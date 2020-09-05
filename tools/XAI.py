import os
import copy
import torch
from tensorboardX import SummaryWriter
import time
import glob
import re
import datetime
import argparse
from pathlib import Path
import torch.distributed as dist
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets.cadc.cadc_bev_visualizer import CADC_BEV
from eval_utils import eval_utils
from pcdet.models import load_data_to_gpu
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.cadc.cadc_dataset import CadcDataset
from pcdet.datasets.kitti.kitti_object_eval_python.eval import d3_box_overlap

# XAI related imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--explained_cfg_file', type=str, default=None,
                        help='specify the config for model to be explained')

    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=80, required=False, help='Number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--mgpus', action='store_true', default=False, help='whether to use multiple gpu')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')
    # # *****************
    # parser.add_argument('--explain', type=str, default='default', required=False, help='indicates the XAI method to be used')
    # # *****************
    args = parser.parse_args()

    # '=' only creates a variable that shares the reference of the original object, hence need to use copy
    # deepcopy creates copy of the original object as well as the nested objects
    x_cfg = copy.deepcopy(cfg)

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    # the model being used to make prediction is the same as the model being explained, unless otherwise specified
    if args.explained_cfg_file is not None:
        print('\n processing the config of the model being explained')
        cfg_from_yaml_file(args.explained_cfg_file, x_cfg)
        x_cfg.TAG = Path(args.explained_cfg_file).stem
        x_cfg.EXP_GROUP_PATH = '/'.join(args.explained_cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    else:
        x_cfg = copy.deepcopy(cfg)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg, x_cfg


def eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=False):
    # load checkpoint
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()

    # start evaluation
    eval_utils.eval_one_epoch(
        cfg, model, test_loader, epoch_id, logger, dist_test=dist_test,
        result_dir=eval_output_dir, save_to_file=args.save_to_file
    )


def get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args):
    ckpt_list = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    evaluated_ckpt_list = [float(x.strip()) for x in open(ckpt_record_file, 'r').readlines()]

    for cur_ckpt in ckpt_list:
        num_list = re.findall('checkpoint_epoch_(.*).pth', cur_ckpt)
        if num_list.__len__() == 0:
            continue

        epoch_id = num_list[-1]
        if 'optim' in epoch_id:
            continue
        if float(epoch_id) not in evaluated_ckpt_list and int(float(epoch_id)) >= args.start_epoch:
            return epoch_id, cur_ckpt
    return -1, None


def repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=False):
    # evaluated ckpt record
    ckpt_record_file = eval_output_dir / ('eval_list_%s.txt' % cfg.DATA_CONFIG.DATA_SPLIT['test'])
    with open(ckpt_record_file, 'a'):
        pass

    # tensorboard log
    if cfg.LOCAL_RANK == 0:
        tb_log = SummaryWriter(log_dir=str(eval_output_dir / ('tensorboard_%s' % cfg.DATA_CONFIG.DATA_SPLIT['test'])))
    total_time = 0
    first_eval = True

    while True:
        # check whether there is checkpoint which is not evaluated
        cur_epoch_id, cur_ckpt = get_no_evaluated_ckpt(ckpt_dir, ckpt_record_file, args)
        if cur_epoch_id == -1 or int(float(cur_epoch_id)) < args.start_epoch:
            wait_second = 30
            print('Wait %s seconds for next check (progress: %.1f / %d minutes): %s \r'
                  % (wait_second, total_time * 1.0 / 60, args.max_waiting_mins, ckpt_dir), end='', flush=True)
            time.sleep(wait_second)
            total_time += 30
            if total_time > args.max_waiting_mins * 60 and (first_eval is False):
                break
            continue

        total_time = 0
        first_eval = False

        model.load_params_from_file(filename=cur_ckpt, logger=logger, to_cpu=dist_test)
        model.cuda()

        # start evaluation
        cur_result_dir = eval_output_dir / ('epoch_%s' % cur_epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        tb_dict = eval_utils.eval_one_epoch(
            cfg, model, test_loader, cur_epoch_id, logger, dist_test=dist_test,
            result_dir=cur_result_dir, save_to_file=args.save_to_file
        )

        if cfg.LOCAL_RANK == 0:
            for key, val in tb_dict.items():
                tb_log.add_scalar(key, val, cur_epoch_id)

        # record this epoch which has been evaluated
        with open(ckpt_record_file, 'a') as f:
            print('%s' % cur_epoch_id, file=f)
        logger.info('Epoch %s has been evaluated' % cur_epoch_id)


def get_gt_infos(cfg, dataset):
    '''
    :param dataset: dataset object
    :param cfg: object containing model config information
    :return: gt_infos--containing gt boxes that have labels corresponding to the classes of interest, as well as the
                labels themselves
    '''
    # dataset_cfg = cfg.DATA_CONFIG
    # class_names = cfg.CLASS_NAMES
    # kitti = KittiDataset(
    #     dataset_cfg=dataset_cfg,
    #     class_names=class_names,
    #     training=False
    # )
    gt_infos = []
    dataset_infos = []
    dataset_name = cfg.DATA_CONFIG.DATASET
    if dataset_name == 'CadcDataset':
        dataset_infos = dataset.cadc_infos
    elif dataset_name == 'KittiDataset':
        dataset_infos = dataset.kitti_infos
    for info in dataset_infos:
        box3d_lidar = np.array(info['annos']['gt_boxes_lidar'])
        labels = np.array(info['annos']['name'])

        interested = []
        for i in range(len(labels)):
            label = labels[i]
            if label in cfg.CLASS_NAMES:
                interested.append(i)
        interested = np.array(interested)

        # box3d_lidar[:,2] -= box3d_lidar[:,5] / 2
        gt_info = {
            'boxes': box3d_lidar[interested],
            'labels': info['annos']['name'][interested],
        }
        gt_infos.append(gt_info)
    return gt_infos


def calculate_iou(gt_boxes, pred_boxes):
    # see pcdet/datasets/kitti/kitti_object_eval_python/eval.py for explanation
    z_axis = 2
    z_center = 0.5
    if cfg.DATA_CONFIG.DATASET == 'KittiDataset':
        z_axis = 1
        z_center = 1.0
    overlap = d3_box_overlap(gt_boxes, pred_boxes, z_axis=z_axis, z_center=z_center)
    # pick max iou wrt to each detection box
    iou, gt_index = np.max(overlap, axis=0), np.argmax(overlap, axis=0)
    return iou, gt_index

def scale_3d_array(orig_array, factor):
    '''

    :param orig_array: Input 3D array.
    :param factor: The factor by which the input is scaled
    :return: new_array: The resulting rescaled array
    '''
    new_layer_list = []
    channels = orig_array.shape[2]
    for i in range(channels):
        layer_i = orig_array[:, :, i]
        new_layer_i = np.kron(layer_i, np.ones((factor, factor), dtype=int))
        new_layer_list.append(new_layer_i)
    new_layer_tuple = tuple(new_layer_list)
    new_array = np.dstack(new_layer_tuple)
    return new_array

def main():
    '''
    important variables:
        batches_to_analyze: The number of batches for which explanations are generated
        method: Explanation method used.
        high_rez: Whether to use higher resolution (in the case of CADC, this means 2000x2000 instead of 400x400)
        overlay_orig_bev: If True, overlay attributions onto the original point cloud BEV. If False, overlay
            attributions onto the 2D pseudoimage.
        mult_by_inputs: Whether to pointwise-multiply Integrated Gradients attributions with the input being explained
            (i.e., the 2D pseudoimage in the case of PointPillar).
        gray_scale_overlay: Whether to convert the input image into gray scale.
        orig_bev_w: Width of the original BEV in # pixels. For CADC, width = height. Note that this HAS to be an
            integer multiple of pseudo_img_w.
        orig_bev_h: Height of the original BEV in # pixels. For CADC, width = height.
    :return:
    '''
    batches_to_analyze = 10
    method = 'Saliency'
    high_rez = True
    overlay_orig_bev = True
    mult_by_inputs = False
    gray_scale_overlay = False
    orig_bev_w = 2000
    orig_bev_h = 2000
    pseudo_img_w = 400
    scaling_factor = int(orig_bev_w/pseudo_img_w)
    args, cfg, x_cfg = parse_config()
    dataset_name = cfg.DATA_CONFIG.DATASET
    if args.launcher == 'none':
        dist_test = False
    else:
        args.batch_size, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.batch_size, args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    if not args.eval_all:
        num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if args.eval_tag is not None:
        eval_output_dir = eval_output_dir / args.eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        total_gpus = dist.get_world_size()
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

    # Create dummy `dataset` object to pass necessary information to `build_network`.
    # test_set = EasyDict({
    #     'class_names': cfg.CLASS_NAMES,
    #     'grid_size': cfg.GRID_SIZE,
    #     'voxel_size': cfg.VOXEL_SIZE,
    #     'point_cloud_range': cfg.POINT_CLOUD_RANGE,
    #     'point_feature_encoder': {
    #         'num_point_features': cfg.NUM_POINT_FEATURES
    #     }
    # })

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )
    print("grid size for 2D pseudoimage: {}".format(test_set.grid_size))

    # # ********** debug message **************
    # print('\n Checking if the 2d network has VFE')
    # if x_cfg.MODEL.get('VFE', None) is None:
    #     print('\n no VFE')
    # else:
    #     print('\n has VFE')
    # # ********** debug message **************
    cadc_bev = CADC_BEV(dataset=test_set, scale_to_pseudoimg=(not high_rez), background='black', width_pix = orig_bev_w)
    print('\n \n building the 2d network')
    model2D = build_network(model_cfg=x_cfg.MODEL, num_class=len(x_cfg.CLASS_NAMES), dataset=test_set)
    print('\n \n building the full network')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

    # # Comment out to just run XAI stuff
    # with torch.no_grad():
    #     if args.eval_all:
    #         repeat_eval_ckpt(model, test_loader, args, eval_output_dir, logger, ckpt_dir, dist_test=dist_test)
    #     else:
    #         eval_single_ckpt(model, test_loader, args, eval_output_dir, logger, epoch_id, dist_test=dist_test)

    # *****************
    # start experimenting with saliency map
    '''
    TODO:
    what is the input?      A certain point cloud with a specific index
    what is the target?     A certain label with a specific index
    what does np.transpose do for grads?
    need to understand viz.visualize_image_attr better/
    '''
    saliency2D = Saliency(model2D)
    saliency = Saliency(model)
    ig2D = IntegratedGradients(model2D, multiply_by_inputs=mult_by_inputs)
    steps = 24  # number of steps for IG
    # load checkpoint
    print('\n \n loading parameters for the 2d network')
    model2D.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model2D.cuda()
    model2D.eval()
    print('\n \n loading parameters for the full network')
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()
    model.eval()

    class_name_dict = {
        0: 'car',
        1: 'pedestrian',
        2: 'cyclist'
    }

    # get the date and time to create a folder for the specific time when this script is run
    now = datetime.datetime.now()
    dt_string = now.strftime("%b_%d_%Y_%H_%M_%S")
    # get current working directory
    cwd = os.getcwd()
    rez_string = 'SameResolutionAsPseudoimage'
    if high_rez:
        rez_string = 'SameResolutionAsOriginalBEV'
    # create directory to store results just for this run, include the method in folder name
    XAI_result_path = os.path.join(cwd, 'XAI_results/{}_{}_{}_{}'.format(dt_string, method, rez_string, dataset_name))
    if method == "IG":
        if mult_by_inputs:
            XAI_result_path = os.path.join(cwd, 'XAI_results/{}_{}_{}_{}steps_{}_{}'.format(
                dt_string, method, 'multiply_by_inputs', steps, rez_string, dataset_name))
        else:
            XAI_result_path = os.path.join(cwd, 'XAI_results/{}_{}_{}_{}steps_{}_{}'.format(
                dt_string, method, 'no_multiply_by_inputs', steps, rez_string, dataset_name))
    print('\nXAI_result_path: {}'.format(XAI_result_path))
    XAI_res_path_str = str(XAI_result_path)
    os.mkdir(XAI_result_path)
    os.chmod(XAI_res_path_str, 0o777)

    for batch_num, batch_dict in enumerate(test_loader):
        XAI_batch_path_str = XAI_res_path_str + '/batch_{}'.format(batch_num)
        os.mkdir(XAI_batch_path_str)
        # run the forward pass once to generate outputs and intermediate representations
        dummy_tensor = 0
        load_data_to_gpu(batch_dict)  # this function is designed for dict, don't use for other data types!
        with torch.no_grad():
            boxes_with_classes = model(dummy_tensor, batch_dict)
        pred_dicts = batch_dict['pred_dicts']
        '''
        Note:
        - In Captum, the forward function of the model is called in such a way: forward_func(input, addtional_input_args)
        - We are using the batch_dict as the addtional_input_args
        - Hence needed a dummy_tensor before batch_dict when we are not in explain mode
        - Alternatively, can modify Captum, but I prefer not to
        '''
        # note: boxes_with_cls_scores contains class scores for each box identified
        # this is the pseudo image used as input for the 2D backbone
        PseudoImage2D = batch_dict['spatial_features']
        # now, need one prediction, which serves as the target
        DetectionHead = model.module_list[
            3]  # note, the index 3 is specifically for pointpillar, can be other number for other models
        BoxPrediction = DetectionHead.forward_ret_dict['cls_preds']
        BoxGroundTruth = batch_dict['gt_boxes']
        # TODO: need to somehow zero out all other predictions and keep just one (idea: single activated neuron)
        # # need to understand better what the target meant in the original Captum example
        # generate explanation for a single 2D pseudoimage at a time
        # `target = 1`: attempting to generate
        SinglePseudoImage2D = PseudoImage2D[0].unsqueeze(0)
        # print('SinglePseudoImage2D dimensions: ' + str(SinglePseudoImage2D.shape))
        # transform the original image to have shape H x W x C, where C means channels
        original_image = np.transpose(PseudoImage2D[0].cpu().detach().numpy(), (1, 2, 0))
        print('orginal_image data type: {}'.format(type(original_image)))
        # print('original_image dimensions: ' + str(original_image.shape))
        # print('\n \n strutcture of the 2D network:')
        # print(model2D)
        # print('\n \n structure of the full network: ')
        # print(model)

        # target dimensions: batch_size and maximum number of boxes per image
        # 1 should correspond to the pedestrain class
        # target = torch.ones(4,50) # in config file, the max num of boxes is 500, I chose 50 instead

        # Separate TP and FP
        score_thres, iou_thres = 0.5, 0.7
        gt_infos = get_gt_infos(cfg, test_set)
        gt_dict = gt_infos[batch_num * args.batch_size:(batch_num + 1) * args.batch_size]
        conf_mat = []
        for i in range(args.batch_size):  # i is input image id in the batch
            conf_mat_frame = []
            pred_boxes = pred_dicts[i]['pred_boxes'].cpu().numpy()
            iou, gt_index = calculate_iou(gt_dict[i]['boxes'], pred_boxes)
            for j in range(len(pred_dicts[i]['pred_scores'])):  # j is prediction box id in the i-th image
                if pred_dicts[i]['pred_scores'][j].cpu().numpy() >= score_thres:
                    adjusted_pred_boxes_labels = pred_dicts[i]['pred_labels'][j].cpu().numpy() - 1
                    if iou[j] >= iou_thres and gt_dict[i]['labels'][gt_index[j]] == class_name_dict[
                        adjusted_pred_boxes_labels]:
                        conf_mat_frame.append('TP')
                    else:
                        conf_mat_frame.append('FP')
                else:
                    conf_mat_frame.append('ignore')  # these boxes do not meet score thresh, ignore for now
            conf_mat.append(conf_mat_frame)
            # print("sample id: {}".format(i))
            # print("len(pred_dicts[i]['pred_scores']): {}".format(len(pred_dicts[i]['pred_scores'])))
            # print("len(pred_dicts[i]['pred_labels']): {}".format(len(pred_dicts[i]['pred_labels'])))
            # print('batch_dict[\'box_count\'][i]: {}'.format(batch_dict['box_count'][i]))
            # print("len(conf_mat_frame): {}".format(len(conf_mat_frame)))
            # print('\n')
            # break

        for i in range(batch_dict['batch_size']):  # iterate through each sample in the batch
            img_idx = batch_num * args.batch_size + i
            bev_fig, bev_fig_data = cadc_bev.get_bev_image(img_idx)
            bev_image = np.transpose(PseudoImage2D[i].cpu().detach().numpy(), (1, 2, 0))
            for k in range(3):  # iterate through the 3 classes
                '''
                Note:
                - In batch_2, sample_0, as the program iterates through the boxes batch_dict['box_count'][0] was
                  originally 35
                - But since the second iteration of j (i.e. j <= 1), batch_dict['box_count'][0] somehow became 36
                - Therefore need a fix for that
                '''
                num_boxes = min(batch_dict['box_count'][i], len(conf_mat[i]))
                for j in range(num_boxes):  # iterate through each box in this sample
                    # print('i: {}'.format(i))
                    # print('j: {}'.format(j))
                    # print('num_boxes: {}'.format(num_boxes))
                    # print('len(conf_mat[i]): {}'.format(len(conf_mat[i])))
                    if k + 1 == pred_dicts[i]['pred_labels'][j] and conf_mat[i][j] != 'ignore':
                        # compute contribution for the positive class only, k+1 because PCDet labels start at 1
                        target = (j, k)  # i.e., generate reason as to why the j-th box is classified as the k-th class
                        grads = None
                        sign = "absolute_value"
                        if method == 'Saliency':
                            grads = copy.deepcopy(
                                saliency2D.attribute(PseudoImage2D, target=target, additional_forward_args=batch_dict))
                        if method == 'IG':
                            grads = copy.deepcopy(ig2D.attribute(PseudoImage2D, baselines=PseudoImage2D * 0,
                                                                 target=target, additional_forward_args=batch_dict,
                                                                 n_steps=steps,
                                                                 internal_batch_size=batch_dict['batch_size']))
                            sign = "all"
                        grad = np.transpose(grads[i].squeeze().cpu().detach().numpy(), (1, 2, 0))
                        # print('grad.shape: {}'.format(grad.shape))
                        # print('type(grad): {}'.format(type(grad)))
                        figure_size = (8, 6)
                        if high_rez:
                            # upscale the attributions if we are using high resolution input bev
                            # also increase figure size to accommodate the higher resolution
                            figure_size = (24, 18)
                            if dataset_name == 'CadcDataset':
                                grad = scale_3d_array(grad, factor=scaling_factor)
                            # TODO: implement for kitti as well
                        # print('grad.shape: {}'.format(grad.shape))
                        # print('type(grad): {}'.format(type(grad)))
                        # print('bev_fig_data.shape: {}'.format(bev_fig_data.shape))
                        # print('bev_image.shape before: {}'.format(bev_image.shape))
                        if overlay_orig_bev:
                            # need to horizontally flip the original bev image and rotate 90 degrees ccw to match location
                            # of explanations
                            bev_image_raw = np.flip(bev_fig_data, 0)
                            bev_image = np.rot90(bev_image_raw, k=1, axes=(0,1))
                        # print('bev_image.shape after: {}'.format(bev_image.shape))

                        grad_viz = viz.visualize_image_attr(grad, bev_image, method="blended_heat_map", sign=sign,
                                                            show_colorbar=True, title="Overlaid {}".format(method),
                                                            gray_scale=gray_scale_overlay, alpha_overlay=0.1,
                                                            fig_size=figure_size)
                        XAI_cls_path_str = XAI_batch_path_str + '/explanation_for_{}/sample_{}'.format(
                            class_name_dict[k], i)
                        if not os.path.exists(XAI_cls_path_str):
                            os.makedirs(XAI_cls_path_str)
                        os.chmod(XAI_cls_path_str, 0o777)
                        XAI_box_relative_path_str = XAI_cls_path_str.split("tools/", 1)[1] +\
                                                    '/box_{}_{}.png'.format(j, conf_mat[i][j])
                        # print('XAI_box_path_str: {}'.format(XAI_box_path_str))
                        grad_viz[0].savefig(XAI_box_relative_path_str, bbox_inches='tight', pad_inches=0.0)
                        # print('box_{}_{}.png is saved in {}'.format(j,conf_mat[i][j],XAI_cls_path_str))
            break # only processing one sample to save time

        # if batch_num == batches_to_analyze:
        #     break  # just process a limited number of batches
        break
        # just need one explanation for example


if __name__ == '__main__':
    main()
