import os
import copy
import torch
from tensorboardX import SummaryWriter
import time
import glob
import re
import h5py
import datetime
import argparse
import csv
import math
from pathlib import Path
import torch.distributed as dist
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets.kitti.kitti_bev_visualizer import KITTI_BEV
from pcdet.datasets.cadc.cadc_bev_visualizer import CADC_BEV
from pcdet.utils import box_utils
from eval_utils import eval_utils
from XAI_utils.bbox_utils import *
from XAI_utils.tp_fp import *
from XAI_utils.XQ_utils import *
from pcdet.models import load_data_to_gpu
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.cadc.cadc_dataset import CadcDataset
from pcdet.datasets.kitti.kitti_object_eval_python.eval import d3_box_overlap

# XAI related imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def calculate_iou(gt_boxes, pred_boxes):
    # see pcdet/datasets/kitti/kitti_object_eval_python/eval.py for explanation
    z_axis = 2
    z_center = 0.5
    if cfg.DATA_CONFIG.DATASET == 'KittiDataset':
        z_axis = 1
        z_center = 1.0
    overlap = d3_box_overlap(gt_boxes, pred_boxes, z_axis=z_axis, z_center=z_center)
    # pick max iou wrt to each detection box
    print('overlap.shape: {}'.format(overlap.shape))
    iou, gt_index = np.max(overlap, axis=0), np.argmax(overlap, axis=0)
    return iou, gt_index


def calculate_bev_iou(gt_boxes, pred_boxes):
    overlap = bboxes3d_nearest_bev_iou(pred_boxes[:, 0:7], gt_boxes[:, 0:7])
    iou, gt_index = np.max(overlap.numpy(), axis=1), np.argmax(overlap.numpy(), axis=1)
    return iou, gt_index


def find_missing_gt(gt_dict, pred_boxes, iou_unmatching_thresholds):
    # TODO: merge this into calculate_bev_iou later to avoid repeated calculations
    # TODO: pass class-specific iou_thresh  find_missing_gt(gt_dict[i], pred_boxes, iou_unmatching_thresholds)
    gt_boxes = gt_dict['boxes']
    gt_labels = gt_dict['labels']
    overlap = bboxes3d_nearest_bev_iou(gt_boxes[:, 0:7], pred_boxes[:, 0:7])
    # pick max iou wrt to each detection box
    print('overlap.shape: {}'.format(overlap.shape))
    ious = np.max(overlap.numpy(), axis=1)
    ind = 0
    missed_gt_idx = []
    for iou in ious:
        if iou < iou_unmatching_thresholds[gt_labels[ind]]:
            print('gt lable is : {}'.format(gt_labels[ind]))
            print('maximum iou is: {}'.format(iou))
            print('matching thresh is: {}'.format(iou_unmatching_thresholds[gt_labels[ind]]))
            missed_gt_idx.append(ind)
        ind += 1
    print('missed_gt_idx: {}'.format(missed_gt_idx))
    return missed_gt_idx


def main():
    """
    important variables:
        max_obj_cnt: The maximum number of objects in an image, right now set to 50
        batches_to_analyze: The number of batches for which explanations are generated
        method: Explanation method used.
        attr_shown: What type of attributions are shown, can be 'absolute_value', 'positive', 'negative', or 'all'.
        high_rez: Whether to use higher resolution (in the case of CADC, this means 2000x2000 instead of 400x400)
        overlay_orig_bev: If True, overlay attributions onto the original point cloud BEV. If False, overlay
            attributions onto the 2D pseudoimage.
        mult_by_inputs: Whether to pointwise-multiply Integrated Gradients attributions with the input being explained
            (i.e., the 2D pseudoimage in the case of PointPillar).
        channel_xai: Whether to generate channel-wise attribution heat map for the pseudoimage
        gray_scale_overlay: Whether to convert the input image into gray scale.
        plot_class_wise: Whether to generate plots for each class.
        box_margin: Margin for the bounding boxes in number of pixels
        orig_bev_w: Width of the original BEV in # pixels. For CADC, width = height. Note that this HAS to be an
            integer multiple of pseudo_img_w.
        orig_bev_h: Height of the original BEV in # pixels. For CADC, width = height.
        dpi_division_factor: Divide image dimension in pixels by this number to get dots per inch (dpi). Lower this
            parameter to get higher dpi.
    :return:
    """
    box_cnt = 0
    start_time = time.time()
    max_obj_cnt = 50
    batches_to_analyze = 1
    method = 'IG'
    attr_shown = 'positive'
    high_rez = True
    overlay_orig_bev = True
    mult_by_inputs = True
    channel_xai = False
    gray_scale_overlay = True
    plot_class_wise = False
    color_map = 'jet'
    box_margin = 5.0
    if gray_scale_overlay:
        color_map = 'gray'
    scaling_factor = 5
    args, cfg, x_cfg = parse_config()
    dataset_name = cfg.DATA_CONFIG.DATASET
    num_channels_viz = min(32, cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES)
    pseudo_img_w = 0
    pseudo_img_h = 0
    total_anchors = 0
    if dataset_name == 'CadcDataset':
        pseudo_img_w = 400
        pseudo_img_h = pseudo_img_w
        total_anchors = 240000
    elif dataset_name == 'KittiDataset':
        pseudo_img_w = 432
        pseudo_img_h = 496
        total_anchors = 321408
    attr_shape = (0, pseudo_img_h, pseudo_img_w)
    max_shape = (None, pseudo_img_h, pseudo_img_w)
    orig_bev_w = pseudo_img_w * 5
    orig_bev_h = pseudo_img_h * 5
    dpi_division_factor = 20.0
    use_anchor_idx = x_cfg.MODEL.POST_PROCESSING.OUTPUT_ANCHOR_BOXES
    if args.launcher == 'none':
        dist_test = False
    else:
        args.batch_size, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.batch_size, args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    cls_score_list = []
    dist_list = []  # stores distance to ego vehicle
    label_list = []
    TP_score_list = []
    TP_XQ_list = []
    TP_dist_list = []
    TP_label_list = []
    FP_score_list = []
    FP_XQ_list = []
    FP_dist_list = []
    FP_label_list = []
    XQ_list = []
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

    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )
    print("grid size for 2D pseudoimage: {}".format(test_set.grid_size))

    cadc_bev = CADC_BEV(dataset=test_set, scale_to_pseudoimg=(not high_rez), background='black',
                        scale=scaling_factor, cmap=color_map, dpi_factor=dpi_division_factor)
    kitti_bev = KITTI_BEV(dataset=test_set, scale_to_pseudoimg=(not high_rez), background='black',
                          result_path='output/kitti_models/pointpillar/default/eval/epoch_7728/val/default/result.pkl',
                          scale=scaling_factor, cmap=color_map, dpi_factor=dpi_division_factor)
    print('\n \n building the 2d network')
    model2D = build_network(model_cfg=x_cfg.MODEL, num_class=len(x_cfg.CLASS_NAMES), dataset=test_set)
    print('\n \n building the full network')
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)

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

    class_name_list = cfg.CLASS_NAMES

    # get the date and time to create a folder for the specific time when this script is run
    now = datetime.datetime.now()
    dt_string = now.strftime("%b_%d_%Y_%H_%M_%S")
    # get current working directory
    cwd = os.getcwd()
    rez_string = 'LowResolution'
    if high_rez:
        rez_string = 'HighResolution'
    # create directory to store results just for this run, include the method in folder name
    XAI_result_path = os.path.join(cwd, 'XAI_results/{}_{}_{}_{}batches'.format(
        dt_string, method, dataset_name, batches_to_analyze))
    XAI_attr_path = os.path.join(cwd, 'XAI_attributions/{}_{}_{}_{}batches'.format(
        dt_string, method, dataset_name, batches_to_analyze))

    print('\nXAI_result_path: {}'.format(XAI_result_path))
    XAI_res_path_str = str(XAI_result_path)
    os.mkdir(XAI_result_path)
    os.chmod(XAI_res_path_str, 0o777)
    XAI_attr_path_str = str(XAI_attr_path)
    os.mkdir(XAI_attr_path)
    os.chmod(XAI_attr_path_str, 0o777)
    info_file = XAI_res_path_str + "/XAI_information.txt"
    f = open(info_file, "w")
    f.write('High Resolution: {}\n'.format(high_rez))
    f.write('DPI Division Factor: {}\n'.format(dpi_division_factor))
    f.write('Attributions Visualized: {}\n'.format(attr_shown))
    f.write('Bounding box margin: {}\n'.format(box_margin))
    info_file_2 = XAI_attr_path_str + "/XAI_information.txt"
    f2 = open(info_file_2, "w")
    f2.write('High Resolution: {}\n'.format(high_rez))
    f2.write('DPI Division Factor: {}\n'.format(dpi_division_factor))
    f2.write('Attributions Visualized: {}\n'.format(attr_shown))
    f2.write('Bounding box margin: {}\n'.format(box_margin))
    if overlay_orig_bev:
        f.write('Background Image: BEV of the original point cloud\n')
        f2.write('Background Image: BEV of the original point cloud\n')
    else:
        f.write('Background Image: BEV of the PointPillar Pseudoimage\n')
        f2.write('Background Image: BEV of the PointPillar Pseudoimage\n')
    if method == "IG":
        f.write("IG # of Steps: {}\n".format(steps))
        f.write("Multiply by Input: {}\n".format(mult_by_inputs))
        f2.write("IG # of Steps: {}\n".format(steps))
        f2.write("Multiply by Input: {}\n".format(mult_by_inputs))
    f.write('Channel-wise Explanation: {}\n'.format(channel_xai))
    f2.write('Channel-wise Explanation: {}\n'.format(channel_xai))
    f.write('Recording XQ values: \n')
    os.chmod(info_file, 0o777)
    os.chmod(info_file_2, 0o777)
    try:
        for batch_num, batch_dict in enumerate(test_loader):
            print("\nbatch_num: {}\n".format(batch_num))
            # if batch_num != 25:
            #     continue
            # print('\nlen(batch_dict): {}\n'.format(len(batch_dict)))
            XAI_batch_path_str = XAI_res_path_str + '/batch_{}'.format(batch_num)
            os.mkdir(XAI_batch_path_str)
            os.chmod(XAI_batch_path_str, 0o777)
            XAI_attr_batch_path_str = XAI_attr_path_str + '/batch_{}'.format(batch_num)
            os.mkdir(XAI_attr_batch_path_str)
            os.chmod(XAI_attr_batch_path_str, 0o777)
            # run the forward pass once to generate outputs and intermediate representations
            dummy_tensor = 0
            load_data_to_gpu(batch_dict)  # this function is designed for dict, don't use for other data types!
            with torch.no_grad():
                anchors_scores = model(dummy_tensor, batch_dict)
            pred_dicts = batch_dict['pred_dicts']
            # print('\nlen(batch_dict[\'pred_dicts\']): {}\n'.format(len(batch_dict['pred_dicts'])))
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
            original_image = np.transpose(PseudoImage2D[0].cpu().detach().numpy(), (1, 2, 0))
            # print('orginal_image data type: {}'.format(type(original_image)))

            # Separate TP and FP
            # TODO: get thresh from config file
            score_thres = 0.1
            anchor_generator_cfg = cfg.MODEL.DENSE_HEAD.ANCHOR_GENERATOR_CONFIG
            # use the thresholds in PCDet's config file for the model
            iou_unmatching_thresholds = {}
            for config in anchor_generator_cfg:
                iou_unmatching_thresholds[config['class_name']] = config['unmatched_threshold']
            gt_infos = get_gt_infos(cfg, test_set)
            gt_dict = gt_infos[batch_num * args.batch_size:(batch_num + 1) * args.batch_size]
            conf_mat = []
            missed_boxes = []
            missed_boxes_plotted = []
            batch_pred_scores = []

            # for i in range(args.batch_size):
            #     anchor_box_loc_file = XAI_res_path_str + "/anchors_{}.csv".format(i)
            #     with open(anchor_box_loc_file, 'w', newline='') as csvfile:
            #         field_name = ['box_id', 'x', 'y']
            #         writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=field_name)
            #         writer.writeheader()
            #         boxes = batch_dict['anchor_boxes'][i].cpu().numpy()
            #         anchor_cnt = 0
            #         if dataset_name == 'CadcDataset':
            #             anchor_cnt = int(400 * 400 * 1.5)
            #         elif dataset_name == 'KittiDataset':
            #             anchor_cnt = int(496 * 432 * 1.5)
            #         for j in range(0, len(boxes), 1):
            #             x = boxes[j][0]
            #             y = boxes[j][1]
            #             writer.writerow({field_name[0]: j, field_name[1]: x, field_name[2]: y})
            for i in range(args.batch_size):  # i is input image id in the batch
                missed_boxes_plotted = []
                for c in range(len(class_name_list)):
                    missed_boxes_plotted.append(False)
                conf_mat_frame = []
                pred_boxes = pred_dicts[i]['pred_boxes'].cpu().numpy()
                iou, gt_index = calculate_bev_iou(gt_dict[i]['boxes'], pred_boxes)
                missed_gt_index = find_missing_gt(gt_dict[i], pred_boxes, iou_unmatching_thresholds)
                missed_boxes.append(missed_gt_index)
                sample_pred_scores = pred_dicts[i]['pred_scores'].cpu().numpy()
                batch_pred_scores.append(sample_pred_scores)
                for j in range(len(pred_dicts[i]['pred_scores'])):  # j is prediction box id in the i-th image
                    gt_cls = gt_dict[i]['labels'][gt_index[j]]
                    iou_thresh_2d = iou_unmatching_thresholds[gt_cls]
                    curr_pred_score = pred_dicts[i]['pred_scores'][j].cpu().numpy()
                    if curr_pred_score >= score_thres:
                        adjusted_pred_boxes_label = pred_dicts[i]['pred_labels'][j].cpu().numpy() - 1
                        if iou[j] >= iou_thresh_2d and gt_dict[i]['labels'][gt_index[j]] == class_name_list[
                            adjusted_pred_boxes_label]:
                            conf_mat_frame.append('TP')
                        else:
                            conf_mat_frame.append('FP')
                    else:
                        conf_mat_frame.append('ignore')  # these boxes do not meet score thresh, ignore for now
                conf_mat.append(conf_mat_frame)
            attributions = []  # a 2D dictionary that stores the gradients in this batch
            # initialize attributions
            for j in range(max_obj_cnt):  # the maximum number of objects in one image
                new_list = []
                for k in range(len(class_name_list)):
                    new_list.append(torch.zeros(1))
                    # for i in range(batch_dict['batch_size']):
                    #     attributions[(j, k)][i] = None\
                attributions.append(new_list)

            for i in range(batch_dict['batch_size']):  # iterate through each sample in the batch
                img_idx = batch_num * args.batch_size + i
                bev_fig, bev_fig_data = None, None
                pred_boxes_vertices = None
                gt_boxes_vertices = None
                gt_boxes_loc = None
                pred_boxes_loc = None
                total_attr = 0              # total amount of attributions in a box
                if dataset_name == 'CadcDataset':
                    bev_fig, bev_fig_data = cadc_bev.get_bev_image(img_idx)
                    pred_boxes_vertices = cadc_bev.pred_poly
                    gt_boxes_vertices = cadc_bev.gt_poly
                    gt_boxes_loc = cadc_bev.gt_loc
                    pred_boxes_loc = cadc_bev.pred_loc
                elif dataset_name == 'KittiDataset':
                    bev_fig, bev_fig_data = kitti_bev.get_bev_image(img_idx)
                    pred_boxes_vertices = kitti_bev.pred_poly
                    gt_boxes_vertices = kitti_bev.gt_poly
                    gt_boxes_loc = kitti_bev.gt_loc
                    pred_boxes_loc = kitti_bev.pred_loc
                bev_image = np.transpose(PseudoImage2D[i].cpu().detach().numpy(), (1, 2, 0))
                pred_boxes = pred_dicts[i]['pred_boxes'].cpu().numpy()
                pred_scores = pred_dicts[i]['pred_scores'].cpu().numpy()
                '''
                format for pred_boxes[box_index]: x,y,z,l,w,h,theta, for both CADC and KITTI
                format for pred_boxes_vertices[box_index]: ((x1,y1), ... ,(x4,y4))
                '''
                # print("pred_boxes.shape[0]: {} \n"
                #       "len(pred_boxes_vertices): {}".format(pred_boxes.shape[0], len(pred_boxes_vertices)))
                box_cnt_list = []
                box_cnt_list.append(batch_dict['box_count'][i])
                box_cnt_list.append(len(conf_mat[i]))
                box_cnt_list.append(max_obj_cnt)
                box_cnt_list.append(len(batch_dict['anchor_selections'][i]))

                XAI_sample_path_str = XAI_batch_path_str + '/sample_{}'.format(i)
                # if not os.path.exists(XAI_sample_path_str):
                os.makedirs(XAI_sample_path_str)
                os.chmod(XAI_sample_path_str, 0o777)

                XAI_attr_sample_path_str = XAI_attr_batch_path_str + '/sample_{}'.format(i)
                # if not os.path.exists(XAI_attr_sample_path_str):
                os.makedirs(XAI_attr_sample_path_str)
                os.chmod(XAI_attr_sample_path_str, 0o777)

                for k in range(3):  # iterate through the 3 classes
                    num_boxes = min(box_cnt_list) - 1
                    print('num_boxes: {}'.format(num_boxes))
                    XAI_cls_path_str = XAI_sample_path_str + '/explanation_for_{}'.format(
                        class_name_list[k])
                    XAI_attr_cls_path_str = XAI_attr_sample_path_str + '/explanation_for_{}'.format(
                        class_name_list[k])
                    # if not os.path.exists(XAI_cls_path_str):
                    os.makedirs(XAI_cls_path_str)
                    os.chmod(XAI_cls_path_str, 0o777)
                    # if not os.path.exists(XAI_attr_cls_path_str):
                    os.makedirs(XAI_attr_cls_path_str)
                    os.chmod(XAI_attr_cls_path_str, 0o777)

                    # initialize the attributions files
                    attr_file = h5py.File(
                        XAI_attr_cls_path_str + '/attr_for_{}.hdf5'.format(class_name_list[k]), 'w')
                    pos_attr_values = attr_file.create_dataset(
                        "pos_attr", attr_shape, maxshape=max_shape, dtype="float32", chunks=True)
                    neg_attr_values = attr_file.create_dataset(
                        "neg_attr", attr_shape, maxshape=max_shape, dtype="float32", chunks=True)
                    stored_pred_boxes = attr_file.create_dataset(
                        "pred_boxes", (0, 4, 2), maxshape=(None, 4, 2), dtype="float32", chunks=True)
                    # encode box types with numbers: 0-FP, 1-TP
                    boxes_type = attr_file.create_dataset(
                        "box_type", (0), maxshape=(None), dtype="uint8", chunks=True)
                    pred_boxes_location = attr_file.create_dataset(
                        "pred_boxes_loc", (0, 2), maxshape=(None, 2), dtype="float32", chunks=True)

                    for j in range(num_boxes):  # iterate through each box in this sample
                        if not box_validation(pred_boxes[j], pred_boxes_vertices[j], dataset_name):
                            continue  # can't compute XQ if the boxes don't match
                        '''
                        Note:
                        Sometimes the size of batch_dict['anchor_selections'][i] changes for no reason.
                        Hence need this double check here
                        Also, num_boxes is already decremented by 1, just to be safe
                        '''
                        if j >= batch_dict['box_count'][i] or j >= len(batch_dict['anchor_selections'][i]):
                            break
                        box_w = pred_boxes[j][3]
                        box_l = pred_boxes[j][4]
                        box_x = pred_boxes_loc[j][0]
                        box_y = pred_boxes_loc[j][1]
                        dist_to_ego = np.sqrt(box_x * box_x + box_y * box_y)

                        if k + 1 == pred_dicts[i]['pred_labels'][j] and conf_mat[i][j] != 'ignore':
                            # compute contribution for the positive class only, k+1 because PCDet labels start at 1
                            target = (j, k)  # i.e., generate reason as to why the j-th box is classified as the k-th class
                            anchor_id = batch_dict['anchor_selections'][i][j]

                            if use_anchor_idx:
                                # if we are using anchor box outputs, need to use the indices in for anchor boxes
                                target = (anchor_id, k)
                            sign = attr_shown
                            if method == 'Saliency':
                                if len(attributions[j][k].shape) == 1:  # compute attributions only when necessary
                                    attributions[j][k] = saliency2D.attribute(
                                        PseudoImage2D, target=target, additional_forward_args=batch_dict)
                            if method == 'IG':
                                if len(attributions[j][k].shape) == 1:  # compute attributions only when necessary
                                    attributions[j][k] = ig2D.attribute(
                                        PseudoImage2D, baselines=PseudoImage2D * 0, target=target,
                                        additional_forward_args=batch_dict, n_steps=steps,
                                        internal_batch_size=batch_dict['batch_size'])
                                # sign = "all"
                                # sign = "positive"
                            grad = np.transpose(attributions[j][k][i].squeeze().cpu().detach().numpy(), (1, 2, 0))

                            pos_grad = np.sum((grad > 0) * grad, axis=2)
                            neg_grad = np.sum((grad < 0) * grad, axis=2)
                            # print('pos_grad.shape: {}'.format(pos_grad.shape))

                            new_size = attr_file["pos_attr"].shape[0] + 1
                            attr_file["pos_attr"].resize(new_size, axis=0)
                            attr_file["neg_attr"].resize(new_size, axis=0)
                            attr_file["pred_boxes"].resize(new_size, axis=0)
                            attr_file["box_type"].resize(new_size, axis=0)
                            attr_file["pred_boxes_loc"].resize(new_size, axis=0)

                            attr_file["pos_attr"][new_size-1] = pos_grad
                            attr_file["neg_attr"][new_size - 1] = neg_grad
                            attr_file["pred_boxes"][new_size-1] = pred_boxes_vertices[j]
                            attr_file["pred_boxes_loc"][new_size - 1] = pred_boxes_loc[j]

                            XQ = get_sum_XQ(grad, pred_boxes_vertices[j], dataset_name, box_w, box_l, sign,
                                            high_rez=high_rez, scaling_factor=scaling_factor, margin=box_margin)
                            XQ_list.append(XQ)
                            cls_score_list.append(pred_scores[j])
                            dist_list.append(dist_to_ego)
                            label_list.append(k)
                            box_cnt += 1
                            if conf_mat[i][j] == 'TP':
                                TP_XQ_list.append(XQ)
                                TP_score_list.append(pred_scores[j])
                                TP_dist_list.append(dist_to_ego)
                                TP_label_list.append(k)
                                attr_file["box_type"][new_size-1] = 1
                            elif conf_mat[i][j] == 'FP':
                                FP_XQ_list.append(XQ)
                                FP_score_list.append(pred_scores[j])
                                FP_dist_list.append(dist_to_ego)
                                FP_label_list.append(k)
                                attr_file["box_type"][new_size - 1] = 0
                            box_explained = flip_xy(pred_boxes_vertices[j])
                            figure_size = (8, 6)
                            # figure_size = (24, 18)
                            if high_rez:
                                # upscale the attributions if we are using high resolution input bev
                                # also increase figure size to accommodate the higher resolution
                                figure_size = (24, 18)
                                '''
                                note: it is more efficient to do the upscaling inside Captum's visualize_image_attr
                                only need to upscale a single 2d array instead of 64 such arrays
                                '''
                                # TODO: implement for kitti as well
                            overlay = 0.4
                            if overlay_orig_bev:
                                # need to horizontally flip the original bev image and rotate 90 degrees ccw to match location
                                # of explanations
                                bev_image_raw = np.flip(bev_fig_data, 0)
                                bev_image = np.rot90(bev_image_raw, k=1, axes=(0, 1))
                                if not gray_scale_overlay:
                                    overlay = 0.2


                            if channel_xai:  # generates channel-wise explanation
                                # TODO: this part needs fixing
                                for c in range(num_channels_viz):
                                    grad_viz = viz.visualize_image_attr(
                                        grad[:, :, c], bev_image, method="blended_heat_map", sign=sign, show_colorbar=True,
                                        title="Overlaid {}".format(method),
                                        alpha_overlay=overlay, fig_size=figure_size, upscale=high_rez)
                                    XAI_sample_path_str = XAI_cls_path_str + '/sample_{}'.format(i)
                                    if not os.path.exists(XAI_sample_path_str):
                                        os.makedirs(XAI_sample_path_str)
                                    os.chmod(XAI_sample_path_str, 0o777)

                                    XAI_box_relative_path_str = XAI_sample_path_str.split("tools/", 1)[1] + \
                                                                '/box_{}_{}_channel_{}_XQ_{}.png'.format(
                                                                    j, conf_mat[i][j], c, XQ)
                                    # print('XAI_box_path_str: {}'.format(XAI_box_path_str))
                                    grad_viz[0].savefig(XAI_box_relative_path_str, bbox_inches='tight', pad_inches=0.0)
                                    os.chmod(XAI_box_relative_path_str, 0o777)
                            else:
                                grad_viz = viz.visualize_image_attr(grad, bev_image, method="blended_heat_map", sign=sign,
                                                                    show_colorbar=True, title="Overlaid {}".format(method),
                                                                    alpha_overlay=overlay,
                                                                    fig_size=figure_size, upscale=high_rez)
                                XAI_box_relative_path_str = XAI_cls_path_str.split("tools/", 1)[1] + \
                                                            '/box_{}_{}_XQ_{}.png'.format(
                                                                j, conf_mat[i][j], XQ)
                                # print('XAI_box_path_str: {}'.format(XAI_box_path_str))
                                f.write('{}\n'.format(XQ))
                                # if high_rez:
                                #     box_explained = box_explained * 3
                                # box_explained = box_explained * 1.5
                                print("\nbox_explained: {}\n".format(box_explained))
                                polys = patches.Polygon(box_explained,
                                                        closed=True, fill=False, edgecolor='y', linewidth=1)
                                grad_viz[1].add_patch(polys)
                                grad_viz[0].savefig(XAI_box_relative_path_str, bbox_inches='tight', pad_inches=0.0)

                                if not missed_boxes_plotted[k] and len(missed_boxes[i]) > 0:
                                    missed_boxes_plotted[k] = True
                                    print('len(gt_boxes_vertices): {}'.format(len(gt_boxes_vertices)))
                                    print('missed_boxes[i]: {}'.format(missed_boxes[i]))
                                    for index in missed_boxes[i]:
                                        if index >= len(gt_boxes_vertices):
                                            break

                                        # obtain the missed gt box
                                        missed_box = gt_boxes_vertices[index]
                                        box_vertices = transform_box_coord(grad.shape[0], grad.shape[1], missed_box,
                                                                           dataset_name, high_rez, scaling_factor)
                                        box_vertices = flip_xy(box_vertices)
                                        # print("\nthe missed gt box: {}\n".format(missed_box))
                                        missed_ploys = patches.Polygon(box_vertices,
                                                                       closed=True, fill=False, edgecolor='c', linewidth=1)
                                        grad_viz[1].add_patch(missed_ploys)

                                        if len(missed_box) != 0:
                                            print("\nfinding anchor boxes matching the missed gt boxes")
                                            # obtain the anchor box matching that gt_box
                                            # this is valid in (x, y), and in meters, with upper left as (0,0)
                                            missed_box_loc = gt_boxes_loc[index]
                                            print('missed_box_loc: {}'.format(missed_box_loc))
                                            # y, x = transform_point_coord(grad.shape[0], grad.shape[1], missed_box_loc,
                                            #                                    dataset_name, high_rez, scaling_factor)
                                            # print('transformed box location (y,x): ({}, {})'.format(y, x))
                                            anchor_ind = find_anchor_index(dataset_name, missed_box_loc[1],
                                                                           missed_box_loc[0])
                                            print('the anchor index is: {}'.format(anchor_ind))
                                            lower_bound = max(anchor_ind - 4, 0)
                                            upper_bound = min(total_anchors, anchor_ind + 4)
                                            matched_anchor_vertices = None
                                            max_iou = 0
                                            experiment = False
                                            if experiment:
                                                box_gpu = batch_dict['anchor_boxes'][i][anchor_ind]
                                                box = box_gpu.cpu().detach().numpy()
                                                print("\nthe FN anchor that we want to visualize: {}\n".
                                                      format(box))
                                                box_expand = np.expand_dims(box, axis=0)
                                                anchor_vertices = kitti_bev.cuboid_to_bev(box[0], box[1], box[2], box[3],
                                                                                          box[4], box[5], box[6])
                                                if dataset_name == 'CadcDataset':
                                                    for vert_ind in range(len(anchor_vertices)):
                                                        anchor_vertices[vert_ind][0] += 50
                                                        anchor_vertices[vert_ind][1] += 50
                                                elif dataset_name == 'KittiDataset':
                                                    for vert_ind in range(len(anchor_vertices)):
                                                        # TODO: need to confirm the correctness of this
                                                        # anchor_vertices[vert_ind][0] += 50
                                                        anchor_vertices[vert_ind][1] += 39.68

                                                print("\nthe FN anchor_vertices: {}\n".format(anchor_vertices))
                                                viz_vertices = transform_box_coord(
                                                    grad.shape[0], grad.shape[1], anchor_vertices, dataset_name,
                                                    high_rez, scaling_factor
                                                )
                                                print("\nthe FN anchor_vertices scaled: {}\n".format(viz_vertices))
                                                viz_vertices = flip_xy(viz_vertices)
                                                # print("\nthe FN box with flipped vertices: {}\n".format(viz_vertices))
                                                viz_poly = patches.Polygon(
                                                    viz_vertices, closed=True, fill=False, edgecolor='m', linewidth=1)
                                                grad_viz[1].add_patch(viz_poly)
                                            for ind in range(lower_bound, upper_bound, 1):
                                                print('the anchor index to search: {}'.format(ind))
                                                box_gpu = batch_dict['anchor_boxes'][i][ind]
                                                box = box_gpu.cpu().detach().numpy()
                                                box_expand = np.expand_dims(box, axis=0)
                                                anchor_vertices = kitti_bev.cuboid_to_bev(box[0], box[1], box[2], box[3],
                                                                                          box[4], box[5], box[6])
                                                if dataset_name == 'CadcDataset':
                                                    for vert_ind in range(len(anchor_vertices)):
                                                        anchor_vertices[vert_ind][0] += 50
                                                        anchor_vertices[vert_ind][1] += 50
                                                elif dataset_name == 'KittiDataset':
                                                    for vert_ind in range(len(anchor_vertices)):
                                                        # TODO: need to confirm the correctness of this
                                                        # anchor_vertices[vert_ind][0] += 50
                                                        anchor_vertices[vert_ind][1] += 39.68
                                                gt_box = gt_dict[i]['boxes'][index]
                                                gt_box_expand = np.expand_dims(gt_box, axis=0)
                                                curr_iou, _ = calculate_bev_iou(gt_box_expand, box_expand)
                                                print('iou between the missed gt box and the corresponding anchor: {}'
                                                      .format(curr_iou))
                                                if curr_iou[0] > max_iou:
                                                    matched_anchor_vertices = anchor_vertices
                                                    max_iou = curr_iou[0]
                                            if matched_anchor_vertices is not None:
                                                matched_anchor_vertices = transform_box_coord(
                                                    grad.shape[0], grad.shape[1], matched_anchor_vertices, dataset_name,
                                                    high_rez, scaling_factor)
                                                matched_anchor_vertices = flip_xy(matched_anchor_vertices)
                                                matched_anchor_polys = patches.Polygon(
                                                    matched_anchor_vertices, closed=True, fill=False, edgecolor='m', linewidth=1)
                                                grad_viz[1].add_patch(matched_anchor_polys)
                                    XAI_missed_boxes_str = XAI_cls_path_str.split("tools/", 1)[1] + \
                                                           '/missed_gt_boxes.png'
                                    grad_viz[0].savefig(XAI_missed_boxes_str, bbox_inches='tight', pad_inches=0.0)
                                plt.close('all')
                                os.chmod(XAI_box_relative_path_str, 0o777)
                                # print('box_{}_{}.png is saved in {}'.format(j,conf_mat[i][j],XAI_cls_path_str))
                    pos_attr_file.close()
                    neg_attr_file.close()
            #     break  # only processing one sample to save time
            #
            if batch_num == batches_to_analyze - 1:
                break  # just process a limited number of batches
            # # break
            # # just need one explanation for example
    finally:
        f.write("total number of boxes analyzed: {}".format(box_cnt))
        f.close()

        # plotting
        all_xq = XAI_res_path_str + "/all_xq.csv"
        tp_xq = XAI_res_path_str + "/tp_xq.csv"
        fp_xq = XAI_res_path_str + "/fp_xq.csv"
        fnames = ['class_score', 'XQ', 'dist_to_ego', 'class_label']
        write_to_csv(all_xq, fnames, cls_score_list, XQ_list, dist_list, label_list)
        write_to_csv(tp_xq, fnames, TP_score_list, TP_XQ_list, TP_dist_list, TP_label_list)
        write_to_csv(fp_xq, fnames, FP_score_list, FP_XQ_list, FP_dist_list, FP_label_list)

        fig, axs = plt.subplots(3, figsize=(10, 20))
        axs[0].scatter(cls_score_list, XQ_list)
        axs[0].set_title('All Boxes')
        axs[1].scatter(TP_score_list, TP_XQ_list)
        axs[1].set_title('TP Boxes')
        axs[2].scatter(FP_score_list, FP_XQ_list)
        axs[2].set_title('FP Boxes')
        for ax in axs:
            ax.set(xlabel='class scores', ylabel='XQ')
        plt.savefig("{}/XQ_class_score.png".format(XAI_result_path))
        plt.close()

        fig, axs = plt.subplots(3, figsize=(10, 20))
        axs[0].hist(XQ_list, bins=20)
        axs[0].set_title('All Boxes')
        axs[1].hist(TP_XQ_list, bins=20)
        axs[1].set_title('TP Boxes')
        axs[2].hist(FP_XQ_list, bins=20)
        axs[2].set_title('FP Boxes')
        for ax in axs:
            ax.set(xlabel='XQ', ylabel='box_count')
        plt.savefig("{}/pred_box_XQ_histograms.png".format(XAI_result_path))
        plt.close()

        fig, axs = plt.subplots(3, figsize=(10, 20))
        axs[0].scatter(dist_list, XQ_list)
        axs[0].set_title('All Boxes')
        axs[1].scatter(TP_dist_list, TP_XQ_list)
        axs[1].set_title('TP Boxes')
        axs[2].scatter(FP_dist_list, FP_XQ_list)
        axs[2].set_title('FP Boxes')
        for ax in axs:
            ax.set(xlabel='distance to ego', ylabel='XQ')
        plt.savefig("{}/XQ_distance_to_ego.png".format(XAI_result_path))
        plt.close()

        if plot_class_wise:
            # generate 3 plots for each class
            for i in range(len(class_name_list)):
                class_name = class_name_list[i]
                selected = [j for j in range(len(label_list)) if label_list[j] == i]
                selected_TP = [j for j in range(len(TP_label_list)) if TP_label_list[j] == i]
                selected_FP = [j for j in range(len(FP_label_list)) if FP_label_list[j] == i]
                cls_score_list_i = list_selection(cls_score_list, selected)
                XQ_list_i = list_selection(XQ_list, selected)
                dist_list_i = list_selection(dist_list, selected)
                TP_score_list_i = list_selection(TP_score_list, selected_TP)
                TP_XQ_list_i = list_selection(TP_XQ_list, selected_TP)
                TP_dist_list_i = list_selection(TP_dist_list, selected_TP)
                FP_score_list_i = list_selection(FP_score_list, selected_FP)
                FP_XQ_list_i = list_selection(FP_XQ_list, selected_FP)
                FP_dist_list_i = list_selection(FP_dist_list, selected_FP)

                fig, axs = plt.subplots(3, figsize=(10, 20))
                axs[0].scatter(cls_score_list_i, XQ_list_i)
                axs[0].set_title('All Boxes')
                axs[1].scatter(TP_score_list_i, TP_XQ_list_i)
                axs[1].set_title('TP Boxes')
                axs[2].scatter(FP_score_list_i, FP_XQ_list_i)
                axs[2].set_title('FP Boxes')
                for ax in axs:
                    ax.set(xlabel='class scores', ylabel='XQ')
                plt.savefig("{}/XQ_class_score_{}.png".format(XAI_result_path, class_name))
                plt.close()

                fig, axs = plt.subplots(3, figsize=(10, 20))
                axs[0].hist(XQ_list_i, bins=20)
                axs[0].set_title('All Boxes')
                axs[1].hist(TP_XQ_list_i, bins=20)
                axs[1].set_title('TP Boxes')
                axs[2].hist(FP_XQ_list_i, bins=20)
                axs[2].set_title('FP Boxes')
                for ax in axs:
                    ax.set(xlabel='XQ', ylabel='box_count')
                plt.savefig("{}/pred_box_XQ_histograms_{}.png".format(XAI_result_path, class_name))
                plt.close()

                fig, axs = plt.subplots(3, figsize=(10, 20))
                axs[0].scatter(dist_list_i, XQ_list_i)
                axs[0].set_title('All Boxes')
                axs[1].scatter(TP_dist_list_i, TP_XQ_list_i)
                axs[1].set_title('TP Boxes')
                axs[2].scatter(FP_dist_list_i, FP_XQ_list_i)
                axs[2].set_title('FP Boxes')
                for ax in axs:
                    ax.set(xlabel='distance to ego', ylabel='XQ')
                plt.savefig("{}/XQ_distance_to_ego_{}.png".format(XAI_result_path, class_name))
                plt.close()

        print("{} boxes analyzed".format(box_cnt))
        print("--- {} seconds ---".format(time.time() - start_time))


if __name__ == '__main__':
    main()
