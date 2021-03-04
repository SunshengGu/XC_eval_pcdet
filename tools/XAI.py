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


def calculate_iou(gt_boxes, pred_boxes, dataset_name, ret_overlap=False):
    # see pcdet/datasets/kitti/kitti_object_eval_python/eval.py for explanation
    z_axis = 1  # welp... it is really the y-axis
    z_center = 1.0
    if dataset_name == 'CadcDataset':
        # z_axis = 2
        z_center = 0.5
    overlap = d3_box_overlap(gt_boxes, pred_boxes, z_axis=z_axis, z_center=z_center)
    # pick max iou wrt to each detection box
    # print("\nverification in calculate_iou:")
    # print("number of gt_boxes: {}".format(len(gt_boxes)))
    # print("number of pred_boxes: {}".format(len(pred_boxes)))
    # print('overlap.shape: {}'.format(overlap.shape))
    iou, gt_index = np.max(overlap, axis=0), np.argmax(overlap, axis=0)
    if ret_overlap:
        return iou, gt_index, overlap
    return iou, gt_index


def calculate_overlaps(gt_boxes, pred_boxes, dataset_name):
    # see pcdet/datasets/kitti/kitti_object_eval_python/eval.py for explanation
    z_axis = 1
    z_center = 1.0
    if dataset_name == 'CadcDataset':
        # z_axis = 2
        z_center = 0.5
    overlaps = d3_box_overlap(gt_boxes, pred_boxes, z_axis=z_axis, z_center=z_center)
    return overlaps


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
    # print('overlap.shape: {}'.format(overlap.shape))
    ious = np.max(overlap.numpy(), axis=1)
    ind = 0
    missed_gt_idx = []
    missed_gt_label = []
    for iou in ious:
        if iou < iou_unmatching_thresholds[gt_labels[ind]]:
            # print('gt label is : {}'.format(gt_labels[ind]))
            # print('maximum iou is: {}'.format(iou))
            # print('matching thresh is: {}'.format(iou_unmatching_thresholds[gt_labels[ind]]))
            missed_gt_idx.append(ind)
            missed_gt_label.append(gt_labels[ind])
        ind += 1
    # print('missed_gt_idx: {}'.format(missed_gt_idx))
    return missed_gt_idx, missed_gt_label


def find_missing_gt_3d(gt_dict, pred_labels, overlap, iou_unmatching_thresholds, dataset_name, cls_names):
    # TODO: avoid hard coding the distance threshold for Kitti
    gt_boxes = gt_dict['boxes']
    gt_labels = gt_dict['labels']
    # print("\nverification in find_missing_gt_3d:")
    # print('overlap.shape: {}'.format(overlap.shape))
    ious, pred_idx = np.max(overlap, axis=1), np.argmax(overlap, axis=1)
    print("\nnumber of gt boxes according to the `info` attribute of the dataset object: {}\n".format(len(gt_boxes)))
    print("\ngt boxes according to the `info` attribute of the dataset object: {}\n".format(gt_boxes))
    # print("number of pred boxes: {}".format(len(pred_boxes)))
    # print("len(ious): {}".format(len(ious)))
    ind = 0
    missed_gt_idx = []
    missed_gt_label = []
    missed_gt_iou = []
    partial_missed_gt_idx = []
    partial_missed_gt_label = []
    partial_missed_gt_iou = []
    misclassified_gt_idx = []
    misclassified_gt_label = []
    misclassified_gt_iou = []
    for iou, pred_id in zip(ious, pred_idx):
        # # this is the highest IoU among all pred boxes with the particular gt box
        # print("iou: {}".format(iou))
        # print("iou thresh: {}".format(iou_unmatching_thresholds[gt_labels[ind]]))
        if iou <= 0.0:
            # these gt boxes are totally missed
            missed_gt_idx.append(ind)
            missed_gt_label.append(gt_labels[ind])
            missed_gt_iou.append(iou)
        elif iou < iou_unmatching_thresholds[gt_labels[ind]]:
            if cls_names[pred_labels[pred_id]] == gt_labels[ind]:
                # these gt boxes are correctly classified but partially missed
                partial_missed_gt_idx.append(ind)
                partial_missed_gt_label.append(gt_labels[ind])
                partial_missed_gt_iou.append(iou)
            else:
                # these gt boxes are incorrectly classified and partially missed
                missed_gt_idx.append(ind)
                missed_gt_label.append(gt_labels[ind])
                missed_gt_iou.append(iou)
        elif cls_names[pred_labels[pred_id]] != gt_labels[ind]:
            # exceeded iou thresholds but missclassifed
            misclassified_gt_idx.append(ind)
            misclassified_gt_label.append(gt_labels[ind])
            misclassified_gt_iou.append(iou)
        ind += 1
    # print('missed_gt_idx: {}'.format(missed_gt_idx))
    return missed_gt_idx, missed_gt_label, missed_gt_iou, partial_missed_gt_idx, partial_missed_gt_label, partial_missed_gt_iou, misclassified_gt_idx, misclassified_gt_label, misclassified_gt_iou


def main():
    """
    important variables:
        Note:
            number code for box types:
            0 - FP
            1 - TP
            2 - FN_missed
            3 - FN_partially_missed
            4 - FN_misclassified
        FN_analysis: indicates if we are doing FN analysis
        skip_TP_FP: indicates if we are skipping TP and FP analysis
        max_obj_cnt: The maximum number of objects in an image, right now set to 50
        batches_to_analyze: The number of batches for which explanations are generated
        method: Explanation method used.
        attr_shown: What type of attributions are shown, can be 'absolute_value', 'positive', 'negative', or 'all'.
        high_rez: Whether to use higher resolution (in the case of CADC, this means 2000x2000 instead of 400x400)
        overlay_orig_bev: If True, overlay attributions onto the original point cloud BEV. If False, overlay
            attributions onto the 2D pseudoimage.
        overlay: Parameter for attr visualization
        mult_by_inputs: Whether to pointwise-multiply Integrated Gradients attributions with the input being explained
            (i.e., the 2D pseudoimage in the case of PointPillar).
        channel_xai: Whether to generate channel-wise attribution heat map for the pseudoimage
        gray_scale_overlay: Whether to convert the input image into gray scale.
        IOU_3D: Indicate if we are using 3D or 2D IoU to find anchors that match the FN boxes
        d3_iou_thresh: iou thresholds from the evaluation script
        FN_search_range: The vicinity within within which we search for the highest-IoU anchor box
        plot_class_wise: Whether to generate plots for each class.
        box_margin: Margin for the bounding boxes in meters
        orig_bev_w: Width of the original BEV in # pixels. For CADC, width = height. Note that this HAS to be an
            integer multiple of pseudo_img_w.
        orig_bev_h: Height of the original BEV in # pixels. For CADC, width = height.
        dpi_division_factor: Divide image dimension in pixels by this number to get dots per inch (dpi). Lower this
            parameter to get higher dpi.
    :return:
    """
    box_debug = False
    run_all = False
    plot_enlarged_pred = True
    unmatched_TP_FP_pred = 0
    # use_anchor_directly = True
    FN_analysis = False
    TP_FP_analysis = True
    box_cnt = 0
    TP_box_cnt = 0
    FP_box_cnt = 0
    missed_box_cnt = 0
    partly_missed_box_cnt = 0
    misclassified_box_cnt = 0
    missed_box_analyzed = 0
    partly_missed_box_analyzed = 0
    misclassified_box_analyzed = 0
    start_time = time.time()
    max_obj_cnt = 100
    batches_to_analyze = 55
    method = 'IG'
    use_trapezoid = False
    ignore_thresh = 0.0
    verify_box = False
    attr_to_csv = False
    attr_shown = 'positive'
    high_rez = True
    overlay_orig_bev = True
    overlay = 0.4
    mult_by_inputs = True
    channel_xai = False
    gray_scale_overlay = True
    plot_class_wise = False
    IOU_3D = True
    tight_iou = False
    d3_iou_thresh = []
    d3_iou_thresh_dict = {}
    FN_search_range = 5000
    color_map = 'jet'
    box_margin = 0.2
    if gray_scale_overlay:
        color_map = 'gist_yarg'
    scaling_factor = 5.0
    args, cfg, x_cfg = parse_config()
    dataset_name = cfg.DATA_CONFIG.DATASET
    if dataset_name == 'KittiDataset':
        if tight_iou:
            d3_iou_thresh = [0.7, 0.5, 0.5]
            d3_iou_thresh_dict = {'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}
        else:
            d3_iou_thresh = [0.5, 0.25, 0.25]
            d3_iou_thresh_dict = {'Car': 0.5, 'Pedestrian': 0.25, 'Cyclist': 0.25}
    elif dataset_name == 'CadcDataset':
        if tight_iou:
            d3_iou_thresh = [0.7, 0.5, 0.7]
            d3_iou_thresh_dict = {'Car': 0.7, 'Pedestrian': 0.5, 'Truck': 0.7}
        else:
            d3_iou_thresh = [0.5, 0.25, 0.5]
            d3_iou_thresh_dict = {'Car': 0.5, 'Pedestrian': 0.25, 'Truck': 0.5}
    num_channels_viz = min(32, cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES)
    figure_size = (8, 6)
    # figure_size = (24, 18)
    if high_rez:
        # upscale the attributions if we are using high resolution input bev
        # also increase figure size to accommodate the higher resolution
        figure_size = (24, 18)
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
    dpi_division_factor = 20.0  # 20.0
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
    class_name_list = cfg.CLASS_NAMES

    cadc_bev = CADC_BEV(dataset=test_set, scale_to_pseudoimg=(not high_rez), class_name=class_name_list,
                        background='black',
                        scale=scaling_factor, cmap=color_map, dpi_factor=dpi_division_factor, margin=box_margin)
    kitti_bev = KITTI_BEV(dataset=test_set, scale_to_pseudoimg=(not high_rez), class_name=class_name_list,
                          background='black',
                          result_path='output/kitti_models/pointpillar/default/eval/epoch_7728/val/default/result.pkl',
                          scale=scaling_factor, cmap=color_map, dpi_factor=dpi_division_factor, margin=box_margin)
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

    # get the date and time to create a folder for the specific time when this script is run
    now = datetime.datetime.now()
    dt_string = now.strftime("%b_%d_%Y_%H_%M_%S")
    # get current working directory
    cwd = os.getcwd()
    rez_string = 'LowResolution'
    if high_rez:
        rez_string = 'HighResolution'
    # create directory to store results just for this run, include the method in folder name
    XAI_result_path = os.path.join(cwd, 'XAI_results/{}_{}_{}_{}_batches'.format(
        dt_string, method, dataset_name, batches_to_analyze))
    XAI_attr_path = os.path.join(cwd, 'XAI_attributions/{}_{}_{}_{}_batches'.format(
        dt_string, method, dataset_name, batches_to_analyze))
    if run_all:
        XAI_result_path = os.path.join(cwd, 'XAI_results/{}_{}_{}_{}_batches'.format(
            dt_string, method, dataset_name, "all"))
        XAI_attr_path = os.path.join(cwd, 'XAI_attributions/{}_{}_{}_{}_batches'.format(
            dt_string, method, dataset_name, "all"))

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
    f.write('XQ ignore threshold: {}\n'.format(ignore_thresh))
    f.write('DPI Division Factor: {}\n'.format(dpi_division_factor))
    f.write('Attributions Visualized: {}\n'.format(attr_shown))
    f.write('Bounding box margin: {}\n'.format(box_margin))
    f.write('FN search range: {}\n'.format(FN_search_range))
    info_file_2 = XAI_attr_path_str + "/XAI_information.txt"
    f2 = open(info_file_2, "w")
    f2.write('High Resolution: {}\n'.format(high_rez))
    f2.write('XQ ignore threshold: {}\n'.format(ignore_thresh))
    f2.write('DPI Division Factor: {}\n'.format(dpi_division_factor))
    f2.write('Attributions Visualized: {}\n'.format(attr_shown))
    f2.write('Bounding box margin: {}\n'.format(box_margin))
    f2.write('FN search range: {}\n'.format(FN_search_range))
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
    # f.write('Recording XQ values: \n')
    if IOU_3D:
        f.write("IoU for FN anchor: 3D\n")
    else:
        f.write("IoU for FN anchor: 2D\n")
    f.write("Search vicinity for FN anchor: {}\n".format(FN_search_range))
    os.chmod(info_file, 0o777)
    os.chmod(info_file_2, 0o777)
    try:
        for batch_num, batch_dict in enumerate(test_loader):
            if (not run_all) and batch_num == batches_to_analyze:
                break  # just process a limited number of batches
            print("\nbatch_num: {}\n".format(batch_num))
            if batch_num != 49:
                continue
            # check_list = [459]
            # if batch_num not in check_list:
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
            anchors_scores = None
            with torch.no_grad():
                anchors_scores = model(dummy_tensor, batch_dict)
            pred_dicts = batch_dict['pred_dicts']
            batch_anchors = batch_dict['anchor_boxes']
            batch_anchor_labels = batch_dict['anchor_labels']
            batch_anchor_sig = batch_dict['sigmoid_anchor_scores']

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
            missed_boxes_label = []
            missed_boxes_iou = []
            partial_missed_boxes = []
            partial_missed_boxes_label = []
            partial_missed_boxes_iou = []
            misclassified_boxes = []
            misclassified_boxes_label = []
            misclassified_boxes_iou = []
            missed_boxes_plotted = []
            batch_pred_scores = []
            gt_exist = []

            pred_box_iou = []
            batch_num_boxes = []
            batch_pred_boxes = []
            batch_pred_labels = []
            batch_pred_scores = []

            for i in range(batch_dict['batch_size']):  # i is input image id in the batch
                anchor_select_i = batch_dict['anchor_selections'][i]
                anchor_select_i_cpu = batch_dict['anchor_selections'][i].cpu().detach().numpy()
                # pred_boxes_i = None
                # pred_labels_i = None
                box_cnt_list = []
                box_cnt_list.append(batch_dict['box_count'][i])
                # box_cnt_list.append(len(conf_mat[i]))
                box_cnt_list.append(max_obj_cnt)
                box_cnt_list.append(len(batch_dict['anchor_selections'][i]))
                # print("\nlen(batch_dict['anchor_selections'][i]): {}\n".format(len(batch_dict['anchor_selections'][i])))
                num_boxes = min(box_cnt_list)
                batch_num_boxes.append(num_boxes)
                # print("\nshowing box counts:")
                # print("batch_dict['box_count'][i]: {}".format(batch_dict['box_count'][i]))
                # # print("len(conf_mat[i]): {}".format(len(conf_mat[i])))
                # print("len(batch_dict['anchor_selections'][i]): {}".format(len(batch_dict['anchor_selections'][i])))

                anchor_boxes_i = batch_anchors[i][anchor_select_i].cpu().numpy()
                anchor_boxes_all_i = batch_anchors[i].cpu().numpy()
                anchor_labels_i = batch_anchor_labels[i][anchor_select_i].cpu().numpy()
                anchor_labels_all_i = batch_anchor_labels[i].cpu().numpy()

                anchor_sig_all_i = batch_anchor_sig[i].cpu().numpy()

                # pred_boxes = anchor_boxes_i[:num_boxes, :]
                # pred_labels = anchor_labels_i[:num_boxes]
                pred_boxes = pred_dicts[i]['pred_boxes'].cpu().numpy()
                pred_labels = pred_dicts[i]['pred_labels'].cpu().numpy() - 1
                pred_scores = pred_dicts[i]['pred_scores'].cpu().numpy()

                batch_pred_boxes.append(pred_boxes)
                batch_pred_labels.append(pred_labels)
                batch_pred_scores.append(pred_scores)

                scores_for_anchors = anchors_scores[i].cpu().detach().numpy()
                # print("\nscores_for_anchors.shape: {}\n".format(scores_for_anchors.shape))
                # # print("type(gt_dict[i]['boxes']) before filtering: {}".format(type(gt_dict[i]['boxes'])))
                # print("type(gt_dict[i]['labels']) before filtering: {}".format(type(gt_dict[i]['labels'])))
                missed_boxes_plotted = []
                for c in range(len(class_name_list)):
                    missed_boxes_plotted.append(False)
                conf_mat_frame = []
                # pred_boxes = pred_dicts[i]['pred_boxes'].cpu().numpy()
                # pred_labels = pred_dicts[i]['pred_labels'].cpu().numpy() - 1
                # filter out out-of-range gt boxes for Kitti
                if dataset_name == 'KittiDataset':
                    filtered_gt_boxes = []
                    filtered_gt_labels = []
                    for gt_ind in range(len(gt_dict[i]['boxes'])):
                        x, y = gt_dict[i]['boxes'][gt_ind][0], gt_dict[i]['boxes'][gt_ind][1]
                        if (-1.5 < x < 70.5) and (-40.5 < y < 40.5):
                            filtered_gt_boxes.append(gt_dict[i]['boxes'][gt_ind])
                            filtered_gt_labels.append(gt_dict[i]['labels'][gt_ind])
                    if len(filtered_gt_boxes) != 0:
                        gt_dict[i]['boxes'] = np.vstack(filtered_gt_boxes)
                    else:
                        gt_dict[i]['boxes'] = filtered_gt_boxes
                    gt_dict[i]['labels'] = filtered_gt_labels
                # print("type(gt_dict[i]['boxes']) after filtering: {}".format(type(gt_dict[i]['boxes'])))

                # need to handle the case when no gt boxes exist
                gt_present = True
                if len(gt_dict[i]['boxes']) == 0:
                    gt_present = False
                gt_exist.append(gt_present)
                if not gt_present:
                    for j in range(len(pred_scores)):
                        curr_pred_score = pred_scores[j]
                        adjusted_pred_boxes_label = pred_labels[j]
                        pred_name = class_name_list[adjusted_pred_boxes_label]
                        if curr_pred_score >= score_thres:
                            conf_mat_frame.append('FP')
                            # print("pred_box_ind: {}, pred_label: {}, didn't match any gt boxes".format(j, pred_name))
                        else:
                            conf_mat_frame.append('ignore')  # these boxes do not meet score thresh, ignore for now
                            print("pred_box {} has score below threshold: {}".format(j, curr_pred_score))
                    conf_mat.append(conf_mat_frame)
                    continue

                # when we indeed have some gt boxes
                iou, gt_index, overlaps = calculate_iou(gt_dict[i]['boxes'], pred_boxes, dataset_name, ret_overlap=True)
                pred_box_iou.append(iou)
                # *************************** FN identification start ******************************************** #
                missed_gt_index, missed_gt_label, missed_gt_iou, part_missed_gt_id, part_missed_gt_label, part_missed_gt_iou, miscls_gt_id, miscls_gt_label, miscls_gt_iou = find_missing_gt_3d(
                    gt_dict[i], pred_labels, overlaps, d3_iou_thresh_dict, dataset_name, class_name_list)
                missed_boxes.append(missed_gt_index)
                missed_boxes_label.append(missed_gt_label)
                missed_boxes_iou.append(missed_gt_iou)
                partial_missed_boxes.append(part_missed_gt_id)
                partial_missed_boxes_label.append(part_missed_gt_label)
                partial_missed_boxes_iou.append(part_missed_gt_iou)
                misclassified_boxes.append(miscls_gt_id)
                misclassified_boxes_label.append(miscls_gt_label)
                misclassified_boxes_iou.append(miscls_gt_iou)
                # *************************** FN identification end ******************************************** #
                # sample_pred_scores = pred_dicts[i]['pred_scores'].cpu().numpy()
                # batch_pred_scores.append(sample_pred_scores)
                # *************************** FP TP identification start ******************************************** #
                # print("\nnumber of pred boxes according to len(pred_scores): {}\n".format(
                #     len(pred_scores)))

                for j in range(len(pred_scores)):  # j is prediction box id in the i-th image
                    gt_cls = gt_dict[i]['labels'][gt_index[j]]
                    iou_thresh_3d = d3_iou_thresh_dict[gt_cls]
                    curr_pred_score = pred_scores[j]
                    if curr_pred_score >= score_thres:
                        adjusted_pred_boxes_label = pred_labels[j]
                        pred_name = class_name_list[adjusted_pred_boxes_label]
                        if iou[j] >= iou_thresh_3d:
                            if gt_cls == pred_name:
                                conf_mat_frame.append('TP')
                            else:
                                conf_mat_frame.append('FP')
                            if box_debug:
                                print("pred_box_ind: {}, pred_label: {}, gt_ind is {}, gt_label: {}, iou: {}".format(
                                    j, pred_name, gt_index[j], gt_cls, iou[j]))
                        elif iou[j] > 0:
                            conf_mat_frame.append('FP')
                            if box_debug:
                                print("pred_box_ind: {}, pred_label: {}, gt_ind is {}, gt_label: {}, iou: {}".format(
                                    j, pred_name, gt_index[j], gt_cls, iou[j]))
                        else:
                            conf_mat_frame.append('FP')
                            if box_debug:
                                print("pred_box_ind: {}, pred_label: {}, didn't match any gt boxes".format(j, pred_name))
                    else:
                        conf_mat_frame.append('ignore')  # these boxes do not meet score thresh, ignore for now
                        print("pred_box {} has score below threshold: {}".format(j, curr_pred_score))
                conf_mat.append(conf_mat_frame)
                # *************************** FP TP identification end ******************************************** #

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
                padded_pred_boxes_vertices = None
                gt_boxes_vertices = None
                gt_boxes_loc = None
                pred_boxes_loc = None
                total_attr = 0  # total amount of attributions in a box
                missed_box_counted_i = False
                partly_missed_box_counted_i = False
                misclassified_box_counted_i = False

                anchor_boxes_all_i = batch_anchors[i].cpu().numpy()
                anchor_labels_all_i = batch_anchor_labels[i].cpu().numpy()

                # pred_boxes_i = batch_pred_boxes[i]
                # pred_labels_i = batch_pred_labels[i]

                pred_boxes_i = pred_dicts[i]['pred_boxes'].cpu().numpy()
                pred_labels_i = pred_dicts[i]['pred_labels'].cpu().numpy() - 1
                pred_boxes_for_pts = None
                if dataset_name == 'CadcDataset':
                    # TODO: implement box padding for Cadc
                    # cadc_bev.set_pred_box(pred_boxes_i, pred_labels_i)
                    bev_fig, bev_fig_data = cadc_bev.get_bev_image(img_idx)
                    pred_boxes_vertices = cadc_bev.pred_poly
                    pred_boxes_for_pts = cadc_bev.pred_boxes_for_cnt
                    padded_pred_boxes_vertices = cadc_bev.pred_poly_expand
                    gt_boxes_vertices = cadc_bev.gt_poly
                    gt_boxes_loc = cadc_bev.gt_loc
                    pred_boxes_loc = cadc_bev.pred_loc
                elif dataset_name == 'KittiDataset':
                    kitti_bev.set_pred_box(pred_boxes_i, pred_labels_i)
                    pred_boxes_for_pts = kitti_bev.pred_boxes_for_cnt
                    bev_fig, bev_fig_data = kitti_bev.get_bev_image(img_idx)
                    pred_boxes_vertices = kitti_bev.pred_poly
                    padded_pred_boxes_vertices = kitti_bev.pred_poly_expand
                    gt_boxes_vertices = kitti_bev.gt_poly
                    gt_boxes_loc = kitti_bev.gt_loc
                    pred_boxes_loc = kitti_bev.pred_loc
                bev_image = np.transpose(PseudoImage2D[i].cpu().detach().numpy(), (1, 2, 0))
                if overlay_orig_bev:
                    # need to horizontally flip the original bev image and rotate 90 degrees ccw to match location
                    # of explanations
                    bev_image_raw = np.flip(bev_fig_data, 0)
                    bev_image = np.rot90(bev_image_raw, k=1, axes=(0, 1))
                    if not gray_scale_overlay:
                        overlay = 0.2
                # print("\ngt box locations:")
                # for gt_ind in range(len(gt_boxes_loc)):
                #     print("location of gt_box {}: {}".format(gt_ind, gt_boxes_loc[gt_ind]))
                pred_boxes = pred_boxes_i
                pred_labels = pred_labels_i
                # pred_scores = batch_pred_scores[i]
                pred_scores = pred_dicts[i]['pred_scores'].cpu().numpy()


                '''
                format for pred_boxes[box_index]: x,y,z,l,w,h,theta, for both CADC and KITTI
                format for pred_boxes_vertices[box_index]: ((x1,y1), ... ,(x4,y4))
                '''
                # print("pred_boxes.shape[0]: {} \n"
                #       "len(pred_boxes_vertices): {}".format(pred_boxes.shape[0], len(pred_boxes_vertices)))

                XAI_sample_path_str = XAI_batch_path_str + '/sample_{}'.format(i)
                # if not os.path.exists(XAI_sample_path_str):
                os.makedirs(XAI_sample_path_str)
                os.chmod(XAI_sample_path_str, 0o777)

                XAI_attr_sample_path_str = XAI_attr_batch_path_str + '/sample_{}'.format(i)
                # if not os.path.exists(XAI_attr_sample_path_str):
                os.makedirs(XAI_attr_sample_path_str)
                os.chmod(XAI_attr_sample_path_str, 0o777)

                # generating corners for counting points in box
                pred_boxes_corners = box_utils.boxes_to_corners_3d(pred_boxes)
                pred_boxes_for_pts_corners = box_utils.boxes_to_corners_3d(pred_boxes_for_pts)
                # TODO: get the points
                points = None
                if dataset_name == "KittiDataset":
                    points = kitti_bev.lidar_data
                elif dataset_name == "CadcDataset":
                    points = cadc_bev.lidar_data

                for k in range(3):  # iterate through the 3 classes
                    # num_boxes = min(box_cnt_list) - 1
                    # print('num_boxes: {}'.format(num_boxes))
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
                    expanded_boxes = attr_file.create_dataset(
                        "pred_boxes_expand", (0, 4, 2), maxshape=(None, 4, 2), dtype="float32", chunks=True)
                    # encode box types with numbers: 0-FP, 1-TP, 2-FN_missed, 3-FN_partially_missed, 4-FN_misclassified
                    boxes_type = attr_file.create_dataset(
                        "box_type", (0, 1), maxshape=(None, 1), dtype="uint8", chunks=True)
                    pred_boxes_location = attr_file.create_dataset(
                        "pred_boxes_loc", (0, 2), maxshape=(None, 2), dtype="float32", chunks=True)
                    pred_boxes_score = attr_file.create_dataset(
                        "box_score", (0, 1), maxshape=(None, 1), dtype="float32", chunks=True)
                    pred_box_points = attr_file.create_dataset(
                        "points_in_box", (0, 1), maxshape=(None, 1), dtype="int32", chunks=True)
                    pred_boxes_score_all = attr_file.create_dataset(
                        "box_score_all", (0, 3), maxshape=(None, 3), dtype="float32", chunks=True)

                    if TP_FP_analysis:
                        for j in range(batch_num_boxes[i]):  # iterate through each box in this sample
                            # if j > 5:
                            #     # save time for debugging
                            #     break
                            if not box_validation(pred_boxes[j], pred_boxes_vertices[j], dataset_name):
                                # print("pred_boxes[{}] didn't matched the computed vertices".format(j))
                                continue  # can't compute XQ if the boxes don't match
                            '''
                            Note:
                            Sometimes the size of batch_dict['anchor_selections'][i] changes for no reason.
                            Hence need this double check here
                            '''
                            if j >= batch_dict['box_count'][i] or j >= len(batch_dict['anchor_selections'][i]):
                                break
                            box_w = pred_boxes[j][3]
                            box_l = pred_boxes[j][4]
                            box_x = pred_boxes_loc[j][0]
                            box_y = pred_boxes_loc[j][1]
                            dist_to_ego = np.sqrt(box_x * box_x + box_y * box_y)

                            # if k + 1 == pred_dicts[i]['pred_labels'][j] and conf_mat[i][j] != 'ignore':
                            if k == pred_labels[j] and conf_mat[i][j] != 'ignore':
                                # compute contribution for the positive class only, k+1 because PCDet labels start at 1
                                # i.e., generate reason as to why the j-th box is classified as the k-th class
                                target = (j, k)
                                anchor_id = batch_dict['anchor_selections'][i][j]
                                anchor_score_all = anchor_sig_all_i[anchor_id]
                                anchor_score = np.max(anchor_score_all)
                                anchor_cls_label = anchor_labels_all_i[anchor_id]
                                anchor_class = class_name_list[anchor_cls_label]
                                # print("\nanchor_id for the {}th pred box is {}".format(j, anchor_id))
                                # print("type(j): {}".format(type(j)))
                                # print("type(anchor_id): {}".format(type(anchor_id)))
                                # print("this is the anchor box: {}".format(anchor_boxes_all_i[anchor_id]))
                                # print("this is the anchor label: {}".format(anchor_class))
                                # print("this is the anchor score: {}".format(anchor_score))
                                # print("this is the anchor box: {}".format(anchor_boxes_i[anchor_id]))
                                if use_anchor_idx:
                                    # if we are using anchor box outputs, need to use the indices in for anchor boxes
                                    target = (anchor_id, k)
                                alternate = [j-4, j-3, j-2, j-1, j+1, j+2, j+3, j+4]
                                # target_cls = k
                                box_matched = True
                                box_dist_match_thresh = 0.0001
                                box_score_match_thresh = 1e-6
                                if (abs(box_x - anchor_boxes_all_i[anchor_id][0]) >= box_dist_match_thresh or \
                                        abs(box_y - anchor_boxes_all_i[anchor_id][1]) >= box_dist_match_thresh or \
                                        anchor_cls_label != k or abs(anchor_score - pred_scores[j]) >= box_score_match_thresh):
                                    box_matched = False
                                    for box_id in alternate:
                                        # print("batch_num_boxes[i]: {}".format(batch_num_boxes[i]))
                                        # print("len(batch_dict['anchor_selections'][i]: {}".format(
                                        #     len(batch_dict['anchor_selections'][i])))
                                        limit = min(len(batch_dict['anchor_selections'][i]), batch_num_boxes[i])
                                        if 0 <= box_id < limit:
                                            anchor_id = batch_dict['anchor_selections'][i][box_id]
                                            anchor_score_all = anchor_sig_all_i[anchor_id]
                                            anchor_score = np.max(anchor_score_all)
                                            anchor_cls_label = anchor_labels_all_i[anchor_id]
                                            anchor_class = class_name_list[anchor_cls_label]
                                            if (abs(box_x - anchor_boxes_all_i[anchor_id][0]) < box_dist_match_thresh and \
                                                    abs(box_y - anchor_boxes_all_i[anchor_id][1]) < box_dist_match_thresh and \
                                                    anchor_cls_label == k and \
                                                    abs(anchor_score - pred_scores[j]) < box_score_match_thresh):
                                                # found the right box, update target
                                                target = (anchor_id, k)
                                                box_matched = True
                                                pred_box_id = box_id
                                                break

                                if not box_matched:
                                    unmatched_TP_FP_pred += 1
                                    f.write("{}th pred box in batch {} image {} does not have a matching anchor\n".format(
                                        j, batch_num, i
                                    ))
                                    continue

                                sign = attr_shown
                                if method == 'Saliency':
                                    if len(attributions[j][k].shape) == 1:  # compute attributions only when necessary
                                        attributions[j][k] = saliency2D.attribute(
                                            PseudoImage2D, target=target, additional_forward_args=batch_dict)
                                if method == 'IG':
                                    if len(attributions[j][k].shape) == 1:  # compute attributions only when necessary
                                        if use_trapezoid:
                                            attributions[j][k] = ig2D.attribute(
                                                PseudoImage2D, baselines=PseudoImage2D * 0, target=target,
                                                additional_forward_args=batch_dict, n_steps=steps,
                                                internal_batch_size=batch_dict['batch_size'], method='riemann_trapezoid')
                                        else:
                                            attributions[j][k] = ig2D.attribute(
                                                PseudoImage2D, baselines=PseudoImage2D * 0, target=target,
                                                additional_forward_args=batch_dict, n_steps=steps,
                                                internal_batch_size=batch_dict['batch_size'])
                                    # sign = "all"
                                    # sign = "positive"
                                grad = np.transpose(attributions[j][k][i].squeeze().cpu().detach().numpy(), (1, 2, 0))
                                # pos_grad = np.sum(np.where(grad < 0, 0, grad), axis=2)
                                # neg_grad = np.sum(-1 * np.where(grad > 0, 0, grad), axis=2)
                                pos_grad = np.sum((grad > 0) * grad, axis=2)
                                neg_grad = np.sum(-1 * (grad < 0) * grad, axis=2)
                                pos_grad_copy = copy.deepcopy(pos_grad)
                                neg_grad_copy = copy.deepcopy(neg_grad)
                                box_type = None
                                if conf_mat[i][j] == 'TP':
                                    box_type = 1
                                    TP_box_cnt += 1
                                elif conf_mat[i][j] == 'FP':
                                    box_type = 0
                                    FP_box_cnt += 1
                                points_flag = box_utils.in_hull(points[:, 0:3], pred_boxes_for_pts_corners[j])
                                num_pts = points_flag.sum()
                                # print("type(num_pts): {}".format(type(num_pts)))
                                new_size = attr_file["pos_attr"].shape[0] + 1
                                attr_file["pos_attr"].resize(new_size, axis=0)
                                attr_file["neg_attr"].resize(new_size, axis=0)
                                attr_file["pred_boxes"].resize(new_size, axis=0)
                                attr_file["pred_boxes_expand"].resize(new_size, axis=0)
                                attr_file["box_type"].resize(new_size, axis=0)
                                attr_file["pred_boxes_loc"].resize(new_size, axis=0)
                                attr_file["box_score"].resize(new_size, axis=0)
                                attr_file["points_in_box"].resize(new_size, axis=0)
                                attr_file["box_score_all"].resize(new_size, axis=0)

                                attr_file["pos_attr"][new_size - 1] = pos_grad
                                attr_file["neg_attr"][new_size - 1] = neg_grad
                                attr_file["pred_boxes"][new_size - 1] = pred_boxes_vertices[j]
                                attr_file["pred_boxes_expand"][new_size - 1] = padded_pred_boxes_vertices[j]
                                attr_file["box_type"][new_size - 1] = box_type
                                attr_file["pred_boxes_loc"][new_size - 1] = pred_boxes_loc[j]
                                attr_file["box_score"][new_size - 1] = pred_scores[j]
                                # print("pred_scores[{}]: {}".format(j, pred_scores[j]))
                                attr_file["points_in_box"][new_size - 1] = num_pts
                                attr_file["box_score_all"][new_size - 1] = anchor_score_all

                                # XQ = get_sum_XQ(grad, pred_boxes_vertices[j], dataset_name, box_w, box_l, sign,
                                #                 high_rez=high_rez, scaling_factor=scaling_factor, margin=box_margin)
                                grad_copy = pos_grad_copy if attr_shown == "positive" else neg_grad_copy
                                # print(
                                #     "pred_boxes_vertices[j] prior to XQ calculation: {}".format(pred_boxes_vertices[j]))
                                XQ = get_sum_XQ_analytics(pos_grad, neg_grad, padded_pred_boxes_vertices[j],
                                                          dataset_name,
                                                          sign, ignore_thresh=ignore_thresh, high_rez=high_rez,
                                                          scaling_factor=scaling_factor, grad_copy=grad_copy)
                                # print(
                                #     "pred_boxes_vertices[j] after XQ calculation: {}".format(pred_boxes_vertices[j]))
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
                                    attr_file["box_type"][new_size - 1] = 1
                                elif conf_mat[i][j] == 'FP':
                                    FP_XQ_list.append(XQ)
                                    FP_score_list.append(pred_scores[j])
                                    FP_dist_list.append(dist_to_ego)
                                    FP_label_list.append(k)
                                    attr_file["box_type"][new_size - 1] = 0
                                box_explained = flip_xy(padded_pred_boxes_vertices[j])

                                if channel_xai:  # generates channel-wise explanation
                                    # TODO: this part needs fixing
                                    for c in range(num_channels_viz):
                                        grad_viz = viz.visualize_image_attr(
                                            grad[:, :, c], bev_image, method="blended_heat_map", sign=sign,
                                            show_colorbar=True,
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
                                        grad_viz[0].savefig(XAI_box_relative_path_str, bbox_inches='tight',
                                                            pad_inches=0.0)
                                        os.chmod(XAI_box_relative_path_str, 0o777)
                                else:
                                    grad_viz = viz.visualize_image_attr(grad, bev_image, method="blended_heat_map",
                                                                        sign=sign,
                                                                        show_colorbar=True,
                                                                        title="Overlaid {}".format(method),
                                                                        alpha_overlay=overlay,
                                                                        fig_size=figure_size, upscale=high_rez)
                                    iou_for_box = 0.0
                                    if gt_exist[i]:
                                        iou_for_box = pred_box_iou[i][j]

                                    XAI_box_relative_path_str = XAI_cls_path_str.split("tools/", 1)[1] + \
                                                                '/box_{}_{}_XQ_{}_points_{}_top_iou_{}_loc{}.png'.format(
                                                                    j, conf_mat[i][j], XQ, num_pts,
                                                                    iou_for_box, pred_boxes_loc[j])

                                    XAI_attr_csv_str = XAI_cls_path_str + \
                                                       '/box_{}_{}_XQ_{}.csv'.format(j, conf_mat[i][j], XQ)
                                    verts = copy.deepcopy(padded_pred_boxes_vertices[j])
                                    if high_rez:
                                        verts = verts / scaling_factor
                                    if attr_to_csv:
                                        if verify_box:
                                            write_attr_to_csv(XAI_attr_csv_str, grad_copy, verts)
                                        else:
                                            if attr_shown == "positive":
                                                write_attr_to_csv(XAI_attr_csv_str, pos_grad, verts)
                                            elif attr_shown == "negative":
                                                write_attr_to_csv(XAI_attr_csv_str, neg_grad, verts)
                                    if plot_enlarged_pred:
                                        polys = patches.Polygon(box_explained,
                                                                closed=True, fill=False, edgecolor='y', linewidth=1)
                                        grad_viz[1].add_patch(polys)
                                    grad_viz[0].savefig(XAI_box_relative_path_str, bbox_inches='tight', pad_inches=0.0)
                                    plt.close('all')
                                    os.chmod(XAI_box_relative_path_str, 0o777)

                    if FN_analysis and gt_exist[i]:
                        # no need to do FN analysis when there're are no gt boxes in range
                        if not missed_boxes_plotted[k] and len(missed_boxes[i]) > 0:
                            missed_boxes_plotted[k] = True
                            # print('len(gt_boxes_vertices): {}'.format(len(gt_boxes_vertices)))
                            # print('missed_boxes[i]: {}'.format(missed_boxes[i]))
                            print("\nnumber of missed boxes: {}\n".format(len(missed_boxes[i])))
                            for FN_id in range(3):
                                FN_type = FN_id + 2
                                FN_boxes, FN_labels, FN_iou = None, None, None
                                box_type_str = None
                                if FN_type == 2:
                                    FN_boxes = missed_boxes[i]
                                    FN_labels = missed_boxes_label[i]
                                    FN_iou = missed_boxes_iou[i]
                                    box_type_str = "missed"
                                    if (not missed_box_counted_i) and len(FN_boxes) > 0:
                                        missed_box_counted_i = True
                                        missed_box_cnt += len(FN_boxes)
                                        f.write("{} missed boxes in batch {}\nlabels: {}\niou: {}\n".format(
                                            len(FN_boxes), batch_num, FN_labels, FN_iou))
                                if FN_type == 3:
                                    FN_boxes = partial_missed_boxes[i]
                                    FN_labels = partial_missed_boxes_label[i]
                                    FN_iou = partial_missed_boxes_iou[i]
                                    box_type_str = "partly_missed"
                                    if (not partly_missed_box_counted_i) and len(FN_boxes) > 0:
                                        partly_missed_box_counted_i = True
                                        partly_missed_box_cnt += len(FN_boxes)
                                        f.write("{} partly missed boxes in batch {}\nlabels: {}\niou: {}\n".format(
                                            len(FN_boxes), batch_num, FN_labels, FN_iou))
                                if FN_type == 4:
                                    # don't need to find a match for misclassified boxes for now
                                    FN_boxes = misclassified_boxes[i]
                                    FN_labels = misclassified_boxes_label[i]
                                    if (not misclassified_box_counted_i) and len(FN_boxes) > 0:
                                        misclassified_box_counted_i = True
                                        misclassified_box_cnt += len(FN_boxes)
                                        f.write("{} misclassified boxes in batch {}\nlabels: {}\n".format(
                                            len(FN_boxes), batch_num, FN_labels))
                                    break
                                    # FN_iou = misclassified_boxes_iou[i]
                                    # box_type_str = "misclassified"
                                for index, label, top_iou in zip(FN_boxes, FN_labels, FN_iou):
                                    # print("\nlabel is: {}\n".format(label))
                                    print("\nFN gt_box {} for {}".format(index, label))
                                    if index >= len(gt_boxes_vertices):
                                        print("len(gt_boxes_vertices): {}".format(len(gt_boxes_vertices)))
                                        break
                                    if label != class_name_list[k]:
                                        continue  # only analyzing the class k right now
                                    print("\nFN gt_box {} for {} after filtering".format(index, label))
                                    # obtain the missed gt box
                                    missed_box = gt_boxes_vertices[index]
                                    box_vertices = transform_box_coord(pseudo_img_h, pseudo_img_w, missed_box,
                                                                       dataset_name, high_rez, scaling_factor)
                                    box_vertices = flip_xy(box_vertices)

                                    # if len(missed_box) != 0:
                                    # print("\nfinding anchor boxes matching the missed gt boxes")
                                    # obtain the anchor box matching that gt_box
                                    # this is valid in (x, y), and in meters, with upper left as (0,0)
                                    missed_box_loc = gt_boxes_loc[index]
                                    print('missed_box_loc being used to find anchor index: {}'.format(missed_box_loc))
                                    f.write('missed_box_loc: {}\n'.format(missed_box_loc))
                                    # y, x = transform_point_coord(grad.shape[0], grad.shape[1], missed_box_loc,
                                    #                                    dataset_name, high_rez, scaling_factor)
                                    # print('transformed box location (y,x): ({}, {})'.format(y, x))
                                    anchor_ind = find_anchor_index(dataset_name, missed_box_loc[1],
                                                                   missed_box_loc[0])
                                    print('the anchor index is: {}'.format(anchor_ind))
                                    lower_bound = max(anchor_ind - FN_search_range, 0)
                                    upper_bound = min(total_anchors, anchor_ind + FN_search_range + 1)
                                    matched_anchor_vertices = None
                                    has_match = False
                                    max_iou = 0
                                    best_FN_anchor = None
                                    best_FN_anchor_label = None
                                    best_FN_anchor_exp = None
                                    bev_tool = None
                                    best_FN_anchor_ind = None
                                    match_dist = 1.0
                                    check_heading = True
                                    if label == "Pedestrian" or label == "Cyclist":
                                        match_dist = 0.5
                                        check_heading = False
                                    if dataset_name == 'KittiDataset':
                                        bev_tool = kitti_bev
                                    elif dataset_name == 'CadcDataset':
                                        bev_tool = cadc_bev
                                    gt_box = gt_dict[i]['boxes'][index]
                                    print("\ngt_box heading: {}".format(gt_box[6]))
                                    dist_filter = False
                                    heading_filter = False
                                    for ind in range(lower_bound, upper_bound, 1):
                                        # print('the anchor index to search: {}'.format(ind))
                                        box_gpu = batch_dict['anchor_boxes'][i][ind]
                                        scores = anchors_scores[i][ind].cpu().detach().numpy()
                                        anchor_label = np.argmax(scores)
                                        # **** Note ***** #
                                        # anchor labels indeed goes from 0 to 2
                                        if anchor_label != k:
                                            # only process anchors that matching the specific class label
                                            continue
                                        box = box_gpu.cpu().detach().numpy()
                                        if abs(box[0] - gt_box[0]) > match_dist or abs(
                                                box[1] - gt_box[1]) > match_dist or \
                                                abs(box[2] - gt_box[2]) > match_dist:
                                            continue  # early stopping
                                        dist_filter = True
                                        if check_heading and abs(abs(box[6]) - abs(gt_box[6])) > 0.5:
                                            continue  # don't check for misaligned boxes
                                        heading_filter = True
                                        gt_box_expand = np.expand_dims(gt_box, axis=0)
                                        box_expand = np.expand_dims(box, axis=0)
                                        curr_iou = None
                                        if IOU_3D:
                                            curr_iou, _ = calculate_iou(gt_box_expand, box_expand, dataset_name)
                                        else:
                                            curr_iou, _ = calculate_bev_iou(gt_box_expand, box_expand)
                                        if curr_iou[0] > max_iou:
                                            has_match = True
                                            max_iou = curr_iou[0]
                                            best_FN_anchor = box
                                            best_FN_anchor_exp = box_expand
                                            best_FN_anchor_ind = ind
                                            best_FN_anchor_label = anchor_label
                                        if curr_iou[0] >= 0.85:
                                            break
                                    if dist_filter:
                                        print("\nFN box {} passed the dist filter!".format(index))
                                    if heading_filter:
                                        print("\nFN box {} passed the heading filter!".format(index))
                                    if has_match:
                                        # TODO: save necessary info to attr file, show XQ and num points in box in image
                                        print("\nFound matching anchor for an FN box {}! \n".format(index))
                                        f.write("batch {} image {}\nFound matching anchor for an FN box {}!\n".format(
                                            batch_num, i, index))
                                        if FN_type == 2:
                                            missed_box_analyzed += 1
                                        if FN_type == 3:
                                            partly_missed_box_analyzed += 1
                                        FN_target = (anchor_ind, k)
                                        FN_attr = ig2D.attribute(
                                            PseudoImage2D, baselines=PseudoImage2D * 0, target=FN_target,
                                            additional_forward_args=batch_dict, n_steps=steps,
                                            internal_batch_size=batch_dict['batch_size'])
                                        FN_all_grad = np.transpose(FN_attr[i].squeeze().cpu().detach().numpy(),
                                                                   (1, 2, 0))
                                        FN_pos_grad = np.sum((FN_all_grad > 0) * FN_all_grad, axis=2)
                                        FN_neg_grad = np.sum(-1 * (FN_all_grad < 0) * FN_all_grad, axis=2)
                                        FN_grad = None
                                        if attr_shown == 'positive':
                                            FN_grad = FN_pos_grad
                                        elif attr_shown == 'negative':
                                            FN_grad = FN_neg_grad
                                        x, y, z, w, l, h, yaw = best_FN_anchor[0], best_FN_anchor[1], \
                                                                best_FN_anchor[2], best_FN_anchor[3], \
                                                                best_FN_anchor[4], best_FN_anchor[5], \
                                                                best_FN_anchor[6]
                                        matched_anchor_vertices = bev_tool.cuboid_to_bev(x, y, z, w, l, h, yaw)
                                        matched_anchor_vertices += bev_tool.offset
                                        orig_matched_vertices = copy.deepcopy(matched_anchor_vertices)
                                        w_big = w + 2 * box_margin
                                        l_big = l + 2 * box_margin
                                        if box_margin != 0:
                                            matched_anchor_vertices = bev_tool.cuboid_to_bev(x, y, z, w_big, l_big, h,
                                                                                             yaw)
                                            matched_anchor_vertices += bev_tool.offset
                                        orig_matched_vertices = transform_box_coord(
                                            pseudo_img_h, pseudo_img_w, orig_matched_vertices, dataset_name, high_rez,
                                            scaling_factor)
                                        XQ = get_sum_XQ_analytics(FN_pos_grad, FN_neg_grad, matched_anchor_vertices,
                                                                  dataset_name,
                                                                  attr_shown, ignore_thresh=ignore_thresh,
                                                                  high_rez=high_rez,
                                                                  scaling_factor=scaling_factor)
                                        padded_anchor_shown = flip_xy(matched_anchor_vertices)
                                        anchor_shown = flip_xy(orig_matched_vertices)
                                        missed_polys = patches.Polygon(
                                            box_vertices, closed=True, fill=False, edgecolor='c', linewidth=1)
                                        matched_anchor_polys = patches.Polygon(
                                            padded_anchor_shown, closed=True, fill=False, edgecolor='y',
                                            linewidth=1)
                                        matched_anchor_polys_orig = patches.Polygon(
                                            anchor_shown, closed=True, fill=False, edgecolor='m',
                                            linewidth=1)
                                        grad_viz = viz.visualize_image_attr(FN_all_grad, bev_image,
                                                                            method="blended_heat_map",
                                                                            sign=attr_shown,
                                                                            show_colorbar=True,
                                                                            title="Overlaid {}".format(method),
                                                                            alpha_overlay=overlay,
                                                                            fig_size=figure_size, upscale=high_rez)
                                        grad_viz[1].add_patch(matched_anchor_polys)
                                        # grad_viz[1].add_patch(matched_anchor_polys_orig)
                                        # grad_viz[1].add_patch(missed_polys)
                                        best_FN_corners = box_utils.boxes_to_corners_3d(best_FN_anchor_exp)
                                        points_flag = box_utils.in_hull(points[:, 0:3], best_FN_corners[0])
                                        num_pts = points_flag.sum()
                                        anchor_loc = np.array([best_FN_anchor[0], best_FN_anchor[1]])
                                        new_size = attr_file["pos_attr"].shape[0] + 1
                                        attr_file["pos_attr"].resize(new_size, axis=0)
                                        attr_file["neg_attr"].resize(new_size, axis=0)
                                        attr_file["pred_boxes"].resize(new_size, axis=0)
                                        attr_file["pred_boxes_expand"].resize(new_size, axis=0)
                                        attr_file["box_type"].resize(new_size, axis=0)
                                        attr_file["pred_boxes_loc"].resize(new_size, axis=0)
                                        attr_file["box_score"].resize(new_size, axis=0)
                                        attr_file["points_in_box"].resize(new_size, axis=0)

                                        attr_file["pos_attr"][new_size - 1] = FN_pos_grad
                                        attr_file["neg_attr"][new_size - 1] = FN_neg_grad
                                        attr_file["pred_boxes"][new_size - 1] = orig_matched_vertices
                                        attr_file["pred_boxes_expand"][new_size - 1] = matched_anchor_vertices
                                        attr_file["box_type"][new_size - 1] = FN_type
                                        attr_file["pred_boxes_loc"][new_size - 1] = anchor_loc
                                        matched_anchor_score = batch_dict['sigmoid_anchor_scores'][i][
                                            best_FN_anchor_ind].cpu().detach().numpy()
                                        # print("anchors_scores[{}][{}]: {}".format(
                                        #     i, best_FN_anchor_ind, matched_anchor_score[k]))
                                        attr_file["box_score"][new_size - 1] = matched_anchor_score[k]
                                        attr_file["points_in_box"][new_size - 1] = num_pts
                                        XAI_missed_boxes_str = XAI_cls_path_str.split("tools/", 1)[1] + \
                                                               '/FN_{}_gt_box_{}_XQ_{}_points_{}_top_iou_{}.png'.format(
                                                                   box_type_str, index, XQ, num_pts, top_iou)
                                        grad_viz[0].savefig(XAI_missed_boxes_str, bbox_inches='tight', pad_inches=0.0)
                                        box_cnt += 1
                                    # missed_ploys = patches.Polygon(box_vertices,
                                    #                                closed=True, fill=False, edgecolor='c', linewidth=1)
                                    # grad_viz[1].add_patch(missed_ploys)
                    attr_file.close()
    finally:
        f.write("total number of boxes analyzed: {}\n".format(box_cnt))
        f.write("TP_box_cnt: {}\n".format(TP_box_cnt))
        f.write("FP_box_cnt: {}\n".format(FP_box_cnt))
        f.write("missed_box_cnt: {}\n".format(missed_box_cnt))
        f.write("partly_missed_box_cnt: {}\n".format(partly_missed_box_cnt))
        f.write("misclassified_box_cnt: {}\n".format(misclassified_box_cnt))
        f.write("number of missed boxes analyzed: {}\n".format(missed_box_analyzed))
        f.write("number of partly missed boxes analyzed: {}\n".format(partly_missed_box_analyzed))
        f.write("number of pred boxes without matching anchors: {}\n".format(unmatched_TP_FP_pred))
        f.close()

        # plotting
        all_xq = XAI_res_path_str + "/all_xq.csv"
        tp_xq = XAI_res_path_str + "/tp_xq.csv"
        fp_xq = XAI_res_path_str + "/fp_xq.csv"
        fnames = ['class_score', 'XQ', 'dist_to_ego', 'class_label']
        write_to_csv(all_xq, fnames, [cls_score_list, XQ_list, dist_list, label_list])
        write_to_csv(tp_xq, fnames, [TP_score_list, TP_XQ_list, TP_dist_list, TP_label_list])
        write_to_csv(fp_xq, fnames, [FP_score_list, FP_XQ_list, FP_dist_list, FP_label_list])

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
