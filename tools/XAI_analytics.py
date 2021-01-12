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
from scipy.stats import gaussian_kde
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
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')

    parser.add_argument('--batch_size', type=int, default=16, required=False, help='batch size for training')
    parser.add_argument('--attr_path', type=str, default=None, required=True,
                        help='the folder where attribution values are saved')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    return args, cfg


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


def tp_fp_density_plotting(y_list, x_list, TP_y_list, TP_x_list, FP_y_list, FP_x_list, fig_name, x_label, x_log=False):
    x_max = max(x_list)
    x_min = min(x_list)
    fig, axs = plt.subplots(3, figsize=(10, 20))
    fig.tight_layout(pad=8.0)
    y_arr = y_list
    x_arr = x_list
    if isinstance(y_list, list):
        y_arr = np.array(y_list)
    if isinstance(x_list, list):
        x_arr = np.array(x_list)
    x_n_y = np.vstack([x_arr, y_arr])
    z_all = gaussian_kde(x_n_y)(x_n_y)
    idx = z_all.argsort()
    x, y, z = x_arr[idx], y_arr[idx], z_all[idx]
    axs[0].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
    axs[0].set_title('All Boxes', fontsize=20)

    TP_x_arr = TP_x_list
    TP_y_arr = TP_y_list
    if isinstance(TP_y_list, list):
        TP_y_arr = np.array(TP_y_list)
    if isinstance(TP_x_list, list):
        TP_x_arr = np.array(TP_x_list)
    TP_x_n_y = np.vstack([TP_x_arr, TP_y_arr])
    z_all = gaussian_kde(TP_x_n_y)(TP_x_n_y)
    idx = z_all.argsort()
    x, y, z = TP_x_arr[idx], TP_y_arr[idx], z_all[idx]
    axs[1].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
    axs[1].set_title('TP Boxes', fontsize=20)

    FP_x_arr = FP_x_list
    FP_y_arr = FP_y_list
    if isinstance(FP_y_list, list):
        FP_x_arr = np.array(FP_x_list)
    if isinstance(FP_x_list, list):
        FP_y_arr = np.array(FP_y_list)
    FP_x_n_y = np.vstack([FP_x_arr, FP_y_arr])
    z_all = gaussian_kde(FP_x_n_y)(FP_x_n_y)
    idx = z_all.argsort()
    x, y, z = FP_x_arr[idx], FP_y_arr[idx], z_all[idx]
    axs[2].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
    axs[2].set_title('FP Boxes', fontsize=20)
    for ax in axs:
        ax.axis(ymin=0.0, ymax=1.0)
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel('XQ', fontsize=20)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        # ax.set(xlabel=x_label, ylabel='XQ', fontsize=20)
        if x_log:
            ax.set_xscale('log')
            ax.axis(xmin=1, xmax=x_max)
        else:
            ax.axis(xmin=x_min, xmax=x_max)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    plt.savefig(fig_name)
    plt.close()


def main():
    """
    important variables:
        ignore_thresh_list: a list of ignore thresholds to try for XQ calculations
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
    FN_analysis = False
    use_margin = True
    XAI_sum = False
    XAI_cnt = not XAI_sum
    ignore_thresh_list = [0.0]
    # ignore_thresh_list = [0.0333, 0.0667, 0.1, 0.1333, 0.1667, 0.2]
    box_cnt = 0
    start_time = time.time()
    method = 'IG'
    attr_shown = 'positive'
    method_str = ""
    if XAI_sum:
        method_str = "summing"
    if XAI_cnt:
        method_str = "counting"
    high_rez = True
    overlay_orig_bev = True
    mult_by_inputs = True
    channel_xai = False
    gray_scale_overlay = True
    plot_class_wise = True
    color_map = 'jet'
    box_margin = 0.2
    if gray_scale_overlay:
        color_map = 'gray'
    scaling_factor = 5
    args, cfg = parse_config()
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
    steps = 24

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = output_dir / 'eval'

    eval_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)

    # ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else output_dir / 'ckpt'

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
    XAI_attr_path = args.attr_path
    attr_folder = XAI_attr_path.split("XAI_attributions/", 1)[1]
    XAI_result_path = os.path.join(cwd, 'XAI_results/{}_analytics_{}_{}_attr_by_{}'.format(
        attr_folder, dt_string, attr_shown, method_str))

    print('\nXAI_result_path: {}'.format(XAI_result_path))
    XAI_res_path_str = str(XAI_result_path)
    os.mkdir(XAI_result_path)
    os.chmod(XAI_res_path_str, 0o777)

    info_file = XAI_res_path_str + "/XAI_information.txt"
    f = open(info_file, "w")
    f.write('High Resolution: {}\n'.format(high_rez))
    f.write('DPI Division Factor: {}\n'.format(dpi_division_factor))
    f.write('Attributions Visualized: {}\n'.format(attr_shown))
    f.write('Bounding box margin: {}\n'.format(box_margin))

    if overlay_orig_bev:
        f.write('Background Image: BEV of the original point cloud\n')
    else:
        f.write('Background Image: BEV of the PointPillar Pseudoimage\n')
    if method == "IG":
        f.write("IG # of Steps: {}\n".format(steps))
        f.write("Multiply by Input: {}\n".format(mult_by_inputs))
    f.write('Channel-wise Explanation: {}\n'.format(channel_xai))
    f.write('Recording XQ values: \n')
    os.chmod(info_file, 0o777)

    if XAI_sum:
        f.write('Analyze XQ by: sum\n')
    else:
        f.write('Analyze XQ by: count\n')

    label_dict = None  # maps class name to label
    vicinity_dict = None  # Search vicinity for the generate_box_mask function
    if dataset_name == 'KittiDataset':
        label_dict = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
        vicinity_dict = {"Car": 20, "Pedestrian": 5, "Cyclist": 9}
    elif dataset_name == 'CadcDataset':
        label_dict = {"Car": 0, "Pedestrian": 1, "Truck": 2}
        vicinity_dict = {"Car": 13, "Pedestrian": 3, "Truck": 19}

    try:
        for ignore_thresh in ignore_thresh_list:
            cls_score_list = []
            cls_0_scores = []
            cls_1_scores = []
            cls_2_scores = []
            dist_list = []  # stores distance to ego vehicle
            label_list = []
            pts_count_list = []
            TP_cls_0_scores = []
            TP_cls_1_scores = []
            TP_cls_2_scores = []
            TP_score_list = []
            TP_XQ_list = []
            TP_dist_list = []
            TP_label_list = []
            TP_pts_count_list = []
            FP_cls_0_scores = []
            FP_cls_1_scores = []
            FP_cls_2_scores = []
            FP_score_list = []
            FP_XQ_list = []
            FP_dist_list = []
            FP_label_list = []
            FP_pts_count_list = []
            FN_cls_0_scores = []
            FN_cls_1_scores = []
            FN_cls_2_scores = []
            FN_score_list = []
            FN_XQ_list = []
            FN_dist_list = []
            FN_label_list = []
            FN_pts_count_list = []
            XQ_list = []
            for root, dirs, files in os.walk(XAI_attr_path):
                # print('processing files: ')
                for name in files:
                    # print(os.path.join(root, name))
                    if name.endswith(".hdf5"):
                        # read in the file
                        label = 255
                        sign = None
                        vicinity = 0  # Search vicinity for the generate_box_mask function

                        if "Car" in name:
                            label = label_dict["Car"]
                            vicinity = vicinity_dict["Car"]
                        elif "Pedestrian" in name:
                            label = label_dict["Pedestrian"]
                            vicinity = vicinity_dict["Pedestrian"]
                        elif "Cyclist" in name:
                            label = label_dict["Cyclist"]
                            vicinity = vicinity_dict["Cyclist"]
                        elif "Truck" in name:
                            label = label_dict["Truck"]
                            vicinity = vicinity_dict["Truck"]

                        sign = attr_shown
                        with h5py.File(os.path.join(root, name), 'r') as attr_data_file:
                            prediction_boxes = attr_data_file["pred_boxes"]
                            expanded_pred_boxes = attr_data_file["pred_boxes_expand"]
                            pos_attr = attr_data_file["pos_attr"]
                            neg_attr = attr_data_file["neg_attr"]
                            boxes_type = attr_data_file["box_type"]
                            pred_boxes_loc = attr_data_file["pred_boxes_loc"]
                            pred_scores = attr_data_file["box_score"]
                            num_pts_in_pred_box = attr_data_file["points_in_box"]
                            pred_scores_all = attr_data_file["box_score_all"]

                            # print("\nnum_pts_in_pred_box.shape: {}\n".format(num_pts_in_pred_box.shape))
                            pred_boxes = prediction_boxes
                            if use_margin:
                                pred_boxes = expanded_pred_boxes
                            for j in range(len(pred_boxes)):
                                # print("box id is: {}".format(j))
                                # print("pred_boxes_loc[{}]: {}".format(j, pred_boxes_loc[j]))
                                # print("prediction_boxes[{}]: {}".format(j, prediction_boxes[j]))
                                box_x = pred_boxes_loc[j][0]
                                box_y = pred_boxes_loc[j][1]
                                dist_to_ego = np.sqrt(box_x * box_x + box_y * box_y)
                                XQ = 0
                                if XAI_sum:
                                    XQ = get_sum_XQ_analytics_fast(
                                        pos_attr[j], neg_attr[j], pred_boxes[j], dataset_name, sign,
                                        ignore_thresh, pred_boxes_loc[j], vicinity)
                                    # XQ = get_sum_XQ_analytics(pos_attr[j], neg_attr[j], pred_boxes[j], dataset_name,
                                    #                           sign, ignore_thresh, high_rez=high_rez,
                                    #                           scaling_factor=scaling_factor)
                                if XAI_cnt:
                                    XQ = get_cnt_XQ_analytics_fast(
                                        pos_attr[j], neg_attr[j], pred_boxes[j], dataset_name, sign,
                                        ignore_thresh, pred_boxes_loc[j], vicinity)
                                    # XQ = get_cnt_XQ_analytics(pos_attr[j], neg_attr[j], pred_boxes[j], dataset_name,
                                    #                           sign, ignore_thresh, high_rez=high_rez,
                                    #                           scaling_factor=scaling_factor)
                                XQ_list.append(XQ)
                                pts_count_list.append(num_pts_in_pred_box[j][0])
                                cls_score_list.append(pred_scores[j][0])
                                cls_0_scores.append(pred_scores_all[j][0])
                                cls_1_scores.append(pred_scores_all[j][1])
                                cls_2_scores.append(pred_scores_all[j][2])
                                dist_list.append(dist_to_ego)
                                label_list.append(label)
                                box_cnt += 1
                                if boxes_type[j] == 1:  # 1 is TP
                                    TP_XQ_list.append(XQ)
                                    TP_pts_count_list.append(num_pts_in_pred_box[j][0])
                                    TP_score_list.append(pred_scores[j][0])
                                    TP_dist_list.append(dist_to_ego)
                                    TP_label_list.append(label)
                                    TP_cls_0_scores.append(pred_scores_all[j][0])
                                    TP_cls_1_scores.append(pred_scores_all[j][1])
                                    TP_cls_2_scores.append(pred_scores_all[j][2])
                                elif boxes_type[j] == 0:  # 0 is FP
                                    FP_XQ_list.append(XQ)
                                    FP_pts_count_list.append(num_pts_in_pred_box[j][0])
                                    FP_score_list.append(pred_scores[j][0])
                                    FP_dist_list.append(dist_to_ego)
                                    FP_label_list.append(label)
                                    FP_cls_0_scores.append(pred_scores_all[j][0])
                                    FP_cls_1_scores.append(pred_scores_all[j][1])
                                    FP_cls_2_scores.append(pred_scores_all[j][2])
                                elif FN_analysis and boxes_type[j] <= 2:  # 2,3,4 are FN
                                    FN_XQ_list.append(XQ)
                                    FN_pts_count_list.append(num_pts_in_pred_box[j][0])
                                    FN_score_list.append(pred_scores[j][0])
                                    FN_dist_list.append(dist_to_ego)
                                    FN_label_list.append(label)
                                    FN_cls_0_scores.append(pred_scores_all[j][0])
                                    FN_cls_1_scores.append(pred_scores_all[j][1])
                                    FN_cls_2_scores.append(pred_scores_all[j][2])
                # print('processing dirs: ')
                # for name in dirs:
                #     print(os.path.join(root, name))
            f.write("total number of boxes analyzed: {}\n".format(box_cnt))

            print("final processing!")

            # plotting
            all_xq = XAI_res_path_str + "/all_xq_thresh{}.csv".format(ignore_thresh)
            tp_xq = XAI_res_path_str + "/tp_xq_thresh{}.csv".format(ignore_thresh)
            fp_xq = XAI_res_path_str + "/fp_xq_thresh{}.csv".format(ignore_thresh)
            fn_xq = XAI_res_path_str + "/fn_xq_thresh{}.csv".format(ignore_thresh)
            fnames = ['class_score', 'XQ', 'dist_to_ego', 'class_label', 'class_0_score', 'class_1_score',
                      'class_2_score', 'pts_in_box']
            write_to_csv(
                all_xq, fnames, [cls_score_list, XQ_list, dist_list, label_list, cls_0_scores, cls_1_scores,
                                 cls_2_scores, pts_count_list])
            write_to_csv(
                tp_xq, fnames, [TP_score_list, TP_XQ_list, TP_dist_list, TP_label_list, TP_cls_0_scores,
                                TP_cls_1_scores, TP_cls_2_scores, TP_pts_count_list])
            write_to_csv(
                fp_xq, fnames, [FP_score_list, FP_XQ_list, FP_dist_list, FP_label_list, FP_cls_0_scores,
                                FP_cls_1_scores, FP_cls_2_scores, FP_pts_count_list])

            if FN_analysis:
                write_to_csv(fn_xq, fnames, [FN_score_list, FN_XQ_list, FN_dist_list, FN_label_list])

                ### generate 4 plots for each analysis, all, FP, TP, FN ###

                # class score vs. XQ plot
                fig, axs = plt.subplots((2, 2), figsize=(20, 20))
                cls_score_arr = np.array(cls_score_list)
                XQ_arr = np.array(XQ_list)
                cls_n_XQ = np.vstack([cls_score_arr, XQ_arr])
                z_all = gaussian_kde(cls_n_XQ)(cls_n_XQ)
                idx = z_all.argsort()
                x, y, z = cls_score_arr[idx], XQ_arr[idx], z_all[idx]
                axs[0, 0].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[0, 0].set_title('All Boxes')

                TP_score_arr = np.array(TP_score_list)
                TP_XQ_arr = np.array(TP_XQ_list)
                TP_cls_n_XQ = np.vstack([TP_score_arr, TP_XQ_arr])
                z_all = gaussian_kde(TP_cls_n_XQ)(TP_cls_n_XQ)
                idx = z_all.argsort()
                x, y, z = TP_score_arr[idx], TP_XQ_arr[idx], z_all[idx]
                axs[0, 1].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[0, 1].set_title('TP Boxes')

                FP_score_arr = np.array(FP_score_list)
                FP_XQ_arr = np.array(FP_XQ_list)
                FP_cls_n_XQ = np.vstack([FP_score_arr, FP_XQ_arr])
                z_all = gaussian_kde(FP_cls_n_XQ)(FP_cls_n_XQ)
                idx = z_all.argsort()
                x, y, z = FP_score_arr[idx], FP_XQ_arr[idx], z_all[idx]
                axs[1, 0].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[1, 0].set_title('FP Boxes')

                FN_score_arr = np.array(FN_score_list)
                FN_XQ_arr = np.array(FN_XQ_list)
                FN_cls_n_XQ = np.vstack([FN_score_arr, FN_XQ_arr])
                z_all = gaussian_kde(FN_cls_n_XQ)(FN_cls_n_XQ)
                idx = z_all.argsort()
                x, y, z = FN_score_arr[idx], FN_XQ_arr[idx], z_all[idx]
                axs[1, 1].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[1, 1].set_title('FN Boxes')
                for ax in axs:
                    ax.set(xlabel='class scores', ylabel='XQ')
                plt.savefig("{}/XQ_class_score_density_thresh{}.png".format(XAI_result_path, ignore_thresh))
                plt.close()

                # XQ distribution
                fig, axs = plt.subplots((2, 2), figsize=(20, 20))
                axs[0, 0].hist(XQ_list, bins=20, range=(0.0, 1.0))
                axs[0, 0].set_title('All Boxes')
                axs[0, 1].hist(TP_XQ_list, bins=20, range=(0.0, 1.0))
                axs[0, 1].set_title('TP Boxes')
                axs[1, 0].hist(FP_XQ_list, bins=20, range=(0.0, 1.0))
                axs[1, 0].set_title('FP Boxes')
                axs[1, 1].hist(FN_XQ_list, bins=20, range=(0.0, 1.0))
                axs[1, 1].set_title('FN Boxes')
                for ax in axs:
                    ax.set(xlabel='XQ', ylabel='box_count')
                plt.savefig("{}/pred_box_XQ_histograms_thresh{}.png".format(XAI_result_path, ignore_thresh))
                plt.close()

                # distance to ego vs. XQ plot
                fig, axs = plt.subplots((2, 2), figsize=(20, 20))
                dist_arr = np.array(dist_list)
                dist_n_XQ = np.vstack([dist_arr, XQ_arr])
                z_all = gaussian_kde(dist_n_XQ)(dist_n_XQ)
                idx = z_all.argsort()
                x, y, z = dist_arr[idx], XQ_arr[idx], z_all[idx]
                axs[0, 0].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[0, 0].set_title('All Boxes')

                TP_dist_arr = np.array(TP_dist_list)
                TP_dist_n_XQ = np.vstack([TP_dist_arr, TP_XQ_arr])
                z_all = gaussian_kde(TP_dist_n_XQ)(TP_dist_n_XQ)
                idx = z_all.argsort()
                x, y, z = TP_dist_arr[idx], TP_XQ_arr[idx], z_all[idx]
                axs[0, 1].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[0, 1].set_title('TP Boxes')

                FP_dist_arr = np.array(FP_dist_list)
                FP_dist_n_XQ = np.vstack([FP_dist_arr, FP_XQ_arr])
                z_all = gaussian_kde(FP_dist_n_XQ)(FP_dist_n_XQ)
                idx = z_all.argsort()
                x, y, z = FP_dist_arr[idx], FP_XQ_arr[idx], z_all[idx]
                axs[1, 0].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[1, 0].set_title('FP Boxes')

                FN_dist_arr = np.array(FN_dist_list)
                FN_dist_n_XQ = np.vstack([FN_dist_arr, FN_XQ_arr])
                z_all = gaussian_kde(FN_dist_n_XQ)(FN_dist_n_XQ)
                idx = z_all.argsort()
                x, y, z = FN_dist_arr[idx], FN_XQ_arr[idx], z_all[idx]
                axs[1, 1].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[1, 1].set_title('FN Boxes')
                for ax in axs:
                    ax.set(xlabel='distance to ego', ylabel='XQ')
                plt.savefig("{}/XQ_distance_to_ego_thresh{}.png".format(XAI_result_path, ignore_thresh))
                plt.close()

                # num of lidar points in box vs. XQ plot
                fig, axs = plt.subplots((2, 2), figsize=(20, 20))
                pts_arr = np.array(pts_count_list)
                pts_n_XQ = np.vstack([pts_arr, XQ_arr])
                z_all = gaussian_kde(pts_n_XQ)(pts_n_XQ)
                idx = z_all.argsort()
                x, y, z = pts_arr[idx], XQ_arr[idx], z_all[idx]
                axs[0, 0].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[0, 0].set_title('All Boxes')

                TP_pts_arr = np.array(TP_pts_count_list)
                TP_dist_n_XQ = np.vstack([TP_pts_arr, TP_XQ_arr])
                z_all = gaussian_kde(TP_dist_n_XQ)(TP_dist_n_XQ)
                idx = z_all.argsort()
                x, y, z = TP_dist_arr[idx], TP_XQ_arr[idx], z_all[idx]
                axs[0, 1].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[0, 1].set_title('TP Boxes')

                FP_pts_arr = np.array(FP_pts_count_list)
                FP_pts_n_XQ = np.vstack([FP_pts_arr, FP_XQ_arr])
                z_all = gaussian_kde(FP_pts_n_XQ)(FP_pts_n_XQ)
                idx = z_all.argsort()
                x, y, z = FP_dist_arr[idx], FP_XQ_arr[idx], z_all[idx]
                axs[1, 0].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[1, 0].set_title('FP Boxes')

                FN_pts_arr = np.array(FN_pts_count_list)
                FN_pts_n_XQ = np.vstack([FN_pts_arr, FN_XQ_arr])
                z_all = gaussian_kde(FN_pts_n_XQ)(FN_pts_n_XQ)
                idx = z_all.argsort()
                x, y, z = FN_dist_arr[idx], FN_XQ_arr[idx], z_all[idx]
                axs[1, 1].scatter(x, y, c=z, s=10, cmap='jet', label=None, picker=True, zorder=2, marker='.')
                axs[1, 1].set_title('FN Boxes')
                for ax in axs:
                    ax.set(xlabel='distance to ego', ylabel='XQ')
                plt.savefig("{}/XQ_points_in_box_thresh{}.png".format(XAI_result_path, ignore_thresh))
                plt.close()
            else:
                ### generate 3 plots for each analysis, all, FP, TP ###

                # class score vs. XQ plot
                fig_name = "{}/XQ_class_score_density_thresh{}.png".format(XAI_result_path, ignore_thresh)
                x_label = "class scores"
                tp_fp_density_plotting(XQ_list, cls_score_list, TP_XQ_list, TP_score_list, FP_XQ_list,
                                       FP_score_list, fig_name, x_label)

                # XQ distribution
                fig, axs = plt.subplots(3, figsize=(10, 20))
                fig.tight_layout(pad=8.0)
                axs[0].hist(XQ_list, bins=20, range=(0.0, 1.0))
                axs[0].set_title('All Boxes', fontsize=20)
                axs[1].hist(TP_XQ_list, bins=20, range=(0.0, 1.0))
                axs[1].set_title('TP Boxes', fontsize=20)
                axs[2].hist(FP_XQ_list, bins=20, range=(0.0, 1.0))
                axs[2].set_title('FP Boxes', fontsize=20)
                for ax in axs:
                    # ax.set(xlabel='XQ', ylabel='box_count')
                    ax.set_xlabel('XQ', fontsize=20)
                    ax.set_ylabel('box_count', fontsize=20)
                    ax.tick_params(axis='x', labelsize=16)
                    ax.tick_params(axis='y', labelsize=16)
                plt.savefig("{}/pred_box_XQ_histograms_thresh{}.png".format(XAI_result_path, ignore_thresh))
                plt.close()

                # distance to ego vs. XQ plot
                fig_name = "{}/XQ_distance_to_ego_thresh{}.png".format(XAI_result_path, ignore_thresh)
                x_label = 'distance to ego'
                tp_fp_density_plotting(XQ_list, dist_list, TP_XQ_list, TP_dist_list, FP_XQ_list,
                                       FP_dist_list, fig_name, x_label)

                # num of lidar points in box vs. XQ plot
                fig_name = "{}/XQ_points_in_box_thresh{}.png".format(XAI_result_path, ignore_thresh)
                x_label = 'points in box'
                tp_fp_density_plotting(XQ_list, pts_count_list, TP_XQ_list, TP_pts_count_list, FP_XQ_list,
                                       FP_pts_count_list, fig_name, x_label, x_log=True)

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
                    pts_count_list_i = list_selection(pts_count_list, selected)
                    TP_score_list_i = list_selection(TP_score_list, selected_TP)
                    TP_XQ_list_i = list_selection(TP_XQ_list, selected_TP)
                    TP_dist_list_i = list_selection(TP_dist_list, selected_TP)
                    TP_pts_count_list_i = list_selection(TP_pts_count_list, selected_TP)
                    FP_score_list_i = list_selection(FP_score_list, selected_FP)
                    FP_XQ_list_i = list_selection(FP_XQ_list, selected_FP)
                    FP_dist_list_i = list_selection(FP_dist_list, selected_FP)
                    FP_pts_count_list_i = list_selection(FP_pts_count_list, selected_FP)

                    if FN_analysis:
                        selected_FN = [j for j in range(len(FN_label_list)) if FN_label_list[j] == i]
                        FN_score_list_i = list_selection(FN_score_list, selected_FN)
                        FN_XQ_list_i = list_selection(FN_XQ_list, selected_FN)
                        FN_dist_list_i = list_selection(FN_dist_list, selected_FN)
                        FN_pts_count_list_i = list_selection(FN_pts_count_list, selected_FN)

                        fig, axs = plt.subplots((2, 2), figsize=(10, 20))
                        axs[0, 0].scatter(cls_score_list_i, XQ_list_i)
                        axs[0, 0].set_title('All Boxes')
                        axs[0, 1].scatter(TP_score_list_i, TP_XQ_list_i)
                        axs[0, 1].set_title('TP Boxes')
                        axs[1, 0].scatter(FP_score_list_i, FP_XQ_list_i)
                        axs[1, 0].set_title('FP Boxes')
                        axs[1, 1].scatter(FN_score_list_i, FN_XQ_list_i)
                        axs[1, 1].set_title('FN Boxes')
                        for ax in axs:
                            ax.set(xlabel='class scores', ylabel='XQ')
                        plt.savefig("{}/XQ_class_score_{}_thresh{}.png".format(XAI_result_path, class_name, ignore_thresh))
                        plt.close()

                        fig, axs = plt.subplots((2, 2), figsize=(10, 20))
                        axs[0, 0].hist(XQ_list_i, bins=20, range=(0.0, 1.0))
                        axs[0, 0].set_title('All Boxes')
                        axs[0, 1].hist(TP_XQ_list_i, bins=20, range=(0.0, 1.0))
                        axs[0, 1].set_title('TP Boxes')
                        axs[1, 0].hist(FP_XQ_list_i, bins=20, range=(0.0, 1.0))
                        axs[1, 0].set_title('FP Boxes')
                        axs[1, 1].hist(FN_XQ_list_i, bins=20, range=(0.0, 1.0))
                        axs[1, 1].set_title('FN Boxes')
                        for ax in axs:
                            ax.set(xlabel='XQ', ylabel='box_count')
                        plt.savefig("{}/pred_box_XQ_histograms_{}_thresh{}.png".format(
                            XAI_result_path, class_name, ignore_thresh))
                        plt.close()

                        fig, axs = plt.subplots((2, 2), figsize=(10, 20))
                        axs[0, 0].scatter(dist_list_i, XQ_list_i)
                        axs[0, 0].set_title('All Boxes')
                        axs[0, 1].scatter(TP_dist_list_i, TP_XQ_list_i)
                        axs[0, 1].set_title('TP Boxes')
                        axs[1, 0].scatter(FP_dist_list_i, FP_XQ_list_i)
                        axs[1, 0].set_title('FP Boxes')
                        axs[1, 1].scatter(FN_dist_list_i, FN_XQ_list_i)
                        axs[1, 1].set_title('FN Boxes')
                        for ax in axs:
                            ax.set(xlabel='distance to ego', ylabel='XQ')
                        plt.savefig("{}/XQ_distance_to_ego_{}_thresh{}.png".format(
                            XAI_result_path, class_name, ignore_thresh))
                        plt.close()

                        fig, axs = plt.subplots((2, 2), figsize=(10, 20))
                        axs[0, 0].scatter(pts_count_list_i, XQ_list_i)
                        axs[0, 0].set_title('All Boxes')
                        axs[0, 1].scatter(TP_pts_count_list_i, TP_XQ_list_i)
                        axs[0, 1].set_title('TP Boxes')
                        axs[1, 0].scatter(FP_pts_count_list_i, FP_XQ_list_i)
                        axs[1, 0].set_title('FP Boxes')
                        axs[1, 1].scatter(FN_pts_count_list_i, FN_XQ_list_i)
                        axs[1, 1].set_title('FN Boxes')
                        for ax in axs:
                            ax.set(xlabel='points in box', ylabel='XQ')
                        plt.savefig("{}/XQ_points_in_box_{}_thresh{}.png".format(
                            XAI_result_path, class_name, ignore_thresh))
                        plt.close()
                    else:
                        fig_name = "{}/XQ_class_score_{}_thresh{}.png".format(XAI_result_path, class_name, ignore_thresh)
                        x_label = "class scores"
                        tp_fp_density_plotting(XQ_list_i, cls_score_list_i, TP_XQ_list_i, TP_score_list_i, FP_XQ_list_i,
                                               FP_score_list_i, fig_name, x_label)

                        fig, axs = plt.subplots(3, figsize=(10, 20))
                        fig.tight_layout(pad=8.0)
                        axs[0].hist(XQ_list_i, bins=20, range=(0.0, 1.0))
                        axs[0].set_title('All Boxes', fontsize=20)
                        axs[1].hist(TP_XQ_list_i, bins=20, range=(0.0, 1.0))
                        axs[1].set_title('TP Boxes', fontsize=20)
                        axs[2].hist(FP_XQ_list_i, bins=20, range=(0.0, 1.0))
                        axs[2].set_title('FP Boxes', fontsize=20)
                        for ax in axs:
                            # ax.set(xlabel='XQ', ylabel='box_count')
                            ax.set_xlabel('XQ', fontsize=20)
                            ax.set_ylabel('box_count', fontsize=20)
                            ax.tick_params(axis='x', labelsize=16)
                            ax.tick_params(axis='y', labelsize=16)
                        plt.savefig("{}/pred_box_XQ_histograms_{}_thresh{}.png".format(
                            XAI_result_path, class_name, ignore_thresh))
                        plt.close()

                        fig_name = "{}/XQ_distance_to_ego_{}_thresh{}.png".format(XAI_result_path, class_name, ignore_thresh)
                        x_label = 'distance to ego'
                        tp_fp_density_plotting(XQ_list_i, dist_list_i, TP_XQ_list_i, TP_dist_list_i, FP_XQ_list_i,
                                               FP_dist_list_i, fig_name, x_label)

                        fig_name = "{}/XQ_points_in_box_{}_thresh{}.png".format(XAI_result_path, class_name, ignore_thresh)
                        x_label = 'points in box'
                        tp_fp_density_plotting(XQ_list_i, pts_count_list_i, TP_XQ_list_i, TP_pts_count_list_i, FP_XQ_list_i,
                                               FP_pts_count_list_i, fig_name, x_label, x_log=True)
            print("finished analysis for threshold = {}".format(ignore_thresh))
    finally:
        f.close()
        print("{} boxes analyzed".format(box_cnt))
        print("--- {} seconds ---".format(time.time() - start_time))


if __name__ == '__main__':
    main()
