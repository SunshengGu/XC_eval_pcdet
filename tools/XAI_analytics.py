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
    XAI_result_path = os.path.join(cwd, 'XAI_results/{}_analytics_{}'.format(attr_folder, dt_string))

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

    label_dict = None # maps class name to label
    if dataset_name == 'KittiDataset':
        label_dict = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    elif dataset_name == 'CadcDataset':
        label_dict = {"Car": 0, "Pedestrian": 1, "Truck": 2}

    try:
        for root, dirs, files in os.walk(XAI_attr_path):
            print('processing files: ')
            for name in files:
                print(os.path.join(root, name))
                if name.endwith(".h5df"):
                    # read in the file
                    label = 255
                    sign = None

                    if "Car" in name:
                        label = label_dict["Car"]
                    elif "Pedestrian" in name:
                        label = label_dict["Pedestrian"]
                    elif "Cyclist" in name:
                        label = label_dict["Cyclist"]
                    elif "Truck" in name:
                        label = label_dict["Truck"]

                    if "negative" in name:
                        sign = "negative"
                    elif "positive" in name:
                        sign = "positive"
                    attr_data_file = h5py.File(os.path.join(root, name), 'r')
                    pred_boxes = attr_data_file["pred_boxes"]
                    pos_attr = attr_data_file["pos_attr"]
                    neg_attr = attr_data_file["neg_attr"]
                    boxes_type = attr_data_file["box_type"]
                    pred_boxes_loc = attr_data_file["pred_boxes_loc"]
                    # need to process this data file
                    for j in range(len(pred_boxes)):
                        box_w = pred_boxes[j][3]
                        box_l = pred_boxes[j][4]
                        box_x = pred_boxes_loc[j][0]
                        box_y = pred_boxes_loc[j][1]
                        dist_to_ego = np.sqrt(box_x * box_x + box_y * box_y)
                        XQ = get_sum_XQ_analytics(pos_attr[j], neg_attr[j], pred_boxes[j], dataset_name,
                                                  box_w, box_l, sign, high_rez=high_rez,
                                                  scaling_factor=scaling_factor, margin=box_margin)
                        XQ_list.append(XQ)
                        cls_score_list.append(pred_scores[j])
                        dist_list.append(dist_to_ego)
                        label_list.append(label)
                        box_cnt += 1
                        if boxes_type[j] == 'TP':
                            TP_XQ_list.append(XQ)
                            TP_score_list.append(pred_scores[j])
                            TP_dist_list.append(dist_to_ego)
                            TP_label_list.append(label)
                        elif boxes_type[j] == 'FP':
                            FP_XQ_list.append(XQ)
                            FP_score_list.append(pred_scores[j])
                            FP_dist_list.append(dist_to_ego)
                            FP_label_list.append(label)
            print('processing dirs: ')
            for name in dirs:
                print(os.path.join(root, name))

    finally:
        f.write("total number of boxes analyzed: {}".format(box_cnt))
        f.close()

        print("final processing!")

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
