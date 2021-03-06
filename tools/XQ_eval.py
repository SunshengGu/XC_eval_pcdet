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
from XAI_utils.metrics import *
from XAI_analytics import tp_fp_density_plotting
from pcdet.models import load_data_to_gpu
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from pcdet.datasets.cadc.cadc_dataset import CadcDataset
from pcdet.datasets.kitti.kitti_object_eval_python.eval import d3_box_overlap

# XAI related imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import torch


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--XC_path', type=str, default=None, required=True,
                        help='the folder where all XC values are saved')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    args = parser.parse_args()
    return args


def evaluate_metric(score_arr, label_arr, save_path, file_name, cols, score_id, thresh, cls_names, cls_labels):
    """

    :param score_arr: array of the score being evaluated
    :param label_arr: array indicating TP/FP boxes
    :param save_path:
    :param file_name:
    :param cols:
    :param score_id: name of the score being evaluated (e.g., class score, XC, etc.)
    :param thresh: XC threshold
    :param cls_names: list of class names specific to a dataset
    :param cls_labels: class labels for the boxes (e.g., car, pedestrian, cyclist, etc.)
    :return:
    """
    # plot_roc(score_arr, label_arr, save_path=save_path, thresh=thresh, measure=score_id)
    # plot_pr(score_arr, label_arr, save_path=save_path, thresh=thresh, measure=score_id)
    # plot_pr(score_arr, label_arr, save_path=save_path, thresh=thresh, measure=score_id, flip=True)
    if score_id == 'far_attr' or score_id == 'PAP_norm':
        score_arr = -1*score_arr
    eval_dict = get_summary_statistics(score_arr, label_arr, thresh, score_id, "all")
    eval_file = os.path.join(save_path, file_name)
    print("{} objects in total".format(len(score_arr)))
    cls_eval_dicts = []
    class_wise_score = []
    class_wise_label = []
    with open(eval_file, 'w') as evalfile:
        writer = csv.DictWriter(evalfile, fieldnames=cols)
        writer.writeheader()
        writer.writerow(eval_dict)
        for cls in range(3):
            cls_name = cls_names[cls]
            cls_file_name = "{}_{}".format(cls_name, file_name)
            positions = np.where(cls_labels == cls)
            cls_score_arr = score_arr[positions]
            cls_tp_fp_arr = label_arr[positions]
            class_wise_score.append(cls_score_arr)
            class_wise_label.append(cls_tp_fp_arr)
            print("{} {} objects".format(len(cls_score_arr), cls_name))
            # plot_roc(cls_score_arr, cls_tp_fp_arr, save_path=save_path, thresh=thresh, measure=score_id, cls_name=cls_name)
            # plot_pr(cls_score_arr, cls_tp_fp_arr, save_path=save_path, thresh=thresh, measure=score_id, cls_name=cls_name)
            # plot_pr(cls_score_arr, cls_tp_fp_arr, save_path=save_path, thresh=thresh, measure=score_id, cls_name=cls_name,
            #         flip=True)
            cls_eval_dict = get_summary_statistics(cls_score_arr, cls_tp_fp_arr, thresh, score_id, cls_name)
            cls_eval_dicts.append(eval_dict)
            # eval_file = os.path.join(save_path, cls_file_name)
            writer.writerow(cls_eval_dict)
    plot_multi_roc(cls_names, score_arr, label_arr, class_wise_score, class_wise_label,
                   save_path=save_path, thresh=thresh, measure=score_id)
    plot_multi_pr(cls_names, score_arr, label_arr, class_wise_score, class_wise_label,
                   save_path=save_path, thresh=thresh, measure=score_id)
    plot_multi_pr(cls_names, score_arr, label_arr, class_wise_score, class_wise_label,
                  save_path=save_path, thresh=thresh, measure=score_id, flip=True)
    return eval_dict, cls_eval_dicts


def wsum_experiment(all_cls_score_arr, XC_arr, label_arr, save_path, file_name, cols, score_id, thresh, cls_names, cls_labels):
    """

    :param cls_score_arr: array of the class scores
    :param label_arr: array indicating TP/FP boxes
    :param save_path:
    :param file_name:
    :param cols:
    :param score_id: name of the score being evaluated (e.g., class score, XC, etc.)
    :param thresh: XC threshold
    :param cls_names: list of class names specific to a dataset
    :param cls_labels: class labels for the boxes (e.g., car, pedestrian, cyclist, etc.)
    :return:
    """
    eval_file = os.path.join(save_path, file_name)
    min_fpr_95 = 100
    max_aupr = 0
    max_aupr_opposite = 0
    max_auroc = 0
    min_fpr_95_cls = [100, 100, 100]
    max_aupr_cls = [0, 0, 0]
    max_aupr_opposite_cls = [0, 0, 0]
    max_auroc_cls = [0, 0, 0]
    min_fpr_95_w = -1
    max_aupr_w = -1
    max_aupr_opposite_w = -1
    max_auroc_w = -1
    min_fpr_95_cls_w = [-1, -1, -1]
    max_aupr_cls_w = [-1, -1, -1]
    max_aupr_opposite_cls_w = [-1, -1, -1]
    max_auroc_cls_w = [-1, -1, -1]
    with open(eval_file, 'w') as evalfile:
        writer = csv.DictWriter(evalfile, fieldnames=cols)
        writer.writeheader()
        for i in range(0, 101, 1):
            w_XC = i * 0.01
            w_cls_score = 1.00 - w_XC
            score_arr = np.multiply(XC_arr, w_XC) + np.multiply(all_cls_score_arr, w_cls_score)
            eval_dict = get_summary_statistics_wsum(score_arr, label_arr, thresh, score_id, "all", w_XC, w_cls_score)
            # print("{} objects in total".format(len(score_arr)))
            if eval_dict['fpr_at_95_tpr'] < min_fpr_95:
                min_fpr_95 = eval_dict['fpr_at_95_tpr']
                min_fpr_95_w = w_XC
            if eval_dict['auroc'] > max_auroc:
                max_auroc = eval_dict['auroc']
                max_auroc_w = w_XC
            if eval_dict['aupr_in'] > max_aupr:
                max_aupr = eval_dict['aupr_in']
                max_aupr_w = w_XC
            if eval_dict['aupr_out'] > max_aupr_opposite:
                max_aupr_opposite = eval_dict['aupr_out']
                max_aupr_opposite_w = w_XC
            cls_eval_dicts = []
            writer.writerow(eval_dict)
            for cls in range(3):
                cls_name = cls_names[cls]
                positions = np.where(cls_labels == cls)
                cls_score_arr = score_arr[positions]
                cls_tp_fp_arr = label_arr[positions]
                cls_eval_dict = get_summary_statistics_wsum(cls_score_arr, cls_tp_fp_arr, thresh, score_id, cls_name, w_XC, w_cls_score)
                if cls_eval_dict['fpr_at_95_tpr'] < min_fpr_95_cls[cls]:
                    min_fpr_95_cls[cls] = cls_eval_dict['fpr_at_95_tpr']
                    min_fpr_95_cls_w[cls] = w_XC
                if cls_eval_dict['auroc'] > max_auroc_cls[cls]:
                    max_auroc_cls[cls] = cls_eval_dict['auroc']
                    max_auroc_cls_w[cls] = w_XC
                if cls_eval_dict['aupr_in'] > max_aupr_cls[cls]:
                    max_aupr_cls[cls] = cls_eval_dict['aupr_in']
                    max_aupr_cls_w[cls] = w_XC
                if cls_eval_dict['aupr_out'] > max_aupr_opposite_cls[cls]:
                    max_aupr_opposite_cls[cls] = cls_eval_dict['aupr_out']
                    max_aupr_opposite_cls_w[cls] = w_XC
                cls_eval_dicts.append(eval_dict)
                writer.writerow(cls_eval_dict)
    result_dict = {
        'min_fpr_95': min_fpr_95,
        'max_aupr': max_aupr,
        'max_aupr_opposite': max_aupr_opposite,
        'max_auroc': max_auroc,
        'min_fpr_95_cls': min_fpr_95_cls,
        'max_aupr_cls': max_aupr_cls,
        'max_aupr_opposite_cls': max_aupr_opposite_cls,
        'max_auroc_cls': max_auroc_cls,
        'min_fpr_95_w': min_fpr_95_w,
        'max_aupr_w': max_aupr_w,
        'max_aupr_opposite_w': max_aupr_opposite_w,
        'max_auroc_w': max_auroc_w,
        'min_fpr_95_cls_w': min_fpr_95_cls_w,
        'max_aupr_cls_w': max_aupr_cls_w,
        'max_aupr_opposite_cls_w': max_aupr_opposite_cls_w,
        'max_auroc_cls_w': max_auroc_cls_w
    }
    for entry in result_dict:
        print("{}: {}".format(entry, result_dict[entry]))


def main():
    """
    important variables:
    :return:
    """
    use_XQ = True
    XC_only = True
    scatter_plot = False
    legacy_file = True # whether the data file was pre jan10 2021
    comb_plot = True
    w_sum_explore = False
    xc_term = 'XQ'
    if not use_XQ:
        xc_term = 'xc'
    dataset_name = "KITTI"
    cls_name_list = []
    if dataset_name == "KITTI":
        cls_name_list = ['Car', 'Pedestrian', 'Cyclist']
    elif dataset_name == "CADC":
        cls_name_list = ['Car', 'Pedestrian', 'Truck']
    use_margin = True
    XAI_sum = False
    XAI_cnt = not XAI_sum
    ignore_thresh_list = [0.0]
    # ignore_thresh_list = [0.0, 0.0333, 0.0667, 0.1, 0.1333, 0.1667, 0.2]
    start_time = time.time()
    max_obj_cnt = 50
    batches_to_analyze = 1
    method = 'IG'
    attr_shown = 'negative'
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
    args = parse_config()
    dpi_division_factor = 20.0
    steps = 24

    # get the date and time to create a folder for the specific time when this script is run
    now = datetime.datetime.now()
    dt_string = now.strftime("%b_%d_%Y_%H_%M_%S")
    # get current working directory
    cwd = os.getcwd()
    rez_string = 'LowResolution'
    if high_rez:
        rez_string = 'HighResolution'
    # create directory to store results just for this run, include the method in folder name
    XC_path = args.XC_path
    XC_folder = XC_path.split("XAI_results/", 1)[1]
    metric_result_path = os.path.join(cwd, 'XAI_results/{}_metrics_analysis_{}'.format(XC_folder, dt_string))

    print('\nmetric_result_path: {}'.format(metric_result_path))
    metric_res_path_str = str(metric_result_path)
    os.mkdir(metric_result_path)
    os.chmod(metric_res_path_str, 0o777)

    try:
        """
        1. Read in XC from the TP file
        2. Read in XC from the FP file
        3. Concatenate into one array, with TP labeled 1 and FP labeled 0
        """
        # print("start trying")
        XC_thresh_list = ['0.1'] # ['0.0', '0.0333', '0.0667', '0.1', '0.1333', '0.1667', '0.2']
        for thresh in XC_thresh_list:
            XC_list = []
            far_attr_list = []
            PAP_list = []
            PAP_norm_list = []
            score_list = []
            TP_FP_label = []
            cls_label_list = []
            pts_list = []
            dist_list = []
            found = False
            tp_name = "tp_xq_thresh{}.csv".format(thresh)
            fp_name = "fp_xq_thresh{}.csv".format(thresh)
            tp_data = None
            fp_data = None
            for root, dirs, files in os.walk(XC_path):
                # print('processing files: ')
                for name in files:
                    # print(os.path.join(root, name))
                    if name == tp_name:
                        found = True
                        tp_data = pd.read_csv(os.path.join(root, name))
                        # print("type(tp_data[xc_term]): {}".format(type(tp_data[xc_term])))
                        XC_list.append(tp_data[xc_term])
                        if not legacy_file:
                            far_attr_list.append(tp_data['far_attr_cnt'])
                            PAP_list.append(tp_data['PAP'])
                            PAP_norm_list.append(tp_data['PAP_norm'])
                        score_list.append(tp_data['class_score'])
                        TP_FP_label.append(np.ones(len(tp_data[xc_term])))
                        cls_label_list.append(tp_data['class_label'])
                        pts_list.append(tp_data['pts_in_box'])
                        dist_list.append(tp_data['dist_to_ego'])
                        print("Number of TP instances for each class:")
                        print("class 0: {}".format(np.count_nonzero(tp_data['class_label'] == 0)))
                        print("class 1: {}".format(np.count_nonzero(tp_data['class_label'] == 1)))
                        print("class 2: {}".format(np.count_nonzero(tp_data['class_label'] == 2)))
                    elif name == fp_name:
                        found = True
                        fp_data = pd.read_csv(os.path.join(root, name))
                        XC_list.append(fp_data[xc_term])
                        if not legacy_file:
                            far_attr_list.append(fp_data['far_attr_cnt'])
                            PAP_list.append(fp_data['PAP'])
                            PAP_norm_list.append(fp_data['PAP_norm'])
                        score_list.append(fp_data['class_score'])
                        TP_FP_label.append(np.zeros(len(fp_data[xc_term])))
                        cls_label_list.append(fp_data['class_label'])
                        pts_list.append(fp_data['pts_in_box'])
                        dist_list.append(fp_data['dist_to_ego'])
                        print("Number of FP instances for each class:")
                        print("class 0: {}".format(np.count_nonzero(fp_data['class_label'] == 0)))
                        print("class 1: {}".format(np.count_nonzero(fp_data['class_label'] == 1)))
                        print("class 2: {}".format(np.count_nonzero(fp_data['class_label'] == 2)))
            if found:
                XC_arr = np.concatenate(XC_list)
                if not legacy_file:
                    far_attr_arr = np.concatenate(far_attr_list)
                    PAP_arr = np.concatenate(PAP_list)
                    PAP_norm_arr = np.concatenate(PAP_norm_list)
                score_arr = np.concatenate(score_list)
                TP_FP_arr = np.concatenate(TP_FP_label)
                cls_label_arr = np.concatenate(cls_label_list)
                pts_arr = np.concatenate(pts_list)
                dist_arr = np.concatenate(dist_list)
                print("len(XC_arr): {}".format(len(XC_arr)))
                print("len(TP_FP_arr): {}".format(len(TP_FP_arr)))
                if scatter_plot:
                    # class score vs. XC plot
                    fig_name = "{}/XC_class_score_density_thresh{}.png".format(metric_result_path, thresh)
                    x_label = "class scores"
                    tp_fp_density_plotting(XC_arr, score_arr, tp_data[xc_term], tp_data['class_score'], fp_data[xc_term],
                                           fp_data['class_score'], fig_name, x_label)

                    # distance to ego vs. XC plot
                    fig_name = "{}/XC_distance_to_ego_thresh{}.png".format(metric_result_path, thresh)
                    x_label = 'distance to ego'
                    tp_fp_density_plotting(XC_arr, dist_arr, tp_data[xc_term], tp_data['dist_to_ego'], fp_data[xc_term],
                                           fp_data['dist_to_ego'], fig_name, x_label)

                    # num of lidar points in box vs. XC plot
                    fig_name = "{}/XC_points_in_box_thresh{}.png".format(metric_result_path, thresh)
                    x_label = 'points in box'
                    tp_fp_density_plotting(XC_arr, pts_arr, tp_data[xc_term], tp_data['pts_in_box'], fp_data[xc_term],
                                           fp_data['pts_in_box'], fig_name, x_label, x_log=True)

                eval_cols = ['XQ_thresh', 'measure', 'class', 'fpr_at_95_tpr', 'detection_error',
                             'auroc', 'aupr_out', 'aupr_in']
                XC_eval_file = "XQ_eval_metrics_thresh{}.csv".format(thresh)
                pts_eval_file = "pts_eval_metrics_thresh{}.csv".format(thresh)
                dist_eval_file = "dist_eval_metrics_thresh{}.csv".format(thresh)
                if not legacy_file:
                    far_attr_eval_file = "far_attr_eval_metrics_thresh{}.csv".format(thresh)
                    PAP_eval_file = "PAP_eval_metrics_thresh{}.csv".format(thresh)
                    PAP_norm_eval_file = "PAP_norm_eval_metrics_thresh{}.csv".format(thresh)
                XC_dict, XC_cls_dicts = evaluate_metric(XC_arr, TP_FP_arr, metric_result_path, XC_eval_file,
                                                       eval_cols, 'XQ', thresh,
                                                       cls_name_list, cls_label_arr)
                pts_dict, pts_cls_dicts = evaluate_metric(
                    pts_arr, TP_FP_arr, metric_result_path, pts_eval_file, eval_cols, 'pts', thresh, cls_name_list,
                    cls_label_arr)
                dist_dict, dist_cls_dicts = evaluate_metric(
                    dist_arr, TP_FP_arr, metric_result_path, dist_eval_file, eval_cols, 'dist', thresh, cls_name_list,
                    cls_label_arr)
                if not legacy_file:
                    far_attr_dict, far_attr_cls_dicts = evaluate_metric(far_attr_arr, TP_FP_arr, metric_result_path,
                                                            far_attr_eval_file, eval_cols, 'far_attr', thresh,
                                                            cls_name_list, cls_label_arr)
                    PAP_dict, PAP_cls_dicts = evaluate_metric(PAP_arr, TP_FP_arr, metric_result_path,
                                                              far_attr_eval_file, eval_cols, 'PAP', thresh,
                                                              cls_name_list, cls_label_arr)
                    PAP_norm_dict, PAP_norm_cls_dicts = evaluate_metric(PAP_norm_arr, TP_FP_arr, metric_result_path,
                                                              far_attr_eval_file, eval_cols, 'PAP_norm', thresh,
                                                              cls_name_list, cls_label_arr)
                # cls_score_arr, XC_arr, label_arr, save_path, file_name, cols, score_id, thresh, cls_names, cls_labels
                if w_sum_explore:
                    exp_cols = ['w_xq', 'w_cls_score', 'XQ_thresh', 'measure', 'class', 'fpr_at_95_tpr',
                                'detection_error', 'auroc', 'aupr_out', 'aupr_in']
                    w_sum_experiment_file = "cls_score_and_XC_weighted_sum_experiment.csv"
                    wsum_experiment(score_arr, XC_arr, TP_FP_arr, metric_result_path, w_sum_experiment_file,
                                    exp_cols, "weighted_sum", thresh, cls_name_list, cls_label_arr)
                if not XC_only:
                    score_eval_file = "class_score_eval_metrics.csv"
                    score_dict, score_cls_dicts = evaluate_metric(
                        score_arr, TP_FP_arr, metric_result_path, score_eval_file, eval_cols, 'cls_score',
                        thresh, cls_name_list, cls_label_arr)

                    score_XC_euclidean_arr = -1 * np.sqrt(np.multiply(1-XC_arr, 1-XC_arr) + np.multiply(1-score_arr, 1-score_arr))
                    euclidean_eval_file = "cls_score_XC_euclidean_eval_metrics_thresh{}.csv".format(thresh)
                    euc_dict, euc_cls_dicts = evaluate_metric(
                        score_XC_euclidean_arr, TP_FP_arr, metric_result_path, euclidean_eval_file,
                        eval_cols, 'cls_score_XC_euclidean', thresh, cls_name_list, cls_label_arr)

                    XC_w = 0.11
                    score_w = 0.89
                    score_XC_wsum_arr = np.multiply(XC_arr, XC_w) + np.multiply(score_arr, score_w)
                    wsum_eval_file = "cls_score_{}_XC_{}_weighted_sum_eval_metrics_thresh{}.csv".format(
                            score_w, XC_w, thresh)
                    wsum_dict, wsum_cls_dicts = evaluate_metric(
                        score_XC_wsum_arr, TP_FP_arr, metric_result_path, wsum_eval_file,
                        eval_cols, 'cls_score_XC_weighted_sum', thresh, cls_name_list, cls_label_arr)

                    mult_arr = np.multiply(XC_arr, score_arr)
                    mult_file = "cls_score_XC_multiply_eval_metrics_thresh{}.csv".format(thresh)
                    mult_dict, mult_cls_dicts = evaluate_metric(
                        mult_arr, TP_FP_arr, metric_result_path, mult_file,
                        eval_cols, 'cls_score_XC_multiply', thresh, cls_name_list, cls_label_arr)
    finally:
        print("--- {} seconds ---".format(time.time() - start_time))


if __name__ == '__main__':
    main()
