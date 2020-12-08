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
    parser.add_argument('--XQ_path', type=str, default=None, required=True,
                        help='the folder where all XQ values are saved')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    args = parser.parse_args()
    return args


def main():
    """
    important variables:
    :return:
    """
    use_margin = True
    XAI_sum = False
    XAI_cnt = not XAI_sum
    # ignore_thresh_list = [0.1]
    ignore_thresh_list = [0.0, 0.0333, 0.0667, 0.1, 0.1333, 0.1667, 0.2]
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
    XQ_path = args.XQ_path
    XQ_folder = XQ_path.split("XAI_results/", 1)[1]
    metric_result_path = os.path.join(cwd, 'XAI_results/{}_metrics_analysis_{}'.format(XQ_folder, dt_string))

    print('\nmetric_result_path: {}'.format(metric_result_path))
    metric_res_path_str = str(metric_result_path)
    os.mkdir(metric_result_path)
    os.chmod(metric_res_path_str, 0o777)

    try:
        """
        1. Read in XQ from the TP file
        2. Read in XQ from the FP file
        3. Concatenate into one array, with TP labeled 1 and FP labeled 0
        """
        XQ_thresh_list = ['0.0333', '0.0667', '0.1', '0.1333', '0.1667', '0.2']
        for thresh in XQ_thresh_list:
            XQ_list = []
            score_list = []
            TP_FP_label = []
            found = False
            tp_name = "tp_xq_thresh{}.csv".format(thresh)
            fp_name = "fp_xq_thresh{}.csv".format(thresh)
            for root, dirs, files in os.walk(XQ_path):
                # print('processing files: ')
                for name in files:
                    # print(os.path.join(root, name))
                    if name == tp_name:
                        found = True
                        tp_data = pd.read_csv(os.path.join(root, name))
                        # print("type(tp_data['XQ']): {}".format(type(tp_data['XQ'])))
                        XQ_list.append(tp_data['XQ'])
                        score_list.append(tp_data['class_score'])
                        TP_FP_label.append(np.ones(len(tp_data['XQ'])))
                    elif name == fp_name:
                        found = True
                        fp_data = pd.read_csv(os.path.join(root, name))
                        XQ_list.append(fp_data['XQ'])
                        score_list.append(fp_data['class_score'])
                        TP_FP_label.append(np.zeros(len(fp_data['XQ'])))
            if found:
                XQ_arr = np.concatenate(XQ_list)
                score_arr = np.concatenate(score_list)
                TP_FP_arr = np.concatenate(TP_FP_label)
                print("len(XQ_arr): {}".format(len(XQ_arr)))
                print("len(TP_FP_arr): {}".format(len(TP_FP_arr)))

                eval_cols = ['fpr_at_95_tpr', 'detection_error', 'auroc', 'aupr_out', 'aupr_in']
                XQ_eval_dict = get_summary_statistics(XQ_arr, TP_FP_arr)
                plot_roc(XQ_arr, TP_FP_arr, save_path=metric_result_path, thresh=thresh, measure='XQ')
                plot_pr(XQ_arr, TP_FP_arr, save_path=metric_result_path, thresh=thresh, measure='XQ')
                eval_file = os.path.join(metric_res_path_str, "XQ_eval_metrics_thresh{}.csv".format(thresh))
                with open(eval_file, 'w') as evalfile:
                    writer = csv.DictWriter(evalfile, fieldnames=eval_cols)
                    writer.writeheader()
                    writer.writerow(XQ_eval_dict)

                score_eval_dict = get_summary_statistics(score_arr, TP_FP_arr)
                plot_roc(score_arr, TP_FP_arr, save_path=metric_result_path, thresh=thresh, measure='cls_score')
                plot_pr(score_arr, TP_FP_arr, save_path=metric_result_path, thresh=thresh, measure='cls_score')
                score_eval_file = os.path.join(metric_res_path_str, "class_score_eval_metrics.csv")
                with open(score_eval_file, 'w') as evalfile:
                    writer = csv.DictWriter(evalfile, fieldnames=eval_cols)
                    writer.writeheader()
                    writer.writerow(score_eval_dict)

                score_XQ_euclidean_arr = -1 * np.sqrt(np.multiply(1-XQ_arr, 1-XQ_arr) + np.multiply(1-score_arr, 1-score_arr))
                # max_val = np.max(score_XQ_euclidean_arr_raw)
                # score_XQ_euclidean_arr = np.divide(score_XQ_euclidean_arr_raw, max_val)
                euclidean_eval_dict = get_summary_statistics(score_XQ_euclidean_arr, TP_FP_arr)
                plot_roc(score_XQ_euclidean_arr, TP_FP_arr, save_path=metric_result_path, thresh=thresh, measure='cls_score_XQ_euclidean')
                plot_pr(score_XQ_euclidean_arr, TP_FP_arr, save_path=metric_result_path, thresh=thresh, measure='cls_score_XQ_euclidean')
                euclidean_eval_file = os.path.join(metric_res_path_str, "cls_score_XQ_euclidean_eval_metrics_thresh{}.csv".format(thresh))
                with open(euclidean_eval_file, 'w') as evalfile:
                    writer = csv.DictWriter(evalfile, fieldnames=eval_cols)
                    writer.writeheader()
                    writer.writerow(euclidean_eval_dict)

                XQ_w = 0.05
                score_w = 0.95
                score_XQ_wsum_arr = np.multiply(XQ_arr, XQ_w) + np.multiply(score_arr, score_w)
                # max_val = np.max(score_XQ_wsum_arr)
                wsum_eval_dict = get_summary_statistics(score_XQ_wsum_arr, TP_FP_arr)
                plot_roc(score_XQ_wsum_arr, TP_FP_arr, save_path=metric_result_path, thresh=thresh,
                         measure='cls_score_XQ_weighted_sum')
                plot_pr(score_XQ_wsum_arr, TP_FP_arr, save_path=metric_result_path, thresh=thresh,
                        measure='cls_score_XQ_weighted_sum')
                wsum_eval_file = os.path.join(
                    metric_res_path_str, "cls_score_{}_XQ_{}_weighted_sum_eval_metrics_thresh{}.csv".format(
                        score_w, XQ_w, thresh))
                with open(wsum_eval_file, 'w') as evalfile:
                    writer = csv.DictWriter(evalfile, fieldnames=eval_cols)
                    writer.writeheader()
                    writer.writerow(wsum_eval_dict)

                mult_arr = np.multiply(XQ_arr, score_arr)
                # max_val = np.max(score_XQ_wsum_arr)
                mult_eval_dict = get_summary_statistics(mult_arr, TP_FP_arr)
                plot_roc(mult_arr, TP_FP_arr, save_path=metric_result_path, thresh=thresh,
                         measure='cls_score_XQ_multiply')
                plot_pr(mult_arr, TP_FP_arr, save_path=metric_result_path, thresh=thresh,
                        measure='cls_score_XQ_multiply')
                mult_file = os.path.join(
                    metric_res_path_str, "cls_score_XQ_multiply_eval_metrics_thresh{}.csv".format(thresh))
                with open(mult_file, 'w') as evalfile:
                    writer = csv.DictWriter(evalfile, fieldnames=eval_cols)
                    writer.writeheader()
                    writer.writerow(mult_eval_dict)
    finally:
        print("--- {} seconds ---".format(time.time() - start_time))


if __name__ == '__main__':
    main()
