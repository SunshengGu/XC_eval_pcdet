import os
import copy
import time
import glob
import re
import h5py
import datetime
import argparse
import csv
import math
from pathlib import Path

# XAI related imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--XC_path', type=str, default=None, required=True,
                        help='the folder where all XC values are saved')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    args = parser.parse_args()
    return args

def main():
    start_time = time.time()
    args = parse_config()
    # get the date and time to create a folder for the specific time when this script is run
    now = datetime.datetime.now()
    dt_string = now.strftime("%b_%d_%Y_%H_%M_%S")
    # get current working directory
    cwd = os.getcwd()
    # create directory to store results just for this run, include the method in folder name
    XC_path = args.XC_path
    XC_folder = XC_path.split("XAI_results/", 1)[1]
    metric_result_path = os.path.join(cwd, 'XAI_results/{}_xc_hist_{}'.format(XC_folder, dt_string))
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
                print('processing files: ')
                for name in files:
                    # print(os.path.join(root, name))
                    if name == tp_name:
                        found = True
                        tp_data = pd.read_csv(os.path.join(root, name))
                        # print("type(tp_data['XQ']): {}".format(type(tp_data['XQ'])))
                        XC_list.append(tp_data['XQ'])
                        score_list.append(tp_data['class_score'])
                        TP_FP_label.append(np.ones(len(tp_data['XQ'])))
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
                        XC_list.append(fp_data['XQ'])
                        score_list.append(fp_data['class_score'])
                        TP_FP_label.append(np.zeros(len(fp_data['XQ'])))
                        cls_label_list.append(fp_data['class_label'])
                        pts_list.append(fp_data['pts_in_box'])
                        dist_list.append(fp_data['dist_to_ego'])
                        print("Number of FP instances for each class:")
                        print("class 0: {}".format(np.count_nonzero(fp_data['class_label'] == 0)))
                        print("class 1: {}".format(np.count_nonzero(fp_data['class_label'] == 1)))
                        print("class 2: {}".format(np.count_nonzero(fp_data['class_label'] == 2)))
            if found:
                XC_arr = np.concatenate(XC_list)
                score_arr = np.concatenate(score_list)
                TP_FP_arr = np.concatenate(TP_FP_label)
                cls_label_arr = np.concatenate(cls_label_list)
                pts_arr = np.concatenate(pts_list)
                dist_arr = np.concatenate(dist_list)
                print("len(XC_arr): {}".format(len(XC_arr)))
                print("len(TP_FP_arr): {}".format(len(TP_FP_arr)))

                fig, axs = plt.subplots(1, figsize=(10, 7))
                fig.tight_layout(pad=8.0)
                axs.hist(tp_data['XQ'], bins=20, alpha=0.5, range=(0.0, 1.0), label="TP")
                axs.hist(fp_data['XQ'], bins=20, alpha=0.5, range=(0.0, 1.0), label="FP")
                axs.set_title('XC Distribution', fontsize=20)
                axs.legend(loc='upper right', fontsize=20)
                # axs[1].hist(tp_data['XQ'], bins=20, range=(0.0, 1.0))
                # axs[1].set_title('TP Boxes', fontsize=20)
                # axs[2].hist(fp_data['XQ'], bins=20, range=(0.0, 1.0))
                # axs[2].set_title('FP Boxes', fontsize=20)
                # for ax in axs:
                axs.set_xlabel('XC', fontsize=20)
                axs.set_ylabel('box_count', fontsize=20)
                axs.tick_params(axis='x', labelsize=16)
                axs.tick_params(axis='y', labelsize=16)
                plt.savefig("{}/pred_box_XC_histograms_thresh{}.png".format(metric_result_path, thresh))
                plt.close()

    finally:
        print("--- {} seconds ---".format(time.time() - start_time))


if __name__ == '__main__':
    main()