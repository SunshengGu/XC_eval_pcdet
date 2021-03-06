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
import scipy

# XAI related imports
import scikitplot as skplt
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
    parser.add_argument('--pick_class', type=int, default=-1, help='the class to plot')
    args = parser.parse_args()
    return args

def main():
    old_file = True
    XC_term = "xc"
    label_term = "pred_label"
    score_term = "pred_score"
    if old_file:
        XC_term = "XQ_cnt^+"
        label_term = "class_label"
        score_term = "class_score"
    start_time = time.time()
    args = parse_config()
    # get the date and time to create a folder for the specific time when this script is run
    now = datetime.datetime.now()
    dt_string = now.strftime("%b_%d_%Y_%H_%M_%S")
    # get current working directory
    cwd = os.getcwd()
    # create directory to store results just for this run, include the method in folder name
    XC_path = args.XC_path
    interested_cls = args.pick_class
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
                        if interested_cls != -1:
                            tp_data = tp_data.loc[tp_data[label_term] == interested_cls]
                        # print("type(tp_data[XC_term]): {}".format(type(tp_data[XC_term])))
                        XC_list.append(tp_data[XC_term])
                        score_list.append(tp_data[score_term])
                        TP_FP_label.append(np.ones(len(tp_data[XC_term])))
                        cls_label_list.append(tp_data[label_term])
                        # pts_list.append(tp_data['pts_in_box'])
                        # dist_list.append(tp_data['dist_to_ego'])
                        print("Number of TP instances for each class:")
                        print("class 0: {}".format(np.count_nonzero(tp_data[label_term] == 0)))
                        print("class 1: {}".format(np.count_nonzero(tp_data[label_term] == 1)))
                        print("class 2: {}".format(np.count_nonzero(tp_data[label_term] == 2)))
                    elif name == fp_name:
                        found = True
                        fp_data = pd.read_csv(os.path.join(root, name))
                        if interested_cls != -1:
                            fp_data = fp_data.loc[fp_data[label_term] == interested_cls]
                        XC_list.append(fp_data[XC_term])
                        score_list.append(fp_data[score_term])
                        TP_FP_label.append(np.zeros(len(fp_data[XC_term])))
                        cls_label_list.append(fp_data[label_term])
                        # pts_list.append(fp_data['pts_in_box'])
                        # dist_list.append(fp_data['dist_to_ego'])
                        print("Number of FP instances for each class:")
                        print("class 0: {}".format(np.count_nonzero(fp_data[label_term] == 0)))
                        print("class 1: {}".format(np.count_nonzero(fp_data[label_term] == 1)))
                        print("class 2: {}".format(np.count_nonzero(fp_data[label_term] == 2)))
            if found:
                XC_arr = np.concatenate(XC_list)
                score_arr = np.concatenate(score_list)
                TP_FP_arr = np.concatenate(TP_FP_label)
                cls_label_arr = np.concatenate(cls_label_list)
                # pts_arr = np.concatenate(pts_list)
                # dist_arr = np.concatenate(dist_list)
                print("len(XC_arr): {}".format(len(XC_arr)))
                print("len(TP_FP_arr): {}".format(len(TP_FP_arr)))

                n_bins = 50
                fig, axs = plt.subplots(2, 1, figsize=(10, 10))
                fig.tight_layout(pad=8.0)
                axs[0].hist(tp_data[XC_term], bins=n_bins, alpha=0.5, range=(0.0, 1.0), label="TP")
                axs[0].hist(fp_data[XC_term], bins=n_bins, alpha=0.5, range=(0.0, 1.0), label="FP")
                axs[0].set_title('IG XC Distribution', fontsize=30)
                axs[0].legend(loc='upper right', fontsize=15)
                axs[0].set_xlabel('XC', fontsize=30)
                axs[0].set_ylabel('box_count', fontsize=30)
                axs[0].tick_params(axis='x', labelsize=26)
                axs[0].tick_params(axis='y', labelsize=26, rotation=60)
                axs[1].hist(tp_data[XC_term], bins=n_bins*2, density=True, cumulative=True, alpha=0.5, range=(0.0, 1.0),
                            label="TP", histtype='step', linewidth=2)
                axs[1].hist(fp_data[XC_term], bins=n_bins*2, density=True, cumulative=True, alpha=0.5, range=(0.0, 1.0),
                            label="FP", histtype='step', linewidth=2)
                ks, pval = scipy.stats.ks_2samp(tp_data[XC_term], fp_data[XC_term])
                print("ks distance is: {}".format(ks))
                axs[1].set_title('IG XC Empirical CDF, KS = {:.3f}'.format(ks), fontsize=30)
                axs[1].legend(loc='lower right', fontsize=15)
                axs[1].set_xlabel('XC', fontsize=30)
                axs[1].set_ylabel('box_count', fontsize=30)
                axs[1].tick_params(axis='x', labelsize=26)
                axs[1].tick_params(axis='y', labelsize=26)
                plt.savefig("{}/pred_box_XC_histograms_thresh{}.png".format(metric_result_path, thresh))
                plt.close()

    finally:
        print("--- {} seconds ---".format(time.time() - start_time))


if __name__ == '__main__':
    main()