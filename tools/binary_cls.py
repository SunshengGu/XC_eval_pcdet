import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
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

# binary classification
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.svm import SVC, LinearSVC


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--XQ_path', type=str, default=None, required=True,
                        help='the folder where all XQ values are saved')
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
    :param score_id: name of the score being evaluated (e.g., class score, XQ, etc.)
    :param thresh: XQ threshold
    :param cls_names: list of class names specific to a dataset
    :param cls_labels: class labels for the boxes (e.g., car, pedestrian, cyclist, etc.)
    :return:
    """
    # plot_roc(score_arr, label_arr, save_path=save_path, thresh=thresh, measure=score_id)
    # plot_pr(score_arr, label_arr, save_path=save_path, thresh=thresh, measure=score_id)
    # plot_pr(score_arr, label_arr, save_path=save_path, thresh=thresh, measure=score_id, flip=True)
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


class MyDataset(Dataset):
    # load the dataset
    def __init__(self, X, y):
        # store the inputs and outputs
        self.X = X
        self.y = np.expand_dims(y, axis=1)
        # y = torch.from_numpy(y)
        # print("y.dtype: {}".format(y.dtype))
        # self.y = nn.functional.one_hot(y.to(torch.int64), 2)

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.hidden1 = nn.Linear(2, 3)
        #self.bm1 = nn.BatchNorm1d(3)
        self.act1 = nn.ReLU()
        self.drop = nn.Dropout()
        #self.hidden2 = nn.Linear(3, 3)
        #self.bm2 = nn.BatchNorm1d(3)
        #self.act2 = nn.ReLU()
        # Output layer, 10 units - one for each digit
        self.output = nn.Linear(3, 1)
        # self.sig = nn.Sigmoid()

    def forward(self, x):
        # Pass the input tensor through each of our operations
        #x = self.drop(x)
        x = self.hidden1(x)
        #x = self.bm1(x)
        x = self.act1(x)
        #x = self.hidden2(x)
        #x = self.bm2(x)
        #x = self.act2(x)
        x = self.output(x)
        # x = self.sig(x)
        return x


def main():
    """
    important variables:
    :return:
    """
    model = "MLP"
    single_score = False
    batch_size = 8
    epochs = 20
    trials = 5
    kfolds = 5
    interested_class = 1 # 0-car, 1-pedestrian, 2-cyclist
    dataset_name = "KITTI"
    show_distribution = True
    cls_name_list = []
    if dataset_name == "KITTI":
        cls_name_list = ['Car', 'Pedestrian', 'Cyclist']
    elif dataset_name == "CADC":
        cls_name_list = ['Car', 'Pedestrian', 'Truck']
    elif dataset_name == "WAYMO":
        cls_name_list = ['Vehicle', 'Pedestrian', 'Cyclist']
    ignore_thresh_list = [0.0]
    # ignore_thresh_list = [0.0, 0.0333, 0.0667, 0.1, 0.1333, 0.1667, 0.2]
    start_time = time.time()
    high_rez = True
    args = parse_config()

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
    # os.mkdir(metric_result_path)
    # os.chmod(metric_res_path_str, 0o777)

    try:
        """
        1. Read in XQ from the TP file
        2. Read in XQ from the FP file
        3. Concatenate into one array, with TP labeled 1 and FP labeled 0
        """
        XQ_thresh_list = ['0.1']
        for thresh in XQ_thresh_list:
            pred_type_list = []
            found = False
            tp_name = "tp_xq_thresh{}.csv".format(thresh)
            fp_name = "fp_xq_thresh{}.csv".format(thresh)
            tp_data = None
            fp_data = None
            tp_len = 0
            for root, dirs, files in os.walk(XQ_path):
                # print('processing files: ')
                for name in files:
                    # print(os.path.join(root, name))
                    if name == tp_name:
                        found = True
                        tp_data = pd.read_csv(os.path.join(root, name))
                        print('tp_len before: {}'.format(len(tp_data['pred_score'])))
                        tp_data = tp_data.loc[tp_data['pred_label'] == interested_class]
                        tp_len = len(tp_data['pred_score'])
                        print('tp_len after selecting the class: {}'.format(tp_len))
                        pred_type_list.append(np.ones(len(tp_data['pred_score'])))
            for root, dirs, files in os.walk(XQ_path):
                # print('processing files: ')
                for name in files:
                    # print(os.path.join(root, name))
                    if name == fp_name:
                        found = True
                        fp_data = pd.read_csv(os.path.join(root, name))
                        #print('fp_data pre-shuffle: {}'.format(fp_data.head()))
                        print('fp_len before: {}'.format(len(fp_data['pred_score'])))
                        fp_data = fp_data.iloc[np.random.permutation(len(fp_data))]
                        fp_data = fp_data.reset_index(drop=True)
                        #print('fp_data post-shuffle: {}'.format(fp_data.head()))
                        fp_data = fp_data.loc[fp_data['pred_label'] == interested_class]
                        fp_data = fp_data.reset_index(drop=True)
                        #print('fp_data after class selection: {}'.format(fp_data.head()))
                        fp_len = len(fp_data['pred_score'])
                        print('fp_len after selecting the class: {}'.format(fp_len))
                        fp_data = fp_data.drop(labels=range(tp_len,fp_len), axis=0)
                        print('fp_len after matching tp_len: {}'.format(len(fp_data['pred_score'])))
                        pred_type_list.append(np.zeros(tp_len))
            if found:
                all_avg_acc, all_avg_auroc, all_avg_aupr, all_avg_aupr_op = 0.0, 0.0, 0.0, 0.0
                for t in range(trials):
                    # input_size = 2
                    # hidden_sizes = [4,3]
                    # output_size = 2
                    # fcnn = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_sizes[0]),
                    #                            torch.nn.ReLU(),
                    #                            torch.nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                    #                            torch.nn.ReLU(),
                    #                            torch.nn.Linear(hidden_sizes[1], output_size),
                    #                            torch.nn.Softmax(dim=1))
                    frames = [tp_data, fp_data]
                    data_df = pd.concat(frames)
                    #new_df = data_df[['pred_score']]
                    #new_df = data_df[['xc_neg_cnt']]
                    #new_df = data_df[['pred_score', 'dist']]
                    #new_df = data_df[['pred_score', 'pts']]
                    #new_df = data_df[['pred_score', 'xc_neg_cnt']]
                    # new_df = data_df[['pred_score', 'xc_pos_cnt']]
                    # new_df = data_df[['pred_score', 'xc_neg_sum']]
                    # new_df = data_df[['pred_score', 'xc_pos_sum']]
                    # new_df = data_df[['pred_score', 'pts', 'dist']]
                    new_df = data_df[['pred_score', 'xc_neg_cnt']]
                    #new_df = data_df[['pred_score', 'xc_neg_cnt', 'pts', 'dist']]
                    #new_df = data_df[['pred_score', 'xc_neg_cnt', 'xc_pos_cnt']]
                    #new_df = data_df[['xc_neg_cnt', 'xc_pos_cnt']]
                    #new_df = data_df[['xc_neg_cnt', 'xc_pos_cnt', 'xc_neg_sum', 'xc_pos_sum']]
                    #new_df = data_df[['pred_score', 'xc_neg_cnt', 'xc_neg_sum', 'xc_pos_sum', 'xc_pos_cnt']]
                    #new_df = data_df[['pred_score', 'xc_neg_cnt', 'xc_neg_sum', 'xc_pos_sum','xc_pos_cnt', 'pts', 'dist']]

                    # new_df = data_df[['XQ_cnt^+', 'class_score']]
                    # new_df = data_df[['XQ_cnt^+', 'class_score', 'pts_in_box']]
                    # new_df = data_df[['XQ_cnt^-', 'XQ_sum^-', 'XQ_cnt^+', 'XQ_sum^+', 'class_score']]
                    # new_df = data_df # [['XQ_cnt^-', 'XQ_sum^-', 'XQ_cnt^+', 'XQ_sum^+', 'class_score']] #

                    #new_df = new_df.iloc[np.random.permutation(len(new_df))]
                    #new_df.reset_index(drop=True)
                    all_data = new_df.values
                    pred_type = np.concatenate(pred_type_list)
                    print("number of entries in all_data: {}".format(len(new_df)))
                    print("number of entries in pred_type: {}".format(len(pred_type)))
                    print("shape of all_data: {}".format(all_data.shape))
                    X_train_, X_test, y_train_, y_test = train_test_split(
                        all_data, pred_type, test_size=0.001, shuffle=True, random_state=42)

                    print("training data before normalization: {}".format(new_df.head(n=10)))
                    # print("training data before normalization: {}".format(X_train_[:10]))
                    scaler = StandardScaler()
                    X_train_ = scaler.fit_transform(X_train_)
                    #y_train_ = pred_type
                    # X_train_ = scaler.fit_transform(X_train_)
                    # X_test = scaler.fit_transform(X_test)
                    # print("training data after normalization: {}".format(X_train_[:10]))
                    # X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, test_size=0.25, random_state=42)
                    k = kfolds
                    kf = KFold(n_splits=k) # , shuffle=True, random_state=42
                    C_list = [0.1, 0.5, 1, 2, 5, 10, 20, 50]
                    neigh = [5, 10, 20, 50]
                    # C_list = [0.1, 1, 5, 20]
                    val_acc = []  # validation accuracies
                    val_acc_avg = []  # mean validation accuracy for each C value
                    acc_sum = 0
                    fold = 0
                    print('\n')
                    for train_index, val_index in kf.split(X_train_):
                        fold += 1
                        X_train, X_val = X_train_[train_index], X_train_[val_index]
                        y_train, y_val = y_train_[train_index], y_train_[val_index]
                        if show_distribution:
                            train_pos = np.count_nonzero(y_train == 1)
                            val_pos = np.count_nonzero(y_val == 1)
                            pos_portion_train = train_pos/len(y_train)
                            pos_portion_val = val_pos/len(y_val)
                            print("proportion of positives in training: {}".format(pos_portion_train))
                            print("proportion of positives in validation: {}".format(pos_portion_val))
                        k_acc = []  # accuracy values for for each split
                        if model == "SVM":
                            for c in C_list:
                                clf = LinearSVC(C=c)
                                clf.fit(X_train, y_train)
                                k_acc.append(clf.score(X_val, y_val))
                                print("tried c={}".format(c))
                        elif model == "KNN":
                            for n_neigh in neigh:
                                clf = KNN(n_neighbors=n_neigh)
                                clf.fit(X_train, y_train)
                                k_acc.append(clf.score(X_val, y_val))
                                print("tried n_neigh={}".format(n_neigh))
                        elif model == "MLP":
                            train_set = MyDataset(X_train, y_train)
                            val_set = MyDataset(X_val, y_val)
                            mlp = MLP()
                            train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
                            val_dl = DataLoader(val_set, batch_size=batch_size, shuffle=False)
                            # Training
                            criterion = nn.BCEWithLogitsLoss() # binary cross entropy loss
                            # optimizer = torch.optim.SGD(mlp.parameters(), lr=0.001, momentum=0.9)
                            optimizer = torch.optim.Adam(mlp.parameters(), lr=0.001)
                            mlp.train()
                            if not single_score:
                                for epoch in range(epochs):
                                    correct = 0
                                    for i, (inputs, targets) in enumerate(train_dl):
                                        optimizer.zero_grad()
                                        inputs = inputs.float()
                                        targets = targets.float()
                                        y_hat = mlp(inputs)
                                        # print("inputs.shape: {}".format(inputs.shape))
                                        # print("y_hat.shape: {}".format(y_hat.shape))
                                        # print("targets.shape: {}".format(targets.shape))
                                        loss = criterion(y_hat, targets)
                                        loss.backward()
                                        optimizer.step()
                                        y_pred_tag = torch.round(torch.sigmoid(y_hat))
                                        correct += (y_pred_tag == targets).sum().float()
                                    train_acc = correct / len(train_set)
                                    print("epoch {} training accuracy: {}".format(epoch, train_acc))
                                    # print("correct predictions: {}".format(correct))
                            # Validation
                            mlp.eval()
                            predictions, actuals, scores = list(), list(), list()
                            for i, (inputs, targets) in enumerate(val_dl):
                                # evaluate the model on the test set
                                inputs = inputs.float()
                                targets = targets.float()
                                if not single_score:
                                    yhat = mlp(inputs)
                                    ysig = torch.sigmoid(yhat)
                                    yhat = torch.round(ysig)
                                    # retrieve numpy array
                                    yhat = yhat.detach().numpy()
                                    ysig_val = ysig.detach().numpy()
                                    actual = targets.numpy()
                                
                                    # # convert to class labels
                                    # yhat = np.argmax(yhat, axis=1)
                                    # reshape for stacking
                                    score = ysig_val.reshape((len(ysig_val),1))
                                    actual = actual.reshape((len(actual), 1))
                                    yhat = yhat.reshape((len(yhat), 1))
                                    # store
                                    predictions.append(yhat)
                                    actuals.append(actual)
                                    scores.append(score)
                                    #if i == 0:
                                        #print("actual: {}".format(actual))
                                else:
                                    score = inputs.detach().numpy()
                                    actual = targets.numpy()
                                    score = score.reshape((len(score), 1))
                                    actual = actual.reshape((len(actual), 1))
                                    predictions.append(actual)
                                    actuals.append(actual)
                                    scores.append(score)
                            predictions, actuals, scores = np.vstack(predictions), np.vstack(actuals), np.vstack(scores)
                            # print("sample predictions: {}".format(predictions[:10]))
                            # print("sample ground_truths: {}".format(actuals[:10]))
                            pos_pred_cnt = np.count_nonzero(predictions == 1)
                            val_pos_pred = pos_pred_cnt/len(predictions)
                            print("proportion of positive predictions in validation: {}".format(val_pos_pred))
                            # calculate accuracy
                            acc = accuracy_score(actuals, predictions)
                            acc_sum += acc
                            all_avg_acc += acc
                            auroc_ = auroc(scores, actuals)
                            aupr_ = aupr(scores, actuals)
                            scores = [-s for s in scores]
                            actuals = [1-a for a in actuals]
                            aupr_op = aupr(scores, actuals)
                            all_avg_auroc += auroc_
                            all_avg_aupr += aupr_
                            all_avg_aupr_op += aupr_op
                            print("the accuracy for this MLP is: {}".format(acc))
                            print("the auroc for this MLP is: {}".format(auroc_))
                            print("the aupr for this MLP is: {}".format(aupr_))
                            print("the aupr_op for this MLP is: {}".format(aupr_op))
                        val_acc.append(k_acc)
                        if acc_sum != 0:
                            avg_acc = acc_sum/fold
                            print("average accuracy for MLP is: {}".format(avg_acc))
                        print("finished one set of validation")
                    '''
                    Note on accuracy_score:
                    This is #(correct predictions)/#(all predictions)
                    Source: https://github.com/scikit-learn/scikit-learn/blob/42aff4e2e/sklearn/metrics/_classification.py#L140
                    See the `accuracy_score` function
                    '''
                    if model != "MLP":
                        for j in range(len(val_acc[0])):
                            # sum_acc = sum(val_acc[:][j])
                            sum_acc = 0
                            for i in range(len(val_acc)):
                                sum_acc += val_acc[i][j]
                            avg_acc = sum_acc / k
                            val_acc_avg.append(avg_acc)
                        print(val_acc_avg)
                all_avg_acc = all_avg_acc / (kfolds * trials)
                all_avg_auroc = all_avg_auroc / (kfolds * trials)
                all_avg_aupr = all_avg_aupr / (kfolds * trials)
                all_avg_aupr_op = all_avg_aupr_op / (kfolds * trials)
                print('\n\navg accuracy through {} trials: {}'.format(trials, all_avg_acc))
                print('\navg auroc through {} trials: {}'.format(trials, all_avg_auroc))
                print('\navg aupr through {} trials: {}'.format(trials, all_avg_aupr))
                print('\navg aupr_op through {} trials: {}'.format(trials, all_avg_aupr_op))
    finally:
        print("--- {} seconds ---".format(time.time() - start_time))


if __name__ == '__main__':
    main()
