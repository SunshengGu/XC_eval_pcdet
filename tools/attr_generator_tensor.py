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
from pcdet.datasets.waymo.waymo_dataset import WaymoDataset
from pcdet.datasets.kitti.kitti_object_eval_python.eval import d3_box_overlap
import random

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
from captum.attr import GuidedBackprop
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz

from scipy.spatial.transform import Rotation as R


class AttributionGeneratorTensor:
    def __init__(self, model, dataset_name, class_name_list, xai_method, output_path, gt_infos, pred_score_file_name, pred_score_field_name,
                 ig_steps=24, margin=0.2, num_boxes=3, selection='top', ignore_thresh=0.1, score_thresh=0.1, debug=False,
                 full_model=None):
        """
        You should have the data loader ready and the specific batch dictionary available before computing any
        attributions.
        Also, you need to have the folder XAI_utils in the `tools` folder as well for this object to work

        :param model: the model for which we are generating explanations for
        :param dataset_name:
        :param class_name_list: the list of cared classes of objects
        :param xai_method: a string indicating which explanation technique to use
        :param output_path: directory for storing relevant outputs
        :param gt_infos: ground truth information for the particular dataset
        :param ig_steps: number of intermediate steps to use for IG
        :param boxes: number of boxes to explain per frame
        :param margin: margin appended to box boundaries for XC computation
        :param num_boxes: number of predicted boxes for which we are generating explanations for
        :param selection: whether to compute XC for the most confident ('top')  or least confident ('bottom') boxes
        :param ignore_thresh: used in XC calculation to filter out small attributions
        :param score_thresh: threshold for filtering out low confidence predictions
        :param debug: whether to show debug messages
        :param full_model: the full pointpillars model
        """

        # Load model configurations
        self.model = model
        self.dataset_name = dataset_name
        self.class_name_list = class_name_list
        self.gt_infos = gt_infos
        self.score_thresh = score_thresh
        self.full_model = full_model if (full_model is not None) else None

        self.label_dict = None  # maps class name to label
        self.vicinity_dict = None  # Search vicinity for the generate_box_mask function
        if self.dataset_name == 'KittiDataset':
            self.label_dict = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
            self.vicinity_dict = {"Car": 20, "Pedestrian": 5, "Cyclist": 9}
        elif self.dataset_name == 'CadcDataset':
            self.label_dict = {"Car": 0, "Pedestrian": 1, "Truck": 2}
            self.vicinity_dict = {"Car": 13, "Pedestrian": 3, "Truck": 19}
        elif self.dataset_name == 'WaymoDataset':
            self.label_dict = {"Vehicle": 0, "Pedestrian": 1, "Cyclist": 2}
            self.vicinity_dict = {"Vehicle": 11, "Pedestrian": 4, "Cyclist": 6}
        else:
            raise NotImplementedError

        # Other initialization stuff
        self.debug = debug
        self.margin = margin
        self.output_path = output_path
        self.ig_steps = ig_steps
        self.num_boxes = num_boxes
        self.selection = selection
        self.ignore_thresh = ignore_thresh
        self.pred_score_file_name = pred_score_file_name
        self.pred_score_field_name = pred_score_field_name
        self.box_debug = True
        self.tight_iou = False
        self.batch_dict = None
        self.pred_boxes = None
        self.pred_labels = None
        self.pred_scores = None
        self.pred_vertices = None  # vertices of predicted boxes with a specified margin
        self.pred_loc = None
        self.selected_anchors = None
        self.explainer = None
        self.tp_fp = None
        self.gt_dict = None
        self.batch_cared_ind = []
        self.batch_cared_tp_ind = None # tp pred box indices for the cared boxes for which explanations are generated
        self.batch_cared_fp_ind = None  # tp pred box indices for the cared boxes for which explanations are generated
        self.batch_size = 1
        self.cur_it = 0
        self.cur_epoch = 0
        # the limits are self.num_boxes, unless the max_num_tp/fp per frame is less than self.num_boxes
        self.tp_limit = 0
        self.fp_limit = 0
        self.xai_method = xai_method
        if xai_method == 'Saliency':
            self.explainer = Saliency(self.model)
        elif xai_method == 'IntegratedGradients':
            self.explainer = IntegratedGradients(self.model, multiply_by_inputs=True)
        elif xai_method == 'GuidedBackprop':
            self.explainer = GuidedBackprop(self.model)
        elif xai_method == 'GuidedGradCam':
            # haven't tried GuidedGradCam yet, not sure if it will work or not
            self.explainer = GuidedGradCam(self.model, self.model.conv_cls)
        else:
            raise NotImplementedError

    def cuboid_to_bev(self, x, y, z, w, l, h, yaw):
        """
        Converts a cuboid in lidar coordinates to the four corners
        :param x,y,z,w,l,h,yaw : Double, specification of the bounding box in lidar frame
        :return:  np.array([[x1, y1], [x2,y2], ...[x4,y4]]) polygon in BEV view
        """

        T_Lidar_Cuboid = torch.eye(4)  # identify matrix
        T_Lidar_Cuboid = T_Lidar_Cuboid.cuda()
        # print("yaw.device: {}".format(yaw.device))
        T_Lidar_Cuboid[0:3, 0:3] = torch.from_numpy(R.from_euler('z', yaw.item(), degrees=False).as_dcm()).cuda() # rotate the identity matrix
        T_Lidar_Cuboid[0][3] = x  # center of the tracklet, from cuboid to lidar
        T_Lidar_Cuboid[1][3] = y
        T_Lidar_Cuboid[2][3] = z

        radius = 3

        # the top view of the tracklet in the "cuboid frame". The cuboid frame is a cuboid with origin (0,0,0)
        # we are making a cuboid that has the dimensions of the tracklet but is located at the origin
        front_right_top = torch.tensor(
            [[1, 0, 0, l / 2],
             [0, 1, 0, w / 2],
             [0, 0, 1, h / 2],
             [0, 0, 0, 1]]).cuda()

        front_left_top = torch.tensor(
            [[1, 0, 0, l / 2],
             [0, 1, 0, -w / 2],
             [0, 0, 1, h / 2],
             [0, 0, 0, 1]]).cuda()

        back_right_top = torch.tensor(
            [[1, 0, 0, -l / 2],
             [0, 1, 0, w / 2],
             [0, 0, 1, h / 2],
             [0, 0, 0, 1]]).cuda()

        back_left_top = torch.tensor(
            [[1, 0, 0, -l / 2],
             [0, 1, 0, -w / 2],
             [0, 0, 1, h / 2],
             [0, 0, 0, 1]]).cuda()

        # Project to lidar
        f_r_t = torch.matmul(T_Lidar_Cuboid, front_right_top)
        f_l_t = torch.matmul(T_Lidar_Cuboid, front_left_top)
        b_r_t = torch.matmul(T_Lidar_Cuboid, back_right_top)
        b_l_t = torch.matmul(T_Lidar_Cuboid, back_left_top)

        # print("f_r_t.device: {}".format(f_r_t.device))

        x1 = f_r_t[0][3]
        y1 = f_r_t[1][3]
        x2 = f_l_t[0][3]
        y2 = f_l_t[1][3]
        x3 = b_r_t[0][3]
        y3 = b_r_t[1][3]
        x4 = b_l_t[0][3]
        y4 = b_l_t[1][3]

        # # to use for the plot
        # x_img_tracklet = -1 * y  # in the image to plot, the negative lidar y axis is the img x axis
        # y_img_tracklet = x  # the lidar x axis is the img y axis
        poly = torch.tensor([[-1 * y1, x1], [-1 * y2, x2], [-1 * y4, x4], [-1 * y3, x3]]).cuda()
        # print("poly.dtype: {}".format(poly.dtype))
        # print("poly.device: {}".format(poly.device))
        return poly

    def calculate_iou(self, gt_boxes, pred_boxes_raw, dataset_name, ret_overlap=False):
        # convert to numpy arrays to be compatible with d3_box_overlap
        pred_boxes = pred_boxes_raw.detach().cpu().numpy()
        # see pcdet/datasets/kitti/kitti_object_eval_python/eval.py for explanation
        z_axis = 1  # welp... it is really the y-axis
        z_center = 1.0
        if dataset_name == 'CadcDataset':
            z_center = 0.5
        if self.debug:
            print("type(gt_boxes): {}".format(type(gt_boxes)))
            print("type(pred_boxes): {}".format(type(pred_boxes)))
        overlap = d3_box_overlap(gt_boxes, pred_boxes, z_axis=z_axis, z_center=z_center)
        #TODO: modify the following line to working torch tensor
        iou, gt_index = np.max(overlap, axis=0), np.argmax(overlap, axis=0)
        if ret_overlap:
            return iou, gt_index, overlap
        return iou, gt_index

    def get_preds_single_frame(self, pred_dicts):
        pred_boxes = pred_dicts[0]['pred_boxes']
        pred_labels = pred_dicts[0]['pred_labels'] - 1
        self.pred_boxes = pred_boxes
        self.pred_labels = pred_labels
        self.pred_scores = pred_dicts[0]['pred_scores']
        self.selected_anchors = self.batch_dict['anchor_selections'][0]

    def get_preds(self):
        # # Debug message
        # print("keys in self.batch_dict:")
        # for key in self.batch_dict:
        #     print('key: {}'.format(key))

        pred_dicts = self.batch_dict['pred_dicts']
        # print('len(pred_dicts): {}'.format(len(pred_dicts)))
        self.batch_size = self.batch_dict['batch_size']
        # print('self.batch_size: {}'.format(self.batch_size))
        pred_boxes, pred_labels, pred_scores, selected_anchors = [], [], [], []
        for i in range(self.batch_size):
            # print("\nkeys in pred_dicts[{}]:".format(i))
            # for key in pred_dicts[i]:
                # print('key: {}'.format(key))
            frame_pred_boxes = pred_dicts[i]['pred_boxes']
            # print('len(frame_pred_boxes): {}'.format(len(frame_pred_boxes)))
            frame_pred_labels = pred_dicts[i]['pred_labels'] - 1 # this operation is valid
            # print('len(frame_pred_labels): {}'.format(len(frame_pred_labels)))
            pred_boxes.append(frame_pred_boxes)
            pred_labels.append(frame_pred_labels)
            pred_scores.append(pred_dicts[i]['pred_scores'])
            selected_anchors.append(self.batch_dict['anchor_selections'][i])
        self.pred_boxes = pred_boxes
        self.pred_labels = pred_labels
        self.pred_scores = pred_scores
        self.selected_anchors = selected_anchors

        # print("self.pred_boxes: {}".format(self.pred_boxes))
        # print("self.pred_labels: {}".format(self.pred_labels))
        # print("self.pred_scores: {}".format(self.pred_scores))

    def reset(self):
        """
        Call this when you have finished using this object for a specific batch
        """
        self.batch_dict = None
        self.pred_boxes = None
        self.pred_labels = None
        self.pred_scores = None
        self.pred_vertices = None
        self.pred_loc = None
        self.selected_anchors = None
        self.tp_fp = None
        self.gt_dict = None
        self.tp_limit = 0
        self.fp_limit = 0
        self.cur_it = 0
        self.batch_cared_ind = []
        self.batch_cared_tp_ind = None
        self.batch_cared_fp_ind = None

    def get_attr(self, target):
        """
        Unless the batch_size is 1, there will be repeated work done in a batch since the same target is being applied
        to all samples in the same batch
        """
        PseudoImage2D = self.batch_dict['spatial_features']
        # print("PseudoImage2D.size(): {}".format(PseudoImage2D.size()))
        attr_size = PseudoImage2D[0].size()
        d, h, w = attr_size[0], attr_size[1], attr_size[2]
        zero_tensor = torch.zeros((h, w, d), dtype=PseudoImage2D[0].dtype).cuda()
        if self.xai_method == "IntegratedGradients":
            print("type(self.batch_dict['batch_size']): {}".format(type(self.batch_dict['batch_size'])))
            print("self.batch_dict['batch_size']: {}".format(self.batch_dict['batch_size']))
            print("type(self.ig_steps): {}".format(type(self.ig_steps)))
            print("self.ig_steps: {}".format(self.ig_steps))
            time.sleep(0.001)
            batch_grad = self.explainer.attribute(PseudoImage2D, baselines=PseudoImage2D * 0, target=target,
                                                  additional_forward_args=self.batch_dict, n_steps=self.ig_steps,
                                                  internal_batch_size=self.batch_dict['batch_size'])
        elif self.xai_method == "Saliency":
            batch_grad = self.explainer.attribute(PseudoImage2D, target=target, additional_forward_args=self.batch_dict)

        grads = [gradients.squeeze().permute(1, 2, 0) for gradients in batch_grad]
        print("\ngrads[0].size(): {}".format(grads[0].size()))
        pos_grad = [torch.sum(torch.where(grad > 0, grad, zero_tensor), axis=2) for grad in grads]
        neg_grad = [torch.sum(-1 * torch.where(grad < 0, grad, zero_tensor), axis=2) for grad in grads]
        print("\nsummarized grad")
        return pos_grad, neg_grad

    def compute_pred_box_vertices_single_frame(self):
        # Generate the vertices
        expanded_preds = []
        pred_loc = []
        for cuboid in self.pred_boxes:
            x, y, z, w, l, h, yaw = cuboid
            w_big = w + 2 * self.margin
            l_big = l + 2 * self.margin
            expanded_preds.append(self.cuboid_to_bev(x, y, z, w_big, l_big, h, yaw))
            pred_loc.append(np.array([x, y]))

        # Limiting the viewing range
        s1, s2, f1, f2 = 0, 0, 0, 0
        if self.dataset_name == 'KittiDataset':
            s1, s2, f1, f2 = 39.68, 39.68, 0.0, 69.12
        elif self.dataset_name == 'CadcDataset':
            s1, s2, f1, f2 = 50.0, 50.0, 50.0, 50.0
        elif self.dataset_name == 'WaymoDataset':
            s1, s2, f1, f2 = 75.2, 75.2, 75.2, 75.2
        else:
            raise NotImplementedError
        side_range = [-s1, s2]
        fwd_range = [-f1, f2]
        offset = np.array([[-side_range[0], -fwd_range[0]]] * 4)
        expanded_preds = [poly + offset for poly in expanded_preds]
        self.pred_vertices = expanded_preds
        self.pred_loc = pred_loc

    def compute_pred_box_vertices(self):
        """
        Given the predicted boxes in [x, y, z, w, l, h, yaw] format, return the expanded predicted boxes in terms of
        box vertices: np.array([[x1, y1], [x2,y2], ...[x4,y4]])
        """
        for i in range(self.batch_size):
            for j in range(len(self.pred_boxes[i])):
                self.pred_boxes[i][j][6] += np.pi / 2
        batch_expanded_preds = []
        batch_pred_loc = []
        # Generate the vertices
        for i in range(self.batch_size):
            expanded_preds = []
            pred_loc = []
            for cuboid in self.pred_boxes[i]:
                # print("type(cuboid): {}".format(type(cuboid))) # I made it tensor
                x, y, z, w, l, h, yaw = cuboid
                # print("type(x): {}".format(type(x))) # I made it tensor
                w_big = w + 2 * self.margin
                l_big = l + 2 * self.margin
                expanded_preds.append(self.cuboid_to_bev(x, y, z, w_big, l_big, h, yaw))
                pred_loc.append(np.array([x, y]))
                # print("passed self.cuboid_to_bev")
            # Limiting the viewing range
            s1, s2, f1, f2 = 0, 0, 0, 0
            if self.dataset_name == 'KittiDataset':
                s1, s2, f1, f2 = 39.68, 39.68, 0.0, 69.12
            elif self.dataset_name == 'CadcDataset':
                s1, s2, f1, f2 = 50.0, 50.0, 50.0, 50.0
            elif self.dataset_name == 'WaymoDataset':
                s1, s2, f1, f2 = 75.2, 75.2, 75.2, 75.2
            else:
                raise NotImplementedError
            side_range = [-s1, s2]
            fwd_range = [-f1, f2]
            offset = np.array([[-side_range[0], -fwd_range[0]]] * 4, dtype="float32")
            offset = torch.from_numpy(offset).cuda()
            # print("offset.dtype: {}".format(offset.dtype))
            expanded_preds = [poly + offset for poly in expanded_preds]
            batch_expanded_preds.append(expanded_preds)
            batch_pred_loc.append(pred_loc)
        # self.pred_vertices = np.array(batch_expanded_preds)
        # self.pred_loc = np.array(batch_pred_loc)
        self.pred_vertices = batch_expanded_preds
        self.pred_loc = batch_pred_loc

    def get_tp_fp_indices(self, cur_it):
        # obtain gt information
        self.gt_dict = self.gt_infos[cur_it * self.batch_size:(cur_it + 1) * self.batch_size]
        conf_mat = []
        gt_exist = []
        pred_box_iou = []
        d3_iou_thresh = []
        d3_iou_thresh_dict = {}

        if self.dataset_name == 'KittiDataset':
            if self.tight_iou:
                d3_iou_thresh = [0.7, 0.5, 0.5]
                d3_iou_thresh_dict = {'Car': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.5}
            else:
                d3_iou_thresh = [0.5, 0.25, 0.25]
                d3_iou_thresh_dict = {'Car': 0.5, 'Pedestrian': 0.25, 'Cyclist': 0.25}
        elif self.dataset_name == 'CadcDataset':
            if self.tight_iou:
                d3_iou_thresh = [0.7, 0.5, 0.7]
                d3_iou_thresh_dict = {'Car': 0.7, 'Pedestrian': 0.5, 'Truck': 0.7}
            else:
                d3_iou_thresh = [0.5, 0.25, 0.5]
                d3_iou_thresh_dict = {'Car': 0.5, 'Pedestrian': 0.25, 'Truck': 0.5}
        elif self.dataset_name == 'WaymoDataset':
            if self.tight_iou:
                d3_iou_thresh = [0.7, 0.5, 0.7]
                d3_iou_thresh_dict = {'Vehicle': 0.7, 'Pedestrian': 0.5, 'Cyclist': 0.7}
            else:
                d3_iou_thresh = [0.5, 0.25, 0.5]
                d3_iou_thresh_dict = {'Vehicle': 0.5, 'Pedestrian': 0.25, 'Cyclist': 0.5}

        # filtering out out-of-range gt boxes
        for i in range(self.batch_size):
            conf_mat_frame = []
            x_low = -1.5
            x_high = 70.5
            y_low = -40.5
            y_high = 40.5
            if self.dataset_name == 'WaymoDataset':
                x_low = -76
                x_high = 76
                y_low = -76
                y_high = 76
            # filter out out-of-range gt boxes for Kitti
            filtered_gt_boxes = []
            filtered_gt_labels = []
            for gt_ind in range(len(self.gt_dict[i]['boxes'])):
                x, y = self.gt_dict[i]['boxes'][gt_ind][0], self.gt_dict[i]['boxes'][gt_ind][1]
                if (x_low < x < x_high) and (y_low < y < y_high):
                    filtered_gt_boxes.append(self.gt_dict[i]['boxes'][gt_ind])
                    filtered_gt_labels.append(self.gt_dict[i]['labels'][gt_ind])
            if len(filtered_gt_boxes) != 0:
                print("len(filtered_gt_boxes): {}".format(len(filtered_gt_boxes)))
                self.gt_dict[i]['boxes'] = np.vstack(filtered_gt_boxes)
            else:
                self.gt_dict[i]['boxes'] = filtered_gt_boxes
            self.gt_dict[i]['labels'] = filtered_gt_labels

            # need to handle the case when no gt boxes exist
            gt_present = True
            if len(self.gt_dict[i]['boxes']) == 0:
                gt_present = False
            gt_exist.append(gt_present)
            if not gt_present:
                for j in range(len(self.pred_scores)):
                    curr_pred_score = self.pred_scores[i][j]
                    if curr_pred_score >= self.score_thresh:
                        conf_mat_frame.append('FP')
                        # print("pred_box_ind: {}, pred_label: {}, didn't match any gt boxes".format(j, pred_name))
                    else:
                        conf_mat_frame.append('ignore')  # these boxes do not meet score thresh, ignore for now
                        print("pred_box {} has score below threshold: {}".format(j, curr_pred_score))
                conf_mat.append(conf_mat_frame)
                continue

            # when we indeed have some gt boxes

            iou, gt_index, overlaps = self.calculate_iou(
                self.gt_dict[i]['boxes'], self.pred_boxes[i], self.dataset_name, ret_overlap=True)
            pred_box_iou.append(iou)
            print("len(self.pred_scores[i]): {}".format(len(self.pred_scores[i])))
            for j in range(len(self.pred_scores[i])):  # j is prediction box id in the i-th image
                gt_cls = self.gt_dict[i]['labels'][gt_index[j]]
                print("gt class name: {}".format(gt_cls))
                # print("self.class_name_list: {}".format(self.class_name_list))
                iou_thresh_3d = d3_iou_thresh_dict[gt_cls]
                curr_pred_score = self.pred_scores[i][j]
                if curr_pred_score >= self.score_thresh:
                    adjusted_pred_boxes_label = self.pred_labels[i][j]
                    pred_name = self.class_name_list[adjusted_pred_boxes_label]
                    print("pred_name: {}".format(pred_name))
                    if iou[j] >= iou_thresh_3d:
                        if gt_cls == pred_name:
                            conf_mat_frame.append('TP')
                        else:
                            conf_mat_frame.append('FP')
                        if self.box_debug:
                            print("pred_box_ind: {}, pred_label: {}, gt_ind is {}, gt_label: {}, iou: {}".format(
                                j, pred_name, gt_index[j], gt_cls, iou[j]))
                    elif iou[j] > 0:
                        conf_mat_frame.append('FP')
                        if self.box_debug:
                            print("pred_box_ind: {}, pred_label: {}, gt_ind is {}, gt_label: {}, iou: {}".format(
                                j, pred_name, gt_index[j], gt_cls, iou[j]))
                    else:
                        conf_mat_frame.append('FP')
                        if self.box_debug:
                            print("pred_box_ind: {}, pred_label: {}, didn't match any gt boxes".format(j, pred_name))
                else:
                    conf_mat_frame.append('ignore')  # these boxes do not meet score thresh, ignore for now
                    print("pred_box {} has score below threshold: {}".format(j, curr_pred_score))
            conf_mat.append(conf_mat_frame)
        self.tp_fp = conf_mat

    def record_pred_results(self, cared_xc, cared_far_attr, cared_pap, cared_fp_xc, cared_fp_far_attr, cared_fp_pap):
        """
        Record the predicted labels and scores
        :return:
        """
        with open(self.pred_score_file_name, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=self.pred_score_field_name)
            if self.selection == "tp/fp" or self.selection == "tp/fp_all":
                for i in range(len(self.batch_cared_tp_ind)):
                    # note that the length of self.batch_cared_tp_ind[i] will not exceed the actual number of tps
                    # in a frame
                    for j in range(len(self.batch_cared_tp_ind[i])):
                        ind = self.batch_cared_tp_ind[i][j]
                        data_dict = {"epoch": self.cur_epoch, "batch": self.cur_it, "tp/fp": "tp",
                                     "pred_label": self.pred_labels[i][ind].item(), "pred_score": self.pred_scores[i][ind].item(),
                                     "xc": cared_xc[i][j].item(), "far_attr": cared_far_attr[i][j].item(), 
                                     "pap": cared_pap[i][j].item()}
                        writer.writerow(data_dict)
                for i in range(len(self.batch_cared_fp_ind)):
                    # note that the length of self.batch_cared_fp_ind[i] will not exceed the actual number of fps
                    # in a frame
                    for j in range(len(self.batch_cared_fp_ind[i])):
                        ind = self.batch_cared_fp_ind[i][j]
                        data_dict = {"epoch": self.cur_epoch, "batch": self.cur_it, "tp/fp": "fp",
                                     "pred_label": self.pred_labels[i][ind].item(), "pred_score": self.pred_scores[i][ind].item(),
                                     "xc": cared_fp_xc[i][j].item(), "far_attr": cared_fp_far_attr[i][j].item(), 
                                     "pap": cared_fp_pap[i][j].item()}
                        writer.writerow(data_dict)
            elif self.selection == "tp":
                for i in range(len(self.batch_cared_tp_ind)):
                    # note that the length of self.batch_cared_tp_ind[i] will not exceed the actual number of tps
                    # in a frame
                    for j in range(len(self.batch_cared_tp_ind[i])):
                        ind = self.batch_cared_tp_ind[i][j]
                        data_dict = {"epoch": self.cur_epoch, "batch": self.cur_it, "tp/fp": "tp",
                                     "pred_label": self.pred_labels[i][ind].item(), "pred_score": self.pred_scores[i][ind].item(),
                                     "xc": cared_xc[i][j].item(), "far_attr": cared_far_attr[i][j].item(), 
                                     "pap": cared_pap[i][j].item()}
                        writer.writerow(data_dict)
            else:
                for i in range(len(self.batch_cared_ind)):
                    for j in range(len(self.batch_cared_ind[i])):
                        ind = self.batch_cared_ind[i][j]
                        data_dict = {"epoch": self.cur_epoch, "batch": self.cur_it,
                                     "pred_label": self.pred_labels[i][ind].item(), "pred_score": self.pred_scores[i][ind].item(),
                                     "xc": cared_xc[i][j].item(), "far_attr": cared_far_attr[i][j].item(), 
                                     "pap": cared_pap[i][j].item()}
                        writer.writerow(data_dict)

    def generate_targets_single_frame(self, epoch_obj_cnt):
        """
        only gets called when the batch_size is 1
        This won't work at the moment
        :return:
        """
        target_list = []
        cared_labels = None
        cared_box_vertices = None
        cared_loc = None
        fp_target_list = []
        fp_labels = None
        fp_box_vertices = None
        fp_loc = None
        if self.selection == "tp/fp" or self.selection == "tp" or self.selection == "tp/fp_all":
            cared_indices = [i for i in range(len(self.tp_fp)) if self.tp_fp[i] == "TP"]
            if self.selection != "tp/fp_all" and len(cared_indices) > self.num_boxes:
                cared_indices = random.sample(cared_indices, self.num_boxes)
            self.batch_cared_tp_ind = cared_indices
            cared_labels = [self.pred_labels[i] for i in cared_indices]
            cared_box_vertices = [self.pred_vertices[i] for i in cared_indices]
            cared_loc = [self.pred_loc[i] for i in cared_indices]
            fp_indices = [i for i in range(len(self.tp_fp)) if self.tp_fp[i] == "FP"]
            if self.selection != "tp/fp_all" and len(fp_indices) > self.num_boxes:
                fp_indices = random.sample(fp_indices, self.num_boxes)
            self.batch_cared_fp_ind = fp_indices
            fp_labels = [self.pred_labels[i] for i in fp_indices]
            fp_box_vertices = [self.pred_vertices[i] for i in fp_indices]
            fp_loc = [self.pred_loc[i] for i in fp_indices]
            if self.debug:
                print("cared tp indices: {}".format(cared_indices))
                print("cared fp indices: {}".format(fp_indices))
        elif self.num_boxes >= len(self.pred_labels):
            self.batch_cared_ind = range(len(self.pred_labels))
            cared_labels = self.pred_labels
            cared_box_vertices = self.pred_vertices
            cared_loc = self.pred_loc
        elif self.selection == "top":
            self.batch_cared_ind = range(self.num_boxes)
            cared_labels = self.pred_labels[:self.num_boxes]
            cared_box_vertices = self.pred_vertices[:self.num_boxes]
            cared_loc = self.pred_loc[:self.num_boxes]
        elif self.selection == "bottom":
            self.batch_cared_ind = range(len(self.pred_labels) - self.num_boxes, len(self.pred_labels))
            cared_labels = self.pred_labels[-1 * self.num_boxes:]
            cared_box_vertices = self.pred_vertices[-1 * self.num_boxes:]
            cared_loc = self.pred_loc[-1 * self.num_boxes:]

        for k in range(len(self.class_name_list)):
            epoch_obj_cnt[k] += np.count_nonzero(cared_labels == k)
        # Generate targets
        for ind, label in enumerate(cared_labels):
            if self.selection == "tp/fp" or self.selection == "tp" or self.selection == "tp/fp_all":
                pred_id = self.batch_cared_tp_ind[ind]
                target_list.append((self.selected_anchors[pred_id], label))
            else:
                target_list.append((self.selected_anchors[ind], label))
        if self.selection == "tp/fp" or self.selection == "tp" or self.selection == "tp/fp_all":
            for ind, label in enumerate(fp_labels):
                pred_id = self.batch_cared_fp_ind[ind]
                fp_target_list.append((self.selected_anchors[pred_id], label))
            return target_list, cared_box_vertices, cared_loc, fp_target_list, fp_box_vertices, fp_loc
        return target_list, cared_box_vertices, cared_loc

    def generate_targets(self, epoch_obj_cnt, epoch_tp_obj_cnt, epoch_fp_obj_cnt):
        """
        Generates targets for explanation given the batch
        Note that the class label target for each box corresponds to the top confidence predicted class
        :return:
        """
        # Get pred labels
        batch_target_list = []
        batch_fp_target_list = []
        batch_cared_labels = []
        batch_cared_box_vertices = []
        batch_cared_loc = []
        batch_fp_box_vertices = []
        batch_fp_loc = []
        batch_tp_indices = []
        batch_fp_indices = []
        batch_cared_indices = []
        max_num_tp = 0
        max_num_fp = 0
        if self.selection == "tp/fp" or self.selection == "tp" or self.selection == "tp/fp_all":
            for f in range(self.batch_size):
                print("frame id: {}\ntp_fp list: {}".format(f, self.tp_fp[f]))
                tp_indices = [k for k in range(len(self.tp_fp[f])) if self.tp_fp[f][k] == "TP"]
                fp_indices = [k for k in range(len(self.tp_fp[f])) if self.tp_fp[f][k] == "FP"]
                print("tp indices: {}".format(tp_indices))
                print("fp indices: {}".format(fp_indices))
                batch_tp_indices.append(tp_indices)
                batch_fp_indices.append(fp_indices)
                max_num_tp = max(len(tp_indices), max_num_tp)
                max_num_fp = max(len(fp_indices), max_num_fp)
            if self.selection == "tp/fp_all":
                self.tp_limit = max_num_tp
                self.fp_limit = max_num_fp
            else:
                self.tp_limit = min(self.num_boxes, max_num_tp)
                self.fp_limit = min(self.num_boxes, max_num_fp)
            print("self.tp_limit: {} self.fp_limit: {}".format(self.tp_limit, self.fp_limit))
        self.batch_cared_tp_ind = batch_tp_indices
        self.batch_cared_fp_ind = batch_fp_indices
        for i in range(self.batch_size):
            target_list = []
            cared_labels = None
            cared_labels_arr = None
            cared_box_vertices = None
            cared_loc = None
            fp_target_list = []
            fp_labels = None
            fp_labels_arr = None
            fp_box_vertices = None
            fp_loc = None
            if self.selection == "tp/fp" or self.selection == "tp" or self.selection == "tp/fp_all":
                # tp selection
                cared_indices = batch_tp_indices[i]
                if self.selection != "tp/fp_all" and len(cared_indices) > self.tp_limit: # too many boxes
                    cared_indices = random.sample(cared_indices, self.tp_limit)
                self.batch_cared_tp_ind[i] = cared_indices
                cared_labels = [self.pred_labels[i][ind] for ind in cared_indices]
                cared_labels_arr = np.array(cared_labels)
                cared_box_vertices = [self.pred_vertices[i][ind] for ind in cared_indices]
                cared_loc = [self.pred_loc[i][ind] for ind in cared_indices]
                # fp selection
                fp_indices = batch_fp_indices[i]
                if self.selection != "tp/fp_all" and len(fp_indices) > self.fp_limit: # too many boxes
                    fp_indices = random.sample(fp_indices, self.fp_limit)
                self.batch_cared_fp_ind[i] = fp_indices
                fp_labels = [self.pred_labels[i][ind] for ind in fp_indices]
                fp_labels_arr = np.array(fp_labels)
                fp_box_vertices = [self.pred_vertices[i][ind] for ind in fp_indices]
                fp_loc = [self.pred_loc[i][ind] for ind in fp_indices]
                if self.debug:
                    print("cared tp indices: {}".format(cared_indices))
                    print("cared fp indices: {}".format(fp_indices))
                #     print("cared tp labels: {}".format(cared_labels))
                #     print("cared fp labels: {}".format(fp_labels))
                #     print("cared tp box vertices: {}".format(cared_box_vertices))
                #     print("cared fp box vertices: {}".format(fp_box_vertices))
                #     print("cared tp locs: {}".format(cared_loc))
                #     print("cared fp locs: {}".format(fp_loc))
            elif self.num_boxes >= len(self.pred_labels[i]):
                batch_cared_indices.append(range(len(self.pred_labels[i])))
                cared_labels = self.pred_labels[i]
                cared_box_vertices = self.pred_vertices[i]
                cared_loc = self.pred_loc[i]
            elif self.selection == "top":
                # As long as self.num_boxes is small (say 3), it's OK to just brute force picking the top few elements
                batch_cared_indices.append(range(self.num_boxes))
                cared_labels = self.pred_labels[i][:self.num_boxes]
                cared_box_vertices = self.pred_vertices[i][:self.num_boxes]
                cared_loc = self.pred_loc[i][:self.num_boxes]
            elif self.selection == "bottom":
                frame_cared_indices = range(len(self.pred_labels[i]) - self.num_boxes, len(self.pred_labels[i]))
                batch_cared_indices.append(frame_cared_indices)
                # TODO: recreate the following 3 variables again using frame_cared_indices
                cared_labels = self.pred_labels[i][-1 * self.num_boxes:]
                cared_box_vertices = self.pred_vertices[i][-1 * self.num_boxes:]
                cared_loc = self.pred_loc[i][-1 * self.num_boxes:]
                # cared_labels = [self.pred_labels[i][ind] for ind in frame_cared_indices]
                # cared_labels_arr = np.array(cared_labels)
                # cared_box_vertices = [self.pred_vertices[i][ind] for ind in frame_cared_indices]
                # cared_loc = [self.pred_loc[i][ind] for ind in frame_cared_indices]
                if self.debug:
                    print("type(frame_cared_indices): {}".format(type(frame_cared_indices)))
                    print("frame {}  frame_cared_indices: {}".format(i, frame_cared_indices))
                    print("len(cared_labels): {}".format(len(cared_labels)))
                    print("len(cared_box_vertices): {}".format(len(cared_box_vertices)))
                    print("len(cared_loc): {}".format(len(cared_loc)))
            # print("\ncared_labels.device: {}\n".format(cared_labels.device))
            # print("type(cared_labels): {} \ncared_labels.dtype: {} \ncared_labels.size(): {}".format(type(cared_labels), cared_labels.dtype, cared_labels.size()))

            if self.selection == "tp/fp" or self.selection == "tp" or self.selection == "tp/fp_all":
                for k in range(len(self.class_name_list)):
                    if self.debug:
                        print("type(cared_labels): {}".format(type(cared_labels)))
                        print("type(fp_labels): {}".format(type(fp_labels)))
                    epoch_tp_obj_cnt[k] += cared_labels.count(k) if isinstance(cared_labels, list) else torch.sum(cared_labels == k)
                    epoch_fp_obj_cnt[k] += fp_labels.count(k) if isinstance(fp_labels, list) else torch.sum(fp_labels == k)
                    epoch_obj_cnt[k] = epoch_tp_obj_cnt[k] + epoch_fp_obj_cnt[k]
            else:
                for k in range(len(self.class_name_list)):
                    epoch_obj_cnt[k] += cared_labels.count(k) if isinstance(cared_labels, list) else torch.sum(cared_labels == k)
            if self.debug:
                print("epoch_obj_cnt: {}".format(epoch_obj_cnt))
                print("epoch_tp_obj_cnt: {}".format(epoch_tp_obj_cnt))
                print("epoch_fp_obj_cnt: {}".format(epoch_fp_obj_cnt))
            # Generate targets
            for ind, label in enumerate(cared_labels):
                if self.selection == "tp/fp" or self.selection == "tp" or self.selection == "tp/fp_all":
                    print("pred_ids:")
                    pred_id = self.batch_cared_tp_ind[i][ind]
                    print("pred_id: {}".format(pred_id))
                    print("pred_label: {}".format(label))
                    target_list.append((self.selected_anchors[i][pred_id], label))
                elif self.selection == "top":
                    target_list.append((self.selected_anchors[i][ind], label))
                elif self.selection == "bottom":
                    pred_id = batch_cared_indices[i][ind]
                    target_list.append((self.selected_anchors[i][pred_id], label))
            if self.selection == "tp/fp" or self.selection == "tp" or self.selection == "tp/fp_all":
                print("fp pred_ids:")
                for ind, label in enumerate(fp_labels):
                    pred_id = self.batch_cared_fp_ind[i][ind]
                    print("pred_id: {}".format(pred_id))
                    print("pred_label: {}".format(label))
                    fp_target_list.append((self.selected_anchors[i][pred_id], label))
                print("frame {} tp target_list: {}".format(i, target_list))
                print("frame {} fp target_list: {}".format(i, fp_target_list))
                if len(target_list) < self.tp_limit: # too few boxes
                    for c in range(len(target_list), self.tp_limit):
                        # creating some dummy entries
                        print("appending None to the tp vertices")
                        target_list.append((0, 0))
                        cared_box_vertices.append(None)
                        cared_loc.append(None)
                if len(fp_target_list) < self.fp_limit: # too few boxes
                    for c in range(len(fp_target_list), self.fp_limit):
                        # creating some dummy entries
                        print("appending None to the fp vertices")
                        fp_target_list.append((0, 0))
                        fp_box_vertices.append(None)
                        fp_loc.append(None)
                batch_fp_target_list.append(fp_target_list)
                batch_fp_box_vertices.append(fp_box_vertices)
                batch_fp_loc.append(fp_loc)
            batch_target_list.append(target_list)
            batch_cared_box_vertices.append(cared_box_vertices)
            batch_cared_loc.append(cared_loc)
        self.batch_cared_ind = batch_cared_indices
        if self.selection == "tp/fp" or self.selection == "tp/fp_all":
            return batch_target_list, batch_cared_box_vertices, batch_cared_loc, batch_fp_target_list, batch_fp_box_vertices, batch_fp_loc
        return batch_target_list, batch_cared_box_vertices, batch_cared_loc

    def compute_xc_single(self, pos_grad, neg_grad, box_vertices, dataset_name, sign, ignore_thresh, box_loc,
                          vicinity, method):
        """
        :param vicinity: search vicinity w.r.t. the box_center, class dependent
        :param ignore_thresh: the threshold below which the attributions would be ignored
        :param box_loc: location of the predicted box
        :param sign: indicates the type of attributions shown, positive or negative
        :param neg_grad: numpy array containing sum of negative gradients at each location
        :param pos_grad: numpy array containing sum of positive gradients at each location
        :param dataset_name:
        :param box_vertices: The vertices of the predicted box
        :param method: either "cnt" or "sum"
        :return: XC for a single box
        """
        XC, attr_in_box, far_attr, total_attr = 0.0, 0.0, 0.0, 0.0
        XC_lst, far_attr_lst = [], []
        for i in range(self.batch_size):
            if box_vertices[i] is None:
                XC_lst.append(torch.tensor(float('nan')).cuda())
                far_attr_lst.append(torch.tensor(float('nan')).cuda())
            else:
                XC_, far_attr_ = 0, 0
                if method == "cnt":
                    XC_, attr_in_box_, far_attr_, total_attr_ = get_cnt_XQ_analytics_fast_tensor(
                        pos_grad[i], neg_grad[i], box_vertices[i], dataset_name, sign, ignore_thresh, box_loc[i],
                        vicinity[i])
                elif method == "sum":
                    XC_, attr_in_box_, far_attr_, total_attr_ = get_sum_XQ_analytics_fast_tensor(
                        pos_grad[i], neg_grad[i], box_vertices[i], dataset_name, sign, ignore_thresh, box_loc[i],
                        vicinity[i])
                XC_lst.append(XC_)
                far_attr_lst.append(far_attr_)
                # debug message
                if self.debug:
                    print("frame id in batch: {} XC: {}".format(i, XC_))
        return XC_lst, far_attr_lst

    def get_PAP_single(self, pos_grad, neg_grad, sign):
        grad = None
        if sign == 'positive':
            grad = pos_grad
        elif sign == 'negative':
            grad = neg_grad
        pap_loss = []

        for i in range(self.batch_size):
            diff_1 = grad[i][1:, :] - grad[i][:-1, :]
            diff_2 = grad[i][:, 1:] - grad[i][:, :-1]
            pap_loss.append(torch.sum(torch.abs(diff_1)) + torch.sum(torch.abs(diff_2)))
        return pap_loss

    def xc_preprocess(self, batch_dict, epoch_obj_cnt, epoch_tp_obj_cnt, epoch_fp_obj_cnt, cur_it):
        # Get the predictions, compute box vertices, and generate targets
        self.batch_dict = batch_dict
        self.cur_it = cur_it
        if self.full_model is not None:
            print("full_model is not None")
            load_data_to_gpu(self.batch_dict)
            dummy_tensor = 0
            with torch.no_grad():
                # run the model once to populate the batch dict
                anchor_scores = self.full_model(dummy_tensor, self.batch_dict)
        self.get_preds()
        if self.selection == "tp/fp" or self.selection == "tp" or self.selection == "tp/fp_all":
            self.get_tp_fp_indices(cur_it)
        self.compute_pred_box_vertices()
        return self.generate_targets(epoch_obj_cnt, epoch_tp_obj_cnt, epoch_fp_obj_cnt)

    def compute_xc(self, batch_dict, epoch_obj_cnt, epoch_tp_obj_cnt, epoch_fp_obj_cnt, cur_it, cur_epoch,
                   method="cnt", sign="positive"):
        """
        This is the ONLY function that the user should call

        :param cur_epoch: current epoch
        :param cur_it: current batch id
        :param epoch_fp_obj_cnt: tp object counts by class
        :param epoch_tp_obj_cnt: tp object counts by class
        :param batch_dict: The input batch for which we are generating explanations
        :param method: Either by counting ("cnt") or by summing ("sum")
        :param epoch_obj_cnt: count of objects in each class up until the current epoch
        :param sign: Analyze either the positive or the negative attributions

        :return:
        output dimensions: B x N x whatever, where B is batch size, and N is the max number of boxes in a frame in this
        batch
        """
        # Get the predictions, compute box vertices, and generate targets
        targets, cared_vertices, cared_locs, fp_targets, fp_vertices, fp_locs = None, None, None, None, None, None
        if self.selection == "tp/fp" or self.selection == "tp/fp_all":
            targets, cared_vertices, cared_locs, fp_targets, fp_vertices, fp_locs = self.xc_preprocess(
                batch_dict, epoch_obj_cnt, epoch_tp_obj_cnt, epoch_fp_obj_cnt, cur_it)
        else:
            targets, cared_vertices, cared_locs = self.xc_preprocess(
                batch_dict, epoch_obj_cnt, epoch_tp_obj_cnt, epoch_fp_obj_cnt, cur_it)
        if self.debug:
            print("len(targets): {} len(cared_vertices): {} len(cared_locs): {}".format(
                len(targets), len(cared_vertices), len(cared_locs)))

        # Compute gradients, XC, and pap
        total_XC_lst, total_far_attr_lst, total_pap_lst, total_fp_XC_lst, total_fp_far_attr_lst, total_fp_pap_lst = \
            [], [], [], [], [], []
        total_XC, total_far_attr, total_pap, total_fp_XC, total_fp_far_attr, total_fp_pap = \
            None, None, None, None, None, None
        if self.selection == "tp/fp" or self.selection == "tp" or self.selection == "tp/fp_all":
            # get the tp related metrics
            for i in range(self.tp_limit):
                # The i-th target for each frame in the batch
                new_targets = [frame_targets[i] for frame_targets in targets]
                pos_grad, neg_grad = self.get_attr(new_targets)
                class_names, vicinities = [], []
                for target in new_targets:
                    cls_name = self.class_name_list[target[1]]
                    vici = self.vicinity_dict[cls_name]
                    class_names.append(cls_name)
                    vicinities.append(vici)
                # print("cared tp vertices: {}".format(cared_vertices))
                # print("cared tp locs: {}".format(cared_locs))
                # print("type(cared tp vertices): {}".format(type(cared_vertices)))
                # print("type(cared tp locs): {}".format(type(cared_locs)))
                new_cared_vertices = [frame_vertices[i] for frame_vertices in cared_vertices]
                new_cared_locs = [frame_locs[i] for frame_locs in cared_locs]
                if self.debug:
                    print("tp pred_box_id: {}".format(i))
                    print("tp type(new_targets): {}".format(type(new_targets)))  # Should be List
                    print("tp new_targets: {}".format(new_targets))
                    print("tp new_cared_vertices: {}".format(new_cared_vertices))
                    print("tp new_cared_locs: {}".format(new_cared_locs))
                XC, far_attr = self.compute_xc_single(
                    pos_grad, neg_grad, new_cared_vertices, self.dataset_name, sign, self.ignore_thresh,
                    new_cared_locs, vicinities, method)
                pap = self.get_PAP_single(pos_grad, neg_grad, sign)
                total_XC_lst.append(torch.stack(XC))
                total_far_attr_lst.append(torch.stack(far_attr))
                total_pap_lst.append(torch.stack(pap))
            if self.selection == "tp/fp" or self.selection == "tp/fp_all":
                # get the fp related metrics
                for i in range(self.fp_limit):
                    # The i-th target for each frame in the batch
                    new_targets = [frame_targets[i] for frame_targets in fp_targets]
                    pos_grad, neg_grad = self.get_attr(new_targets)
                    class_names, vicinities = [], []
                    for target in new_targets:
                        cls_name = self.class_name_list[target[1]]
                        vici = self.vicinity_dict[cls_name]
                        class_names.append(cls_name)
                        vicinities.append(vici)
                    new_cared_vertices = [frame_vertices[i] for frame_vertices in fp_vertices]
                    new_cared_locs = [frame_locs[i] for frame_locs in fp_locs]
                    if self.debug:
                        print("fp pred_box_id: {}".format(i))
                        print("fp type(new_targets): {}".format(type(new_targets)))  # Should be List
                        print("fp new_targets: {}".format(new_targets))
                        print("fp new_cared_vertices: {}".format(new_cared_vertices))
                        print("fp new_cared_locs: {}".format(new_cared_locs))
                    XC, far_attr = self.compute_xc_single(
                        pos_grad, neg_grad, new_cared_vertices, self.dataset_name, sign, self.ignore_thresh,
                        new_cared_locs, vicinities, method)
                    pap = self.get_PAP_single(pos_grad, neg_grad, sign)
                    total_fp_XC_lst.append(torch.stack(XC))
                    total_fp_far_attr_lst.append(torch.stack(far_attr))
                    total_fp_pap_lst.append(torch.stack(pap))
        else:
            min_num_preds = 10000
            for i in range(self.batch_size):
                # In case some frames doesn't have enough predictions to match number of predictions in other frames
                # of the same batch
                min_num_preds = min(len(targets[i]), min_num_preds)
            if self.debug:
                print("min_num_preds: {}".format(min_num_preds))
            for i in range(min_num_preds):
                # The i-th target for each frame in the batch
                new_targets = [frame_targets[i] for frame_targets in targets]
                pos_grad, neg_grad = self.get_attr(new_targets)
                class_names, vicinities = [], []
                for target in new_targets:
                    cls_name = self.class_name_list[target[1]]
                    vici = self.vicinity_dict[cls_name]
                    class_names.append(cls_name)
                    vicinities.append(vici)
                new_cared_vertices = [frame_vertices[i] for frame_vertices in cared_vertices]
                new_cared_locs = [frame_locs[i] for frame_locs in cared_locs]
                if self.debug:
                    print("pred_box_id: {}".format(i))
                    print("type(new_targets): {}".format(type(new_targets)))  # Should be List
                    print("new_targets: {}".format(new_targets))
                    # print("new_cared_vertices: {}".format(new_cared_vertices))
                    # print("new_cared_locs: {}".format(new_cared_locs))
                XC, far_attr = self.compute_xc_single(
                    pos_grad, neg_grad, new_cared_vertices, self.dataset_name, sign, self.ignore_thresh,
                    new_cared_locs, vicinities, method)
                # note: pap is aggregated, unlike XC and far_attr which are per object
                pap = self.get_PAP_single(pos_grad, neg_grad, sign)
                # print("type(XC[0]): {} XC[0].device: {} XC[0].dtype: {}".format(type(XC[0]), XC[0].device, XC[0].dtype))
                # print("type(pap[0]): {} pap[0].device: {} pap[0].dtype: {}".format(type(pap[0]), pap[0].device, pap[0].dtype))
                total_XC_lst.append(torch.stack(XC))
                total_far_attr_lst.append(torch.stack(far_attr))
                total_pap_lst.append(torch.stack(pap))
        # normalizing by the batch size
        if len(total_XC_lst) == 0: # to account for the case where we don't have any TP in a frame in the tp and tp/fp modes
            total_XC_lst.append(torch.full((1,1), float('nan')).cuda())
            total_far_attr_lst.append(torch.full((1,1), float('nan')).cuda())
            total_pap_lst.append(torch.full((1,1), float('nan')).cuda())
        total_XC = torch.stack(total_XC_lst)
        total_far_attr = torch.stack(total_far_attr_lst)
        total_pap_raw = torch.stack(total_pap_lst)
        nan_tensor = torch.full(total_XC.size(), float('nan')).cuda()
        total_pap_ = torch.where(torch.isnan(total_XC), nan_tensor, total_pap_raw)
        total_pap = torch.where(total_pap_ == 0, nan_tensor, total_pap_)
        print("\ntotal_XC.size(): {}\ntotal_pap.size(): {}".format(total_XC.size(), total_pap.size()))
        print("\nsuccessfully reformatted the XC, far_attr, and pap values from lists to tensors\n")
        if self.selection == "tp/fp" or self.selection == "tp/fp_all":
            total_fp_XC = torch.stack(total_fp_XC_lst)
            total_fp_far_attr = torch.stack(total_fp_far_attr_lst)
            total_fp_pap_raw = torch.stack(total_fp_pap_lst)
            fp_nan_tensor = torch.full(total_fp_XC.size(), float('nan')).cuda()
            total_fp_pap_ = torch.where(torch.isnan(total_fp_XC), fp_nan_tensor, total_fp_pap_raw)
            total_fp_pap = torch.where(total_fp_pap_ == 0, fp_nan_tensor, total_fp_pap_)
            total_XC = total_XC.transpose(0, 1)
            total_far_attr = total_far_attr.transpose(0, 1)
            total_pap = total_pap.transpose(0, 1)
            if self.selection == "tp/fp" or self.selection == "tp/fp_all":
                total_fp_XC = total_fp_XC.transpose(0, 1)
                total_fp_far_attr = total_fp_far_attr.transpose(0, 1)
                total_fp_pap = total_fp_pap.transpose(0, 1)

        # record the cared predictions:
        self.cur_epoch = cur_epoch
        self.record_pred_results(total_XC, total_far_attr, total_pap, total_fp_XC, total_fp_far_attr, total_fp_pap)

        if self.selection == "tp/fp" or self.selection == "tp/fp_all":
            return total_XC, total_far_attr, total_pap, total_fp_XC, total_fp_far_attr, total_fp_pap
        return total_XC, total_far_attr, total_pap

    def compute_PAP(self, batch_dict, epoch_obj_cnt, epoch_tp_obj_cnt, epoch_fp_obj_cnt, cur_it, cur_epoch, sign="positive"):
        """
        User shall call the function if they wish to not compute XC, just PAP only
        :return:
        """
        # Get the predictions, compute box vertices, and generate targets
        targets, cared_vertices, cared_locs = self.xc_preprocess(
            batch_dict, epoch_obj_cnt, epoch_tp_obj_cnt, epoch_fp_obj_cnt, cur_it)
        if self.debug:
            print("len(targets): {} len(cared_vertices): {} len(cared_locs): {}".format(
                len(targets), len(cared_vertices), len(cared_locs)))

        # record the cared predictions:
        self.cur_epoch = cur_epoch
        self.record_pred_results()

        # Compute gradients, XC, and pap
        total_pap = 0
        min_num_preds = 10000
        for i in range(self.batch_size):
            # In case some frames doesn't have enough predictions to match number of predictions in other frames
            # of the same batch
            min_num_preds = min(len(targets[i]), min_num_preds)
        if self.debug:
            print("min_num_preds: {}".format(min_num_preds))
        for i in range(min_num_preds):
            # The i-th target for each frame in the batch
            new_targets = [frame_targets[i] for frame_targets in targets]
            pos_grad, neg_grad = self.get_attr(new_targets)
            if self.debug:
                print("pred_box_id: {}".format(i))
            pap = self.get_PAP_single(pos_grad, neg_grad, sign)
            total_pap += pap
        # normalizing by the batch size
        total_pap = total_pap / self.batch_size
        return total_pap
