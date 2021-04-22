import os
import copy
import torch
from tensorboardX import SummaryWriter
import time
import timeit
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
from captum.attr import GuidedBackprop
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz

from scipy.spatial.transform import Rotation as R

class AttributionGenerator:
    def __init__(self, model_ckpt, full_model_cfg_file, model_cfg_file, data_set, xai_method, output_path, ig_steps=24,
                 margin=0.2, num_boxes=3, selection='top', ignore_thresh=0.1, double_model=True, debug=False):
        """
        You should have the data loader ready and the specific batch dictionary available before computing any
        attributions.
        Also, you need to have the folder XAI_utils in the `tools` folder as well for this object to work

        :param model_ckpt: directory for the model checkpoint
        :param full_model_cfg_file: directory for the full model config file
        :param model_cfg_file: directory for the model config file (model to be explained)
        :param data_set:
        :param xai_method: a string indicating which explanation technique to use
        :param output_path: directory for storing relevant outputs
        :param ig_steps: number of intermediate steps to use for IG
        :param boxes: number of boxes to explain per frame
        :param margin: margin appended to box boundaries for XC computation
        :param num_boxes: number of predicted boxes for which we are generating explanations for
        :param selection: whether to compute XC for the most confident ('top')  or least confident ('bottom') boxes
        :param ignore_thresh: used in XC calculation to filter out small attributions
        :param double_model: indicates whether a secondary model (in the case of PointPillars, that would be the full
         model) is required as well
        :param debug: whether to show debug messages
        """

        # Load model configurations
        full_cfg = copy.deepcopy(cfg)
        cfg_from_yaml_file(model_cfg_file, cfg)
        cfg.TAG = Path(model_cfg_file).stem
        cfg.EXP_GROUP_PATH = '/'.join(model_cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
        self.cfg = cfg
        self.dataset_name = cfg.DATA_CONFIG.DATASET
        self.class_name_list = cfg.CLASS_NAMES
        self.double_model = double_model

        if self.double_model:
            cfg_from_yaml_file(full_model_cfg_file, full_cfg)
            self.full_cfg = full_cfg

        self.label_dict = None  # maps class name to label
        self.vicinity_dict = None  # Search vicinity for the generate_box_mask function
        if self.dataset_name == 'KittiDataset':
            self.label_dict = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
            self.vicinity_dict = {"Car": 20, "Pedestrian": 5, "Cyclist": 9}
        elif self.dataset_name == 'CadcDataset':
            self.label_dict = {"Car": 0, "Pedestrian": 1, "Truck": 2}
            self.vicinity_dict = {"Car": 13, "Pedestrian": 3, "Truck": 19}
        else:
            raise NotImplementedError

        # Create output directories, not very useful, just to be compatible with the load_params_from_file
        # method defined in pcdet/models/detectors/detector3d_template.py, which requires a logger
        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / 'default'
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_output_dir = output_dir / 'eval'
        eval_all = False
        eval_tag = 'default'

        if not eval_all:
            num_list = re.findall(r'\d+', model_ckpt) if model_ckpt is not None else []
            epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
            eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
        else:
            eval_output_dir = eval_output_dir / 'eval_all_default'

        if eval_tag is not None:
            eval_output_dir = eval_output_dir / eval_tag

        eval_output_dir.mkdir(parents=True, exist_ok=True)

        # Create logger, not very useful, just to be compatible with PCDet's structures
        log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

        # Build the model
        self.full_model = build_network(model_cfg=full_cfg.MODEL, num_class=len(full_cfg.CLASS_NAMES), dataset=data_set)
        self.full_model.load_params_from_file(filename=model_ckpt, logger=logger, to_cpu=False)
        self.full_model.cuda()
        self.full_model.eval()
        self.model = self.full_model.forward_model2D
        # self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=data_set)
        # self.model.load_params_from_file(filename=model_ckpt, logger=logger, to_cpu=False)
        # self.model.cuda()
        # self.model.eval()
        # if self.double_model:
        #     self.full_model = build_network(model_cfg=full_cfg.MODEL, num_class=len(full_cfg.CLASS_NAMES), dataset=data_set)
        #     self.full_model.load_params_from_file(filename=model_ckpt, logger=logger, to_cpu=False)
        #     self.full_model.cuda()
        #     self.full_model.eval()

        # Other initialization stuff
        self.debug = debug
        self.margin = margin
        self.output_path = output_path
        self.ig_steps = ig_steps
        self.num_boxes = num_boxes
        self.selection = selection
        self.ignore_thresh = ignore_thresh
        self.batch_dict = None
        self.pred_boxes = None
        self.pred_labels = None
        self.pred_scores = None
        self.pred_vertices = None # vertices of predicted boxes with a specified margin
        self.pred_loc = None
        self.selected_anchors = None
        self.explainer = None
        self.batch_size = 1
        self.time_study = False
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

        T_Lidar_Cuboid = np.eye(4)  # identify matrix
        T_Lidar_Cuboid[0:3, 0:3] = R.from_euler('z', yaw, degrees=False).as_dcm()  # rotate the identity matrix
        T_Lidar_Cuboid[0][3] = x  # center of the tracklet, from cuboid to lidar
        T_Lidar_Cuboid[1][3] = y
        T_Lidar_Cuboid[2][3] = z

        radius = 3

        # the top view of the tracklet in the "cuboid frame". The cuboid frame is a cuboid with origin (0,0,0)
        # we are making a cuboid that has the dimensions of the tracklet but is located at the origin
        front_right_top = np.array(
            [[1, 0, 0, l / 2],
             [0, 1, 0, w / 2],
             [0, 0, 1, h / 2],
             [0, 0, 0, 1]])

        front_left_top = np.array(
            [[1, 0, 0, l / 2],
             [0, 1, 0, -w / 2],
             [0, 0, 1, h / 2],
             [0, 0, 0, 1]])

        back_right_top = np.array(
            [[1, 0, 0, -l / 2],
             [0, 1, 0, w / 2],
             [0, 0, 1, h / 2],
             [0, 0, 0, 1]])

        back_left_top = np.array(
            [[1, 0, 0, -l / 2],
             [0, 1, 0, -w / 2],
             [0, 0, 1, h / 2],
             [0, 0, 0, 1]])

        # Project to lidar
        f_r_t = np.matmul(T_Lidar_Cuboid, front_right_top)
        f_l_t = np.matmul(T_Lidar_Cuboid, front_left_top)
        b_r_t = np.matmul(T_Lidar_Cuboid, back_right_top)
        b_l_t = np.matmul(T_Lidar_Cuboid, back_left_top)

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
        poly = np.array([[-1 * y1, x1], [-1 * y2, x2], [-1 * y4, x4], [-1 * y3, x3]])
        return poly

    def get_preds_single_frame(self, pred_dicts):
        pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
        for i in range(len(pred_boxes)):
            pred_boxes[i][6] += np.pi / 2
        self.pred_boxes = pred_boxes
        self.pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy() - 1
        self.pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()
        self.selected_anchors = self.batch_dict['anchor_selections'][0]

    def get_preds(self):
        # # Debug message
        # print("keys in self.batch_dict:")
        # for key in self.batch_dict:
        #     print('key: {}'.format(key))

        pred_dicts = self.batch_dict['pred_dicts']
        self.batch_size = self.batch_dict['batch_size']
        if self.batch_size == 1:
            self.get_preds_single_frame(pred_dicts)
        else:
            pred_boxes, pred_labels, pred_scores, selected_anchors = [], [], [], []
            for i in range(self.batch_size):
                frame_pred_boxes = pred_dicts[i]['pred_boxes'].cpu().numpy()
                for j in range(len(frame_pred_boxes)):
                    frame_pred_boxes[j][6] += np.pi / 2
                pred_boxes.append(frame_pred_boxes)
                pred_labels.append(pred_dicts[i]['pred_labels'].cpu().numpy() - 1)
                pred_scores.append(pred_dicts[i]['pred_scores'].cpu().numpy())
                selected_anchors.append(self.batch_dict['anchor_selections'][i])
            self.pred_boxes = pred_boxes
            self.pred_labels = pred_labels
            self.pred_scores = pred_scores
            self.selected_anchors = selected_anchors
            # # debug message
            # print("len(selected_anchors): {}".format(len(selected_anchors)))
            # print("len(selected_anchors[0]): {}".format(len(selected_anchors[0])))

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

    def get_attr(self, target):
        """
        Unless the batch_size is 1, there will be repeated work done in a batch since the same target is being applied
        to all samples in the same batch
        """
        start1 = timeit.default_timer()
        PseudoImage2D = self.batch_dict['spatial_features']
        batch_grad = self.explainer.attribute(PseudoImage2D, baselines=PseudoImage2D * 0, target=target,
                                              additional_forward_args=self.batch_dict, n_steps=self.ig_steps,
                                              internal_batch_size=self.batch_dict['batch_size'])
        end1 = timeit.default_timer()
        attr_time = end1 - start1

        start2 = timeit.default_timer()
        if self.batch_size == 1:
            grad = np.transpose(batch_grad[0].squeeze().cpu().detach().numpy(), (1, 2, 0))
            pos_grad = np.sum((grad > 0) * grad, axis=2)
            neg_grad = np.sum(-1 * (grad < 0) * grad, axis=2)
        else:
            grads = [np.transpose(gradients.squeeze().cpu().detach().numpy(), (1, 2, 0)) for gradients in batch_grad]
            pos_grad = [np.sum((grad > 0) * grad, axis=2) for grad in grads]
            neg_grad = [np.sum(-1 * (grad < 0) * grad, axis=2) for grad in grads]
        end2 = timeit.default_timer()
        aggr_time = end2 - start2
        if self.time_study:
            return pos_grad, neg_grad, attr_time, aggr_time
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
        if self.batch_size == 1:
            self.compute_pred_box_vertices_single_frame()
        else:
            batch_expanded_preds = []
            batch_pred_loc = []
            # Generate the vertices
            for i in range(self.batch_size):
                expanded_preds = []
                pred_loc = []
                for cuboid in self.pred_boxes[i]:
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
                else:
                    raise NotImplementedError
                side_range = [-s1, s2]
                fwd_range = [-f1, f2]
                offset = np.array([[-side_range[0], -fwd_range[0]]] * 4)
                expanded_preds = [poly + offset for poly in expanded_preds]
                batch_expanded_preds.append(expanded_preds)
                batch_pred_loc.append(pred_loc)
            self.pred_vertices = batch_expanded_preds
            self.pred_loc = batch_pred_loc

    def generate_targets_single_frame(self):
        target_list = []
        cared_labels = None
        cared_box_vertices = None
        cared_loc = None
        if self.num_boxes >= len(self.pred_labels):
            cared_labels = self.pred_labels
            cared_box_vertices = self.pred_vertices
            cared_loc = self.pred_loc
        elif self.selection == "top":
            cared_labels = self.pred_labels[:self.num_boxes]
            cared_box_vertices = self.pred_vertices[:self.num_boxes]
            cared_loc = self.pred_loc[:self.num_boxes]
        elif self.selection == "bottom":
            cared_labels = self.pred_labels[-1 * self.num_boxes:]
            cared_box_vertices = self.pred_vertices[-1 * self.num_boxes:]
            cared_loc = self.pred_loc[-1 * self.num_boxes:]

        # Generate targets
        for ind, label in enumerate(cared_labels):
            target_list.append((self.selected_anchors[ind], label))
        return target_list, cared_box_vertices, cared_loc

    def generate_targets(self):
        """
        Generates targets for explanation given the batch
        Note that the class label target for each box corresponds to the top confidence predicted class
        :return:
        """
        # Get pred labels
        batch_target_list = []
        batch_cared_labels = []
        batch_cared_box_vertices = []
        batch_cared_loc = []
        if self.batch_size == 1:
            batch_target_list, batch_cared_box_vertices, batch_cared_loc = self.generate_targets_single_frame()
        else:
            for i in range(self.batch_size):
                target_list = []
                cared_labels = None
                cared_box_vertices = None
                cared_loc = None
                if self.num_boxes >= len(self.pred_labels[i]):
                    cared_labels = self.pred_labels[i]
                    cared_box_vertices = self.pred_vertices[i]
                    cared_loc = self.pred_loc[i]
                elif self.selection == "top":
                    cared_labels = self.pred_labels[i][:self.num_boxes]
                    cared_box_vertices = self.pred_vertices[i][:self.num_boxes]
                    cared_loc = self.pred_loc[i][:self.num_boxes]
                elif self.selection == "bottom":
                    cared_labels = self.pred_labels[i][-1*self.num_boxes:]
                    cared_box_vertices = self.pred_vertices[i][-1*self.num_boxes:]
                    cared_loc = self.pred_loc[i][-1*self.num_boxes:]

                # Generate targets
                for ind, label in enumerate(cared_labels):
                    target_list.append((self.selected_anchors[i][ind], label))
                batch_target_list.append(target_list)
                batch_cared_box_vertices.append(cared_box_vertices)
                batch_cared_loc.append(cared_loc)
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
        XC, attr_in_box, far_attr, total_attr = None, None, None, None
        if self.batch_size == 1:
            if method == "cnt":
                XC, attr_in_box, far_attr, total_attr = get_cnt_XQ_analytics_fast(
                    pos_grad, neg_grad, box_vertices, dataset_name, sign, ignore_thresh, box_loc, vicinity)
            elif method == "sum":
                XC, attr_in_box, far_attr, total_attr = get_sum_XQ_analytics_fast(
                    pos_grad, neg_grad, box_vertices, dataset_name, sign, ignore_thresh, box_loc, vicinity)
        else:
            XC_lst, far_attr_lst = [], []
            for i in range(self.batch_size):
                if method == "cnt":
                    XC_, attr_in_box_, far_attr_, total_attr_ = get_cnt_XQ_analytics_fast(
                        pos_grad[i], neg_grad[i], box_vertices[i], dataset_name, sign, ignore_thresh, box_loc[i],
                        vicinity[i])
                elif method == "sum":
                    XC_, attr_in_box_, far_attr_, total_attr_ = get_sum_XQ_analytics_fast(
                        pos_grad[i], neg_grad[i], box_vertices[i], dataset_name, sign, ignore_thresh, box_loc[i],
                        vicinity[i])
                XC_lst.append(XC_)
                far_attr_lst.append(far_attr_)
                # debug message
                if self.debug:
                    print("frame id in batch: {} XC: {}".format(i, XC_))
            XC = np.asarray(XC_lst)
            far_attr = np.asarray(far_attr_lst)
        return XC, far_attr

    def get_PAP_single(self, pos_grad, neg_grad, sign):
        grad = None
        if sign == 'positive':
            grad = pos_grad
        elif sign == 'negative':
            grad = neg_grad
        pap_loss = 0.0
        if self.batch_size == 1:
            diff_1 = grad[1:, :] - grad[:-1, :]
            diff_2 = grad[:, 1:] - grad[:, :-1]
            pap_loss = np.sum(np.abs(diff_1)) + np.sum(np.abs(diff_2))
        else:
            for i in range(self.batch_size):
                diff_1 = grad[i][1:, :] - grad[i][:-1, :]
                diff_2 = grad[i][:, 1:] - grad[i][:, :-1]
                pap_loss += np.sum(np.abs(diff_1)) + np.sum(np.abs(diff_2))
        return pap_loss

    def xc_preprocess(self, batch_dict):
        # Get the predictions, compute box vertices, and generate targets
        self.batch_dict = batch_dict
        load_data_to_gpu(self.batch_dict)
        dummy_tensor = 0
        if self.double_model:
            with torch.no_grad():
                # run the model once to populate the batch dict
                anchor_scores = self.full_model(dummy_tensor, self.batch_dict)
        self.get_preds()
        self.compute_pred_box_vertices()
        return self.generate_targets()

    def compute_xc(self, batch_dict, method="cnt", sign="positive"):
        """
        This is the ONLY function that the user should call

        :param batch_dict: The input batch for which we are generating explanations
        :param method: Either by counting ("cnt") or by summing ("sum")
        :param sign: Analyze either the positive or the negative attributions
        :return:
        """
        # Get the predictions, compute box vertices, and generate targets
        start_time = timeit.default_timer()
        targets, cared_vertices, cared_locs = self.xc_preprocess(batch_dict)
        after_preprocess = timeit.default_timer()
        xc_preprocess_time = after_preprocess - start_time
        if self.debug:
            print("len(targets): {} len(cared_vertices): {} len(cared_locs): {}".format(
                len(targets), len(cared_vertices), len(cared_locs)))
        # Compute gradients, XC, and pap
        total_XC_lst, total_far_attr_lst, total_pap_lst = [], [], []
        total_XC, total_far_attr, total_pap = None, None, None

        attr_time, aggr_time, xc_time, pap_time, misc_time = 0.0, 0.0, 0.0, 0.0, 0.0
        if self.batch_size == 1:
            for i in range(len(targets)):
                pos_grad, neg_grad = self.get_attr(targets[i])
                class_name = self.class_name_list[targets[i][1]]
                vicinity = self.vicinity_dict[class_name]
                XC, far_attr = self.compute_xc_single(
                    pos_grad, neg_grad, cared_vertices[i], self.dataset_name, sign, self.ignore_thresh,
                    cared_locs[i], vicinity, method)
                pap = self.get_PAP_single(pos_grad, neg_grad, sign)
                total_XC_lst.append(XC)
                total_far_attr_lst.append(far_attr)
                total_pap_lst.append(pap)
                # debug message
                if self.debug:
                    print("\nPred_box id: {} XC: {}".format(targets[i][0], XC))
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
                if self.debug:
                    print("type(new_targets): {}".format(type(new_targets)))  # Should be List

                if self.time_study:
                    pos_grad, neg_grad, attr_time_i, aggr_time_i = self.get_attr(new_targets)
                    attr_time += attr_time_i
                    aggr_time += aggr_time_i
                else:
                    pos_grad, neg_grad = self.get_attr(new_targets)

                class_names, vicinities = [], []
                for target in new_targets:
                    cls_name = self.class_name_list[target[1]]
                    vici = self.vicinity_dict[cls_name]
                    class_names.append(cls_name)
                    vicinities.append(vici)
                if self.debug:
                    print("pred_box_id: {}".format(i))
                new_cared_vertices = [frame_vertices[i] for frame_vertices in cared_vertices]
                new_cared_locs = [frame_locs[i] for frame_locs in cared_locs]

                xc_start = timeit.default_timer()
                XC, far_attr = self.compute_xc_single(
                    pos_grad, neg_grad, new_cared_vertices, self.dataset_name, sign, self.ignore_thresh,
                    new_cared_locs, vicinities, method)
                xc_end = timeit.default_timer()
                xc_time += xc_end - xc_start

                pap_start = timeit.default_timer()
                pap = self.get_PAP_single(pos_grad, neg_grad, sign)
                pap_end = timeit.default_timer()
                pap_time += pap_end - pap_start

                total_XC_lst.append(XC)
                total_far_attr_lst.append(far_attr)
                total_pap_lst.append(pap)
        total_XC = np.asarray(total_XC_lst)
        total_far_attr = np.asarray(total_far_attr_lst)
        total_pap = np.asarray(total_pap_lst)
        if self.batch_size > 1:
            total_XC = np.transpose(total_XC)
            total_far_attr = np.transpose(total_far_attr)
            total_pap = np.transpose(total_pap)
        end_time = timeit.default_timer()
        total_time = end_time - start_time
        misc_time = total_time - attr_time - xc_time - pap_time - xc_preprocess_time - aggr_time
        if self.time_study:
            print("XC preprocessing: {}".format(xc_preprocess_time))
            print("attr generation: {} \nattr aggregation: {} \nxc: {} \npap: {} \nmiscellaneous: {}".format(
                attr_time, aggr_time, xc_time, pap_time, misc_time))
        return total_XC, total_far_attr, total_pap

    def compute_PAP(self, batch_dict, sign="positive"):
        """
        User shall call the function if they wish to not compute XC, just PAP only
        :return:
        """
        # Get the predictions, compute box vertices, and generate targets
        targets, cared_vertices, cared_locs = self.xc_preprocess(batch_dict)
        if self.debug:
            print("len(targets): {} len(cared_vertices): {} len(cared_locs): {}".format(
                len(targets), len(cared_vertices), len(cared_locs)))

        # Compute gradients, XC, and pap
        total_pap = 0
        if self.batch_size == 1:
            for i in range(len(targets)):
                pos_grad, neg_grad = self.get_attr(targets[i])
                pap = self.get_PAP_single(pos_grad, neg_grad, sign)
                total_pap += pap
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
                if self.debug:
                    print("pred_box_id: {}".format(i))
                pap = self.get_PAP_single(pos_grad, neg_grad, sign)
                total_pap += pap
            # normalizing by the batch size
            total_pap = total_pap / self.batch_size
        return total_pap
