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
from captum.attr import GuidedBackprop
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz


class AttributionGenerator:
    def __init__(self, model_ckpt, model_cfg_file, data_set, xai_method, output_path, ig_steps=24, boxes=5, margin=0.2):
        """
        You should have the data loader ready and the specific batch dictionary available before computing any
        attributions.
        Also, you need to have the folder XAI_utils in the `tools` folder as well for this object to work

        :param model_ckpt: directory for the model checkpoint
        :param model_cfg_file: directory for the model config file
        :param data_set:
        :param xai_method: a string indicating which explanation technique to use
        :param output_path: directory for storing relevant outputs
        :param ig_steps: number of intermediate steps to use for IG
        :param boxes: number of boxes to explain per frame
        :param margin: margin appended to box boundaries for XC computation
        """

        # Load model configurations
        cfg_from_yaml_file(model_cfg_file, cfg)
        self.cfg = cfg
        self.dataset_name = cfg.DATA_CONFIG.DATASET

        # Create output directories, not very useful, just to be compatible with the load_params_from_file
        # method defined in pcdet/models/detectors/detector3d_template.py, which requires a logger
        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / 'default'
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_output_dir = output_dir / 'eval'

        # Create logger, not very useful, just to be compatible with PCDet's structures
        log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

        # Build the model
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=data_set)
        self.model.load_params_from_file(filename=model_ckpt, logger=logger, to_cpu=False)
        self.model.cuda()
        self.model.eval()

        # Other initialization stuff
        self.margin = margin
        self.output_path = output_path
        self.ig_steps = ig_steps
        self.batch_dict = {}
        self.pred_boxes = None
        self.pred_labels = None
        self.pred_scores = None
        self.pred_vertices = None # vertices of predicted boxes with a specified margin
        self.explainer = None
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

    def get_preds(self):
        pred_dicts = self.batch_dict['pred_dicts']
        pred_boxes = pred_dicts[0]['pred_boxes'].cpu().numpy()
        for i in range(len(pred_boxes)):
            pred_boxes[i][6] += np.pi / 2
        self.pred_boxes = pred_boxes
        self.pred_labels = pred_dicts[0]['pred_labels'].cpu().numpy() - 1
        self.pred_scores = pred_dicts[0]['pred_scores'].cpu().numpy()

    def reset(self):
        """
        Call this when you have finished using this object for a specific batch
        """
        self.batch_dict = {}
        self.pred_boxes = None
        self.pred_labels = None
        self.pred_scores = None
        self.pred_vertices = None

    def get_attr(self, batch_dict, target):
        """
        Unless the batch_size is 1, there will be repeated work done in a batch since the same target is being applied
        to all samples in the same batch
        """

        # get the predictions
        self.batch_dict = batch_dict
        self.get_preds()

        # get the attributions
        PseudoImage2D = batch_dict['spatial_features']
        batch_grad = self.explainer.attribute(PseudoImage2D, baselines=PseudoImage2D * 0, target=target,
                                              additional_forward_args=batch_dict, n_steps=self.ig_steps,
                                              internal_batch_size=batch_dict['batch_size'])
        grad = np.transpose(batch_grad[0].squeeze().cpu().detach().numpy(), (1, 2, 0))
        pos_grad = np.sum((grad > 0) * grad, axis=2)
        neg_grad = np.sum(-1 * (grad < 0) * grad, axis=2)
        return pos_grad, neg_grad

    def compute_pred_box_vertices(self):
        """
        Given the predicted boxes in [x, y, z, w, l, h, yaw] format, return the expanded predicted boxes in terms of
        box vertices: np.array([[x1, y1], [x2,y2], ...[x4,y4]])
        """

        # Generate the vertices
        expanded_preds = []
        for cuboid in self.pred_boxes:
            x, y, z, w, l, h, yaw = cuboid
            w_big = w + 2 * self.margin
            l_big = l + 2 * self.margin
            expanded_preds.append(self.cuboid_to_bev(x, y, z, w_big, l_big, h, yaw))

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