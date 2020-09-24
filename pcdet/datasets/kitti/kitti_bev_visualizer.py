import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.patches as patches
import os
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from math import pi, cos, sin

import yaml
from pathlib import Path
from easydict import EasyDict

import time
import datetime


class KITTI_BEV:
    def __init__(self, dataset, repo_dir='/root/pcdet', scale_to_pseudoimg=False,
                 class_name=['Car', 'Pedestrian', 'Cyclist'],
                 result_path='output/kitti_models/pointpillar/default/eval/epoch_2/val/default/result.pkl',
                 output_path='data/kitti/', background='black', width_pix=432 * 5, height_pix=496 * 5, cmap='jet',
                 dpi_factor=20.0):
        self.repo_dir = repo_dir
        self.scale_to_pseudoimg = scale_to_pseudoimg
        self.class_name = class_name
        self.result_path = os.path.join(repo_dir, result_path)
        self.dataset = dataset
        self.results = np.load(self.result_path, allow_pickle=True)
        self.background = background
        self.width_pix = width_pix
        self.height_pix = height_pix
        self.cmap = cmap
        self.dpi_factor = dpi_factor
        self.pred_poly = []  # store the predicted polygons
        now = datetime.datetime.now()
        dt_string = now.strftime("%b_%d_%Y_%H_%M_%S")
        output_path = output_path + '{}_bev_pred'.format(dt_string)
        self.output_path = os.path.join(repo_dir, output_path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def load_gt_anns(self, result_frame, unique_id):
        kitti_annotations = self.dataset.get_label(result_frame[unique_id])
        calib = self.dataset.get_calib(result_frame[unique_id])
        # Convert to CADC
        annotations = {
            'cuboids': []
        }
        for obj in kitti_annotations:
            loc_lidar = calib.rect_to_lidar(obj.loc.reshape(1, 3))[0]
            cuboid = {
                'label': obj.cls_type,
                'position': {
                    'x': loc_lidar[0],
                    'y': loc_lidar[1],
                    'z': loc_lidar[2]
                },
                'dimensions': {
                    'x': obj.l,
                    'y': obj.w,
                    'z': obj.h
                },
                'yaw': -obj.ry
            }
            annotations['cuboids'].append(cuboid)
            # exit()

        # Filter GT boxes
        gt = []
        for cuboid in annotations['cuboids']:
            if cuboid['label'] in self.class_name:
                gt.append(cuboid)
        annotations['cuboids'] = gt

        return annotations

    def get_preds(self, result_frame):
        # filter predictions
        box_preds = []
        # box_var_preds = []
        print("len(result_frame['boxes_lidar']): {}".format(len(result_frame['boxes_lidar'])))
        for i in range(len(result_frame['boxes_lidar'])):
            # Rotate prediction
            result_frame['boxes_lidar'][i][6] += np.pi / 2
            box_preds.append(result_frame['boxes_lidar'][i])
            # # Exponentiate to get variances
            # box_var_preds.append(np.exp(result_frame['boxes_lidar_log_var'][i]))
        return box_preds

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

        # to use for the plot
        x_img_tracklet = -1 * y  # in the image to plot, the negative lidar y axis is the img x axis
        y_img_tracklet = x  # the lidar x axis is the img y axis

        poly = np.array([[-1 * y1, x1], [-1 * y2, x2], [-1 * y4, x4], [-1 * y3, x3]])
        return poly

    def draw_bev(self, lidar, annotations, predictions, output_path, s1=39.68, s2=39.68, f1=0.0, f2=69.12):
        '''
        :param lidar : Lidar data as an np.array
        :param annotations: annotations json for the desired frame
        :param predictions: [[x,y,z,w,l,h,yaw]...] List of lidar bounding boxes
        :param output_path: String
        :return:
        '''
        # limit the viewing range
        side_range = [-s1, s2]
        fwd_range = [-f1, f2]

        lidar_x = lidar[:, 0]
        lidar_y = lidar[:, 1]
        lidar_z = lidar[:, 2]

        lidar_x_trunc = []
        lidar_y_trunc = []
        lidar_z_trunc = []

        for i in range(len(lidar_x)):
            if lidar_x[i] > fwd_range[0] and lidar_x[i] < fwd_range[1]:  # get the lidar coordinates
                if lidar_y[i] > side_range[0] and lidar_y[i] < side_range[1]:
                    lidar_x_trunc.append(lidar_x[i])
                    lidar_y_trunc.append(lidar_y[i])
                    lidar_z_trunc.append(lidar_z[i])

        # to use for the plot
        x_img = [i * -1 for i in lidar_y_trunc]  # in the image plot, the negative lidar y axis is the img x axis
        y_img = lidar_x_trunc  # the lidar x axis is the img y axis
        pixel_values = lidar_z_trunc

        # shift values such that 0,0 is the minimum
        x_img = [i - side_range[0] for i in x_img]
        y_img = [i - fwd_range[0] for i in y_img]
        gt_poly = []
        pred_poly = []

        for cuboid in annotations['cuboids']:
            x = cuboid['position']['x']
            y = cuboid['position']['y']
            z = cuboid['position']['z']
            w = cuboid['dimensions']['x']
            l = cuboid['dimensions']['y']
            h = cuboid['dimensions']['z']
            yaw = cuboid['yaw']

            if (x < fwd_range[0] or x > fwd_range[1] or y < side_range[0] or y > side_range[1]):
                continue  # out of bounds

            gt_poly.append(self.cuboid_to_bev(x, y, z, w, l, h, yaw))

        for cuboid in predictions:
            x, y, z, w, l, h, yaw = cuboid

            if (x < fwd_range[0] or x > fwd_range[1] or y < side_range[0] or y > side_range[1]):
                continue  # out of bounds

            pred_poly.append(self.cuboid_to_bev(x, y, z, w, l, h, yaw))

        # Transform all polygons so 0,0 is the minimum
        offset = np.array([[-side_range[0], -fwd_range[0]]] * 4)

        gt_poly = [poly + offset for poly in gt_poly]
        pred_poly = [poly + offset for poly in pred_poly]
        self.pred_poly = pred_poly
        # print('self.pred_poly: {}'.format(self.pred_poly))
        # PLOT THE IMAGE
        cmap = self.cmap  # Color map to use # 'jet' originally
        x_max = side_range[1] - side_range[0]
        y_max = fwd_range[1] - fwd_range[0]
        if self.scale_to_pseudoimg:
            self.width_pix = 432
            self.height_pix = 496
        dpi = self.width_pix / self.dpi_factor  # Image resolution, dots per inch
        fig, ax = plt.subplots(figsize=(self.height_pix / dpi, self.width_pix / dpi), dpi=dpi)

        for poly in gt_poly:  # plot the tracklets
            polys = patches.Polygon(poly, closed=True, fill=False, edgecolor='g', linewidth=1)
            ax.add_patch(polys)
        for poly in pred_poly:
            polys = patches.Polygon(poly, closed=True, fill=False, edgecolor='r', linewidth=1)
            ax.add_patch(polys)

        ax.scatter(x_img, y_img, s=1, c=pixel_values, alpha=1.0, cmap=cmap)  # Plot Lidar points
        if self.background == 'black':
            ax.set_facecolor((0, 0, 0))
        elif self.background == 'white':
            ax.set_facecolor((1, 1, 1))
        ax.axis('scaled')  # {equal, scaled}
        ax.xaxis.set_visible(False)  # Do not draw axis tick marks
        ax.yaxis.set_visible(False)  # Do not draw axis tick marks
        plt.xlim([0, x_max])
        plt.ylim([0, y_max])
        size = fig.get_size_inches() * fig.dpi
        # print('bev image size: {}'.format(size))
        plt.tight_layout(pad=0)  # must have this line to ensure that image margin is zero
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close('all')
        return fig

    def get_bev_image(self, frame_idx):
        '''

        :param frame_idx:
        :return: bev_fig--Matplotlib.figure.Figure object
                 bev_fig_array--np.array containing data for this figure
        '''
        result_frame = self.results[frame_idx]

        # TODO double check, this is either sample_idx or frame_id depending on
        # code in generate_prediction_dicts() in cadc_dataset.py
        unique_id = 'frame_id'
        if len(result_frame[unique_id]) == 0:
            print('frame has no id')
            return None

        lidar_data = self.dataset.get_lidar(result_frame[unique_id])
        annotations = self.load_gt_anns(result_frame, unique_id)
        box_preds = self.get_preds(result_frame)
        scores = result_frame['score']

        print("Processing Sample: %s" % result_frame[unique_id])

        bev_fig = self.draw_bev(lidar_data, annotations, box_preds,
                                os.path.join(self.output_path, "%s.png" % result_frame[unique_id]))

        bev_fig.canvas.draw()
        bev_fig_data = np.fromstring(bev_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        bev_fig_array = bev_fig_data.reshape(bev_fig.canvas.get_width_height()[::-1] + (3,))
        return bev_fig, bev_fig_array
        # return bev_fig, bev_fig_data
