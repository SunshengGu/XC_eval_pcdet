import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.patches as patches
import os
from pcdet.datasets.cadc.cadc_dataset import CadcDataset

import yaml
from pathlib import Path
from easydict import EasyDict

import time
import datetime


class CADC_BEV:
    def __init__(self, dataset, repo_dir='/root/pcdet', scale_to_pseudoimg=False, class_name=['Car', 'Pedestrian', 'Cyclist'],
                result_path='output/cadc_models/pointpillar/default/eval/epoch_4/val/default/result.pkl',
                output_path = 'data/cadc/', background='black', scale=5, cmap='jet', dpi_factor=20.0, margin=0.0):
        self.repo_dir = repo_dir
        self.scale_to_pseudoimg = scale_to_pseudoimg
        self.class_name = class_name
        self.result_path = os.path.join(repo_dir, result_path)
        self.dataset = dataset
        self.results = np.load(self.result_path, allow_pickle=True)
        self.background = background
        self.width_pix = 400 * scale
        self.cmap = cmap
        self.dpi_factor = dpi_factor
        now = datetime.datetime.now()
        dt_string = now.strftime("%b_%d_%Y_%H_%M_%S")
        output_path = output_path + '{}_bev_pred'.format(dt_string)
        self.output_path = os.path.join(repo_dir, output_path)
        self.pred_poly = [] # store the predicted polygons
        self.pred_poly_expand = []  # store the expanded predicted polygons
        self.pred_loc = []
        self.gt_poly = []   # store the ground truth polygons
        self.gt_loc = []    # store the gt box locations
        self.margin = margin
        self.lidar_data = None
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

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

    def draw_bev(self, lidar, annotations, predictions, output_path, s1=50, s2=50, f1=50, f2=50):
        '''
        :param lidar : Lidar data as an np.array
        :param annotations: annotations json for the desired frame
        :param predictions: [[x,y,z,w,l,h,yaw]...] List of lidar bounding boxes
        :param output_path: String
        :return:
        '''
        # limit the viewing range
        side_range = [-s1, s2]  # 15 meters from either side of the car
        fwd_range = [-f1, f2]  # 15 m infront of the car

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
        pred_poly_expand = []
        gt_loc = []
        pred_loc = []

        for cuboid in annotations['cuboids']:
            x = cuboid['position']['x']
            y = cuboid['position']['y']
            z = cuboid['position']['z']
            w = cuboid['dimensions']['x']
            l = cuboid['dimensions']['y']
            h = cuboid['dimensions']['z']
            yaw = cuboid['yaw']

            # if (x < fwd_range[0] or x > fwd_range[1] or y < side_range[0] or y > side_range[1]):
            #     continue  # out of bounds
            gt_loc.append(np.array([x, y]))
            gt_poly.append(self.cuboid_to_bev(x, y, z, w, l, h, yaw))
        # print('in visualization script: len(predictions): {}'.format(len(predictions)))
        for cuboid in predictions:
            x, y, z, w, l, h, yaw = cuboid
            if self.margin != 0.0:
                w_big = w + 2 * self.margin
                l_big = l + 2 * self.margin
                pred_poly_expand.append(self.cuboid_to_bev(x, y, z, w_big, l_big, h, yaw))

            # if (x < fwd_range[0] or x > fwd_range[1] or y < side_range[0] or y > side_range[1]):
            #     continue  # out of bounds

            pred_poly.append(self.cuboid_to_bev(x, y, z, w, l, h, yaw))
            pred_loc.append(np.array([x, y]))

        # Transform all polygons so 0,0 is the minimum
        offset = np.array([[-side_range[0], -fwd_range[0]]] * 4)

        gt_poly = [poly + offset for poly in gt_poly]
        pred_poly = [poly + offset for poly in pred_poly]
        pred_poly_expand = [poly + offset for poly in pred_poly_expand]
        gt_loc = [loc + offset[0] for loc in gt_loc]
        self.pred_poly = pred_poly
        if self.margin==0.0:
            self.pred_poly_expand = pred_poly
        else:
            self.pred_poly_expand = pred_poly_expand
        self.gt_poly = gt_poly
        self.gt_loc = gt_loc
        self.pred_loc = pred_loc

        # print('in visualization script: len(self.pred_poly): {}'.format(len(self.pred_poly)))
        # PLOT THE IMAGE
        cmap = self.cmap  # Color map to use
        x_max = side_range[1] - side_range[0]
        y_max = fwd_range[1] - fwd_range[0]
        img_size_pix = self.width_pix  # image size in num of pixels
        if self.scale_to_pseudoimg:
            img_size_pix = 400
        dpi = img_size_pix / self.dpi_factor  # Image resolution, dots per inch
        fig, ax = plt.subplots(figsize=(img_size_pix / dpi, img_size_pix / dpi), dpi=dpi)

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
        plt.tight_layout(pad=0) # must have this line to ensure that image margin is zero
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.0)
        plt.close('all')
        return fig

    def get_bev_image(self, frame_idx):
        """

        :param frame_idx:
        :return: bev_fig--Matplotlib.figure.Figure object
                 bev_fig_array--np.array containing data for this figure
        """
        result_frame = self.results[frame_idx]

        # TODO double check, this is either sample_idx or frame_id depending on
        # code in generate_prediction_dicts() in cadc_dataset.py
        unique_id = 'frame_id'
        if len(result_frame[unique_id]) == 0:
            print('frame has no id')
            return None

        # TODO uncomment this if the unique id is modified to only be a number
        # sample_idx = result_frame[unique_id][0]
        # date, run, frame = str(sample_idx[0]), str(sample_idx[1]).zfill(4), str(sample_idx[2]).zfill(10)
        # date = date[0:4] + '_' + date[4:6] + '_' + date[6:]

        # For now since we send over three variables we can do it this way
        date, run, frame = result_frame[unique_id]

        lidar_data = self.dataset.get_lidar([date, run, frame])
        self.lidar_data = lidar_data
        annotations = self.dataset.get_label([date, run, frame])[int(frame)]

        point_count_threshold, distance_threshold, score_threshold = self.dataset.get_threshold()
        # print('point_count_threshold: {}'.format(point_count_threshold))
        # print('distance_threshold: {}'.format(distance_threshold))
        # print('score_threshold: {}'.format(score_threshold))
        # filter gt
        gt = []
        for cuboid in annotations['cuboids']:
            x, y, z = cuboid['position']['x'], cuboid['position']['y'], cuboid['position']['z']
            distance = np.sqrt(np.square(x) + np.square(y) + np.square(z))
            if cuboid['label'] in self.class_name and \
                    (cuboid['label'] != 'Truck' or cuboid['attributes']['truck_type'] == 'Pickup_Truck') and \
                    cuboid['points_count'] >= point_count_threshold[cuboid['label']] and \
                    distance < distance_threshold:
                gt.append(cuboid)
        annotations['cuboids'] = gt

        # filter prediction
        predictions = result_frame['boxes_lidar']
        # print("\nlen(result_frame['boxes_lidar']) in visualization script: {}".format(len(result_frame['boxes_lidar'])))
        # for i in range(len(result_frame['boxes_lidar'])):
        #     x, y, z = result_frame['location'][i][0], result_frame['location'][i][1], result_frame['location'][i][2]
        #     distance = np.sqrt(np.square(x) + np.square(y) + np.square(z))
        #     if result_frame['score'][i] >= score_threshold and distance < distance_threshold:
        #         predictions.append(result_frame['boxes_lidar'][i])

        print("\nProcessing Sample: %s_%s_%s" % (date, run, frame))

        bev_fig = self.draw_bev(lidar_data, annotations, predictions,
                             os.path.join(self.output_path, "%s_%s_%s.png" % (date, run, frame)))

        bev_fig.canvas.draw()
        bev_fig_data = np.fromstring(bev_fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        bev_fig_array = bev_fig_data.reshape(bev_fig.canvas.get_width_height()[::-1] + (3,))
        return bev_fig, bev_fig_array