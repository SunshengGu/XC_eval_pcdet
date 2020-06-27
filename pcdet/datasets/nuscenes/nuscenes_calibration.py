import numpy as np
import yaml
import os

from nuscenes.utils.geometry_utils import transform_matrix, view_points
from pyquaternion import Quaternion

class Calibration(object):
    def __init__(self, ego_pose, cam_calibrated, lidar_calibrated):
        """
        :param ego_pose: ego_pose dictionary from NuScenes
        :param cam_calibrated: calibrated_sensor dictionary from NuScenes
        :param lidar_calibrated: calibrated_sensor dictionary from NuScenes
        """
        self.t_global_car = transform_matrix(ego_pose['translation'],
                                    Quaternion(ego_pose['rotation']), inverse=False)
        self.t_car_global = transform_matrix(ego_pose['translation'],
                                    Quaternion(ego_pose['rotation']), inverse=True)
        self.t_car_lidar = transform_matrix(lidar_calibrated['translation'],
                                    Quaternion(lidar_calibrated['rotation']), inverse=False)
        self.t_lidar_car = transform_matrix(lidar_calibrated['translation'],
                                    Quaternion(lidar_calibrated['rotation']), inverse=True)
        self.t_car_cam = transform_matrix(cam_calibrated['translation'],
                                    Quaternion(cam_calibrated['rotation']), inverse=False)
        self.t_cam_car = transform_matrix(cam_calibrated['translation'],
                                    Quaternion(cam_calibrated['rotation']), inverse=True)
        camera_intrinsic = np.array(cam_calibrated['camera_intrinsic'])
        self.t_img_cam = np.eye(4)
        self.t_img_cam[:camera_intrinsic.shape[0], :camera_intrinsic.shape[1]] = camera_intrinsic
        
        self.ego_pose = ego_pose
        self.lidar_calibrated = lidar_calibrated
        self.cam_calibrated = cam_calibrated

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        if (len(pts) == 0):
            return pts
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        Note: Using the front camera as rectified frame
        """
        if (len(pts_rect) == 0):
            return pts

        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        t_lidar_cam = np.dot(self.t_lidar_car, self.t_car_cam)
        return np.matmul(t_lidar_cam,pts_rect_hom.T).T[:,0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        if (len(pts_lidar) == 0):
            return pts
            
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        t_cam_lidar = np.dot(self.t_cam_car, self.t_car_lidar)
        return np.matmul(t_cam_lidar,pts_lidar_hom.T).T[:,0:3]

    def rect_to_img(self, pts_rect, cam=0):
        """
        :param pts_rect: (N, 3)
        :param cam: Int, camera number to project onto
        :return pts_img: (N, 2)
        """
        if (len(pts_rect) == 0):
            return pts
            
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(self.t_img_cam, pts_rect_hom.T).T
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.t_img_cam.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar, cam=0):
        """
        :param pts_lidar: (N, 3)
        :param cam: Int, camera number to project onto
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect, cam=cam)
        return pts_img, pts_depth

    def lidar_to_global(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_global: (N, 3)
        """
        if (len(pts_lidar) == 0):
            return pts
            
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        t_global_lidar = np.dot(self.t_global_car, self.t_car_lidar)
        return np.matmul(t_global_lidar,pts_lidar_hom.T).T[:,0:3]
    
    def velo_global_to_lidar(self, velo_xy):
        """
        :param velo_xy: (N, 2)
        :return velo_lidar: (N, 2)
        """
        velo_hom = self.cart_to_hom(velo_xy)
        
        l2e_r = self.lidar_calibrated['rotation']
        e2g_r = self.ego_pose['rotation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        
        velo_lidar = velo_hom @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
            l2e_r_mat).T
        velo_lidar = velo_lidar[:,:2].reshape(-1, 2)
        
        return velo_lidar
        
    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        raise NotImplementedError

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        raise NotImplementedError
