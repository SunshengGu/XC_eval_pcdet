import numpy as np
import math
import copy
import torch
from numba import jit, cuda

from pcdet.utils import common_utils


def bboxes3d_nearest_bev_iou(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_a = torch.from_numpy(boxes_a)
    boxes_b = torch.from_numpy(boxes_b)
    boxes_a = boxes_a.float()
    boxes_b = boxes_b.float()
    boxes_bev_a = boxes3d_lidar_to_aligned_bev_boxes(boxes_a)
    boxes_bev_b = boxes3d_lidar_to_aligned_bev_boxes(boxes_b)

    return boxes_iou_normal(boxes_bev_a, boxes_bev_b)


def box_preprocess(box_vertices):
    """
    :param box_vertices:
    :return:
        AB: one edge of the box in vector form
        AD: another edge of the box in vector form
        AB_dot_AB: scalar, dot product
        AD_dot_AD: scalar, dot product
    """
    A = box_vertices[0]
    B = box_vertices[1]
    D = box_vertices[3]
    AB, AD = np.zeros(2), np.zeros(2)
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]
    AD[0] = D[0] - A[0]
    AD[1] = D[1] - A[1]
    AB_dot_AB = np.dot(AB, AB)
    AD_dot_AD = np.dot(AD, AD)
    return AB, AD, AB_dot_AB, AD_dot_AD


def box_preprocess_tensor(box_vertices):
    """
    :param box_vertices:
    :return:
        AB: one edge of the box in vector form
        AD: another edge of the box in vector form
        AB_dot_AB: scalar, dot product
        AD_dot_AD: scalar, dot product
    """
    A = box_vertices[0]
    B = box_vertices[1]
    D = box_vertices[3]
    AB, AD = torch.zeros(2).cuda(), torch.zeros(2).cuda()
    AB[0] = B[0] - A[0]
    AB[1] = B[1] - A[1]
    AD[0] = D[0] - A[0]
    AD[1] = D[1] - A[1]
    AB_dot_AB = torch.dot(AB, AB)
    # print("AB_dot_AB.device: {}".format(AB_dot_AB.device)) # it is already loaded onto the gpu
    AD_dot_AD = torch.dot(AD, AD)
    return AB, AD, AB_dot_AB, AD_dot_AD


def box_validation(box, box_vertices, dataset_name):
    """
    :param box: a bounding box of the form (x,y,z,l,w,h,theta)
    :param box_vertices: vertices of the 2d box ((x1,y1), (x2,y1), (x2,y2) ,(x1,y2))
    :return: boolean indicating if the box matches the vertices (to be implemented)
    """

    l1 = get_dist(box_vertices[0], box_vertices[1])
    l2 = get_dist(box_vertices[1], box_vertices[2])
    l3 = get_dist(box_vertices[2], box_vertices[3])
    l4 = get_dist(box_vertices[3], box_vertices[0])

    if (abs(box[3] - l1) < 0.00001 or abs(box[3] - l2) < 0.00001) and \
            (abs(box[4] - l1) < 0.00001 or abs(box[4] - l2) < 0.00001):
        # print('Boxes matched!')
        return True
    else:
        # print('Mismatch!')
        return False


def boxes3d_lidar_to_aligned_bev_boxes(boxes3d):
    """
    Args:
        boxes3d: (N, 7 + C) [x, y, z, dx, dy, dz, heading] in lidar coordinate

    Returns:
        aligned_bev_boxes: (N, 4) [x1, y1, x2, y2] in the above lidar coordinate
    """
    # rot_angle_raw = common_utils.limit_period(boxes3d[:, 6], offset=0.5, period=np.pi)
    # print("type(rot_angle_raw): {}".format(type(rot_angle_raw)))
    rot_angle = common_utils.limit_period(boxes3d[:, 6], offset=0.5, period=np.pi).abs()
    choose_dims = torch.where(rot_angle[:, None] < np.pi / 4, boxes3d[:, [3, 4]], boxes3d[:, [4, 3]])
    aligned_bev_boxes = torch.cat((boxes3d[:, 0:2] - choose_dims / 2, boxes3d[:, 0:2] + choose_dims / 2), dim=1)
    return aligned_bev_boxes


def boxes_iou_normal(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 4) [x1, y1, x2, y2]
        boxes_b: (M, 4) [x1, y1, x2, y2]

    Returns:

    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    x_min = torch.max(boxes_a[:, 0, None], boxes_b[None, :, 0])
    x_max = torch.min(boxes_a[:, 2, None], boxes_b[None, :, 2])
    y_min = torch.max(boxes_a[:, 1, None], boxes_b[None, :, 1])
    y_max = torch.min(boxes_a[:, 3, None], boxes_b[None, :, 3])
    x_len = torch.clamp_min(x_max - x_min, min=0)
    y_len = torch.clamp_min(y_max - y_min, min=0)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    a_intersect_b = x_len * y_len
    iou = a_intersect_b / torch.clamp_min(area_a[:, None] + area_b[None, :] - a_intersect_b, min=1e-6)
    return iou


def find_anchor_index(dataset_name, y, x):
    """
    Given y and x coordinate of the center of an anchor box,
    find the index for that anchor
    :param dataset_name:
    :param y:
    :param x:
    :return:
    """
    if dataset_name == 'KittiDataset':
        H_grid = 248
        W_grid = 1296
        H_step = 80 / H_grid
        W_step = 69.12 / W_grid
        row_id = math.floor(y / H_step)
        col_id = math.floor(x / W_step)
        anchor_id = row_id * W_grid + col_id
        return anchor_id
    elif dataset_name == 'CadcDataset':
        H_grid = 200
        W_grid = 1200
        H_step = 100 / H_grid
        W_step = 100 / W_grid
        row_id = math.floor(y / H_step)
        col_id = math.floor(x / W_step)
        anchor_id = row_id * W_grid + col_id
        return anchor_id


def flip_xy(box_vertices):
    box_vertices_copy = copy.deepcopy(box_vertices)
    for vertex in box_vertices_copy:
        temp = vertex[0]
        vertex[0] = vertex[1]
        vertex[1] = temp
    return box_vertices_copy


def generate_box_mask(box_mask, box_center, vicinity, corner_A, AB, AD, AB_dot_AB, AD_dot_AD):
    """
    :param vicinity: search vicinity w.r.t. the box_center, class dependent
    :param box_center: center of the pred box we wish to explain
    :param box_mask: numpy array, a mask for the box to be generated here
    :param corner_A: a corner of the box
    :param AB:
    :param AD:
    :param AB_dot_AB:
    :param AD_dot_AD:
    Idea:
    - Create a mask for the box within the box center's vicinity
    - The size of the vicinity is determined by the class of the object
    :return:
    """
    H = box_mask.shape[0]
    W = box_mask.shape[1]
    y_start = max(0, int(box_center[0]-vicinity))
    y_end = min(H, int(box_center[0]+vicinity+1))
    x_start = max(0, int(box_center[1] - vicinity))
    x_end = min(W, int(box_center[1] + vicinity + 1))
    for i in range(y_start, y_end, 1):
        for j in range(x_start, x_end, 1):
            if in_box(corner_A, i, j, AB, AD, AB_dot_AB, AD_dot_AD):
                box_mask[i][j] = 1


def generate_box_mask_tensor(box_mask, box_center, vicinity, corner_A, AB, AD, AB_dot_AB, AD_dot_AD):
    """
    :param vicinity: search vicinity w.r.t. the box_center, class dependent
    :param box_center: center of the pred box we wish to explain
    :param box_mask: numpy array, a mask for the box to be generated here
    :param corner_A: a corner of the box
    :param AB:
    :param AD:
    :param AB_dot_AB:
    :param AD_dot_AD:
    Idea:
    - Create a mask for the box within the box center's vicinity
    - The size of the vicinity is determined by the class of the object
    :return:
    """
    H = box_mask.shape[0]
    W = box_mask.shape[1]
    y_start = max(0, int(box_center[0]-vicinity))
    y_end = min(H, int(box_center[0]+vicinity+1))
    x_start = max(0, int(box_center[1] - vicinity))
    x_end = min(W, int(box_center[1] + vicinity + 1))
    for i in range(y_start, y_end, 1):
        for j in range(x_start, x_end, 1):
            if in_box_tensor(corner_A, i, j, AB, AD, AB_dot_AB, AD_dot_AD):
                box_mask[i][j] = 1


def get_box_center(vertices):
    y_center = np.mean(vertices[:, 0])
    x_center = np.mean(vertices[:, 1])
    return y_center, x_center


def get_box_scale(dataset_name):
    """
    Give a dataset_name, compute the conversion factor from meters to pixels
    Matching the pointpillar 2d pseudoimage
    :param dataset_name:
    :return:
    """
    if dataset_name == 'CadcDataset':
        return 400 / 100
    if dataset_name == 'KittiDataset':
        return 496 / 79.36
    if dataset_name == 'WaymoDataset':
        return 456 / 150.4


def get_dist(p1, p2):
    """
    :param p1: point 1 in xy plane
    :param p2: point 2 in xy plane
    :return: the distance between the points
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.sqrt(dx * dx + dy * dy)


def in_box(A, y, x, AB, AD, AB_dot_AB, AD_dot_AD):
    """
    reference: https://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle
    :param A: first vertex of the box
    :param y: y coordinate of M
    :param x: x coordinate of M
    :return: If point M(y,x) is in the box
    """
    AM = np.zeros(2)
    AM[0] = y - A[0]
    AM[1] = x - A[1]
    AM_dot_AB = np.dot(AM, AB)
    if AM_dot_AB < 0 or AM_dot_AB > AB_dot_AB:
        return False
    AM_dot_AD = np.dot(AM, AD)
    if AM_dot_AD < 0 or AM_dot_AD > AD_dot_AD:
        return False
    return True


def in_box_tensor(A, y, x, AB, AD, AB_dot_AB, AD_dot_AD):
    """
    reference: https://math.stackexchange.com/questions/190111/how-to-check-if-a-point-is-inside-a-rectangle
    :param A: first vertex of the box
    :param y: y coordinate of M
    :param x: x coordinate of M
    :return: If point M(y,x) is in the box
    """
    AM = torch.zeros(2).cuda()
    AM[0] = y - A[0]
    AM[1] = x - A[1]
    AM_dot_AB = torch.dot(AM, AB)
    if AM_dot_AB < 0 or AM_dot_AB > AB_dot_AB:
        return False
    AM_dot_AD = torch.dot(AM, AD)
    if AM_dot_AD < 0 or AM_dot_AD > AD_dot_AD:
        return False
    return True


def rotate(ox, oy, point, angle):
    """
    Reference: https://stackoverflow.com/questions/34372480/rotate-point-about-another-point-in-degrees-python
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in radians.
    """
    px, py = point[0], point[1]
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    point[0], point[1] = qx, qy


def rotate_and_flip(box_vertices, dataset_name, angle):
    angle = angle * np.pi / 180.0  # convert to radian
    box_vertices_copy = copy.deepcopy(box_vertices)
    x_max = 0.0
    y_max = 0.0
    if dataset_name == 'CadcDataset':
        x_max = 400.0
        y_max = 400.0
    if dataset_name == 'KittiDataset':
        x_max = 496.0
        y_max = 432.0
    if dataset_name == 'WaymoDataset':
        x_max = 456.0
        y_max = 456.0
    # TODO: verify the following for waymo
    vert_center = y_max/2.0
    horiz_center = x_max/2.0
    if dataset_name == 'KittiDataset':
        for vertex in box_vertices_copy:
            # first, rotate ccw by a certain degrees
            rotate(vert_center, horiz_center, vertex, angle)
            # then, flip about the vertical axis
            '''Note: x is the vertical direction'''
            vert_dist = vertex[1] - vert_center
            vertex[1] = vert_center - vert_dist
    else: # waymo or cadc
        for vertex in box_vertices_copy:
            # first, rotate ccw by a certain degrees
            rotate(vert_center, horiz_center, vertex, angle)
    return box_vertices_copy


def scale_3d_array(orig_array, factor):
    '''

    :param orig_array: Input 3D array.
    :param factor: The factor by which the input is scaled
    :return: new_array: The resulting rescaled array
    '''
    new_layer_list = []
    channels = orig_array.shape[2]
    for i in range(channels):
        layer_i = orig_array[:, :, i]
        new_layer_i = np.kron(layer_i, np.ones((factor, factor), dtype=int))
        new_layer_list.append(new_layer_i)
    new_layer_tuple = tuple(new_layer_list)
    new_array = np.dstack(new_layer_tuple)
    return new_array


def transform_box_center_coord(coord, dataset_name, high_rez=False, scaling_factor=1):
    """
    :param coord: coordinate of the center of the box in meters (x, y) w.r.t. ego location
    :param dataset_name:
    :param high_rez: indicate if we are transforming to the high resolution coordinates
    :param scaling_factor:
    :return:
    """
    if dataset_name == 'CadcDataset':
        # x_range = 100.0
        y_range = 100.0
        H = 400
        if high_rez:
            H = H * scaling_factor
        #TODO: the remaining part of this if statement may need to be changed
        new_scale = H / y_range
        # print('H: {}'.format(H))
        y = coord[1] * new_scale
        # y = H - y
        x = coord[0] * new_scale
        return np.array([y,x])
    elif dataset_name == 'KittiDataset':
        '''Note: the range for Kitti is different now'''
        # x_range = 70.4
        y_range = 79.36
        H = 496
        if high_rez:
            H = H * scaling_factor
        new_scale = H / y_range
        y = coord[1] + 39.68
        y = y * new_scale
        # y = H - y
        x = coord[0] * new_scale
        return np.array([y,x])
    elif dataset_name == 'WaymoDataset':
        # x_range = 100.0
        y_range = 150.4
        H = 456
        if high_rez:
            H = H * scaling_factor
        #TODO: the remaining part of this if statement may need to be changed
        new_scale = H / y_range
        y = y_orig + coord[0]
        x = x_orig - coord[1]
        y = y_range - y
        y = y * new_scale
        x = x * new_scale
        return np.array([y,x])


def transform_box_center_coord_tensor(coord, dataset_name, high_rez=False, scaling_factor=1):
    """
    :param coord: coordinate of the center of the box in meters (x, y) w.r.t. ego location
    :param dataset_name:
    :param high_rez: indicate if we are transforming to the high resolution coordinates
    :param scaling_factor:
    :return:
    """
    if dataset_name == 'CadcDataset':
        # x_range = 100.0
        y_range = 100.0
        H = 400
        if high_rez:
            H = H * scaling_factor
        #TODO: the remaining part of this if statement may need to be changed
        new_scale = H / y_range
        # print('H: {}'.format(H))
        y = coord[1] * new_scale
        # y = H - y
        x = coord[0] * new_scale
        return torch.tensor([y, x]).cuda()
    elif dataset_name == 'KittiDataset':
        '''Note: the range for Kitti is different now'''
        # x_range = 70.4
        y_range = 79.36
        H = 496
        if high_rez:
            H = H * scaling_factor
        new_scale = H / y_range
        y = coord[1] + 39.68
        y = y * new_scale
        # y = H - y
        x = coord[0] * new_scale
        return torch.tensor([y, x]).cuda()
    elif dataset_name == 'WaymoDataset':
        # TODO: need to check if the following is valid
        # origin coordinate in 
        x_orig = 75.2
        y_orig = 75.2
        y_range = 150.4
        H = 456
        if high_rez:
            H = H * scaling_factor
        new_scale = H / y_range
        y = y_orig + coord[0]
        x = x_orig - coord[1]
        x = y_range - x
        y = y * new_scale
        x = x * new_scale
        return torch.tensor([x, y]).cuda()


def transform_box_coord(H, W, box_vertices, dataset_name, high_rez=False, scaling_factor=1):
    """
    Transform box_vertices to match the coordinate system of the attributions
    :param H: Desired height of image
    :param W: Desired width of image
    :param box_vertices:
    :param dataset_name:
    :param high_rez:
    :param scaling_factor:
    :return: transformed box_vertices
    """
    if high_rez:
        H = H * scaling_factor
        # W = W * scaling_factor
    x_range, y_range = None, None
    if dataset_name == 'CadcDataset':
        # x_range = 100.0
        y_range = 100.0
    elif dataset_name == 'KittiDataset':
        '''Note: the range for Kitti is different now'''
        # x_range = 70.4
        y_range = 79.36
    elif dataset_name == 'WaymoDataset':
        y_range = 150.4
    new_scale = H / y_range
    # print('H: {}'.format(H))
    if dataset_name == 'KittiDataset':
        for vertex in box_vertices:
            vertex[0] = vertex[0] * new_scale
            vertex[0] = H - vertex[0]
            vertex[1] = vertex[1] * new_scale
    else:
        for vertex in box_vertices:
            vertex[0] = vertex[0] * new_scale
            vertex[0] = H - vertex[0]
            vertex[1] = vertex[1] * new_scale
    return box_vertices


def transform_box_coord_pseudo(H, W, box_vertices, dataset_name):
    """
    Transform box_vertices to match the coordinate system of the attributions
    :param H: Desired height of image
    :param W: Desired width of image
    :param box_vertices:
    :param dataset_name:
    :param high_rez:
    :param scaling_factor:
    :return: transformed box_vertices
    """

    y_range = None
    if dataset_name == 'CadcDataset':
        y_range = 100.0
    elif dataset_name == 'KittiDataset':
        '''Note: the range for Kitti is different now'''
        y_range = 79.36
    elif dataset_name == 'WaymoDataset':
        y_range = 150.4
    new_scale = H / y_range
    # print('H: {}'.format(H))
    # TODO: verify the following for waymo
    if dataset_name == 'KittiDataset':
        for vertex in box_vertices:
            vertex[0] = vertex[0] * new_scale
            vertex[0] = H - vertex[0]
            vertex[1] = vertex[1] * new_scale
    else:
	# print("\n")
        for vertex in box_vertices:
            vertex[0] = vertex[0] * new_scale
            vertex[0] = H - vertex[0]
            vertex[1] = vertex[1] * new_scale
            # print("vertex: {}".format(vertex))
    return box_vertices


def transform_point_coord(H, W, coord, dataset_name, high_rez=False, scaling_factor=1):
    if high_rez:
        H = H * scaling_factor
        # W = W * scaling_factor
    x_range, y_range = None, None
    if dataset_name == 'CadcDataset':
        # x_range = 100.0
        y_range = 100.0
    elif dataset_name == 'KittiDataset':
        '''Note: the range for Kitti is different now'''
        # x_range = 70.4
        y_range = 79.36
    elif dataset_name == 'WaymoDataset':
        y_range = 150.4
    new_scale = H / y_range
    # TODO: verify the following for waymo
    # print('H: {}'.format(H))
    y = coord[0] * new_scale
    # y = H - y
    x = coord[1] * new_scale
    return y, x


def transform_pred_point_coord(H, W, coord, dataset_name, high_rez=False, scaling_factor=1):
    if high_rez:
        H = H * scaling_factor
        # W = W * scaling_factor
    x_range, y_range = None, None
    if dataset_name == 'CadcDataset':
        # x_range = 100.0
        y_range = 100.0
    elif dataset_name == 'KittiDataset':
        '''Note: the range for Kitti is different now'''
        # x_range = 70.4
        y_range = 79.36
    elif dataset_name == 'WaymoDataset':
        y_range = 150.4
    new_scale = H / y_range
    # TODO: verify the following for waymo
    # print('H: {}'.format(H))
    y = coord[0] * new_scale
    if dataset_name == 'KittiDataset':
        y = H - y
    x = coord[1] * new_scale
    return y, x


if __name__ == '__main__':
    pass
