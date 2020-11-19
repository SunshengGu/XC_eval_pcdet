from .bbox_utils import *
import numpy as np
from numba import jit, cuda


def get_sum_XQ(grad, box_vertices, dataset_name, box_w, box_l, sign, high_rez=False, scaling_factor=1,
               grad_copy=None):
    """
    Calculates XQ based on the sum of attributions, using the original attribution values
    :param high_rez: Whether to upscale the resolution or use pseudoimage resolution
    :param scaling_factor:
    :param dataset_name:
    :param grad: The attributions generated from 2D pseudoimage
    :param box_vertices: The vertices of the predicted box
    :return: Explanation Quality (XQ)
    """
    # print('box_vertices before transformation: {}'.format(box_vertices))
    '''1) transform the box coordinates to match with grad dimensions'''
    H, W = grad.shape[0], grad.shape[1]  # image height and width
    box_vertices = transform_box_coord(H, W, box_vertices, dataset_name, high_rez, scaling_factor)
    # print('box_vertices after transformation: {}'.format(box_vertices))
    # print('\ngrad.shape: {}'.format(grad.shape))
    '''2) preprocess the box to get important parameters'''
    AB, AD, AB_dot_AB, AD_dot_AD = box_preprocess(box_vertices)
    # box_w = get_dist(box_vertices[0], box_vertices[1])
    # box_l = get_dist(box_vertices[0], box_vertices[3])
    box_scale = get_box_scale(dataset_name)
    box_w = box_w * box_scale
    box_l = box_l * box_scale

    '''3) compute XQ'''
    ignore_thresh = 0.1
    # margin = 2  # margin for the box
    attr_in_box = 0.0
    total_attr = 0
    # max_xq = 0
    # if sign == 'positive':
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            curr_sum = 0
            # print('grad[i][j].shape: {}'.format(grad[i][j].shape))
            if sign == 'positive':
                # both summing schemes give the same results
                curr_sum = np.sum((grad[i][j] > 0) * grad[i][j])
                # curr_sum = np.sum(np.where(grad[i][j] < 0, 0, grad[i][j]))
            elif sign == 'negative':
                curr_sum -= np.sum((grad[i][j] < 0) * grad[i][j])
                # curr_sum -= np.sum(np.where(grad[i][j] > 0, 0, grad[i][j]))
            else:
                curr_sum = np.sum(abs(grad[i][j]))
            if curr_sum < ignore_thresh:  # ignore small attributions
                continue
            total_attr += curr_sum
            # max_xq = max(max_xq, curr_sum)
            y = i
            x = j
            if high_rez:
                y = i * scaling_factor
                x = j * scaling_factor
            if in_box(box_vertices[0], y, x, AB, AD, AB_dot_AB, AD_dot_AD):
                attr_in_box += curr_sum
    # print("maximum xq is : {}".format(max_xq))
    # box area matching pseudoimage dimensions (i.e. grad)
    box_area = box_w * box_l
    avg_in_box_attr = attr_in_box / box_area
    avg_attr = total_attr / (grad.shape[0] * grad.shape[1])
    print("avg_attr: {}".format(avg_attr))
    print("avg_in_box_attr: {}".format(avg_in_box_attr))
    if total_attr == 0:
        print("No attributions present!")
        return 0
    XQ = attr_in_box / total_attr
    print("XQ: {}".format(XQ))
    return XQ


def get_sum_XQ_analytics(pos_grad, neg_grad, box_vertices, dataset_name, sign, ignore_thresh, box_w=None, box_l=None,
                         high_rez=False, scaling_factor=1, grad_copy=None):
    """
    Calculates XQ based on the sum of attributions, resolution considered
    :param high_rez: Whether to upscale the resolution or use pseudoimage resolution
    :param scaling_factor:
    :param dataset_name:
    :param grad: The attributions generated from 2D pseudoimage
    :param box_vertices: The vertices of the predicted box
    :return: Explanation Quality (XQ)
    """
    # print('box_vertices before transformation: {}'.format(box_vertices))
    grad = None
    if sign == 'positive':
        grad = pos_grad
    elif sign == 'negative':
        grad = neg_grad
    # print("type(grad): {}".format(type(grad)))
    '''1) transform the box coordinates to match with grad dimensions'''
    H, W = grad.shape[0], grad.shape[1]  # image height and width
    box_vertices = transform_box_coord(H, W, box_vertices, dataset_name, high_rez, scaling_factor)
    # print('box_vertices after transformation: {}'.format(box_vertices))
    # print('\ngrad.shape: {}'.format(grad.shape))
    '''2) preprocess the box to get important parameters'''
    AB, AD, AB_dot_AB, AD_dot_AD = box_preprocess(box_vertices)
    # box_w = get_dist(box_vertices[0], box_vertices[1])
    # box_l = get_dist(box_vertices[0], box_vertices[3])
    box_scale = get_box_scale(dataset_name)

    '''3) compute XQ'''
    # ignore_thresh = 0
    # margin = 2  # margin for the box
    attr_in_box = 0.0
    total_attr = 0
    # max_xq = 0
    # if sign == 'positive':
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            if grad[i][j] < ignore_thresh:
                continue
            total_attr += grad[i][j]
            # max_xq = max(max_xq, curr_sum)
            y = copy.deepcopy(i)
            x = copy.deepcopy(j)
            if high_rez:
                # print("high rez is true")
                y = i * scaling_factor
                x = j * scaling_factor
            if in_box(box_vertices[0], y, x, AB, AD, AB_dot_AB, AD_dot_AD):
                attr_in_box += grad[i][j]
                # if grad_copy is not None:
                #     grad_copy[i][j] = 1
    # print("maximum xq is : {}".format(max_xq))
    # box area matching pseudoimage dimensions (i.e. grad)
    if box_w != None and box_l != None:
        box_w = box_w * box_scale
        box_l = box_l * box_scale
        box_area = box_w * box_l
        avg_in_box_attr = attr_in_box / box_area
        avg_attr = total_attr / (grad.shape[0] * grad.shape[1])
        # print("avg_attr: {}".format(avg_attr))
        # print("avg_in_box_attr: {}".format(avg_in_box_attr))
    if total_attr == 0:
        # print("No attributions present!")
        return 0
    XQ = attr_in_box / total_attr
    # print("XQ: {}".format(XQ))
    return XQ


def get_sum_XQ_analytics_fast(pos_grad, neg_grad, box_vertices, dataset_name, sign, ignore_thresh, box_loc, vicinity):
    """
    Calculates XQ based on the sum of attributions, resolution not considered
    :param vicinity: search vicinity w.r.t. the box_center, class dependent
    :param ignore_thresh: the threshold below which the attributions would be ignored
    :param box_loc: location of the predicted box
    :param sign: indicates the type of attributions shown, positive or negative
    :param neg_grad: numpy array containing sum of negative gradients at each location
    :param pos_grad: numpy array containing sum of positive gradients at each location
    :param dataset_name:
    :param box_vertices: The vertices of the predicted box
    :return: Explanation Quality (XQ)
    """
    # print('box_vertices before transformation: {}'.format(box_vertices))
    grad = None
    if sign == 'positive':
        grad = pos_grad
    elif sign == 'negative':
        grad = neg_grad
    '''1) transform the box coordinates to match with grad dimensions'''
    H, W = grad.shape[0], grad.shape[1]  # image height and width
    box_vertices = transform_box_coord_pseudo(H, W, box_vertices, dataset_name)
    box_loc = transform_box_center_coord(box_loc, dataset_name)
    '''2) preprocess the box to get important parameters'''
    AB, AD, AB_dot_AB, AD_dot_AD = box_preprocess(box_vertices)
    '''3) compute XQ'''
    box_mask = np.zeros((H, W))
    generate_box_mask(box_mask, box_loc, vicinity, box_vertices[0], AB, AD, AB_dot_AB, AD_dot_AD)
    # grad[grad >= ignore_thresh] is an 1D array, shape doesn't match grad
    total_attr = np.sum(grad[grad >= ignore_thresh])
    masked_attr = grad[box_mask == 1]
    attr_in_box = np.sum(masked_attr[masked_attr >= ignore_thresh])
    if total_attr == 0:
        # print("No attributions present!")
        return 0
    XQ = attr_in_box / total_attr
    # print("XQ: {}".format(XQ))
    return XQ


def get_cnt_XQ_analytics(pos_grad, neg_grad, box_vertices, dataset_name, sign, ignore_thresh, box_w=None, box_l=None,
                         high_rez=False, scaling_factor=1, grad_copy=None):
    """
    Calculates XQ based on the sum of attributions
    :param high_rez: Whether to upscale the resolution or use pseudoimage resolution
    :param scaling_factor:
    :param dataset_name:
    :param grad: The attributions generated from 2D pseudoimage
    :param box_vertices: The vertices of the predicted box
    :return: Explanation Quality (XQ)
    """
    # print('box_vertices before transformation: {}'.format(box_vertices))
    grad = None
    if sign == 'positive':
        grad = pos_grad
    elif sign == 'negative':
        grad = neg_grad
    print("type(grad): {}".format(type(grad)))
    '''1) transform the box coordinates to match with grad dimensions'''
    H, W = grad.shape[0], grad.shape[1]  # image height and width
    box_vertices = transform_box_coord(H, W, box_vertices, dataset_name, high_rez, scaling_factor)
    # print('box_vertices after transformation: {}'.format(box_vertices))
    # print('\ngrad.shape: {}'.format(grad.shape))
    '''2) preprocess the box to get important parameters'''
    AB, AD, AB_dot_AB, AD_dot_AD = box_preprocess(box_vertices)
    # box_w = get_dist(box_vertices[0], box_vertices[1])
    # box_l = get_dist(box_vertices[0], box_vertices[3])
    box_scale = get_box_scale(dataset_name)

    '''3) compute XQ'''
    # ignore_thresh = 0.1
    # margin = 2  # margin for the box
    attr_in_box = 0
    total_attr = 0
    # max_xq = 0
    # if sign == 'positive':
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            if grad[i][j] < ignore_thresh:
                continue
            total_attr += 1
            # max_xq = max(max_xq, curr_sum)
            y = copy.deepcopy(i)
            x = copy.deepcopy(j)
            if high_rez:
                y = i * scaling_factor
                x = j * scaling_factor
            if in_box(box_vertices[0], y, x, AB, AD, AB_dot_AB, AD_dot_AD):
                attr_in_box += 1
    # print("maximum xq is : {}".format(max_xq))
    # box area matching pseudoimage dimensions (i.e. grad)
    if box_w != None and box_l != None:
        box_w = box_w * box_scale
        box_l = box_l * box_scale
        box_area = box_w * box_l
        avg_in_box_attr = attr_in_box / box_area
        avg_attr = total_attr / (grad.shape[0] * grad.shape[1])
        print("avg_attr: {}".format(avg_attr))
        print("avg_in_box_attr: {}".format(avg_in_box_attr))
    if total_attr == 0:
        print("No attributions present!")
        return 0
    XQ = attr_in_box / total_attr
    print("XQ: {}".format(XQ))
    return XQ


def get_cnt_XQ_analytics_fast(pos_grad, neg_grad, box_vertices, dataset_name, sign, ignore_thresh, box_loc, vicinity):
    """
    Calculates XQ based on the count of pixels exceeding certain attr threshold, resolution not considered
    :param vicinity: search vicinity w.r.t. the box_center, class dependent
    :param ignore_thresh: the threshold below which the attributions would be ignored
    :param box_loc: location of the predicted box
    :param sign: indicates the type of attributions shown, positive or negative
    :param neg_grad: numpy array containing sum of negative gradients at each location
    :param pos_grad: numpy array containing sum of positive gradients at each location
    :param dataset_name:
    :param box_vertices: The vertices of the predicted box
    :return: Explanation Quality (XQ)
    """
    # print('box_vertices before transformation: {}'.format(box_vertices))
    grad = None
    if sign == 'positive':
        grad = pos_grad
    elif sign == 'negative':
        grad = neg_grad
    '''1) transform the box coordinates to match with grad dimensions'''
    H, W = grad.shape[0], grad.shape[1]  # image height and width
    box_vertices = transform_box_coord_pseudo(H, W, box_vertices, dataset_name)
    box_loc = transform_box_center_coord(box_loc, dataset_name)
    '''2) preprocess the box to get important parameters'''
    AB, AD, AB_dot_AB, AD_dot_AD = box_preprocess(box_vertices)
    '''3) compute XQ'''
    box_mask = np.zeros((H, W))
    generate_box_mask(box_mask, box_loc, vicinity, box_vertices[0], AB, AD, AB_dot_AB, AD_dot_AD)
    # grad[grad >= ignore_thresh] is an 1D array, shape doesn't match grad
    total_attr = np.count_nonzero(grad[grad >= ignore_thresh])
    masked_attr = grad[box_mask == 1]
    attr_in_box = np.count_nonzero(masked_attr[masked_attr >= ignore_thresh])
    if total_attr == 0:
        # print("No attributions present!")
        return 0
    XQ = attr_in_box / total_attr
    # print("XQ: {}".format(XQ))
    return XQ


def get_cnt_XQ(grad, box_vertices, dataset_name, box_w, box_l, sign, high_rez=False, scaling_factor=1,
               grad_copy=None):
    """
    Calculates XQ based on the count of pixels with attr sum exceeding a certain threshold
    :param margin: Margin for the bounding box. e.g., if margin = 1, then attributions within one pixel distance
        to the box boundary are counted as in box.
    :param high_rez: Whether to upscale the resolution or use pseudoimage resolution
    :param scaling_factor:
    :param dataset_name:
    :param grad: The attributions generated from 2D pseudoimage
    :param box_vertices: The vertices of the predicted box
    :return: Explanation Quality (XQ)
    """
    # print('box_vertices before transformation: {}'.format(box_vertices))
    '''1) transform the box coordinates to match with grad dimensions'''
    H, W = grad.shape[0], grad.shape[1]  # image height and width
    box_vertices = transform_box_coord(H, W, box_vertices, dataset_name, high_rez, scaling_factor)
    # print('box_vertices after transformation: {}'.format(box_vertices))
    # print('\ngrad.shape: {}'.format(grad.shape))
    '''2) preprocess the box to get important parameters'''
    AB, AD, AB_dot_AB, AD_dot_AD = box_preprocess(box_vertices)
    # box_w = get_dist(box_vertices[0], box_vertices[1])
    # box_l = get_dist(box_vertices[0], box_vertices[3])
    box_scale = get_box_scale(dataset_name)
    box_w = box_w * box_scale
    box_l = box_l * box_scale

    '''3) compute XQ'''
    ignore_thresh = 0.1
    # margin = 2  # margin for the box
    attr_in_box = 0
    total_attr = 0
    # if total_attr == 0:
    for i in range(grad.shape[0]):
        for j in range(grad.shape[1]):
            curr_sum = np.sum(grad[i][j])  # sum up attributions in all channels at this location
            if curr_sum < ignore_thresh:  # ignore small attributions
                continue
            total_attr += 1
            y = i
            x = j
            if high_rez:
                y = i * scaling_factor
                x = j * scaling_factor
            if in_box(box_vertices[0], y, x, AB, AD, AB_dot_AB, AD_dot_AD):
                attr_in_box += 1
    # box area matching pseudoimage dimensions (i.e. grad)
    # box_area = box_w * box_l
    # avg_in_box_attr = attr_in_box / box_area
    # avg_attr = total_attr / (grad.shape[0] * grad.shape[1])
    # print("avg_attr: {}".format(avg_attr))
    # print("avg_in_box_attr: {}".format(avg_in_box_attr))
    if total_attr == 0:
        print("No attributions present!")
        return 0
    XQ = attr_in_box / total_attr
    print("XQ: {}".format(XQ))
    return XQ
