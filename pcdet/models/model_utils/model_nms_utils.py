import torch

from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    # print('\n entering class_agnostic_nms of model_nms_utils.py')
    src_box_scores = box_scores
    # print("\nnumber of pred boxes before score thresholding--len(src_box_scores): {}".format(len(src_box_scores)))
    # print("\nsample scores--src_box_scores[:5]: {}".format(src_box_scores[:5]))
    # print("\ntorch.mean(src_box_scores): {}".format(torch.mean(src_box_scores)))
    # print("\ntorch.max(src_box_scores): {}".format(torch.max(src_box_scores)))
    # print("\ntorch.min(src_box_scores): {}".format(torch.min(src_box_scores)))
    # print("\nsrc_box_scores.shape: {}".format(src_box_scores.shape))
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    # print("\nnumber of pred boxes after score thresholding--len(box_scores): {}".format(len(box_scores)))
    if box_scores.shape[0] > 0:
        # for PointPillar:
        # NMS_PRE_MAXSIZE: 4096
        # NMS_POST_MAXSIZE: 500
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]), sorted=True)
        # box_scores_nms: box scores exceeding the threshold
        # indices: indices of these box scores
        # boxes_for_nms: corresponding boxes, each with 7 parameters
        boxes_for_nms = box_preds[indices]
        # print("\nnumber of pred boxes for which nms is to be applied--len(indices): {}".format(len(indices)))
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]
        # print("\nnumber of pred boxes after nms--len(selected): {}".format(len(selected)))
        # print('selected type: ' + str(type(selected)))
        # print('selected shape: ' + str(selected.shape))
        # size of `selected` matches with `num_out` in nms_gpu of pcdet/ops/iou3d_nms/iou3d_nms_utils.py

    if score_thresh is not None: # get rid of components that fail to exceed the score threshold
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected] # indices of the top boxes picked by nms_gpu
    # print('\n exiting class_agnostic_nms of model_nms_utils.py')
    return selected, src_box_scores[selected]


def multi_classes_nms(cls_scores, box_preds, nms_config, score_thresh=None):
    """
    Args:
        cls_scores: (N, num_class)
        box_preds: (N, 7 + C)
        nms_config:
        score_thresh:

    Returns:

    """
    pred_scores, pred_labels, pred_boxes = [], [], []
    for k in range(cls_scores.shape[1]):
        if score_thresh is not None:
            scores_mask = (cls_scores[:, k] >= score_thresh)
            box_scores = cls_scores[scores_mask, k]
            cur_box_preds = box_preds[scores_mask]
        else:
            box_scores = cls_scores[:, k]

        selected = []
        if box_scores.shape[0] > 0:
            box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
            boxes_for_nms = cur_box_preds[indices]
            keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
                    boxes_for_nms[:, 0:7], box_scores_nms, nms_config.NMS_THRESH, **nms_config
            )
            selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]

        pred_scores.append(box_scores[selected])
        pred_labels.append(box_scores.new_ones(len(selected)).long() * k)
        pred_boxes.append(cur_box_preds[selected])

    pred_scores = torch.cat(pred_scores, dim=0)
    pred_labels = torch.cat(pred_labels, dim=0)
    pred_boxes = torch.cat(pred_boxes, dim=0)

    return pred_scores, pred_labels, pred_boxes
