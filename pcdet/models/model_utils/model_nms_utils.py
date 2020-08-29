import torch
from ...ops.iou3d_nms import iou3d_nms_utils


def class_agnostic_nms(box_scores, box_preds, nms_config, score_thresh=None):
    # print('\n entering class_agnostic_nms of model_nms_utils.py')
    src_box_scores = box_scores
    if score_thresh is not None:
        scores_mask = (box_scores >= score_thresh)
        box_scores = box_scores[scores_mask]
        box_preds = box_preds[scores_mask]

    selected = []
    if box_scores.shape[0] > 0:
        # for PointPillar:
        # NMS_PRE_MAXSIZE: 4096
        # NMS_POST_MAXSIZE: 500
        box_scores_nms, indices = torch.topk(box_scores, k=min(nms_config.NMS_PRE_MAXSIZE, box_scores.shape[0]))
        # box_scores_nms: box scores exceeding the threshold
        # indices: indices of these box scores
        # boxes_for_nms: corresponding boxes, each with 7 parameters
        boxes_for_nms = box_preds[indices]
        keep_idx, selected_scores = getattr(iou3d_nms_utils, nms_config.NMS_TYPE)(
            boxes_for_nms, box_scores_nms, nms_config.NMS_THRESH, **nms_config
        )
        selected = indices[keep_idx[:nms_config.NMS_POST_MAXSIZE]]
        # print('selected type: ' + str(type(selected)))
        # print('selected shape: ' + str(selected.shape))
        # size of `selected` matches with `num_out` in nms_gpu of pcdet/ops/iou3d_nms/iou3d_nms_utils.py

    if score_thresh is not None: # get rid of components that fail to exceed the score threshold
        original_idxs = scores_mask.nonzero().view(-1)
        selected = original_idxs[selected] # indices of the top boxes picked by nms_gpu
    # print('\n exiting class_agnostic_nms of model_nms_utils.py')
    return selected, src_box_scores[selected]
