from XAI_utils.bbox_utils import *
from attr_generator_tensor import AttributionGeneratorTensor
import numpy as np

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import GuidedBackprop
from captum.attr import GuidedGradCam
from captum.attr import visualization as viz


def attr_func_draft(explained_model, explainer, batch):
    '''

    :param explained_model: the model being explained
    :param explainer: parameters for the attribution generator
    :param batch: the batch of input data we are explaining
    :return:
    '''
    attr_generator = None
    if explainer['method'] == 'IntegratedGradients':
        attr_generator = IntegratedGradients(explained_model, multiply_by_inputs=explainer['mult_inputs'])
    PseudoImage2D = batch['spatial_features']
    top_confidence_ind = batch['anchor_selections'][0][0]  # 0th frame in batch, 0th predicted box's anchor index
    top_confidence_label = batch['anchor_labels'][0][0]  # 0th frame in batch, 0th predicted box's anchor label
    target = (top_confidence_ind, top_confidence_label)
    print("\nexplanation target: {}\n".format(target))
    target = (0, 0)
    batch_attributions = attr_generator.attribute(
        PseudoImage2D, baselines=PseudoImage2D * 0, target=target,
        additional_forward_args=batch, n_steps=explainer['steps_ig'],
        internal_batch_size=batch['batch_size'])
    print("\nIn tools/XAI_utils/attr_util.py, batch_attributions[0].shape: {}".format(batch_attributions[0].shape))
    return 0.00002


def attr_func(explained_model, explainer, batch, dataset_name, cls_names, cur_it, gt_infos, score_thresh,
              object_cnt, tp_object_cnt, fp_object_cnt, box_selection, pred_score_file_name, pred_score_field_name,
              cur_epoch, aggre_method, attr_sign, pap_only=False):
    '''
    :param explained_model: the model being explained
    :param explainer: parameters for the attribution generator
    :param batch: the batch of input data we are explaining
    :param cur_it: the current batch id, starting from 0
    :param score_thresh: the threshold to filter out low confidence predictions
    :param box_selection: either "tp/fp", "top", or "bottom"
    :param cur_epoch: the current epoch
    :param aggre_method: the aggregation method, "sum" or "cnt"
    :param attr_sign: sign of attribution, "positive" or "negative"
    :return:
    '''
    myExplainer = AttributionGeneratorTensor(
        explained_model, dataset_name, cls_names, explainer['method'], None, gt_infos,
        pred_score_file_name=pred_score_file_name, pred_score_field_name=pred_score_field_name,
        score_thresh=score_thresh, debug=True, selection=box_selection)
    if not pap_only:
        if box_selection == "tp/fp":
            tp_XC, tp_far_attr, tp_pap, fp_XC, fp_far_attr, fp_pap = myExplainer.compute_xc(
                batch, object_cnt, tp_object_cnt, fp_object_cnt, cur_it, cur_epoch=cur_epoch, method=aggre_method,
                sign=attr_sign)
            return tp_XC, tp_far_attr, tp_pap, fp_XC, fp_far_attr, fp_pap
        else:
            XC, far_attr, pap = myExplainer.compute_xc(
                batch, object_cnt, tp_object_cnt, fp_object_cnt, cur_it, cur_epoch=cur_epoch, method=aggre_method,
                sign=attr_sign)
            return XC, far_attr, pap
    else:
        pap = myExplainer.compute_PAP(
            batch, object_cnt, tp_object_cnt, fp_object_cnt, cur_it, cur_epoch=cur_epoch, sign=attr_sign)
        return pap
