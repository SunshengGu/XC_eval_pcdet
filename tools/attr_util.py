from XAI_utils.bbox_utils import *
from attr_generator_train import AttributionGeneratorTrain
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
    if explainer['method'] == 'IG':
        attr_generator = IntegratedGradients(explained_model, multiply_by_inputs=explainer['mult_inputs'])
    PseudoImage2D = batch['spatial_features']
    top_confidence_ind = batch['anchor_selections'][0][0] # 0th frame in batch, 0th predicted box's anchor index
    top_confidence_label = batch['anchor_labels'][0][0]  # 0th frame in batch, 0th predicted box's anchor label
    target = (top_confidence_ind, top_confidence_label)
    print("\nexplanation target: {}\n".format(target))
    target = (0,0)
    batch_attributions = attr_generator.attribute(
        PseudoImage2D, baselines=PseudoImage2D * 0, target=target,
        additional_forward_args=batch, n_steps=explainer['steps_ig'],
        internal_batch_size=batch['batch_size'])
    print("\nIn tools/XAI_utils/attr_util.py, batch_attributions[0].shape: {}".format(batch_attributions[0].shape))
    return 0.00002

def attr_func(explained_model, explainer, batch):
    '''

    :param explained_model: the model being explained
    :param explainer: parameters for the attribution generator
    :param batch: the batch of input data we are explaining
    :return:
    '''
    attr_generator = None
    if explainer['method'] == 'IG':
        attr_generator = IntegratedGradients(explained_model, multiply_by_inputs=explainer['mult_inputs'])
    PseudoImage2D = batch['spatial_features']
    top_confidence_ind = batch['anchor_selections'][0][0] # 0th frame in batch, 0th predicted box's anchor index
    top_confidence_label = batch['anchor_labels'][0][0]  # 0th frame in batch, 0th predicted box's anchor label
    target = (top_confidence_ind, top_confidence_label)
    print("\nexplanation target: {}\n".format(target))
    target = (0,0)
    batch_attributions = attr_generator.attribute(
        PseudoImage2D, baselines=PseudoImage2D * 0, target=target,
        additional_forward_args=batch, n_steps=explainer['steps_ig'],
        internal_batch_size=batch['batch_size'])
    print("\nIn tools/XAI_utils/attr_util.py, batch_attributions[0].shape: {}".format(batch_attributions[0].shape))
    return 0.00002