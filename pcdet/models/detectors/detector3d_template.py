import os

import torch
import torch.nn as nn
import copy

from ...ops.iou3d_nms import iou3d_nms_utils
from .. import backbones_2d, backbones_3d, dense_heads, roi_heads
from ..backbones_2d import map_to_bev
from ..backbones_3d import pfe, vfe
from ..model_utils import model_nms_utils


class Detector3DTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__()
        # self.explain = explain # additional argument indicating if we are in explain mode
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.dataset = dataset
        self.class_names = dataset.class_names
        self.register_buffer('global_step', torch.LongTensor(1).zero_())

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head', 'point_head', 'roi_head'
        ]

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        # # ********** debug message **************
        # print("\n keys in self.state_dict before model is built:")
        # for key in self.state_dict():
        #     print(key)
        # # ********** debug message **************
        model_info_dict = {
            'module_list': [],
            'num_rawpoint_features': self.dataset.point_feature_encoder.num_point_features,
            'num_point_features': self.dataset.point_feature_encoder.num_point_features,
            'grid_size': self.dataset.grid_size,
            'point_cloud_range': self.dataset.point_cloud_range,
            'voxel_size': self.dataset.voxel_size
        }

        # # ********** debug message **************
        # print("showing module names in self.module_topology")
        # for mod_name in self.module_topology:
        #     print(mod_name)
        # # ********** debug message **************
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
            self.add_module(module_name, module)
        # # ********** debug message **************
        # print("\n keys in self.state_dict after model is built:")
        # for key in self.state_dict():
        #     print(key)
        # # ********** debug message **************
        return model_info_dict['module_list']

    def build_vfe(self, model_info_dict):
        if self.model_cfg.get('VFE', None) is None:
            # ********** debug message **************
            # print('\n no VFE')
            # ********** debug message **************
            return None, model_info_dict

        vfe_module = vfe.__all__[self.model_cfg.VFE.NAME](
            model_cfg=self.model_cfg.VFE,
            num_point_features=model_info_dict['num_rawpoint_features'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            voxel_size=model_info_dict['voxel_size']
        )
        model_info_dict['num_point_features'] = vfe_module.get_output_feature_dim()
        model_info_dict['module_list'].append(vfe_module)
        return vfe_module, model_info_dict

    def build_backbone_3d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_3D', None) is None:
            # ********** debug message **************
            # print('\n no 3D backbone')
            # ********** debug message **************
            return None, model_info_dict

        backbone_3d_module = backbones_3d.__all__[self.model_cfg.BACKBONE_3D.NAME](
            model_cfg=self.model_cfg.BACKBONE_3D,
            input_channels=model_info_dict['num_point_features'],
            grid_size=model_info_dict['grid_size'],
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range']
        )
        model_info_dict['module_list'].append(backbone_3d_module)
        model_info_dict['num_point_features'] = backbone_3d_module.num_point_features
        return backbone_3d_module, model_info_dict

    def build_map_to_bev_module(self, model_info_dict):
        if self.model_cfg.get('MAP_TO_BEV', None) is None:
            # ********** debug message **************
            # print('\n no map_to_bev_module')
            # ********** debug message **************
            return None, model_info_dict

        map_to_bev_module = map_to_bev.__all__[self.model_cfg.MAP_TO_BEV.NAME](
            model_cfg=self.model_cfg.MAP_TO_BEV,
            grid_size=model_info_dict['grid_size']
        )
        model_info_dict['module_list'].append(map_to_bev_module)
        model_info_dict['num_bev_features'] = map_to_bev_module.num_bev_features
        return map_to_bev_module, model_info_dict

    def build_backbone_2d(self, model_info_dict):
        if self.model_cfg.get('BACKBONE_2D', None) is None:
            # ********** debug message **************
            # print('\n no 2D backbone')
            # ********** debug message **************
            return None, model_info_dict
        if 'num_bev_features' not in model_info_dict:
            model_info_dict['num_bev_features'] = 64
        backbone_2d_module = backbones_2d.__all__[self.model_cfg.BACKBONE_2D.NAME](
            model_cfg=self.model_cfg.BACKBONE_2D,
            input_channels=model_info_dict['num_bev_features']
            # # TODO: hard code just for the sake of building a simpler pointpillar, need to change back later
            # input_channels=64
        )
        model_info_dict['module_list'].append(backbone_2d_module)
        model_info_dict['num_bev_features'] = backbone_2d_module.num_bev_features
        return backbone_2d_module, model_info_dict

    def build_pfe(self, model_info_dict):
        if self.model_cfg.get('PFE', None) is None:
            # ********** debug message **************
            # print('\n no pfe')
            # ********** debug message **************
            return None, model_info_dict

        pfe_module = pfe.__all__[self.model_cfg.PFE.NAME](
            model_cfg=self.model_cfg.PFE,
            voxel_size=model_info_dict['voxel_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            num_bev_features=model_info_dict['num_bev_features'],
            num_rawpoint_features=model_info_dict['num_rawpoint_features']
        )
        model_info_dict['module_list'].append(pfe_module)
        model_info_dict['num_point_features'] = pfe_module.num_point_features
        model_info_dict['num_point_features_before_fusion'] = pfe_module.num_point_features_before_fusion
        return pfe_module, model_info_dict

    def build_dense_head(self, model_info_dict):
        if self.model_cfg.get('DENSE_HEAD', None) is None:
            return None, model_info_dict
        dense_head_module = dense_heads.__all__[self.model_cfg.DENSE_HEAD.NAME](
            model_cfg=self.model_cfg.DENSE_HEAD,
            input_channels=model_info_dict['num_bev_features'],
            num_class=self.num_class if not self.model_cfg.DENSE_HEAD.CLASS_AGNOSTIC else 1,
            class_names=self.class_names,
            grid_size=model_info_dict['grid_size'],
            point_cloud_range=model_info_dict['point_cloud_range'],
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )
        model_info_dict['module_list'].append(dense_head_module)
        return dense_head_module, model_info_dict

    def build_point_head(self, model_info_dict):
        if self.model_cfg.get('POINT_HEAD', None) is None:
            return None, model_info_dict

        if self.model_cfg.POINT_HEAD.get('USE_POINT_FEATURES_BEFORE_FUSION', False):
            num_point_features = model_info_dict['num_point_features_before_fusion']
        else:
            num_point_features = model_info_dict['num_point_features']

        point_head_module = dense_heads.__all__[self.model_cfg.POINT_HEAD.NAME](
            model_cfg=self.model_cfg.POINT_HEAD,
            input_channels=num_point_features,
            num_class=self.num_class if not self.model_cfg.POINT_HEAD.CLASS_AGNOSTIC else 1,
            predict_boxes_when_training=self.model_cfg.get('ROI_HEAD', False)
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def build_roi_head(self, model_info_dict):
        if self.model_cfg.get('ROI_HEAD', None) is None:
            return None, model_info_dict
        point_head_module = roi_heads.__all__[self.model_cfg.ROI_HEAD.NAME](
            model_cfg=self.model_cfg.ROI_HEAD,
            input_channels=model_info_dict['num_point_features'],
            num_class=self.num_class if not self.model_cfg.ROI_HEAD.CLASS_AGNOSTIC else 1,
        )

        model_info_dict['module_list'].append(point_head_module)
        return point_head_module, model_info_dict

    def forward(self, **kwargs):
        raise NotImplementedError

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds

            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = torch.sigmoid(cls_preds)
            else:
                cls_preds = [x[batch_mask] for x in batch_dict['batch_cls_preds']]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [torch.sigmoid(x) for x in cls_preds]

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [torch.arange(1, self.num_class, device=cls_preds[0].device)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']

                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = model_nms_utils.multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=post_process_cfg.NMS_CONFIG,
                        score_thresh=post_process_cfg.SCORE_THRESH
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = torch.cat(pred_scores, dim=0)
                final_labels = torch.cat(pred_labels, dim=0)
                final_boxes = torch.cat(pred_boxes, dim=0)
            else:
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                if batch_dict.get('has_class_labels', False):
                    label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                    label_preds = batch_dict[label_key][index]
                else:
                    label_preds = label_preds + 1
                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )

                if post_process_cfg.OUTPUT_RAW_SCORE:
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict

    def post_processing_xai(self, tensor_values, batch_dict, box_limit=30):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:
        :param box_limit:
        :param batch_dict:
        :param tensor_values:

        """
        # print('\n starting the post_processing() function')
        # tensor_values is just for compatibility with Captum, only useful when in explain mode
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        boxes_with_cls_scores = []
        # all_anchor_boxes = []
        # boxes_params = []
        anchor_selections = []
        batch_dict['box_count'] = {}  # store the number of boxes for each image in the sample
        batch_dict['sigmoid_anchor_scores'] = []
        output_anchor = post_process_cfg.OUTPUT_ANCHOR_BOXES  # indicates if we output anchor boxes
        anchor_scores = []  # store class scores for individual anchor boxes
        anchor_boxes = []
        anchor_labels = []
        # max_box_ind = 0 # index of the input in the batch with most number of boxes
        max_num_boxes = box_limit
        for index in range(batch_size):
            # the 'None' here just means return None if key not found
            if batch_dict.get('batch_index', None) is not None:
                # print('\n batch_dict has the \'bactch_index\' entry!')
                # print('\n shape of batch_dict[\'batch_cls_preds\']' + str(batch_dict['batch_cls_preds'].shape))
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                # print('\n batch_dict does NOT have the \'bactch_index\' entry!')
                # print('\n shape of batch_dict[\'batch_cls_preds\']' + str(batch_dict['batch_cls_preds'].shape))
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index

            # inside the for loop, we only care about one particular sample, not the entire mini-batch
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            cls_preds = batch_dict['batch_cls_preds'][batch_mask]

            if str(type(tensor_values)) == 'torch.Tensor':
                cls_preds = tensor_values[batch_mask]

            src_cls_preds = cls_preds
            src_box_preds = box_preds
            # print("src_box_preds.shape: {}".format(src_box_preds.shape))

            anchor_scores.append(src_cls_preds)
            anchor_boxes.append(src_box_preds)
            # print('src_box_preds.shape before nms: {}'.format(src_box_preds.shape))
            # print('src_cls_preds.shape before nms: {}'.format(src_cls_preds.shape))
            # the second dimension of cls_preds should be the same as the number of classes
            assert cls_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)
                batch_dict['sigmoid_anchor_scores'].append(torch.sigmoid(src_cls_preds))
            else:
                batch_dict['sigmoid_anchor_scores'].append(src_cls_preds)

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                # in python, -1 means the last dimension
                # torch.max(input, dim, keepdim=False, out=None) returns a tuple:
                # 1. the maximum values in the indicated dimension
                # 2. the indices of the maximum values in the indicated dimension
                # now, for each box, we have a class prediction
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                # orig_label_preds = label_preds + 1
                # orig_cls_preds = cls_preds
                anchor_labels.append(label_preds)
                label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels',
                                                                                False) else label_preds + 1
                if batch_dict.get('has_class_labels', False):
                    print('\n no key named \'has_class_labels\' in batch_dict')
                # print('\n shape of label_preds after: ' + str(label_preds.shape))

                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
                anchor_selections.append(selected)
                # print("\nlen(selected): {}\n".format(len(selected)))
                if post_process_cfg.OUTPUT_RAW_SCORE:  # no need to worry about this, false by default
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores # this is the original code
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

                # # the following modifications did nothing
                # final_scores = orig_cls_preds[selected]
                # final_labels = orig_label_preds[selected]
                # final_boxes = src_box_preds[selected]

                # for label in final_labels:
                #     print('label is {}'.format(label))

                batch_dict['box_count'][index] = final_scores.shape[0]
                # if final_scores.shape[0] > max_num_boxes:
                #     max_box_ind = index
                #     max_num_boxes = final_scores.shape[0]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

            # print('src_cls_pred[selected] data type: ' + str(type(src_cls_preds[selected])))
            # print('src_cls_pred[selected] shape: ' + str(src_cls_preds[selected].shape))
            boxes_with_cls_scores.append(src_cls_preds[selected])
            # boxes_params.append(src_box_preds[selected])
        batch_dict['pred_dicts'] = pred_dicts
        batch_dict['recall_dict'] = recall_dict
        batch_dict['anchor_selections'] = anchor_selections
        # # note: torch.stack only works if every dimension except for dimension 0 matches
        # boxes_with_cls_scores = torch.stack(boxes_with_cls_scores)

        if output_anchor:
            anchor_scores = torch.stack(anchor_scores)
            batch_dict['anchor_scores'] = anchor_scores
            batch_dict['anchor_boxes'] = anchor_boxes
            batch_dict['anchor_labels'] = anchor_labels
            return anchor_scores

        # pad each output in the batch to match dimensions with the maximum length output
        # then stack the individual outputs together to get a tensor as the batch outout
        for i in range(len(boxes_with_cls_scores)):
            if boxes_with_cls_scores[i].shape[0] > max_num_boxes:
                # more than max_num_boxes boxes detected
                boxes_with_cls_scores[i] = boxes_with_cls_scores[i][:max_num_boxes]
            elif boxes_with_cls_scores[i].shape[0] < max_num_boxes:
                # less than max_num_boxes boxes detected
                padding_size = max_num_boxes - boxes_with_cls_scores[i].shape[0]
                padding = torch.zeros(padding_size, 3)
                padding = padding.float().cuda()  # load `padding` to GPU
                boxes_with_cls_scores[i] = torch.cat((boxes_with_cls_scores[i], padding), 0)
            else:
                continue
        boxes_with_cls_scores = torch.stack(boxes_with_cls_scores)
        # boxes_params = torch.stack(boxes_params)
        # print('\n finishing the post_processing() function')
        return boxes_with_cls_scores

    def post_processing_tensor(self, tensor_values, batch_dict, box_limit=30):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:
        :param box_limit:
        :param batch_dict:
        :param tensor_values:

        """
        # print('\n starting the post_processing() function')
        # tensor_values is just for compatibility with Captum, only useful when in explain mode
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        boxes_with_cls_scores = []
        # all_anchor_boxes = []
        # boxes_params = []
        anchor_selections = []
        batch_dict['box_count'] = {}  # store the number of boxes for each image in the sample
        batch_dict['sigmoid_anchor_scores'] = []
        output_anchor = post_process_cfg.OUTPUT_ANCHOR_BOXES  # indicates if we output anchor boxes
        anchor_scores = []  # store class scores for individual anchor boxes
        anchor_boxes = []
        anchor_labels = []
        # max_box_ind = 0 # index of the input in the batch with most number of boxes
        max_num_boxes = box_limit
        for index in range(batch_size):
            # the 'None' here just means return None if key not found
            if batch_dict.get('batch_index', None) is not None:
                # print('\n batch_dict has the \'bactch_index\' entry!')
                # print('\n shape of batch_dict[\'batch_cls_preds\']' + str(batch_dict['batch_cls_preds'].shape))
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                # print('\n batch_dict does NOT have the \'bactch_index\' entry!')
                # print('\n shape of batch_dict[\'batch_cls_preds\']' + str(batch_dict['batch_cls_preds'].shape))
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index

            # inside the for loop, we only care about one particular sample, not the entire mini-batch
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            cls_preds = batch_dict['batch_cls_preds'][batch_mask]

            if str(type(tensor_values)) == 'torch.Tensor':
                cls_preds = tensor_values[batch_mask]

            src_cls_preds = cls_preds
            src_box_preds = box_preds
            # print("src_box_preds.shape: {}".format(src_box_preds.shape))

            anchor_scores.append(src_cls_preds)
            anchor_boxes.append(src_box_preds)
            # print('src_box_preds.shape before nms: {}'.format(src_box_preds.shape))
            # print('src_cls_preds.shape before nms: {}'.format(src_cls_preds.shape))
            # the second dimension of cls_preds should be the same as the number of classes
            assert cls_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)
                batch_dict['sigmoid_anchor_scores'].append(torch.sigmoid(src_cls_preds))
            else:
                batch_dict['sigmoid_anchor_scores'].append(src_cls_preds)

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                # in python, -1 means the last dimension
                # torch.max(input, dim, keepdim=False, out=None) returns a tuple:
                # 1. the maximum values in the indicated dimension
                # 2. the indices of the maximum values in the indicated dimension
                # now, for each box, we have a class prediction
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                # orig_label_preds = label_preds + 1
                # orig_cls_preds = cls_preds
                anchor_labels.append(label_preds)
                label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels',
                                                                                False) else label_preds + 1
                if batch_dict.get('has_class_labels', False):
                    print('\n no key named \'has_class_labels\' in batch_dict')
                # print('\n shape of label_preds after: ' + str(label_preds.shape))

                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
                anchor_selections.append(selected)

                if post_process_cfg.OUTPUT_RAW_SCORE:  # no need to worry about this, false by default
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]
                # print("len(selected): {}".format(len(selected)))
                final_scores = selected_scores # this is the original code
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]
                # print("len(final_labels): {}".format(len(final_labels)))
                # print("len(final_scores): {}".format(len(final_scores)))

                batch_dict['box_count'][index] = final_scores.shape[0]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

            # print('src_cls_pred[selected] data type: ' + str(type(src_cls_preds[selected])))
            # print('src_cls_pred[selected] shape: ' + str(src_cls_preds[selected].shape))
            boxes_with_cls_scores.append(src_cls_preds[selected])
            # boxes_params.append(src_box_preds[selected])
        batch_dict['pred_dicts'] = pred_dicts
        batch_dict['recall_dict'] = recall_dict
        batch_dict['anchor_selections'] = anchor_selections
        # # note: torch.stack only works if every dimension except for dimension 0 matches
        # boxes_with_cls_scores = torch.stack(boxes_with_cls_scores)

        anchor_scores = torch.stack(anchor_scores)
        batch_dict['anchor_scores'] = anchor_scores
        batch_dict['anchor_boxes'] = anchor_boxes
        batch_dict['anchor_labels'] = anchor_labels
        return pred_dicts, recall_dict

    def post_processing_v2(self, batch_dict, box_limit=30):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                roi_labels: (B, num_rois)  1 .. num_classes
        Returns:
        :param box_limit:
        :param batch_dict:
        :param tensor_values:

        """
        # print('\n starting the post_processing() function')
        # tensor_values is just for compatibility with Captum, only useful when in explain mode
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        boxes_with_cls_scores = []
        # all_anchor_boxes = []
        # boxes_params = []
        anchor_selections = []
        batch_dict['box_count'] = {}  # store the number of boxes for each image in the sample
        batch_dict['sigmoid_anchor_scores'] = []
        output_anchor = post_process_cfg.OUTPUT_ANCHOR_BOXES  # indicates if we output anchor boxes
        anchor_scores = []  # store class scores for individual anchor boxes
        anchor_boxes = []
        anchor_labels = []
        # max_box_ind = 0 # index of the input in the batch with most number of boxes
        max_num_boxes = box_limit
        for index in range(batch_size):
            # the 'None' here just means return None if key not found
            if batch_dict.get('batch_index', None) is not None:
                # print('\n batch_dict has the \'bactch_index\' entry!')
                # print('\n shape of batch_dict[\'batch_cls_preds\']' + str(batch_dict['batch_cls_preds'].shape))
                assert batch_dict['batch_cls_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                # print('\n batch_dict does NOT have the \'bactch_index\' entry!')
                # print('\n shape of batch_dict[\'batch_cls_preds\']' + str(batch_dict['batch_cls_preds'].shape))
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index

            # inside the for loop, we only care about one particular sample, not the entire mini-batch
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            cls_preds = batch_dict['batch_cls_preds'][batch_mask]

            src_cls_preds = cls_preds
            src_box_preds = box_preds
            # print("src_box_preds.shape: {}".format(src_box_preds.shape))

            anchor_scores.append(src_cls_preds)
            anchor_boxes.append(src_box_preds)
            # print('src_box_preds.shape before nms: {}'.format(src_box_preds.shape))
            # print('src_cls_preds.shape before nms: {}'.format(src_cls_preds.shape))
            # the second dimension of cls_preds should be the same as the number of classes
            assert cls_preds.shape[1] in [1, self.num_class]

            if not batch_dict['cls_preds_normalized']:
                cls_preds = torch.sigmoid(cls_preds)
                batch_dict['sigmoid_anchor_scores'].append(torch.sigmoid(src_cls_preds))
            else:
                batch_dict['sigmoid_anchor_scores'].append(src_cls_preds)

            if post_process_cfg.NMS_CONFIG.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                # in python, -1 means the last dimension
                # torch.max(input, dim, keepdim=False, out=None) returns a tuple:
                # 1. the maximum values in the indicated dimension
                # 2. the indices of the maximum values in the indicated dimension
                # now, for each box, we have a class prediction
                cls_preds, label_preds = torch.max(cls_preds, dim=-1)
                # orig_label_preds = label_preds + 1
                # orig_cls_preds = cls_preds
                anchor_labels.append(label_preds)
                label_preds = batch_dict['roi_labels'][index] if batch_dict.get('has_class_labels',
                                                                                False) else label_preds + 1
                if batch_dict.get('has_class_labels', False):
                    print('\n no key named \'has_class_labels\' in batch_dict')
                # print('\n shape of label_preds after: ' + str(label_preds.shape))

                selected, selected_scores = model_nms_utils.class_agnostic_nms(
                    box_scores=cls_preds, box_preds=box_preds,
                    nms_config=post_process_cfg.NMS_CONFIG,
                    score_thresh=post_process_cfg.SCORE_THRESH
                )
                # print("\nlen(selected): {}\n".format(len(selected)))
                anchor_selections.append(selected)

                if post_process_cfg.OUTPUT_RAW_SCORE:  # no need to worry about this, false by default
                    max_cls_preds, _ = torch.max(src_cls_preds, dim=-1)
                    selected_scores = max_cls_preds[selected]

                final_scores = selected_scores # this is the original code
                final_labels = label_preds[selected]
                final_boxes = box_preds[selected]

                batch_dict['box_count'][index] = final_scores.shape[0]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

            # print('src_cls_pred[selected] data type: ' + str(type(src_cls_preds[selected])))
            # print('src_cls_pred[selected] shape: ' + str(src_cls_preds[selected].shape))
            boxes_with_cls_scores.append(src_cls_preds[selected])
            # boxes_params.append(src_box_preds[selected])
        batch_dict['pred_dicts'] = pred_dicts
        batch_dict['recall_dict'] = recall_dict
        batch_dict['anchor_selections'] = anchor_selections
        # # note: torch.stack only works if every dimension except for dimension 0 matches
        # boxes_with_cls_scores = torch.stack(boxes_with_cls_scores)

        if output_anchor:
            anchor_scores = torch.stack(anchor_scores)
            batch_dict['anchor_scores'] = anchor_scores
            batch_dict['anchor_boxes'] = anchor_boxes
            batch_dict['anchor_labels'] = anchor_labels
            return anchor_scores

        # pad each output in the batch to match dimensions with the maximum length output
        # then stack the individual outputs together to get a tensor as the batch outout
        for i in range(len(boxes_with_cls_scores)):
            if boxes_with_cls_scores[i].shape[0] > max_num_boxes:
                # more than max_num_boxes boxes detected
                boxes_with_cls_scores[i] = boxes_with_cls_scores[i][:max_num_boxes]
            elif boxes_with_cls_scores[i].shape[0] < max_num_boxes:
                # less than max_num_boxes boxes detected
                padding_size = max_num_boxes - boxes_with_cls_scores[i].shape[0]
                padding = torch.zeros(padding_size, 3)
                padding = padding.float().cuda()  # load `padding` to GPU
                boxes_with_cls_scores[i] = torch.cat((boxes_with_cls_scores[i], padding), 0)
            else:
                continue
        boxes_with_cls_scores = torch.stack(boxes_with_cls_scores)
        # boxes_params = torch.stack(boxes_params)
        # print('\n finishing the post_processing() function')
        return boxes_with_cls_scores

    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = torch.zeros((0, cur_gt.shape[0]))

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(dim=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = box_preds.new_zeros(box_preds.shape[0])
        return recall_dict

    def load_params_from_file(self, filename, logger, to_cpu=False):
        # file name is a checkpoint file
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        model_state_disk = checkpoint['model_state']

        if 'version' in checkpoint:
            logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])

        # # ********** debug message **************
        # print("keys in self.state_dict before processing ckpt data")
        # for key in self.state_dict():
        #     print(key)
        # # ********** debug message **************

        update_model_state = {}
        for key, val in model_state_disk.items():
            if key in self.state_dict() and self.state_dict()[key].shape == model_state_disk[key].shape:
                update_model_state[key] = val
                # logger.info('Update weight %s: %s' % (key, str(val.shape)))

        state_dict = self.state_dict()
        state_dict.update(update_model_state)
        self.load_state_dict(state_dict)

        for key in state_dict:
            if key not in update_model_state:
                logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

        logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(self.state_dict())))

    def load_params_with_optimizer(self, filename, to_cpu=False, optimizer=None, logger=None):
        if not os.path.isfile(filename):
            raise FileNotFoundError

        logger.info('==> Loading parameters from checkpoint %s to %s' % (filename, 'CPU' if to_cpu else 'GPU'))
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch.load(filename, map_location=loc_type)
        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)

        self.load_state_dict(checkpoint['model_state'])

        if optimizer is not None:
            print("optimizer argument present!")
            if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
                logger.info('==> Loading optimizer parameters from checkpoint %s to %s'
                            % (filename, 'CPU' if to_cpu else 'GPU'))
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            else:
                print("optimizer_state NOT in checkpoint!")
                assert filename[-4] == '.', filename
                src_file, ext = filename[:-4], filename[-3:]
                optimizer_filename = '%s_optim.%s' % (src_file, ext)
                if os.path.exists(optimizer_filename):
                    print("optimizer_filename {} exists!".format(optimizer_filename))
                    optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                    optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])
                else:
                    print("optimizer_filename {} does not exist!".format(optimizer_filename))

        if 'version' in checkpoint:
            print('==> Checkpoint trained from version: %s' % checkpoint['version'])
        logger.info('==> Done')

        return it, epoch
