import numpy as np
import torch.nn as nn
from .anchor_head_template import AnchorHeadTemplate


class AnchorHeadSingleXAI(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, tensor_values, data_dict):
        # tensor_values is just for compatibility with Captum, only useful when in explain mode
        spatial_features_2d = data_dict['spatial_features_2d']
        if str(type(tensor_values)) == 'torch.Tensor':
            spatial_features_2d = tensor_values

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        # print('cls_preds data type: ' + str(type(cls_preds)))
        # print('cls_preds shape: ' + str(cls_preds.shape))
        # print('cls_preds[0] shape: ' + str(cls_preds[0].shape))
        # print('box_preds data type: ' + str(type(box_preds)))
        # print('box_preds shape: ' + str(box_preds.shape))
        # print('box_preds[0] shape: ' + str(box_preds[0].shape))

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None

        if self.training:
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            # The update() method updates the dictionary with the elements from the another dictionary object or from
            # an iterable of key / value pairs.
            self.forward_ret_dict.update(targets_dict)
        '''
        # previous code:
        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
            # print('batch_cls_preds data type: ' + str(type(batch_cls_preds)))
            # print('batch_cls_preds shape: ' + str(batch_cls_preds.shape))
            # print('batch_cls_preds[0] shape: ' + str(batch_cls_preds[0].shape))
            # print('batch_box_preds data type: ' + str(type(batch_box_preds)))
            # print('batch_box_preds shape: ' + str(batch_box_preds.shape))
            # print('batch_box_preds[0] shape: ' + str(batch_box_preds[0].shape))
        '''

        # start of new code
        batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
            batch_size=data_dict['batch_size'],
            cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
        )
        data_dict['batch_cls_preds'] = batch_cls_preds
        data_dict['batch_box_preds'] = batch_box_preds
        data_dict['cls_preds_normalized'] = False
        # end of new code

        # if str(type(tensor_values)) == 'torch.Tensor':
        #     tensor_values = batch_cls_preds
        tensor_values = batch_cls_preds
        return tensor_values, data_dict
