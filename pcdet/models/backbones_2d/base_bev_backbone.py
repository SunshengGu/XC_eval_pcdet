import torch
import torch.nn as nn


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == \
               len(self.model_cfg.NUM_FILTERS) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
        layer_nums = self.model_cfg.LAYER_NUMS
        layer_strides = self.model_cfg.LAYER_STRIDES
        num_filters = self.model_cfg.NUM_FILTERS
        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))

            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    num_filters[idx], num_upsample_filters[idx],
                    upsample_strides[idx],
                    stride=upsample_strides[idx], bias=False
                ),
                nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, tensor_values, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        # tensor_values is just for compatibility with Captum, only useful when in explain mode
        spatial_features = data_dict['spatial_features']
        print('\n shape of spatial_features in base_bev_backbone.py')
        print(spatial_features.shape)
        ups = []
        ret_dict = {}
        x = spatial_features
        # TODO: use tensor_values instead of x when in explain mode
        if str(type(tensor_values)) == 'torch.Tensor': # not a dummy tensor
            for i in range(len(self.blocks)):
                tensor_values = self.blocks[i](tensor_values)

                stride = int(spatial_features.shape[2] / tensor_values.shape[2])
                ret_dict['spatial_features_%dtensor_values' % stride] = tensor_values
                ups.append(self.deblocks[i](tensor_values))

            if len(ups) > 1:
                tensor_values = torch.cat(ups, dim=1)
            else:
                tensor_values = ups[0]
            if len(self.deblocks) > len(self.blocks):
                tensor_values = self.deblocks[-1](tensor_values)
            data_dict['spatial_features_2d'] = tensor_values
            return tensor_values, data_dict

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            ups.append(self.deblocks[i](x))

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        else:
            x = ups[0]
        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return tensor_values, data_dict
