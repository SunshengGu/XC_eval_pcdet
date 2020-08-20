from .detector3d_template import Detector3DTemplate


class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset, explain):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset, explain=explain)
        self.module_list = self.build_networks()

    def forward(self, tensor_values, batch_dict):
        # use batch_dict only when not in explanation mode
        for cur_module in self.module_list:
            tensor_values, batch_dict = cur_module(tensor_values, batch_dict)
        # if self.explain(): # in explain mode, need a tensor_values in tensor format to be compatible with Captum
        #     for cur_module in self.module_list:
        #         tensor_values, batch_dict = cur_module(tensor_values, batch_dict)
        # else: # in other modes, just use the batch_dict is sufficient
        #     for cur_module in self.module_list:
        #         batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            if self.explain: # in explain mode
                boxes_with_cls_scores = self.post_processing(tensor_values, batch_dict)
                return boxes_with_cls_scores
            else: # in test mode
                pred_dicts, recall_dicts = self.post_processing(tensor_values, batch_dict)
                return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
