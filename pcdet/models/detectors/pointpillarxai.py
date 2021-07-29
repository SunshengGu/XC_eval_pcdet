from .detector3d_template import Detector3DTemplate


class PointPillarXAI(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, tensor_values, batch_dict):
        # use batch_dict only when not in explanation mode
        for cur_module in self.module_list:
            tensor_values, batch_dict = cur_module(tensor_values, batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            # actually, using post_processing_v2 and post_processing_tensor lead to the same result here
            # also, disabling post processing does not change the results either
            self.post_processing_v2(batch_dict)
            return ret_dict, tb_dict, disp_dict
        else:
            print("\ncalling post_processing_tensor\n")
            pred_dicts, recall_dicts = self.post_processing_tensor(tensor_values, batch_dict)
            return pred_dicts, recall_dicts

    def forward_model2D(self, tensor_values, batch_dict):
        out_put, out_batch_dict = self.backbone_2d(tensor_values, batch_dict)
        out_put, out_batch_dict = self.dense_head(out_put, out_batch_dict)
        print("\ncalling post_processing_xai\n")
        out_put = self.post_processing_xai(out_put, batch_dict)
        return out_put

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
