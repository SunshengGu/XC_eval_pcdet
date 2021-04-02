from .detector3d_template import Detector3DTemplate


class PointPillarXAINewLoss(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, tensor_values, batch_dict):
        # use batch_dict only when not in explanation mode
        for cur_module in self.module_list:
            tensor_values, batch_dict = cur_module(tensor_values, batch_dict)
        return self.post_processing_xai(tensor_values, batch_dict)

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict