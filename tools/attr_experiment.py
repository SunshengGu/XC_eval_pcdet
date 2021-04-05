import numpy as np
from attr_generator import *

# def get_PAP(pos_grad, neg_grad, sign):
#     grad = None
#     if sign == 'positive':
#         grad = pos_grad
#     elif sign == 'negative':
#         grad = neg_grad
#     diff_1 = grad[1:, :] - grad[:-1, :]
#     diff_2 = grad[:, 1:] - grad[:, :-1]
#     pap_loss = np.sum(np.abs(diff_1)) + np.sum(np.abs(diff_2))
#     return pap_loss

def main():
    '''________________________User Input Begin________________________'''
    dataset_name = 'KittiDataset'  # change to your dataset, follow naming convention in the config yaml files in PCDet
    method = 'IntegratedGradients'  # explanation method
    attr_shown = 'positive'  # show positive or negative attributions

    # IG specific parameters
    mult_by_inputs = True  # whether to show attributions only at where some input exists
    steps = 24  # number of intermediate steps for IG

    # config file for the full PointPillars model
    cfg_file = 'cfgs/kitti_models/pointpillar_xai.yaml'
    # config file for the truncated PointPillars model: 2D backbone and SSD
    explained_cfg_file = 'cfgs/kitti_models/pointpillar_2DBackbone_DetHead_xai.yaml'

    # model checkpoint, change to your checkpoint, make sure it's a well-trained model
    ckpt = '../output/kitti_models/pointpillar/default/ckpt/pointpillar_7728.pth'

    num_batchs = 13

    '''________________________User Input End________________________'''
    # data set prepration, use the validation set (called 'test_set' here)
    # arguments for dataloader
    batch_size = 1
    workers = 4
    dist_test = False
    cfg_from_yaml_file(explained_cfg_file, cfg)
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        dist=dist_test, workers=workers, training=False
    )

    myXCCalculator = AttributionGenerator(model_ckpt=ckpt, full_model_cfg_file=cfg_file,
                                          model_cfg_file=explained_cfg_file,
                                          data_set=test_set, xai_method=method, output_path="unspecified",
                                          ignore_thresh=0.0)

    for batch_num, batch_dictionary in enumerate(test_loader):
        if batch_num == num_batchs:
            break
        print("\nAnalyzing the {}th batch".format(batch_num))
        batch_XC, batch_far_attr, batch_total_pap = myXCCalculator.compute_xc(batch_dictionary, method="sum")
        myXCCalculator.reset()


if __name__ == '__main__':
    main()