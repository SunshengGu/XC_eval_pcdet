import numpy as np
# from attr_generator_train import *
import pickle
from attr_generator_tensor import *
from XAI_utils.tp_fp import get_gt_infos

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
    # dataset_name = 'KittiDataset'  # change to your dataset, follow naming convention in the config yaml files in PCDet
    method = 'IntegratedGradients'  # explanation method
    attr_shown = 'positive'  # show positive or negative attributions
    aggre_method = 'sum'
    check_all = True
    debugging = False

    # IG specific parameters
    mult_by_inputs = True  # whether to show attributions only at where some input exists
    steps = 24  # number of intermediate steps for IG

    # config file for the full PointPillars model
    model_cfg_file = 'cfgs/waymo_models/pointpillar_nick_xai.yaml'
    # model_cfg_file = 'cfgs/kitti_models/pointpillar.yaml'
    # model checkpoint, change to your checkpoint, make sure it's a well-trained model
    # model_ckpt = '../output/kitti_models/pointpillar/default/ckpt/pointpillar_7728.pth'
    model_ckpt = '../output/waymo_models/pointpillar/nick_models/ckpt/checkpoint_epoch_30.pth'
    # info_path = '../output/kitti_models/pointpillar/default/eval/epoch_2/val/default/result.pkl'
    num_batchs = 40

    '''________________________User Input End________________________'''
    # data set prepration, use the validation set (called 'test_set' here)
    # arguments for dataloader
    batch_size = 1
    workers = 2
    dist_test = False

    cfg_from_yaml_file(model_cfg_file, cfg)
    cfg.TAG = Path(model_cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(model_cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    # Create output directories, not very useful, just to be compatible with the load_params_from_file
    # method defined in pcdet/models/detectors/detector3d_template.py, which requires a logger
    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / 'default'
    print("\noutput_dir: {}\n".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir = output_dir / 'eval'
    eval_all = False
    eval_tag = 'default'

    if not eval_all:
        num_list = re.findall(r'\d+', model_ckpt) if model_ckpt is not None else []
        epoch_id = num_list[-1] if num_list.__len__() > 0 else 'no_number'
        eval_output_dir = eval_output_dir / ('epoch_%s' % epoch_id) / cfg.DATA_CONFIG.DATA_SPLIT['test']
    else:
        eval_output_dir = eval_output_dir / 'eval_all_default'

    if eval_tag is not None:
        eval_output_dir = eval_output_dir / eval_tag

    eval_output_dir.mkdir(parents=True, exist_ok=True)

    # Create logger, not very useful, just to be compatible with PCDet's structures
    log_file = eval_output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # Creating the dataset
    test_set, test_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=batch_size,
        logger=logger,
        dist=dist_test, workers=workers, training=False
    )

    # Build the model
    full_model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=test_set)
    full_model.load_params_from_file(filename=model_ckpt, logger=logger, to_cpu=False)
    full_model.cuda()
    full_model.eval()
    explained_model = full_model.forward_model2D

    gt_infos = get_gt_infos(cfg, test_set)
    infos = None
    # with open(info_path, 'rb') as f:
    #     infos = pickle.load(f)

    # myXCCalculator = AttributionGenerator(model_ckpt=ckpt, full_model_cfg_file=cfg_file,
    #                                       model_cfg_file=explained_cfg_file,
    #                                       data_set=test_set, xai_method=method, output_path="unspecified",
    #                                       ignore_thresh=0.0, debug=True)
    selection = "tp/fp_all"
    pred_score_file_name = output_dir / 'pts_{}.csv'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    pred_score_field_name = ['epoch', 'batch', 'tp/fp', 'pred_label', 'pred_score', 'dist', 'pts']
    if selection != "tp/fp" and selection != "tp" and selection != "tp/fp_all":
        pred_score_field_name = ['epoch', 'batch', 'pred_label', 'pred_score', 'xc', 'far_attr', 'pap']
    with open(pred_score_file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=pred_score_field_name)
        writer.writeheader()
    myXCCalculator = AttributionGeneratorTensor(
        explained_model, cfg.DATA_CONFIG.DATASET, cfg.CLASS_NAMES, method, None, gt_infos, infos=infos,
        pred_score_file_name=pred_score_file_name, pred_score_field_name=pred_score_field_name, dataset=test_set,
        pts_file_name=pred_score_file_name, pts_field_name=pred_score_field_name,
        score_thresh=cfg.MODEL.POST_PROCESSING.SCORE_THRESH, selection=selection, debug=debugging, full_model=full_model,
        margin=0.2, ignore_thresh=0.1)
    epoch_obj_cnt = {}
    epoch_tp_obj_cnt = {}
    epoch_fp_obj_cnt = {}
    for i in range(3):
        epoch_obj_cnt[i] = 0
        epoch_tp_obj_cnt[i] = 0
        epoch_fp_obj_cnt[i] = 0
    for batch_num, batch_dictionary in enumerate(test_loader):
        print("batch_num: {}".format(batch_num))
        if (not check_all) and batch_num >= num_batchs:
            break
        if (batch_num % 10 != 0):
            continue  # only analyze 10% of the dataset
        print("\n\nAnalyzing the {}th batch\n".format(batch_num))
        if selection == "tp/fp" or selection == "tp/fp_all":
            batch_tp_pts, batch_fp_pts, batch_tp_dist, batch_fp_dist = myXCCalculator.compute_pts(
                batch_dictionary, batch_num, epoch_obj_cnt, epoch_tp_obj_cnt, epoch_fp_obj_cnt)
            print("\nTP lidar points count for the batch:\n {}".format(batch_tp_pts))
            print("\nFP lidar points count for the batch:\n {}".format(batch_fp_pts))
            print("\nTP dist for the batch:\n {}".format(batch_tp_dist))
            print("\nFP dist for the batch:\n {}".format(batch_fp_dist))
        myXCCalculator.reset()


if __name__ == '__main__':
    main()
