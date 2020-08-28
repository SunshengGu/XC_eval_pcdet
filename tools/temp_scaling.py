from pcdet.datasets.cadc.cadc_dataset import CadcDataset
from pcdet.config import cfg, cfg_from_yaml_file
from tools.eval_utils.ece import ece
import torch

import numpy as np


"""
Note that in order to use temperature scaling, the scores for all classes needs to be
outputed. This can be done by modifying the post_processing function in 
pcdet/models/detectors/detector3d_template.py.
"""

def calibrate(config_path, result_path):
    preds = np.load(result_path, allow_pickle=True)
    config = cfg_from_yaml_file(config_path, cfg)
    gts = get_cadc_gt_infos(config)

    index_to_class = ['Car', 'Pedestrian', 'Truck']

    results = []
    best, best_T = 1, None
    for _ in range(20):
        T = np.random.uniform(low=0.01, high=10)
        for i in range(len(preds)):
            score_all = preds[i]['score_all']
            label_idx, score = temp_scale(score_all, T)
            labels = [index_to_class[int(j)] for j in label_idx]

            preds[i]['score'] = np.array(score)
            preds[i]['label'] = np.array(labels)

        acc, ECE = ece(gts, preds, iou_thres=0.6)
        results.append((T, ECE))
        if ECE < best:
            best, best_T = ECE, T

    print('(Temperature, ECE) pairs:')
    print(results)
    print('Best T is {}, corresponding ECE is {}'.format(best_T, best))

def temp_scale(raw_scores, T, act_func='sigmoid'):
    raw_scores = torch.Tensor(raw_scores)
    if act_func == 'softmax':
        softmax = torch.softmax(raw_scores / T, dim=-1)
        score, label_idx = torch.max(softmax, dim=-1)
    elif act_func == 'sigmoid':
        sigmoid = torch.sigmoid(raw_scores / T)
        score, label_idx = torch.max(sigmoid, dim=-1)
    else:
        raise NameError("Activation function should be 'sigmoid' or 'softmax', you entered {}.".format(act_func))
    return label_idx.cpu().numpy(), score.cpu().numpy()


def get_cadc_gt_infos(cfg):
    dataset_cfg = cfg.DATA_CONFIG
    class_names = cfg.CLASS_NAMES
    cadc = CadcDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        training=False
    )
    gt_infos = []
    for info in cadc.cadc_infos:
        box3d_lidar = np.array(info['annos']['gt_boxes_lidar'])
        labels = np.array(info['annos']['name'])

        relevant_objs = [i for i in range(len(labels)) if labels[i] in cfg.CLASS_NAMES]
        relevant_objs = np.array(relevant_objs)

        # In CADC, z-axis of the gt is at the center of the centroid;
        # z-axis of the pred is at the bottom of the centroid.
        # Subtracting z-axis of the gt by half of its height to make the
        # the z-axis aligned.
        box3d_lidar[:,2] -= box3d_lidar[:,5] / 2
        gt_info = {
            'box' : box3d_lidar[relevant_objs],
            'label' : info['annos']['name'][relevant_objs],
        }
        gt_infos.append(gt_info)
    return gt_infos

def main():
    config_path = "/root/pcdet/tools/cfgs/cadc_models/pointpillar.yaml"
    result_path = "/root/pcdet/output/cadc_models/pointpillar/default/eval/epoch_80/val/default/result.pkl"
    calibrate(config_path, result_path)

if __name__ == "__main__":
    main()