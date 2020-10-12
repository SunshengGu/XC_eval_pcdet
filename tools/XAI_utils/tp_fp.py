import numpy as np
import csv


def get_gt_infos(cfg, dataset):
    '''
    :param dataset: dataset object
    :param cfg: object containing model config information
    :return: gt_infos--containing gt boxes that have labels corresponding to the classes of interest, as well as the
                labels themselves
    '''
    # dataset_cfg = cfg.DATA_CONFIG
    # class_names = cfg.CLASS_NAMES
    # kitti = KittiDataset(
    #     dataset_cfg=dataset_cfg,
    #     class_names=class_names,
    #     training=False
    # )
    gt_infos = []
    dataset_infos = []
    dataset_name = cfg.DATA_CONFIG.DATASET
    if dataset_name == 'CadcDataset':
        dataset_infos = dataset.cadc_infos
    elif dataset_name == 'KittiDataset':
        dataset_infos = dataset.kitti_infos
    for info in dataset_infos:
        box3d_lidar = np.array(info['annos']['gt_boxes_lidar'])
        labels = np.array(info['annos']['name'])

        interested = []
        for i in range(len(labels)):
            label = labels[i]
            if label in cfg.CLASS_NAMES:
                interested.append(i)
        interested = np.array(interested)

        # box3d_lidar[:,2] -= box3d_lidar[:,5] / 2
        gt_info = {
            'boxes': box3d_lidar[interested],
            'labels': info['annos']['name'][interested],
        }
        gt_infos.append(gt_info)
    return gt_infos


def write_to_csv(file_name, field_name, data_1, data_2, data_3):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=field_name)
        name1 = field_name[0]
        name2 = field_name[1]
        name3 = field_name[2]
        writer.writeheader()
        for i in range(len(data_1)):
            writer.writerow({name1 : data_1[i], name2 : data_2[i], name3 : data_3[i]})


if __name__ == '__main__':
    pass