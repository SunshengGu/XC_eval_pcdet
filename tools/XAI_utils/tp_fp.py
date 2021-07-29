import numpy as np
import csv
# import matplotlib.pyplot as plt


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
    elif dataset_name == 'WaymoDataset':
        dataset_infos = dataset.infos
    # print("dataset_infos: {}".format(dataset_infos))
    for info in dataset_infos:
        box3d_lidar = np.array(info['annos']['gt_boxes_lidar'])
        labels = np.array(info['annos']['name'])
        # print("info['annos']: ".format(info['annos']))
        interested = []
        for i in range(len(labels)):
            label = labels[i]
            if label in cfg.CLASS_NAMES:
                interested.append(i)
        interested = np.array(interested)
        print("len(box3d_lidar): {}".format(len(box3d_lidar)))
        print("len(labels): {}".format(len(labels)))
        print("len(interested): {}".format(len(interested)))
        print("type(interested): {}".format(type(interested)))
        
        # TODO: handle the case where len(interested) == 0
        if (len(interested) == 0):
            gt_info = {
                'boxes': [],
                'labels': [],
            }
        else:
            # box3d_lidar[:,2] -= box3d_lidar[:,5] / 2
            gt_info = {
                'boxes': box3d_lidar[interested],
                'labels': info['annos']['name'][interested],
            }
        gt_infos.append(gt_info)
    return gt_infos


def list_selection(input_list, selections):
    new_list = []
    for ind in selections:
        new_list.append(input_list[ind])
    return new_list


# def write_to_csv(file_name, field_name, data_1, data_2, data_3, data_4):
#     with open(file_name, 'w', newline='') as csvfile:
#         writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=field_name)
#         name1 = field_name[0]
#         name2 = field_name[1]
#         name3 = field_name[2]
#         name4 = field_name[3]
#         writer.writeheader()
#         for i in range(len(data_1)):
#             writer.writerow({name1: data_1[i], name2: data_2[i], name3: data_3[i], name4: data_4[i]})


def write_to_csv(file_name, field_name, data_list):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=field_name)
        writer.writeheader()
        for i in range(len(data_list[0])):
            data_dict = {}
            for cnt, name in enumerate(field_name):
                data_dict[name] = data_list[cnt][i]
            writer.writerow(data_dict)


def write_attr_to_csv(file_name, grad, box_vertices):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        if len(grad.shape)==3 and grad.shape[2]==64:
            pos_grad = np.sum((grad > 0) * grad, axis=2)
            neg_grad = np.sum((grad > 0) * grad, axis=2)
            for row in pos_grad:
                writer.writerow(row)
            for vert in box_vertices:
                writer.writerow(vert)
            for row in neg_grad:
                writer.writerow(row)
        else:
            for row in grad:
                writer.writerow(row)
            for vert in box_vertices:
                writer.writerow(vert)


if __name__ == '__main__':
    pass
