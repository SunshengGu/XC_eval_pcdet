import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import json
import matplotlib.patches as patches
import os
from pcdet.datasets.kitti.kitti_dataset import KittiDataset
from math import pi, cos, sin
import copy

import yaml
from pathlib import Path
from easydict import EasyDict

import time
import datetime

import csv

val_set_size = 3769
result_path = '/root/pcdet/output/kitti_models/pointpillar/default/eval/epoch_2/val/default/result.pkl'
csv_name = '/root/pcdet/tools/val_frame_idx.csv'

def main():
    lidar_results = np.load(result_path, allow_pickle=True)
    field_name = ['val_batch_id', 'orig_frame_id']
    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=field_name)
        writer.writeheader()
        for i in range(val_set_size):
            data_dict = {'val_batch_id': i, 'orig_frame_id': lidar_results[i]['frame_id']}
            writer.writerow(data_dict)


if __name__ == '__main__':
    main()