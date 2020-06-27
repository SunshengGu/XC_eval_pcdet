import os
import sys
import pickle
import copy
import numpy as np
import json
from skimage import io
from pathlib import Path
import torch
import spconv

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval

from pcdet.utils import box_utils, common_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.config import cfg
from pcdet.datasets.data_augmentation.dbsampler import DataBaseSampler
from pcdet.datasets import DatasetTemplate
from pcdet.datasets.nuscenes import nuscenes_calibration


class BaseNuScenesDataset(DatasetTemplate):
    NameMapping = {
        'movable_object.barrier': 'Barrier',
        'vehicle.bicycle': 'Bicycle',
        'vehicle.bus.bendy': 'Bus',
        'vehicle.bus.rigid': 'Bus',
        'vehicle.car': 'Car',
        'vehicle.construction': 'Construction_vehicle',
        'vehicle.motorcycle': 'Motorcycle',
        'human.pedestrian.adult': 'Pedestrian',
        'human.pedestrian.child': 'Pedestrian',
        'human.pedestrian.construction_worker': 'Pedestrian',
        'human.pedestrian.police_officer': 'Pedestrian',
        'movable_object.trafficcone': 'Traffic_cone',
        'vehicle.trailer': 'Trailer',
        'vehicle.truck': 'Truck'
    }
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.parked",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    
    def __init__(self, root_path, split='train', init_nusc=True):
        super().__init__()
        self.root_path = root_path
        self.split = split
        if (init_nusc):
            self.nusc = NuScenes(version='v1.0-trainval', dataroot=root_path, verbose=True)
            
        splits = create_splits_scenes()
        split_scenes = splits[split]
        all_scene_names = [scene['name'] for scene in self.nusc.scene]
        split_scene_tokens = [self.nusc.scene[all_scene_names.index(scene_name)]['token'] for scene_name in split_scenes]
        
        self.sample_id_list = self.get_sample_tokens_from_scenes(split_scene_tokens)
        
        try:
            self.max_sweeps = cfg.DATA_CONFIG.MAX_SWEEPS
        except:
            self.max_sweeps = 10
            
        self.with_velocity = True
        
    def get_sample_tokens_from_scenes(self, scene_tokens):
        """
        :param scene_tokens: List of scene tokens
        :ret: List of sample tokens
        """
        sample_tokens = []
        for token in scene_tokens:
            sample_tokens.append(self.nusc.get('scene', token)['first_sample_token'])
            sample = self.nusc.get('sample', sample_tokens[-1])
            while (sample['next'] != ''):
                sample_tokens.append(sample['next'])
                sample = self.nusc.get('sample', sample_tokens[-1])

        return sample_tokens
        
    
    def set_split(self, split):
        self.__init__(self.root_path, split, False)
    
    def get_lidar(self, sample_token):
        lidar_token = self.nusc.get('sample', sample_token)['data']['LIDAR_TOP']
        sample_data = self.nusc.get('sample_data', lidar_token)
        lidar_file = os.path.join(self.root_path, sample_data['filename'])
        assert os.path.exists(lidar_file)
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 5)
        points[:, 3] /= 255
        points[:, 4] = 0
        
        sweep_points_list = self.get_sweeps(sample_token, self.max_sweeps)
        sweep_points_list.append(points)
        points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]
        
        return points


    def get_sweeps(self, sample_token, max_sweeps):
        lidar_token = self.nusc.get('sample', sample_token)['data']['LIDAR_TOP']
        sample_data = self.nusc.get('sample_data', lidar_token)
        
        # Transforms for putting all seeps in the same frame
        cs_record = self.nusc.get('calibrated_sensor',
                             sample_data['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
        l2e_r = cs_record['rotation']
        l2e_t = cs_record['translation']
        e2g_r = pose_record['rotation']
        e2g_t = pose_record['translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        
        timestamp = sample_data['timestamp'] / 1e6
        sweep_points_list = []
        
        # Accumulate sweeps 
        while len(sweep_points_list) < max_sweeps:
            if sample_data['prev'] == "":
                break
        
            sample_data = self.nusc.get('sample_data', sample_data['prev'])
            cs_record = self.nusc.get('calibrated_sensor',
                                    sample_data['calibrated_sensor_token'])
            pose_record = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            lidar_path = self.nusc.get_sample_data_path(sample_data['token'])
            sweep_ts = sample_data["timestamp"]
            l2e_r_s = cs_record['rotation']
            l2e_t_s = cs_record['translation']
            e2g_r_s = pose_record['rotation']
            e2g_t_s = pose_record['translation']
            # sweep->ego->global->ego'->lidar
            l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
            e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

            R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
            sweep2lidar_rotation = R.T  # points @ R.T + T
            sweep2lidar_translation = T
            
            points_sweep = np.fromfile( str(lidar_path), dtype=np.float32, count=-1).reshape([-1, 5])
            points_sweep[:, 3] /= 255
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep2lidar_rotation.T
            points_sweep[:, :3] += sweep2lidar_translation
            points_sweep[:, 4] = timestamp - sweep_ts
            sweep_points_list.append(points_sweep)
            
        return sweep_points_list

    def get_image_shape(self, sample_token):
        img_token = self.nusc.get('sample', sample_token)['data']['CAM_FRONT']
        img_file = os.path.join(self.root_path, self.nusc.get('sample_data', img_token)['filename'])
        assert os.path.exists(img_file)
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, sample_token, sensor='LIDAR_TOP'):
        sensor_token = self.nusc.get('sample', sample_token)['data'][sensor]
        data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sensor_token)
        return boxes

    def get_calib(self, sample_token):
        cam_token = self.nusc.get('sample', sample_token)['data']['CAM_FRONT']
        lidar_token = self.nusc.get('sample', sample_token)['data']['LIDAR_TOP']
        cam_calibrated_token = self.nusc.get('sample_data', cam_token)['calibrated_sensor_token']
        lidar_calibrated_token = self.nusc.get('sample_data', lidar_token)['calibrated_sensor_token']
        ego_pose_token = self.nusc.get('sample_data', cam_token)['ego_pose_token']
            
        cam_calibrated =  self.nusc.get('calibrated_sensor', cam_calibrated_token)
        lidar_calibrated =  self.nusc.get('calibrated_sensor', lidar_calibrated_token)
        ego_pose =  self.nusc.get('ego_pose', ego_pose_token)
        
        return nuscenes_calibration.Calibration(ego_pose, cam_calibrated, lidar_calibrated)

    def get_road_plane(self, sample_token):
        """
        plane_file = os.path.join(self.root_path, 'planes', '%s.txt' % idx)
        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane
        """
        # Currently unsupported in NuScenes
        raise NotImplementedError

    def get_annotation_from_label(self, calib, sample_token):
        box_list = self.get_label(sample_token, sensor='LIDAR_TOP')
        if (len(box_list) == 0):
            annotations = {}
            annotations['name'] = annotations['num_points_in_gt'] = annotations['gt_boxes_lidar'] = \
                annotations['token'] = annotations['location'] = annotations['rotation_y'] = \
                annotations['dimensions'] = annotations['score'] = annotations['difficulty'] = \
                annotations['truncated'] = annotations['occluded'] = annotations['alpha'] = \
                annotations['bbox'] = annotations['gt_velocity'] = np.array([])
            return None

        annotations = {}
        gt_names = np.array([self.NameMapping[box.name] if box.name in self.NameMapping else 'DontCare' for box in box_list])
        num_points_in_gt = np.array([self.nusc.get('sample_annotation', box.token)['num_lidar_pts'] for box in box_list])
        
        loc_lidar = np.array([box.center for box in box_list]) 
        dims =  np.array([box.wlh for box in box_list]) 
        #loc_lidar[:,2] -= dims[:,2] / 2 # Translate true center to bottom middle coordinate
        rots = np.array([box.orientation.yaw_pitch_roll[0] for box in box_list])
        gt_boxes_lidar = np.concatenate([loc_lidar, dims, rots[..., np.newaxis]], axis=1)
        
        velocity_global = np.array([self.nusc.box_velocity(box.token)[:2] for box in box_list]).reshape(-1, 2) # x,y 
        nan_mask = np.isnan(velocity_global[:, 0])
        velocity_global[nan_mask] = [0.0, 0.0]
        gt_velocity = calib.velo_global_to_lidar(velocity_global)
        
        if self.with_velocity:
            gt_boxes_lidar = np.concatenate([gt_boxes_lidar, gt_velocity], axis=-1)
            
        annotations['name'] = gt_names
        annotations['num_points_in_gt'] = num_points_in_gt
        annotations['gt_boxes_lidar'] = gt_boxes_lidar
        annotations['gt_velocity'] = gt_velocity
        annotations['token'] = np.array([box.token for box in box_list])
        
        # in CAM_FRONT frame. Probably meaningless as most objects aren't in frame.
        annotations['location'] = calib.lidar_to_rect(loc_lidar) 
        annotations['rotation_y'] = rots
        annotations['dimensions'] = np.array([[box.wlh[1], box.wlh[2], box.wlh[0]] for box in box_list])  # lhw format
        
        occluded = np.zeros([num_points_in_gt.shape[0]])
        easy_mask = num_points_in_gt > 15
        moderate_mask = num_points_in_gt > 7
        occluded = np.zeros([num_points_in_gt.shape[0]])
        occluded[:] = 2
        occluded[moderate_mask] = 1
        occluded[easy_mask] = 0
        
        gt_boxes_camera = box_utils.boxes3d_lidar_to_camera(gt_boxes_lidar, calib)
        assert len(gt_boxes_camera) == len(gt_boxes_lidar) == len(box_list)
        # Currently unused for NuScenes, and don't make too much since as we primarily use 360 degree 3d LIDAR boxes.
        annotations['score'] = np.array([1 for _ in box_list])
        annotations['difficulty'] = np.array([0 for _ in box_list], np.int32)
        annotations['truncated'] = np.array([0 for _ in box_list])
        annotations['occluded'] = occluded
        annotations['alpha'] = np.array([-np.arctan2(-gt_boxes_lidar[i][1], gt_boxes_lidar[i][0]) + gt_boxes_camera[i][6] for i in range(len(gt_boxes_camera))]) 
        annotations['bbox'] = gt_boxes_camera
        
        return annotations
    
    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        '''
        Valid point should be in the image (and in the PC_AREA_SCOPE)
        :param pts_rect:
        :param img_shape:
        :return:
        '''
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag

    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_token):
            
            print('%s sample_token: %s ' % (self.split, sample_token))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_token}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_token, 'image_shape': self.get_image_shape(sample_token)}
            info['image'] = image_info
            calib = self.get_calib(sample_token)
            
            calib_info = {'T_IMG_CAM': calib.t_img_cam, 'T_CAR_CAM': calib.t_car_cam, 'T_CAR_LIDAR': calib.t_car_lidar, 'T_GLOBAL_CAR': calib.t_global_car}
            info['calib'] = calib_info

            if has_label:
                annotations = self.get_annotation_from_label(calib, sample_token)
                if (annotations == None):
                    return None
                info['annos'] = annotations
            return info

        # temp = process_single_scene(self.sample_id_list[0])
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        # Remove samples with no gt boxes
        infos = [sample for sample in infos if sample] 
            
        return list(infos)

    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('nuscenes_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']
            
            if (len(gt_boxes) == 0):
                continue
            
            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes[:,:7])
            ).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                if (used_classes is None) or names[i] in used_classes:
                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                               'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                               'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dict(input_dict, index, record_dict):
        # finally generate predictions.
        sample_idx = input_dict['sample_idx'][index] if 'sample_idx' in input_dict else -1
        boxes3d_lidar_preds = record_dict['boxes'].cpu().numpy()

        if boxes3d_lidar_preds.shape[0] == 0:
            return {'sample_idx': sample_idx}

        calib = input_dict['calib'][index]
        image_shape = input_dict['image_shape'][index]

        boxes3d_camera_preds = box_utils.boxes3d_lidar_to_camera(boxes3d_lidar_preds, calib)
        boxes2d_image_preds = box_utils.boxes3d_camera_to_imageboxes(boxes3d_camera_preds, calib,
                                                                     image_shape=image_shape)
        # predictions
        predictions_dict = {
            'bbox': boxes2d_image_preds,
            'box3d_camera': boxes3d_camera_preds,
            'box3d_lidar': boxes3d_lidar_preds,
            'scores': record_dict['scores'].cpu().numpy(),
            'label_preds': record_dict['labels'].cpu().numpy(),
            'sample_idx': sample_idx,
        }
        return predictions_dict

    @staticmethod
    def generate_annotations(input_dict, pred_dicts, class_names, save_to_file=False, output_dir=None):
        def get_empty_prediction():
            ret_dict = {
                'name': np.array([]), 'truncated': np.array([]), 'occluded': np.array([]),
                'alpha': np.array([]), 'bbox': np.zeros([0, 4]), 'dimensions': np.zeros([0, 3]),
                'location': np.zeros([0, 3]), 'rotation_y': np.array([]), 'score': np.array([]),
                'boxes_lidar': np.zeros([0, 7])
            }
            return ret_dict

        def generate_single_anno(idx, box_dict):
            num_example = 0
            if 'bbox' not in box_dict:
                return get_empty_prediction(), num_example

            sample_idx = box_dict['sample_idx']
            box_preds_image = box_dict['bbox']
            box_preds_camera = box_dict['box3d_camera']
            box_preds_lidar = box_dict['box3d_lidar']
            scores = box_dict['scores']
            label_preds = box_dict['label_preds']

            anno = {'name': [], 'truncated': [], 'occluded': [], 'alpha': [], 'bbox': [], 'dimensions': [],
                    'location': [], 'rotation_y': [], 'score': [], 'boxes_lidar': []}

            for box_camera, box_lidar, bbox, score, label in zip(box_preds_camera, box_preds_lidar, box_preds_image,
                                                                 scores, label_preds):

                if not (np.all(box_lidar[3:6] > -0.1)):
                    print('Invalid size(sample %s): ' % str(sample_idx), box_lidar)
                    continue

                anno['name'].append(class_names[int(label - 1)])
                anno['truncated'].append(0.0)
                anno['occluded'].append(0)
                anno['alpha'].append(-np.arctan2(-box_lidar[1], box_lidar[0]) + box_camera[6])
                anno['bbox'].append(bbox)
                anno['dimensions'].append(box_camera[3:6])
                anno['location'].append(box_camera[:3])
                anno['rotation_y'].append(box_camera[6])
                anno['score'].append(score)
                anno['boxes_lidar'].append(box_lidar)

                num_example += 1

            if num_example != 0:
                anno = {k: np.stack(v) for k, v in anno.items()}
            else:
                anno = get_empty_prediction()

            return anno, num_example

        annos = []
        for i, box_dict in enumerate(pred_dicts):
            sample_idx = box_dict['sample_idx']
            single_anno, num_example = generate_single_anno(i, box_dict)
            single_anno['num_example'] = num_example
            single_anno['sample_idx'] = np.array([sample_idx] * num_example)
            annos.append(single_anno)
            if save_to_file:
                cur_det_file = os.path.join(output_dir, '%s.json' % (sample_idx))
                boxes_lidar = single_anno['boxes_lidar'] # x y z w l h yaw
                pred_json = {}
                pred_json['cuboids'] = []
                for idx in range(len(boxes_lidar)):
                    data['cuboids'].append({
                        'label': single_anno['name'][idx],
                        'position': {
                            'x': boxes_lidar[idx][0],
                            'y': boxes_lidar[idx][1],
                            'z': boxes_lidar[idx][2],
                        },
                        'dimension': {
                            'x': boxes_lidar[idx][3],
                            'y': boxes_lidar[idx][4],
                            'z': boxes_lidar[idx][5],
                        },
                        "yaw": boxes_lidar[idx][6],
                        "score": single_anno['score'][idx]
                    })
                with open(cur_det_file, 'w') as f:
                    json.dump(pred_json, f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):

        eval_det_annos = copy.deepcopy(det_annos)
    
        # Create NuScenes JSON output file
        nusc_annos = {}
        for sample in eval_det_annos:
            try:
                sample_idx = sample['sample_idx'][0]
            except:
                continue
            
            sample_results = []
            
            calib = self.get_calib(sample_idx)
            
            sample['boxes_lidar'] = np.array(sample['boxes_lidar'])
            positions = sample['boxes_lidar'][:,:3]
            dimensions = sample['boxes_lidar'][:,3:6]
            rotations = sample['boxes_lidar'][:,6]
                
            velocities = [(np.nan, np.nan, np.nan) for _ in sample['boxes_lidar']]
            if sample['boxes_lidar'].shape[-1] == 9:
                velocities = sample['boxes_lidar'][:,7:9]
            
            for center, dimension, yaw, label, score, velocity in zip(positions, dimensions, rotations, sample['name'], sample['score'], velocities):
                
                quaternion = Quaternion(axis=[0, 0, 1], radians=yaw)
                
                box = Box(center, dimension, quaternion, velocity=(*velocity, 0.0))
                # Move box to ego vehicle coord system
                box.rotate(Quaternion(calib.lidar_calibrated['rotation']))
                box.translate(np.array(calib.lidar_calibrated['translation']))
                # Move box to global coord system
                box.rotate(Quaternion(calib.ego_pose['rotation']))
                box.translate(np.array(calib.ego_pose['translation']))
                
                if (float(score) < 0):
                    score = 0
                if (float(score) > 1):
                    score = 1
                if (label == 'Cyclist'):
                    label = 'bicycle'
                sample_results.append({
                    "sample_token": sample_idx,
                    "translation": box.center.tolist(), 
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "lidar_yaw": float(yaw),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": label.lower(),
                    "detection_score": float(score),
                    "attribute_name": self.DefaultAttribute[label.lower()],
                })
                
            nusc_annos[sample_idx] = sample_results
        
        for sample_id in self.sample_id_list:
            if sample_id not in nusc_annos:
                nusc_annos[sample_id] = []
        
        nusc_submission = {
            "meta": {
                "use_camera": False,
                "use_lidar": True,
                "use_radar": False,
                "use_map": False,
                "use_external": False,
            },
            "results": nusc_annos,
        }
        eval_file = os.path.join(kwargs['output_dir'], 'nusc_results.json')
        with open(eval_file, "w") as f:
            json.dump(nusc_submission, f, indent=2)
        
        # Call NuScenes evaluation
        cfg = config_factory('detection_cvpr_2019')
        nusc_eval = DetectionEval(self.nusc, config=cfg, result_path=eval_file, eval_set=self.split, 
                                output_dir=kwargs['output_dir'], verbose=True)
        metric_summary = nusc_eval.main(plot_examples = 10, render_curves=True)

        # Reformat the metrics summary a bit for the tensorboard logger
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        result = {}
        result['mean_ap'] = metric_summary['mean_ap']
        for tp_name, tp_val in metric_summary['tp_errors'].items():
            result[tp_name] = tp_val
            
        class_aps = metric_summary['mean_dist_aps']
        class_tps = metric_summary['label_tp_errors']
        for class_name in class_aps.keys():
            result['mAP_' + class_name] = class_aps[class_name]
            for key, val in err_name_mapping.items():
                result[val + '_' + class_name] = class_tps[class_name][key]
        
        return str(result), result

class NuScenesDataset(BaseNuScenesDataset):
    def __init__(self, root_path, class_names, split, training, logger=None):
        """
        :param root_path: NuScenes data path
        :param split:
        """
        super().__init__(root_path=root_path, split=split)

        self.class_names = class_names
        self.training = training
        self.logger = logger

        self.mode = 'TRAIN' if self.training else 'TEST'

        self.nuscenes_infos = []
        self.include_nuscenes_data(self.mode, logger)
        self.dataset_init(class_names, logger)

    def include_nuscenes_data(self, mode, logger):
        if cfg.LOCAL_RANK == 0 and logger is not None:
            logger.info('Loading NuScenes dataset')
        nuscenes_infos = []

        for info_path in cfg.DATA_CONFIG[mode].INFO_PATH:
            info_path = cfg.ROOT_DIR / info_path
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                nuscenes_infos.extend(infos)
        
        #if (self.mode == 'TRAIN'):
        #    print("Using subset of training data")
        #    nuscenes_infos = nuscenes_infos[::4]    
        
        self.nuscenes_infos.extend(nuscenes_infos)

        if cfg.LOCAL_RANK == 0 and logger is not None:
            logger.info('Total samples for NuScenes dataset: %d' % (len(nuscenes_infos)))

    def dataset_init(self, class_names, logger):
        self.db_sampler = None
        db_sampler_cfg = cfg.DATA_CONFIG.AUGMENTATION.DB_SAMPLER
        if self.training and db_sampler_cfg.ENABLED:
            db_infos = []
            for db_info_path in db_sampler_cfg.DB_INFO_PATH:
                db_info_path = cfg.ROOT_DIR / db_info_path
                with open(str(db_info_path), 'rb') as f:
                    infos = pickle.load(f)
                    if db_infos.__len__() == 0:
                        db_infos = infos
                    else:
                        [db_infos[cls].extend(infos[cls]) for cls in db_infos.keys()]

            self.db_sampler = DataBaseSampler(
                db_infos=db_infos, sampler_cfg=db_sampler_cfg, class_names=class_names, logger=logger
            )

        voxel_generator_cfg = cfg.DATA_CONFIG.VOXEL_GENERATOR

        # Support spconv 1.0 and 1.1
        points = np.zeros((1, 3))
        try:
            self.voxel_generator = spconv.utils.VoxelGenerator(
                voxel_size=voxel_generator_cfg.VOXEL_SIZE,
                point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                max_num_points=voxel_generator_cfg.MAX_POINTS_PER_VOXEL,
                max_voxels=cfg.DATA_CONFIG[self.mode].MAX_NUMBER_OF_VOXELS
            )
            voxels, coordinates, num_points = self.voxel_generator.generate(points)
        except:
            self.voxel_generator = spconv.utils.VoxelGeneratorV2(
                voxel_size=voxel_generator_cfg.VOXEL_SIZE,
                point_cloud_range=cfg.DATA_CONFIG.POINT_CLOUD_RANGE,
                max_num_points=voxel_generator_cfg.MAX_POINTS_PER_VOXEL,
                max_voxels=cfg.DATA_CONFIG[self.mode].MAX_NUMBER_OF_VOXELS
            )
            voxel_grid = self.voxel_generator.generate(points)


    def __len__(self):
        return len(self.nuscenes_infos)

    def __getitem__(self, index):
        # index = 4
        info = copy.deepcopy(self.nuscenes_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']

        points = self.get_lidar(sample_idx)
        calib = self.get_calib(sample_idx)

        img_shape = info['image']['image_shape']
        if cfg.DATA_CONFIG.FOV_POINTS_ONLY:
            pts_rect = calib.lidar_to_rect(points[:, 0:3])
            fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
            points = points[fov_flag]

        input_dict = {
            'points': points,
            'sample_idx': sample_idx,
            'calib': calib,
        }

        if 'annos' in info:
            annos = info['annos']
            #annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            bbox = annos['bbox']
            gt_boxes = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            if 'gt_boxes_lidar' in annos:
                gt_boxes_lidar = annos['gt_boxes_lidar']
            else:
                gt_boxes_lidar = box_utils.boxes3d_camera_to_lidar(gt_boxes, calib)

            input_dict.update({
                'gt_boxes': gt_boxes,
                'gt_names': gt_names,
                'gt_box2d': bbox,
                'gt_boxes_lidar': gt_boxes_lidar
            })

        example = self.prepare_data(input_dict=input_dict, has_label='annos' in info)

        example['sample_idx'] = sample_idx
        example['image_shape'] = img_shape

        return example


def create_nuscenes_infos(data_path, save_path, workers=4):
    dataset = BaseNuScenesDataset(root_path=data_path)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('nuscenes_infos_%s.pkl' % train_split)
    val_filename = save_path / ('nuscenes_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'nuscenes_infos_trainval.pkl'
    test_filename = save_path / 'nuscenes_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    nuscenes_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(nuscenes_infos_train, f)
    print('NuScenes info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    nuscenes_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(nuscenes_infos_val, f)
    print('NuScenes info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(nuscenes_infos_train + nuscenes_infos_val, f)
    print('NuScenes info trainval file is saved to %s' % trainval_filename)

    #dataset.set_split('test')
    #nuscenes_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    #with open(test_filename, 'wb') as f:
    #    pickle.dump(nuscenes_infos_test, f)
    #print('NuScenes info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_nuscenes_infos':
        create_nuscenes_infos(
            data_path=cfg.ROOT_DIR / 'data' / 'nuscenes',
            save_path=cfg.ROOT_DIR / 'data' / 'nuscenes'
            
        )
    else:
        A = NuScenesDataset(root_path='data/nuscenes', class_names=cfg.CLASS_NAMES, split='train', training=True)
        import pdb
        pdb.set_trace()
        ans = A[1]


