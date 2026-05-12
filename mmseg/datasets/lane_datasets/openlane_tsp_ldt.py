import os
import pickle

import numpy as np

from ..builder import DATASETS
from ..tools.utils import homography_g2im_extrinsic, projection_g2im_extrinsic
from .openlane import OpenlaneDataset


@DATASETS.register_module()
class OpenlaneTSPDataset(OpenlaneDataset):
    def __init__(self,
                 pipeline,
                 data_root,
                 tsp_cache_dir='lane3d_1000_v1.3/tsp_ldt_cache',
                 tsp_channels=4,
                 tsp_height=128,
                 tsp_width=48,
                 tsp_strict=True,
                 **kwargs):
        self.tsp_cache_dir = os.path.join(data_root, tsp_cache_dir)
        self.tsp_shape = (int(tsp_channels), int(tsp_height), int(tsp_width))
        self.tsp_strict = bool(tsp_strict)
        super(OpenlaneTSPDataset, self).__init__(pipeline, data_root, **kwargs)

    def load_annotations(self):
        print('Now loading annotations...')
        self.img_infos = []
        with open(self.data_list, 'r') as anno_obj:
            for sample_id in [s.strip() for s in anno_obj.readlines()]:
                self.img_infos.append({
                    'filename': os.path.join(self.img_dir, sample_id + self.img_suffix),
                    'anno_file': os.path.join(self.cache_dir, sample_id + '.pkl'),
                    'tsp_file': os.path.join(self.tsp_cache_dir, sample_id + '.npz'),
                })
        print('after load annotation')
        print('find {} samples in {}.'.format(len(self.img_infos), self.data_list))

    def load_tsp_teacher(self, filename):
        if not os.path.exists(filename):
            if self.tsp_strict:
                raise FileNotFoundError('missing TSP-LDT cache: {}'.format(filename))
            return self.empty_tsp_teacher()
        data = np.load(filename)
        if 'tsp_teacher' in data:
            teacher = data['tsp_teacher'].astype(np.float32)
        else:
            teacher = np.stack([
                data['tsp_occ'], data['tsp_sdf'],
                data['tsp_z'], data['tsp_quality'],
            ], axis=0).astype(np.float32)
        valid_key = 'tsp_valid' if 'tsp_valid' in data else 'valid'
        if valid_key not in data:
            if self.tsp_strict:
                raise KeyError('missing tsp_valid in {}'.format(filename))
            valid = np.ones((1, teacher.shape[1], teacher.shape[2]), dtype=np.float32)
        else:
            valid = data[valid_key].astype(np.float32)
        if valid.ndim == 2:
            valid = valid[None, ...]
        if teacher.ndim != 3 or teacher.shape != self.tsp_shape:
            raise ValueError(
                'bad tsp_teacher shape {} in {}, expected {}'.format(
                    teacher.shape, filename, self.tsp_shape))
        expected_valid_shape = (1, self.tsp_shape[1], self.tsp_shape[2])
        if valid.ndim != 3 or valid[:1].shape != expected_valid_shape:
            raise ValueError(
                'bad tsp_valid shape {} in {}, expected {}'.format(
                    valid.shape, filename, expected_valid_shape))
        return teacher, valid[:1]

    def empty_tsp_teacher(self):
        teacher = np.zeros(self.tsp_shape, dtype=np.float32)
        valid = np.zeros((1, self.tsp_shape[1], self.tsp_shape[2]), dtype=np.float32)
        return teacher, valid

    def __getitem__(self, idx, transform=False):
        results = self.img_infos[idx].copy()
        results['img_info'] = {}
        results['img_info']['filename'] = results['filename']
        results['ori_filename'] = results['filename']
        results['ori_shape'] = (self.h_org, self.w_org)
        results['flip'] = False
        results['flip_direction'] = None
        with open(results['anno_file'], 'rb') as f:
            obj = pickle.load(f)
            results.update(obj)
        if self.no_cls:
            results['gt_3dlanes'][:, 1] = results['gt_3dlanes'][:, 1] > 0
        if self.max_lanes > -1:
            results['gt_3dlanes'] = results['gt_3dlanes'][:self.max_lanes]
        results['img_metas'] = {'ori_shape': results['ori_shape']}
        results['gt_project_matrix'] = projection_g2im_extrinsic(
            results['gt_camera_extrinsic'], results['gt_camera_intrinsic'])
        results['gt_homography_matrix'] = homography_g2im_extrinsic(
            results['gt_camera_extrinsic'], results['gt_camera_intrinsic'])
        results['tsp_teacher'], results['tsp_valid'] = self.load_tsp_teacher(
            results['tsp_file'])
        return self.pipeline(results)
