import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import to_tensor
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData, PixelData


@TRANSFORMS.register_module()
class PackDualDetInputsStage2(BaseTransform):
    """Stage2 专用 pack。

    支持：
    - RGB-only
    - RGB-D
    - RGB-D + low/high freq depth
    """

    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks'
    }

    def __init__(self,
                 meta_keys=('img_id', 'img_path', 'aux_img_path', 'ori_shape',
                            'img_shape', 'aux_img_shape', 'scale_factor',
                            'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def _to_chw_tensor(self, arr: np.ndarray, name: str):
        if not isinstance(arr, np.ndarray):
            raise TypeError(f'`{name}` must be np.ndarray, but got {type(arr)}')
        if len(arr.shape) < 3:
            arr = np.expand_dims(arr, -1)
        arr = np.ascontiguousarray(arr.transpose(2, 0, 1))
        return to_tensor(arr)

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        if 'img' not in results:
            raise KeyError('`img` not found in results, cannot pack RGB input.')

        packed_results['inputs'] = self._to_chw_tensor(results['img'], 'img')

        if 'aux_img' in results:
            packed_results['aux_inputs'] = self._to_chw_tensor(results['aux_img'], 'aux_img')

        if 'aux_img_low' in results:
            packed_results['aux_low_inputs'] = self._to_chw_tensor(results['aux_img_low'], 'aux_img_low')

        if 'aux_img_high' in results:
            packed_results['aux_high_inputs'] = self._to_chw_tensor(results['aux_img_high'], 'aux_img_high')

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()

        gt_ignore_flags = results.get('gt_ignore_flags', None)
        valid_idx = None
        ignore_idx = None
        if gt_ignore_flags is not None:
            valid_idx = np.where(gt_ignore_flags == 0)[0]
            ignore_idx = np.where(gt_ignore_flags == 1)[0]

        for key, mapped_key in self.mapping_table.items():
            if key not in results:
                continue

            value = results[key]

            if key == 'gt_bboxes':
                if gt_ignore_flags is not None:
                    instance_data[mapped_key] = value[valid_idx]
                    ignore_instance_data[mapped_key] = value[ignore_idx]
                else:
                    instance_data[mapped_key] = value

            elif key == 'gt_masks':
                if gt_ignore_flags is not None:
                    instance_data[mapped_key] = value[valid_idx]
                    ignore_instance_data[mapped_key] = value[ignore_idx]
                else:
                    instance_data[mapped_key] = value

            elif key == 'gt_bboxes_labels':
                value = to_tensor(value)
                if gt_ignore_flags is not None:
                    instance_data[mapped_key] = value[valid_idx]
                    ignore_instance_data[mapped_key] = value[ignore_idx]
                else:
                    instance_data[mapped_key] = value

        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict()
            gt_seg_map = results['gt_seg_map'][None, ...].copy()
            gt_sem_seg_data['sem_seg'] = to_tensor(gt_seg_map)
            data_sample.gt_sem_seg = PixelData(**gt_sem_seg_data)

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)

        packed_results['data_samples'] = data_sample
        return packed_results

    def __repr__(self):
        return f'{self.__class__.__name__}(meta_keys={self.meta_keys})'
