import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms import to_tensor
from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData, PixelData


@TRANSFORMS.register_module()
class PackDualDetInputs(BaseTransform):
    """Pack inputs for RGB-only or RGB-D detection.

    单样本输出格式：

    RGB-only:
    {
        'inputs': Tensor[C, H, W],
        'data_samples': DetDataSample
    }

    RGB-D:
    {
        'inputs': Tensor[C, H, W],
        'aux_inputs': Tensor[C, H, W],
        'data_samples': DetDataSample
    }
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

    def transform(self, results: dict) -> dict:
        packed_results = dict()

        if 'img' not in results:
            raise KeyError('`img` not found in results, cannot pack RGB input.')

        img = results['img']
        if not isinstance(img, np.ndarray):
            raise TypeError(f'`img` must be np.ndarray, but got {type(img)}')

        aux_img = results.get('aux_img', None)
        if aux_img is not None and not isinstance(aux_img, np.ndarray):
            raise TypeError(f'`aux_img` must be np.ndarray, but got {type(aux_img)}')

        # HWC -> CHW
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1))
        packed_results['inputs'] = to_tensor(img)

        if aux_img is not None:
            if len(aux_img.shape) < 3:
                aux_img = np.expand_dims(aux_img, -1)
            aux_img = np.ascontiguousarray(aux_img.transpose(2, 0, 1))
            packed_results['aux_inputs'] = to_tensor(aux_img)

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
