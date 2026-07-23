import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type


@TRANSFORMS.register_module()
class DualRandomFlipStage2(BaseTransform):
    """Stage2 专用 flip，同步处理：
    - img
    - aux_img
    - aux_img_low
    - aux_img_high
    - annotations
    """

    def __init__(self, prob=0.5, direction='horizontal'):
        self.prob = prob
        self.direction = direction

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        flip = np.random.rand() < self.prob
        results['flip'] = flip
        results['flip_direction'] = self.direction if flip else None

        if not flip:
            return results

        results['img'] = mmcv.imflip(results['img'], direction=self.direction)

        if 'aux_img' in results:
            results['aux_img'] = mmcv.imflip(results['aux_img'], direction=self.direction)

        if 'aux_img_low' in results:
            results['aux_img_low'] = mmcv.imflip(results['aux_img_low'], direction=self.direction)

        if 'aux_img_high' in results:
            results['aux_img_high'] = mmcv.imflip(results['aux_img_high'], direction=self.direction)

        results['img_shape'] = results['img'].shape[:2]

        if 'aux_img' in results:
            results['aux_img_shape'] = results['aux_img'].shape[:2]

        if 'gt_bboxes' in results:
            results['gt_bboxes'].flip_(results['img_shape'], direction=self.direction)

        if 'gt_masks' in results:
            results['gt_masks'] = results['gt_masks'].flip(self.direction)

        if 'gt_seg_map' in results:
            results['gt_seg_map'] = mmcv.imflip(results['gt_seg_map'], direction=self.direction)

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob}, direction={self.direction})'
