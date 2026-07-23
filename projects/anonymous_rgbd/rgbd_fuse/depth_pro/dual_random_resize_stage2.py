import random
import mmcv
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type


@TRANSFORMS.register_module()
class DualRandomResizeStage2(BaseTransform):
    """Stage2 专用随机 resize，同步处理：
    - img
    - aux_img
    - aux_img_low
    - aux_img_high
    - annotations
    """

    def __init__(self,
                 scale,
                 ratio_range=(0.1, 2.0),
                 keep_ratio=True,
                 interpolation='bilinear'):
        self.scale = scale
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.interpolation = interpolation

    def _sample_scale(self):
        min_ratio, max_ratio = self.ratio_range
        ratio = random.uniform(min_ratio, max_ratio)
        target_w = int(self.scale[0] * ratio)
        target_h = int(self.scale[1] * ratio)
        target_w = max(target_w, 1)
        target_h = max(target_h, 1)
        return (target_w, target_h)

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        img = results['img']
        ori_h, ori_w = img.shape[:2]

        target_scale = self._sample_scale()

        if self.keep_ratio:
            resized_img, _ = mmcv.imrescale(
                img,
                target_scale,
                interpolation=self.interpolation,
                return_scale=True,
                backend='cv2'
            )
            new_h, new_w = resized_img.shape[:2]
            w_scale = new_w / ori_w
            h_scale = new_h / ori_h
        else:
            resized_img, w_scale, h_scale = mmcv.imresize(
                img,
                target_scale,
                interpolation=self.interpolation,
                return_scale=True,
                backend='cv2'
            )
            new_h, new_w = resized_img.shape[:2]

        results['img'] = resized_img
        results['img_shape'] = resized_img.shape[:2]
        results['scale_factor'] = (w_scale, h_scale)
        results['keep_ratio'] = self.keep_ratio

        if 'aux_img' in results:
            results['aux_img'] = mmcv.imresize(
                results['aux_img'],
                (new_w, new_h),
                interpolation=self.interpolation,
                backend='cv2'
            )
            results['aux_img_shape'] = results['aux_img'].shape[:2]

        if 'aux_img_low' in results:
            results['aux_img_low'] = mmcv.imresize(
                results['aux_img_low'],
                (new_w, new_h),
                interpolation=self.interpolation,
                backend='cv2'
            )

        if 'aux_img_high' in results:
            results['aux_img_high'] = mmcv.imresize(
                results['aux_img_high'],
                (new_w, new_h),
                interpolation=self.interpolation,
                backend='cv2'
            )

        if 'gt_bboxes' in results:
            results['gt_bboxes'].rescale_((w_scale, h_scale))

        if 'gt_masks' in results:
            results['gt_masks'] = results['gt_masks'].resize((new_h, new_w))

        if 'gt_seg_map' in results:
            results['gt_seg_map'] = mmcv.imresize(
                results['gt_seg_map'],
                (new_w, new_h),
                interpolation='nearest',
                backend='cv2'
            )

        return results

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'scale={self.scale}, '
            f'ratio_range={self.ratio_range}, '
            f'keep_ratio={self.keep_ratio}, '
            f'interpolation={self.interpolation})'
        )
