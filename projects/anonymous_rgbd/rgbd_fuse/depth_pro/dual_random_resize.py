import random
import mmcv
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type


@TRANSFORMS.register_module()
class DualRandomResize(BaseTransform):
    """对 img / aux_img / annotations 使用同一随机 resize。

    目标：尽量对齐原始 Mask2Former 的 RandomResize 行为。

    Args:
        scale (tuple[int, int]): base image size, e.g. (1024, 1024)
        ratio_range (tuple[float, float]): random ratio range
        keep_ratio (bool): whether to keep aspect ratio
        interpolation (str): interpolation for image/aux image
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
            aux_resized = mmcv.imresize(
                results['aux_img'],
                (new_w, new_h),
                interpolation=self.interpolation,
                backend='cv2'
            )
            results['aux_img'] = aux_resized
            results['aux_img_shape'] = aux_resized.shape[:2]

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
