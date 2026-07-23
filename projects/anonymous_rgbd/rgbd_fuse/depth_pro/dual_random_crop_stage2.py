import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import autocast_box_type


@TRANSFORMS.register_module()
class DualRandomCropStage2(BaseTransform):
    """Stage2 专用 crop，同步处理：
    - img
    - aux_img
    - aux_img_low
    - aux_img_high
    - annotations
    """

    def __init__(self,
                 crop_size,
                 crop_type='absolute',
                 recompute_bbox=True,
                 allow_negative_crop=False):
        assert crop_type == 'absolute', \
            'Current DualRandomCropStage2 only supports crop_type="absolute".'
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.recompute_bbox = recompute_bbox
        self.allow_negative_crop = allow_negative_crop

    def _recompute_boxes_from_masks(self, masks, ref_bboxes):
        num_masks = len(masks)

        if num_masks == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            if hasattr(ref_bboxes, 'new_box'):
                return ref_bboxes.new_box(empty)
            elif hasattr(ref_bboxes, 'tensor'):
                return ref_bboxes.__class__(
                    empty,
                    dtype=ref_bboxes.tensor.dtype,
                    device=ref_bboxes.tensor.device)
            return empty

        boxes = []

        if hasattr(masks, 'masks') and isinstance(masks.masks, np.ndarray):
            bitmap_masks = masks.masks
            for m in bitmap_masks:
                ys, xs = np.where(m > 0)
                if len(xs) == 0 or len(ys) == 0:
                    boxes.append([0., 0., 0., 0.])
                else:
                    boxes.append([
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max()),
                        float(ys.max())
                    ])
        elif hasattr(masks, 'masks') and isinstance(masks.masks, list):
            polygon_masks = masks.masks
            for polys in polygon_masks:
                if len(polys) == 0:
                    boxes.append([0., 0., 0., 0.])
                    continue

                xs_all = []
                ys_all = []
                for poly in polys:
                    poly = np.asarray(poly, dtype=np.float32)
                    if poly.size < 6:
                        continue
                    xs_all.append(poly[0::2])
                    ys_all.append(poly[1::2])

                if len(xs_all) == 0 or len(ys_all) == 0:
                    boxes.append([0., 0., 0., 0.])
                else:
                    xs = np.concatenate(xs_all)
                    ys = np.concatenate(ys_all)
                    boxes.append([
                        float(xs.min()),
                        float(ys.min()),
                        float(xs.max()),
                        float(ys.max())
                    ])
        else:
            raise TypeError(f'Unsupported mask type for bbox recomputation: {type(masks)}')

        boxes = np.asarray(boxes, dtype=np.float32)

        if hasattr(ref_bboxes, 'new_box'):
            return ref_bboxes.new_box(boxes)
        elif hasattr(ref_bboxes, 'tensor'):
            return ref_bboxes.__class__(
                boxes,
                dtype=ref_bboxes.tensor.dtype,
                device=ref_bboxes.tensor.device)
        else:
            return boxes

    @autocast_box_type()
    def transform(self, results: dict):
        img = results['img']
        h, w = img.shape[:2]
        crop_h, crop_w = self.crop_size

        crop_h_eff = min(crop_h, h)
        crop_w_eff = min(crop_w, w)

        margin_h = max(h - crop_h_eff, 0)
        margin_w = max(w - crop_w_eff, 0)

        offset_h = np.random.randint(0, margin_h + 1) if margin_h > 0 else 0
        offset_w = np.random.randint(0, margin_w + 1) if margin_w > 0 else 0

        y1, y2 = offset_h, offset_h + crop_h_eff
        x1, x2 = offset_w, offset_w + crop_w_eff

        results['img'] = results['img'][y1:y2, x1:x2, ...]
        results['img_shape'] = results['img'].shape[:2]

        if 'aux_img' in results:
            results['aux_img'] = results['aux_img'][y1:y2, x1:x2, ...]
            results['aux_img_shape'] = results['aux_img'].shape[:2]

        if 'aux_img_low' in results:
            results['aux_img_low'] = results['aux_img_low'][y1:y2, x1:x2, ...]

        if 'aux_img_high' in results:
            results['aux_img_high'] = results['aux_img_high'][y1:y2, x1:x2, ...]

        if 'gt_bboxes' in results:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-x1, -y1])
            bboxes.clip_(results['img_shape'])
            valid_inds = bboxes.is_inside(results['img_shape']).numpy()

            if not valid_inds.any() and not self.allow_negative_crop:
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if 'gt_bboxes_labels' in results:
                results['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid_inds]

            if 'gt_ignore_flags' in results:
                results['gt_ignore_flags'] = results['gt_ignore_flags'][valid_inds]

            if 'gt_masks' in results:
                results['gt_masks'] = results['gt_masks'][valid_inds].crop(
                    np.asarray([x1, y1, x2, y2], dtype=np.int32)
                )
                if self.recompute_bbox:
                    results['gt_bboxes'] = self._recompute_boxes_from_masks(
                        results['gt_masks'],
                        results['gt_bboxes']
                    )

        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][y1:y2, x1:x2]

        return results

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'crop_size={self.crop_size}, '
            f'crop_type={self.crop_type}, '
            f'recompute_bbox={self.recompute_bbox}, '
            f'allow_negative_crop={self.allow_negative_crop})'
        )
