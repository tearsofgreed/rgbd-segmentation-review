import os
import os.path as osp

import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SaveDualDebugImages(BaseTransform):
    """将预处理后的 img 和 aux_img 保存到本地，便于检查同步情况。

    默认会保存：
    - rgb
    - depth
    - overlay

    注意：
    这是调试用 transform，不建议正式训练时保留。
    """

    def __init__(self, save_dir='work_dirs/dual_debug', max_save=50):
        self.save_dir = save_dir
        self.max_save = max_save
        self.save_count = 0
        os.makedirs(self.save_dir, exist_ok=True)

    def _to_uint8(self, img):
        img = img.copy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img

    def _normalize_to_uint8(self, img):
        img = img.copy().astype(np.float32)
        if img.ndim == 3 and img.shape[2] == 3:
            min_v, max_v = img.min(), img.max()
            if max_v - min_v < 1e-6:
                return np.zeros_like(img, dtype=np.uint8)
            img = (img - min_v) / (max_v - min_v) * 255.0
            return img.astype(np.uint8)

        min_v, max_v = img.min(), img.max()
        if max_v - min_v < 1e-6:
            return np.zeros_like(img, dtype=np.uint8)
        img = (img - min_v) / (max_v - min_v) * 255.0
        return img.astype(np.uint8)

    def _make_overlay(self, rgb, aux):
        rgb = self._to_uint8(rgb)
        aux = self._normalize_to_uint8(aux)

        if aux.ndim == 2:
            aux = cv2.cvtColor(aux, cv2.COLOR_GRAY2BGR)

        return cv2.addWeighted(rgb, 0.6, aux, 0.4, 0)

    def transform(self, results: dict) -> dict:
        if self.save_count >= self.max_save:
            return results

        img = results['img']
        aux_img = results['aux_img']
        img_path = results['img_path']

        stem = osp.splitext(osp.basename(img_path))[0]

        rgb = self._to_uint8(img)
        aux = self._normalize_to_uint8(aux_img)
        overlay = self._make_overlay(rgb, aux)

        rgb_path = osp.join(self.save_dir, f'{self.save_count:03d}_{stem}_rgb.jpg')
        aux_path = osp.join(self.save_dir, f'{self.save_count:03d}_{stem}_depth.jpg')
        overlay_path = osp.join(self.save_dir, f'{self.save_count:03d}_{stem}_overlay.jpg')

        # mmcv.imwrite / cv2.imwrite 都按 BGR 保存，所以这里直接存当前数组
        mmcv.imwrite(rgb, rgb_path)
        mmcv.imwrite(aux, aux_path)
        mmcv.imwrite(overlay, overlay_path)

        print('=' * 80)
        print(f'[SaveDualDebugImages] save index : {self.save_count}')
        print(f'[SaveDualDebugImages] img_path   : {results.get("img_path", "")}')
        print(f'[SaveDualDebugImages] aux_path   : {results.get("aux_img_path", "")}')
        print(f'[SaveDualDebugImages] img_shape  : {results["img"].shape}')
        print(f'[SaveDualDebugImages] aux_shape  : {results["aux_img"].shape}')
        print(f'[SaveDualDebugImages] saved rgb  : {rgb_path}')
        print(f'[SaveDualDebugImages] saved dep  : {aux_path}')
        print(f'[SaveDualDebugImages] saved over : {overlay_path}')

        self.save_count += 1
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(save_dir={self.save_dir}, max_save={self.max_save})'
