import os.path as osp

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadAuxImageFromFile(BaseTransform):
    """根据主图 img_path 自动加载对应深度图到 aux_img。

    目录约定：
        data/seed1/images/000001.jpg
        data/seed1/depth/000001.png

    新增字段：
        aux_img_path / aux_img / aux_img_shape / ori_aux_img_shape

    关键修复 (2025-06):
        - 替换 mmcv.imread 为 cv2.imdecode + np.fromfile，支持 Windows 中文路径
        - 新增 flag 参数 (unchanged/grayscale/color)，用于控制 depth 读取方式
        - depth 默认以 unchanged 读取，保留 16-bit 连续动态范围
        - 不再做 /255.0、uint8 化、RGB normalize
        - 新增 debug 模式，输出加载统计信息
    """

    def __init__(
        self,
        aux_folder='depth',
        aux_suffix='.png',
        to_float32=True,
        color_type=None,
        imdecode_backend=None,
        flag='unchanged',
        depth_scale=1.0,
        squeeze_channel=True,
        debug=False,
    ):
        super().__init__()
        self.aux_folder = str(aux_folder)
        self.aux_suffix = str(aux_suffix)
        self.to_float32 = bool(to_float32)
        self.depth_scale = float(depth_scale)
        self.squeeze_channel = bool(squeeze_channel)
        self.debug = bool(debug)

        # --- flag 解析：优先使用 flag，向后兼容 color_type ---
        if flag is not None:
            self.flag = str(flag)
        elif color_type is not None:
            self.flag = str(color_type)
        else:
            self.flag = 'unchanged'

        if self.flag not in ('unchanged', 'grayscale', 'color'):
            raise ValueError(
                f'flag must be one of (unchanged, grayscale, color), got {self.flag!r}'
            )

        # 保留旧参数以保持兼容性（不会被使用，但不报错）
        self.color_type = self.flag
        self.imdecode_backend = 'cv2'

        # --- flag -> cv2 flag 映射 ---
        _FLAG_MAP = {
            'unchanged': cv2.IMREAD_UNCHANGED,
            'grayscale': cv2.IMREAD_GRAYSCALE,
            'color': cv2.IMREAD_COLOR,
        }
        self._cv2_flag = _FLAG_MAP[self.flag]

    @staticmethod
    def _imread_unicode(path, cv2_flag):
        """Windows 中文路径安全读取，使用 np.fromfile + cv2.imdecode"""
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2_flag)

    def transform(self, results: dict) -> dict:
        img_path = results['img_path']

        # img_path: .../data/seed1/images/000001.jpg
        # aux_path: .../data/seed1/depth/000001.png
        img_dir = osp.dirname(img_path)
        data_root = osp.dirname(img_dir)
        stem = osp.splitext(osp.basename(img_path))[0]
        aux_img_path = osp.join(data_root, self.aux_folder, stem + self.aux_suffix)

        if not osp.exists(aux_img_path):
            raise FileNotFoundError(f'Cannot find aux image: {aux_img_path}')

        # --- 核心修复：使用 cv2.imdecode + IMREAD_UNCHANGED ---
        aux_img = self._imread_unicode(aux_img_path, self._cv2_flag)

        if aux_img is None:
            raise ValueError(f'Failed to read aux image: {aux_img_path}')

        original_dtype = aux_img.dtype
        original_shape = aux_img.shape

        # --- 通道处理 ---
        if self.squeeze_channel and aux_img.ndim == 3:
            if aux_img.shape[2] == 1:
                aux_img = aux_img[:, :, 0]  # HxWx1 -> HxW
            elif aux_img.shape[2] == 3:
                # 三通道：检查是否完全相同
                ch0 = aux_img[:, :, 0]
                ch1 = aux_img[:, :, 1]
                ch2 = aux_img[:, :, 2]
                if np.array_equal(ch0, ch1) and np.array_equal(ch0, ch2):
                    aux_img = ch0  # 三通道完全相同，取第 0 通道
                else:
                    raise ValueError(
                        f'Depth image has 3 non-identical channels at {aux_img_path}. '
                        f'If this is intentional, set squeeze_channel=False.'
                    )

        # --- 转换为 float32 ---
        if self.to_float32:
            aux_img = aux_img.astype(np.float32)

        # --- 显式缩放（仅当 depth_scale != 1.0 时） ---
        if self.depth_scale != 1.0:
            aux_img = aux_img * self.depth_scale

        # --- 保存结果 ---
        results['aux_img_path'] = aux_img_path
        results['aux_img'] = aux_img
        results['aux_img_shape'] = aux_img.shape[:2]
        results['ori_aux_img_shape'] = aux_img.shape[:2]

        # --- debug 统计 ---
        if self.debug:
            flat = aux_img.ravel()
            # 只对前 200000 个 finite 值统计，避免超慢
            finite_vals = flat[np.isfinite(flat)][:200000]
            if len(finite_vals) > 0:
                debug_info = {
                    'path': aux_img_path,
                    'flag': self.flag,
                    'original_dtype': str(original_dtype),
                    'original_shape': original_shape,
                    'final_dtype': str(aux_img.dtype),
                    'final_shape': aux_img.shape,
                    'min': float(np.min(finite_vals)),
                    'p50': float(np.percentile(finite_vals, 50)),
                    'p95': float(np.percentile(finite_vals, 95)),
                    'p99': float(np.percentile(finite_vals, 99)),
                    'max': float(np.max(finite_vals)),
                    'std': float(np.std(finite_vals)),
                    'nonzero_frac': float(np.mean(np.abs(finite_vals) > 1e-6)),
                    'near_zero_frac': float(np.mean(np.abs(finite_vals) <= 1e-6)),
                    'unique_approx': int(len(np.unique(finite_vals.astype(np.float32)))),
                }
            else:
                debug_info = {
                    'path': aux_img_path,
                    'flag': self.flag,
                    'error': 'no finite values',
                }
            results['aux_load_debug'] = debug_info

        return results

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'aux_folder={self.aux_folder}, '
            f'aux_suffix={self.aux_suffix}, '
            f'to_float32={self.to_float32}, '
            f'flag={self.flag!r}, '
            f'depth_scale={self.depth_scale}, '
            f'squeeze_channel={self.squeeze_channel}, '
            f'debug={self.debug})'
        )
