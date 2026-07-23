from typing import Optional, Sequence, Tuple

import cv2
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS

from .depth_freq_split import split_depth_low_high


@TRANSFORMS.register_module()
class DepthPreprocess(BaseTransform):
    """对 aux_img 做深度专属预处理，并可选产生高低频分量。

    新增：
        output_mode:
            - 'base': aux_img 保持为基础修复后的 depth
            - 'low' : aux_img 替换为 low-frequency depth
            - 'high': aux_img 替换为 high-frequency depth
    """

    def __init__(self,
                 key: str = 'aux_img',
                 to_float32: bool = True,
                 squeeze_if_single_channel: bool = True,
                 to_gray: bool = False,
                 clip_min: Optional[float] = None,
                 clip_max: Optional[float] = None,
                 invalid_values: Optional[Sequence[float]] = None,
                 invalid_min: Optional[float] = None,
                 invalid_fill_value: float = 0.0,
                 normalize_mode: Optional[str] = None,
                 norm_mean: Optional[float] = None,
                 norm_std: Optional[float] = None,
                 scale_factor: Optional[float] = None,
                 apply_log1p: bool = False,
                 median_ksize: Optional[int] = None,
                 percentile_clip: Optional[Tuple[float, float]] = None,
                 # -------- 高低频分离 --------
                 enable_freq_split: bool = False,
                 low_key: str = 'aux_img_low',
                 high_key: str = 'aux_img_high',
                 output_mode: str = 'base',
                 low_mode: str = 'gaussian',
                 gaussian_ksize: int = 5,
                 gaussian_sigma: float = 0.0,
                 blur_ksize: int = 5,
                 high_mode: str = 'residual_abs',
                 high_percentile_clip: Optional[Tuple[float, float]] = (1.0, 99.0),
                 normalize_low: bool = True,
                 normalize_high: bool = True,
                 eps: float = 1e-6):
        self.key = key
        self.to_float32 = to_float32
        self.squeeze_if_single_channel = squeeze_if_single_channel
        self.to_gray = to_gray

        self.clip_min = clip_min
        self.clip_max = clip_max

        self.invalid_values = invalid_values
        self.invalid_min = invalid_min
        self.invalid_fill_value = invalid_fill_value

        self.normalize_mode = normalize_mode
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.scale_factor = scale_factor
        self.apply_log1p = apply_log1p

        self.median_ksize = median_ksize
        self.percentile_clip = percentile_clip

        # freq split
        self.enable_freq_split = enable_freq_split
        self.low_key = low_key
        self.high_key = high_key
        self.output_mode = output_mode

        self.low_mode = low_mode
        self.gaussian_ksize = gaussian_ksize
        self.gaussian_sigma = gaussian_sigma
        self.blur_ksize = blur_ksize
        self.high_mode = high_mode
        self.high_percentile_clip = high_percentile_clip
        self.normalize_low = normalize_low
        self.normalize_high = normalize_high

        self.eps = eps

        if self.output_mode not in ['base', 'low', 'high']:
            raise ValueError(
                f'Unsupported output_mode: {self.output_mode}. '
                "Supported: 'base' / 'low' / 'high'"
            )

    def _to_gray(self, depth: np.ndarray) -> np.ndarray:
        if depth.ndim == 2:
            return depth
        if depth.ndim == 3 and depth.shape[2] == 1:
            return depth[:, :, 0]
        if depth.ndim == 3 and depth.shape[2] == 3:
            return cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        raise ValueError(f'Unsupported depth shape for to_gray: {depth.shape}')

    def _replace_invalid(self, depth: np.ndarray) -> np.ndarray:
        if self.invalid_values is None:
            return depth

        invalid_mask = np.zeros(depth.shape, dtype=bool)
        for v in self.invalid_values:
            invalid_mask |= (depth == v)

        depth = depth.copy()
        depth[invalid_mask] = self.invalid_fill_value
        return depth

    def _build_valid_mask(self, depth: np.ndarray) -> np.ndarray:
        if self.invalid_min is None:
            return np.ones(depth.shape, dtype=bool)
        return depth > self.invalid_min

    def _median_filter(self, depth: np.ndarray) -> np.ndarray:
        if self.median_ksize is None or self.median_ksize <= 1:
            return depth
        if self.median_ksize % 2 == 0:
            raise ValueError('`median_ksize` must be odd.')
        return cv2.medianBlur(depth, self.median_ksize)

    def _percentile_clip_on_valid(self,
                                  depth: np.ndarray,
                                  valid_mask: np.ndarray) -> np.ndarray:
        if self.percentile_clip is None:
            return depth

        low_p, high_p = self.percentile_clip
        if not (0.0 <= low_p <= 100.0 and 0.0 <= high_p <= 100.0):
            raise ValueError('percentile_clip values must be in [0, 100].')
        if high_p <= low_p:
            raise ValueError('percentile_clip high must be greater than low.')

        valid_values = depth[valid_mask]
        if valid_values.size == 0:
            return depth

        low = np.percentile(valid_values, low_p)
        high = np.percentile(valid_values, high_p)

        depth = depth.copy()
        depth[valid_mask] = np.clip(depth[valid_mask], low, high)
        return depth

    def _clip(self, depth: np.ndarray) -> np.ndarray:
        if self.clip_min is not None:
            depth = np.maximum(depth, self.clip_min)
        if self.clip_max is not None:
            depth = np.minimum(depth, self.clip_max)
        return depth

    def _normalize(self, depth: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        if self.normalize_mode is None:
            return depth

        if self.normalize_mode == 'zscore':
            if self.norm_mean is None or self.norm_std is None:
                raise ValueError(
                    '`norm_mean` and `norm_std` must be set when '
                    "normalize_mode='zscore'"
                )
            depth = (depth - self.norm_mean) / (self.norm_std + self.eps)

        elif self.normalize_mode == 'minmax':
            dmin = float(depth.min())
            dmax = float(depth.max())
            depth = (depth - dmin) / (dmax - dmin + self.eps)

        elif self.normalize_mode == 'fixed_range':
            if self.clip_min is None or self.clip_max is None:
                raise ValueError(
                    '`clip_min` and `clip_max` must be set when '
                    "normalize_mode='fixed_range'"
                )
            depth = (depth - self.clip_min) / (
                self.clip_max - self.clip_min + self.eps
            )

        elif self.normalize_mode == 'minmax_on_valid':
            if valid_mask is None:
                raise ValueError(
                    "`valid_mask` must be provided when "
                    "normalize_mode='minmax_on_valid'"
                )

            valid_values = depth[valid_mask]
            out = np.zeros_like(depth, dtype=np.float32)

            if valid_values.size == 0:
                return out

            dmin = float(valid_values.min())
            dmax = float(valid_values.max())

            if dmax <= dmin:
                out[valid_mask] = 0.0
            else:
                out[valid_mask] = (
                    (depth[valid_mask] - dmin) / (dmax - dmin + self.eps)
                )

            depth = out

        else:
            raise ValueError(
                f'Unsupported normalize_mode: {self.normalize_mode}. '
                "Supported: None / 'zscore' / 'minmax' / "
                "'fixed_range' / 'minmax_on_valid'"
            )

        return depth

    def transform(self, results: dict) -> dict:
        if self.key not in results:
            raise KeyError(f'`{self.key}` not found in results.')

        depth = results[self.key]

        if not isinstance(depth, np.ndarray):
            depth = np.array(depth)

        if self.to_gray:
            depth = self._to_gray(depth)

        if (self.squeeze_if_single_channel and depth.ndim == 3
                and depth.shape[2] == 1):
            depth = depth[:, :, 0]

        if self.to_float32 and depth.dtype != np.float32:
            depth = depth.astype(np.float32)

        if self.scale_factor is not None:
            depth = depth * self.scale_factor

        # 1) invalid exact replace
        depth = self._replace_invalid(depth)

        # 2) denoise
        depth = self._median_filter(depth)

        # 3) valid region
        valid_mask = self._build_valid_mask(depth)

        # 4) percentile clip on valid
        depth = self._percentile_clip_on_valid(depth, valid_mask)

        # 5) normal clip
        depth = self._clip(depth)

        # 6) optional log
        if self.apply_log1p:
            depth = np.log1p(np.maximum(depth, 0.0))

        # 7) normalize
        depth = self._normalize(depth, valid_mask=valid_mask)

        if self.invalid_min is not None:
            depth = depth.copy()
            depth[~valid_mask] = self.invalid_fill_value

        # base repaired depth
        results[self.key] = depth
        results['aux_img_shape'] = depth.shape[:2]

        # 8) frequency split
        if self.enable_freq_split:
            if self.high_mode in ('residual_valid_norm', 'sobel_valid_norm', 'laplacian_valid_norm'):
                from .depth_freq_split import _split_depth_new_high_mode
                # Re-normalize depth with valid-aware robust norm for new modes
                valid_mask_dense = valid_mask if valid_mask.sum() >= 10 else (depth > self.invalid_fill_value)
                depth_norm = depth.copy()
                if self.normalize_mode == 'minmax_on_valid' or True:
                    from .depth_freq_split import _robust_normalize_depth
                    depth_norm = _robust_normalize_depth(depth_norm, valid_mask_dense, p_low=1.0, p_high=99.0)
                low, high = _split_depth_new_high_mode(
                    depth=depth_norm, low=low if 'low' in dir() else depth_norm,
                    valid=valid_mask_dense, high_mode=self.high_mode,
                    blur_kernel=getattr(self, 'blur_ksize', 7),
                    eps=self.eps)
            else:
                low, high = split_depth_low_high(
                    depth=depth,
                low_mode=self.low_mode,
                gaussian_ksize=self.gaussian_ksize,
                gaussian_sigma=self.gaussian_sigma,
                blur_ksize=self.blur_ksize,
                high_mode=self.high_mode,
                high_percentile_clip=self.high_percentile_clip,
                normalize_low=self.normalize_low,
                normalize_high=self.normalize_high,
                eps=self.eps
            )
            results[self.low_key] = low
            results[self.high_key] = high

            # 9) decide which one becomes aux_img
            if self.output_mode == 'low':
                results[self.key] = results[self.low_key]
                results['aux_img_shape'] = results[self.low_key].shape[:2]
            elif self.output_mode == 'high':
                results[self.key] = results[self.high_key]
                results['aux_img_shape'] = results[self.high_key].shape[:2]

        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'key={self.key}, '
            f'to_float32={self.to_float32}, '
            f'squeeze_if_single_channel={self.squeeze_if_single_channel}, '
            f'to_gray={self.to_gray}, '
            f'clip_min={self.clip_min}, '
            f'clip_max={self.clip_max}, '
            f'invalid_values={self.invalid_values}, '
            f'invalid_min={self.invalid_min}, '
            f'invalid_fill_value={self.invalid_fill_value}, '
            f'normalize_mode={self.normalize_mode}, '
            f'norm_mean={self.norm_mean}, '
            f'norm_std={self.norm_std}, '
            f'scale_factor={self.scale_factor}, '
            f'apply_log1p={self.apply_log1p}, '
            f'median_ksize={self.median_ksize}, '
            f'percentile_clip={self.percentile_clip}, '
            f'enable_freq_split={self.enable_freq_split}, '
            f'low_key={self.low_key}, '
            f'high_key={self.high_key}, '
            f'output_mode={self.output_mode}, '
            f'low_mode={self.low_mode}, '
            f'high_mode={self.high_mode})'
        )
