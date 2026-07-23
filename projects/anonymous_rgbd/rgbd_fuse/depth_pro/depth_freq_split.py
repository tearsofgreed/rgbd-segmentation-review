from typing import Optional, Tuple

import cv2
import numpy as np


def _normalize_to_01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.float32)
    return (x - xmin) / (xmax - xmin + eps)


def _percentile_clip(x: np.ndarray,
                     percentile_clip: Optional[Tuple[float, float]] = None) -> np.ndarray:
    if percentile_clip is None:
        return x

    low_p, high_p = percentile_clip
    if not (0.0 <= low_p <= 100.0 and 0.0 <= high_p <= 100.0):
        raise ValueError('percentile_clip values must be in [0, 100].')
    if high_p <= low_p:
        raise ValueError('percentile_clip high must be greater than low.')

    low = np.percentile(x, low_p)
    high = np.percentile(x, high_p)

    x = np.clip(x, low, high)
    return x


def split_depth_low_high(depth: np.ndarray,
                         low_mode: str = 'gaussian',
                         gaussian_ksize: int = 5,
                         gaussian_sigma: float = 0.0,
                         blur_ksize: int = 5,
                         high_mode: str = 'residual_abs',
                         high_percentile_clip: Optional[Tuple[float, float]] = (1.0, 99.0),
                         normalize_low: bool = True,
                         normalize_high: bool = True,
                         eps: float = 1e-6):
    """对单通道 depth 做高低频分离。

    Args:
        depth: [H, W], 建议已经完成基础修复并归一化
        low_mode: 当前支持 'gaussian' / 'box'
        gaussian_ksize: 高斯核大小，必须为奇数
        gaussian_sigma: 高斯 sigma
        blur_ksize: box blur 核大小
        high_mode: 当前支持 'residual_abs'
        high_percentile_clip: 对 high 做分位数裁剪
        normalize_low: 是否把 low 归一化到 [0,1]
        normalize_high: 是否把 high 归一化到 [0,1]

    Returns:
        low, high: 两个 [H, W] float32 数组
    """
    if depth.ndim != 2:
        raise ValueError(f'Expected depth shape [H, W], but got {depth.shape}')

    depth = depth.astype(np.float32)

    # -------- low frequency --------
    if low_mode == 'gaussian':
        if gaussian_ksize % 2 == 0:
            raise ValueError('gaussian_ksize must be odd.')
        low = cv2.GaussianBlur(depth, (gaussian_ksize, gaussian_ksize), gaussian_sigma)
    elif low_mode == 'box':
        if blur_ksize <= 1:
            low = depth.copy()
        else:
            low = cv2.blur(depth, (blur_ksize, blur_ksize))
    else:
        raise ValueError(f'Unsupported low_mode: {low_mode}')

    # -------- high frequency --------
    if high_mode == 'residual_abs':
        high = np.abs(depth - low)
    else:
        raise ValueError(f'Unsupported high_mode: {high_mode}')

    # 对高频做裁剪，避免少量强响应主导整图
    high = _percentile_clip(high, high_percentile_clip)

    if normalize_low:
        low = _normalize_to_01(low, eps=eps)
    if normalize_high:
        high = _normalize_to_01(high, eps=eps)

    return low.astype(np.float32), high.astype(np.float32)


# ---- New valid-aware high modes (smoke-only, default unchanged) ----

def _robust_normalize_depth(depth: "np.ndarray", valid: "np.ndarray",
                            p_low: float = 1.0, p_high: float = 99.0,
                            eps: float = 1e-6) -> "np.ndarray":
    """Robust normalize depth using valid pixels only."""
    out = np.zeros_like(depth, dtype=np.float32)
    vals = depth[valid]
    if vals.size < 10 or p_high <= p_low:
        return out
    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)
    rng = hi - lo
    if rng < eps:
        return out
    out[valid] = np.clip((depth[valid] - lo) / rng, 0.0, 1.0)
    return out


def _normalize_high(high: "np.ndarray", valid: "np.ndarray",
                    high_norm_percentile: float = 99.0,
                    eps: float = 1e-6) -> "np.ndarray":
    """Normalize high map using non-zero valid pixels."""
    out = np.zeros_like(high, dtype=np.float32)
    mask = valid & (high > eps)
    vals = high[mask]
    if vals.size < 10:
        return out
    p99 = np.percentile(vals, high_norm_percentile)
    if p99 < eps:
        return out
    out[mask] = np.clip(high[mask] / p99, 0.0, 1.0)
    return out


def _split_depth_new_high_mode(depth, low, valid, high_mode, blur_kernel, eps):
    """Compute high-depth for new high_modes (residual_valid_norm, sobel_valid_norm, laplacian_valid_norm).

    All new modes assume depth has been robust-normalized to [0,1] on valid pixels.
    """
    import cv2
    if high_mode == 'residual_valid_norm':
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blurred = cv2.GaussianBlur(depth, (blur_kernel, blur_kernel), 0)
        high = np.abs(depth - blurred)
        high = _normalize_high(high, valid, high_norm_percentile=99.0, eps=eps)
        low = blurred.copy()
        return low.astype(np.float32), high.astype(np.float32)

    elif high_mode == 'sobel_valid_norm':
        grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        high = np.sqrt(grad_x**2 + grad_y**2)
        high = _normalize_high(high, valid, high_norm_percentile=99.0, eps=eps)
        low = depth.copy()
        return low.astype(np.float32), high.astype(np.float32)

    elif high_mode == 'laplacian_valid_norm':
        lap = cv2.Laplacian(depth, cv2.CV_32F, ksize=3)
        high = np.abs(lap)
        high = _normalize_high(high, valid, high_norm_percentile=99.0, eps=eps)
        low = depth.copy()
        return low.astype(np.float32), high.astype(np.float32)

    raise ValueError(f'Unsupported new high_mode: {high_mode}')
