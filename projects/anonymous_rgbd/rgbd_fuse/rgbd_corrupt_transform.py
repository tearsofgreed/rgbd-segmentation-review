# -*- coding: utf-8 -*-
"""
Unified RGB-D corruption utility.

It has two usage paths:

1) MMDetection/MMEngine transform registration:
   Put this file at:
       <mmdetection_root>/rgbd_fuse/rgbd_corrupt_transform.py

   Add to config custom_imports:
       'rgbd_fuse.rgbd_corrupt_transform'

   Then use:
       dict(type='RGBDCorruption', mode='depth_noise', depth_noise_sigma=10.0, ...)

2) Terminal preview/export:
   Run directly to preview 10 samples or export corrupted copies:
       python rgbd_corrupt_transform.py preview --rgb-dir ... --depth-dir ... --out-dir ... --mode depth_noise --depth-noise-sigma 10
       python rgbd_corrupt_transform.py export  --rgb-dir ... --depth-dir ... --out-rgb-dir ... --out-depth-dir ... --mode rgb_gamma --gamma 1.5

The CLI and the registered transform call the same RGBDCorruption class, so the same parameter values produce the same corruption behavior.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    from mmdet.registry import TRANSFORMS
except Exception:
    TRANSFORMS = None

try:
    from mmcv.transforms import BaseTransform
except Exception:
    class BaseTransform:  # type: ignore
        def __call__(self, results):
            return self.transform(results)


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
DEPTH_EXTS = {'.png', '.tif', '.tiff', '.npy', '.exr'}


# =============================================================================
# Core image operations
# =============================================================================

def _to_float01(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    if x.size > 0 and np.nanmax(x) > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _back_to_like_img(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    if np.issubdtype(ref.dtype, np.integer):
        return np.round(x * 255.0).astype(ref.dtype)
    # MMDetection LoadImageFromFile(to_float32=True) normally returns float32 in [0, 255].
    if ref.size > 0 and np.nanmax(ref) > 1.5:
        return (x * 255.0).astype(ref.dtype)
    return x.astype(ref.dtype)


def _apply_brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return _back_to_like_img(_to_float01(img) * float(factor), img)


def _apply_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    x = _to_float01(img)
    mean = x.mean(axis=(0, 1), keepdims=True)
    y = (x - mean) * float(factor) + mean
    return _back_to_like_img(y, img)


def _apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    if gamma <= 0:
        raise ValueError('gamma must be positive.')
    x = _to_float01(img)
    return _back_to_like_img(np.power(x, float(gamma)), img)


def _apply_white_balance(
    img: np.ndarray,
    r_gain: float,
    g_gain: float,
    b_gain: float,
    channel_order: str = 'bgr',
) -> np.ndarray:
    x = _to_float01(img)
    channel_order = channel_order.lower()
    if channel_order == 'rgb':
        gains = np.array([r_gain, g_gain, b_gain], dtype=np.float32)
    elif channel_order == 'bgr':
        gains = np.array([b_gain, g_gain, r_gain], dtype=np.float32)
    else:
        raise ValueError("channel_order must be 'rgb' or 'bgr'.")
    y = x * gains.reshape(1, 1, 3)
    return _back_to_like_img(y, img)


def _squeeze_depth(depth: np.ndarray) -> Tuple[np.ndarray, bool]:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        return depth[..., 0], True
    return depth, False


def _restore_depth(depth_2d: np.ndarray, had_channel: bool) -> np.ndarray:
    if had_channel:
        return depth_2d[..., None]
    return depth_2d


def _depth_like(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref_2d, had_channel = _squeeze_depth(ref)
    y_2d, _ = _squeeze_depth(y)
    if np.issubdtype(ref_2d.dtype, np.integer):
        info = np.iinfo(ref_2d.dtype)
        out = np.clip(np.round(y_2d), info.min, info.max).astype(ref_2d.dtype)
    else:
        out = y_2d.astype(ref_2d.dtype)
    return _restore_depth(out, had_channel)


def _depth_valid_mask(depth_2d: np.ndarray, invalid_value: Union[int, float]) -> np.ndarray:
    d = depth_2d.astype(np.float32)
    return np.isfinite(d) & (d != invalid_value)


def _apply_depth_blur(
    depth: np.ndarray,
    ksize: int,
    sigma: float,
    invalid_value: Union[int, float] = 0,
    preserve_invalid: bool = True,
) -> np.ndarray:
    d0, had_channel = _squeeze_depth(depth)
    if ksize <= 1:
        return depth.copy()
    if ksize % 2 == 0:
        ksize += 1

    d = d0.astype(np.float32)
    if not preserve_invalid:
        y = cv2.GaussianBlur(d, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        return _restore_depth(_depth_like(y, d0), had_channel)

    valid = _depth_valid_mask(d0, invalid_value)
    valid_f = valid.astype(np.float32)
    d_fill = np.where(valid, d, 0.0).astype(np.float32)

    num = cv2.GaussianBlur(d_fill, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    den = cv2.GaussianBlur(valid_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    y = np.where(den > 1e-6, num / np.maximum(den, 1e-6), invalid_value)
    y[~valid] = invalid_value
    out = _depth_like(y, d0)
    return _restore_depth(out, had_channel)


def _apply_depth_noise(
    depth: np.ndarray,
    sigma: float,
    relative: bool,
    invalid_value: Union[int, float],
    rng: np.random.Generator,
) -> np.ndarray:
    d0, had_channel = _squeeze_depth(depth)
    d = d0.astype(np.float32)
    valid = _depth_valid_mask(d0, invalid_value)

    if relative:
        std = np.abs(d) * float(sigma)
    else:
        std = float(sigma)

    noise = rng.normal(0.0, std, size=d.shape).astype(np.float32)
    y = d.copy()
    y[valid] = d[valid] + noise[valid]
    y[~valid] = invalid_value

    if np.any(valid):
        lo = float(np.nanmin(d[valid]))
        hi = float(np.nanmax(d[valid]))
        y[valid] = np.clip(y[valid], lo, hi)

    out = _depth_like(y, d0)
    return _restore_depth(out, had_channel)


def _apply_depth_random_dropout(
    depth: np.ndarray,
    drop_prob: float,
    invalid_value: Union[int, float],
    rng: np.random.Generator,
) -> np.ndarray:
    d0, had_channel = _squeeze_depth(depth)
    y = d0.copy()
    valid = _depth_valid_mask(d0, invalid_value)
    drop = rng.random(d0.shape[:2]) < float(drop_prob)
    y[valid & drop] = invalid_value
    return _restore_depth(y, had_channel)


def _apply_depth_block_dropout(
    depth: np.ndarray,
    num_blocks: int,
    block_ratio: Tuple[float, float],
    invalid_value: Union[int, float],
    rng: np.random.Generator,
) -> np.ndarray:
    d0, had_channel = _squeeze_depth(depth)
    y = d0.copy()
    h, w = y.shape[:2]
    base = min(h, w)
    lo, hi = block_ratio

    for _ in range(int(num_blocks)):
        bh = max(1, min(h, int(base * rng.uniform(lo, hi))))
        bw = max(1, min(w, int(base * rng.uniform(lo, hi))))
        y0 = int(rng.integers(0, max(1, h - bh + 1)))
        x0 = int(rng.integers(0, max(1, w - bw + 1)))
        y[y0:y0 + bh, x0:x0 + bw] = invalid_value

    return _restore_depth(y, had_channel)


# =============================================================================
# Registered transform
# =============================================================================

class _RGBDCorruptionImpl(BaseTransform):
    """Config-controllable RGB-D corruption transform.

    Expected default keys in your pipeline:
      - RGB image: results['img']
      - depth image: results['aux_img']

    Recommended location in pipeline:
      LoadImageFromFile -> LoadAuxImageFromFile -> RGBDCorruption -> DepthPreprocess
    """

    def __init__(
        self,
        enabled: bool = True,
        mode: str = 'none',
        prob: float = 1.0,

        # keys and channel convention
        img_key: str = 'img',
        depth_key: str = 'aux_img',
        channel_order: str = 'bgr',
        invalid_value: Union[int, float] = 0,

        # RGB params
        brightness: float = 1.0,
        contrast: float = 1.0,
        gamma: float = 1.0,
        r_gain: float = 1.0,
        g_gain: float = 1.0,
        b_gain: float = 1.0,

        # Depth params
        depth_blur_ksize: int = 5,
        depth_blur_sigma: float = 1.5,
        depth_blur_preserve_invalid: bool = True,
        depth_noise_sigma: float = 2.0,
        depth_noise_relative: bool = False,
        depth_dropout_prob: float = 0.05,
        depth_blocks: int = 3,
        depth_block_min: float = 0.05,
        depth_block_max: float = 0.15,

        # reproducibility
        seed: Optional[int] = None,
        record_meta: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.mode = mode
        self.prob = float(prob)
        self.img_key = img_key
        self.depth_key = depth_key
        self.channel_order = channel_order
        self.invalid_value = invalid_value

        self.brightness = brightness
        self.contrast = contrast
        self.gamma = gamma
        self.r_gain = r_gain
        self.g_gain = g_gain
        self.b_gain = b_gain

        self.depth_blur_ksize = depth_blur_ksize
        self.depth_blur_sigma = depth_blur_sigma
        self.depth_blur_preserve_invalid = depth_blur_preserve_invalid
        self.depth_noise_sigma = depth_noise_sigma
        self.depth_noise_relative = depth_noise_relative
        self.depth_dropout_prob = depth_dropout_prob
        self.depth_blocks = depth_blocks
        self.depth_block_min = depth_block_min
        self.depth_block_max = depth_block_max

        self.rng = np.random.default_rng(seed)
        self.record_meta = record_meta

    def _apply_to_arrays(self, img: Optional[np.ndarray], depth: Optional[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        mode = self.mode
        out_img = img
        out_depth = depth

        if mode in ('none', None):
            return out_img, out_depth

        if mode == 'rgb_brightness':
            if img is None:
                raise KeyError(f"RGB key '{self.img_key}' not found.")
            out_img = _apply_brightness(img, self.brightness)
        elif mode == 'rgb_contrast':
            if img is None:
                raise KeyError(f"RGB key '{self.img_key}' not found.")
            out_img = _apply_contrast(img, self.contrast)
        elif mode == 'rgb_gamma':
            if img is None:
                raise KeyError(f"RGB key '{self.img_key}' not found.")
            out_img = _apply_gamma(img, self.gamma)
        elif mode == 'rgb_white_balance':
            if img is None:
                raise KeyError(f"RGB key '{self.img_key}' not found.")
            out_img = _apply_white_balance(
                img,
                r_gain=self.r_gain,
                g_gain=self.g_gain,
                b_gain=self.b_gain,
                channel_order=self.channel_order,
            )
        elif mode == 'depth_blur':
            if depth is None:
                raise KeyError(f"Depth key '{self.depth_key}' not found.")
            out_depth = _apply_depth_blur(
                depth,
                ksize=self.depth_blur_ksize,
                sigma=self.depth_blur_sigma,
                invalid_value=self.invalid_value,
                preserve_invalid=self.depth_blur_preserve_invalid,
            )
        elif mode == 'depth_noise':
            if depth is None:
                raise KeyError(f"Depth key '{self.depth_key}' not found.")
            out_depth = _apply_depth_noise(
                depth,
                sigma=self.depth_noise_sigma,
                relative=self.depth_noise_relative,
                invalid_value=self.invalid_value,
                rng=self.rng,
            )
        elif mode == 'depth_random_dropout':
            if depth is None:
                raise KeyError(f"Depth key '{self.depth_key}' not found.")
            out_depth = _apply_depth_random_dropout(
                depth,
                drop_prob=self.depth_dropout_prob,
                invalid_value=self.invalid_value,
                rng=self.rng,
            )
        elif mode == 'depth_block_dropout':
            if depth is None:
                raise KeyError(f"Depth key '{self.depth_key}' not found.")
            out_depth = _apply_depth_block_dropout(
                depth,
                num_blocks=self.depth_blocks,
                block_ratio=(self.depth_block_min, self.depth_block_max),
                invalid_value=self.invalid_value,
                rng=self.rng,
            )
        else:
            raise ValueError(
                f'Unsupported mode: {mode}. Supported: none, rgb_brightness, '
                'rgb_contrast, rgb_gamma, rgb_white_balance, depth_blur, '
                'depth_noise, depth_random_dropout, depth_block_dropout.'
            )

        return out_img, out_depth

    def transform(self, results: Dict) -> Dict:
        if (not self.enabled) or self.mode in ('none', None):
            return results
        if self.rng.random() > self.prob:
            return results

        img = results.get(self.img_key, None)
        depth = results.get(self.depth_key, None)
        out_img, out_depth = self._apply_to_arrays(img, depth)

        if out_img is not None:
            results[self.img_key] = out_img
        if out_depth is not None:
            results[self.depth_key] = out_depth

        if self.record_meta:
            results.setdefault('rgbd_corruption', {})
            results['rgbd_corruption'].update(self.to_plain_dict())

        return results

    def __call__(self, results: Dict) -> Dict:
        return self.transform(results)

    def to_plain_dict(self) -> Dict:
        return dict(
            enabled=self.enabled,
            mode=self.mode,
            prob=self.prob,
            img_key=self.img_key,
            depth_key=self.depth_key,
            channel_order=self.channel_order,
            invalid_value=self.invalid_value,
            brightness=self.brightness,
            contrast=self.contrast,
            gamma=self.gamma,
            r_gain=self.r_gain,
            g_gain=self.g_gain,
            b_gain=self.b_gain,
            depth_blur_ksize=self.depth_blur_ksize,
            depth_blur_sigma=self.depth_blur_sigma,
            depth_blur_preserve_invalid=self.depth_blur_preserve_invalid,
            depth_noise_sigma=self.depth_noise_sigma,
            depth_noise_relative=self.depth_noise_relative,
            depth_dropout_prob=self.depth_dropout_prob,
            depth_blocks=self.depth_blocks,
            depth_block_min=self.depth_block_min,
            depth_block_max=self.depth_block_max,
        )

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(enabled={self.enabled}, mode={self.mode}, prob={self.prob})'


if TRANSFORMS is not None:
    @TRANSFORMS.register_module()
    class RGBDCorruption(_RGBDCorruptionImpl):
        pass
else:
    class RGBDCorruption(_RGBDCorruptionImpl):
        pass


# =============================================================================
# CLI helpers
# =============================================================================

def imread_rgb_bgr_order(path: Union[str, Path]) -> np.ndarray:
    """Read RGB image in BGR order, matching MMDetection's default loading convention."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Cannot read image: {path}')
    return img.astype(np.float32)


def imwrite_bgr(path: Union[str, Path], img_bgr: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.clip(img_bgr, 0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(path), out)
    if not ok:
        raise IOError(f'Failed to write image: {path}')


def imread_depth(path: Union[str, Path]) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == '.npy':
        return np.load(path)
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(f'Cannot read depth: {path}')
    if d.ndim == 3 and d.shape[-1] != 1:
        d = d[..., 0]
    return d.astype(np.float32)


def imwrite_depth(path: Union[str, Path], depth: np.ndarray, ref_dtype: Optional[np.dtype] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == '.npy':
        np.save(path, depth)
        return

    out = depth
    if ref_dtype is not None and np.issubdtype(ref_dtype, np.integer):
        info = np.iinfo(ref_dtype)
        out = np.clip(np.round(out), info.min, info.max).astype(ref_dtype)
    elif ref_dtype is not None:
        out = out.astype(ref_dtype)

    ok = cv2.imwrite(str(path), out)
    if not ok:
        raise IOError(f'Failed to write depth: {path}')


def list_files(root: Path, exts: Sequence[str]) -> List[Path]:
    return sorted([p for p in root.rglob('*') if p.suffix.lower() in exts])


def find_depth_for_rgb(rgb_path: Path, rgb_dir: Path, depth_dir: Path, aux_suffix: str = '.png') -> Optional[Path]:
    rel = rgb_path.relative_to(rgb_dir)
    candidates = [depth_dir / rel.with_suffix(aux_suffix)]
    candidates += [depth_dir / rel.with_suffix(ext) for ext in DEPTH_EXTS]
    candidates += [depth_dir / f'{rgb_path.stem}{ext}' for ext in DEPTH_EXTS]
    for p in candidates:
        if p.exists():
            return p
    return None


def depth_to_vis(depth: np.ndarray, invalid_value: Union[int, float] = 0) -> np.ndarray:
    d, _ = _squeeze_depth(depth)
    d = d.astype(np.float32)
    valid = np.isfinite(d) & (d != invalid_value)
    if not np.any(valid):
        gray = np.zeros(d.shape[:2], dtype=np.uint8)
    else:
        lo, hi = np.percentile(d[valid], [1, 99])
        if hi <= lo:
            hi = lo + 1
        gray = np.clip((d - lo) / (hi - lo), 0, 1)
        gray = (gray * 255).astype(np.uint8)
        gray[~valid] = 0
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def label_panel(img_bgr: np.ndarray, text: str) -> np.ndarray:
    out = np.clip(img_bgr, 0, 255).astype(np.uint8).copy()
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def make_preview(
    rgb_before: Optional[np.ndarray],
    rgb_after: Optional[np.ndarray],
    depth_before: Optional[np.ndarray],
    depth_after: Optional[np.ndarray],
    mode: str,
    invalid_value: Union[int, float],
) -> np.ndarray:
    panels = []
    ref_hw = None
    if rgb_before is not None and rgb_after is not None:
        panels.append(label_panel(rgb_before, 'RGB original'))
        panels.append(label_panel(rgb_after, f'RGB {mode}'))
        ref_hw = rgb_before.shape[:2]

    if depth_before is not None and depth_after is not None:
        d0 = depth_to_vis(depth_before, invalid_value=invalid_value)
        d1 = depth_to_vis(depth_after, invalid_value=invalid_value)
        if ref_hw is not None:
            h, w = ref_hw
            d0 = cv2.resize(d0, (w, h), interpolation=cv2.INTER_NEAREST)
            d1 = cv2.resize(d1, (w, h), interpolation=cv2.INTER_NEAREST)
        panels.append(label_panel(d0, 'Depth original'))
        panels.append(label_panel(d1, f'Depth {mode}'))

    if not panels:
        raise ValueError('Nothing to preview.')
    return np.concatenate(panels, axis=1)


def add_transform_args(p: argparse.ArgumentParser) -> None:
    p.add_argument('--mode', default='none', choices=[
        'none',
        'rgb_brightness',
        'rgb_contrast',
        'rgb_gamma',
        'rgb_white_balance',
        'depth_blur',
        'depth_noise',
        'depth_random_dropout',
        'depth_block_dropout',
    ])
    p.add_argument('--prob', type=float, default=1.0)
    p.add_argument('--channel-order', default='bgr', choices=['bgr', 'rgb'])
    p.add_argument('--invalid-value', type=float, default=0)

    p.add_argument('--brightness', type=float, default=1.0)
    p.add_argument('--contrast', type=float, default=1.0)
    p.add_argument('--gamma', type=float, default=1.0)
    p.add_argument('--r-gain', type=float, default=1.0)
    p.add_argument('--g-gain', type=float, default=1.0)
    p.add_argument('--b-gain', type=float, default=1.0)

    p.add_argument('--depth-blur-ksize', type=int, default=5)
    p.add_argument('--depth-blur-sigma', type=float, default=1.5)
    p.add_argument('--no-depth-blur-preserve-invalid', action='store_true')
    p.add_argument('--depth-noise-sigma', type=float, default=2.0)
    p.add_argument('--depth-noise-relative', action='store_true')
    p.add_argument('--depth-dropout-prob', type=float, default=0.05)
    p.add_argument('--depth-blocks', type=int, default=3)
    p.add_argument('--depth-block-min', type=float, default=0.05)
    p.add_argument('--depth-block-max', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)


def transform_from_args(args: argparse.Namespace) -> RGBDCorruption:
    return RGBDCorruption(
        enabled=True,
        mode=args.mode,
        prob=args.prob,
        img_key='img',
        depth_key='aux_img',
        channel_order=args.channel_order,
        invalid_value=args.invalid_value,
        brightness=args.brightness,
        contrast=args.contrast,
        gamma=args.gamma,
        r_gain=args.r_gain,
        g_gain=args.g_gain,
        b_gain=args.b_gain,
        depth_blur_ksize=args.depth_blur_ksize,
        depth_blur_sigma=args.depth_blur_sigma,
        depth_blur_preserve_invalid=not args.no_depth_blur_preserve_invalid,
        depth_noise_sigma=args.depth_noise_sigma,
        depth_noise_relative=args.depth_noise_relative,
        depth_dropout_prob=args.depth_dropout_prob,
        depth_blocks=args.depth_blocks,
        depth_block_min=args.depth_block_min,
        depth_block_max=args.depth_block_max,
        seed=args.seed,
    )


def maybe_load_transform_from_config(args: argparse.Namespace) -> Optional[RGBDCorruption]:
    """Optional: build transform from cfg.rgbd_corruption_cfg if --mmdet-config is provided."""
    cfg_path = getattr(args, 'mmdet_config', None)
    if not cfg_path:
        return None
    try:
        from mmengine.config import Config
        cfg = Config.fromfile(cfg_path)
        raw = dict(cfg.get('rgbd_corruption_cfg', {}))
        if not raw:
            print('[WARN] No rgbd_corruption_cfg found in config. Falling back to CLI args.')
            return None
        raw.pop('type', None)
        raw['enabled'] = True
        # CLI --mode can override config mode if explicitly set to a non-default value.
        if getattr(args, 'mode', 'none') != 'none':
            raw['mode'] = args.mode
        return RGBDCorruption(**raw)
    except Exception as e:
        print(f'[WARN] Failed to load mmdet config: {e}. Falling back to CLI args.')
        return None


def choose_samples(files: Sequence[Path], num_samples: int, seed: int) -> List[Path]:
    files = list(files)
    if num_samples <= 0 or num_samples >= len(files):
        return files
    rng = random.Random(seed)
    return rng.sample(files, num_samples)


def cmd_preview(args: argparse.Namespace) -> None:
    rgb_dir = Path(args.rgb_dir) if args.rgb_dir else None
    depth_dir = Path(args.depth_dir) if args.depth_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t = maybe_load_transform_from_config(args) or transform_from_args(args)

    if rgb_dir is not None:
        rgb_files = list_files(rgb_dir, IMG_EXTS)
        samples = choose_samples(rgb_files, args.num_samples, args.seed)
    elif depth_dir is not None:
        depth_files = list_files(depth_dir, DEPTH_EXTS)
        samples = choose_samples(depth_files, args.num_samples, args.seed)
    else:
        raise ValueError('Provide --rgb-dir and/or --depth-dir.')

    manifest = []
    for i, p in enumerate(samples):
        rgb = None
        depth = None
        depth_path = None

        if rgb_dir is not None:
            rgb_path = p
            rgb = imread_rgb_bgr_order(rgb_path)
            if depth_dir is not None:
                depth_path = find_depth_for_rgb(rgb_path, rgb_dir, depth_dir, aux_suffix=args.aux_suffix)
                if depth_path is not None:
                    depth = imread_depth(depth_path)
        else:
            rgb_path = None
            depth_path = p
            depth = imread_depth(depth_path)

        results = {}
        if rgb is not None:
            results['img'] = rgb.copy()
        if depth is not None:
            results['aux_img'] = depth.copy()

        out = t(results)
        rgb_after = out.get('img', None)
        depth_after = out.get('aux_img', None)

        panel = make_preview(rgb, rgb_after, depth, depth_after, t.mode, t.invalid_value)
        out_path = out_dir / f'preview_{i:03d}_{p.stem}_{t.mode}.jpg'
        imwrite_bgr(out_path, panel)

        manifest.append({
            'preview': str(out_path),
            'rgb_path': str(rgb_path) if rgb_dir is not None else None,
            'depth_path': str(depth_path) if depth_path is not None else None,
            'corruption': t.to_plain_dict(),
        })
        print(f'Wrote {out_path}')

    (out_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote manifest: {out_dir / "manifest.json"}')


def cmd_export(args: argparse.Namespace) -> None:
    rgb_dir = Path(args.rgb_dir) if args.rgb_dir else None
    depth_dir = Path(args.depth_dir) if args.depth_dir else None
    out_rgb_dir = Path(args.out_rgb_dir) if args.out_rgb_dir else None
    out_depth_dir = Path(args.out_depth_dir) if args.out_depth_dir else None

    t = maybe_load_transform_from_config(args) or transform_from_args(args)

    if rgb_dir is None and depth_dir is None:
        raise ValueError('Provide --rgb-dir and/or --depth-dir.')

    manifest = []

    if rgb_dir is not None and out_rgb_dir is not None:
        for p in list_files(rgb_dir, IMG_EXTS):
            rgb = imread_rgb_bgr_order(p)
            results = {'img': rgb.copy()}
            if depth_dir is not None:
                dp = find_depth_for_rgb(p, rgb_dir, depth_dir, aux_suffix=args.aux_suffix)
                if dp is not None:
                    results['aux_img'] = imread_depth(dp)
            out = t(results)
            rel = p.relative_to(rgb_dir)
            imwrite_bgr(out_rgb_dir / rel, out['img'])
            manifest.append({'rgb': str(p), 'out_rgb': str(out_rgb_dir / rel)})

    if depth_dir is not None and out_depth_dir is not None:
        for p in list_files(depth_dir, DEPTH_EXTS):
            depth_raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.suffix.lower() != '.npy' else np.load(p)
            depth = imread_depth(p)
            results = {'aux_img': depth.copy()}
            out = t(results)
            rel = p.relative_to(depth_dir)
            ref_dtype = depth_raw.dtype if isinstance(depth_raw, np.ndarray) else depth.dtype
            imwrite_depth(out_depth_dir / rel, out['aux_img'], ref_dtype=ref_dtype)
            manifest.append({'depth': str(p), 'out_depth': str(out_depth_dir / rel)})

    manifest_path = (out_rgb_dir or out_depth_dir or Path('.')) / 'corruption_manifest.json'
    manifest_path.write_text(json.dumps({
        'corruption': t.to_plain_dict(),
        'files': manifest,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote manifest: {manifest_path}')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Unified RGB-D corruption transform and CLI preview/export tool.')
    sub = parser.add_subparsers(dest='command', required=True)

    p1 = sub.add_parser('preview', help='Preview corrupted RGB-D samples using the same transform class as MMDetection.')
    p1.add_argument('--rgb-dir', default=None)
    p1.add_argument('--depth-dir', default=None)
    p1.add_argument('--out-dir', required=True)
    p1.add_argument('--num-samples', type=int, default=10)
    p1.add_argument('--aux-suffix', default='.png')
    p1.add_argument('--mmdet-config', default=None, help='Optional config path. Uses rgbd_corruption_cfg if present.')
    add_transform_args(p1)
    p1.set_defaults(func=cmd_preview)

    p2 = sub.add_parser('export', help='Export corrupted RGB/depth copies.')
    p2.add_argument('--rgb-dir', default=None)
    p2.add_argument('--depth-dir', default=None)
    p2.add_argument('--out-rgb-dir', default=None)
    p2.add_argument('--out-depth-dir', default=None)
    p2.add_argument('--aux-suffix', default='.png')
    p2.add_argument('--mmdet-config', default=None, help='Optional config path. Uses rgbd_corruption_cfg if present.')
    add_transform_args(p2)
    p2.set_defaults(func=cmd_export)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
