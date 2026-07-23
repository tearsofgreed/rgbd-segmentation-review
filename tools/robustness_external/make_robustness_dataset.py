#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline RGB-D Robustness Dataset Generator (External / Zero-Invasive).

Supported corruption modes:
  clean, rgb_brightness, rgb_white_balance, rgb_local_shadow,
  rgb_specular_highlight, depth_random_holes, depth_block_holes,
  depth_edge_holes, depth_holes_with_fill
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Copy helpers
# ============================================================================

def _copy_file(src: str, dst: str, mode: str, overwrite: bool) -> str:
    os.makedirs(osp.dirname(dst), exist_ok=True)
    if osp.exists(dst) and not overwrite:
        return "skip"
    if osp.exists(dst) or osp.islink(dst):
        os.unlink(dst)
    methods = ["symlink", "hardlink", "copy"] if mode == "auto" else [mode]
    last_err = None
    for m in methods:
        try:
            if m == "symlink":
                os.symlink(osp.abspath(src), dst); return "symlink"
            elif m == "hardlink":
                os.link(src, dst); return "hardlink"
            elif m == "copy":
                shutil.copy2(src, dst); return "copy"
        except OSError as e:
            last_err = e; continue
    raise OSError(f"Failed copy {src} -> {dst}: {last_err}")


def _copy_annotation(src_json: str, dst_json: str) -> None:
    os.makedirs(osp.dirname(dst_json), exist_ok=True)
    shutil.copy2(src_json, dst_json)

def save_json(obj, path):
    """Save object as JSON with UTF-8 encoding (safe for Windows non-ASCII paths)."""
    import json
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)




# ============================================================================
# Windows-safe image I/O (handles Chinese paths)
# ============================================================================

def imread_unicode(path, flags):
    """Read an image with cv2.imdecode, safe for Windows paths with non-ASCII chars."""
    import cv2
    import numpy as np
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")

    with p.open("rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)

    if data.size == 0:
        raise IOError(f"Empty or unreadable file: {p}")

    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"cv2.imdecode failed for flags={flags}: {p}")

    return img


def imwrite_unicode(path, img, params=None):
    """Write an image with cv2.imencode, safe for Windows paths with non-ASCII chars.

    Important:
    Do not use ndarray.tofile(path) here. On Windows, numpy.tofile can fail
    with OSError [Errno 22] under non-ASCII project paths. Use Python's
    Unicode-aware file handle instead.
    """
    import cv2
    import os
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ext = p.suffix
    if not ext:
        ext = ".png"

    if params is None:
        params = []

    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"cv2.imencode failed: {p}")

    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("wb") as f:
        f.write(buf.tobytes())

    os.replace(str(tmp), str(p))
    return True

# ============================================================================
# RGB corruption functions
# ============================================================================

def apply_rgb_brightness(img_bgr: "np.ndarray", factor: float) -> "np.ndarray":
    import numpy as np
    if factor <= 0:
        raise ValueError(f"brightness_factor must be > 0, got {factor}")
    if factor == 1.0:
        return img_bgr.copy()
    img_f = img_bgr.astype(np.float32) * float(factor)
    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8)


def apply_rgb_white_balance(
    img_bgr: "np.ndarray", r_gain: float, g_gain: float, b_gain: float
) -> "np.ndarray":
    """Multiply BGR channels: B * b_gain, G * g_gain, R * r_gain."""
    import numpy as np
    img_f = img_bgr.astype(np.float32)
    img_f[:, :, 0] *= b_gain  # B
    img_f[:, :, 1] *= g_gain  # G
    img_f[:, :, 2] *= r_gain  # R
    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8)


def _generate_soft_blob_mask(
    h: int, w: int, num_blobs: int, area_ratio: float,
    soft_sigma: float, rng: "np.random.Generator",
) -> "np.ndarray":
    """Generate soft blob mask [0,1] HxW."""
    import numpy as np
    import cv2
    mask = np.zeros((h, w), dtype=np.float32)
    base = min(h, w)
    for _ in range(num_blobs):
        cx = rng.integers(w // 6, 5 * w // 6)
        cy = rng.integers(h // 6, 5 * h // 6)
        a = rng.integers(max(10, int(base * 0.03)), max(11, int(base * 0.15)))
        b = rng.integers(max(8, int(base * 0.02)), max(9, int(base * 0.12)))
        angle = rng.uniform(0, 2 * np.pi)
        yy, xx = np.ogrid[:h, :w]
        xr = (xx - cx) * np.cos(angle) + (yy - cy) * np.sin(angle)
        yr = -(xx - cx) * np.sin(angle) + (yy - cy) * np.cos(angle)
        blob = ((xr / a) ** 2 + (yr / b) ** 2 <= 1.0).astype(np.float32)
        mask = np.maximum(mask, blob)
    if soft_sigma > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=soft_sigma, sigmaY=soft_sigma)
    mask = np.clip(mask, 0, 1)
    # Limit area
    if mask.sum() > area_ratio * h * w * 2:
        thr = np.percentile(mask[mask > 0], 50)
        mask[mask < thr] = 0
    return mask.astype(np.float32)


def apply_rgb_local_shadow(
    img_bgr: "np.ndarray", shadow_factor: float, area_ratio: float,
    num_blobs: int, soft_sigma: float, rng: "np.random.Generator",
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Returns (corrupted_img, shadow_mask)."""
    import cv2
    import numpy as np
    h, w = img_bgr.shape[:2]
    blob_mask = _generate_soft_blob_mask(h, w, num_blobs, area_ratio, soft_sigma, rng)
    shadow_mask = 1.0 - blob_mask * (1.0 - shadow_factor)
    img_f = img_bgr.astype(np.float32)
    for c in range(3):
        img_f[:, :, c] = img_f[:, :, c] * shadow_mask
    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8), shadow_mask.astype(np.float32)


def apply_rgb_specular_highlight(
    img_bgr: "np.ndarray", highlight_value: float, area_ratio: float,
    num_blobs: int, soft_sigma: float, rng: "np.random.Generator",
) -> Tuple["np.ndarray", "np.ndarray"]:
    """Returns (corrupted_img, highlight_mask)."""
    import numpy as np
    h, w = img_bgr.shape[:2]
    mask = _generate_soft_blob_mask(h, w, num_blobs, area_ratio, soft_sigma, rng)
    alpha = mask[:, :, None]
    img_f = img_bgr.astype(np.float32)
    img_f = (1 - alpha) * img_f + alpha * highlight_value
    img_f = np.clip(img_f, 0, 255)
    return img_f.astype(np.uint8), mask.astype(np.float32)


# ============================================================================
# Depth corruption functions
# ============================================================================

DEPTH_VALID_MIN = 2.0
DEPTH_INVALID_VAL = 0


def _depth_valid(depth: "np.ndarray") -> "np.ndarray":
    import numpy as np
    d = depth.astype(np.float32)
    return (d > DEPTH_VALID_MIN) & np.isfinite(d)


def apply_depth_random_holes(
    depth: "np.ndarray", hole_ratio: float, rng: "np.random.Generator",
) -> Tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    h, w = depth.shape[:2]
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    valid = _depth_valid(depth)
    hole_mask = np.zeros((h, w), dtype=bool)
    n_valid = valid.sum()
    n_holes = max(1, int(n_valid * hole_ratio))
    valid_idx = np.where(valid.ravel())[0]
    chosen = rng.choice(valid_idx, size=min(n_holes, len(valid_idx)), replace=False)
    hole_mask.ravel()[chosen] = True
    corrupted = depth.copy()
    corrupted[hole_mask] = DEPTH_INVALID_VAL
    return corrupted, hole_mask.astype(np.uint8)


def apply_depth_block_holes(
    depth: "np.ndarray", num_blocks: int, block_min_ratio: float,
    block_max_ratio: float, rng: "np.random.Generator",
) -> Tuple["np.ndarray", "np.ndarray"]:
    import numpy as np
    h, w = depth.shape[:2]
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    valid = _depth_valid(depth)
    base = min(h, w)
    hole_mask = np.zeros((h, w), dtype=bool)
    corrupted = depth.copy()
    for _ in range(int(num_blocks)):
        bh = max(1, min(h, int(base * rng.uniform(block_min_ratio, block_max_ratio))))
        bw = max(1, min(w, int(base * rng.uniform(block_min_ratio, block_max_ratio))))
        y0 = int(rng.integers(0, max(1, h - bh + 1)))
        x0 = int(rng.integers(0, max(1, w - bw + 1)))
        block_valid = valid[y0:y0 + bh, x0:x0 + bw]
        hole_mask[y0:y0 + bh, x0:x0 + bw] = block_valid
        corrupted[y0:y0 + bh, x0:x0 + bw] = np.where(
            block_valid, DEPTH_INVALID_VAL, corrupted[y0:y0 + bh, x0:x0 + bw]
        )
    return corrupted, hole_mask.astype(np.uint8)


def apply_depth_edge_holes(
    depth: "np.ndarray", edge_frac: float, hole_prob: float,
    rng: "np.random.Generator",
) -> Tuple["np.ndarray", "np.ndarray"]:
    import cv2
    import numpy as np
    h, w = depth.shape[:2]
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    valid = _depth_valid(depth)
    d_f = depth.astype(np.float32)
    grad_x = cv2.Sobel(d_f, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(d_f, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    grad_mag[~valid] = 0
    valid_grad = grad_mag[valid]
    if valid_grad.size < 10:
        return depth.copy(), np.zeros((h, w), dtype=np.uint8)
    threshold = np.percentile(valid_grad, 100 - edge_frac * 100)
    edge_mask = (grad_mag > threshold) & valid
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    edge_dilated = cv2.dilate(edge_mask.astype(np.uint8), kernel).astype(bool)
    drop = (rng.random((h, w)) < hole_prob) & edge_dilated & valid
    corrupted = depth.copy()
    corrupted[drop] = DEPTH_INVALID_VAL
    return corrupted, drop.astype(np.uint8)


def fill_depth_median(
    depth: "np.ndarray", hole_mask: "np.ndarray", kernel_size: int = 11,
) -> "np.ndarray":
    import numpy as np
    h, w = depth.shape
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    filled = depth.copy().astype(np.float32)
    half = kernel_size // 2
    hole_ys, hole_xs = np.where(hole_mask)
    for y, x in zip(hole_ys, hole_xs):
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        x0, x1 = max(0, x - half), min(w, x + half + 1)
        patch = filled[y0:y1, x0:x1]
        pv = patch[(patch > DEPTH_VALID_MIN) & (patch != DEPTH_INVALID_VAL)]
        if len(pv) > 0:
            filled[y, x] = np.median(pv)
        else:
            filled[y, x] = DEPTH_INVALID_VAL
    return np.clip(filled, 0, None).astype(depth.dtype)


def fill_depth_inpaint(
    depth: "np.ndarray", hole_mask: "np.ndarray", method: str = "telea",
) -> "np.ndarray":
    import cv2
    import numpy as np
    h, w = depth.shape
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    valid = _depth_valid(depth) & (hole_mask == 0)
    if not valid.any():
        return depth.copy()
    vmin = np.percentile(depth[valid], 1)
    vmax = np.percentile(depth[valid], 99)
    if vmax <= vmin:
        return depth.copy()
    depth_norm = np.zeros((h, w), dtype=np.uint8)
    depth_norm[valid] = np.clip(
        (depth[valid].astype(np.float32) - vmin) / (vmax - vmin) * 255, 0, 255
    ).astype(np.uint8)
    inpaint_mask = (hole_mask.astype(np.uint8) * 255)
    if method == "telea":
        filled_norm = cv2.inpaint(depth_norm, inpaint_mask, 3, cv2.INPAINT_TELEA)
    else:
        filled_norm = cv2.inpaint(depth_norm, inpaint_mask, 3, cv2.INPAINT_NS)
    filled = depth.copy().astype(np.float32)
    filled[hole_mask > 0] = filled_norm[hole_mask > 0].astype(np.float32) / 255.0 * (vmax - vmin) + vmin
    return np.clip(filled, 0, None).astype(depth.dtype)


def _deterministic_seed(file_stem: str, global_seed: int, corruption: str, severity: int) -> int:
    """Generate a deterministic per-image seed from file name + global seed."""
    h = hash(f"{file_stem}_{global_seed}_{corruption}_{severity}")
    return h % (2 ** 31)


# ============================================================================
# Sanity check
# ============================================================================

def save_sanity_image(
    src_img: str, dst_img: str, src_depth: str, dst_depth: str,
    out_path: str, corruption: str, mask_src: Optional[str] = None,
    extra_info: str = "",
) -> None:
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb_src = imread_unicode(src_img, cv2.IMREAD_COLOR)
    rgb_dst = imread_unicode(dst_img, cv2.IMREAD_COLOR)
    d_src = imread_unicode(src_depth, cv2.IMREAD_UNCHANGED)
    d_dst = imread_unicode(dst_depth, cv2.IMREAD_UNCHANGED)
    mask = imread_unicode(mask_src, cv2.IMREAD_GRAYSCALE) if mask_src else None

    ncols = 3 if mask is not None else 2
    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))

    def show_rgb(ax, img, title):
        if img is not None: ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=9); ax.axis("off")

    def show_depth(ax, d, title):
        if d is not None:
            dv = d.astype(np.float32); vv = dv[dv > 0]
            vm = np.percentile(vv, 1) if vv.size > 0 else 0
            vx = np.percentile(vv, 99) if vv.size > 0 else 1
            ax.imshow(dv, cmap="inferno", vmin=vm, vmax=vx)
        ax.set_title(title, fontsize=9); ax.axis("off")

    show_rgb(axes[0, 0], rgb_src, "Original RGB")
    show_rgb(axes[0, 1], rgb_dst, f"Corrupted ({corruption})")
    if mask is not None:
        axes[0, 2].imshow(mask, cmap="gray")
        axes[0, 2].set_title("Mask", fontsize=9); axes[0, 2].axis("off")

    show_depth(axes[1, 0], d_src, "Original Depth")
    show_depth(axes[1, 1], d_dst, "Depth Output")

    fig.suptitle(f"{corruption} {extra_info}", fontsize=11)
    plt.tight_layout()
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Corruption dispatcher for hole+fill mode
# ============================================================================

def _generate_hole(
    depth: "np.ndarray", hole_type: str, params: Dict, rng: "np.random.Generator",
) -> Tuple["np.ndarray", "np.ndarray"]:
    if hole_type == "random":
        return apply_depth_random_holes(depth, params["hole_ratio"], rng)
    elif hole_type == "block":
        return apply_depth_block_holes(
            depth, params["block_num"], params["block_min_ratio"],
            params["block_max_ratio"], rng)
    elif hole_type == "edge":
        return apply_depth_edge_holes(
            depth, params["edge_frac"], params["edge_hole_prob"], rng)
    raise ValueError(f"Unknown hole_type: {hole_type}")


# ============================================================================
# Main dataset builder
# ============================================================================

def build_dataset(args: "argparse.Namespace") -> Dict:
    import cv2
    import numpy as np

    data_root = osp.abspath(args.data_root)
    ann_file = args.ann_file if osp.isabs(args.ann_file) else osp.join(data_root, args.ann_file)
    out_root = osp.abspath(args.out_root)
    corruption = args.corruption
    severity = args.severity
    seed = args.seed if args.seed is not None else 42
    max_samples = args.max_samples
    copy_mode = args.copy_mode
    overwrite = args.overwrite

    if not osp.isfile(ann_file):
        raise FileNotFoundError(f"Annotation not found: {ann_file}")

    with open(ann_file, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    if not images:
        raise ValueError("No images in annotation.")
    if max_samples is not None and max_samples > 0:
        images = images[:int(max_samples)]

    n_total = len(images)
    n_rgb_ok = 0; n_rgb_corrupted = 0; n_depth_ok = 0; n_depth_corrupted = 0

    os.makedirs(osp.join(out_root, "images"), exist_ok=True)
    os.makedirs(osp.join(out_root, "depth"), exist_ok=True)
    os.makedirs(osp.join(out_root, "annotations"), exist_ok=True)
    if corruption in ("depth_random_holes", "depth_block_holes", "depth_edge_holes",
                       "depth_holes_with_fill", "rgb_local_shadow", "rgb_specular_highlight"):
        os.makedirs(osp.join(out_root, "masks"), exist_ok=True)

    sanity_dir = osp.join(out_root, "sanity")
    if args.save_sanity:
        os.makedirs(sanity_dir, exist_ok=True)

    for idx, img_info in enumerate(images):
        file_name = img_info["file_name"]
        stem, ext = osp.splitext(file_name)
        img_seed = _deterministic_seed(stem, seed, corruption, severity)
        rng = np.random.default_rng(img_seed)

        src_img = osp.join(data_root, "images", file_name)
        src_depth = osp.join(data_root, "depth", stem + ".png")
        dst_img = osp.join(out_root, "images", file_name)
        dst_depth = osp.join(out_root, "depth", stem + ".png")

        if not osp.isfile(src_img):
            raise FileNotFoundError(f"RGB missing: {src_img}")
        if not osp.isfile(src_depth):
            raise FileNotFoundError(f"Depth missing: {src_depth}")

        mask_path: Optional[str] = None

        # ---- Clean ----
        if corruption == "clean":
            _copy_file(src_img, dst_img, copy_mode, overwrite)
            _copy_file(src_depth, dst_depth, copy_mode, overwrite)
            n_rgb_ok += 1; n_depth_ok += 1

        # ---- RGB brightness ----
        elif corruption == "rgb_brightness":
            img_bgr = imread_unicode(src_img, cv2.IMREAD_COLOR)
            corrupted = apply_rgb_brightness(img_bgr, args.brightness_factor)
            imwrite_unicode(dst_img, corrupted)
            _copy_file(src_depth, dst_depth, copy_mode, overwrite)
            n_rgb_corrupted += 1; n_depth_ok += 1

        # ---- RGB white balance ----
        elif corruption == "rgb_white_balance":
            img_bgr = imread_unicode(src_img, cv2.IMREAD_COLOR)
            corrupted = apply_rgb_white_balance(
                img_bgr, args.r_gain, args.g_gain, args.b_gain)
            imwrite_unicode(dst_img, corrupted)
            _copy_file(src_depth, dst_depth, copy_mode, overwrite)
            n_rgb_corrupted += 1; n_depth_ok += 1

        # ---- RGB local shadow ----
        elif corruption == "rgb_local_shadow":
            img_bgr = imread_unicode(src_img, cv2.IMREAD_COLOR)
            corrupted, shadow_mask = apply_rgb_local_shadow(
                img_bgr, args.shadow_factor, args.shadow_area_ratio,
                args.shadow_num_blobs, 20.0, rng)
            imwrite_unicode(dst_img, corrupted)
            mask_path = osp.join(out_root, "masks", stem + ".png")
            imwrite_unicode(mask_path, (shadow_mask * 255).astype(np.uint8))
            _copy_file(src_depth, dst_depth, copy_mode, overwrite)
            n_rgb_corrupted += 1; n_depth_ok += 1

        # ---- RGB specular highlight ----
        elif corruption == "rgb_specular_highlight":
            img_bgr = imread_unicode(src_img, cv2.IMREAD_COLOR)
            corrupted, hl_mask = apply_rgb_specular_highlight(
                img_bgr, args.highlight_value, args.highlight_area_ratio,
                args.highlight_num_blobs, 10.0, rng)
            imwrite_unicode(dst_img, corrupted)
            mask_path = osp.join(out_root, "masks", stem + ".png")
            imwrite_unicode(mask_path, (hl_mask * 255).astype(np.uint8))
            _copy_file(src_depth, dst_depth, copy_mode, overwrite)
            n_rgb_corrupted += 1; n_depth_ok += 1

        # ---- Depth random holes ----
        elif corruption == "depth_random_holes":
            d = imread_unicode(src_depth, cv2.IMREAD_UNCHANGED)
            corrupted, hole_mask = apply_depth_random_holes(d, args.hole_ratio, rng)
            imwrite_unicode(dst_depth, corrupted)
            mask_path = osp.join(out_root, "masks", stem + ".png")
            imwrite_unicode(mask_path, hole_mask * 255)
            _copy_file(src_img, dst_img, copy_mode, overwrite)
            n_rgb_ok += 1; n_depth_corrupted += 1

        # ---- Depth block holes ----
        elif corruption == "depth_block_holes":
            d = imread_unicode(src_depth, cv2.IMREAD_UNCHANGED)
            corrupted, hole_mask = apply_depth_block_holes(
                d, args.block_num, args.block_min_ratio, args.block_max_ratio, rng)
            imwrite_unicode(dst_depth, corrupted)
            mask_path = osp.join(out_root, "masks", stem + ".png")
            imwrite_unicode(mask_path, hole_mask * 255)
            _copy_file(src_img, dst_img, copy_mode, overwrite)
            n_rgb_ok += 1; n_depth_corrupted += 1

        # ---- Depth edge holes ----
        elif corruption == "depth_edge_holes":
            d = imread_unicode(src_depth, cv2.IMREAD_UNCHANGED)
            corrupted, hole_mask = apply_depth_edge_holes(
                d, args.edge_frac, args.edge_hole_prob, rng)
            imwrite_unicode(dst_depth, corrupted)
            mask_path = osp.join(out_root, "masks", stem + ".png")
            imwrite_unicode(mask_path, hole_mask * 255)
            _copy_file(src_img, dst_img, copy_mode, overwrite)
            n_rgb_ok += 1; n_depth_corrupted += 1

        # ---- Depth holes with fill ----
        elif corruption == "depth_holes_with_fill":
            d = imread_unicode(src_depth, cv2.IMREAD_UNCHANGED)
            hole_params = {
                "hole_ratio": args.hole_ratio,
                "block_num": args.block_num,
                "block_min_ratio": args.block_min_ratio,
                "block_max_ratio": args.block_max_ratio,
                "edge_frac": args.edge_frac,
                "edge_hole_prob": args.edge_hole_prob,
            }
            hole_d, hole_mask_arr = _generate_hole(d, args.hole_type, hole_params, rng)
            mask_path = osp.join(out_root, "masks", stem + ".png")
            imwrite_unicode(mask_path, hole_mask_arr.astype(np.uint8) * 255)

            if args.fill_method == "none":
                imwrite_unicode(dst_depth, hole_d)
            elif args.fill_method == "median":
                filled = fill_depth_median(hole_d, hole_mask_arr)
                imwrite_unicode(dst_depth, filled)
            elif args.fill_method.startswith("inpaint"):
                m = "telea" if "telea" in args.fill_method else "ns"
                filled = fill_depth_inpaint(hole_d, hole_mask_arr, m)
                imwrite_unicode(dst_depth, filled)
            else:
                raise ValueError(f"Unknown fill_method: {args.fill_method}")
            _copy_file(src_img, dst_img, copy_mode, overwrite)
            n_rgb_ok += 1; n_depth_corrupted += 1

        else:
            raise ValueError(f"Unsupported corruption: {corruption}")

        # ---- Sanity ----
        if args.save_sanity and idx < int(args.sanity_num):
            sp = osp.join(sanity_dir, f"{idx:03d}_{stem}.png")
            try:
                save_sanity_image(
                    src_img, dst_img, src_depth, dst_depth, sp,
                    corruption=corruption, mask_src=mask_path,
                    extra_info=f"s{severity} {getattr(args,'fill_method','')}",
                )
            except Exception as e:
                print(f"  [WARN] sanity: {e}")

        if (idx + 1) % 50 == 0 or idx < 3 or idx == n_total - 1:
            print(f"  [{idx + 1}/{n_total}] {file_name}")

    # Annotation
    dst_ann = osp.join(out_root, "annotations", "val.json")
    _copy_annotation(ann_file, dst_ann)

    # Metadata
    metadata = {
        "corruption": corruption, "severity": severity,
        "brightness_factor": getattr(args, "brightness_factor", None),
        "r_gain": getattr(args, "r_gain", None), "g_gain": getattr(args, "g_gain", None),
        "b_gain": getattr(args, "b_gain", None),
        "shadow_factor": getattr(args, "shadow_factor", None),
        "shadow_area_ratio": getattr(args, "shadow_area_ratio", None),
        "shadow_num_blobs": getattr(args, "shadow_num_blobs", None),
        "highlight_value": getattr(args, "highlight_value", None),
        "highlight_area_ratio": getattr(args, "highlight_area_ratio", None),
        "highlight_num_blobs": getattr(args, "highlight_num_blobs", None),
        "hole_ratio": getattr(args, "hole_ratio", None),
        "block_num": getattr(args, "block_num", None),
        "block_min_ratio": getattr(args, "block_min_ratio", None),
        "block_max_ratio": getattr(args, "block_max_ratio", None),
        "edge_frac": getattr(args, "edge_frac", None),
        "edge_hole_prob": getattr(args, "edge_hole_prob", None),
        "hole_type": getattr(args, "hole_type", None),
        "fill_method": getattr(args, "fill_method", None),
        "seed": seed, "data_root": data_root, "ann_file": ann_file,
        "out_root": out_root, "n_images": n_total,
        "n_rgb_ok": n_rgb_ok, "n_rgb_corrupted": n_rgb_corrupted,
        "n_depth_ok": n_depth_ok, "n_depth_corrupted": n_depth_corrupted,
        "copy_mode": copy_mode, "created_time": datetime.now().isoformat(),
    }
    save_json({k: v for k, v in metadata.items() if v is not None},
              osp.join(out_root, "metadata.json"))
    return metadata


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> "argparse.Namespace":
    p = argparse.ArgumentParser(description="Offline RGB-D Robustness Dataset Generator")
    p.add_argument("--config", default=None, help="(Reserved)")
    p.add_argument("--data-root", required=True)
    p.add_argument("--ann-file", default="annotations/val.json")
    p.add_argument("--out-root", required=True)
    p.add_argument("--corruption", default="clean",
                   choices=["clean", "rgb_brightness", "rgb_white_balance",
                            "rgb_local_shadow", "rgb_specular_highlight",
                            "depth_random_holes", "depth_block_holes",
                            "depth_edge_holes", "depth_holes_with_fill"])
    p.add_argument("--severity", type=int, default=0)
    p.add_argument("--brightness-factor", type=float, default=0.70)
    p.add_argument("--r-gain", type=float, default=1.0)
    p.add_argument("--g-gain", type=float, default=1.0)
    p.add_argument("--b-gain", type=float, default=1.0)
    p.add_argument("--shadow-factor", type=float, default=0.50)
    p.add_argument("--shadow-area-ratio", type=float, default=0.20)
    p.add_argument("--shadow-num-blobs", type=int, default=3)
    p.add_argument("--highlight-value", type=float, default=240.0)
    p.add_argument("--highlight-area-ratio", type=float, default=0.06)
    p.add_argument("--highlight-num-blobs", type=int, default=5)
    p.add_argument("--hole-ratio", type=float, default=0.15)
    p.add_argument("--block-num", type=int, default=5)
    p.add_argument("--block-min-ratio", type=float, default=0.05)
    p.add_argument("--block-max-ratio", type=float, default=0.15)
    p.add_argument("--edge-frac", type=float, default=0.15)
    p.add_argument("--edge-hole-prob", type=float, default=0.45)
    p.add_argument("--hole-type", default="random", choices=["random", "block", "edge"])
    p.add_argument("--fill-method", default="none",
                   choices=["none", "median", "inpaint_telea", "inpaint_ns"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--copy-mode", default="auto",
                   choices=["auto", "symlink", "hardlink", "copy"])
    p.add_argument("--save-sanity", action="store_true")
    p.add_argument("--sanity-num", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 60)
    print(f"  RGB-D Robustness Dataset Generator")
    print(f"  Corruption: {args.corruption}  Severity: {args.severity}")
    print(f"  Out: {args.out_root}")
    print("=" * 60)
    metadata = build_dataset(args)
    print(f"\nDone! {metadata['n_images']} images -> {metadata['out_root']}")
    print(f"  RGB: {metadata['n_rgb_ok']} ok + {metadata['n_rgb_corrupted']} corrupted")
    print(f"  Depth: {metadata['n_depth_ok']} ok + {metadata['n_depth_corrupted']} corrupted")


if __name__ == "__main__":
    main()
