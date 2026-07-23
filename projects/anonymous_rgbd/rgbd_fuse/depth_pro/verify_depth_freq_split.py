import os
import argparse
from pathlib import Path

import cv2
import numpy as np

from rgbd_fuse.depth_pro.depth_preprocess import DepthPreprocess


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def to_uint8_vis(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    xmin = float(x.min())
    xmax = float(x.max())
    if xmax <= xmin:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - xmin) / (xmax - xmin + 1e-6)
    x = (x * 255.0).clip(0, 255).astype(np.uint8)
    return x


def save_vis(img: np.ndarray, out_path: Path):
    vis = to_uint8_vis(img)
    cv2.imwrite(str(out_path), vis)


def infer_aux_path_from_img_path(img_path: Path,
                                 aux_folder='depth',
                                 aux_suffix='.png') -> Path:
    img_dir = img_path.parent
    data_root = img_dir.parent
    stem = img_path.stem
    return data_root / aux_folder / f'{stem}{aux_suffix}'


def load_depth(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f'Failed to read depth image: {depth_path}')
    return depth.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True,
                        help='depth 文件夹路径，例如 data/seed1/depth')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='输出可视化目录')
    parser.add_argument('--num-samples', type=int, default=5,
                        help='保存多少张样本')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    repaired_dir = output_dir / 'repaired_depth'
    low_dir = output_dir / 'low_freq'
    high_dir = output_dir / 'high_freq'

    ensure_dir(repaired_dir)
    ensure_dir(low_dir)
    ensure_dir(high_dir)

    depth_files = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    depth_files = depth_files[:args.num_samples]

    preprocess = DepthPreprocess(
        key='aux_img',
        to_float32=True,
        squeeze_if_single_channel=True,
        to_gray=False,
        invalid_min=2.0,
        invalid_fill_value=0.0,
        median_ksize=3,
        percentile_clip=(2.0, 98.0),
        normalize_mode='minmax_on_valid',
        apply_log1p=False,
        enable_freq_split=True,
        low_key='aux_img_low',
        high_key='aux_img_high',
        low_mode='gaussian',
        gaussian_ksize=5,
        gaussian_sigma=0.0,
        high_mode='residual_abs',
        high_percentile_clip=(1.0, 99.0),
        normalize_low=True,
        normalize_high=True
    )

    for depth_path in depth_files:
        depth = load_depth(depth_path)

        results = {
            'aux_img': depth,
            'aux_img_path': str(depth_path),
        }

        results = preprocess.transform(results)

        repaired = results['aux_img']
        low = results['aux_img_low']
        high = results['aux_img_high']

        stem = depth_path.stem
        save_vis(repaired, repaired_dir / f'{stem}.png')
        save_vis(low, low_dir / f'{stem}.png')
        save_vis(high, high_dir / f'{stem}.png')

        print(f'[OK] {depth_path.name}')
        print(f'     saved repaired -> {repaired_dir / f"{stem}.png"}')
        print(f'     saved low      -> {low_dir / f"{stem}.png"}')
        print(f'     saved high     -> {high_dir / f"{stem}.png"}')

    print('[DONE] verify_depth_freq_split finished.')


if __name__ == '__main__':
    main()
