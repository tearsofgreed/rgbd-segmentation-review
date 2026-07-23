#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Five-fold external robustness evaluation orchestrator.

This script:
- Uses checkpoints 1.pth ... 5.pth under checkpoint_dir.
- Generates corrupted mirror datasets once per selected experiment group.
- Runs each test in a separate tools/test.py subprocess.
- Saves results under results_5fold/fold{n}/{exp_name}/.
- Sleeps and prints GPU status after each test to reduce CUDA memory issues.

Run from the MMDetection repository root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import os.path as osp
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.robustness_external.robustness_utils import (
    ensure_dir,
    build_test_command,
    build_subprocess_env,
    run_command_and_log,
    parse_metrics_from_log,
    save_json,
    load_json,
    compute_drop_retention,
    compute_recovery,
    append_csv_row,
)


# ============================================================
# Experiment definitions
# ============================================================

RGB_BRIGHTNESS = [
    {"group": "rgb_brightness", "exp_name": "rgb_brightness_s1", "corruption": "rgb_brightness", "severity": 1, "name": "s1", "params": {"brightness_factor": 0.90}},
    {"group": "rgb_brightness", "exp_name": "rgb_brightness_s2", "corruption": "rgb_brightness", "severity": 2, "name": "s2", "params": {"brightness_factor": 0.80}},
    {"group": "rgb_brightness", "exp_name": "rgb_brightness_s3", "corruption": "rgb_brightness", "severity": 3, "name": "s3", "params": {"brightness_factor": 0.70}},
    {"group": "rgb_brightness", "exp_name": "rgb_brightness_s4", "corruption": "rgb_brightness", "severity": 4, "name": "s4", "params": {"brightness_factor": 0.60}},
    {"group": "rgb_brightness", "exp_name": "rgb_brightness_s5", "corruption": "rgb_brightness", "severity": 5, "name": "s5", "params": {"brightness_factor": 0.45}},
]

RGB_WHITE_BALANCE = [
    {"group": "rgb_white_balance", "exp_name": "rgb_white_balance_warm_s1", "corruption": "rgb_white_balance", "severity": 1, "name": "warm_s1", "params": {"r_gain": 1.05, "g_gain": 1.00, "b_gain": 0.95}},
    {"group": "rgb_white_balance", "exp_name": "rgb_white_balance_warm_s2", "corruption": "rgb_white_balance", "severity": 2, "name": "warm_s2", "params": {"r_gain": 1.10, "g_gain": 1.00, "b_gain": 0.90}},
    {"group": "rgb_white_balance", "exp_name": "rgb_white_balance_warm_s3", "corruption": "rgb_white_balance", "severity": 3, "name": "warm_s3", "params": {"r_gain": 1.15, "g_gain": 1.00, "b_gain": 0.85}},
    {"group": "rgb_white_balance", "exp_name": "rgb_white_balance_warm_s4", "corruption": "rgb_white_balance", "severity": 4, "name": "warm_s4", "params": {"r_gain": 1.25, "g_gain": 0.95, "b_gain": 0.75}},
    {"group": "rgb_white_balance", "exp_name": "rgb_white_balance_warm_s5", "corruption": "rgb_white_balance", "severity": 5, "name": "warm_s5", "params": {"r_gain": 1.35, "g_gain": 0.90, "b_gain": 0.65}},

    {"group": "rgb_white_balance", "exp_name": "rgb_white_balance_cool_s1", "corruption": "rgb_white_balance", "severity": 1, "name": "cool_s1", "params": {"r_gain": 0.95, "g_gain": 1.00, "b_gain": 1.05}},
    {"group": "rgb_white_balance", "exp_name": "rgb_white_balance_cool_s2", "corruption": "rgb_white_balance", "severity": 2, "name": "cool_s2", "params": {"r_gain": 0.90, "g_gain": 1.00, "b_gain": 1.10}},
    {"group": "rgb_white_balance", "exp_name": "rgb_white_balance_cool_s3", "corruption": "rgb_white_balance", "severity": 3, "name": "cool_s3", "params": {"r_gain": 0.85, "g_gain": 1.00, "b_gain": 1.15}},
    {"group": "rgb_white_balance", "exp_name": "rgb_white_balance_cool_s4", "corruption": "rgb_white_balance", "severity": 4, "name": "cool_s4", "params": {"r_gain": 0.75, "g_gain": 0.95, "b_gain": 1.25}},
    {"group": "rgb_white_balance", "exp_name": "rgb_white_balance_cool_s5", "corruption": "rgb_white_balance", "severity": 5, "name": "cool_s5", "params": {"r_gain": 0.65, "g_gain": 0.90, "b_gain": 1.35}},
]

RGB_LOCAL_SHADOW = [
    {"group": "rgb_local_shadow", "exp_name": "rgb_local_shadow_s1", "corruption": "rgb_local_shadow", "severity": 1, "name": "s1", "params": {"shadow_factor": 0.85, "shadow_area_ratio": 0.05, "shadow_num_blobs": 1}},
    {"group": "rgb_local_shadow", "exp_name": "rgb_local_shadow_s2", "corruption": "rgb_local_shadow", "severity": 2, "name": "s2", "params": {"shadow_factor": 0.75, "shadow_area_ratio": 0.10, "shadow_num_blobs": 2}},
    {"group": "rgb_local_shadow", "exp_name": "rgb_local_shadow_s3", "corruption": "rgb_local_shadow", "severity": 3, "name": "s3", "params": {"shadow_factor": 0.65, "shadow_area_ratio": 0.15, "shadow_num_blobs": 3}},
    {"group": "rgb_local_shadow", "exp_name": "rgb_local_shadow_s4", "corruption": "rgb_local_shadow", "severity": 4, "name": "s4", "params": {"shadow_factor": 0.50, "shadow_area_ratio": 0.20, "shadow_num_blobs": 4}},
    {"group": "rgb_local_shadow", "exp_name": "rgb_local_shadow_s5", "corruption": "rgb_local_shadow", "severity": 5, "name": "s5", "params": {"shadow_factor": 0.35, "shadow_area_ratio": 0.30, "shadow_num_blobs": 5}},
]

RGB_SPECULAR_HIGHLIGHT = [
    {"group": "rgb_specular_highlight", "exp_name": "rgb_specular_highlight_s1", "corruption": "rgb_specular_highlight", "severity": 1, "name": "s1", "params": {"highlight_value": 200, "highlight_area_ratio": 0.015, "highlight_num_blobs": 1}},
    {"group": "rgb_specular_highlight", "exp_name": "rgb_specular_highlight_s2", "corruption": "rgb_specular_highlight", "severity": 2, "name": "s2", "params": {"highlight_value": 220, "highlight_area_ratio": 0.030, "highlight_num_blobs": 3}},
    {"group": "rgb_specular_highlight", "exp_name": "rgb_specular_highlight_s3", "corruption": "rgb_specular_highlight", "severity": 3, "name": "s3", "params": {"highlight_value": 235, "highlight_area_ratio": 0.050, "highlight_num_blobs": 4}},
    {"group": "rgb_specular_highlight", "exp_name": "rgb_specular_highlight_s4", "corruption": "rgb_specular_highlight", "severity": 4, "name": "s4", "params": {"highlight_value": 245, "highlight_area_ratio": 0.075, "highlight_num_blobs": 6}},
    {"group": "rgb_specular_highlight", "exp_name": "rgb_specular_highlight_s5", "corruption": "rgb_specular_highlight", "severity": 5, "name": "s5", "params": {"highlight_value": 255, "highlight_area_ratio": 0.100, "highlight_num_blobs": 8}},
]

DEPTH_HOLES = [
    {"group": "depth_holes", "exp_name": "depth_random_holes_s1", "corruption": "depth_random_holes", "severity": 1, "name": "s1", "params": {"hole_ratio": 0.03}},
    {"group": "depth_holes", "exp_name": "depth_random_holes_s2", "corruption": "depth_random_holes", "severity": 2, "name": "s2", "params": {"hole_ratio": 0.05}},
    {"group": "depth_holes", "exp_name": "depth_random_holes_s3", "corruption": "depth_random_holes", "severity": 3, "name": "s3", "params": {"hole_ratio": 0.10}},
    {"group": "depth_holes", "exp_name": "depth_random_holes_s4", "corruption": "depth_random_holes", "severity": 4, "name": "s4", "params": {"hole_ratio": 0.20}},
    {"group": "depth_holes", "exp_name": "depth_random_holes_s5", "corruption": "depth_random_holes", "severity": 5, "name": "s5", "params": {"hole_ratio": 0.30}},

    {"group": "depth_holes", "exp_name": "depth_block_holes_s1", "corruption": "depth_block_holes", "severity": 1, "name": "s1", "params": {"block_num": 2, "block_min_ratio": 0.03, "block_max_ratio": 0.06}},
    {"group": "depth_holes", "exp_name": "depth_block_holes_s2", "corruption": "depth_block_holes", "severity": 2, "name": "s2", "params": {"block_num": 3, "block_min_ratio": 0.05, "block_max_ratio": 0.10}},
    {"group": "depth_holes", "exp_name": "depth_block_holes_s3", "corruption": "depth_block_holes", "severity": 3, "name": "s3", "params": {"block_num": 5, "block_min_ratio": 0.08, "block_max_ratio": 0.13}},
    {"group": "depth_holes", "exp_name": "depth_block_holes_s4", "corruption": "depth_block_holes", "severity": 4, "name": "s4", "params": {"block_num": 7, "block_min_ratio": 0.10, "block_max_ratio": 0.17}},
    {"group": "depth_holes", "exp_name": "depth_block_holes_s5", "corruption": "depth_block_holes", "severity": 5, "name": "s5", "params": {"block_num": 9, "block_min_ratio": 0.12, "block_max_ratio": 0.22}},

    {"group": "depth_holes", "exp_name": "depth_edge_holes_s1", "corruption": "depth_edge_holes", "severity": 1, "name": "s1", "params": {"edge_frac": 0.05, "edge_hole_prob": 0.15}},
    {"group": "depth_holes", "exp_name": "depth_edge_holes_s2", "corruption": "depth_edge_holes", "severity": 2, "name": "s2", "params": {"edge_frac": 0.10, "edge_hole_prob": 0.25}},
    {"group": "depth_holes", "exp_name": "depth_edge_holes_s3", "corruption": "depth_edge_holes", "severity": 3, "name": "s3", "params": {"edge_frac": 0.15, "edge_hole_prob": 0.40}},
    {"group": "depth_holes", "exp_name": "depth_edge_holes_s4", "corruption": "depth_edge_holes", "severity": 4, "name": "s4", "params": {"edge_frac": 0.20, "edge_hole_prob": 0.55}},
    {"group": "depth_holes", "exp_name": "depth_edge_holes_s5", "corruption": "depth_edge_holes", "severity": 5, "name": "s5", "params": {"edge_frac": 0.25, "edge_hole_prob": 0.70}},
]


def build_depth_fill() -> List[Dict[str, Any]]:
    exps: List[Dict[str, Any]] = []
    hole_params = {
        "block": {
            1: {"block_num": 2, "block_min_ratio": 0.03, "block_max_ratio": 0.06},
            2: {"block_num": 3, "block_min_ratio": 0.05, "block_max_ratio": 0.10},
            3: {"block_num": 5, "block_min_ratio": 0.08, "block_max_ratio": 0.13},
            4: {"block_num": 7, "block_min_ratio": 0.10, "block_max_ratio": 0.17},
            5: {"block_num": 9, "block_min_ratio": 0.12, "block_max_ratio": 0.22},
        },
        "edge": {
            1: {"edge_frac": 0.05, "edge_hole_prob": 0.15},
            2: {"edge_frac": 0.10, "edge_hole_prob": 0.25},
            3: {"edge_frac": 0.15, "edge_hole_prob": 0.40},
            4: {"edge_frac": 0.20, "edge_hole_prob": 0.55},
            5: {"edge_frac": 0.25, "edge_hole_prob": 0.70},
        },
    }
    for hole_type in ["block", "edge"]:
        for severity in [1, 2, 3, 4, 5]:
            for fill_method in ["none", "median", "inpaint_telea"]:
                params = dict(hole_params[hole_type][severity])
                params["hole_type"] = hole_type
                params["fill_method"] = fill_method
                exps.append({
                    "group": "depth_fill",
                    "exp_name": f"depth_fill_{hole_type}_s{severity}_{fill_method}",
                    "corruption": "depth_holes_with_fill",
                    "severity": severity,
                    "name": f"s{severity}_{fill_method}",
                    "hole_type": hole_type,
                    "fill_method": fill_method,
                    "params": params,
                })
    return exps


DEPTH_FILL = build_depth_fill()

GROUPS = {
    "rgb_brightness": RGB_BRIGHTNESS,
    "rgb_white_balance": RGB_WHITE_BALANCE,
    "rgb_local_shadow": RGB_LOCAL_SHADOW,
    "rgb_specular_highlight": RGB_SPECULAR_HIGHLIGHT,
    "depth_holes": DEPTH_HOLES,
    "depth_fill": DEPTH_FILL,
    "rgb": RGB_BRIGHTNESS + RGB_WHITE_BALANCE + RGB_LOCAL_SHADOW + RGB_SPECULAR_HIGHLIGHT,
    "all": RGB_BRIGHTNESS + RGB_WHITE_BALANCE + RGB_LOCAL_SHADOW + RGB_SPECULAR_HIGHLIGHT + DEPTH_HOLES + DEPTH_FILL,
}

CSV_FIELDS = [
    "fold", "checkpoint", "exp_name", "group", "corruption", "severity", "severity_name",
    "fill_method", "hole_type", "data_root", "ann_file", "log_file",
    "segm_mAP", "segm_mAP_50", "segm_mAP_75",
    "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
    "clean_segm_mAP", "AP_drop", "AP_retention",
    "AP50_drop", "AP50_retention", "AP75_drop", "AP75_retention",
    "recovery_AP", "recovery_AP50", "recovery_AP75",
    "status", "error",
]


# ============================================================
# Helpers
# ============================================================

def to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("none", "nan", "?"):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def has_core_metrics(metrics: Dict[str, Any]) -> bool:
    return to_float_or_none(metrics.get("coco/segm_mAP")) is not None


def fmt4(x: Any) -> str:
    v = to_float_or_none(x)
    return f"{v:.4f}" if v is not None else ""


def load_metrics_if_valid(path: str) -> Optional[Dict[str, Any]]:
    if not osp.isfile(path):
        return None
    try:
        metrics = load_json(path)
    except Exception:
        return None
    if has_core_metrics(metrics):
        return metrics
    return None


def print_gpu_status(title: str = "") -> None:
    print("\n" + "=" * 80)
    print(f"GPU STATUS {title}".strip())
    print("=" * 80)
    try:
        subprocess.run(["nvidia-smi"], check=False)
    except Exception as e:
        print(f"[WARN] nvidia-smi failed: {e}")


def kill_stale_gpu_python() -> None:
    """Kill stale python processes still registered as GPU compute apps.

    The orchestrator itself normally does not use GPU, so it should not appear
    in nvidia-smi compute apps. This is optional and should be used with care.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        current_pid = os.getpid()
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            parts = [x.strip() for x in line.split(",", 1)]
            if len(parts) < 2:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            pname = parts[1].lower()
            if pid != current_pid and "python" in pname:
                print(f"[GPU CLEAN] Killing stale GPU python PID={pid}, process={parts[1]}")
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
    except Exception as e:
        print(f"[WARN] kill_stale_gpu_python failed: {e}")


def sleep_and_clean(args: argparse.Namespace, after_what: str) -> None:
    if args.kill_stale_gpu_python:
        kill_stale_gpu_python()
    if args.show_gpu:
        print_gpu_status(f"after {after_what}")
    if args.sleep_after_test > 0:
        print(f"[SLEEP] {args.sleep_after_test}s after {after_what}")
        time.sleep(args.sleep_after_test)


def write_summary(result: Dict[str, Any], path: str) -> None:
    ensure_dir(osp.dirname(path))
    fields = [
        "fold", "checkpoint", "exp_name", "group", "corruption", "severity",
        "severity_name", "fill_method", "hole_type", "data_root", "ann_file",
        "log_file", "segm_mAP", "segm_mAP_50", "segm_mAP_75",
        "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
        "clean_segm_mAP", "AP_drop", "AP_retention",
        "AP50_drop", "AP50_retention", "AP75_drop", "AP75_retention",
        "recovery_AP", "recovery_AP50", "recovery_AP75",
        "status", "error",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for k in fields:
            v = result.get(k)
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {'' if v is None else v}\n")


def selected_experiments(groups: List[str]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for g in groups:
        if g not in GROUPS:
            raise ValueError(f"Unknown group: {g}")
        for e in GROUPS[g]:
            if e["exp_name"] not in seen:
                out.append(e)
                seen.add(e["exp_name"])
    return out


def clean_experiment() -> Dict[str, Any]:
    return {
        "group": "clean",
        "exp_name": "clean_mirror",
        "corruption": "clean",
        "severity": 0,
        "name": "clean",
        "params": {},
    }


def dataset_valid(dataset_root: str, exp_name: str) -> bool:
    root = osp.join(dataset_root, exp_name)
    return (
        osp.isdir(osp.join(root, "images"))
        and osp.isdir(osp.join(root, "depth"))
        and osp.isfile(osp.join(root, "annotations", "val.json"))
    )


def generate_dataset(exp: Dict[str, Any], args: argparse.Namespace) -> int:
    exp_name = exp["exp_name"]
    out_root = osp.join(args.datasets_root, exp_name)

    if dataset_valid(args.datasets_root, exp_name) and not args.regenerate_datasets:
        print(f"[DATASET SKIP] {exp_name}")
        return 0

    cmd = [
        args.python_exe,
        "tools/robustness_external/make_robustness_dataset.py",
        "--data-root", args.data_root,
        "--ann-file", args.ann_file,
        "--out-root", out_root,
        "--corruption", exp["corruption"],
        "--severity", str(exp["severity"]),
        "--copy-mode", args.copy_mode,
        "--overwrite",
    ]

    for k, v in exp.get("params", {}).items():
        cmd.extend(["--" + k.replace("_", "-"), str(v)])

    if args.save_sanity:
        cmd.extend(["--save-sanity", "--sanity-num", str(args.sanity_num)])

    print("\n" + "=" * 80)
    print(f"[DATASET] {exp_name}")
    print(" ".join(cmd))
    print("=" * 80)

    if args.dry_run:
        return 0

    proc = subprocess.run(cmd, cwd=_PROJECT_ROOT, check=False)
    return proc.returncode


def prepare_datasets(exps: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    all_datasets = [clean_experiment()] + exps
    for exp in all_datasets:
        rc = generate_dataset(exp, args)
        if rc != 0:
            raise RuntimeError(f"Dataset generation failed: {exp['exp_name']} rc={rc}")


def checkpoint_path(args: argparse.Namespace, fold: int) -> str:
    return osp.join(args.checkpoint_dir, f"{fold}.pth")


def load_clean_metrics(fold_results_root: str) -> Dict[str, Any]:
    p = osp.join(fold_results_root, "clean_mirror", "metrics.json")
    metrics = load_metrics_if_valid(p)
    return metrics or {}


def run_test_for_exp(
    fold: int,
    ckpt: str,
    exp: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    exp_name = exp["exp_name"]
    fold_results_root = osp.join(args.results_root, f"fold{fold}")
    result_dir = ensure_dir(osp.join(fold_results_root, exp_name))
    log_file = osp.join(result_dir, "test.log")
    metrics_file = osp.join(result_dir, "metrics.json")
    summary_file = osp.join(result_dir, "summary.txt")

    if args.skip_existing_tests:
        old_metrics = load_metrics_if_valid(metrics_file)
        if old_metrics is not None:
            print(f"[TEST SKIP] fold{fold} {exp_name}: valid metrics already exist")
            return {
                "fold": fold,
                "checkpoint": ckpt,
                "exp_name": exp_name,
                "status": "skipped",
                "segm_mAP": old_metrics.get("coco/segm_mAP"),
            }

    data_root = osp.join(args.datasets_root, exp_name)
    ann_file = osp.join(data_root, "annotations", "val.json")
    work_dir = result_dir

    cmd = build_test_command(
        args.python_exe,
        args.config,
        ckpt,
        data_root,
        ann_file,
        work_dir,
    )
    env = build_subprocess_env(_PROJECT_ROOT)
    env["PYTORCH_CUDA_ALLOC_CONF"] = args.cuda_alloc_conf

    print("\n" + "=" * 80)
    print(f"[TEST] fold{fold} | {exp_name}")
    print(f"checkpoint: {ckpt}")
    print(f"result_dir: {result_dir}")
    print("=" * 80)

    if args.dry_run:
        return {
            "fold": fold,
            "checkpoint": ckpt,
            "exp_name": exp_name,
            "status": "dry_run",
        }

    rc = run_command_and_log(cmd, log_file, _PROJECT_ROOT, env)
    metrics = parse_metrics_from_log(log_file)
    save_json(metrics, metrics_file)

    clean_metrics = load_clean_metrics(fold_results_root)
    result: Dict[str, Any] = {
        "fold": fold,
        "checkpoint": ckpt,
        "exp_name": exp_name,
        "group": exp.get("group"),
        "corruption": exp.get("corruption"),
        "severity": exp.get("severity"),
        "severity_name": exp.get("name"),
        "fill_method": exp.get("fill_method", "none"),
        "hole_type": exp.get("hole_type", ""),
        "data_root": data_root,
        "ann_file": ann_file,
        "log_file": log_file,
        "segm_mAP": metrics.get("coco/segm_mAP"),
        "segm_mAP_50": metrics.get("coco/segm_mAP_50"),
        "segm_mAP_75": metrics.get("coco/segm_mAP_75"),
        "bbox_mAP": metrics.get("coco/bbox_mAP"),
        "bbox_mAP_50": metrics.get("coco/bbox_mAP_50"),
        "bbox_mAP_75": metrics.get("coco/bbox_mAP_75"),
        "clean_segm_mAP": clean_metrics.get("coco/segm_mAP"),
        "status": "success" if has_core_metrics(metrics) else "failed",
        "error": None,
    }

    if rc != 0 and has_core_metrics(metrics):
        result["status"] = "success"
        result["error"] = f"test rc={rc}, but metrics parsed"
    elif rc != 0:
        result["status"] = "failed"
        result["error"] = f"test rc={rc}"

    if exp_name != "clean_mirror" and clean_metrics:
        aliases = [
            ("coco/segm_mAP", "AP"),
            ("coco/segm_mAP_50", "AP50"),
            ("coco/segm_mAP_75", "AP75"),
        ]
        for mk, alias in aliases:
            drop, ret = compute_drop_retention(clean_metrics, metrics, mk)
            result[f"{alias}_drop"] = drop
            result[f"{alias}_retention"] = ret

    if exp.get("group") == "depth_fill" and exp.get("fill_method") != "none":
        hole_name = f"depth_fill_{exp.get('hole_type')}_s{exp.get('severity')}_none"
        hole_metrics_path = osp.join(fold_results_root, hole_name, "metrics.json")
        hole_metrics = load_metrics_if_valid(hole_metrics_path) or {}
        aliases = [
            ("coco/segm_mAP", "AP"),
            ("coco/segm_mAP_50", "AP50"),
            ("coco/segm_mAP_75", "AP75"),
        ]
        for mk, alias in aliases:
            result[f"recovery_{alias}"] = compute_recovery(clean_metrics, hole_metrics, metrics, mk)

    write_summary(result, summary_file)

    fold_csv = osp.join(fold_results_root, "robustness_results.csv")
    all_csv = osp.join(args.results_root, "all_results.csv")
    append_csv_row(fold_csv, result, CSV_FIELDS)
    append_csv_row(all_csv, result, CSV_FIELDS)

    print(f"[DONE] fold{fold} {exp_name} status={result['status']} AP={result.get('segm_mAP')}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Five-fold robustness orchestrator")
    p.add_argument("--config", default="configs/mask2former/config_fusion_all_stage_strict.py")
    p.add_argument("--checkpoint-dir", default="work_dirs/test_pth")
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--groups", nargs="+", default=["rgb_brightness"],
                   choices=list(GROUPS.keys()))
    p.add_argument("--data-root", default="data/seed1")
    p.add_argument("--ann-file", default="annotations/val.json")
    p.add_argument("--datasets-root", default="robustness_external/datasets")
    p.add_argument("--results-root", default="robustness_external/results_5fold")
    p.add_argument("--copy-mode", default="copy")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--cuda-alloc-conf", default="max_split_size_mb:128")
    p.add_argument("--sleep-after-test", type=int, default=25)
    p.add_argument("--sleep-after-fold", type=int, default=60)
    p.add_argument("--regenerate-datasets", action="store_true")
    p.add_argument("--skip-existing-tests", action="store_true")
    p.add_argument("--save-sanity", action="store_true")
    p.add_argument("--sanity-num", type=int, default=5)
    p.add_argument("--show-gpu", action="store_true")
    p.add_argument("--kill-stale-gpu-python", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--max-experiments", type=int, default=None,
                   help="Only run first N selected experiments, useful for smoke tests.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.results_root)

    exps = selected_experiments(args.groups)
    if args.max_experiments is not None:
        exps = exps[:args.max_experiments]

    print("=" * 80)
    print("Five-fold robustness orchestrator")
    print(f"project_root     : {_PROJECT_ROOT}")
    print(f"checkpoint_dir  : {args.checkpoint_dir}")
    print(f"folds           : {args.folds}")
    print(f"groups          : {args.groups}")
    print(f"experiments     : {len(exps)}")
    print(f"datasets_root   : {args.datasets_root}")
    print(f"results_root    : {args.results_root}")
    print(f"sleep_after_test: {args.sleep_after_test}s")
    print("=" * 80)
    for e in exps:
        print(f"  - {e['exp_name']}")

    print_gpu_status("before start")

    prepare_datasets(exps, args)

    for fold in args.folds:
        ckpt = checkpoint_path(args, fold)
        if not osp.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found for fold{fold}: {ckpt}")

        fold_results_root = ensure_dir(osp.join(args.results_root, f"fold{fold}"))
        print("\n" + "#" * 80)
        print(f"# FOLD {fold}")
        print(f"# checkpoint: {ckpt}")
        print(f"# results   : {fold_results_root}")
        print("#" * 80)

        try:
            # Clean baseline first, per checkpoint.
            run_test_for_exp(fold, ckpt, clean_experiment(), args)
            sleep_and_clean(args, f"fold{fold} clean_mirror")

            # Selected robustness experiments.
            for exp in exps:
                run_test_for_exp(fold, ckpt, exp, args)
                sleep_and_clean(args, f"fold{fold} {exp['exp_name']}")

        except Exception as e:
            print(f"[ERROR] fold{fold}: {e}")
            if args.stop_on_error:
                raise

        if args.sleep_after_fold > 0:
            print(f"[SLEEP] {args.sleep_after_fold}s after fold{fold}")
            time.sleep(args.sleep_after_fold)

    print("\nALL DONE.")
    print(f"Global CSV: {osp.join(args.results_root, 'all_results.csv')}")
    print_gpu_status("after all")


if __name__ == "__main__":
    main()