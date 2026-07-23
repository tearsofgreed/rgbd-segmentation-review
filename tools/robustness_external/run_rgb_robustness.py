#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Automated RGB robustness experiments (external, zero-invasive)."""

from __future__ import annotations

import argparse
import os
import os.path as osp
import subprocess
import sys
from pathlib import Path

# Add project root so we can import sibling modules
_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.robustness_external.robustness_utils import (
    ensure_dir, get_project_root, build_test_command, build_subprocess_env,
    run_command_and_log, parse_metrics_from_log, save_json, load_json,
    append_csv_row, load_clean_baseline, compute_drop_retention,
    has_core_metrics, write_experiment_summary,
)

# ============================================================================
# Experiment matrix
# ============================================================================

RGB_EXPERIMENTS = {
    "rgb_brightness": [
        {"severity": 1, "name": "s1", "brightness_factor": 0.90},
        {"severity": 2, "name": "s2", "brightness_factor": 0.80},
        {"severity": 3, "name": "s3", "brightness_factor": 0.70},
        {"severity": 4, "name": "s4", "brightness_factor": 0.60},
        {"severity": 5, "name": "s5", "brightness_factor": 0.45},
    ],
    "rgb_white_balance": [
        {"severity": 1, "name": "warm_s1", "r_gain": 1.05, "g_gain": 1.00, "b_gain": 0.95},
        {"severity": 2, "name": "warm_s2", "r_gain": 1.10, "g_gain": 1.00, "b_gain": 0.90},
        {"severity": 3, "name": "warm_s3", "r_gain": 1.15, "g_gain": 1.00, "b_gain": 0.85},
        {"severity": 4, "name": "warm_s4", "r_gain": 1.25, "g_gain": 0.95, "b_gain": 0.75},
        {"severity": 5, "name": "warm_s5", "r_gain": 1.35, "g_gain": 0.90, "b_gain": 0.65},
        {"severity": 1, "name": "cool_s1", "r_gain": 0.95, "g_gain": 1.00, "b_gain": 1.05},
        {"severity": 2, "name": "cool_s2", "r_gain": 0.90, "g_gain": 1.00, "b_gain": 1.10},
        {"severity": 3, "name": "cool_s3", "r_gain": 0.85, "g_gain": 1.00, "b_gain": 1.15},
        {"severity": 4, "name": "cool_s4", "r_gain": 0.75, "g_gain": 0.95, "b_gain": 1.25},
        {"severity": 5, "name": "cool_s5", "r_gain": 0.65, "g_gain": 0.90, "b_gain": 1.35},
    ],
    "rgb_local_shadow": [
        {"severity": 1, "name": "s1", "shadow_factor": 0.85, "shadow_area_ratio": 0.05, "shadow_num_blobs": 1},
        {"severity": 2, "name": "s2", "shadow_factor": 0.75, "shadow_area_ratio": 0.10, "shadow_num_blobs": 2},
        {"severity": 3, "name": "s3", "shadow_factor": 0.65, "shadow_area_ratio": 0.15, "shadow_num_blobs": 3},
        {"severity": 4, "name": "s4", "shadow_factor": 0.50, "shadow_area_ratio": 0.20, "shadow_num_blobs": 4},
        {"severity": 5, "name": "s5", "shadow_factor": 0.35, "shadow_area_ratio": 0.30, "shadow_num_blobs": 5},
    ],
    "rgb_specular_highlight": [
        {"severity": 1, "name": "s1", "highlight_value": 200, "highlight_area_ratio": 0.015, "highlight_num_blobs": 1},
        {"severity": 2, "name": "s2", "highlight_value": 220, "highlight_area_ratio": 0.030, "highlight_num_blobs": 3},
        {"severity": 3, "name": "s3", "highlight_value": 235, "highlight_area_ratio": 0.050, "highlight_num_blobs": 4},
        {"severity": 4, "name": "s4", "highlight_value": 245, "highlight_area_ratio": 0.075, "highlight_num_blobs": 6},
        {"severity": 5, "name": "s5", "highlight_value": 255, "highlight_area_ratio": 0.100, "highlight_num_blobs": 8},
    ],
}

CSV_FIELDS = [
    "exp_name", "group", "corruption", "severity", "severity_name",
    "fill_method", "data_root", "log_file",
    "segm_mAP", "segm_mAP_50", "segm_mAP_75",
    "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
    "clean_segm_mAP", "AP_drop", "AP_retention",
    "AP50_drop", "AP50_retention", "AP75_drop", "AP75_retention",
    "recovery_AP", "status", "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RGB Robustness Runner")
    p.add_argument("--config", default="configs/mask2former/config_fusion_all_stage_strict.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data/seed1")
    p.add_argument("--ann-file", default="annotations/val.json")
    p.add_argument("--out-root", default="robustness_external/datasets")
    p.add_argument("--results-root", default="robustness_external/results")
    p.add_argument("--copy-mode", default="copy")
    p.add_argument("--corruptions", nargs="*", default=None)
    p.add_argument("--severities", type=int, nargs="*", default=None)
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--save-sanity", action="store_true")
    p.add_argument("--sanity-num", type=int, default=5)
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--stop-on-error", action="store_true")
    return p.parse_args()


def build_experiments(args: argparse.Namespace) -> list:
    exps = []
    for corr, variants in RGB_EXPERIMENTS.items():
        if args.corruptions and corr not in args.corruptions:
            continue
        for v in variants:
            if args.severities and v["severity"] not in args.severities:
                continue
            v["corruption"] = corr
            exps.append(v)
    return exps


def run_one_experiment(exp: dict, args: argparse.Namespace, project_root: str) -> dict:
    corr = exp["corruption"]
    sev = exp["severity"]
    name = exp["name"]
    exp_name = f"{corr}_{name}"

    data_root_path = osp.join(args.out_root, exp_name)
    results_dir = ensure_dir(osp.join(args.results_root, exp_name))
    log_file = osp.join(results_dir, "test.log")
    metrics_file = osp.join(results_dir, "metrics.json")

    if args.skip_existing and osp.isfile(metrics_file):
        existing = load_json(metrics_file)
        if has_core_metrics(existing):
            print(f"  [SKIP] {exp_name} already has valid metrics")
            return {"status": "skipped", "exp_name": exp_name}
        else:
            print(f"  [RE-RUN] {exp_name} metrics.json exists but empty")

    # 1) Generate dataset
    if not args.skip_generate:
        gen_cmd = [
            sys.executable, "tools/robustness_external/make_robustness_dataset.py",
            "--data-root", args.data_root, "--ann-file", args.ann_file,
            "--out-root", data_root_path, "--corruption", corr,
            "--severity", str(sev), "--copy-mode", args.copy_mode,
            "--overwrite",
        ]
        for k in ["brightness_factor", "r_gain", "g_gain", "b_gain",
                   "shadow_factor", "shadow_area_ratio", "shadow_num_blobs",
                   "highlight_value", "highlight_area_ratio", "highlight_num_blobs"]:
            if k in exp:
                gen_cmd.extend(["--" + k.replace("_", "-"), str(exp[k])])
        if args.save_sanity:
            gen_cmd.extend(["--save-sanity", "--sanity-num", str(args.sanity_num)])
        print(f"  Generating: {exp_name}")
        rc = subprocess.run(gen_cmd, cwd=project_root, capture_output=False)
        if rc.returncode != 0:
            return {"exp_name": exp_name, "status": "failed", "error": f"gen rc={rc.returncode}"}

    # 2) Test
    ann = osp.join(data_root_path, "annotations", "val.json")
    test_cmd = build_test_command(args.python_exe, args.config, args.checkpoint,
                                   data_root_path, ann, results_dir)
    env = build_subprocess_env(project_root)
    print(f"  Testing: {exp_name}")
    rc = run_command_and_log(test_cmd, log_file, project_root, env)

    # 3) Parse metrics
    metrics = parse_metrics_from_log(log_file)
    save_json(metrics, metrics_file)

    clean = load_clean_baseline(args.results_root)
    result = {
        "exp_name": exp_name, "group": "rgb", "corruption": corr,
        "severity": sev, "severity_name": name, "fill_method": "none",
        "data_root": data_root_path, "log_file": log_file,
        "segm_mAP": metrics.get("coco/segm_mAP"),
        "segm_mAP_50": metrics.get("coco/segm_mAP_50"),
        "segm_mAP_75": metrics.get("coco/segm_mAP_75"),
        "bbox_mAP": metrics.get("coco/bbox_mAP"),
        "bbox_mAP_50": metrics.get("coco/bbox_mAP_50"),
        "bbox_mAP_75": metrics.get("coco/bbox_mAP_75"),
        "clean_segm_mAP": clean.get("coco/segm_mAP"),
        "status": "success" if rc == 0 else "failed",
        "error": f"test rc={rc}" if rc != 0 else None,
    }

    if clean:
        metric_aliases = [
            ("coco/segm_mAP", "AP"),
            ("coco/segm_mAP_50", "AP50"),
            ("coco/segm_mAP_75", "AP75"),
        ]
        for mk, alias in metric_aliases:
            drop, ret = compute_drop_retention(clean, metrics, mk)
            result[f"{alias}_drop"] = drop
            result[f"{alias}_retention"] = ret

    if result["status"] == "failed" and has_core_metrics(metrics):
        result["status"] = "success"
        result["error"] = f"test rc={rc}, but metrics parsed"

    result["checkpoint"] = args.checkpoint
    result["ann_file"] = ann
    write_experiment_summary(result, osp.join(results_dir, "summary.txt"))
    return result


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    exps = build_experiments(args)

    print(f"Experiments: {len(exps)}")
    for e in exps:
        print(f"  {e['corruption']}_{e['name']}")

    if args.dry_run:
        print("[DRY RUN] Done.")
        return

    csv_path = osp.join(args.results_root, "robustness_rgb_results.csv")
    for exp in exps:
        try:
            result = run_one_experiment(exp, args, project_root)
            append_csv_row(csv_path, result, CSV_FIELDS)
            print(f"  [{result['status']}] {result['exp_name']} "
                  f"AP={result.get('segm_mAP','?')}")
        except Exception as e:
            print(f"  [FAIL] {exp['corruption']}_{exp['name']}: {e}")
            if args.stop_on_error:
                raise

    print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    main()
