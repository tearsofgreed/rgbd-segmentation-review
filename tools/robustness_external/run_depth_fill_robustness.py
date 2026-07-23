#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Automated Depth Holes+Filling robustness experiments (external)."""

from __future__ import annotations

import argparse, os, os.path as osp, subprocess, sys

_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.robustness_external.robustness_utils import (
    ensure_dir, get_project_root, build_test_command, build_subprocess_env,
    run_command_and_log, parse_metrics_from_log, save_json, append_csv_row,
    load_clean_baseline, compute_drop_retention, compute_recovery,
)

HOLE_PARAMS = {
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

FILL_METHODS = ["none", "median", "inpaint_telea"]

CSV_FIELDS = [
    "exp_name", "group", "corruption", "severity", "severity_name",
    "fill_method", "hole_type", "data_root", "log_file",
    "segm_mAP", "segm_mAP_50", "segm_mAP_75",
    "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
    "clean_segm_mAP", "AP_drop", "AP_retention",
    "AP50_drop", "AP50_retention", "AP75_drop", "AP75_retention",
    "recovery_AP", "recovery_AP50", "recovery_AP75",
    "status", "error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Depth Fill Robustness Runner")
    p.add_argument("--config", default="configs/mask2former/config_fusion_all_stage_strict.py")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data/seed1")
    p.add_argument("--ann-file", default="annotations/val.json")
    p.add_argument("--out-root", default="robustness_external/datasets")
    p.add_argument("--results-root", default="robustness_external/results")
    p.add_argument("--copy-mode", default="copy")
    p.add_argument("--hole-types", nargs="*", default=None, choices=["block", "edge"])
    p.add_argument("--severities", type=int, nargs="*", default=None)
    p.add_argument("--fill-methods", nargs="*", default=None)
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
    hole_types = args.hole_types or ["block", "edge"]
    severities = args.severities or [1, 2, 3, 4, 5]
    fill_methods = args.fill_methods or FILL_METHODS
    for ht in hole_types:
        for sev in severities:
            if sev not in HOLE_PARAMS.get(ht, {}):
                continue
            params = HOLE_PARAMS[ht][sev]
            for fm in fill_methods:
                exp = {
                    "hole_type": ht, "severity": sev, "fill_method": fm,
                    "corruption": f"depth_fill_{ht}",
                    "name": f"s{sev}_{fm}", "severity_name": f"s{sev}",
                }
                exp.update(params)
                exps.append(exp)
    return exps


def run_one_experiment(exp: dict, args: argparse.Namespace, project_root: str) -> dict:
    ht = exp["hole_type"]; sev = exp["severity"]; fm = exp["fill_method"]
    exp_name = f"depth_fill_{ht}_s{sev}_{fm}"
    data_root_path = osp.join(args.out_root, exp_name)
    results_dir = ensure_dir(osp.join(args.results_root, exp_name))
    log_file = osp.join(results_dir, "test.log")
    metrics_file = osp.join(results_dir, "metrics.json")

    if args.skip_existing and osp.isfile(metrics_file):
        print(f"  [SKIP] {exp_name}")
        return {"status": "skipped", "exp_name": exp_name}

    if not args.skip_generate:
        gen_cmd = [
            sys.executable, "tools/robustness_external/make_robustness_dataset.py",
            "--data-root", args.data_root, "--ann-file", args.ann_file,
            "--out-root", data_root_path,
            "--corruption", "depth_holes_with_fill",
            "--hole-type", ht, "--fill-method", fm,
            "--severity", str(sev),
            "--copy-mode", args.copy_mode, "--overwrite",
        ]
        for k in ["block_num", "block_min_ratio", "block_max_ratio",
                   "edge_frac", "edge_hole_prob"]:
            if k in exp:
                gen_cmd.extend(["--" + k.replace("_", "-"), str(exp[k])])
        if args.save_sanity:
            gen_cmd.extend(["--save-sanity", "--sanity-num", str(args.sanity_num)])
        print(f"  Generating: {exp_name}")
        rc = subprocess.run(gen_cmd, cwd=project_root, capture_output=False)
        if rc.returncode != 0:
            return {"exp_name": exp_name, "status": "failed", "error": f"gen rc={rc.returncode}"}

    ann = osp.join(data_root_path, "annotations", "val.json")
    test_cmd = build_test_command(args.python_exe, args.config, args.checkpoint,
                                   data_root_path, ann, results_dir)
    env = build_subprocess_env(project_root)
    print(f"  Testing: {exp_name}")
    rc = run_command_and_log(test_cmd, log_file, project_root, env)

    metrics = parse_metrics_from_log(log_file)
    save_json(metrics, metrics_file)
    clean = load_clean_baseline(args.results_root)

    result = {
        "exp_name": exp_name, "group": "depth_fill", "corruption": f"depth_fill_{ht}",
        "severity": sev, "severity_name": f"s{sev}", "fill_method": fm,
        "hole_type": ht, "data_root": data_root_path, "log_file": log_file,
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
        for mk in ["coco/segm_mAP", "coco/segm_mAP_50", "coco/segm_mAP_75"]:
            drop, ret = compute_drop_retention(clean, metrics, mk)
            short = mk.split("/")[-1]
            result[f"{short}_drop"] = drop
            result[f"{short}_retention"] = ret
    return result


def compute_all_recoveries(results: list) -> list:
    """Post-hoc: compute Recovery for fill experiments."""
    for r in results:
        if r.get("fill_method") == "none" or r["status"] != "success":
            continue
        hole = next((x for x in results
                     if x["hole_type"] == r["hole_type"]
                     and x["severity"] == r["severity"]
                     and x["fill_method"] == "none"
                     and x["status"] == "success"), None)
        if not hole:
            continue
        clean_metrics = {}
        cm = r.get("clean_segm_mAP")
        if cm: clean_metrics["coco/segm_mAP"] = cm
        hole_metrics = {"coco/segm_mAP": hole.get("segm_mAP"),
                        "coco/segm_mAP_50": hole.get("segm_mAP_50"),
                        "coco/segm_mAP_75": hole.get("segm_mAP_75")}
        fill_metrics = {"coco/segm_mAP": r.get("segm_mAP"),
                        "coco/segm_mAP_50": r.get("segm_mAP_50"),
                        "coco/segm_mAP_75": r.get("segm_mAP_75")}
        for mk in ["coco/segm_mAP", "coco/segm_mAP_50", "coco/segm_mAP_75"]:
            rec = compute_recovery(clean_metrics, hole_metrics, fill_metrics, mk)
            r[f"recovery_{mk.split('/')[-1]}"] = rec
    return results


def main() -> None:
    args = parse_args()
    project_root = get_project_root()
    exps = build_experiments(args)
    print(f"Experiments: {len(exps)}")
    for e in exps:
        print(f"  depth_fill_{e['hole_type']}_s{e['severity']}_{e['fill_method']}")
    if args.dry_run: return print("[DRY RUN] Done.")

    csv_path = osp.join(args.results_root, "robustness_fill_results.csv")
    all_results = []
    for exp in exps:
        try:
            result = run_one_experiment(exp, args, project_root)
            all_results.append(result)
            append_csv_row(csv_path, result, CSV_FIELDS)
            print(f"  [{result['status']}] {result['exp_name']} AP={result.get('segm_mAP','?')}")
        except Exception as e:
            print(f"  [FAIL] {exp.get('hole_type','?')}: {e}")
            if args.stop_on_error: raise

    all_results = compute_all_recoveries(all_results)
    # Re-write CSV with recovery values
    for r in all_results:
        append_csv_row(csv_path + ".with_recovery.csv", r, CSV_FIELDS)
    print(f"\nResults: {csv_path}")


if __name__ == "__main__":
    main()
