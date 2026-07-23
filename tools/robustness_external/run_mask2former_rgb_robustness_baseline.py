#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run RGB-only Mask2Former robustness evaluation on already generated RGB corrupted datasets.

This script does NOT generate datasets.
It only runs tools/test.py on:
- clean_mirror
- RGB corruptions:
  rgb_brightness
  rgb_white_balance warm/cool
  rgb_local_shadow
  rgb_specular_highlight

Expected checkpoint layout:
  work_dirs/mask2former_rgb_pth/1.pth
  work_dirs/mask2former_rgb_pth/2.pth
  work_dirs/mask2former_rgb_pth/3.pth

Example:
  python tools/robustness_external/run_mask2former_rgb_robustness_baseline.py ^
    --config configs/mask2former/mask2former_rgb_baseline.py ^
    --checkpoint-dir work_dirs/mask2former_rgb_pth ^
    --folds 1 2 3 ^
    --groups rgb ^
    --skip-existing-tests ^
    --show-gpu ^
    --sleep-after-test 45
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


PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Experiment definitions: RGB only
# ============================================================

RGB_BRIGHTNESS = [
    {"group": "rgb_brightness", "exp_name": f"rgb_brightness_s{i}", "severity": i}
    for i in range(1, 6)
]

RGB_WHITE_BALANCE = (
    [{"group": "rgb_white_balance", "exp_name": f"rgb_white_balance_warm_s{i}", "severity": i, "direction": "warm"} for i in range(1, 6)]
    + [{"group": "rgb_white_balance", "exp_name": f"rgb_white_balance_cool_s{i}", "severity": i, "direction": "cool"} for i in range(1, 6)]
)

RGB_LOCAL_SHADOW = [
    {"group": "rgb_local_shadow", "exp_name": f"rgb_local_shadow_s{i}", "severity": i}
    for i in range(1, 6)
]

RGB_SPECULAR_HIGHLIGHT = [
    {"group": "rgb_specular_highlight", "exp_name": f"rgb_specular_highlight_s{i}", "severity": i}
    for i in range(1, 6)
]

GROUPS = {
    "rgb_brightness": RGB_BRIGHTNESS,
    "rgb_white_balance": RGB_WHITE_BALANCE,
    "rgb_local_shadow": RGB_LOCAL_SHADOW,
    "rgb_specular_highlight": RGB_SPECULAR_HIGHLIGHT,
    "rgb": RGB_BRIGHTNESS + RGB_WHITE_BALANCE + RGB_LOCAL_SHADOW + RGB_SPECULAR_HIGHLIGHT,
}


CSV_FIELDS = [
    "method", "fold", "checkpoint", "exp_name", "group", "severity", "direction",
    "data_root", "ann_file", "log_file",
    "segm_mAP", "segm_mAP_50", "segm_mAP_75",
    "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
    "clean_segm_mAP", "AP_drop", "AP_retention",
    "AP50_drop", "AP50_retention", "AP75_drop", "AP75_retention",
    "status", "error",
]


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("none", "nan", "na", "?"):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(osp.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def append_csv_row(path: str, row: Dict[str, Any], fields: List[str]) -> None:
    ensure_dir(osp.dirname(path))
    exists = osp.isfile(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fields})


def has_core_metrics(metrics: Dict[str, Any]) -> bool:
    return to_float_or_none(metrics.get("coco/segm_mAP")) is not None


def load_metrics_if_valid(path: str) -> Optional[Dict[str, Any]]:
    if not osp.isfile(path):
        return None
    try:
        metrics = load_json(path)
    except Exception:
        return None
    return metrics if has_core_metrics(metrics) else None


def parse_metrics_from_log(log_file: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if not osp.isfile(log_file):
        return metrics

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    keys = [
        "coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75",
        "coco/bbox_mAP_s", "coco/bbox_mAP_m", "coco/bbox_mAP_l",
        "coco/segm_mAP", "coco/segm_mAP_50", "coco/segm_mAP_75",
        "coco/segm_mAP_s", "coco/segm_mAP_m", "coco/segm_mAP_l",
    ]

    import re
    for k in keys:
        pattern = re.escape(k) + r"[:=]\s*([0-9]*\.?[0-9]+)"
        found = re.findall(pattern, text)
        if found:
            metrics[k] = float(found[-1])

    return metrics


def write_summary(row: Dict[str, Any], path: str) -> None:
    ensure_dir(osp.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for k in CSV_FIELDS:
            v = row.get(k)
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {'' if v is None else v}\n")


def normalize_cfg_path(path: str, is_dir: bool = False) -> str:
    s = path.replace("\\", "/")
    if is_dir and not s.endswith("/"):
        s += "/"
    return s


def build_env(cuda_alloc_conf: str) -> Dict[str, str]:
    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + old_pp if old_pp else "")
    env["PYTORCH_CUDA_ALLOC_CONF"] = cuda_alloc_conf
    return env


def build_test_command(
    python_exe: str,
    config: str,
    checkpoint: str,
    data_root: str,
    ann_file: str,
    work_dir: str,
    set_data_prefix: bool = True,
) -> List[str]:
    data_root_norm = normalize_cfg_path(data_root, is_dir=True)
    ann_file_norm = normalize_cfg_path(ann_file, is_dir=False)

    cmd = [
        python_exe,
        "tools/test.py",
        config,
        checkpoint,
        "--work-dir",
        work_dir,
        "--cfg-options",
        f"test_dataloader.dataset.data_root={data_root_norm}",
        f"val_dataloader.dataset.data_root={data_root_norm}",
        f"test_evaluator.ann_file={ann_file_norm}",
        f"val_evaluator.ann_file={ann_file_norm}",
    ]

    # Most MMDetection RGB configs use data_prefix=dict(img='images/').
    # If your baseline config already uses images/, this is harmless.
    # If your config uses a different key, disable with --no-set-data-prefix.
    if set_data_prefix:
        cmd += [
            "test_dataloader.dataset.data_prefix.img=images/",
            "val_dataloader.dataset.data_prefix.img=images/",
        ]

    return cmd


def run_command_and_log(cmd: List[str], log_file: str, cwd: str, env: Dict[str, str]) -> int:
    ensure_dir(osp.dirname(log_file))

    command_txt = osp.join(osp.dirname(log_file), "command.txt")
    with open(command_txt, "w", encoding="utf-8") as f:
        f.write("repr:\n")
        f.write(repr(cmd) + "\n\n")
        f.write("human:\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("args:\n")
        for i, a in enumerate(cmd):
            f.write(f"[{i}] {a}\n")

    with open(log_file, "w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
        proc.wait()
        return int(proc.returncode)


def print_gpu_status(title: str = "") -> None:
    print("\n" + "=" * 80)
    print(f"GPU STATUS {title}".strip())
    print("=" * 80)
    subprocess.run(["nvidia-smi"], check=False)


def kill_stale_gpu_python() -> None:
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
                print(f"[GPU CLEAN] killing stale python PID={pid}")
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False)
    except Exception as e:
        print(f"[WARN] GPU cleanup failed: {e}")


def selected_experiments(groups: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for g in groups:
        for e in GROUPS[g]:
            if e["exp_name"] not in seen:
                out.append(e)
                seen.add(e["exp_name"])
    return out


def clean_experiment() -> Dict[str, Any]:
    return {
        "group": "clean",
        "exp_name": "clean_mirror",
        "severity": 0,
        "direction": "",
    }


def checkpoint_path(checkpoint_dir: str, fold: int) -> str:
    return osp.join(checkpoint_dir, f"{fold}.pth")


def data_root_for_exp(datasets_root: str, exp_name: str, fallback_clean_root: str) -> str:
    root = osp.join(datasets_root, exp_name)
    if exp_name == "clean_mirror" and not osp.isdir(root):
        return fallback_clean_root
    return root


def ann_file_for_data_root(data_root: str) -> str:
    return osp.join(data_root, "annotations", "val.json")


def dataset_ready(data_root: str) -> bool:
    return (
        osp.isdir(osp.join(data_root, "images"))
        and osp.isfile(osp.join(data_root, "annotations", "val.json"))
    )


def get_clean_metrics(fold_results_root: str) -> Dict[str, Any]:
    p = osp.join(fold_results_root, "clean_mirror", "metrics.json")
    return load_metrics_if_valid(p) or {}


def compute_drop_ret(clean: Dict[str, Any], cur: Dict[str, Any], clean_key: str, cur_key: str):
    c = to_float_or_none(clean.get(clean_key))
    v = to_float_or_none(cur.get(cur_key))
    if c is None or v is None:
        return None, None
    return c - v, v / c if c != 0 else None


def run_one(
    fold: int,
    ckpt: str,
    exp: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    exp_name = exp["exp_name"]
    fold_results_root = ensure_dir(osp.join(args.results_root, f"fold{fold}"))
    result_dir = ensure_dir(osp.join(fold_results_root, exp_name))

    metrics_file = osp.join(result_dir, "metrics.json")
    summary_file = osp.join(result_dir, "summary.txt")
    log_file = osp.join(result_dir, "test.log")

    old_metrics = load_metrics_if_valid(metrics_file)
    if args.skip_existing_tests and old_metrics is not None:
        print(f"[SKIP] fold{fold} {exp_name}: valid metrics already exist")
        return {
            "method": args.method_name,
            "fold": fold,
            "checkpoint": ckpt,
            "exp_name": exp_name,
            "group": exp.get("group"),
            "severity": exp.get("severity"),
            "direction": exp.get("direction", ""),
            "segm_mAP": old_metrics.get("coco/segm_mAP"),
            "status": "skipped",
        }

    data_root = data_root_for_exp(args.datasets_root, exp_name, args.clean_data_root)
    ann_file = ann_file_for_data_root(data_root)

    if not dataset_ready(data_root):
        raise FileNotFoundError(
            f"Dataset not ready for {exp_name}: {data_root}\n"
            f"Expected images/ and annotations/val.json"
        )

    cmd = build_test_command(
        python_exe=args.python_exe,
        config=args.config,
        checkpoint=ckpt,
        data_root=data_root,
        ann_file=ann_file,
        work_dir=result_dir,
        set_data_prefix=not args.no_set_data_prefix,
    )

    print("\n" + "=" * 80)
    print(f"[TEST] {args.method_name} | fold{fold} | {exp_name}")
    print(f"checkpoint: {ckpt}")
    print(f"data_root : {data_root}")
    print(f"result_dir: {result_dir}")
    print("=" * 80)

    if args.dry_run:
        print(" ".join(cmd))
        return {
            "method": args.method_name,
            "fold": fold,
            "checkpoint": ckpt,
            "exp_name": exp_name,
            "group": exp.get("group"),
            "severity": exp.get("severity"),
            "direction": exp.get("direction", ""),
            "status": "dry_run",
        }

    env = build_env(args.cuda_alloc_conf)
    rc = run_command_and_log(cmd, log_file, PROJECT_ROOT, env)
    metrics = parse_metrics_from_log(log_file)
    save_json(metrics, metrics_file)

    clean_metrics = get_clean_metrics(fold_results_root)

    row: Dict[str, Any] = {
        "method": args.method_name,
        "fold": fold,
        "checkpoint": ckpt,
        "exp_name": exp_name,
        "group": exp.get("group"),
        "severity": exp.get("severity"),
        "direction": exp.get("direction", ""),
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
        row["status"] = "success"
        row["error"] = f"test rc={rc}, but metrics parsed"
    elif rc != 0:
        row["status"] = "failed"
        row["error"] = f"test rc={rc}"

    if exp_name != "clean_mirror" and clean_metrics:
        d, r = compute_drop_ret(clean_metrics, metrics, "coco/segm_mAP", "coco/segm_mAP")
        row["AP_drop"] = d
        row["AP_retention"] = r

        d, r = compute_drop_ret(clean_metrics, metrics, "coco/segm_mAP_50", "coco/segm_mAP_50")
        row["AP50_drop"] = d
        row["AP50_retention"] = r

        d, r = compute_drop_ret(clean_metrics, metrics, "coco/segm_mAP_75", "coco/segm_mAP_75")
        row["AP75_drop"] = d
        row["AP75_retention"] = r

    write_summary(row, summary_file)

    append_csv_row(osp.join(fold_results_root, "robustness_results.csv"), row, CSV_FIELDS)
    append_csv_row(osp.join(args.results_root, "all_results.csv"), row, CSV_FIELDS)

    print(f"[DONE] fold{fold} {exp_name} status={row['status']} AP={row.get('segm_mAP')}")
    return row


def sleep_and_clean(args: argparse.Namespace, title: str) -> None:
    if args.kill_stale_gpu_python:
        kill_stale_gpu_python()
    if args.show_gpu:
        print_gpu_status(f"after {title}")
    if args.sleep_after_test > 0:
        print(f"[SLEEP] {args.sleep_after_test}s")
        time.sleep(args.sleep_after_test)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--method-name", default="Mask2Former-RGB")
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint-dir", default="work_dirs/mask2former_rgb_pth")
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--groups", nargs="+", default=["rgb"], choices=list(GROUPS.keys()))
    p.add_argument("--datasets-root", default="robustness_external/datasets")
    p.add_argument("--clean-data-root", default="data/seed1")
    p.add_argument("--results-root", default="robustness_external/results_mask2former_rgb_3fold")
    p.add_argument("--python-exe", default=sys.executable)
    p.add_argument("--cuda-alloc-conf", default="max_split_size_mb:128")
    p.add_argument("--sleep-after-test", type=int, default=45)
    p.add_argument("--skip-existing-tests", action="store_true")
    p.add_argument("--show-gpu", action="store_true")
    p.add_argument("--kill-stale-gpu-python", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--stop-on-error", action="store_true")
    p.add_argument("--max-experiments", type=int, default=None)
    p.add_argument("--no-set-data-prefix", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.results_root)

    exps = selected_experiments(args.groups)
    if args.max_experiments is not None:
        exps = exps[:args.max_experiments]

    print("=" * 80)
    print("Mask2Former RGB robustness baseline runner")
    print(f"project_root  : {PROJECT_ROOT}")
    print(f"config        : {args.config}")
    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"folds         : {args.folds}")
    print(f"groups        : {args.groups}")
    print(f"experiments   : {len(exps)}")
    print(f"datasets_root : {args.datasets_root}")
    print(f"results_root  : {args.results_root}")
    print("=" * 80)

    if args.show_gpu:
        print_gpu_status("before start")

    for fold in args.folds:
        ckpt = checkpoint_path(args.checkpoint_dir, fold)
        if not osp.isfile(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        print("\n" + "#" * 80)
        print(f"# FOLD {fold}")
        print(f"# checkpoint: {ckpt}")
        print("#" * 80)

        try:
            run_one(fold, ckpt, clean_experiment(), args)
            sleep_and_clean(args, f"fold{fold} clean_mirror")

            for exp in exps:
                run_one(fold, ckpt, exp, args)
                sleep_and_clean(args, f"fold{fold} {exp['exp_name']}")

        except Exception as e:
            print(f"[ERROR] fold{fold}: {e}")
            if args.stop_on_error:
                raise

    print("\nALL DONE.")
    print(f"Global CSV: {osp.join(args.results_root, 'all_results.csv')}")


if __name__ == "__main__":
    main()