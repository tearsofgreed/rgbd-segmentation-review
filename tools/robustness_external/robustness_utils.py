#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared utilities for external robustness evaluation scripts."""

from __future__ import annotations

import csv
import json
import os
import os.path as osp
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Filesystem helpers
# ============================================================================

def ensure_dir(path: str) -> str:
    """Create directory if not exists, return path."""
    os.makedirs(path, exist_ok=True)
    return path


def normalize_cfg_path(path: str) -> str:
    """Normalize a path for use in --cfg-options (forward slashes)."""
    p = path.replace("\\", "/")
    # Ensure trailing slash for directory-like paths (data_root)
    if not p.endswith("/") and not p.endswith(".json"):
        p += "/"
    return p


def get_project_root() -> str:
    """Determine the MMDetection project root from this script's location."""
    # tools/robustness_external/robustness_utils.py -> project root is 3 levels up
    script_dir = osp.dirname(osp.abspath(__file__))
    return osp.abspath(osp.join(script_dir, "..", ".."))


# ============================================================================
# Test command builder
# ============================================================================

def build_test_command(
    python_exe: str,
    config: str,
    checkpoint: str,
    data_root: str,
    ann_file: str,
    work_dir: str,
) -> List[str]:
    """Build a subprocess-safe test command list.

    Returns a list[str] suitable for subprocess.run (no shell=True).
    Overrides data_root for both dataloader and evaluator.
    """
    data_root_norm = normalize_cfg_path(data_root)
    ann_file_norm = normalize_cfg_path(ann_file)

    return [
        python_exe,
        "tools/test.py",
        config,
        checkpoint,
        "--work-dir", work_dir,
        "--cfg-options",
        f"test_dataloader.dataset.data_root={data_root_norm}",
        f"val_dataloader.dataset.data_root={data_root_norm}",
        f"test_evaluator.ann_file={ann_file_norm}",
        f"val_evaluator.ann_file={ann_file_norm}",
    ]


# ============================================================================
# Subprocess runner
# ============================================================================

def build_subprocess_env(project_root: str) -> Dict[str, str]:
    """Clone os.environ and prepend project_root to PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    if existing:
        env["PYTHONPATH"] = project_root + osp.pathsep + existing
    else:
        env["PYTHONPATH"] = project_root
    return env


def debug_log_command(cmd: List[str], log_file: str) -> None:
    """Write command structure to a debug file."""
    import os
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("# CMD repr\n" + repr(cmd) + "\n\n")
        f.write("# CMD human-readable\n")
        parts = []
        for c in cmd:
            if " " in c or "=" in c:
                parts.append('"' + c + '"')
            else:
                parts.append(c)
        f.write(" ".join(parts) + "\n\n")
        f.write("# CMD numbered\n")
        for i, c in enumerate(cmd):
            f.write("[" + str(i) + "] " + c + "\n")


def run_command_and_log(
    cmd: List[str],
    log_file: str,
    cwd: str,
    env: Optional[Dict[str, str]] = None,
) -> int:
    """Run a subprocess command, tee stdout/stderr to a log file and terminal.

    Returns the process return code.
    """
    debug_log_command(cmd, osp.join(osp.dirname(log_file), "command.txt"))

    with open(log_file, "w", encoding="utf-8", errors="replace") as lf:
        lf.write(f"CMD: {' '.join(cmd)}\n")
        lf.write(f"CWD: {cwd}\n")
        lf.write("-" * 60 + "\n")
        lf.flush()

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

        if proc.stdout is None:
            proc.wait()
            return proc.returncode

        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
            lf.flush()

        proc.wait()

    return proc.returncode


# ============================================================================
# Metrics parsing
# ============================================================================

def parse_metrics_from_log(log_file: str) -> Dict[str, Optional[float]]:
    """Parse COCO metrics from a test log file.

    Searches for lines like:
        coco/segm_mAP: 0.8170  coco/segm_mAP_50: 0.9240  ...
    Also tries copypaste format:
        Average Precision  (AP) @[ IoU=0.50:0.95 | ... ] = 0.817
    """
    if not osp.isfile(log_file):
        return {}

    keys_order = [
        "coco/segm_mAP", "coco/segm_mAP_50", "coco/segm_mAP_75",
        "coco/segm_mAP_s", "coco/segm_mAP_m", "coco/segm_mAP_l",
        "coco/bbox_mAP", "coco/bbox_mAP_50", "coco/bbox_mAP_75",
    ]

    metrics: Dict[str, Optional[float]] = {}

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Try copypaste format first (from mmdet evaluation)
    segm_map_pat = re.compile(
        r"coco/segm_mAP(?:_(\w+))?\s*[:-]\s*([\d.]+)"
    )
    for m in segm_map_pat.finditer(content):
        suffix = m.group(1)
        key = f"coco/segm_mAP_{suffix}" if suffix else "coco/segm_mAP"
        try:
            metrics[key] = float(m.group(2))
        except ValueError:
            pass

    bbox_map_pat = re.compile(
        r"coco/bbox_mAP(?:_(\w+))?\s*[:-]\s*([\d.]+)"
    )
    for m in bbox_map_pat.finditer(content):
        suffix = m.group(1)
        key = f"coco/bbox_mAP_{suffix}" if suffix else "coco/bbox_mAP"
        try:
            metrics[key] = float(m.group(2))
        except ValueError:
            pass

    # Also try the classic mmdet print format
    # OrderedDict([('coco/segm_mAP', 0.817), ('coco/segm_mAP_50', 0.924), ...])
    od_pat = re.compile(r"OrderedDict\(\[(.*?)\]\)", re.DOTALL)
    od_match = od_pat.search(content)
    if od_match:
        for kv in re.finditer(r"\('([^']+)',\s*([\d.]+)\)", od_match.group(1)):
            key = kv.group(1)
            try:
                metrics[key] = float(kv.group(2))
            except ValueError:
                pass

    return metrics


# ============================================================================
# JSON / CSV helpers
# ============================================================================

def has_core_metrics(metrics: Dict) -> bool:
    """Check if metrics dict contains the core coco/segm_mAP key."""
    return isinstance(metrics, dict) and metrics.get("coco/segm_mAP") is not None


def write_experiment_summary(result: Dict[str, Any], path: str) -> None:
    """Write experiment summary to a text file."""
    ensure_dir(osp.dirname(path))
    keys = [
        "exp_name", "group", "corruption", "severity", "severity_name",
        "fill_method", "checkpoint", "data_root", "ann_file", "log_file",
        "segm_mAP", "segm_mAP_50", "segm_mAP_75",
        "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
        "clean_segm_mAP",
        "AP_drop", "AP_retention", "AP50_drop", "AP50_retention",
        "AP75_drop", "AP75_retention",
        "recovery_AP", "recovery_AP50", "recovery_AP75",
        "status", "error",
    ]
    with open(path, "w", encoding="utf-8") as f:
        for k in keys:
            v = result.get(k, "")
            if isinstance(v, float):
                f.write(f"{k}: {v:.4f}\n")
            else:
                f.write(f"{k}: {v}\n")


def save_json(obj: Any, path: str) -> None:
    """Save object as JSON with UTF-8 encoding."""
    ensure_dir(osp.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def append_csv_row(csv_path: str, row: Dict[str, Any], fieldnames: List[str]) -> None:
    """Append a row to a CSV file; write header if file is new."""
    ensure_dir(osp.dirname(csv_path))
    write_header = not osp.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ============================================================================
# Baseline / stats helpers
# ============================================================================

def load_clean_baseline(results_root: str) -> Dict[str, Optional[float]]:
    """Load clean baseline metrics from clean_mirror metrics.json."""
    metrics_path = osp.join(results_root, "clean_mirror", "metrics.json")
    if osp.isfile(metrics_path):
        return load_json(metrics_path)
    # Fallback: try clean_mirror test.log
    log_path = osp.join(results_root, "clean_mirror", "test.log")
    if osp.isfile(log_path):
        return parse_metrics_from_log(log_path)
    return {}


def compute_drop_retention(
    clean_metrics: Dict[str, Optional[float]],
    exp_metrics: Dict[str, Optional[float]],
    metric_key: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Compute AP_drop and Retention for a single metric key."""
    clean_val = clean_metrics.get(metric_key)
    exp_val = exp_metrics.get(metric_key)
    if clean_val is None or exp_val is None or clean_val == 0:
        return None, None
    drop = clean_val - exp_val
    retention = exp_val / clean_val
    return drop, retention


def compute_recovery(
    clean_metrics: Dict[str, Optional[float]],
    hole_metrics: Dict[str, Optional[float]],
    fill_metrics: Dict[str, Optional[float]],
    metric_key: str,
) -> Optional[float]:
    """Compute Recovery = (AP_filled - AP_hole) / (AP_clean - AP_hole)."""
    ap_clean = clean_metrics.get(metric_key)
    ap_hole = hole_metrics.get(metric_key)
    ap_fill = fill_metrics.get(metric_key)
    if ap_clean is None or ap_hole is None or ap_fill is None:
        return None
    denom = ap_clean - ap_hole
    if denom == 0:
        return 0.0
    return (ap_fill - ap_hole) / denom
