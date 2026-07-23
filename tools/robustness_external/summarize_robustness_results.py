#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Scan robustness_external/results/ and produce summary CSV/JSON/TXT."""

from __future__ import annotations

import argparse, csv, json, os, os.path as osp, sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tools.robustness_external.robustness_utils import (
    ensure_dir, parse_metrics_from_log, load_json, save_json,
    load_clean_baseline,
)

RESULT_KEYS = [
    "segm_mAP", "segm_mAP_50", "segm_mAP_75",
    "segm_mAP_s", "segm_mAP_m", "segm_mAP_l",
    "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
]


# ============================================================================
# Safe value helpers
# ============================================================================

def to_float_or_none(x) -> Optional[float]:
    """Safely convert any value to float or None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        try:
            return float(x)
        except (ValueError, OverflowError):
            return None
    if isinstance(x, str):
        x = x.strip()
        if x == "" or x.lower() in ("none", "nan", "?", "na"):
            return None
        try:
            return float(x)
        except ValueError:
            return None
    return None


def fmt4(x) -> str:
    """Format a value as .4f or 'NA'."""
    v = to_float_or_none(x)
    return f"{v:.4f}" if v is not None else "NA"


def has_core_metric(row: Dict) -> bool:
    """Check if a row has a valid segm_mAP value."""
    return to_float_or_none(row.get("segm_mAP")) is not None


# ============================================================================
# Scan results
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize robustness results")
    p.add_argument("--results-root", default="robustness_external/results")
    p.add_argument("--out-csv", default=None)
    p.add_argument("--out-json", default=None)
    p.add_argument("--out-txt", default=None)
    p.add_argument("--groups", nargs="*", default=None,
                   choices=["rgb", "depth_holes", "depth_fill"])
    return p.parse_args()


def scan_results(results_root: str, groups: Optional[List[str]] = None) -> List[Dict]:
    """Scan all subdirectories for metrics.json or test.log."""
    rows = []
    if not osp.isdir(results_root):
        return rows

    for sub in sorted(os.listdir(results_root)):
        sub_path = osp.join(results_root, sub)
        if not osp.isdir(sub_path):
            continue

        metrics_file = osp.join(sub_path, "metrics.json")
        log_file = osp.join(sub_path, "test.log")

        row = {"exp_name": sub, "data_root": "", "log_file": log_file,
               "status": "unknown", "group": ""}

        # 1. Try metrics.json first
        metrics = {}
        if osp.isfile(metrics_file):
            try:
                metrics = load_json(metrics_file)
            except Exception:
                metrics = {}
            if not isinstance(metrics, dict):
                metrics = {}

        # 2. If no valid segm_mAP, fallback to test.log parsing
        if not has_core_metric({"segm_mAP": metrics.get("coco/segm_mAP")}):
            if osp.isfile(log_file):
                metrics = parse_metrics_from_log(log_file)

        # 3. Fill RESULT_KEYS
        for k in RESULT_KEYS:
            row[k] = metrics.get(f"coco/{k}") or metrics.get(k)

        # 4. Status
        row["status"] = "success" if has_core_metric(row) else "no_metric"

        # 5. Determine group
        if sub.startswith("rgb_"):
            row["group"] = "rgb"
        elif sub.startswith("depth_fill_"):
            row["group"] = "depth_fill"
        elif sub.startswith("depth_"):
            row["group"] = "depth_holes"
        elif sub == "clean_mirror" or sub.startswith("original"):
            row["group"] = "clean"

        # 6. Extract severity and fill_method
        parts = sub.split("_")
        row["fill_method"] = "none"
        for p in parts:
            if p in ("none", "median", "inpaint_telea", "inpaint_ns"):
                row["fill_method"] = p
            if p.startswith("s") and len(p) <= 3 and p[1:].isdigit():
                row["severity_name"] = p
        row["corruption"] = "_".join(parts[:2]) if len(parts) >= 2 else sub

        rows.append(row)

    if groups:
        rows = [r for r in rows if r["group"] in groups]
    return rows


# ============================================================================
# Add stats
# ============================================================================

def add_stats(rows: List[Dict], clean_metrics: Dict) -> List[Dict]:
    """Add drop, retention, recovery to each row with safe value handling."""
    metric_aliases = [
        ("segm_mAP", "AP"),
        ("segm_mAP_50", "AP50"),
        ("segm_mAP_75", "AP75"),
    ]

    clean_segm = to_float_or_none(clean_metrics.get("coco/segm_mAP"))

    for r in rows:
        r["clean_segm_mAP"] = clean_segm

        for metric_name, alias in metric_aliases:
            exp_val = to_float_or_none(r.get(metric_name))
            clean_val = to_float_or_none(clean_metrics.get(f"coco/{metric_name}"))
            if clean_val is not None and exp_val is not None and clean_val != 0:
                r[f"{alias}_drop"] = clean_val - exp_val
                r[f"{alias}_retention"] = exp_val / clean_val
            else:
                r[f"{alias}_drop"] = None
                r[f"{alias}_retention"] = None

    # Recovery for fill experiments
    fill_rows = [r for r in rows if r["group"] == "depth_fill" and r.get("fill_method") != "none"]
    for r in fill_rows:
        hole = next((x for x in rows
                     if x["group"] == "depth_fill"
                     and x.get("fill_method") == "none"
                     and x["corruption"] == r["corruption"]
                     and x["severity_name"] == r["severity_name"]
                     and x["status"] == "success"), None)
        if hole and has_core_metric(r):
            for metric_name, alias in metric_aliases:
                ap_clean = to_float_or_none(clean_metrics.get(f"coco/{metric_name}"))
                ap_hole = to_float_or_none(hole.get(metric_name))
                ap_fill = to_float_or_none(r.get(metric_name))
                if ap_clean and ap_hole is not None and ap_fill is not None:
                    denom = ap_clean - ap_hole
                    if denom != 0:
                        r[f"recovery_{alias}"] = (ap_fill - ap_hole) / denom

    return rows


# ============================================================================
# Print summary
# ============================================================================

def print_summary(rows: List[Dict]) -> None:
    """Print group-level summary with safe filtering."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        if has_core_metric(r):
            groups[r["group"]].append(r)

    print("\n=== Group Summary ===")
    for g, items in groups.items():
        aps = [to_float_or_none(x.get("segm_mAP")) for x in items]
        aps = [x for x in aps if x is not None]
        rets = [to_float_or_none(x.get("AP_retention")) for x in items]
        rets = [x for x in rets if x is not None]
        recs = [to_float_or_none(x.get("recovery_AP")) for x in items]
        recs = [x for x in recs if x is not None]
        print(f"  {g}: {len(items)} experiments")
        if aps:
            print(f"    AP range: {min(aps):.4f} - {max(aps):.4f}")
        if rets:
            print(f"    Retention range: {min(rets):.4f} - {max(rets):.4f}")
        if recs:
            print(f"    Recovery range: {min(recs):.4f} - {max(recs):.4f}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    args = parse_args()
    results_root = args.results_root
    out_csv = args.out_csv or osp.join(results_root, "robustness_results.csv")
    out_json = args.out_json or osp.join(results_root, "robustness_results.json")
    out_txt = args.out_txt or osp.join(results_root, "robustness_summary.txt")

    rows = scan_results(results_root, args.groups)
    clean_metrics = load_clean_baseline(results_root)
    rows = add_stats(rows, clean_metrics)

    # Save CSV
    all_keys = [
        "exp_name", "group", "corruption", "severity_name", "fill_method", "status",
        "segm_mAP", "segm_mAP_50", "segm_mAP_75", "segm_mAP_s", "segm_mAP_m", "segm_mAP_l",
        "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
        "clean_segm_mAP",
        "AP_drop", "AP_retention", "AP50_drop", "AP50_retention",
        "AP75_drop", "AP75_retention",
        "recovery_AP", "recovery_AP50", "recovery_AP75",
    ]
    ensure_dir(osp.dirname(out_csv))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            # convert None back to "" for CSV
            row_out = {}
            for k in all_keys:
                v = r.get(k)
                row_out[k] = v if v is not None else ""
            writer.writerow(row_out)

    # Save JSON with safe float conversion
    json_rows = []
    for r in rows:
        jr = {}
        for k, v in r.items():
            if isinstance(v, float):
                jr[k] = v
            elif v is None:
                jr[k] = None
            else:
                jr[k] = v
        json_rows.append(jr)
    save_json({"clean_metrics": clean_metrics, "results": json_rows}, out_json)

    # Save TXT (safe format)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=== Robustness Summary ===\n")
        f.write(f"Clean segm_mAP: {fmt4(clean_metrics.get('coco/segm_mAP'))}\n")
        f.write(f"Total experiments: {len(rows)}\n\n")
        for r in rows:
            f.write(
                f"{r.get('exp_name',''):42s} {r.get('group',''):15s} "
                f"AP={fmt4(r.get('segm_mAP'))} "
                f"ret={fmt4(r.get('AP_retention'))} "
                f"drop={fmt4(r.get('AP_drop'))} "
                f"rec={fmt4(r.get('recovery_AP'))} "
                f"[{r.get('status','unknown')}]\n"
            )

    print_summary(rows)
    print(f"\nCSV: {out_csv}")
    print(f"JSON: {out_json}")
    print(f"TXT: {out_txt}")


if __name__ == "__main__":
    main()
