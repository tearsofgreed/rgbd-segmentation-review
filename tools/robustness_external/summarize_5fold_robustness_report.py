#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Summarize five-fold RGB-D robustness experiments.

This script is read-only:
- It does not run inference.
- It does not regenerate datasets.
- It does not modify original data.
- It reads results_5fold and robustness_external/datasets, then writes report files.

Default input:
  robustness_external/results_5fold
  robustness_external/datasets

Default output:
  robustness_external/reports_5fold
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import os.path as osp
import re
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Basic helpers
# ============================================================

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float, np.integer, np.floating)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in ("none", "nan", "na", "?"):
            return None
        try:
            v = float(s)
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        except ValueError:
            return None
    return None


def fmt4(x: Any) -> str:
    v = to_float_or_none(x)
    return f"{v:.4f}" if v is not None else "NA"


def safe_read_json(path: str) -> Dict[str, Any]:
    if not osp.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def safe_write_json(obj: Any, path: str) -> None:
    ensure_dir(osp.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_summary_txt(path: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if not osp.isfile(path):
        return out
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


# ============================================================
# Expected experiment list
# ============================================================

def expected_experiment_names() -> List[str]:
    names = ["clean_mirror"]

    for i in range(1, 6):
        names.append(f"rgb_brightness_s{i}")

    for prefix in ("warm", "cool"):
        for i in range(1, 6):
            names.append(f"rgb_white_balance_{prefix}_s{i}")

    for i in range(1, 6):
        names.append(f"rgb_local_shadow_s{i}")

    for i in range(1, 6):
        names.append(f"rgb_specular_highlight_s{i}")

    for kind in ("random", "block", "edge"):
        for i in range(1, 6):
            names.append(f"depth_{kind}_holes_s{i}")

    for hole_type in ("block", "edge"):
        for i in range(1, 6):
            for fill in ("none", "median", "inpaint_telea"):
                names.append(f"depth_fill_{hole_type}_s{i}_{fill}")

    return names


def infer_metadata(exp_name: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "exp_name": exp_name,
        "family": "",
        "series": "",
        "series_label": "",
        "group": "",
        "severity": None,
        "severity_name": "",
        "direction": "",
        "hole_type": "",
        "fill_method": "none",
    }

    if exp_name == "clean_mirror":
        meta.update({
            "family": "clean",
            "series": "clean",
            "series_label": "Clean",
            "group": "clean",
            "severity": 0,
            "severity_name": "clean",
        })
        return meta

    m = re.search(r"_s([1-5])(?:_|$)", exp_name)
    if m:
        meta["severity"] = int(m.group(1))
        meta["severity_name"] = f"s{m.group(1)}"

    if exp_name.startswith("rgb_brightness_"):
        meta.update({
            "family": "rgb_brightness",
            "series": "rgb_brightness",
            "series_label": "RGB brightness",
            "group": "rgb",
        })
    elif exp_name.startswith("rgb_white_balance_warm_"):
        meta.update({
            "family": "rgb_white_balance",
            "series": "rgb_white_balance_warm",
            "series_label": "RGB white balance warm",
            "group": "rgb",
            "direction": "warm",
        })
    elif exp_name.startswith("rgb_white_balance_cool_"):
        meta.update({
            "family": "rgb_white_balance",
            "series": "rgb_white_balance_cool",
            "series_label": "RGB white balance cool",
            "group": "rgb",
            "direction": "cool",
        })
    elif exp_name.startswith("rgb_local_shadow_"):
        meta.update({
            "family": "rgb_local_shadow",
            "series": "rgb_local_shadow",
            "series_label": "RGB local shadow",
            "group": "rgb",
        })
    elif exp_name.startswith("rgb_specular_highlight_"):
        meta.update({
            "family": "rgb_specular_highlight",
            "series": "rgb_specular_highlight",
            "series_label": "RGB specular highlight",
            "group": "rgb",
        })
    elif exp_name.startswith("depth_random_holes_"):
        meta.update({
            "family": "depth_holes",
            "series": "depth_random_holes",
            "series_label": "Depth random holes",
            "group": "depth_holes",
            "hole_type": "random",
        })
    elif exp_name.startswith("depth_block_holes_"):
        meta.update({
            "family": "depth_holes",
            "series": "depth_block_holes",
            "series_label": "Depth block holes",
            "group": "depth_holes",
            "hole_type": "block",
        })
    elif exp_name.startswith("depth_edge_holes_"):
        meta.update({
            "family": "depth_holes",
            "series": "depth_edge_holes",
            "series_label": "Depth edge holes",
            "group": "depth_holes",
            "hole_type": "edge",
        })
    elif exp_name.startswith("depth_fill_"):
        # depth_fill_block_s3_median
        parts = exp_name.split("_")
        hole_type = parts[2] if len(parts) > 2 else ""
        fill_method = "_".join(parts[4:]) if len(parts) > 4 else "none"
        meta.update({
            "family": "depth_fill",
            "series": f"depth_fill_{hole_type}_{fill_method}",
            "series_label": f"Depth fill {hole_type} / {fill_method}",
            "group": "depth_fill",
            "hole_type": hole_type,
            "fill_method": fill_method,
        })

    return meta


# ============================================================
# Load and clean results
# ============================================================

def load_rows_from_all_csv(results_root: str) -> pd.DataFrame:
    csv_path = osp.join(results_root, "all_results.csv")
    if not osp.isfile(csv_path):
        return pd.DataFrame()
    return pd.read_csv(csv_path, encoding="utf-8")


def load_rows_by_scanning(results_root: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    root = Path(results_root)
    for fold_dir in sorted(root.glob("fold*")):
        if not fold_dir.is_dir():
            continue
        fold_match = re.search(r"fold(\d+)", fold_dir.name)
        if not fold_match:
            continue
        fold = int(fold_match.group(1))

        for exp_dir in sorted(fold_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            exp_name = exp_dir.name
            metrics_path = str(exp_dir / "metrics.json")
            summary_path = str(exp_dir / "summary.txt")
            log_path = str(exp_dir / "test.log")

            metrics = safe_read_json(metrics_path)
            summary = parse_summary_txt(summary_path)
            meta = infer_metadata(exp_name)

            row: Dict[str, Any] = {}
            row.update(meta)
            row.update(summary)

            row["fold"] = fold
            row["exp_name"] = exp_name
            row["log_file"] = row.get("log_file") or log_path
            row["status"] = row.get("status") or ("success" if to_float_or_none(metrics.get("coco/segm_mAP")) is not None else "no_metric")

            row["segm_mAP"] = metrics.get("coco/segm_mAP", row.get("segm_mAP"))
            row["segm_mAP_50"] = metrics.get("coco/segm_mAP_50", row.get("segm_mAP_50"))
            row["segm_mAP_75"] = metrics.get("coco/segm_mAP_75", row.get("segm_mAP_75"))
            row["bbox_mAP"] = metrics.get("coco/bbox_mAP", row.get("bbox_mAP"))
            row["bbox_mAP_50"] = metrics.get("coco/bbox_mAP_50", row.get("bbox_mAP_50"))
            row["bbox_mAP_75"] = metrics.get("coco/bbox_mAP_75", row.get("bbox_mAP_75"))

            rows.append(row)

    return pd.DataFrame(rows)


def normalize_dataframe(df: pd.DataFrame, results_root: str) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    df["_row_order"] = np.arange(len(df))

    # Make sure basic columns exist.
    for col in [
        "fold", "exp_name", "group", "family", "series", "series_label",
        "severity", "severity_name", "direction", "hole_type", "fill_method",
        "status", "error", "checkpoint", "data_root", "ann_file", "log_file",
        "segm_mAP", "segm_mAP_50", "segm_mAP_75",
        "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
        "clean_segm_mAP", "AP_drop", "AP_retention",
        "AP50_drop", "AP50_retention", "AP75_drop", "AP75_retention",
        "recovery_AP", "recovery_AP50", "recovery_AP75",
    ]:
        if col not in df.columns:
            df[col] = None

    # Infer fold from path if needed.
    for idx, row in df.iterrows():
        if pd.isna(row.get("fold")):
            s = str(row.get("log_file", ""))
            m = re.search(r"fold(\d+)", s)
            if m:
                df.at[idx, "fold"] = int(m.group(1))

    # Infer metadata from exp_name.
    for idx, row in df.iterrows():
        exp_name = str(row["exp_name"])
        meta = infer_metadata(exp_name)
        for k, v in meta.items():
            if k not in df.columns:
                df[k] = None
            cur = row.get(k)
            if cur is None or (isinstance(cur, float) and pd.isna(cur)) or str(cur) == "":
                df.at[idx, k] = v

    # Numeric conversion.
    numeric_cols = [
        "fold", "severity",
        "segm_mAP", "segm_mAP_50", "segm_mAP_75",
        "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75",
        "clean_segm_mAP", "AP_drop", "AP_retention",
        "AP50_drop", "AP50_retention", "AP75_drop", "AP75_retention",
        "recovery_AP", "recovery_AP50", "recovery_AP75",
    ]
    for col in numeric_cols:
        df[col] = df[col].map(to_float_or_none)

    df["fold"] = df["fold"].astype("Int64")
    df["severity"] = df["severity"].astype("Int64")

    # Deduplicate: keep best valid row per fold+exp_name.
    # This avoids old failed rows appended before successful reruns.
    def status_rank(s: Any) -> int:
        ss = str(s).lower()
        if ss == "success":
            return 3
        if ss == "skipped":
            return 2
        if "parsed" in ss:
            return 2
        if ss in ("failed", "no_metric"):
            return 0
        return 1

    df["_status_rank"] = df["status"].map(status_rank)
    df["_has_ap"] = df["segm_mAP"].notna().astype(int)

    df = (
        df.sort_values(["fold", "exp_name", "_has_ap", "_status_rank", "_row_order"])
          .groupby(["fold", "exp_name"], as_index=False)
          .tail(1)
          .reset_index(drop=True)
    )

    # Per-fold clean baseline.
    clean_map = {}
    clean_rows = df[df["exp_name"] == "clean_mirror"]
    for _, r in clean_rows.iterrows():
        if pd.notna(r["fold"]) and to_float_or_none(r["segm_mAP"]) is not None:
            clean_map[int(r["fold"])] = float(r["segm_mAP"])

    for idx, row in df.iterrows():
        fold = to_float_or_none(row.get("fold"))
        if fold is None:
            continue
        fold_i = int(fold)
        clean_val = clean_map.get(fold_i)

        if clean_val is not None and row["exp_name"] != "clean_mirror":
            df.at[idx, "clean_segm_mAP"] = clean_val

            ap = to_float_or_none(row.get("segm_mAP"))
            ap50 = to_float_or_none(row.get("segm_mAP_50"))
            ap75 = to_float_or_none(row.get("segm_mAP_75"))

            if ap is not None:
                df.at[idx, "AP_drop"] = clean_val - ap
                df.at[idx, "AP_retention"] = ap / clean_val if clean_val != 0 else None

            # Use clean_mirror AP50/AP75 if available.
            clean_row = clean_rows[clean_rows["fold"] == fold_i]
            if not clean_row.empty:
                clean50 = to_float_or_none(clean_row.iloc[0].get("segm_mAP_50"))
                clean75 = to_float_or_none(clean_row.iloc[0].get("segm_mAP_75"))
                if clean50 is not None and ap50 is not None:
                    df.at[idx, "AP50_drop"] = clean50 - ap50
                    df.at[idx, "AP50_retention"] = ap50 / clean50 if clean50 != 0 else None
                if clean75 is not None and ap75 is not None:
                    df.at[idx, "AP75_drop"] = clean75 - ap75
                    df.at[idx, "AP75_retention"] = ap75 / clean75 if clean75 != 0 else None

    # Mark success by metric availability.
    df.loc[df["segm_mAP"].notna(), "status"] = "success"

    return df


def add_recovery(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for idx, row in df.iterrows():
        if row.get("group") != "depth_fill":
            continue
        if str(row.get("fill_method")) == "none":
            continue

        fold = row.get("fold")
        hole_type = row.get("hole_type")
        sev = row.get("severity")
        fill_exp = row.get("exp_name")
        none_exp = f"depth_fill_{hole_type}_s{int(sev)}_none"

        clean_rows = df[(df["fold"] == fold) & (df["exp_name"] == "clean_mirror")]
        hole_rows = df[(df["fold"] == fold) & (df["exp_name"] == none_exp)]
        if clean_rows.empty or hole_rows.empty:
            continue

        clean = clean_rows.iloc[0]
        hole = hole_rows.iloc[0]

        for metric, alias in [
            ("segm_mAP", "AP"),
            ("segm_mAP_50", "AP50"),
            ("segm_mAP_75", "AP75"),
        ]:
            clean_v = to_float_or_none(clean.get(metric))
            hole_v = to_float_or_none(hole.get(metric))
            fill_v = to_float_or_none(row.get(metric))
            if clean_v is None or hole_v is None or fill_v is None:
                continue
            denom = clean_v - hole_v
            rec = 0.0 if denom == 0 else (fill_v - hole_v) / denom
            df.at[idx, f"recovery_{alias}"] = rec

    return df


# ============================================================
# Summaries and judgement
# ============================================================

def summarize_experiments(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "segm_mAP", "segm_mAP_50", "segm_mAP_75",
        "bbox_mAP", "AP_drop", "AP_retention",
        "AP50_drop", "AP50_retention", "AP75_drop", "AP75_retention",
        "recovery_AP", "recovery_AP50", "recovery_AP75",
    ]

    for exp_name, g in df.groupby("exp_name"):
        meta = infer_metadata(exp_name)
        row: Dict[str, Any] = dict(meta)
        row["n_folds"] = int(g["fold"].nunique())
        row["n_success"] = int(g["segm_mAP"].notna().sum())

        for m in metrics:
            vals = [to_float_or_none(x) for x in g[m].tolist() if to_float_or_none(x) is not None]
            row[f"{m}_mean"] = float(np.mean(vals)) if vals else None
            row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0 if len(vals) == 1 else None
            row[f"{m}_min"] = float(np.min(vals)) if vals else None
            row[f"{m}_max"] = float(np.max(vals)) if vals else None

        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["group", "series", "severity", "exp_name"], na_position="first")
    return out


def classify_retention(ret: Optional[float]) -> str:
    if ret is None:
        return "无有效结果"
    if ret >= 0.98:
        return "非常稳定"
    if ret >= 0.95:
        return "稳定"
    if ret >= 0.90:
        return "轻中度下降"
    if ret >= 0.80:
        return "明显下降"
    return "严重下降"


def summarize_series(exp_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = exp_summary[exp_summary["group"] != "clean"].copy()
    for series, g in valid.groupby("series"):
        g = g.sort_values("severity")
        vals = [to_float_or_none(x) for x in g["AP_retention_mean"].tolist()]
        drops = [to_float_or_none(x) for x in g["AP_drop_mean"].tolist()]
        aps = [to_float_or_none(x) for x in g["segm_mAP_mean"].tolist()]

        vals2 = [v for v in vals if v is not None]
        drops2 = [v for v in drops if v is not None]
        aps2 = [v for v in aps if v is not None]

        s1 = None
        s5 = None
        if 1 in set(g["severity"].dropna().astype(int).tolist()):
            s1_row = g[g["severity"] == 1]
            if not s1_row.empty:
                s1 = to_float_or_none(s1_row.iloc[0]["AP_retention_mean"])
        if 5 in set(g["severity"].dropna().astype(int).tolist()):
            s5_row = g[g["severity"] == 5]
            if not s5_row.empty:
                s5 = to_float_or_none(s5_row.iloc[0]["AP_retention_mean"])

        worst_ret = min(vals2) if vals2 else None
        mean_ret = float(np.mean(vals2)) if vals2 else None
        max_drop = max(drops2) if drops2 else None
        degradation_span = None
        if s1 is not None and s5 is not None:
            degradation_span = s1 - s5

        first = g.iloc[0]
        rows.append({
            "series": series,
            "series_label": first.get("series_label"),
            "group": first.get("group"),
            "family": first.get("family"),
            "n_experiments": len(g),
            "mean_AP": float(np.mean(aps2)) if aps2 else None,
            "mean_AP_retention": mean_ret,
            "worst_AP_retention": worst_ret,
            "max_AP_drop": max_drop,
            "S1_retention": s1,
            "S5_retention": s5,
            "S1_to_S5_retention_drop": degradation_span,
            "judgement": classify_retention(worst_ret),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["group", "series"])
    return out


def find_failed_or_missing(df: pd.DataFrame, folds: List[int]) -> pd.DataFrame:
    expected = expected_experiment_names()
    rows = []

    existing = set()
    for _, r in df.iterrows():
        fold = to_float_or_none(r.get("fold"))
        if fold is None:
            continue
        existing.add((int(fold), r["exp_name"]))

        if r["exp_name"] != "clean_mirror" and to_float_or_none(r.get("segm_mAP")) is None:
            rows.append({
                "fold": int(fold),
                "exp_name": r["exp_name"],
                "issue": "no_valid_AP",
                "status": r.get("status"),
                "error": r.get("error"),
                "log_file": r.get("log_file"),
            })

    for fold in folds:
        for exp_name in expected:
            if (fold, exp_name) not in existing:
                rows.append({
                    "fold": fold,
                    "exp_name": exp_name,
                    "issue": "missing",
                    "status": "",
                    "error": "",
                    "log_file": "",
                })

    return pd.DataFrame(rows).sort_values(["fold", "exp_name"]) if rows else pd.DataFrame()


def write_judgement(series_summary: pd.DataFrame, failed_df: pd.DataFrame, out_path: str) -> None:
    ensure_dir(osp.dirname(out_path))
    lines: List[str] = []
    lines.append("=== Five-fold RGB-D Robustness Judgement ===")
    lines.append("")

    if failed_df.empty:
        lines.append("Completeness: all expected experiments have valid or discoverable records.")
    else:
        lines.append(f"Completeness warning: {len(failed_df)} failed or missing records were detected.")
    lines.append("")

    if not series_summary.empty:
        lines.append("Series-level judgement:")
        for _, r in series_summary.iterrows():
            lines.append(
                f"- {r['series_label']}: "
                f"mean retention={fmt4(r['mean_AP_retention'])}, "
                f"worst retention={fmt4(r['worst_AP_retention'])}, "
                f"max AP drop={fmt4(r['max_AP_drop'])}, "
                f"judgement={r['judgement']}"
            )

        lines.append("")
        worst = series_summary.sort_values("worst_AP_retention", na_position="last").head(5)
        lines.append("Most sensitive series:")
        for _, r in worst.iterrows():
            lines.append(
                f"- {r['series_label']}: worst retention={fmt4(r['worst_AP_retention'])}, "
                f"S1->S5 drop={fmt4(r['S1_to_S5_retention_drop'])}"
            )

        lines.append("")
        stable = series_summary.sort_values("worst_AP_retention", ascending=False, na_position="last").head(5)
        lines.append("Most stable series:")
        for _, r in stable.iterrows():
            lines.append(
                f"- {r['series_label']}: worst retention={fmt4(r['worst_AP_retention'])}, "
                f"mean retention={fmt4(r['mean_AP_retention'])}"
            )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# Plotting
# ============================================================

def plot_series_curves(exp_summary: pd.DataFrame, out_dir: str, metric_mean: str, metric_std: str, ylabel: str, suffix: str) -> List[str]:
    ensure_dir(out_dir)
    paths: List[str] = []

    valid = exp_summary[(exp_summary["group"] != "clean") & exp_summary["severity"].notna()].copy()
    for series, g in valid.groupby("series"):
        g = g.sort_values("severity")
        x = [int(v) for v in g["severity"].tolist()]
        y = [to_float_or_none(v) for v in g[metric_mean].tolist()]
        yerr = [to_float_or_none(v) for v in g[metric_std].tolist()]

        if not any(v is not None for v in y):
            continue

        yy = [np.nan if v is None else v for v in y]
        ee = [0.0 if v is None else v for v in yerr]

        label = str(g.iloc[0].get("series_label", series))
        plt.figure(figsize=(7, 4.5))
        plt.errorbar(x, yy, yerr=ee, marker="o", capsize=3)
        plt.xticks([1, 2, 3, 4, 5])
        plt.xlabel("Severity level")
        plt.ylabel(ylabel)
        plt.title(label)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        path = osp.join(out_dir, f"{series}_{suffix}.png")
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    return paths


def plot_group_retention(exp_summary: pd.DataFrame, out_dir: str) -> List[str]:
    ensure_dir(out_dir)
    paths = []
    valid = exp_summary[(exp_summary["group"] != "clean") & exp_summary["severity"].notna()].copy()

    for group, g0 in valid.groupby("group"):
        plt.figure(figsize=(9, 5.5))
        for series, g in g0.groupby("series"):
            g = g.sort_values("severity")
            x = [int(v) for v in g["severity"].tolist()]
            y = [to_float_or_none(v) for v in g["AP_retention_mean"].tolist()]
            yy = [np.nan if v is None else v for v in y]
            label = str(g.iloc[0].get("series_label", series))
            plt.plot(x, yy, marker="o", label=label)

        plt.xticks([1, 2, 3, 4, 5])
        plt.xlabel("Severity level")
        plt.ylabel("AP retention")
        plt.title(f"{group}: AP retention curves")
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()

        path = osp.join(out_dir, f"group_{group}_retention_curves.png")
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    return paths


def plot_recovery_curves(exp_summary: pd.DataFrame, out_dir: str) -> List[str]:
    ensure_dir(out_dir)
    paths = []
    g0 = exp_summary[(exp_summary["group"] == "depth_fill") & (exp_summary["fill_method"] != "none")].copy()
    if g0.empty:
        return paths

    for hole_type, g1 in g0.groupby("hole_type"):
        plt.figure(figsize=(8, 5))
        for fill_method, g in g1.groupby("fill_method"):
            g = g.sort_values("severity")
            x = [int(v) for v in g["severity"].tolist()]
            y = [to_float_or_none(v) for v in g["recovery_AP_mean"].tolist()]
            yy = [np.nan if v is None else v for v in y]
            plt.plot(x, yy, marker="o", label=fill_method)

        plt.xticks([1, 2, 3, 4, 5])
        plt.xlabel("Severity level")
        plt.ylabel("Recovery on AP")
        plt.title(f"Depth fill recovery: {hole_type}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        path = osp.join(out_dir, f"depth_fill_{hole_type}_recovery.png")
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    return paths


# ============================================================
# Visual contact sheets
# ============================================================

def list_sanity_images(datasets_root: str, exp_name: str, max_images: int) -> List[str]:
    sanity_dir = Path(datasets_root) / exp_name / "sanity"
    if not sanity_dir.is_dir():
        return []
    imgs = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        imgs.extend([str(p) for p in sorted(sanity_dir.glob(ext))])
    return imgs[:max_images]


def open_image_safe(path: str):
    try:
        from PIL import Image
        img = Image.open(path)
        return np.array(img.convert("RGB"))
    except Exception:
        try:
            import matplotlib.image as mpimg
            return mpimg.imread(path)
        except Exception:
            return None


def make_contact_sheet_for_series(
    datasets_root: str,
    series: str,
    exp_names: List[str],
    out_dir: str,
    max_images_per_exp: int = 1,
) -> Optional[str]:
    ensure_dir(out_dir)

    selected = []
    # Prefer S1/S3/S5 if present.
    for target in ("_s1", "_s3", "_s5"):
        matched = [e for e in exp_names if target in e]
        if matched:
            selected.append(matched[0])
    if not selected:
        selected = exp_names[:3]

    panels = []
    titles = []
    for exp_name in selected:
        imgs = list_sanity_images(datasets_root, exp_name, max_images_per_exp)
        for img_path in imgs:
            arr = open_image_safe(img_path)
            if arr is not None:
                panels.append(arr)
                titles.append(exp_name)

    if not panels:
        return None

    n = len(panels)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))

    plt.figure(figsize=(5 * cols, 4 * rows))
    for i, arr in enumerate(panels):
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(arr)
        ax.set_title(titles[i], fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    path = osp.join(out_dir, f"{series}_sanity_contact.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def make_visual_contact_sheets(exp_summary: pd.DataFrame, datasets_root: str, out_dir: str, max_images_per_exp: int) -> List[str]:
    ensure_dir(out_dir)
    paths = []

    valid = exp_summary[(exp_summary["group"] != "clean") & exp_summary["series"].notna()].copy()
    for series, g in valid.groupby("series"):
        exp_names = g.sort_values("severity")["exp_name"].tolist()
        p = make_contact_sheet_for_series(
            datasets_root=datasets_root,
            series=series,
            exp_names=exp_names,
            out_dir=out_dir,
            max_images_per_exp=max_images_per_exp,
        )
        if p:
            paths.append(p)
    return paths


# ============================================================
# HTML report
# ============================================================

def relpath(path: str, base: str) -> str:
    return osp.relpath(path, base).replace("\\", "/")


def df_to_html_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "<p>None.</p>"
    show = df.head(max_rows).copy()
    return show.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4f}")


def write_html_report(
    out_dir: str,
    df: pd.DataFrame,
    exp_summary: pd.DataFrame,
    series_summary: pd.DataFrame,
    failed_df: pd.DataFrame,
    figure_paths: List[str],
    visual_paths: List[str],
) -> str:
    ensure_dir(out_dir)
    html_path = osp.join(out_dir, "index.html")

    figs_html = "\n".join(
        f'<div class="card"><img src="{relpath(p, out_dir)}" /></div>'
        for p in figure_paths
    )
    visuals_html = "\n".join(
        f'<div class="card"><img src="{relpath(p, out_dir)}" /></div>'
        for p in visual_paths
    )

    clean_rows = df[df["exp_name"] == "clean_mirror"]
    clean_html = df_to_html_table(clean_rows[[
        "fold", "exp_name", "segm_mAP", "segm_mAP_50", "segm_mAP_75",
        "bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "status"
    ]] if not clean_rows.empty else clean_rows)

    series_cols = [
        "series_label", "group", "n_experiments",
        "mean_AP", "mean_AP_retention", "worst_AP_retention",
        "max_AP_drop", "S1_retention", "S5_retention",
        "S1_to_S5_retention_drop", "judgement"
    ]
    series_html = df_to_html_table(series_summary[series_cols] if not series_summary.empty else series_summary)

    failed_html = df_to_html_table(failed_df, max_rows=500)

    exp_cols = [
        "exp_name", "group", "series_label", "severity",
        "segm_mAP_mean", "segm_mAP_std",
        "AP_drop_mean", "AP_retention_mean",
        "AP50_retention_mean", "AP75_retention_mean",
        "recovery_AP_mean", "n_success", "n_folds"
    ]
    exp_html = df_to_html_table(exp_summary[exp_cols] if not exp_summary.empty else exp_summary, max_rows=500)

    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Five-fold Robustness Report</title>
<style>
body {{
  font-family: Arial, "Microsoft YaHei", sans-serif;
  margin: 24px;
  line-height: 1.45;
}}
h1, h2 {{
  border-bottom: 1px solid #ddd;
  padding-bottom: 6px;
}}
table {{
  border-collapse: collapse;
  font-size: 13px;
  margin: 12px 0;
}}
th, td {{
  border: 1px solid #ccc;
  padding: 4px 8px;
}}
th {{
  background: #f3f3f3;
}}
.card {{
  margin: 12px 0 24px 0;
}}
.card img {{
  max-width: 100%;
  border: 1px solid #ddd;
}}
.note {{
  background: #f7f7f7;
  border-left: 4px solid #888;
  padding: 10px 12px;
}}
</style>
</head>
<body>
<h1>Five-fold RGB-D Robustness Report</h1>

<div class="note">
<p>本报告读取五折鲁棒性实验结果，统计 AP、AP drop、AP retention、depth fill recovery，并生成下降曲线与 sanity 可视化拼图。</p>
<p>判定规则：retention ≥ 0.98 为非常稳定；≥ 0.95 为稳定；≥ 0.90 为轻中度下降；≥ 0.80 为明显下降；低于 0.80 为严重下降。</p>
</div>

<h2>1. Clean baseline by fold</h2>
{clean_html}

<h2>2. Series-level judgement</h2>
{series_html}

<h2>3. Failed or missing records</h2>
{failed_html}

<h2>4. Experiment summary</h2>
{exp_html}

<h2>5. Degradation / response curves</h2>
{figs_html}

<h2>6. Visual sanity contact sheets</h2>
{visuals_html}

</body>
</html>
"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(content)
    return html_path


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Summarize five-fold robustness results")
    p.add_argument("--results-root", default="robustness_external/results_5fold")
    p.add_argument("--datasets-root", default="robustness_external/datasets")
    p.add_argument("--out-dir", default="robustness_external/reports_5fold")
    p.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--max-sanity-per-exp", type=int, default=1)
    p.add_argument("--no-visuals", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = ensure_dir(args.out_dir)
    fig_dir = ensure_dir(osp.join(out_dir, "figures"))
    visual_dir = ensure_dir(osp.join(out_dir, "visuals"))

    print("=" * 80)
    print("Five-fold Robustness Summary")
    print(f"results_root : {args.results_root}")
    print(f"datasets_root: {args.datasets_root}")
    print(f"out_dir      : {args.out_dir}")
    print("=" * 80)

    df = load_rows_from_all_csv(args.results_root)
    if df.empty:
        print("[INFO] all_results.csv not found or empty. Scanning fold directories...")
        df = load_rows_by_scanning(args.results_root)

    if df.empty:
        raise RuntimeError(f"No results found under {args.results_root}")

    df = normalize_dataframe(df, args.results_root)
    df = add_recovery(df)

    exp_summary = summarize_experiments(df)
    series_summary = summarize_series(exp_summary)
    failed_df = find_failed_or_missing(df, args.folds)

    # Save tables.
    df.to_csv(osp.join(out_dir, "all_results_cleaned.csv"), index=False, encoding="utf-8-sig")
    exp_summary.to_csv(osp.join(out_dir, "experiment_summary.csv"), index=False, encoding="utf-8-sig")
    series_summary.to_csv(osp.join(out_dir, "series_summary.csv"), index=False, encoding="utf-8-sig")
    failed_df.to_csv(osp.join(out_dir, "failed_or_missing.csv"), index=False, encoding="utf-8-sig")

    safe_write_json(
        {
            "n_rows": int(len(df)),
            "n_experiments": int(exp_summary["exp_name"].nunique()) if not exp_summary.empty else 0,
            "n_series": int(series_summary["series"].nunique()) if not series_summary.empty else 0,
            "failed_or_missing": int(len(failed_df)),
        },
        osp.join(out_dir, "report_meta.json"),
    )

    write_judgement(series_summary, failed_df, osp.join(out_dir, "judgement.txt"))

    # Figures.
    figure_paths: List[str] = []
    figure_paths += plot_group_retention(exp_summary, fig_dir)
    figure_paths += plot_series_curves(exp_summary, fig_dir, "segm_mAP_mean", "segm_mAP_std", "Mask AP", "mask_ap")
    figure_paths += plot_series_curves(exp_summary, fig_dir, "AP_drop_mean", "AP_drop_std", "AP drop", "ap_drop")
    figure_paths += plot_series_curves(exp_summary, fig_dir, "AP_retention_mean", "AP_retention_std", "AP retention", "ap_retention")
    figure_paths += plot_recovery_curves(exp_summary, fig_dir)

    # Visuals.
    visual_paths: List[str] = []
    if not args.no_visuals:
        visual_paths = make_visual_contact_sheets(
            exp_summary=exp_summary,
            datasets_root=args.datasets_root,
            out_dir=visual_dir,
            max_images_per_exp=args.max_sanity_per_exp,
        )

    html_path = write_html_report(
        out_dir=out_dir,
        df=df,
        exp_summary=exp_summary,
        series_summary=series_summary,
        failed_df=failed_df,
        figure_paths=figure_paths,
        visual_paths=visual_paths,
    )

    print("\nDone.")
    print(f"HTML report      : {html_path}")
    print(f"Cleaned results  : {osp.join(out_dir, 'all_results_cleaned.csv')}")
    print(f"Experiment table : {osp.join(out_dir, 'experiment_summary.csv')}")
    print(f"Series table     : {osp.join(out_dir, 'series_summary.csv')}")
    print(f"Judgement        : {osp.join(out_dir, 'judgement.txt')}")
    print(f"Failed/missing   : {osp.join(out_dir, 'failed_or_missing.csv')}")
    print(f"Figures          : {fig_dir}")
    print(f"Visuals          : {visual_dir}")


if __name__ == "__main__":
    main()