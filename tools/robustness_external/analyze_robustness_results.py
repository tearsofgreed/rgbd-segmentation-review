#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline robustness result analyzer.

This script only reads saved robustness results and generates:
1. CSV tables
2. AP / retention / recovery curves
3. Markdown report

Expected result structure:
robustness_external/results/{exp_name}/metrics.json
robustness_external/results/{exp_name}/summary.txt
robustness_external/results/{exp_name}/test.log

Optional metadata:
robustness_external/datasets/{exp_name}/metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import os.path as osp
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RESULT_KEYS = [
    "segm_mAP",
    "segm_mAP_50",
    "segm_mAP_75",
    "segm_mAP_s",
    "segm_mAP_m",
    "segm_mAP_l",
    "bbox_mAP",
    "bbox_mAP_50",
    "bbox_mAP_75",
    "bbox_mAP_s",
    "bbox_mAP_m",
    "bbox_mAP_l",
]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_json_safe(path: str) -> Dict[str, Any]:
    if not osp.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return None
        return float(x)
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


def parse_metrics_from_log(log_file: str) -> Dict[str, float]:
    """Fallback parser for MMDetection logs."""
    if not osp.isfile(log_file):
        return {}

    with open(log_file, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    metrics: Dict[str, float] = {}

    # Example:
    # coco/segm_mAP: 0.8140
    pat = re.compile(r"(coco/[A-Za-z0-9_]+)\s*[:=]\s*([0-9.]+)")
    for m in pat.finditer(content):
        key = m.group(1)
        try:
            metrics[key] = float(m.group(2))
        except ValueError:
            pass

    # Example:
    # OrderedDict([('coco/segm_mAP', 0.817), ...])
    od_pat = re.compile(r"\('([^']+)',\s*([0-9.]+)\)")
    for m in od_pat.finditer(content):
        key = m.group(1)
        if key.startswith("coco/"):
            try:
                metrics[key] = float(m.group(2))
            except ValueError:
                pass

    return metrics


def parse_summary_txt(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not osp.isfile(path):
        return data

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            data[k.strip()] = v.strip()
    return data


def extract_severity(exp_name: str) -> Optional[int]:
    m = re.search(r"(?:^|_)s(\d+)(?:_|$)", exp_name)
    if not m:
        return None
    return int(m.group(1))


def infer_group_and_names(exp_name: str) -> Dict[str, Any]:
    """Infer group / corruption / curve_name from experiment name."""
    severity = extract_severity(exp_name)

    info: Dict[str, Any] = {
        "exp_name": exp_name,
        "group": "unknown",
        "corruption": exp_name,
        "curve_name": exp_name,
        "severity": severity,
        "severity_name": f"s{severity}" if severity is not None else "",
        "variant": "",
        "hole_type": "",
        "fill_method": "none",
    }

    if exp_name in ("clean_mirror", "original_clean"):
        info.update({
            "group": "clean",
            "corruption": exp_name,
            "curve_name": exp_name,
        })
        return info

    # RGB
    if exp_name.startswith("rgb_brightness_"):
        info.update({
            "group": "rgb",
            "corruption": "rgb_brightness",
            "curve_name": "rgb_brightness",
        })
        return info

    if exp_name.startswith("rgb_white_balance_warm_"):
        info.update({
            "group": "rgb",
            "corruption": "rgb_white_balance",
            "curve_name": "rgb_white_balance_warm",
            "variant": "warm",
        })
        return info

    if exp_name.startswith("rgb_white_balance_cool_"):
        info.update({
            "group": "rgb",
            "corruption": "rgb_white_balance",
            "curve_name": "rgb_white_balance_cool",
            "variant": "cool",
        })
        return info

    if exp_name.startswith("rgb_local_shadow_"):
        info.update({
            "group": "rgb",
            "corruption": "rgb_local_shadow",
            "curve_name": "rgb_local_shadow",
        })
        return info

    if exp_name.startswith("rgb_specular_highlight_"):
        info.update({
            "group": "rgb",
            "corruption": "rgb_specular_highlight",
            "curve_name": "rgb_specular_highlight",
        })
        return info

    # Depth holes
    for corr in ["depth_random_holes", "depth_block_holes", "depth_edge_holes"]:
        if exp_name.startswith(corr + "_"):
            info.update({
                "group": "depth_holes",
                "corruption": corr,
                "curve_name": corr,
            })
            return info

    # Depth fill:
    # depth_fill_block_s1_none
    # depth_fill_edge_s3_inpaint_telea
    if exp_name.startswith("depth_fill_"):
        parts = exp_name.split("_")
        hole_type = parts[2] if len(parts) > 2 else ""
        fill_method = "none"
        if exp_name.endswith("_inpaint_telea"):
            fill_method = "inpaint_telea"
        elif exp_name.endswith("_median"):
            fill_method = "median"
        elif exp_name.endswith("_none"):
            fill_method = "none"

        curve_name = f"depth_fill_{hole_type}_{fill_method}"

        info.update({
            "group": "depth_fill",
            "corruption": f"depth_fill_{hole_type}",
            "curve_name": curve_name,
            "hole_type": hole_type,
            "fill_method": fill_method,
        })
        return info

    return info


def read_experiment(
    exp_dir: str,
    datasets_root: str,
) -> Dict[str, Any]:
    exp_name = osp.basename(exp_dir)
    row = infer_group_and_names(exp_name)

    metrics_path = osp.join(exp_dir, "metrics.json")
    summary_path = osp.join(exp_dir, "summary.txt")
    log_path = osp.join(exp_dir, "test.log")
    metadata_path = osp.join(datasets_root, exp_name, "metadata.json")

    metrics = load_json_safe(metrics_path)
    if to_float_or_none(metrics.get("coco/segm_mAP")) is None:
        metrics = parse_metrics_from_log(log_path)

    summary = parse_summary_txt(summary_path)
    metadata = load_json_safe(metadata_path)

    row["metrics_json"] = metrics_path if osp.isfile(metrics_path) else ""
    row["summary_txt"] = summary_path if osp.isfile(summary_path) else ""
    row["test_log"] = log_path if osp.isfile(log_path) else ""
    row["metadata_json"] = metadata_path if osp.isfile(metadata_path) else ""

    # Metrics
    for k in RESULT_KEYS:
        row[k] = to_float_or_none(metrics.get(f"coco/{k}") or metrics.get(k))

    # Summary fields
    for k in [
        "checkpoint",
        "data_root",
        "ann_file",
        "log_file",
        "status",
        "error",
    ]:
        row[k] = summary.get(k, "")

    # Metadata fields, if available
    for k in [
        "brightness_factor",
        "r_gain",
        "g_gain",
        "b_gain",
        "shadow_factor",
        "shadow_area_ratio",
        "shadow_num_blobs",
        "highlight_value",
        "highlight_area_ratio",
        "highlight_num_blobs",
        "hole_ratio",
        "block_num",
        "block_min_ratio",
        "block_max_ratio",
        "edge_frac",
        "edge_hole_prob",
        "seed",
        "created_time",
    ]:
        row[k] = metadata.get(k, "")

    # Status
    if to_float_or_none(row.get("segm_mAP")) is not None:
        row["metric_status"] = "success"
    else:
        row["metric_status"] = "no_metric"

    return row


def load_all_results(results_root: str, datasets_root: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if not osp.isdir(results_root):
        return pd.DataFrame()

    for sub in sorted(os.listdir(results_root)):
        exp_dir = osp.join(results_root, sub)
        if not osp.isdir(exp_dir):
            continue
        rows.append(read_experiment(exp_dir, datasets_root))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Stable sorting
    sort_cols = [c for c in ["group", "curve_name", "severity", "fill_method", "exp_name"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)
    return df


def get_clean_metrics(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Prefer clean_mirror, then original_clean, then fixed fallback."""
    for name in ["clean_mirror", "original_clean"]:
        sub = df[df["exp_name"] == name]
        if not sub.empty:
            r = sub.iloc[0]
            clean_ap = to_float_or_none(r.get("segm_mAP"))
            clean_ap50 = to_float_or_none(r.get("segm_mAP_50"))
            clean_ap75 = to_float_or_none(r.get("segm_mAP_75"))
            if clean_ap is not None:
                return {
                    "segm_mAP": clean_ap,
                    "segm_mAP_50": clean_ap50,
                    "segm_mAP_75": clean_ap75,
                }

    return {
        "segm_mAP": 0.817,
        "segm_mAP_50": 0.924,
        "segm_mAP_75": 0.857,
    }


def add_drop_retention(df: pd.DataFrame, clean: Dict[str, Optional[float]]) -> pd.DataFrame:
    df = df.copy()

    aliases = [
        ("segm_mAP", "AP"),
        ("segm_mAP_50", "AP50"),
        ("segm_mAP_75", "AP75"),
    ]

    for metric_name, alias in aliases:
        clean_val = to_float_or_none(clean.get(metric_name))
        drops = []
        rets = []

        for _, r in df.iterrows():
            v = to_float_or_none(r.get(metric_name))
            if clean_val is not None and clean_val != 0 and v is not None:
                drops.append(clean_val - v)
                rets.append(v / clean_val)
            else:
                drops.append(None)
                rets.append(None)

        df[f"{alias}_drop"] = drops
        df[f"{alias}_retention"] = rets

    df["clean_segm_mAP"] = clean.get("segm_mAP")
    df["clean_segm_mAP_50"] = clean.get("segm_mAP_50")
    df["clean_segm_mAP_75"] = clean.get("segm_mAP_75")

    return df


def add_recovery(df: pd.DataFrame, clean: Dict[str, Optional[float]]) -> pd.DataFrame:
    df = df.copy()
    for col in ["recovery_AP", "recovery_AP50", "recovery_AP75"]:
        df[col] = None

    metrics = [
        ("segm_mAP", "recovery_AP"),
        ("segm_mAP_50", "recovery_AP50"),
        ("segm_mAP_75", "recovery_AP75"),
    ]

    fill_df = df[df["group"] == "depth_fill"].copy()
    if fill_df.empty:
        return df

    for hole_type in sorted(fill_df["hole_type"].dropna().unique()):
        for severity in sorted(fill_df["severity"].dropna().unique()):
            base = fill_df[
                (fill_df["hole_type"] == hole_type)
                & (fill_df["severity"] == severity)
                & (fill_df["fill_method"] == "none")
            ]
            if base.empty:
                continue
            base_row = base.iloc[0]

            for idx, r in df[
                (df["group"] == "depth_fill")
                & (df["hole_type"] == hole_type)
                & (df["severity"] == severity)
                & (df["fill_method"] != "none")
            ].iterrows():
                for metric_name, out_col in metrics:
                    clean_val = to_float_or_none(clean.get(metric_name))
                    hole_val = to_float_or_none(base_row.get(metric_name))
                    fill_val = to_float_or_none(r.get(metric_name))

                    if clean_val is None or hole_val is None or fill_val is None:
                        rec = None
                    else:
                        denom = clean_val - hole_val
                        rec = 0.0 if denom == 0 else (fill_val - hole_val) / denom

                    df.at[idx, out_col] = rec

    return df


def save_tables(df: pd.DataFrame, out_dir: str) -> None:
    table_dir = ensure_dir(osp.join(out_dir, "tables"))

    df.to_csv(osp.join(table_dir, "robustness_all_results.csv"), index=False, encoding="utf-8-sig")

    for group in ["rgb", "depth_holes", "depth_fill", "clean", "unknown"]:
        sub = df[df["group"] == group]
        if not sub.empty:
            sub.to_csv(
                osp.join(table_dir, f"{group}_results.csv"),
                index=False,
                encoding="utf-8-sig",
            )

    # Compact main table
    keep_cols = [
        "exp_name", "group", "corruption", "curve_name", "severity",
        "variant", "hole_type", "fill_method",
        "segm_mAP", "segm_mAP_50", "segm_mAP_75",
        "AP_drop", "AP_retention",
        "AP50_drop", "AP50_retention",
        "AP75_drop", "AP75_retention",
        "recovery_AP", "recovery_AP50", "recovery_AP75",
        "metric_status",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(
        osp.join(table_dir, "robustness_main_table.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # Group summary
    valid = df[(df["metric_status"] == "success") & (df["group"].isin(["rgb", "depth_holes", "depth_fill"]))].copy()
    if not valid.empty:
        summary = valid.groupby(["group", "curve_name"], dropna=False).agg(
            n=("exp_name", "count"),
            AP_mean=("segm_mAP", "mean"),
            AP_min=("segm_mAP", "min"),
            AP_max=("segm_mAP", "max"),
            AP_retention_mean=("AP_retention", "mean"),
            AP_retention_min=("AP_retention", "min"),
            AP_drop_mean=("AP_drop", "mean"),
            AP_drop_max=("AP_drop", "max"),
        ).reset_index()
        summary.to_csv(
            osp.join(table_dir, "robustness_group_summary.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    # Pivot tables for curves
    for group in ["rgb", "depth_holes"]:
        sub = df[(df["group"] == group) & (df["metric_status"] == "success")]
        if not sub.empty:
            pivot_ap = sub.pivot_table(
                index="severity",
                columns="curve_name",
                values="segm_mAP",
                aggfunc="mean",
            )
            pivot_ret = sub.pivot_table(
                index="severity",
                columns="curve_name",
                values="AP_retention",
                aggfunc="mean",
            )
            pivot_ap.to_csv(osp.join(table_dir, f"{group}_ap_pivot.csv"), encoding="utf-8-sig")
            pivot_ret.to_csv(osp.join(table_dir, f"{group}_retention_pivot.csv"), encoding="utf-8-sig")

    sub = df[(df["group"] == "depth_fill") & (df["metric_status"] == "success")]
    if not sub.empty:
        pivot_ap = sub.pivot_table(
            index="severity",
            columns="curve_name",
            values="segm_mAP",
            aggfunc="mean",
        )
        pivot_rec = sub.pivot_table(
            index="severity",
            columns="curve_name",
            values="recovery_AP",
            aggfunc="mean",
        )
        pivot_ap.to_csv(osp.join(table_dir, "depth_fill_ap_pivot.csv"), encoding="utf-8-sig")
        pivot_rec.to_csv(osp.join(table_dir, "depth_fill_recovery_pivot.csv"), encoding="utf-8-sig")


def plot_metric_curves(
    df: pd.DataFrame,
    group: str,
    y_col: str,
    title: str,
    ylabel: str,
    out_path: str,
    exclude_fill_none: bool = False,
) -> None:
    sub = df[(df["group"] == group) & (df["metric_status"] == "success")].copy()

    if exclude_fill_none and "fill_method" in sub.columns:
        sub = sub[sub["fill_method"] != "none"]

    if sub.empty:
        return

    plt.figure(figsize=(9, 6))

    has_line = False
    for curve_name in sorted(sub["curve_name"].dropna().unique()):
        cdf = sub[sub["curve_name"] == curve_name].copy()
        cdf = cdf.sort_values("severity")
        x = [to_float_or_none(v) for v in cdf["severity"].tolist()]
        y = [to_float_or_none(v) for v in cdf[y_col].tolist()]
        pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
        if not pairs:
            continue
        xs, ys = zip(*pairs)
        plt.plot(xs, ys, marker="o", label=curve_name)
        has_line = True

    if not has_line:
        plt.close()
        return

    plt.title(title)
    plt.xlabel("Severity")
    plt.ylabel(ylabel)
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_summary_bar(df: pd.DataFrame, out_path: str) -> None:
    valid = df[
        (df["metric_status"] == "success")
        & (df["group"].isin(["rgb", "depth_holes", "depth_fill"]))
    ].copy()

    if valid.empty:
        return

    summary = valid.groupby("curve_name", dropna=False).agg(
        AP_retention_mean=("AP_retention", "mean"),
        AP_retention_min=("AP_retention", "min"),
    ).reset_index()

    summary = summary.sort_values("AP_retention_mean", ascending=True)

    plt.figure(figsize=(10, max(5, 0.35 * len(summary))))
    plt.barh(summary["curve_name"], summary["AP_retention_mean"])
    plt.xlabel("Mean AP retention")
    plt.ylabel("Corruption")
    plt.title("Mean AP Retention by Corruption")
    plt.grid(True, axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_figures(df: pd.DataFrame, out_dir: str) -> None:
    fig_dir = ensure_dir(osp.join(out_dir, "figures"))

    plot_metric_curves(
        df, "rgb", "segm_mAP",
        "RGB Robustness: AP Curves",
        "segm mAP",
        osp.join(fig_dir, "rgb_ap_curves.png"),
    )

    plot_metric_curves(
        df, "rgb", "AP_retention",
        "RGB Robustness: AP Retention Curves",
        "AP retention",
        osp.join(fig_dir, "rgb_retention_curves.png"),
    )

    plot_metric_curves(
        df, "depth_holes", "segm_mAP",
        "Depth Holes Robustness: AP Curves",
        "segm mAP",
        osp.join(fig_dir, "depth_holes_ap_curves.png"),
    )

    plot_metric_curves(
        df, "depth_holes", "AP_retention",
        "Depth Holes Robustness: AP Retention Curves",
        "AP retention",
        osp.join(fig_dir, "depth_holes_retention_curves.png"),
    )

    plot_metric_curves(
        df, "depth_fill", "segm_mAP",
        "Depth Fill: AP Curves",
        "segm mAP",
        osp.join(fig_dir, "depth_fill_ap_curves.png"),
    )

    plot_metric_curves(
        df, "depth_fill", "recovery_AP",
        "Depth Fill: Recovery Curves",
        "Recovery",
        osp.join(fig_dir, "depth_fill_recovery_curves.png"),
        exclude_fill_none=True,
    )

    plot_summary_bar(
        df,
        osp.join(fig_dir, "mean_ap_retention_bar.png"),
    )


def write_markdown_report(df: pd.DataFrame, clean: Dict[str, Optional[float]], out_dir: str) -> None:
    report_path = osp.join(out_dir, "robustness_report.md")
    table_dir = osp.join(out_dir, "tables")
    fig_dir = osp.join(out_dir, "figures")

    total = len(df)
    valid = df[df["metric_status"] == "success"]
    failed = df[df["metric_status"] != "success"]

    lines: List[str] = []
    lines.append("# Robustness Analysis Report")
    lines.append("")
    lines.append("## Clean baseline")
    lines.append("")
    lines.append(f"- Clean segm_mAP: {fmt4(clean.get('segm_mAP'))}")
    lines.append(f"- Clean segm_mAP_50: {fmt4(clean.get('segm_mAP_50'))}")
    lines.append(f"- Clean segm_mAP_75: {fmt4(clean.get('segm_mAP_75'))}")
    lines.append("")
    lines.append("## Result status")
    lines.append("")
    lines.append(f"- Total result directories: {total}")
    lines.append(f"- Valid metric results: {len(valid)}")
    lines.append(f"- Missing or invalid metrics: {len(failed)}")
    lines.append("")

    if not failed.empty:
        lines.append("### Missing or invalid metric directories")
        lines.append("")
        for _, r in failed.iterrows():
            lines.append(f"- {r.get('exp_name', '')}: {r.get('metric_status', '')}")
        lines.append("")

    lines.append("## Tables")
    lines.append("")
    lines.append(f"- `tables/robustness_all_results.csv`")
    lines.append(f"- `tables/robustness_main_table.csv`")
    lines.append(f"- `tables/robustness_group_summary.csv`")
    lines.append(f"- `tables/rgb_results.csv`")
    lines.append(f"- `tables/depth_holes_results.csv`")
    lines.append(f"- `tables/depth_fill_results.csv`")
    lines.append("")

    lines.append("## Figures")
    lines.append("")
    figure_files = [
        "rgb_ap_curves.png",
        "rgb_retention_curves.png",
        "depth_holes_ap_curves.png",
        "depth_holes_retention_curves.png",
        "depth_fill_ap_curves.png",
        "depth_fill_recovery_curves.png",
        "mean_ap_retention_bar.png",
    ]
    for fn in figure_files:
        p = osp.join(fig_dir, fn)
        if osp.isfile(p):
            lines.append(f"### {fn}")
            lines.append("")
            lines.append(f"![{fn}](figures/{fn})")
            lines.append("")

    # compact group summary preview
    group_summary_path = osp.join(table_dir, "robustness_group_summary.csv")
    if osp.isfile(group_summary_path):
        gs = pd.read_csv(group_summary_path)
        lines.append("## Group summary preview")
        lines.append("")
        preview_cols = [
            "group",
            "curve_name",
            "n",
            "AP_mean",
            "AP_min",
            "AP_retention_mean",
            "AP_retention_min",
        ]
        preview_cols = [c for c in preview_cols if c in gs.columns]
        lines.append(gs[preview_cols].to_markdown(index=False))
        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved robustness results.")
    parser.add_argument("--results-root", default="robustness_external/results")
    parser.add_argument("--datasets-root", default="robustness_external/datasets")
    parser.add_argument("--out-dir", default="robustness_external/analysis")
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    df = load_all_results(args.results_root, args.datasets_root)
    if df.empty:
        print(f"No results found under: {args.results_root}")
        return

    clean = get_clean_metrics(df)
    df = add_drop_retention(df, clean)
    df = add_recovery(df, clean)

    save_tables(df, args.out_dir)
    save_figures(df, args.out_dir)
    write_markdown_report(df, clean, args.out_dir)

    print("=" * 60)
    print("Robustness analysis done.")
    print(f"Results root: {args.results_root}")
    print(f"Output dir:   {args.out_dir}")
    print(f"Clean AP:     {fmt4(clean.get('segm_mAP'))}")
    print("=" * 60)

    valid = df[df["metric_status"] == "success"]
    print(f"Total dirs:   {len(df)}")
    print(f"Valid rows:   {len(valid)}")
    print(f"No metric:    {len(df) - len(valid)}")
    print("")
    print("Generated:")
    print(f"  {osp.join(args.out_dir, 'tables')}")
    print(f"  {osp.join(args.out_dir, 'figures')}")
    print(f"  {osp.join(args.out_dir, 'robustness_report.md')}")


if __name__ == "__main__":
    main()