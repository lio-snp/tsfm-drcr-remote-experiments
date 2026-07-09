#!/usr/bin/env python
"""Plot native TSFM/baseline comparisons for coverage-aware DRCR variants."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".omx" / "matplotlib"))

import matplotlib.pyplot as plt

SKILL_HELPER = Path.home() / ".codex" / "skills" / "scientific-figure-pro" / "scripts" / "scientific_figure_pro.py"
SUMMARY = ROOT / "results" / "aaai_stress" / "drcr_smooth_coverage_aware_gate_summary.csv"
WINDOWS = ROOT / "results" / "aaai_stress" / "drcr_smooth_coverage_aware_gate_windows.csv"
OUT_DIR = ROOT / "figures" / "aaai_stress"


def load_style_helper():
    spec = importlib.util.spec_from_file_location("scientific_figure_pro", SKILL_HELPER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load helper: {SKILL_HELPER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["candidate_id"], row["group"]): row for row in rows if row["phase"] == "test"}


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def group_native_coverage(rows: list[dict[str, str]]) -> dict[str, float]:
    selected = [
        row for row in rows
        if row["phase"] == "test" and row["candidate_id"] == "interval_cap_s1.10"
    ]

    def values_for(group: str) -> list[float]:
        if group == "overall":
            subset = selected
        elif group.startswith("role:"):
            role = group.split(":", 1)[1]
            subset = [row for row in selected if row["role"] == role]
        elif group.startswith("target_id:"):
            target = group.split(":", 1)[1]
            subset = [row for row in selected if row["target_id"] == target]
        elif group.startswith("evidence_tier:") and "|role:" in group:
            tier, role = group.replace("evidence_tier:", "").split("|role:")
            subset = [
                row for row in selected
                if row["evidence_tier"] == tier and row["role"] == role
            ]
        else:
            subset = []
        return [finite_float(row["model_coverage_q10_q90"], float("nan")) for row in subset]

    groups = [
        "overall",
        "role:failure_target",
        "role:positive_control",
        "role:stress_target",
        "target_id:finance_fred_stress",
        "evidence_tier:q9_fullgrid|role:failure_target",
    ]
    return {group: mean(values_for(group)) for group in groups}


def label(ax, prefix: str, title: str) -> None:
    ax.set_title(f"{prefix}. {title}", loc="left", fontweight="bold", pad=8, fontsize=13)


def annotate_values(ax, bars, fmt: str = "{:.2f}", dy: float = 0.04) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=7.5,
            rotation=0,
        )


def plot_wql_groups(ax, summaries: dict[tuple[str, str], dict[str, str]], palette: dict[str, str]) -> None:
    groups = [
        ("overall", "Overall"),
        ("role:failure_target", "Failure"),
        ("role:positive_control", "Positive"),
        ("role:stress_target", "Stress"),
        ("target_id:finance_fred_stress", "Finance"),
    ]
    series = [
        ("Native TSFM", None, "model_median_wql_rer", "#222222"),
        ("Full", "full_drcr_smooth", "repair_median_wql_rer", palette["red_strong"]),
        ("Point", "point_only_all", "repair_median_wql_rer", palette["neutral"]),
        ("Cap 1.10", "interval_cap_s1.10", "repair_median_wql_rer", palette["blue_main"]),
    ]
    x = np.arange(len(groups))
    width = 0.19
    for idx, (name, cid, metric, color) in enumerate(series):
        vals = []
        for group, _ in groups:
            row = summaries[("interval_cap_s1.10", group)] if cid is None else summaries[(cid, group)]
            vals.append(finite_float(row[metric]))
        bars = ax.bar(x + (idx - 1.5) * width, vals, width, label=name, color=color)
        if idx in {0, 3}:
            annotate_values(ax, bars, dy=0.035)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.6, label="Classical baseline (=1)")
    ax.set_ylabel("Median WQL-RER (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in groups])
    ax.set_ylim(0, 1.55)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.legend(loc="upper right", fontsize=8.5, ncols=2)
    label(ax, "A", "Native TSFM vs repair variants")


def plot_q9_failure(ax, summaries: dict[tuple[str, str], dict[str, str]], palette: dict[str, str]) -> None:
    group = "evidence_tier:q9_fullgrid|role:failure_target"
    bars_spec = [
        ("Native", summaries[("interval_cap_s1.10", group)]["model_median_wql_rer"], "#222222"),
        ("Full", summaries[("full_drcr_smooth", group)]["repair_median_wql_rer"], palette["red_strong"]),
        ("Point", summaries[("point_only_all", group)]["repair_median_wql_rer"], palette["neutral"]),
        ("Cap 1.10", summaries[("interval_cap_s1.10", group)]["repair_median_wql_rer"], palette["blue_main"]),
    ]
    x = np.arange(len(bars_spec))
    vals = [finite_float(value) for _, value, _ in bars_spec]
    colors = [color for _, _, color in bars_spec]
    bars = ax.bar(x, vals, color=colors, width=0.58)
    annotate_values(ax, bars, dy=0.07)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.6, label="Classical baseline (=1)")
    ax.set_ylabel("q9 failure median WQL-RER")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _, _ in bars_spec])
    ax.set_ylim(0, 3.8)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.legend(loc="upper right", fontsize=8.5)
    label(ax, "B", "Failure-side repair")


def plot_coverage(
    ax,
    summaries: dict[tuple[str, str], dict[str, str]],
    native_coverage: dict[str, float],
    palette: dict[str, str],
) -> None:
    groups = [
        ("overall", "Overall"),
        ("role:failure_target", "Failure"),
        ("role:positive_control", "Positive"),
        ("target_id:finance_fred_stress", "Finance"),
    ]
    series = [
        ("Native TSFM", None, "model_mean_coverage", "#222222"),
        ("Full", "full_drcr_smooth", "repair_mean_coverage", palette["red_strong"]),
        ("Point", "point_only_all", "repair_mean_coverage", palette["neutral"]),
        ("Cap 1.10", "interval_cap_s1.10", "repair_mean_coverage", palette["blue_main"]),
    ]
    x = np.arange(len(groups))
    width = 0.19
    for idx, (name, cid, metric, color) in enumerate(series):
        vals = []
        for group, _ in groups:
            if cid is None:
                vals.append(native_coverage[group])
            else:
                vals.append(finite_float(summaries[(cid, group)][metric]))
        bars = ax.bar(x + (idx - 1.5) * width, [100 * value for value in vals], width, label=name, color=color)
        if idx in {0, 3}:
            annotate_values(ax, bars, fmt="{:.0f}", dy=1.1)
    ax.axhline(80, color="black", linestyle=":", linewidth=1.6, label="Target 80%")
    ax.set_ylabel("q10-q90 coverage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in groups])
    ax.set_ylim(45, 100)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.legend(loc="lower right", fontsize=8.5, ncols=2)
    label(ax, "C", "Interval coverage")


def main() -> None:
    helper = load_style_helper()
    helper.apply_publication_style(helper.FigureStyle(font_size=12, axes_linewidth=1.8))
    rows = read_csv(SUMMARY)
    window_rows = read_csv(WINDOWS)
    summaries = lookup(rows)
    native_coverage = group_native_coverage(window_rows)
    palette = helper.PALETTE
    fig, axes = helper.create_subplots(1, 3, figsize=(17.2, 4.9))
    plot_wql_groups(axes[0], summaries, palette)
    plot_q9_failure(axes[1], summaries, palette)
    plot_coverage(axes[2], summaries, native_coverage, palette)
    fig.tight_layout(w_pad=2.8)
    saved = helper.finalize_figure(
        fig,
        OUT_DIR / "latest_coverage_aware_baseline_comparison.png",
        formats=["png", "pdf"],
        dpi=450,
        pad=0.08,
    )
    print(json.dumps({"saved": [str(path.relative_to(ROOT)) for path in saved]}, indent=2))


if __name__ == "__main__":
    main()
