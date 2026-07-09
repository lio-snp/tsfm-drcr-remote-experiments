#!/usr/bin/env python
"""Plot the latest factorized failure-family results."""

from __future__ import annotations

import csv
import importlib.util
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".omx" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np


SKILL_HELPER = Path.home() / ".codex" / "skills" / "scientific-figure-pro" / "scripts" / "scientific_figure_pro.py"
OUT_DIR = ROOT / "figures" / "factorized_failure_family"


def load_style_helper():
    if not SKILL_HELPER.exists():
        raise SystemExit(f"Missing scientific figure helper: {SKILL_HELPER}")
    spec = importlib.util.spec_from_file_location("scientific_figure_pro", SKILL_HELPER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load scientific figure helper: {SKILL_HELPER}")
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


def pct(value: float) -> float:
    return 100.0 * finite_float(value)


def annotate_bar(ax, bars, fmt="{:.0f}", dy=2.0, fontsize=9) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def add_panel_label(ax, label: str, title: str) -> None:
    ax.set_title(f"{label}. {title}", loc="left", fontweight="bold", pad=10)


def plot_interaction_ladder(ax, rows, palette) -> None:
    rows = sorted(rows, key=lambda row: int(finite_float(row["active_ex_ante_factor_count"])))
    x = np.asarray([int(finite_float(row["active_ex_ante_factor_count"])) for row in rows], dtype=float)
    failure = np.asarray([pct(row["failure_rate_delta_005"]) for row in rows], dtype=float)
    coverage_bad = np.asarray([pct(row["bad_coverage_rate_lt_070"]) for row in rows], dtype=float)
    flat_bad = np.asarray([pct(row["over_smoothing_rate_flatness_ge_060"]) for row in rows], dtype=float)
    n = [int(finite_float(row["n_windows"])) for row in rows]

    ax.plot(x, failure, marker="o", lw=2.8, color=palette["red_strong"], label="MAE-RER failure")
    ax.plot(x, coverage_bad, marker="s", lw=2.8, color=palette["blue_main"], label="Coverage < 0.70")
    ax.plot(x, flat_bad, marker="^", lw=2.8, color=palette["teal"], label="Flatness >= 0.60")
    for xi, yi, ni in zip(x, failure, n):
        ax.text(xi, yi + 3.5, f"n={ni}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xlabel("Number of active ex-ante factors")
    ax.set_ylabel("Window rate (%)")
    ax.set_ylim(0, max(85, float(np.nanmax([failure.max(), coverage_bad.max(), flat_bad.max()])) + 12))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    add_panel_label(ax, "A", "Failure-regime interaction ladder")


def plot_multimetric_heatmap(ax, rows, mod) -> None:
    selected_groups = [
        "positive_controls",
        "outside_factorized_regime",
        "factorized_regime_n_ge_2",
        "noncontrol_failure_or_stress",
        "denominator_fragile",
    ]
    metrics = [
        ("mae_rer_failure_rate", "MAE\nfail"),
        ("rmse_rer_failure_rate", "RMSE\nfail"),
        ("mase_rer_failure_rate", "MASE\nfail"),
        ("coverage_bad_rate_lt_070", "Coverage\nbad"),
        ("shape_bad_rate", "Shape\nbad"),
    ]
    by_group = {row["group"]: row for row in rows}
    matrix = [[finite_float(by_group[group][metric]) for metric, _ in metrics] for group in selected_groups]
    y_labels = ["Positive\ncontrols", "Outside\nregime", "Regime\nn>=2", "Non-control\nstress", "Denominator\nfragile"]
    x_labels = [label for _, label in metrics]
    image = mod.make_heatmap(
        ax,
        matrix,
        x_labels=x_labels,
        y_labels=y_labels,
        cmap="YlOrRd",
        cbar_label="Rate",
        annotate=False,
    )
    image.set_clim(0.0, 1.0)
    for row_idx, row in enumerate(matrix):
        for col_idx, value in enumerate(row):
            color = "white" if value >= 0.58 else "black"
            ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=9, color=color)
    add_panel_label(ax, "B", "Multi-metric robustness")


def plot_tsfm_synthetic(ax, rows, palette) -> None:
    by_factor_value = {(row["factor"], row["value"]): row for row in rows}
    categories = ["context 24", "context 48", "context 96", "decay 0", "decay .05"]
    failure = [
        pct(by_factor_value[("context_length", "24")]["failure_rate_delta_005"]),
        pct(by_factor_value[("context_length", "48")]["failure_rate_delta_005"]),
        pct(by_factor_value[("context_length", "96")]["failure_rate_delta_005"]),
        pct(by_factor_value[("decay_rate", "0.0")]["failure_rate_delta_005"]),
        pct(by_factor_value[("decay_rate", "0.05")]["failure_rate_delta_005"]),
    ]
    coverage_bad = [
        100.0 - pct(by_factor_value[("context_length", "24")]["mean_empirical_coverage_90"]),
        100.0 - pct(by_factor_value[("context_length", "48")]["mean_empirical_coverage_90"]),
        100.0 - pct(by_factor_value[("context_length", "96")]["mean_empirical_coverage_90"]),
        100.0 - pct(by_factor_value[("decay_rate", "0.0")]["mean_empirical_coverage_90"]),
        100.0 - pct(by_factor_value[("decay_rate", "0.05")]["mean_empirical_coverage_90"]),
    ]
    x = np.arange(len(categories))
    width = 0.36
    b1 = ax.bar(x - width / 2, failure, width, color=palette["red_strong"], label="Failure")
    b2 = ax.bar(x + width / 2, coverage_bad, width, color=palette["blue_secondary"], label="1 - coverage")
    annotate_bar(ax, b1, dy=1.5, fontsize=8)
    annotate_bar(ax, b2, dy=1.5, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    add_panel_label(ax, "C", "Chronos-Bolt tiny on controlled synthetic windows")


def plot_repair(ax, rows, palette) -> None:
    groups = ["failure_target", "stress_target", "positive_control", "overall"]
    labels = ["Failure\ntarget", "Stress\ntarget", "Positive\ncontrol", "Overall"]
    by_group = {row["group"]: row for row in rows}
    model = [pct(by_group[group]["model_failure_rate_delta_005"]) for group in groups]
    repair = [pct(by_group[group]["repair_failure_rate_delta_005"]) for group in groups]
    gate = [pct(by_group[group]["gate_rate"]) for group in groups]
    x = np.arange(len(groups))
    width = 0.35
    b1 = ax.bar(x - width / 2, model, width, color=palette["neutral"], edgecolor="white", label="TSFM")
    b2 = ax.bar(x + width / 2, repair, width, color=palette["green_3"], edgecolor="white", label="Gated mixture")
    annotate_bar(ax, b1, dy=1.2, fontsize=8)
    annotate_bar(ax, b2, dy=1.2, fontsize=8)
    for xi, gi in zip(x, gate):
        ax.text(xi, max(model[int(xi)], repair[int(xi)]) + 9, f"gate {gi:.0f}%", ha="center", fontsize=8, color="#444444")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Failure rate (%)")
    ax.set_ylim(0, max(70, max(model + repair) + 18))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)
    add_panel_label(ax, "D", "Held-out repair before vs after")


def main() -> None:
    mod = load_style_helper()
    mod.apply_publication_style(mod.FigureStyle(font_size=12, axes_linewidth=1.8))
    palette = mod.PALETTE

    ladder = read_csv(ROOT / "results" / "failure_family" / "factor_interaction_ladder.csv")
    robustness = read_csv(ROOT / "results" / "failure_family" / "multimetric_robustness_summary.csv")
    synthetic = read_csv(ROOT / "results" / "failure_family" / "tsfm_synthetic_ablation_summary.csv")
    repair = read_csv(ROOT / "results" / "repair" / "validation_calibrated_gate_repair_summary.csv")

    fig, axes = mod.create_subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    plot_interaction_ladder(axes[0], ladder, palette)
    plot_multimetric_heatmap(axes[1], robustness, mod)
    plot_tsfm_synthetic(axes[2], synthetic, palette)
    plot_repair(axes[3], repair, palette)
    fig.suptitle("Factorized failure family: latest evidence snapshot", fontweight="bold", y=1.02)
    paths = mod.finalize_figure(
        fig,
        OUT_DIR / "latest_factorized_failure_family_dashboard",
        formats=["png", "svg", "pdf"],
        dpi=400,
        pad=0.08,
    )
    print("\n".join(str(path.relative_to(ROOT)) for path in paths))


if __name__ == "__main__":
    main()
