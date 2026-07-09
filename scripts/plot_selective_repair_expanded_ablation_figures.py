#!/usr/bin/env python
"""Plot selective repair and expanded controlled-ablation results."""

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
OUT_DIR = ROOT / "figures" / "selective_repair"


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


def pct(value: object) -> float:
    return 100.0 * finite_float(value)


def add_panel_label(ax, label: str, title: str) -> None:
    ax.set_title(f"{label}. {title}", loc="left", fontweight="bold", pad=10)


def annotate_bars(ax, bars, dy: float = 1.2, fontsize: int = 8) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def plot_expanded_synthetic(ax, synthetic_rows, palette) -> None:
    order = [
        ("context_length", "24", "ctx\n24"),
        ("context_length", "48", "ctx\n48"),
        ("context_length", "96", "ctx\n96"),
        ("seasonality", "0.0", "seas\n0"),
        ("seasonality", "0.5", "seas\n.5"),
        ("seasonality", "1.5", "seas\n1.5"),
        ("spike_size", "0.0", "spike\n0"),
        ("spike_size", "5.0", "spike\n5"),
        ("decay_rate", "0.0", "decay\n0"),
        ("decay_rate", "0.05", "decay\n.05"),
    ]
    by_key = {(row["factor"], row["value"]): row for row in synthetic_rows}
    failure = [pct(by_key[(factor, value)]["failure_rate_delta_005"]) for factor, value, _ in order]
    coverage_bad = [100.0 - pct(by_key[(factor, value)]["mean_empirical_coverage_90"]) for factor, value, _ in order]
    labels = [label for _, _, label in order]
    x = np.arange(len(labels))
    width = 0.38
    b1 = ax.bar(x - width / 2, failure, width, color=palette["red_strong"], label="Failure")
    b2 = ax.bar(x + width / 2, coverage_bad, width, color=palette["blue_secondary"], label="1 - coverage")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    add_panel_label(ax, "A", "Expanded Chronos synthetic ablation")


def by_strategy_group(summary_rows):
    return {(row["strategy_id"], row["group"]): row for row in summary_rows}


def plot_strategy_tradeoff(ax, summary_rows, palette) -> None:
    lookup = by_strategy_group(summary_rows)
    strategies = [
        ("selective_margin_pareto", "Selective\nPareto"),
        ("previous_high_recall", "High\nrecall"),
        ("global_blend_w0.75", "Global\nblend .75"),
        ("strict_safe_gate", "Strict\nsafe"),
    ]
    overall_failure = [pct(lookup[(strategy, "overall")]["repair_failure_rate_delta_005"]) for strategy, _ in strategies]
    gate_rate = [pct(lookup[(strategy, "overall")]["gate_rate"]) for strategy, _ in strategies]
    x = np.arange(len(strategies))
    width = 0.36
    b1 = ax.bar(x - width / 2, overall_failure, width, color=palette["green_3"], label="Overall failure")
    b2 = ax.bar(x + width / 2, gate_rate, width, color=palette["neutral"], label="Gate rate")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in strategies])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    add_panel_label(ax, "B", "Repair strength vs selectivity")


def plot_selective_before_after(ax, summary_rows, palette) -> None:
    lookup = by_strategy_group(summary_rows)
    groups = [
        ("failure_target", "Failure\ntarget"),
        ("stress_target", "Stress\ntarget"),
        ("positive_control", "Positive\ncontrol"),
        ("overall", "Overall"),
    ]
    model = [pct(lookup[("selective_margin_pareto", group)]["model_failure_rate_delta_005"]) for group, _ in groups]
    repair = [pct(lookup[("selective_margin_pareto", group)]["repair_failure_rate_delta_005"]) for group, _ in groups]
    x = np.arange(len(groups))
    width = 0.36
    b1 = ax.bar(x - width / 2, model, width, color=palette["neutral"], label="TSFM")
    b2 = ax.bar(x + width / 2, repair, width, color=palette["green_3"], label="Selective repair")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in groups])
    ax.set_ylabel("Failure rate (%)")
    ax.set_ylim(0, 62)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)
    add_panel_label(ax, "C", "Selective Pareto repair before vs after")


def plot_positive_control_margin(ax, summary_rows, palette) -> None:
    lookup = by_strategy_group(summary_rows)
    strategies = [
        ("selective_margin_pareto", "Selective\nPareto"),
        ("previous_high_recall", "High\nrecall"),
        ("global_blend_w0.25", "Global\n.25"),
        ("global_blend_w0.50", "Global\n.50"),
        ("global_blend_w0.75", "Global\n.75"),
    ]
    gate = [pct(lookup[(strategy, "positive_control")]["gate_rate"]) for strategy, _ in strategies]
    rer_delta = [finite_float(lookup[(strategy, "positive_control")]["median_rer_delta"]) for strategy, _ in strategies]
    x = np.arange(len(strategies))
    ax2 = ax.twinx()
    bars = ax.bar(x, gate, width=0.52, color=palette["blue_secondary"], alpha=0.8, label="Gate rate")
    line = ax2.plot(x, rer_delta, color=palette["red_strong"], marker="o", lw=2.5, label="Median RER delta")
    annotate_bars(ax, bars, dy=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in strategies])
    ax.set_ylabel("Positive-control gate rate (%)")
    ax2.set_ylabel("Positive-control median RER delta")
    ax.set_ylim(0, 108)
    ax2.set_ylim(-0.05, max(0.45, max(rer_delta) + 0.08))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    handles = [bars, line[0]]
    ax.legend(handles, ["Gate rate", "Median RER delta"], loc="upper left", fontsize=9)
    add_panel_label(ax, "D", "Positive-control margin diagnostics")


def main() -> None:
    mod = load_style_helper()
    mod.apply_publication_style(mod.FigureStyle(font_size=12, axes_linewidth=1.8))
    palette = mod.PALETTE
    synthetic = read_csv(ROOT / "results" / "failure_family" / "tsfm_synthetic_ablation_summary.csv")
    summary = read_csv(ROOT / "results" / "repair" / "selective_gate_strategy_comparison.csv")

    fig, axes = mod.create_subplots(2, 2, figsize=(16, 10.5), constrained_layout=True)
    plot_expanded_synthetic(axes[0], synthetic, palette)
    plot_strategy_tradeoff(axes[1], summary, palette)
    plot_selective_before_after(axes[2], summary, palette)
    plot_positive_control_margin(axes[3], summary, palette)
    fig.suptitle("Selective repair and expanded controlled ablation", fontweight="bold", y=1.02)
    saved = mod.finalize_figure(
        fig,
        OUT_DIR / "latest_selective_repair_expanded_ablation_dashboard",
        formats=["png", "svg", "pdf"],
        dpi=400,
        pad=0.08,
    )
    print("\n".join(str(path.relative_to(ROOT)) for path in saved))


if __name__ == "__main__":
    main()
