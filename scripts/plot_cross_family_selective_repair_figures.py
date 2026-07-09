#!/usr/bin/env python
"""Plot cross-family selective repair validation results."""

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


def annotate_bars(ax, bars, dy: float = 1.1, fontsize: int = 8) -> None:
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


def by_strategy_group(rows):
    return {(row["strategy_id"], row["group"]): row for row in rows}


def plot_family_before_after(ax, lookup, palette) -> None:
    families = [("family:chronos", "Chronos"), ("family:moirai", "Moirai"), ("family:timesfm", "TimesFM")]
    strategy = "risk_controlled_leave_family_out"
    model = [pct(lookup[(strategy, group)]["model_failure_rate_delta_005"]) for group, _ in families]
    repair = [pct(lookup[(strategy, group)]["repair_failure_rate_delta_005"]) for group, _ in families]
    x = np.arange(len(families))
    width = 0.36
    b1 = ax.bar(x - width / 2, model, width, color=palette["neutral"], label="TSFM")
    b2 = ax.bar(x + width / 2, repair, width, color=palette["green_3"], label="Shielded repair")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in families])
    ax.set_ylabel("Failure rate (%)")
    ax.set_ylim(0, 82)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    add_panel_label(ax, "A", "Cross-family repair by model family")


def plot_strategy_tradeoff(ax, lookup, palette) -> None:
    strategies = [
        ("risk_controlled_leave_family_out", "Shielded\nLFO"),
        ("cross_family_margin_pareto", "Unshielded\nfrontier"),
        ("previous_high_recall", "High\nrecall"),
        ("global_blend_w0.75", "Global\nblend"),
        ("strict_safe_gate", "Strict\nsafe"),
    ]
    failure = [pct(lookup[(strategy, "overall")]["repair_failure_rate_delta_005"]) for strategy, _ in strategies]
    gate = [pct(lookup[(strategy, "overall")]["gate_rate"]) for strategy, _ in strategies]
    x = np.arange(len(strategies))
    width = 0.34
    b1 = ax.bar(x - width / 2, failure, width, color=palette["green_3"], label="Overall failure")
    b2 = ax.bar(x + width / 2, gate, width, color=palette["blue_secondary"], label="Gate rate")
    annotate_bars(ax, b1)
    annotate_bars(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in strategies])
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    add_panel_label(ax, "B", "Repair strength vs selectivity")


def plot_positive_control_margin(ax, lookup, palette) -> None:
    strategies = [
        ("risk_controlled_leave_family_out", "Shielded\nLFO"),
        ("cross_family_margin_pareto", "Unshielded\nfrontier"),
        ("previous_high_recall", "High\nrecall"),
        ("global_blend_w0.50", "Global\n.50"),
        ("global_blend_w0.75", "Global\n.75"),
        ("strict_safe_gate", "Strict\nsafe"),
    ]
    gate = [pct(lookup[(strategy, "positive_control")]["gate_rate"]) for strategy, _ in strategies]
    rer_delta = [finite_float(lookup[(strategy, "positive_control")]["median_rer_delta"]) for strategy, _ in strategies]
    x = np.arange(len(strategies))
    ax2 = ax.twinx()
    bars = ax.bar(x, gate, width=0.52, color=palette["blue_secondary"], alpha=0.82)
    line = ax2.plot(x, rer_delta, color=palette["red_strong"], marker="o", lw=2.5)
    annotate_bars(ax, bars)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in strategies])
    ax.set_ylabel("Positive-control gate rate (%)")
    ax2.set_ylabel("Median RER delta")
    ax.set_ylim(0, 108)
    ax2.set_ylim(-0.05, max(0.42, max(rer_delta) + 0.08))
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend([bars, line[0]], ["Gate rate", "Median RER delta"], loc="upper left", fontsize=9)
    add_panel_label(ax, "C", "Positive-control margin cost")


def plot_family_reduction(ax, lookup, palette) -> None:
    families = [("family:chronos", "Chronos"), ("family:moirai", "Moirai"), ("family:timesfm", "TimesFM")]
    strategies = [
        ("risk_controlled_leave_family_out", "Shielded"),
        ("previous_high_recall", "High recall"),
        ("global_blend_w0.75", "Global .75"),
    ]
    x = np.arange(len(families))
    width = 0.24
    colors = [palette["green_3"], palette["blue_main"], palette["red_2"]]
    for offset, (strategy, label) in enumerate(strategies):
        values = [pct(lookup[(strategy, group)]["failure_rate_reduction"]) for group, _ in families]
        bars = ax.bar(x + (offset - 1) * width, values, width, color=colors[offset], label=label)
        annotate_bars(ax, bars, dy=0.8, fontsize=8)
    ax.axhline(0, color="#333333", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in families])
    ax.set_ylabel("Failure-rate reduction (pp)")
    ax.set_ylim(-4, 30)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", fontsize=9)
    add_panel_label(ax, "D", "Transfer boundary by family")


def main() -> None:
    mod = load_style_helper()
    mod.apply_publication_style(mod.FigureStyle(font_size=12, axes_linewidth=1.8))
    palette = mod.PALETTE
    summary = read_csv(ROOT / "results" / "repair" / "cross_family_selective_repair_strategy_summary.csv")
    lookup = by_strategy_group(summary)

    fig, axes = mod.create_subplots(2, 2, figsize=(16, 10.4), constrained_layout=True)
    plot_family_before_after(axes[0], lookup, palette)
    plot_strategy_tradeoff(axes[1], lookup, palette)
    plot_positive_control_margin(axes[2], lookup, palette)
    plot_family_reduction(axes[3], lookup, palette)
    fig.suptitle("Cross-family validation of selective failure-aware repair", fontweight="bold", y=1.02)
    saved = mod.finalize_figure(
        fig,
        OUT_DIR / "latest_cross_family_selective_repair_dashboard",
        formats=["png", "svg", "pdf"],
        dpi=400,
        pad=0.08,
    )
    print("\n".join(str(path.relative_to(ROOT)) for path in saved))


if __name__ == "__main__":
    main()
