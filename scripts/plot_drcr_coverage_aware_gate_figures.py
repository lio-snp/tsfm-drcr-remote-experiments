#!/usr/bin/env python
"""Plot the coverage-aware DRCR-Smooth safety-gate tradeoff."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".omx" / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

SKILL_HELPER = Path.home() / ".codex" / "skills" / "scientific-figure-pro" / "scripts" / "scientific_figure_pro.py"
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


def pct(value: object) -> float:
    return 100.0 * finite_float(value)


def label(ax, prefix: str, title: str) -> None:
    ax.set_title(f"{prefix}. {title}", loc="left", fontweight="bold", pad=8, fontsize=13)


def annotate_bars(ax, bars, suffix: str = "%", dy: float = 1.0) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            f"{height:.0f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def selected_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["candidate_id"]: row for row in rows}


def summary_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["candidate_id"], row["group"]): row for row in rows if row["phase"] == "test"}


def plot_calibration_tradeoff(ax, selected_rows: list[dict[str, str]], palette: dict[str, str]) -> None:
    for row in selected_rows:
        cid = row["candidate_id"]
        is_selected = int(row["selected"]) == 1
        mode = row["candidate_mode"]
        color = palette["blue_main"] if is_selected else (
            palette["red_strong"] if mode == "full" else palette["neutral"]
        )
        marker = "D" if is_selected else ("X" if mode == "full" else "o")
        size = 120 if is_selected else 55
        ax.scatter(
            pct(row["abs_undercoverage_empirical_risk"]),
            pct(row["wql_harm_empirical_risk"]),
            s=size,
            color=color,
            marker=marker,
            edgecolor="black" if is_selected else "white",
            linewidth=1.2 if is_selected else 0.7,
            zorder=3,
        )
        if cid in {"full_drcr_smooth", "point_only_all", "interval_cap_s1.10", "interval_cap_s1.35"}:
            short = {
                "full_drcr_smooth": "full",
                "point_only_all": "point",
                "interval_cap_s1.10": "cap 1.10",
                "interval_cap_s1.35": "cap 1.35",
            }[cid]
            ax.annotate(short, (pct(row["abs_undercoverage_empirical_risk"]), pct(row["wql_harm_empirical_risk"])),
                        textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axhline(20, color=palette["red_strong"], linestyle="--", linewidth=1.4, alpha=0.8)
    ax.axvline(18, color=palette["blue_secondary"], linestyle="--", linewidth=1.4, alpha=0.8)
    ax.set_xlabel("Strict absolute UC risk (%)")
    ax.set_ylabel("Calibration WQL harm (%)")
    ax.set_xlim(8, 23)
    ax.set_ylim(0, 28)
    ax.grid(alpha=0.22, linestyle="--")
    label(ax, "A", "Calibration tradeoff")


def plot_test_harm(ax, summaries: dict[tuple[str, str], dict[str, str]], palette: dict[str, str]) -> None:
    candidates = [
        ("full_drcr_smooth", "Full"),
        ("point_only_all", "Point"),
        ("interval_cap_s1.10", "Cap 1.10"),
    ]
    groups = [
        ("overall", "Overall"),
        ("role:positive_control", "Positive"),
        ("role:stress_target", "Stress"),
        ("target_id:finance_fred_stress", "Finance"),
    ]
    x = np.arange(len(groups))
    width = 0.25
    colors = [palette["red_strong"], palette["neutral"], palette["blue_main"]]
    for offset, ((cid, name), color) in enumerate(zip(candidates, colors)):
        vals = [pct(summaries[(cid, group)]["repair_wql_noninferiority_harm_rate"]) for group, _ in groups]
        bars = ax.bar(x + (offset - 1) * width, vals, width, label=name, color=color)
        annotate_bars(ax, bars, dy=1.2)
    ax.axhline(20, color=palette["red_strong"], linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_ylabel("Test WQL harm (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in groups])
    ax.set_ylim(0, 82)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    ax.legend(loc="upper left", fontsize=9)
    label(ax, "B", "Stress/control harm")


def plot_failure_and_coverage(ax, summaries: dict[tuple[str, str], dict[str, str]], palette: dict[str, str]) -> None:
    candidates = [
        ("full_drcr_smooth", "Full"),
        ("point_only_all", "Point"),
        ("interval_cap_s1.10", "Cap 1.10"),
    ]
    x = np.arange(len(candidates))
    q9_repair = [
        finite_float(summaries[(cid, "evidence_tier:q9_fullgrid|role:failure_target")]["repair_median_wql_rer"])
        for cid, _ in candidates
    ]
    q9_model = finite_float(
        summaries[("interval_cap_s1.10", "evidence_tier:q9_fullgrid|role:failure_target")]["model_median_wql_rer"]
    )
    coverage = [pct(summaries[(cid, "overall")]["repair_mean_coverage"]) for cid, _ in candidates]
    bars = ax.bar(x, q9_repair, color=[palette["red_strong"], palette["neutral"], palette["blue_main"]], width=0.55)
    ax.axhline(q9_model, color="black", linestyle=":", linewidth=1.5, label="Native q9 failure")
    ax.set_ylabel("q9 failure median WQL-RER")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in candidates])
    ax.set_ylim(0, max(q9_model * 1.12, 3.8))
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    for bar, cov in zip(bars, coverage):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.08,
            f"cov {cov:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.legend(loc="upper right", fontsize=9)
    label(ax, "C", "Failure repair")


def main() -> None:
    helper = load_style_helper()
    helper.apply_publication_style(helper.FigureStyle(font_size=12, axes_linewidth=1.8))
    selected_rows = read_csv(ROOT / "results" / "aaai_stress" / "drcr_smooth_coverage_aware_gate_selected_configs.csv")
    summary_rows = read_csv(ROOT / "results" / "aaai_stress" / "drcr_smooth_coverage_aware_gate_summary.csv")
    summaries = summary_lookup(summary_rows)
    palette = helper.PALETTE
    fig, axes = helper.create_subplots(1, 3, figsize=(16.5, 4.8))
    plot_calibration_tradeoff(axes[0], selected_rows, palette)
    plot_test_harm(axes[1], summaries, palette)
    plot_failure_and_coverage(axes[2], summaries, palette)
    fig.tight_layout(w_pad=3.0)
    saved = helper.finalize_figure(
        fig,
        OUT_DIR / "latest_coverage_aware_gate_tradeoff.png",
        formats=["png", "pdf"],
        dpi=450,
        pad=0.08,
    )
    print(json.dumps({"saved": [str(path.relative_to(ROOT)) for path in saved]}, indent=2))


if __name__ == "__main__":
    main()
