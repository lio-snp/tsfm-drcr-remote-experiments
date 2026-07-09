#!/usr/bin/env python
"""Plot the DRCR multi-action selector probe with objective baselines."""

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
from matplotlib.patches import Patch

SKILL_HELPER = Path.home() / ".codex" / "skills" / "scientific-figure-pro" / "scripts" / "scientific_figure_pro.py"
SUMMARY = ROOT / "results" / "aaai_stress" / "drcr_multi_action_selector_probe_summary.csv"
CANDIDATES = ROOT / "results" / "aaai_stress" / "drcr_multi_action_selector_probe_candidates.csv"
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


def label(ax, prefix: str, title: str) -> None:
    ax.set_title(f"{prefix}. {title}", loc="left", fontweight="bold", pad=8, fontsize=13)


def annotate_bars(ax, bars, fmt: str = "{:.2f}", dy: float = 0.035, fontsize: float = 7.2) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0,
        )


def summary_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["candidate_id"], row["group"]): row for row in rows if row["phase"] == "test"}


def candidate_lookup(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["candidate_id"]: row for row in rows}


METHODS = [
    ("native_tsfm", "Native"),
    ("classical_deterministic", "Classical"),
    ("drcr_full", "Full"),
    ("drcr_point", "Point"),
    ("drcr_cap_1.10", "Cap\n1.10"),
    ("drcr_expert_pull_1.25_cap_1.10", "EP\n1.25"),
    ("drcr_expert_pull_1.50_cap_1.10", "EP\n1.50"),
    ("drcr_width_veto_expert_pull_1.50_cap_1.10", "WV\n1.50"),
    ("drcr_score_floor_0.60_cap_1.00", "SF\n0.60"),
]


def method_colors(palette: dict[str, str]) -> dict[str, str]:
    return {
        "native_tsfm": "#222222",
        "classical_deterministic": palette["teal"],
        "drcr_full": palette["red_strong"],
        "drcr_point": palette["neutral"],
        "drcr_cap_1.10": palette["blue_secondary"],
        "drcr_expert_pull_1.25_cap_1.10": palette["blue_main"],
        "drcr_expert_pull_1.50_cap_1.10": palette["green_3"],
        "drcr_width_veto_expert_pull_1.50_cap_1.10": palette["highlight"],
        "drcr_score_floor_0.60_cap_1.00": palette["violet"],
    }


def plot_wql_bar(
    ax,
    rows: dict[tuple[str, str], dict[str, str]],
    palette: dict[str, str],
    group: str,
    title: str,
    ylim: tuple[float, float],
    prefix: str,
) -> None:
    colors = method_colors(palette)
    values = [finite_float(rows[(cid, group)]["repair_median_wql_rer"]) for cid, _ in METHODS]
    x = np.arange(len(METHODS))
    bars = ax.bar(
        x,
        values,
        color=[colors[cid] for cid, _ in METHODS],
        edgecolor="black",
        linewidth=0.7,
        width=0.72,
    )
    selected_index = [cid for cid, _ in METHODS].index("drcr_expert_pull_1.25_cap_1.10")
    frontier_index = [cid for cid, _ in METHODS].index("drcr_width_veto_expert_pull_1.50_cap_1.10")
    bars[selected_index].set_hatch("//")
    bars[selected_index].set_linewidth(1.5)
    bars[frontier_index].set_hatch("..")
    bars[frontier_index].set_linewidth(1.5)
    ax.axhline(1.0, color="black", linestyle=":", linewidth=1.5)
    ax.text(len(METHODS) - 0.35, 1.0 + 0.035, "Classical = 1", ha="right", va="bottom", fontsize=8)
    annotate_bars(ax, bars, dy=0.035 if ylim[1] < 2 else 0.085)
    ax.set_ylabel("Median WQL-RER\n(lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in METHODS], fontsize=8)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    label(ax, prefix, title)


def plot_coverage(ax, rows: dict[tuple[str, str], dict[str, str]], palette: dict[str, str]) -> None:
    selected_methods = [
        ("native_tsfm", "Native"),
        ("classical_deterministic", "Classical"),
        ("drcr_full", "Full"),
        ("drcr_cap_1.10", "Cap"),
        ("drcr_expert_pull_1.25_cap_1.10", "ExpertPull"),
        ("drcr_width_veto_expert_pull_1.50_cap_1.10", "WidthVeto"),
    ]
    groups = [
        ("overall", "Overall"),
        ("evidence_tier:q9_fullgrid|role:failure_target", "q9 failure"),
        ("role:stress_target", "Stress"),
        ("target_id:finance_fred_stress", "Finance"),
    ]
    colors = method_colors(palette)
    x = np.arange(len(groups))
    width = 0.13
    for idx, (cid, name) in enumerate(selected_methods):
        vals = [100.0 * finite_float(rows[(cid, group)]["repair_mean_coverage"]) for group, _ in groups]
        bars = ax.bar(
            x + (idx - 2.5) * width,
            vals,
            width,
            label=name,
            color=colors[cid],
            edgecolor="black",
            linewidth=0.5,
        )
        if cid in {"classical_deterministic", "drcr_width_veto_expert_pull_1.50_cap_1.10"}:
            annotate_bars(ax, bars, fmt="{:.0f}", dy=1.2, fontsize=6.5)
    ax.axhline(80, color="black", linestyle=":", linewidth=1.5)
    ax.text(len(groups) - 0.25, 81.5, "Nominal 80%", ha="right", va="bottom", fontsize=8)
    ax.set_ylabel("q10-q90 coverage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in groups])
    ax.set_ylim(0, 102)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    label(ax, "D", "Why pure fallback is not a probabilistic repair")


def plot_harm_tradeoff(ax, rows: dict[tuple[str, str], dict[str, str]], palette: dict[str, str]) -> None:
    methods = [
        ("drcr_full", "Full"),
        ("drcr_cap_1.10", "Cap"),
        ("drcr_expert_pull_1.25_cap_1.10", "ExpertPull"),
        ("drcr_expert_pull_1.50_cap_1.10", "Pull 1.50"),
        ("drcr_width_veto_expert_pull_1.50_cap_1.10", "WidthVeto"),
        ("drcr_score_floor_0.60_cap_1.00", "ScoreFloor"),
    ]
    groups = [
        ("overall", "Overall"),
        ("role:stress_target", "Stress"),
        ("target_id:finance_fred_stress", "Finance"),
    ]
    colors = method_colors(palette)
    x = np.arange(len(groups))
    width = 0.12
    for idx, (cid, name) in enumerate(methods):
        vals = [100.0 * finite_float(rows[(cid, group)]["wql_harm_rate"]) for group, _ in groups]
        bars = ax.bar(
            x + (idx - 2.5) * width,
            vals,
            width,
            label=name,
            color=colors[cid],
            edgecolor="black",
            linewidth=0.5,
        )
        if cid in {"drcr_full", "drcr_width_veto_expert_pull_1.50_cap_1.10"}:
            annotate_bars(ax, bars, fmt="{:.0f}", dy=1.1, fontsize=6.5)
    ax.axhline(20, color="black", linestyle=":", linewidth=1.5)
    ax.text(len(groups) - 0.15, 21.3, "Risk cap 20%", ha="right", va="bottom", fontsize=8)
    ax.set_ylabel("WQL harm rate (%)\n(lower is safer)")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in groups])
    ax.set_ylim(0, 78)
    ax.grid(axis="y", alpha=0.22, linestyle="--")
    label(ax, "E", "Safety constraint rejects attractive but unsafe actions")


def plot_calibration_screen(ax, candidates: dict[str, dict[str, str]], palette: dict[str, str]) -> None:
    methods = [cid for cid, _ in METHODS]
    risk_names = ["wql_harm_empirical_risk", "protected_harm_empirical_risk", "undercoverage_harm_empirical_risk"]
    risk_labels = ["WQL harm", "Protected harm", "UC harm"]
    values = np.array([[100.0 * finite_float(candidates[cid][risk]) for risk in risk_names] for cid in methods])
    im = ax.imshow(values, aspect="auto", cmap="Blues", vmin=0, vmax=35)
    for row_idx, cid in enumerate(methods):
        for col_idx in range(values.shape[1]):
            value = values[row_idx, col_idx]
            color = "white" if value > 18 else "#222222"
            ax.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", fontsize=7, color=color)
        tri_ok = int(finite_float(candidates[cid]["tri_risk_accepted"])) == 1
        selected = int(finite_float(candidates[cid]["selected"])) == 1
        marker = "PASS" if tri_ok else "REJECT"
        marker_color = palette["blue_main"] if selected else ("#1F7A3A" if tri_ok else palette["red_strong"])
        ax.text(values.shape[1] + 0.12, row_idx, marker + ("*" if selected else ""), ha="left", va="center", fontsize=7.2, color=marker_color)
    ax.set_xticks(np.arange(len(risk_labels)))
    ax.set_xticklabels(risk_labels, fontsize=8)
    ax.set_yticks(np.arange(len(METHODS)))
    ax.set_yticklabels([name.replace("\n", " ") for _, name in METHODS], fontsize=7.3)
    ax.set_xlim(-0.5, values.shape[1] + 1.25)
    ax.set_title("Calibration risk (%)", loc="left", fontsize=9)
    label(ax, "C", "LTT/CRC candidate screen")
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    cbar.ax.tick_params(labelsize=7)


def main() -> None:
    helper = load_style_helper()
    helper.apply_publication_style(helper.FigureStyle(font_size=12, axes_linewidth=1.8))
    summary_rows = summary_lookup(read_csv(SUMMARY))
    candidate_rows = candidate_lookup(read_csv(CANDIDATES))
    palette = helper.PALETTE

    fig, axes = helper.create_subplots(2, 3, figsize=(18.8, 9.3), gridspec_kw={"width_ratios": [1.35, 1.35, 1.1]})
    plot_wql_bar(axes[0], summary_rows, palette, "overall", "Overall WQL vs objective baselines", (0, 1.33), "A")
    plot_wql_bar(
        axes[1],
        summary_rows,
        palette,
        "evidence_tier:q9_fullgrid|role:failure_target",
        "Hard failure slice",
        (0, 3.85),
        "B",
    )
    plot_calibration_screen(axes[2], candidate_rows, palette)
    plot_coverage(axes[3], summary_rows, palette)
    plot_harm_tradeoff(axes[4], summary_rows, palette)
    axes[5].axis("off")
    axes[5].text(
        0.02,
        0.88,
        "Takeaway",
        fontsize=13,
        fontweight="bold",
        transform=axes[5].transAxes,
    )
    axes[5].text(
        0.02,
        0.72,
        "Classical fallback repairs point error\nbut destroys interval coverage.\n\n"
        "Full interval repair improves coverage\nbut violates WQL harm constraints.\n\n"
        "ExpertPull 1.25 is the conservative\ncalibrated selection.\n\n"
        "WidthVeto Pull1.50 is the stronger\nPareto probe: lower overall/q9 error,\nlow stress harm, zero finance harm.",
        fontsize=9.5,
        linespacing=1.25,
        transform=axes[5].transAxes,
        va="top",
    )
    colors = method_colors(palette)
    legend_handles = [
        Patch(facecolor=colors["native_tsfm"], edgecolor="black", label="Native TSFM"),
        Patch(facecolor=colors["classical_deterministic"], edgecolor="black", label="Classical fallback"),
        Patch(facecolor=colors["drcr_full"], edgecolor="black", label="Full repair"),
        Patch(facecolor=colors["drcr_cap_1.10"], edgecolor="black", label="Cap 1.10"),
        Patch(facecolor=colors["drcr_expert_pull_1.25_cap_1.10"], edgecolor="black", hatch="//", label="ExpertPull 1.25"),
        Patch(facecolor=colors["drcr_expert_pull_1.50_cap_1.10"], edgecolor="black", label="ExpertPull 1.50"),
        Patch(facecolor=colors["drcr_width_veto_expert_pull_1.50_cap_1.10"], edgecolor="black", hatch="..", label="WidthVeto Pull1.50"),
        Patch(facecolor=colors["drcr_score_floor_0.60_cap_1.00"], edgecolor="black", label="ScoreFloor 0.60"),
    ]
    axes[5].legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 0.015),
        ncols=2,
        fontsize=7.3,
        handlelength=1.35,
        handletextpad=0.55,
        columnspacing=0.8,
    )
    fig.tight_layout(w_pad=2.2, h_pad=2.6)
    saved = helper.finalize_figure(
        fig,
        OUT_DIR / "latest_multi_action_selector_probe.png",
        formats=["png", "pdf"],
        dpi=450,
        pad=0.06,
    )
    print(json.dumps({"saved": [str(path.relative_to(ROOT)) for path in saved]}, indent=2))


if __name__ == "__main__":
    main()
