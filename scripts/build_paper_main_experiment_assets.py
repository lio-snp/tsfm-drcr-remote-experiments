#!/usr/bin/env python
"""Build paper-facing main experiment table and visualization assets.

This script is intentionally downstream of the locked final-main artifacts. It
does not rerun models; it only reads the current result CSV and regenerates the
paper table/figure so deferred q9 reruns can be ingested later without manual
table editing.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


RESULTS_PATH = ROOT / "results" / "aaai_stress" / "final_main_results.csv"
OUT_CSV = ROOT / "results" / "aaai_stress" / "paper_main_experiment_big_table.csv"
OUT_TEX = ROOT / "paper" / "tables" / "main_experiment_big_table.tex"
FIG_BASE = ROOT / "figures" / "aaai_stress" / "paper_main_experiment_summary"
RADAR_BASE = ROOT / "figures" / "aaai_stress" / "paper_main_radar_tradeoff"
SCENARIO_RADAR_BASE = ROOT / "figures" / "aaai_stress" / "paper_main_scenario_metric_radars"
SCENARIO_RADAR_CSV = ROOT / "results" / "aaai_stress" / "paper_scenario_radar_values.csv"
REPORT_PATH = ROOT / "docs" / "paper_main_experiment_assets.md"

SELECTED = "drcr_expert_pull_1.25_cap_1.10"
WIDTH_VETO = "drcr_width_veto_expert_pull_1.50_cap_1.10"

METHODS = [
    ("native_tsfm", "Native TSFM", "reference", "Reference"),
    ("classical_deterministic", "Classical point", "baseline", "Unsafe coverage"),
    ("classical_residual_calibrated", "Classical calibrated", "baseline", "Unsafe harm"),
    ("global_blend_w0.50", "Global blend", "baseline", "Strong but harmful"),
    ("smooth_score_gate_t0.50_w1.00", "Score gate", "gate", "Failure-only gate"),
    ("width_gate_t0.10_w1.00", "Width gate", "gate", "Unsafe coverage"),
    ("drcr_full", "Full DRCR", "ablation", "Too aggressive"),
    ("drcr_point", "Point-only DRCR", "ablation", "Point repair"),
    ("drcr_cap_1.10", "Cap 1.10", "ablation", "Conservative DRCR"),
    (SELECTED, "DRCR selected", "proposed", "Selected"),
    (WIDTH_VETO, "WidthVeto probe", "probe", "Unfrozen probe"),
    ("oracle_native_classical_drcr", "Oracle upper", "oracle", "Upper bound"),
]

GROUPS = {
    "overall": "overall",
    "q9": "evidence_tier:q9_fullgrid|role:failure_target",
    "positive": "role:positive_control",
    "stress": "role:stress_target",
    "finance": "target_id:finance_fred_stress",
}

SHORT_LABELS = {
    "native_tsfm": "Native",
    "classical_deterministic": "C-pt",
    "classical_residual_calibrated": "C-cal",
    "global_blend_w0.50": "Blend",
    "smooth_score_gate_t0.50_w1.00": "Score",
    "width_gate_t0.10_w1.00": "Width",
    "drcr_full": "Full",
    "drcr_point": "Point",
    "drcr_cap_1.10": "Cap",
    SELECTED: "DRCR",
    WIDTH_VETO: "WV",
    "oracle_native_classical_drcr": "Oracle",
}

RADAR_LABELS = {
    "native_tsfm": "Native",
    "classical_deterministic": "C-pt",
    "classical_residual_calibrated": "C-cal",
    "global_blend_w0.50": "Blend",
    "smooth_score_gate_t0.50_w1.00": "Score",
    "width_gate_t0.10_w1.00": "Width",
    "drcr_full": "Full",
    "drcr_point": "Point",
    "drcr_cap_1.10": "Cap",
    SELECTED: "DRCR",
    WIDTH_VETO: "WV",
    "oracle_native_classical_drcr": "Oracle",
}

NAVY = "#1A1A3A"
CREAM = "#FBFAF4"
GRID = "#DFDED5"

COLORS = {
    "reference": "#BDBCB5",
    "baseline": "#E8E8F5",
    "gate": "#D7DBEE",
    "ablation": "#E8F0E8",
    "proposed": "#8EAE98",
    "probe": "#C9DCCB",
    "oracle": "#D8D1E8",
}

TEXT_COLORS = {
    "reference": NAVY,
    "baseline": NAVY,
    "gate": NAVY,
    "ablation": NAVY,
    "proposed": NAVY,
    "probe": NAVY,
    "oracle": NAVY,
}

RADAR_COLORS = {
    "native_tsfm": "#9E9E9E",
    "classical_deterministic": "#C7C7C7",
    "classical_residual_calibrated": "#B8B8B8",
    "global_blend_w0.50": "#AFAFAF",
    "smooth_score_gate_t0.50_w1.00": "#D0D0D0",
    "width_gate_t0.10_w1.00": "#D0D0D0",
    "drcr_full": "#E8F0E8",
    "drcr_point": "#D5E4D7",
    "drcr_cap_1.10": "#BED4C2",
    SELECTED: "#7EA188",
    WIDTH_VETO: "#A8C4AD",
    "oracle_native_classical_drcr": "#CFCFCF",
}

RADAR_LINESTYLES = {
    "native_tsfm": (0, (4, 2)),
    "oracle_native_classical_drcr": (0, (2, 2)),
}

SCENARIO_LABELS = {
    "overall": "Overall",
    "evidence_tier:q9_fullgrid|role:failure_target": "q9 failure",
    "role:positive_control": "Positive control",
    "role:stress_target": "Stress target",
    "target_id:finance_fred_stress": "Finance stress",
}

SCENARIO_RADAR_METHODS = [
    "native_tsfm",
    "classical_deterministic",
    "classical_residual_calibrated",
    "global_blend_w0.50",
    "drcr_full",
    "drcr_cap_1.10",
    SELECTED,
]

SCENARIO_METRICS = [
    ("WQL", "repair_median_wql_rer", "error"),
    ("MAE", "repair_median_mae_rer", "error"),
    ("RMSE", "repair_median_rmse_rer", "error"),
    ("WAPE", "repair_median_wape_rer", "error"),
    ("Coverage", "repair_mean_coverage", "coverage"),
    ("Safety", "wql_harm_rate", "safety"),
]


def load_style_module():
    skill_path = Path.home() / ".codex" / "skills" / "scientific-figure-pro" / "scripts" / "scientific_figure_pro.py"
    if not skill_path.exists():
        return None
    spec = importlib.util.spec_from_file_location("scientific_figure_pro", skill_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def finite(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return parsed if math.isfinite(parsed) else float("nan")


def fmt_num(value: object, digits: int = 3) -> str:
    parsed = finite(value)
    return "--" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def fmt_pct(value: object, digits: int = 1) -> str:
    parsed = finite(value)
    return "--" if not math.isfinite(parsed) else f"{100.0 * parsed:.{digits}f}"


def latex_escape(text: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def row_lookup(df: pd.DataFrame, candidate_id: str, group: str) -> pd.Series:
    rows = df[(df["phase"] == "test") & (df["candidate_id"] == candidate_id) & (df["group"] == group)]
    if rows.empty:
        raise KeyError(f"Missing result for {candidate_id=} {group=}")
    return rows.iloc[0]


def build_summary() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    rows: list[dict[str, object]] = []
    for candidate_id, name, method_type, verdict in METHODS:
        overall = row_lookup(df, candidate_id, GROUPS["overall"])
        q9 = row_lookup(df, candidate_id, GROUPS["q9"])
        positive = row_lookup(df, candidate_id, GROUPS["positive"])
        stress = row_lookup(df, candidate_id, GROUPS["stress"])
        finance = row_lookup(df, candidate_id, GROUPS["finance"])
        rows.append(
            {
                "candidate_id": candidate_id,
                "method": name,
                "type": method_type,
                "overall_wql_rer": finite(overall["repair_median_wql_rer"]),
                "overall_harm_rate": finite(overall["wql_harm_rate"]),
                "q9_wql_rer": finite(q9["repair_median_wql_rer"]),
                "q9_coverage": finite(q9["repair_mean_coverage"]),
                "q9_harm_rate": finite(q9["wql_harm_rate"]),
                "positive_harm_rate": finite(positive["wql_harm_rate"]),
                "stress_harm_rate": finite(stress["wql_harm_rate"]),
                "finance_wql_rer": finite(finance["repair_median_wql_rer"]),
                "finance_harm_rate": finite(finance["wql_harm_rate"]),
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def write_big_table(summary: pd.DataFrame) -> None:
    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_CSV, index=False)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Main experiment table on the locked final-main test split. WQL-RER and harm are lower-is-better; coverage is higher-is-better. We do not bold per-metric winners because several low-WQL baselines achieve this by harming protected windows. The proposed row is the candidate selected by the calibration screen.}",
        r"\label{tab:main_experiment_big}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccccl}",
        r"\toprule",
        r"Method & Type & \multicolumn{2}{c}{Overall} & \multicolumn{3}{c}{q9 failure} & \multicolumn{2}{c}{Safety / stress} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
        r"& & WQL$\downarrow$ & Harm$\downarrow$ & WQL$\downarrow$ & Cov.$\uparrow$ & Harm$\downarrow$ & PC/Stress/Fin harm$\downarrow$ & Finance WQL$\downarrow$ \\",
        r"\midrule",
    ]

    for _, row in summary.iterrows():
        bold = row["candidate_id"] == SELECTED
        values = [
            latex_escape(row["method"]),
            latex_escape(row["type"]),
            fmt_num(row["overall_wql_rer"]),
            fmt_pct(row["overall_harm_rate"]),
            fmt_num(row["q9_wql_rer"]),
            fmt_pct(row["q9_coverage"]),
            fmt_pct(row["q9_harm_rate"]),
            f"{fmt_pct(row['positive_harm_rate'])}/{fmt_pct(row['stress_harm_rate'])}/{fmt_pct(row['finance_harm_rate'])}",
            fmt_num(row["finance_wql_rer"]),
        ]
        if bold:
            values = [rf"\textbf{{{value}}}" for value in values]
        line = " & ".join(values) + r" \\"
        if row["candidate_id"] == SELECTED:
            lines.append(r"\midrule")
        lines.append(line)
        if row["candidate_id"] == WIDTH_VETO:
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\vspace{0.25em}",
            r"\caption*{\footnotesize PC/Stress/Fin harm reports WQL harm on positive-control, stress-target, and finance-stress slices. Moirai/TimesFM q9 full-grid refresh remains external-compute deferred and is excluded until ingestion passes.}",
            r"\end{table*}",
            "",
        ]
    )
    OUT_TEX.write_text("\n".join(lines))


def make_bar(ax, x, vals, colors, ylabel, title, ylim=None, annotate=True):
    bars = ax.bar(x, vals, color=colors, edgecolor=NAVY, linewidth=0.9)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(1.0, color=NAVY, lw=1.0, ls="--", alpha=0.55)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    if annotate:
        top = ax.get_ylim()[1]
        for bar, val in zip(bars, vals):
            label = f"{val:.2f}" if math.isfinite(val) else "--"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(bar.get_height() + top * 0.02, top * 0.96),
                label,
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if len(vals) > 8 else 0,
            )


def clipped_score(value: float, bad: float, good: float) -> float:
    """Map a lower-is-better value to a 0--100 higher-is-better score."""
    if not math.isfinite(value) or bad == good:
        return float("nan")
    score = 100.0 * (bad - value) / (bad - good)
    return float(np.clip(score, 0.0, 100.0))


def build_radar_scores(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert heterogeneous metrics into higher-is-better radar spokes.

    The anchors are fixed to the current paper protocol rather than fit per
    plotted subset: WQL scores use the observed bad end of the locked suite and
    the oracle-near good end, while harm and coverage are natural percentages.
    """
    rows: list[dict[str, object]] = []
    for _, row in summary.iterrows():
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "label": RADAR_LABELS[row["candidate_id"]],
                "Overall fit": clipped_score(float(row["overall_wql_rer"]), bad=1.20, good=0.65),
                "Failure repair": clipped_score(float(row["q9_wql_rer"]), bad=3.50, good=0.70),
                "Coverage": float(np.clip(100.0 * row["q9_coverage"], 0.0, 100.0)),
                "Overall safety": float(np.clip(100.0 * (1.0 - row["overall_harm_rate"]), 0.0, 100.0)),
                "PC safety": float(np.clip(100.0 * (1.0 - row["positive_harm_rate"]), 0.0, 100.0)),
                "Finance safety": float(np.clip(100.0 * (1.0 - row["finance_harm_rate"]), 0.0, 100.0)),
            }
        )
    return pd.DataFrame(rows)


def draw_radar_panel(ax, radar: pd.DataFrame, candidate_ids: list[str], title: str) -> None:
    metrics = ["Overall fit", "Failure repair", "Coverage", "Overall safety", "PC safety", "Finance safety"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_facecolor(CREAM)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=9, color=NAVY)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], fontsize=8, color=NAVY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.spines["polar"].set_color(NAVY)
    ax.spines["polar"].set_linewidth(1.0)
    ax.set_title(title, y=1.06, color=NAVY, fontsize=12)

    handles = []
    labels = []
    radar_by_id = radar.set_index("candidate_id")
    for candidate_id in candidate_ids:
        values = [float(radar_by_id.loc[candidate_id, metric]) for metric in metrics]
        values += values[:1]
        is_selected = candidate_id == SELECTED
        is_drcr_family = candidate_id in {"drcr_full", "drcr_point", "drcr_cap_1.10", SELECTED, WIDTH_VETO}
        linewidth = 2.8 if is_selected else (1.8 if is_drcr_family else 1.2)
        alpha = 0.95 if is_selected else (0.75 if is_drcr_family else 0.55)
        linestyle = RADAR_LINESTYLES.get(candidate_id, "solid")
        color = RADAR_COLORS[candidate_id]
        (line,) = ax.plot(angles, values, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
        if is_selected:
            ax.fill(angles, values, color=color, alpha=0.18)
        elif is_drcr_family:
            ax.fill(angles, values, color=color, alpha=0.07)
        handles.append(line)
        labels.append(RADAR_LABELS[candidate_id])

    ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)


def write_radar_figure(summary: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "axes.edgecolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "figure.facecolor": CREAM,
            "axes.facecolor": CREAM,
            "savefig.facecolor": CREAM,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    radar = build_radar_scores(summary)
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.6), subplot_kw={"polar": True}, facecolor=CREAM)

    draw_radar_panel(
        axes[0],
        radar,
        [
            "native_tsfm",
            "classical_deterministic",
            "classical_residual_calibrated",
            "global_blend_w0.50",
            SELECTED,
        ],
        "A. Baselines vs. selected DRCR",
    )
    draw_radar_panel(
        axes[1],
        radar,
        [
            "native_tsfm",
            "drcr_full",
            "drcr_point",
            "drcr_cap_1.10",
            SELECTED,
            WIDTH_VETO,
        ],
        "B. DRCR family variants",
    )

    fig.suptitle("Multi-objective repair trade-off (outer is better on every spoke)", y=0.985, fontsize=13, color=NAVY)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.82, bottom=0.16, wspace=0.32)
    RADAR_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(RADAR_BASE.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(RADAR_BASE.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def format_raw_value(metric_kind: str, value: float) -> str:
    if not math.isfinite(value):
        return "--"
    if metric_kind in {"coverage", "safety"}:
        display = value if metric_kind == "coverage" else (1.0 - value)
        return f"{100.0 * display:.0f}%"
    return f"{value:.2f}"


def scenario_score(values: pd.Series, value: float, metric_kind: str) -> float:
    if not math.isfinite(value):
        return float("nan")
    if metric_kind == "coverage":
        return float(np.clip(100.0 * value, 0.0, 100.0))
    if metric_kind == "safety":
        return float(np.clip(100.0 * (1.0 - value), 0.0, 100.0))

    finite_values = np.array([v for v in values.to_numpy(float) if math.isfinite(float(v))])
    if finite_values.size == 0:
        return float("nan")
    good = float(np.nanmin(finite_values))
    bad = float(np.nanmax(finite_values))
    if abs(bad - good) < 1e-12:
        return 70.0
    return float(np.clip(100.0 * (bad - value) / (bad - good), 0.0, 100.0))


def build_scenario_radar_values() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_PATH)
    rows: list[dict[str, object]] = []
    for group, scenario in SCENARIO_LABELS.items():
        subset = df[(df["phase"] == "test") & (df["group"] == group) & (df["candidate_id"].isin(SCENARIO_RADAR_METHODS))]
        for candidate_id in SCENARIO_RADAR_METHODS:
            method_row = subset[subset["candidate_id"] == candidate_id]
            if method_row.empty:
                continue
            method_row = method_row.iloc[0]
            for metric_label, col, metric_kind in SCENARIO_METRICS:
                raw = finite(method_row[col])
                score = scenario_score(subset[col], raw, metric_kind)
                rows.append(
                    {
                        "scenario": scenario,
                        "group": group,
                        "candidate_id": candidate_id,
                        "method": RADAR_LABELS[candidate_id],
                        "metric": metric_label,
                        "metric_kind": metric_kind,
                        "raw_value": raw,
                        "display_value": format_raw_value(metric_kind, raw),
                        "score_outer_better": score,
                    }
                )
    out = pd.DataFrame(rows)
    SCENARIO_RADAR_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SCENARIO_RADAR_CSV, index=False)
    return out


def draw_scenario_panel(ax, values: pd.DataFrame, scenario: str, title: str) -> list[object]:
    metrics = [metric for metric, _, _ in SCENARIO_METRICS]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    selected_values = values[(values["scenario"] == scenario) & (values["candidate_id"] == SELECTED)]
    selected_by_metric = selected_values.set_index("metric")
    axis_labels = []
    for metric, _, metric_kind in SCENARIO_METRICS:
        suffix = selected_by_metric.loc[metric, "display_value"] if metric in selected_by_metric.index else "--"
        arrow = "\u2191" if metric_kind == "coverage" else "\u2193"
        if metric_kind == "safety":
            axis_labels.append(f"{metric}\nDRCR={suffix}")
        else:
            axis_labels.append(f"{metric}{arrow}\nDRCR={suffix}")

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_facecolor(CREAM)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=7.5, color=NAVY)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["", "50", "", "100"], fontsize=7, color=NAVY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.spines["polar"].set_color(NAVY)
    ax.spines["polar"].set_linewidth(1.0)
    ax.set_title(title, y=1.11, color=NAVY, fontsize=10.5)

    handles = []
    panel = values[values["scenario"] == scenario]
    for candidate_id in SCENARIO_RADAR_METHODS:
        method = panel[panel["candidate_id"] == candidate_id]
        if method.empty:
            continue
        method = method.set_index("metric").loc[metrics].reset_index()
        scores = method["score_outer_better"].to_numpy(float).tolist()
        scores += scores[:1]
        is_selected = candidate_id == SELECTED
        is_drcr = candidate_id.startswith("drcr_")
        color = RADAR_COLORS[candidate_id]
        linewidth = 2.6 if is_selected else (1.6 if is_drcr else 1.0)
        alpha = 0.95 if is_selected else (0.65 if is_drcr else 0.35)
        linestyle = RADAR_LINESTYLES.get(candidate_id, "solid")
        (line,) = ax.plot(angles, scores, color=color, linewidth=linewidth, alpha=alpha, linestyle=linestyle)
        if is_selected:
            ax.fill(angles, scores, color=color, alpha=0.16)
        elif is_drcr:
            ax.fill(angles, scores, color=color, alpha=0.04)
        handles.append(line)
    return handles


def write_scenario_radar_figure() -> None:
    values = build_scenario_radar_values()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "axes.edgecolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "figure.facecolor": CREAM,
            "axes.facecolor": CREAM,
            "savefig.facecolor": CREAM,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 8.8), subplot_kw={"polar": True}, facecolor=CREAM)
    axes_flat = axes.flatten()
    titles = [
        ("Overall", "A. Overall"),
        ("q9 failure", "B. q9 failure"),
        ("Positive control", "C. Positive control"),
        ("Stress target", "D. Stress target"),
        ("Finance stress", "E. Finance stress"),
    ]
    legend_handles = None
    for ax, (scenario, title) in zip(axes_flat[:5], titles):
        handles = draw_scenario_panel(ax, values, scenario, title)
        if legend_handles is None:
            legend_handles = handles

    axes_flat[5].set_axis_off()
    labels = [RADAR_LABELS[cid] for cid in SCENARIO_RADAR_METHODS]
    if legend_handles is not None:
        axes_flat[5].legend(
            legend_handles,
            labels,
            loc="center",
            ncol=1,
            fontsize=9,
            frameon=False,
            title="Methods",
            title_fontsize=10,
        )
    fig.suptitle(
        "Scenario-wise metric radars: proposed variants share sage green, baselines are muted gray",
        y=0.985,
        fontsize=13,
        color=NAVY,
    )
    fig.text(
        0.5,
        0.035,
        "Each panel is one task scenario. Spokes are metrics; outer is better after per-scenario scaling. Axis labels show selected DRCR's raw value; full raw values are exported to CSV.",
        ha="center",
        va="center",
        fontsize=9,
        color=NAVY,
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.90, bottom=0.09, hspace=0.42, wspace=0.25)
    SCENARIO_RADAR_BASE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCENARIO_RADAR_BASE.with_suffix(".png"), dpi=450, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(SCENARIO_RADAR_BASE.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def write_figure(summary: pd.DataFrame) -> None:
    style = load_style_module()
    if style is not None:
        style.apply_publication_style(style.FigureStyle(font_size=10, axes_linewidth=1.5))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 10,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "axes.edgecolor": NAVY,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "figure.facecolor": CREAM,
            "axes.facecolor": CREAM,
            "savefig.facecolor": CREAM,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    plot_ids = [
        "native_tsfm",
        "classical_deterministic",
        "classical_residual_calibrated",
        "global_blend_w0.50",
        "smooth_score_gate_t0.50_w1.00",
        "drcr_full",
        "drcr_cap_1.10",
        SELECTED,
        WIDTH_VETO,
        "oracle_native_classical_drcr",
    ]
    plot_df = summary.set_index("candidate_id").loc[plot_ids].reset_index()
    x = np.arange(len(plot_df))
    labels = [SHORT_LABELS[cid] for cid in plot_df["candidate_id"]]
    colors = [COLORS[row["type"]] for _, row in plot_df.iterrows()]

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.2), facecolor=CREAM)
    axes = axes.flatten()

    make_bar(
        axes[0],
        x,
        plot_df["overall_wql_rer"].to_numpy(float),
        colors,
        "WQL-RER",
        "A. Overall error (lower is better)",
        ylim=(0, 1.15),
    )
    make_bar(
        axes[1],
        x,
        plot_df["q9_wql_rer"].to_numpy(float),
        colors,
        "WQL-RER",
        "B. Severe failure error (lower is better)",
        ylim=(0, 3.65),
    )

    width = 0.38
    axes[2].bar(
        x - width / 2,
        100 * plot_df["overall_harm_rate"].to_numpy(float),
        width,
        label="Overall harm",
        color="#D2D4EA",
        edgecolor=NAVY,
        linewidth=0.7,
    )
    axes[2].bar(
        x + width / 2,
        100 * plot_df["finance_harm_rate"].to_numpy(float),
        width,
        label="Finance harm",
        color="#E8F0E8",
        edgecolor=NAVY,
        linewidth=0.7,
    )
    axes[2].set_ylabel("Harm rate (%)")
    axes[2].set_title("C. Hidden harm (lower is better)")
    axes[2].grid(axis="y", color=GRID, linewidth=0.7)
    axes[2].legend(ncol=2, loc="upper right", fontsize=8)

    width = 0.38
    axes[3].bar(
        x - width / 2,
        100 * plot_df["q9_coverage"].to_numpy(float),
        width,
        label="q9 coverage",
        color="#E8E8F5",
        edgecolor=NAVY,
        linewidth=0.7,
    )
    axes[3].bar(
        x + width / 2,
        100 * plot_df["positive_harm_rate"].to_numpy(float),
        width,
        label="PC harm",
        color="#C9DCCB",
        edgecolor=NAVY,
        linewidth=0.7,
    )
    axes[3].set_ylabel("Rate (%)")
    axes[3].set_title("D. Coverage vs. safety")
    axes[3].grid(axis="y", color=GRID, linewidth=0.7)
    axes[3].legend(ncol=1, loc="upper right", fontsize=8, frameon=False)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(NAVY)
        ax.spines["bottom"].set_color(NAVY)
        ax.tick_params(colors=NAVY)

    for ax in axes[:2]:
        ax.set_xticklabels([])
        ax.set_xlabel("")
    for ax in axes[2:]:
        ax.tick_params(axis="x", labelsize=9)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.90, bottom=0.12, hspace=0.55, wspace=0.20)
    fig.suptitle("DRCR repairs severe probabilistic failures without blanket fallback harm", y=0.975, fontsize=13, color=NAVY)
    FIG_BASE.parent.mkdir(parents=True, exist_ok=True)
    if style is not None:
        style.finalize_figure(fig, FIG_BASE, formats=["png", "pdf"], dpi=450, pad=0.08)
    else:
        fig.tight_layout(pad=2)
        fig.savefig(FIG_BASE.with_suffix(".png"), dpi=450, bbox_inches="tight")
        fig.savefig(FIG_BASE.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def write_report(summary: pd.DataFrame) -> None:
    selected = summary[summary["candidate_id"] == SELECTED].iloc[0]
    native = summary[summary["candidate_id"] == "native_tsfm"].iloc[0]
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Paper Main Experiment Assets",
                "",
                "Generated from `results/aaai_stress/final_main_results.csv`.",
                "",
                "## Outputs",
                "",
                f"- `{OUT_CSV.relative_to(ROOT)}`",
                f"- `{OUT_TEX.relative_to(ROOT)}`",
                f"- `{FIG_BASE.with_suffix('.png').relative_to(ROOT)}`",
                f"- `{FIG_BASE.with_suffix('.pdf').relative_to(ROOT)}`",
                f"- `{RADAR_BASE.with_suffix('.png').relative_to(ROOT)}`",
                f"- `{RADAR_BASE.with_suffix('.pdf').relative_to(ROOT)}`",
                f"- `{SCENARIO_RADAR_BASE.with_suffix('.png').relative_to(ROOT)}`",
                f"- `{SCENARIO_RADAR_BASE.with_suffix('.pdf').relative_to(ROOT)}`",
                f"- `{SCENARIO_RADAR_CSV.relative_to(ROOT)}`",
                "",
                "## Key Numbers",
                "",
                f"- Overall WQL-RER: native `{native['overall_wql_rer']:.3f}` -> DRCR `{selected['overall_wql_rer']:.3f}`.",
                f"- q9 failure WQL-RER: native `{native['q9_wql_rer']:.3f}` -> DRCR `{selected['q9_wql_rer']:.3f}`.",
                f"- DRCR harm rates PC/stress/finance: `{100*selected['positive_harm_rate']:.1f}%` / `{100*selected['stress_harm_rate']:.1f}%` / `{100*selected['finance_harm_rate']:.1f}%`.",
                "",
                "## Boundary",
                "",
                "Moirai/TimesFM q9 full-grid refresh remains external-compute deferred and is not counted in these current-suite results.",
                "",
            ]
        )
    )


def main() -> None:
    summary = build_summary()
    write_big_table(summary)
    write_figure(summary)
    write_radar_figure(summary)
    write_scenario_radar_figure()
    write_report(summary)
    print(f"Wrote {OUT_TEX.relative_to(ROOT)}")
    print(f"Wrote {FIG_BASE.with_suffix('.png').relative_to(ROOT)}")
    print(f"Wrote {FIG_BASE.with_suffix('.pdf').relative_to(ROOT)}")
    print(f"Wrote {RADAR_BASE.with_suffix('.png').relative_to(ROOT)}")
    print(f"Wrote {RADAR_BASE.with_suffix('.pdf').relative_to(ROOT)}")
    print(f"Wrote {SCENARIO_RADAR_BASE.with_suffix('.png').relative_to(ROOT)}")
    print(f"Wrote {SCENARIO_RADAR_BASE.with_suffix('.pdf').relative_to(ROOT)}")
    print(f"Wrote {SCENARIO_RADAR_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
