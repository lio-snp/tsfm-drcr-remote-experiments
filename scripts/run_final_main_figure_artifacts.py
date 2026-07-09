#!/usr/bin/env python
"""Build paper-facing final-main-figure artifacts for DRCR.

This script keeps the current DRCR multi-action probe intact and adds the
reviewer-facing pieces needed for a final main figure:

1. a residual-calibrated probabilistic classical baseline,
2. point-metric robustness on the same split (MAE/RMSE/WAPE RER),
3. paired bootstrap confidence intervals, and
4. a draft multi-panel figure for visual inspection.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_chronos_fullgrid_cpr_wql_goal as fullgrid  # noqa: E402
import run_drcr_multi_action_selector_probe as probe  # noqa: E402
import run_drcr_smooth_coverage_aware_gate_goal as coverage_gate  # noqa: E402
import run_drcr_smooth_safety_gate_goal as safety  # noqa: E402
import run_drcr_smooth_score_head_search_goal as smooth  # noqa: E402

from low_snr_tsfm.metrics import mae, relative_error_ratio, rmse, wape  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
FIG_DIR = ROOT / "figures" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "final_main_figure_results_report.md"
WINDOW_OUT = OUT_DIR / "final_main_figure_windows.csv"
SUMMARY_OUT = OUT_DIR / "final_main_figure_summary.csv"
CANDIDATE_OUT = OUT_DIR / "final_main_figure_candidates.csv"
CALIBRATION_OUT = OUT_DIR / "final_main_figure_calibration_tests.csv"
STATUS_OUT = OUT_DIR / "final_main_figure_status.json"
FIGURE_OUT = FIG_DIR / "final_main_figure_draft.png"

EPS = 1e-12
BOOTSTRAP_N = 1000
BOOTSTRAP_ALPHA = 0.05
MIN_RESIDUALS = 40
WQL_RER_CAP = 10.0

PROB_CLASSICAL_ID = "classical_residual_calibrated"
SELECTED_METHOD_ID = "drcr_expert_pull_1.25_cap_1.10"
FRONTIER_PROBE_ID = "drcr_width_veto_expert_pull_1.50_cap_1.10"

DISPLAY_ORDER = [
    "native_tsfm",
    "classical_deterministic",
    PROB_CLASSICAL_ID,
    "global_blend_w0.50",
    "smooth_score_gate_t0.50_w1.00",
    "width_gate_t0.10_w1.00",
    "oracle_native_classical_drcr",
    "drcr_full",
    "drcr_point",
    "drcr_cap_1.10",
    SELECTED_METHOD_ID,
    FRONTIER_PROBE_ID,
]

SHORT_LABEL = {
    "native_tsfm": "Native",
    "classical_deterministic": "Classical\npoint",
    PROB_CLASSICAL_ID: "Classical\ncalib.",
    "global_blend_w0.50": "Global\nblend",
    "smooth_score_gate_t0.50_w1.00": "Score\ngate",
    "width_gate_t0.10_w1.00": "Width\ngate",
    "oracle_native_classical_drcr": "Oracle\nupper",
    "drcr_full": "Full\nDRCR",
    "drcr_point": "Point\nonly",
    "drcr_cap_1.10": "Cap\n1.10",
    SELECTED_METHOD_ID: "EP1.25\nselected",
    "drcr_expert_pull_1.50_cap_1.10": "EP1.50",
    FRONTIER_PROBE_ID: "WidthVeto\nprobe",
    "drcr_score_floor_0.60_cap_1.00": "ScoreFloor",
}

NMI_PASTEL = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_tiny": "#E4E4F0",
    "ours_base": "#E4CCD8",
    "ours_large": "#F0C0CC",
    "bg_peach": "#F0E0D0",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#A8A8A8",
    "neutral_dark": "#606060",
    "delta_up": "#2E9E44",
    "delta_down": "#B64342",
}

METHOD_COLORS = {
    "native_tsfm": NMI_PASTEL["baseline_dark"],
    "classical_deterministic": "#B9B3AA",
    PROB_CLASSICAL_ID: "#D9C7A6",
    "global_blend_w0.50": "#CFC6D6",
    "smooth_score_gate_t0.50_w1.00": "#C7D6C6",
    "width_gate_t0.10_w1.00": "#C4D4DD",
    "oracle_native_classical_drcr": "#F3D58A",
    "drcr_full": NMI_PASTEL["baseline_soft"],
    "drcr_point": NMI_PASTEL["ours_tiny"],
    "drcr_cap_1.10": "#AEBBDD",
    SELECTED_METHOD_ID: NMI_PASTEL["baseline_mid"],
    "drcr_expert_pull_1.50_cap_1.10": "#8D9BC9",
    FRONTIER_PROBE_ID: "#8AB48A",
    "drcr_score_floor_0.60_cap_1.00": NMI_PASTEL["ours_base"],
}

METHOD_EDGES = {
    "native_tsfm": "#2F2F55",
    "classical_deterministic": "#7B756E",
    PROB_CLASSICAL_ID: "#9A7C45",
    "global_blend_w0.50": "#84708F",
    "smooth_score_gate_t0.50_w1.00": "#718F70",
    "width_gate_t0.10_w1.00": "#668596",
    "oracle_native_classical_drcr": "#A87916",
    "drcr_full": "#6C7BAA",
    "drcr_point": "#AEB7D8",
    "drcr_cap_1.10": "#5D6F9D",
    SELECTED_METHOD_ID: "#26396D",
    FRONTIER_PROBE_ID: "#4F7D4F",
}

RISK_COLORS = {
    "wql": "#C97874",
    "protected": "#A88BB7",
    "undercoverage": "#7796C5",
}

TEXT_MUTED = "#606060"
AXIS_DARK = "#272727"
GRID_LIGHT = "#E8E8E8"
REJECT_COLOR = NMI_PASTEL["delta_down"]
SELECTED_COLOR = METHOD_EDGES[SELECTED_METHOD_ID]


def method_color(candidate_id: str) -> str:
    return METHOD_COLORS.get(candidate_id, NMI_PASTEL["neutral_mid"])


def method_edge(candidate_id: str) -> str:
    return METHOD_EDGES.get(candidate_id, "#FFFFFF")


def method_linewidth(candidate_id: str) -> float:
    return 1.35 if candidate_id in {SELECTED_METHOD_ID, FRONTIER_PROBE_ID} else 0.75


def style_method_bars(bars, candidate_id: str) -> None:
    for bar in bars:
        bar.set_facecolor(method_color(candidate_id))
        bar.set_edgecolor(method_edge(candidate_id))
        bar.set_linewidth(method_linewidth(candidate_id))


def style_axes(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=1.0, color=GRID_LIGHT, linestyle="-", linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(AXIS_DARK)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(axis="both", colors=AXIS_DARK, width=0.8, length=3)


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def clipped_wql_rer(value: object) -> float:
    parsed = finite_float(value, float("nan"))
    return min(parsed, WQL_RER_CAP) if math.isfinite(parsed) else float("nan")


def log1p_wql_rer(value: object) -> float:
    parsed = finite_float(value, float("nan"))
    return math.log1p(max(0.0, parsed)) if math.isfinite(parsed) else float("nan")


def rate(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def pct(value: object, digits: int = 1) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{100.0 * parsed:.{digits}f}%"


def num(value: object, digits: int = 3) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key, _ in columns) + " |")
    return "\n".join(lines)


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("||".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def bootstrap_ci(
    rows: list[dict[str, object]],
    stat_fn: Callable[[list[dict[str, object]]], float],
    *,
    seed_key: str,
    n_boot: int = BOOTSTRAP_N,
) -> tuple[float, float]:
    if not rows:
        return float("nan"), float("nan")
    point = stat_fn(rows)
    if len(rows) < 2 or not math.isfinite(point):
        return point, point
    rng = np.random.default_rng(stable_seed(seed_key))
    stats: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(rows), size=len(rows))
        sample = [rows[int(i)] for i in idx]
        value = stat_fn(sample)
        if math.isfinite(value):
            stats.append(value)
    if not stats:
        return float("nan"), float("nan")
    low, high = np.quantile(np.asarray(stats, dtype=float), [BOOTSTRAP_ALPHA / 2, 1.0 - BOOTSTRAP_ALPHA / 2])
    return float(low), float(high)


def window_uid(window: dict[str, object]) -> tuple[str, str, str]:
    return (str(window["source"]), str(window["series_id"]), str(window["window_index"]))


class ResidualQuantileCalibrator:
    """Residual-calibrated classical forecast intervals fit on calibration windows."""

    def __init__(self, windows: list[dict[str, object]]) -> None:
        self.records: list[dict[str, object]] = []
        for window in windows:
            residuals = np.asarray(window["actual"], dtype=float) - np.asarray(window["baseline_forecast"], dtype=float)
            self.records.append(
                {
                    "uid": window_uid(window),
                    "target_id": str(window["target_id"]),
                    "role": str(window["role"]),
                    "family": str(window["family"]),
                    "residuals": residuals,
                }
            )
        self.global_residuals = self._concat(self.records)

    @staticmethod
    def _concat(records: list[dict[str, object]]) -> np.ndarray:
        arrays = [np.asarray(record["residuals"], dtype=float) for record in records]
        return np.concatenate(arrays) if arrays else np.zeros(1, dtype=float)

    def _pool(self, window: dict[str, object], *, exclude_uid: tuple[str, str, str] | None) -> np.ndarray:
        target_id = str(window["target_id"])
        role = str(window["role"])
        family = str(window["family"])

        filters = [
            lambda record: record["target_id"] == target_id,
            lambda record: record["role"] == role and record["family"] == family,
            lambda record: record["role"] == role,
            lambda record: True,
        ]
        for predicate in filters:
            records = [
                record
                for record in self.records
                if (exclude_uid is None or record["uid"] != exclude_uid) and predicate(record)
            ]
            residuals = self._concat(records)
            if residuals.size >= MIN_RESIDUALS:
                return residuals
        return self.global_residuals

    def grid_for(self, window: dict[str, object], *, phase: str) -> np.ndarray:
        exclude_uid = window_uid(window) if phase == "calibration" else None
        residuals = self._pool(window, exclude_uid=exclude_uid)
        levels = np.asarray(window["quantile_levels"], dtype=float)
        residual_quantiles = np.quantile(residuals, levels)
        baseline = np.asarray(window["baseline_forecast"], dtype=float)
        return np.sort(baseline[:, None] + residual_quantiles[None, :], axis=1)


def candidate_configs() -> list[dict[str, object]]:
    configs = []
    for candidate in probe.candidate_configs():
        copied = dict(candidate)
        candidate_id = str(copied["candidate_id"])
        copied["eligible_for_selection"] = int(candidate_id not in {"native_tsfm", "classical_deterministic"})
        configs.append(copied)
        if candidate_id == "classical_deterministic":
            configs.append(
                {
                    "candidate_id": PROB_CLASSICAL_ID,
                    "mode": "classical_residual",
                    "weight_multiplier": "",
                    "interval_cap": "",
                    "score_floor": "",
                    "score_threshold": "",
                    "eligible_for_selection": 0,
                    "description": "classical point forecast plus residual-calibrated quantile intervals fit on calibration windows",
                }
            )
            configs.extend(
                [
                    {
                        "candidate_id": "global_blend_w0.50",
                        "mode": "global_blend",
                        "blend_weight": 0.50,
                        "weight_multiplier": "",
                        "interval_cap": "",
                        "score_floor": "",
                        "score_threshold": "",
                        "eligible_for_selection": 0,
                        "description": "objective baseline: fixed 50/50 blend of native TSFM and deterministic classical quantile grids",
                    },
                    {
                        "candidate_id": "smooth_score_gate_t0.50_w1.00",
                        "mode": "smooth_score_gate",
                        "gate_threshold": 0.50,
                        "blend_weight": 1.00,
                        "weight_multiplier": "",
                        "interval_cap": "",
                        "score_floor": "",
                        "score_threshold": "",
                        "eligible_for_selection": 0,
                        "description": "objective baseline: simple threshold gate on the DRCR smooth diagnostic score",
                    },
                    {
                        "candidate_id": "width_gate_t0.10_w1.00",
                        "mode": "width_gate",
                        "gate_threshold": 0.10,
                        "blend_weight": 1.00,
                        "weight_multiplier": "",
                        "interval_cap": "",
                        "score_floor": "",
                        "score_threshold": "",
                        "eligible_for_selection": 0,
                        "description": "objective baseline: simple threshold gate on native interval width ratio",
                    },
                    {
                        "candidate_id": "oracle_native_classical_drcr",
                        "mode": "oracle_upper_bound",
                        "weight_multiplier": "",
                        "interval_cap": "",
                        "score_floor": "",
                        "score_threshold": "",
                        "eligible_for_selection": 0,
                        "description": "oracle upper bound choosing the lowest WQL among native, classical, selected DRCR, and WidthVeto per window",
                    },
                ]
            )
    return configs


def q50_point(window: dict[str, object], grid: np.ndarray) -> np.ndarray:
    levels = [float(level) for level in window["quantile_levels"]]
    q50_idx = min(range(len(levels)), key=lambda idx: abs(levels[idx] - 0.5))
    return np.asarray(grid[:, q50_idx], dtype=float)


def point_metric_bundle(window: dict[str, object], forecast: np.ndarray) -> dict[str, float]:
    actual = np.asarray(window["actual"], dtype=float)
    baseline = np.asarray(window["baseline_forecast"], dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    forecast_mae = mae(actual, forecast)
    baseline_mae = mae(actual, baseline)
    forecast_rmse = rmse(actual, forecast)
    baseline_rmse = rmse(actual, baseline)
    forecast_wape = wape(actual, forecast)
    baseline_wape = wape(actual, baseline)
    return {
        "mae": forecast_mae,
        "mae_rer": relative_error_ratio(forecast_mae, baseline_mae),
        "rmse": forecast_rmse,
        "rmse_rer": relative_error_ratio(forecast_rmse, baseline_rmse),
        "wape": forecast_wape,
        "wape_rer": relative_error_ratio(forecast_wape, baseline_wape),
    }


def quantile_grid_for(
    window: dict[str, object],
    base: dict[str, object],
    candidate: dict[str, object],
    calibrator: ResidualQuantileCalibrator,
    *,
    phase: str,
) -> tuple[np.ndarray, str, float, float, int]:
    mode = str(candidate["mode"])
    model_grid = np.asarray(window["quantile_grid"], dtype=float)
    classical_grid = smooth.crc.fullgrid.deterministic_baseline_grid(window)
    if str(candidate["candidate_id"]) == PROB_CLASSICAL_ID:
        return calibrator.grid_for(window, phase=phase), "classical_residual_calibrated", 1.0, 1.0, 1
    if mode == "global_blend":
        weight = finite_float(candidate["blend_weight"])
        return model_grid + weight * (classical_grid - model_grid), "global_blend", weight, 1.0, int(weight > 0.0)
    if mode == "smooth_score_gate":
        score = finite_float(base["smooth_score"])
        threshold = finite_float(candidate["gate_threshold"])
        weight = finite_float(candidate["blend_weight"]) if score >= threshold else 0.0
        return model_grid + weight * (classical_grid - model_grid), "smooth_score_gate", weight, 1.0, int(weight > 0.0)
    if mode == "width_gate":
        width_ratio = smooth.crc.native_width_ratio(window)
        threshold = finite_float(candidate["gate_threshold"])
        weight = finite_float(candidate["blend_weight"]) if width_ratio >= threshold else 0.0
        return model_grid + weight * (classical_grid - model_grid), "width_gate", weight, 1.0, int(weight > 0.0)
    if mode == "oracle_upper_bound":
        selected = next(item for item in probe.candidate_configs() if str(item["candidate_id"]) == SELECTED_METHOD_ID)
        frontier = next(item for item in probe.candidate_configs() if str(item["candidate_id"]) == FRONTIER_PROBE_ID)
        selected_grid, _, _, _, _ = probe.quantile_grid_for(window, base, selected)
        frontier_grid, _, _, _, _ = probe.quantile_grid_for(window, base, frontier)
        options = [
            ("native", model_grid, 0.0),
            ("classical", classical_grid, 1.0),
            ("selected_drcr", selected_grid, finite_float(base["effective_weight"])),
            ("width_veto_probe", frontier_grid, finite_float(base["effective_weight"])),
        ]
        best_mode, best_grid, best_weight = min(
            options,
            key=lambda item: fullgrid.quantile_metrics(window, np.asarray(item[1], dtype=float))["wql_rer"],
        )
        return np.asarray(best_grid, dtype=float), f"oracle_{best_mode}", best_weight, 1.0, 1
    return probe.quantile_grid_for(window, base, candidate)


def apply_candidate_to_window(
    window: dict[str, object],
    policy: dict[str, object],
    smooth_candidate: dict[str, object],
    candidate: dict[str, object],
    calibrator: ResidualQuantileCalibrator,
    *,
    phase: str,
) -> dict[str, object]:
    base = safety.base_intervention(window, policy, smooth_candidate)
    model_grid = np.asarray(window["quantile_grid"], dtype=float)
    repair_grid, deployed_mode, weight, scale, veto = quantile_grid_for(
        window,
        base,
        candidate,
        calibrator,
        phase=phase,
    )
    model_quantile = fullgrid.quantile_metrics(window, model_grid)
    repair_quantile = fullgrid.quantile_metrics(window, repair_grid)
    model_point = point_metric_bundle(window, np.asarray(window["model_forecast"], dtype=float))
    repair_point = point_metric_bundle(window, q50_point(window, repair_grid))
    model_undercoverage = max(0.0, smooth.crc.NOMINAL_COVERAGE - model_quantile["coverage"])
    repair_undercoverage = max(0.0, smooth.crc.NOMINAL_COVERAGE - repair_quantile["coverage"])
    undercoverage_delta = max(0.0, repair_undercoverage - model_undercoverage)

    row: dict[str, object] = {
        "phase": phase,
        "candidate_id": candidate["candidate_id"],
        "candidate_mode": candidate["mode"],
        "candidate_description": candidate["description"],
        "eligible_for_selection": int(candidate.get("eligible_for_selection", 1)),
        "deployed_mode": deployed_mode,
        "effective_weight": weight,
        "interval_scale": scale,
        "veto_active": veto,
        "family": window["family"],
        "source": window["source"],
        "dataset": window["dataset"],
        "model": window["model"],
        "series_id": window["series_id"],
        "window_index": window["window_index"],
        "role": window["role"],
        "target_id": window["target_id"],
        "evidence_tier": window["evidence_tier"],
        "quantile_grid_n_levels": window["quantile_grid_n_levels"],
        "protected_signal": int(base["protected_signal"]),
        "smooth_width_score": base["smooth_score"],
        "native_width_ratio": smooth.crc.native_width_ratio(window),
        "model_wql": model_quantile["wql"],
        "repair_wql": repair_quantile["wql"],
        "model_wql_rer": model_quantile["wql_rer"],
        "repair_wql_rer": repair_quantile["wql_rer"],
        "wql_rer_delta_vs_model": repair_quantile["wql_rer"] - model_quantile["wql_rer"],
        "model_coverage_q10_q90": model_quantile["coverage"],
        "repair_coverage_q10_q90": repair_quantile["coverage"],
        "model_undercoverage_risk": model_undercoverage,
        "repair_undercoverage_risk": repair_undercoverage,
        "undercoverage_risk_delta_vs_model": undercoverage_delta,
        "undercoverage_noninferiority_harm": int(undercoverage_delta > probe.UNDERCOVERAGE_HARM_MARGIN),
        "model_wql_failure_delta005": int(model_quantile["wql_rer"] > 1.05),
        "repair_wql_failure_delta005": int(repair_quantile["wql_rer"] > 1.05),
        "repair_win_vs_model": int(repair_quantile["wql_rer"] < model_quantile["wql_rer"]),
        "repair_wql_noninferiority_harm": int(
            repair_quantile["wql_rer"] > model_quantile["wql_rer"] + smooth.WQL_HARM_MARGIN
        ),
    }
    for prefix, metrics in [("model", model_point), ("repair", repair_point)]:
        for metric_name, value in metrics.items():
            row[f"{prefix}_{metric_name}"] = value
        row[f"{prefix}_mae_failure_delta005"] = int(metrics["mae_rer"] > 1.05)
        row[f"{prefix}_rmse_failure_delta005"] = int(metrics["rmse_rer"] > 1.05)
        row[f"{prefix}_wape_failure_delta005"] = int(metrics["wape_rer"] > 1.05)
    return row


def group_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    return {
        "overall": rows,
        "role:failure_target": [row for row in rows if row["role"] == "failure_target"],
        "role:positive_control": [row for row in rows if row["role"] == "positive_control"],
        "role:stress_target": [row for row in rows if row["role"] == "stress_target"],
        "evidence_tier:q9_fullgrid|role:failure_target": [
            row
            for row in rows
            if row["evidence_tier"] == "q9_fullgrid" and row["role"] == "failure_target"
        ],
        "target_id:finance_fred_stress": [row for row in rows if row["target_id"] == "finance_fred_stress"],
        "family:timesfm|role:failure_target": [
            row for row in rows if row["family"] == "timesfm" and row["role"] == "failure_target"
        ],
    }


def metric_reduction(rows: list[dict[str, object]], metric: str) -> float:
    return rate([int(row[f"model_{metric}_failure_delta005"]) for row in rows]) - rate(
        [int(row[f"repair_{metric}_failure_delta005"]) for row in rows]
    )


def summarize_subset(candidate_id: str, phase: str, group: str, rows: list[dict[str, object]]) -> dict[str, object]:
    def stat(name: str, fn: Callable[[list[dict[str, object]]], float]) -> tuple[float, float, float]:
        point = fn(rows)
        low, high = bootstrap_ci(rows, fn, seed_key=f"{candidate_id}|{phase}|{group}|{name}")
        return point, low, high

    repair_wql, repair_wql_low, repair_wql_high = stat(
        "repair_wql",
        lambda sample: median([finite_float(row["repair_wql_rer"], float("nan")) for row in sample]),
    )
    repair_wql_clipped, repair_wql_clipped_low, repair_wql_clipped_high = stat(
        "repair_wql_clipped",
        lambda sample: median([clipped_wql_rer(row["repair_wql_rer"]) for row in sample]),
    )
    delta_wql, delta_wql_low, delta_wql_high = stat(
        "delta_wql",
        lambda sample: median([finite_float(row["wql_rer_delta_vs_model"], float("nan")) for row in sample]),
    )
    delta_wql_clipped, delta_wql_clipped_low, delta_wql_clipped_high = stat(
        "delta_wql_clipped",
        lambda sample: mean(
            [
                clipped_wql_rer(row["repair_wql_rer"]) - clipped_wql_rer(row["model_wql_rer"])
                for row in sample
            ]
        ),
    )
    delta_wql_log, delta_wql_log_low, delta_wql_log_high = stat(
        "delta_wql_log",
        lambda sample: mean(
            [
                log1p_wql_rer(row["repair_wql_rer"]) - log1p_wql_rer(row["model_wql_rer"])
                for row in sample
            ]
        ),
    )
    coverage, coverage_low, coverage_high = stat(
        "coverage",
        lambda sample: mean([finite_float(row["repair_coverage_q10_q90"], float("nan")) for row in sample]),
    )
    harm, harm_low, harm_high = stat(
        "wql_harm",
        lambda sample: rate([int(row["repair_wql_noninferiority_harm"]) for row in sample]),
    )
    mae_r, mae_low, mae_high = stat(
        "mae",
        lambda sample: median([finite_float(row["repair_mae_rer"], float("nan")) for row in sample]),
    )
    rmse_r, rmse_low, rmse_high = stat(
        "rmse",
        lambda sample: median([finite_float(row["repair_rmse_rer"], float("nan")) for row in sample]),
    )
    wape_r, wape_low, wape_high = stat(
        "wape",
        lambda sample: median([finite_float(row["repair_wape_rer"], float("nan")) for row in sample]),
    )

    return {
        "candidate_id": candidate_id,
        "candidate_label": SHORT_LABEL.get(candidate_id, candidate_id),
        "candidate_mode": rows[0]["candidate_mode"],
        "phase": phase,
        "group": group,
        "n_windows": len(rows),
        "model_median_wql_rer": median([finite_float(row["model_wql_rer"], float("nan")) for row in rows]),
        "repair_median_wql_rer": repair_wql,
        "repair_median_wql_rer_ci_low": repair_wql_low,
        "repair_median_wql_rer_ci_high": repair_wql_high,
        "repair_median_clipped_wql_rer": repair_wql_clipped,
        "repair_median_clipped_wql_rer_ci_low": repair_wql_clipped_low,
        "repair_median_clipped_wql_rer_ci_high": repair_wql_clipped_high,
        "paired_median_wql_delta_vs_model": delta_wql,
        "paired_median_wql_delta_vs_model_ci_low": delta_wql_low,
        "paired_median_wql_delta_vs_model_ci_high": delta_wql_high,
        "paired_mean_clipped_wql_delta_vs_model": delta_wql_clipped,
        "paired_mean_clipped_wql_delta_vs_model_ci_low": delta_wql_clipped_low,
        "paired_mean_clipped_wql_delta_vs_model_ci_high": delta_wql_clipped_high,
        "paired_mean_log1p_wql_delta_vs_model": delta_wql_log,
        "paired_mean_log1p_wql_delta_vs_model_ci_low": delta_wql_log_low,
        "paired_mean_log1p_wql_delta_vs_model_ci_high": delta_wql_log_high,
        "wql_harm_rate": harm,
        "wql_harm_rate_ci_low": harm_low,
        "wql_harm_rate_ci_high": harm_high,
        "undercoverage_harm_rate": rate([int(row["undercoverage_noninferiority_harm"]) for row in rows]),
        "repair_mean_coverage": coverage,
        "repair_mean_coverage_ci_low": coverage_low,
        "repair_mean_coverage_ci_high": coverage_high,
        "wql_failure_reduction_vs_model": rate([int(row["model_wql_failure_delta005"]) for row in rows])
        - rate([int(row["repair_wql_failure_delta005"]) for row in rows]),
        "repair_win_rate_vs_model": rate([int(row["repair_win_vs_model"]) for row in rows]),
        "model_median_mae_rer": median([finite_float(row["model_mae_rer"], float("nan")) for row in rows]),
        "repair_median_mae_rer": mae_r,
        "repair_median_mae_rer_ci_low": mae_low,
        "repair_median_mae_rer_ci_high": mae_high,
        "model_median_rmse_rer": median([finite_float(row["model_rmse_rer"], float("nan")) for row in rows]),
        "repair_median_rmse_rer": rmse_r,
        "repair_median_rmse_rer_ci_low": rmse_low,
        "repair_median_rmse_rer_ci_high": rmse_high,
        "model_median_wape_rer": median([finite_float(row["model_wape_rer"], float("nan")) for row in rows]),
        "repair_median_wape_rer": wape_r,
        "repair_median_wape_rer_ci_low": wape_low,
        "repair_median_wape_rer_ci_high": wape_high,
        "mae_failure_reduction_vs_model": metric_reduction(rows, "mae"),
        "rmse_failure_reduction_vs_model": metric_reduction(rows, "rmse"),
        "wape_failure_reduction_vs_model": metric_reduction(rows, "wape"),
    }


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for candidate_id in sorted({str(row["candidate_id"]) for row in rows}):
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        for phase in ["calibration", "test"]:
            phase_rows = [row for row in candidate_rows if row["phase"] == phase]
            if not phase_rows:
                continue
            for group, subset in group_rows(phase_rows).items():
                if subset:
                    output.append(summarize_subset(candidate_id, phase, group, subset))
    return output


def calibration_tests(
    candidates: list[dict[str, object]],
    candidate_calibration_rows: dict[str, list[dict[str, object]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    risk_rows, screen = probe.calibration_tests(candidate_calibration_rows)
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    eligible = [
        cid
        for cid, row in screen.items()
        if int(row["tri_risk_accepted"]) and int(by_id[cid].get("eligible_for_selection", 1))
    ]
    selected_id = max(eligible, key=lambda cid: finite_float(screen[cid]["calibration_utility"], -1e9)) if eligible else "native_tsfm"
    candidate_rows = []
    for candidate in candidates:
        cid = str(candidate["candidate_id"])
        candidate_rows.append(
            {
                **candidate,
                **screen[cid],
                "selected": int(cid == selected_id),
                "candidate_label": SHORT_LABEL.get(cid, cid),
            }
        )
    return risk_rows, candidate_rows, selected_id


def row_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], dict[str, object]]:
    return {(str(row["candidate_id"]), str(row["phase"]), str(row["group"])): row for row in rows}


def load_figure_helper():
    helper = Path.home() / ".codex/skills/scientific-figure-pro/scripts/scientific_figure_pro.py"
    spec = importlib.util.spec_from_file_location("scientific_figure_pro", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load figure helper from {helper}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def err_from_ci(point: float, low: float, high: float) -> list[float]:
    if not all(math.isfinite(value) for value in [point, low, high]):
        return [0.0, 0.0]
    return [max(0.0, point - low), max(0.0, high - point)]


def plot_final_figure(summary_rows: list[dict[str, object]], candidate_rows: list[dict[str, object]]) -> list[Path]:
    mod = load_figure_helper()
    mod.apply_publication_style(mod.FigureStyle(font_size=12, axes_linewidth=1.0))
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["legend.frameon"] = False
    fig, axes = mod.create_subplots(2, 3, figsize=(21, 10.5))
    fig.patch.set_facecolor("white")
    lookup = row_lookup(summary_rows)
    cand_lookup = {str(row["candidate_id"]): row for row in candidate_rows}
    candidates = [cid for cid in DISPLAY_ORDER if cid in cand_lookup]
    x = np.arange(len(candidates))
    labels = [SHORT_LABEL.get(cid, cid) for cid in candidates]

    # A. Calibration screen.
    ax = axes[0]
    risk_series = [
        ("wql_harm_empirical_risk", "WQL harm", RISK_COLORS["wql"]),
        ("protected_harm_empirical_risk", "Protected", RISK_COLORS["protected"]),
        ("undercoverage_harm_empirical_risk", "UC harm", RISK_COLORS["undercoverage"]),
    ]
    width = 0.24
    for i, (field, label, color) in enumerate(risk_series):
        vals = [100.0 * finite_float(cand_lookup[cid][field], float("nan")) for cid in candidates]
        ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.92,
        )
    ax.axhline(20, color=TEXT_MUTED, linestyle="--", linewidth=1.0)
    for i, cid in enumerate(candidates):
        if int(cand_lookup[cid].get("selected", 0)):
            ax.text(i, 26, "selected", ha="center", va="bottom", fontsize=10, color=SELECTED_COLOR, weight="bold")
        elif not int(cand_lookup[cid].get("tri_risk_accepted", 0)):
            ax.text(i, 26, "reject", ha="center", va="bottom", fontsize=10, color=REJECT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Calibration risk (%)")
    ax.set_title("A. Tri-risk calibration screen")
    ax.legend(ncol=3, loc="upper left", fontsize=9)
    style_axes(ax)

    # B. q9 failure WQL-RER.
    ax = axes[1]
    group = "evidence_tier:q9_fullgrid|role:failure_target"
    vals = [finite_float(lookup[(cid, "test", group)]["repair_median_wql_rer"], float("nan")) for cid in candidates]
    yerr = np.asarray(
        [
            err_from_ci(
                finite_float(lookup[(cid, "test", group)]["repair_median_wql_rer"]),
                finite_float(lookup[(cid, "test", group)]["repair_median_clipped_wql_rer_ci_low"]),
                finite_float(lookup[(cid, "test", group)]["repair_median_clipped_wql_rer_ci_high"]),
            )
            for cid in candidates
        ]
    ).T
    colors = [method_color(cid) for cid in candidates]
    bars = ax.bar(x, vals, yerr=yerr, capsize=3, color=colors, error_kw={"elinewidth": 1.0, "capthick": 1.0})
    for bar, cid in zip(bars, candidates):
        bar.set_edgecolor(method_edge(cid))
        bar.set_linewidth(method_linewidth(cid))
    ax.axhline(1.0, color=TEXT_MUTED, linewidth=1.0, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, WQL_RER_CAP + 0.8)
    ax.text(
        0.02,
        0.95,
        f"95% CI clipped at {WQL_RER_CAP:g}; raw CI in table",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        color=TEXT_MUTED,
    )
    ax.set_ylabel("q9 failure WQL-RER")
    ax.set_title("B. Failure-window repair")
    style_axes(ax)

    # C. Harm by safety group.
    ax = axes[2]
    harm_candidates = [
        "classical_deterministic",
        PROB_CLASSICAL_ID,
        "drcr_full",
        "drcr_cap_1.10",
        SELECTED_METHOD_ID,
        FRONTIER_PROBE_ID,
    ]
    harm_groups = [("overall", "Overall"), ("role:stress_target", "Stress"), ("target_id:finance_fred_stress", "Finance")]
    width = 0.12
    xg = np.arange(len(harm_groups))
    for i, cid in enumerate(harm_candidates):
        vals_h = [100.0 * finite_float(lookup[(cid, "test", group_key)]["wql_harm_rate"]) for group_key, _ in harm_groups]
        bars = ax.bar(
            xg + (i - (len(harm_candidates) - 1) / 2) * width,
            vals_h,
            width=width,
            label=SHORT_LABEL[cid],
        )
        style_method_bars(bars, cid)
    ax.axhline(20, color=TEXT_MUTED, linestyle="--", linewidth=1.0)
    ax.set_xticks(xg)
    ax.set_xticklabels([label for _, label in harm_groups])
    ax.set_ylabel("WQL harm rate (%)")
    ax.set_title("C. Harm is the constraint")
    style_axes(ax)

    # D. Coverage by group.
    ax = axes[3]
    coverage_groups = [
        ("overall", "Overall"),
        ("evidence_tier:q9_fullgrid|role:failure_target", "q9 fail"),
        ("role:stress_target", "Stress"),
        ("target_id:finance_fred_stress", "Finance"),
    ]
    coverage_candidates = ["native_tsfm", "classical_deterministic", PROB_CLASSICAL_ID, "drcr_full", SELECTED_METHOD_ID, FRONTIER_PROBE_ID]
    width = 0.12
    xg = np.arange(len(coverage_groups))
    for i, cid in enumerate(coverage_candidates):
        vals_c = [100.0 * finite_float(lookup[(cid, "test", group_key)]["repair_mean_coverage"]) for group_key, _ in coverage_groups]
        bars = ax.bar(
            xg + (i - (len(coverage_candidates) - 1) / 2) * width,
            vals_c,
            width=width,
            label=SHORT_LABEL[cid],
        )
        style_method_bars(bars, cid)
    ax.axhline(80, color=TEXT_MUTED, linestyle="--", linewidth=1.0)
    ax.set_xticks(xg)
    ax.set_xticklabels([label for _, label in coverage_groups])
    ax.set_ylabel("q10-q90 coverage (%)")
    ax.set_title("D. Interval behavior")
    style_axes(ax)

    # E. Point metric robustness.
    ax = axes[4]
    point_candidates = ["native_tsfm", PROB_CLASSICAL_ID, "drcr_cap_1.10", SELECTED_METHOD_ID, FRONTIER_PROBE_ID]
    metrics = [("repair_median_mae_rer", "MAE"), ("repair_median_rmse_rer", "RMSE"), ("repair_median_wape_rer", "WAPE")]
    xg = np.arange(len(metrics))
    width = 0.14
    for i, cid in enumerate(point_candidates):
        vals_p = [finite_float(lookup[(cid, "test", "overall")][field]) for field, _ in metrics]
        bars = ax.bar(
            xg + (i - (len(point_candidates) - 1) / 2) * width,
            vals_p,
            width=width,
            label=SHORT_LABEL[cid],
        )
        style_method_bars(bars, cid)
    ax.axhline(1.0, color=TEXT_MUTED, linestyle=":", linewidth=1.0)
    ax.set_xticks(xg)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Point metric RER")
    ax.set_title("E. Point-metric robustness")
    style_axes(ax)

    # F. Main selected method summary.
    ax = axes[5]
    ax.axis("off")
    selected = lookup[(SELECTED_METHOD_ID, "test", "evidence_tier:q9_fullgrid|role:failure_target")]
    frontier = lookup[(FRONTIER_PROBE_ID, "test", "evidence_tier:q9_fullgrid|role:failure_target")]
    selected_overall = lookup[(SELECTED_METHOD_ID, "test", "overall")]
    lines = [
        "F. Reading guide",
        "",
        f"Selected action: {SHORT_LABEL[SELECTED_METHOD_ID].replace(chr(10), ' ')}",
        f"q9 WQL-RER: {num(selected['model_median_wql_rer'])} -> {num(selected['repair_median_wql_rer'])}",
        f"overall harm: {pct(selected_overall['wql_harm_rate'])}",
        f"overall coverage: {pct(selected_overall['repair_mean_coverage'])}",
        "",
        "WidthVeto is shown as a frontier probe:",
        f"q9 WQL-RER {num(frontier['repair_median_wql_rer'])}, harm {pct(lookup[(FRONTIER_PROBE_ID, 'test', 'overall')]['wql_harm_rate'])}",
        "",
        "Do not claim strict coverage certification.",
    ]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left", fontsize=14)
    legend_ids = [
        "native_tsfm",
        "classical_deterministic",
        PROB_CLASSICAL_ID,
        "drcr_full",
        "drcr_cap_1.10",
        SELECTED_METHOD_ID,
        FRONTIER_PROBE_ID,
    ]
    handles = [
        Patch(
            facecolor=method_color(cid),
            edgecolor=method_edge(cid),
            linewidth=method_linewidth(cid),
            label=SHORT_LABEL[cid].replace("\n", " "),
        )
        for cid in legend_ids
    ]
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.02, 0.03),
        ncol=2,
        fontsize=9,
        handlelength=1.6,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    fig.suptitle("Risk-controlled DRCR action selection for low-structure forecast failure", y=1.02, fontsize=18, weight="bold")
    fig.tight_layout()
    return mod.finalize_figure(fig, FIGURE_OUT, formats=["png", "pdf", "svg"], dpi=350, pad=0.08)


def rows_for_candidate_screen(candidate_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for cid in DISPLAY_ORDER:
        row = next(item for item in candidate_rows if item["candidate_id"] == cid)
        rows.append(
            {
                "Candidate": SHORT_LABEL.get(cid, cid).replace("\n", " "),
                "Selected": "yes" if int(row["selected"]) else "",
                "TriOK": "yes" if int(row["tri_risk_accepted"]) else "no",
                "WQL": pct(row["wql_harm_empirical_risk"]),
                "Prot": pct(row["protected_harm_empirical_risk"]),
                "UC": pct(row["undercoverage_harm_empirical_risk"]),
                "Q9Gain": num(row["calibration_q9_gain"]),
                "Utility": num(row["calibration_utility"], 4),
            }
        )
    return rows


def rows_for_summary(summary_rows: list[dict[str, object]], group: str) -> list[dict[str, object]]:
    lookup = row_lookup(summary_rows)
    rows = []
    for cid in DISPLAY_ORDER:
        row = lookup[(cid, "test", group)]
        rows.append(
            {
                "Candidate": SHORT_LABEL.get(cid, cid).replace("\n", " "),
                "N": int(row["n_windows"]),
                "WQL": num(row["repair_median_wql_rer"]),
                "WQL CI": f"[{num(row['repair_median_wql_rer_ci_low'])}, {num(row['repair_median_wql_rer_ci_high'])}]",
                "Harm": pct(row["wql_harm_rate"]),
                "Coverage": pct(row["repair_mean_coverage"]),
                "MAE": num(row["repair_median_mae_rer"]),
                "RMSE": num(row["repair_median_rmse_rer"]),
                "WAPE": num(row["repair_median_wape_rer"]),
            }
        )
    return rows


def rows_for_denominator_sensitivity(summary_rows: list[dict[str, object]], group: str) -> list[dict[str, object]]:
    lookup = row_lookup(summary_rows)
    rows = []
    for cid in DISPLAY_ORDER:
        row = lookup[(cid, "test", group)]
        rows.append(
            {
                "Candidate": SHORT_LABEL.get(cid, cid).replace("\n", " "),
                "Raw": num(row["repair_median_wql_rer"]),
                "Raw CI": f"[{num(row['repair_median_wql_rer_ci_low'])}, {num(row['repair_median_wql_rer_ci_high'])}]",
                "Clipped": num(row["repair_median_clipped_wql_rer"]),
                "Clipped CI": (
                    f"[{num(row['repair_median_clipped_wql_rer_ci_low'])}, "
                    f"{num(row['repair_median_clipped_wql_rer_ci_high'])}]"
                ),
                "Mean log d": num(row["paired_mean_log1p_wql_delta_vs_model"], 4),
                "Mean log CI": (
                    f"[{num(row['paired_mean_log1p_wql_delta_vs_model_ci_low'], 4)}, "
                    f"{num(row['paired_mean_log1p_wql_delta_vs_model_ci_high'], 4)}]"
                ),
            }
        )
    return rows


def write_report(
    status: dict[str, object],
    candidate_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Final Main-Figure Results",
        "",
        f"Generated: 2026-07-08; run timestamp `{status['timestamp']}`.",
        "",
        "## Purpose",
        "",
        "This run adds the missing reviewer-facing pieces for the final DRCR main figure: a residual-calibrated probabilistic classical baseline, point-metric robustness on the same split, paired bootstrap confidence intervals, and a draft multi-panel figure.",
        "",
        "## Data",
        "",
        f"- Combined windows: `{status['n_windows']}`.",
        f"- Calibration/test split: `{status['n_calibration_windows']}` / `{status['n_test_windows']}`.",
        f"- Candidates: `{status['n_candidates']}`.",
        f"- Selected method under locked eligibility: `{status['selected_candidate_id']}`.",
        f"- Frontier probe: `{FRONTIER_PROBE_ID}`.",
        "",
        "## Calibration Screen",
        "",
        markdown_table(
            rows_for_candidate_screen(candidate_rows),
            [
                ("Candidate", "Candidate"),
                ("Selected", "Selected"),
                ("TriOK", "Tri ok"),
                ("WQL", "WQL harm"),
                ("Prot", "Protected"),
                ("UC", "UC harm"),
                ("Q9Gain", "Q9 gain"),
                ("Utility", "Utility"),
            ],
        ),
        "",
        "## Test: q9 Full-Grid Failure",
        "",
        markdown_table(
            rows_for_summary(summary_rows, "evidence_tier:q9_fullgrid|role:failure_target"),
            [
                ("Candidate", "Candidate"),
                ("N", "N"),
                ("WQL", "WQL-RER"),
                ("WQL CI", "95% CI"),
                ("Harm", "Harm"),
                ("Coverage", "Coverage"),
                ("MAE", "MAE-RER"),
                ("RMSE", "RMSE-RER"),
                ("WAPE", "WAPE-RER"),
            ],
        ),
        "",
        "## Denominator-Fragility Check",
        "",
        f"Raw WQL-RER is retained, but the q9 failure slice has denominator-fragile windows. The plot therefore uses the same median WQL-RER with bootstrap CI clipped at `{WQL_RER_CAP:g}` for readability, while this table reports raw CI, clipped CI, and paired mean log1p delta versus native TSFM.",
        "",
        markdown_table(
            rows_for_denominator_sensitivity(summary_rows, "evidence_tier:q9_fullgrid|role:failure_target"),
            [
                ("Candidate", "Candidate"),
                ("Raw", "Raw WQL"),
                ("Raw CI", "Raw 95% CI"),
                ("Clipped", "Clipped WQL"),
                ("Clipped CI", "Clipped 95% CI"),
                ("Mean log d", "Mean log d"),
                ("Mean log CI", "Mean log 95% CI"),
            ],
        ),
        "",
        "## Test: Overall",
        "",
        markdown_table(
            rows_for_summary(summary_rows, "overall"),
            [
                ("Candidate", "Candidate"),
                ("N", "N"),
                ("WQL", "WQL-RER"),
                ("WQL CI", "95% CI"),
                ("Harm", "Harm"),
                ("Coverage", "Coverage"),
                ("MAE", "MAE-RER"),
                ("RMSE", "RMSE-RER"),
                ("WAPE", "WAPE-RER"),
            ],
        ),
        "",
        "## Interpretation",
        "",
        f"- `{SELECTED_METHOD_ID}` remains the selected method when classical baselines are marked as ineligible comparison baselines rather than selectable repairs.",
        "- `classical_residual_calibrated` is a fairer probabilistic classical baseline than the deterministic fallback: its intervals are fit from calibration residuals and evaluated on the test split.",
        f"- `{FRONTIER_PROBE_ID}` remains a stronger-looking Pareto probe but is not the selected method unless a Pareto selection objective is explicitly frozen and rerun.",
        "- The main claim should remain risk-controlled probabilistic repair plus point-metric robustness, not universal improvement on every metric.",
        "- The q9 failure panel should be read with the denominator-fragility table: raw RER values are preserved, and clipped/log summaries are robustness checks rather than replacement metrics.",
        "",
        "## Artifacts",
        "",
        f"- `{WINDOW_OUT.relative_to(ROOT)}`",
        f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
        f"- `{CANDIDATE_OUT.relative_to(ROOT)}`",
        f"- `{CALIBRATION_OUT.relative_to(ROOT)}`",
        f"- `{FIGURE_OUT.relative_to(ROOT)}`",
        f"- `{FIGURE_OUT.with_suffix('.pdf').relative_to(ROOT)}`",
        f"- `{FIGURE_OUT.with_suffix('.svg').relative_to(ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    smooth.validate_protocol_config()
    windows, counts = coverage_gate.combined_windows()
    calibration_windows = [window for window in windows if smooth.crc.split_bucket(window) == 0]
    test_windows = [window for window in windows if smooth.crc.split_bucket(window) == 1]
    calibrator = ResidualQuantileCalibrator(calibration_windows)
    policy = fullgrid.common_cpr_policy()
    smooth_candidate = probe.selected_smooth_candidate()
    candidates = candidate_configs()

    all_rows: list[dict[str, object]] = []
    calibration_by_candidate: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        candidate_rows: list[dict[str, object]] = []
        for window in windows:
            phase = "calibration" if smooth.crc.split_bucket(window) == 0 else "test"
            row = apply_candidate_to_window(
                window,
                policy,
                smooth_candidate,
                candidate,
                calibrator,
                phase=phase,
            )
            candidate_rows.append(row)
            all_rows.append(row)
        calibration_by_candidate[str(candidate["candidate_id"])] = [
            row for row in candidate_rows if row["phase"] == "calibration"
        ]

    calibration_rows, candidate_rows, selected_id = calibration_tests(candidates, calibration_by_candidate)
    summary_rows = build_summary(all_rows)
    saved_figures = plot_final_figure(summary_rows, candidate_rows)

    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows": len(windows),
        "n_calibration_windows": len(calibration_windows),
        "n_test_windows": len(test_windows),
        "n_candidates": len(candidates),
        "selected_candidate_id": selected_id,
        "frontier_probe_id": FRONTIER_PROBE_ID,
        "bootstrap_n": BOOTSTRAP_N,
        "wql_rer_cap_for_clipped_sensitivity": WQL_RER_CAP,
        "probabilistic_classical_baseline": PROB_CLASSICAL_ID,
        **counts,
        "windows": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "candidates": str(CANDIDATE_OUT.relative_to(ROOT)),
        "calibration_tests": str(CALIBRATION_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
        "figures": [str(path.relative_to(ROOT)) for path in saved_figures],
    }

    write_csv(WINDOW_OUT, all_rows)
    write_csv(SUMMARY_OUT, summary_rows)
    write_csv(CANDIDATE_OUT, candidate_rows)
    write_csv(CALIBRATION_OUT, calibration_rows)
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")
    write_report(status, candidate_rows, summary_rows)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
