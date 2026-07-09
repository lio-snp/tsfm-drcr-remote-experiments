#!/usr/bin/env python
"""Build paper-faithful metric robustness tables for RC-RR-CSSR.

The main repair claim remains the unified MAE-RER failure reduction. This
script adds a companion robustness layer that follows the metric families used
by the original model papers and GIFT-Eval where the current artifacts allow it.
When the raw artifacts only expose q10/q50/q90, probabilistic metrics are marked
as proxies rather than exact paper metrics.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_cross_family_selective_repair_goal as cross  # noqa: E402
import run_factorized_failure_family_goal as base  # noqa: E402

from low_snr_tsfm.metrics import (  # noqa: E402
    empirical_coverage,
    mae,
    mean_weighted_quantile_loss,
    relative_error_ratio,
    rmse,
    smape,
    wape,
)
from low_snr_tsfm.quantile_artifacts import quantile_matrix_from_rows  # noqa: E402


OUT_DIR = ROOT / "results" / "repair"
AAAI_OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "paper_faithful_metric_robustness_report.md"
WINDOW_OUT = OUT_DIR / "paper_faithful_metric_robustness_windows.csv"
SUMMARY_OUT = OUT_DIR / "paper_faithful_metric_robustness_summary.csv"
REGISTRY_OUT = OUT_DIR / "paper_faithful_metric_registry.csv"
GIFT_EXACT_OUT = OUT_DIR / "paper_faithful_gift_exact_metric_summary.csv"
STATUS_OUT = OUT_DIR / "paper_faithful_metric_robustness_status.json"

PRIMARY_STRATEGY = "rc_rr_cssr_leave_source_calibrated"
CPR_STRATEGY = "cpr_ltt_leave_source"
NOMINAL_Q10_Q90_COVERAGE = 0.80
EPS = 1e-12


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def optional_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def rate(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def p90(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.percentile(finite, 90)) if finite else float("nan")


def pct(value: object) -> str:
    return f"{100.0 * finite_float(value):.1f}%"


def num(value: object, digits: int = 3) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def is_finite_cell(value: object) -> bool:
    return math.isfinite(finite_float(value, float("nan")))


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def registry_rows() -> list[dict[str, object]]:
    return [
        {
            "family_scope": "GIFT-Eval / Moirai",
            "paper_or_benchmark": "GIFT-Eval; Moirai / Uni2TS leaderboard-style evaluation",
            "primary_source": "https://arxiv.org/abs/2410.10393; https://arxiv.org/abs/2402.02592",
            "paper_faithful_metrics": "MASE[0.5]; mean weighted sum quantile loss; MSIS; sMAPE/ND/NRMSE where available",
            "current_exact_support": "Aggregate locked-slice exact MASE and WQL from GIFT-Eval result CSVs",
            "current_window_repair_support": "MASE reconstructed from stored baseline MASE scale; q10/q50/q90 WQL proxy; available-grid WQL when full forecast_q* columns are rerun; q10-q90 coverage",
            "main_claim_use": "robustness only",
        },
        {
            "family_scope": "Chronos / Chronos-Bolt",
            "paper_or_benchmark": "Chronos paper and Chronos-Bolt official benchmark artifacts",
            "primary_source": "https://arxiv.org/abs/2403.07815; https://github.com/amazon-science/chronos-forecasting",
            "paper_faithful_metrics": "WQL / quantile loss family; MASE; probabilistic calibration summaries",
            "current_exact_support": "Aggregate GIFT-style MASE/WQL where the locked scores contain them",
            "current_window_repair_support": "q10/q50/q90 WQL proxy; available-grid WQL after richer quantile reruns; q10-q90 coverage; MASE/RMSE/MAE point metrics",
            "main_claim_use": "robustness only",
        },
        {
            "family_scope": "TimesFM",
            "paper_or_benchmark": "TimesFM official reports and benchmark tables",
            "primary_source": "https://arxiv.org/abs/2310.10688; https://github.com/google-research/timesfm",
            "paper_faithful_metrics": "Point forecast metrics such as MAE, RMSE, sMAPE/MAPE-style normalized errors; benchmark-scaled point errors",
            "current_exact_support": "Aggregate GIFT-style MAE/RMSE/sMAPE/MAPE/ND/NRMSE in locked score CSVs where present",
            "current_window_repair_support": "Exact window MAE/RMSE/sMAPE/WAPE; MASE from stored scale; q10/q50/q90 proxy or available-grid WQL if richer quantiles are present",
            "main_claim_use": "robustness only",
        },
    ]


def window_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("model", ""),
        row.get("series_id", ""),
        row.get("origin", ""),
        str(row.get("window_index", "")),
    )


def repair_key_from_parts(
    family: object,
    source: object,
    dataset: object,
    series_id: object,
    origin: object,
    window_index: object,
) -> tuple[str, str, str, str, str, str]:
    return (
        str(family),
        str(source),
        str(dataset),
        str(series_id),
        str(origin),
        str(window_index),
    )


def metric_path_for_raw(raw_path: Path) -> Path:
    return ROOT / "results" / "window_metrics" / f"{raw_path.stem}_metrics.csv"


def metric_scale_rows(metric_path: Path) -> dict[tuple[str, str, str, str, str], dict[str, object]]:
    if not metric_path.exists():
        return {}
    rows: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    for row in read_csv(metric_path):
        baseline_mae = optional_float(row.get("baseline_mae"))
        baseline_mase = optional_float(row.get("baseline_mase"))
        scale = None
        if baseline_mae is not None and baseline_mase is not None and abs(baseline_mase) > EPS:
            scale = baseline_mae / baseline_mase
        rows[window_key(row)] = {
            "baseline_mase": baseline_mase,
            "model_mase": optional_float(row.get("mase")),
            "mase_scale": scale,
            "source_metric_path": str(metric_path.relative_to(ROOT)),
        }
    return rows


def load_windows() -> dict[tuple[str, str, str, str, str, str], dict[str, object]]:
    windows: dict[tuple[str, str, str, str, str, str], dict[str, object]] = {}
    for item in cross.INPUTS:
        raw_path = ROOT / str(item["raw"])
        if not raw_path.exists():
            continue
        raw_groups = base.raw_window_map(raw_path)
        scale_rows = metric_scale_rows(metric_path_for_raw(raw_path))
        for raw_key, rows in raw_groups.items():
            dataset, model_name, series_id, origin, window_index = raw_key
            scale_row = scale_rows.get(raw_key, {})
            actual = np.asarray([finite_float(row.get("actual")) for row in rows], dtype=float)
            model = np.asarray([finite_float(row.get("forecast_mean")) for row in rows], dtype=float)
            baseline = base.raw_baseline_values(rows)
            q10 = np.asarray([finite_float(row.get("forecast_q10")) for row in rows], dtype=float)
            q50 = np.asarray(
                [
                    finite_float(
                        row.get(
                            "forecast_q50",
                            row.get("forecast_median", row.get("forecast_mean")),
                        )
                    )
                    for row in rows
                ],
                dtype=float,
            )
            q90 = np.asarray([finite_float(row.get("forecast_q90")) for row in rows], dtype=float)
            try:
                quantile_levels, quantile_grid = quantile_matrix_from_rows(rows)
            except ValueError:
                quantile_levels = [0.1, 0.5, 0.9]
                quantile_grid = np.column_stack([q10, q50, q90])
            key = repair_key_from_parts(
                item["family"],
                item["source"],
                dataset,
                series_id,
                origin,
                window_index,
            )
            windows[key] = {
                "family": item["family"],
                "source": item["source"],
                "role": item["role"],
                "dataset": dataset,
                "model": model_name,
                "series_id": series_id,
                "origin": origin,
                "window_index": window_index,
                "domain": rows[0].get("domain", ""),
                "regime": rows[0].get("regime", ""),
                "horizon": len(rows),
                "actual": actual,
                "model_forecast": model,
                "baseline_forecast": baseline,
                "q10": q10,
                "q50": q50,
                "q90": q90,
                "quantile_levels": quantile_levels,
                "quantile_grid": quantile_grid,
                "mase_scale": scale_row.get("mase_scale"),
                "baseline_mase_from_metrics": scale_row.get("baseline_mase"),
                "model_mase_from_metrics": scale_row.get("model_mase"),
                "source_metric_path": scale_row.get("source_metric_path", ""),
                "raw_forecast_path": str(raw_path.relative_to(ROOT)),
            }
    return windows


def quantile_loss_proxy(actual: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> float:
    quantiles = np.column_stack([q10, q50, q90])
    return mean_weighted_quantile_loss(actual, quantiles, [0.1, 0.5, 0.9])


def ordered_quantiles(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stacked = np.vstack([q10, q50, q90])
    ordered = np.sort(stacked, axis=0)
    return ordered[0], ordered[1], ordered[2]


def mase_from_scale(error: float, scale: object) -> float:
    parsed = finite_float(scale, float("nan"))
    if not math.isfinite(parsed) or abs(parsed) <= EPS:
        return float("nan")
    return float(error / parsed)


def point_metric_block(actual: np.ndarray, forecast: np.ndarray, scale: object) -> dict[str, float]:
    mae_value = mae(actual, forecast)
    return {
        "mae": mae_value,
        "rmse": rmse(actual, forecast),
        "smape": smape(actual, forecast),
        "wape": wape(actual, forecast),
        "mase": mase_from_scale(mae_value, scale),
    }


def ratio(value: float, baseline: float) -> float:
    if not math.isfinite(value) or not math.isfinite(baseline):
        return float("nan")
    return relative_error_ratio(value, baseline)


def apply_repair_quantiles(
    q10: np.ndarray,
    q50: np.ndarray,
    q90: np.ndarray,
    baseline: np.ndarray,
    weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    repair_q10 = q10 + weight * (baseline - q10)
    repair_q50 = q50 + weight * (baseline - q50)
    repair_q90 = q90 + weight * (baseline - q90)
    return ordered_quantiles(repair_q10, repair_q50, repair_q90)


def apply_repair_quantile_grid(quantiles: np.ndarray, baseline: np.ndarray, weight: float) -> np.ndarray:
    repaired = quantiles + weight * (baseline[:, None] - quantiles)
    return np.sort(repaired, axis=1)


def quantile_grid_status(levels: list[float]) -> str:
    interior = [level for level in levels if 0.0 < float(level) < 1.0]
    if len(interior) > 3:
        return "full_grid_from_available_quantile_columns"
    if len(interior) == 3 and all(level in {0.1, 0.5, 0.9} for level in interior):
        return "proxy_from_available_three_quantiles"
    return "limited_available_quantile_grid"


def repair_metric_sources() -> list[Path]:
    sources = [OUT_DIR / "risk_calibrated_rr_cssr_windows.csv"]
    cpr_path = AAAI_OUT_DIR / "cpr_ltt_windows.csv"
    if cpr_path.exists():
        sources.append(cpr_path)
    return sources


def build_window_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    windows = load_windows()
    repair_rows: list[dict[str, str]] = []
    source_paths = repair_metric_sources()
    for path in source_paths:
        for row in read_csv(path):
            row = dict(row)
            row["repair_metric_source_path"] = str(path.relative_to(ROOT))
            repair_rows.append(row)
    output: list[dict[str, object]] = []
    missing = 0
    for repair_row in repair_rows:
        key = repair_key_from_parts(
            repair_row.get("family"),
            repair_row.get("source"),
            repair_row.get("dataset"),
            repair_row.get("series_id"),
            repair_row.get("origin"),
            repair_row.get("window_index"),
        )
        window = windows.get(key)
        if window is None:
            missing += 1
            continue
        actual = np.asarray(window["actual"], dtype=float)
        model = np.asarray(window["model_forecast"], dtype=float)
        baseline = np.asarray(window["baseline_forecast"], dtype=float)
        q10 = np.asarray(window["q10"], dtype=float)
        q50 = np.asarray(window["q50"], dtype=float)
        q90 = np.asarray(window["q90"], dtype=float)
        quantile_levels = [float(level) for level in window["quantile_levels"]]
        quantile_grid = np.asarray(window["quantile_grid"], dtype=float)
        weight = finite_float(repair_row.get("effective_weight"))
        repair_point = model + weight * (baseline - model)
        rq10, rq50, rq90 = apply_repair_quantiles(q10, q50, q90, baseline, weight)
        repair_quantile_grid = apply_repair_quantile_grid(quantile_grid, baseline, weight)
        baseline_quantile_grid = np.repeat(baseline[:, None], len(quantile_levels), axis=1)

        baseline_point = point_metric_block(actual, baseline, window["mase_scale"])
        model_point = point_metric_block(actual, model, window["mase_scale"])
        repair_point_metrics = point_metric_block(actual, repair_point, window["mase_scale"])

        baseline_wql = quantile_loss_proxy(actual, baseline, baseline, baseline)
        model_wql = quantile_loss_proxy(actual, q10, q50, q90)
        repair_wql = quantile_loss_proxy(actual, rq10, rq50, rq90)
        baseline_grid_wql = mean_weighted_quantile_loss(actual, baseline_quantile_grid, quantile_levels)
        model_grid_wql = mean_weighted_quantile_loss(actual, quantile_grid, quantile_levels)
        repair_grid_wql = mean_weighted_quantile_loss(actual, repair_quantile_grid, quantile_levels)
        grid_status = quantile_grid_status(quantile_levels)

        model_coverage = empirical_coverage(actual, q10, q90)
        repair_coverage = empirical_coverage(actual, rq10, rq90)
        model_coverage_error = abs(model_coverage - NOMINAL_Q10_Q90_COVERAGE)
        repair_coverage_error = abs(repair_coverage - NOMINAL_Q10_Q90_COVERAGE)

        row: dict[str, object] = {
            "strategy_id": repair_row["strategy_id"],
            "split_id": repair_row.get("split_id", ""),
            "config_id": repair_row.get("config_id", repair_row.get("selected_policy_id", "")),
            "selected_policy_id": repair_row.get("selected_policy_id", ""),
            "repair_metric_source_path": repair_row.get("repair_metric_source_path", ""),
            "family": window["family"],
            "source": window["source"],
            "role": window["role"],
            "dataset": window["dataset"],
            "domain": window["domain"],
            "regime": window["regime"],
            "model": window["model"],
            "series_id": window["series_id"],
            "origin": window["origin"],
            "window_index": window["window_index"],
            "horizon": window["horizon"],
            "effective_weight": weight,
            "gate_active": repair_row.get("gate_active", ""),
            "shield_active": repair_row.get("shield_active", ""),
            "conflict_override": repair_row.get("conflict_override", ""),
            "source_metric_path": window["source_metric_path"],
            "raw_forecast_path": window["raw_forecast_path"],
            "mase_scale_available": int(math.isfinite(finite_float(window["mase_scale"], float("nan")))),
            "model_coverage_q10_q90": model_coverage,
            "repair_coverage_q10_q90": repair_coverage,
            "model_coverage_abs_error_q10_q90": model_coverage_error,
            "repair_coverage_abs_error_q10_q90": repair_coverage_error,
            "coverage_abs_error_delta_vs_model": repair_coverage_error - model_coverage_error,
            "model_interval_width_q10_q90": float(np.mean(q90 - q10)),
            "repair_interval_width_q10_q90": float(np.mean(rq90 - rq10)),
            "quantile_grid_n_levels": len(quantile_levels),
            "quantile_grid_levels": ";".join(f"{level:.4g}" for level in quantile_levels),
            "wql_quantile_grid_artifact_status": grid_status,
            "baseline_wql_proxy_q10_q50_q90": baseline_wql,
            "model_wql_proxy_q10_q50_q90": model_wql,
            "repair_wql_proxy_q10_q50_q90": repair_wql,
            "model_wql_proxy_q10_q50_q90_rer": ratio(model_wql, baseline_wql),
            "repair_wql_proxy_q10_q50_q90_rer": ratio(repair_wql, baseline_wql),
            "wql_proxy_q10_q50_q90_rer_delta_vs_model": ratio(repair_wql, baseline_wql)
            - ratio(model_wql, baseline_wql),
            "baseline_wql_quantile_grid": baseline_grid_wql,
            "model_wql_quantile_grid": model_grid_wql,
            "repair_wql_quantile_grid": repair_grid_wql,
            "model_wql_quantile_grid_rer": ratio(model_grid_wql, baseline_grid_wql),
            "repair_wql_quantile_grid_rer": ratio(repair_grid_wql, baseline_grid_wql),
            "wql_quantile_grid_rer_delta_vs_model": ratio(repair_grid_wql, baseline_grid_wql)
            - ratio(model_grid_wql, baseline_grid_wql),
        }
        for metric_id, baseline_value in baseline_point.items():
            model_value = model_point[metric_id]
            repair_value = repair_point_metrics[metric_id]
            row[f"baseline_{metric_id}"] = baseline_value
            row[f"model_{metric_id}"] = model_value
            row[f"repair_{metric_id}"] = repair_value
            row[f"model_{metric_id}_rer"] = ratio(model_value, baseline_value)
            row[f"repair_{metric_id}_rer"] = ratio(repair_value, baseline_value)
            row[f"{metric_id}_rer_delta_vs_model"] = ratio(repair_value, baseline_value) - ratio(
                model_value,
                baseline_value,
            )
            row[f"model_{metric_id}_failure_delta_005"] = int(ratio(model_value, baseline_value) > 1.05)
            row[f"repair_{metric_id}_failure_delta_005"] = int(ratio(repair_value, baseline_value) > 1.05)
        row["model_wql_proxy_q10_q50_q90_failure_delta_005"] = int(
            row["model_wql_proxy_q10_q50_q90_rer"] > 1.05
        )
        row["repair_wql_proxy_q10_q50_q90_failure_delta_005"] = int(
            row["repair_wql_proxy_q10_q50_q90_rer"] > 1.05
        )
        row["model_wql_quantile_grid_failure_delta_005"] = int(row["model_wql_quantile_grid_rer"] > 1.05)
        row["repair_wql_quantile_grid_failure_delta_005"] = int(row["repair_wql_quantile_grid_rer"] > 1.05)
        output.append(row)
    status = {
        "n_cross_windows": len(windows),
        "n_repair_strategy_rows": len(repair_rows),
        "n_robustness_rows": len(output),
        "n_missing_repair_rows": missing,
        "quantile_grid_level_counts": sorted(
            {int(row.get("quantile_grid_n_levels", 0)) for row in output if int(row.get("quantile_grid_n_levels", 0)) > 0}
        ),
        "n_full_grid_quantile_rows": sum(
            1 for row in output if row.get("wql_quantile_grid_artifact_status") == "full_grid_from_available_quantile_columns"
        ),
        "repair_metric_sources": [str(path.relative_to(ROOT)) for path in source_paths],
    }
    return output, status


METRIC_SPECS = [
    {
        "metric_id": "mae",
        "label": "MAE-RER",
        "paper_scope": "Unified main metric; TimesFM point robustness",
        "artifact_status": "exact_window",
        "model_ratio_col": "model_mae_rer",
        "repair_ratio_col": "repair_mae_rer",
        "model_value_col": "model_mae",
        "repair_value_col": "repair_mae",
        "model_failure_col": "model_mae_failure_delta_005",
        "repair_failure_col": "repair_mae_failure_delta_005",
        "lower_is_better": True,
    },
    {
        "metric_id": "rmse",
        "label": "RMSE-RER",
        "paper_scope": "TimesFM/GIFT-style point robustness",
        "artifact_status": "exact_window",
        "model_ratio_col": "model_rmse_rer",
        "repair_ratio_col": "repair_rmse_rer",
        "model_value_col": "model_rmse",
        "repair_value_col": "repair_rmse",
        "model_failure_col": "model_rmse_failure_delta_005",
        "repair_failure_col": "repair_rmse_failure_delta_005",
        "lower_is_better": True,
    },
    {
        "metric_id": "mase",
        "label": "MASE-RER",
        "paper_scope": "GIFT-Eval/Moirai/Chronos scale-normalized robustness",
        "artifact_status": "reconstructed_from_stored_baseline_mase_scale",
        "model_ratio_col": "model_mase_rer",
        "repair_ratio_col": "repair_mase_rer",
        "model_value_col": "model_mase",
        "repair_value_col": "repair_mase",
        "model_failure_col": "model_mase_failure_delta_005",
        "repair_failure_col": "repair_mase_failure_delta_005",
        "lower_is_better": True,
    },
    {
        "metric_id": "smape",
        "label": "sMAPE-RER",
        "paper_scope": "TimesFM/GIFT-style point robustness",
        "artifact_status": "exact_window",
        "model_ratio_col": "model_smape_rer",
        "repair_ratio_col": "repair_smape_rer",
        "model_value_col": "model_smape",
        "repair_value_col": "repair_smape",
        "model_failure_col": "model_smape_failure_delta_005",
        "repair_failure_col": "repair_smape_failure_delta_005",
        "lower_is_better": True,
    },
    {
        "metric_id": "wape",
        "label": "WAPE-RER",
        "paper_scope": "TimesFM point robustness / normalized absolute error proxy",
        "artifact_status": "exact_window",
        "model_ratio_col": "model_wape_rer",
        "repair_ratio_col": "repair_wape_rer",
        "model_value_col": "model_wape",
        "repair_value_col": "repair_wape",
        "model_failure_col": "model_wape_failure_delta_005",
        "repair_failure_col": "repair_wape_failure_delta_005",
        "lower_is_better": True,
    },
    {
        "metric_id": "wql_proxy_q10_q50_q90",
        "label": "WQL-proxy-RER(q10/q50/q90)",
        "paper_scope": "Chronos/Moirai/GIFT quantile-loss robustness",
        "artifact_status": "proxy_from_available_three_quantiles",
        "model_ratio_col": "model_wql_proxy_q10_q50_q90_rer",
        "repair_ratio_col": "repair_wql_proxy_q10_q50_q90_rer",
        "model_value_col": "model_wql_proxy_q10_q50_q90",
        "repair_value_col": "repair_wql_proxy_q10_q50_q90",
        "model_failure_col": "model_wql_proxy_q10_q50_q90_failure_delta_005",
        "repair_failure_col": "repair_wql_proxy_q10_q50_q90_failure_delta_005",
        "lower_is_better": True,
    },
    {
        "metric_id": "wql_quantile_grid",
        "label": "WQL-RER(available quantile grid)",
        "paper_scope": "Chronos/Moirai/GIFT quantile-loss robustness; exact only when full-grid columns are present",
        "artifact_status": "dynamic_from_quantile_grid_columns",
        "artifact_status_col": "wql_quantile_grid_artifact_status",
        "model_ratio_col": "model_wql_quantile_grid_rer",
        "repair_ratio_col": "repair_wql_quantile_grid_rer",
        "model_value_col": "model_wql_quantile_grid",
        "repair_value_col": "repair_wql_quantile_grid",
        "model_failure_col": "model_wql_quantile_grid_failure_delta_005",
        "repair_failure_col": "repair_wql_quantile_grid_failure_delta_005",
        "lower_is_better": True,
    },
    {
        "metric_id": "coverage_abs_error_q10_q90",
        "label": "Coverage error(q10-q90 vs 0.80)",
        "paper_scope": "probabilistic calibration robustness",
        "artifact_status": "proxy_from_available_interval",
        "model_ratio_col": "",
        "repair_ratio_col": "",
        "model_value_col": "model_coverage_abs_error_q10_q90",
        "repair_value_col": "repair_coverage_abs_error_q10_q90",
        "model_failure_col": "",
        "repair_failure_col": "",
        "lower_is_better": True,
    },
]


def group_specs(rows: list[dict[str, object]]) -> list[tuple[str, str, list[dict[str, object]]]]:
    specs: list[tuple[str, str, list[dict[str, object]]]] = [("overall", "overall", rows)]
    for role in sorted({str(row["role"]) for row in rows}):
        specs.append((f"role:{role}", "role", [row for row in rows if row["role"] == role]))
    for family in sorted({str(row["family"]) for row in rows}):
        specs.append((f"family:{family}", "family", [row for row in rows if row["family"] == family]))
    for family in sorted({str(row["family"]) for row in rows}):
        for role in sorted({str(row["role"]) for row in rows if row["family"] == family}):
            subset = [row for row in rows if row["family"] == family and row["role"] == role]
            specs.append((f"family:{family}|role:{role}", "family_role", subset))
    return specs


def consistency_label(
    *,
    failure_reduction: float | None,
    mean_delta: float,
    median_delta: float,
    lower_is_better: bool,
) -> str:
    if not lower_is_better:
        return "reported_only"
    improves_mean = mean_delta < -EPS
    improves_median = median_delta < -EPS
    if failure_reduction is not None and math.isfinite(failure_reduction):
        if failure_reduction > 0.01 and (improves_mean or improves_median):
            return "consistent_improvement"
        if failure_reduction > 0.01:
            return "failure_reduction_but_mean_mixed"
        if failure_reduction < -0.01:
            return "inconsistent_worse"
    if improves_mean and improves_median:
        return "weak_consistent_no_threshold_gain"
    if mean_delta > EPS and median_delta > EPS:
        return "inconsistent_worse"
    return "mixed_or_neutral"


def summarize_metric_group(
    rows: list[dict[str, object]],
    strategy_id: str,
    group: str,
    group_type: str,
    spec: dict[str, object],
) -> dict[str, object]:
    model_values = [finite_float(row[spec["model_value_col"]], float("nan")) for row in rows]
    repair_values = [finite_float(row[spec["repair_value_col"]], float("nan")) for row in rows]
    deltas = [repair - model for repair, model in zip(repair_values, model_values, strict=True)]
    model_ratio_col = str(spec["model_ratio_col"])
    repair_ratio_col = str(spec["repair_ratio_col"])
    model_ratios = [finite_float(row[model_ratio_col], float("nan")) for row in rows] if model_ratio_col else []
    repair_ratios = [finite_float(row[repair_ratio_col], float("nan")) for row in rows] if repair_ratio_col else []
    ratio_deltas = [
        repair - model
        for repair, model in zip(repair_ratios, model_ratios, strict=True)
        if math.isfinite(repair) and math.isfinite(model)
    ]
    failure_reduction: float | None = None
    if spec["model_failure_col"] and spec["repair_failure_col"]:
        model_fail = [int(finite_float(row[spec["model_failure_col"]])) for row in rows]
        repair_fail = [int(finite_float(row[spec["repair_failure_col"]])) for row in rows]
        failure_reduction = rate(model_fail) - rate(repair_fail)
    win_rate = rate(
        [
            int(repair < model)
            for repair, model in zip(repair_values, model_values, strict=True)
            if math.isfinite(repair) and math.isfinite(model)
        ]
    )
    mean_delta = mean(deltas)
    median_delta = median(deltas)
    mean_consistency_delta = mean(ratio_deltas) if ratio_deltas else mean_delta
    median_consistency_delta = median(ratio_deltas) if ratio_deltas else median_delta
    consistency = consistency_label(
        failure_reduction=failure_reduction,
        mean_delta=mean_consistency_delta,
        median_delta=median_consistency_delta,
        lower_is_better=bool(spec["lower_is_better"]),
    )
    if str(spec["metric_id"]).startswith("wql_proxy"):
        model_median_ratio = median(model_ratios)
        repair_median_ratio = median(repair_ratios)
        if win_rate < 0.33 or (
            math.isfinite(model_median_ratio)
            and math.isfinite(repair_median_ratio)
            and repair_median_ratio > model_median_ratio + EPS
        ):
            consistency = "mixed_probabilistic_not_strong"
    if str(spec["metric_id"]) == "wql_quantile_grid":
        model_median_ratio = median(model_ratios)
        repair_median_ratio = median(repair_ratios)
        if win_rate < 0.33 or (
            math.isfinite(model_median_ratio)
            and math.isfinite(repair_median_ratio)
            and repair_median_ratio > model_median_ratio + EPS
        ):
            consistency = "mixed_probabilistic_not_strong"
    if str(spec["metric_id"]).startswith("coverage_abs_error") and win_rate < 0.33:
        consistency = "mixed_calibration_not_strong"
    artifact_status = str(spec["artifact_status"])
    artifact_status_col = str(spec.get("artifact_status_col", ""))
    if artifact_status_col:
        statuses = sorted({str(row.get(artifact_status_col, "")) for row in rows if row.get(artifact_status_col, "")})
        if len(statuses) == 1:
            artifact_status = statuses[0]
        elif statuses:
            artifact_status = "mixed:" + ",".join(statuses)
    return {
        "strategy_id": strategy_id,
        "group": group,
        "group_type": group_type,
        "metric_id": spec["metric_id"],
        "metric_label": spec["label"],
        "paper_scope": spec["paper_scope"],
        "artifact_status": artifact_status,
        "n_windows": len(rows),
        "model_mean": mean(model_values),
        "repair_mean": mean(repair_values),
        "model_median": median(model_values),
        "repair_median": median(repair_values),
        "mean_delta_vs_model": mean_delta,
        "median_delta_vs_model": median_delta,
        "p90_delta_vs_model": p90(deltas),
        "model_median_rer": median(model_ratios),
        "repair_median_rer": median(repair_ratios),
        "mean_rer_delta_vs_model": mean(ratio_deltas),
        "median_rer_delta_vs_model": median(ratio_deltas),
        "p90_rer_delta_vs_model": p90(ratio_deltas),
        "failure_rate_reduction_delta005": failure_reduction if failure_reduction is not None else "",
        "repair_win_rate_vs_model": win_rate,
        "consistency": consistency,
    }


def build_summary_rows(window_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for strategy_id in sorted({str(row["strategy_id"]) for row in window_rows}):
        strategy_rows = [row for row in window_rows if row["strategy_id"] == strategy_id]
        for group, group_type, group_rows in group_specs(strategy_rows):
            if not group_rows:
                continue
            for spec in METRIC_SPECS:
                summaries.append(summarize_metric_group(group_rows, strategy_id, group, group_type, spec))
    return summaries


def build_gift_exact_summary() -> list[dict[str, object]]:
    path = ROOT / "results" / "failure_mining" / "gift_eval_aggregate_failures.csv"
    if not path.exists():
        return []
    rows = read_csv(path)
    output: list[dict[str, object]] = []
    for model in sorted({row.get("local_model_key", "") for row in rows}):
        subset = [row for row in rows if row.get("local_model_key", "") == model]
        for metric_id, label, ratio_col in [
            ("gift_exact_mase", "GIFT exact MASE-RER", "mase_relative_error_ratio"),
            ("gift_exact_wql", "GIFT exact WQL-RER", "wql_relative_error_ratio"),
        ]:
            ratios = [finite_float(row.get(ratio_col), float("nan")) for row in subset]
            failures = [int(value > 1.05) for value in ratios if math.isfinite(value)]
            output.append(
                {
                    "model": model,
                    "metric_id": metric_id,
                    "metric_label": label,
                    "n_aggregate_rows": len([value for value in ratios if math.isfinite(value)]),
                    "median_ratio": median(ratios),
                    "mean_ratio": mean(ratios),
                    "failure_rate_delta005": rate(failures),
                    "artifact_status": "exact_gift_eval_aggregate_no_repair",
                }
            )
    return output


def rows_for_report(summary_rows: list[dict[str, object]], strategy_id: str = PRIMARY_STRATEGY) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for metric_id in [
        "mae",
        "rmse",
        "mase",
        "smape",
        "wape",
        "wql_proxy_q10_q50_q90",
        "wql_quantile_grid",
        "coverage_abs_error_q10_q90",
    ]:
        row = next(
            (
                item
                for item in summary_rows
                if item["strategy_id"] == strategy_id
                and item["group"] == "overall"
                and item["metric_id"] == metric_id
            ),
            None,
        )
        if row is None:
            continue
        selected.append(
            {
                "Metric": row["metric_label"],
                "Status": row["artifact_status"],
                "Model": num(row["model_median_rer"])
                if is_finite_cell(row["model_median_rer"])
                else num(row["model_median"]),
                "Repair": num(row["repair_median_rer"])
                if is_finite_cell(row["repair_median_rer"])
                else num(row["repair_median"]),
                "Delta": num(row["median_rer_delta_vs_model"])
                if is_finite_cell(row["median_rer_delta_vs_model"])
                else num(row["median_delta_vs_model"]),
                "FailRed": ""
                if row["failure_rate_reduction_delta005"] == ""
                else pct(row["failure_rate_reduction_delta005"]),
                "Win": pct(row["repair_win_rate_vs_model"]),
                "Consistency": row["consistency"],
            }
        )
    return selected


def family_rows_for_report(summary_rows: list[dict[str, object]], strategy_id: str = PRIMARY_STRATEGY) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for family in ["chronos", "moirai", "timesfm"]:
        for metric_id in ["mae", "rmse", "wql_proxy_q10_q50_q90", "wql_quantile_grid", "coverage_abs_error_q10_q90"]:
            row = next(
                (
                    item
                    for item in summary_rows
                    if item["strategy_id"] == strategy_id
                    and item["group"] == f"family:{family}"
                    and item["metric_id"] == metric_id
                ),
                None,
            )
            if row is None:
                continue
            selected.append(
                {
                    "Family": family,
                    "Metric": row["metric_label"],
                    "Delta": num(row["median_rer_delta_vs_model"])
                    if is_finite_cell(row["median_rer_delta_vs_model"])
                    else num(row["median_delta_vs_model"]),
                    "FailRed": ""
                    if row["failure_rate_reduction_delta005"] == ""
                    else pct(row["failure_rate_reduction_delta005"]),
                    "Consistency": row["consistency"],
                }
            )
    return selected


def gift_rows_for_report(gift_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    for row in gift_rows:
        selected.append(
            {
                "Model": row["model"],
                "Metric": row["metric_label"],
                "Rows": row["n_aggregate_rows"],
                "Median": num(row["median_ratio"]),
                "FailRate": pct(row["failure_rate_delta005"]),
            }
        )
    return selected[:12]


def interval_head_rows_for_report() -> list[dict[str, object]]:
    path = AAAI_OUT_DIR / "cpr_interval_recalibration_summary.csv"
    if not path.exists():
        return []
    rows = read_csv(path)
    selected: list[dict[str, object]] = []
    for strategy_id in [
        "cpr_shifted_quantiles",
        "cpr_width_preserve_s1_00",
        "cpr_width_preserve_s1_25",
        "cpr_width_calibrated_balanced",
        "cpr_width_calibrated_coverage",
    ]:
        row = next(
            (
                item
                for item in rows
                if item["strategy_id"] == strategy_id
                and item["split_protocol"] == "leave_source"
                and item["group"] == "overall"
            ),
            None,
        )
        if row is None:
            continue
        selected.append(
            {
                "Strategy": strategy_id,
                "Coverage": num(row["repair_mean_coverage"]),
                "CovErr": num(row["repair_mean_coverage_abs_error"]),
                "CovErrRed": pct(row["coverage_abs_error_relative_reduction_vs_shifted"]),
                "WQL-RER": num(row["repair_median_wql_proxy_rer"]),
                "dWQL": num(row["median_wql_proxy_rer_delta_vs_shifted"]),
                "WQLFailDelta": pct(row["wql_proxy_failure_delta_vs_shifted"]),
            }
        )
    return selected


def write_report(
    summary_rows: list[dict[str, object]],
    gift_rows: list[dict[str, object]],
    status: dict[str, object],
) -> None:
    main_rows = rows_for_report(summary_rows)
    by_family = family_rows_for_report(summary_rows)
    cpr_rows = rows_for_report(summary_rows, CPR_STRATEGY)
    cpr_by_family = family_rows_for_report(summary_rows, CPR_STRATEGY)
    gift_report_rows = gift_rows_for_report(gift_rows)
    interval_head_rows = interval_head_rows_for_report()
    probabilistic_not_strong = any(
        row["Metric"].startswith("WQL") and row["Consistency"] != "consistent_improvement"
        for row in main_rows
    )
    coverage_not_strong = any(
        row["Metric"].startswith("Coverage error") and row["Consistency"] != "consistent_improvement"
        for row in main_rows
    )
    caveat = (
        "On the current artifacts, at least one probabilistic robustness metric is mixed rather than directionally strong; "
        "therefore the repair claim should remain: it mainly improves point MAE-style failure and does not "
        "necessarily improve distribution metrics."
        if probabilistic_not_strong or coverage_not_strong
        else "On the current artifacts, the primary point metrics improve and the available probabilistic proxies do not overturn the claim; keep them as robustness, not the headline."
    )
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Paper-Faithful Metric Robustness",
                "",
                "## Claim Boundary",
                "",
                "Main paper claim: use unified MAE-RER to discover and repair the low-local-structure failure family across model families and datasets.",
                "",
                "Robustness claim: check whether the same direction holds under metrics closer to the original papers and benchmarks. If a robustness metric disagrees, report it as a boundary condition rather than changing the headline post hoc.",
                "",
                "## Metric Registry",
                "",
                markdown_table(
                    [
                        {
                            "Scope": row["family_scope"],
                            "Metrics": row["paper_faithful_metrics"],
                            "Support": row["current_window_repair_support"],
                        }
                        for row in registry_rows()
                    ],
                    [("Scope", "Scope"), ("Metrics", "Paper/benchmark metrics"), ("Support", "Current repair support")],
                ),
                "",
                "## Primary Strategy Robustness",
                "",
                f"Strategy: `{PRIMARY_STRATEGY}` over {status['n_cross_windows']} locked cross-family windows. For RER metrics the table shows median RER; for coverage it shows median absolute error from the nominal q10-q90 80% interval. Lower values are better for every row below.",
                "",
                markdown_table(
                    main_rows,
                    [
                        ("Metric", "Metric"),
                        ("Status", "Artifact status"),
                        ("Model", "Model median"),
                        ("Repair", "Repair median"),
                        ("Delta", "Median delta"),
                        ("FailRed", "Failure reduction"),
                        ("Win", "Win rate"),
                        ("Consistency", "Consistency"),
                    ],
                ),
                "",
                "## CPR Strategy Robustness",
                "",
                f"Strategy: `{CPR_STRATEGY}`. This table evaluates the LTT-certified CPR policy using the same paper-faithful metric layer, so the new method is not judged only on MAE-RER.",
                "",
                markdown_table(
                    cpr_rows,
                    [
                        ("Metric", "Metric"),
                        ("Status", "Artifact status"),
                        ("Model", "Model median"),
                        ("Repair", "Repair median"),
                        ("Delta", "Median delta"),
                        ("FailRed", "Failure reduction"),
                        ("Win", "Win rate"),
                        ("Consistency", "Consistency"),
                    ],
                ),
                "",
                "## CPR Interval Head Follow-Up",
                "",
                "The CPR point policy has a separate interval-head follow-up because shifted quantiles can collapse q10-q90 width when the point-repair weight is large. These rows are still q10/q50/q90 proxy metrics, but they evaluate a mechanism-aware location/uncertainty decoupling rather than another point blend.",
                "",
                markdown_table(
                    interval_head_rows,
                    [
                        ("Strategy", "Strategy"),
                        ("Coverage", "Mean coverage"),
                        ("CovErr", "Mean cov. error"),
                        ("CovErrRed", "CovErr red. vs shifted"),
                        ("WQL-RER", "Median WQL-RER"),
                        ("dWQL", "dWQL-RER vs shifted"),
                        ("WQLFailDelta", "WQL fail-rate delta"),
                    ],
                )
                if interval_head_rows
                else "Not run yet; see `scripts/run_cpr_interval_recalibration_goal.py`.",
                "",
                "## Family Breakdown",
                "",
                markdown_table(
                    by_family,
                    [
                        ("Family", "Family"),
                        ("Metric", "Metric"),
                        ("Delta", "Median delta"),
                        ("FailRed", "Failure reduction"),
                        ("Consistency", "Consistency"),
                    ],
                ),
                "",
                "## CPR Family Breakdown",
                "",
                markdown_table(
                    cpr_by_family,
                    [
                        ("Family", "Family"),
                        ("Metric", "Metric"),
                        ("Delta", "Median delta"),
                        ("FailRed", "Failure reduction"),
                        ("Consistency", "Consistency"),
                    ],
                ),
                "",
                "## Exact GIFT-Eval Aggregate Check",
                "",
                "These rows use the existing locked aggregate GIFT-Eval result CSVs and are exact for aggregate MASE/WQL ratios, but they do not evaluate repair because aggregate files do not contain repaired forecasts.",
                "",
                markdown_table(
                    gift_report_rows,
                    [
                        ("Model", "Model"),
                        ("Metric", "Metric"),
                        ("Rows", "Rows"),
                        ("Median", "Median ratio"),
                        ("FailRate", "Failure rate"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                caveat,
                "",
                "MASE-RER is mathematically tied to MAE-RER at the window level once both forecasts share the same stored MASE scale, so it is a paper-faithful scale-normalized view but not independent evidence from MAE. The independent stress tests here are RMSE/sMAPE/WAPE, the q10/q50/q90 WQL proxy, the available-grid WQL row when richer quantile columns exist, and q10-q90 coverage error.",
                "",
                "## Limitations",
                "",
                "- Exact full-paper WQL/CRPS/MSIS cannot be reconstructed from current raw repair artifacts because the existing rerun artifacts mostly persist only q10/q50/q90, not the full quantile grid or sample paths. The code path now detects richer forecast_q* columns and promotes WQL to available-grid status after reruns.",
                "- Repair quantiles are post-hoc residual-shifted toward the deterministic reference forecast; this is a robustness probe, not a learned probabilistic calibration method.",
                "- TimesFM robustness is strongest for point metrics; any probabilistic TimesFM row is proxy-only unless richer quantile artifacts are regenerated.",
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{REGISTRY_OUT.relative_to(ROOT)}`",
                f"- `{GIFT_EXACT_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    registry = registry_rows()
    window_rows, status = build_window_rows()
    summary = build_summary_rows(window_rows)
    gift_exact = build_gift_exact_summary()
    write_csv(REGISTRY_OUT, registry)
    write_csv(WINDOW_OUT, window_rows)
    write_csv(SUMMARY_OUT, summary)
    if gift_exact:
        write_csv(GIFT_EXACT_OUT, gift_exact)
    status.update(
        {
            "status": "ok",
            "timestamp": int(time.time()),
            "primary_strategy": PRIMARY_STRATEGY,
            "registry": str(REGISTRY_OUT.relative_to(ROOT)),
            "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
            "summary": str(SUMMARY_OUT.relative_to(ROOT)),
            "gift_exact_summary": str(GIFT_EXACT_OUT.relative_to(ROOT)) if gift_exact else "",
            "report": str(DOC_PATH.relative_to(ROOT)),
            "limitations": [
                "existing rerun artifacts are mostly q10/q50/q90; full WQL/CRPS/MSIS require richer forecast artifacts",
                "available-grid WQL is computed automatically when richer forecast_q* columns are present",
                "MASE repair values are reconstructed from stored baseline MASE scales",
                "distribution-metric repair claim is robustness-only, not headline",
            ],
        }
    )
    write_report(summary, gift_exact, status)
    STATUS_OUT.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
