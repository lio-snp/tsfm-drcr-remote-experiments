#!/usr/bin/env python
"""Build failure-family evidence and a gated baseline-mixture repair."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.failure_selection import infer_raw_path, read_metric_rows  # noqa: E402
from low_snr_tsfm.metrics import (  # noqa: E402
    empirical_coverage,
    flatness_score,
    forecast_variance_ratio,
    mae,
    pinball_loss,
    prediction_amplitude_ratio,
    relative_error_ratio,
    rmse,
    spike_recall,
)
from low_snr_tsfm.repair import blended_interval, convex_mixture, hull_interval  # noqa: E402


METRIC_INPUTS = [
    "results/window_metrics/chronos_bolt_small_bizitobs_application_short_auto_arima_metrics.csv",
    "results/window_metrics/chronos_bolt_small_covid_deaths_short_auto_ets_metrics.csv",
    "results/window_metrics/chronos_bolt_small_solar_short_seasonal_naive_metrics.csv",
    "results/window_metrics/chronos_bolt_small_loop_seattle_short_seasonal_naive_metrics.csv",
    "results/window_metrics/chronos_bolt_small_finance_fred_stress_metrics.csv",
    "results/window_metrics/timesfm_2_5_m16_covid_deaths_short_auto_ets_metrics.csv",
    "results/window_metrics/moirai2_ctx1680_m12_covid_deaths_short_auto_ets_metrics.csv",
    "results/window_metrics/moirai2_ctx1680_solar_m16_solar_short_seasonal_naive_metrics.csv",
    "results/traffic/chronos_bolt_small_metr_la_traffic_bma_window_metrics.csv",
    "results/traffic/chronos_bolt_small_pems_bay_traffic_bma_window_metrics.csv",
]

FEATURE_INPUTS = [
    ("results/failure_mining/chronos_bolt_small_bizitobs_application_short_auto_arima_predictor_features.csv", "bizitobs_auto_arima"),
    ("results/failure_mining/chronos_bolt_small_covid_deaths_short_auto_ets_predictor_features.csv", "chronos_covid_auto_ets"),
    ("results/failure_mining/timesfm_2_5_m16_covid_deaths_short_auto_ets_predictor_features.csv", "timesfm_covid_auto_ets"),
    ("results/failure_mining/moirai2_ctx1680_m12_covid_deaths_short_auto_ets_predictor_features.csv", "moirai_covid_auto_ets"),
    ("results/failure_mining/chronos_bolt_small_solar_short_seasonal_naive_predictor_features.csv", "chronos_solar_seasonal_naive"),
    ("results/failure_mining/moirai2_ctx1680_solar_m16_solar_short_seasonal_naive_predictor_features.csv", "moirai_solar_seasonal_naive"),
    ("results/failure_mining/chronos_bolt_small_loop_seattle_short_seasonal_naive_predictor_features.csv", "chronos_loop_seattle"),
    ("results/failure_mining/chronos_bolt_small_finance_fred_stress_predictor_features.csv", "chronos_finance_fred"),
    ("results/failure_mining/chronos_bolt_small_metr_la_traffic_bma_predictor_features.csv", "metr_la_traffic"),
    ("results/failure_mining/chronos_bolt_small_pems_bay_traffic_bma_predictor_features.csv", "pems_bay_traffic"),
]

REPAIR_INPUTS = [
    {
        "raw": "results/raw_forecasts/chronos_bolt_small_bizitobs_application_short_auto_arima.csv",
        "features": "results/failure_mining/chronos_bolt_small_bizitobs_application_short_auto_arima_predictor_features.csv",
        "role": "failure_target",
    },
    {
        "raw": "results/raw_forecasts/chronos_bolt_small_covid_deaths_short_auto_ets.csv",
        "features": "results/failure_mining/chronos_bolt_small_covid_deaths_short_auto_ets_predictor_features.csv",
        "role": "failure_target",
    },
    {
        "raw": "results/raw_forecasts/chronos_bolt_small_finance_fred_stress.csv",
        "features": "results/failure_mining/chronos_bolt_small_finance_fred_stress_predictor_features.csv",
        "role": "stress_target",
    },
    {
        "raw": "results/raw_forecasts/chronos_bolt_small_solar_short_seasonal_naive.csv",
        "features": "results/failure_mining/chronos_bolt_small_solar_short_seasonal_naive_predictor_features.csv",
        "role": "positive_control",
    },
    {
        "raw": "results/raw_forecasts/chronos_bolt_small_loop_seattle_short_seasonal_naive.csv",
        "features": "results/failure_mining/chronos_bolt_small_loop_seattle_short_seasonal_naive_predictor_features.csv",
        "role": "positive_control",
    },
]

FACTOR_NAMES = [
    "seasonality_strength",
    "trend_strength",
    "horizon_context_ratio",
    "spike_frequency",
    "zero_ratio",
    "changepoint_density",
    "coefficient_of_variation",
    "spectral_entropy",
    "kurtosis_excess",
]


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


def raw_path_for_metric(path: Path) -> Path | None:
    inferred = infer_raw_path(path)
    if inferred:
        full = ROOT / inferred
        return full if full.exists() else None
    if path.parent.name == "traffic" and path.name.endswith("_window_metrics.csv"):
        full = path.with_name(f"{path.name.removesuffix('_window_metrics.csv')}_raw.csv")
        return full if full.exists() else None
    return None


def window_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("model", ""),
        row.get("series_id", ""),
        row.get("origin", ""),
        str(row.get("window_index", "")),
    )


def raw_window_map(raw_path: Path) -> dict[tuple[str, str, str, str, str], list[dict[str, str]]]:
    rows = read_csv(raw_path)
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(window_key(row), []).append(row)
    for key in groups:
        groups[key].sort(key=lambda row: finite_float(row.get("horizon_index")))
    return groups


def raw_baseline_values(rows: list[dict[str, str]]) -> np.ndarray:
    if rows and "baseline_forecast" in rows[0]:
        return np.asarray([finite_float(row.get("baseline_forecast")) for row in rows], dtype=float)
    if rows and "historical_mean" in rows[0]:
        return np.asarray([finite_float(row.get("historical_mean")) for row in rows], dtype=float)
    if rows and "bma_mean" in rows[0]:
        return np.asarray([finite_float(row.get("bma_mean")) for row in rows], dtype=float)
    raise ValueError("Raw rows do not contain a recognized baseline forecast column")


def normalize_metric_row(row: dict[str, str], raw_groups: dict[tuple[str, str, str, str, str], list[dict[str, str]]]) -> dict[str, object]:
    raw_rows = raw_groups.get(window_key(row), [])
    actual = np.asarray([finite_float(raw.get("actual")) for raw in raw_rows], dtype=float)
    model_forecast = np.asarray([finite_float(raw.get("forecast_mean")) for raw in raw_rows], dtype=float)
    q10 = np.asarray([finite_float(raw.get("forecast_q10")) for raw in raw_rows], dtype=float)
    q50 = np.asarray([finite_float(raw.get("forecast_q50"), finite_float(raw.get("forecast_mean"))) for raw in raw_rows], dtype=float)
    q90 = np.asarray([finite_float(raw.get("forecast_q90")) for raw in raw_rows], dtype=float)
    baseline = raw_baseline_values(raw_rows) if raw_rows else np.asarray([], dtype=float)

    model_mae = finite_float(row.get("mae", row.get("model_mae")))
    baseline_mae = finite_float(row.get("baseline_mae", row.get("historical_mae")))
    model_rmse = finite_float(row.get("rmse", row.get("model_rmse")))
    baseline_rmse = finite_float(row.get("baseline_rmse", row.get("historical_rmse")))
    if raw_rows:
        model_mae = mae(actual, model_forecast)
        baseline_mae = mae(actual, baseline)
        model_rmse = rmse(actual, model_forecast)
        baseline_rmse = rmse(actual, baseline)

    rer = relative_error_ratio(model_mae, baseline_mae)
    rmse_rer = relative_error_ratio(model_rmse, baseline_rmse)
    normalizer = float(np.mean(np.abs(actual))) if actual.size else 0.0
    pinball_q10 = pinball_loss(actual, q10, 0.1) if actual.size and q10.size else float("nan")
    pinball_q50 = pinball_loss(actual, q50, 0.5) if actual.size and q50.size else float("nan")
    pinball_q90 = pinball_loss(actual, q90, 0.9) if actual.size and q90.size else float("nan")
    wql_proxy = (pinball_q10 + pinball_q50 + pinball_q90) / (normalizer + 1e-12) if actual.size else float("nan")

    return {
        "dataset": row.get("dataset", ""),
        "domain": row.get("domain", ""),
        "regime": row.get("regime", ""),
        "model": row.get("model", ""),
        "baseline": row.get("baseline", row.get("baseline_mode", "")),
        "series_id": row.get("series_id", ""),
        "origin": row.get("origin", ""),
        "window_index": row.get("window_index", ""),
        "horizon": int(finite_float(row.get("horizon"), len(raw_rows))),
        "context_length": int(finite_float(row.get("context_length"))),
        "mae": model_mae,
        "baseline_mae": baseline_mae,
        "absolute_error_gap": model_mae - baseline_mae,
        "relative_error_ratio": rer,
        "rmse": model_rmse,
        "baseline_rmse": baseline_rmse,
        "rmse_relative_error_ratio": rmse_rer,
        "mase": optional_float(row.get("mase")),
        "baseline_mase": optional_float(row.get("baseline_mase")),
        "mase_relative_error_ratio": relative_error_ratio(
            finite_float(row.get("mase")),
            finite_float(row.get("baseline_mase")),
        )
        if row.get("mase") not in {None, ""} and row.get("baseline_mase") not in {None, ""}
        else None,
        "pinball_q10": pinball_q10,
        "pinball_q50": pinball_q50,
        "pinball_q90": pinball_q90,
        "wql_proxy_q10_q50_q90": wql_proxy,
        "empirical_coverage_90": empirical_coverage(actual, q10, q90) if actual.size and q10.size else optional_float(row.get("empirical_coverage_90")),
        "interval_width_q10_q90": float(np.mean(q90 - q10)) if q10.size and q90.size else None,
        "forecast_variance_ratio": forecast_variance_ratio(actual, model_forecast) if actual.size else optional_float(row.get("forecast_variance_ratio")),
        "prediction_amplitude_ratio": prediction_amplitude_ratio(actual, model_forecast) if actual.size else optional_float(row.get("prediction_amplitude_ratio")),
        "flatness_score": flatness_score(actual, model_forecast) if actual.size else optional_float(row.get("flatness_score")),
        "spike_recall": spike_recall(actual, model_forecast, k=3) if actual.size else optional_float(row.get("spike_recall")),
        "failure_delta_005": int(rer > 1.05),
        "denominator_fragile": int(baseline_mae < 1e-6),
        "source_metric_path": row.get("source_metric_path", ""),
        "raw_forecast_path": row.get("raw_forecast_path", ""),
    }


def build_multimetric_table(metric_paths: list[Path], out_dir: Path) -> tuple[Path, dict[str, object]]:
    metric_rows = read_metric_rows(metric_paths)
    raw_cache: dict[str, dict[tuple[str, str, str, str, str], list[dict[str, str]]]] = {}
    enriched = []
    for row in metric_rows:
        metric_path = Path(row["source_metric_path"])
        raw_path = raw_path_for_metric(ROOT / metric_path)
        if raw_path is None:
            continue
        row["raw_forecast_path"] = str(raw_path.relative_to(ROOT))
        cache_key = str(raw_path)
        if cache_key not in raw_cache:
            raw_cache[cache_key] = raw_window_map(raw_path)
        normalized = normalize_metric_row(row, raw_cache[cache_key])
        enriched.append(normalized)

    output = out_dir / "multimetric_failure_table.csv"
    write_csv(output, enriched)
    labels = np.asarray([int(row["failure_delta_005"]) for row in enriched], dtype=int)
    report = {
        "status": "ok",
        "n_rows": len(enriched),
        "n_metric_inputs": len(metric_paths),
        "failure_rate_delta_005": float(np.mean(labels)) if labels.size else 0.0,
        "denominator_fragile_rate": float(np.mean([int(row["denominator_fragile"]) for row in enriched])) if enriched else 0.0,
        "output": str(output.relative_to(ROOT)),
    }
    return output, report


def read_feature_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature_path, source in FEATURE_INPUTS:
        path = ROOT / feature_path
        if not path.exists():
            continue
        for row in read_csv(path):
            normalized: dict[str, object] = {
                "source": source,
                "dataset": row.get("dataset", source),
                "domain": row.get("domain", row.get("original_domain", "")),
                "regime": row.get("regime", ""),
                "series_id": row.get("series_id", ""),
                "window_index": row.get("window_index", ""),
                "failure_delta_005": int(finite_float(row.get("failure_delta_005")) > 0.5),
                "relative_error_ratio": finite_float(row.get("relative_error_ratio")),
                "empirical_coverage_90": optional_float(row.get("empirical_coverage_90")),
                "flatness_score": optional_float(row.get("flatness_score")),
            }
            for factor in FACTOR_NAMES:
                normalized[factor] = finite_float(row.get(factor))
            rows.append(normalized)
    return rows


def quantile_bins(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(np.quantile(arr, 1 / 3)), float(np.quantile(arr, 2 / 3))


def assign_bin(value: float, low_cut: float, high_cut: float) -> str:
    if value <= low_cut:
        return "low"
    if value <= high_cut:
        return "mid"
    return "high"


def mean_optional(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    return float(np.mean(finite)) if finite else None


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.median(finite)) if finite else float("nan")


def build_factor_analysis(out_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    rows = read_feature_rows()
    feature_table = out_dir / "factor_feature_table.csv"
    write_csv(feature_table, rows)

    analysis_rows: list[dict[str, object]] = []
    for factor in FACTOR_NAMES:
        values = [float(row[factor]) for row in rows]
        low_cut, high_cut = quantile_bins(values)
        for bin_name in ["low", "mid", "high"]:
            subset = [
                row
                for row in rows
                if assign_bin(float(row[factor]), low_cut, high_cut) == bin_name
            ]
            if not subset:
                continue
            labels = [int(row["failure_delta_005"]) for row in subset]
            analysis_rows.append(
                {
                    "factor": factor,
                    "bin": bin_name,
                    "low_cut": low_cut,
                    "high_cut": high_cut,
                    "n_windows": len(subset),
                    "failure_rate_delta_005": float(np.mean(labels)),
                    "median_relative_error_ratio": median([float(row["relative_error_ratio"]) for row in subset]),
                    "mean_relative_error_ratio": float(np.mean([float(row["relative_error_ratio"]) for row in subset])),
                    "mean_empirical_coverage_90": mean_optional([row.get("empirical_coverage_90") for row in subset]),
                    "mean_flatness_score": mean_optional([row.get("flatness_score") for row in subset]),
                    "risk_hypothesis": "low_is_risky"
                    if factor in {"seasonality_strength", "trend_strength"}
                    else "high_is_risky",
                }
            )
    factor_output = out_dir / "factor_analysis.csv"
    write_csv(factor_output, analysis_rows)
    return feature_table, factor_output, {
        "status": "ok",
        "n_feature_rows": len(rows),
        "n_factor_rows": len(analysis_rows),
        "feature_table": str(feature_table.relative_to(ROOT)),
        "factor_analysis": str(factor_output.relative_to(ROOT)),
    }


def synthetic_series(
    *,
    context_length: int,
    horizon: int,
    seasonality: float,
    noise: float,
    spike_size: float,
    decay_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    period = 24
    t_context = np.arange(-context_length, 0, dtype=float)
    t_future = np.arange(1, horizon + 1, dtype=float)
    level = 10.0
    context = level + seasonality * np.sin(2 * np.pi * t_context / period)
    future = level + seasonality * np.sin(2 * np.pi * t_future / period)
    context += rng.normal(0.0, noise, size=context_length)
    future += rng.normal(0.0, noise, size=horizon)
    if spike_size > 0:
        future[horizon // 3] += spike_size
    if decay_rate > 0:
        future *= np.exp(-decay_rate * t_future)
    return context, future


def proxy_forecasts(context: np.ndarray, horizon: int, seasonality: float) -> tuple[np.ndarray, np.ndarray]:
    period = 24
    steps = np.arange(1, horizon + 1, dtype=float)
    tail = context[-min(context.size, 24) :]
    if tail.size >= 2:
        x = np.arange(tail.size, dtype=float)
        slope = float(np.polyfit(x, tail, 1)[0])
    else:
        slope = 0.0
    structure = min(1.0, max(0.0, seasonality))
    model = context[-1] + (1.0 - structure) * slope * steps
    if context.size >= 2 * period and seasonality > 0:
        seasonal_tail = context[-period:]
        repeated = np.resize(seasonal_tail - np.mean(seasonal_tail), horizon)
        model = np.mean(seasonal_tail) + structure * repeated + (1.0 - structure) * (model - context[-1])
    if structure < 0.5:
        spurious = 0.25 * (1.0 - structure) * np.std(tail) * np.sin(2 * np.pi * steps / 6)
        model = model + spurious
    baseline = np.full(horizon, context[-1], dtype=float)
    if context.size >= period:
        baseline = np.resize(context[-period:], horizon)
    return model, baseline


def build_synthetic_ablation(out_dir: Path, seeds: int = 32) -> tuple[Path, Path, dict[str, object]]:
    horizon = 24
    base = {
        "context_length": 96,
        "seasonality": 0.5,
        "noise": 1.0,
        "spike_size": 0.0,
        "decay_rate": 0.0,
        "baseline_noise": 0.0,
    }
    sweeps = {
        "context_length": [24, 48, 96, 192],
        "seasonality": [0.0, 0.25, 0.5, 1.0, 2.0],
        "noise": [0.1, 0.5, 1.0, 2.0],
        "spike_size": [0.0, 2.0, 5.0, 10.0],
        "decay_rate": [0.0, 0.02, 0.05, 0.10],
        "baseline_noise": [1.0, 0.1, 0.01, 0.0],
    }
    metric_rows: list[dict[str, object]] = []
    for factor, values in sweeps.items():
        for value in values:
            params = dict(base)
            params[factor] = value
            for seed in range(seeds):
                context, actual = synthetic_series(
                    context_length=int(params["context_length"]),
                    horizon=horizon,
                    seasonality=float(params["seasonality"]),
                    noise=float(params["noise"]),
                    spike_size=float(params["spike_size"]),
                    decay_rate=float(params["decay_rate"]),
                    seed=10_000 + seed,
                )
                model, baseline = proxy_forecasts(context, horizon, float(params["seasonality"]))
                if factor == "baseline_noise":
                    rng = np.random.default_rng(20_000 + seed)
                    baseline = actual + rng.normal(0.0, float(params["baseline_noise"]), size=horizon)
                model_mae = mae(actual, model)
                baseline_mae = mae(actual, baseline)
                rer = relative_error_ratio(model_mae, baseline_mae)
                spread = np.std(context[-min(context.size, 48) :]) + 1e-6
                q10 = model - 1.64 * spread
                q90 = model + 1.64 * spread
                metric_rows.append(
                    {
                        "factor": factor,
                        "value": value,
                        "seed": seed,
                        "context_length": int(params["context_length"]),
                        "seasonality": float(params["seasonality"]),
                        "noise": float(params["noise"]),
                        "spike_size": float(params["spike_size"]),
                        "decay_rate": float(params["decay_rate"]),
                        "baseline_noise": float(params["baseline_noise"]),
                        "model_mae": model_mae,
                        "baseline_mae": baseline_mae,
                        "relative_error_ratio": rer,
                        "failure_delta_005": int(rer > 1.05),
                        "empirical_coverage_90": empirical_coverage(actual, q10, q90),
                        "forecast_variance_ratio": forecast_variance_ratio(actual, model),
                        "prediction_amplitude_ratio": prediction_amplitude_ratio(actual, model),
                        "flatness_score": flatness_score(actual, model),
                    }
                )
    summary_rows: list[dict[str, object]] = []
    for factor in sweeps:
        for value in sweeps[factor]:
            subset = [row for row in metric_rows if row["factor"] == factor and row["value"] == value]
            summary_rows.append(
                {
                    "factor": factor,
                    "value": value,
                    "n": len(subset),
                    "failure_rate_delta_005": float(np.mean([row["failure_delta_005"] for row in subset])),
                    "median_relative_error_ratio": median([float(row["relative_error_ratio"]) for row in subset]),
                    "mean_empirical_coverage_90": float(np.mean([float(row["empirical_coverage_90"]) for row in subset])),
                    "mean_forecast_variance_ratio": float(np.mean([float(row["forecast_variance_ratio"]) for row in subset])),
                    "mean_prediction_amplitude_ratio": float(np.mean([float(row["prediction_amplitude_ratio"]) for row in subset])),
                }
            )
    metrics_path = out_dir / "synthetic_ablation_metrics.csv"
    summary_path = out_dir / "synthetic_ablation_summary.csv"
    write_csv(metrics_path, metric_rows)
    write_csv(summary_path, summary_rows)
    return metrics_path, summary_path, {
        "status": "ok",
        "n_metric_rows": len(metric_rows),
        "n_summary_rows": len(summary_rows),
        "metrics": str(metrics_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)),
    }


def feature_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("dataset", ""), row.get("series_id", ""), str(row.get("window_index", "")))


def danger_gate(feature: dict[str, str]) -> tuple[bool, float, str]:
    season = finite_float(feature.get("seasonality_strength"))
    trend = finite_float(feature.get("trend_strength"))
    hcr = finite_float(feature.get("horizon_context_ratio"))
    spike = finite_float(feature.get("spike_frequency"))
    zero = finite_float(feature.get("zero_ratio"))
    change = finite_float(feature.get("changepoint_density"))
    cv = finite_float(feature.get("coefficient_of_variation"))
    zero_short = zero >= 0.20 and hcr >= 0.10
    low_structure_bursty = season < 0.20 and trend < 0.10 and (spike >= 0.022 or change >= 0.08)
    finance_like = cv >= 3.0 and season < 0.15
    if zero_short:
        return True, 0.75, "zero_inflated+short_context_relative_to_horizon"
    if finance_like:
        return True, 0.75, "high_cv+low_seasonality"
    if low_structure_bursty:
        return True, 0.5, "low_local_structure+bursty_or_changepoint"
    return False, 0.0, "safe"


def run_gated_repair(out_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    metric_rows: list[dict[str, object]] = []
    raw_rows_out: list[dict[str, object]] = []
    for repair_input in REPAIR_INPUTS:
        raw_path = ROOT / str(repair_input["raw"])
        feature_path = ROOT / str(repair_input["features"])
        raw_groups = raw_window_map(raw_path)
        feature_rows = {feature_key(row): row for row in read_csv(feature_path)}
        for key, rows in raw_groups.items():
            dataset, model_name, series_id, origin, window_index = key
            feature = feature_rows.get((dataset, series_id, window_index)) or feature_rows.get(("", series_id, window_index), {})
            gated, weight, reason = danger_gate(feature)
            actual = np.asarray([finite_float(row.get("actual")) for row in rows], dtype=float)
            model = np.asarray([finite_float(row.get("forecast_mean")) for row in rows], dtype=float)
            baseline = raw_baseline_values(rows)
            q10 = np.asarray([finite_float(row.get("forecast_q10")) for row in rows], dtype=float)
            q90 = np.asarray([finite_float(row.get("forecast_q90")) for row in rows], dtype=float)
            repaired = convex_mixture(model, baseline, weight)
            blend_q10, blend_q90 = blended_interval(q10, q90, baseline, weight)
            hull_q10, hull_q90 = hull_interval(q10, q90, baseline)
            model_mae = mae(actual, model)
            baseline_mae = mae(actual, baseline)
            repair_mae = mae(actual, repaired)
            model_rer = relative_error_ratio(model_mae, baseline_mae)
            repair_rer = relative_error_ratio(repair_mae, baseline_mae)
            metric_rows.append(
                {
                    "dataset": dataset,
                    "domain": rows[0].get("domain", ""),
                    "regime": rows[0].get("regime", ""),
                    "role": repair_input["role"],
                    "model": model_name,
                    "series_id": series_id,
                    "origin": origin,
                    "window_index": window_index,
                    "gate_active": int(gated),
                    "gate_weight": weight,
                    "gate_reason": reason,
                    "model_mae": model_mae,
                    "baseline_mae": baseline_mae,
                    "repair_mae": repair_mae,
                    "model_relative_error_ratio": model_rer,
                    "repair_relative_error_ratio": repair_rer,
                    "model_failure_delta_005": int(model_rer > 1.05),
                    "repair_failure_delta_005": int(repair_rer > 1.05),
                    "repair_improves_model": int(repair_mae < model_mae),
                    "repair_beats_baseline": int(repair_mae < baseline_mae),
                    "model_empirical_coverage_90": empirical_coverage(actual, q10, q90),
                    "repair_blend_empirical_coverage_90": empirical_coverage(actual, blend_q10, blend_q90),
                    "repair_hull_empirical_coverage_90": empirical_coverage(actual, hull_q10, hull_q90),
                    "model_forecast_variance_ratio": forecast_variance_ratio(actual, model),
                    "repair_forecast_variance_ratio": forecast_variance_ratio(actual, repaired),
                    "model_prediction_amplitude_ratio": prediction_amplitude_ratio(actual, model),
                    "repair_prediction_amplitude_ratio": prediction_amplitude_ratio(actual, repaired),
                    "model_flatness_score": flatness_score(actual, model),
                    "repair_flatness_score": flatness_score(actual, repaired),
                }
            )
            for row, repair_value, lo, hi in zip(rows, repaired, blend_q10, blend_q90):
                raw_rows_out.append(
                    {
                        "dataset": dataset,
                        "domain": row.get("domain", ""),
                        "regime": row.get("regime", ""),
                        "model": model_name,
                        "series_id": series_id,
                        "origin": origin,
                        "window_index": window_index,
                        "horizon_index": row.get("horizon_index", ""),
                        "actual": row.get("actual", ""),
                        "forecast_mean": row.get("forecast_mean", ""),
                        "baseline_forecast": baseline[int(finite_float(row.get("horizon_index"), 1)) - 1],
                        "repair_mean": float(repair_value),
                        "repair_q10": float(lo),
                        "repair_q90": float(hi),
                        "gate_active": int(gated),
                        "gate_weight": weight,
                        "gate_reason": reason,
                    }
                )
    summary_rows: list[dict[str, object]] = []
    for role in sorted({str(row["role"]) for row in metric_rows}):
        subset = [row for row in metric_rows if row["role"] == role]
        summary_rows.append(summarize_repair(subset, role))
    for dataset in sorted({str(row["dataset"]) for row in metric_rows}):
        subset = [row for row in metric_rows if row["dataset"] == dataset]
        summary_rows.append(summarize_repair(subset, f"dataset:{dataset}"))
    summary_rows.append(summarize_repair(metric_rows, "overall"))

    raw_path = out_dir / "failure_aware_mixture_raw.csv"
    metrics_path = out_dir / "failure_aware_mixture_metrics.csv"
    summary_path = out_dir / "failure_aware_mixture_summary.csv"
    write_csv(raw_path, raw_rows_out)
    write_csv(metrics_path, metric_rows)
    write_csv(summary_path, summary_rows)
    return metrics_path, summary_path, {
        "status": "ok",
        "n_windows": len(metric_rows),
        "gated_rate": float(np.mean([int(row["gate_active"]) for row in metric_rows])),
        "metrics": str(metrics_path.relative_to(ROOT)),
        "summary": str(summary_path.relative_to(ROOT)),
        "raw": str(raw_path.relative_to(ROOT)),
    }


def summarize_repair(rows: list[dict[str, object]], group: str) -> dict[str, object]:
    return {
        "group": group,
        "n_windows": len(rows),
        "gate_rate": float(np.mean([int(row["gate_active"]) for row in rows])) if rows else 0.0,
        "model_failure_rate_delta_005": float(np.mean([int(row["model_failure_delta_005"]) for row in rows])) if rows else 0.0,
        "repair_failure_rate_delta_005": float(np.mean([int(row["repair_failure_delta_005"]) for row in rows])) if rows else 0.0,
        "model_median_relative_error_ratio": median([float(row["model_relative_error_ratio"]) for row in rows]),
        "repair_median_relative_error_ratio": median([float(row["repair_relative_error_ratio"]) for row in rows]),
        "repair_win_rate_vs_model": float(np.mean([int(row["repair_improves_model"]) for row in rows])) if rows else 0.0,
        "model_mean_empirical_coverage_90": float(np.mean([float(row["model_empirical_coverage_90"]) for row in rows])) if rows else 0.0,
        "repair_blend_mean_empirical_coverage_90": float(np.mean([float(row["repair_blend_empirical_coverage_90"]) for row in rows])) if rows else 0.0,
        "repair_hull_mean_empirical_coverage_90": float(np.mean([float(row["repair_hull_empirical_coverage_90"]) for row in rows])) if rows else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/failure_family")
    parser.add_argument("--repair-dir", default="results/repair")
    parser.add_argument("--synthetic-seeds", type=int, default=32)
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    repair_dir = ROOT / args.repair_dir
    metric_paths = [ROOT / path for path in METRIC_INPUTS if (ROOT / path).exists()]
    if len(metric_paths) < 6:
        raise SystemExit("Too few metric inputs for failure-family evidence")

    multimetric_path, multimetric_report = build_multimetric_table(metric_paths, out_dir)
    feature_table, factor_path, factor_report = build_factor_analysis(out_dir)
    synthetic_metrics, synthetic_summary, synthetic_report = build_synthetic_ablation(out_dir, seeds=args.synthetic_seeds)
    repair_metrics, repair_summary, repair_report = run_gated_repair(repair_dir)

    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "multimetric": multimetric_report,
        "factor_analysis": factor_report,
        "synthetic_ablation": synthetic_report,
        "repair": repair_report,
        "required_outputs": [
            str(multimetric_path.relative_to(ROOT)),
            str(feature_table.relative_to(ROOT)),
            str(factor_path.relative_to(ROOT)),
            str(synthetic_metrics.relative_to(ROOT)),
            str(synthetic_summary.relative_to(ROOT)),
            str(repair_metrics.relative_to(ROOT)),
            str(repair_summary.relative_to(ROOT)),
        ],
    }
    status_path = out_dir / "failure_family_goal_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
