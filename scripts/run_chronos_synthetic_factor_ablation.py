#!/usr/bin/env python
"""Run a compact Chronos-Bolt synthetic factor ablation.

The goal is not to optimize Chronos, but to put at least one real zero-shot
TSFM through controlled knobs that mirror the failure-regime taxonomy. If the
local Chronos runtime is unavailable, the script writes a structured blocker
artifact and exits successfully so the research ledger remains reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
EXTERNAL_CHRONOS_SRC = ROOT / "external" / "chronos-forecasting" / "src"
if EXTERNAL_CHRONOS_SRC.exists():
    sys.path.insert(0, str(EXTERNAL_CHRONOS_SRC))

from low_snr_tsfm.baselines import seasonal_naive_forecast  # noqa: E402
from low_snr_tsfm.metrics import (  # noqa: E402
    empirical_coverage,
    flatness_score,
    forecast_variance_ratio,
    mae,
    mase,
    prediction_amplitude_ratio,
    relative_error_ratio,
    rmse,
    spike_recall,
)


OUT_DIR = ROOT / "results" / "failure_family"
RAW_PATH = OUT_DIR / "tsfm_synthetic_ablation_raw.csv"
METRICS_PATH = OUT_DIR / "tsfm_synthetic_ablation_metrics.csv"
SUMMARY_PATH = OUT_DIR / "tsfm_synthetic_ablation_summary.csv"
STATUS_PATH = OUT_DIR / "tsfm_synthetic_ablation_status.json"


def write_status(status: str, detail: dict[str, object]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "timestamp": int(time.time()), **detail}
    STATUS_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def check_imports() -> list[dict[str, str]]:
    missing = []
    for package in ["torch", "transformers", "accelerate", "chronos"]:
        try:
            __import__(package)
        except Exception as exc:  # noqa: BLE001 - optional runtime blocker
            missing.append({"package": package, "error": f"{type(exc).__name__}: {exc}"})
    return missing


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def controlled_series(
    *,
    context_length: int,
    horizon: int,
    seasonality: float,
    noise: float,
    spike_size: float,
    decay_rate: float,
    seed: int,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    total = context_length + horizon
    t = np.arange(total, dtype=float)
    level = 10.0 + 0.01 * t
    seasonal = seasonality * np.sin(2 * np.pi * t / period)
    values = level + seasonal + rng.normal(0.0, noise, size=total)

    future_idx = np.arange(context_length, total)
    if spike_size > 0:
        values[context_length + horizon // 3] += spike_size
    if decay_rate > 0:
        step = np.arange(1, horizon + 1, dtype=float)
        values[future_idx] = values[future_idx] * np.exp(-decay_rate * step)
    return values[:context_length], values[context_length:]


def run_model(
    context: np.ndarray,
    horizon: int,
    pipeline: object,
    quantile_levels: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.no_grad():
        quantiles, mean = pipeline.predict_quantiles(
            torch.tensor(context, dtype=torch.float32),
            prediction_length=horizon,
            quantile_levels=quantile_levels,
        )
    return quantiles[0].detach().cpu().numpy(), mean[0].detach().cpu().numpy()


def build_sweep() -> list[dict[str, object]]:
    base = {
        "context_length": 96,
        "seasonality": 1.0,
        "noise": 0.5,
        "spike_size": 0.0,
        "decay_rate": 0.0,
    }
    sweeps = {
        "context_length": [24, 48, 96],
        "seasonality": [0.0, 0.5, 1.5],
        "spike_size": [0.0, 5.0],
        "decay_rate": [0.0, 0.05],
    }
    configs: list[dict[str, object]] = []
    for factor, values in sweeps.items():
        for value in values:
            params = dict(base)
            params[factor] = value
            params["factor"] = factor
            params["value"] = value
            configs.append(params)
    return configs


def summarize(metric_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    groups = sorted({(str(row["factor"]), str(row["value"])) for row in metric_rows})
    for factor, value in groups:
        subset = [row for row in metric_rows if str(row["factor"]) == factor and str(row["value"]) == value]
        summary_rows.append(
            {
                "factor": factor,
                "value": value,
                "n_windows": len(subset),
                "failure_rate_delta_005": float(np.mean([int(row["failure_delta_005"]) for row in subset])),
                "median_relative_error_ratio": median([float(row["relative_error_ratio"]) for row in subset]),
                "median_mase_relative_error_ratio": median([float(row["mase_relative_error_ratio"]) for row in subset]),
                "median_rmse_relative_error_ratio": median([float(row["rmse_relative_error_ratio"]) for row in subset]),
                "mean_empirical_coverage_90": float(np.mean([float(row["empirical_coverage_90"]) for row in subset])),
                "mean_forecast_variance_ratio": float(np.mean([float(row["forecast_variance_ratio"]) for row in subset])),
                "mean_flatness_score": float(np.mean([float(row["flatness_score"]) for row in subset])),
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="amazon/chronos-bolt-tiny")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--period", type=int, default=24)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--max-cases", type=int, default=0)
    args = parser.parse_args()

    missing = check_imports()
    if missing:
        write_status(
            "blocked_missing_dependencies",
            {
                "missing": missing,
                "setup_command": "scripts/setup_chronos_env.sh",
                "intended_outputs": [
                    str(RAW_PATH.relative_to(ROOT)),
                    str(METRICS_PATH.relative_to(ROOT)),
                    str(SUMMARY_PATH.relative_to(ROOT)),
                ],
            },
        )
        return

    try:
        from chronos import BaseChronosPipeline

        pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map="cpu")
        configs = build_sweep()
        if args.max_cases > 0:
            configs = configs[: args.max_cases]
        quantile_levels = [0.1, 0.5, 0.9]
        raw_rows: list[dict[str, object]] = []
        metric_rows: list[dict[str, object]] = []
        run_id = f"chronos_synthetic_factor_ablation_{int(time.time())}"

        for config_index, params in enumerate(configs):
            for seed_index in range(args.seeds):
                seed = 20260704 + config_index * 100 + seed_index
                context, actual = controlled_series(
                    context_length=int(params["context_length"]),
                    horizon=args.horizon,
                    seasonality=float(params["seasonality"]),
                    noise=float(params["noise"]),
                    spike_size=float(params["spike_size"]),
                    decay_rate=float(params["decay_rate"]),
                    seed=seed,
                    period=args.period,
                )
                baseline = seasonal_naive_forecast(context, args.horizon, args.period)
                quantiles, mean_forecast = run_model(context, args.horizon, pipeline, quantile_levels)

                model_mae = mae(actual, mean_forecast)
                baseline_mae = mae(actual, baseline)
                model_rmse = rmse(actual, mean_forecast)
                baseline_rmse = rmse(actual, baseline)
                model_mase = mase(actual, mean_forecast, context, season_length=args.period)
                baseline_mase = mase(actual, baseline, context, season_length=args.period)
                rer = relative_error_ratio(model_mae, baseline_mae)
                metric_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": "synthetic_factor_ablation",
                        "model": "chronos_bolt_tiny",
                        "model_id": args.model_id,
                        "baseline": f"seasonal_naive_{args.period}",
                        "factor": params["factor"],
                        "value": params["value"],
                        "seed": seed,
                        "context_length": params["context_length"],
                        "horizon": args.horizon,
                        "seasonality": params["seasonality"],
                        "noise": params["noise"],
                        "spike_size": params["spike_size"],
                        "decay_rate": params["decay_rate"],
                        "mae": model_mae,
                        "baseline_mae": baseline_mae,
                        "relative_error_ratio": rer,
                        "rmse": model_rmse,
                        "baseline_rmse": baseline_rmse,
                        "rmse_relative_error_ratio": relative_error_ratio(model_rmse, baseline_rmse),
                        "mase": model_mase,
                        "baseline_mase": baseline_mase,
                        "mase_relative_error_ratio": relative_error_ratio(model_mase, baseline_mase),
                        "empirical_coverage_90": empirical_coverage(actual, quantiles[:, 0], quantiles[:, 2]),
                        "forecast_variance_ratio": forecast_variance_ratio(actual, mean_forecast),
                        "prediction_amplitude_ratio": prediction_amplitude_ratio(actual, mean_forecast),
                        "flatness_score": flatness_score(actual, mean_forecast),
                        "spike_recall": spike_recall(actual, mean_forecast, k=3),
                        "failure_delta_005": int(rer > 1.05),
                    }
                )
                for horizon_index in range(args.horizon):
                    raw_rows.append(
                        {
                            "run_id": run_id,
                            "dataset": "synthetic_factor_ablation",
                            "model": "chronos_bolt_tiny",
                            "factor": params["factor"],
                            "value": params["value"],
                            "seed": seed,
                            "context_length": params["context_length"],
                            "horizon": args.horizon,
                            "horizon_index": horizon_index + 1,
                            "actual": float(actual[horizon_index]),
                            "forecast_mean": float(mean_forecast[horizon_index]),
                            "forecast_q10": float(quantiles[horizon_index, 0]),
                            "forecast_q50": float(quantiles[horizon_index, 1]),
                            "forecast_q90": float(quantiles[horizon_index, 2]),
                            "baseline_forecast": float(baseline[horizon_index]),
                        }
                    )

        summary_rows = summarize(metric_rows)
        write_csv(RAW_PATH, raw_rows)
        write_csv(METRICS_PATH, metric_rows)
        write_csv(SUMMARY_PATH, summary_rows)
        write_status(
            "ok",
            {
                "model_id": args.model_id,
                "n_windows": len(metric_rows),
                "n_summary_rows": len(summary_rows),
                "raw": str(RAW_PATH.relative_to(ROOT)),
                "metrics": str(METRICS_PATH.relative_to(ROOT)),
                "summary": str(SUMMARY_PATH.relative_to(ROOT)),
            },
        )
    except Exception as exc:  # noqa: BLE001 - convert runtime limits into a reproducible blocker
        write_status(
            "blocked_runtime_error",
            {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
                "repro_command": ".venv-chronos/bin/python scripts/run_chronos_synthetic_factor_ablation.py --seeds 2",
                "intended_outputs": [
                    str(RAW_PATH.relative_to(ROOT)),
                    str(METRICS_PATH.relative_to(ROOT)),
                    str(SUMMARY_PATH.relative_to(ROOT)),
                ],
            },
        )


if __name__ == "__main__":
    main()
