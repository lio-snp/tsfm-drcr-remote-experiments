#!/usr/bin/env python
"""Run a tiny Chronos-Bolt raw-forecast contract smoke test when available.

If dependencies are missing, write a structured blocker artifact and exit 0 so
the research ledger can capture the next concrete action.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.metrics import (
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
from low_snr_tsfm.baselines import naive_forecast
from low_snr_tsfm.synthetic import seasonal_ar


RAW_DIR = ROOT / "results" / "raw_forecasts"
METRIC_DIR = ROOT / "results" / "window_metrics"
STATUS_PATH = RAW_DIR / "chronos_bolt_smoke_status.json"


def write_status(status: str, detail: dict[str, object]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "timestamp": int(time.time()), **detail}
    STATUS_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def check_imports():
    missing = []
    for name in ["torch", "transformers", "accelerate", "chronos"]:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - report exact optional-dep blocker
            missing.append({"package": name, "error": f"{type(exc).__name__}: {exc}"})
    return missing


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    missing = check_imports()
    if missing:
        write_status(
            "blocked_missing_dependencies",
            {
                "missing": missing,
                "setup_command": "scripts/setup_chronos_env.sh",
                "intended_outputs": [
                    "results/raw_forecasts/chronos_bolt_smoke.csv",
                    "results/window_metrics/chronos_bolt_smoke_metrics.csv",
                ],
            },
        )
        return

    import torch
    from chronos import BaseChronosPipeline

    model_id = "amazon/chronos-bolt-tiny"
    context_length = 64
    horizon = 12
    origin = context_length
    series = seasonal_ar(
        length=context_length + horizon,
        phi=0.3,
        seasonal_amplitude=1.0,
        period=12,
        sigma=0.25,
        seed=20260703,
    )
    context = series[:context_length]
    target = series[context_length : context_length + horizon]
    baseline = naive_forecast(context, horizon)

    pipeline = BaseChronosPipeline.from_pretrained(model_id, device_map="cpu")
    quantiles, mean = pipeline.predict_quantiles(
        torch.tensor(context, dtype=torch.float32),
        prediction_length=horizon,
        quantile_levels=[0.1, 0.5, 0.9],
    )
    q = quantiles[0].detach().cpu().numpy()
    mean_forecast = mean[0].detach().cpu().numpy()

    run_id = f"chronos_bolt_smoke_{int(time.time())}"
    raw_rows = []
    for idx in range(horizon):
        raw_rows.append(
            {
                "run_id": run_id,
                "dataset": "synthetic_seasonal_smoke",
                "series_id": "synthetic_0",
                "domain": "controlled",
                "regime": "seasonal_high_structure",
                "model": "chronos_bolt_tiny",
                "baseline_family": "naive",
                "origin": origin,
                "horizon_index": idx + 1,
                "context_length": context_length,
                "horizon": horizon,
                "actual": float(target[idx]),
                "forecast_mean": float(mean_forecast[idx]),
                "forecast_median": float(q[idx, 1]),
                "forecast_q10": float(q[idx, 0]),
                "forecast_q50": float(q[idx, 1]),
                "forecast_q90": float(q[idx, 2]),
                "source_commit": "external/chronos-forecasting",
                "model_id": model_id,
                "model_version": getattr(pipeline.model.config, "_name_or_path", model_id),
            }
        )

    model_mae = mae(target, mean_forecast)
    baseline_mae = mae(target, baseline)
    model_mase = mase(target, mean_forecast, context, season_length=12)
    baseline_mase = mase(target, baseline, context, season_length=12)
    metric_rows = [
        {
            "run_id": run_id,
            "dataset": "synthetic_seasonal_smoke",
            "series_id": "synthetic_0",
            "domain": "controlled",
            "regime": "seasonal_high_structure",
            "model": "chronos_bolt_tiny",
            "baseline": "naive",
            "origin": origin,
            "context_length": context_length,
            "horizon": horizon,
            "mae": model_mae,
            "rmse": rmse(target, mean_forecast),
            "mase": model_mase,
            "baseline_mae": baseline_mae,
            "baseline_mase": baseline_mase,
            "relative_error_ratio": relative_error_ratio(model_mae, baseline_mae),
            "forecast_variance_ratio": forecast_variance_ratio(target, mean_forecast),
            "prediction_amplitude_ratio": prediction_amplitude_ratio(target, mean_forecast),
            "flatness_score": flatness_score(target, mean_forecast),
            "spike_recall": spike_recall(target, mean_forecast, k=3),
            "empirical_coverage_90": empirical_coverage(target, q[:, 0], q[:, 2]),
        }
    ]
    write_csv(RAW_DIR / "chronos_bolt_smoke.csv", raw_rows)
    write_csv(METRIC_DIR / "chronos_bolt_smoke_metrics.csv", metric_rows)
    write_status(
        "ok",
        {
            "model_id": model_id,
            "raw_forecasts": str(RAW_DIR / "chronos_bolt_smoke.csv"),
            "window_metrics": str(METRIC_DIR / "chronos_bolt_smoke_metrics.csv"),
        },
    )


if __name__ == "__main__":
    main()
