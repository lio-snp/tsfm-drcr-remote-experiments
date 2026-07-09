#!/usr/bin/env python
"""Run or stage a traffic-regime BMA reproduction branch.

When METR-LA/PEMS-BAY data is missing, this script writes an explicit blocked
status. Once a supported traffic matrix is present, it runs a CPU-bounded
Chronos-Bolt + historical-conditional + BMA slice.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.evaluation import rolling_windows
from low_snr_tsfm.features import feature_vector
from low_snr_tsfm.metrics import empirical_coverage, mae, relative_error_ratio, rmse
from low_snr_tsfm.repair import hull_interval
from low_snr_tsfm.traffic import (
    TrafficRegimeConfig,
    historical_conditional_forecast,
    label_traffic_regime,
    load_traffic_matrix,
)


OUT_DIR = ROOT / "results" / "traffic"
FAILURE_DIR = ROOT / "results" / "failure_mining"
DEFAULT_CANDIDATES = [
    ROOT / "data" / "traffic" / "metr-la.npz",
    ROOT / "data" / "traffic" / "pems-bay.npz",
    ROOT / "data" / "traffic" / "metr-la.h5",
    ROOT / "data" / "traffic" / "pems-bay.h5",
    ROOT / "data" / "traffic" / "metr-la.csv",
    ROOT / "data" / "traffic" / "pems-bay.csv",
]


def clean_slug(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").replace(".", "_").lower()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_status(path: Path, status: str, detail: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "timestamp": int(time.time()), **detail}
    path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def repo_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best effort
        return "unknown"


def resolve_data_path(value: str | None) -> Path | None:
    if value:
        path = ROOT / value
        return path if path.exists() else None
    for path in DEFAULT_CANDIDATES:
        if path.exists():
            return path
    return None


def mean_of(rows: list[dict[str, object]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / max(len(rows), 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dataset-name", default="metr-la")
    parser.add_argument("--model-id", default="amazon/chronos-bolt-small")
    parser.add_argument("--model-name", default="chronos_bolt_small")
    parser.add_argument("--max-sensors", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=288)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--step", type=int, default=12)
    parser.add_argument("--period", type=int, default=288)
    parser.add_argument("--bma-weight", type=float, default=0.5)
    parser.add_argument("--low-speed", type=float, default=30.0)
    parser.add_argument("--high-speed", type=float, default=55.0)
    parser.add_argument("--transition-range", type=float, default=15.0)
    args = parser.parse_args()

    output_slug = f"{clean_slug(args.model_name)}_{clean_slug(args.dataset_name)}_traffic_bma"
    status_path = OUT_DIR / f"{output_slug}_status.json"
    data_path = resolve_data_path(args.data_path)
    if data_path is None:
        write_status(
            status_path,
            "blocked_missing_data",
            {
                "expected_paths": [str(path.relative_to(ROOT)) for path in DEFAULT_CANDIDATES],
                "source_hint": (
                    "DCRNN README points to a Google Drive folder containing metr-la.h5 "
                    "and pems-bay.h5; place one file under data/traffic/."
                ),
                "paper": "https://arxiv.org/abs/2606.18367",
            },
        )
        return

    missing = []
    for package in ["torch", "chronos"]:
        try:
            __import__(package)
        except Exception as exc:  # noqa: BLE001 - optional runtime dependency report
            missing.append({"package": package, "error": f"{type(exc).__name__}: {exc}"})
    if missing:
        write_status(status_path, "blocked_missing_dependencies", {"missing": missing})
        return

    import torch
    from chronos import BaseChronosPipeline

    matrix = load_traffic_matrix(data_path)
    sensor_count = min(args.max_sensors, matrix.shape[1])
    pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map="cpu")
    quantile_levels = [0.1, 0.5, 0.9]
    regime_config = TrafficRegimeConfig(
        low_speed=args.low_speed,
        high_speed=args.high_speed,
        transition_range=args.transition_range,
    )
    run_id = f"{output_slug}_{int(time.time())}"
    source_commit = f"chronos-forecasting:{repo_commit(ROOT / 'external' / 'chronos-forecasting')}"
    raw_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []

    for sensor_idx in range(sensor_count):
        series = matrix[:, sensor_idx]
        windows = list(
            rolling_windows(
                series,
                context_length=args.context_length,
                horizon=args.horizon,
                step=args.step,
            )
        )[-args.max_windows :]
        for window_index, window in enumerate(windows):
            context = np.asarray(window.context, dtype=float)
            target = np.asarray(window.target, dtype=float)
            hist = historical_conditional_forecast(context, horizon=args.horizon, period=args.period)
            hist_mean = hist["mean"]
            hist_q10 = hist["q10"]
            hist_q90 = hist["q90"]
            quantiles, mean = pipeline.predict_quantiles(
                torch.tensor(context, dtype=torch.float32),
                prediction_length=args.horizon,
                quantile_levels=quantile_levels,
            )
            q = quantiles[0].detach().cpu().numpy()
            mean_forecast = mean[0].detach().cpu().numpy()
            bma_mean = (1.0 - args.bma_weight) * mean_forecast + args.bma_weight * hist_mean
            bma_q10, bma_q90 = hull_interval(q[:, 0], q[:, 2], hist_mean)
            model_mae = mae(target, mean_forecast)
            hist_mae = mae(target, hist_mean)
            bma_mae = mae(target, bma_mean)
            model_rer = relative_error_ratio(model_mae, hist_mae)
            bma_rer = relative_error_ratio(bma_mae, hist_mae)
            regime = label_traffic_regime(target, regime_config)
            metric_rows.append(
                {
                    "run_id": run_id,
                    "dataset": args.dataset_name,
                    "series_id": f"sensor_{sensor_idx}",
                    "domain": "traffic",
                    "regime": regime,
                    "model": args.model_name,
                    "baseline": "historical_conditional",
                    "origin": window.origin,
                    "window_index": window_index,
                    "context_length": args.context_length,
                    "horizon": args.horizon,
                    "model_mae": model_mae,
                    "historical_mae": hist_mae,
                    "bma_mae": bma_mae,
                    "model_rmse": rmse(target, mean_forecast),
                    "historical_rmse": rmse(target, hist_mean),
                    "bma_rmse": rmse(target, bma_mean),
                    "model_relative_error_ratio": model_rer,
                    "bma_relative_error_ratio": bma_rer,
                    "model_empirical_coverage_90": empirical_coverage(target, q[:, 0], q[:, 2]),
                    "historical_empirical_coverage_80": empirical_coverage(target, hist_q10, hist_q90),
                    "bma_hull_empirical_coverage_90": empirical_coverage(target, bma_q10, bma_q90),
                    "model_failure_delta_005": int(model_rer > 1.05),
                    "bma_failure_delta_005": int(bma_rer > 1.05),
                    "bma_improves_model": int(bma_mae < model_mae),
                }
            )
            feature_rows.append(
                {
                    "run_id": run_id,
                    "dataset": args.dataset_name,
                    "series_id": f"sensor_{sensor_idx}",
                    "domain": "Traffic",
                    "regime": regime,
                    "origin": window.origin,
                    "window_index": window_index,
                    "context_length": args.context_length,
                    "horizon": args.horizon,
                    "failure_delta_005": int(model_rer > 1.05),
                    "relative_error_ratio": model_rer,
                    "bma_relative_error_ratio": bma_rer,
                    "bma_improves_model": int(bma_mae < model_mae),
                    **feature_vector(
                        context,
                        horizon=args.horizon,
                        context_length=args.context_length,
                        period=args.period,
                    ),
                }
            )
            for idx in range(args.horizon):
                raw_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": args.dataset_name,
                        "series_id": f"sensor_{sensor_idx}",
                        "domain": "traffic",
                        "regime": regime,
                        "model": args.model_name,
                        "baseline_family": "historical_conditional",
                        "origin": window.origin,
                        "window_index": window_index,
                        "horizon_index": idx + 1,
                        "context_length": args.context_length,
                        "horizon": args.horizon,
                        "actual": float(target[idx]),
                        "forecast_mean": float(mean_forecast[idx]),
                        "forecast_q10": float(q[idx, 0]),
                        "forecast_q50": float(q[idx, 1]),
                        "forecast_q90": float(q[idx, 2]),
                        "historical_mean": float(hist_mean[idx]),
                        "historical_q10": float(hist_q10[idx]),
                        "historical_q90": float(hist_q90[idx]),
                        "bma_mean": float(bma_mean[idx]),
                        "bma_q10": float(bma_q10[idx]),
                        "bma_q90": float(bma_q90[idx]),
                        "source_commit": source_commit,
                        "model_id": args.model_id,
                        "model_version": getattr(pipeline.model.config, "_name_or_path", args.model_id),
                    }
                )

    summary_rows = []
    for regime in ["overall", *sorted({str(row["regime"]) for row in metric_rows})]:
        rows = metric_rows if regime == "overall" else [row for row in metric_rows if row["regime"] == regime]
        if not rows:
            continue
        summary_rows.append(
            {
                "run_id": run_id,
                "dataset": args.dataset_name,
                "regime": regime,
                "n_windows": len(rows),
                "model_failure_rate_delta_005": mean_of(rows, "model_failure_delta_005"),
                "bma_failure_rate_delta_005": mean_of(rows, "bma_failure_delta_005"),
                "bma_win_rate_vs_model": mean_of(rows, "bma_improves_model"),
                "model_mean_relative_error_ratio": mean_of(rows, "model_relative_error_ratio"),
                "bma_mean_relative_error_ratio": mean_of(rows, "bma_relative_error_ratio"),
                "model_mean_empirical_coverage_90": mean_of(rows, "model_empirical_coverage_90"),
                "bma_hull_mean_empirical_coverage_90": mean_of(rows, "bma_hull_empirical_coverage_90"),
            }
        )

    raw_path = OUT_DIR / f"{output_slug}_raw.csv"
    metric_path = OUT_DIR / f"{output_slug}_window_metrics.csv"
    summary_path = OUT_DIR / f"{output_slug}_summary.csv"
    manifest_path = OUT_DIR / f"{output_slug}_manifest.json"
    feature_path = FAILURE_DIR / f"{output_slug}_predictor_features.csv"
    write_csv(raw_path, raw_rows)
    write_csv(metric_path, metric_rows)
    write_csv(summary_path, summary_rows)
    write_csv(feature_path, feature_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "data_path": str(data_path),
                "matrix_shape": list(matrix.shape),
                "max_sensors": args.max_sensors,
                "max_windows": args.max_windows,
                "context_length": args.context_length,
                "horizon": args.horizon,
                "period": args.period,
                "bma_weight": args.bma_weight,
                "regime_config": {
                    "low_speed": args.low_speed,
                    "high_speed": args.high_speed,
                    "transition_range": args.transition_range,
                },
                "limitations": [
                    "CPU-bounded first reproduction slice",
                    "historical conditional baseline approximates the target paper until exact code is available",
                ],
            },
            indent=2,
        )
    )
    write_status(
        status_path,
        "ok",
        {
            "data_path": str(data_path),
            "windows": len(metric_rows),
            "raw_rows": len(raw_rows),
            "raw_forecasts": str(raw_path),
            "window_metrics": str(metric_path),
            "summary": str(summary_path),
            "predictor_features": str(feature_path),
            "manifest": str(manifest_path),
            "regime_counts": {
                regime: sum(1 for row in metric_rows if row["regime"] == regime)
                for regime in sorted({str(row["regime"]) for row in metric_rows})
            },
        },
    )


if __name__ == "__main__":
    main()
