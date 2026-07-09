#!/usr/bin/env python
"""Self-contained TimesFM raw rerun payload for remote jobs.

This file is intended to be passed to Hugging Face Jobs as a UV script, or run
on any machine with enough RAM. It prints a compact JSON summary and writes the
same raw/metric/status artifact shapes used by the local runners.
"""

# /// script
# dependencies = [
#   "datasets==2.17.1",
#   "huggingface-hub>=0.23.0",
#   "numpy>=1.26.4",
#   "statsforecast>=1.7.8",
#   "timesfm[torch]==2.0.0",
# ]
# ///

from __future__ import annotations

import csv
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, snapshot_download


OUT = Path(os.environ.get("TSFM_OUT_DIR", "timesfm_covid_raw_rerun"))
DATASET_NAME = os.environ.get("TSFM_DATASET_NAME", "covid_deaths")
MODEL_ID = os.environ.get("TSFM_MODEL_ID", "google/timesfm-2.5-200m-pytorch")
MODEL_NAME = os.environ.get("TSFM_MODEL_NAME", "timesfm_2_5")
CONTEXT_CAP = int(os.environ.get("TSFM_CONTEXT_CAP", "128"))
MAX_SERIES = int(os.environ.get("TSFM_MAX_SERIES", "1"))
MAX_WINDOWS = int(os.environ.get("TSFM_MAX_WINDOWS", "1"))
TERM = os.environ.get("TSFM_TERM", "short")
HF_OUTPUT_REPO_ID = os.environ.get("HF_OUTPUT_REPO_ID", "")
HF_OUTPUT_REPO_PRIVATE = os.environ.get("HF_OUTPUT_REPO_PRIVATE", "1").lower() not in {"0", "false", "no"}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def forward_fill_nan(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not finite.any():
        return np.zeros_like(arr)
    last = float(arr[finite][0])
    for idx, value in enumerate(arr):
        if np.isfinite(value):
            last = float(value)
        else:
            arr[idx] = last
    return arr


def finite_pair_mask(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    return np.isfinite(actual) & np.isfinite(forecast)


def prediction_length(dataset_name: str, freq: str, term: str) -> int:
    multiplier = {"short": 1, "medium": 10, "long": 15}[term]
    unit = "".join(ch for ch in str(freq) if not ch.isdigit()).upper()
    unit = {"MIN": "T", "H": "H", "D": "D", "S": "S", "W": "W", "M": "M"}.get(unit, unit)
    base = {"M": 12, "W": 8, "D": 30, "H": 48, "T": 48, "S": 60}[unit]
    return multiplier * base


def window_count(min_series_length: int, pred_length: int) -> int:
    return min(max(1, math.ceil(0.1 * int(min_series_length) / int(pred_length))), 20)


def test_windows(series: np.ndarray, pred_length: int, windows: int) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    values = np.asarray(series, dtype=float)
    test_start = values.size - pred_length * windows
    output = []
    for idx in range(windows):
        origin = test_start + idx * pred_length
        output.append((idx, origin, values[:origin], values[origin : origin + pred_length]))
    return output


def iter_univariate_targets(target: np.ndarray, item_id: str):
    values = np.asarray(target, dtype=float)
    if values.ndim == 1:
        yield item_id, values
    else:
        for dim in range(values.shape[0]):
            yield f"{item_id}_dim{dim}", values[dim]


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mase(actual: np.ndarray, forecast: np.ndarray, context: np.ndarray, season_length: int) -> float:
    context = np.asarray(context, dtype=float)
    season = max(1, min(season_length, context.size - 1))
    scale = float(np.mean(np.abs(context[season:] - context[:-season]))) if context.size > season else 0.0
    return mae(actual, forecast) / max(scale, 1e-12)


def empirical_coverage(actual: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    return float(np.mean((actual >= low) & (actual <= high)))


def forecast_variance_ratio(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.var(np.diff(forecast)) / max(np.var(np.diff(actual)), 1e-12)) if actual.size > 1 else 0.0


def prediction_amplitude_ratio(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(forecast))) / max(np.mean(np.abs(np.diff(actual))), 1e-12)) if actual.size > 1 else 0.0


def flatness_score(actual: np.ndarray, forecast: np.ndarray) -> float:
    actual_amp = np.mean(np.abs(np.diff(actual))) if actual.size > 1 else 0.0
    forecast_amp = np.mean(np.abs(np.diff(forecast))) if forecast.size > 1 else 0.0
    return float(max(0.0, 1.0 - forecast_amp / max(actual_amp, 1e-12)))


def spike_recall(actual: np.ndarray, forecast: np.ndarray, k: int = 3) -> float:
    if actual.size < 2:
        return 0.0
    k = min(k, actual.size)
    actual_idx = set(np.argsort(np.abs(actual - np.median(actual)))[-k:])
    forecast_idx = set(np.argsort(np.abs(forecast - np.median(forecast)))[-k:])
    return float(len(actual_idx & forecast_idx) / max(len(actual_idx), 1))


def auto_ets_forecast(context: np.ndarray, horizon: int) -> np.ndarray:
    from statsforecast.models import AutoETS

    model = AutoETS(season_length=1)
    forecast = model.forecast(y=forward_fill_nan(context).astype(np.float64), h=horizon)
    return np.asarray(forecast["mean"], dtype=float)


def quantile_triplet(quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(quantiles, dtype=float)
    if q.shape[1] >= 10:
        return q[:, 1], q[:, 5], q[:, 9]
    return q[:, 0], q[:, q.shape[1] // 2], q[:, -1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    local_data = OUT / "gift-eval"
    snapshot_download(
        repo_id="Salesforce/GiftEval",
        repo_type="dataset",
        allow_patterns=[f"{DATASET_NAME}/*"],
        local_dir=str(local_data),
    )

    from datasets import load_from_disk
    import timesfm

    hf_dataset = load_from_disk(str(local_data / DATASET_NAME))
    freq = str(hf_dataset[0]["freq"])
    horizon = prediction_length(DATASET_NAME, freq, TERM)
    raw_lengths = [np.asarray(item["target"], dtype=float).shape[-1] for item in hf_dataset]
    gift_windows = window_count(min(raw_lengths), horizon)
    windows_to_run = min(gift_windows, max(1, MAX_WINDOWS))

    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
    model.compile(
        timesfm.ForecastConfig(
            max_context=CONTEXT_CAP,
            max_horizon=horizon,
            normalize_inputs=True,
            use_continuous_quantile_head=True,
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    raw_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    run_id = f"{MODEL_NAME}_{DATASET_NAME}_{TERM}_auto_ets_{int(time.time())}"
    processed_series = 0
    for item_idx, item in enumerate(hf_dataset):
        item_id = str(item.get("item_id", f"item_{item_idx}"))
        for series_id, values in iter_univariate_targets(item["target"], item_id):
            if processed_series >= MAX_SERIES:
                break
            processed_series += 1
            for window_index, origin, full_context, target in test_windows(values, horizon, gift_windows)[:windows_to_run]:
                full_context = forward_fill_nan(full_context)
                context = full_context[-CONTEXT_CAP:]
                baseline_context = full_context[-512:]
                target = np.asarray(target, dtype=float)
                baseline = auto_ets_forecast(baseline_context, horizon)
                point, quantiles = model.forecast(horizon=horizon, inputs=[context])
                forecast = np.asarray(point[0], dtype=float)
                q10, q50, q90 = quantile_triplet(np.asarray(quantiles[0], dtype=float))
                mask = finite_pair_mask(target, forecast)
                model_mae = mae(target[mask], forecast[mask])
                baseline_mae = mae(target[mask], baseline[mask])
                rer = model_mae / max(baseline_mae, 1e-12)
                metric_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": f"{DATASET_NAME}/{freq}/{TERM}",
                        "series_id": series_id,
                        "domain": "Healthcare",
                        "regime": "medium_snr_persistent",
                        "model": MODEL_NAME,
                        "baseline": "auto_ets",
                        "origin": origin,
                        "window_index": window_index,
                        "context_length": int(context.size),
                        "horizon": horizon,
                        "mae": model_mae,
                        "rmse": rmse(target[mask], forecast[mask]),
                        "mase": mase(target[mask], forecast[mask], context, horizon),
                        "baseline_mae": baseline_mae,
                        "baseline_mase": mase(target[mask], baseline[mask], context, horizon),
                        "relative_error_ratio": rer,
                        "forecast_variance_ratio": forecast_variance_ratio(target[mask], forecast[mask]),
                        "prediction_amplitude_ratio": prediction_amplitude_ratio(target[mask], forecast[mask]),
                        "flatness_score": flatness_score(target[mask], forecast[mask]),
                        "spike_recall": spike_recall(target[mask], forecast[mask]),
                        "empirical_coverage_90": empirical_coverage(target[mask], q10[mask], q90[mask]),
                        "failure_delta_005": int(rer > 1.05),
                    }
                )
                for idx in range(horizon):
                    raw_rows.append(
                        {
                            "run_id": run_id,
                            "dataset": f"{DATASET_NAME}/{freq}/{TERM}",
                            "series_id": series_id,
                            "domain": "Healthcare",
                            "regime": "medium_snr_persistent",
                            "model": MODEL_NAME,
                            "baseline_family": "auto_ets",
                            "origin": origin,
                            "window_index": window_index,
                            "horizon_index": idx + 1,
                            "context_length": int(context.size),
                            "horizon": horizon,
                            "actual": float(target[idx]),
                            "baseline_forecast": float(baseline[idx]),
                            "forecast_mean": float(forecast[idx]),
                            "forecast_median": float(q50[idx]),
                            "forecast_q10": float(q10[idx]),
                            "forecast_q50": float(q50[idx]),
                            "forecast_q90": float(q90[idx]),
                            "model_id": MODEL_ID,
                            "model_version": MODEL_ID,
                        }
                    )
        if processed_series >= MAX_SERIES:
            break

    if not metric_rows:
        raise RuntimeError("No TimesFM windows were evaluated; check dataset/window limits.")

    raw_artifact = OUT / "timesfm_2_5_covid_deaths_short_auto_ets_raw.csv"
    metrics_artifact = OUT / "timesfm_2_5_covid_deaths_short_auto_ets_metrics.csv"
    status_artifact = OUT / "timesfm_2_5_covid_deaths_short_auto_ets_status.json"
    summary_artifact = OUT / "timesfm_2_5_covid_deaths_short_auto_ets_summary.json"
    summary = {
        "status": "ok",
        "run_id": run_id,
        "dataset": f"{DATASET_NAME}/{freq}/{TERM}",
        "model_id": MODEL_ID,
        "windows_run": len(metric_rows),
        "raw_rows": len(raw_rows),
        "raw_forecasts": str(raw_artifact),
        "window_metrics": str(metrics_artifact),
        "status_artifact": str(status_artifact),
        "summary_artifact": str(summary_artifact),
        "failure_rate_delta_005": float(np.mean([row["failure_delta_005"] for row in metric_rows])),
        "mean_relative_error_ratio": float(np.mean([row["relative_error_ratio"] for row in metric_rows])),
        "max_relative_error_ratio": float(max(row["relative_error_ratio"] for row in metric_rows)),
        "output_repo_requested": bool(HF_OUTPUT_REPO_ID),
        "output_repo_private": bool(HF_OUTPUT_REPO_PRIVATE) if HF_OUTPUT_REPO_ID else None,
    }
    write_csv(raw_artifact, raw_rows)
    write_csv(metrics_artifact, metric_rows)
    write_json(status_artifact, summary)
    write_json(summary_artifact, summary)

    if HF_OUTPUT_REPO_ID:
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(
            repo_id=HF_OUTPUT_REPO_ID,
            repo_type="dataset",
            private=HF_OUTPUT_REPO_PRIVATE,
            exist_ok=True,
        )
        api.upload_folder(
            repo_id=HF_OUTPUT_REPO_ID,
            repo_type="dataset",
            folder_path=str(OUT),
            path_in_repo=f"timesfm-covid/{run_id}",
        )
        summary["uploaded_to"] = f"{HF_OUTPUT_REPO_ID}/timesfm-covid/{run_id}"
        write_json(status_artifact, summary)
        write_json(summary_artifact, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
