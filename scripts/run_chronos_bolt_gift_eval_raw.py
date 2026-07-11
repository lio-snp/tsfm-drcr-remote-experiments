#!/usr/bin/env python
"""Run a narrow Chronos-Bolt raw rerun on a real GIFT-Eval dataset window."""

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

from low_snr_tsfm.baselines import (
    all_simple_baselines,
    naive_forecast,
    rolling_origin_select_baseline,
    seasonal_naive_forecast,
)
from low_snr_tsfm.gift_eval_windowing import (
    finite_pair_mask,
    forward_fill_nan,
    iter_univariate_targets,
    prediction_length,
    test_windows,
    window_count,
)
from low_snr_tsfm.forecast_export import history_context_sidecar_row
from low_snr_tsfm.metrics import (
    classify_degeneration,
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
from low_snr_tsfm.quantile_artifacts import (
    clean_quantile_levels,
    quantile_row_values,
    quantile_triplet_from_matrix,
)


RAW_DIR = ROOT / "results" / "raw_forecasts"
METRIC_DIR = ROOT / "results" / "window_metrics"
FAILURE_DIR = ROOT / "results" / "failure_mining"

LOCKED_SLICE_METADATA = {
    "bizitobs_application": ("Web/CloudOps", "intermittent_bursty"),
    "bitbrains_rnd": ("Web/CloudOps", "intermittent_bursty"),
    "car_parts": ("Sales", "intermittent_bursty"),
    "covid_deaths": ("Healthcare", "medium_snr_persistent"),
    "electricity": ("Energy", "seasonal_high_structure"),
    "ett1": ("Energy", "seasonal_high_structure"),
    "hospital": ("Healthcare", "medium_snr_persistent"),
    "loop_seattle": ("Transport", "medium_snr_persistent"),
    "m4_hourly": ("Econ/Fin", "noisy_low_signal"),
    "m4_monthly": ("Econ/Fin", "noisy_low_signal"),
    "m4_weekly": ("Econ/Fin", "noisy_low_signal"),
    "solar": ("Energy", "seasonal_high_structure"),
}


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


def missing_imports() -> list[dict[str, str]]:
    missing = []
    for name in ["datasets", "torch", "chronos"]:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - optional runtime dependency report
            missing.append({"package": name, "error": f"{type(exc).__name__}: {exc}"})
    return missing


def repo_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort for local clones
        return "unknown"


def clean_slug(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").lower()


def load_window_manifest(
    manifest_path: Path | None,
    dataset_name: str,
    term: str,
) -> dict[str, set[int]] | None:
    if manifest_path is None:
        return None
    if not manifest_path.exists():
        raise FileNotFoundError(f"window manifest not found: {manifest_path}")

    selected: dict[str, set[int]] = {}
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_dataset = row.get("dataset_name") or str(row.get("dataset", "")).split("/")[0]
            row_term = row.get("term") or str(row.get("dataset", "")).split("/")[-1]
            if clean_slug(str(row_dataset)) != clean_slug(dataset_name):
                continue
            if str(row_term) != str(term):
                continue
            series_id = str(row["series_id"])
            selected.setdefault(series_id, set()).add(int(row["window_index"]))
    return selected


def mean_of(rows: list[dict[str, object]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / len(rows))


def select_best_baseline(
    context: np.ndarray,
    target: np.ndarray,
    horizon: int,
    season_length: int,
) -> tuple[str, np.ndarray, float]:
    candidates = all_simple_baselines(
        context,
        horizon,
        season_length=season_length,
        ar_lags=min(24, max(1, context.size - 1)),
    )
    best_name = ""
    best_values = np.zeros(horizon, dtype=float)
    best_error = float("inf")
    for candidate in candidates:
        mask = finite_pair_mask(target, candidate.values)
        if not mask.any():
            continue
        error = mae(target[mask], candidate.values[mask])
        if error < best_error:
            best_name = candidate.name
            best_values = candidate.values
            best_error = error
    if not best_name:
        raise ValueError("No finite baseline comparison is available")
    return best_name, best_values, best_error


def auto_arima_forecast(context: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
    try:
        from statsforecast.models import AutoARIMA
    except Exception as exc:  # noqa: BLE001 - optional local baseline dependency
        raise RuntimeError(
            "statsforecast is required for --baseline-mode auto_arima; "
            "run scripts/setup_chronos_env.sh or install statsforecast"
        ) from exc

    values = forward_fill_nan(context).astype(np.float64)
    model = AutoARIMA(season_length=season_length)
    forecast = model.forecast(y=values, h=horizon)
    return np.asarray(forecast["mean"], dtype=float)


def auto_ets_forecast(context: np.ndarray, horizon: int, season_length: int) -> np.ndarray:
    try:
        from statsforecast.models import AutoETS
    except Exception as exc:  # noqa: BLE001 - optional local baseline dependency
        raise RuntimeError(
            "statsforecast is required for --baseline-mode auto_ets; "
            "run scripts/setup_chronos_env.sh or install statsforecast"
        ) from exc

    values = forward_fill_nan(context).astype(np.float64)
    model = AutoETS(season_length=season_length)
    forecast = model.forecast(y=values, h=horizon)
    return np.asarray(forecast["mean"], dtype=float)


def rolling_pre_origin_forecast(
    context: np.ndarray,
    horizon: int,
    season_length: int,
    *,
    n_folds: int = 3,
) -> tuple[str, np.ndarray]:
    validation_horizon = min(
        int(horizon),
        max(1, int(season_length)),
        max(1, int(np.asarray(context).size // (n_folds + 2))),
    )
    forecasters = {
        "naive": lambda values, length: naive_forecast(values, length),
        f"seasonal_naive_{season_length}": lambda values, length: seasonal_naive_forecast(
            values, length, season_length
        ),
        "auto_ets": lambda values, length: auto_ets_forecast(values, length, season_length),
        "auto_arima": lambda values, length: auto_arima_forecast(values, length, season_length),
    }
    selected = rolling_origin_select_baseline(
        context,
        horizon,
        forecasters,
        validation_horizon=validation_horizon,
        n_folds=n_folds,
    )
    return selected.name, selected.values


def select_baseline(
    mode: str,
    context: np.ndarray,
    target: np.ndarray,
    horizon: int,
    season_length: int,
) -> tuple[str, np.ndarray, float]:
    if mode == "best_simple":
        return select_best_baseline(context, target, horizon, season_length)
    if mode == "auto_arima":
        values = auto_arima_forecast(context, horizon, season_length)
        mask = finite_pair_mask(target, values)
        if not mask.any():
            raise ValueError("AutoARIMA produced no finite comparison points")
        return "auto_arima", values, mae(target[mask], values[mask])
    if mode == "auto_ets":
        values = auto_ets_forecast(context, horizon, season_length)
        mask = finite_pair_mask(target, values)
        if not mask.any():
            raise ValueError("AutoETS produced no finite comparison points")
        return "auto_ets", values, mae(target[mask], values[mask])
    if mode == "rolling_pre_origin":
        name, values = rolling_pre_origin_forecast(context, horizon, season_length)
        mask = finite_pair_mask(target, values)
        if not mask.any():
            raise ValueError("Rolling-selected baseline produced no finite comparison points")
        return name, values, mae(target[mask], values[mask])
    named = {candidate.name: candidate.values for candidate in all_simple_baselines(
        context,
        horizon,
        season_length=season_length,
        ar_lags=min(24, max(1, context.size - 1)),
    )}
    if mode == "seasonal_naive":
        named_key = f"seasonal_naive_{season_length}"
    else:
        named_key = mode
    if named_key in named:
        values = named[named_key]
        mask = finite_pair_mask(target, values)
        if not mask.any():
            raise ValueError(f"{named_key} produced no finite comparison points")
        return named_key, values, mae(target[mask], values[mask])
    raise ValueError(f"Unknown baseline mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/gift-eval")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dataset-name", default="bizitobs_application")
    parser.add_argument("--term", default="short", choices=["short", "medium", "long"])
    parser.add_argument("--model-id", default="amazon/chronos-bolt-small")
    parser.add_argument("--model-name", default="chronos_bolt_small")
    parser.add_argument("--max-series", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--context-cap", type=int, default=2048)
    parser.add_argument(
        "--baseline-mode",
        default="best_simple",
        choices=[
            "best_simple",
            "auto_arima",
            "auto_ets",
            "zero",
            "mean",
            "naive",
            "drift",
            "linear_ar",
            "seasonal_naive",
            "rolling_pre_origin",
        ],
    )
    parser.add_argument("--baseline-context-cap", type=int, default=None)
    parser.add_argument("--baseline-season-length", type=int, default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--regime", default=None)
    parser.add_argument("--quantile-levels", default="0.1,0.5,0.9")
    parser.add_argument("--window-manifest", default=None)
    parser.add_argument("--output-slug", default=None)
    parser.add_argument(
        "--export-history-sidecar",
        action="store_true",
        help="Write one sidecar row per forecast origin with exact context/baseline history values.",
    )
    args = parser.parse_args()

    dataset_slug = clean_slug(args.dataset_name)
    model_slug = clean_slug(args.model_name)
    term_slug = clean_slug(args.term)
    default_domain, default_regime = LOCKED_SLICE_METADATA.get(args.dataset_name, ("unknown", "unknown"))
    domain = args.domain or default_domain
    regime = args.regime or default_regime
    output_slug = args.output_slug or f"{model_slug}_{dataset_slug}_{term_slug}"
    if args.output_slug is None and args.baseline_mode != "best_simple":
        output_slug = f"{output_slug}_{clean_slug(args.baseline_mode)}"
    status_path = RAW_DIR / f"{output_slug}_status.json"

    missing = missing_imports()
    if missing:
        write_status(
            status_path,
            "blocked_missing_dependencies",
            {
                "missing": missing,
                "setup_command": ".venv-chronos/bin/python -m pip install datasets==2.17.1",
            },
        )
        return

    data_path = ROOT / args.data_path if args.data_path is not None else ROOT / args.data_root / args.dataset_name
    if not data_path.exists():
        write_status(
            status_path,
            "blocked_missing_data",
            {
                "dataset_dir": str(data_path),
                "setup_command": ".venv-chronos/bin/python scripts/download_gift_eval_subset.py --dataset-name "
                + args.dataset_name,
            },
        )
        return

    import torch
    from chronos import BaseChronosPipeline
    from datasets import load_from_disk

    quantile_levels = clean_quantile_levels([float(value) for value in args.quantile_levels.split(",")])

    hf_dataset = load_from_disk(str(data_path))
    first = hf_dataset[0]
    freq = str(first["freq"])
    horizon = prediction_length(args.dataset_name, freq, args.term)
    baseline_season_length = args.baseline_season_length
    if baseline_season_length is None:
        baseline_season_length = 1 if args.baseline_mode in {"auto_arima", "auto_ets"} else horizon
    baseline_context_cap = args.baseline_context_cap
    if baseline_context_cap is None:
        baseline_context_cap = 512 if args.baseline_mode in {"auto_arima", "auto_ets"} else args.context_cap
    raw_lengths = []
    for item in hf_dataset:
        target = np.asarray(item["target"], dtype=float)
        raw_lengths.append(target.shape[-1])
    gift_windows = window_count(min(raw_lengths), horizon)
    windows_to_run = min(gift_windows, max(1, args.max_windows))
    manifest_path = ROOT / args.window_manifest if args.window_manifest is not None else None
    selected_windows = load_window_manifest(manifest_path, args.dataset_name, args.term)
    manifest_requested_windows = (
        sum(len(window_indexes) for window_indexes in selected_windows.values())
        if selected_windows is not None
        else None
    )

    pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map="cpu")

    raw_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    run_id = f"{output_slug}_{int(time.time())}"
    source_commit = (
        f"gift-eval:{repo_commit(ROOT / 'external' / 'gift-eval')};"
        f"chronos-forecasting:{repo_commit(ROOT / 'external' / 'chronos-forecasting')}"
    )
    processed_series = 0
    matched_manifest_windows = 0

    for item_idx, item in enumerate(hf_dataset):
        item_id = str(item.get("item_id", f"item_{item_idx}"))
        for series_id, values in iter_univariate_targets(item["target"], item_id):
            if selected_windows is None:
                if processed_series >= args.max_series:
                    break
                windows = test_windows(values, horizon, gift_windows)[:windows_to_run]
            else:
                wanted = selected_windows.get(series_id)
                if not wanted:
                    continue
                windows = [window for window in test_windows(values, horizon, gift_windows) if window.window_index in wanted]
                if not windows:
                    continue
            processed_series += 1
            for window in windows:
                if selected_windows is not None:
                    matched_manifest_windows += 1
                full_context = forward_fill_nan(window.context)
                context = full_context[-args.context_cap :]
                target = np.asarray(window.target, dtype=float)
                baseline_context = full_context[-baseline_context_cap:]
                baseline_name, baseline_values, baseline_mae = select_baseline(
                    mode=args.baseline_mode,
                    context=baseline_context,
                    target=target,
                    horizon=horizon,
                    season_length=baseline_season_length,
                )

                quantiles, mean = pipeline.predict_quantiles(
                    torch.tensor(context, dtype=torch.float32),
                    prediction_length=horizon,
                    quantile_levels=quantile_levels,
                )
                q = quantiles[0].detach().cpu().numpy()
                mean_forecast = mean[0].detach().cpu().numpy()
                q10, q50, q90 = quantile_triplet_from_matrix(q, quantile_levels)
                mask = finite_pair_mask(target, mean_forecast)
                if not mask.any():
                    continue

                model_mae = mae(target[mask], mean_forecast[mask])
                model_mase = mase(target[mask], mean_forecast[mask], context, season_length=horizon)
                baseline_mase = mase(target[mask], baseline_values[mask], context, season_length=horizon)
                flags = classify_degeneration(
                    target[mask],
                    mean_forecast[mask],
                    model_error=model_mae,
                    baseline_error=baseline_mae,
                )
                metric_row = {
                    "run_id": run_id,
                    "dataset": f"{args.dataset_name}/{freq}/{args.term}",
                    "series_id": series_id,
                    "domain": domain,
                    "regime": regime,
                    "model": args.model_name,
                    "baseline": baseline_name,
                    "baseline_mode": args.baseline_mode,
                    "baseline_context_length": int(baseline_context.size),
                    "baseline_season_length": int(baseline_season_length),
                    "origin": window.origin,
                    "window_index": window.window_index,
                    "context_length": int(context.size),
                    "full_context_length": int(full_context.size),
                    "horizon": horizon,
                    "mae": model_mae,
                    "rmse": rmse(target[mask], mean_forecast[mask]),
                    "mase": model_mase,
                    "baseline_mae": baseline_mae,
                    "baseline_mase": baseline_mase,
                    "relative_error_ratio": relative_error_ratio(model_mae, baseline_mae),
                    "forecast_variance_ratio": forecast_variance_ratio(target[mask], mean_forecast[mask]),
                    "prediction_amplitude_ratio": prediction_amplitude_ratio(target[mask], mean_forecast[mask]),
                    "flatness_score": flatness_score(target[mask], mean_forecast[mask]),
                    "spike_recall": spike_recall(target[mask], mean_forecast[mask], k=3),
                    "empirical_coverage_90": empirical_coverage(target[mask], q10[mask], q90[mask]),
                    "failure_delta_005": int(relative_error_ratio(model_mae, baseline_mae) > 1.05),
                    "excess_variance": int(flags.excess_variance),
                    "over_smoothing": int(flags.over_smoothing),
                }
                metric_rows.append(metric_row)
                failure_rows.append(
                    {
                        **metric_row,
                        "selector": "real_gift_eval_window",
                        "failed": int(metric_row["failure_delta_005"]),
                    }
                )
                if args.export_history_sidecar:
                    history_rows.append(
                        history_context_sidecar_row(
                            run_id=run_id,
                            dataset=f"{args.dataset_name}/{freq}/{args.term}",
                            series_id=series_id,
                            model=args.model_name,
                            baseline_family=baseline_name,
                            baseline_mode=args.baseline_mode,
                            origin=window.origin,
                            window_index=window.window_index,
                            context=context,
                            full_context=full_context,
                            baseline_context=baseline_context,
                            target=target,
                            baseline_season_length=baseline_season_length,
                            baseline_context_cap=baseline_context_cap,
                            source_commit=source_commit,
                            model_id=args.model_id,
                        )
                    )

                for idx in range(horizon):
                    raw_rows.append(
                        {
                            "run_id": run_id,
                            "dataset": f"{args.dataset_name}/{freq}/{args.term}",
                            "series_id": series_id,
                            "domain": domain,
                            "regime": regime,
                            "model": args.model_name,
                            "baseline_family": baseline_name,
                            "baseline_mode": args.baseline_mode,
                            "baseline_context_length": int(baseline_context.size),
                            "baseline_season_length": int(baseline_season_length),
                            "origin": window.origin,
                            "window_index": window.window_index,
                            "horizon_index": idx + 1,
                            "context_length": int(context.size),
                            "full_context_length": int(full_context.size),
                            "horizon": horizon,
                            "actual": float(target[idx]),
                            "baseline_forecast": float(baseline_values[idx]),
                            "forecast_mean": float(mean_forecast[idx]),
                            "forecast_median": float(q50[idx]),
                            "forecast_q10": float(q10[idx]),
                            "forecast_q50": float(q50[idx]),
                            "forecast_q90": float(q90[idx]),
                            **quantile_row_values(q, quantile_levels, idx),
                            "source_commit": source_commit,
                            "model_id": args.model_id,
                            "model_version": getattr(pipeline.model.config, "_name_or_path", args.model_id),
                        }
                    )
        if selected_windows is not None and manifest_requested_windows is not None:
            if matched_manifest_windows >= manifest_requested_windows:
                break
        if selected_windows is None and processed_series >= args.max_series:
            break

    if not raw_rows or not metric_rows:
        write_status(
            status_path,
            "blocked_no_rows",
            {
                "dataset_dir": str(data_path),
                "dataset_name": args.dataset_name,
                "term": args.term,
                "model_id": args.model_id,
                "window_manifest": str(manifest_path) if manifest_path is not None else None,
                "manifest_requested_windows": manifest_requested_windows,
                "manifest_matched_windows": matched_manifest_windows if manifest_path is not None else None,
            },
        )
        return

    raw_path = RAW_DIR / f"{output_slug}.csv"
    history_path = RAW_DIR / f"{output_slug}_history_context.csv"
    metric_path = METRIC_DIR / f"{output_slug}_metrics.csv"
    failure_path = FAILURE_DIR / f"{output_slug}_window_failures.csv"
    summary_path = FAILURE_DIR / f"{output_slug}_failure_summary.csv"
    failure_rows.sort(key=lambda row: float(row["relative_error_ratio"]), reverse=True)
    summary_rows = [
        {
            "run_id": run_id,
            "dataset": f"{args.dataset_name}/{freq}/{args.term}",
            "model": args.model_name,
            "n_windows": len(metric_rows),
            "n_series": processed_series,
            "prediction_length": horizon,
            "baseline_mode": args.baseline_mode,
            "baseline_season_length": int(baseline_season_length),
            "baseline_context_cap": int(baseline_context_cap),
            "failure_rate_delta_005": mean_of(metric_rows, "failure_delta_005"),
            "excess_variance_rate": mean_of(metric_rows, "excess_variance"),
            "over_smoothing_rate": mean_of(metric_rows, "over_smoothing"),
            "mean_relative_error_ratio": mean_of(metric_rows, "relative_error_ratio"),
            "max_relative_error_ratio": max(float(row["relative_error_ratio"]) for row in metric_rows),
            "mean_flatness_score": mean_of(metric_rows, "flatness_score"),
            "mean_forecast_variance_ratio": mean_of(metric_rows, "forecast_variance_ratio"),
            "mean_prediction_amplitude_ratio": mean_of(metric_rows, "prediction_amplitude_ratio"),
            "mean_empirical_coverage_90": mean_of(metric_rows, "empirical_coverage_90"),
            "top_failure_series_id": failure_rows[0]["series_id"],
            "top_failure_window_index": failure_rows[0]["window_index"],
            "top_failure_baseline": failure_rows[0]["baseline"],
        }
    ]
    write_csv(raw_path, raw_rows)
    if args.export_history_sidecar:
        write_csv(history_path, history_rows)
    write_csv(metric_path, metric_rows)
    write_csv(failure_path, failure_rows)
    write_csv(summary_path, summary_rows)
    write_status(
        status_path,
        "ok",
        {
            "dataset_dir": str(data_path),
            "dataset": f"{args.dataset_name}/{freq}/{args.term}",
            "model_id": args.model_id,
            "prediction_length": horizon,
            "gift_eval_windows": gift_windows,
            "windows_run": len(metric_rows),
            "series_run": processed_series,
            "baseline_mode": args.baseline_mode,
            "baseline_season_length": int(baseline_season_length),
            "baseline_context_cap": int(baseline_context_cap),
            "quantile_levels": quantile_levels,
            "window_manifest": str(manifest_path) if manifest_path is not None else None,
            "manifest_requested_windows": manifest_requested_windows,
            "manifest_matched_windows": matched_manifest_windows if manifest_path is not None else None,
            "raw_forecasts": str(raw_path),
            "history_context_sidecar": str(history_path) if args.export_history_sidecar else None,
            "window_metrics": str(metric_path),
            "failure_mining": str(failure_path),
            "failure_summary": str(summary_path),
            "limitations": [
                "limited-series CPU rerun",
                (
                    f"aggregate-comparable local {args.baseline_mode} baseline"
                    if args.baseline_mode in {"auto_arima", "auto_ets"}
                    else "local simple baseline comparator"
                ),
                "context capped for local runtime",
            ],
        },
    )


if __name__ == "__main__":
    main()
