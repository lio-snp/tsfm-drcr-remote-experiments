#!/usr/bin/env python
"""Run or preflight a TimesFM raw rerun on a real GIFT-Eval dataset window."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from low_snr_tsfm.gift_eval_windowing import (  # noqa: E402
    finite_pair_mask,
    forward_fill_nan,
    iter_univariate_targets,
    prediction_length,
    test_windows,
    window_count,
)
from low_snr_tsfm.forecast_export import history_context_sidecar_row  # noqa: E402
from low_snr_tsfm.metrics import (  # noqa: E402
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
from low_snr_tsfm.quantile_artifacts import (  # noqa: E402
    quantile_row_values,
    quantile_triplet_from_matrix,
)
from low_snr_tsfm.system_memory import available_ram_gb  # noqa: E402
from run_chronos_bolt_gift_eval_raw import (  # noqa: E402
    LOCKED_SLICE_METADATA,
    RAW_DIR,
    FAILURE_DIR,
    METRIC_DIR,
    clean_slug,
    load_window_manifest,
    mean_of,
    repo_commit,
    select_baseline,
    write_csv,
    write_status,
)


def missing_imports() -> list[dict[str, str]]:
    missing = []
    for name in ["datasets", "torch", "timesfm"]:
        if importlib.util.find_spec(name) is None:
            missing.append({"package": name, "error": "module spec not found"})
    return missing


def timesfm_quantile_levels(quantiles: np.ndarray) -> list[float]:
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"Expected TimesFM quantiles with shape horizon x quantile, got {q.shape}")
    if q.shape[1] >= 10:
        return [idx / 10.0 for idx in range(q.shape[1])]
    if q.shape[1] >= 9:
        return [(idx + 1) / 10.0 for idx in range(q.shape[1])]
    raise ValueError(f"TimesFM quantile output has too few columns: {q.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/gift-eval")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--dataset-name", default="covid_deaths")
    parser.add_argument("--term", default="short", choices=["short", "medium", "long"])
    parser.add_argument("--model-id", default="google/timesfm-2.5-200m-pytorch")
    parser.add_argument("--model-name", default="timesfm_2_5")
    parser.add_argument("--max-series", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--context-cap", type=int, default=1024)
    parser.add_argument(
        "--baseline-mode",
        default="auto_ets",
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
        ],
    )
    parser.add_argument("--baseline-context-cap", type=int, default=None)
    parser.add_argument("--baseline-season-length", type=int, default=None)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--regime", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-memory-preflight", action="store_true")
    parser.add_argument("--min-available-ram-gb", type=float, default=1.56)
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
                "setup_command": ".venv-chronos/bin/python -m pip install -e 'external/timesfm[torch]'",
            },
        )
        return

    available = available_ram_gb()
    preflight = {
        "available_ram_gb": round(available, 3),
        "min_available_ram_gb": args.min_available_ram_gb,
        "model_id": args.model_id,
        "dataset_name": args.dataset_name,
        "term": args.term,
        "context_cap": args.context_cap,
        "max_series": args.max_series,
        "max_windows": args.max_windows,
    }
    if not args.skip_memory_preflight and available < args.min_available_ram_gb:
        write_status(
            status_path,
            "blocked_insufficient_memory",
            {
                **preflight,
                "detail": "TimesFM 2.5 local load is not attempted because available RAM is below the preflight floor.",
            },
        )
        return
    if args.preflight_only:
        write_status(status_path, "preflight_ok", preflight)
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

    try:
        import timesfm
        from datasets import load_from_disk
    except Exception as exc:  # noqa: BLE001 - optional runtime dependency report
        write_status(
            status_path,
            "blocked_import_error",
            {
                **preflight,
                "error": f"{type(exc).__name__}: {exc}",
                "detail": "TimesFM runtime imports failed after dependency specs were found.",
            },
        )
        return

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
    raw_lengths = [np.asarray(item["target"], dtype=float).shape[-1] for item in hf_dataset]
    gift_windows = window_count(min(raw_lengths), horizon)
    windows_to_run = min(gift_windows, max(1, args.max_windows))
    manifest_path = ROOT / args.window_manifest if args.window_manifest is not None else None
    selected_windows = load_window_manifest(manifest_path, args.dataset_name, args.term)
    manifest_requested_windows = (
        sum(len(window_indexes) for window_indexes in selected_windows.values())
        if selected_windows is not None
        else None
    )

    try:
        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
        model.compile(
            timesfm.ForecastConfig(
                max_context=args.context_cap,
                max_horizon=horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 - preserve blocked model-load evidence
        write_status(
            status_path,
            "blocked_model_load_error",
            {
                **preflight,
                "error": f"{type(exc).__name__}: {exc}",
                "detail": "TimesFM model load or compile failed before raw forecasts could be generated.",
            },
        )
        return

    raw_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []
    run_id = f"{output_slug}_{int(time.time())}"
    source_commit = (
        f"gift-eval:{repo_commit(ROOT / 'external' / 'gift-eval')};"
        f"timesfm:{repo_commit(ROOT / 'external' / 'timesfm')}"
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

                point, quantiles = model.forecast(horizon=horizon, inputs=[context])
                mean_forecast = np.asarray(point[0], dtype=float)
                quantile_matrix = np.asarray(quantiles[0], dtype=float)
                quantile_levels = timesfm_quantile_levels(quantile_matrix)
                q10, q50, q90 = quantile_triplet_from_matrix(quantile_matrix, quantile_levels)
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
                failure_rows.append({**metric_row, "selector": "real_gift_eval_window", "failed": int(metric_row["failure_delta_005"])})
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
                            **quantile_row_values(quantile_matrix, quantile_levels, idx),
                            "source_commit": source_commit,
                            "model_id": args.model_id,
                            "model_version": args.model_id,
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
            "failure_rate_delta_005": mean_of(metric_rows, "failure_delta_005"),
            "excess_variance_rate": mean_of(metric_rows, "excess_variance"),
            "over_smoothing_rate": mean_of(metric_rows, "over_smoothing"),
            "mean_relative_error_ratio": mean_of(metric_rows, "relative_error_ratio"),
            "max_relative_error_ratio": max(float(row["relative_error_ratio"]) for row in metric_rows),
            "mean_empirical_coverage_90": mean_of(metric_rows, "empirical_coverage_90"),
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
            "quantile_levels": quantile_levels if raw_rows else [],
            "window_manifest": str(manifest_path) if manifest_path is not None else None,
            "manifest_requested_windows": manifest_requested_windows,
            "manifest_matched_windows": matched_manifest_windows if manifest_path is not None else None,
            "raw_forecasts": str(raw_path),
            "history_context_sidecar": str(history_path) if args.export_history_sidecar else None,
            "window_metrics": str(metric_path),
            "failure_mining": str(failure_path),
            "failure_summary": str(summary_path),
            "limitations": ["limited-series CPU rerun", "TimesFM local memory-gated path"],
        },
    )


if __name__ == "__main__":
    main()
