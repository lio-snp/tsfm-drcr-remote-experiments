#!/usr/bin/env python
"""Run a small FRED-based finance stress-test mirror.

The goal is not to make a trading claim. This script creates a reproducible
finance seed that mirrors the raw forecast contract used by the GIFT-Eval
reruns: raw forecasts, window metrics, failure summary, and DM/FDR statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.baselines import all_simple_baselines
from low_snr_tsfm.evaluation import ForecastWindow, rolling_windows
from low_snr_tsfm.features import feature_vector
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
from low_snr_tsfm.stats import benjamini_hochberg, diebold_mariano


RAW_DIR = ROOT / "results" / "raw_forecasts"
METRIC_DIR = ROOT / "results" / "window_metrics"
FAILURE_DIR = ROOT / "results" / "failure_mining"
STAT_DIR = ROOT / "results" / "statistics"
DATA_DIR = ROOT / "data" / "finance_fred"

FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


@dataclass(frozen=True)
class FredSpec:
    series_id: str
    label: str
    domain: str
    regime: str
    target: str


DEFAULT_SPECS = [
    FredSpec("SP500", "sp500_return", "finance_equity", "low_snr_returns", "log_return"),
    FredSpec("NASDAQCOM", "nasdaq_return", "finance_equity", "low_snr_returns", "log_return"),
    FredSpec("DJIA", "dow_jones_return", "finance_equity", "low_snr_returns", "log_return"),
    FredSpec("DEXUSEU", "eur_usd_return", "finance_fx", "low_snr_returns", "log_return"),
    FredSpec("DEXJPUS", "jpy_usd_return", "finance_fx", "low_snr_returns", "log_return"),
    FredSpec("DEXUSUK", "gbp_usd_return", "finance_fx", "low_snr_returns", "log_return"),
    FredSpec("DEXCAUS", "cad_usd_return", "finance_fx", "low_snr_returns", "log_return"),
    FredSpec("DEXCHUS", "cny_usd_return", "finance_fx", "low_snr_returns", "log_return"),
    FredSpec("DCOILWTICO", "wti_oil_return", "finance_commodity", "low_snr_returns", "log_return"),
    FredSpec("DHHNGSP", "henry_hub_gas_return", "finance_commodity", "low_snr_returns", "log_return"),
    FredSpec("DCOILBRENTEU", "brent_oil_return", "finance_commodity", "low_snr_returns", "log_return"),
    FredSpec("VIXCLS", "vix_log_level", "finance_volatility", "medium_snr_volatility", "log_level"),
    FredSpec("OVXCLS", "oil_vol_log_level", "finance_volatility", "medium_snr_volatility", "log_level"),
    FredSpec("GVZCLS", "gold_vol_log_level", "finance_volatility", "medium_snr_volatility", "log_level"),
]


def clean_slug(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").lower()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
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


def load_window_manifest(path: str | None) -> set[tuple[str, int]] | None:
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    allowed: set[tuple[str, int]] = set()
    with manifest_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                window_index = int(row["window_index"])
            except (KeyError, ValueError):
                continue
            series_id = row.get("series_id", "")
            if series_id:
                allowed.add((series_id, window_index))
    if not allowed:
        raise ValueError(f"No usable finance windows found in manifest: {manifest_path}")
    return allowed


def repo_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - provenance is best effort for local clones
        return "unknown"


def missing_imports(backend: str) -> list[dict[str, str]]:
    missing = []
    packages = ["torch", "chronos"] if backend == "chronos" else ["torch", "timesfm"]
    for name in packages:
        try:
            __import__(name)
        except Exception as exc:  # noqa: BLE001 - optional runtime dependency report
            missing.append({"package": name, "error": f"{type(exc).__name__}: {exc}"})
    return missing


def quantile_triplet(quantiles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(quantiles, dtype=float)
    if q.ndim != 2:
        raise ValueError(f"Expected quantiles with shape horizon x quantile, got {q.shape}")
    if q.shape[1] >= 10:
        return q[:, 1], q[:, 5], q[:, 9]
    if q.shape[1] >= 9:
        return q[:, 0], q[:, 4], q[:, 8]
    if q.shape[1] >= 3:
        return q[:, 0], q[:, 1], q[:, 2]
    raise ValueError(f"Quantile output has too few columns: {q.shape}")


def download_fred_csv(series_id: str, cache_dir: Path, refresh: bool) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{series_id}.csv"
    if path.exists() and not refresh:
        return path
    url = FRED_URL.format(series_id=series_id)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - locked public FRED URL
            content = response.read()
        path.write_bytes(content)
    except Exception:
        if path.exists():
            return path
        raise
    return path


def read_fred_values(path: Path, series_id: str, start_date: str, end_date: str) -> tuple[list[str], np.ndarray]:
    dates: list[str] = []
    values: list[float] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            obs_date = row["observation_date"]
            if obs_date < start_date or obs_date > end_date:
                continue
            raw_value = row.get(series_id, "")
            if raw_value in {"", "."}:
                continue
            try:
                value = float(raw_value)
            except ValueError:
                continue
            if not np.isfinite(value) or value <= 0.0:
                continue
            dates.append(obs_date)
            values.append(value)
    return dates, np.asarray(values, dtype=float)


def transform_series(dates: list[str], values: np.ndarray, target: str) -> tuple[list[str], np.ndarray]:
    if target == "log_return":
        transformed = np.diff(np.log(values))
        return dates[1:], transformed.astype(float)
    if target == "log_level":
        return dates, np.log(values).astype(float)
    raise ValueError(f"Unknown target transform: {target}")


def select_best_baseline(
    context: np.ndarray,
    target: np.ndarray,
    horizon: int,
    season_length: int | None,
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
        error = mae(target, candidate.values)
        if error < best_error:
            best_name = candidate.name
            best_values = candidate.values
            best_error = error
    if not best_name:
        raise ValueError("No finite baseline comparison is available")
    return best_name, best_values, best_error


def choose_tail_windows(
    values: np.ndarray,
    context_length: int,
    horizon: int,
    step: int,
    max_windows: int,
) -> list[ForecastWindow]:
    windows = list(rolling_windows(values, context_length=context_length, horizon=horizon, step=step))
    if max_windows > 0:
        windows = windows[-max_windows:]
    return windows


def mean_of(rows: list[dict[str, object]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / max(len(rows), 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["chronos", "timesfm"], default="chronos")
    parser.add_argument("--model-id", default="amazon/chronos-bolt-small")
    parser.add_argument("--model-name", default="chronos_bolt_small")
    parser.add_argument("--start-date", default="2016-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--step", type=int, default=63)
    parser.add_argument("--max-windows-per-series", type=int, default=4)
    parser.add_argument("--season-length", type=int, default=5)
    parser.add_argument("--window-manifest", default=None)
    parser.add_argument("--output-slug", default=None)
    parser.add_argument(
        "--export-history-sidecar",
        action="store_true",
        help="Write one sidecar row per finance forecast origin with exact context/baseline history values.",
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    model_slug = clean_slug(args.model_name)
    output_slug = args.output_slug or f"{model_slug}_finance_fred_stress"
    status_path = RAW_DIR / f"{output_slug}_status.json"
    allowed_windows = load_window_manifest(args.window_manifest)

    missing = missing_imports(args.backend)
    if missing:
        write_status(
            status_path,
            "blocked_missing_dependencies",
            {"missing": missing, "setup_command": "scripts/setup_chronos_env.sh or install external/timesfm[torch]"},
        )
        return

    import torch
    if args.backend == "chronos":
        from chronos import BaseChronosPipeline

        pipeline = BaseChronosPipeline.from_pretrained(args.model_id, device_map="cpu")
        model_version = getattr(pipeline.model.config, "_name_or_path", args.model_id)
        source_commit = f"chronos-forecasting:{repo_commit(ROOT / 'external' / 'chronos-forecasting')}"
    else:
        import timesfm

        pipeline = timesfm.TimesFM_2p5_200M_torch.from_pretrained(args.model_id)
        pipeline.compile(
            timesfm.ForecastConfig(
                max_context=args.context_length,
                max_horizon=args.horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        model_version = args.model_id
        source_commit = f"timesfm:{repo_commit(ROOT / 'external' / 'timesfm')}"
    quantile_levels = [0.1, 0.5, 0.9]
    run_id = f"{output_slug}_{int(time.time())}"

    raw_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    data_rows: list[dict[str, object]] = []

    for spec in DEFAULT_SPECS:
        try:
            csv_path = download_fred_csv(spec.series_id, DATA_DIR, refresh=args.refresh)
            dates, values = read_fred_values(csv_path, spec.series_id, args.start_date, args.end_date)
        except Exception as exc:  # noqa: BLE001 - public data mirror should degrade per series
            data_rows.append(
                {
                    "series_id": spec.series_id,
                    "label": spec.label,
                    "status": "skipped_download_or_parse_failed",
                    "target": spec.target,
                    "error": f"{type(exc).__name__}: {exc}",
                    "source_url": FRED_URL.format(series_id=spec.series_id),
                }
            )
            continue
        target_dates, series = transform_series(dates, values, spec.target)
        if series.size < args.context_length + args.horizon:
            data_rows.append(
                {
                    "series_id": spec.series_id,
                    "label": spec.label,
                    "status": "skipped_too_short",
                    "n_points": int(series.size),
                }
            )
            continue
        windows = choose_tail_windows(
            series,
            context_length=args.context_length,
            horizon=args.horizon,
            step=args.step,
            max_windows=args.max_windows_per_series,
        )
        data_rows.append(
            {
                "series_id": spec.series_id,
                "label": spec.label,
                "status": "ok",
                "target": spec.target,
                "n_points": int(series.size),
                "first_date": target_dates[0],
                "last_date": target_dates[-1],
                "windows_run": len(windows),
                "source_url": FRED_URL.format(series_id=spec.series_id),
            }
        )

        for window_index, window in enumerate(windows):
            if allowed_windows is not None and (spec.label, window_index) not in allowed_windows:
                continue
            origin_date = target_dates[window.origin]
            context = np.asarray(window.context, dtype=float)
            target = np.asarray(window.target, dtype=float)
            baseline_name, baseline_values, baseline_mae = select_best_baseline(
                context,
                target,
                horizon=args.horizon,
                season_length=args.season_length,
            )
            if args.backend == "chronos":
                quantiles, mean = pipeline.predict_quantiles(
                    torch.tensor(context, dtype=torch.float32),
                    prediction_length=args.horizon,
                    quantile_levels=quantile_levels,
                )
                q = quantiles[0].detach().cpu().numpy()
                mean_forecast = mean[0].detach().cpu().numpy()
            else:
                point, quantiles = pipeline.forecast(horizon=args.horizon, inputs=[context])
                mean_forecast = np.asarray(point[0], dtype=float)
                q10, q50, q90 = quantile_triplet(np.asarray(quantiles[0], dtype=float))
                q = np.stack([q10, q50, q90], axis=1)
            model_mae = mae(target, mean_forecast)
            model_mase = mase(target, mean_forecast, context, season_length=1)
            baseline_mase = mase(target, baseline_values, context, season_length=1)
            flags = classify_degeneration(
                target,
                mean_forecast,
                model_error=model_mae,
                baseline_error=baseline_mae,
            )
            rer = relative_error_ratio(model_mae, baseline_mae)
            metric_row = {
                "run_id": run_id,
                "dataset": "FRED finance stress seed",
                "series_id": spec.label,
                "fred_series_id": spec.series_id,
                "domain": spec.domain,
                "regime": spec.regime,
                "target_transform": spec.target,
                "model": args.model_name,
                "baseline": baseline_name,
                "origin": int(window.origin),
                "origin_date": origin_date,
                "window_index": window_index,
                "context_length": args.context_length,
                "horizon": args.horizon,
                "mae": model_mae,
                "rmse": rmse(target, mean_forecast),
                "mase": model_mase,
                "baseline_mae": baseline_mae,
                "baseline_mase": baseline_mase,
                "relative_error_ratio": rer,
                "forecast_variance_ratio": forecast_variance_ratio(target, mean_forecast),
                "prediction_amplitude_ratio": prediction_amplitude_ratio(target, mean_forecast),
                "flatness_score": flatness_score(target, mean_forecast),
                "spike_recall": spike_recall(target, mean_forecast, k=3),
                "empirical_coverage_90": empirical_coverage(target, q[:, 0], q[:, 2]),
                "failure_delta_005": int(rer > 1.05),
                "excess_variance": int(flags.excess_variance),
                "over_smoothing": int(flags.over_smoothing),
            }
            metric_rows.append(metric_row)
            feature_rows.append(
                {
                    **{
                        key: metric_row[key]
                        for key in [
                            "run_id",
                            "series_id",
                            "fred_series_id",
                            "domain",
                            "regime",
                            "target_transform",
                            "origin",
                            "origin_date",
                            "window_index",
                            "horizon",
                            "context_length",
                            "failure_delta_005",
                            "relative_error_ratio",
                        ]
                    },
                    **feature_vector(
                        context,
                        horizon=args.horizon,
                        context_length=args.context_length,
                        period=args.season_length,
                    ),
                }
            )
            if args.export_history_sidecar:
                history_rows.append(
                    history_context_sidecar_row(
                        run_id=run_id,
                        dataset="FRED finance stress seed",
                        series_id=spec.label,
                        model=args.model_name,
                        baseline_family=baseline_name,
                        baseline_mode="best_simple",
                        origin=int(window.origin),
                        window_index=window_index,
                        context=context,
                        full_context=context,
                        baseline_context=context,
                        target=target,
                        baseline_season_length=args.season_length,
                        baseline_context_cap=args.context_length,
                        source_commit=source_commit,
                        model_id=args.model_id,
                    )
                )

            for idx in range(args.horizon):
                raw_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": "FRED finance stress seed",
                        "series_id": spec.label,
                        "fred_series_id": spec.series_id,
                        "domain": spec.domain,
                        "regime": spec.regime,
                        "target_transform": spec.target,
                        "model": args.model_name,
                        "baseline_family": baseline_name,
                        "baseline_mode": "best_simple",
                        "origin": int(window.origin),
                        "origin_date": origin_date,
                        "target_date": target_dates[window.origin + idx],
                        "window_index": window_index,
                        "horizon_index": idx + 1,
                        "context_length": args.context_length,
                        "horizon": args.horizon,
                        "actual": float(target[idx]),
                        "forecast_mean": float(mean_forecast[idx]),
                        "forecast_median": float(q[idx, 1]),
                        "forecast_q10": float(q[idx, 0]),
                        "forecast_q50": float(q[idx, 1]),
                        "forecast_q90": float(q[idx, 2]),
                        "baseline_forecast": float(baseline_values[idx]),
                        "source_commit": source_commit,
                        "source_url": FRED_URL.format(series_id=spec.series_id),
                        "model_id": args.model_id,
                        "model_version": model_version,
                    }
                )

    if not raw_rows or not metric_rows:
        write_status(
            status_path,
            "blocked_no_rows",
            {
                "start_date": args.start_date,
                "end_date": args.end_date,
                "data_manifest": data_rows,
            },
        )
        return

    stat_rows = []
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in raw_rows:
        grouped.setdefault(str(row["series_id"]), []).append(row)
    p_values = []
    for series_id, rows in sorted(grouped.items()):
        model_loss = np.asarray([abs(float(row["actual"]) - float(row["forecast_mean"])) for row in rows])
        baseline_loss = np.asarray([abs(float(row["actual"]) - float(row["baseline_forecast"])) for row in rows])
        dm = diebold_mariano(model_loss, baseline_loss, horizon=args.horizon, alternative="two-sided")
        stat_row = {
            "run_id": run_id,
            "series_id": series_id,
            "domain": rows[0]["domain"],
            "regime": rows[0]["regime"],
            "n_loss_points": len(rows),
            "mean_model_loss": float(np.mean(model_loss)),
            "mean_baseline_loss": float(np.mean(baseline_loss)),
            "mean_loss_ratio": relative_error_ratio(float(np.mean(model_loss)), float(np.mean(baseline_loss))),
            "dm_statistic": dm.statistic,
            "dm_p_value": dm.p_value,
        }
        stat_rows.append(stat_row)
        p_values.append(dm.p_value)
    rejected, q_values = benjamini_hochberg(np.asarray(p_values), alpha=0.10)
    for row, reject, q_value in zip(stat_rows, rejected, q_values):
        row["bh_fdr_q_value"] = float(q_value)
        row["bh_fdr_reject_010"] = int(bool(reject))

    failure_rows = sorted(metric_rows, key=lambda row: float(row["relative_error_ratio"]), reverse=True)
    summary_rows = [
        {
            "run_id": run_id,
            "model": args.model_name,
            "dataset": "FRED finance stress seed",
            "n_series": len({row["series_id"] for row in metric_rows}),
            "n_windows": len(metric_rows),
            "n_raw_rows": len(raw_rows),
            "horizon": args.horizon,
            "context_length": args.context_length,
            "failure_rate_delta_005": mean_of(metric_rows, "failure_delta_005"),
            "mean_relative_error_ratio": mean_of(metric_rows, "relative_error_ratio"),
            "max_relative_error_ratio": max(float(row["relative_error_ratio"]) for row in metric_rows),
            "over_smoothing_rate": mean_of(metric_rows, "over_smoothing"),
            "excess_variance_rate": mean_of(metric_rows, "excess_variance"),
            "mean_empirical_coverage_90": mean_of(metric_rows, "empirical_coverage_90"),
            "mean_flatness_score": mean_of(metric_rows, "flatness_score"),
            "top_failure_series_id": failure_rows[0]["series_id"],
            "top_failure_origin_date": failure_rows[0]["origin_date"],
            "top_failure_baseline": failure_rows[0]["baseline"],
        }
    ]
    for regime in sorted({str(row["regime"]) for row in metric_rows}):
        group = [row for row in metric_rows if row["regime"] == regime]
        summary_rows.append(
            {
                "run_id": run_id,
                "model": args.model_name,
                "dataset": "FRED finance stress seed",
                "regime": regime,
                "n_series": len({row["series_id"] for row in group}),
                "n_windows": len(group),
                "n_raw_rows": len(group) * args.horizon,
                "horizon": args.horizon,
                "context_length": args.context_length,
                "failure_rate_delta_005": mean_of(group, "failure_delta_005"),
                "mean_relative_error_ratio": mean_of(group, "relative_error_ratio"),
                "max_relative_error_ratio": max(float(row["relative_error_ratio"]) for row in group),
                "over_smoothing_rate": mean_of(group, "over_smoothing"),
                "excess_variance_rate": mean_of(group, "excess_variance"),
                "mean_empirical_coverage_90": mean_of(group, "empirical_coverage_90"),
                "mean_flatness_score": mean_of(group, "flatness_score"),
                "top_failure_series_id": max(group, key=lambda row: float(row["relative_error_ratio"]))["series_id"],
                "top_failure_origin_date": max(group, key=lambda row: float(row["relative_error_ratio"]))[
                    "origin_date"
                ],
                "top_failure_baseline": max(group, key=lambda row: float(row["relative_error_ratio"]))["baseline"],
            }
        )

    raw_path = RAW_DIR / f"{output_slug}.csv"
    metric_path = METRIC_DIR / f"{output_slug}_metrics.csv"
    failure_path = FAILURE_DIR / f"{output_slug}_window_failures.csv"
    summary_path = FAILURE_DIR / f"{output_slug}_failure_summary.csv"
    feature_path = FAILURE_DIR / f"{output_slug}_predictor_features.csv"
    stat_path = STAT_DIR / f"{output_slug}_dm_fdr.csv"
    history_path = RAW_DIR / f"{output_slug}_history_context.csv"
    manifest_path = ROOT / "results" / "reproduction" / "finance_fred_stress_manifest.json"

    write_csv(raw_path, raw_rows)
    write_csv(metric_path, metric_rows)
    write_csv(failure_path, failure_rows)
    write_csv(summary_path, summary_rows)
    write_csv(feature_path, feature_rows)
    write_csv(stat_path, stat_rows)
    if args.export_history_sidecar:
        write_csv(history_path, history_rows)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source": "FRED public CSV via fred.stlouisfed.org",
                "backend": args.backend,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "model_id": args.model_id,
                "context_length": args.context_length,
                "horizon": args.horizon,
                "step": args.step,
                "max_windows_per_series": args.max_windows_per_series,
                "window_manifest": args.window_manifest,
                "history_context_sidecar": str(history_path) if args.export_history_sidecar else None,
                "series": data_rows,
                "limitations": [
                    "small public FRED fallback, not point-in-time equity universe",
                    "daily close/index data only",
                    "not a trading or deployment claim",
                    f"limited CPU {args.backend} seed for finance alignment",
                ],
            },
            indent=2,
        )
    )

    write_status(
        status_path,
        "ok",
        {
            "backend": args.backend,
            "model_id": args.model_id,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "series_run": len({row["series_id"] for row in metric_rows}),
            "windows_run": len(metric_rows),
            "raw_rows": len(raw_rows),
            "raw_forecasts": str(raw_path),
            "window_metrics": str(metric_path),
            "window_manifest": args.window_manifest,
            "failure_mining": str(failure_path),
            "failure_summary": str(summary_path),
            "predictor_features": str(feature_path),
            "dm_fdr": str(stat_path),
            "history_context_sidecar": str(history_path) if args.export_history_sidecar else None,
            "manifest": str(manifest_path),
            "limitations": [
                "small public FRED fallback, not point-in-time equity universe",
                "daily close/index data only",
                "not a trading or deployment claim",
                f"limited CPU {args.backend} seed for finance alignment",
            ],
        },
    )


if __name__ == "__main__":
    main()
