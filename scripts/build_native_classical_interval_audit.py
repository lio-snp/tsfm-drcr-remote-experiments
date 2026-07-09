#!/usr/bin/env python3
"""Fit sidecar-backed native-ish classical interval baselines.

This audit uses the recovered history/context sidecars to build a stronger
classical probabilistic baseline than deterministic point fallback.  By default
it uses sidecar empirical residual quantiles around the stored classical point
forecast; a bounded state-space ETS probe can be enabled by increasing
MAX_NATIVE_ETS_FITS.  The output is intentionally reported as an audit, not as a
strict AutoARIMA/AutoETS-native-interval reproduction.
"""

from __future__ import annotations

import ast
import csv
import math
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.metrics import empirical_coverage, mae, mean_weighted_quantile_loss, relative_error_ratio, rmse, wape  # noqa: E402

csv.field_size_limit(1024 * 1024 * 1024)

OUT = ROOT / "results" / "aaai_stress"
RAW = ROOT / "results" / "raw_forecasts"
DOCS = ROOT / "docs"

SPLIT = OUT / "split_manifest.csv"
SIDECAR_STATUS = OUT / "history_sidecar_reconstruction_status.csv"
WINDOWS_OUT = OUT / "native_classical_interval_audit_windows.csv"
SUMMARY_OUT = OUT / "native_classical_interval_audit_summary.csv"
STATUS_OUT = OUT / "native_classical_interval_audit_status.csv"
DOC_OUT = DOCS / "native_classical_interval_audit.md"

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NOMINAL = 0.8
EPS = 1e-12
MAX_NATIVE_ETS_FITS = 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_values(text: str) -> np.ndarray:
    values = ast.literal_eval(text)
    return np.asarray(values, dtype=float)


def finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def raw_window_groups(path: Path) -> dict[tuple[str, int], list[dict[str, str]]]:
    groups: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(path):
        groups[(row["series_id"], int(row["window_index"]))].append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: int(row.get("horizon_index") or 0))
    return groups


def residual_quantile_grid(point: np.ndarray, context: np.ndarray, season_length: int) -> tuple[np.ndarray, str]:
    history = finite(context)
    if history.size < max(8, season_length + 2):
        residuals = history - np.nanmedian(history) if history.size else np.zeros(1)
    elif season_length > 1 and history.size > season_length:
        residuals = history[season_length:] - history[:-season_length]
    else:
        residuals = np.diff(history)
    residuals = finite(residuals)
    if residuals.size < 4:
        scale = float(np.nanstd(history)) if history.size else 0.0
        residuals = np.asarray([-scale, 0.0, scale], dtype=float)
    quantile_offsets = np.quantile(residuals, LEVELS)
    return np.sort(point[:, None] + quantile_offsets[None, :], axis=1), "empirical_residual_fallback"


def statsmodels_ets_grid(context: np.ndarray, horizon: int, season_length: int) -> tuple[np.ndarray, np.ndarray, str]:
    history = finite(context)
    if history.size < 12:
        raise ValueError("too few history points for state-space ETS")
    seasonal = int(season_length) if season_length and season_length > 1 and history.size >= 3 * season_length else None
    trend = bool(history.size >= 16)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        model = ExponentialSmoothing(
            history,
            trend=trend,
            damped_trend=trend,
            seasonal=seasonal,
            initialization_method="estimated",
        )
        result = model.fit(disp=False, maxiter=15, cov_type="none", low_memory=True)
        forecast = result.get_forecast(horizon)
        mean = np.asarray(forecast.predicted_mean, dtype=float)
        grids: list[np.ndarray] = []
        for level in LEVELS:
            alpha = 2.0 * min(level, 1.0 - level)
            if alpha <= 0:
                raise ValueError("invalid quantile level")
            interval = np.asarray(forecast.conf_int(alpha=alpha), dtype=float)
            if level < 0.5:
                grids.append(interval[:, 0])
            elif level > 0.5:
                grids.append(interval[:, 1])
            else:
                grids.append(mean)
        grid = np.column_stack(grids)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(grid)):
        raise ValueError("non-finite statsmodels ETS forecast")
    return mean, np.sort(grid, axis=1), "statsmodels_state_ets"


def native_classical_grid(
    *,
    point: np.ndarray,
    context: np.ndarray,
    horizon: int,
    season_length: int,
    allow_statsmodels: bool,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if not allow_statsmodels:
        grid, method = residual_quantile_grid(point, context, season_length)
        return point, grid, method, "native_fit_budget"
    try:
        mean, grid, method = statsmodels_ets_grid(context, horizon, season_length)
        return mean, grid, method, ""
    except Exception as exc:  # noqa: BLE001 - audit records fallback reasons
        grid, method = residual_quantile_grid(point, context, season_length)
        return point, grid, method, type(exc).__name__


def metric_row(
    *,
    actual: np.ndarray,
    baseline_point: np.ndarray,
    model_grid: np.ndarray,
    model_levels: list[float],
    classical_point: np.ndarray,
    classical_grid: np.ndarray,
) -> dict[str, float]:
    baseline_wql = mean_weighted_quantile_loss(
        actual,
        np.repeat(baseline_point[:, None], len(LEVELS), axis=1),
        LEVELS,
    )
    model_wql = mean_weighted_quantile_loss(actual, model_grid, model_levels)
    classical_wql = mean_weighted_quantile_loss(actual, classical_grid, LEVELS)
    q10 = classical_grid[:, 0]
    q50 = classical_grid[:, 4]
    q90 = classical_grid[:, 8]
    point_mae = mae(actual, classical_point)
    baseline_mae = mae(actual, baseline_point)
    point_rmse = rmse(actual, classical_point)
    baseline_rmse = rmse(actual, baseline_point)
    point_wape = wape(actual, classical_point)
    baseline_wape = wape(actual, baseline_point)
    return {
        "baseline_wql": baseline_wql,
        "model_wql": model_wql,
        "native_classical_wql": classical_wql,
        "native_classical_wql_rer": relative_error_ratio(classical_wql, baseline_wql),
        "model_wql_rer": relative_error_ratio(model_wql, baseline_wql),
        "native_classical_wql_delta_vs_model": classical_wql - model_wql,
        "native_classical_win_vs_model": float(classical_wql < model_wql),
        "native_classical_wql_harm": float(classical_wql > model_wql * 1.05),
        "native_classical_coverage_q10_q90": empirical_coverage(actual, q10, q90),
        "native_classical_interval_width_q10_q90": float(np.mean(q90 - q10)),
        "native_classical_mae": point_mae,
        "native_classical_mae_rer": relative_error_ratio(point_mae, baseline_mae),
        "native_classical_rmse": point_rmse,
        "native_classical_rmse_rer": relative_error_ratio(point_rmse, baseline_rmse),
        "native_classical_wape": point_wape,
        "native_classical_wape_rer": relative_error_ratio(point_wape, baseline_wape),
        "native_classical_q50_mae_rer": relative_error_ratio(mae(actual, q50), baseline_mae),
    }


def model_grid_from_rows(rows: list[dict[str, str]]) -> tuple[list[float], np.ndarray]:
    levels = []
    columns = []
    for level in LEVELS:
        key = f"forecast_q{int(level * 100)}"
        if key in rows[0]:
            values = [float(row[key]) for row in rows]
            if all(math.isfinite(value) for value in values):
                levels.append(level)
                columns.append(values)
    if not levels:
        levels = [0.1, 0.5, 0.9]
        columns = [
            [float(row["forecast_q10"]) for row in rows],
            [float(row.get("forecast_q50") or row.get("forecast_median") or row["forecast_mean"]) for row in rows],
            [float(row["forecast_q90"]) for row in rows],
        ]
    return levels, np.asarray(columns, dtype=float).T


def summarize(rows: list[dict[str, object]], group_key: str, group_value: str) -> dict[str, object]:
    selected = [row for row in rows if group_key == "overall" or str(row.get(group_key)) == group_value]
    if not selected:
        raise ValueError(f"empty group {group_key}:{group_value}")
    values = lambda key: [float(row[key]) for row in selected if math.isfinite(float(row[key]))]
    method_counts = Counter(str(row["interval_method"]) for row in selected)
    return {
        "group": "overall" if group_key == "overall" else f"{group_key}:{group_value}",
        "n_windows": len(selected),
        "median_native_classical_wql_rer": median(values("native_classical_wql_rer")),
        "median_model_wql_rer": median(values("model_wql_rer")),
        "median_native_classical_mae_rer": median(values("native_classical_mae_rer")),
        "median_native_classical_rmse_rer": median(values("native_classical_rmse_rer")),
        "median_native_classical_wape_rer": median(values("native_classical_wape_rer")),
        "mean_native_classical_coverage": float(np.mean(values("native_classical_coverage_q10_q90"))),
        "mean_native_classical_width": float(np.mean(values("native_classical_interval_width_q10_q90"))),
        "native_classical_win_rate_vs_model": float(np.mean(values("native_classical_win_vs_model"))),
        "native_classical_harm_rate_vs_model": float(np.mean(values("native_classical_wql_harm"))),
        "statsmodels_state_ets_rate": method_counts.get("statsmodels_state_ets", 0) / len(selected),
        "empirical_fallback_rate": method_counts.get("empirical_residual_fallback", 0) / len(selected),
    }


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")).replace("|", "\\|") for key, _ in columns) + " |")
    return "\n".join(lines)


def fmt(value: object, digits: int = 3) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{parsed:.{digits}f}" if math.isfinite(parsed) else "nan"


def pct(value: object, digits: int = 1) -> str:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{100.0 * parsed:.{digits}f}%" if math.isfinite(parsed) else "nan"


def main() -> None:
    split_rows = read_csv(SPLIT)
    split_by_key = {
        (row["source"], row["series_id"], int(row["window_index"])): row
        for row in split_rows
    }
    status_rows = read_csv(SIDECAR_STATUS)

    all_window_rows: list[dict[str, object]] = []
    source_status_rows: list[dict[str, object]] = []
    native_fit_attempts = 0
    for status in status_rows:
        source = status["source"]
        raw_path = RAW / f"{source}.csv"
        sidecar_path = ROOT / status["sidecar"]
        if not raw_path.exists() or not sidecar_path.exists():
            source_status_rows.append({"source": source, "status": "missing_raw_or_sidecar"})
            continue
        raw_groups = raw_window_groups(raw_path)
        source_rows: list[dict[str, object]] = []
        fallback_reasons: Counter[str] = Counter()
        for sidecar in read_csv(sidecar_path):
            key = (source, sidecar["series_id"], int(sidecar["window_index"]))
            split = split_by_key.get(key)
            if split is None:
                continue
            raw_rows = raw_groups.get((sidecar["series_id"], int(sidecar["window_index"])), [])
            if not raw_rows:
                continue
            actual = np.asarray([float(row["actual"]) for row in raw_rows], dtype=float)
            baseline_point = np.asarray([float(row["baseline_forecast"]) for row in raw_rows], dtype=float)
            model_levels, model_grid = model_grid_from_rows(raw_rows)
            context = parse_values(sidecar["baseline_context_values"] or sidecar["context_values"])
            horizon = len(raw_rows)
            season_length = int(float(sidecar.get("baseline_season_length") or 1))
            allow_statsmodels = native_fit_attempts < MAX_NATIVE_ETS_FITS and split["split"] == "test"
            if allow_statsmodels:
                native_fit_attempts += 1
            classical_point, classical_grid, method, fallback_reason = native_classical_grid(
                point=baseline_point,
                context=context,
                horizon=horizon,
                season_length=season_length,
                allow_statsmodels=allow_statsmodels,
            )
            if fallback_reason:
                fallback_reasons[fallback_reason] += 1
            metrics = metric_row(
                actual=actual,
                baseline_point=baseline_point,
                model_grid=model_grid,
                model_levels=model_levels,
                classical_point=classical_point,
                classical_grid=classical_grid,
            )
            window_row: dict[str, object] = {
                "source": source,
                "dataset": split["dataset"],
                "family": split["family"],
                "model": split["model"],
                "series_id": sidecar["series_id"],
                "window_index": int(sidecar["window_index"]),
                "split": split["split"],
                "role": split["role"],
                "target_id": split["target_id"],
                "evidence_tier": split["evidence_tier"],
                "quantile_grid_n_levels": split["quantile_grid_n_levels"],
                "baseline_mode": sidecar.get("baseline_mode", ""),
                "baseline_family": sidecar.get("baseline_family", ""),
                "baseline_season_length": season_length,
                "horizon": horizon,
                "context_length": len(context),
                "interval_method": method,
                "fallback_reason": fallback_reason,
                **metrics,
            }
            source_rows.append(window_row)
        all_window_rows.extend(source_rows)
        method_counts = Counter(str(row["interval_method"]) for row in source_rows)
        source_status_rows.append(
            {
                "source": source,
                "status": "ok" if source_rows else "no_matched_windows",
                "windows": len(source_rows),
                "statsmodels_state_ets": method_counts.get("statsmodels_state_ets", 0),
                "empirical_residual_fallback": method_counts.get("empirical_residual_fallback", 0),
                "fallback_reasons": ";".join(f"{key}:{value}" for key, value in fallback_reasons.most_common()),
                "sidecar_status": status["status"],
                "sidecar_mismatched_actual": status.get("mismatched_actual", "0"),
            }
        )

    if len(all_window_rows) < 1000:
        raise SystemExit(f"too few matched windows for native classical interval audit: {len(all_window_rows)}")

    write_csv(WINDOWS_OUT, all_window_rows)
    summary_rows = [
        summarize(all_window_rows, "overall", "overall"),
        summarize(all_window_rows, "split", "test"),
        summarize(all_window_rows, "split", "calibration"),
        summarize(all_window_rows, "role", "failure_target"),
        summarize(all_window_rows, "role", "positive_control"),
        summarize(all_window_rows, "role", "stress_target"),
        summarize(all_window_rows, "family", "chronos"),
        summarize(all_window_rows, "family", "moirai"),
        summarize(all_window_rows, "family", "timesfm"),
        summarize(all_window_rows, "evidence_tier", "q9_fullgrid"),
        summarize(all_window_rows, "evidence_tier", "q3_interval_proxy"),
    ]
    write_csv(SUMMARY_OUT, summary_rows)
    write_csv(STATUS_OUT, source_status_rows)

    display_rows = []
    for row in summary_rows:
        display_rows.append(
            {
                "Group": row["group"],
                "N": row["n_windows"],
                "Classical WQL-RER": fmt(row["median_native_classical_wql_rer"]),
                "Native TSFM WQL-RER": fmt(row["median_model_wql_rer"]),
                "Coverage": pct(row["mean_native_classical_coverage"]),
                "Harm vs TSFM": pct(row["native_classical_harm_rate_vs_model"]),
                "ETS fit": pct(row["statsmodels_state_ets_rate"]),
                "Fallback": pct(row["empirical_fallback_rate"]),
            }
        )
    lines = [
        "# Native Classical Interval Audit",
        "",
        "This audit uses recovered history/context sidecars to fit a sidecar-backed empirical classical probabilistic baseline. It uses empirical residual quantiles around the stored classical point forecast; a bounded state-space ETS probe is available in the script but disabled by default because local state-space fitting is too slow on this machine. It is a reviewer-facing robustness audit, not a strict reproduction of every benchmark's native AutoARIMA/AutoETS interval implementation.",
        "",
        "## Summary",
        "",
        f"- Windows audited: `{len(all_window_rows)}`.",
        f"- Sources audited: `{len(source_status_rows)}`.",
        f"- Statsmodels state-space ETS fit rate: `{pct(summary_rows[0]['statsmodels_state_ets_rate'])}`.",
        f"- Statsmodels native-fit budget: `{MAX_NATIVE_ETS_FITS}` test windows.",
        f"- Empirical residual fallback rate: `{pct(summary_rows[0]['empirical_fallback_rate'])}`.",
        f"- One known sidecar warning remains for finance FRED: `{next((row['sidecar_mismatched_actual'] for row in source_status_rows if row['source'] == 'timesfm_2_5_finance_fred_finance_fred_stress'), '0')}` mismatched raw actual values.",
        "",
        "## Main Groups",
        "",
        markdown_table(
            display_rows,
            [
                ("Group", "Group"),
                ("N", "N"),
                ("Classical WQL-RER", "Classical WQL-RER"),
                ("Native TSFM WQL-RER", "Native TSFM WQL-RER"),
                ("Coverage", "Coverage"),
                ("Harm vs TSFM", "Harm vs TSFM"),
                ("ETS fit", "ETS fit"),
                ("Fallback", "Fallback"),
            ],
        ),
        "",
        "## Paper-Safe Interpretation",
        "",
        "- This closes the prior artifact-level history/context blocker for current-suite classical probabilistic interval audits.",
        "- It does not justify claiming exact benchmark-native AutoARIMA/AutoETS interval parity, because the original raw runners only stored point classical forecasts.",
        "- Use this as an objective classical probabilistic baseline robustness table; keep strict native-interval parity as a future/external-run gap.",
        "",
        "## Artifacts",
        "",
        f"- `{WINDOWS_OUT.relative_to(ROOT)}`",
        f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
        f"- `{STATUS_OUT.relative_to(ROOT)}`",
        f"- `{DOC_OUT.relative_to(ROOT)}`",
        "",
    ]
    DOC_OUT.write_text("\n".join(lines))
    print({"status": "ok", "windows": len(all_window_rows), "sources": len(source_status_rows)})


if __name__ == "__main__":
    main()
