#!/usr/bin/env python
"""Run a BMA-style post-hoc repair probe on raw forecast artifacts.

This is a local transfer probe for the traffic-regime BMA idea: mix the TSFM
forecast with the already-computed non-leaky local baseline forecast. It is not
a claim of exact paper reproduction because it uses deterministic baseline
forecasts rather than traffic-specific historical conditional samples.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.metrics import (
    empirical_coverage,
    flatness_score,
    forecast_variance_ratio,
    mae,
    prediction_amplitude_ratio,
    relative_error_ratio,
    rmse,
)
from low_snr_tsfm.repair import blended_interval, convex_mixture, hull_interval


OUT_DIR = ROOT / "results" / "repair"
REQUIRED_RAW_FIELDS = {
    "actual",
    "baseline_forecast",
    "forecast_mean",
    "forecast_q10",
    "forecast_q90",
    "horizon_index",
    "series_id",
    "window_index",
}


def read_raw(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(REQUIRED_RAW_FIELDS - set(reader.fieldnames or []))
        if missing:
            raise SystemExit(f"{path} is missing required raw fields for repair probe: {missing}")
        return list(reader)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_windows(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (
            row.get("dataset", ""),
            row["series_id"],
            row.get("origin", ""),
            row["window_index"],
        )
        groups.setdefault(key, []).append(row)
    for key in groups:
        groups[key].sort(key=lambda row: int(row["horizon_index"]))
    return groups


def mean_of(rows: list[dict[str, object]], key: str) -> float:
    return float(sum(float(row[key]) for row in rows) / max(len(rows), 1))


def rate_of(rows: list[dict[str, object]], key: str) -> float:
    return mean_of(rows, key)


def summarize(window_rows: list[dict[str, object]], weight: float) -> dict[str, object]:
    model_rers = [float(row["model_relative_error_ratio"]) for row in window_rows]
    repair_rers = [float(row["repair_relative_error_ratio"]) for row in window_rows]
    return {
        "weight": weight,
        "n_windows": len(window_rows),
        "model_failure_rate_delta_005": float(np.mean(np.asarray(model_rers) > 1.05)),
        "repair_failure_rate_delta_005": float(np.mean(np.asarray(repair_rers) > 1.05)),
        "model_mean_relative_error_ratio": float(np.mean(model_rers)),
        "model_median_relative_error_ratio": float(statistics.median(model_rers)),
        "repair_mean_relative_error_ratio": float(np.mean(repair_rers)),
        "repair_median_relative_error_ratio": float(statistics.median(repair_rers)),
        "repair_win_rate_vs_model": rate_of(window_rows, "repair_improves_model"),
        "repair_win_rate_vs_baseline": rate_of(window_rows, "repair_beats_baseline"),
        "model_mean_empirical_coverage_90": mean_of(window_rows, "model_empirical_coverage_90"),
        "repair_blend_mean_empirical_coverage_90": mean_of(window_rows, "repair_blend_empirical_coverage_90"),
        "repair_hull_mean_empirical_coverage_90": mean_of(window_rows, "repair_hull_empirical_coverage_90"),
        "repair_mean_flatness_score": mean_of(window_rows, "repair_flatness_score"),
        "repair_mean_forecast_variance_ratio": mean_of(window_rows, "repair_forecast_variance_ratio"),
        "repair_mean_prediction_amplitude_ratio": mean_of(window_rows, "repair_prediction_amplitude_ratio"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Raw forecast CSV containing baseline_forecast")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--weights", default="0.25,0.5,0.75")
    args = parser.parse_args()

    raw_path = ROOT / args.raw
    weights = [float(value) for value in args.weights.split(",")]
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise SystemExit("--weights must contain values in [0, 1]")

    raw_rows = read_raw(raw_path)
    groups = group_windows(raw_rows)
    window_metric_rows: list[dict[str, object]] = []
    repaired_raw_rows: list[dict[str, object]] = []

    for key, rows in sorted(groups.items()):
        dataset, series_id, origin, window_index = key
        actual = np.asarray([float(row["actual"]) for row in rows])
        model = np.asarray([float(row["forecast_mean"]) for row in rows])
        baseline = np.asarray([float(row["baseline_forecast"]) for row in rows])
        q10 = np.asarray([float(row["forecast_q10"]) for row in rows])
        q90 = np.asarray([float(row["forecast_q90"]) for row in rows])
        model_mae = mae(actual, model)
        baseline_mae = mae(actual, baseline)
        model_rer = relative_error_ratio(model_mae, baseline_mae)
        model_coverage = empirical_coverage(actual, q10, q90)

        for weight in weights:
            repaired = convex_mixture(model, baseline, weight)
            blend_q10, blend_q90 = blended_interval(q10, q90, baseline, weight)
            hull_q10, hull_q90 = hull_interval(q10, q90, baseline)
            repair_mae = mae(actual, repaired)
            repair_rer = relative_error_ratio(repair_mae, baseline_mae)
            metric_row = {
                "dataset": dataset,
                "series_id": series_id,
                "domain": rows[0].get("domain", ""),
                "regime": rows[0].get("regime", ""),
                "model": rows[0].get("model", ""),
                "baseline_family": rows[0].get("baseline_family", ""),
                "origin": origin,
                "window_index": window_index,
                "weight": weight,
                "horizon": len(rows),
                "model_mae": model_mae,
                "baseline_mae": baseline_mae,
                "repair_mae": repair_mae,
                "model_rmse": rmse(actual, model),
                "repair_rmse": rmse(actual, repaired),
                "model_relative_error_ratio": model_rer,
                "repair_relative_error_ratio": repair_rer,
                "repair_to_model_error_ratio": relative_error_ratio(repair_mae, model_mae),
                "repair_improves_model": int(repair_mae < model_mae),
                "repair_beats_baseline": int(repair_mae < baseline_mae),
                "model_empirical_coverage_90": model_coverage,
                "repair_blend_empirical_coverage_90": empirical_coverage(actual, blend_q10, blend_q90),
                "repair_hull_empirical_coverage_90": empirical_coverage(actual, hull_q10, hull_q90),
                "repair_forecast_variance_ratio": forecast_variance_ratio(actual, repaired),
                "repair_prediction_amplitude_ratio": prediction_amplitude_ratio(actual, repaired),
                "repair_flatness_score": flatness_score(actual, repaired),
                "repair_failure_delta_005": int(repair_rer > 1.05),
            }
            window_metric_rows.append(metric_row)

            for row, repair_value, lo, hi, hull_lo, hull_hi in zip(
                rows,
                repaired,
                blend_q10,
                blend_q90,
                hull_q10,
                hull_q90,
            ):
                repaired_raw_rows.append(
                    {
                        "dataset": dataset,
                        "series_id": series_id,
                        "domain": row.get("domain", ""),
                        "regime": row.get("regime", ""),
                        "model": row.get("model", ""),
                        "baseline_family": row.get("baseline_family", ""),
                        "origin": origin,
                        "window_index": window_index,
                        "horizon_index": row["horizon_index"],
                        "weight": weight,
                        "actual": row["actual"],
                        "forecast_mean": row["forecast_mean"],
                        "baseline_forecast": row["baseline_forecast"],
                        "repair_mean": float(repair_value),
                        "repair_blend_q10": float(lo),
                        "repair_blend_q90": float(hi),
                        "repair_hull_q10": float(hull_lo),
                        "repair_hull_q90": float(hull_hi),
                    }
                )

    summary_rows = []
    for weight in weights:
        rows = [row for row in window_metric_rows if float(row["weight"]) == weight]
        summary_rows.append(
            {
                **summarize(rows, weight),
                "raw_input": str(raw_path),
                "limitations": (
                    "BMA-style deterministic transfer probe; exact traffic-paper BMA requires "
                    "traffic regime labels and historical conditional sample distributions"
                ),
            }
        )

    raw_out = OUT_DIR / f"{args.output_prefix}_bma_repair_raw.csv"
    metric_out = OUT_DIR / f"{args.output_prefix}_bma_repair_window_metrics.csv"
    summary_out = OUT_DIR / f"{args.output_prefix}_bma_repair_summary.csv"
    status_out = OUT_DIR / f"{args.output_prefix}_bma_repair_status.json"
    write_csv(raw_out, repaired_raw_rows)
    write_csv(metric_out, window_metric_rows)
    write_csv(summary_out, summary_rows)
    status_out.write_text(
        json.dumps(
            {
                "status": "ok",
                "timestamp": int(time.time()),
                "raw_input": str(raw_path),
                "weights": weights,
                "windows": len(groups),
                "raw_rows": len(repaired_raw_rows),
                "window_metrics": str(metric_out),
                "summary": str(summary_out),
            },
            indent=2,
        )
    )
    print(status_out.read_text())


if __name__ == "__main__":
    main()
