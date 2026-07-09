"""Predefined selectors for paper-ready forecast failure examples."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import median
from typing import Callable, Iterable


MetricRow = dict[str, str]


def read_metric_rows(paths: Iterable[Path]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row = dict(row)
                row["source_metric_path"] = str(path)
                row["raw_forecast_path"] = infer_raw_path(path)
                rows.append(row)
    return rows


def infer_raw_path(metric_path: Path) -> str:
    name = metric_path.name
    if metric_path.parent.name == "window_metrics" and name.endswith("_metrics.csv"):
        return str(Path("results/raw_forecasts") / f"{name.removesuffix('_metrics.csv')}.csv")
    if metric_path.parent.name == "traffic" and name.endswith("_window_metrics.csv"):
        return str(metric_path.with_name(f"{name.removesuffix('_window_metrics.csv')}_raw.csv"))
    return ""


def numeric(row: MetricRow, key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def failed(row: MetricRow) -> bool:
    return numeric(row, "failure_delta_005") > 0.5


def selector_identity(row: MetricRow) -> tuple[str, str, str, str, str, str]:
    return (
        row.get("dataset", ""),
        row.get("model", ""),
        row.get("baseline", row.get("baseline_mode", "")),
        row.get("series_id", ""),
        row.get("origin", ""),
        row.get("window_index", ""),
    )


def selected_case(selector: str, metric: str, value: float, row: MetricRow) -> MetricRow:
    output = dict(row)
    output["selector"] = selector
    output["selection_metric"] = metric
    output["selection_value"] = f"{value:.12g}"
    output["case_key"] = "|".join(selector_identity(row))
    return output


def argmax(rows: list[MetricRow], key: Callable[[MetricRow], float]) -> MetricRow | None:
    return max(rows, key=key) if rows else None


def argmin(rows: list[MetricRow], key: Callable[[MetricRow], float]) -> MetricRow | None:
    return min(rows, key=key) if rows else None


def representative_cases(rows: list[MetricRow]) -> list[MetricRow]:
    """Return deterministic example rows selected by predefined paper rules."""

    failures = [row for row in rows if failed(row)]
    wins = [row for row in rows if not failed(row)]
    cases: list[MetricRow] = []

    largest_rer = argmax(failures, lambda row: numeric(row, "relative_error_ratio"))
    if largest_rer is not None:
        cases.append(
            selected_case(
                "largest_relative_error_failure",
                "relative_error_ratio",
                numeric(largest_rer, "relative_error_ratio"),
                largest_rer,
            )
        )

    largest_abs_gap = argmax(failures, lambda row: numeric(row, "mae") - numeric(row, "baseline_mae"))
    if largest_abs_gap is not None:
        value = numeric(largest_abs_gap, "mae") - numeric(largest_abs_gap, "baseline_mae")
        cases.append(
            selected_case(
                "largest_absolute_error_gap_failure",
                "mae_minus_baseline_mae",
                value,
                largest_abs_gap,
            )
        )

    smoothing_pool = [row for row in failures if numeric(row, "over_smoothing") > 0.5] or failures
    smoothing = argmax(
        smoothing_pool,
        lambda row: (numeric(row, "flatness_score"), numeric(row, "relative_error_ratio")),
    )
    if smoothing is not None:
        cases.append(
            selected_case(
                "largest_over_smoothing_failure",
                "flatness_score",
                numeric(smoothing, "flatness_score"),
                smoothing,
            )
        )

    excess_pool = [row for row in failures if numeric(row, "excess_variance") > 0.5] or failures
    excess = argmax(
        excess_pool,
        lambda row: (
            max(numeric(row, "forecast_variance_ratio"), numeric(row, "prediction_amplitude_ratio")),
            numeric(row, "relative_error_ratio"),
        ),
    )
    if excess is not None:
        value = max(numeric(excess, "forecast_variance_ratio"), numeric(excess, "prediction_amplitude_ratio"))
        cases.append(selected_case("largest_excess_variance_failure", "max_fvr_par", value, excess))

    coverage = argmin(
        failures,
        lambda row: (numeric(row, "empirical_coverage_90", 1.0), -numeric(row, "relative_error_ratio")),
    )
    if coverage is not None:
        cases.append(
            selected_case(
                "worst_coverage_failure",
                "empirical_coverage_90",
                numeric(coverage, "empirical_coverage_90", 1.0),
                coverage,
            )
        )

    spike = argmin(
        failures,
        lambda row: (numeric(row, "spike_recall", 1.0), -numeric(row, "relative_error_ratio")),
    )
    if spike is not None:
        cases.append(
            selected_case(
                "lowest_spike_recall_failure",
                "spike_recall",
                numeric(spike, "spike_recall", 1.0),
                spike,
            )
        )

    if failures:
        rer_median = median(numeric(row, "relative_error_ratio") for row in failures)
        median_failure = argmin(
            failures,
            lambda row: (
                abs(numeric(row, "relative_error_ratio") - rer_median),
                -numeric(row, "mae"),
            ),
        )
        if median_failure is not None:
            cases.append(
                selected_case(
                    "median_relative_error_failure",
                    "relative_error_ratio_distance_to_failed_median",
                    abs(numeric(median_failure, "relative_error_ratio") - rer_median),
                    median_failure,
                )
            )

    positive_control_pool = [
        row
        for row in wins
        if row.get("regime") in {"seasonal_high_structure", "medium_snr_persistent", "free_flow"}
        or row.get("domain") in {"Transport", "traffic", "Energy"}
    ] or wins
    positive = argmin(
        positive_control_pool,
        lambda row: (numeric(row, "relative_error_ratio", 999.0), -numeric(row, "empirical_coverage_90")),
    )
    if positive is not None:
        cases.append(
            selected_case(
                "strongest_positive_control_win",
                "relative_error_ratio",
                numeric(positive, "relative_error_ratio", 999.0),
                positive,
            )
        )

    return cases


def write_rows(path: Path, rows: list[MetricRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    leading = [
        "selector",
        "selection_metric",
        "selection_value",
        "case_key",
        "dataset",
        "domain",
        "regime",
        "model",
        "baseline",
        "baseline_mode",
        "series_id",
        "origin",
        "window_index",
        "relative_error_ratio",
        "failure_delta_005",
        "flatness_score",
        "forecast_variance_ratio",
        "prediction_amplitude_ratio",
        "empirical_coverage_90",
        "spike_recall",
        "source_metric_path",
        "raw_forecast_path",
    ]
    all_fields = sorted({key for row in rows for key in row})
    fieldnames = leading + [key for key in all_fields if key not in leading]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
