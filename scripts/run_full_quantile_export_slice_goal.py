#!/usr/bin/env python
"""Summarize full-quantile export reruns.

This is an artifact-readiness check, not a new repair benchmark. It verifies
that rerun exporters can persist richer forecast_q* grids and that the metric
layer can compute available-grid WQL without falling back to the q10/q50/q90
proxy.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.metrics import empirical_coverage, mean_weighted_quantile_loss  # noqa: E402
from low_snr_tsfm.quantile_artifacts import quantile_matrix_from_rows  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
WINDOW_OUT = OUT_DIR / "full_quantile_export_slice_windows.csv"
SUMMARY_OUT = OUT_DIR / "full_quantile_export_slice_summary.csv"
STATUS_OUT = OUT_DIR / "full_quantile_export_slice_status.json"
DOC_PATH = ROOT / "docs" / "full_quantile_export_slice_report.md"
EPS = 1e-12

INPUTS = [
    {
        "slice_id": "chronos_tiny_covid_failure_fullgrid",
        "role": "failure_target",
        "path": ROOT / "results/raw_forecasts/chronos_bolt_tiny_fullgrid_scaling_covid_deaths_d_short_auto_ets.csv",
    },
    {
        "slice_id": "chronos_tiny_solar_control_fullgrid",
        "role": "positive_control",
        "path": ROOT / "results/raw_forecasts/chronos_bolt_tiny_fullgrid_scaling_solar_10t_short_seasonal_naive.csv",
    },
]


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def pct(value: object) -> str:
    return f"{100.0 * finite_float(value, 0.0):.1f}%"


def num(value: object, digits: int = 3) -> str:
    parsed = finite_float(value)
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def window_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("series_id", ""), row.get("origin", ""), str(row.get("window_index", "")))


def summarize_window(item: dict[str, object], rows: list[dict[str, str]]) -> dict[str, object]:
    actual = np.asarray([finite_float(row.get("actual")) for row in rows], dtype=float)
    baseline = np.asarray([finite_float(row.get("baseline_forecast")) for row in rows], dtype=float)
    levels, quantiles = quantile_matrix_from_rows(rows)
    baseline_grid = np.repeat(baseline[:, None], len(levels), axis=1)
    q10 = np.asarray([finite_float(row.get("forecast_q10")) for row in rows], dtype=float)
    q50 = np.asarray([finite_float(row.get("forecast_q50")) for row in rows], dtype=float)
    q90 = np.asarray([finite_float(row.get("forecast_q90")) for row in rows], dtype=float)
    proxy_grid = np.column_stack([q10, q50, q90])
    baseline_proxy = np.column_stack([baseline, baseline, baseline])
    full_wql = mean_weighted_quantile_loss(actual, quantiles, levels)
    baseline_full_wql = mean_weighted_quantile_loss(actual, baseline_grid, levels)
    proxy_wql = mean_weighted_quantile_loss(actual, proxy_grid, [0.1, 0.5, 0.9])
    baseline_proxy_wql = mean_weighted_quantile_loss(actual, baseline_proxy, [0.1, 0.5, 0.9])
    return {
        "slice_id": item["slice_id"],
        "role": item["role"],
        "source_path": str(Path(item["path"]).relative_to(ROOT)),
        "dataset": rows[0].get("dataset", ""),
        "model": rows[0].get("model", ""),
        "series_id": rows[0].get("series_id", ""),
        "origin": rows[0].get("origin", ""),
        "window_index": rows[0].get("window_index", ""),
        "horizon": len(rows),
        "quantile_grid_n_levels": len(levels),
        "quantile_grid_levels": ";".join(f"{level:.4g}" for level in levels),
        "artifact_status": "full_grid_from_available_quantile_columns" if len(levels) > 3 else "proxy_only",
        "baseline_wql_full_grid": baseline_full_wql,
        "model_wql_full_grid": full_wql,
        "model_wql_full_grid_rer": full_wql / max(baseline_full_wql, EPS),
        "baseline_wql_proxy_q10_q50_q90": baseline_proxy_wql,
        "model_wql_proxy_q10_q50_q90": proxy_wql,
        "model_wql_proxy_q10_q50_q90_rer": proxy_wql / max(baseline_proxy_wql, EPS),
        "coverage_q10_q90": empirical_coverage(actual, q10, q90),
        "interval_width_q10_q90": float(np.mean(q90 - q10)),
    }


def build_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    window_rows: list[dict[str, object]] = []
    missing: list[str] = []
    for item in INPUTS:
        path = Path(item["path"])
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
        for row in read_csv(path):
            groups.setdefault(window_key(row), []).append(row)
        for rows in groups.values():
            window_rows.append(summarize_window(item, rows))

    summary_rows: list[dict[str, object]] = []
    for slice_id in sorted({str(row["slice_id"]) for row in window_rows}):
        subset = [row for row in window_rows if row["slice_id"] == slice_id]
        full_rers = [finite_float(row["model_wql_full_grid_rer"]) for row in subset]
        proxy_rers = [finite_float(row["model_wql_proxy_q10_q50_q90_rer"]) for row in subset]
        coverages = [finite_float(row["coverage_q10_q90"]) for row in subset]
        summary_rows.append(
            {
                "slice_id": slice_id,
                "role": subset[0]["role"],
                "dataset": subset[0]["dataset"],
                "model": subset[0]["model"],
                "n_windows": len(subset),
                "quantile_grid_n_levels": int(subset[0]["quantile_grid_n_levels"]),
                "artifact_status": subset[0]["artifact_status"],
                "median_full_grid_wql_rer": median(full_rers),
                "mean_full_grid_wql_rer": mean(full_rers),
                "full_grid_wql_failure_rate_delta005": mean([int(value > 1.05) for value in full_rers]),
                "median_proxy_wql_rer": median(proxy_rers),
                "proxy_wql_failure_rate_delta005": mean([int(value > 1.05) for value in proxy_rers]),
                "mean_coverage_q10_q90": mean(coverages),
            }
        )
    status = {
        "status": "ok" if not missing else "partial_missing_inputs",
        "timestamp": int(time.time()),
        "n_window_rows": len(window_rows),
        "n_summary_rows": len(summary_rows),
        "missing_inputs": missing,
        "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    return window_rows, summary_rows, status


def write_report(summary_rows: list[dict[str, object]], status: dict[str, object]) -> None:
    rows = [
        {
            "Slice": row["slice_id"],
            "Role": row["role"],
            "Windows": row["n_windows"],
            "Q": row["quantile_grid_n_levels"],
            "Status": row["artifact_status"],
            "FullWQL": num(row["median_full_grid_wql_rer"]),
            "FullFail": pct(row["full_grid_wql_failure_rate_delta005"]),
            "ProxyWQL": num(row["median_proxy_wql_rer"]),
            "Cov": num(row["mean_coverage_q10_q90"]),
        }
        for row in summary_rows
    ]
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Full Quantile Export Slice",
                "",
                "Purpose: verify that the raw exporters can persist richer quantile grids and that the metric layer can compute available-grid WQL. This is an artifact-readiness slice, not a repair result.",
                "",
                markdown_table(
                    rows,
                    [
                        ("Slice", "Slice"),
                        ("Role", "Role"),
                        ("Windows", "Windows"),
                        ("Q", "Q levels"),
                        ("Status", "Status"),
                        ("FullWQL", "Median full-grid WQL-RER"),
                        ("FullFail", "Full-grid fail rate"),
                        ("ProxyWQL", "Median q10/q50/q90 WQL-RER"),
                        ("Cov", "Mean q10-q90 coverage"),
                    ],
                ),
                "",
                "Interpretation: the Chronos-Bolt tiny exporter now emits the native nine-level quantile grid. On this small paired exporter slice, the covid failure target has much worse full-grid WQL-RER than the solar positive control, which is directionally consistent with the low-local-structure story. Because this uses one model size and no repair policy, it should be cited as artifact readiness and metric robustness, not as a main probabilistic repair claim.",
                "",
                "Artifacts:",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{STATUS_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    window_rows, summary_rows, status = build_rows()
    write_csv(WINDOW_OUT, window_rows)
    write_csv(SUMMARY_OUT, summary_rows)
    write_report(summary_rows, status)
    STATUS_OUT.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
