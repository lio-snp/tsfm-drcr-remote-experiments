#!/usr/bin/env python
"""Select predefined representative forecast failure cases for paper figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from low_snr_tsfm.failure_selection import read_metric_rows, representative_cases, write_rows  # noqa: E402


DEFAULT_METRICS = [
    Path("results/window_metrics/chronos_bolt_small_bizitobs_application_short_auto_arima_metrics.csv"),
    Path("results/window_metrics/chronos_bolt_small_covid_deaths_short_auto_ets_metrics.csv"),
    Path("results/window_metrics/chronos_bolt_small_solar_short_seasonal_naive_metrics.csv"),
    Path("results/window_metrics/chronos_bolt_small_loop_seattle_short_seasonal_naive_metrics.csv"),
    Path("results/window_metrics/chronos_bolt_small_finance_fred_stress_metrics.csv"),
    Path("results/traffic/chronos_bolt_small_metr_la_traffic_bma_window_metrics.csv"),
    Path("results/traffic/chronos_bolt_small_pems_bay_traffic_bma_window_metrics.csv"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=[str(path) for path in DEFAULT_METRICS],
        help="Metric CSV paths to scan. Defaults to the current locked raw-rerun story inputs.",
    )
    parser.add_argument(
        "--output",
        default="results/failure_mining/representative_failure_cases.csv",
        help="Output CSV for predefined representative cases.",
    )
    parser.add_argument(
        "--report",
        default="results/failure_mining/representative_failure_cases_report.json",
        help="Output JSON report.",
    )
    args = parser.parse_args()

    metric_paths = [Path(path) for path in args.metrics]
    missing = [str(path) for path in metric_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing metric inputs: {missing}")

    rows = read_metric_rows(metric_paths)
    cases = representative_cases(rows)
    if len(cases) < 6:
        raise SystemExit(f"Expected at least 6 representative cases, found {len(cases)}")

    output = Path(args.output)
    report_path = Path(args.report)
    write_rows(output, cases)

    report = {
        "status": "ok",
        "n_metric_inputs": len(metric_paths),
        "n_metric_rows": len(rows),
        "n_failure_rows": sum(int(float(row.get("failure_delta_005", "0"))) for row in rows),
        "n_selected_cases": len(cases),
        "selectors": [row["selector"] for row in cases],
        "output": str(output),
        "metric_inputs": [str(path) for path in metric_paths],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
