#!/usr/bin/env python
"""Plot raw forecast panels for predefined representative failure cases."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG = ROOT / ".omx" / "matplotlib"
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SELECTORS = [
    "largest_relative_error_failure",
    "largest_absolute_error_gap_failure",
    "largest_over_smoothing_failure",
    "largest_excess_variance_failure",
    "worst_coverage_failure",
    "lowest_spike_recall_failure",
    "median_relative_error_failure",
    "strongest_positive_control_win",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def raw_rows_for_case(case: dict[str, str]) -> list[dict[str, str]]:
    raw_path = ROOT / case["raw_forecast_path"]
    rows = read_rows(raw_path)
    filtered = []
    for row in rows:
        if row.get("dataset") != case.get("dataset"):
            continue
        if row.get("model") != case.get("model"):
            continue
        if row.get("series_id") != case.get("series_id"):
            continue
        if row.get("origin") != case.get("origin"):
            continue
        if row.get("window_index") != case.get("window_index"):
            continue
        if case.get("run_id") and row.get("run_id") != case.get("run_id"):
            continue
        filtered.append(row)
    filtered.sort(key=lambda row: numeric(row, "horizon_index"))
    if not filtered:
        raise ValueError(f"No raw forecast rows matched selector {case['selector']}: {case}")
    return filtered


def title_for(case: dict[str, str]) -> str:
    selector = case["selector"].replace("_", " ")
    dataset = case.get("dataset", "")
    rer = numeric(case, "relative_error_ratio")
    return f"{selector}\n{dataset} | RER={rer:.3g}"


def plot_case(ax, case: dict[str, str], rows: list[dict[str, str]]) -> None:
    x = [int(numeric(row, "horizon_index")) for row in rows]
    actual = [numeric(row, "actual") for row in rows]
    forecast = [numeric(row, "forecast_mean") for row in rows]
    baseline = [numeric(row, "baseline_forecast") for row in rows]
    q10 = [numeric(row, "forecast_q10") for row in rows]
    q90 = [numeric(row, "forecast_q90") for row in rows]

    ax.fill_between(x, q10, q90, color="#9ecae1", alpha=0.28, linewidth=0, label="q10-q90")
    ax.plot(x, actual, color="#111111", linewidth=1.8, label="actual")
    ax.plot(x, forecast, color="#2b6cb0", linewidth=1.5, label="TSFM mean")
    ax.plot(x, baseline, color="#2f855a", linewidth=1.3, linestyle="--", label="baseline")
    ax.set_title(title_for(case), fontsize=8)
    ax.grid(True, color="#d9d9d9", linewidth=0.5, alpha=0.7)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlabel("horizon", fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="results/failure_mining/representative_failure_cases.csv")
    parser.add_argument("--output", default="figures/representative_failure_cases_panel.svg")
    parser.add_argument("--png-output", default="figures/representative_failure_cases_panel.png")
    parser.add_argument("--report", default="figures/representative_failure_cases_panel_report.json")
    parser.add_argument("--selectors", nargs="*", default=DEFAULT_SELECTORS)
    args = parser.parse_args()

    cases = read_rows(ROOT / args.cases)
    by_selector = {case["selector"]: case for case in cases}
    selected = []
    missing_selectors = []
    for selector in args.selectors:
        if selector in by_selector:
            selected.append(by_selector[selector])
        else:
            missing_selectors.append(selector)
    if missing_selectors:
        raise SystemExit(f"Missing selectors in case table: {missing_selectors}")

    ncols = 2
    nrows = (len(selected) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.35 * nrows), squeeze=False)
    plotted = []
    for ax, case in zip(axes.ravel(), selected):
        rows = raw_rows_for_case(case)
        plot_case(ax, case, rows)
        plotted.append(
            {
                "selector": case["selector"],
                "dataset": case.get("dataset", ""),
                "series_id": case.get("series_id", ""),
                "window_index": case.get("window_index", ""),
                "raw_rows": len(rows),
                "raw_forecast_path": case.get("raw_forecast_path", ""),
            }
        )
    for ax in axes.ravel()[len(selected) :]:
        ax.axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg")
    png_output = ROOT / args.png_output
    png_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_output, format="png", dpi=180)
    plt.close(fig)

    report = {
        "status": "ok",
        "figure": str(output.relative_to(ROOT)),
        "png_figure": str(png_output.relative_to(ROOT)),
        "case_table": args.cases,
        "n_cases": len(selected),
        "selectors": [case["selector"] for case in selected],
        "plotted": plotted,
    }
    report_path = ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
