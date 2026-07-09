#!/usr/bin/env python
"""Evaluate transferred full-grid CPR/LSSS evidence on Moirai2.

This is deliberately an evidence-line script, not a new tuning pass. It reuses
the common CPR policy and Local-Structure Safety Shield (LSSS) from the Chronos
method line, then applies them to freshly rerun Moirai2 nine-quantile artifacts.
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
sys.path.insert(0, str(ROOT / "scripts"))

import run_chronos_adaptive_fullgrid_cpr_goal as adaptive  # noqa: E402
import run_chronos_fullgrid_cpr_wql_goal as fullgrid  # noqa: E402
import run_factorized_failure_family_goal as base  # noqa: E402

from low_snr_tsfm.metrics import mae, relative_error_ratio  # noqa: E402
from low_snr_tsfm.quantile_artifacts import quantile_matrix_from_rows  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "moirai_fullgrid_evidence_report.md"
STATUS_OUT = OUT_DIR / "moirai_fullgrid_evidence_status.json"
WINDOW_OUT = OUT_DIR / "moirai_fullgrid_evidence_windows.csv"
SUMMARY_OUT = OUT_DIR / "moirai_fullgrid_evidence_summary.csv"
INVENTORY_OUT = OUT_DIR / "moirai_fullgrid_evidence_inventory.csv"

SOURCES = [
    {
        "slug": "moirai2_fullgrid_ctx1680_m64_covid_deaths_short_auto_ets",
        "role": "failure_target",
        "target_id": "covid_deaths_d_short",
        "size": "R-small",
        "params_m": "",
    },
    {
        "slug": "moirai2_fullgrid_ctx1680_solar_m64_solar_short_seasonal_naive",
        "role": "positive_control",
        "target_id": "solar_10t_short",
        "size": "R-small",
        "params_m": "",
    },
]


def finite_float(value: object, default: float = 0.0) -> float:
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


def num(value: object, digits: int = 3) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def pct(value: object) -> str:
    return f"{100.0 * finite_float(value):.1f}%"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def feature_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (row.get("dataset", ""), row.get("series_id", ""), str(row.get("window_index", "")))


def load_moirai_windows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for source in SOURCES:
        slug = source["slug"]
        raw_path = ROOT / "results" / "raw_forecasts" / f"{slug}.csv"
        feature_path = ROOT / "results" / "failure_mining" / f"{slug}_predictor_features.csv"
        status_path = ROOT / "results" / "raw_forecasts" / f"{slug}_status.json"
        if not raw_path.exists() or not feature_path.exists():
            inventory.append(
                {
                    "slug": slug,
                    "role": source["role"],
                    "target_id": source["target_id"],
                    "n_windows": 0,
                    "status": "missing_raw_or_features",
                }
            )
            continue

        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        feature_rows = {feature_key(row): row for row in read_csv(feature_path)}
        raw_groups = base.raw_window_map(raw_path)
        matched = 0
        skipped = 0
        observed_level_counts: set[int] = set()
        for raw_key, rows in raw_groups.items():
            dataset, model_name, series_id, origin, window_index = raw_key
            feature = feature_rows.get((dataset, series_id, window_index)) or feature_rows.get(
                ("", series_id, window_index)
            )
            if feature is None:
                skipped += 1
                continue
            levels, quantile_grid = quantile_matrix_from_rows(rows)
            observed_level_counts.add(len(levels))
            actual = np.asarray([finite_float(row.get("actual")) for row in rows], dtype=float)
            model = np.asarray([finite_float(row.get("forecast_mean")) for row in rows], dtype=float)
            baseline = base.raw_baseline_values(rows)
            q10 = np.asarray([finite_float(row.get("forecast_q10")) for row in rows], dtype=float)
            q50 = np.asarray([finite_float(row.get("forecast_q50")) for row in rows], dtype=float)
            q90 = np.asarray([finite_float(row.get("forecast_q90")) for row in rows], dtype=float)
            model_mae = mae(actual, model)
            baseline_mae = mae(actual, baseline)
            model_rer = relative_error_ratio(model_mae, baseline_mae)
            windows.append(
                {
                    "family": "moirai",
                    "source": slug,
                    "role": source["role"],
                    "target_id": source["target_id"],
                    "size": source["size"],
                    "params_m": source["params_m"],
                    "dataset": dataset,
                    "model": model_name,
                    "series_id": series_id,
                    "origin": origin,
                    "window_index": window_index,
                    "feature": feature,
                    "actual": actual,
                    "model_forecast": model,
                    "baseline_forecast": baseline,
                    "quantile_levels": levels,
                    "quantile_grid": quantile_grid,
                    "q10": q10,
                    "q50": q50,
                    "q90": q90,
                    "model_mae": model_mae,
                    "baseline_mae": baseline_mae,
                    "model_rer": model_rer,
                    "model_failure": int(model_rer > 1.05),
                }
            )
            matched += 1
        inventory.append(
            {
                "slug": slug,
                "role": source["role"],
                "target_id": source["target_id"],
                "n_windows": matched,
                "skipped_missing_features": skipped,
                "observed_quantile_level_counts": ";".join(str(value) for value in sorted(observed_level_counts)),
                "status_quantile_levels": ";".join(str(value) for value in status.get("quantile_levels", [])),
                "raw_path": str(raw_path.relative_to(ROOT)),
                "feature_path": str(feature_path.relative_to(ROOT)),
                "status": "ok",
            }
        )
    return windows, inventory


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary = [adaptive.summarize(rows, "overall", "overall")]
    for role in sorted({str(row["role"]) for row in rows}):
        summary.append(adaptive.summarize([row for row in rows if row["role"] == role], f"role:{role}", "role"))
    for target in sorted({str(row["target_id"]) for row in rows}):
        summary.append(
            adaptive.summarize([row for row in rows if row["target_id"] == target], f"target:{target}", "target")
        )
    for source in sorted({str(row["source"]) for row in rows}):
        summary.append(adaptive.summarize([row for row in rows if row["source"] == source], f"source:{source}", "source"))
    return summary


def rows_for_report(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    for group in ["overall", "role:failure_target", "role:positive_control"]:
        row = next(item for item in summary if item["group"] == group)
        selected.append(
            {
                "Group": group,
                "N": row["n_windows"],
                "Guard": pct(row["guard_rate"]),
                "Model": num(row["model_median_wql_rer"]),
                "Fixed": num(row["fixed_s125_median_wql_rer"]),
                "Adaptive": num(row["adaptive_median_wql_rer"]),
                "dModel": num(row["adaptive_median_wql_delta_vs_model"]),
                "FailRed": pct(row["adaptive_wql_failure_reduction_vs_model"]),
                "Cov": num(row["adaptive_mean_coverage"]),
                "CovErrRed": num(row["adaptive_coverage_error_reduction_vs_fixed_s125"]),
                "Harm": pct(row["adaptive_safety_harm_rate"]),
            }
        )
    return selected


def write_report(summary: list[dict[str, object]], status: dict[str, object]) -> None:
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Moirai2 Full-Grid Evidence",
                "",
                "## Purpose",
                "",
                "This is the evidence-line follow-up to the Chronos method-line result. It reuses the same transferred CPR policy and Local-Structure Safety Shield (LSSS), without Moirai-specific threshold tuning, on Moirai2 nine-quantile reruns.",
                "",
                "## Design",
                "",
                f"- Sources: `{status['n_sources']}` Moirai2 full-grid reruns.",
                f"- Windows: `{status['n_windows']}` total; covid failure target plus solar positive control.",
                f"- Quantile grid sizes observed: `{status['quantile_grid_n_levels']}`.",
                "- Method transfer: common CPR policy plus LSSS; no dataset-name branch and no family-specific retuning.",
                "",
                "## Result",
                "",
                markdown_table(
                    rows_for_report(summary),
                    [
                        ("Group", "Group"),
                        ("N", "N"),
                        ("Guard", "Guard"),
                        ("Model", "Model WQL-RER"),
                        ("Fixed", "Fixed s=1.25"),
                        ("Adaptive", "Adaptive"),
                        ("dModel", "dAdaptive vs model"),
                        ("FailRed", "Fail reduction"),
                        ("Cov", "Coverage"),
                        ("CovErrRed", "CovErr red. vs fixed"),
                        ("Harm", "Safety harm"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- The expanded Moirai slice verifies that full-grid WQL is now available beyond Chronos.",
                "- The result should be used as cross-family evidence for artifact readiness and limited method transfer, not as the final system-scale validation.",
                "- If adaptive WQL or safety tradeoffs diverge from Chronos, report that as a family-specific limitation rather than hiding it.",
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{INVENTORY_OUT.relative_to(ROOT)}`",
                f"- `{STATUS_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    policy = fullgrid.common_cpr_policy()
    windows, inventory = load_moirai_windows()
    if not windows:
        raise SystemExit("No Moirai full-grid windows available")
    rows = adaptive.baseline_rows(windows, policy, strategy_id="moirai_adaptive_fullgrid_cpr")
    summary = build_summary(rows)
    level_counts = sorted({len(window["quantile_levels"]) for window in windows})
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows": len(rows),
        "n_sources": len({row["source"] for row in rows}),
        "n_families": len({row["family"] for row in rows}),
        "n_targets": len({row["target_id"] for row in rows}),
        "quantile_grid_n_levels": level_counts,
        "selected_policy_id": policy["policy_id"],
        "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "inventory": str(INVENTORY_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, rows)
    write_csv(SUMMARY_OUT, summary)
    write_csv(INVENTORY_OUT, inventory)
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")
    write_report(summary, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
