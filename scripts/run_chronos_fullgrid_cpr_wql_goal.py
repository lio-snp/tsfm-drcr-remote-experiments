#!/usr/bin/env python
"""Evaluate CPR repair on Chronos-Bolt native nine-quantile reruns.

The goal is to move the probabilistic robustness story from q10/q50/q90 proxy
to an available-grid WQL result. The point CPR policy and interval scale are
transferred from earlier locked/expanded CPR experiments rather than tuned here.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_conformal_policy_repair_goal as cpr  # noqa: E402
import run_cpr_interval_recalibration_goal as interval  # noqa: E402
import run_factorized_failure_family_goal as base  # noqa: E402

from low_snr_tsfm.metrics import empirical_coverage, mae, mean_weighted_quantile_loss, relative_error_ratio  # noqa: E402
from low_snr_tsfm.quantile_artifacts import quantile_matrix_from_rows  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "chronos_fullgrid_cpr_wql_report.md"
STATUS_OUT = OUT_DIR / "chronos_fullgrid_cpr_wql_status.json"
WINDOW_OUT = OUT_DIR / "chronos_fullgrid_cpr_wql_windows.csv"
SUMMARY_OUT = OUT_DIR / "chronos_fullgrid_cpr_wql_summary.csv"
POLICY_OUT = OUT_DIR / "chronos_fullgrid_cpr_wql_policy.json"

NOMINAL_COVERAGE = 0.80
INTERVAL_SCALE = 1.25
EPS = 1e-12

SIZES = [
    ("tiny", 9),
    ("mini", 21),
    ("small", 48),
    ("base", 205),
]

TARGETS = [
    {
        "target_id": "covid_deaths_d_short",
        "role": "failure_target",
        "dataset_label": "covid_deaths/D/short",
        "slug_suffix": "covid_deaths_d_short_auto_ets",
    },
    {
        "target_id": "solar_10t_short",
        "role": "positive_control",
        "dataset_label": "solar/10T/short",
        "slug_suffix": "solar_10t_short_seasonal_naive",
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


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def rate(values: list[int]) -> float:
    return float(np.mean(values)) if values else 0.0


def pct(value: object) -> str:
    return f"{100.0 * finite_float(value):.1f}%"


def num(value: object, digits: int = 3) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


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


def fullgrid_slug(size: str, target: dict[str, str]) -> str:
    return f"chronos_bolt_{size}_fullgrid9_scaling_{target['slug_suffix']}"


def common_cpr_policy() -> dict[str, object]:
    selected_path = OUT_DIR / "cpr_ltt_selected_policies.csv"
    selected = read_csv(selected_path)
    policy_id, _ = Counter(row["selected_policy_id"] for row in selected).most_common(1)[0]
    row = next(item for item in selected if item["selected_policy_id"] == policy_id)
    return interval.policy_from_selected(row)


def load_windows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for size, params_m in SIZES:
        for target in TARGETS:
            slug = fullgrid_slug(size, target)
            raw_path = ROOT / "results" / "raw_forecasts" / f"{slug}.csv"
            feature_path = ROOT / "results" / "failure_mining" / f"{slug}_predictor_features.csv"
            status_path = ROOT / "results" / "raw_forecasts" / f"{slug}_status.json"
            if not raw_path.exists() or not feature_path.exists():
                inventory.append(
                    {
                        "slug": slug,
                        "size": size,
                        "target_id": target["target_id"],
                        "n_windows": 0,
                        "status": "missing_raw_or_features",
                    }
                )
                continue
            feature_rows = {feature_key(row): row for row in read_csv(feature_path)}
            raw_groups = base.raw_window_map(raw_path)
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            matched = 0
            skipped = 0
            for raw_key, rows in raw_groups.items():
                dataset, model_name, series_id, origin, window_index = raw_key
                feature = feature_rows.get((dataset, series_id, window_index)) or feature_rows.get(
                    ("", series_id, window_index)
                )
                if feature is None:
                    skipped += 1
                    continue
                actual = np.asarray([finite_float(row.get("actual")) for row in rows], dtype=float)
                model = np.asarray([finite_float(row.get("forecast_mean")) for row in rows], dtype=float)
                baseline = base.raw_baseline_values(rows)
                levels, quantile_grid = quantile_matrix_from_rows(rows)
                q10 = np.asarray([finite_float(row.get("forecast_q10")) for row in rows], dtype=float)
                q50 = np.asarray([finite_float(row.get("forecast_q50")) for row in rows], dtype=float)
                q90 = np.asarray([finite_float(row.get("forecast_q90")) for row in rows], dtype=float)
                model_mae = mae(actual, model)
                baseline_mae = mae(actual, baseline)
                windows.append(
                    {
                        "family": "chronos",
                        "source": slug,
                        "role": target["role"],
                        "target_id": target["target_id"],
                        "size": size,
                        "params_m": params_m,
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
                        "model_rer": relative_error_ratio(model_mae, baseline_mae),
                        "model_failure": int(relative_error_ratio(model_mae, baseline_mae) > 1.05),
                    }
                )
                matched += 1
            inventory.append(
                {
                    "slug": slug,
                    "size": size,
                    "target_id": target["target_id"],
                    "role": target["role"],
                    "n_windows": matched,
                    "skipped_missing_features": skipped,
                    "quantile_levels": ";".join(str(level) for level in status.get("quantile_levels", [])),
                    "raw_path": str(raw_path.relative_to(ROOT)),
                    "feature_path": str(feature_path.relative_to(ROOT)),
                    "status": "ok",
                }
            )
    return windows, inventory


def shifted_quantile_grid(window: dict[str, object], weight: float) -> np.ndarray:
    quantiles = np.asarray(window["quantile_grid"], dtype=float)
    baseline = np.asarray(window["baseline_forecast"], dtype=float)
    return np.sort(quantiles + weight * (baseline[:, None] - quantiles), axis=1)


def interval_head_quantile_grid(window: dict[str, object], weight: float, scale: float = INTERVAL_SCALE) -> np.ndarray:
    quantiles = np.asarray(window["quantile_grid"], dtype=float)
    levels = [float(level) for level in window["quantile_levels"]]
    q50_idx = min(range(len(levels)), key=lambda idx: abs(levels[idx] - 0.5))
    original_center = quantiles[:, q50_idx]
    model = np.asarray(window["model_forecast"], dtype=float)
    baseline = np.asarray(window["baseline_forecast"], dtype=float)
    repaired_center = model + weight * (baseline - model)
    repaired = repaired_center[:, None] + scale * (quantiles - original_center[:, None])
    return np.sort(repaired, axis=1)


def deterministic_baseline_grid(window: dict[str, object]) -> np.ndarray:
    baseline = np.asarray(window["baseline_forecast"], dtype=float)
    levels = [float(level) for level in window["quantile_levels"]]
    return np.repeat(baseline[:, None], len(levels), axis=1)


def quantile_metrics(window: dict[str, object], quantiles: np.ndarray) -> dict[str, float]:
    actual = np.asarray(window["actual"], dtype=float)
    baseline_wql = mean_weighted_quantile_loss(
        actual,
        deterministic_baseline_grid(window),
        window["quantile_levels"],
    )
    wql = mean_weighted_quantile_loss(actual, quantiles, window["quantile_levels"])
    levels = [float(level) for level in window["quantile_levels"]]
    q10_idx = min(range(len(levels)), key=lambda idx: abs(levels[idx] - 0.1))
    q90_idx = min(range(len(levels)), key=lambda idx: abs(levels[idx] - 0.9))
    q10 = quantiles[:, q10_idx]
    q90 = quantiles[:, q90_idx]
    coverage = empirical_coverage(actual, q10, q90)
    return {
        "wql": wql,
        "wql_rer": wql / max(baseline_wql, EPS),
        "coverage": coverage,
        "coverage_abs_error": abs(coverage - NOMINAL_COVERAGE),
        "interval_width_q10_q90": float(np.mean(q90 - q10)),
    }


def point_metrics(window: dict[str, object], weight: float) -> dict[str, float]:
    actual = np.asarray(window["actual"], dtype=float)
    model = np.asarray(window["model_forecast"], dtype=float)
    baseline = np.asarray(window["baseline_forecast"], dtype=float)
    repaired = model + weight * (baseline - model)
    repair_mae = mae(actual, repaired)
    return {
        "repair_mae": repair_mae,
        "repair_rer": relative_error_ratio(repair_mae, finite_float(window["baseline_mae"])),
        "repair_failure": int(relative_error_ratio(repair_mae, finite_float(window["baseline_mae"])) > 1.05),
    }


def build_window_rows(windows: list[dict[str, object]], policy: dict[str, object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for window in windows:
        point_row = cpr.apply_policy_to_window(
            window,
            policy,
            "chronos_fullgrid_cpr",
            "transferred_common_policy",
            "all_windows",
            "common_ltt_policy",
        )
        weight = finite_float(point_row["effective_weight"])
        model_metrics = quantile_metrics(window, np.asarray(window["quantile_grid"], dtype=float))
        shifted_metrics = quantile_metrics(window, shifted_quantile_grid(window, weight))
        interval_metrics = quantile_metrics(window, interval_head_quantile_grid(window, weight))
        point = point_metrics(window, weight)
        row = {
            "family": "chronos",
            "source": window["source"],
            "role": window["role"],
            "target_id": window["target_id"],
            "size": window["size"],
            "params_m": window["params_m"],
            "dataset": window["dataset"],
            "model": window["model"],
            "series_id": window["series_id"],
            "origin": window["origin"],
            "window_index": window["window_index"],
            "horizon": len(window["actual"]),
            "quantile_grid_n_levels": len(window["quantile_levels"]),
            "quantile_grid_levels": ";".join(f"{float(level):.4g}" for level in window["quantile_levels"]),
            "selected_policy_id": policy["policy_id"],
            "interval_scale": INTERVAL_SCALE,
            "gate_active": point_row["gate_active"],
            "shield_active": point_row["shield_active"],
            "conflict_override": point_row["conflict_override"],
            "effective_weight": weight,
            "low_structure_factor_count": point_row["low_structure_factor_count"],
            "model_mae_rer": window["model_rer"],
            "point_repair_mae_rer": point["repair_rer"],
            "model_mae_failure_delta005": int(finite_float(window["model_rer"]) > 1.05),
            "point_repair_mae_failure_delta005": point["repair_failure"],
        }
        for prefix, metrics in [
            ("model", model_metrics),
            ("shifted_cpr", shifted_metrics),
            ("interval_cpr_s125", interval_metrics),
        ]:
            row[f"{prefix}_full_grid_wql"] = metrics["wql"]
            row[f"{prefix}_full_grid_wql_rer"] = metrics["wql_rer"]
            row[f"{prefix}_full_grid_wql_failure_delta005"] = int(metrics["wql_rer"] > 1.05)
            row[f"{prefix}_coverage_q10_q90"] = metrics["coverage"]
            row[f"{prefix}_coverage_abs_error_q10_q90"] = metrics["coverage_abs_error"]
            row[f"{prefix}_interval_width_q10_q90"] = metrics["interval_width_q10_q90"]
        row["shifted_cpr_wql_rer_delta_vs_model"] = (
            row["shifted_cpr_full_grid_wql_rer"] - row["model_full_grid_wql_rer"]
        )
        row["interval_cpr_s125_wql_rer_delta_vs_model"] = (
            row["interval_cpr_s125_full_grid_wql_rer"] - row["model_full_grid_wql_rer"]
        )
        row["interval_cpr_s125_wql_rer_delta_vs_shifted"] = (
            row["interval_cpr_s125_full_grid_wql_rer"] - row["shifted_cpr_full_grid_wql_rer"]
        )
        row["interval_cpr_s125_cov_error_delta_vs_shifted"] = (
            row["interval_cpr_s125_coverage_abs_error_q10_q90"]
            - row["shifted_cpr_coverage_abs_error_q10_q90"]
        )
        output.append(row)
    return output


def summarize_group(rows: list[dict[str, object]], group: str, group_type: str) -> dict[str, object]:
    return {
        "group": group,
        "group_type": group_type,
        "n_windows": len(rows),
        "gate_rate": rate([int(row["gate_active"]) for row in rows]),
        "shield_rate": rate([int(row["shield_active"]) for row in rows]),
        "model_median_wql_rer": median([finite_float(row["model_full_grid_wql_rer"], float("nan")) for row in rows]),
        "shifted_median_wql_rer": median([finite_float(row["shifted_cpr_full_grid_wql_rer"], float("nan")) for row in rows]),
        "interval_median_wql_rer": median(
            [finite_float(row["interval_cpr_s125_full_grid_wql_rer"], float("nan")) for row in rows]
        ),
        "model_wql_failure_rate": rate([int(row["model_full_grid_wql_failure_delta005"]) for row in rows]),
        "shifted_wql_failure_rate": rate([int(row["shifted_cpr_full_grid_wql_failure_delta005"]) for row in rows]),
        "interval_wql_failure_rate": rate(
            [int(row["interval_cpr_s125_full_grid_wql_failure_delta005"]) for row in rows]
        ),
        "interval_wql_failure_reduction_vs_model": rate(
            [int(row["model_full_grid_wql_failure_delta005"]) for row in rows]
        )
        - rate([int(row["interval_cpr_s125_full_grid_wql_failure_delta005"]) for row in rows]),
        "interval_median_wql_rer_delta_vs_model": median(
            [finite_float(row["interval_cpr_s125_wql_rer_delta_vs_model"], float("nan")) for row in rows]
        ),
        "interval_median_wql_rer_delta_vs_shifted": median(
            [finite_float(row["interval_cpr_s125_wql_rer_delta_vs_shifted"], float("nan")) for row in rows]
        ),
        "shifted_mean_coverage": mean([finite_float(row["shifted_cpr_coverage_q10_q90"], float("nan")) for row in rows]),
        "interval_mean_coverage": mean(
            [finite_float(row["interval_cpr_s125_coverage_q10_q90"], float("nan")) for row in rows]
        ),
        "shifted_mean_coverage_abs_error": mean(
            [finite_float(row["shifted_cpr_coverage_abs_error_q10_q90"], float("nan")) for row in rows]
        ),
        "interval_mean_coverage_abs_error": mean(
            [finite_float(row["interval_cpr_s125_coverage_abs_error_q10_q90"], float("nan")) for row in rows]
        ),
        "interval_coverage_error_reduction_vs_shifted": mean(
            [finite_float(row["shifted_cpr_coverage_abs_error_q10_q90"], float("nan")) for row in rows]
        )
        - mean([finite_float(row["interval_cpr_s125_coverage_abs_error_q10_q90"], float("nan")) for row in rows]),
        "model_mae_failure_rate": rate([int(row["model_mae_failure_delta005"]) for row in rows]),
        "point_repair_mae_failure_rate": rate([int(row["point_repair_mae_failure_delta005"]) for row in rows]),
        "point_repair_mae_failure_reduction": rate([int(row["model_mae_failure_delta005"]) for row in rows])
        - rate([int(row["point_repair_mae_failure_delta005"]) for row in rows]),
    }


def build_summary_rows(window_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    summaries.append(summarize_group(window_rows, "overall", "overall"))
    for role in sorted({str(row["role"]) for row in window_rows}):
        subset = [row for row in window_rows if row["role"] == role]
        summaries.append(summarize_group(subset, f"role:{role}", "role"))
    for size in [size for size, _ in SIZES]:
        subset = [row for row in window_rows if row["size"] == size]
        summaries.append(summarize_group(subset, f"size:{size}", "size"))
    for target in [target["target_id"] for target in TARGETS]:
        subset = [row for row in window_rows if row["target_id"] == target]
        summaries.append(summarize_group(subset, f"target:{target}", "target"))
    for size, _ in SIZES:
        for target in [target["target_id"] for target in TARGETS]:
            subset = [row for row in window_rows if row["size"] == size and row["target_id"] == target]
            summaries.append(summarize_group(subset, f"size:{size}|target:{target}", "size_target"))
    return summaries


def rows_for_report(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for size, _ in SIZES:
        for target in [target["target_id"] for target in TARGETS]:
            group = f"size:{size}|target:{target}"
            row = next(item for item in summary_rows if item["group"] == group)
            selected.append(
                {
                    "Size": size,
                    "Target": target,
                    "N": row["n_windows"],
                    "Gate": pct(row["gate_rate"]),
                    "Model": num(row["model_median_wql_rer"]),
                    "Shifted": num(row["shifted_median_wql_rer"]),
                    "Interval": num(row["interval_median_wql_rer"]),
                    "dModel": num(row["interval_median_wql_rer_delta_vs_model"]),
                    "FailRed": pct(row["interval_wql_failure_reduction_vs_model"]),
                    "Cov": num(row["interval_mean_coverage"]),
                    "CovErrRed": num(row["interval_coverage_error_reduction_vs_shifted"]),
                }
            )
    return selected


def compact_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    for group in ["overall", "role:failure_target", "role:positive_control"]:
        row = next(item for item in summary_rows if item["group"] == group)
        selected.append(
            {
                "Group": group,
                "N": row["n_windows"],
                "Model": num(row["model_median_wql_rer"]),
                "Interval": num(row["interval_median_wql_rer"]),
                "FailRed": pct(row["interval_wql_failure_reduction_vs_model"]),
                "MAEFailRed": pct(row["point_repair_mae_failure_reduction"]),
                "Cov": num(row["interval_mean_coverage"]),
                "CovErrRed": num(row["interval_coverage_error_reduction_vs_shifted"]),
            }
        )
    return selected


def write_report(
    summary_rows: list[dict[str, object]],
    inventory: list[dict[str, object]],
    policy: dict[str, object],
    status: dict[str, object],
) -> None:
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Chronos Full-Grid CPR WQL",
                "",
                "## Purpose",
                "",
                "This reruns Chronos-Bolt tiny/mini/small/base on the paired covid failure and solar control slices with the native nine-level quantile grid. It then applies the transferred CPR point policy plus a fixed interval head `s=1.25` and evaluates available-grid WQL-RER.",
                "",
                "This is the first probabilistic robustness result that uses more than q10/q50/q90. It is still Chronos-only, so it should not be phrased as a cross-family TSFM probabilistic claim.",
                "",
                "## Transferred Policy",
                "",
                f"- CPR policy: `{policy['policy_id']}`",
                f"- Interval head: preserve original full-grid quantile deviations around q50, recenter at CPR point forecast, scale deviations by `{INTERVAL_SCALE}`.",
                f"- Windows: `{status['n_windows']}` across `{status['n_sources']}` full-grid reruns.",
                "",
                "## Overall",
                "",
                markdown_table(
                    compact_rows(summary_rows),
                    [
                        ("Group", "Group"),
                        ("N", "N"),
                        ("Model", "Model WQL-RER"),
                        ("Interval", "Interval CPR WQL-RER"),
                        ("FailRed", "WQL fail reduction"),
                        ("MAEFailRed", "MAE fail reduction"),
                        ("Cov", "Interval coverage"),
                        ("CovErrRed", "CovErr red. vs shifted"),
                    ],
                ),
                "",
                "## By Size And Target",
                "",
                markdown_table(
                    rows_for_report(summary_rows),
                    [
                        ("Size", "Size"),
                        ("Target", "Target"),
                        ("N", "N"),
                        ("Gate", "Gate"),
                        ("Model", "Model WQL-RER"),
                        ("Shifted", "Shifted CPR"),
                        ("Interval", "Interval CPR"),
                        ("dModel", "dInterval vs model"),
                        ("FailRed", "Fail reduction"),
                        ("Cov", "Coverage"),
                        ("CovErrRed", "CovErr red."),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- The full-grid WQL metric confirms that covid_deaths is a severe probabilistic failure target for all four Chronos-Bolt sizes.",
                "- The transferred CPR point policy plus interval head gives a real full-grid probabilistic repair signal on the failure target, but the solar control is not a clean universal win under this exact MAE-oriented CPR gate.",
                "- Therefore the defensible claim is: full-grid WQL supports the failure-family diagnosis and shows CPR can repair part of the probabilistic failure on Chronos; it is not yet a universal probabilistic Pareto improvement.",
                "",
                "## Inventory",
                "",
                markdown_table(
                    [
                        {
                            "Slug": row["slug"],
                            "Size": row["size"],
                            "Target": row["target_id"],
                            "Windows": row["n_windows"],
                            "Status": row["status"],
                        }
                        for row in inventory
                    ],
                    [
                        ("Slug", "Slug"),
                        ("Size", "Size"),
                        ("Target", "Target"),
                        ("Windows", "Windows"),
                        ("Status", "Status"),
                    ],
                ),
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{STATUS_OUT.relative_to(ROOT)}`",
                f"- `{POLICY_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    policy = common_cpr_policy()
    windows, inventory = load_windows()
    window_rows = build_window_rows(windows, policy)
    summary_rows = build_summary_rows(window_rows)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows": len(window_rows),
        "n_sources": len({row["source"] for row in window_rows}),
        "n_sizes": len({row["size"] for row in window_rows}),
        "n_targets": len({row["target_id"] for row in window_rows}),
        "quantile_grid_n_levels": sorted({int(row["quantile_grid_n_levels"]) for row in window_rows}),
        "policy_id": policy["policy_id"],
        "interval_scale": INTERVAL_SCALE,
        "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, window_rows)
    write_csv(SUMMARY_OUT, summary_rows)
    POLICY_OUT.write_text(json.dumps(policy, indent=2))
    STATUS_OUT.write_text(json.dumps(status, indent=2))
    write_report(summary_rows, inventory, policy, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
