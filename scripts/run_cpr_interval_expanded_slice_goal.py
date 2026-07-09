#!/usr/bin/env python
"""Evaluate the CPR interval-width head on every local raw slice with features.

This is an expansion check for `run_cpr_interval_recalibration_goal.py`. The
point-repair policy is fixed to the most common LTT-selected CPR policy from the
locked 428-window slice; only the interval-width scale is recalibrated under
leave-source and leave-family splits.
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

from low_snr_tsfm.metrics import mae, relative_error_ratio  # noqa: E402


OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "cpr_interval_expanded_slice_report.md"
STATUS_PATH = OUT_DIR / "cpr_interval_expanded_slice_status.json"
WINDOW_OUT = OUT_DIR / "cpr_interval_expanded_slice_windows.csv"
SUMMARY_OUT = OUT_DIR / "cpr_interval_expanded_slice_summary.csv"
SELECTED_OUT = OUT_DIR / "cpr_interval_expanded_slice_selected.csv"
CANDIDATE_OUT = OUT_DIR / "cpr_interval_expanded_slice_candidates.csv"
INVENTORY_OUT = OUT_DIR / "cpr_interval_expanded_slice_inventory.csv"

EXCLUDE_SOURCE_SUBSTRINGS = ("scaling", "smoke")
BASELINE_COLUMNS = ("baseline_forecast", "historical_mean", "bma_mean")


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


def infer_family(slug: str) -> str:
    if slug.startswith("chronos"):
        return "chronos"
    if slug.startswith("moirai"):
        return "moirai"
    if slug.startswith("timesfm"):
        return "timesfm"
    return "unknown"


def infer_role(slug: str) -> str:
    if "finance" in slug:
        return "stress_target"
    if "solar" in slug and "timesfm_2_5_solar_m8" in slug:
        return "weak_positive_control"
    if "solar" in slug or "loop" in slug:
        return "positive_control"
    return "failure_target"


def eligible_slugs() -> list[str]:
    raw_slugs = {path.stem for path in (ROOT / "results" / "raw_forecasts").glob("*.csv")}
    feature_slugs = {
        path.name.replace("_predictor_features.csv", "")
        for path in (ROOT / "results" / "failure_mining").glob("*_predictor_features.csv")
    }
    metric_slugs = {
        path.name.replace("_metrics.csv", "")
        for path in (ROOT / "results" / "window_metrics").glob("*_metrics.csv")
    }
    return sorted(
        slug
        for slug in raw_slugs & feature_slugs & metric_slugs
        if not any(token in slug for token in EXCLUDE_SOURCE_SUBSTRINGS)
    )


def load_feature_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    return {base.feature_key(row): row for row in read_csv(path)}


def load_expanded_windows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for slug in eligible_slugs():
        raw_path = ROOT / "results" / "raw_forecasts" / f"{slug}.csv"
        feature_path = ROOT / "results" / "failure_mining" / f"{slug}_predictor_features.csv"
        raw_groups = base.raw_window_map(raw_path)
        sample_rows = next(iter(raw_groups.values()), [])
        if sample_rows and not any(column in sample_rows[0] for column in BASELINE_COLUMNS):
            inventory.append(
                {
                    "source": slug,
                    "family": infer_family(slug),
                    "role": infer_role(slug),
                    "n_windows": 0,
                    "skipped_missing_features": 0,
                    "status": "skipped_missing_horizon_baseline",
                    "raw_path": str(raw_path.relative_to(ROOT)),
                    "feature_path": str(feature_path.relative_to(ROOT)),
                }
            )
            continue
        feature_rows = load_feature_rows(feature_path)
        source_windows = 0
        skipped = 0
        for key, rows in raw_groups.items():
            dataset, model_name, series_id, origin, window_index = key
            feature = feature_rows.get((dataset, series_id, window_index)) or feature_rows.get(("", series_id, window_index))
            if not feature:
                skipped += 1
                continue
            actual = np.asarray([finite_float(row.get("actual")) for row in rows], dtype=float)
            model = np.asarray([finite_float(row.get("forecast_mean")) for row in rows], dtype=float)
            baseline = base.raw_baseline_values(rows)
            q10 = np.asarray([finite_float(row.get("forecast_q10")) for row in rows], dtype=float)
            q50 = np.asarray(
                [
                    finite_float(row.get("forecast_q50", row.get("forecast_median", row.get("forecast_mean"))))
                    for row in rows
                ],
                dtype=float,
            )
            q90 = np.asarray([finite_float(row.get("forecast_q90")) for row in rows], dtype=float)
            model_mae = mae(actual, model)
            baseline_mae = mae(actual, baseline)
            model_rer = relative_error_ratio(model_mae, baseline_mae)
            windows.append(
                {
                    "family": infer_family(slug),
                    "source": slug,
                    "role": infer_role(slug),
                    "dataset": dataset,
                    "model": model_name,
                    "series_id": series_id,
                    "origin": origin,
                    "window_index": window_index,
                    "feature": feature,
                    "actual": actual,
                    "model_forecast": model,
                    "baseline_forecast": baseline,
                    "q10": q10,
                    "q50": q50,
                    "q90": q90,
                    "model_mae": model_mae,
                    "baseline_mae": baseline_mae,
                    "model_rer": model_rer,
                    "model_failure": int(model_rer > 1.05),
                }
            )
            source_windows += 1
        inventory.append(
            {
                "source": slug,
                "family": infer_family(slug),
                "role": infer_role(slug),
                "n_windows": source_windows,
                "skipped_missing_features": skipped,
                "status": "ok",
                "raw_path": str(raw_path.relative_to(ROOT)),
                "feature_path": str(feature_path.relative_to(ROOT)),
            }
        )
    return windows, inventory


def common_cpr_policy() -> dict[str, object]:
    selected_path = OUT_DIR / "cpr_ltt_selected_policies.csv"
    selected = read_csv(selected_path)
    policy_id, _ = Counter(row["selected_policy_id"] for row in selected).most_common(1)[0]
    row = next(item for item in selected if item["selected_policy_id"] == policy_id)
    return interval.policy_from_selected(row)


def raw_key(window: dict[str, object]) -> tuple[str, str, str, str, str, str]:
    return (
        str(window["family"]),
        str(window["source"]),
        str(window["dataset"]),
        str(window["series_id"]),
        str(window["origin"]),
        str(window["window_index"]),
    )


def raw_map(windows: list[dict[str, object]]) -> dict[tuple[str, str, str, str, str, str], dict[str, object]]:
    return {raw_key(window): window for window in windows}


def apply_point_policy(
    windows: list[dict[str, object]],
    policy: dict[str, object],
    split_protocol: str,
    split_id: str,
) -> list[dict[str, object]]:
    return [
        cpr.apply_policy_to_window(
            window,
            policy,
            "expanded_interval_point_policy",
            split_protocol,
            split_id,
            "expanded_fixed_common_cpr",
        )
        for window in windows
    ]


def split_values(windows: list[dict[str, object]], key: str) -> list[str]:
    return sorted({str(window[key]) for window in windows})


def candidate_rows_for_train(
    train_windows: list[dict[str, object]],
    train_point_rows: list[dict[str, object]],
    candidates: list[dict[str, object]],
    split_protocol: str,
    split_id: str,
    point_policy_id: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        metrics = [
            interval.interval_metrics(window, point_row, candidate)
            for window, point_row in zip(train_windows, train_point_rows, strict=True)
        ]
        aggregate = interval.aggregate_metrics(metrics)
        objective_balanced = (
            ""
            if candidate["kind"] == "shifted_quantile_baseline"
            else interval.candidate_objective(aggregate, interval.BALANCED_LAMBDA)
        )
        objective_coverage = (
            ""
            if candidate["kind"] == "shifted_quantile_baseline"
            else interval.candidate_objective(aggregate, 0.0)
        )
        rows.append(
            {
                "split_protocol": split_protocol,
                "split_id": split_id,
                "point_policy_id": point_policy_id,
                "interval_policy_id": candidate["policy_id"],
                "interval_policy_class": candidate["kind"],
                "interval_scale": candidate.get("scale", ""),
                "selection_objective_balanced": objective_balanced,
                "selection_objective_coverage": objective_coverage,
                "train_n_windows": len(train_windows),
                **aggregate,
            }
        )
    return rows


def pseudo_selected_row(split_protocol: str, split_id: str, policy: dict[str, object]) -> dict[str, str]:
    return {
        "split_protocol": split_protocol,
        "split_id": split_id,
        "selected_policy_id": str(policy["policy_id"]),
    }


def run_split_protocol(
    windows: list[dict[str, object]],
    split_key: str,
    policy: dict[str, object],
    candidates: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    split_protocol = f"leave_{split_key}"
    candidate_by_id = {str(candidate["policy_id"]): candidate for candidate in candidates}
    window_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for holdout in split_values(windows, split_key):
        split_id = f"holdout_{split_key}:{holdout}"
        train = [window for window in windows if str(window[split_key]) != holdout]
        test = [window for window in windows if str(window[split_key]) == holdout]
        train_point_rows = apply_point_policy(train, policy, split_protocol, split_id)
        test_point_rows = apply_point_policy(test, policy, split_protocol, split_id)
        per_candidate_rows = candidate_rows_for_train(
            train,
            train_point_rows,
            candidates,
            split_protocol,
            split_id,
            str(policy["policy_id"]),
        )
        candidate_rows.extend(per_candidate_rows)
        balanced_selected = interval.choose_candidate(per_candidate_rows, objective_key="selection_objective_balanced")
        coverage_selected = interval.choose_candidate(per_candidate_rows, objective_key="selection_objective_coverage")
        strategies = [
            ("expanded_shifted_quantiles", "fixed_shifted_quantiles", candidate_by_id["shifted_quantile_baseline"], 0.0),
            (
                "expanded_width_preserve_s1_00",
                "fixed_preserve_original_width_s1.00",
                candidate_by_id["preserve_original_width_s1"],
                0.0,
            ),
            (
                "expanded_width_preserve_s1_25",
                "fixed_preserve_original_width_s1.25",
                candidate_by_id["preserve_original_width_s1.25"],
                0.0,
            ),
            (
                "expanded_width_calibrated_balanced",
                f"calibration_min_cov_error_plus_{interval.BALANCED_LAMBDA:.2f}_wql_rer",
                candidate_by_id[str(balanced_selected["interval_policy_id"])],
                finite_float(balanced_selected["selection_objective_balanced"]),
            ),
            (
                "expanded_width_calibrated_coverage",
                "calibration_min_cov_error",
                candidate_by_id[str(coverage_selected["interval_policy_id"])],
                finite_float(coverage_selected["selection_objective_coverage"]),
            ),
        ]
        selected_row = pseudo_selected_row(split_protocol, split_id, policy)
        for strategy_id, objective, selected_candidate, selected_objective in strategies:
            selected_rows.append(
                {
                    "strategy_id": strategy_id,
                    "split_protocol": split_protocol,
                    "split_id": split_id,
                    "selection_objective": objective,
                    "point_policy_id": policy["policy_id"],
                    "interval_policy_id": selected_candidate["policy_id"],
                    "interval_policy_class": selected_candidate["kind"],
                    "interval_scale": selected_candidate.get("scale", ""),
                    "train_selected_objective": selected_objective,
                    "train_n_windows": len(train),
                    "test_n_windows": len(test),
                }
            )
            for window, point_row in zip(test, test_point_rows, strict=True):
                window_rows.append(
                    interval.build_eval_row(
                        raw=window,
                        point_row=point_row,
                        selected_row=selected_row,
                        candidate=selected_candidate,
                        strategy_id=strategy_id,
                        selection_objective=objective,
                        train_selected_objective=selected_objective,
                    )
                )
    return window_rows, selected_rows, candidate_rows


def row_lookup(rows: list[dict[str, object]], strategy: str, split: str, group: str) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["strategy_id"] == strategy and row["split_protocol"] == split and row["group"] == group
    )


def report_rows(summary: list[dict[str, object]], split: str = "leave_source") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for strategy in [
        "expanded_shifted_quantiles",
        "expanded_width_preserve_s1_00",
        "expanded_width_preserve_s1_25",
        "expanded_width_calibrated_balanced",
        "expanded_width_calibrated_coverage",
    ]:
        row = row_lookup(summary, strategy, split, "overall")
        rows.append(
            {
                "Strategy": strategy,
                "Coverage": num(row["repair_mean_coverage"], 3),
                "CovErr": num(row["repair_mean_coverage_abs_error"], 3),
                "CovErrRed": pct(row["coverage_abs_error_relative_reduction_vs_shifted"]),
                "WQL-RER": num(row["repair_median_wql_proxy_rer"], 3),
                "dWQL": num(row["median_wql_proxy_rer_delta_vs_shifted"], 3),
                "Width": num(row["repair_median_interval_width"], 3),
            }
        )
    return rows


def role_rows(summary: list[dict[str, object]], split: str = "leave_source") -> list[dict[str, object]]:
    rows = []
    for group in ["role:failure_target", "role:stress_target", "role:positive_control", "role:weak_positive_control"]:
        try:
            row = row_lookup(summary, "expanded_width_calibrated_balanced", split, group)
        except StopIteration:
            continue
        rows.append(
            {
                "Group": group.replace("role:", ""),
                "Coverage": num(row["repair_mean_coverage"], 3),
                "CovErr": num(row["repair_mean_coverage_abs_error"], 3),
                "CovErrRed": pct(row["coverage_abs_error_relative_reduction_vs_shifted"]),
                "WQL-RER": num(row["repair_median_wql_proxy_rer"], 3),
                "dWQL": num(row["median_wql_proxy_rer_delta_vs_shifted"], 3),
            }
        )
    return rows


def write_report(summary: list[dict[str, object]], inventory: list[dict[str, object]], status: dict[str, object]) -> None:
    transfer = row_lookup(summary, "expanded_width_preserve_s1_25", "leave_source", "overall")
    balanced = row_lookup(summary, "expanded_width_calibrated_balanced", "leave_source", "overall")
    shifted = row_lookup(summary, "expanded_shifted_quantiles", "leave_source", "overall")
    DOC_PATH.write_text(
        "\n".join(
            [
                "# CPR Interval Expanded Slice Report",
                "",
                "## Purpose",
                "",
                "This expands the interval-width head from the locked 428-window slice to every local raw forecast source that also has predictor features and window metrics. The point-repair policy is fixed to the most common LTT-selected CPR policy; only the interval width scale is recalibrated on train splits.",
                "",
                "This remains a q10/q50/q90 proxy experiment. It is not a full WQL/CRPS rerun.",
                "",
                "## Headline",
                "",
                f"- Expanded slice: `{status['n_windows']}` windows across `{status['n_sources']}` sources.",
                f"- Shifted-quantile coverage: `{num(shifted['repair_mean_coverage'])}` with mean coverage error `{num(shifted['repair_mean_coverage_abs_error'])}`.",
                f"- Transferred fixed-width head `s=1.25` coverage: `{num(transfer['repair_mean_coverage'])}` with mean coverage error `{num(transfer['repair_mean_coverage_abs_error'])}`.",
                f"- `s=1.25` reduces coverage error by `{pct(transfer['coverage_abs_error_relative_reduction_vs_shifted'])}` and improves median WQL-proxy RER by `{num(-finite_float(transfer['median_wql_proxy_rer_delta_vs_shifted']))}` vs shifted quantiles.",
                f"- Calibration-selected balanced head is the higher-coverage tradeoff: coverage `{num(balanced['repair_mean_coverage'])}`, coverage-error reduction `{pct(balanced['coverage_abs_error_relative_reduction_vs_shifted'])}`, median WQL-proxy RER delta `{num(balanced['median_wql_proxy_rer_delta_vs_shifted'])}`.",
                "",
                "## Overall Table",
                "",
                markdown_table(
                    report_rows(summary),
                    [
                        ("Strategy", "Strategy"),
                        ("Coverage", "Mean coverage"),
                        ("CovErr", "Mean cov. error"),
                        ("CovErrRed", "CovErr red. vs shifted"),
                        ("WQL-RER", "Median WQL-RER"),
                        ("dWQL", "dWQL-RER vs shifted"),
                        ("Width", "Median width"),
                    ],
                ),
                "",
                "## Balanced Head By Role",
                "",
                markdown_table(
                    role_rows(summary),
                    [
                        ("Group", "Group"),
                        ("Coverage", "Mean coverage"),
                        ("CovErr", "Mean cov. error"),
                        ("CovErrRed", "CovErr red. vs shifted"),
                        ("WQL-RER", "Median WQL-RER"),
                        ("dWQL", "dWQL-RER vs shifted"),
                    ],
                ),
                "",
                "## Inventory",
                "",
                markdown_table(
                    [
                        {
                            "Source": row["source"],
                            "Family": row["family"],
                            "Role": row["role"],
                            "Windows": row["n_windows"],
                        }
                        for row in inventory
                    ],
                    [("Source", "Source"), ("Family", "Family"), ("Role", "Role"), ("Windows", "Windows")],
                ),
                "",
                "## Interpretation",
                "",
                "- The interval-collapse repair still holds when expanded beyond the locked 428-window slice.",
                "- The strongest expanded-slice Pareto point is the transferred fixed width scale `s=1.25`: it improves both coverage error and median WQL-proxy RER relative to shifted quantiles.",
                "- Split-calibrated width selection pushes coverage higher, but it exposes a WQL-proxy tradeoff on the expanded slice. The point policy is fixed, so this is evidence for the interval head rather than another point-policy search result.",
                "- Full WQL/CRPS still requires richer exporter artifacts; this report only uses q10/q50/q90.",
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{SELECTED_OUT.relative_to(ROOT)}`",
                f"- `{CANDIDATE_OUT.relative_to(ROOT)}`",
                f"- `{INVENTORY_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows, inventory = load_expanded_windows()
    policy = common_cpr_policy()
    candidates = interval.interval_candidates()
    window_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    for split_key in ["source", "family"]:
        rows, selected, candidate = run_split_protocol(windows, split_key, policy, candidates)
        window_rows.extend(rows)
        selected_rows.extend(selected)
        candidate_rows.extend(candidate)
    summary = interval.build_summary(window_rows)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows": len(windows),
        "n_sources": len({str(window["source"]) for window in windows}),
        "n_families": len({str(window["family"]) for window in windows}),
        "n_window_rows": len(window_rows),
        "n_selected_rows": len(selected_rows),
        "n_candidate_rows": len(candidate_rows),
        "fixed_point_policy_id": policy["policy_id"],
        "balanced_lambda": interval.BALANCED_LAMBDA,
        "excluded_source_substrings": list(EXCLUDE_SOURCE_SUBSTRINGS),
        "windows": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "selected": str(SELECTED_OUT.relative_to(ROOT)),
        "candidates": str(CANDIDATE_OUT.relative_to(ROOT)),
        "inventory": str(INVENTORY_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, window_rows)
    write_csv(SUMMARY_OUT, summary)
    write_csv(SELECTED_OUT, selected_rows)
    write_csv(CANDIDATE_OUT, candidate_rows)
    write_csv(INVENTORY_OUT, inventory)
    STATUS_PATH.write_text(json.dumps(status, indent=2))
    write_report(summary, inventory, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
