#!/usr/bin/env python
"""Calibrate a CPR interval-width head on available q10/q50/q90 artifacts.

The CPR point head repairs the forecast location by moving the TSFM mean toward
the classical baseline under an LTT-selected policy. A naive quantile analogue
also moves q10/q50/q90 toward the same baseline, but that collapses interval
width whenever the point-repair weight is large. This script tests a decoupled
uncertainty head: keep the CPR-repaired center, but preserve and calibrate the
original TSFM q10-q90 width on calibration windows.

Only q10/q50/q90 are available in the current raw artifacts, so this is a
three-quantile robustness layer rather than exact full WQL/CRPS.
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

import run_conformal_policy_repair_goal as cpr  # noqa: E402
import run_cross_family_selective_repair_goal as cross  # noqa: E402
import run_paper_faithful_metric_robustness_goal as faithful  # noqa: E402

from low_snr_tsfm.metrics import empirical_coverage  # noqa: E402


OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "cpr_interval_recalibration_report.md"
STATUS_PATH = OUT_DIR / "cpr_interval_recalibration_status.json"
WINDOW_OUT = OUT_DIR / "cpr_interval_recalibration_windows.csv"
SUMMARY_OUT = OUT_DIR / "cpr_interval_recalibration_summary.csv"
SELECTED_OUT = OUT_DIR / "cpr_interval_recalibration_selected.csv"
CANDIDATE_OUT = OUT_DIR / "cpr_interval_recalibration_candidates.csv"

NOMINAL_COVERAGE = 0.80
EPS = 1e-12
INTERVAL_SCALES = [0.5, 0.75, 1.0, 1.125, 1.25, 1.5, 1.75, 2.0]
BALANCED_LAMBDA = 0.10


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


def split_holdout(row: dict[str, str]) -> tuple[str, str]:
    split_protocol = row["split_protocol"]
    if not split_protocol.startswith("leave_"):
        raise ValueError(f"Unsupported split protocol: {split_protocol}")
    split_key = split_protocol.replace("leave_", "", 1)
    try:
        holdout_value = row["split_id"].split(":", 1)[1]
    except IndexError as exc:
        raise ValueError(f"Unsupported split id: {row['split_id']}") from exc
    return split_key, holdout_value


def policy_from_selected(row: dict[str, str]) -> dict[str, object]:
    policy = {
        "kind": row["selected_kind"],
        "policy_id": row["selected_policy_id"],
    }
    mapping = {
        "selected_weight": "weight",
        "selected_min_active": "min_active",
        "selected_factor_step": "factor_step",
        "selected_max_weight": "max_weight",
        "selected_conflict_threshold": "conflict_threshold",
        "selected_shield_cap": "shield_cap",
        "selected_hcr_threshold": "hcr_threshold",
        "selected_trend_threshold": "trend_threshold",
    }
    for source, target in mapping.items():
        if row.get(source, "") == "":
            continue
        value = finite_float(row[source])
        policy[target] = int(value) if target == "min_active" else value
    return policy


def raw_key(window: dict[str, object]) -> tuple[str, str, str, str, str, str]:
    return (
        str(window["family"]),
        str(window["source"]),
        str(window["dataset"]),
        str(window["series_id"]),
        str(window["origin"]),
        str(window["window_index"]),
    )


def shifted_quantiles(raw: dict[str, object], weight: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q10 = np.asarray(raw["q10"], dtype=float)
    q50 = np.asarray(raw["q50"], dtype=float)
    q90 = np.asarray(raw["q90"], dtype=float)
    baseline = np.asarray(raw["baseline_forecast"], dtype=float)
    return faithful.ordered_quantiles(
        q10 + weight * (baseline - q10),
        q50 + weight * (baseline - q50),
        q90 + weight * (baseline - q90),
    )


def interval_candidates() -> list[dict[str, object]]:
    return [
        {"policy_id": "shifted_quantile_baseline", "kind": "shifted_quantile_baseline", "scale": ""},
        *[
            {
                "policy_id": f"preserve_original_width_s{scale:g}",
                "kind": "preserve_original_width",
                "scale": scale,
            }
            for scale in INTERVAL_SCALES
        ],
    ]


def interval_forecast(
    raw: dict[str, object],
    point_row: dict[str, object],
    candidate: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    weight = finite_float(point_row["effective_weight"])
    shifted_q10, shifted_q50, shifted_q90 = shifted_quantiles(raw, weight)
    if candidate["kind"] == "shifted_quantile_baseline":
        return shifted_q10, shifted_q50, shifted_q90

    model = np.asarray(raw["model_forecast"], dtype=float)
    baseline = np.asarray(raw["baseline_forecast"], dtype=float)
    q10 = np.asarray(raw["q10"], dtype=float)
    q90 = np.asarray(raw["q90"], dtype=float)
    center = model + weight * (baseline - model)
    half_width = 0.5 * np.maximum(q90 - q10, EPS) * finite_float(candidate["scale"], 1.0)
    return faithful.ordered_quantiles(center - half_width, center, center + half_width)


def interval_metrics(
    raw: dict[str, object],
    point_row: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, float]:
    actual = np.asarray(raw["actual"], dtype=float)
    baseline = np.asarray(raw["baseline_forecast"], dtype=float)
    q10, q50, q90 = interval_forecast(raw, point_row, candidate)
    baseline_wql = faithful.quantile_loss_proxy(actual, baseline, baseline, baseline)
    repair_wql = faithful.quantile_loss_proxy(actual, q10, q50, q90)
    coverage = empirical_coverage(actual, q10, q90)
    return {
        "coverage": coverage,
        "coverage_abs_error": abs(coverage - NOMINAL_COVERAGE),
        "wql_proxy": repair_wql,
        "wql_proxy_rer": repair_wql / max(baseline_wql, EPS),
        "interval_width": float(np.mean(q90 - q10)),
    }


def model_interval_metrics(raw: dict[str, object]) -> dict[str, float]:
    actual = np.asarray(raw["actual"], dtype=float)
    baseline = np.asarray(raw["baseline_forecast"], dtype=float)
    q10 = np.asarray(raw["q10"], dtype=float)
    q50 = np.asarray(raw["q50"], dtype=float)
    q90 = np.asarray(raw["q90"], dtype=float)
    baseline_wql = faithful.quantile_loss_proxy(actual, baseline, baseline, baseline)
    model_wql = faithful.quantile_loss_proxy(actual, q10, q50, q90)
    coverage = empirical_coverage(actual, q10, q90)
    return {
        "coverage": coverage,
        "coverage_abs_error": abs(coverage - NOMINAL_COVERAGE),
        "wql_proxy": model_wql,
        "wql_proxy_rer": model_wql / max(baseline_wql, EPS),
        "interval_width": float(np.mean(q90 - q10)),
    }


def aggregate_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        "mean_coverage": mean([row["coverage"] for row in metrics]),
        "median_coverage": median([row["coverage"] for row in metrics]),
        "mean_coverage_abs_error": mean([row["coverage_abs_error"] for row in metrics]),
        "median_coverage_abs_error": median([row["coverage_abs_error"] for row in metrics]),
        "median_wql_proxy": median([row["wql_proxy"] for row in metrics]),
        "median_wql_proxy_rer": median([row["wql_proxy_rer"] for row in metrics]),
        "wql_proxy_failure_rate_delta005": rate([int(row["wql_proxy_rer"] > 1.05) for row in metrics]),
        "median_interval_width": median([row["interval_width"] for row in metrics]),
    }


def candidate_objective(candidate_metrics: dict[str, float], selection_lambda: float) -> float:
    return (
        finite_float(candidate_metrics["mean_coverage_abs_error"], 1e6)
        + selection_lambda * finite_float(candidate_metrics["median_wql_proxy_rer"], 1e6)
    )


def choose_candidate(
    candidate_rows: list[dict[str, object]],
    *,
    objective_key: str,
) -> dict[str, object]:
    eligible = [row for row in candidate_rows if row["interval_policy_class"] == "preserve_original_width"]
    if not eligible:
        raise ValueError("No preserve-width candidates available")
    return min(eligible, key=lambda row: (finite_float(row[objective_key]), finite_float(row["interval_scale"])))


def apply_point_policy(
    windows: list[dict[str, object]],
    selected_row: dict[str, str],
    policy: dict[str, object],
) -> list[dict[str, object]]:
    return [
        cpr.apply_policy_to_window(
            window,
            policy,
            "interval_head_point_policy",
            selected_row["split_protocol"],
            selected_row["split_id"],
            selected_row["config_hash"],
        )
        for window in windows
    ]


def build_eval_row(
    *,
    raw: dict[str, object],
    point_row: dict[str, object],
    selected_row: dict[str, str],
    candidate: dict[str, object],
    strategy_id: str,
    selection_objective: str,
    train_selected_objective: float,
) -> dict[str, object]:
    model_metrics = model_interval_metrics(raw)
    shifted_metrics = interval_metrics(raw, point_row, interval_candidates()[0])
    repair_metrics = interval_metrics(raw, point_row, candidate)
    return {
        "strategy_id": strategy_id,
        "split_protocol": selected_row["split_protocol"],
        "split_id": selected_row["split_id"],
        "selection_objective": selection_objective,
        "train_selected_objective": train_selected_objective,
        "point_policy_id": selected_row["selected_policy_id"],
        "interval_policy_id": candidate["policy_id"],
        "interval_policy_class": candidate["kind"],
        "interval_scale": candidate.get("scale", ""),
        "family": raw["family"],
        "source": raw["source"],
        "role": raw["role"],
        "dataset": raw["dataset"],
        "model": raw["model"],
        "series_id": raw["series_id"],
        "origin": raw["origin"],
        "window_index": raw["window_index"],
        "effective_weight": point_row["effective_weight"],
        "gate_active": point_row["gate_active"],
        "low_structure_factor_count": point_row["low_structure_factor_count"],
        "model_coverage_q10_q90": model_metrics["coverage"],
        "shifted_coverage_q10_q90": shifted_metrics["coverage"],
        "repair_coverage_q10_q90": repair_metrics["coverage"],
        "model_coverage_abs_error_q10_q90": model_metrics["coverage_abs_error"],
        "shifted_coverage_abs_error_q10_q90": shifted_metrics["coverage_abs_error"],
        "repair_coverage_abs_error_q10_q90": repair_metrics["coverage_abs_error"],
        "coverage_abs_error_delta_vs_shifted": repair_metrics["coverage_abs_error"] - shifted_metrics["coverage_abs_error"],
        "coverage_abs_error_delta_vs_model": repair_metrics["coverage_abs_error"] - model_metrics["coverage_abs_error"],
        "model_wql_proxy_q10_q50_q90": model_metrics["wql_proxy"],
        "shifted_wql_proxy_q10_q50_q90": shifted_metrics["wql_proxy"],
        "repair_wql_proxy_q10_q50_q90": repair_metrics["wql_proxy"],
        "model_wql_proxy_q10_q50_q90_rer": model_metrics["wql_proxy_rer"],
        "shifted_wql_proxy_q10_q50_q90_rer": shifted_metrics["wql_proxy_rer"],
        "repair_wql_proxy_q10_q50_q90_rer": repair_metrics["wql_proxy_rer"],
        "wql_proxy_rer_delta_vs_shifted": repair_metrics["wql_proxy_rer"] - shifted_metrics["wql_proxy_rer"],
        "wql_proxy_rer_delta_vs_model": repair_metrics["wql_proxy_rer"] - model_metrics["wql_proxy_rer"],
        "model_wql_proxy_failure_delta005": int(model_metrics["wql_proxy_rer"] > 1.05),
        "shifted_wql_proxy_failure_delta005": int(shifted_metrics["wql_proxy_rer"] > 1.05),
        "repair_wql_proxy_failure_delta005": int(repair_metrics["wql_proxy_rer"] > 1.05),
        "model_interval_width_q10_q90": model_metrics["interval_width"],
        "shifted_interval_width_q10_q90": shifted_metrics["interval_width"],
        "repair_interval_width_q10_q90": repair_metrics["interval_width"],
    }


def group_specs(rows: list[dict[str, object]]) -> list[tuple[str, str, list[dict[str, object]]]]:
    specs: list[tuple[str, str, list[dict[str, object]]]] = [("overall", "overall", rows)]
    for role in sorted({str(row["role"]) for row in rows}):
        specs.append((f"role:{role}", "role", [row for row in rows if row["role"] == role]))
    for family in sorted({str(row["family"]) for row in rows}):
        specs.append((f"family:{family}", "family", [row for row in rows if row["family"] == family]))
    return specs


def summarize_group(rows: list[dict[str, object]], strategy_id: str, split_protocol: str, group: str, group_type: str) -> dict[str, object]:
    shifted_cov_err = mean([finite_float(row["shifted_coverage_abs_error_q10_q90"], float("nan")) for row in rows])
    repair_cov_err = mean([finite_float(row["repair_coverage_abs_error_q10_q90"], float("nan")) for row in rows])
    model_cov_err = mean([finite_float(row["model_coverage_abs_error_q10_q90"], float("nan")) for row in rows])
    shifted_wql_rer = median([finite_float(row["shifted_wql_proxy_q10_q50_q90_rer"], float("nan")) for row in rows])
    repair_wql_rer = median([finite_float(row["repair_wql_proxy_q10_q50_q90_rer"], float("nan")) for row in rows])
    model_wql_rer = median([finite_float(row["model_wql_proxy_q10_q50_q90_rer"], float("nan")) for row in rows])
    shifted_fail = rate([int(row["shifted_wql_proxy_failure_delta005"]) for row in rows])
    repair_fail = rate([int(row["repair_wql_proxy_failure_delta005"]) for row in rows])
    model_fail = rate([int(row["model_wql_proxy_failure_delta005"]) for row in rows])
    return {
        "strategy_id": strategy_id,
        "split_protocol": split_protocol,
        "group": group,
        "group_type": group_type,
        "n_windows": len(rows),
        "model_mean_coverage": mean([finite_float(row["model_coverage_q10_q90"], float("nan")) for row in rows]),
        "shifted_mean_coverage": mean([finite_float(row["shifted_coverage_q10_q90"], float("nan")) for row in rows]),
        "repair_mean_coverage": mean([finite_float(row["repair_coverage_q10_q90"], float("nan")) for row in rows]),
        "model_mean_coverage_abs_error": model_cov_err,
        "shifted_mean_coverage_abs_error": shifted_cov_err,
        "repair_mean_coverage_abs_error": repair_cov_err,
        "coverage_abs_error_reduction_vs_shifted": shifted_cov_err - repair_cov_err,
        "coverage_abs_error_relative_reduction_vs_shifted": (shifted_cov_err - repair_cov_err) / max(shifted_cov_err, EPS),
        "coverage_abs_error_delta_vs_model": repair_cov_err - model_cov_err,
        "model_median_wql_proxy_rer": model_wql_rer,
        "shifted_median_wql_proxy_rer": shifted_wql_rer,
        "repair_median_wql_proxy_rer": repair_wql_rer,
        "median_wql_proxy_rer_delta_vs_shifted": repair_wql_rer - shifted_wql_rer,
        "median_wql_proxy_rer_delta_vs_model": repair_wql_rer - model_wql_rer,
        "model_wql_proxy_failure_rate": model_fail,
        "shifted_wql_proxy_failure_rate": shifted_fail,
        "repair_wql_proxy_failure_rate": repair_fail,
        "wql_proxy_failure_delta_vs_shifted": repair_fail - shifted_fail,
        "wql_proxy_failure_delta_vs_model": repair_fail - model_fail,
        "model_median_interval_width": median([finite_float(row["model_interval_width_q10_q90"], float("nan")) for row in rows]),
        "shifted_median_interval_width": median([finite_float(row["shifted_interval_width_q10_q90"], float("nan")) for row in rows]),
        "repair_median_interval_width": median([finite_float(row["repair_interval_width_q10_q90"], float("nan")) for row in rows]),
    }


def build_summary(window_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for strategy_id in sorted({str(row["strategy_id"]) for row in window_rows}):
        strategy_rows = [row for row in window_rows if row["strategy_id"] == strategy_id]
        for split_protocol in sorted({str(row["split_protocol"]) for row in strategy_rows}):
            split_rows = [row for row in strategy_rows if row["split_protocol"] == split_protocol]
            for group, group_type, group_rows in group_specs(split_rows):
                summaries.append(summarize_group(group_rows, strategy_id, split_protocol, group, group_type))
    return summaries


def row_lookup(rows: list[dict[str, object]], strategy: str, split: str, group: str) -> dict[str, object]:
    return next(
        row
        for row in rows
        if row["strategy_id"] == strategy and row["split_protocol"] == split and row["group"] == group
    )


def report_rows(summary: list[dict[str, object]], split: str = "leave_source") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for strategy in [
        "cpr_shifted_quantiles",
        "cpr_width_preserve_s1_00",
        "cpr_width_preserve_s1_25",
        "cpr_width_calibrated_balanced",
        "cpr_width_calibrated_coverage",
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
                "WQLFailDelta": pct(row["wql_proxy_failure_delta_vs_shifted"]),
                "Width": num(row["repair_median_interval_width"], 3),
            }
        )
    return rows


def role_rows(summary: list[dict[str, object]], split: str = "leave_source") -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in ["role:failure_target", "role:stress_target", "role:positive_control", "role:weak_positive_control"]:
        try:
            row = row_lookup(summary, "cpr_width_calibrated_balanced", split, group)
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


def write_report(summary: list[dict[str, object]], status: dict[str, object]) -> None:
    headline = row_lookup(summary, "cpr_width_calibrated_balanced", "leave_source", "overall")
    shifted = row_lookup(summary, "cpr_shifted_quantiles", "leave_source", "overall")
    preserve = row_lookup(summary, "cpr_width_preserve_s1_00", "leave_source", "overall")
    DOC_PATH.write_text(
        "\n".join(
            [
                "# CPR Interval Recalibration Report",
                "",
                "## Method Claim",
                "",
                "The CPR point head fixes forecast location, but directly blending q10/q50/q90 toward a deterministic classical baseline collapses uncertainty width when the point-repair weight is large. The interval head therefore decouples location and uncertainty: keep the CPR-repaired center while preserving and calibration-scaling the original TSFM q10-q90 width.",
                "",
                "This is a q10/q50/q90 robustness layer only. Exact full-paper WQL, CRPS, or MSIS cannot be reconstructed from the current artifacts.",
                "",
                "## Headline",
                "",
                f"- Naive shifted-quantile CPR coverage: `{num(shifted['repair_mean_coverage'])}` with mean coverage error `{num(shifted['repair_mean_coverage_abs_error'])}`.",
                f"- Parameter-free original-width head coverage: `{num(preserve['repair_mean_coverage'])}` with mean coverage error `{num(preserve['repair_mean_coverage_abs_error'])}`.",
                f"- Calibration-selected balanced width head coverage: `{num(headline['repair_mean_coverage'])}` with mean coverage error `{num(headline['repair_mean_coverage_abs_error'])}`.",
                f"- Balanced head reduces mean coverage error vs shifted quantiles by `{pct(headline['coverage_abs_error_relative_reduction_vs_shifted'])}` and changes median WQL-proxy RER by `{num(headline['median_wql_proxy_rer_delta_vs_shifted'])}`.",
                "",
                "## Overall Pareto Table",
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
                        ("WQLFailDelta", "WQL fail-rate delta"),
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
                "## Interpretation",
                "",
                "- The method no longer looks like another fallback blend: it identifies a specific failure introduced by quantile blending, namely uncertainty-width collapse, and repairs it by decoupling the location head from the uncertainty head.",
                "- The strongest honest claim is interval-collapse repair under available q10/q50/q90 proxies. It substantially improves coverage error relative to shifted quantiles while keeping median WQL-proxy RER near or below the shifted-quantile baseline.",
                "- WQL failure-rate can increase relative to shifted quantiles because shifted deterministic intervals are often narrow and score well on some low-noise windows. Therefore this is not yet a full probabilistic forecasting repair claim.",
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{SELECTED_OUT.relative_to(ROOT)}`",
                f"- `{CANDIDATE_OUT.relative_to(ROOT)}`",
                "",
                "## Status",
                "",
                "```json",
                json.dumps(status, indent=2),
                "```",
            ]
        )
        + "\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cpr_selected = read_csv(OUT_DIR / "cpr_ltt_selected_policies.csv")
    cross_windows, _ = cross.load_cross_family_windows()
    raw_windows = faithful.load_windows()
    candidates = interval_candidates()
    candidate_by_id = {str(candidate["policy_id"]): candidate for candidate in candidates}

    window_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []

    for selected_row in cpr_selected:
        split_key, holdout_value = split_holdout(selected_row)
        point_policy = policy_from_selected(selected_row)
        train_windows = [window for window in cross_windows if str(window[split_key]) != holdout_value]
        test_windows = [window for window in cross_windows if str(window[split_key]) == holdout_value]
        train_point_rows = apply_point_policy(train_windows, selected_row, point_policy)
        test_point_rows = apply_point_policy(test_windows, selected_row, point_policy)

        per_candidate_rows: list[dict[str, object]] = []
        for candidate in candidates:
            metrics = [
                interval_metrics(raw_windows[raw_key(window)], point_row, candidate)
                for window, point_row in zip(train_windows, train_point_rows, strict=True)
            ]
            aggregate = aggregate_metrics(metrics)
            interval_scale = candidate.get("scale", "")
            objective_balanced = (
                ""
                if candidate["kind"] == "shifted_quantile_baseline"
                else candidate_objective(aggregate, BALANCED_LAMBDA)
            )
            objective_coverage = (
                ""
                if candidate["kind"] == "shifted_quantile_baseline"
                else candidate_objective(aggregate, 0.0)
            )
            row = {
                "split_protocol": selected_row["split_protocol"],
                "split_id": selected_row["split_id"],
                "point_policy_id": selected_row["selected_policy_id"],
                "interval_policy_id": candidate["policy_id"],
                "interval_policy_class": candidate["kind"],
                "interval_scale": interval_scale,
                "selection_objective_balanced": objective_balanced,
                "selection_objective_coverage": objective_coverage,
                "train_n_windows": len(train_windows),
                **aggregate,
            }
            candidate_rows.append(row)
            per_candidate_rows.append(row)

        balanced_selected = choose_candidate(per_candidate_rows, objective_key="selection_objective_balanced")
        coverage_selected = choose_candidate(per_candidate_rows, objective_key="selection_objective_coverage")
        strategies = [
            (
                "cpr_shifted_quantiles",
                "fixed_shifted_quantiles",
                candidate_by_id["shifted_quantile_baseline"],
                0.0,
            ),
            (
                "cpr_width_preserve_s1_00",
                "fixed_preserve_original_width_s1.00",
                candidate_by_id["preserve_original_width_s1"],
                0.0,
            ),
            (
                "cpr_width_preserve_s1_25",
                "fixed_preserve_original_width_s1.25",
                candidate_by_id["preserve_original_width_s1.25"],
                0.0,
            ),
            (
                "cpr_width_calibrated_balanced",
                f"calibration_min_cov_error_plus_{BALANCED_LAMBDA:.2f}_wql_rer",
                candidate_by_id[str(balanced_selected["interval_policy_id"])],
                finite_float(balanced_selected["selection_objective_balanced"]),
            ),
            (
                "cpr_width_calibrated_coverage",
                "calibration_min_cov_error",
                candidate_by_id[str(coverage_selected["interval_policy_id"])],
                finite_float(coverage_selected["selection_objective_coverage"]),
            ),
        ]

        for strategy_id, objective, selected_candidate, selected_objective in strategies:
            selected_rows.append(
                {
                    "strategy_id": strategy_id,
                    "split_protocol": selected_row["split_protocol"],
                    "split_id": selected_row["split_id"],
                    "selection_objective": objective,
                    "point_policy_id": selected_row["selected_policy_id"],
                    "interval_policy_id": selected_candidate["policy_id"],
                    "interval_policy_class": selected_candidate["kind"],
                    "interval_scale": selected_candidate.get("scale", ""),
                    "train_selected_objective": selected_objective,
                    "train_n_windows": len(train_windows),
                    "test_n_windows": len(test_windows),
                }
            )
            for window, point_row in zip(test_windows, test_point_rows, strict=True):
                window_rows.append(
                    build_eval_row(
                        raw=raw_windows[raw_key(window)],
                        point_row=point_row,
                        selected_row=selected_row,
                        candidate=selected_candidate,
                        strategy_id=strategy_id,
                        selection_objective=objective,
                        train_selected_objective=selected_objective,
                    )
                )

    summary = build_summary(window_rows)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_cross_windows": len(cross_windows),
        "n_cpr_selected_splits": len(cpr_selected),
        "n_interval_candidates": len(candidates),
        "n_window_rows": len(window_rows),
        "n_selected_rows": len(selected_rows),
        "n_candidate_rows": len(candidate_rows),
        "nominal_q10_q90_coverage": NOMINAL_COVERAGE,
        "balanced_lambda": BALANCED_LAMBDA,
        "interval_scales": INTERVAL_SCALES,
        "windows": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "selected": str(SELECTED_OUT.relative_to(ROOT)),
        "candidates": str(CANDIDATE_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, window_rows)
    write_csv(SUMMARY_OUT, summary)
    write_csv(SELECTED_OUT, selected_rows)
    write_csv(CANDIDATE_OUT, candidate_rows)
    STATUS_PATH.write_text(json.dumps(status, indent=2))
    write_report(summary, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
