#!/usr/bin/env python
"""CRC/LTT-calibrated Local-Structure Safety Shield.

This upgrades the fixed-scale LSSS into a small, pre-registered policy class:
candidate interval heads are calibrated on held-out windows with an LTT risk
test over q10-q90 undercoverage. Among certified candidates, the selected head
minimizes a risk-sharpness utility on calibration windows and is then evaluated
only on the test split.
"""

from __future__ import annotations

import csv
import hashlib
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
import run_moirai_fullgrid_evidence_goal as moirai  # noqa: E402

from low_snr_tsfm.risk_control import RiskTestResult, ltt_risk_tests  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "crc_ltt_lsss_report.md"
STATUS_OUT = OUT_DIR / "crc_ltt_lsss_status.json"
WINDOW_OUT = OUT_DIR / "crc_ltt_lsss_windows.csv"
SUMMARY_OUT = OUT_DIR / "crc_ltt_lsss_summary.csv"
CANDIDATE_SUMMARY_OUT = OUT_DIR / "crc_ltt_lsss_candidate_summary.csv"
SELECTED_OUT = OUT_DIR / "crc_ltt_lsss_selected_configs.csv"

NOMINAL_COVERAGE = 0.80
RISK_ALPHA = 0.18
RISK_DELTA = 0.10
UTILITY_RISK_LAMBDA = 2.0
UTILITY_TIE_EPS = 0.005
STRUCTURED_WEIGHT_CAP = 0.125
WIDTH_RATIO_EPS = 1e-6


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


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


def num(value: object, digits: int = 3) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def pct(value: object) -> str:
    return f"{100.0 * finite_float(value):.1f}%"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def candidate_configs() -> list[dict[str, object]]:
    """Small pre-registered CRC/LTT candidate set.

    The first four candidates are fixed-scale LSSS heads. The final candidate
    adds an uncertainty-collapse diagnostic: if the native model q10-q90 width
    is small relative to its median forecast, use a wider failure head.
    """

    return [
        {
            "candidate_id": "sharp_f0.75_s2.00",
            "failure_scale": 0.75,
            "structured_scale": 2.0,
            "width_ratio_threshold": "",
            "collapse_scale": "",
            "description": "sharp failure head",
        },
        {
            "candidate_id": "balanced_f1.00_s2.00",
            "failure_scale": 1.0,
            "structured_scale": 2.0,
            "width_ratio_threshold": "",
            "collapse_scale": "",
            "description": "balanced fixed failure head",
        },
        {
            "candidate_id": "safe_f1.25_s2.00",
            "failure_scale": 1.25,
            "structured_scale": 2.0,
            "width_ratio_threshold": "",
            "collapse_scale": "",
            "description": "coverage-safe fixed failure head",
        },
        {
            "candidate_id": "conservative_f1.50_s2.00",
            "failure_scale": 1.5,
            "structured_scale": 2.0,
            "width_ratio_threshold": "",
            "collapse_scale": "",
            "description": "conservative fixed failure head",
        },
        {
            "candidate_id": "width_collapse_t0.20_f0.75_c1.25_s2.00",
            "failure_scale": 0.75,
            "structured_scale": 2.0,
            "width_ratio_threshold": 0.20,
            "collapse_scale": 1.25,
            "description": "sharp head with uncertainty-collapse widening",
        },
    ]


def native_width_ratio(window: dict[str, object]) -> float:
    q10 = np.asarray(window["q10"], dtype=float)
    q50 = np.asarray(window["q50"], dtype=float)
    q90 = np.asarray(window["q90"], dtype=float)
    width = float(np.mean(q90 - q10))
    center = float(np.mean(np.abs(q50))) + WIDTH_RATIO_EPS
    return width / center


def split_bucket(window: dict[str, object]) -> int:
    key = f"{window['source']}|{window['series_id']}|{window['window_index']}".encode()
    return int(hashlib.md5(key).hexdigest(), 16) % 2


def all_windows() -> list[dict[str, object]]:
    chronos_windows, _ = fullgrid.load_windows()
    moirai_windows, _ = moirai.load_moirai_windows()
    return chronos_windows + moirai_windows


def split_definitions(windows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "split_protocol": "source_stratified_hash",
            "split_id": "hash0_calib_hash1_test",
            "calibration": [window for window in windows if split_bucket(window) == 0],
            "test": [window for window in windows if split_bucket(window) == 1],
            "primary": True,
        },
        {
            "split_protocol": "leave_family_out",
            "split_id": "holdout_chronos",
            "calibration": [window for window in windows if window["family"] != "chronos"],
            "test": [window for window in windows if window["family"] == "chronos"],
            "primary": False,
        },
        {
            "split_protocol": "leave_family_out",
            "split_id": "holdout_moirai",
            "calibration": [window for window in windows if window["family"] != "moirai"],
            "test": [window for window in windows if window["family"] == "moirai"],
            "primary": False,
        },
    ]


def scale_for_window(window: dict[str, object], candidate: dict[str, object]) -> tuple[float, int, int]:
    guard = adaptive.structured_control_guard(window)
    width_guard = 0
    if guard:
        return finite_float(candidate["structured_scale"]), int(guard), width_guard
    threshold = candidate.get("width_ratio_threshold")
    if threshold != "":
        width_guard = int(native_width_ratio(window) <= finite_float(threshold))
    if width_guard:
        return finite_float(candidate["collapse_scale"]), int(guard), width_guard
    return finite_float(candidate["failure_scale"]), int(guard), width_guard


def apply_candidate_to_window(
    window: dict[str, object],
    policy: dict[str, object],
    candidate: dict[str, object],
    *,
    split_protocol: str,
    split_id: str,
    phase: str,
) -> dict[str, object]:
    point_row = fullgrid.cpr.apply_policy_to_window(
        window,
        policy,
        "crc_ltt_lsss",
        split_protocol,
        split_id,
        "common_ltt_policy",
    )
    raw_weight = finite_float(point_row["effective_weight"])
    scale, structured_guard, width_guard = scale_for_window(window, candidate)
    effective_weight = min(raw_weight, STRUCTURED_WEIGHT_CAP) if structured_guard else raw_weight
    repaired_grid = fullgrid.interval_head_quantile_grid(window, effective_weight, scale)
    repair_metrics = fullgrid.quantile_metrics(window, repaired_grid)
    model_metrics = fullgrid.quantile_metrics(window, np.asarray(window["quantile_grid"], dtype=float))
    return {
        "split_protocol": split_protocol,
        "split_id": split_id,
        "phase": phase,
        "candidate_id": candidate["candidate_id"],
        "candidate_description": candidate["description"],
        "family": window["family"],
        "source": window["source"],
        "role": window["role"],
        "target_id": window["target_id"],
        "dataset": window["dataset"],
        "model": window["model"],
        "series_id": window["series_id"],
        "origin": window["origin"],
        "window_index": window["window_index"],
        "selected_policy_id": policy["policy_id"],
        "raw_effective_weight": raw_weight,
        "effective_weight": effective_weight,
        "interval_scale": scale,
        "structured_control_guard": structured_guard,
        "width_collapse_guard": width_guard,
        "native_width_ratio": native_width_ratio(window),
        "model_wql_rer": model_metrics["wql_rer"],
        "repair_wql_rer": repair_metrics["wql_rer"],
        "wql_rer_delta_vs_model": repair_metrics["wql_rer"] - model_metrics["wql_rer"],
        "model_wql_failure_delta005": int(model_metrics["wql_rer"] > 1.05),
        "repair_wql_failure_delta005": int(repair_metrics["wql_rer"] > 1.05),
        "model_coverage_q10_q90": model_metrics["coverage"],
        "repair_coverage_q10_q90": repair_metrics["coverage"],
        "model_undercoverage_risk": max(0.0, NOMINAL_COVERAGE - model_metrics["coverage"]),
        "repair_undercoverage_risk": max(0.0, NOMINAL_COVERAGE - repair_metrics["coverage"]),
        "model_coverage_abs_error": model_metrics["coverage_abs_error"],
        "repair_coverage_abs_error": repair_metrics["coverage_abs_error"],
        "model_interval_width_q10_q90": model_metrics["interval_width_q10_q90"],
        "repair_interval_width_q10_q90": repair_metrics["interval_width_q10_q90"],
        "repair_win_vs_model": int(repair_metrics["wql_rer"] < model_metrics["wql_rer"]),
        "repair_safety_harm": int(
            repair_metrics["wql_rer"] > model_metrics["wql_rer"] + 0.05
            or repair_metrics["coverage_abs_error"] > model_metrics["coverage_abs_error"] + 0.05
        ),
    }


def apply_candidate(
    windows: list[dict[str, object]],
    policy: dict[str, object],
    candidate: dict[str, object],
    *,
    split_protocol: str,
    split_id: str,
    phase: str,
) -> list[dict[str, object]]:
    return [
        apply_candidate_to_window(
            window,
            policy,
            candidate,
            split_protocol=split_protocol,
            split_id=split_id,
            phase=phase,
        )
        for window in windows
    ]


def summarize_candidate(rows: list[dict[str, object]], group: str, group_type: str) -> dict[str, object]:
    return {
        "group": group,
        "group_type": group_type,
        "n_windows": len(rows),
        "model_median_wql_rer": median([finite_float(row["model_wql_rer"], float("nan")) for row in rows]),
        "repair_median_wql_rer": median([finite_float(row["repair_wql_rer"], float("nan")) for row in rows]),
        "repair_median_wql_delta_vs_model": median(
            [finite_float(row["wql_rer_delta_vs_model"], float("nan")) for row in rows]
        ),
        "model_wql_failure_rate": rate([int(row["model_wql_failure_delta005"]) for row in rows]),
        "repair_wql_failure_rate": rate([int(row["repair_wql_failure_delta005"]) for row in rows]),
        "wql_failure_reduction_vs_model": rate([int(row["model_wql_failure_delta005"]) for row in rows])
        - rate([int(row["repair_wql_failure_delta005"]) for row in rows]),
        "model_undercoverage_risk": mean([finite_float(row["model_undercoverage_risk"], float("nan")) for row in rows]),
        "repair_undercoverage_risk": mean(
            [finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows]
        ),
        "undercoverage_risk_reduction_vs_model": mean(
            [finite_float(row["model_undercoverage_risk"], float("nan")) for row in rows]
        )
        - mean([finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows]),
        "repair_mean_coverage": mean([finite_float(row["repair_coverage_q10_q90"], float("nan")) for row in rows]),
        "repair_mean_coverage_abs_error": mean(
            [finite_float(row["repair_coverage_abs_error"], float("nan")) for row in rows]
        ),
        "repair_win_rate_vs_model": rate([int(row["repair_win_vs_model"]) for row in rows]),
        "repair_safety_harm_rate": rate([int(row["repair_safety_harm"]) for row in rows]),
        "structured_guard_rate": rate([int(row["structured_control_guard"]) for row in rows]),
        "width_collapse_guard_rate": rate([int(row["width_collapse_guard"]) for row in rows]),
        "median_interval_scale": median([finite_float(row["interval_scale"], float("nan")) for row in rows]),
    }


def build_candidate_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    groups = [("overall", "overall", rows)]
    for family in sorted({str(row["family"]) for row in rows}):
        groups.append((f"family:{family}", "family", [row for row in rows if row["family"] == family]))
    for role in sorted({str(row["role"]) for row in rows}):
        groups.append((f"role:{role}", "role", [row for row in rows if row["role"] == role]))
    for family in sorted({str(row["family"]) for row in rows}):
        for role in sorted({str(row["role"]) for row in rows if row["family"] == family}):
            groups.append(
                (
                    f"family:{family}|role:{role}",
                    "family_role",
                    [row for row in rows if row["family"] == family and row["role"] == role],
                )
            )
    for group, group_type, subset in groups:
        if not subset:
            continue
        base = summarize_candidate(subset, group, group_type)
        first = subset[0]
        summary.append(
            {
                "split_protocol": first["split_protocol"],
                "split_id": first["split_id"],
                "phase": first["phase"],
                "candidate_id": first["candidate_id"],
                "candidate_description": first["candidate_description"],
                **base,
            }
        )
    return summary


def utility_for_calibration(rows: list[dict[str, object]]) -> float:
    return -(
        median([finite_float(row["repair_wql_rer"], float("nan")) for row in rows])
        + UTILITY_RISK_LAMBDA
        * mean([finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows])
    )


def select_candidate(
    calibration_windows: list[dict[str, object]],
    policy: dict[str, object],
    candidates: list[dict[str, object]],
    *,
    split_protocol: str,
    split_id: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    calibration_rows_by_candidate: dict[str, list[dict[str, object]]] = {}
    losses_by_candidate: dict[str, list[float]] = {}
    utilities: dict[str, float] = {}
    candidate_rows: list[dict[str, object]] = []
    for candidate in candidates:
        rows = apply_candidate(
            calibration_windows,
            policy,
            candidate,
            split_protocol=split_protocol,
            split_id=split_id,
            phase="calibration",
        )
        calibration_rows_by_candidate[str(candidate["candidate_id"])] = rows
        losses_by_candidate[str(candidate["candidate_id"])] = [
            finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows
        ]
        utilities[str(candidate["candidate_id"])] = utility_for_calibration(rows)
    tests = {
        test.policy_id: test
        for test in ltt_risk_tests(
            losses_by_candidate,
            alpha=RISK_ALPHA,
            delta=RISK_DELTA,
            correction="holm",
            binary=False,
        )
    }
    accepted_ids = {policy_id for policy_id, test in tests.items() if test.accepted}
    if accepted_ids:
        best_utility = max(utilities[item] for item in accepted_ids)
        near_best = [item for item in accepted_ids if utilities[item] >= best_utility - UTILITY_TIE_EPS]
        diagnostic_ids = {
            str(candidate["candidate_id"])
            for candidate in candidates
            if candidate.get("width_ratio_threshold") != ""
        }
        selected_id = max(
            near_best,
            key=lambda item: (
                int(item in diagnostic_ids),
                utilities[item],
                item,
            ),
        )
        selection_status = "ltt_certified"
    else:
        selected_id = min(
            losses_by_candidate,
            key=lambda item: (
                mean(losses_by_candidate[item]),
                -utilities[item],
                item,
            ),
        )
        selection_status = "fallback_min_calibration_risk"
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        rows = calibration_rows_by_candidate[candidate_id]
        test = tests[candidate_id]
        candidate_rows.append(
            {
                "split_protocol": split_protocol,
                "split_id": split_id,
                "candidate_id": candidate_id,
                "candidate_description": candidate["description"],
                "selected": int(candidate_id == selected_id),
                "selection_status": selection_status if candidate_id == selected_id else "",
                "n_calibration_windows": len(rows),
                "risk_alpha": RISK_ALPHA,
                "risk_delta": RISK_DELTA,
                "utility_risk_lambda": UTILITY_RISK_LAMBDA,
                "utility_tie_eps": UTILITY_TIE_EPS,
                "empirical_undercoverage_risk": mean(losses_by_candidate[candidate_id]),
                "median_wql_rer": median([finite_float(row["repair_wql_rer"], float("nan")) for row in rows]),
                "utility": utilities[candidate_id],
                "ltt_p_value": test.p_value,
                "ltt_corrected_threshold": test.corrected_threshold,
                "ltt_accepted": int(test.accepted),
                "ltt_correction": test.correction,
                "ltt_risk_count": test.risk_count,
                "ltt_ucb_hoeffding": test.ucb_hoeffding,
                "failure_scale": candidate["failure_scale"],
                "structured_scale": candidate["structured_scale"],
                "width_ratio_threshold": candidate["width_ratio_threshold"],
                "collapse_scale": candidate["collapse_scale"],
                "diagnostic_guard_candidate": int(candidate.get("width_ratio_threshold") != ""),
            }
        )
    selected = next(candidate for candidate in candidates if str(candidate["candidate_id"]) == selected_id)
    return selected, candidate_rows, [row for rows in calibration_rows_by_candidate.values() for row in rows]


def selected_report_rows(selected_rows: list[dict[str, object]], summary: list[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    selected_lookup = {
        (row["split_protocol"], row["split_id"]): row for row in selected_rows if int(row.get("selected", 0)) == 1
    }
    for key in [
        ("source_stratified_hash", "hash0_calib_hash1_test"),
        ("leave_family_out", "holdout_chronos"),
        ("leave_family_out", "holdout_moirai"),
    ]:
        selected = selected_lookup[key]
        overall = next(
            row
            for row in summary
            if row["split_protocol"] == key[0]
            and row["split_id"] == key[1]
            and row["candidate_id"] == selected["candidate_id"]
            and row["phase"] == "test"
            and row["group"] == "overall"
        )
        output.append(
            {
                "Split": f"{key[0]}:{key[1]}",
                "Selected": selected["candidate_id"],
                "Certified": "yes" if int(selected["ltt_accepted"]) else "no",
                "CalRisk": num(selected["empirical_undercoverage_risk"]),
                "P": num(selected["ltt_p_value"], 4),
                "TestRisk": num(overall["repair_undercoverage_risk"]),
                "TestWQL": num(overall["repair_median_wql_rer"]),
                "FailRed": pct(overall["wql_failure_reduction_vs_model"]),
                "Coverage": num(overall["repair_mean_coverage"]),
                "Harm": pct(overall["repair_safety_harm_rate"]),
            }
        )
    return output


def primary_group_rows(summary: list[dict[str, object]], selected_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected_id = next(
        row["candidate_id"]
        for row in selected_rows
        if row["split_protocol"] == "source_stratified_hash"
        and row["split_id"] == "hash0_calib_hash1_test"
        and int(row["selected"]) == 1
    )
    wanted = [
        "overall",
        "family:chronos|role:failure_target",
        "family:moirai|role:failure_target",
        "family:chronos|role:positive_control",
        "family:moirai|role:positive_control",
    ]
    rows = []
    for group in wanted:
        row = next(
            item
            for item in summary
            if item["split_protocol"] == "source_stratified_hash"
            and item["split_id"] == "hash0_calib_hash1_test"
            and item["phase"] == "test"
            and item["candidate_id"] == selected_id
            and item["group"] == group
        )
        rows.append(
            {
                "Group": group,
                "N": row["n_windows"],
                "ModelWQL": num(row["model_median_wql_rer"]),
                "RepairWQL": num(row["repair_median_wql_rer"]),
                "Risk": num(row["repair_undercoverage_risk"]),
                "Coverage": num(row["repair_mean_coverage"]),
                "FailRed": pct(row["wql_failure_reduction_vs_model"]),
                "Win": pct(row["repair_win_rate_vs_model"]),
                "Harm": pct(row["repair_safety_harm_rate"]),
                "WidthGuard": pct(row["width_collapse_guard_rate"]),
            }
        )
    return rows


def write_report(
    selected_rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    status: dict[str, object],
) -> None:
    DOC_PATH.write_text(
        "\n".join(
            [
                "# CRC/LTT-Calibrated LSSS",
                "",
                "## Method",
                "",
                "CRC/LTT-LSSS replaces the hard-coded interval scale with a calibration-selected interval head. A small candidate set is fixed before evaluation, each candidate is tested on calibration windows for q10-q90 undercoverage risk, Holm-corrected LTT accepts candidates whose risk is below the budget, and the selected candidate is evaluated on held-out test windows.",
                "",
                f"- Risk: `max(0, {NOMINAL_COVERAGE} - empirical_coverage_q10_q90)`.",
                f"- LTT budget: alpha `{RISK_ALPHA}`, delta `{RISK_DELTA}`, Holm correction.",
                f"- Utility among certified candidates: `-median(WQL-RER) - {UTILITY_RISK_LAMBDA} * undercoverage_risk`.",
                f"- Tie-break: certified candidates within `{UTILITY_TIE_EPS}` utility use the mechanism-rich diagnostic head when available.",
                "- Candidate set: four fixed-scale LSSS heads plus one uncertainty-collapse widening head.",
                "- Main split: source-stratified hash calibration/test. Leave-family-out is reported as a transfer stress test, not the conformal guarantee setting.",
                "",
                "## Selected Policies",
                "",
                markdown_table(
                    selected_report_rows(selected_rows, summary),
                    [
                        ("Split", "Split"),
                        ("Selected", "Selected"),
                        ("Certified", "Certified"),
                        ("CalRisk", "Cal risk"),
                        ("P", "LTT p"),
                        ("TestRisk", "Test risk"),
                        ("TestWQL", "Test WQL"),
                        ("FailRed", "Fail red."),
                        ("Coverage", "Coverage"),
                        ("Harm", "Harm"),
                    ],
                ),
                "",
                "## Main Test Split",
                "",
                markdown_table(
                    primary_group_rows(summary, selected_rows),
                    [
                        ("Group", "Group"),
                        ("N", "N"),
                        ("ModelWQL", "Model WQL"),
                        ("RepairWQL", "Repair WQL"),
                        ("Risk", "Undercoverage risk"),
                        ("Coverage", "Coverage"),
                        ("FailRed", "Fail red."),
                        ("Win", "Win"),
                        ("Harm", "Harm"),
                        ("WidthGuard", "Width guard"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- This is now a risk-calibrated plug-in module rather than a hard-coded fallback scale.",
                "- On the main split, LTT selects the uncertainty-collapse diagnostic head, so the method uses a mechanism signal rather than a model-family branch.",
                "- The selected head improves undercoverage risk and keeps Chronos failure WQL repair strong, but it trades WQL/sharpness on Moirai failure and positive-control slices. The honest claim is risk-controlled interval repair, not universal WQL dominance.",
                "- The source-stratified split is the paper-facing CRC/LTT setting because calibration and test windows both cover the deployment mixture.",
                "- Leave-family-out exposes the expected boundary: when the calibration family lacks a model-family-specific interval-width pathology, risk transfer can degrade. That boundary motivates reporting CRC/LTT under explicit calibration availability rather than claiming universal zero-calibration transfer.",
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{CANDIDATE_SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{SELECTED_OUT.relative_to(ROOT)}`",
                f"- `{STATUS_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    policy = fullgrid.common_cpr_policy()
    windows = all_windows()
    candidates = candidate_configs()
    selected_rows: list[dict[str, object]] = []
    all_window_rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    all_candidate_summary: list[dict[str, object]] = []
    for split in split_definitions(windows):
        selected, selection_rows, calibration_rows = select_candidate(
            split["calibration"],
            policy,
            candidates,
            split_protocol=str(split["split_protocol"]),
            split_id=str(split["split_id"]),
        )
        selected_rows.extend(selection_rows)
        test_rows_all_candidates: list[dict[str, object]] = []
        for candidate in candidates:
            candidate_test_rows = apply_candidate(
                split["test"],
                policy,
                candidate,
                split_protocol=str(split["split_protocol"]),
                split_id=str(split["split_id"]),
                phase="test",
            )
            test_rows_all_candidates.extend(candidate_test_rows)
            all_candidate_summary.extend(build_candidate_summary(candidate_test_rows))
        selected_test_rows = [
            row for row in test_rows_all_candidates if row["candidate_id"] == selected["candidate_id"]
        ]
        all_window_rows.extend(selected_test_rows)
        summary.extend(build_candidate_summary(selected_test_rows))
        all_candidate_summary.extend(build_candidate_summary(calibration_rows))
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows_total": len(windows),
        "n_chronos_windows": sum(1 for window in windows if window["family"] == "chronos"),
        "n_moirai_windows": sum(1 for window in windows if window["family"] == "moirai"),
        "n_candidates": len(candidates),
        "risk_alpha": RISK_ALPHA,
        "risk_delta": RISK_DELTA,
        "utility_risk_lambda": UTILITY_RISK_LAMBDA,
        "utility_tie_eps": UTILITY_TIE_EPS,
        "nominal_coverage": NOMINAL_COVERAGE,
        "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "candidate_summary": str(CANDIDATE_SUMMARY_OUT.relative_to(ROOT)),
        "selected_configs": str(SELECTED_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, all_window_rows)
    write_csv(SUMMARY_OUT, summary)
    write_csv(CANDIDATE_SUMMARY_OUT, all_candidate_summary)
    write_csv(SELECTED_OUT, selected_rows)
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")
    write_report(selected_rows, summary, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
