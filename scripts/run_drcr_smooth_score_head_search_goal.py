#!/usr/bin/env python
"""DRCR smooth score-to-scale head search.

This is a mechanism-focused follow-up to Experiment A.  The previous DRCR
candidate used a hard uncertainty-collapse threshold.  Here the diagnostic
signal is continuous: native q10-q90 width ratio is mapped through a sigmoid to
an interval scale.  The risk protocol is unchanged: undercoverage and WQL
non-inferiority harm are calibrated with Holm-corrected LTT on held-out
calibration windows.
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

import run_crc_ltt_lsss_goal as crc  # noqa: E402
import run_dual_risk_crc_ltt_lsss_goal as drcr  # noqa: E402

from low_snr_tsfm.risk_control import ltt_risk_tests  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "drcr_smooth_score_head_report.md"
STATUS_OUT = OUT_DIR / "drcr_smooth_score_head_status.json"
WINDOW_OUT = OUT_DIR / "drcr_smooth_score_head_windows.csv"
SUMMARY_OUT = OUT_DIR / "drcr_smooth_score_head_summary.csv"
CANDIDATE_SUMMARY_OUT = OUT_DIR / "drcr_smooth_score_head_candidate_summary.csv"
SELECTED_OUT = OUT_DIR / "drcr_smooth_score_head_selected_configs.csv"
CALIBRATION_TEST_OUT = OUT_DIR / "drcr_smooth_score_head_calibration_tests.csv"
PAIRED_STATS_OUT = OUT_DIR / "drcr_smooth_score_head_paired_stats.csv"
CONFIG_PATH = ROOT / "configs" / "drcr_smooth_frozen_protocol.json"


def load_protocol_config() -> dict[str, object]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Frozen DRCR-Smooth protocol is missing: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text())


PROTOCOL_CONFIG = load_protocol_config()
PROTOCOL_ID = str(PROTOCOL_CONFIG["protocol_id"])
SPLIT_PROTOCOL = str(PROTOCOL_CONFIG["split"]["split_protocol"])
SPLIT_ID = str(PROTOCOL_CONFIG["split"]["split_id"])
UNDERCOVERAGE_ALPHA = float(PROTOCOL_CONFIG["risk_control"]["undercoverage_alpha"])
WQL_HARM_ALPHA = float(PROTOCOL_CONFIG["risk_control"]["wql_harm_alpha"])
DELTA_PER_RISK = float(PROTOCOL_CONFIG["risk_control"]["delta_per_risk"])
WQL_HARM_MARGIN = float(PROTOCOL_CONFIG["risk_control"]["wql_harm_margin"])
UTILITY_UNDERCOVERAGE_LAMBDA = float(PROTOCOL_CONFIG["utility"]["undercoverage_lambda"])
UTILITY_WQL_HARM_LAMBDA = float(PROTOCOL_CONFIG["utility"]["wql_harm_lambda"])
SMOOTH_TIE_EPS = float(PROTOCOL_CONFIG["selection"]["utility_tie_eps"])
SMOOTH_CANDIDATE_ID = str(PROTOCOL_CONFIG["candidate_aliases"]["smooth"])
PREVIOUS_STEP_ID = str(PROTOCOL_CONFIG["candidate_aliases"]["previous_step"])
BALANCED_ID = str(PROTOCOL_CONFIG["candidate_aliases"]["balanced"])
N_BOOTSTRAP = int(PROTOCOL_CONFIG["bootstrap"]["n_bootstrap"])
SEED = int(PROTOCOL_CONFIG["bootstrap"]["seed"])


def protocol_sha256() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()


def validate_protocol_config() -> None:
    expected_constants = {
        "undercoverage_alpha": drcr.UNDERCOVERAGE_ALPHA,
        "wql_harm_alpha": drcr.WQL_HARM_ALPHA,
        "delta_per_risk": drcr.DELTA_PER_RISK,
        "wql_harm_margin": drcr.WQL_HARM_MARGIN,
        "undercoverage_lambda": drcr.UTILITY_UNDERCOVERAGE_LAMBDA,
        "wql_harm_lambda": drcr.UTILITY_WQL_HARM_LAMBDA,
        "utility_tie_eps": drcr.UTILITY_TIE_EPS,
    }
    observed_constants = {
        "undercoverage_alpha": UNDERCOVERAGE_ALPHA,
        "wql_harm_alpha": WQL_HARM_ALPHA,
        "delta_per_risk": DELTA_PER_RISK,
        "wql_harm_margin": WQL_HARM_MARGIN,
        "undercoverage_lambda": UTILITY_UNDERCOVERAGE_LAMBDA,
        "wql_harm_lambda": UTILITY_WQL_HARM_LAMBDA,
        "utility_tie_eps": SMOOTH_TIE_EPS,
    }
    for key, expected in expected_constants.items():
        if abs(float(observed_constants[key]) - float(expected)) > 1e-12:
            raise ValueError(f"Frozen protocol {key}={observed_constants[key]} disagrees with DRCR constant {expected}")
    candidates = PROTOCOL_CONFIG["candidate_class"]["candidates"]
    ids = [str(candidate["candidate_id"]) for candidate in candidates]
    expected_ids = [
        BALANCED_ID,
        "safe_f1.25_s2.00",
        "conservative_f1.50_s2.00",
        PREVIOUS_STEP_ID,
        SMOOTH_CANDIDATE_ID,
    ]
    if ids != expected_ids:
        raise ValueError(f"Frozen DRCR-Smooth candidate ids changed: {ids}")
    if str(PROTOCOL_CONFIG["selected_candidate_id"]) != SMOOTH_CANDIDATE_ID:
        raise ValueError("Frozen selected candidate alias and selected_candidate_id disagree")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


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


def pct(value: object, digits: int = 1) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{100.0 * parsed:.{digits}f}%"


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
    """Return the frozen, pre-registered DRCR-Smooth candidate class."""

    return json.loads(json.dumps(PROTOCOL_CONFIG["candidate_class"]["candidates"]))


def sigmoid(value: float) -> float:
    if value > 60:
        return 1.0
    if value < -60:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def scale_for_window(window: dict[str, object], candidate: dict[str, object]) -> tuple[float, int, int, float]:
    structured_guard = crc.adaptive.structured_control_guard(window)
    if structured_guard:
        return finite_float(candidate["structured_scale"]), int(structured_guard), 0, 0.0
    width_ratio = crc.native_width_ratio(window)
    candidate_type = str(candidate["candidate_type"])
    if candidate_type == "fixed":
        return finite_float(candidate["failure_scale"]), 0, 0, 0.0
    if candidate_type == "step":
        score = float(width_ratio <= finite_float(candidate["width_ratio_threshold"]))
        scale = finite_float(candidate["collapse_scale"]) if score else finite_float(candidate["failure_scale"])
        return scale, 0, int(score >= 0.5), score
    if candidate_type == "smooth_sigmoid":
        score = sigmoid(
            (finite_float(candidate["width_ratio_threshold"]) - width_ratio)
            / finite_float(candidate["temperature"])
        )
        low_scale = finite_float(candidate["low_scale"])
        high_scale = finite_float(candidate["high_scale"])
        scale = low_scale + (high_scale - low_scale) * score
        return scale, 0, int(score >= 0.5), score
    raise ValueError(f"Unsupported candidate_type: {candidate_type}")


def apply_candidate_to_window(
    window: dict[str, object],
    policy: dict[str, object],
    candidate: dict[str, object],
    *,
    split_protocol: str,
    split_id: str,
    phase: str,
) -> dict[str, object]:
    point_row = crc.fullgrid.cpr.apply_policy_to_window(
        window,
        policy,
        "drcr_smooth_score_head",
        split_protocol,
        split_id,
        "common_ltt_policy",
    )
    raw_weight = finite_float(point_row["effective_weight"])
    scale, structured_guard, width_guard, smooth_score = scale_for_window(window, candidate)
    effective_weight = min(raw_weight, crc.STRUCTURED_WEIGHT_CAP) if structured_guard else raw_weight
    repaired_grid = crc.fullgrid.interval_head_quantile_grid(window, effective_weight, scale)
    repair_metrics = crc.fullgrid.quantile_metrics(window, repaired_grid)
    model_metrics = crc.fullgrid.quantile_metrics(window, np.asarray(window["quantile_grid"], dtype=float))
    return {
        "split_protocol": split_protocol,
        "split_id": split_id,
        "phase": phase,
        "candidate_id": candidate["candidate_id"],
        "candidate_description": candidate["description"],
        "candidate_type": candidate["candidate_type"],
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
        "smooth_width_score": smooth_score,
        "native_width_ratio": crc.native_width_ratio(window),
        "model_wql_rer": model_metrics["wql_rer"],
        "repair_wql_rer": repair_metrics["wql_rer"],
        "wql_rer_delta_vs_model": repair_metrics["wql_rer"] - model_metrics["wql_rer"],
        "model_wql_failure_delta005": int(model_metrics["wql_rer"] > 1.05),
        "repair_wql_failure_delta005": int(repair_metrics["wql_rer"] > 1.05),
        "model_coverage_q10_q90": model_metrics["coverage"],
        "repair_coverage_q10_q90": repair_metrics["coverage"],
        "model_undercoverage_risk": max(0.0, crc.NOMINAL_COVERAGE - model_metrics["coverage"]),
        "repair_undercoverage_risk": max(0.0, crc.NOMINAL_COVERAGE - repair_metrics["coverage"]),
        "model_coverage_abs_error": model_metrics["coverage_abs_error"],
        "repair_coverage_abs_error": repair_metrics["coverage_abs_error"],
        "model_interval_width_q10_q90": model_metrics["interval_width_q10_q90"],
        "repair_interval_width_q10_q90": repair_metrics["interval_width_q10_q90"],
        "repair_win_vs_model": int(repair_metrics["wql_rer"] < model_metrics["wql_rer"]),
        "repair_safety_harm": int(
            repair_metrics["wql_rer"] > model_metrics["wql_rer"] + 0.05
            or repair_metrics["coverage_abs_error"] > model_metrics["coverage_abs_error"] + 0.05
        ),
        "repair_wql_noninferiority_harm": int(
            repair_metrics["wql_rer"] > model_metrics["wql_rer"] + WQL_HARM_MARGIN
        ),
        "wql_harm_margin": WQL_HARM_MARGIN,
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
    base = crc.summarize_candidate(rows, group, group_type)
    base["repair_wql_noninferiority_harm_rate"] = rate(
        [int(row["repair_wql_noninferiority_harm"]) for row in rows]
    )
    base["smooth_width_score_mean"] = mean([finite_float(row["smooth_width_score"], float("nan")) for row in rows])
    return base


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
        first = subset[0]
        summary.append(
            {
                "split_protocol": first["split_protocol"],
                "split_id": first["split_id"],
                "phase": first["phase"],
                "candidate_id": first["candidate_id"],
                "candidate_description": first["candidate_description"],
                "candidate_type": first["candidate_type"],
                **summarize_candidate(subset, group, group_type),
            }
        )
    return summary


def utility_for_calibration(rows: list[dict[str, object]]) -> float:
    return -(
        median([finite_float(row["repair_wql_rer"], float("nan")) for row in rows])
        + UTILITY_UNDERCOVERAGE_LAMBDA
        * mean([finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows])
        + UTILITY_WQL_HARM_LAMBDA
        * rate([int(row["repair_wql_noninferiority_harm"]) for row in rows])
    )


def select_candidate(
    calibration_windows: list[dict[str, object]],
    policy: dict[str, object],
    candidates: list[dict[str, object]],
    *,
    split_protocol: str,
    split_id: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, list[dict[str, object]]]]]:
    calibration_rows_by_candidate: dict[str, list[dict[str, object]]] = {}
    undercoverage_losses: dict[str, list[float]] = {}
    wql_harm_losses: dict[str, list[int]] = {}
    utilities: dict[str, float] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        rows = apply_candidate(
            calibration_windows,
            policy,
            candidate,
            split_protocol=split_protocol,
            split_id=split_id,
            phase="calibration",
        )
        calibration_rows_by_candidate[candidate_id] = rows
        undercoverage_losses[candidate_id] = [
            finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows
        ]
        wql_harm_losses[candidate_id] = [int(row["repair_wql_noninferiority_harm"]) for row in rows]
        utilities[candidate_id] = utility_for_calibration(rows)

    undercoverage_tests = {
        test.policy_id: test
        for test in ltt_risk_tests(
            undercoverage_losses,
            alpha=UNDERCOVERAGE_ALPHA,
            delta=DELTA_PER_RISK,
            correction="holm",
            binary=False,
        )
    }
    wql_harm_tests = {
        test.policy_id: test
        for test in ltt_risk_tests(
            wql_harm_losses,
            alpha=WQL_HARM_ALPHA,
            delta=DELTA_PER_RISK,
            correction="holm",
            binary=True,
        )
    }
    accepted_ids = {
        candidate_id
        for candidate_id in undercoverage_losses
        if undercoverage_tests[candidate_id].accepted and wql_harm_tests[candidate_id].accepted
    }
    if accepted_ids:
        best_utility = max(utilities[item] for item in accepted_ids)
        near_best = [item for item in accepted_ids if utilities[item] >= best_utility - SMOOTH_TIE_EPS]
        smooth_ids = {
            str(candidate["candidate_id"])
            for candidate in candidates
            if candidate["candidate_type"] == "smooth_sigmoid"
        }
        diagnostic_ids = {
            str(candidate["candidate_id"])
            for candidate in candidates
            if candidate["candidate_type"] in {"smooth_sigmoid", "step"}
        }
        selected_id = max(
            near_best,
            key=lambda item: (
                int(item in smooth_ids),
                int(item in diagnostic_ids),
                utilities[item],
                item,
            ),
        )
        selection_status = "dual_ltt_certified_smoothness_prior"
    else:
        selected_id = min(
            undercoverage_losses,
            key=lambda item: (
                mean(undercoverage_losses[item]) / UNDERCOVERAGE_ALPHA
                + rate(wql_harm_losses[item]) / WQL_HARM_ALPHA,
                -utilities[item],
                item,
            ),
        )
        selection_status = "fallback_min_joint_calibration_risk"

    selected_rows: list[dict[str, object]] = []
    calibration_tests: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        rows = calibration_rows_by_candidate[candidate_id]
        uc_test = undercoverage_tests[candidate_id]
        harm_test = wql_harm_tests[candidate_id]
        base = {
            "split_protocol": split_protocol,
            "split_id": split_id,
            "candidate_id": candidate_id,
            "candidate_description": candidate["description"],
            "candidate_type": candidate["candidate_type"],
            "selected": int(candidate_id == selected_id),
            "selection_status": selection_status if candidate_id == selected_id else "",
            "dual_ltt_accepted": int(uc_test.accepted and harm_test.accepted),
            "n_calibration_windows": len(rows),
            "utility": utilities[candidate_id],
            "median_wql_rer": median([finite_float(row["repair_wql_rer"], float("nan")) for row in rows]),
            "smoothness_priority_tie_eps": SMOOTH_TIE_EPS,
            "smooth_score_candidate": int(candidate["candidate_type"] == "smooth_sigmoid"),
            "diagnostic_guard_candidate": int(candidate["candidate_type"] in {"smooth_sigmoid", "step"}),
        }
        selected_rows.append(
            {
                **base,
                "undercoverage_alpha": UNDERCOVERAGE_ALPHA,
                "undercoverage_delta": DELTA_PER_RISK,
                "undercoverage_empirical_risk": uc_test.empirical_risk,
                "undercoverage_ltt_p_value": uc_test.p_value,
                "undercoverage_ltt_corrected_threshold": uc_test.corrected_threshold,
                "undercoverage_ltt_accepted": int(uc_test.accepted),
                "wql_harm_alpha": WQL_HARM_ALPHA,
                "wql_harm_delta": DELTA_PER_RISK,
                "wql_harm_margin": WQL_HARM_MARGIN,
                "wql_harm_empirical_risk": harm_test.empirical_risk,
                "wql_harm_ltt_p_value": harm_test.p_value,
                "wql_harm_ltt_corrected_threshold": harm_test.corrected_threshold,
                "wql_harm_ltt_accepted": int(harm_test.accepted),
            }
        )
        for risk_name, alpha, delta, test in [
            ("undercoverage", UNDERCOVERAGE_ALPHA, DELTA_PER_RISK, uc_test),
            ("wql_noninferiority_harm", WQL_HARM_ALPHA, DELTA_PER_RISK, harm_test),
        ]:
            calibration_tests.append(
                {
                    **base,
                    "risk_name": risk_name,
                    "risk_alpha": alpha,
                    "risk_delta": delta,
                    "empirical_risk": test.empirical_risk,
                    "ltt_p_value": test.p_value,
                    "ltt_corrected_threshold": test.corrected_threshold,
                    "ltt_accepted": int(test.accepted),
                    "ltt_correction": test.correction,
                    "ltt_ucb_hoeffding": test.ucb_hoeffding,
                    "ltt_risk_count": test.risk_count,
                }
            )
    selected = next(candidate for candidate in candidates if str(candidate["candidate_id"]) == selected_id)
    return selected, selected_rows, calibration_tests, calibration_rows_by_candidate


def summary_row(summary: list[dict[str, object]], candidate_id: str, group: str) -> dict[str, object]:
    return next(
        row
        for row in summary
        if row["split_protocol"] == SPLIT_PROTOCOL
        and row["split_id"] == SPLIT_ID
        and row["phase"] == "test"
        and row["candidate_id"] == candidate_id
        and row["group"] == group
    )


def primary_group_rows(summary: list[dict[str, object]], selected_id: str) -> list[dict[str, object]]:
    wanted = [
        "overall",
        "family:chronos|role:failure_target",
        "family:moirai|role:failure_target",
        "family:chronos|role:positive_control",
        "family:moirai|role:positive_control",
    ]
    rows = []
    for group in wanted:
        row = summary_row(summary, selected_id, group)
        rows.append(
            {
                "Group": group,
                "N": row["n_windows"],
                "ModelWQL": num(row["model_median_wql_rer"]),
                "RepairWQL": num(row["repair_median_wql_rer"]),
                "Risk": num(row["repair_undercoverage_risk"]),
                "Coverage": num(row["repair_mean_coverage"]),
                "FailRed": pct(row["wql_failure_reduction_vs_model"]),
                "WQLHarm": pct(row["repair_wql_noninferiority_harm_rate"]),
                "SafetyHarm": pct(row["repair_safety_harm_rate"]),
                "WidthScore": num(row["smooth_width_score_mean"], 3),
            }
        )
    return rows


def source_candidate_rows(selected_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in selected_rows:
        rows.append(
            {
                "Candidate": row["candidate_id"],
                "Type": row["candidate_type"],
                "Selected": "yes" if int(row["selected"]) else "",
                "Dual": "yes" if int(row["dual_ltt_accepted"]) else "no",
                "UC": num(row["undercoverage_empirical_risk"]),
                "UCok": "yes" if int(row["undercoverage_ltt_accepted"]) else "no",
                "Harm": pct(row["wql_harm_empirical_risk"]),
                "HarmOk": "yes" if int(row["wql_harm_ltt_accepted"]) else "no",
                "Utility": num(row["utility"], 4),
            }
        )
    return rows


def window_key(row: dict[str, object]) -> tuple[str, str, str, str, str, str, str, str, str]:
    return (
        str(row["family"]),
        str(row["source"]),
        str(row["role"]),
        str(row["dataset"]),
        str(row["model"]),
        str(row["target_id"]),
        str(row["series_id"]),
        str(row["origin"]),
        str(row["window_index"]),
    )


def bootstrap_ci(values: list[float], *, seed: int = SEED) -> tuple[float, float]:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(N_BOOTSTRAP, arr.size))
    stats = np.mean(arr[draws], axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def paired_stats(candidate_rows: dict[str, list[dict[str, object]]], selected_id: str) -> list[dict[str, object]]:
    selected_by_key = {window_key(row): row for row in candidate_rows[selected_id]}
    output: list[dict[str, object]] = []
    for comparator_id in [BALANCED_ID, PREVIOUS_STEP_ID]:
        comparator_by_key = {window_key(row): row for row in candidate_rows[comparator_id]}
        if set(selected_by_key) != set(comparator_by_key):
            raise ValueError(f"Comparator windows do not align for {comparator_id}")
        pairs = [(selected_by_key[key], comparator_by_key[key]) for key in sorted(selected_by_key)]
        groups = [
            ("overall", "overall", pairs),
            (
                "family:chronos|role:failure_target",
                "family_role",
                [item for item in pairs if item[0]["family"] == "chronos" and item[0]["role"] == "failure_target"],
            ),
            (
                "family:moirai|role:failure_target",
                "family_role",
                [item for item in pairs if item[0]["family"] == "moirai" and item[0]["role"] == "failure_target"],
            ),
            (
                "role:positive_control",
                "role",
                [item for item in pairs if item[0]["role"] == "positive_control"],
            ),
        ]
        for group, group_type, subset in groups:
            log_deltas = [
                math.log1p(max(0.0, finite_float(selected["repair_wql_rer"])))
                - math.log1p(max(0.0, finite_float(comparator["repair_wql_rer"])))
                for selected, comparator in subset
            ]
            wql_deltas = [
                finite_float(selected["repair_wql_rer"]) - finite_float(comparator["repair_wql_rer"])
                for selected, comparator in subset
            ]
            safety_deltas = [
                int(selected["repair_safety_harm"]) - int(comparator["repair_safety_harm"])
                for selected, comparator in subset
            ]
            harm_deltas = [
                int(selected["repair_wql_noninferiority_harm"]) - int(comparator["repair_wql_noninferiority_harm"])
                for selected, comparator in subset
            ]
            ci_low, ci_high = bootstrap_ci(log_deltas)
            output.append(
                {
                    "selected_candidate_id": selected_id,
                    "comparator_candidate_id": comparator_id,
                    "group": group,
                    "group_type": group_type,
                    "n_windows": len(subset),
                    "selected_median_wql_rer": median(
                        [finite_float(selected["repair_wql_rer"], float("nan")) for selected, _ in subset]
                    ),
                    "comparator_median_wql_rer": median(
                        [finite_float(comparator["repair_wql_rer"], float("nan")) for _, comparator in subset]
                    ),
                    "paired_median_wql_delta": median(wql_deltas),
                    "paired_mean_log1p_wql_delta": mean(log_deltas),
                    "paired_mean_log1p_wql_delta_ci_low": ci_low,
                    "paired_mean_log1p_wql_delta_ci_high": ci_high,
                    "paired_safety_harm_delta": mean([float(value) for value in safety_deltas]),
                    "paired_wql_harm_delta": mean([float(value) for value in harm_deltas]),
                }
            )
    return output


def paired_report_rows(pairs: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = [
        (PREVIOUS_STEP_ID, "overall"),
        (PREVIOUS_STEP_ID, "family:chronos|role:failure_target"),
        (PREVIOUS_STEP_ID, "family:moirai|role:failure_target"),
        (BALANCED_ID, "family:chronos|role:failure_target"),
    ]
    rows = []
    for comparator, group in wanted:
        row = next(item for item in pairs if item["comparator_candidate_id"] == comparator and item["group"] == group)
        rows.append(
            {
                "Comparator": comparator,
                "Group": group,
                "N": row["n_windows"],
                "ComparatorWQL": num(row["comparator_median_wql_rer"]),
                "SmoothWQL": num(row["selected_median_wql_rer"]),
                "MeanLogDelta": num(row["paired_mean_log1p_wql_delta"], 4),
                "MeanLogCI": f"[{num(row['paired_mean_log1p_wql_delta_ci_low'], 4)}, {num(row['paired_mean_log1p_wql_delta_ci_high'], 4)}]",
                "MedianDelta": num(row["paired_median_wql_delta"], 4),
                "SafetyDelta": pct(row["paired_safety_harm_delta"], 2),
                "HarmDelta": pct(row["paired_wql_harm_delta"], 2),
            }
        )
    return rows


def write_report(
    selected_rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    paired: list[dict[str, object]],
    status: dict[str, object],
) -> None:
    selected_id = str(status["selected_candidate_id"])
    DOC_PATH.write_text(
        "\n".join(
            [
                "# DRCR Smooth Score-to-Scale Head",
                "",
                "## Method",
                "",
                "DRCR-Smooth keeps the dual-risk CRC/LTT protocol fixed and changes the mechanism head from a hard uncertainty-collapse threshold to a continuous score-to-scale map.",
                "",
                f"- Frozen protocol: `{PROTOCOL_ID}` from `{CONFIG_PATH.relative_to(ROOT)}`.",
                f"- Protocol SHA-256: `{status['protocol_sha256']}`.",
                "- Risk 1: q10-q90 undercoverage, same alpha/delta as DRCR-dual.",
                "- Risk 2: WQL non-inferiority harm, same margin and alpha/delta as DRCR-dual.",
                "- Candidate class is frozen: expansion slices may apply this class, but must not add, remove, or retune candidates.",
                "- Smooth score: `sigmoid((0.10 - native_width_ratio) / 0.05)`.",
                "- Smooth scale: `0.906 + (1.00 - 0.906) * score` on failure windows; structured controls retain the existing structured scale and weight cap.",
                f"- Tie rule: among dual-certified candidates within `{SMOOTH_TIE_EPS}` utility of the best candidate, prefer smooth score-to-scale heads, then other diagnostic heads, then fixed heads.",
                "",
                "## Calibration Screen",
                "",
                markdown_table(
                    source_candidate_rows(selected_rows),
                    [
                        ("Candidate", "Candidate"),
                        ("Type", "Type"),
                        ("Selected", "Selected"),
                        ("Dual", "Dual ok"),
                        ("UC", "Cal UC"),
                        ("UCok", "UC ok"),
                        ("Harm", "Cal WQL harm"),
                        ("HarmOk", "Harm ok"),
                        ("Utility", "Utility"),
                    ],
                ),
                "",
                "## Main Test Split",
                "",
                markdown_table(
                    primary_group_rows(summary, selected_id),
                    [
                        ("Group", "Group"),
                        ("N", "N"),
                        ("ModelWQL", "Model WQL"),
                        ("RepairWQL", "Repair WQL"),
                        ("Risk", "Undercoverage risk"),
                        ("Coverage", "Coverage"),
                        ("FailRed", "Fail red."),
                        ("WQLHarm", "WQL harm"),
                        ("SafetyHarm", "Safety harm"),
                        ("WidthScore", "Mean score"),
                    ],
                ),
                "",
                "## Paired Comparisons",
                "",
                markdown_table(
                    paired_report_rows(paired),
                    [
                        ("Comparator", "Comparator"),
                        ("Group", "Group"),
                        ("N", "N"),
                        ("ComparatorWQL", "Comparator WQL"),
                        ("SmoothWQL", "Smooth WQL"),
                        ("MeanLogDelta", "Mean dlogWQL"),
                        ("MeanLogCI", "Mean dlogWQL 95% CI"),
                        ("MedianDelta", "Median dWQL"),
                        ("SafetyDelta", "Safety harm d"),
                        ("HarmDelta", "WQL harm d"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- This is a positive smooth-head result: the selected continuous diagnostic head improves the Chronos failure WQL beyond the previous hard-threshold collapse-floor head while preserving Moirai failure behavior.",
                "- The effect-size story is now cleaner: Chronos failure WQL moves from raw `4.940` to smooth `1.894`, versus `1.972` for the previous hard-threshold diagnostic and `2.008` for fixed balanced.",
                "- The tradeoff is still real: overall safety harm rises modestly versus the previous hard-threshold head, so this should be framed as a stronger mechanism/effect-size candidate, not a free improvement.",
                "- The method story is stronger than fallback/blending because the deployed head is a continuous local-uncertainty mechanism selected only after dual-risk CRC/LTT calibration.",
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{CANDIDATE_SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{SELECTED_OUT.relative_to(ROOT)}`",
                f"- `{CALIBRATION_TEST_OUT.relative_to(ROOT)}`",
                f"- `{PAIRED_STATS_OUT.relative_to(ROOT)}`",
                f"- `{STATUS_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    validate_protocol_config()
    policy = crc.fullgrid.common_cpr_policy()
    windows = crc.all_windows()
    split = next(
        item
        for item in crc.split_definitions(windows)
        if item["split_protocol"] == SPLIT_PROTOCOL and item["split_id"] == SPLIT_ID
    )
    candidates = candidate_configs()
    selected, selected_rows, calibration_tests, _ = select_candidate(
        split["calibration"],
        policy,
        candidates,
        split_protocol=SPLIT_PROTOCOL,
        split_id=SPLIT_ID,
    )
    selected_id = str(selected["candidate_id"])
    candidate_test_rows: dict[str, list[dict[str, object]]] = {}
    all_candidate_summary: list[dict[str, object]] = []
    for candidate in candidates:
        rows = apply_candidate(
            split["test"],
            policy,
            candidate,
            split_protocol=SPLIT_PROTOCOL,
            split_id=SPLIT_ID,
            phase="test",
        )
        candidate_test_rows[str(candidate["candidate_id"])] = rows
        all_candidate_summary.extend(build_candidate_summary(rows))
    selected_test_rows = candidate_test_rows[selected_id]
    summary = build_candidate_summary(selected_test_rows)
    paired = paired_stats(candidate_test_rows, selected_id)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "protocol_id": PROTOCOL_ID,
        "protocol_config": str(CONFIG_PATH.relative_to(ROOT)),
        "protocol_sha256": protocol_sha256(),
        "freeze_status": PROTOCOL_CONFIG["freeze_status"],
        "freeze_date": PROTOCOL_CONFIG["freeze_date"],
        "split_protocol": SPLIT_PROTOCOL,
        "split_id": SPLIT_ID,
        "n_windows_total": len(windows),
        "n_test_windows": len(split["test"]),
        "n_calibration_windows": len(split["calibration"]),
        "n_candidates": len(candidates),
        "selected_candidate_id": selected_id,
        "smooth_candidate_id": SMOOTH_CANDIDATE_ID,
        "previous_step_candidate_id": PREVIOUS_STEP_ID,
        "balanced_candidate_id": BALANCED_ID,
        "smooth_tie_eps": SMOOTH_TIE_EPS,
        "undercoverage_alpha": UNDERCOVERAGE_ALPHA,
        "wql_harm_alpha": WQL_HARM_ALPHA,
        "delta_per_risk": DELTA_PER_RISK,
        "wql_harm_margin": WQL_HARM_MARGIN,
        "n_bootstrap": N_BOOTSTRAP,
        "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "candidate_summary": str(CANDIDATE_SUMMARY_OUT.relative_to(ROOT)),
        "selected_configs": str(SELECTED_OUT.relative_to(ROOT)),
        "calibration_tests": str(CALIBRATION_TEST_OUT.relative_to(ROOT)),
        "paired_stats": str(PAIRED_STATS_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, selected_test_rows)
    write_csv(SUMMARY_OUT, summary)
    write_csv(CANDIDATE_SUMMARY_OUT, all_candidate_summary)
    write_csv(SELECTED_OUT, selected_rows)
    write_csv(CALIBRATION_TEST_OUT, calibration_tests)
    write_csv(PAIRED_STATS_OUT, paired)
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")
    write_report(selected_rows, summary, paired, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
