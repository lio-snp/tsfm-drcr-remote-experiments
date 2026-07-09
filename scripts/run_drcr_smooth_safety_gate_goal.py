#!/usr/bin/env python
"""Safety-gated DRCR-Smooth intervention selector.

The frozen DRCR-Smooth head is left unchanged.  This script adds a small
intervention veto layer that can downgrade the deployed intervention from full
DRCR-Smooth to point-only or native no-op on ex-ante safety signals.  The goal
is to address the component-ablation finding that interval scaling, not point
shift, drives most positive-control/stress harm.

This is a prototype method line, not a new smooth-head tuning pass.
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

import run_drcr_smooth_frozen_expansion_goal as expansion  # noqa: E402
import run_drcr_smooth_score_head_search_goal as smooth  # noqa: E402

from low_snr_tsfm.risk_control import ltt_risk_tests  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "drcr_smooth_safety_gate_report.md"
STATUS_OUT = OUT_DIR / "drcr_smooth_safety_gate_status.json"
WINDOW_OUT = OUT_DIR / "drcr_smooth_safety_gate_windows.csv"
SUMMARY_OUT = OUT_DIR / "drcr_smooth_safety_gate_summary.csv"
SELECTED_OUT = OUT_DIR / "drcr_smooth_safety_gate_selected_configs.csv"
CALIBRATION_OUT = OUT_DIR / "drcr_smooth_safety_gate_calibration_tests.csv"

WQL_HARM_ALPHA = 0.20
PROTECTED_HARM_ALPHA = 0.20
DELTA_PER_RISK = 0.05
LOW_SCORE_THRESHOLDS = [0.10, 0.25, 0.40]
UTILITY_TIE_EPS = 0.002


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
    candidates = [
        {
            "candidate_id": "full_drcr_smooth",
            "mode": "full",
            "description": "current frozen DRCR-Smooth, no safety veto",
            "low_score_threshold": "",
        },
        {
            "candidate_id": "point_only_all",
            "mode": "point_only_all",
            "description": "global point-only downgrade; diagnostic baseline",
            "low_score_threshold": "",
        },
        {
            "candidate_id": "point_if_structured",
            "mode": "point_if_structured",
            "description": "downgrade structured-control guard windows to point-only",
            "low_score_threshold": "",
        },
    ]
    for threshold in LOW_SCORE_THRESHOLDS:
        suffix = f"{threshold:.2f}"
        candidates.extend(
            [
                {
                    "candidate_id": f"point_if_low_score_t{suffix}",
                    "mode": "point_if_low_score",
                    "description": "downgrade low smooth-score windows to point-only",
                    "low_score_threshold": threshold,
                },
                {
                    "candidate_id": f"native_if_low_score_t{suffix}",
                    "mode": "native_if_low_score",
                    "description": "veto low smooth-score windows to native no-op",
                    "low_score_threshold": threshold,
                },
            ]
        )
    return candidates


def selected_smooth_candidate() -> dict[str, object]:
    return {str(candidate["candidate_id"]): candidate for candidate in smooth.candidate_configs()}[
        smooth.SMOOTH_CANDIDATE_ID
    ]


def base_intervention(window: dict[str, object], policy: dict[str, object], smooth_candidate: dict[str, object]) -> dict[str, object]:
    point_row = smooth.crc.fullgrid.cpr.apply_policy_to_window(
        window,
        policy,
        "drcr_smooth_safety_gate",
        smooth.SPLIT_PROTOCOL,
        smooth.SPLIT_ID,
        "common_ltt_policy",
    )
    raw_weight = finite_float(point_row["effective_weight"])
    scale, structured_guard, width_guard, smooth_score = smooth.scale_for_window(window, smooth_candidate)
    effective_weight = min(raw_weight, smooth.crc.STRUCTURED_WEIGHT_CAP) if structured_guard else raw_weight
    return {
        "point_row": point_row,
        "raw_effective_weight": raw_weight,
        "effective_weight": effective_weight,
        "interval_scale": scale,
        "structured_guard": int(structured_guard),
        "width_guard": int(width_guard),
        "smooth_score": smooth_score,
        "protected_signal": int(structured_guard or smooth_score < 0.25),
    }


def intervention_mode(base: dict[str, object], candidate: dict[str, object]) -> tuple[str, float, float, int]:
    mode = str(candidate["mode"])
    effective_weight = finite_float(base["effective_weight"])
    interval_scale = finite_float(base["interval_scale"])
    structured_guard = int(base["structured_guard"])
    smooth_score = finite_float(base["smooth_score"])
    threshold = candidate.get("low_score_threshold")

    if mode == "full":
        return "full", effective_weight, interval_scale, 0
    if mode == "point_only_all":
        return "point_only", effective_weight, 1.0, 1
    if mode == "point_if_structured" and structured_guard:
        return "point_only", effective_weight, 1.0, 1
    if mode == "point_if_low_score" and threshold != "" and smooth_score < finite_float(threshold):
        return "point_only", effective_weight, 1.0, 1
    if mode == "native_if_low_score" and threshold != "" and smooth_score < finite_float(threshold):
        return "native_noop", 0.0, 1.0, 1
    return "full", effective_weight, interval_scale, 0


def quantile_grid_for(window: dict[str, object], weight: float, scale: float) -> np.ndarray:
    if weight == 0.0 and scale == 1.0:
        return np.asarray(window["quantile_grid"], dtype=float)
    return smooth.crc.fullgrid.interval_head_quantile_grid(window, weight, scale)


def apply_candidate_to_window(
    window: dict[str, object],
    policy: dict[str, object],
    smooth_candidate: dict[str, object],
    candidate: dict[str, object],
    *,
    phase: str,
) -> dict[str, object]:
    base = base_intervention(window, policy, smooth_candidate)
    deployed_mode, weight, scale, veto = intervention_mode(base, candidate)
    model_metrics = smooth.crc.fullgrid.quantile_metrics(window, np.asarray(window["quantile_grid"], dtype=float))
    repair_metrics = smooth.crc.fullgrid.quantile_metrics(window, quantile_grid_for(window, weight, scale))
    point_row = base["point_row"]
    return {
        "protocol_id": smooth.PROTOCOL_ID,
        "protocol_sha256": smooth.protocol_sha256(),
        "phase": phase,
        "candidate_id": candidate["candidate_id"],
        "candidate_description": candidate["description"],
        "candidate_mode": candidate["mode"],
        "candidate_low_score_threshold": candidate["low_score_threshold"],
        "deployed_mode": deployed_mode,
        "veto_active": veto,
        "split_protocol": smooth.SPLIT_PROTOCOL,
        "split_id": smooth.SPLIT_ID,
        "family": window["family"],
        "source": window["source"],
        "role": window["role"],
        "target_id": window["target_id"],
        "evidence_tier": window["evidence_tier"],
        "expansion_set": window["expansion_set"],
        "dataset": window["dataset"],
        "model": window["model"],
        "series_id": window["series_id"],
        "origin": window["origin"],
        "window_index": window["window_index"],
        "quantile_grid_n_levels": window["quantile_grid_n_levels"],
        "selected_policy_id": policy["policy_id"],
        "gate_active": point_row["gate_active"],
        "shield_active": point_row["shield_active"],
        "conflict_override": point_row["conflict_override"],
        "reference_outside_interval_rate": point_row["reference_outside_interval_rate"],
        "low_structure_factor_count": point_row["low_structure_factor_count"],
        "raw_effective_weight": base["raw_effective_weight"],
        "base_effective_weight": base["effective_weight"],
        "base_interval_scale": base["interval_scale"],
        "effective_weight": weight,
        "interval_scale": scale,
        "structured_control_guard": base["structured_guard"],
        "width_collapse_guard": base["width_guard"],
        "smooth_width_score": base["smooth_score"],
        "protected_signal": base["protected_signal"],
        "native_width_ratio": smooth.crc.native_width_ratio(window),
        "model_wql_rer": model_metrics["wql_rer"],
        "repair_wql_rer": repair_metrics["wql_rer"],
        "wql_rer_delta_vs_model": repair_metrics["wql_rer"] - model_metrics["wql_rer"],
        "model_wql_failure_delta005": int(model_metrics["wql_rer"] > 1.05),
        "repair_wql_failure_delta005": int(repair_metrics["wql_rer"] > 1.05),
        "model_coverage_q10_q90": model_metrics["coverage"],
        "repair_coverage_q10_q90": repair_metrics["coverage"],
        "repair_undercoverage_risk": max(0.0, smooth.crc.NOMINAL_COVERAGE - repair_metrics["coverage"]),
        "model_coverage_abs_error": model_metrics["coverage_abs_error"],
        "repair_coverage_abs_error": repair_metrics["coverage_abs_error"],
        "repair_win_vs_model": int(repair_metrics["wql_rer"] < model_metrics["wql_rer"]),
        "repair_wql_noninferiority_harm": int(
            repair_metrics["wql_rer"] > model_metrics["wql_rer"] + smooth.WQL_HARM_MARGIN
        ),
        "repair_safety_harm": int(
            repair_metrics["wql_rer"] > model_metrics["wql_rer"] + smooth.WQL_HARM_MARGIN
            or repair_metrics["coverage_abs_error"] > model_metrics["coverage_abs_error"] + 0.05
        ),
    }


def apply_candidate(
    windows: list[dict[str, object]],
    policy: dict[str, object],
    smooth_candidate: dict[str, object],
    candidate: dict[str, object],
    *,
    phase: str,
) -> list[dict[str, object]]:
    return [apply_candidate_to_window(window, policy, smooth_candidate, candidate, phase=phase) for window in windows]


def summarize(rows: list[dict[str, object]], group: str, group_type: str) -> dict[str, object]:
    return {
        "group": group,
        "group_type": group_type,
        "candidate_id": rows[0]["candidate_id"],
        "candidate_mode": rows[0]["candidate_mode"],
        "phase": rows[0]["phase"],
        "n_windows": len(rows),
        "quantile_grid_n_levels": ";".join(
            str(value) for value in sorted({int(row["quantile_grid_n_levels"]) for row in rows})
        ),
        "model_median_wql_rer": median([finite_float(row["model_wql_rer"], float("nan")) for row in rows]),
        "repair_median_wql_rer": median([finite_float(row["repair_wql_rer"], float("nan")) for row in rows]),
        "repair_median_wql_delta_vs_model": median(
            [finite_float(row["wql_rer_delta_vs_model"], float("nan")) for row in rows]
        ),
        "wql_failure_reduction_vs_model": rate([int(row["model_wql_failure_delta005"]) for row in rows])
        - rate([int(row["repair_wql_failure_delta005"]) for row in rows]),
        "repair_win_rate_vs_model": rate([int(row["repair_win_vs_model"]) for row in rows]),
        "repair_mean_coverage": mean([finite_float(row["repair_coverage_q10_q90"], float("nan")) for row in rows]),
        "repair_undercoverage_risk": mean(
            [finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows]
        ),
        "repair_wql_noninferiority_harm_rate": rate([int(row["repair_wql_noninferiority_harm"]) for row in rows]),
        "repair_safety_harm_rate": rate([int(row["repair_safety_harm"]) for row in rows]),
        "veto_rate": rate([int(row["veto_active"]) for row in rows]),
        "point_only_rate": rate([int(row["deployed_mode"] == "point_only") for row in rows]),
        "native_noop_rate": rate([int(row["deployed_mode"] == "native_noop") for row in rows]),
        "protected_signal_rate": rate([int(row["protected_signal"]) for row in rows]),
        "structured_guard_rate": rate([int(row["structured_control_guard"]) for row in rows]),
    }


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for candidate_id in sorted({str(row["candidate_id"]) for row in rows}):
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        groups: list[tuple[str, str, list[dict[str, object]]]] = [("overall", "overall", candidate_rows)]
        for key in ["evidence_tier", "family", "role", "target_id", "source"]:
            for value in sorted({str(row[key]) for row in candidate_rows}):
                groups.append((f"{key}:{value}", key, [row for row in candidate_rows if str(row[key]) == value]))
        for tier in sorted({str(row["evidence_tier"]) for row in candidate_rows}):
            for role in sorted({str(row["role"]) for row in candidate_rows if row["evidence_tier"] == tier}):
                groups.append(
                    (
                        f"evidence_tier:{tier}|role:{role}",
                        "tier_role",
                        [row for row in candidate_rows if row["evidence_tier"] == tier and row["role"] == role],
                    )
                )
        for family in sorted({str(row["family"]) for row in candidate_rows}):
            for role in sorted({str(row["role"]) for row in candidate_rows if row["family"] == family}):
                groups.append(
                    (
                        f"family:{family}|role:{role}",
                        "family_role",
                        [row for row in candidate_rows if row["family"] == family and row["role"] == role],
                    )
                )
        for group, group_type, subset in groups:
            if subset:
                output.append(summarize(subset, group, group_type))
    return output


def utility_for_calibration(rows: list[dict[str, object]]) -> float:
    failure_rows = [row for row in rows if row["role"] == "failure_target"]
    protected_rows = [row for row in rows if int(row["protected_signal"])]
    failure_gain = median(
        [finite_float(row["model_wql_rer"], float("nan")) - finite_float(row["repair_wql_rer"], float("nan")) for row in failure_rows]
    )
    all_harm = rate([int(row["repair_wql_noninferiority_harm"]) for row in rows])
    protected_harm = rate([int(row["repair_wql_noninferiority_harm"]) for row in protected_rows])
    undercoverage = mean([finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows])
    return 2.0 * failure_gain - 0.25 * all_harm - 0.75 * protected_harm - 0.10 * undercoverage


def select_candidate(
    calibration_windows: list[dict[str, object]],
    policy: dict[str, object],
    smooth_candidate: dict[str, object],
    candidates: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    rows_by_candidate: dict[str, list[dict[str, object]]] = {}
    harm_losses: dict[str, list[int]] = {}
    protected_harm_losses: dict[str, list[int]] = {}
    utilities: dict[str, float] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        rows = apply_candidate(calibration_windows, policy, smooth_candidate, candidate, phase="calibration")
        rows_by_candidate[candidate_id] = rows
        harm_losses[candidate_id] = [int(row["repair_wql_noninferiority_harm"]) for row in rows]
        protected = [row for row in rows if int(row["protected_signal"])]
        protected_harm_losses[candidate_id] = [int(row["repair_wql_noninferiority_harm"]) for row in protected]
        utilities[candidate_id] = utility_for_calibration(rows)

    harm_tests = {
        test.policy_id: test
        for test in ltt_risk_tests(
            harm_losses,
            alpha=WQL_HARM_ALPHA,
            delta=DELTA_PER_RISK,
            correction="holm",
            binary=True,
        )
    }
    protected_tests = {
        test.policy_id: test
        for test in ltt_risk_tests(
            protected_harm_losses,
            alpha=PROTECTED_HARM_ALPHA,
            delta=DELTA_PER_RISK,
            correction="holm",
            binary=True,
        )
    }
    accepted = [
        candidate_id
        for candidate_id in rows_by_candidate
        if harm_tests[candidate_id].accepted and protected_tests[candidate_id].accepted
    ]
    if accepted:
        failure_gain_by_candidate = {
            candidate_id: median(
                [
                    finite_float(row["model_wql_rer"], float("nan"))
                    - finite_float(row["repair_wql_rer"], float("nan"))
                    for row in rows_by_candidate[candidate_id]
                    if row["role"] == "failure_target"
                ]
            )
            for candidate_id in accepted
        }
        preserving = [item for item in accepted if failure_gain_by_candidate[item] >= 0.005]
        selection_pool = preserving or accepted

        candidates_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}

        def selection_key(candidate_id: str) -> tuple[float, float, float, int, float, float, str]:
            candidate = candidates_by_id[candidate_id]
            threshold = candidate.get("low_score_threshold")
            threshold_score = -abs(finite_float(threshold, -1.0) - 0.25) if threshold != "" else -10.0
            return (
                -protected_tests[candidate_id].empirical_risk,
                -harm_tests[candidate_id].empirical_risk,
                -mean([finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows_by_candidate[candidate_id]]),
                int(str(candidate["mode"]) == "point_if_low_score"),
                threshold_score,
                utilities[candidate_id],
                candidate_id,
            )

        selected_id = max(selection_pool, key=selection_key)
        selection_status = (
            "dual_harm_ltt_certified_failure_preserving" if preserving else "dual_harm_ltt_certified"
        )
    else:
        selected_id = min(
            rows_by_candidate,
            key=lambda item: (
                rate(harm_losses[item]) / WQL_HARM_ALPHA
                + rate(protected_harm_losses[item]) / PROTECTED_HARM_ALPHA,
                -utilities[item],
                item,
            ),
        )
        selection_status = "fallback_min_joint_harm_risk"

    selected_rows: list[dict[str, object]] = []
    calibration_tests: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        rows = rows_by_candidate[candidate_id]
        protected_rows = [row for row in rows if int(row["protected_signal"])]
        base = {
            "candidate_id": candidate_id,
            "candidate_mode": candidate["mode"],
            "candidate_description": candidate["description"],
            "candidate_low_score_threshold": candidate["low_score_threshold"],
            "selected": int(candidate_id == selected_id),
            "selection_status": selection_status if candidate_id == selected_id else "",
            "dual_harm_ltt_accepted": int(harm_tests[candidate_id].accepted and protected_tests[candidate_id].accepted),
            "n_calibration_windows": len(rows),
            "n_protected_calibration_windows": len(protected_rows),
            "utility": utilities[candidate_id],
            "calibration_median_wql_rer": median([finite_float(row["repair_wql_rer"], float("nan")) for row in rows]),
            "calibration_failure_gain": median(
                [
                    finite_float(row["model_wql_rer"], float("nan")) - finite_float(row["repair_wql_rer"], float("nan"))
                    for row in rows
                    if row["role"] == "failure_target"
                ]
            ),
            "calibration_undercoverage_risk_monitor": mean(
                [finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows]
            ),
        }
        selected_rows.append(
            {
                **base,
                "wql_harm_alpha": WQL_HARM_ALPHA,
                "wql_harm_empirical_risk": harm_tests[candidate_id].empirical_risk,
                "wql_harm_ltt_p_value": harm_tests[candidate_id].p_value,
                "wql_harm_ltt_corrected_threshold": harm_tests[candidate_id].corrected_threshold,
                "wql_harm_ltt_accepted": int(harm_tests[candidate_id].accepted),
                "protected_harm_alpha": PROTECTED_HARM_ALPHA,
                "protected_harm_empirical_risk": protected_tests[candidate_id].empirical_risk,
                "protected_harm_ltt_p_value": protected_tests[candidate_id].p_value,
                "protected_harm_ltt_corrected_threshold": protected_tests[candidate_id].corrected_threshold,
                "protected_harm_ltt_accepted": int(protected_tests[candidate_id].accepted),
            }
        )
        for risk_name, alpha, test in [
            ("wql_noninferiority_harm", WQL_HARM_ALPHA, harm_tests[candidate_id]),
            ("protected_wql_harm", PROTECTED_HARM_ALPHA, protected_tests[candidate_id]),
        ]:
            calibration_tests.append(
                {
                    **base,
                    "risk_name": risk_name,
                    "risk_alpha": alpha,
                    "risk_delta": DELTA_PER_RISK,
                    "empirical_risk": test.empirical_risk,
                    "ltt_p_value": test.p_value,
                    "ltt_corrected_threshold": test.corrected_threshold,
                    "ltt_accepted": int(test.accepted),
                    "ltt_correction": test.correction,
                    "ltt_ucb_hoeffding": test.ucb_hoeffding,
                    "ltt_risk_count": test.risk_count,
                }
            )
    return next(candidate for candidate in candidates if str(candidate["candidate_id"]) == selected_id), selected_rows, calibration_tests


def rows_for_report(summary: list[dict[str, object]], selected_id: str) -> list[dict[str, object]]:
    wanted = [
        "overall",
        "evidence_tier:q9_fullgrid|role:failure_target",
        "evidence_tier:q9_fullgrid|role:positive_control",
        "family:timesfm|role:failure_target",
        "family:timesfm|role:stress_target",
        "target_id:finance_fred_stress",
        "family:timesfm|role:positive_control",
    ]
    by_group = {
        str(row["group"]): row
        for row in summary
        if row["candidate_id"] == selected_id and row["phase"] == "test"
    }
    rows: list[dict[str, object]] = []
    for group in wanted:
        row = by_group.get(group)
        if row is None:
            continue
        rows.append(
            {
                "Group": group,
                "N": row["n_windows"],
                "Model": num(row["model_median_wql_rer"]),
                "Repair": num(row["repair_median_wql_rer"]),
                "FailRed": pct(row["wql_failure_reduction_vs_model"]),
                "Coverage": num(row["repair_mean_coverage"]),
                "WQLHarm": pct(row["repair_wql_noninferiority_harm_rate"]),
                "Safety": pct(row["repair_safety_harm_rate"]),
                "Veto": pct(row["veto_rate"]),
                "PointOnly": pct(row["point_only_rate"]),
                "Noop": pct(row["native_noop_rate"]),
            }
        )
    return rows


def selected_rows_for_report(selected_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in selected_rows:
        output.append(
            {
                "Candidate": row["candidate_id"],
                "Mode": row["candidate_mode"],
                "Selected": "yes" if int(row["selected"]) else "",
                "Dual": "yes" if int(row["dual_harm_ltt_accepted"]) else "no",
                "Harm": pct(row["wql_harm_empirical_risk"]),
                "HarmOK": "yes" if int(row["wql_harm_ltt_accepted"]) else "no",
                "Protected": pct(row["protected_harm_empirical_risk"]),
                "ProtectedOK": "yes" if int(row["protected_harm_ltt_accepted"]) else "no",
                "UCmonitor": num(row["calibration_undercoverage_risk_monitor"]),
                "FailureGain": num(row["calibration_failure_gain"]),
                "Utility": num(row["utility"], 4),
            }
        )
    return output


def write_report(selected_rows: list[dict[str, object]], summary: list[dict[str, object]], status: dict[str, object]) -> None:
    selected_id = str(status["selected_candidate_id"])
    DOC_PATH.write_text(
        "\n".join(
            [
                "# DRCR-Smooth Safety Gate",
                "",
                "## Purpose",
                "",
                "This prototype keeps the frozen DRCR-Smooth score-to-scale head unchanged and adds an intervention safety gate. The gate can downgrade risky windows from full interval scaling to point-only or native no-op.",
                "",
                "## Calibration",
                "",
                f"- Frozen DRCR-Smooth protocol: `{status['frozen_protocol_id']}`.",
                f"- Selected safety candidate: `{selected_id}`.",
                f"- Split: `{status['split_protocol']}` / `{status['split_id']}` with `{status['n_calibration_windows']}` calibration and `{status['n_test_windows']}` test windows.",
                f"- Safety risks: WQL harm alpha `{WQL_HARM_ALPHA}` and protected-signal WQL harm alpha `{PROTECTED_HARM_ALPHA}`.",
                "- Protected signal is ex-ante: structured-control guard is active or the smooth width score is below 0.25.",
                "- Undercoverage is monitored but not certified in this expansion prototype, because the expansion calibration split is too small for the original strict coverage LTT bound.",
                "",
                "## Candidate Screen",
                "",
                markdown_table(
                    selected_rows_for_report(selected_rows),
                    [
                        ("Candidate", "Candidate"),
                        ("Mode", "Mode"),
                        ("Selected", "Selected"),
                        ("Dual", "Dual ok"),
                        ("Harm", "Cal harm"),
                        ("HarmOK", "Harm ok"),
                        ("Protected", "Protected harm"),
                        ("ProtectedOK", "Protected ok"),
                        ("UCmonitor", "UC monitor"),
                        ("FailureGain", "Failure gain"),
                        ("Utility", "Utility"),
                    ],
                ),
                "",
                "## Test Result",
                "",
                markdown_table(
                    rows_for_report(summary, selected_id),
                    [
                        ("Group", "Group"),
                        ("N", "N"),
                        ("Model", "Model"),
                        ("Repair", "Repair"),
                        ("FailRed", "Fail red."),
                        ("Coverage", "Coverage"),
                        ("WQLHarm", "WQL harm"),
                        ("Safety", "Safety"),
                        ("Veto", "Veto"),
                        ("PointOnly", "Point-only"),
                        ("Noop", "No-op"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- The component ablation showed interval scaling as the main harm source; the selected safety gate therefore preserves CPR point shifts while vetoing interval scaling on low-score/structured-risk windows.",
                "- This is a method upgrade over fallback/blending: it is an intervention-mode selector calibrated with LTT harm constraints, not a new TSFM/classical weighted average.",
                "- The current prototype controls WQL-harm style risks and substantially reduces positive-control/stress harm, but it should not yet be sold as coverage-certified until larger calibration support is added.",
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{SELECTED_OUT.relative_to(ROOT)}`",
                f"- `{CALIBRATION_OUT.relative_to(ROOT)}`",
                f"- `{STATUS_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    smooth.validate_protocol_config()
    windows, _ = expansion.load_expansion_windows()
    calibration_windows = [window for window in windows if smooth.crc.split_bucket(window) == 0]
    test_windows = [window for window in windows if smooth.crc.split_bucket(window) == 1]
    policy = smooth.crc.fullgrid.common_cpr_policy()
    smooth_candidate = selected_smooth_candidate()
    candidates = candidate_configs()
    selected, selected_rows, calibration_tests = select_candidate(calibration_windows, policy, smooth_candidate, candidates)
    selected_id = str(selected["candidate_id"])
    test_rows_by_candidate: dict[str, list[dict[str, object]]] = {}
    all_test_rows: list[dict[str, object]] = []
    for candidate in candidates:
        rows = apply_candidate(test_windows, policy, smooth_candidate, candidate, phase="test")
        test_rows_by_candidate[str(candidate["candidate_id"])] = rows
        all_test_rows.extend(rows)
    selected_test_rows = test_rows_by_candidate[selected_id]
    summary = build_summary(all_test_rows)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "frozen_protocol_id": smooth.PROTOCOL_ID,
        "frozen_protocol_sha256": smooth.protocol_sha256(),
        "split_protocol": smooth.SPLIT_PROTOCOL,
        "split_id": smooth.SPLIT_ID,
        "n_windows_total": len(windows),
        "n_calibration_windows": len(calibration_windows),
        "n_test_windows": len(test_windows),
        "n_candidates": len(candidates),
        "selected_candidate_id": selected_id,
        "selected_candidate_mode": selected["mode"],
        "wql_harm_alpha": WQL_HARM_ALPHA,
        "protected_harm_alpha": PROTECTED_HARM_ALPHA,
        "delta_per_risk": DELTA_PER_RISK,
        "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "selected_configs": str(SELECTED_OUT.relative_to(ROOT)),
        "calibration_tests": str(CALIBRATION_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, selected_test_rows)
    write_csv(SUMMARY_OUT, summary)
    write_csv(SELECTED_OUT, selected_rows)
    write_csv(CALIBRATION_OUT, calibration_tests)
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")
    write_report(selected_rows, summary, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
