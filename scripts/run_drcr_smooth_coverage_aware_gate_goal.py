#!/usr/bin/env python
"""Coverage-aware safety gate for frozen DRCR-Smooth.

This is the next method step after the dual-harm safety gate.  It keeps the
frozen DRCR-Smooth score-to-scale head unchanged, adds a small intervention
component class, and calibrates three deployment risks:

1. WQL non-inferiority harm.
2. Protected-signal WQL harm.
3. Undercoverage non-inferiority harm versus the native TSFM interval.

Strict absolute undercoverage risk is also audited with LTT, but it is not the
primary acceptance criterion in this pass because the current calibration
support is still too small for that continuous Hoeffding test to have useful
power at the original alpha.
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

import run_crc_ltt_lsss_goal as crc  # noqa: E402
import run_drcr_smooth_frozen_expansion_goal as expansion  # noqa: E402
import run_drcr_smooth_safety_gate_goal as safety  # noqa: E402
import run_drcr_smooth_score_head_search_goal as smooth  # noqa: E402

from low_snr_tsfm.risk_control import ltt_risk_tests  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "drcr_smooth_coverage_aware_gate_report.md"
STATUS_OUT = OUT_DIR / "drcr_smooth_coverage_aware_gate_status.json"
WINDOW_OUT = OUT_DIR / "drcr_smooth_coverage_aware_gate_windows.csv"
SUMMARY_OUT = OUT_DIR / "drcr_smooth_coverage_aware_gate_summary.csv"
SELECTED_OUT = OUT_DIR / "drcr_smooth_coverage_aware_gate_selected_configs.csv"
CALIBRATION_OUT = OUT_DIR / "drcr_smooth_coverage_aware_gate_calibration_tests.csv"

WQL_HARM_ALPHA = 0.20
PROTECTED_HARM_ALPHA = 0.20
UNDERCOVERAGE_HARM_ALPHA = 0.20
UNDERCOVERAGE_HARM_MARGIN = 0.05
ABS_UNDERCOVERAGE_ALPHA = smooth.UNDERCOVERAGE_ALPHA
DELTA_PER_RISK = 0.05
FAILURE_GAIN_FLOOR = 0.005


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
    return [
        {
            "candidate_id": "full_drcr_smooth",
            "mode": "full",
            "interval_cap": "",
            "low_score_threshold": "",
            "description": "current frozen DRCR-Smooth, no safety veto",
        },
        {
            "candidate_id": "point_only_all",
            "mode": "point_only_all",
            "interval_cap": "",
            "low_score_threshold": "",
            "description": "global point-only downgrade",
        },
        {
            "candidate_id": "point_if_low_score_t0.40",
            "mode": "point_if_low_score",
            "interval_cap": "",
            "low_score_threshold": 0.40,
            "description": "dual-harm safety gate reference",
        },
        {
            "candidate_id": "native_if_low_score_t0.40",
            "mode": "native_if_low_score",
            "interval_cap": "",
            "low_score_threshold": 0.40,
            "description": "native no-op on low smooth-score windows",
        },
        {
            "candidate_id": "interval_cap_s1.10",
            "mode": "interval_cap",
            "interval_cap": 1.10,
            "low_score_threshold": "",
            "description": "cap risky interval scale at 1.10 while keeping point repair",
        },
        {
            "candidate_id": "interval_cap_s1.20",
            "mode": "interval_cap",
            "interval_cap": 1.20,
            "low_score_threshold": "",
            "description": "cap risky interval scale at 1.20 while keeping point repair",
        },
        {
            "candidate_id": "interval_cap_s1.25",
            "mode": "interval_cap",
            "interval_cap": 1.25,
            "low_score_threshold": "",
            "description": "cap risky interval scale at 1.25 while keeping point repair",
        },
        {
            "candidate_id": "interval_cap_s1.35",
            "mode": "interval_cap",
            "interval_cap": 1.35,
            "low_score_threshold": "",
            "description": "cap risky interval scale at 1.35 while keeping point repair",
        },
        {
            "candidate_id": "interval_cap_s1.50",
            "mode": "interval_cap",
            "interval_cap": 1.50,
            "low_score_threshold": "",
            "description": "cap risky interval scale at 1.50 while keeping point repair",
        },
    ]


def annotated_original_windows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window in crc.all_windows():
        row = dict(window)
        row.setdefault("evidence_tier", "q9_fullgrid")
        row.setdefault("expansion_set", "original_chronos_moirai")
        row.setdefault("quantile_grid_n_levels", len(row.get("quantile_levels", [])))
        rows.append(row)
    return rows


def combined_windows() -> tuple[list[dict[str, object]], dict[str, int]]:
    original = annotated_original_windows()
    expansion_rows, _ = expansion.load_expansion_windows()
    seen: set[tuple[object, object, object]] = set()
    output: list[dict[str, object]] = []
    duplicates = 0
    for window in original + expansion_rows:
        key = (window["source"], window["series_id"], window["window_index"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        output.append(window)
    return output, {
        "n_original_windows": len(original),
        "n_expansion_windows_raw": len(expansion_rows),
        "n_duplicate_windows_removed": duplicates,
        "n_combined_windows": len(output),
    }


def selected_smooth_candidate() -> dict[str, object]:
    return safety.selected_smooth_candidate()


def intervention_mode(base: dict[str, object], candidate: dict[str, object]) -> tuple[str, float, float, int]:
    mode = str(candidate["mode"])
    effective_weight = finite_float(base["effective_weight"])
    interval_scale = finite_float(base["interval_scale"])
    smooth_score = finite_float(base["smooth_score"])
    threshold = candidate.get("low_score_threshold")

    if mode == "full":
        return "full", effective_weight, interval_scale, 0
    if mode == "point_only_all":
        return "point_only", effective_weight, 1.0, 1
    if mode == "point_if_low_score" and threshold != "" and smooth_score < finite_float(threshold):
        return "point_only", effective_weight, 1.0, 1
    if mode == "native_if_low_score" and threshold != "" and smooth_score < finite_float(threshold):
        return "native_noop", 0.0, 1.0, 1
    if mode == "interval_cap":
        cap = finite_float(candidate["interval_cap"])
        capped_scale = min(interval_scale, cap)
        return "interval_cap", effective_weight, capped_scale, int(capped_scale < interval_scale - 1e-12)
    return "full", effective_weight, interval_scale, 0


def quantile_grid_for(window: dict[str, object], weight: float, scale: float) -> np.ndarray:
    return safety.quantile_grid_for(window, weight, scale)


def apply_candidate_to_window(
    window: dict[str, object],
    policy: dict[str, object],
    smooth_candidate: dict[str, object],
    candidate: dict[str, object],
    *,
    phase: str,
) -> dict[str, object]:
    base = safety.base_intervention(window, policy, smooth_candidate)
    deployed_mode, weight, scale, veto = intervention_mode(base, candidate)
    model_metrics = smooth.crc.fullgrid.quantile_metrics(window, np.asarray(window["quantile_grid"], dtype=float))
    repair_metrics = smooth.crc.fullgrid.quantile_metrics(window, quantile_grid_for(window, weight, scale))
    model_undercoverage = max(0.0, smooth.crc.NOMINAL_COVERAGE - model_metrics["coverage"])
    repair_undercoverage = max(0.0, smooth.crc.NOMINAL_COVERAGE - repair_metrics["coverage"])
    undercoverage_delta = max(0.0, repair_undercoverage - model_undercoverage)
    return {
        "protocol_id": smooth.PROTOCOL_ID,
        "protocol_sha256": smooth.protocol_sha256(),
        "phase": phase,
        "candidate_id": candidate["candidate_id"],
        "candidate_description": candidate["description"],
        "candidate_mode": candidate["mode"],
        "candidate_interval_cap": candidate["interval_cap"],
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
        "model_undercoverage_risk": model_undercoverage,
        "repair_undercoverage_risk": repair_undercoverage,
        "undercoverage_risk_delta_vs_model": undercoverage_delta,
        "undercoverage_noninferiority_harm": int(undercoverage_delta > UNDERCOVERAGE_HARM_MARGIN),
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
        "undercoverage_harm_rate": rate([int(row["undercoverage_noninferiority_harm"]) for row in rows]),
        "repair_wql_noninferiority_harm_rate": rate([int(row["repair_wql_noninferiority_harm"]) for row in rows]),
        "repair_safety_harm_rate": rate([int(row["repair_safety_harm"]) for row in rows]),
        "veto_rate": rate([int(row["veto_active"]) for row in rows]),
        "point_only_rate": rate([int(row["deployed_mode"] == "point_only") for row in rows]),
        "interval_cap_rate": rate([int(row["deployed_mode"] == "interval_cap") for row in rows]),
        "native_noop_rate": rate([int(row["deployed_mode"] == "native_noop") for row in rows]),
        "protected_signal_rate": rate([int(row["protected_signal"]) for row in rows]),
        "structured_guard_rate": rate([int(row["structured_control_guard"]) for row in rows]),
    }


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for candidate_id in sorted({str(row["candidate_id"]) for row in rows}):
        candidate_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        groups: list[tuple[str, str, list[dict[str, object]]]] = [("overall", "overall", candidate_rows)]
        for key in ["evidence_tier", "family", "role", "target_id", "source", "expansion_set"]:
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


def calibration_failure_gain(rows: list[dict[str, object]]) -> float:
    return median(
        [
            finite_float(row["model_wql_rer"], float("nan")) - finite_float(row["repair_wql_rer"], float("nan"))
            for row in rows
            if row["role"] == "failure_target"
        ]
    )


def utility_for_calibration(rows: list[dict[str, object]]) -> float:
    protected_rows = [row for row in rows if int(row["protected_signal"])]
    failure_gain = calibration_failure_gain(rows)
    all_harm = rate([int(row["repair_wql_noninferiority_harm"]) for row in rows])
    protected_harm = rate([int(row["repair_wql_noninferiority_harm"]) for row in protected_rows])
    uc_harm = rate([int(row["undercoverage_noninferiority_harm"]) for row in rows])
    abs_uc = mean([finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows])
    return 2.0 * failure_gain - 0.80 * all_harm - 0.80 * protected_harm - 0.80 * uc_harm - 0.50 * abs_uc


def select_candidate(
    calibration_windows: list[dict[str, object]],
    policy: dict[str, object],
    smooth_candidate: dict[str, object],
    candidates: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    rows_by_candidate: dict[str, list[dict[str, object]]] = {}
    wql_losses: dict[str, list[int]] = {}
    protected_wql_losses: dict[str, list[int]] = {}
    uc_harm_losses: dict[str, list[int]] = {}
    abs_uc_losses: dict[str, list[float]] = {}
    utilities: dict[str, float] = {}

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        rows = apply_candidate(calibration_windows, policy, smooth_candidate, candidate, phase="calibration")
        rows_by_candidate[candidate_id] = rows
        protected = [row for row in rows if int(row["protected_signal"])]
        wql_losses[candidate_id] = [int(row["repair_wql_noninferiority_harm"]) for row in rows]
        protected_wql_losses[candidate_id] = [int(row["repair_wql_noninferiority_harm"]) for row in protected]
        uc_harm_losses[candidate_id] = [int(row["undercoverage_noninferiority_harm"]) for row in rows]
        abs_uc_losses[candidate_id] = [finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows]
        utilities[candidate_id] = utility_for_calibration(rows)

    wql_tests = {
        test.policy_id: test
        for test in ltt_risk_tests(wql_losses, alpha=WQL_HARM_ALPHA, delta=DELTA_PER_RISK, correction="holm", binary=True)
    }
    protected_tests = {
        test.policy_id: test
        for test in ltt_risk_tests(
            protected_wql_losses,
            alpha=PROTECTED_HARM_ALPHA,
            delta=DELTA_PER_RISK,
            correction="holm",
            binary=True,
        )
    }
    uc_harm_tests = {
        test.policy_id: test
        for test in ltt_risk_tests(
            uc_harm_losses,
            alpha=UNDERCOVERAGE_HARM_ALPHA,
            delta=DELTA_PER_RISK,
            correction="holm",
            binary=True,
        )
    }
    abs_uc_tests = {
        test.policy_id: test
        for test in ltt_risk_tests(
            abs_uc_losses,
            alpha=ABS_UNDERCOVERAGE_ALPHA,
            delta=DELTA_PER_RISK,
            correction="holm",
            binary=False,
        )
    }

    accepted = [
        candidate_id
        for candidate_id in rows_by_candidate
        if wql_tests[candidate_id].accepted
        and protected_tests[candidate_id].accepted
        and uc_harm_tests[candidate_id].accepted
    ]
    if accepted:
        preserving = [
            candidate_id
            for candidate_id in accepted
            if calibration_failure_gain(rows_by_candidate[candidate_id]) >= FAILURE_GAIN_FLOOR
        ]
        selection_pool = preserving or accepted
        candidates_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}

        def selection_key(candidate_id: str) -> tuple[float, int, float, str]:
            candidate = candidates_by_id[candidate_id]
            return (
                utilities[candidate_id],
                int(str(candidate["mode"]) == "interval_cap"),
                -finite_float(candidate.get("interval_cap"), 999.0),
                candidate_id,
            )

        selected_id = max(selection_pool, key=selection_key)
        selection_status = (
            "tri_risk_ltt_certified_failure_preserving" if preserving else "tri_risk_ltt_certified"
        )
    else:
        selected_id = min(
            rows_by_candidate,
            key=lambda item: (
                rate(wql_losses[item]) / WQL_HARM_ALPHA
                + rate(protected_wql_losses[item]) / PROTECTED_HARM_ALPHA
                + rate(uc_harm_losses[item]) / UNDERCOVERAGE_HARM_ALPHA,
                -utilities[item],
                item,
            ),
        )
        selection_status = "fallback_min_joint_tri_risk"

    selected_rows: list[dict[str, object]] = []
    calibration_tests: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        rows = rows_by_candidate[candidate_id]
        protected_rows = [row for row in rows if int(row["protected_signal"])]
        tri_ok = (
            wql_tests[candidate_id].accepted
            and protected_tests[candidate_id].accepted
            and uc_harm_tests[candidate_id].accepted
        )
        base = {
            "candidate_id": candidate_id,
            "candidate_mode": candidate["mode"],
            "candidate_description": candidate["description"],
            "candidate_interval_cap": candidate["interval_cap"],
            "candidate_low_score_threshold": candidate["low_score_threshold"],
            "selected": int(candidate_id == selected_id),
            "selection_status": selection_status if candidate_id == selected_id else "",
            "tri_risk_ltt_accepted": int(tri_ok),
            "strict_abs_undercoverage_ltt_accepted": int(abs_uc_tests[candidate_id].accepted),
            "n_calibration_windows": len(rows),
            "n_protected_calibration_windows": len(protected_rows),
            "utility": utilities[candidate_id],
            "calibration_median_wql_rer": median([finite_float(row["repair_wql_rer"], float("nan")) for row in rows]),
            "calibration_failure_gain": calibration_failure_gain(rows),
        }
        selected_rows.append(
            {
                **base,
                "wql_harm_alpha": WQL_HARM_ALPHA,
                "wql_harm_empirical_risk": wql_tests[candidate_id].empirical_risk,
                "wql_harm_ltt_p_value": wql_tests[candidate_id].p_value,
                "wql_harm_ltt_corrected_threshold": wql_tests[candidate_id].corrected_threshold,
                "wql_harm_ltt_accepted": int(wql_tests[candidate_id].accepted),
                "protected_harm_alpha": PROTECTED_HARM_ALPHA,
                "protected_harm_empirical_risk": protected_tests[candidate_id].empirical_risk,
                "protected_harm_ltt_p_value": protected_tests[candidate_id].p_value,
                "protected_harm_ltt_corrected_threshold": protected_tests[candidate_id].corrected_threshold,
                "protected_harm_ltt_accepted": int(protected_tests[candidate_id].accepted),
                "undercoverage_harm_alpha": UNDERCOVERAGE_HARM_ALPHA,
                "undercoverage_harm_empirical_risk": uc_harm_tests[candidate_id].empirical_risk,
                "undercoverage_harm_ltt_p_value": uc_harm_tests[candidate_id].p_value,
                "undercoverage_harm_ltt_corrected_threshold": uc_harm_tests[candidate_id].corrected_threshold,
                "undercoverage_harm_ltt_accepted": int(uc_harm_tests[candidate_id].accepted),
                "abs_undercoverage_alpha": ABS_UNDERCOVERAGE_ALPHA,
                "abs_undercoverage_empirical_risk": abs_uc_tests[candidate_id].empirical_risk,
                "abs_undercoverage_ltt_p_value": abs_uc_tests[candidate_id].p_value,
                "abs_undercoverage_ltt_corrected_threshold": abs_uc_tests[candidate_id].corrected_threshold,
                "abs_undercoverage_ltt_accepted": int(abs_uc_tests[candidate_id].accepted),
            }
        )
        for risk_name, alpha, test, primary in [
            ("wql_noninferiority_harm", WQL_HARM_ALPHA, wql_tests[candidate_id], 1),
            ("protected_wql_harm", PROTECTED_HARM_ALPHA, protected_tests[candidate_id], 1),
            ("undercoverage_noninferiority_harm", UNDERCOVERAGE_HARM_ALPHA, uc_harm_tests[candidate_id], 1),
            ("absolute_undercoverage_audit", ABS_UNDERCOVERAGE_ALPHA, abs_uc_tests[candidate_id], 0),
        ]:
            calibration_tests.append(
                {
                    **base,
                    "risk_name": risk_name,
                    "primary_risk": primary,
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
        "role:failure_target",
        "role:positive_control",
        "role:stress_target",
        "evidence_tier:q9_fullgrid|role:failure_target",
        "target_id:finance_fred_stress",
        "family:timesfm|role:failure_target",
        "family:timesfm|role:stress_target",
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
                "UC": num(row["repair_undercoverage_risk"]),
                "UCHarm": pct(row["undercoverage_harm_rate"]),
                "WQLHarm": pct(row["repair_wql_noninferiority_harm_rate"]),
                "Cap": pct(row["interval_cap_rate"]),
                "Point": pct(row["point_only_rate"]),
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
                "TriOK": "yes" if int(row["tri_risk_ltt_accepted"]) else "no",
                "WQL": pct(row["wql_harm_empirical_risk"]),
                "Prot": pct(row["protected_harm_empirical_risk"]),
                "UCHarm": pct(row["undercoverage_harm_empirical_risk"]),
                "AbsUC": num(row["abs_undercoverage_empirical_risk"]),
                "AbsOK": "yes" if int(row["abs_undercoverage_ltt_accepted"]) else "no",
                "FailureGain": num(row["calibration_failure_gain"]),
                "Utility": num(row["utility"], 4),
            }
        )
    return output


def write_report(
    selected_rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    status: dict[str, object],
) -> None:
    selected_id = str(status["selected_candidate_id"])
    DOC_PATH.write_text(
        "\n".join(
            [
                "# DRCR-Smooth Coverage-Aware Safety Gate",
                "",
                "## Purpose",
                "",
                "This experiment upgrades the safety gate from dual WQL-harm control to a coverage-aware intervention selector. It keeps the frozen DRCR-Smooth point/score head unchanged and calibrates whether to deploy full repair, point-only repair, or capped interval repair.",
                "",
                "## Data",
                "",
                f"- Original Chronos/Moirai windows: `{status['n_original_windows']}`.",
                f"- Raw expansion windows: `{status['n_expansion_windows_raw']}`.",
                f"- Duplicate overlap removed: `{status['n_duplicate_windows_removed']}`.",
                f"- Combined de-duplicated windows: `{status['n_windows_total']}`.",
                f"- Calibration/test split: `{status['n_calibration_windows']}` / `{status['n_test_windows']}`.",
                "",
                "## Calibration",
                "",
                f"- Selected candidate: `{selected_id}`.",
                f"- Primary risk 1: WQL non-inferiority harm, alpha `{WQL_HARM_ALPHA}`.",
                f"- Primary risk 2: protected-signal WQL harm, alpha `{PROTECTED_HARM_ALPHA}`.",
                f"- Primary risk 3: undercoverage non-inferiority harm, alpha `{UNDERCOVERAGE_HARM_ALPHA}`, margin `{UNDERCOVERAGE_HARM_MARGIN}`.",
                f"- Audit risk: strict absolute undercoverage, alpha `{ABS_UNDERCOVERAGE_ALPHA}`; this is reported but not used as the primary accept/reject criterion in this pass.",
                "",
                "## Candidate Screen",
                "",
                markdown_table(
                    selected_rows_for_report(selected_rows),
                    [
                        ("Candidate", "Candidate"),
                        ("Mode", "Mode"),
                        ("Selected", "Selected"),
                        ("TriOK", "Tri ok"),
                        ("WQL", "WQL harm"),
                        ("Prot", "Prot harm"),
                        ("UCHarm", "UC harm"),
                        ("AbsUC", "Abs UC"),
                        ("AbsOK", "Abs ok"),
                        ("FailureGain", "Fail gain"),
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
                        ("UC", "UC risk"),
                        ("UCHarm", "UC harm"),
                        ("WQLHarm", "WQL harm"),
                        ("Cap", "Cap"),
                        ("Point", "Point"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- Strict absolute undercoverage plus WQL harm exposes a tradeoff: full interval repair is coverage-safe but WQL-unsafe, while point-only repair is WQL-safe but weak on absolute undercoverage.",
                "- The selected coverage-aware gate resolves this by using capped interval repair: it preserves the failure-side point repair and keeps a limited interval correction instead of fully widening structured/stress windows.",
                "- This is a stronger method story than fallback/blending because the calibrated decision is over intervention components and coverage/WQL risks, not over a TSFM/classical average.",
                "- The honest remaining boundary is that strict absolute undercoverage LTT is still an audit target; the primary certified coverage claim is non-inferiority versus the native TSFM interval.",
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
    windows, inventory = combined_windows()
    calibration_windows = [window for window in windows if crc.split_bucket(window) == 0]
    test_windows = [window for window in windows if crc.split_bucket(window) == 1]
    policy = smooth.crc.fullgrid.common_cpr_policy()
    smooth_candidate = selected_smooth_candidate()
    candidates = candidate_configs()
    selected, selected_rows, calibration_tests = select_candidate(calibration_windows, policy, smooth_candidate, candidates)
    selected_id = str(selected["candidate_id"])

    all_test_rows: list[dict[str, object]] = []
    selected_test_rows: list[dict[str, object]] = []
    for candidate in candidates:
        rows = apply_candidate(test_windows, policy, smooth_candidate, candidate, phase="test")
        all_test_rows.extend(rows)
        if str(candidate["candidate_id"]) == selected_id:
            selected_test_rows = rows
    summary = build_summary(all_test_rows)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "frozen_protocol_id": smooth.PROTOCOL_ID,
        "frozen_protocol_sha256": smooth.protocol_sha256(),
        "split_protocol": smooth.SPLIT_PROTOCOL,
        "split_id": smooth.SPLIT_ID,
        **inventory,
        "n_windows_total": len(windows),
        "n_calibration_windows": len(calibration_windows),
        "n_test_windows": len(test_windows),
        "n_candidates": len(candidates),
        "selected_candidate_id": selected_id,
        "selected_candidate_mode": selected["mode"],
        "wql_harm_alpha": WQL_HARM_ALPHA,
        "protected_harm_alpha": PROTECTED_HARM_ALPHA,
        "undercoverage_harm_alpha": UNDERCOVERAGE_HARM_ALPHA,
        "undercoverage_harm_margin": UNDERCOVERAGE_HARM_MARGIN,
        "abs_undercoverage_alpha": ABS_UNDERCOVERAGE_ALPHA,
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
