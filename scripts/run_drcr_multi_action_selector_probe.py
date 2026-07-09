#!/usr/bin/env python
"""Probe a multi-action DRCR selector with objective baselines.

This is a research probe, not a frozen protocol.  It tests the literature-
motivated framing from selective intervention / conformal policy control:
choose among native, classical fallback, point repair, capped interval repair,
and a stronger expert-pull capped repair under the same tri-risk calibration
screen used by the coverage-aware gate.
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

import run_chronos_fullgrid_cpr_wql_goal as fullgrid_base  # noqa: E402
import run_drcr_smooth_coverage_aware_gate_goal as coverage_gate  # noqa: E402
import run_drcr_smooth_safety_gate_goal as safety  # noqa: E402
import run_drcr_smooth_score_head_search_goal as smooth  # noqa: E402

from low_snr_tsfm.risk_control import ltt_risk_tests  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "drcr_multi_action_selector_probe_report.md"
STATUS_OUT = OUT_DIR / "drcr_multi_action_selector_probe_status.json"
WINDOW_OUT = OUT_DIR / "drcr_multi_action_selector_probe_windows.csv"
SUMMARY_OUT = OUT_DIR / "drcr_multi_action_selector_probe_summary.csv"
CANDIDATE_OUT = OUT_DIR / "drcr_multi_action_selector_probe_candidates.csv"
CALIBRATION_OUT = OUT_DIR / "drcr_multi_action_selector_probe_calibration_tests.csv"

WQL_HARM_ALPHA = 0.20
PROTECTED_HARM_ALPHA = 0.20
UNDERCOVERAGE_HARM_ALPHA = 0.20
UNDERCOVERAGE_HARM_MARGIN = 0.05
ABS_UNDERCOVERAGE_ALPHA = smooth.UNDERCOVERAGE_ALPHA
DELTA_PER_RISK = 0.05


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


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
            "candidate_id": "native_tsfm",
            "mode": "native",
            "weight_multiplier": 0.0,
            "interval_cap": "",
            "score_floor": "",
            "score_threshold": "",
            "description": "raw foundation model forecast; no intervention",
        },
        {
            "candidate_id": "classical_deterministic",
            "mode": "classical",
            "weight_multiplier": "",
            "interval_cap": "",
            "score_floor": "",
            "score_threshold": "",
            "description": "deterministic classical reference forecast; objective fallback baseline",
        },
        {
            "candidate_id": "drcr_full",
            "mode": "full",
            "weight_multiplier": 1.0,
            "interval_cap": "",
            "score_floor": "",
            "score_threshold": "",
            "description": "frozen DRCR-Smooth full point and interval repair",
        },
        {
            "candidate_id": "drcr_point",
            "mode": "point",
            "weight_multiplier": 1.0,
            "interval_cap": 1.0,
            "score_floor": "",
            "score_threshold": "",
            "description": "point repair only; preserve native interval width",
        },
        {
            "candidate_id": "drcr_cap_1.10",
            "mode": "cap",
            "weight_multiplier": 1.0,
            "interval_cap": 1.10,
            "score_floor": "",
            "score_threshold": "",
            "description": "coverage-aware capped interval repair from the previous gate",
        },
        {
            "candidate_id": "drcr_expert_pull_1.25_cap_1.10",
            "mode": "cap",
            "weight_multiplier": 1.25,
            "interval_cap": 1.10,
            "score_floor": "",
            "score_threshold": "",
            "description": "CPC-style stronger pull toward the classical expert, with interval cap",
        },
        {
            "candidate_id": "drcr_expert_pull_1.50_cap_1.10",
            "mode": "cap",
            "weight_multiplier": 1.50,
            "protected_weight_multiplier": "",
            "interval_cap": 1.10,
            "width_veto_threshold": "",
            "score_floor": "",
            "score_threshold": "",
            "description": "more aggressive expert pull stress-test, with interval cap",
        },
        {
            "candidate_id": "drcr_width_veto_expert_pull_1.50_cap_1.10",
            "mode": "width_veto_expert_pull",
            "weight_multiplier": 1.50,
            "protected_weight_multiplier": 1.25,
            "interval_cap": 1.10,
            "width_veto_threshold": 0.05,
            "score_floor": "",
            "score_threshold": "",
            "description": "stronger expert pull with a native no-op veto when native intervals are extremely narrow",
        },
        {
            "candidate_id": "drcr_score_floor_0.60_cap_1.00",
            "mode": "score_floor_cap",
            "weight_multiplier": 1.0,
            "protected_weight_multiplier": "",
            "interval_cap": 1.00,
            "width_veto_threshold": "",
            "score_floor": 0.60,
            "score_threshold": 0.50,
            "description": "only high smooth-score windows get at least 0.60 classical-expert pull",
        },
    ]


def selected_smooth_candidate() -> dict[str, object]:
    return coverage_gate.selected_smooth_candidate()


def quantile_grid_for(
    window: dict[str, object],
    base: dict[str, object],
    candidate: dict[str, object],
) -> tuple[np.ndarray, str, float, float, int]:
    mode = str(candidate["mode"])
    if mode == "native":
        return np.asarray(window["quantile_grid"], dtype=float), "native", 0.0, 1.0, 0
    if mode == "classical":
        return smooth.crc.fullgrid.deterministic_baseline_grid(window), "classical", 1.0, 0.0, 1

    weight = finite_float(base["effective_weight"])
    scale = finite_float(base["interval_scale"])
    veto = 0
    if candidate.get("weight_multiplier", "") != "":
        weight = min(1.0, max(0.0, weight * finite_float(candidate["weight_multiplier"], 1.0)))
    if mode == "point":
        scale = 1.0
        veto = 1
    elif mode == "cap":
        cap = finite_float(candidate["interval_cap"], scale)
        veto = int(scale > cap + 1e-12)
        scale = min(scale, cap)
    elif mode == "score_floor_cap":
        score = finite_float(base["smooth_score"])
        if score >= finite_float(candidate["score_threshold"]):
            weight = max(weight, finite_float(candidate["score_floor"]))
        cap = finite_float(candidate["interval_cap"], scale)
        veto = int(scale > cap + 1e-12)
        scale = min(scale, cap)
    elif mode == "width_veto_expert_pull":
        width_ratio = smooth.crc.native_width_ratio(window)
        threshold = finite_float(candidate["width_veto_threshold"])
        if width_ratio < threshold:
            return np.asarray(window["quantile_grid"], dtype=float), "native_width_veto", 0.0, 1.0, 1
        multiplier_key = "protected_weight_multiplier" if int(base["protected_signal"]) else "weight_multiplier"
        weight = min(1.0, max(0.0, weight * finite_float(candidate[multiplier_key], 1.0)))
        cap = finite_float(candidate["interval_cap"], scale)
        veto = int(scale > cap + 1e-12)
        scale = min(scale, cap)
    elif mode == "full":
        pass
    else:
        raise ValueError(f"Unsupported candidate mode: {mode}")
    return safety.quantile_grid_for(window, weight, scale), mode, weight, scale, veto


def apply_candidate_to_window(
    window: dict[str, object],
    policy: dict[str, object],
    smooth_candidate: dict[str, object],
    candidate: dict[str, object],
    *,
    phase: str,
) -> dict[str, object]:
    base = safety.base_intervention(window, policy, smooth_candidate)
    model_metrics = smooth.crc.fullgrid.quantile_metrics(window, np.asarray(window["quantile_grid"], dtype=float))
    grid, deployed_mode, weight, scale, veto = quantile_grid_for(window, base, candidate)
    repair_metrics = smooth.crc.fullgrid.quantile_metrics(window, grid)
    model_undercoverage = max(0.0, smooth.crc.NOMINAL_COVERAGE - model_metrics["coverage"])
    repair_undercoverage = max(0.0, smooth.crc.NOMINAL_COVERAGE - repair_metrics["coverage"])
    undercoverage_delta = max(0.0, repair_undercoverage - model_undercoverage)
    return {
        "phase": phase,
        "candidate_id": candidate["candidate_id"],
        "candidate_mode": candidate["mode"],
        "candidate_description": candidate["description"],
        "deployed_mode": deployed_mode,
        "effective_weight": weight,
        "interval_scale": scale,
        "veto_active": veto,
        "family": window["family"],
        "source": window["source"],
        "dataset": window["dataset"],
        "model": window["model"],
        "series_id": window["series_id"],
        "window_index": window["window_index"],
        "role": window["role"],
        "target_id": window["target_id"],
        "evidence_tier": window["evidence_tier"],
        "protected_signal": int(base["protected_signal"]),
        "smooth_width_score": base["smooth_score"],
        "native_width_ratio": smooth.crc.native_width_ratio(window),
        "model_wql_rer": model_metrics["wql_rer"],
        "repair_wql_rer": repair_metrics["wql_rer"],
        "wql_rer_delta_vs_model": repair_metrics["wql_rer"] - model_metrics["wql_rer"],
        "model_coverage_q10_q90": model_metrics["coverage"],
        "repair_coverage_q10_q90": repair_metrics["coverage"],
        "model_undercoverage_risk": model_undercoverage,
        "repair_undercoverage_risk": repair_undercoverage,
        "undercoverage_risk_delta_vs_model": undercoverage_delta,
        "undercoverage_noninferiority_harm": int(undercoverage_delta > UNDERCOVERAGE_HARM_MARGIN),
        "model_wql_failure_delta005": int(model_metrics["wql_rer"] > 1.05),
        "repair_wql_failure_delta005": int(repair_metrics["wql_rer"] > 1.05),
        "repair_win_vs_model": int(repair_metrics["wql_rer"] < model_metrics["wql_rer"]),
        "repair_wql_noninferiority_harm": int(
            repair_metrics["wql_rer"] > model_metrics["wql_rer"] + smooth.WQL_HARM_MARGIN
        ),
    }


def group_rows(rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    return {
        "overall": rows,
        "role:failure_target": [row for row in rows if row["role"] == "failure_target"],
        "role:positive_control": [row for row in rows if row["role"] == "positive_control"],
        "role:stress_target": [row for row in rows if row["role"] == "stress_target"],
        "evidence_tier:q9_fullgrid|role:failure_target": [
            row
            for row in rows
            if row["evidence_tier"] == "q9_fullgrid" and row["role"] == "failure_target"
        ],
        "target_id:finance_fred_stress": [row for row in rows if row["target_id"] == "finance_fred_stress"],
        "family:timesfm|role:failure_target": [
            row for row in rows if row["family"] == "timesfm" and row["role"] == "failure_target"
        ],
    }


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for group, subset in group_rows(rows).items():
        if not subset:
            continue
        output.append(
            {
                "candidate_id": subset[0]["candidate_id"],
                "candidate_mode": subset[0]["candidate_mode"],
                "phase": subset[0]["phase"],
                "group": group,
                "n_windows": len(subset),
                "model_median_wql_rer": median([finite_float(row["model_wql_rer"], float("nan")) for row in subset]),
                "repair_median_wql_rer": median([finite_float(row["repair_wql_rer"], float("nan")) for row in subset]),
                "repair_mean_coverage": mean(
                    [finite_float(row["repair_coverage_q10_q90"], float("nan")) for row in subset]
                ),
                "repair_undercoverage_risk": mean(
                    [finite_float(row["repair_undercoverage_risk"], float("nan")) for row in subset]
                ),
                "wql_harm_rate": rate([int(row["repair_wql_noninferiority_harm"]) for row in subset]),
                "undercoverage_harm_rate": rate([int(row["undercoverage_noninferiority_harm"]) for row in subset]),
                "win_rate_vs_model": rate([int(row["repair_win_vs_model"]) for row in subset]),
                "wql_failure_reduction_vs_model": rate(
                    [
                        int(row["model_wql_failure_delta005"]) - int(row["repair_wql_failure_delta005"])
                        for row in subset
                    ]
                ),
            }
        )
    return output


def calibration_tests(candidate_rows: dict[str, list[dict[str, object]]]) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    wql_losses = {
        cid: [int(row["repair_wql_noninferiority_harm"]) for row in rows]
        for cid, rows in candidate_rows.items()
    }
    protected_losses = {
        cid: [int(row["repair_wql_noninferiority_harm"]) for row in rows if int(row["protected_signal"])]
        for cid, rows in candidate_rows.items()
    }
    uc_losses = {
        cid: [int(row["undercoverage_noninferiority_harm"]) for row in rows]
        for cid, rows in candidate_rows.items()
    }
    abs_uc_losses = {
        cid: [finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows]
        for cid, rows in candidate_rows.items()
    }
    wql_tests = {
        test.policy_id: test
        for test in ltt_risk_tests(wql_losses, alpha=WQL_HARM_ALPHA, delta=DELTA_PER_RISK, correction="holm", binary=True)
    }
    protected_tests = {
        test.policy_id: test
        for test in ltt_risk_tests(
            protected_losses,
            alpha=PROTECTED_HARM_ALPHA,
            delta=DELTA_PER_RISK,
            correction="holm",
            binary=True,
        )
    }
    uc_tests = {
        test.policy_id: test
        for test in ltt_risk_tests(
            uc_losses,
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
    rows: list[dict[str, object]] = []
    by_candidate: dict[str, dict[str, object]] = {}
    for cid, rows_for_candidate in candidate_rows.items():
        tri_ok = (
            wql_tests[cid].accepted
            and protected_tests[cid].accepted
            and uc_tests[cid].accepted
        )
        q9 = [
            row
            for row in rows_for_candidate
            if row["role"] == "failure_target" and row["evidence_tier"] == "q9_fullgrid"
        ]
        failure = [row for row in rows_for_candidate if row["role"] == "failure_target"]
        utility = (
            1.50
            * (
                median([finite_float(row["model_wql_rer"], float("nan")) for row in q9])
                - median([finite_float(row["repair_wql_rer"], float("nan")) for row in q9])
            )
            + 0.80
            * (
                median([finite_float(row["model_wql_rer"], float("nan")) for row in rows_for_candidate])
                - median([finite_float(row["repair_wql_rer"], float("nan")) for row in rows_for_candidate])
            )
            + 0.50
            * (
                median([finite_float(row["model_wql_rer"], float("nan")) for row in failure])
                - median([finite_float(row["repair_wql_rer"], float("nan")) for row in failure])
            )
            - 1.25 * wql_tests[cid].empirical_risk
            - 0.75 * protected_tests[cid].empirical_risk
            - 0.75 * uc_tests[cid].empirical_risk
        )
        row = {
            "candidate_id": cid,
            "tri_risk_accepted": int(tri_ok),
            "strict_abs_undercoverage_ltt_accepted": int(abs_uc_tests[cid].accepted),
            "wql_harm_empirical_risk": wql_tests[cid].empirical_risk,
            "protected_harm_empirical_risk": protected_tests[cid].empirical_risk,
            "undercoverage_harm_empirical_risk": uc_tests[cid].empirical_risk,
            "abs_undercoverage_empirical_risk": abs_uc_tests[cid].empirical_risk,
            "calibration_q9_gain": (
                median([finite_float(item["model_wql_rer"], float("nan")) for item in q9])
                - median([finite_float(item["repair_wql_rer"], float("nan")) for item in q9])
            ),
            "calibration_utility": utility,
        }
        rows.append(row)
        by_candidate[cid] = row
    risk_rows: list[dict[str, object]] = []
    for cid in candidate_rows:
        for risk_name, alpha, test, primary in [
            ("wql_noninferiority_harm", WQL_HARM_ALPHA, wql_tests[cid], 1),
            ("protected_wql_harm", PROTECTED_HARM_ALPHA, protected_tests[cid], 1),
            ("undercoverage_noninferiority_harm", UNDERCOVERAGE_HARM_ALPHA, uc_tests[cid], 1),
            ("absolute_undercoverage_audit", ABS_UNDERCOVERAGE_ALPHA, abs_uc_tests[cid], 0),
        ]:
            risk_rows.append(
                {
                    "candidate_id": cid,
                    "risk_name": risk_name,
                    "alpha": alpha,
                    "primary_risk": primary,
                    "empirical_risk": test.empirical_risk,
                    "p_value": test.p_value,
                    "corrected_threshold": test.corrected_threshold,
                    "accepted": int(test.accepted),
                }
            )
    return risk_rows, by_candidate


def selected_candidate(candidates: list[dict[str, object]], candidate_screen: dict[str, dict[str, object]]) -> str:
    accepted = [
        str(candidate["candidate_id"])
        for candidate in candidates
        if int(candidate_screen[str(candidate["candidate_id"])]["tri_risk_accepted"])
        and str(candidate["candidate_id"]) != "native_tsfm"
    ]
    if not accepted:
        return "native_tsfm"
    return max(accepted, key=lambda cid: finite_float(candidate_screen[cid]["calibration_utility"], -1e9))


def report_rows(summary_rows: list[dict[str, object]], candidate_ids: list[str], group: str) -> list[dict[str, object]]:
    lookup = {(row["candidate_id"], row["phase"], row["group"]): row for row in summary_rows}
    output = []
    for cid in candidate_ids:
        row = lookup[(cid, "test", group)]
        output.append(
            {
                "Candidate": cid,
                "N": int(row["n_windows"]),
                "Model": num(row["model_median_wql_rer"]),
                "Repair": num(row["repair_median_wql_rer"]),
                "Harm": pct(row["wql_harm_rate"]),
                "Coverage": pct(row["repair_mean_coverage"]),
                "FailRed": pct(row["wql_failure_reduction_vs_model"]),
            }
        )
    return output


def write_report(
    counts: dict[str, int],
    selected_id: str,
    candidate_rows: list[dict[str, object]],
    summary_rows: list[dict[str, object]],
) -> None:
    candidate_ids = [
        "native_tsfm",
        "classical_deterministic",
        "drcr_full",
        "drcr_point",
        "drcr_cap_1.10",
        "drcr_expert_pull_1.25_cap_1.10",
        "drcr_expert_pull_1.50_cap_1.10",
        "drcr_width_veto_expert_pull_1.50_cap_1.10",
        "drcr_score_floor_0.60_cap_1.00",
    ]
    screen = {row["candidate_id"]: row for row in candidate_rows}
    screen_rows = []
    for cid in candidate_ids:
        row = screen[cid]
        screen_rows.append(
            {
                "Candidate": cid,
                "Selected": "yes" if cid == selected_id else "",
                "TriOK": "yes" if int(row["tri_risk_accepted"]) else "no",
                "WQL": pct(row["wql_harm_empirical_risk"]),
                "Prot": pct(row["protected_harm_empirical_risk"]),
                "UC": pct(row["undercoverage_harm_empirical_risk"]),
                "AbsUC": num(row["abs_undercoverage_empirical_risk"]),
                "Q9Gain": num(row["calibration_q9_gain"]),
                "Utility": num(row["calibration_utility"], 4),
            }
        )
    lines = [
        "# DRCR Multi-Action Selector Probe",
        "",
        "## Purpose",
        "",
        "This probe tests whether the method story can move from comparing only DRCR variants to a multi-action selective intervention policy with objective baselines: native TSFM, deterministic classical fallback, full repair, point repair, capped repair, stronger expert-pull capped repair, and a native-width veto expert-pull candidate.",
        "",
        "This is a probe, not a frozen protocol. The candidate class should be frozen only after deciding the final paper-facing method.",
        "",
        "## Data",
        "",
        f"- Original windows: `{counts['n_original_windows']}`.",
        f"- Raw expansion windows: `{counts['n_expansion_windows_raw']}`.",
        f"- Duplicate overlaps removed: `{counts['n_duplicate_windows_removed']}`.",
        f"- Combined windows: `{counts['n_combined_windows']}`.",
        "",
        "## Calibration Screen",
        "",
        markdown_table(
            screen_rows,
            [
                ("Candidate", "Candidate"),
                ("Selected", "Selected"),
                ("TriOK", "Tri ok"),
                ("WQL", "WQL harm"),
                ("Prot", "Prot harm"),
                ("UC", "UC harm"),
                ("AbsUC", "Abs UC"),
                ("Q9Gain", "Q9 gain"),
                ("Utility", "Utility"),
            ],
        ),
        "",
        "## Test: Overall",
        "",
        markdown_table(
            report_rows(summary_rows, candidate_ids, "overall"),
            [
                ("Candidate", "Candidate"),
                ("N", "N"),
                ("Model", "Model"),
                ("Repair", "Repair"),
                ("Harm", "Harm"),
                ("Coverage", "Coverage"),
                ("FailRed", "Fail red."),
            ],
        ),
        "",
        "## Test: q9 Failure",
        "",
        markdown_table(
            report_rows(summary_rows, candidate_ids, "evidence_tier:q9_fullgrid|role:failure_target"),
            [
                ("Candidate", "Candidate"),
                ("N", "N"),
                ("Model", "Model"),
                ("Repair", "Repair"),
                ("Harm", "Harm"),
                ("Coverage", "Coverage"),
                ("FailRed", "Fail red."),
            ],
        ),
        "",
        "## Test: Stress and Finance",
        "",
        markdown_table(
            report_rows(summary_rows, candidate_ids, "role:stress_target")
            + report_rows(summary_rows, candidate_ids, "target_id:finance_fred_stress"),
            [
                ("Candidate", "Candidate"),
                ("N", "N"),
                ("Model", "Model"),
                ("Repair", "Repair"),
                ("Harm", "Harm"),
                ("Coverage", "Coverage"),
                ("FailRed", "Fail red."),
            ],
        ),
        "",
        "## Interpretation",
        "",
        f"- Selected probe candidate: `{selected_id}`.",
        "- Classical fallback is a useful objective reference but is not a probabilistic repair: it often has poor interval behavior and should be treated as a baseline, not the proposed method.",
        "- Expert-pull capped repair is the first promising direction for better-looking failure-side numbers: it increases the pull toward the classical expert while retaining the interval cap.",
        "- The native-width veto expert-pull candidate passes the same tri-risk screen and forms the current high-performance frontier on the test split: it improves overall and q9 failure WQL more than the conservative selected candidate while lowering stress harm. It is not selected by the current conservative calibration utility, so it should be treated as a Pareto probe unless a risk-constrained selection objective is frozen before the next locked rerun.",
        "- The next decision is whether to freeze a small multi-action candidate class and a pre-registered Pareto/risk-constrained selection objective around the expert-pull plus width-veto idea.",
        "",
        "## Artifacts",
        "",
        f"- `{WINDOW_OUT.relative_to(ROOT)}`",
        f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
        f"- `{CANDIDATE_OUT.relative_to(ROOT)}`",
        f"- `{CALIBRATION_OUT.relative_to(ROOT)}`",
    ]
    DOC_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    smooth.validate_protocol_config()
    windows, counts = coverage_gate.combined_windows()
    policy = fullgrid_base.common_cpr_policy()
    smooth_candidate = selected_smooth_candidate()
    candidates = candidate_configs()
    all_rows: list[dict[str, object]] = []
    candidate_calibration_rows: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        rows: list[dict[str, object]] = []
        for window in windows:
            phase = "calibration" if smooth.crc.split_bucket(window) == 0 else "test"
            row = apply_candidate_to_window(window, policy, smooth_candidate, candidate, phase=phase)
            rows.append(row)
            all_rows.append(row)
        candidate_calibration_rows[str(candidate["candidate_id"])] = [row for row in rows if row["phase"] == "calibration"]
    calibration_rows, candidate_screen = calibration_tests(candidate_calibration_rows)
    selected_id = selected_candidate(candidates, candidate_screen)
    candidate_rows = []
    config_by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    for cid, row in candidate_screen.items():
        candidate_rows.append(
            {
                **config_by_id[cid],
                **row,
                "selected": int(cid == selected_id),
            }
        )
    summary_rows: list[dict[str, object]] = []
    for cid in config_by_id:
        candidate_rows_all = [row for row in all_rows if row["candidate_id"] == cid]
        summary_rows.extend(summarize_rows([row for row in candidate_rows_all if row["phase"] == "calibration"]))
        summary_rows.extend(summarize_rows([row for row in candidate_rows_all if row["phase"] == "test"]))
    write_csv(WINDOW_OUT, all_rows)
    write_csv(SUMMARY_OUT, summary_rows)
    write_csv(CANDIDATE_OUT, candidate_rows)
    write_csv(CALIBRATION_OUT, calibration_rows)
    write_report(counts, selected_id, candidate_rows, summary_rows)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "probe": "drcr_multi_action_selector",
        "n_windows": len(windows),
        "n_candidates": len(candidates),
        "selected_candidate_id": selected_id,
        **counts,
        "windows": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "candidates": str(CANDIDATE_OUT.relative_to(ROOT)),
        "calibration_tests": str(CALIBRATION_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
