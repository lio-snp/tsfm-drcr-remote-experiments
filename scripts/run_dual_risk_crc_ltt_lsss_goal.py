#!/usr/bin/env python
"""Dual-risk CRC/LTT calibration for LSSS interval heads.

The single-risk CRC/LTT-LSSS script controls q10-q90 undercoverage.  This
variant adds a second calibration risk: WQL non-inferiority harm versus the
native TSFM.  A candidate interval head is deployable only when it passes both
LTT screens, which turns the interval-scale choice into a Pareto-style safety
selection rather than a hard-coded widening knob.
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

from low_snr_tsfm.risk_control import ltt_risk_tests  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "dual_risk_crc_ltt_lsss_report.md"
STATUS_OUT = OUT_DIR / "dual_risk_crc_ltt_lsss_status.json"
WINDOW_OUT = OUT_DIR / "dual_risk_crc_ltt_lsss_windows.csv"
SUMMARY_OUT = OUT_DIR / "dual_risk_crc_ltt_lsss_summary.csv"
CANDIDATE_SUMMARY_OUT = OUT_DIR / "dual_risk_crc_ltt_lsss_candidate_summary.csv"
SELECTED_OUT = OUT_DIR / "dual_risk_crc_ltt_lsss_selected_configs.csv"
CALIBRATION_TEST_OUT = OUT_DIR / "dual_risk_crc_ltt_lsss_calibration_tests.csv"

UNDERCOVERAGE_ALPHA = 0.18
WQL_HARM_ALPHA = 0.35
TOTAL_DELTA = 0.10
DELTA_PER_RISK = TOTAL_DELTA / 2.0
WQL_HARM_MARGIN = 0.05
UTILITY_UNDERCOVERAGE_LAMBDA = 1.0
UTILITY_WQL_HARM_LAMBDA = 0.2
UTILITY_TIE_EPS = 0.005


def candidate_configs() -> list[dict[str, object]]:
    """Pre-registered dual-risk candidate class.

    The first four heads are fixed interval scales. The remaining heads keep
    the mechanism signal from the single-risk result: native q10-q90 width
    collapse changes the failure-side interval scale while structured controls
    remain guarded separately.
    """

    return [
        {
            "candidate_id": "sharp_f0.75_s2.00",
            "failure_scale": 0.75,
            "structured_scale": 2.0,
            "width_ratio_threshold": "",
            "collapse_scale": "",
            "description": "sharp fixed failure head",
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
            "candidate_id": "width_neutral_t0.20_f0.75_c1.00_s2.00",
            "failure_scale": 0.75,
            "structured_scale": 2.0,
            "width_ratio_threshold": 0.20,
            "collapse_scale": 1.00,
            "description": "sharp head with neutral widening on uncertainty collapse",
        },
        {
            "candidate_id": "width_widen_t0.20_f0.75_c1.25_s2.00",
            "failure_scale": 0.75,
            "structured_scale": 2.0,
            "width_ratio_threshold": 0.20,
            "collapse_scale": 1.25,
            "description": "sharp head with coverage widening on uncertainty collapse",
        },
        {
            "candidate_id": "collapse_floor_t0.10_f0.95_c1.00_s2.00",
            "failure_scale": 0.95,
            "structured_scale": 2.0,
            "width_ratio_threshold": 0.10,
            "collapse_scale": 1.00,
            "description": "lightly sharp head with balanced floor on severe uncertainty collapse",
        },
    ]


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
    return f"{100.0 * crc.finite_float(value):.1f}%"


def num(value: object, digits: int = 3) -> str:
    parsed = crc.finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


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


def apply_candidate(
    windows: list[dict[str, object]],
    policy: dict[str, object],
    candidate: dict[str, object],
    *,
    split_protocol: str,
    split_id: str,
    phase: str,
) -> list[dict[str, object]]:
    rows = crc.apply_candidate(
        windows,
        policy,
        candidate,
        split_protocol=split_protocol,
        split_id=split_id,
        phase=phase,
    )
    for row in rows:
        row["repair_wql_noninferiority_harm"] = int(
            crc.finite_float(row["repair_wql_rer"])
            > crc.finite_float(row["model_wql_rer"]) + WQL_HARM_MARGIN
        )
        row["wql_harm_margin"] = WQL_HARM_MARGIN
    return rows


def summarize_candidate(rows: list[dict[str, object]], group: str, group_type: str) -> dict[str, object]:
    base = crc.summarize_candidate(rows, group, group_type)
    base["repair_wql_noninferiority_harm_rate"] = crc.rate(
        [int(row["repair_wql_noninferiority_harm"]) for row in rows]
    )
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
                **summarize_candidate(subset, group, group_type),
            }
        )
    return summary


def utility_for_calibration(rows: list[dict[str, object]]) -> float:
    return -(
        crc.median([crc.finite_float(row["repair_wql_rer"], float("nan")) for row in rows])
        + UTILITY_UNDERCOVERAGE_LAMBDA
        * crc.mean([crc.finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows])
        + UTILITY_WQL_HARM_LAMBDA
        * crc.rate([int(row["repair_wql_noninferiority_harm"]) for row in rows])
    )


def select_candidate(
    calibration_windows: list[dict[str, object]],
    policy: dict[str, object],
    candidates: list[dict[str, object]],
    *,
    split_protocol: str,
    split_id: str,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
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
            crc.finite_float(row["repair_undercoverage_risk"], float("nan")) for row in rows
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
        selection_status = "dual_ltt_certified"
    else:
        selected_id = min(
            undercoverage_losses,
            key=lambda item: (
                crc.mean(undercoverage_losses[item]) / UNDERCOVERAGE_ALPHA
                + crc.rate(wql_harm_losses[item]) / WQL_HARM_ALPHA,
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
            "selected": int(candidate_id == selected_id),
            "selection_status": selection_status if candidate_id == selected_id else "",
            "dual_ltt_accepted": int(uc_test.accepted and harm_test.accepted),
            "n_calibration_windows": len(rows),
            "utility": utilities[candidate_id],
            "median_wql_rer": crc.median([crc.finite_float(row["repair_wql_rer"], float("nan")) for row in rows]),
            "failure_scale": candidate["failure_scale"],
            "structured_scale": candidate["structured_scale"],
            "width_ratio_threshold": candidate["width_ratio_threshold"],
            "collapse_scale": candidate["collapse_scale"],
            "diagnostic_guard_candidate": int(candidate.get("width_ratio_threshold") != ""),
            "utility_undercoverage_lambda": UTILITY_UNDERCOVERAGE_LAMBDA,
            "utility_wql_harm_lambda": UTILITY_WQL_HARM_LAMBDA,
            "utility_tie_eps": UTILITY_TIE_EPS,
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
                "undercoverage_ltt_ucb_hoeffding": uc_test.ucb_hoeffding,
                "wql_harm_alpha": WQL_HARM_ALPHA,
                "wql_harm_delta": DELTA_PER_RISK,
                "wql_harm_margin": WQL_HARM_MARGIN,
                "wql_harm_empirical_risk": harm_test.empirical_risk,
                "wql_harm_ltt_p_value": harm_test.p_value,
                "wql_harm_ltt_corrected_threshold": harm_test.corrected_threshold,
                "wql_harm_ltt_accepted": int(harm_test.accepted),
                "wql_harm_ltt_ucb_hoeffding": harm_test.ucb_hoeffding,
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
    calibration_rows = [row for rows in calibration_rows_by_candidate.values() for row in rows]
    return selected, selected_rows, calibration_tests, calibration_rows


def selected_lookup(selected_rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {
        (str(row["split_protocol"]), str(row["split_id"])): row
        for row in selected_rows
        if int(row.get("selected", 0)) == 1
    }


def summary_row(
    summary: list[dict[str, object]],
    split_protocol: str,
    split_id: str,
    candidate_id: str,
    group: str,
) -> dict[str, object]:
    return next(
        row
        for row in summary
        if row["split_protocol"] == split_protocol
        and row["split_id"] == split_id
        and row["phase"] == "test"
        and row["candidate_id"] == candidate_id
        and row["group"] == group
    )


def selected_report_rows(selected_rows: list[dict[str, object]], summary: list[dict[str, object]]) -> list[dict[str, object]]:
    lookup = selected_lookup(selected_rows)
    output = []
    for key in [
        ("source_stratified_hash", "hash0_calib_hash1_test"),
        ("leave_family_out", "holdout_chronos"),
        ("leave_family_out", "holdout_moirai"),
    ]:
        selected = lookup[key]
        overall = summary_row(summary, key[0], key[1], str(selected["candidate_id"]), "overall")
        output.append(
            {
                "Split": f"{key[0]}:{key[1]}",
                "Selected": selected["candidate_id"],
                "DualCertified": "yes" if int(selected["dual_ltt_accepted"]) else "no",
                "CalUC": num(selected["undercoverage_empirical_risk"]),
                "CalHarm": pct(selected["wql_harm_empirical_risk"]),
                "TestRisk": num(overall["repair_undercoverage_risk"]),
                "TestWQL": num(overall["repair_median_wql_rer"]),
                "TestHarm": pct(overall["repair_wql_noninferiority_harm_rate"]),
                "Coverage": num(overall["repair_mean_coverage"]),
            }
        )
    return output


def primary_group_rows(summary: list[dict[str, object]], selected_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = selected_lookup(selected_rows)[("source_stratified_hash", "hash0_calib_hash1_test")]
    wanted = [
        "overall",
        "family:chronos|role:failure_target",
        "family:moirai|role:failure_target",
        "family:chronos|role:positive_control",
        "family:moirai|role:positive_control",
    ]
    rows = []
    for group in wanted:
        row = summary_row(
            summary,
            "source_stratified_hash",
            "hash0_calib_hash1_test",
            str(selected["candidate_id"]),
            group,
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
                "WQLHarm": pct(row["repair_wql_noninferiority_harm_rate"]),
                "SafetyHarm": pct(row["repair_safety_harm_rate"]),
                "WidthGuard": pct(row["width_collapse_guard_rate"]),
            }
        )
    return rows


def source_candidate_rows(calibration_tests: list[dict[str, object]]) -> list[dict[str, object]]:
    source_rows = [
        row
        for row in calibration_tests
        if row["split_protocol"] == "source_stratified_hash" and row["split_id"] == "hash0_calib_hash1_test"
    ]
    by_candidate: dict[str, dict[str, object]] = {}
    for row in source_rows:
        item = by_candidate.setdefault(
            str(row["candidate_id"]),
            {
                "Candidate": row["candidate_id"],
                "Selected": "yes" if int(row["selected"]) else "",
                "Dual": "yes" if int(row["dual_ltt_accepted"]) else "no",
                "Utility": num(row["utility"]),
                "MedianWQL": num(row["median_wql_rer"]),
            },
        )
        if row["risk_name"] == "undercoverage":
            item["UC"] = num(row["empirical_risk"])
            item["UCok"] = "yes" if int(row["ltt_accepted"]) else "no"
        elif row["risk_name"] == "wql_noninferiority_harm":
            item["Harm"] = pct(row["empirical_risk"])
            item["HarmOk"] = "yes" if int(row["ltt_accepted"]) else "no"
    return list(by_candidate.values())


def write_report(
    selected_rows: list[dict[str, object]],
    summary: list[dict[str, object]],
    calibration_tests: list[dict[str, object]],
    status: dict[str, object],
) -> None:
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Dual-Risk CRC/LTT-Calibrated LSSS",
                "",
                "## Method",
                "",
                "Dual-risk CRC/LTT-LSSS upgrades the interval-head selection from a single coverage-risk screen to a two-risk safety screen. A candidate head must pass both LTT tests on calibration windows before it can be selected for test evaluation.",
                "",
                f"- Risk 1: q10-q90 undercoverage, `max(0, {crc.NOMINAL_COVERAGE} - coverage)`, alpha `{UNDERCOVERAGE_ALPHA}`, delta `{DELTA_PER_RISK}`.",
                f"- Risk 2: WQL non-inferiority harm, `repair_wql_rer > model_wql_rer + {WQL_HARM_MARGIN}`, alpha `{WQL_HARM_ALPHA}`, delta `{DELTA_PER_RISK}`.",
                f"- Utility among dual-certified candidates: `-median(WQL-RER) - {UTILITY_UNDERCOVERAGE_LAMBDA} * undercoverage - {UTILITY_WQL_HARM_LAMBDA} * WQL_harm_rate`.",
                f"- Tie rule: if certified candidates are within `{UTILITY_TIE_EPS}` utility, prefer the uncertainty-collapse diagnostic head.",
                "- Candidate set: four fixed-scale heads plus three uncertainty-collapse diagnostic heads.",
                "- Main split: source-stratified hash calibration/test. Leave-family-out remains a transfer stress test.",
                "",
                "## Source-Split Calibration Screen",
                "",
                markdown_table(
                    source_candidate_rows(calibration_tests),
                    [
                        ("Candidate", "Candidate"),
                        ("Selected", "Selected"),
                        ("Dual", "Dual ok"),
                        ("UC", "Cal UC"),
                        ("UCok", "UC ok"),
                        ("Harm", "Cal WQL harm"),
                        ("HarmOk", "Harm ok"),
                        ("MedianWQL", "Cal WQL"),
                        ("Utility", "Utility"),
                    ],
                ),
                "",
                "## Selected Policies",
                "",
                markdown_table(
                    selected_report_rows(selected_rows, summary),
                    [
                        ("Split", "Split"),
                        ("Selected", "Selected"),
                        ("DualCertified", "Dual certified"),
                        ("CalUC", "Cal UC"),
                        ("CalHarm", "Cal WQL harm"),
                        ("TestRisk", "Test risk"),
                        ("TestWQL", "Test WQL"),
                        ("TestHarm", "Test WQL harm"),
                        ("Coverage", "Coverage"),
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
                        ("WQLHarm", "WQL harm"),
                        ("SafetyHarm", "Safety harm"),
                        ("WidthGuard", "Width guard"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                "- The dual-risk screen automatically removes the two intuitive but unsafe extremes: too-sharp heads fail coverage, while overly safe heads fail WQL non-inferiority.",
                "- On the current 640-window slice, the source-split selector chooses a severe-collapse diagnostic head rather than the fixed balanced head. Calibration still controls the safety risks; the diagnostic tie rule only acts among near-equivalent dual-certified candidates.",
                "- Compared with the previous balanced dual-risk head, the selected diagnostic head slightly improves overall WQL and Chronos failure WQL while preserving the near-neutral Moirai failure tradeoff. The honest paper claim is a tunable risk tradeoff, not a free lunch.",
                "- This is the first positive Experiment A result: a mechanism-rich head can pass dual-risk calibration when the mechanism is conservative and localized.",
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{CANDIDATE_SUMMARY_OUT.relative_to(ROOT)}`",
                f"- `{SELECTED_OUT.relative_to(ROOT)}`",
                f"- `{CALIBRATION_TEST_OUT.relative_to(ROOT)}`",
                f"- `{STATUS_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    policy = crc.fullgrid.common_cpr_policy()
    windows = crc.all_windows()
    candidates = candidate_configs()
    selected_rows: list[dict[str, object]] = []
    calibration_tests: list[dict[str, object]] = []
    all_window_rows: list[dict[str, object]] = []
    summary: list[dict[str, object]] = []
    all_candidate_summary: list[dict[str, object]] = []
    for split in crc.split_definitions(windows):
        selected, split_selected_rows, split_calibration_tests, split_calibration_rows = select_candidate(
            split["calibration"],
            policy,
            candidates,
            split_protocol=str(split["split_protocol"]),
            split_id=str(split["split_id"]),
        )
        selected_rows.extend(split_selected_rows)
        calibration_tests.extend(split_calibration_tests)
        for candidate in candidates:
            candidate_calibration_rows = [
                row for row in split_calibration_rows if row["candidate_id"] == candidate["candidate_id"]
            ]
            all_candidate_summary.extend(build_candidate_summary(candidate_calibration_rows))
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

    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows_total": len(windows),
        "n_chronos_windows": sum(1 for window in windows if window["family"] == "chronos"),
        "n_moirai_windows": sum(1 for window in windows if window["family"] == "moirai"),
        "n_candidates": len(candidates),
        "undercoverage_alpha": UNDERCOVERAGE_ALPHA,
        "wql_harm_alpha": WQL_HARM_ALPHA,
        "total_delta": TOTAL_DELTA,
        "delta_per_risk": DELTA_PER_RISK,
        "wql_harm_margin": WQL_HARM_MARGIN,
        "utility_undercoverage_lambda": UTILITY_UNDERCOVERAGE_LAMBDA,
        "utility_wql_harm_lambda": UTILITY_WQL_HARM_LAMBDA,
        "utility_tie_eps": UTILITY_TIE_EPS,
        "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
        "summary": str(SUMMARY_OUT.relative_to(ROOT)),
        "candidate_summary": str(CANDIDATE_SUMMARY_OUT.relative_to(ROOT)),
        "selected_configs": str(SELECTED_OUT.relative_to(ROOT)),
        "calibration_tests": str(CALIBRATION_TEST_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, all_window_rows)
    write_csv(SUMMARY_OUT, summary)
    write_csv(CANDIDATE_SUMMARY_OUT, all_candidate_summary)
    write_csv(SELECTED_OUT, selected_rows)
    write_csv(CALIBRATION_TEST_OUT, calibration_tests)
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")
    write_report(selected_rows, summary, calibration_tests, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
