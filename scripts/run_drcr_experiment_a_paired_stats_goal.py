#!/usr/bin/env python
"""Paired statistical audit for DRCR Experiment A.

The dual-risk selector now chooses a conservative uncertainty-collapse head.
This script compares that mechanism-rich head against the previous fixed
balanced dual-risk head on the exact same held-out windows.  The output is
deliberately paired: every row has both candidates applied to the same window,
so the confidence intervals and tests quantify the incremental contribution of
the diagnostic head rather than re-measuring the whole repair module.
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
import run_dual_risk_crc_ltt_lsss_goal as drcr  # noqa: E402

OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "drcr_experiment_a_paired_stats_report.md"
STATUS_OUT = OUT_DIR / "drcr_experiment_a_paired_stats_status.json"
WINDOW_OUT = OUT_DIR / "drcr_experiment_a_paired_windows.csv"
STATS_OUT = OUT_DIR / "drcr_experiment_a_paired_stats.csv"

SPLIT_PROTOCOL = "source_stratified_hash"
SPLIT_ID = "hash0_calib_hash1_test"
COMPARATOR_ID = "balanced_f1.00_s2.00"
EXPECTED_DIAGNOSTIC_ID = "collapse_floor_t0.10_f0.95_c1.00_s2.00"
N_BOOTSTRAP = 4000
N_PERMUTATIONS = 4000
SEED = 20260705
EPS = 1e-12
WQL_RER_CAP = 10.0


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


def mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def rate(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def bootstrap_ci(
    values: np.ndarray,
    statistic: str,
    *,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    sample = arr[draws]
    if statistic == "mean":
        stats = np.mean(sample, axis=1)
    elif statistic == "median":
        stats = np.median(sample, axis=1)
    else:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def exact_sign_test_p(values: np.ndarray) -> tuple[int, int, int, float]:
    """Two-sided exact sign test for paired deltas.

    Negative deltas mean the diagnostic head is better because lower WQL/RER is
    better.  Zeros are dropped from the exact binomial test.
    """

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    improved = int(np.sum(arr < -EPS))
    worsened = int(np.sum(arr > EPS))
    n = improved + worsened
    if n == 0:
        return improved, worsened, n, 1.0
    k = min(improved, worsened)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return improved, worsened, n, float(min(1.0, 2.0 * prob))


def paired_permutation_p(
    values: np.ndarray,
    *,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = SEED + 1,
) -> float:
    """Two-sided random sign-flip test for the paired mean delta."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    observed = abs(float(np.mean(arr)))
    if observed <= EPS:
        return 1.0
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_permutations, arr.size))
    null_stats = np.abs(np.mean(signs * arr, axis=1))
    return float((1.0 + np.sum(null_stats >= observed - EPS)) / (n_permutations + 1.0))


def fmt_num(value: object, digits: int = 4) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{parsed:.{digits}f}"


def fmt_pct(value: object, digits: int = 1) -> str:
    parsed = finite_float(value, float("nan"))
    return "nan" if not math.isfinite(parsed) else f"{100.0 * parsed:.{digits}f}%"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


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


def selected_source_candidate() -> dict[str, object]:
    policy = crc.fullgrid.common_cpr_policy()
    windows = crc.all_windows()
    candidates = drcr.candidate_configs()
    split = next(
        item
        for item in crc.split_definitions(windows)
        if item["split_protocol"] == SPLIT_PROTOCOL and item["split_id"] == SPLIT_ID
    )
    selected, _, _, _ = drcr.select_candidate(
        split["calibration"],
        policy,
        candidates,
        split_protocol=SPLIT_PROTOCOL,
        split_id=SPLIT_ID,
    )
    return selected


def paired_windows() -> tuple[list[dict[str, object]], str]:
    policy = crc.fullgrid.common_cpr_policy()
    windows = crc.all_windows()
    candidates = {str(candidate["candidate_id"]): candidate for candidate in drcr.candidate_configs()}
    if COMPARATOR_ID not in candidates:
        raise ValueError(f"Missing comparator candidate {COMPARATOR_ID}")
    selected = selected_source_candidate()
    diagnostic_id = str(selected["candidate_id"])
    if diagnostic_id != EXPECTED_DIAGNOSTIC_ID:
        raise ValueError(f"Expected {EXPECTED_DIAGNOSTIC_ID}, selected {diagnostic_id}")

    split = next(
        item
        for item in crc.split_definitions(windows)
        if item["split_protocol"] == SPLIT_PROTOCOL and item["split_id"] == SPLIT_ID
    )
    balanced_rows = drcr.apply_candidate(
        split["test"],
        policy,
        candidates[COMPARATOR_ID],
        split_protocol=SPLIT_PROTOCOL,
        split_id=SPLIT_ID,
        phase="test",
    )
    diagnostic_rows = drcr.apply_candidate(
        split["test"],
        policy,
        candidates[diagnostic_id],
        split_protocol=SPLIT_PROTOCOL,
        split_id=SPLIT_ID,
        phase="test",
    )
    balanced_by_key = {window_key(row): row for row in balanced_rows}
    diagnostic_by_key = {window_key(row): row for row in diagnostic_rows}
    if set(balanced_by_key) != set(diagnostic_by_key):
        raise ValueError("Balanced and diagnostic windows do not align")

    rows: list[dict[str, object]] = []
    for key in sorted(balanced_by_key):
        balanced = balanced_by_key[key]
        diagnostic = diagnostic_by_key[key]
        balanced_wql = finite_float(balanced["repair_wql_rer"], float("nan"))
        diagnostic_wql = finite_float(diagnostic["repair_wql_rer"], float("nan"))
        balanced_wql_clipped = min(balanced_wql, WQL_RER_CAP)
        diagnostic_wql_clipped = min(diagnostic_wql, WQL_RER_CAP)
        balanced_log_wql = math.log1p(max(0.0, balanced_wql))
        diagnostic_log_wql = math.log1p(max(0.0, diagnostic_wql))
        balanced_uc = finite_float(balanced["repair_undercoverage_risk"], float("nan"))
        diagnostic_uc = finite_float(diagnostic["repair_undercoverage_risk"], float("nan"))
        balanced_cov = finite_float(balanced["repair_coverage_q10_q90"], float("nan"))
        diagnostic_cov = finite_float(diagnostic["repair_coverage_q10_q90"], float("nan"))
        balanced_harm = int(balanced["repair_wql_noninferiority_harm"])
        diagnostic_harm = int(diagnostic["repair_wql_noninferiority_harm"])
        balanced_safety = int(balanced["repair_safety_harm"])
        diagnostic_safety = int(diagnostic["repair_safety_harm"])
        balanced_failure = int(balanced["repair_wql_failure_delta005"])
        diagnostic_failure = int(diagnostic["repair_wql_failure_delta005"])
        rows.append(
            {
                "split_protocol": SPLIT_PROTOCOL,
                "split_id": SPLIT_ID,
                "phase": "test",
                "family": balanced["family"],
                "source": balanced["source"],
                "role": balanced["role"],
                "dataset": balanced["dataset"],
                "model": balanced["model"],
                "target_id": balanced["target_id"],
                "series_id": balanced["series_id"],
                "origin": balanced["origin"],
                "window_index": balanced["window_index"],
                "model_wql_rer": balanced["model_wql_rer"],
                "model_coverage_q10_q90": balanced["model_coverage_q10_q90"],
                "balanced_candidate_id": COMPARATOR_ID,
                "diagnostic_candidate_id": diagnostic_id,
                "balanced_wql_rer": balanced_wql,
                "diagnostic_wql_rer": diagnostic_wql,
                "diagnostic_minus_balanced_wql_rer": diagnostic_wql - balanced_wql,
                "balanced_clipped_wql_rer": balanced_wql_clipped,
                "diagnostic_clipped_wql_rer": diagnostic_wql_clipped,
                "diagnostic_minus_balanced_clipped_wql_rer": diagnostic_wql_clipped - balanced_wql_clipped,
                "balanced_log1p_wql_rer": balanced_log_wql,
                "diagnostic_log1p_wql_rer": diagnostic_log_wql,
                "diagnostic_minus_balanced_log1p_wql_rer": diagnostic_log_wql - balanced_log_wql,
                "balanced_undercoverage_risk": balanced_uc,
                "diagnostic_undercoverage_risk": diagnostic_uc,
                "diagnostic_minus_balanced_undercoverage_risk": diagnostic_uc - balanced_uc,
                "balanced_coverage_q10_q90": balanced_cov,
                "diagnostic_coverage_q10_q90": diagnostic_cov,
                "diagnostic_minus_balanced_coverage_q10_q90": diagnostic_cov - balanced_cov,
                "balanced_wql_noninferiority_harm": balanced_harm,
                "diagnostic_wql_noninferiority_harm": diagnostic_harm,
                "diagnostic_minus_balanced_wql_harm": diagnostic_harm - balanced_harm,
                "balanced_safety_harm": balanced_safety,
                "diagnostic_safety_harm": diagnostic_safety,
                "diagnostic_minus_balanced_safety_harm": diagnostic_safety - balanced_safety,
                "balanced_wql_failure_delta005": balanced_failure,
                "diagnostic_wql_failure_delta005": diagnostic_failure,
                "diagnostic_minus_balanced_failure": diagnostic_failure - balanced_failure,
                "balanced_width_collapse_guard": balanced["width_collapse_guard"],
                "diagnostic_width_collapse_guard": diagnostic["width_collapse_guard"],
                "balanced_interval_scale": balanced["interval_scale"],
                "diagnostic_interval_scale": diagnostic["interval_scale"],
                "native_width_ratio": diagnostic["native_width_ratio"],
                "structured_control_guard": diagnostic["structured_control_guard"],
            }
        )
    return rows, diagnostic_id


def group_specs(rows: list[dict[str, object]]) -> list[tuple[str, str, list[dict[str, object]]]]:
    groups: list[tuple[str, str, list[dict[str, object]]]] = [("overall", "overall", rows)]
    for family in sorted({str(row["family"]) for row in rows}):
        groups.append((f"family:{family}", "family", [row for row in rows if row["family"] == family]))
    for role in sorted({str(row["role"]) for row in rows}):
        groups.append((f"role:{role}", "role", [row for row in rows if row["role"] == role]))
    for family in sorted({str(row["family"]) for row in rows}):
        for role in sorted({str(row["role"]) for row in rows if row["family"] == family}):
            subset = [row for row in rows if row["family"] == family and row["role"] == role]
            if subset:
                groups.append((f"family:{family}|role:{role}", "family_role", subset))
    return groups


def summarize_group(
    rows: list[dict[str, object]],
    group: str,
    group_type: str,
    *,
    diagnostic_id: str,
) -> dict[str, object]:
    wql_delta = np.asarray([finite_float(row["diagnostic_minus_balanced_wql_rer"], float("nan")) for row in rows])
    clipped_wql_delta = np.asarray(
        [finite_float(row["diagnostic_minus_balanced_clipped_wql_rer"], float("nan")) for row in rows]
    )
    log_wql_delta = np.asarray(
        [finite_float(row["diagnostic_minus_balanced_log1p_wql_rer"], float("nan")) for row in rows]
    )
    uc_delta = np.asarray(
        [finite_float(row["diagnostic_minus_balanced_undercoverage_risk"], float("nan")) for row in rows]
    )
    cov_delta = np.asarray(
        [finite_float(row["diagnostic_minus_balanced_coverage_q10_q90"], float("nan")) for row in rows]
    )
    wql_harm_delta = np.asarray([finite_float(row["diagnostic_minus_balanced_wql_harm"]) for row in rows])
    safety_harm_delta = np.asarray([finite_float(row["diagnostic_minus_balanced_safety_harm"]) for row in rows])
    failure_delta = np.asarray([finite_float(row["diagnostic_minus_balanced_failure"]) for row in rows])

    wql_mean_ci = bootstrap_ci(wql_delta, "mean")
    clipped_wql_mean_ci = bootstrap_ci(clipped_wql_delta, "mean", seed=SEED + 7)
    log_wql_mean_ci = bootstrap_ci(log_wql_delta, "mean", seed=SEED + 8)
    wql_median_ci = bootstrap_ci(wql_delta, "median", seed=SEED + 2)
    uc_mean_ci = bootstrap_ci(uc_delta, "mean", seed=SEED + 3)
    safety_mean_ci = bootstrap_ci(safety_harm_delta, "mean", seed=SEED + 4)
    harm_mean_ci = bootstrap_ci(wql_harm_delta, "mean", seed=SEED + 5)
    failure_mean_ci = bootstrap_ci(failure_delta, "mean", seed=SEED + 6)
    improved, worsened, nonzero, sign_p = exact_sign_test_p(wql_delta)
    perm_p = paired_permutation_p(wql_delta)
    log_perm_p = paired_permutation_p(log_wql_delta, seed=SEED + 9)

    return {
        "split_protocol": SPLIT_PROTOCOL,
        "split_id": SPLIT_ID,
        "group": group,
        "group_type": group_type,
        "n_windows": len(rows),
        "balanced_candidate_id": COMPARATOR_ID,
        "diagnostic_candidate_id": diagnostic_id,
        "balanced_median_wql_rer": median(
            np.asarray([finite_float(row["balanced_wql_rer"], float("nan")) for row in rows])
        ),
        "diagnostic_median_wql_rer": median(
            np.asarray([finite_float(row["diagnostic_wql_rer"], float("nan")) for row in rows])
        ),
        "paired_mean_wql_delta": mean(wql_delta),
        "paired_mean_wql_delta_ci_low": wql_mean_ci[0],
        "paired_mean_wql_delta_ci_high": wql_mean_ci[1],
        "paired_mean_clipped_wql_delta": mean(clipped_wql_delta),
        "paired_mean_clipped_wql_delta_ci_low": clipped_wql_mean_ci[0],
        "paired_mean_clipped_wql_delta_ci_high": clipped_wql_mean_ci[1],
        "paired_mean_log1p_wql_delta": mean(log_wql_delta),
        "paired_mean_log1p_wql_delta_ci_low": log_wql_mean_ci[0],
        "paired_mean_log1p_wql_delta_ci_high": log_wql_mean_ci[1],
        "paired_median_wql_delta": median(wql_delta),
        "paired_median_wql_delta_ci_low": wql_median_ci[0],
        "paired_median_wql_delta_ci_high": wql_median_ci[1],
        "wql_delta_improved_count": improved,
        "wql_delta_worsened_count": worsened,
        "wql_delta_nonzero_count": nonzero,
        "wql_delta_exact_sign_p": sign_p,
        "wql_delta_paired_permutation_mean_p": perm_p,
        "log1p_wql_delta_paired_permutation_mean_p": log_perm_p,
        "balanced_undercoverage_risk": mean(
            np.asarray([finite_float(row["balanced_undercoverage_risk"], float("nan")) for row in rows])
        ),
        "diagnostic_undercoverage_risk": mean(
            np.asarray([finite_float(row["diagnostic_undercoverage_risk"], float("nan")) for row in rows])
        ),
        "paired_mean_undercoverage_delta": mean(uc_delta),
        "paired_mean_undercoverage_delta_ci_low": uc_mean_ci[0],
        "paired_mean_undercoverage_delta_ci_high": uc_mean_ci[1],
        "balanced_mean_coverage": mean(
            np.asarray([finite_float(row["balanced_coverage_q10_q90"], float("nan")) for row in rows])
        ),
        "diagnostic_mean_coverage": mean(
            np.asarray([finite_float(row["diagnostic_coverage_q10_q90"], float("nan")) for row in rows])
        ),
        "paired_mean_coverage_delta": mean(cov_delta),
        "balanced_wql_harm_rate": rate(
            np.asarray([finite_float(row["balanced_wql_noninferiority_harm"]) for row in rows])
        ),
        "diagnostic_wql_harm_rate": rate(
            np.asarray([finite_float(row["diagnostic_wql_noninferiority_harm"]) for row in rows])
        ),
        "paired_wql_harm_rate_delta": mean(wql_harm_delta),
        "paired_wql_harm_rate_delta_ci_low": harm_mean_ci[0],
        "paired_wql_harm_rate_delta_ci_high": harm_mean_ci[1],
        "balanced_safety_harm_rate": rate(np.asarray([finite_float(row["balanced_safety_harm"]) for row in rows])),
        "diagnostic_safety_harm_rate": rate(
            np.asarray([finite_float(row["diagnostic_safety_harm"]) for row in rows])
        ),
        "paired_safety_harm_rate_delta": mean(safety_harm_delta),
        "paired_safety_harm_rate_delta_ci_low": safety_mean_ci[0],
        "paired_safety_harm_rate_delta_ci_high": safety_mean_ci[1],
        "balanced_failure_rate": rate(np.asarray([finite_float(row["balanced_wql_failure_delta005"]) for row in rows])),
        "diagnostic_failure_rate": rate(
            np.asarray([finite_float(row["diagnostic_wql_failure_delta005"]) for row in rows])
        ),
        "paired_failure_rate_delta": mean(failure_delta),
        "paired_failure_rate_delta_ci_low": failure_mean_ci[0],
        "paired_failure_rate_delta_ci_high": failure_mean_ci[1],
        "diagnostic_width_guard_rate": rate(
            np.asarray([finite_float(row["diagnostic_width_collapse_guard"]) for row in rows])
        ),
    }


def report_rows(stats: list[dict[str, object]]) -> list[dict[str, object]]:
    wanted = [
        "overall",
        "family:chronos|role:failure_target",
        "family:moirai|role:failure_target",
        "family:chronos|role:positive_control",
        "family:moirai|role:positive_control",
    ]
    lookup = {str(row["group"]): row for row in stats}
    rows = []
    for group in wanted:
        row = lookup[group]
        rows.append(
            {
                "Group": group,
                "N": row["n_windows"],
                "BalancedWQL": fmt_num(row["balanced_median_wql_rer"], 3),
                "DiagnosticWQL": fmt_num(row["diagnostic_median_wql_rer"], 3),
                "LogMeanDelta": fmt_num(row["paired_mean_log1p_wql_delta"], 4),
                "LogMeanCI": f"[{fmt_num(row['paired_mean_log1p_wql_delta_ci_low'], 4)}, {fmt_num(row['paired_mean_log1p_wql_delta_ci_high'], 4)}]",
                "ClippedMeanDelta": fmt_num(row["paired_mean_clipped_wql_delta"], 4),
                "MedianDelta": f"{fmt_num(row['paired_median_wql_delta'], 4)} [{fmt_num(row['paired_median_wql_delta_ci_low'], 4)}, {fmt_num(row['paired_median_wql_delta_ci_high'], 4)}]",
                "Signs": f"{row['wql_delta_improved_count']}/{row['wql_delta_worsened_count']}",
                "SignP": fmt_num(row["wql_delta_exact_sign_p"], 4),
                "LogPermP": fmt_num(row["log1p_wql_delta_paired_permutation_mean_p"], 4),
                "SafetyDelta": fmt_pct(row["paired_safety_harm_rate_delta"], 2),
                "HarmDelta": fmt_pct(row["paired_wql_harm_rate_delta"], 2),
                "WidthGuard": fmt_pct(row["diagnostic_width_guard_rate"], 1),
            }
        )
    return rows


def write_report(rows: list[dict[str, object]], stats: list[dict[str, object]], status: dict[str, object]) -> None:
    overall = next(row for row in stats if row["group"] == "overall")
    chronos_failure = next(row for row in stats if row["group"] == "family:chronos|role:failure_target")
    moirai_failure = next(row for row in stats if row["group"] == "family:moirai|role:failure_target")
    log_ci_crosses_zero = (
        finite_float(overall["paired_mean_log1p_wql_delta_ci_low"]) <= 0.0
        <= finite_float(overall["paired_mean_log1p_wql_delta_ci_high"])
    )
    DOC_PATH.write_text(
        "\n".join(
            [
                "# DRCR Experiment A Paired Statistical Audit",
                "",
                "## Question",
                "",
                "Does the mechanism-rich DRCR head add measurable value over the previous fixed balanced dual-risk head, or does it merely make the method easier to narrate?",
                "",
                "The comparison is paired on the locked source-stratified held-out test split. Each window is evaluated with both heads, and all intervals/tests are computed on `diagnostic - balanced`; negative WQL deltas favor the mechanism-rich head.",
                "",
                "## Candidates",
                "",
                f"- Comparator: `{COMPARATOR_ID}`.",
                f"- Mechanism-rich Experiment A head: `{status['diagnostic_candidate_id']}`.",
                f"- Test windows: `{status['n_test_windows']}`.",
                f"- Bootstrap resamples: `{N_BOOTSTRAP}`; paired sign-flip permutations: `{N_PERMUTATIONS}`.",
                f"- Raw WQL-RER means are retained in CSV but not used as the headline statistic because covid-style denominator fragility creates values above `1e17`; the main table uses median WQL plus log/clipped paired deltas with cap `{WQL_RER_CAP}`.",
                "",
                "## Paired Results",
                "",
                markdown_table(
                    report_rows(stats),
                    [
                        ("Group", "Group"),
                        ("N", "N"),
                        ("BalancedWQL", "Balanced WQL"),
                        ("DiagnosticWQL", "Diagnostic WQL"),
                        ("LogMeanDelta", "Mean dlogWQL"),
                        ("LogMeanCI", "Mean dlogWQL 95% CI"),
                        ("ClippedMeanDelta", "Mean clipped dWQL"),
                        ("MedianDelta", "Median dWQL 95% CI"),
                        ("Signs", "Improved/worse"),
                        ("SignP", "Sign p"),
                        ("LogPermP", "Log perm p"),
                        ("SafetyDelta", "Safety harm d"),
                        ("HarmDelta", "WQL harm d"),
                        ("WidthGuard", "Width guard"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                f"- Overall median WQL moves from `{fmt_num(overall['balanced_median_wql_rer'], 3)}` to `{fmt_num(overall['diagnostic_median_wql_rer'], 3)}`. Among non-tied windows, the diagnostic head improves/worsens `{overall['wql_delta_improved_count']}/{overall['wql_delta_worsened_count']}` windows, with sign-test p `{fmt_num(overall['wql_delta_exact_sign_p'], 4)}`.",
                f"- Chronos failure windows are the cleanest win: median WQL moves from `{fmt_num(chronos_failure['balanced_median_wql_rer'], 3)}` to `{fmt_num(chronos_failure['diagnostic_median_wql_rer'], 3)}`; paired median delta is `{fmt_num(chronos_failure['paired_median_wql_delta'], 4)}` with 95% bootstrap CI `{fmt_num(chronos_failure['paired_median_wql_delta_ci_low'], 4)}` to `{fmt_num(chronos_failure['paired_median_wql_delta_ci_high'], 4)}`.",
                f"- Moirai failure remains near-neutral: median WQL moves from `{fmt_num(moirai_failure['balanced_median_wql_rer'], 3)}` to `{fmt_num(moirai_failure['diagnostic_median_wql_rer'], 3)}`.",
                f"- The cost is explicit: overall safety harm changes by `{fmt_pct(overall['paired_safety_harm_rate_delta'], 2)}` and WQL non-inferiority harm changes by `{fmt_pct(overall['paired_wql_harm_rate_delta'], 2)}` versus the fixed balanced head.",
                "- This is not a large effect-size win. It is a calibrated mechanism-shape win with a small paired WQL improvement, a localized Chronos-failure benefit, near-neutral Moirai behavior, and a small safety-harm cost.",
                "- Because the overall log-WQL paired bootstrap CI "
                + ("crosses zero, the paper should frame overall superiority as exploratory rather than statistically locked." if log_ci_crosses_zero else "excludes zero, the current slice supports a statistically locked paired log-WQL improvement.")
                + " The stronger claim remains that dual-risk CRC/LTT can certify a mechanism-rich head without relying on an ad hoc fallback threshold.",
                "",
                "## Artifacts",
                "",
                f"- `{WINDOW_OUT.relative_to(ROOT)}`",
                f"- `{STATS_OUT.relative_to(ROOT)}`",
                f"- `{STATUS_OUT.relative_to(ROOT)}`",
            ]
        )
        + "\n"
    )


def main() -> None:
    rows, diagnostic_id = paired_windows()
    stats = [
        summarize_group(subset, group, group_type, diagnostic_id=diagnostic_id)
        for group, group_type, subset in group_specs(rows)
    ]
    overall = next(row for row in stats if row["group"] == "overall")
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "split_protocol": SPLIT_PROTOCOL,
        "split_id": SPLIT_ID,
        "n_test_windows": len(rows),
        "balanced_candidate_id": COMPARATOR_ID,
        "diagnostic_candidate_id": diagnostic_id,
        "expected_diagnostic_candidate_id": EXPECTED_DIAGNOSTIC_ID,
        "n_bootstrap": N_BOOTSTRAP,
        "n_permutations": N_PERMUTATIONS,
        "seed": SEED,
        "overall_paired_mean_wql_delta": overall["paired_mean_wql_delta"],
        "overall_paired_mean_wql_delta_ci_low": overall["paired_mean_wql_delta_ci_low"],
        "overall_paired_mean_wql_delta_ci_high": overall["paired_mean_wql_delta_ci_high"],
        "overall_paired_mean_clipped_wql_delta": overall["paired_mean_clipped_wql_delta"],
        "overall_paired_mean_clipped_wql_delta_ci_low": overall["paired_mean_clipped_wql_delta_ci_low"],
        "overall_paired_mean_clipped_wql_delta_ci_high": overall["paired_mean_clipped_wql_delta_ci_high"],
        "overall_paired_mean_log1p_wql_delta": overall["paired_mean_log1p_wql_delta"],
        "overall_paired_mean_log1p_wql_delta_ci_low": overall["paired_mean_log1p_wql_delta_ci_low"],
        "overall_paired_mean_log1p_wql_delta_ci_high": overall["paired_mean_log1p_wql_delta_ci_high"],
        "overall_paired_safety_harm_rate_delta": overall["paired_safety_harm_rate_delta"],
        "overall_paired_wql_harm_rate_delta": overall["paired_wql_harm_rate_delta"],
        "wql_rer_cap_for_clipped_delta": WQL_RER_CAP,
        "window_metrics": str(WINDOW_OUT.relative_to(ROOT)),
        "paired_stats": str(STATS_OUT.relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    write_csv(WINDOW_OUT, rows)
    write_csv(STATS_OUT, stats)
    STATUS_OUT.write_text(json.dumps(status, indent=2) + "\n")
    write_report(rows, stats, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
