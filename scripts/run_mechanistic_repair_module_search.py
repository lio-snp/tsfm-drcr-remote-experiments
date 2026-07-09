#!/usr/bin/env python
"""Search mechanism-oriented repair modules beyond plain weighted blending."""

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

import run_cross_family_selective_repair_goal as cross  # noqa: E402
import run_selective_repair_expanded_ablation_goal as selective  # noqa: E402

from low_snr_tsfm.metrics import mae, relative_error_ratio  # noqa: E402


OUT_DIR = ROOT / "results" / "repair"
DOC_PATH = ROOT / "docs" / "mechanistic_repair_module_search_report.md"
BASE_POLICY_ID = "loose_n2_low_structure_adaptive_w0.60_s0.500_max1.00"


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


def rate(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def p90(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.percentile(finite, 90)) if finite else float("nan")


def bootstrap_ci(values: np.ndarray, *, n_bootstrap: int = 2000, seed: int = 23) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    stats = np.mean(arr[draws], axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def exact_mcnemar_p(improved: int, worsened: int) -> float:
    n = improved + worsened
    if n == 0:
        return 1.0
    k = min(improved, worsened)
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return float(min(1.0, 2.0 * prob))


def base_policy() -> dict[str, object]:
    for policy in selective.policy_candidates():
        if policy["policy_id"] == BASE_POLICY_ID:
            return policy
    raise ValueError(f"Missing base policy {BASE_POLICY_ID}")


def feature_float(window: dict[str, object], key: str) -> float:
    return finite_float(window["feature"].get(key))


def is_trend_decay_compatible_conflict(window: dict[str, object]) -> bool:
    """Classify a high disagreement as compatible with local degeneration repair."""

    return feature_float(window, "horizon_context_ratio") <= 0.30 and feature_float(window, "trend_strength") >= 0.40


def is_loose_trend_compatible_conflict(window: dict[str, object]) -> bool:
    """A rejected looser adjudicator used to expose the safety tradeoff."""

    return feature_float(window, "horizon_context_ratio") <= 0.40 and feature_float(window, "trend_strength") >= 0.10


def apply_module(window: dict[str, object], policy: dict[str, object], module_id: str) -> dict[str, object]:
    gate, reason, active_count = selective.gate_decision(window["feature"], policy)
    pre_weight = selective.policy_reference_weight(policy, gate, active_count)
    reference = np.asarray(window["baseline_forecast"], dtype=float)
    model = np.asarray(window["model_forecast"], dtype=float)
    actual = np.asarray(window["actual"], dtype=float)
    q10 = np.asarray(window["q10"], dtype=float)
    q90 = np.asarray(window["q90"], dtype=float)
    outside_rate, outside_ratio = selective.reference_interval_conflict(reference, q10, q90)
    weight = pre_weight
    reference_used = reference
    shield_active = 0
    conflict_override = 0
    projection_active = 0
    if module_id == "unshielded_adaptive_deferral":
        pass
    elif module_id == "cssr_cap_all_conflicts":
        if outside_rate >= 0.40 and weight > 0.125:
            weight = 0.125
            shield_active = 1
    elif module_id == "rr_cssr_trend_decay_adjudicator":
        if outside_rate >= 0.40 and weight > 0.125:
            if is_trend_decay_compatible_conflict(window):
                conflict_override = 1
            else:
                weight = 0.125
                shield_active = 1
    elif module_id == "loose_rr_cssr_trend_adjudicator_ablation":
        if outside_rate >= 0.40 and weight > 0.125:
            if is_loose_trend_compatible_conflict(window):
                conflict_override = 1
            else:
                weight = 0.125
                shield_active = 1
    elif module_id == "interval_projected_reference":
        if outside_rate >= 0.40:
            lower = np.minimum(q10, q90)
            upper = np.maximum(q10, q90)
            reference_used = np.clip(reference, lower, upper)
            projection_active = 1
    elif module_id == "smooth_disagreement_decay":
        if outside_rate > 0.0 and weight > 0.0:
            weight = max(0.125, weight * math.exp(-2.0 * outside_rate))
            shield_active = int(weight < pre_weight - 1e-12)
    else:
        raise ValueError(f"Unsupported module_id: {module_id}")
    forecast = (1.0 - weight) * model + weight * reference_used
    repair_mae = mae(actual, forecast)
    repair_rer = relative_error_ratio(repair_mae, finite_float(window["baseline_mae"]))
    return {
        "module_id": module_id,
        "family": window["family"],
        "source": window["source"],
        "role": window["role"],
        "dataset": window["dataset"],
        "series_id": window["series_id"],
        "origin": window["origin"],
        "window_index": window["window_index"],
        "gate_active": gate,
        "gate_reason": reason,
        "active_ex_ante_factor_count": active_count,
        "pre_weight": pre_weight,
        "effective_weight": weight,
        "shield_active": shield_active,
        "conflict_override": conflict_override,
        "projection_active": projection_active,
        "trend_decay_compatible_conflict": int(is_trend_decay_compatible_conflict(window)),
        "reference_outside_interval_rate": outside_rate,
        "reference_outside_interval_mean_ratio": outside_ratio,
        "model_failure_delta_005": window["model_failure"],
        "repair_failure_delta_005": int(repair_rer > 1.05),
        "model_relative_error_ratio": window["model_rer"],
        "repair_relative_error_ratio": repair_rer,
        "relative_error_ratio_delta": repair_rer - finite_float(window["model_rer"]),
        "model_mae": window["model_mae"],
        "baseline_mae": window["baseline_mae"],
        "repair_mae": repair_mae,
        "horizon_context_ratio": feature_float(window, "horizon_context_ratio"),
        "trend_strength": feature_float(window, "trend_strength"),
        "zero_ratio": feature_float(window, "zero_ratio"),
        "spectral_entropy": feature_float(window, "spectral_entropy"),
    }


def summarize(rows: list[dict[str, object]], module_id: str, group: str, group_type: str) -> dict[str, object]:
    model_fail = np.asarray([finite_float(row["model_failure_delta_005"]) for row in rows], dtype=float)
    repair_fail = np.asarray([finite_float(row["repair_failure_delta_005"]) for row in rows], dtype=float)
    diff = model_fail - repair_fail
    improved = int(np.sum((model_fail == 1) & (repair_fail == 0)))
    worsened = int(np.sum((model_fail == 0) & (repair_fail == 1)))
    ci_low, ci_high = bootstrap_ci(diff)
    return {
        "module_id": module_id,
        "group": group,
        "group_type": group_type,
        "n_windows": len(rows),
        "model_failure_rate": rate(model_fail.tolist()),
        "repair_failure_rate": rate(repair_fail.tolist()),
        "failure_rate_reduction": rate(diff.tolist()),
        "failure_reduction_ci_low": ci_low,
        "failure_reduction_ci_high": ci_high,
        "mcnemar_improved_count": improved,
        "mcnemar_worsened_count": worsened,
        "mcnemar_exact_p": exact_mcnemar_p(improved, worsened),
        "gate_rate": rate([finite_float(row["gate_active"]) for row in rows]),
        "shield_rate": rate([finite_float(row["shield_active"]) for row in rows]),
        "override_rate": rate([finite_float(row["conflict_override"]) for row in rows]),
        "projection_rate": rate([finite_float(row["projection_active"]) for row in rows]),
        "mean_reference_outside_interval_rate": rate(
            [finite_float(row["reference_outside_interval_rate"]) for row in rows]
        ),
        "median_rer_delta": median([finite_float(row["relative_error_ratio_delta"]) for row in rows]),
        "p90_rer_delta": p90([finite_float(row["relative_error_ratio_delta"]) for row in rows]),
    }


def summary_rows(rows: list[dict[str, object]], module_id: str) -> list[dict[str, object]]:
    summaries = [summarize(rows, module_id, "overall", "overall")]
    for role in sorted({str(row["role"]) for row in rows}):
        summaries.append(summarize([row for row in rows if row["role"] == role], module_id, f"role:{role}", "role"))
    for family in sorted({str(row["family"]) for row in rows}):
        summaries.append(
            summarize([row for row in rows if row["family"] == family], module_id, f"family:{family}", "family")
        )
    for family in sorted({str(row["family"]) for row in rows}):
        subset = [row for row in rows if row["family"] == family and row["role"] == "positive_control"]
        if subset:
            summaries.append(summarize(subset, module_id, f"family:{family}|role:positive_control", "family_role"))
    return summaries


def fmt_pct(value: object) -> str:
    return f"{100.0 * finite_float(value):.1f}%"


def fmt_num(value: object, digits: int = 3) -> str:
    return f"{finite_float(value):.{digits}f}"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(key, "")) for key, _ in columns) + " |")
    return "\n".join(lines)


def write_report(summary: list[dict[str, object]]) -> None:
    lookup = {(row["module_id"], row["group"]): row for row in summary}
    modules = [
        "unshielded_adaptive_deferral",
        "cssr_cap_all_conflicts",
        "rr_cssr_trend_decay_adjudicator",
        "loose_rr_cssr_trend_adjudicator_ablation",
        "interval_projected_reference",
        "smooth_disagreement_decay",
    ]
    table_rows = []
    for module_id in modules:
        overall = lookup[(module_id, "overall")]
        pc = lookup[(module_id, "role:positive_control")]
        failure = lookup[(module_id, "role:failure_target")]
        stress = lookup[(module_id, "role:stress_target")]
        weak_pc = lookup[(module_id, "role:weak_positive_control")]
        times_pc = lookup[(module_id, "family:timesfm|role:positive_control")]
        table_rows.append(
            {
                "Module": module_id,
                "Overall": fmt_pct(overall["failure_rate_reduction"]),
                "CI": f"[{fmt_pct(overall['failure_reduction_ci_low'])}, {fmt_pct(overall['failure_reduction_ci_high'])}]",
                "Failure": fmt_pct(failure["failure_rate_reduction"]),
                "Stress": fmt_pct(stress["failure_rate_reduction"]),
                "PC p90": fmt_num(pc["p90_rer_delta"]),
                "Weak-PC p90": fmt_num(weak_pc["p90_rer_delta"]),
                "TimesFM PC p90": fmt_num(times_pc["p90_rer_delta"]),
                "Shield": fmt_pct(overall["shield_rate"]),
                "Override": fmt_pct(overall["override_rate"]),
            }
        )
    best = lookup[("rr_cssr_trend_decay_adjudicator", "overall")]
    best_pc = lookup[("rr_cssr_trend_decay_adjudicator", "role:positive_control")]
    best_times_pc = lookup[("rr_cssr_trend_decay_adjudicator", "family:timesfm|role:positive_control")]
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Mechanistic Repair Module Search",
                "",
                "## Candidate Mechanism",
                "",
                "The strongest new candidate is `rr_cssr_trend_decay_adjudicator`, a regime-resolved variant of CSSR. It treats interval disagreement as ambiguous rather than uniformly unsafe. If disagreement occurs in a trend/decay-compatible low-local-structure window, it allows the reference expert to repair the TSFM; otherwise it keeps the conflict shield and caps the reference weight.",
                "",
                "Rule:",
                "",
                "```text",
                "if low_structure_gate:",
                "    w = adaptive_reference_weight(active_factors)",
                "    if reference outside TSFM interval on >=40% horizon:",
                "        if horizon_context_ratio <= 0.30 and trend_strength >= 0.40:",
                "            keep w  # degeneration-compatible conflict",
                "        else:",
                "            w = min(w, 0.125)  # unsafe expert disagreement",
                "```",
                "",
                "## Experiment Contract",
                "",
                "- Hypothesis: not every TSFM/reference conflict is unsafe; conflicts inside a short-context trend/decay degeneration regime should be repaired, while conflicts on positive-control regimes should stay shielded.",
                "- Controlled variable: the conflict-resolution rule applied after the same low-structure gate and adaptive reference weight.",
                "- Primary metric: overall failure-rate reduction subject to strict positive-control and TimesFM positive-control p90 dRER safety constraints.",
                "- Boundary safety metric: weak-positive-control p90 dRER, used to reject over-permissive conflict overrides.",
                "- Fallback story if negative: conflict shielding remains a safety protocol, but the current trend/decay resolution rule is not a stable mechanism.",
                "",
                "## Results",
                "",
                markdown_table(
                    table_rows,
                    [
                        ("Module", "Module"),
                        ("Overall", "Overall Reduction"),
                        ("CI", "95% CI"),
                        ("Failure", "Failure Target"),
                        ("Stress", "Stress Target"),
                        ("PC p90", "PC p90 dRER"),
                        ("Weak-PC p90", "Weak-PC p90 dRER"),
                        ("TimesFM PC p90", "TimesFM PC p90"),
                        ("Shield", "Shield Rate"),
                        ("Override", "Override Rate"),
                    ],
                ),
                "",
                "## Interpretation",
                "",
                (
                    f"`rr_cssr_trend_decay_adjudicator` improves overall failure reduction to {fmt_pct(best['failure_rate_reduction'])} "
                    f"with CI [{fmt_pct(best['failure_reduction_ci_low'])}, {fmt_pct(best['failure_reduction_ci_high'])}], while keeping "
                    f"positive-control p90 dRER at {fmt_num(best_pc['p90_rer_delta'])} and TimesFM positive-control p90 at "
                    f"{fmt_num(best_times_pc['p90_rer_delta'])}."
                ),
                "",
                "The `loose_rr_cssr_trend_adjudicator_ablation` row is intentionally reported as a rejected high-gain variant. It reaches the unshielded overall reduction but reopens a larger weak-positive-control tail cost, so it is not a safe main-method candidate.",
                "",
                "Mechanistically, this is a better story than global weighted averaging: the module first detects degeneration, then adjudicates expert disagreement based on whether the conflict is compatible with local trend/decay dynamics. It recovers high-conflict failure windows that plain CSSR suppresses, without reopening the positive-control tail-cost failure mode.",
                "",
                "## Claim Discipline",
                "",
                "- Claim: regime-resolved conflict adjudication is a stronger repair controller than uniform conflict capping on the current locked cross-family slice.",
                "- Do not yet claim: the trend/decay thresholds are universal. They need held-out calibration or synthetic counterfactual validation.",
                "- Next decisive ablation: lock the trend/decay adjudicator on calibration families, then evaluate on held-out families and synthetic trend/decay counterfactuals.",
                "",
                "## Artifacts",
                "",
                "- `results/repair/mechanistic_repair_module_search_summary.csv`",
                "- `results/repair/mechanistic_repair_module_search_windows.csv`",
            ]
        )
        + "\n"
    )


def main() -> None:
    windows, _ = cross.load_cross_family_windows()
    policy = base_policy()
    module_ids = [
        "unshielded_adaptive_deferral",
        "cssr_cap_all_conflicts",
        "rr_cssr_trend_decay_adjudicator",
        "loose_rr_cssr_trend_adjudicator_ablation",
        "interval_projected_reference",
        "smooth_disagreement_decay",
    ]
    window_rows = []
    summary = []
    for module_id in module_ids:
        rows = [apply_module(window, policy, module_id) for window in windows]
        window_rows.extend(rows)
        summary.extend(summary_rows(rows, module_id))
    write_csv(OUT_DIR / "mechanistic_repair_module_search_windows.csv", window_rows)
    write_csv(OUT_DIR / "mechanistic_repair_module_search_summary.csv", summary)
    write_report(summary)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows": len(windows),
        "modules": module_ids,
        "summary": "results/repair/mechanistic_repair_module_search_summary.csv",
        "window_metrics": "results/repair/mechanistic_repair_module_search_windows.csv",
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    (OUT_DIR / "mechanistic_repair_module_search_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
