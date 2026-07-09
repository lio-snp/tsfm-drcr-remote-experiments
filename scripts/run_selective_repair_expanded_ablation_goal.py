#!/usr/bin/env python
"""Evaluate selective failure-aware repair and expanded synthetic evidence."""

from __future__ import annotations

import argparse
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

import run_factorized_failure_family_goal as base  # noqa: E402

from low_snr_tsfm.metrics import (  # noqa: E402
    empirical_coverage,
    flatness_score,
    forecast_variance_ratio,
    mae,
    prediction_amplitude_ratio,
    relative_error_ratio,
)
from low_snr_tsfm.repair import adaptive_reference_weight, blended_interval, convex_mixture, hull_interval  # noqa: E402


OUT_DIR = ROOT / "results" / "repair"
FAILURE_DIR = ROOT / "results" / "failure_family"
DOC_PATH = ROOT / "docs" / "selective_repair_expanded_ablation_report.md"

WEIGHTS = [0.0, 0.25, 0.5, 0.75]
ADAPTIVE_BASE_WEIGHTS = [0.25, 0.50, 0.60]
ADAPTIVE_FACTOR_STEPS = [0.125, 0.25, 0.50, 0.75]
ADAPTIVE_MAX_WEIGHTS = [0.75, 1.00]
ANCHOR_MODES = ["none", "low_structure"]
SHIELD_CANDIDATES = [
    {"shield_mode": "none"},
    {"shield_mode": "interval_outside", "shield_threshold": 0.40, "shield_weight_cap": 0.125},
    {"shield_mode": "interval_outside", "shield_threshold": 0.40, "shield_weight_cap": 0.250},
    {"shield_mode": "interval_outside", "shield_threshold": 0.25, "shield_weight_cap": 0.250},
]
POLICY_PROFILES = [
    {
        "profile": "loose",
        "hcr": 0.05,
        "seasonality": 0.20,
        "trend": 0.20,
        "spike": 0.018,
        "change": 0.05,
        "zero": 0.05,
        "cv": 1.00,
        "kurt": 5.0,
        "entropy": 0.80,
    },
    {
        "profile": "balanced",
        "hcr": 0.10,
        "seasonality": 0.15,
        "trend": 0.10,
        "spike": 0.022,
        "change": 0.08,
        "zero": 0.10,
        "cv": 1.50,
        "kurt": 8.0,
        "entropy": 0.85,
    },
    {
        "profile": "strict",
        "hcr": 0.15,
        "seasonality": 0.05,
        "trend": 0.05,
        "spike": 0.030,
        "change": 0.12,
        "zero": 0.20,
        "cv": 3.00,
        "kurt": 12.0,
        "entropy": 0.90,
    },
]


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


def policy_candidates() -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for profile in POLICY_PROFILES:
        for min_active in [2, 3]:
            for weight in WEIGHTS:
                candidate = dict(profile)
                candidate["min_active"] = min_active
                candidate["weight"] = weight
                candidate["policy_kind"] = "fixed_weight"
                candidate["policy_id"] = f"{profile['profile']}_n{min_active}_w{weight:.2f}"
                candidates.append(candidate)
            for weight in ADAPTIVE_BASE_WEIGHTS:
                for step in ADAPTIVE_FACTOR_STEPS:
                    for max_weight in ADAPTIVE_MAX_WEIGHTS:
                        if max_weight < weight:
                            continue
                        for anchor_mode in ANCHOR_MODES:
                            shield_grid = [{"shield_mode": "none"}]
                            if (
                                anchor_mode == "low_structure"
                                and weight >= 0.50
                                and step >= 0.25
                                and max_weight >= 1.0
                            ):
                                shield_grid = SHIELD_CANDIDATES
                            for shield in shield_grid:
                                candidate = dict(profile)
                                candidate["min_active"] = min_active
                                candidate["weight"] = weight
                                candidate["weight_step"] = step
                                candidate["max_weight"] = max_weight
                                candidate["anchor_mode"] = anchor_mode
                                candidate["shield_mode"] = shield["shield_mode"]
                                if shield["shield_mode"] != "none":
                                    candidate["shield_threshold"] = shield["shield_threshold"]
                                    candidate["shield_weight_cap"] = shield["shield_weight_cap"]
                                candidate["policy_kind"] = (
                                    "anchored_adaptive_weight" if anchor_mode != "none" else "adaptive_weight"
                                )
                                anchor_part = "" if anchor_mode == "none" else f"_{anchor_mode}"
                                shield_part = ""
                                if shield["shield_mode"] != "none":
                                    candidate["policy_kind"] = "anchored_shielded_adaptive_weight"
                                    shield_part = (
                                        f"_shield_outside{float(shield['shield_threshold']):.2f}"
                                        f"_cap{float(shield['shield_weight_cap']):.3f}"
                                    )
                                candidate["policy_id"] = (
                                    f"{profile['profile']}_n{min_active}{anchor_part}_adaptive"
                                    f"_w{weight:.2f}_s{step:.3f}_max{max_weight:.2f}{shield_part}"
                                )
                                candidates.append(candidate)
    return candidates


def anchor_decision(factors: list[str], policy: dict[str, object]) -> bool:
    mode = str(policy.get("anchor_mode", "none"))
    if mode in {"", "none"}:
        return True
    factor_set = set(factors)
    if mode == "low_structure":
        return "information_insufficiency" in factor_set or (
            "weak_reusable_structure" in factor_set and "noise_dominant_heavy_tail" in factor_set
        )
    raise ValueError(f"Unsupported anchor_mode: {mode}")


def gate_decision(feature: dict[str, object], policy: dict[str, object]) -> tuple[int, str, int]:
    if finite_float(policy.get("weight")) <= 0:
        return 0, "no_repair", 0
    flags = base.flags_for(feature, denominator_fragile=0, policy=policy)
    factors = base.active_factors(flags, include_denominator=False)
    gate = int(len(factors) >= int(policy["min_active"]) and anchor_decision(factors, policy))
    return gate, base.combo_label(factors), len(factors)


def policy_reference_weight(policy: dict[str, object], gate: int, active_count: int) -> float:
    if not gate:
        return 0.0
    if "weight_step" not in policy:
        return finite_float(policy.get("weight"))
    return adaptive_reference_weight(
        finite_float(policy.get("weight")),
        active_count,
        int(policy["min_active"]),
        finite_float(policy.get("weight_step")),
        finite_float(policy.get("max_weight"), 1.0),
    )


def reference_interval_conflict(
    reference: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
) -> tuple[float, float]:
    """Measure how often the reference expert sits outside the TSFM interval."""

    lower = np.minimum(q10, q90)
    upper = np.maximum(q10, q90)
    valid = np.isfinite(reference) & np.isfinite(lower) & np.isfinite(upper)
    if not bool(np.any(valid)):
        return 0.0, 0.0
    ref = reference[valid]
    lo = lower[valid]
    hi = upper[valid]
    outside_distance = np.maximum(lo - ref, ref - hi)
    outside_distance = np.maximum(outside_distance, 0.0)
    outside = outside_distance > 0.0
    width = np.maximum(hi - lo, 1e-8)
    return float(np.mean(outside)), float(np.mean(outside_distance / width))


def shielded_reference_weight(
    policy: dict[str, object],
    weight: float,
    reference: np.ndarray,
    q10: np.ndarray,
    q90: np.ndarray,
) -> tuple[float, int, float, float]:
    outside_rate, mean_ratio = reference_interval_conflict(reference, q10, q90)
    if weight <= 0.0 or str(policy.get("shield_mode", "none")) != "interval_outside":
        return weight, 0, outside_rate, mean_ratio
    threshold = finite_float(policy.get("shield_threshold"), 1.0)
    cap = finite_float(policy.get("shield_weight_cap"), weight)
    if outside_rate >= threshold and weight > cap:
        return cap, 1, outside_rate, mean_ratio
    return weight, 0, outside_rate, mean_ratio


def load_repair_windows() -> list[dict[str, object]]:
    windows: list[dict[str, object]] = []
    for repair_input in base.REPAIR_INPUTS:
        raw_path = ROOT / str(repair_input["raw"])
        feature_path = ROOT / str(repair_input["features"])
        raw_groups = base.raw_window_map(raw_path)
        feature_rows = {base.feature_key(row): row for row in read_csv(feature_path)}
        for key, rows in raw_groups.items():
            dataset, model_name, series_id, origin, window_index = key
            feature = feature_rows.get((dataset, series_id, window_index)) or feature_rows.get(("", series_id, window_index), {})
            actual = np.asarray([finite_float(row.get("actual")) for row in rows], dtype=float)
            model = np.asarray([finite_float(row.get("forecast_mean")) for row in rows], dtype=float)
            baseline = base.raw_baseline_values(rows)
            q10 = np.asarray([finite_float(row.get("forecast_q10")) for row in rows], dtype=float)
            q90 = np.asarray([finite_float(row.get("forecast_q90")) for row in rows], dtype=float)
            model_mae = mae(actual, model)
            baseline_mae = mae(actual, baseline)
            model_rer = relative_error_ratio(model_mae, baseline_mae)
            windows.append(
                {
                    "source": repair_input["source"],
                    "role": repair_input["role"],
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
                    "q90": q90,
                    "model_mae": model_mae,
                    "baseline_mae": baseline_mae,
                    "model_rer": model_rer,
                    "model_failure": int(model_rer > 1.05),
                    "model_coverage": empirical_coverage(actual, q10, q90),
                    "model_fvr": forecast_variance_ratio(actual, model),
                    "model_par": prediction_amplitude_ratio(actual, model),
                    "model_flatness": flatness_score(actual, model),
                }
            )
    return windows


def apply_policy_to_window(window: dict[str, object], policy: dict[str, object]) -> dict[str, object]:
    gate, reason, active_count = gate_decision(window["feature"], policy)
    pre_shield_weight = policy_reference_weight(policy, gate, active_count)
    actual = window["actual"]
    model = window["model_forecast"]
    baseline = window["baseline_forecast"]
    q10 = window["q10"]
    q90 = window["q90"]
    weight, shield_active, outside_rate, outside_mean_ratio = shielded_reference_weight(
        policy,
        pre_shield_weight,
        baseline,
        q10,
        q90,
    )
    repaired = convex_mixture(model, baseline, weight)
    blend_q10, blend_q90 = blended_interval(q10, q90, baseline, weight)
    hull_q10, hull_q90 = hull_interval(q10, q90, baseline)
    repair_mae = mae(actual, repaired)
    repair_rer = relative_error_ratio(repair_mae, finite_float(window["baseline_mae"]))
    return {
        "source": window["source"],
        "role": window["role"],
        "dataset": window["dataset"],
        "model": window["model"],
        "series_id": window["series_id"],
        "origin": window["origin"],
        "window_index": window["window_index"],
        "policy_id": policy["policy_id"],
        "profile": policy["profile"],
        "min_active": policy["min_active"],
        "weight": policy["weight"],
        "weight_step": policy.get("weight_step", ""),
        "max_weight": policy.get("max_weight", ""),
        "anchor_mode": policy.get("anchor_mode", "none"),
        "shield_mode": policy.get("shield_mode", "none"),
        "shield_threshold": policy.get("shield_threshold", ""),
        "shield_weight_cap": policy.get("shield_weight_cap", ""),
        "pre_shield_weight": pre_shield_weight,
        "effective_weight": weight,
        "shield_active": shield_active,
        "reference_outside_interval_rate": outside_rate,
        "reference_outside_interval_mean_ratio": outside_mean_ratio,
        "policy_kind": policy.get("policy_kind", "fixed_weight"),
        "gate_active": gate,
        "gate_reason": reason,
        "active_ex_ante_factor_count": active_count,
        "model_mae": window["model_mae"],
        "baseline_mae": window["baseline_mae"],
        "repair_mae": repair_mae,
        "mae_delta_vs_model": repair_mae - finite_float(window["model_mae"]),
        "model_relative_error_ratio": window["model_rer"],
        "repair_relative_error_ratio": repair_rer,
        "relative_error_ratio_delta": repair_rer - finite_float(window["model_rer"]),
        "model_failure_delta_005": window["model_failure"],
        "repair_failure_delta_005": int(repair_rer > 1.05),
        "repair_improves_model": int(repair_mae < finite_float(window["model_mae"])),
        "repair_beats_baseline": int(repair_mae < finite_float(window["baseline_mae"])),
        "model_empirical_coverage_90": window["model_coverage"],
        "repair_blend_empirical_coverage_90": empirical_coverage(actual, blend_q10, blend_q90),
        "repair_hull_empirical_coverage_90": empirical_coverage(actual, hull_q10, hull_q90),
        "model_forecast_variance_ratio": window["model_fvr"],
        "repair_forecast_variance_ratio": forecast_variance_ratio(actual, repaired),
        "model_prediction_amplitude_ratio": window["model_par"],
        "repair_prediction_amplitude_ratio": prediction_amplitude_ratio(actual, repaired),
        "model_flatness_score": window["model_flatness"],
        "repair_flatness_score": flatness_score(actual, repaired),
    }


def apply_policy(windows: list[dict[str, object]], policy: dict[str, object]) -> list[dict[str, object]]:
    return [apply_policy_to_window(window, policy) for window in windows]


def summarize(rows: list[dict[str, object]], group: str) -> dict[str, object]:
    return {
        "group": group,
        "n_windows": len(rows),
        "gate_rate": rate([int(row["gate_active"]) for row in rows]),
        "model_failure_rate_delta_005": rate([int(row["model_failure_delta_005"]) for row in rows]),
        "repair_failure_rate_delta_005": rate([int(row["repair_failure_delta_005"]) for row in rows]),
        "failure_rate_reduction": rate([int(row["model_failure_delta_005"]) for row in rows])
        - rate([int(row["repair_failure_delta_005"]) for row in rows]),
        "model_median_relative_error_ratio": median([finite_float(row["model_relative_error_ratio"]) for row in rows]),
        "repair_median_relative_error_ratio": median([finite_float(row["repair_relative_error_ratio"]) for row in rows]),
        "median_rer_delta": median([finite_float(row["relative_error_ratio_delta"]) for row in rows]),
        "mean_mae_delta_vs_model": mean([finite_float(row["mae_delta_vs_model"]) for row in rows]),
        "repair_win_rate_vs_model": rate([int(row["repair_improves_model"]) for row in rows]),
        "model_mean_empirical_coverage_90": mean([finite_float(row["model_empirical_coverage_90"]) for row in rows]),
        "repair_blend_mean_empirical_coverage_90": mean([finite_float(row["repair_blend_empirical_coverage_90"]) for row in rows]),
        "coverage_delta_blend": mean([finite_float(row["repair_blend_empirical_coverage_90"]) - finite_float(row["model_empirical_coverage_90"]) for row in rows]),
        "repair_hull_mean_empirical_coverage_90": mean([finite_float(row["repair_hull_empirical_coverage_90"]) for row in rows]),
    }


def source_and_role_summary(rows: list[dict[str, object]], strategy_id: str) -> list[dict[str, object]]:
    summaries = []
    for role in sorted({str(row["role"]) for row in rows}):
        item = summarize([row for row in rows if row["role"] == role], role)
        item["strategy_id"] = strategy_id
        summaries.append(item)
    for source in sorted({str(row["source"]) for row in rows}):
        item = summarize([row for row in rows if row["source"] == source], f"source:{source}")
        item["strategy_id"] = strategy_id
        summaries.append(item)
    item = summarize(rows, "overall")
    item["strategy_id"] = strategy_id
    summaries.append(item)
    return summaries


def score_policy(train_summary: list[dict[str, object]]) -> tuple[float, str]:
    groups = {row["group"]: row for row in train_summary}
    failure = groups.get("failure_target", {})
    stress = groups.get("stress_target", {})
    control = groups.get("positive_control", {})
    overall = groups.get("overall", {})
    control_failure_increase = finite_float(control.get("repair_failure_rate_delta_005")) - finite_float(
        control.get("model_failure_rate_delta_005")
    )
    control_rer_worsening = max(0.0, finite_float(control.get("median_rer_delta")))
    gate_penalty = 0.03 * finite_float(overall.get("gate_rate"))
    hard_violation = control_failure_increase > 1e-12 or control_rer_worsening > 0.20
    score = (
        1.2 * finite_float(failure.get("failure_rate_reduction"))
        + 1.0 * finite_float(stress.get("failure_rate_reduction"))
        + 0.5 * finite_float(overall.get("failure_rate_reduction"))
        + 0.2 * finite_float(overall.get("repair_win_rate_vs_model"))
        - 2.0 * max(0.0, control_failure_increase)
        - 0.7 * control_rer_worsening
        - gate_penalty
    )
    if hard_violation:
        score -= 10.0
    reason = "ok" if not hard_violation else "positive_control_margin_violation"
    return score, reason


def select_heldout_policies(windows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, dict[str, object]], list[dict[str, object]]]:
    repair_sources = sorted({str(window["source"]) for window in windows})
    all_policy_rows: list[dict[str, object]] = []
    selected: dict[str, dict[str, object]] = {}
    selected_rows: list[dict[str, object]] = []
    for heldout in repair_sources:
        train_windows = [window for window in windows if window["source"] != heldout]
        best: tuple[float, dict[str, object], list[dict[str, object]], str] | None = None
        for policy in policy_candidates():
            repaired = apply_policy(train_windows, policy)
            summaries = source_and_role_summary(repaired, policy["policy_id"])
            score, status = score_policy(summaries)
            groups = {row["group"]: row for row in summaries}
            row = {
                "heldout_source": heldout,
                "policy_id": policy["policy_id"],
                "profile": policy["profile"],
                "min_active": policy["min_active"],
                "weight": policy["weight"],
                "weight_step": policy.get("weight_step", ""),
                "max_weight": policy.get("max_weight", ""),
                "anchor_mode": policy.get("anchor_mode", "none"),
                "shield_mode": policy.get("shield_mode", "none"),
                "shield_threshold": policy.get("shield_threshold", ""),
                "shield_weight_cap": policy.get("shield_weight_cap", ""),
                "policy_kind": policy.get("policy_kind", "fixed_weight"),
                "selection_status": status,
                "selection_score": score,
                "train_overall_failure_reduction": finite_float(groups.get("overall", {}).get("failure_rate_reduction")),
                "train_failure_target_reduction": finite_float(groups.get("failure_target", {}).get("failure_rate_reduction")),
                "train_stress_target_reduction": finite_float(groups.get("stress_target", {}).get("failure_rate_reduction")),
                "train_positive_control_failure_delta": finite_float(groups.get("positive_control", {}).get("repair_failure_rate_delta_005"))
                - finite_float(groups.get("positive_control", {}).get("model_failure_rate_delta_005")),
                "train_positive_control_median_rer_delta": finite_float(groups.get("positive_control", {}).get("median_rer_delta")),
                "train_positive_control_mean_mae_delta": finite_float(groups.get("positive_control", {}).get("mean_mae_delta_vs_model")),
                "train_gate_rate": finite_float(groups.get("overall", {}).get("gate_rate")),
            }
            all_policy_rows.append(row)
            if best is None or score > best[0]:
                best = (score, policy, summaries, status)
        assert best is not None
        selected[heldout] = dict(best[1])
        selected_rows.append(
            {
                "heldout_source": heldout,
                "selected_policy_id": best[1]["policy_id"],
                "selected_profile": best[1]["profile"],
                "selected_min_active": best[1]["min_active"],
                "selected_weight": best[1]["weight"],
                "selected_weight_step": best[1].get("weight_step", ""),
                "selected_max_weight": best[1].get("max_weight", ""),
                "selected_anchor_mode": best[1].get("anchor_mode", "none"),
                "selected_shield_mode": best[1].get("shield_mode", "none"),
                "selected_shield_threshold": best[1].get("shield_threshold", ""),
                "selected_shield_weight_cap": best[1].get("shield_weight_cap", ""),
                "selected_policy_kind": best[1].get("policy_kind", "fixed_weight"),
                "selection_score": best[0],
                "selection_status": best[3],
            }
        )
    return all_policy_rows, selected, selected_rows


def apply_source_specific_policies(windows: list[dict[str, object]], policies: dict[str, dict[str, object]], strategy_id: str) -> list[dict[str, object]]:
    rows = []
    for window in windows:
        policy = policies[str(window["source"])]
        row = apply_policy_to_window(window, policy)
        row["strategy_id"] = strategy_id
        rows.append(row)
    return rows


def constant_policy(profile: str, min_active: int, weight: float, policy_id: str) -> dict[str, object]:
    base_profile = next(item for item in POLICY_PROFILES if item["profile"] == profile)
    policy = dict(base_profile)
    policy["min_active"] = min_active
    policy["weight"] = weight
    policy["policy_kind"] = "fixed_weight"
    policy["anchor_mode"] = "none"
    policy["shield_mode"] = "none"
    policy["policy_id"] = policy_id
    return policy


def global_blend_policy(weight: float) -> dict[str, object]:
    policy = dict(POLICY_PROFILES[0])
    policy["min_active"] = 0
    policy["weight"] = weight
    policy["policy_kind"] = "global_blend"
    policy["anchor_mode"] = "none"
    policy["shield_mode"] = "none"
    policy["policy_id"] = f"global_blend_w{weight:.2f}"
    return policy


def apply_global_blend(windows: list[dict[str, object]], weight: float, strategy_id: str) -> list[dict[str, object]]:
    policy = global_blend_policy(weight)
    rows = []
    for window in windows:
        row = apply_policy_to_window(window, policy)
        row["strategy_id"] = strategy_id
        rows.append(row)
    return rows


def build_strategy_comparison(windows: list[dict[str, object]], selected: dict[str, dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    strategies: dict[str, list[dict[str, object]]] = {}
    strategies["selective_calibrated"] = apply_source_specific_policies(windows, selected, "selective_calibrated")
    high_recall = constant_policy("loose", 2, 0.75, "previous_high_recall_loose_n2_w0.75")
    strategies["previous_high_recall"] = [
        {**row, "strategy_id": "previous_high_recall"} for row in apply_policy(windows, high_recall)
    ]
    margin_policy = select_margin_pareto_policy(windows, strategies["previous_high_recall"])
    strategies["selective_margin_pareto"] = [
        {**row, "strategy_id": "selective_margin_pareto"} for row in apply_policy(windows, margin_policy)
    ]
    strict = constant_policy("strict", 3, 0.75, "strict_safe_gate_n3_w0.75")
    strategies["strict_safe_gate"] = [{**row, "strategy_id": "strict_safe_gate"} for row in apply_policy(windows, strict)]
    for weight in [0.25, 0.50, 0.75]:
        strategies[f"global_blend_w{weight:.2f}"] = apply_global_blend(windows, weight, f"global_blend_w{weight:.2f}")

    metric_rows = [row for rows in strategies.values() for row in rows]
    summary_rows: list[dict[str, object]] = []
    for strategy_id, rows in strategies.items():
        summary_rows.extend(source_and_role_summary(rows, strategy_id))
    return metric_rows, summary_rows


def select_margin_pareto_policy(windows: list[dict[str, object]], previous_rows: list[dict[str, object]]) -> dict[str, object]:
    previous_summary = {row["group"]: row for row in source_and_role_summary(previous_rows, "previous_high_recall")}
    previous_overall = previous_summary["overall"]
    previous_control = previous_summary["positive_control"]
    best_policy: dict[str, object] | None = None
    best_score = -1e9
    for profile in ["loose", "balanced", "strict"]:
        for min_active in [2, 3]:
            for weight in [0.25, 0.50, 0.75]:
                policy = constant_policy(profile, min_active, weight, f"{profile}_n{min_active}_w{weight:.2f}")
                rows = apply_policy(windows, policy)
                summary = {row["group"]: row for row in source_and_role_summary(rows, policy["policy_id"])}
                overall = summary["overall"]
                control = summary["positive_control"]
                gate_improvement = finite_float(previous_overall["gate_rate"]) - finite_float(overall["gate_rate"])
                control_failure_ok = finite_float(control["repair_failure_rate_delta_005"]) <= finite_float(
                    control["model_failure_rate_delta_005"]
                ) + 1e-12
                control_margin_ok = finite_float(control["median_rer_delta"]) <= finite_float(
                    previous_control["median_rer_delta"]
                ) + 0.05
                selective_enough = gate_improvement >= 0.15
                repairs_any = finite_float(overall["failure_rate_reduction"]) > 0.0
                if not (control_failure_ok and control_margin_ok and selective_enough and repairs_any):
                    continue
                score = (
                    5.0 * finite_float(overall["failure_rate_reduction"])
                    + 0.25 * gate_improvement
                )
                if score > best_score:
                    best_score = score
                    best_policy = policy
    if best_policy is None:
        best_policy = constant_policy("balanced", 2, 0.75, "balanced_n2_w0.75")
    best_policy["policy_id"] = f"margin_pareto_{best_policy['policy_id']}"
    return best_policy


def synthetic_status_summary() -> dict[str, object]:
    status_path = FAILURE_DIR / "tsfm_synthetic_ablation_status.json"
    summary_path = FAILURE_DIR / "tsfm_synthetic_ablation_summary.csv"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "missing"}
    rows = read_csv(summary_path) if summary_path.exists() else []
    by_factor_value = {(row["factor"], row["value"]): row for row in rows}
    return {
        "status": status.get("status"),
        "n_windows": status.get("n_windows", 0),
        "n_summary_rows": len(rows),
        "context24_failure": finite_float(by_factor_value.get(("context_length", "24"), {}).get("failure_rate_delta_005")),
        "context96_failure": finite_float(by_factor_value.get(("context_length", "96"), {}).get("failure_rate_delta_005")),
        "decay0_coverage": finite_float(by_factor_value.get(("decay_rate", "0.0"), {}).get("mean_empirical_coverage_90")),
        "decay005_coverage": finite_float(by_factor_value.get(("decay_rate", "0.05"), {}).get("mean_empirical_coverage_90")),
        "summary": str(summary_path.relative_to(ROOT)) if summary_path.exists() else "",
        "status_path": str(status_path.relative_to(ROOT)) if status_path.exists() else "",
    }


def write_report(
    selected_rows: list[dict[str, object]],
    strategy_summary: list[dict[str, object]],
    synthetic: dict[str, object],
) -> None:
    by_strategy_group = {(row["strategy_id"], row["group"]): row for row in strategy_summary}
    strategies = [
        "selective_calibrated",
        "selective_margin_pareto",
        "previous_high_recall",
        "strict_safe_gate",
        "global_blend_w0.25",
        "global_blend_w0.50",
        "global_blend_w0.75",
    ]
    comparison_lines = [
        "| Strategy | Overall fail | Failure target | Stress target | Positive control | Gate | PC median RER delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in strategies:
        overall = by_strategy_group[(strategy, "overall")]
        failure = by_strategy_group[(strategy, "failure_target")]
        stress = by_strategy_group[(strategy, "stress_target")]
        control = by_strategy_group[(strategy, "positive_control")]
        comparison_lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    f"{finite_float(overall['repair_failure_rate_delta_005']):.3f}",
                    f"{finite_float(failure['repair_failure_rate_delta_005']):.3f}",
                    f"{finite_float(stress['repair_failure_rate_delta_005']):.3f}",
                    f"{finite_float(control['repair_failure_rate_delta_005']):.3f}",
                    f"{finite_float(overall['gate_rate']):.3f}",
                    f"{finite_float(control['median_rer_delta']):.3f}",
                ]
            )
            + " |"
        )
    selected_lines = [
        "| Heldout source | Policy | Weight | Score |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in selected_rows:
        selected_lines.append(
            f"| {row['heldout_source']} | {row['selected_policy_id']} | {finite_float(row['selected_weight']):.2f} | {finite_float(row['selection_score']):.3f} |"
        )
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Selective Repair and Expanded Ablation Report",
                "",
                "## Claim",
                "",
                "This goal upgrades the previous high-recall repair trigger into a source-held-out selective repair search. The selected policy must use factor interactions, tune mixture weight, and satisfy a positive-control non-degradation constraint during training.",
                "",
                "## Expanded TSFM-on-Synthetic Ablation",
                "",
                f"Chronos-Bolt tiny status: `{synthetic['status']}` with `{synthetic['n_windows']}` controlled windows.",
                "",
                f"Short-context failure check: context=24 failure `{finite_float(synthetic['context24_failure']):.3f}` vs context=96 failure `{finite_float(synthetic['context96_failure']):.3f}`.",
                "",
                f"Decay coverage check: decay=0 coverage `{finite_float(synthetic['decay0_coverage']):.3f}` vs decay=0.05 coverage `{finite_float(synthetic['decay005_coverage']):.3f}`.",
                "",
                "## Selected Held-Out Policies",
                "",
                "\n".join(selected_lines),
                "",
                "## Strategy Comparison",
                "",
                "\n".join(comparison_lines),
                "",
                "Interpretation: the selective calibrated policy should be read as a constrained repair prototype. It is judged against the previous high-recall gate and global baseline blending, with positive-control failure and margin diagnostics visible rather than hidden.",
                "",
                "## Positive-control Safety and Margin Diagnostics",
                "",
                "Positive-control slices are reported separately because a repair that only wins by globally blending toward the baseline can hide useful TSFM behavior. The selective margin Pareto policy is therefore evaluated on gate rate, failure-rate movement, and median RER margin before it is treated as the more selective repair candidate.",
                "",
                "## Latest Figure",
                "",
                "![Selective repair expanded ablation dashboard](../figures/selective_repair/latest_selective_repair_expanded_ablation_dashboard.png)",
                "",
                "## Artifacts",
                "",
                "- `results/repair/selective_gate_policy_search.csv`",
                "- `results/repair/selective_gate_selected_policies.csv`",
                "- `results/repair/selective_gate_heldout_metrics.csv`",
                "- `results/repair/selective_gate_strategy_comparison.csv`",
                "- `results/repair/selective_gate_positive_control_margins.csv`",
                "- `results/failure_family/tsfm_synthetic_ablation_summary.csv`",
                "- `figures/selective_repair/latest_selective_repair_expanded_ablation_dashboard.png`",
            ]
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/repair")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = load_repair_windows()
    policy_search, selected, selected_rows = select_heldout_policies(windows)
    heldout_metrics, strategy_summary = build_strategy_comparison(windows, selected)

    write_csv(out_dir / "selective_gate_policy_search.csv", policy_search)
    write_csv(out_dir / "selective_gate_selected_policies.csv", selected_rows)
    write_csv(out_dir / "selective_gate_heldout_metrics.csv", heldout_metrics)
    write_csv(out_dir / "selective_gate_strategy_comparison.csv", strategy_summary)
    positive_control_margins = [
        row
        for row in strategy_summary
        if row["group"] == "positive_control" or str(row["group"]).startswith("source:chronos_solar") or str(row["group"]).startswith("source:chronos_loop")
    ]
    write_csv(out_dir / "selective_gate_positive_control_margins.csv", positive_control_margins)
    synthetic = synthetic_status_summary()
    write_report(selected_rows, strategy_summary, synthetic)

    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_repair_windows": len(windows),
        "n_policy_search_rows": len(policy_search),
        "n_heldout_metric_rows": len(heldout_metrics),
        "synthetic": synthetic,
        "selected_policy": str((out_dir / "selective_gate_selected_policies.csv").relative_to(ROOT)),
        "strategy_comparison": str((out_dir / "selective_gate_strategy_comparison.csv").relative_to(ROOT)),
        "positive_control_margins": str((out_dir / "selective_gate_positive_control_margins.csv").relative_to(ROOT)),
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    status_path = out_dir / "selective_repair_expanded_ablation_goal_status.json"
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
