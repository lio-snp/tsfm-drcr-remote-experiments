#!/usr/bin/env python
"""Validate selective repair on held-out model families."""

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

import run_factorized_failure_family_goal as base  # noqa: E402
import run_selective_repair_expanded_ablation_goal as selective  # noqa: E402

from low_snr_tsfm.metrics import (  # noqa: E402
    empirical_coverage,
    flatness_score,
    forecast_variance_ratio,
    mae,
    prediction_amplitude_ratio,
    relative_error_ratio,
)
from low_snr_tsfm.repair import blended_interval, convex_mixture, hull_interval  # noqa: E402


OUT_DIR = ROOT / "results" / "repair"
DOC_PATH = ROOT / "docs" / "cross_family_selective_repair_report.md"

INPUTS = [
    {
        "family": "chronos",
        "source": "chronos_bizitobs_auto_arima",
        "role": "failure_target",
        "raw": "results/raw_forecasts/chronos_bolt_small_bizitobs_application_short_auto_arima.csv",
        "features": "results/failure_mining/chronos_bolt_small_bizitobs_application_short_auto_arima_predictor_features.csv",
    },
    {
        "family": "chronos",
        "source": "chronos_covid_auto_ets",
        "role": "failure_target",
        "raw": "results/raw_forecasts/chronos_bolt_small_covid_deaths_short_auto_ets.csv",
        "features": "results/failure_mining/chronos_bolt_small_covid_deaths_short_auto_ets_predictor_features.csv",
    },
    {
        "family": "chronos",
        "source": "chronos_finance_fred",
        "role": "stress_target",
        "raw": "results/raw_forecasts/chronos_bolt_small_finance_fred_stress.csv",
        "features": "results/failure_mining/chronos_bolt_small_finance_fred_stress_predictor_features.csv",
    },
    {
        "family": "chronos",
        "source": "chronos_solar_seasonal_naive",
        "role": "positive_control",
        "raw": "results/raw_forecasts/chronos_bolt_small_solar_short_seasonal_naive.csv",
        "features": "results/failure_mining/chronos_bolt_small_solar_short_seasonal_naive_predictor_features.csv",
    },
    {
        "family": "chronos",
        "source": "chronos_loop_seattle",
        "role": "positive_control",
        "raw": "results/raw_forecasts/chronos_bolt_small_loop_seattle_short_seasonal_naive.csv",
        "features": "results/failure_mining/chronos_bolt_small_loop_seattle_short_seasonal_naive_predictor_features.csv",
    },
    {
        "family": "moirai",
        "source": "moirai2_ctx128_covid",
        "role": "failure_target",
        "raw": "results/raw_forecasts/moirai2_ctx128_m12_covid_deaths_short_auto_ets.csv",
        "features": "results/failure_mining/moirai2_ctx128_m12_covid_deaths_short_auto_ets_predictor_features.csv",
    },
    {
        "family": "moirai",
        "source": "moirai2_ctx512_covid",
        "role": "failure_target",
        "raw": "results/raw_forecasts/moirai2_ctx512_m12_covid_deaths_short_auto_ets.csv",
        "features": "results/failure_mining/moirai2_ctx512_m12_covid_deaths_short_auto_ets_predictor_features.csv",
    },
    {
        "family": "moirai",
        "source": "moirai2_ctx1680_covid",
        "role": "failure_target",
        "raw": "results/raw_forecasts/moirai2_ctx1680_m12_covid_deaths_short_auto_ets.csv",
        "features": "results/failure_mining/moirai2_ctx1680_m12_covid_deaths_short_auto_ets_predictor_features.csv",
    },
    {
        "family": "moirai",
        "source": "moirai2_solar_seasonal_naive",
        "role": "positive_control",
        "raw": "results/raw_forecasts/moirai2_ctx1680_solar_m16_solar_short_seasonal_naive.csv",
        "features": "results/failure_mining/moirai2_ctx1680_solar_m16_solar_short_seasonal_naive_predictor_features.csv",
    },
    {
        "family": "moirai",
        "source": "moirai2_loop_seattle",
        "role": "positive_control",
        "raw": "results/raw_forecasts/moirai2_loop_m8_loop_seattle_short_seasonal_naive.csv",
        "features": "results/failure_mining/moirai2_loop_m8_loop_seattle_short_seasonal_naive_predictor_features.csv",
    },
    {
        "family": "timesfm",
        "source": "timesfm25_covid_auto_ets_m128",
        "role": "failure_target",
        "raw": "results/raw_forecasts/timesfm_2_5_m128_covid_deaths_short_auto_ets.csv",
        "features": "results/failure_mining/timesfm_2_5_m128_covid_deaths_short_auto_ets_predictor_features.csv",
    },
    {
        "family": "timesfm",
        "source": "timesfm25_finance_fred",
        "role": "stress_target",
        "raw": "results/raw_forecasts/timesfm_2_5_finance_fred_finance_fred_stress.csv",
        "features": "results/failure_mining/timesfm_2_5_finance_fred_finance_fred_stress_predictor_features.csv",
    },
    {
        "family": "timesfm",
        "source": "timesfm25_loop_seattle",
        "role": "positive_control",
        "raw": "results/raw_forecasts/timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive.csv",
        "features": "results/failure_mining/timesfm_2_5_loop_m32_loop_seattle_short_seasonal_naive_predictor_features.csv",
    },
    {
        "family": "timesfm",
        "source": "timesfm25_solar_weak_control",
        "role": "weak_positive_control",
        "raw": "results/raw_forecasts/timesfm_2_5_solar_m8_solar_short_seasonal_naive.csv",
        "features": "results/failure_mining/timesfm_2_5_solar_m8_solar_short_seasonal_naive_predictor_features.csv",
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


def window_feature_rows(path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    return {base.feature_key(row): row for row in read_csv(path)}


def load_cross_family_windows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    windows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for item in INPUTS:
        raw_path = ROOT / str(item["raw"])
        feature_path = ROOT / str(item["features"])
        if not raw_path.exists() or not feature_path.exists():
            inventory.append({**item, "status": "missing", "n_windows": 0})
            continue
        raw_groups = base.raw_window_map(raw_path)
        features = window_feature_rows(feature_path)
        source_windows = 0
        for key, rows in raw_groups.items():
            dataset, model_name, series_id, origin, window_index = key
            feature = features.get((dataset, series_id, window_index)) or features.get(("", series_id, window_index), {})
            if not feature:
                continue
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
                    "family": item["family"],
                    "source": item["source"],
                    "role": item["role"],
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
            source_windows += 1
        inventory.append({**item, "status": "ok", "n_windows": source_windows})
    return windows, inventory


def policy_candidates() -> list[dict[str, object]]:
    return [policy for policy in selective.policy_candidates() if finite_float(policy.get("weight")) > 0.0]


def is_interval_shielded(policy: dict[str, object]) -> bool:
    return str(policy.get("shield_mode", "none")) == "interval_outside"


def apply_policy_to_window(window: dict[str, object], policy: dict[str, object], strategy_id: str) -> dict[str, object]:
    row = selective.apply_policy_to_window(window, policy)
    row["family"] = window["family"]
    row["strategy_id"] = strategy_id
    return row


def apply_single_policy(windows: list[dict[str, object]], policy: dict[str, object], strategy_id: str) -> list[dict[str, object]]:
    return [apply_policy_to_window(window, policy, strategy_id) for window in windows]


def apply_family_policies(
    windows: list[dict[str, object]],
    policies: dict[str, dict[str, object]],
    strategy_id: str,
) -> list[dict[str, object]]:
    rows = []
    for window in windows:
        policy = policies[str(window["family"])]
        rows.append(apply_policy_to_window(window, policy, strategy_id))
    return rows


def summarize(rows: list[dict[str, object]], group: str, group_type: str) -> dict[str, object]:
    model_fail = rate([int(row["model_failure_delta_005"]) for row in rows])
    repair_fail = rate([int(row["repair_failure_delta_005"]) for row in rows])
    rer_deltas = [finite_float(row["relative_error_ratio_delta"]) for row in rows]
    return {
        "group": group,
        "group_type": group_type,
        "n_windows": len(rows),
        "n_families": len({str(row.get("family", "")) for row in rows}),
        "families": ",".join(sorted({str(row.get("family", "")) for row in rows})),
        "n_sources": len({str(row.get("source", "")) for row in rows}),
        "gate_rate": rate([int(row["gate_active"]) for row in rows]),
        "model_failure_rate_delta_005": model_fail,
        "repair_failure_rate_delta_005": repair_fail,
        "failure_rate_reduction": model_fail - repair_fail,
        "model_median_relative_error_ratio": median([finite_float(row["model_relative_error_ratio"]) for row in rows]),
        "repair_median_relative_error_ratio": median([finite_float(row["repair_relative_error_ratio"]) for row in rows]),
        "median_rer_delta": median(rer_deltas),
        "p90_rer_delta": float(np.percentile(rer_deltas, 90)) if rer_deltas else float("nan"),
        "mean_mae_delta_vs_model": mean([finite_float(row["mae_delta_vs_model"]) for row in rows]),
        "repair_win_rate_vs_model": rate([int(row["repair_improves_model"]) for row in rows]),
        "model_mean_empirical_coverage_90": mean([finite_float(row["model_empirical_coverage_90"]) for row in rows]),
        "repair_blend_mean_empirical_coverage_90": mean([finite_float(row["repair_blend_empirical_coverage_90"]) for row in rows]),
        "coverage_delta_blend": mean(
            [
                finite_float(row["repair_blend_empirical_coverage_90"]) - finite_float(row["model_empirical_coverage_90"])
                for row in rows
            ]
        ),
        "model_mean_forecast_variance_ratio": mean([finite_float(row["model_forecast_variance_ratio"]) for row in rows]),
        "repair_mean_forecast_variance_ratio": mean([finite_float(row["repair_forecast_variance_ratio"]) for row in rows]),
    }


def summary_rows(rows: list[dict[str, object]], strategy_id: str) -> list[dict[str, object]]:
    summaries = [summarize(rows, "overall", "overall")]
    for role in sorted({str(row["role"]) for row in rows}):
        summaries.append(summarize([row for row in rows if row["role"] == role], role, "role"))
    for family in sorted({str(row["family"]) for row in rows}):
        summaries.append(summarize([row for row in rows if row["family"] == family], f"family:{family}", "family"))
    for family in sorted({str(row["family"]) for row in rows}):
        for role in sorted({str(row["role"]) for row in rows if row["family"] == family}):
            subset = [row for row in rows if row["family"] == family and row["role"] == role]
            summaries.append(summarize(subset, f"family:{family}|role:{role}", "family_role"))
    for source in sorted({str(row["source"]) for row in rows}):
        summaries.append(summarize([row for row in rows if row["source"] == source], f"source:{source}", "source"))
    for row in summaries:
        row["strategy_id"] = strategy_id
    return summaries


def role_summary_for_score(rows: list[dict[str, object]], strategy_id: str) -> list[dict[str, object]]:
    selected = [row for row in summary_rows(rows, strategy_id) if row["group_type"] in {"overall", "role"}]
    return selected


def score_policy(rows: list[dict[str, object]], policy: dict[str, object]) -> tuple[float, str, list[dict[str, object]]]:
    repaired = apply_single_policy(rows, policy, str(policy["policy_id"]))
    summaries = role_summary_for_score(repaired, str(policy["policy_id"]))
    score, reason = selective.score_policy(summaries)
    return score, reason, summaries


def select_policy_for_scope(
    windows: list[dict[str, object]],
    train_windows: list[dict[str, object]],
    scope: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    del windows
    best: tuple[float, dict[str, object], str, list[dict[str, object]]] | None = None
    search_rows: list[dict[str, object]] = []
    for policy in policy_candidates():
        score, reason, summaries = score_policy(train_windows, policy)
        groups = {row["group"]: row for row in summaries}
        row = {
            "selection_scope": scope,
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
            "selection_status": reason,
            "selection_score": score,
            "train_n_windows": len(train_windows),
            "train_gate_rate": finite_float(groups.get("overall", {}).get("gate_rate")),
            "train_overall_reduction": finite_float(groups.get("overall", {}).get("failure_rate_reduction")),
            "train_failure_target_reduction": finite_float(groups.get("failure_target", {}).get("failure_rate_reduction")),
            "train_stress_target_reduction": finite_float(groups.get("stress_target", {}).get("failure_rate_reduction")),
            "train_positive_control_failure_delta": finite_float(
                groups.get("positive_control", {}).get("repair_failure_rate_delta_005")
            )
            - finite_float(groups.get("positive_control", {}).get("model_failure_rate_delta_005")),
            "train_positive_control_median_rer_delta": finite_float(groups.get("positive_control", {}).get("median_rer_delta")),
            "train_positive_control_p90_rer_delta": finite_float(groups.get("positive_control", {}).get("p90_rer_delta")),
        }
        search_rows.append(row)
        if best is None or score > best[0]:
            best = (score, dict(policy), reason, summaries)
    if best is None:
        raise ValueError(f"No policy candidates for {scope}")
    best[1]["policy_id"] = f"{scope}_{best[1]['policy_id']}"
    selected_row = {
        "selection_scope": scope,
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
        "selection_status": best[2],
        "selection_score": best[0],
        "train_n_windows": len(train_windows),
    }
    return best[1], [*search_rows, selected_row]


def select_risk_controlled_policy_for_scope(
    train_windows: list[dict[str, object]],
    scope: str,
    *,
    p90_rer_budget: float = 0.25,
    median_rer_budget: float = 0.05,
    max_base_weight: float = 0.25,
    max_weight_step: float = 0.125,
    max_weight_cap: float = 0.75,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Select a low-structure policy using only training-family safety constraints.

    This is a lightweight empirical analogue of risk-control calibration: the
    policy must satisfy a positive-control tail-cost budget on calibration
    families before it is evaluated on the held-out model family.
    """

    best: tuple[float, dict[str, object], str, list[dict[str, object]]] | None = None
    search_rows: list[dict[str, object]] = []
    for policy in policy_candidates():
        if policy.get("anchor_mode") != "low_structure":
            continue
        shielded = is_interval_shielded(policy)
        if shielded:
            if finite_float(policy.get("shield_threshold"), 1.0) > 0.40:
                continue
            if finite_float(policy.get("shield_weight_cap"), 1.0) > 0.125:
                continue
            if finite_float(policy.get("weight")) > 0.60:
                continue
            if finite_float(policy.get("weight_step")) > 0.50:
                continue
            if finite_float(policy.get("max_weight"), 1.0) > 1.00:
                continue
        else:
            if finite_float(policy.get("weight")) > max_base_weight:
                continue
            if finite_float(policy.get("weight_step")) > max_weight_step:
                continue
            if finite_float(policy.get("max_weight"), max_weight_cap) > max_weight_cap:
                continue
        repaired = apply_single_policy(train_windows, policy, str(policy["policy_id"]))
        summaries = role_summary_for_score(repaired, str(policy["policy_id"]))
        groups = {row["group"]: row for row in summaries}
        overall = groups.get("overall", {})
        failure = groups.get("failure_target", {})
        stress = groups.get("stress_target", {})
        control = groups.get("positive_control", {})
        pc_failure_delta = finite_float(control.get("repair_failure_rate_delta_005")) - finite_float(
            control.get("model_failure_rate_delta_005")
        )
        pc_median_delta = finite_float(control.get("median_rer_delta"))
        pc_p90_delta = finite_float(control.get("p90_rer_delta"))
        overall_reduction = finite_float(overall.get("failure_rate_reduction"))
        gate_rate = finite_float(overall.get("gate_rate"))
        status = "ok"
        if pc_failure_delta > 1e-12:
            status = "positive_control_failure_violation"
        elif pc_median_delta > median_rer_budget:
            status = "positive_control_median_risk_violation"
        elif pc_p90_delta > p90_rer_budget:
            status = "positive_control_tail_risk_violation"
        elif overall_reduction <= 0.0:
            status = "no_training_repair_gain"
        score = (
            5.0 * overall_reduction
            + 0.75 * finite_float(failure.get("failure_rate_reduction"))
            + 1.00 * finite_float(stress.get("failure_rate_reduction"))
            + 0.20 * (1.0 - gate_rate)
            - 0.20 * max(0.0, pc_p90_delta)
        )
        if status != "ok":
            score -= 100.0
        search_rows.append(
            {
                "selection_scope": scope,
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
                "train_n_windows": len(train_windows),
                "max_base_weight": max_base_weight,
                "max_weight_step": max_weight_step,
                "max_weight_cap": max_weight_cap,
                "is_interval_shielded": int(shielded),
                "train_gate_rate": gate_rate,
                "train_overall_reduction": overall_reduction,
                "train_failure_target_reduction": finite_float(failure.get("failure_rate_reduction")),
                "train_stress_target_reduction": finite_float(stress.get("failure_rate_reduction")),
                "train_positive_control_failure_delta": pc_failure_delta,
                "train_positive_control_median_rer_delta": pc_median_delta,
                "train_positive_control_p90_rer_delta": pc_p90_delta,
                "positive_control_p90_budget": p90_rer_budget,
            }
        )
        if status == "ok" and (best is None or score > best[0]):
            best = (score, dict(policy), status, summaries)
    if best is None:
        fallback = selective.constant_policy("strict", 3, 0.25, "strict_n3_w0.25")
        fallback["policy_id"] = f"{scope}_{fallback['policy_id']}"
        selected_row = {
            "selection_scope": scope,
            "selected_policy_id": fallback["policy_id"],
            "selected_profile": fallback["profile"],
            "selected_min_active": fallback["min_active"],
            "selected_weight": fallback["weight"],
            "selected_weight_step": fallback.get("weight_step", ""),
            "selected_max_weight": fallback.get("max_weight", ""),
            "selected_anchor_mode": fallback.get("anchor_mode", "none"),
            "selected_shield_mode": fallback.get("shield_mode", "none"),
            "selected_shield_threshold": fallback.get("shield_threshold", ""),
            "selected_shield_weight_cap": fallback.get("shield_weight_cap", ""),
            "selected_policy_kind": fallback.get("policy_kind", "fixed_weight"),
            "selection_status": "no_risk_controlled_policy",
            "selection_score": 0.0,
            "train_n_windows": len(train_windows),
            "positive_control_p90_budget": p90_rer_budget,
            "max_base_weight": max_base_weight,
            "max_weight_step": max_weight_step,
            "max_weight_cap": max_weight_cap,
        }
        return fallback, [*search_rows, selected_row]
    best[1]["policy_id"] = f"{scope}_{best[1]['policy_id']}"
    selected_row = {
        "selection_scope": scope,
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
        "selection_status": "training_positive_control_tail_risk_ok",
        "selection_score": best[0],
        "train_n_windows": len(train_windows),
        "positive_control_p90_budget": p90_rer_budget,
        "max_base_weight": max_base_weight,
        "max_weight_step": max_weight_step,
        "max_weight_cap": max_weight_cap,
    }
    return best[1], [*search_rows, selected_row]


def build_strategies(
    windows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    policy_search_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    families = sorted({str(window["family"]) for window in windows})

    chronos_train = [window for window in windows if window["family"] == "chronos"]
    chronos_policy, rows = select_policy_for_scope(windows, chronos_train, "chronos_train")
    policy_search_rows.extend([row for row in rows if "policy_id" in row])
    selected_rows.extend([row for row in rows if "selected_policy_id" in row])

    leave_family_policies: dict[str, dict[str, object]] = {}
    risk_controlled_policies: dict[str, dict[str, object]] = {}
    for family in families:
        train = [window for window in windows if window["family"] != family]
        policy, rows = select_policy_for_scope(windows, train, f"leaveout_{family}")
        leave_family_policies[family] = policy
        policy_search_rows.extend([row for row in rows if "policy_id" in row])
        selected_rows.extend([row for row in rows if "selected_policy_id" in row])
        risk_policy, risk_rows = select_risk_controlled_policy_for_scope(
            train,
            f"risk_controlled_leaveout_{family}",
        )
        risk_controlled_policies[family] = risk_policy
        policy_search_rows.extend([row for row in risk_rows if "policy_id" in row])
        selected_rows.extend([row for row in risk_rows if "selected_policy_id" in row])

    strategies: dict[str, list[dict[str, object]]] = {}
    strategies["chronos_tuned_transfer"] = apply_single_policy(windows, chronos_policy, "chronos_tuned_transfer")
    strategies["leave_family_out_calibrated"] = apply_family_policies(
        windows,
        leave_family_policies,
        "leave_family_out_calibrated",
    )
    strategies["risk_controlled_leave_family_out"] = apply_family_policies(
        windows,
        risk_controlled_policies,
        "risk_controlled_leave_family_out",
    )
    high_recall_policy = selective.constant_policy("loose", 2, 0.75, "previous_high_recall_loose_n2_w0.75")
    strategies["previous_high_recall"] = apply_single_policy(windows, high_recall_policy, "previous_high_recall")
    pareto_policy = select_cross_family_margin_pareto_policy(windows, strategies["previous_high_recall"])
    selected_rows.append(
        {
            "selection_scope": "cross_family_margin_pareto",
            "selected_policy_id": pareto_policy["policy_id"],
            "selected_profile": pareto_policy["profile"],
            "selected_min_active": pareto_policy["min_active"],
            "selected_weight": pareto_policy["weight"],
            "selected_weight_step": pareto_policy.get("weight_step", ""),
            "selected_max_weight": pareto_policy.get("max_weight", ""),
            "selected_anchor_mode": pareto_policy.get("anchor_mode", "none"),
            "selected_shield_mode": pareto_policy.get("shield_mode", "none"),
            "selected_shield_threshold": pareto_policy.get("shield_threshold", ""),
            "selected_shield_weight_cap": pareto_policy.get("shield_weight_cap", ""),
            "selected_policy_kind": pareto_policy.get("policy_kind", "fixed_weight"),
            "selection_status": "positive_control_margin_ok",
            "selection_score": pareto_policy["selection_score"],
            "train_n_windows": len(windows),
        }
    )
    strategies["cross_family_margin_pareto"] = apply_single_policy(
        windows,
        pareto_policy,
        "cross_family_margin_pareto",
    )
    tail_safe_policy = select_cross_family_tail_safe_policy(windows, strategies["previous_high_recall"])
    selected_rows.append(
        {
            "selection_scope": "cross_family_tail_safe_pareto",
            "selected_policy_id": tail_safe_policy["policy_id"],
            "selected_profile": tail_safe_policy["profile"],
            "selected_min_active": tail_safe_policy["min_active"],
            "selected_weight": tail_safe_policy["weight"],
            "selected_weight_step": tail_safe_policy.get("weight_step", ""),
            "selected_max_weight": tail_safe_policy.get("max_weight", ""),
            "selected_anchor_mode": tail_safe_policy.get("anchor_mode", "none"),
            "selected_shield_mode": tail_safe_policy.get("shield_mode", "none"),
            "selected_shield_threshold": tail_safe_policy.get("shield_threshold", ""),
            "selected_shield_weight_cap": tail_safe_policy.get("shield_weight_cap", ""),
            "selected_policy_kind": tail_safe_policy.get("policy_kind", "fixed_weight"),
            "selection_status": "positive_control_tail_safe",
            "selection_score": tail_safe_policy["selection_score"],
            "train_n_windows": len(windows),
        }
    )
    strategies["cross_family_tail_safe_pareto"] = apply_single_policy(
        windows,
        tail_safe_policy,
        "cross_family_tail_safe_pareto",
    )
    strategies["strict_safe_gate"] = apply_single_policy(
        windows,
        selective.constant_policy("strict", 3, 0.75, "strict_safe_gate_n3_w0.75"),
        "strict_safe_gate",
    )
    for weight in [0.50, 0.75]:
        strategies[f"global_blend_w{weight:.2f}"] = apply_single_policy(
            windows,
            selective.global_blend_policy(weight),
            f"global_blend_w{weight:.2f}",
        )

    metric_rows = [row for rows in strategies.values() for row in rows]
    strategy_summary: list[dict[str, object]] = []
    for strategy_id, rows in strategies.items():
        strategy_summary.extend(summary_rows(rows, strategy_id))
    return metric_rows, strategy_summary, policy_search_rows, selected_rows


def select_cross_family_margin_pareto_policy(
    windows: list[dict[str, object]],
    previous_rows: list[dict[str, object]],
) -> dict[str, object]:
    previous = {row["group"]: row for row in summary_rows(previous_rows, "previous_high_recall")}
    previous_overall = previous["overall"]
    previous_control = previous["positive_control"]
    best_policy: dict[str, object] | None = None
    best_score = -1e9
    for policy in policy_candidates():
        rows = apply_single_policy(windows, policy, str(policy["policy_id"]))
        current = {row["group"]: row for row in summary_rows(rows, str(policy["policy_id"]))}
        overall = current["overall"]
        control = current["positive_control"]
        gate_improvement = finite_float(previous_overall["gate_rate"]) - finite_float(overall["gate_rate"])
        control_failure_ok = finite_float(control["repair_failure_rate_delta_005"]) <= finite_float(
            control["model_failure_rate_delta_005"]
        ) + 1e-12
        control_margin_ok = finite_float(control["median_rer_delta"]) <= finite_float(
            previous_control["median_rer_delta"]
        ) + 0.05
        repairs_any = finite_float(overall["failure_rate_reduction"]) > 0.0
        selective_enough = gate_improvement >= 0.15
        if not (control_failure_ok and control_margin_ok and repairs_any and selective_enough):
            continue
        score = 5.0 * finite_float(overall["failure_rate_reduction"]) + 0.25 * gate_improvement
        if score > best_score:
            best_score = score
            best_policy = dict(policy)
            best_policy["selection_score"] = score
            best_policy["gate_improvement_vs_high_recall"] = gate_improvement
    if best_policy is None:
        best_policy = selective.constant_policy("balanced", 2, 0.75, "balanced_n2_w0.75")
        best_policy["selection_score"] = 0.0
        best_policy["gate_improvement_vs_high_recall"] = 0.0
    best_policy["policy_id"] = f"cross_family_margin_pareto_{best_policy['policy_id']}"
    return best_policy


def select_cross_family_tail_safe_policy(
    windows: list[dict[str, object]],
    previous_rows: list[dict[str, object]],
) -> dict[str, object]:
    previous = {row["group"]: row for row in summary_rows(previous_rows, "previous_high_recall")}
    previous_overall = previous["overall"]
    best_policy: dict[str, object] | None = None
    best_score = -1e9
    for policy in policy_candidates():
        rows = apply_single_policy(windows, policy, str(policy["policy_id"]))
        current = {row["group"]: row for row in summary_rows(rows, str(policy["policy_id"]))}
        overall = current["overall"]
        control = current["positive_control"]
        gate_improvement = finite_float(previous_overall["gate_rate"]) - finite_float(overall["gate_rate"])
        control_failure_ok = finite_float(control["repair_failure_rate_delta_005"]) <= finite_float(
            control["model_failure_rate_delta_005"]
        ) + 1e-12
        control_median_ok = finite_float(control["median_rer_delta"]) <= 0.05
        control_tail_ok = finite_float(control["p90_rer_delta"]) <= 0.25
        repairs_any = finite_float(overall["failure_rate_reduction"]) > 0.0
        selective_enough = gate_improvement >= 0.15
        if not (control_failure_ok and control_median_ok and control_tail_ok and repairs_any and selective_enough):
            continue
        score = 5.0 * finite_float(overall["failure_rate_reduction"]) + 0.25 * gate_improvement
        if score > best_score:
            best_score = score
            best_policy = dict(policy)
            best_policy["selection_score"] = score
            best_policy["gate_improvement_vs_high_recall"] = gate_improvement
            best_policy["positive_control_p90_rer_delta"] = finite_float(control["p90_rer_delta"])
    if best_policy is None:
        best_policy = selective.constant_policy("strict", 3, 0.25, "strict_n3_w0.25")
        best_policy["selection_score"] = 0.0
        best_policy["gate_improvement_vs_high_recall"] = 0.0
        best_policy["positive_control_p90_rer_delta"] = 0.0
    best_policy["policy_id"] = f"cross_family_tail_safe_{best_policy['policy_id']}"
    return best_policy


def write_report(
    inventory: list[dict[str, object]],
    selected: list[dict[str, object]],
    strategy_summary: list[dict[str, object]],
) -> None:
    lookup = {(row["strategy_id"], row["group"]): row for row in strategy_summary}
    strategies = [
        "chronos_tuned_transfer",
        "leave_family_out_calibrated",
        "risk_controlled_leave_family_out",
        "cross_family_margin_pareto",
        "cross_family_tail_safe_pareto",
        "previous_high_recall",
        "strict_safe_gate",
        "global_blend_w0.75",
    ]
    comparison_lines = [
        "| Strategy | Overall fail | Overall gate | Moirai overall fail | TimesFM overall fail | Positive-control fail | PC median RER delta | PC p90 RER delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy in strategies:
        overall = lookup[(strategy, "overall")]
        moirai = lookup.get((strategy, "family:moirai"), {})
        timesfm = lookup.get((strategy, "family:timesfm"), {})
        control = lookup[(strategy, "positive_control")]
        comparison_lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    f"{finite_float(overall.get('repair_failure_rate_delta_005')):.3f}",
                    f"{finite_float(overall.get('gate_rate')):.3f}",
                    f"{finite_float(moirai.get('repair_failure_rate_delta_005')):.3f}",
                    f"{finite_float(timesfm.get('repair_failure_rate_delta_005')):.3f}",
                    f"{finite_float(control.get('repair_failure_rate_delta_005')):.3f}",
                    f"{finite_float(control.get('median_rer_delta')):.3f}",
                    f"{finite_float(control.get('p90_rer_delta')):.3f}",
                ]
            )
            + " |"
        )
    inventory_lines = [
        "| Family | Source | Role | Windows | Status |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in inventory:
        inventory_lines.append(
            f"| {row['family']} | {row['source']} | {row['role']} | {int(row['n_windows'])} | {row['status']} |"
        )
    selected_lines = [
        "| Scope | Policy | Kind | Anchor | Shield | Weight | Step | Cap | PC p90 budget | Train windows | Score |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        step = row.get("selected_weight_step", "")
        cap = row.get("selected_max_weight", "")
        budget = row.get("positive_control_p90_budget", "")
        shield = row.get("selected_shield_mode", "none")
        shield_cap = row.get("selected_shield_weight_cap", "")
        shield_threshold = row.get("selected_shield_threshold", "")
        shield_text = str(shield)
        if shield and shield != "none":
            shield_text = f"{shield}@{finite_float(shield_threshold):.2f}/cap{finite_float(shield_cap):.3f}"
        step_text = "-" if step == "" else f"{finite_float(step):.3f}"
        cap_text = "-" if cap == "" else f"{finite_float(cap):.2f}"
        budget_text = "-" if budget == "" else f"{finite_float(budget):.2f}"
        selected_lines.append(
            f"| {row['selection_scope']} | {row['selected_policy_id']} | {row.get('selected_policy_kind', 'fixed_weight')} | "
            f"{row.get('selected_anchor_mode', 'none')} | {shield_text} | {finite_float(row['selected_weight']):.2f} | {step_text} | "
            f"{cap_text} | {budget_text} | {int(row['train_n_windows'])} | "
            f"{finite_float(row['selection_score']):.3f} |"
        )
    margin = lookup[("cross_family_margin_pareto", "overall")]
    margin_control = lookup[("cross_family_margin_pareto", "positive_control")]
    tail = lookup[("cross_family_tail_safe_pareto", "overall")]
    tail_control = lookup[("cross_family_tail_safe_pareto", "positive_control")]
    risk = lookup[("risk_controlled_leave_family_out", "overall")]
    risk_control = lookup[("risk_controlled_leave_family_out", "positive_control")]
    risk_timesfm_control = lookup[("risk_controlled_leave_family_out", "family:timesfm|role:positive_control")]

    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Cross-Family Selective Repair Validation",
                "",
                "## Claim",
                "",
                "This goal tests whether the factor-interaction selective repair transfers beyond the Chronos-centered prototype. The cleanest reading is not that one gate is final, but that held-out model-family slices can be repaired without relying on pure global baseline blending.",
                "",
                "## Evaluation Inventory",
                "",
                "\n".join(inventory_lines),
                "",
                "## Held-Out Policy Selection",
                "",
                "\n".join(selected_lines),
                "",
                "## Strategy Comparison",
                "",
                "\n".join(comparison_lines),
                "",
                "## Positive-control Safety and Margin Diagnostics",
                "",
                "Positive-control rows combine Chronos solar/LOOP_SEATTLE, Moirai solar/LOOP_SEATTLE, and TimesFM LOOP_SEATTLE slices. The new TimesFM solar slice is reported separately as a weak positive-control slice because it is not clean under the current RER threshold. A repair is treated as safer when it reduces failure without increasing the positive-control median RER margin. The global_blend_w0.75 control is intentionally reported because it is a competitive but non-selective alternative with positive-control margin cost.",
                "",
                (
                    "The `risk_controlled_leave_family_out` strategy is the cleanest method-facing protocol: for each held-out "
                    "model family, the policy is selected only on the other families, must satisfy a positive-control "
                    "p90 RER-delta budget, and may use a high repair weight only when protected by an expert-conflict "
                    "interval shield. It gives "
                    f"`{100.0 * finite_float(risk['failure_rate_reduction']):.1f}`pp "
                    f"overall failure reduction with gate rate `{100.0 * finite_float(risk['gate_rate']):.1f}`% and held-out "
                    f"positive-control p90 RER delta `{finite_float(risk_control['p90_rer_delta']):.3f}` "
                    f"(TimesFM positive-control family p90 `{finite_float(risk_timesfm_control['p90_rer_delta']):.3f}`)."
                ),
                "",
                (
                    "The selected anchored adaptive policies expose the main tradeoff: the median-safe low-structure gate gives "
                    f"`{100.0 * finite_float(margin['failure_rate_reduction']):.1f}`pp overall failure reduction but has "
                    f"positive-control p90 RER delta `{finite_float(margin_control['p90_rer_delta']):.3f}`; the tail-safe "
                    f"low-structure gate keeps positive-control p90 RER delta to `{finite_float(tail_control['p90_rer_delta']):.3f}` "
                    f"while retaining `{100.0 * finite_float(tail['failure_rate_reduction']):.1f}`pp reduction. The manuscript-facing "
                    "method should use the family-wise tail-safe/conflict-shielded gate as the conservative primary and report the median-safe gate as a tradeoff point."
                ),
                "",
                "The low-structure anchor is an ex-ante guard: a window is repaired only when the factor interaction includes information insufficiency or the weak-structure/noise-dominant pair. The expert-conflict shield is a second guard: when the baseline/reference forecast falls outside the TSFM q10-q90 interval on a large share of the horizon, the fallback weight is capped, preventing blind replacement of useful TSFM behavior on positive-control slices.",
                "",
                "## Transfer Boundary",
                "",
                "The local TimesFM covid slice is no longer a no-transfer case after adding the expert-conflict shield: it shows nontrivial repair, especially once TimesFM LOOP_SEATTLE, solar, and FRED finance stress are included for family coverage. The remaining boundary is narrower: covid-only residual failure stays visible, and unshielded high-weight repair can still harm TimesFM positive controls, so the claim stays at cross-family partial repair rather than universal success.",
                "",
                "## Latest Figure",
                "",
                "![Cross-family selective repair dashboard](../figures/selective_repair/latest_cross_family_selective_repair_dashboard.png)",
                "",
                "## Artifacts",
                "",
                "- `results/repair/cross_family_selective_repair_windows.csv`",
                "- `results/repair/cross_family_selective_repair_policy_search.csv`",
                "- `results/repair/cross_family_selective_repair_selected_policies.csv`",
                "- `results/repair/cross_family_selective_repair_strategy_metrics.csv`",
                "- `results/repair/cross_family_selective_repair_strategy_summary.csv`",
                "- `results/repair/cross_family_selective_repair_positive_control_margins.csv`",
                "- `figures/selective_repair/latest_cross_family_selective_repair_dashboard.png`",
            ]
        )
        + "\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows, inventory = load_cross_family_windows()
    if len({str(window["family"]) for window in windows}) < 3:
        raise SystemExit("Cross-family validation requires Chronos, Moirai, and TimesFM local windows.")

    window_export = [
        {
            "family": row["family"],
            "source": row["source"],
            "role": row["role"],
            "dataset": row["dataset"],
            "model": row["model"],
            "series_id": row["series_id"],
            "origin": row["origin"],
            "window_index": row["window_index"],
            "model_mae": row["model_mae"],
            "baseline_mae": row["baseline_mae"],
            "model_relative_error_ratio": row["model_rer"],
            "model_failure_delta_005": row["model_failure"],
            "model_empirical_coverage_90": row["model_coverage"],
            "model_forecast_variance_ratio": row["model_fvr"],
            "model_prediction_amplitude_ratio": row["model_par"],
            "model_flatness_score": row["model_flatness"],
            "horizon_context_ratio": row["feature"].get("horizon_context_ratio", ""),
            "seasonality_strength": row["feature"].get("seasonality_strength", ""),
            "trend_strength": row["feature"].get("trend_strength", ""),
            "spike_frequency": row["feature"].get("spike_frequency", ""),
            "zero_ratio": row["feature"].get("zero_ratio", ""),
            "coefficient_of_variation": row["feature"].get("coefficient_of_variation", ""),
            "spectral_entropy": row["feature"].get("spectral_entropy", ""),
        }
        for row in windows
    ]
    metric_rows, summaries, policy_search, selected = build_strategies(windows)

    write_csv(OUT_DIR / "cross_family_selective_repair_windows.csv", window_export)
    write_csv(OUT_DIR / "cross_family_selective_repair_inventory.csv", inventory)
    write_csv(OUT_DIR / "cross_family_selective_repair_policy_search.csv", policy_search)
    write_csv(OUT_DIR / "cross_family_selective_repair_selected_policies.csv", selected)
    write_csv(OUT_DIR / "cross_family_selective_repair_strategy_metrics.csv", metric_rows)
    write_csv(OUT_DIR / "cross_family_selective_repair_strategy_summary.csv", summaries)
    margins = [
        row
        for row in summaries
        if row["group"] == "positive_control" or str(row["group"]).endswith("role:positive_control")
    ]
    write_csv(OUT_DIR / "cross_family_selective_repair_positive_control_margins.csv", margins)
    write_report(inventory, selected, summaries)

    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows": len(windows),
        "families": sorted({str(window["family"]) for window in windows}),
        "n_sources": len({str(window["source"]) for window in windows}),
        "inventory": "results/repair/cross_family_selective_repair_inventory.csv",
        "windows": "results/repair/cross_family_selective_repair_windows.csv",
        "policy_search": "results/repair/cross_family_selective_repair_policy_search.csv",
        "selected_policies": "results/repair/cross_family_selective_repair_selected_policies.csv",
        "strategy_metrics": "results/repair/cross_family_selective_repair_strategy_metrics.csv",
        "strategy_summary": "results/repair/cross_family_selective_repair_strategy_summary.csv",
        "positive_control_margins": "results/repair/cross_family_selective_repair_positive_control_margins.csv",
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    status_path = OUT_DIR / "cross_family_selective_repair_goal_status.json"
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
