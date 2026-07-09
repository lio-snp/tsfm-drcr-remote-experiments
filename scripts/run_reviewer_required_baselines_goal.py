#!/usr/bin/env python
"""Run reviewer-facing repair baselines under matched calibration/test splits.

This pass answers the "is it just fallback/blending?" objection before the
Conformal Policy Repair refactor.  It compares the current fixed RR-CSSR rules
against same-split validation-tuned global blends, learned gates, and a
conflict-only selector.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_cross_family_selective_repair_goal as cross  # noqa: E402
import run_risk_calibrated_rr_cssr_goal as rr_cssr  # noqa: E402
import run_selective_repair_expanded_ablation_goal as selective  # noqa: E402

from low_snr_tsfm.metrics import mae, relative_error_ratio  # noqa: E402


OUT_DIR = ROOT / "results" / "aaai_stress"
DOC_PATH = ROOT / "docs" / "reviewer_required_baselines_report.md"
STATUS_PATH = OUT_DIR / "reviewer_required_baselines_status.json"

FEATURE_NAMES = [
    "horizon_context_ratio",
    "seasonality_strength",
    "trend_strength",
    "zero_ratio",
    "coefficient_of_variation",
    "spectral_entropy",
    "spike_frequency",
    "changepoint_density",
    "kurtosis_excess",
    "autocorrelation_strength",
    "missingness",
    "reference_outside_interval_rate",
]

WEIGHT_GRID = [0.0, 0.125, 0.25, 0.50, 0.75, 1.00]
GATE_WEIGHT_GRID = [0.125, 0.25, 0.50, 0.75]
PROB_THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
CONFLICT_THRESHOLDS = [0.00, 0.10, 0.25, 0.40, 0.60, 0.80, 1.00]
PC_TAIL_BUDGET = 0.05
PC_MEDIAN_BUDGET = 0.02
WEAK_PC_TAIL_BUDGET = 0.35


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.median(finite)) if finite else float("nan")


def p90(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.percentile(finite, 90)) if finite else float("nan")


def rate(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def bootstrap_ci(values: list[float], *, n_bootstrap: int = 2000, seed: int = 97) -> tuple[float, float]:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
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


def reference_conflict(window: dict[str, object]) -> tuple[float, float]:
    return selective.reference_interval_conflict(
        np.asarray(window["baseline_forecast"], dtype=float),
        np.asarray(window["q10"], dtype=float),
        np.asarray(window["q90"], dtype=float),
    )


def feature_value(window: dict[str, object], name: str) -> float:
    if name == "reference_outside_interval_rate":
        return reference_conflict(window)[0]
    return finite_float(window["feature"].get(name))


def feature_matrix(windows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([[feature_value(window, name) for name in FEATURE_NAMES] for window in windows], dtype=float)


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic_gate(train_windows: list[dict[str, object]]) -> dict[str, object]:
    x_raw = feature_matrix(train_windows)
    y = np.asarray([int(window["model_failure"]) for window in train_windows], dtype=float)
    mu = np.nanmean(x_raw, axis=0)
    sigma = np.nanstd(x_raw, axis=0)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    x = np.nan_to_num((x_raw - mu) / sigma, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.column_stack([np.ones(x.shape[0]), x])
    beta = np.zeros(x.shape[1], dtype=float)
    pos = float(np.sum(y == 1.0))
    neg = float(np.sum(y == 0.0))
    if pos == 0.0 or neg == 0.0:
        prior = min(max(float(np.mean(y)), 1e-4), 1.0 - 1e-4)
        beta[0] = math.log(prior / (1.0 - prior))
        return {"beta": beta, "mu": mu, "sigma": sigma}
    sample_weight = np.where(y == 1.0, 0.5 / pos, 0.5 / neg)
    lr = 0.35
    l2 = 0.02
    for _ in range(900):
        pred = stable_sigmoid(x @ beta)
        grad = x.T @ ((pred - y) * sample_weight)
        grad[1:] += l2 * beta[1:]
        beta -= lr * grad
    return {"beta": beta, "mu": mu, "sigma": sigma}


def logistic_prob(window: dict[str, object], model: dict[str, object]) -> float:
    x = np.asarray([feature_value(window, name) for name in FEATURE_NAMES], dtype=float)
    x = np.nan_to_num((x - model["mu"]) / model["sigma"], nan=0.0, posinf=0.0, neginf=0.0)
    x = np.concatenate([[1.0], x])
    return float(stable_sigmoid(np.asarray([float(x @ model["beta"])]))[0])


def low_structure_factor_count(window: dict[str, object]) -> int:
    policy = {
        "hcr": 0.10,
        "seasonality": 0.15,
        "trend": 0.10,
        "spike": 0.022,
        "change": 0.08,
        "zero": 0.10,
        "cv": 1.50,
        "kurt": 8.0,
        "entropy": 0.85,
    }
    flags = selective.base.flags_for(window["feature"], denominator_fragile=0, policy=policy)
    return len(selective.base.active_factors(flags, include_denominator=False))


def policy_decision(window: dict[str, object], spec: dict[str, object]) -> tuple[int, float, str, float]:
    kind = str(spec["kind"])
    weight = finite_float(spec.get("weight"))
    if kind == "no_repair":
        return 0, 0.0, "no_repair", 0.0
    if kind == "global_blend":
        return int(weight > 0.0), weight, "global_blend", weight
    if kind == "conflict_only":
        outside_rate, _ = reference_conflict(window)
        threshold = finite_float(spec["conflict_threshold"])
        gate = int(outside_rate <= threshold and weight > 0.0)
        return gate, weight if gate else 0.0, "conflict_compatible", outside_rate
    if kind == "logistic_gate":
        prob = logistic_prob(window, spec["model"])
        threshold = finite_float(spec["prob_threshold"])
        gate = int(prob >= threshold and weight > 0.0)
        return gate, weight if gate else 0.0, "learned_logistic", prob
    if kind == "stump_gate":
        value = feature_value(window, str(spec["feature_name"]))
        threshold = finite_float(spec["threshold"])
        direction = str(spec["direction"])
        gate = int((value >= threshold if direction == "ge" else value <= threshold) and weight > 0.0)
        return gate, weight if gate else 0.0, f"stump_{direction}", value
    raise ValueError(f"Unsupported policy kind: {kind}")


def apply_simple_policy_to_window(
    window: dict[str, object],
    spec: dict[str, object],
    strategy_id: str,
    split_protocol: str,
    split_id: str,
) -> dict[str, object]:
    gate, weight, reason, score = policy_decision(window, spec)
    actual = np.asarray(window["actual"], dtype=float)
    model = np.asarray(window["model_forecast"], dtype=float)
    baseline = np.asarray(window["baseline_forecast"], dtype=float)
    repaired = model + weight * (baseline - model)
    repair_mae = mae(actual, repaired)
    repair_rer = relative_error_ratio(repair_mae, finite_float(window["baseline_mae"]))
    outside_rate, outside_ratio = reference_conflict(window)
    return {
        "strategy_id": strategy_id,
        "split_protocol": split_protocol,
        "split_id": split_id,
        "selected_policy_id": spec["policy_id"],
        "policy_class": spec["kind"],
        "family": window["family"],
        "source": window["source"],
        "role": window["role"],
        "dataset": window["dataset"],
        "model": window["model"],
        "series_id": window["series_id"],
        "origin": window["origin"],
        "window_index": window["window_index"],
        "gate_active": gate,
        "gate_reason": reason,
        "gate_score": score,
        "effective_weight": weight,
        "reference_outside_interval_rate": outside_rate,
        "reference_outside_interval_mean_ratio": outside_ratio,
        "low_structure_factor_count": low_structure_factor_count(window),
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
        "horizon_context_ratio": feature_value(window, "horizon_context_ratio"),
        "trend_strength": feature_value(window, "trend_strength"),
        "zero_ratio": feature_value(window, "zero_ratio"),
        "coefficient_of_variation": feature_value(window, "coefficient_of_variation"),
        "spectral_entropy": feature_value(window, "spectral_entropy"),
    }


def apply_simple_policy(
    windows: list[dict[str, object]],
    spec: dict[str, object],
    strategy_id: str,
    split_protocol: str,
    split_id: str,
) -> list[dict[str, object]]:
    return [apply_simple_policy_to_window(window, spec, strategy_id, split_protocol, split_id) for window in windows]


def apply_rr_cssr_policy(
    windows: list[dict[str, object]],
    config: dict[str, object],
    strategy_id: str,
    split_protocol: str,
    split_id: str,
) -> list[dict[str, object]]:
    policy = rr_cssr.base_policy()
    rows: list[dict[str, object]] = []
    for window in windows:
        row = rr_cssr.apply_config_to_window(window, policy, config, strategy_id, split_id)
        row["split_protocol"] = split_protocol
        row["selected_policy_id"] = str(config["config_id"])
        row["policy_class"] = "fixed_rr_cssr"
        row["mae_delta_vs_model"] = finite_float(row["repair_mae"]) - finite_float(row["model_mae"])
        row["repair_improves_model"] = int(finite_float(row["repair_mae"]) < finite_float(row["model_mae"]))
        rows.append(row)
    return rows


def metric_bundle(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {
            "n_windows": 0,
            "model_failure_rate": float("nan"),
            "repair_failure_rate": float("nan"),
            "failure_rate_reduction": float("nan"),
            "median_rer_delta": float("nan"),
            "p90_rer_delta": float("nan"),
            "gate_rate": float("nan"),
            "mean_mae_delta_vs_model": float("nan"),
            "repair_win_rate": float("nan"),
        }
    model_fail = [finite_float(row["model_failure_delta_005"]) for row in rows]
    repair_fail = [finite_float(row["repair_failure_delta_005"]) for row in rows]
    diff = [m - r for m, r in zip(model_fail, repair_fail, strict=True)]
    return {
        "n_windows": len(rows),
        "model_failure_rate": rate(model_fail),
        "repair_failure_rate": rate(repair_fail),
        "failure_rate_reduction": rate(diff),
        "median_rer_delta": median([finite_float(row["relative_error_ratio_delta"]) for row in rows]),
        "p90_rer_delta": p90([finite_float(row["relative_error_ratio_delta"]) for row in rows]),
        "gate_rate": rate([finite_float(row["gate_active"]) for row in rows]),
        "mean_mae_delta_vs_model": mean([finite_float(row["mae_delta_vs_model"]) for row in rows]),
        "repair_win_rate": rate([finite_float(row["repair_improves_model"]) for row in rows]),
    }


def selection_metrics(rows: list[dict[str, object]]) -> dict[str, float]:
    overall = metric_bundle(rows)
    failure = metric_bundle([row for row in rows if row["role"] == "failure_target"])
    stress = metric_bundle([row for row in rows if row["role"] == "stress_target"])
    pc = metric_bundle([row for row in rows if row["role"] == "positive_control"])
    weak_rows = [row for row in rows if row["role"] == "weak_positive_control"]
    weak = metric_bundle(weak_rows) if weak_rows else {"p90_rer_delta": float("nan")}
    return {
        "overall_reduction": overall["failure_rate_reduction"],
        "failure_target_reduction": failure["failure_rate_reduction"],
        "stress_target_reduction": stress["failure_rate_reduction"],
        "gate_rate": overall["gate_rate"],
        "positive_control_failure_delta": pc["repair_failure_rate"] - pc["model_failure_rate"],
        "positive_control_median_rer_delta": pc["median_rer_delta"],
        "positive_control_p90_rer_delta": pc["p90_rer_delta"],
        "weak_positive_control_p90_rer_delta": finite_float(weak.get("p90_rer_delta"), float("nan")),
    }


def selection_score(metrics: dict[str, float]) -> tuple[float, str]:
    if metrics["positive_control_failure_delta"] > 1e-12:
        return -1000.0, "positive_control_failure_violation"
    if metrics["positive_control_p90_rer_delta"] > PC_TAIL_BUDGET:
        return -900.0, "positive_control_tail_violation"
    if metrics["positive_control_median_rer_delta"] > PC_MEDIAN_BUDGET:
        return -800.0, "positive_control_median_violation"
    weak_p90 = metrics["weak_positive_control_p90_rer_delta"]
    if math.isfinite(weak_p90) and weak_p90 > WEAK_PC_TAIL_BUDGET:
        return -700.0, "weak_positive_control_tail_violation"
    score = (
        5.0 * metrics["overall_reduction"]
        + 0.75 * metrics["failure_target_reduction"]
        + 0.75 * metrics["stress_target_reduction"]
        - 0.15 * metrics["gate_rate"]
        - 0.25 * max(0.0, metrics["positive_control_p90_rer_delta"])
    )
    if metrics["overall_reduction"] < -1e-12:
        return score - 100.0, "negative_training_utility"
    return score, "ok"


def no_repair_spec(policy_id: str = "no_repair") -> dict[str, object]:
    return {"kind": "no_repair", "policy_id": policy_id, "weight": 0.0}


def global_blend_candidates() -> list[dict[str, object]]:
    return [
        {"kind": "global_blend", "policy_id": f"global_blend_w{weight:.3f}", "weight": weight}
        for weight in WEIGHT_GRID
    ]


def conflict_only_candidates() -> list[dict[str, object]]:
    candidates = [no_repair_spec()]
    for threshold in CONFLICT_THRESHOLDS:
        for weight in GATE_WEIGHT_GRID:
            candidates.append(
                {
                    "kind": "conflict_only",
                    "policy_id": f"conflict_only_le{threshold:.2f}_w{weight:.3f}",
                    "conflict_threshold": threshold,
                    "weight": weight,
                }
            )
    return candidates


def logistic_candidates(train_windows: list[dict[str, object]]) -> list[dict[str, object]]:
    model = fit_logistic_gate(train_windows)
    candidates = [no_repair_spec()]
    for threshold in PROB_THRESHOLDS:
        for weight in GATE_WEIGHT_GRID:
            candidates.append(
                {
                    "kind": "logistic_gate",
                    "policy_id": f"logistic_p{threshold:.2f}_w{weight:.3f}",
                    "prob_threshold": threshold,
                    "weight": weight,
                    "model": model,
                }
            )
    return candidates


def stump_candidates(train_windows: list[dict[str, object]]) -> list[dict[str, object]]:
    candidates = [no_repair_spec()]
    x = feature_matrix(train_windows)
    for index, name in enumerate(FEATURE_NAMES):
        values = x[:, index]
        values = values[np.isfinite(values)]
        if values.size < 4 or float(np.max(values) - np.min(values)) < 1e-12:
            continue
        for threshold in sorted(set(float(q) for q in np.quantile(values, [0.2, 0.4, 0.6, 0.8]))):
            for direction in ["ge", "le"]:
                for weight in GATE_WEIGHT_GRID:
                    candidates.append(
                        {
                            "kind": "stump_gate",
                            "policy_id": f"stump_{name}_{direction}{threshold:.4g}_w{weight:.3f}",
                            "feature_name": name,
                            "direction": direction,
                            "threshold": threshold,
                            "weight": weight,
                        }
                    )
    return candidates


def select_simple_candidate(
    train_windows: list[dict[str, object]],
    strategy_id: str,
    split_protocol: str,
    split_id: str,
    candidate_factory: Callable[[list[dict[str, object]]], list[dict[str, object]]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    best: tuple[float, dict[str, object], str, dict[str, float]] | None = None
    rows: list[dict[str, object]] = []
    for spec in candidate_factory(train_windows):
        repaired = apply_simple_policy(train_windows, spec, strategy_id, split_protocol, split_id)
        metrics = selection_metrics(repaired)
        score, status = selection_score(metrics)
        candidate_row = {
            "strategy_id": strategy_id,
            "split_protocol": split_protocol,
            "split_id": split_id,
            "candidate_policy_id": spec["policy_id"],
            "policy_class": spec["kind"],
            "selection_score": score,
            "selection_status": status,
            **metrics,
        }
        for key in [
            "weight",
            "prob_threshold",
            "conflict_threshold",
            "feature_name",
            "direction",
            "threshold",
        ]:
            if key in spec:
                candidate_row[key] = spec[key]
        rows.append(candidate_row)
        if status == "ok" and (best is None or score > best[0]):
            best = (score, dict(spec), status, metrics)
    if best is None:
        fallback = no_repair_spec()
        best = (0.0, fallback, "fallback_no_safe_candidate", selection_metrics(apply_simple_policy(train_windows, fallback, strategy_id, split_protocol, split_id)))
    selected = dict(best[1])
    selected_row = {
        "strategy_id": strategy_id,
        "split_protocol": split_protocol,
        "split_id": split_id,
        "selected_policy_id": selected["policy_id"],
        "policy_class": selected["kind"],
        "selection_score": best[0],
        "selection_status": f"selected_{best[2]}",
        "train_n_windows": len(train_windows),
        **best[3],
    }
    for key in ["weight", "prob_threshold", "conflict_threshold", "feature_name", "direction", "threshold"]:
        if key in selected:
            selected_row[f"selected_{key}"] = selected[key]
    rows.append(selected_row)
    return selected, rows


def split_values(windows: list[dict[str, object]], split_key: str) -> list[str]:
    return sorted({str(window[split_key]) for window in windows})


def run_simple_strategy(
    windows: list[dict[str, object]],
    strategy_id: str,
    split_key: str,
    candidate_factory: Callable[[list[dict[str, object]]], list[dict[str, object]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    split_protocol = f"leave_{split_key}"
    for value in split_values(windows, split_key):
        split_id = f"holdout_{split_key}:{value}"
        train = [window for window in windows if str(window[split_key]) != value]
        test = [window for window in windows if str(window[split_key]) == value]
        spec, rows = select_simple_candidate(train, strategy_id, split_protocol, split_id, candidate_factory)
        selection_rows.extend(rows)
        all_rows.extend(apply_simple_policy(test, spec, strategy_id, split_protocol, split_id))
    return all_rows, selection_rows


def run_fixed_rr_strategy(
    windows: list[dict[str, object]],
    strategy_id: str,
    split_key: str,
    config: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows: list[dict[str, object]] = []
    selected_rows: list[dict[str, object]] = []
    split_protocol = f"leave_{split_key}"
    for value in split_values(windows, split_key):
        split_id = f"holdout_{split_key}:{value}"
        test = [window for window in windows if str(window[split_key]) == value]
        rows = apply_rr_cssr_policy(test, config, strategy_id, split_protocol, split_id)
        all_rows.extend(rows)
        selected_rows.append(
            {
                "strategy_id": strategy_id,
                "split_protocol": split_protocol,
                "split_id": split_id,
                "selected_policy_id": config["config_id"],
                "policy_class": "fixed_rr_cssr",
                "selection_status": "fixed_pre_registered",
                "train_n_windows": len(windows) - len(test),
                "test_n_windows": len(test),
                "selected_conflict_threshold": config["conflict_threshold"],
                "selected_shield_cap": config["shield_cap"],
                "selected_hcr_threshold": config["hcr_threshold"],
                "selected_trend_threshold": config["trend_threshold"],
            }
        )
    return all_rows, selected_rows


def summary_for_group(
    rows: list[dict[str, object]],
    strategy_id: str,
    split_protocol: str,
    group: str,
    group_type: str,
) -> dict[str, object]:
    metrics = metric_bundle(rows)
    model_fail = np.asarray([finite_float(row["model_failure_delta_005"]) for row in rows], dtype=float)
    repair_fail = np.asarray([finite_float(row["repair_failure_delta_005"]) for row in rows], dtype=float)
    diff = (model_fail - repair_fail).tolist()
    improved = int(np.sum((model_fail == 1.0) & (repair_fail == 0.0)))
    worsened = int(np.sum((model_fail == 0.0) & (repair_fail == 1.0)))
    ci_low, ci_high = bootstrap_ci(diff)
    return {
        "strategy_id": strategy_id,
        "split_protocol": split_protocol,
        "group": group,
        "group_type": group_type,
        **metrics,
        "failure_reduction_ci_low": ci_low,
        "failure_reduction_ci_high": ci_high,
        "mcnemar_improved_count": improved,
        "mcnemar_worsened_count": worsened,
        "mcnemar_exact_p": exact_mcnemar_p(improved, worsened),
    }


def build_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for (strategy_id, split_protocol), base_rows in sorted(
        groupby_map(rows, ["strategy_id", "split_protocol"]).items()
    ):
        out.append(summary_for_group(base_rows, strategy_id, split_protocol, "overall", "overall"))
        for role, role_rows in sorted(groupby_map(base_rows, ["role"]).items()):
            out.append(summary_for_group(role_rows, strategy_id, split_protocol, f"role:{role}", "role"))
        for family, family_rows in sorted(groupby_map(base_rows, ["family"]).items()):
            out.append(summary_for_group(family_rows, strategy_id, split_protocol, f"family:{family}", "family"))
        for source, source_rows in sorted(groupby_map(base_rows, ["source"]).items()):
            out.append(summary_for_group(source_rows, strategy_id, split_protocol, f"source:{source}", "source"))
    return out


def groupby_map(rows: list[dict[str, object]], keys: list[str]) -> dict[object, list[dict[str, object]]]:
    groups: dict[object, list[dict[str, object]]] = {}
    for row in rows:
        key_values = tuple(str(row.get(key, "")) for key in keys)
        key: object = key_values[0] if len(key_values) == 1 else key_values
        groups.setdefault(key, []).append(row)
    return groups


def threshold_stability(selection_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = [row for row in selection_rows if str(row.get("selected_policy_id", ""))]
    rows: list[dict[str, object]] = []
    for (strategy_id, split_protocol), group_rows in sorted(groupby_map(selected, ["strategy_id", "split_protocol"]).items()):
        policies = [str(row.get("selected_policy_id", "")) for row in group_rows]
        counts = Counter(policies)
        most_common, count = counts.most_common(1)[0]
        weights = [finite_float(row.get("selected_weight"), float("nan")) for row in group_rows if "selected_weight" in row]
        prob_thresholds = [
            finite_float(row.get("selected_prob_threshold"), float("nan"))
            for row in group_rows
            if "selected_prob_threshold" in row
        ]
        conflict_thresholds = [
            finite_float(row.get("selected_conflict_threshold"), float("nan"))
            for row in group_rows
            if "selected_conflict_threshold" in row
        ]
        hcr_thresholds = [
            finite_float(row.get("selected_hcr_threshold"), float("nan"))
            for row in group_rows
            if "selected_hcr_threshold" in row
        ]
        trend_thresholds = [
            finite_float(row.get("selected_trend_threshold"), float("nan"))
            for row in group_rows
            if "selected_trend_threshold" in row
        ]
        rows.append(
            {
                "strategy_id": strategy_id,
                "split_protocol": split_protocol,
                "n_splits": len(group_rows),
                "unique_selected_policy_count": len(counts),
                "most_common_policy": most_common,
                "most_common_policy_share": count / len(group_rows),
                "no_repair_selection_rate": rate([1.0 if policy == "no_repair" else 0.0 for policy in policies]),
                "selected_weight_mean": mean(weights),
                "selected_weight_std": float(np.nanstd(weights)) if weights else float("nan"),
                "selected_prob_threshold_mean": mean(prob_thresholds),
                "selected_conflict_threshold_mean": mean(conflict_thresholds),
                "selected_hcr_threshold_mean": mean(hcr_thresholds),
                "selected_trend_threshold_mean": mean(trend_thresholds),
            }
        )
    return rows


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


def report(summary: list[dict[str, object]], stability: list[dict[str, object]]) -> None:
    lookup = {(row["strategy_id"], row["split_protocol"], row["group"]): row for row in summary}
    strategies = [
        "fixed_cssr_cap_all_conflicts",
        "fixed_rr_cssr_trend_decay_adjudicator",
        "validation_tuned_global_blend",
        "conflict_only_selector",
        "learned_logistic_gate",
        "learned_stump_gate",
    ]
    main_rows: list[dict[str, object]] = []
    for split_protocol in ["leave_source", "leave_family"]:
        for strategy in strategies:
            row = lookup[(strategy, split_protocol, "overall")]
            pc = lookup.get((strategy, split_protocol, "role:positive_control"), {})
            weak = lookup.get((strategy, split_protocol, "role:weak_positive_control"), {})
            main_rows.append(
                {
                    "Split": split_protocol,
                    "Strategy": strategy,
                    "N": int(row["n_windows"]),
                    "Reduction": fmt_pct(row["failure_rate_reduction"]),
                    "95% CI": f"[{fmt_pct(row['failure_reduction_ci_low'])}, {fmt_pct(row['failure_reduction_ci_high'])}]",
                    "PC p90": fmt_num(pc.get("p90_rer_delta", float("nan"))),
                    "Weak-PC p90": fmt_num(weak.get("p90_rer_delta", float("nan"))),
                    "Gate": fmt_pct(row["gate_rate"]),
                    "McNemar p": fmt_num(row["mcnemar_exact_p"], 4),
                }
            )
    stability_rows = []
    for row in stability:
        stability_rows.append(
            {
                "Split": row["split_protocol"],
                "Strategy": row["strategy_id"],
                "Policies": int(row["unique_selected_policy_count"]),
                "Top share": fmt_pct(row["most_common_policy_share"]),
                "No repair": fmt_pct(row["no_repair_selection_rate"]),
                "Mean w": fmt_num(row["selected_weight_mean"], 3),
                "Mean prob": fmt_num(row["selected_prob_threshold_mean"], 3),
                "Mean conflict": fmt_num(row["selected_conflict_threshold_mean"], 3),
                "Mean HCR": fmt_num(row["selected_hcr_threshold_mean"], 3),
                "Mean trend": fmt_num(row["selected_trend_threshold_mean"], 3),
            }
        )
    source_lookup = {(row["Strategy"], row["Split"]): row for row in main_rows}
    rr = source_lookup[("fixed_rr_cssr_trend_decay_adjudicator", "leave_source")]
    learned_logistic = source_lookup[("learned_logistic_gate", "leave_source")]
    learned_stump = source_lookup[("learned_stump_gate", "leave_source")]
    tuned_blend = source_lookup[("validation_tuned_global_blend", "leave_source")]
    DOC_PATH.write_text(
        "\n".join(
            [
                "# Reviewer-Required Repair Baselines",
                "",
                "## What This Adds",
                "",
                "This run keeps the method refactor separate from the evidence gap. It evaluates the current repair against strong same-split controls: validation-tuned global blending, a conflict-only selector, and two learned gates trained only on calibration windows.",
                "",
                "The key review question is whether the current router/shield/baseline mixture is merely a tuned fallback. If learned routers or global blending dominate under the same split, the method story is weak; if they do not, the next CPR/LTT step has a cleaner target: turn the surviving operating point into a risk-controlled algorithm rather than another heuristic.",
                "",
                "## Matched-Split Baseline Table",
                "",
                markdown_table(
                    main_rows,
                    [
                        ("Split", "Split"),
                        ("Strategy", "Strategy"),
                        ("N", "N"),
                        ("Reduction", "Failure Reduction"),
                        ("95% CI", "95% CI"),
                        ("PC p90", "PC p90 dRER"),
                        ("Weak-PC p90", "Weak-PC p90"),
                        ("Gate", "Gate"),
                        ("McNemar p", "McNemar p"),
                    ],
                ),
                "",
                "## Threshold / Policy Stability",
                "",
                markdown_table(
                    stability_rows,
                    [
                        ("Split", "Split"),
                        ("Strategy", "Strategy"),
                        ("Policies", "Unique Policies"),
                        ("Top share", "Top Share"),
                        ("No repair", "No-Repair Rate"),
                        ("Mean w", "Mean Weight"),
                        ("Mean prob", "Mean Prob"),
                        ("Mean conflict", "Mean Conflict"),
                        ("Mean HCR", "Mean HCR"),
                        ("Mean trend", "Mean Trend"),
                    ],
                ),
                "",
                "## Immediate Interpretation",
                "",
                f"- On the primary leave-source split, fixed RR-CSSR gives {rr['Reduction']} failure-rate reduction; validation-tuned global blending gives {tuned_blend['Reduction']}.",
                f"- Learned logistic and stump gates give {learned_logistic['Reduction']} and {learned_stump['Reduction']} respectively under the same safety constraints.",
                f"- Their held-out positive-control p90 dRER values are {learned_logistic['PC p90']} and {learned_stump['PC p90']}, so the learned routers do not certify the safety story even when they reduce some failures.",
                "- A learned gate that loses to fixed RR-CSSR or breaks held-out positive-control tail safety is evidence that the current small slice does not yet support a reliable cross-domain predictor; this motivates risk-controlled policy calibration rather than an unconstrained learned router.",
                "- These results still do not solve the method-novelty critique. They are the control layer that the CPR/LTT implementation must beat or certify against.",
                "",
                "## Artifacts",
                "",
                "- `results/aaai_stress/reviewer_baseline_windows.csv`",
                "- `results/aaai_stress/reviewer_baseline_summary.csv`",
                "- `results/aaai_stress/reviewer_baseline_policy_search.csv`",
                "- `results/aaai_stress/reviewer_threshold_stability.csv`",
            ]
        )
        + "\n"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    windows, _ = cross.load_cross_family_windows()
    fixed_cssr = {
        "config_id": "fixed_cssr_cap_all_conflicts",
        "conflict_threshold": 0.40,
        "shield_cap": 0.125,
        "hcr_threshold": -1.0,
        "trend_threshold": 2.0,
    }
    fixed_rr = {
        "config_id": "fixed_rr_cssr_trend_decay_adjudicator",
        "conflict_threshold": 0.40,
        "shield_cap": 0.125,
        "hcr_threshold": 0.30,
        "trend_threshold": 0.40,
    }
    all_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for split_key in ["source", "family"]:
        for strategy_id, config in [
            ("fixed_cssr_cap_all_conflicts", fixed_cssr),
            ("fixed_rr_cssr_trend_decay_adjudicator", fixed_rr),
        ]:
            rows, selected = run_fixed_rr_strategy(windows, strategy_id, split_key, config)
            all_rows.extend(rows)
            selection_rows.extend(selected)
        for strategy_id, factory in [
            ("validation_tuned_global_blend", lambda train: global_blend_candidates()),
            ("conflict_only_selector", lambda train: conflict_only_candidates()),
            ("learned_logistic_gate", logistic_candidates),
            ("learned_stump_gate", stump_candidates),
        ]:
            rows, selected = run_simple_strategy(windows, strategy_id, split_key, factory)
            all_rows.extend(rows)
            selection_rows.extend(selected)
    summary = build_summary(all_rows)
    stability = threshold_stability(selection_rows)
    write_csv(OUT_DIR / "reviewer_baseline_windows.csv", all_rows)
    write_csv(OUT_DIR / "reviewer_baseline_summary.csv", summary)
    write_csv(OUT_DIR / "reviewer_baseline_policy_search.csv", selection_rows)
    write_csv(OUT_DIR / "reviewer_threshold_stability.csv", stability)
    report(summary, stability)
    status = {
        "status": "ok",
        "timestamp": int(time.time()),
        "n_windows": len(windows),
        "n_window_rows": len(all_rows),
        "n_policy_search_rows": len(selection_rows),
        "n_summary_rows": len(summary),
        "summary": "results/aaai_stress/reviewer_baseline_summary.csv",
        "windows": "results/aaai_stress/reviewer_baseline_windows.csv",
        "policy_search": "results/aaai_stress/reviewer_baseline_policy_search.csv",
        "threshold_stability": "results/aaai_stress/reviewer_threshold_stability.csv",
        "report": str(DOC_PATH.relative_to(ROOT)),
    }
    STATUS_PATH.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
