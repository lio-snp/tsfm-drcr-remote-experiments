"""Frozen, outcome-blind mechanics for external Benefit-Selective DRCR."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .features import feature_vector
from .gift_eval_windowing import forward_fill_nan


Array = np.ndarray
EPS = 1e-12


def stable_sigmoid(value: float) -> float:
    if value > 60.0:
        return 1.0
    if value < -60.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def native_width_ratio(quantile_levels: list[float], quantile_grid: Array) -> float:
    levels = np.asarray(quantile_levels, dtype=float)
    grid = np.asarray(quantile_grid, dtype=float)
    q10 = grid[:, int(np.argmin(np.abs(levels - 0.1)))]
    q50 = grid[:, int(np.argmin(np.abs(levels - 0.5)))]
    q90 = grid[:, int(np.argmin(np.abs(levels - 0.9)))]
    return float(np.mean(q90 - q10) / (np.mean(np.abs(q50)) + EPS))


def structured_guard(features: dict[str, float], head: dict[str, Any]) -> bool:
    return (
        features["horizon_context_ratio"] <= float(head["structured_guard_hcr_threshold"])
        and features["trend_strength"] <= float(head["structured_guard_trend_threshold"])
    )


def smooth_width_score(width_ratio: float, features: dict[str, float], head: dict[str, Any]) -> float:
    if structured_guard(features, head):
        return 0.0
    return stable_sigmoid(
        (float(head["native_width_ratio_threshold"]) - width_ratio) / float(head["temperature"])
    )


def pre_origin_feature_vector(
    context: Array,
    horizon: int,
    quantile_levels: list[float],
    quantile_grid: Array,
    family: str,
    head: dict[str, Any],
) -> dict[str, float]:
    clean_context = forward_fill_nan(np.asarray(context, dtype=float))
    features = feature_vector(
        clean_context,
        horizon=int(horizon),
        context_length=int(clean_context.size),
        period=int(horizon),
    )
    features["context_length"] = float(clean_context.size)
    features["horizon"] = float(horizon)
    width_ratio = native_width_ratio(quantile_levels, quantile_grid)
    features["native_width_ratio"] = width_ratio
    features["smooth_width_score"] = smooth_width_score(width_ratio, features, head)
    features["family_is_timesfm"] = float(family == "timesfm")
    return features


def low_structure_factors(
    features: dict[str, float], taxonomy: dict[str, Any]
) -> dict[str, bool]:
    threshold = taxonomy["thresholds"]
    seasonality = features["seasonality_strength"]
    trend = features["trend_strength"]
    entropy = features["spectral_entropy"]
    return {
        "information_insufficiency": features["horizon_context_ratio"] >= float(threshold["horizon_context_ratio"]),
        "weak_reusable_structure": (
            seasonality <= float(threshold["seasonality_strength"])
            and trend <= float(threshold["trend_strength"])
        ) or (
            seasonality <= float(threshold["severe_weak_seasonality"])
            and entropy >= float(threshold["spectral_entropy"])
        ),
        "pathological_dynamics": (
            features["spike_frequency"] >= float(threshold["spike_frequency"])
            or features["changepoint_density"] >= float(threshold["changepoint_density"])
        ),
        "sparse_count_behavior": features["zero_ratio"] >= float(threshold["zero_ratio"]),
        "noise_dominant_history": (
            features["coefficient_of_variation"] >= float(threshold["coefficient_of_variation"])
            or features["kurtosis_excess"] >= float(threshold["kurtosis_excess"])
            or (
                entropy >= float(threshold["spectral_entropy"])
                and seasonality <= float(threshold["seasonality_strength"])
            )
        ),
    }


def low_structure_count(features: dict[str, float], taxonomy: dict[str, Any]) -> int:
    return sum(low_structure_factors(features, taxonomy).values())


def reference_interval_conflict(
    reference: Array,
    quantile_levels: list[float],
    quantile_grid: Array,
) -> tuple[float, float]:
    levels = np.asarray(quantile_levels, dtype=float)
    grid = np.asarray(quantile_grid, dtype=float)
    lower = grid[:, int(np.argmin(np.abs(levels - 0.1)))]
    upper = grid[:, int(np.argmin(np.abs(levels - 0.9)))]
    reference = np.asarray(reference, dtype=float)
    valid = np.isfinite(reference) & np.isfinite(lower) & np.isfinite(upper)
    if not np.any(valid):
        return 0.0, 0.0
    lo = np.minimum(lower[valid], upper[valid])
    hi = np.maximum(lower[valid], upper[valid])
    distance = np.maximum(np.maximum(lo - reference[valid], reference[valid] - hi), 0.0)
    width = np.maximum(hi - lo, 1e-8)
    return float(np.mean(distance > 0.0)), float(np.mean(distance / width))


def base_cpr_weight(
    features: dict[str, float],
    reference: Array,
    quantile_levels: list[float],
    quantile_grid: Array,
    taxonomy: dict[str, Any],
    policy: dict[str, Any],
) -> tuple[float, dict[str, float | int]]:
    factor_count = low_structure_count(features, taxonomy)
    gate = int(factor_count >= int(policy["min_active"]))
    outside_rate, outside_ratio = reference_interval_conflict(reference, quantile_levels, quantile_grid)
    if not gate:
        return 0.0, {
            "low_structure_factor_count": factor_count,
            "base_gate_active": 0,
            "shield_active": 0,
            "conflict_override": 0,
            "reference_outside_interval_rate": outside_rate,
            "reference_outside_interval_mean_ratio": outside_ratio,
        }

    weight = float(policy["weight"]) + max(0, factor_count - int(policy["min_active"])) * float(
        policy["factor_step"]
    )
    weight = min(float(policy["max_weight"]), weight)
    shield = 0
    override = 0
    if outside_rate >= float(policy["conflict_threshold"]) and weight > float(policy["shield_cap"]):
        degeneracy_compatible = (
            features["horizon_context_ratio"] <= float(policy["degeneracy_hcr_threshold"])
            and features["trend_strength"] >= float(policy["degeneracy_trend_threshold"])
        )
        if degeneracy_compatible:
            override = 1
        else:
            weight = float(policy["shield_cap"])
            shield = 1
    return weight, {
        "low_structure_factor_count": factor_count,
        "base_gate_active": gate,
        "shield_active": shield,
        "conflict_override": override,
        "reference_outside_interval_rate": outside_rate,
        "reference_outside_interval_mean_ratio": outside_ratio,
    }


def interval_head_grid(
    native_mean: Array,
    reference: Array,
    quantile_levels: list[float],
    quantile_grid: Array,
    weight: float,
    scale: float,
) -> Array:
    levels = np.asarray(quantile_levels, dtype=float)
    grid = np.asarray(quantile_grid, dtype=float)
    native_center = grid[:, int(np.argmin(np.abs(levels - 0.5)))]
    repaired_center = np.asarray(native_mean, dtype=float) + weight * (
        np.asarray(reference, dtype=float) - np.asarray(native_mean, dtype=float)
    )
    return np.sort(repaired_center[:, None] + scale * (grid - native_center[:, None]), axis=1)


def frozen_action_grids(
    native_mean: Array,
    reference: Array,
    quantile_levels: list[float],
    quantile_grid: Array,
    features: dict[str, float],
    taxonomy: dict[str, Any],
    method: dict[str, Any],
) -> tuple[dict[str, Array], dict[str, float | int]]:
    policy = method["base_cpr_policy"]
    head = method["smooth_interval_head"]
    weight, diagnostics = base_cpr_weight(
        features,
        reference,
        quantile_levels,
        quantile_grid,
        taxonomy,
        policy,
    )
    if structured_guard(features, head):
        weight = min(weight, float(head["structured_weight_cap"]))
        smooth_scale = float(head["structured_scale"])
    else:
        score = smooth_width_score(features["native_width_ratio"], features, head)
        smooth_scale = float(head["low_scale"]) + (
            float(head["high_scale"]) - float(head["low_scale"])
        ) * score
    expert_weight = min(1.0, 1.25 * weight)
    grids = {
        "native_tsfm": np.asarray(quantile_grid, dtype=float),
        "drcr_point": interval_head_grid(
            native_mean, reference, quantile_levels, quantile_grid, weight=weight, scale=1.0
        ),
        "drcr_expert_pull_1.25_cap_1.10": interval_head_grid(
            native_mean,
            reference,
            quantile_levels,
            quantile_grid,
            weight=expert_weight,
            scale=min(smooth_scale, 1.10),
        ),
    }
    diagnostics.update(
        {
            "base_effective_weight": weight,
            "expert_effective_weight": expert_weight,
            "smooth_interval_scale": smooth_scale,
            "structured_guard": int(structured_guard(features, head)),
        }
    )
    return grids, diagnostics
