"""Post-hoc forecast repair probes.

These helpers intentionally implement a conservative BMA-style probe rather
than the full traffic-paper BMA method. The probe mixes a TSFM forecast with a
non-leaky local reference forecast and separately reports an interval hull.
"""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def adaptive_reference_weight(
    reference_weight: float,
    active_factor_count: int,
    min_active_factors: int,
    factor_step: float = 0.0,
    max_reference_weight: float | None = None,
) -> float:
    """Return a clamped reference weight that grows with active risk factors."""

    if not 0.0 <= reference_weight <= 1.0:
        raise ValueError("reference_weight must be in [0, 1]")
    if factor_step < 0.0:
        raise ValueError("factor_step must be non-negative")
    cap = 1.0 if max_reference_weight is None else max_reference_weight
    if not 0.0 <= cap <= 1.0:
        raise ValueError("max_reference_weight must be in [0, 1]")
    if active_factor_count < min_active_factors:
        return 0.0
    excess = max(0, active_factor_count - min_active_factors)
    return min(cap, reference_weight + factor_step * excess)


def convex_mixture(model_forecast: Array, reference_forecast: Array, reference_weight: float) -> Array:
    """Blend model and reference forecasts with a fixed reference weight."""

    if not 0.0 <= reference_weight <= 1.0:
        raise ValueError("reference_weight must be in [0, 1]")
    model = np.asarray(model_forecast, dtype=float)
    reference = np.asarray(reference_forecast, dtype=float)
    if model.shape != reference.shape:
        raise ValueError(f"Shape mismatch: {model.shape} != {reference.shape}")
    return (1.0 - reference_weight) * model + reference_weight * reference


def blended_interval(lower: Array, upper: Array, reference_forecast: Array, reference_weight: float) -> tuple[Array, Array]:
    """Blend model quantile bounds toward a deterministic reference forecast."""

    lower_blend = convex_mixture(lower, reference_forecast, reference_weight)
    upper_blend = convex_mixture(upper, reference_forecast, reference_weight)
    return np.minimum(lower_blend, upper_blend), np.maximum(lower_blend, upper_blend)


def hull_interval(lower: Array, upper: Array, reference_forecast: Array) -> tuple[Array, Array]:
    """Return the smallest interval containing model bounds and the reference."""

    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    reference = np.asarray(reference_forecast, dtype=float)
    if not (lo.shape == hi.shape == reference.shape):
        raise ValueError("lower, upper, and reference_forecast must have the same shape")
    return np.minimum(lo, reference), np.maximum(hi, reference)
