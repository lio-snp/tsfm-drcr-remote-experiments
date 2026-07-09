"""Accuracy, degeneration, and calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPS = 1e-12
Array = np.ndarray


def _pair(y_true: Array, y_pred: Array) -> tuple[Array, Array]:
    y = np.asarray(y_true, dtype=float)
    f = np.asarray(y_pred, dtype=float)
    if y.shape != f.shape:
        raise ValueError(f"Shape mismatch: {y.shape} != {f.shape}")
    return y, f


def mae(y_true: Array, y_pred: Array) -> float:
    y, f = _pair(y_true, y_pred)
    return float(np.mean(np.abs(y - f)))


def rmse(y_true: Array, y_pred: Array) -> float:
    y, f = _pair(y_true, y_pred)
    return float(np.sqrt(np.mean((y - f) ** 2)))


def smape(y_true: Array, y_pred: Array, eps: float = EPS) -> float:
    y, f = _pair(y_true, y_pred)
    return float(np.mean(2.0 * np.abs(y - f) / (np.abs(y) + np.abs(f) + eps)))


def wape(y_true: Array, y_pred: Array, eps: float = EPS) -> float:
    y, f = _pair(y_true, y_pred)
    return float(np.sum(np.abs(y - f)) / (np.sum(np.abs(y)) + eps))


def mase(y_true: Array, y_pred: Array, insample: Array, season_length: int = 1, eps: float = EPS) -> float:
    y, f = _pair(y_true, y_pred)
    x = np.asarray(insample, dtype=float)
    if x.size <= season_length:
        scale = np.mean(np.abs(x - np.nanmean(x))) if x.size else 0.0
    else:
        scale = np.mean(np.abs(x[season_length:] - x[:-season_length]))
    return float(np.mean(np.abs(y - f)) / (scale + eps))


def relative_error_ratio(model_error: float, baseline_error: float, eps: float = EPS) -> float:
    return float(model_error / (baseline_error + eps))


def first_difference(values: Array) -> Array:
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return np.zeros(0, dtype=float)
    return np.diff(arr)


def forecast_variance_ratio(y_true: Array, y_pred: Array, eps: float = EPS) -> float:
    dy = first_difference(y_true)
    df = first_difference(y_pred)
    return float(np.var(df) / (np.var(dy) + eps))


def prediction_amplitude_ratio(y_true: Array, y_pred: Array, eps: float = EPS) -> float:
    dy = first_difference(y_true)
    df = first_difference(y_pred)
    return float(np.mean(np.abs(df)) / (np.mean(np.abs(dy)) + eps))


def excess_movement_score(y_true: Array, y_pred: Array) -> float:
    return float(max(0.0, prediction_amplitude_ratio(y_true, y_pred) - 1.0))


def flatness_score(y_true: Array, y_pred: Array) -> float:
    ratio = prediction_amplitude_ratio(y_true, y_pred)
    return float(1.0 - min(1.0, ratio))


def spike_recall(y_true: Array, y_pred: Array, k: int) -> float:
    dy = np.abs(first_difference(y_true))
    df = np.abs(first_difference(y_pred))
    if dy.size == 0 or df.size == 0:
        return 0.0
    k = int(max(1, min(k, dy.size, df.size)))
    true_idx = set(np.argpartition(dy, -k)[-k:].tolist())
    pred_idx = set(np.argpartition(df, -k)[-k:].tolist())
    return float(len(true_idx & pred_idx) / k)


def peak_timing_error(y_true: Array, y_pred: Array, k: int) -> float:
    dy = np.abs(first_difference(y_true))
    df = np.abs(first_difference(y_pred))
    if dy.size == 0 or df.size == 0:
        return float("nan")
    k = int(max(1, min(k, df.size)))
    true_peak = int(np.argmax(dy))
    pred_top = np.argpartition(df, -k)[-k:]
    return float(np.min(np.abs(pred_top - true_peak)))


def pinball_loss(y_true: Array, q_pred: Array, tau: float) -> float:
    if not 0 < tau < 1:
        raise ValueError("tau must be in (0, 1)")
    y, q = _pair(y_true, q_pred)
    err = y - q
    return float(np.mean(np.maximum(tau * err, (tau - 1.0) * err)))


def mean_weighted_quantile_loss(
    y_true: Array,
    quantile_predictions: Array,
    levels: list[float] | Array,
    eps: float = EPS,
) -> float:
    """Mean weighted quantile loss over an available quantile grid.

    The denominator follows the usual WQL normalization by the target
    magnitude. Endpoint levels 0 and 1 are skipped because pinball loss is
    defined for open-interval quantiles.
    """

    y = np.asarray(y_true, dtype=float)
    q = np.asarray(quantile_predictions, dtype=float)
    taus = np.asarray(levels, dtype=float).reshape(-1)
    if q.ndim != 2:
        raise ValueError("quantile_predictions must have shape (horizon, n_quantiles)")
    if q.shape != (y.size, taus.size):
        raise ValueError(f"Shape mismatch: y={y.shape}, quantiles={q.shape}, levels={taus.shape}")
    valid = [(idx, float(tau)) for idx, tau in enumerate(taus) if 0.0 < float(tau) < 1.0]
    if not valid:
        raise ValueError("At least one quantile level in (0, 1) is required")
    normalizer = float(np.mean(np.abs(y))) if y.size else 0.0
    if normalizer <= eps:
        normalizer = eps
    losses = [pinball_loss(y, q[:, idx], tau) for idx, tau in valid]
    return float(2.0 * np.mean(losses) / normalizer)


def empirical_coverage(y_true: Array, lower: Array, upper: Array) -> float:
    y = np.asarray(y_true, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError("Coverage arrays must have the same shape")
    return float(np.mean((lo <= y) & (y <= hi)))


def interval_width(lower: Array, upper: Array) -> float:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if lo.shape != hi.shape:
        raise ValueError("Interval bounds must have the same shape")
    return float(np.mean(hi - lo))


def sample_crps(samples: Array, y_true: Array) -> float:
    """CRPS for sample forecasts using the standard empirical estimator."""

    sample_arr = np.asarray(samples, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if sample_arr.ndim != 2:
        raise ValueError("samples must have shape (num_samples, horizon)")
    if sample_arr.shape[1] != y.size:
        raise ValueError("sample horizon must match y_true")
    term1 = np.mean(np.abs(sample_arr - y[None, :]), axis=0)
    pairwise = np.abs(sample_arr[:, None, :] - sample_arr[None, :, :])
    term2 = 0.5 * np.mean(pairwise, axis=(0, 1))
    return float(np.mean(term1 - term2))


@dataclass(frozen=True)
class DegenerationFlags:
    relative_error_ratio: float
    excess_variance: bool
    over_smoothing: bool
    calibration_failure: bool | None = None


def classify_degeneration(
    y_true: Array,
    y_pred: Array,
    model_error: float,
    baseline_error: float,
    delta: float = 0.05,
    tau_var: float = 2.0,
    tau_amp: float = 1.5,
    tau_flat: float = 0.6,
) -> DegenerationFlags:
    rer = relative_error_ratio(model_error, baseline_error)
    failed = rer > 1.0 + delta
    fvr = forecast_variance_ratio(y_true, y_pred)
    par = prediction_amplitude_ratio(y_true, y_pred)
    flat = flatness_score(y_true, y_pred)
    return DegenerationFlags(
        relative_error_ratio=rer,
        excess_variance=bool(failed and (fvr > tau_var or par > tau_amp)),
        over_smoothing=bool(failed and flat > tau_flat),
        calibration_failure=None,
    )


def horizon_degradation_slope(relative_errors: Array) -> float:
    values = np.asarray(relative_errors, dtype=float)
    if values.size < 2:
        return 0.0
    x = np.arange(1, values.size + 1, dtype=float)
    x = x - x.mean()
    y = values - values.mean()
    return float(np.dot(x, y) / (np.dot(x, x) + EPS))
