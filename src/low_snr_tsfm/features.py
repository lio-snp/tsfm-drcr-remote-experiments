"""Observable features for ex-ante failure prediction."""

from __future__ import annotations

import numpy as np


Array = np.ndarray
EPS = 1e-12


def autocorrelation(values: Array, lag: int = 1) -> float:
    x = np.asarray(values, dtype=float)
    if lag <= 0:
        raise ValueError("lag must be positive")
    if x.size <= lag:
        return 0.0
    a = x[:-lag] - np.nanmean(x[:-lag])
    b = x[lag:] - np.nanmean(x[lag:])
    denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
    if denom <= EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def autocorrelation_strength(values: Array, max_lag: int = 24) -> float:
    x = np.asarray(values, dtype=float)
    max_lag = int(max(1, min(max_lag, x.size - 2)))
    if max_lag <= 0:
        return 0.0
    return float(max(abs(autocorrelation(x, lag)) for lag in range(1, max_lag + 1)))


def spectral_entropy(values: Array) -> float:
    x = np.asarray(values, dtype=float)
    if x.size < 4:
        return 0.0
    centered = x - np.nanmean(x)
    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    spectrum = spectrum[1:]
    total = np.sum(spectrum)
    if total <= EPS:
        return 0.0
    p = spectrum / total
    entropy = -np.sum(p * np.log(p + EPS))
    return float(entropy / np.log(p.size + EPS))


def coefficient_of_variation(values: Array) -> float:
    x = np.asarray(values, dtype=float)
    return float(np.nanstd(x) / (abs(np.nanmean(x)) + EPS))


def zero_ratio(values: Array, atol: float = 1e-12) -> float:
    x = np.asarray(values, dtype=float)
    if x.size == 0:
        return 0.0
    return float(np.mean(np.isclose(x, 0.0, atol=atol)))


def spike_frequency(values: Array, z_threshold: float = 3.0) -> float:
    x = np.asarray(values, dtype=float)
    if x.size < 3:
        return 0.0
    diffs = np.diff(x)
    scale = np.nanstd(diffs)
    if scale <= EPS:
        return 0.0
    z = np.abs((diffs - np.nanmean(diffs)) / scale)
    return float(np.mean(z > z_threshold))


def trend_strength(values: Array) -> float:
    x = np.asarray(values, dtype=float)
    if x.size < 3:
        return 0.0
    t = np.arange(x.size, dtype=float)
    design = np.column_stack([np.ones_like(t), t])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    fitted = design @ coef
    resid_var = np.var(x - fitted)
    total_var = np.var(x)
    return float(max(0.0, 1.0 - resid_var / (total_var + EPS)))


def seasonality_strength(values: Array, period: int) -> float:
    x = np.asarray(values, dtype=float)
    if period <= 1 or x.size < 2 * period:
        return 0.0
    centered = x - np.nanmean(x)
    return abs(autocorrelation(centered, period))


def changepoint_density(values: Array, z_threshold: float = 3.5) -> float:
    x = np.asarray(values, dtype=float)
    if x.size < 8:
        return 0.0
    diffs = np.diff(x)
    median = np.median(diffs)
    mad = np.median(np.abs(diffs - median)) + EPS
    robust_z = 0.6745 * np.abs(diffs - median) / mad
    return float(np.mean(robust_z > z_threshold))


def kurtosis_excess(values: Array) -> float:
    x = np.asarray(values, dtype=float)
    if x.size < 4:
        return 0.0
    centered = x - np.nanmean(x)
    var = np.nanmean(centered**2)
    if var <= EPS:
        return 0.0
    return float(np.nanmean(centered**4) / (var**2) - 3.0)


def feature_vector(
    values: Array,
    horizon: int,
    context_length: int,
    period: int | None = None,
) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    features = {
        "autocorrelation_strength": autocorrelation_strength(x),
        "spectral_entropy": spectral_entropy(x),
        "coefficient_of_variation": coefficient_of_variation(x),
        "zero_ratio": zero_ratio(x),
        "spike_frequency": spike_frequency(x),
        "trend_strength": trend_strength(x),
        "changepoint_density": changepoint_density(x),
        "kurtosis_excess": kurtosis_excess(x),
        "missingness": float(np.mean(~np.isfinite(x))) if x.size else 0.0,
        "horizon_context_ratio": float(horizon / max(context_length, 1)),
    }
    if period is not None:
        features["seasonality_strength"] = seasonality_strength(x, period)
    return features
