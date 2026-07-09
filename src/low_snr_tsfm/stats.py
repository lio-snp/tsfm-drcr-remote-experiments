"""Statistical tests for forecast-comparison reports."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, erfc, sqrt

import numpy as np


Array = np.ndarray
EPS = 1e-12


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _normal_sf(x: float) -> float:
    return 0.5 * erfc(x / sqrt(2.0))


def _chi2_df1_sf(x: float) -> float:
    return erfc(sqrt(max(x, 0.0) / 2.0))


@dataclass(frozen=True)
class TestResult:
    statistic: float
    p_value: float


def diebold_mariano(
    loss_model: Array,
    loss_baseline: Array,
    horizon: int = 1,
    alternative: str = "two-sided",
) -> TestResult:
    """Diebold-Mariano test with a simple Newey-West variance estimate."""

    lm = np.asarray(loss_model, dtype=float)
    lb = np.asarray(loss_baseline, dtype=float)
    if lm.shape != lb.shape:
        raise ValueError("loss arrays must have the same shape")
    if lm.size < 2:
        raise ValueError("Need at least two loss differences")
    if alternative not in {"two-sided", "less", "greater"}:
        raise ValueError("alternative must be two-sided, less, or greater")

    d = lm - lb
    n = d.size
    d_mean = float(np.mean(d))
    max_lag = max(0, int(horizon) - 1)
    centered = d - d_mean
    gamma0 = float(np.dot(centered, centered) / n)
    var = gamma0
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1)
        var += 2.0 * weight * cov
    se = np.sqrt(max(var, EPS) / n)
    statistic = d_mean / se

    if alternative == "two-sided":
        p_value = 2.0 * _normal_sf(abs(statistic))
    elif alternative == "less":
        p_value = _normal_cdf(statistic)
    else:
        p_value = _normal_sf(statistic)
    return TestResult(float(statistic), float(p_value))


def benjamini_hochberg(p_values: Array, alpha: float = 0.10) -> tuple[Array, Array]:
    """Return rejection flags and BH-adjusted q-values."""

    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    m = p.size
    if m == 0:
        return np.asarray([], dtype=bool), np.asarray([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = alpha * np.arange(1, m + 1) / m
    below = ranked <= thresholds
    rejected_sorted = np.zeros(m, dtype=bool)
    if np.any(below):
        k = np.max(np.where(below)[0])
        rejected_sorted[: k + 1] = True
    q_sorted = np.minimum.accumulate((m / np.arange(m, 0, -1)) * ranked[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    rejected = np.zeros(m, dtype=bool)
    q_values = np.zeros(m, dtype=float)
    rejected[order] = rejected_sorted
    q_values[order] = q_sorted
    return rejected, q_values


def kupiec_pof(exceptions: Array, alpha: float) -> TestResult:
    """Kupiec proportion-of-failures unconditional coverage test."""

    exc = np.asarray(exceptions, dtype=int)
    if exc.ndim != 1 or exc.size == 0:
        raise ValueError("exceptions must be a non-empty one-dimensional array")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    n = exc.size
    x = int(np.sum(exc))
    phat = np.clip(x / n, EPS, 1.0 - EPS)
    alpha_c = np.clip(alpha, EPS, 1.0 - EPS)
    ll_null = (n - x) * np.log(1 - alpha_c) + x * np.log(alpha_c)
    ll_alt = (n - x) * np.log(1 - phat) + x * np.log(phat)
    lr = -2.0 * (ll_null - ll_alt)
    return TestResult(float(lr), float(_chi2_df1_sf(lr)))


def christoffersen_independence(exceptions: Array) -> TestResult:
    """Christoffersen independence test for exception clustering."""

    exc = np.asarray(exceptions, dtype=int)
    if exc.ndim != 1 or exc.size < 2:
        raise ValueError("exceptions must contain at least two observations")
    n00 = n01 = n10 = n11 = 0
    for prev, curr in zip(exc[:-1], exc[1:]):
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        else:
            n11 += 1

    pi = np.clip((n01 + n11) / max(n00 + n01 + n10 + n11, 1), EPS, 1.0 - EPS)
    pi01 = np.clip(n01 / max(n00 + n01, 1), EPS, 1.0 - EPS)
    pi11 = np.clip(n11 / max(n10 + n11, 1), EPS, 1.0 - EPS)

    ll_null = (n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
    ll_alt = n00 * np.log(1 - pi01) + n01 * np.log(pi01) + n10 * np.log(1 - pi11) + n11 * np.log(pi11)
    lr = -2.0 * (ll_null - ll_alt)
    return TestResult(float(lr), float(_chi2_df1_sf(lr)))
