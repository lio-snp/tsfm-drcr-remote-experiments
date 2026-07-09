"""Finite-sample risk-control helpers for selective forecast repair."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class RiskTestResult:
    """Result of a one-sided LTT risk test for one policy."""

    policy_id: str
    n: int
    empirical_risk: float
    alpha: float
    p_value: float
    corrected_threshold: float
    accepted: bool
    correction: str
    risk_count: int
    ucb_hoeffding: float


def _as_bounded_array(losses: Sequence[float]) -> np.ndarray:
    arr = np.asarray(losses, dtype=float)
    if arr.ndim != 1:
        raise ValueError("losses must be one-dimensional")
    if arr.size == 0:
        raise ValueError("losses must be non-empty")
    if np.any(~np.isfinite(arr)):
        raise ValueError("losses must be finite")
    if np.any((arr < -1e-12) | (arr > 1.0 + 1e-12)):
        raise ValueError("losses must be bounded in [0, 1]")
    return np.clip(arr, 0.0, 1.0)


def hoeffding_upper_bound(empirical_risk: float, n: int, delta: float) -> float:
    """One-sided Hoeffding upper confidence bound for a bounded mean."""

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    radius = math.sqrt(math.log(1.0 / delta) / (2.0 * n))
    return min(1.0, float(empirical_risk) + radius)


def binomial_lower_tail_p_value(risk_count: int, n: int, alpha: float) -> float:
    """Return P[Binomial(n, alpha) <= risk_count].

    For binary LTT losses, this is the least-favorable p-value for testing
    H0: risk > alpha against the alternative risk <= alpha.  The lower-tail
    probability decreases when the observed number of harms is unusually small.
    """

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    k = int(risk_count)
    if k < 0 or k > n:
        raise ValueError("risk_count must lie in [0, n]")
    terms = []
    for i in range(k + 1):
        terms.append(math.comb(n, i) * (alpha**i) * ((1.0 - alpha) ** (n - i)))
    return float(min(1.0, sum(terms)))


def hoeffding_lower_tail_p_value(empirical_risk: float, n: int, alpha: float) -> float:
    """Conservative lower-tail p-value for bounded non-binary losses."""

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if empirical_risk > alpha:
        return 1.0
    return float(math.exp(-2.0 * n * (alpha - empirical_risk) ** 2))


def ltt_risk_tests(
    losses_by_policy: Mapping[str, Sequence[float]],
    *,
    alpha: float,
    delta: float,
    correction: str = "holm",
    binary: bool = True,
) -> list[RiskTestResult]:
    """Run Learn-Then-Test style multiple testing over fixed policies.

    A policy is accepted when the null hypothesis ``risk > alpha`` is rejected
    after the requested family-wise error correction.
    """

    if not losses_by_policy:
        raise ValueError("losses_by_policy must be non-empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if correction not in {"bonferroni", "holm"}:
        raise ValueError("correction must be 'bonferroni' or 'holm'")

    base_rows = []
    for policy_id, losses in losses_by_policy.items():
        arr = _as_bounded_array(losses)
        empirical = float(np.mean(arr))
        if binary:
            rounded = np.rint(arr)
            if np.any(np.abs(arr - rounded) > 1e-8):
                raise ValueError("binary=True requires 0/1 losses")
            risk_count = int(np.sum(rounded))
            p_value = binomial_lower_tail_p_value(risk_count, int(arr.size), alpha)
        else:
            risk_count = int(round(float(np.sum(arr))))
            p_value = hoeffding_lower_tail_p_value(empirical, int(arr.size), alpha)
        base_rows.append(
            {
                "policy_id": str(policy_id),
                "n": int(arr.size),
                "empirical_risk": empirical,
                "risk_count": risk_count,
                "p_value": p_value,
                "ucb_hoeffding": hoeffding_upper_bound(empirical, int(arr.size), delta),
            }
        )

    m = len(base_rows)
    accepted: set[str] = set()
    thresholds: dict[str, float] = {}
    if correction == "bonferroni":
        threshold = delta / m
        for row in base_rows:
            thresholds[row["policy_id"]] = threshold
            if row["p_value"] <= threshold:
                accepted.add(row["policy_id"])
    else:
        ordered = sorted(base_rows, key=lambda row: (float(row["p_value"]), str(row["policy_id"])))
        for index, row in enumerate(ordered):
            threshold = delta / (m - index)
            thresholds[row["policy_id"]] = threshold
            if row["p_value"] <= threshold:
                accepted.add(row["policy_id"])
            else:
                for later in ordered[index + 1 :]:
                    thresholds[later["policy_id"]] = delta / (m - index)
                break

    return [
        RiskTestResult(
            policy_id=row["policy_id"],
            n=int(row["n"]),
            empirical_risk=float(row["empirical_risk"]),
            alpha=alpha,
            p_value=float(row["p_value"]),
            corrected_threshold=float(thresholds.get(str(row["policy_id"]), delta / m)),
            accepted=str(row["policy_id"]) in accepted,
            correction=correction,
            risk_count=int(row["risk_count"]),
            ucb_hoeffding=float(row["ucb_hoeffding"]),
        )
        for row in sorted(base_rows, key=lambda item: str(item["policy_id"]))
    ]


def select_highest_utility_certified(
    tests: Sequence[RiskTestResult],
    utilities: Mapping[str, float],
    *,
    fallback_policy_id: str,
) -> str:
    """Select the highest-utility certified policy, or a fallback."""

    accepted = [test for test in tests if test.accepted]
    if not accepted:
        return fallback_policy_id
    accepted_ids = {test.policy_id for test in accepted}
    if fallback_policy_id not in utilities:
        raise ValueError("fallback_policy_id must have a utility entry")
    candidates = accepted_ids & set(utilities)
    if not candidates:
        return fallback_policy_id
    return max(candidates, key=lambda policy_id: (float(utilities[policy_id]), policy_id))
