"""Paired scaling summaries for Chronos-Bolt capacity experiments."""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Iterable

import numpy as np


def finite_float(value: object, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def linear_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.dot(x, x))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def percentile_interval(values: Iterable[float], low: float = 2.5, high: float = 97.5) -> dict[str, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return {"low": float("nan"), "high": float("nan")}
    return {"low": float(np.percentile(arr, low)), "high": float(np.percentile(arr, high))}


def log1p_nonnegative(value: object) -> float:
    numeric = finite_float(value)
    if not np.isfinite(numeric):
        return float("nan")
    return float(np.log1p(max(0.0, numeric)))


def trimmed_mean(values: Iterable[float], proportion: float = 0.1) -> float:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=float)
    if arr.size == 0:
        return float("nan")
    if arr.size < 3 or proportion <= 0:
        return float(np.mean(arr))
    cut = int(np.floor(arr.size * proportion))
    if cut == 0 or cut * 2 >= arr.size:
        return float(np.mean(arr))
    arr.sort()
    return float(np.mean(arr[cut:-cut]))


def model_rate_slope(
    rows: list[dict[str, object]],
    *,
    outcome: str,
    x_key: str = "log10_params_m",
) -> float:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)

    xs: list[float] = []
    ys: list[float] = []
    for model_rows in grouped.values():
        x = finite_float(model_rows[0].get(x_key))
        values = [finite_float(row.get(outcome)) for row in model_rows]
        values = [value for value in values if np.isfinite(value)]
        if np.isfinite(x) and values:
            xs.append(x)
            ys.append(float(mean(values)))
    return linear_slope(xs, ys)


def bootstrap_target_slopes(
    rows: list[dict[str, object]],
    *,
    target_key: str,
    outcome: str,
    x_key: str = "log10_params_m",
    n_bootstrap: int = 1000,
    seed: int = 7,
) -> tuple[float, list[float]]:
    target_rows = [row for row in rows if row.get("target_key") == target_key]
    point = model_rate_slope(target_rows, outcome=outcome, x_key=x_key)

    by_unit: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in target_rows:
        by_unit[str(row["paired_unit_id"])].append(row)
    units = sorted(by_unit)
    if not units:
        return point, []

    rng = np.random.default_rng(seed)
    slopes: list[float] = []
    for _ in range(n_bootstrap):
        sampled_rows: list[dict[str, object]] = []
        sampled_units = rng.choice(units, size=len(units), replace=True)
        for draw_idx, unit in enumerate(sampled_units):
            for row in by_unit[str(unit)]:
                sampled = dict(row)
                sampled["paired_unit_id"] = f"{draw_idx}:{unit}"
                sampled_rows.append(sampled)
        slope = model_rate_slope(sampled_rows, outcome=outcome, x_key=x_key)
        if np.isfinite(slope):
            slopes.append(slope)
    return point, slopes


def bootstrap_interactions(
    rows: list[dict[str, object]],
    *,
    failure_target_key: str,
    control_target_keys: list[str],
    outcome: str,
    x_key: str = "log10_params_m",
    n_bootstrap: int = 1000,
    seed: int = 7,
) -> dict[str, object]:
    failure_point, failure_draws = bootstrap_target_slopes(
        rows,
        target_key=failure_target_key,
        outcome=outcome,
        x_key=x_key,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    target_results: dict[str, object] = {
        failure_target_key: {
            "point_slope": failure_point,
            "bootstrap_ci": percentile_interval(failure_draws),
            "n_bootstrap": len(failure_draws),
        }
    }
    interactions: dict[str, object] = {}
    for offset, control_key in enumerate(control_target_keys, start=1):
        control_point, control_draws = bootstrap_target_slopes(
            rows,
            target_key=control_key,
            outcome=outcome,
            x_key=x_key,
            n_bootstrap=n_bootstrap,
            seed=seed + offset,
        )
        target_results[control_key] = {
            "point_slope": control_point,
            "bootstrap_ci": percentile_interval(control_draws),
            "n_bootstrap": len(control_draws),
        }
        point_interaction = failure_point - control_point
        n_draws = min(len(failure_draws), len(control_draws))
        draws = [failure_draws[idx] - control_draws[idx] for idx in range(n_draws)]
        interactions[f"{failure_target_key}_minus_{control_key}"] = {
            "point_slope_difference": point_interaction,
            "bootstrap_ci": percentile_interval(draws),
            "n_bootstrap": len(draws),
        }
    return {
        "outcome": outcome,
        "x_key": x_key,
        "failure_target_key": failure_target_key,
        "control_target_keys": control_target_keys,
        "target_slopes": target_results,
        "interactions": interactions,
    }
