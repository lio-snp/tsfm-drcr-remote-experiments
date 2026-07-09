"""Simple baselines that every TSFM comparison must beat."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class BaselineForecast:
    name: str
    values: Array


def _as_1d(context: Array) -> Array:
    values = np.asarray(context, dtype=float)
    if values.ndim != 1:
        raise ValueError("Expected a univariate context array")
    if values.size == 0:
        raise ValueError("Context must contain at least one value")
    return values


def zero_forecast(context: Array, horizon: int) -> Array:
    _as_1d(context)
    return np.zeros(horizon, dtype=float)


def mean_forecast(context: Array, horizon: int) -> Array:
    values = _as_1d(context)
    return np.full(horizon, np.nanmean(values), dtype=float)


def naive_forecast(context: Array, horizon: int) -> Array:
    values = _as_1d(context)
    return np.full(horizon, values[-1], dtype=float)


def seasonal_naive_forecast(context: Array, horizon: int, season_length: int) -> Array:
    values = _as_1d(context)
    if season_length <= 0:
        raise ValueError("season_length must be positive")
    if values.size < season_length:
        return naive_forecast(values, horizon)
    pattern = values[-season_length:]
    reps = int(np.ceil(horizon / season_length))
    return np.tile(pattern, reps)[:horizon].astype(float)


def drift_forecast(context: Array, horizon: int) -> Array:
    values = _as_1d(context)
    if values.size < 2:
        return naive_forecast(values, horizon)
    slope = (values[-1] - values[0]) / (values.size - 1)
    return values[-1] + slope * np.arange(1, horizon + 1)


def linear_ar_forecast(context: Array, horizon: int, lags: int = 12, ridge: float = 1e-8) -> Array:
    """Recursive linear autoregression fit only on the supplied context."""

    values = _as_1d(context)
    lags = int(max(1, min(lags, values.size - 1)))
    if values.size <= lags + 1:
        return mean_forecast(values, horizon)

    rows = []
    target = []
    for idx in range(lags, values.size):
        rows.append(values[idx - lags : idx][::-1])
        target.append(values[idx])
    x = np.asarray(rows)
    y = np.asarray(target)
    x_design = np.column_stack([np.ones(x.shape[0]), x])
    penalty = ridge * np.eye(x_design.shape[1])
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ y)

    history = list(values.astype(float))
    preds = []
    for _ in range(horizon):
        lag_values = np.asarray(history[-lags:][::-1])
        pred = float(np.r_[1.0, lag_values] @ coef)
        preds.append(pred)
        history.append(pred)
    return np.asarray(preds)


def all_simple_baselines(
    context: Array,
    horizon: int,
    season_length: int | None = None,
    ar_lags: int = 12,
) -> list[BaselineForecast]:
    """Return the locked minimal baseline menu for a forecast window."""

    forecasts = [
        BaselineForecast("zero", zero_forecast(context, horizon)),
        BaselineForecast("mean", mean_forecast(context, horizon)),
        BaselineForecast("naive", naive_forecast(context, horizon)),
        BaselineForecast("drift", drift_forecast(context, horizon)),
        BaselineForecast("linear_ar", linear_ar_forecast(context, horizon, lags=ar_lags)),
    ]
    if season_length is not None:
        forecasts.append(
            BaselineForecast(
                f"seasonal_naive_{season_length}",
                seasonal_naive_forecast(context, horizon, season_length),
            )
        )
    return forecasts
