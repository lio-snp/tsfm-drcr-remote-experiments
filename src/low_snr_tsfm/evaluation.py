"""Rolling-origin evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

from .baselines import BaselineForecast, all_simple_baselines
from .metrics import mae, rmse, relative_error_ratio


Array = np.ndarray
Forecaster = Callable[[Array, int], Array]


@dataclass(frozen=True)
class ForecastWindow:
    origin: int
    context: Array
    target: Array


def rolling_windows(series: Array, context_length: int, horizon: int, step: int | None = None) -> Iterable[ForecastWindow]:
    values = np.asarray(series, dtype=float)
    if context_length <= 0 or horizon <= 0:
        raise ValueError("context_length and horizon must be positive")
    step = horizon if step is None else step
    if step <= 0:
        raise ValueError("step must be positive")
    last_origin = values.size - horizon
    for origin in range(context_length, last_origin + 1, step):
        yield ForecastWindow(
            origin=origin,
            context=values[origin - context_length : origin],
            target=values[origin : origin + horizon],
        )


def evaluate_baselines(
    series: Array,
    context_length: int,
    horizon: int,
    step: int | None = None,
    season_length: int | None = None,
    ar_lags: int = 12,
) -> object:
    """Evaluate baselines and return a pandas DataFrame when pandas is installed."""

    rows = evaluate_baselines_records(
        series,
        context_length,
        horizon,
        step=step,
        season_length=season_length,
        ar_lags=ar_lags,
    )
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pandas is required for evaluate_baselines; use evaluate_baselines_records instead") from exc
    return pd.DataFrame(rows)


def evaluate_baselines_records(
    series: Array,
    context_length: int,
    horizon: int,
    step: int | None = None,
    season_length: int | None = None,
    ar_lags: int = 12,
) -> list[dict[str, float | int | str]]:
    rows = []
    for window in rolling_windows(series, context_length, horizon, step):
        for forecast in all_simple_baselines(
            window.context,
            horizon,
            season_length=season_length,
            ar_lags=ar_lags,
        ):
            rows.append(
                {
                    "origin": window.origin,
                    "model": forecast.name,
                    "mae": mae(window.target, forecast.values),
                    "rmse": rmse(window.target, forecast.values),
                }
            )
    return rows


def best_baseline_by_validation_records(
    validation_rows: list[dict[str, float | int | str]],
    metric: str = "mae",
) -> str:
    if not validation_rows:
        raise ValueError("validation_rows is empty")
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in validation_rows:
        model = str(row["model"])
        sums[model] = sums.get(model, 0.0) + float(row[metric])
        counts[model] = counts.get(model, 0) + 1
    return min(sums, key=lambda name: sums[name] / counts[name])


def best_baseline_by_validation(validation_frame: object, metric: str = "mae") -> str:
    if hasattr(validation_frame, "empty"):
        if validation_frame.empty:
            raise ValueError("validation_frame is empty")
        grouped = validation_frame.groupby("model", as_index=False)[metric].mean()
        return str(grouped.sort_values(metric).iloc[0]["model"])
    return best_baseline_by_validation_records(validation_frame, metric=metric)


def evaluate_forecaster_against_baseline(
    series: Array,
    forecaster_name: str,
    forecaster: Forecaster,
    baseline_name: str,
    baseline_factory: Callable[[Array, int], Array],
    context_length: int,
    horizon: int,
    step: int | None = None,
) -> object:
    rows = []
    for window in rolling_windows(series, context_length, horizon, step):
        model_pred = forecaster(window.context, horizon)
        base_pred = baseline_factory(window.context, horizon)
        model_mae = mae(window.target, model_pred)
        base_mae = mae(window.target, base_pred)
        rows.append(
            {
                "origin": window.origin,
                "model": forecaster_name,
                "baseline": baseline_name,
                "mae": model_mae,
                "baseline_mae": base_mae,
                "relative_error_ratio": relative_error_ratio(model_mae, base_mae),
            }
        )
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return rows
    return pd.DataFrame(rows)


def forecast_to_frame(origin: int, name: str, forecast: BaselineForecast, target: Array) -> object:
    rows = [
        {
            "origin": origin,
            "model": name,
            "horizon_index": int(idx),
            "forecast": float(pred),
            "actual": float(actual),
        }
        for idx, (pred, actual) in enumerate(zip(forecast.values, target), start=1)
    ]
    try:
        import pandas as pd
    except ModuleNotFoundError:
        return rows
    return pd.DataFrame(rows)
