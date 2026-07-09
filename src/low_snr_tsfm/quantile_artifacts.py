"""Helpers for persisting richer quantile forecast artifacts."""

from __future__ import annotations

import math
import re

import numpy as np

_FORECAST_Q_RE = re.compile(r"^forecast_q(?P<percent>[0-9]+(?:p[0-9]+)?)$")


def clean_quantile_levels(levels: list[float] | np.ndarray) -> list[float]:
    parsed = [float(value) for value in np.asarray(levels, dtype=float).reshape(-1)]
    if not parsed:
        raise ValueError("At least one quantile level is required")
    for level in parsed:
        if not 0.0 <= level <= 1.0:
            raise ValueError(f"Quantile level out of [0, 1]: {level}")
    return parsed


def quantile_column_name(level: float) -> str:
    percent = 100.0 * float(level)
    rounded = round(percent)
    if math.isclose(percent, rounded, abs_tol=1e-9):
        return f"forecast_q{int(rounded):02d}"
    suffix = f"{percent:.4f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"forecast_q{suffix}"


def quantile_level_from_column_name(column: str) -> float:
    match = _FORECAST_Q_RE.match(column)
    if match is None:
        raise ValueError(f"Not a forecast quantile column: {column}")
    level = float(match.group("percent").replace("p", ".")) / 100.0
    clean_quantile_levels([level])
    return level


def forecast_quantile_columns(columns: list[str] | tuple[str, ...] | set[str]) -> list[tuple[float, str]]:
    pairs: list[tuple[float, str]] = []
    for column in columns:
        try:
            level = quantile_level_from_column_name(str(column))
        except ValueError:
            continue
        pairs.append((level, str(column)))
    return sorted(pairs, key=lambda item: (item[0], item[1]))


def quantile_matrix_from_rows(rows: list[dict[str, object]]) -> tuple[list[float], np.ndarray]:
    if not rows:
        raise ValueError("At least one row is required")
    columns = forecast_quantile_columns({key for row in rows for key in row})
    levels: list[float] = []
    values_by_level: list[list[float]] = []
    for level, column in columns:
        values: list[float] = []
        usable = True
        for row in rows:
            try:
                value = float(row.get(column, "nan"))
            except (TypeError, ValueError):
                usable = False
                break
            if not math.isfinite(value):
                usable = False
                break
            values.append(value)
        if usable:
            levels.append(level)
            values_by_level.append(values)
    if not levels:
        raise ValueError("No complete forecast_q* quantile columns found")
    return levels, np.asarray(values_by_level, dtype=float).T


def nearest_quantile_index(levels: list[float], target: float) -> int:
    clean = clean_quantile_levels(levels)
    return min(range(len(clean)), key=lambda idx: abs(clean[idx] - target))


def orient_quantile_matrix(values: np.ndarray, levels: list[float]) -> np.ndarray:
    """Return a horizon x quantile matrix for the given quantile levels."""

    q = np.asarray(values, dtype=float)
    clean = clean_quantile_levels(levels)
    if q.ndim != 2:
        raise ValueError(f"Expected a 2D quantile matrix, got {q.shape}")
    if q.shape[1] == len(clean):
        return q
    if q.shape[0] == len(clean):
        return q.T
    raise ValueError(f"Quantile level count {len(clean)} does not match forecast shape {q.shape}")


def quantile_triplet_from_matrix(values: np.ndarray, levels: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = orient_quantile_matrix(values, levels)
    return (
        q[:, nearest_quantile_index(levels, 0.1)],
        q[:, nearest_quantile_index(levels, 0.5)],
        q[:, nearest_quantile_index(levels, 0.9)],
    )


def quantile_row_values(values: np.ndarray, levels: list[float], horizon_index: int) -> dict[str, float]:
    q = orient_quantile_matrix(values, levels)
    idx = int(horizon_index)
    if idx < 0 or idx >= q.shape[0]:
        raise IndexError(f"horizon_index {idx} out of range for quantile matrix with horizon {q.shape[0]}")
    return {quantile_column_name(level): float(q[idx, level_idx]) for level_idx, level in enumerate(levels)}
