"""Small GIFT-Eval split helpers used by local raw-forecast reruns."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np


TEST_SPLIT = 0.1
MAX_WINDOW = 20

M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

TERM_MULTIPLIER = {
    "short": 1,
    "medium": 10,
    "long": 15,
}


@dataclass(frozen=True)
class GiftEvalWindow:
    window_index: int
    origin: int
    context: np.ndarray
    target: np.ndarray


def canonical_freq_unit(freq: str) -> str:
    """Return the GIFT-Eval frequency unit key used by prediction maps."""

    freq_str = str(freq).strip()
    unit = re.sub(r"^[0-9]+", "", freq_str)
    unit = {
        "min": "T",
        "h": "H",
        "s": "S",
        "YE": "A",
        "Y": "A",
        "QE": "Q",
        "ME": "M",
    }.get(unit, unit)
    unit = unit.upper()
    if unit == "MIN":
        return "T"
    return unit


def prediction_length(dataset_name: str, freq: str, term: str) -> int:
    unit = canonical_freq_unit(freq)
    multiplier = TERM_MULTIPLIER[term]
    base = M4_PRED_LENGTH_MAP[unit] if "m4" in dataset_name else PRED_LENGTH_MAP[unit]
    return multiplier * base


def window_count(min_series_length: int, pred_length: int) -> int:
    windows = math.ceil(TEST_SPLIT * int(min_series_length) / int(pred_length))
    return min(max(1, windows), MAX_WINDOW)


def test_windows(series: np.ndarray, pred_length: int, windows: int) -> list[GiftEvalWindow]:
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("GIFT-Eval raw rerun windows require a univariate series")
    if values.size < pred_length * windows + 1:
        raise ValueError("Series is too short for requested GIFT-Eval windows")

    test_start = values.size - pred_length * windows
    generated = []
    for idx in range(windows):
        origin = test_start + idx * pred_length
        generated.append(
            GiftEvalWindow(
                window_index=idx,
                origin=origin,
                context=values[:origin],
                target=values[origin : origin + pred_length],
            )
        )
    return generated


def iter_univariate_targets(target: np.ndarray, item_id: str) -> Iterable[tuple[str, np.ndarray]]:
    values = np.asarray(target, dtype=float)
    if values.ndim == 1:
        yield item_id, values
        return
    if values.ndim != 2:
        raise ValueError(f"Unsupported target shape: {values.shape}")
    for dim in range(values.shape[0]):
        yield f"{item_id}_dim{dim}", values[dim]


def forward_fill_nan(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    if not finite.any():
        return np.zeros_like(arr)
    last = float(arr[finite][0])
    for idx, value in enumerate(arr):
        if np.isfinite(value):
            last = float(value)
        else:
            arr[idx] = last
    return arr


def finite_pair_mask(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    y = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    if y.shape != f.shape:
        raise ValueError(f"Shape mismatch: {y.shape} != {f.shape}")
    return np.isfinite(y) & np.isfinite(f)
