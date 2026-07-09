"""Export helpers for raw forecast reruns."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable

import numpy as np


def _clean_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def serialize_values(values: Iterable[object]) -> str:
    """Serialize numeric values for compact CSV sidecars."""

    return json.dumps([_clean_float(value) for value in np.asarray(list(values), dtype=object).reshape(-1)])


def history_context_sidecar_row(
    *,
    run_id: str,
    dataset: str,
    series_id: str,
    model: str,
    baseline_family: str,
    baseline_mode: str,
    origin: int,
    window_index: int,
    context: np.ndarray,
    full_context: np.ndarray,
    baseline_context: np.ndarray,
    target: np.ndarray,
    baseline_season_length: int,
    baseline_context_cap: int,
    source_commit: str,
    model_id: str,
) -> dict[str, object]:
    """Return one auditable sidecar row per forecast origin.

    Native classical prediction intervals need the exact history supplied to the
    classical model.  Keeping this as a sidecar avoids duplicating long context
    arrays for every horizon step in the raw forecast CSV.
    """

    return {
        "run_id": run_id,
        "dataset": dataset,
        "series_id": series_id,
        "model": model,
        "baseline_family": baseline_family,
        "baseline_mode": baseline_mode,
        "origin": int(origin),
        "window_index": int(window_index),
        "context_length": int(np.asarray(context).size),
        "full_context_length": int(np.asarray(full_context).size),
        "baseline_context_length": int(np.asarray(baseline_context).size),
        "baseline_context_cap": int(baseline_context_cap),
        "baseline_season_length": int(baseline_season_length),
        "horizon": int(np.asarray(target).size),
        "context_values": serialize_values(context),
        "full_context_values": serialize_values(full_context),
        "baseline_context_values": serialize_values(baseline_context),
        "target_values": serialize_values(target),
        "source_commit": source_commit,
        "model_id": model_id,
    }
