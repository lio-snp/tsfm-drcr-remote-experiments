"""Traffic reproduction helpers for regime-stratified TSFM audits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class TrafficRegimeConfig:
    low_speed: float = 30.0
    high_speed: float = 55.0
    transition_range: float = 15.0


def _first_h5_dataset(handle: object) -> Array:
    import h5py

    found = []

    def visitor(_name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset):
            found.append(np.asarray(obj))

    handle.visititems(visitor)
    if not found:
        raise ValueError("HDF5 file does not contain a dataset")
    return found[0]


def _as_time_sensor_matrix(values: Array) -> Array:
    matrix = np.asarray(values, dtype=float)
    matrix = np.squeeze(matrix)
    if matrix.ndim == 3:
        matrix = matrix[:, :, 0]
    if matrix.ndim != 2:
        raise ValueError(f"Expected a time x sensor matrix, got shape {matrix.shape}")
    if matrix.shape[0] < matrix.shape[1]:
        matrix = matrix.T
    return matrix.astype(float)


def load_traffic_matrix(path: str | Path) -> Array:
    """Load traffic speed data from .npz, .npy, .csv, or .h5/.hdf5."""

    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix == ".npz":
        archive = np.load(data_path)
        preferred = [key for key in ["data", "x", "arr_0"] if key in archive.files]
        key = preferred[0] if preferred else archive.files[0]
        return _as_time_sensor_matrix(archive[key])
    if suffix == ".npy":
        return _as_time_sensor_matrix(np.load(data_path))
    if suffix == ".csv":
        try:
            import pandas as pd

            return _as_time_sensor_matrix(pd.read_csv(data_path).to_numpy())
        except Exception:
            return _as_time_sensor_matrix(np.genfromtxt(data_path, delimiter=","))
    if suffix in {".h5", ".hdf5"}:
        try:
            import pandas as pd

            return _as_time_sensor_matrix(pd.read_hdf(data_path).to_numpy())
        except Exception:
            import h5py

            with h5py.File(data_path, "r") as handle:
                return _as_time_sensor_matrix(_first_h5_dataset(handle))
    raise ValueError(f"Unsupported traffic file type: {data_path}")


def label_traffic_regime(target: Array, config: TrafficRegimeConfig | None = None) -> str:
    """Classify a target horizon as free-flow, congested, or transition."""

    cfg = config or TrafficRegimeConfig()
    values = np.asarray(target, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "unknown"
    low = float(np.min(values))
    high = float(np.max(values))
    mean = float(np.mean(values))
    if high - low >= cfg.transition_range or (low <= cfg.low_speed and high >= cfg.high_speed):
        return "transition"
    if mean >= cfg.high_speed:
        return "free_flow"
    if mean <= cfg.low_speed:
        return "congested"
    return "transition"


def historical_conditional_samples(
    context: Array,
    horizon: int,
    period: int = 288,
    max_samples: int = 64,
) -> Array:
    """Return same-time-of-day historical samples with shape sample x horizon."""

    values = np.asarray(context, dtype=float)
    if values.ndim != 1:
        raise ValueError("context must be univariate")
    if horizon <= 0 or period <= 0:
        raise ValueError("horizon and period must be positive")
    if values.size == 0:
        raise ValueError("context must not be empty")

    columns = []
    for step in range(horizon):
        slot = (values.size + step) % period
        candidates = values[np.arange(values.size) % period == slot]
        candidates = candidates[np.isfinite(candidates)]
        columns.append(candidates[-max_samples:])
    common = min((col.size for col in columns), default=0)
    if common == 0:
        fallback = np.resize(values[np.isfinite(values)][-1:], horizon)
        return fallback.reshape(1, horizon)
    aligned = [col[-common:] for col in columns]
    return np.column_stack(aligned)


def historical_conditional_forecast(
    context: Array,
    horizon: int,
    period: int = 288,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[str, Array]:
    """Summarize historical conditional samples into mean and quantiles."""

    samples = historical_conditional_samples(context, horizon=horizon, period=period)
    result: dict[str, Array] = {
        "samples": samples,
        "mean": np.mean(samples, axis=0),
    }
    for quantile in quantiles:
        result[f"q{int(round(quantile * 100)):02d}"] = np.quantile(samples, quantile, axis=0)
    return result
