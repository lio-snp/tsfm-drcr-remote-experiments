"""Interfaces for optional TSFM adapters.

Concrete adapters should live in this module family but remain optional so the
protocol and baseline diagnostics can run on machines without GPU or model
weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


Array = np.ndarray


class PointForecaster(Protocol):
    name: str

    def forecast(self, context: Array, horizon: int) -> Array:
        """Return a point forecast with shape `(horizon,)`."""


class ProbabilisticForecaster(Protocol):
    name: str

    def sample(self, context: Array, horizon: int, num_samples: int) -> Array:
        """Return samples with shape `(num_samples, horizon)`."""


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    package: str
    model_id: str
    forecast_type: str
    status: str


REQUIRED_ADAPTERS = [
    AdapterSpec("chronos_bolt", "chronos-forecasting", "amazon/chronos-bolt-small", "probabilistic", "planned"),
    AdapterSpec("timesfm", "timesfm", "google/timesfm-2.0-500m-pytorch", "point_or_quantile", "planned"),
    AdapterSpec("moirai", "uni2ts", "Salesforce/moirai-1.1-R-small", "probabilistic", "planned"),
]
