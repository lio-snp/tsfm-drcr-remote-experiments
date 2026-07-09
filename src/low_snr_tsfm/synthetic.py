"""Synthetic processes for controlled predictability sweeps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SyntheticSpec:
    """A reproducible synthetic-series configuration."""

    process: str
    length: int
    seed: int
    params: dict[str, float | int]


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def white_noise(length: int, sigma: float = 1.0, seed: int | None = None) -> Array:
    """Generate a zero-mean white-noise series."""

    return _rng(seed).normal(0.0, sigma, size=length)


def ar1(
    length: int,
    phi: float,
    sigma: float = 1.0,
    seed: int | None = None,
    burn_in: int = 200,
) -> Array:
    """Generate a stationary AR(1) process."""

    if abs(phi) >= 1:
        raise ValueError("AR(1) phi must be in (-1, 1) for the locked stationary sweep.")
    generator = _rng(seed)
    values = np.zeros(length + burn_in)
    noise = generator.normal(0.0, sigma, size=length + burn_in)
    for idx in range(1, len(values)):
        values[idx] = phi * values[idx - 1] + noise[idx]
    return values[burn_in:]


def seasonal_ar(
    length: int,
    phi: float,
    seasonal_amplitude: float,
    period: int,
    sigma: float = 1.0,
    seed: int | None = None,
) -> Array:
    """Generate AR(1) dynamics plus sinusoidal seasonality."""

    if period <= 1:
        raise ValueError("period must be greater than 1")
    base = ar1(length, phi=phi, sigma=sigma, seed=seed)
    t = np.arange(length)
    seasonal = seasonal_amplitude * np.sin(2 * np.pi * t / period)
    return base + seasonal


def local_level(
    length: int,
    level_sigma: float,
    obs_sigma: float,
    seed: int | None = None,
) -> Array:
    """Generate a local-level process observed with noise."""

    generator = _rng(seed)
    level_shocks = generator.normal(0.0, level_sigma, size=length)
    levels = np.cumsum(level_shocks)
    observations = levels + generator.normal(0.0, obs_sigma, size=length)
    return observations


def garch11(
    length: int,
    omega: float = 0.05,
    alpha: float = 0.08,
    beta: float = 0.90,
    seed: int | None = None,
    burn_in: int = 300,
) -> Array:
    """Generate zero-mean GARCH(1,1) returns."""

    if omega <= 0:
        raise ValueError("omega must be positive")
    if alpha < 0 or beta < 0 or alpha + beta >= 1:
        raise ValueError("Require alpha >= 0, beta >= 0, and alpha + beta < 1")
    generator = _rng(seed)
    total = length + burn_in
    values = np.zeros(total)
    variances = np.full(total, omega / (1.0 - alpha - beta))
    shocks = generator.normal(size=total)
    for idx in range(1, total):
        variances[idx] = omega + alpha * values[idx - 1] ** 2 + beta * variances[idx - 1]
        values[idx] = np.sqrt(max(variances[idx], 0.0)) * shocks[idx]
    return values[burn_in:]


def regime_switching_ar(
    length: int,
    phi_low: float,
    phi_high: float,
    switch_prob: float,
    sigma: float = 1.0,
    seed: int | None = None,
) -> tuple[Array, Array]:
    """Generate an AR process whose persistence switches between two regimes."""

    if not 0 <= switch_prob <= 1:
        raise ValueError("switch_prob must be in [0, 1]")
    if abs(phi_low) >= 1 or abs(phi_high) >= 1:
        raise ValueError("regime phis must be stationary")
    generator = _rng(seed)
    values = np.zeros(length)
    regimes = np.zeros(length, dtype=int)
    shocks = generator.normal(0.0, sigma, size=length)
    for idx in range(1, length):
        if generator.random() < switch_prob:
            regimes[idx] = 1 - regimes[idx - 1]
        else:
            regimes[idx] = regimes[idx - 1]
        phi = phi_high if regimes[idx] else phi_low
        values[idx] = phi * values[idx - 1] + shocks[idx]
    return values, regimes


def generate(spec: SyntheticSpec) -> Array:
    """Generate a synthetic series from a serializable spec."""

    params = dict(spec.params)
    registry: dict[str, Callable[..., Array | tuple[Array, Array]]] = {
        "white_noise": white_noise,
        "ar1": ar1,
        "seasonal_ar": seasonal_ar,
        "local_level": local_level,
        "garch11": garch11,
        "regime_switching_ar": regime_switching_ar,
    }
    if spec.process not in registry:
        raise KeyError(f"Unknown synthetic process: {spec.process}")
    generated = registry[spec.process](length=spec.length, seed=spec.seed, **params)
    if isinstance(generated, tuple):
        return generated[0]
    return generated


def default_snr_sweep(length: int = 2048, seed: int = 0) -> list[SyntheticSpec]:
    """Create a compact preregistration-friendly synthetic sweep."""

    specs: list[SyntheticSpec] = []
    for idx, phi in enumerate([0.0, 0.1, 0.3, 0.6, 0.9]):
        specs.append(SyntheticSpec("ar1", length, seed + idx, {"phi": phi, "sigma": 1.0}))
    for idx, amp in enumerate([0.0, 0.25, 0.5, 1.0, 2.0]):
        specs.append(
            SyntheticSpec(
                "seasonal_ar",
                length,
                seed + 100 + idx,
                {"phi": 0.2, "seasonal_amplitude": amp, "period": 24, "sigma": 1.0},
            )
        )
    for idx, ratio in enumerate([0.01, 0.05, 0.10, 0.25, 0.50]):
        specs.append(
            SyntheticSpec(
                "local_level",
                length,
                seed + 200 + idx,
                {"level_sigma": ratio, "obs_sigma": 1.0},
            )
        )
    specs.append(SyntheticSpec("white_noise", length, seed + 300, {"sigma": 1.0}))
    specs.append(
        SyntheticSpec(
            "garch11",
            length,
            seed + 400,
            {"omega": 0.05, "alpha": 0.08, "beta": 0.90},
        )
    )
    specs.append(
        SyntheticSpec(
            "regime_switching_ar",
            length,
            seed + 500,
            {"phi_low": 0.05, "phi_high": 0.8, "switch_prob": 0.03, "sigma": 1.0},
        )
    )
    return specs
