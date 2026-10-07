"""Causal post-preparation inputs for the fixed reactor information experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import causal_features


@dataclass(frozen=True)
class ProbeReadout:
    coefficient_K_per_kg: float
    standard_error_K_per_kg: float
    residual_rmse_K: float
    invalid: bool
    rank: int
    condition_number: float
    physical_times_s: np.ndarray
    design: np.ndarray
    observed_temperature_K: np.ndarray

    @property
    def coordinates(self) -> np.ndarray:
        return np.asarray(
            (
                self.coefficient_K_per_kg,
                self.standard_error_K_per_kg,
                self.residual_rmse_K,
                float(self.invalid),
            ),
            dtype=np.float64,
        )


def _check_prefix(
    observations: np.ndarray,
    stages: np.ndarray,
    exposure: np.ndarray,
    t0: int,
    callback: int,
) -> None:
    if (
        type(t0) is not int
        or type(callback) is not int
        or t0 < 0
        or t0 % 10
        or callback not in (t0 // 10 + 33, t0 // 10 + 93)
        or observations.shape[1:] != (4,)
        or stages.shape[1:] != (4,)
        or exposure.ndim != 3
        or exposure.shape[2] != 4
        or len(observations) <= callback
        or len(stages) < callback
        or len(exposure) < callback
        or not np.isfinite(observations[: callback + 1]).all()
        or not np.isfinite(stages[:callback]).all()
        or not np.isfinite(exposure[:callback]).all()
    ):
        raise ValueError("probe inputs violate the fixed causal prefix")
    if not np.array_equal(observations[: callback + 1, 0], np.arange(callback + 1) * 10):
        raise ValueError("probe observations have a shifted callback clock")
    # Dose exhaustion may reduce realized feed below the applied stage.
    if np.any(exposure[:callback, :, 2] > stages[:callback, None, 2] + 1e-12):
        raise ValueError("realized feed exceeds applied stage")


def probe_readout(
    observations: np.ndarray,
    stages: np.ndarray,
    exposure: np.ndarray,
    t0: int,
    callback: int,
) -> ProbeReadout:
    """OLS on the 31 delayed observations for physical t0 through t0+300.

    The sensor at callback ``t0+10`` describes physical ``t0``. The readout
    is first available at the fixed ``q=t0+330`` commitment, never at t0.
    """
    _check_prefix(observations, stages, exposure, t0, callback)
    start = t0 // 10
    if callback < start + 31:
        raise ValueError("the final delayed probe observation is unavailable")
    sensor = observations[start + 1 : start + 32]
    dt = exposure[start : start + 30, :, 1]
    realized = exposure[start : start + 30, :, 2]
    if not np.all(dt > 0) or not np.all(realized >= 0):
        raise ValueError("probe exposure has invalid duration or feed")
    increments = np.sum(dt * (realized - 0.016), axis=1)
    cumulative = np.r_[0.0, np.cumsum(increments)]
    elapsed = np.arange(31, dtype=np.float64) * 10
    design = np.column_stack(
        (
            np.ones(31),
            elapsed / 300,
            (elapsed / 300) ** 2,
            sensor[:, 2] / 40,
            cumulative,
        )
    )
    response = sensor[:, 1].copy()
    operator, _, rank, singular = np.linalg.lstsq(design, response, rcond=1e-12)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
    invalid = rank != 5 or not np.isfinite(condition) or condition > 1e8
    if invalid:
        coefficient = standard_error = rmse = 0.0
    else:
        residual = response - design @ operator
        rmse = float(np.sqrt(np.mean(residual**2)))
        sigma2 = float(np.dot(residual, residual) / (len(response) - design.shape[1]))
        covariance = np.linalg.inv(design.T @ design) * sigma2
        coefficient = float(operator[4])
        standard_error = float(np.sqrt(max(0.0, covariance[4, 4])))
    return ProbeReadout(
        coefficient,
        standard_error,
        rmse,
        bool(invalid),
        int(rank),
        condition,
        np.arange(t0, t0 + 301, 10, dtype=np.float64),
        design,
        response,
    )


def action_only_features(
    observations: np.ndarray,
    requests: np.ndarray,
    stages: np.ndarray,
    exposure: np.ndarray,
    t0: int,
    callback: int,
) -> np.ndarray:
    """Retain pre-t0 sensors, endpoint sensors and all known action summaries."""
    _check_prefix(observations, stages, exposure, t0, callback)
    if (
        requests.shape[1:] != (2,)
        or len(requests) < callback
        or not np.isfinite(requests[:callback]).all()
    ):
        raise ValueError("requested action tape is incomplete")
    start = t0 // 10
    before = causal_features(observations, stages, start)
    current = causal_features(observations, stages, callback)[:6]
    active = slice(start, callback)
    actual_mass = exposure[active, :, 1] * exposure[active, :, 2]
    duration = (callback - start) * 10
    whole_mass = float(np.sum(actual_mass))
    recent = stages[max(start, callback - 6) : callback, 2]
    window = stages[max(start, callback - 30) : callback, 2]
    endpoint = causal_features(observations, stages, callback)
    action = np.asarray(
        (
            duration / 300,
            whole_mass / 287.3,
            float(np.mean(recent)) / 0.032,
            float(np.mean(window)) / 0.032,
            endpoint[14],
            endpoint[15],
            float(np.sum(requests[active, 0]) * 10) / 287.3,
            float(np.sum(stages[active, 0]) * 10) / 287.3,
            float(np.sum(stages[active, 2]) * 10) / 287.3,
            float(np.mean(requests[active, 1]) - 316) / 40,
            float(np.mean(stages[active, 1]) - 316) / 40,
            float(np.mean(stages[active, 3]) - 316) / 40,
        ),
        dtype=np.float64,
    )
    # Full custody retains the complete requested/accepted/applied/realized
    # tape; the fixed model encoder also exposes its declared action summaries.
    return np.asarray(np.round(np.r_[before, current, action], 9), dtype=np.float64)


def full_readout_features(
    observations: np.ndarray,
    requests: np.ndarray,
    stages: np.ndarray,
    exposure: np.ndarray,
    t0: int,
    callback: int,
    stored_readout: ProbeReadout | None = None,
) -> np.ndarray:
    """Action-only inputs plus post-t0 sensor history and the fixed OLS readout."""
    common = action_only_features(observations, requests, stages, exposure, t0, callback)
    if callback == t0 // 10 + 93 and stored_readout is None:
        raise ValueError("persistence prediction requires the readout sealed at q")
    readout = stored_readout or probe_readout(observations, stages, exposure, t0, callback)
    # At q+600 the caller supplies the readout sealed at q; it is never refit.
    sensor_history = causal_features(observations, stages, callback)[6:]
    return np.asarray(
        np.round(np.r_[common, sensor_history, readout.coordinates], 9), dtype=np.float64
    )
