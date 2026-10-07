"""Pinned actuator-only extraction and privileged post-run label projection."""

from __future__ import annotations
from hashlib import sha256
import json
from types import SimpleNamespace
from typing import Any
import numpy as np
from empirical_lawhood.adapters.simulators.reactor_prefix_response.forecast import _native_actuator_projection
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import PARAMS_SHA256, PLANT_SHA256
from .actuator_statements import project_stages
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation, Projection, menu


class Actuator:
    """No plant import, state propagation or mechanistic response forecast."""

    def __init__(self, plant: bytes | None = None, params: bytes | None = None) -> None:
        if plant is None and params is None:
            self.project_stages = project_stages
            self.params = SimpleNamespace(
                dose_window_s=(600.0, 25200.0),
                dose_total_kg=287.3,
                sample_dt_s=10.0,
                feed_max_kg_s=0.032,
                feed_rate_kg_s2=0.002,
                tj_cmd_min_k=279.3,
                tj_cmd_max_k=347.6,
                tj_cmd_rate_k_s=0.23,
            )
            return
        if plant is None or params is None:
            raise ValueError("both pinned sources required")
        if sha256(plant).hexdigest() != PLANT_SHA256 or sha256(params).hexdigest() != PARAMS_SHA256:
            raise ValueError("actuator source pin differs")
        self.project_stages = _native_actuator_projection(plant, {"np": np})
        spec = json.loads(params)
        a, d, t = spec["actuators"], spec["dosing"], spec["timing"]
        self.params = SimpleNamespace(
            dose_window_s=d["window_s"],
            dose_total_kg=d["total_mass_kg"],
            sample_dt_s=t["sample_dt_s"],
            **a,
        )

    def project(
        self, observation: Observation, previous: tuple[float, float], dt: float = 1.0
    ) -> tuple[Projection, ...]:
        if dt not in (1.0, 0.5):
            raise ValueError("undeclared numerical view")
        return tuple(
            self.project_request(observation, previous, word, i, dt)
            for i, word in enumerate(menu(previous[1]))
        )

    def project_request(
        self,
        observation: Observation,
        previous: tuple[float, float],
        requested: tuple[float, float],
        action: int,
        dt: float = 1.0,
    ) -> Projection:
        if dt not in (1.0, 0.5) or not all(np.isfinite(requested)):
            raise ValueError("invalid native projection input")
        accepted_f, accepted_j, feed, jacket = self.project_stages(
            self.params, observation.time, observation.dose, *previous, *requested
        )
        dose = observation.dose
        profile = []
        for k in range(int(10 / dt)):
            realized = min(feed, max(287.3 - dose, 0.0) / dt)
            profile.append((observation.time + k * dt, dt, realized, jacket))
            dose += realized * dt
        return Projection(
            action, requested, (accepted_f, accepted_j), (feed, jacket), tuple(profile), dose
        )


def measured_labels(
    time: np.ndarray, temperature: np.ndarray, conversion: np.ndarray, dose: np.ndarray, dt: float
) -> np.ndarray:
    """ACCESS-01 post-run measurement boundary. Never call from online features."""
    expected = np.arange(int(28800 / dt) + 1, dtype=float) * dt
    if dt not in (1, 0.5) or not np.array_equal(time, expected):
        raise ValueError("complete native grid required, including last interval")
    if any(
        x.shape != time.shape or not np.isfinite(x).all() for x in (temperature, conversion, dose)
    ):
        raise ValueError("missing/nonfinite measurement labels")
    stride = int(10 / dt)
    return np.array(
        [
            (max(temperature[k : k + stride + 1]), conversion[k + stride], dose[k + stride])
            for k in range(0, len(time) - 1, stride)
        ],
        dtype=float,
    )


def bridge_advance(bridge: Any, command: Any) -> Any:
    """Preserve the existing bridge's final-delivery record without scalar reduction."""
    return bridge.advance(command)


def project_tape(requests: np.ndarray, dt: float) -> np.ndarray:
    """Known-input full tape projection; no response labels or native state."""
    return project_tape_stages(requests, dt)[1]


def project_tape_stages(requests: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Preserve accepted and applied stages separately from realized exposures."""
    if requests.shape != (2880, 2) or not np.isfinite(requests).all() or dt not in (1, 0.5):
        raise ValueError("complete finite request tape required")
    actuator = Actuator()
    feed, jacket, dose = 0.0, 316.0, 0.0
    result = np.empty((2880, int(10 / dt), 4))
    stages = np.empty((2880, 4))
    for k, (requested_feed, requested_jacket) in enumerate(requests):
        accepted_f, accepted_j, feed, jacket = actuator.project_stages(
            actuator.params, k * 10, dose, feed, jacket, requested_feed, requested_jacket
        )
        stages[k] = (accepted_f, accepted_j, feed, jacket)
        for step in range(int(10 / dt)):
            exposure = min(feed, max(287.3 - dose, 0.0) / dt)
            result[k, step] = (k * 10 + step * dt, dt, exposure, jacket)
            dose += exposure * dt
    return stages, result
