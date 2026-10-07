"""Exact +400 preparation-policy preparation chart and handoff instrument."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame, _observation_window
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ComplexArray

from .instruments import REFERENCE_INSTRUMENT, reference_sketch
from .preparation_policy_contracts import FiniteResponseLawPreparationSchedule

Array = npt.NDArray[np.float64]
BASELINE = np.asarray((2 / 3, 22 / 3), dtype=np.float64)


def preparation_policy_preparation_schedule(
    schedule: FiniteResponseLawPreparationSchedule, *, refinement: int
) -> Array:
    """Return the declared 400-tick Y-only preparation in native units."""
    if type(refinement) is not int or refinement not in (1, 2):
        raise ValueError("preparation-policy preparation requires one of the two declared views")
    result = np.tile(BASELINE, (400 * refinement, 1))
    if schedule.schedule_id == "hold":
        return np.frombuffer(result.tobytes(), dtype=np.float64).reshape(result.shape)
    ramp_ticks, dwell_ticks = {
        128: (48, 32),
        256: (96, 64),
    }[schedule.duration_ticks]
    start = schedule.pulse_start_tick * refinement
    ramp = ramp_ticks * refinement
    dwell = dwell_ticks * refinement
    sign = schedule.sign * float(schedule.excursion)
    result[start : start + ramp, 1] += sign * np.arange(1, ramp + 1) / ramp
    result[start + ramp : start + ramp + dwell, 1] += sign
    result[start + ramp + dwell : start + 2 * ramp + dwell, 1] += sign * (
        1 - np.arange(1, ramp + 1) / ramp
    )
    return np.frombuffer(result.tobytes(), dtype=np.float64).reshape(result.shape)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyCompactInterface:
    """Same frozen 24-channel instrument at the distinct +400 cutoff."""

    frame_cutoff_tick: int
    cutoff_tick: int
    backward_tick: int
    values: Array

    def __post_init__(self) -> None:
        if (
            self.frame_cutoff_tick != 4096
            or self.cutoff_tick != 4496
            or self.backward_tick != 4464
            or self.values.shape != (24,)
            or self.values.dtype != np.dtype("float64")
            or not np.isfinite(self.values).all()
        ):
            raise ValueError("preparation-policy handoff interface changes its cutoff or recipe")
        object.__setattr__(self, "values", np.frombuffer(self.values.tobytes(), dtype=np.float64))


def preparation_policy_compact_interface(
    *,
    frame: PreparedPortFrame,
    ticks: tuple[int, ...],
    positions: ComplexArray,
    momenta: ComplexArray,
) -> FiniteResponseLawPreparationPolicyCompactInterface:
    if frame.cutoff_tick != 4096 or ticks != tuple(range(4016, 4497, 16)):
        raise ValueError("preparation-policy instrument requires its exact retained-prefix/+400 window")
    history = _observation_window(frame, ticks, positions, momenta)
    backward_index = ticks.index(4464)
    current = reference_sketch(positions[-1], frame)
    previous = reference_sketch(positions[backward_index], frame)
    elapsed = float(REFERENCE_INSTRUMENT.reference_tick_duration * 32)
    values = np.r_[
        history[-1, list(REFERENCE_INSTRUMENT.snapshot_channels)],
        current,
        (current - previous) / elapsed,
    ]
    return FiniteResponseLawPreparationPolicyCompactInterface(4096, 4496, 4464, values)
