"""Outcome-blind native pulse projection, including all shutdown commands."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.config import GUARD, Pulse
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import ReactorExposureWordMap
from empirical_lawhood.adapters.simulators.reactor_prefix_response.finite_words import feed_action_word, project_feed_commands, overlay_feed_tape
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.serialization import CanonicalRecord

CHART = "reactor-finite-control-frontier-finite-pulse-chart"
RECEIVER = "reactor-frontier-paired-peak-reduction"


def scalar_word(
    maps: tuple[ReactorExposureWordMap, ...], *, word_id: str, history_id: str, horizon_id: str
) -> OccurrenceActionWord:
    if len(maps) != GUARD // 10:
        raise ValueError("frontier delivery word must include every guard-window command")
    return feed_action_word(
        maps,
        word_id=word_id,
        history_id=history_id,
        horizon_id=horizon_id,
        chart_id=CHART,
        receiver_id=RECEIVER,
    )


@dataclass(frozen=True, slots=True)
class PulseProjection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-finite-control-frontier/pulse-projection'
    pulse: Pulse
    mappings: tuple[ReactorExposureWordMap, ...]
    word: OccurrenceActionWord
    fixed_jacket_K: D

    def __post_init__(self) -> None:
        if self.word != scalar_word(
            self.mappings,
            word_id=self.word.word_id,
            history_id=self.word.retained_history_id,
            horizon_id=self.word.horizon_id,
        ):
            raise ValueError("pulse lost its exact scalar/native compatibility map")
        start = self.mappings[0].command.time_s
        for offset, m in enumerate(self.mappings):
            if (
                m.command.time_s != start + 10 * offset
                or m.command.feed_kg_s != self.pulse.request(10 * offset)
                or m.command.jacket_k != self.fixed_jacket_K
                or m.accepted[1] != self.fixed_jacket_K
                or m.applied[1] != self.fixed_jacket_K
                or any(row[3] != self.fixed_jacket_K for row in m.exposure)
            ):
                raise ValueError("pulse changed command order or fixed jacket")

    @property
    def applied_mass_kg(self) -> D:
        return sum((10 * m.applied[0] for m in self.mappings), D(0))

    @property
    def realized_mass_kg(self) -> D:
        return sum((dt * feed for m in self.mappings for _, dt, feed, _ in m.exposure), D(0))


def project_pulse(
    observation: Observation,
    previous: tuple[float, float],
    pulse: Pulse,
    *,
    decision_id: str,
    history_id: str,
    horizon_id: str,
) -> PulseProjection:
    """Propagate known dose and actuator only; never propagate temperature/state."""
    if (
        observation.time % 10
        or not 600 <= observation.time <= 25200 - GUARD
        or not 0 <= previous[0] <= 0.016
        or not np.isfinite((*previous, observation.time, observation.dose)).all()
    ):
        raise ValueError("pulse is outside the causal native support")
    maps = project_feed_commands(
        observation,
        previous,
        tuple(pulse.request(i) for i in range(0, GUARD, 10)),
        decision_id=decision_id,
        history_id=history_id,
        horizon_id=horizon_id,
    )
    word = scalar_word(
        maps, word_id=f"{decision_id}.word", history_id=history_id, horizon_id=horizon_id
    )
    return PulseProjection(pulse, maps, word, D(repr(float(previous[1]))))


def pulse_tape(
    donor_requests: np.ndarray, callback: int, pulse: Pulse, fixed_jacket_K: float
) -> np.ndarray:
    if (
        donor_requests.shape != (2880, 2)
        or not np.isfinite(donor_requests).all()
        or type(callback) is not int
        or not 60 <= callback <= (25200 - GUARD) // 10
        or not np.isfinite(fixed_jacket_K)
    ):
        raise ValueError("pulse lacks its complete native donor tape")
    return overlay_feed_tape(
        donor_requests,
        callback,
        tuple(pulse.request(i) for i in range(0, GUARD, 10)),
        fixed_jacket_K,
    )
