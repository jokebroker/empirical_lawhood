"""Pure repeated-command feed projection shared by bounded reactor consumers."""

from dataclasses import replace
from decimal import Decimal as D

import numpy as np

from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import ReactorExposureWordMap
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
from empirical_lawhood.kernel.action_contracts import ActionOccurrenceGroup, OccurrenceActionWord, ActionWordMode


def feed_action_word(
    maps: tuple[ReactorExposureWordMap, ...],
    *,
    word_id: str,
    history_id: str,
    horizon_id: str,
    chart_id: str,
    receiver_id: str,
) -> OccurrenceActionWord:
    if not 1 <= len(maps) <= 24:
        raise ValueError("finite native word exceeds its bounded 24-command horizon")
    occurrences = tuple(
        replace(o, channel=replace(o.channel, support_contract_id=chart_id))
        for m in maps
        for o in m.word.occurrences
        if o.channel.controller_quantity_id == "reactor-feed"
    )
    keys = {o.occurrence_id for o in occurrences}
    groups = tuple(
        ActionOccurrenceGroup(g.group_id, tuple(x for x in g.members if x.occurrence_id in keys))
        for m in maps
        for g in m.word.groups
    )
    first = maps[0].word
    return OccurrenceActionWord(
        word_id,
        ActionWordMode.SEQUENTIAL,
        occurrences,
        groups,
        first.ordering_clock_id,
        first.ordering_time_unit,
        first.ordering_coordinate_frame,
        first.ordering_origin,
        chart_id,
        history_id,
        receiver_id,
        horizon_id,
        tuple(f"{word_id}.prefix-{i:02d}" for i in range(len(occurrences) + 1)),
        first.support_status,
        (),
    )


def project_feed_commands(
    observation: Observation,
    previous: tuple[float, float],
    rates: tuple[D, ...],
    *,
    decision_id: str,
    history_id: str,
    horizon_id: str,
) -> tuple[ReactorExposureWordMap, ...]:
    """Propagate only the known actuator and dose; no temperature prediction."""
    if (
        not 1 <= len(rates) <= 24
        or any(type(r) is not D or not r.is_finite() or not D(0) <= r <= D(".09") for r in rates)
        or not np.isfinite((*previous, observation.time, observation.dose)).all()
        or observation.time % 10
        or not 600 <= observation.time <= 25200 - 10 * len(rates)
    ):
        raise ValueError("finite feed request exceeds its native window or chart")
    actuator = Actuator()
    mappings = []
    current, applied = observation, previous
    for index, rate in enumerate(rates):
        projection = actuator.project_request(current, applied, (float(rate), previous[1]), index)
        mappings.append(
            ReactorExposureWordMap.from_projection(
                projection,
                decision_id=f"{decision_id}.command-{index:02d}",
                history_id=history_id,
                horizon_id=horizon_id,
            )
        )
        applied = projection.applied
        current = replace(current, time=current.time + 10, dose=projection.next_dose)
    return tuple(mappings)


def overlay_feed_tape(
    donor: np.ndarray,
    callback: int,
    rates: tuple[D, ...],
    jacket: float,
) -> np.ndarray:
    if (
        donor.shape != (2880, 2)
        or not np.isfinite(donor).all()
        or type(callback) is not int
        or not 1 <= len(rates) <= 24
        or not 60 <= callback <= 2520 - len(rates)
        or not np.isfinite(jacket)
        or any(type(r) is not D or not r.is_finite() for r in rates)
    ):
        raise ValueError("finite word lacks a complete donor tape and native window")
    tape = donor.copy()
    for offset, rate in enumerate(rates):
        tape[callback + offset] = (float(rate), jacket)
    return tape


def measure_feed_delivery(
    anchor: np.ndarray,
    requested: np.ndarray,
    stages: np.ndarray,
    exposure: np.ndarray,
    rates: tuple[D, ...],
    dt: float,
) -> tuple[bool, float, float]:
    count = len(rates)
    if (
        dt not in (1.0, 0.5)
        or not 1 <= count <= 24
        or anchor.shape != (6,)
        or requested.shape != (count, 2)
        or stages.shape != (count, 4)
        or exposure.shape != (count, int(10 / dt), 4)
        or any(not np.isfinite(a).all() for a in (anchor, requested, stages, exposure))
    ):
        return False, 0.0, 0.0
    actuator = Actuator()
    previous = (float(anchor[4]), float(anchor[5]))
    dose, start, jacket = float(anchor[3]), float(anchor[0]), float(anchor[5])
    valid = True
    for index, rate in enumerate(rates):
        observation = Observation(start + 10 * index, float(anchor[1]), float(anchor[2]), dose)
        request = (float(rate), jacket)
        projected = actuator.project_request(observation, previous, request, index, dt)
        valid &= (
            np.array_equal(requested[index], request)
            and np.array_equal(stages[index], (*projected.accepted, *projected.applied))
            and np.array_equal(exposure[index], projected.exposure)
        )
        previous, dose = projected.applied, projected.next_dose
    return (
        bool(valid),
        float(10 * stages[:, 2].sum()),
        float(np.sum(exposure[:, :, 1] * exposure[:, :, 2])),
    )
