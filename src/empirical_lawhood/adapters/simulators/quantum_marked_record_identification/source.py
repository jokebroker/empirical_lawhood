"""Event-driven passive monitored-chain source for quantum marked record identification.

The conditional checkpoint is a source-restart operand only.  Feature and
compiler modules accept event records, never this checkpoint type.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
from scipy.sparse import csr_matrix

from .basis import ComplexVector, FixedNumberBasis, Propagator, SparseKrylovPropagator, apply_jump, normalize_state, particle_number_error, site_probabilities
from .contracts import PLAN_ID, PreparationUnit


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_index: int
    event_time: float
    site: int

    def __post_init__(self) -> None:
        if self.event_index < 0 or not math.isfinite(self.event_time) or self.event_time < 0:
            raise ValueError("event identity or time is invalid")
        if not 0 <= self.site < 12:
            raise ValueError("event site leaves the fixed chain")


@dataclass(frozen=True, slots=True)
class IntervalResult:
    state: ComplexVector
    events: tuple[EventRecord, ...]
    total_event_count: int
    maximum_norm_error: float
    maximum_particle_number_error: float


@dataclass(frozen=True, slots=True)
class PrefixCheckpoint:
    scientific_seed: int
    unit_id: str
    roster_id: str
    denominator_id: str
    k_left: int
    initial_state: int
    cutoff_time: float
    state: ComplexVector
    history_events: tuple[EventRecord, ...]
    total_event_count: int
    stream_id: str
    maximum_norm_error: float
    maximum_particle_number_error: float

    @property
    def state_sha256(self) -> str:
        return sha256(np.asarray(self.state, dtype="<c16").tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class PassiveFuture:
    scientific_seed: int
    unit_id: str
    start_time: float
    end_time: float
    events: tuple[EventRecord, ...]
    total_event_count: int
    stream_id: str
    maximum_norm_error: float
    maximum_particle_number_error: float


def _rng(seed: int) -> np.random.Generator:
    if type(seed) is not int or not 0 <= seed < 2**128:
        raise ValueError("source scientific seed must be an explicit unsigned integer of the declared width")
    return np.random.Generator(np.random.Philox(seed))


def simulate_interval(
    *,
    basis: FixedNumberBasis,
    propagator: Propagator,
    initial_state: ComplexVector,
    start_time: float,
    duration: float,
    gamma: float,
    rng: np.random.Generator,
    retention_start: float | None = None,
) -> IntervalResult:
    """Simulate a right-open interval with exact constant total hazard."""

    if not math.isfinite(start_time) or start_time < 0:
        raise ValueError("interval start is invalid")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("interval duration is invalid")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("monitoring strength is invalid")
    end_time = start_time + duration
    retained_from = start_time if retention_start is None else retention_start
    if not start_time <= retained_from <= end_time:
        raise ValueError("retention start leaves the simulated interval")
    state, initial_norm_error = normalize_state(initial_state.copy())
    maximum_norm_error = initial_norm_error
    maximum_particle_error = particle_number_error(basis, state)
    time_value = start_time
    events: list[EventRecord] = []
    event_count = 0
    hazard = gamma * basis.particles
    while True:
        waiting = float(rng.exponential(1.0 / hazard))
        event_time = time_value + waiting
        if event_time >= end_time:
            state = propagator.propagate(state, end_time - time_value)
            state, norm_error = normalize_state(state)
            maximum_norm_error = max(maximum_norm_error, norm_error)
            maximum_particle_error = max(
                maximum_particle_error,
                particle_number_error(basis, state),
            )
            break
        state = propagator.propagate(state, waiting)
        state, norm_error = normalize_state(state)
        probabilities = site_probabilities(basis, state)
        site = int(rng.choice(basis.l_sites, p=probabilities))
        state, jump_norm_error = apply_jump(basis, state, site)
        maximum_norm_error = max(maximum_norm_error, norm_error, jump_norm_error)
        maximum_particle_error = max(
            maximum_particle_error,
            particle_number_error(basis, state),
        )
        if event_time >= retained_from:
            events.append(EventRecord(event_count, event_time, site))
        event_count += 1
        time_value = event_time
    if any(
        right.event_time <= left.event_time for left, right in zip(events, events[1:], strict=False)
    ):
        raise AssertionError("event times are not strictly increasing")
    if any(not retained_from <= event.event_time < end_time for event in events):
        raise AssertionError("event leaves the right-open retained window")
    return IntervalResult(
        state=state,
        events=tuple(events),
        total_event_count=event_count,
        maximum_norm_error=maximum_norm_error,
        maximum_particle_number_error=maximum_particle_error,
    )


def prefix_stream_id(unit: PreparationUnit, denominator_id: str) -> str:
    digest = sha256(f"{PLAN_ID}:{unit.unit_id}:{denominator_id}:prefix".encode()).hexdigest()[:24]
    return f"stream.quantum-marked-record-identification.prefix.{digest}"


def future_stream_id(prefix: PrefixCheckpoint) -> str:
    digest = sha256(
        f"{PLAN_ID}:{prefix.roster_id}:{prefix.unit_id}:hold:future".encode()
    ).hexdigest()[:24]
    return f"stream.quantum-marked-record-identification.future.{digest}"


def prepare_prefix(
    *,
    scientific_seed: int,
    unit: PreparationUnit,
    denominator_id: str,
    basis: FixedNumberBasis,
    hamiltonian: csr_matrix,
    gamma: float,
    cutoff: float = 200.0,
    retained_start: float = 176.0,
) -> PrefixCheckpoint:
    if not 0 <= retained_start < cutoff:
        raise ValueError("retained prefix interval is invalid")
    stream_id = prefix_stream_id(unit, denominator_id)
    rng = _rng(scientific_seed)
    result = simulate_interval(
        basis=basis,
        propagator=SparseKrylovPropagator(hamiltonian),
        initial_state=basis.state_vector(unit.initial_state),
        start_time=0.0,
        duration=cutoff,
        gamma=gamma,
        rng=rng,
        retention_start=retained_start,
    )
    return PrefixCheckpoint(
        scientific_seed=scientific_seed,
        unit_id=unit.unit_id,
        roster_id=unit.roster_id,
        denominator_id=denominator_id,
        k_left=unit.k_left,
        initial_state=unit.initial_state,
        cutoff_time=cutoff,
        state=result.state,
        history_events=result.events,
        total_event_count=result.total_event_count,
        stream_id=stream_id,
        maximum_norm_error=result.maximum_norm_error,
        maximum_particle_number_error=result.maximum_particle_number_error,
    )


def continue_passive(
    *,
    scientific_seed: int,
    prefix: PrefixCheckpoint,
    basis: FixedNumberBasis,
    hamiltonian: csr_matrix,
    gamma: float,
    horizon: float = 4.0,
) -> PassiveFuture:
    if prefix.cutoff_time != 200.0:
        raise ValueError("passive continuation requires the frozen cutoff")
    stream_id = future_stream_id(prefix)
    result = simulate_interval(
        basis=basis,
        propagator=SparseKrylovPropagator(hamiltonian),
        initial_state=prefix.state,
        start_time=prefix.cutoff_time,
        duration=horizon,
        gamma=gamma,
        rng=_rng(scientific_seed),
        retention_start=prefix.cutoff_time,
    )
    return PassiveFuture(
        scientific_seed=scientific_seed,
        unit_id=prefix.unit_id,
        start_time=prefix.cutoff_time,
        end_time=prefix.cutoff_time + horizon,
        events=result.events,
        total_event_count=result.total_event_count,
        stream_id=stream_id,
        maximum_norm_error=max(
            prefix.maximum_norm_error,
            result.maximum_norm_error,
        ),
        maximum_particle_number_error=max(
            prefix.maximum_particle_number_error,
            result.maximum_particle_number_error,
        ),
    )


__all__ = [
    "EventRecord",
    "IntervalResult",
    "PassiveFuture",
    "PrefixCheckpoint",
    "continue_passive",
    "future_stream_id",
    "prepare_prefix",
    "prefix_stream_id",
    "simulate_interval",
]
