"""Event-driven monitored-chain source with direct quantum response control transport."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .basis import ComplexVector, FixedNumberBasis, Propagator, SparseKrylovPropagator, apply_jump, half_count_diagonal, normalize_state, particle_number_error, site_probabilities
from .contracts import Action, PLAN_ID, PreparationUnit
from .transport import TransportSummary, finalize_transport, integrate_current_segment_refined, q_a


@dataclass(frozen=True, slots=True)
class EventRecord:
    event_index: int
    event_time: float
    site: int

    def __post_init__(self) -> None:
        if self.event_index < 0 or not math.isfinite(self.event_time) or self.event_time < 0:
            raise ValueError("event identity or clock is invalid")
        if self.site < 0:
            raise ValueError("event site is invalid")


@dataclass(frozen=True, slots=True)
class IntervalResult:
    state: ComplexVector
    events: tuple[EventRecord, ...]
    total_event_count: int
    start_time: float
    end_time: float
    maximum_norm_error: float
    maximum_particle_number_error: float
    transport: TransportSummary | None


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
    preparation_rng_state: Mapping[str, object]
    stream_id: str
    maximum_norm_error: float
    maximum_particle_number_error: float

    @property
    def state_sha256(self) -> str:
        return sha256(np.asarray(self.state, dtype="<c16").tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class ActionLedger:
    requested_action: Action
    accepted_action: Action
    applied_action: Action
    requested_start: float
    requested_end: float
    realized_start: float
    realized_end: float
    hamiltonian_sha256: str
    delivery_discrepancy: float


@dataclass(frozen=True, slots=True)
class FutureDraw:
    scientific_seed: int
    unit_id: str
    action: Action
    draw_index: int
    stream_id: str
    events: tuple[EventRecord, ...]
    terminal_state: ComplexVector | None
    ledger: ActionLedger
    work_on: float
    work_off: float
    work_abs: float
    effort: float
    terminal_q_a: float
    terminal_energy: float
    transport: TransportSummary
    maximum_norm_error: float
    maximum_particle_number_error: float


def _rng(seed: int) -> np.random.Generator:
    if type(seed) is not int or not 0 <= seed < 2**128:
        raise ValueError("source scientific seed must be an explicit unsigned integer of the declared width")
    return np.random.Generator(np.random.Philox(seed))


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def rng_state(rng: np.random.Generator) -> Mapping[str, object]:
    state = _json_safe(rng.bit_generator.state)
    if not isinstance(state, Mapping):
        raise AssertionError("PRNG state must be an object")
    json.dumps(state, sort_keys=True, allow_nan=False)
    return state


def restore_rng(state: Mapping[str, object]) -> np.random.Generator:
    generator = _rng(0)
    generator.bit_generator.state = dict(state)
    return generator


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
    current_operator: csr_matrix | None = None,
    quadrature_order: int = 8,
    quadrature_sentinel_order: int = 16,
    continuity_tolerance: float = 1e-8,
) -> IntervalResult:
    """Simulate one right-open interval under the fixed total hazard."""

    if not math.isfinite(start_time) or start_time < 0:
        raise ValueError("interval start is invalid")
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("interval duration must be positive and finite")
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("monitoring strength must be positive and finite")
    end_time = start_time + duration
    if retention_start is None:
        retention_start = start_time
    if not start_time <= retention_start <= end_time:
        raise ValueError("retention cutoff leaves the simulated interval")
    state, initial_norm_error = normalize_state(initial_state.copy())
    max_norm_error = initial_norm_error
    max_particle_error = particle_number_error(basis, state)
    time_value = start_time
    events: list[EventRecord] = []
    event_count = 0
    hazard = gamma * basis.particles
    half_count = half_count_diagonal(basis)
    q_start_value = q_a(state, half_count)
    coherent = 0.0
    coherent_abs = 0.0
    sentinel = 0.0
    sentinel_abs = 0.0
    innovation = 0.0

    def integrate_segment(segment_duration: float) -> None:
        nonlocal coherent, coherent_abs, sentinel, sentinel_abs
        if current_operator is None:
            return
        local_tolerance = continuity_tolerance * segment_duration / duration
        primary, refined = integrate_current_segment_refined(
            propagator=propagator,
            state=state,
            duration=segment_duration,
            current_operator=current_operator,
            production_order=quadrature_order,
            sentinel_order=quadrature_sentinel_order,
            tolerance=max(local_tolerance, np.finfo(np.float64).eps),
        )
        coherent += primary.signed_current
        coherent_abs += primary.absolute_current
        sentinel += refined.signed_current
        sentinel_abs += refined.absolute_current

    while True:
        waiting = float(rng.exponential(1.0 / hazard))
        event_time = time_value + waiting
        if event_time >= end_time:
            final_duration = end_time - time_value
            integrate_segment(final_duration)
            state = propagator.propagate(state, final_duration)
            state, norm_error = normalize_state(state)
            max_norm_error = max(max_norm_error, norm_error)
            max_particle_error = max(
                max_particle_error,
                particle_number_error(basis, state),
            )
            break
        integrate_segment(waiting)
        state = propagator.propagate(state, waiting)
        state, norm_error = normalize_state(state)
        max_norm_error = max(max_norm_error, norm_error)
        probabilities = site_probabilities(basis, state)
        site = int(rng.choice(basis.l_sites, p=probabilities))
        q_before = q_a(state, half_count)
        state, jump_norm_error = apply_jump(basis, state, site)
        q_after = q_a(state, half_count)
        innovation += q_after - q_before
        max_norm_error = max(max_norm_error, jump_norm_error)
        max_particle_error = max(
            max_particle_error,
            particle_number_error(basis, state),
        )
        if event_time >= retention_start:
            events.append(EventRecord(event_count, event_time, site))
        event_count += 1
        time_value = event_time
    if any(
        right.event_time <= left.event_time for left, right in zip(events, events[1:], strict=False)
    ):
        raise AssertionError("trajectory event clocks are not strictly increasing")
    if any(not retention_start <= event.event_time < end_time for event in events):
        raise AssertionError("retained event leaves the right-open record window")
    transport = None
    if current_operator is not None:
        transport = finalize_transport(
            coherent_signed=coherent,
            coherent_absolute=coherent_abs,
            measurement_innovation=innovation,
            q_start_value=q_start_value,
            q_end_value=q_a(state, half_count),
            sentinel_signed=sentinel,
            sentinel_absolute=sentinel_abs,
            continuity_tolerance=continuity_tolerance,
        )
    return IntervalResult(
        state=state,
        events=tuple(events),
        total_event_count=event_count,
        start_time=start_time,
        end_time=end_time,
        maximum_norm_error=max_norm_error,
        maximum_particle_number_error=max_particle_error,
        transport=transport,
    )


def prefix_stream_id(unit: PreparationUnit, denominator_id: str) -> str:
    digest = sha256(f"{PLAN_ID}:{unit.unit_id}:{denominator_id}:prefix".encode()).hexdigest()[:24]
    return f"stream.quantum-response-control.prefix.{digest}"


def prepare_prefix(
    *,
    scientific_seed: int,
    unit: PreparationUnit,
    denominator_id: str,
    basis: FixedNumberBasis,
    hold_hamiltonian: csr_matrix,
    gamma: float,
    cutoff: float = 200.0,
    retained_history: float = 20.0,
) -> PrefixCheckpoint:
    if retained_history <= 0 or retained_history > cutoff:
        raise ValueError("retained history window is invalid")
    stream_id = prefix_stream_id(unit, denominator_id)
    rng = _rng(scientific_seed)
    result = simulate_interval(
        basis=basis,
        propagator=SparseKrylovPropagator(hold_hamiltonian),
        initial_state=basis.state_vector(unit.initial_state),
        start_time=0.0,
        duration=cutoff,
        gamma=gamma,
        rng=rng,
        retention_start=cutoff - retained_history,
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
        preparation_rng_state=rng_state(rng),
        stream_id=stream_id,
        maximum_norm_error=result.maximum_norm_error,
        maximum_particle_number_error=result.maximum_particle_number_error,
    )


def future_stream_id(
    prefix: PrefixCheckpoint,
    action: Action,
    draw_index: int,
    purpose: str = "future",
) -> str:
    if draw_index < 0 or not purpose:
        raise ValueError("future stream identity is invalid")
    digest = sha256(
        f"{PLAN_ID}:{prefix.roster_id}:{prefix.unit_id}:{action.value}:"
        f"{draw_index}:{purpose}".encode()
    ).hexdigest()[:24]
    return f"stream.quantum-response-control.{purpose}.{digest}"


def simulate_future(
    *,
    scientific_seed: int,
    prefix: PrefixCheckpoint,
    basis: FixedNumberBasis,
    hamiltonians: Mapping[Action, csr_matrix],
    current_operators: Mapping[Action, csr_matrix],
    hamiltonian_fingerprints: Mapping[Action, str],
    gamma: float,
    action: Action,
    draw_index: int,
    horizon: float,
    epsilon: float,
    quadrature_order: int = 8,
    quadrature_sentinel_order: int = 16,
    continuity_tolerance: float = 1e-8,
    purpose: str = "future",
    retain_terminal_state: bool = True,
) -> FutureDraw:
    if set(hamiltonians) != set(Action) or set(current_operators) != set(Action):
        raise ValueError("future source requires the complete action chart")
    if set(hamiltonian_fingerprints) != set(Action):
        raise ValueError("future source lacks action implementation identities")
    stream_id = future_stream_id(prefix, action, draw_index, purpose)
    rng = _rng(scientific_seed)
    result = simulate_interval(
        basis=basis,
        propagator=SparseKrylovPropagator(hamiltonians[action]),
        initial_state=prefix.state,
        start_time=prefix.cutoff_time,
        duration=horizon,
        gamma=gamma,
        rng=rng,
        current_operator=current_operators[action],
        quadrature_order=quadrature_order,
        quadrature_sentinel_order=quadrature_sentinel_order,
        continuity_tolerance=continuity_tolerance,
    )
    if result.transport is None:
        raise AssertionError("future interval omitted required transport")
    hold = hamiltonians[Action.HOLD]
    delta_h = hamiltonians[action] - hold
    work_on = float(np.real(np.vdot(prefix.state, delta_h @ prefix.state)))
    work_off = float(np.real(np.vdot(result.state, (-delta_h) @ result.state)))
    terminal_energy = float(np.real(np.vdot(result.state, hold @ result.state)))
    ledger = ActionLedger(
        requested_action=action,
        accepted_action=action,
        applied_action=action,
        requested_start=prefix.cutoff_time,
        requested_end=prefix.cutoff_time + horizon,
        realized_start=prefix.cutoff_time,
        realized_end=prefix.cutoff_time + horizon,
        hamiltonian_sha256=hamiltonian_fingerprints[action],
        delivery_discrepancy=0.0,
    )
    return FutureDraw(
        scientific_seed=scientific_seed,
        unit_id=prefix.unit_id,
        action=action,
        draw_index=draw_index,
        stream_id=stream_id,
        events=result.events,
        terminal_state=result.state if retain_terminal_state else None,
        ledger=ledger,
        work_on=work_on,
        work_off=work_off,
        work_abs=abs(work_on) + abs(work_off),
        effort=(action.sign * epsilon) ** 2 * horizon,
        terminal_q_a=result.transport.q_end,
        terminal_energy=terminal_energy,
        transport=result.transport,
        maximum_norm_error=max(prefix.maximum_norm_error, result.maximum_norm_error),
        maximum_particle_number_error=max(
            prefix.maximum_particle_number_error,
            result.maximum_particle_number_error,
        ),
    )


def simulate_future_block(
    *,
    prefix: PrefixCheckpoint,
    basis: FixedNumberBasis,
    hamiltonians: Mapping[Action, csr_matrix],
    current_operators: Mapping[Action, csr_matrix],
    hamiltonian_fingerprints: Mapping[Action, str],
    gamma: float,
    actions: Sequence[Action],
    draw_start: int,
    draw_stop: int,
    scientific_future_seeds: tuple[tuple[Action, int, int], ...],
    horizon: float,
    epsilon: float,
    quadrature_order: int,
    quadrature_sentinel_order: int,
    continuity_tolerance: float,
    purpose: str,
    retain_terminal_state: bool = True,
) -> tuple[FutureDraw, ...]:
    if draw_start < 0 or draw_stop <= draw_start:
        raise ValueError("future draw block is invalid")
    expected = tuple((action, draw) for action in actions for draw in range(draw_start, draw_stop))
    if (
        not isinstance(scientific_future_seeds, tuple)
        or len(scientific_future_seeds) != len(expected)
        or any(not isinstance(row, tuple) or len(row) != 3 for row in scientific_future_seeds)
        or tuple((action, draw) for action, draw, _ in scientific_future_seeds) != expected
        or any(not isinstance(action, Action) or type(draw) is not int for action, draw, _ in scientific_future_seeds)
        or len(set(expected)) != len(expected)
        or any(type(seed) is not int or not 0 <= seed < 2**128 for _, _, seed in scientific_future_seeds)
        or len({row[2] for row in scientific_future_seeds}) != len(expected)
    ):
        raise ValueError("future scientific seed census differs from the exact action/draw block")
    seeds = {(action, draw): seed for action, draw, seed in scientific_future_seeds}
    return tuple(
        simulate_future(
            scientific_seed=seeds[action, draw_index],
            prefix=prefix,
            basis=basis,
            hamiltonians=hamiltonians,
            current_operators=current_operators,
            hamiltonian_fingerprints=hamiltonian_fingerprints,
            gamma=gamma,
            action=action,
            draw_index=draw_index,
            horizon=horizon,
            epsilon=epsilon,
            quadrature_order=quadrature_order,
            quadrature_sentinel_order=quadrature_sentinel_order,
            continuity_tolerance=continuity_tolerance,
            purpose=purpose,
            retain_terminal_state=retain_terminal_state,
        )
        for action in actions
        for draw_index in range(draw_start, draw_stop)
    )


__all__ = [
    "ActionLedger",
    "EventRecord",
    "FutureDraw",
    "IntervalResult",
    "PrefixCheckpoint",
    "future_stream_id",
    "prefix_stream_id",
    "prepare_prefix",
    "restore_rng",
    "rng_state",
    "simulate_future",
    "simulate_future_block",
    "simulate_interval",
]
