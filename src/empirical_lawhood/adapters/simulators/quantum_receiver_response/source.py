"""Event-driven monitored-chain source for quantum receiver response."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .basis import ComplexVector, FixedNumberBasis, Propagator, SparseKrylovPropagator, apply_jump, normalize_state, particle_number_error, site_probabilities
from .contracts import Action, PreparationUnit


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
    start_time: float
    end_time: float
    maximum_norm_error: float
    maximum_particle_number_error: float


@dataclass(frozen=True, slots=True)
class PrefixCheckpoint:
    scientific_seed: int
    unit_id: str
    denominator_id: str
    k_left: int
    initial_state: int
    cutoff_time: float
    state: ComplexVector
    history_events: tuple[EventRecord, ...]
    preparation_rng_state: Mapping[str, object]
    maximum_norm_error: float
    maximum_particle_number_error: float

    @property
    def state_sha256(self) -> str:
        array = np.asarray(self.state, dtype="<c16")
        return sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class FutureDraw:
    scientific_seed: int
    unit_id: str
    action: Action
    draw_index: int
    stream_id: str
    events: tuple[EventRecord, ...]
    terminal_state: ComplexVector | None
    requested_action: Action
    accepted_action: Action
    applied_action: Action
    realized_action_start: float
    realized_action_end: float
    work_on: float
    work_off: float
    work_abs: float
    effort: float
    terminal_q_a: float
    maximum_norm_error: float
    maximum_particle_number_error: float


def _rng(seed: int) -> np.random.Generator:
    if type(seed) is not int or not 0 <= seed < 2**64:
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
) -> IntervalResult:
    """Simulate one right-open event interval under fixed total hazard."""

    if not math.isfinite(start_time) or start_time < 0.0:
        raise ValueError("interval start is invalid")
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("interval duration must be positive and finite")
    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("monitoring strength must be positive and finite")
    end_time = start_time + duration
    time_value = start_time
    state, initial_norm_error = normalize_state(initial_state.copy())
    max_norm_error = initial_norm_error
    max_particle_error = particle_number_error(basis, state)
    events: list[EventRecord] = []
    hazard = gamma * basis.particles
    while True:
        waiting = float(rng.exponential(1.0 / hazard))
        event_time = time_value + waiting
        if event_time >= end_time:
            state = propagator.propagate(state, end_time - time_value)
            state, norm_error = normalize_state(state)
            max_norm_error = max(max_norm_error, norm_error)
            max_particle_error = max(
                max_particle_error,
                particle_number_error(basis, state),
            )
            break
        state = propagator.propagate(state, waiting)
        state, norm_error = normalize_state(state)
        max_norm_error = max(max_norm_error, norm_error)
        probabilities = site_probabilities(basis, state)
        site = int(rng.choice(basis.l_sites, p=probabilities))
        state, jump_norm_error = apply_jump(basis, state, site)
        max_norm_error = max(max_norm_error, jump_norm_error)
        max_particle_error = max(
            max_particle_error,
            particle_number_error(basis, state),
        )
        events.append(
            EventRecord(
                event_index=len(events),
                event_time=event_time,
                site=site,
            )
        )
        time_value = event_time
    if any(
        right.event_time <= left.event_time for left, right in zip(events, events[1:], strict=False)
    ):
        raise AssertionError("trajectory event clocks are not strictly increasing")
    if any(event.event_time >= end_time for event in events):
        raise AssertionError("right-open interval admitted an endpoint event")
    return IntervalResult(
        state=state,
        events=tuple(events),
        start_time=start_time,
        end_time=end_time,
        maximum_norm_error=max_norm_error,
        maximum_particle_number_error=max_particle_error,
    )


def prepare_prefix(
    *,
    scientific_seed: int,
    unit: PreparationUnit,
    denominator_id: str,
    basis: FixedNumberBasis,
    hold_hamiltonian: csr_matrix,
    gamma: float,
    burn_in: float,
    history_horizon: float,
) -> PrefixCheckpoint:
    rng = _rng(scientific_seed)
    propagator = SparseKrylovPropagator(hold_hamiltonian)
    initial = basis.state_vector(unit.initial_state)
    burn = simulate_interval(
        basis=basis,
        propagator=propagator,
        initial_state=initial,
        start_time=0.0,
        duration=burn_in,
        gamma=gamma,
        rng=rng,
    )
    history = simulate_interval(
        basis=basis,
        propagator=propagator,
        initial_state=burn.state,
        start_time=burn_in,
        duration=history_horizon,
        gamma=gamma,
        rng=rng,
    )
    return PrefixCheckpoint(
        scientific_seed=scientific_seed,
        unit_id=unit.unit_id,
        denominator_id=denominator_id,
        k_left=unit.k_left,
        initial_state=unit.initial_state,
        cutoff_time=burn_in + history_horizon,
        state=history.state,
        history_events=history.events,
        preparation_rng_state=rng_state(rng),
        maximum_norm_error=max(
            burn.maximum_norm_error,
            history.maximum_norm_error,
        ),
        maximum_particle_number_error=max(
            burn.maximum_particle_number_error,
            history.maximum_particle_number_error,
        ),
    )


def burn_in_comparison(
    *,
    scientific_seed: int,
    unit: PreparationUnit,
    basis: FixedNumberBasis,
    hold_hamiltonian: csr_matrix,
    gamma: float,
    burn_in: float,
    extension: float,
    observation_horizon: float,
) -> tuple[PrefixCheckpoint, PrefixCheckpoint]:
    """Return matched windows after base and extended burn-in."""

    if extension <= observation_horizon:
        raise ValueError("burn-in extension must exceed the observation window")
    rng = _rng(scientific_seed)
    propagator = SparseKrylovPropagator(hold_hamiltonian)
    first_burn = simulate_interval(
        basis=basis,
        propagator=propagator,
        initial_state=basis.state_vector(unit.initial_state),
        start_time=0.0,
        duration=burn_in - observation_horizon,
        gamma=gamma,
        rng=rng,
    )
    first_window = simulate_interval(
        basis=basis,
        propagator=propagator,
        initial_state=first_burn.state,
        start_time=burn_in - observation_horizon,
        duration=observation_horizon,
        gamma=gamma,
        rng=rng,
    )
    first_rng_state = rng_state(rng)
    between = simulate_interval(
        basis=basis,
        propagator=propagator,
        initial_state=first_window.state,
        start_time=burn_in,
        duration=extension - observation_horizon,
        gamma=gamma,
        rng=rng,
    )
    second_window = simulate_interval(
        basis=basis,
        propagator=propagator,
        initial_state=between.state,
        start_time=burn_in + extension - observation_horizon,
        duration=observation_horizon,
        gamma=gamma,
        rng=rng,
    )
    second_rng_state = rng_state(rng)

    def checkpoint(
        result: IntervalResult,
        token: str,
        saved_rng_state: Mapping[str, object],
    ) -> PrefixCheckpoint:
        return PrefixCheckpoint(
            scientific_seed=scientific_seed,
            unit_id=f"{unit.unit_id}.{token}",
            denominator_id=f"gamma-{gamma:g}",
            k_left=unit.k_left,
            initial_state=unit.initial_state,
            cutoff_time=result.end_time,
            state=result.state,
            history_events=result.events,
            preparation_rng_state=saved_rng_state,
            maximum_norm_error=result.maximum_norm_error,
            maximum_particle_number_error=result.maximum_particle_number_error,
        )

    return (
        checkpoint(first_window, "base", first_rng_state),
        checkpoint(second_window, "extended", second_rng_state),
    )


def future_stream_id(unit_id: str, action: Action, draw_index: int) -> str:
    if draw_index < 0:
        raise ValueError("future draw index must be nonnegative")
    digest = sha256(f"{unit_id}:{action.value}:{draw_index}".encode()).hexdigest()[:20]
    return f"stream.quantum-receiver-response.{digest}"


def simulate_future(
    *,
    scientific_seed: int,
    prefix: PrefixCheckpoint,
    basis: FixedNumberBasis,
    hamiltonians: Mapping[Action, csr_matrix],
    gamma: float,
    action: Action,
    draw_index: int,
    horizon: float,
    epsilon: float,
) -> FutureDraw:
    if set(hamiltonians) != set(Action):
        raise ValueError("future source requires the complete action chart")
    stream_id = future_stream_id(prefix.unit_id, action, draw_index)
    rng = _rng(scientific_seed)
    result = simulate_interval(
        basis=basis,
        propagator=SparseKrylovPropagator(hamiltonians[action]),
        initial_state=prefix.state,
        start_time=prefix.cutoff_time,
        duration=horizon,
        gamma=gamma,
        rng=rng,
    )
    delta_h = hamiltonians[action] - hamiltonians[Action.HOLD]
    work_on = float(np.real(np.vdot(prefix.state, delta_h @ prefix.state)))
    work_off = float(np.real(np.vdot(result.state, (-delta_h) @ result.state)))
    half_count_operator = basis.occupations[:, : basis.l_sites // 2].sum(axis=1)
    terminal_q_a = float((np.abs(result.state) ** 2) @ half_count_operator)
    return FutureDraw(
        scientific_seed=scientific_seed,
        unit_id=prefix.unit_id,
        action=action,
        draw_index=draw_index,
        stream_id=stream_id,
        events=result.events,
        terminal_state=result.state,
        requested_action=action,
        accepted_action=action,
        applied_action=action,
        realized_action_start=prefix.cutoff_time,
        realized_action_end=prefix.cutoff_time + horizon,
        work_on=work_on,
        work_off=work_off,
        work_abs=abs(work_on) + abs(work_off),
        effort=(action.sign * epsilon) ** 2 * horizon,
        terminal_q_a=terminal_q_a,
        maximum_norm_error=max(
            prefix.maximum_norm_error,
            result.maximum_norm_error,
        ),
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
    gamma: float,
    actions: Sequence[Action],
    draw_start: int,
    draw_stop: int,
    scientific_future_seeds: tuple[tuple[Action, int, int], ...],
    horizon: float,
    epsilon: float,
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
        or any(type(seed) is not int or not 0 <= seed < 2**64 for _, _, seed in scientific_future_seeds)
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
            gamma=gamma,
            action=action,
            draw_index=draw_index,
            horizon=horizon,
            epsilon=epsilon,
        )
        for action in actions
        for draw_index in range(draw_start, draw_stop)
    )


__all__ = [
    "EventRecord",
    "FutureDraw",
    "IntervalResult",
    "PrefixCheckpoint",
    "burn_in_comparison",
    "future_stream_id",
    "prepare_prefix",
    "restore_rng",
    "rng_state",
    "simulate_future",
    "simulate_future_block",
    "simulate_interval",
]
