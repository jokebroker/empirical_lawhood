"""Event-driven continuous trajectory evolution with explicit PRNG accounting."""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from typing import Mapping, Sequence

import numpy as np

from .basis import FixedNumberBasis
from .jumps import apply_jump, jump_masses, sample_mark
from .operators import DenseEigensystemPropagator, Propagator, SparseKrylovPropagator, normalize_state, particle_number_residual
from .types import Action, ComplexVector, Denominator, EventRecord, JumpDiagnostic, TrajectoryResult, View


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


class CountingRandom:
    def __init__(
        self,
        seed: int,
        *,
        state: Mapping[str, object] | None = None,
        draw_count: int = 0,
    ) -> None:
        self._generator = np.random.Generator(np.random.Philox(seed))
        if state is not None:
            self._generator.bit_generator.state = dict(state)
        self.draw_count = draw_count

    def exponential(self, scale: float) -> float:
        self.draw_count += 1
        return float(self._generator.exponential(scale))

    def uniform(self) -> float:
        self.draw_count += 1
        return float(self._generator.random())

    @property
    def state(self) -> Mapping[str, object]:
        result = _json_safe(self._generator.bit_generator.state)
        if not isinstance(result, Mapping):
            raise AssertionError("Philox state must serialize as an object")
        json.dumps(result, sort_keys=True, allow_nan=False)
        return result


def action_at_time(
    *,
    actions: Sequence[Action],
    switch_times: Sequence[float],
    event_time: float,
) -> Action:
    """Return the action on right-open segments with switches applied first."""

    if len(actions) != len(switch_times) + 1:
        raise ValueError("one action is required for every schedule segment")
    switches = tuple(float(value) for value in switch_times)
    if (
        any(not math.isfinite(value) for value in switches)
        or tuple(sorted(set(switches))) != switches
        or not math.isfinite(event_time)
        or event_time < 0.0
    ):
        raise ValueError("action schedule clocks must be finite and increasing")
    return actions[bisect_right(switches, event_time)]


def propagator_for_view(view: View, hamiltonian: object) -> Propagator:
    from scipy.sparse import csr_matrix

    if not isinstance(hamiltonian, csr_matrix):
        raise TypeError("hamiltonian must be CSR")
    if view == View.SPARSE:
        return SparseKrylovPropagator(hamiltonian)
    if view == View.DENSE:
        return DenseEigensystemPropagator(hamiltonian)
    raise ValueError("analytic view cannot run a trajectory")


def simulate_trajectory(
    *,
    unit_id: str,
    initial_state_id: int,
    preparation_seed: int,
    basis: FixedNumberBasis,
    propagator: Propagator,
    denominator: Denominator,
    action: Action,
    view: View,
    endpoints: Sequence[float],
    initial_state: ComplexVector | None = None,
    start_time: float = 0.0,
    prior_events: Sequence[EventRecord] = (),
    prior_diagnostics: Sequence[JumpDiagnostic] = (),
    rng_state: Mapping[str, object] | None = None,
    rng_draw_count: int = 0,
) -> TrajectoryResult:
    ordered_endpoints = tuple(float(value) for value in endpoints)
    if (
        not ordered_endpoints
        or any(not math.isfinite(value) for value in ordered_endpoints)
        or any(value <= start_time for value in ordered_endpoints)
        or tuple(sorted(set(ordered_endpoints))) != ordered_endpoints
    ):
        raise ValueError("trajectory endpoints must be unique and increasing")
    state = (
        basis.state_vector(initial_state_id)
        if initial_state is None
        else np.asarray(initial_state, dtype=np.complex128).copy()
    )
    state, initial_norm_residual = normalize_state(state)
    maximum_prop = initial_norm_residual
    maximum_post = 0.0
    maximum_mass = 0.0
    maximum_prob = 0.0
    maximum_particle = particle_number_residual(basis, state)
    rng = CountingRandom(
        preparation_seed,
        state=rng_state,
        draw_count=rng_draw_count,
    )
    events = list(prior_events)
    diagnostics = list(prior_diagnostics)
    time_value = float(start_time)
    hazard = denominator.gamma * basis.particles
    snapshots: dict[float, ComplexVector] = {}
    checkpoint_states: dict[float, Mapping[str, object]] = {}
    checkpoint_counts: dict[float, int] = {}
    for endpoint in ordered_endpoints:
        while True:
            waiting = rng.exponential(1.0 / hazard)
            event_time = time_value + waiting
            if event_time >= endpoint:
                propagated = propagator.propagate(state, endpoint - time_value)
                state, prop_residual = normalize_state(propagated)
                maximum_prop = max(maximum_prop, prop_residual)
                maximum_particle = max(
                    maximum_particle,
                    particle_number_residual(basis, state),
                )
                time_value = endpoint
                break
            propagated = propagator.propagate(state, waiting)
            state, prop_residual = normalize_state(propagated)
            masses = jump_masses(basis, state)
            mark_uniform = rng.uniform()
            site = sample_mark(masses.mark_probabilities, mark_uniform)
            applied = apply_jump(
                basis,
                state,
                site,
                expected_mass=float(masses.projected_masses[site]),
            )
            state = applied.state
            particle_residual = particle_number_residual(basis, state)
            event_index = len(events)
            events.append(
                EventRecord(
                    event_index=event_index,
                    event_time=event_time,
                    site=site,
                )
            )
            diagnostics.append(
                JumpDiagnostic(
                    event_index=event_index,
                    event_time=event_time,
                    site=site,
                    propagator_norm_residual=prop_residual,
                    projected_mass_expectation=applied.projected_mass_expectation,
                    projected_vector_norm_squared=applied.projected_vector_norm_squared,
                    projected_mass_identity_residual=(applied.projected_mass_identity_residual),
                    selected_mark_probability=float(masses.mark_probabilities[site]),
                    mass_sum_residual=masses.mass_sum_residual,
                    mark_probability_sum_residual=masses.probability_sum_residual,
                    post_jump_norm_residual=applied.post_jump_norm_residual,
                    particle_number_residual=particle_residual,
                )
            )
            maximum_prop = max(maximum_prop, prop_residual)
            maximum_post = max(maximum_post, applied.post_jump_norm_residual)
            maximum_mass = max(
                maximum_mass,
                applied.projected_mass_identity_residual,
            )
            maximum_prob = max(maximum_prob, masses.probability_sum_residual)
            maximum_particle = max(maximum_particle, particle_residual)
            time_value = event_time
        snapshots[endpoint] = state.copy()
        checkpoint_states[endpoint] = rng.state
        checkpoint_counts[endpoint] = rng.draw_count
    if any(
        right.event_time <= left.event_time for left, right in zip(events, events[1:], strict=False)
    ):
        raise AssertionError("trajectory event clocks are not strictly increasing")
    if any(event.event_time >= ordered_endpoints[-1] for event in events):
        raise AssertionError("right-open trajectory admitted an endpoint event")
    return TrajectoryResult(
        unit_id=unit_id,
        denominator=denominator,
        action=action,
        view=view,
        l_sites=basis.l_sites,
        particles=basis.particles,
        initial_state=initial_state_id,
        preparation_seed=preparation_seed,
        end_time=ordered_endpoints[-1],
        terminal_state=state,
        events=tuple(events),
        diagnostics=tuple(diagnostics),
        snapshots=snapshots,
        checkpoint_rng_states=checkpoint_states,
        checkpoint_rng_draw_counts=checkpoint_counts,
        maximum_propagator_norm_residual=maximum_prop,
        maximum_post_jump_norm_residual=maximum_post,
        maximum_projected_mass_identity_residual=maximum_mass,
        maximum_mark_probability_sum_residual=maximum_prob,
        maximum_particle_number_residual=maximum_particle,
    )


__all__ = [
    "CountingRandom",
    "action_at_time",
    "propagator_for_view",
    "simulate_trajectory",
]
