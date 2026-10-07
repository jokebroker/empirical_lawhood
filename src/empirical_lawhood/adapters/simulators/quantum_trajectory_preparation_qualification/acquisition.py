"""Continuous strong-source acquisition and exact observation reduction."""

from __future__ import annotations

from typing import Sequence

from ..quantum_scientific_seeds import validate_scientific_seed

from .basis import build_basis
from .checkpoint import checkpoint_from_result, trajectory_from_bytes, trajectory_to_bytes
from .evolution import propagator_for_view, simulate_trajectory
from .joint_inference import PreparedObservation
from .operators import build_hamiltonian
from .receivers import observe
from .schemas import QuantumTrajectoryPreparationQualificationConfig
from .types import Action, Denominator, PreparationUnit, View


def development_endpoints(config: QuantumTrajectoryPreparationQualificationConfig) -> tuple[float, ...]:
    extension = float(config.burnin_extension)
    return tuple(
        sorted(
            {
                endpoint
                for candidate_decimal in config.burnin_candidates
                for endpoint in (
                    float(candidate_decimal),
                    float(candidate_decimal) + extension,
                    2.0 * float(candidate_decimal),
                    2.0 * float(candidate_decimal) + extension,
                )
            }
        )
    )


def evaluation_endpoints(
    selected_burnin: float,
    extension: float,
) -> tuple[float, ...]:
    return tuple(
        sorted(
            {
                selected_burnin,
                selected_burnin + extension,
                2.0 * selected_burnin,
                2.0 * selected_burnin + extension,
            }
        )
    )


def trajectory_worker(
    unit: PreparationUnit,
    *,
    scientific_seed: int,
    endpoints: Sequence[float],
) -> bytes:
    validate_scientific_seed(scientific_seed, bits=64)
    basis = build_basis(unit.l_sites, unit.particles)
    hamiltonian = build_hamiltonian(
        basis,
        action=Action.HOLD,
        epsilon=0.20,
        j_xy=1.0,
        j_z=1.0,
    )
    trajectory = simulate_trajectory(
        unit_id=unit.unit_id,
        initial_state_id=unit.initial_state,
        preparation_seed=scientific_seed,
        basis=basis,
        propagator=propagator_for_view(View.SPARSE, hamiltonian),
        denominator=Denominator.STRONG,
        action=Action.HOLD,
        view=View.SPARSE,
        endpoints=endpoints,
    )
    return trajectory_to_bytes(trajectory)


def continue_trajectory_worker(
    payload: bytes,
    *,
    endpoints: Sequence[float],
) -> bytes:
    previous = trajectory_from_bytes(payload)
    checkpoint = checkpoint_from_result(previous, previous.end_time)
    basis = build_basis(previous.l_sites, previous.particles)
    hamiltonian = build_hamiltonian(
        basis,
        action=previous.action,
        epsilon=0.20,
        j_xy=1.0,
        j_z=1.0,
    )
    result = simulate_trajectory(
        unit_id=previous.unit_id,
        initial_state_id=previous.initial_state,
        preparation_seed=previous.preparation_seed,
        basis=basis,
        propagator=propagator_for_view(previous.view, hamiltonian),
        denominator=previous.denominator,
        action=previous.action,
        view=previous.view,
        endpoints=endpoints,
        initial_state=checkpoint.state,
        start_time=checkpoint.clock,
        prior_events=checkpoint.events,
        prior_diagnostics=checkpoint.diagnostics,
        rng_state=checkpoint.rng_state,
        rng_draw_count=checkpoint.rng_draw_count,
    )
    return trajectory_to_bytes(result)


def observations_from_trajectory(
    payload: bytes,
    unit: PreparationUnit,
    *,
    config: QuantumTrajectoryPreparationQualificationConfig,
) -> tuple[PreparedObservation, ...]:
    trajectory = trajectory_from_bytes(payload)
    if trajectory.unit_id != unit.unit_id:
        raise ValueError("trajectory parent unit differs from roster")
    if (
        trajectory.denominator != Denominator.STRONG
        or trajectory.action != Action.HOLD
        or trajectory.view != View.SPARSE
    ):
        raise ValueError("trajectory denominator/action/view differs from preparation qualification")
    basis = build_basis(unit.l_sites, unit.particles)
    hold = build_hamiltonian(
        basis,
        action=Action.HOLD,
        epsilon=float(config.epsilon),
        j_xy=float(config.j_xy),
        j_z=float(config.j_z),
    )
    valid = bool(
        trajectory.maximum_propagator_norm_residual <= float(config.propagator_norm_tolerance)
        and trajectory.maximum_post_jump_norm_residual <= float(config.post_jump_norm_tolerance)
        and trajectory.maximum_projected_mass_identity_residual
        <= float(config.projected_mass_tolerance)
        and trajectory.maximum_mark_probability_sum_residual
        <= float(config.mark_probability_sum_tolerance)
        and trajectory.maximum_particle_number_residual <= float(config.particle_number_tolerance)
    )
    rows: list[PreparedObservation] = []
    for endpoint in sorted(trajectory.snapshots):
        observation = observe(
            events=trajectory.events,
            state=trajectory.snapshots[endpoint],
            endpoint=endpoint,
            basis=basis,
            hold_hamiltonian=hold,
            gamma=trajectory.denominator.gamma,
            observation_horizon=float(config.observation_horizon),
        )
        rows.append(
            PreparedObservation(
                unit_id=unit.unit_id,
                unit_index=unit.unit_index,
                roster_id=unit.roster_id,
                k_left=unit.k_left,
                boundary_occupation=unit.boundary_occupation,
                translation_orbit=unit.translation_orbit,
                denominator=trajectory.denominator,
                endpoint=endpoint,
                values=observation.values,
                energy=observation.energy,
                event_rate=observation.event_rate,
                state_sha256=observation.state_sha256,
                valid=valid,
                reason_code="OK" if valid else "PATH_NUMERICAL_VALIDITY_FAILED",
            )
        )
    return tuple(rows)


__all__ = [
    "continue_trajectory_worker",
    "development_endpoints",
    "evaluation_endpoints",
    "observations_from_trajectory",
    "trajectory_worker",
]
