"""Development/evaluation path reductions and immutable burn-in selection."""

from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence

from ..quantum_scientific_seeds import validate_scientific_seed

from .basis import build_basis
from .checkpoint import checkpoint_from_result, trajectory_from_bytes, trajectory_to_bytes
from .evolution import propagator_for_view, simulate_trajectory
from .inference import ComparisonSpec, PreparedObservation, preparation_memory, stationarity_intervals
from .operators import build_hamiltonian
from .receivers import observe
from .schemas import QuantumTrajectoryBurninScreenConfig
from .types import Action, Denominator, MemoryResult, PreparationUnit, Stage, View


def development_endpoints(config: QuantumTrajectoryBurninScreenConfig) -> tuple[float, ...]:
    endpoints = {float(candidate) for candidate in config.burnin_candidates}
    for candidate in config.burnin_candidates:
        value = float(candidate)
        endpoints.update(
            {
                value + float(config.burnin_extension),
                2.0 * value,
                2.0 * value + float(config.burnin_extension),
            }
        )
    return tuple(sorted(endpoints))


def evaluation_endpoints(
    selected_burnin: float,
    extension: float,
) -> tuple[float, ...]:
    return (
        selected_burnin,
        selected_burnin + extension,
        2.0 * selected_burnin,
        2.0 * selected_burnin + extension,
    )


def trajectory_worker(
    unit: PreparationUnit,
    denominator: Denominator,
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
    result = simulate_trajectory(
        unit_id=unit.unit_id,
        initial_state_id=unit.initial_state,
        preparation_seed=scientific_seed,
        basis=basis,
        propagator=propagator_for_view(View.SPARSE, hamiltonian),
        denominator=denominator,
        action=Action.HOLD,
        view=View.SPARSE,
        endpoints=endpoints,
    )
    return trajectory_to_bytes(result)


def continue_trajectory_worker(
    payload: bytes,
    *,
    endpoints: Sequence[float],
) -> bytes:
    """Continue one exact path without changing its preparation or PRNG stream."""

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
    config: QuantumTrajectoryBurninScreenConfig,
) -> tuple[PreparedObservation, ...]:
    trajectory = trajectory_from_bytes(payload)
    if trajectory.unit_id != unit.unit_id:
        raise ValueError("trajectory parent unit differs from roster")
    basis = build_basis(unit.l_sites, unit.particles)
    hold = build_hamiltonian(
        basis,
        action=Action.HOLD,
        epsilon=float(config.epsilon),
        j_xy=float(config.j_xy),
        j_z=float(config.j_z),
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
                unit_index=unit.unit_index,
                unit_id=unit.unit_id,
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
                valid=bool(
                    trajectory.maximum_propagator_norm_residual
                    <= float(config.propagator_norm_tolerance)
                    and trajectory.maximum_post_jump_norm_residual
                    <= float(config.post_jump_norm_tolerance)
                    and trajectory.maximum_projected_mass_identity_residual
                    <= float(config.projected_mass_tolerance)
                    and trajectory.maximum_mark_probability_sum_residual
                    <= float(config.mark_probability_sum_tolerance)
                    and trajectory.maximum_particle_number_residual
                    <= float(config.particle_number_tolerance)
                ),
                reason_code="OK",
            )
        )
    return tuple(rows)


def _candidate_comparisons(
    candidate: float,
    denominator: Denominator,
    extension: float,
) -> tuple[ComparisonSpec, ComparisonSpec]:
    return (
        ComparisonSpec(
            denominator=denominator,
            comparison_id=f"candidate-{candidate:g}",
            earlier=candidate,
            later=candidate + extension,
        ),
        ComparisonSpec(
            denominator=denominator,
            comparison_id=f"sentinel-{2.0 * candidate:g}",
            earlier=2.0 * candidate,
            later=2.0 * candidate + extension,
        ),
    )


def analyze_candidate(
    records: Sequence[PreparedObservation],
    *,
    candidate: float,
    denominators: Sequence[Denominator],
    config: QuantumTrajectoryBurninScreenConfig,
    stage: Stage,
) -> dict[str, object]:
    comparisons = tuple(
        comparison
        for denominator in denominators
        for comparison in _candidate_comparisons(
            candidate,
            denominator,
            float(config.burnin_extension),
        )
    )
    intervals = stationarity_intervals(
        records,
        comparisons,
        config=config,
        stage=stage,
    )
    memory: list[MemoryResult] = []
    for denominator in denominators:
        memory.extend(
            preparation_memory(
                records,
                denominator=denominator,
                endpoint=candidate,
                clock_id=f"candidate-{candidate:g}",
                config=config,
                stage=stage,
            )
        )
        memory.extend(
            preparation_memory(
                records,
                denominator=denominator,
                endpoint=2.0 * candidate,
                clock_id=f"sentinel-{2.0 * candidate:g}",
                config=config,
                stage=stage,
            )
        )
    denominator_pass: dict[str, bool] = {}
    for denominator in denominators:
        denominator_intervals = [row for row in intervals if row.denominator == denominator]
        denominator_memory = [row for row in memory if row.denominator == denominator]
        denominator_pass[denominator.value] = bool(
            denominator_intervals
            and all(row.passed for row in denominator_intervals)
            and not any(row.material for row in denominator_memory)
        )
    return {
        "candidate": candidate,
        "sentinel": 2.0 * candidate,
        "stage": stage.value,
        "denominator_pass": denominator_pass,
        "intervals": [asdict(row) for row in intervals],
        "memory": [asdict(row) for row in memory],
    }


def select_development_burnins(
    records: Sequence[PreparedObservation],
    *,
    denominators: Sequence[Denominator],
    config: QuantumTrajectoryBurninScreenConfig,
) -> dict[str, object]:
    selected: dict[str, float] = {}
    candidate_results: list[dict[str, object]] = []
    for candidate_decimal in config.burnin_candidates:
        candidate = float(candidate_decimal)
        result = analyze_candidate(
            records,
            candidate=candidate,
            denominators=denominators,
            config=config,
            stage=Stage.DEVELOPMENT,
        )
        candidate_results.append(result)
        denominator_pass = result["denominator_pass"]
        if not isinstance(denominator_pass, Mapping):
            raise AssertionError("candidate denominator pass map is invalid")
        for denominator in denominators:
            if (
                denominator.value not in selected
                and denominator_pass.get(denominator.value) is True
            ):
                selected[denominator.value] = candidate
        if len(selected) == len(denominators):
            break
    return {
        "stage": Stage.DEVELOPMENT.value,
        "eligible_denominators": [value.value for value in denominators],
        "selected_burnin": selected,
        "candidate_results": candidate_results,
    }


def analyze_evaluation(
    records: Sequence[PreparedObservation],
    *,
    selected_burnins: Mapping[Denominator, float],
    config: QuantumTrajectoryBurninScreenConfig,
) -> dict[str, object]:
    comparisons = tuple(
        comparison
        for denominator, selected in selected_burnins.items()
        for comparison in _candidate_comparisons(
            selected,
            denominator,
            float(config.burnin_extension),
        )
    )
    intervals = stationarity_intervals(
        records,
        comparisons,
        config=config,
        stage=Stage.EVALUATION,
    )
    memory: list[MemoryResult] = []
    denominator_pass: dict[str, bool] = {}
    for denominator, selected in selected_burnins.items():
        memory.extend(
            preparation_memory(
                records,
                denominator=denominator,
                endpoint=selected,
                clock_id=f"selected-{selected:g}",
                config=config,
                stage=Stage.EVALUATION,
            )
        )
        memory.extend(
            preparation_memory(
                records,
                denominator=denominator,
                endpoint=2.0 * selected,
                clock_id=f"sentinel-{2.0 * selected:g}",
                config=config,
                stage=Stage.EVALUATION,
            )
        )
    for denominator in selected_burnins:
        denominator_pass[denominator.value] = bool(
            all(row.passed for row in intervals if row.denominator == denominator)
            and not any(row.material for row in memory if row.denominator == denominator)
        )
    return {
        "stage": Stage.EVALUATION.value,
        "selected_burnin": {
            denominator.value: value for denominator, value in selected_burnins.items()
        },
        "denominator_pass": denominator_pass,
        "intervals": [asdict(row) for row in intervals],
        "memory": [asdict(row) for row in memory],
    }


__all__ = [
    "analyze_candidate",
    "analyze_evaluation",
    "continue_trajectory_worker",
    "development_endpoints",
    "evaluation_endpoints",
    "observations_from_trajectory",
    "select_development_burnins",
    "trajectory_worker",
]
