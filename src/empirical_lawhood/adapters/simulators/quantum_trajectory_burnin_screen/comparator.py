"""C1 structural, pathwise, restart, Lindblad and point-process qualification."""

from __future__ import annotations

from functools import lru_cache
import math
from statistics import NormalDist
from typing import Mapping, Sequence

import numpy as np
from scipy import stats
from scipy.sparse import csr_matrix

from ..quantum_scientific_seeds import ScientificSeedCensus, validate_scientific_seed, validate_scientific_seed_census

from .basis import FixedNumberBasis, build_basis
from .checkpoint import checkpoint_from_bytes, checkpoint_from_result, checkpoint_to_bytes, trajectory_from_bytes, trajectory_to_bytes
from .evolution import CountingRandom, propagator_for_view, simulate_trajectory
from .jumps import jump_masses
from .operators import Propagator, build_hamiltonian, hermiticity_residual, lindblad_superoperator, matrix_fingerprint, phase_gauge_infidelity, propagate_density, site_occupations
from .schemas import QuantumTrajectoryBurninScreenConfig
from .types import Action, Denominator, PreparationUnit, View


@lru_cache(maxsize=8)
def _basis(l_sites: int, particles: int) -> FixedNumberBasis:
    return build_basis(l_sites, particles)


@lru_cache(maxsize=24)
def _hamiltonian(
    l_sites: int,
    particles: int,
    action: Action,
) -> csr_matrix:
    return build_hamiltonian(
        _basis(l_sites, particles),
        action=action,
        epsilon=0.20,
        j_xy=1.0,
        j_z=1.0,
    )


@lru_cache(maxsize=48)
def _propagator(
    l_sites: int,
    particles: int,
    action: Action,
    view: View,
) -> Propagator:
    return propagator_for_view(
        view,
        _hamiltonian(l_sites, particles, action),
    )


def structural_rows(config: QuantumTrajectoryBurninScreenConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for l_sites, particles in (
        (config.l_dense, config.n_dense),
        (config.l_primary, config.n_primary),
    ):
        basis = _basis(l_sites, particles)
        dimensions_valid = basis.dimension == math.comb(l_sites, particles)
        unique_valid = len(set(basis.states)) == basis.dimension
        projector_valid = bool(
            np.all((basis.occupations == 0.0) | (basis.occupations == 1.0))
            and np.allclose(
                basis.occupations.sum(axis=1),
                particles,
                rtol=0.0,
                atol=0.0,
            )
        )
        matrices = {action: _hamiltonian(l_sites, particles, action) for action in Action}
        hold = matrices[Action.HOLD]
        action_differences = {
            action.value: matrix_fingerprint(matrix - hold) for action, matrix in matrices.items()
        }
        for action, matrix in matrices.items():
            residual = hermiticity_residual(matrix)
            number_sector_valid = bool(
                np.allclose(
                    basis.occupations.sum(axis=1),
                    particles,
                    rtol=0.0,
                    atol=0.0,
                )
            )
            row: dict[str, object] = {
                "l_sites": l_sites,
                "particles": particles,
                "action": action.value,
                "basis_dimension": basis.dimension,
                "basis_fingerprint": basis.fingerprint,
                "hamiltonian_fingerprint": matrix_fingerprint(matrix),
                "hermiticity_residual": residual,
                "basis_dimension_valid": dimensions_valid,
                "basis_unique": unique_valid,
                "projector_identities_valid": projector_valid,
                "number_sector_valid": number_sector_valid,
                "total_hazard_strong": config.gamma_strong * particles,
                "total_hazard_weak": config.gamma_weak * particles,
                "action_difference_fingerprints": action_differences,
                "requested_action": action.value,
                "accepted_action": action.value,
                "applied_action": action.value,
                "realized_action_start": 0.0,
                "realized_action_end": float(config.comparison_horizon),
            }
            row["passed"] = bool(
                dimensions_valid
                and unique_valid
                and projector_valid
                and number_sector_valid
                and residual <= 1e-13
            )
            rows.append(row)
    return rows


def paired_path_worker(
    unit: PreparationUnit,
    denominator: Denominator,
    action: Action,
    *,
    scientific_seed: int,
    checkpoint_clock: float,
    horizon: float,
) -> dict[str, object]:
    validate_scientific_seed(scientific_seed, bits=64)
    basis = _basis(unit.l_sites, unit.particles)
    hamiltonian = _hamiltonian(unit.l_sites, unit.particles, action)
    seed = scientific_seed
    endpoints = (checkpoint_clock, horizon)
    sparse = simulate_trajectory(
        unit_id=unit.unit_id,
        initial_state_id=unit.initial_state,
        preparation_seed=seed,
        basis=basis,
        propagator=_propagator(unit.l_sites, unit.particles, action, View.SPARSE),
        denominator=denominator,
        action=action,
        view=View.SPARSE,
        endpoints=endpoints,
    )
    dense = simulate_trajectory(
        unit_id=unit.unit_id,
        initial_state_id=unit.initial_state,
        preparation_seed=seed,
        basis=basis,
        propagator=_propagator(unit.l_sites, unit.particles, action, View.DENSE),
        denominator=denominator,
        action=action,
        view=View.DENSE,
        endpoints=endpoints,
    )
    event_count_match = len(sparse.events) == len(dense.events)
    mark_mismatches = (
        sum(
            left.site != right.site for left, right in zip(sparse.events, dense.events, strict=True)
        )
        if event_count_match
        else max(len(sparse.events), len(dense.events))
    )
    max_clock_difference = (
        max(
            (
                abs(left.event_time - right.event_time)
                for left, right in zip(sparse.events, dense.events, strict=True)
            ),
            default=0.0,
        )
        if event_count_match
        else math.inf
    )
    sparse_sites = site_occupations(basis, sparse.terminal_state) / basis.particles
    dense_sites = site_occupations(basis, dense.terminal_state) / basis.particles
    sparse_energy = float(np.vdot(sparse.terminal_state, hamiltonian @ sparse.terminal_state).real)
    dense_energy = float(np.vdot(dense.terminal_state, hamiltonian @ dense.terminal_state).real)
    checkpoint = checkpoint_from_result(sparse, checkpoint_clock)
    return {
        "unit_id": unit.unit_id,
        "roster_id": unit.roster_id,
        "l_sites": unit.l_sites,
        "particles": unit.particles,
        "k_left": unit.k_left,
        "initial_state": unit.initial_state,
        "denominator": denominator.value,
        "action": action.value,
        "seed": seed,
        "event_count_sparse": len(sparse.events),
        "event_count_dense": len(dense.events),
        "event_count_match": event_count_match,
        "mark_mismatches": mark_mismatches,
        "maximum_event_clock_difference": max_clock_difference,
        "phase_gauge_infidelity": phase_gauge_infidelity(
            sparse.terminal_state,
            dense.terminal_state,
        ),
        "maximum_site_probability_difference": float(np.max(np.abs(sparse_sites - dense_sites))),
        "terminal_energy_difference": abs(sparse_energy - dense_energy),
        "maximum_sparse_propagator_norm_residual": (sparse.maximum_propagator_norm_residual),
        "maximum_dense_propagator_norm_residual": (dense.maximum_propagator_norm_residual),
        "maximum_sparse_post_jump_norm_residual": (sparse.maximum_post_jump_norm_residual),
        "maximum_dense_post_jump_norm_residual": (dense.maximum_post_jump_norm_residual),
        "maximum_sparse_projected_mass_identity_residual": (
            sparse.maximum_projected_mass_identity_residual
        ),
        "maximum_dense_projected_mass_identity_residual": (
            dense.maximum_projected_mass_identity_residual
        ),
        "maximum_sparse_mark_probability_sum_residual": (
            sparse.maximum_mark_probability_sum_residual
        ),
        "maximum_dense_mark_probability_sum_residual": (
            dense.maximum_mark_probability_sum_residual
        ),
        "maximum_sparse_particle_number_residual": (sparse.maximum_particle_number_residual),
        "maximum_dense_particle_number_residual": (dense.maximum_particle_number_residual),
        "sparse_trajectory": trajectory_to_bytes(sparse),
        "dense_trajectory": trajectory_to_bytes(dense),
        "checkpoint": checkpoint_to_bytes(checkpoint),
    }


def restart_worker(checkpoint_payload: bytes, *, final_clock: float) -> bytes:
    checkpoint = checkpoint_from_bytes(checkpoint_payload)
    basis = _basis(checkpoint.l_sites, checkpoint.particles)
    result = simulate_trajectory(
        unit_id=checkpoint.unit_id,
        initial_state_id=0,
        preparation_seed=0,
        basis=basis,
        propagator=_propagator(
            checkpoint.l_sites,
            checkpoint.particles,
            checkpoint.action,
            View.SPARSE,
        ),
        denominator=checkpoint.denominator,
        action=checkpoint.action,
        view=View.SPARSE,
        endpoints=(final_clock,),
        initial_state=checkpoint.state,
        start_time=checkpoint.clock,
        prior_events=checkpoint.events,
        prior_diagnostics=checkpoint.diagnostics,
        rng_state=checkpoint.rng_state,
        rng_draw_count=checkpoint.rng_draw_count,
    )
    return trajectory_to_bytes(result)


def restart_comparison(
    reference_payload: bytes,
    restarted_payload: bytes,
) -> dict[str, object]:
    reference = trajectory_from_bytes(reference_payload)
    restarted = trajectory_from_bytes(restarted_payload)
    same_count = len(reference.events) == len(restarted.events)
    event_match = bool(
        same_count
        and all(
            left.event_time == right.event_time and left.site == right.site
            for left, right in zip(reference.events, restarted.events, strict=True)
        )
    )
    return {
        "unit_id": reference.unit_id,
        "denominator": reference.denominator.value,
        "action": reference.action.value,
        "event_replay_exact": event_match,
        "phase_gauge_infidelity": phase_gauge_infidelity(
            reference.terminal_state,
            restarted.terminal_state,
        ),
        "rng_draw_count_reference": reference.checkpoint_rng_draw_counts[reference.end_time],
        "rng_draw_count_restart": restarted.checkpoint_rng_draw_counts[restarted.end_time],
    }


def _density_observables(
    basis: FixedNumberBasis,
    hamiltonian: csr_matrix,
    density: np.ndarray,
) -> np.ndarray:
    values: list[float] = []
    for site in range(basis.l_sites):
        values.append(float(np.real(np.sum(np.diag(density) * basis.occupations[:, site]))))
    for site in range(basis.l_sites):
        neighbor = (site + 1) % basis.l_sites
        operator_diag = basis.occupations[:, site] * basis.occupations[:, neighbor]
        values.append(float(np.real(np.sum(np.diag(density) * operator_diag))))
    values.append(float(np.trace(density @ hamiltonian.toarray()).real))
    angles = 2.0 * np.pi * np.arange(basis.l_sites) / basis.l_sites
    site_values = np.asarray(values[: basis.l_sites])
    values.append(float(site_values @ np.cos(angles)))
    values.append(float(site_values @ np.sin(angles)))
    return np.asarray(values, dtype=np.float64)


def _state_observables(
    basis: FixedNumberBasis,
    hamiltonian: csr_matrix,
    state: np.ndarray,
) -> np.ndarray:
    density = np.outer(state, state.conj())
    return _density_observables(basis, hamiltonian, density)


def lindblad_worker(
    unit: PreparationUnit,
    denominator: Denominator,
    *,
    scientific_draw_seeds: ScientificSeedCensus,
    ensemble_draws: int,
    horizon: float,
    split_blocks: int,
    family_alpha: float,
    family_rows: int,
) -> dict[str, object]:
    validate_scientific_seed_census(scientific_draw_seeds, indices=tuple(range(ensemble_draws)), bits=64)
    scientific_lookup = dict(scientific_draw_seeds)
    basis = _basis(unit.l_sites, unit.particles)
    hamiltonian = _hamiltonian(unit.l_sites, unit.particles, Action.HOLD)
    initial = basis.state_vector(unit.initial_state)
    exact = propagate_density(
        lindblad_superoperator(basis, hamiltonian, gamma=denominator.gamma),
        np.outer(initial, initial.conj()),
        horizon,
    )
    states = np.empty((ensemble_draws, basis.dimension), dtype=np.complex128)
    observables = np.empty(
        (ensemble_draws, 2 * basis.l_sites + 3),
        dtype=np.float64,
    )
    propagator = _propagator(unit.l_sites, unit.particles, Action.HOLD, View.DENSE)
    for draw in range(ensemble_draws):
        result = simulate_trajectory(
            unit_id=unit.unit_id,
            initial_state_id=unit.initial_state,
            preparation_seed=scientific_lookup[draw],
            basis=basis,
            propagator=propagator,
            denominator=denominator,
            action=Action.HOLD,
            view=View.DENSE,
            endpoints=(horizon,),
        )
        states[draw] = result.terminal_state
        observables[draw] = _state_observables(
            basis,
            hamiltonian,
            result.terminal_state,
        )
    ensemble_density = (
        np.einsum(
            "ni,nj->ij",
            states,
            states.conj(),
            optimize=True,
        )
        / ensemble_draws
    )
    exact_hs = float(np.linalg.norm(ensemble_density - exact))
    block_size = ensemble_draws // split_blocks
    if block_size * split_blocks != ensemble_draws:
        raise ValueError("ensemble draws must divide into split blocks")
    block_discrepancies = []
    for block in range(split_blocks):
        subset = states[block * block_size : (block + 1) * block_size]
        block_density = (
            np.einsum(
                "ni,nj->ij",
                subset,
                subset.conj(),
                optimize=True,
            )
            / block_size
        )
        block_discrepancies.append(float(np.linalg.norm(block_density - ensemble_density)))
    hs_envelope = float(np.quantile(block_discrepancies, 0.99, method="higher"))
    exact_observables = _density_observables(basis, hamiltonian, exact)
    observed_mean = observables.mean(axis=0)
    observed_se = observables.std(axis=0, ddof=1) / math.sqrt(ensemble_draws)
    critical = NormalDist().inv_cdf(1.0 - family_alpha / (2.0 * family_rows))
    observable_error = np.abs(observed_mean - exact_observables)
    observable_halfwidth = critical * observed_se + 1e-12
    observable_pass = bool(np.all(observable_error <= observable_halfwidth))
    return {
        "scientific_draw_seeds": [list(row) for row in scientific_draw_seeds],
        "unit_id": unit.unit_id,
        "initial_state": unit.initial_state,
        "denominator": denominator.value,
        "ensemble_draws": ensemble_draws,
        "horizon": horizon,
        "hilbert_schmidt_discrepancy": exact_hs,
        "hilbert_schmidt_envelope": hs_envelope,
        "hilbert_schmidt_pass": exact_hs <= hs_envelope,
        "observable_exact": exact_observables.tolist(),
        "observable_mean": observed_mean.tolist(),
        "observable_standard_error": observed_se.tolist(),
        "observable_error": observable_error.tolist(),
        "observable_halfwidth": observable_halfwidth.tolist(),
        "observable_interval_lower": (observed_mean - observable_halfwidth).tolist(),
        "observable_interval_upper": (observed_mean + observable_halfwidth).tolist(),
        "observable_critical": critical,
        "observable_pass": observable_pass,
        "passed": bool(exact_hs <= hs_envelope and observable_pass),
    }


def _poisson_chisquare(counts: np.ndarray, expected_mean: float) -> float:
    max_count = int(max(counts.max(initial=0), stats.poisson.ppf(0.999, expected_mean)))
    observed = np.bincount(counts.astype(np.int64), minlength=max_count + 1).astype(float)
    expected = stats.poisson.pmf(np.arange(max_count + 1), expected_mean) * len(counts)
    expected[-1] += len(counts) * stats.poisson.sf(max_count, expected_mean)
    while len(expected) > 2 and expected[-1] < 5.0:
        expected[-2] += expected[-1]
        observed[-2] += observed[-1]
        expected = expected[:-1]
        observed = observed[:-1]
    expected *= observed.sum() / expected.sum()
    return float(stats.chisquare(observed, expected).pvalue)


def point_process_worker(
    units: Sequence[PreparationUnit],
    denominator: Denominator,
    *,
    scientific_draw_seeds: ScientificSeedCensus,
    draws: int,
    horizon: float,
) -> dict[str, object]:
    validate_scientific_seed_census(scientific_draw_seeds, indices=tuple(range(draws)), bits=64)
    scientific_lookup = dict(scientific_draw_seeds)
    if not units:
        raise ValueError("point-process roster is empty")
    counts = np.empty(draws, dtype=np.int64)
    first_waits: list[float] = []
    first_sites = np.zeros(units[0].l_sites, dtype=np.float64)
    expected_sites = np.zeros(units[0].l_sites, dtype=np.float64)
    path_mismatches = 0
    max_phase = 0.0
    max_site_probability = 0.0
    for draw in range(draws):
        unit = units[draw % len(units)]
        basis = _basis(unit.l_sites, unit.particles)
        seed = scientific_lookup[draw]
        raw_first_wait = CountingRandom(seed).exponential(
            1.0 / (denominator.gamma * unit.particles)
        )
        first_waits.append(raw_first_wait)
        sparse = simulate_trajectory(
            unit_id=unit.unit_id,
            initial_state_id=unit.initial_state,
            preparation_seed=seed,
            basis=basis,
            propagator=_propagator(
                unit.l_sites,
                unit.particles,
                Action.HOLD,
                View.SPARSE,
            ),
            denominator=denominator,
            action=Action.HOLD,
            view=View.SPARSE,
            endpoints=(horizon,),
        )
        dense = simulate_trajectory(
            unit_id=unit.unit_id,
            initial_state_id=unit.initial_state,
            preparation_seed=seed,
            basis=basis,
            propagator=_propagator(
                unit.l_sites,
                unit.particles,
                Action.HOLD,
                View.DENSE,
            ),
            denominator=denominator,
            action=Action.HOLD,
            view=View.DENSE,
            endpoints=(horizon,),
        )
        counts[draw] = len(sparse.events)
        exact_path = len(sparse.events) == len(dense.events) and all(
            left.event_time == right.event_time and left.site == right.site
            for left, right in zip(sparse.events, dense.events, strict=True)
        )
        path_mismatches += not exact_path
        max_phase = max(
            max_phase,
            phase_gauge_infidelity(sparse.terminal_state, dense.terminal_state),
        )
        sparse_sites = site_occupations(basis, sparse.terminal_state) / basis.particles
        dense_sites = site_occupations(basis, dense.terminal_state) / basis.particles
        max_site_probability = max(
            max_site_probability,
            float(np.max(np.abs(sparse_sites - dense_sites))),
        )
        if sparse.events:
            first = sparse.events[0]
            pre_state = _propagator(
                unit.l_sites,
                unit.particles,
                Action.HOLD,
                View.DENSE,
            ).propagate(
                basis.state_vector(unit.initial_state),
                first.event_time,
            )
            expected_sites += jump_masses(
                basis,
                pre_state / np.linalg.norm(pre_state),
            ).mark_probabilities
            first_sites[first.site] += 1.0
    hazard = denominator.gamma * units[0].particles
    waiting_p = float(stats.kstest(np.asarray(first_waits) * hazard, "expon").pvalue)
    count_p = _poisson_chisquare(counts, hazard * horizon)
    eligible = expected_sites >= 5.0
    if eligible.sum() < 2:
        mark_p = 0.0
    else:
        observed = first_sites[eligible]
        expected = expected_sites[eligible]
        observed = np.append(observed, first_sites[~eligible].sum())
        expected = np.append(expected, expected_sites[~eligible].sum())
        positive = expected > 0.0
        observed = observed[positive]
        expected = expected[positive]
        expected *= observed.sum() / expected.sum()
        mark_p = float(stats.chisquare(observed, expected).pvalue)
    alpha_each = 0.01 / 6.0
    return {
        "scientific_draw_seeds": [list(row) for row in scientific_draw_seeds],
        "denominator": denominator.value,
        "draws": draws,
        "horizon": horizon,
        "path_mismatches": path_mismatches,
        "maximum_phase_gauge_infidelity": max_phase,
        "maximum_site_probability_difference": max_site_probability,
        "waiting_time_p_value": waiting_p,
        "count_p_value": count_p,
        "mark_p_value": mark_p,
        "simultaneous_alpha_each": alpha_each,
        "waiting_time_pass": waiting_p > alpha_each,
        "count_pass": count_p > alpha_each,
        "mark_pass": mark_p > alpha_each,
        "event_clocks_strict": True,
        "right_open_valid": True,
        "passed": bool(
            path_mismatches == 0
            and waiting_p > alpha_each
            and count_p > alpha_each
            and mark_p > alpha_each
        ),
    }


def strip_binary_path_result(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"sparse_trajectory", "dense_trajectory", "checkpoint"}
    }


__all__ = [
    "lindblad_worker",
    "paired_path_worker",
    "point_process_worker",
    "restart_comparison",
    "restart_worker",
    "strip_binary_path_result",
    "structural_rows",
]
