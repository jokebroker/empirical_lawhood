"""Excluded source, propagation and ensemble qualification."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from statistics import NormalDist
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.stats import chisquare, kstest, poisson
from scipy.sparse import csr_matrix

from .basis import ComplexVector, DenseEigensystemPropagator, FixedNumberBasis, Propagator, SparseKrylovPropagator, apply_jump, build_hamiltonian, hermiticity_residual, lindblad_superoperator, normalize_state, particle_number_error, phase_gauge_infidelity, propagate_density, site_probabilities, sparse_matrix_fingerprint
from ..quantum_scientific_seeds import ScientificSeedCensus, validate_scientific_seed, validate_scientific_seed_census
from .contracts import Action, QuantumReceiverResponseConfig, stable_json_bytes
from .source import simulate_interval


@dataclass(frozen=True, slots=True)
class CoupledPathResult:
    scientific_seed: int
    action: Action
    event_count: int
    maximum_site_probability_error: float
    maximum_phase_infidelity: float
    maximum_primary_norm_error: float
    maximum_reference_norm_error: float
    maximum_particle_number_error: float
    passed: bool


@dataclass(frozen=True, slots=True)
class EnsembleQualification:
    scientific_seed_census_sha256: str
    initial_state: int
    draws: int
    horizon: float
    observable_ids: tuple[str, ...]
    trajectory_means: tuple[float, ...]
    exact_values: tuple[float, ...]
    simultaneous_halfwidths: tuple[float, ...]
    hilbert_schmidt_discrepancy: float
    hilbert_schmidt_envelope: float
    waiting_time_p_value: float
    total_count_p_value: float
    first_site_p_value: float
    passed: bool


def structural_audit(
    *,
    basis: FixedNumberBasis,
    hamiltonians: Mapping[Action, csr_matrix],
    gamma: float,
) -> dict[str, object]:
    if set(hamiltonians) != set(Action):
        raise ValueError("source structural audit requires all actions")
    states_ascending = tuple(sorted(basis.states)) == basis.states
    unique_states = len(set(basis.states)) == basis.dimension
    popcounts_valid = all(state.bit_count() == basis.particles for state in basis.states)
    hermiticity = {
        action.value: hermiticity_residual(hamiltonian)
        for action, hamiltonian in hamiltonians.items()
    }
    fingerprints = {
        action.value: sparse_matrix_fingerprint(hamiltonian)
        for action, hamiltonian in hamiltonians.items()
    }
    action_differences = {
        action.value: int((hamiltonians[action] - hamiltonians[Action.HOLD]).nnz)
        for action in (Action.MINUS, Action.PLUS)
    }
    expected_boundary_bonds = {
        (basis.l_sites // 2 - 1, basis.l_sites // 2),
        (0, basis.l_sites - 1),
    }
    action_difference_bonds = {}
    for action in (Action.MINUS, Action.PLUS):
        difference = (hamiltonians[action] - hamiltonians[Action.HOLD]).tocoo()
        bonds: set[tuple[int, int]] = set()
        for row, column, value in zip(
            difference.row,
            difference.col,
            difference.data,
            strict=True,
        ):
            if abs(value) == 0.0:
                continue
            changed = basis.states[int(row)] ^ basis.states[int(column)]
            sites = tuple(site for site in range(basis.l_sites) if (changed >> site) & 1)
            if len(sites) != 2:
                bonds.add((-1, -1))
            else:
                bonds.add(
                    (
                        min(sites[0], sites[1]),
                        max(sites[0], sites[1]),
                    )
                )
        action_difference_bonds[action.value] = sorted(bonds)
    boundary_bonds_valid = all(
        set(action_difference_bonds[action.value]) == expected_boundary_bonds
        for action in (Action.MINUS, Action.PLUS)
    )
    occupations = basis.occupations
    projectors_idempotent = bool(np.all((occupations * occupations) == occupations))
    total_hazard_error = float(
        np.max(np.abs(gamma * occupations.sum(axis=1) - gamma * basis.particles))
    )
    passed = (
        states_ascending
        and unique_states
        and popcounts_valid
        and max(hermiticity.values()) == 0.0
        and all(value > 0 for value in action_differences.values())
        and boundary_bonds_valid
        and projectors_idempotent
        and total_hazard_error == 0.0
    )
    return {
        "basis_dimension": basis.dimension,
        "basis_fingerprint": basis.fingerprint,
        "states_ascending": states_ascending,
        "unique_states": unique_states,
        "popcounts_valid": popcounts_valid,
        "hermiticity_residuals": hermiticity,
        "hamiltonian_fingerprints": fingerprints,
        "action_difference_nnz": action_differences,
        "action_difference_bonds": action_difference_bonds,
        "boundary_bonds_valid": boundary_bonds_valid,
        "jump_projectors_idempotent": projectors_idempotent,
        "total_hazard_error": total_hazard_error,
        "passed": passed,
    }


def coupled_path(
    *,
    basis: FixedNumberBasis,
    primary: Propagator,
    reference: Propagator,
    initial_state: ComplexVector,
    action: Action,
    gamma: float,
    horizon: float,
    seed: int,
    config: QuantumReceiverResponseConfig,
) -> CoupledPathResult:
    """Compare propagators while applying the sparse view's canonical jump."""

    validate_scientific_seed(seed, bits=64)
    rng = np.random.Generator(np.random.Philox(seed))
    primary_state, primary_norm = normalize_state(initial_state.copy())
    reference_state, reference_norm = normalize_state(initial_state.copy())
    time_value = 0.0
    event_count = 0
    max_probability_error = 0.0
    max_infidelity = 0.0
    max_particle_error = 0.0
    hazard = gamma * basis.particles
    while True:
        waiting = float(rng.exponential(1.0 / hazard))
        event_time = time_value + waiting
        duration = min(event_time, horizon) - time_value
        primary_state, error = normalize_state(primary.propagate(primary_state, duration))
        primary_norm = max(primary_norm, error)
        reference_state, error = normalize_state(reference.propagate(reference_state, duration))
        reference_norm = max(reference_norm, error)
        max_infidelity = max(
            max_infidelity,
            phase_gauge_infidelity(primary_state, reference_state),
        )
        max_particle_error = max(
            max_particle_error,
            particle_number_error(basis, primary_state),
            particle_number_error(basis, reference_state),
        )
        if event_time >= horizon:
            break
        primary_probabilities = site_probabilities(basis, primary_state)
        reference_probabilities = site_probabilities(basis, reference_state)
        max_probability_error = max(
            max_probability_error,
            float(np.max(np.abs(primary_probabilities - reference_probabilities))),
        )
        site = int(rng.choice(basis.l_sites, p=primary_probabilities))
        if min(primary_probabilities[site], reference_probabilities[site]) <= 1e-15:
            raise RuntimeError("coupled source jump selected a zero-probability site")
        primary_state, error = apply_jump(basis, primary_state, site)
        primary_norm = max(primary_norm, error)
        reference_state, error = apply_jump(basis, reference_state, site)
        reference_norm = max(reference_norm, error)
        max_infidelity = max(
            max_infidelity,
            phase_gauge_infidelity(primary_state, reference_state),
        )
        event_count += 1
        time_value = event_time
    passed = (
        max_probability_error <= float(config.source_conformance_site_probability_tolerance)
        and max_infidelity <= float(config.source_conformance_phase_infidelity_tolerance)
        and primary_norm <= float(config.source_conformance_norm_tolerance)
        and reference_norm <= float(config.source_conformance_norm_tolerance)
        and max_particle_error <= float(config.source_conformance_particle_tolerance)
    )
    return CoupledPathResult(
        scientific_seed=seed,
        action=action,
        event_count=event_count,
        maximum_site_probability_error=max_probability_error,
        maximum_phase_infidelity=max_infidelity,
        maximum_primary_norm_error=primary_norm,
        maximum_reference_norm_error=reference_norm,
        maximum_particle_number_error=max_particle_error,
        passed=passed,
    )


def _observables(
    *,
    basis: FixedNumberBasis,
    hamiltonian: NDArray[np.complex128],
) -> tuple[tuple[str, ...], NDArray[np.complex128]]:
    rows: list[NDArray[np.complex128]] = []
    names: list[str] = []
    for site in range(basis.l_sites):
        rows.append(np.diag(basis.occupations[:, site]).astype(np.complex128))
        names.append(f"occupation_{site}")
    for site in range(basis.l_sites):
        neighbor = (site + 1) % basis.l_sites
        rows.append(
            np.diag(basis.occupations[:, site] * basis.occupations[:, neighbor]).astype(
                np.complex128
            )
        )
        names.append(f"density_correlation_{site}_{neighbor}")
    rows.append(hamiltonian)
    names.append("energy")
    angles = 2.0 * math.pi * np.arange(basis.l_sites) / basis.l_sites
    for label, weights in (("fourier_cos_1", np.cos(angles)), ("fourier_sin_1", np.sin(angles))):
        diagonal = basis.occupations @ weights
        rows.append(np.diag(diagonal).astype(np.complex128))
        names.append(label)
    return tuple(names), np.asarray(rows, dtype=np.complex128)


def _poisson_count_p_value(counts: NDArray[np.int64], mean: float) -> float:
    maximum = max(int(np.max(counts)), int(math.ceil(mean + 6.0 * math.sqrt(mean))))
    observed = np.bincount(counts, minlength=maximum + 1).astype(np.float64)
    probabilities = poisson.pmf(np.arange(maximum + 1), mean)
    probabilities[-1] += 1.0 - float(probabilities.sum())
    expected = probabilities * len(counts)
    while len(expected) > 2 and expected[-1] < 5.0:
        expected[-2] += expected[-1]
        observed[-2] += observed[-1]
        expected = expected[:-1]
        observed = observed[:-1]
    expected *= observed.sum() / expected.sum()
    return float(chisquare(observed, expected).pvalue)


def ensemble_lindblad_qualification(
    *,
    basis: FixedNumberBasis,
    initial_state: int,
    config: QuantumReceiverResponseConfig,
    scientific_seeds: ScientificSeedCensus | None = None,
    scientific_waiting_seed: int | None = None,
) -> EnsembleQualification:
    """Compare a 4,096-path ensemble with the exact finite Lindblad solution."""

    draws = config.source_conformance_ensemble_draws
    if type(draws) is not int or draws < 8:
        raise ValueError("ensemble qualification requires at least eight declared draws")
    validate_scientific_seed_census(scientific_seeds, indices=tuple(range(draws)), bits=64)
    validate_scientific_seed(scientific_waiting_seed, bits=64)
    if scientific_waiting_seed in {seed for _, seed in scientific_seeds}:
        raise ValueError("waiting-time stream duplicates an ensemble draw allocation")
    scientific_digest = sha256(stable_json_bytes({
        "ensemble": scientific_seeds,
        "waiting": scientific_waiting_seed,
    })).hexdigest()
    hamiltonian = build_hamiltonian(basis)
    propagator = SparseKrylovPropagator(hamiltonian)
    horizon = float(config.source_conformance_dense_horizon)
    terminal_states = np.empty(
        (draws, basis.dimension),
        dtype=np.complex128,
    )
    event_counts = np.empty(draws, dtype=np.int64)
    first_sites: list[int] = []
    initial = basis.state_vector(initial_state)
    for draw_index in range(draws):
        rng = np.random.Generator(
            np.random.Philox(scientific_seeds[draw_index][1])
        )
        result = simulate_interval(
            basis=basis,
            propagator=propagator,
            initial_state=initial,
            start_time=0.0,
            duration=horizon,
            gamma=1.0,
            rng=rng,
        )
        terminal_states[draw_index] = result.state
        event_counts[draw_index] = len(result.events)
        if result.events:
            first_sites.append(result.events[0].site)

    names, observables = _observables(
        basis=basis,
        hamiltonian=np.asarray(hamiltonian.toarray(), dtype=np.complex128),
    )
    trajectory_values = np.real(
        np.einsum(
            "di,oij,dj->do",
            terminal_states.conj(),
            observables,
            terminal_states,
            optimize=True,
        )
    )
    means = trajectory_values.mean(axis=0)
    standard_errors = trajectory_values.std(axis=0, ddof=1) / math.sqrt(draws)
    critical = NormalDist().inv_cdf(1.0 - float(config.source_conformance_family_alpha) / (2.0 * len(names)))
    halfwidths = critical * standard_errors + 1e-12

    generator = lindblad_superoperator(basis, hamiltonian, gamma=1.0)
    exact_density = propagate_density(
        generator,
        np.outer(initial, initial.conj()),
        horizon,
    )
    exact_values = np.real(np.einsum("oij,ji->o", observables, exact_density, optimize=True))
    observable_pass = bool(np.all(np.abs(means - exact_values) <= halfwidths))

    ensemble_density = (
        np.einsum(
            "di,dj->ij",
            terminal_states,
            terminal_states.conj(),
            optimize=True,
        )
        / draws
    )
    discrepancy = float(np.linalg.norm(ensemble_density - exact_density))
    block_size = draws // 8
    block_discrepancies = []
    for block_index in range(8):
        local = terminal_states[block_index * block_size : (block_index + 1) * block_size]
        local_density = np.einsum(
            "di,dj->ij",
            local,
            local.conj(),
            optimize=True,
        ) / len(local)
        block_discrepancies.append(float(np.linalg.norm(local_density - ensemble_density)))
    envelope = float(config.source_conformance_hs_split_multiplier) * float(
        np.quantile(block_discrepancies, 0.99, method="higher")
    )

    hazard = basis.particles
    waiting_rng = np.random.Generator(
        np.random.Philox(scientific_waiting_seed)
    )
    waiting_times = waiting_rng.exponential(1.0 / hazard, size=draws)
    waiting_p = float(kstest(waiting_times, "expon", args=(0.0, 1.0 / hazard)).pvalue)
    count_p = _poisson_count_p_value(event_counts, hazard * horizon)
    initial_probabilities = site_probabilities(basis, initial)
    active_sites = np.flatnonzero(initial_probabilities > 0.0)
    observed_sites = np.asarray(
        [sum(site == value for site in first_sites) for value in active_sites],
        dtype=np.float64,
    )
    expected_sites = (
        initial_probabilities[active_sites]
        / initial_probabilities[active_sites].sum()
        * observed_sites.sum()
    )
    first_site_p = float(chisquare(observed_sites, expected_sites).pvalue)
    process_alpha = float(config.source_conformance_family_alpha) / (4.0 * 3.0)
    passed = (
        observable_pass
        and discrepancy <= envelope
        and waiting_p >= process_alpha
        and count_p >= process_alpha
        and first_site_p >= process_alpha
    )
    return EnsembleQualification(
        scientific_seed_census_sha256=scientific_digest,
        initial_state=initial_state,
        draws=draws,
        horizon=horizon,
        observable_ids=names,
        trajectory_means=tuple(float(value) for value in means),
        exact_values=tuple(float(value) for value in exact_values),
        simultaneous_halfwidths=tuple(float(value) for value in halfwidths),
        hilbert_schmidt_discrepancy=discrepancy,
        hilbert_schmidt_envelope=envelope,
        waiting_time_p_value=waiting_p,
        total_count_p_value=count_p,
        first_site_p_value=first_site_p,
        passed=passed,
    )


def free_running_dense_sparse(
    *,
    basis: FixedNumberBasis,
    initial_state: int,
    gamma: float,
    horizon: float,
    draws: int,
    scientific_seeds: ScientificSeedCensus | None = None,
) -> dict[str, object]:
    if type(draws) is not int or draws <= 0:
        raise ValueError("free-running comparison requires positive declared draws")
    validate_scientific_seed_census(scientific_seeds, indices=tuple(range(draws)), bits=64)
    hamiltonian = build_hamiltonian(basis)
    propagators: tuple[Propagator, Propagator] = (
        SparseKrylovPropagator(hamiltonian),
        DenseEigensystemPropagator(hamiltonian),
    )
    counts = np.empty((2, draws, basis.l_sites), dtype=np.float64)
    initial = basis.state_vector(initial_state)
    for method_index, propagator in enumerate(propagators):
        for draw_index in range(draws):
            rng = np.random.Generator(
                np.random.Philox(scientific_seeds[draw_index][1])
            )
            result = simulate_interval(
                basis=basis,
                propagator=propagator,
                initial_state=initial,
                start_time=0.0,
                duration=horizon,
                gamma=gamma,
                rng=rng,
            )
            counts[method_index, draw_index] = np.bincount(
                [event.site for event in result.events],
                minlength=basis.l_sites,
            )
    mean_difference = float(np.max(np.abs(counts[0].mean(axis=0) - counts[1].mean(axis=0))))
    exact_path_fraction = float(np.mean(np.all(counts[0] == counts[1], axis=1)))
    return {
        "draws": draws,
        "scientific_seed_census_sha256": sha256(stable_json_bytes(scientific_seeds)).hexdigest(),
        "maximum_mean_site_count_difference": mean_difference,
        "exact_path_receiver_fraction": exact_path_fraction,
        "passed": mean_difference <= 1e-10,
    }


__all__ = [
    "CoupledPathResult",
    "EnsembleQualification",
    "coupled_path",
    "ensemble_lindblad_qualification",
    "free_running_dense_sparse",
    "structural_audit",
]
