"""Independent deterministic Lindblad reference routes."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import time
from typing import Callable, Mapping

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix

from .basis import FixedNumberBasis, build_basis
from .bounded_intervals import density_observables, observable_supports, support_rows
from .fixture_seed_commitments import generator_fixture_seed
from .operators import build_hamiltonian, lindblad_superoperator, matrix_fingerprint, propagate_density
from .schemas import IntegratorSpec, QuantumTrajectoryReferenceValidationConfig
from .types import Action, Denominator, PreparationUnit, Validity


def _density_sha256(density: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(density, dtype="<c16"))
    return sha256(canonical.tobytes()).hexdigest()


def _direct_rhs_factory(
    basis: FixedNumberBasis,
    hamiltonian: csr_matrix,
    gamma: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Assemble the matrix RHS without the Kronecker superoperator."""

    dense_hamiltonian = np.asarray(hamiltonian.toarray(), dtype=np.complex128)
    dephasing_coefficient = gamma * (basis.occupations @ basis.occupations.T - basis.particles)

    def rhs(density: np.ndarray) -> np.ndarray:
        matrix = np.asarray(density, dtype=np.complex128)
        return (
            -1j * (dense_hamiltonian @ matrix - matrix @ dense_hamiltonian)
            + dephasing_coefficient * matrix
        )

    return rhs


def _stack(density: np.ndarray) -> np.ndarray:
    flat = np.asarray(density, dtype=np.complex128).reshape(-1)
    return np.concatenate((flat.real, flat.imag))


def _unstack(vector: np.ndarray, dimension: int) -> np.ndarray:
    half = dimension * dimension
    return np.asarray(
        vector[:half] + 1j * vector[half:],
        dtype=np.complex128,
    ).reshape((dimension, dimension))


def _integrate_direct(
    rhs: Callable[[np.ndarray], np.ndarray],
    initial_density: np.ndarray,
    horizon: float,
    spec: IntegratorSpec,
) -> tuple[np.ndarray, dict[str, object]]:
    dimension = initial_density.shape[0]

    def stacked_rhs(_time: float, value: np.ndarray) -> np.ndarray:
        return _stack(rhs(_unstack(value, dimension)))

    started = time.perf_counter()
    solution = solve_ivp(
        stacked_rhs,
        (0.0, horizon),
        _stack(initial_density),
        method="DOP853",
        rtol=float(spec.rtol),
        atol=float(spec.atol),
        max_step=float(spec.max_step),
        t_eval=(horizon,),
    )
    elapsed = time.perf_counter() - started
    if not solution.success or solution.y.shape[1] != 1:
        raise RuntimeError(f"DOP853 reference failed: {solution.message}")
    return _unstack(solution.y[:, -1], dimension), {
        "solver": "scipy.integrate.solve_ivp/DOP853",
        "rtol": str(spec.rtol),
        "atol": str(spec.atol),
        "max_step": str(spec.max_step),
        "function_evaluations": int(solution.nfev),
        "jacobian_evaluations": int(solution.njev),
        "lu_decompositions": int(solution.nlu),
        "solver_message": str(solution.message),
        "wall_seconds": elapsed,
    }


def density_diagnostics(
    basis: FixedNumberBasis,
    density: np.ndarray,
) -> dict[str, float]:
    array = np.asarray(density, dtype=np.complex128)
    trace = complex(np.trace(array))
    hermitian = 0.5 * (array + array.conj().T)
    eigenvalues = np.linalg.eigvalsh(hermitian)
    number_diagonal = basis.occupations.sum(axis=1)
    particle_expectation = complex(np.sum(np.diag(array) * number_diagonal))
    return {
        "trace_residual": abs(trace - 1.0),
        "hermiticity_residual": float(np.max(np.abs(array - array.conj().T))),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "particle_number_residual": abs(particle_expectation - basis.particles),
    }


def generator_conformance(
    basis: FixedNumberBasis,
    hamiltonian: csr_matrix,
    denominator: Denominator,
    *,
    scientific_seed: int | None = None,
) -> dict[str, object]:
    scientific_seed = generator_fixture_seed(
        basis.l_sites, basis.particles, denominator.gamma, scientific_seed,
    )
    generator = lindblad_superoperator(
        basis,
        hamiltonian,
        gamma=denominator.gamma,
    )
    direct_rhs = _direct_rhs_factory(basis, hamiltonian, denominator.gamma)
    fixtures: list[np.ndarray] = []
    fixtures.append(np.eye(basis.dimension, dtype=np.complex128) / basis.dimension)
    rng = np.random.Generator(np.random.Philox(scientific_seed))
    raw = rng.normal(size=(basis.dimension, basis.dimension)) + 1j * rng.normal(
        size=(basis.dimension, basis.dimension)
    )
    hermitian = raw + raw.conj().T
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    positive = (
        eigenvectors @ np.diag(np.exp(eigenvalues - eigenvalues.max())) @ (eigenvectors.conj().T)
    )
    fixtures.append(positive / np.trace(positive))
    discrepancies: list[float] = []
    fixture_hashes: list[str] = []
    for fixture in fixtures:
        vector = fixture.reshape(-1, order="F")
        kron_action = np.asarray(generator @ vector).reshape(
            fixture.shape,
            order="F",
        )
        direct_action = direct_rhs(fixture)
        discrepancies.append(float(np.linalg.norm(kron_action - direct_action)))
        fixture_hashes.append(_density_sha256(fixture))
    maximum = max(discrepancies)
    return {
        "scientific_fixture_seed": scientific_seed,
        "fixture_hashes": fixture_hashes,
        "hilbert_schmidt_discrepancies": discrepancies,
        "maximum_hilbert_schmidt_discrepancy": maximum,
        "passed": maximum <= 1e-12,
    }


def reference_cell(
    unit: PreparationUnit,
    denominator: Denominator,
    config: QuantumTrajectoryReferenceValidationConfig,
    *,
    scientific_seed: int | None = None,
) -> dict[str, object]:
    if (unit.l_sites, unit.particles) != (config.l_dense, config.n_dense):
        raise ValueError("deterministic reference requires the frozen dense basis")
    scientific_seed = generator_fixture_seed(
        unit.l_sites, unit.particles, denominator.gamma, scientific_seed,
    )
    basis = build_basis(unit.l_sites, unit.particles)
    hamiltonian = build_hamiltonian(
        basis,
        action=Action.HOLD,
        epsilon=float(config.epsilon),
        j_xy=float(config.j_xy),
        j_z=float(config.j_z),
    )
    supports = observable_supports(basis, hamiltonian)
    initial_state = basis.state_vector(unit.initial_state)
    initial_density = np.outer(initial_state, initial_state.conj())
    generator = lindblad_superoperator(
        basis,
        hamiltonian,
        gamma=denominator.gamma,
    )
    conformance = generator_conformance(
        basis, hamiltonian, denominator, scientific_seed=scientific_seed,
    )

    started = time.perf_counter()
    sparse_density = propagate_density(
        generator,
        initial_density,
        float(config.dense_horizon),
    )
    sparse_wall = time.perf_counter() - started
    direct_rhs = _direct_rhs_factory(basis, hamiltonian, denominator.gamma)
    tight_density, tight_solver = _integrate_direct(
        direct_rhs,
        initial_density,
        float(config.dense_horizon),
        config.reference_tight,
    )
    tighter_density, tighter_solver = _integrate_direct(
        direct_rhs,
        initial_density,
        float(config.dense_horizon),
        config.reference_tighter,
    )

    sparse_observables = density_observables(basis, hamiltonian, sparse_density)
    tight_observables = density_observables(basis, hamiltonian, tight_density)
    tighter_observables = density_observables(basis, hamiltonian, tighter_density)
    widths = np.asarray([support.width for support in supports], dtype=np.float64)
    route_hs = float(np.linalg.norm(sparse_density - tight_density))
    self_hs = float(np.linalg.norm(tight_density - tighter_density))
    normalized_route = np.divide(
        np.abs(sparse_observables - tight_observables),
        widths,
        out=np.zeros_like(widths),
        where=widths > 0.0,
    )
    envelope_radius = np.maximum.reduce(
        (
            np.abs(sparse_observables - tight_observables),
            np.abs(tight_observables - tighter_observables),
            1e-12 * widths,
        )
    )
    diagnostics = {
        "sparse": density_diagnostics(basis, sparse_density),
        "tight": density_diagnostics(basis, tight_density),
        "tighter": density_diagnostics(basis, tighter_density),
    }
    valid_densities = all(
        value["trace_residual"] <= float(config.reference_trace_tolerance)
        and value["hermiticity_residual"] <= float(config.reference_hermiticity_tolerance)
        and value["minimum_eigenvalue"] >= float(config.reference_minimum_eigenvalue)
        and value["particle_number_residual"] <= float(config.particle_number_tolerance)
        for value in diagnostics.values()
    )
    passed = bool(
        conformance["passed"]
        and valid_densities
        and route_hs <= float(config.reference_route_hs_tolerance)
        and self_hs <= float(config.reference_self_convergence_tolerance)
        and float(np.max(normalized_route))
        <= float(config.reference_observable_normalized_tolerance)
    )
    return {
        "schema": 'empirical-lawhood/simulators/quantum-trajectory/density-reference-cell',
        "version": '1.0.0',
        "unit_id": unit.unit_id,
        "roster_id": unit.roster_id,
        "initial_state": unit.initial_state,
        "initial_density_sha256": _density_sha256(initial_density),
        "l_sites": unit.l_sites,
        "particles": unit.particles,
        "denominator": denominator.value,
        "action": Action.HOLD.value,
        "horizon": float(config.dense_horizon),
        "hamiltonian_sha256": matrix_fingerprint(hamiltonian),
        "liouvillian_sha256": matrix_fingerprint(csr_matrix(generator)),
        "generator_conformance": conformance,
        "supports": support_rows(supports),
        "sparse": {
            "route": "sparse-kron-expm-multiply",
            "density_sha256": _density_sha256(sparse_density),
            "observables": sparse_observables.tolist(),
            "diagnostics": diagnostics["sparse"],
            "wall_seconds": sparse_wall,
        },
        "tight": {
            "route": "direct-matrix-dop853",
            "density_sha256": _density_sha256(tight_density),
            "observables": tight_observables.tolist(),
            "diagnostics": diagnostics["tight"],
            "solver": tight_solver,
        },
        "tighter": {
            "route": "direct-matrix-dop853",
            "density_sha256": _density_sha256(tighter_density),
            "observables": tighter_observables.tolist(),
            "diagnostics": diagnostics["tighter"],
            "solver": tighter_solver,
        },
        "route_hilbert_schmidt_difference": route_hs,
        "tight_tighter_hilbert_schmidt_difference": self_hs,
        "maximum_normalized_observable_difference": float(np.max(normalized_route)),
        "reference_observables": sparse_observables.tolist(),
        "reference_envelope_radius": envelope_radius.tolist(),
        "reference_envelope_lower": (sparse_observables - envelope_radius).tolist(),
        "reference_envelope_upper": (sparse_observables + envelope_radius).tolist(),
        "validity": Validity.VALID.value if passed else Validity.INVALID.value,
        "reason_code": "OK" if passed else "REFERENCE_TOLERANCE_FAILURE",
        "passed": passed,
        "_density_arrays": {
            "sparse": sparse_density,
            "tight": tight_density,
            "tighter": tighter_density,
        },
    }


def strip_density_arrays(row: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key != "_density_arrays"}


def support_manifest(config: QuantumTrajectoryReferenceValidationConfig) -> dict[str, object]:
    basis = build_basis(config.l_dense, config.n_dense)
    hamiltonian = build_hamiltonian(
        basis,
        action=Action.HOLD,
        epsilon=float(config.epsilon),
        j_xy=float(config.j_xy),
        j_z=float(config.j_z),
    )
    supports = observable_supports(basis, hamiltonian)
    return {
        "schema": 'empirical-lawhood/simulators/quantum-trajectory/observable-support-manifest',
        "version": '1.0.0',
        "value": {
            "basis_sha256": basis.fingerprint,
            "hamiltonian_sha256": matrix_fingerprint(hamiltonian),
            "supports": [asdict(value) for value in supports],
        },
    }


__all__ = [
    "density_diagnostics",
    "generator_conformance",
    "reference_cell",
    "strip_density_arrays",
    "support_manifest",
]
