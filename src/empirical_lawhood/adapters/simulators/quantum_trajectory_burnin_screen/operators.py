"""Independent sparse/dense operator and propagation views for quantum trajectory burnin screen."""

from __future__ import annotations

from hashlib import sha256
from typing import Protocol

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import csc_matrix, csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply

from .basis import FixedNumberBasis
from .types import Action, ComplexVector


class Propagator(Protocol):
    def propagate(self, state: ComplexVector, duration: float) -> ComplexVector: ...


def build_hamiltonian(
    basis: FixedNumberBasis,
    *,
    action: Action,
    epsilon: float,
    j_xy: float,
    j_z: float,
) -> csr_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    boundary_bonds = {
        (basis.l_sites // 2 - 1, basis.l_sites // 2),
        (basis.l_sites - 1, 0),
    }
    for column, state in enumerate(basis.states):
        diagonal = 0.0
        for site in range(basis.l_sites):
            neighbor = (site + 1) % basis.l_sites
            occupied = (state >> site) & 1
            neighbor_occupied = (state >> neighbor) & 1
            diagonal += j_z * occupied * neighbor_occupied
            if occupied == neighbor_occupied:
                continue
            multiplier = 1.0 + action.sign * epsilon if (site, neighbor) in boundary_bonds else 1.0
            target = state ^ (1 << site) ^ (1 << neighbor)
            rows.append(basis.index[target])
            columns.append(column)
            values.append(complex(0.5 * j_xy * multiplier))
        rows.append(column)
        columns.append(column)
        values.append(complex(diagonal))
    result = csr_matrix(
        (
            np.asarray(values, dtype=np.complex128),
            (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
        ),
        shape=(basis.dimension, basis.dimension),
        dtype=np.complex128,
    )
    result.sum_duplicates()
    result.sort_indices()
    return result


def matrix_fingerprint(matrix: csr_matrix) -> str:
    canonical = matrix.copy()
    canonical.sum_duplicates()
    canonical.sort_indices()
    payload = b"".join(
        (
            np.asarray(canonical.shape, dtype="<i8").tobytes(),
            np.asarray(canonical.indptr, dtype="<i8").tobytes(),
            np.asarray(canonical.indices, dtype="<i8").tobytes(),
            np.asarray(canonical.data, dtype="<c16").tobytes(),
        )
    )
    return sha256(payload).hexdigest()


def hermiticity_residual(matrix: csr_matrix) -> float:
    difference = matrix - matrix.getH()
    return float(np.max(np.abs(difference.data))) if difference.nnz else 0.0


class SparseKrylovPropagator:
    def __init__(self, hamiltonian: csr_matrix) -> None:
        self._hamiltonian = hamiltonian
        self._trace = complex(hamiltonian.diagonal().sum())

    def propagate(self, state: ComplexVector, duration: float) -> ComplexVector:
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("propagation duration must be finite and nonnegative")
        if duration == 0.0:
            return state.copy()
        scale = -1j * duration
        result = expm_multiply(
            scale * self._hamiltonian,
            state,
            traceA=scale * self._trace,
        )
        return np.asarray(result, dtype=np.complex128)


class DenseEigensystemPropagator:
    def __init__(self, hamiltonian: csr_matrix) -> None:
        eigenvalues, eigenvectors = eigh(
            np.asarray(hamiltonian.toarray(), dtype=np.complex128),
            check_finite=True,
            driver="evd",
        )
        self._eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        self._eigenvectors = np.asarray(eigenvectors, dtype=np.complex128)

    def propagate(self, state: ComplexVector, duration: float) -> ComplexVector:
        if not np.isfinite(duration) or duration < 0.0:
            raise ValueError("propagation duration must be finite and nonnegative")
        coefficients = self._eigenvectors.conj().T @ state
        coefficients *= np.exp(-1j * self._eigenvalues * duration)
        return np.asarray(self._eigenvectors @ coefficients, dtype=np.complex128)


def normalize_state(state: ComplexVector) -> tuple[ComplexVector, float]:
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("conditional state norm is invalid")
    return np.asarray(state / norm, dtype=np.complex128), abs(norm - 1.0)


def particle_number_residual(basis: FixedNumberBasis, state: ComplexVector) -> float:
    weights = np.abs(state) ** 2
    expected = float(weights @ basis.occupations.sum(axis=1))
    return abs(expected - basis.particles)


def site_occupations(
    basis: FixedNumberBasis,
    state: ComplexVector,
) -> np.ndarray:
    result = (np.abs(state) ** 2) @ basis.occupations
    return np.asarray(result, dtype=np.float64)


def phase_gauge_infidelity(left: ComplexVector, right: ComplexVector) -> float:
    overlap = complex(np.vdot(left, right))
    return max(0.0, 1.0 - abs(overlap) ** 2)


def phase_aligned_max_error(left: ComplexVector, right: ComplexVector) -> float:
    overlap = complex(np.vdot(left, right))
    phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
    return float(np.max(np.abs(left - np.conj(phase) * right)))


def lindblad_superoperator(
    basis: FixedNumberBasis,
    hamiltonian: csr_matrix,
    *,
    gamma: float,
) -> csc_matrix:
    dimension = basis.dimension
    identity = eye(dimension, dtype=np.complex128, format="csc")
    h_csc = hamiltonian.tocsc()
    generator = -1j * (
        kron(identity, h_csc, format="csc") - kron(h_csc.transpose(), identity, format="csc")
    )
    for site in range(basis.l_sites):
        projector = csc_matrix(np.diag(basis.occupations[:, site]).astype(np.complex128))
        generator += gamma * (
            kron(projector.transpose(), projector, format="csc")
            - 0.5 * kron(identity, projector, format="csc")
            - 0.5 * kron(projector.transpose(), identity, format="csc")
        )
    generator.sum_duplicates()
    generator.sort_indices()
    return generator


def propagate_density(
    generator: csc_matrix,
    density: np.ndarray,
    duration: float,
) -> np.ndarray:
    if density.ndim != 2 or density.shape[0] != density.shape[1]:
        raise ValueError("density matrix must be square")
    dimension = density.shape[0]
    vector = np.asarray(density, dtype=np.complex128).reshape(-1, order="F")
    result = expm_multiply(generator * duration, vector)
    return np.asarray(result, dtype=np.complex128).reshape(
        (dimension, dimension),
        order="F",
    )


__all__ = [
    "DenseEigensystemPropagator",
    "Propagator",
    "SparseKrylovPropagator",
    "build_hamiltonian",
    "hermiticity_residual",
    "lindblad_superoperator",
    "matrix_fingerprint",
    "normalize_state",
    "particle_number_residual",
    "phase_aligned_max_error",
    "phase_gauge_infidelity",
    "propagate_density",
    "site_occupations",
]
