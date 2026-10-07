"""Fixed-number basis, operators and independent propagation views for quantum receiver response."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse import (
    csc_matrix,
    csr_matrix,
    eye,
    kron,
)
from scipy.sparse.linalg import expm_multiply

from .contracts import Action, basis_states


ComplexVector = NDArray[np.complex128]
ComplexMatrix = NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class FixedNumberBasis:
    l_sites: int
    particles: int
    states: tuple[int, ...]
    index: dict[int, int]
    occupations: NDArray[np.float64]

    @property
    def dimension(self) -> int:
        return len(self.states)

    @property
    def fingerprint(self) -> str:
        payload = np.asarray(self.states, dtype="<u8").tobytes()
        return sha256(payload).hexdigest()

    def state_vector(self, state: int) -> ComplexVector:
        vector = np.zeros(self.dimension, dtype=np.complex128)
        try:
            vector[self.index[state]] = 1.0
        except KeyError as error:
            raise ValueError("initial state is outside the fixed-number basis") from error
        return vector


def build_basis(l_sites: int, particles: int) -> FixedNumberBasis:
    if l_sites <= 1 or particles <= 0 or particles >= l_sites:
        raise ValueError("fixed-number basis dimensions are invalid")
    states = basis_states(l_sites, particles)
    occupations = np.asarray(
        [[(state >> site) & 1 for site in range(l_sites)] for state in states],
        dtype=np.float64,
    )
    return FixedNumberBasis(
        l_sites=l_sites,
        particles=particles,
        states=states,
        index={state: index for index, state in enumerate(states)},
        occupations=occupations,
    )


def build_hamiltonian(
    basis: FixedNumberBasis,
    *,
    action: Action = Action.HOLD,
    epsilon: float = 0.20,
    j_xy: float = 1.0,
    j_z: float = 1.0,
) -> csr_matrix:
    """Build the periodic hard-core-boson/XXZ matrix in canonical basis order."""

    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    l_sites = basis.l_sites
    boundary_bonds = {(l_sites // 2 - 1, l_sites // 2), (l_sites - 1, 0)}
    for column, state in enumerate(basis.states):
        diagonal = 0.0
        for site in range(l_sites):
            neighbor = (site + 1) % l_sites
            occupied = (state >> site) & 1
            neighbor_occupied = (state >> neighbor) & 1
            diagonal += j_z * occupied * neighbor_occupied
            if occupied == neighbor_occupied:
                continue
            hopping_scale = (
                1.0 + action.sign * epsilon if (site, neighbor) in boundary_bonds else 1.0
            )
            target = state ^ (1 << site) ^ (1 << neighbor)
            rows.append(basis.index[target])
            columns.append(column)
            values.append(complex(0.5 * j_xy * hopping_scale))
        rows.append(column)
        columns.append(column)
        values.append(complex(diagonal))
    matrix = csr_matrix(
        (np.asarray(values, dtype=np.complex128), (rows, columns)),
        shape=(basis.dimension, basis.dimension),
        dtype=np.complex128,
    )
    matrix.sum_duplicates()
    matrix.sort_indices()
    return matrix


def sparse_matrix_fingerprint(matrix: csr_matrix) -> str:
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


class Propagator(Protocol):
    def propagate(self, state: ComplexVector, duration: float) -> ComplexVector: ...


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
    """Independently implemented dense Hermitian propagation sentinel."""

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


def site_probabilities(
    basis: FixedNumberBasis,
    state: ComplexVector,
) -> NDArray[np.float64]:
    probabilities = (np.abs(state) ** 2) @ basis.occupations
    probabilities = np.asarray(probabilities / basis.particles, dtype=np.float64)
    probabilities = np.maximum(probabilities, 0.0)
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("jump-site probability vector is invalid")
    return probabilities / total


def apply_jump(
    basis: FixedNumberBasis,
    state: ComplexVector,
    site: int,
) -> tuple[ComplexVector, float]:
    if not 0 <= site < basis.l_sites:
        raise ValueError("jump site leaves the chain")
    projected = np.asarray(state * basis.occupations[:, site], dtype=np.complex128)
    return normalize_state(projected)


def particle_number_error(
    basis: FixedNumberBasis,
    state: ComplexVector,
) -> float:
    weights = np.abs(state) ** 2
    expected = float(weights @ basis.occupations.sum(axis=1))
    return abs(expected - basis.particles)


def phase_gauge_infidelity(left: ComplexVector, right: ComplexVector) -> float:
    overlap = complex(np.vdot(left, right))
    return max(0.0, 1.0 - abs(overlap) ** 2)


def phase_aligned_max_error(left: ComplexVector, right: ComplexVector) -> float:
    overlap = complex(np.vdot(left, right))
    phase = overlap / abs(overlap) if abs(overlap) > 0.0 else 1.0 + 0.0j
    return float(np.max(np.abs(left - np.conj(phase) * right)))


def number_projectors(basis: FixedNumberBasis) -> tuple[csr_matrix, ...]:
    return tuple(
        csr_matrix(np.diag(basis.occupations[:, site]).astype(np.complex128))
        for site in range(basis.l_sites)
    )


def lindblad_superoperator(
    basis: FixedNumberBasis,
    hamiltonian: csr_matrix,
    *,
    gamma: float,
) -> csc_matrix:
    """Return the column-major vectorized Lindblad generator."""

    dimension = basis.dimension
    identity = eye(dimension, dtype=np.complex128, format="csc")
    h_csc = hamiltonian.tocsc()
    generator = -1j * (
        kron(identity, h_csc, format="csc") - kron(h_csc.transpose(), identity, format="csc")
    )
    for projector in number_projectors(basis):
        n_csc = projector.tocsc()
        n2 = n_csc @ n_csc
        generator += gamma * (
            kron(n_csc.transpose(), n_csc, format="csc")
            - 0.5 * kron(identity, n2, format="csc")
            - 0.5 * kron(n2.transpose(), identity, format="csc")
        )
    generator.sum_duplicates()
    generator.sort_indices()
    return generator


def propagate_density(
    generator: csc_matrix,
    density: ComplexMatrix,
    duration: float,
) -> ComplexMatrix:
    dimension = density.shape[0]
    if density.shape != (dimension, dimension):
        raise ValueError("density matrix must be square")
    vector = np.asarray(density, dtype=np.complex128).reshape(-1, order="F")
    result = expm_multiply(generator * duration, vector)
    return np.asarray(result, dtype=np.complex128).reshape(
        (dimension, dimension),
        order="F",
    )


__all__ = [
    "ComplexMatrix",
    "ComplexVector",
    "DenseEigensystemPropagator",
    "FixedNumberBasis",
    "Propagator",
    "SparseKrylovPropagator",
    "apply_jump",
    "build_basis",
    "build_hamiltonian",
    "hermiticity_residual",
    "lindblad_superoperator",
    "normalize_state",
    "number_projectors",
    "particle_number_error",
    "phase_aligned_max_error",
    "phase_gauge_infidelity",
    "propagate_density",
    "site_probabilities",
    "sparse_matrix_fingerprint",
]
