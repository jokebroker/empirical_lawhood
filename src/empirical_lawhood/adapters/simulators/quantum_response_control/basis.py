"""Fixed-number basis and Hamiltonian operators for quantum response control."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply

from .contracts import Action


ComplexVector = NDArray[np.complex128]


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
        return sha256(np.asarray(self.states, dtype="<u8").tobytes()).hexdigest()

    def state_vector(self, encoded_state: int) -> ComplexVector:
        vector = np.zeros(self.dimension, dtype=np.complex128)
        try:
            vector[self.index[encoded_state]] = 1.0
        except KeyError as error:
            raise ValueError("initial state is outside the fixed-number basis") from error
        return vector


def build_basis(l_sites: int, particles: int) -> FixedNumberBasis:
    if l_sites <= 1 or particles <= 0 or particles >= l_sites:
        raise ValueError("fixed-number basis dimensions are invalid")
    states = tuple(
        sorted(
            sum(1 << site for site in occupied)
            for occupied in combinations(range(l_sites), particles)
        )
    )
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
    """Periodic hard-core-boson/XXZ Hamiltonian in canonical basis order."""

    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    half = basis.l_sites // 2
    receiver_boundary_bonds = {(half - 1, half), (basis.l_sites - 1, 0)}
    for column, encoded in enumerate(basis.states):
        diagonal = 0.0
        for site in range(basis.l_sites):
            neighbor = (site + 1) % basis.l_sites
            occupied = (encoded >> site) & 1
            neighbor_occupied = (encoded >> neighbor) & 1
            diagonal += j_z * occupied * neighbor_occupied
            if occupied == neighbor_occupied:
                continue
            hopping_scale = (
                1.0 + action.sign * epsilon if (site, neighbor) in receiver_boundary_bonds else 1.0
            )
            target = encoded ^ (1 << site) ^ (1 << neighbor)
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


def build_boundary_current_operator(
    basis: FixedNumberBasis,
    *,
    action: Action,
    epsilon: float = 0.20,
    j_xy: float = 1.0,
) -> csr_matrix:
    """Signed coherent current into the receiver half A.

    The convention satisfies d<q_A>/dt=<I_into_A> under Schrödinger
    evolution.  The two periodic boundary bonds are oriented 6->5 and 11->0.
    """

    rows: list[int] = []
    columns: list[int] = []
    values: list[complex] = []
    half = basis.l_sites // 2
    oriented = ((half, half - 1), (basis.l_sites - 1, 0))
    scale = 1.0 + action.sign * epsilon
    hopping = 0.5 * j_xy * scale
    for column, encoded in enumerate(basis.states):
        for source, target_site in oriented:
            source_occupied = (encoded >> source) & 1
            target_occupied = (encoded >> target_site) & 1
            if source_occupied and not target_occupied:
                target = encoded ^ (1 << source) ^ (1 << target_site)
                rows.append(basis.index[target])
                columns.append(column)
                values.append(complex(-1j * hopping))
            if target_occupied and not source_occupied:
                target = encoded ^ (1 << source) ^ (1 << target_site)
                rows.append(basis.index[target])
                columns.append(column)
                values.append(complex(1j * hopping))
    operator = csr_matrix(
        (np.asarray(values, dtype=np.complex128), (rows, columns)),
        shape=(basis.dimension, basis.dimension),
        dtype=np.complex128,
    )
    operator.sum_duplicates()
    operator.sort_indices()
    return operator


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
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("propagation duration must be finite and nonnegative")
        if duration == 0:
            return state.copy()
        scale = -1j * duration
        result = expm_multiply(
            scale * self._hamiltonian,
            state,
            traceA=scale * self._trace,
        )
        return np.asarray(result, dtype=np.complex128)


class DenseEigensystemPropagator:
    """Independent dense Hermitian sentinel used only in instrument conformance."""

    def __init__(self, hamiltonian: csr_matrix) -> None:
        eigenvalues, eigenvectors = eigh(
            np.asarray(hamiltonian.toarray(), dtype=np.complex128),
            check_finite=True,
            driver="evd",
        )
        self._eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        self._eigenvectors = np.asarray(eigenvectors, dtype=np.complex128)

    def propagate(self, state: ComplexVector, duration: float) -> ComplexVector:
        if not np.isfinite(duration) or duration < 0:
            raise ValueError("propagation duration must be finite and nonnegative")
        coefficients = self._eigenvectors.conj().T @ state
        coefficients *= np.exp(-1j * self._eigenvalues * duration)
        return np.asarray(self._eigenvectors @ coefficients, dtype=np.complex128)


def normalize_state(state: ComplexVector) -> tuple[ComplexVector, float]:
    norm = float(np.linalg.norm(state))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("conditional state norm is invalid")
    return np.asarray(state / norm, dtype=np.complex128), abs(norm - 1.0)


def site_probabilities(
    basis: FixedNumberBasis,
    state: ComplexVector,
) -> NDArray[np.float64]:
    probabilities = (np.abs(state) ** 2) @ basis.occupations
    probabilities = np.maximum(
        np.asarray(probabilities / basis.particles, dtype=np.float64),
        0.0,
    )
    total = float(probabilities.sum())
    if not np.isfinite(total) or total <= 0:
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
    normalized, _projection_norm = normalize_state(projected)
    # The projection norm is the physical jump likelihood, not a numerical
    # state-normalization error.  The receiver-response contract previously mixed those quantities.
    return normalized, abs(float(np.linalg.norm(normalized)) - 1.0)


def particle_number_error(basis: FixedNumberBasis, state: ComplexVector) -> float:
    weights = np.abs(state) ** 2
    expected = float(weights @ basis.occupations.sum(axis=1))
    return abs(expected - basis.particles)


def phase_gauge_infidelity(left: ComplexVector, right: ComplexVector) -> float:
    overlap = complex(np.vdot(left, right))
    return max(0.0, 1.0 - abs(overlap) ** 2)


def phase_aligned_max_error(left: ComplexVector, right: ComplexVector) -> float:
    overlap = complex(np.vdot(left, right))
    phase = overlap / abs(overlap) if abs(overlap) else 1.0 + 0.0j
    return float(np.max(np.abs(left - np.conj(phase) * right)))


def half_count_diagonal(basis: FixedNumberBasis) -> NDArray[np.float64]:
    return np.asarray(
        basis.occupations[:, : basis.l_sites // 2].sum(axis=1),
        dtype=np.float64,
    )


def expectation(state: ComplexVector, operator: csr_matrix) -> float:
    value = complex(np.vdot(state, operator @ state))
    if abs(value.imag) > 1e-10:
        raise ValueError("Hermitian expectation has a material imaginary component")
    return float(value.real)


__all__ = [
    "ComplexVector",
    "DenseEigensystemPropagator",
    "FixedNumberBasis",
    "Propagator",
    "SparseKrylovPropagator",
    "apply_jump",
    "build_basis",
    "build_boundary_current_operator",
    "build_hamiltonian",
    "expectation",
    "half_count_diagonal",
    "hermiticity_residual",
    "normalize_state",
    "particle_number_error",
    "phase_aligned_max_error",
    "phase_gauge_infidelity",
    "site_probabilities",
    "sparse_matrix_fingerprint",
]
