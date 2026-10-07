"""Numerical state and exact SU(2) backgrounds for the Six-matrix response matrix world."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt


ComplexArray = npt.NDArray[np.complex128]


def hermitian_part(value: ComplexArray) -> ComplexArray:
    """Project a square complex array onto the Hermitian real vector space."""

    return np.asarray((value + np.swapaxes(value.conj(), -1, -2)) * 0.5, dtype="<c16")


def hermiticity_residual(value: ComplexArray) -> float:
    numerator = float(np.linalg.norm(value - np.swapaxes(value.conj(), -1, -2)))
    denominator = max(float(np.linalg.norm(value)), 1.0)
    return numerator / denominator


@dataclass(frozen=True, slots=True)
class SixMatrixState:
    """One in-memory phase-space point; persisted identity lives in a checkpoint."""

    q: int
    positions: ComplexArray
    momenta: ComplexArray
    step_index: int
    alpha_tilde_x: float
    alpha_tilde_y: float

    def __post_init__(self) -> None:
        if self.q not in {2, 3, 4}:
            raise ValueError("Six-matrix response state q must be one of 2, 3 or 4")
        n = self.q**2
        for name in ("positions", "momenta"):
            value = np.asarray(getattr(self, name))
            if value.shape != (2, 3, n, n) or value.dtype != np.dtype("complex128"):
                raise ValueError(f"Six-matrix response {name} must have exact (2,3,n,n) complex128 shape")
            if hermiticity_residual(value) > 1e-12:
                raise ValueError(f"Six-matrix response {name} are not Hermitian")
            copied = np.ascontiguousarray(value, dtype="<c16")
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)
        if self.step_index < 0:
            raise ValueError("Six-matrix response state step index cannot be negative")

    @property
    def n(self) -> int:
        return self.q**2

    @property
    def finite(self) -> bool:
        return bool(
            np.isfinite(self.positions).all()
            and np.isfinite(self.momenta).all()
            and np.isfinite((self.alpha_tilde_x, self.alpha_tilde_y)).all()
        )


def su2_generators(q: int) -> ComplexArray:
    """Return the q-dimensional irreducible Hermitian generators."""

    if q < 2:
        raise ValueError("SU(2) representation dimension must be at least two")
    spin = (q - 1) / 2.0
    raising = np.zeros((q, q), dtype="<c16")
    for column in range(1, q):
        m = spin - column
        raising[column - 1, column] = np.sqrt((spin - m) * (spin + m + 1.0))
    lowering = raising.conj().T
    x = (raising + lowering) * 0.5
    y = (raising - lowering) / (2.0j)
    z = np.diag(np.asarray([spin - index for index in range(q)], dtype=np.float64))
    return np.ascontiguousarray(np.stack((x, y, z)), dtype="<c16")


def ideal_positions(
    *,
    q: int,
    alpha_tilde_x: float,
    alpha_tilde_y: float,
    constitution: Literal["00", "10", "01", "11"],
) -> ComplexArray:
    """Construct exact zero/one-factor/product published backgrounds."""

    if constitution not in {"00", "10", "01", "11"}:
        raise ValueError("Six-matrix response ideal constitution is outside 00/10/01/11")
    generators = su2_generators(q)
    identity = np.eye(q, dtype="<c16")
    n = q**2
    positions = np.zeros((2, 3, n, n), dtype="<c16")
    if constitution[0] == "1":
        alpha_x = alpha_tilde_x / q
        positions[0] = np.stack(
            tuple((2.0 * alpha_x / 3.0) * np.kron(value, identity) for value in generators)
        )
    if constitution[1] == "1":
        alpha_y = alpha_tilde_y / q
        positions[1] = np.stack(
            tuple((2.0 * alpha_y / 3.0) * np.kron(identity, value) for value in generators)
        )
    return np.ascontiguousarray(positions, dtype="<c16")


def ideal_state(
    *,
    q: int,
    alpha_tilde_x: float,
    alpha_tilde_y: float,
    constitution: Literal["00", "10", "01", "11"],
) -> SixMatrixState:
    positions = ideal_positions(
        q=q,
        alpha_tilde_x=alpha_tilde_x,
        alpha_tilde_y=alpha_tilde_y,
        constitution=constitution,
    )
    return SixMatrixState(
        q=q,
        positions=positions,
        momenta=np.zeros_like(positions),
        step_index=0,
        alpha_tilde_x=alpha_tilde_x,
        alpha_tilde_y=alpha_tilde_y,
    )


def conjugate_state(state: SixMatrixState, unitary: ComplexArray) -> SixMatrixState:
    if unitary.shape != (state.n, state.n):
        raise ValueError("unitary dimension differs from Six-matrix response state")
    identity = np.eye(state.n, dtype="<c16")
    if float(np.linalg.norm(unitary.conj().T @ unitary - identity)) > 1e-10:
        raise ValueError("Six-matrix response conjugation matrix is not unitary")
    positions = unitary @ state.positions @ unitary.conj().T
    momenta = unitary @ state.momenta @ unitary.conj().T
    return SixMatrixState(
        q=state.q,
        positions=hermitian_part(positions),
        momenta=hermitian_part(momenta),
        step_index=state.step_index,
        alpha_tilde_x=state.alpha_tilde_x,
        alpha_tilde_y=state.alpha_tilde_y,
    )


__all__ = [
    "ComplexArray",
    'SixMatrixState',
    "conjugate_state",
    "hermitian_part",
    "hermiticity_residual",
    "ideal_positions",
    "ideal_state",
    "su2_generators",
]
