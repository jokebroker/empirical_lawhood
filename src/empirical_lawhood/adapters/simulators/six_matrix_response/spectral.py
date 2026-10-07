"""Adjoint-Laplacian and invariant spectral receivers for Six-matrix response."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import exp
from typing import ClassVar, Literal

import numpy as np

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_decimal, validate_stable_id

from .model import ComplexArray


RealArray = np.ndarray


@dataclass(frozen=True, slots=True)
class SixMatrixResponseLaplacianSpectrum(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-laplacian-spectrum'

    spectrum_id: str
    sector: str
    q: int
    eigenvalues: tuple[Decimal, ...]
    kernel_tolerance: Decimal
    kernel_dimension: int
    heat_trace_t1_normalized: Decimal
    finite: bool
    valid: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.spectrum_id, field_name="spectrum_id")
        if self.sector not in {"X", "Y", "JOINT"}:
            raise ValueError("Six-matrix response Laplacian sector must be X, Y or JOINT")
        if self.q not in {2, 3, 4} or len(self.eigenvalues) != self.q**4:
            raise ValueError("Six-matrix response spectrum has another q/ambient dimension")
        for index, value in enumerate(self.eigenvalues):
            validate_decimal(value, field_name=f"eigenvalues[{index}]", minimum=Decimal(0))
        if tuple(sorted(self.eigenvalues)) != self.eigenvalues:
            raise ValueError("Six-matrix response spectrum must be ascending")
        validate_decimal(self.kernel_tolerance, field_name="kernel_tolerance", minimum=Decimal(0))
        validate_decimal(
            self.heat_trace_t1_normalized,
            field_name="heat_trace_t1_normalized",
            minimum=Decimal(0),
        )
        expected_kernel = sum(value <= self.kernel_tolerance for value in self.eigenvalues)
        if self.kernel_dimension != expected_kernel:
            raise ValueError("Six-matrix response kernel dimension differs from its spectrum/tolerance")
        if self.valid != self.finite:
            raise ValueError("Six-matrix response spectral validity differs from finiteness")


def _check_triplet(triplet: ComplexArray) -> int:
    if triplet.ndim != 3 or triplet.shape[0] != 3 or triplet.shape[1] != triplet.shape[2]:
        raise ValueError("Six-matrix response Laplacian requires one square matrix triplet")
    if triplet.dtype != np.dtype("complex128") or not np.isfinite(triplet).all():
        raise ValueError("Six-matrix response Laplacian triplet must be finite complex128")
    return int(triplet.shape[1])


def adjoint_laplacian_explicit(triplet: ComplexArray) -> ComplexArray:
    """Kronecker transcription of A -> sum_a [X_a,[X_a,A]]."""

    n = _check_triplet(triplet)
    identity = np.eye(n, dtype="<c16")
    result = np.zeros((n * n, n * n), dtype="<c16")
    for matrix in triplet:
        commutator_map = np.kron(identity, matrix) - np.kron(matrix.T, identity)
        result += commutator_map @ commutator_map
    return np.asarray((result + result.conj().T) * 0.5, dtype="<c16")


def adjoint_laplacian_basis(triplet: ComplexArray) -> ComplexArray:
    """Independent basis-column construction used as the optimized cross-check."""

    n = _check_triplet(triplet)
    result = np.zeros((n * n, n * n), dtype="<c16")
    for column in range(n * n):
        probe = np.zeros((n, n), dtype="<c16")
        row_index, column_index = np.unravel_index(column, (n, n), order="F")
        probe[row_index, column_index] = 1.0
        image = np.zeros_like(probe)
        for matrix in triplet:
            first = matrix @ probe - probe @ matrix
            image += matrix @ first - first @ matrix
        result[:, column] = image.reshape(n * n, order="F")
    return np.asarray((result + result.conj().T) * 0.5, dtype="<c16")


def laplacian_eigenvalues(operator: ComplexArray) -> RealArray:
    if operator.ndim != 2 or operator.shape[0] != operator.shape[1]:
        raise ValueError("Six-matrix response Laplacian operator must be square")
    values = np.asarray(np.linalg.eigvalsh(operator), dtype=np.float64)
    floor = 1e-12 * max(1.0, float(np.max(np.abs(values))))
    if float(np.min(values)) < -floor:
        raise FloatingPointError("Six-matrix response adjoint Laplacian is materially nonpositive")
    values[np.abs(values) <= floor] = 0.0
    return values


def ideal_laplacian_eigenvalues(
    *,
    q: int,
    alpha_tilde: float,
    active: bool,
) -> RealArray:
    """Exact one-sector spectrum including the other tensor-factor spectator."""

    if q not in {2, 3, 4}:
        raise ValueError("Six-matrix response ideal spectrum q must be 2, 3 or 4")
    if not active:
        return np.zeros(q**4, dtype=np.float64)
    scale_squared = (2.0 * (alpha_tilde / q) / 3.0) ** 2
    values = [
        scale_squared * ell * (ell + 1) for ell in range(q) for _ in range((2 * ell + 1) * q**2)
    ]
    return np.asarray(sorted(values), dtype=np.float64)


def ideal_joint_laplacian_eigenvalues(
    *,
    q: int,
    alpha_tilde_x: float,
    alpha_tilde_y: float,
    constitution: Literal["00", "10", "01", "11"],
) -> RealArray:
    if constitution == "00":
        return np.zeros(q**4, dtype=np.float64)
    scale_x = (2.0 * (alpha_tilde_x / q) / 3.0) ** 2
    scale_y = (2.0 * (alpha_tilde_y / q) / 3.0) ** 2
    values: list[float] = []
    for ell_x in range(q):
        for ell_y in range(q):
            value = (scale_x * ell_x * (ell_x + 1) if constitution[0] == "1" else 0.0) + (
                scale_y * ell_y * (ell_y + 1) if constitution[1] == "1" else 0.0
            )
            multiplicity = (2 * ell_x + 1) * (2 * ell_y + 1)
            values.extend((value,) * multiplicity)
    return np.asarray(sorted(values), dtype=np.float64)


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("Six-matrix response spectrum contains a nonfinite value")
    return Decimal(repr(float(value)))


def spectral_receiver(
    *,
    receiver_prefix: str,
    q: int,
    positions: ComplexArray,
    kernel_relative_tolerance: float = 1e-10,
) -> tuple[SixMatrixResponseLaplacianSpectrum, ...]:
    if positions.shape != (2, 3, q**2, q**2):
        raise ValueError("Six-matrix response spectral receiver positions have another shape")
    x_operator = adjoint_laplacian_basis(positions[0])
    y_operator = adjoint_laplacian_basis(positions[1])
    outputs = []
    for sector, operator in (
        ("JOINT", x_operator + y_operator),
        ("X", x_operator),
        ("Y", y_operator),
    ):
        eigenvalues = laplacian_eigenvalues(operator)
        tolerance = kernel_relative_tolerance * max(1.0, float(eigenvalues[-1]))
        outputs.append(
            SixMatrixResponseLaplacianSpectrum(
                spectrum_id=f"{receiver_prefix}.{sector.lower()}",
                sector=sector,
                q=q,
                eigenvalues=tuple(_decimal(value) for value in eigenvalues),
                kernel_tolerance=_decimal(tolerance),
                kernel_dimension=int(np.count_nonzero(eigenvalues <= tolerance)),
                heat_trace_t1_normalized=_decimal(
                    sum(exp(-float(value)) for value in eigenvalues) / q**4
                ),
                finite=bool(np.isfinite(eigenvalues).all()),
                valid=bool(np.isfinite(eigenvalues).all()),
            )
        )
    return tuple(outputs)


__all__ = [
    'SixMatrixResponseLaplacianSpectrum',
    'adjoint_laplacian_basis',
    'adjoint_laplacian_explicit',
    'ideal_joint_laplacian_eigenvalues',
    'ideal_laplacian_eigenvalues',
    'laplacian_eigenvalues',
    'spectral_receiver',
]
