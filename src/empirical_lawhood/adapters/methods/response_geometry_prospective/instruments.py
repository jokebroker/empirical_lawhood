"""Finite factor and causal-observation diagnostics for the Response geometry assay.

These operands qualify measurements. They neither construct a response law nor
grant support, admission, actuation authority or an experiment adjudication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from empirical_lawhood.adapters.simulators.six_matrix_response.model import ComplexArray
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import traceless_operator
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import traceless_hermitian_basis
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import ResponseGeometryNativeFactorMetrics, ResponseGeometryNativeGeometrySample


@dataclass(frozen=True, slots=True)
class FactorDiagnostic:
    boundary_gap: float
    gap_floor: float
    slow_basis: ComplexArray | None
    product_defect: float | None
    adjoint_defect: float | None
    center_dimension: int | None
    commutant_dimension: int | None
    candidate: bool | None


def _nullity(matrix: ComplexArray, *, relative_floor: float) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    return int(matrix.shape[1]) - int(
        np.sum(singular > relative_floor * max(1.0, float(singular[0])))
    )


def factor_diagnostic(y: ComplexArray) -> FactorDiagnostic:
    """Test identity plus the resolved slow triplet as a complex *-algebra.

    The normalized Frobenius norm of the complete product-residual tensor is
    independent of the orthonormal basis selected within the triplet. A small
    defect nominates an approximate factor; it is not proof of exact closure.
    """
    operator = traceless_operator(y)
    values, vectors = np.linalg.eigh(operator)
    gap = float(values[3] - values[2])
    floor = 1e-6 * max(1.0, float(np.linalg.norm(operator, 2)))
    if gap <= floor:
        return FactorDiagnostic(gap, floor, None, None, None, None, None, None)
    slow = np.einsum("ki,kab->iab", vectors[:, :3], traceless_hermitian_basis(4))
    basis = np.concatenate((np.eye(4, dtype="<c16")[None] / 2, slow), axis=0)
    columns = basis.reshape(4, 16).T
    products = np.einsum("iab,jbc->ijac", basis, basis).reshape(16, 16).T
    adjoints = basis.conj().transpose(0, 2, 1).reshape(4, 16).T
    product_defect = float(
        np.linalg.norm(products - columns @ (columns.conj().T @ products))
        / np.linalg.norm(products)
    )
    adjoint_defect = float(
        np.linalg.norm(adjoints - columns @ (columns.conj().T @ adjoints))
        / np.linalg.norm(adjoints)
    )
    commutators = np.stack(
        [np.kron(matrix, np.eye(4)) - np.kron(np.eye(4), matrix.T) for matrix in basis]
    ).reshape(64, 16)
    center_dimension = _nullity(commutators @ columns, relative_floor=1e-6)
    commutant_dimension = _nullity(commutators, relative_floor=1e-6)
    candidate = (
        product_defect <= 1e-6
        and adjoint_defect <= 1e-6
        and center_dimension == 1
        and commutant_dimension == 4
    )
    frozen = np.frombuffer(slow.tobytes(), dtype="<c16").reshape(3, 4, 4)
    return FactorDiagnostic(
        gap,
        floor,
        frozen,
        product_defect,
        adjoint_defect,
        center_dimension,
        commutant_dimension,
        candidate,
    )


def slow_projector_distance(left: ComplexArray, right: ComplexArray) -> float:
    """Operator distance of two rank-three HS projectors in the SAME frame."""
    for value in (left, right):
        if value.shape != (3, 4, 4) or not np.isfinite(value).all():
            raise ValueError("Response geometry projector basis must have finite rank-three geometry")
        columns = value.reshape(3, 16).T
        if np.linalg.norm(columns.conj().T @ columns - np.eye(3)) > 1e-10:
            raise ValueError("Response geometry projector basis must be HS orthonormal")
    a, b = left.reshape(3, 16).T, right.reshape(3, 16).T
    return float(np.linalg.norm(a @ a.conj().T - b @ b.conj().T, 2))


@dataclass(frozen=True, slots=True)
class CausalGeometry:
    full_g: bool | None
    robust_00: bool | None
    label: str


def structural_window(
    samples: tuple[ResponseGeometryNativeGeometrySample, ...],
) -> tuple[bool | None, bool | None, bool | None]:
    """Original 12-of-16 radius/closure and endpoint/all-window kernel rules."""
    if not samples:
        return None, None, None
    expected = tuple(range(samples[-1].reference_tick - 240, samples[-1].reference_tick + 1, 16))
    if len(samples) != 16:
        return None, None, None
    if tuple(value.reference_tick for value in samples) != expected:
        raise ValueError("Response geometry structural window must use sixteen consecutive causal samples")
    if not all(value.x.known and value.y.known for value in samples):
        return None, None, None

    def radius_closure(value: ResponseGeometryNativeFactorMetrics) -> bool:
        assert value.phi is not None and value.closure_ratio is not None
        return 0.35 <= float(value.phi) <= 0.95 and float(value.closure_ratio) <= 0.30

    flags = []
    for factors in (tuple(value.x for value in samples), tuple(value.y for value in samples)):
        last = factors[-1]
        assert last.kernel_ratio is not None
        flags.append(
            radius_closure(last)
            and float(last.kernel_ratio) <= 0.25
            and sum(radius_closure(value) for value in factors) >= 12
        )
    y_strict = flags[1] and all(
        value.y.kernel_ratio is not None and float(value.y.kernel_ratio) <= 0.25
        for value in samples
    )
    return flags[0], flags[1], y_strict


def causal_geometry(
    *,
    x_geometric: bool | None,
    y_geometric: bool | None,
    y_strict_window: bool | None,
    passive_pass: bool | None,
    acquisition_tick: int,
    evidence_ready_tick: int,
    decision_tick: int,
) -> CausalGeometry:
    """Combine already measured rolling predicates with explicit unknowns.

    All operands must be available at the proposed decision. Strict Y includes
    the declared 16-sample persistence/kernel window; the mature passive operand
    includes prefix invariants and resolved normalization. Raw predicates survive
    even if they overlap. No truth is inferred from absent/late measurements.
    """
    ticks = (acquisition_tick, evidence_ready_tick, decision_tick)
    if any(type(tick) is not int or tick < 0 for tick in ticks):
        raise ValueError("Response geometry geometry clocks must be nonnegative reference ticks")
    if evidence_ready_tick < acquisition_tick:
        raise ValueError("Response geometry evidence cannot precede its acquisition")
    operands = (x_geometric, y_geometric, y_strict_window, passive_pass)
    if any(value is not None and type(value) is not bool for value in operands):
        raise ValueError("Response geometry geometry predicates must be measured bool or unknown")
    if evidence_ready_tick > decision_tick:
        return CausalGeometry(None, None, "U")
    # Require every full-G operand to be known, including when a different
    # operand is false: invalid observation is distinct from observed absence.
    full = (
        None
        if any(value is None for value in (x_geometric, y_strict_window, passive_pass))
        else (not x_geometric and bool(y_strict_window) and bool(passive_pass))
    )
    robust = (
        None
        if x_geometric is None or y_geometric is None
        else (not x_geometric and not y_geometric)
    )
    label = "G" if full is True else "N" if robust is True and full is False else "U"
    return CausalGeometry(full, robust, label)
