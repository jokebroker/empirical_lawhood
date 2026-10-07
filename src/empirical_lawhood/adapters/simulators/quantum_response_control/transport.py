"""Pathwise coherent-current and measurement-innovation accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.sparse import csr_matrix

from .basis import ComplexVector, Propagator


@dataclass(frozen=True, slots=True)
class SegmentTransport:
    signed_current: float
    absolute_current: float
    duration: float


@dataclass(frozen=True, slots=True)
class TransportSummary:
    coherent_signed: float
    coherent_absolute: float
    measurement_innovation: float
    q_start: float
    q_end: float
    continuity_residual: float
    refinement_error_signed: float
    refinement_error_absolute: float
    valid: bool


def q_a(state: ComplexVector, half_count: np.ndarray) -> float:
    weights = np.abs(state) ** 2
    value = float(weights @ half_count)
    if not math.isfinite(value):
        raise ValueError("receiver occupation is nonfinite")
    return value


def integrate_current_segment(
    *,
    propagator: Propagator,
    state: ComplexVector,
    duration: float,
    current_operator: csr_matrix,
    order: int,
) -> SegmentTransport:
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("transport segment duration is invalid")
    if order < 2:
        raise ValueError("quadrature order must be at least two")
    if duration == 0:
        return SegmentTransport(0.0, 0.0, 0.0)
    nodes, weights = leggauss(order)
    times = 0.5 * duration * (nodes + 1.0)
    values = np.empty(order, dtype=np.float64)
    for index, time_value in enumerate(times):
        local = propagator.propagate(state, float(time_value))
        expectation = complex(np.vdot(local, current_operator @ local))
        if abs(expectation.imag) > 1e-9:
            raise ValueError("current expectation is materially complex")
        values[index] = expectation.real
    scale = 0.5 * duration
    return SegmentTransport(
        signed_current=float(scale * np.dot(weights, values)),
        absolute_current=float(scale * np.dot(weights, np.abs(values))),
        duration=duration,
    )


def integrate_current_segment_refined(
    *,
    propagator: Propagator,
    state: ComplexVector,
    duration: float,
    current_operator: csr_matrix,
    production_order: int,
    sentinel_order: int,
    tolerance: float,
    maximum_depth: int = 16,
) -> tuple[SegmentTransport, SegmentTransport]:
    """Adaptively partition a segment until the frozen 8/16 sentinel closes.

    The absolute-current integrand has cusps where the signed expectation
    crosses zero.  Merely increasing Gauss--Legendre order on the whole
    inter-jump segment does not reliably resolve those cusps.  Deterministic
    bisection preserves the declared production/sentinel orders while making
    the refinement test meaningful for both signed and absolute transport.
    """

    if tolerance <= 0 or not math.isfinite(tolerance):
        raise ValueError("quadrature refinement tolerance is invalid")
    if maximum_depth < 0:
        raise ValueError("quadrature refinement depth is invalid")

    def recurse(
        local_state: ComplexVector,
        local_duration: float,
        local_tolerance: float,
        depth: int,
    ) -> tuple[SegmentTransport, SegmentTransport]:
        production = integrate_current_segment(
            propagator=propagator,
            state=local_state,
            duration=local_duration,
            current_operator=current_operator,
            order=production_order,
        )
        sentinel = integrate_current_segment(
            propagator=propagator,
            state=local_state,
            duration=local_duration,
            current_operator=current_operator,
            order=sentinel_order,
        )
        effective_tolerance = max(
            local_tolerance,
            # Repeated sparse Krylov propagation makes sub-picounit
            # 8/16 differences roundoff-dominated well before the requested
            # path-level 1e-8 tolerance.  This local floor remains four orders
            # below that path-level gate even across O(10^3) leaves.
            1e-12,
        )
        closed = (
            abs(production.signed_current - sentinel.signed_current) <= effective_tolerance
            and abs(production.absolute_current - sentinel.absolute_current) <= effective_tolerance
        )
        if closed:
            return production, sentinel
        if depth >= maximum_depth:
            raise RuntimeError("quadrature refinement did not converge")
        half = 0.5 * local_duration
        left = recurse(local_state, half, 0.5 * local_tolerance, depth + 1)
        midpoint = propagator.propagate(local_state, half)
        right = recurse(midpoint, half, 0.5 * local_tolerance, depth + 1)
        return (
            SegmentTransport(
                left[0].signed_current + right[0].signed_current,
                left[0].absolute_current + right[0].absolute_current,
                local_duration,
            ),
            SegmentTransport(
                left[1].signed_current + right[1].signed_current,
                left[1].absolute_current + right[1].absolute_current,
                local_duration,
            ),
        )

    return recurse(state, duration, tolerance, 0)


def finalize_transport(
    *,
    coherent_signed: float,
    coherent_absolute: float,
    measurement_innovation: float,
    q_start_value: float,
    q_end_value: float,
    sentinel_signed: float,
    sentinel_absolute: float,
    continuity_tolerance: float,
) -> TransportSummary:
    values = (
        coherent_signed,
        coherent_absolute,
        measurement_innovation,
        q_start_value,
        q_end_value,
        sentinel_signed,
        sentinel_absolute,
        continuity_tolerance,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("transport summary contains a nonfinite operand")
    residual = q_end_value - q_start_value - coherent_signed - measurement_innovation
    refinement_signed = abs(sentinel_signed - coherent_signed)
    refinement_absolute = abs(sentinel_absolute - coherent_absolute)
    valid = (
        abs(residual) <= continuity_tolerance
        and refinement_signed <= continuity_tolerance
        and refinement_absolute <= continuity_tolerance
        and coherent_absolute + continuity_tolerance >= abs(coherent_signed)
    )
    return TransportSummary(
        coherent_signed=coherent_signed,
        coherent_absolute=coherent_absolute,
        measurement_innovation=measurement_innovation,
        q_start=q_start_value,
        q_end=q_end_value,
        continuity_residual=residual,
        refinement_error_signed=refinement_signed,
        refinement_error_absolute=refinement_absolute,
        valid=valid,
    )


__all__ = [
    "SegmentTransport",
    "TransportSummary",
    "finalize_transport",
    "integrate_current_segment",
    "integrate_current_segment_refined",
    "q_a",
]
