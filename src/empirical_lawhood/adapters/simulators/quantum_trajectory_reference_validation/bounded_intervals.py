"""Exact observable supports and frozen empirical-Bernstein intervals."""

from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal, localcontext
from hashlib import sha256
import math
from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .basis import FixedNumberBasis
from .operators import matrix_fingerprint
from .schemas import stable_json_bytes
from .types import BoundedInterval, ObservableSupport, Validity


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    payload = b"".join(
        (
            str(array.dtype).encode(),
            np.asarray(array.shape, dtype="<i8").tobytes(),
            array.tobytes(),
        )
    )
    return sha256(payload).hexdigest()


def _support(
    *,
    index: int,
    coordinate: str,
    lower: float,
    upper: float,
    operator_sha256: str,
    route: str,
) -> ObservableSupport:
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        raise ValueError(f"invalid support for {coordinate}")
    support_document = {
        "coordinate": coordinate,
        "index": index,
        "lower": lower,
        "operator_sha256": operator_sha256,
        "route": route,
        "upper": upper,
    }
    return ObservableSupport(
        index=index,
        coordinate=coordinate,
        lower=lower,
        upper=upper,
        operator_sha256=operator_sha256,
        support_sha256=sha256(stable_json_bytes(support_document)).hexdigest(),
        route=route,
    )


def observable_supports(
    basis: FixedNumberBasis,
    hamiltonian: csr_matrix,
) -> tuple[ObservableSupport, ...]:
    """Derive all 19 frozen supports from exact operators, never samples."""

    result: list[ObservableSupport] = []
    index = 0
    for site in range(basis.l_sites):
        diagonal = np.asarray(basis.occupations[:, site], dtype="<f8")
        result.append(
            _support(
                index=index,
                coordinate=f"n{site}",
                lower=0.0,
                upper=1.0,
                operator_sha256=_array_sha256(diagonal),
                route="analytic-projector-spectrum",
            )
        )
        index += 1
    for site in range(basis.l_sites):
        neighbor = (site + 1) % basis.l_sites
        diagonal = np.asarray(
            basis.occupations[:, site] * basis.occupations[:, neighbor],
            dtype="<f8",
        )
        result.append(
            _support(
                index=index,
                coordinate=f"nn{site}_{neighbor}",
                lower=0.0,
                upper=1.0,
                operator_sha256=_array_sha256(diagonal),
                route="analytic-projector-product-spectrum",
            )
        )
        index += 1
    eigenvalues = np.linalg.eigvalsh(hamiltonian.toarray())
    result.append(
        _support(
            index=index,
            coordinate="energy",
            lower=float(eigenvalues[0]),
            upper=float(eigenvalues[-1]),
            operator_sha256=matrix_fingerprint(hamiltonian),
            route="hermitian-spectral-extrema",
        )
    )
    index += 1
    angles = 2.0 * np.pi * np.arange(basis.l_sites) / basis.l_sites
    for name, weights in (
        ("fourier_cos_1", np.cos(angles)),
        ("fourier_sin_1", np.sin(angles)),
    ):
        diagonal = np.asarray(basis.occupations @ weights, dtype="<f8")
        result.append(
            _support(
                index=index,
                coordinate=name,
                lower=float(np.min(diagonal)),
                upper=float(np.max(diagonal)),
                operator_sha256=_array_sha256(diagonal),
                route="fixed-number-basis-diagonal-extrema",
            )
        )
        index += 1
    expected = 2 * basis.l_sites + 3
    if len(result) != expected:
        raise AssertionError("observable support family is incomplete")
    return tuple(result)


def density_observables(
    basis: FixedNumberBasis,
    hamiltonian: csr_matrix,
    density: np.ndarray,
) -> np.ndarray:
    density_array = np.asarray(density, dtype=np.complex128)
    if density_array.shape != (basis.dimension, basis.dimension):
        raise ValueError("density shape differs from the fixed-number basis")
    diagonal = np.real(np.diag(density_array))
    site_values = diagonal @ basis.occupations
    pair_values = np.asarray(
        [
            diagonal
            @ (basis.occupations[:, site] * basis.occupations[:, (site + 1) % basis.l_sites])
            for site in range(basis.l_sites)
        ],
        dtype=np.float64,
    )
    energy = float(np.trace(density_array @ hamiltonian.toarray()).real)
    angles = 2.0 * np.pi * np.arange(basis.l_sites) / basis.l_sites
    return np.concatenate(
        (
            np.asarray(site_values, dtype=np.float64),
            pair_values,
            np.asarray(
                [
                    energy,
                    float(site_values @ np.cos(angles)),
                    float(site_values @ np.sin(angles)),
                ],
                dtype=np.float64,
            ),
        )
    )


def state_observables(
    basis: FixedNumberBasis,
    hamiltonian: csr_matrix,
    state: np.ndarray,
) -> np.ndarray:
    state_array = np.asarray(state, dtype=np.complex128)
    if state_array.shape != (basis.dimension,):
        raise ValueError("state shape differs from the fixed-number basis")
    return density_observables(
        basis,
        hamiltonian,
        np.outer(state_array, state_array.conj()),
    )


def empirical_bernstein_halfwidth(
    normalized_variance: np.ndarray | float,
    *,
    sample_size: int,
    alpha: float,
    family_cells: int,
    family_looks: int,
) -> np.ndarray:
    if sample_size < 2:
        raise ValueError("empirical-Bernstein intervals require n >= 2")
    if not (0.0 < alpha < 1.0) or family_cells <= 0 or family_looks <= 0:
        raise ValueError("invalid simultaneous-family operands")
    variance = np.asarray(normalized_variance, dtype=np.float64)
    if np.any(~np.isfinite(variance)) or np.any(variance < -1e-15):
        raise ValueError("normalized variance is invalid")
    variance = np.maximum(variance, 0.0)
    delta = alpha / (family_cells * family_looks)
    logarithm = math.log(2.0 / delta)
    return np.sqrt(2.0 * variance * logarithm / sample_size) + (
        7.0 * logarithm / (3.0 * (sample_size - 1))
    )


def scalar_decimal_halfwidth(
    normalized_values: Sequence[Decimal],
    *,
    alpha: Decimal,
    family_cells: int,
    family_looks: int,
) -> Decimal:
    "Readable high-precision reference for deterministic bounded-interval fixtures."

    sample_size = len(normalized_values)
    if sample_size < 2:
        raise ValueError("empirical-Bernstein intervals require n >= 2")
    with localcontext() as context:
        context.prec = 80
        mean = sum(normalized_values, Decimal(0)) / Decimal(sample_size)
        variance = sum(
            ((value - mean) * (value - mean) for value in normalized_values),
            Decimal(0),
        ) / Decimal(sample_size - 1)
        delta = alpha / Decimal(family_cells * family_looks)
        logarithm = (Decimal(2) / delta).ln()
        return (Decimal(2) * variance * logarithm / Decimal(sample_size)).sqrt() + (
            Decimal(7) * logarithm / (Decimal(3) * Decimal(sample_size - 1))
        )


def intervals_for_observations(
    observations: np.ndarray,
    supports: Sequence[ObservableSupport],
    *,
    alpha: float,
    family_cells: int,
    family_looks: int,
) -> tuple[BoundedInterval, ...]:
    values = np.asarray(observations, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(supports):
        raise ValueError("observation array does not match the support family")
    if values.shape[0] < 2:
        raise ValueError("at least two observations are required")
    intervals: list[BoundedInterval] = []
    delta = alpha / (family_cells * family_looks)
    support_tolerance = 64.0 * np.finfo(np.float64).eps
    for column, support in enumerate(supports):
        sample = values[:, column]
        width = support.width
        if np.any(~np.isfinite(sample)):
            intervals.append(
                BoundedInterval(
                    coordinate=support.coordinate,
                    sample_size=len(sample),
                    estimate=math.nan,
                    unbiased_variance_normalized=math.nan,
                    delta=delta,
                    normalized_halfwidth=math.inf,
                    native_halfwidth=math.inf,
                    lower_unclipped=-math.inf,
                    upper_unclipped=math.inf,
                    lower_display=support.lower,
                    upper_display=support.upper,
                    support_lower=support.lower,
                    support_upper=support.upper,
                    validity=Validity.UNEVALUABLE,
                    reason_code="NONFINITE_OBSERVATION",
                )
            )
            continue
        if np.any(sample < support.lower - support_tolerance) or np.any(
            sample > support.upper + support_tolerance
        ):
            intervals.append(
                BoundedInterval(
                    coordinate=support.coordinate,
                    sample_size=len(sample),
                    estimate=float(np.mean(sample)),
                    unbiased_variance_normalized=math.nan,
                    delta=delta,
                    normalized_halfwidth=math.inf,
                    native_halfwidth=math.inf,
                    lower_unclipped=-math.inf,
                    upper_unclipped=math.inf,
                    lower_display=support.lower,
                    upper_display=support.upper,
                    support_lower=support.lower,
                    support_upper=support.upper,
                    validity=Validity.UNEVALUABLE,
                    reason_code="SUPPORT_VIOLATION",
                )
            )
            continue
        estimate = float(np.mean(sample, dtype=np.float64))
        if width == 0.0:
            constant = bool(np.all(sample == support.lower))
            intervals.append(
                BoundedInterval(
                    coordinate=support.coordinate,
                    sample_size=len(sample),
                    estimate=estimate,
                    unbiased_variance_normalized=0.0,
                    delta=delta,
                    normalized_halfwidth=0.0 if constant else math.inf,
                    native_halfwidth=0.0 if constant else math.inf,
                    lower_unclipped=support.lower if constant else -math.inf,
                    upper_unclipped=support.upper if constant else math.inf,
                    lower_display=support.lower,
                    upper_display=support.upper,
                    support_lower=support.lower,
                    support_upper=support.upper,
                    validity=Validity.VALID if constant else Validity.UNEVALUABLE,
                    reason_code="EXACT_CONSTANT" if constant else "CONSTANT_RANGE_VIOLATION",
                )
            )
            continue
        normalized = (sample - support.lower) / width
        variance = float(np.var(normalized, ddof=1, dtype=np.float64))
        normalized_halfwidth = float(
            empirical_bernstein_halfwidth(
                variance,
                sample_size=len(sample),
                alpha=alpha,
                family_cells=family_cells,
                family_looks=family_looks,
            )
        )
        native_halfwidth = width * normalized_halfwidth
        lower = estimate - native_halfwidth
        upper = estimate + native_halfwidth
        intervals.append(
            BoundedInterval(
                coordinate=support.coordinate,
                sample_size=len(sample),
                estimate=estimate,
                unbiased_variance_normalized=variance,
                delta=delta,
                normalized_halfwidth=normalized_halfwidth,
                native_halfwidth=native_halfwidth,
                lower_unclipped=lower,
                upper_unclipped=upper,
                lower_display=max(lower, support.lower),
                upper_display=min(upper, support.upper),
                support_lower=support.lower,
                support_upper=support.upper,
                validity=Validity.VALID,
                reason_code="OK",
            )
        )
    return tuple(intervals)


def support_rows(supports: Sequence[ObservableSupport]) -> list[dict[str, object]]:
    return [asdict(support) for support in supports]


__all__ = [
    "density_observables",
    "empirical_bernstein_halfwidth",
    "intervals_for_observations",
    "observable_supports",
    "scalar_decimal_halfwidth",
    "state_observables",
    "support_rows",
]
