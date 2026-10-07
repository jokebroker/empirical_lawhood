# SPDX-License-Identifier: MPL-2.0
"""Rigorous threshold-rank brackets of the entered finite binary64 matrix.

The target is #{sigma_i(A) > tau}, where
tau = max(A.shape) * 2**-160 * max(1, ||A||_2).
All error propagation and comparisons stay in Arb. Exact rational rank can
only reduce the upper bound; it never promotes a subthreshold direction.
See docs/scientific-integrity.md for the enclosure argument and limitations.
"""

from dataclasses import dataclass
from decimal import Decimal
from threading import RLock

import numpy as np
import numpy.typing as npt
from flint import arb, arb_mat, ctx, fmpq, fmpq_mat

from ._arb import endpoint_decimal

ALGORITHM_ID = "arb-inertia-exact-dyadic-v2"
FloatArray = npt.NDArray[np.float64]

# python-flint exposes a shared precision context. Serialize this module's
# calls, including restoration on failure; mpmath/Decimal contexts are unused.
_ARB_CONTEXT_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class RankCertificate:
    lower: int
    upper: int
    residual: Decimal
    precision_bits: int
    tau_lower: Decimal
    tau_upper: Decimal


class _UnresolvedInertia(ArithmeticError):
    """Interval elimination could not certify a pivot sign."""


def _inertia_above(gram: arb_mat, threshold: arb) -> int:
    """Positive inertia of G-t**2 I by interval-enclosed LDL^T congruence.

    Every division uses a pivot whose interval excludes zero. A symmetric
    permutation preserves inertia. If scalar pivots cannot establish a sign,
    leave the rank unresolved (including exact equality at the threshold).
    """
    size = gram.nrows()
    shift = threshold * threshold
    shifted = [
        [gram[i, j] - (shift if i == j else 0) for j in range(size)]
        for i in range(size)
    ]
    positive = 0
    for k in range(size):
        candidates = [i for i in range(k, size) if not shifted[i][i].contains(0)]
        if not candidates:
            raise _UnresolvedInertia("no certified nonzero diagonal pivot")
        selected = max(candidates, key=lambda i: shifted[i][i].abs_lower())
        if selected != k:
            shifted[k], shifted[selected] = shifted[selected], shifted[k]
            for row in shifted:
                row[k], row[selected] = row[selected], row[k]
        pivot = shifted[k][k]
        if pivot > 0:
            positive += 1
        elif not pivot < 0:
            raise _UnresolvedInertia("pivot sign is unresolved")
        column = [shifted[i][k] for i in range(k + 1, size)]
        for i in range(k + 1, size):
            for j in range(i, size):
                entry = shifted[i][j] - column[i - k - 1] * column[j - k - 1] / pivot
                shifted[i][j] = entry
                shifted[j][i] = entry
    return positive


def _enclosures(entered: arb_mat) -> tuple[arb_mat, arb, arb, arb]:
    """Return Gram enclosure, threshold endpoints and a residual upper bound.

    For exact G and ball midpoints M, |G_ij-M_ij| <= radius_ij, hence
    ||G-M||_2 <= ||G-M||_F <= sqrt(sum radius_ij**2).
    The reported residual is an upward bound on the square root of this
    Gram error. It is not a QR or continuous-model reconstruction error.
    """
    rows, columns = entered.nrows(), entered.ncols()
    gram = entered.transpose() * entered if rows >= columns else entered * entered.transpose()
    norm_lower = arb(1)
    squared_frobenius = arb(0)
    for j in range(columns):
        squared_column = arb(0)
        for i in range(rows):
            squared_column += entered[i, j] * entered[i, j]
        norm_lower = max(norm_lower, squared_column.sqrt().lower())
        squared_frobenius += squared_column
    norm_upper = max(arb(1), squared_frobenius.sqrt().upper())
    factor = arb((max(rows, columns), -160))
    tau_lower = (factor * norm_lower).lower()
    tau_upper = (factor * norm_upper).upper()
    squared_radii = arb(0)
    for i in range(gram.nrows()):
        for j in range(gram.ncols()):
            radius = gram[i, j].rad()
            squared_radii += radius * radius
    residual = squared_radii.sqrt().sqrt().upper()
    return gram, tau_lower, tau_upper, residual


def binary64_matrix_snapshot(matrix: FloatArray) -> FloatArray:
    """Own one conversion of the input before validation or later computation.

    Inspect the captured dtype before binary64 conversion so complex inputs
    cannot lose their imaginary part. Copy even binary64 arrays: callers may
    mutate their storage while certification or record assembly proceeds.
    """
    captured = np.array(matrix, copy=True, order="C")
    if np.iscomplexobj(captured):
        raise ValueError("discrete-rank input matrix must be real")
    entered = np.asarray(captured, dtype=np.float64)
    if entered.ndim != 2 or not entered.size or not np.all(np.isfinite(entered)):
        raise ValueError("discrete-rank input matrix is invalid")
    entered.setflags(write=False)
    return entered


def certify_rank(matrix: FloatArray, *, precision_bits: int = 256) -> RankCertificate:
    """Enclose threshold rank; retry once, then retain conservative bounds.

    Inputs are real matrices explicitly entered as binary64. Any finite
    binary64 magnitude is supported, including subnormals. The precision
    budget controls resolution, never the validity of the bracket. A failed
    residual budget or unresolved pivot cannot certify a narrower rank.
    """
    if type(precision_bits) is not int or precision_bits < 256:
        raise ValueError("discrete rank requires at least 256 integer bits")
    entered_array = binary64_matrix_snapshot(matrix)
    rows, columns = entered_array.shape
    rationals = [fmpq(*float(value).as_integer_ratio()) for value in entered_array.ravel(order="C")]
    exact_rank = fmpq_mat(rows, columns, rationals).rank()
    start = max(precision_bits, 512 if min(rows, columns) >= 128 else 384)
    with _ARB_CONTEXT_LOCK:
        for bits in (start, start + 256):
            with ctx.workprec(bits):
                entered = arb_mat(rows, columns, [arb(value) for value in rationals])
                gram, tau_lower, tau_upper, residual = _enclosures(entered)
                lower, upper = 0, exact_rank
                budget_met = residual * 4 <= tau_lower
                if budget_met:
                    try:
                        lower = _inertia_above(gram, tau_upper)
                        upper = min(_inertia_above(gram, tau_lower), exact_rank)
                    except _UnresolvedInertia:
                        lower, upper = 0, exact_rank
                        budget_met = False
                if not 0 <= lower <= upper <= exact_rank:
                    raise ArithmeticError("Arb inertia contradicts exact dyadic rank")
                result = RankCertificate(
                    lower=lower,
                    upper=upper,
                    residual=endpoint_decimal(residual, upper=True),
                    precision_bits=bits,
                    tau_lower=endpoint_decimal(tau_lower, upper=False),
                    tau_upper=endpoint_decimal(tau_upper, upper=True),
                )
                if budget_met:
                    return result
    return result


def residual_certified_rank_bracket(
    matrix: FloatArray, *, precision_bits: int = 256,
) -> tuple[int, int, Decimal, Decimal | None]:
    """Return the rigorous bracket, Gram error bound and no separation claim."""
    result = certify_rank(matrix, precision_bits=precision_bits)
    return result.lower, result.upper, result.residual, None
