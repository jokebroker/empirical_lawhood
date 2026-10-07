"""Exact receiver-geometry reference fixtures for RICQ.

The nominated Keller construction is treated only as a truth-known stressor for
local-versus-global receiver identifiability.  This module has no simulator,
filesystem, network or controller authority.  Its two construction routes are
deliberately independent:

* a sparse-polynomial expansion verifies the explicit map and its Jacobian;
* an incidence/factorization chart verifies the normalized resultant variety
  and reconstructs the same map through cubic multiplication.

All claim-bearing identities use exact :class:`fractions.Fraction` arithmetic.
Numerical continuation belongs to the receiver-geometry method adapter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations
from typing import Final, TypeAlias


Rational: TypeAlias = Fraction
Exponent: TypeAlias = tuple[int, int, int]


def _q(value: int | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(value)


@dataclass(frozen=True, slots=True)
class SparsePolynomial3:
    """A dependency-free exact polynomial in three ordered variables."""

    terms: tuple[tuple[Exponent, Fraction], ...]

    def __post_init__(self) -> None:
        normalized = tuple(
            sorted(
                ((exponent, coefficient) for exponent, coefficient in self.terms if coefficient),
                key=lambda value: value[0],
            )
        )
        if normalized != self.terms:
            raise ValueError("sparse polynomial terms must be sorted, unique and nonzero")
        if len({exponent for exponent, _coefficient in self.terms}) != len(self.terms):
            raise ValueError("sparse polynomial repeats an exponent")
        if any(len(exponent) != 3 or min(exponent) < 0 for exponent, _ in self.terms):
            raise ValueError("sparse polynomial has an invalid exponent")

    @classmethod
    def constant(cls, value: int | Fraction) -> SparsePolynomial3:
        coefficient = _q(value)
        return cls(()) if coefficient == 0 else cls((((0, 0, 0), coefficient),))

    @classmethod
    def variable(cls, index: int) -> SparsePolynomial3:
        if index not in {0, 1, 2}:
            raise ValueError("polynomial variable index is outside three variables")
        exponent = ((1, 0, 0), (0, 1, 0), (0, 0, 1))[index]
        return cls(((exponent, Fraction(1)),))

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[Exponent, int | Fraction],
    ) -> SparsePolynomial3:
        return cls(
            tuple(
                sorted(
                    (
                        (exponent, _q(coefficient))
                        for exponent, coefficient in values.items()
                        if coefficient
                    ),
                    key=lambda value: value[0],
                )
            )
        )

    def _mapping(self) -> dict[Exponent, Fraction]:
        return dict(self.terms)

    def __add__(self, other: SparsePolynomial3 | int) -> SparsePolynomial3:
        right = other if isinstance(other, SparsePolynomial3) else SparsePolynomial3.constant(other)
        values = self._mapping()
        for exponent, coefficient in right.terms:
            values[exponent] = values.get(exponent, Fraction(0)) + coefficient
        return SparsePolynomial3.from_mapping(values)

    def __radd__(self, other: int) -> SparsePolynomial3:
        return self + other

    def __neg__(self) -> SparsePolynomial3:
        return SparsePolynomial3(
            tuple((exponent, -coefficient) for exponent, coefficient in self.terms)
        )

    def __sub__(self, other: SparsePolynomial3 | int) -> SparsePolynomial3:
        right = other if isinstance(other, SparsePolynomial3) else SparsePolynomial3.constant(other)
        return self + (-right)

    def __rsub__(self, other: int) -> SparsePolynomial3:
        return SparsePolynomial3.constant(other) - self

    def __mul__(self, other: SparsePolynomial3 | int | Fraction) -> SparsePolynomial3:
        right = other if isinstance(other, SparsePolynomial3) else SparsePolynomial3.constant(other)
        values: dict[Exponent, Fraction] = {}
        for left_exponent, left_coefficient in self.terms:
            for right_exponent, right_coefficient in right.terms:
                exponent: Exponent = (
                    left_exponent[0] + right_exponent[0],
                    left_exponent[1] + right_exponent[1],
                    left_exponent[2] + right_exponent[2],
                )
                values[exponent] = (
                    values.get(exponent, Fraction(0)) + left_coefficient * right_coefficient
                )
        return SparsePolynomial3.from_mapping(values)

    def __rmul__(self, other: int | Fraction) -> SparsePolynomial3:
        return self * other

    def __pow__(self, exponent: int) -> SparsePolynomial3:
        if exponent < 0:
            raise ValueError("polynomial exponent must be nonnegative")
        result = SparsePolynomial3.constant(1)
        factor = self
        remaining = exponent
        while remaining:
            if remaining & 1:
                result = result * factor
            factor = factor * factor
            remaining >>= 1
        return result

    def derivative(self, index: int) -> SparsePolynomial3:
        if index not in {0, 1, 2}:
            raise ValueError("polynomial derivative index is outside three variables")
        values: dict[Exponent, Fraction] = {}
        for exponent, coefficient in self.terms:
            power = exponent[index]
            if power == 0:
                continue
            derived = list(exponent)
            derived[index] -= 1
            derived_exponent: Exponent = (derived[0], derived[1], derived[2])
            values[derived_exponent] = coefficient * power
        return SparsePolynomial3.from_mapping(values)

    def evaluate(self, x: Fraction, y: Fraction, z: Fraction) -> Fraction:
        coordinates = (x, y, z)
        return sum(
            (
                coefficient
                * coordinates[0] ** exponent[0]
                * coordinates[1] ** exponent[1]
                * coordinates[2] ** exponent[2]
                for exponent, coefficient in self.terms
            ),
            Fraction(0),
        )


def determinant3(
    rows: tuple[
        tuple[SparsePolynomial3, SparsePolynomial3, SparsePolynomial3],
        tuple[SparsePolynomial3, SparsePolynomial3, SparsePolynomial3],
        tuple[SparsePolynomial3, SparsePolynomial3, SparsePolynomial3],
    ],
) -> SparsePolynomial3:
    """Return the exact determinant without relying on symbolic libraries."""

    total = SparsePolynomial3.constant(0)
    for order in permutations(range(3)):
        inversions = sum(
            1 for left in range(3) for right in range(left + 1, 3) if order[left] > order[right]
        )
        term = SparsePolynomial3.constant(-1 if inversions % 2 else 1)
        for row, column in enumerate(order):
            term = term * rows[row][column]
        total = total + term
    return total


def keller_polynomial_map() -> tuple[
    SparsePolynomial3,
    SparsePolynomial3,
    SparsePolynomial3,
]:
    """Return the explicit three-variable map as exact sparse polynomials."""

    x = SparsePolynomial3.variable(0)
    y = SparsePolynomial3.variable(1)
    z = SparsePolynomial3.variable(2)
    one_plus_xy = 1 + x * y
    return (
        one_plus_xy**3 * z + y**2 * one_plus_xy * (4 + 3 * x * y),
        y + 3 * x * one_plus_xy**2 * z + 3 * x * y**2 * (4 + 3 * x * y),
        2 * x - 3 * x**2 * y - x**3 * z,
    )


def keller_jacobian_determinant() -> SparsePolynomial3:
    """Mechanically expand the exact Jacobian determinant."""

    coordinate_polynomials = keller_polynomial_map()
    matrix = tuple(
        tuple(polynomial.derivative(index) for index in range(3))
        for polynomial in coordinate_polynomials
    )
    return determinant3(matrix)  # type: ignore[arg-type]


def keller_map(
    x: Fraction,
    y: Fraction,
    z: Fraction,
) -> tuple[Fraction, Fraction, Fraction]:
    """Evaluate the explicit map directly over the rationals."""

    one_plus_xy = 1 + x * y
    return (
        one_plus_xy**3 * z + y**2 * one_plus_xy * (4 + 3 * x * y),
        y + 3 * x * one_plus_xy**2 * z + 3 * x * y**2 * (4 + 3 * x * y),
        2 * x - 3 * x**2 * y - x**3 * z,
    )


@dataclass(frozen=True, slots=True)
class IncidenceChartPoint:
    """Normalized linear/quadratic factor pair in the affine chart."""

    a: Fraction
    b: Fraction
    c: Fraction
    d: Fraction
    e: Fraction

    @property
    def resultant(self) -> Fraction:
        return self.a**2 * self.e - self.a * self.b * self.d + self.c * self.b**2

    @property
    def slice_coordinate(self) -> Fraction:
        return self.a * self.d + self.b * self.c

    @property
    def cubic_coefficients(self) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        return (
            self.a * self.c,
            self.a * self.d + self.b * self.c,
            self.a * self.e + self.b * self.d,
            self.b * self.e,
        )


def incidence_chart(a: Fraction, y: Fraction, z: Fraction) -> IncidenceChartPoint:
    """Parameterize the normalized resultant slice by three affine coordinates."""

    b = 1 + a * y
    c = 1 - Fraction(3, 2) * a * y + a**2 * z
    d = Fraction(1, 2) * y - a * z + Fraction(3, 2) * a * y**2 - a**2 * y * z
    e = -2 * z + 4 * y**2 - 4 * a * y * z + 3 * a * y**3 - 2 * a**2 * y**2 * z
    point = IncidenceChartPoint(a=a, b=b, c=c, d=d, e=e)
    if point.resultant != 1 or point.slice_coordinate != 1:
        raise AssertionError("incidence chart failed its defining exact identities")
    return point


def incidence_keller_map(
    x: Fraction,
    y: Fraction,
    z: Fraction,
) -> tuple[Fraction, Fraction, Fraction]:
    """Reconstruct the explicit map through the independent incidence chart."""

    f, g, h, i = incidence_chart(x, y, -z / 2).cubic_coefficients
    if g != 1:
        raise AssertionError("incidence cubic escaped its affine target slice")
    return i, 2 * h, 2 * f


def marked_root_fiber_polynomial(
    target: tuple[Fraction, Fraction, Fraction],
) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    """Return coefficients of ``C*T^3 + 2*T^2 + B*T + 2*A``."""

    a, b, c = target
    return c, Fraction(2), b, 2 * a


def evaluate_cubic(
    coefficients: tuple[Fraction, Fraction, Fraction, Fraction],
    value: Fraction,
) -> Fraction:
    c3, c2, c1, c0 = coefficients
    return ((c3 * value + c2) * value + c1) * value + c0


def marked_root_for_preimage(
    preimage: tuple[Fraction, Fraction, Fraction],
) -> Fraction | None:
    """Return the affine marked root; ``None`` represents the chart boundary."""

    x, y, _z = preimage
    return None if x == 0 else -(1 + x * y) / x


def cubic_discriminant(
    coefficients: tuple[Fraction, Fraction, Fraction, Fraction],
) -> Fraction:
    """Return the exact discriminant of a cubic, including degenerate cubics."""

    a, b, c, d = coefficients
    return b**2 * c**2 - 4 * a * c**3 - 4 * b**3 * d - 27 * a**2 * d**2 + 18 * a * b * c * d


def keller_discriminant_locus_value(
    target: tuple[Fraction, Fraction, Fraction],
) -> Fraction:
    """Return the compact target-locus polynomial whose negative fourfold is Δ."""

    a, b, c = target
    return 27 * a**2 * c**2 - 18 * a * b * c + 16 * a + b**3 * c - b**2


KELLER_WITNESS_PREIMAGES: Final = (
    (Fraction(0), Fraction(0), Fraction(-1, 4)),
    (Fraction(1), Fraction(-3, 2), Fraction(13, 2)),
    (Fraction(-1), Fraction(3, 2), Fraction(13, 2)),
)
KELLER_WITNESS_TARGET: Final = (Fraction(-1, 4), Fraction(0), Fraction(0))


def verify_keller_exact_fixture() -> tuple[str, ...]:
    """Run the dependency-light exact conformance checks and return fact IDs."""

    determinant = keller_jacobian_determinant()
    if determinant != SparsePolynomial3.constant(-2):
        raise AssertionError("explicit Keller map does not have constant Jacobian -2")
    for preimage in KELLER_WITNESS_PREIMAGES:
        if keller_map(*preimage) != KELLER_WITNESS_TARGET:
            raise AssertionError("declared Keller collision does not reproduce")
        if incidence_keller_map(*preimage) != KELLER_WITNESS_TARGET:
            raise AssertionError("incidence reconstruction does not reproduce the collision")
        marked_root = marked_root_for_preimage(preimage)
        if marked_root is not None and evaluate_cubic(
            marked_root_fiber_polynomial(KELLER_WITNESS_TARGET),
            marked_root,
        ):
            raise AssertionError("preimage marked root misses the target fiber polynomial")
    if cubic_discriminant(marked_root_fiber_polynomial(KELLER_WITNESS_TARGET)) != (
        -4 * keller_discriminant_locus_value(KELLER_WITNESS_TARGET)
    ):
        raise AssertionError("fiber discriminant differs from the target-locus identity")
    return (
        "INCIDENCE_CHART_RESULTANT_ONE",
        "INCIDENCE_TARGET_SLICE_G_ONE",
        "JACOBIAN_CONSTANT_MINUS_TWO",
        "MARKED_ROOT_FIBER_POLYNOMIAL",
        "THREE_DECLARED_PREIMAGES_COLLIDE",
        "TWO_INDEPENDENT_EXACT_CONSTRUCTIONS_AGREE",
    )


__all__ = [
    "IncidenceChartPoint",
    "KELLER_WITNESS_PREIMAGES",
    "KELLER_WITNESS_TARGET",
    "SparsePolynomial3",
    "cubic_discriminant",
    "evaluate_cubic",
    "incidence_chart",
    "incidence_keller_map",
    "keller_discriminant_locus_value",
    "keller_jacobian_determinant",
    "keller_map",
    "keller_polynomial_map",
    "marked_root_fiber_polynomial",
    "marked_root_for_preimage",
    "verify_keller_exact_fixture",
]
