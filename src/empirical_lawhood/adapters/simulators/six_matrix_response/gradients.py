"""Independent action implementations and termwise analytic Six-matrix response gradients."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import ComplexArray, hermitian_part


def commutator(left: ComplexArray, right: ComplexArray) -> ComplexArray:
    return left @ right - right @ left


@dataclass(frozen=True, slots=True)
class SixMatrixParameters:
    q: int
    mass_x: float
    mass_y: float
    gamma: float
    alpha_tilde_x: float
    alpha_tilde_y: float

    def __post_init__(self) -> None:
        if self.q not in {2, 3, 4}:
            raise ValueError("Six-matrix response parameters q must be one of 2, 3 or 4")
        if min(self.mass_x, self.mass_y, self.gamma) <= 0:
            raise ValueError("Six-matrix response masses and cross coupling must be positive")
        if min(self.alpha_tilde_x, self.alpha_tilde_y) < 0:
            raise ValueError("Six-matrix response scaled couplings cannot be negative")

    @property
    def n(self) -> int:
        return self.q**2

    @property
    def c2(self) -> float:
        return (self.q**2 - 1.0) / 4.0

    @property
    def alpha_x(self) -> float:
        return self.alpha_tilde_x / self.q

    @property
    def alpha_y(self) -> float:
        return self.alpha_tilde_y / self.q

    def mu(self, mass: float) -> float:
        return 2.0 * (4.0 * self.c2 * mass - 1.0) / 9.0

    @property
    def beta_x(self) -> float:
        return -(self.alpha_x**2) * self.mu(self.mass_x)

    @property
    def beta_y(self) -> float:
        return -(self.alpha_y**2) * self.mu(self.mass_y)


@dataclass(frozen=True, slots=True)
class SixMatrixActionTerms:
    x_commutator: float
    x_cubic: float
    x_beta: float
    x_quartic: float
    y_commutator: float
    y_cubic: float
    y_beta: float
    y_quartic: float
    cross_commutator: float

    @property
    def total(self) -> float:
        return sum(
            (
                self.x_commutator,
                self.x_cubic,
                self.x_beta,
                self.x_quartic,
                self.y_commutator,
                self.y_cubic,
                self.y_beta,
                self.y_quartic,
                self.cross_commutator,
            )
        )


@dataclass(frozen=True, slots=True)
class SixMatrixGradientTerms:
    x_commutator: ComplexArray
    x_cubic: ComplexArray
    x_beta: ComplexArray
    x_quartic: ComplexArray
    y_commutator: ComplexArray
    y_cubic: ComplexArray
    y_beta: ComplexArray
    y_quartic: ComplexArray
    cross_commutator: ComplexArray

    @property
    def total(self) -> ComplexArray:
        return hermitian_part(
            self.x_commutator
            + self.x_cubic
            + self.x_beta
            + self.x_quartic
            + self.y_commutator
            + self.y_cubic
            + self.y_beta
            + self.y_quartic
            + self.cross_commutator
        )


def _real_trace(value: ComplexArray, *, label: str) -> float:
    scalar = complex(np.trace(value))
    if abs(scalar.imag) > 1e-9 * max(1.0, abs(scalar.real)):
        raise FloatingPointError(f"{label} has a material imaginary residual")
    return float(scalar.real)


def _epsilon(a: int, b: int, c: int) -> int:
    if len({a, b, c}) != 3:
        return 0
    return 1 if (a, b, c) in {(0, 1, 2), (1, 2, 0), (2, 0, 1)} else -1


def reference_action_terms(
    positions: ComplexArray,
    parameters: SixMatrixParameters,
) -> SixMatrixActionTerms:
    """Slow literal transcription of every published summation."""

    if positions.shape != (2, 3, parameters.n, parameters.n):
        raise ValueError("Six-matrix response action positions have another shape")
    values: list[tuple[float, float, float, float]] = []
    for sector, alpha, beta, mass in (
        (positions[0], parameters.alpha_x, parameters.beta_x, parameters.mass_x),
        (positions[1], parameters.alpha_y, parameters.beta_y, parameters.mass_y),
    ):
        commutator_sum = 0.0
        cubic_sum = 0.0j
        radius = np.zeros((parameters.n, parameters.n), dtype="<c16")
        for a in range(3):
            radius += sector[a] @ sector[a]
            for b in range(3):
                pair = sector[a] @ sector[b] - sector[b] @ sector[a]
                commutator_sum += _real_trace(pair @ pair, label="reference commutator")
                for c in range(3):
                    cubic_sum += _epsilon(a, b, c) * np.trace(sector[a] @ sector[b] @ sector[c])
        cubic = complex((2.0j * alpha / 3.0) * cubic_sum)
        if abs(cubic.imag) > 1e-9 * max(1.0, abs(cubic.real)):
            raise FloatingPointError("reference cubic action is not real")
        values.append(
            (
                parameters.n * (-0.25 * commutator_sum),
                parameters.n * float(cubic.real),
                parameters.n * beta * _real_trace(radius, label="reference beta"),
                parameters.n * mass * _real_trace(radius @ radius, label="reference quartic"),
            )
        )
    cross_sum = 0.0
    for a in range(3):
        for b in range(3):
            pair = positions[0, a] @ positions[1, b] - positions[1, b] @ positions[0, a]
            cross_sum += _real_trace(pair @ pair, label="reference cross commutator")
    return SixMatrixActionTerms(
        x_commutator=values[0][0],
        x_cubic=values[0][1],
        x_beta=values[0][2],
        x_quartic=values[0][3],
        y_commutator=values[1][0],
        y_cubic=values[1][1],
        y_beta=values[1][2],
        y_quartic=values[1][3],
        cross_commutator=parameters.n * (-0.5 * parameters.gamma * cross_sum),
    )


def optimized_action_terms(
    positions: ComplexArray,
    parameters: SixMatrixParameters,
) -> SixMatrixActionTerms:
    """Algebraically reduced action used by the integrator."""

    if positions.shape != (2, 3, parameters.n, parameters.n):
        raise ValueError("Six-matrix response action positions have another shape")
    values: list[tuple[float, float, float, float]] = []
    for sector, alpha, beta, mass in (
        (positions[0], parameters.alpha_x, parameters.beta_x, parameters.mass_x),
        (positions[1], parameters.alpha_y, parameters.beta_y, parameters.mass_y),
    ):
        pair_commutators = tuple(
            commutator(sector[a], sector[b]) for a in range(3) for b in range(a + 1, 3)
        )
        commutator_value = -0.5 * sum(
            _real_trace(value @ value, label="optimized commutator") for value in pair_commutators
        )
        cubic_value = _real_trace(
            2.0j * alpha * sector[0] @ commutator(sector[1], sector[2]),
            label="optimized cubic",
        )
        radius = np.sum(sector @ sector, axis=0)
        values.append(
            (
                parameters.n * commutator_value,
                parameters.n * cubic_value,
                parameters.n * beta * _real_trace(radius, label="optimized beta"),
                parameters.n * mass * _real_trace(radius @ radius, label="optimized quartic"),
            )
        )
    cross = tuple(commutator(positions[0, a], positions[1, b]) for a in range(3) for b in range(3))
    cross_value = (
        -0.5
        * parameters.gamma
        * sum(_real_trace(value @ value, label="optimized cross") for value in cross)
    )
    return SixMatrixActionTerms(
        x_commutator=values[0][0],
        x_cubic=values[0][1],
        x_beta=values[0][2],
        x_quartic=values[0][3],
        y_commutator=values[1][0],
        y_cubic=values[1][1],
        y_beta=values[1][2],
        y_quartic=values[1][3],
        cross_commutator=parameters.n * cross_value,
    )


def analytic_gradient_terms(
    positions: ComplexArray,
    parameters: SixMatrixParameters,
) -> SixMatrixGradientTerms:
    """Return every analytic gradient contribution in the Hermitian metric."""

    if positions.shape != (2, 3, parameters.n, parameters.n):
        raise ValueError("Six-matrix response gradient positions have another shape")
    shape = positions.shape

    def zero() -> ComplexArray:
        return np.zeros(shape, dtype="<c16")

    x_comm, x_cubic, x_beta, x_quartic = zero(), zero(), zero(), zero()
    y_comm, y_cubic, y_beta, y_quartic = zero(), zero(), zero(), zero()
    cross = zero()
    sector_terms = (
        (0, parameters.alpha_x, parameters.beta_x, parameters.mass_x),
        (1, parameters.alpha_y, parameters.beta_y, parameters.mass_y),
    )
    destinations = (
        (x_comm, x_cubic, x_beta, x_quartic),
        (y_comm, y_cubic, y_beta, y_quartic),
    )
    for (sector_id, alpha, beta, mass), destination in zip(sector_terms, destinations, strict=True):
        sector = positions[sector_id]
        radius = np.sum(sector @ sector, axis=0)
        comm_term, cubic_term, beta_term, quartic_term = destination
        for a in range(3):
            comm_term[sector_id, a] = sum(
                (commutator(sector[b], commutator(sector[b], sector[a])) for b in range(3)),
                start=np.zeros((parameters.n, parameters.n), dtype="<c16"),
            )
            b, c = ((1, 2), (2, 0), (0, 1))[a]
            cubic_term[sector_id, a] = 2.0j * alpha * commutator(sector[b], sector[c])
            beta_term[sector_id, a] = 2.0 * beta * sector[a]
            quartic_term[sector_id, a] = 2.0 * mass * (radius @ sector[a] + sector[a] @ radius)
    for a in range(3):
        cross[0, a] = parameters.gamma * sum(
            (
                commutator(positions[1, b], commutator(positions[1, b], positions[0, a]))
                for b in range(3)
            ),
            start=np.zeros((parameters.n, parameters.n), dtype="<c16"),
        )
        cross[1, a] = parameters.gamma * sum(
            (
                commutator(positions[0, b], commutator(positions[0, b], positions[1, a]))
                for b in range(3)
            ),
            start=np.zeros((parameters.n, parameters.n), dtype="<c16"),
        )
    scale = float(parameters.n)
    return SixMatrixGradientTerms(
        x_commutator=hermitian_part(scale * x_comm),
        x_cubic=hermitian_part(scale * x_cubic),
        x_beta=hermitian_part(scale * x_beta),
        x_quartic=hermitian_part(scale * x_quartic),
        y_commutator=hermitian_part(scale * y_comm),
        y_cubic=hermitian_part(scale * y_cubic),
        y_beta=hermitian_part(scale * y_beta),
        y_quartic=hermitian_part(scale * y_quartic),
        cross_commutator=hermitian_part(scale * cross),
    )


def coupling_derivative_density(
    positions: ComplexArray,
    parameters: SixMatrixParameters,
) -> tuple[float, float]:
    """Return partial(S/n)/partial(alpha_tilde_x/y), including beta(alpha)."""

    values = []
    for sector, alpha, mass in (
        (positions[0], parameters.alpha_x, parameters.mass_x),
        (positions[1], parameters.alpha_y, parameters.mass_y),
    ):
        radius = np.sum(sector @ sector, axis=0)
        cubic_derivative = _real_trace(
            (2.0j / parameters.q) * sector[0] @ commutator(sector[1], sector[2]),
            label="coupling cubic derivative",
        )
        beta_derivative = -2.0 * alpha * parameters.mu(mass) / parameters.q
        values.append(
            cubic_derivative
            + beta_derivative * _real_trace(radius, label="coupling beta derivative")
        )
    return values[0], values[1]


__all__ = [
    'SixMatrixActionTerms',
    'SixMatrixGradientTerms',
    'SixMatrixParameters',
    'analytic_gradient_terms',
    "commutator",
    'coupling_derivative_density',
    'optimized_action_terms',
    'reference_action_terms',
]
