"""Pure UEG scaling, transverse response and slab-receiver calculations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from math import acosh, exp, isfinite, log, log1p, pi
from typing import ClassVar, Iterable, Mapping, Protocol, Sequence

from scipy.constants import (
    elementary_charge,
    hbar,
    m_e,
    mu_0,
    physical_constants,
)

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_decimal

from .contracts import ActionCurrentRow, FiniteQEstimate, Vector3


class FiniteQDesign(Protocol):
    """The scientific inputs needed for the signed finite-q estimate."""

    u0: Decimal

    def threshold(self, key: str) -> Decimal: ...


def _d(value: float | int | str | Decimal) -> Decimal:
    result = value if isinstance(value, Decimal) else Decimal(str(value))
    validate_decimal(result, field_name="finite_decimal")
    return result


def _float(value: Decimal, *, field_name: str) -> float:
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field_name} is not representable as a finite float")
    return result


@dataclass(frozen=True, slots=True)
class UniformElectronGasScales(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/uniform-electron-gas-response/uniform-electron-gas-scales'

    r_s: Decimal
    density_m3: Decimal
    k_f_m1: Decimal
    e_f_J: Decimal
    e_f_eV: Decimal
    v_f_m_s: Decimal
    diamagnetic_kernel_A_T_m3: Decimal
    constants_source: str

    def __post_init__(self) -> None:
        for name in (
            "r_s",
            "density_m3",
            "k_f_m1",
            "e_f_J",
            "e_f_eV",
            "v_f_m_s",
            "diamagnetic_kernel_A_T_m3",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal("0"))
        if any(
            getattr(self, name) == 0
            for name in (
                "r_s",
                "density_m3",
                "k_f_m1",
                "e_f_J",
                "v_f_m_s",
                "diamagnetic_kernel_A_T_m3",
            )
        ):
            raise ValueError("UEG scales must be strictly positive")


def ueg_scales(r_s: Decimal) -> UniformElectronGasScales:
    """Return free-electron 3D UEG scales for a Wigner-Seitz radius."""

    validate_decimal(r_s, field_name="r_s", minimum=Decimal("0"))
    if r_s == 0:
        raise ValueError("r_s must be positive")
    a0 = physical_constants["Bohr radius"][0]
    radius = _float(r_s, field_name="r_s") * a0
    density = 3.0 / (4.0 * pi * radius**3)
    k_f = (3.0 * pi**2 * density) ** (1.0 / 3.0)
    e_f = hbar**2 * k_f**2 / (2.0 * m_e)
    v_f = hbar * k_f / m_e
    diamagnetic = density * elementary_charge**2 / m_e
    return UniformElectronGasScales(
        r_s=r_s,
        density_m3=_d(density),
        k_f_m1=_d(k_f),
        e_f_J=_d(e_f),
        e_f_eV=_d(e_f / elementary_charge),
        v_f_m_s=_d(v_f),
        diamagnetic_kernel_A_T_m3=_d(diamagnetic),
        constants_source="scipy-constants-codata",
    )


def vector_scale(vector: Vector3, scale: Decimal) -> Vector3:
    validate_decimal(scale, field_name="scale")
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def vector_subtract(left: Vector3, right: Vector3) -> Vector3:
    return (left[0] - right[0], left[1] - right[1], left[2] - right[2])


def dot(left: Vector3, right: Vector3) -> Decimal:
    return sum((a * b for a, b in zip(left, right, strict=True)), Decimal("0"))


def norm(vector: Vector3) -> Decimal:
    with localcontext() as context:
        context.prec = 40
        return dot(vector, vector).sqrt()


def relative_vector_difference(left: Vector3, right: Vector3, *, floor: Decimal) -> Decimal:
    validate_decimal(floor, field_name="floor", minimum=Decimal("0"))
    return norm(vector_subtract(left, right)) / max(norm(left), norm(right), floor)


def transverse_relative_dot(q_vector: Vector3, a_vector: Vector3) -> Decimal:
    denominator = norm(q_vector) * norm(a_vector)
    if denominator == 0:
        return Decimal("0")
    return abs(dot(q_vector, a_vector)) / denominator


def vector_potential_from_u(u: Decimal, scales: UniformElectronGasScales) -> Decimal:
    """Map dimensionless action u to signed SI vector-potential amplitude."""

    validate_decimal(u, field_name="u")
    return u * scales.e_f_J / (_d(elementary_charge) * scales.v_f_m_s)


def magnetic_field_amplitude(
    q_over_kf: Decimal,
    realized_A_T: Vector3,
    scales: UniformElectronGasScales,
) -> Decimal:
    """Return |q x A| for the frozen orthogonal plane-wave chart."""

    q_abs = q_over_kf * scales.k_f_m1
    return q_abs * norm(realized_A_T)


def project_current(current: Vector3, direction: Vector3) -> Decimal:
    direction_norm = norm(direction)
    if direction_norm == 0:
        raise ValueError("current projection direction is zero")
    return dot(current, direction) / direction_norm


def row_map(
    rows: Sequence[ActionCurrentRow],
) -> dict[tuple[Decimal, Decimal], ActionCurrentRow]:
    result: dict[tuple[Decimal, Decimal], ActionCurrentRow] = {}
    for row in rows:
        key = (row.q_over_kf, row.u)
        if key in result:
            raise ValueError("action panel has a duplicate q/u row")
        result[key] = row
    return result


def finite_q_estimate(
    *,
    panel_id: str,
    q_over_kf: Decimal,
    rows: Mapping[tuple[Decimal, Decimal], ActionCurrentRow],
    config: FiniteQDesign,
    action_direction: Vector3,
) -> FiniteQEstimate:
    """Estimate K_A from signed currents using realized SI A amplitudes.

    The plan's finite-difference coordinate is dimensionless ``u``.  London
    stiffness, however, is a derivative with respect to SI vector potential.
    This function therefore divides by the realized SI ``A(+u)-A(-u)`` before
    any penetration-depth transformation.
    """

    def one(u: Decimal) -> ActionCurrentRow:
        try:
            return rows[(q_over_kf, u)]
        except KeyError as error:
            raise ValueError(f"panel lacks q={q_over_kf}, u={u}") from error

    zero = one(Decimal("0"))
    plus_half = one(config.u0 / 2)
    minus_half = one(-config.u0 / 2)
    plus_full = one(config.u0)
    minus_full = one(-config.u0)

    def estimate(plus: ActionCurrentRow, minus: ActionCurrentRow) -> tuple[Decimal, Decimal]:
        current_plus = project_current(plus.current_density, action_direction)
        current_minus = project_current(minus.current_density, action_direction)
        a_plus = project_current(plus.realized_A_T, action_direction)
        a_minus = project_current(minus.realized_A_T, action_direction)
        denominator = a_plus - a_minus
        if denominator <= 0:
            raise ValueError("signed realized vector potentials are not ordered")
        kernel = -(current_plus - current_minus) / denominator
        error = (plus.current_error_bound_A_m2 + minus.current_error_bound_A_m2) / abs(denominator)
        return kernel, error

    kernel_half, error_half = estimate(plus_half, minus_half)
    kernel_full, error_full = estimate(plus_full, minus_full)
    j0 = project_current(zero.current_density, action_direction)

    def even(plus: ActionCurrentRow, minus: ActionCurrentRow) -> Decimal:
        return (
            project_current(plus.current_density, action_direction)
            + project_current(minus.current_density, action_direction)
            - Decimal("2") * j0
        ) / 2

    even_half = even(plus_half, minus_half)
    even_full = even(plus_full, minus_full)
    kernel_floor = max(error_half, error_full, Decimal("1e-300"))
    locality_denominator = max(abs(kernel_half), abs(kernel_full), kernel_floor)
    locality = abs(kernel_half - kernel_full) / locality_denominator
    response_half = (
        abs(
            project_current(plus_half.current_density, action_direction)
            - project_current(minus_half.current_density, action_direction)
        )
        / 2
    )
    response_full = (
        abs(
            project_current(plus_full.current_density, action_direction)
            - project_current(minus_full.current_density, action_direction)
        )
        / 2
    )
    numerical_floor = max(
        row.current_error_bound_A_m2 for row in (zero, plus_half, minus_half, plus_full, minus_full)
    )
    zero_offset_bound = max(
        config.threshold("even_remainder_floor_multiplier") * zero.current_error_bound_A_m2,
        config.threshold("baseline_relative") * max(response_half, response_full),
    )
    even_bound_half = max(
        config.threshold("even_remainder_floor_multiplier") * numerical_floor,
        config.threshold("even_remainder_response_relative") * response_half,
    )
    even_bound_full = max(
        config.threshold("even_remainder_floor_multiplier") * numerical_floor,
        config.threshold("even_remainder_response_relative") * response_full,
    )
    return FiniteQEstimate(
        estimate_id=f"estimate-{panel_id}-{str(q_over_kf).replace('.', 'p')}",
        q_over_kf=q_over_kf,
        kernel_half=kernel_half,
        kernel_full=kernel_full,
        kernel_error_bound=max(error_half, error_full),
        offset_current_A_m2=j0,
        even_remainder_half_A_m2=even_half,
        even_remainder_full_A_m2=even_full,
        locality_relative=locality,
        locality_pass=(
            locality + (error_half + error_full) / locality_denominator
            <= config.threshold("amplitude_locality_relative")
        ),
        even_remainder_pass=(
            abs(even_half) <= even_bound_half and abs(even_full) <= even_bound_full
        ),
        zero_offset_pass=abs(j0) <= zero_offset_bound,
    )


@dataclass(frozen=True, slots=True)
class InterceptEstimate:
    kernel0: Decimal
    c2: Decimal
    error_bound: Decimal


def fit_even_q_intercept(estimates: Sequence[FiniteQEstimate]) -> InterceptEstimate:
    """Fit the frozen K(q)=K0+c2*(q/kF)^2 model without adaptive terms."""

    if len(estimates) < 2:
        raise ValueError("q intercept requires at least two finite-q points")
    x = tuple(value.q_over_kf**2 for value in estimates)
    y = tuple(value.kernel_full for value in estimates)
    n = Decimal(len(estimates))
    mean_x = sum(x, Decimal("0")) / n
    mean_y = sum(y, Decimal("0")) / n
    s_xx = sum(((value - mean_x) ** 2 for value in x), Decimal("0"))
    if s_xx == 0:
        raise ValueError("q intercept design has zero x variance")
    c2 = (
        sum(
            (
                (x_value - mean_x) * (y_value - mean_y)
                for x_value, y_value in zip(x, y, strict=True)
            ),
            Decimal("0"),
        )
        / s_xx
    )
    kernel0 = mean_y - c2 * mean_x
    weights = tuple(Decimal("1") / n - mean_x * (value - mean_x) / s_xx for value in x)
    error_bound = sum(
        (
            abs(weight) * estimate.kernel_error_bound
            for weight, estimate in zip(weights, estimates, strict=True)
        ),
        Decimal("0"),
    )
    return InterceptEstimate(kernel0=kernel0, c2=c2, error_bound=error_bound)


def q_intercept_stability(
    estimates: Sequence[FiniteQEstimate], *, kernel_floor: Decimal
) -> tuple[InterceptEstimate, InterceptEstimate, Decimal]:
    if len(estimates) != 4:
        raise ValueError("frozen q stability requires exactly four q points")
    ordered = tuple(sorted(estimates, key=lambda value: value.q_over_kf))
    all_fit = fit_even_q_intercept(ordered)
    small_fit = fit_even_q_intercept(ordered[:3])
    stability = abs(all_fit.kernel0 - small_fit.kernel0) / max(abs(all_fit.kernel0), kernel_floor)
    return all_fit, small_fit, stability


def kernel_from_penetration_depth(penetration_depth_m: Decimal) -> Decimal:
    validate_decimal(
        penetration_depth_m,
        field_name="penetration_depth_m",
        minimum=Decimal("0"),
    )
    if penetration_depth_m == 0:
        raise ValueError("penetration depth must be positive")
    return Decimal("1") / (_d(mu_0) * penetration_depth_m**2)


def penetration_depth_from_kernel(kernel: Decimal) -> Decimal:
    validate_decimal(kernel, field_name="kernel")
    if kernel <= 0:
        raise ValueError("positive stiffness is required for penetration depth")
    with localcontext() as context:
        context.prec = 40
        return (Decimal("1") / (_d(mu_0) * kernel)).sqrt()


def shielding_score(thickness_m: Decimal, penetration_depth_m: Decimal) -> Decimal:
    for name, value in (
        ("thickness_m", thickness_m),
        ("penetration_depth_m", penetration_depth_m),
    ):
        validate_decimal(value, field_name=name, minimum=Decimal("0"))
        if value == 0:
            raise ValueError(f"{name} must be positive")
    argument = abs(
        _float(
            thickness_m / (Decimal("2") * penetration_depth_m),
            field_name="slab_argument",
        )
    )
    # 2 exp(-x)/(1+exp(-2x)) is sech(x) without cosh overflow.
    decaying = exp(-argument)
    return _d(1.0 - (2.0 * decaying) / (1.0 + decaying * decaying))


def _log_cosh(value: float) -> float:
    absolute = abs(value)
    return absolute + log1p(exp(-2.0 * absolute)) - log(2.0)


def slab_profile(
    *, thickness_m: Decimal, penetration_depth_m: Decimal, points: int
) -> tuple[tuple[Decimal, Decimal], ...]:
    for name, value in (
        ("thickness_m", thickness_m),
        ("penetration_depth_m", penetration_depth_m),
    ):
        validate_decimal(value, field_name=name, minimum=Decimal("0"))
        if value == 0:
            raise ValueError(f"{name} must be positive")
    if isinstance(points, bool) or not isinstance(points, int) or points < 3 or points % 2 == 0:
        raise ValueError("slab profile requires an odd point count >= 3")
    half = thickness_m / 2
    denominator_log = _log_cosh(
        _float(half / penetration_depth_m, field_name="slab_denominator_argument")
    )
    values = []
    for index in range(points):
        fraction = Decimal(index) / Decimal(points - 1)
        x = -half + thickness_m * fraction
        ratio = exp(
            _log_cosh(_float(x / penetration_depth_m, field_name="slab_profile_argument"))
            - denominator_log
        )
        values.append((x, _d(ratio)))
    return tuple(values)


def fit_penetration_depth_from_slab(
    *, thickness_m: Decimal, profile: Sequence[tuple[Decimal, Decimal]]
) -> Decimal:
    if len(profile) < 3 or len(profile) % 2 == 0:
        raise ValueError("slab fit requires an odd profile with a centre point")
    centre_x, centre_ratio = profile[len(profile) // 2]
    if centre_x != 0 or not (Decimal("0") < centre_ratio < Decimal("1")):
        raise ValueError("slab profile has an invalid centre")
    value = _float(Decimal("1") / centre_ratio, field_name="inverse_centre_ratio")
    return thickness_m / _d(2.0 * acosh(value))


def maximum_field(rows: Iterable[ActionCurrentRow], scales: UniformElectronGasScales) -> Decimal:
    return max(
        (magnetic_field_amplitude(row.q_over_kf, row.realized_A_T, scales) for row in rows),
        default=Decimal("0"),
    )


__all__ = [
    "InterceptEstimate",
    'UniformElectronGasScales',
    "dot",
    "finite_q_estimate",
    "fit_even_q_intercept",
    "fit_penetration_depth_from_slab",
    "kernel_from_penetration_depth",
    "magnetic_field_amplitude",
    "maximum_field",
    "norm",
    "penetration_depth_from_kernel",
    "project_current",
    "q_intercept_stability",
    "relative_vector_difference",
    "row_map",
    "shielding_score",
    "slab_profile",
    "transverse_relative_dot",
    "ueg_scales",
    "vector_potential_from_u",
    "vector_scale",
]
