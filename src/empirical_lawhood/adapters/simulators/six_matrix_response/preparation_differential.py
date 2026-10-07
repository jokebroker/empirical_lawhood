"""Matrix-preparation baseline derivatives of the actual fixed-coupling native map.

These pure instruments own no RNG, source access, execution or scientific
qualification. Future-path tangents and secants are privileged diagnostics.
The historical Response geometry force and Hessian contracts remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

import numpy as np

from .contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView, MatrixIntegratorKind
from .model import ComplexArray, SixMatrixState, hermiticity_residual, hermitian_part
from .response_hessian import _hessian_image
from .simulation import BAOABGradientCache, baoab_step_with_hermitian_noise


def _matrix(value: ComplexArray, shape: tuple[int, ...], name: str) -> ComplexArray:
    if value.shape != shape or value.dtype != np.dtype("complex128"):
        raise ValueError(f"preparation {name} requires complex128 geometry {shape}")
    if not np.isfinite(value).all() or hermiticity_residual(value) > 1e-12:
        raise ValueError(f"preparation {name} must be finite and Hermitian")
    return value


def _mode(value: ComplexArray) -> ComplexArray:
    result = _matrix(value, (3, 4, 4), "mode")
    if abs(float(np.vdot(result, result).real) - 1.0) > 1e-12:
        raise ValueError("preparation mode must have unit real HS norm")
    return result


def _state(value: SixMatrixState) -> None:
    if value.q != 2 or not value.finite:
        raise ValueError("preparation requires a finite q=2 native state")
    if abs(value.alpha_tilde_x - 2 / 3) > 1e-12 or abs(value.alpha_tilde_y - 22 / 3) > 1e-12:
        raise ValueError("preparation downstream map requires common nominal couplings")


def _timestep(view: SixMatrixResponseNumericalView) -> float:
    if (
        view.timestep not in (Decimal("0.001"), Decimal("0.0005"))
        or view.friction_gamma != Decimal(1)
        or view.bath_temperature != Decimal(1)
        or view.integrator is not MatrixIntegratorKind.BAOAB_UNDERDAMPED_LANGEVIN
    ):
        raise ValueError("preparation map requires its two declared Langevin views")
    return float(view.timestep)


def preparation_pulse_amplitude(
    *, native_step: int, invocation_tick: int, refinement: int, sign: int, magnitude: float
) -> float:
    """One value for both kicks in a completed 64-reference-tick interval."""
    if type(native_step) is not int or native_step < 0:
        raise ValueError("native step must be a nonnegative integer")
    if type(invocation_tick) is not int or invocation_tick < 0:
        raise ValueError("invocation must be a nonnegative reference tick")
    if type(refinement) is not int or refinement not in (1, 2):
        raise ValueError("preparation permits only primary and half views")
    if type(sign) is not int or sign not in (-1, 0, 1):
        raise ValueError("preparation sign must be NEG, HOLD or POS")
    if type(magnitude) is not float or magnitude not in (0.5, 1.0, 2.0, 8.0):
        raise ValueError("preparation amplitude is outside the finite diagnostic chart")
    start = invocation_tick * refinement
    return sign * magnitude if start <= native_step < start + 64 * refinement else 0.0


def preparation_force_step(
    state: SixMatrixState,
    *,
    amplitude: float,
    mode: ComplexArray,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    standardized_noise: ComplexArray,
    gradient_cache: BAOABGradientCache | None = None,
) -> SixMatrixState:
    """Continuous force instrument used by the chart and independent differences.

    The issued source owns the finite chart. This pure map also accepts finite
    perturbed force values to permit differentiation checks; that capability
    does not expand the action chart or grant execution authority.
    """
    _state(state)
    direction = _mode(mode)
    dt = _timestep(numerical_view)
    if not np.isfinite(amplitude):
        raise ValueError("preparation force must be finite")
    if (
        member.mass_x != Decimal("0.5")
        or member.mass_y != Decimal("0.5")
        or member.cross_coupling_gamma != Decimal(1)
    ):
        raise ValueError("preparation map changes the imported native member")
    kick = np.zeros_like(state.momenta)
    kick[0] = 0.5 * dt * amplitude * direction
    incoming = (
        state if amplitude == 0 else replace(state, momenta=hermitian_part(state.momenta + kick))
    )
    advanced = baoab_step_with_hermitian_noise(
        incoming,
        member=member,
        numerical_view=numerical_view,
        next_alpha_tilde_x=state.alpha_tilde_x,
        next_alpha_tilde_y=state.alpha_tilde_y,
        standardized_noise=standardized_noise,
        gradient_cache=gradient_cache,
    )
    return (
        advanced
        if amplitude == 0
        else replace(advanced, momenta=hermitian_part(advanced.momenta + kick))
    )


@dataclass(frozen=True, slots=True)
class NativeStateDifferential:
    positions: ComplexArray
    momenta: ComplexArray

    def __post_init__(self) -> None:
        for field in ("positions", "momenta"):
            value = _matrix(getattr(self, field), (2, 3, 4, 4), field)
            frozen = np.frombuffer(value.tobytes(), dtype=np.complex128).reshape(value.shape)
            object.__setattr__(self, field, frozen)

    @classmethod
    def zero(cls) -> NativeStateDifferential:
        return cls(
            np.zeros((2, 3, 4, 4), dtype=np.complex128), np.zeros((2, 3, 4, 4), dtype=np.complex128)
        )

    def receiver(self, mode: ComplexArray) -> float:
        return float(np.vdot(_mode(mode), self.positions[0]).real)


def _coupled_step(
    differential: NativeStateDifferential,
    *,
    first_image: ComplexArray,
    dt: float,
    direction: ComplexArray,
    force_derivative: float,
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    kick = np.zeros_like(differential.positions)
    kick[0] = 0.5 * dt * force_derivative * direction
    momentum = hermitian_part(differential.momenta + kick - 0.5 * dt * first_image)
    position = hermitian_part(differential.positions + 0.5 * dt * momentum)
    momentum = hermitian_part(np.exp(-dt) * momentum)
    position = hermitian_part(position + 0.5 * dt * momentum)
    return position, momentum, kick


def discrete_tangent_step(
    state: SixMatrixState,
    advanced: SixMatrixState,
    differential: NativeStateDifferential,
    *,
    mode: ComplexArray,
    numerical_view: SixMatrixResponseNumericalView,
    force_derivative: float,
) -> NativeStateDifferential:
    """Differentiate F_dt at the supplied native trajectory and fixed noise.

    `force_derivative` is one during an amplitude derivative's active interval,
    zero after it. The stochastic innovation derivative is exactly zero. Both
    native gradient locations and both external force kicks are retained.
    """
    _state(state)
    _state(advanced)
    if advanced.step_index != state.step_index + 1 or not np.isfinite(force_derivative):
        raise ValueError("tangent requires consecutive states and a finite input derivative")
    dt = _timestep(numerical_view)
    position, momentum, kick = _coupled_step(
        differential,
        first_image=_hessian_image(state.positions, differential.positions),
        dt=dt,
        direction=_mode(mode),
        force_derivative=force_derivative,
    )
    momentum = hermitian_part(
        momentum - 0.5 * dt * _hessian_image(advanced.positions, position) + kick
    )
    return NativeStateDifferential(position, momentum)


def secant_hessian_image(
    minus: ComplexArray, plus: ComplexArray, perturbation: ComplexArray
) -> ComplexArray:
    """Exact two-node integral for the native quartic potential's Hessian.

    H is degree at most two along the chord. Gauss--Legendre integration on
    [0,1] is exact through degree three. This is a future-path identity and
    contains no prediction or infinitesimal approximation.
    """
    for value in (minus, plus, perturbation):
        _matrix(value, (2, 3, 4, 4), "secant operand")
    middle = (minus + plus) / 2
    displacement = (plus - minus) / (2 * np.sqrt(3.0))
    return hermitian_part(
        (
            _hessian_image(middle - displacement, perturbation)
            + _hessian_image(middle + displacement, perturbation)
        )
        / 2
    )


def discrete_secant_step(
    minus: SixMatrixState,
    plus: SixMatrixState,
    advanced_minus: SixMatrixState,
    advanced_plus: SixMatrixState,
    reconstructed: NativeStateDifferential,
    *,
    mode: ComplexArray,
    numerical_view: SixMatrixResponseNumericalView,
    normalized_force: float,
) -> NativeStateDifferential:
    """Propagate (S_plus-S_minus)/(2a) without substituting the observed difference."""
    for state in (minus, plus, advanced_minus, advanced_plus):
        _state(state)
    if (
        minus.step_index != plus.step_index
        or advanced_minus.step_index != minus.step_index + 1
        or advanced_plus.step_index != plus.step_index + 1
        or not np.isfinite(normalized_force)
    ):
        raise ValueError("secant requires aligned consecutive signed states")
    dt = _timestep(numerical_view)
    position, momentum, kick = _coupled_step(
        reconstructed,
        first_image=secant_hessian_image(
            minus.positions, plus.positions, reconstructed.positions
        ),
        dt=dt,
        direction=_mode(mode),
        force_derivative=normalized_force,
    )
    momentum = hermitian_part(
        momentum
        - 0.5
        * dt
        * secant_hessian_image(advanced_minus.positions, advanced_plus.positions, position)
        + kick
    )
    return NativeStateDifferential(position, momentum)


def scalar_hold_tangent_step(
    state: SixMatrixState,
    advanced: SixMatrixState,
    *,
    position: float,
    momentum: float,
    mode: ComplexArray,
    numerical_view: SixMatrixResponseNumericalView,
    force_derivative: float,
) -> tuple[float, float]:
    """Scalar reduction at both actual HOLD gradient positions, same discrete map."""
    _state(state)
    _state(advanced)
    if advanced.step_index != state.step_index + 1 or not all(
        np.isfinite(value) for value in (position, momentum, force_derivative)
    ):
        raise ValueError("scalar tangent requires consecutive states and finite operands")
    dt = _timestep(numerical_view)
    full_direction = np.zeros_like(state.positions)
    full_direction[0] = _mode(mode)
    first = float(np.vdot(full_direction, _hessian_image(state.positions, full_direction)).real)
    last = float(np.vdot(full_direction, _hessian_image(advanced.positions, full_direction)).real)
    momentum += 0.5 * dt * (force_derivative - first * position)
    position += 0.5 * dt * momentum
    momentum *= float(np.exp(-dt))
    position += 0.5 * dt * momentum
    momentum += 0.5 * dt * (force_derivative - last * position)
    return position, momentum
