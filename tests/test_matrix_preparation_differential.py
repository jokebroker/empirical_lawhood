"""Independent exposed native-map checks, migrated from the original differential tests.

Source: icf-yolo 70ad670, test_preparation_differential.py. No campaign work.
"""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import (
    BetaCouplingRule,
    SixMatrixResponseModelFamilyMember,
    SixMatrixResponseNumericalView,
    MatrixIntegratorKind,
    MatrixPrecision,
)
from empirical_lawhood.adapters.simulators.six_matrix_response.gradients import (
    SixMatrixParameters,
    analytic_gradient_terms,
)
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ideal_state
from empirical_lawhood.adapters.simulators.six_matrix_response.preparation_differential import (
    NativeStateDifferential,
    discrete_secant_step,
    discrete_tangent_step,
    preparation_force_step,
    preparation_pulse_amplitude,
    scalar_hold_tangent_step,
    secant_hessian_image,
)
from empirical_lawhood.adapters.simulators.six_matrix_response.response_assay import (
    ResponseGeometryNativeForcePulse,
    force_step,
)
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import hermitian_noise


@pytest.fixture
def operands():
    rng = np.random.default_rng(98124)
    member = SixMatrixResponseModelFamilyMember(
        "prep.fixture.member",
        Decimal("0.5"),
        Decimal("0.5"),
        Decimal(1),
        BetaCouplingRule.PUBLISHED_DETERMINISTIC,
    )
    view = SixMatrixResponseNumericalView(
        "prep.fixture.primary",
        MatrixIntegratorKind.BAOAB_UNDERDAMPED_LANGEVIN,
        Decimal("0.001"),
        "dimensionless-langevin-time",
        Decimal(1),
        Decimal(1),
        MatrixPrecision.COMPLEX128,
        "numpy.pcg64dxsm",
        "1.0.0",
        "prep.fixture.stream",
        512,
    )
    base = ideal_state(q=2, alpha_tilde_x=2 / 3, alpha_tilde_y=22 / 3, constitution="01")
    state = replace(
        base,
        positions=base.positions + 0.1 * hermitian_noise(rng=rng, q=2),
        momenta=hermitian_noise(rng=rng, q=2),
    )
    mode = hermitian_noise(rng=rng, q=2)[0]
    mode /= np.linalg.norm(mode)
    return rng, member, view, state, mode


@pytest.mark.parametrize("refinement", (1, 2))
@pytest.mark.parametrize("tick", (4, 5, 68, 69))
def test_force_matches_imported_map_at_on_off_edges(operands, refinement, tick):
    rng, member, view, state, mode = operands
    view = replace(view, timestep=Decimal("0.001") / refinement)
    state = replace(state, step_index=tick * refinement)
    noise = hermitian_noise(rng=rng, q=2)
    for sign in (-1, 0, 1):
        amplitude = preparation_pulse_amplitude(
            native_step=state.step_index,
            invocation_tick=5,
            refinement=refinement,
            sign=sign,
            magnitude=8.0,
        )
        current = preparation_force_step(
            state,
            amplitude=amplitude,
            mode=mode,
            member=member,
            numerical_view=view,
            standardized_noise=noise,
        )
        imported = force_step(
            state,
            pulse=ResponseGeometryNativeForcePulse("prep.fixture.pulse", "short-pulse-response", sign, 5),
            mode=mode,
            member=member,
            numerical_view=view,
            standardized_noise=noise,
        )
        np.testing.assert_array_equal(current.positions, imported.positions)
        np.testing.assert_array_equal(current.momenta, imported.momenta)


@pytest.mark.parametrize("refinement", (1, 2))
@pytest.mark.parametrize("active", (False, True))
def test_full_state_and_receiver_derivative_by_independent_differences(
    operands, refinement, active
):
    rng, member, view, state, mode = operands
    view = replace(view, timestep=Decimal("0.001") / refinement)
    noise = hermitian_noise(rng=rng, q=2)
    tangent = NativeStateDifferential(
        hermitian_noise(rng=rng, q=2), hermitian_noise(rng=rng, q=2)
    )

    def advance(value, force):
        return preparation_force_step(
            value,
            amplitude=force,
            mode=mode,
            member=member,
            numerical_view=view,
            standardized_noise=noise,
        )

    analytic = discrete_tangent_step(
        state,
        advance(state, 0.0),
        tangent,
        mode=mode,
        numerical_view=view,
        force_derivative=float(active),
    )
    for epsilon in (1e-4, 1e-5, 1e-6):
        differences = []
        for sign in (-1, 1):
            perturbed = replace(
                state,
                positions=state.positions + sign * epsilon * tangent.positions,
                momenta=state.momenta + sign * epsilon * tangent.momenta,
            )
            differences.append(advance(perturbed, sign * epsilon * active))
        minus, plus = differences
        for field in ("positions", "momenta"):
            finite = (getattr(plus, field) - getattr(minus, field)) / (2 * epsilon)
            np.testing.assert_allclose(getattr(analytic, field), finite, rtol=1e-5, atol=1e-8)
        receiver = float(
            np.vdot(mode, (plus.positions[0] - minus.positions[0]) / (2 * epsilon)).real
        )
        assert analytic.receiver(mode) == pytest.approx(receiver, rel=1e-5, abs=1e-8)
    with pytest.raises(ValueError):
        analytic.positions.setflags(write=True)


def test_secant_integral_equals_native_gradient_difference(operands):
    rng, _, _, state, _ = operands
    minus = state.positions
    plus = minus + 0.7 * hermitian_noise(rng=rng, q=2)
    parameters = SixMatrixParameters(2, 0.5, 0.5, 1, 2 / 3, 22 / 3)
    expected = (
        analytic_gradient_terms(plus, parameters).total
        - analytic_gradient_terms(minus, parameters).total
    )
    np.testing.assert_allclose(
        secant_hessian_image(minus, plus, plus - minus), expected, atol=1e-10, rtol=1e-11
    )


@pytest.mark.parametrize("refinement", (1, 2))
def test_secant_reconstructs_entire_finite_pulse_across_force_cessation(operands, refinement):
    rng, member, view, state, mode = operands
    view = replace(view, timestep=Decimal("0.001") / refinement)
    minus = plus = state
    reconstruction = NativeStateDifferential.zero()
    for step in range(68 * refinement):
        noise = hermitian_noise(rng=rng, q=2)
        active = float(step < 64 * refinement)
        following = [
            preparation_force_step(
                value,
                amplitude=sign * 8.0 * active,
                mode=mode,
                member=member,
                numerical_view=view,
                standardized_noise=noise,
            )
            for value, sign in ((minus, -1), (plus, 1))
        ]
        reconstruction = discrete_secant_step(
            minus,
            plus,
            following[0],
            following[1],
            reconstruction,
            mode=mode,
            numerical_view=view,
            normalized_force=active,
        )
        minus, plus = following
    for field in ("positions", "momenta"):
        np.testing.assert_allclose(
            getattr(reconstruction, field),
            (getattr(plus, field) - getattr(minus, field)) / 16,
            atol=1e-12,
            rtol=1e-9,
        )


def test_discrete_tangent_and_scalar_are_gauge_covariant(operands):
    rng, member, view, state, mode = operands
    noise = hermitian_noise(rng=rng, q=2)
    tangent = NativeStateDifferential(
        hermitian_noise(rng=rng, q=2), hermitian_noise(rng=rng, q=2)
    )
    unitary = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0]

    def rotate(value):
        return unitary @ value @ unitary.conj().T

    def calculate(current, direction, innovation, differential):
        after = preparation_force_step(
            current,
            amplitude=0.0,
            mode=direction,
            member=member,
            numerical_view=view,
            standardized_noise=innovation,
        )
        full = discrete_tangent_step(
            current, after, differential, mode=direction, numerical_view=view, force_derivative=1.0
        )
        scalar = scalar_hold_tangent_step(
            current,
            after,
            position=0.2,
            momentum=0.3,
            mode=direction,
            numerical_view=view,
            force_derivative=1.0,
        )
        return full, scalar

    original, scalar = calculate(state, mode, noise, tangent)
    rotated, scalar_rotated = calculate(
        replace(state, positions=rotate(state.positions), momenta=rotate(state.momenta)),
        rotate(mode),
        rotate(noise),
        NativeStateDifferential(rotate(tangent.positions), rotate(tangent.momenta)),
    )
    np.testing.assert_allclose(rotated.positions, rotate(original.positions), atol=1e-12)
    np.testing.assert_allclose(rotated.momenta, rotate(original.momenta), atol=1e-12)
    np.testing.assert_allclose(scalar_rotated, scalar, atol=1e-12)


def test_negative_curvature_is_retained_and_unsafe_semantics_rejected(operands):
    _, _, view, _, _ = operands
    state = ideal_state(q=2, alpha_tilde_x=2 / 3, alpha_tilde_y=22 / 3, constitution="01")
    mode = np.zeros((3, 4, 4), dtype=np.complex128)
    mode[0] = np.eye(4) / 2
    after = replace(state, step_index=1)
    position, momentum = scalar_hold_tangent_step(
        state,
        after,
        position=1.0,
        momentum=0.0,
        mode=mode,
        numerical_view=view,
        force_derivative=0.0,
    )
    assert position > 1.0 and momentum > 0.0
    with pytest.raises(ValueError, match="nominal"):
        discrete_tangent_step(
            replace(state, alpha_tilde_x=0.7),
            after,
            NativeStateDifferential.zero(),
            mode=mode,
            numerical_view=view,
            force_derivative=1.0,
        )
    with pytest.raises(ValueError, match="finite diagnostic"):
        preparation_pulse_amplitude(
            native_step=0, invocation_tick=0, refinement=1, sign=1, magnitude=4.0
        )
    with pytest.raises(ValueError, match="reference tick"):
        preparation_pulse_amplitude(
            native_step=0, invocation_tick=True, refinement=1, sign=1, magnitude=8.0
        )
