'Bounded independent equations for the retained quantum trajectory burn-in-screen native operators.'

from __future__ import annotations

from math import cos, exp, sin, sqrt

import numpy as np
import pytest

from empirical_lawhood.adapters.simulators.quantum_trajectory_burnin_screen.basis import build_basis
from empirical_lawhood.adapters.simulators.quantum_trajectory_burnin_screen.jumps import ZeroProjectedMassError, apply_jump, jump_masses, sample_mark
from empirical_lawhood.adapters.simulators.quantum_trajectory_burnin_screen.operators import DenseEigensystemPropagator, SparseKrylovPropagator, build_hamiltonian, lindblad_superoperator, propagate_density
from empirical_lawhood.adapters.simulators.quantum_trajectory_burnin_screen.types import Action


def test_two_site_action_and_cutoff_follow_analytic_rabi_clock() -> None:
    basis = build_basis(2, 1)
    initial = basis.state_vector(1)
    before_action = 0.2
    after_action = 0.4
    hold_h = build_hamiltonian(basis, action=Action.HOLD, epsilon=0.2, j_xy=1, j_z=0)
    hold_at_cutoff = SparseKrylovPropagator(hold_h).propagate(initial, before_action)

    for action, frequency in ((Action.MINUS, 0.8), (Action.HOLD, 1.0), (Action.PLUS, 1.2)):
        hamiltonian = build_hamiltonian(
            basis, action=action, epsilon=0.2, j_xy=1, j_z=0
        )
        expected_after = np.asarray(
            [
                cos(before_action) * cos(frequency * after_action)
                - sin(before_action) * sin(frequency * after_action),
                -1j
                * (
                    sin(before_action) * cos(frequency * after_action)
                    + cos(before_action) * sin(frequency * after_action)
                ),
            ]
        )
        for propagator in (
            SparseKrylovPropagator(hamiltonian),
            DenseEigensystemPropagator(hamiltonian),
        ):
            observed = propagator.propagate(hold_at_cutoff, after_action)
            np.testing.assert_allclose(observed, expected_after, rtol=0, atol=2e-13)
            np.testing.assert_allclose(np.linalg.norm(observed), 1, rtol=0, atol=2e-13)

    assert not np.allclose(
        SparseKrylovPropagator(build_hamiltonian(basis, action=Action.MINUS, epsilon=0.2, j_xy=1, j_z=0)).propagate(
            hold_at_cutoff, after_action
        ),
        SparseKrylovPropagator(build_hamiltonian(basis, action=Action.PLUS, epsilon=0.2, j_xy=1, j_z=0)).propagate(
            hold_at_cutoff, after_action
        ),
    )


def test_projected_jump_mass_is_distinct_from_normalized_post_jump_state() -> None:
    basis = build_basis(2, 1)
    state = (sqrt(0.25) * basis.state_vector(1)) + (
        sqrt(0.75) * basis.state_vector(2)
    )
    masses = jump_masses(basis, state)
    np.testing.assert_allclose(masses.projected_masses, [0.25, 0.75], rtol=0, atol=1e-15)
    np.testing.assert_allclose(masses.mark_probabilities, [0.25, 0.75], rtol=0, atol=1e-15)
    assert sample_mark(masses.mark_probabilities, 0.25) == 1

    jumped = apply_jump(basis, state, 0, expected_mass=0.25)
    np.testing.assert_allclose(jumped.state, basis.state_vector(1), rtol=0, atol=1e-15)
    assert jumped.projected_vector_norm_squared == pytest.approx(0.25)
    assert jumped.post_jump_norm_residual == pytest.approx(0)
    with pytest.raises(ZeroProjectedMassError, match="ZERO_PROJECTED_MASS"):
        apply_jump(basis, basis.state_vector(2), 0)
    with pytest.raises(ValueError, match="normalized"):
        sample_mark(np.asarray([0.25, 0.5]), 0.5)


def test_two_site_lindblad_dephasing_preserves_population_and_trace() -> None:
    basis = build_basis(2, 1)
    hamiltonian = build_hamiltonian(
        basis, action=Action.HOLD, epsilon=0, j_xy=0, j_z=0
    )
    generator = lindblad_superoperator(basis, hamiltonian, gamma=0.3)
    initial = np.full((2, 2), 0.5, dtype=np.complex128)
    observed = propagate_density(generator, initial, 2.0)
    expected = np.asarray([[0.5, 0.5 * exp(-0.6)], [0.5 * exp(-0.6), 0.5]])
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-13)
    assert np.trace(observed) == pytest.approx(1)
