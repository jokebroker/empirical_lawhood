"""Invocation-only Gaussian reduction for six-matrix response forecasts.

The calculation accepts only supplied initial phases and causal observations.
Historical result readers and report writers are deliberately excluded.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from empirical_lawhood.adapters.simulators.six_matrix_response.gradients import SixMatrixParameters, analytic_gradient_terms
from empirical_lawhood.adapters.simulators.six_matrix_response.response_hessian import _hessian_image
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import hermitian_basis

PARAMETERS = SixMatrixParameters(2, 0.5, 0.5, 1.0, 2 / 3, 22 / 3)
FEATURES = (
    "receiver_position", "receiver_velocity", "position_X_spectral_norm_squared",
    "position_Y_spectral_norm_squared", "momentum_X_spectral_norm_squared",
    "momentum_Y_spectral_norm_squared", "position_X_norm_history_slope",
    "receiver_history_acceleration", "invocation_offset_time",
)


def matrices(coordinates):
    return np.einsum(
        "ijk,kab->ijab", np.asarray(coordinates).reshape(2, 3, 16), hermitian_basis(4)
    )


def coordinates(fields):
    return np.einsum("kab,ijab->ijk", hermitian_basis(4).conj(), fields).real.ravel()


def full_hessian(position):
    basis = hermitian_basis(4)
    result = np.empty((96, 96))
    for column in range(96):
        sector, component, coordinate = np.unravel_index(column, (2, 3, 16))
        direction = np.zeros_like(position)
        direction[sector, component] = basis[coordinate]
        result[:, column] = coordinates(_hessian_image(position, direction))
    error = np.linalg.norm(result - result.T) / max(1.0, np.linalg.norm(result))
    if error > 1e-10:
        raise ValueError("Native Hessian symmetry failed")
    return (result + result.T) / 2, float(error)


def microscopic_initial(phase, direction, zero_hessian=None):
    """Exact initial-only native quantities; no trajectory is accepted."""
    phase = np.asarray(phase)
    if phase.shape != (192,) or not np.isfinite(phase).all():
        raise ValueError("Expected only a finite 192-coordinate invocation phase")
    if direction.shape != (2, 3, 4, 4) or not np.isfinite(direction).all():
        raise ValueError("Invalid native direction")
    if abs(np.vdot(direction, direction).real - 1) > 1e-11:
        raise ValueError("Direction must be unit normalized")
    z = matrices(phase[:96])
    h, h_error = full_hessian(z)
    if zero_hessian is None:
        zero_hessian, _ = full_hessian(np.zeros_like(z))
    hp, hp_error = full_hessian(direction)
    hm, hm_error = full_hessian(-direction)
    b = hp + hm - 2 * zero_hessian
    gradient = analytic_gradient_terms(z, PARAMETERS).total
    a = coordinates(
        analytic_gradient_terms(z + direction, PARAMETERS).total
        + analytic_gradient_terms(z - direction, PARAMETERS).total
        - 2 * gradient
    )
    m = coordinates(direction)
    eigenvalues, rotation = np.linalg.eigh(h)
    return dict(
        k0=float(m @ h @ m),
        a=a,
        b=b,
        eigenvalues=eigenvalues,
        modal_a=rotation.T @ a,
        modal_b=rotation.T @ b @ rotation,
        modal_force=rotation.T @ -coordinates(gradient),
        modal_velocity=rotation.T @ phase[96:],
        modal_direction=rotation.T @ m,
        symmetry_error=max(h_error, hp_error, hm_error),
    )


def modal_step(eigenvalues, dt):
    """Exact affine unit-force and noise integrals, including unstable modes."""
    count = len(eigenvalues)
    transition = np.empty((count, 2, 2))
    affine = np.empty((count, 2))
    noise = np.empty((count, 2, 2))
    diffusion = np.array([[0.0, 0.0], [0.0, 2.0]])
    for j, value in enumerate(eigenvalues):
        drift = np.array([[0.0, 1.0], [-value, -1.0]])
        augmented = np.zeros((3, 3))
        augmented[:2, :2] = drift
        augmented[1, 2] = 1
        propagated = expm(augmented * dt)
        transition[j], affine[j] = propagated[:2, :2], propagated[:2, 2]
        van_loan = np.block([[drift, diffusion], [np.zeros((2, 2)), -drift.T]])
        integral = expm(van_loan * dt)[:2, 2:] @ transition[j].T
        noise[j] = (integral + integral.T) / 2
    return transition, affine, noise


def step_moments(mean, covariance, operators, force):
    transition, affine, noise = operators
    new_mean = np.einsum("nij,nj->ni", transition, mean) + affine * force[:, None]
    new_covariance = np.einsum("nij,njk,nlk->nil", transition, covariance, transition) + noise
    return new_mean, new_covariance


def expected_curvature(initial, mean, covariance):
    x = mean[:, 0]
    deterministic = initial["k0"] + initial["modal_a"] @ x + 0.5 * x @ initial["modal_b"] @ x
    thermal = 0.5 * np.diag(initial["modal_b"]) @ covariance[:, 0, 0]
    return np.array([deterministic + thermal, deterministic])


def harmonic_forecast(initial, dt=0.001):
    """Local Gaussian curvature and pulse gain; all coefficients fixed at invocation."""
    if dt not in (0.001, 0.0005):
        raise ValueError("Undeclared propagation step")
    count = int(round(0.320 / dt))
    full = modal_step(initial["eigenvalues"], dt)
    half = modal_step(initial["eigenvalues"], dt / 2)
    mean = np.column_stack((np.zeros_like(initial["modal_velocity"]), initial["modal_velocity"]))
    covariance = np.zeros((len(mean), 2, 2))
    tangent = np.zeros_like(mean)
    response = np.tile([0.0, 0.0, 1.0], (2, 1))
    curves = np.empty((2, count + 1))
    gains = np.empty((2, count + 1))
    coupled = np.zeros(count + 1)
    curves[:, 0], gains[:, 0] = initial["k0"], 0.0
    covariance_minimum = 0.0
    for step in range(count):
        mid_mean, mid_covariance = step_moments(mean, covariance, half, initial["modal_force"])
        k_mid = expected_curvature(initial, mid_mean, mid_covariance)
        forcing = 8.0 if step < round(0.064 / dt) else 0.0
        for model in range(2):
            matrix = np.array([[0.0, 1.0, 0.0], [-k_mid[model], -1.0, forcing], [0.0, 0.0, 0.0]])
            response[model] = expm(matrix * dt) @ response[model]
        tangent = (
            np.einsum("nij,nj->ni", full[0], tangent)
            + full[1] * (forcing * initial["modal_direction"])[:, None]
        )
        mean, covariance = step_moments(mean, covariance, full, initial["modal_force"])
        covariance_minimum = min(covariance_minimum, float(np.linalg.eigvalsh(covariance).min()))
        curves[:, step + 1] = expected_curvature(initial, mean, covariance)
        gains[:, step + 1] = response[:, 0]
        coupled[step + 1] = initial["modal_direction"] @ tangent[:, 0]
    return curves, gains, coupled, covariance_minimum


def observable_features(scalar, spectra, assay_preparation_history, radius_history, relative_ticks, offset):
    """Deployed feature boundary: scalar/spectral observations and causal histories only."""
    if np.shape(scalar) != (2,) or np.shape(spectra) != (4, 3, 4):
        raise ValueError("Expected two receiver scalars and 48 spectral observations")
    if np.shape(assay_preparation_history) != (16,) or np.shape(radius_history) != (16,):
        raise ValueError("Expected 16 scalar history samples")
    if not np.array_equal(relative_ticks, np.arange(-240, 1, 16)):
        raise ValueError("History violates the invocation cutoff or clock")
    t = np.asarray(relative_ticks) * 0.001
    slope = np.linalg.lstsq(np.column_stack((np.ones(16), t)), radius_history, rcond=None)[0][1]
    acceleration = (
        2 * np.linalg.lstsq(np.column_stack((np.ones(16), t, t * t)), assay_preparation_history, rcond=None)[0][2]
    )
    result = np.r_[
        scalar, np.sum(np.asarray(spectra) ** 2, axis=(1, 2)), slope, acceleration, offset
    ]
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite observable at the invocation boundary")
    return result
