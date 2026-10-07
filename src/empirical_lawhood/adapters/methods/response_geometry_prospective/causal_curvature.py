"""Invocation-only curvature and signed-gain forecast calculations.

These functions consume caller-supplied finite phases and forcing directions;
they do not load a historical panel or issue a scientific claim.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.linalg import expm

from empirical_lawhood.adapters.simulators.six_matrix_response.response_hessian import _hessian_image
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import hermitian_basis

TICKS = np.array([0, 32, 64, 96, 128, 192, 256, 320])
TIMES = TICKS * 0.001


def stiffness(positions, direction):
    return float(np.vdot(direction, _hessian_image(positions, direction)).real)


def initial_features(phase, direction):
    """Only the 192 invocation coordinates enter; unit-mass momentum is velocity."""
    phase = np.asarray(phase)
    if phase.shape != (192,) or not np.isfinite(phase).all():
        raise ValueError("Expected the finite invocation phase, not a trajectory")
    if direction.shape != (2, 3, 4, 4) or not np.isfinite(direction).all():
        raise ValueError("Invalid fixed direction")
    if abs(np.vdot(direction, direction).real - 1) > 1e-11:
        raise ValueError("Forcing direction is not unit normalized")
    matrices = np.einsum("ijk,kab->ijab", phase.reshape(4, 3, 16), hermitian_basis(4))
    positions, velocity = matrices[:2], matrices[2:]
    k0 = stiffness(positions, direction)
    h = 1e-5 * max(1.0, np.linalg.norm(positions)) / max(1.0, np.linalg.norm(velocity))
    rates = [
        (
            stiffness(positions + e * velocity, direction)
            - stiffness(positions - e * velocity, direction)
        )
        / (2 * e)
        for e in (h, h / 2)
    ]
    refinement_error = abs(rates[0] - rates[1]) / max(1.0, abs(rates[1]))
    return k0, rates[0], float(refinement_error)


def gain_path(curvatures, max_step=0.001):
    """Propagate the finite short-pulse-response signed-gain tangent; negative curvature is retained."""
    curvatures = np.asarray(curvatures, dtype=float)
    if curvatures.shape != (8,) or not np.isfinite(curvatures).all():
        return np.full(8, np.nan)
    if not 0 < max_step <= 0.001:
        raise ValueError("Propagation step is outside the declared numerical family")
    state = np.array([0.0, 0.0, 1.0])
    values = [0.0]
    for j, span in enumerate(np.diff(TIMES)):
        count = max(1, math.ceil(span / max_step - 1e-10))
        dt = span / count
        for substep in range(count):
            fraction = (substep + 0.5) / count
            k = (1 - fraction) * curvatures[j] + fraction * curvatures[j + 1]
            matrix = np.array(
                [[0.0, 1.0, 0.0], [-k, -1.0, 8.0 if TICKS[j] < 64 else 0.0], [0.0, 0.0, 0.0]]
            )
            with np.errstate(over="ignore", invalid="ignore"):
                state = expm(matrix * dt) @ state
            if not np.isfinite(state).all():
                return np.full(8, np.nan)
        values.append(state[0])
    return np.asarray(values)


def forecast(phase, direction, max_step=0.001):
    k0, rate, derivative_error = initial_features(phase, direction)
    predicted_k = k0 + TIMES * rate
    paths = np.stack((gain_path(np.full(8, k0), max_step), gain_path(predicted_k, max_step)))
    return paths, predicted_k, (k0, rate, derivative_error)
