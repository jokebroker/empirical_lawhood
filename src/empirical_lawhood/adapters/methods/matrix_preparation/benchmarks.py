"""Invocation-only privileged benchmarks with the imported Gaussian semantics.

The historical standalone Gaussian/calibration analyses remain immutable. This
new scientific owner retains their continuous local-harmonic benchmark on the
new readout grid; it is distinct from the actual native-map tangent diagnostic.
No future trajectory is accepted by the benchmark interface.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from empirical_lawhood.adapters.simulators.six_matrix_response.gradients import SixMatrixParameters, analytic_gradient_terms
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ComplexArray, hermiticity_residual
from empirical_lawhood.adapters.simulators.six_matrix_response.response_hessian import _hessian_image
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import hermitian_basis


RealArray = npt.NDArray[np.float64]
PARAMETERS = SixMatrixParameters(2, 0.5, 0.5, 1.0, 2 / 3, 22 / 3)
PRIVILEGED_MODELS = ("scalar", "full_hessian", "microscopic")


def real_coordinates(fields: ComplexArray) -> RealArray:
    return np.asarray(
        np.einsum("kab,ijab->ijk", hermitian_basis(4).conj(), fields).real.ravel(),
        dtype=np.float64,
    )


def full_hessian(positions: ComplexArray) -> tuple[RealArray, float]:
    if positions.shape != (2, 3, 4, 4) or positions.dtype != np.dtype("complex128"):
        raise ValueError("privileged benchmark requires exact q=2 positions")
    if not np.isfinite(positions).all() or hermiticity_residual(positions) > 1e-12:
        raise ValueError("privileged benchmark positions must be finite Hermitian")
    basis, result = hermitian_basis(4), np.empty((96, 96))
    for column in range(96):
        sector, spatial, coordinate = np.unravel_index(column, (2, 3, 16))
        direction = np.zeros_like(positions)
        direction[sector, spatial] = basis[coordinate]
        result[:, column] = real_coordinates(_hessian_image(positions, direction))
    error = float(np.linalg.norm(result - result.T)) / max(1.0, float(np.linalg.norm(result)))
    if error > 1e-10:
        raise ValueError("native Hessian symmetry is unresolved")
    return (result + result.T) / 2, error


@dataclass(frozen=True)
class MicroscopicInitial:
    curvature: float
    eigenvalues: RealArray
    modal_curvature_gradient: RealArray
    modal_curvature_hessian: RealArray
    modal_force: RealArray
    modal_velocity: RealArray
    modal_direction: RealArray
    symmetry_error: float


def microscopic_initial(
    positions: ComplexArray,
    momenta: ComplexArray,
    mode: ComplexArray,
    curvature_quadratic: RealArray | None = None,
) -> MicroscopicInitial:
    """Full hidden state is explicitly privileged, with unit-mass momentum."""
    if (
        momenta.shape != (2, 3, 4, 4)
        or momenta.dtype != np.dtype("complex128")
        or not np.isfinite(momenta).all()
    ):
        raise ValueError("privileged benchmark requires finite q=2 momenta")
    if mode.shape != (3, 4, 4) or abs(float(np.vdot(mode, mode).real) - 1) > 1e-12:
        raise ValueError("privileged benchmark requires the frozen unit mode")
    direction = np.zeros_like(positions)
    direction[0] = mode
    hessian, error = full_hessian(positions)
    if curvature_quadratic is None:
        curvature_quadratic = curvature_quadratic(mode)
    if curvature_quadratic.shape != (96, 96) or not np.isfinite(curvature_quadratic).all():
        raise ValueError("privileged curvature quadratic has another geometry")
    gradient = analytic_gradient_terms(positions, PARAMETERS).total
    derivative = real_coordinates(
        analytic_gradient_terms(positions + direction, PARAMETERS).total
        + analytic_gradient_terms(positions - direction, PARAMETERS).total
        - 2 * gradient
    )
    vector = real_coordinates(direction)
    eigenvalues, rotation = np.linalg.eigh(hessian)
    return MicroscopicInitial(
        float(vector @ hessian @ vector),
        eigenvalues,
        rotation.T @ derivative,
        rotation.T @ curvature_quadratic @ rotation,
        rotation.T @ -real_coordinates(gradient),
        rotation.T @ real_coordinates(momenta),
        rotation.T @ vector,
        error,
    )


def curvature_quadratic(mode: ComplexArray) -> RealArray:
    direction = np.zeros((2, 3, 4, 4), dtype=np.complex128)
    direction[0] = mode
    zero, _ = full_hessian(np.zeros_like(direction))
    positive, _ = full_hessian(direction)
    negative, _ = full_hessian(-direction)
    return positive + negative - 2 * zero


def modal_operators(eigenvalues: RealArray, dt: float) -> tuple[RealArray, RealArray, RealArray]:
    """Exact mean/noise integrals including negative modes, without clipping."""
    transition, affine, noise = (
        np.empty((len(eigenvalues), 2, 2)),
        np.empty((len(eigenvalues), 2)),
        np.empty((len(eigenvalues), 2, 2)),
    )
    diffusion = np.array([[0.0, 0.0], [0.0, 2.0]])
    for index, value in enumerate(eigenvalues):
        drift = np.array([[0.0, 1.0], [-value, -1.0]])
        augmented = np.zeros((3, 3))
        augmented[:2, :2], augmented[1, 2] = drift, 1.0
        propagated = expm(augmented * dt)
        transition[index], affine[index] = propagated[:2, :2], propagated[:2, 2]
        van_loan = np.block([[drift, diffusion], [np.zeros((2, 2)), -drift.T]])
        integral = expm(van_loan * dt)[:2, 2:] @ transition[index].T
        noise[index] = (integral + integral.T) / 2
    return transition, affine, noise


def _moments(
    mean: RealArray,
    covariance: RealArray,
    operators: tuple[RealArray, RealArray, RealArray],
    force: RealArray,
) -> tuple[RealArray, RealArray]:
    transition, affine, noise = operators
    return (
        np.einsum("nij,nj->ni", transition, mean) + affine * force[:, None],
        np.einsum("nij,njk,nlk->nil", transition, covariance, transition) + noise,
    )


def _curvature(initial: MicroscopicInitial, mean: RealArray, covariance: RealArray) -> float:
    position = mean[:, 0]
    return float(
        initial.curvature
        + initial.modal_curvature_gradient @ position
        + 0.5 * position @ initial.modal_curvature_hessian @ position
        + 0.5 * np.diag(initial.modal_curvature_hessian) @ covariance[:, 0, 0]
    )


@dataclass(frozen=True)
class PrivilegedPrediction:
    gains: RealArray
    microscopic_curvature: RealArray
    minimum_covariance_eigenvalue: float
    symmetry_error: float


def privileged_prediction(
    initial: MicroscopicInitial, refinement: int
) -> PrivilegedPrediction:
    if type(refinement) is not int or refinement not in (1, 2):
        raise ValueError("privileged benchmark changes its declared numerical view")
    dt, count = 0.001 / refinement, 320 * refinement
    full, half = (
        modal_operators(initial.eigenvalues, dt),
        modal_operators(initial.eigenvalues, dt / 2),
    )
    mean = np.column_stack((np.zeros_like(initial.modal_velocity), initial.modal_velocity))
    covariance, tangent = np.zeros((len(mean), 2, 2)), np.zeros_like(mean)
    scalar, microscopic = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0])
    gains, curves = np.zeros((3, count + 1)), np.empty(count + 1)
    curves[0], minimum = initial.curvature, 0.0
    # Scalar invocation curvature is fixed. Caching its two force regimes does
    # not change the imported continuous propagation equations.
    scalar_matrices = tuple(
        expm(np.array([[0.0, 1.0, 0.0], [-initial.curvature, -1.0, force], [0.0, 0.0, 0.0]]) * dt)
        for force in (0.0, 8.0)
    )
    for step in range(count):
        middle_mean, middle_covariance = _moments(mean, covariance, half, initial.modal_force)
        middle_curvature = _curvature(initial, middle_mean, middle_covariance)
        active = int(step < 64 * refinement)
        force = 8.0 * active
        scalar = scalar_matrices[active] @ scalar
        microscopic = (
            expm(
                np.array([[0.0, 1.0, 0.0], [-middle_curvature, -1.0, force], [0.0, 0.0, 0.0]]) * dt
            )
            @ microscopic
        )
        tangent = (
            np.einsum("nij,nj->ni", full[0], tangent)
            + full[1] * (force * initial.modal_direction)[:, None]
        )
        mean, covariance = _moments(mean, covariance, full, initial.modal_force)
        minimum = min(minimum, float(np.linalg.eigvalsh(covariance).min()))
        gains[:, step + 1] = scalar[0], initial.modal_direction @ tangent[:, 0], microscopic[0]
        curves[step + 1] = _curvature(initial, mean, covariance)
    if not np.isfinite(gains).all() or not np.isfinite(curves).all():
        raise FloatingPointError("privileged local forecast became nonfinite")
    return PrivilegedPrediction(gains, curves, minimum, initial.symmetry_error)
