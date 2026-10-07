"""Full real-HS endogenous Hessian diagnostic for the exact Response geometry member."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .gradients import SixMatrixParameters, analytic_gradient_terms, commutator
from .model import ComplexArray, hermiticity_residual, hermitian_part
from .shooting import hermitian_basis


@dataclass(frozen=True, slots=True)
class EndogenousHessian:
    matrix: npt.NDArray[np.float64]
    refinement_error: float
    symmetry_error: float
    resolved: bool


def _hessian_image(positions: ComplexArray, perturbation: ComplexArray) -> ComplexArray:
    parameters = SixMatrixParameters(2, 0.5, 0.5, 1.0, 2 / 3, 22 / 3)
    output = np.zeros_like(positions)
    for sector, alpha, beta in (
        (0, parameters.alpha_x, parameters.beta_x),
        (1, parameters.alpha_y, parameters.beta_y),
    ):
        z, v = positions[sector], perturbation[sector]
        other, w = positions[1 - sector], perturbation[1 - sector]
        radius = np.sum(z @ z, axis=0)
        delta_radius = np.sum(v @ z + z @ v, axis=0)
        for a in range(3):
            result = 2 * beta * v[a]
            b, c = ((1, 2), (2, 0), (0, 1))[a]
            result = result + 2j * alpha * (commutator(v[b], z[c]) + commutator(z[b], v[c]))
            result = (
                result + delta_radius @ z[a] + z[a] @ delta_radius + radius @ v[a] + v[a] @ radius
            )
            for b in range(3):
                result = result + commutator(v[b], commutator(z[b], z[a]))
                result = result + commutator(z[b], commutator(v[b], z[a]))
                result = result + commutator(z[b], commutator(z[b], v[a]))
                result = result + commutator(w[b], commutator(other[b], z[a]))
                result = result + commutator(other[b], commutator(w[b], z[a]))
                result = result + commutator(other[b], commutator(other[b], v[a]))
            output[sector, a] = 4 * result
    return hermitian_part(output)


def endogenous_hessian(positions: ComplexArray) -> EndogenousHessian:
    """Apply frozen common assay's three fixed differences to all 96 position coordinates.

    The output retains identity/trace coordinates, both coupled sectors and the
    leading n in the installed full-potential gradient. It does not project native
    dynamics or infer a stable law from an instantaneous Hessian.
    """
    if positions.shape != (2, 3, 4, 4) or positions.dtype != np.dtype("complex128"):
        raise ValueError("Response geometry Hessian requires q=2 complex128 positions")
    if not np.isfinite(positions).all() or hermiticity_residual(positions) > 1e-12:
        raise ValueError("Response geometry Hessian requires finite Hermitian positions")
    parameters = SixMatrixParameters(2, 0.5, 0.5, 1.0, 2 / 3, 22 / 3)
    basis = hermitian_basis(4)
    matrices = []
    analytic = np.empty((96, 96), dtype=np.float64)
    position_scale = max(1.0, float(np.linalg.norm(positions)))
    for relative_step in (1e-4, 1e-5, 1e-6):
        epsilon = relative_step * position_scale
        matrix = np.empty((96, 96), dtype=np.float64)
        for column in range(96):
            sector, component, coordinate = np.unravel_index(column, (2, 3, 16))
            direction = np.zeros_like(positions)
            direction[sector, component] = basis[coordinate]
            if relative_step == 1e-5:
                analytic[:, column] = np.einsum(
                    "kab,ijab->ijk", basis.conj(), _hessian_image(positions, direction)
                ).real.ravel()
            image = (
                analytic_gradient_terms(positions + epsilon * direction, parameters).total
                - analytic_gradient_terms(positions - epsilon * direction, parameters).total
            ) / (2 * epsilon)
            matrix[:, column] = np.einsum("kab,ijab->ijk", basis.conj(), image).real.ravel()
        matrices.append(matrix)
    scale = max(1.0, float(np.linalg.norm(matrices[1], 2)))
    refinement_error = max(
        float(np.linalg.norm(matrices[1] - other, 2)) / scale
        for other in (matrices[0], matrices[2], analytic)
    )
    symmetry_error = float(np.linalg.norm(matrices[1] - matrices[1].T, 2)) / scale
    matrix = (matrices[1] + matrices[1].T) / 2
    frozen = np.frombuffer(matrix.tobytes(), dtype=np.float64).reshape(96, 96)
    return EndogenousHessian(
        frozen,
        refinement_error,
        symmetry_error,
        refinement_error <= 1e-6 and symmetry_error <= 1e-6,
    )
