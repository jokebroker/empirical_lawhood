"""Pure two-port instruments and exact finite control clocks.

These functions do not acquire roots, grant authority, select a controller or
decide scientific support. Native tensors end at this instrument boundary.
"""

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.simulators.six_matrix_response.gradients import SixMatrixParameters, analytic_gradient_terms
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ComplexArray, SixMatrixState, hermitian_part, hermiticity_residual
from empirical_lawhood.adapters.simulators.six_matrix_response.response_assay import select_past_mode
from empirical_lawhood.adapters.simulators.six_matrix_response.response_hessian import _hessian_image
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView
from empirical_lawhood.adapters.simulators.six_matrix_response.preparation_differential import preparation_force_step
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import BAOABGradientCache
from empirical_lawhood.adapters.methods.matrix_preparation.projection import observable_features

from .contracts import DIRECTIONS, PARENTS, PreparedForceWord, PreparedRoot

Array = npt.NDArray[np.float64]


Scalar = TypeVar("Scalar", bound=np.generic)


def _frozen(value: npt.NDArray[Scalar]) -> npt.NDArray[Scalar]:
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


@dataclass(frozen=True, slots=True)
class PreparedPortFrame:
    cutoff_tick: int
    modes: ComplexArray

    def __post_init__(self) -> None:
        value = self.modes
        if (
            type(self.cutoff_tick) is not int
            or self.cutoff_tick < 240
            or value.shape != (2, 3, 4, 4)
            or value.dtype != np.dtype("complex128")
            or not np.isfinite(value).all()
            or hermiticity_residual(value) > 1e-12
        ):
            raise ValueError("prepared frame requires two finite causal Hermitian X modes")
        gram = np.einsum("iabc,jabc->ij", value.conj(), value).real
        if not np.allclose(gram, np.eye(2), atol=1e-12, rtol=0):
            raise ValueError("prepared frame is not real-HS orthonormal")
        object.__setattr__(self, "modes", _frozen(np.asarray(value, dtype="<c16")))


def select_prepared_ports(
    *, ticks: tuple[int, ...], observations: ComplexArray, cutoff_tick: int
) -> PreparedPortFrame | None:
    first = select_past_mode(ticks=ticks, observations=observations, cutoff_tick=cutoff_tick)
    if first is None:
        return None
    centered = observations - observations.mean(axis=0)
    overlaps = np.einsum("sabc,abc->s", centered.conj(), first).real
    residual = centered - overlaps[:, None, None, None] * first
    rows = residual.reshape(16, -1)
    gram = (rows.conj() @ rows.T).real / 15
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues[-1] - eigenvalues[-2] <= 1e-6 * max(1.0, float(np.trace(gram))):
        return None
    second = np.einsum("s,sabc->abc", eigenvectors[:, -1], residual)
    second -= float(np.vdot(first, second).real) * first
    norm = float(np.linalg.norm(second))
    if not np.isfinite(norm) or norm <= 1e-10:
        return None
    second = hermitian_part(second / norm)
    overlaps = np.einsum("sabc,abc->s", centered.conj(), second).real
    resolved = np.flatnonzero(abs(overlaps) > 1e-10)
    if not len(resolved):
        return None
    if overlaps[resolved[0]] < 0:
        second = -second
    return PreparedPortFrame(cutoff_tick, np.stack((first, second)))


def prepared_parent_schedule(*, parent: str, refinement: int) -> Array:
    """Post-interval couplings: returned by 256, then sixteen baseline ticks."""
    if parent not in PARENTS or type(refinement) is not int or refinement not in (1, 2):
        raise ValueError("prepared parent/refinement is outside its native contract")
    result = np.tile(np.array([2 / 3, 22 / 3]), (272 * refinement, 1))
    if parent != "hold":
        duration = 2 if parent.endswith("256") else 1
        ramp, dwell = 48 * duration * refinement, 32 * duration * refinement
        sign = -1 if "negative" in parent else 1
        result[:ramp, 1] += sign * 0.125 * np.arange(1, ramp + 1) / ramp
        result[ramp : ramp + dwell, 1] += sign * 0.125
        result[ramp + dwell : 2 * ramp + dwell, 1] += (
            sign * 0.125 * (1 - np.arange(1, ramp + 1) / ramp)
        )
    return _frozen(result)


def prepared_force_components(
    word: PreparedForceWord, *, native_step: int, invocation_tick: int, refinement: int
) -> Array:
    if (
        type(refinement) is not int
        or refinement not in (1, 2)
        or type(native_step) is not int
        or native_step < 0
        or type(invocation_tick) is not int
        or invocation_tick < 0
    ):
        raise ValueError("prepared force clock is outside its declared two numerical views")
    components = np.zeros(2)
    start = invocation_tick * refinement
    if word.sign and start <= native_step < start + 64 * refinement:
        direction = np.array(DIRECTIONS[word.direction_index], dtype=float)
        components = word.sign * float(word.magnitude) * direction / np.linalg.norm(direction)
    return _frozen(components)


def prepared_receiver(frame: PreparedPortFrame, displacement: ComplexArray) -> Array:
    if (
        displacement.shape != (3, 4, 4)
        or displacement.dtype != np.dtype("complex128")
        or not np.isfinite(displacement).all()
        or hermiticity_residual(displacement) > 1e-12
    ):
        raise ValueError("prepared receiver requires a finite Hermitian X displacement")
    return _frozen(np.einsum("pabc,abc->p", frame.modes.conj(), displacement).real)


def prepared_force_step(
    state: SixMatrixState,
    *,
    word: PreparedForceWord,
    frame: PreparedPortFrame,
    invocation_tick: int,
    refinement: int,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    standardized_noise: ComplexArray,
    gradient_cache: BAOABGradientCache | None = None,
    expected_parent_ticks: int = 272,
) -> SixMatrixState:
    if (
        type(expected_parent_ticks) is not int
        or expected_parent_ticks <= 0
        or invocation_tick != frame.cutoff_tick + expected_parent_ticks
        or state.step_index < invocation_tick * refinement
    ):
        raise ValueError("prepared force must follow the common parent/wait/delay boundary")
    if float(numerical_view.timestep) != 0.001 / refinement:
        raise ValueError("prepared force view disagrees with its native interval clock")
    components = prepared_force_components(
        word, native_step=state.step_index, invocation_tick=invocation_tick, refinement=refinement
    )
    amplitude = float(np.linalg.norm(components))
    mode = (
        frame.modes[0]
        if amplitude == 0
        else np.einsum("p,pabc->abc", components / amplitude, frame.modes)
    )
    return preparation_force_step(
        state,
        amplitude=amplitude,
        mode=mode,
        member=member,
        numerical_view=numerical_view,
        standardized_noise=standardized_noise,
        gradient_cache=gradient_cache,
    )


def prepared_observation_history(
    *,
    frame: PreparedPortFrame,
    ticks: tuple[int, ...],
    positions: ComplexArray,
    momenta: ComplexArray,
    cutoff_tick: int,
) -> Array:
    """Twelve channels at sixteen samples, each with its own causal past fit.

    Thirty-one native samples supply the first row's sixteen-sample history.
    The port frame is fixed at pre-parent time even for historical projections.
    No acceleration row uses a later sample to fit its past derivative.
    """
    if (
        cutoff_tick not in (frame.cutoff_tick, frame.cutoff_tick + 272)
        or type(cutoff_tick) is not int
        or ticks != tuple(range(cutoff_tick - 480, cutoff_tick + 1, 16))
        or any(type(tick) is not int for tick in ticks)
    ):
        raise ValueError("prepared observer requires the exact pre-parent or handoff causal window")
    return _observation_window(frame, ticks, positions, momenta)


def prepared_training_observation(
    *,
    root: PreparedRoot,
    frame: PreparedPortFrame,
    ticks: tuple[int, ...],
    positions: ComplexArray,
    momenta: ComplexArray,
    cutoff_tick: int,
) -> Array:
    """A dependent refinement-only scalar transition label, never a deployable future input.

    Each row is computed by the same causal instrument as a handoff row.
    The native projection owner retains these arrays; a learner receives only
    the twelve scalars, with root/action/view/time labels.
    """
    if (
        root.stage != 'development'
        or frame.cutoff_tick != root.landmark
        or type(cutoff_tick) is not int
        or cutoff_tick - root.landmark - 272 not in range(0, 321, 16)
        or ticks != tuple(range(cutoff_tick - 480, cutoff_tick + 1, 16))
        or any(type(tick) is not int for tick in ticks)
    ):
        raise ValueError("prepared transition label requires the declared dependent refinement causal cutoff")
    return _frozen(_observation_window(frame, ticks, positions, momenta, last_only=True)[0])


def _observation_window(
    frame: PreparedPortFrame,
    ticks: tuple[int, ...],
    positions: ComplexArray,
    momenta: ComplexArray,
    *,
    last_only: bool = False,
) -> Array:
    for value in (positions, momenta):
        if (
            value.shape != (31, 2, 3, 4, 4)
            or value.dtype != np.dtype("complex128")
            or not np.isfinite(value).all()
            or hermiticity_residual(value) > 1e-12
        ):
            raise ValueError("prepared observer requires thirty-one finite native samples")
    receiver_positions = np.einsum("pabc,sabc->sp", frame.modes.conj(), positions[:, 0]).real
    receiver_velocities = np.einsum("pabc,sabc->sp", frame.modes.conj(), momenta[:, 0]).real
    radii = np.sum(abs(positions[:, 0]) ** 2, axis=(1, 2, 3))
    relative = np.arange(-240, 1, 16, dtype=float)
    time = relative * 0.001
    quadratic = np.column_stack((np.ones(16), time, time * time))
    result = np.empty((1 if last_only else 16, 12))
    for row, sample in enumerate(range(30 if last_only else 15, 31)):
        window = slice(sample - 15, sample + 1)
        spectra = np.linalg.eigvalsh(np.concatenate((positions[sample], momenta[sample])))
        original = observable_features(
            np.array([receiver_positions[sample, 0], receiver_velocities[sample, 0]]),
            spectra,
            receiver_positions[window, 0],
            radii[window],
            relative,
            (ticks[sample] - frame.cutoff_tick) * 0.001,
        )
        acceleration = (
            2 * np.linalg.lstsq(quadratic, receiver_positions[window, 1], rcond=None)[0][2]
        )
        result[row] = np.r_[
            original, receiver_positions[sample, 1], receiver_velocities[sample, 1], acceleration
        ]
    return _frozen(result)


def prepared_mechanism_sketch(state: SixMatrixState, frame: PreparedPortFrame) -> Array:
    """Two force, four Hessian and two full-space residual-norm scalars; two HVPs."""
    if (
        state.q != 2
        or not state.finite
        or abs(state.alpha_tilde_x - 2 / 3) > 1e-12
        or abs(state.alpha_tilde_y - 22 / 3) > 1e-12
    ):
        raise ValueError("prepared sketch requires the declared baseline member")
    parameters = SixMatrixParameters(2, 0.5, 0.5, 1, 2 / 3, 22 / 3)
    gradient = analytic_gradient_terms(state.positions, parameters).total
    force = -prepared_receiver(frame, gradient[0])
    basis = np.zeros((2, 2, 3, 4, 4), dtype=np.complex128)
    basis[:, 0] = frame.modes
    images = np.stack([_hessian_image(state.positions, direction) for direction in basis])
    projected = np.einsum("iabc,jabc->ij", frame.modes.conj(), images[:, 0]).real
    residual = images - np.einsum("ij,iabcd->jabcd", projected, basis)
    norms = np.linalg.norm(residual.reshape(2, -1), axis=1)
    result = np.r_[force, projected.ravel(), norms]
    if not np.isfinite(result).all():
        raise ValueError("prepared mechanism instrument is unresolved")
    return _frozen(result)
