"""Response geometry native force instruments; no source access or scientific verdicts.

The force is a new action contract, not an extension of frozen Six-matrix response coupling
requests. All clocks below count reference ticks of 0.001 Langevin time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView, MatrixIntegratorKind
from .controlled_branch import build_action_schedule
from .model import ComplexArray, SixMatrixState, hermiticity_residual, hermitian_part
from .simulation import BAOABGradientCache, baoab_step_with_hermitian_noise


@dataclass(frozen=True, slots=True)
class ResponseGeometryNativeForcePulse(CanonicalRecord):
    """One finite short-pulse-response/extended-pulse-response action occurrence, fixed before its first interval."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-native-force-pulse'

    occurrence_id: str
    assay: str
    sign: int
    invocation_tick: int

    def __post_init__(self) -> None:
        validate_stable_id(self.occurrence_id, field_name="occurrence_id")
        if self.assay not in {"short-pulse-response", "extended-pulse-response"}:
            raise ValueError("Response geometry force assay must be short-pulse-response or extended-pulse-response")
        if type(self.sign) is not int or self.sign not in {-1, 0, 1}:
            raise ValueError("Response geometry force sign must be NEG, HOLD or POS")
        if type(self.invocation_tick) is not int or self.invocation_tick < 0:
            raise ValueError("Response geometry invocation must be a nonnegative reference tick")

    @property
    def pulse_ticks(self) -> int:
        return 64 if self.assay == "short-pulse-response" else 128

    @property
    def horizon_ticks(self) -> int:
        return 320 if self.assay == "short-pulse-response" else 640

    @property
    def amplitude(self) -> float:
        return 8.0 if self.assay == "short-pulse-response" else 4.0

    def interval_force(self, *, native_step: int, refinement: int) -> float:
        """The SAME value belongs to both kicks of this completed interval."""
        if type(refinement) is not int or refinement not in {1, 2, 4}:
            raise ValueError("Response geometry permits only primary, half and quarter steps")
        if type(native_step) is not int or native_step < 0:
            raise ValueError("Response geometry native interval index must be nonnegative")
        start = self.invocation_tick * refinement
        end = start + self.pulse_ticks * refinement
        return self.sign * self.amplitude if start <= native_step < end else 0.0


def _triplet(value: ComplexArray, *, name: str) -> ComplexArray:
    array = np.asarray(value)
    if array.shape != (3, 4, 4) or array.dtype != np.dtype("complex128"):
        raise ValueError(f"Response geometry {name} must have q=2 Hermitian triplet geometry")
    if not np.isfinite(array).all() or hermiticity_residual(array) > 1e-12:
        raise ValueError(f"Response geometry {name} must be finite and Hermitian")
    return array


def force_step(
    state: SixMatrixState,
    *,
    pulse: ResponseGeometryNativeForcePulse,
    mode: ComplexArray,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    standardized_noise: ComplexArray,
    gradient_cache: BAOABGradientCache | None = None,
) -> SixMatrixState:
    """Apply a unit-mass X force outside the existing conservative cache.

    Couplings are fixed during this assay. The source runner must bind the mode,
    invocation, realization ledger and complete checkpoint to the issued action.
    This helper has no authority, RNG, scheduler, persistence or retry policy.
    """
    direction = _triplet(mode, name="force mode")
    if abs(float(np.vdot(direction, direction).real) - 1.0) > 1e-12:
        raise ValueError("Response geometry force mode must have unit real HS norm")
    if state.q != 2 or not state.finite:
        raise ValueError("Response geometry force state must be finite q=2")
    if abs(state.alpha_tilde_x - 2 / 3) > 1e-12 or abs(state.alpha_tilde_y - 22 / 3) > 1e-12:
        raise ValueError("Response geometry force requires the declared target couplings after the parent")
    if (
        member.mass_x != Decimal("0.5")
        or member.mass_y != Decimal("0.5")
        or member.cross_coupling_gamma != Decimal(1)
        or numerical_view.friction_gamma != Decimal(1)
        or numerical_view.bath_temperature != Decimal(1)
        or numerical_view.integrator is not MatrixIntegratorKind.BAOAB_UNDERDAMPED_LANGEVIN
    ):
        raise ValueError("Response geometry force requires the declared M=0.5, gamma=bath=friction=1")
    refinements = {Decimal("0.001"): 1, Decimal("0.0005"): 2, Decimal("0.00025"): 4}
    if numerical_view.timestep not in refinements:
        raise ValueError("Response geometry force timestep is outside the declared views")
    amplitude = pulse.interval_force(
        native_step=state.step_index, refinement=refinements[numerical_view.timestep]
    )
    kick = np.zeros_like(state.momenta)
    kick[0] = 0.5 * float(numerical_view.timestep) * amplitude * direction
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


def select_past_mode(
    *, ticks: tuple[int, ...], observations: ComplexArray, cutoff_tick: int
) -> ComplexArray | None:
    """Freeze frozen common assay's oriented largest-variance X direction, or return unresolved.

    Only the exact sixteen-sample prefix ending at the pre-parent cutoff enters.
    Reject supplied future data rather than silently select a favorable window.
    """
    if type(cutoff_tick) is not int or cutoff_tick < 240:
        raise ValueError("Response geometry mode cutoff lacks a complete past window")
    expected = tuple(range(cutoff_tick - 240, cutoff_tick + 1, 16))
    if ticks != expected or any(type(tick) is not int for tick in ticks):
        raise ValueError("Response geometry mode requires the exact causal 16-tick window")
    values = np.asarray(observations)
    if values.shape != (16, 3, 4, 4) or values.dtype != np.dtype("complex128"):
        raise ValueError("Response geometry mode observations have another shape or dtype")
    if not np.isfinite(values).all() or hermiticity_residual(values) > 1e-12:
        return None
    centered = values - values.mean(axis=0)
    # The 16x16 Gram eigensystem yields the nonzero spectrum of the real HS
    # covariance without choosing a coordinate frame in the Hermitian space.
    rows = centered.reshape(16, -1)
    gram = (rows.conj() @ rows.T).real / 15.0
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    if eigenvalues[-1] - eigenvalues[-2] <= 1e-6 * max(1.0, float(np.trace(gram))):
        return None
    mode = np.einsum("s,sabc->abc", eigenvectors[:, -1], centered)
    norm = float(np.linalg.norm(mode))
    if not np.isfinite(norm) or norm <= 1e-10:
        return None
    mode = hermitian_part(mode / norm)
    overlaps = np.einsum("sabc,abc->s", centered.conj(), mode).real
    resolved = np.flatnonzero(np.abs(overlaps) > 1e-10)
    if not len(resolved):
        return None
    if overlaps[resolved[0]] < 0:
        mode = -mode
    return np.frombuffer(mode.tobytes(), dtype="<c16").reshape(3, 4, 4)


def signed_displacement(
    *, mode: ComplexArray, origin: ComplexArray, current: ComplexArray
) -> tuple[float, float, float]:
    """Return total, trace and traceless X displacement in native HS units."""
    direction = _triplet(mode, name="receiver mode")
    if abs(float(np.vdot(direction, direction).real) - 1.0) > 1e-12:
        raise ValueError("Response geometry receiver mode must have unit HS norm")
    displacement = _triplet(current, name="current X") - _triplet(origin, name="origin X")
    total = float(np.vdot(direction, displacement).real)
    trace = float(
        np.vdot(
            np.trace(direction, axis1=-2, axis2=-1), np.trace(displacement, axis1=-2, axis2=-1)
        ).real
        / 4
    )
    return total, trace, total - trace


def parent_schedule(*, parent: str, refinement: int) -> npt.NDArray[np.float64]:
    """frozen common assay's existing five 384-tick coupling paths, with a quarter diagnostic.

    Columns are post-interval (alpha_tilde_X, alpha_tilde_Y). Primary and half
    views reuse the installed schedule bytes. Quarter midpoints linearly split
    each half interval; no frozen predecessor validator is relaxed.
    """
    if type(refinement) is not int or refinement not in {1, 2, 4}:
        raise ValueError("Response geometry parent refinement differs")
    base_refinement = min(refinement, 2)
    source = build_action_schedule(
        action_word=parent,
        branch_start_step=0,
        trigger_parent_step=None if parent == "hold" else 0,
        total_primary_steps=384,
        baseline_x=2 / 3,
        baseline_y=22 / 3,
        timestep=0.001 / base_refinement,
        multiplier=base_refinement,
    )
    array = np.column_stack((source.alpha_x, source.alpha_y))
    if refinement == 4:
        start = np.array([[2 / 3, 22 / 3]])
        preceding = np.concatenate((start, array[:-1]))
        quarter = np.empty((1536, 2), dtype=np.float64)
        quarter[::2], quarter[1::2] = (preceding + array) / 2, array
        array = quarter
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(384 * refinement, 2)
