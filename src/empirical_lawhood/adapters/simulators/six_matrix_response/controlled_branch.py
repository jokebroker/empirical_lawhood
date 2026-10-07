"""Exact replay, bounded pulse, and common-noise branch mechanics for six-matrix controlled invariance."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView
from .gradients import SixMatrixParameters, coupling_derivative_density
from .model import ComplexArray, SixMatrixState, ideal_state
from .shooting import brownian_bridge_split, anisotropic_feasibility_rollout_state_sha256
from .simulation import BAOABGradientCache, baoab_step_and_capture_hermitian_noise, baoab_step_with_hermitian_noise, derive_rng_stream, fast_receiver, hermitian_noise
from .scientific_inputs import SixMatrixResponseScientificSeedInput, require_six_matrix_scientific_seed_input
from .spectral import spectral_receiver

ACTION_WORDS = (
    "hold",
    "x-negative",
    "x-positive",
    "y-negative",
    "y-positive",
)


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("six-matrix controlled invariance cannot encode a nonfinite value")
    return Decimal(repr(float(value)))


def _float_array_sha256(*values: np.ndarray) -> str:
    digest = sha256()
    for value in values:
        array = np.ascontiguousarray(value, dtype="<f8")
        digest.update(len(array.tobytes()).to_bytes(8, "big"))
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SixMatrixResponseTransientControlledInvarianceReplayPath:
    """One exact anisotropic feasibility path retained in memory for controlled-invariance probe/control preparation."""

    rollout_id: str
    history_id: str
    seed_index: int
    alpha_x_index: int
    alpha_y_index: int
    states: tuple[SixMatrixState, ...]
    coarse_noises: ComplexArray
    rng_seed_sha256: str
    final_state_sha256: str

    def __post_init__(self) -> None:
        for name in ("rollout_id", "history_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("rng_seed_sha256", "final_state_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        noises = np.asarray(self.coarse_noises)
        if (
            self.seed_index < 0
            or min(self.alpha_x_index, self.alpha_y_index) < 0
            or len(self.states) != 1025
            or tuple(value.step_index for value in self.states) != tuple(range(1025))
            or noises.shape != (1024, 2, 3, 4, 4)
            or noises.dtype != np.dtype("complex128")
            or not np.isfinite(noises).all()
        ):
            raise ValueError("six-matrix controlled invariance replay path geometry differs")
        copied_noises = np.ascontiguousarray(noises, dtype="<c16")
        copied_noises.setflags(write=False)
        object.__setattr__(self, "coarse_noises", copied_noises)

    def state_at(self, step: int) -> SixMatrixState:
        if not 0 <= step <= 1024:
            raise ValueError("six-matrix controlled invariance replay step lies outside 0..1024")
        return self.states[step]

    def y_path(self, *, start_step: int, end_step: int) -> ComplexArray:
        if not 0 <= start_step < end_step <= 1024:
            raise ValueError("six-matrix controlled invariance replay Y-path bounds differ")
        return np.ascontiguousarray(
            np.stack(
                tuple(self.states[step].positions[1] for step in range(start_step, end_step + 1))
            ),
            dtype="<c16",
        )


def _history_couplings(
    *, history_id: str, step: int, ramp_steps: int, target_x: float, target_y: float
) -> tuple[float, float]:
    fraction = min(step, ramp_steps) / ramp_steps
    if history_id == "matrix-history.joint-increasing-coupling":
        return target_x * fraction, target_y * fraction
    first = min(1.0, 2.0 * fraction)
    second = max(0.0, min(1.0, 2.0 * fraction - 1.0))
    if history_id == "matrix-history.x-first-increasing-coupling":
        return target_x * first, target_y * second
    if history_id == "matrix-history.y-first-increasing-coupling":
        return target_x * second, target_y * first
    raise ValueError("six-matrix controlled invariance replay history is outside the feasibility roster")


def _anisotropic_feasibility_grid_target(index: int, supplied: Decimal) -> Decimal:
    """Recover the original Decimal representation used in the original anisotropic feasibility state digest."""

    if index not in range(13):
        raise ValueError("six-matrix controlled invariance anisotropic feasibility grid index lies outside 0..12")
    increment = (Decimal(8) - Decimal(0)) / Decimal(12)
    reconstructed = Decimal(0) + increment * index
    if reconstructed != supplied:
        raise ValueError("six-matrix controlled invariance supplied target differs from its frozen anisotropic feasibility grid index")
    return reconstructed


def replay_anisotropic_feasibility_rollout_path(
    *,
    rollout_id: str,
    history_id: str,
    seed_index: int,
    alpha_x_index: int,
    alpha_y_index: int,
    target_x: Decimal,
    target_y: Decimal,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    scientific_seed_input: SixMatrixResponseScientificSeedInput | None = None,
    total_steps: int = 1024,
    ramp_steps: int = 256,
) -> SixMatrixResponseTransientControlledInvarianceReplayPath:
    """Reproduce one original two-dimensional anisotropic feasibility path including historical receiver calls."""

    scientific_seed = require_six_matrix_scientific_seed_input(
        scientific_seed_input, scientific_role="controlled-parent-replay",
        current_root_id=rollout_id,
        current_context_sha256=numerical_view.fingerprint(), stream_index=seed_index,
    )
    gradient_cache_state = BAOABGradientCache()
    if total_steps != 1024 or ramp_steps != 256 or min(alpha_x_index, alpha_y_index) < 0:
        raise ValueError("six-matrix controlled invariance replay denominator differs")
    # Canonical JSON preserves Decimal value but not exponent/trailing-zero
    # representation.  The original anisotropic feasibility digest included ``str(task.alpha)``;
    # reconstruct the exact design-grid Decimal before hashing.
    target_x = _anisotropic_feasibility_grid_target(alpha_x_index, target_x)
    target_y = _anisotropic_feasibility_grid_target(alpha_y_index, target_y)
    rng, receipt = derive_rng_stream(
        seed_root_id="seed.matrix-anisotropic-feasibility",
        purpose_id=f"purpose.{rollout_id}",
        stream_index=seed_index,
        derivation_rule_id=numerical_view.stream_derivation_rule_id,
        scientific_seed_sha256=scientific_seed.full_seed_sha256,
    )
    state = ideal_state(q=2, alpha_tilde_x=0.0, alpha_tilde_y=0.0, constitution="00")
    states = [state]
    noises: list[ComplexArray] = []
    for step in range(1, total_steps + 1):
        alpha_x, alpha_y = _history_couplings(
            history_id=history_id,
            step=step,
            ramp_steps=ramp_steps,
            target_x=float(target_x),
            target_y=float(target_y),
        )
        state, noise = baoab_step_and_capture_hermitian_noise(
            state,
            member=member,
            numerical_view=numerical_view,
            next_alpha_tilde_x=alpha_x,
            next_alpha_tilde_y=alpha_y,
            rng=rng,
            gradient_cache=gradient_cache_state,
        )
        noises.append(noise)
        if step > 768 and step % 16 == 0:
            fast_receiver(state=state, member=member)
        if step > 768 and step % 256 == 0:
            spectral_receiver(
                receiver_prefix=f"spectrum.{rollout_id}.{step}",
                q=2,
                positions=state.positions,
            )
        states.append(state)
    return SixMatrixResponseTransientControlledInvarianceReplayPath(
        rollout_id=rollout_id,
        history_id=history_id,
        seed_index=seed_index,
        alpha_x_index=alpha_x_index,
        alpha_y_index=alpha_y_index,
        states=tuple(states),
        coarse_noises=np.ascontiguousarray(np.stack(noises), dtype="<c16"),
        rng_seed_sha256=receipt.derived_seed_sha256,
        final_state_sha256=anisotropic_feasibility_rollout_state_sha256(
            state=state,
            target_x=target_x,
            target_y=target_y,
        ),
    )


def brownian_bridge_replay_y_path(
    *,
    replay: SixMatrixResponseTransientControlledInvarianceReplayPath,
    member: SixMatrixResponseModelFamilyMember,
    primary_view: SixMatrixResponseNumericalView,
    half_view: SixMatrixResponseNumericalView,
    target_x: Decimal,
    target_y: Decimal,
    config_fingerprint: str,
    scientific_seed_input: SixMatrixResponseScientificSeedInput | None = None,
) -> tuple[ComplexArray, str]:
    """Replay the same coarse Wiener path at dt/2 and return coarse-aligned Y states."""

    gradient_cache_state = BAOABGradientCache()
    if (
        primary_view.timestep != Decimal("0.001")
        or half_view.timestep != Decimal("0.0005")
        or replay.states[0].step_index != 0
    ):
        raise ValueError("six-matrix controlled invariance half-step replay denominator differs")
    bridge, bridge_seed = derive_noise_block(
        namespace=f"matrix-controlled-invariance.feasibility-bridge.{replay.rollout_id}",
        checkpoint_sha256=config_fingerprint,
        block_index=0,
        step_count=1024,
        scientific_seed_input=scientific_seed_input,
    )
    half_decay = float(
        np.exp(-float(primary_view.friction_gamma) * float(primary_view.timestep) / 2.0)
    )
    state = ideal_state(q=2, alpha_tilde_x=0.0, alpha_tilde_y=0.0, constitution="00")
    y_states = [state.positions[1]]
    for coarse_step, (coarse_noise, bridge_noise) in enumerate(
        zip(replay.coarse_noises, bridge, strict=True), 1
    ):
        first, second = brownian_bridge_split(
            coarse_noise=coarse_noise,
            bridge_noise=bridge_noise,
            half_decay=half_decay,
        )
        for half_offset, noise in enumerate((first, second), 1):
            half_step = 2 * (coarse_step - 1) + half_offset
            alpha_x, alpha_y = _history_couplings(
                history_id=replay.history_id,
                step=half_step,
                ramp_steps=512,
                target_x=float(target_x),
                target_y=float(target_y),
            )
            state = baoab_step_with_hermitian_noise(
                state,
                member=member,
                numerical_view=half_view,
                next_alpha_tilde_x=alpha_x,
                next_alpha_tilde_y=alpha_y,
                standardized_noise=noise,
                gradient_cache=gradient_cache_state,
            )
            y_states.append(state.positions[1])
        if coarse_step > 768 and coarse_step % 16 == 0:
            fast_receiver(state=state, member=member)
        if coarse_step > 768 and coarse_step % 256 == 0:
            spectral_receiver(
                receiver_prefix=f"spectrum.half.{replay.rollout_id}.{coarse_step}",
                q=2,
                positions=state.positions,
            )
    return np.ascontiguousarray(np.stack(y_states), dtype="<c16"), bridge_seed


@dataclass(frozen=True, slots=True)
class SixMatrixResponseTransientControlledInvarianceActionSchedule:
    """One explicit requested schedule; arrays remain native simulator inputs."""

    action_word: str
    trigger_parent_step: int | None
    baseline_x: float
    baseline_y: float
    timestep: float
    alpha_x: np.ndarray
    alpha_y: np.ndarray
    requested_sha256: str

    def __post_init__(self) -> None:
        if self.action_word not in ACTION_WORDS or self.timestep <= 0:
            raise ValueError("six-matrix controlled invariance action schedule identity differs")
        x = np.asarray(self.alpha_x)
        y = np.asarray(self.alpha_y)
        if (
            x.ndim != 1
            or x.shape != y.shape
            or x.size < 1
            or x.dtype != np.dtype("float64")
            or y.dtype != np.dtype("float64")
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("six-matrix controlled invariance action schedule arrays differ")
        validate_sha256(self.requested_sha256, field_name="requested_sha256")
        if self.requested_sha256 != _float_array_sha256(x, y):
            raise ValueError("six-matrix controlled invariance requested schedule hash differs")
        if self.action_word == "hold" and self.trigger_parent_step is not None:
            raise ValueError("six-matrix controlled invariance HOLD cannot have a trigger")
        if self.action_word != "hold" and self.trigger_parent_step is None:
            raise ValueError("six-matrix controlled invariance active schedule requires a trigger")
        copied_x = np.ascontiguousarray(x, dtype="<f8")
        copied_x.setflags(write=False)
        copied_y = np.ascontiguousarray(y, dtype="<f8")
        copied_y.setflags(write=False)
        object.__setattr__(self, "alpha_x", copied_x)
        object.__setattr__(self, "alpha_y", copied_y)


def build_action_schedule(
    *,
    action_word: str,
    branch_start_step: int,
    trigger_parent_step: int | None,
    total_primary_steps: int,
    baseline_x: float,
    baseline_y: float,
    timestep: float,
    multiplier: int = 1,
    delta: float = 0.125,
) -> SixMatrixResponseTransientControlledInvarianceActionSchedule:
    """Build the 48/32/48 pulse or native HOLD at aligned physical times."""

    if (
        action_word not in ACTION_WORDS
        or branch_start_step < 0
        or total_primary_steps < 1
        or multiplier not in {1, 2}
        or timestep <= 0
        or delta != 0.125
    ):
        raise ValueError("six-matrix controlled invariance schedule request differs")
    length = total_primary_steps * multiplier
    x = np.full(length, baseline_x, dtype=np.float64)
    y = np.full(length, baseline_y, dtype=np.float64)
    if action_word != "hold":
        assert trigger_parent_step is not None
        trigger_offset = (trigger_parent_step - branch_start_step) * multiplier
        ramp = 48 * multiplier
        dwell = 32 * multiplier
        returning = 48 * multiplier
        if trigger_offset < 0 or trigger_offset + ramp + dwell + returning > length:
            raise ValueError("six-matrix controlled invariance active pulse does not fit its branch")
        sign = -1.0 if action_word.endswith("negative") else 1.0
        channel = x if action_word.startswith("x-") else y
        baseline = baseline_x if action_word.startswith("x-") else baseline_y
        for index in range(ramp):
            channel[trigger_offset + index] = baseline + sign * delta * (index + 1) / ramp
        channel[trigger_offset + ramp : trigger_offset + ramp + dwell] = baseline + sign * delta
        for index in range(returning):
            channel[trigger_offset + ramp + dwell + index] = baseline + sign * delta * (
                1.0 - (index + 1) / returning
            )
    if min(float(np.min(x)), float(np.min(y))) < 0 or max(float(np.max(x)), float(np.max(y))) > 8:
        raise ValueError("six-matrix controlled invariance schedule exceeds native alpha domain")
    return SixMatrixResponseTransientControlledInvarianceActionSchedule(
        action_word=action_word,
        trigger_parent_step=trigger_parent_step,
        baseline_x=baseline_x,
        baseline_y=baseline_y,
        timestep=timestep,
        alpha_x=x,
        alpha_y=y,
        requested_sha256=_float_array_sha256(x, y),
    )


def derive_noise_block(
    *, namespace: str, checkpoint_sha256: str, block_index: int, step_count: int, q: int = 2,
    scientific_seed_input: SixMatrixResponseScientificSeedInput | None = None,
) -> tuple[ComplexArray, str]:
    validate_stable_id(namespace, field_name="namespace")
    validate_sha256(checkpoint_sha256, field_name="checkpoint_sha256")
    if block_index < 0 or step_count < 1 or q != 2:
        raise ValueError("six-matrix controlled invariance noise-block geometry differs")
    scientific_seed = require_six_matrix_scientific_seed_input(
        scientific_seed_input, scientific_role="controlled-noise-block",
        current_root_id=namespace,
        current_context_sha256=checkpoint_sha256, stream_index=block_index,
    )
    digest = bytes.fromhex(scientific_seed.full_seed_sha256)
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:16], "big")))
    noises = np.ascontiguousarray(
        np.stack(tuple(hermitian_noise(rng=rng, q=q) for _ in range(step_count))),
        dtype="<c16",
    )
    return noises, digest.hex()


def bridge_noise_block(
    *,
    coarse_noises: ComplexArray,
    namespace: str,
    checkpoint_sha256: str,
    block_index: int,
    half_decay: float,
    scientific_seed_input: SixMatrixResponseScientificSeedInput | None = None,
) -> tuple[ComplexArray, str]:
    coarse = np.asarray(coarse_noises)
    if coarse.ndim != 5 or coarse.shape[1:] != (2, 3, 4, 4):
        raise ValueError("six-matrix controlled invariance bridge coarse-noise geometry differs")
    bridge, digest = derive_noise_block(
        namespace=f"{namespace}.bridge",
        checkpoint_sha256=checkpoint_sha256,
        block_index=block_index,
        step_count=coarse.shape[0],
        scientific_seed_input=scientific_seed_input,
    )
    outputs = np.empty((coarse.shape[0] * 2, 2, 3, 4, 4), dtype="<c16")
    for index, (coarse_value, bridge_value) in enumerate(zip(coarse, bridge, strict=True)):
        first, second = brownian_bridge_split(
            coarse_noise=coarse_value,
            bridge_noise=bridge_value,
            half_decay=half_decay,
        )
        outputs[2 * index] = first
        outputs[2 * index + 1] = second
    return outputs, digest


@dataclass(frozen=True, slots=True)
class SixMatrixResponseTransientControlledInvarianceActionLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-transient-controlled-invariance-action-ledger'

    ledger_id: str
    action_word: str
    trigger_parent_step: int | None
    requested_sha256: str
    accepted_sha256: str
    applied_sha256: str
    realized_sha256: str
    maximum_excursion: Decimal
    maximum_increment: Decimal
    total_variation: Decimal
    squared_action_energy: Decimal
    generalized_absolute_work: Decimal
    pulse_count: int
    exact_baseline_return: bool
    clipped: bool
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        if self.action_word not in ACTION_WORDS:
            raise ValueError("six-matrix controlled invariance ledger action differs")
        for name in (
            "requested_sha256",
            "accepted_sha256",
            "applied_sha256",
            "realized_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        for name in (
            "maximum_excursion",
            "maximum_increment",
            "total_variation",
            "squared_action_energy",
            "generalized_absolute_work",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.pulse_count not in {0, 1} or self.clipped:
            raise ValueError("six-matrix controlled invariance ledger pulse/clipping disposition differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_valid = (
            not self.reason_codes
            and self.requested_sha256
            == self.accepted_sha256
            == self.applied_sha256
            == self.realized_sha256
            and self.exact_baseline_return
        )
        if self.valid != expected_valid:
            raise ValueError("six-matrix controlled invariance action-ledger validity differs")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseTransientControlledInvarianceBranchTrace:
    """Native branch output; scientific labels remain method-owned."""

    branch_id: str
    block_index: int
    numerical_view_id: str
    parent_step_multiplier: int
    start_state: SixMatrixState
    final_state: SixMatrixState
    receiver_states: tuple[SixMatrixState, ...]
    y_path: ComplexArray
    noise_seed_sha256: str
    noise_block_sha256: str
    action_ledger: SixMatrixResponseTransientControlledInvarianceActionLedger
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("branch_id", "numerical_view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("noise_seed_sha256", "noise_block_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        path = np.asarray(self.y_path)
        if (
            self.block_index < 0
            or self.parent_step_multiplier not in {1, 2}
            or path.ndim != 4
            or path.shape[1:] != (3, 4, 4)
            or path.dtype != np.dtype("complex128")
            or path.shape[0] != self.final_state.step_index - self.start_state.step_index + 1
            or not np.isfinite(path).all()
        ):
            raise ValueError("six-matrix controlled invariance branch trace geometry differs")
        cadence = 16 * self.parent_step_multiplier
        expected_steps = tuple(
            range(self.start_state.step_index + cadence, self.final_state.step_index + 1, cadence)
        )
        if tuple(value.step_index for value in self.receiver_states) != expected_steps:
            raise ValueError("six-matrix controlled invariance branch receiver cadence differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.valid != (self.action_ledger.valid and not self.reason_codes):
            raise ValueError("six-matrix controlled invariance branch validity differs")
        copied = np.ascontiguousarray(path, dtype="<c16")
        copied.setflags(write=False)
        object.__setattr__(self, "y_path", copied)


def _parameters(state: SixMatrixState, member: SixMatrixResponseModelFamilyMember) -> SixMatrixParameters:
    return SixMatrixParameters(
        q=state.q,
        mass_x=float(member.mass_x),
        mass_y=float(member.mass_y),
        gamma=float(member.cross_coupling_gamma),
        alpha_tilde_x=state.alpha_tilde_x,
        alpha_tilde_y=state.alpha_tilde_y,
    )


def execute_controlled_branch(
    *,
    branch_id: str,
    block_index: int,
    start_state: SixMatrixState,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    schedule: SixMatrixResponseTransientControlledInvarianceActionSchedule,
    noises: ComplexArray,
    noise_seed_sha256: str,
    parent_step_multiplier: int = 1,
    squared_action_energy_ceiling: float = 0.001001,
    generalized_work_ceiling: float = 32.0,
) -> SixMatrixResponseTransientControlledInvarianceBranchTrace:
    """Execute one arm with supplied common noise and four-stage action evidence."""

    gradient_cache_state = BAOABGradientCache()
    validate_stable_id(branch_id, field_name="branch_id")
    validate_sha256(noise_seed_sha256, field_name="noise_seed_sha256")
    innovations = np.asarray(noises)
    if (
        innovations.ndim != 5
        or innovations.shape[1:] != (2, 3, 4, 4)
        or innovations.shape[0] != schedule.alpha_x.size
        or innovations.dtype != np.dtype("complex128")
        or not np.isfinite(innovations).all()
        or parent_step_multiplier not in {1, 2}
        or schedule.alpha_x.size % parent_step_multiplier
        or float(numerical_view.timestep) != schedule.timestep
        or (parent_step_multiplier == 1 and numerical_view.timestep != Decimal("0.001"))
        or (parent_step_multiplier == 2 and numerical_view.timestep != Decimal("0.0005"))
        or start_state.step_index % parent_step_multiplier
        or squared_action_energy_ceiling <= 0
        or generalized_work_ceiling <= 0
    ):
        raise ValueError("six-matrix controlled invariance branch noise geometry differs")
    state = start_state
    states = [state]
    receivers: list[SixMatrixState] = []
    realized_x: list[float] = []
    realized_y: list[float] = []
    work = 0.0
    reasons: set[str] = set()
    for index, noise in enumerate(innovations):
        next_x = float(schedule.alpha_x[index])
        next_y = float(schedule.alpha_y[index])
        derivative_x, derivative_y = coupling_derivative_density(
            state.positions, _parameters(state, member)
        )
        work += abs(derivative_x * (next_x - state.alpha_tilde_x))
        work += abs(derivative_y * (next_y - state.alpha_tilde_y))
        state = baoab_step_with_hermitian_noise(
            state,
            member=member,
            numerical_view=numerical_view,
            next_alpha_tilde_x=next_x,
            next_alpha_tilde_y=next_y,
            standardized_noise=noise,
            gradient_cache=gradient_cache_state,
        )
        states.append(state)
        realized_x.append(state.alpha_tilde_x)
        realized_y.append(state.alpha_tilde_y)
        if (state.step_index - start_state.step_index) % (16 * parent_step_multiplier) == 0:
            receivers.append(state)
    realized_x_array = np.asarray(realized_x, dtype=np.float64)
    realized_y_array = np.asarray(realized_y, dtype=np.float64)
    realized_sha256 = _float_array_sha256(realized_x_array, realized_y_array)
    baseline_x = schedule.baseline_x
    baseline_y = schedule.baseline_y
    excursion = max(
        float(np.max(np.abs(realized_x_array - baseline_x))),
        float(np.max(np.abs(realized_y_array - baseline_y))),
    )
    with_initial_x = np.concatenate(([baseline_x], realized_x_array))
    with_initial_y = np.concatenate(([baseline_y], realized_y_array))
    increment = max(
        float(np.max(np.abs(np.diff(with_initial_x)))),
        float(np.max(np.abs(np.diff(with_initial_y)))),
    )
    total_variation = float(np.sum(np.abs(np.diff(with_initial_x)))) + float(
        np.sum(np.abs(np.diff(with_initial_y)))
    )
    energy = schedule.timestep * float(
        np.sum((realized_x_array - baseline_x) ** 2 + (realized_y_array - baseline_y) ** 2)
    )
    exact_return = bool(realized_x_array[-1] == baseline_x and realized_y_array[-1] == baseline_y)
    if schedule.requested_sha256 != realized_sha256:
        reasons.add("requested-applied-realized-mismatch")
    if excursion > 0.125 + 1e-15:
        reasons.add("pulse-amplitude-exceeded")
    maximum_increment = 1.0 / 384.0 if numerical_view.timestep == Decimal("0.001") else 1.0 / 768.0
    if increment > maximum_increment + 1e-15:
        reasons.add("pulse-rate-exceeded")
    if total_variation > 0.25 + 1e-14:
        reasons.add("pulse-total-variation-exceeded")
    if energy > squared_action_energy_ceiling + 1e-15:
        reasons.add("squared-action-energy-exceeded")
    if work > generalized_work_ceiling + 1e-12:
        reasons.add("generalized-work-exceeded")
    if not exact_return:
        reasons.add("baseline-return-failed")
    if not state.finite:
        reasons.add("nonfinite-final-state")
    action_ledger = SixMatrixResponseTransientControlledInvarianceActionLedger(
        ledger_id=f"ledger.{branch_id}",
        action_word=schedule.action_word,
        trigger_parent_step=schedule.trigger_parent_step,
        requested_sha256=schedule.requested_sha256,
        accepted_sha256=schedule.requested_sha256,
        applied_sha256=schedule.requested_sha256,
        realized_sha256=realized_sha256,
        maximum_excursion=_decimal(excursion),
        maximum_increment=_decimal(increment),
        total_variation=_decimal(total_variation),
        squared_action_energy=_decimal(energy),
        generalized_absolute_work=_decimal(work),
        pulse_count=int(schedule.action_word != "hold"),
        exact_baseline_return=exact_return,
        clipped=False,
        valid=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )
    noise_block_sha256 = sha256(
        np.ascontiguousarray(innovations, dtype="<c16").tobytes()
    ).hexdigest()
    return SixMatrixResponseTransientControlledInvarianceBranchTrace(
        branch_id=branch_id,
        block_index=block_index,
        numerical_view_id=numerical_view.view_id,
        parent_step_multiplier=parent_step_multiplier,
        start_state=start_state,
        final_state=state,
        receiver_states=tuple(receivers),
        y_path=np.ascontiguousarray(
            np.stack(tuple(value.positions[1] for value in states)), dtype="<c16"
        ),
        noise_seed_sha256=noise_seed_sha256,
        noise_block_sha256=noise_block_sha256,
        action_ledger=action_ledger,
        valid=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


def checkpoint_state_sha256(state: SixMatrixState) -> str:
    digest = sha256()
    digest.update(state.positions.tobytes(order="C"))
    digest.update(state.momenta.tobytes(order="C"))
    digest.update(str(state.step_index).encode())
    digest.update(repr(state.alpha_tilde_x).encode())
    digest.update(repr(state.alpha_tilde_y).encode())
    return digest.hexdigest()


def canonical_state_summary(state: SixMatrixState) -> dict[str, object]:
    """Compact native state summary for HDF5/receipt metadata, not a verdict."""

    return {
        "q": state.q,
        "step_index": state.step_index,
        "alpha_tilde_x": repr(state.alpha_tilde_x),
        "alpha_tilde_y": repr(state.alpha_tilde_y),
        "positions_sha256": sha256(state.positions.tobytes(order="C")).hexdigest(),
        "momenta_sha256": sha256(state.momenta.tobytes(order="C")).hexdigest(),
        "state_sha256": checkpoint_state_sha256(state),
    }


def rng_state_sha256(rng: np.random.Generator) -> str:
    payload = json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


__all__ = [
    "ACTION_WORDS",
    'SixMatrixResponseTransientControlledInvarianceActionLedger',
    'SixMatrixResponseTransientControlledInvarianceActionSchedule',
    'SixMatrixResponseTransientControlledInvarianceBranchTrace',
    'SixMatrixResponseTransientControlledInvarianceReplayPath',
    'bridge_noise_block',
    'brownian_bridge_replay_y_path',
    'build_action_schedule',
    'canonical_state_summary',
    'checkpoint_state_sha256',
    'derive_noise_block',
    'execute_controlled_branch',
    'replay_anisotropic_feasibility_rollout_path',
    'rng_state_sha256',
]
