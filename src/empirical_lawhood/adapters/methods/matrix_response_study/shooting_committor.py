"""Scientific estimators and sole finalizer for the bounded Matrix shooting event assay."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from math import exp, log2, sqrt
from statistics import NormalDist, median
from typing import Any, Callable, ClassVar

import numpy as np
from scipy.optimize import linear_sum_assignment

from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView
from empirical_lawhood.adapters.simulators.six_matrix_response.gradients import SixMatrixParameters, analytic_gradient_terms
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ComplexArray, SixMatrixState, hermiticity_residual, ideal_state
from empirical_lawhood.adapters.simulators.six_matrix_response.scientific_inputs import SixMatrixResponseScientificSeedInput, require_six_matrix_scientific_seed_input
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import BRANCH_DERIVATION_RULE_ID, BRIDGE_DERIVATION_RULE_ID, SixMatrixResponseShootingCommittorCheckpoint, SixMatrixResponseShootingCommittorReplayData, brownian_bridge_split, decode_hermitian_matrices, derive_branch_rng, derive_bridge_rng, encode_hermitian_matrices, state_from_shooting_checkpoint, traceless_hermitian_basis
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import BAOABGradientCache, SixMatrixResponseFastReceiver, baoab_step, baoab_step_with_hermitian_noise, fast_receiver, hermitian_noise
from empirical_lawhood.adapters.simulators.six_matrix_response.spectral import SixMatrixResponseLaplacianSpectrum, spectral_receiver
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .anisotropic_feasibility import MatrixResponseAnisotropicFeasibilityQualificationReport, MatrixResponseAnisotropicFeasibilityRolloutSummary, MatrixResponseAnisotropicFeasibilityStage


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("Matrix shooting scientific metric is nonfinite")
    return Decimal(repr(float(value)))


def _optional_decimal(value: float | None) -> Decimal | None:
    return None if value is None else _decimal(value)


class MatrixResponseShootingCommittorBranchTerminal(StrEnum):
    G_FIRST_WITHIN_HORIZON = "G_FIRST_WITHIN_HORIZON"
    ROBUST_00_FIRST_WITHIN_HORIZON = "ROBUST_00_FIRST_WITHIN_HORIZON"
    RIGHT_CENSORED_NO_TARGET_WITHIN_HORIZON = "RIGHT_CENSORED_NO_TARGET_WITHIN_HORIZON"
    INVALID_NUMERICAL_FUTURE = "INVALID_NUMERICAL_FUTURE"


class MatrixResponseShootingCommittorNumericalDisposition(StrEnum):
    EVENT_PATH_NUMERICALLY_CONCORDANT = "EVENT_PATH_NUMERICALLY_CONCORDANT"
    EVENT_NUMERICAL_VIEW_SENSITIVE = "EVENT_NUMERICAL_VIEW_SENSITIVE"
    PARENT_REPLAY_MISMATCH = "PARENT_REPLAY_MISMATCH"


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorWilsonInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-wilson-interval'

    successes: int
    denominator: int
    confidence_level: Decimal
    estimate: Decimal | None
    lower: Decimal | None
    upper: Decimal | None

    def __post_init__(self) -> None:
        if self.denominator < 0 or not 0 <= self.successes <= self.denominator:
            raise ValueError("Matrix shooting Wilson counts differ")
        validate_decimal(self.confidence_level, field_name="confidence_level", minimum=Decimal(0))
        if not Decimal(0) < self.confidence_level < Decimal(1):
            raise ValueError("Matrix shooting confidence level must lie in (0,1)")
        values = (self.estimate, self.lower, self.upper)
        if self.denominator == 0:
            if any(value is not None for value in values):
                raise ValueError("empty Matrix shooting Wilson interval must be unevaluable")
        elif any(value is None for value in values):
            raise ValueError("nonempty Matrix shooting Wilson interval is incomplete")
        for name in ("estimate", "lower", "upper"):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name, minimum=Decimal(0))
                if value > Decimal(1):
                    raise ValueError("Matrix shooting Wilson value exceeds one")


def wilson_interval(
    successes: int, denominator: int, *, confidence_level: float
) -> MatrixResponseShootingCommittorWilsonInterval:
    if denominator == 0:
        return MatrixResponseShootingCommittorWilsonInterval(
            successes=0,
            denominator=0,
            confidence_level=_decimal(confidence_level),
            estimate=None,
            lower=None,
            upper=None,
        )
    if not 0 <= successes <= denominator or not 0.0 < confidence_level < 1.0:
        raise ValueError("Matrix shooting Wilson inputs differ")
    z = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    p = successes / denominator
    denominator_term = 1.0 + z * z / denominator
    center = (p + z * z / (2.0 * denominator)) / denominator_term
    radius = (
        z * sqrt(p * (1.0 - p) / denominator + z * z / (4.0 * denominator**2)) / denominator_term
    )
    return MatrixResponseShootingCommittorWilsonInterval(
        successes=successes,
        denominator=denominator,
        confidence_level=_decimal(confidence_level),
        estimate=_decimal(p),
        lower=_decimal(max(0.0, center - radius)),
        upper=_decimal(min(1.0, center + radius)),
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorFactorObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-factor-observation'

    factor_id: str
    phi: Decimal
    closure_ratio: Decimal
    kernel_band_ratio: Decimal
    radius_closure_pass: bool
    instantaneous_geometric: bool
    valid: bool

    def __post_init__(self) -> None:
        if self.factor_id not in {"X", "Y"}:
            raise ValueError("Matrix shooting factor must be X or Y")
        for name in ("phi", "closure_ratio", "kernel_band_ratio"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.instantaneous_geometric and not (self.radius_closure_pass and self.valid):
            raise ValueError("Matrix shooting instantaneous geometry lacks valid radius/closure")


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-observation'

    observation_id: str
    local_step: int
    local_time: Decimal
    native_receiver: SixMatrixResponseFastReceiver
    factor_x: MatrixResponseShootingCommittorFactorObservation
    factor_y: MatrixResponseShootingCommittorFactorObservation
    spectral_valid: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        if self.local_step < 0:
            raise ValueError("Matrix shooting observation step cannot be negative")
        validate_decimal(self.local_time, field_name="local_time", minimum=Decimal(0))
        if self.spectral_valid and not (self.factor_x.valid and self.factor_y.valid):
            raise ValueError("Matrix shooting valid spectrum requires valid factor spectra")


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorRollingLabel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-rolling-label'

    endpoint_step: int
    endpoint_time: Decimal
    x_geometric: bool
    y_geometric: bool
    label: str
    strict_window_x_geometric: bool
    strict_window_y_geometric: bool

    def __post_init__(self) -> None:
        if self.endpoint_step < 1 or self.label not in {"00", "10", "01", "11"}:
            raise ValueError("Matrix shooting rolling label geometry differs")
        validate_decimal(self.endpoint_time, field_name="endpoint_time", minimum=Decimal(0))
        if self.label != f"{int(self.x_geometric)}{int(self.y_geometric)}":
            raise ValueError("Matrix shooting rolling label differs from factor flags")


def _kernel_band_ratio(spectrum: SixMatrixResponseLaplacianSpectrum, q: int) -> float:
    values = np.asarray([float(value) for value in spectrum.eigenvalues], dtype=np.float64)
    denominator = float(values[q**2])
    if denominator <= np.finfo(np.float64).eps:
        return 1.0
    return float(values[q**2 - 1] / denominator)


def _factor_observation(
    *,
    factor_id: str,
    alpha_tilde: float,
    q: int,
    radius: float,
    closure_residual: float,
    spectrum: SixMatrixResponseLaplacianSpectrum,
    config: Any,
) -> MatrixResponseShootingCommittorFactorObservation:
    c2 = (q**2 - 1.0) / 4.0
    phi_denominator = (alpha_tilde / q) ** 2 * c2
    normalization_valid = bool(
        np.isfinite(phi_denominator) and phi_denominator > np.finfo(np.float64).tiny
    )
    # A zero coupling has no factor-amplitude normalization.  Preserve that as
    # an invalid observation instead of raising before the caller can retain
    # its declared invalid/unevaluable geometry state.
    phi = sqrt(max(radius, 0.0) / phi_denominator) if normalization_valid else 0.0
    scale = 2.0 * (alpha_tilde / q) / 3.0
    rms_norm = abs(scale) * sqrt((q**2) * max(radius, 0.0) / 3.0)
    closure = closure_residual / max(rms_norm, np.finfo(np.float64).tiny)
    kernel = _kernel_band_ratio(spectrum, q)
    valid = bool(
        normalization_valid and spectrum.valid and np.isfinite((phi, closure, kernel)).all()
    )
    radius_closure = bool(
        valid
        and float(config.phi_min) <= phi <= float(config.phi_max)
        and closure <= float(config.closure_ratio_max)
    )
    return MatrixResponseShootingCommittorFactorObservation(
        factor_id=factor_id,
        phi=_decimal(phi),
        closure_ratio=_decimal(closure),
        kernel_band_ratio=_decimal(kernel),
        radius_closure_pass=radius_closure,
        instantaneous_geometric=bool(
            radius_closure and kernel <= float(config.kernel_band_ratio_max)
        ),
        valid=valid,
    )


def observation_from_receivers(
    *,
    observation_id: str,
    local_step: int,
    local_time: float,
    receiver: SixMatrixResponseFastReceiver,
    spectra: tuple[SixMatrixResponseLaplacianSpectrum, ...],
    config: Any,
) -> MatrixResponseShootingCommittorObservation:
    by_sector = {value.sector: value for value in spectra}
    if set(by_sector) != {"JOINT", "X", "Y"}:
        raise ValueError("Matrix shooting observation requires exact X/Y/JOINT spectra")
    x = _factor_observation(
        factor_id="X",
        alpha_tilde=float(receiver.alpha_tilde_x),
        q=config.q,
        radius=float(receiver.radius_x),
        closure_residual=float(receiver.closure_residual_x),
        spectrum=by_sector["X"],
        config=config,
    )
    y = _factor_observation(
        factor_id="Y",
        alpha_tilde=float(receiver.alpha_tilde_y),
        q=config.q,
        radius=float(receiver.radius_y),
        closure_residual=float(receiver.closure_residual_y),
        spectrum=by_sector["Y"],
        config=config,
    )
    return MatrixResponseShootingCommittorObservation(
        observation_id=observation_id,
        local_step=local_step,
        local_time=_decimal(local_time),
        native_receiver=receiver,
        factor_x=x,
        factor_y=y,
        spectral_valid=x.valid and y.valid and by_sector["JOINT"].valid,
    )


def observe_state(
    *,
    observation_id: str,
    local_step: int,
    local_time: float,
    state: SixMatrixState,
    member: SixMatrixResponseModelFamilyMember,
    config: Any,
) -> MatrixResponseShootingCommittorObservation:
    receiver = fast_receiver(state=state, member=member)
    spectra = spectral_receiver(
        receiver_prefix=f"spectrum.{observation_id}",
        q=state.q,
        positions=state.positions,
    )
    return observation_from_receivers(
        observation_id=observation_id,
        local_step=local_step,
        local_time=local_time,
        receiver=receiver,
        spectra=spectra,
        config=config,
    )


def rolling_labels(
    observations: tuple[MatrixResponseShootingCommittorObservation, ...], *, config: Any
) -> tuple[MatrixResponseShootingCommittorRollingLabel, ...]:
    window_size = config.rolling_window_samples
    if tuple(value.local_step for value in observations) != tuple(
        sorted(set(value.local_step for value in observations))
    ):
        raise ValueError("Matrix shooting observations must have increasing unique local steps")
    outputs: list[MatrixResponseShootingCommittorRollingLabel] = []
    for end in range(window_size, len(observations) + 1):
        window = observations[end - window_size : end]
        endpoint = window[-1]
        if not all(value.factor_x.valid and value.factor_y.valid for value in window):
            continue
        flags: list[tuple[bool, bool]] = []
        for factor_name in ("factor_x", "factor_y"):
            factors = tuple(getattr(value, factor_name) for value in window)
            persistence = sum(value.radius_closure_pass for value in factors)
            primary = bool(
                factors[-1].instantaneous_geometric and persistence >= config.persistence_pass_count
            )
            strict = bool(
                primary
                and max(float(value.kernel_band_ratio) for value in factors)
                <= float(config.kernel_band_ratio_max)
            )
            flags.append((primary, strict))
        outputs.append(
            MatrixResponseShootingCommittorRollingLabel(
                endpoint_step=endpoint.local_step,
                endpoint_time=endpoint.local_time,
                x_geometric=flags[0][0],
                y_geometric=flags[1][0],
                label=f"{int(flags[0][0])}{int(flags[1][0])}",
                strict_window_x_geometric=flags[0][1],
                strict_window_y_geometric=flags[1][1],
            )
        )
    return tuple(outputs)


def _first_time(labels: tuple[MatrixResponseShootingCommittorRollingLabel, ...], target: str) -> Decimal | None:
    return next((value.endpoint_time for value in labels if value.label == target), None)


def _lag_autocorrelation(values: tuple[bool, ...], lag: int) -> float | None:
    if len(values) <= lag:
        return None
    left = np.asarray(values[:-lag], dtype=np.float64)
    right = np.asarray(values[lag:], dtype=np.float64)
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorBranchResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-branch-result'

    branch_id: str
    checkpoint_id: str
    checkpoint_step: int
    branch_index: int
    seed_document_sha256: str
    bit_generator_id: str
    action_disposition: str
    requested_alpha_x: Decimal
    accepted_alpha_x: Decimal
    applied_alpha_x: Decimal
    realized_alpha_x: Decimal
    requested_alpha_y: Decimal
    accepted_alpha_y: Decimal
    applied_alpha_y: Decimal
    realized_alpha_y: Decimal
    observations: tuple[MatrixResponseShootingCommittorObservation, ...]
    rolling_labels: tuple[MatrixResponseShootingCommittorRollingLabel, ...]
    terminal: MatrixResponseShootingCommittorBranchTerminal
    tau_g: Decimal | None
    tau_00: Decimal | None
    tau_00_after_g: Decimal | None
    first_y_loss: Decimal | None
    first_y_regain: Decimal | None
    y_transition_count: int
    y_occupancy: Decimal | None
    rolling_01_occupancy: Decimal | None
    lag1_autocorrelation: Decimal | None
    lag2_autocorrelation: Decimal | None
    terminal_positions_sha256: str
    terminal_momenta_sha256: str
    terminal_rng_state_sha256: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("branch_id", "checkpoint_id", "bit_generator_id", "action_disposition"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.action_disposition != "hold" or not 0 <= self.branch_index < 64:
            raise ValueError("Matrix shooting branch action/index differs")
        validate_sha256(self.seed_document_sha256, field_name="seed_document_sha256")
        for name in (
            "terminal_positions_sha256",
            "terminal_momenta_sha256",
            "terminal_rng_state_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        for name in (
            "requested_alpha_x",
            "accepted_alpha_x",
            "applied_alpha_x",
            "realized_alpha_x",
            "requested_alpha_y",
            "accepted_alpha_y",
            "applied_alpha_y",
            "realized_alpha_y",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if not (
            self.requested_alpha_x
            == self.accepted_alpha_x
            == self.applied_alpha_x
            == self.realized_alpha_x
            and self.requested_alpha_y
            == self.accepted_alpha_y
            == self.applied_alpha_y
            == self.realized_alpha_y
        ):
            raise ValueError("Matrix shooting HOLD action stages differ")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.terminal is MatrixResponseShootingCommittorBranchTerminal.INVALID_NUMERICAL_FUTURE:
            if not self.reason_codes:
                raise ValueError("invalid Matrix shooting branch requires reasons")
        elif self.reason_codes:
            raise ValueError("valid Matrix shooting branch cannot retain failure reasons")
        if self.y_transition_count < 0:
            raise ValueError("Matrix shooting transition count cannot be negative")


@dataclass(frozen=True, slots=True)
class _BranchTask:
    checkpoint: SixMatrixResponseShootingCommittorCheckpoint
    member: SixMatrixResponseModelFamilyMember
    numerical_view: SixMatrixResponseNumericalView
    config: Any
    branch_index: int
    scientific_seed_input: SixMatrixResponseScientificSeedInput


def _require_complete_shooting_seed_inputs(
    scientific_seed_inputs: tuple[SixMatrixResponseScientificSeedInput, ...],
    *,
    scientific_role: str,
    current_root_id: str,
    current_context_sha256: str,
    stream_indices: range,
) -> None:
    """Bind a complete externally authenticated numerical export before work."""
    if type(scientific_seed_inputs) is not tuple or len(scientific_seed_inputs) != len(stream_indices):
        raise ValueError("shooting requires the complete ordered typed scientific seed census")
    original_custody = None
    for stream_index, value in zip(stream_indices, scientific_seed_inputs, strict=True):
        value = require_six_matrix_scientific_seed_input(
            value,
            scientific_role=scientific_role,
            current_root_id=current_root_id,
            current_context_sha256=current_context_sha256,
            stream_index=stream_index,
        )
        custody = (value.source_original_context_sha256, value.original_source, value.export_receipt)
        if original_custody is None:
            original_custody = custody
        elif custody != original_custody:
            raise ValueError("shooting scientific seed census mixes original source or export custody")


def _branch_state_hashes(state: SixMatrixState, rng: np.random.Generator) -> tuple[str, str, str]:
    positions = state.positions.tobytes(order="C")
    momenta = state.momenta.tobytes(order="C")
    rng_state = json_dumps_rng_state(rng)
    return (
        sha256(positions).hexdigest(),
        sha256(momenta).hexdigest(),
        sha256(rng_state.encode("utf-8")).hexdigest(),
    )


def json_dumps_rng_state(rng: np.random.Generator) -> str:
    import json

    return json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":"))


def _run_branch_task(task: _BranchTask) -> MatrixResponseShootingCommittorBranchResult:
    require_six_matrix_scientific_seed_input(
        task.scientific_seed_input,
        scientific_role="shooting-branch",
        current_root_id=f"checkpoint-sha256.{task.checkpoint.combined_state_sha256}",
        current_context_sha256=task.checkpoint.combined_state_sha256,
        stream_index=task.branch_index,
    )
    gradient_cache_state = BAOABGradientCache()
    checkpoint_state, _ = state_from_shooting_checkpoint(task.checkpoint)
    state = SixMatrixState(
        q=checkpoint_state.q,
        positions=checkpoint_state.positions,
        momenta=checkpoint_state.momenta,
        step_index=0,
        alpha_tilde_x=checkpoint_state.alpha_tilde_x,
        alpha_tilde_y=checkpoint_state.alpha_tilde_y,
    )
    rng, seed_hash = derive_branch_rng(
        checkpoint_combined_sha256=task.checkpoint.combined_state_sha256,
        branch_index=task.branch_index,
        scientific_seed_input=task.scientific_seed_input,
    )
    prefix = f"matrix-shooting.branch.step-{task.checkpoint.parent_step:04d}.b{task.branch_index:02d}"
    observations: list[MatrixResponseShootingCommittorObservation] = []
    sample_states: list[SixMatrixState] = []
    checkpoint_observation: MatrixResponseShootingCommittorObservation | None = None
    reasons: set[str] = set()
    try:
        checkpoint_state = state
        for _ in range(task.config.branch_steps):
            state = baoab_step(
                state,
                member=task.member,
                numerical_view=task.numerical_view,
                next_alpha_tilde_x=float(task.config.target_alpha_tilde_x),
                next_alpha_tilde_y=float(task.config.target_alpha_tilde_y),
                rng=rng,
                gradient_cache=gradient_cache_state,
            )
            if not state.finite:
                raise FloatingPointError("nonfinite branch state")
            if state.step_index % task.config.receiver_cadence_steps == 0:
                sample_states.append(state)
        # Evaluate the immutable snapshots only after the future is complete.
        # This prevents linked LAPACK calls from altering subsequent dynamics.
        checkpoint_observation = observe_state(
            observation_id=f"{prefix}.sample-0000",
            local_step=0,
            local_time=0.0,
            state=checkpoint_state,
            member=task.member,
            config=task.config,
        )
        for sample_state in sample_states:
            observation = observe_state(
                observation_id=f"{prefix}.sample-{sample_state.step_index:04d}",
                local_step=sample_state.step_index,
                local_time=sample_state.step_index * float(task.numerical_view.timestep),
                state=sample_state,
                member=task.member,
                config=task.config,
            )
            observations.append(observation)
            if not observation.spectral_valid or not observation.native_receiver.valid:
                raise FloatingPointError("invalid branch observation")
    except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
        reasons.add("invalid-numerical-future")
    labels = rolling_labels(tuple(observations), config=task.config)
    tau_g = _first_time(labels, task.config.geometric_target_label)
    tau_00 = _first_time(labels, task.config.return_target_label)
    if reasons:
        terminal = MatrixResponseShootingCommittorBranchTerminal.INVALID_NUMERICAL_FUTURE
    elif tau_g is not None and (tau_00 is None or tau_g < tau_00):
        terminal = MatrixResponseShootingCommittorBranchTerminal.G_FIRST_WITHIN_HORIZON
    elif tau_00 is not None and (tau_g is None or tau_00 < tau_g):
        terminal = MatrixResponseShootingCommittorBranchTerminal.ROBUST_00_FIRST_WITHIN_HORIZON
    else:
        terminal = MatrixResponseShootingCommittorBranchTerminal.RIGHT_CENSORED_NO_TARGET_WITHIN_HORIZON
    tau_after = (
        next(
            (
                value.endpoint_time
                for value in labels
                if tau_g is not None
                and value.endpoint_time > tau_g
                and value.label == task.config.return_target_label
            ),
            None,
        )
        if tau_g is not None
        else None
    )
    sequence: list[tuple[Decimal, bool]] = []
    if checkpoint_observation is not None:
        sequence.append((Decimal(0), checkpoint_observation.factor_y.instantaneous_geometric))
    sequence.extend(
        (value.local_time, value.factor_y.instantaneous_geometric) for value in observations
    )
    transitions = [
        (sequence[index][0], sequence[index - 1][1], sequence[index][1])
        for index in range(1, len(sequence))
        if sequence[index - 1][1] != sequence[index][1]
    ]
    first_loss = next((time for time, before, after in transitions if before and not after), None)
    first_regain = next(
        (
            time
            for time, before, after in transitions
            if not before and after and (first_loss is None or time > first_loss)
        ),
        None,
    )
    future_y = tuple(value.factor_y.instantaneous_geometric for value in observations)
    y_occupancy = sum(future_y) / len(future_y) if future_y else None
    rolling_occupancy = (
        sum(value.label == task.config.geometric_target_label for value in labels) / len(labels)
        if labels
        else None
    )
    hashes = _branch_state_hashes(state, rng)
    alpha_x = task.config.target_alpha_tilde_x
    alpha_y = task.config.target_alpha_tilde_y
    return MatrixResponseShootingCommittorBranchResult(
        branch_id=prefix,
        checkpoint_id=task.checkpoint.checkpoint_id,
        checkpoint_step=task.checkpoint.parent_step,
        branch_index=task.branch_index,
        seed_document_sha256=seed_hash,
        bit_generator_id="numpy.pcg64dxsm",
        action_disposition=task.config.branch_action_disposition,
        requested_alpha_x=alpha_x,
        accepted_alpha_x=alpha_x,
        applied_alpha_x=alpha_x,
        realized_alpha_x=alpha_x,
        requested_alpha_y=alpha_y,
        accepted_alpha_y=alpha_y,
        applied_alpha_y=alpha_y,
        realized_alpha_y=alpha_y,
        observations=tuple(observations),
        rolling_labels=labels,
        terminal=terminal,
        tau_g=tau_g,
        tau_00=tau_00,
        tau_00_after_g=tau_after,
        first_y_loss=first_loss,
        first_y_regain=first_regain,
        y_transition_count=len(transitions),
        y_occupancy=_optional_decimal(y_occupancy),
        rolling_01_occupancy=_optional_decimal(rolling_occupancy),
        lag1_autocorrelation=_optional_decimal(_lag_autocorrelation(future_y, 1)),
        lag2_autocorrelation=_optional_decimal(_lag_autocorrelation(future_y, 2)),
        terminal_positions_sha256=hashes[0],
        terminal_momenta_sha256=hashes[1],
        terminal_rng_state_sha256=hashes[2],
        reason_codes=tuple(sorted(reasons)),
    )


def execute_shooting_cohort(
    *,
    checkpoint: SixMatrixResponseShootingCommittorCheckpoint,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    config: Any,
    scientific_seed_inputs: tuple[SixMatrixResponseScientificSeedInput, ...],
    workers: int,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[MatrixResponseShootingCommittorBranchResult, ...]:
    _require_complete_shooting_seed_inputs(
        scientific_seed_inputs,
        scientific_role="shooting-branch",
        current_root_id=f"checkpoint-sha256.{checkpoint.combined_state_sha256}",
        current_context_sha256=checkpoint.combined_state_sha256,
        stream_indices=range(config.branch_count_per_checkpoint),
    )
    if (
        config.branch_seed_derivation_rule_id != BRANCH_DERIVATION_RULE_ID
        or config.branch_action_disposition != "hold"
        or config.branch_count_per_checkpoint != 64
        or member.member_id != config.member_id
        or member.fingerprint() != config.member_fingerprint
        or numerical_view.view_id != config.primary_view_id
        or numerical_view.fingerprint() != config.primary_view_fingerprint
        or checkpoint.parent_step not in config.checkpoint_steps
        or checkpoint.parent_report_sha256 != config.parent_anisotropic_feasibility_report_sha256
        or checkpoint.parent_rollout_id != config.parent_rollout_id
    ):
        raise ValueError("Matrix shooting branch execution semantic binding differs")
    tasks = tuple(
        _BranchTask(checkpoint, member, numerical_view, config, branch_index, scientific_seed_inputs[branch_index])
        for branch_index in range(config.branch_count_per_checkpoint)
    )
    if not 1 <= workers <= 8:
        raise ValueError("Matrix shooting workers must lie in 1..8")
    if workers == 1:
        values = []
        for index, task in enumerate(tasks, 1):
            values.append(_run_branch_task(task))
            if progress is not None:
                progress(index, len(tasks))
        return tuple(values)
    outputs: list[MatrixResponseShootingCommittorBranchResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, value in enumerate(executor.map(_run_branch_task, tasks), 1):
            outputs.append(value)
            if progress is not None:
                progress(index, len(tasks))
    return tuple(sorted(outputs, key=lambda value: value.branch_index))


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorSurvivalPoint(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-survival-point'

    time: Decimal
    at_risk: int
    events: int
    censored: int
    survival: Decimal

    def __post_init__(self) -> None:
        validate_decimal(self.time, field_name="time", minimum=Decimal(0))
        validate_decimal(self.survival, field_name="survival", minimum=Decimal(0))
        if self.survival > Decimal(1) or min(self.at_risk, self.events, self.censored) < 0:
            raise ValueError("Matrix shooting survival point differs")


def _kaplan_meier(durations: tuple[tuple[float, bool], ...]) -> tuple[MatrixResponseShootingCommittorSurvivalPoint, ...]:
    if not durations:
        return ()
    survival = 1.0
    at_risk = len(durations)
    outputs: list[MatrixResponseShootingCommittorSurvivalPoint] = []
    for time in sorted({value[0] for value in durations}):
        events = sum(duration == time and event for duration, event in durations)
        censored = sum(duration == time and not event for duration, event in durations)
        if events:
            survival *= 1.0 - events / at_risk
        outputs.append(
            MatrixResponseShootingCommittorSurvivalPoint(
                time=_decimal(time),
                at_risk=at_risk,
                events=events,
                censored=censored,
                survival=_decimal(survival),
            )
        )
        at_risk -= events + censored
    return tuple(outputs)


def _survival_at(curve: tuple[MatrixResponseShootingCommittorSurvivalPoint, ...], time: float) -> float:
    value = 1.0
    for point in curve:
        if float(point.time) > time:
            break
        value = float(point.survival)
    return value


def _rmst(curve: tuple[MatrixResponseShootingCommittorSurvivalPoint, ...], horizon: float) -> float:
    area = 0.0
    previous_time = 0.0
    previous_survival = 1.0
    for point in curve:
        time = min(float(point.time), horizon)
        area += max(0.0, time - previous_time) * previous_survival
        if float(point.time) >= horizon:
            return area
        previous_time = float(point.time)
        previous_survival = float(point.survival)
    return area + max(0.0, horizon - previous_time) * previous_survival


def _quantile_time(
    curve: tuple[MatrixResponseShootingCommittorSurvivalPoint, ...], survival_threshold: float
) -> float | None:
    return next(
        (float(point.time) for point in curve if float(point.survival) <= survival_threshold),
        None,
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorCohortResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-cohort-result'

    cohort_id: str
    config_fingerprint: str
    checkpoint: SixMatrixResponseShootingCommittorCheckpoint
    branches: tuple[MatrixResponseShootingCommittorBranchResult, ...]
    g_first_count: int
    robust_00_first_count: int
    right_censored_count: int
    invalid_count: int
    unconditional_committor: MatrixResponseShootingCommittorWilsonInterval
    simultaneous_committor: MatrixResponseShootingCommittorWilsonInterval
    resolved_committor: MatrixResponseShootingCommittorWilsonInterval
    resolution_fraction: MatrixResponseShootingCommittorWilsonInterval
    censoring_fraction: MatrixResponseShootingCommittorWilsonInterval
    g_first_survival: tuple[MatrixResponseShootingCommittorSurvivalPoint, ...]
    landmark_branch_count: int
    landmark_rmst: Decimal | None
    landmark_survival: Decimal | None
    residence_median: Decimal | None
    residence_q1: Decimal | None
    residence_q3: Decimal | None
    y_transition_histogram: tuple[int, ...]
    valid_branch_count: int
    complete: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cohort_id, field_name="cohort_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        require_sorted_unique_ids(self.branches, attribute="branch_id", field_name="branches")
        total = (
            self.g_first_count
            + self.robust_00_first_count
            + self.right_censored_count
            + self.invalid_count
        )
        if total != len(self.branches) or len(self.branches) != 64:
            raise ValueError("Matrix shooting cohort terminal denominator differs")
        if self.valid_branch_count != len(self.branches) - self.invalid_count:
            raise ValueError("Matrix shooting valid branch count differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.complete != (self.invalid_count == 0 and not self.reason_codes):
            raise ValueError("Matrix shooting cohort completeness differs")


def reduce_shooting_cohort(
    *,
    checkpoint: SixMatrixResponseShootingCommittorCheckpoint,
    branches: tuple[MatrixResponseShootingCommittorBranchResult, ...],
    config: Any,
) -> MatrixResponseShootingCommittorCohortResult:
    if tuple(value.branch_index for value in branches) != tuple(range(64)):
        raise ValueError("Matrix shooting cohort branch roster differs")
    if any(
        value.checkpoint_id != checkpoint.checkpoint_id
        or value.checkpoint_step != checkpoint.parent_step
        for value in branches
    ):
        raise ValueError("Matrix shooting cohort contains a foreign checkpoint parent")
    counts = {
        terminal: sum(value.terminal is terminal for value in branches)
        for terminal in MatrixResponseShootingCommittorBranchTerminal
    }
    g = counts[MatrixResponseShootingCommittorBranchTerminal.G_FIRST_WITHIN_HORIZON]
    robust = counts[MatrixResponseShootingCommittorBranchTerminal.ROBUST_00_FIRST_WITHIN_HORIZON]
    censored = counts[MatrixResponseShootingCommittorBranchTerminal.RIGHT_CENSORED_NO_TARGET_WITHIN_HORIZON]
    invalid = counts[MatrixResponseShootingCommittorBranchTerminal.INVALID_NUMERICAL_FUTURE]
    confidence = float(config.confidence_level)
    simultaneous = 1.0 - (1.0 - confidence) / config.simultaneous_family_size
    g_durations: list[tuple[float, bool]] = []
    fixed: list[tuple[float, bool]] = []
    horizon = config.branch_steps * float(config.primary_timestep)
    landmark = float(config.residence_landmark)
    for branch in branches:
        if branch.terminal is not MatrixResponseShootingCommittorBranchTerminal.G_FIRST_WITHIN_HORIZON:
            continue
        assert branch.tau_g is not None
        tau_g = float(branch.tau_g)
        if branch.tau_00_after_g is None:
            duration, event = horizon - tau_g, False
        else:
            duration, event = float(branch.tau_00_after_g) - tau_g, True
        g_durations.append((duration, event))
        if tau_g <= horizon - landmark + 1e-15:
            fixed.append((min(duration, landmark), event and duration <= landmark))
    survival = _kaplan_meier(tuple(g_durations))
    fixed_curve = _kaplan_meier(tuple(fixed))
    histogram_size = max((value.y_transition_count for value in branches), default=0) + 1
    histogram = tuple(
        sum(value.y_transition_count == index for value in branches)
        for index in range(histogram_size)
    )
    reasons = ("committor-numerically-incomplete",) if invalid else ()
    return MatrixResponseShootingCommittorCohortResult(
        cohort_id=f"matrix-shooting.cohort.step-{checkpoint.parent_step:04d}",
        config_fingerprint=config.fingerprint(),
        checkpoint=checkpoint,
        branches=branches,
        g_first_count=g,
        robust_00_first_count=robust,
        right_censored_count=censored,
        invalid_count=invalid,
        unconditional_committor=wilson_interval(g, 64, confidence_level=confidence),
        simultaneous_committor=wilson_interval(g, 64, confidence_level=simultaneous),
        resolved_committor=wilson_interval(g, g + robust, confidence_level=confidence),
        resolution_fraction=wilson_interval(g + robust, 64, confidence_level=confidence),
        censoring_fraction=wilson_interval(censored, 64, confidence_level=confidence),
        g_first_survival=survival,
        landmark_branch_count=len(fixed),
        landmark_rmst=(_decimal(_rmst(fixed_curve, landmark)) if fixed else None),
        landmark_survival=(_decimal(_survival_at(fixed_curve, landmark)) if fixed else None),
        residence_median=_optional_decimal(_quantile_time(survival, 0.5)),
        residence_q1=_optional_decimal(_quantile_time(survival, 0.75)),
        residence_q3=_optional_decimal(_quantile_time(survival, 0.25)),
        y_transition_histogram=histogram,
        valid_branch_count=64 - invalid,
        complete=invalid == 0,
        reason_codes=reasons,
    )


def primary_observations_from_replay(
    replay: SixMatrixResponseShootingCommittorReplayData, *, config: Any
) -> tuple[MatrixResponseShootingCommittorObservation, ...]:
    "Materialize the study diagnostics from the replay's complete prefix."

    terminal_checkpoint = replay.checkpoints[-1]
    spectra_by_step: dict[int, list[SixMatrixResponseLaplacianSpectrum]] = {}
    for spectrum in terminal_checkpoint.spectral_receiver_prefix:
        try:
            step = int(spectrum.spectrum_id.rsplit(".", 2)[-2])
        except (IndexError, ValueError) as error:
            raise ValueError("Matrix shooting replay spectrum identity lacks a sample step") from error
        spectra_by_step.setdefault(step, []).append(spectrum)
    outputs = []
    for receiver in terminal_checkpoint.fast_receiver_prefix:
        step = receiver.sample_step
        outputs.append(
            observation_from_receivers(
                observation_id=f"matrix-shooting.primary.sample-{step:04d}",
                local_step=step,
                local_time=step * float(config.primary_timestep),
                receiver=receiver,
                spectra=tuple(spectra_by_step[step]),
                config=config,
            )
        )
    return tuple(outputs)


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorReplayResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-replay-result'

    result_id: str
    config_fingerprint: str
    implementation_commit: str
    parent_report_sha256: str
    parent_implementation_commit: str
    parent_rollout_id: str
    derived_rng_seed_sha256: str
    expected_final_state_sha256: str
    observed_final_state_sha256: str
    parent_replay_exact: bool
    checkpoint_restart_exact: bool
    checkpoints: tuple[SixMatrixResponseShootingCommittorCheckpoint, ...]
    primary_observations: tuple[MatrixResponseShootingCommittorObservation, ...]
    primary_rolling_labels: tuple[MatrixResponseShootingCommittorRollingLabel, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("result_id", "parent_rollout_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "config_fingerprint",
            "parent_report_sha256",
            "derived_rng_seed_sha256",
            "expected_final_state_sha256",
            "observed_final_state_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        for name in ("implementation_commit", "parent_implementation_commit"):
            value = getattr(self, name)
            if len(value) != 40:
                raise ValueError(f"Matrix shooting {name} must be a Git SHA-1")
            int(value, 16)
        if tuple(value.parent_step for value in self.checkpoints) != (
            816,
            848,
            880,
            896,
            928,
            960,
            1024,
        ):
            raise ValueError("Matrix shooting replay checkpoint roster differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_exact = not self.reason_codes
        if self.parent_replay_exact != expected_exact:
            raise ValueError("Matrix shooting parent replay disposition differs")


def build_replay_result(
    *,
    replay: SixMatrixResponseShootingCommittorReplayData,
    config: Any,
    implementation_commit: str,
    parent_report: MatrixResponseAnisotropicFeasibilityQualificationReport,
    checkpoint_restart_exact: bool,
) -> MatrixResponseShootingCommittorReplayResult:
    observations = primary_observations_from_replay(replay, config=config)
    reasons = set()
    if replay.final_rollout_state_sha256 != config.parent_final_state_sha256:
        reasons.add("parent-final-state-hash-mismatch")
    if replay.rng_stream.derived_seed_sha256 != next(
        value.rng_seed_sha256
        for value in parent_report.rollouts
        if value.rollout_id == config.parent_rollout_id
    ):
        reasons.add("parent-rng-seed-mismatch")
    if not checkpoint_restart_exact:
        reasons.add("checkpoint-restart-mismatch")
    return MatrixResponseShootingCommittorReplayResult(
        result_id="matrix-shooting.reconstructed-checkpoints",
        config_fingerprint=config.fingerprint(),
        implementation_commit=implementation_commit,
        parent_report_sha256=config.parent_anisotropic_feasibility_report_sha256,
        parent_implementation_commit=parent_report.implementation_commit,
        parent_rollout_id=config.parent_rollout_id,
        derived_rng_seed_sha256=replay.rng_stream.derived_seed_sha256,
        expected_final_state_sha256=config.parent_final_state_sha256,
        observed_final_state_sha256=replay.final_rollout_state_sha256,
        parent_replay_exact=not reasons,
        checkpoint_restart_exact=checkpoint_restart_exact,
        checkpoints=replay.checkpoints,
        primary_observations=observations,
        primary_rolling_labels=rolling_labels(observations, config=config),
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorBridgeAlignment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-bridge-alignment'

    parent_step: int
    primary_phi_y: Decimal
    half_phi_y: Decimal
    absolute_phi_y_difference: Decimal
    primary_closure_y: Decimal
    half_closure_y: Decimal
    primary_kernel_y: Decimal
    half_kernel_y: Decimal
    primary_rolling_label: str
    half_rolling_label: str

    def __post_init__(self) -> None:
        if self.parent_step not in {816, 848, 880, 896, 928, 960, 1024}:
            raise ValueError("Matrix shooting bridge alignment step differs")
        for name in (
            "primary_phi_y",
            "half_phi_y",
            "absolute_phi_y_difference",
            "primary_closure_y",
            "half_closure_y",
            "primary_kernel_y",
            "half_kernel_y",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for value in (self.primary_rolling_label, self.half_rolling_label):
            if value not in {"00", "10", "01", "11"}:
                raise ValueError("Matrix shooting bridge rolling label differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorBridgeResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-bridge-result'

    result_id: str
    config_fingerprint: str
    parent_rollout_id: str
    bridge_seed_aggregate_sha256: str
    half_step_count: int
    half_observations: tuple[MatrixResponseShootingCommittorObservation, ...]
    half_rolling_labels: tuple[MatrixResponseShootingCommittorRollingLabel, ...]
    alignments: tuple[MatrixResponseShootingCommittorBridgeAlignment, ...]
    maximum_checkpoint_phi_y_difference: Decimal
    maximum_half_hermiticity_residual: Decimal
    half_final_positions_sha256: str
    half_final_momenta_sha256: str
    half_final_rolling_label: str
    x_false_admission_count: int
    disposition: MatrixResponseShootingCommittorNumericalDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("result_id", "parent_rollout_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "config_fingerprint",
            "bridge_seed_aggregate_sha256",
            "half_final_positions_sha256",
            "half_final_momenta_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.half_step_count != 2048 or len(self.half_observations) != 64:
            raise ValueError("Matrix shooting half-step replay cardinality differs")
        validate_decimal(
            self.maximum_checkpoint_phi_y_difference,
            field_name="maximum_checkpoint_phi_y_difference",
            minimum=Decimal(0),
        )
        validate_decimal(
            self.maximum_half_hermiticity_residual,
            field_name="maximum_half_hermiticity_residual",
            minimum=Decimal(0),
        )
        if self.half_final_rolling_label not in {"00", "10", "01", "11"}:
            raise ValueError("Matrix shooting half final rolling label differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        concordant = (
            self.disposition is MatrixResponseShootingCommittorNumericalDisposition.EVENT_PATH_NUMERICALLY_CONCORDANT
        )
        if concordant != (not self.reason_codes):
            raise ValueError("Matrix shooting bridge disposition differs from reasons")


def execute_brownian_bridge_replay(
    *,
    replay: SixMatrixResponseShootingCommittorReplayData,
    replay_result: MatrixResponseShootingCommittorReplayResult,
    member: SixMatrixResponseModelFamilyMember,
    primary_view: SixMatrixResponseNumericalView,
    half_view: SixMatrixResponseNumericalView,
    config: Any,
    scientific_seed_inputs: tuple[SixMatrixResponseScientificSeedInput, ...],
) -> MatrixResponseShootingCommittorBridgeResult:
    current_context_sha256 = config.fingerprint()
    _require_complete_shooting_seed_inputs(
        scientific_seed_inputs,
        scientific_role="shooting-bridge",
        current_root_id=config.parent_rollout_id,
        current_context_sha256=current_context_sha256,
        stream_indices=range(1, config.parent_total_steps + 1),
    )
    gradient_cache_state = BAOABGradientCache()
    if not replay_result.parent_replay_exact:
        raise ValueError("Matrix shooting bridge cannot run after parent replay mismatch")
    if (
        config.bridge_seed_derivation_rule_id != BRIDGE_DERIVATION_RULE_ID
        or config.half_step_action_interpolation_id != "matrix-shooting.parent-binary-float-physical-time"
        or member.member_id != config.member_id
        or member.fingerprint() != config.member_fingerprint
        or primary_view.view_id != config.primary_view_id
        or primary_view.fingerprint() != config.primary_view_fingerprint
        or half_view.view_id != config.secondary_view_id
        or half_view.fingerprint() != config.secondary_view_fingerprint
        or len(replay.coarse_noises) != config.parent_total_steps
    ):
        raise ValueError("Matrix shooting Brownian-bridge semantic binding differs")
    state = ideal_state(
        q=config.q,
        alpha_tilde_x=0.0,
        alpha_tilde_y=0.0,
        constitution="00",
    )
    observations: list[MatrixResponseShootingCommittorObservation] = []
    sample_states: list[SixMatrixState] = []
    seed_hashes: list[str] = []
    maximum_hermiticity = 0.0
    half_decay = exp(-float(primary_view.friction_gamma) * float(primary_view.timestep) / 2.0)
    for coarse_step, coarse_noise in enumerate(replay.coarse_noises, 1):
        bridge_rng, seed_hash = derive_bridge_rng(
            parent_rollout_id=config.parent_rollout_id,
            coarse_step_index=coarse_step,
            current_context_sha256=current_context_sha256,
            scientific_seed_input=scientific_seed_inputs[coarse_step - 1],
        )
        seed_hashes.append(seed_hash)
        zeta = hermitian_noise(rng=bridge_rng, q=config.q)
        first, second = brownian_bridge_split(
            coarse_noise=coarse_noise,
            bridge_noise=zeta,
            half_decay=half_decay,
        )
        for half_offset, noise in enumerate((first, second), 1):
            half_step = 2 * (coarse_step - 1) + half_offset
            # Mirror the parent's binary-float action arithmetic at every
            # coarse endpoint while interpolating the same action in physical
            # time at the intervening half step.
            half_ramp_steps = 2 * config.parent_ramp_steps
            fraction = min(half_step, half_ramp_steps) / half_ramp_steps
            state = baoab_step_with_hermitian_noise(
                state,
                member=member,
                numerical_view=half_view,
                next_alpha_tilde_x=float(config.target_alpha_tilde_x) * fraction,
                next_alpha_tilde_y=float(config.target_alpha_tilde_y) * fraction,
                standardized_noise=noise,
                gradient_cache=gradient_cache_state,
            )
            maximum_hermiticity = max(
                maximum_hermiticity,
                hermiticity_residual(state.positions),
                hermiticity_residual(state.momenta),
            )
        if coarse_step % config.receiver_cadence_steps == 0:
            sample_states.append(state)
    for sample_state in sample_states:
        coarse_step = sample_state.step_index // 2
        observations.append(
            observe_state(
                observation_id=f"matrix-shooting.half.sample-{coarse_step:04d}",
                local_step=coarse_step,
                local_time=coarse_step * float(config.primary_timestep),
                state=sample_state,
                member=member,
                config=config,
            )
        )
    half_labels = rolling_labels(tuple(observations), config=config)
    primary_by_step = {value.local_step: value for value in replay_result.primary_observations}
    half_by_step = {value.local_step: value for value in observations}
    primary_labels = {
        value.endpoint_step: value.label for value in replay_result.primary_rolling_labels
    }
    half_label_map = {value.endpoint_step: value.label for value in half_labels}
    alignments = tuple(
        MatrixResponseShootingCommittorBridgeAlignment(
            parent_step=step,
            primary_phi_y=primary_by_step[step].factor_y.phi,
            half_phi_y=half_by_step[step].factor_y.phi,
            absolute_phi_y_difference=_decimal(
                abs(
                    float(primary_by_step[step].factor_y.phi)
                    - float(half_by_step[step].factor_y.phi)
                )
            ),
            primary_closure_y=primary_by_step[step].factor_y.closure_ratio,
            half_closure_y=half_by_step[step].factor_y.closure_ratio,
            primary_kernel_y=primary_by_step[step].factor_y.kernel_band_ratio,
            half_kernel_y=half_by_step[step].factor_y.kernel_band_ratio,
            primary_rolling_label=primary_labels[step],
            half_rolling_label=half_label_map[step],
        )
        for step in config.checkpoint_steps
    )
    maximum_phi = max(float(value.absolute_phi_y_difference) for value in alignments)
    x_false = sum(value.half_rolling_label.startswith("1") for value in alignments)
    final_label = half_labels[-1].label
    reasons = set()
    if maximum_hermiticity > float(config.hermiticity_residual_max):
        reasons.add("half-step-hermiticity-exceeded")
    if maximum_phi > float(config.bridge_phi_y_difference_max):
        reasons.add("half-step-y-radius-not-concordant")
    if final_label != config.geometric_target_label:
        reasons.add("half-step-final-rolling-label-not-01")
    if x_false:
        reasons.add("half-step-false-x-admission")
    if not state.finite or not all(value.spectral_valid for value in observations):
        reasons.add("half-step-path-invalid")
    seed_aggregate = sha256("".join(seed_hashes).encode("ascii")).hexdigest()
    return MatrixResponseShootingCommittorBridgeResult(
        result_id="matrix-shooting.brownian-bridge-replay",
        config_fingerprint=config.fingerprint(),
        parent_rollout_id=config.parent_rollout_id,
        bridge_seed_aggregate_sha256=seed_aggregate,
        half_step_count=state.step_index,
        half_observations=tuple(observations),
        half_rolling_labels=half_labels,
        alignments=alignments,
        maximum_checkpoint_phi_y_difference=_decimal(maximum_phi),
        maximum_half_hermiticity_residual=_decimal(maximum_hermiticity),
        half_final_positions_sha256=sha256(state.positions.tobytes(order="C")).hexdigest(),
        half_final_momenta_sha256=sha256(state.momenta.tobytes(order="C")).hexdigest(),
        half_final_rolling_label=final_label,
        x_false_admission_count=x_false,
        disposition=(
            MatrixResponseShootingCommittorNumericalDisposition.EVENT_PATH_NUMERICALLY_CONCORDANT
            if not reasons
            else MatrixResponseShootingCommittorNumericalDisposition.EVENT_NUMERICAL_VIEW_SENSITIVE
        ),
        reason_codes=tuple(sorted(reasons)),
    )


class MatrixResponseShootingCommittorStabilityDisposition(StrEnum):
    ONE_QUOTIENT_UNSTABLE_DIRECTION = "ONE_QUOTIENT_UNSTABLE_DIRECTION"
    MULTIPLE_QUOTIENT_UNSTABLE_DIRECTIONS = "MULTIPLE_QUOTIENT_UNSTABLE_DIRECTIONS"
    NO_QUOTIENT_UNSTABLE_DIRECTION = "NO_QUOTIENT_UNSTABLE_DIRECTION"
    LINEARIZATION_NUMERICALLY_UNSTABLE = "LINEARIZATION_NUMERICALLY_UNSTABLE"
    LOCAL_RECEIVER_PROJECTION_UNEVALUABLE = "LOCAL_RECEIVER_PROJECTION_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorComplexEigenvalue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-complex-eigenvalue'

    real: Decimal
    imaginary: Decimal

    def __post_init__(self) -> None:
        validate_decimal(self.real, field_name="real")
        validate_decimal(self.imaginary, field_name="imaginary")


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorLinearizationScale(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-linearization-scale'

    step_size: Decimal
    raw_hessian_sha256: str
    raw_hessian_antisymmetry: Decimal
    quotient_rank: int
    quotient_dimension: int
    unstable_eigenvalues: tuple[MatrixResponseShootingCommittorComplexEigenvalue, ...]

    def __post_init__(self) -> None:
        validate_decimal(self.step_size, field_name="step_size", minimum=Decimal(0))
        validate_sha256(self.raw_hessian_sha256, field_name="raw_hessian_sha256")
        validate_decimal(
            self.raw_hessian_antisymmetry,
            field_name="raw_hessian_antisymmetry",
            minimum=Decimal(0),
        )
        if self.quotient_rank < 0 or self.quotient_dimension < 1:
            raise ValueError("Matrix shooting quotient dimensions differ")


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorReceiverDerivative(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-receiver-derivative'

    receiver_id: str
    active_relative_gap: Decimal | None
    differentiable: bool
    h_gradient_norm: Decimal | None
    half_h_gradient_norm: Decimal | None
    relative_change: Decimal | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receiver_id, field_name="receiver_id")
        for name in (
            "active_relative_gap",
            "h_gradient_norm",
            "half_h_gradient_norm",
            "relative_change",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name, minimum=Decimal(0))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.differentiable == bool(self.reason_codes):
            raise ValueError("Matrix shooting receiver derivative disposition differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorUnstableProjection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-unstable-projection'

    mode_index: int
    eigenvalue: MatrixResponseShootingCommittorComplexEigenvalue
    y_structural_squared_projection: Decimal
    x_radius_squared_projection: Decimal
    cross_commutator_squared_projection: Decimal | None
    observed_union_squared_projection: Decimal
    unobserved_squared_projection: Decimal

    def __post_init__(self) -> None:
        if self.mode_index < 0:
            raise ValueError("Matrix shooting unstable mode index cannot be negative")
        for name in (
            "y_structural_squared_projection",
            "x_radius_squared_projection",
            "cross_commutator_squared_projection",
            "observed_union_squared_projection",
            "unobserved_squared_projection",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name, minimum=Decimal(0))
                if value > Decimal("1.0000000001"):
                    raise ValueError("Matrix shooting squared projection exceeds one")


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorLinearizationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-linearization-result'

    result_id: str
    config_fingerprint: str
    checkpoint_id: str
    checkpoint_step: int
    position_dimension: int
    phase_space_dimension: int
    conjugation_generator_count: int
    quotient_tangent_rank: int
    primary_scale: MatrixResponseShootingCommittorLinearizationScale
    half_scale: MatrixResponseShootingCommittorLinearizationScale
    maximum_unstable_eigenvalue_relative_change: Decimal | None
    receiver_derivatives: tuple[MatrixResponseShootingCommittorReceiverDerivative, ...]
    unstable_projections: tuple[MatrixResponseShootingCommittorUnstableProjection, ...]
    y_structural_overlap_pass: bool
    disposition: MatrixResponseShootingCommittorStabilityDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("result_id", "checkpoint_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        if (
            self.checkpoint_step,
            self.position_dimension,
            self.phase_space_dimension,
            self.conjugation_generator_count,
        ) != (896, 96, 192, 15):
            raise ValueError("Matrix shooting linearization dimension/point differs")
        if self.quotient_tangent_rank < 0:
            raise ValueError("Matrix shooting quotient tangent rank cannot be negative")
        if self.maximum_unstable_eigenvalue_relative_change is not None:
            validate_decimal(
                self.maximum_unstable_eigenvalue_relative_change,
                field_name="maximum_unstable_eigenvalue_relative_change",
                minimum=Decimal(0),
            )
        require_sorted_unique_ids(
            self.receiver_derivatives,
            attribute="receiver_id",
            field_name="receiver_derivatives",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def _gradient_vector(
    coordinates: np.ndarray,
    *,
    n: int,
    q: int,
    member: SixMatrixResponseModelFamilyMember,
    alpha_x: float,
    alpha_y: float,
) -> np.ndarray:
    positions = decode_hermitian_matrices(coordinates.reshape((2, 3, n * n)), n=n)
    parameters = SixMatrixParameters(
        q=q,
        mass_x=float(member.mass_x),
        mass_y=float(member.mass_y),
        gamma=float(member.cross_coupling_gamma),
        alpha_tilde_x=alpha_x,
        alpha_tilde_y=alpha_y,
    )
    gradient = analytic_gradient_terms(positions, parameters).total
    return encode_hermitian_matrices(gradient).reshape(-1)


def _central_jacobian(
    function: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float
) -> np.ndarray:
    output_size = function(x).size
    result = np.empty((output_size, x.size), dtype=np.float64)
    for index in range(x.size):
        direction = np.zeros_like(x)
        direction[index] = h
        result[:, index] = (function(x + direction) - function(x - direction)) / (2.0 * h)
    return result


def _conjugation_slice(
    state: SixMatrixState, *, relative_cutoff: float
) -> tuple[np.ndarray, int]:
    columns = []
    for generator in traceless_hermitian_basis(state.n):
        dx = 1.0j * (generator @ state.positions - state.positions @ generator)
        dp = 1.0j * (generator @ state.momenta - state.momenta @ generator)
        columns.append(
            np.concatenate(
                (
                    encode_hermitian_matrices(np.asarray(dx, dtype="<c16")).reshape(-1),
                    encode_hermitian_matrices(np.asarray(dp, dtype="<c16")).reshape(-1),
                )
            )
        )
    tangent = np.stack(columns, axis=1)
    u, singular, _ = np.linalg.svd(tangent, full_matrices=True)
    threshold = relative_cutoff * max(float(singular[0]), 1.0)
    rank = int(np.count_nonzero(singular > threshold))
    return u[:, rank:], rank


@dataclass(frozen=True, slots=True)
class _DriftSpectrum:
    scale: MatrixResponseShootingCommittorLinearizationScale
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    quotient_basis: np.ndarray


def _drift_spectrum(
    *,
    hessian: np.ndarray,
    raw_hessian: np.ndarray,
    step_size: float,
    quotient_basis: np.ndarray,
    quotient_rank: int,
    unstable_cutoff: float,
) -> _DriftSpectrum:
    dimension = hessian.shape[0]
    zero = np.zeros_like(hessian)
    identity = np.eye(dimension)
    drift = np.block([[zero, identity], [-hessian, -identity]])
    quotient = quotient_basis.T @ drift @ quotient_basis
    eigenvalues, eigenvectors = np.linalg.eig(quotient)
    unstable = np.where(eigenvalues.real > unstable_cutoff)[0]
    ordered = unstable[np.argsort(eigenvalues[unstable].real)]
    antisymmetry = float(
        float(np.linalg.norm(raw_hessian - raw_hessian.T))
        / max(float(np.linalg.norm(raw_hessian)), 1.0)
    )
    scale = MatrixResponseShootingCommittorLinearizationScale(
        step_size=_decimal(step_size),
        raw_hessian_sha256=sha256(
            np.ascontiguousarray(raw_hessian, dtype="<f8").tobytes(order="C")
        ).hexdigest(),
        raw_hessian_antisymmetry=_decimal(antisymmetry),
        quotient_rank=quotient_rank,
        quotient_dimension=quotient.shape[0],
        unstable_eigenvalues=tuple(
            MatrixResponseShootingCommittorComplexEigenvalue(
                _decimal(eigenvalues[index].real), _decimal(eigenvalues[index].imag)
            )
            for index in ordered
        ),
    )
    return _DriftSpectrum(scale, eigenvalues, eigenvectors, quotient_basis)


def _constituent_metrics(
    positions: ComplexArray, *, q: int, alpha_x: float, alpha_y: float
) -> tuple[dict[str, float], dict[str, float | None]]:
    n = q**2
    radii = [float(np.trace(np.sum(sector @ sector, axis=0)).real / n) for sector in positions]
    closures: list[list[float]] = []
    for sector, alpha_tilde in zip(positions, (alpha_x, alpha_y), strict=True):
        scale = 2.0 * (alpha_tilde / q) / 3.0
        residuals = []
        for a, (b, c) in enumerate(((1, 2), (2, 0), (0, 1))):
            residuals.append(
                float(
                    np.linalg.norm(
                        sector[b] @ sector[c] - sector[c] @ sector[b] - 1.0j * scale * sector[a]
                    )
                )
            )
        closures.append(residuals)
    closure_y_sorted = sorted(closures[1], reverse=True)
    scale_y = 2.0 * (alpha_y / q) / 3.0
    closure_y = closure_y_sorted[0] / max(
        abs(scale_y) * sqrt(n * max(radii[1], 0.0) / 3.0),
        np.finfo(np.float64).tiny,
    )
    spectra = spectral_receiver(
        receiver_prefix="matrix-shooting.linearization.receiver",
        q=q,
        positions=positions,
    )
    y_spectrum = next(value for value in spectra if value.sector == "Y")
    eigenvalues = np.asarray([float(value) for value in y_spectrum.eigenvalues])
    kernel = _kernel_band_ratio(y_spectrum, q)
    kernel_neighbors = (
        eigenvalues[q**2 - 2],
        eigenvalues[q**2 - 1],
        eigenvalues[q**2],
        eigenvalues[q**2 + 1],
    )
    cross_values = sorted(
        (
            float(
                np.linalg.norm(
                    positions[0, a] @ positions[1, b] - positions[1, b] @ positions[0, a]
                )
            )
            for a in range(3)
            for b in range(3)
        ),
        reverse=True,
    )

    def relative_gap(high: float, low: float) -> float:
        return (high - low) / max(abs(high), 1.0)

    kernel_gap = min(
        relative_gap(kernel_neighbors[1], kernel_neighbors[0]),
        relative_gap(kernel_neighbors[2], kernel_neighbors[1]),
        relative_gap(kernel_neighbors[3], kernel_neighbors[2]),
    )
    values = {
        "matrix-shooting.receiver.y-radius": radii[1],
        "matrix-shooting.receiver.y-closure": closure_y,
        "matrix-shooting.receiver.y-kernel": kernel,
        "matrix-shooting.receiver.x-radius": radii[0],
        "matrix-shooting.receiver.cross-commutator": cross_values[0],
    }
    gaps: dict[str, float | None] = {
        "matrix-shooting.receiver.y-radius": None,
        "matrix-shooting.receiver.y-closure": relative_gap(closure_y_sorted[0], closure_y_sorted[1]),
        "matrix-shooting.receiver.y-kernel": kernel_gap,
        "matrix-shooting.receiver.x-radius": None,
        "matrix-shooting.receiver.cross-commutator": relative_gap(cross_values[0], cross_values[1]),
    }
    return values, gaps


def _scalar_gradient(
    function: Callable[[np.ndarray], float], x: np.ndarray, h: float
) -> np.ndarray:
    result = np.empty_like(x)
    for index in range(x.size):
        direction = np.zeros_like(x)
        direction[index] = h
        result[index] = (function(x + direction) - function(x - direction)) / (2.0 * h)
    return result


def _orthonormal_span(columns: list[np.ndarray], dimension: int) -> np.ndarray:
    if not columns:
        return np.empty((dimension, 0), dtype=np.float64)
    matrix = np.stack(columns, axis=1)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=False)
    rank = int(np.count_nonzero(singular > 1e-12 * max(float(singular[0]), 1.0)))
    return u[:, :rank]


def _squared_projection(vector: np.ndarray, basis: np.ndarray) -> float:
    if basis.shape[1] == 0:
        return 0.0
    numerator = float(np.linalg.norm(basis.T @ vector)) ** 2
    denominator = max(float(np.linalg.norm(vector)) ** 2, 1e-30)
    return numerator / denominator


def execute_local_linearization(
    *,
    checkpoint: SixMatrixResponseShootingCommittorCheckpoint,
    member: SixMatrixResponseModelFamilyMember,
    config: Any,
) -> MatrixResponseShootingCommittorLinearizationResult:
    if (
        checkpoint.parent_step != config.linearization_checkpoint_step
        or config.hermitian_coordinate_basis_id != "matrix-shooting.hs-hermitian-coordinate-basis"
        or member.member_id != config.member_id
        or member.fingerprint() != config.member_fingerprint
    ):
        raise ValueError("Matrix shooting primary linearization must use step 896")
    state, _ = state_from_shooting_checkpoint(checkpoint)
    n = state.n
    x = encode_hermitian_matrices(state.positions).reshape(-1)
    h = float(config.hessian_scale) * max(1.0, float(np.linalg.norm(x)) / sqrt(x.size))

    def gradient_function(value: np.ndarray) -> np.ndarray:
        return _gradient_vector(
            value,
            n=n,
            q=state.q,
            member=member,
            alpha_x=state.alpha_tilde_x,
            alpha_y=state.alpha_tilde_y,
        )

    raw_h = _central_jacobian(gradient_function, x, h)
    raw_half = _central_jacobian(gradient_function, x, h / 2.0)
    hessian = (raw_h + raw_h.T) * 0.5
    half_hessian = (raw_half + raw_half.T) * 0.5
    quotient_basis, tangent_rank = _conjugation_slice(
        state, relative_cutoff=float(config.quotient_relative_cutoff)
    )
    primary = _drift_spectrum(
        hessian=hessian,
        raw_hessian=raw_h,
        step_size=h,
        quotient_basis=quotient_basis,
        quotient_rank=tangent_rank,
        unstable_cutoff=float(config.unstable_real_part_min),
    )
    half = _drift_spectrum(
        hessian=half_hessian,
        raw_hessian=raw_half,
        step_size=h / 2.0,
        quotient_basis=quotient_basis,
        quotient_rank=tangent_rank,
        unstable_cutoff=float(config.unstable_real_part_min),
    )
    unstable_primary = np.where(primary.eigenvalues.real > float(config.unstable_real_part_min))[0]
    unstable_half = np.where(half.eigenvalues.real > float(config.unstable_real_part_min))[0]
    maximum_change: float | None = None
    matching_stable = len(unstable_primary) == len(unstable_half)
    if matching_stable and len(unstable_primary):
        cost = np.abs(
            primary.eigenvalues[unstable_primary][:, None]
            - half.eigenvalues[unstable_half][None, :]
        )
        rows, columns = linear_sum_assignment(cost)
        changes = [
            abs(
                primary.eigenvalues[unstable_primary[row]].real
                - half.eigenvalues[unstable_half[column]].real
            )
            / max(abs(primary.eigenvalues[unstable_primary[row]].real), 1e-15)
            for row, column in zip(rows, columns, strict=True)
        ]
        maximum_change = max(changes, default=0.0)
        matching_stable = maximum_change <= float(config.unstable_eigenvalue_change_max)
    base_values, gaps = _constituent_metrics(
        state.positions,
        q=state.q,
        alpha_x=state.alpha_tilde_x,
        alpha_y=state.alpha_tilde_y,
    )
    receiver_records: list[MatrixResponseShootingCommittorReceiverDerivative] = []
    gradients: dict[str, np.ndarray] = {}
    for receiver_id in sorted(base_values):
        gap = gaps[receiver_id]
        if gap is not None and gap <= float(config.receiver_derivative_gap_min):
            receiver_records.append(
                MatrixResponseShootingCommittorReceiverDerivative(
                    receiver_id=receiver_id,
                    active_relative_gap=_decimal(max(gap, 0.0)),
                    differentiable=False,
                    h_gradient_norm=None,
                    half_h_gradient_norm=None,
                    relative_change=None,
                    reason_codes=("active-constituent-gap-insufficient",),
                )
            )
            continue

        def scalar(value: np.ndarray, receiver_name: str = receiver_id) -> float:
            return _constituent_metrics(
                decode_hermitian_matrices(value.reshape((2, 3, n * n)), n=n),
                q=state.q,
                alpha_x=state.alpha_tilde_x,
                alpha_y=state.alpha_tilde_y,
            )[0][receiver_name]

        gradient_h = _scalar_gradient(scalar, x, h)
        gradient_half = _scalar_gradient(scalar, x, h / 2.0)
        relative_change = float(
            float(np.linalg.norm(gradient_h - gradient_half))
            / max(float(np.linalg.norm(gradient_half)), 1e-12)
        )
        differentiable = relative_change <= float(config.receiver_derivative_change_max)
        receiver_records.append(
            MatrixResponseShootingCommittorReceiverDerivative(
                receiver_id=receiver_id,
                active_relative_gap=None if gap is None else _decimal(max(gap, 0.0)),
                differentiable=differentiable,
                h_gradient_norm=_decimal(float(np.linalg.norm(gradient_h))),
                half_h_gradient_norm=_decimal(float(np.linalg.norm(gradient_half))),
                relative_change=_decimal(relative_change),
                reason_codes=(
                    () if differentiable else ("finite-difference-gradient-not-converged",)
                ),
            )
        )
        if differentiable:
            gradients[receiver_id] = gradient_half

    def phase_gradient(value: np.ndarray) -> np.ndarray:
        return np.concatenate((value, np.zeros_like(value)))

    y_ids = (
        "matrix-shooting.receiver.y-radius",
        "matrix-shooting.receiver.y-closure",
        "matrix-shooting.receiver.y-kernel",
    )
    y_basis = _orthonormal_span(
        [phase_gradient(gradients[value]) for value in y_ids if value in gradients],
        192,
    )
    x_basis = _orthonormal_span(
        [phase_gradient(gradients["matrix-shooting.receiver.x-radius"])]
        if "matrix-shooting.receiver.x-radius" in gradients
        else [],
        192,
    )
    cross_basis = _orthonormal_span(
        [phase_gradient(gradients["matrix-shooting.receiver.cross-commutator"])]
        if "matrix-shooting.receiver.cross-commutator" in gradients
        else [],
        192,
    )
    union_basis = _orthonormal_span([phase_gradient(value) for value in gradients.values()], 192)
    ordered_unstable = unstable_primary[np.argsort(primary.eigenvalues[unstable_primary].real)]
    projections: list[MatrixResponseShootingCommittorUnstableProjection] = []
    for mode_index, eigen_index in enumerate(ordered_unstable):
        # Complex-conjugate modes of a real projected drift are legitimate
        # real invariant planes, not a numerical failure. The full complex
        # vector gives phase-invariant Hermitian-norm projections; selecting
        # only its real part does not.
        vector = quotient_basis @ primary.eigenvectors[:, eigen_index]
        vector /= max(float(np.linalg.norm(vector)), 1e-30)
        observed = min(1.0, _squared_projection(vector, union_basis))
        projections.append(
            MatrixResponseShootingCommittorUnstableProjection(
                mode_index=mode_index,
                eigenvalue=MatrixResponseShootingCommittorComplexEigenvalue(
                    _decimal(primary.eigenvalues[eigen_index].real),
                    _decimal(primary.eigenvalues[eigen_index].imag),
                ),
                y_structural_squared_projection=_decimal(
                    min(1.0, _squared_projection(vector, y_basis))
                ),
                x_radius_squared_projection=_decimal(
                    min(1.0, _squared_projection(vector, x_basis))
                ),
                cross_commutator_squared_projection=(
                    None
                    if cross_basis.shape[1] == 0
                    else _decimal(min(1.0, _squared_projection(vector, cross_basis)))
                ),
                observed_union_squared_projection=_decimal(observed),
                unobserved_squared_projection=_decimal(max(0.0, 1.0 - observed)),
            )
        )
    reasons = set()
    if (
        float(primary.scale.raw_hessian_antisymmetry) > float(config.hessian_antisymmetry_max)
        or float(half.scale.raw_hessian_antisymmetry) > float(config.hessian_antisymmetry_max)
        or not matching_stable
    ):
        reasons.add("linearization-numerically-unstable")
    required_derivatives = {
        "matrix-shooting.receiver.y-radius",
        "matrix-shooting.receiver.y-closure",
        "matrix-shooting.receiver.y-kernel",
        "matrix-shooting.receiver.x-radius",
        "matrix-shooting.receiver.cross-commutator",
    }
    derivative_complete = required_derivatives <= set(gradients)
    if not derivative_complete:
        reasons.add("local-receiver-projection-unevaluable")
    if "linearization-numerically-unstable" in reasons:
        disposition = MatrixResponseShootingCommittorStabilityDisposition.LINEARIZATION_NUMERICALLY_UNSTABLE
    elif not derivative_complete:
        disposition = MatrixResponseShootingCommittorStabilityDisposition.LOCAL_RECEIVER_PROJECTION_UNEVALUABLE
    elif len(unstable_primary) == 0:
        disposition = MatrixResponseShootingCommittorStabilityDisposition.NO_QUOTIENT_UNSTABLE_DIRECTION
    elif len(unstable_primary) == 1:
        disposition = MatrixResponseShootingCommittorStabilityDisposition.ONE_QUOTIENT_UNSTABLE_DIRECTION
    else:
        disposition = MatrixResponseShootingCommittorStabilityDisposition.MULTIPLE_QUOTIENT_UNSTABLE_DIRECTIONS
    y_pass = bool(
        len(projections) == 1
        and float(projections[0].y_structural_squared_projection) >= float(config.y_projection_min)
    )
    return MatrixResponseShootingCommittorLinearizationResult(
        result_id="matrix-shooting.local-linearization.step-0896",
        config_fingerprint=config.fingerprint(),
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_step=checkpoint.parent_step,
        position_dimension=96,
        phase_space_dimension=192,
        conjugation_generator_count=15,
        quotient_tangent_rank=tangent_rank,
        primary_scale=primary.scale,
        half_scale=half.scale,
        maximum_unstable_eigenvalue_relative_change=_optional_decimal(maximum_change),
        receiver_derivatives=tuple(receiver_records),
        unstable_projections=tuple(projections),
        y_structural_overlap_pass=y_pass,
        disposition=disposition,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorNullCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-null-cell'

    cell_id: str
    amplitude_band: Decimal
    stratum_id: str
    denominator: int
    closure_count: int
    kernel_count: int
    persistence_count: int
    closure_kernel_count: int
    closure_persistence_count: int
    kernel_persistence_count: int
    joint_count: int
    frozen_geometric_count: int
    closure_interval: MatrixResponseShootingCommittorWilsonInterval
    kernel_interval: MatrixResponseShootingCommittorWilsonInterval
    persistence_interval: MatrixResponseShootingCommittorWilsonInterval
    joint_interval: MatrixResponseShootingCommittorWilsonInterval

    def __post_init__(self) -> None:
        for name in ("cell_id", "stratum_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_decimal(self.amplitude_band, field_name="amplitude_band", minimum=Decimal(0))
        counts = (
            self.closure_count,
            self.kernel_count,
            self.persistence_count,
            self.closure_kernel_count,
            self.closure_persistence_count,
            self.kernel_persistence_count,
            self.joint_count,
            self.frozen_geometric_count,
        )
        if self.denominator < 0 or any(not 0 <= value <= self.denominator for value in counts):
            raise ValueError("Matrix shooting null cell counts differ")


class MatrixResponseShootingCommittorNullDisposition(StrEnum):
    ALGEBRAIC_ORGANIZATION_EXCEPTIONAL_AT_MATCHED_AMPLITUDE = (
        "ALGEBRAIC_ORGANIZATION_EXCEPTIONAL_AT_MATCHED_AMPLITUDE"
    )
    ALGEBRAIC_ORGANIZATION_NOT_EXCEPTIONAL = "ALGEBRAIC_ORGANIZATION_NOT_EXCEPTIONAL"
    AMPLITUDE_NULL_SPARSE = "AMPLITUDE_NULL_SPARSE"
    AMPLITUDE_NULL_UNEVALUABLE = "AMPLITUDE_NULL_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorAmplitudeNullResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-amplitude-null-result'
    VERSION: ClassVar[str] = "2.0.0"

    result_id: str
    config_fingerprint: str
    parent_report_sha256: str
    selected_rollout_id: str
    target_phi_y: Decimal
    eligible_reference_count: int
    cells: tuple[MatrixResponseShootingCommittorNullCell, ...]
    reference_inventory_complete: bool
    disposition: MatrixResponseShootingCommittorNullDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("result_id", "selected_rollout_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("config_fingerprint", "parent_report_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        validate_decimal(self.target_phi_y, field_name="target_phi_y", minimum=Decimal(0))
        if self.eligible_reference_count < 0:
            raise ValueError("Matrix shooting null eligible count cannot be negative")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def _eligible_null_rows(
    report: MatrixResponseAnisotropicFeasibilityQualificationReport, *, config: Any,
    selected_parent: MatrixResponseAnisotropicFeasibilityRolloutSummary | None = None,
) -> tuple[MatrixResponseAnisotropicFeasibilityRolloutSummary, ...]:
    return tuple(
        value
        for value in report.rollouts
        if value.stage is MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY
        and value.q == 2
        and value.valid
        and value.alpha_tilde_y > 0
        and value.factor_y.final_phi is not None
        and value.factor_y.final_closure_ratio is not None
        and not (value.rollout_id == config.parent_rollout_id
            and (selected_parent is None or value.rng_seed_sha256 == selected_parent.rng_seed_sha256))
    )


def execute_amplitude_conditioned_null(
    *, report: MatrixResponseAnisotropicFeasibilityQualificationReport, config: Any,
    selected_parent: MatrixResponseAnisotropicFeasibilityRolloutSummary | None = None,
) -> MatrixResponseShootingCommittorAmplitudeNullResult:
    target = selected_parent if selected_parent is not None else next(
        value for value in report.rollouts if value.rollout_id == config.parent_rollout_id
    )
    if target.factor_y.final_phi != config.null_phi_target:
        raise ValueError("Matrix shooting null target amplitude differs from authenticated parent")
    if target.rollout_id != config.parent_rollout_id or not target.valid:
        raise ValueError("Matrix shooting amplitude null has another or invalid parent")
    target_joint = bool(
        target.factor_y.final_closure_ratio is not None
        and target.factor_y.final_closure_ratio <= config.closure_ratio_max
        and target.factor_y.maximum_kernel_band_ratio <= config.kernel_band_ratio_max
        and target.factor_y.persistence_fraction
        >= Decimal(config.persistence_pass_count) / Decimal(config.rolling_window_samples)
    )
    rows = _eligible_null_rows(report, config=config, selected_parent=selected_parent)
    confidence = float(config.confidence_level)
    cells: list[MatrixResponseShootingCommittorNullCell] = []
    for band in config.null_bands:
        band_rows = tuple(
            value
            for value in rows
            if abs(value.factor_y.final_phi - config.null_phi_target) <= band
        )
        strata = {
            "all-c1a": band_rows,
            "same-member": tuple(
                value for value in band_rows if value.member_id == config.member_id
            ),
            "same-member-y11": tuple(
                value
                for value in band_rows
                if value.member_id == config.member_id
                and value.alpha_y_index == config.alpha_y_index
            ),
            "same-member-x01-y11": tuple(
                value
                for value in band_rows
                if value.member_id == config.member_id
                and value.alpha_x_index == config.alpha_x_index
                and value.alpha_y_index == config.alpha_y_index
            ),
        }
        for stratum_id in config.null_strata:
            selected = strata[stratum_id]
            closure = tuple(
                value.factor_y.final_closure_ratio <= config.closure_ratio_max for value in selected
            )
            kernel = tuple(
                value.factor_y.maximum_kernel_band_ratio <= config.kernel_band_ratio_max
                for value in selected
            )
            persistence = tuple(
                value.factor_y.persistence_fraction
                >= Decimal(config.persistence_pass_count) / Decimal(config.rolling_window_samples)
                for value in selected
            )
            count = len(selected)
            closure_count = sum(closure)
            kernel_count = sum(kernel)
            persistence_count = sum(persistence)
            joint_count = sum(
                left and middle and right
                for left, middle, right in zip(closure, kernel, persistence, strict=True)
            )
            band_id = str(band).replace("0.", "0p")
            cells.append(
                MatrixResponseShootingCommittorNullCell(
                    cell_id=f"matrix-shooting.null.band-{band_id}.{stratum_id}",
                    amplitude_band=band,
                    stratum_id=stratum_id,
                    denominator=count,
                    closure_count=closure_count,
                    kernel_count=kernel_count,
                    persistence_count=persistence_count,
                    closure_kernel_count=sum(
                        left and right for left, right in zip(closure, kernel, strict=True)
                    ),
                    closure_persistence_count=sum(
                        left and right for left, right in zip(closure, persistence, strict=True)
                    ),
                    kernel_persistence_count=sum(
                        left and right for left, right in zip(kernel, persistence, strict=True)
                    ),
                    joint_count=joint_count,
                    frozen_geometric_count=sum(value.factor_y.geometric for value in selected),
                    closure_interval=wilson_interval(
                        closure_count, count, confidence_level=confidence
                    ),
                    kernel_interval=wilson_interval(
                        kernel_count, count, confidence_level=confidence
                    ),
                    persistence_interval=wilson_interval(
                        persistence_count, count, confidence_level=confidence
                    ),
                    joint_interval=wilson_interval(
                        joint_count, count, confidence_level=confidence
                    ),
                )
            )
    cells_tuple = tuple(sorted(cells, key=lambda value: value.cell_id))
    primary_member = next(
        value
        for value in cells_tuple
        if value.amplitude_band == Decimal("0.01") and value.stratum_id == "same-member"
    )
    # Completeness and identity are current operand contracts. Historical
    # observed-null counts are never a validity condition for new outcomes.
    reference = tuple(row for row in report.rollouts
        if row.stage is MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY)
    coordinates = {(row.member_id, row.alpha_x_index, row.alpha_y_index, row.history_id, row.seed_index)
        for row in reference}
    members = {row.member_id for row in reference}
    expected = {(member, x, y, history, seed) for member in members
        for x in range(13) for y in range(13)
        for history in ("matrix-history.joint-increasing-coupling", "matrix-history.x-first-increasing-coupling", "matrix-history.y-first-increasing-coupling")
        for seed in range(3)}
    calibration = bool(report.source_config.object_fingerprint == config.parent_source_fingerprint
        and len(reference) == 9126 and len(members) == 6 and coordinates == expected
        and len({row.rollout_id for row in report.rollouts}) == len(report.rollouts))
    reasons = set()
    if not calibration:
        reasons.add("amplitude-null-parent-decode-mismatch")
    if not target_joint:
        reasons.add("selected-target-conjunction-mismatch")
    if reasons:
        disposition = MatrixResponseShootingCommittorNullDisposition.AMPLITUDE_NULL_UNEVALUABLE
    elif primary_member.denominator < config.null_same_member_minimum:
        disposition = MatrixResponseShootingCommittorNullDisposition.AMPLITUDE_NULL_SPARSE
    elif primary_member.joint_count > 0:
        disposition = MatrixResponseShootingCommittorNullDisposition.ALGEBRAIC_ORGANIZATION_NOT_EXCEPTIONAL
    else:
        disposition = MatrixResponseShootingCommittorNullDisposition.ALGEBRAIC_ORGANIZATION_EXCEPTIONAL_AT_MATCHED_AMPLITUDE
    return MatrixResponseShootingCommittorAmplitudeNullResult(
        result_id="matrix-shooting.amplitude-conditioned-null",
        config_fingerprint=config.fingerprint(),
        parent_report_sha256=config.parent_anisotropic_feasibility_report_sha256,
        selected_rollout_id=config.parent_rollout_id,
        target_phi_y=config.null_phi_target,
        eligible_reference_count=len(rows),
        cells=cells_tuple,
        reference_inventory_complete=calibration,
        disposition=disposition,
        reason_codes=tuple(sorted(reasons)),
    )


def _contains_consecutive(values: tuple[bool, ...], count: int) -> bool:
    return any(all(values[start : start + count]) for start in range(len(values) - count + 1))


def adjudicate_committor_morphology(
    cohorts: tuple[MatrixResponseShootingCommittorCohortResult, ...], *, config: Any
) -> str:
    if (
        len(cohorts) != 7
        or tuple(value.checkpoint.parent_step for value in cohorts) != config.checkpoint_steps
        or any(
            not value.complete or value.config_fingerprint != config.fingerprint()
            for value in cohorts
        )
    ):
        return "DYNAMICAL_OBJECT_UNEVALUABLE"
    if (
        sum(
            float(value.resolution_fraction.estimate or Decimal(0))
            < float(config.resolution_fraction_min)
            for value in cohorts
        )
        >= config.unresolved_checkpoint_min
    ):
        return "FINITE_HORIZON_UNRESOLVED"
    probabilities = tuple(float(value.unconditional_committor.estimate) for value in cohorts)  # type: ignore[arg-type]
    index_by_step = {cohort.checkpoint.parent_step: index for index, cohort in enumerate(cohorts)}
    formation_indices = tuple(index_by_step[step] for step in config.formation_checkpoint_steps)
    formation = tuple(probabilities[index] for index in formation_indices)
    rises = tuple(formation[index + 1] - formation[index] for index in range(3))
    transition_edge = max(range(3), key=lambda index: (rises[index], -index))
    entropies = tuple(
        0.0 if value in {0.0, 1.0} else -(value * log2(value) + (1.0 - value) * log2(1.0 - value))
        for value in probabilities
    )
    entropy_index = max(range(7), key=lambda index: (entropies[index], -index))
    edge_indices = (
        formation_indices[transition_edge],
        formation_indices[transition_edge + 1],
    )
    edge_interval_crosses_half = any(
        float(cohorts[index].unconditional_committor.lower or Decimal(0))
        <= 0.5
        <= float(cohorts[index].unconditional_committor.upper or Decimal(0))
        for index in edge_indices
    )
    sharp = bool(
        rises[transition_edge] >= float(config.sharp_rise_min)
        and edge_interval_crosses_half
        and entropy_index
        in set(
            range(
                max(0, edge_indices[0] - 1),
                min(len(cohorts), edge_indices[1] + 2),
            )
        )
    )
    source = tuple(cohorts[index_by_step[step]] for step in config.source_admitted_checkpoint_steps)
    plateau_flags = tuple(
        float(value.unconditional_committor.lower) > float(config.plateau_lower_min)  # type: ignore[arg-type]
        for value in source
    )
    plateau = _contains_consecutive(plateau_flags, config.plateau_consecutive_min)
    low = all(
        float(value.simultaneous_committor.upper)  # type: ignore[arg-type]
        < float(config.low_commitment_upper_max)
        for value in source
    )
    changes = tuple(probabilities[index + 1] - probabilities[index] for index in range(6))
    nonmonotone = any(
        changes[first] >= float(config.nonmonotone_change_min)
        and changes[second] <= -float(config.nonmonotone_change_min)
        or changes[first] <= -float(config.nonmonotone_change_min)
        and changes[second] >= float(config.nonmonotone_change_min)
        for first in range(6)
        for second in range(first + 1, 6)
    )
    patterns = [
        label
        for label, applies in (
            ("SHARP_FORMATION_TRANSITION", sharp),
            ("HIGH_COMMITMENT_PLATEAU", plateau),
            ("LOW_COMMITMENT_EVERYWHERE", low),
            ("NONMONOTONE_COMMITMENT", nonmonotone),
        )
        if applies
    ]
    if len(patterns) > 1:
        nested = set(patterns) == {"SHARP_FORMATION_TRANSITION", "HIGH_COMMITMENT_PLATEAU"}
        return (
            "SHARP_TRANSITION_WITH_HIGH_COMMITMENT_PLATEAU"
            if nested
            else "COMMITTOR_MORPHOLOGY_MIXED"
        )
    return patterns[0] if patterns else "MONTE_CARLO_RESOLUTION_INSUFFICIENT"


def adjudicate_residence(cohorts: tuple[MatrixResponseShootingCommittorCohortResult, ...], *, config: Any) -> str:
    if (
        len(cohorts) != 7
        or tuple(value.checkpoint.parent_step for value in cohorts) != config.checkpoint_steps
        or any(
            not value.complete or value.config_fingerprint != config.fingerprint()
            for value in cohorts
        )
    ):
        return "RESIDENCE_UNEVALUABLE"
    index_by_step = {cohort.checkpoint.parent_step: index for index, cohort in enumerate(cohorts)}
    source_indices = tuple(index_by_step[step] for step in config.source_admitted_checkpoint_steps)
    persistent_flags = tuple(
        value.landmark_branch_count >= config.residence_branch_min
        and value.landmark_rmst is not None
        and value.landmark_survival is not None
        and value.landmark_rmst >= config.residence_rmst_min
        and value.landmark_survival >= config.residence_survival_min
        for value in (cohorts[index] for index in source_indices)
    )
    persistent = _contains_consecutive(persistent_flags, 3)
    first_window = (
        Decimal(config.receiver_cadence_steps * config.rolling_window_samples)
        * config.primary_timestep
    )
    short_flags = []
    oscillatory_flags = []
    for cohort in cohorts:
        valid = tuple(
            value
            for value in cohort.branches
            if value.terminal is not MatrixResponseShootingCommittorBranchTerminal.INVALID_NUMERICAL_FUTURE
        )
        short_count = sum(
            value.first_y_loss is not None
            and value.first_y_loss <= config.short_loss_time_max
            and value.tau_00 == first_window
            for value in valid
        )
        short_flags.append(
            bool(valid) and short_count / len(valid) >= float(config.short_loss_fraction_min)
        )
        oscillatory_count = sum(
            value.y_transition_count >= config.oscillation_transition_min for value in valid
        )
        occupancies = sorted(
            float(value.y_occupancy) for value in valid if value.y_occupancy is not None
        )
        oscillatory_flags.append(
            bool(valid)
            and oscillatory_count / len(valid) >= float(config.oscillation_branch_fraction_min)
            and bool(occupancies)
            and float(config.oscillation_occupancy_min)
            <= median(occupancies)
            <= float(config.oscillation_occupancy_max)
        )
    short = all(short_flags[index] for index in source_indices)
    oscillatory = sum(oscillatory_flags) >= 3
    patterns = [
        label
        for label, applies in (
            ("FINITE_HORIZON_PERSISTENT_RESIDENCE", persistent),
            ("SHORT_LIVED_ALIGNMENT", short),
            ("OSCILLATORY_ADMISSION", oscillatory),
        )
        if applies
    ]
    if len(patterns) > 1:
        return "RESIDENCE_MIXED"
    return patterns[0] if patterns else "RESIDENCE_MIXED"


@dataclass(frozen=True, slots=True)
class MatrixResponseShootingCommittorTerminalReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-terminal-report'

    report_id: str
    config_fingerprint: str
    implementation_commit: str
    parent_report_sha256: str
    replay_fingerprint: str
    bridge_fingerprint: str
    cohort_fingerprints: tuple[str, ...]
    linearization_fingerprint: str | None
    amplitude_null_fingerprint: str
    numerical_axis: str
    committor_morphology_axis: str
    residence_axis: str
    stability_axis: str
    amplitude_null_axis: str
    combined_interpretation: str
    scientific_terminal: str
    operational_complete: bool
    condition_false_descendants: tuple[str, ...]
    evidence_ceiling: str
    reason_codes: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("report_id", "evidence_ceiling"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "config_fingerprint",
            "parent_report_sha256",
            "replay_fingerprint",
            "bridge_fingerprint",
            "amplitude_null_fingerprint",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if len(self.implementation_commit) != 40:
            raise ValueError("Matrix shooting terminal report requires a Git SHA-1")
        int(self.implementation_commit, 16)
        for value in self.cohort_fingerprints:
            validate_sha256(value, field_name="cohort_fingerprints")
        if self.linearization_fingerprint is not None:
            validate_sha256(self.linearization_fingerprint, field_name="linearization_fingerprint")
        require_sorted_unique_strings(
            self.condition_false_descendants,
            field_name="condition_false_descendants",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not self.operational_complete or self.grants_authority:
            raise ValueError("Matrix shooting terminal must be operationally complete and nonauthorizing")


def finalize_matrix_response_study_shooting_committor(
    *,
    implementation_commit: str,
    config: Any,
    replay: MatrixResponseShootingCommittorReplayResult,
    bridge: MatrixResponseShootingCommittorBridgeResult,
    cohorts: tuple[MatrixResponseShootingCommittorCohortResult, ...],
    linearization: MatrixResponseShootingCommittorLinearizationResult | None,
    amplitude_null: MatrixResponseShootingCommittorAmplitudeNullResult,
) -> MatrixResponseShootingCommittorTerminalReport:
    numerical_axis = bridge.disposition.value
    condition_false: set[str] = {
        "six-matrix-response-rescue",
        "control-stabilization",
        "base-system-prospective-evaluation",
        "response-law",
    }
    reasons = set(bridge.reason_codes)
    reasons.update(amplitude_null.reason_codes)
    for cohort in cohorts:
        reasons.update(cohort.reason_codes)
    if linearization is not None:
        reasons.update(linearization.reason_codes)
    cohort_roster_complete = bool(
        len(cohorts) == 7
        and tuple(value.checkpoint.parent_step for value in cohorts) == config.checkpoint_steps
        and all(
            value.complete and value.config_fingerprint == config.fingerprint() for value in cohorts
        )
    )
    if bridge.disposition is MatrixResponseShootingCommittorNumericalDisposition.EVENT_PATH_NUMERICALLY_CONCORDANT:
        if not cohort_roster_complete:
            reasons.add("shooting-corpus-incomplete")
        if linearization is None:
            reasons.add("local-linearization-missing")
    if bridge.disposition is not MatrixResponseShootingCommittorNumericalDisposition.EVENT_PATH_NUMERICALLY_CONCORDANT:
        condition_false.update(("shooting", "local-linearization"))
        morphology = "DYNAMICAL_OBJECT_UNEVALUABLE"
        residence = "RESIDENCE_UNEVALUABLE"
        stability = "CONDITION_FALSE"
        combined = "DYNAMICAL_OBJECT_UNEVALUABLE"
        scientific_terminal = bridge.disposition.value
    else:
        morphology = adjudicate_committor_morphology(cohorts, config=config)
        residence = adjudicate_residence(cohorts, config=config)
        stability = (
            linearization.disposition.value if linearization is not None else "CONDITION_FALSE"
        )
        if linearization is None or not cohort_roster_complete:
            combined = "DYNAMICAL_OBJECT_UNEVALUABLE"
        else:
            tube = bool(
                morphology
                in {"SHARP_FORMATION_TRANSITION", "SHARP_TRANSITION_WITH_HIGH_COMMITMENT_PLATEAU"}
                and linearization.disposition
                is MatrixResponseShootingCommittorStabilityDisposition.ONE_QUOTIENT_UNSTABLE_DIRECTION
                and linearization.y_structural_overlap_pass
            )
            basin = bool(
                morphology
                in {"HIGH_COMMITMENT_PLATEAU", "SHARP_TRANSITION_WITH_HIGH_COMMITMENT_PLATEAU"}
                and residence == "FINITE_HORIZON_PERSISTENT_RESIDENCE"
            )
            oscillation = bool(
                morphology == "NONMONOTONE_COMMITMENT" and residence == "OSCILLATORY_ADMISSION"
            )
            accidental = bool(
                morphology == "LOW_COMMITMENT_EVERYWHERE" and residence == "SHORT_LIVED_ALIGNMENT"
            )
            if tube and basin:
                combined = "TRANSITION_INTO_FINITE_HORIZON_BASIN_LIKE"
            elif tube:
                combined = "TRANSITION_TUBE_LIKE"
            elif basin:
                combined = "FINITE_HORIZON_BASIN_LIKE"
            elif oscillation:
                combined = "PREPARATION_OSCILLATION_LIKE"
            elif accidental:
                combined = "ACCIDENTAL_ALIGNMENT_LIKE"
            elif morphology in {
                "DYNAMICAL_OBJECT_UNEVALUABLE",
                "FINITE_HORIZON_UNRESOLVED",
                "MONTE_CARLO_RESOLUTION_INSUFFICIENT",
            }:
                combined = "DYNAMICAL_OBJECT_UNEVALUABLE"
            else:
                combined = "DYNAMICAL_OBJECT_MIXED"
        scientific_terminal = combined
    return MatrixResponseShootingCommittorTerminalReport(
        report_id="matrix-shooting.terminal-report",
        config_fingerprint=config.fingerprint(),
        implementation_commit=implementation_commit,
        parent_report_sha256=config.parent_anisotropic_feasibility_report_sha256,
        replay_fingerprint=replay.fingerprint(),
        bridge_fingerprint=bridge.fingerprint(),
        cohort_fingerprints=tuple(value.fingerprint() for value in cohorts),
        linearization_fingerprint=(None if linearization is None else linearization.fingerprint()),
        amplitude_null_fingerprint=amplitude_null.fingerprint(),
        numerical_axis=numerical_axis,
        committor_morphology_axis=morphology,
        residence_axis=residence,
        stability_axis=stability,
        amplitude_null_axis=amplitude_null.disposition.value,
        combined_interpretation=combined,
        scientific_terminal=scientific_terminal,
        operational_complete=True,
        condition_false_descendants=tuple(sorted(condition_false)),
        evidence_ceiling="outcome-visible-event-selected-nonpromotable",
        reason_codes=tuple(sorted(reasons)),
        grants_authority=False,
    )


__all__ = [
    'MatrixResponseShootingCommittorAmplitudeNullResult',
    'MatrixResponseShootingCommittorBranchResult',
    'MatrixResponseShootingCommittorBranchTerminal',
    'MatrixResponseShootingCommittorBridgeResult',
    'MatrixResponseShootingCommittorCohortResult',
    'MatrixResponseShootingCommittorFactorObservation',
    'MatrixResponseShootingCommittorLinearizationResult',
    'MatrixResponseShootingCommittorNullDisposition',
    'MatrixResponseShootingCommittorNumericalDisposition',
    'MatrixResponseShootingCommittorObservation',
    'MatrixResponseShootingCommittorReplayResult',
    'MatrixResponseShootingCommittorRollingLabel',
    'MatrixResponseShootingCommittorStabilityDisposition',
    'MatrixResponseShootingCommittorTerminalReport',
    'MatrixResponseShootingCommittorWilsonInterval',
    'adjudicate_committor_morphology',
    'adjudicate_residence',
    'build_replay_result',
    'execute_amplitude_conditioned_null',
    'execute_brownian_bridge_replay',
    'execute_local_linearization',
    'execute_shooting_cohort',
    'finalize_matrix_response_study_shooting_committor',
    'observe_state',
    'primary_observations_from_replay',
    'reduce_shooting_cohort',
    'rolling_labels',
    'wilson_interval',
]
