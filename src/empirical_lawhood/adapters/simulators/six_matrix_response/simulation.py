"""Bounded native action, BAOAB, checkpoint and receiver semantics for Six-matrix response."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView, SixMatrixResponseSixMatrixSourceConfig
from .gradients import SixMatrixParameters, analytic_gradient_terms, coupling_derivative_density, optimized_action_terms
from .model import ComplexArray, SixMatrixState, hermitian_part, hermiticity_residual, ideal_state

MATRIX_RESPONSE_CANONICAL_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
MATRIX_RESPONSE_MAXIMUM_REQUEST_BYTES = 4 * 1024 * 1024
_ALPHA_MIN = Decimal("0")
_ALPHA_MAX = Decimal("8")
_MAX_ALPHA_RATE_PER_STEP = Decimal(1) / Decimal(384)


class SixMatrixResponseConstitution(StrEnum):
    EMPTY = "00"
    X_ONLY = "10"
    Y_ONLY = "01"
    PRODUCT = "11"


class SixMatrixResponseActionDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    CLIPPED = "CLIPPED"
    HOLD = "HOLD"
    REFUSED = "REFUSED"


class SixMatrixResponseEpisodeTerminal(StrEnum):
    COMPLETED = "COMPLETED"
    PARTIAL_CHECKPOINT = "PARTIAL_CHECKPOINT"
    ACTION_REFUSED = "ACTION_REFUSED"
    INVALID_INITIAL_STATE = "INVALID_INITIAL_STATE"
    INVALID_CHECKPOINT = "INVALID_CHECKPOINT"
    NONFINITE_STATE = "NONFINITE_STATE"


@dataclass(frozen=True, slots=True)
class SixMatrixResponseScaledCouplings(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-scaled-couplings'

    alpha_tilde_x: Decimal
    alpha_tilde_y: Decimal

    def __post_init__(self) -> None:
        for name in ("alpha_tilde_x", "alpha_tilde_y"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class SixMatrixResponseNativeActionRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-native-action-request'

    action_id: str
    requested_start: SixMatrixResponseScaledCouplings
    requested_target: SixMatrixResponseScaledCouplings
    ramp_steps: int
    allow_clipping: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.action_id, field_name="action_id")
        if self.ramp_steps < 1:
            raise ValueError("Six-matrix response action ramp must contain at least one step")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseNativeActionLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-native-action-ledger'

    action_id: str
    disposition: SixMatrixResponseActionDisposition
    requested: SixMatrixResponseScaledCouplings
    accepted: SixMatrixResponseScaledCouplings
    applied: SixMatrixResponseScaledCouplings
    realized: SixMatrixResponseScaledCouplings
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.action_id, field_name="action_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition in {SixMatrixResponseActionDisposition.ACCEPTED, SixMatrixResponseActionDisposition.HOLD}:
            if self.reason_codes:
                raise ValueError("accepted/HOLD Six-matrix response actions cannot retain failure reasons")
        elif not self.reason_codes:
            raise ValueError("clipped/refused Six-matrix response actions require typed reasons")
        if self.disposition is SixMatrixResponseActionDisposition.HOLD and not (
            self.requested == self.accepted == self.applied == self.realized
        ):
            raise ValueError("Six-matrix response HOLD must preserve all four action stages")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseRNGStreamReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-rng-stream-receipt'

    seed_root_id: str
    purpose_id: str
    stream_index: int
    derivation_rule_id: str
    derived_seed_sha256: str
    bit_generator_id: str

    def __post_init__(self) -> None:
        for name in ("seed_root_id", "purpose_id", "derivation_rule_id", "bit_generator_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.stream_index < 0:
            raise ValueError("Six-matrix response RNG stream index cannot be negative")
        validate_sha256(self.derived_seed_sha256, field_name="derived_seed_sha256")
        if self.bit_generator_id != "numpy.pcg64dxsm":
            raise ValueError("Six-matrix response baseline fixes NumPy PCG64DXSM")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseCheckpoint(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-checkpoint'

    checkpoint_id: str
    request: ObjectIdentity
    q: int
    step_index: int
    total_steps: int
    alpha_tilde_x: Decimal
    alpha_tilde_y: Decimal
    positions_base64: str
    momenta_base64: str
    positions_sha256: str
    momenta_sha256: str
    state_sha256: str
    rng_stream: SixMatrixResponseRNGStreamReceipt
    rng_state_json: str
    rng_state_sha256: str
    receivers: tuple[SixMatrixResponseFastReceiver, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.checkpoint_id, field_name="checkpoint_id")
        if self.q not in {2, 3, 4} or not 0 <= self.step_index <= self.total_steps:
            raise ValueError("Six-matrix response checkpoint q/step bounds differ")
        for name in ("alpha_tilde_x", "alpha_tilde_y"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        decoded: dict[str, bytes] = {}
        expected_bytes = 2 * 3 * (self.q**2) ** 2 * 16
        for field_name, digest_name in (
            ("positions_base64", "positions_sha256"),
            ("momenta_base64", "momenta_sha256"),
        ):
            try:
                raw = base64.b64decode(getattr(self, field_name), validate=True)
            except ValueError as error:
                raise ValueError(f"Six-matrix response checkpoint {field_name} is not canonical base64") from error
            if len(raw) != expected_bytes or base64.b64encode(raw).decode("ascii") != getattr(
                self, field_name
            ):
                raise ValueError(f"Six-matrix response checkpoint {field_name} has another byte geometry")
            validate_sha256(getattr(self, digest_name), field_name=digest_name)
            if sha256(raw).hexdigest() != getattr(self, digest_name):
                raise ValueError(f"Six-matrix response checkpoint {field_name} hash differs")
            decoded[field_name] = raw
        validate_sha256(self.rng_state_sha256, field_name="rng_state_sha256")
        if sha256(self.rng_state_json.encode("utf-8")).hexdigest() != self.rng_state_sha256:
            raise ValueError("Six-matrix response checkpoint RNG state hash differs")
        try:
            rng_document = json.loads(self.rng_state_json)
        except json.JSONDecodeError as error:
            raise ValueError("Six-matrix response checkpoint RNG state is not JSON") from error
        canonical_rng = json.dumps(rng_document, sort_keys=True, separators=(",", ":"))
        if canonical_rng != self.rng_state_json:
            raise ValueError("Six-matrix response checkpoint RNG state is not canonical")
        expected_state = _state_sha256(
            q=self.q,
            step_index=self.step_index,
            alpha_x=self.alpha_tilde_x,
            alpha_y=self.alpha_tilde_y,
            positions=decoded["positions_base64"],
            momenta=decoded["momenta_base64"],
            rng_state=self.rng_state_json,
        )
        validate_sha256(self.state_sha256, field_name="state_sha256")
        if expected_state != self.state_sha256:
            raise ValueError("Six-matrix response checkpoint combined state identity differs")
        sample_steps = tuple(value.sample_step for value in self.receivers)
        if sample_steps != tuple(sorted(set(sample_steps))) or any(
            value > self.step_index for value in sample_steps
        ):
            raise ValueError("Six-matrix response checkpoint receiver prefix differs from its step")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseFastReceiver(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-fast-receiver'

    receiver_id: str
    sample_step: int
    alpha_tilde_x: Decimal
    alpha_tilde_y: Decimal
    action_density: Decimal
    radius_x: Decimal
    radius_y: Decimal
    closure_residual_x: Decimal
    closure_residual_y: Decimal
    cross_commutator_norm: Decimal
    kinetic_density: Decimal
    coupling_derivative_x: Decimal
    coupling_derivative_y: Decimal
    hermiticity_residual: Decimal
    finite: bool
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receiver_id, field_name="receiver_id")
        if self.sample_step < 0:
            raise ValueError("Six-matrix response receiver sample step cannot be negative")
        for field_name in (
            "alpha_tilde_x",
            "alpha_tilde_y",
            "action_density",
            "radius_x",
            "radius_y",
            "closure_residual_x",
            "closure_residual_y",
            "cross_commutator_norm",
            "kinetic_density",
            "coupling_derivative_x",
            "coupling_derivative_y",
            "hermiticity_residual",
        ):
            validate_decimal(getattr(self, field_name), field_name=field_name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.valid != (self.finite and not self.reason_codes):
            raise ValueError("Six-matrix response receiver validity differs from its finite/reason record")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseEpisodeRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-episode-request'

    request_id: str
    task_id: str
    output_id: str
    source_config: ObjectIdentity
    family_member: SixMatrixResponseModelFamilyMember
    numerical_view: SixMatrixResponseNumericalView
    q: int
    seed_root_id: str
    rng_purpose_id: str
    rng_stream_index: int
    scientific_seed_sha256: str
    initial_constitution: SixMatrixResponseConstitution
    action: SixMatrixResponseNativeActionRequest
    integration_steps: int
    receiver_cadence_steps: int
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("request_id", "task_id", "output_id", "seed_root_id", "rng_purpose_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.scientific_seed_sha256, field_name="scientific_seed_sha256")
        if self.q not in {2, 3, 4}:
            raise ValueError("Six-matrix response request q must be 2, 3 or 4")
        if self.rng_stream_index < 0 or self.integration_steps < 1:
            raise ValueError("Six-matrix response request RNG/step bounds differ")
        if not 1 <= self.receiver_cadence_steps <= self.integration_steps:
            raise ValueError("Six-matrix response receiver cadence is outside the episode")
        if self.action.ramp_steps > self.integration_steps:
            raise ValueError("Six-matrix response action ramp cannot exceed the episode")
        if self.evidence_ceiling is not EvidenceCeiling.MEASUREMENT:
            raise ValueError("Six-matrix response simulator emits measurement evidence only")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("Six-matrix response source request has excessive outcome access")
        if self.grants_authority:
            raise ValueError("Six-matrix response episode requests cannot grant authority")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseEpisodeResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-episode-result'

    episode_id: str
    task_id: str
    output_id: str
    request: ObjectIdentity
    terminal: SixMatrixResponseEpisodeTerminal
    completed_steps: int
    total_steps: int
    action_ledger: SixMatrixResponseNativeActionLedger
    rng_stream: SixMatrixResponseRNGStreamReceipt
    checkpoint: SixMatrixResponseCheckpoint | None
    receivers: tuple[SixMatrixResponseFastReceiver, ...]
    reason_codes: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    scientific_verdict_ids: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("episode_id", "task_id", "output_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if not 0 <= self.completed_steps <= self.total_steps:
            raise ValueError("Six-matrix response episode completed-step count differs")
        if tuple(value.sample_step for value in self.receivers) != tuple(
            sorted(set(value.sample_step for value in self.receivers))
        ):
            raise ValueError("Six-matrix response receivers must have increasing unique sample steps")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.scientific_verdict_ids or self.grants_authority:
            raise ValueError("Six-matrix response native simulator cannot adjudicate or grant authority")
        if self.checkpoint is not None and self.checkpoint.step_index != self.completed_steps:
            raise ValueError("Six-matrix response episode/checkpoint step identity differs")
        if self.terminal is SixMatrixResponseEpisodeTerminal.COMPLETED and (
            self.completed_steps != self.total_steps or self.checkpoint is None
        ):
            raise ValueError("completed Six-matrix response episode lacks its terminal checkpoint")


def _as_decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("cannot encode a nonfinite Six-matrix response numerical value")
    return Decimal(repr(float(value)))


def derive_rng_stream(
    *, seed_root_id: str, purpose_id: str, stream_index: int, derivation_rule_id: str,
    scientific_seed_sha256: str,
) -> tuple[np.random.Generator, SixMatrixResponseRNGStreamReceipt]:
    for name, value in (
        ("seed_root_id", seed_root_id),
        ("purpose_id", purpose_id),
        ("derivation_rule_id", derivation_rule_id),
    ):
        validate_stable_id(value, field_name=name)
    if stream_index < 0:
        raise ValueError("Six-matrix response stream index cannot be negative")
    validate_sha256(scientific_seed_sha256, field_name="scientific_seed_sha256")
    digest = bytes.fromhex(scientific_seed_sha256)
    receipt = SixMatrixResponseRNGStreamReceipt(
        seed_root_id=seed_root_id,
        purpose_id=purpose_id,
        stream_index=stream_index,
        derivation_rule_id=derivation_rule_id,
        derived_seed_sha256=digest.hex(),
        bit_generator_id="numpy.pcg64dxsm",
    )
    return np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:16], "big"))), receipt


def hermitian_noise(*, rng: np.random.Generator, q: int) -> ComplexArray:
    """Sample six isotropic Hermitian Gaussian matrices in the HS real metric."""

    n = q**2
    result = np.zeros((2, 3, n, n), dtype="<c16")
    for sector in range(2):
        for component in range(3):
            matrix = result[sector, component]
            matrix[np.diag_indices(n)] = rng.normal(size=n)
            rows, columns = np.triu_indices(n, 1)
            values = (rng.normal(size=len(rows)) + 1.0j * rng.normal(size=len(rows))) / np.sqrt(2)
            matrix[rows, columns] = values
            matrix[columns, rows] = values.conj()
    return result


def _parameters(state: SixMatrixState, member: SixMatrixResponseModelFamilyMember) -> SixMatrixParameters:
    return SixMatrixParameters(
        q=state.q,
        mass_x=float(member.mass_x),
        mass_y=float(member.mass_y),
        gamma=float(member.cross_coupling_gamma),
        alpha_tilde_x=state.alpha_tilde_x,
        alpha_tilde_y=state.alpha_tilde_y,
    )


class BAOABGradientCache:
    """One trajectory's last conservative force, keyed by every force input.

    The key and result own immutable bytes, so even externally changed array
    storage cannot make a hit stale. Momenta and the random stream are not force
    inputs. The ordinary uncached step remains available as the reference.
    """

    def __init__(self) -> None:
        self._positions: bytes | None = None
        self._parameters: SixMatrixParameters | None = None
        self._gradient: ComplexArray | None = None

    def evaluate(self, positions: ComplexArray, parameters: SixMatrixParameters) -> ComplexArray:
        key = positions.tobytes()
        if self._gradient is not None and self._positions == key and self._parameters == parameters:
            return self._gradient
        gradient = analytic_gradient_terms(positions, parameters).total
        self._gradient = np.frombuffer(gradient.tobytes(), dtype="<c16").reshape(gradient.shape)
        self._positions = key
        self._parameters = parameters
        return self._gradient


def _gradient(
    positions: ComplexArray,
    parameters: SixMatrixParameters,
    cache: BAOABGradientCache | None,
) -> ComplexArray:
    if cache is not None:
        return cache.evaluate(positions, parameters)
    return analytic_gradient_terms(positions, parameters).total


def baoab_step(
    state: SixMatrixState,
    *,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    next_alpha_tilde_x: float,
    next_alpha_tilde_y: float,
    rng: np.random.Generator,
    gradient_cache: BAOABGradientCache | None = None,
) -> SixMatrixState:
    """Advance one BAOAB step while preserving the historical RNG stream exactly."""

    # Preserve the exact historical operation order.  NumPy's normal sampler
    # can alter floating-point status on some builds, so sampling before the
    # first B/A operations is not byte-equivalent even with the same draw.
    dt = float(numerical_view.timestep)
    friction = float(numerical_view.friction_gamma)
    temperature = float(numerical_view.bath_temperature)
    gradient = _gradient(state.positions, _parameters(state, member), gradient_cache)
    momentum = hermitian_part(state.momenta - 0.5 * dt * gradient)
    position = hermitian_part(state.positions + 0.5 * dt * momentum)
    decay = float(np.exp(-friction * dt))
    noise_scale = float(np.sqrt(temperature * (1.0 - decay**2)))
    momentum = hermitian_part(
        decay * momentum + noise_scale * hermitian_noise(rng=rng, q=state.q)
    )
    position = hermitian_part(position + 0.5 * dt * momentum)
    next_parameters = SixMatrixParameters(
        q=state.q,
        mass_x=float(member.mass_x),
        mass_y=float(member.mass_y),
        gamma=float(member.cross_coupling_gamma),
        alpha_tilde_x=next_alpha_tilde_x,
        alpha_tilde_y=next_alpha_tilde_y,
    )
    next_gradient = _gradient(position, next_parameters, gradient_cache)
    momentum = hermitian_part(momentum - 0.5 * dt * next_gradient)
    return SixMatrixState(
        q=state.q,
        positions=position,
        momenta=momentum,
        step_index=state.step_index + 1,
        alpha_tilde_x=next_alpha_tilde_x,
        alpha_tilde_y=next_alpha_tilde_y,
    )


def baoab_step_with_hermitian_noise(
    state: SixMatrixState,
    *,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    next_alpha_tilde_x: float,
    next_alpha_tilde_y: float,
    standardized_noise: ComplexArray,
    gradient_cache: BAOABGradientCache | None = None,
) -> SixMatrixState:
    """Advance one ordered BAOAB step with one supplied HS-standard innovation.

    This simulator-local seam owns no RNG, branch selection, persistence or
    scientific verdict.  The ordinary :func:`baoab_step` remains the sole
    historical RNG-driven wrapper.
    """

    noise = np.asarray(standardized_noise)
    expected_shape = (2, 3, state.n, state.n)
    if noise.shape != expected_shape or noise.dtype != np.dtype("complex128"):
        raise ValueError("Six-matrix response supplied BAOAB noise has another shape or dtype")
    if not np.isfinite(noise).all():
        raise ValueError("Six-matrix response supplied BAOAB noise must be finite")
    if hermiticity_residual(noise) > 1e-12:
        raise ValueError("Six-matrix response supplied BAOAB noise must be Hermitian")

    dt = float(numerical_view.timestep)
    friction = float(numerical_view.friction_gamma)
    temperature = float(numerical_view.bath_temperature)
    gradient = _gradient(state.positions, _parameters(state, member), gradient_cache)
    momentum = hermitian_part(state.momenta - 0.5 * dt * gradient)
    position = hermitian_part(state.positions + 0.5 * dt * momentum)
    decay = float(np.exp(-friction * dt))
    noise_scale = float(np.sqrt(temperature * (1.0 - decay**2)))
    momentum = hermitian_part(decay * momentum + noise_scale * noise)
    position = hermitian_part(position + 0.5 * dt * momentum)
    next_parameters = SixMatrixParameters(
        q=state.q,
        mass_x=float(member.mass_x),
        mass_y=float(member.mass_y),
        gamma=float(member.cross_coupling_gamma),
        alpha_tilde_x=next_alpha_tilde_x,
        alpha_tilde_y=next_alpha_tilde_y,
    )
    next_gradient = _gradient(position, next_parameters, gradient_cache)
    momentum = hermitian_part(momentum - 0.5 * dt * next_gradient)
    return SixMatrixState(
        q=state.q,
        positions=position,
        momenta=momentum,
        step_index=state.step_index + 1,
        alpha_tilde_x=next_alpha_tilde_x,
        alpha_tilde_y=next_alpha_tilde_y,
    )


def baoab_step_and_capture_hermitian_noise(
    state: SixMatrixState,
    *,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    next_alpha_tilde_x: float,
    next_alpha_tilde_y: float,
    rng: np.random.Generator,
    gradient_cache: BAOABGradientCache | None = None,
) -> tuple[SixMatrixState, ComplexArray]:
    """Advance the historical path and expose its innovation without reordering it."""

    dt = float(numerical_view.timestep)
    friction = float(numerical_view.friction_gamma)
    temperature = float(numerical_view.bath_temperature)
    gradient = _gradient(state.positions, _parameters(state, member), gradient_cache)
    momentum = hermitian_part(state.momenta - 0.5 * dt * gradient)
    position = hermitian_part(state.positions + 0.5 * dt * momentum)
    decay = float(np.exp(-friction * dt))
    noise_scale = float(np.sqrt(temperature * (1.0 - decay**2)))
    noise = hermitian_noise(rng=rng, q=state.q)
    momentum = hermitian_part(decay * momentum + noise_scale * noise)
    position = hermitian_part(position + 0.5 * dt * momentum)
    next_parameters = SixMatrixParameters(
        q=state.q,
        mass_x=float(member.mass_x),
        mass_y=float(member.mass_y),
        gamma=float(member.cross_coupling_gamma),
        alpha_tilde_x=next_alpha_tilde_x,
        alpha_tilde_y=next_alpha_tilde_y,
    )
    next_gradient = _gradient(position, next_parameters, gradient_cache)
    momentum = hermitian_part(momentum - 0.5 * dt * next_gradient)
    return (
        SixMatrixState(
            q=state.q,
            positions=position,
            momenta=momentum,
            step_index=state.step_index + 1,
            alpha_tilde_x=next_alpha_tilde_x,
            alpha_tilde_y=next_alpha_tilde_y,
        ),
        noise,
    )


def _resolve_action(
    request: SixMatrixResponseNativeActionRequest,
) -> tuple[SixMatrixResponseScaledCouplings, SixMatrixResponseActionDisposition, tuple[str, ...]]:
    start = request.requested_start
    target = request.requested_target
    if target == start:
        return target, SixMatrixResponseActionDisposition.HOLD, ()
    target_values = (target.alpha_tilde_x, target.alpha_tilde_y)
    start_values = (start.alpha_tilde_x, start.alpha_tilde_y)
    reasons: set[str] = set()
    bounded = []
    for value in target_values:
        if value < _ALPHA_MIN or value > _ALPHA_MAX:
            reasons.add("action-domain-exceeded")
        bounded.append(min(_ALPHA_MAX, max(_ALPHA_MIN, value)))
    max_delta = _MAX_ALPHA_RATE_PER_STEP * request.ramp_steps
    rate_bounded = []
    for initial, value in zip(start_values, bounded, strict=True):
        if abs(value - initial) > max_delta:
            reasons.add("action-rate-exceeded")
            value = initial + (max_delta if value > initial else -max_delta)
        rate_bounded.append(value)
    if reasons and not request.allow_clipping:
        return start, SixMatrixResponseActionDisposition.REFUSED, tuple(sorted(reasons))
    accepted = SixMatrixResponseScaledCouplings(rate_bounded[0], rate_bounded[1])
    return (
        accepted,
        (SixMatrixResponseActionDisposition.CLIPPED if reasons else SixMatrixResponseActionDisposition.ACCEPTED),
        tuple(sorted(reasons)),
    )


def _state_sha256(
    *,
    q: int,
    step_index: int,
    alpha_x: Decimal,
    alpha_y: Decimal,
    positions: bytes,
    momenta: bytes,
    rng_state: str,
) -> str:
    digest = sha256()
    for value in (
        str(q).encode(),
        str(step_index).encode(),
        str(alpha_x).encode(),
        str(alpha_y).encode(),
        positions,
        momenta,
        rng_state.encode(),
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def checkpoint_from_state(
    *,
    request: SixMatrixResponseEpisodeRequest,
    state: SixMatrixState,
    rng: np.random.Generator,
    rng_stream: SixMatrixResponseRNGStreamReceipt,
    receivers: tuple[SixMatrixResponseFastReceiver, ...],
) -> SixMatrixResponseCheckpoint:
    return checkpoint_from_phase_state(
        checkpoint_id=f"checkpoint.{request.request_id}.{state.step_index}",
        request=ObjectIdentity.from_record(request.request_id, request),
        total_steps=request.integration_steps,
        state=state,
        rng=rng,
        rng_stream=rng_stream,
        receivers=receivers,
    )


def checkpoint_from_phase_state(
    *,
    checkpoint_id: str,
    request: ObjectIdentity,
    total_steps: int,
    state: SixMatrixState,
    rng: np.random.Generator,
    rng_stream: SixMatrixResponseRNGStreamReceipt,
    receivers: tuple[SixMatrixResponseFastReceiver, ...],
) -> SixMatrixResponseCheckpoint:
    """Encode the existing native phase checkpoint for an exact request identity.

    Additional action/observer state belongs to the requesting source's enclosing
    checkpoint. This helper does not select a request, effect, authority or route.
    """
    positions = state.positions.tobytes(order="C")
    momenta = state.momenta.tobytes(order="C")
    rng_state = json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":"))
    alpha_x = _as_decimal(state.alpha_tilde_x)
    alpha_y = _as_decimal(state.alpha_tilde_y)
    return SixMatrixResponseCheckpoint(
        checkpoint_id=checkpoint_id,
        request=request,
        q=state.q,
        step_index=state.step_index,
        total_steps=total_steps,
        alpha_tilde_x=alpha_x,
        alpha_tilde_y=alpha_y,
        positions_base64=base64.b64encode(positions).decode("ascii"),
        momenta_base64=base64.b64encode(momenta).decode("ascii"),
        positions_sha256=sha256(positions).hexdigest(),
        momenta_sha256=sha256(momenta).hexdigest(),
        state_sha256=_state_sha256(
            q=state.q,
            step_index=state.step_index,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            positions=positions,
            momenta=momenta,
            rng_state=rng_state,
        ),
        rng_stream=rng_stream,
        rng_state_json=rng_state,
        rng_state_sha256=sha256(rng_state.encode()).hexdigest(),
        receivers=receivers,
    )


def state_from_checkpoint(
    checkpoint: SixMatrixResponseCheckpoint,
) -> tuple[SixMatrixState, np.random.Generator]:
    shape = (2, 3, checkpoint.q**2, checkpoint.q**2)
    positions = np.frombuffer(base64.b64decode(checkpoint.positions_base64), dtype="<c16").reshape(
        shape
    )
    momenta = np.frombuffer(base64.b64decode(checkpoint.momenta_base64), dtype="<c16").reshape(
        shape
    )
    state = SixMatrixState(
        q=checkpoint.q,
        positions=positions,
        momenta=momenta,
        step_index=checkpoint.step_index,
        alpha_tilde_x=float(checkpoint.alpha_tilde_x),
        alpha_tilde_y=float(checkpoint.alpha_tilde_y),
    )
    rng = np.random.Generator(np.random.PCG64DXSM())
    rng.bit_generator.state = json.loads(checkpoint.rng_state_json)
    return state, rng


def fast_receiver(
    *, state: SixMatrixState, member: SixMatrixResponseModelFamilyMember
) -> SixMatrixResponseFastReceiver:
    parameters = _parameters(state, member)
    action = optimized_action_terms(state.positions, parameters).total / parameters.n
    radii = tuple(
        float(np.trace(np.sum(sector @ sector, axis=0)).real / parameters.n)
        for sector in state.positions
    )
    closure = []
    for sector, alpha in zip(
        state.positions, (parameters.alpha_x, parameters.alpha_y), strict=True
    ):
        scale = 2.0 * alpha / 3.0
        residuals = []
        for a, (b, c) in enumerate(((1, 2), (2, 0), (0, 1))):
            lhs = sector[b] @ sector[c] - sector[c] @ sector[b]
            residuals.append(float(np.linalg.norm(lhs - 1.0j * scale * sector[a])))
        closure.append(max(residuals))
    cross = max(
        float(
            np.linalg.norm(
                state.positions[0, a] @ state.positions[1, b]
                - state.positions[1, b] @ state.positions[0, a]
            )
        )
        for a in range(3)
        for b in range(3)
    )
    kinetic = float(np.vdot(state.momenta, state.momenta).real / (2.0 * parameters.n))
    derivative = coupling_derivative_density(state.positions, parameters)
    hermitian = max(hermiticity_residual(state.positions), hermiticity_residual(state.momenta))
    values = (action, *radii, *closure, cross, kinetic, *derivative, hermitian)
    finite = all(np.isfinite(value) for value in values)
    reasons = () if finite else ("nonfinite-fast-receiver",)
    if not finite:
        raise FloatingPointError("Six-matrix response fast receiver is nonfinite")
    return SixMatrixResponseFastReceiver(
        receiver_id=f"receiver.fast.{state.step_index}",
        sample_step=state.step_index,
        alpha_tilde_x=_as_decimal(state.alpha_tilde_x),
        alpha_tilde_y=_as_decimal(state.alpha_tilde_y),
        action_density=_as_decimal(action),
        radius_x=_as_decimal(radii[0]),
        radius_y=_as_decimal(radii[1]),
        closure_residual_x=_as_decimal(closure[0]),
        closure_residual_y=_as_decimal(closure[1]),
        cross_commutator_norm=_as_decimal(cross),
        kinetic_density=_as_decimal(kinetic),
        coupling_derivative_x=_as_decimal(derivative[0]),
        coupling_derivative_y=_as_decimal(derivative[1]),
        hermiticity_residual=_as_decimal(hermitian),
        finite=finite,
        valid=finite and not reasons,
        reason_codes=reasons,
    )


class SixMatrixResponseSixMatrixEpisodeEngine:
    """In-process numerical engine; it emits native measurements, never verdicts."""

    def __init__(self, source_config: SixMatrixResponseSixMatrixSourceConfig) -> None:
        self.source_config = source_config

    def _validate_request(self, request: SixMatrixResponseEpisodeRequest) -> None:
        expected_source = ObjectIdentity.from_record(
            self.source_config.config_id, self.source_config
        )
        if request.source_config != expected_source:
            raise ValueError("Six-matrix response request source configuration identity differs")
        if request.family_member not in self.source_config.anisotropic_model.family_members:
            raise ValueError("Six-matrix response request family member is outside the frozen family")
        if request.numerical_view not in {
            self.source_config.primary_view,
            self.source_config.secondary_view,
        }:
            raise ValueError("Six-matrix response request numerical view is not installed")
        if request.integration_steps > self.source_config.maximum_total_integration_steps:
            raise ValueError("Six-matrix response request exceeds the source integration ceiling")

    def run(
        self,
        request: SixMatrixResponseEpisodeRequest,
        *,
        checkpoint: SixMatrixResponseCheckpoint | None = None,
        stop_after_steps: int | None = None,
        initial_state_override: SixMatrixState | None = None,
    ) -> SixMatrixResponseEpisodeResult:
        gradient_cache_state = BAOABGradientCache()
        self._validate_request(request)
        if checkpoint is not None and initial_state_override is not None:
            raise ValueError("Six-matrix response resume cannot also override initial state")
        accepted, disposition, action_reasons = _resolve_action(request.action)
        ledger_terminal = (
            request.action.requested_start
            if disposition is SixMatrixResponseActionDisposition.REFUSED
            else accepted
        )
        base_ledger = SixMatrixResponseNativeActionLedger(
            action_id=request.action.action_id,
            disposition=disposition,
            requested=request.action.requested_target,
            accepted=accepted,
            applied=ledger_terminal,
            realized=ledger_terminal,
            reason_codes=action_reasons,
        )
        rng, rng_receipt = derive_rng_stream(
            seed_root_id=request.seed_root_id,
            purpose_id=request.rng_purpose_id,
            stream_index=request.rng_stream_index,
            derivation_rule_id=request.numerical_view.stream_derivation_rule_id,
            scientific_seed_sha256=request.scientific_seed_sha256,
        )
        if disposition is SixMatrixResponseActionDisposition.REFUSED:
            return self._result(
                request=request,
                terminal=SixMatrixResponseEpisodeTerminal.ACTION_REFUSED,
                completed_steps=0,
                ledger=base_ledger,
                rng_stream=rng_receipt,
                checkpoint=None,
                receivers=(),
                reasons=action_reasons,
            )
        if checkpoint is None:
            start = request.action.requested_start
            state = initial_state_override or ideal_state(
                q=request.q,
                alpha_tilde_x=float(start.alpha_tilde_x),
                alpha_tilde_y=float(start.alpha_tilde_y),
                constitution=request.initial_constitution.value,
            )
            if (
                state.q != request.q
                or state.step_index != 0
                or not state.finite
                or _as_decimal(state.alpha_tilde_x) != start.alpha_tilde_x
                or _as_decimal(state.alpha_tilde_y) != start.alpha_tilde_y
            ):
                return self._result(
                    request=request,
                    terminal=SixMatrixResponseEpisodeTerminal.INVALID_INITIAL_STATE,
                    completed_steps=0,
                    ledger=base_ledger,
                    rng_stream=rng_receipt,
                    checkpoint=None,
                    receivers=(),
                    reasons=("invalid-initial-state",),
                )
        else:
            if (
                checkpoint.request != ObjectIdentity.from_record(request.request_id, request)
                or checkpoint.total_steps != request.integration_steps
                or checkpoint.rng_stream != rng_receipt
            ):
                return self._result(
                    request=request,
                    terminal=SixMatrixResponseEpisodeTerminal.INVALID_CHECKPOINT,
                    completed_steps=0,
                    ledger=base_ledger,
                    rng_stream=rng_receipt,
                    checkpoint=None,
                    receivers=(),
                    reasons=("checkpoint-request-identity-mismatch",),
                )
            state, rng = state_from_checkpoint(checkpoint)
        remaining = request.integration_steps - state.step_index
        advance = remaining if stop_after_steps is None else min(remaining, stop_after_steps)
        if advance < 1 and remaining:
            raise ValueError("Six-matrix response stop_after_steps must be positive")
        receivers: list[SixMatrixResponseFastReceiver] = list(
            checkpoint.receivers if checkpoint is not None else ()
        )
        start = request.action.requested_start
        try:
            for _ in range(advance):
                next_step = state.step_index + 1
                fraction = Decimal(min(next_step, request.action.ramp_steps)) / Decimal(
                    request.action.ramp_steps
                )
                next_x = start.alpha_tilde_x + fraction * (
                    accepted.alpha_tilde_x - start.alpha_tilde_x
                )
                next_y = start.alpha_tilde_y + fraction * (
                    accepted.alpha_tilde_y - start.alpha_tilde_y
                )
                state = baoab_step(
                    state,
                    member=request.family_member,
                    numerical_view=request.numerical_view,
                    next_alpha_tilde_x=float(next_x),
                    next_alpha_tilde_y=float(next_y),
                    rng=rng,
                    gradient_cache=gradient_cache_state,
                )
                if not state.finite:
                    raise FloatingPointError("Six-matrix response BAOAB state became nonfinite")
                if (
                    state.step_index % request.receiver_cadence_steps == 0
                    or state.step_index == request.integration_steps
                ):
                    receivers.append(fast_receiver(state=state, member=request.family_member))
        except (FloatingPointError, OverflowError):
            return self._result(
                request=request,
                terminal=SixMatrixResponseEpisodeTerminal.NONFINITE_STATE,
                completed_steps=state.step_index,
                ledger=base_ledger,
                rng_stream=rng_receipt,
                checkpoint=None,
                receivers=tuple(receivers),
                reasons=("nonfinite-state",),
            )
        stored = checkpoint_from_state(
            request=request,
            state=state,
            rng=rng,
            rng_stream=rng_receipt,
            receivers=tuple(receivers),
        )
        realized = SixMatrixResponseScaledCouplings(
            _as_decimal(state.alpha_tilde_x), _as_decimal(state.alpha_tilde_y)
        )
        applied_fraction = Decimal(min(state.step_index, request.action.ramp_steps)) / Decimal(
            request.action.ramp_steps
        )
        applied = SixMatrixResponseScaledCouplings(
            start.alpha_tilde_x + applied_fraction * (accepted.alpha_tilde_x - start.alpha_tilde_x),
            start.alpha_tilde_y + applied_fraction * (accepted.alpha_tilde_y - start.alpha_tilde_y),
        )
        ledger = SixMatrixResponseNativeActionLedger(
            action_id=request.action.action_id,
            disposition=disposition,
            requested=request.action.requested_target,
            accepted=accepted,
            applied=applied,
            realized=realized,
            reason_codes=action_reasons,
        )
        terminal = (
            SixMatrixResponseEpisodeTerminal.COMPLETED
            if state.step_index == request.integration_steps
            else SixMatrixResponseEpisodeTerminal.PARTIAL_CHECKPOINT
        )
        return self._result(
            request=request,
            terminal=terminal,
            completed_steps=state.step_index,
            ledger=ledger,
            rng_stream=rng_receipt,
            checkpoint=stored,
            receivers=tuple(receivers),
            reasons=(),
        )

    @staticmethod
    def _result(
        *,
        request: SixMatrixResponseEpisodeRequest,
        terminal: SixMatrixResponseEpisodeTerminal,
        completed_steps: int,
        ledger: SixMatrixResponseNativeActionLedger,
        rng_stream: SixMatrixResponseRNGStreamReceipt,
        checkpoint: SixMatrixResponseCheckpoint | None,
        receivers: tuple[SixMatrixResponseFastReceiver, ...],
        reasons: tuple[str, ...],
    ) -> SixMatrixResponseEpisodeResult:
        return SixMatrixResponseEpisodeResult(
            episode_id=f"episode.{request.request_id}",
            task_id=request.task_id,
            output_id=request.output_id,
            request=ObjectIdentity.from_record(request.request_id, request),
            terminal=terminal,
            completed_steps=completed_steps,
            total_steps=request.integration_steps,
            action_ledger=ledger,
            rng_stream=rng_stream,
            checkpoint=checkpoint,
            receivers=receivers,
            reason_codes=tuple(sorted(reasons)),
            evidence_ceiling=request.evidence_ceiling,
            outcome_access=request.outcome_access,
            visibility_ceiling=request.visibility_ceiling,
            scientific_verdict_ids=(),
            grants_authority=False,
        )


__all__ = [
    'SixMatrixResponseActionDisposition',
    'SixMatrixResponseCheckpoint',
    'SixMatrixResponseConstitution',
    'SixMatrixResponseEpisodeRequest',
    'SixMatrixResponseEpisodeResult',
    'SixMatrixResponseEpisodeTerminal',
    'SixMatrixResponseFastReceiver',
    'SixMatrixResponseNativeActionLedger',
    'SixMatrixResponseNativeActionRequest',
    'SixMatrixResponseRNGStreamReceipt',
    'SixMatrixResponseScaledCouplings',
    'SixMatrixResponseSixMatrixEpisodeEngine',
    'baoab_step_and_capture_hermitian_noise',
    'baoab_step',
    'baoab_step_with_hermitian_noise',
    'checkpoint_from_state',
    'derive_rng_stream',
    'fast_receiver',
    'hermitian_noise',
    'state_from_checkpoint',
]
