"""Exact checkpoint, innovation-coupling and coordinate mechanics for six-matrix shooting committor."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
from hashlib import sha256
from math import sqrt
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_unique_ids,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView
from .model import ComplexArray, SixMatrixState, hermiticity_residual, ideal_state
from .simulation import BAOABGradientCache, SixMatrixResponseFastReceiver, SixMatrixResponseRNGStreamReceipt, baoab_step_and_capture_hermitian_noise, derive_rng_stream, fast_receiver
from .scientific_inputs import SixMatrixResponseScientificSeedInput, require_six_matrix_scientific_seed_input
from .spectral import SixMatrixResponseLaplacianSpectrum, spectral_receiver

BRANCH_DERIVATION_RULE_ID = "matrix-shooting.branch-scientific-input"
BRIDGE_DERIVATION_RULE_ID = "matrix-shooting.bridge-scientific-input"


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("six-matrix shooting committor cannot encode a nonfinite value")
    return Decimal(repr(float(value)))


def _canonical_rng_state(rng: np.random.Generator) -> str:
    return json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":"))


def _combined_checkpoint_sha256(
    *,
    parent_rollout_id: str,
    parent_step: int,
    alpha_tilde_x: Decimal,
    alpha_tilde_y: Decimal,
    positions: bytes,
    momenta: bytes,
    rng_state_json: str,
) -> str:
    digest = sha256()
    for value in (
        parent_rollout_id.encode("utf-8"),
        str(parent_step).encode("ascii"),
        str(alpha_tilde_x).encode("ascii"),
        str(alpha_tilde_y).encode("ascii"),
        positions,
        momenta,
        rng_state_json.encode("utf-8"),
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SixMatrixResponseShootingCommittorCheckpoint(CanonicalRecord):
    """One complete selected-event phase-space checkpoint and replay prefix."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-shooting-committor-checkpoint'

    checkpoint_id: str
    parent_rollout_id: str
    parent_report_sha256: str
    q: int
    parent_step: int
    parent_time: Decimal
    alpha_tilde_x: Decimal
    alpha_tilde_y: Decimal
    positions_base64: str
    momenta_base64: str
    positions_sha256: str
    momenta_sha256: str
    rng_state_json: str
    rng_state_sha256: str
    combined_state_sha256: str
    fast_receiver_prefix: tuple[SixMatrixResponseFastReceiver, ...]
    spectral_receiver_prefix: tuple[SixMatrixResponseLaplacianSpectrum, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.checkpoint_id, field_name="checkpoint_id")
        validate_stable_id(self.parent_rollout_id, field_name="parent_rollout_id")
        validate_sha256(self.parent_report_sha256, field_name="parent_report_sha256")
        if self.q != 2 or self.parent_step < 1:
            raise ValueError("six-matrix shooting committor checkpoint must be a positive q=2 parent step")
        validate_decimal(self.parent_time, field_name="parent_time", minimum=Decimal(0))
        for name in ("alpha_tilde_x", "alpha_tilde_y"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        expected_bytes = 2 * 3 * (self.q**2) ** 2 * 16
        decoded: list[bytes] = []
        for data_name, hash_name in (
            ("positions_base64", "positions_sha256"),
            ("momenta_base64", "momenta_sha256"),
        ):
            try:
                raw = base64.b64decode(getattr(self, data_name), validate=True)
            except ValueError as error:
                raise ValueError(f"six-matrix shooting committor {data_name} is not canonical base64") from error
            if len(raw) != expected_bytes or base64.b64encode(raw).decode("ascii") != getattr(
                self, data_name
            ):
                raise ValueError(f"six-matrix shooting committor {data_name} has another byte geometry")
            validate_sha256(getattr(self, hash_name), field_name=hash_name)
            if sha256(raw).hexdigest() != getattr(self, hash_name):
                raise ValueError(f"six-matrix shooting committor {data_name} hash differs")
            decoded.append(raw)
        validate_sha256(self.rng_state_sha256, field_name="rng_state_sha256")
        if sha256(self.rng_state_json.encode("utf-8")).hexdigest() != self.rng_state_sha256:
            raise ValueError("six-matrix shooting committor checkpoint RNG-state hash differs")
        try:
            rng_document = json.loads(self.rng_state_json)
        except json.JSONDecodeError as error:
            raise ValueError("six-matrix shooting committor checkpoint RNG state is not JSON") from error
        if json.dumps(rng_document, sort_keys=True, separators=(",", ":")) != self.rng_state_json:
            raise ValueError("six-matrix shooting committor checkpoint RNG state is not canonical")
        validate_sha256(self.combined_state_sha256, field_name="combined_state_sha256")
        expected = _combined_checkpoint_sha256(
            parent_rollout_id=self.parent_rollout_id,
            parent_step=self.parent_step,
            alpha_tilde_x=self.alpha_tilde_x,
            alpha_tilde_y=self.alpha_tilde_y,
            positions=decoded[0],
            momenta=decoded[1],
            rng_state_json=self.rng_state_json,
        )
        if expected != self.combined_state_sha256:
            raise ValueError("six-matrix shooting committor combined checkpoint identity differs")
        fast_steps = tuple(value.sample_step for value in self.fast_receiver_prefix)
        if fast_steps != tuple(sorted(set(fast_steps))) or any(
            step > self.parent_step for step in fast_steps
        ):
            raise ValueError("six-matrix shooting committor checkpoint fast-receiver prefix differs")
        require_unique_ids(
            self.spectral_receiver_prefix,
            attribute="spectrum_id",
            field_name="spectral_receiver_prefix",
        )


@dataclass(frozen=True, slots=True)
class SixMatrixResponseShootingCommittorReplayData:
    """In-memory exact replay data; scientific persistence uses canonical records."""

    checkpoints: tuple[SixMatrixResponseShootingCommittorCheckpoint, ...]
    checkpoint_states: tuple[SixMatrixState, ...]
    coarse_noises: tuple[ComplexArray, ...]
    final_state: SixMatrixState
    final_rng_state_json: str
    rng_stream: SixMatrixResponseRNGStreamReceipt
    final_rollout_state_sha256: str


def checkpoint_from_replay_state(
    *,
    parent_rollout_id: str,
    parent_report_sha256: str,
    parent_time: Decimal,
    state: SixMatrixState,
    rng: np.random.Generator,
    fast_receiver_prefix: tuple[SixMatrixResponseFastReceiver, ...],
    spectral_receiver_prefix: tuple[SixMatrixResponseLaplacianSpectrum, ...],
) -> SixMatrixResponseShootingCommittorCheckpoint:
    positions = state.positions.tobytes(order="C")
    momenta = state.momenta.tobytes(order="C")
    rng_state_json = _canonical_rng_state(rng)
    alpha_x = _decimal(state.alpha_tilde_x)
    alpha_y = _decimal(state.alpha_tilde_y)
    return SixMatrixResponseShootingCommittorCheckpoint(
        checkpoint_id=f"matrix-shooting.checkpoint.step-{state.step_index:04d}",
        parent_rollout_id=parent_rollout_id,
        parent_report_sha256=parent_report_sha256,
        q=state.q,
        parent_step=state.step_index,
        parent_time=parent_time,
        alpha_tilde_x=alpha_x,
        alpha_tilde_y=alpha_y,
        positions_base64=base64.b64encode(positions).decode("ascii"),
        momenta_base64=base64.b64encode(momenta).decode("ascii"),
        positions_sha256=sha256(positions).hexdigest(),
        momenta_sha256=sha256(momenta).hexdigest(),
        rng_state_json=rng_state_json,
        rng_state_sha256=sha256(rng_state_json.encode("utf-8")).hexdigest(),
        combined_state_sha256=_combined_checkpoint_sha256(
            parent_rollout_id=parent_rollout_id,
            parent_step=state.step_index,
            alpha_tilde_x=alpha_x,
            alpha_tilde_y=alpha_y,
            positions=positions,
            momenta=momenta,
            rng_state_json=rng_state_json,
        ),
        fast_receiver_prefix=fast_receiver_prefix,
        spectral_receiver_prefix=spectral_receiver_prefix,
    )


def state_from_shooting_checkpoint(
    checkpoint: SixMatrixResponseShootingCommittorCheckpoint,
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
        step_index=checkpoint.parent_step,
        alpha_tilde_x=float(checkpoint.alpha_tilde_x),
        alpha_tilde_y=float(checkpoint.alpha_tilde_y),
    )
    rng = np.random.Generator(np.random.PCG64DXSM())
    rng.bit_generator.state = json.loads(checkpoint.rng_state_json)
    return state, rng


def anisotropic_feasibility_rollout_state_sha256(
    *, state: SixMatrixState, target_x: Decimal, target_y: Decimal
) -> str:
    """Reproduce the exact state digest used by the original anisotropic feasibility report."""

    digest = sha256()
    digest.update(state.positions.tobytes(order="C"))
    digest.update(state.momenta.tobytes(order="C"))
    digest.update(str(state.step_index).encode("ascii"))
    digest.update(str(target_x).encode("ascii"))
    digest.update(str(target_y).encode("ascii"))
    return digest.hexdigest()


def _co_anneal_couplings(
    *, step: int, ramp_steps: int, target_x: Decimal, target_y: Decimal
) -> tuple[float, float]:
    # Preserve the original anisotropic feasibility preparation arithmetic: its ramp fraction was
    # binary float before multiplication by the decoded Decimal target.
    fraction = min(step, ramp_steps) / ramp_steps
    return float(target_x) * fraction, float(target_y) * fraction


def replay_selected_co_anneal_event(
    *,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    parent_rollout_id: str,
    parent_report_sha256: str,
    target_x: Decimal,
    target_y: Decimal,
    seed_index: int,
    scientific_seed_sha256: str,
    checkpoint_steps: tuple[int, ...],
    total_steps: int = 1024,
    ramp_steps: int = 256,
    receiver_cadence_steps: int = 16,
) -> SixMatrixResponseShootingCommittorReplayData:
    """Replay the selected anisotropic feasibility path once and retain every coarse innovation."""

    gradient_cache_state = BAOABGradientCache()
    if (
        checkpoint_steps != tuple(sorted(set(checkpoint_steps)))
        or checkpoint_steps[-1] > total_steps
    ):
        raise ValueError("six-matrix shooting committor checkpoint roster differs")
    purpose_id = f"purpose.{parent_rollout_id}"
    rng, receipt = derive_rng_stream(
        seed_root_id="seed.matrix-anisotropic-feasibility",
        purpose_id=purpose_id,
        stream_index=seed_index,
        derivation_rule_id=numerical_view.stream_derivation_rule_id,
        scientific_seed_sha256=scientific_seed_sha256,
    )
    state = ideal_state(q=2, alpha_tilde_x=0.0, alpha_tilde_y=0.0, constitution="00")
    noises: list[ComplexArray] = []
    sample_states: list[SixMatrixState] = []
    states: list[SixMatrixState] = []
    checkpoint_rng_states: list[str] = []
    checkpoint_set = set(checkpoint_steps)
    for step in range(1, total_steps + 1):
        alpha_x, alpha_y = _co_anneal_couplings(
            step=step,
            ramp_steps=ramp_steps,
            target_x=target_x,
            target_y=target_y,
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
        # The original anisotropic feasibility denominator includes its original measurement
        # cadence.  On the pinned numerical stack, receiver BLAS operations
        # can alter later floating-point status even though they do not mutate
        # the state arrays, so replay the historical calls at their exact
        # locations before constructing the denser follow-up view post hoc.
        if step > 768 and step % 16 == 0:
            fast_receiver(state=state, member=member)
        if step > 768 and step % 256 == 0:
            spectral_receiver(
                receiver_prefix=f"spectrum.{parent_rollout_id}.{step}",
                q=2,
                positions=state.positions,
            )
        if step % receiver_cadence_steps == 0:
            sample_states.append(state)
        if step in checkpoint_set:
            states.append(state)
            checkpoint_rng_states.append(_canonical_rng_state(rng))
    # Receiver evaluation is deliberately post-replay.  Some linked BLAS/LAPACK
    # builds alter floating-point status during eigensolves; inserting spectra
    # into the historical integration loop changes later bytes despite no array
    # mutation.  Retaining immutable sample states preserves both exact dynamics
    # and a complete auditable receiver prefix.
    fast = [fast_receiver(state=value, member=member) for value in sample_states]
    spectra_by_step = {
        value.step_index: spectral_receiver(
            receiver_prefix=f"spectrum.{parent_rollout_id}.{value.step_index}",
            q=2,
            positions=value.positions,
        )
        for value in sample_states
    }
    checkpoints = []
    for checkpoint_state, rng_state_json in zip(states, checkpoint_rng_states, strict=True):
        checkpoint_rng = np.random.Generator(np.random.PCG64DXSM())
        checkpoint_rng.bit_generator.state = json.loads(rng_state_json)
        prefix_fast = tuple(
            value for value in fast if value.sample_step <= checkpoint_state.step_index
        )
        prefix_spectra = tuple(
            spectrum
            for sample_step in sorted(spectra_by_step)
            if sample_step <= checkpoint_state.step_index
            for spectrum in spectra_by_step[sample_step]
        )
        checkpoints.append(
            checkpoint_from_replay_state(
                parent_rollout_id=parent_rollout_id,
                parent_report_sha256=parent_report_sha256,
                parent_time=Decimal(checkpoint_state.step_index) * numerical_view.timestep,
                state=checkpoint_state,
                rng=checkpoint_rng,
                fast_receiver_prefix=prefix_fast,
                spectral_receiver_prefix=prefix_spectra,
            )
        )
    return SixMatrixResponseShootingCommittorReplayData(
        checkpoints=tuple(checkpoints),
        checkpoint_states=tuple(states),
        coarse_noises=tuple(noises),
        final_state=state,
        final_rng_state_json=_canonical_rng_state(rng),
        rng_stream=receipt,
        final_rollout_state_sha256=anisotropic_feasibility_rollout_state_sha256(
            state=state,
            target_x=target_x,
            target_y=target_y,
        ),
    )


def derive_branch_rng(
    *, checkpoint_combined_sha256: str, branch_index: int,
    scientific_seed_input: SixMatrixResponseScientificSeedInput | None = None,
) -> tuple[np.random.Generator, str]:
    validate_sha256(
        checkpoint_combined_sha256,
        field_name="checkpoint_combined_sha256",
    )
    if not 0 <= branch_index < 64:
        raise ValueError("six-matrix shooting committor branch index must be in 0..63")
    scientific_seed = require_six_matrix_scientific_seed_input(
        scientific_seed_input, scientific_role="shooting-branch",
        current_root_id=f"checkpoint-sha256.{checkpoint_combined_sha256}",
        current_context_sha256=checkpoint_combined_sha256, stream_index=branch_index,
    )
    digest = bytes.fromhex(scientific_seed.full_seed_sha256)
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:16], "big")))
    return rng, digest.hex()


def derive_bridge_rng(
    *, parent_rollout_id: str, coarse_step_index: int, current_context_sha256: str,
    scientific_seed_input: SixMatrixResponseScientificSeedInput | None = None,
) -> tuple[np.random.Generator, str]:
    validate_stable_id(parent_rollout_id, field_name="parent_rollout_id")
    if coarse_step_index < 1:
        raise ValueError("six-matrix shooting committor bridge step must be positive")
    scientific_seed = require_six_matrix_scientific_seed_input(
        scientific_seed_input, scientific_role="shooting-bridge",
        current_root_id=parent_rollout_id,
        current_context_sha256=current_context_sha256, stream_index=coarse_step_index,
    )
    digest = bytes.fromhex(scientific_seed.full_seed_sha256)
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:16], "big")))
    return rng, digest.hex()


def brownian_bridge_split(
    *, coarse_noise: ComplexArray, bridge_noise: ComplexArray, half_decay: float
) -> tuple[ComplexArray, ComplexArray]:
    """Conditionally split one coarse standardized OU innovation into two halves."""

    left = np.asarray(coarse_noise)
    bridge = np.asarray(bridge_noise)
    if (
        left.shape != bridge.shape
        or left.dtype != np.dtype("complex128")
        or bridge.dtype != np.dtype("complex128")
    ):
        raise ValueError("six-matrix shooting committor bridge innovations have another shape or dtype")
    if not (0.0 < half_decay < 1.0):
        raise ValueError("six-matrix shooting committor half decay must lie strictly between zero and one")
    if not np.isfinite(left).all() or not np.isfinite(bridge).all():
        raise ValueError("six-matrix shooting committor bridge innovations must be finite")
    if max(hermiticity_residual(left), hermiticity_residual(bridge)) > 1e-12:
        raise ValueError("six-matrix shooting committor bridge innovations must be Hermitian")
    normalizer = sqrt(1.0 + half_decay**2)
    first = np.asarray((half_decay * left + bridge) / normalizer, dtype="<c16")
    second = np.asarray((left - half_decay * bridge) / normalizer, dtype="<c16")
    return first, second


@lru_cache(maxsize=16)
def hermitian_basis(n: int) -> ComplexArray:
    """Return the fixed Hilbert--Schmidt-orthonormal Hermitian basis."""

    if n < 1:
        raise ValueError("Hermitian basis dimension must be positive")
    values: list[ComplexArray] = []
    for index in range(n):
        matrix = np.zeros((n, n), dtype="<c16")
        matrix[index, index] = 1.0
        values.append(matrix)
    symmetric_values: list[ComplexArray] = []
    antisymmetric_values: list[ComplexArray] = []
    for row in range(n):
        for column in range(row + 1, n):
            symmetric = np.zeros((n, n), dtype="<c16")
            symmetric[row, column] = symmetric[column, row] = 1.0 / sqrt(2.0)
            symmetric_values.append(symmetric)
            antisymmetric = np.zeros((n, n), dtype="<c16")
            antisymmetric[row, column] = 1.0j / sqrt(2.0)
            antisymmetric[column, row] = -1.0j / sqrt(2.0)
            antisymmetric_values.append(antisymmetric)
    values.extend(symmetric_values)
    values.extend(antisymmetric_values)
    value = np.ascontiguousarray(np.stack(values), dtype="<c16")
    return np.frombuffer(value.tobytes(), dtype="<c16").reshape(value.shape)


@lru_cache(maxsize=16)
def traceless_hermitian_basis(n: int) -> ComplexArray:
    """Return a fixed HS-orthonormal basis of traceless Hermitian generators."""

    basis = hermitian_basis(n)
    off_diagonal = basis[n:]
    diagonal: list[ComplexArray] = []
    for k in range(1, n):
        matrix = np.zeros((n, n), dtype="<c16")
        matrix[np.arange(k), np.arange(k)] = 1.0
        matrix[k, k] = -float(k)
        matrix /= sqrt(k * (k + 1.0))
        diagonal.append(matrix)
    value = np.ascontiguousarray(np.concatenate((np.stack(diagonal), off_diagonal)), dtype="<c16")
    return np.frombuffer(value.tobytes(), dtype="<c16").reshape(value.shape)


def encode_hermitian_matrices(value: ComplexArray) -> np.ndarray:
    matrices = np.asarray(value)
    if matrices.ndim < 2 or matrices.shape[-1] != matrices.shape[-2]:
        raise ValueError("six-matrix shooting committor Hermitian encoding requires square matrices")
    if matrices.dtype != np.dtype("complex128") or not np.isfinite(matrices).all():
        raise ValueError("six-matrix shooting committor Hermitian encoding requires finite complex128")
    if hermiticity_residual(matrices) > 1e-12:
        raise ValueError("six-matrix shooting committor Hermitian encoding requires Hermitian matrices")
    n = matrices.shape[-1]
    basis = hermitian_basis(n)
    flat = matrices.reshape((-1, n, n))
    encoded = np.asarray(
        [[float(np.vdot(generator, matrix).real) for generator in basis] for matrix in flat],
        dtype=np.float64,
    )
    return encoded.reshape(matrices.shape[:-2] + (n * n,))


def decode_hermitian_matrices(value: np.ndarray, *, n: int) -> ComplexArray:
    coordinates = np.asarray(value)
    if coordinates.ndim < 1 or coordinates.shape[-1] != n * n:
        raise ValueError("six-matrix shooting committor Hermitian coordinates have another shape")
    if coordinates.dtype != np.dtype("float64") or not np.isfinite(coordinates).all():
        raise ValueError("six-matrix shooting committor Hermitian coordinates must be finite float64")
    decoded = np.tensordot(coordinates, hermitian_basis(n), axes=([-1], [0]))
    return np.ascontiguousarray(decoded, dtype="<c16")


__all__ = [
    "BRANCH_DERIVATION_RULE_ID",
    "BRIDGE_DERIVATION_RULE_ID",
    'SixMatrixResponseShootingCommittorCheckpoint',
    'SixMatrixResponseShootingCommittorReplayData',
    'brownian_bridge_split',
    'anisotropic_feasibility_rollout_state_sha256',
    'checkpoint_from_replay_state',
    'decode_hermitian_matrices',
    'derive_branch_rng',
    'derive_bridge_rng',
    'encode_hermitian_matrices',
    'hermitian_basis',
    'replay_selected_co_anneal_event',
    'state_from_shooting_checkpoint',
    'traceless_hermitian_basis',
]
