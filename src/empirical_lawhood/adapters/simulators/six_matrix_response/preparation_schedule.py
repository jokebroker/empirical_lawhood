"""Exact six-matrix preparation scheduling preparation schedules, tapes, and trajectory execution.

This module owns simulator-native native measurements and custody only.  It does
not evaluate response factors, select a schedule, qualify a law, or grant
execution authority.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from io import BytesIO
from math import prod
from typing import Any, ClassVar

import numpy as np

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.response_experiment import ResponseAcquisitionView

from .artifact_inventory import MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_TRAJECTORY_DECODED_DATASET_BYTES, MATRIX_RESPONSE_PREPARATION_SCHEDULING_MEASUREMENT_HDF5_SCHEMA, SixMatrixResponsePreparationWindowSchedulingTrajectoryArtifactReservation
from .contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView, SixMatrixResponseSixMatrixSourceConfig
from .model import ComplexArray, SixMatrixState, ideal_state
from .scientific_inputs import SixMatrixResponseScientificSeedInput, require_six_matrix_scientific_seed_input
from .simulation import BAOABGradientCache, SixMatrixResponseScaledCouplings, baoab_step_with_hermitian_noise, derive_rng_stream, hermitian_noise

MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_CONFIG_BYTES = 64 * 1024
MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_REQUEST_BYTES = 16 * 1024
MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_RESULT_BYTES = 64 * 1024
MATRIX_RESPONSE_PREPARATION_SCHEDULING_PRIMARY_TIMESTEP = Decimal("0.001")
MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_X = Decimal("0.6666666666666666666666666667")
MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_Y = Decimal("7.333333333333333333333333334")
MATRIX_RESPONSE_PREPARATION_SCHEDULING_POST_ARRIVAL_STEPS = 768
MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS = (
    "m64-r075",
    "m64-r100",
    "m64-r125",
    "p00-r075",
    "p00-r100",
    "p00-r125",
    "p64-r075",
    "p64-r100",
    "p64-r125",
)

_SCHEDULE_TABLE: dict[str, tuple[int, int, int, int, int]] = {
    "m64-r075": (-64, 64, 0, 192, 320),
    "m64-r100": (-64, 64, 0, 256, 320),
    "m64-r125": (-64, 64, 0, 320, 320),
    "p00-r075": (0, 0, 0, 192, 256),
    "p00-r100": (0, 0, 0, 256, 256),
    "p00-r125": (0, 0, 0, 320, 320),
    "p64-r075": (64, 0, 64, 192, 256),
    "p64-r100": (64, 0, 64, 256, 320),
    "p64-r125": (64, 0, 64, 320, 384),
}


def six_matrix_response_preparation_window_scheduling_trajectory_decoded_dataset_bytes(
    schedule_id: str,
    *,
    view_step_multiplier: int,
) -> int:
    """Exact uncompressed decoded HDF5 dataset bytes for one trajectory task."""

    if schedule_id not in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS or view_step_multiplier != 1:
        raise ValueError("six-matrix preparation scheduling decoded-byte coordinate differs")
    arrival = _SCHEDULE_TABLE[schedule_id][-1] * view_step_multiplier
    states = arrival + MATRIX_RESPONSE_PREPARATION_SCHEDULING_POST_ARRIVAL_STEPS * view_step_multiplier + 1
    matrix_elements = 2 * 3 * 4 * 4
    tape_steps = (384 + MATRIX_RESPONSE_PREPARATION_SCHEDULING_POST_ARRIVAL_STEPS) * view_step_multiplier
    tape_bytes = tape_steps * matrix_elements * 16
    phase_path_bytes = states * 2 * matrix_elements * 16
    alpha_and_local_step_bytes = states * (2 * 8 + 8)
    return tape_bytes + phase_path_bytes + alpha_and_local_step_bytes


_PARENT_TERMINAL_SCHEMAS = tuple(
    sorted(
        (
            'empirical-lawhood/methods/matrix-response-study/matrix-response-shooting-committor-terminal-report',
            # Current predecessor formats are required here. Historical terminal
            # bytes require an authenticated external export/import disposition.
            "empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-joint-terminal-report",
            'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-terminal-report',
            'empirical-lawhood/methods/matrix-response-study/matrix-response-causal-intersection-residence-terminal-report',
            'empirical-lawhood/methods/matrix-response-study/matrix-response-numerical-qualification-qualification-report',
            'empirical-lawhood/methods/matrix-response-study/matrix-response-anisotropic-feasibility-qualification-report',
        )
    )
)


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("six-matrix preparation scheduling cannot encode a nonfinite numerical value")
    return Decimal(repr(float(value)))


def _array_sha256(value: np.ndarray, *, dtype: str) -> str:
    array = np.ascontiguousarray(value, dtype=dtype)
    digest = sha256()
    digest.update(len(array.shape).to_bytes(2, "big"))
    for dimension in array.shape:
        digest.update(dimension.to_bytes(8, "big"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _rng_state_sha256(rng: np.random.Generator) -> str:
    payload = json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _root_seed_sha256(
    *, seed_root_id: str, root_block_id: str, block_index: int,
    current_context_sha256: str,
    scientific_seed_input: SixMatrixResponseScientificSeedInput | None,
) -> str:
    validate_stable_id(seed_root_id, field_name="seed_root_id")
    return require_six_matrix_scientific_seed_input(
        scientific_seed_input, scientific_role="preparation-root-identity",
        current_root_id=root_block_id, current_context_sha256=current_context_sha256,
        stream_index=block_index,
    ).full_seed_sha256


def _require_preparation_scientific_inputs(
    inputs: tuple[SixMatrixResponseScientificSeedInput, ...], *,
    root_block_id: str, seed_root_id: str, block_index: int,
    current_context_sha256: str,
) -> tuple[SixMatrixResponseScientificSeedInput, ...]:
    roles = ("preparation-root-identity", "preparation-tape", "observation-tape")
    if not isinstance(inputs, tuple) or len(inputs) != 3:
        raise ValueError("preparation root requires all three explicit numeric commitments and their external export custody")
    for ordinal, role in enumerate(roles):
        require_six_matrix_scientific_seed_input(
            inputs[ordinal], scientific_role=role,
            current_root_id=root_block_id if ordinal == 0 else seed_root_id,
            current_context_sha256=current_context_sha256, stream_index=block_index,
        )
    if len({row.source_original_context_sha256 for row in inputs}) != 1 or len({row.original_source for row in inputs}) != 1 or len({row.export_receipt for row in inputs}) != 1:
        raise ValueError("preparation root inputs require one complete original context/source/export census")
    if inputs[1].full_seed_sha256 == inputs[2].full_seed_sha256:
        raise ValueError("preparation and observation require distinct original numeric streams")
    return inputs


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingPreparationWord(CanonicalRecord):
    """One word in the closed nine-word preparation chart."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-preparation-word'

    schedule_id: str
    delta_start_steps: int
    x_start_step: int
    y_start_step: int
    x_duration_steps: int
    y_duration_steps: int
    final_arrival_step: int
    target_alpha_tilde_x: Decimal
    target_alpha_tilde_y: Decimal
    central_control: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.schedule_id, field_name="schedule_id")
        expected = _SCHEDULE_TABLE.get(self.schedule_id)
        if expected is None:
            raise ValueError("six-matrix preparation scheduling preparation word is outside the closed chart")
        observed = (
            self.delta_start_steps,
            self.x_start_step,
            self.y_start_step,
            self.y_duration_steps,
            self.final_arrival_step,
        )
        if observed != expected or self.x_duration_steps != 256:
            raise ValueError("six-matrix preparation scheduling preparation word differs from the exact chart")
        for name in ("target_alpha_tilde_x", "target_alpha_tilde_y"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if (
            self.target_alpha_tilde_x,
            self.target_alpha_tilde_y,
        ) != (MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_X, MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_Y):
            raise ValueError("six-matrix preparation scheduling preparation word has another final coordinate")
        if self.central_control != (self.schedule_id == "p00-r100"):
            raise ValueError("six-matrix preparation scheduling central schedule identity differs")
        if self.final_arrival_step != max(
            self.x_start_step + self.x_duration_steps,
            self.y_start_step + self.y_duration_steps,
        ):
            raise ValueError("six-matrix preparation scheduling arrival is not the first common-final step")


def six_matrix_response_preparation_window_scheduling_preparation_words() -> tuple[SixMatrixResponsePreparationWindowSchedulingPreparationWord, ...]:
    """Return the exact canonical chart in schedule-ID order."""

    words = []
    for schedule_id in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS:
        delta, x_start, y_start, y_duration, arrival = _SCHEDULE_TABLE[schedule_id]
        words.append(
            SixMatrixResponsePreparationWindowSchedulingPreparationWord(
                schedule_id=schedule_id,
                delta_start_steps=delta,
                x_start_step=x_start,
                y_start_step=y_start,
                x_duration_steps=256,
                y_duration_steps=y_duration,
                final_arrival_step=arrival,
                target_alpha_tilde_x=MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_X,
                target_alpha_tilde_y=MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_Y,
                central_control=schedule_id == "p00-r100",
            )
        )
    return tuple(words)


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingStudyConfig(CanonicalRecord):
    """Strict denominator and resource contract for direct preparation scheduling DEVELOPMENT only."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-study-config'

    config_id: str
    config_version: str
    plan_id: str
    source_config: ObjectIdentity
    parent_terminals: tuple[ObjectIdentity, ...]
    accepted_platform_receipt: ObjectIdentity
    family_member: SixMatrixResponseModelFamilyMember
    primary_view: SixMatrixResponseNumericalView
    primary_numerical_member_id: str
    q: int
    matrix_dimension_n: int
    target_alpha_tilde_x: Decimal
    target_alpha_tilde_y: Decimal
    schedules: tuple[SixMatrixResponsePreparationWindowSchedulingPreparationWord, ...]
    central_schedule_id: str
    post_arrival_steps: int
    receiver_cadence_steps: int
    spectral_receiver_cadence_steps: int
    rolling_history_samples: int
    first_eligible_post_arrival_step: int
    development_root_blocks: int
    development_primary_trajectory_count: int
    development_issue_id: str
    development_seed_namespace_id: str
    development_root_namespace_id: str
    preparation_tape_purpose_id: str
    observation_tape_purpose_id: str
    tape_derivation_rule_id: str
    splice_rule_id: str
    maximum_worker_processes: int
    preferred_wall_seconds: int
    hard_wall_seconds: int
    maximum_memory_bytes: int
    artifact_resource_scope_id: str
    maximum_trajectory_decoded_bytes: int
    maximum_trajectory_hdf5_bytes: int
    maximum_trajectory_result_bytes: int
    maximum_attempt_hdf5_bytes: int
    maximum_trajectory_integration_steps: int
    maximum_total_integration_steps: int
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "plan_id",
            "central_schedule_id",
            "primary_numerical_member_id",
            "development_issue_id",
            "development_seed_namespace_id",
            "development_root_namespace_id",
            "preparation_tape_purpose_id",
            "observation_tape_purpose_id",
            "tape_derivation_rule_id",
            "splice_rule_id",
            "artifact_resource_scope_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.config_version)
        if self.plan_id != "matrix-preparation-window-scheduling":
            raise ValueError("six-matrix preparation scheduling plan identity differs")
        if self.source_config.object_schema != SixMatrixResponseSixMatrixSourceConfig.SCHEMA:
            raise ValueError("six-matrix preparation scheduling source-config lineage differs")
        require_sorted_unique_ids(
            self.parent_terminals,
            attribute="object_schema",
            field_name="parent_terminals",
        )
        if (
            tuple(value.object_schema for value in self.parent_terminals)
            != _PARENT_TERMINAL_SCHEMAS
        ):
            raise ValueError("six-matrix preparation scheduling predecessor terminal roster differs")
        if self.accepted_platform_receipt.object_id in {
            value.object_id for value in self.parent_terminals
        }:
            raise ValueError("six-matrix preparation scheduling platform receipt impersonates a predecessor")
        if (
            self.family_member.member_id != "six-matrix-response.member.mass-0p5.cross-coupling-1"
            or self.family_member.mass_x != Decimal("0.5")
            or self.family_member.mass_y != Decimal("0.5")
            or self.family_member.cross_coupling_gamma != Decimal("1")
        ):
            raise ValueError("six-matrix preparation scheduling model member differs")
        if (
            self.primary_view.timestep != MATRIX_RESPONSE_PREPARATION_SCHEDULING_PRIMARY_TIMESTEP
            or self.primary_view.friction_gamma != Decimal("1")
            or self.primary_view.bath_temperature != Decimal("1")
            or self.primary_numerical_member_id != "matrix-preparation-scheduling.numerical.primary-dt-0p001"
        ):
            raise ValueError("six-matrix preparation scheduling primary numerical denominator differs")
        for name in ("target_alpha_tilde_x", "target_alpha_tilde_y"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if (
            self.q,
            self.matrix_dimension_n,
            self.target_alpha_tilde_x,
            self.target_alpha_tilde_y,
        ) != (
            2,
            4,
            MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_X,
            MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_Y,
        ):
            raise ValueError("six-matrix preparation scheduling q/final-coordinate denominator differs")
        require_sorted_unique_ids(
            self.schedules,
            attribute="schedule_id",
            field_name="schedules",
        )
        if self.schedules != six_matrix_response_preparation_window_scheduling_preparation_words():
            raise ValueError("six-matrix preparation scheduling study config lacks the exact nine-word chart")
        if (
            self.central_schedule_id,
            self.post_arrival_steps,
            self.receiver_cadence_steps,
            self.spectral_receiver_cadence_steps,
            self.rolling_history_samples,
            self.first_eligible_post_arrival_step,
            self.development_root_blocks,
            self.development_primary_trajectory_count,
        ) != (
            "p00-r100",
            768,
            16,
            256,
            16,
            256,
            32,
            288,
        ):
            raise ValueError("six-matrix preparation scheduling direct DEVELOPMENT design differs")
        if (
            not self.development_issue_id.startswith("matrix-preparation-scheduling.issue.development.")
            or not self.development_seed_namespace_id.startswith("matrix-preparation-scheduling.seed.development.")
            or not self.development_root_namespace_id.startswith("matrix-preparation-scheduling.root.development.")
        ):
            raise ValueError("six-matrix preparation scheduling development issue/seed/root namespace differs")
        if (
            self.preparation_tape_purpose_id,
            self.observation_tape_purpose_id,
            self.splice_rule_id,
        ) != (
            "matrix-preparation-scheduling.preparation-tape",
            "matrix-preparation-scheduling.observation-tape",
            "matrix-preparation-scheduling.independent-tape-splice",
        ):
            raise ValueError("six-matrix preparation scheduling common-noise/splice semantics differ")
        if (
            self.preparation_tape_purpose_id == self.observation_tape_purpose_id
            or self.tape_derivation_rule_id != self.primary_view.stream_derivation_rule_id
        ):
            raise ValueError("six-matrix preparation scheduling tape derivation/reuse contract differs")
        if (
            self.maximum_worker_processes != 8
            or self.preferred_wall_seconds != 14_400
            or self.hard_wall_seconds != 43_200
            or self.maximum_memory_bytes != 8 * 1024**3
            or self.artifact_resource_scope_id != "matrix-preparation-scheduling.development-per-trajectory-and-attempt"
            or self.maximum_trajectory_decoded_bytes
            != MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_TRAJECTORY_DECODED_DATASET_BYTES
            or self.maximum_trajectory_hdf5_bytes != 6 * 1024**2
            or self.maximum_trajectory_result_bytes != MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_RESULT_BYTES
            or self.maximum_attempt_hdf5_bytes != 288 * 6 * 1024**2
            or self.maximum_trajectory_integration_steps != 1_152
            or self.maximum_total_integration_steps != 309_248
        ):
            raise ValueError("six-matrix preparation scheduling DEVELOPMENT resource closure differs")
        if self.grants_authority:
            raise ValueError("six-matrix preparation scheduling study config cannot grant authority")


def build_six_matrix_response_preparation_window_scheduling_study_config(
    *,
    source_config: SixMatrixResponseSixMatrixSourceConfig,
    parent_terminals: tuple[ObjectIdentity, ...],
    accepted_platform_receipt: ObjectIdentity,
    development_issue_id: str,
    development_seed_namespace_id: str,
    development_root_namespace_id: str,
) -> SixMatrixResponsePreparationWindowSchedulingStudyConfig:
    """Construct the fixed direct-development design from authenticated lineage."""

    members = tuple(
        value
        for value in source_config.anisotropic_model.family_members
        if value.member_id == "six-matrix-response.member.mass-0p5.cross-coupling-1"
    )
    if len(members) != 1:
        raise ValueError("six-matrix preparation scheduling exact m0p5-g1 member is not uniquely installed")
    return SixMatrixResponsePreparationWindowSchedulingStudyConfig(
        config_id="matrix-preparation-scheduling.development-study",
        config_version="1.0.0",
        plan_id="matrix-preparation-window-scheduling",
        source_config=ObjectIdentity.from_record(source_config.config_id, source_config),
        parent_terminals=tuple(sorted(parent_terminals, key=lambda value: value.object_schema)),
        accepted_platform_receipt=accepted_platform_receipt,
        family_member=members[0],
        primary_view=source_config.primary_view,
        primary_numerical_member_id="matrix-preparation-scheduling.numerical.primary-dt-0p001",
        q=2,
        matrix_dimension_n=4,
        target_alpha_tilde_x=MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_X,
        target_alpha_tilde_y=MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_Y,
        schedules=six_matrix_response_preparation_window_scheduling_preparation_words(),
        central_schedule_id="p00-r100",
        post_arrival_steps=768,
        receiver_cadence_steps=16,
        spectral_receiver_cadence_steps=256,
        rolling_history_samples=16,
        first_eligible_post_arrival_step=256,
        development_root_blocks=32,
        development_primary_trajectory_count=288,
        development_issue_id=development_issue_id,
        development_seed_namespace_id=development_seed_namespace_id,
        development_root_namespace_id=development_root_namespace_id,
        preparation_tape_purpose_id="matrix-preparation-scheduling.preparation-tape",
        observation_tape_purpose_id="matrix-preparation-scheduling.observation-tape",
        tape_derivation_rule_id=source_config.primary_view.stream_derivation_rule_id,
        splice_rule_id="matrix-preparation-scheduling.independent-tape-splice",
        maximum_worker_processes=8,
        preferred_wall_seconds=14_400,
        hard_wall_seconds=43_200,
        maximum_memory_bytes=8 * 1024**3,
        artifact_resource_scope_id="matrix-preparation-scheduling.development-per-trajectory-and-attempt",
        maximum_trajectory_decoded_bytes=MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_TRAJECTORY_DECODED_DATASET_BYTES,
        maximum_trajectory_hdf5_bytes=6 * 1024**2,
        maximum_trajectory_result_bytes=MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_RESULT_BYTES,
        maximum_attempt_hdf5_bytes=288 * 6 * 1024**2,
        maximum_trajectory_integration_steps=1_152,
        maximum_total_integration_steps=309_248,
        grants_authority=False,
    )


def decode_six_matrix_response_preparation_window_scheduling_study_config(payload: bytes) -> SixMatrixResponsePreparationWindowSchedulingStudyConfig:
    """Strict, bounded canonical codec for the issued preparation scheduling design record."""

    return decode_canonical_bytes(
        payload,
        SixMatrixResponsePreparationWindowSchedulingStudyConfig,
        maximum_bytes=MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_CONFIG_BYTES,
    )


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingPreparationWaveform:
    """In-memory independent-channel waveform; arrays never enter config bytes."""

    word: SixMatrixResponsePreparationWindowSchedulingPreparationWord
    view_step_multiplier: int
    alpha_tilde_x: np.ndarray
    alpha_tilde_y: np.ndarray
    waveform_sha256: str

    def __post_init__(self) -> None:
        x = np.asarray(self.alpha_tilde_x)
        y = np.asarray(self.alpha_tilde_y)
        expected_length = self.word.final_arrival_step * self.view_step_multiplier
        if (
            self.view_step_multiplier != 1
            or x.shape != (expected_length,)
            or y.shape != x.shape
            or x.dtype != np.dtype("float64")
            or y.dtype != np.dtype("float64")
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("six-matrix preparation scheduling waveform geometry differs")
        validate_sha256(self.waveform_sha256, field_name="waveform_sha256")
        if self.waveform_sha256 != _schedule_sha256(x, y):
            raise ValueError("six-matrix preparation scheduling waveform digest differs")
        if (
            x[-1] != float(self.word.target_alpha_tilde_x)
            or y[-1] != float(self.word.target_alpha_tilde_y)
            or np.any(np.diff(x) < 0)
            or np.any(np.diff(y) < 0)
        ):
            raise ValueError("six-matrix preparation scheduling waveform does not arrive monotonically at its target")
        copied_x = np.ascontiguousarray(x, dtype="<f8")
        copied_y = np.ascontiguousarray(y, dtype="<f8")
        copied_x.setflags(write=False)
        copied_y.setflags(write=False)
        object.__setattr__(self, "alpha_tilde_x", copied_x)
        object.__setattr__(self, "alpha_tilde_y", copied_y)


def _schedule_sha256(alpha_x: np.ndarray, alpha_y: np.ndarray) -> str:
    digest = sha256()
    for value in (alpha_x, alpha_y):
        array = np.ascontiguousarray(value, dtype="<f8")
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _ramp_value(*, step: int, start_step: int, duration_steps: int, target: Decimal) -> float:
    if step <= start_step:
        return 0.0
    if step >= start_step + duration_steps:
        return float(target)
    return float(target) * ((step - start_step) / duration_steps)


def build_six_matrix_response_preparation_window_scheduling_preparation_waveform(
    word: SixMatrixResponsePreparationWindowSchedulingPreparationWord,
    *,
    view_step_multiplier: int = 1,
) -> SixMatrixResponsePreparationWindowSchedulingPreparationWaveform:
    """Compile independent X/Y starts and durations at aligned physical times."""

    if view_step_multiplier != 1:
        raise ValueError("six-matrix preparation scheduling DEVELOPMENT uses only the primary numerical view")
    length = word.final_arrival_step * view_step_multiplier
    x_start = word.x_start_step * view_step_multiplier
    y_start = word.y_start_step * view_step_multiplier
    x_duration = word.x_duration_steps * view_step_multiplier
    y_duration = word.y_duration_steps * view_step_multiplier
    x = np.asarray(
        [
            _ramp_value(
                step=step,
                start_step=x_start,
                duration_steps=x_duration,
                target=word.target_alpha_tilde_x,
            )
            for step in range(1, length + 1)
        ],
        dtype="<f8",
    )
    y = np.asarray(
        [
            _ramp_value(
                step=step,
                start_step=y_start,
                duration_steps=y_duration,
                target=word.target_alpha_tilde_y,
            )
            for step in range(1, length + 1)
        ],
        dtype="<f8",
    )
    return SixMatrixResponsePreparationWindowSchedulingPreparationWaveform(
        word=word,
        view_step_multiplier=view_step_multiplier,
        alpha_tilde_x=x,
        alpha_tilde_y=y,
        waveform_sha256=_schedule_sha256(x, y),
    )


class SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind(StrEnum):
    PREPARATION = "PREPARATION"
    OBSERVATION = "OBSERVATION"


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingNoiseTapeReceipt(CanonicalRecord):
    """Exact derivation and byte identity of one shared root-block tape."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-noise-tape-receipt'

    tape_id: str
    root_block_id: str
    seed_root_id: str
    tape_kind: SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind
    numerical_view_id: str
    view_step_multiplier: int
    purpose_id: str
    stream_index: int
    derivation_rule_id: str
    coarse_seed_sha256: str
    scientific_input_fingerprint: str
    derivation_rng_initial_state_sha256: str
    derivation_rng_final_state_sha256: str
    noise_sha256: str
    step_count: int
    consumed_start_step: int
    consumed_stop_step: int
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "tape_id",
            "root_block_id",
            "seed_root_id",
            "numerical_view_id",
            "purpose_id",
            "derivation_rule_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "coarse_seed_sha256",
            "scientific_input_fingerprint",
            "derivation_rng_initial_state_sha256",
            "derivation_rng_final_state_sha256",
            "noise_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            self.stream_index < 0
            or self.step_count < 1
            or self.consumed_start_step != 0
            or self.consumed_stop_step != self.step_count
            or self.view_step_multiplier != 1
        ):
            raise ValueError("six-matrix preparation scheduling tape range or view differs")
        if self.grants_authority:
            raise ValueError("six-matrix preparation scheduling tape receipt cannot grant authority")


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingNoiseTape:
    receipt: SixMatrixResponsePreparationWindowSchedulingNoiseTapeReceipt
    noises: ComplexArray

    def __post_init__(self) -> None:
        value = np.asarray(self.noises)
        if (
            value.shape != (self.receipt.step_count, 2, 3, 4, 4)
            or value.dtype != np.dtype("complex128")
            or not np.isfinite(value).all()
            or _array_sha256(value, dtype="<c16") != self.receipt.noise_sha256
        ):
            raise ValueError("six-matrix preparation scheduling tape bytes differ from its receipt")
        copied = np.ascontiguousarray(value, dtype="<c16")
        copied.setflags(write=False)
        object.__setattr__(self, "noises", copied)


def derive_six_matrix_response_preparation_window_scheduling_noise_tape(
    *,
    config: SixMatrixResponsePreparationWindowSchedulingStudyConfig,
    root_block_id: str,
    seed_root_id: str,
    block_index: int,
    tape_kind: SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind,
    scientific_seed_input: SixMatrixResponseScientificSeedInput | None = None,
) -> SixMatrixResponsePreparationWindowSchedulingNoiseTape:
    """Derive one primary tape; preparation and observation use distinct streams."""

    validate_stable_id(root_block_id, field_name="root_block_id")
    validate_stable_id(seed_root_id, field_name="seed_root_id")
    if block_index < 0:
        raise ValueError("six-matrix preparation scheduling root block index cannot be negative")
    if not isinstance(tape_kind, SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind):
        raise ValueError("preparation tape requires its explicit physical tape role")
    scientific_seed = require_six_matrix_scientific_seed_input(
        scientific_seed_input,
        scientific_role="preparation-tape" if tape_kind is SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind.PREPARATION else "observation-tape",
        current_root_id=seed_root_id, current_context_sha256=config.fingerprint(),
        stream_index=block_index,
    )
    purpose_prefix = (
        config.preparation_tape_purpose_id
        if tape_kind is SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind.PREPARATION
        else config.observation_tape_purpose_id
    )
    purpose_id = f"{purpose_prefix}.{root_block_id}"
    step_count = (
        max(word.final_arrival_step for word in config.schedules)
        if tape_kind is SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind.PREPARATION
        else config.post_arrival_steps
    )
    rng, stream = derive_rng_stream(
        seed_root_id=seed_root_id,
        purpose_id=purpose_id,
        stream_index=block_index,
        derivation_rule_id=config.tape_derivation_rule_id,
        scientific_seed_sha256=scientific_seed.full_seed_sha256,
    )
    initial_rng = _rng_state_sha256(rng)
    noises = np.ascontiguousarray(
        np.stack(tuple(hermitian_noise(rng=rng, q=config.q) for _ in range(step_count))),
        dtype="<c16",
    )
    receipt = SixMatrixResponsePreparationWindowSchedulingNoiseTapeReceipt(
        tape_id=(f"tape.{seed_root_id}.{tape_kind.value.lower()}.{config.primary_view.view_id}"),
        root_block_id=root_block_id,
        seed_root_id=seed_root_id,
        tape_kind=tape_kind,
        numerical_view_id=config.primary_view.view_id,
        view_step_multiplier=1,
        purpose_id=purpose_id,
        stream_index=block_index,
        derivation_rule_id=config.tape_derivation_rule_id,
        coarse_seed_sha256=stream.derived_seed_sha256,
        scientific_input_fingerprint=scientific_seed.fingerprint(),
        derivation_rng_initial_state_sha256=initial_rng,
        derivation_rng_final_state_sha256=_rng_state_sha256(rng),
        noise_sha256=_array_sha256(noises, dtype="<c16"),
        step_count=step_count,
        consumed_start_step=0,
        consumed_stop_step=step_count,
        grants_authority=False,
    )
    return SixMatrixResponsePreparationWindowSchedulingNoiseTape(receipt=receipt, noises=noises)


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingScheduleLedger(CanonicalRecord):
    """Requested/accepted/applied/realized preparation and tape custody."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-schedule-ledger'

    ledger_id: str
    root_block_id: str
    schedule_id: str
    numerical_view_id: str
    view_step_multiplier: int
    preparation_word: ObjectIdentity
    requested_schedule_sha256: str
    accepted_schedule_sha256: str
    applied_schedule_sha256: str
    realized_schedule_sha256: str
    requested_arrival_step: int
    accepted_arrival_step: int
    applied_arrival_step: int
    realized_arrival_step: int
    requested_final: SixMatrixResponseScaledCouplings
    accepted_final: SixMatrixResponseScaledCouplings
    applied_final: SixMatrixResponseScaledCouplings
    realized_final: SixMatrixResponseScaledCouplings
    preparation_tape: ObjectIdentity
    preparation_consumed_start_step: int
    preparation_consumed_stop_step: int
    observation_tape: ObjectIdentity
    observation_consumed_start_step: int
    observation_consumed_stop_step: int
    splice_rule_id: str
    splice_sha256: str
    clipped: bool
    complete_delivery: bool
    valid: bool
    reason_codes: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "ledger_id",
            "root_block_id",
            "schedule_id",
            "numerical_view_id",
            "splice_rule_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.schedule_id not in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS:
            raise ValueError("six-matrix preparation scheduling schedule ledger names an unknown word")
        if self.preparation_word.object_schema != SixMatrixResponsePreparationWindowSchedulingPreparationWord.SCHEMA:
            raise ValueError("six-matrix preparation scheduling ledger preparation-word identity differs")
        if (
            self.preparation_tape.object_schema != SixMatrixResponsePreparationWindowSchedulingNoiseTapeReceipt.SCHEMA
            or self.observation_tape.object_schema != SixMatrixResponsePreparationWindowSchedulingNoiseTapeReceipt.SCHEMA
            or self.preparation_tape == self.observation_tape
        ):
            raise ValueError("six-matrix preparation scheduling ledger tape identities differ or are reused")
        for name in (
            "requested_schedule_sha256",
            "accepted_schedule_sha256",
            "applied_schedule_sha256",
            "realized_schedule_sha256",
            "splice_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if self.view_step_multiplier != 1:
            raise ValueError("six-matrix preparation scheduling ledger view multiplier differs")
        if (
            min(
                self.requested_arrival_step,
                self.accepted_arrival_step,
                self.applied_arrival_step,
                self.realized_arrival_step,
                self.preparation_consumed_start_step,
                self.preparation_consumed_stop_step,
                self.observation_consumed_start_step,
                self.observation_consumed_stop_step,
            )
            < 0
        ):
            raise ValueError("six-matrix preparation scheduling ledger contains a negative step")
        if (
            self.preparation_consumed_start_step != 0
            or self.observation_consumed_start_step != 0
            or self.preparation_consumed_stop_step != self.realized_arrival_step
        ):
            raise ValueError("six-matrix preparation scheduling consumed tape ranges differ from realized delivery")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        requested_target = SixMatrixResponseScaledCouplings(
            MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_X,
            MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_Y,
        )
        native_target = SixMatrixResponseScaledCouplings(
            _decimal(float(MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_X)),
            _decimal(float(MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_Y)),
        )
        if self.valid:
            expected_arrival = _SCHEDULE_TABLE[self.schedule_id][-1] * self.view_step_multiplier
            if (
                self.reason_codes
                or self.clipped
                or not self.complete_delivery
                or len(
                    {
                        self.requested_schedule_sha256,
                        self.accepted_schedule_sha256,
                        self.applied_schedule_sha256,
                        self.realized_schedule_sha256,
                    }
                )
                != 1
                or (
                    self.requested_arrival_step,
                    self.accepted_arrival_step,
                    self.applied_arrival_step,
                    self.realized_arrival_step,
                )
                != (expected_arrival,) * 4
                or self.requested_final != requested_target
                or self.accepted_final != requested_target
                or self.applied_final != native_target
                or self.realized_final != native_target
                or self.observation_consumed_stop_step
                != MATRIX_RESPONSE_PREPARATION_SCHEDULING_POST_ARRIVAL_STEPS * self.view_step_multiplier
            ):
                raise ValueError("six-matrix preparation scheduling valid ledger lacks exact four-stage delivery")
        elif not self.reason_codes:
            raise ValueError("six-matrix preparation scheduling invalid schedule ledger requires typed reasons")
        if self.grants_authority:
            raise ValueError("six-matrix preparation scheduling schedule ledger cannot grant authority")


class SixMatrixResponsePreparationWindowSchedulingPhaseStateKind(StrEnum):
    ARRIVAL = "ARRIVAL"
    FINAL = "FINAL"


def _phase_state_sha256(
    *,
    q: int,
    global_step_index: int,
    post_arrival_step: int,
    alpha_x: Decimal,
    alpha_y: Decimal,
    positions: bytes,
    momenta: bytes,
) -> str:
    digest = sha256()
    for value in (
        str(q).encode("ascii"),
        str(global_step_index).encode("ascii"),
        str(post_arrival_step).encode("ascii"),
        str(alpha_x).encode("ascii"),
        str(alpha_y).encode("ascii"),
        positions,
        momenta,
    ):
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingPhaseState(CanonicalRecord):
    """Complete positions and momenta at arrival or final observation time."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-phase-state'

    state_id: str
    root_block_id: str
    schedule_id: str
    numerical_view_id: str
    state_kind: SixMatrixResponsePreparationWindowSchedulingPhaseStateKind
    q: int
    global_step_index: int
    post_arrival_step: int
    alpha_tilde_x: Decimal
    alpha_tilde_y: Decimal
    positions_base64: str
    momenta_base64: str
    positions_sha256: str
    momenta_sha256: str
    state_sha256: str

    def __post_init__(self) -> None:
        for name in ("state_id", "root_block_id", "schedule_id", "numerical_view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.schedule_id not in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS or self.q != 2:
            raise ValueError("six-matrix preparation scheduling phase-state coordinate differs")
        if min(self.global_step_index, self.post_arrival_step) < 0:
            raise ValueError("six-matrix preparation scheduling phase-state step cannot be negative")
        if self.state_kind is SixMatrixResponsePreparationWindowSchedulingPhaseStateKind.ARRIVAL and self.post_arrival_step != 0:
            raise ValueError("six-matrix preparation scheduling arrival state must be at local s=0")
        if self.state_kind is SixMatrixResponsePreparationWindowSchedulingPhaseStateKind.FINAL and self.post_arrival_step < 1:
            raise ValueError("six-matrix preparation scheduling final state must follow arrival")
        for name in ("alpha_tilde_x", "alpha_tilde_y"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        expected_bytes = prod((2, 3, self.q**2, self.q**2)) * 16
        decoded: dict[str, bytes] = {}
        for data_name, digest_name in (
            ("positions_base64", "positions_sha256"),
            ("momenta_base64", "momenta_sha256"),
        ):
            try:
                raw = base64.b64decode(getattr(self, data_name), validate=True)
            except ValueError as error:
                raise ValueError(f"six-matrix preparation scheduling {data_name} is not canonical base64") from error
            if len(raw) != expected_bytes or base64.b64encode(raw).decode("ascii") != getattr(
                self, data_name
            ):
                raise ValueError(f"six-matrix preparation scheduling {data_name} has another state geometry")
            validate_sha256(getattr(self, digest_name), field_name=digest_name)
            if sha256(raw).hexdigest() != getattr(self, digest_name):
                raise ValueError(f"six-matrix preparation scheduling {data_name} digest differs")
            decoded[data_name] = raw
        validate_sha256(self.state_sha256, field_name="state_sha256")
        expected_state = _phase_state_sha256(
            q=self.q,
            global_step_index=self.global_step_index,
            post_arrival_step=self.post_arrival_step,
            alpha_x=self.alpha_tilde_x,
            alpha_y=self.alpha_tilde_y,
            positions=decoded["positions_base64"],
            momenta=decoded["momenta_base64"],
        )
        if self.state_sha256 != expected_state:
            raise ValueError("six-matrix preparation scheduling combined phase-state digest differs")


def six_matrix_response_preparation_window_scheduling_phase_state_from_state(
    *,
    root_block_id: str,
    schedule_id: str,
    numerical_view_id: str,
    state_kind: SixMatrixResponsePreparationWindowSchedulingPhaseStateKind,
    state: SixMatrixState,
    post_arrival_step: int,
) -> SixMatrixResponsePreparationWindowSchedulingPhaseState:
    if not state.finite:
        raise ValueError("six-matrix preparation scheduling cannot persist a nonfinite phase state")
    positions = state.positions.tobytes(order="C")
    momenta = state.momenta.tobytes(order="C")
    alpha_x = _decimal(state.alpha_tilde_x)
    alpha_y = _decimal(state.alpha_tilde_y)
    return SixMatrixResponsePreparationWindowSchedulingPhaseState(
        state_id=(
            f"state.{root_block_id}.{schedule_id}.{numerical_view_id}.{state_kind.value.lower()}"
        ),
        root_block_id=root_block_id,
        schedule_id=schedule_id,
        numerical_view_id=numerical_view_id,
        state_kind=state_kind,
        q=state.q,
        global_step_index=state.step_index,
        post_arrival_step=post_arrival_step,
        alpha_tilde_x=alpha_x,
        alpha_tilde_y=alpha_y,
        positions_base64=base64.b64encode(positions).decode("ascii"),
        momenta_base64=base64.b64encode(momenta).decode("ascii"),
        positions_sha256=sha256(positions).hexdigest(),
        momenta_sha256=sha256(momenta).hexdigest(),
        state_sha256=_phase_state_sha256(
            q=state.q,
            global_step_index=state.step_index,
            post_arrival_step=post_arrival_step,
            alpha_x=alpha_x,
            alpha_y=alpha_y,
            positions=positions,
            momenta=momenta,
        ),
    )


class SixMatrixResponsePreparationWindowSchedulingScheduleTerminal(StrEnum):
    COMPLETED = "COMPLETED"
    NUMERICAL_INVALID = "NUMERICAL_INVALID"


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingScheduleTraceResult(CanonicalRecord):
    """Canonical native trace summary; full paths live in the reserved artifact."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-schedule-trace-result'

    trace_id: str
    root_block_id: str
    block_index: int
    schedule_id: str
    numerical_view_id: str
    view_step_multiplier: int
    terminal: SixMatrixResponsePreparationWindowSchedulingScheduleTerminal
    completed_preparation_steps: int
    completed_observation_steps: int
    trajectory_state_count: int
    trajectory_positions_sha256: str
    trajectory_momenta_sha256: str
    schedule_ledger: SixMatrixResponsePreparationWindowSchedulingScheduleLedger
    arrival_state: SixMatrixResponsePreparationWindowSchedulingPhaseState | None
    final_state: SixMatrixResponsePreparationWindowSchedulingPhaseState | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("trace_id", "root_block_id", "schedule_id", "numerical_view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.block_index < 0
            or self.schedule_id not in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS
            or self.view_step_multiplier != 1
            or min(
                self.completed_preparation_steps,
                self.completed_observation_steps,
                self.trajectory_state_count,
            )
            < 0
        ):
            raise ValueError("six-matrix preparation scheduling trace coordinate/count differs")
        for name in ("trajectory_positions_sha256", "trajectory_momenta_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.schedule_ledger.schedule_id != self.schedule_id
            or self.schedule_ledger.root_block_id != self.root_block_id
            or self.schedule_ledger.numerical_view_id != self.numerical_view_id
        ):
            raise ValueError("six-matrix preparation scheduling trace/ledger schedule identity differs")
        for phase_state in (self.arrival_state, self.final_state):
            if phase_state is not None and (
                phase_state.root_block_id != self.root_block_id
                or phase_state.schedule_id != self.schedule_id
                or phase_state.numerical_view_id != self.numerical_view_id
            ):
                raise ValueError("six-matrix preparation scheduling trace phase-state identity differs")
        if self.trajectory_state_count != (
            self.completed_preparation_steps + self.completed_observation_steps + 1
        ):
            raise ValueError("six-matrix preparation scheduling trace state count differs from completed integration")
        if self.terminal is SixMatrixResponsePreparationWindowSchedulingScheduleTerminal.COMPLETED:
            expected_preparation = _SCHEDULE_TABLE[self.schedule_id][-1]
            expected_preparation *= self.view_step_multiplier
            expected_observation = MATRIX_RESPONSE_PREPARATION_SCHEDULING_POST_ARRIVAL_STEPS * self.view_step_multiplier
            if (
                self.reason_codes
                or not self.schedule_ledger.valid
                or self.completed_preparation_steps != expected_preparation
                or self.completed_observation_steps != expected_observation
                or self.arrival_state is None
                or self.final_state is None
            ):
                raise ValueError("six-matrix preparation scheduling completed trace is incomplete")
        elif not self.reason_codes or self.schedule_ledger.valid or self.final_state is not None:
            raise ValueError("six-matrix preparation scheduling numerical-invalid trace disposition differs")


def _measurement_dataset_paths() -> tuple[str, ...]:
    return (
        "/noise/observation",
        "/noise/preparation",
        "/trajectory/alpha_tilde",
        "/trajectory/momenta",
        "/trajectory/positions",
        "/trajectory/post_arrival_step",
    )


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingMeasurementArtifactReceipt(CanonicalRecord):
    """Exact custody receipt for the full-path root-block HDF5 payload."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-measurement-artifact-receipt'

    receipt_id: str
    task_id: str
    root_block_id: str
    output_id: str
    payload_schema: str
    media_type: str
    reservation: SixMatrixResponsePreparationWindowSchedulingTrajectoryArtifactReservation
    schedule_ids: tuple[str, ...]
    dataset_paths: tuple[str, ...]
    content_sha256: str
    size_bytes: int
    decoded_dataset_bytes: int
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("receipt_id", "task_id", "root_block_id", "output_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.payload_schema != MATRIX_RESPONSE_PREPARATION_SCHEDULING_MEASUREMENT_HDF5_SCHEMA
            or self.media_type != "application/x-hdf5"
            or self.dataset_paths != _measurement_dataset_paths()
        ):
            raise ValueError("six-matrix preparation scheduling measurement artifact contract differs")
        require_sorted_unique_strings(
            self.schedule_ids, field_name="schedule_ids", allow_empty=False
        )
        if not set(self.schedule_ids).issubset(MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS):
            raise ValueError("six-matrix preparation scheduling measurement artifact names an unknown schedule")
        if (
            self.reservation.task_id != self.task_id
            or self.reservation.root_block_id != self.root_block_id
            or self.reservation.output_id != self.output_id
            or self.reservation.payload_schema != self.payload_schema
        ):
            raise ValueError("six-matrix preparation scheduling measurement receipt differs from its reservation")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        if not 1 <= self.size_bytes <= self.reservation.maximum_bytes:
            raise ValueError("six-matrix preparation scheduling measurement artifact exceeds its reservation")
        if not 1 <= self.decoded_dataset_bytes <= self.reservation.maximum_decoded_bytes:
            raise ValueError("six-matrix preparation scheduling decoded datasets exceed their separate ceiling")
        if self.grants_authority:
            raise ValueError("six-matrix preparation scheduling measurement artifact cannot grant authority")


class SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"


def _development_views(
    *,
    root_ids: tuple[str, ...],
    numerical_member_id: str,
    numerical_view_id: str,
) -> tuple[ResponseAcquisitionView, ...]:
    return tuple(
        sorted(
            (
                ResponseAcquisitionView(
                    view_id=f"view.{root_id}.{schedule_id}.{numerical_view_id}",
                    acquisition_group_id=(
                        f"acquisition.{root_id}.{schedule_id}.{numerical_view_id}"
                    ),
                    physical_independent_unit_id=root_id,
                    numerical_member_id=numerical_member_id,
                    action_word_id=schedule_id,
                )
                for root_id in root_ids
                for schedule_id in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS
            ),
            key=lambda value: value.view_id,
        )
    )


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster(CanonicalRecord):
    """Closed outcome-blind physical-unit/action/acquisition/view roster."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-development-roster'

    roster_id: str
    study_config: ObjectIdentity
    issue_id: str
    seed_namespace_id: str
    root_namespace_id: str
    physical_independent_unit_ids: tuple[str, ...]
    schedule_ids: tuple[str, ...]
    numerical_member_id: str
    numerical_view_id: str
    views: tuple[ResponseAcquisitionView, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "roster_id",
            "issue_id",
            "seed_namespace_id",
            "root_namespace_id",
            "numerical_member_id",
            "numerical_view_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.study_config.object_schema != SixMatrixResponsePreparationWindowSchedulingStudyConfig.SCHEMA:
            raise ValueError("six-matrix preparation scheduling roster requires its exact study config")
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.schedule_ids,
            field_name="schedule_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.views, attribute="view_id", field_name="views")
        expected_roots = tuple(f"{self.root_namespace_id}.{index:02d}" for index in range(32))
        expected_views = _development_views(
            root_ids=expected_roots,
            numerical_member_id=self.numerical_member_id,
            numerical_view_id=self.numerical_view_id,
        )
        if (
            self.physical_independent_unit_ids != expected_roots
            or self.schedule_ids != MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS
            or self.views != expected_views
            or len(self.views) != 288
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
            or self.grants_authority
        ):
            raise ValueError("six-matrix preparation scheduling development roster differs from the closed 32x9 design")


def build_six_matrix_response_preparation_window_scheduling_development_roster(
    config: SixMatrixResponsePreparationWindowSchedulingStudyConfig,
) -> SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster:
    roots = tuple(f"{config.development_root_namespace_id}.{index:02d}" for index in range(32))
    return SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster(
        roster_id=f"matrix-preparation-scheduling.development-roster.{config.development_issue_id}",
        study_config=ObjectIdentity.from_record(config.config_id, config),
        issue_id=config.development_issue_id,
        seed_namespace_id=config.development_seed_namespace_id,
        root_namespace_id=config.development_root_namespace_id,
        physical_independent_unit_ids=roots,
        schedule_ids=MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS,
        numerical_member_id=config.primary_numerical_member_id,
        numerical_view_id=config.primary_view.view_id,
        views=_development_views(
            root_ids=roots,
            numerical_member_id=config.primary_numerical_member_id,
            numerical_view_id=config.primary_view.view_id,
        ),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        grants_authority=False,
    )


def decode_six_matrix_response_preparation_window_scheduling_development_roster(payload: bytes) -> SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster:
    return decode_canonical_bytes(
        payload,
        SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster,
        maximum_bytes=2 * 1024 * 1024,
    )


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingTrajectoryRequest(CanonicalRecord):
    """One action-homogeneous preparation scheduling trajectory task coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-trajectory-request'

    request_id: str
    task_id: str
    output_id: str
    measurement_output_id: str
    measurement_artifact_reservation: SixMatrixResponsePreparationWindowSchedulingTrajectoryArtifactReservation
    root_block_id: str
    block_index: int
    tranche: SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche
    issue_id: str
    seed_namespace_id: str
    development_roster: ObjectIdentity
    issued_nested_view: ResponseAcquisitionView
    study_config: ObjectIdentity
    source_config: ObjectIdentity
    numerical_view_id: str
    numerical_member_id: str
    scientific_view_id: str
    physical_independent_unit_id: str
    acquisition_group_id: str
    seed_root_id: str
    scientific_inputs: tuple[SixMatrixResponseScientificSeedInput, ...]
    schedule_id: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        _require_preparation_scientific_inputs(
            self.scientific_inputs, root_block_id=self.root_block_id,
            seed_root_id=self.seed_root_id, block_index=self.block_index,
            current_context_sha256=self.study_config.object_fingerprint,
        )
        for name in (
            "request_id",
            "task_id",
            "output_id",
            "measurement_output_id",
            "root_block_id",
            "issue_id",
            "seed_namespace_id",
            "numerical_view_id",
            "numerical_member_id",
            "scientific_view_id",
            "physical_independent_unit_id",
            "acquisition_group_id",
            "seed_root_id",
            "schedule_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.output_id == self.measurement_output_id:
            raise ValueError("six-matrix preparation scheduling trajectory outputs must be distinct")
        if (
            self.physical_independent_unit_id != self.root_block_id
            or self.development_roster.object_schema != SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster.SCHEMA
            or self.issued_nested_view.view_id != self.scientific_view_id
            or self.issued_nested_view.acquisition_group_id != self.acquisition_group_id
            or self.issued_nested_view.physical_independent_unit_id
            != self.physical_independent_unit_id
            or self.issued_nested_view.numerical_member_id != self.numerical_member_id
            or self.issued_nested_view.action_word_id != self.schedule_id
            or len(
                {
                    self.physical_independent_unit_id,
                    self.acquisition_group_id,
                    self.scientific_view_id,
                }
            )
            != 3
        ):
            raise ValueError("six-matrix preparation scheduling trajectory differs from its issued physical-unit/action/acquisition/view")
        if (
            self.tranche is not SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche.DEVELOPMENT
            or not 0 <= self.block_index < 32
            or self.schedule_id not in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS
        ):
            raise ValueError("six-matrix preparation scheduling trajectory coordinate lies outside the frozen roster")
        if (
            not self.issue_id.startswith("matrix-preparation-scheduling.issue.development.")
            or not self.seed_namespace_id.startswith("matrix-preparation-scheduling.seed.development.")
            or self.seed_root_id != f"{self.seed_namespace_id}.{self.physical_independent_unit_id}"
        ):
            raise ValueError("six-matrix preparation scheduling trajectory issue/root/seed topology differs")
        if self.study_config.object_schema != SixMatrixResponsePreparationWindowSchedulingStudyConfig.SCHEMA:
            raise ValueError("six-matrix preparation scheduling trajectory study-config identity differs")
        if self.source_config.object_schema != SixMatrixResponseSixMatrixSourceConfig.SCHEMA:
            raise ValueError("six-matrix preparation scheduling trajectory source-config identity differs")
        reservation = self.measurement_artifact_reservation
        if (
            reservation.task_id != self.task_id
            or reservation.root_block_id != self.root_block_id
            or reservation.output_id != self.measurement_output_id
        ):
            raise ValueError("six-matrix preparation scheduling trajectory request differs from its reservation")
        if self.evidence_ceiling is not EvidenceCeiling.MEASUREMENT:
            raise ValueError("six-matrix preparation scheduling trajectory request is not capped at native measurement")
        if (self.outcome_access, self.visibility_ceiling) != (
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            VisibilityCeiling.DEVELOPMENT_ONLY,
        ):
            raise ValueError("six-matrix preparation scheduling trajectory visibility differs from its tranche")
        if self.grants_authority:
            raise ValueError("six-matrix preparation scheduling trajectory request cannot grant authority")

    @property
    def schedule_ids(self) -> tuple[str, ...]:
        """Compatibility view for the pure schedule executor."""

        return (self.schedule_id,)


def decode_six_matrix_response_preparation_window_scheduling_trajectory_request(payload: bytes) -> SixMatrixResponsePreparationWindowSchedulingTrajectoryRequest:
    return decode_canonical_bytes(
        payload,
        SixMatrixResponsePreparationWindowSchedulingTrajectoryRequest,
        maximum_bytes=MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_REQUEST_BYTES,
    )


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingTrajectoryResult(CanonicalRecord):
    """One persisted native trajectory; scientific aggregation occurs later."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-trajectory-result'

    result_id: str
    task_id: str
    output_id: str
    request: ObjectIdentity
    root_block_id: str
    block_index: int
    tranche: SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche
    issue_id: str
    seed_namespace_id: str
    development_roster: ObjectIdentity
    issued_nested_view: ResponseAcquisitionView
    schedule_id: str
    numerical_view_id: str
    numerical_member_id: str
    scientific_view_id: str
    physical_independent_unit_id: str
    acquisition_group_id: str
    view_step_multiplier: int
    root_seed_sha256: str
    study_config: ObjectIdentity
    scientific_inputs: tuple[SixMatrixResponseScientificSeedInput, ...]
    preparation_tape: SixMatrixResponsePreparationWindowSchedulingNoiseTapeReceipt
    observation_tape: SixMatrixResponsePreparationWindowSchedulingNoiseTapeReceipt
    measurement_artifact: SixMatrixResponsePreparationWindowSchedulingMeasurementArtifactReceipt
    schedule_trace: SixMatrixResponsePreparationWindowSchedulingScheduleTraceResult
    planned_integration_updates: int
    completed_integration_updates: int
    terminal: SixMatrixResponsePreparationWindowSchedulingScheduleTerminal
    reason_codes: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    scientific_verdict_ids: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "result_id",
            "task_id",
            "output_id",
            "root_block_id",
            "issue_id",
            "seed_namespace_id",
            "schedule_id",
            "numerical_view_id",
            "numerical_member_id",
            "scientific_view_id",
            "physical_independent_unit_id",
            "acquisition_group_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.request.object_schema != SixMatrixResponsePreparationWindowSchedulingTrajectoryRequest.SCHEMA:
            raise ValueError("six-matrix preparation scheduling trajectory result request identity differs")
        if (
            self.tranche is not SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche.DEVELOPMENT
            or not self.issue_id.startswith("matrix-preparation-scheduling.issue.development.")
            or not self.seed_namespace_id.startswith("matrix-preparation-scheduling.seed.development.")
        ):
            raise ValueError("six-matrix preparation scheduling trajectory result issue/root topology differs")
        if (
            self.physical_independent_unit_id != self.root_block_id
            or self.development_roster.object_schema != SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster.SCHEMA
            or self.issued_nested_view.view_id != self.scientific_view_id
            or self.issued_nested_view.acquisition_group_id != self.acquisition_group_id
            or self.issued_nested_view.physical_independent_unit_id
            != self.physical_independent_unit_id
            or self.issued_nested_view.numerical_member_id != self.numerical_member_id
            or self.issued_nested_view.action_word_id != self.schedule_id
        ):
            raise ValueError("six-matrix preparation scheduling trajectory result differs from its development view")
        if (
            self.schedule_id not in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS
            or not 0 <= self.block_index < 32
            or self.view_step_multiplier != 1
            or self.schedule_trace.schedule_id != self.schedule_id
            or self.schedule_trace.root_block_id != self.root_block_id
        ):
            raise ValueError("six-matrix preparation scheduling trajectory result coordinate differs")
        validate_sha256(self.root_seed_sha256, field_name="root_seed_sha256")
        expected_seed_root = f"{self.seed_namespace_id}.{self.physical_independent_unit_id}"
        if self.study_config.object_schema != SixMatrixResponsePreparationWindowSchedulingStudyConfig.SCHEMA:
            raise ValueError("preparation result requires its exact current study context")
        _require_preparation_scientific_inputs(
            self.scientific_inputs, root_block_id=self.root_block_id,
            seed_root_id=expected_seed_root, block_index=self.block_index,
            current_context_sha256=self.study_config.object_fingerprint,
        )
        expected_preparation_purpose = f"matrix-preparation-scheduling.preparation-tape.{self.root_block_id}"
        expected_observation_purpose = f"matrix-preparation-scheduling.observation-tape.{self.root_block_id}"
        if (
            self.preparation_tape.tape_kind is not SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind.PREPARATION
            or self.observation_tape.tape_kind is not SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind.OBSERVATION
            or self.preparation_tape.root_block_id != self.root_block_id
            or self.observation_tape.root_block_id != self.root_block_id
            or self.preparation_tape.seed_root_id != expected_seed_root
            or self.observation_tape.seed_root_id != expected_seed_root
            or self.preparation_tape.numerical_view_id != self.numerical_view_id
            or self.observation_tape.numerical_view_id != self.numerical_view_id
            or self.preparation_tape.view_step_multiplier != self.view_step_multiplier
            or self.observation_tape.view_step_multiplier != self.view_step_multiplier
            or self.preparation_tape.stream_index != self.block_index
            or self.observation_tape.stream_index != self.block_index
            or self.preparation_tape.purpose_id != expected_preparation_purpose
            or self.observation_tape.purpose_id != expected_observation_purpose
            or self.preparation_tape.coarse_seed_sha256 == self.observation_tape.coarse_seed_sha256
            or self.preparation_tape.noise_sha256 == self.observation_tape.noise_sha256
            or self.preparation_tape.scientific_input_fingerprint != self.scientific_inputs[1].fingerprint()
            or self.observation_tape.scientific_input_fingerprint != self.scientific_inputs[2].fingerprint()
            or self.preparation_tape.coarse_seed_sha256 != self.scientific_inputs[1].full_seed_sha256
            or self.observation_tape.coarse_seed_sha256 != self.scientific_inputs[2].full_seed_sha256
            or self.root_seed_sha256
            != _root_seed_sha256(
                seed_root_id=expected_seed_root,
                root_block_id=self.root_block_id,
                block_index=self.block_index,
                current_context_sha256=self.study_config.object_fingerprint,
                scientific_seed_input=self.scientific_inputs[0],
            )
        ):
            raise ValueError("six-matrix preparation scheduling trajectory tape custody/independence differs")
        expected_planned = (
            _SCHEDULE_TABLE[self.schedule_id][-1] + MATRIX_RESPONSE_PREPARATION_SCHEDULING_POST_ARRIVAL_STEPS
        ) * self.view_step_multiplier
        expected_completed = (
            self.schedule_trace.completed_preparation_steps
            + self.schedule_trace.completed_observation_steps
        )
        if (
            self.planned_integration_updates != expected_planned
            or self.completed_integration_updates != expected_completed
            or not 0 <= expected_completed <= expected_planned
        ):
            raise ValueError("six-matrix preparation scheduling trajectory integration arithmetic differs")
        if (
            self.measurement_artifact.task_id != self.task_id
            or self.measurement_artifact.root_block_id != self.root_block_id
            or self.measurement_artifact.output_id == self.output_id
            or self.measurement_artifact.schedule_ids != (self.schedule_id,)
            or self.measurement_artifact.decoded_dataset_bytes
            != six_matrix_response_preparation_window_scheduling_trajectory_decoded_dataset_bytes(
                self.schedule_id,
                view_step_multiplier=self.view_step_multiplier,
            )
        ):
            raise ValueError("six-matrix preparation scheduling trajectory measurement custody differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.terminal is SixMatrixResponsePreparationWindowSchedulingScheduleTerminal.COMPLETED:
            if (
                self.schedule_trace.terminal is not self.terminal
                or self.reason_codes
                or expected_completed != expected_planned
            ):
                raise ValueError("six-matrix preparation scheduling completed trajectory is incomplete")
        elif self.schedule_trace.terminal is not self.terminal or not self.reason_codes:
            raise ValueError("six-matrix preparation scheduling invalid trajectory disposition differs")
        if (
            self.evidence_ceiling is not EvidenceCeiling.MEASUREMENT
            or (self.outcome_access, self.visibility_ceiling)
            != (OutcomeAccess.DEVELOPMENT_VISIBLE, VisibilityCeiling.DEVELOPMENT_ONLY)
            or self.scientific_verdict_ids
            or self.grants_authority
        ):
            raise ValueError("six-matrix preparation scheduling trajectory cannot adjudicate or grant authority")


def decode_six_matrix_response_preparation_window_scheduling_trajectory_result(payload: bytes) -> SixMatrixResponsePreparationWindowSchedulingTrajectoryResult:
    return decode_canonical_bytes(
        payload,
        SixMatrixResponsePreparationWindowSchedulingTrajectoryResult,
        maximum_bytes=MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_RESULT_BYTES,
    )


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingScheduleTrace:
    """In-memory complete path paired with its canonical artifact summary."""

    word: SixMatrixResponsePreparationWindowSchedulingPreparationWord
    states: tuple[SixMatrixState, ...]
    result: SixMatrixResponsePreparationWindowSchedulingScheduleTraceResult

    def __post_init__(self) -> None:
        if not self.states or self.result.trajectory_state_count != len(self.states):
            raise ValueError("six-matrix preparation scheduling in-memory trace state count differs")
        if tuple(value.step_index for value in self.states) != tuple(range(len(self.states))):
            raise ValueError("six-matrix preparation scheduling in-memory trace steps are not contiguous")
        if self.word.schedule_id != self.result.schedule_id:
            raise ValueError("six-matrix preparation scheduling in-memory trace schedule identity differs")
        positions = np.stack(tuple(value.positions for value in self.states))
        momenta = np.stack(tuple(value.momenta for value in self.states))
        if (
            _array_sha256(positions, dtype="<c16") != self.result.trajectory_positions_sha256
            or _array_sha256(momenta, dtype="<c16") != self.result.trajectory_momenta_sha256
        ):
            raise ValueError("six-matrix preparation scheduling in-memory path bytes differ from the result")


def _readonly_array(value: np.ndarray, *, dtype: str) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingPersistedScheduleTrace:
    """Fresh-process complete schedule path decoded from the custody artifact."""

    result: SixMatrixResponsePreparationWindowSchedulingScheduleTraceResult
    positions: np.ndarray
    momenta: np.ndarray
    alpha_tilde: np.ndarray
    post_arrival_steps: np.ndarray

    def __post_init__(self) -> None:
        count = self.result.trajectory_state_count
        geometry = (count, 2, 3, 4, 4)
        if (
            self.positions.shape != geometry
            or self.momenta.shape != geometry
            or self.alpha_tilde.shape != (count, 2)
            or self.post_arrival_steps.shape != (count,)
            or self.positions.dtype != np.dtype("<c16")
            or self.momenta.dtype != np.dtype("<c16")
            or self.alpha_tilde.dtype != np.dtype("<f8")
            or self.post_arrival_steps.dtype != np.dtype("<i8")
        ):
            raise ValueError("six-matrix preparation scheduling persisted trace geometry or dtype differs")
        if any(
            value.flags.writeable
            for value in (
                self.positions,
                self.momenta,
                self.alpha_tilde,
                self.post_arrival_steps,
            )
        ):
            raise ValueError("six-matrix preparation scheduling persisted trace arrays must be immutable")
        arrival_index = self.result.completed_preparation_steps
        expected_local_steps = np.arange(count, dtype="<i8") - arrival_index
        if (
            not np.array_equal(self.post_arrival_steps, expected_local_steps)
            or _array_sha256(self.positions, dtype="<c16")
            != self.result.trajectory_positions_sha256
            or _array_sha256(self.momenta, dtype="<c16") != self.result.trajectory_momenta_sha256
        ):
            raise ValueError("six-matrix preparation scheduling persisted trace bytes differ from its receipt")
        if not np.isfinite(self.alpha_tilde).all():
            raise ValueError("six-matrix preparation scheduling persisted trace has nonfinite couplings")


def _encode_six_matrix_response_preparation_window_scheduling_measurement_hdf5(
    *,
    request: SixMatrixResponsePreparationWindowSchedulingTrajectoryRequest,
    numerical_view_id: str,
    view_step_multiplier: int,
    traces: tuple[SixMatrixResponsePreparationWindowSchedulingScheduleTrace, ...],
    preparation_tape: SixMatrixResponsePreparationWindowSchedulingNoiseTape,
    observation_tape: SixMatrixResponsePreparationWindowSchedulingNoiseTape,
) -> bytes:
    """Encode one bounded full trajectory without external I/O."""

    if len(traces) != 1:
        raise ValueError("six-matrix preparation scheduling HDF5 payload must contain exactly one trajectory")

    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - installed product dependency
        raise RuntimeError("six-matrix preparation scheduling HDF5 encoding requires h5py") from error
    buffer = BytesIO()
    with h5py.File(buffer, "w", libver=("earliest", "v114"), track_order=False) as handle:
        attributes = {
            "empirical_lawhood_clocks": '{"noise":"matrix-preparation-scheduling.root-tape-step","trajectory":"matrix-preparation-scheduling.integrator-step"}',
            "empirical_lawhood_frames": '{"noise":"standardized-hermitian-noise","trajectory":"six-matrix-native"}',
            "empirical_lawhood_keys": '["post_arrival_step"]',
            "empirical_lawhood_payload_schema": MATRIX_RESPONSE_PREPARATION_SCHEDULING_MEASUREMENT_HDF5_SCHEMA,
            "empirical_lawhood_units": '{"alpha_tilde":"1","momenta":"native","noise":"standardized","positions":"native","post_arrival_step":"step"}',
        }
        for name in sorted(attributes):
            encoded = attributes[name].encode("utf-8")
            handle.attrs.create(name, encoded, dtype=f"S{len(encoded)}")
        noise_group = handle.create_group("noise", track_order=False)
        noise_group.create_dataset(
            "observation",
            data=np.ascontiguousarray(observation_tape.noises, dtype="<c16"),
            dtype="<c16",
            chunks=None,
            compression=None,
            shuffle=False,
            fletcher32=False,
            track_times=False,
        )
        noise_group.create_dataset(
            "preparation",
            data=np.ascontiguousarray(preparation_tape.noises, dtype="<c16"),
            dtype="<c16",
            chunks=None,
            compression=None,
            shuffle=False,
            fletcher32=False,
            track_times=False,
        )
        group = handle.create_group("trajectory", track_order=False)
        trace = traces[0]
        states = trace.states
        arrival_index = trace.result.completed_preparation_steps
        datasets = (
            (
                "alpha_tilde",
                np.asarray(
                    tuple((state.alpha_tilde_x, state.alpha_tilde_y) for state in states),
                    dtype="<f8",
                ),
                "<f8",
            ),
            (
                "momenta",
                np.ascontiguousarray(
                    np.stack(tuple(state.momenta for state in states)),
                    dtype="<c16",
                ),
                "<c16",
            ),
            (
                "positions",
                np.ascontiguousarray(
                    np.stack(tuple(state.positions for state in states)),
                    dtype="<c16",
                ),
                "<c16",
            ),
            (
                "post_arrival_step",
                np.arange(len(states), dtype="<i8") - arrival_index,
                "<i8",
            ),
        )
        for name, value, dtype in datasets:
            group.create_dataset(
                name,
                data=value,
                dtype=dtype,
                chunks=None,
                compression=None,
                shuffle=False,
                fletcher32=False,
                track_times=False,
            )
    return buffer.getvalue()


def _validate_six_matrix_response_preparation_window_scheduling_hdf5_objects(
    handle: Any,
    *,
    multiplier: int,
    traces: tuple[SixMatrixResponsePreparationWindowSchedulingScheduleTraceResult, ...],
) -> None:
    import h5py

    if len(traces) != 1:
        raise ValueError("six-matrix preparation scheduling HDF5 validator requires one trajectory result")
    expected_groups = {"noise", "trajectory"}
    expected_datasets: dict[str, tuple[tuple[int, ...], np.dtype[np.generic]]] = {
        "noise/preparation": ((384 * multiplier, 2, 3, 4, 4), np.dtype("<c16")),
        "noise/observation": ((768 * multiplier, 2, 3, 4, 4), np.dtype("<c16")),
    }
    count = traces[0].trajectory_state_count
    expected_datasets.update(
        {
            "trajectory/alpha_tilde": ((count, 2), np.dtype("<f8")),
            "trajectory/momenta": ((count, 2, 3, 4, 4), np.dtype("<c16")),
            "trajectory/positions": ((count, 2, 3, 4, 4), np.dtype("<c16")),
            "trajectory/post_arrival_step": ((count,), np.dtype("<i8")),
        }
    )
    observed: set[str] = set()

    def validate_object(name: str, value: object) -> None:
        observed.add(name)
        link = handle.get(name, getlink=True)
        if not isinstance(link, h5py.HardLink):
            raise ValueError("six-matrix preparation scheduling measurement HDF5 forbids non-hard links")
        if name in expected_groups:
            if not isinstance(value, h5py.Group) or tuple(value.attrs):
                raise ValueError("six-matrix preparation scheduling measurement HDF5 group contract differs")
            return
        spec = expected_datasets.get(name)
        if spec is None or not isinstance(value, h5py.Dataset):
            raise ValueError("six-matrix preparation scheduling measurement HDF5 object inventory differs")
        shape, dtype = spec
        if (
            value.shape != shape
            or value.maxshape != shape
            or value.dtype != dtype
            or value.chunks is not None
            or value.compression is not None
            or value.shuffle
            or value.fletcher32
            or value.scaleoffset is not None
            or value.external
            or tuple(value.attrs)
        ):
            raise ValueError("six-matrix preparation scheduling measurement HDF5 dataset contract differs")

    handle.visititems(validate_object)
    if observed != expected_groups | set(expected_datasets):
        raise ValueError("six-matrix preparation scheduling measurement HDF5 object roster differs")


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingPersistedTrajectory:
    """One independently decoded action-homogeneous trajectory and root tapes."""

    result: SixMatrixResponsePreparationWindowSchedulingTrajectoryResult
    preparation_noise: np.ndarray
    observation_noise: np.ndarray
    trace: SixMatrixResponsePreparationWindowSchedulingPersistedScheduleTrace

    def __post_init__(self) -> None:
        multiplier = self.result.view_step_multiplier
        if (
            self.preparation_noise.shape != (384 * multiplier, 2, 3, 4, 4)
            or self.observation_noise.shape != (768 * multiplier, 2, 3, 4, 4)
            or self.preparation_noise.dtype != np.dtype("<c16")
            or self.observation_noise.dtype != np.dtype("<c16")
            or self.preparation_noise.flags.writeable
            or self.observation_noise.flags.writeable
            or self.trace.result != self.result.schedule_trace
        ):
            raise ValueError("six-matrix preparation scheduling persisted trajectory geometry or identity differs")
        if (
            _array_sha256(self.preparation_noise, dtype="<c16")
            != self.result.preparation_tape.noise_sha256
            or _array_sha256(self.observation_noise, dtype="<c16")
            != self.result.observation_tape.noise_sha256
        ):
            raise ValueError("six-matrix preparation scheduling persisted trajectory tapes differ from custody")


def decode_six_matrix_response_preparation_window_scheduling_trajectory_hdf5(
    payload: bytes,
    *,
    reservation: SixMatrixResponsePreparationWindowSchedulingTrajectoryArtifactReservation,
    result: SixMatrixResponsePreparationWindowSchedulingTrajectoryResult,
) -> SixMatrixResponsePreparationWindowSchedulingPersistedTrajectory:
    """Strictly reconstruct one trajectory in a fresh method process."""

    receipt = result.measurement_artifact
    if (
        not isinstance(payload, bytes)
        or receipt.reservation != reservation
        or len(payload) != receipt.size_bytes
        or len(payload) > reservation.maximum_bytes
        or sha256(payload).hexdigest() != receipt.content_sha256
    ):
        raise ValueError("six-matrix preparation scheduling trajectory payload differs from its custody receipt")
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - installed product dependency
        raise RuntimeError("six-matrix preparation scheduling HDF5 decoding requires h5py") from error
    expected_attributes = {
        "empirical_lawhood_clocks": '{"noise":"matrix-preparation-scheduling.root-tape-step","trajectory":"matrix-preparation-scheduling.integrator-step"}',
        "empirical_lawhood_frames": '{"noise":"standardized-hermitian-noise","trajectory":"six-matrix-native"}',
        "empirical_lawhood_keys": '["post_arrival_step"]',
        "empirical_lawhood_payload_schema": MATRIX_RESPONSE_PREPARATION_SCHEDULING_MEASUREMENT_HDF5_SCHEMA,
        "empirical_lawhood_units": '{"alpha_tilde":"1","momenta":"native","noise":"standardized","positions":"native","post_arrival_step":"step"}',
    }
    with h5py.File(BytesIO(payload), "r") as handle:
        if tuple(sorted(handle.attrs)) != tuple(sorted(expected_attributes)):
            raise ValueError("six-matrix preparation scheduling trajectory HDF5 root attributes differ")
        for name, expected in expected_attributes.items():
            observed = handle.attrs[name]
            if not isinstance(observed, (bytes, np.bytes_)) or bytes(observed).decode() != expected:
                raise ValueError("six-matrix preparation scheduling trajectory HDF5 attribute bytes differ")
        _validate_six_matrix_response_preparation_window_scheduling_hdf5_objects(
            handle,
            multiplier=result.view_step_multiplier,
            traces=(result.schedule_trace,),
        )
        preparation = _readonly_array(handle["/noise/preparation"][...], dtype="<c16")
        observation = _readonly_array(handle["/noise/observation"][...], dtype="<c16")
        prefix = "/trajectory"
        trace = SixMatrixResponsePreparationWindowSchedulingPersistedScheduleTrace(
            result=result.schedule_trace,
            positions=_readonly_array(handle[f"{prefix}/positions"][...], dtype="<c16"),
            momenta=_readonly_array(handle[f"{prefix}/momenta"][...], dtype="<c16"),
            alpha_tilde=_readonly_array(handle[f"{prefix}/alpha_tilde"][...], dtype="<f8"),
            post_arrival_steps=_readonly_array(
                handle[f"{prefix}/post_arrival_step"][...], dtype="<i8"
            ),
        )
    return SixMatrixResponsePreparationWindowSchedulingPersistedTrajectory(
        result=result,
        preparation_noise=preparation,
        observation_noise=observation,
        trace=trace,
    )


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingExecutedTrajectory:
    result: SixMatrixResponsePreparationWindowSchedulingTrajectoryResult
    measurement_payload: bytes
    trace: SixMatrixResponsePreparationWindowSchedulingScheduleTrace
    preparation_tape: SixMatrixResponsePreparationWindowSchedulingNoiseTape
    observation_tape: SixMatrixResponsePreparationWindowSchedulingNoiseTape

    def __post_init__(self) -> None:
        if (
            self.trace.result != self.result.schedule_trace
            or self.preparation_tape.receipt != self.result.preparation_tape
            or self.observation_tape.receipt != self.result.observation_tape
            or len(self.measurement_payload) != self.result.measurement_artifact.size_bytes
            or sha256(self.measurement_payload).hexdigest()
            != self.result.measurement_artifact.content_sha256
        ):
            raise ValueError("six-matrix preparation scheduling executed trajectory differs from its receipts")


def _splice_sha256(
    *,
    rule_id: str,
    schedule_id: str,
    numerical_view_id: str,
    preparation: ComplexArray,
    observation: ComplexArray,
) -> str:
    digest = sha256()
    for identity_bytes in (
        rule_id.encode("utf-8"),
        schedule_id.encode("utf-8"),
        numerical_view_id.encode("utf-8"),
    ):
        digest.update(len(identity_bytes).to_bytes(8, "big"))
        digest.update(identity_bytes)
    for noise_array in (preparation, observation):
        array = np.ascontiguousarray(noise_array, dtype="<c16")
        payload = array.tobytes(order="C")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _trajectory_sha256(
    states: tuple[SixMatrixState, ...],
    *,
    attribute: str,
) -> str:
    values = np.ascontiguousarray(
        np.stack(tuple(getattr(value, attribute) for value in states)),
        dtype="<c16",
    )
    return _array_sha256(values, dtype="<c16")


def execute_six_matrix_response_preparation_window_scheduling_schedule(
    *,
    config: SixMatrixResponsePreparationWindowSchedulingStudyConfig,
    request: SixMatrixResponsePreparationWindowSchedulingTrajectoryRequest,
    word: SixMatrixResponsePreparationWindowSchedulingPreparationWord,
    numerical_view: SixMatrixResponseNumericalView,
    view_step_multiplier: int,
    preparation_tape: SixMatrixResponsePreparationWindowSchedulingNoiseTape,
    observation_tape: SixMatrixResponsePreparationWindowSchedulingNoiseTape,
) -> SixMatrixResponsePreparationWindowSchedulingScheduleTrace:
    """Execute one word with supplied root-shared preparation/observation tapes."""

    if request.study_config != ObjectIdentity.from_record(config.config_id, config):
        raise ValueError("preparation executor requires the exact current issued study context")
    _require_preparation_scientific_inputs(
        request.scientific_inputs, root_block_id=request.root_block_id,
        seed_root_id=request.seed_root_id, block_index=request.block_index,
        current_context_sha256=config.fingerprint(),
    )
    if (
        preparation_tape.receipt.scientific_input_fingerprint != request.scientific_inputs[1].fingerprint()
        or observation_tape.receipt.scientific_input_fingerprint != request.scientific_inputs[2].fingerprint()
        or preparation_tape.receipt.coarse_seed_sha256 != request.scientific_inputs[1].full_seed_sha256
        or observation_tape.receipt.coarse_seed_sha256 != request.scientific_inputs[2].full_seed_sha256
        or preparation_tape.receipt.seed_root_id != request.seed_root_id
        or observation_tape.receipt.seed_root_id != request.seed_root_id
        or preparation_tape.receipt.stream_index != request.block_index
        or observation_tape.receipt.stream_index != request.block_index
    ):
        raise ValueError("preparation executor requires the complete original numeric tape census")
    gradient_cache_state = BAOABGradientCache()
    if word not in config.schedules or word.schedule_id not in request.schedule_ids:
        raise ValueError("six-matrix preparation scheduling schedule is outside the requested closed chart")
    if (
        view_step_multiplier != 1
        or preparation_tape.receipt.tape_kind is not SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind.PREPARATION
        or observation_tape.receipt.tape_kind is not SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind.OBSERVATION
        or preparation_tape.receipt.root_block_id != request.root_block_id
        or observation_tape.receipt.root_block_id != request.root_block_id
        or preparation_tape.receipt.view_step_multiplier != view_step_multiplier
        or observation_tape.receipt.view_step_multiplier != view_step_multiplier
        or preparation_tape.receipt.numerical_view_id != numerical_view.view_id
        or observation_tape.receipt.numerical_view_id != numerical_view.view_id
        or preparation_tape.receipt.coarse_seed_sha256
        == observation_tape.receipt.coarse_seed_sha256
        or np.shares_memory(preparation_tape.noises, observation_tape.noises)
    ):
        raise ValueError("six-matrix preparation scheduling schedule tapes are not the exact independent root pair")
    waveform = build_six_matrix_response_preparation_window_scheduling_preparation_waveform(
        word,
        view_step_multiplier=view_step_multiplier,
    )
    preparation_steps = waveform.alpha_tilde_x.size
    observation_steps = config.post_arrival_steps * view_step_multiplier
    if (
        preparation_steps > preparation_tape.noises.shape[0]
        or observation_steps > observation_tape.noises.shape[0]
    ):
        raise ValueError("six-matrix preparation scheduling shared tape is shorter than the schedule exposure")
    state = ideal_state(q=config.q, alpha_tilde_x=0.0, alpha_tilde_y=0.0, constitution="00")
    states = [state]
    completed_preparation = 0
    completed_observation = 0
    reasons: set[str] = set()
    for alpha_x, alpha_y, noise in zip(
        waveform.alpha_tilde_x,
        waveform.alpha_tilde_y,
        preparation_tape.noises[:preparation_steps],
        strict=True,
    ):
        try:
            state = baoab_step_with_hermitian_noise(
                state,
                member=config.family_member,
                numerical_view=numerical_view,
                next_alpha_tilde_x=float(alpha_x),
                next_alpha_tilde_y=float(alpha_y),
                standardized_noise=noise,
                gradient_cache=gradient_cache_state,
            )
        except (FloatingPointError, OverflowError):
            reasons.add("nonfinite-preparation-step")
            break
        states.append(state)
        completed_preparation += 1
        if not state.finite:
            reasons.add("nonfinite-preparation-state")
            break
    arrival_native: SixMatrixState | None = None
    if completed_preparation == preparation_steps and not reasons:
        arrival_native = state
        for noise in observation_tape.noises[:observation_steps]:
            try:
                state = baoab_step_with_hermitian_noise(
                    state,
                    member=config.family_member,
                    numerical_view=numerical_view,
                    next_alpha_tilde_x=float(config.target_alpha_tilde_x),
                    next_alpha_tilde_y=float(config.target_alpha_tilde_y),
                    standardized_noise=noise,
                    gradient_cache=gradient_cache_state,
                )
            except (FloatingPointError, OverflowError):
                reasons.add("nonfinite-observation-step")
                break
            states.append(state)
            completed_observation += 1
            if not state.finite:
                reasons.add("nonfinite-observation-state")
                break
    complete = (
        completed_preparation == preparation_steps
        and completed_observation == observation_steps
        and not reasons
    )
    realized_preparation_x = np.asarray(
        tuple(value.alpha_tilde_x for value in states[1 : completed_preparation + 1]),
        dtype="<f8",
    )
    realized_preparation_y = np.asarray(
        tuple(value.alpha_tilde_y for value in states[1 : completed_preparation + 1]),
        dtype="<f8",
    )
    realized_schedule_sha256 = _schedule_sha256(
        realized_preparation_x,
        realized_preparation_y,
    )
    applied_final = SixMatrixResponseScaledCouplings(
        _decimal(states[completed_preparation].alpha_tilde_x),
        _decimal(states[completed_preparation].alpha_tilde_y),
    )
    target = SixMatrixResponseScaledCouplings(
        config.target_alpha_tilde_x,
        config.target_alpha_tilde_y,
    )
    preparation_slice = preparation_tape.noises[:completed_preparation]
    observation_slice = observation_tape.noises[:completed_observation]
    ledger_reasons = tuple(sorted(reasons))
    ledger = SixMatrixResponsePreparationWindowSchedulingScheduleLedger(
        ledger_id=(f"ledger.{request.root_block_id}.{word.schedule_id}.{numerical_view.view_id}"),
        root_block_id=request.root_block_id,
        schedule_id=word.schedule_id,
        numerical_view_id=numerical_view.view_id,
        view_step_multiplier=view_step_multiplier,
        preparation_word=ObjectIdentity.from_record(word.schedule_id, word),
        requested_schedule_sha256=waveform.waveform_sha256,
        accepted_schedule_sha256=waveform.waveform_sha256,
        applied_schedule_sha256=realized_schedule_sha256,
        realized_schedule_sha256=realized_schedule_sha256,
        requested_arrival_step=preparation_steps,
        accepted_arrival_step=preparation_steps,
        applied_arrival_step=completed_preparation,
        realized_arrival_step=completed_preparation,
        requested_final=target,
        accepted_final=target,
        applied_final=applied_final,
        realized_final=applied_final,
        preparation_tape=ObjectIdentity.from_record(
            preparation_tape.receipt.tape_id,
            preparation_tape.receipt,
        ),
        preparation_consumed_start_step=0,
        preparation_consumed_stop_step=completed_preparation,
        observation_tape=ObjectIdentity.from_record(
            observation_tape.receipt.tape_id,
            observation_tape.receipt,
        ),
        observation_consumed_start_step=0,
        observation_consumed_stop_step=completed_observation,
        splice_rule_id=config.splice_rule_id,
        splice_sha256=_splice_sha256(
            rule_id=config.splice_rule_id,
            schedule_id=word.schedule_id,
            numerical_view_id=numerical_view.view_id,
            preparation=preparation_slice,
            observation=observation_slice,
        ),
        clipped=False,
        complete_delivery=complete,
        valid=complete,
        reason_codes=ledger_reasons,
        grants_authority=False,
    )
    state_tuple = tuple(states)
    arrival = (
        six_matrix_response_preparation_window_scheduling_phase_state_from_state(
            root_block_id=request.root_block_id,
            schedule_id=word.schedule_id,
            numerical_view_id=numerical_view.view_id,
            state_kind=SixMatrixResponsePreparationWindowSchedulingPhaseStateKind.ARRIVAL,
            state=arrival_native,
            post_arrival_step=0,
        )
        if arrival_native is not None
        else None
    )
    final = (
        six_matrix_response_preparation_window_scheduling_phase_state_from_state(
            root_block_id=request.root_block_id,
            schedule_id=word.schedule_id,
            numerical_view_id=numerical_view.view_id,
            state_kind=SixMatrixResponsePreparationWindowSchedulingPhaseStateKind.FINAL,
            state=state,
            post_arrival_step=completed_observation,
        )
        if complete
        else None
    )
    terminal = (
        SixMatrixResponsePreparationWindowSchedulingScheduleTerminal.COMPLETED if complete else SixMatrixResponsePreparationWindowSchedulingScheduleTerminal.NUMERICAL_INVALID
    )
    result = SixMatrixResponsePreparationWindowSchedulingScheduleTraceResult(
        trace_id=(f"trace.{request.root_block_id}.{word.schedule_id}.{numerical_view.view_id}"),
        root_block_id=request.root_block_id,
        block_index=request.block_index,
        schedule_id=word.schedule_id,
        numerical_view_id=numerical_view.view_id,
        view_step_multiplier=view_step_multiplier,
        terminal=terminal,
        completed_preparation_steps=completed_preparation,
        completed_observation_steps=completed_observation,
        trajectory_state_count=len(state_tuple),
        trajectory_positions_sha256=_trajectory_sha256(state_tuple, attribute="positions"),
        trajectory_momenta_sha256=_trajectory_sha256(state_tuple, attribute="momenta"),
        schedule_ledger=ledger,
        arrival_state=arrival,
        final_state=final,
        reason_codes=ledger_reasons,
    )
    return SixMatrixResponsePreparationWindowSchedulingScheduleTrace(word=word, states=state_tuple, result=result)


class SixMatrixResponsePreparationWindowSchedulingTrajectoryEngine:
    """Source-free one-request/one-action-homogeneous-trajectory engine."""

    def __init__(
        self,
        *,
        source_config: SixMatrixResponseSixMatrixSourceConfig,
        study_config: SixMatrixResponsePreparationWindowSchedulingStudyConfig,
        development_roster: SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster,
    ) -> None:
        if study_config.source_config != ObjectIdentity.from_record(
            source_config.config_id,
            source_config,
        ):
            raise ValueError("six-matrix preparation scheduling engine source/study identities differ")
        if study_config.family_member not in source_config.anisotropic_model.family_members:
            raise ValueError("six-matrix preparation scheduling engine member is absent from the source family")
        if study_config.primary_view != source_config.primary_view:
            raise ValueError("six-matrix preparation scheduling engine numerical views differ from the source")
        if (
            development_roster.study_config
            != ObjectIdentity.from_record(study_config.config_id, study_config)
            or development_roster.issue_id != study_config.development_issue_id
            or development_roster.seed_namespace_id != study_config.development_seed_namespace_id
            or development_roster.root_namespace_id != study_config.development_root_namespace_id
            or development_roster.numerical_member_id != study_config.primary_numerical_member_id
            or development_roster.numerical_view_id != study_config.primary_view.view_id
        ):
            raise ValueError("six-matrix preparation scheduling engine roster differs from its study config")
        self.source_config = source_config
        self.study_config = study_config
        self.development_roster = development_roster

    def _view(
        self,
        request: SixMatrixResponsePreparationWindowSchedulingTrajectoryRequest,
    ) -> tuple[SixMatrixResponseNumericalView, int]:
        if request.numerical_view_id == self.study_config.primary_view.view_id:
            return self.study_config.primary_view, 1
        raise ValueError("six-matrix preparation scheduling DEVELOPMENT requires the primary numerical view")

    def run_trajectory(
        self,
        request: SixMatrixResponsePreparationWindowSchedulingTrajectoryRequest,
    ) -> SixMatrixResponsePreparationWindowSchedulingExecutedTrajectory:
        """Execute one platform action-homogeneous trajectory task."""

        if (
            request.study_config
            != ObjectIdentity.from_record(self.study_config.config_id, self.study_config)
            or request.source_config != self.study_config.source_config
            or request.development_roster
            != ObjectIdentity.from_record(
                self.development_roster.roster_id,
                self.development_roster,
            )
            or request.issued_nested_view not in self.development_roster.views
        ):
            raise ValueError("six-matrix preparation scheduling trajectory config/roster differs from the engine")
        numerical_view, multiplier = self._view(request)
        if (
            request.tranche is not SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche.DEVELOPMENT
            or request.issue_id != self.study_config.development_issue_id
            or request.seed_namespace_id != self.study_config.development_seed_namespace_id
            or request.numerical_member_id != self.study_config.primary_numerical_member_id
            or request.root_block_id
            != self.development_roster.physical_independent_unit_ids[request.block_index]
            or multiplier != 1
        ):
            raise ValueError("six-matrix preparation scheduling trajectory lies outside issued DEVELOPMENT")
        _require_preparation_scientific_inputs(
            request.scientific_inputs, root_block_id=request.root_block_id,
            seed_root_id=request.seed_root_id, block_index=request.block_index,
            current_context_sha256=self.study_config.fingerprint(),
        )
        preparation = derive_six_matrix_response_preparation_window_scheduling_noise_tape(
            config=self.study_config,
            root_block_id=request.root_block_id,
            seed_root_id=request.seed_root_id,
            block_index=request.block_index,
            tape_kind=SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind.PREPARATION,
            scientific_seed_input=request.scientific_inputs[1],
        )
        observation = derive_six_matrix_response_preparation_window_scheduling_noise_tape(
            config=self.study_config,
            root_block_id=request.root_block_id,
            seed_root_id=request.seed_root_id,
            block_index=request.block_index,
            tape_kind=SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind.OBSERVATION,
            scientific_seed_input=request.scientific_inputs[2],
        )
        word = next(
            value
            for value in self.study_config.schedules
            if value.schedule_id == request.schedule_id
        )
        trace = execute_six_matrix_response_preparation_window_scheduling_schedule(
            config=self.study_config,
            request=request,
            word=word,
            numerical_view=numerical_view,
            view_step_multiplier=multiplier,
            preparation_tape=preparation,
            observation_tape=observation,
        )
        measurement_payload = _encode_six_matrix_response_preparation_window_scheduling_measurement_hdf5(
            request=request,
            numerical_view_id=numerical_view.view_id,
            view_step_multiplier=multiplier,
            traces=(trace,),
            preparation_tape=preparation,
            observation_tape=observation,
        )
        reservation = request.measurement_artifact_reservation
        if len(measurement_payload) > min(
            reservation.maximum_bytes,
            self.study_config.maximum_trajectory_hdf5_bytes,
        ):
            raise ValueError("six-matrix preparation scheduling trajectory exceeds its reserved shard ceiling")
        measurement_artifact = SixMatrixResponsePreparationWindowSchedulingMeasurementArtifactReceipt(
            receipt_id=f"matrix-preparation-scheduling.measurement-receipt.{request.task_id}",
            task_id=request.task_id,
            root_block_id=request.root_block_id,
            output_id=request.measurement_output_id,
            payload_schema=MATRIX_RESPONSE_PREPARATION_SCHEDULING_MEASUREMENT_HDF5_SCHEMA,
            media_type="application/x-hdf5",
            reservation=reservation,
            schedule_ids=(request.schedule_id,),
            dataset_paths=_measurement_dataset_paths(),
            content_sha256=sha256(measurement_payload).hexdigest(),
            size_bytes=len(measurement_payload),
            decoded_dataset_bytes=six_matrix_response_preparation_window_scheduling_trajectory_decoded_dataset_bytes(
                request.schedule_id, view_step_multiplier=multiplier
            ),
            grants_authority=False,
        )
        planned_updates = (
            word.final_arrival_step + self.study_config.post_arrival_steps
        ) * multiplier
        if planned_updates > self.study_config.maximum_trajectory_integration_steps:
            raise ValueError("six-matrix preparation scheduling trajectory exceeds its frozen integration ceiling")
        result = SixMatrixResponsePreparationWindowSchedulingTrajectoryResult(
            result_id=f"result.{request.task_id}",
            task_id=request.task_id,
            output_id=request.output_id,
            request=ObjectIdentity.from_record(request.request_id, request),
            root_block_id=request.root_block_id,
            block_index=request.block_index,
            tranche=request.tranche,
            issue_id=request.issue_id,
            seed_namespace_id=request.seed_namespace_id,
            development_roster=request.development_roster,
            issued_nested_view=request.issued_nested_view,
            schedule_id=request.schedule_id,
            numerical_view_id=numerical_view.view_id,
            numerical_member_id=request.numerical_member_id,
            scientific_view_id=request.scientific_view_id,
            physical_independent_unit_id=request.physical_independent_unit_id,
            acquisition_group_id=request.acquisition_group_id,
            view_step_multiplier=multiplier,
            root_seed_sha256=_root_seed_sha256(
                seed_root_id=request.seed_root_id,
                root_block_id=request.root_block_id,
                block_index=request.block_index,
                current_context_sha256=self.study_config.fingerprint(),
                scientific_seed_input=request.scientific_inputs[0],
            ),
            study_config=request.study_config,
            scientific_inputs=request.scientific_inputs,
            preparation_tape=preparation.receipt,
            observation_tape=observation.receipt,
            measurement_artifact=measurement_artifact,
            schedule_trace=trace.result,
            planned_integration_updates=planned_updates,
            completed_integration_updates=(
                trace.result.completed_preparation_steps + trace.result.completed_observation_steps
            ),
            terminal=trace.result.terminal,
            reason_codes=trace.result.reason_codes,
            evidence_ceiling=request.evidence_ceiling,
            outcome_access=request.outcome_access,
            visibility_ceiling=request.visibility_ceiling,
            scientific_verdict_ids=(),
            grants_authority=False,
        )
        if len(result.canonical_bytes()) > min(
            MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_RESULT_BYTES,
            self.study_config.maximum_trajectory_result_bytes,
        ):
            raise ValueError("six-matrix preparation scheduling trajectory result exceeds its output ceiling")
        return SixMatrixResponsePreparationWindowSchedulingExecutedTrajectory(
            result=result,
            measurement_payload=measurement_payload,
            trace=trace,
            preparation_tape=preparation,
            observation_tape=observation,
        )


__all__ = [
    "MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_CONFIG_BYTES",
    "MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_REQUEST_BYTES",
    "MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_RESULT_BYTES",
    "MATRIX_RESPONSE_PREPARATION_SCHEDULING_POST_ARRIVAL_STEPS",
    "MATRIX_RESPONSE_PREPARATION_SCHEDULING_PRIMARY_TIMESTEP",
    "MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS",
    "MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_X",
    "MATRIX_RESPONSE_PREPARATION_SCHEDULING_TARGET_ALPHA_TILDE_Y",
    'SixMatrixResponsePreparationWindowSchedulingExecutedTrajectory',
    'SixMatrixResponsePreparationWindowSchedulingMeasurementArtifactReceipt',
    'SixMatrixResponsePreparationWindowSchedulingNoiseTapeKind',
    'SixMatrixResponsePreparationWindowSchedulingNoiseTapeReceipt',
    'SixMatrixResponsePreparationWindowSchedulingNoiseTape',
    'SixMatrixResponsePreparationWindowSchedulingPhaseStateKind',
    'SixMatrixResponsePreparationWindowSchedulingPhaseState',
    'SixMatrixResponsePreparationWindowSchedulingPersistedScheduleTrace',
    'SixMatrixResponsePreparationWindowSchedulingPersistedTrajectory',
    'SixMatrixResponsePreparationWindowSchedulingPreparationWaveform',
    'SixMatrixResponsePreparationWindowSchedulingPreparationWord',
    'SixMatrixResponsePreparationWindowSchedulingTrajectoryEngine',
    'SixMatrixResponsePreparationWindowSchedulingScheduleLedger',
    'SixMatrixResponsePreparationWindowSchedulingScheduleTerminal',
    'SixMatrixResponsePreparationWindowSchedulingScheduleTraceResult',
    'SixMatrixResponsePreparationWindowSchedulingScheduleTrace',
    'SixMatrixResponsePreparationWindowSchedulingStudyConfig',
    'SixMatrixResponsePreparationWindowSchedulingTrajectoryRequest',
    'SixMatrixResponsePreparationWindowSchedulingTrajectoryResult',
    'SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche',
    'build_six_matrix_response_preparation_window_scheduling_preparation_waveform',
    'build_six_matrix_response_preparation_window_scheduling_study_config',
    'six_matrix_response_preparation_window_scheduling_phase_state_from_state',
    'six_matrix_response_preparation_window_scheduling_preparation_words',
    'six_matrix_response_preparation_window_scheduling_trajectory_decoded_dataset_bytes',
    'decode_six_matrix_response_preparation_window_scheduling_trajectory_hdf5',
    'decode_six_matrix_response_preparation_window_scheduling_trajectory_request',
    'decode_six_matrix_response_preparation_window_scheduling_trajectory_result',
    'decode_six_matrix_response_preparation_window_scheduling_study_config',
    'derive_six_matrix_response_preparation_window_scheduling_noise_tape',
    'execute_six_matrix_response_preparation_window_scheduling_schedule',
]
