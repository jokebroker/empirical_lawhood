"""Typed contracts for the CubeSpec fine-steering-mirror empirical slice."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.atlases import AtlasGap
from empirical_lawhood.kernel.evidence import EvidenceRung, VisibilityCeiling
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus, ScientificStatus


class FineSteeringMirrorTransportStatus(StrEnum):
    """Whether simulator-to-physical transport was part of the slice."""

    NOT_APPLICABLE_NO_SIMULATOR = "NOT_APPLICABLE_NO_SIMULATOR"


@dataclass(frozen=True, slots=True)
class FineSteeringMirrorObservationOperatorSpec(CanonicalRecord):
    """Frozen map from source arrays to one block-level frequency response."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/fine-steering-mirror/fine-steering-mirror-observation-operator-spec'

    operator_id: str
    sample_count: int
    sampling_frequency_hz: Decimal
    input_channels: int
    output_channels: int
    realizations_per_block: int
    periods_per_realization: int
    frequency_band_hz: tuple[Decimal, Decimal]
    period_aggregation: str
    matrix_estimator: str
    receiver_estimator: str

    def __post_init__(self) -> None:
        validate_stable_id(self.operator_id, field_name="operator_id")
        for name, value in (
            ("sample_count", self.sample_count),
            ("input_channels", self.input_channels),
            ("output_channels", self.output_channels),
            ("realizations_per_block", self.realizations_per_block),
            ("periods_per_realization", self.periods_per_realization),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        validate_decimal(
            self.sampling_frequency_hz,
            field_name="sampling_frequency_hz",
            minimum=Decimal("0"),
        )
        lower, upper = self.frequency_band_hz
        validate_decimal(lower, field_name="frequency_band_hz[0]", minimum=Decimal("0"))
        validate_decimal(upper, field_name="frequency_band_hz[1]", minimum=lower)
        if lower == upper:
            raise ValueError("frequency band must have positive width")
        if self.period_aggregation != "complex_arithmetic_mean":
            raise ValueError("Fine-steering-mirror slice requires the frozen complex period mean")
        if self.matrix_estimator != "output_fft_times_inverse_input_fft":
            raise ValueError("Fine-steering-mirror slice requires the frozen square orthogonal-block estimator")
        if self.receiver_estimator != "row_l2_norm_argmax":
            raise ValueError("Fine-steering-mirror slice requires the frozen row-norm peak estimator")


@dataclass(frozen=True, slots=True)
class FineSteeringMirrorProtocolSpec(CanonicalRecord):
    """Complete development-frozen empirical protocol."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/fine-steering-mirror/fine-steering-mirror-protocol-spec'

    protocol_id: str
    source_config_sha256: str
    source_audit_sha256: str
    development_acquisition_sha256: str
    observation_operator: FineSteeringMirrorObservationOperatorSpec
    action_levels_volts: tuple[Decimal, ...]
    primary_receiver_index: int
    maximum_input_matrix_condition: Decimal
    maximum_period_relative_difference: Decimal
    maximum_within_action_peak_range_hz: Decimal
    maximum_heldout_absolute_error_hz: Decimal
    minimum_primary_wrong_action_advantage_hz: Decimal
    maximum_primary_slope_hz_per_v: Decimal
    minimum_law_qualification_evaluation_blocks_per_action: int
    minimum_law_qualification_denominator_exchanges: int
    expected_development_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.protocol_id, field_name="protocol_id")
        for sha_name, digest in (
            ("source_config_sha256", self.source_config_sha256),
            ("source_audit_sha256", self.source_audit_sha256),
            ("development_acquisition_sha256", self.development_acquisition_sha256),
        ):
            validate_sha256(digest, field_name=sha_name)
        if tuple(sorted(set(self.action_levels_volts))) != self.action_levels_volts:
            raise ValueError("action levels must be sorted and unique")
        if len(self.action_levels_volts) != 3:
            raise ValueError("Fine-steering-mirror slice requires the exact three-level action chart")
        for index, level in enumerate(self.action_levels_volts):
            validate_decimal(level, field_name=f"action_levels_volts[{index}]")
        if self.primary_receiver_index not in range(self.observation_operator.output_channels):
            raise ValueError("primary receiver index is out of range")
        self._validate_thresholds()
        self._validate_evidence_counts()

    def _validate_thresholds(self) -> None:
        for gate_name, gate_value in (
            ("maximum_input_matrix_condition", self.maximum_input_matrix_condition),
            ("maximum_period_relative_difference", self.maximum_period_relative_difference),
            ("maximum_within_action_peak_range_hz", self.maximum_within_action_peak_range_hz),
            ("maximum_heldout_absolute_error_hz", self.maximum_heldout_absolute_error_hz),
            (
                "minimum_primary_wrong_action_advantage_hz",
                self.minimum_primary_wrong_action_advantage_hz,
            ),
        ):
            validate_decimal(gate_value, field_name=gate_name, minimum=Decimal("0"))
        validate_decimal(
            self.maximum_primary_slope_hz_per_v,
            field_name="maximum_primary_slope_hz_per_v",
        )
        if self.maximum_primary_slope_hz_per_v >= 0:
            raise ValueError("the predeclared primary action response must have negative slope")

    def _validate_evidence_counts(self) -> None:
        if self.minimum_law_qualification_evaluation_blocks_per_action < 2:
            raise ValueError("Local law recurrence cannot require fewer than two evaluation blocks")
        if self.minimum_law_qualification_denominator_exchanges < 1:
            raise ValueError("Local law coordinate adequacy requires a denominator exchange")
        if len(self.expected_development_sha256) != 6:
            raise ValueError("protocol must bind the six development arrays")
        require_sorted_unique_strings(
            self.expected_development_sha256,
            field_name="expected_development_sha256",
            allow_empty=False,
        )
        for index, digest in enumerate(self.expected_development_sha256):
            validate_sha256(digest, field_name=f"expected_development_sha256[{index}]")


@dataclass(frozen=True, slots=True)
class FineSteeringMirrorBlockResponse(CanonicalRecord):
    """One physical independent block experiment reduced to its receiver vector."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/fine-steering-mirror/fine-steering-mirror-block-response'

    block_id: str
    action_level_volts: Decimal
    receiver_peak_hz: tuple[Decimal, Decimal, Decimal]
    maximum_input_matrix_condition: Decimal
    period_relative_difference: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.block_id, field_name="block_id")
        validate_decimal(self.action_level_volts, field_name="action_level_volts")
        for index, value in enumerate(self.receiver_peak_hz):
            validate_decimal(value, field_name=f"receiver_peak_hz[{index}]")
        validate_decimal(
            self.maximum_input_matrix_condition,
            field_name="maximum_input_matrix_condition",
            minimum=Decimal("0"),
        )
        validate_decimal(
            self.period_relative_difference,
            field_name="period_relative_difference",
            minimum=Decimal("0"),
        )


@dataclass(frozen=True, slots=True)
class FineSteeringMirrorDevelopmentModel(CanonicalRecord):
    """Frozen linear susceptibility summary fitted only on development blocks."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/fine-steering-mirror/fine-steering-mirror-development-model'

    model_id: str
    intercept_hz: tuple[Decimal, Decimal, Decimal]
    slope_hz_per_v: tuple[Decimal, Decimal, Decimal]
    training_rmse_hz: tuple[Decimal, Decimal, Decimal]
    leave_one_block_max_error_hz: tuple[Decimal, Decimal, Decimal]
    reversed_action_rmse_hz: tuple[Decimal, Decimal, Decimal]
    prediction_hz_by_action: tuple[tuple[Decimal, Decimal, Decimal], ...]
    fit_block_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.model_id, field_name="model_id")
        require_sorted_unique_strings(
            self.fit_block_ids, field_name="fit_block_ids", allow_empty=False
        )
        if len(self.prediction_hz_by_action) != 3:
            raise ValueError("model must commit one receiver prediction per action level")


@dataclass(frozen=True, slots=True)
class FineSteeringMirrorRungAssessment(CanonicalRecord):
    "One explicit evidence-stage status with orthogonal readiness and science axes."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/fine-steering-mirror/fine-steering-mirror-rung-assessment'

    assessment_id: str
    rung: EvidenceRung
    scientific_status: ScientificStatus
    readiness_status: ReadinessStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class FineSteeringMirrorAdversarialAudit(CanonicalRecord):
    """Independent construction-oriented audit of the immutable slice result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/fine-steering-mirror/fine-steering-mirror-adversarial-audit'

    audit_id: str
    check_ids: tuple[str, ...]
    failed_check_ids: tuple[str, ...]
    conclusion_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        require_sorted_unique_strings(self.check_ids, field_name="check_ids", allow_empty=False)
        require_sorted_unique_strings(self.failed_check_ids, field_name="failed_check_ids")
        require_sorted_unique_strings(
            self.conclusion_codes, field_name="conclusion_codes", allow_empty=False
        )
        unknown = set(self.failed_check_ids) - set(self.check_ids)
        if unknown:
            raise ValueError(f"failed audit checks are not registered: {sorted(unknown)}")


@dataclass(frozen=True, slots=True)
class FineSteeringMirrorVerticalSliceResult(CanonicalRecord):
    """Terminal physical evidence result; it cannot imply a missing law/controller."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/fine-steering-mirror/fine-steering-mirror-vertical-slice-result'

    result_id: str
    system_id: str
    protocol_sha256: str
    freeze_commit: str
    source_member_sha256: tuple[str, ...]
    development_model: FineSteeringMirrorDevelopmentModel
    evaluation_blocks: tuple[FineSteeringMirrorBlockResponse, ...]
    heldout_absolute_error_hz: tuple[tuple[Decimal, Decimal, Decimal], ...]
    primary_correct_action_rmse_hz: Decimal
    primary_reversed_action_rmse_hz: Decimal
    rung_assessments: tuple[FineSteeringMirrorRungAssessment, ...]
    atlas_gap: AtlasGap
    transport_status: FineSteeringMirrorTransportStatus
    controller_status: ScientificStatus
    visibility_ceiling: VisibilityCeiling
    adversarial_audit: FineSteeringMirrorAdversarialAudit

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.system_id, field_name="system_id")
        validate_sha256(self.protocol_sha256, field_name="protocol_sha256")
        if len(self.freeze_commit) != 40:
            raise ValueError("freeze commit must be a full Git SHA-1")
        require_sorted_unique_strings(
            self.source_member_sha256,
            field_name="source_member_sha256",
            allow_empty=False,
        )
        for index, digest in enumerate(self.source_member_sha256):
            validate_sha256(digest, field_name=f"source_member_sha256[{index}]")
        require_sorted_unique_ids(
            self.evaluation_blocks,
            attribute="block_id",
            field_name="evaluation_blocks",
        )
        if len(self.heldout_absolute_error_hz) != len(self.evaluation_blocks):
            raise ValueError("held-out errors must align one-to-one with evaluation blocks")
        validate_decimal(
            self.primary_correct_action_rmse_hz,
            field_name="primary_correct_action_rmse_hz",
            minimum=Decimal("0"),
        )
        validate_decimal(
            self.primary_reversed_action_rmse_hz,
            field_name="primary_reversed_action_rmse_hz",
            minimum=Decimal("0"),
        )
        require_sorted_unique_ids(
            self.rung_assessments,
            attribute="assessment_id",
            field_name="rung_assessments",
        )
        if tuple(item.rung for item in self.rung_assessments) != tuple(EvidenceRung):
            raise ValueError("vertical slice must retain exactly one assessment for every ordered evidence stage")
        if self.controller_status is not ScientificStatus.NOT_TESTED:
            raise ValueError("Fine-steering-mirror slice has no authority or admission basis for a controller")
        if self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE:
            raise ValueError("revealed public evaluation results are outcome-visible")
