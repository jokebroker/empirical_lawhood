"Additive correction of the retained cross-target disposition.\n\nThe retained executable cross-target record kept prediction recurrence separate\nfrom restrictiveness. The issued target construct validation plan requires both\ntarget predictions and the comparator conjunction before recurrence can be\ncalled supported. This module preserves that record and emits a compact\ninterpretation record; it never reopens target-native payloads.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .cross_target import TargetConstructValidationCrossTargetAdjudication, TargetConstructValidationRecurrenceDisposition
from .target_adjudication import TargetConstructValidationPredictionAxis, TargetConstructValidationRestrictivenessAxis, TargetConstructValidationTargetHandoff


class TargetConstructValidationCorrectedRecurrenceDisposition(StrEnum):
    SUPPORTED = "TWO_TARGET_RECURRENCE_SUPPORTED"
    NOT_DISTINGUISHED = "TWO_TARGET_RECURRENCE_NOT_DISTINGUISHED"
    MIXED = "TWO_TARGET_RECURRENCE_MIXED"
    OPPOSED = "TWO_TARGET_RECURRENCE_OPPOSED"
    UNEVALUABLE = "TWO_TARGET_RECURRENCE_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationCrossTargetInterpretationCorrection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-cross-target-interpretation-correction'

    correction_id: str
    source_adjudication: ObjectIdentity
    target_handoffs: tuple[ObjectIdentity, ...]
    target_ids: tuple[str, ...]
    stored_recurrence_disposition: TargetConstructValidationRecurrenceDisposition
    corrected_recurrence_disposition: TargetConstructValidationCorrectedRecurrenceDisposition
    restrictiveness_disposition: TargetConstructValidationRestrictivenessAxis
    predeclared_support_conjunction_met: bool
    reason_codes: tuple[str, ...]
    pooled_coefficient_count: int
    pooled_threshold_count: int
    pooled_sample_count: int
    native_numeric_value_count: int
    raw_payload_reference_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.correction_id, field_name="correction_id")
        require_sorted_unique_ids(
            self.target_handoffs,
            attribute="object_id",
            field_name="target_handoffs",
        )
        require_sorted_unique_strings(
            self.target_ids,
            field_name="target_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if len(self.target_handoffs) != 2 or len(self.target_ids) != 2:
            raise ValueError("cross-target correction requires exactly two handoffs")
        if self.predeclared_support_conjunction_met != (
            self.corrected_recurrence_disposition is TargetConstructValidationCorrectedRecurrenceDisposition.SUPPORTED
        ):
            raise ValueError("corrected support is not conjunction-derived")
        if (
            self.corrected_recurrence_disposition is TargetConstructValidationCorrectedRecurrenceDisposition.SUPPORTED
            and self.restrictiveness_disposition is not TargetConstructValidationRestrictivenessAxis.SUPPORTED
        ):
            raise ValueError("recurrence support requires predictive restrictiveness")
        if any(
            (
                self.pooled_coefficient_count,
                self.pooled_threshold_count,
                self.pooled_sample_count,
                self.native_numeric_value_count,
                self.raw_payload_reference_count,
            )
        ):
            raise ValueError("cross-target correction cannot pool or expose native payloads")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("cross-target correction requires revealed handoffs")


def correct_cross_target_interpretation(
    source: TargetConstructValidationCrossTargetAdjudication,
    first: TargetConstructValidationTargetHandoff,
    second: TargetConstructValidationTargetHandoff,
) -> TargetConstructValidationCrossTargetInterpretationCorrection:
    """Apply the predeclared prediction-plus-comparator support conjunction."""

    values = tuple(sorted((first, second), key=lambda value: value.target_id))
    identities = tuple(ObjectIdentity.from_record(value.handoff_id, value) for value in values)
    if identities != source.target_handoffs or tuple(value.target_id for value in values) != (
        source.target_ids
    ):
        raise ValueError("correction handoffs differ from the source adjudication")

    reasons: set[str] = set()
    prediction = tuple(value.prediction_axis for value in values)
    restrictive = tuple(value.restrictiveness_axis for value in values)
    if any(value.exact_construct_valid_counterexample for value in values):
        corrected = TargetConstructValidationCorrectedRecurrenceDisposition.OPPOSED
        reasons.add("EXACT_CONSTRUCT_VALID_COUNTEREXAMPLE")
    elif any(value is TargetConstructValidationPredictionAxis.UNEVALUABLE for value in prediction) or any(
        value is TargetConstructValidationRestrictivenessAxis.UNEVALUABLE for value in restrictive
    ):
        corrected = TargetConstructValidationCorrectedRecurrenceDisposition.UNEVALUABLE
        reasons.add("TARGET_AXIS_UNEVALUABLE")
    elif all(value is TargetConstructValidationPredictionAxis.SUPPORTED for value in prediction):
        if all(value is TargetConstructValidationRestrictivenessAxis.SUPPORTED for value in restrictive):
            corrected = TargetConstructValidationCorrectedRecurrenceDisposition.SUPPORTED
        else:
            corrected = TargetConstructValidationCorrectedRecurrenceDisposition.NOT_DISTINGUISHED
            reasons.add("TARGET_REGULARITY_WITHOUT_DISTINCTIVE_STRUCTURAL_RECURRENCE_VALUE")
    elif all(value is TargetConstructValidationPredictionAxis.OPPOSED for value in prediction):
        corrected = TargetConstructValidationCorrectedRecurrenceDisposition.OPPOSED
        reasons.add("BOTH_TARGET_PREDICTIONS_OPPOSED")
    else:
        corrected = TargetConstructValidationCorrectedRecurrenceDisposition.MIXED
        reasons.add("TARGET_PREDICTION_DISPOSITIONS_DIFFER")

    stored_as_corrected = TargetConstructValidationCorrectedRecurrenceDisposition(source.recurrence_disposition.value)
    if stored_as_corrected is not corrected:
        reasons.add("STORED_RECURRENCE_LABEL_DID_NOT_ENFORCE_PREDECLARED_COMPARATOR_CONJUNCTION")
    return TargetConstructValidationCrossTargetInterpretationCorrection(
        correction_id="target-construct-validation.two-target-interpretation-correction",
        source_adjudication=ObjectIdentity.from_record(source.adjudication_id, source),
        target_handoffs=identities,
        target_ids=source.target_ids,
        stored_recurrence_disposition=source.recurrence_disposition,
        corrected_recurrence_disposition=corrected,
        restrictiveness_disposition=source.restrictiveness_disposition,
        predeclared_support_conjunction_met=(
            corrected is TargetConstructValidationCorrectedRecurrenceDisposition.SUPPORTED
        ),
        reason_codes=tuple(sorted(reasons)),
        pooled_coefficient_count=0,
        pooled_threshold_count=0,
        pooled_sample_count=0,
        native_numeric_value_count=0,
        raw_payload_reference_count=0,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


__all__ = [
    'TargetConstructValidationCorrectedRecurrenceDisposition',
    'TargetConstructValidationCrossTargetInterpretationCorrection',
    "correct_cross_target_interpretation",
]
