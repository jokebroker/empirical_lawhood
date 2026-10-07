"""Non-pooling two-target recurrence and restrictiveness adjudication."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .contracts import TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID
from .target_adjudication import TargetConstructValidationPredictionAxis, TargetConstructValidationRestrictivenessAxis, TargetConstructValidationTargetHandoff


class TargetConstructValidationRecurrenceDisposition(StrEnum):
    SUPPORTED = "TWO_TARGET_RECURRENCE_SUPPORTED"
    MIXED = "TWO_TARGET_RECURRENCE_MIXED"
    OPPOSED = "TWO_TARGET_RECURRENCE_OPPOSED"
    UNEVALUABLE = "TWO_TARGET_RECURRENCE_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationCrossTargetAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-cross-target-adjudication'

    adjudication_id: str
    primary_relation_id: str
    target_handoffs: tuple[ObjectIdentity, ...]
    target_ids: tuple[str, ...]
    recurrence_disposition: TargetConstructValidationRecurrenceDisposition
    restrictiveness_disposition: TargetConstructValidationRestrictivenessAxis
    reason_codes: tuple[str, ...]
    pooled_coefficient_count: int
    pooled_threshold_count: int
    pooled_sample_count: int
    native_numeric_value_count: int
    raw_payload_reference_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        validate_stable_id(self.primary_relation_id, field_name="primary_relation_id")
        require_sorted_unique_strings(
            self.target_ids,
            field_name="target_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if len(self.target_handoffs) != 2 or len(self.target_ids) != 2:
            raise ValueError("cross-target adjudication requires exactly two handoffs")
        if self.primary_relation_id != TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID:
            raise ValueError("cross-target adjudication names a nonprimary relation")
        if any(
            (
                self.pooled_coefficient_count,
                self.pooled_threshold_count,
                self.pooled_sample_count,
                self.native_numeric_value_count,
                self.raw_payload_reference_count,
            )
        ):
            raise ValueError("cross-target adjudication cannot pool or expose native payloads")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("cross-target adjudication requires revealed handoffs")


def adjudicate_cross_target(
    first: TargetConstructValidationTargetHandoff,
    second: TargetConstructValidationTargetHandoff,
) -> TargetConstructValidationCrossTargetAdjudication:
    """Apply frozen counterexample precedence to two compact handoffs."""

    values = tuple(sorted((first, second), key=lambda value: value.target_id))
    if values[0].target_id == values[1].target_id:
        raise ValueError("INSUFFICIENT_ELIGIBLE_TARGETS: duplicate target")
    for attribute in ("domain_id", "solver_family_id", "generator_family_id"):
        if getattr(values[0], attribute) == getattr(values[1], attribute):
            raise ValueError(f"INSUFFICIENT_ELIGIBLE_TARGETS: duplicate {attribute}")
    if any(not value.eligible_for_cross_target for value in values):
        raise ValueError("INSUFFICIENT_ELIGIBLE_TARGETS: ineligible handoff")
    if {value.primary_relation_id for value in values} != {TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID}:
        raise ValueError("CROSS_TARGET_RELATION_NOT_PREDECLARED")

    reasons: set[str] = set()
    if any(value.exact_construct_valid_counterexample for value in values):
        recurrence = TargetConstructValidationRecurrenceDisposition.OPPOSED
        reasons.add("EXACT_CONSTRUCT_VALID_COUNTEREXAMPLE")
    elif all(value.prediction_axis is TargetConstructValidationPredictionAxis.SUPPORTED for value in values):
        recurrence = TargetConstructValidationRecurrenceDisposition.SUPPORTED
    elif all(value.prediction_axis is TargetConstructValidationPredictionAxis.OPPOSED for value in values):
        recurrence = TargetConstructValidationRecurrenceDisposition.OPPOSED
        reasons.add("BOTH_TARGET_PREDICTIONS_OPPOSED")
    else:
        recurrence = TargetConstructValidationRecurrenceDisposition.MIXED
        reasons.add("TARGET_PREDICTION_DISPOSITIONS_DIFFER")

    restrictive = tuple(value.restrictiveness_axis for value in values)
    if all(value is TargetConstructValidationRestrictivenessAxis.SUPPORTED for value in restrictive):
        restrictiveness = TargetConstructValidationRestrictivenessAxis.SUPPORTED
    elif any(value is TargetConstructValidationRestrictivenessAxis.OPPOSED for value in restrictive):
        restrictiveness = TargetConstructValidationRestrictivenessAxis.OPPOSED
        reasons.add("TARGET_RESTRICTIVENESS_OPPOSED")
    else:
        restrictiveness = TargetConstructValidationRestrictivenessAxis.NOT_DISTINGUISHED
        reasons.add("TARGET_RESTRICTIVENESS_NOT_DISTINGUISHED")

    identities = tuple(ObjectIdentity.from_record(value.handoff_id, value) for value in values)
    return TargetConstructValidationCrossTargetAdjudication(
        adjudication_id="target-construct-validation.two-target-adjudication",
        primary_relation_id=TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID,
        target_handoffs=identities,
        target_ids=tuple(value.target_id for value in values),
        recurrence_disposition=recurrence,
        restrictiveness_disposition=restrictiveness,
        reason_codes=tuple(sorted(reasons)),
        pooled_coefficient_count=0,
        pooled_threshold_count=0,
        pooled_sample_count=0,
        native_numeric_value_count=0,
        raw_payload_reference_count=0,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
