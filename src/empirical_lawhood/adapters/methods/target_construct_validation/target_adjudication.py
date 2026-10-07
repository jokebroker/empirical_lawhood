"""Independent target-local axes and compact cross-target handoff."""

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


class TargetConstructValidationConstructAxis(StrEnum):
    PASSED = "CONSTRUCT_VALIDITY_PASSED"
    FAILED = "CONSTRUCT_VALIDITY_FAILED"
    UNEVALUABLE = "CONSTRUCT_VALIDITY_UNEVALUABLE"


class TargetConstructValidationPredictionAxis(StrEnum):
    SUPPORTED = "TARGET_PREDICTION_SUPPORTED"
    MIXED = "TARGET_PREDICTION_MIXED"
    OPPOSED = "TARGET_PREDICTION_OPPOSED"
    UNEVALUABLE = "TARGET_PREDICTION_UNEVALUABLE"


class TargetConstructValidationRestrictivenessAxis(StrEnum):
    SUPPORTED = "PREDICTIVE_RESTRICTIVENESS_SUPPORTED"
    NOT_DISTINGUISHED = "PREDICTIVE_RESTRICTIVENESS_NOT_DISTINGUISHED"
    OPPOSED = "PREDICTIVE_RESTRICTIVENESS_OPPOSED"
    UNEVALUABLE = "PREDICTIVE_RESTRICTIVENESS_UNEVALUABLE"


class TargetConstructValidationTopologyAxis(StrEnum):
    TOPOLOGY_STRONGER = "TOPOLOGY_STRONGER"
    EQUIVALENT_OR_UNEVALUABLE = "EQUIVALENT_OR_UNEVALUABLE"
    METRIC_STRONGER_OR_TOPOLOGY_OPPOSED = "METRIC_STRONGER_OR_TOPOLOGY_OPPOSED"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationTargetAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-adjudication'

    adjudication_id: str
    target_id: str
    primary_relation_id: str
    construct_axis: TargetConstructValidationConstructAxis
    prediction_axis: TargetConstructValidationPredictionAxis
    restrictiveness_axis: TargetConstructValidationRestrictivenessAxis
    topology_axis: TargetConstructValidationTopologyAxis
    exact_construct_valid_counterexample: bool
    complete_unit_inference_closed: bool
    power_adequate: bool
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("adjudication_id", "target_id", "primary_relation_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.primary_relation_id != TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID:
            raise ValueError("target adjudication names a nonprimary relation")
        if self.construct_axis is not TargetConstructValidationConstructAxis.PASSED:
            if self.prediction_axis is not TargetConstructValidationPredictionAxis.UNEVALUABLE:
                raise ValueError("prediction is unevaluable when construct validity fails")
            if self.restrictiveness_axis is not TargetConstructValidationRestrictivenessAxis.UNEVALUABLE:
                raise ValueError("restrictiveness is unevaluable when construct validity fails")
        if not self.complete_unit_inference_closed or not self.power_adequate:
            if self.prediction_axis is not TargetConstructValidationPredictionAxis.UNEVALUABLE:
                raise ValueError("prediction requires complete-unit inference and power")
            if self.restrictiveness_axis is not TargetConstructValidationRestrictivenessAxis.UNEVALUABLE:
                raise ValueError("restrictiveness requires complete-unit inference and power")
        if self.exact_construct_valid_counterexample:
            if self.construct_axis is not TargetConstructValidationConstructAxis.PASSED:
                raise ValueError("a decisive counterexample must first be construct-valid")
            if self.prediction_axis is not TargetConstructValidationPredictionAxis.OPPOSED:
                raise ValueError("a decisive counterexample has precedence over support")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("target adjudication requires revealed evaluation records")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationTargetHandoff(CanonicalRecord):
    """Compact categorical handoff; native numeric values are forbidden."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-handoff'

    handoff_id: str
    target_id: str
    domain_id: str
    solver_family_id: str
    generator_family_id: str
    primary_relation_id: str
    relation_binding: ObjectIdentity
    target_adjudication: ObjectIdentity
    construct_axis: TargetConstructValidationConstructAxis
    prediction_axis: TargetConstructValidationPredictionAxis
    restrictiveness_axis: TargetConstructValidationRestrictivenessAxis
    topology_axis: TargetConstructValidationTopologyAxis
    exact_construct_valid_counterexample: bool
    eligible_for_cross_target: bool
    reason_codes: tuple[str, ...]
    native_numeric_value_count: int
    raw_payload_reference_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "handoff_id",
            "target_id",
            "domain_id",
            "solver_family_id",
            "generator_family_id",
            "primary_relation_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.primary_relation_id != TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID:
            raise ValueError("target handoff names a nonprimary relation")
        if self.native_numeric_value_count or self.raw_payload_reference_count:
            raise ValueError("cross-target handoff cannot expose native numbers or payloads")
        expected_eligible = self.construct_axis is TargetConstructValidationConstructAxis.PASSED and (
            self.prediction_axis is not TargetConstructValidationPredictionAxis.UNEVALUABLE
            and self.restrictiveness_axis is not TargetConstructValidationRestrictivenessAxis.UNEVALUABLE
        )
        if self.eligible_for_cross_target != expected_eligible:
            raise ValueError("cross-target eligibility is not axis-derived")
        if self.exact_construct_valid_counterexample and (
            self.prediction_axis is not TargetConstructValidationPredictionAxis.OPPOSED
        ):
            raise ValueError("counterexample handoff must retain opposition")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("handoff can be built only after target adjudication")
