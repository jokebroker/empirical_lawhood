"Plan-scoped structural-response-class contracts for structural response-class assessment.\n\n``INTERFACE_STATE_DEPENDENT_GLOBALIZATION`` is deliberately not added to the\nfrozen kernel structural-class enumeration.  Membership belongs to a complete\nprepared response context and requires a noncompensating conjunction of eight\naxes.  These records also prevent a deliberately matched circuit world from\nbeing counted as independent recurrence in matter.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Sequence

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


ISDG_SIGNATURE_ID = "interface-state-dependent-globalization"
ISDG_AXIS_IDS = (
    "a1.valid-delivery-material-local-response",
    "a2.accurate-supported-local-sections",
    "a3.minimal-measured-interface-globalizes",
    "a4.ablation-excess-global-failure-local-eligible",
    "a5.causal-interface-localization",
    "a6.frozen-interface-restoration-rescues",
    "a7.receiver-unit-clock-gauge-counterfeits-rejected",
    "a8.transformation-preparation-lot-stability",
)


class AxisDisposition(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class StructuralMechanismDisposition(StrEnum):
    COMPATIBLE_GLOBALIZATION = "COMPATIBLE_GLOBALIZATION"
    RECEIVER_COBOUNDARY_ONLY = "RECEIVER_COBOUNDARY_ONLY"
    INTERFACE_STATE_DEPENDENT_GLOBALIZATION = (
        "INTERFACE_STATE_DEPENDENT_GLOBALIZATION"
    )
    PERSISTENT_OBSTRUCTION_CONTRAST = "PERSISTENT_OBSTRUCTION_CONTRAST"
    APPARENT_ANALOGY_ONLY = "APPARENT_ANALOGY_ONLY"
    INVALID = "INVALID"
    UNEVALUABLE = "UNEVALUABLE"


class StructuralClassOutcome(StrEnum):
    STRUCTURAL_CLASS_CANDIDATE_SUPPORTED = "STRUCTURAL_CLASS_CANDIDATE_SUPPORTED"
    SUBSTRATE_CLASS_ONLY = "SUBSTRATE_CLASS_ONLY"
    LOCAL_SIGNATURE_ONLY = "LOCAL_SIGNATURE_ONLY"
    ENGINEERED_REALIZATION_ONLY = "ENGINEERED_REALIZATION_ONLY"
    STRUCTURAL_CLASS_SIGNATURE_OPPOSED = "STRUCTURAL_CLASS_SIGNATURE_OPPOSED"
    APPARENT_ANALOGY_ONLY = "APPARENT_ANALOGY_ONLY"
    STRUCTURAL_CLASS_UNEVALUABLE = "STRUCTURAL_CLASS_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class StructuralAxisDefinition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-axis-definition'

    axis_id: str
    estimand_id: str
    pass_rule: str
    falsifier_ids: tuple[str, ...]
    native_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.axis_id, field_name="axis_id")
        validate_stable_id(self.estimand_id, field_name="estimand_id")
        validate_nonempty(self.pass_rule, field_name="pass_rule")
        require_sorted_unique_strings(self.falsifier_ids, field_name="falsifier_ids")
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class StructuralResponseSignature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-response-signature'

    signature_id: str
    axes: tuple[StructuralAxisDefinition, ...]
    required_contrast_ids: tuple[str, ...]
    evidence_ceiling: str
    membership_key: str
    maximum_promotion: str

    def __post_init__(self) -> None:
        if self.signature_id != ISDG_SIGNATURE_ID:
            raise ValueError("Structural response-class assessment exposes only the frozen ISDG signature")
        require_sorted_unique_ids(self.axes, attribute="axis_id", field_name="axes")
        if tuple(axis.axis_id for axis in self.axes) != ISDG_AXIS_IDS:
            raise ValueError("ISDG requires exactly the eight frozen axes")
        require_sorted_unique_strings(
            self.required_contrast_ids,
            field_name="required_contrast_ids",
        )
        validate_nonempty(self.evidence_ceiling, field_name="evidence_ceiling")
        if self.membership_key != "PREPARED_RESPONSE_CONTEXT":
            raise ValueError("substrate-name membership is forbidden")
        if self.maximum_promotion != "STRUCTURAL_CLASS_CANDIDATE":
            raise ValueError("Structural response-class assessment cannot expose a universality promotion")


@dataclass(frozen=True, slots=True)
class StructuralClassNomination(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-class-nomination'

    nomination_id: str
    signature_id: str
    source_context_ids: tuple[str, ...]
    source_evidence_act_ids: tuple[str, ...]
    information_cutoff: str
    retrospective_only: bool
    excluded_promotion_ids: tuple[str, ...]
    counterfeit_dispositions: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.nomination_id, field_name="nomination_id")
        if self.signature_id != ISDG_SIGNATURE_ID:
            raise ValueError("nomination signature differs")
        require_sorted_unique_strings(self.source_context_ids, field_name="source_context_ids")
        require_sorted_unique_strings(
            self.source_evidence_act_ids,
            field_name="source_evidence_act_ids",
        )
        if len(self.source_context_ids) != len(self.source_evidence_act_ids):
            raise ValueError("source context/evidence-act cardinalities differ")
        validate_nonempty(self.information_cutoff, field_name="information_cutoff")
        require_sorted_unique_strings(
            self.excluded_promotion_ids,
            field_name="excluded_promotion_ids",
        )
        if "structural-universality-class" not in self.excluded_promotion_ids:
            raise ValueError("universality must be excluded explicitly")
        require_sorted_unique_strings(
            self.counterfeit_dispositions,
            field_name="counterfeit_dispositions",
        )


@dataclass(frozen=True, slots=True)
class StructuralClassTransportWitness(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-class-transport-witness'

    witness_id: str
    source_context_id: str
    target_context_id: str
    map_ids: tuple[str, ...]
    nontransported_coordinate_ids: tuple[str, ...]
    support_map_valid: bool
    uncertainty_map_valid: bool
    independently_prepared_target: bool
    target_is_matched_twin: bool
    counts_as_independent_member: bool
    native_coefficient_pooling_permitted: bool
    status: str

    def __post_init__(self) -> None:
        for name, value in (
            ("witness_id", self.witness_id),
            ("source_context_id", self.source_context_id),
            ("target_context_id", self.target_context_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.map_ids, field_name="map_ids")
        require_sorted_unique_strings(
            self.nontransported_coordinate_ids,
            field_name="nontransported_coordinate_ids",
            allow_empty=True,
        )
        validate_nonempty(self.status, field_name="status")
        if self.native_coefficient_pooling_permitted:
            raise ValueError("cross-substrate native coefficient pooling is forbidden")
        if self.target_is_matched_twin and self.counts_as_independent_member:
            raise ValueError("a deliberately matched twin is not independent recurrence")
        if self.counts_as_independent_member and not self.independently_prepared_target:
            raise ValueError("independent membership requires independent preparation")


@dataclass(frozen=True, slots=True)
class ClassConditionedPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/class-conditioned-prediction'

    prediction_id: str
    signature_id: str
    protected_target_id: str
    target_binding_sha256: str
    prediction_sha256: str
    predicted_disposition: StructuralMechanismDisposition
    localization_id: str
    interface_subset_id: str
    scoring_rule_id: str
    abstention_reason_ids: tuple[str, ...]
    issued_before_target_outcomes: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("prediction_id", self.prediction_id),
            ("protected_target_id", self.protected_target_id),
            ("localization_id", self.localization_id),
            ("interface_subset_id", self.interface_subset_id),
            ("scoring_rule_id", self.scoring_rule_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.signature_id != ISDG_SIGNATURE_ID:
            raise ValueError("prediction signature differs")
        validate_sha256(self.target_binding_sha256, field_name="target_binding_sha256")
        validate_sha256(self.prediction_sha256, field_name="prediction_sha256")
        require_sorted_unique_strings(
            self.abstention_reason_ids,
            field_name="abstention_reason_ids",
            allow_empty=True,
        )
        if not self.issued_before_target_outcomes:
            raise ValueError("outcome-visible predictions cannot support promotion")


@dataclass(frozen=True, slots=True)
class StructuralAxisResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-axis-result'

    axis_id: str
    disposition: AxisDisposition
    estimate: Decimal | None
    threshold: Decimal | None
    independent_unit_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.axis_id not in ISDG_AXIS_IDS:
            raise ValueError("unknown ISDG axis")
        if self.estimate is not None:
            validate_decimal(self.estimate, field_name="estimate")
        if self.threshold is not None:
            validate_decimal(self.threshold, field_name="threshold", minimum=Decimal("0"))
        if self.independent_unit_count < 0:
            raise ValueError("independent-unit count must be nonnegative")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes", allow_empty=True)
        if self.disposition is not AxisDisposition.PASS and not self.reason_codes:
            raise ValueError("nonpassing axis requires a reason")


@dataclass(frozen=True, slots=True)
class LotRecurrenceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/lot-recurrence-result'

    lot_id: str
    independent_assembly_count: int
    direction_supported: bool
    valid: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.lot_id, field_name="lot_id")
        if self.independent_assembly_count < 1:
            raise ValueError("lot recurrence requires at least one assembly")
        if self.direction_supported and not self.valid:
            raise ValueError("an invalid lot cannot support recurrence")


@dataclass(frozen=True, slots=True)
class StructuralClassAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-class-adjudication'

    adjudication_id: str
    context_id: str
    signature_id: str
    axis_results: tuple[StructuralAxisResult, ...]
    mechanism_disposition: StructuralMechanismDisposition
    lot_results: tuple[LotRecurrenceResult, ...]
    withheld_prediction_passed: bool | None
    transport_valid: bool
    independent_prospective_member: bool
    source_member_eligible: bool
    outcome: StructuralClassOutcome
    cohomology_disposition: str
    missingness_reason_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        validate_stable_id(self.context_id, field_name="context_id")
        if self.signature_id != ISDG_SIGNATURE_ID:
            raise ValueError("adjudication signature differs")
        require_sorted_unique_ids(
            self.axis_results,
            attribute="axis_id",
            field_name="axis_results",
        )
        if tuple(result.axis_id for result in self.axis_results) != ISDG_AXIS_IDS:
            raise ValueError("adjudication requires exactly eight ordered axes")
        require_sorted_unique_ids(self.lot_results, attribute="lot_id", field_name="lot_results")
        require_sorted_unique_strings(
            self.missingness_reason_ids,
            field_name="missingness_reason_ids",
            allow_empty=True,
        )
        validate_nonempty(self.cohomology_disposition, field_name="cohomology_disposition")
        all_axes_pass = all(result.disposition is AxisDisposition.PASS for result in self.axis_results)
        if (
            self.mechanism_disposition
            is StructuralMechanismDisposition.INTERFACE_STATE_DEPENDENT_GLOBALIZATION
        ) != all_axes_pass:
            raise ValueError("ISDG disposition and noncompensating axes disagree")
        if self.outcome is StructuralClassOutcome.STRUCTURAL_CLASS_CANDIDATE_SUPPORTED:
            if not (
                all_axes_pass
                and self.transport_valid
                and self.independent_prospective_member
                and self.source_member_eligible
                and self.withheld_prediction_passed is True
                and len(self.lot_results) >= 2
                and all(item.valid and item.direction_supported for item in self.lot_results)
            ):
                raise ValueError("structural-class candidate prerequisites are incomplete")


def adjudicate_structural_class(
    *,
    adjudication_id: str,
    context_id: str,
    axis_results: Sequence[StructuralAxisResult],
    mechanism_disposition: StructuralMechanismDisposition,
    lot_results: Sequence[LotRecurrenceResult] = (),
    withheld_prediction_passed: bool | None = None,
    transport_valid: bool = False,
    independent_prospective_member: bool = False,
    source_member_eligible: bool = False,
    engineered_target: bool = False,
    cohomology_disposition: str = "NOT_APPLICABLE",
    missingness_reason_ids: Sequence[str] = (),
) -> StructuralClassAdjudication:
    """Apply the frozen noncompensating structural ladder."""

    ordered_axes = tuple(sorted(axis_results, key=lambda item: item.axis_id))
    ordered_lots = tuple(sorted(lot_results, key=lambda item: item.lot_id))
    all_pass = (
        len(ordered_axes) == len(ISDG_AXIS_IDS)
        and all(item.disposition is AxisDisposition.PASS for item in ordered_axes)
    )
    has_invalid = any(item.disposition is AxisDisposition.INVALID for item in ordered_axes)
    has_unevaluable = any(
        item.disposition is AxisDisposition.UNEVALUABLE for item in ordered_axes
    )
    lots_recur = len(ordered_lots) >= 2 and all(
        item.valid and item.direction_supported for item in ordered_lots
    )
    if has_invalid or has_unevaluable:
        outcome = StructuralClassOutcome.STRUCTURAL_CLASS_UNEVALUABLE
    elif not all_pass:
        outcome = (
            StructuralClassOutcome.APPARENT_ANALOGY_ONLY
            if mechanism_disposition is StructuralMechanismDisposition.APPARENT_ANALOGY_ONLY
            else StructuralClassOutcome.STRUCTURAL_CLASS_SIGNATURE_OPPOSED
        )
    elif (
        transport_valid
        and independent_prospective_member
        and source_member_eligible
        and withheld_prediction_passed is True
        and lots_recur
    ):
        outcome = StructuralClassOutcome.STRUCTURAL_CLASS_CANDIDATE_SUPPORTED
    elif engineered_target:
        outcome = StructuralClassOutcome.ENGINEERED_REALIZATION_ONLY
    elif lots_recur:
        outcome = StructuralClassOutcome.SUBSTRATE_CLASS_ONLY
    else:
        outcome = StructuralClassOutcome.LOCAL_SIGNATURE_ONLY
    return StructuralClassAdjudication(
        adjudication_id=adjudication_id,
        context_id=context_id,
        signature_id=ISDG_SIGNATURE_ID,
        axis_results=ordered_axes,
        mechanism_disposition=mechanism_disposition,
        lot_results=ordered_lots,
        withheld_prediction_passed=withheld_prediction_passed,
        transport_valid=transport_valid,
        independent_prospective_member=independent_prospective_member,
        source_member_eligible=source_member_eligible,
        outcome=outcome,
        cohomology_disposition=cohomology_disposition,
        missingness_reason_ids=tuple(sorted(set(missingness_reason_ids))),
    )


__all__ = [
    "ISDG_AXIS_IDS",
    "ISDG_SIGNATURE_ID",
    "AxisDisposition",
    "ClassConditionedPrediction",
    "LotRecurrenceResult",
    "StructuralAxisDefinition",
    "StructuralAxisResult",
    "StructuralClassAdjudication",
    "StructuralClassNomination",
    "StructuralClassOutcome",
    "StructuralClassTransportWitness",
    "StructuralMechanismDisposition",
    "StructuralResponseSignature",
    "adjudicate_structural_class",
]
