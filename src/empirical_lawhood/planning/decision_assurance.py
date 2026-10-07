"""Complete-unit, representation-aware decision assurance records."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryCellDisposition, MetatheoryEvidenceCeiling, MetatheoryMethodSelection


class DecisionAssuranceApplicability(StrEnum):
    REFERENCE_COMPARISON_REQUIRED = "REFERENCE_COMPARISON_REQUIRED"
    DIRECT_NATIVE_COMPLETE_UNIT = "DIRECT_NATIVE_COMPLETE_UNIT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class DecisionDisposition(StrEnum):
    ACTIVE_ACTION = "ACTIVE_ACTION"
    HOLD = "HOLD"
    NONATTEMPT = "NONATTEMPT"
    UNSAFE = "UNSAFE"
    UNEVALUABLE = "UNEVALUABLE"


class DecisionErrorKind(StrEnum):
    FALSE_ADMISSION = "FALSE_ADMISSION"
    FALSE_SAFE_HOLD = "FALSE_SAFE_HOLD"
    FALSE_HOLD = "FALSE_HOLD"
    FALSE_NONATTEMPT = "FALSE_NONATTEMPT"
    ACTION_SUBSTITUTION = "ACTION_SUBSTITUTION"
    DECISION_MISMATCH = "DECISION_MISMATCH"


@dataclass(frozen=True, slots=True)
class DecisionAssuranceTarget(CanonicalRecord):
    """One predeclared complete-unit/action-fibre comparison cell."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/decision-assurance-target'

    target_id: str
    physical_unit_id: str
    action_fibre_id: str
    member_id: str
    view_id: str

    def __post_init__(self) -> None:
        for name in (
            "target_id",
            "physical_unit_id",
            "action_fibre_id",
            "member_id",
            "view_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class DecisionErrorRule(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/decision-error-rule'

    error_kind: DecisionErrorKind
    decisive_veto: bool
    maximum_complete_unit_rate: NamedDecimal
    interval_rule_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.interval_rule_id, field_name="interval_rule_id")
        if self.maximum_complete_unit_rate.unit != "1" or not (
            Decimal(0) <= self.maximum_complete_unit_rate.value <= Decimal(1)
        ):
            raise ValueError("decision error maximum rate must be a fraction")


@dataclass(frozen=True, slots=True)
class DecisionAssuranceSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/decision-assurance-spec'

    spec_id: str
    applicability: DecisionAssuranceApplicability
    applicability_reason_codes: tuple[str, ...]
    candidate_representation: ObjectIdentity | None
    reference_representation: ObjectIdentity | None
    direct_native_proof: ObjectIdentity | None
    target_semantics_id: str | None
    sink_semantics_id: str | None
    gate_semantics_id: str | None
    direction_ids: tuple[str, ...]
    physical_unit_ids: tuple[str, ...]
    targets: tuple[DecisionAssuranceTarget, ...]
    candidate_evaluator: MetatheoryMethodSelection | None
    reference_evaluator: MetatheoryMethodSelection | None
    uncertainty_method: MetatheoryMethodSelection | None
    comparison_reveal_route: ObjectIdentity | None
    error_rules: tuple[DecisionErrorRule, ...]
    multiplicity_rule_id: str | None
    aggregation_rule_id: str | None
    measured_hold_semantics_id: str | None
    nonattempt_semantics_id: str | None
    falsifier_ids: tuple[str, ...]
    maximum_ordinary_evidence_ceiling: EvidenceCeiling
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        require_sorted_unique_strings(
            self.applicability_reason_codes,
            field_name="applicability_reason_codes",
        )
        require_sorted_unique_strings(
            self.direction_ids,
            field_name="direction_ids",
        )
        require_sorted_unique_strings(
            self.physical_unit_ids,
            field_name="physical_unit_ids",
        )
        require_sorted_unique_ids(self.targets, attribute="target_id", field_name="targets")
        require_sorted_unique_ids(
            self.error_rules,
            attribute="error_kind",
            field_name="error_rules",
        )
        require_sorted_unique_strings(self.falsifier_ids, field_name="falsifier_ids")
        required_visibility = inherited_visibility((), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(required_visibility):
            raise ValueError("decision assurance visibility understates outcome access")

        if self.applicability is DecisionAssuranceApplicability.NOT_APPLICABLE:
            self._validate_not_applicable()
            return
        self._validate_decision_semantics()
        if self.applicability is DecisionAssuranceApplicability.DIRECT_NATIVE_COMPLETE_UNIT:
            self._validate_direct_native()
        else:
            self._validate_reference_comparison()

    def _validate_not_applicable(self) -> None:
        if not self.applicability_reason_codes:
            raise ValueError("N/A decision assurance requires typed reasons")
        executing = (
            self.candidate_representation,
            self.reference_representation,
            self.direct_native_proof,
            self.target_semantics_id,
            self.sink_semantics_id,
            self.gate_semantics_id,
            self.candidate_evaluator,
            self.reference_evaluator,
            self.uncertainty_method,
            self.comparison_reveal_route,
            self.multiplicity_rule_id,
            self.aggregation_rule_id,
            self.measured_hold_semantics_id,
            self.nonattempt_semantics_id,
        )
        if any(value is not None for value in executing) or any(
            (self.direction_ids, self.physical_unit_ids, self.targets, self.error_rules)
        ):
            raise ValueError("N/A decision assurance must remain nonexecuting")

    def _validate_decision_semantics(self) -> None:
        if self.applicability_reason_codes:
            raise ValueError("applicable decision assurance cannot carry N/A reasons")
        for name in (
            "target_semantics_id",
            "sink_semantics_id",
            "gate_semantics_id",
            "multiplicity_rule_id",
            "aggregation_rule_id",
            "measured_hold_semantics_id",
            "nonattempt_semantics_id",
        ):
            value = getattr(self, name)
            if value is None:
                raise ValueError(f"applicable decision assurance lacks {name}")
            validate_stable_id(value, field_name=name)
        if self.candidate_representation is None or self.candidate_evaluator is None:
            raise ValueError("applicable decision assurance lacks candidate binding")
        if not self.direction_ids or not self.physical_unit_ids or not self.targets:
            raise ValueError("applicable decision assurance requires complete targets")
        target_units = {value.physical_unit_id for value in self.targets}
        if target_units != set(self.physical_unit_ids):
            raise ValueError("decision targets do not cover exact physical units")

    def _validate_direct_native(self) -> None:
        if self.direct_native_proof is None:
            raise ValueError("direct-native decision assurance requires exact proof")
        if (
            any(
                value is not None
                for value in (
                    self.reference_representation,
                    self.reference_evaluator,
                    self.uncertainty_method,
                    self.comparison_reveal_route,
                )
            )
            or self.error_rules
        ):
            raise ValueError("direct-native applicability cannot invent a comparison")

    def _validate_reference_comparison(self) -> None:
        if self.direct_native_proof is not None:
            raise ValueError("reference comparison cannot carry direct-native proof")
        if any(
            value is None
            for value in (
                self.reference_representation,
                self.reference_evaluator,
                self.uncertainty_method,
                self.comparison_reveal_route,
            )
        ):
            raise ValueError("reference comparison lacks sealed evaluator binding")
        if {value.error_kind for value in self.error_rules} != set(DecisionErrorKind):
            raise ValueError("reference comparison requires every decision error rule")


@dataclass(frozen=True, slots=True)
class DecisionComparisonEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/decision-comparison-evidence'

    evidence_id: str
    assurance_spec: ObjectIdentity
    target: ObjectIdentity
    physical_unit_id: str
    action_fibre_id: str
    candidate_decision: DecisionDisposition
    reference_decision: DecisionDisposition
    candidate_action_occurrence: ObjectIdentity | None
    reference_action_occurrence: ObjectIdentity | None
    candidate_delivery: ObjectIdentity | None
    reference_delivery: ObjectIdentity | None
    candidate_gate_operands: ObjectIdentity
    reference_gate_operands: ObjectIdentity
    reference_hold_supported_and_viable: bool | None
    candidate_evaluator: MetatheoryMethodSelection
    reference_evaluator: MetatheoryMethodSelection
    uncertainty_method: MetatheoryMethodSelection
    publication: ObjectIdentity
    recovery: ObjectIdentity
    evidence_links: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        if self.assurance_spec.object_schema != DecisionAssuranceSpec.SCHEMA:
            raise ValueError("decision evidence names another assurance schema")
        if self.target.object_schema != DecisionAssuranceTarget.SCHEMA:
            raise ValueError("decision evidence names another target schema")
        validate_stable_id(self.physical_unit_id, field_name="physical_unit_id")
        validate_stable_id(self.action_fibre_id, field_name="action_fibre_id")
        for decision, occurrence, delivery, role in (
            (
                self.candidate_decision,
                self.candidate_action_occurrence,
                self.candidate_delivery,
                "candidate",
            ),
            (
                self.reference_decision,
                self.reference_action_occurrence,
                self.reference_delivery,
                "reference",
            ),
        ):
            if decision is DecisionDisposition.ACTIVE_ACTION:
                if occurrence is None or delivery is None:
                    raise ValueError(f"active {role} decision lacks occurrence/delivery")
            elif occurrence is not None or delivery is not None:
                raise ValueError(f"nonactive {role} decision cannot carry occurrence/delivery")
        if self.reference_decision is DecisionDisposition.HOLD:
            if self.reference_hold_supported_and_viable is None:
                raise ValueError("reference HOLD requires independent viability status")
        elif self.reference_hold_supported_and_viable is not None:
            raise ValueError("non-HOLD reference cannot carry HOLD viability")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )


@dataclass(frozen=True, slots=True)
class DecisionAssuranceCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/decision-assurance-cell'

    cell_id: str
    assurance_spec: ObjectIdentity
    target: ObjectIdentity
    evidence: ObjectIdentity
    physical_unit_id: str
    action_fibre_id: str
    candidate_decision: DecisionDisposition
    reference_decision: DecisionDisposition
    error_kinds: tuple[DecisionErrorKind, ...]
    decisive_veto: bool
    disposition: MetatheoryCellDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        validate_stable_id(self.physical_unit_id, field_name="physical_unit_id")
        validate_stable_id(self.action_fibre_id, field_name="action_fibre_id")
        if tuple(sorted(set(self.error_kinds), key=lambda value: value.value)) != self.error_kinds:
            raise ValueError("decision error kinds must be sorted and unique")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is MetatheoryCellDisposition.SUPPORTED:
            if self.error_kinds or self.decisive_veto or self.reason_codes:
                raise ValueError("supported decision cell cannot carry errors")
        elif self.disposition is MetatheoryCellDisposition.OPPOSED:
            if not self.error_kinds or not self.reason_codes:
                raise ValueError("opposed decision cell requires derived errors")
        elif self.disposition is MetatheoryCellDisposition.UNEVALUABLE:
            if self.error_kinds or self.decisive_veto or not self.reason_codes:
                raise ValueError("unevaluable decision cell cannot assert decision errors")
        else:
            raise ValueError("comparison cell cannot be NOT_APPLICABLE")


@dataclass(frozen=True, slots=True)
class DecisionErrorInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/decision-error-interval'

    error_kind: DecisionErrorKind
    complete_unit_count: int
    lower_rate: NamedDecimal
    upper_rate: NamedDecimal
    interval_rule_id: str

    def __post_init__(self) -> None:
        if type(self.complete_unit_count) is not int or self.complete_unit_count < 0:
            raise ValueError("decision interval count must be nonnegative")
        validate_stable_id(self.interval_rule_id, field_name="interval_rule_id")
        for value in (self.lower_rate, self.upper_rate):
            if value.unit != "1" or not (Decimal(0) <= value.value <= Decimal(1)):
                raise ValueError("decision interval rate must be a fraction")
        if self.lower_rate.value > self.upper_rate.value:
            raise ValueError("decision interval lower rate exceeds upper rate")


@dataclass(frozen=True, slots=True)
class DecisionAssuranceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/decision-assurance-result'

    result_id: str
    assurance_spec: ObjectIdentity
    applicability: DecisionAssuranceApplicability
    applicability_proof: ObjectIdentity | None
    comparison_performed: bool
    evidence: tuple[DecisionComparisonEvidence, ...]
    cells: tuple[DecisionAssuranceCell, ...]
    complete_physical_unit_count: int
    false_admission_count: int | None
    false_safe_hold_count: int | None
    false_hold_count: int | None
    false_nonattempt_count: int | None
    action_substitution_count: int | None
    decision_mismatch_count: int | None
    error_intervals: tuple[DecisionErrorInterval, ...]
    decisive_veto_cell_ids: tuple[str, ...]
    disposition: MetatheoryAggregateDisposition
    achieved_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    evidence_links: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_sorted_unique_ids(self.evidence, attribute="evidence_id", field_name="evidence")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_ids(
            self.error_intervals,
            attribute="error_kind",
            field_name="error_intervals",
        )
        require_sorted_unique_strings(
            self.decisive_veto_cell_ids,
            field_name="decisive_veto_cell_ids",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )
        if (
            type(self.complete_physical_unit_count) is not int
            or self.complete_physical_unit_count < 0
        ):
            raise ValueError("decision result complete-unit count must be nonnegative")
        counts = (
            self.false_admission_count,
            self.false_safe_hold_count,
            self.false_hold_count,
            self.false_nonattempt_count,
            self.action_substitution_count,
            self.decision_mismatch_count,
        )
        if self.comparison_performed:
            if (
                self.applicability
                is not DecisionAssuranceApplicability.REFERENCE_COMPARISON_REQUIRED
            ):
                raise ValueError("only reference mode may perform decision comparison")
            if (
                self.applicability_proof is not None
                or not self.cells
                or len(self.cells) != len(self.evidence)
            ):
                raise ValueError("decision comparison result has inconsistent evidence")
            if any(type(value) is not int or value < 0 for value in counts):
                raise ValueError("decision comparison counts must be nonnegative integers")
            if len(self.error_intervals) != len(DecisionErrorKind):
                raise ValueError("decision comparison lacks error intervals")
        else:
            if self.evidence or self.cells or self.error_intervals or self.decisive_veto_cell_ids:
                raise ValueError("applicability result cannot contain comparison evidence")
            if any(value is not None for value in counts):
                raise ValueError("applicability proof cannot masquerade as zero-error comparison")
            if self.applicability is DecisionAssuranceApplicability.DIRECT_NATIVE_COMPLETE_UNIT:
                if (
                    self.applicability_proof is None
                    or self.disposition is not MetatheoryAggregateDisposition.SUPPORTED
                ):
                    raise ValueError("direct-native result requires supported applicability proof")
            elif (
                self.applicability_proof is not None
                or self.disposition is not MetatheoryAggregateDisposition.PREREQUISITE_NONATTEMPT
            ):
                raise ValueError("N/A decision result must remain a prerequisite nonattempt")


__all__ = [
    'DecisionAssuranceApplicability',
    'DecisionAssuranceCell',
    'DecisionAssuranceResult',
    'DecisionAssuranceSpec',
    'DecisionAssuranceTarget',
    'DecisionComparisonEvidence',
    'DecisionDisposition',
    'DecisionErrorInterval',
    'DecisionErrorKind',
    'DecisionErrorRule',
]
