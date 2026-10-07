"""Pure validation and decisions for an issued static fixed-round graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import EvidenceCeiling, EvidenceRung, OutcomeAccess
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.planning.adaptive_acquisition import AcquisitionCausalDomain, ActionPreparationRecurrenceQualificationRequirement, BoundedQueryAcquisitionPlan, FixedRoundAcquisitionPlan


class FixedRoundDisposition(StrEnum):
    SELECTED = "SELECTED"
    SKIPPED = "SKIPPED"


@dataclass(frozen=True, slots=True)
class FixedRoundArmAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/fixed-round-arm-assessment'

    assessment_id: str
    arm_id: str
    eligible: bool
    priority_numerator: int
    priority_denominator: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.arm_id, field_name="arm_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.priority_denominator < 1:
            raise ValueError("fixed-round priority denominator must be positive")
        if self.eligible == bool(self.reason_codes):
            raise ValueError("fixed-round eligibility and reasons are inconsistent")


@dataclass(frozen=True, slots=True)
class FixedRoundDecisionItem(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/fixed-round-decision-item'

    arm_id: str
    disposition: FixedRoundDisposition
    selection_ordinal: int | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.arm_id, field_name="arm_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is FixedRoundDisposition.SELECTED:
            if self.selection_ordinal is None or self.selection_ordinal < 1 or self.reason_codes:
                raise ValueError("selected fixed-round arm is malformed")
        elif self.selection_ordinal is not None or not self.reason_codes:
            raise ValueError("skipped fixed-round arm is malformed")


@dataclass(frozen=True, slots=True)
class FixedRoundAcquisitionDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/fixed-round-acquisition-decision'

    decision_id: str
    plan: ObjectIdentity
    round_number: int
    checkpoint: ObjectIdentity | None
    items: tuple[FixedRoundDecisionItem, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        if self.round_number < 1:
            raise ValueError("fixed-round number must be positive")
        require_sorted_unique_ids(self.items, attribute="arm_id", field_name="items")


@dataclass(frozen=True, slots=True)
class FixedRoundAcquisitionReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/fixed-round-acquisition-receipt'

    receipt_id: str
    decision: ObjectIdentity
    round_number: int
    completed_arm_ids: tuple[str, ...]
    skipped_arm_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(
            self.completed_arm_ids,
            field_name="completed_arm_ids",
        )
        require_sorted_unique_strings(self.skipped_arm_ids, field_name="skipped_arm_ids")
        if set(self.completed_arm_ids) & set(self.skipped_arm_ids):
            raise ValueError("fixed-round receipt both completes and skips one arm")


@dataclass(frozen=True, slots=True)
class FixedRoundCompletenessReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/fixed-round-completeness-receipt'

    receipt_id: str
    plan: ObjectIdentity
    round_receipts: tuple[ObjectIdentity, ...]
    completed_round_count: int
    complete: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        receipt_ids = tuple(value.object_id for value in self.round_receipts)
        if len(set(receipt_ids)) != len(receipt_ids):
            raise ValueError("fixed-round completeness receipt identities are duplicated")
        if self.completed_round_count != len(self.round_receipts) or not self.complete:
            raise ValueError("fixed-round completeness receipt is incomplete")


def select_fixed_round(
    *,
    plan: FixedRoundAcquisitionPlan,
    round_number: int,
    assessments: tuple[FixedRoundArmAssessment, ...],
    prior_selection_counts: tuple[tuple[str, int], ...],
    checkpoint: ObjectIdentity | None = None,
) -> FixedRoundAcquisitionDecision:
    """Return a deterministic decision only; this function cannot create a task."""

    if not 1 <= round_number <= plan.maximum_rounds:
        raise ValueError("fixed-round decision is outside the issued round roster")
    if plan.checkpoint_ids:
        if checkpoint is None or checkpoint.object_id != plan.checkpoint_ids[round_number - 1]:
            raise ValueError("fixed-round decision binds another checkpoint")
    elif checkpoint is not None:
        raise ValueError("exhaust-round decision cannot bind a checkpoint")
    require_sorted_unique_ids(
        assessments,
        attribute="arm_id",
        field_name="assessments",
    )
    if tuple(value.arm_id for value in assessments) != tuple(value.arm_id for value in plan.arms):
        raise ValueError("fixed-round assessment roster differs from plan")
    count_map = dict(prior_selection_counts)
    if len(count_map) != len(prior_selection_counts) or set(count_map) != {
        value.arm_id for value in plan.arms
    }:
        raise ValueError("fixed-round prior-count roster differs from plan")
    if any(value < 0 for value in count_map.values()):
        raise ValueError("fixed-round prior selection count cannot be negative")
    eligible = [
        value
        for value in assessments
        if value.eligible
        and count_map[value.arm_id]
        < next(arm.maximum_selections for arm in plan.arms if arm.arm_id == value.arm_id)
    ]
    ordered = sorted(
        eligible,
        key=lambda value: (
            -Fraction(value.priority_numerator, value.priority_denominator),
            value.arm_id,
        ),
    )
    selected = {
        value.arm_id: ordinal
        for ordinal, value in enumerate(ordered[: plan.selections_per_round], start=1)
    }
    items = tuple(
        FixedRoundDecisionItem(
            arm_id=assessment.arm_id,
            disposition=(
                FixedRoundDisposition.SELECTED
                if assessment.arm_id in selected
                else FixedRoundDisposition.SKIPPED
            ),
            selection_ordinal=selected.get(assessment.arm_id),
            reason_codes=(
                ()
                if assessment.arm_id in selected
                else (assessment.reason_codes or ("NOT_SELECTED_BY_FROZEN_PRIORITY",))
            ),
        )
        for assessment in assessments
    )
    return FixedRoundAcquisitionDecision(
        decision_id=f"fixed-round-decision.{plan.acquisition_plan_id}.{round_number:03d}",
        plan=ObjectIdentity.from_record(plan.acquisition_plan_id, plan),
        round_number=round_number,
        checkpoint=checkpoint,
        items=items,
    )


def validate_fixed_round_completeness(
    *,
    plan: FixedRoundAcquisitionPlan,
    decisions: tuple[FixedRoundAcquisitionDecision, ...],
    receipts: tuple[FixedRoundAcquisitionReceipt, ...],
) -> FixedRoundCompletenessReceipt:
    """Prove the static maximum graph is exactly accounted for, including skips."""

    expected_rounds = tuple(range(1, plan.maximum_rounds + 1))
    if tuple(value.round_number for value in decisions) != expected_rounds:
        raise ValueError("fixed-round decision trace is incomplete or out of order")
    if tuple(value.round_number for value in receipts) != expected_rounds:
        raise ValueError("fixed-round receipt trace is incomplete or out of order")
    plan_identity = ObjectIdentity.from_record(plan.acquisition_plan_id, plan)
    arm_ids = tuple(value.arm_id for value in plan.arms)
    receipt_identities: list[ObjectIdentity] = []
    for decision, receipt in zip(decisions, receipts, strict=True):
        if decision.plan != plan_identity:
            raise ValueError("fixed-round decision binds another plan")
        if tuple(value.arm_id for value in decision.items) != arm_ids:
            raise ValueError("fixed-round decision does not account for every arm")
        decision_identity = ObjectIdentity.from_record(decision.decision_id, decision)
        if receipt.decision != decision_identity:
            raise ValueError("fixed-round receipt binds another decision")
        selected = tuple(
            sorted(
                value.arm_id
                for value in decision.items
                if value.disposition is FixedRoundDisposition.SELECTED
            )
        )
        skipped = tuple(
            sorted(
                value.arm_id
                for value in decision.items
                if value.disposition is FixedRoundDisposition.SKIPPED
            )
        )
        if receipt.completed_arm_ids != selected or receipt.skipped_arm_ids != skipped:
            raise ValueError("fixed-round receipt differs from selected/skipped truth")
        receipt_identities.append(ObjectIdentity.from_record(receipt.receipt_id, receipt))
    return FixedRoundCompletenessReceipt(
        receipt_id=f"fixed-round-completeness.{plan.acquisition_plan_id}",
        plan=plan_identity,
        round_receipts=tuple(receipt_identities),
        completed_round_count=plan.maximum_rounds,
        complete=True,
    )


class AcquisitionDecisionKind(StrEnum):
    QUERY = "QUERY"
    STOP_SUFFICIENT = "STOP_SUFFICIENT"
    STOP_NO_LEGAL_QUERY = "STOP_NO_LEGAL_QUERY"


class QueryBundleReceiptDisposition(StrEnum):
    EXECUTED = "EXECUTED"
    ZERO_CALL_SKIP = "ZERO_CALL_SKIP"


class AcquisitionEvidenceRole(StrEnum):
    COMMON_PANEL = "COMMON_PANEL"
    ADAPTIVE_QUERY = "ADAPTIVE_QUERY"
    MAPPED_DEVELOPMENT_DONOR = "MAPPED_DEVELOPMENT_DONOR"


class RecurrenceEvidenceDomain(StrEnum):
    ACQUISITION = "ACQUISITION"
    EXCLUDED_MAPPED_DEVELOPMENT = "EXCLUDED_MAPPED_DEVELOPMENT"


class RecurrenceEvidenceSourceKind(StrEnum):
    ACQUISITION_TRACE = "ACQUISITION_TRACE"
    MAPPED_DEVELOPMENT_SUPPORT_ATLAS = "MAPPED_DEVELOPMENT_SUPPORT_ATLAS"


class ActionPreparationRecurrenceStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


class RecurrenceQualificationDisposition(StrEnum):
    RECURRENCE_BLOCKED = "RECURRENCE_BLOCKED"
    FINALIZED_NO_LAW = "FINALIZED_NO_LAW"
    FINALIZED_SUPPORTED_LAW = "FINALIZED_SUPPORTED_LAW"


class AcquisitionCheckpointCellDisposition(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNEVALUABLE = "UNEVALUABLE"


class AcquisitionPriorityTier(StrEnum):
    MISSING_SECOND_PREPARATION = "MISSING_SECOND_PREPARATION"
    SEVERE_FALSE_SAFETY_OR_ADMISSION_BOUNDARY = "SEVERE_FALSE_SAFETY_OR_ADMISSION_BOUNDARY"
    UNRESOLVED_CATEGORICAL_DISPOSITION = "UNRESOLVED_CATEGORICAL_DISPOSITION"
    METHOD_UNCERTAINTY_PER_COMPLETE_BUNDLE = "METHOD_UNCERTAINTY_PER_COMPLETE_BUNDLE"


_ACQUISITION_PRIORITY_ORDER = tuple(AcquisitionPriorityTier)


@dataclass(frozen=True, slots=True)
class AcquisitionPriorityComponent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/acquisition-priority-component'

    component_id: str
    tier: AcquisitionPriorityTier
    score_numerator: int
    score_denominator: int
    evidence_inputs: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.component_id, field_name="component_id")
        if self.score_numerator < 0 or self.score_denominator < 1:
            raise ValueError("acquisition component score must be nonnegative and finite")
        require_sorted_unique_ids(
            self.evidence_inputs,
            attribute="object_id",
            field_name="evidence_inputs",
        )
        if not self.evidence_inputs:
            raise ValueError("acquisition priority component requires causal evidence")


@dataclass(frozen=True, slots=True)
class AcquisitionCheckpointCell(CanonicalRecord):
    """One explicit provisional action/member disposition at a checkpoint."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/acquisition-checkpoint-cell'

    cell_id: str
    action_word: ObjectIdentity
    model_member_id: str
    disposition: AcquisitionCheckpointCellDisposition
    claimed_active: bool
    recurrence_covered: bool
    uncertainty_resolved: bool
    safe: bool
    evidence_inputs: tuple[ObjectIdentity, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        validate_stable_id(self.model_member_id, field_name="model_member_id")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("checkpoint cell requires an exact ActionWord")
        require_sorted_unique_ids(
            self.evidence_inputs,
            attribute="object_id",
            field_name="evidence_inputs",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not self.evidence_inputs:
            raise ValueError("checkpoint cell requires generic compile/gate evidence")
        if self.disposition is AcquisitionCheckpointCellDisposition.PASS:
            if self.reason_codes or not self.uncertainty_resolved or not self.safe:
                raise ValueError("checkpoint PASS must be resolved, safe and reason-free")
        elif not self.reason_codes:
            raise ValueError("checkpoint non-PASS requires a typed reason")
        if self.claimed_active and (
            self.disposition is not AcquisitionCheckpointCellDisposition.PASS
            or not self.recurrence_covered
        ):
            raise ValueError("claimed checkpoint action lacks pass/recurrence support")


@dataclass(frozen=True, slots=True)
class AcquisitionCheckpointAssessment(CanonicalRecord):
    """Unticked generic checkpoint compile from which stopping is derived."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/acquisition-checkpoint-assessment'

    checkpoint_id: str
    task_id: str
    arm_id: str
    round_number: int
    trace_parent_receipts: tuple[ObjectIdentity, ...]
    provisional_study: ObjectIdentity
    provisional_compile: ObjectIdentity
    cells: tuple[AcquisitionCheckpointCell, ...]
    study_action_word_ids: tuple[str, ...]
    support_sufficient: bool
    outcome_access: OutcomeAccess
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("checkpoint_id", self.checkpoint_id),
            ("task_id", self.task_id),
            ("arm_id", self.arm_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.round_number not in {1, 2}:
            raise ValueError("checkpoint assessment is outside the two-round graph")
        require_sorted_unique_ids(
            self.trace_parent_receipts,
            attribute="object_id",
            field_name="trace_parent_receipts",
        )
        if not self.trace_parent_receipts:
            raise ValueError("checkpoint assessment lacks its causal trace prefix")
        if self.provisional_study.object_schema != (
            'empirical-lawhood/planning/admission-controller-study'
        ) or self.provisional_compile.object_schema != (
            'empirical-lawhood/runtime/compiled-admission-controller-study'
        ):
            raise ValueError("checkpoint assessment must bind the generic admission/controller-use route")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if not self.cells:
            raise ValueError("checkpoint assessment lacks action/member cells")
        require_sorted_unique_strings(
            self.study_action_word_ids,
            field_name='study_action_word_ids',
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        cell_action_ids = {value.action_word.object_id for value in self.cells}
        if not set(self.study_action_word_ids) <= cell_action_ids:
            raise ValueError("checkpoint programme action is outside its assessed chart")
        programme_cells = tuple(
            value
            for value in self.cells
            if value.action_word.object_id in self.study_action_word_ids
        )
        expected_sufficient = (
            all(
                value.disposition is not AcquisitionCheckpointCellDisposition.UNEVALUABLE
                and value.uncertainty_resolved
                for value in self.cells
            )
            and all(
                value.disposition is AcquisitionCheckpointCellDisposition.PASS
                and value.safe
                and value.recurrence_covered
                for value in programme_cells
            )
            and bool(programme_cells)
        )
        if self.support_sufficient is not expected_sufficient:
            raise ValueError("checkpoint sufficiency is not disposition/recurrence-derived")
        if self.support_sufficient == bool(self.reason_codes):
            raise ValueError("checkpoint sufficiency and reasons are inconsistent")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        }:
            raise ValueError("checkpoint assessment cannot inspect evaluation outcomes")


@dataclass(frozen=True, slots=True)
class AcquisitionBundleAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/acquisition-bundle-assessment'

    assessment_id: str
    query_bundle_id: str
    legal: bool
    priority_components: tuple[AcquisitionPriorityComponent, ...]
    evidence_inputs: tuple[ObjectIdentity, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.query_bundle_id, field_name="query_bundle_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if tuple(value.tier for value in self.priority_components) != (_ACQUISITION_PRIORITY_ORDER):
            raise ValueError("acquisition bundle lacks the exact ordered priority components")
        if self.legal == bool(self.reason_codes):
            raise ValueError("acquisition bundle legality and reasons disagree")
        if not self.evidence_inputs:
            raise ValueError("acquisition bundle assessment requires causal inputs")
        if len({value.object_id for value in self.evidence_inputs}) != len(self.evidence_inputs):
            raise ValueError("acquisition bundle assessment duplicates an input identity")
        component_inputs = {
            value for component in self.priority_components for value in component.evidence_inputs
        }
        if set(self.evidence_inputs) != component_inputs:
            raise ValueError("acquisition bundle evidence differs from component evidence")


@dataclass(frozen=True, slots=True)
class AcquisitionRoundState(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/acquisition-round-state'

    state_id: str
    plan: ObjectIdentity
    round_number: int
    prior_round_receipt: ObjectIdentity | None
    queried_bundle_ids: tuple[str, ...]
    cumulative_episode_charge: int
    checkpoint_assessment: AcquisitionCheckpointAssessment
    bundle_assessments: tuple[AcquisitionBundleAssessment, ...]
    legal_remaining_bundle_ids: tuple[str, ...]
    causal_domain: AcquisitionCausalDomain
    information_cutoff_id: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.state_id, field_name="state_id")
        validate_stable_id(self.information_cutoff_id, field_name="information_cutoff_id")
        if self.plan.object_schema != BoundedQueryAcquisitionPlan.SCHEMA:
            raise ValueError("acquisition state requires a bounded-query plan")
        if self.round_number not in {1, 2}:
            raise ValueError("acquisition state is outside the two-round graph")
        if (self.round_number == 1) != (self.prior_round_receipt is None):
            raise ValueError("only the first acquisition round lacks a parent receipt")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        }:
            raise ValueError("acquisition state cannot inspect evaluation outcomes")
        if (
            self.checkpoint_assessment.round_number != self.round_number
            or self.checkpoint_assessment.outcome_access is not self.outcome_access
        ):
            raise ValueError("acquisition state substitutes its checkpoint assessment")
        require_sorted_unique_strings(
            self.queried_bundle_ids,
            field_name="queried_bundle_ids",
        )
        require_sorted_unique_ids(
            self.bundle_assessments,
            attribute="query_bundle_id",
            field_name="bundle_assessments",
        )
        require_sorted_unique_strings(
            self.legal_remaining_bundle_ids,
            field_name="legal_remaining_bundle_ids",
        )
        expected_legal = tuple(
            value.query_bundle_id for value in self.bundle_assessments if value.legal
        )
        if self.legal_remaining_bundle_ids != expected_legal:
            raise ValueError("acquisition legal roster is not assessment-derived")
        if set(self.queried_bundle_ids) & set(self.legal_remaining_bundle_ids):
            raise ValueError("an already queried bundle cannot remain legal")
        if self.cumulative_episode_charge < 0:
            raise ValueError("acquisition charge cannot be negative")
        if self.causal_domain is not AcquisitionCausalDomain.ACQUISITION:
            raise ValueError("acquisition state crosses its causal domain")


@dataclass(frozen=True, slots=True)
class AcquisitionDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/acquisition-decision'

    decision_id: str
    state: ObjectIdentity
    round_number: int
    kind: AcquisitionDecisionKind
    query_bundle_id: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.state.object_schema != AcquisitionRoundState.SCHEMA:
            raise ValueError("acquisition decision requires an exact round state")
        if self.round_number not in {1, 2}:
            raise ValueError("acquisition decision is outside the static graph")
        if self.kind is AcquisitionDecisionKind.QUERY:
            if self.query_bundle_id is None or self.reason_codes:
                raise ValueError("QUERY requires exactly one bundle and no stop reason")
            validate_stable_id(self.query_bundle_id, field_name="query_bundle_id")
        elif self.query_bundle_id is not None or not self.reason_codes:
            raise ValueError("acquisition STOP cannot carry a query bundle")


@dataclass(frozen=True, slots=True)
class QueryCoordinateOutput(CanonicalRecord):
    """One exact member-episode output for a selected query coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/query-coordinate-output'

    output_id: str
    query_coordinate: ObjectIdentity
    episode_receipt: ObjectIdentity
    delivery_receipt: ObjectIdentity
    response_evidence: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.output_id, field_name="output_id")
        if self.query_coordinate.object_schema != (
            'empirical-lawhood/planning/acquisition-query-coordinate'
        ):
            raise ValueError("query output requires an exact acquisition coordinate")


@dataclass(frozen=True, slots=True)
class QueryBundleReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/query-bundle-receipt'

    receipt_id: str
    decision: ObjectIdentity
    round_number: int
    query_bundle_id: str | None
    disposition: QueryBundleReceiptDisposition
    executed_coordinate_ids: tuple[str, ...]
    zero_call_slot_ids: tuple[str, ...]
    outputs: tuple[QueryCoordinateOutput, ...]
    episode_charge: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.decision.object_schema != AcquisitionDecision.SCHEMA:
            raise ValueError("query receipt requires an acquisition decision")
        require_sorted_unique_strings(
            self.executed_coordinate_ids,
            field_name="executed_coordinate_ids",
        )
        require_sorted_unique_strings(self.zero_call_slot_ids, field_name="zero_call_slot_ids")
        require_sorted_unique_ids(self.outputs, attribute="output_id", field_name="outputs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is QueryBundleReceiptDisposition.EXECUTED:
            if (
                self.query_bundle_id is None
                or not self.executed_coordinate_ids
                or self.zero_call_slot_ids
                or len(self.outputs) != len(self.executed_coordinate_ids)
                or self.episode_charge != len(self.executed_coordinate_ids)
                or self.reason_codes
            ):
                raise ValueError("executed query bundle receipt is incomplete")
            if {value.query_coordinate.object_id for value in self.outputs} != set(
                self.executed_coordinate_ids
            ):
                raise ValueError("query outputs differ from executed coordinates")
        elif (
            self.query_bundle_id is not None
            or self.executed_coordinate_ids
            or not self.zero_call_slot_ids
            or self.outputs
            or self.episode_charge != 0
            or not self.reason_codes
        ):
            raise ValueError("zero-call query receipt fabricates execution")


@dataclass(frozen=True, slots=True)
class AcquisitionTraceReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/acquisition-trace-receipt'

    trace_id: str
    plan: ObjectIdentity
    states: tuple[AcquisitionRoundState, ...]
    decisions: tuple[AcquisitionDecision, ...]
    round_receipts: tuple[QueryBundleReceipt, ...]
    common_panel_outputs: tuple[QueryCoordinateOutput, ...]
    common_panel_completed_coordinate_ids: tuple[str, ...]
    common_panel_bundle_ids: tuple[str, ...]
    queried_bundle_ids: tuple[str, ...]
    adaptively_completed_coordinate_ids: tuple[str, ...]
    never_selected_coordinate_ids: tuple[str, ...]
    common_panel_episode_charge: int
    adaptive_episode_charge: int
    total_episode_charge: int
    total_bundle_charge: int
    complete: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.trace_id, field_name="trace_id")
        if self.plan.object_schema != BoundedQueryAcquisitionPlan.SCHEMA:
            raise ValueError("acquisition trace requires a bounded-query plan")
        if (
            tuple(value.round_number for value in self.states) != (1, 2)
            or tuple(value.round_number for value in self.decisions) != (1, 2)
            or tuple(value.round_number for value in self.round_receipts) != (1, 2)
        ):
            raise ValueError("acquisition trace must close both static rounds")
        require_sorted_unique_strings(
            self.queried_bundle_ids,
            field_name="queried_bundle_ids",
        )
        require_sorted_unique_ids(
            self.common_panel_outputs,
            attribute="output_id",
            field_name="common_panel_outputs",
        )
        require_sorted_unique_strings(
            self.common_panel_completed_coordinate_ids,
            field_name="common_panel_completed_coordinate_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.common_panel_bundle_ids,
            field_name="common_panel_bundle_ids",
            allow_empty=False,
        )
        if len(self.common_panel_bundle_ids) != 4:
            raise ValueError("acquisition trace requires exactly four common bundles")
        if {value.query_coordinate.object_id for value in self.common_panel_outputs} != set(
            self.common_panel_completed_coordinate_ids
        ) or len(self.common_panel_outputs) != len(self.common_panel_completed_coordinate_ids):
            raise ValueError("acquisition trace common outputs differ from exact coordinates")
        if (
            len({value.episode_receipt for value in self.common_panel_outputs})
            != len(self.common_panel_outputs)
            or len({value.delivery_receipt for value in self.common_panel_outputs})
            != len(self.common_panel_outputs)
            or len({value.response_evidence for value in self.common_panel_outputs})
            != len(self.common_panel_outputs)
        ):
            raise ValueError("common panel reuses episode, delivery or response evidence")
        require_sorted_unique_strings(
            self.adaptively_completed_coordinate_ids,
            field_name="adaptively_completed_coordinate_ids",
        )
        require_sorted_unique_strings(
            self.never_selected_coordinate_ids,
            field_name="never_selected_coordinate_ids",
        )
        if set(self.adaptively_completed_coordinate_ids) & set(self.never_selected_coordinate_ids):
            raise ValueError("acquisition trace both completes and skips one legal coordinate")
        adaptive_charge = sum(value.episode_charge for value in self.round_receipts)
        if (
            self.common_panel_episode_charge != len(self.common_panel_outputs)
            or self.adaptive_episode_charge != adaptive_charge
            or self.total_episode_charge
            != self.common_panel_episode_charge + self.adaptive_episode_charge
            or self.total_bundle_charge
            != len(self.common_panel_bundle_ids) + len(self.queried_bundle_ids)
        ):
            raise ValueError("acquisition trace charge is not receipt-derived")
        if not self.complete:
            raise ValueError("acquisition trace cannot remain operationally incomplete")


def decide_bounded_query(
    *,
    plan: BoundedQueryAcquisitionPlan,
    state: AcquisitionRoundState,
) -> AcquisitionDecision:
    """Apply the exact three-way decision without launching a query."""

    plan_identity = ObjectIdentity.from_record(plan.acquisition_plan_id, plan)
    if (
        state.plan != plan_identity
        or state.information_cutoff_id != plan.information_cutoff_id
        or state.outcome_access is not plan.outcome_access
    ):
        raise ValueError("acquisition state substitutes plan or cutoff")
    if set(state.queried_bundle_ids) - set(plan.query_bundle_priority_ids):
        raise ValueError("acquisition state contains an unissued query bundle")
    if tuple(value.query_bundle_id for value in state.bundle_assessments) != (
        plan.query_bundle_priority_ids
    ):
        raise ValueError("acquisition state assessment roster differs from the plan")
    if state.cumulative_episode_charge > plan.maximum_episode_charge:
        raise ValueError("acquisition state already exceeds the episode ceiling")
    state_identity = ObjectIdentity.from_record(state.state_id, state)
    checkpoint = state.checkpoint_assessment
    expected_cells = {
        (action, member) for action in plan.action_words for member in plan.model_member_ids
    }
    if (
        checkpoint.task_id != plan.task_ids[0]
        or checkpoint.arm_id != plan.arm_ids[0]
        or {(value.action_word, value.model_member_id) for value in checkpoint.cells}
        != expected_cells
        or len(checkpoint.cells) != len(expected_cells)
    ):
        raise ValueError("checkpoint assessment differs from the issued action/member product")
    if checkpoint.support_sufficient:
        return AcquisitionDecision(
            decision_id=f"acquisition-decision.{plan.acquisition_plan_id}.{state.round_number}",
            state=state_identity,
            round_number=state.round_number,
            kind=AcquisitionDecisionKind.STOP_SUFFICIENT,
            query_bundle_id=None,
            reason_codes=("SUPPORTED_ACTION_RECURRENCE_SUFFICIENT",),
        )
    if (
        not state.legal_remaining_bundle_ids
        or len(state.queried_bundle_ids) >= plan.maximum_query_bundles
    ):
        return AcquisitionDecision(
            decision_id=f"acquisition-decision.{plan.acquisition_plan_id}.{state.round_number}",
            state=state_identity,
            round_number=state.round_number,
            kind=AcquisitionDecisionKind.STOP_NO_LEGAL_QUERY,
            query_bundle_id=None,
            reason_codes=("NO_LEGAL_QUERY_WITHIN_FROZEN_CEILING",),
        )
    priority = {value: index for index, value in enumerate(plan.query_bundle_priority_ids)}
    assessment = min(
        (
            value
            for value in state.bundle_assessments
            if value.query_bundle_id in state.legal_remaining_bundle_ids
        ),
        key=lambda value: (
            *(
                -Fraction(component.score_numerator, component.score_denominator)
                for component in value.priority_components
            ),
            priority[value.query_bundle_id],
            value.query_bundle_id,
        ),
    )
    coordinates = tuple(
        value
        for value in plan.query_coordinates
        if value.query_bundle_id == assessment.query_bundle_id
    )
    if state.cumulative_episode_charge + len(coordinates) > plan.maximum_episode_charge:
        raise ValueError("selected acquisition query exceeds the episode ceiling")
    return AcquisitionDecision(
        decision_id=f"acquisition-decision.{plan.acquisition_plan_id}.{state.round_number}",
        state=state_identity,
        round_number=state.round_number,
        kind=AcquisitionDecisionKind.QUERY,
        query_bundle_id=assessment.query_bundle_id,
        reason_codes=(),
    )


def close_query_round(
    *,
    plan: BoundedQueryAcquisitionPlan,
    state: AcquisitionRoundState,
    decision: AcquisitionDecision,
    outputs: tuple[QueryCoordinateOutput, ...] = (),
) -> QueryBundleReceipt:
    if decision.state != ObjectIdentity.from_record(state.state_id, state):
        raise ValueError("query round decision substitutes another state")
    if decision.round_number != state.round_number:
        raise ValueError("query round decision changes its static round")
    if decision.kind is AcquisitionDecisionKind.QUERY:
        if decision.query_bundle_id is None:  # pragma: no cover - decision invariant
            raise AssertionError("QUERY lost its bundle")
        executed = tuple(
            value.coordinate_id
            for value in plan.query_coordinates
            if value.query_bundle_id == decision.query_bundle_id
        )
        coordinate_identities = {
            value.coordinate_id: ObjectIdentity.from_record(value.coordinate_id, value)
            for value in plan.query_coordinates
            if value.query_bundle_id == decision.query_bundle_id
        }
        if {
            value.query_coordinate.object_id: value.query_coordinate for value in outputs
        } != coordinate_identities:
            raise ValueError("query outputs substitute the selected scientific coordinates")
        return QueryBundleReceipt(
            receipt_id=f"query-receipt.{plan.acquisition_plan_id}.{state.round_number}",
            decision=ObjectIdentity.from_record(decision.decision_id, decision),
            round_number=state.round_number,
            query_bundle_id=decision.query_bundle_id,
            disposition=QueryBundleReceiptDisposition.EXECUTED,
            executed_coordinate_ids=executed,
            zero_call_slot_ids=(),
            outputs=outputs,
            episode_charge=len(executed),
            reason_codes=(),
        )
    if outputs:
        raise ValueError("STOP decision cannot publish query outputs")
    zero_call_slots = tuple(
        f"query-slot.{plan.acquisition_plan_id}.{state.round_number:02d}.{member}"
        for member in plan.model_member_ids
    )
    return QueryBundleReceipt(
        receipt_id=f"query-receipt.{plan.acquisition_plan_id}.{state.round_number}",
        decision=ObjectIdentity.from_record(decision.decision_id, decision),
        round_number=state.round_number,
        query_bundle_id=None,
        disposition=QueryBundleReceiptDisposition.ZERO_CALL_SKIP,
        executed_coordinate_ids=(),
        zero_call_slot_ids=zero_call_slots,
        outputs=(),
        episode_charge=0,
        reason_codes=decision.reason_codes,
    )


def validate_acquisition_trace(
    *,
    plan: BoundedQueryAcquisitionPlan,
    states: tuple[AcquisitionRoundState, ...],
    decisions: tuple[AcquisitionDecision, ...],
    receipts: tuple[QueryBundleReceipt, ...],
    common_panel_outputs: tuple[QueryCoordinateOutput, ...],
) -> AcquisitionTraceReceipt:
    if len(states) != 2 or len(decisions) != 2 or len(receipts) != 2:
        raise ValueError("bounded acquisition must close exactly two static rounds")
    plan_identity = ObjectIdentity.from_record(plan.acquisition_plan_id, plan)
    queried: list[str] = []
    charge = 0
    prior: QueryBundleReceipt | None = None
    stopped = False
    for round_number, (state, decision, receipt) in enumerate(
        zip(states, decisions, receipts, strict=True),
        start=1,
    ):
        if (
            state.round_number != round_number
            or state.plan != plan_identity
            or state.queried_bundle_ids != tuple(sorted(queried))
            or state.cumulative_episode_charge != charge
            or decision != decide_bounded_query(plan=plan, state=state)
            or receipt.decision != ObjectIdentity.from_record(decision.decision_id, decision)
        ):
            raise ValueError("bounded acquisition round is not causally derived")
        if round_number == 2 and (
            prior is None
            or state.prior_round_receipt != ObjectIdentity.from_record(prior.receipt_id, prior)
        ):
            raise ValueError("bounded acquisition round two lacks its durable parent receipt")
        expected_receipt = close_query_round(
            plan=plan,
            state=state,
            decision=decision,
            outputs=receipt.outputs,
        )
        if receipt != expected_receipt:
            raise ValueError("bounded acquisition receipt changes execution or zero-call truth")
        if stopped and decision.kind is AcquisitionDecisionKind.QUERY:
            raise ValueError("bounded acquisition cannot query after a terminal stop")
        if decision.kind is AcquisitionDecisionKind.QUERY:
            if decision.query_bundle_id in queried:
                raise ValueError("bounded acquisition queries one bundle twice")
            if decision.query_bundle_id is None:  # pragma: no cover
                raise AssertionError("QUERY lost its bundle")
            queried.append(decision.query_bundle_id)
        else:
            stopped = True
        charge += receipt.episode_charge
        prior = receipt
    if charge > plan.maximum_episode_charge or len(queried) > plan.maximum_query_bundles:
        raise ValueError("bounded acquisition trace exceeds its frozen ceiling")
    completed = tuple(
        sorted(
            {
                coordinate_id
                for receipt in receipts
                for coordinate_id in receipt.executed_coordinate_ids
            }
        )
    )
    legal = {value.coordinate_id for value in plan.query_coordinates}
    if not set(completed) <= legal:
        raise ValueError("bounded acquisition completed an unissued query coordinate")
    never_selected = tuple(sorted(legal - set(completed)))
    common_coordinate_identities = {
        value.coordinate_id: ObjectIdentity.from_record(value.coordinate_id, value)
        for value in plan.common_panel_coordinates
    }
    if {
        value.query_coordinate.object_id: value.query_coordinate for value in common_panel_outputs
    } != common_coordinate_identities:
        raise ValueError("common-panel outputs differ from the issued exact coordinates")
    common_bundle_ids = tuple(
        sorted({value.query_bundle_id for value in plan.common_panel_coordinates})
    )
    common_coordinate_ids = tuple(sorted(common_coordinate_identities))
    return AcquisitionTraceReceipt(
        trace_id=f"acquisition-trace.{plan.acquisition_plan_id}",
        plan=plan_identity,
        states=states,
        decisions=decisions,
        round_receipts=receipts,
        common_panel_outputs=tuple(sorted(common_panel_outputs, key=lambda value: value.output_id)),
        common_panel_completed_coordinate_ids=common_coordinate_ids,
        common_panel_bundle_ids=common_bundle_ids,
        queried_bundle_ids=tuple(sorted(queried)),
        adaptively_completed_coordinate_ids=completed,
        never_selected_coordinate_ids=never_selected,
        common_panel_episode_charge=len(common_panel_outputs),
        adaptive_episode_charge=charge,
        total_episode_charge=len(common_panel_outputs) + charge,
        total_bundle_charge=len(common_bundle_ids) + len(queried),
        complete=True,
    )


@dataclass(frozen=True, slots=True)
class ActionPreparationEvidenceCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/action-preparation-evidence-cell'

    cell_id: str
    task_id: str
    arm_id: str
    preparation_unit_id: str
    requested_action_word: ObjectIdentity
    model_member_id: str
    source_fingerprint: str
    accepted_action: ObjectIdentity
    applied_action: ObjectIdentity
    realized_action: ObjectIdentity
    delivery_receipt: ObjectIdentity
    response_evidence: ObjectIdentity
    method_config: ObjectIdentity
    recurrence_predicate: ObjectIdentity
    uncertainty_rule: ObjectIdentity
    acquisition_role: AcquisitionEvidenceRole
    matched_clone: bool
    delivery_valid: bool
    recurrence_passed: bool
    causal_domain: RecurrenceEvidenceDomain
    outcome_access: OutcomeAccess
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("task_id", self.task_id),
            ("arm_id", self.arm_id),
            ("preparation_unit_id", self.preparation_unit_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.requested_action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("recurrence cell requires an exact requested ActionWord")
        validate_sha256(self.source_fingerprint, field_name="source_fingerprint")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.matched_clone and self.delivery_valid and self.recurrence_passed:
            if self.reason_codes:
                raise ValueError("passing recurrence cell cannot carry reasons")
        elif not self.reason_codes:
            raise ValueError("failed recurrence cell requires a typed reason")
        expected_role_domain = {
            AcquisitionEvidenceRole.COMMON_PANEL: RecurrenceEvidenceDomain.ACQUISITION,
            AcquisitionEvidenceRole.ADAPTIVE_QUERY: RecurrenceEvidenceDomain.ACQUISITION,
            AcquisitionEvidenceRole.MAPPED_DEVELOPMENT_DONOR: (
                RecurrenceEvidenceDomain.EXCLUDED_MAPPED_DEVELOPMENT
            ),
        }
        if self.causal_domain is not expected_role_domain[self.acquisition_role]:
            raise ValueError("recurrence evidence role crosses its causal domain")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        }:
            raise ValueError("evaluation-visible evidence cannot establish action/preparation recurrence")

    @property
    def action_word_id(self) -> str:
        return self.requested_action_word.object_id


@dataclass(frozen=True, slots=True)
class ActionPreparationRecurrenceEvidenceSource(CanonicalRecord):
    "Typed precommitment evidence source shared by generated and mapped action/preparation recurrence."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/action-preparation-recurrence-evidence-source'

    source_id: str
    source_kind: RecurrenceEvidenceSourceKind
    source_record: ObjectIdentity
    causal_domain: RecurrenceEvidenceDomain
    outcome_access: OutcomeAccess
    evidence_inputs: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.source_id, field_name="source_id")
        require_sorted_unique_ids(
            self.evidence_inputs,
            attribute="object_id",
            field_name="evidence_inputs",
        )
        if not self.evidence_inputs:
            raise ValueError("recurrence evidence source cannot be empty")
        expected = {
            RecurrenceEvidenceSourceKind.ACQUISITION_TRACE: (
                AcquisitionTraceReceipt.SCHEMA,
                RecurrenceEvidenceDomain.ACQUISITION,
            ),
            RecurrenceEvidenceSourceKind.MAPPED_DEVELOPMENT_SUPPORT_ATLAS: (
                'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-development-support-atlas',
                RecurrenceEvidenceDomain.EXCLUDED_MAPPED_DEVELOPMENT,
            ),
        }[self.source_kind]
        if self.source_record.object_schema != expected[0] or self.causal_domain is not expected[1]:
            raise ValueError("recurrence evidence source changes its typed causal domain")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.DEVELOPMENT_VISIBLE,
        }:
            raise ValueError("post-reference evidence cannot create or repair action/preparation recurrence")


def recurrence_evidence_source_from_acquisition_trace(
    trace: AcquisitionTraceReceipt,
) -> ActionPreparationRecurrenceEvidenceSource:
    evidence_inputs = {
        *(
            identity
            for output in trace.common_panel_outputs
            for identity in (
                ObjectIdentity.from_record(output.output_id, output),
                output.episode_receipt,
                output.delivery_receipt,
                output.response_evidence,
            )
        ),
        *(
            identity
            for receipt in trace.round_receipts
            for output in receipt.outputs
            for identity in (
                ObjectIdentity.from_record(output.output_id, output),
                output.episode_receipt,
                output.delivery_receipt,
                output.response_evidence,
            )
        ),
    }
    return ActionPreparationRecurrenceEvidenceSource(
        source_id=f"recurrence-evidence-source.{trace.trace_id}",
        source_kind=RecurrenceEvidenceSourceKind.ACQUISITION_TRACE,
        source_record=ObjectIdentity.from_record(trace.trace_id, trace),
        causal_domain=RecurrenceEvidenceDomain.ACQUISITION,
        outcome_access=trace.states[0].outcome_access,
        evidence_inputs=tuple(sorted(evidence_inputs, key=lambda value: value.object_id)),
    )


@dataclass(frozen=True, slots=True)
class ActionPreparationRecurrenceProofOwner(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/action-preparation-recurrence-proof-owner'

    owner_id: str
    obligation_id: str
    capability_key: str
    capability_version: str
    config_sha256: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.owner_id, field_name="owner_id")
        validate_stable_id(self.obligation_id, field_name="obligation_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(self.config_sha256, field_name="config_sha256")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.obligation_id != "order-relation.action-preparation-recurrence":
            raise ValueError("recurrence proof owner does not own the named action/preparation recurrence obligation")


@dataclass(frozen=True, slots=True)
class ActionPreparationRecurrenceReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/action-preparation-recurrence-receipt'

    receipt_id: str
    evidence_source: ActionPreparationRecurrenceEvidenceSource
    task_id: str
    arm_id: str
    active_action_word: ObjectIdentity
    matched_hold_action_word: ObjectIdentity
    preparation_unit_ids: tuple[str, ...]
    model_member_ids: tuple[str, ...]
    evidence_cells: tuple[ActionPreparationEvidenceCell, ...]
    method_config: ObjectIdentity
    proof_owner: ObjectIdentity
    status: ActionPreparationRecurrenceStatus
    evidence_ceiling: EvidenceCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("task_id", self.task_id),
            ("arm_id", self.arm_id),
        ):
            validate_stable_id(value, field_name=name)
        if (
            self.active_action_word.object_schema != OccurrenceActionWord.SCHEMA
            or self.matched_hold_action_word.object_schema != OccurrenceActionWord.SCHEMA
        ):
            raise ValueError("recurrence receipt requires exact active and HOLD ActionWords")
        if self.active_action_word == self.matched_hold_action_word:
            raise ValueError("active recurrence cannot use HOLD as its active word")
        if self.proof_owner.object_schema != ActionPreparationRecurrenceProofOwner.SCHEMA:
            raise ValueError("recurrence receipt requires the named proof owner")
        require_sorted_unique_strings(
            self.preparation_unit_ids,
            field_name="preparation_unit_ids",
        )
        require_sorted_unique_strings(
            self.model_member_ids,
            field_name="model_member_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.evidence_cells,
            attribute="cell_id",
            field_name="evidence_cells",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is ActionPreparationRecurrenceStatus.SUPPORTED:
            if len(self.preparation_unit_ids) < 2 or self.reason_codes:
                raise ValueError("supported active recurrence requires two preparations")
            expected = {
                (preparation, action, member)
                for preparation in self.preparation_unit_ids
                for action in (
                    self.active_action_word,
                    self.matched_hold_action_word,
                )
                for member in self.model_member_ids
            }
            observed = {
                (
                    value.preparation_unit_id,
                    value.requested_action_word,
                    value.model_member_id,
                )
                for value in self.evidence_cells
            }
            preparation_fingerprints = {
                preparation: {
                    value.source_fingerprint
                    for value in self.evidence_cells
                    if value.preparation_unit_id == preparation
                }
                for preparation in self.preparation_unit_ids
            }
            if (
                observed != expected
                or len(self.evidence_cells) != len(expected)
                or any(
                    value.task_id != self.task_id
                    or value.arm_id != self.arm_id
                    or value.method_config != self.method_config
                    or value.causal_domain is not self.evidence_source.causal_domain
                    or value.outcome_access is not self.evidence_source.outcome_access
                    or not value.matched_clone
                    or not value.delivery_valid
                    or not value.recurrence_passed
                    for value in self.evidence_cells
                )
                or len({value.delivery_receipt for value in self.evidence_cells})
                != len(self.evidence_cells)
                or len({value.response_evidence for value in self.evidence_cells})
                != len(self.evidence_cells)
                or len({value.recurrence_predicate for value in self.evidence_cells}) != 1
                or len({value.uncertainty_rule for value in self.evidence_cells}) != 1
                or any(len(values) != 1 for values in preparation_fingerprints.values())
                or len(
                    {next(iter(values)) for values in preparation_fingerprints.values() if values}
                )
                != len(self.preparation_unit_ids)
                or any(
                    value.delivery_receipt not in self.evidence_source.evidence_inputs
                    or value.response_evidence not in self.evidence_source.evidence_inputs
                    for value in self.evidence_cells
                )
            ):
                raise ValueError("supported recurrence receipt is not its exact product proof")
        elif not self.reason_codes:
            raise ValueError("unsupported active recurrence requires decisive reasons")
        if self.evidence_ceiling is not EvidenceCeiling.ORDER_RELATION:
            raise ValueError("action/preparation recurrence cannot exceed the order-relation evidence ceiling")

    @property
    def active_action_word_id(self) -> str:
        return self.active_action_word.object_id

    @property
    def matched_hold_action_word_id(self) -> str:
        return self.matched_hold_action_word.object_id


def recurrence_qualification_obstructions(
    requirement: ActionPreparationRecurrenceQualificationRequirement,
    receipts: tuple[ActionPreparationRecurrenceReceipt, ...],
) -> tuple[str, ...]:
    "Derive the complete noncompensating action/preparation recurrence obstruction set."

    reasons: set[str] = set()
    receipt_action_ids = tuple(value.active_action_word_id for value in receipts)
    if len(set(receipt_action_ids)) != len(receipt_action_ids) or set(receipt_action_ids) != set(
        requirement.active_action_word_ids
    ):
        reasons.add("RECURRENCE_RECEIPT_ROSTER_INCOMPLETE")
    if any(
        value.task_id != requirement.task_id
        or value.arm_id != requirement.arm_id
        or value.active_action_word not in requirement.active_action_words
        or value.matched_hold_action_word != requirement.matched_hold_action_word
        or value.model_member_ids != requirement.model_member_ids
        or value.method_config != requirement.method_config
        or value.proof_owner != requirement.proof_owner
        for value in receipts
    ):
        reasons.add("RECURRENCE_RECEIPT_BINDING_MISMATCH")
    if any(value.status is not ActionPreparationRecurrenceStatus.SUPPORTED for value in receipts):
        reasons.add("RECURRENCE_RECEIPT_NOT_SUPPORTED")
    return tuple(sorted(reasons))


@dataclass(frozen=True, slots=True)
class ActionPreparationRecurrenceQualificationBinding(CanonicalRecord):
    "Exact action/preparation recurrence receipts consumed immediately before the sole finalizer."

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/runtime/action-preparation-recurrence-qualification-binding'
    )

    binding_id: str
    requirement: ActionPreparationRecurrenceQualificationRequirement
    recurrence_receipts: tuple[ActionPreparationRecurrenceReceipt, ...]
    recurrence_receipt_fingerprints: tuple[str, ...]
    qualification_result: LawQualificationResult | None
    response_law: ObjectIdentity | None
    disposition: RecurrenceQualificationDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        ordered = tuple(
            sorted(
                self.recurrence_receipts,
                key=lambda value: value.active_action_word_id,
            )
        )
        if self.recurrence_receipts != ordered:
            raise ValueError("recurrence qualification receipts must be action-sorted")
        require_sorted_unique_strings(
            self.recurrence_receipt_fingerprints,
            field_name="recurrence_receipt_fingerprints",
        )
        expected_fingerprints = tuple(
            sorted(value.fingerprint() for value in self.recurrence_receipts)
        )
        if self.recurrence_receipt_fingerprints != expected_fingerprints:
            raise ValueError("recurrence qualification changes an exact receipt digest")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        obstructions = recurrence_qualification_obstructions(
            self.requirement,
            self.recurrence_receipts,
        )
        if obstructions:
            if (
                self.disposition is not RecurrenceQualificationDisposition.RECURRENCE_BLOCKED
                or self.qualification_result is not None
                or self.response_law is not None
                or self.reason_codes != obstructions
            ):
                raise ValueError("under-qualified recurrence reached the law finalizer")
            return
        result = self.qualification_result
        if result is None or result.config != self.requirement.method_config:
            raise ValueError("qualified recurrence lacks its exact finalizer result/config")
        law = result.response_law
        expected_law = ObjectIdentity.from_record(law.law_id, law) if law is not None else None
        if self.response_law != expected_law or self.reason_codes != result.reason_codes:
            raise ValueError("recurrence qualification rewrites terminal law truth")
        if law is None:
            if self.disposition is not RecurrenceQualificationDisposition.FINALIZED_NO_LAW:
                raise ValueError("law-free finalizer result has the wrong disposition")
        elif (
            self.disposition is not RecurrenceQualificationDisposition.FINALIZED_SUPPORTED_LAW
            or result.scientific_status is not ScientificStatus.SUPPORTED
            or result.highest_supported_rung is not EvidenceRung.LOCAL_LAW
        ):
            raise ValueError("recurrence-qualified law is not an exact supported local law result")


def bind_action_preparation_recurrence_qualification(
    *,
    requirement: ActionPreparationRecurrenceQualificationRequirement,
    receipts: tuple[ActionPreparationRecurrenceReceipt, ...],
    qualification_result: LawQualificationResult | None,
) -> ActionPreparationRecurrenceQualificationBinding:
    ordered = tuple(sorted(receipts, key=lambda value: value.active_action_word_id))
    obstructions = recurrence_qualification_obstructions(requirement, ordered)
    if obstructions:
        if qualification_result is not None:
            raise ValueError("blocked recurrence cannot bind a finalizer result")
        disposition = RecurrenceQualificationDisposition.RECURRENCE_BLOCKED
        response_law = None
        reasons = obstructions
    else:
        if qualification_result is None:
            raise ValueError("supported recurrence requires the sole finalizer result")
        response_law = (
            ObjectIdentity.from_record(
                qualification_result.response_law.law_id,
                qualification_result.response_law,
            )
            if qualification_result.response_law is not None
            else None
        )
        disposition = (
            RecurrenceQualificationDisposition.FINALIZED_SUPPORTED_LAW
            if response_law is not None
            else RecurrenceQualificationDisposition.FINALIZED_NO_LAW
        )
        reasons = qualification_result.reason_codes
    return ActionPreparationRecurrenceQualificationBinding(
        binding_id=f"recurrence-qualification-binding.{requirement.requirement_id}",
        requirement=requirement,
        recurrence_receipts=ordered,
        recurrence_receipt_fingerprints=tuple(sorted(value.fingerprint() for value in ordered)),
        qualification_result=qualification_result,
        response_law=response_law,
        disposition=disposition,
        reason_codes=reasons,
    )


def evaluate_action_preparation_recurrence(
    *,
    evidence_source: ActionPreparationRecurrenceEvidenceSource,
    task_id: str,
    arm_id: str,
    active_action_word: ObjectIdentity,
    matched_hold_action_word: ObjectIdentity,
    model_member_ids: tuple[str, ...],
    evidence_cells: tuple[ActionPreparationEvidenceCell, ...],
    method_config: ObjectIdentity,
    proof_owner: ActionPreparationRecurrenceProofOwner,
) -> ActionPreparationRecurrenceReceipt:
    if (
        active_action_word.object_schema != OccurrenceActionWord.SCHEMA
        or matched_hold_action_word.object_schema != OccurrenceActionWord.SCHEMA
        or active_action_word == matched_hold_action_word
    ):
        raise ValueError("recurrence evaluation requires distinct exact active/HOLD words")
    require_sorted_unique_ids(
        evidence_cells,
        attribute="cell_id",
        field_name="evidence_cells",
    )
    preparations = tuple(
        sorted(
            {
                value.preparation_unit_id
                for value in evidence_cells
                if value.requested_action_word == active_action_word
            }
        )
    )
    expected = {
        (preparation, action, member)
        for preparation in preparations
        for action in (active_action_word, matched_hold_action_word)
        for member in model_member_ids
    }
    observed = {
        (
            value.preparation_unit_id,
            value.requested_action_word,
            value.model_member_id,
        )
        for value in evidence_cells
    }
    reasons: set[str] = set()
    if len(preparations) < 2:
        reasons.add("ACTIVE_ACTION_UNDER_REPLICATED")
    if observed != expected or len(evidence_cells) != len(expected):
        reasons.add("ACTION_HOLD_MEMBER_PRODUCT_INCOMPLETE")
    if any(
        value.task_id != task_id
        or value.arm_id != arm_id
        or value.method_config != method_config
        or value.causal_domain is not evidence_source.causal_domain
        or value.outcome_access is not evidence_source.outcome_access
        for value in evidence_cells
    ):
        reasons.add("RECURRENCE_EVIDENCE_BINDING_OR_DOMAIN_MISMATCH")
    if len({value.delivery_receipt for value in evidence_cells}) != len(evidence_cells) or len(
        {value.response_evidence for value in evidence_cells}
    ) != len(evidence_cells):
        reasons.add("RECURRENCE_CELL_EVIDENCE_REUSED")
    if (
        len({value.recurrence_predicate for value in evidence_cells}) != 1
        or len({value.uncertainty_rule for value in evidence_cells}) != 1
    ):
        reasons.add("RECURRENCE_RULE_BINDING_MISMATCH")
    if any(not value.matched_clone for value in evidence_cells):
        reasons.add("RECURRENCE_MATCHED_CLONE_FAILED")
    if any(not value.delivery_valid for value in evidence_cells):
        reasons.add("RECURRENCE_DELIVERY_INVALID")
    if any(not value.recurrence_passed for value in evidence_cells):
        reasons.add("ACTION_HOLD_RECURRENCE_FAILED")
    preparation_fingerprints = {
        preparation: {
            value.source_fingerprint
            for value in evidence_cells
            if value.preparation_unit_id == preparation
        }
        for preparation in preparations
    }
    if any(len(values) != 1 for values in preparation_fingerprints.values()) or len(
        {next(iter(values)) for values in preparation_fingerprints.values() if values}
    ) != len(preparations):
        reasons.add("RECURRENCE_PREPARATION_FINGERPRINT_NOT_DISTINCT")
    source_inputs = set(evidence_source.evidence_inputs)
    if any(
        value.delivery_receipt not in source_inputs or value.response_evidence not in source_inputs
        for value in evidence_cells
    ):
        reasons.add("RECURRENCE_EVIDENCE_NOT_IN_PRECOMMITMENT_SOURCE")
    expected_roles = {
        RecurrenceEvidenceSourceKind.ACQUISITION_TRACE: {
            AcquisitionEvidenceRole.COMMON_PANEL,
            AcquisitionEvidenceRole.ADAPTIVE_QUERY,
        },
        RecurrenceEvidenceSourceKind.MAPPED_DEVELOPMENT_SUPPORT_ATLAS: {
            AcquisitionEvidenceRole.MAPPED_DEVELOPMENT_DONOR,
        },
    }[evidence_source.source_kind]
    if any(value.acquisition_role not in expected_roles for value in evidence_cells):
        reasons.add("RECURRENCE_EVIDENCE_ROLE_SOURCE_MISMATCH")
    status = (
        ActionPreparationRecurrenceStatus.SUPPORTED
        if not reasons
        else ActionPreparationRecurrenceStatus.NOT_SUPPORTED
    )
    return ActionPreparationRecurrenceReceipt(
        receipt_id=(
            f"action-preparation-recurrence.{task_id}.{arm_id}.{active_action_word.object_id}"
        ),
        evidence_source=evidence_source,
        task_id=task_id,
        arm_id=arm_id,
        active_action_word=active_action_word,
        matched_hold_action_word=matched_hold_action_word,
        preparation_unit_ids=preparations,
        model_member_ids=model_member_ids,
        evidence_cells=evidence_cells,
        method_config=method_config,
        proof_owner=ObjectIdentity.from_record(proof_owner.owner_id, proof_owner),
        status=status,
        evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
        reason_codes=tuple(sorted(reasons)),
    )


__all__ = [
    'AcquisitionBundleAssessment',
    'AcquisitionCheckpointAssessment',
    "AcquisitionCheckpointCellDisposition",
    'AcquisitionCheckpointCell',
    "AcquisitionDecisionKind",
    'AcquisitionDecision',
    "AcquisitionEvidenceRole",
    'AcquisitionPriorityComponent',
    "AcquisitionPriorityTier",
    'AcquisitionRoundState',
    'AcquisitionTraceReceipt',
    'ActionPreparationEvidenceCell',
    'ActionPreparationRecurrenceEvidenceSource',
    'ActionPreparationRecurrenceQualificationBinding',
    'ActionPreparationRecurrenceProofOwner',
    'ActionPreparationRecurrenceReceipt',
    "ActionPreparationRecurrenceStatus",
    "RecurrenceEvidenceDomain",
    "RecurrenceEvidenceSourceKind",
    "FixedRoundAcquisitionDecision",
    "FixedRoundAcquisitionReceipt",
    "FixedRoundArmAssessment",
    "FixedRoundCompletenessReceipt",
    "FixedRoundDecisionItem",
    "FixedRoundDisposition",
    "QueryBundleReceiptDisposition",
    'QueryBundleReceipt',
    'QueryCoordinateOutput',
    "RecurrenceQualificationDisposition",
    "bind_action_preparation_recurrence_qualification",
    "close_query_round",
    "decide_bounded_query",
    "evaluate_action_preparation_recurrence",
    "recurrence_evidence_source_from_acquisition_trace",
    "recurrence_qualification_obstructions",
    "select_fixed_round",
    "validate_fixed_round_completeness",
    "validate_acquisition_trace",
]
