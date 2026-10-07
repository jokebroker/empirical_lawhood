"Outcome-visible qualification of one exact finite action occurrence.\n\nThis module is intentionally below and disjoint from admission reachability and from\ncontroller authoring.  It answers only whether the already-issued finite\nactive/HOLD coordinate was represented, delivered and observed exactly enough\nto be admitted to the historical-fidelity recurrence experiment.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage, ActionStageEvent, OccurrenceActionWord, ActionWordMode, ActionWordSupportStatus, ObservedActionOccurrence
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.kernel.time import ClockCoordinate
from empirical_lawhood.planning.finite_action_alternate_branch_selection import FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_BRANCH_ID, FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_REASON_PRECEDENCE, FiniteActionAlternateBranchSelectionBridge
from empirical_lawhood.planning.preissue_branch_selection import TOKAMAK_FINITE_ACTION_RECURRENCE_BRANCH_ID, TOKAMAK_FINITE_ACTION_RECURRENCE_REASON_CODE_PRECEDENCE, PreissueBranchSelectionDisposition, PreissueBranchSelectionReceipt


FiniteActionBranchSelection = PreissueBranchSelectionReceipt | FiniteActionAlternateBranchSelectionBridge


def finite_action_branch_selection_receipt_id(
    selection: FiniteActionBranchSelection,
) -> str:
    if isinstance(selection, FiniteActionAlternateBranchSelectionBridge):
        return selection.receipt_id
    return selection.selection_receipt_id


def finite_action_branch_selection_identity(
    selection: FiniteActionBranchSelection,
) -> ObjectIdentity:
    return ObjectIdentity.from_record(
        finite_action_branch_selection_receipt_id(selection),
        selection,
    )


def finite_action_branch_operator_identity(
    selection: FiniteActionBranchSelection,
) -> ObjectIdentity:
    if isinstance(selection, FiniteActionAlternateBranchSelectionBridge):
        return selection.g2_operator_feasibility
    return ObjectIdentity.from_record(
        selection.operator_feasibility.assessment_id,
        selection.operator_feasibility,
    )


def _is_historical_fidelity_recurrence_selection(selection: FiniteActionBranchSelection) -> bool:
    if isinstance(selection, FiniteActionAlternateBranchSelectionBridge):
        return (
            selection.selected_branch_id == FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_BRANCH_ID
            and selection.controlling_reason_code in FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_REASON_PRECEDENCE
        )
    return (
        selection.disposition is PreissueBranchSelectionDisposition.ALTERNATE_SELECTED
        and selection.selected_branch_id == TOKAMAK_FINITE_ACTION_RECURRENCE_BRANCH_ID
        and selection.controlling_reason_code in TOKAMAK_FINITE_ACTION_RECURRENCE_REASON_CODE_PRECEDENCE
    )


FINITE_ACTION_OCCURRENCE_COUNT = 6
FINITE_ACTION_PREFIX_COUNT = 7
FINITE_ACTION_DELIVERY_TOLERANCE_A = Decimal("0.5")
FINITE_ACTION_FUTURE_HOLD_TOLERANCE = Decimal("1e-12")
FINITE_ACTION_ACTIVE_REQUEST_COORDINATES = tuple(Decimal(value) for value in range(104, 110))
FINITE_ACTION_ACTIVE_RESPONSE_COORDINATES = tuple(Decimal(value) for value in range(105, 111))
FINITE_ACTION_FUTURE_REQUEST_COORDINATES = tuple(Decimal(value) for value in range(111, 117))
FINITE_ACTION_FUTURE_RESPONSE_COORDINATES = tuple(Decimal(value) for value in range(112, 118))
_FINITE_ACTION_OCCURRENCE_QUALIFICATION_RECEIPT_SCHEMA = (
    'empirical-lawhood/planning/finite-action-occurrence-qualification-receipt'
)

FINITE_ACTION_TOKAMAK_PREPARED_DENOMINATOR_ID = 'denominator.tokamak-control.gym-torax-prepared'
FINITE_ACTION_TOKAMAK_LOCAL_SUPPORT_IDS = (
    'local-support.tokamak-control.joint-depth-lower',
    'local-support.tokamak-control.joint-depth-middle',
    'local-support.tokamak-control.joint-depth-upper',
)
FINITE_ACTION_TOKAMAK_ACTIVE_WORD_ID = 'action-word.tokamak-control.lower-ip'
FINITE_ACTION_TOKAMAK_HOLD_WORD_ID = 'action-word.tokamak-control.native-hold'
FINITE_ACTION_TOKAMAK_WRONG_SIGN_WORD_ID = 'action-word.tokamak-control.wrong-sign-ip'
FINITE_ACTION_TOKAMAK_FUTURE_WORD_ID = 'action-word.tokamak-control.future-ip'

FINITE_ACTION_OCCURRENCE_REASON_PRECEDENCE = (
    "FINITE_ACTION_LOCAL_LAW_BINDING_UNSUPPORTED",
    "FINITE_ACTION_WORD_CONTRACT_MISMATCH",
    "FINITE_ACTION_OCCURRENCE_ROSTER_INCOMPLETE",
    "FINITE_ACTION_DELIVERY_STAGE_INCOMPLETE",
    "FINITE_ACTION_DELIVERY_MISMATCH",
    "FINITE_ACTION_DELIVERY_REJECTED_CLIPPED_SUBSTITUTED_OR_TERMINATED",
    "FINITE_ACTION_ACTIVE_CAUSAL_ONSET_TOO_EARLY",
    "FINITE_ACTION_FUTURE_NULL_VIOLATED_THROUGH_STATE_110",
    "FINITE_ACTION_FUTURE_CAUSAL_ONSET_TOO_EARLY",
    "FINITE_ACTION_NONCOMPENSATING_PREDICATE_FAILED",
    "FINITE_ACTION_NONCOMPENSATING_PREDICATE_UNEVALUABLE",
    "FINITE_ACTION_CAUSAL_OPERAND_UNAVAILABLE",
)


class FiniteActionWordRole(StrEnum):
    ACTIVE = "ACTIVE"
    FUTURE_CAUSAL_FALSIFIER = "FUTURE_CAUSAL_FALSIFIER"
    MATCHED_HOLD = "MATCHED_HOLD"
    WRONG_SIGN_FALSIFIER = "WRONG_SIGN_FALSIFIER"


class FiniteActionOccurrencePredicateKind(StrEnum):
    AUTHORITY = "AUTHORITY"
    BASELINE_PRESERVATION = "BASELINE_PRESERVATION"
    EFFORT = "EFFORT"
    OBSERVATION_VALIDITY = "OBSERVATION_VALIDITY"
    SINK = "SINK"
    TARGET = "TARGET"
    UNCERTAINTY = "UNCERTAINTY"


FINITE_ACTION_REQUIRED_PREDICATE_KINDS = tuple(FiniteActionOccurrencePredicateKind)


class FiniteActionOccurrenceDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class FiniteActionOccurrenceClockProjection(CanonicalRecord):
    """One exact occurrence -> request/state-clock projection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-occurrence-clock-projection'

    projection_id: str
    occurrence_id: str
    request_coordinate: ClockCoordinate
    response_coordinate: ClockCoordinate

    def __post_init__(self) -> None:
        validate_stable_id(self.projection_id, field_name="projection_id")
        validate_stable_id(self.occurrence_id, field_name="occurrence_id")
        if self.response_coordinate.coordinate <= self.request_coordinate.coordinate:
            raise ValueError("finite-action response projection must follow its request")


@dataclass(frozen=True, slots=True)
class FiniteActionOccurrencePredicateSpec(CanonicalRecord):
    """One independently decisive occurrence-level predicate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-occurrence-predicate-spec'

    predicate_id: str
    kind: FiniteActionOccurrencePredicateKind

    def __post_init__(self) -> None:
        validate_stable_id(self.predicate_id, field_name="predicate_id")


def _word_coordinates(word: OccurrenceActionWord) -> tuple[Decimal, ...]:
    return tuple(value.requested.coordinate.coordinate for value in word.occurrences)


def _validate_exact_word(
    word: OccurrenceActionWord,
    *,
    word_id: str,
    coordinates: tuple[Decimal, ...],
) -> None:
    if (
        word.word_id != word_id
        or word.mode is not ActionWordMode.SEQUENTIAL
        or word.support_status is not ActionWordSupportStatus.SUPPORTED
        or word.reason_codes
        or len(word.occurrences) != FINITE_ACTION_OCCURRENCE_COUNT
        or len(word.groups) != FINITE_ACTION_OCCURRENCE_COUNT
        or len(word.prefix_support_ids) != FINITE_ACTION_PREFIX_COUNT
        or _word_coordinates(word) != coordinates
    ):
        raise ValueError("finite-action word differs from the exact six-occurrence chart")
    if any(
        len(group.members) != 1
        or group.members[0].occurrence_id != occurrence.occurrence_id
        or occurrence.duration != Decimal(1)
        or occurrence.duration_unit != "s"
        for group, occurrence in zip(word.groups, word.occurrences, strict=True)
    ):
        raise ValueError("finite-action word changes singleton order or one-second duration")


def _same_word_context(left: OccurrenceActionWord, right: OccurrenceActionWord) -> bool:
    return (
        left.denominator_id,
        left.retained_history_id,
        left.receiver_id,
        left.horizon_id,
        left.ordering_clock_id,
        left.ordering_time_unit,
        left.ordering_coordinate_frame,
        left.ordering_origin,
    ) == (
        right.denominator_id,
        right.retained_history_id,
        right.receiver_id,
        right.horizon_id,
        right.ordering_clock_id,
        right.ordering_time_unit,
        right.ordering_coordinate_frame,
        right.ordering_origin,
    )


def _validate_projection_roster(
    word: OccurrenceActionWord,
    projections: tuple[FiniteActionOccurrenceClockProjection, ...],
    *,
    response_coordinates: tuple[Decimal, ...],
) -> None:
    require_sorted_unique_ids(
        projections,
        attribute="occurrence_id",
        field_name="projections",
    )
    expected = {value.occurrence_id: value for value in word.occurrences}
    if len(projections) != FINITE_ACTION_OCCURRENCE_COUNT or set(
        value.occurrence_id for value in projections
    ) != set(expected):
        raise ValueError("finite-action projection roster differs from the word")
    ordered = sorted(
        projections,
        key=lambda value: expected[value.occurrence_id].requested.coordinate.coordinate,
    )
    if (
        tuple(value.request_coordinate for value in ordered)
        != tuple(value.requested.coordinate for value in word.occurrences)
        or tuple(value.response_coordinate.coordinate for value in ordered) != response_coordinates
    ):
        raise ValueError("finite-action occurrence projections change native coordinates")


@dataclass(frozen=True, slots=True)
class FiniteActionOccurrenceQualificationTemplate(CanonicalRecord):
    "Preissue occurrence design containing no branch or local-law outcome record."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-occurrence-qualification-template'

    template_id: str
    expected_qualification_spec_id: str
    expected_qualification_receipt_id: str
    expected_qualification_receipt_schema: str
    expected_branch_selection_receipt_id: str
    expected_branch_selection_receipt_schema: str
    expected_law_qualification_result_id: str
    expected_law_qualification_result_schema: str
    local_support_id: str
    model_member_id: str
    active_word: OccurrenceActionWord
    matched_hold_word: OccurrenceActionWord
    wrong_sign_word: OccurrenceActionWord
    future_word: OccurrenceActionWord
    active_projections: tuple[FiniteActionOccurrenceClockProjection, ...]
    future_projections: tuple[FiniteActionOccurrenceClockProjection, ...]
    predicates: tuple[FiniteActionOccurrencePredicateSpec, ...]
    retained_history: ObjectIdentity
    receiver: ObjectIdentity
    horizon: ObjectIdentity
    active_first_allowed_response_state: int
    hold_equality_last_state: int
    future_first_allowed_response_state: int
    delivery_tolerance_a: Decimal
    future_hold_tolerance: Decimal
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.template_id, field_name="template_id")
        validate_stable_id(
            self.expected_qualification_spec_id,
            field_name="expected_qualification_spec_id",
        )
        validate_stable_id(
            self.expected_qualification_receipt_id,
            field_name="expected_qualification_receipt_id",
        )
        validate_stable_id(
            self.expected_branch_selection_receipt_id,
            field_name="expected_branch_selection_receipt_id",
        )
        validate_stable_id(self.expected_law_qualification_result_id, field_name='expected_law_qualification_result_id')
        validate_schema(self.expected_qualification_receipt_schema)
        validate_schema(self.expected_branch_selection_receipt_schema)
        validate_schema(self.expected_law_qualification_result_schema)
        if (
            self.expected_qualification_receipt_schema
            != _FINITE_ACTION_OCCURRENCE_QUALIFICATION_RECEIPT_SCHEMA
            or self.expected_branch_selection_receipt_schema
            not in {
                PreissueBranchSelectionReceipt.SCHEMA,
                FiniteActionAlternateBranchSelectionBridge.SCHEMA,
            }
            or self.expected_law_qualification_result_schema != LawQualificationResult.SCHEMA
        ):
            raise ValueError("finite-action template expects another receipt/branch/local-law schema")
        _validate_occurrence_design(self)
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("finite-action occurrence template must be outcome-blind")

    def word(self, role: FiniteActionWordRole) -> OccurrenceActionWord:
        return {
            FiniteActionWordRole.ACTIVE: self.active_word,
            FiniteActionWordRole.FUTURE_CAUSAL_FALSIFIER: self.future_word,
            FiniteActionWordRole.MATCHED_HOLD: self.matched_hold_word,
            FiniteActionWordRole.WRONG_SIGN_FALSIFIER: self.wrong_sign_word,
        }[role]


@dataclass(frozen=True, slots=True)
class FiniteActionOccurrenceQualificationTemplateSet(CanonicalRecord):
    "The complete preissue three-region by two-member finite-action recurrence occurrence design."

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/planning/finite-action-occurrence-qualification-template-set'
    )

    template_set_id: str
    templates: tuple[FiniteActionOccurrenceQualificationTemplate, ...]
    robust_intersection_required: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.template_set_id, field_name="template_set_id")
        require_sorted_unique_ids(
            self.templates,
            attribute="template_id",
            field_name="templates",
        )
        require_sorted_unique_strings(
            self.expected_qualification_receipt_ids,
            field_name="expected_qualification_receipt_ids",
            allow_empty=False,
        )
        members = tuple(sorted({value.model_member_id for value in self.templates}))
        expected_products = {
            (support_id, member_id)
            for support_id in FINITE_ACTION_TOKAMAK_LOCAL_SUPPORT_IDS
            for member_id in members
        }
        if (
            len(members) != 2
            or len(self.templates) != 6
            or {(value.local_support_id, value.model_member_id) for value in self.templates}
            != expected_products
        ):
            raise ValueError("finite-action template set requires the exact 3x2 product")
        first = self.templates[0]
        if any(
            (
                value.expected_branch_selection_receipt_id,
                value.expected_branch_selection_receipt_schema,
                value.expected_law_qualification_result_id,
                value.expected_law_qualification_result_schema,
                value.active_word,
                value.matched_hold_word,
                value.wrong_sign_word,
                value.future_word,
                value.retained_history,
                value.receiver,
                value.horizon,
            )
            != (
                first.expected_branch_selection_receipt_id,
                first.expected_branch_selection_receipt_schema,
                first.expected_law_qualification_result_id,
                first.expected_law_qualification_result_schema,
                first.active_word,
                first.matched_hold_word,
                first.wrong_sign_word,
                first.future_word,
                first.retained_history,
                first.receiver,
                first.horizon,
            )
            for value in self.templates[1:]
        ):
            raise ValueError("finite-action template set changes shared branch/local-law/action context")
        if (
            not self.robust_intersection_required
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("finite-action template set must freeze a blind intersection")

    @property
    def expected_qualification_receipt_ids(self) -> tuple[str, ...]:
        """Exact outcome-free receipt roster declared by the six templates."""

        return tuple(sorted(value.expected_qualification_receipt_id for value in self.templates))


@dataclass(frozen=True, slots=True)
class FiniteActionOccurrenceQualificationSpec(CanonicalRecord):
    "Exact occurrence specification after local-law qualification bound to its preissue template."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-occurrence-qualification-spec'

    qualification_spec_id: str
    expected_qualification_receipt_id: str
    expected_qualification_receipt_schema: str
    frozen_template: ObjectIdentity
    branch_selection: FiniteActionBranchSelection
    law_qualification_result: LawQualificationResult
    local_support_id: str
    model_member_id: str
    active_word: OccurrenceActionWord
    matched_hold_word: OccurrenceActionWord
    wrong_sign_word: OccurrenceActionWord
    future_word: OccurrenceActionWord
    active_projections: tuple[FiniteActionOccurrenceClockProjection, ...]
    future_projections: tuple[FiniteActionOccurrenceClockProjection, ...]
    predicates: tuple[FiniteActionOccurrencePredicateSpec, ...]
    retained_history: ObjectIdentity
    receiver: ObjectIdentity
    horizon: ObjectIdentity
    active_first_allowed_response_state: int
    hold_equality_last_state: int
    future_first_allowed_response_state: int
    delivery_tolerance_a: Decimal
    future_hold_tolerance: Decimal
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_spec_id, field_name="qualification_spec_id")
        validate_stable_id(
            self.expected_qualification_receipt_id,
            field_name="expected_qualification_receipt_id",
        )
        validate_schema(self.expected_qualification_receipt_schema)
        if (
            self.expected_qualification_receipt_schema
            != _FINITE_ACTION_OCCURRENCE_QUALIFICATION_RECEIPT_SCHEMA
        ):
            raise ValueError("finite-action specification expects another receipt schema")
        if not _is_historical_fidelity_recurrence_selection(self.branch_selection):
            raise ValueError("finite-action qualification requires the sealed input-output operator feasibility alternate-branch selection")
        law = self.law_qualification_result.response_law
        if (
            self.law_qualification_result.scientific_status is not ScientificStatus.SUPPORTED
            or law is None
            or self.local_support_id not in law.obligations.support.denominator_cell_ids
            or FINITE_ACTION_TOKAMAK_PREPARED_DENOMINATOR_ID
            not in law.obligations.support.denominator_cell_ids
        ):
            raise ValueError("finite-action coordinate lacks exact support of an exactly qualified local law")
        if self.local_support_id not in FINITE_ACTION_TOKAMAK_LOCAL_SUPPORT_IDS:
            raise ValueError("finite-action coordinate is outside the three J3-BT regions")
        _validate_occurrence_design(self)
        template = _template_from_occurrence_spec(self)
        if self.frozen_template != ObjectIdentity.from_record(template.template_id, template):
            raise ValueError("finite-action specification differs from its frozen template")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("finite-action specification after local-law qualification has the wrong access ceiling")

    def word(self, role: FiniteActionWordRole) -> OccurrenceActionWord:
        return {
            FiniteActionWordRole.ACTIVE: self.active_word,
            FiniteActionWordRole.FUTURE_CAUSAL_FALSIFIER: self.future_word,
            FiniteActionWordRole.MATCHED_HOLD: self.matched_hold_word,
            FiniteActionWordRole.WRONG_SIGN_FALSIFIER: self.wrong_sign_word,
        }[role]


def _validate_occurrence_design(
    value: FiniteActionOccurrenceQualificationTemplate
    | FiniteActionOccurrenceQualificationSpec,
) -> None:
    validate_stable_id(value.local_support_id, field_name="local_support_id")
    validate_stable_id(value.model_member_id, field_name="model_member_id")
    if value.local_support_id not in FINITE_ACTION_TOKAMAK_LOCAL_SUPPORT_IDS:
        raise ValueError("finite-action coordinate is outside the three J3-BT regions")
    _validate_exact_word(
        value.active_word,
        word_id=FINITE_ACTION_TOKAMAK_ACTIVE_WORD_ID,
        coordinates=FINITE_ACTION_ACTIVE_REQUEST_COORDINATES,
    )
    _validate_exact_word(
        value.matched_hold_word,
        word_id=FINITE_ACTION_TOKAMAK_HOLD_WORD_ID,
        coordinates=FINITE_ACTION_ACTIVE_REQUEST_COORDINATES,
    )
    _validate_exact_word(
        value.wrong_sign_word,
        word_id=FINITE_ACTION_TOKAMAK_WRONG_SIGN_WORD_ID,
        coordinates=FINITE_ACTION_ACTIVE_REQUEST_COORDINATES,
    )
    _validate_exact_word(
        value.future_word,
        word_id=FINITE_ACTION_TOKAMAK_FUTURE_WORD_ID,
        coordinates=FINITE_ACTION_FUTURE_REQUEST_COORDINATES,
    )
    if not all(
        _same_word_context(value.active_word, word)
        for word in (value.matched_hold_word, value.wrong_sign_word, value.future_word)
    ):
        raise ValueError("finite-action chart changes prepared D/history/receiver/horizon")
    if value.active_word.denominator_id != FINITE_ACTION_TOKAMAK_PREPARED_DENOMINATOR_ID:
        raise ValueError("finite-action word changes the prepared denominator")
    if (
        value.retained_history.object_id != value.active_word.retained_history_id
        or value.receiver.object_id != value.active_word.receiver_id
        or value.horizon.object_id != value.active_word.horizon_id
    ):
        raise ValueError("finite-action semantic identities differ from its word")
    _validate_projection_roster(
        value.active_word,
        value.active_projections,
        response_coordinates=FINITE_ACTION_ACTIVE_RESPONSE_COORDINATES,
    )
    _validate_projection_roster(
        value.future_word,
        value.future_projections,
        response_coordinates=FINITE_ACTION_FUTURE_RESPONSE_COORDINATES,
    )
    require_sorted_unique_ids(
        value.predicates,
        attribute="predicate_id",
        field_name="predicates",
    )
    if tuple(sorted(item.kind for item in value.predicates)) != tuple(
        sorted(FINITE_ACTION_REQUIRED_PREDICATE_KINDS)
    ):
        raise ValueError("finite-action qualification requires all seven predicates once")
    validate_decimal(
        value.delivery_tolerance_a,
        field_name="delivery_tolerance_a",
        minimum=Decimal(0),
    )
    validate_decimal(
        value.future_hold_tolerance,
        field_name="future_hold_tolerance",
        minimum=Decimal(0),
    )
    if (
        value.active_first_allowed_response_state != 105
        or value.hold_equality_last_state != 110
        or value.future_first_allowed_response_state != 112
        or value.delivery_tolerance_a != FINITE_ACTION_DELIVERY_TOLERANCE_A
        or value.future_hold_tolerance != FINITE_ACTION_FUTURE_HOLD_TOLERANCE
    ):
        raise ValueError("finite-action causal/delivery constants differ from the frozen occurrence contract")


def _template_from_occurrence_spec(
    spec: FiniteActionOccurrenceQualificationSpec,
) -> FiniteActionOccurrenceQualificationTemplate:
    return FiniteActionOccurrenceQualificationTemplate(
        template_id=spec.frozen_template.object_id,
        expected_qualification_spec_id=spec.qualification_spec_id,
        expected_qualification_receipt_id=spec.expected_qualification_receipt_id,
        expected_qualification_receipt_schema=spec.expected_qualification_receipt_schema,
        expected_branch_selection_receipt_id=(
            finite_action_branch_selection_receipt_id(spec.branch_selection)
        ),
        expected_branch_selection_receipt_schema=spec.branch_selection.SCHEMA,
        expected_law_qualification_result_id=spec.law_qualification_result.result_id,
        expected_law_qualification_result_schema=LawQualificationResult.SCHEMA,
        local_support_id=spec.local_support_id,
        model_member_id=spec.model_member_id,
        active_word=spec.active_word,
        matched_hold_word=spec.matched_hold_word,
        wrong_sign_word=spec.wrong_sign_word,
        future_word=spec.future_word,
        active_projections=spec.active_projections,
        future_projections=spec.future_projections,
        predicates=spec.predicates,
        retained_history=spec.retained_history,
        receiver=spec.receiver,
        horizon=spec.horizon,
        active_first_allowed_response_state=spec.active_first_allowed_response_state,
        hold_equality_last_state=spec.hold_equality_last_state,
        future_first_allowed_response_state=spec.future_first_allowed_response_state,
        delivery_tolerance_a=spec.delivery_tolerance_a,
        future_hold_tolerance=spec.future_hold_tolerance,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


@dataclass(frozen=True, slots=True)
class FiniteActionOccurrenceTemplateBindingReceipt(CanonicalRecord):
    "Identity-only deterministic attachment of branch and supported local law records."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-occurrence-template-binding-receipt'

    receipt_id: str
    template: ObjectIdentity
    branch_selection: ObjectIdentity
    law_qualification_result: ObjectIdentity
    qualification_spec: ObjectIdentity
    expected_qualification_receipt_id: str
    expected_qualification_receipt_schema: str
    binding_implementation_id: str
    binding_implementation_sha256: str
    outcome_dependent_choice: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    scientific_verdict: None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.expected_qualification_receipt_id,
            field_name="expected_qualification_receipt_id",
        )
        validate_schema(self.expected_qualification_receipt_schema)
        if (
            self.expected_qualification_receipt_schema
            != _FINITE_ACTION_OCCURRENCE_QUALIFICATION_RECEIPT_SCHEMA
        ):
            raise ValueError("finite-action binding expects another qualification receipt schema")
        validate_stable_id(
            self.binding_implementation_id,
            field_name="binding_implementation_id",
        )
        validate_sha256(
            self.binding_implementation_sha256,
            field_name="binding_implementation_sha256",
        )
        expected_schemas = (
            (self.template, FiniteActionOccurrenceQualificationTemplate.SCHEMA),
            (self.law_qualification_result, LawQualificationResult.SCHEMA),
            (self.qualification_spec, FiniteActionOccurrenceQualificationSpec.SCHEMA),
        )
        if self.branch_selection.object_schema not in {
            PreissueBranchSelectionReceipt.SCHEMA,
            FiniteActionAlternateBranchSelectionBridge.SCHEMA,
        } or any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("finite-action occurrence binding contains another schema")
        if self.outcome_dependent_choice or self.scientific_verdict is not None:
            raise ValueError("finite-action occurrence binding cannot select or adjudicate")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("finite-action occurrence binding is evaluator-only after local-law qualification")


def bind_finite_action_occurrence_template(
    *,
    receipt_id: str,
    template: FiniteActionOccurrenceQualificationTemplate,
    branch_selection: FiniteActionBranchSelection,
    law_qualification_result: LawQualificationResult,
    binding_implementation_id: str,
    binding_implementation_sha256: str,
) -> tuple[
    FiniteActionOccurrenceQualificationSpec,
    FiniteActionOccurrenceTemplateBindingReceipt,
]:
    "Attach authenticated records after local-law qualification without changing the frozen design."

    if (
        finite_action_branch_selection_receipt_id(branch_selection)
        != template.expected_branch_selection_receipt_id
        or branch_selection.SCHEMA != template.expected_branch_selection_receipt_schema
        or law_qualification_result.result_id != template.expected_law_qualification_result_id
    ):
        raise ValueError("finite-action occurrence binding substitutes an expected record ID")
    spec = FiniteActionOccurrenceQualificationSpec(
        qualification_spec_id=template.expected_qualification_spec_id,
        expected_qualification_receipt_id=template.expected_qualification_receipt_id,
        expected_qualification_receipt_schema=(template.expected_qualification_receipt_schema),
        frozen_template=ObjectIdentity.from_record(template.template_id, template),
        branch_selection=branch_selection,
        law_qualification_result=law_qualification_result,
        local_support_id=template.local_support_id,
        model_member_id=template.model_member_id,
        active_word=template.active_word,
        matched_hold_word=template.matched_hold_word,
        wrong_sign_word=template.wrong_sign_word,
        future_word=template.future_word,
        active_projections=template.active_projections,
        future_projections=template.future_projections,
        predicates=template.predicates,
        retained_history=template.retained_history,
        receiver=template.receiver,
        horizon=template.horizon,
        active_first_allowed_response_state=template.active_first_allowed_response_state,
        hold_equality_last_state=template.hold_equality_last_state,
        future_first_allowed_response_state=template.future_first_allowed_response_state,
        delivery_tolerance_a=template.delivery_tolerance_a,
        future_hold_tolerance=template.future_hold_tolerance,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    receipt = FiniteActionOccurrenceTemplateBindingReceipt(
        receipt_id=receipt_id,
        template=ObjectIdentity.from_record(template.template_id, template),
        branch_selection=finite_action_branch_selection_identity(branch_selection),
        law_qualification_result=ObjectIdentity.from_record(law_qualification_result.result_id, law_qualification_result),
        qualification_spec=ObjectIdentity.from_record(spec.qualification_spec_id, spec),
        expected_qualification_receipt_id=spec.expected_qualification_receipt_id,
        expected_qualification_receipt_schema=spec.expected_qualification_receipt_schema,
        binding_implementation_id=binding_implementation_id,
        binding_implementation_sha256=binding_implementation_sha256,
        outcome_dependent_choice=False,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    return spec, receipt


class FiniteActionOccurrenceTemplateBinder:
    """Code-owned deterministic wrapper suitable for direct service composition."""

    def __init__(self, *, implementation_id: str, implementation_sha256: str) -> None:
        validate_stable_id(implementation_id, field_name="implementation_id")
        validate_sha256(implementation_sha256, field_name="implementation_sha256")
        self._implementation_id = implementation_id
        self._implementation_sha256 = implementation_sha256

    def bind(
        self,
        *,
        receipt_id: str,
        template: FiniteActionOccurrenceQualificationTemplate,
        branch_selection: FiniteActionBranchSelection,
        law_qualification_result: LawQualificationResult,
    ) -> tuple[
        FiniteActionOccurrenceQualificationSpec,
        FiniteActionOccurrenceTemplateBindingReceipt,
    ]:
        return bind_finite_action_occurrence_template(
            receipt_id=receipt_id,
            template=template,
            branch_selection=branch_selection,
            law_qualification_result=law_qualification_result,
            binding_implementation_id=self._implementation_id,
            binding_implementation_sha256=self._implementation_sha256,
        )


@dataclass(frozen=True, slots=True)
class FiniteActionWordDeliveryEvidence(CanonicalRecord):
    """Observed native delivery for one entire six-occurrence word."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-word-delivery-evidence'

    delivery_evidence_id: str
    role: FiniteActionWordRole
    action_word: ObjectIdentity
    occurrences: tuple[ObservedActionOccurrence, ...]
    clipped: bool
    rejected: bool
    substituted: bool
    early_terminated: bool
    trace: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.delivery_evidence_id, field_name="delivery_evidence_id")
        require_sorted_unique_ids(
            self.occurrences,
            attribute="expected_occurrence_id",
            field_name="occurrences",
        )


@dataclass(frozen=True, slots=True)
class FiniteActionOccurrencePredicateEvidence(CanonicalRecord):
    """One gate result; a pass can never compensate another gate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-occurrence-predicate-evidence'

    evidence_id: str
    predicate_id: str
    kind: FiniteActionOccurrencePredicateKind
    status: GateStatus
    observed: NamedDecimal | None
    evidence_identity: ObjectIdentity
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        validate_stable_id(self.predicate_id, field_name="predicate_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is GateStatus.PASS:
            if self.reason_codes:
                raise ValueError("passing finite-action predicate cannot carry reasons")
        elif not self.reason_codes:
            raise ValueError("nonpassing finite-action predicate requires a reason")


@dataclass(frozen=True, slots=True)
class FiniteActionOccurrenceEvidence(CanonicalRecord):
    """Outcome-visible source translation for the frozen occurrence spec."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-occurrence-evidence'

    evidence_id: str
    qualification_spec: ObjectIdentity
    word_deliveries: tuple[FiniteActionWordDeliveryEvidence, ...]
    predicate_evidence: tuple[FiniteActionOccurrencePredicateEvidence, ...]
    active_first_difference_state: int | None
    future_first_difference_state: int | None
    future_hold_max_abs_difference_through_state_110: Decimal | None
    decisive_evidence: tuple[ObjectIdentity, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        require_sorted_unique_ids(
            self.word_deliveries,
            attribute="delivery_evidence_id",
            field_name="word_deliveries",
        )
        if tuple(value.role for value in self.word_deliveries) != tuple(
            sorted(FiniteActionWordRole, key=lambda value: value.value)
        ):
            raise ValueError("finite-action evidence requires the exact four-role roster")
        require_sorted_unique_ids(
            self.predicate_evidence,
            attribute="evidence_id",
            field_name="predicate_evidence",
        )
        require_sorted_unique_ids(
            self.decisive_evidence,
            attribute="object_id",
            field_name="decisive_evidence",
        )
        if not self.decisive_evidence:
            raise ValueError("finite-action evidence requires decisive evidence identities")
        if self.future_hold_max_abs_difference_through_state_110 is not None:
            validate_decimal(
                self.future_hold_max_abs_difference_through_state_110,
                field_name="future_hold_max_abs_difference_through_state_110",
                minimum=Decimal(0),
            )
        if (
            self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("finite-action occurrence evidence is evaluator-visible only")


@dataclass(frozen=True, slots=True)
class FiniteActionOccurrenceDecisiveOperand(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-action-occurrence-decisive-operand'

    operand_id: str
    status: GateStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.operand_id, field_name="operand_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is GateStatus.PASS and self.reason_codes:
            raise ValueError("passing finite-action operand cannot carry reasons")
        if self.status is not GateStatus.PASS and not self.reason_codes:
            raise ValueError("nonpassing finite-action operand requires reasons")


def _event_matches(
    expected: ActionStageEvent, observed: ActionStageEvent, *, tolerance: Decimal
) -> bool:
    return (
        expected.stage is observed.stage
        and expected.quantity_id == observed.quantity_id
        and expected.native_unit == observed.native_unit
        and expected.native_action_frame == observed.native_action_frame
        and expected.native_direction == observed.native_direction
        and expected.coordinate == observed.coordinate
        and abs(expected.value - observed.value) <= tolerance
    )


def _delivery_operand(
    *,
    spec: FiniteActionOccurrenceQualificationSpec,
    delivery: FiniteActionWordDeliveryEvidence,
) -> FiniteActionOccurrenceDecisiveOperand:
    word = spec.word(delivery.role)
    if delivery.action_word != ObjectIdentity.from_record(word.word_id, word):
        return FiniteActionOccurrenceDecisiveOperand(
            operand_id=f"operand.{delivery.delivery_evidence_id}",
            status=GateStatus.UNEVALUABLE,
            reason_codes=("FINITE_ACTION_WORD_CONTRACT_MISMATCH",),
        )
    observed_by_id = {value.expected_occurrence_id: value for value in delivery.occurrences}
    if set(observed_by_id) != {value.occurrence_id for value in word.occurrences}:
        return FiniteActionOccurrenceDecisiveOperand(
            operand_id=f"operand.{delivery.delivery_evidence_id}",
            status=GateStatus.UNEVALUABLE,
            reason_codes=("FINITE_ACTION_OCCURRENCE_ROSTER_INCOMPLETE",),
        )
    if any(not value.complete for value in delivery.occurrences):
        return FiniteActionOccurrenceDecisiveOperand(
            operand_id=f"operand.{delivery.delivery_evidence_id}",
            status=GateStatus.UNEVALUABLE,
            reason_codes=("FINITE_ACTION_DELIVERY_STAGE_INCOMPLETE",),
        )
    for expected in word.occurrences:
        observed = observed_by_id[expected.occurrence_id]
        expected_events = (
            expected.requested,
            expected.accepted,
            expected.applied,
            expected.realized,
        )
        observed_events = (
            observed.requested,
            observed.accepted,
            observed.applied,
            observed.realized,
        )
        if any(
            actual is None
            or not _event_matches(
                reference,
                actual,
                tolerance=(
                    Decimal(0)
                    if reference.stage is ActionDeliveryStage.REQUESTED
                    else spec.delivery_tolerance_a
                ),
            )
            for reference, actual in zip(expected_events, observed_events, strict=True)
        ):
            return FiniteActionOccurrenceDecisiveOperand(
                operand_id=f"operand.{delivery.delivery_evidence_id}",
                status=GateStatus.FAIL,
                reason_codes=("FINITE_ACTION_DELIVERY_MISMATCH",),
            )
    if delivery.clipped or delivery.rejected or delivery.substituted or delivery.early_terminated:
        return FiniteActionOccurrenceDecisiveOperand(
            operand_id=f"operand.{delivery.delivery_evidence_id}",
            status=GateStatus.FAIL,
            reason_codes=("FINITE_ACTION_DELIVERY_REJECTED_CLIPPED_SUBSTITUTED_OR_TERMINATED",),
        )
    return FiniteActionOccurrenceDecisiveOperand(
        operand_id=f"operand.{delivery.delivery_evidence_id}",
        status=GateStatus.PASS,
        reason_codes=(),
    )


def _causal_operands(
    spec: FiniteActionOccurrenceQualificationSpec,
    evidence: FiniteActionOccurrenceEvidence,
) -> tuple[FiniteActionOccurrenceDecisiveOperand, ...]:
    active_state = evidence.active_first_difference_state
    future_state = evidence.future_first_difference_state
    equality = evidence.future_hold_max_abs_difference_through_state_110
    if active_state is None:
        active = FiniteActionOccurrenceDecisiveOperand(
            operand_id="operand.finite-action.active-causal-onset",
            status=GateStatus.UNEVALUABLE,
            reason_codes=("FINITE_ACTION_CAUSAL_OPERAND_UNAVAILABLE",),
        )
    else:
        active_passed = active_state >= spec.active_first_allowed_response_state
        active = FiniteActionOccurrenceDecisiveOperand(
            operand_id="operand.finite-action.active-causal-onset",
            status=GateStatus.PASS if active_passed else GateStatus.FAIL,
            reason_codes=() if active_passed else ("FINITE_ACTION_ACTIVE_CAUSAL_ONSET_TOO_EARLY",),
        )
    if equality is None:
        future_equality = FiniteActionOccurrenceDecisiveOperand(
            operand_id="operand.finite-action.future-hold-equality-through-0110",
            status=GateStatus.UNEVALUABLE,
            reason_codes=("FINITE_ACTION_CAUSAL_OPERAND_UNAVAILABLE",),
        )
    else:
        equality_passed = equality <= spec.future_hold_tolerance
        future_equality = FiniteActionOccurrenceDecisiveOperand(
            operand_id="operand.finite-action.future-hold-equality-through-0110",
            status=GateStatus.PASS if equality_passed else GateStatus.FAIL,
            reason_codes=(
                () if equality_passed else ("FINITE_ACTION_FUTURE_NULL_VIOLATED_THROUGH_STATE_110",)
            ),
        )
    if future_state is None:
        future = FiniteActionOccurrenceDecisiveOperand(
            operand_id="operand.finite-action.future-causal-onset",
            status=GateStatus.UNEVALUABLE,
            reason_codes=("FINITE_ACTION_CAUSAL_OPERAND_UNAVAILABLE",),
        )
    else:
        future_passed = future_state >= spec.future_first_allowed_response_state
        future = FiniteActionOccurrenceDecisiveOperand(
            operand_id="operand.finite-action.future-causal-onset",
            status=GateStatus.PASS if future_passed else GateStatus.FAIL,
            reason_codes=() if future_passed else ("FINITE_ACTION_FUTURE_CAUSAL_ONSET_TOO_EARLY",),
        )
    return active, future_equality, future


def _occurrence_operands(
    spec: FiniteActionOccurrenceQualificationSpec,
    evidence: FiniteActionOccurrenceEvidence,
) -> tuple[FiniteActionOccurrenceDecisiveOperand, ...]:
    spec_identity = ObjectIdentity.from_record(spec.qualification_spec_id, spec)
    if evidence.qualification_spec != spec_identity:
        raise ValueError("finite-action evidence substitutes another specification")
    deliveries = tuple(
        _delivery_operand(spec=spec, delivery=value) for value in evidence.word_deliveries
    )
    predicate_specs = {value.predicate_id: value for value in spec.predicates}
    observed_ids = {value.predicate_id for value in evidence.predicate_evidence}
    predicates: tuple[FiniteActionOccurrenceDecisiveOperand, ...]
    if observed_ids != set(predicate_specs) or len(evidence.predicate_evidence) != len(
        predicate_specs
    ):
        predicates = (
            FiniteActionOccurrenceDecisiveOperand(
                operand_id="operand.finite-action.predicate-roster",
                status=GateStatus.UNEVALUABLE,
                reason_codes=("FINITE_ACTION_NONCOMPENSATING_PREDICATE_UNEVALUABLE",),
            ),
        )
    else:
        predicate_values: list[FiniteActionOccurrenceDecisiveOperand] = []
        for value in evidence.predicate_evidence:
            expected = predicate_specs[value.predicate_id]
            reasons: tuple[str, ...]
            if value.kind is not expected.kind:
                status = GateStatus.UNEVALUABLE
                reasons = ("FINITE_ACTION_NONCOMPENSATING_PREDICATE_UNEVALUABLE",)
            elif value.status is GateStatus.PASS:
                status = GateStatus.PASS
                reasons = ()
            elif value.status is GateStatus.FAIL:
                status = GateStatus.FAIL
                reasons = ("FINITE_ACTION_NONCOMPENSATING_PREDICATE_FAILED",)
            else:
                status = GateStatus.UNEVALUABLE
                reasons = ("FINITE_ACTION_NONCOMPENSATING_PREDICATE_UNEVALUABLE",)
            predicate_values.append(
                FiniteActionOccurrenceDecisiveOperand(
                    operand_id=f"operand.{value.evidence_id}",
                    status=status,
                    reason_codes=reasons,
                )
            )
        predicates = tuple(predicate_values)
    return tuple(
        sorted(
            (*deliveries, *_causal_operands(spec, evidence), *predicates),
            key=lambda value: value.operand_id,
        )
    )


def _occurrence_decision(
    operands: tuple[FiniteActionOccurrenceDecisiveOperand, ...],
) -> tuple[FiniteActionOccurrenceDisposition, tuple[str, ...]]:
    reasons = tuple(sorted({reason for value in operands for reason in value.reason_codes}))
    if any(value.status is GateStatus.UNEVALUABLE for value in operands):
        return FiniteActionOccurrenceDisposition.UNEVALUABLE, reasons
    if any(value.status is GateStatus.FAIL for value in operands):
        return FiniteActionOccurrenceDisposition.NOT_SUPPORTED, reasons
    return FiniteActionOccurrenceDisposition.SUPPORTED, ()


@dataclass(frozen=True, slots=True)
class FiniteActionOccurrenceQualificationReceipt(CanonicalRecord):
    "Sole finite-occurrence qualification receipt; never admission or reachability."

    SCHEMA: ClassVar[str] = _FINITE_ACTION_OCCURRENCE_QUALIFICATION_RECEIPT_SCHEMA

    receipt_id: str
    qualification_spec: FiniteActionOccurrenceQualificationSpec
    evidence: FiniteActionOccurrenceEvidence
    occurrence_ids: tuple[str, ...]
    decisive_operands: tuple[FiniteActionOccurrenceDecisiveOperand, ...]
    disposition: FiniteActionOccurrenceDisposition
    reason_codes: tuple[str, ...]
    g2_operator_feasibility: ObjectIdentity
    g2_controlling_reason_code: str
    claim_ceiling: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if (
            self.receipt_id != self.qualification_spec.expected_qualification_receipt_id
            or self.qualification_spec.expected_qualification_receipt_schema != self.SCHEMA
        ):
            raise ValueError(
                "finite-action receipt differs from the expected qualification receipt"
            )
        validate_nonempty(self.claim_ceiling, field_name="claim_ceiling")
        require_sorted_unique_strings(
            self.occurrence_ids,
            field_name="occurrence_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.decisive_operands,
            attribute="operand_id",
            field_name="decisive_operands",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_occurrence_ids = tuple(
            sorted(
                occurrence.occurrence_id
                for role in FiniteActionWordRole
                for occurrence in self.qualification_spec.word(role).occurrences
            )
        )
        if self.occurrence_ids != expected_occurrence_ids:
            raise ValueError("finite-action receipt omits or adds kernel occurrences")
        expected_operands = _occurrence_operands(self.qualification_spec, self.evidence)
        expected_disposition, expected_reasons = _occurrence_decision(expected_operands)
        if (
            self.decisive_operands != expected_operands
            or self.disposition is not expected_disposition
            or self.reason_codes != expected_reasons
        ):
            raise ValueError("finite-action receipt is not mechanically derived")
        branch = self.qualification_spec.branch_selection
        if (
            self.g2_operator_feasibility != finite_action_branch_operator_identity(branch)
            or self.g2_controlling_reason_code != branch.controlling_reason_code
        ):
            raise ValueError("finite-action receipt drops its controlling input-output operator feasibility obstruction")
        if (
            self.claim_ceiling != "FINITE_ACTION_OCCURRENCE_ONLY"
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("finite-action occurrence receipt exceeds its evidence ceiling")


class FiniteActionOccurrenceEvaluator:
    """Pure evaluator with no admission, reachability, action or issue authority."""

    def evaluate(
        self,
        *,
        receipt_id: str,
        spec: FiniteActionOccurrenceQualificationSpec,
        evidence: FiniteActionOccurrenceEvidence,
    ) -> FiniteActionOccurrenceQualificationReceipt:
        if receipt_id != spec.expected_qualification_receipt_id:
            raise ValueError(
                "finite-action evaluator substitutes the expected qualification receipt ID"
            )
        operands = _occurrence_operands(spec, evidence)
        disposition, reasons = _occurrence_decision(operands)
        branch = spec.branch_selection
        controlling = branch.controlling_reason_code
        if controlling is None:  # pragma: no cover - narrowed by the frozen spec
            raise AssertionError("Finite-action recurrence selection lost its controlling input-output operator feasibility obstruction")
        return FiniteActionOccurrenceQualificationReceipt(
            receipt_id=receipt_id,
            qualification_spec=spec,
            evidence=evidence,
            occurrence_ids=tuple(
                sorted(
                    occurrence.occurrence_id
                    for role in FiniteActionWordRole
                    for occurrence in spec.word(role).occurrences
                )
            ),
            decisive_operands=operands,
            disposition=disposition,
            reason_codes=reasons,
            g2_operator_feasibility=finite_action_branch_operator_identity(branch),
            g2_controlling_reason_code=controlling,
            claim_ceiling="FINITE_ACTION_OCCURRENCE_ONLY",
            evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        )


__all__ = [
    "FINITE_ACTION_ACTIVE_REQUEST_COORDINATES",
    "FINITE_ACTION_ACTIVE_RESPONSE_COORDINATES",
    "FINITE_ACTION_DELIVERY_TOLERANCE_A",
    "FINITE_ACTION_FUTURE_HOLD_TOLERANCE",
    "FINITE_ACTION_FUTURE_REQUEST_COORDINATES",
    "FINITE_ACTION_FUTURE_RESPONSE_COORDINATES",
    'FINITE_ACTION_TOKAMAK_LOCAL_SUPPORT_IDS',
    "FINITE_ACTION_OCCURRENCE_COUNT",
    "FINITE_ACTION_OCCURRENCE_REASON_PRECEDENCE",
    "FINITE_ACTION_PREFIX_COUNT",
    "FINITE_ACTION_REQUIRED_PREDICATE_KINDS",
    'FiniteActionOccurrenceClockProjection',
    'FiniteActionOccurrenceDecisiveOperand',
    'FiniteActionOccurrenceDisposition',
    'FiniteActionOccurrenceEvaluator',
    'FiniteActionOccurrenceEvidence',
    'FiniteActionOccurrencePredicateEvidence',
    'FiniteActionOccurrencePredicateKind',
    'FiniteActionOccurrencePredicateSpec',
    'FiniteActionOccurrenceQualificationReceipt',
    'FiniteActionOccurrenceQualificationSpec',
    'FiniteActionOccurrenceQualificationTemplateSet',
    'FiniteActionOccurrenceQualificationTemplate',
    'FiniteActionOccurrenceTemplateBinder',
    'FiniteActionOccurrenceTemplateBindingReceipt',
    'FiniteActionWordDeliveryEvidence',
    'FiniteActionWordRole',
    'bind_finite_action_occurrence_template',
]
