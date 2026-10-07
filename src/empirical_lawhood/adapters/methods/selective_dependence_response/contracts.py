"Strict target-neutral contracts for selective dependence response.\n\nThe package deliberately describes a finite experimental grammar rather than\ntransporting target values or a structural recurrence answer word.  Target physics remains in\nsimulator adapters; these records carry only typed scientific relations,\ncomplete-unit observations and compact adjudication state.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar
import re

from empirical_lawhood.adapters.independent_source_exports import IndependentSourceExport
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID = "selective-context-and-action-fibre-lawhood"
SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS = ("A", "D", "H", "R", "tau")
SELECTIVE_DEPENDENCE_RESPONSE_COMPONENT_IDS = (
    "action-fibre",
    "finite-law",
    "selective-exchange",
    "support-boundary",
)


def digest_ids(values: tuple[str, ...]) -> str:
    """Canonical digest for one already sorted identity roster."""

    require_sorted_unique_strings(values, field_name="values")
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


class SelectiveDependenceResponseTargetKey(StrEnum):
    CANTERA = "cantera"
    FIPY = "fipy"


class SelectiveDependenceResponsePhase(StrEnum):
    CANARY = "CANARY"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


class SelectiveDependenceResponseExchangeExpectation(StrEnum):
    ACTIVE = "ACTIVE"
    INVARIANT = "INVARIANT"
    SUPPORT_BOUNDARY = "SUPPORT_BOUNDARY"


class SelectiveDependenceResponseDisposition(StrEnum):
    ACTION_AVAILABLE = "ACTION_AVAILABLE"
    HOLD_ONLY = "HOLD_ONLY"
    NONATTEMPT = "NONATTEMPT"
    UNEVALUABLE = "UNEVALUABLE"


class SelectiveDependenceResponseAxisName(StrEnum):
    CONSTRUCT = "CONSTRUCT"
    MEASUREMENT_ACTION_CHAIN = "MEASUREMENT_ACTION_CHAIN"
    LOCAL_RESPONSE_LAW = "LOCAL_RESPONSE_LAW"
    SELECTIVE_DEPENDENCE = "SELECTIVE_DEPENDENCE"
    SUPPORT_BOUNDARY = "SUPPORT_BOUNDARY"
    RECEIVER_ADMISSION = "RECEIVER_ADMISSION"
    HOLD_VIABILITY = "HOLD_VIABILITY"
    PREDICTIVE_DISTINCTIVENESS = "PREDICTIVE_DISTINCTIVENESS"


class SelectiveDependenceResponseAxisState(StrEnum):
    DISTINGUISHED = "DISTINGUISHED"
    INVALID = "INVALID"
    MIXED = "MIXED"
    NOT_DISTINGUISHED = "NOT_DISTINGUISHED"
    NOT_VIABLE = "NOT_VIABLE"
    OPPOSED = "OPPOSED"
    PARTIAL = "PARTIAL"
    QUALIFIED = "QUALIFIED"
    SUPPORTED = "SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"
    VALID = "VALID"
    VIABLE = "VIABLE"


class SelectiveDependenceResponseRelationState(StrEnum):
    MIXED = "MIXED"
    NOT_DISTINGUISHED = "NOT_DISTINGUISHED"
    OPPOSED = "OPPOSED"
    SUPPORTED = "SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


class SelectiveDependenceResponseCaseState(StrEnum):
    HIGH = "HIGH"
    LOW = "LOW"
    NEUTRAL = "NEUTRAL"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    UNEVALUABLE = "UNEVALUABLE"


class SelectiveDependenceResponseStopCode(StrEnum):
    ACTION_CHAIN_UNRESOLVED = "ACTION_CHAIN_UNRESOLVED"
    COMPARATOR_IDENTICAL_BY_CONSTRUCTION = "COMPARATOR_IDENTICAL_BY_CONSTRUCTION"
    CONSTRUCT_INVALID = "CONSTRUCT_INVALID"
    CONSTRUCT_REVIEW_INDEPENDENCE_UNRESOLVED = "CONSTRUCT_REVIEW_INDEPENDENCE_UNRESOLVED"
    EVALUATION_REVEAL_BARRIER_VIOLATED = "EVALUATION_REVEAL_BARRIER_VIOLATED"
    LOCAL_LAW_UNQUALIFIED = "LOCAL_LAW_UNQUALIFIED"
    MAPPING_UNRESOLVED = "MAPPING_UNRESOLVED"
    NO_PREDICTIVELY_DISTINGUISHING_CHALLENGE = "NO_PREDICTIVELY_DISTINGUISHING_CHALLENGE"
    POWER_OR_RESOLUTION_UNATTAINABLE = "POWER_OR_RESOLUTION_UNATTAINABLE"
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    UNEVALUABLE_COMPLETE_UNIT_INFERENCE = "UNEVALUABLE_COMPLETE_UNIT_INFERENCE"


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseMethodQuestionFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-method-question-freeze'

    freeze_id: str
    primary_relation_id: str
    relation_statement: str
    component_ids: tuple[str, ...]
    required_role_ids: tuple[str, ...]
    primary_question_ids: tuple[str, ...]
    decisive_falsifier_ids: tuple[str, ...]
    excluded_scope_ids: tuple[str, ...]
    outcome_visible_predecessor_ids: tuple[str, ...]
    maximum_claim: str
    protected_target_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        validate_stable_id(self.primary_relation_id, field_name="primary_relation_id")
        validate_nonempty(self.relation_statement, field_name="relation_statement")
        validate_nonempty(self.maximum_claim, field_name="maximum_claim")
        for name in (
            "component_ids",
            "required_role_ids",
            "primary_question_ids",
            "decisive_falsifier_ids",
            "excluded_scope_ids",
            "outcome_visible_predecessor_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        if self.primary_relation_id != SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID:
            raise ValueError("selective dependence response primary relation differs")
        if self.component_ids != SELECTIVE_DEPENDENCE_RESPONSE_COMPONENT_IDS:
            raise ValueError("selective dependence response component roster differs")
        if self.required_role_ids != SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS:
            raise ValueError("selective dependence response role roster differs")
        if self.protected_target_outcome_access_count:
            raise ValueError("method freeze cannot access selective dependence response outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("method freeze must be outcome blind")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseInspectedPilotInventory(CanonicalRecord):
    """Current role inventory exported from the original inspected exposures."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/selective-dependence-response/inspected-pilot-inventory"

    inventory_id: str
    selected_target_ids: tuple[str, ...]
    inspected_pilot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.inventory_id, field_name="inventory_id")
        require_sorted_unique_strings(self.inspected_pilot_ids, field_name="inspected_pilot_ids", allow_empty=False)
        if self.selected_target_ids != (
            "target.cantera-selective-dependence-response", "target.fipy-selective-dependence-response"
        ) or len(self.inspected_pilot_ids) != 7:
            raise ValueError("inspected pilot inventory requires the exact two targets and seven distinct original exposure mappings")
        if any(re.search(r"(?:^|[._-])v[0-9]+(?:$|[._-])", value) for value in self.inspected_pilot_ids):
            raise ValueError("inspected pilot inventory requires current descriptive roles and a separately authenticated original mapping")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseContaminationLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-contamination-ledger'

    ledger_id: str
    selected_target_ids: tuple[str, ...]
    visible_predecessor_ids: tuple[str, ...]
    visible_target_pilot_ids: tuple[str, ...]
    inspected_pilot_inventory: SelectiveDependenceResponseInspectedPilotInventory
    inspected_pilot_export: IndependentSourceExport
    prohibited_reuse_ids: tuple[str, ...]
    selection_claim: str
    independent_investigator_claim_permitted: bool
    external_task_owner_claim_permitted: bool
    target_pilot_outcome_access_occurred: bool
    claim_bearing_design_response_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        if (
            type(self.inspected_pilot_inventory) is not SelectiveDependenceResponseInspectedPilotInventory
            or type(self.inspected_pilot_export) is not IndependentSourceExport
            or self.visible_target_pilot_ids != self.inspected_pilot_inventory.inspected_pilot_ids
            or self.selected_target_ids != self.inspected_pilot_inventory.selected_target_ids
        ):
            raise ValueError("contamination ledger requires its exact current inspected pilot inventory and separately verified original export custody")
        self.inspected_pilot_export.validate_source(
            ObjectIdentity.from_record(self.inspected_pilot_inventory.inventory_id, self.inspected_pilot_inventory),
            self.visible_target_pilot_ids,
        )
        validate_nonempty(self.selection_claim, field_name="selection_claim")
        for name in (
            "selected_target_ids",
            "visible_predecessor_ids",
            "visible_target_pilot_ids",
            "prohibited_reuse_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        if self.selected_target_ids != (
            "target.cantera-selective-dependence-response",
            "target.fipy-selective-dependence-response",
        ):
            raise ValueError("selective dependence response target nomination differs")
        if self.independent_investigator_claim_permitted:
            raise ValueError("outcome-visible nomination is not investigator independent")
        if self.external_task_owner_claim_permitted:
            raise ValueError("repository-authored tasks are not external-owner replications")
        if not self.target_pilot_outcome_access_occurred:
            raise ValueError("contamination ledger must disclose inspected selective-response pilots")
        if self.claim_bearing_design_response_count:
            raise ValueError("contamination ledger must precede final-design response")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("contamination ledger must remain selective-response-outcome blind")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetNativeDossier(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-native-dossier'

    dossier_id: str
    target_id: str
    source_family_id: str
    native_task_statement: str
    complete_unit_definition: str
    causal_cutoff_definition: str
    action_stage_definition: str
    receiver_definition: str
    failure_definition: str
    native_unit_ids: tuple[str, ...]
    native_clock_ids: tuple[str, ...]
    uses_metatheory_role_vocabulary: bool
    contains_target_forecast: bool
    claim_bearing_design_response_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("dossier_id", "target_id", "source_family_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "native_task_statement",
            "complete_unit_definition",
            "causal_cutoff_definition",
            "action_stage_definition",
            "receiver_definition",
            "failure_definition",
        ):
            validate_nonempty(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.native_unit_ids, field_name="native_unit_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.native_clock_ids, field_name="native_clock_ids", allow_empty=False
        )
        if self.uses_metatheory_role_vocabulary or self.contains_target_forecast:
            raise ValueError("native dossier cannot encode the selective-response role map or forecast")
        if self.claim_bearing_design_response_count:
            raise ValueError("native dossier must precede final-design response")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("native dossier must be outcome blind")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseFiniteRoleMap(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-finite-role-map'

    map_id: str
    target_id: str
    dossier: ObjectIdentity
    role_ids: tuple[str, ...]
    native_binding_ids: tuple[str, ...]
    semantic_loss_ids: tuple[str, ...]
    requested_accepted_applied_realized_distinct: bool
    receiver_direction_preserved: bool
    complete_unit_preserved: bool
    causal_cutoff_preserved: bool
    selected_before_development: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("map_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("role_ids", "native_binding_ids", "semantic_loss_ids"):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name == "semantic_loss_ids",
            )
        if self.role_ids != SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS:
            raise ValueError("finite role map lacks D/H/A/R/tau")
        if not all(
            (
                self.requested_accepted_applied_realized_distinct,
                self.receiver_direction_preserved,
                self.complete_unit_preserved,
                self.causal_cutoff_preserved,
                self.selected_before_development,
            )
        ):
            raise ValueError("finite role map loses a mandatory semantic invariant")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("finite role map must be outcome blind")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseFiniteRoleMapSelection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-finite-role-map-selection'

    selection_id: str
    target_id: str
    candidate_maps: tuple[ObjectIdentity, ...]
    selected_map: ObjectIdentity
    semantic_preservation_rule: str
    resolved: bool
    claim_bearing_design_response_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("selection_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.candidate_maps,
            attribute="object_id",
            field_name="candidate_maps",
        )
        if not 1 <= len(self.candidate_maps) <= 3:
            raise ValueError("finite role-map candidate count must lie inside [1, 3]")
        if self.selected_map not in self.candidate_maps or not self.resolved:
            raise ValueError("finite role-map selection is unresolved")
        validate_nonempty(
            self.semantic_preservation_rule,
            field_name="semantic_preservation_rule",
        )
        if self.claim_bearing_design_response_count:
            raise ValueError("finite role-map selection must precede final design response")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("finite role-map selection must remain outcome blind")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseConstructReviewAttestation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-construct-review-attestation'

    attestation_id: str
    target_id: str
    dossier: ObjectIdentity
    role_map: ObjectIdentity
    role_map_selection: ObjectIdentity
    reviewer_id: str
    forecast_author_id: str
    reviewer_accountable: bool
    reviewer_distinct_from_forecast_author: bool
    hostile_case_ids: tuple[str, ...]
    all_hostile_cases_passed: bool
    review_statement: str
    claim_bearing_design_response_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "attestation_id",
            "target_id",
            "reviewer_id",
            "forecast_author_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.review_statement, field_name="review_statement")
        if self.role_map_selection.object_schema != SelectiveDependenceResponseFiniteRoleMapSelection.SCHEMA:
            raise ValueError("construct review lacks the finite role-map selection")
        require_sorted_unique_strings(
            self.hostile_case_ids, field_name="hostile_case_ids", allow_empty=False
        )
        if self.reviewer_id == self.forecast_author_id:
            raise ValueError("construct reviewer and forecast author must differ")
        if not all(
            (
                self.reviewer_accountable,
                self.reviewer_distinct_from_forecast_author,
                self.all_hostile_cases_passed,
            )
        ):
            raise ValueError("construct review is not claim-bearing")
        if self.claim_bearing_design_response_count:
            raise ValueError("construct review must precede final-design response")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("construct review must be outcome blind")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponsePreparationDistributionFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-preparation-distribution-freeze'

    freeze_id: str
    target_id: str
    variable_ids: tuple[str, ...]
    distribution_statement: str
    canary_unit_ids: tuple[str, ...]
    development_unit_ids: tuple[str, ...]
    evaluation_unit_ids: tuple[str, ...]
    reserve_unit_ids: tuple[str, ...]
    all_unit_ids_sha256: str
    nested_conditions_count_as_units: bool
    materially_varying_preparations: bool
    seed_family: str
    development_response_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("freeze_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.distribution_statement, field_name="distribution_statement")
        require_sorted_unique_strings(
            self.variable_ids, field_name="variable_ids", allow_empty=False
        )
        rosters = (
            self.canary_unit_ids,
            self.development_unit_ids,
            self.evaluation_unit_ids,
            self.reserve_unit_ids,
        )
        for index, roster in enumerate(rosters):
            require_sorted_unique_strings(
                roster,
                field_name=f"unit_roster_{index}",
                allow_empty=index == 3,
            )
        if any(
            set(left) & set(right) for i, left in enumerate(rosters) for right in rosters[i + 1 :]
        ):
            raise ValueError("preparation partitions overlap")
        all_ids = tuple(sorted(value for roster in rosters for value in roster))
        validate_sha256(self.all_unit_ids_sha256, field_name="all_unit_ids_sha256")
        if self.all_unit_ids_sha256 != digest_ids(all_ids):
            raise ValueError("complete-unit roster digest differs")
        if self.nested_conditions_count_as_units:
            raise ValueError("nested conditions cannot inflate replication")
        if not self.materially_varying_preparations:
            raise ValueError("roundoff jitter is not a preparation distribution")
        if self.seed_family != "explicit-full-seed-pcg64":
            raise ValueError("preparation seed family differs")
        if self.development_response_count:
            raise ValueError("preparation distribution must precede development")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("preparation distribution must be outcome blind")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseActionRealization(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-action-realization'

    realization_id: str
    action_id: str
    requested_value: Decimal
    requested_unit: str
    accepted_value: Decimal
    accepted_unit: str
    applied_value: Decimal
    applied_unit: str
    realized_value: Decimal
    realized_unit: str
    requested_clock: Decimal
    accepted_clock: Decimal
    applied_clock: Decimal
    realized_clock: Decimal
    acceptance_state: str

    def __post_init__(self) -> None:
        for name in ("realization_id", "action_id", "acceptance_state"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "requested_value",
            "accepted_value",
            "applied_value",
            "realized_value",
            "requested_clock",
            "accepted_clock",
            "applied_clock",
            "realized_clock",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        for name in ("requested_unit", "accepted_unit", "applied_unit", "realized_unit"):
            validate_nonempty(getattr(self, name), field_name=name)
        if not (
            self.requested_clock <= self.accepted_clock <= self.applied_clock <= self.realized_clock
        ):
            raise ValueError("action realization clocks are not causal")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseReceiverObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-receiver-observation'

    observation_id: str
    receiver_id: str
    value: Decimal
    native_unit: str
    valid: bool

    def __post_init__(self) -> None:
        for name in ("observation_id", "receiver_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_decimal(self.value, field_name="value")
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseConditionResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-condition-result'

    condition_id: str
    denominator_id: str
    history_id: str
    action_id: str
    horizon_id: str
    support_state: str
    action: SelectiveDependenceResponseActionRealization
    receivers: tuple[SelectiveDependenceResponseReceiverObservation, ...]
    gate_margins: tuple[NamedDecimal, ...]
    fibre_admitted: bool | None
    disposition: SelectiveDependenceResponseDisposition
    stopped: bool
    stop_code: str | None

    def __post_init__(self) -> None:
        for name in (
            "condition_id",
            "denominator_id",
            "history_id",
            "action_id",
            "horizon_id",
            "support_state",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(self.receivers, attribute="receiver_id", field_name="receivers")
        require_sorted_unique_ids(
            self.gate_margins, attribute="value_id", field_name="gate_margins"
        )
        if self.action.action_id != self.action_id:
            raise ValueError("condition and action realization differ")
        if self.stopped != (self.stop_code is not None):
            raise ValueError("condition stop state and code differ")
        if self.stopped != (self.fibre_admitted is None):
            raise ValueError("condition fibre evaluability and stop state differ")
        if self.stopped and self.disposition is not SelectiveDependenceResponseDisposition.UNEVALUABLE:
            raise ValueError("stopped condition must be unevaluable")
        if self.support_state == "outside-support" and self.fibre_admitted:
            raise ValueError("an outside-support action fibre cannot be admitted")
        if not self.stopped:
            expected_admitted = (
                self.support_state != "outside-support"
                and bool(self.gate_margins)
                and all(value.value >= 0 for value in self.gate_margins)
            )
            if self.fibre_admitted != expected_admitted:
                raise ValueError("action-fibre admission is not the exact gate intersection")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseContextDecision(CanonicalRecord):
    """Categorical active-set, measured-hold and policy result for one context.

    This is deliberately separate from a condition's exact action-fibre result:
    a viable hold can coexist with one or more admitted active actions, while the
    policy branch is still ``ACTION_AVAILABLE``.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-context-decision'

    decision_id: str
    denominator_id: str
    history_id: str
    horizon_id: str
    admitted_active_action_ids: tuple[str, ...]
    active_fibres_complete: bool
    hold_action_id: str
    hold_viable: bool | None
    disposition: SelectiveDependenceResponseDisposition

    def __post_init__(self) -> None:
        for name in (
            "decision_id",
            "denominator_id",
            "history_id",
            "horizon_id",
            "hold_action_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.admitted_active_action_ids,
            field_name="admitted_active_action_ids",
        )
        if self.hold_action_id in self.admitted_active_action_ids:
            raise ValueError("hold cannot be counted as an active action")
        expected = (
            SelectiveDependenceResponseDisposition.ACTION_AVAILABLE
            if self.admitted_active_action_ids
            else (
                SelectiveDependenceResponseDisposition.UNEVALUABLE
                if not self.active_fibres_complete or self.hold_viable is None
                else (
                    SelectiveDependenceResponseDisposition.HOLD_ONLY
                    if self.hold_viable
                    else SelectiveDependenceResponseDisposition.NONATTEMPT
                )
            )
        )
        if self.disposition is not expected:
            raise ValueError("context policy is not derived from active fibres then hold")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseCompleteUnitResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-complete-unit-result'

    result_id: str
    target_id: str
    complete_unit_id: str
    phase: SelectiveDependenceResponsePhase
    preparation_values: tuple[NamedDecimal, ...]
    conditions: tuple[SelectiveDependenceResponseConditionResult, ...]
    context_decisions: tuple[SelectiveDependenceResponseContextDecision, ...]
    expected_condition_ids_sha256: str
    source_version: str
    solver_id: str
    nested_conditions_count_as_units: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("result_id", "target_id", "complete_unit_id", "solver_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.source_version, field_name="source_version")
        require_sorted_unique_ids(
            self.preparation_values,
            attribute="value_id",
            field_name="preparation_values",
        )
        require_sorted_unique_ids(
            self.conditions, attribute="condition_id", field_name="conditions"
        )
        require_sorted_unique_ids(
            self.context_decisions,
            attribute="decision_id",
            field_name="context_decisions",
        )
        validate_sha256(
            self.expected_condition_ids_sha256,
            field_name="expected_condition_ids_sha256",
        )
        condition_ids = tuple(value.condition_id for value in self.conditions)
        if self.expected_condition_ids_sha256 != digest_ids(condition_ids):
            raise ValueError("complete-unit condition roster differs")
        condition_contexts = {
            (value.denominator_id, value.history_id, value.horizon_id) for value in self.conditions
        }
        decision_contexts = {
            (value.denominator_id, value.history_id, value.horizon_id)
            for value in self.context_decisions
        }
        if condition_contexts != decision_contexts:
            raise ValueError("complete-unit context decisions do not cover conditions")
        for decision in self.context_decisions:
            branches = tuple(
                value
                for value in self.conditions
                if (
                    value.denominator_id,
                    value.history_id,
                    value.horizon_id,
                )
                == (
                    decision.denominator_id,
                    decision.history_id,
                    decision.horizon_id,
                )
            )
            hold = tuple(value for value in branches if value.action_id == decision.hold_action_id)
            if len(hold) != 1 or hold[0].fibre_admitted is not decision.hold_viable:
                raise ValueError("context hold viability differs from its native fibre")
            admitted = tuple(
                sorted(
                    value.action_id
                    for value in branches
                    if value.action_id != decision.hold_action_id
                    and value.support_state != "outside-support"
                    and value.fibre_admitted
                )
            )
            if admitted != decision.admitted_active_action_ids:
                raise ValueError("context active set differs from exact action fibres")
            expected_complete = all(
                value.fibre_admitted is not None
                for value in branches
                if value.action_id != decision.hold_action_id
                and value.support_state != "outside-support"
            )
            if expected_complete != decision.active_fibres_complete:
                raise ValueError("context active-fibre completeness differs")
        if self.nested_conditions_count_as_units:
            raise ValueError("conditions cannot become independent units")
        expected_access = {
            SelectiveDependenceResponsePhase.CANARY: OutcomeAccess.DEVELOPMENT_VISIBLE,
            SelectiveDependenceResponsePhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            SelectiveDependenceResponsePhase.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.phase]
        if self.outcome_access is not expected_access:
            raise ValueError("complete-unit phase and outcome access differ")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetPanel(CanonicalRecord):
    """Exact full fan-in over complete independent preparation units."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-panel'

    panel_id: str
    target_id: str
    phase: SelectiveDependenceResponsePhase
    complete_units: tuple[SelectiveDependenceResponseCompleteUnitResult, ...]
    expected_complete_unit_ids: tuple[str, ...]
    expected_complete_unit_ids_sha256: str
    nested_conditions_count_as_units: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("panel_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.complete_units,
            attribute="complete_unit_id",
            field_name="complete_units",
        )
        require_sorted_unique_strings(
            self.expected_complete_unit_ids,
            field_name="expected_complete_unit_ids",
            allow_empty=False,
        )
        validate_sha256(
            self.expected_complete_unit_ids_sha256,
            field_name="expected_complete_unit_ids_sha256",
        )
        if self.expected_complete_unit_ids_sha256 != digest_ids(self.expected_complete_unit_ids):
            raise ValueError("target-panel roster digest differs")
        observed = tuple(value.complete_unit_id for value in self.complete_units)
        if observed != self.expected_complete_unit_ids:
            raise ValueError("target panel does not have exact full fan-in")
        if any(
            value.target_id != self.target_id or value.phase is not self.phase
            for value in self.complete_units
        ):
            raise ValueError("target panel crosses target or phase")
        if self.nested_conditions_count_as_units:
            raise ValueError("target panel cannot inflate nested conditions")
        expected_access = {
            SelectiveDependenceResponsePhase.CANARY: OutcomeAccess.DEVELOPMENT_VISIBLE,
            SelectiveDependenceResponsePhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            SelectiveDependenceResponsePhase.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.phase]
        if self.outcome_access is not expected_access:
            raise ValueError("target-panel phase and visibility differ")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseExchangeForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-exchange-forecast'

    exchange_id: str
    target_id: str
    expectation: SelectiveDependenceResponseExchangeExpectation
    coordinate_role_id: str
    left_cell_id: str
    right_cell_id: str
    estimand_id: str
    equivalence_margin: Decimal
    minimum_active_difference: Decimal
    native_unit: str

    def __post_init__(self) -> None:
        for name in (
            "exchange_id",
            "target_id",
            "left_cell_id",
            "right_cell_id",
            "estimand_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.coordinate_role_id not in SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS:
            raise ValueError("exchange coordinate is outside D/H/A/R/tau")
        validate_decimal(
            self.equivalence_margin,
            field_name="equivalence_margin",
            minimum=Decimal(0),
        )
        validate_decimal(
            self.minimum_active_difference,
            field_name="minimum_active_difference",
            minimum=Decimal(0),
        )
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.left_cell_id == self.right_cell_id:
            raise ValueError("exchange endpoints must differ")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseCellForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-cell-forecast'

    cell_id: str
    denominator_id: str
    history_id: str
    action_id: str
    receiver_id: str
    horizon_id: str
    expected_state: SelectiveDependenceResponseCaseState
    expected_fibre_admitted: bool | None
    expected_disposition: SelectiveDependenceResponseDisposition

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "denominator_id",
            "history_id",
            "action_id",
            "receiver_id",
            "horizon_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if (self.expected_fibre_admitted is None) != (
            self.expected_disposition is SelectiveDependenceResponseDisposition.UNEVALUABLE
        ):
            raise ValueError("forecast fibre and disposition evaluability differ")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseSelectiveLawForecast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-selective-law-forecast'

    forecast_id: str
    target_id: str
    method_question: ObjectIdentity
    contamination_ledger: ObjectIdentity
    construct_review: ObjectIdentity
    selected_role_map: ObjectIdentity
    role_map_selection: ObjectIdentity
    preparation_freeze: ObjectIdentity
    analysis_freeze: ObjectIdentity
    design: ObjectIdentity
    development_result: ObjectIdentity
    denominator_selection: ObjectIdentity
    target_relation_id: str
    exchange_familywise_alpha: Decimal
    exchange_interval_method: str
    exchange_forecasts: tuple[SelectiveDependenceResponseExchangeForecast, ...]
    cell_forecasts: tuple[SelectiveDependenceResponseCellForecast, ...]
    context_forecasts: tuple[SelectiveDependenceResponseContextDecision, ...]
    comparator_ids: tuple[str, ...]
    evaluation_unit_ids_sha256: str
    predictively_distinguishing: bool
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("forecast_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.target_relation_id != SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID:
            raise ValueError("forecast target relation differs")
        validate_decimal(
            self.exchange_familywise_alpha,
            field_name="exchange_familywise_alpha",
            minimum=Decimal(0),
        )
        if not Decimal(0) < self.exchange_familywise_alpha < Decimal(1):
            raise ValueError("exchange familywise alpha must lie inside (0, 1)")
        validate_nonempty(self.exchange_interval_method, field_name="exchange_interval_method")
        require_sorted_unique_ids(
            self.exchange_forecasts,
            attribute="exchange_id",
            field_name="exchange_forecasts",
        )
        require_sorted_unique_ids(
            self.cell_forecasts, attribute="cell_id", field_name="cell_forecasts"
        )
        require_sorted_unique_ids(
            self.context_forecasts,
            attribute="decision_id",
            field_name="context_forecasts",
        )
        if not self.context_forecasts:
            raise ValueError("forecast lacks categorical context decisions")
        require_sorted_unique_strings(
            self.comparator_ids, field_name="comparator_ids", allow_empty=False
        )
        validate_sha256(
            self.evaluation_unit_ids_sha256,
            field_name="evaluation_unit_ids_sha256",
        )
        expectations = {value.expectation for value in self.exchange_forecasts}
        if not {
            SelectiveDependenceResponseExchangeExpectation.ACTIVE,
            SelectiveDependenceResponseExchangeExpectation.INVARIANT,
            SelectiveDependenceResponseExchangeExpectation.SUPPORT_BOUNDARY,
        }.issubset(expectations):
            raise ValueError("forecast lacks active, invariant or support exchange")
        if not self.predictively_distinguishing:
            raise ValueError("non-distinguishing forecast cannot parent evaluation")
        if self.evaluation_outcome_count:
            raise ValueError("forecast cannot access evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("forecast must remain development visible")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseAxisAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-axis-adjudication'

    adjudication_id: str
    axis: SelectiveDependenceResponseAxisName
    state: SelectiveDependenceResponseAxisState
    decisive_case_ids: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_strings(self.decisive_case_ids, field_name="decisive_case_ids")
        validate_nonempty(self.reason, field_name="reason")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-adjudication'

    adjudication_id: str
    target_id: str
    forecast: ObjectIdentity
    evaluation_unit_ids_sha256: str
    issued_unit_count: int
    source_valid_unit_count: int
    evaluable_unit_count: int
    stopped_unit_count: int
    missing_unit_count: int
    axes: tuple[SelectiveDependenceResponseAxisAdjudication, ...]
    exact_counterexample_ids: tuple[str, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    maximum_claim: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("adjudication_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(
            self.evaluation_unit_ids_sha256,
            field_name="evaluation_unit_ids_sha256",
        )
        for name in (
            "issued_unit_count",
            "source_valid_unit_count",
            "evaluable_unit_count",
            "stopped_unit_count",
            "missing_unit_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.issued_unit_count != (
            self.evaluable_unit_count + self.stopped_unit_count + self.missing_unit_count
        ):
            raise ValueError("target unit accounting does not close")
        if self.source_valid_unit_count < self.evaluable_unit_count:
            raise ValueError("evaluable units cannot exceed source-valid units")
        require_sorted_unique_ids(self.axes, attribute="adjudication_id", field_name="axes")
        observed_axes = tuple(sorted(value.axis for value in self.axes))
        if observed_axes != tuple(sorted(SelectiveDependenceResponseAxisName)):
            raise ValueError("target adjudication lacks an axis")
        require_sorted_unique_strings(
            self.exact_counterexample_ids,
            field_name="exact_counterexample_ids",
        )
        validate_nonempty(self.maximum_claim, field_name="maximum_claim")
        if self.maximum_evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("selective dependence response target ceiling must remain local law")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("target adjudication must retain revealed visibility")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetHandoff(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-handoff'

    handoff_id: str
    target_id: str
    source_family_id: str
    solver_family_id: str
    primary_relation_id: str
    component_states: tuple[SelectiveDependenceResponseAxisAdjudication, ...]
    measurement_action_state: SelectiveDependenceResponseAxisState
    hold_state: SelectiveDependenceResponseAxisState
    distinctiveness_state: SelectiveDependenceResponseAxisState
    forecast_policy_dispositions: tuple[SelectiveDependenceResponseDisposition, ...]
    observed_policy_dispositions: tuple[SelectiveDependenceResponseDisposition, ...]
    construct_valid: bool
    target_eligible: bool
    ineligibility_reason_codes: tuple[str, ...]
    exact_counterexample_ids: tuple[str, ...]
    native_numeric_value_count: int
    target_adjudication: ObjectIdentity
    maximum_claim: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("handoff_id", "target_id", "source_family_id", "solver_family_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.primary_relation_id != SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID:
            raise ValueError("handoff relation differs")
        require_sorted_unique_ids(
            self.component_states,
            attribute="adjudication_id",
            field_name="component_states",
        )
        require_sorted_unique_strings(
            self.exact_counterexample_ids,
            field_name="exact_counterexample_ids",
        )
        require_sorted_unique_strings(
            self.ineligibility_reason_codes,
            field_name="ineligibility_reason_codes",
        )
        for name in ("forecast_policy_dispositions", "observed_policy_dispositions"):
            values = getattr(self, name)
            if not values or tuple(sorted(set(values), key=lambda value: value.value)) != values:
                raise ValueError(f"handoff {name} must be nonempty, sorted and unique")
        expected_eligible = (
            self.construct_valid and self.measurement_action_state is SelectiveDependenceResponseAxisState.QUALIFIED
        )
        if self.target_eligible != expected_eligible:
            raise ValueError("target handoff eligibility is not axis-derived")
        if self.target_eligible == bool(self.ineligibility_reason_codes):
            raise ValueError("target handoff eligibility reasons are inconsistent")
        if self.native_numeric_value_count:
            raise ValueError("cross-target handoff cannot carry native numeric values")
        validate_nonempty(self.maximum_claim, field_name="maximum_claim")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("target handoff must retain revealed visibility")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseCrossTargetAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-cross-target-adjudication'

    adjudication_id: str
    primary_relation_id: str
    eligibility: ObjectIdentity
    target_handoffs: tuple[ObjectIdentity, ...]
    target_ids: tuple[str, ...]
    relation_state: SelectiveDependenceResponseRelationState
    decisive_target_ids: tuple[str, ...]
    pooled_native_numeric_value_count: int
    maximum_claim: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        if self.primary_relation_id != SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID:
            raise ValueError("cross-target relation differs")
        require_sorted_unique_ids(
            self.target_handoffs,
            attribute="object_id",
            field_name="target_handoffs",
        )
        require_sorted_unique_strings(self.target_ids, field_name="target_ids", allow_empty=False)
        require_sorted_unique_strings(self.decisive_target_ids, field_name="decisive_target_ids")
        if len(self.target_ids) != 2 or len(self.target_handoffs) != 2:
            raise ValueError("cross-target adjudication requires two exact handoffs")
        if self.pooled_native_numeric_value_count:
            raise ValueError("cross-target adjudication cannot pool native values")
        validate_nonempty(self.maximum_claim, field_name="maximum_claim")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("cross-target adjudication must remain outcome visible")


__all__ = [
    "SELECTIVE_DEPENDENCE_RESPONSE_COMPONENT_IDS",
    "SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID",
    'SelectiveDependenceResponseActionRealization',
    'SelectiveDependenceResponseAxisAdjudication',
    'SelectiveDependenceResponseAxisName',
    'SelectiveDependenceResponseAxisState',
    'SelectiveDependenceResponseCaseState',
    'SelectiveDependenceResponseCellForecast',
    'SelectiveDependenceResponseCompleteUnitResult',
    'SelectiveDependenceResponseConditionResult',
    'SelectiveDependenceResponseContextDecision',
    'SelectiveDependenceResponseConstructReviewAttestation',
    'SelectiveDependenceResponseContaminationLedger',
    'SelectiveDependenceResponseCrossTargetAdjudication',
    'SelectiveDependenceResponseDisposition',
    'SelectiveDependenceResponseExchangeExpectation',
    'SelectiveDependenceResponseExchangeForecast',
    'SelectiveDependenceResponseFiniteRoleMap',
    'SelectiveDependenceResponseFiniteRoleMapSelection',
    'SelectiveDependenceResponseMethodQuestionFreeze',
    'SelectiveDependenceResponsePhase',
    'SelectiveDependenceResponsePreparationDistributionFreeze',
    'SelectiveDependenceResponseReceiverObservation',
    'SelectiveDependenceResponseRelationState',
    'SelectiveDependenceResponseSelectiveLawForecast',
    'SelectiveDependenceResponseStopCode',
    'SelectiveDependenceResponseTargetAdjudication',
    'SelectiveDependenceResponseTargetHandoff',
    'SelectiveDependenceResponseTargetKey',
    'SelectiveDependenceResponseTargetNativeDossier',
    'SelectiveDependenceResponseTargetPanel',
    "digest_ids",
]
