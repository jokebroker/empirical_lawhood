"Outcome-blind templates for materializing the admission controller route.\n\nThe records in this module freeze topology and decision rules before scientific\noutcomes exist.  They deliberately carry expected IDs and schemas, rather than\nfingerprints for a future atlas, raw admission corpus, calibration, selected anchor\nroster, or observer config.  One deterministic binder after admission attaches those\nrecords and constructs the existing :class:`AdmissionControllerSynthesisPlan`; the\nexisting evidence-derived programme author remains the sole admission-controller author.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.admission import (
    AdmissionGateKind,
    GateStatus,
    ReachabilityStatus,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ExecutableReference, NamedDecimal
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
from empirical_lawhood.kernel.status import AdmissionStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.controller_study import CONTROLLER_ROLES, AdmissionActionCandidate, CandidatePriorityOrientation, AdmissionControllerStudy, AdmissionControllerSynthesisPlan, AdmissionActionCandidateChart, ImplementationBinding, ImplementationRole, AdmissionMeasuredHoldFibre, OnlineSupportMonitorSpec, ControllerActionBinding, ReidentificationTriggerSpec
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionSpec, ControlledMapReachabilitySpec, GatePredicateKind, GatePredicateSpec, ControlledMapAdmissionReceiptCorpus, ReceiptAdmissionReachabilityReferenceKind, ReceiptAdmissionUtilityDirection, ReceiptAdmissionUtilityStatus, PreservationCompatibilityRule, derive_receipt_admission_comparison, derive_controlled_map_reachability_comparison, ControlledMapReachabilityComparison
from empirical_lawhood.planning.geometry import AdmissionComparison
from empirical_lawhood.planning.local_support_compatibility import PreparedDenominatorLocalSupportCompatibilityDisposition, PreparedDenominatorLocalSupportCompatibility
from empirical_lawhood.planning.native_hold_decision_cell_calibration import NativeHoldCalibrationDisposition, NativeHoldCalibrationSelectedRoster, NativeHoldDecisionCellCalibrationReceipt


MAX_ADMISSION_CONTROLLER_PROGRAMME_TEMPLATE_BYTES = 16 * 1024 * 1024


_ROLE_PORT_SCHEMAS: dict[
    ImplementationRole,
    tuple[tuple[str, ...], tuple[str, ...]],
] = {
    ImplementationRole.ADMISSION_DERIVER: (
        (ReceiptAdmissionSpec.SCHEMA,),
        (AdmissionComparison.SCHEMA,),
    ),
    ImplementationRole.REACHABILITY_DERIVER: (
        (ControlledMapReachabilitySpec.SCHEMA,),
        (ControlledMapReachabilityComparison.SCHEMA,),
    ),
    ImplementationRole.SYNTHESIZER: (
        (AdmissionControllerStudy.SCHEMA,),
        ('empirical-lawhood/runtime/compiled-admission-controller-study',),
    ),
    ImplementationRole.OBSERVER: (
        ('empirical-lawhood/runtime/runtime-observation',),
        ('empirical-lawhood/runtime/observer-evaluation',),
    ),
    ImplementationRole.ONLINE_GATE_EVALUATOR: (
        (AdmissionControllerStudy.SCHEMA,),
        ('empirical-lawhood/runtime/admission-live-gate-evaluation',),
    ),
    ImplementationRole.DELIVERY: (
        (OccurrenceActionWord.SCHEMA,),
        ('empirical-lawhood/runtime/exact-action-delivery-trace',),
    ),
}


def _bytewise_sorted(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(values, key=lambda value: value.encode("utf-8")))


@dataclass(frozen=True, slots=True)
class ControllerImplementationBindingTemplate(CanonicalRecord):
    """One code/config binding whose config digest is materialized later."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controller-implementation-binding-template'

    binding_id: str
    role: ImplementationRole
    reference: ExecutableReference
    expected_config_id: str
    expected_config_schema: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.expected_config_id, field_name="expected_config_id")
        validate_schema(self.expected_config_schema)
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )
        if self.role not in CONTROLLER_ROLES:
            raise ValueError("controller binding template uses a role outside its admission controller contract")
        expected_inputs, expected_outputs = _ROLE_PORT_SCHEMAS[self.role]
        if (
            not self.reference.deterministic
            or self.reference.input_schema not in expected_inputs
            or self.reference.output_schema not in expected_outputs
        ):
            raise ValueError("controller binding template changes its admission controller port contract")

    def materialize(self, config: ObjectIdentity) -> ImplementationBinding:
        if (
            config.object_id != self.expected_config_id
            or config.object_schema != self.expected_config_schema
        ):
            raise ValueError("controller implementation config substitutes a frozen identity")
        return ImplementationBinding(
            binding_id=self.binding_id,
            role=self.role,
            reference=self.reference,
            config_sha256=config.object_fingerprint,
            implementation_sha256=self.implementation_sha256,
        )


class AdmissionControllerCellSelectionRule(StrEnum):
    UNSIGNED_UTF8_BYTES_FIRST = "UNSIGNED_UTF8_BYTES_FIRST"


class AdmissionControllerUtilityRole(StrEnum):
    ACTIVE = "ACTIVE"
    SIBLING_NATIVE_HOLD = "SIBLING_NATIVE_HOLD"


class AdmissionControllerTargetComparison(StrEnum):
    STRICT_GREATER_THAN = "STRICT_GREATER_THAN"


class ControllerSupportMonitorSource(StrEnum):
    ADMITTED_LOCAL_SUPPORT = "ADMITTED_LOCAL_SUPPORT"
    SELECTED_ANCHOR_FINGERPRINTS = "SELECTED_ANCHOR_FINGERPRINTS"


class ControllerReidentificationCategory(StrEnum):
    SOURCE_RUNTIME_MEMBER_DRIFT = "SOURCE_RUNTIME_MEMBER_DRIFT"
    PREACTION_CLOCK_PREFIX_OR_FIELD_INVALID = "PREACTION_CLOCK_PREFIX_OR_FIELD_INVALID"
    SCIENTIFIC_BINDING_FINGERPRINT_MISMATCH = "SCIENTIFIC_BINDING_FINGERPRINT_MISMATCH"
    AUTHORITY_REVOCATION = "AUTHORITY_REVOCATION"


@dataclass(frozen=True, slots=True)
class AdmissionControllerCellPairTemplate(CanonicalRecord):
    "One predeclared active/sibling-HOLD admission pair on one local support."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-controller-cell-pair-template'

    tier_id: str
    support_cell_id: str
    active_candidate_id: str
    active_decision_cell_id: str
    active_admission_candidate_cell_id: str
    sibling_hold_admission_candidate_cell_id: str

    def __post_init__(self) -> None:
        for name in (
            "tier_id",
            "support_cell_id",
            "active_candidate_id",
            "active_decision_cell_id",
            "active_admission_candidate_cell_id",
            "sibling_hold_admission_candidate_cell_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.active_admission_candidate_cell_id == self.sibling_hold_admission_candidate_cell_id:
            raise ValueError("active and sibling-HOLD admission cells must remain distinct")


@dataclass(frozen=True, slots=True)
class AdmissionSiblingHoldCellSelectorTemplate(CanonicalRecord):
    """Frozen complete-set selector; it cannot inspect favorable outcomes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-sibling-hold-cell-selector-template'

    selector_id: str
    eligible_candidate_cell_ids: tuple[str, ...]
    selected_candidate_cell_id: str
    selection_rule: AdmissionControllerCellSelectionRule
    requires_complete_active_and_hold_grid: bool
    outcome_dependent_selection: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.selector_id, field_name="selector_id")
        require_sorted_unique_strings(
            self.eligible_candidate_cell_ids,
            field_name="eligible_candidate_cell_ids",
            allow_empty=False,
        )
        validate_stable_id(
            self.selected_candidate_cell_id,
            field_name="selected_candidate_cell_id",
        )
        if (
            self.selection_rule is not AdmissionControllerCellSelectionRule.UNSIGNED_UTF8_BYTES_FIRST
            or self.eligible_candidate_cell_ids
            != _bytewise_sorted(self.eligible_candidate_cell_ids)
            or self.selected_candidate_cell_id != self.eligible_candidate_cell_ids[0]
            or not self.requires_complete_active_and_hold_grid
            or self.outcome_dependent_selection
        ):
            raise ValueError("sibling-HOLD selector is not the frozen bytewise rule")


@dataclass(frozen=True, slots=True)
class AdmissionControllerUtilityRule(CanonicalRecord):
    """Action-local utility identity and noninterchangeable native rule."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-controller-utility-rule'

    rule_id: str
    role: AdmissionControllerUtilityRole
    action_binding_id: str
    expected_utility_definition_id: str
    expected_utility_definition_schema: str
    direction: ReceiptAdmissionUtilityDirection
    minimum_utility: NamedDecimal
    derivation_rule: str

    def __post_init__(self) -> None:
        for name in (
            "rule_id",
            "action_binding_id",
            "expected_utility_definition_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_schema(self.expected_utility_definition_schema)
        validate_nonempty(self.derivation_rule, field_name="derivation_rule")
        if self.direction is not ReceiptAdmissionUtilityDirection.HIGHER_IS_BETTER:
            raise ValueError("current admission controller utility template requires HIGHER_IS_BETTER")
        if self.role is AdmissionControllerUtilityRole.SIBLING_NATIVE_HOLD and (
            self.minimum_utility.value != 0 or self.derivation_rule != "DECLARED_NATIVE_HOLD_NULL"
        ):
            raise ValueError("sibling-HOLD utility must remain an exact native-null rule")
        if self.role is AdmissionControllerUtilityRole.ACTIVE and (
            self.minimum_utility.value <= 0
            or self.derivation_rule != "PREDICTED_MINUS_MATCHED_HOLD_MINUS_UNCERTAINTY"
        ):
            raise ValueError("active utility must retain its matched-HOLD lower rule")


@dataclass(frozen=True, slots=True)
class AdmissionControllerTargetRule(CanonicalRecord):
    """Sole Boolean TARGET with strict active and native-HOLD semantics."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-controller-target-rule'

    rule_id: str
    predicate: GatePredicateSpec
    active_action_binding_id: str
    hold_action_binding_id: str
    active_materiality: NamedDecimal
    active_comparison: AdmissionControllerTargetComparison
    equality_passes: bool
    hold_null_tolerance: NamedDecimal
    hold_requires_complete_exact_delivery: bool
    offline_certificate_replay_only: bool
    retains_native_operands_and_margins: bool

    def __post_init__(self) -> None:
        for name in (
            "rule_id",
            "active_action_binding_id",
            "hold_action_binding_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.active_action_binding_id == self.hold_action_binding_id
            or self.predicate.gate_kind is not AdmissionGateKind.TARGET
            or self.predicate.predicate_kind is not GatePredicateKind.BOOLEAN_EQUALS
            or self.predicate.expected_boolean is not True
            or self.active_materiality.value <= 0
            or self.active_materiality.unit != self.hold_null_tolerance.unit
            or self.hold_null_tolerance.value < 0
            or self.active_comparison is not AdmissionControllerTargetComparison.STRICT_GREATER_THAN
            or self.equality_passes
            or not self.hold_requires_complete_exact_delivery
            or not self.offline_certificate_replay_only
            or not self.retains_native_operands_and_margins
        ):
            raise ValueError("admission TARGET rule changes strict action/HOLD Boolean semantics")


@dataclass(frozen=True, slots=True)
class OnlineScalarGateRequirement(CanonicalRecord):
    """One non-TARGET pre-action scalar input and exact inclusive threshold."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/online-scalar-gate-requirement'

    requirement_id: str
    gate_kind: AdmissionGateKind
    observation_quantity_id: str
    minimum: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.requirement_id, field_name="requirement_id")
        validate_stable_id(
            self.observation_quantity_id,
            field_name="observation_quantity_id",
        )
        if self.gate_kind is AdmissionGateKind.TARGET:
            raise ValueError("TARGET is Boolean and cannot enter the scalar gate roster")
        if self.minimum.value not in {Decimal(0), Decimal(1)} or self.minimum.unit != "1":
            raise ValueError("online scalar requirement must use an exact unit-1 zero/one floor")


@dataclass(frozen=True, slots=True)
class ControllerSupportMonitorTemplate(CanonicalRecord):
    "Monitor declared before measurement, whose five anchor fingerprints are attached later."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controller-support-monitor-template'

    monitor_id: str
    decision_cell_id: str
    source: ControllerSupportMonitorSource
    expected_support_ids: tuple[str, ...]
    anchor_slot_ids: tuple[str, ...]
    check_rule: str
    invalidation_reason_code: str

    def __post_init__(self) -> None:
        validate_stable_id(self.monitor_id, field_name="monitor_id")
        validate_stable_id(self.decision_cell_id, field_name="decision_cell_id")
        validate_stable_id(
            self.invalidation_reason_code,
            field_name="invalidation_reason_code",
        )
        validate_nonempty(self.check_rule, field_name="check_rule")
        require_sorted_unique_strings(
            self.expected_support_ids,
            field_name="expected_support_ids",
        )
        require_sorted_unique_strings(
            self.anchor_slot_ids,
            field_name="anchor_slot_ids",
        )
        if self.source is ControllerSupportMonitorSource.ADMITTED_LOCAL_SUPPORT:
            if len(self.expected_support_ids) != 1 or self.anchor_slot_ids:
                raise ValueError("active monitor must bind exactly one local support")
        elif self.expected_support_ids or len(self.anchor_slot_ids) != 5:
            raise ValueError("HOLD monitor must parameterize exactly five anchor slots")


@dataclass(frozen=True, slots=True)
class ControllerReidentificationTriggerTemplate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controller-reidentification-trigger-template'

    category: ControllerReidentificationCategory
    trigger: ReidentificationTriggerSpec


@dataclass(frozen=True, slots=True)
class AdmissionControllerStudyTemplate(CanonicalRecord):
    "Topology declared before measurement for one exact three-cell admission controller programme."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-controller-study-template'

    template_id: str
    expected_synthesis_id: str
    expected_study_request_id: str
    expected_study_id: str
    compiler_release_id: str
    expected_system_id: str
    expected_system_schema: str
    expected_atlas_id: str
    expected_atlas_schema: str
    expected_admission_corpus_id: str
    expected_admission_corpus_schema: str
    expected_admission_spec_id: str
    expected_admission_spec_schema: str
    expected_reachability_spec_id: str
    expected_reachability_spec_schema: str
    expected_model_set_id: str
    expected_model_set_schema: str
    expected_response_law_ids: tuple[str, ...]
    expected_measured_hold_fibre_id: str
    expected_measured_hold_fibre_schema: str
    expected_native_hold_calibration_receipt_id: str
    expected_native_hold_calibration_receipt_schema: str
    expected_selected_anchor_roster_id: str
    expected_selected_anchor_roster_schema: str
    expected_local_support_compatibility_receipt_id: str
    expected_local_support_compatibility_receipt_schema: str
    local_support_compatibility_evaluator: ObjectIdentity
    expected_observer_config_id: str
    expected_observer_config_schema: str
    expected_observer_template_binding_receipt_id: str
    expected_observer_template_binding_receipt_schema: str
    expected_online_gate_config_id: str
    expected_online_gate_config_schema: str
    prepared_denominator_id: str
    expected_candidate_version_id: str
    model_member_ids: tuple[str, ...]
    nominal_denominator_member_id: str
    active_action_binding_id: str
    active_action_word_id: str
    hold_action_binding_id: str
    hold_action_word_id: str
    candidate_chart_id: str
    cell_pairs: tuple[AdmissionControllerCellPairTemplate, ...]
    sibling_hold_selector: AdmissionSiblingHoldCellSelectorTemplate
    hold_decision_cell_id: str
    utility_rules: tuple[AdmissionControllerUtilityRule, ...]
    target_rule: AdmissionControllerTargetRule
    gate_predicates: tuple[GatePredicateSpec, ...]
    scalar_gate_requirements: tuple[OnlineScalarGateRequirement, ...]
    baseline_compatibility_rule: PreservationCompatibilityRule
    complete_baseline_evidence_required: bool
    preparation_quantity_ids: tuple[str, ...]
    runtime_state_quantity_id: str
    observation_quantity_ids: tuple[str, ...]
    support_monitor_templates: tuple[ControllerSupportMonitorTemplate, ...]
    reidentification_trigger_templates: tuple[
        ControllerReidentificationTriggerTemplate,
        ...,
    ]
    implementation_templates: tuple[ControllerImplementationBindingTemplate, ...]
    authority_policy_id: str
    worst_case_latency_seconds: Decimal
    deadline_seconds: Decimal
    template_binder_implementation_id: str
    template_binder_implementation_sha256: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        _validate_admission_controller_study_template(self)


def _validate_admission_controller_study_template(
    value: AdmissionControllerStudyTemplate,
) -> None:
    stable_ids = (
        "template_id",
        "expected_synthesis_id",
        'expected_study_request_id',
        'expected_study_id',
        "compiler_release_id",
        "expected_system_id",
        "expected_atlas_id",
        "expected_admission_corpus_id",
        "expected_admission_spec_id",
        "expected_reachability_spec_id",
        "expected_model_set_id",
        "expected_measured_hold_fibre_id",
        "expected_native_hold_calibration_receipt_id",
        "expected_selected_anchor_roster_id",
        "expected_local_support_compatibility_receipt_id",
        "expected_observer_config_id",
        "expected_observer_template_binding_receipt_id",
        "expected_online_gate_config_id",
        "prepared_denominator_id",
        "expected_candidate_version_id",
        "nominal_denominator_member_id",
        "active_action_binding_id",
        "active_action_word_id",
        "hold_action_binding_id",
        "hold_action_word_id",
        "candidate_chart_id",
        "hold_decision_cell_id",
        "runtime_state_quantity_id",
        "authority_policy_id",
        "template_binder_implementation_id",
    )
    for name in stable_ids:
        validate_stable_id(getattr(value, name), field_name=name)
    for schema in (
        value.expected_system_schema,
        value.expected_atlas_schema,
        value.expected_admission_corpus_schema,
        value.expected_admission_spec_schema,
        value.expected_reachability_spec_schema,
        value.expected_model_set_schema,
        value.expected_measured_hold_fibre_schema,
        value.expected_native_hold_calibration_receipt_schema,
        value.expected_selected_anchor_roster_schema,
        value.expected_local_support_compatibility_receipt_schema,
        value.expected_observer_config_schema,
        value.expected_observer_template_binding_receipt_schema,
        value.expected_online_gate_config_schema,
    ):
        validate_schema(schema)
    expected_schemas = (
        (value.expected_system_schema, SystemSpec.SCHEMA),
        (value.expected_atlas_schema, 'empirical-lawhood/kernel/response-atlas'),
        (value.expected_admission_corpus_schema, ControlledMapAdmissionReceiptCorpus.SCHEMA),
        (value.expected_admission_spec_schema, ReceiptAdmissionSpec.SCHEMA),
        (
            value.expected_reachability_spec_schema,
            ControlledMapReachabilitySpec.SCHEMA,
        ),
        (value.expected_model_set_schema, 'empirical-lawhood/kernel/model-set-spec'),
        (value.expected_measured_hold_fibre_schema, AdmissionMeasuredHoldFibre.SCHEMA),
        (
            value.expected_native_hold_calibration_receipt_schema,
            NativeHoldDecisionCellCalibrationReceipt.SCHEMA,
        ),
        (
            value.expected_selected_anchor_roster_schema,
            NativeHoldCalibrationSelectedRoster.SCHEMA,
        ),
        (
            value.expected_local_support_compatibility_receipt_schema,
            PreparedDenominatorLocalSupportCompatibility.SCHEMA,
        ),
    )
    if any(observed != expected for observed, expected in expected_schemas):
        raise ValueError("admission controller template expects another current record schema")
    exact_adapter_schemas = (
        (
            value.expected_observer_config_schema,
            'empirical-lawhood/control/bounded-region-finite-anchor-observer-config',
        ),
        (
            value.expected_observer_template_binding_receipt_schema,
            'empirical-lawhood/control/bounded-region-finite-anchor-observer-template-binding-receipt',
        ),
        (
            value.expected_online_gate_config_schema,
            'empirical-lawhood/control/evidence-bound-boolean-online-gate-config-template',
        ),
    )
    if any(observed != expected for observed, expected in exact_adapter_schemas):
        raise ValueError("admission controller template expects another observer/online-gate port")
    validate_sha256(
        value.template_binder_implementation_sha256,
        field_name="template_binder_implementation_sha256",
    )
    require_sorted_unique_strings(
        value.expected_response_law_ids,
        field_name="expected_response_law_ids",
        allow_empty=False,
    )
    require_sorted_unique_strings(
        value.model_member_ids,
        field_name="model_member_ids",
        allow_empty=False,
    )
    if (
        len(value.model_member_ids) != 2
        or len(value.expected_response_law_ids) != 2
        or (value.nominal_denominator_member_id not in value.model_member_ids)
    ):
        raise ValueError("admission controller template requires two members and one exact nominal")
    require_sorted_unique_ids(value.cell_pairs, attribute="tier_id", field_name="cell_pairs")
    if (
        len(value.cell_pairs) != 3
        or len({item.support_cell_id for item in value.cell_pairs}) != 3
        or len({item.active_candidate_id for item in value.cell_pairs}) != 3
        or len({item.active_decision_cell_id for item in value.cell_pairs}) != 3
        or len({item.active_admission_candidate_cell_id for item in value.cell_pairs}) != 3
        or len({item.sibling_hold_admission_candidate_cell_id for item in value.cell_pairs}) != 3
        or value.hold_decision_cell_id
        in {item.active_decision_cell_id for item in value.cell_pairs}
    ):
        raise ValueError("admission controller template requires exactly three distinct active pairs")
    hold_cell_ids = _bytewise_sorted(
        tuple(item.sibling_hold_admission_candidate_cell_id for item in value.cell_pairs)
    )
    if value.sibling_hold_selector.eligible_candidate_cell_ids != hold_cell_ids:
        raise ValueError("admission controller selector differs from the three sibling-HOLD cells")
    require_sorted_unique_ids(value.utility_rules, attribute="rule_id", field_name="utility_rules")
    utility_by_role = {item.role: item for item in value.utility_rules}
    if (
        len(value.utility_rules) != 2
        or set(utility_by_role) != set(AdmissionControllerUtilityRole)
        or utility_by_role[AdmissionControllerUtilityRole.ACTIVE].action_binding_id
        != value.active_action_binding_id
        or utility_by_role[AdmissionControllerUtilityRole.SIBLING_NATIVE_HOLD].action_binding_id
        != value.hold_action_binding_id
        or utility_by_role[AdmissionControllerUtilityRole.ACTIVE].minimum_utility
        != value.target_rule.active_materiality
        or utility_by_role[AdmissionControllerUtilityRole.SIBLING_NATIVE_HOLD].minimum_utility.unit
        != value.target_rule.hold_null_tolerance.unit
    ):
        raise ValueError("admission controller utility rules are incomplete or interchangeable")
    if (
        value.target_rule.active_action_binding_id != value.active_action_binding_id
        or value.target_rule.hold_action_binding_id != value.hold_action_binding_id
    ):
        raise ValueError("admission controller TARGET changes its action/HOLD bindings")
    require_sorted_unique_ids(
        value.gate_predicates,
        attribute="predicate_id",
        field_name="gate_predicates",
    )
    predicate_by_kind = {item.gate_kind: item for item in value.gate_predicates}
    if len(value.gate_predicates) != len(AdmissionGateKind) or set(predicate_by_kind) != set(
        AdmissionGateKind
    ):
        raise ValueError("admission controller template requires exactly nine gate predicates")
    if predicate_by_kind[AdmissionGateKind.TARGET] != value.target_rule.predicate:
        raise ValueError("admission controller template has two TARGET predicates")
    require_sorted_unique_ids(
        value.scalar_gate_requirements,
        attribute="requirement_id",
        field_name="scalar_gate_requirements",
    )
    scalar_by_kind = {item.gate_kind: item for item in value.scalar_gate_requirements}
    non_target = set(AdmissionGateKind) - {AdmissionGateKind.TARGET}
    if len(value.scalar_gate_requirements) != 8 or set(scalar_by_kind) != non_target:
        raise ValueError("admission controller template requires the eight non-TARGET scalars")
    for kind, requirement in scalar_by_kind.items():
        predicate = predicate_by_kind[kind]
        if (
            predicate.predicate_kind is not GatePredicateKind.SCALAR_AT_LEAST
            or predicate.quantity_id != requirement.observation_quantity_id
            or predicate.lower != requirement.minimum
        ):
            raise ValueError("online scalar requirement differs from its admission predicate")
    baseline = predicate_by_kind[AdmissionGateKind.BASELINE_PRESERVATION]
    if (
        value.baseline_compatibility_rule
        is not PreservationCompatibilityRule.EXACT_SEMANTIC_EQUALITY
        or not value.complete_baseline_evidence_required
        or baseline.temporal_semantics is None
    ):
        raise ValueError("baseline preservation loses exact law qualification/admission compatibility")
    require_sorted_unique_strings(
        value.preparation_quantity_ids,
        field_name="preparation_quantity_ids",
        allow_empty=False,
    )
    require_sorted_unique_strings(
        value.observation_quantity_ids,
        field_name="observation_quantity_ids",
        allow_empty=False,
    )
    expected_observations = tuple(
        sorted(
            {
                *(item.observation_quantity_id for item in value.scalar_gate_requirements),
                *value.preparation_quantity_ids,
                value.runtime_state_quantity_id,
            }
        )
    )
    if len(value.preparation_quantity_ids) != 4 or (
        value.observation_quantity_ids != expected_observations
        or len(value.observation_quantity_ids) != 13
    ):
        raise ValueError("admission controller observation vector is not the exact 8+4+1 roster")
    require_sorted_unique_ids(
        value.support_monitor_templates,
        attribute="monitor_id",
        field_name="support_monitor_templates",
    )
    active_monitors = {
        item.decision_cell_id: item
        for item in value.support_monitor_templates
        if item.source is ControllerSupportMonitorSource.ADMITTED_LOCAL_SUPPORT
    }
    hold_monitors = tuple(
        item
        for item in value.support_monitor_templates
        if item.source is ControllerSupportMonitorSource.SELECTED_ANCHOR_FINGERPRINTS
    )
    if (
        len(value.support_monitor_templates) != 4
        or set(active_monitors) != {item.active_decision_cell_id for item in value.cell_pairs}
        or any(
            active_monitors[item.active_decision_cell_id].expected_support_ids
            != (item.support_cell_id,)
            for item in value.cell_pairs
        )
        or len(hold_monitors) != 1
        or hold_monitors[0].decision_cell_id != value.hold_decision_cell_id
    ):
        raise ValueError("admission controller support-monitor topology differs")
    require_sorted_unique_ids(
        value.reidentification_trigger_templates,
        attribute="category",
        field_name="reidentification_trigger_templates",
    )
    if len(value.reidentification_trigger_templates) != len(
        ControllerReidentificationCategory
    ) or {item.category for item in value.reidentification_trigger_templates} != set(
        ControllerReidentificationCategory
    ):
        raise ValueError("admission controller reidentification categories are incomplete")
    require_sorted_unique_ids(
        value.implementation_templates,
        attribute="binding_id",
        field_name="implementation_templates",
    )
    implementation_by_role = {item.role: item for item in value.implementation_templates}
    if (
        len(value.implementation_templates) != len(CONTROLLER_ROLES)
        or set(implementation_by_role) != CONTROLLER_ROLES
        or len({item.expected_config_id for item in value.implementation_templates})
        != len(value.implementation_templates)
        or implementation_by_role[ImplementationRole.OBSERVER].expected_config_id
        != value.expected_observer_config_id
        or implementation_by_role[ImplementationRole.OBSERVER].expected_config_schema
        != value.expected_observer_config_schema
        or implementation_by_role[ImplementationRole.ONLINE_GATE_EVALUATOR].expected_config_id
        != value.expected_online_gate_config_id
        or implementation_by_role[ImplementationRole.ONLINE_GATE_EVALUATOR].expected_config_schema
        != value.expected_online_gate_config_schema
    ):
        raise ValueError("admission controller implementation role set is not exact")
    validate_decimal(
        value.worst_case_latency_seconds,
        field_name="worst_case_latency_seconds",
        minimum=Decimal(0),
    )
    validate_decimal(
        value.deadline_seconds,
        field_name="deadline_seconds",
        minimum=Decimal(0),
    )
    if (
        value.deadline_seconds <= 0
        or value.worst_case_latency_seconds > value.deadline_seconds
        or value.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        or value.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        or value.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
    ):
        raise ValueError("admission controller template overclaims timing, evidence or visibility")


@dataclass(frozen=True, slots=True)
class AdmissionControllerStudyTemplateBindingReceipt(CanonicalRecord):
    "Outcome-blind receipt for the deterministic eligible admission materialization."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-controller-study-template-binding-receipt'

    receipt_id: str
    template: ObjectIdentity
    system: ObjectIdentity
    atlas: ObjectIdentity
    admission_corpus: ObjectIdentity
    admission_spec: ObjectIdentity
    reachability_spec: ObjectIdentity
    model_set: ObjectIdentity
    selected_anchor_roster: ObjectIdentity
    native_hold_calibration: ObjectIdentity
    measured_hold_fibre: ObjectIdentity
    local_support_compatibility: ObjectIdentity
    observer_config: ObjectIdentity
    observer_template_binding_receipt: ObjectIdentity
    online_gate_config: ObjectIdentity
    selected_sibling_hold_admission_cell: ObjectIdentity
    candidate_chart: ObjectIdentity
    synthesis: ObjectIdentity
    implementation_bindings: tuple[ObjectIdentity, ...]
    binding_implementation_id: str
    binding_implementation_sha256: str
    outcome_dependent_candidate_selection: bool
    prospective_evaluation_embedded: bool
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    scientific_verdict: None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.binding_implementation_id,
            field_name="binding_implementation_id",
        )
        validate_sha256(
            self.binding_implementation_sha256,
            field_name="binding_implementation_sha256",
        )
        expected = (
            (self.template, AdmissionControllerStudyTemplate.SCHEMA),
            (self.system, SystemSpec.SCHEMA),
            (self.atlas, 'empirical-lawhood/kernel/response-atlas'),
            (self.admission_corpus, ControlledMapAdmissionReceiptCorpus.SCHEMA),
            (self.admission_spec, ReceiptAdmissionSpec.SCHEMA),
            (self.reachability_spec, ControlledMapReachabilitySpec.SCHEMA),
            (self.model_set, 'empirical-lawhood/kernel/model-set-spec'),
            (
                self.selected_anchor_roster,
                NativeHoldCalibrationSelectedRoster.SCHEMA,
            ),
            (
                self.native_hold_calibration,
                NativeHoldDecisionCellCalibrationReceipt.SCHEMA,
            ),
            (self.measured_hold_fibre, AdmissionMeasuredHoldFibre.SCHEMA),
            (
                self.local_support_compatibility,
                PreparedDenominatorLocalSupportCompatibility.SCHEMA,
            ),
            (self.candidate_chart, AdmissionActionCandidateChart.SCHEMA),
            (self.synthesis, AdmissionControllerSynthesisPlan.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected):
            raise ValueError("admission controller template receipt contains another schema")
        require_sorted_unique_ids(
            self.implementation_bindings,
            attribute="object_id",
            field_name="implementation_bindings",
        )
        if any(
            identity.object_schema != ImplementationBinding.SCHEMA
            for identity in self.implementation_bindings
        ):
            raise ValueError("admission controller receipt contains another implementation schema")
        if (
            self.outcome_dependent_candidate_selection
            or self.prospective_evaluation_embedded
            or self.scientific_verdict is not None
            or self.evidence_ceiling is not EvidenceCeiling.ADMISSION
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("admission controller binding receipt overclaims choice or outcome access")


def _require_expected_identity(
    identity: ObjectIdentity,
    *,
    expected_id: str,
    expected_schema: str,
    field_name: str,
) -> None:
    if identity.object_id != expected_id or identity.object_schema != expected_schema:
        raise ValueError(f'admission controller binding substitutes{field_name}')


def _utility_rule_by_role(
    template: AdmissionControllerStudyTemplate,
    role: AdmissionControllerUtilityRole,
) -> AdmissionControllerUtilityRule:
    return next(item for item in template.utility_rules if item.role is role)


def bind_admission_controller_study_template(
    *,
    receipt_id: str,
    template: AdmissionControllerStudyTemplate,
    system: SystemSpec,
    action_bindings: tuple[ControllerActionBinding, ...],
    admission: ReceiptAdmissionSpec,
    reachability: ControlledMapReachabilitySpec,
    measured_hold_fibre: AdmissionMeasuredHoldFibre,
    native_hold_calibration: NativeHoldDecisionCellCalibrationReceipt,
    selected_anchor_roster: NativeHoldCalibrationSelectedRoster,
    local_support_compatibility: PreparedDenominatorLocalSupportCompatibility,
    observer_config: ObjectIdentity,
    observer_template_binding_receipt: ObjectIdentity,
    online_gate_config: ObjectIdentity,
    implementation_configs: tuple[ObjectIdentity, ...],
    binding_implementation_id: str,
    binding_implementation_sha256: str,
) -> tuple[
    AdmissionControllerSynthesisPlan,
    tuple[ImplementationBinding, ...],
    AdmissionControllerStudyTemplateBindingReceipt,
]:
    "Attach exact eligible admission objects without selecting a favorable subset."

    if (
        binding_implementation_id != template.template_binder_implementation_id
        or binding_implementation_sha256 != template.template_binder_implementation_sha256
    ):
        raise ValueError("admission controller binding uses another frozen implementation")
    corpus = admission.corpus
    atlas = corpus.plan.atlas
    model_set = corpus.plan.model_set
    identities = {
        "system": ObjectIdentity.from_record(system.system_id, system),
        "atlas": ObjectIdentity.from_record(atlas.atlas_id, atlas),
        "controller admission corpus": ObjectIdentity.from_record(corpus.corpus_id, corpus),
        "admission spec": ObjectIdentity.from_record(admission.evaluation_id, admission),
        "reachability spec": ObjectIdentity.from_record(
            reachability.evaluation_id,
            reachability,
        ),
        "model set": ObjectIdentity.from_record(model_set.model_set_id, model_set),
        "selected anchor roster": ObjectIdentity.from_record(
            selected_anchor_roster.roster_id,
            selected_anchor_roster,
        ),
        "native-HOLD calibration": ObjectIdentity.from_record(
            native_hold_calibration.receipt_id,
            native_hold_calibration,
        ),
        "measured-HOLD fibre": ObjectIdentity.from_record(
            measured_hold_fibre.hold_fibre_id,
            measured_hold_fibre,
        ),
        "local-support compatibility": ObjectIdentity.from_record(
            local_support_compatibility.receipt_id,
            local_support_compatibility,
        ),
    }
    expected = {
        "system": (template.expected_system_id, template.expected_system_schema),
        "atlas": (template.expected_atlas_id, template.expected_atlas_schema),
        "controller admission corpus": (
            template.expected_admission_corpus_id,
            template.expected_admission_corpus_schema,
        ),
        "admission spec": (
            template.expected_admission_spec_id,
            template.expected_admission_spec_schema,
        ),
        "reachability spec": (
            template.expected_reachability_spec_id,
            template.expected_reachability_spec_schema,
        ),
        "model set": (
            template.expected_model_set_id,
            template.expected_model_set_schema,
        ),
        "selected anchor roster": (
            template.expected_selected_anchor_roster_id,
            template.expected_selected_anchor_roster_schema,
        ),
        "native-HOLD calibration": (
            template.expected_native_hold_calibration_receipt_id,
            template.expected_native_hold_calibration_receipt_schema,
        ),
        "measured-HOLD fibre": (
            template.expected_measured_hold_fibre_id,
            template.expected_measured_hold_fibre_schema,
        ),
        "local-support compatibility": (
            template.expected_local_support_compatibility_receipt_id,
            template.expected_local_support_compatibility_receipt_schema,
        ),
    }
    for field_name, identity in identities.items():
        expected_id, expected_schema = expected[field_name]
        _require_expected_identity(
            identity,
            expected_id=expected_id,
            expected_schema=expected_schema,
            field_name=field_name,
        )
    _require_expected_identity(
        observer_config,
        expected_id=template.expected_observer_config_id,
        expected_schema=template.expected_observer_config_schema,
        field_name="observer config",
    )
    _require_expected_identity(
        observer_template_binding_receipt,
        expected_id=template.expected_observer_template_binding_receipt_id,
        expected_schema=template.expected_observer_template_binding_receipt_schema,
        field_name="observer template binding receipt",
    )
    _require_expected_identity(
        online_gate_config,
        expected_id=template.expected_online_gate_config_id,
        expected_schema=template.expected_online_gate_config_schema,
        field_name="online-gate config",
    )
    if (
        reachability.admission_spec != admission
        or system.system_id != atlas.system_id
        or model_set.target_world_id != system.world.world_id
        or tuple(sorted(item.law_id for item in atlas.laws)) != template.expected_response_law_ids
        or tuple(sorted(item.denominator_member_id for item in model_set.members))
        != template.model_member_ids
        or corpus.plan.nominal_denominator_member_id != template.nominal_denominator_member_id
    ):
        raise ValueError("admission controller binding changes system/law/member parent geometry")
    support_ids = tuple(item.support_cell_id for item in template.cell_pairs)
    support_cells = tuple(sorted(corpus.plan.support_cells, key=lambda item: item.cell_id))
    exact_action_word_ids = {
        template.active_action_word_id,
        template.hold_action_word_id,
    }
    expected_action_bound_ids = tuple(
        sorted({bound for cell in support_cells for bound in cell.action_bound_ids})
    )
    atlas_law_identities = {ObjectIdentity.from_record(item.law_id, item) for item in atlas.laws}
    if (
        local_support_compatibility.evaluator_implementation
        != template.local_support_compatibility_evaluator
        or local_support_compatibility.disposition
        is not PreparedDenominatorLocalSupportCompatibilityDisposition.COMPATIBLE
        or local_support_compatibility.prepared_denominator_id != template.prepared_denominator_id
        or tuple(item.cell_id for item in support_cells) != tuple(sorted(support_ids))
        or local_support_compatibility.admission_support_cells != support_cells
        or local_support_compatibility.payload_local_support_ids != tuple(sorted(support_ids))
        or tuple(item.local_support_id for item in local_support_compatibility.local_supports)
        != tuple(sorted(support_ids))
        or local_support_compatibility.law_support_ids
        != tuple(sorted((template.prepared_denominator_id, *support_ids)))
        or local_support_compatibility.law_action_bound_ids != expected_action_bound_ids
        or {item.object_id for item in local_support_compatibility.action_words}
        != exact_action_word_ids
        or local_support_compatibility.response_law not in atlas_law_identities
    ):
        raise ValueError("admission controller binding loses prepared-D/local-support compatibility")
    action_fibre_ids = tuple(item.action_binding_id for item in corpus.plan.action_fibres)
    if (
        action_fibre_ids
        != tuple(sorted((template.active_action_binding_id, template.hold_action_binding_id)))
        or any(
            item.denominator_cell_id != template.prepared_denominator_id for item in support_cells
        )
        or any(
            item.candidate_version_ids != (template.expected_candidate_version_id,)
            for item in model_set.members
        )
        or len(corpus.plan.coordinates) != 12
    ):
        raise ValueError("admission controller binding changes the 2x1x2x3 Cartesian plan")

    admission_comparison = derive_receipt_admission_comparison(admission)
    reachability_comparison = derive_controlled_map_reachability_comparison(reachability)
    if (
        admission_comparison.nominal.status is not AdmissionStatus.ADMITTED
        or admission_comparison.robust.status is not AdmissionStatus.ADMITTED
        or not admission_comparison.structurally_stable
        or reachability_comparison.nominal.status is not ReachabilityStatus.REACHABLE
        or reachability_comparison.robust.status is not ReachabilityStatus.REACHABLE
        or not reachability_comparison.structurally_stable
    ):
        raise ValueError("admission controller template materializes only on complete robust admission")

    admission_candidate_cells_by_id = {item.candidate_cell_id: item for item in admission.candidate_cells}
    expected_cell_ids = {
        cell_id
        for pair in template.cell_pairs
        for cell_id in (
            pair.active_admission_candidate_cell_id,
            pair.sibling_hold_admission_candidate_cell_id,
        )
    }
    if set(admission_candidate_cells_by_id) != expected_cell_ids:
        raise ValueError("admission controller binding adds or omits an active/HOLD admission cell")
    for pair in template.cell_pairs:
        active = admission_candidate_cells_by_id[pair.active_admission_candidate_cell_id]
        hold = admission_candidate_cells_by_id[pair.sibling_hold_admission_candidate_cell_id]
        if (
            active.action_fibre.object_id != template.active_action_binding_id
            or hold.action_fibre.object_id != template.hold_action_binding_id
            or active.support_cell.object_id != pair.support_cell_id
            or hold.support_cell.object_id != pair.support_cell_id
            or active.planned_coordinate_ids == hold.planned_coordinate_ids
            or len(active.planned_coordinate_ids) != 2
            or len(hold.planned_coordinate_ids) != 2
            or {
                next(
                    coordinate.denominator_member_id
                    for coordinate in corpus.plan.coordinates
                    if coordinate.coordinate_id == coordinate_id
                )
                for coordinate_id in active.planned_coordinate_ids
            }
            != set(template.model_member_ids)
            or {
                next(
                    coordinate.denominator_member_id
                    for coordinate in corpus.plan.coordinates
                    if coordinate.coordinate_id == coordinate_id
                )
                for coordinate_id in hold.planned_coordinate_ids
            }
            != set(template.model_member_ids)
        ):
            raise ValueError("admission controller active/HOLD cell pairing differs")
    selected_hold_cell = admission_candidate_cells_by_id[template.sibling_hold_selector.selected_candidate_cell_id]

    action_by_id = {item.action_binding_id: item for item in action_bindings}
    if set(action_by_id) != {
        template.active_action_binding_id,
        template.hold_action_binding_id,
    }:
        raise ValueError("admission controller action-binding roster differs")
    if (
        action_by_id[template.active_action_binding_id].action_word.word_id
        != template.active_action_word_id
        or action_by_id[template.hold_action_binding_id].action_word.word_id
        != template.hold_action_word_id
    ):
        raise ValueError("admission controller action word identity differs")
    plan_fibres = {item.action_binding_id: item.action_word for item in corpus.plan.action_fibres}
    if {key: value.action_word for key, value in action_by_id.items()} != plan_fibres:
        raise ValueError("admission controller action binding changes the raw admission fibre")

    if corpus.plan.gate_predicates != template.gate_predicates:
        raise ValueError("admission controller raw gate predicates differ from the template")
    for reachability_receipt in corpus.reachability_receipts:
        action_id = reachability_receipt.planned_coordinate.action_fibre.object_id
        if action_id == template.active_action_binding_id:
            if (
                reachability_receipt.reference_kind
                is not ReceiptAdmissionReachabilityReferenceKind.ACTIVE_MINUS_QUALIFIED_HOLD
                or reachability_receipt.qualified_hold_action_fibre is None
                or reachability_receipt.qualified_hold_action_fibre.object_id
                != template.hold_action_binding_id
            ):
                raise ValueError("admission controller active reachability loses qualified HOLD")
        elif (
            action_id != template.hold_action_binding_id
            or reachability_receipt.reference_kind
            is not ReceiptAdmissionReachabilityReferenceKind.DECLARED_NATIVE_REFERENCE
            or reachability_receipt.declared_native_reference_direction_id is None
        ):
            raise ValueError("admission controller sibling-HOLD reachability is not native-reference")
    target_receipts = tuple(
        item
        for item in corpus.gate_receipts
        if item.predicate.gate_kind is AdmissionGateKind.TARGET
    )
    baseline_receipts = tuple(
        item
        for item in corpus.gate_receipts
        if item.predicate.gate_kind is AdmissionGateKind.BASELINE_PRESERVATION
    )
    if not target_receipts or any(
        item.predicate != template.target_rule.predicate
        or item.observed_boolean is not True
        or item.observed_scalar is not None
        or item.observed_identity is not None
        or item.status is not GateStatus.PASS
        or not item.input_artifacts
        or not item.evidence_links
        for item in target_receipts
    ):
        raise ValueError("admission controller TARGET receipt grid is not exact Boolean evidence")
    if not baseline_receipts or any(
        item.baseline_compatibility is None
        or item.baseline_compatibility.rule is not template.baseline_compatibility_rule
        or not item.baseline_compatibility.evidence_link_ids
        for item in baseline_receipts
    ):
        raise ValueError("admission controller baseline grid loses exact law qualification compatibility")

    utility_by_role = {
        role: _utility_rule_by_role(template, role) for role in AdmissionControllerUtilityRole
    }
    role_by_action = {
        template.active_action_binding_id: AdmissionControllerUtilityRole.ACTIVE,
        template.hold_action_binding_id: AdmissionControllerUtilityRole.SIBLING_NATIVE_HOLD,
    }
    for receipt in corpus.utility_receipts:
        role = role_by_action[receipt.planned_coordinate.action_fibre.object_id]
        rule = utility_by_role[role]
        if (
            receipt.utility_definition.object_id != rule.expected_utility_definition_id
            or receipt.utility_definition.object_schema != rule.expected_utility_definition_schema
            or receipt.direction is not rule.direction
            or receipt.minimum_utility != rule.minimum_utility
            or receipt.status is not ReceiptAdmissionUtilityStatus.VIABLE
        ):
            raise ValueError("admission controller utility receipt changes its action-local rule")

    selected_roster_identity = identities["selected anchor roster"]
    if (
        native_hold_calibration.selected_roster != selected_roster_identity
        or native_hold_calibration.disposition is not NativeHoldCalibrationDisposition.SUPPORTED
        or native_hold_calibration.selected_sibling_hold_candidate_key_id
        != selected_hold_cell.candidate_cell_id
        or native_hold_calibration.hold_decision_cell_id != template.hold_decision_cell_id
        or native_hold_calibration.active_decision_cell_ids
        != tuple(sorted(item.active_decision_cell_id for item in template.cell_pairs))
        or measured_hold_fibre.admission_candidate_cell
        != ObjectIdentity.from_record(
            selected_hold_cell.candidate_cell_id,
            selected_hold_cell,
        )
        or measured_hold_fibre.decision_cell_id != template.hold_decision_cell_id
        or measured_hold_fibre.calibration_id != native_hold_calibration.receipt_id
        or measured_hold_fibre.calibration_evidence_links != native_hold_calibration.evidence_links
        or measured_hold_fibre.null_tolerance != template.target_rule.hold_null_tolerance
        or not measured_hold_fibre.delivery_equivalence.require_exact_word_identity
        or not measured_hold_fibre.delivery_equivalence.require_all_delivery_stages
    ):
        raise ValueError("admission controller measured-HOLD/calibration lineage differs")
    hold_monitor = next(
        item
        for item in template.support_monitor_templates
        if item.source is ControllerSupportMonitorSource.SELECTED_ANCHOR_FINGERPRINTS
    )
    selected_by_slot = {item.slot_id: item for item in selected_anchor_roster.selections}
    if set(selected_by_slot) != set(hold_monitor.anchor_slot_ids):
        raise ValueError("admission controller selected anchor slots differ")

    config_by_id = {item.object_id: item for item in implementation_configs}
    if len(config_by_id) != len(implementation_configs):
        raise ValueError("admission controller implementation config identity is duplicated")
    expected_config_ids = {item.expected_config_id for item in template.implementation_templates}
    if set(config_by_id) != expected_config_ids:
        raise ValueError("admission controller implementation config roster differs")
    implementations = tuple(
        sorted(
            (
                item.materialize(config_by_id[item.expected_config_id])
                for item in template.implementation_templates
            ),
            key=lambda item: item.binding_id,
        )
    )
    by_role = {item.role: item for item in implementations}
    if (
        by_role[ImplementationRole.OBSERVER].config_sha256 != observer_config.object_fingerprint
        or by_role[ImplementationRole.ONLINE_GATE_EVALUATOR].config_sha256
        != online_gate_config.object_fingerprint
    ):
        raise ValueError("admission controller observer/online-gate config binding differs")

    candidates = tuple(
        sorted(
            (
                AdmissionActionCandidate(
                    candidate_id=item.active_candidate_id,
                    decision_cell_id=item.active_decision_cell_id,
                    admission_candidate_cell=ObjectIdentity.from_record(
                        item.active_admission_candidate_cell_id,
                        admission_candidate_cells_by_id[item.active_admission_candidate_cell_id],
                    ),
                    priority_rank=0,
                )
                for item in template.cell_pairs
            ),
            key=lambda item: item.candidate_id,
        )
    )
    active_utility = utility_by_role[AdmissionControllerUtilityRole.ACTIVE]
    first_active_utility = next(
        item
        for item in corpus.utility_receipts
        if item.planned_coordinate.action_fibre.object_id == template.active_action_binding_id
    )
    chart = AdmissionActionCandidateChart(
        chart_id=template.candidate_chart_id,
        model_set=identities["model set"],
        nominal_denominator_member_id=template.nominal_denominator_member_id,
        utility_definition=first_active_utility.utility_definition,
        utility_direction=active_utility.direction,
        minimum_robust_utility=active_utility.minimum_utility,
        priority_orientation=CandidatePriorityOrientation.EARLIER_WINS,
        candidates=candidates,
    )
    monitors = tuple(
        sorted(
            (
                OnlineSupportMonitorSpec(
                    monitor_id=item.monitor_id,
                    support_ids=(
                        item.expected_support_ids
                        if item.source is ControllerSupportMonitorSource.ADMITTED_LOCAL_SUPPORT
                        else tuple(
                            sorted(
                                selected_by_slot[slot_id].selected.preparation_fingerprint
                                for slot_id in item.anchor_slot_ids
                            )
                        )
                    ),
                    check_rule=item.check_rule,
                    invalidation_reason_code=item.invalidation_reason_code,
                )
                for item in template.support_monitor_templates
            ),
            key=lambda item: item.monitor_id,
        )
    )
    triggers = tuple(
        sorted(
            (item.trigger for item in template.reidentification_trigger_templates),
            key=lambda item: item.trigger_id,
        )
    )
    visibility = VisibilityCeiling.most_restrictive(
        corpus.plan.visibility_ceiling,
        atlas.visibility_ceiling,
        model_set.visibility_ceiling,
        admission.visibility_ceiling,
        reachability.visibility_ceiling,
    )
    synthesis = AdmissionControllerSynthesisPlan(
        synthesis_id=template.expected_synthesis_id,
        atlas=identities["atlas"],
        admission_corpus=identities["controller admission corpus"],
        admission_comparison=ObjectIdentity.from_record(
            admission_comparison.comparison_id,
            admission_comparison,
        ),
        reachability_comparison=ObjectIdentity.from_record(
            reachability_comparison.comparison_id,
            reachability_comparison,
        ),
        model_set=identities["model set"],
        candidate_chart=chart,
        admission_deriver_binding_id=by_role[ImplementationRole.ADMISSION_DERIVER].binding_id,
        reachability_deriver_binding_id=by_role[ImplementationRole.REACHABILITY_DERIVER].binding_id,
        synthesizer_binding_id=by_role[ImplementationRole.SYNTHESIZER].binding_id,
        observer_binding_id=by_role[ImplementationRole.OBSERVER].binding_id,
        online_gate_evaluator_binding_id=by_role[
            ImplementationRole.ONLINE_GATE_EVALUATOR
        ].binding_id,
        delivery_binding_id=by_role[ImplementationRole.DELIVERY].binding_id,
        observation_quantity_ids=template.observation_quantity_ids,
        support_monitors=monitors,
        reidentification_triggers=triggers,
        authority_policy_id=template.authority_policy_id,
        worst_case_latency_seconds=template.worst_case_latency_seconds,
        deadline_seconds=template.deadline_seconds,
        evidence_ceiling=EvidenceCeiling.ADMISSION,
        visibility_ceiling=visibility,
    )
    binding_receipt = AdmissionControllerStudyTemplateBindingReceipt(
        receipt_id=receipt_id,
        template=ObjectIdentity.from_record(template.template_id, template),
        system=identities["system"],
        atlas=identities["atlas"],
        admission_corpus=identities["controller admission corpus"],
        admission_spec=identities["admission spec"],
        reachability_spec=identities["reachability spec"],
        model_set=identities["model set"],
        selected_anchor_roster=selected_roster_identity,
        native_hold_calibration=identities["native-HOLD calibration"],
        measured_hold_fibre=identities["measured-HOLD fibre"],
        local_support_compatibility=identities["local-support compatibility"],
        observer_config=observer_config,
        observer_template_binding_receipt=observer_template_binding_receipt,
        online_gate_config=online_gate_config,
        selected_sibling_hold_admission_cell=ObjectIdentity.from_record(
            selected_hold_cell.candidate_cell_id,
            selected_hold_cell,
        ),
        candidate_chart=ObjectIdentity.from_record(chart.chart_id, chart),
        synthesis=ObjectIdentity.from_record(synthesis.synthesis_id, synthesis),
        implementation_bindings=tuple(
            ObjectIdentity.from_record(item.binding_id, item) for item in implementations
        ),
        binding_implementation_id=binding_implementation_id,
        binding_implementation_sha256=binding_implementation_sha256,
        outcome_dependent_candidate_selection=False,
        prospective_evaluation_embedded=False,
        evidence_ceiling=EvidenceCeiling.ADMISSION,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return synthesis, implementations, binding_receipt


class AdmissionControllerStudyTemplateBinder:
    "Code-owned deterministic wrapper around eligible admission materialization."

    def __init__(self, *, implementation_id: str, implementation_sha256: str) -> None:
        validate_stable_id(implementation_id, field_name="implementation_id")
        validate_sha256(implementation_sha256, field_name="implementation_sha256")
        self._implementation_id = implementation_id
        self._implementation_sha256 = implementation_sha256

    def bind(
        self,
        *,
        receipt_id: str,
        template: AdmissionControllerStudyTemplate,
        system: SystemSpec,
        action_bindings: tuple[ControllerActionBinding, ...],
        admission: ReceiptAdmissionSpec,
        reachability: ControlledMapReachabilitySpec,
        measured_hold_fibre: AdmissionMeasuredHoldFibre,
        native_hold_calibration: NativeHoldDecisionCellCalibrationReceipt,
        selected_anchor_roster: NativeHoldCalibrationSelectedRoster,
        local_support_compatibility: PreparedDenominatorLocalSupportCompatibility,
        observer_config: ObjectIdentity,
        observer_template_binding_receipt: ObjectIdentity,
        online_gate_config: ObjectIdentity,
        implementation_configs: tuple[ObjectIdentity, ...],
    ) -> tuple[
        AdmissionControllerSynthesisPlan,
        tuple[ImplementationBinding, ...],
        AdmissionControllerStudyTemplateBindingReceipt,
    ]:
        return bind_admission_controller_study_template(
            receipt_id=receipt_id,
            template=template,
            system=system,
            action_bindings=action_bindings,
            admission=admission,
            reachability=reachability,
            measured_hold_fibre=measured_hold_fibre,
            native_hold_calibration=native_hold_calibration,
            selected_anchor_roster=selected_anchor_roster,
            local_support_compatibility=local_support_compatibility,
            observer_config=observer_config,
            observer_template_binding_receipt=observer_template_binding_receipt,
            online_gate_config=online_gate_config,
            implementation_configs=implementation_configs,
            binding_implementation_id=self._implementation_id,
            binding_implementation_sha256=self._implementation_sha256,
        )


def decode_admission_controller_study_template(
    payload: bytes,
) -> AdmissionControllerStudyTemplate:
    return decode_canonical_bytes(
        payload,
        AdmissionControllerStudyTemplate,
        maximum_bytes=MAX_ADMISSION_CONTROLLER_PROGRAMME_TEMPLATE_BYTES,
    )


def decode_admission_controller_study_template_binding_receipt(
    payload: bytes,
) -> AdmissionControllerStudyTemplateBindingReceipt:
    return decode_canonical_bytes(
        payload,
        AdmissionControllerStudyTemplateBindingReceipt,
        maximum_bytes=MAX_ADMISSION_CONTROLLER_PROGRAMME_TEMPLATE_BYTES,
    )


__all__ = [
    'ControllerImplementationBindingTemplate',
    'ControllerReidentificationCategory',
    'ControllerReidentificationTriggerTemplate',
    'ControllerSupportMonitorSource',
    'ControllerSupportMonitorTemplate',
    "MAX_ADMISSION_CONTROLLER_PROGRAMME_TEMPLATE_BYTES",
    'OnlineScalarGateRequirement',
    'AdmissionControllerCellPairTemplate',
    'AdmissionControllerCellSelectionRule',
    'AdmissionControllerStudyTemplateBinder',
    'AdmissionControllerStudyTemplateBindingReceipt',
    'AdmissionControllerStudyTemplate',
    'AdmissionControllerTargetComparison',
    'AdmissionControllerTargetRule',
    'AdmissionControllerUtilityRole',
    'AdmissionControllerUtilityRule',
    'AdmissionSiblingHoldCellSelectorTemplate',
    'bind_admission_controller_study_template',
    'decode_admission_controller_study_template_binding_receipt',
    'decode_admission_controller_study_template',
]
