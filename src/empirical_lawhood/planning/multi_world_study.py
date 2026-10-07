"""Outcome-blind archive-to-simulator multi-world flagship contracts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)


PRIMARY_MORPHISM_PROPERTY_ID = "receiver-history-conditioned-two-action-response-class-refinement"


class MultiWorldChildRole(StrEnum):
    FAIR_MAST_ARCHIVE = "FAIR_MAST_ARCHIVE"
    MAPPED_DIRECT_TORAX = "MAPPED_DIRECT_TORAX"
    GENERATED_GYM_TORAX = "GENERATED_GYM_TORAX"


class MultiWorldOutcomeDomain(StrEnum):
    ARCHIVE_PREACTION = "ARCHIVE_PREACTION"
    ARCHIVE_ACTION_TRACE = "ARCHIVE_ACTION_TRACE"
    MAPPED_TORAX_OUTCOME = "MAPPED_TORAX_OUTCOME"
    GENERATED_TORAX_OUTCOME = "GENERATED_TORAX_OUTCOME"
    ARCHIVE_RECEIVER_OUTCOME = "ARCHIVE_RECEIVER_OUTCOME"


class MorphismFieldOrigin(StrEnum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    ASSUMED = "ASSUMED"
    MARGINALIZED = "MARGINALIZED"


class PropertyExpectationKind(StrEnum):
    TESTED_PRIMARY = "TESTED_PRIMARY"
    MANDATORY_DIAGNOSTIC = "MANDATORY_DIAGNOSTIC"
    NOT_ENTAILED = "NOT_ENTAILED"
    EXACT_MAPPING_DISPOSITION = "EXACT_MAPPING_DISPOSITION"


class MorphismNegativeControlKind(StrEnum):
    ACTION_SWAP = "ACTION_SWAP"
    PREACTION_TIME_SHIFT = "PREACTION_TIME_SHIFT"
    RADIAL_REVERSE = "RADIAL_REVERSE"


@dataclass(frozen=True, slots=True)
class MultiWorldStudyChildScientificBinding(CanonicalRecord):
    "World-local scientific identities retained by the bundle issue."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/multi-world-study-child-scientific-binding'

    child_id: str
    role: MultiWorldChildRole
    base_candidate: ObjectIdentity
    independent_unit_roster: ObjectIdentity
    evidence_world: ObjectIdentity
    law: ObjectIdentity
    maximum_evidence_ceiling: EvidenceCeiling
    protected_outcome_domain_id: str
    execution_authority_subject: ObjectIdentity
    reveal_authority_subject: ObjectIdentity
    maximum_outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.child_id, field_name="child_id")
        validate_stable_id(
            self.protected_outcome_domain_id,
            field_name="protected_outcome_domain_id",
        )
        if self.base_candidate.object_schema != 'empirical-lawhood/runtime/study-candidate':
            raise ValueError("bundle child science must bind a base standard candidate")
        expected_ceiling = (
            EvidenceCeiling.LOCAL_LAW
            if self.role is MultiWorldChildRole.FAIR_MAST_ARCHIVE
            else EvidenceCeiling.CONTROLLER_USE
        )
        if self.maximum_evidence_ceiling is not expected_ceiling:
            raise ValueError("bundle child evidence ceiling differs from its world-local role")
        if self.maximum_outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("protected child requires evaluator-only reveal access")


@dataclass(frozen=True, slots=True)
class ArchiveOutcomeProtectionPlan(CanonicalRecord):
    """Separate archive state/action staging from late receiver outcome reveal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/archive-outcome-protection-plan'

    plan_id: str
    archive_child_id: str
    preaction_domain_id: str
    action_trace_domain_id: str
    receiver_outcome_domain_id: str
    preaction_field_ids: tuple[str, ...]
    action_trace_field_ids: tuple[str, ...]
    receiver_outcome_field_ids: tuple[str, ...]
    outcome_blind_mapping_input_domains: tuple[MultiWorldOutcomeDomain, ...]
    prohibited_before_receiver_reveal: tuple[str, ...]
    one_event_per_independent_unit: bool
    receiver_reveal_is_late: bool
    mapping_maximum_outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("plan_id", self.plan_id),
            ("archive_child_id", self.archive_child_id),
            ("preaction_domain_id", self.preaction_domain_id),
            ("action_trace_domain_id", self.action_trace_domain_id),
            ("receiver_outcome_domain_id", self.receiver_outcome_domain_id),
        ):
            validate_stable_id(value, field_name=name)
        if (
            len(
                {
                    self.preaction_domain_id,
                    self.action_trace_domain_id,
                    self.receiver_outcome_domain_id,
                }
            )
            != 3
        ):
            raise ValueError("archive protection domains must remain distinct")
        for name, values in (
            ("preaction_field_ids", self.preaction_field_ids),
            ("action_trace_field_ids", self.action_trace_field_ids),
            ("receiver_outcome_field_ids", self.receiver_outcome_field_ids),
            ("prohibited_before_receiver_reveal", self.prohibited_before_receiver_reveal),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if set(self.preaction_field_ids) & set(self.receiver_outcome_field_ids) or set(
            self.action_trace_field_ids
        ) & set(self.receiver_outcome_field_ids):
            raise ValueError("archive receiver fields leaked into pre-reveal staging")
        if self.outcome_blind_mapping_input_domains != (
            MultiWorldOutcomeDomain.ARCHIVE_PREACTION,
            MultiWorldOutcomeDomain.ARCHIVE_ACTION_TRACE,
        ):
            raise ValueError("archive mapping inputs include a protected receiver domain")
        if (
            not set(self.receiver_outcome_field_ids).issubset(
                self.prohibited_before_receiver_reveal
            )
            or not self.one_event_per_independent_unit
            or not self.receiver_reveal_is_late
            or self.mapping_maximum_outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("archive outcome-protection contract is incomplete")


@dataclass(frozen=True, slots=True)
class MultiWorldOutcomeBarrierSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/multi-world-outcome-barrier-spec'

    barrier_id: str
    domain: MultiWorldOutcomeDomain
    child_id: str
    predecessor_barrier_ids: tuple[str, ...]
    authority_subject: ObjectIdentity
    prerequisite_execution_authority: ObjectIdentity
    reveal_grantee_id: str
    reveal_scope_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.barrier_id, field_name="barrier_id")
        validate_stable_id(self.child_id, field_name="child_id")
        validate_stable_id(self.reveal_grantee_id, field_name="reveal_grantee_id")
        validate_stable_id(self.reveal_scope_id, field_name="reveal_scope_id")
        require_sorted_unique_strings(
            self.predecessor_barrier_ids,
            field_name="predecessor_barrier_ids",
        )
        if self.domain in {
            MultiWorldOutcomeDomain.ARCHIVE_PREACTION,
            MultiWorldOutcomeDomain.ARCHIVE_ACTION_TRACE,
        }:
            raise ValueError("staged archive inputs are not reveal barriers")
        if self.prerequisite_execution_authority.object_schema != (
            'empirical-lawhood/planning/study-operation-authority'
        ):
            raise ValueError("reveal barrier lacks its exact execution-authority prerequisite")


@dataclass(frozen=True, slots=True)
class MultiWorldOutcomeBarrierPlan(CanonicalRecord):
    """One reveal per protected domain, with both simulators before archive."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/multi-world-outcome-barrier-plan'

    plan_id: str
    barriers: tuple[MultiWorldOutcomeBarrierSpec, ...]
    simulator_child_ids: tuple[str, ...]
    archive_child_id: str
    one_transition_per_domain: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        validate_stable_id(self.archive_child_id, field_name="archive_child_id")
        require_sorted_unique_ids(self.barriers, attribute="barrier_id", field_name="barriers")
        require_sorted_unique_strings(
            self.simulator_child_ids,
            field_name="simulator_child_ids",
            allow_empty=False,
        )
        if len(self.simulator_child_ids) != 2 or self.archive_child_id in self.simulator_child_ids:
            raise ValueError(
                "multi-world barrier plan requires two simulator children and one archive"
            )
        by_domain = {value.domain: value for value in self.barriers}
        required = {
            MultiWorldOutcomeDomain.MAPPED_TORAX_OUTCOME,
            MultiWorldOutcomeDomain.GENERATED_TORAX_OUTCOME,
            MultiWorldOutcomeDomain.ARCHIVE_RECEIVER_OUTCOME,
        }
        if len(self.barriers) != 3 or set(by_domain) != required:
            raise ValueError("multi-world reveal barrier roster is incomplete")
        simulator_barrier_ids = {
            by_domain[MultiWorldOutcomeDomain.MAPPED_TORAX_OUTCOME].barrier_id,
            by_domain[MultiWorldOutcomeDomain.GENERATED_TORAX_OUTCOME].barrier_id,
        }
        archive = by_domain[MultiWorldOutcomeDomain.ARCHIVE_RECEIVER_OUTCOME]
        if (
            {
                value.child_id
                for domain, value in by_domain.items()
                if domain is not MultiWorldOutcomeDomain.ARCHIVE_RECEIVER_OUTCOME
            }
            != set(self.simulator_child_ids)
            or archive.child_id != self.archive_child_id
            or set(archive.predecessor_barrier_ids) != simulator_barrier_ids
            or any(
                value.predecessor_barrier_ids
                for domain, value in by_domain.items()
                if domain is not MultiWorldOutcomeDomain.ARCHIVE_RECEIVER_OUTCOME
            )
            or len({value.authority_subject for value in self.barriers}) != 3
            or not self.one_transition_per_domain
        ):
            raise ValueError("multi-world barriers do not enforce simulator-before-archive reveal")


@dataclass(frozen=True, slots=True)
class MorphismFieldBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/morphism-field-binding'

    binding_id: str
    archive_field_id: str
    simulator_field_id: str | None
    origin: MorphismFieldOrigin
    mapping_rule_id: str
    acceptance_predicate_id: str
    rejection_disposition_id: str
    archive_unit: str
    simulator_unit: str | None
    transported_value: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("archive_field_id", self.archive_field_id),
            ("mapping_rule_id", self.mapping_rule_id),
            ("acceptance_predicate_id", self.acceptance_predicate_id),
            ("rejection_disposition_id", self.rejection_disposition_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.archive_unit, field_name="archive_unit")
        if self.simulator_field_id is not None:
            validate_stable_id(self.simulator_field_id, field_name="simulator_field_id")
        if self.simulator_unit is not None:
            validate_nonempty(self.simulator_unit, field_name="simulator_unit")
        if self.origin is MorphismFieldOrigin.MARGINALIZED:
            if self.simulator_field_id is not None or self.transported_value:
                raise ValueError("marginalized field cannot transport a simulator value")
        elif self.simulator_field_id is None or self.simulator_unit is None:
            raise ValueError("mapped field lacks its simulator coordinate/unit")


@dataclass(frozen=True, slots=True)
class PropertyTransportExpectation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/property-transport-expectation'

    expectation_id: str
    property_id: str
    kind: PropertyExpectationKind
    archive_operand: str
    simulator_operand: str
    equivalence_relation: str
    per_preparation_reducer: str
    decisive_falsifier: str
    required_member_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.expectation_id, field_name="expectation_id")
        validate_stable_id(self.property_id, field_name="property_id")
        for name, value in (
            ("archive_operand", self.archive_operand),
            ("simulator_operand", self.simulator_operand),
            ("equivalence_relation", self.equivalence_relation),
            ("per_preparation_reducer", self.per_preparation_reducer),
            ("decisive_falsifier", self.decisive_falsifier),
        ):
            validate_nonempty(value, field_name=name)
        expected_members = (
            6
            if self.kind
            in {
                PropertyExpectationKind.TESTED_PRIMARY,
                PropertyExpectationKind.MANDATORY_DIAGNOSTIC,
            }
            else 0
        )
        if self.required_member_count != expected_members:
            raise ValueError("property expectation changes its member reducer domain")
        if self.kind is PropertyExpectationKind.TESTED_PRIMARY and self.property_id != (
            PRIMARY_MORPHISM_PROPERTY_ID
        ):
            raise ValueError("another property cannot become the primary morphism test")


@dataclass(frozen=True, slots=True)
class ArchiveToSimulatorPartialMorphismSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/archive-to-simulator-partial-morphism-spec'

    morphism_id: str
    archive_child_id: str
    simulator_child_id: str
    field_bindings: tuple[MorphismFieldBinding, ...]
    action_correspondence: tuple[str, ...]
    expectations: tuple[PropertyTransportExpectation, ...]
    nontransported_property_ids: tuple[str, ...]
    acceptance_predicate_ids: tuple[str, ...]
    mapping_disposition_ids: tuple[str, ...]
    physical_hold_transport_forbidden: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("morphism_id", self.morphism_id),
            ("archive_child_id", self.archive_child_id),
            ("simulator_child_id", self.simulator_child_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.archive_child_id == self.simulator_child_id:
            raise ValueError("archive morphism endpoints must remain world-local")
        require_sorted_unique_ids(
            self.field_bindings,
            attribute="binding_id",
            field_name="field_bindings",
        )
        require_sorted_unique_ids(
            self.expectations,
            attribute="expectation_id",
            field_name="expectations",
        )
        for name, values in (
            ("nontransported_property_ids", self.nontransported_property_ids),
            ("acceptance_predicate_ids", self.acceptance_predicate_ids),
            ("mapping_disposition_ids", self.mapping_disposition_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if self.action_correspondence != ("NBI_DOWN:TORAX_DOWN", "NBI_UP:TORAX_UP"):
            raise ValueError("morphism action correspondence must exclude HOLD")
        expectation_by_property = {value.property_id: value for value in self.expectations}
        if (
            expectation_by_property.get(PRIMARY_MORPHISM_PROPERTY_ID) is None
            or expectation_by_property[PRIMARY_MORPHISM_PROPERTY_ID].kind
            is not PropertyExpectationKind.TESTED_PRIMARY
        ):
            raise ValueError("partial morphism omits the fixed primary property")
        required_nontransport = {
            "coefficient-or-response-magnitude",
            "uncertainty-threshold-materiality",
            "physical-or-numerical-hold",
            "law-qualification-controller-admission-reachability-utility-commitment-prospective-controller-evaluation",
        }
        if (
            not required_nontransport.issubset(self.nontransported_property_ids)
            or set(self.mapping_disposition_ids) != {"ACCEPTED", "REJECTED", "UNEVALUABLE"}
            or not self.physical_hold_transport_forbidden
        ):
            raise ValueError("partial morphism transport exclusions are incomplete")


@dataclass(frozen=True, slots=True)
class ArchiveOverlapQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/archive-overlap-qualification'

    qualification_id: str
    minimum_evaluable_states: int
    minimum_per_campaign: int
    minimum_per_action: int
    selected_recurrence_wilson_lower_exclusive: Decimal
    minimum_refinement_contact_rows: int
    minimum_selected_correct_contact_rows: int
    minimum_selected_advantage_rows: int

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        if (
            self.minimum_evaluable_states != 12
            or self.minimum_per_campaign != 3
            or self.minimum_per_action != 6
            or self.selected_recurrence_wilson_lower_exclusive != Decimal("0.65")
            or self.minimum_refinement_contact_rows != 8
            or self.minimum_selected_correct_contact_rows != 6
            or self.minimum_selected_advantage_rows != 4
        ):
            raise ValueError("archive overlap thresholds differ from the frozen H10 contract")


@dataclass(frozen=True, slots=True)
class MorphismNegativeControlPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/morphism-negative-control-plan'

    plan_id: str
    controls: tuple[MorphismNegativeControlKind, ...]
    common_denominator_required: bool
    same_six_member_reducer: bool
    additional_simulator_launches: int
    minimum_preserved_count_advantage: int
    minimum_preserved_fraction_advantage: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        if (
            self.controls != tuple(MorphismNegativeControlKind)
            or not self.common_denominator_required
            or not self.same_six_member_reducer
            or self.additional_simulator_launches != 0
            or self.minimum_preserved_count_advantage != 4
            or self.minimum_preserved_fraction_advantage != Decimal("0.15")
        ):
            raise ValueError("morphism negative-control contract differs from H10")


@dataclass(frozen=True, slots=True)
class MultiWorldJointAdjudicationPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/multi-world-joint-adjudication-plan'

    plan_id: str
    archive_child_id: str
    mapped_child_id: str
    generated_child_id: str
    primary_property_id: str
    overlap_qualification: ArchiveOverlapQualification
    negative_controls: MorphismNegativeControlPlan
    mapped_prospective_attempt_count: int
    minimum_accepted_states: int
    minimum_accepted_fraction: Decimal
    maximum_map_unevaluable_fraction: Decimal
    minimum_contacted_states: int
    maximum_property_unevaluable_fraction: Decimal
    preservation_wilson_lower_exclusive: Decimal
    archive_h9_required: bool
    simulator_success_cannot_rescue_archive: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        for name, value in (
            ("archive_child_id", self.archive_child_id),
            ("mapped_child_id", self.mapped_child_id),
            ("generated_child_id", self.generated_child_id),
        ):
            validate_stable_id(value, field_name=name)
        if len({self.archive_child_id, self.mapped_child_id, self.generated_child_id}) != 3:
            raise ValueError("joint adjudication children must remain distinct")
        if (
            self.primary_property_id != PRIMARY_MORPHISM_PROPERTY_ID
            or self.mapped_prospective_attempt_count != 45
            or self.minimum_accepted_states != 24
            or self.minimum_accepted_fraction != Decimal("0.50")
            or self.maximum_map_unevaluable_fraction != Decimal("0.50")
            or self.minimum_contacted_states != 18
            or self.maximum_property_unevaluable_fraction != Decimal("0.50")
            or self.preservation_wilson_lower_exclusive != Decimal("0.80")
            or not self.archive_h9_required
            or not self.simulator_success_cannot_rescue_archive
        ):
            raise ValueError("joint adjudication thresholds differ from frozen H10")


__all__ = [
    'ArchiveOutcomeProtectionPlan',
    'ArchiveOverlapQualification',
    'ArchiveToSimulatorPartialMorphismSpec',
    'MultiWorldJointAdjudicationPlan',
    'MorphismFieldBinding',
    "MorphismFieldOrigin",
    "MorphismNegativeControlKind",
    'MorphismNegativeControlPlan',
    "MultiWorldChildRole",
    'MultiWorldOutcomeBarrierPlan',
    'MultiWorldOutcomeBarrierSpec',
    "MultiWorldOutcomeDomain",
    "PRIMARY_MORPHISM_PROPERTY_ID",
    'MultiWorldStudyChildScientificBinding',
    "PropertyExpectationKind",
    'PropertyTransportExpectation',
]
