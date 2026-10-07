"""Deterministic archive-overlap, morphism and non-pooling joint adjudication."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.multi_world_study import ArchiveOverlapQualification, MultiWorldJointAdjudicationPlan, MorphismNegativeControlKind, MorphismNegativeControlPlan, PRIMARY_MORPHISM_PROPERTY_ID, PropertyExpectationKind, PropertyTransportExpectation


_WILSON_Z_95 = Decimal("1.959963984540054")


class ArchiveResponseClass(StrEnum):
    DOWN_OR_NEGATIVE = "DOWN_OR_NEGATIVE"
    WITHIN_MATERIAL_EQUIVALENCE = "WITHIN_MATERIAL_EQUIVALENCE"
    UP_OR_POSITIVE = "UP_OR_POSITIVE"


class TwoActionMorphismAction(StrEnum):
    DOWN = "DOWN"
    UP = "UP"


class MappingDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNEVALUABLE = "UNEVALUABLE"


class PropertyMorphismDisposition(StrEnum):
    PRESERVED = "PRESERVED"
    REJECTED = "REJECTED"
    NOT_ENTAILED = "NOT_ENTAILED"
    UNEVALUABLE = "UNEVALUABLE"


class JointMultiWorldStudyDisposition(StrEnum):
    FULL_JOINT_SUCCESS = "FULL_JOINT_SUCCESS"
    ARCHIVE_BRIDGE_WITHOUT_FULL_SIMULATOR_SUCCESS = "ARCHIVE_BRIDGE_WITHOUT_FULL_SIMULATOR_SUCCESS"
    WORLD_LOCAL_WITHOUT_BRIDGE = "WORLD_LOCAL_WITHOUT_BRIDGE"
    ARCHIVE_LOCAL_ONLY = "ARCHIVE_LOCAL_ONLY"
    SIMULATOR_ONLY_MIXED = "SIMULATOR_ONLY_MIXED"
    NEGATIVE = "NEGATIVE"
    UNEVALUABLE = "UNEVALUABLE"


def wilson_lower_bound_95(successes: int, total: int) -> Decimal:
    if successes < 0 or total < 0 or successes > total:
        raise ValueError("Wilson counts are invalid")
    if total == 0:
        return Decimal(0)
    n = Decimal(total)
    p = Decimal(successes) / n
    z2 = _WILSON_Z_95 * _WILSON_Z_95
    denominator = Decimal(1) + z2 / n
    centre = p + z2 / (Decimal(2) * n)
    radicand = p * (Decimal(1) - p) / n + z2 / (Decimal(4) * n * n)
    return (centre - _WILSON_Z_95 * radicand.sqrt()) / denominator


@dataclass(frozen=True, slots=True)
class ArchiveOverlapObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/archive-overlap-observation'

    state_id: str
    campaign_id: str
    realized_action: TwoActionMorphismAction
    endpoint_evaluable: bool
    selected_declared: bool
    selected_supported: bool
    selected_class: ArchiveResponseClass | None
    action_only_class: ArchiveResponseClass | None
    observed_class: ArchiveResponseClass | None

    def __post_init__(self) -> None:
        validate_stable_id(self.state_id, field_name="state_id")
        if self.campaign_id not in {"M7", "M8", "M9"}:
            raise ValueError("archive overlap campaign is outside M7/M8/M9")
        if self.endpoint_evaluable != (self.observed_class is not None):
            raise ValueError("archive overlap endpoint state and class differ")
        if self.selected_declared != (self.selected_class is not None):
            raise ValueError("selected archive declaration and class differ")

    @property
    def selected_correct(self) -> bool:
        return bool(
            self.endpoint_evaluable
            and self.selected_declared
            and self.selected_supported
            and self.selected_class is self.observed_class
        )

    @property
    def action_only_correct(self) -> bool:
        return bool(
            self.endpoint_evaluable
            and self.action_only_class is not None
            and self.action_only_class is self.observed_class
        )

    @property
    def refinement_contacted(self) -> bool:
        return bool(
            self.selected_declared
            and self.selected_supported
            and self.action_only_class is not None
            and self.selected_class is not self.action_only_class
        )


@dataclass(frozen=True, slots=True)
class ArchiveOverlapQualificationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/archive-overlap-qualification-result'

    result_id: str
    qualification: ObjectIdentity
    observations: tuple[ArchiveOverlapObservation, ...]
    endpoint_evaluable_count: int
    campaign_evaluable_counts: tuple[str, ...]
    action_evaluable_counts: tuple[str, ...]
    selected_correct_count: int
    selected_recurrence_wilson_lower: Decimal
    refinement_contact_count: int
    selected_correct_contact_count: int
    action_only_correct_contact_count: int
    selected_advantage_rows: int
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.qualification.object_schema != ArchiveOverlapQualification.SCHEMA:
            raise ValueError("archive overlap result binds another qualification schema")
        require_sorted_unique_ids(
            self.observations,
            attribute="state_id",
            field_name="observations",
        )
        require_sorted_unique_strings(
            self.campaign_evaluable_counts,
            field_name="campaign_evaluable_counts",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.action_evaluable_counts,
            field_name="action_evaluable_counts",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.passed == bool(self.reason_codes):
            raise ValueError("archive overlap decision and reasons differ")


def evaluate_archive_overlap_qualification(
    *,
    result_id: str,
    qualification: ArchiveOverlapQualification,
    observations: tuple[ArchiveOverlapObservation, ...],
) -> ArchiveOverlapQualificationResult:
    ordered = tuple(sorted(observations, key=lambda value: value.state_id))
    if len({value.state_id for value in ordered}) != len(ordered):
        raise ValueError("archive overlap contains duplicate accepted states")
    evaluable = tuple(value for value in ordered if value.endpoint_evaluable)
    campaign_counts = Counter(value.campaign_id for value in evaluable)
    action_counts = Counter(value.realized_action.value for value in evaluable)
    selected_correct = sum(value.selected_correct for value in evaluable)
    contacted = tuple(value for value in evaluable if value.refinement_contacted)
    selected_contact_correct = sum(value.selected_correct for value in contacted)
    action_contact_correct = sum(value.action_only_correct for value in contacted)
    advantage = selected_contact_correct - action_contact_correct
    lower = wilson_lower_bound_95(selected_correct, len(evaluable))
    reasons: set[str] = set()
    if len(evaluable) < qualification.minimum_evaluable_states:
        reasons.add("ARCHIVE_OVERLAP_INSUFFICIENT_EVALUABLE_STATES")
    if any(
        campaign_counts[value] < qualification.minimum_per_campaign for value in ("M7", "M8", "M9")
    ):
        reasons.add("ARCHIVE_OVERLAP_CAMPAIGN_SUPPORT_INSUFFICIENT")
    if any(
        action_counts[value.value] < qualification.minimum_per_action
        for value in TwoActionMorphismAction
    ):
        reasons.add("ARCHIVE_OVERLAP_ACTION_SUPPORT_INSUFFICIENT")
    if lower <= qualification.selected_recurrence_wilson_lower_exclusive:
        reasons.add("ARCHIVE_OVERLAP_RECURRENCE_WILSON_FAILED")
    if len(contacted) < qualification.minimum_refinement_contact_rows:
        reasons.add("ARCHIVE_OVERLAP_REFINEMENT_NOT_CONTACTED")
    if selected_contact_correct < qualification.minimum_selected_correct_contact_rows:
        reasons.add("ARCHIVE_OVERLAP_SELECTED_CONTACT_CORRECTNESS_FAILED")
    if advantage < qualification.minimum_selected_advantage_rows:
        reasons.add("ARCHIVE_OVERLAP_ACTION_ONLY_ADVANTAGE_FAILED")
    return ArchiveOverlapQualificationResult(
        result_id=result_id,
        qualification=ObjectIdentity.from_record(qualification.qualification_id, qualification),
        observations=ordered,
        endpoint_evaluable_count=len(evaluable),
        campaign_evaluable_counts=tuple(
            sorted(f"{key}:{campaign_counts[key]}" for key in ("M7", "M8", "M9"))
        ),
        action_evaluable_counts=tuple(
            sorted(f"{key.value}:{action_counts[key.value]}" for key in TwoActionMorphismAction)
        ),
        selected_correct_count=selected_correct,
        selected_recurrence_wilson_lower=lower,
        refinement_contact_count=len(contacted),
        selected_correct_contact_count=selected_contact_correct,
        action_only_correct_contact_count=action_contact_correct,
        selected_advantage_rows=advantage,
        passed=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class ArchiveTwoActionClassPair(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/archive-two-action-class-pair'

    down: ArchiveResponseClass
    up: ArchiveResponseClass


@dataclass(frozen=True, slots=True)
class ToraxMemberTwoActionClassPair(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/torax-member-two-action-class-pair'

    member_id: str
    down: ArchiveResponseClass | None
    up: ArchiveResponseClass | None
    valid: bool
    nominal_reference_profile: ObjectIdentity | None

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        complete = (
            self.down is not None
            and self.up is not None
            and self.nominal_reference_profile is not None
        )
        if self.valid != complete:
            raise ValueError("TORAX member validity differs from its complete class/profile pair")


@dataclass(frozen=True, slots=True)
class PropertyMorphismStateInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/property-morphism-state-input'

    state_id: str
    property_id: str
    mapping_disposition: MappingDisposition
    archive_supported: bool
    mapped_down_supported: bool
    mapped_up_supported: bool
    archive_selected_pair: ArchiveTwoActionClassPair | None
    archive_action_only_pair: ArchiveTwoActionClassPair | None
    torax_members: tuple[ToraxMemberTwoActionClassPair, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.state_id, field_name="state_id")
        validate_stable_id(self.property_id, field_name="property_id")
        require_sorted_unique_ids(
            self.torax_members,
            attribute="member_id",
            field_name="torax_members",
        )
        if self.mapping_disposition is MappingDisposition.ACCEPTED and len(self.torax_members) != 6:
            raise ValueError("accepted morphism state requires the exact six-member TORAX product")


@dataclass(frozen=True, slots=True)
class PropertyMorphismMemberVerdict(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/property-morphism-member-verdict'

    member_id: str
    down_equal: bool | None
    up_equal: bool | None
    valid: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        if self.valid != (self.down_equal is not None and self.up_equal is not None):
            raise ValueError("morphism member validity differs from its equality verdicts")


@dataclass(frozen=True, slots=True)
class PropertyMorphismVerdict(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/property-morphism-verdict'

    verdict_id: str
    expectation: ObjectIdentity
    state_input: ObjectIdentity
    property_id: str
    property_contacted: bool
    member_verdicts: tuple[PropertyMorphismMemberVerdict, ...]
    disposition: PropertyMorphismDisposition
    decisive_falsifier_member_id: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.verdict_id, field_name="verdict_id")
        validate_stable_id(self.property_id, field_name="property_id")
        if self.expectation.object_schema != PropertyTransportExpectation.SCHEMA:
            raise ValueError("morphism verdict binds another expectation schema")
        if self.state_input.object_schema != PropertyMorphismStateInput.SCHEMA:
            raise ValueError("morphism verdict binds another state-input schema")
        require_sorted_unique_ids(
            self.member_verdicts,
            attribute="member_id",
            field_name="member_verdicts",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.decisive_falsifier_member_id is not None:
            validate_stable_id(
                self.decisive_falsifier_member_id,
                field_name="decisive_falsifier_member_id",
            )
        if self.disposition is PropertyMorphismDisposition.REJECTED:
            if self.decisive_falsifier_member_id is None:
                raise ValueError("rejected morphism verdict lacks its member falsifier")
        elif self.decisive_falsifier_member_id is not None:
            raise ValueError("non-rejected morphism verdict cannot name a falsifier")


def evaluate_property_morphism(
    *,
    verdict_id: str,
    expectation: PropertyTransportExpectation,
    state: PropertyMorphismStateInput,
) -> PropertyMorphismVerdict:
    expectation_identity = ObjectIdentity.from_record(expectation.expectation_id, expectation)
    state_identity = ObjectIdentity.from_record(state.state_id, state)
    if expectation.property_id != state.property_id:
        raise ValueError("property morphism input differs from its preissued expectation")
    if expectation.kind is PropertyExpectationKind.NOT_ENTAILED:
        return PropertyMorphismVerdict(
            verdict_id=verdict_id,
            expectation=expectation_identity,
            state_input=state_identity,
            property_id=state.property_id,
            property_contacted=False,
            member_verdicts=(),
            disposition=PropertyMorphismDisposition.NOT_ENTAILED,
            decisive_falsifier_member_id=None,
            reason_codes=("PROPERTY_OUTSIDE_DECLARED_MORPHISM_DOMAIN",),
        )
    selected = state.archive_selected_pair
    action_only = state.archive_action_only_pair
    contacted = bool(
        selected is not None
        and action_only is not None
        and (selected.down is not action_only.down or selected.up is not action_only.up)
    )
    complete_support = (
        state.mapping_disposition is MappingDisposition.ACCEPTED
        and state.archive_supported
        and state.mapped_down_supported
        and state.mapped_up_supported
        and selected is not None
    )
    member_verdicts = tuple(
        PropertyMorphismMemberVerdict(
            member_id=value.member_id,
            down_equal=(
                None if not value.valid or selected is None else value.down is selected.down
            ),
            up_equal=(None if not value.valid or selected is None else value.up is selected.up),
            valid=value.valid and selected is not None,
        )
        for value in state.torax_members
    )
    reasons: set[str] = set()
    if not complete_support:
        reasons.add("MORPHISM_OPERAND_UNSUPPORTED_OR_UNEVALUABLE")
    if expectation.kind is PropertyExpectationKind.TESTED_PRIMARY and not contacted:
        reasons.add("PRIMARY_MORPHISM_PROPERTY_NOT_CONTACTED")
    if len(member_verdicts) != expectation.required_member_count or any(
        not value.valid for value in member_verdicts
    ):
        reasons.add("MORPHISM_MEMBER_PRODUCT_UNEVALUABLE")
    if reasons:
        disposition = PropertyMorphismDisposition.UNEVALUABLE
        falsifier = None
    else:
        mismatches = tuple(
            value
            for value in member_verdicts
            if value.down_equal is False or value.up_equal is False
        )
        disposition = (
            PropertyMorphismDisposition.REJECTED
            if mismatches
            else PropertyMorphismDisposition.PRESERVED
        )
        falsifier = None if not mismatches else mismatches[0].member_id
        if mismatches:
            reasons.add("VALID_MEMBER_ACTION_CLASS_MISMATCH")
    return PropertyMorphismVerdict(
        verdict_id=verdict_id,
        expectation=expectation_identity,
        state_input=state_identity,
        property_id=state.property_id,
        property_contacted=contacted,
        member_verdicts=member_verdicts,
        disposition=disposition,
        decisive_falsifier_member_id=falsifier,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class MorphismControlStateObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/morphism-control-state-observation'

    observation_id: str
    state_id: str
    control: MorphismNegativeControlKind
    genuine_verdict: ObjectIdentity
    genuine_preserved: bool
    control_preserved: bool | None
    common_denominator_id: str
    raw_nominal_reference_profiles: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        validate_stable_id(self.state_id, field_name="state_id")
        validate_stable_id(self.common_denominator_id, field_name="common_denominator_id")
        if self.genuine_verdict.object_schema != PropertyMorphismVerdict.SCHEMA:
            raise ValueError("morphism control binds another genuine verdict schema")


@dataclass(frozen=True, slots=True)
class MorphismControlContrast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/morphism-control-contrast'

    control: MorphismNegativeControlKind
    contacted_count: int
    genuine_preserved_count: int
    control_preserved_count: int
    preserved_count_advantage: int
    preserved_fraction_advantage: Decimal
    evaluable: bool
    passed: bool

    def __post_init__(self) -> None:
        if self.contacted_count <= 0:
            raise ValueError("morphism control contrast requires contacted states")
        if self.passed and not self.evaluable:
            raise ValueError("unevaluable morphism control cannot pass")


@dataclass(frozen=True, slots=True)
class MorphismControlContrastReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/morphism-control-contrast-receipt'

    receipt_id: str
    plan: ObjectIdentity
    contacted_state_ids: tuple[str, ...]
    observations: tuple[MorphismControlStateObservation, ...]
    contrasts: tuple[MorphismControlContrast, ...]
    common_denominator_id: str
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.common_denominator_id, field_name="common_denominator_id")
        if self.plan.object_schema != MorphismNegativeControlPlan.SCHEMA:
            raise ValueError("morphism contrast receipt binds another control plan")
        require_sorted_unique_strings(
            self.contacted_state_ids,
            field_name="contacted_state_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.observations,
            attribute="observation_id",
            field_name="observations",
        )
        if {value.control for value in self.contrasts} != set(MorphismNegativeControlKind):
            raise ValueError("morphism contrast roster is incomplete")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.passed == bool(self.reason_codes):
            raise ValueError("morphism contrast decision and reasons differ")


def evaluate_morphism_negative_controls(
    *,
    receipt_id: str,
    plan: MorphismNegativeControlPlan,
    contacted_state_ids: tuple[str, ...],
    observations: tuple[MorphismControlStateObservation, ...],
) -> MorphismControlContrastReceipt:
    states = tuple(sorted(contacted_state_ids))
    require_sorted_unique_strings(states, field_name="contacted_state_ids", allow_empty=False)
    ordered = tuple(sorted(observations, key=lambda value: value.observation_id))
    if len({value.observation_id for value in ordered}) != len(ordered):
        raise ValueError("morphism control observations contain duplicate identities")
    expected = {(state_id, control) for state_id in states for control in plan.controls}
    actual = {(value.state_id, value.control) for value in ordered}
    if actual != expected or len(ordered) != len(expected):
        raise ValueError("morphism controls do not retain the exact common denominator")
    denominator_ids = {value.common_denominator_id for value in ordered}
    if len(denominator_ids) != 1:
        raise ValueError("morphism controls changed their common denominator identity")
    genuine_by_state: dict[str, bool] = {}
    for value in ordered:
        previous = genuine_by_state.setdefault(value.state_id, value.genuine_preserved)
        if previous != value.genuine_preserved:
            raise ValueError("morphism control changed the genuine-map state verdict")
    genuine_count = sum(genuine_by_state.values())
    contrasts: list[MorphismControlContrast] = []
    reasons: set[str] = set()
    for control in plan.controls:
        rows = tuple(value for value in ordered if value.control is control)
        evaluable = all(value.control_preserved is not None for value in rows)
        control_count = sum(value.control_preserved is True for value in rows)
        count_advantage = genuine_count - control_count
        fraction_advantage = Decimal(count_advantage) / Decimal(len(states))
        passed = bool(
            evaluable
            and count_advantage >= plan.minimum_preserved_count_advantage
            and fraction_advantage >= plan.minimum_preserved_fraction_advantage
        )
        if not evaluable:
            reasons.add(f"MORPHISM_CONTROL_UNEVALUABLE_{control.value}")
        elif not passed:
            reasons.add(f"MORPHISM_CONTROL_CONTRAST_FAILED_{control.value}")
        contrasts.append(
            MorphismControlContrast(
                control=control,
                contacted_count=len(states),
                genuine_preserved_count=genuine_count,
                control_preserved_count=control_count,
                preserved_count_advantage=count_advantage,
                preserved_fraction_advantage=fraction_advantage,
                evaluable=evaluable,
                passed=passed,
            )
        )
    return MorphismControlContrastReceipt(
        receipt_id=receipt_id,
        plan=ObjectIdentity.from_record(plan.plan_id, plan),
        contacted_state_ids=states,
        observations=ordered,
        contrasts=tuple(sorted(contrasts, key=lambda value: value.control.value)),
        common_denominator_id=next(iter(denominator_ids)),
        passed=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class WorldLocalStudyResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/world-local-study-result'

    result_id: str
    child_id: str
    result: ObjectIdentity
    h6_compiler_closure_passed: bool | None
    prospective_controller_evaluation_passed: bool | None
    h9_archive_law_passed: bool | None
    archive_temporal_placebo_passed: bool | None
    terminal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.child_id, field_name="child_id")
        if not self.terminal:
            raise ValueError("joint adjudication requires terminal child results")
        archive = self.h9_archive_law_passed is not None
        simulator = self.h6_compiler_closure_passed is not None
        if archive == simulator:
            raise ValueError("world-local result collapses archive and simulator hypotheses")
        if archive != (self.archive_temporal_placebo_passed is not None):
            raise ValueError("archive H9 result lacks its temporal-placebo verdict")
        if simulator != (self.prospective_controller_evaluation_passed is not None):
            raise ValueError("simulator result lacks its H6/H7 pair")


@dataclass(frozen=True, slots=True)
class MultiWorldJointAdjudicationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-joint-adjudication-result'

    result_id: str
    plan: ObjectIdentity
    child_results: tuple[WorldLocalStudyResult, ...]
    overlap_result: ObjectIdentity
    morphism_verdicts: tuple[PropertyMorphismVerdict, ...]
    control_contrast: ObjectIdentity
    mapped_prospective_accepted_count: int
    mapped_prospective_map_unevaluable_count: int
    property_contacted_count: int
    property_preserved_count: int
    property_unevaluable_count: int
    property_preservation_wilson_lower: Decimal
    h9_passed: bool
    h10_passed: bool
    mapped_simulator_passed: bool
    generated_simulator_passed: bool
    disposition: JointMultiWorldStudyDisposition
    reason_codes: tuple[str, ...]
    terminal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.plan.object_schema != MultiWorldJointAdjudicationPlan.SCHEMA:
            raise ValueError("joint result binds another adjudication plan")
        require_sorted_unique_ids(
            self.child_results,
            attribute="child_id",
            field_name="child_results",
        )
        require_sorted_unique_ids(
            self.morphism_verdicts,
            attribute="verdict_id",
            field_name="morphism_verdicts",
        )
        if self.overlap_result.object_schema != ArchiveOverlapQualificationResult.SCHEMA:
            raise ValueError("joint result binds another archive-overlap result schema")
        if self.control_contrast.object_schema != MorphismControlContrastReceipt.SCHEMA:
            raise ValueError("joint result binds another control-contrast schema")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not self.terminal:
            raise ValueError("joint flagship adjudication must be terminal")
        if self.disposition is JointMultiWorldStudyDisposition.FULL_JOINT_SUCCESS and not (
            self.h9_passed
            and self.h10_passed
            and self.mapped_simulator_passed
            and self.generated_simulator_passed
        ):
            raise ValueError("simulator-local success rescued an incomplete joint result")


def adjudicate_joint_multi_world_study(
    *,
    result_id: str,
    plan: MultiWorldJointAdjudicationPlan,
    child_results: tuple[WorldLocalStudyResult, ...],
    overlap_result: ArchiveOverlapQualificationResult,
    morphism_verdicts: tuple[PropertyMorphismVerdict, ...],
    control_contrast: MorphismControlContrastReceipt,
    mapped_prospective_accepted_count: int,
    mapped_prospective_map_unevaluable_count: int,
) -> MultiWorldJointAdjudicationResult:
    children = {value.child_id: value for value in child_results}
    if len(children) != 3 or set(children) != {
        plan.archive_child_id,
        plan.mapped_child_id,
        plan.generated_child_id,
    }:
        raise ValueError("joint adjudication child result roster differs from its plan")
    archive = children[plan.archive_child_id]
    mapped = children[plan.mapped_child_id]
    generated = children[plan.generated_child_id]
    h9 = bool(archive.h9_archive_law_passed and archive.archive_temporal_placebo_passed)
    mapped_pass = bool(mapped.h6_compiler_closure_passed and mapped.prospective_controller_evaluation_passed)
    generated_pass = bool(
        generated.h6_compiler_closure_passed and generated.prospective_controller_evaluation_passed
    )
    verdicts = tuple(sorted(morphism_verdicts, key=lambda value: value.verdict_id))
    primary = tuple(
        value for value in verdicts if value.property_id == PRIMARY_MORPHISM_PROPERTY_ID
    )
    contacted = tuple(value for value in primary if value.property_contacted)
    preserved = sum(
        value.disposition is PropertyMorphismDisposition.PRESERVED for value in contacted
    )
    unevaluable = sum(
        value.disposition is PropertyMorphismDisposition.UNEVALUABLE for value in primary
    )
    wilson = wilson_lower_bound_95(preserved, len(contacted))
    reasons: set[str] = set()
    if not h9:
        reasons.add("PREREQUISITE_ARCHIVE_PROPERTY_NOT_QUALIFIED")
    if (
        mapped_prospective_accepted_count < plan.minimum_accepted_states
        or Decimal(mapped_prospective_accepted_count) / Decimal(plan.mapped_prospective_attempt_count)
        < plan.minimum_accepted_fraction
    ):
        reasons.add("MAPPED_OVERLAP_INSUFFICIENT")
    if (
        mapped_prospective_map_unevaluable_count < 0
        or Decimal(mapped_prospective_map_unevaluable_count) / Decimal(plan.mapped_prospective_attempt_count)
        > plan.maximum_map_unevaluable_fraction
    ):
        reasons.add("MAPPED_UNEVALUABLE_FRACTION_EXCEEDED")
    if not overlap_result.passed:
        reasons.add("ARCHIVE_PROPERTY_NOT_QUALIFIED_ON_MAPPED_OVERLAP")
    if len(contacted) < plan.minimum_contacted_states:
        reasons.add("PRIMARY_MORPHISM_PROPERTY_NOT_CONTACTED")
    if mapped_prospective_accepted_count <= 0 or Decimal(unevaluable) / Decimal(mapped_prospective_accepted_count) > (
        plan.maximum_property_unevaluable_fraction
    ):
        reasons.add("PRIMARY_MORPHISM_UNEVALUABLE_FRACTION_EXCEEDED")
    if wilson <= plan.preservation_wilson_lower_exclusive:
        reasons.add("PRIMARY_MORPHISM_PRESERVATION_WILSON_FAILED")
    if not control_contrast.passed:
        reasons.add("MORPHISM_CONTROL_CONTRAST_FAILED")
    h10 = not reasons
    if h9 and h10 and mapped_pass and generated_pass:
        disposition = JointMultiWorldStudyDisposition.FULL_JOINT_SUCCESS
    elif h9 and h10:
        disposition = JointMultiWorldStudyDisposition.ARCHIVE_BRIDGE_WITHOUT_FULL_SIMULATOR_SUCCESS
    elif h9 and (mapped_pass or generated_pass):
        disposition = JointMultiWorldStudyDisposition.WORLD_LOCAL_WITHOUT_BRIDGE
    elif h9:
        disposition = JointMultiWorldStudyDisposition.ARCHIVE_LOCAL_ONLY
    elif mapped_pass or generated_pass:
        disposition = JointMultiWorldStudyDisposition.SIMULATOR_ONLY_MIXED
    elif any(value.disposition is PropertyMorphismDisposition.UNEVALUABLE for value in primary):
        disposition = JointMultiWorldStudyDisposition.UNEVALUABLE
    else:
        disposition = JointMultiWorldStudyDisposition.NEGATIVE
    return MultiWorldJointAdjudicationResult(
        result_id=result_id,
        plan=ObjectIdentity.from_record(plan.plan_id, plan),
        child_results=tuple(sorted(child_results, key=lambda value: value.child_id)),
        overlap_result=ObjectIdentity.from_record(overlap_result.result_id, overlap_result),
        morphism_verdicts=verdicts,
        control_contrast=ObjectIdentity.from_record(control_contrast.receipt_id, control_contrast),
        mapped_prospective_accepted_count=mapped_prospective_accepted_count,
        mapped_prospective_map_unevaluable_count=mapped_prospective_map_unevaluable_count,
        property_contacted_count=len(contacted),
        property_preserved_count=preserved,
        property_unevaluable_count=unevaluable,
        property_preservation_wilson_lower=wilson,
        h9_passed=h9,
        h10_passed=h10,
        mapped_simulator_passed=mapped_pass,
        generated_simulator_passed=generated_pass,
        disposition=disposition,
        reason_codes=tuple(sorted(reasons)),
        terminal=True,
    )


__all__ = [
    'ArchiveOverlapObservation',
    'ArchiveOverlapQualificationResult',
    "ArchiveResponseClass",
    'ArchiveTwoActionClassPair',
    'MultiWorldJointAdjudicationResult',
    'JointMultiWorldStudyDisposition',
    "MappingDisposition",
    'MorphismControlContrastReceipt',
    'MorphismControlContrast',
    'MorphismControlStateObservation',
    "PropertyMorphismDisposition",
    'PropertyMorphismMemberVerdict',
    'PropertyMorphismStateInput',
    'PropertyMorphismVerdict',
    'ToraxMemberTwoActionClassPair',
    "TwoActionMorphismAction",
    'WorldLocalStudyResult',
    'adjudicate_joint_multi_world_study',
    "evaluate_archive_overlap_qualification",
    "evaluate_morphism_negative_controls",
    "evaluate_property_morphism",
    "wilson_lower_bound_95",
]
