"Registered nested controller-use reduction with sealed correctness and HOLD efficacy."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from threading import RLock
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.control import ScientificCommitmentKind, OperationalDeliveryState
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    require_unique_ids,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy, ImplementationBinding, ImplementationRole, UtilityDirection
from empirical_lawhood.planning.nested_controller_evaluation import CommonStartControllerEvaluationPlan, CoupledRealizationControllerEvaluationPlan, PreparedInterfaceEvaluationSlice, prepared_interface_evaluation_slice, PreparedInterfaceReducerRegistration, PreparedFutureSlot, PreparedFutureRole, NestedEvaluationCellLocator, NestedMemberReducer, RepeatedDeliveryReducerRegistration, NestedRepeatReducer, RepeatedDeliveryControllerEvaluationPlan, ActionAwareControllerEvaluationPlan, NestedReplicateSpec, ProspectiveEvaluationPrecommitment
from empirical_lawhood.planning.finite_response_geometry import FiniteTaskFunctionalSpec, FiniteResponseSet, FiniteResponseCoordinate, FiniteSetDisposition
from empirical_lawhood.runtime.controller_compiler import CompiledDeliveryControllerStudy, FINITE_COMPILED_COMMITMENT_SCHEMAS, CompiledAdmissionControllerStudy, HoldFibreDisposition, ProspectiveBindingDisposition
from empirical_lawhood.runtime.controller_evaluation import (
    ControllerUseResult,
    OutcomePredicateEvaluation,
    evaluate_controller_outcome_predicate,
)
from empirical_lawhood.runtime.finite_chart_reference import ReferenceClassMembershipReceipt, ReferenceDispositionClass, ReferenceMembershipStatus, evaluate_reference_class_membership
from empirical_lawhood.runtime.controller_runtime import CommitmentDisposition, DeliveryControllerDecisionCommitment, DeliveryControllerTickReceipt, AdmissionControllerTickReceipt, ExactActionDeliveryTrace, StageAwareActionDeliveryTrace


from empirical_lawhood.planning.trajectory_controller_evaluation import TrajectoryControllerEvaluationPlan, TrajectoryReducerRegistration
from empirical_lawhood.runtime.controller_evaluation_trajectory import RevealedTrajectoryBundle, TrajectoryOwnerRecordReader, TrajectoryUnitEvaluation, TrajectoryCohortEvaluation, evaluate_trajectory
from empirical_lawhood.planning.bounded_sequence_evaluation import BoundedSequenceEvaluationPlan
from empirical_lawhood.runtime.controller_evaluation_sequence import BoundedSequenceUnitEvaluation, RevealedBoundedSequence, evaluate_sequence


MAX_NESTED_CONTROLLER_EVALUATION_BUNDLE_BYTES = 64 * 1024 * 1024


class ReferenceSafetyDisposition(StrEnum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNEVALUABLE = "UNEVALUABLE"


class DispositionCorrectness(StrEnum):
    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    UNSAFE = "UNSAFE"
    UNEVALUABLE = "UNEVALUABLE"


class NestedRepeatDisposition(StrEnum):
    ACTIVE_EVALUABLE = "ACTIVE_EVALUABLE"
    QUALIFIED_HOLD = "QUALIFIED_HOLD"
    NONATTEMPT = "NONATTEMPT"
    REFERENCE_INCORRECT = "REFERENCE_INCORRECT"
    UNSAFE = "UNSAFE"
    DELIVERY_INVALID = "DELIVERY_INVALID"
    TECHNICAL_INVALID = "TECHNICAL_INVALID"
    UNEVALUABLE = "UNEVALUABLE"


class NestedControllerUnitDisposition(StrEnum):
    ACTIVE_EVALUABLE = "ACTIVE_EVALUABLE"
    QUALIFIED_HOLD = "QUALIFIED_HOLD"
    NONATTEMPT = "NONATTEMPT"
    REFERENCE_INCORRECT = "REFERENCE_INCORRECT"
    UNSAFE = "UNSAFE"
    DELIVERY_INVALID = "DELIVERY_INVALID"
    TECHNICAL_INVALID = "TECHNICAL_INVALID"
    PARTIAL_HETEROGENEOUS = "PARTIAL_HETEROGENEOUS"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class SealedNestedReferenceLocator(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sealed-nested-reference-locator'

    locator_id: str
    replicate_id: str
    finite_chart_reference_plan: ObjectIdentity
    artifact: ArtifactIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.locator_id, field_name="locator_id")
        validate_stable_id(self.replicate_id, field_name="replicate_id")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("nested reference locator must remain sealed")


@dataclass(frozen=True, slots=True)
class SealedNestedOutcomeLocator(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sealed-nested-outcome-locator'

    locator_id: str
    replicate_id: str
    branch_id: str
    model_member_id: str
    artifact: ArtifactIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("locator_id", self.locator_id),
            ("replicate_id", self.replicate_id),
            ("branch_id", self.branch_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("nested outcome locator must remain sealed")


@dataclass(frozen=True, slots=True)
class SealedNestedReplicate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sealed-nested-replicate'

    replicate: NestedReplicateSpec
    common_initial_state_fingerprint: str
    common_prefix_fingerprint: str
    tick: AdmissionControllerTickReceipt
    expected_controller_word: OccurrenceActionWord | None
    reference_locator: SealedNestedReferenceLocator
    outcome_locators: tuple[SealedNestedOutcomeLocator, ...]

    def __post_init__(self) -> None:
        validate_sha256(
            self.common_initial_state_fingerprint,
            field_name="common_initial_state_fingerprint",
        )
        validate_sha256(
            self.common_prefix_fingerprint,
            field_name="common_prefix_fingerprint",
        )
        if (
            self.tick.commitment.observation.independent_unit_id
            != self.replicate.independent_unit_id
        ):
            raise ValueError("nested tick rewrites its physical independent unit")
        expected_word = (
            self.tick.commitment.action_binding.action_word
            if self.tick.commitment.action_binding is not None
            else None
        )
        if self.expected_controller_word != expected_word:
            raise ValueError("nested replicate expected word differs from its commitment")
        if (
            self.reference_locator.replicate_id != self.replicate.replicate_id
            or self.reference_locator.finite_chart_reference_plan.object_schema
            != 'empirical-lawhood/planning/finite-chart-reference-plan'
        ):
            raise ValueError("nested replicate reference locator differs")
        require_sorted_unique_ids(
            self.outcome_locators,
            attribute="locator_id",
            field_name="outcome_locators",
        )
        if any(
            value.replicate_id != self.replicate.replicate_id for value in self.outcome_locators
        ):
            raise ValueError("nested outcome locator uses another replicate")


@dataclass(frozen=True, slots=True)
class SealedRepeatedDeliveryControllerBundle(CanonicalRecord):
    """Complete sealed nested campaign; outcomes remain inaccessible."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sealed-repeated-delivery-controller-bundle'

    bundle_id: str
    evaluation_plan: RepeatedDeliveryControllerEvaluationPlan
    compiled_study: ObjectIdentity
    cohort_id: str
    replicates: tuple[SealedNestedReplicate, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        validate_stable_id(self.cohort_id, field_name="cohort_id")
        if self.compiled_study.object_schema != CompiledAdmissionControllerStudy.SCHEMA:
            raise ValueError("nested bundle requires a compiled admission controller study")
        require_sorted_unique_strings(
            tuple(value.replicate.replicate_id for value in self.replicates),
            field_name="replicates",
            allow_empty=False,
        )
        expected_replicates = {value.replicate_id for value in self.evaluation_plan.replicates}
        observed_replicates = {value.replicate.replicate_id for value in self.replicates}
        if observed_replicates != expected_replicates or len(self.replicates) != len(
            expected_replicates
        ):
            raise ValueError("sealed nested replicate roster differs from plan")
        branches = {
            self.evaluation_plan.controller_branch_id,
            self.evaluation_plan.reference_branch_id,
        }
        if self.evaluation_plan.hold_baseline_branch_id is not None:
            branches.add(self.evaluation_plan.hold_baseline_branch_id)
        tick_ids: set[str] = set()
        artifact_ids: set[str] = set()
        for sealed in self.replicates:
            planned = next(
                value
                for value in self.evaluation_plan.replicates
                if value.replicate_id == sealed.replicate.replicate_id
            )
            if (
                sealed.replicate != planned
                or sealed.tick.compiled_study != self.compiled_study
                or sealed.reference_locator.finite_chart_reference_plan
                != self.evaluation_plan.finite_chart_reference_plan
            ):
                raise ValueError("sealed nested replicate changes plan/compiled identities")
            if sealed.tick.tick_id in tick_ids:
                raise ValueError("nested repetitions must use distinct controller ticks")
            tick_ids.add(sealed.tick.tick_id)
            expected = {
                (branch, member)
                for branch in branches
                for member in self.evaluation_plan.model_member_ids
            }
            observed = {
                (value.branch_id, value.model_member_id) for value in sealed.outcome_locators
            }
            if observed != expected or len(sealed.outcome_locators) != len(expected):
                raise ValueError("sealed nested branch/member locator grid is incomplete")
            local_artifacts = {
                sealed.reference_locator.artifact.artifact_id,
                *(value.artifact.artifact_id for value in sealed.outcome_locators),
            }
            if artifact_ids & local_artifacts:
                raise ValueError("sealed nested locators reuse outcome artifact identity")
            artifact_ids.update(local_artifacts)


@dataclass(frozen=True, slots=True)
class RevealedReplicateDispositionReference(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-replicate-disposition-reference'

    reference_id: str
    sealed_locator: ObjectIdentity
    replicate_id: str
    safety: ReferenceSafetyDisposition
    expected_commitment: CommitmentDisposition | None
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.reference_id, field_name="reference_id")
        validate_stable_id(self.replicate_id, field_name="replicate_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.outcome_access not in {
            OutcomeAccess.EVALUATOR_REVEAL,
            OutcomeAccess.EVALUATION_REVEALED,
        }:
            raise ValueError("nested disposition reference is not revealed")
        if self.safety is ReferenceSafetyDisposition.SAFE:
            if self.expected_commitment is None or self.reason_codes:
                raise ValueError("safe disposition reference requires one expected commitment")
        elif self.expected_commitment is not None or not self.reason_codes:
            raise ValueError("unsafe/unevaluable disposition reference is malformed")


@dataclass(frozen=True, slots=True)
class RevealedNestedBranchOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-nested-branch-outcome'

    outcome_id: str
    sealed_locator: ObjectIdentity
    replicate_id: str
    branch_id: str
    model_member_id: str
    initial_state_fingerprint: str
    prefix_fingerprint: str
    action_word: OccurrenceActionWord | None
    delivery_trace: ExactActionDeliveryTrace
    raw_outcomes: tuple[NamedDecimal, ...]
    technical_reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("outcome_id", self.outcome_id),
            ("replicate_id", self.replicate_id),
            ("branch_id", self.branch_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(self.initial_state_fingerprint, field_name="initial_state_fingerprint")
        validate_sha256(self.prefix_fingerprint, field_name="prefix_fingerprint")
        require_sorted_unique_ids(
            self.raw_outcomes,
            attribute="value_id",
            field_name="raw_outcomes",
        )
        require_sorted_unique_strings(
            self.technical_reason_codes,
            field_name="technical_reason_codes",
        )
        if self.outcome_access not in {
            OutcomeAccess.EVALUATOR_REVEAL,
            OutcomeAccess.EVALUATION_REVEALED,
        }:
            raise ValueError("nested branch outcome is not evaluator-revealed")
        if self.action_word != self.delivery_trace.expected_action_word:
            raise ValueError("nested branch action differs from its fresh delivery trace")
        if self.delivery_trace.commitment_kind is ScientificCommitmentKind.NONATTEMPT:
            if self.action_word is not None or self.raw_outcomes:
                raise ValueError("nested NONATTEMPT fabricates action/outcome evidence")


@dataclass(frozen=True, slots=True)
class RevealedRepeatedDeliveryControllerBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-repeated-delivery-controller-bundle'

    reveal_id: str
    sealed_bundle: SealedRepeatedDeliveryControllerBundle
    reveal_authorization: ObjectIdentity
    disposition_references: tuple[RevealedReplicateDispositionReference, ...]
    outcomes: tuple[RevealedNestedBranchOutcome, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.reveal_id, field_name="reveal_id")
        require_sorted_unique_ids(
            self.disposition_references,
            attribute="reference_id",
            field_name="disposition_references",
        )
        require_sorted_unique_ids(self.outcomes, attribute="outcome_id", field_name="outcomes")
        sealed_by_replicate = {
            value.replicate.replicate_id: value for value in self.sealed_bundle.replicates
        }
        if {value.replicate_id for value in self.disposition_references} != set(
            sealed_by_replicate
        ) or len(self.disposition_references) != len(sealed_by_replicate):
            raise ValueError("revealed disposition-reference roster differs from sealed plan")
        for reference in self.disposition_references:
            locator = sealed_by_replicate[reference.replicate_id].reference_locator
            if reference.sealed_locator != ObjectIdentity.from_record(
                locator.locator_id,
                locator,
            ):
                raise ValueError("revealed disposition binds another sealed reference")
        locator_map = {
            (locator.replicate_id, locator.branch_id, locator.model_member_id): locator
            for sealed in self.sealed_bundle.replicates
            for locator in sealed.outcome_locators
        }
        observed = {
            (value.replicate_id, value.branch_id, value.model_member_id) for value in self.outcomes
        }
        if observed != set(locator_map) or len(self.outcomes) != len(locator_map):
            raise ValueError("revealed nested outcome grid differs from sealed locators")
        for outcome in self.outcomes:
            outcome_locator = locator_map[
                (outcome.replicate_id, outcome.branch_id, outcome.model_member_id)
            ]
            if outcome.sealed_locator != ObjectIdentity.from_record(
                outcome_locator.locator_id,
                outcome_locator,
            ):
                raise ValueError("revealed nested outcome binds another sealed locator")
        plan = self.sealed_bundle.evaluation_plan
        traces: dict[tuple[str, str], ExactActionDeliveryTrace] = {}
        trace_ids: dict[str, tuple[str, str]] = {}
        for outcome in self.outcomes:
            key = (outcome.replicate_id, outcome.branch_id)
            prior = traces.setdefault(key, outcome.delivery_trace)
            if prior != outcome.delivery_trace:
                raise ValueError("one physical branch delivery has several member traces")
            prior_key = trace_ids.setdefault(outcome.delivery_trace.trace_id, key)
            if prior_key != key:
                raise ValueError("nested repetitions/branches reuse a delivery trace")
            sealed = sealed_by_replicate[outcome.replicate_id]
            expected_word = (
                sealed.expected_controller_word
                if outcome.branch_id == plan.controller_branch_id
                else (
                    plan.reference_action_word
                    if outcome.branch_id == plan.reference_branch_id
                    else None
                )
            )
            if outcome.branch_id == plan.hold_baseline_branch_id:
                expected_word = None
            if (
                outcome.branch_id != plan.hold_baseline_branch_id
                and outcome.action_word != expected_word
            ):
                raise ValueError("revealed nested branch changes its frozen expected action word")
            if (
                outcome.branch_id == plan.controller_branch_id
                and outcome.delivery_trace.trace_id == sealed.tick.delivery_trace.trace_id
            ):
                raise ValueError("nested controller use repetition reuses the commitment-tick delivery trace")
            if (
                outcome.branch_id == plan.controller_branch_id
                and outcome.delivery_trace.commitment
                != ObjectIdentity.from_record(
                    sealed.tick.commitment.commitment_id,
                    sealed.tick.commitment,
                )
            ):
                raise ValueError("fresh controller delivery binds another commitment")


class NestedMemberRepeatDisposition(StrEnum):
    EVALUABLE = "EVALUABLE"
    QUALIFIED_HOLD = "QUALIFIED_HOLD"
    NONATTEMPT = "NONATTEMPT"
    UNSAFE = "UNSAFE"
    DELIVERY_INVALID = "DELIVERY_INVALID"
    TECHNICAL_INVALID = "TECHNICAL_INVALID"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class NestedMemberRepeatEffect(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/nested-member-repeat-effect'

    effect_id: str
    replicate_id: str
    model_member_id: str
    disposition: NestedMemberRepeatDisposition
    effect: NamedDecimal | None
    predicate_evaluations: tuple[OutcomePredicateEvaluation, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("effect_id", self.effect_id),
            ("replicate_id", self.replicate_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.predicate_evaluations,
            attribute="evaluation_id",
            field_name="predicate_evaluations",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is NestedMemberRepeatDisposition.EVALUABLE:
            if self.effect is None or self.reason_codes:
                raise ValueError("evaluable nested member/repeat requires one clean effect")
        elif self.effect is not None or not self.reason_codes:
            raise ValueError("nonevaluable nested member/repeat cannot carry an effect")


@dataclass(frozen=True, slots=True)
class NestedRepeatEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/nested-repeat-evaluation'

    repeat_evaluation_id: str
    replicate: NestedReplicateSpec
    observed_commitment: CommitmentDisposition
    expected_commitment: CommitmentDisposition | None
    correctness: DispositionCorrectness
    disposition: NestedRepeatDisposition
    member_effects: tuple[NestedMemberRepeatEffect, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.repeat_evaluation_id, field_name="repeat_evaluation_id")
        require_sorted_unique_ids(
            self.member_effects,
            attribute="model_member_id",
            field_name="member_effects",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.correctness is DispositionCorrectness.CORRECT:
            if self.expected_commitment != self.observed_commitment:
                raise ValueError("correct nested disposition comparison differs")
        elif self.correctness is DispositionCorrectness.INCORRECT:
            if (
                self.expected_commitment is None
                or self.expected_commitment == self.observed_commitment
            ):
                raise ValueError("incorrect nested disposition comparison is not a mismatch")
        elif self.expected_commitment is not None:
            raise ValueError("unsafe/unevaluable reference cannot supply expected commitment")
        if self.disposition is NestedRepeatDisposition.ACTIVE_EVALUABLE:
            if (
                not self.member_effects
                or self.reason_codes
                or any(
                    value.disposition is not NestedMemberRepeatDisposition.EVALUABLE
                    for value in self.member_effects
                )
            ):
                raise ValueError("active nested repeat lacks a complete clean member effect grid")
        elif not self.reason_codes:
            raise ValueError("nonactive nested repeat requires reasons")


@dataclass(frozen=True, slots=True)
class RepeatedDeliveryMemberEffect(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/repeated-delivery-member-effect'

    model_member_id: str
    repeat_count: int
    effect: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.model_member_id, field_name="model_member_id")
        if self.repeat_count < 1:
            raise ValueError("nested member effect requires at least one repeat")


@dataclass(frozen=True, slots=True)
class RepeatedDeliveryControllerUnitEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/repeated-delivery-controller-unit-evaluation'

    unit_evaluation_id: str
    independent_unit_id: str
    task_id: str
    preparation_unit_id: str
    evaluation_plan: ObjectIdentity
    compiled_study: ObjectIdentity
    bundle: ObjectIdentity
    reducer_registration: ObjectIdentity
    disposition: NestedControllerUnitDisposition
    repeat_evaluations: tuple[NestedRepeatEvaluation, ...]
    member_effects: tuple[RepeatedDeliveryMemberEffect, ...]
    robust_effect: NamedDecimal | None
    independent_unit_count: int
    nested_repeat_count: int
    nested_member_count: int
    correct_reference_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("unit_evaluation_id", self.unit_evaluation_id),
            ("independent_unit_id", self.independent_unit_id),
            ("task_id", self.task_id),
            ("preparation_unit_id", self.preparation_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            tuple(value.replicate.replicate_id for value in self.repeat_evaluations),
            field_name="repeat_evaluations",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.member_effects,
            attribute="model_member_id",
            field_name="member_effects",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.independent_unit_count != 1:
            raise ValueError("nested unit evaluation must count exactly one physical unit")
        if self.nested_repeat_count != len(self.repeat_evaluations):
            raise ValueError("nested repeat count differs from repeat-evaluation roster")
        if self.nested_member_count < 1:
            raise ValueError("nested unit evaluation requires a declared member roster")
        if not 0 <= self.correct_reference_count <= self.nested_repeat_count:
            raise ValueError("nested correctness count is outside the repeat roster")
        if self.disposition is NestedControllerUnitDisposition.ACTIVE_EVALUABLE:
            if (
                not self.member_effects
                or self.robust_effect is None
                or self.reason_codes
                or self.robust_effect.value
                != min(value.effect.value for value in self.member_effects)
            ):
                raise ValueError("active nested unit is not a complete memberwise reduction")
        elif self.member_effects or self.robust_effect is not None or not self.reason_codes:
            raise ValueError("nonactive nested unit cannot carry an efficacy estimate")


@dataclass(frozen=True, slots=True)
class NestedUnitDispositionCount(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/nested-unit-disposition-count'

    disposition: NestedControllerUnitDisposition
    count: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("nested unit disposition count cannot be negative")


@dataclass(frozen=True, slots=True)
class RepeatedDeliveryControllerCohortAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/repeated-delivery-controller-cohort-adjudication'

    adjudication_id: str
    evaluation_plan: ObjectIdentity
    compiled_study: ObjectIdentity
    bundle: ObjectIdentity
    reducer_registration: ObjectIdentity
    units: tuple[RepeatedDeliveryControllerUnitEvaluation, ...]
    disposition_counts: tuple[NestedUnitDispositionCount, ...]
    rostered_independent_unit_count: int
    effective_independent_unit_count: int
    nested_repeat_count: int
    nested_member_count: int
    mean_effect: NamedDecimal | None
    one_sided_lower_bound: NamedDecimal | None
    active_coverage: Decimal
    reference_correctness_coverage: Decimal
    result: ControllerUseResult
    claim_ceiling: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(self.units, attribute="independent_unit_id", field_name="units")
        if tuple(value.disposition for value in self.disposition_counts) != tuple(
            NestedControllerUnitDisposition
        ):
            raise ValueError("nested cohort disposition count roster is incomplete")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.rostered_independent_unit_count != len(self.units):
            raise ValueError("nested cohort roster counts something other than physical units")
        if sum(value.count for value in self.disposition_counts) != len(self.units):
            raise ValueError("nested cohort disposition counts do not partition physical units")
        expected_effective = sum(
            value.disposition is NestedControllerUnitDisposition.ACTIVE_EVALUABLE
            for value in self.units
        )
        if self.effective_independent_unit_count != expected_effective:
            raise ValueError("nested members/repeats inflated effective independent-unit count")
        if self.nested_repeat_count != sum(value.nested_repeat_count for value in self.units):
            raise ValueError("nested cohort repeat audit count differs")
        if self.nested_member_count < 1:
            raise ValueError("nested cohort requires members")
        for name, amount in (
            ("active_coverage", self.active_coverage),
            ("reference_correctness_coverage", self.reference_correctness_coverage),
        ):
            validate_decimal(amount, field_name=name, minimum=Decimal(0))
            if amount > 1:
                raise ValueError(f"{name} cannot exceed one")
        if self.effective_independent_unit_count:
            if self.mean_effect is None or self.one_sided_lower_bound is None:
                raise ValueError("evaluable nested cohort requires physical-unit summaries")
        elif self.mean_effect is not None or self.one_sided_lower_bound is not None:
            raise ValueError("nonevaluable nested cohort cannot carry efficacy summaries")


def _branches(plan: RepeatedDeliveryControllerEvaluationPlan) -> tuple[str, ...]:
    values = [plan.controller_branch_id, plan.reference_branch_id]
    if plan.hold_baseline_branch_id is not None:
        values.append(plan.hold_baseline_branch_id)
    return tuple(sorted(values))


class NestedControllerUseEvaluator:
    "Outcome-visible evaluation record; it cannot select, compile, tick or deliver."

    def __init__(
        self,
        implementation_binding: ImplementationBinding,
        reducer_registration: RepeatedDeliveryReducerRegistration
        | PreparedInterfaceReducerRegistration
        | TrajectoryReducerRegistration,
    ) -> None:
        if implementation_binding.role is not ImplementationRole.OUTCOME_EVALUATOR:
            raise ValueError("nested controller use evaluator uses another implementation role")
        if (
            reducer_registration.capability_key != implementation_binding.reference.capability_key
            or reducer_registration.capability_version
            != implementation_binding.reference.capability_version
            or reducer_registration.implementation_sha256
            != implementation_binding.implementation_sha256
            or reducer_registration.config_sha256 != implementation_binding.config_sha256
        ):
            raise ValueError("nested reducer registration differs from evaluator binding")
        self.implementation_binding = implementation_binding
        self.reducer_registration = reducer_registration

    def evaluate_sequence_unit(
        self, *, plan: BoundedSequenceEvaluationPlan, bundle: RevealedBoundedSequence
    ) -> BoundedSequenceUnitEvaluation:
        if (
            plan.reducer != self.reducer_registration
            or plan.evaluator != self.implementation_binding
        ):
            raise ValueError("sequence evaluation requires its exact registered prepared owner")
        return evaluate_sequence(plan, bundle, self)

    def evaluate_trajectory_unit(
        self,
        *,
        plan: TrajectoryControllerEvaluationPlan,
        bundle: RevealedTrajectoryBundle,
        reader: TrajectoryOwnerRecordReader,
    ) -> TrajectoryUnitEvaluation:
        if (
            not isinstance(self.reducer_registration, TrajectoryReducerRegistration)
            or plan.evaluator != self.implementation_binding
        ):
            raise ValueError("trajectory evaluation requires its exact registered owner")
        return evaluate_trajectory(plan, bundle, reader)

    def reduce_trajectory_cohort(
        self,
        *,
        plan: TrajectoryControllerEvaluationPlan,
        units: tuple[TrajectoryUnitEvaluation, ...],
    ) -> TrajectoryCohortEvaluation:
        if (
            not isinstance(self.reducer_registration, TrajectoryReducerRegistration)
            or plan.evaluator != self.implementation_binding
        ):
            raise ValueError("trajectory cohort requires its exact registered owner")
        ordered = tuple(sorted(units, key=lambda u: u.root))
        ratios = tuple(u.completion_ratio for u in ordered)
        mean = (
            sum((r for r in ratios if r is not None), Decimal(0)) / len(ratios)
            if ratios and all(r is not None for r in ratios)
            else None
        )
        return TrajectoryCohortEvaluation(
            plan,
            ordered,
            len(ordered),
            sum(u.A for u in ordered),
            sum(u.J for u in ordered),
            sum(u.C for u in ordered),
            mean,
            mean is not None and mean <= plan.panel_mean_completion_ratio,
        )

    def reduce_prepared_cohort(
        self,
        *,
        plan: CommonStartControllerEvaluationPlan,
        units: tuple[PreparedPolicyUnitEvaluation, ...],
    ) -> PreparedControllerCohortEvaluation:
        if (
            plan.reducer != self.reducer_registration
            or plan.evaluator_boundary.outcome_evaluator_binding_id
            != self.implementation_binding.binding_id
        ):
            raise ValueError("prepared cohort reducer differs from its pre-parent frozen design")
        ordered = tuple(sorted(units, key=lambda unit: unit.evaluation_id))
        return PreparedControllerCohortEvaluation(
            evaluation_id=f"prepared-cohort.{plan.evaluation_plan_id}",
            plan=plan,
            evaluator=self.implementation_binding,
            units=ordered,
            censuses=_prepared_censuses(ordered),
            independent_root_count=len(plan.roots),
        )

    @staticmethod
    def _prepared_intersection(values: tuple[GateStatus, ...]) -> GateStatus:
        if any(value is GateStatus.FAIL for value in values):
            return GateStatus.FAIL
        if not values or any(value is not GateStatus.PASS for value in values):
            return GateStatus.UNEVALUABLE
        return GateStatus.PASS

    @staticmethod
    def _prepared_target_status(
        task: FiniteTaskFunctionalSpec,
        outcome: RevealedPreparedFuture,
    ) -> GateStatus:
        observed = {r.coordinate.coordinate_id: r for r in outcome.readouts}
        required = tuple(i.coordinate for i in task.boxes[0].intervals)
        if any(
            c.coordinate_id not in observed or observed[c.coordinate_id].coordinate != c
            for c in required
        ):
            return GateStatus.UNEVALUABLE
        return (
            GateStatus.PASS
            if any(
                all(
                    i.lower <= observed[i.coordinate.coordinate_id].value <= i.upper
                    for i in box.intervals
                )
                for box in task.boxes
            )
            else GateStatus.FAIL
        )

    @staticmethod
    def _prepared_numerical_status(
        plan: PreparedInterfaceEvaluationSlice,
        outcomes: tuple[RevealedPreparedFuture, ...],
    ) -> GateStatus:
        if tuple(sorted(o.view_id for o in outcomes)) != plan.numerical_view_ids:
            return GateStatus.UNEVALUABLE
        by_view = [{r.coordinate.coordinate_id: r for r in o.readouts} for o in outcomes]
        if not by_view or not by_view[0] or any(set(v) != set(by_view[0]) for v in by_view):
            return GateStatus.UNEVALUABLE
        tolerances = {t.coordinate.coordinate_id: t for t in plan.readout_tolerances}
        for coordinate_id in by_view[0]:
            tolerance = tolerances.get(coordinate_id)
            values = tuple(view[coordinate_id] for view in by_view)
            if tolerance is None or any(r.coordinate != tolerance.coordinate for r in values):
                return GateStatus.UNEVALUABLE
            if (
                max(r.value for r in values) - min(r.value for r in values)
                > tolerance.numerical_tolerance
                or max(r.numerical_floor for r in values) > tolerance.numerical_tolerance
            ):
                return GateStatus.FAIL
        return GateStatus.PASS

    def _prepared_preservation_status(
        self,
        plan: PreparedInterfaceEvaluationSlice,
        raw: tuple[NamedDecimal, ...],
        predicate_ids: tuple[str, ...],
    ) -> GateStatus:
        values = {value.value_id: value for value in raw}
        predicates = tuple(
            predicate
            for predicate in plan.evaluator_boundary.outcome_predicates
            if predicate.predicate_id in predicate_ids
        )
        required_quantities = {predicate.quantity_id for predicate in predicates}
        if len(predicates) != len(predicate_ids) or set(values) != required_quantities:
            return GateStatus.UNEVALUABLE
        return self._prepared_intersection(
            tuple(
                evaluate_controller_outcome_predicate(
                    evaluation_id=f"prepared-preservation.{p.predicate_id}", predicate=p, raw=values
                ).status
                for p in predicates
            )
        )

    def _prepared_matched_hold_status(
        self,
        plan: PreparedInterfaceEvaluationSlice,
        slot: PreparedFutureSlot,
        outcomes: tuple[RevealedPreparedFuture, ...],
        compiled: CompiledDeliveryControllerStudy | None,
    ) -> GateStatus:
        """Authenticate the task's evaluator-only same-innovation HOLD reference."""

        if tuple(sorted(outcome.view_id for outcome in outcomes)) != plan.numerical_view_ids:
            return GateStatus.UNEVALUABLE
        expected_word = ObjectIdentity.from_record(
            plan.matched_hold_word.word_id, plan.matched_hold_word
        )
        expected_commitment = ObjectIdentity.from_record(slot.slot_id, slot)
        delivery_statuses: list[GateStatus] = []
        for outcome in outcomes:
            trace = outcome.delivery_trace
            if trace is None:
                delivery_statuses.append(GateStatus.UNEVALUABLE)
                continue
            if (
                trace.commitment != expected_commitment
                or trace.commitment_kind is not ScientificCommitmentKind.MEASURED_HOLD
                or trace.expected_action_word is None
                or ObjectIdentity.from_record(
                    trace.expected_action_word.word_id, trace.expected_action_word
                )
                != expected_word
            ):
                raise ValueError("prepared matched HOLD substitutes its frozen reference word")
            if compiled is not None and trace.delivery != compiled.implementation(
                ImplementationRole.DELIVERY
            ):
                raise ValueError("prepared matched HOLD substitutes the delivery implementation")
            if isinstance(trace, StageAwareActionDeliveryTrace) and (
                compiled is None
                or trace.delivery_equivalence != compiled.study.delivery_equivalence
            ):
                raise ValueError("prepared matched HOLD changes the delivery tolerances")
            delivery_statuses.append(
                GateStatus.PASS
                if trace.operational_state is OperationalDeliveryState.DELIVERED
                else GateStatus.FAIL
            )
        return self._prepared_intersection(
            (
                self._prepared_numerical_status(plan, outcomes),
                *delivery_statuses,
            )
        )

    def evaluate_prepared_unit(
        self,
        *,
        revealed: RevealedPreparedPolicyBundle,
        compiled: CompiledDeliveryControllerStudy | None,
    ) -> PreparedPolicyUnitEvaluation:
        """Reduce exact native points and paired views; enumerate no counterfactual actions."""
        sealed = revealed.sealed
        design = sealed.design
        if isinstance(sealed, SealedPreparedForecastPolicyBundle) and compiled is not None:
            sealed.forecast_parent.validate_binding(design, compiled)
        plan = design.evaluation_plan
        plan_id = plan.identity
        if (
            design.evaluator_binding != self.implementation_binding
            or plan.reducer != self.reducer_registration
        ):
            raise ValueError(
                "prepared controller use evaluator differs from the pre-parent frozen registration"
            )
        if sealed.instance is None:
            if compiled is not None:
                raise ValueError(
                    "prepared controller use cannot invent a compiled child for an unbound episode"
                )
        else:
            if (
                compiled is None
                or sealed.instance.compiled_study
                != ObjectIdentity.from_record(compiled.compiled_study_id, compiled)
                or compiled.study.prospective_evaluation != plan_id
                or compiled.prospective_evaluation_binding.disposition is not ProspectiveBindingDisposition.BOUND
                or compiled.implementation(ImplementationRole.OUTCOME_EVALUATOR)
                != self.implementation_binding
            ):
                raise ValueError("prepared controller use substitutes its frozen compiled child or evaluator")
        root = next(r for r in plan.roots if r.root_id == design.root_id)
        slots = tuple(
            f for f in plan.futures if f.root_id == root.root_id and f.policy_id == design.policy_id
        )
        task_slot = next(f for f in slots if f.role is PreparedFutureRole.COMMITTED_TASK)
        matched_hold_slot = next(f for f in slots if f.role is PreparedFutureRole.MATCHED_HOLD)
        tasks = tuple(o for o in revealed.outcomes if o.slot_id == task_slot.slot_id)
        matched_holds = tuple(
            o for o in revealed.outcomes if o.slot_id == matched_hold_slot.slot_id
        )
        handoff = sealed.parent_return.disposition is PreparedParentDisposition.HANDOFF_AVAILABLE
        early_decision = isinstance(sealed, SealedPreparedForecastPolicyBundle)
        commitment = (
            sealed.forecast_parent.decision
            if isinstance(sealed, SealedPreparedForecastPolicyBundle)
            else sealed.commitment
        )
        admitted = (
            (early_decision or handoff)
            and commitment is not None
            and commitment.disposition is not CommitmentDisposition.NONATTEMPT
        )
        target = (
            self._prepared_intersection(
                tuple(self._prepared_target_status(sealed.task, o) for o in tasks)
            )
            if sealed.task is not None
            else GateStatus.UNEVALUABLE
        )
        task_delivery: list[GateStatus] = []
        for outcome in tasks:
            trace = outcome.delivery_trace
            if trace is None or commitment is None or commitment.action_binding is None:
                task_delivery.append(GateStatus.UNEVALUABLE)
                continue
            if (
                trace.commitment != ObjectIdentity.from_record(commitment.commitment_id, commitment)
                or trace.expected_action_word != commitment.action_binding.action_word
            ):
                raise ValueError("prepared native task substitutes the committed word or decision")
            if compiled is not None and trace.delivery != compiled.implementation(
                ImplementationRole.DELIVERY
            ):
                raise ValueError("prepared native task substitutes the delivery implementation")
            if isinstance(trace, StageAwareActionDeliveryTrace):
                if (
                    compiled is None
                    or trace.delivery_equivalence != compiled.study.delivery_equivalence
                ):
                    raise ValueError("prepared task changes the frozen stage delivery tolerances")
            task_delivery.append(
                GateStatus.PASS
                if trace.operational_state is OperationalDeliveryState.DELIVERED
                else GateStatus.FAIL
            )
        delivery = self._prepared_intersection(tuple(task_delivery))
        parent_preservation = self._prepared_preservation_status(
            plan,
            revealed.parent_preservation_outcomes,
            plan.parent_preservation_predicate_ids,
        )
        matched_hold_status = self._prepared_matched_hold_status(
            plan, matched_hold_slot, matched_holds, compiled
        )
        preservation = self._prepared_intersection(
            (
                parent_preservation,
                matched_hold_status,
                *(
                    self._prepared_preservation_status(
                        plan,
                        o.preservation_outcomes,
                        plan.future_preservation_predicate_ids,
                    )
                    for o in tasks
                ),
            )
        )
        numerics = self._prepared_numerical_status(plan, tasks)
        adequate: list[GateStatus] = []
        covered: list[GateStatus] = []
        sharp: list[GateStatus] = []
        tolerances = {t.coordinate.coordinate_id: t for t in plan.readout_tolerances}
        probes = {p.slot.slot_id: p for p in sealed.probes}
        for slot in slots:
            if slot.role is not PreparedFutureRole.AUDIT_PROBE:
                continue
            probe = probes.get(slot.slot_id)
            outcomes = tuple(o for o in revealed.outcomes if o.slot_id == slot.slot_id)
            if probe is None or probe.forecast.disposition is not FiniteSetDisposition.AVAILABLE:
                adequate.append(GateStatus.UNEVALUABLE)
                covered.append(GateStatus.UNEVALUABLE)
                sharp.append(GateStatus.UNEVALUABLE)
                continue
            forecast = probe.forecast
            if compiled is None:
                raise ValueError("prepared probe has no compiled frozen law package")
            instance = compiled.study.instance_binding
            calibration = ObjectIdentity.from_record(
                forecast.joint_calibration.calibration_id,
                forecast.joint_calibration,
            )
            if (
                forecast.law_payload not in instance.law_payloads
                or calibration not in instance.joint_calibrations
                or forecast.public_handoff != instance.public_handoff
                or forecast.frame_translation != instance.frame_translation
                or forecast.information_cutoff != instance.information_cutoff
            ):
                raise ValueError("prepared probe substitutes the frozen law/calibration package")
            bounds = {
                (b.qualification_view_id, b.coordinate.coordinate_id): b for b in forecast.bounds
            }
            observed = {
                (o.view_id, r.coordinate.coordinate_id): r for o in outcomes for r in o.readouts
            }
            if set(bounds) != set(observed) or any(
                bounds[key].coordinate != observed[key].coordinate for key in bounds
            ):
                adequate.append(GateStatus.UNEVALUABLE)
                covered.append(GateStatus.UNEVALUABLE)
                sharp.append(GateStatus.UNEVALUABLE)
                continue
            if any(
                key[1] not in tolerances
                or tolerances[key[1]].coordinate != observed[key].coordinate
                for key in bounds
            ):
                raise ValueError("prepared probe substitutes an unqualified readout contract")
            errors = all(
                abs(observed[key].value - bound.absolute_mean)
                <= tolerances[key[1]].adequacy_epsilon
                for key, bound in bounds.items()
            )
            widths = all(
                (bound.upper - bound.lower) / 2 + bound.numerical_floor
                <= tolerances[key[1]].adequacy_epsilon
                for key, bound in bounds.items()
            )
            coverage = all(
                bound.lower - bound.numerical_floor
                <= observed[key].value
                <= bound.upper + bound.numerical_floor
                for key, bound in bounds.items()
            )
            probe_delivery: list[GateStatus] = []
            for outcome in outcomes:
                trace = outcome.delivery_trace
                if trace is None:
                    probe_delivery.append(GateStatus.UNEVALUABLE)
                elif (
                    trace.expected_action_word is None
                    or ObjectIdentity.from_record(
                        trace.expected_action_word.word_id, trace.expected_action_word
                    )
                    != slot.assigned_probe_word
                ) or trace.commitment != ObjectIdentity.from_record(probe.probe_id, probe):
                    raise ValueError(
                        "prepared probe delivery substitutes its independent preassigned word"
                    )
                else:
                    probe_delivery.append(
                        GateStatus.PASS
                        if trace.operational_state is OperationalDeliveryState.DELIVERED
                        else GateStatus.FAIL
                    )
            support = self._prepared_intersection(
                (
                    parent_preservation,
                    self._prepared_numerical_status(plan, outcomes),
                    *probe_delivery,
                    *(
                        self._prepared_preservation_status(
                            plan,
                            o.preservation_outcomes,
                            plan.future_preservation_predicate_ids,
                        )
                        for o in outcomes
                    ),
                )
            )
            adequate.append(
                self._prepared_intersection(
                    (support, GateStatus.PASS if errors and widths else GateStatus.FAIL)
                )
            )
            covered.append(
                self._prepared_intersection(
                    (support, GateStatus.PASS if coverage else GateStatus.FAIL)
                )
            )
            sharp.append(GateStatus.PASS if widths else GateStatus.FAIL)
        success = (
            handoff
            and admitted
            and all(s is GateStatus.PASS for s in (target, delivery, preservation, numerics))
        )
        probe_adequacy = self._prepared_intersection(tuple(adequate))
        probe_coverage = self._prepared_intersection(tuple(covered))
        probe_sharpness = self._prepared_intersection(tuple(sharp))
        reasons = {code for locator in sealed.locators for code in locator.reason_codes}
        reasons.update(sealed.parent_return.reason_codes)
        if not admitted:
            reasons.add("ASSIGNED_ROOT_NOT_ADMITTED")
        for label, status in (
            ("NATIVE_TARGET", target),
            ("DELIVERY", delivery),
            ("PRESERVATION", preservation),
            ("NUMERICS", numerics),
            ("PROBE_ADEQUACY", probe_adequacy),
        ):
            if status is not GateStatus.PASS:
                reasons.add(f"{label}_{status.value}")
        return PreparedPolicyUnitEvaluation(
            evaluation_id=f"prepared-unit.{root.root_id}.{design.policy_id}",
            root_id=root.root_id,
            policy_id=design.policy_id,
            context_id=root.context_id,
            problem_id=root.problem_id,
            common_checkpoint=design.common_checkpoint,
            evaluation_plan=plan_id,
            revealed_bundle=ObjectIdentity.from_record(revealed.reveal_id, revealed),
            independent_unit_count=1,
            handoff_available=handoff,
            admitted=admitted,
            native_target=target,
            delivery=delivery,
            preservation=preservation,
            numerics=numerics,
            task_success=success,
            admitted_failure=admitted and not success,
            probe_adequacy=probe_adequacy,
            probe_coverage=probe_coverage,
            probe_sharpness=probe_sharpness,
            dispositions=tuple(sorted({locator.disposition for locator in sealed.locators})),
            reason_codes=tuple(sorted(reasons)),
        )

    def _validate_context(
        self,
        *,
        compiled: CompiledAdmissionControllerStudy,
        revealed: RevealedRepeatedDeliveryControllerBundle,
    ) -> RepeatedDeliveryControllerEvaluationPlan:
        plan = revealed.sealed_bundle.evaluation_plan
        plan_identity = ObjectIdentity.from_record(plan.evaluation_plan_id, plan)
        compiled_identity = ObjectIdentity.from_record(
            compiled.compiled_study_id,
            compiled,
        )
        if (
            compiled.study.prospective_evaluation != plan_identity
            or compiled.prospective_evaluation_binding.disposition is not ProspectiveBindingDisposition.BOUND
            or compiled.prospective_evaluation_binding.evaluation_plan != plan_identity
            or compiled.implementation(ImplementationRole.OUTCOME_EVALUATOR)
            != self.implementation_binding
            or plan.evaluator_boundary.outcome_evaluator_binding_id
            != self.implementation_binding.binding_id
            or plan.reducer != self.reducer_registration
            or revealed.sealed_bundle.compiled_study != compiled_identity
        ):
            raise ValueError("nested controller use evaluator/context differs from frozen programme")
        if plan.measured_hold_fibre is not None:
            hold = compiled.hold_fibre
            if (
                hold is None
                or hold.disposition is not HoldFibreDisposition.QUALIFIED
                or ObjectIdentity.from_record(
                    hold.source.hold_fibre_id,
                    hold.source,
                )
                != plan.measured_hold_fibre
            ):
                raise ValueError("nested controller use efficacy lacks its exact qualified HOLD baseline")
        return plan

    @staticmethod
    def _correctness(
        sealed: SealedNestedReplicate,
        reference: RevealedReplicateDispositionReference,
    ) -> tuple[DispositionCorrectness, tuple[str, ...]]:
        observed = sealed.tick.commitment.disposition
        if reference.safety is ReferenceSafetyDisposition.UNSAFE:
            return DispositionCorrectness.UNSAFE, ("FINITE_CHART_REFERENCE_UNSAFE",)
        if reference.safety is ReferenceSafetyDisposition.UNEVALUABLE:
            return DispositionCorrectness.UNEVALUABLE, ("FINITE_CHART_REFERENCE_UNEVALUABLE",)
        if reference.expected_commitment is observed:
            return DispositionCorrectness.CORRECT, ()
        return DispositionCorrectness.INCORRECT, (
            "COMMITMENT_DISPOSITION_DIFFERS_FROM_SEALED_REFERENCE",
        )

    @staticmethod
    def _reduce_repeats(
        values: tuple[Decimal, ...],
        reducer: NestedRepeatReducer,
    ) -> Decimal:
        if not values:
            raise ValueError("cannot reduce an empty nested repeat roster")
        if reducer is NestedRepeatReducer.ARITHMETIC_MEAN:
            return sum(values, Decimal(0)) / Decimal(len(values))
        return min(values)

    def evaluate_unit(
        self,
        *,
        compiled: CompiledAdmissionControllerStudy,
        revealed: RevealedRepeatedDeliveryControllerBundle,
        independent_unit_id: str,
    ) -> RepeatedDeliveryControllerUnitEvaluation:
        validate_stable_id(independent_unit_id, field_name="independent_unit_id")
        plan = self._validate_context(compiled=compiled, revealed=revealed)
        unit = next(
            (
                value
                for value in plan.independent_units
                if value.independent_unit_id == independent_unit_id
            ),
            None,
        )
        if unit is None:
            raise ValueError("nested unit is outside the frozen physical-unit roster")
        sealed_by_id = {
            value.replicate.replicate_id: value
            for value in revealed.sealed_bundle.replicates
            if value.replicate.independent_unit_id == independent_unit_id
        }
        references = {value.replicate_id: value for value in revealed.disposition_references}
        outcomes = {
            (value.replicate_id, value.branch_id, value.model_member_id): value
            for value in revealed.outcomes
        }
        repeat_results: list[NestedRepeatEvaluation] = []
        for replicate in (
            value for value in plan.replicates if value.independent_unit_id == independent_unit_id
        ):
            sealed = sealed_by_id[replicate.replicate_id]
            reference = references[replicate.replicate_id]
            correctness, correctness_reasons = self._correctness(sealed, reference)
            observed_commitment = sealed.tick.commitment.disposition
            repeat_member_effects: list[NestedMemberRepeatEffect] = []
            repeat_reasons: set[str] = set(correctness_reasons)
            failure_dispositions: set[NestedRepeatDisposition] = set()
            if correctness is DispositionCorrectness.UNSAFE:
                failure_dispositions.add(NestedRepeatDisposition.UNSAFE)
            elif correctness is DispositionCorrectness.INCORRECT:
                failure_dispositions.add(NestedRepeatDisposition.REFERENCE_INCORRECT)
            elif correctness is DispositionCorrectness.UNEVALUABLE:
                failure_dispositions.add(NestedRepeatDisposition.UNEVALUABLE)
            if observed_commitment is CommitmentDisposition.NONATTEMPT:
                disposition = (
                    self._repeat_failure(failure_dispositions)
                    if failure_dispositions
                    else NestedRepeatDisposition.NONATTEMPT
                )
                repeat_reasons.add("CONTROLLER_NONATTEMPT")
            else:
                for member in plan.model_member_ids:
                    controller = outcomes[
                        (replicate.replicate_id, plan.controller_branch_id, member)
                    ]
                    reference_outcome = outcomes[
                        (replicate.replicate_id, plan.reference_branch_id, member)
                    ]
                    predicates = tuple(
                        evaluate_controller_outcome_predicate(
                            evaluation_id=(
                                f"nested-outcome-gate.{replicate.replicate_id}."
                                f"{member}.{predicate.predicate_id}"
                            ),
                            predicate=predicate,
                            raw={value.value_id: value for value in controller.raw_outcomes},
                        )
                        for predicate in plan.evaluator_boundary.outcome_predicates
                    )
                    reasons: set[str] = set()
                    member_disposition = NestedMemberRepeatDisposition.EVALUABLE
                    effect: NamedDecimal | None = None
                    branch_outcomes = [controller, reference_outcome]
                    if plan.hold_baseline_branch_id is not None:
                        branch_outcomes.append(
                            outcomes[
                                (
                                    replicate.replicate_id,
                                    plan.hold_baseline_branch_id,
                                    member,
                                )
                            ]
                        )
                    hold_action = (
                        compiled.hold_fibre.action_binding.action_word
                        if compiled.hold_fibre is not None
                        else None
                    )
                    if any(
                        value.initial_state_fingerprint != sealed.common_initial_state_fingerprint
                        or value.prefix_fingerprint != sealed.common_prefix_fingerprint
                        for value in branch_outcomes
                    ):
                        member_disposition = NestedMemberRepeatDisposition.UNEVALUABLE
                        reasons.add("MATCHED_PREPARATION_OR_PREFIX_MISMATCH")
                    elif any(not value.delivery_trace.exact for value in branch_outcomes):
                        member_disposition = NestedMemberRepeatDisposition.DELIVERY_INVALID
                        reasons.add("FRESH_DELIVERY_INVALID")
                    elif (
                        plan.hold_baseline_branch_id is not None
                        and branch_outcomes[-1].action_word != hold_action
                    ):
                        member_disposition = NestedMemberRepeatDisposition.DELIVERY_INVALID
                        reasons.add("HOLD_BASELINE_ACTION_WORD_MISMATCH")
                    elif any(value.technical_reason_codes for value in branch_outcomes):
                        member_disposition = NestedMemberRepeatDisposition.TECHNICAL_INVALID
                        reasons.update(
                            reason
                            for value in branch_outcomes
                            for reason in value.technical_reason_codes
                        )
                    elif any(value.status is GateStatus.FAIL for value in predicates):
                        member_disposition = NestedMemberRepeatDisposition.UNSAFE
                        reasons.add("CONTROLLER_OUTCOME_UNSAFE")
                    elif any(value.status is GateStatus.UNEVALUABLE for value in predicates):
                        member_disposition = NestedMemberRepeatDisposition.UNEVALUABLE
                        reasons.add("CONTROLLER_OUTCOME_UNEVALUABLE")
                    elif observed_commitment is CommitmentDisposition.MEASURED_HOLD_COMMITTED:
                        member_disposition = NestedMemberRepeatDisposition.QUALIFIED_HOLD
                        reasons.add("CONTROLLER_COMMITTED_MEASURED_HOLD")
                    elif plan.hold_baseline_branch_id is None:
                        member_disposition = NestedMemberRepeatDisposition.UNEVALUABLE
                        reasons.add("QUALIFIED_HOLD_EFFICACY_BASELINE_ABSENT")
                    else:
                        hold_outcome = branch_outcomes[-1]
                        controller_raw = {
                            value.value_id: value for value in controller.raw_outcomes
                        }
                        hold_raw = {value.value_id: value for value in hold_outcome.raw_outcomes}
                        controller_value = controller_raw.get(plan.effect_quantity_id)
                        hold_value = hold_raw.get(plan.effect_quantity_id)
                        if (
                            controller_value is None
                            or hold_value is None
                            or controller_value.unit != plan.effect_native_unit
                            or hold_value.unit != plan.effect_native_unit
                        ):
                            member_disposition = NestedMemberRepeatDisposition.UNEVALUABLE
                            reasons.add("HOLD_EFFICACY_EFFECT_MISSING_OR_WRONG_UNIT")
                        else:
                            effect_value = (
                                controller_value.value - hold_value.value
                                if plan.favorable_direction is UtilityDirection.HIGHER_IS_BETTER
                                else hold_value.value - controller_value.value
                            )
                            effect = NamedDecimal(
                                value_id=f"nested-effect.{replicate.replicate_id}.{member}",
                                value=effect_value,
                                unit=plan.effect_native_unit,
                            )
                    if member_disposition is not NestedMemberRepeatDisposition.EVALUABLE:
                        mapped = {
                            NestedMemberRepeatDisposition.UNSAFE: NestedRepeatDisposition.UNSAFE,
                            NestedMemberRepeatDisposition.DELIVERY_INVALID: (
                                NestedRepeatDisposition.DELIVERY_INVALID
                            ),
                            NestedMemberRepeatDisposition.TECHNICAL_INVALID: (
                                NestedRepeatDisposition.TECHNICAL_INVALID
                            ),
                            NestedMemberRepeatDisposition.UNEVALUABLE: (
                                NestedRepeatDisposition.UNEVALUABLE
                            ),
                            NestedMemberRepeatDisposition.QUALIFIED_HOLD: (
                                NestedRepeatDisposition.QUALIFIED_HOLD
                            ),
                            NestedMemberRepeatDisposition.NONATTEMPT: (
                                NestedRepeatDisposition.NONATTEMPT
                            ),
                        }.get(member_disposition)
                        if mapped is not None:
                            failure_dispositions.add(mapped)
                        repeat_reasons.update(reasons)
                    repeat_member_effects.append(
                        NestedMemberRepeatEffect(
                            effect_id=f"nested-member-repeat.{replicate.replicate_id}.{member}",
                            replicate_id=replicate.replicate_id,
                            model_member_id=member,
                            disposition=member_disposition,
                            effect=effect,
                            predicate_evaluations=tuple(
                                sorted(predicates, key=lambda value: value.evaluation_id)
                            ),
                            reason_codes=tuple(sorted(reasons)),
                        )
                    )
                if not failure_dispositions:
                    disposition = NestedRepeatDisposition.ACTIVE_EVALUABLE
                    repeat_reasons.clear()
                elif failure_dispositions == {NestedRepeatDisposition.QUALIFIED_HOLD}:
                    disposition = NestedRepeatDisposition.QUALIFIED_HOLD
                else:
                    disposition = self._repeat_failure(failure_dispositions)
            repeat_results.append(
                NestedRepeatEvaluation(
                    repeat_evaluation_id=f"nested-repeat-evaluation.{replicate.replicate_id}",
                    replicate=replicate,
                    observed_commitment=observed_commitment,
                    expected_commitment=reference.expected_commitment,
                    correctness=correctness,
                    disposition=disposition,
                    member_effects=tuple(
                        sorted(
                            repeat_member_effects,
                            key=lambda value: value.model_member_id,
                        )
                    ),
                    reason_codes=tuple(sorted(repeat_reasons)),
                )
            )
        repeats = tuple(sorted(repeat_results, key=lambda value: value.replicate.replicate_id))
        dispositions = {value.disposition for value in repeats}
        unit_reasons = {reason for value in repeats for reason in value.reason_codes}
        aggregated_member_effects: tuple[RepeatedDeliveryMemberEffect, ...] = ()
        robust: NamedDecimal | None = None
        if dispositions == {NestedRepeatDisposition.ACTIVE_EVALUABLE}:
            aggregates = []
            for member in plan.model_member_ids:
                values_list: list[Decimal] = []
                for repeat in repeats:
                    repeat_effect = next(
                        value for value in repeat.member_effects if value.model_member_id == member
                    )
                    if repeat_effect.effect is None:
                        raise ValueError("nested repeat/member effect grid is incomplete")
                    values_list.append(repeat_effect.effect.value)
                values = tuple(values_list)
                aggregates.append(
                    RepeatedDeliveryMemberEffect(
                        model_member_id=member,
                        repeat_count=len(values),
                        effect=NamedDecimal(
                            value_id=f"nested-member-effect.{independent_unit_id}.{member}",
                            value=self._reduce_repeats(
                                values,
                                plan.reducer.repeat_reducer,
                            ),
                            unit=plan.effect_native_unit,
                        ),
                    )
                )
            aggregated_member_effects = tuple(
                sorted(aggregates, key=lambda value: value.model_member_id)
            )
            if plan.reducer.member_reducer is not NestedMemberReducer.WORST_MEMBER:
                raise ValueError("nested member reducer is not registered")
            robust = NamedDecimal(
                value_id=f"nested-robust-effect.{independent_unit_id}",
                value=min(value.effect.value for value in aggregated_member_effects),
                unit=plan.effect_native_unit,
            )
            unit_disposition = NestedControllerUnitDisposition.ACTIVE_EVALUABLE
            unit_reasons.clear()
        elif dispositions == {NestedRepeatDisposition.QUALIFIED_HOLD}:
            unit_disposition = NestedControllerUnitDisposition.QUALIFIED_HOLD
            unit_reasons.add("ALL_REPETITIONS_COMMITTED_QUALIFIED_HOLD")
        elif dispositions == {NestedRepeatDisposition.NONATTEMPT}:
            unit_disposition = NestedControllerUnitDisposition.NONATTEMPT
            unit_reasons.add("ALL_REPETITIONS_NONATTEMPT")
        elif NestedRepeatDisposition.UNSAFE in dispositions:
            unit_disposition = NestedControllerUnitDisposition.UNSAFE
        elif NestedRepeatDisposition.DELIVERY_INVALID in dispositions:
            unit_disposition = NestedControllerUnitDisposition.DELIVERY_INVALID
        elif NestedRepeatDisposition.TECHNICAL_INVALID in dispositions:
            unit_disposition = NestedControllerUnitDisposition.TECHNICAL_INVALID
        elif NestedRepeatDisposition.REFERENCE_INCORRECT in dispositions:
            unit_disposition = NestedControllerUnitDisposition.REFERENCE_INCORRECT
        elif len(dispositions) > 1:
            unit_disposition = NestedControllerUnitDisposition.PARTIAL_HETEROGENEOUS
            unit_reasons.add("REPETITION_DISPOSITIONS_HETEROGENEOUS")
        else:
            unit_disposition = NestedControllerUnitDisposition.UNEVALUABLE
        return RepeatedDeliveryControllerUnitEvaluation(
            unit_evaluation_id=f"nested-unit-evaluation.{independent_unit_id}",
            independent_unit_id=independent_unit_id,
            task_id=unit.task_id,
            preparation_unit_id=unit.preparation_unit_id,
            evaluation_plan=ObjectIdentity.from_record(plan.evaluation_plan_id, plan),
            compiled_study=ObjectIdentity.from_record(
                compiled.compiled_study_id,
                compiled,
            ),
            bundle=ObjectIdentity.from_record(
                revealed.sealed_bundle.bundle_id,
                revealed.sealed_bundle,
            ),
            reducer_registration=ObjectIdentity.from_record(
                plan.reducer.registration_id,
                plan.reducer,
            ),
            disposition=unit_disposition,
            repeat_evaluations=repeats,
            member_effects=aggregated_member_effects,
            robust_effect=robust,
            independent_unit_count=1,
            nested_repeat_count=len(repeats),
            nested_member_count=len(plan.model_member_ids),
            correct_reference_count=sum(
                value.correctness is DispositionCorrectness.CORRECT for value in repeats
            ),
            reason_codes=tuple(sorted(unit_reasons)),
        )

    @staticmethod
    def _repeat_failure(
        dispositions: set[NestedRepeatDisposition],
    ) -> NestedRepeatDisposition:
        return next(
            value
            for value in (
                NestedRepeatDisposition.UNSAFE,
                NestedRepeatDisposition.DELIVERY_INVALID,
                NestedRepeatDisposition.TECHNICAL_INVALID,
                NestedRepeatDisposition.REFERENCE_INCORRECT,
                NestedRepeatDisposition.UNEVALUABLE,
                NestedRepeatDisposition.QUALIFIED_HOLD,
                NestedRepeatDisposition.NONATTEMPT,
            )
            if value in dispositions
        )

    def adjudicate_cohort(
        self,
        *,
        compiled: CompiledAdmissionControllerStudy,
        revealed: RevealedRepeatedDeliveryControllerBundle,
        units: tuple[RepeatedDeliveryControllerUnitEvaluation, ...],
    ) -> RepeatedDeliveryControllerCohortAdjudication:
        plan = self._validate_context(compiled=compiled, revealed=revealed)
        expected_unit_ids = tuple(value.independent_unit_id for value in plan.independent_units)
        if tuple(value.independent_unit_id for value in units) != expected_unit_ids:
            raise ValueError("nested cohort units differ from frozen physical-unit roster")
        plan_identity = ObjectIdentity.from_record(plan.evaluation_plan_id, plan)
        compiled_identity = ObjectIdentity.from_record(
            compiled.compiled_study_id,
            compiled,
        )
        bundle_identity = ObjectIdentity.from_record(
            revealed.sealed_bundle.bundle_id,
            revealed.sealed_bundle,
        )
        reducer_identity = ObjectIdentity.from_record(
            plan.reducer.registration_id,
            plan.reducer,
        )
        if any(
            value.evaluation_plan != plan_identity
            or value.compiled_study != compiled_identity
            or value.bundle != bundle_identity
            or value.reducer_registration != reducer_identity
            or value.independent_unit_count != 1
            or value.nested_member_count != len(plan.model_member_ids)
            or any(
                repeat.member_effects
                and tuple(effect.model_member_id for effect in repeat.member_effects)
                != plan.model_member_ids
                for repeat in value.repeat_evaluations
            )
            for value in units
        ):
            raise ValueError("nested cohort unit changes plan/evaluator/physical-unit identity")
        counts = {
            disposition: sum(value.disposition is disposition for value in units)
            for disposition in NestedControllerUnitDisposition
        }
        effects = tuple(
            value.robust_effect.value
            for value in units
            if value.disposition is NestedControllerUnitDisposition.ACTIVE_EVALUABLE
            and value.robust_effect is not None
        )
        mean: NamedDecimal | None = None
        lower: NamedDecimal | None = None
        if effects:
            mean_value = sum(effects, Decimal(0)) / Decimal(len(effects))
            if len(effects) == 1:
                standard_error = Decimal(0)
            else:
                variance = sum(
                    ((value - mean_value) ** 2 for value in effects),
                    Decimal(0),
                ) / Decimal(len(effects) - 1)
                standard_error = (variance / Decimal(len(effects))).sqrt()
            lower_value = mean_value - plan.one_sided_critical_value * standard_error
            mean = NamedDecimal(
                value_id=f"nested-mean-effect.{plan.evaluation_plan_id}",
                value=mean_value,
                unit=plan.effect_native_unit,
            )
            lower = NamedDecimal(
                value_id=f"nested-lower-bound.{plan.evaluation_plan_id}",
                value=lower_value,
                unit=plan.effect_native_unit,
            )
        active_count = counts[NestedControllerUnitDisposition.ACTIVE_EVALUABLE]
        qualified_hold_count = counts[NestedControllerUnitDisposition.QUALIFIED_HOLD]
        evaluable_commitment_count = active_count + qualified_hold_count
        active_coverage = Decimal(active_count) / Decimal(len(units))
        total_repeats = sum(value.nested_repeat_count for value in units)
        correct_references = sum(value.correct_reference_count for value in units)
        correctness_coverage = Decimal(correct_references) / Decimal(total_repeats)
        stratum_failure = any(
            sum(
                value.disposition
                in {
                    NestedControllerUnitDisposition.ACTIVE_EVALUABLE,
                    NestedControllerUnitDisposition.QUALIFIED_HOLD,
                }
                for value in units
                if value.independent_unit_id in stratum.independent_unit_ids
            )
            < stratum.minimum_attempted_units
            for stratum in plan.strata
        )
        reasons: set[str] = set()
        if counts[NestedControllerUnitDisposition.UNSAFE]:
            result = ControllerUseResult.CONTROLLER_USE_UNSAFE
            reasons.add("UNSAFE_PHYSICAL_UNIT_PRESENT")
        elif counts[NestedControllerUnitDisposition.DELIVERY_INVALID]:
            result = ControllerUseResult.CONTROLLER_USE_DELIVERY_INVALID
            reasons.add("DELIVERY_INVALID_PHYSICAL_UNIT_PRESENT")
        elif counts[NestedControllerUnitDisposition.TECHNICAL_INVALID]:
            result = ControllerUseResult.CONTROLLER_USE_TECHNICAL_FAILURE
            reasons.add("TECHNICAL_INVALID_PHYSICAL_UNIT_PRESENT")
        elif evaluable_commitment_count < plan.minimum_evaluable_units:
            result = ControllerUseResult.CONTROLLER_USE_UNEVALUABLE
            reasons.add("MINIMUM_EVALUABLE_PHYSICAL_UNITS_NOT_MET")
        elif (
            counts[NestedControllerUnitDisposition.REFERENCE_INCORRECT]
            or counts[NestedControllerUnitDisposition.PARTIAL_HETEROGENEOUS]
            or counts[NestedControllerUnitDisposition.NONATTEMPT]
            or counts[NestedControllerUnitDisposition.UNEVALUABLE]
            or stratum_failure
            or correctness_coverage < Decimal(1)
        ):
            result = ControllerUseResult.CONTROLLER_USE_PARTIAL_OR_HETEROGENEOUS
            reasons.add("REFERENCE_OR_STRATUM_OR_REPETITION_HETEROGENEITY")
        elif qualified_hold_count > active_count:
            result = ControllerUseResult.CONTROLLER_USE_HOLD_DOMINANT
            reasons.add("QUALIFIED_HOLD_DOMINATES_ACTIVE_PHYSICAL_UNITS")
        elif active_coverage < plan.minimum_active_coverage:
            result = ControllerUseResult.CONTROLLER_USE_UNEVALUABLE
            reasons.add("MINIMUM_ACTIVE_PHYSICAL_UNIT_COVERAGE_NOT_MET")
        elif lower is not None and lower.value > plan.materiality.value:
            result = ControllerUseResult.CONTROLLER_USE_VALIDATED
        elif mean is not None and mean.value > 0:
            result = ControllerUseResult.CONTROLLER_USE_POSITIVE_BUT_BELOW_MATERIALITY
            reasons.add("LOWER_BOUND_DOES_NOT_EXCEED_MATERIALITY")
        elif effects:
            result = ControllerUseResult.CONTROLLER_USE_NEGATIVE
            reasons.add("NONPOSITIVE_PHYSICAL_UNIT_MEAN_EFFECT")
        else:
            result = ControllerUseResult.CONTROLLER_USE_UNEVALUABLE
            reasons.add("NO_EVALUABLE_PHYSICAL_UNIT_EFFECTS")
        return RepeatedDeliveryControllerCohortAdjudication(
            adjudication_id=f"nested-cohort-adjudication.{plan.evaluation_plan_id}",
            evaluation_plan=plan_identity,
            compiled_study=compiled_identity,
            bundle=bundle_identity,
            reducer_registration=reducer_identity,
            units=units,
            disposition_counts=tuple(
                NestedUnitDispositionCount(disposition=value, count=counts[value])
                for value in NestedControllerUnitDisposition
            ),
            rostered_independent_unit_count=len(units),
            effective_independent_unit_count=active_count,
            nested_repeat_count=total_repeats,
            nested_member_count=len(plan.model_member_ids),
            mean_effect=mean,
            one_sided_lower_bound=lower,
            active_coverage=active_coverage,
            reference_correctness_coverage=correctness_coverage,
            result=result,
            claim_ceiling=plan.maximum_claim_ceiling,
            reason_codes=tuple(sorted(reasons)),
        )


class ProspectiveExecutionEventKind(StrEnum):
    PROGRAMME_ISSUED = "PROGRAMME_ISSUED"
    COMPILED = "COMPILED"
    EVALUATOR_BOUND = "EVALUATOR_BOUND"
    TICK_RESERVED = "TICK_RESERVED"
    REFERENCE_RESERVED = "REFERENCE_RESERVED"
    CONFIRMATORY_RESERVED = "CONFIRMATORY_RESERVED"

    @property
    def protected(self) -> bool:
        return self in {
            self.TICK_RESERVED,
            self.REFERENCE_RESERVED,
            self.CONFIRMATORY_RESERVED,
        }


@dataclass(frozen=True, slots=True)
class ProspectiveExecutionEvent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prospective-execution-event'

    event_id: str
    sequence_number: int
    kind: ProspectiveExecutionEventKind
    coordinate_id: str
    subject: ObjectIdentity
    parent: ObjectIdentity | None
    occurred_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.event_id, field_name="event_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        if self.sequence_number < 1:
            raise ValueError("prospective event sequence number must be positive")
        parse_utc_timestamp(self.occurred_at_utc, field_name="occurred_at_utc")


def _prospective_prefix_sha256(events: tuple[ProspectiveExecutionEvent, ...]) -> str:
    digest = sha256()
    for event in events:
        digest.update(event.canonical_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ProspectiveExecutionEventPrefix(CanonicalRecord):
    """Exact compare-and-append root for one task/arm execution coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prospective-execution-event-prefix'

    prefix_id: str
    coordinate_id: str
    events: tuple[ProspectiveExecutionEvent, ...]
    event_root_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.prefix_id, field_name="prefix_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_sha256(self.event_root_sha256, field_name="event_root_sha256")
        require_unique_ids(self.events, attribute="event_id", field_name="events")
        if tuple(value.sequence_number for value in self.events) != tuple(
            range(1, len(self.events) + 1)
        ):
            raise ValueError("prospective event prefix sequence is not contiguous")
        if any(value.coordinate_id != self.coordinate_id for value in self.events):
            raise ValueError("prospective event prefix mixes coordinates")
        if self.event_root_sha256 != _prospective_prefix_sha256(self.events):
            raise ValueError("prospective event prefix root is not derived")

    @classmethod
    def from_events(
        cls,
        *,
        prefix_id: str,
        coordinate_id: str,
        events: tuple[ProspectiveExecutionEvent, ...],
    ) -> ProspectiveExecutionEventPrefix:
        return cls(
            prefix_id=prefix_id,
            coordinate_id=coordinate_id,
            events=events,
            event_root_sha256=_prospective_prefix_sha256(events),
        )

    def append(self, event: ProspectiveExecutionEvent) -> ProspectiveExecutionEventPrefix:
        if (
            event.coordinate_id != self.coordinate_id
            or event.sequence_number != len(self.events) + 1
        ):
            raise ValueError("prospective event is not the next event for this coordinate")
        return self.from_events(
            prefix_id=self.prefix_id,
            coordinate_id=self.coordinate_id,
            events=(*self.events, event),
        )


class ProspectiveBindingConflict(ValueError):
    """The durable prefix no longer permits the requested transition."""


class DurableProspectiveExecutionEventStore(Protocol):
    def create(self, prefix: ProspectiveExecutionEventPrefix) -> None: ...

    def load(self, prefix_id: str) -> ProspectiveExecutionEventPrefix: ...

    def compare_and_append(
        self,
        *,
        expected_prefix_sha256: str,
        updated: ProspectiveExecutionEventPrefix,
    ) -> None: ...


class InMemoryProspectiveExecutionEventStore:
    """Thread-safe conformance store; production injects an external durable port."""

    def __init__(self) -> None:
        self._prefixes: dict[str, ProspectiveExecutionEventPrefix] = {}
        self._prepared_prefixes: dict[str, PreparedExecutionEventPrefix] = {}
        self._prepared_records: dict[
            tuple[str, str], tuple[ObjectIdentity, ArtifactIdentity, bytes]
        ] = {}
        self._lock = RLock()

    def publish_prepared_record(
        self,
        *,
        prefix_id: str,
        object_id: str,
        record: CanonicalRecord,
    ) -> ArtifactIdentity:
        validate_stable_id(prefix_id, field_name="prefix_id")
        validate_stable_id(object_id, field_name="object_id")
        payload = record.canonical_bytes()
        identity = ObjectIdentity.from_record(object_id, record)
        artifact = ArtifactIdentity(
            artifact_id=object_id,
            role="prepared-control-record",
            payload_schema=record.SCHEMA,
            sha256=sha256(payload).hexdigest(),
            media_type="application/json",
            size_bytes=len(payload),
        )
        with self._lock:
            key = (prefix_id, object_id)
            value = (identity, artifact, payload)
            if self._prepared_records.setdefault(key, value) != value:
                raise ProspectiveBindingConflict("PREPARED_IMMUTABLE_RECORD_CONFLICT")
        return artifact

    def verify_prepared_subject(
        self,
        *,
        prefix_id: str,
        subject: ObjectIdentity,
        artifact: ArtifactIdentity,
    ) -> None:
        with self._lock:
            stored = self._prepared_records.get((prefix_id, subject.object_id))
            if stored is None or stored[:2] != (subject, artifact):
                raise ProspectiveBindingConflict("PREPARED_SUBJECT_NOT_PERSISTED")

    def read_prepared_record(
        self,
        *,
        prefix_id: str,
        subject: ObjectIdentity,
        artifact: ArtifactIdentity,
    ) -> bytes:
        with self._lock:
            self.verify_prepared_subject(prefix_id=prefix_id, subject=subject, artifact=artifact)
            return self._prepared_records[(prefix_id, subject.object_id)][2]

    def create_prepared(self, prefix: PreparedExecutionEventPrefix) -> None:
        with self._lock:
            if len(prefix.events) != 1:
                raise ProspectiveBindingConflict("PREPARED_INITIAL_PREFIX_REQUIRED")
            self.verify_prepared_subject(
                prefix_id=prefix.prefix_id,
                subject=prefix.events[0].subject,
                artifact=prefix.events[0].subject_artifact,
            )
            prior = self._prepared_prefixes.get(prefix.prefix_id)
            if prior is not None and (
                prior.root_id != prefix.root_id
                or prior.policy_id != prefix.policy_id
                or prior.events[:1] != prefix.events
            ):
                raise ProspectiveBindingConflict("PREPARED_DESIGN_CONFLICT")
            if prior is None:
                self._prepared_prefixes[prefix.prefix_id] = prefix

    def load_prepared(self, prefix_id: str) -> PreparedExecutionEventPrefix:
        with self._lock:
            return self._prepared_prefixes[prefix_id]

    def compare_and_append_prepared(
        self,
        *,
        expected_prefix_sha256: str,
        updated: PreparedExecutionEventPrefix,
    ) -> None:
        with self._lock:
            current = self._prepared_prefixes.get(updated.prefix_id)
            if (
                current is None
                or current.fingerprint() != expected_prefix_sha256
                or updated.events[:-1] != current.events
                or updated.root_id != current.root_id
                or updated.policy_id != current.policy_id
            ):
                raise ProspectiveBindingConflict("PREPARED_BINDING_CONFLICT")
            event = updated.events[-1]
            self.verify_prepared_subject(
                prefix_id=updated.prefix_id, subject=event.subject, artifact=event.subject_artifact
            )
            self._prepared_prefixes[updated.prefix_id] = updated

    def create(self, prefix: ProspectiveExecutionEventPrefix) -> None:
        with self._lock:
            prior = self._prefixes.get(prefix.prefix_id)
            if prior is not None and prior != prefix:
                raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
            self._prefixes[prefix.prefix_id] = prefix

    def load(self, prefix_id: str) -> ProspectiveExecutionEventPrefix:
        with self._lock:
            return self._prefixes[prefix_id]

    def compare_and_append(
        self,
        *,
        expected_prefix_sha256: str,
        updated: ProspectiveExecutionEventPrefix,
    ) -> None:
        with self._lock:
            current = self._prefixes.get(updated.prefix_id)
            if (
                current is None
                or current.fingerprint() != expected_prefix_sha256
                or updated.events[:-1] != current.events
            ):
                raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
            self._prefixes[updated.prefix_id] = updated


@dataclass(frozen=True, slots=True)
class ProspectiveEvaluationBindingReceipt(CanonicalRecord):
    """Actual programme/compiler binding accepted before any protected event."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prospective-evaluation-binding-receipt'

    binding_id: str
    coordinate_id: str
    issued_manifest: ObjectIdentity
    extension_set: ObjectIdentity
    sealed_acquisition_trace: ObjectIdentity
    precommitment: ProspectiveEvaluationPrecommitment
    study: ObjectIdentity
    compiled_study: ObjectIdentity
    reference_design: ObjectIdentity
    nested_plan: ObjectIdentity
    evaluator_binding: ImplementationBinding
    causal_cutoff_id: str
    acquisition_domain_id: str
    reference_domain_id: str
    confirmatory_domain_id: str
    prefix_before: ProspectiveExecutionEventPrefix
    accepted_event: ProspectiveExecutionEvent
    accepted_prefix_root_sha256: str
    bound_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_stable_id(self.causal_cutoff_id, field_name="causal_cutoff_id")
        validate_sha256(
            self.accepted_prefix_root_sha256,
            field_name="accepted_prefix_root_sha256",
        )
        parse_utc_timestamp(self.bound_at_utc, field_name="bound_at_utc")
        if self.study.object_schema != AdmissionControllerStudy.SCHEMA:
            raise ValueError("prospective binding requires admission controller study")
        if self.compiled_study.object_schema != CompiledAdmissionControllerStudy.SCHEMA:
            raise ValueError("prospective binding requires a compiled admission controller study")
        if self.nested_plan.object_schema != ActionAwareControllerEvaluationPlan.SCHEMA:
            raise ValueError("prospective binding requires the action-aware controller evaluation schema")
        if self.accepted_event.kind is not ProspectiveExecutionEventKind.EVALUATOR_BOUND:
            raise ValueError("prospective binding receipt has another event kind")
        if (
            self.accepted_event.coordinate_id != self.coordinate_id
            or self.accepted_event.subject != self.compiled_study
            or self.accepted_event.parent
            != ObjectIdentity.from_record(
                self.precommitment.precommitment_id,
                self.precommitment,
            )
        ):
            raise ValueError("prospective binding event changes coordinate/compiled/precommitment")
        if (
            self.issued_manifest != self.precommitment.issued_manifest
            or self.extension_set != self.precommitment.extension_set
            or self.sealed_acquisition_trace != self.precommitment.sealed_acquisition_trace
            or self.reference_design != self.precommitment.reference_design
            or self.nested_plan
            != ObjectIdentity.from_record(
                self.precommitment.nested_plan.evaluation_plan_id,
                self.precommitment.nested_plan,
            )
            or self.evaluator_binding != self.precommitment.evaluator_binding
            or self.causal_cutoff_id != self.precommitment.causal_cutoff_id
            or self.acquisition_domain_id != self.precommitment.acquisition_domain_id
            or self.reference_domain_id != self.precommitment.reference_domain_id
            or self.confirmatory_domain_id != self.precommitment.confirmatory_domain_id
            or self.bound_at_utc != self.accepted_event.occurred_at_utc
        ):
            raise ValueError("prospective binding receipt changes its frozen precommitment")
        if (
            self.prefix_before.coordinate_id != self.coordinate_id
            or any(value.kind.protected for value in self.prefix_before.events)
            or self.accepted_event.sequence_number != len(self.prefix_before.events) + 1
            or self.prefix_before.append(self.accepted_event).event_root_sha256
            != self.accepted_prefix_root_sha256
        ):
            raise ValueError("prospective binding receipt is not derived from a clean prefix")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("prospective binding receipt cannot reveal outcomes")


class ProspectiveEvaluationBindingCoordinator:
    """Atomic pre-outcome binding and protected-event reservation service."""

    def __init__(
        self,
        store: DurableProspectiveExecutionEventStore | None = None,
        *,
        prepared_store: DurablePreparedExecutionEventStore | None = None,
    ) -> None:
        if store is None and prepared_store is None:
            raise ValueError("prospective binding requires its versioned durable store")
        self._prospective_event_store = store
        self._prepared_store = prepared_store

    @property
    def _store(self) -> DurableProspectiveExecutionEventStore:
        if self._prospective_event_store is None:
            raise ValueError("prospective binding requires its durable prospective-event store")
        return self._prospective_event_store

    def _prepared(self) -> DurablePreparedExecutionEventStore:
        if self._prepared_store is None:
            raise ValueError("prepared execution requires a durable prepared-event store")
        return self._prepared_store

    def freeze_prepared_design(
        self,
        *,
        prefix_id: str,
        binding: PreparedDesignBindingReceipt,
        plan: CommonStartControllerEvaluationPlan,
    ) -> PreparedExecutionEventPrefix:
        expected_slice = prepared_interface_evaluation_slice(
            plan, root_id=binding.root_id, policy_id=binding.policy_id
        )
        if binding.evaluation_plan != expected_slice:
            raise ValueError("prepared design slice differs from the complete frozen census")
        store = self._prepared()
        anchor = PreparedCommonStartBinding.for_design(binding)
        store.publish_prepared_record(
            prefix_id=anchor.binding_id, object_id=anchor.binding_id, record=anchor
        )
        artifact = store.publish_prepared_record(
            prefix_id=prefix_id, object_id=binding.binding_id, record=binding
        )
        event = PreparedExecutionEvent(
            event_id=f"{prefix_id}.event.1",
            sequence_number=1,
            kind=PreparedExecutionEventKind.DESIGN_FROZEN,
            slot_id=prefix_id,
            subject=ObjectIdentity.from_record(binding.binding_id, binding),
            subject_artifact=artifact,
            parent=None,
            occurred_at_utc=binding.frozen_at_utc,
        )
        prefix = PreparedExecutionEventPrefix(
            prefix_id, binding.root_id, binding.policy_id, (event,)
        )
        store.create_prepared(prefix)
        return store.load_prepared(prefix_id)

    def _append_prepared(
        self,
        *,
        prefix: PreparedExecutionEventPrefix,
        kind: PreparedExecutionEventKind,
        slot_id: str,
        subject: ObjectIdentity,
        artifact: ArtifactIdentity,
        parent: PreparedExecutionEvent,
        occurred_at_utc: str,
    ) -> PreparedExecutionEventPrefix:
        store = self._prepared()
        store.verify_prepared_subject(
            prefix_id=prefix.prefix_id, subject=subject, artifact=artifact
        )
        event = PreparedExecutionEvent(
            event_id=f"{prefix.prefix_id}.event.{len(prefix.events) + 1}",
            sequence_number=len(prefix.events) + 1,
            kind=kind,
            slot_id=slot_id,
            subject=subject,
            subject_artifact=artifact,
            parent=ObjectIdentity.from_record(parent.event_id, parent),
            occurred_at_utc=occurred_at_utc,
        )
        updated = prefix.append(event)
        store.compare_and_append_prepared(
            expected_prefix_sha256=prefix.fingerprint(), updated=updated
        )
        return updated

    def _prepared_design_event(
        self,
        prefix: PreparedExecutionEventPrefix,
        design: PreparedDesignBindingReceipt,
    ) -> PreparedExecutionEvent:
        if (
            prefix.root_id != design.root_id
            or prefix.policy_id != design.policy_id
            or prefix.events[0].subject != ObjectIdentity.from_record(design.binding_id, design)
        ):
            raise ProspectiveBindingConflict("PREPARED_DESIGN_SUBSTITUTED")
        anchor = PreparedCommonStartBinding.for_design(design)
        artifact = ArtifactIdentity(
            anchor.binding_id,
            "prepared-control-record",
            anchor.SCHEMA,
            anchor.fingerprint(),
            "application/json",
            len(anchor.canonical_bytes()),
        )
        self._prepared().verify_prepared_subject(
            prefix_id=anchor.binding_id,
            subject=ObjectIdentity.from_record(anchor.binding_id, anchor),
            artifact=artifact,
        )
        return prefix.events[0]

    def record_prepared_boundary(
        self,
        *,
        prefix_id: str,
        design: PreparedDesignBindingReceipt,
        kind: PreparedExecutionEventKind,
        subject: ObjectIdentity,
        artifact: ArtifactIdentity,
        occurred_at_utc: str,
        slot_id: str | None = None,
    ) -> PreparedExecutionEventPrefix:
        """Persist parent/probe/terminal events; task launch has its own stricter seam."""
        prefix = self._prepared().load_prepared(prefix_id)
        first = self._prepared_design_event(prefix, design)
        k = PreparedExecutionEventKind
        allowed = {k.PARENT_RESERVED}
        if kind not in allowed:
            raise ValueError("prepared child/probe/completion/census requires its typed method")
        if subject != design.parent_commitment:
            raise ValueError("parent reservation must bind the exact pre-parent commitment")
        if slot_id is not None:
            raise ValueError("parent reservation cannot masquerade as a future slot")
        return self._append_prepared(
            prefix=prefix,
            kind=kind,
            slot_id=prefix_id,
            subject=subject,
            artifact=artifact,
            parent=first,
            occurred_at_utc=occurred_at_utc,
        )

    def record_prepared_completion(
        self,
        *,
        prefix_id: str,
        design: PreparedDesignBindingReceipt,
        completion: PreparedFutureCompletion,
        occurred_at_utc: str,
    ) -> PreparedExecutionEventPrefix:
        prefix = self._prepared().load_prepared(prefix_id)
        self._prepared_design_event(prefix, design)
        slot = next(
            (
                f
                for f in design.evaluation_plan.futures
                if f.root_id == design.root_id
                and f.policy_id == design.policy_id
                and ObjectIdentity.from_record(f.slot_id, f) == completion.slot
            ),
            None,
        )
        if (
            slot is None
            or tuple(sorted(locator.view_id for locator in completion.locators))
            != design.evaluation_plan.numerical_view_ids
        ):
            raise ValueError("prepared completion changes the assigned future/view roster")
        k = PreparedExecutionEventKind
        reservation_kind = {
            PreparedFutureRole.COMMITTED_TASK: k.TASK_RESERVED,
            PreparedFutureRole.AUDIT_PROBE: k.PROBE_RESERVED,
            PreparedFutureRole.MATCHED_HOLD: k.MATCHED_HOLD_RESERVED,
        }[slot.role]
        parent = next(
            (e for e in prefix.events if e.kind is reservation_kind and e.slot_id == slot.slot_id),
            None,
        )
        if parent is None:
            raise ProspectiveBindingConflict("PREPARED_COMPLETION_WITHOUT_RESERVATION")
        artifact = self._prepared().publish_prepared_record(
            prefix_id=prefix_id, object_id=completion.completion_id, record=completion
        )
        return self._append_prepared(
            prefix=prefix,
            kind={
                PreparedFutureRole.COMMITTED_TASK: k.TASK_COMPLETED,
                PreparedFutureRole.AUDIT_PROBE: k.PROBE_COMPLETED,
                PreparedFutureRole.MATCHED_HOLD: k.MATCHED_HOLD_COMPLETED,
            }[slot.role],
            slot_id=slot.slot_id,
            subject=ObjectIdentity.from_record(completion.completion_id, completion),
            artifact=artifact,
            parent=parent,
            occurred_at_utc=occurred_at_utc,
        )

    def close_prepared_execution(
        self,
        *,
        prefix_id: str,
        design: PreparedDesignBindingReceipt,
        census: PreparedExecutionCensus,
        occurred_at_utc: str,
    ) -> PreparedExecutionEventPrefix:
        prefix = self._prepared().load_prepared(prefix_id)
        self._prepared_design_event(prefix, design)
        slots = {
            f.slot_id: f
            for f in design.evaluation_plan.futures
            if f.root_id == design.root_id and f.policy_id == design.policy_id
        }
        expected = {
            (slot_id, view)
            for slot_id in slots
            for view in design.evaluation_plan.numerical_view_ids
        }
        if (
            census.design_binding != ObjectIdentity.from_record(design.binding_id, design)
            or {(locator.slot_id, locator.view_id) for locator in census.locators} != expected
            or len(census.locators) != len(expected)
        ):
            raise ValueError("terminal census drops or substitutes an assigned future/view")
        k = PreparedExecutionEventKind
        for slot_id in slots:
            locators = tuple(locator for locator in census.locators if locator.slot_id == slot_id)
            reserved = any(
                e.slot_id == slot_id
                and e.kind in {k.TASK_RESERVED, k.PROBE_RESERVED, k.MATCHED_HOLD_RESERVED}
                for e in prefix.events
            )
            completed = next(
                (
                    e
                    for e in prefix.events
                    if e.slot_id == slot_id
                    and e.kind in {k.TASK_COMPLETED, k.PROBE_COMPLETED, k.MATCHED_HOLD_COMPLETED}
                ),
                None,
            )
            if completed is not None:
                payload = self._prepared().read_prepared_record(
                    prefix_id=prefix_id,
                    subject=completed.subject,
                    artifact=completed.subject_artifact,
                )
                receipt = decode_canonical_bytes(
                    payload, PreparedFutureCompletion, maximum_bytes=1024 * 1024
                )
                if receipt.locators != locators:
                    raise ValueError("terminal census rewrites the immutable source completion")
            elif reserved:
                if any(
                    locator.disposition is not PreparedFutureDisposition.COMPLETION_UNKNOWN
                    for locator in locators
                ):
                    raise ValueError(
                        "reserved future without source completion must remain completion-unknown"
                    )
            elif any(
                locator.source_receipt is not None
                or locator.artifact is not None
                or locator.disposition
                in {
                    PreparedFutureDisposition.COMPLETED,
                    PreparedFutureDisposition.COMPLETION_UNKNOWN,
                }
                for locator in locators
            ):
                raise ValueError("unreserved future cannot claim source contact or completion")
        artifact = self._prepared().publish_prepared_record(
            prefix_id=prefix_id, object_id=census.census_id, record=census
        )
        return self._append_prepared(
            prefix=prefix,
            kind=k.TERMINAL,
            slot_id=prefix_id,
            subject=ObjectIdentity.from_record(census.census_id, census),
            artifact=artifact,
            parent=prefix.events[-1],
            occurred_at_utc=occurred_at_utc,
        )

    def reserve_prepared_probe(
        self,
        *,
        prefix_id: str,
        design: PreparedDesignBindingReceipt,
        returned: PreparedParentReturn,
        probe: PreparedProbeCommitment,
        occurred_at_utc: str,
    ) -> PreparedExecutionEventPrefix:
        prefix = self._prepared().load_prepared(prefix_id)
        self._prepared_design_event(prefix, design)
        parent = next(
            (e for e in prefix.events if e.kind is PreparedExecutionEventKind.PARENT_RETURNED), None
        )
        if (
            parent is None
            or parent.subject != ObjectIdentity.from_record(returned.return_id, returned)
            or returned.disposition is not PreparedParentDisposition.HANDOFF_AVAILABLE
            or probe.design_binding != ObjectIdentity.from_record(design.binding_id, design)
            or probe.parent_return != parent.subject
            or probe.forecast.public_handoff != returned.public_handoff
            or probe.slot not in design.evaluation_plan.futures
            or probe.slot.root_id != design.root_id
            or probe.slot.policy_id != design.policy_id
            or probe.forecast.joint_calibration.qualification_view_ids
            != design.evaluation_plan.numerical_view_ids
        ):
            raise ProspectiveBindingConflict("PREPARED_PROBE_CHANGES_HANDOFF_OR_ASSIGNMENT")
        artifact = self._prepared().publish_prepared_record(
            prefix_id=prefix_id, object_id=probe.probe_id, record=probe
        )
        return self._append_prepared(
            prefix=prefix,
            kind=PreparedExecutionEventKind.PROBE_RESERVED,
            slot_id=probe.slot.slot_id,
            subject=ObjectIdentity.from_record(probe.probe_id, probe),
            artifact=artifact,
            parent=parent,
            occurred_at_utc=occurred_at_utc,
        )

    def reserve_prepared_matched_hold(
        self,
        *,
        prefix_id: str,
        design: PreparedDesignBindingReceipt,
        returned: PreparedParentReturn,
        slot: PreparedFutureSlot,
        occurred_at_utc: str,
    ) -> PreparedExecutionEventPrefix:
        """Reserve the evaluator-only same-handoff, same-innovation HOLD."""

        prefix = self._prepared().load_prepared(prefix_id)
        self._prepared_design_event(prefix, design)
        parent = next(
            (e for e in prefix.events if e.kind is PreparedExecutionEventKind.PARENT_RETURNED), None
        )
        expected_hold = ObjectIdentity.from_record(
            design.evaluation_plan.matched_hold_word.word_id,
            design.evaluation_plan.matched_hold_word,
        )
        if (
            parent is None
            or parent.subject != ObjectIdentity.from_record(returned.return_id, returned)
            or returned.disposition is not PreparedParentDisposition.HANDOFF_AVAILABLE
            or slot.role is not PreparedFutureRole.MATCHED_HOLD
            or slot not in design.evaluation_plan.futures
            or slot.root_id != design.root_id
            or slot.policy_id != design.policy_id
            or slot.assigned_probe_word != expected_hold
        ):
            raise ProspectiveBindingConflict("PREPARED_MATCHED_HOLD_CHANGES_HANDOFF_OR_ASSIGNMENT")
        artifact = self._prepared().publish_prepared_record(
            prefix_id=prefix_id, object_id=slot.slot_id, record=slot
        )
        return self._append_prepared(
            prefix=prefix,
            kind=PreparedExecutionEventKind.MATCHED_HOLD_RESERVED,
            slot_id=slot.slot_id,
            subject=ObjectIdentity.from_record(slot.slot_id, slot),
            artifact=artifact,
            parent=parent,
            occurred_at_utc=occurred_at_utc,
        )

    def record_prepared_parent_return(
        self,
        *,
        prefix_id: str,
        design: PreparedDesignBindingReceipt,
        returned: PreparedParentReturn,
        occurred_at_utc: str,
    ) -> PreparedExecutionEventPrefix:
        prefix = self._prepared().load_prepared(prefix_id)
        self._prepared_design_event(prefix, design)
        parent = next(
            (e for e in prefix.events if e.kind is PreparedExecutionEventKind.PARENT_RESERVED), None
        )
        if (
            parent is None
            or returned.design_binding != ObjectIdentity.from_record(design.binding_id, design)
            or returned.common_checkpoint != design.common_checkpoint
            or returned.parent_commitment != design.parent_commitment
        ):
            raise ProspectiveBindingConflict("PREPARED_PARENT_RETURN_SUBSTITUTED")
        artifact = self._prepared().publish_prepared_record(
            prefix_id=prefix_id, object_id=returned.return_id, record=returned
        )
        return self._append_prepared(
            prefix=prefix,
            kind=PreparedExecutionEventKind.PARENT_RETURNED,
            slot_id=prefix_id,
            subject=ObjectIdentity.from_record(returned.return_id, returned),
            artifact=artifact,
            parent=parent,
            occurred_at_utc=occurred_at_utc,
        )

    def bind_prepared_instance(
        self,
        *,
        prefix_id: str,
        binding_id: str,
        design: PreparedDesignBindingReceipt,
        compiled: CompiledDeliveryControllerStudy,
        task: FiniteTaskFunctionalSpec,
        returned: PreparedParentReturn,
        occurred_at_utc: str,
    ) -> PreparedInstanceBindingReceipt:
        return self._bind_prepared_instance(
            prefix_id=prefix_id,
            binding_id=binding_id,
            design=design,
            compiled=compiled,
            task=task,
            returned=returned,
            occurred_at_utc=occurred_at_utc,
            forecast_parent=None,
        )

    def freeze_prepared_forecast_design(
        self,
        *,
        prefix_id: str,
        binding: PreparedDesignBindingReceipt,
        plan: CommonStartControllerEvaluationPlan,
        forecast_parent: PreparedForecastParentCommitment,
        compiled: CompiledDeliveryControllerStudy,
    ) -> PreparedExecutionEventPrefix:
        """Publish the child decision before the design/parent can reserve effects.

        The parent's committed identity contains the whole early decision. This
        makes the existing DESIGN_FROZEN -> PARENT_RESERVED dependency sufficient;
        no new event order, scheduler or backdated event is introduced.
        """
        forecast_parent.validate_binding(binding, compiled)
        store = self._prepared()
        for object_id, record in (
            (compiled.compiled_study_id, compiled),
            (forecast_parent.decision.commitment_id, forecast_parent.decision),
            (forecast_parent.commitment_id, forecast_parent),
        ):
            store.publish_prepared_record(prefix_id=prefix_id, object_id=object_id, record=record)
        return self.freeze_prepared_design(prefix_id=prefix_id, binding=binding, plan=plan)

    def bind_prepared_forecast_instance(
        self,
        *,
        prefix_id: str,
        binding_id: str,
        design: PreparedDesignBindingReceipt,
        compiled: CompiledDeliveryControllerStudy,
        returned: PreparedParentReturn,
        forecast_parent: PreparedForecastParentCommitment,
        occurred_at_utc: str,
    ) -> PreparedInstanceBindingReceipt:
        forecast_parent.validate_binding(design, compiled)
        # Read authentic stored bytes; a newly supplied lookalike is insufficient.
        prefix = self._prepared().load_prepared(prefix_id)
        reserved = next(
            (e for e in prefix.events if e.kind is PreparedExecutionEventKind.PARENT_RESERVED), None
        )
        if reserved is None or reserved.subject != design.parent_commitment:
            raise ProspectiveBindingConflict("PREPARED_FORECAST_PARENT_NOT_RESERVED")
        stored = self._prepared().read_prepared_record(
            prefix_id=prefix_id,
            subject=ObjectIdentity.from_record(
                forecast_parent.commitment_id,
                forecast_parent,
            ),
            artifact=reserved.subject_artifact,
        )
        if stored != forecast_parent.canonical_bytes():
            raise ProspectiveBindingConflict("PREPARED_FORECAST_LOCK_CHANGED")
        return self._bind_prepared_instance(
            prefix_id=prefix_id,
            binding_id=binding_id,
            design=design,
            compiled=compiled,
            task=forecast_parent.task,
            returned=returned,
            occurred_at_utc=occurred_at_utc,
            forecast_parent=forecast_parent,
        )

    def _bind_prepared_instance(
        self,
        *,
        prefix_id: str,
        binding_id: str,
        design: PreparedDesignBindingReceipt,
        compiled: CompiledDeliveryControllerStudy,
        task: FiniteTaskFunctionalSpec,
        returned: PreparedParentReturn,
        occurred_at_utc: str,
        forecast_parent: PreparedForecastParentCommitment | None,
    ) -> PreparedInstanceBindingReceipt:
        prefix = self._prepared().load_prepared(prefix_id)
        self._prepared_design_event(prefix, design)
        parent = next(
            (e for e in prefix.events if e.kind is PreparedExecutionEventKind.PARENT_RETURNED), None
        )
        if (
            parent is None
            or parent.subject != ObjectIdentity.from_record(returned.return_id, returned)
            or returned.design_binding != ObjectIdentity.from_record(design.binding_id, design)
            or returned.disposition is not PreparedParentDisposition.HANDOFF_AVAILABLE
        ):
            raise ProspectiveBindingConflict("PREPARED_PARENT_NOT_RETURNED")
        public_handoff = returned.public_handoff
        if public_handoff is None:
            raise ValueError("available parent return lost its public handoff")
        plan = design.evaluation_plan
        plan_identity = plan.identity
        policy = next(p for p in plan.policies if p.policy_id == design.policy_id)
        instance = compiled.study.instance_binding
        if (
            compiled.study.prospective_evaluation != plan_identity
            or compiled.prospective_evaluation_binding.disposition is not ProspectiveBindingDisposition.BOUND
            or compiled.prospective_evaluation_binding.evaluation_plan != plan_identity
            or compiled.implementation(ImplementationRole.OUTCOME_EVALUATOR)
            != design.evaluator_binding
            or instance.frozen_recipe != policy.child_recipe
            or (forecast_parent is None and instance.public_handoff != public_handoff)
            or instance.task_functional != ObjectIdentity.from_record(task.functional_id, task)
            or task.predeclared_family != plan.target_family
        ):
            raise ValueError("late child changes the frozen recipe/evaluator/handoff/target design")
        slot = next(
            f
            for f in plan.futures
            if f.root_id == design.root_id
            and f.policy_id == design.policy_id
            and f.role is PreparedFutureRole.COMMITTED_TASK
        )
        receipt = PreparedInstanceBindingReceipt(
            binding_id=binding_id,
            design_binding=ObjectIdentity.from_record(design.binding_id, design),
            parent_return=parent.subject,
            parent_return_event=ObjectIdentity.from_record(parent.event_id, parent),
            public_handoff=public_handoff,
            target_reveal=task.target_reveal,
            task_functional=instance.task_functional,
            compiled_study=ObjectIdentity.from_record(compiled.compiled_study_id, compiled),
            controller_instance=ObjectIdentity.from_record(instance.instance_id, instance),
            task_slot=ObjectIdentity.from_record(slot.slot_id, slot),
            bound_at_utc=occurred_at_utc,
        )
        store = self._prepared()
        # Both the full compiled object and the late binding precede any task reservation.
        store.publish_prepared_record(
            prefix_id=prefix_id, object_id=compiled.compiled_study_id, record=compiled
        )
        artifact = store.publish_prepared_record(
            prefix_id=prefix_id, object_id=receipt.binding_id, record=receipt
        )
        self._append_prepared(
            prefix=prefix,
            kind=PreparedExecutionEventKind.INSTANCE_BOUND,
            slot_id=slot.slot_id,
            subject=ObjectIdentity.from_record(receipt.binding_id, receipt),
            artifact=artifact,
            parent=parent,
            occurred_at_utc=occurred_at_utc,
        )
        return receipt

    def store_prepared_commitment(
        self,
        *,
        prefix_id: str,
        instance: PreparedInstanceBindingReceipt,
        commitment: DeliveryControllerDecisionCommitment,
        occurred_at_utc: str,
    ) -> PreparedExecutionEventPrefix:
        prefix = self._prepared().load_prepared(prefix_id)
        reserved = next(
            (e for e in prefix.events if e.kind is PreparedExecutionEventKind.PARENT_RESERVED), None
        )
        if (
            reserved is not None
            and reserved.subject.object_schema == PreparedForecastParentCommitment.SCHEMA
        ):
            raw = self._prepared().read_prepared_record(
                prefix_id=prefix_id,
                subject=reserved.subject,
                artifact=reserved.subject_artifact,
            )
            early = decode_canonical_bytes(
                raw, PreparedForecastParentCommitment, maximum_bytes=128 * 1024 * 1024
            )
            if commitment != early.decision:
                raise ProspectiveBindingConflict("PREPARED_FORECAST_DECISION_RESELECTION")
        parent = next(
            (e for e in prefix.events if e.kind is PreparedExecutionEventKind.INSTANCE_BOUND), None
        )
        if (
            parent is None
            or parent.subject != ObjectIdentity.from_record(instance.binding_id, instance)
            or commitment.compiled_study != instance.compiled_study
            or ObjectIdentity.from_record(
                commitment.instance_binding.instance_id, commitment.instance_binding
            )
            != instance.controller_instance
        ):
            raise ProspectiveBindingConflict("PREPARED_COMMITMENT_INSTANCE_MISMATCH")
        existing = next(
            (e for e in prefix.events if e.kind is PreparedExecutionEventKind.COMMITMENT_STORED),
            None,
        )
        if existing is not None:
            if existing.subject != ObjectIdentity.from_record(commitment.commitment_id, commitment):
                raise ProspectiveBindingConflict("PREPARED_COMMITMENT_REWRITE")
            self._prepared().verify_prepared_subject(
                prefix_id=prefix_id, subject=existing.subject, artifact=existing.subject_artifact
            )
            return prefix
        artifact = self._prepared().publish_prepared_record(
            prefix_id=prefix_id, object_id=commitment.commitment_id, record=commitment
        )
        return self._append_prepared(
            prefix=prefix,
            kind=PreparedExecutionEventKind.COMMITMENT_STORED,
            slot_id=instance.task_slot.object_id,
            subject=ObjectIdentity.from_record(commitment.commitment_id, commitment),
            artifact=artifact,
            parent=parent,
            occurred_at_utc=occurred_at_utc,
        )

    def store_prepared_delivery(
        self, *, prefix_id: str, tick: DeliveryControllerTickReceipt
    ) -> ArtifactIdentity:
        prefix = self._prepared().load_prepared(prefix_id)
        commitment = ObjectIdentity.from_record(tick.commitment.commitment_id, tick.commitment)
        if not any(
            e.kind is PreparedExecutionEventKind.COMMITMENT_STORED and e.subject == commitment
            for e in prefix.events
        ):
            raise ProspectiveBindingConflict("PREPARED_DELIVERY_WITHOUT_DURABLE_COMMITMENT")
        if tick.commitment.disposition is not CommitmentDisposition.NONATTEMPT and not any(
            e.kind is PreparedExecutionEventKind.TASK_RESERVED and e.subject == commitment
            for e in prefix.events
        ):
            raise ProspectiveBindingConflict("PREPARED_DELIVERY_WITHOUT_TASK_RESERVATION")
        return self._prepared().publish_prepared_record(
            prefix_id=prefix_id, object_id=tick.tick_id, record=tick
        )

    def reserve_prepared_task(
        self,
        *,
        prefix_id: str,
        commitment: DeliveryControllerDecisionCommitment,
        occurred_at_utc: str,
    ) -> PreparedExecutionEventPrefix:
        prefix = self._prepared().load_prepared(prefix_id)
        parent = next(
            (e for e in prefix.events if e.kind is PreparedExecutionEventKind.COMMITMENT_STORED),
            None,
        )
        if (
            parent is None
            or parent.subject != ObjectIdentity.from_record(commitment.commitment_id, commitment)
            or commitment.disposition is CommitmentDisposition.NONATTEMPT
        ):
            raise ProspectiveBindingConflict("PREPARED_TASK_HAS_NO_EXECUTABLE_DURABLE_COMMITMENT")
        return self._append_prepared(
            prefix=prefix,
            kind=PreparedExecutionEventKind.TASK_RESERVED,
            slot_id=parent.slot_id,
            subject=parent.subject,
            artifact=parent.subject_artifact,
            parent=parent,
            occurred_at_utc=occurred_at_utc,
        )

    def initialize(
        self,
        *,
        prefix_id: str,
        coordinate_id: str,
        study: AdmissionControllerStudy,
        compiled: CompiledAdmissionControllerStudy,
        issued_at_utc: str,
        compiled_at_utc: str,
    ) -> ProspectiveExecutionEventPrefix:
        programme_identity = ObjectIdentity.from_record(study.study_id, study)
        compiled_identity = ObjectIdentity.from_record(
            compiled.compiled_study_id,
            compiled,
        )
        if compiled.study != study:
            raise ValueError("compiled programme substitutes another programme")
        prefix = ProspectiveExecutionEventPrefix.from_events(
            prefix_id=prefix_id,
            coordinate_id=coordinate_id,
            events=(
                ProspectiveExecutionEvent(
                    event_id=f"event.{coordinate_id}.programme-issued",
                    sequence_number=1,
                    kind=ProspectiveExecutionEventKind.PROGRAMME_ISSUED,
                    coordinate_id=coordinate_id,
                    subject=programme_identity,
                    parent=None,
                    occurred_at_utc=issued_at_utc,
                ),
                ProspectiveExecutionEvent(
                    event_id=f"event.{coordinate_id}.compiled",
                    sequence_number=2,
                    kind=ProspectiveExecutionEventKind.COMPILED,
                    coordinate_id=coordinate_id,
                    subject=compiled_identity,
                    parent=programme_identity,
                    occurred_at_utc=compiled_at_utc,
                ),
            ),
        )
        self._store.create(prefix)
        return prefix

    @staticmethod
    def _receipt(
        *,
        precommitment: ProspectiveEvaluationPrecommitment,
        study: AdmissionControllerStudy,
        compiled: CompiledAdmissionControllerStudy,
        prefix_before: ProspectiveExecutionEventPrefix,
        accepted_event: ProspectiveExecutionEvent,
        accepted_prefix: ProspectiveExecutionEventPrefix,
        bound_at_utc: str,
    ) -> ProspectiveEvaluationBindingReceipt:
        nested_plan = precommitment.nested_plan
        return ProspectiveEvaluationBindingReceipt(
            binding_id=f"prospective-binding.{prefix_before.coordinate_id}",
            coordinate_id=prefix_before.coordinate_id,
            issued_manifest=precommitment.issued_manifest,
            extension_set=precommitment.extension_set,
            sealed_acquisition_trace=precommitment.sealed_acquisition_trace,
            precommitment=precommitment,
            study=ObjectIdentity.from_record(study.study_id, study),
            compiled_study=ObjectIdentity.from_record(
                compiled.compiled_study_id,
                compiled,
            ),
            reference_design=precommitment.reference_design,
            nested_plan=ObjectIdentity.from_record(
                nested_plan.evaluation_plan_id,
                nested_plan,
            ),
            evaluator_binding=precommitment.evaluator_binding,
            causal_cutoff_id=precommitment.causal_cutoff_id,
            acquisition_domain_id=precommitment.acquisition_domain_id,
            reference_domain_id=precommitment.reference_domain_id,
            confirmatory_domain_id=precommitment.confirmatory_domain_id,
            prefix_before=prefix_before,
            accepted_event=accepted_event,
            accepted_prefix_root_sha256=accepted_prefix.event_root_sha256,
            bound_at_utc=bound_at_utc,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )

    def bind(
        self,
        *,
        prefix_id: str,
        precommitment: ProspectiveEvaluationPrecommitment,
        study: AdmissionControllerStudy,
        compiled: CompiledAdmissionControllerStudy,
        bound_at_utc: str,
    ) -> ProspectiveEvaluationBindingReceipt:
        if (
            study.prospective_evaluation is not None
            or compiled.prospective_evaluation_binding.disposition is not ProspectiveBindingDisposition.NOT_DECLARED
            or compiled.prospective_evaluation_binding.evaluation_plan is not None
            or compiled.prospective_evaluation_binding.evaluator is not None
        ):
            raise ValueError("action-aware prospective binding requires compiled controller use NOT_DECLARED")
        if compiled.study != study:
            raise ValueError("prospective binding substitutes another programme")
        if any(
            value.role is ImplementationRole.OUTCOME_EVALUATOR
            for value in study.implementations
        ):
            raise ValueError("programme capped at admission cannot embed the separate evaluator")
        current = self._store.load(prefix_id)
        programme_identity = ObjectIdentity.from_record(study.study_id, study)
        compiled_identity = ObjectIdentity.from_record(
            compiled.compiled_study_id,
            compiled,
        )
        expected_initial = (
            (ProspectiveExecutionEventKind.PROGRAMME_ISSUED, programme_identity, None),
            (ProspectiveExecutionEventKind.COMPILED, compiled_identity, programme_identity),
        )
        observed_initial = tuple(
            (value.kind, value.subject, value.parent) for value in current.events[:2]
        )
        if observed_initial != expected_initial:
            raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
        existing = next(
            (
                value
                for value in current.events
                if value.kind is ProspectiveExecutionEventKind.EVALUATOR_BOUND
            ),
            None,
        )
        if existing is not None:
            if (
                existing.subject != compiled_identity
                or existing.parent
                != ObjectIdentity.from_record(
                    precommitment.precommitment_id,
                    precommitment,
                )
                or any(
                    value.kind.protected for value in current.events[: existing.sequence_number - 1]
                )
            ):
                raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
            before = ProspectiveExecutionEventPrefix.from_events(
                prefix_id=current.prefix_id,
                coordinate_id=current.coordinate_id,
                events=current.events[: existing.sequence_number - 1],
            )
            accepted = ProspectiveExecutionEventPrefix.from_events(
                prefix_id=current.prefix_id,
                coordinate_id=current.coordinate_id,
                events=current.events[: existing.sequence_number],
            )
            return self._receipt(
                precommitment=precommitment,
                study=study,
                compiled=compiled,
                prefix_before=before,
                accepted_event=existing,
                accepted_prefix=accepted,
                bound_at_utc=existing.occurred_at_utc,
            )
        if len(current.events) != 2 or any(value.kind.protected for value in current.events):
            raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
        event = ProspectiveExecutionEvent(
            event_id=f"event.{current.coordinate_id}.evaluator-bound",
            sequence_number=3,
            kind=ProspectiveExecutionEventKind.EVALUATOR_BOUND,
            coordinate_id=current.coordinate_id,
            subject=compiled_identity,
            parent=ObjectIdentity.from_record(
                precommitment.precommitment_id,
                precommitment,
            ),
            occurred_at_utc=bound_at_utc,
        )
        updated = current.append(event)
        self._store.compare_and_append(
            expected_prefix_sha256=current.fingerprint(),
            updated=updated,
        )
        return self._receipt(
            precommitment=precommitment,
            study=study,
            compiled=compiled,
            prefix_before=current,
            accepted_event=event,
            accepted_prefix=updated,
            bound_at_utc=bound_at_utc,
        )

    def reserve_protected_event(
        self,
        *,
        prefix_id: str,
        receipt: ProspectiveEvaluationBindingReceipt,
        event_id: str,
        kind: ProspectiveExecutionEventKind,
        subject: ObjectIdentity,
        occurred_at_utc: str,
    ) -> ProspectiveExecutionEventPrefix:
        if not kind.protected:
            raise ValueError("requested event is not a protected execution event")
        current = self._store.load(prefix_id)
        bound = next(
            (
                value
                for value in current.events
                if value.kind is ProspectiveExecutionEventKind.EVALUATOR_BOUND
            ),
            None,
        )
        if (
            bound is None
            or bound != receipt.accepted_event
            or receipt.accepted_prefix_root_sha256
            != _prospective_prefix_sha256(current.events[: bound.sequence_number])
        ):
            raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
        event = ProspectiveExecutionEvent(
            event_id=event_id,
            sequence_number=len(current.events) + 1,
            kind=kind,
            coordinate_id=current.coordinate_id,
            subject=subject,
            parent=ObjectIdentity.from_record(receipt.binding_id, receipt),
            occurred_at_utc=occurred_at_utc,
        )
        updated = current.append(event)
        self._store.compare_and_append(
            expected_prefix_sha256=current.fingerprint(),
            updated=updated,
        )
        return updated


@dataclass(frozen=True, slots=True)
class SealedNestedPhysicalCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sealed-nested-physical-cell'

    physical_cell_id: str
    artifact: ArtifactIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.physical_cell_id, field_name="physical_cell_id")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("action-aware nested physical cell must remain sealed")


@dataclass(frozen=True, slots=True)
class SealedActionAwareControllerBundle(CanonicalRecord):
    """Sealed action-aware grid bound to the accepted prospective receipt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sealed-action-aware-controller-bundle'

    bundle_id: str
    evaluation_plan: ActionAwareControllerEvaluationPlan
    binding_receipt: ProspectiveEvaluationBindingReceipt
    compiled_study: ObjectIdentity
    ticks: tuple[AdmissionControllerTickReceipt, ...]
    physical_cells: tuple[SealedNestedPhysicalCell, ...]
    logical_locators: tuple[NestedEvaluationCellLocator, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        if (
            self.compiled_study != self.binding_receipt.compiled_study
            or self.binding_receipt.nested_plan
            != ObjectIdentity.from_record(
                self.evaluation_plan.evaluation_plan_id,
                self.evaluation_plan,
            )
        ):
            raise ValueError("action-aware nested bundle substitutes binding or plan")
        require_sorted_unique_ids(self.ticks, attribute="tick_id", field_name="ticks")
        require_sorted_unique_ids(
            self.physical_cells,
            attribute="physical_cell_id",
            field_name="physical_cells",
        )
        require_sorted_unique_ids(
            self.logical_locators,
            attribute="locator_id",
            field_name="logical_locators",
        )
        if self.logical_locators != self.evaluation_plan.cell_locators:
            raise ValueError("action-aware nested logical locator product differs from plan")
        expected_physical = {value.physical_cell_id for value in self.logical_locators}
        observed_physical = {value.physical_cell_id for value in self.physical_cells}
        if observed_physical != expected_physical or len(self.physical_cells) != len(
            expected_physical
        ):
            raise ValueError("action-aware nested physical cell product is incomplete")
        expected_units = {value.independent_unit_id for value in self.evaluation_plan.task_units}
        observed_units = {value.commitment.observation.independent_unit_id for value in self.ticks}
        if observed_units != expected_units or len(self.ticks) != len(expected_units):
            raise ValueError("action-aware nested tick roster differs from task units")
        if any(value.compiled_study != self.compiled_study for value in self.ticks):
            raise ValueError("action-aware nested tick binds another compiled programme")


@dataclass(frozen=True, slots=True)
class RevealedReferenceClassMembership(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-reference-class-membership'

    reference_id: str
    independent_unit_id: str
    reference_class: ReferenceDispositionClass
    membership: ReferenceClassMembershipReceipt

    def __post_init__(self) -> None:
        validate_stable_id(self.reference_id, field_name="reference_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        if self.membership.reference_class != ObjectIdentity.from_record(
            self.reference_class.class_id,
            self.reference_class,
        ):
            raise ValueError("revealed reference membership binds another class")


@dataclass(frozen=True, slots=True)
class RevealedNestedPhysicalCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-nested-physical-cell'

    reveal_id: str
    sealed_cell: ObjectIdentity
    physical_cell_id: str
    delivery_trace: ExactActionDeliveryTrace
    executed: bool
    response: NamedDecimal | None
    safe: bool
    delivery_valid: bool
    technical_valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.reveal_id, field_name="reveal_id")
        validate_stable_id(self.physical_cell_id, field_name="physical_cell_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.sealed_cell.object_schema != SealedNestedPhysicalCell.SCHEMA:
            raise ValueError("revealed nested cell binds another sealed schema")
        if not self.executed:
            if (
                self.response is not None
                or self.safe
                or self.delivery_valid
                or self.technical_valid
                or not self.reason_codes
            ):
                raise ValueError("zero-call nested cell must retain a typed skip")
        elif self.safe and self.delivery_valid and self.technical_valid:
            if self.response is None or self.reason_codes:
                raise ValueError("valid revealed nested cell lacks its response")
        elif not self.reason_codes:
            raise ValueError("invalid revealed nested cell requires a typed reason")


@dataclass(frozen=True, slots=True)
class RevealedActionAwareControllerBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-action-aware-controller-bundle'

    reveal_id: str
    sealed_bundle: SealedActionAwareControllerBundle
    references: tuple[RevealedReferenceClassMembership, ...]
    physical_cells: tuple[RevealedNestedPhysicalCell, ...]
    reveal_authorization: ObjectIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.reveal_id, field_name="reveal_id")
        require_sorted_unique_ids(
            self.references,
            attribute="independent_unit_id",
            field_name="references",
        )
        require_sorted_unique_ids(
            self.physical_cells,
            attribute="physical_cell_id",
            field_name="physical_cells",
        )
        expected_units = tuple(
            value.independent_unit_id for value in self.sealed_bundle.evaluation_plan.task_units
        )
        if tuple(value.independent_unit_id for value in self.references) != expected_units:
            raise ValueError("revealed action-aware nested references differ from task roster")
        sealed = {value.physical_cell_id: value for value in self.sealed_bundle.physical_cells}
        if set(sealed) != {value.physical_cell_id for value in self.physical_cells}:
            raise ValueError("revealed action-aware nested physical grid is incomplete")
        if any(
            value.sealed_cell
            != ObjectIdentity.from_record(
                sealed[value.physical_cell_id].physical_cell_id,
                sealed[value.physical_cell_id],
            )
            for value in self.physical_cells
        ):
            raise ValueError("revealed action-aware nested cell substitutes its seal")
        trace_ids = tuple(sorted(value.delivery_trace.trace_id for value in self.physical_cells))
        if len(trace_ids) != len(set(trace_ids)):
            raise ValueError("fresh confirmatory cells cannot reuse a delivery trace")
        tick_trace_ids = {value.delivery_trace.trace_id for value in self.sealed_bundle.ticks}
        if tick_trace_ids & set(trace_ids):
            raise ValueError("confirmatory evidence cannot copy the decision tick trace")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("revealed action-aware nested bundle is evaluator-only")


@dataclass(frozen=True, slots=True)
class PreparationMedianMemberEffect(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/preparation-median-member-effect'

    model_member_id: str
    preparation_count: int
    median_effect: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.model_member_id, field_name="model_member_id")
        if self.preparation_count != 3:
            raise ValueError("action-aware nested member effect requires exactly three preparations")


@dataclass(frozen=True, slots=True)
class ActionAwareControllerUnitEvaluation(CanonicalRecord):
    """One task row after action membership and median/member-min reduction."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/action-aware-controller-unit-evaluation'

    unit_evaluation_id: str
    independent_unit_id: str
    task_id: str
    arm_id: str
    evaluation_plan: ObjectIdentity
    binding_receipt: ObjectIdentity
    compiled_study: ObjectIdentity
    bundle: ObjectIdentity
    reference_membership: ReferenceClassMembershipReceipt
    disposition: NestedControllerUnitDisposition
    member_effects: tuple[PreparationMedianMemberEffect, ...]
    robust_effect: NamedDecimal | None
    independent_unit_count: int
    nested_preparation_count: int
    nested_member_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("unit_evaluation_id", self.unit_evaluation_id),
            ("independent_unit_id", self.independent_unit_id),
            ("task_id", self.task_id),
            ("arm_id", self.arm_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.member_effects,
            attribute="model_member_id",
            field_name="member_effects",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.independent_unit_count != 1:
            raise ValueError("nested coordinates cannot inflate the task count")
        if self.nested_preparation_count != 3:
            raise ValueError("action-aware controller unit requires three nested preparations")
        if self.nested_member_count < 1:
            raise ValueError("action-aware controller unit requires model members")
        if self.disposition is NestedControllerUnitDisposition.ACTIVE_EVALUABLE:
            if (
                self.robust_effect is None
                or len(self.member_effects) != self.nested_member_count
                or self.reason_codes
                or self.reference_membership.status is not ReferenceMembershipStatus.MEMBER
            ):
                raise ValueError("active action-aware controller unit is incomplete")
        elif self.robust_effect is not None or self.member_effects:
            raise ValueError("nonevaluable action-aware controller unit cannot retain effects")


@dataclass(frozen=True, slots=True)
class ActionAwareControllerCohortAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/action-aware-controller-cohort-adjudication'

    adjudication_id: str
    evaluation_plan: ObjectIdentity
    binding_receipt: ObjectIdentity
    compiled_study: ObjectIdentity
    bundle: ObjectIdentity
    units: tuple[ActionAwareControllerUnitEvaluation, ...]
    disposition_counts: tuple[NestedUnitDispositionCount, ...]
    rostered_independent_unit_count: int
    effective_independent_unit_count: int
    nested_preparation_count: int
    nested_member_count: int
    mean_effect: NamedDecimal | None
    active_coverage: Decimal
    result: ControllerUseResult
    claim_ceiling: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(
            self.units,
            attribute="independent_unit_id",
            field_name="units",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.rostered_independent_unit_count != len(self.units):
            raise ValueError("action-aware controller cohort roster count differs from task rows")
        if not 0 <= self.effective_independent_unit_count <= len(self.units):
            raise ValueError("action-aware controller cohort effective task count is invalid")
        validate_decimal(self.active_coverage, field_name="active_coverage", minimum=Decimal(0))
        if self.active_coverage > 1:
            raise ValueError("action-aware controller cohort active coverage cannot exceed one")


def _median_decimal(values: tuple[Decimal, ...]) -> Decimal:
    if not values or len(values) % 2 == 0:
        raise ValueError("registered nested median requires a nonempty odd roster")
    ordered = tuple(sorted(values))
    return ordered[len(ordered) // 2]


class ActionAwareNestedControllerUseEvaluator:
    "Pure stage-aware controller evaluator; never participates in compile or runtime decisions."

    def __init__(self, implementation_binding: ImplementationBinding) -> None:
        if implementation_binding.role is not ImplementationRole.OUTCOME_EVALUATOR:
            raise ValueError("action-aware evaluator requires the outcome-evaluator role")
        self.implementation_binding = implementation_binding

    def _validate_context(
        self,
        *,
        compiled: CompiledAdmissionControllerStudy,
        revealed: RevealedActionAwareControllerBundle,
    ) -> ActionAwareControllerEvaluationPlan:
        bundle = revealed.sealed_bundle
        receipt = bundle.binding_receipt
        if (
            compiled.prospective_evaluation_binding.disposition is not ProspectiveBindingDisposition.NOT_DECLARED
            or compiled.study.prospective_evaluation is not None
        ):
            raise ValueError("action-aware evaluator requires compiled controller capped at admission admission controller")
        compiled_identity = ObjectIdentity.from_record(
            compiled.compiled_study_id,
            compiled,
        )
        if bundle.compiled_study != compiled_identity:
            raise ValueError("action-aware bundle substitutes another compiled programme")
        if receipt.compiled_study != compiled_identity:
            raise ValueError("action-aware evaluator binding substitutes compiled identity")
        if receipt.evaluator_binding != self.implementation_binding:
            raise ValueError("action-aware evaluator implementation differs from receipt")
        plan = bundle.evaluation_plan
        if receipt.nested_plan != ObjectIdentity.from_record(plan.evaluation_plan_id, plan):
            raise ValueError("action-aware evaluator binding substitutes nested plan")
        return plan

    @staticmethod
    def _tick_for_unit(
        bundle: SealedActionAwareControllerBundle,
        independent_unit_id: str,
    ) -> AdmissionControllerTickReceipt:
        return next(
            value
            for value in bundle.ticks
            if value.commitment.observation.independent_unit_id == independent_unit_id
        )

    @staticmethod
    def _physical_semantics(
        plan: ActionAwareControllerEvaluationPlan,
    ) -> dict[str, NestedEvaluationCellLocator]:
        semantics: dict[str, NestedEvaluationCellLocator] = {}
        for locator in plan.cell_locators:
            prior = semantics.setdefault(locator.physical_cell_id, locator)
            if (
                prior.independent_unit_id,
                prior.occurrence_id,
                prior.action_word_id,
                prior.model_member_id,
            ) != (
                locator.independent_unit_id,
                locator.occurrence_id,
                locator.action_word_id,
                locator.model_member_id,
            ):
                raise ValueError("logical locator aliases change physical semantics")
        return semantics

    def evaluate_unit(
        self,
        *,
        compiled: CompiledAdmissionControllerStudy,
        revealed: RevealedActionAwareControllerBundle,
        independent_unit_id: str,
    ) -> ActionAwareControllerUnitEvaluation:
        plan = self._validate_context(compiled=compiled, revealed=revealed)
        unit = next(
            (
                value
                for value in plan.task_units
                if value.independent_unit_id == independent_unit_id
            ),
            None,
        )
        if unit is None:
            raise ValueError("action-aware controller unit is outside the frozen task roster")
        tick = self._tick_for_unit(revealed.sealed_bundle, independent_unit_id)
        reference = next(
            value
            for value in revealed.references
            if value.independent_unit_id == independent_unit_id
        )
        if reference.reference_class.reference_design != plan.finite_chart_reference_design:
            raise ValueError("revealed reference class substitutes another frozen design")
        committed_action_id = (
            tick.commitment.action_binding.action_word.word_id
            if tick.commitment.action_binding is not None
            else None
        )
        expected_membership = evaluate_reference_class_membership(
            reference_class=reference.reference_class,
            commitment_disposition=tick.commitment.disposition,
            committed_action_word_id=committed_action_id,
            receipt_id=reference.membership.receipt_id,
        )
        if reference.membership != expected_membership:
            raise ValueError("revealed reference membership is not exactly derived")

        def result_record(
            disposition: NestedControllerUnitDisposition,
            member_effects: tuple[PreparationMedianMemberEffect, ...],
            robust_effect: NamedDecimal | None,
            reason_codes: tuple[str, ...],
        ) -> ActionAwareControllerUnitEvaluation:
            return ActionAwareControllerUnitEvaluation(
                unit_evaluation_id=f"action-aware-controller-unit.{independent_unit_id}",
                independent_unit_id=independent_unit_id,
                task_id=unit.task_id,
                arm_id=unit.arm_id,
                evaluation_plan=ObjectIdentity.from_record(
                    plan.evaluation_plan_id,
                    plan,
                ),
                binding_receipt=ObjectIdentity.from_record(
                    revealed.sealed_bundle.binding_receipt.binding_id,
                    revealed.sealed_bundle.binding_receipt,
                ),
                compiled_study=ObjectIdentity.from_record(
                    compiled.compiled_study_id,
                    compiled,
                ),
                bundle=ObjectIdentity.from_record(
                    revealed.sealed_bundle.bundle_id,
                    revealed.sealed_bundle,
                ),
                reference_membership=reference.membership,
                disposition=disposition,
                member_effects=member_effects,
                robust_effect=robust_effect,
                independent_unit_count=1,
                nested_preparation_count=3,
                nested_member_count=len(plan.model_member_ids),
                reason_codes=reason_codes,
            )

        def terminal(
            disposition: NestedControllerUnitDisposition,
            reason: str,
        ) -> ActionAwareControllerUnitEvaluation:
            return result_record(disposition, (), None, (reason,))

        if reference.membership.status is not ReferenceMembershipStatus.MEMBER:
            return terminal(
                NestedControllerUnitDisposition.REFERENCE_INCORRECT,
                "COMMITMENT_OUTSIDE_REFERENCE_CLASS",
            )
        action_words = {value.word_id: value for value in plan.action_words}
        semantics = self._physical_semantics(plan)
        outcomes = {value.physical_cell_id: value for value in revealed.physical_cells}
        unit_cells = {
            cell_id: outcomes[cell_id]
            for cell_id, locator in semantics.items()
            if locator.independent_unit_id == independent_unit_id
        }
        executed = tuple(value for value in unit_cells.values() if value.executed)
        selected_action_ids: set[str] = set()
        if tick.commitment.disposition is CommitmentDisposition.ACTION_COMMITTED:
            if committed_action_id is not None:
                selected_action_ids.add(committed_action_id)
            if plan.measured_hold_word_id is not None:
                selected_action_ids.add(plan.measured_hold_word_id)
            if reference.reference_class.canonical_action_word_id is not None:
                selected_action_ids.add(reference.reference_class.canonical_action_word_id)
        elif tick.commitment.disposition is CommitmentDisposition.MEASURED_HOLD_COMMITTED:
            if plan.measured_hold_word_id is None:
                return terminal(
                    NestedControllerUnitDisposition.UNEVALUABLE,
                    "MEASURED_HOLD_WORD_ABSENT",
                )
            selected_action_ids.add(plan.measured_hold_word_id)
        expected_executed = {
            cell_id
            for cell_id, locator in semantics.items()
            if locator.independent_unit_id == independent_unit_id
            and locator.action_word_id in selected_action_ids
        }
        observed_executed = {cell_id for cell_id, value in unit_cells.items() if value.executed}
        if observed_executed != expected_executed:
            return terminal(
                NestedControllerUnitDisposition.UNEVALUABLE,
                "CONFIRMATORY_EXECUTED_UNION_DIFFERS_FROM_FROZEN_RULE",
            )
        if any(not value.safe for value in executed):
            return terminal(
                NestedControllerUnitDisposition.UNSAFE,
                "UNSAFE_CONFIRMATORY_CELL_PRESENT",
            )
        if any(not value.delivery_valid for value in executed):
            return terminal(
                NestedControllerUnitDisposition.DELIVERY_INVALID,
                "DELIVERY_INVALID_CONFIRMATORY_CELL_PRESENT",
            )
        if any(not value.technical_valid for value in executed):
            return terminal(
                NestedControllerUnitDisposition.TECHNICAL_INVALID,
                "TECHNICAL_INVALID_CONFIRMATORY_CELL_PRESENT",
            )
        for physical_id in observed_executed:
            locator = semantics[physical_id]
            cell = outcomes[physical_id]
            if cell.delivery_trace.expected_action_word != action_words[locator.action_word_id]:
                return terminal(
                    NestedControllerUnitDisposition.DELIVERY_INVALID,
                    "CONFIRMATORY_DELIVERY_ACTION_IDENTITY_MISMATCH",
                )
        if tick.commitment.disposition is CommitmentDisposition.NONATTEMPT:
            return terminal(
                NestedControllerUnitDisposition.NONATTEMPT,
                "CORRECT_NONATTEMPT",
            )
        if tick.commitment.disposition is CommitmentDisposition.MEASURED_HOLD_COMMITTED:
            return terminal(
                NestedControllerUnitDisposition.QUALIFIED_HOLD,
                "CORRECT_MEASURED_HOLD",
            )
        if committed_action_id is None or plan.measured_hold_word_id is None:
            return terminal(
                NestedControllerUnitDisposition.UNEVALUABLE,
                "QUALIFIED_HOLD_EFFICACY_BASELINE_ABSENT",
            )
        action_word = action_words[committed_action_id]
        hold_word = action_words[plan.measured_hold_word_id]
        occurrences = tuple(
            value
            for value in plan.preparation_occurrences
            if value.independent_unit_id == independent_unit_id
        )
        member_effects: list[PreparationMedianMemberEffect] = []
        required_trace_ids: set[str] = set()
        for member in plan.model_member_ids:
            effects: list[Decimal] = []
            for occurrence in occurrences:
                pair: dict[str, RevealedNestedPhysicalCell] = {}
                for requested_action in (committed_action_id, plan.measured_hold_word_id):
                    physical_id = next(
                        cell_id
                        for cell_id, locator in semantics.items()
                        if locator.independent_unit_id == independent_unit_id
                        and locator.occurrence_id == occurrence.occurrence_id
                        and locator.action_word_id == requested_action
                        and locator.model_member_id == member
                    )
                    pair[requested_action] = outcomes[physical_id]
                action_cell = pair[committed_action_id]
                hold_cell = pair[plan.measured_hold_word_id]
                if not action_cell.executed or not hold_cell.executed:
                    return terminal(
                        NestedControllerUnitDisposition.UNEVALUABLE,
                        "REQUIRED_CONFIRMATORY_ACTION_HOLD_GRID_INCOMPLETE",
                    )
                if (
                    action_cell.delivery_trace.expected_action_word != action_word
                    or hold_cell.delivery_trace.expected_action_word != hold_word
                ):
                    return terminal(
                        NestedControllerUnitDisposition.DELIVERY_INVALID,
                        "CONFIRMATORY_DELIVERY_ACTION_IDENTITY_MISMATCH",
                    )
                required_trace_ids.update(
                    {
                        action_cell.delivery_trace.trace_id,
                        hold_cell.delivery_trace.trace_id,
                    }
                )
                if (
                    action_cell.response is None
                    or hold_cell.response is None
                    or action_cell.response.unit != plan.effect_native_unit
                    or hold_cell.response.unit != plan.effect_native_unit
                    or action_cell.response.value_id != plan.effect_quantity_id
                    or hold_cell.response.value_id != plan.effect_quantity_id
                ):
                    return terminal(
                        NestedControllerUnitDisposition.UNEVALUABLE,
                        "CONFIRMATORY_EFFECT_MISSING_OR_WRONG_UNIT",
                    )
                effect = (
                    action_cell.response.value - hold_cell.response.value
                    if plan.favorable_direction is UtilityDirection.HIGHER_IS_BETTER
                    else hold_cell.response.value - action_cell.response.value
                )
                effects.append(effect)
            member_effects.append(
                PreparationMedianMemberEffect(
                    model_member_id=member,
                    preparation_count=len(effects),
                    median_effect=NamedDecimal(
                        value_id=f"member-median.{independent_unit_id}.{member}",
                        value=_median_decimal(tuple(effects)),
                        unit=plan.effect_native_unit,
                    ),
                )
            )
        if len(required_trace_ids) != 2 * len(occurrences) * len(plan.model_member_ids):
            return terminal(
                NestedControllerUnitDisposition.DELIVERY_INVALID,
                "CONFIRMATORY_DELIVERY_TRACE_REUSED",
            )
        ordered_effects = tuple(sorted(member_effects, key=lambda value: value.model_member_id))
        robust = NamedDecimal(
            value_id=f"robust-effect.{independent_unit_id}",
            value=min(value.median_effect.value for value in ordered_effects),
            unit=plan.effect_native_unit,
        )
        return result_record(
            NestedControllerUnitDisposition.ACTIVE_EVALUABLE,
            ordered_effects,
            robust,
            (),
        )

    def adjudicate_cohort(
        self,
        *,
        compiled: CompiledAdmissionControllerStudy,
        revealed: RevealedActionAwareControllerBundle,
        units: tuple[ActionAwareControllerUnitEvaluation, ...],
    ) -> ActionAwareControllerCohortAdjudication:
        plan = self._validate_context(compiled=compiled, revealed=revealed)
        expected = tuple(value.independent_unit_id for value in plan.task_units)
        if tuple(value.independent_unit_id for value in units) != expected:
            raise ValueError("action-aware controller cohort units differ from the frozen task roster")
        plan_identity = ObjectIdentity.from_record(plan.evaluation_plan_id, plan)
        receipt_identity = ObjectIdentity.from_record(
            revealed.sealed_bundle.binding_receipt.binding_id,
            revealed.sealed_bundle.binding_receipt,
        )
        compiled_identity = ObjectIdentity.from_record(
            compiled.compiled_study_id,
            compiled,
        )
        bundle_identity = ObjectIdentity.from_record(
            revealed.sealed_bundle.bundle_id,
            revealed.sealed_bundle,
        )
        if any(
            value.evaluation_plan != plan_identity
            or value.binding_receipt != receipt_identity
            or value.compiled_study != compiled_identity
            or value.bundle != bundle_identity
            or value.independent_unit_count != 1
            for value in units
        ):
            raise ValueError("action-aware controller cohort unit changes frozen identities or replication")
        counts = {
            disposition: sum(value.disposition is disposition for value in units)
            for disposition in NestedControllerUnitDisposition
        }
        effects = tuple(
            value.robust_effect.value
            for value in units
            if value.disposition is NestedControllerUnitDisposition.ACTIVE_EVALUABLE
            and value.robust_effect is not None
        )
        mean = (
            NamedDecimal(
                value_id=f"cohort-mean.{plan.evaluation_plan_id}",
                value=sum(effects, Decimal(0)) / Decimal(len(effects)),
                unit=plan.effect_native_unit,
            )
            if effects
            else None
        )
        active_count = counts[NestedControllerUnitDisposition.ACTIVE_EVALUABLE]
        active_coverage = Decimal(active_count) / Decimal(len(units))
        reasons: set[str] = set()
        if counts[NestedControllerUnitDisposition.UNSAFE]:
            result = ControllerUseResult.CONTROLLER_USE_UNSAFE
            reasons.add("UNSAFE_TASK_PRESENT")
        elif counts[NestedControllerUnitDisposition.DELIVERY_INVALID]:
            result = ControllerUseResult.CONTROLLER_USE_DELIVERY_INVALID
            reasons.add("DELIVERY_INVALID_TASK_PRESENT")
        elif counts[NestedControllerUnitDisposition.TECHNICAL_INVALID]:
            result = ControllerUseResult.CONTROLLER_USE_TECHNICAL_FAILURE
            reasons.add("TECHNICAL_INVALID_TASK_PRESENT")
        elif active_count < plan.minimum_evaluable_units:
            result = ControllerUseResult.CONTROLLER_USE_UNEVALUABLE
            reasons.add("MINIMUM_EVALUABLE_TASK_COUNT_NOT_MET")
        elif any(
            counts[value]
            for value in (
                NestedControllerUnitDisposition.REFERENCE_INCORRECT,
                NestedControllerUnitDisposition.PARTIAL_HETEROGENEOUS,
                NestedControllerUnitDisposition.UNEVALUABLE,
            )
        ):
            result = ControllerUseResult.CONTROLLER_USE_PARTIAL_OR_HETEROGENEOUS
            reasons.add("TASK_CORRECTNESS_OR_EVALUABILITY_HETEROGENEOUS")
        elif mean is not None and mean.value > plan.materiality.value:
            result = ControllerUseResult.CONTROLLER_USE_VALIDATED
        elif mean is not None and mean.value > 0:
            result = ControllerUseResult.CONTROLLER_USE_POSITIVE_BUT_BELOW_MATERIALITY
            reasons.add("FINITE_ROSTER_MEAN_BELOW_MATERIALITY")
        elif mean is not None:
            result = ControllerUseResult.CONTROLLER_USE_NEGATIVE
            reasons.add("FINITE_ROSTER_MEAN_NONPOSITIVE")
        else:
            result = ControllerUseResult.CONTROLLER_USE_UNEVALUABLE
            reasons.add("NO_EVALUABLE_ACTIVE_TASK_EFFECTS")
        return ActionAwareControllerCohortAdjudication(
            adjudication_id=f"action-aware-controller-cohort.{plan.evaluation_plan_id}",
            evaluation_plan=plan_identity,
            binding_receipt=receipt_identity,
            compiled_study=compiled_identity,
            bundle=bundle_identity,
            units=units,
            disposition_counts=tuple(
                NestedUnitDispositionCount(disposition=value, count=counts[value])
                for value in NestedControllerUnitDisposition
            ),
            rostered_independent_unit_count=len(units),
            effective_independent_unit_count=active_count,
            nested_preparation_count=3,
            nested_member_count=len(plan.model_member_ids),
            mean_effect=mean,
            active_coverage=active_coverage,
            result=result,
            claim_ceiling="FINITE_ROSTER_DETERMINISTIC_CONTROLLER_USE",
            reason_codes=tuple(sorted(reasons)),
        )


def decode_sealed_repeated_delivery_controller_bundle(payload: bytes) -> SealedRepeatedDeliveryControllerBundle:
    return decode_canonical_bytes(
        payload,
        SealedRepeatedDeliveryControllerBundle,
        maximum_bytes=MAX_NESTED_CONTROLLER_EVALUATION_BUNDLE_BYTES,
    )


def decode_revealed_repeated_delivery_controller_bundle(
    payload: bytes,
) -> RevealedRepeatedDeliveryControllerBundle:
    return decode_canonical_bytes(
        payload,
        RevealedRepeatedDeliveryControllerBundle,
        maximum_bytes=MAX_NESTED_CONTROLLER_EVALUATION_BUNDLE_BYTES,
    )


def decode_sealed_action_aware_controller_bundle(payload: bytes) -> SealedActionAwareControllerBundle:
    return decode_canonical_bytes(
        payload,
        SealedActionAwareControllerBundle,
        maximum_bytes=MAX_NESTED_CONTROLLER_EVALUATION_BUNDLE_BYTES,
    )


def decode_revealed_action_aware_controller_bundle(
    payload: bytes,
) -> RevealedActionAwareControllerBundle:
    return decode_canonical_bytes(
        payload,
        RevealedActionAwareControllerBundle,
        maximum_bytes=MAX_NESTED_CONTROLLER_EVALUATION_BUNDLE_BYTES,
    )


__all__ = [
    "ActionAwareNestedControllerUseEvaluator",
    'RepeatedDeliveryControllerCohortAdjudication',
    'ActionAwareControllerCohortAdjudication',
    'RepeatedDeliveryControllerUnitEvaluation',
    'ActionAwareControllerUnitEvaluation',
    "DurableProspectiveExecutionEventStore",
    "DispositionCorrectness",
    "MAX_NESTED_CONTROLLER_EVALUATION_BUNDLE_BYTES",
    "NestedControllerUseEvaluator",
    "NestedControllerUnitDisposition",
    'RepeatedDeliveryMemberEffect',
    'PreparationMedianMemberEffect',
    "NestedMemberRepeatDisposition",
    "NestedMemberRepeatEffect",
    "NestedRepeatDisposition",
    "NestedRepeatEvaluation",
    "NestedUnitDispositionCount",
    'SealedRepeatedDeliveryControllerBundle',
    'SealedActionAwareControllerBundle',
    "ProspectiveBindingConflict",
    "ProspectiveEvaluationBindingCoordinator",
    'ProspectiveEvaluationBindingReceipt',
    "ProspectiveExecutionEventKind",
    'ProspectiveExecutionEventPrefix',
    'ProspectiveExecutionEvent',
    "ReferenceSafetyDisposition",
    'RevealedReplicateDispositionReference',
    'RevealedReferenceClassMembership',
    "RevealedNestedBranchOutcome",
    'RevealedRepeatedDeliveryControllerBundle',
    'RevealedActionAwareControllerBundle',
    'RevealedNestedPhysicalCell',
    'SealedNestedPhysicalCell',
    "SealedNestedOutcomeLocator",
    "SealedNestedReferenceLocator",
    "SealedNestedReplicate",
    'decode_sealed_repeated_delivery_controller_bundle',
    'decode_sealed_action_aware_controller_bundle',
    'decode_revealed_repeated_delivery_controller_bundle',
    'decode_revealed_action_aware_controller_bundle',
    "InMemoryProspectiveExecutionEventStore",
]


class PreparedExecutionEventKind(StrEnum):
    DESIGN_FROZEN = "DESIGN_FROZEN"
    PARENT_RESERVED = "PARENT_RESERVED"
    PARENT_RETURNED = "PARENT_RETURNED"
    INSTANCE_BOUND = "INSTANCE_BOUND"
    COMMITMENT_STORED = "COMMITMENT_STORED"
    TASK_RESERVED = "TASK_RESERVED"
    TASK_COMPLETED = "TASK_COMPLETED"
    PROBE_RESERVED = "PROBE_RESERVED"
    PROBE_COMPLETED = "PROBE_COMPLETED"
    MATCHED_HOLD_RESERVED = "MATCHED_HOLD_RESERVED"
    MATCHED_HOLD_COMPLETED = "MATCHED_HOLD_COMPLETED"
    TERMINAL = "TERMINAL"


@dataclass(frozen=True, slots=True)
class PreparedExecutionEvent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-execution-event'

    event_id: str
    sequence_number: int
    kind: PreparedExecutionEventKind
    slot_id: str
    subject: ObjectIdentity
    subject_artifact: ArtifactIdentity
    parent: ObjectIdentity | None
    occurred_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.event_id, field_name="event_id")
        validate_stable_id(self.slot_id, field_name="slot_id")
        if self.sequence_number < 1 or not isinstance(self.kind, PreparedExecutionEventKind):
            raise ValueError("prepared event requires a positive sequence and typed kind")
        if (
            self.subject_artifact.sha256 != self.subject.object_fingerprint
            or self.subject_artifact.payload_schema != self.subject.object_schema
        ):
            raise ValueError("prepared event artifact differs from its exact canonical subject")
        schemas = {
            PreparedExecutionEventKind.DESIGN_FROZEN: PreparedDesignBindingReceipt.SCHEMA,
            PreparedExecutionEventKind.PARENT_RETURNED: PreparedParentReturn.SCHEMA,
            PreparedExecutionEventKind.INSTANCE_BOUND: PreparedInstanceBindingReceipt.SCHEMA,
            PreparedExecutionEventKind.COMMITMENT_STORED: DeliveryControllerDecisionCommitment.SCHEMA,
            PreparedExecutionEventKind.TASK_RESERVED: DeliveryControllerDecisionCommitment.SCHEMA,
            PreparedExecutionEventKind.PROBE_RESERVED: PreparedProbeCommitment.SCHEMA,
            PreparedExecutionEventKind.MATCHED_HOLD_RESERVED: PreparedFutureSlot.SCHEMA,
            PreparedExecutionEventKind.TASK_COMPLETED: PreparedFutureCompletion.SCHEMA,
            PreparedExecutionEventKind.PROBE_COMPLETED: PreparedFutureCompletion.SCHEMA,
            PreparedExecutionEventKind.MATCHED_HOLD_COMPLETED: PreparedFutureCompletion.SCHEMA,
            PreparedExecutionEventKind.TERMINAL: PreparedExecutionCensus.SCHEMA,
        }
        if self.kind in schemas and self.subject.object_schema != schemas[self.kind]:
            raise ValueError("prepared event kind substitutes an incompatible subject schema")
        parse_utc_timestamp(self.occurred_at_utc, field_name="occurred_at_utc")


@dataclass(frozen=True, slots=True)
class PreparedExecutionEventPrefix(CanonicalRecord):
    """Hash-linked root/policy log with separate parent and late-child boundaries."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-execution-event-prefix'

    prefix_id: str
    root_id: str
    policy_id: str
    events: tuple[PreparedExecutionEvent, ...]

    def __post_init__(self) -> None:
        for name in ("prefix_id", "root_id", "policy_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_unique_ids(self.events, attribute="event_id", field_name="events")
        if not self.events or len(self.events) > 256:
            raise ValueError("prepared event prefix must be nonempty and bounded")
        seen: dict[tuple[PreparedExecutionEventKind, str], PreparedExecutionEvent] = {}
        root_kinds: dict[PreparedExecutionEventKind, PreparedExecutionEvent] = {}
        k = PreparedExecutionEventKind
        prerequisite = {
            k.PARENT_RESERVED: k.DESIGN_FROZEN,
            k.PARENT_RETURNED: k.PARENT_RESERVED,
            k.INSTANCE_BOUND: k.PARENT_RETURNED,
            k.COMMITMENT_STORED: k.INSTANCE_BOUND,
            k.TASK_RESERVED: k.COMMITMENT_STORED,
            k.TASK_COMPLETED: k.TASK_RESERVED,
            k.PROBE_RESERVED: k.PARENT_RETURNED,
            k.PROBE_COMPLETED: k.PROBE_RESERVED,
            k.MATCHED_HOLD_RESERVED: k.PARENT_RETURNED,
            k.MATCHED_HOLD_COMPLETED: k.MATCHED_HOLD_RESERVED,
        }
        prior: PreparedExecutionEvent | None = None
        for index, event in enumerate(self.events, 1):
            if event.sequence_number != index:
                raise ValueError("prepared event sequence is not contiguous")
            if prior is not None and parse_utc_timestamp(
                event.occurred_at_utc, field_name="event time"
            ) < parse_utc_timestamp(prior.occurred_at_utc, field_name="prior time"):
                raise ValueError("prepared event chronology runs backwards")
            if (event.kind, event.slot_id) in seen or k.TERMINAL in root_kinds:
                raise ValueError("prepared event repeats a reservation or follows terminal closure")
            if index == 1:
                if event.kind is not k.DESIGN_FROZEN or event.parent is not None:
                    raise ValueError("prepared execution must begin with the frozen design")
            elif event.kind is k.DESIGN_FROZEN:
                raise ValueError("prepared design is immutable after its first event")
            elif event.kind is k.TERMINAL:
                if prior is None or event.parent != ObjectIdentity.from_record(
                    prior.event_id, prior
                ):
                    raise ValueError("terminal event must bind the exact preceding event")
            else:
                required = prerequisite[event.kind]
                predecessor = (
                    seen.get((required, event.slot_id))
                    if event.kind in {k.PROBE_COMPLETED, k.TASK_COMPLETED, k.MATCHED_HOLD_COMPLETED}
                    else root_kinds.get(required)
                )
                if predecessor is None or event.parent != ObjectIdentity.from_record(
                    predecessor.event_id, predecessor
                ):
                    raise ValueError("prepared event precedes or substitutes its causal boundary")
            if event.kind not in {k.PROBE_RESERVED, k.PROBE_COMPLETED}:
                if event.kind in root_kinds:
                    raise ValueError("root/policy has more than one parent or committed task")
                root_kinds[event.kind] = event
            seen[(event.kind, event.slot_id)] = event
            prior = event

    def append(self, event: PreparedExecutionEvent) -> PreparedExecutionEventPrefix:
        return PreparedExecutionEventPrefix(
            self.prefix_id, self.root_id, self.policy_id, (*self.events, event)
        )


class DurablePreparedExecutionEventStore(Protocol):
    def publish_prepared_record(
        self,
        *,
        prefix_id: str,
        object_id: str,
        record: CanonicalRecord,
    ) -> ArtifactIdentity: ...

    def verify_prepared_subject(
        self,
        *,
        prefix_id: str,
        subject: ObjectIdentity,
        artifact: ArtifactIdentity,
    ) -> None: ...

    def read_prepared_record(
        self,
        *,
        prefix_id: str,
        subject: ObjectIdentity,
        artifact: ArtifactIdentity,
    ) -> bytes: ...

    def create_prepared(self, prefix: PreparedExecutionEventPrefix) -> None: ...

    def load_prepared(self, prefix_id: str) -> PreparedExecutionEventPrefix: ...

    def compare_and_append_prepared(
        self,
        *,
        expected_prefix_sha256: str,
        updated: PreparedExecutionEventPrefix,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class PreparedDesignBindingReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-design-binding-receipt'

    binding_id: str
    issued_manifest: ObjectIdentity
    evaluation_plan: PreparedInterfaceEvaluationSlice
    root_id: str
    policy_id: str
    common_checkpoint: ObjectIdentity
    parent_commitment: ObjectIdentity
    evaluator_binding: ImplementationBinding
    frozen_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if self.root_id not in {r.root_id for r in self.evaluation_plan.roots}:
            raise ValueError("prepared binding substitutes its assigned root")
        if self.policy_id not in {p.policy_id for p in self.evaluation_plan.policies}:
            raise ValueError("prepared binding substitutes its assigned policy")
        reducer = self.evaluation_plan.reducer
        binding = self.evaluator_binding
        if (
            binding.role is not ImplementationRole.OUTCOME_EVALUATOR
            or binding.binding_id
            != self.evaluation_plan.evaluator_boundary.outcome_evaluator_binding_id
            or binding.reference.capability_key != reducer.capability_key
            or binding.reference.capability_version != reducer.capability_version
            or binding.config_sha256 != reducer.config_sha256
            or binding.implementation_sha256 != reducer.implementation_sha256
        ):
            raise ValueError("prepared design substitutes the frozen evaluator")
        parse_utc_timestamp(self.frozen_at_utc, field_name="frozen_at_utc")


@dataclass(frozen=True, slots=True)
class PreparedForecastParentCommitment(CanonicalRecord):
    """Parent command with a hash-bound child decision already fixed beforehand.

    Native execution unwraps only parent_action. decision/task are dependencies,
    not extra actuators; the later measured handoff is deliberately absent.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-forecast-parent-commitment'

    commitment_id: str
    evaluation_plan: ObjectIdentity
    root_id: str
    policy_id: str
    common_checkpoint: ObjectIdentity
    parent_action: ObjectIdentity
    decision: DeliveryControllerDecisionCommitment
    task: FiniteTaskFunctionalSpec
    locked_at_utc: str

    def __post_init__(self) -> None:
        for name in ("commitment_id", "root_id", "policy_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        parse_utc_timestamp(self.locked_at_utc, field_name="locked_at_utc")
        if self.evaluation_plan.object_schema not in (
            CommonStartControllerEvaluationPlan.SCHEMA,
            CoupledRealizationControllerEvaluationPlan.SCHEMA,
        ):
            raise ValueError("forecast parent requires its exact prepared evaluation plan")
        if self.decision.instance_binding.task_functional != ObjectIdentity.from_record(
            self.task.functional_id,
            self.task,
        ):
            raise ValueError("forecast parent changes the pre-parent request")

    def validate_binding(
        self,
        design: PreparedDesignBindingReceipt,
        compiled: CompiledDeliveryControllerStudy,
    ) -> None:
        policy = next(p for p in design.evaluation_plan.policies if p.policy_id == design.policy_id)
        if (
            design.parent_commitment != ObjectIdentity.from_record(self.commitment_id, self)
            or self.evaluation_plan != design.evaluation_plan.identity
            or self.root_id != design.root_id
            or self.policy_id != design.policy_id
            or self.common_checkpoint != design.common_checkpoint
            or parse_utc_timestamp(self.locked_at_utc, field_name="locked_at_utc")
            > parse_utc_timestamp(design.frozen_at_utc, field_name="frozen_at_utc")
            or self.decision.compiled_study
            != ObjectIdentity.from_record(compiled.compiled_study_id, compiled)
            or self.decision.instance_binding != compiled.study.instance_binding
            or (
                self.decision.selected_cell_action is not None
                and self.decision.selected_cell_action not in compiled.action_table
            )
            or compiled.study.prospective_evaluation != self.evaluation_plan
            or compiled.prospective_evaluation_binding.disposition is not ProspectiveBindingDisposition.BOUND
            or compiled.prospective_evaluation_binding.evaluation_plan != self.evaluation_plan
            or compiled.implementation(ImplementationRole.OUTCOME_EVALUATOR)
            != design.evaluator_binding
            or self.decision.instance_binding.frozen_recipe != policy.child_recipe
            or self.task.predeclared_family != design.evaluation_plan.target_family
        ):
            raise ValueError(
                "forecast parent substitutes its root, parent scope or frozen decision"
            )


@dataclass(frozen=True, slots=True)
class PreparedInstanceBindingReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-instance-binding-receipt'

    binding_id: str
    design_binding: ObjectIdentity
    parent_return: ObjectIdentity
    parent_return_event: ObjectIdentity
    public_handoff: ObjectIdentity
    target_reveal: ObjectIdentity
    task_functional: ObjectIdentity
    compiled_study: ObjectIdentity
    controller_instance: ObjectIdentity
    task_slot: ObjectIdentity
    bound_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        schemas = (
            (self.design_binding, PreparedDesignBindingReceipt.SCHEMA),
            (self.parent_return_event, PreparedExecutionEvent.SCHEMA),
            (self.task_functional, 'empirical-lawhood/planning/finite-task-functional-spec'),
            (self.controller_instance, 'empirical-lawhood/planning/controller-instance-binding'),
            (self.task_slot, PreparedFutureSlot.SCHEMA),
        )
        if (
            any(identity.object_schema != schema for identity, schema in schemas)
            or self.compiled_study.object_schema not in FINITE_COMPILED_COMMITMENT_SCHEMAS
        ):
            raise ValueError("prepared late binding uses an incompatible record version")
        parse_utc_timestamp(self.bound_at_utc, field_name="bound_at_utc")


class PreparedParentDisposition(StrEnum):
    HANDOFF_AVAILABLE = "HANDOFF_AVAILABLE"
    NO_HANDOFF = "NO_HANDOFF"
    INVALID_OBSERVATION = "INVALID_OBSERVATION"
    OPERATIONAL_STOP = "OPERATIONAL_STOP"
    COMPLETION_UNKNOWN = "COMPLETION_UNKNOWN"


@dataclass(frozen=True, slots=True)
class PreparedParentReturn(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-parent-return'

    return_id: str
    design_binding: ObjectIdentity
    common_checkpoint: ObjectIdentity
    parent_commitment: ObjectIdentity
    source_receipt: ObjectIdentity
    source_artifact: ArtifactIdentity
    public_handoff: ObjectIdentity | None
    disposition: PreparedParentDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.return_id, field_name="return_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.design_binding.object_schema != PreparedDesignBindingReceipt.SCHEMA:
            raise ValueError("prepared parent return requires its pre-parent design binding")
        if not isinstance(self.disposition, PreparedParentDisposition):
            raise ValueError("prepared parent disposition must be typed")
        ready = self.disposition is PreparedParentDisposition.HANDOFF_AVAILABLE
        if ready != (self.public_handoff is not None) or ready == bool(self.reason_codes):
            raise ValueError("prepared parent return/handoff/reasons are inconsistent")


class PreparedFutureDisposition(StrEnum):
    COMPLETED = "COMPLETED"
    NO_HANDOFF = "NO_HANDOFF"
    RANK_REFUSED = "RANK_REFUSED"
    INVALID_OBSERVATION = "INVALID_OBSERVATION"
    NONATTEMPT = "NONATTEMPT"
    OPERATIONAL_STOP = "OPERATIONAL_STOP"
    COMPLETION_UNKNOWN = "COMPLETION_UNKNOWN"


@dataclass(frozen=True, slots=True)
class PreparedProbeCommitment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-probe-commitment'

    probe_id: str
    slot: PreparedFutureSlot
    design_binding: ObjectIdentity
    parent_return: ObjectIdentity
    forecast: FiniteResponseSet

    def __post_init__(self) -> None:
        validate_stable_id(self.probe_id, field_name="probe_id")
        if (
            self.slot.role is not PreparedFutureRole.AUDIT_PROBE
            or self.slot.assigned_probe_word is None
        ):
            raise ValueError("prepared probe commitment requires its preassigned audit word")
        word = self.slot.assigned_probe_word
        if any(b.action_word != word for b in self.forecast.evaluation_bindings):
            raise ValueError("prepared probe forecast substitutes the preassigned native word")
        if self.design_binding.object_schema != PreparedDesignBindingReceipt.SCHEMA:
            raise ValueError("prepared probe lost its pre-parent design")
        if self.parent_return.object_schema != PreparedParentReturn.SCHEMA:
            raise ValueError("prepared probe lost its immutable handoff")


@dataclass(frozen=True, slots=True)
class SealedPreparedFutureLocator(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sealed-prepared-future-locator'

    locator_id: str
    slot_id: str
    view_id: str
    disposition: PreparedFutureDisposition
    source_receipt: ObjectIdentity | None
    artifact: ArtifactIdentity | None
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("locator_id", "slot_id", "view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not isinstance(self.disposition, PreparedFutureDisposition):
            raise ValueError("prepared future disposition must be typed")
        completed = self.disposition is PreparedFutureDisposition.COMPLETED
        if completed and (
            self.source_receipt is None or self.artifact is None or self.reason_codes
        ):
            raise ValueError("completed prepared future requires immutable native custody")
        if not completed and not self.reason_codes:
            raise ValueError("unexecuted or unresolved prepared future requires a reason")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("prepared future locator must remain sealed")


@dataclass(frozen=True, slots=True)
class SealedPreparedPolicyBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sealed-prepared-policy-bundle'
    FORECAST_LOCK_REQUIRED: ClassVar[bool] = False

    bundle_id: str
    design: PreparedDesignBindingReceipt
    parent_return: PreparedParentReturn
    instance: PreparedInstanceBindingReceipt | None
    task: FiniteTaskFunctionalSpec | None
    commitment: DeliveryControllerDecisionCommitment | None
    probes: tuple[PreparedProbeCommitment, ...]
    event_prefix: PreparedExecutionEventPrefix
    census: PreparedExecutionCensus
    completions: tuple[PreparedFutureCompletion, ...]

    @property
    def locators(self) -> tuple[SealedPreparedFutureLocator, ...]:
        return self.census.locators

    def __post_init__(self) -> None:
        if (
            self.design.parent_commitment.object_schema == PreparedForecastParentCommitment.SCHEMA
        ) != self.FORECAST_LOCK_REQUIRED:
            raise ValueError("prepared forecast design requires its explicit sealed lock bundle")
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        require_sorted_unique_ids(self.locators, attribute="locator_id", field_name="locators")
        require_sorted_unique_ids(self.probes, attribute="probe_id", field_name="probes")
        require_sorted_unique_ids(
            self.completions, attribute="completion_id", field_name="completions"
        )
        design_id = ObjectIdentity.from_record(self.design.binding_id, self.design)
        returned_id = ObjectIdentity.from_record(self.parent_return.return_id, self.parent_return)
        prefix = self.event_prefix
        plan = self.design.evaluation_plan
        if (
            prefix.root_id != self.design.root_id
            or prefix.policy_id != self.design.policy_id
            or prefix.events[0].subject != design_id
            or self.parent_return.design_binding != design_id
            or self.parent_return.common_checkpoint != self.design.common_checkpoint
            or self.parent_return.parent_commitment != self.design.parent_commitment
        ):
            raise ValueError("prepared bundle changes the root/policy/checkpoint/design")
        k = PreparedExecutionEventKind
        if (
            prefix.events[-1].kind is not k.TERMINAL
            or prefix.events[-1].subject
            != ObjectIdentity.from_record(self.census.census_id, self.census)
            or self.census.design_binding != design_id
        ):
            raise ValueError("sealed prepared bundle requires exact persisted terminal census")
        completed_events = {
            e.subject
            for e in prefix.events
            if e.kind in {k.TASK_COMPLETED, k.PROBE_COMPLETED, k.MATCHED_HOLD_COMPLETED}
        }
        if completed_events != {
            ObjectIdentity.from_record(c.completion_id, c) for c in self.completions
        }:
            raise ValueError("sealed bundle changes its known native completions")
        for completion in self.completions:
            if completion.locators != tuple(
                locator for locator in self.locators if locator.slot_id == completion.slot.object_id
            ):
                raise ValueError("sealed census changes immutable native completion locators")
        if not any(e.kind is k.PARENT_RETURNED and e.subject == returned_id for e in prefix.events):
            raise ValueError("prepared bundle lacks its persisted parent-return boundary")
        slots = {
            f.slot_id: f
            for f in plan.futures
            if f.root_id == self.design.root_id and f.policy_id == self.design.policy_id
        }
        expected = {(slot, view) for slot in slots for view in plan.numerical_view_ids}
        if {(locator.slot_id, locator.view_id) for locator in self.locators} != expected or len(
            self.locators
        ) != len(expected):
            raise ValueError("prepared bundle drops or duplicates an assigned future/view")
        ready = self.parent_return.disposition is PreparedParentDisposition.HANDOFF_AVAILABLE
        if not ready and any(
            locator.disposition is PreparedFutureDisposition.COMPLETED for locator in self.locators
        ):
            raise ValueError("prepared bundle fabricates a task/probe after absent handoff")
        if self.instance is None:
            if self.task is not None or self.commitment is not None:
                raise ValueError("prepared bundle supplies a child without its concrete binding")
        else:
            if self.task is None or self.commitment is None or not ready:
                raise ValueError("prepared instance requires its target, commitment and handoff")
            instance_id = ObjectIdentity.from_record(self.instance.binding_id, self.instance)
            commitment_id = ObjectIdentity.from_record(
                self.commitment.commitment_id, self.commitment
            )
            if (
                self.instance.design_binding != design_id
                or self.instance.parent_return != returned_id
                or self.instance.public_handoff != self.parent_return.public_handoff
                or self.instance.target_reveal != self.task.target_reveal
                or self.instance.task_functional
                != ObjectIdentity.from_record(self.task.functional_id, self.task)
                or self.instance.compiled_study != self.commitment.compiled_study
                or self.instance.controller_instance
                != ObjectIdentity.from_record(
                    self.commitment.instance_binding.instance_id, self.commitment.instance_binding
                )
                or not any(
                    e.kind is k.INSTANCE_BOUND and e.subject == instance_id for e in prefix.events
                )
                or not any(
                    e.kind is k.COMMITMENT_STORED and e.subject == commitment_id
                    for e in prefix.events
                )
            ):
                raise ValueError("prepared bundle breaks its immutable child/commitment chain")
        probe_slots: set[str] = set()
        for probe in self.probes:
            if (
                slots.get(probe.slot.slot_id) != probe.slot
                or probe.slot.slot_id in probe_slots
                or probe.design_binding != design_id
                or probe.parent_return != returned_id
                or probe.forecast.public_handoff != self.parent_return.public_handoff
                or not any(
                    e.kind is k.PROBE_RESERVED
                    and e.slot_id == probe.slot.slot_id
                    and e.subject == ObjectIdentity.from_record(probe.probe_id, probe)
                    for e in prefix.events
                )
            ):
                raise ValueError("prepared bundle substitutes an independently assigned probe")
            probe_slots.add(probe.slot.slot_id)
        for locator in self.locators:
            slot = slots[locator.slot_id]
            if locator.disposition is not PreparedFutureDisposition.COMPLETED:
                continue
            reservation = {
                PreparedFutureRole.COMMITTED_TASK: k.TASK_RESERVED,
                PreparedFutureRole.AUDIT_PROBE: k.PROBE_RESERVED,
                PreparedFutureRole.MATCHED_HOLD: k.MATCHED_HOLD_RESERVED,
            }[slot.role]
            if not any(e.kind is reservation and e.slot_id == slot.slot_id for e in prefix.events):
                raise ValueError("native future completion predates its durable reservation")
            if slot.role is PreparedFutureRole.AUDIT_PROBE and slot.slot_id not in probe_slots:
                raise ValueError("completed audit probe has no immutable pre-future forecast")
            if slot.role is PreparedFutureRole.MATCHED_HOLD and not any(
                e.kind is k.MATCHED_HOLD_RESERVED
                and e.slot_id == slot.slot_id
                and e.subject == ObjectIdentity.from_record(slot.slot_id, slot)
                for e in prefix.events
            ):
                raise ValueError("completed matched HOLD has no immutable evaluator reservation")
            if slot.role is PreparedFutureRole.COMMITTED_TASK:
                if (
                    self.commitment is None
                    or self.commitment.disposition is CommitmentDisposition.NONATTEMPT
                ):
                    raise ValueError("NONATTEMPT cannot acquire a task outcome")


@dataclass(frozen=True, slots=True)
class SealedPreparedForecastPolicyBundle(SealedPreparedPolicyBundle):
    """Unchanged native outcome census with the explicit pre-parent decision join."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sealed-prepared-forecast-policy-bundle'
    FORECAST_LOCK_REQUIRED: ClassVar[bool] = True

    forecast_parent: PreparedForecastParentCommitment

    def __post_init__(self) -> None:
        SealedPreparedPolicyBundle.__post_init__(self)
        lock = self.forecast_parent
        lock_id = ObjectIdentity.from_record(lock.commitment_id, lock)
        if (
            self.design.parent_commitment != lock_id
            or lock.evaluation_plan != self.design.evaluation_plan.identity
            or lock.root_id != self.design.root_id
            or lock.policy_id != self.design.policy_id
            or lock.common_checkpoint != self.design.common_checkpoint
            or parse_utc_timestamp(lock.locked_at_utc, field_name="locked_at_utc")
            > parse_utc_timestamp(self.design.frozen_at_utc, field_name="frozen_at_utc")
            or (
                self.instance is not None
                and (self.commitment != lock.decision or self.task != lock.task)
            )
            or not any(
                e.kind is PreparedExecutionEventKind.PARENT_RESERVED and e.subject == lock_id
                for e in self.event_prefix.events
            )
        ):
            raise ValueError("sealed forecast bundle changes its pre-parent dependency or decision")


@dataclass(frozen=True, slots=True)
class PreparedNativeReadout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-native-readout'

    coordinate: FiniteResponseCoordinate
    value: Decimal
    numerical_floor: Decimal

    def __post_init__(self) -> None:
        validate_decimal(self.value, field_name="value")
        validate_decimal(self.numerical_floor, field_name="numerical_floor", minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class RevealedPreparedFuture(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-prepared-future'

    outcome_id: str
    sealed_locator: ObjectIdentity
    slot_id: str
    view_id: str
    readouts: tuple[PreparedNativeReadout, ...]
    preservation_outcomes: tuple[NamedDecimal, ...]
    delivery_trace: ExactActionDeliveryTrace | StageAwareActionDeliveryTrace | None
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("outcome_id", "slot_id", "view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            tuple(r.coordinate.coordinate_id for r in self.readouts), field_name="native readouts"
        )
        require_sorted_unique_ids(
            self.preservation_outcomes, attribute="value_id", field_name="preservation_outcomes"
        )
        if self.sealed_locator.object_schema != SealedPreparedFutureLocator.SCHEMA:
            raise ValueError("prepared native outcome requires its exact sealed locator")
        if self.outcome_access not in {
            OutcomeAccess.EVALUATOR_REVEAL,
            OutcomeAccess.EVALUATION_REVEALED,
        }:
            raise ValueError("prepared native outcome requires separate reveal access")


@dataclass(frozen=True, slots=True)
class RevealedPreparedPolicyBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-prepared-policy-bundle'
    SEALED_TYPE: ClassVar[type[SealedPreparedPolicyBundle]] = SealedPreparedPolicyBundle

    reveal_id: str
    sealed: SealedPreparedPolicyBundle
    reveal_authorization: ObjectIdentity
    outcomes: tuple[RevealedPreparedFuture, ...]
    parent_preservation_outcomes: tuple[NamedDecimal, ...]

    def __post_init__(self) -> None:
        if type(self.sealed) is not self.SEALED_TYPE:
            raise ValueError("prepared reveal requires its exact sealed bundle schema")
        validate_stable_id(self.reveal_id, field_name="reveal_id")
        require_sorted_unique_ids(self.outcomes, attribute="outcome_id", field_name="outcomes")
        require_sorted_unique_ids(
            self.parent_preservation_outcomes,
            attribute="value_id",
            field_name="parent_preservation_outcomes",
        )
        locators = {(locator.slot_id, locator.view_id): locator for locator in self.sealed.locators}
        if {(o.slot_id, o.view_id) for o in self.outcomes} != set(locators) or len(
            self.outcomes
        ) != len(locators):
            raise ValueError("prepared reveal drops an assigned future/view")
        for outcome in self.outcomes:
            locator = locators[(outcome.slot_id, outcome.view_id)]
            if outcome.sealed_locator != ObjectIdentity.from_record(locator.locator_id, locator):
                raise ValueError("prepared reveal substitutes native custody")
            if locator.disposition is not PreparedFutureDisposition.COMPLETED:
                if (
                    outcome.readouts
                    or outcome.preservation_outcomes
                    or outcome.delivery_trace is not None
                ):
                    raise ValueError("unexecuted prepared future fabricates native observations")
            elif not outcome.readouts or outcome.delivery_trace is None:
                raise ValueError("completed prepared future lacks native readouts or delivery")


@dataclass(frozen=True, slots=True)
class RevealedPreparedForecastPolicyBundle(RevealedPreparedPolicyBundle):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-prepared-forecast-policy-bundle'
    SEALED_TYPE: ClassVar[type[SealedPreparedPolicyBundle]] = SealedPreparedForecastPolicyBundle

    sealed: SealedPreparedForecastPolicyBundle


@dataclass(frozen=True, slots=True)
class PreparedPolicyUnitEvaluation(CanonicalRecord):
    "All-assigned-root controller use and independent probe events, never view-level n."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-policy-unit-evaluation'

    evaluation_id: str
    root_id: str
    policy_id: str
    context_id: str
    problem_id: str
    common_checkpoint: ObjectIdentity
    evaluation_plan: ObjectIdentity
    revealed_bundle: ObjectIdentity
    independent_unit_count: int
    handoff_available: bool
    admitted: bool
    native_target: GateStatus
    delivery: GateStatus
    preservation: GateStatus
    numerics: GateStatus
    task_success: bool
    admitted_failure: bool
    probe_adequacy: GateStatus
    probe_coverage: GateStatus
    probe_sharpness: GateStatus
    dispositions: tuple[PreparedFutureDisposition, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("evaluation_id", "root_id", "policy_id", "context_id", "problem_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.independent_unit_count != 1:
            raise ValueError("prepared policy evaluation counts exactly one assigned root")
        expected = (
            self.handoff_available
            and self.admitted
            and all(
                value is GateStatus.PASS
                for value in (self.native_target, self.delivery, self.preservation, self.numerics)
            )
        )
        if self.task_success != expected or self.admitted_failure != (
            self.admitted and not expected
        ):
            raise ValueError("prepared task event is not the frozen conjunctive assigned-root rule")
        if not self.dispositions or tuple(sorted(set(self.dispositions))) != self.dispositions:
            raise ValueError(
                "prepared evaluation requires its complete distinct execution dispositions"
            )


@dataclass(frozen=True, slots=True)
class PreparedFutureCompletion(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-future-completion'

    completion_id: str
    slot: ObjectIdentity
    locators: tuple[SealedPreparedFutureLocator, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.completion_id, field_name="completion_id")
        require_sorted_unique_ids(self.locators, attribute="locator_id", field_name="locators")
        if self.slot.object_schema != PreparedFutureSlot.SCHEMA or not self.locators:
            raise ValueError("prepared future completion requires its exact assigned slot")
        if any(locator.slot_id != self.slot.object_id for locator in self.locators):
            raise ValueError("prepared completion mixes native futures")
        if any(locator.source_receipt is None for locator in self.locators):
            raise ValueError("known native completion requires a source receipt for every view")
        if any(
            locator.disposition is PreparedFutureDisposition.COMPLETION_UNKNOWN
            for locator in self.locators
        ):
            raise ValueError("unknown completion cannot masquerade as known source completion")


@dataclass(frozen=True, slots=True)
class PreparedExecutionCensus(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-execution-census'

    census_id: str
    design_binding: ObjectIdentity
    locators: tuple[SealedPreparedFutureLocator, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.census_id, field_name="census_id")
        require_sorted_unique_ids(self.locators, attribute="locator_id", field_name="locators")
        if (
            self.design_binding.object_schema != PreparedDesignBindingReceipt.SCHEMA
            or not self.locators
        ):
            raise ValueError("prepared census requires its pre-parent design and assigned futures")


@dataclass(frozen=True, slots=True)
class PreparedContextPolicyCensus(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-context-policy-census'

    census_id: str
    context_id: str
    policy_id: str
    assigned_root_ids: tuple[str, ...]
    admitted_count: int
    task_success_count: int
    admitted_failure_count: int
    probe_adequate_count: int
    probe_coverage_count: int
    probe_sharp_count: int
    unresolved_task_count: int

    def __post_init__(self) -> None:
        for name in ("census_id", "context_id", "policy_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.assigned_root_ids, field_name="assigned_root_ids", allow_empty=False
        )
        for count in (
            self.admitted_count,
            self.task_success_count,
            self.admitted_failure_count,
            self.probe_adequate_count,
            self.probe_coverage_count,
            self.probe_sharp_count,
            self.unresolved_task_count,
        ):
            if not 0 <= count <= len(self.assigned_root_ids):
                raise ValueError("prepared census inflates assigned-root counts")
        if self.task_success_count + self.admitted_failure_count != self.admitted_count:
            raise ValueError("prepared admission denominator drops unsuccessful admitted episodes")


def _prepared_censuses(
    units: tuple[PreparedPolicyUnitEvaluation, ...],
) -> tuple[PreparedContextPolicyCensus, ...]:
    groups = tuple(sorted({(unit.context_id, unit.policy_id) for unit in units}))
    result = []
    for context_id, policy_id in groups:
        grouped = tuple(
            unit for unit in units if unit.context_id == context_id and unit.policy_id == policy_id
        )
        result.append(
            PreparedContextPolicyCensus(
                census_id=f"prepared-census.{context_id}.{policy_id}",
                context_id=context_id,
                policy_id=policy_id,
                assigned_root_ids=tuple(sorted(unit.root_id for unit in grouped)),
                admitted_count=sum(unit.admitted for unit in grouped),
                task_success_count=sum(unit.task_success for unit in grouped),
                admitted_failure_count=sum(unit.admitted_failure for unit in grouped),
                probe_adequate_count=sum(
                    unit.probe_adequacy is GateStatus.PASS for unit in grouped
                ),
                probe_coverage_count=sum(
                    unit.probe_coverage is GateStatus.PASS for unit in grouped
                ),
                probe_sharp_count=sum(unit.probe_sharpness is GateStatus.PASS for unit in grouped),
                unresolved_task_count=sum(
                    unit.native_target is GateStatus.UNEVALUABLE for unit in grouped
                ),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class PreparedControllerCohortEvaluation(CanonicalRecord):
    """Exact root census; inferential promotion needs the separately frozen statistics."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-controller-cohort-evaluation'

    evaluation_id: str
    plan: CommonStartControllerEvaluationPlan | CoupledRealizationControllerEvaluationPlan
    evaluator: ImplementationBinding
    units: tuple[PreparedPolicyUnitEvaluation, ...]
    censuses: tuple[PreparedContextPolicyCensus, ...]
    independent_root_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        require_sorted_unique_ids(self.units, attribute="evaluation_id", field_name="units")
        require_sorted_unique_ids(self.censuses, attribute="census_id", field_name="censuses")
        roots = {root.root_id: root for root in self.plan.roots}
        expected = {
            (root_id, policy.policy_id) for root_id in roots for policy in self.plan.policies
        }
        if {(unit.root_id, unit.policy_id) for unit in self.units} != expected or len(
            self.units
        ) != len(expected):
            raise ValueError("prepared cohort omits or duplicates an assigned root/policy")
        if any(
            len({unit.common_checkpoint for unit in self.units if unit.root_id == root_id}) != 1
            for root_id in roots
        ):
            raise ValueError(
                "paired policies do not share the same immutable common-start checkpoint"
            )
        plan_id = ObjectIdentity.from_record(self.plan.evaluation_plan_id, self.plan)
        if any(
            unit.evaluation_plan != plan_id
            or unit.context_id != roots[unit.root_id].context_id
            or unit.problem_id != roots[unit.root_id].problem_id
            for unit in self.units
        ):
            raise ValueError("prepared cohort rewrites frozen context/problem assignments")
        if self.independent_root_count != len(roots) or self.censuses != _prepared_censuses(
            self.units
        ):
            raise ValueError(
                "prepared cohort counts are not derived from its complete assigned census"
            )
        if self.evaluator.binding_id != self.plan.evaluator_boundary.outcome_evaluator_binding_id:
            raise ValueError("prepared cohort substitutes its evaluator binding")


@dataclass(frozen=True, slots=True)
class PreparedCommonStartBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prepared-common-start-binding'

    binding_id: str
    evaluation_plan: ObjectIdentity
    root_assignment: ObjectIdentity
    common_checkpoint: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if (
            self.evaluation_plan.object_schema
            not in (
                CommonStartControllerEvaluationPlan.SCHEMA,
                CoupledRealizationControllerEvaluationPlan.SCHEMA,
            )
            or self.root_assignment.object_schema != 'empirical-lawhood/planning/prepared-root-assignment'
        ):
            raise ValueError(
                "common-start anchor requires its exact complete plan and assigned root"
            )

    @classmethod
    def for_design(cls, design: PreparedDesignBindingReceipt) -> PreparedCommonStartBinding:
        scope = design.evaluation_plan
        token = sha256(
            f"{scope.source_plan.object_fingerprint}:{scope.root.root_id}".encode()
        ).hexdigest()
        return cls(
            f"prepared-common-start.{token}",
            scope.source_plan,
            ObjectIdentity.from_record(scope.root.root_id, scope.root),
            design.common_checkpoint,
        )
