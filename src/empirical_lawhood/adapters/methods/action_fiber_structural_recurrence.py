'Action-fibered recurrence and measured-hold policy for action-fiber structural recurrence.\n\nThe module is a plan-scoped additive repair.  It reuses the frozen categorical structural recurrence\nontology and native unit records but never changes their schemas.  Receiver\nrestrictions are derived on one action fiber at a time, and ``hold`` is treated\nas an empirically evaluated fiber rather than as a safe default.\n'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.methods import structural_recurrence as core
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetDesignFreeze, StructuralRecurrenceTargetStage, _all_outcomes, _categorical_structure, _median
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)


ACTION_FIBER_VALID_STATE_COUNT = 10_000
ACTION_FIBER_PREDICTED_STATE_COUNT = 1
ACTION_FIBER_SHARPNESS = Decimal("0.9999")
_ONE_SIDED_95_Z = Decimal("1.6448536269514722")


class HoldViabilityDisposition(StrEnum):
    VIABLE = "VIABLE"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"
    ABSENT = "ABSENT"


class PolicyValidationDisposition(StrEnum):
    VALIDATED = "VALIDATED"
    OPPOSED = "OPPOSED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"


class ActionFiberStructuralRecurrenceTerminalVerdict(StrEnum):
    METHOD_NOT_QUALIFIED = "ACTION_FIBER_STRUCTURAL_RECURRENCE_METHOD_NOT_QUALIFIED"
    SAFETY_TYPING_OPPOSED = "ACTION_FIBER_STRUCTURAL_RECURRENCE_SAFETY_TYPING_OPPOSED"
    SUBSTRATE_LOCAL = "ACTION_FIBER_STRUCTURAL_RECURRENCE_SUBSTRATE_LOCAL"
    BOUNDED_TWO_TARGET_SAFETY_TYPED_RECURRENCE = (
        "ACTION_FIBER_STRUCTURAL_RECURRENCE_BOUNDED_TWO_TARGET_SAFETY_TYPED_RECURRENCE"
    )
    BOUNDED_THREE_TARGET_SAFETY_TYPED_RECURRENCE = (
        "ACTION_FIBER_STRUCTURAL_RECURRENCE_BOUNDED_THREE_TARGET_SAFETY_TYPED_RECURRENCE"
    )
    BOUNDED_FOUR_TARGET_SAFETY_TYPED_RECURRENCE = (
        "ACTION_FIBER_STRUCTURAL_RECURRENCE_BOUNDED_FOUR_TARGET_SAFETY_TYPED_RECURRENCE"
    )


@dataclass(frozen=True, slots=True)
class ActionFiberSignature(CanonicalRecord):
    'The complete admission intersection and restriction on one native action.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-signature'

    fiber_id: str
    target_slot_id: str
    stage: StructuralRecurrenceTargetStage
    action_id: str
    development_rank: int
    hold_semantics: bool
    requested_median: Decimal
    accepted_median: Decimal
    applied_median: Decimal
    realized_median: Decimal
    realization_role: core.ActionRole
    operands: tuple[core.AdmissionOperandFact, ...]
    receiver_role: core.ReceiverRole
    minimum_binary_lcb: Decimal
    admitted: bool
    independent_unit_count: int
    evidence: ObjectIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("fiber_id", "target_slot_id", "action_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.development_rank < 1 or self.independent_unit_count < 1:
            raise ValueError("action fiber requires positive rank and independent-unit count")
        if self.hold_semantics != (self.action_id == "hold"):
            raise ValueError("hold semantics must be attached only to the exact hold action")
        for name in (
            "requested_median",
            "accepted_median",
            "applied_median",
            "realized_median",
            "minimum_binary_lcb",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        require_sorted_unique_ids(self.operands, attribute="operand_id", field_name="operands")
        if tuple(value.role for value in self.operands) != core.ADMISSION_ROLES:
            raise ValueError('action fiber requires the exact ten-role admission intersection')
        expected_receiver = _receiver_role(self.operands)
        if self.receiver_role is not expected_receiver:
            raise ValueError("receiver restriction is not derived from this action fiber")
        expected_admitted = (
            self.realization_role is core.ActionRole.REALIZATION_QUALIFIED
            and all(value.status is core.OperandStatus.PASS for value in self.operands)
        )
        if self.admitted != expected_admitted:
            raise ValueError("action-fiber admission is not the noncompensating intersection")
        expected_access = (
            OutcomeAccess.DEVELOPMENT_VISIBLE
            if self.stage is StructuralRecurrenceTargetStage.DEVELOPMENT
            else OutcomeAccess.EVALUATOR_REVEAL
        )
        if self.outcome_access is not expected_access:
            raise ValueError("action fiber has the wrong outcome-access boundary")

    def operand(self, role: core.UniversalRole) -> core.AdmissionOperandFact:
        return next(value for value in self.operands if value.role is role)

    @property
    def category_key(self) -> tuple[object, ...]:
        return (
            self.action_id,
            self.realization_role,
            tuple(value.status for value in self.operands),
            self.receiver_role,
            self.admitted,
        )


@dataclass(frozen=True, slots=True)
class HoldViabilitySignature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/hold-viability-signature'

    signature_id: str
    hold_fiber: ActionFiberSignature
    disposition: HoldViabilityDisposition
    measured: bool
    default_assumed_safe: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.signature_id, field_name="signature_id")
        if not self.hold_fiber.hold_semantics:
            raise ValueError("hold viability requires the exact hold fiber")
        statuses = tuple(value.status for value in self.hold_fiber.operands)
        expected = (
            HoldViabilityDisposition.ABSENT
            if any(value is core.OperandStatus.ABSENT for value in statuses)
            else HoldViabilityDisposition.UNEVALUABLE
            if any(value is core.OperandStatus.UNEVALUABLE for value in statuses)
            or self.hold_fiber.realization_role is core.ActionRole.UNEVALUABLE
            else HoldViabilityDisposition.VIABLE
            if self.hold_fiber.admitted
            else HoldViabilityDisposition.OPPOSED
        )
        if self.disposition is not expected:
            raise ValueError("hold disposition is not derived from measured hold operands")
        if not self.measured or self.default_assumed_safe:
            raise ValueError('action-fiber structural recurrence forbids an unmeasured or default-safe hold')


@dataclass(frozen=True, slots=True)
class PolicySafetySignature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/policy-safety-signature'

    signature_id: str
    target_slot_id: str
    stage: StructuralRecurrenceTargetStage
    denominator_structure: core.DenominatorStructure
    action_fibers: tuple[ActionFiberSignature, ...]
    hold_viability: HoldViabilitySignature
    policy_branch: core.PolicyBranch
    selected_action_id: str
    selected_fiber_id: str
    primary_reason: core.StructuralReason
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.signature_id, field_name="signature_id")
        validate_stable_id(self.target_slot_id, field_name="target_slot_id")
        validate_stable_id(self.selected_action_id, field_name="selected_action_id")
        validate_stable_id(self.selected_fiber_id, field_name="selected_fiber_id")
        require_sorted_unique_ids(self.action_fibers, attribute="fiber_id", field_name="action_fibers")
        if any(value.hold_semantics for value in self.action_fibers):
            raise ValueError("non-hold action fibers cannot contain hold")
        if any(
            value.target_slot_id != self.target_slot_id
            or value.stage is not self.stage
            or value.outcome_access is not self.outcome_access
            for value in (*self.action_fibers, self.hold_viability.hold_fiber)
        ):
            raise ValueError("policy signature crosses target, stage, or access boundary")
        expected = _policy_decision(
            self.denominator_structure,
            self.action_fibers,
            self.hold_viability,
        )
        observed = (
            self.policy_branch,
            self.selected_action_id,
            self.selected_fiber_id,
            self.primary_reason,
        )
        if observed != expected:
            raise ValueError("policy signature differs from the action-fibered decision rule")

    @property
    def selected_fiber(self) -> ActionFiberSignature | None:
        return next(
            (value for value in self.action_fibers if value.action_id == self.selected_action_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrenceMethodFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-method-freeze'

    freeze_id: str
    plan: ObjectIdentity
    config: ObjectIdentity
    categorical_terminal_integrity_audit: ObjectIdentity
    method_source: ObjectIdentity
    runner_source: ObjectIdentity
    target_designs: tuple[ObjectIdentity, ...]
    required_fixture_count: int
    required_control_count: int
    binary_confidence_level: Decimal
    policy_validation_units_per_target: int
    action_identity_required: bool
    measured_hold_required: bool
    evaluator_frozen_before_target_execution: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        require_sorted_unique_ids(self.target_designs, attribute="object_id", field_name="target_designs")
        validate_decimal(self.binary_confidence_level, field_name="binary_confidence_level")
        if (
            len(self.target_designs) != 4
            or self.required_fixture_count != 24
            or self.required_control_count != 13
            or self.binary_confidence_level != Decimal("0.95")
            or self.policy_validation_units_per_target < 12
            or not self.action_identity_required
            or not self.measured_hold_required
            or not self.evaluator_frozen_before_target_execution
            or self.protected_outcome_access_count
        ):
            raise ValueError('action-fiber structural recurrence method freeze differs from the predeclared repair')


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrenceOperationalClosureFreeze(ActionFiberStructuralRecurrenceMethodFreeze):
    """Pre-target operational reissue with the complete executable closure."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-operational-closure-freeze'

    conformance_source: ObjectIdentity
    custody_source: ObjectIdentity


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrenceCanonicalClosureFreeze(ActionFiberStructuralRecurrenceMethodFreeze):
    """Canonical pre-target reissue with the complete executable closure."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-canonical-closure-freeze'

    conformance_source: ObjectIdentity
    custody_source: ObjectIdentity


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrenceVisibilityPreflightedClosureFreeze(ActionFiberStructuralRecurrenceMethodFreeze):
    """Visibility-preflighted reissue with the complete executable closure."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-visibility-preflighted-closure-freeze'

    conformance_source: ObjectIdentity
    custody_source: ObjectIdentity


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrenceFixtureExecution(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-fixture-execution'

    fixture_id: str
    fixture_kind: str
    expected_policy_branch: core.PolicyBranch
    observed_policy_branch: core.PolicyBranch
    expected_selected_action_id: str
    observed_selected_action_id: str
    safety_veto_passed: bool
    passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.fixture_id, field_name="fixture_id")
        validate_stable_id(self.fixture_kind, field_name="fixture_kind")
        validate_stable_id(self.expected_selected_action_id, field_name="expected_selected_action_id")
        validate_stable_id(self.observed_selected_action_id, field_name="observed_selected_action_id")
        expected = (
            self.expected_policy_branch is self.observed_policy_branch
            and self.expected_selected_action_id == self.observed_selected_action_id
            and self.safety_veto_passed
        )
        if self.passed != expected:
            raise ValueError("fixture pass is not mechanically derived")


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrenceControlResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-control-result'

    control_id: str
    control_kind: str
    observed_failure_code: str
    passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.control_id, field_name="control_id")
        validate_stable_id(self.control_kind, field_name="control_kind")
        validate_stable_id(self.observed_failure_code, field_name="observed_failure_code")
        if not self.passed:
            raise ValueError("failed negative control cannot qualify the method")


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrenceConformance(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-conformance'

    conformance_id: str
    method_freeze: ObjectIdentity
    execution_authority: ObjectIdentity
    fixtures: tuple[ActionFiberStructuralRecurrenceFixtureExecution, ...]
    controls: tuple[ActionFiberStructuralRecurrenceControlResult, ...]
    fixture_pass_count: int
    control_pass_count: int
    contradictory_positive_constructible: bool
    method_qualified: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.conformance_id, field_name="conformance_id")
        require_sorted_unique_ids(self.fixtures, attribute="fixture_id", field_name="fixtures")
        require_sorted_unique_ids(self.controls, attribute="control_id", field_name="controls")
        expected_qualified = (
            len(self.fixtures) == self.fixture_pass_count == 24
            and len(self.controls) == self.control_pass_count == 13
            and all(value.passed for value in self.fixtures)
            and all(value.passed for value in self.controls)
            and not self.contradictory_positive_constructible
        )
        if self.method_qualified != expected_qualified:
            raise ValueError("method qualification is not conformance-derived")
        if self.outcome_access is not OutcomeAccess.PRIVILEGED_TRUTH:
            raise ValueError("truth-known conformance requires privileged truth labels")


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrencePredictionIssue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-prediction-issue'

    issue_id: str
    method_freeze: ObjectIdentity
    conformance: ObjectIdentity
    target_design: ObjectIdentity
    development_evidence: ObjectIdentity
    target_slot_id: str
    denominator_structure: core.DenominatorStructure
    history_clock_quotient: core.HistoryClockQuotient
    support_transport: core.SupportTransport
    admission_topology: core.AdmissionTopology
    policy_safety: PolicySafetySignature
    prospective_validation_disposition: core.ProspectiveDisposition
    validation_disposition: PolicyValidationDisposition
    valid_state_count: int
    predicted_state_count: int
    sharpness: Decimal
    comparator_ids: tuple[str, ...]
    published_before_evaluation: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        validate_stable_id(self.target_slot_id, field_name="target_slot_id")
        require_sorted_unique_strings(self.comparator_ids, field_name="comparator_ids", allow_empty=False)
        if (
            self.policy_safety.target_slot_id != self.target_slot_id
            or self.policy_safety.stage is not StructuralRecurrenceTargetStage.DEVELOPMENT
        ):
            raise ValueError("prediction safety does not come from development")
        expected_prospective_validation, expected_validation = _expected_dispositions(self.policy_safety.policy_branch)
        if self.prospective_validation_disposition is not expected_prospective_validation or self.validation_disposition is not expected_validation:
            raise ValueError("prediction dispositions differ from its policy branch")
        if (
            self.valid_state_count != ACTION_FIBER_VALID_STATE_COUNT
            or self.predicted_state_count != ACTION_FIBER_PREDICTED_STATE_COUNT
            or self.sharpness != ACTION_FIBER_SHARPNESS
            or self.comparator_ids != ("ASSUME_HOLD_SAFE", 'CATEGORICAL_TARGET_WIDE_RESTRICTION')
            or not self.published_before_evaluation
            or self.protected_outcome_access_count
        ):
            raise ValueError("prediction issue differs from the frozen singleton contract")


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrenceAdmissionHandoff(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-admission-handoff'

    handoff_id: str
    method_freeze: ObjectIdentity
    prediction_issue: ObjectIdentity
    evaluation_evidence: ObjectIdentity
    reveal_authority: ObjectIdentity
    target_slot_id: str
    denominator_structure: core.DenominatorStructure
    history_clock_quotient: core.HistoryClockQuotient
    support_transport: core.SupportTransport
    admission_topology: core.AdmissionTopology
    policy_safety: PolicySafetySignature
    independent_unit_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.handoff_id, field_name="handoff_id")
        validate_stable_id(self.target_slot_id, field_name="target_slot_id")
        if (
            self.policy_safety.target_slot_id != self.target_slot_id
            or self.policy_safety.stage is not StructuralRecurrenceTargetStage.EVALUATION
            or self.independent_unit_count < 1
            or self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        ):
            raise ValueError('admission handoff lacks revealed independent evaluation evidence')
        if self.admission_topology is not _admission_topology(self.policy_safety.action_fibers):
            raise ValueError('admission topology is not action-fiber derived')


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrenceTargetResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-target-result'

    result_id: str
    admission_handoff: ActionFiberStructuralRecurrenceAdmissionHandoff
    validation_evidence: ObjectIdentity | None
    validation_authority: ObjectIdentity
    validation_policy_safety: PolicySafetySignature | None
    prospective_validation_disposition: core.ProspectiveDisposition
    validation_disposition: PolicyValidationDisposition
    policy_action_executed: bool
    hold_validation_executed: bool
    nonattempt_executed_neither: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        branch = self.admission_handoff.policy_safety.policy_branch
        expected_evidence = branch in {core.PolicyBranch.EXACT_ACTION, core.PolicyBranch.HOLD}
        if expected_evidence != (self.validation_evidence is not None):
            raise ValueError('policy validation evidence differs from the admission branch')
        if expected_evidence != (self.validation_policy_safety is not None):
            raise ValueError('policy validation signature differs from the admission branch')
        if self.validation_policy_safety is not None and self.validation_policy_safety.stage is not StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION:
            raise ValueError('policy validation must use the fresh prospective validation roster')
        if self.policy_action_executed != (branch is core.PolicyBranch.EXACT_ACTION):
            raise ValueError('exact-action execution differs from the admission branch')
        if self.hold_validation_executed != (branch is core.PolicyBranch.HOLD):
            raise ValueError('hold validation execution differs from the admission branch')
        if self.nonattempt_executed_neither != (branch is core.PolicyBranch.NONATTEMPT):
            raise ValueError("nonattempt must execute neither action nor hold")
        if branch is core.PolicyBranch.NONATTEMPT:
            expected_validation = PolicyValidationDisposition.NOT_ATTEMPTED
            expected_prospective_validation = core.ProspectiveDisposition.NONATTEMPT
        elif branch is core.PolicyBranch.HOLD:
            assert self.validation_policy_safety is not None
            viable = (
                self.validation_policy_safety.hold_viability.disposition
                is HoldViabilityDisposition.VIABLE
            )
            expected_validation = (
                PolicyValidationDisposition.VALIDATED
                if viable
                else PolicyValidationDisposition.OPPOSED
            )
            expected_prospective_validation = core.ProspectiveDisposition.CONDITION_FALSE
        else:
            assert self.validation_policy_safety is not None
            selected = self.validation_policy_safety.selected_fiber
            viable = (
                selected is not None
                and selected.admitted
                and self.validation_policy_safety.hold_viability.disposition
                is HoldViabilityDisposition.VIABLE
            )
            expected_validation = (
                PolicyValidationDisposition.VALIDATED
                if viable
                else PolicyValidationDisposition.OPPOSED
            )
            expected_prospective_validation = (
                core.ProspectiveDisposition.VALIDATED
                if viable
                else core.ProspectiveDisposition.OPPOSED
            )
        if self.validation_disposition is not expected_validation or self.prospective_validation_disposition is not expected_prospective_validation:
            raise ValueError("terminal policy disposition is not fresh-validation derived")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("target result requires evaluator reveal")


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrenceTargetMatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-target-match'

    match_id: str
    prediction_issue: ObjectIdentity
    target_result: ObjectIdentity
    target_slot_id: str
    core_field_match_count: int
    core_exact: bool
    selected_action_exact: bool
    selected_action_safety_exact: bool
    hold_safety_exact: bool
    all_action_fibers_exact: bool
    validation_exact: bool
    false_action_count: int
    hidden_sink_admission_count: int
    hold_default_error_count: int
    target_wide_comparator_rejected: bool
    assume_hold_safe_comparator_rejected: bool
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.match_id, field_name="match_id")
        validate_stable_id(self.target_slot_id, field_name="target_slot_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.core_field_match_count < 0 or self.core_field_match_count > 6:
            raise ValueError("core match count lies outside the six frozen fields")
        if self.core_exact != (self.core_field_match_count == 6):
            raise ValueError("core exactness is not six-field derived")
        if any(
            value not in {0, 1}
            for value in (
                self.false_action_count,
                self.hidden_sink_admission_count,
                self.hold_default_error_count,
            )
        ):
            raise ValueError("per-target safety errors must be binary")
        expected_pass = all(
            (
                self.core_exact,
                self.selected_action_exact,
                self.selected_action_safety_exact,
                self.hold_safety_exact,
                self.validation_exact,
                self.false_action_count == 0,
                self.hidden_sink_admission_count == 0,
                self.hold_default_error_count == 0,
            )
        )
        if self.passed != expected_pass:
            raise ValueError("positive target match is incompatible with its safety fields")


@dataclass(frozen=True, slots=True)
class ActionFiberStructuralRecurrenceAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-fiber-structural-recurrence-adjudication'

    adjudication_id: str
    method_conformance: ObjectIdentity
    target_matches: tuple[ObjectIdentity, ...]
    eligible_target_count: int
    passed_target_count: int
    failed_target_count: int
    false_action_count: int
    hidden_sink_admission_count: int
    hold_default_error_count: int
    action_fiber_defect_resolved: bool
    hold_viability_defect_resolved: bool
    target_wide_comparator_rejection_count: int
    assume_hold_safe_comparator_rejection_count: int
    verdict: ActionFiberStructuralRecurrenceTerminalVerdict
    positive_claim_eligible: bool
    same_substrate_classes_as_defect_discovery: bool
    independent_generality_claim_eligible: bool
    no_cross_target_pooling: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(self.target_matches, attribute="object_id", field_name="target_matches")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.eligible_target_count != len(self.target_matches):
            raise ValueError("eligible count differs from the target match roster")
        if self.passed_target_count + self.failed_target_count != self.eligible_target_count:
            raise ValueError("target pass/fail partition is incomplete")
        if any(
            value < 0
            for value in (
                self.false_action_count,
                self.hidden_sink_admission_count,
                self.hold_default_error_count,
                self.target_wide_comparator_rejection_count,
                self.assume_hold_safe_comparator_rejection_count,
            )
        ):
            raise ValueError("adjudication counts must be nonnegative")
        safety_clean = not (
            self.false_action_count
            or self.hidden_sink_admission_count
            or self.hold_default_error_count
        )
        if self.action_fiber_defect_resolved != (
            self.false_action_count == 0 and self.hidden_sink_admission_count == 0
        ):
            raise ValueError("action-fiber defect status is not safety-count derived")
        if self.hold_viability_defect_resolved != (self.hold_default_error_count == 0):
            raise ValueError("hold defect status is not measured-hold derived")
        expected_positive = (
            self.passed_target_count == self.eligible_target_count == 4 and safety_clean
        )
        if self.positive_claim_eligible != expected_positive:
            raise ValueError("positive claim eligibility conflicts with target safety")
        if (
            not self.same_substrate_classes_as_defect_discovery
            or self.independent_generality_claim_eligible
            or not self.no_cross_target_pooling
        ):
            raise ValueError('action-fiber structural recurrence cannot claim independent generality or pool substrates')


def wilson_lower_bound(successes: int, total: int) -> Decimal:
    """One-sided 95% Wilson lower bound in exact Decimal arithmetic."""

    if total < 1 or successes < 0 or successes > total:
        raise ValueError("invalid binomial count")
    n = Decimal(total)
    proportion = Decimal(successes) / n
    z2 = _ONE_SIDED_95_Z * _ONE_SIDED_95_Z
    denominator = Decimal(1) + z2 / n
    centre = proportion + z2 / (Decimal(2) * n)
    spread = _ONE_SIDED_95_Z * (
        (proportion * (Decimal(1) - proportion) / n + z2 / (Decimal(4) * n * n)).sqrt()
    )
    return ((centre - spread) / denominator).quantize(Decimal("0.000000000001"))


def _binary_status(values: tuple[bool, ...], minimum: Decimal) -> tuple[core.OperandStatus, Decimal]:
    lower = wilson_lower_bound(sum(values), len(values))
    return (core.OperandStatus.PASS if lower >= minimum else core.OperandStatus.FAIL, lower)


def _receiver_role(operands: tuple[core.AdmissionOperandFact, ...]) -> core.ReceiverRole:
    by_role = {value.role: value.status for value in operands}
    if by_role[core.UniversalRole.RECEIVER_SINK] is not core.OperandStatus.PASS:
        return core.ReceiverRole.SINK_LIMITING
    if by_role[core.UniversalRole.PRESERVATION] is not core.OperandStatus.PASS:
        return core.ReceiverRole.PRESERVATION_LIMITING
    if by_role[core.UniversalRole.RECEIVER_TARGET] is not core.OperandStatus.PASS:
        return core.ReceiverRole.TARGET_LIMITING
    return core.ReceiverRole.NONLIMITING


def _first_failure(fiber: ActionFiberSignature) -> core.StructuralReason:
    if fiber.realization_role is not core.ActionRole.REALIZATION_QUALIFIED:
        return core.StructuralReason.ACTION_REALIZATION
    return next(
        (
            core.StructuralReason.from_role(value.role)
            for value in fiber.operands
            if value.status is not core.OperandStatus.PASS
        ),
        core.StructuralReason.HOLD,
    )


def build_action_fiber(
    *,
    design: StructuralRecurrenceTargetDesignFreeze,
    evidence: StructuralRecurrenceStageEvidence,
    action_id: str,
    development_rank: int,
    outcome_access: OutcomeAccess,
) -> ActionFiberSignature:
    if evidence.stage is StructuralRecurrenceTargetStage.DEVELOPMENT:
        if outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("development fiber requires development-visible access")
    elif outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
        raise ValueError('evaluation/prospective validation fiber requires evaluator reveal')
    outcomes = _all_outcomes(evidence, action_id)
    minimum = design.threshold("unit-pass-rate-min")
    realization_status, realization_lcb = _binary_status(
        tuple(
            value.realization_error <= design.threshold("realization-error-max")
            for value in outcomes
        ),
        minimum,
    )
    realization = (
        core.ActionRole.REALIZATION_QUALIFIED
        if realization_status is core.OperandStatus.PASS
        else core.ActionRole.MISMATCH
    )
    binary: dict[core.UniversalRole, tuple[core.OperandStatus, Decimal]] = {
        core.UniversalRole.SUPPORT: _binary_status(
            tuple(value.support_preserved for value in outcomes), minimum
        ),
        core.UniversalRole.RECEIVER_SINK: _binary_status(
            tuple(value.sink_margin >= design.threshold("sink-margin-min") for value in outcomes),
            minimum,
        ),
        core.UniversalRole.EFFORT: _binary_status(
            tuple(value.effort <= design.threshold("effort-max") for value in outcomes), minimum
        ),
        core.UniversalRole.VALIDITY: _binary_status(
            tuple(value.validity_passed for value in outcomes), minimum
        ),
        core.UniversalRole.UNCERTAINTY: _binary_status(
            tuple(value.uncertainty_evaluable for value in outcomes), minimum
        ),
        core.UniversalRole.PRESERVATION: _binary_status(
            tuple(value.preservation_passed for value in outcomes), minimum
        ),
        core.UniversalRole.DYNAMICS: _binary_status(
            tuple(value.dynamics_passed for value in outcomes), minimum
        ),
        core.UniversalRole.REACHABILITY: _binary_status(
            tuple(value.reachability_passed for value in outcomes), minimum
        ),
        core.UniversalRole.AUTHORITY: _binary_status(
            tuple(value.authority_passed for value in outcomes), minimum
        ),
    }
    target_median = _median(tuple(value.target_effect for value in outcomes))
    target_pass = (
        target_median >= -design.threshold("target-effect-min")
        if action_id == "hold"
        else target_median >= design.threshold("target-effect-min")
    )
    evidence_identity = ObjectIdentity.from_record(evidence.evidence_id, evidence)
    operands = tuple(
        core.AdmissionOperandFact(
            operand_id=(
                f"{design.target_slot.target_slot_id}.{evidence.stage.value.lower()}."
                f"{action_id}.action-fiber.{index:02d}"
            ),
            role=role,
            status=(
                core.OperandStatus.PASS
                if role is core.UniversalRole.RECEIVER_TARGET and target_pass
                else core.OperandStatus.FAIL
                if role is core.UniversalRole.RECEIVER_TARGET
                else binary[role][0]
            ),
            evidence=evidence_identity,
            reason_codes=(
                ()
                if (
                    (role is core.UniversalRole.RECEIVER_TARGET and target_pass)
                    or (role is not core.UniversalRole.RECEIVER_TARGET and binary[role][0] is core.OperandStatus.PASS)
                )
                else (f"ACTION_FIBER_{role.value}_FAILED",)
            ),
        )
        for index, role in enumerate(core.ADMISSION_ROLES)
    )
    receiver = _receiver_role(operands)
    admitted = realization is core.ActionRole.REALIZATION_QUALIFIED and all(
        value.status is core.OperandStatus.PASS for value in operands
    )
    return ActionFiberSignature(
        fiber_id=(
            f"{design.target_slot.target_slot_id}.{evidence.stage.value.lower()}."
            f"{action_id}.action-fiber"
        ),
        target_slot_id=design.target_slot.target_slot_id,
        stage=evidence.stage,
        action_id=action_id,
        development_rank=development_rank,
        hold_semantics=action_id == "hold",
        requested_median=_median(tuple(value.requested_action for value in outcomes)),
        accepted_median=_median(tuple(value.accepted_action for value in outcomes)),
        applied_median=_median(tuple(value.applied_action for value in outcomes)),
        realized_median=_median(tuple(value.realized_action for value in outcomes)),
        realization_role=realization,
        operands=operands,
        receiver_role=receiver,
        minimum_binary_lcb=min(realization_lcb, *(value[1] for value in binary.values())),
        admitted=admitted,
        independent_unit_count=evidence.independent_unit_count,
        evidence=evidence_identity,
        outcome_access=outcome_access,
    )


def _admission_topology(fibers: tuple[ActionFiberSignature, ...]) -> core.AdmissionTopology:
    count = sum(value.admitted for value in fibers)
    return (
        core.AdmissionTopology.EMPTY
        if count == 0
        else core.AdmissionTopology.SINGLETON
        if count == 1
        else core.AdmissionTopology.MULTIPLE
    )


def _policy_decision(
    denominator: core.DenominatorStructure,
    action_fibers: tuple[ActionFiberSignature, ...],
    hold: HoldViabilitySignature,
) -> tuple[core.PolicyBranch, str, str, core.StructuralReason]:
    if denominator is core.DenominatorStructure.INCOMPATIBLE:
        return (
            core.PolicyBranch.NONATTEMPT,
            "nonattempt",
            "none",
            core.StructuralReason.PREPARED_DENOMINATOR,
        )
    admitted = tuple(value for value in action_fibers if value.admitted)
    if admitted:
        selected = min(admitted, key=lambda value: (value.development_rank, value.action_id))
        return core.PolicyBranch.EXACT_ACTION, selected.action_id, selected.fiber_id, core.StructuralReason.NONE
    if hold.disposition is HoldViabilityDisposition.VIABLE:
        reason = (
            _first_failure(min(action_fibers, key=lambda value: (value.development_rank, value.action_id)))
            if action_fibers
            else core.StructuralReason.HOLD
        )
        return core.PolicyBranch.HOLD, "hold", hold.hold_fiber.fiber_id, reason
    return (
        core.PolicyBranch.NONATTEMPT,
        "nonattempt",
        "none",
        _first_failure(hold.hold_fiber),
    )


def build_policy_signature(
    *,
    design: StructuralRecurrenceTargetDesignFreeze,
    evidence: StructuralRecurrenceStageEvidence,
    development_ranks: dict[str, int] | None = None,
    nonhold_action_ids: tuple[str, ...] | None = None,
    outcome_access: OutcomeAccess,
) -> tuple[PolicySafetySignature, core.HistoryClockQuotient, core.SupportTransport]:
    denominator, history, support = _categorical_structure(evidence, design)
    chart_nonhold = tuple(value for value in design.native_action_ids if value != "hold")
    nonhold = chart_nonhold if nonhold_action_ids is None else nonhold_action_ids
    if tuple(sorted(set(nonhold))) != nonhold or not set(nonhold) <= set(chart_nonhold):
        raise ValueError("validation action subset differs from the frozen non-hold chart")
    if development_ranks is None:
        ranked = sorted(
            (
                action_id,
                _median(tuple(value.target_effect for value in _all_outcomes(evidence, action_id))),
            )
            for action_id in nonhold
        )
        ranks = {
            action_id: index + 1
            for index, (action_id, _) in enumerate(
                sorted(ranked, key=lambda value: (-value[1], value[0]))
            )
        }
    else:
        if not set(nonhold) <= set(development_ranks):
            raise ValueError("development ranks omit a validation action")
        ranks = {value: development_ranks[value] for value in nonhold}
    fibers = tuple(
        sorted(
            (
                build_action_fiber(
                    design=design,
                    evidence=evidence,
                    action_id=action_id,
                    development_rank=ranks[action_id],
                    outcome_access=outcome_access,
                )
                for action_id in nonhold
            ),
            key=lambda value: value.fiber_id,
        )
    )
    hold_fiber = build_action_fiber(
        design=design,
        evidence=evidence,
        action_id="hold",
        development_rank=len(nonhold) + 1,
        outcome_access=outcome_access,
    )
    statuses = tuple(value.status for value in hold_fiber.operands)
    hold_disposition = (
        HoldViabilityDisposition.ABSENT
        if any(value is core.OperandStatus.ABSENT for value in statuses)
        else HoldViabilityDisposition.UNEVALUABLE
        if any(value is core.OperandStatus.UNEVALUABLE for value in statuses)
        else HoldViabilityDisposition.VIABLE
        if hold_fiber.admitted
        else HoldViabilityDisposition.OPPOSED
    )
    hold = HoldViabilitySignature(
        signature_id=(
            f"{design.target_slot.target_slot_id}.{evidence.stage.value.lower()}."
            'action-fiber.hold-viability'
        ),
        hold_fiber=hold_fiber,
        disposition=hold_disposition,
        measured=True,
        default_assumed_safe=False,
    )
    branch, selected, selected_fiber, reason = _policy_decision(denominator, fibers, hold)
    return (
        PolicySafetySignature(
            signature_id=(
                f"{design.target_slot.target_slot_id}.{evidence.stage.value.lower()}."
                'action-fiber.policy-safety'
            ),
            target_slot_id=design.target_slot.target_slot_id,
            stage=evidence.stage,
            denominator_structure=denominator,
            action_fibers=fibers,
            hold_viability=hold,
            policy_branch=branch,
            selected_action_id=selected,
            selected_fiber_id=selected_fiber,
            primary_reason=reason,
            outcome_access=outcome_access,
        ),
        history,
        support,
    )


def _expected_dispositions(
    branch: core.PolicyBranch,
) -> tuple[core.ProspectiveDisposition, PolicyValidationDisposition]:
    if branch is core.PolicyBranch.EXACT_ACTION:
        return core.ProspectiveDisposition.VALIDATED, PolicyValidationDisposition.VALIDATED
    if branch is core.PolicyBranch.HOLD:
        return core.ProspectiveDisposition.CONDITION_FALSE, PolicyValidationDisposition.VALIDATED
    return core.ProspectiveDisposition.NONATTEMPT, PolicyValidationDisposition.NOT_ATTEMPTED


def build_prediction_issue(
    *,
    issue_id: str,
    method_freeze: ActionFiberStructuralRecurrenceMethodFreeze,
    conformance: ActionFiberStructuralRecurrenceConformance,
    design: StructuralRecurrenceTargetDesignFreeze,
    development: StructuralRecurrenceStageEvidence,
) -> ActionFiberStructuralRecurrencePredictionIssue:
    policy, history, support = build_policy_signature(
        design=design,
        evidence=development,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    prospective_validation, validation = _expected_dispositions(policy.policy_branch)
    return ActionFiberStructuralRecurrencePredictionIssue(
        issue_id=issue_id,
        method_freeze=ObjectIdentity.from_record(method_freeze.freeze_id, method_freeze),
        conformance=ObjectIdentity.from_record(conformance.conformance_id, conformance),
        target_design=ObjectIdentity.from_record(design.design_id, design),
        development_evidence=ObjectIdentity.from_record(development.evidence_id, development),
        target_slot_id=design.target_slot.target_slot_id,
        denominator_structure=policy.denominator_structure,
        history_clock_quotient=history,
        support_transport=support,
        admission_topology=_admission_topology(policy.action_fibers),
        policy_safety=policy,
        prospective_validation_disposition=prospective_validation,
        validation_disposition=validation,
        valid_state_count=ACTION_FIBER_VALID_STATE_COUNT,
        predicted_state_count=ACTION_FIBER_PREDICTED_STATE_COUNT,
        sharpness=ACTION_FIBER_SHARPNESS,
        comparator_ids=("ASSUME_HOLD_SAFE", 'CATEGORICAL_TARGET_WIDE_RESTRICTION'),
        published_before_evaluation=True,
        protected_outcome_access_count=0,
    )


def evaluate_admission(
    *,
    method_freeze: ActionFiberStructuralRecurrenceMethodFreeze,
    prediction: ActionFiberStructuralRecurrencePredictionIssue,
    design: StructuralRecurrenceTargetDesignFreeze,
    evaluation: StructuralRecurrenceStageEvidence,
    reveal_authority: ObjectIdentity,
) -> ActionFiberStructuralRecurrenceAdmissionHandoff:
    if evaluation.stage is not StructuralRecurrenceTargetStage.EVALUATION:
        raise ValueError('admission requires the sealed evaluation roster')
    ranks = {value.action_id: value.development_rank for value in prediction.policy_safety.action_fibers}
    policy, history, support = build_policy_signature(
        design=design,
        evidence=evaluation,
        development_ranks=ranks,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    return ActionFiberStructuralRecurrenceAdmissionHandoff(
        handoff_id=f"{design.target_slot.target_slot_id}.action-fiber-admission-handoff",
        method_freeze=ObjectIdentity.from_record(method_freeze.freeze_id, method_freeze),
        prediction_issue=ObjectIdentity.from_record(prediction.issue_id, prediction),
        evaluation_evidence=ObjectIdentity.from_record(evaluation.evidence_id, evaluation),
        reveal_authority=reveal_authority,
        target_slot_id=design.target_slot.target_slot_id,
        denominator_structure=policy.denominator_structure,
        history_clock_quotient=history,
        support_transport=support,
        admission_topology=_admission_topology(policy.action_fibers),
        policy_safety=policy,
        independent_unit_count=evaluation.independent_unit_count,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )


def finalize_target(
    *,
    admission_handoff: ActionFiberStructuralRecurrenceAdmissionHandoff,
    design: StructuralRecurrenceTargetDesignFreeze,
    validation_evidence: StructuralRecurrenceStageEvidence | None,
    validation_authority: ObjectIdentity,
) -> ActionFiberStructuralRecurrenceTargetResult:
    branch = admission_handoff.policy_safety.policy_branch
    validation_policy = None
    if validation_evidence is not None:
        ranks = {value.action_id: value.development_rank for value in admission_handoff.policy_safety.action_fibers}
        validation_policy, _, _ = build_policy_signature(
            design=design,
            evidence=validation_evidence,
            development_ranks=ranks,
            nonhold_action_ids=(admission_handoff.policy_safety.selected_action_id,)
            if branch is core.PolicyBranch.EXACT_ACTION
            else (),
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        )
        if branch is core.PolicyBranch.EXACT_ACTION:
            selected = validation_policy.selected_fiber
            viable = (
                selected is not None
                and selected.action_id == admission_handoff.policy_safety.selected_action_id
                and selected.admitted
                and validation_policy.hold_viability.disposition is HoldViabilityDisposition.VIABLE
            )
            validation = (
                PolicyValidationDisposition.VALIDATED
                if viable
                else PolicyValidationDisposition.OPPOSED
            )
            prospective_validation = core.ProspectiveDisposition.VALIDATED if viable else core.ProspectiveDisposition.OPPOSED
        else:
            viable = validation_policy.hold_viability.disposition is HoldViabilityDisposition.VIABLE
            validation = (
                PolicyValidationDisposition.VALIDATED
                if viable
                else PolicyValidationDisposition.OPPOSED
            )
            prospective_validation = core.ProspectiveDisposition.CONDITION_FALSE
    else:
        validation = PolicyValidationDisposition.NOT_ATTEMPTED
        prospective_validation = core.ProspectiveDisposition.NONATTEMPT
    return ActionFiberStructuralRecurrenceTargetResult(
        result_id=f"{design.target_slot.target_slot_id}.action-fiber-target-result",
        admission_handoff=admission_handoff,
        validation_evidence=(
            ObjectIdentity.from_record(validation_evidence.evidence_id, validation_evidence)
            if validation_evidence is not None
            else None
        ),
        validation_authority=validation_authority,
        validation_policy_safety=validation_policy,
        prospective_validation_disposition=prospective_validation,
        validation_disposition=validation,
        policy_action_executed=branch is core.PolicyBranch.EXACT_ACTION,
        hold_validation_executed=branch is core.PolicyBranch.HOLD,
        nonattempt_executed_neither=branch is core.PolicyBranch.NONATTEMPT,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )


def match_target(
    prediction: ActionFiberStructuralRecurrencePredictionIssue,
    result: ActionFiberStructuralRecurrenceTargetResult,
) -> ActionFiberStructuralRecurrenceTargetMatch:
    observed = result.admission_handoff
    predicted_policy = prediction.policy_safety
    observed_policy = observed.policy_safety
    core_pairs = (
        (prediction.denominator_structure, observed.denominator_structure),
        (prediction.history_clock_quotient, observed.history_clock_quotient),
        (prediction.support_transport, observed.support_transport),
        (prediction.admission_topology, observed.admission_topology),
        (predicted_policy.policy_branch, observed_policy.policy_branch),
        (prediction.prospective_validation_disposition, result.prospective_validation_disposition),
    )
    core_count = sum(left is right for left, right in core_pairs)
    selected_exact = predicted_policy.selected_action_id == observed_policy.selected_action_id
    predicted_selected = predicted_policy.selected_fiber
    observed_selected = observed_policy.selected_fiber
    selected_safety_exact = (
        predicted_policy.policy_branch is not core.PolicyBranch.EXACT_ACTION
        and observed_policy.policy_branch is not core.PolicyBranch.EXACT_ACTION
    ) or (
        predicted_selected is not None
        and observed_selected is not None
        and predicted_selected.category_key == observed_selected.category_key
    )
    hold_exact = (
        predicted_policy.hold_viability.disposition
        is observed_policy.hold_viability.disposition
        and predicted_policy.hold_viability.hold_fiber.category_key
        == observed_policy.hold_viability.hold_fiber.category_key
    )
    predicted_fibers = {value.action_id: value.category_key for value in predicted_policy.action_fibers}
    observed_fibers = {value.action_id: value.category_key for value in observed_policy.action_fibers}
    all_fibers_exact = predicted_fibers == observed_fibers
    validation_exact = prediction.validation_disposition is result.validation_disposition
    observed_selected_safe = (
        observed_policy.policy_branch is core.PolicyBranch.EXACT_ACTION
        and observed_selected is not None
        and observed_selected.admitted
        and observed_selected.receiver_role is core.ReceiverRole.NONLIMITING
    )
    false_action = int(
        predicted_policy.policy_branch is core.PolicyBranch.EXACT_ACTION
        and (not observed_selected_safe or not selected_exact)
    )
    hidden_sink = int(
        predicted_policy.policy_branch is core.PolicyBranch.EXACT_ACTION
        and observed_selected is not None
        and observed_selected.receiver_role
        in {core.ReceiverRole.SINK_LIMITING, core.ReceiverRole.PRESERVATION_LIMITING}
    )
    hold_default_error = int(
        observed_policy.policy_branch is core.PolicyBranch.HOLD
        and observed_policy.hold_viability.disposition is not HoldViabilityDisposition.VIABLE
    )
    target_wide_receiver = (
        core.ReceiverRole.SINK_LIMITING
        if any(
            value.receiver_role is core.ReceiverRole.SINK_LIMITING
            for value in observed_policy.action_fibers
        )
        else core.ReceiverRole.PRESERVATION_LIMITING
        if any(
            value.receiver_role is core.ReceiverRole.PRESERVATION_LIMITING
            for value in observed_policy.action_fibers
        )
        else core.ReceiverRole.TARGET_LIMITING
        if any(
            value.receiver_role is core.ReceiverRole.TARGET_LIMITING
            for value in observed_policy.action_fibers
        )
        else core.ReceiverRole.NONLIMITING
    )
    comparator_target_wide_rejected = (
        observed_selected is not None
        and observed_selected.receiver_role is core.ReceiverRole.NONLIMITING
        and target_wide_receiver is not core.ReceiverRole.NONLIMITING
    )
    comparator_hold_rejected = (
        observed_policy.hold_viability.disposition is not HoldViabilityDisposition.VIABLE
    )
    reasons: set[str] = set()
    if core_count != 6:
        reasons.add("CORE_COVERAGE_FAILED")
    if not selected_exact:
        reasons.add("ACTION_IDENTITY_MISMATCH")
    if not selected_safety_exact:
        reasons.add("SELECTED_ACTION_SAFETY_MISMATCH")
    if not hold_exact:
        reasons.add("HOLD_VIABILITY_MISMATCH")
    if not validation_exact:
        reasons.add("POLICY_VALIDATION_MISMATCH")
    if false_action:
        reasons.add("FALSE_ACTION")
    if hidden_sink:
        reasons.add("HIDDEN_SINK_ADMISSION")
    if hold_default_error:
        reasons.add("HOLD_DEFAULT_ERROR")
    passed = all(
        (
            core_count == 6,
            selected_exact,
            selected_safety_exact,
            hold_exact,
            validation_exact,
            not false_action,
            not hidden_sink,
            not hold_default_error,
        )
    )
    return ActionFiberStructuralRecurrenceTargetMatch(
        match_id=f"{prediction.target_slot_id}.action-fiber-target-match",
        prediction_issue=ObjectIdentity.from_record(prediction.issue_id, prediction),
        target_result=ObjectIdentity.from_record(result.result_id, result),
        target_slot_id=prediction.target_slot_id,
        core_field_match_count=core_count,
        core_exact=core_count == 6,
        selected_action_exact=selected_exact,
        selected_action_safety_exact=selected_safety_exact,
        hold_safety_exact=hold_exact,
        all_action_fibers_exact=all_fibers_exact,
        validation_exact=validation_exact,
        false_action_count=false_action,
        hidden_sink_admission_count=hidden_sink,
        hold_default_error_count=hold_default_error,
        target_wide_comparator_rejected=comparator_target_wide_rejected,
        assume_hold_safe_comparator_rejected=comparator_hold_rejected,
        passed=passed,
        reason_codes=tuple(sorted(reasons)),
    )


def adjudicate(
    *,
    conformance: ActionFiberStructuralRecurrenceConformance,
    matches: tuple[ActionFiberStructuralRecurrenceTargetMatch, ...],
) -> ActionFiberStructuralRecurrenceAdjudication:
    ordered = tuple(sorted(matches, key=lambda value: value.match_id))
    passed = sum(value.passed for value in ordered)
    false_actions = sum(value.false_action_count for value in ordered)
    hidden_sinks = sum(value.hidden_sink_admission_count for value in ordered)
    hold_errors = sum(value.hold_default_error_count for value in ordered)
    safety_clean = not (false_actions or hidden_sinks or hold_errors)
    if not conformance.method_qualified:
        verdict = ActionFiberStructuralRecurrenceTerminalVerdict.METHOD_NOT_QUALIFIED
    elif not safety_clean:
        verdict = ActionFiberStructuralRecurrenceTerminalVerdict.SAFETY_TYPING_OPPOSED
    elif passed == 4:
        verdict = ActionFiberStructuralRecurrenceTerminalVerdict.BOUNDED_FOUR_TARGET_SAFETY_TYPED_RECURRENCE
    elif passed == 3:
        verdict = ActionFiberStructuralRecurrenceTerminalVerdict.BOUNDED_THREE_TARGET_SAFETY_TYPED_RECURRENCE
    elif passed == 2:
        verdict = ActionFiberStructuralRecurrenceTerminalVerdict.BOUNDED_TWO_TARGET_SAFETY_TYPED_RECURRENCE
    else:
        verdict = ActionFiberStructuralRecurrenceTerminalVerdict.SUBSTRATE_LOCAL
    reasons = tuple(
        sorted(
            {
                code
                for value in ordered
                for code in value.reason_codes
            }
            | {"SAME_SUBSTRATE_CLASSES_AS_DEFECT_DISCOVERY"}
        )
    )
    return ActionFiberStructuralRecurrenceAdjudication(
        adjudication_id='action-fiber-structural-recurrence-canonical-closure.cross-target-adjudication',
        method_conformance=ObjectIdentity.from_record(conformance.conformance_id, conformance),
        target_matches=tuple(ObjectIdentity.from_record(value.match_id, value) for value in ordered),
        eligible_target_count=len(ordered),
        passed_target_count=passed,
        failed_target_count=len(ordered) - passed,
        false_action_count=false_actions,
        hidden_sink_admission_count=hidden_sinks,
        hold_default_error_count=hold_errors,
        action_fiber_defect_resolved=false_actions == 0 and hidden_sinks == 0,
        hold_viability_defect_resolved=hold_errors == 0,
        target_wide_comparator_rejection_count=sum(
            value.target_wide_comparator_rejected for value in ordered
        ),
        assume_hold_safe_comparator_rejection_count=sum(
            value.assume_hold_safe_comparator_rejected for value in ordered
        ),
        verdict=verdict,
        positive_claim_eligible=passed == len(ordered) == 4 and safety_clean,
        same_substrate_classes_as_defect_discovery=True,
        independent_generality_claim_eligible=False,
        no_cross_target_pooling=True,
        reason_codes=reasons,
    )


def terminal_positive_allowed(
    *,
    false_action_count: int,
    hidden_sink_admission_count: int,
    hold_default_error_count: int,
) -> bool:
    return not (false_action_count or hidden_sink_admission_count or hold_default_error_count)


def source_semantics_sha256() -> str:
    """Stable semantic label included in the method freeze."""

    return sha256(
        b'action-fiber-structural-recurrence:action-fibered:measured-hold:one-sided-95-wilson:zero-safety-veto'
    ).hexdigest()


__all__ = [
    "ActionFiberSignature",
    "HoldViabilityDisposition",
    "HoldViabilitySignature",
    'ActionFiberStructuralRecurrenceAdjudication',
    'ActionFiberStructuralRecurrenceConformance',
    'ActionFiberStructuralRecurrenceControlResult',
    'ActionFiberStructuralRecurrenceFixtureExecution',
    'ActionFiberStructuralRecurrenceMethodFreeze',
    'ActionFiberStructuralRecurrenceAdmissionHandoff',
    'ActionFiberStructuralRecurrencePredictionIssue',
    'ActionFiberStructuralRecurrenceOperationalClosureFreeze',
    'ActionFiberStructuralRecurrenceCanonicalClosureFreeze',
    'ActionFiberStructuralRecurrenceVisibilityPreflightedClosureFreeze',
    'ActionFiberStructuralRecurrenceTargetMatch',
    'ActionFiberStructuralRecurrenceTargetResult',
    'ActionFiberStructuralRecurrenceTerminalVerdict',
    "PolicySafetySignature",
    "PolicyValidationDisposition",
    "adjudicate",
    "build_action_fiber",
    "build_policy_signature",
    "build_prediction_issue",
    'evaluate_admission',
    "finalize_target",
    "match_target",
    "terminal_positive_allowed",
    "wilson_lower_bound",
]
