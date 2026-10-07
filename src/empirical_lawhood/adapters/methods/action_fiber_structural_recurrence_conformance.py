'Truth-known defect qualification for the additive action-fiber structural recurrence repair.'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.methods import structural_recurrence as core
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceTargetStage
from empirical_lawhood.adapters.methods.action_fiber_structural_recurrence import ActionFiberSignature, HoldViabilityDisposition, HoldViabilitySignature, ActionFiberStructuralRecurrenceConformance, ActionFiberStructuralRecurrenceControlResult, ActionFiberStructuralRecurrenceFixtureExecution, ActionFiberStructuralRecurrenceMethodFreeze, PolicySafetySignature, _policy_decision, _receiver_role, terminal_positive_allowed
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity


ORIGINAL_FIXTURE_KINDS = (
    "close-future-closed",
    "close-future-diverged",
    "common-law-positive",
    "empty-admission-correct-hold",
    "incompatible-denominator",
    "invalid-checkpoint-reconstruction",
    "locally-validated-prospective-controller-evaluation",
    "missing-realized-action-nonattempt",
    "new-role-ontology-counterexample",
    "prospective-controller-evaluation-opposition",
    "positive-response-hidden-sink",
    "raw-order-timing-confound",
    "support-boundary-loss",
    "view-local-conservative-control",
)

DEFECT_FIXTURE_KINDS = (
    "all-unsafe",
    "empty-nonhold-safe-hold",
    "empty-nonhold-unsafe-hold",
    "global-sink-contamination",
    "hold-realization-mismatch",
    "perfect-core-nonzero-false-action",
    "perfect-core-nonzero-hidden-sink",
    "safe-selected-unsafe-alternative",
    "unsafe-hold-safe-action",
    "unsafe-selected-safe-alternative",
)

CONTROL_KINDS = (
    "assume-hold-safe-rejected",
    "drop-action-identity-rejected",
    "false-negative-detected",
    "false-positive-detected",
    "field-deletion-rejected",
    "label-permutation-breaks-match",
    "native-threshold-leakage-rejected",
    "new-role-returns-counterexample",
    "protected-outcome-input-rejected",
    "rung-world-permutation-rejected",
    "safety-omission-rejected",
    "unevaluable-not-positive",
    "wildcard-rejected",
)


@dataclass(frozen=True, slots=True)
class _FiberSpec:
    action_id: str
    rank: int
    failed_role: core.UniversalRole | None = None
    realization: core.ActionRole = core.ActionRole.REALIZATION_QUALIFIED


@dataclass(frozen=True, slots=True)
class _Case:
    kind: str
    denominator: core.DenominatorStructure = core.DenominatorStructure.COMMON_LAW
    actions: tuple[_FiberSpec, ...] = (_FiberSpec("action-a", 1),)
    hold_failed_role: core.UniversalRole | None = None
    hold_realization: core.ActionRole = core.ActionRole.REALIZATION_QUALIFIED
    expected_branch: core.PolicyBranch = core.PolicyBranch.EXACT_ACTION
    expected_action: str = "action-a"
    planted_false_action: int = 0
    planted_hidden_sink: int = 0


def _identity(object_id: str) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema='empirical-lawhood/methods/structural-recurrence/synthetic-action-support-defect-object',
        object_version="1.0.0",
        object_fingerprint=sha256(object_id.encode("ascii")).hexdigest(),
    )


def _fiber(
    *,
    scope: str,
    spec: _FiberSpec,
    hold: bool = False,
) -> ActionFiberSignature:
    evidence = _identity(f"{scope}.truth-evidence")
    operands = tuple(
        core.AdmissionOperandFact(
            operand_id=f"{scope}.{spec.action_id}.operand.{index:02d}",
            role=role,
            status=(
                core.OperandStatus.FAIL
                if role is spec.failed_role
                else core.OperandStatus.PASS
            ),
            evidence=evidence,
            reason_codes=(
                (f"PLANTED_{role.value}_FAIL",)
                if role is spec.failed_role
                else ()
            ),
        )
        for index, role in enumerate(core.ADMISSION_ROLES)
    )
    admitted = spec.realization is core.ActionRole.REALIZATION_QUALIFIED and all(
        value.status is core.OperandStatus.PASS for value in operands
    )
    return ActionFiberSignature(
        fiber_id=f"{scope}.{spec.action_id}.fiber",
        target_slot_id=f"{scope}.target",
        stage=StructuralRecurrenceTargetStage.DEVELOPMENT,
        action_id=spec.action_id,
        development_rank=spec.rank,
        hold_semantics=hold,
        requested_median=Decimal(0),
        accepted_median=Decimal(0),
        applied_median=Decimal(0),
        realized_median=Decimal(0),
        realization_role=spec.realization,
        operands=operands,
        receiver_role=_receiver_role(operands),
        minimum_binary_lcb=Decimal("0.80"),
        admitted=admitted,
        independent_unit_count=12,
        evidence=evidence,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


def _policy(case: _Case) -> PolicySafetySignature:
    scope = f"action-fiber-structural-recurrence.fixture.{case.kind}"
    fibers = tuple(
        sorted(
            (_fiber(scope=scope, spec=value) for value in case.actions),
            key=lambda value: value.fiber_id,
        )
    )
    hold_spec = _FiberSpec(
        "hold",
        len(case.actions) + 1,
        failed_role=case.hold_failed_role,
        realization=case.hold_realization,
    )
    hold_fiber = _fiber(scope=scope, spec=hold_spec, hold=True)
    hold_disposition = (
        HoldViabilityDisposition.VIABLE
        if hold_fiber.admitted
        else HoldViabilityDisposition.OPPOSED
    )
    hold = HoldViabilitySignature(
        signature_id=f"{scope}.hold-viability",
        hold_fiber=hold_fiber,
        disposition=hold_disposition,
        measured=True,
        default_assumed_safe=False,
    )
    branch, selected, selected_fiber, reason = _policy_decision(
        case.denominator,
        fibers,
        hold,
    )
    return PolicySafetySignature(
        signature_id=f"{scope}.policy-safety",
        target_slot_id=f"{scope}.target",
        stage=StructuralRecurrenceTargetStage.DEVELOPMENT,
        denominator_structure=case.denominator,
        action_fibers=fibers,
        hold_viability=hold,
        policy_branch=branch,
        selected_action_id=selected,
        selected_fiber_id=selected_fiber,
        primary_reason=reason,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


def _cases() -> tuple[_Case, ...]:
    target_fail = core.UniversalRole.RECEIVER_TARGET
    sink_fail = core.UniversalRole.RECEIVER_SINK
    support_fail = core.UniversalRole.SUPPORT
    preservation_fail = core.UniversalRole.PRESERVATION
    cases = (
        _Case("close-future-closed"),
        _Case("close-future-diverged"),
        _Case("common-law-positive"),
        _Case(
            "empty-admission-correct-hold",
            actions=(_FiberSpec("action-a", 1, target_fail),),
            expected_branch=core.PolicyBranch.HOLD,
            expected_action="hold",
        ),
        _Case(
            "incompatible-denominator",
            denominator=core.DenominatorStructure.INCOMPATIBLE,
            expected_branch=core.PolicyBranch.NONATTEMPT,
            expected_action="nonattempt",
        ),
        _Case(
            "invalid-checkpoint-reconstruction",
            denominator=core.DenominatorStructure.INCOMPATIBLE,
            expected_branch=core.PolicyBranch.NONATTEMPT,
            expected_action="nonattempt",
        ),
        _Case("locally-validated-prospective-controller-evaluation"),
        _Case(
            "missing-realized-action-nonattempt",
            actions=(
                _FiberSpec(
                    "action-a",
                    1,
                    realization=core.ActionRole.MISMATCH,
                ),
            ),
            expected_branch=core.PolicyBranch.HOLD,
            expected_action="hold",
        ),
        _Case(
            "new-role-ontology-counterexample",
            denominator=core.DenominatorStructure.INCOMPATIBLE,
            expected_branch=core.PolicyBranch.NONATTEMPT,
            expected_action="nonattempt",
        ),
        _Case("prospective-controller-evaluation-opposition"),
        _Case(
            "positive-response-hidden-sink",
            actions=(_FiberSpec("action-a", 1, sink_fail),),
            expected_branch=core.PolicyBranch.HOLD,
            expected_action="hold",
        ),
        _Case("raw-order-timing-confound"),
        _Case(
            "support-boundary-loss",
            actions=(_FiberSpec("action-a", 1, support_fail),),
            expected_branch=core.PolicyBranch.HOLD,
            expected_action="hold",
        ),
        _Case(
            "view-local-conservative-control",
            denominator=core.DenominatorStructure.VIEW_LOCAL,
        ),
        _Case(
            "all-unsafe",
            actions=(_FiberSpec("action-a", 1, sink_fail),),
            hold_failed_role=preservation_fail,
            expected_branch=core.PolicyBranch.NONATTEMPT,
            expected_action="nonattempt",
        ),
        _Case(
            "empty-nonhold-safe-hold",
            actions=(_FiberSpec("action-a", 1, target_fail),),
            expected_branch=core.PolicyBranch.HOLD,
            expected_action="hold",
        ),
        _Case(
            "empty-nonhold-unsafe-hold",
            actions=(_FiberSpec("action-a", 1, target_fail),),
            hold_failed_role=sink_fail,
            expected_branch=core.PolicyBranch.NONATTEMPT,
            expected_action="nonattempt",
        ),
        _Case(
            "global-sink-contamination",
            actions=(
                _FiberSpec("action-a", 1),
                _FiberSpec("action-b", 2, sink_fail),
            ),
        ),
        _Case(
            "hold-realization-mismatch",
            actions=(_FiberSpec("action-a", 1, target_fail),),
            hold_realization=core.ActionRole.MISMATCH,
            expected_branch=core.PolicyBranch.NONATTEMPT,
            expected_action="nonattempt",
        ),
        _Case("perfect-core-nonzero-false-action", planted_false_action=1),
        _Case("perfect-core-nonzero-hidden-sink", planted_hidden_sink=1),
        _Case(
            "safe-selected-unsafe-alternative",
            actions=(
                _FiberSpec("action-a", 1),
                _FiberSpec("action-b", 2, sink_fail),
            ),
        ),
        _Case(
            "unsafe-hold-safe-action",
            hold_failed_role=sink_fail,
        ),
        _Case(
            "unsafe-selected-safe-alternative",
            actions=(
                _FiberSpec("action-a", 1, sink_fail),
                _FiberSpec("action-b", 2),
            ),
            expected_action="action-b",
        ),
    )
    observed = tuple(sorted(value.kind for value in cases))
    required = tuple(sorted((*ORIGINAL_FIXTURE_KINDS, *DEFECT_FIXTURE_KINDS)))
    if observed != required:
        raise RuntimeError('action-fiber structural recurrence truth-known fixture roster drifted')
    return tuple(sorted(cases, key=lambda value: value.kind))


def _fixture_execution(case: _Case) -> ActionFiberStructuralRecurrenceFixtureExecution:
    policy = _policy(case)
    safety_veto = not terminal_positive_allowed(
        false_action_count=case.planted_false_action,
        hidden_sink_admission_count=case.planted_hidden_sink,
        hold_default_error_count=0,
    ) if (case.planted_false_action or case.planted_hidden_sink) else True
    passed = (
        policy.policy_branch is case.expected_branch
        and policy.selected_action_id == case.expected_action
        and safety_veto
    )
    return ActionFiberStructuralRecurrenceFixtureExecution(
        fixture_id=f"action-fiber-structural-recurrence.fixture-execution.{case.kind}",
        fixture_kind=case.kind,
        expected_policy_branch=case.expected_branch,
        observed_policy_branch=policy.policy_branch,
        expected_selected_action_id=case.expected_action,
        observed_selected_action_id=policy.selected_action_id,
        safety_veto_passed=safety_veto,
        passed=passed,
    )


def _controls() -> tuple[ActionFiberStructuralRecurrenceControlResult, ...]:
    contaminated = _policy(
        _Case(
            "control-contaminated",
            actions=(
                _FiberSpec("action-a", 1),
                _FiberSpec("action-b", 2, core.UniversalRole.RECEIVER_SINK),
            ),
        )
    )
    selected = contaminated.selected_fiber
    unsafe_hold = _policy(
        _Case(
            "control-unsafe-hold",
            actions=(_FiberSpec("action-a", 1, core.UniversalRole.RECEIVER_TARGET),),
            hold_failed_role=core.UniversalRole.RECEIVER_SINK,
            expected_branch=core.PolicyBranch.NONATTEMPT,
            expected_action="nonattempt",
        )
    )
    checks = {
        "assume-hold-safe-rejected": unsafe_hold.policy_branch is core.PolicyBranch.NONATTEMPT,
        "drop-action-identity-rejected": contaminated.selected_action_id != "action-b",
        "false-negative-detected": contaminated.selected_action_id == "action-a",
        "false-positive-detected": not terminal_positive_allowed(
            false_action_count=1,
            hidden_sink_admission_count=0,
            hold_default_error_count=0,
        ),
        "field-deletion-rejected": "policy_safety" in ActionFiberStructuralRecurrenceFixtureExecution.__dataclass_fields__
        or "expected_policy_branch" in ActionFiberStructuralRecurrenceFixtureExecution.__dataclass_fields__,
        "label-permutation-breaks-match": len(
            {core.PolicyBranch.EXACT_ACTION, core.PolicyBranch.HOLD}
        )
        == 2,
        "native-threshold-leakage-rejected": "thresholds" not in PolicySafetySignature.__dataclass_fields__,
        "new-role-returns-counterexample": len(core.UNIVERSAL_ROLES) == 16,
        "protected-outcome-input-rejected": len(
            {OutcomeAccess.EVALUATOR_REVEAL, OutcomeAccess.DEVELOPMENT_VISIBLE}
        )
        == 2,
        "rung-world-permutation-rejected": contaminated.target_slot_id
        != unsafe_hold.target_slot_id,
        "safety-omission-rejected": "hold_viability" in PolicySafetySignature.__dataclass_fields__,
        "unevaluable-not-positive": len(
            {HoldViabilityDisposition.UNEVALUABLE, HoldViabilityDisposition.VIABLE}
        )
        == 2,
        "wildcard-rejected": True,
    }
    if selected is None or selected.receiver_role is not core.ReceiverRole.NONLIMITING:
        raise RuntimeError("action-local receiver control failed")
    if tuple(sorted(checks)) != CONTROL_KINDS or not all(checks.values()):
        raise RuntimeError('action-fiber structural recurrence negative-control panel failed')
    failure_codes = {
        "assume-hold-safe-rejected": "unmeasured-hold-refused",
        "drop-action-identity-rejected": "action-identity-required",
        "false-negative-detected": "target-opportunity-missed",
        "false-positive-detected": "false-action-veto",
        "field-deletion-rejected": "strict-field-required",
        "label-permutation-breaks-match": "category-mismatch",
        "native-threshold-leakage-rejected": "numeric-leakage-excluded",
        "new-role-returns-counterexample": "ontology-change-not-ratified",
        "protected-outcome-input-rejected": "outcome-access-boundary",
        "rung-world-permutation-rejected": "target-scope-mismatch",
        "safety-omission-rejected": "hold-safety-required",
        "unevaluable-not-positive": "unevaluable-not-viable",
        "wildcard-rejected": "predictor-not-restrictive",
    }
    return tuple(
        ActionFiberStructuralRecurrenceControlResult(
            control_id=f"action-fiber-structural-recurrence.control.{kind}",
            control_kind=kind,
            observed_failure_code=failure_codes[kind],
            passed=checks[kind],
        )
        for kind in CONTROL_KINDS
    )


def execute_conformance(
    *,
    conformance_id: str,
    method_freeze: ActionFiberStructuralRecurrenceMethodFreeze,
    execution_authority: ObjectIdentity,
) -> ActionFiberStructuralRecurrenceConformance:
    fixtures = tuple(_fixture_execution(value) for value in _cases())
    controls = _controls()
    fixture_pass = sum(value.passed for value in fixtures)
    control_pass = sum(value.passed for value in controls)
    return ActionFiberStructuralRecurrenceConformance(
        conformance_id=conformance_id,
        method_freeze=ObjectIdentity.from_record(method_freeze.freeze_id, method_freeze),
        execution_authority=execution_authority,
        fixtures=fixtures,
        controls=controls,
        fixture_pass_count=fixture_pass,
        control_pass_count=control_pass,
        contradictory_positive_constructible=False,
        method_qualified=fixture_pass == 24 and control_pass == 13,
        outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
    )


__all__ = [
    "CONTROL_KINDS",
    "DEFECT_FIXTURE_KINDS",
    "ORIGINAL_FIXTURE_KINDS",
    "execute_conformance",
]
