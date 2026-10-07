"""Outcome-blind authorization records and candidate-to-frozen conversion."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.worlds import WorldKind

from .design import ExperimentProposal


class AuthorizationDecision(StrEnum):
    APPROVED_NONACTUATING = "APPROVED_NONACTUATING"
    REFUSED = "REFUSED"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    PAUSED = "PAUSED"


@dataclass(frozen=True, slots=True)
class AuthorizationRecord(CanonicalRecord):
    """One immutable policy-gate decision, never scientific evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/authorization-record'

    authorization_id: str
    policy: ObjectIdentity
    proposal: ObjectIdentity
    experiment: ObjectIdentity
    decision: AuthorizationDecision
    action: AuthorityAction
    world_kind: WorldKind
    source_access: SourceAccessClass
    requested_scope_id: str
    requested_budget: ResourceBudget
    passed_gate_ids: tuple[str, ...]
    failed_gate_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    proposer_id: str
    approver_id: str
    decided_at_utc: str
    implementation_commit: str
    outcome_access: OutcomeAccess
    plan_mutated: bool
    grants_claim_promotion: bool
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("authorization_id", self.authorization_id),
            ("requested_scope_id", self.requested_scope_id),
            ("proposer_id", self.proposer_id),
            ("approver_id", self.approver_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.passed_gate_ids, field_name="passed_gate_ids")
        require_sorted_unique_strings(self.failed_gate_ids, field_name="failed_gate_ids")
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )
        if set(self.passed_gate_ids) & set(self.failed_gate_ids):
            raise ValueError("an authorization gate cannot both pass and fail")
        validate_nonempty(self.decided_at_utc, field_name="decided_at_utc")
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit) is None:
            raise ValueError("implementation_commit must be a lowercase Git SHA-1")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("an authorization gate cannot read outcome values")
        if self.plan_mutated:
            raise ValueError("an authorization gate cannot edit the proposed plan")
        if self.grants_claim_promotion:
            raise ValueError("authorization cannot grant scientific promotion")
        if self.decision is AuthorizationDecision.APPROVED_NONACTUATING:
            if self.failed_gate_ids:
                raise ValueError("an approval cannot contain failed gates")
        require_extensions(self.extensions)


@dataclass(frozen=True, slots=True)
class ApprovalRequest(CanonicalRecord):
    """Outcome-blind input to the separately scoped nonactuating gate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/approval-request'

    authorization_id: str
    action: AuthorityAction
    world_kind: WorldKind
    source_access: SourceAccessClass
    requested_scope_id: str
    requested_budget: ResourceBudget
    passed_gate_ids: tuple[str, ...]
    proposer_id: str
    approver_id: str
    decided_at_utc: str
    implementation_commit: str

    def __post_init__(self) -> None:
        for name, value in (
            ("authorization_id", self.authorization_id),
            ("requested_scope_id", self.requested_scope_id),
            ("proposer_id", self.proposer_id),
            ("approver_id", self.approver_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.passed_gate_ids,
            field_name="passed_gate_ids",
        )
        validate_nonempty(self.decided_at_utc, field_name="decided_at_utc")
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit) is None:
            raise ValueError("implementation_commit must be a lowercase Git SHA-1")


class ApprovalGate:
    "Contained request gate; it cannot produce a new authorization act.\n\n    Frozen records remain decodable for historical/reference reproduction.\n    New approval must use :class:`planning.approval.CompleteApprovalService`.\n    "

    def decide(
        self,
        policy: AuthorityPolicy,
        proposal: ExperimentProposal,
        request: ApprovalRequest,
    ) -> AuthorizationRecord:
        reason_codes = ("COMPLETE_APPROVAL_ENVELOPE_REQUIRED",)
        return AuthorizationRecord(
            authorization_id=request.authorization_id,
            policy=ObjectIdentity.from_record(policy.policy_id, policy),
            proposal=ObjectIdentity.from_record(proposal.proposal_id, proposal),
            experiment=ObjectIdentity.from_record(
                proposal.candidate_experiment.experiment_id,
                proposal.candidate_experiment,
            ),
            decision=AuthorizationDecision.AUTHORITY_REQUIRED,
            action=request.action,
            world_kind=request.world_kind,
            source_access=request.source_access,
            requested_scope_id=request.requested_scope_id,
            requested_budget=request.requested_budget,
            passed_gate_ids=(),
            failed_gate_ids=policy.required_gate_ids,
            reason_codes=reason_codes,
            proposer_id=request.proposer_id,
            approver_id=request.approver_id,
            decided_at_utc=request.decided_at_utc,
            implementation_commit=request.implementation_commit,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            plan_mutated=False,
            grants_claim_promotion=False,
        )


def authorize_experiment(
    policy: AuthorityPolicy,
    proposal: ExperimentProposal,
    record: AuthorizationRecord,
) -> ExperimentSpec:
    """Create a new authorized spec after replaying the immutable policy check."""

    expected_policy = ObjectIdentity.from_record(policy.policy_id, policy)
    expected_proposal = ObjectIdentity.from_record(proposal.proposal_id, proposal)
    expected_experiment = ObjectIdentity.from_record(
        proposal.candidate_experiment.experiment_id,
        proposal.candidate_experiment,
    )
    if record.policy != expected_policy:
        raise ValueError("authorization record binds the wrong policy")
    if record.proposal != expected_proposal:
        raise ValueError("authorization record binds the wrong proposal")
    if record.experiment != expected_experiment:
        raise ValueError("authorization record binds the wrong candidate experiment")
    if record.decision is not AuthorizationDecision.APPROVED_NONACTUATING:
        raise ValueError("only an approved non-actuating record can authorize")
    reasons = policy.refusal_reasons(
        action=record.action,
        world_kind=record.world_kind,
        source_access=record.source_access,
        passed_gate_ids=frozenset(record.passed_gate_ids),
        requested_budget=record.requested_budget,
        requested_outcome_access=record.outcome_access,
        requested_scope_id=record.requested_scope_id,
        proposer_id=record.proposer_id,
        approver_id=record.approver_id,
        at_utc=record.decided_at_utc,
    )
    if reasons:
        raise ValueError(f"authorization policy refuses the request: {reasons}")
    return replace(
        proposal.candidate_experiment,
        authorization_record_id=record.authorization_id,
        readiness=ReadinessStatus.READY,
    )
