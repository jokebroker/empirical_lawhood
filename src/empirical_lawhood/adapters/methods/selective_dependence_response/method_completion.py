"""Outcome-blind terminal gate for the selective dependence response truth-world method act.

The conformance report itself remains privileged truth.  This module is the
only compatibility boundary into target work: it verifies the exact report,
task receipt, atomic publication and recovery closure, then emits a compact
record containing no truth-case outcomes and no target response values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import (
    ArtifactPublicationCommit,
    CanonicalTaskReceipt,
    artifact_publication_member,
)
from empirical_lawhood.runtime.recovery import RecoveryTerminalDisposition, CandidateRunRecoveryIndex, CandidateRunRecoveryTerminalEvent

from .conformance import SelectiveDependenceResponseMethodConformanceReport
from .contracts import SelectiveDependenceResponseMethodQuestionFreeze


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseMethodCompletionEnvelope(CanonicalRecord):
    """Compact proof that no-target method conformance closed successfully."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-method-completion-envelope'

    envelope_id: str
    method_question: ObjectIdentity
    conformance_report: ObjectIdentity
    conformance_task_receipt: ObjectIdentity
    publication_commit: ObjectIdentity
    recovery_index: ObjectIdentity
    recovery_terminal_event: ObjectIdentity
    method_implementation_sha256: str
    conformance_case_count: int
    all_passed: bool
    target_response_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        expected_schemas = (
            (self.method_question, SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA),
            (self.conformance_report, SelectiveDependenceResponseMethodConformanceReport.SCHEMA),
            (self.conformance_task_receipt, CanonicalTaskReceipt.SCHEMA),
            (self.publication_commit, ArtifactPublicationCommit.SCHEMA),
            (self.recovery_index, CandidateRunRecoveryIndex.SCHEMA),
            (self.recovery_terminal_event, CandidateRunRecoveryTerminalEvent.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("method completion schema differs")
        validate_sha256(
            self.method_implementation_sha256,
            field_name="method_implementation_sha256",
        )
        if self.conformance_case_count != 14:
            raise ValueError("method completion conformance case count differs")
        if not self.all_passed:
            raise ValueError("failed method conformance cannot open target work")
        if self.target_response_count:
            raise ValueError("method completion cannot contain target responses")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("method completion must be outcome blind")


def build_method_completion_envelope(
    *,
    envelope_id: str,
    method_question: SelectiveDependenceResponseMethodQuestionFreeze,
    conformance_report: SelectiveDependenceResponseMethodConformanceReport,
    conformance_task_receipt: CanonicalTaskReceipt,
    publication_commit: ArtifactPublicationCommit,
    recovery_index: CandidateRunRecoveryIndex,
    recovery_terminal_event: CandidateRunRecoveryTerminalEvent,
    method_implementation_sha256: str,
) -> SelectiveDependenceResponseMethodCompletionEnvelope:
    """Verify terminal privileged conformance without carrying its outcomes onward."""

    validate_sha256(
        method_implementation_sha256,
        field_name="method_implementation_sha256",
    )
    question_identity = ObjectIdentity.from_record(method_question.freeze_id, method_question)
    if conformance_report.method_question != question_identity:
        raise ValueError("method conformance report binds another question")
    if (
        not conformance_report.all_passed
        or len(conformance_report.case_results) != 14
        or conformance_report.target_response_count
        or conformance_report.outcome_access is not OutcomeAccess.PRIVILEGED_TRUTH
    ):
        raise ValueError("method conformance report cannot open target work")
    if conformance_task_receipt.operational_status is not OperationalStatus.SUCCEEDED:
        raise ValueError("method conformance receipt did not succeed")
    logical = tuple(conformance_task_receipt.output_logical_artifacts)
    materializations = tuple(conformance_task_receipt.output_materializations)
    if (
        len(logical) != 1
        or len(materializations) != 1
        or logical[0].payload_schema != SelectiveDependenceResponseMethodConformanceReport.SCHEMA
        or logical[0].content_sha256 != conformance_report.fingerprint()
        or logical[0].outcome_access is not OutcomeAccess.PRIVILEGED_TRUTH
        or materializations[0].logical_artifact_id != logical[0].logical_artifact_id
    ):
        raise ValueError("method receipt does not bind the exact privileged report")
    expected_members = (artifact_publication_member(logical[0], materializations[0]),)
    if publication_commit.members != expected_members:
        raise ValueError("method publication commit differs from receipt output")
    if (
        recovery_index.run_id != conformance_task_receipt.run_id
        or recovery_index.implementation_commit != conformance_task_receipt.implementation_commit
    ):
        raise ValueError("method recovery index belongs to another run")
    try:
        task_binding = recovery_index.task(conformance_task_receipt.task_id)
    except KeyError as error:
        raise ValueError("method conformance task is absent from recovery index") from error
    if task_binding.output_schema_ids != (
        SelectiveDependenceResponseMethodConformanceReport.SCHEMA,
    ) or conformance_task_receipt.attempt_id not in {
        value.attempt_id for value in task_binding.attempts
    }:
        raise ValueError("method recovery task output/attempt differs")
    if (
        recovery_terminal_event.recovery_index
        != ObjectIdentity.from_record(recovery_index.recovery_index_id, recovery_index)
        or recovery_terminal_event.run_id != conformance_task_receipt.run_id
        or recovery_terminal_event.operational_status != "SUCCEEDED"
        or conformance_task_receipt.task_id not in recovery_terminal_event.completed_task_ids
    ):
        raise ValueError("method recovery terminal closure differs")
    terminal_attempts = tuple(
        value
        for value in recovery_terminal_event.attempts
        if value.task_id == conformance_task_receipt.task_id
        and value.disposition is RecoveryTerminalDisposition.SUCCEEDED
    )
    if len(terminal_attempts) != 1 or (
        terminal_attempts[0].attempt_id != conformance_task_receipt.attempt_id
        or terminal_attempts[0].receipt_id != conformance_task_receipt.receipt_id
        or terminal_attempts[0].receipt_sha256 != conformance_task_receipt.fingerprint()
    ):
        raise ValueError("method recovery terminal receipt differs")
    return SelectiveDependenceResponseMethodCompletionEnvelope(
        envelope_id=envelope_id,
        method_question=question_identity,
        conformance_report=ObjectIdentity.from_record(
            conformance_report.report_id,
            conformance_report,
        ),
        conformance_task_receipt=ObjectIdentity.from_record(
            conformance_task_receipt.receipt_id,
            conformance_task_receipt,
        ),
        publication_commit=ObjectIdentity.from_record(
            publication_commit.publication_batch_id,
            publication_commit,
        ),
        recovery_index=ObjectIdentity.from_record(
            recovery_index.recovery_index_id,
            recovery_index,
        ),
        recovery_terminal_event=ObjectIdentity.from_record(
            recovery_terminal_event.terminal_event_id,
            recovery_terminal_event,
        ),
        method_implementation_sha256=method_implementation_sha256,
        conformance_case_count=len(conformance_report.case_results),
        all_passed=conformance_report.all_passed,
        target_response_count=conformance_report.target_response_count,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


__all__ = ['SelectiveDependenceResponseMethodCompletionEnvelope', "build_method_completion_envelope"]
