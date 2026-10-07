"""Exact operational closure required before an selective dependence response target handoff.

The cross-target stage must not accept a scientifically plausible record that
has not also closed its receipt, publication and bounded-recovery lineage.  The
builder below verifies the supplied runtime records once and emits a compact
canonical envelope; it performs no storage reads or writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import (
    ArtifactPublicationCommit,
    CanonicalTaskReceipt,
    artifact_publication_member,
)
from empirical_lawhood.runtime.recovery import RecoveryTerminalDisposition, CandidateRunRecoveryIndex, CandidateRunRecoveryTerminalEvent

from .contracts import SelectiveDependenceResponseTargetHandoff


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetCompletionEnvelope(CanonicalRecord):
    """Compact proof that one handoff is scientifically and operationally terminal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-completion-envelope'

    envelope_id: str
    target_id: str
    handoff: SelectiveDependenceResponseTargetHandoff
    handoff_receipt: ObjectIdentity
    publication_commit: ObjectIdentity
    recovery_index: ObjectIdentity
    recovery_terminal_event: ObjectIdentity
    handoff_materialization_ids: tuple[str, ...]
    native_numeric_value_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("envelope_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.handoff.target_id != self.target_id:
            raise ValueError("completion envelope crosses targets")
        expected_schemas = (
            (self.handoff_receipt, CanonicalTaskReceipt.SCHEMA),
            (self.publication_commit, ArtifactPublicationCommit.SCHEMA),
            (self.recovery_index, CandidateRunRecoveryIndex.SCHEMA),
            (self.recovery_terminal_event, CandidateRunRecoveryTerminalEvent.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("completion envelope operational schema differs")
        require_sorted_unique_strings(
            self.handoff_materialization_ids,
            field_name="handoff_materialization_ids",
            allow_empty=False,
        )
        if self.native_numeric_value_count or self.handoff.native_numeric_value_count:
            raise ValueError("completion envelope cannot carry native numeric values")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("completion envelope must remain outcome visible")


def build_target_completion_envelope(
    *,
    envelope_id: str,
    handoff: SelectiveDependenceResponseTargetHandoff,
    handoff_receipt: CanonicalTaskReceipt,
    publication_commit: ArtifactPublicationCommit,
    recovery_index: CandidateRunRecoveryIndex,
    recovery_terminal_event: CandidateRunRecoveryTerminalEvent,
) -> SelectiveDependenceResponseTargetCompletionEnvelope:
    """Validate an exact terminal runtime lineage and issue its compact envelope."""

    if handoff_receipt.operational_status is not OperationalStatus.SUCCEEDED:
        raise ValueError("target handoff receipt did not succeed")
    logical = tuple(
        value
        for value in handoff_receipt.output_logical_artifacts
        if value.payload_schema == SelectiveDependenceResponseTargetHandoff.SCHEMA
    )
    if (
        len(handoff_receipt.output_logical_artifacts) != 1
        or len(handoff_receipt.output_materializations) != 1
        or len(logical) != 1
        or logical[0].content_sha256 != handoff.fingerprint()
        or logical[0].outcome_access is not OutcomeAccess.EVALUATION_REVEALED
    ):
        raise ValueError("target handoff receipt does not bind the exact handoff bytes")
    materializations = tuple(
        value
        for value in handoff_receipt.output_materializations
        if value.logical_artifact_id == logical[0].logical_artifact_id
    )
    if len(materializations) != 1:
        raise ValueError("target handoff receipt materialization roster differs")
    expected_members = tuple(
        sorted(
            (
                artifact_publication_member(logical_value, materialization)
                for logical_value, materialization in zip(
                    handoff_receipt.output_logical_artifacts,
                    handoff_receipt.output_materializations,
                    strict=True,
                )
            ),
            key=lambda value: value.materialization_id,
        )
    )
    if publication_commit.members != expected_members:
        raise ValueError("target handoff publication commit differs from receipt outputs")
    if (
        recovery_index.run_id != handoff_receipt.run_id
        or recovery_index.implementation_commit != handoff_receipt.implementation_commit
    ):
        raise ValueError("target handoff recovery index belongs to another run")
    try:
        task_binding = recovery_index.task(handoff_receipt.task_id)
    except KeyError as error:
        raise ValueError("target handoff task is absent from recovery index") from error
    if task_binding.output_schema_ids != (SelectiveDependenceResponseTargetHandoff.SCHEMA,) or (
        handoff_receipt.attempt_id not in {value.attempt_id for value in task_binding.attempts}
    ):
        raise ValueError("target handoff recovery task does not declare the handoff output")
    if (
        recovery_terminal_event.recovery_index
        != ObjectIdentity.from_record(recovery_index.recovery_index_id, recovery_index)
        or recovery_terminal_event.run_id != handoff_receipt.run_id
        or recovery_terminal_event.operational_status != "SUCCEEDED"
        or handoff_receipt.task_id not in recovery_terminal_event.completed_task_ids
    ):
        raise ValueError("target handoff recovery terminal closure differs")
    terminal_attempts = tuple(
        value
        for value in recovery_terminal_event.attempts
        if value.task_id == handoff_receipt.task_id
        and value.disposition is RecoveryTerminalDisposition.SUCCEEDED
    )
    if len(terminal_attempts) != 1 or (
        terminal_attempts[0].attempt_id != handoff_receipt.attempt_id
        or terminal_attempts[0].receipt_id != handoff_receipt.receipt_id
        or terminal_attempts[0].receipt_sha256 != handoff_receipt.fingerprint()
    ):
        raise ValueError("target handoff recovery terminal receipt differs")
    return SelectiveDependenceResponseTargetCompletionEnvelope(
        envelope_id=envelope_id,
        target_id=handoff.target_id,
        handoff=handoff,
        handoff_receipt=ObjectIdentity.from_record(handoff_receipt.receipt_id, handoff_receipt),
        publication_commit=ObjectIdentity.from_record(
            publication_commit.publication_batch_id, publication_commit
        ),
        recovery_index=ObjectIdentity.from_record(recovery_index.recovery_index_id, recovery_index),
        recovery_terminal_event=ObjectIdentity.from_record(
            recovery_terminal_event.terminal_event_id, recovery_terminal_event
        ),
        handoff_materialization_ids=tuple(value.materialization_id for value in materializations),
        native_numeric_value_count=0,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


__all__ = ['SelectiveDependenceResponseTargetCompletionEnvelope', "build_target_completion_envelope"]
