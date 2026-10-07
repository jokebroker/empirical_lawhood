"""Terminal custody envelope between source canaries and development.

Source qualification is an excluded, development-visible act.  Development is
a separate issue/authority act and may consume the source result only after the
qualification and exact canary panel have a successful shared task receipt,
atomic publication commit and terminal recovery record.  This module verifies
that boundary without reading or writing storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import (
    ArtifactPublicationCommit,
    CanonicalTaskReceipt,
    artifact_publication_member,
)
from empirical_lawhood.runtime.recovery import RecoveryTerminalDisposition, CandidateRunRecoveryIndex, CandidateRunRecoveryTerminalEvent

from .contracts import SelectiveDependenceResponsePhase, SelectiveDependenceResponsePreparationDistributionFreeze, SelectiveDependenceResponseTargetPanel


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseSourceCanaryCompletionEnvelope(CanonicalRecord):
    """Compact proof that excluded source qualification is terminal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-source-canary-completion-envelope'

    envelope_id: str
    target_id: str
    design: ObjectIdentity
    preparation_freeze: ObjectIdentity
    analysis_freeze: ObjectIdentity
    construct_review: ObjectIdentity
    source_qualification: ObjectIdentity
    canary_panel: ObjectIdentity
    source_task_receipt: ObjectIdentity
    publication_commit: ObjectIdentity
    recovery_index: ObjectIdentity
    recovery_terminal_event: ObjectIdentity
    canary_complete_unit_ids: tuple[str, ...]
    canary_complete_unit_ids_sha256: str
    source_implementation_sha256: str
    development_complete_unit_count: int
    source_ready: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("envelope_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        expected_schemas = (
            (self.source_task_receipt, CanonicalTaskReceipt.SCHEMA),
            (self.publication_commit, ArtifactPublicationCommit.SCHEMA),
            (self.recovery_index, CandidateRunRecoveryIndex.SCHEMA),
            (self.recovery_terminal_event, CandidateRunRecoveryTerminalEvent.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("source completion operational schema differs")
        require_sorted_unique_strings(
            self.canary_complete_unit_ids,
            field_name="canary_complete_unit_ids",
            allow_empty=False,
        )
        validate_sha256(
            self.canary_complete_unit_ids_sha256,
            field_name="canary_complete_unit_ids_sha256",
        )
        validate_sha256(
            self.source_implementation_sha256,
            field_name="source_implementation_sha256",
        )
        from .contracts import digest_ids

        if self.canary_complete_unit_ids_sha256 != digest_ids(self.canary_complete_unit_ids):
            raise ValueError("source completion canary roster digest differs")
        if self.development_complete_unit_count:
            raise ValueError("source completion cannot contain development units")
        if not self.source_ready:
            raise ValueError("an unqualified source cannot open development")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("source completion must retain excluded development visibility")


def build_source_canary_completion_envelope(
    *,
    envelope_id: str,
    target_id: str,
    design: CanonicalRecord,
    design_id: str,
    preparation: SelectiveDependenceResponsePreparationDistributionFreeze,
    analysis_freeze: CanonicalRecord,
    analysis_freeze_id: str,
    construct_review: CanonicalRecord,
    construct_review_id: str,
    source_qualification: CanonicalRecord,
    source_qualification_id: str,
    canary_panel: SelectiveDependenceResponseTargetPanel,
    source_task_receipt: CanonicalTaskReceipt,
    publication_commit: ArtifactPublicationCommit,
    recovery_index: CandidateRunRecoveryIndex,
    recovery_terminal_event: CandidateRunRecoveryTerminalEvent,
    source_implementation_sha256: str,
) -> SelectiveDependenceResponseSourceCanaryCompletionEnvelope:
    """Verify shared output custody and emit the development prerequisite."""

    validate_sha256(source_implementation_sha256, field_name="source_implementation_sha256")
    if (
        target_id != preparation.target_id
        or canary_panel.target_id != target_id
        or canary_panel.phase is not SelectiveDependenceResponsePhase.CANARY
        or canary_panel.expected_complete_unit_ids != preparation.canary_unit_ids
    ):
        raise ValueError("source completion target/canary roster differs")
    qualification_target = getattr(source_qualification, "target_id", target_id)
    if qualification_target != target_id or not bool(
        getattr(source_qualification, "source_ready", False)
    ):
        raise ValueError("source completion qualification is not target-ready")
    if getattr(source_qualification, "canary_panel", None) != ObjectIdentity.from_record(
        canary_panel.panel_id, canary_panel
    ):
        raise ValueError("source qualification does not bind the exact canary panel")
    expected_result_identities = tuple(
        ObjectIdentity.from_record(value.result_id, value) for value in canary_panel.complete_units
    )
    if getattr(source_qualification, "canary_result_identities", None) != (
        expected_result_identities
    ):
        raise ValueError("source qualification canary result roster differs")
    if source_task_receipt.operational_status is not OperationalStatus.SUCCEEDED:
        raise ValueError("source qualification receipt did not succeed")
    expected_records = {
        source_qualification.SCHEMA: source_qualification,
        SelectiveDependenceResponseTargetPanel.SCHEMA: canary_panel,
    }
    logical = tuple(source_task_receipt.output_logical_artifacts)
    materializations = tuple(source_task_receipt.output_materializations)
    if (
        len(logical) != 2
        or len(materializations) != 2
        or {value.payload_schema for value in logical} != set(expected_records)
    ):
        raise ValueError("source receipt must jointly bind qualification and canary panel")
    for value in logical:
        if (
            value.content_sha256 != expected_records[value.payload_schema].fingerprint()
            or value.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("source receipt output bytes/access differ")
    expected_members = tuple(
        sorted(
            (
                artifact_publication_member(logical_value, materialization)
                for logical_value, materialization in zip(
                    logical,
                    materializations,
                    strict=True,
                )
            ),
            key=lambda value: value.materialization_id,
        )
    )
    if publication_commit.members != expected_members:
        raise ValueError("source publication commit differs from receipt outputs")
    if (
        recovery_index.run_id != source_task_receipt.run_id
        or recovery_index.implementation_commit != source_task_receipt.implementation_commit
    ):
        raise ValueError("source recovery index belongs to another run")
    try:
        task_binding = recovery_index.task(source_task_receipt.task_id)
    except KeyError as error:
        raise ValueError("source task is absent from recovery index") from error
    if set(task_binding.output_schema_ids) != set(
        expected_records
    ) or source_task_receipt.attempt_id not in {
        value.attempt_id for value in task_binding.attempts
    }:
        raise ValueError("source recovery task output/attempt differs")
    if task_binding.capability_implementation_sha256 != source_implementation_sha256:
        raise ValueError("source completion implementation differs from recovery authority")
    if (
        recovery_terminal_event.recovery_index
        != ObjectIdentity.from_record(recovery_index.recovery_index_id, recovery_index)
        or recovery_terminal_event.run_id != source_task_receipt.run_id
        or recovery_terminal_event.operational_status != "SUCCEEDED"
        or source_task_receipt.task_id not in recovery_terminal_event.completed_task_ids
    ):
        raise ValueError("source recovery terminal closure differs")
    terminal_attempts = tuple(
        value
        for value in recovery_terminal_event.attempts
        if value.task_id == source_task_receipt.task_id
        and value.disposition is RecoveryTerminalDisposition.SUCCEEDED
    )
    if len(terminal_attempts) != 1 or (
        terminal_attempts[0].attempt_id != source_task_receipt.attempt_id
        or terminal_attempts[0].receipt_id != source_task_receipt.receipt_id
        or terminal_attempts[0].receipt_sha256 != source_task_receipt.fingerprint()
    ):
        raise ValueError("source recovery terminal receipt differs")
    from .contracts import digest_ids

    return SelectiveDependenceResponseSourceCanaryCompletionEnvelope(
        envelope_id=envelope_id,
        target_id=target_id,
        design=ObjectIdentity.from_record(design_id, design),
        preparation_freeze=ObjectIdentity.from_record(preparation.freeze_id, preparation),
        analysis_freeze=ObjectIdentity.from_record(analysis_freeze_id, analysis_freeze),
        construct_review=ObjectIdentity.from_record(construct_review_id, construct_review),
        source_qualification=ObjectIdentity.from_record(
            source_qualification_id, source_qualification
        ),
        canary_panel=ObjectIdentity.from_record(canary_panel.panel_id, canary_panel),
        source_task_receipt=ObjectIdentity.from_record(
            source_task_receipt.receipt_id, source_task_receipt
        ),
        publication_commit=ObjectIdentity.from_record(
            publication_commit.publication_batch_id, publication_commit
        ),
        recovery_index=ObjectIdentity.from_record(recovery_index.recovery_index_id, recovery_index),
        recovery_terminal_event=ObjectIdentity.from_record(
            recovery_terminal_event.terminal_event_id, recovery_terminal_event
        ),
        canary_complete_unit_ids=canary_panel.expected_complete_unit_ids,
        canary_complete_unit_ids_sha256=digest_ids(canary_panel.expected_complete_unit_ids),
        source_implementation_sha256=source_implementation_sha256,
        development_complete_unit_count=0,
        source_ready=True,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


__all__ = [
    'SelectiveDependenceResponseSourceCanaryCompletionEnvelope',
    "build_source_canary_completion_envelope",
]
