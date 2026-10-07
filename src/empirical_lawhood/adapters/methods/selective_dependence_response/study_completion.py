"""Terminal custody between the programme forecast gate and evaluation issue."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import (
    ArtifactPublicationCommit,
    CanonicalTaskReceipt,
    artifact_publication_member,
)
from empirical_lawhood.runtime.recovery import RecoveryTerminalDisposition, CandidateRunRecoveryIndex, CandidateRunRecoveryTerminalEvent

from .forecast import SelectiveDependenceResponseStudyForecastQualification
from .study_forecast_protocol import SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_CAPABILITY_KEY, SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_VERSION


_TASK_ID = "qualify-programme-forecast"


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseStudyForecastCompletionEnvelope(CanonicalRecord):
    """Exact programme result plus proof of terminal receipt and recovery."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-study-forecast-completion-envelope'

    envelope_id: str
    study_qualification: SelectiveDependenceResponseStudyForecastQualification
    study_task_receipt: ObjectIdentity
    publication_commit: ObjectIdentity
    recovery_index: ObjectIdentity
    recovery_terminal_event: ObjectIdentity
    study_implementation_sha256: str
    completed_task_ids: tuple[str, ...]
    qualified: bool
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        validate_sha256(
            self.study_implementation_sha256,
            field_name='study_implementation_sha256',
        )
        expected_schemas = (
            (self.study_task_receipt, CanonicalTaskReceipt.SCHEMA),
            (self.publication_commit, ArtifactPublicationCommit.SCHEMA),
            (self.recovery_index, CandidateRunRecoveryIndex.SCHEMA),
            (self.recovery_terminal_event, CandidateRunRecoveryTerminalEvent.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("programme completion operational schema differs")
        if self.completed_task_ids != (_TASK_ID,):
            raise ValueError("programme completion task roster differs")
        if self.qualified != self.study_qualification.qualified:
            raise ValueError("programme completion qualification flag differs")
        if self.evaluation_outcome_count or (self.study_qualification.evaluation_outcome_count):
            raise ValueError("programme completion cannot contain evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("programme completion must remain development visible")


def build_study_forecast_completion_envelope(
    *,
    envelope_id: str,
    study_qualification: SelectiveDependenceResponseStudyForecastQualification,
    study_task_receipt: CanonicalTaskReceipt,
    publication_commit: ArtifactPublicationCommit,
    recovery_index: CandidateRunRecoveryIndex,
    recovery_terminal_event: CandidateRunRecoveryTerminalEvent,
    study_implementation_sha256: str,
) -> SelectiveDependenceResponseStudyForecastCompletionEnvelope:
    """Verify the exact one-node programme result custody without I/O."""

    validate_sha256(
        study_implementation_sha256,
        field_name='study_implementation_sha256',
    )
    if study_task_receipt.operational_status is not OperationalStatus.SUCCEEDED:
        raise ValueError("programme forecast receipt did not succeed")
    if study_task_receipt.task_id != _TASK_ID:
        raise ValueError("programme completion requires the forecast task receipt")
    logical = tuple(study_task_receipt.output_logical_artifacts)
    materializations = tuple(study_task_receipt.output_materializations)
    if len(logical) != 1 or len(materializations) != 1:
        raise ValueError("programme receipt must bind exactly one qualification")
    logical_value = logical[0]
    materialization = materializations[0]
    if (
        logical_value.payload_schema != SelectiveDependenceResponseStudyForecastQualification.SCHEMA
        or logical_value.content_sha256 != study_qualification.fingerprint()
        or logical_value.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        or logical_value.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
        or materialization.logical_artifact_id != logical_value.logical_artifact_id
        or materialization.physical_sha256 != study_qualification.fingerprint()
    ):
        raise ValueError("programme receipt does not bind exact qualification bytes")
    expected_member = artifact_publication_member(logical_value, materialization)
    if publication_commit.members != (expected_member,):
        raise ValueError("programme publication commit differs from receipt output")
    if (
        publication_commit.publication_scope.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        or publication_commit.publication_scope.visibility_ceiling
        is not VisibilityCeiling.DEVELOPMENT_ONLY
    ):
        raise ValueError("programme publication visibility differs")
    if (
        recovery_index.run_id != study_task_receipt.run_id
        or recovery_index.implementation_commit != study_task_receipt.implementation_commit
        or tuple(value.task_id for value in recovery_index.tasks) != (_TASK_ID,)
    ):
        raise ValueError("programme recovery index differs")
    task = recovery_index.task(_TASK_ID)
    if (
        task.dependency_task_ids
        or task.capability_key != SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_CAPABILITY_KEY
        or task.capability_version != SELECTIVE_DEPENDENCE_RESPONSE_PROGRAMME_FORECAST_VERSION
        or task.capability_implementation_sha256 != study_implementation_sha256
        or task.output_schema_ids != (SelectiveDependenceResponseStudyForecastQualification.SCHEMA,)
        or study_task_receipt.attempt_id not in {value.attempt_id for value in task.attempts}
    ):
        raise ValueError("programme recovery task binding differs")
    if (
        recovery_terminal_event.recovery_index
        != ObjectIdentity.from_record(recovery_index.recovery_index_id, recovery_index)
        or recovery_terminal_event.run_id != study_task_receipt.run_id
        or recovery_terminal_event.operational_status != "SUCCEEDED"
        or recovery_terminal_event.completed_task_ids != (_TASK_ID,)
        or recovery_terminal_event.failed_task_ids
        or recovery_terminal_event.blocked_task_ids
    ):
        raise ValueError("programme recovery terminal closure differs")
    terminal_attempts = tuple(
        value
        for value in recovery_terminal_event.attempts
        if value.task_id == _TASK_ID and value.disposition is RecoveryTerminalDisposition.SUCCEEDED
    )
    if len(terminal_attempts) != 1 or (
        terminal_attempts[0].attempt_id != study_task_receipt.attempt_id
        or terminal_attempts[0].receipt_id != study_task_receipt.receipt_id
        or terminal_attempts[0].receipt_sha256 != study_task_receipt.fingerprint()
    ):
        raise ValueError("programme recovery terminal receipt differs")
    return SelectiveDependenceResponseStudyForecastCompletionEnvelope(
        envelope_id=envelope_id,
        study_qualification=study_qualification,
        study_task_receipt=ObjectIdentity.from_record(
            study_task_receipt.receipt_id,
            study_task_receipt,
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
        study_implementation_sha256=study_implementation_sha256,
        completed_task_ids=(_TASK_ID,),
        qualified=study_qualification.qualified,
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


__all__ = [
    'SelectiveDependenceResponseStudyForecastCompletionEnvelope',
    'build_study_forecast_completion_envelope',
]
