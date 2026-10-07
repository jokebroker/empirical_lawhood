"""Terminal custody between target development and programme forecast.

The development bundle is constructed inside ``analyze-development`` before
that task has a canonical receipt.  Its dependency receipt IDs therefore
cannot prove publication or recovery of the bundle itself.  This module closes
that exact boundary without storage access or writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
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

from .forecast import SelectiveDependenceResponseDevelopmentBundle
from .method_completion import SelectiveDependenceResponseMethodCompletionEnvelope
from .source_completion import SelectiveDependenceResponseSourceCanaryCompletionEnvelope


_DEVELOPMENT_TASK_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "analyze-development": (
        "bind-construct-review",
        "freeze-target-analysis",
        "freeze-target-design",
        "generate-development-panel",
    ),
    "bind-construct-review": (),
    "freeze-target-analysis": (),
    "freeze-target-design": (),
    "generate-development-panel": (
        "bind-construct-review",
        "freeze-target-analysis",
        "freeze-target-design",
    ),
}
_DEVELOPMENT_TASK_IDS = tuple(sorted(_DEVELOPMENT_TASK_DEPENDENCIES))
_TARGET_BY_SLUG = {
    "cantera": "target.cantera-selective-dependence-response",
    "fipy": "target.fipy-selective-dependence-response",
}


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseDevelopmentCompletionEnvelope(CanonicalRecord):
    """Exact bundle bytes plus proof of their terminal operational custody."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-development-completion-envelope'

    envelope_id: str
    target_id: str
    target_slug: str
    development_bundle: SelectiveDependenceResponseDevelopmentBundle
    development_task_receipt: ObjectIdentity
    publication_commit: ObjectIdentity
    recovery_index: ObjectIdentity
    recovery_terminal_event: ObjectIdentity
    development_implementation_sha256: str
    completed_task_ids: tuple[str, ...]
    evaluation_eligible: bool
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("envelope_id", "target_id", "target_slug"):
            validate_stable_id(getattr(self, name), field_name=name)
        if _TARGET_BY_SLUG.get(self.target_slug) != self.target_id:
            raise ValueError("development completion target slug/identity differs")
        if self.development_bundle.target_id != self.target_id:
            raise ValueError("development completion bundle crosses targets")
        validate_sha256(
            self.development_implementation_sha256,
            field_name="development_implementation_sha256",
        )
        if (
            self.development_bundle.parent.implementation_sha256
            != self.development_implementation_sha256
        ):
            raise ValueError("development completion implementation differs from bundle")
        expected_schemas = (
            (self.development_task_receipt, CanonicalTaskReceipt.SCHEMA),
            (self.publication_commit, ArtifactPublicationCommit.SCHEMA),
            (self.recovery_index, CandidateRunRecoveryIndex.SCHEMA),
            (self.recovery_terminal_event, CandidateRunRecoveryTerminalEvent.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("development completion operational schema differs")
        if (
            self.development_bundle.parent.source_completion.object_schema
            != SelectiveDependenceResponseSourceCanaryCompletionEnvelope.SCHEMA
            or self.development_bundle.parent.method_completion.object_schema
            != SelectiveDependenceResponseMethodCompletionEnvelope.SCHEMA
        ):
            raise ValueError("development completion prerequisite lineage differs")
        require_sorted_unique_strings(
            self.completed_task_ids,
            field_name="completed_task_ids",
            allow_empty=False,
        )
        if self.completed_task_ids != _DEVELOPMENT_TASK_IDS:
            raise ValueError("development completion task roster differs")
        if self.evaluation_eligible != self.development_bundle.evaluation_eligible:
            raise ValueError("development completion eligibility is not bundle-derived")
        if self.evaluation_outcome_count:
            raise ValueError("development completion cannot contain evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("development completion must remain development visible")


def build_development_completion_envelope(
    *,
    envelope_id: str,
    target_slug: str,
    development_bundle: SelectiveDependenceResponseDevelopmentBundle,
    development_task_receipt: CanonicalTaskReceipt,
    publication_commit: ArtifactPublicationCommit,
    recovery_index: CandidateRunRecoveryIndex,
    recovery_terminal_event: CandidateRunRecoveryTerminalEvent,
    development_implementation_sha256: str,
) -> SelectiveDependenceResponseDevelopmentCompletionEnvelope:
    """Verify exact task/publication/recovery closure for one bundle."""

    validate_sha256(
        development_implementation_sha256,
        field_name="development_implementation_sha256",
    )
    target_id = development_bundle.target_id
    if _TARGET_BY_SLUG.get(target_slug) != target_id:
        raise ValueError("development completion target slug/identity differs")
    if development_bundle.parent.implementation_sha256 != development_implementation_sha256:
        raise ValueError("development completion implementation differs from bundle")
    if development_task_receipt.operational_status is not OperationalStatus.SUCCEEDED:
        raise ValueError("development analysis receipt did not succeed")
    if development_task_receipt.task_id != "analyze-development":
        raise ValueError("development completion requires the analysis task receipt")
    logical = tuple(development_task_receipt.output_logical_artifacts)
    materializations = tuple(development_task_receipt.output_materializations)
    if len(logical) != 1 or len(materializations) != 1:
        raise ValueError("development analysis receipt must bind exactly one bundle")
    logical_bundle = logical[0]
    materialized_bundle = materializations[0]
    if (
        logical_bundle.payload_schema != SelectiveDependenceResponseDevelopmentBundle.SCHEMA
        or logical_bundle.content_sha256 != development_bundle.fingerprint()
        or logical_bundle.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        or logical_bundle.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
        or materialized_bundle.logical_artifact_id != logical_bundle.logical_artifact_id
        or materialized_bundle.physical_sha256 != development_bundle.fingerprint()
    ):
        raise ValueError("development analysis receipt does not bind exact bundle bytes")
    if not set(development_bundle.parent.development_materialization_ids).issubset(
        development_task_receipt.input_materialization_ids
    ):
        raise ValueError("development bundle dependencies differ from analysis receipt")
    expected_member = artifact_publication_member(logical_bundle, materialized_bundle)
    if publication_commit.members != (expected_member,):
        raise ValueError("development publication commit differs from receipt output")
    if (
        publication_commit.publication_scope.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        or publication_commit.publication_scope.visibility_ceiling
        is not VisibilityCeiling.DEVELOPMENT_ONLY
    ):
        raise ValueError("development publication visibility differs")
    if (
        recovery_index.run_id != development_task_receipt.run_id
        or recovery_index.implementation_commit != development_task_receipt.implementation_commit
    ):
        raise ValueError("development recovery index belongs to another run")
    if tuple(value.task_id for value in recovery_index.tasks) != _DEVELOPMENT_TASK_IDS:
        raise ValueError("development recovery task roster differs")
    for task in recovery_index.tasks:
        if task.dependency_task_ids != _DEVELOPMENT_TASK_DEPENDENCIES[task.task_id]:
            raise ValueError("development recovery graph differs")
        if task.capability_implementation_sha256 != development_implementation_sha256:
            raise ValueError("development recovery implementation differs")
    analysis_task = recovery_index.task("analyze-development")
    if (
        analysis_task.capability_key != f"simulator.selective-dependence-response.{target_slug}.analyze-development"
        or analysis_task.capability_version != "1.0.0"
        or analysis_task.output_schema_ids != (SelectiveDependenceResponseDevelopmentBundle.SCHEMA,)
        or development_task_receipt.attempt_id
        not in {value.attempt_id for value in analysis_task.attempts}
    ):
        raise ValueError("development recovery analysis binding differs")
    if (
        recovery_terminal_event.recovery_index
        != ObjectIdentity.from_record(recovery_index.recovery_index_id, recovery_index)
        or recovery_terminal_event.run_id != development_task_receipt.run_id
        or recovery_terminal_event.operational_status != "SUCCEEDED"
        or recovery_terminal_event.completed_task_ids != _DEVELOPMENT_TASK_IDS
        or recovery_terminal_event.failed_task_ids
        or recovery_terminal_event.blocked_task_ids
    ):
        raise ValueError("development recovery terminal closure differs")
    terminal_attempts = tuple(
        value
        for value in recovery_terminal_event.attempts
        if value.task_id == development_task_receipt.task_id
        and value.disposition is RecoveryTerminalDisposition.SUCCEEDED
    )
    if len(terminal_attempts) != 1 or (
        terminal_attempts[0].attempt_id != development_task_receipt.attempt_id
        or terminal_attempts[0].receipt_id != development_task_receipt.receipt_id
        or terminal_attempts[0].receipt_sha256 != development_task_receipt.fingerprint()
    ):
        raise ValueError("development recovery terminal receipt differs")
    return SelectiveDependenceResponseDevelopmentCompletionEnvelope(
        envelope_id=envelope_id,
        target_id=target_id,
        target_slug=target_slug,
        development_bundle=development_bundle,
        development_task_receipt=ObjectIdentity.from_record(
            development_task_receipt.receipt_id,
            development_task_receipt,
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
        development_implementation_sha256=development_implementation_sha256,
        completed_task_ids=_DEVELOPMENT_TASK_IDS,
        evaluation_eligible=development_bundle.evaluation_eligible,
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


__all__ = [
    'SelectiveDependenceResponseDevelopmentCompletionEnvelope',
    "build_development_completion_envelope",
]
