"""Bounded external run recovery and compact operational reconstruction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Final

from empirical_lawhood.infrastructure.execution_resource_envelopes import DEFAULT_EXECUTION_RESOURCE_ENVELOPE_STATE_ROOT, ExternalExecutionResourceEnvelopeStore
from empirical_lawhood.infrastructure.artifacts import (
    ArtifactIdentityConflict,
    ExternalArtifactPlane,
)
from empirical_lawhood.infrastructure.bounded_io import (
    BoundedFileIOError,
    MAX_CONTROL_PLANE_JSON_BYTES,
    read_bounded_bytes,
)
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    CanonicalizationError,
    validate_document_shape,
)
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ArtifactWriteRequest,
    CanonicalTaskReceipt,
    lineage_parent_sort_key,
    most_restrictive_outcome_access,
)
from empirical_lawhood.runtime.execution import (
    OperationalFailureClass,
    OperationalAttempt,
    OperationalRepository,
    TaskAttemptDisposition,
    TaskBlockReason,
    TaskReceiptStore,
)
from empirical_lawhood.runtime.execution_envelope import ExecutionEnvelopeEvent, ExecutionEnvelopeSpec, ExecutionResourceEnvelopeEvent, ExecutionResourceEnvelopeSpec, JitGraphSignatureManifest, RunExecutionEnvelope, RunExecutionResourceEnvelope, reconstruct_execution_envelope, reconstruct_execution_resource_envelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan, ProtocolExecutionTask
from empirical_lawhood.runtime.retry_amendment import LeaseExpiryRetryAmendment, RetainedCustodyCompletionAmendment
from empirical_lawhood.runtime.recovery import MAX_RECOVERY_CANDIDATES, OperationalFailureDiagnostic, RecoveryTerminalAttempt, RecoveryTerminalDisposition, RecoveryAttemptCandidate, RecoveryTaskBinding, ProtocolRunRecoveryIndex, CandidateRunRecoveryIndex, EnvelopeRunRecoveryIndex, RunRecoveryIndex, RunRecoveryReport, ProtocolRunRecoveryTerminalEvent, CandidateRunRecoveryTerminalEvent, EnvelopeRunRecoveryTerminalEvent, RunRecoveryTerminalEvent, TaskRecoveryDisposition, ProtocolTaskRecoveryEvent, TaskRecoveryEvent


MAX_RECOVERY_INDEX_BYTES: Final[int] = 32 * 1024**2
MAX_RECOVERY_EVENT_BYTES: Final[int] = 2 * 1024**2
MAX_EXECUTION_ENVELOPE_EVENTS: Final[int] = 100_000


class RunRecoveryError(RuntimeError):
    """External recovery authority, custody or state is inconsistent."""


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CanonicalizationError(f"{field_name} must be a string-keyed mapping")
    return value


def _record(
    value: object,
    record_type: type[CanonicalRecord],
    fields: frozenset[str],
) -> Mapping[str, object]:
    return validate_document_shape(
        _mapping(value, record_type.SCHEMA),
        expected_schema=record_type.SCHEMA,
        expected_version=record_type.VERSION,
        field_names=fields,
    )


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise CanonicalizationError(f"{field_name} must be a string")
    return value


def _optional_string(value: object, field_name: str) -> str | None:
    return None if value is None else _string(value, field_name)


def _integer(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanonicalizationError(f"{field_name} must be an integer")
    return value


def _optional_integer(value: object, field_name: str) -> int | None:
    return None if value is None else _integer(value, field_name)


def _list(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise CanonicalizationError(f"{field_name} must be a JSON array")
    return value


def _boolean(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise CanonicalizationError(f"{field_name} must be a boolean")
    return value


def _strings(value: object, field_name: str) -> tuple[str, ...]:
    return tuple(_string(item, field_name) for item in _list(value, field_name))


def _identity(document: object) -> ObjectIdentity:
    value = _record(
        document,
        ObjectIdentity,
        frozenset(
            {
                "object_id",
                "object_schema",
                "object_version",
                "object_fingerprint",
            }
        ),
    )
    return ObjectIdentity(
        object_id=_string(value["object_id"], "object_id"),
        object_schema=_string(value["object_schema"], "object_schema"),
        object_version=_string(value["object_version"], "object_version"),
        object_fingerprint=_string(value["object_fingerprint"], "object_fingerprint"),
    )


def _attempt_candidate(document: object) -> RecoveryAttemptCandidate:
    value = _record(
        document,
        RecoveryAttemptCandidate,
        frozenset(
            {
                "attempt_id",
                "ordinal",
                "receipt_relative_path",
                "event_relative_path",
                "receipt_schema",
                "event_schema",
            }
        ),
    )
    return RecoveryAttemptCandidate(
        attempt_id=_string(value["attempt_id"], "attempt_id"),
        ordinal=_integer(value["ordinal"], "ordinal"),
        receipt_relative_path=_string(
            value["receipt_relative_path"],
            "receipt_relative_path",
        ),
        event_relative_path=_string(value["event_relative_path"], "event_relative_path"),
        receipt_schema=_string(value["receipt_schema"], "receipt_schema"),
        event_schema=_string(value["event_schema"], "event_schema"),
    )


def _task_binding(document: object) -> RecoveryTaskBinding:
    value = _record(
        document,
        RecoveryTaskBinding,
        frozenset(
            {
                "task_id",
                "topological_ordinal",
                "dependency_task_ids",
                "maximum_attempts",
                "capability_key",
                "capability_version",
                "capability_implementation_sha256",
                "output_schema_ids",
                "attempts",
                "nonattempt_event_relative_path",
            }
        ),
    )
    return RecoveryTaskBinding(
        task_id=_string(value["task_id"], "task_id"),
        topological_ordinal=_integer(
            value["topological_ordinal"],
            "topological_ordinal",
        ),
        dependency_task_ids=_strings(
            value["dependency_task_ids"],
            "dependency_task_ids",
        ),
        maximum_attempts=_integer(value["maximum_attempts"], "maximum_attempts"),
        capability_key=_string(value["capability_key"], "capability_key"),
        capability_version=_string(value["capability_version"], "capability_version"),
        capability_implementation_sha256=_string(
            value["capability_implementation_sha256"],
            "capability_implementation_sha256",
        ),
        output_schema_ids=_strings(value["output_schema_ids"], "output_schema_ids"),
        attempts=tuple(_attempt_candidate(item) for item in _list(value["attempts"], "attempts")),
        nonattempt_event_relative_path=_string(
            value["nonattempt_event_relative_path"],
            "nonattempt_event_relative_path",
        ),
    )


def _json_document(payload: bytes, *, label: str, maximum_bytes: int) -> object:
    if not payload or len(payload) > maximum_bytes:
        raise CanonicalizationError(f"{label} violates its byte bound")
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise CanonicalizationError(f"{label} is not UTF-8 JSON") from error


def decode_run_recovery_index(payload: bytes) -> ProtocolRunRecoveryIndex:
    document = _json_document(
        payload,
        label="run recovery index",
        maximum_bytes=MAX_RECOVERY_INDEX_BYTES,
    )
    schema = _string(_mapping(document, "run recovery index").get("schema"), "schema")
    record_type: type[ProtocolRunRecoveryIndex]
    if schema == ProtocolRunRecoveryIndex.SCHEMA:
        record_type = ProtocolRunRecoveryIndex
    elif schema == CandidateRunRecoveryIndex.SCHEMA:
        record_type = CandidateRunRecoveryIndex
    elif schema == EnvelopeRunRecoveryIndex.SCHEMA:
        return decode_canonical_bytes(
            payload,
            EnvelopeRunRecoveryIndex,
            maximum_bytes=MAX_RECOVERY_INDEX_BYTES,
        )
    elif schema == RunRecoveryIndex.SCHEMA:
        return decode_canonical_bytes(
            payload,
            RunRecoveryIndex,
            maximum_bytes=MAX_RECOVERY_INDEX_BYTES,
        )
    else:
        raise CanonicalizationError("run recovery index schema is unsupported")
    value = _record(
        document,
        record_type,
        frozenset(
            {
                "recovery_index_id",
                "wave_id",
                "run_id",
                "execution_plan",
                "authority_identities",
                "registry_sha256",
                "implementation_commit",
                "index_relative_path",
                "terminal_event_relative_path",
                "terminal_event_schema",
                "tasks",
            }
        ),
    )
    result = record_type(
        recovery_index_id=_string(
            value["recovery_index_id"],
            "recovery_index_id",
        ),
        wave_id=_string(value["wave_id"], "wave_id"),
        run_id=_string(value["run_id"], "run_id"),
        execution_plan=_identity(value["execution_plan"]),
        authority_identities=tuple(
            _identity(item)
            for item in _list(
                value["authority_identities"],
                "authority_identities",
            )
        ),
        registry_sha256=_string(value["registry_sha256"], "registry_sha256"),
        implementation_commit=_string(
            value["implementation_commit"],
            "implementation_commit",
        ),
        index_relative_path=_string(
            value["index_relative_path"],
            "index_relative_path",
        ),
        terminal_event_relative_path=_string(
            value["terminal_event_relative_path"],
            "terminal_event_relative_path",
        ),
        terminal_event_schema=_string(
            value["terminal_event_schema"],
            "terminal_event_schema",
        ),
        tasks=tuple(_task_binding(item) for item in _list(value["tasks"], "tasks")),
    )
    if result.canonical_bytes() != payload:
        raise CanonicalizationError("run recovery index bytes are not canonical")
    return result


def _failure_diagnostic(document: object) -> OperationalFailureDiagnostic | None:
    if document is None:
        return None
    value = _record(
        document,
        OperationalFailureDiagnostic,
        frozenset(
            {
                "failure_class",
                "retryable",
                "sanitized_exception_type",
                "sanitized_message",
                "diagnostic_sha256",
                "scientific_evidence",
            }
        ),
    )
    try:
        failure_class = OperationalFailureClass(_string(value["failure_class"], "failure_class"))
    except ValueError as error:
        raise CanonicalizationError("operational failure class is unsupported") from error
    return OperationalFailureDiagnostic(
        failure_class=failure_class,
        retryable=_boolean(value["retryable"], "retryable"),
        sanitized_exception_type=_optional_string(
            value["sanitized_exception_type"],
            "sanitized_exception_type",
        ),
        sanitized_message=_optional_string(value["sanitized_message"], "sanitized_message"),
        diagnostic_sha256=_string(value["diagnostic_sha256"], "diagnostic_sha256"),
        scientific_evidence=_boolean(value["scientific_evidence"], "scientific_evidence"),
    )


def decode_task_recovery_event(payload: bytes) -> TaskRecoveryEvent:
    document = _json_document(
        payload,
        label="task recovery event",
        maximum_bytes=MAX_RECOVERY_EVENT_BYTES,
    )
    schema = _string(_mapping(document, "task recovery event").get("schema"), "schema")
    if schema != TaskRecoveryEvent.SCHEMA:
        raise CanonicalizationError("task recovery event schema is unsupported")
    value = _record(
        document,
        TaskRecoveryEvent,
        frozenset(
            {
                "event_id",
                "recovery_index",
                "run_id",
                "task_id",
                "attempt_id",
                "attempt_ordinal",
                "disposition",
                "receipt_id",
                "receipt_relative_path",
                "receipt_sha256",
                "receipt_schema",
                "output_materialization_ids",
                "reason_codes",
                "failure",
            }
        ),
    )
    try:
        disposition = TaskRecoveryDisposition(_string(value["disposition"], "disposition"))
    except ValueError as error:
        raise CanonicalizationError("task recovery disposition is unsupported") from error
    result = TaskRecoveryEvent(
        event_id=_string(value["event_id"], "event_id"),
        recovery_index=_identity(value["recovery_index"]),
        run_id=_string(value["run_id"], "run_id"),
        task_id=_string(value["task_id"], "task_id"),
        attempt_id=_optional_string(value["attempt_id"], "attempt_id"),
        attempt_ordinal=_optional_integer(
            value["attempt_ordinal"],
            "attempt_ordinal",
        ),
        disposition=disposition,
        receipt_id=_optional_string(value["receipt_id"], "receipt_id"),
        receipt_relative_path=_optional_string(
            value["receipt_relative_path"],
            "receipt_relative_path",
        ),
        receipt_sha256=_optional_string(value["receipt_sha256"], "receipt_sha256"),
        receipt_schema=_optional_string(value["receipt_schema"], "receipt_schema"),
        output_materialization_ids=_strings(
            value["output_materialization_ids"],
            "output_materialization_ids",
        ),
        reason_codes=_strings(value["reason_codes"], "reason_codes"),
        failure=_failure_diagnostic(value["failure"]),
    )
    if result.canonical_bytes() != payload:
        raise CanonicalizationError("task recovery event bytes are not canonical")
    return result


def decode_execution_envelope_event(payload: bytes) -> ExecutionEnvelopeEvent:
    """Strictly decode one canonical, bounded execution-envelope event."""

    return decode_canonical_bytes(
        payload,
        ExecutionEnvelopeEvent,
        maximum_bytes=MAX_RECOVERY_EVENT_BYTES,
    )


def decode_execution_resource_envelope_event(
    payload: bytes,
) -> ExecutionResourceEnvelopeEvent:
    """Strictly decode one canonical deadline-free resource event."""

    return decode_canonical_bytes(
        payload,
        ExecutionResourceEnvelopeEvent,
        maximum_bytes=MAX_RECOVERY_EVENT_BYTES,
    )


def reconstruct_execution_envelope_from_external_events(
    *,
    index: EnvelopeRunRecoveryIndex,
    envelope_id: str,
    spec: ExecutionEnvelopeSpec,
    execution_plan: ObjectIdentity,
    event_payloads: tuple[bytes, ...],
) -> RunExecutionEnvelope:
    "Reconstruct exact state from externally retained canonical event bytes."

    if len(event_payloads) > MAX_EXECUTION_ENVELOPE_EVENTS:
        raise RunRecoveryError("execution-envelope event count exceeds its bound")
    if (
        index.execution_envelope_spec != ObjectIdentity.from_record(spec.envelope_spec_id, spec)
        or index.execution_plan != execution_plan
        or spec.issued_study_extensions != index.issued_extension_set
    ):
        raise RunRecoveryError("execution-envelope recovery inputs differ from the index")
    try:
        events = tuple(decode_execution_envelope_event(value) for value in event_payloads)
        if tuple(value.sequence_number for value in events) != tuple(range(1, len(events) + 1)):
            raise ValueError("execution-envelope external events are incomplete or out of order")
        return reconstruct_execution_envelope(
            envelope_id=envelope_id,
            spec=spec,
            execution_plan=execution_plan,
            events=events,
        )
    except (CanonicalizationError, ValueError) as error:
        raise RunRecoveryError("execution-envelope external event reconstruction failed") from error


def reconstruct_execution_resource_envelope_from_external_events(
    *,
    index: RunRecoveryIndex,
    envelope_id: str,
    spec: ExecutionResourceEnvelopeSpec,
    execution_plan: ObjectIdentity,
    jit_graph_signature_manifest: JitGraphSignatureManifest | None,
    event_payloads: tuple[bytes, ...],
) -> RunExecutionResourceEnvelope:
    "Reconstruct exact state from externally retained event bytes."

    if len(event_payloads) > MAX_EXECUTION_ENVELOPE_EVENTS:
        raise RunRecoveryError("resource-envelope event count exceeds its bound")
    if (
        index.execution_resource_envelope_spec
        != ObjectIdentity.from_record(spec.envelope_spec_id, spec)
        or index.execution_plan != execution_plan
        or spec.issued_study_extensions != index.issued_extension_set
        or index.jit_graph_signature_manifest
        != (
            None
            if jit_graph_signature_manifest is None
            else ObjectIdentity.from_record(
                jit_graph_signature_manifest.manifest_id,
                jit_graph_signature_manifest,
            )
        )
    ):
        raise RunRecoveryError("resource-envelope recovery inputs differ from the index")
    try:
        events = tuple(
            decode_execution_resource_envelope_event(value) for value in event_payloads
        )
        if tuple(value.sequence_number for value in events) != tuple(range(1, len(events) + 1)):
            raise ValueError("resource-envelope external events are incomplete or out of order")
        return reconstruct_execution_resource_envelope(
            envelope_id=envelope_id,
            spec=spec,
            execution_plan=execution_plan,
            events=events,
            jit_graph_signature_manifest=jit_graph_signature_manifest,
        )
    except (CanonicalizationError, ValueError) as error:
        raise RunRecoveryError("resource-envelope external event reconstruction failed") from error


def _terminal_attempt(document: object) -> RecoveryTerminalAttempt:
    value = _record(
        document,
        RecoveryTerminalAttempt,
        frozenset(
            {
                "attempt_id",
                "task_id",
                "ordinal",
                "disposition",
                "reason_code",
                "receipt_id",
                "receipt_sha256",
            }
        ),
    )
    try:
        disposition = RecoveryTerminalDisposition(_string(value["disposition"], "disposition"))
    except ValueError as error:
        raise CanonicalizationError("terminal attempt disposition is unsupported") from error
    return RecoveryTerminalAttempt(
        attempt_id=_string(value["attempt_id"], "attempt_id"),
        task_id=_string(value["task_id"], "task_id"),
        ordinal=_integer(value["ordinal"], "ordinal"),
        disposition=disposition,
        reason_code=_optional_string(value["reason_code"], "reason_code"),
        receipt_id=_optional_string(value["receipt_id"], "receipt_id"),
        receipt_sha256=_optional_string(value["receipt_sha256"], "receipt_sha256"),
    )


def decode_run_recovery_terminal_event(payload: bytes) -> ProtocolRunRecoveryTerminalEvent:
    document = _json_document(
        payload,
        label="run recovery terminal event",
        maximum_bytes=MAX_RECOVERY_INDEX_BYTES,
    )
    schema = _string(
        _mapping(document, "run recovery terminal event").get("schema"),
        "schema",
    )
    record_type: type[ProtocolRunRecoveryTerminalEvent]
    if schema == ProtocolRunRecoveryTerminalEvent.SCHEMA:
        record_type = ProtocolRunRecoveryTerminalEvent
    elif schema == CandidateRunRecoveryTerminalEvent.SCHEMA:
        record_type = CandidateRunRecoveryTerminalEvent
    elif schema == EnvelopeRunRecoveryTerminalEvent.SCHEMA:
        return decode_canonical_bytes(
            payload,
            EnvelopeRunRecoveryTerminalEvent,
            maximum_bytes=MAX_RECOVERY_INDEX_BYTES,
        )
    elif schema == RunRecoveryTerminalEvent.SCHEMA:
        return decode_canonical_bytes(
            payload,
            RunRecoveryTerminalEvent,
            maximum_bytes=MAX_RECOVERY_INDEX_BYTES,
        )
    else:
        raise CanonicalizationError("run recovery terminal event schema is unsupported")
    value = _record(
        document,
        record_type,
        frozenset(
            {
                "terminal_event_id",
                "recovery_index",
                "run_id",
                "operational_status",
                "attempts",
                "completed_task_ids",
                "failed_task_ids",
                "blocked_task_ids",
            }
        ),
    )
    result = record_type(
        terminal_event_id=_string(value["terminal_event_id"], "terminal_event_id"),
        recovery_index=_identity(value["recovery_index"]),
        run_id=_string(value["run_id"], "run_id"),
        operational_status=_string(
            value["operational_status"],
            "operational_status",
        ),
        attempts=tuple(_terminal_attempt(item) for item in _list(value["attempts"], "attempts")),
        completed_task_ids=_strings(
            value["completed_task_ids"],
            "completed_task_ids",
        ),
        failed_task_ids=_strings(value["failed_task_ids"], "failed_task_ids"),
        blocked_task_ids=_strings(value["blocked_task_ids"], "blocked_task_ids"),
    )
    if result.canonical_bytes() != payload:
        raise CanonicalizationError("run recovery terminal event bytes are not canonical")
    return result


class ExternalRunRecoveryStore:
    """One external, scan-free recovery stream for a frozen execution wave."""

    def __init__(
        self,
        plane: ExternalArtifactPlane,
        *,
        minimum_free_bytes: int = 0,
    ) -> None:
        if minimum_free_bytes < 0:
            raise ValueError("recovery free-space floor must be nonnegative")
        self.plane = plane
        self.minimum_free_bytes = minimum_free_bytes

    @staticmethod
    def _index_parents(index: ProtocolRunRecoveryIndex) -> tuple[ArtifactLineageParent, ...]:
        values = (
            ArtifactLineageParent(
                identity=index.execution_plan,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            ),
            *(
                ArtifactLineageParent(
                    identity=authority,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                )
                for authority in index.authority_identities
            ),
        )
        return tuple(sorted(values, key=lineage_parent_sort_key))

    def freeze(self, index: ProtocolRunRecoveryIndex) -> None:
        parents = self._index_parents(index)
        self.plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=index.recovery_index_id,
                relative_path=index.index_relative_path,
                payload_schema=index.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=f"run-recovery.{index.run_id}.{index.wave_id}",
                publication_scope_relative_root=(f"runs/{index.run_id}/recovery/{index.wave_id}"),
                payload=index.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                parent_visibility_ceilings=tuple(parent.visibility_ceiling for parent in parents),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                logical_content_sha256=index.fingerprint(),
                lineage_parents=parents,
                minimum_free_bytes=self.minimum_free_bytes,
            )
        )
        observed = self.read_index(
            index.index_relative_path,
            expected_schema=index.SCHEMA,
        )
        if observed != index:
            raise RunRecoveryError("persisted run recovery index differs after freeze")

    def _read_payload(
        self,
        relative_path: str,
        *,
        expected_schema: str,
        maximum_bytes: int,
    ) -> bytes | None:
        path = self.plane.root.resolve(relative_path, for_write=False)
        manifest_path = self.plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        if not path.exists() and not manifest_path.exists():
            return None
        if not path.is_file() or not manifest_path.is_file():
            raise RunRecoveryError("recovery artifact/manifest pair is partial")
        try:
            manifest = decode_artifact_manifest(
                read_bounded_bytes(
                    manifest_path,
                    maximum_bytes=MAX_CONTROL_PLANE_JSON_BYTES,
                )
            )
            if (
                manifest.materialization.relative_path != relative_path
                or manifest.materialization.size_bytes > maximum_bytes
                or manifest.logical.payload_schema != expected_schema
                or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
                or manifest.logical.media_type != "application/json"
            ):
                raise RunRecoveryError("recovery artifact violates its bounded contract")
            self.plane.verify_manifest(manifest)
            payload = read_bounded_bytes(path, maximum_bytes=maximum_bytes)
        except (
            ArtifactIdentityConflict,
            BoundedFileIOError,
            OSError,
            ValueError,
        ) as error:
            raise RunRecoveryError("recovery artifact custody is invalid") from error
        if (
            len(payload) != manifest.materialization.size_bytes
            or hashlib.sha256(payload).hexdigest() != manifest.logical.content_sha256
        ):
            raise RunRecoveryError("recovery artifact bytes differ from their manifest")
        return payload

    def read_index(
        self,
        relative_path: str,
        *,
        expected_schema: str,
    ) -> ProtocolRunRecoveryIndex:
        if expected_schema not in {
            ProtocolRunRecoveryIndex.SCHEMA,
            CandidateRunRecoveryIndex.SCHEMA,
            EnvelopeRunRecoveryIndex.SCHEMA,
            RunRecoveryIndex.SCHEMA,
        }:
            raise RunRecoveryError("run recovery index schema is unsupported")
        payload = self._read_payload(
            relative_path,
            expected_schema=expected_schema,
            maximum_bytes=MAX_RECOVERY_INDEX_BYTES,
        )
        if payload is None:
            raise RunRecoveryError("run recovery index is missing")
        try:
            index = decode_run_recovery_index(payload)
        except (CanonicalizationError, ValueError) as error:
            raise RunRecoveryError("run recovery index is invalid") from error
        if index.index_relative_path != relative_path:
            raise RunRecoveryError("run recovery index path identity differs")
        return index

    def _event_visibility(
        self,
        receipt: CanonicalTaskReceipt | None,
    ) -> tuple[VisibilityCeiling, OutcomeAccess]:
        if receipt is None or not receipt.output_logical_artifacts:
            return VisibilityCeiling.PROSPECTIVE, OutcomeAccess.OUTCOME_BLIND
        return (
            VisibilityCeiling.most_restrictive(
                VisibilityCeiling.PROSPECTIVE,
                *(value.visibility_ceiling for value in receipt.output_logical_artifacts),
            ),
            most_restrictive_outcome_access(
                OutcomeAccess.OUTCOME_BLIND,
                *(value.outcome_access for value in receipt.output_logical_artifacts),
            ),
        )

    def append_event(
        self,
        index: ProtocolRunRecoveryIndex,
        event: ProtocolTaskRecoveryEvent,
        *,
        receipt: CanonicalTaskReceipt | None = None,
    ) -> None:
        path = self._event_path(index, event)
        visibility, outcome_access = self._event_visibility(receipt)
        parents = [
            ArtifactLineageParent(
                identity=ObjectIdentity.from_record(index.recovery_index_id, index),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
        ]
        if receipt is not None:
            parents.append(
                ArtifactLineageParent(
                    identity=ObjectIdentity.from_record(receipt.receipt_id, receipt),
                    visibility_ceiling=visibility,
                    outcome_access=outcome_access,
                )
            )
        lineage = tuple(sorted(parents, key=lineage_parent_sort_key))
        self.plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=event.event_id,
                relative_path=path,
                payload_schema=event.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=f"run-recovery-events.{index.run_id}.{index.wave_id}",
                publication_scope_relative_root=(
                    f"runs/{index.run_id}/recovery/{index.wave_id}/events"
                ),
                payload=event.canonical_bytes(),
                visibility_ceiling=visibility,
                parent_visibility_ceilings=tuple(parent.visibility_ceiling for parent in lineage),
                outcome_access=outcome_access,
                logical_content_sha256=event.fingerprint(),
                lineage_parents=lineage,
                minimum_free_bytes=self.minimum_free_bytes,
            )
        )

    @staticmethod
    def _envelope_event_path(
        index: EnvelopeRunRecoveryIndex,
        event: ExecutionEnvelopeEvent,
    ) -> str:
        return (
            f"{index.envelope_event_root_relative_path}/"
            f"{event.sequence_number:09d}.{event.event_id}.json"
        )

    def append_envelope_event(
        self,
        index: EnvelopeRunRecoveryIndex,
        event: ExecutionEnvelopeEvent,
        *,
        visibility_ceiling: VisibilityCeiling,
        outcome_access: OutcomeAccess,
    ) -> str:
        """Persist one immutable envelope event before it is trusted for recovery."""

        relative_path = self._envelope_event_path(index, event)
        parent = ArtifactLineageParent(
            identity=ObjectIdentity.from_record(index.recovery_index_id, index),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        self.plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=event.event_id,
                relative_path=relative_path,
                payload_schema=ExecutionEnvelopeEvent.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=(f"run-envelope-events.{index.run_id}.{index.wave_id}"),
                publication_scope_relative_root=index.envelope_event_root_relative_path,
                payload=event.canonical_bytes(),
                visibility_ceiling=visibility_ceiling,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                outcome_access=outcome_access,
                logical_content_sha256=event.fingerprint(),
                lineage_parents=(parent,),
                minimum_free_bytes=self.minimum_free_bytes,
            )
        )
        observed = self.read_envelope_event(index, relative_path)
        if observed != event:
            raise RunRecoveryError("persisted execution-envelope event differs")
        return relative_path

    def read_envelope_event(
        self,
        index: EnvelopeRunRecoveryIndex,
        relative_path: str,
    ) -> ExecutionEnvelopeEvent:
        expected_prefix = f"{index.envelope_event_root_relative_path}/"
        if not relative_path.startswith(expected_prefix):
            raise RunRecoveryError("execution-envelope event lies outside its root")
        payload = self._read_payload(
            relative_path,
            expected_schema=ExecutionEnvelopeEvent.SCHEMA,
            maximum_bytes=MAX_RECOVERY_EVENT_BYTES,
        )
        if payload is None:
            raise RunRecoveryError("execution-envelope event is missing")
        try:
            event = decode_execution_envelope_event(payload)
        except (CanonicalizationError, ValueError) as error:
            raise RunRecoveryError("execution-envelope event is invalid") from error
        if self._envelope_event_path(index, event) != relative_path:
            raise RunRecoveryError("execution-envelope event path identity differs")
        return event

    def reconstruct_execution_envelope(
        self,
        *,
        index: EnvelopeRunRecoveryIndex,
        envelope_id: str,
        spec: ExecutionEnvelopeSpec,
        execution_plan: ObjectIdentity,
        event_relative_paths: tuple[str, ...],
    ) -> RunExecutionEnvelope:
        """Read an exact caller-bound event roster; never scan or infer missing events."""

        if len(event_relative_paths) > MAX_EXECUTION_ENVELOPE_EVENTS:
            raise RunRecoveryError("execution-envelope event count exceeds its bound")
        events = tuple(
            self.read_envelope_event(index, relative_path) for relative_path in event_relative_paths
        )
        return reconstruct_execution_envelope_from_external_events(
            index=index,
            envelope_id=envelope_id,
            spec=spec,
            execution_plan=execution_plan,
            event_payloads=tuple(value.canonical_bytes() for value in events),
        )

    def reconstruct_execution_resource_envelope(
        self,
        *,
        index: RunRecoveryIndex,
        envelope_id: str,
        spec: ExecutionResourceEnvelopeSpec,
        execution_plan: ObjectIdentity,
        jit_graph_signature_manifest: JitGraphSignatureManifest | None,
    ) -> RunExecutionResourceEnvelope:
        """Authenticate the same journal used for durable reservation admission."""

        if (
            index.resource_envelope_event_root_relative_path
            != DEFAULT_EXECUTION_RESOURCE_ENVELOPE_STATE_ROOT
            or envelope_id != f"run-resource-envelope.{index.run_id}"
            or index.execution_resource_envelope_spec
            != ObjectIdentity.from_record(spec.envelope_spec_id, spec)
            or index.execution_plan != execution_plan
            or spec.issued_study_extensions != index.issued_extension_set
            or index.jit_graph_signature_manifest != spec.jit_graph_signature_manifest
            or spec.jit_graph_signature_manifest
            != (
                None
                if jit_graph_signature_manifest is None
                else ObjectIdentity.from_record(
                    jit_graph_signature_manifest.manifest_id, jit_graph_signature_manifest
                )
            )
        ):
            raise RunRecoveryError("resource-envelope recovery inputs differ from the index")
        try:
            envelope = ExternalExecutionResourceEnvelopeStore(self.plane.root).load(envelope_id)
        except (KeyError, OSError, ValueError) as error:
            raise RunRecoveryError("resource-envelope journal recovery failed") from error
        if envelope.spec != spec or envelope.execution_plan != execution_plan:
            raise RunRecoveryError("resource-envelope journal substitutes the issued plan")
        return envelope

    @staticmethod
    def _event_path(index: ProtocolRunRecoveryIndex, event: ProtocolTaskRecoveryEvent) -> str:
        task = index.task(event.task_id)
        if event.disposition is TaskRecoveryDisposition.NOT_ATTEMPTED:
            return task.nonattempt_event_relative_path
        assert event.attempt_id is not None
        for candidate in task.attempts:
            if candidate.attempt_id == event.attempt_id:
                return candidate.event_relative_path
        raise RunRecoveryError("task recovery event lies outside its index")

    def read_event(
        self,
        index: ProtocolRunRecoveryIndex,
        relative_path: str,
    ) -> TaskRecoveryEvent | None:
        payload = self._read_payload(
            relative_path,
            expected_schema=index.TASK_EVENT_SCHEMA,
            maximum_bytes=MAX_RECOVERY_EVENT_BYTES,
        )
        if payload is None:
            return None
        try:
            event = decode_task_recovery_event(payload)
        except (CanonicalizationError, ValueError) as error:
            raise RunRecoveryError("task recovery event is invalid") from error
        if (
            event.recovery_index != ObjectIdentity.from_record(index.recovery_index_id, index)
            or event.run_id != index.run_id
            or self._event_path(index, event) != relative_path
        ):
            raise RunRecoveryError("task recovery event differs from its index")
        return event

    def record_receipt(
        self,
        index: ProtocolRunRecoveryIndex,
        receipt: CanonicalTaskReceipt,
    ) -> None:
        event = TaskRecoveryEvent.from_receipt(index, receipt)
        self.append_event(index, event, receipt=receipt)

    def record_failure(
        self,
        index: ProtocolRunRecoveryIndex,
        *,
        task_id: str,
        attempt_id: str,
        reason_code: str,
        failure: OperationalFailureDiagnostic,
        blocked: bool = False,
    ) -> None:
        task = index.task(task_id)
        candidate = next(
            (value for value in task.attempts if value.attempt_id == attempt_id),
            None,
        )
        if candidate is None:
            raise RunRecoveryError("failed attempt lies outside its recovery index")
        event = TaskRecoveryEvent(
            event_id=f"recovery-event.{attempt_id}.failed",
            recovery_index=ObjectIdentity.from_record(index.recovery_index_id, index),
            run_id=index.run_id,
            task_id=task_id,
            attempt_id=attempt_id,
            attempt_ordinal=candidate.ordinal,
            disposition=(
                TaskRecoveryDisposition.BLOCKED if blocked else TaskRecoveryDisposition.FAILED
            ),
            receipt_id=None,
            receipt_relative_path=None,
            receipt_sha256=None,
            receipt_schema=None,
            output_materialization_ids=(),
            reason_codes=(reason_code,),
            failure=failure,
        )
        self.append_event(index, event)

    def record_not_attempted(
        self,
        index: ProtocolRunRecoveryIndex,
        *,
        task_id: str,
        reason: TaskBlockReason,
    ) -> None:
        if reason.kind.value != "DURABLE":
            return
        event = TaskRecoveryEvent(
            event_id=f"recovery-event.{index.run_id}.{task_id}.not-attempted",
            recovery_index=ObjectIdentity.from_record(index.recovery_index_id, index),
            run_id=index.run_id,
            task_id=task_id,
            attempt_id=None,
            attempt_ordinal=None,
            disposition=TaskRecoveryDisposition.NOT_ATTEMPTED,
            receipt_id=None,
            receipt_relative_path=None,
            receipt_sha256=None,
            receipt_schema=None,
            output_materialization_ids=(),
            reason_codes=(reason.value,),
            failure=None,
        )
        self.append_event(index, event)

    def read_terminal(
        self,
        index: ProtocolRunRecoveryIndex,
    ) -> ProtocolRunRecoveryTerminalEvent | None:
        payload = self._read_payload(
            index.terminal_event_relative_path,
            expected_schema=index.terminal_event_schema,
            maximum_bytes=MAX_RECOVERY_INDEX_BYTES,
        )
        if payload is None:
            return None
        try:
            event = decode_run_recovery_terminal_event(payload)
        except (CanonicalizationError, ValueError) as error:
            raise RunRecoveryError("run recovery terminal event is invalid") from error
        if (
            event.recovery_index != ObjectIdentity.from_record(index.recovery_index_id, index)
            or event.run_id != index.run_id
        ):
            raise RunRecoveryError("run recovery terminal event differs from its index")
        return event

    def record_terminal(
        self,
        index: ProtocolRunRecoveryIndex,
        *,
        status: OperationalStatus,
        attempts: tuple[OperationalAttempt, ...],
        receipts: tuple[CanonicalTaskReceipt, ...],
    ) -> None:
        if status not in {
            OperationalStatus.SUCCEEDED,
            OperationalStatus.FAILED,
            OperationalStatus.BLOCKED,
        }:
            raise RunRecoveryError("run recovery closure requires a terminal status")
        latest_operational = {
            attempt.task_id: attempt
            for attempt in sorted(attempts, key=lambda value: (value.task_id, value.ordinal))
        }
        if status is OperationalStatus.BLOCKED:
            for attempt in latest_operational.values():
                if (
                    attempt.disposition is not TaskAttemptDisposition.BLOCKED
                    or attempt.block_kind is None
                    or attempt.block_kind.value != "RETRYABLE"
                ):
                    continue
                binding = index.task(attempt.task_id)
                candidate_ids = {value.attempt_id for value in binding.attempts}
                consumed = sum(
                    value.task_id == attempt.task_id
                    and value.attempt_id in candidate_ids
                    and value.disposition
                    in {
                        TaskAttemptDisposition.FAILED,
                        TaskAttemptDisposition.BLOCKED,
                    }
                    for value in attempts
                )
                if attempt.attempt_id not in candidate_ids or consumed < binding.maximum_attempts:
                    return
        receipts_by_attempt = {receipt.attempt_id: receipt for receipt in receipts}
        terminal_attempts: list[RecoveryTerminalAttempt] = []
        for attempt in sorted(attempts, key=lambda value: (value.task_id, value.ordinal)):
            if attempt.disposition is TaskAttemptDisposition.RUNNING:
                raise RunRecoveryError("terminal recovery closure contains a live attempt")
            if attempt.disposition is TaskAttemptDisposition.SUCCEEDED:
                receipt = receipts_by_attempt.get(attempt.attempt_id)
                if receipt is None:
                    raise RunRecoveryError("successful terminal attempt lacks its receipt")
                disposition = RecoveryTerminalDisposition.SUCCEEDED
                reason_code = None
                receipt_id = receipt.receipt_id
                receipt_sha256 = receipt.fingerprint()
            else:
                if attempt.reason_code is None:
                    raise RunRecoveryError("non-success terminal attempt lacks its reason")
                disposition = RecoveryTerminalDisposition(attempt.disposition.value)
                reason_code = attempt.reason_code
                receipt_id = None
                receipt_sha256 = None
            terminal_attempts.append(
                RecoveryTerminalAttempt(
                    attempt_id=attempt.attempt_id,
                    task_id=attempt.task_id,
                    ordinal=attempt.ordinal,
                    disposition=disposition,
                    reason_code=reason_code,
                    receipt_id=receipt_id,
                    receipt_sha256=receipt_sha256,
                )
            )
        if set(receipts_by_attempt) != {
            value.attempt_id
            for value in terminal_attempts
            if value.disposition is RecoveryTerminalDisposition.SUCCEEDED
        }:
            raise RunRecoveryError("terminal receipt set differs from successful attempts")
        latest = {attempt.task_id: attempt.disposition for attempt in terminal_attempts}
        event_type = (
            RunRecoveryTerminalEvent
            if isinstance(index, RunRecoveryIndex)
            else EnvelopeRunRecoveryTerminalEvent
            if isinstance(index, EnvelopeRunRecoveryIndex)
            else CandidateRunRecoveryTerminalEvent
            if isinstance(index, CandidateRunRecoveryIndex)
            else ProtocolRunRecoveryTerminalEvent
        )
        event = event_type(
            terminal_event_id=f"recovery-terminal.{index.run_id}.{index.wave_id}",
            recovery_index=ObjectIdentity.from_record(index.recovery_index_id, index),
            run_id=index.run_id,
            operational_status=status.value,
            attempts=tuple(terminal_attempts),
            completed_task_ids=tuple(
                sorted(
                    task_id
                    for task_id, disposition in latest.items()
                    if disposition is RecoveryTerminalDisposition.SUCCEEDED
                )
            ),
            failed_task_ids=tuple(
                sorted(
                    task_id
                    for task_id, disposition in latest.items()
                    if disposition is RecoveryTerminalDisposition.FAILED
                )
            ),
            blocked_task_ids=tuple(
                sorted(
                    task_id
                    for task_id, disposition in latest.items()
                    if disposition is RecoveryTerminalDisposition.BLOCKED
                )
            ),
        )
        visibility, outcome_access = self._event_visibility(next(iter(receipts), None))
        if receipts:
            visibility = VisibilityCeiling.most_restrictive(
                VisibilityCeiling.PROSPECTIVE,
                *(
                    logical.visibility_ceiling
                    for receipt in receipts
                    for logical in receipt.output_logical_artifacts
                ),
            )
            outcome_access = most_restrictive_outcome_access(
                OutcomeAccess.OUTCOME_BLIND,
                *(
                    logical.outcome_access
                    for receipt in receipts
                    for logical in receipt.output_logical_artifacts
                ),
            )
        # The terminal event is the bounded aggregate commitment to every
        # attempted receipt: each successful attempt carries the exact receipt
        # ID and fingerprint, while the recovery index fixes its candidate path.
        # Repeating every receipt as a manifest lineage parent makes terminal
        # publication grow past the artifact-manifest bound for otherwise valid
        # large DAGs.  Keep direct lineage at the aggregate index and preserve
        # the most restrictive receipt evidence class on the terminal artifact.
        parents = (
            ArtifactLineageParent(
                identity=ObjectIdentity.from_record(index.recovery_index_id, index),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            ),
        )
        self.plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=event.terminal_event_id,
                relative_path=index.terminal_event_relative_path,
                payload_schema=event.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id=f"run-recovery.{index.run_id}.{index.wave_id}",
                publication_scope_relative_root=(f"runs/{index.run_id}/recovery/{index.wave_id}"),
                payload=event.canonical_bytes(),
                visibility_ceiling=visibility,
                parent_visibility_ceilings=tuple(parent.visibility_ceiling for parent in parents),
                outcome_access=outcome_access,
                logical_content_sha256=event.fingerprint(),
                lineage_parents=parents,
                minimum_free_bytes=self.minimum_free_bytes,
            )
        )
        if self.read_terminal(index) != event:
            raise RunRecoveryError("persisted terminal recovery event differs")

    @staticmethod
    def _validate_receipt_static(
        index: ProtocolRunRecoveryIndex,
        task: ProtocolExecutionTask,
        candidate: RecoveryAttemptCandidate,
        receipt: CanonicalTaskReceipt,
    ) -> None:
        logical = tuple(
            sorted(
                (
                    value.logical_artifact_id,
                    value.payload_schema,
                )
                for value in receipt.output_logical_artifacts
            )
        )
        expected = tuple(
            sorted(
                (
                    value.logical_artifact_id,
                    value.payload_schema,
                )
                for value in task.outputs
            )
        )
        if (
            receipt.run_id != index.run_id
            or receipt.task_id != task.task_id
            or receipt.attempt_id != candidate.attempt_id
            or receipt.implementation_commit != index.implementation_commit
            or receipt.operational_status is not OperationalStatus.SUCCEEDED
            or not receipt.checks
            or any(not check.passed for check in receipt.checks)
            or receipt.reason_codes
            or logical != expected
            or len(receipt.output_materializations) != len(task.outputs)
        ):
            raise RunRecoveryError("task receipt differs from the frozen recovery index")

    @staticmethod
    def _restore_attempt(
        repository: OperationalRepository,
        *,
        run_id: str,
        attempt_id: str,
        task_id: str,
        disposition: TaskAttemptDisposition,
        reason_code: str | None,
        existing: dict[str, OperationalAttempt],
    ) -> None:
        prior = existing.get(attempt_id)
        if prior is not None:
            if prior.task_id != task_id or prior.run_id != run_id:
                raise RunRecoveryError("operational attempt identity conflicts with recovery")
            if prior.disposition is TaskAttemptDisposition.RUNNING:
                if disposition is TaskAttemptDisposition.SUCCEEDED:
                    repository.complete_attempt(attempt_id)
                elif disposition is TaskAttemptDisposition.FAILED:
                    assert reason_code is not None
                    repository.fail_attempt(attempt_id, reason_code)
                else:
                    assert reason_code is not None
                    repository.block_attempt(attempt_id, TaskBlockReason(reason_code))
                return
            if prior.disposition is not disposition or prior.reason_code != reason_code:
                raise RunRecoveryError("operational attempt differs from external recovery")
            return
        repository.start_attempt(
            run_id=run_id,
            task_id=task_id,
            attempt_id=attempt_id,
            lease_id=f"lease.recovered.{attempt_id}",
            owner_id="external-run-recovery",
            expires_epoch_seconds=1,
        )
        if disposition is TaskAttemptDisposition.SUCCEEDED:
            repository.complete_attempt(attempt_id)
        elif disposition is TaskAttemptDisposition.FAILED:
            assert reason_code is not None
            repository.fail_attempt(attempt_id, reason_code)
        else:
            assert reason_code is not None
            repository.block_attempt(attempt_id, TaskBlockReason(reason_code))

    def reconcile(
        self,
        index: ProtocolRunRecoveryIndex,
        plan: ProtocolExecutionPlan,
        repository: OperationalRepository,
        receipt_store: TaskReceiptStore,
        *,
        retry_amendment: LeaseExpiryRetryAmendment | None = None,
    ) -> RunRecoveryReport:
        """Rebuild bounded run/task state without directory scans or workers."""

        try:
            index.validate_plan(plan, retry_amendment=retry_amendment)
        except ValueError as error:
            raise RunRecoveryError("run recovery index does not authorize this plan") from error
        if (
            self.read_index(
                index.index_relative_path,
                expected_schema=index.SCHEMA,
            )
            != index
        ):
            raise RunRecoveryError("external recovery index differs from the supplied index")
        repository.register_run(index.run_id, plan.fingerprint())
        existing = {attempt.attempt_id: attempt for attempt in repository.attempts(index.run_id)}
        tasks = {task.task_id: task for task in plan.tasks}
        candidate_count = sum(len(task.attempts) for task in index.tasks)
        if candidate_count > MAX_RECOVERY_CANDIDATES:
            raise RunRecoveryError("recovery index exceeds its aggregate work bound")
        retained: dict[str, str] = {}
        if isinstance(retry_amendment, RetainedCustodyCompletionAmendment):
            prior = decode_run_recovery_terminal_event(read_bounded_bytes(
                self.plane.root.resolve(retry_amendment.retained_terminal_path, for_write=False),
                maximum_bytes=MAX_CONTROL_PLANE_JSON_BYTES,
            ))
            if (
                ObjectIdentity.from_record(prior.terminal_event_id, prior) != retry_amendment.prior_repair_terminal
                or prior.run_id != index.run_id
                or prior.operational_status != "FAILED"
                or prior.failed_task_ids != retry_amendment.task_ids
                or prior.blocked_task_ids
            ):
                raise RunRecoveryError("minimal recovery lacks its exact retained failed terminal")
            retained = {a.attempt_id: a.receipt_sha256 for a in prior.attempts if a.receipt_sha256 is not None}
        terminal = self.read_terminal(index)
        if terminal is not None:
            return self._reconcile_terminal(
                index=index,
                plan=plan,
                terminal=terminal,
                repository=repository,
                receipt_store=receipt_store,
                existing=existing,
                candidate_count=candidate_count,
                retained=retained,
            )
        receipt_probes = 0
        event_probes = 0
        recovered_attempts = 0
        recovered_tasks: set[str] = set()
        recovered_receipts: dict[str, CanonicalTaskReceipt] = {}
        completed: set[str] = set()
        exhausted: set[str] = set()
        durable_blocked: set[str] = set()

        for task_id in plan.topological_task_ids():
            binding = index.task(task_id)
            task = tasks[task_id]
            failed = 0
            terminal_attempts = 0
            observed_attempt = False
            success_seen = False
            for candidate in binding.attempts:
                receipt_probes += 1
                event_probes += 1
                try:
                    receipt = receipt_store.read(
                        index.run_id,
                        task_id,
                        candidate.attempt_id,
                    )
                    event = self.read_event(index, candidate.event_relative_path)
                except (ArtifactIdentityConflict, OSError, ValueError) as error:
                    raise RunRecoveryError("bounded task recovery probe failed") from error
                if receipt is not None:
                    if success_seen or candidate.ordinal != terminal_attempts + 1:
                        raise RunRecoveryError(
                            "successful receipt has a missing or later terminal predecessor"
                        )
                    self._validate_receipt_static(index, task, candidate, receipt)
                    expected_event = TaskRecoveryEvent.from_receipt(index, receipt)
                    if candidate.attempt_id in retained:
                        if receipt.fingerprint() != retained[candidate.attempt_id]:
                            raise RunRecoveryError("minimal recovery changes a retained receipt")
                        if event is not None and event != expected_event:
                            raise RunRecoveryError("retained receipt conflicts with a recovery event")
                    elif event is None:
                        self.append_event(index, expected_event, receipt=receipt)
                    elif event != expected_event:
                        raise RunRecoveryError("task receipt conflicts with its recovery event")
                    self._restore_attempt(
                        repository,
                        run_id=index.run_id,
                        attempt_id=candidate.attempt_id,
                        task_id=task_id,
                        disposition=TaskAttemptDisposition.SUCCEEDED,
                        reason_code=None,
                        existing=existing,
                    )
                    recovered_attempts += 1
                    recovered_tasks.add(task_id)
                    recovered_receipts[receipt.receipt_id] = receipt
                    completed.add(task_id)
                    observed_attempt = True
                    success_seen = True
                    terminal_attempts += 1
                    continue
                if event is None:
                    continue
                if event.disposition not in {
                    TaskRecoveryDisposition.FAILED,
                    TaskRecoveryDisposition.BLOCKED,
                }:
                    raise RunRecoveryError("receipt-free attempt has a nonterminal event")
                if success_seen or candidate.ordinal != terminal_attempts + 1:
                    raise RunRecoveryError(
                        "failed recovery event has a missing or successful predecessor"
                    )
                reason_code = event.reason_codes[0]
                disposition = (
                    TaskAttemptDisposition.BLOCKED
                    if event.disposition is TaskRecoveryDisposition.BLOCKED
                    else TaskAttemptDisposition.FAILED
                )
                self._restore_attempt(
                    repository,
                    run_id=index.run_id,
                    attempt_id=candidate.attempt_id,
                    task_id=task_id,
                    disposition=disposition,
                    reason_code=reason_code,
                    existing=existing,
                )
                if disposition is TaskAttemptDisposition.FAILED:
                    failed += 1
                elif TaskBlockReason(reason_code).kind.value == "DURABLE":
                    durable_blocked.add(task_id)
                recovered_attempts += 1
                recovered_tasks.add(task_id)
                observed_attempt = True
                terminal_attempts += 1
            event_probes += 1
            nonattempt = self.read_event(index, binding.nonattempt_event_relative_path)
            if nonattempt is not None and not observed_attempt:
                reason = TaskBlockReason(nonattempt.reason_codes[0])
                if reason.kind.value != "DURABLE":
                    raise RunRecoveryError("nonattempt recovery event is not terminal")
                prior_blocks = tuple(
                    attempt
                    for attempt in existing.values()
                    if attempt.task_id == task_id
                    and attempt.disposition is TaskAttemptDisposition.BLOCKED
                )
                if prior_blocks:
                    if any(attempt.reason_code != reason.value for attempt in prior_blocks):
                        raise RunRecoveryError("operational block differs from external recovery")
                else:
                    repository.block_task(index.run_id, task_id, reason)
                    recovered_attempts += 1
                    recovered_tasks.add(task_id)
                durable_blocked.add(task_id)
            if failed >= binding.maximum_attempts:
                exhausted.add(task_id)

        if len(completed) == len(index.tasks):
            status = OperationalStatus.SUCCEEDED
        elif exhausted:
            status = OperationalStatus.FAILED
        elif durable_blocked:
            status = OperationalStatus.BLOCKED
        else:
            status = OperationalStatus.PENDING
        repository.set_run_status(index.run_id, status)
        return RunRecoveryReport(
            candidate_count=candidate_count,
            receipt_probe_count=receipt_probes,
            event_probe_count=event_probes,
            recovered_attempt_count=recovered_attempts,
            recovered_task_ids=tuple(sorted(recovered_tasks)),
            recovered_receipt_ids=tuple(sorted(recovered_receipts)),
            recovered_receipts=tuple(
                recovered_receipts[receipt_id] for receipt_id in sorted(recovered_receipts)
            ),
            terminal_status=None,
            completed_task_ids=(),
            failed_task_ids=(),
            blocked_task_ids=(),
        )

    def _reconcile_terminal(
        self,
        *,
        index: ProtocolRunRecoveryIndex,
        plan: ProtocolExecutionPlan,
        terminal: ProtocolRunRecoveryTerminalEvent,
        repository: OperationalRepository,
        receipt_store: TaskReceiptStore,
        existing: dict[str, OperationalAttempt],
        candidate_count: int,
        retained: dict[str, str],
    ) -> RunRecoveryReport:
        summaries = {attempt.attempt_id: attempt for attempt in terminal.attempts}
        consumed: set[str] = set()
        recovered_receipts: dict[str, CanonicalTaskReceipt] = {}
        tasks = {task.task_id: task for task in plan.tasks}
        receipt_probes = 0
        event_probes = 0
        for binding in index.tasks:
            task = tasks[binding.task_id]
            for candidate in binding.attempts:
                receipt_probes += 1
                event_probes += 1
                try:
                    receipt = receipt_store.read(
                        index.run_id,
                        binding.task_id,
                        candidate.attempt_id,
                    )
                    event = self.read_event(index, candidate.event_relative_path)
                except (ArtifactIdentityConflict, OSError, ValueError) as error:
                    raise RunRecoveryError("terminal task recovery custody is invalid") from error
                summary = summaries.get(candidate.attempt_id)
                if summary is None:
                    if receipt is not None or event is not None:
                        raise RunRecoveryError(
                            "terminal recovery closure omits a persisted attempt"
                        )
                    continue
                consumed.add(candidate.attempt_id)
                # Candidate ordinals count executable attempts. Terminal
                # ordinals count the complete operational history, including
                # prior authority/dependency blocks. Identity is therefore
                # carried by the bounded attempt ID and task; the terminal
                # record independently requires contiguous task-local order.
                if summary.task_id != binding.task_id:
                    raise RunRecoveryError("terminal attempt differs from its bounded candidate")
                if summary.disposition is RecoveryTerminalDisposition.SUCCEEDED:
                    if receipt is None:
                        raise RunRecoveryError("terminal task receipt is missing")
                    self._validate_receipt_static(index, task, candidate, receipt)
                    expected_event = TaskRecoveryEvent.from_receipt(index, receipt)
                    if (
                        (event != expected_event and not (
                            event is None and retained.get(candidate.attempt_id) == receipt.fingerprint()
                        ))
                        or summary.receipt_id != receipt.receipt_id
                        or summary.receipt_sha256 != receipt.fingerprint()
                    ):
                        raise RunRecoveryError(
                            "terminal task receipt or recovery event was substituted"
                        )
                    recovered_receipts[receipt.receipt_id] = receipt
                elif summary.disposition is RecoveryTerminalDisposition.FAILED:
                    if (
                        receipt is not None
                        or event is None
                        or event.disposition is not TaskRecoveryDisposition.FAILED
                        or event.reason_codes != (summary.reason_code,)
                    ):
                        raise RunRecoveryError(
                            "terminal failed attempt lacks exact external evidence"
                        )
                elif receipt is not None or (
                    event is not None
                    and (
                        event.disposition is not TaskRecoveryDisposition.BLOCKED
                        or event.reason_codes != (summary.reason_code,)
                    )
                ):
                    raise RunRecoveryError("terminal blocked attempt has conflicting evidence")
            event_probes += 1
            try:
                nonattempt = self.read_event(
                    index,
                    binding.nonattempt_event_relative_path,
                )
            except (ArtifactIdentityConflict, OSError, ValueError) as error:
                raise RunRecoveryError("terminal nonattempt recovery custody is invalid") from error
            if nonattempt is not None:
                matching = tuple(
                    attempt
                    for attempt in terminal.attempts
                    if attempt.task_id == binding.task_id
                    and attempt.disposition is RecoveryTerminalDisposition.BLOCKED
                    and attempt.reason_code == nonattempt.reason_codes[0]
                )
                if not matching:
                    raise RunRecoveryError("terminal recovery closure omits its nonattempt event")
        if consumed != {
            attempt.attempt_id
            for attempt in terminal.attempts
            if attempt.attempt_id
            in {candidate.attempt_id for binding in index.tasks for candidate in binding.attempts}
        }:
            raise RunRecoveryError("terminal recovery closure contains unknown attempts")
        for summary in terminal.attempts:
            if summary.task_id not in tasks:
                raise RunRecoveryError("terminal recovery closure names an unknown task")
            disposition = TaskAttemptDisposition(summary.disposition.value)
            self._restore_attempt(
                repository,
                run_id=index.run_id,
                attempt_id=summary.attempt_id,
                task_id=summary.task_id,
                disposition=disposition,
                reason_code=summary.reason_code,
                existing=existing,
            )
        repository.set_run_status(
            index.run_id,
            OperationalStatus(terminal.operational_status),
        )
        return RunRecoveryReport(
            candidate_count=candidate_count,
            receipt_probe_count=receipt_probes,
            event_probe_count=event_probes,
            recovered_attempt_count=len(terminal.attempts),
            recovered_task_ids=tuple(
                sorted(
                    {
                        *terminal.completed_task_ids,
                        *terminal.failed_task_ids,
                        *terminal.blocked_task_ids,
                    }
                )
            ),
            recovered_receipt_ids=tuple(sorted(recovered_receipts)),
            recovered_receipts=tuple(
                recovered_receipts[receipt_id] for receipt_id in sorted(recovered_receipts)
            ),
            terminal_status=terminal.operational_status,
            completed_task_ids=terminal.completed_task_ids,
            failed_task_ids=terminal.failed_task_ids,
            blocked_task_ids=terminal.blocked_task_ids,
        )


__all__ = [
    "ExternalRunRecoveryStore",
    "RunRecoveryError",
    "decode_run_recovery_index",
    "decode_run_recovery_terminal_event",
    "decode_task_recovery_event",
    "decode_execution_envelope_event",
    'decode_execution_resource_envelope_event',
    "reconstruct_execution_envelope_from_external_events",
    'reconstruct_execution_resource_envelope_from_external_events',
]
