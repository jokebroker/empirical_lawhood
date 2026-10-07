"""Capability-scoped source and append-only stream adapter contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.authority import ResourceBudget, SourceAccessClass
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)

from .artifacts import ArtifactMaterialization


class SourceMode(StrEnum):
    ARCHIVAL = "ARCHIVAL"
    STREAMING = "STREAMING"
    SIMULATED = "SIMULATED"


@dataclass(frozen=True, slots=True)
class SourceCapabilityManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-capability-manifest'

    capability_key: str
    capability_version: str
    mode: SourceMode
    source_access: SourceAccessClass
    output_schema_ids: tuple[str, ...]
    maximum_evidence: EvidenceCeiling
    maximum_outcome_access: OutcomeAccess
    deterministic: bool
    implementation_sha256: str
    resource_budget: ResourceBudget

    def __post_init__(self) -> None:
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        require_sorted_unique_strings(
            self.output_schema_ids,
            field_name="output_schema_ids",
            allow_empty=False,
        )
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")


@dataclass(frozen=True, slots=True)
class SourceRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-request'

    request_id: str
    source_id: str
    expected_sha256: str | None
    expected_size_bytes: int | None
    licence_id: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (("request_id", self.request_id), ("source_id", self.source_id)):
            validate_stable_id(value, field_name=name)
        if self.expected_sha256 is not None:
            validate_sha256(self.expected_sha256, field_name="expected_sha256")
        if self.expected_size_bytes is not None and self.expected_size_bytes < 0:
            raise ValueError("expected source size must be nonnegative")
        validate_nonempty(self.licence_id, field_name="licence_id")


@dataclass(frozen=True, slots=True)
class SourceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-result'

    request_id: str
    source_id: str
    payload_schema: str
    materialization: ArtifactMaterialization
    observed_sha256: str
    observed_size_bytes: int
    custody_status: str

    def __post_init__(self) -> None:
        for name, value in (("request_id", self.request_id), ("source_id", self.source_id)):
            validate_stable_id(value, field_name=name)
        validate_schema(self.payload_schema)
        validate_sha256(self.observed_sha256, field_name="observed_sha256")
        if self.observed_size_bytes < 0:
            raise ValueError("observed source size must be nonnegative")
        validate_stable_id(self.custody_status, field_name="custody_status")


@dataclass(frozen=True, slots=True)
class StreamCursor(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/stream-cursor'

    stream_id: str
    next_sequence: int
    predecessor_chunk_sha256: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.stream_id, field_name="stream_id")
        if self.next_sequence < 0:
            raise ValueError("stream sequence must be nonnegative")
        if self.predecessor_chunk_sha256 is not None:
            validate_sha256(
                self.predecessor_chunk_sha256,
                field_name="predecessor_chunk_sha256",
            )


@dataclass(frozen=True, slots=True)
class StreamChunk(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/stream-chunk'

    stream_id: str
    sequence: int
    payload_schema: str
    chunk_sha256: str
    predecessor_chunk_sha256: str | None
    materialization: ArtifactMaterialization
    terminal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.stream_id, field_name="stream_id")
        if self.sequence < 0:
            raise ValueError("stream sequence must be nonnegative")
        validate_schema(self.payload_schema)
        validate_sha256(self.chunk_sha256, field_name="chunk_sha256")
        if self.predecessor_chunk_sha256 is not None:
            validate_sha256(
                self.predecessor_chunk_sha256,
                field_name="predecessor_chunk_sha256",
            )


class SourceAdapter(Protocol):
    manifest: SourceCapabilityManifest

    def acquire(self, request: SourceRequest) -> SourceResult: ...


class StreamAdapter(Protocol):
    manifest: SourceCapabilityManifest

    def read_chunk(self, cursor: StreamCursor) -> StreamChunk: ...


_OUTCOME_ACCESS_RANK = {
    OutcomeAccess.OUTCOME_BLIND: 0,
    OutcomeAccess.EVALUATION_SEALED: 0,
    OutcomeAccess.DEVELOPMENT_VISIBLE: 1,
    OutcomeAccess.EVALUATOR_REVEAL: 2,
    OutcomeAccess.EVALUATION_REVEALED: 2,
    OutcomeAccess.PRIVILEGED_TRUTH: 3,
}


class SourceConformanceError(ValueError):
    pass


def source_conformance_reasons(
    manifest: SourceCapabilityManifest,
    request: SourceRequest,
    result: SourceResult,
) -> tuple[str, ...]:
    """Return deterministic reasons why an archival/simulated result is inadmissible."""

    reasons: list[str] = []
    if manifest.mode is SourceMode.STREAMING:
        reasons.append("SOURCE_MODE_MISMATCH")
    if result.request_id != request.request_id:
        reasons.append("REQUEST_ID_MISMATCH")
    if result.source_id != request.source_id:
        reasons.append("SOURCE_ID_MISMATCH")
    if result.payload_schema not in manifest.output_schema_ids:
        reasons.append("OUTPUT_SCHEMA_NOT_ALLOWED")
    if result.custody_status != "verified":
        reasons.append("CUSTODY_NOT_VERIFIED")
    if (
        _OUTCOME_ACCESS_RANK[request.outcome_access]
        > _OUTCOME_ACCESS_RANK[manifest.maximum_outcome_access]
    ):
        reasons.append("OUTCOME_ACCESS_EXCEEDED")
    if result.observed_sha256 != result.materialization.physical_sha256:
        reasons.append("MATERIALIZATION_HASH_MISMATCH")
    if result.observed_size_bytes != result.materialization.size_bytes:
        reasons.append("MATERIALIZATION_SIZE_MISMATCH")
    if request.expected_sha256 is not None and result.observed_sha256 != request.expected_sha256:
        reasons.append("EXPECTED_HASH_MISMATCH")
    if (
        request.expected_size_bytes is not None
        and result.observed_size_bytes != request.expected_size_bytes
    ):
        reasons.append("EXPECTED_SIZE_MISMATCH")
    return tuple(sorted(reasons))


def validate_source_result(
    manifest: SourceCapabilityManifest,
    request: SourceRequest,
    result: SourceResult,
) -> None:
    reasons = source_conformance_reasons(manifest, request, result)
    if reasons:
        raise SourceConformanceError(",".join(reasons))


def stream_conformance_reasons(
    manifest: SourceCapabilityManifest,
    cursor: StreamCursor,
    chunk: StreamChunk,
) -> tuple[str, ...]:
    """Return deterministic reasons for rejecting a broken append-only stream link."""

    reasons: list[str] = []
    if manifest.mode is not SourceMode.STREAMING:
        reasons.append("SOURCE_MODE_MISMATCH")
    if chunk.stream_id != cursor.stream_id:
        reasons.append("STREAM_ID_MISMATCH")
    if chunk.sequence != cursor.next_sequence:
        reasons.append("STREAM_SEQUENCE_MISMATCH")
    if chunk.predecessor_chunk_sha256 != cursor.predecessor_chunk_sha256:
        reasons.append("STREAM_PREDECESSOR_MISMATCH")
    if chunk.payload_schema not in manifest.output_schema_ids:
        reasons.append("OUTPUT_SCHEMA_NOT_ALLOWED")
    if chunk.chunk_sha256 != chunk.materialization.physical_sha256:
        reasons.append("MATERIALIZATION_HASH_MISMATCH")
    return tuple(sorted(reasons))


def validate_stream_chunk(
    manifest: SourceCapabilityManifest,
    cursor: StreamCursor,
    chunk: StreamChunk,
) -> None:
    reasons = stream_conformance_reasons(manifest, cursor, chunk)
    if reasons:
        raise SourceConformanceError(",".join(reasons))
