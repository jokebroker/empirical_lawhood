"""Bounded FAIR-MAST inspection and external-only materialization service."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar, Protocol
from urllib.parse import urlsplit

from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)

from .contracts import (
    FAIR_MAST_LICENSE_ID,
    FAIR_MAST_SHOTS_METADATA_SHA256,
    FAIR_MAST_SOURCES_METADATA_SHA256,
    FairMastChunkReceipt,
    FairMastMaterializationReceipt,
    FairMastSignalSpec,
    FairMastSourceDeclaration,
    MastCampaign,
    MastMaterializationState,
)


MAX_OPERATOR_CONFIG_BYTES = 64 * 1024
_ALLOWED_FAIR_MAST_HOSTS = frozenset({"mastapp.site"})


@dataclass(frozen=True, slots=True)
class FairMastOperatorConfig(CanonicalRecord):
    """Operator-owned endpoint binding; never an experiment/scientific input."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-operator-config'

    config_id: str
    service_id: str
    base_url: str
    api_version: str
    request_timeout_seconds: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.service_id, field_name="service_id")
        validate_stable_id(self.api_version, field_name="api_version")
        parts = urlsplit(self.base_url)
        if (
            parts.scheme != "https"
            or parts.hostname not in _ALLOWED_FAIR_MAST_HOSTS
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
            or parts.path not in {"", "/"}
        ):
            raise ValueError("operator FAIR-MAST endpoint is outside the static allowlist")
        if not 1 <= self.request_timeout_seconds <= 300:
            raise ValueError("FAIR-MAST request timeout is outside the bounded range")


def decode_fair_mast_operator_config(payload: bytes) -> FairMastOperatorConfig:
    return decode_canonical_bytes(
        payload,
        FairMastOperatorConfig,
        maximum_bytes=MAX_OPERATOR_CONFIG_BYTES,
    )


@dataclass(frozen=True, slots=True)
class FairMastMetadataSnapshot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-metadata-snapshot'

    snapshot_id: str
    shots_metadata_sha256: str
    sources_metadata_sha256: str
    license_id: str
    campaigns: tuple[MastCampaign, ...]
    groups: tuple[str, ...]
    signals: tuple[FairMastSignalSpec, ...]
    preview_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.snapshot_id, field_name="snapshot_id")
        validate_sha256(self.shots_metadata_sha256, field_name="shots_metadata_sha256")
        validate_sha256(self.sources_metadata_sha256, field_name="sources_metadata_sha256")
        validate_stable_id(self.license_id, field_name="license_id")
        require_sorted_unique_strings(self.groups, field_name="groups", allow_empty=False)
        require_sorted_unique_ids(self.signals, attribute="signal_id", field_name="signals")
        if self.preview_bytes < 0:
            raise ValueError("preview_bytes must be nonnegative")


@dataclass(frozen=True, slots=True)
class FairMastPreview(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-preview'

    preview_id: str
    source: ObjectIdentity
    metadata: FairMastMetadataSnapshot
    requested_budget: ResourceBudget
    conforms: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.preview_id, field_name="preview_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.conforms == bool(self.reason_codes):
            raise ValueError("FAIR-MAST preview conformance differs from its reasons")
        if self.metadata.preview_bytes > self.requested_budget.source_scan_bytes:
            raise ValueError("FAIR-MAST preview exceeds the authorized byte budget")


@dataclass(frozen=True, slots=True)
class FairMastChunkSelection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-chunk-selection'

    chunk_id: str
    signal_id: str
    chunk_index: int
    expected_source_sha256: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.chunk_id, field_name="chunk_id")
        validate_stable_id(self.signal_id, field_name="signal_id")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be nonnegative")
        if self.expected_source_sha256 is not None:
            validate_sha256(self.expected_source_sha256, field_name="expected_source_sha256")


@dataclass(frozen=True, slots=True)
class FairMastMaterializationRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-materialization-request'

    request_id: str
    shot_id: int
    campaign: MastCampaign
    chunks: tuple[FairMastChunkSelection, ...]
    destination_prefix: str
    external_root: ObjectIdentity
    source_authorization: ObjectIdentity
    write_authorization: ObjectIdentity
    budget: ResourceBudget

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        if self.shot_id <= 0:
            raise ValueError("shot_id must be positive")
        require_sorted_unique_ids(self.chunks, attribute="chunk_id", field_name="chunks")
        if not self.chunks:
            raise ValueError("materialization request requires declared chunks")
        validate_relative_locator(self.destination_prefix)
        if not self.destination_prefix.startswith("mast-torax/fair-mast/"):
            raise ValueError("FAIR-MAST destination is outside its external-store namespace")
        if self.source_authorization == self.write_authorization:
            raise ValueError("source acquisition and external write require distinct authority")


@dataclass(frozen=True, slots=True)
class FairMastFetchedChunk:
    """Ephemeral gateway response; bytes must be published before a receipt exists."""

    selection: FairMastChunkSelection
    payload: bytes
    media_type: str
    payload_schema: str

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes):
            raise TypeError("FAIR-MAST chunk payload must be immutable bytes")
        validate_nonempty(self.media_type, field_name="media_type")
        if (
            self.selection.expected_source_sha256 is not None
            and sha256(self.payload).hexdigest() != self.selection.expected_source_sha256
        ):
            raise ValueError("FAIR-MAST source drift detected")

    @property
    def source_sha256(self) -> str:
        return sha256(self.payload).hexdigest()


class FairMastGateway(Protocol):
    def inspect_metadata(
        self,
        operator: FairMastOperatorConfig,
        declaration: FairMastSourceDeclaration,
        *,
        maximum_bytes: int,
    ) -> FairMastMetadataSnapshot: ...

    def fetch_shot_chunks(
        self,
        operator: FairMastOperatorConfig,
        request: FairMastMaterializationRequest,
    ) -> tuple[FairMastFetchedChunk, ...]: ...


class FairMastAuthorityVerifier(Protocol):
    def require(
        self,
        authorization: ObjectIdentity,
        action: AuthorityAction,
        scope_id: str,
        budget: ResourceBudget,
    ) -> None: ...


class ExternalScientificSink(Protocol):
    @property
    def root_identity(self) -> ObjectIdentity: ...

    def lookup(self, relative_locator: str) -> ArtifactIdentity | None: ...

    def publish(
        self,
        relative_locator: str,
        payload: bytes,
        *,
        artifact_id: str,
        role: str,
        payload_schema: str,
        media_type: str,
    ) -> ArtifactIdentity: ...


class FairMastSourceAdapter:
    """No-network adapter core; transport, authority and persistence are injected."""

    def __init__(
        self,
        *,
        gateway: FairMastGateway,
        authority: FairMastAuthorityVerifier,
        sink: ExternalScientificSink,
    ) -> None:
        self._gateway = gateway
        self._authority = authority
        self._sink = sink

    def preview(
        self,
        *,
        operator: FairMastOperatorConfig,
        declaration: FairMastSourceDeclaration,
        authorization: ObjectIdentity,
        budget: ResourceBudget,
    ) -> FairMastPreview:
        if operator.request_timeout_seconds > budget.wall_time_seconds:
            raise ValueError("operator timeout exceeds the authorized wall-time budget")
        self._authority.require(
            authorization,
            AuthorityAction.PUBLIC_SOURCE_ACQUISITION,
            declaration.source_id,
            budget,
        )
        metadata = self._gateway.inspect_metadata(
            operator,
            declaration,
            maximum_bytes=budget.source_scan_bytes,
        )
        reasons = _metadata_reasons(declaration, metadata)
        return FairMastPreview(
            preview_id=f"preview.{metadata.snapshot_id}",
            source=ObjectIdentity.from_record(declaration.source_id, declaration),
            metadata=metadata,
            requested_budget=budget,
            conforms=not reasons,
            reason_codes=tuple(sorted(reasons)),
        )

    def materialize(
        self,
        *,
        operator: FairMastOperatorConfig,
        declaration: FairMastSourceDeclaration,
        request: FairMastMaterializationRequest,
    ) -> FairMastMaterializationReceipt:
        self._authority.require(
            request.source_authorization,
            AuthorityAction.PUBLIC_SOURCE_ACQUISITION,
            declaration.source_id,
            request.budget,
        )
        self._authority.require(
            request.write_authorization,
            AuthorityAction.DATASET_REGISTRATION,
            declaration.source_id,
            request.budget,
        )
        if request.external_root != self._sink.root_identity:
            raise ValueError("external scientific root identity differs")
        if operator.request_timeout_seconds > request.budget.wall_time_seconds:
            raise ValueError("operator timeout exceeds the authorized wall-time budget")
        fetched = self._gateway.fetch_shot_chunks(operator, request)
        expected = tuple(value.chunk_id for value in request.chunks)
        actual = tuple(value.selection.chunk_id for value in fetched)
        if actual != expected:
            raise ValueError("gateway returned an undeclared, missing or reordered chunk")
        total_bytes = sum(len(value.payload) for value in fetched)
        if total_bytes > request.budget.source_scan_bytes:
            raise ValueError("FAIR-MAST acquisition exceeds the authorized byte budget")

        index_payload = canonical_json_bytes(
            {
                "campaign": request.campaign.value,
                "chunks": tuple(
                    {
                        "chunk_id": value.selection.chunk_id,
                        "sha256": value.source_sha256,
                        "signal_id": value.selection.signal_id,
                    }
                    for value in fetched
                ),
                "shot_id": request.shot_id,
                "source_id": declaration.source_id,
            }
        )
        if total_bytes + len(index_payload) > request.budget.output_bytes:
            raise ValueError("FAIR-MAST materialization exceeds the authorized output budget")

        receipts: list[FairMastChunkReceipt] = []
        reused = True
        for item in fetched:
            locator = (
                f"{request.destination_prefix}/shot-{request.shot_id}/raw/"
                f"{item.selection.chunk_id}.bin"
            )
            artifact_id = f"raw.{request.shot_id}.{item.selection.chunk_id}"
            artifact, was_reused = _publish_idempotently(
                self._sink,
                locator,
                item.payload,
                artifact_id=artifact_id,
                role="raw-source-chunk",
                payload_schema=item.payload_schema,
                media_type=item.media_type,
            )
            reused = reused and was_reused
            receipts.append(
                FairMastChunkReceipt(
                    chunk_id=item.selection.chunk_id,
                    shot_id=request.shot_id,
                    signal_id=item.selection.signal_id,
                    chunk_index=item.selection.chunk_index,
                    source_sha256=item.source_sha256,
                    raw_artifact=artifact,
                    raw_locator=locator,
                )
            )

        canonical_locator = (
            f"{request.destination_prefix}/shot-{request.shot_id}/canonical/index.json"
        )
        canonical_artifact, canonical_reused = _publish_idempotently(
            self._sink,
            canonical_locator,
            index_payload,
            artifact_id=f"canonical.{request.shot_id}.index",
            role="canonical-shot-index",
            payload_schema='empirical-lawhood/physical/mast-archive/canonical-index',
            media_type="application/json",
        )
        reused = reused and canonical_reused
        return FairMastMaterializationReceipt(
            receipt_id=f"receipt.{request.request_id}",
            source=ObjectIdentity.from_record(declaration.source_id, declaration),
            shot_id=request.shot_id,
            campaign=request.campaign,
            chunk_receipts=tuple(receipts),
            canonical_artifact=canonical_artifact,
            canonical_locator=canonical_locator,
            external_root=request.external_root,
            state=(
                MastMaterializationState.IDEMPOTENT_REUSE
                if reused
                else MastMaterializationState.MATERIALIZED
            ),
            total_source_bytes=total_bytes,
        )


def _metadata_reasons(
    declaration: FairMastSourceDeclaration,
    metadata: FairMastMetadataSnapshot,
) -> set[str]:
    reasons: set[str] = set()
    if metadata.shots_metadata_sha256 != FAIR_MAST_SHOTS_METADATA_SHA256:
        reasons.add("SHOTS_METADATA_IDENTITY_MISMATCH")
    if metadata.sources_metadata_sha256 != FAIR_MAST_SOURCES_METADATA_SHA256:
        reasons.add("SOURCES_METADATA_IDENTITY_MISMATCH")
    if metadata.license_id != FAIR_MAST_LICENSE_ID:
        reasons.add("LICENSE_MISMATCH")
    if metadata.campaigns != declaration.campaigns:
        reasons.add("CAMPAIGN_SET_MISMATCH")
    if not set(declaration.expected_groups).issubset(metadata.groups):
        reasons.add("EXPECTED_GROUP_MISSING")
    declared_signals = {value.signal_id: value for value in metadata.signals}
    for signal in metadata.signals:
        if signal.expected_group not in metadata.groups:
            reasons.add("SIGNAL_GROUP_MISSING")
    if set(declared_signals) != {"electron-density", "electron-temperature", "nbi-power"}:
        reasons.add("SIGNAL_GAUGE_MISMATCH")
    return reasons


def _publish_idempotently(
    sink: ExternalScientificSink,
    locator: str,
    payload: bytes,
    *,
    artifact_id: str,
    role: str,
    payload_schema: str,
    media_type: str,
) -> tuple[ArtifactIdentity, bool]:
    validate_relative_locator(locator)
    expected_sha256 = sha256(payload).hexdigest()
    existing = sink.lookup(locator)
    if existing is not None:
        expected = (artifact_id, role, payload_schema, expected_sha256, media_type, len(payload))
        actual = (
            existing.artifact_id,
            existing.role,
            existing.payload_schema,
            existing.sha256,
            existing.media_type,
            existing.size_bytes,
        )
        if actual != expected:
            raise ValueError("external scientific artifact drift detected")
        return existing, True
    result = sink.publish(
        locator,
        payload,
        artifact_id=artifact_id,
        role=role,
        payload_schema=payload_schema,
        media_type=media_type,
    )
    if result.sha256 != expected_sha256 or result.size_bytes != len(payload):
        raise ValueError("external sink publication receipt differs from written bytes")
    return result, False


__all__ = [
    "ExternalScientificSink",
    "FairMastAuthorityVerifier",
    "FairMastChunkSelection",
    "FairMastFetchedChunk",
    "FairMastGateway",
    "FairMastMaterializationRequest",
    "FairMastMetadataSnapshot",
    "FairMastOperatorConfig",
    "FairMastPreview",
    "FairMastSourceAdapter",
    "decode_fair_mast_operator_config",
]
