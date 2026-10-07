"""Ephemeral payload plane for disk-independent truth-known conformance only."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ExecutableReference
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt


@dataclass(frozen=True, slots=True)
class ConformancePayloadEvent(CanonicalRecord):
    """Synthetic publication/recovery event; never empirical evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/conformance-payload-event'

    event_id: str
    event_kind_id: str
    content_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.event_id, field_name="event_id")
        validate_stable_id(self.event_kind_id, field_name="event_kind_id")
        validate_sha256(self.content_sha256, field_name="content_sha256")


@dataclass(slots=True)
class EphemeralCandidatePayloadPlane:
    """Injected memory-only plane used by ordinary tests and planted worlds."""

    _payloads: dict[str, bytes] = field(default_factory=dict)

    def publish_candidate_payload(
        self,
        *,
        payload: bytes,
        evaluator: ExecutableReference,
        implementation: ObjectIdentity,
        decoder_schema: str,
        decoder_version: str,
        maximum_decode_bytes: int,
    ) -> CandidatePayloadPublicationReceipt:
        digest = hashlib.sha256(payload).hexdigest()
        if digest != evaluator.payload.sha256:
            raise ValueError("conformance payload differs from its evaluator artifact")
        receipt_id = f"payload-publication.{evaluator.payload.artifact_id}"
        existing = self._payloads.get(receipt_id)
        if existing is not None and existing != payload:
            raise ValueError("conformance payload publication identity conflict")
        self._payloads[receipt_id] = payload
        publication = ConformancePayloadEvent(
            event_id=f"event.publish.{evaluator.payload.artifact_id}",
            event_kind_id="synthetic-publication",
            content_sha256=digest,
        )
        recovery = ConformancePayloadEvent(
            event_id=f"event.recovery.{evaluator.payload.artifact_id}",
            event_kind_id="synthetic-recovery",
            content_sha256=digest,
        )
        return CandidatePayloadPublicationReceipt(
            receipt_id=receipt_id,
            artifact=evaluator.payload,
            external_root_contract_id="ephemeral-conformance-plane",
            authoritative_relative_locator=(
                f"conformance/{evaluator.payload.artifact_id}.canonical.json"
            ),
            content_sha256=digest,
            payload_format=evaluator.payload_format,
            decoder_schema=decoder_schema,
            decoder_version=decoder_version,
            candidate_evaluator=evaluator,
            implementation=implementation,
            publication_receipt=ObjectIdentity.from_record(publication.event_id, publication),
            recovery_identity=ObjectIdentity.from_record(recovery.event_id, recovery),
            maximum_decode_bytes=maximum_decode_bytes,
        )

    def read_candidate_payload(self, receipt: CandidatePayloadPublicationReceipt) -> bytes:
        try:
            return self._payloads[receipt.receipt_id]
        except KeyError as error:
            raise FileNotFoundError(receipt.receipt_id) from error
