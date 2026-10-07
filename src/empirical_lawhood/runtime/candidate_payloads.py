"""Runtime-owned candidate-payload custody records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final, Protocol

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_relative_locator,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)


CANDIDATE_EVALUATOR_IMPLEMENTATION_SCHEMA: Final = (
    'empirical-lawhood/methods/candidate-evaluator-implementation'
)


@dataclass(frozen=True, slots=True)
class CandidatePayloadPublicationReceipt(CanonicalRecord):
    """Durable, bounded reconstruction receipt for a candidate evaluator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-payload-publication-receipt'

    receipt_id: str
    artifact: ArtifactIdentity
    external_root_contract_id: str
    authoritative_relative_locator: str
    content_sha256: str
    payload_format: SafePayloadFormat
    decoder_schema: str
    decoder_version: str
    candidate_evaluator: ExecutableReference
    implementation: ObjectIdentity
    publication_receipt: ObjectIdentity
    recovery_identity: ObjectIdentity
    maximum_decode_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.external_root_contract_id,
            field_name="external_root_contract_id",
        )
        validate_relative_locator(self.authoritative_relative_locator)
        validate_sha256(self.content_sha256, field_name="content_sha256")
        validate_schema(self.decoder_schema)
        validate_semantic_version(self.decoder_version)
        if self.maximum_decode_bytes <= 0:
            raise ValueError("candidate payload decode bound must be positive")
        if self.artifact.sha256 != self.content_sha256:
            raise ValueError("candidate payload content digest differs from its artifact")
        if self.candidate_evaluator.payload != self.artifact:
            raise ValueError("candidate evaluator binds another payload")
        if self.candidate_evaluator.payload_format is not self.payload_format:
            raise ValueError("candidate payload format differs from its evaluator")
        if self.implementation.object_schema != CANDIDATE_EVALUATOR_IMPLEMENTATION_SCHEMA:
            raise ValueError("candidate payload requires an exact evaluator implementation")


class CandidatePayloadPlane(Protocol):
    """Inward custody port implemented by infrastructure or injected fakes."""

    def publish_candidate_payload(
        self,
        *,
        payload: bytes,
        evaluator: ExecutableReference,
        implementation: ObjectIdentity,
        decoder_schema: str,
        decoder_version: str,
        maximum_decode_bytes: int,
    ) -> CandidatePayloadPublicationReceipt: ...

    def read_candidate_payload(
        self,
        receipt: CandidatePayloadPublicationReceipt,
    ) -> bytes: ...


__all__ = [
    "CANDIDATE_EVALUATOR_IMPLEMENTATION_SCHEMA",
    "CandidatePayloadPlane",
    "CandidatePayloadPublicationReceipt",
]
