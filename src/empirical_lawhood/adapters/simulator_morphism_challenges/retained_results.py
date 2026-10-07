"""Receipt-primary current RC result and phase prerequisite custody."""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.runtime.artifacts import ArtifactManifest, ArtifactWriter, CanonicalTaskReceipt


@dataclass(frozen=True, slots=True)
class RCChallengeResultAuthorityContext(CanonicalRecord):
    """References to actual current grants; this record itself grants no access."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/simulator-morphism-challenges/result-authority-context"
    context_id: str
    issued_study: ObjectIdentity
    execution_authority: ObjectIdentity
    reveal_authority: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.context_id)
        from empirical_lawhood.planning.study_issue import StudyOperationAuthority

        if any(value.object_schema != StudyOperationAuthority.SCHEMA for value in (self.execution_authority, self.reveal_authority)):
            raise ValueError("RC protected result requires typed current authority references")


@dataclass(frozen=True, slots=True)
class RCChallengeRetainedResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/simulator-morphism-challenges/retained-result"
    result_id: str
    canonical_payload_text: str
    output_manifest: ArtifactManifest
    task_receipt: CanonicalTaskReceipt
    receipt_manifest: ArtifactManifest
    authority_context: RCChallengeResultAuthorityContext | None = None

    @property
    def requires_outcome_authority(self) -> bool:
        return self.output_manifest.logical.outcome_access in (OutcomeAccess.EVALUATION_SEALED, OutcomeAccess.EVALUATOR_REVEAL, OutcomeAccess.EVALUATION_REVEALED)

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id)
        payload = self.canonical_payload_text.encode("utf-8")
        if not 0 < len(payload) <= 8 * 1024**2:
            raise ValueError("retained RC result exceeds its closed payload bound")
        output = self.output_manifest
        receipt = self.task_receipt
        if output.publication is None or self.receipt_manifest.publication is None:
            raise ValueError("RC result/prerequisite requires actual committed external publications")
        if output.logical.content_sha256 != sha256(payload).hexdigest() or output.materialization.physical_sha256 != sha256(payload).hexdigest() or output.materialization.size_bytes != len(payload):
            raise ValueError("retained RC output bytes differ from their exact manifest")
        if output.materialization not in receipt.output_materializations or output.logical not in receipt.output_logical_artifacts:
            raise ValueError("retained RC result is absent from its exact task receipt output census")
        receipt_bytes = receipt.canonical_bytes()
        if self.receipt_manifest.logical.payload_schema != receipt.SCHEMA or self.receipt_manifest.logical.content_sha256 != receipt.fingerprint() or self.receipt_manifest.materialization.physical_sha256 != receipt.fingerprint() or self.receipt_manifest.materialization.size_bytes != len(receipt_bytes):
            raise ValueError("retained RC task receipt differs from its committed bytes")

    def authenticate(self, writer: ArtifactWriter) -> None:
        writer.verify_manifests((self.output_manifest, self.receipt_manifest))

    def decode(self) -> CanonicalRecord:
        from .runtime_provider import _RECORD_TYPES

        kind = _RECORD_TYPES.get(self.output_manifest.logical.payload_schema)
        if kind is None:
            raise ValueError("retained RC result schema is outside the closed current family")
        return decode_canonical_bytes(self.canonical_payload_text.encode("utf-8"), kind, maximum_bytes=8 * 1024**2)

    def require_prerequisite(self, record: CanonicalRecord) -> None:
        if self.decode() != record or self.task_receipt.operational_status is not OperationalStatus.SUCCEEDED or not self.task_receipt.checks or not all(check.passed for check in self.task_receipt.checks):
            raise ValueError("RC dependent phase requires the exact successful upstream receipt, never a copied result alone")
