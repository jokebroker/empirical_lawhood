"""Durable external candidate-payload publication for empirical law consumers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath

from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import (
    MAX_ARTIFACT_MANIFEST_BYTES,
    read_bounded_bytes,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ExecutableReference, SafePayloadFormat
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactManifest,
    ArtifactProfile,
    ArtifactWriteRequest,
)
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt


@dataclass(slots=True)
class ExternalCandidatePayloadPlane:
    """Publish and recover canonical candidate models on a guarded artifact plane.

    The scientific method remains injected with the narrow publisher/reader
    protocols.  This class owns only durable byte custody and reconstructable
    publication identity; it does not fit, qualify, or select candidates.
    """

    plane: ExternalArtifactPlane
    relative_root: str
    publication_scope_id: str
    visibility_ceiling: VisibilityCeiling
    outcome_access: OutcomeAccess
    lineage_parents: tuple[ArtifactLineageParent, ...]
    minimum_free_bytes: int = 100 * 1024**3

    def __post_init__(self) -> None:
        if not self.relative_root or PurePosixPath(self.relative_root).is_absolute():
            raise ValueError("candidate payload relative root must be nonempty and relative")
        if (
            tuple(
                sorted(
                    self.lineage_parents,
                    key=lambda value: (
                        value.identity.object_id,
                        value.identity.object_schema,
                        value.identity.object_version,
                        value.identity.object_fingerprint,
                    ),
                )
            )
            != self.lineage_parents
        ):
            raise ValueError("candidate payload lineage parents must be canonically ordered")

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
        if evaluator.payload_format is not SafePayloadFormat.CANONICAL_JSON:
            raise ValueError("external candidate plane accepts canonical JSON models only")
        digest = hashlib.sha256(payload).hexdigest()
        if (
            digest != evaluator.payload.sha256
            or len(payload) != evaluator.payload.size_bytes
            or len(payload) > maximum_decode_bytes
        ):
            raise ValueError("candidate payload differs from its evaluator identity or bound")
        relative_path = f"{self.relative_root}/{evaluator.payload.artifact_id}.canonical.json"
        result = self.plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=evaluator.payload.artifact_id,
                relative_path=relative_path,
                payload_schema=evaluator.payload.payload_schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type=evaluator.payload.media_type,
                publication_scope_id=self.publication_scope_id,
                publication_scope_relative_root=self.relative_root,
                payload=payload,
                visibility_ceiling=self.visibility_ceiling,
                parent_visibility_ceilings=tuple(
                    value.visibility_ceiling for value in self.lineage_parents
                ),
                outcome_access=self.outcome_access,
                logical_content_sha256=digest,
                lineage_parents=self.lineage_parents,
                minimum_free_bytes=self.minimum_free_bytes,
            )
        )
        manifest_path = self.plane.root.resolve(
            f"{relative_path}.manifest.json",
            for_write=False,
        )
        manifest_bytes = read_bounded_bytes(
            manifest_path,
            maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
        )
        manifest = decode_canonical_bytes(
            manifest_bytes,
            ArtifactManifest,
            maximum_bytes=len(manifest_bytes),
        )
        self.plane.verify_manifest(manifest)
        if (
            manifest.logical != result.logical
            or manifest.materialization != result.materialization
            or manifest.publication is None
        ):
            raise ValueError("candidate payload publication manifest is incomplete")
        return CandidatePayloadPublicationReceipt(
            receipt_id=f"payload-publication.{evaluator.payload.artifact_id}",
            artifact=evaluator.payload,
            external_root_contract_id=self.plane.root.contract.storage_root_id,
            authoritative_relative_locator=relative_path,
            content_sha256=digest,
            payload_format=evaluator.payload_format,
            decoder_schema=decoder_schema,
            decoder_version=decoder_version,
            candidate_evaluator=evaluator,
            implementation=implementation,
            publication_receipt=ObjectIdentity.from_record(
                manifest.publication.publication_batch_id,
                manifest.publication,
            ),
            recovery_identity=ObjectIdentity.from_record(
                manifest.materialization.materialization_id,
                manifest.materialization,
            ),
            maximum_decode_bytes=maximum_decode_bytes,
        )

    def read_candidate_payload(self, receipt: CandidatePayloadPublicationReceipt) -> bytes:
        if receipt.external_root_contract_id != self.plane.root.contract.storage_root_id:
            raise FileNotFoundError("candidate payload names another external root")
        expected_prefix = (*PurePosixPath(self.relative_root).parts,)
        observed_parts = PurePosixPath(receipt.authoritative_relative_locator).parts
        if observed_parts[: len(expected_prefix)] != expected_prefix:
            raise FileNotFoundError("candidate payload lies outside the configured namespace")
        path = self.plane.root.resolve(
            receipt.authoritative_relative_locator,
            for_write=False,
        )
        payload = read_bounded_bytes(path, maximum_bytes=receipt.maximum_decode_bytes)
        if (
            len(payload) != receipt.artifact.size_bytes
            or hashlib.sha256(payload).hexdigest() != receipt.content_sha256
        ):
            raise ValueError("candidate payload recovery identity drifted")
        return payload


__all__ = ["ExternalCandidatePayloadPlane"]
