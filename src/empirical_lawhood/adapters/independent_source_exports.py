"""Explicit original-source exports with distinct current target custody."""

from dataclasses import dataclass
import re
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_sha256,
    validate_nonempty,
)


@dataclass(frozen=True, slots=True)
class OriginalObjectReference(CanonicalRecord):
    """Opaque original identity facts, never a current scientific interface."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/adapters/original-object-reference"
    object_id: str
    object_schema: str
    object_version: str
    object_fingerprint: str

    def __post_init__(self) -> None:
        for name in ("object_id", "object_schema", "object_version"):
            validate_nonempty(getattr(self, name), field_name=name)
        validate_sha256(self.object_fingerprint, field_name="object_fingerprint")


@dataclass(frozen=True, slots=True)
class OriginalArtifactReference(CanonicalRecord):
    """Exact archived bytes and raw payload schema, without current authority."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/adapters/original-artifact-reference"
    artifact_id: str
    role: str
    payload_schema: str
    sha256: str
    media_type: str
    size_bytes: int

    def __post_init__(self) -> None:
        for name in ("artifact_id", "role", "payload_schema", "media_type"):
            validate_nonempty(getattr(self, name), field_name=name)
        validate_sha256(self.sha256)
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("original artifact requires its exact nonnegative byte count")


@dataclass(frozen=True, slots=True)
class IndependentSourceExport(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/adapters/independent-source-export"
    original_source: OriginalObjectReference
    original_payload_version: str
    original_artifact: OriginalArtifactReference
    original_relative_path: str
    original_task_receipt: OriginalObjectReference
    original_manifest_sha256: str
    original_publication_commit_sha256: str
    original_source_commit: str
    interpretation_sha256: str
    verified_export: ObjectIdentity
    target_source: ObjectIdentity
    target_artifact: ArtifactIdentity
    target_relative_path: str
    target_task_receipt: ObjectIdentity
    target_manifest_sha256: str
    target_publication_commit_sha256: str
    original_physical_unit_ids: tuple[str, ...]
    target_physical_unit_ids: tuple[str, ...]
    grants_authority: bool = False

    def __post_init__(self) -> None:
        if any(
            type(getattr(self, name)) is not ObjectIdentity
            for name in (
                "verified_export",
                "target_source", "target_task_receipt",
            )
        ) or any(
            type(getattr(self, name)) is not ArtifactIdentity
            for name in ("target_artifact",)
        ) or type(self.original_source) is not OriginalObjectReference or type(self.original_task_receipt) is not OriginalObjectReference or type(self.original_artifact) is not OriginalArtifactReference:
            raise ValueError("source export requires its exact typed original references and current source/artifact/custody identities")
        validate_nonempty(self.original_payload_version, field_name="original_payload_version")
        for name in ("original_physical_unit_ids", "target_physical_unit_ids"):
            require_sorted_unique_strings(
                getattr(self, name), field_name=name, allow_empty=False
            )
        for name in ("original_relative_path", "target_relative_path"):
            validate_relative_locator(getattr(self, name))
        for name in (
            "original_manifest_sha256",
            "original_publication_commit_sha256",
            "interpretation_sha256",
            "target_manifest_sha256",
            "target_publication_commit_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            self.original_payload_version != self.original_source.object_version
            or self.original_source.object_schema != self.original_artifact.payload_schema
            or len(self.original_physical_unit_ids) != len(self.target_physical_unit_ids)
            or re.fullmatch(r"[0-9a-f]{40}", self.original_source_commit) is None
            or self.original_source.object_id == self.target_source.object_id
            or self.original_artifact.artifact_id == self.target_artifact.artifact_id
            or self.original_task_receipt.object_id == self.target_task_receipt.object_id
            or self.verified_export.object_schema
            != "empirical-lawhood/migration/verified-source-export"
            or self.verified_export.object_version != "1.0.0"
            or self.target_source.object_schema != self.target_artifact.payload_schema
            or self.target_source.object_version != "1.0.0"
            or self.target_source.object_fingerprint != self.target_artifact.sha256
            or self.target_task_receipt.object_schema
            != "empirical-lawhood/runtime/canonical-task-receipt"
            or self.target_task_receipt.object_version != "1.0.0"
            or self.grants_authority is not False
        ):
            raise ValueError(
                "retained source requires an independently verified original export and distinct current target custody; qualifications and grants do not transfer"
            )

    def validate_source(
        self, source: ObjectIdentity, physical_unit_ids: tuple[str, ...]
    ) -> None:
        if (
            self.target_source != source
            or self.target_physical_unit_ids != physical_unit_ids
        ):
            raise ValueError(
                "retained export does not bind the exact current source and physical units"
            )

    @property
    def lineage_identities(self) -> tuple[ObjectIdentity, ...]:
        return (
            ObjectIdentity.from_record(
                f"export.{self.target_artifact.artifact_id}", self
            ),
            self.verified_export,
            ObjectIdentity.from_record(
                f"original-source.{self.target_artifact.artifact_id}", self.original_source
            ),
            ObjectIdentity.from_record(
                f"original-artifact.{self.target_artifact.artifact_id}", self.original_artifact
            ),
            ObjectIdentity.from_record(
                f"original-receipt.{self.target_artifact.artifact_id}", self.original_task_receipt
            ),
            self.target_task_receipt,
        )
