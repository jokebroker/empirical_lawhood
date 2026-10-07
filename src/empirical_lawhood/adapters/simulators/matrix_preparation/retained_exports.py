"""Required custody bindings for independently verified retained-source exports."""

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
)


@dataclass(frozen=True, slots=True)
class PreparationRetainedSourceExport(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/simulators/matrix-preparation/preparation-retained-source-export"
    original_source: ObjectIdentity
    original_artifact: ArtifactIdentity
    original_relative_path: str
    original_task_receipt: ObjectIdentity
    original_manifest_sha256: str
    original_publication_commit_sha256: str
    original_source_commit: str
    interpretation_sha256: str
    verified_export: ObjectIdentity
    target_artifact: ArtifactIdentity
    target_relative_path: str
    target_task_receipt: ObjectIdentity
    target_manifest_sha256: str
    target_publication_commit_sha256: str
    original_physical_unit_ids: tuple[str, ...]
    target_physical_unit_ids: tuple[str, ...]
    grants_authority: bool = False

    def __post_init__(self) -> None:
        require_sorted_unique_strings(self.original_physical_unit_ids, field_name="original_physical_unit_ids", allow_empty=False)
        require_sorted_unique_strings(self.target_physical_unit_ids, field_name="target_physical_unit_ids", allow_empty=False)
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
            len(self.original_physical_unit_ids) != len(self.target_physical_unit_ids)
            or re.fullmatch(r"[0-9a-f]{40}", self.original_source_commit) is None
            or self.original_artifact == self.target_artifact
            or self.original_task_receipt == self.target_task_receipt
            or self.verified_export.object_schema
            != "empirical-lawhood/migration/verified-source-export"
            or self.verified_export.object_version != "1.0.0"
            or self.target_task_receipt.object_schema
            != "empirical-lawhood/runtime/canonical-task-receipt"
            or self.target_task_receipt.object_version != "1.0.0"
            or self.grants_authority is not False
        ):
            raise ValueError(
                "retained source requires independently verified export and separate target custody; qualification and grants do not transfer"
            )

    def validate_target(
        self,
        artifact: ArtifactIdentity,
        relative_path: str,
        task_receipt: ObjectIdentity,
        manifest_sha256: str,
        publication_commit_sha256: str,
        physical_unit_ids: tuple[str, ...],
    ) -> None:
        if (
            self.target_physical_unit_ids != physical_unit_ids
            or self.target_artifact != artifact
            or self.target_relative_path != relative_path
            or self.target_task_receipt != task_receipt
            or self.target_manifest_sha256 != manifest_sha256
            or self.target_publication_commit_sha256 != publication_commit_sha256
        ):
            raise ValueError("retained source export does not bind the exact current input custody")

    @property
    def lineage_identities(self) -> tuple[ObjectIdentity, ...]:
        return (
            ObjectIdentity.from_record(f"export.{self.target_artifact.artifact_id}", self),
            self.verified_export,
            self.original_source,
            ObjectIdentity.from_record(self.original_artifact.artifact_id, self.original_artifact),
            self.original_task_receipt,
        )
