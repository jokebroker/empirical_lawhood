"Additive provenance for reusable measurement through controller use surgical extraction.\n\nThis manifest is independent of the frozen tokamak control replication extraction manifest.  It binds\nreviewed donor bytes and current worktree bytes, but deliberately cannot stand\nin for a clean-commit ``ImplementationSourceClosure`` at future issue time.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import ClassVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)


MAX_REUSABLE_EXTRACTION_MANIFEST_BYTES = 8 * 1024 * 1024


class ReusableExtractionKind(StrEnum):
    EXACT_DONOR = "EXACT_DONOR"
    REVIEWED_REWRITE = "REVIEWED_REWRITE"
    NEW_INTEGRATION = "NEW_INTEGRATION"


class ReusableExtractionLayer(StrEnum):
    PLANNING = "PLANNING"
    RUNTIME = "RUNTIME"
    ADAPTER = "ADAPTER"
    TEST = "TEST"


@dataclass(frozen=True, slots=True)
class ReusableExtractionFileDigest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reusable-extraction-file-digest'

    file_id: str
    relative_path: str
    sha256: str
    size_bytes: int
    role: str

    def __post_init__(self) -> None:
        validate_stable_id(self.file_id, field_name="file_id")
        validate_relative_locator(self.relative_path)
        validate_sha256(self.sha256, field_name="sha256")
        if self.size_bytes < 1:
            raise ValueError("source-closure file must be nonempty")
        validate_stable_id(self.role, field_name="role")


@dataclass(frozen=True, slots=True)
class ReusableResponseWorktreeSourceClosure(CanonicalRecord):
    """Exact current bytes pending a future clean-commit source closure."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reusable-response-worktree-source-closure'

    closure_id: str
    baseline_commit: str
    files: tuple[ReusableExtractionFileDigest, ...]
    clean_commit_closure_available: bool
    issueable_as_implementation_source_closure: bool
    grants_issue_or_execution_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.closure_id, field_name="closure_id")
        if re.fullmatch(r"[0-9a-f]{40}", self.baseline_commit) is None:
            raise ValueError("worktree closure baseline must be a full Git SHA-1")
        require_sorted_unique_ids(self.files, attribute="relative_path", field_name="files")
        if not self.files:
            raise ValueError("worktree source closure requires exact files")
        if (
            self.clean_commit_closure_available
            or self.issueable_as_implementation_source_closure
            or self.grants_issue_or_execution_authority
        ):
            raise ValueError("dirty-worktree source closure cannot issue or authorize")


@dataclass(frozen=True, slots=True)
class ReusableExtractionEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reusable-extraction-entry'

    entry_id: str
    donor_commit: str | None
    donor_path: str | None
    donor_sha256: str | None
    target_path: str
    target_sha256: str
    extraction_kind: ReusableExtractionKind
    retained_symbol_ids: tuple[str, ...]
    semantic_delta: str
    rejected_donor_semantics: tuple[str, ...]
    owning_layer: ReusableExtractionLayer
    scientific_owner_id: str
    first_canonical_artifact_schema: str
    current_consumer_ids: tuple[str, ...]
    focused_test_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.entry_id, field_name="entry_id")
        validate_relative_locator(self.target_path)
        validate_sha256(self.target_sha256, field_name="target_sha256")
        require_sorted_unique_strings(
            self.retained_symbol_ids,
            field_name="retained_symbol_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.rejected_donor_semantics,
            field_name="rejected_donor_semantics",
        )
        require_sorted_unique_strings(
            self.current_consumer_ids,
            field_name="current_consumer_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.focused_test_ids,
            field_name="focused_test_ids",
            allow_empty=False,
        )
        validate_nonempty(self.semantic_delta, field_name="semantic_delta")
        validate_stable_id(self.scientific_owner_id, field_name="scientific_owner_id")
        validate_schema(self.first_canonical_artifact_schema)
        donor_values = (self.donor_commit, self.donor_path, self.donor_sha256)
        if self.extraction_kind is ReusableExtractionKind.NEW_INTEGRATION:
            if any(value is not None for value in donor_values):
                raise ValueError("new integration cannot fabricate donor bytes")
        else:
            if any(value is None for value in donor_values):
                raise ValueError("donor extraction requires exact donor provenance")
            assert self.donor_commit is not None
            assert self.donor_path is not None
            assert self.donor_sha256 is not None
            if re.fullmatch(r"[0-9a-f]{40}", self.donor_commit) is None:
                raise ValueError("donor commit must be a full Git SHA-1")
            validate_relative_locator(self.donor_path)
            validate_sha256(self.donor_sha256, field_name="donor_sha256")
        if (
            self.extraction_kind is ReusableExtractionKind.EXACT_DONOR
            and self.donor_sha256 != self.target_sha256
        ):
            raise ValueError("exact-donor extraction bytes differ")


@dataclass(frozen=True, slots=True)
class ReusableResponseExtractionManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reusable-response-extraction-manifest'

    manifest_id: str
    trusted_base_commit: str
    donor_commit: str
    entries: tuple[ReusableExtractionEntry, ...]
    worktree_source_closure: ReusableResponseWorktreeSourceClosure
    donor_branch_imports_prohibited: bool
    historical_semantic_imports_prohibited: bool
    adapters_import_infrastructure_prohibited: bool
    experiment_specific_generic_constants_prohibited: bool
    grants_execution_authority: bool
    grants_scientific_promotion: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        for name in ("trusted_base_commit", "donor_commit"):
            if re.fullmatch(r"[0-9a-f]{40}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be a full Git SHA-1")
        require_sorted_unique_ids(
            self.entries,
            attribute="target_path",
            field_name="entries",
        )
        if not self.entries:
            raise ValueError("reusable extraction manifest requires entries")
        donor_commits = {
            value.donor_commit for value in self.entries if value.donor_commit is not None
        }
        if donor_commits != {self.donor_commit}:
            raise ValueError("reusable extraction entries name another donor commit")
        entry_paths = {value.target_path for value in self.entries}
        closure_paths = {value.relative_path for value in self.worktree_source_closure.files}
        if not entry_paths.issubset(closure_paths):
            raise ValueError("worktree source closure omits an extracted/current source")
        if not (
            self.donor_branch_imports_prohibited
            and self.historical_semantic_imports_prohibited
            and self.adapters_import_infrastructure_prohibited
            and self.experiment_specific_generic_constants_prohibited
        ):
            raise ValueError("reusable extraction manifest weakens an architecture prohibition")
        if self.grants_execution_authority or self.grants_scientific_promotion:
            raise ValueError("reusable extraction manifest cannot authorize or promote science")


def decode_reusable_source_qualification_prospective_extraction_manifest(
    payload: bytes,
) -> ReusableResponseExtractionManifest:
    return decode_canonical_bytes(
        payload,
        ReusableResponseExtractionManifest,
        maximum_bytes=MAX_REUSABLE_EXTRACTION_MANIFEST_BYTES,
    )


__all__ = [
    "MAX_REUSABLE_EXTRACTION_MANIFEST_BYTES",
    'ReusableExtractionEntry',
    'ReusableExtractionFileDigest',
    'ReusableExtractionKind',
    'ReusableExtractionLayer',
    'ReusableResponseExtractionManifest',
    'ReusableResponseWorktreeSourceClosure',
    'decode_reusable_source_qualification_prospective_extraction_manifest',
]
