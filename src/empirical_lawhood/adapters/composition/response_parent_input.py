"""Shared prepared response historical-parent witness and pre-outcome authority boundary."""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from .response_parent_custody import ResponseParentTargetAuthorityStore, ResponseParentTargetGrant

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class ImportedResponseParentCustody(CanonicalRecord):
    """Declared source inventory; validity alone never grants target custody."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/imported-response-parent-custody'
    source_schema: str
    source_manifest_sha256: str
    artifact_sha256s: tuple[str, ...]
    original_root_ids: tuple[str, ...]
    receipt_sha256s: tuple[str, ...]
    evidence_role: str

    def __post_init__(self) -> None:
        segments = self.source_schema.split("/")
        if len(self.source_schema) > 256 or not 3 <= len(segments) <= 8:
            raise ValueError("prepared response parent source schema is not bounded")
        for segment in segments:
            validate_stable_id(segment, field_name="source_schema segment")
        validate_sha256(
            self.source_manifest_sha256, field_name="source_manifest_sha256"
        )
        for name in ("artifact_sha256s", "receipt_sha256s"):
            values = getattr(self, name)
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            for value in values:
                validate_sha256(value, field_name=name)
        require_sorted_unique_strings(
            self.original_root_ids, field_name="original_root_ids", allow_empty=False
        )
        for value in self.original_root_ids:
            validate_stable_id(value, field_name="original_root_id")
        if self.evidence_role not in (
            "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
            "EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE",
        ):
            raise ValueError("RESPONSE_PARENT_ROLE_INVALID")


def require_target_parent_authority(
    *,
    route: str,
    source_root: Path | None,
    parent_manifest: Path | None,
    custody: Path | None,
    reveal_record: Path | None,
    analysis_record: Path | None,
    authority_store: 'ResponseParentTargetAuthorityStore | None' = None,
) -> 'ResponseParentTargetGrant':
    """Refuse source-only paths before reading parent or outcome bytes.

    Resolve all three grants before opening a parent or manifest. Importers
    subsequently authenticate the complete source inventory against this grant.
    """

    if route not in ("dependent-refinement", "fresh-response-calibration", "response-composition", "finite-response-law-supplemental-development", "finite-response-law-informative-composition", "finite-response-law-calibration-method", "finite-response-law-prospective-parent", "finite-response-law-preparation-screening"):
        raise ValueError("RESPONSE_PARENT_ROUTE_INVALID")
    code = {
        "dependent-refinement": "DEPENDENT_REFINEMENT",
        "fresh-response-calibration": "FRESH_RESPONSE_CALIBRATION",
        "response-composition": "RESPONSE_COMPOSITION",
        "finite-response-law-supplemental-development": "FINITE_RESPONSE_LAW_SUPPLEMENTAL_DEVELOPMENT",
        "finite-response-law-informative-composition": "FINITE_RESPONSE_LAW_INFORMATIVE_COMPOSITION",
        "finite-response-law-calibration-method": "FINITE_RESPONSE_LAW_CALIBRATION_METHOD",
        "finite-response-law-prospective-parent": "FINITE_RESPONSE_LAW_PROSPECTIVE_PARENT",
        "finite-response-law-preparation-screening": "FINITE_RESPONSE_LAW_PREPARATION_SCREENING",
    }[route]
    if (
        source_root is None
        or not source_root.is_absolute()
        or any(path.is_symlink() for path in (source_root, *source_root.parents))
        or not source_root.is_dir()
    ):
        raise ValueError(f"{code}_SOURCE_ROOT_REQUIRED")
    root = source_root.resolve(strict=True)
    if parent_manifest is None:
        raise ValueError(f"{code}_PARENT_MANIFEST_REQUIRED")
    if (
        not parent_manifest.is_absolute()
        or any(
            path.is_symlink() for path in (parent_manifest, *parent_manifest.parents)
        )
        or not parent_manifest.is_file()
        or not parent_manifest.resolve(strict=True).is_relative_to(root)
        or not 0 < parent_manifest.stat().st_size <= 128 * 1024**2
    ):
        raise ValueError(f"{code}_PARENT_MANIFEST_INVALID")
    for name, path in (
        ("TARGET_CUSTODY", custody),
        ("TARGET_REVEAL", reveal_record),
        ("TARGET_ANALYSIS", analysis_record),
    ):
        if path is None:
            raise ValueError(f"{code}_{name}_REQUIRED")
        if not path.is_absolute() or path.is_symlink():
            raise ValueError(f"{code}_{name}_LOCATOR_INVALID")
    if authority_store is None:
        raise ValueError(
            f"{code}_TARGET_CUSTODY_STORE_REQUIRED: native_tasks=0 analysis_tasks=0"
        )
    from .response_parent_custody import _resolve_grants

    assert (
        custody is not None
        and reveal_record is not None
        and analysis_record is not None
    )
    return _resolve_grants(
        route=route,
        custody=custody,
        reveal_record=reveal_record,
        analysis_record=analysis_record,
        authority_store=authority_store,
    )


__all__ = ['ImportedResponseParentCustody', "require_target_parent_authority"]
