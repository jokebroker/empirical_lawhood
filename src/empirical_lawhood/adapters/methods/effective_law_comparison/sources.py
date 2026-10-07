"""Frozen source roster and byte-level qualification for the post-hoc tranche."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from empirical_lawhood.adapters.independent_source_exports import IndependentSourceExport
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.runtime.artifacts import ArtifactManifest, CanonicalTaskReceipt
from empirical_lawhood.kernel.serialization import (
    validate_relative_locator,
    validate_semantic_version,
)
from empirical_lawhood.kernel.worlds import WorldKind


@dataclass(frozen=True, slots=True)
class SourceMemberDefinition:
    artifact_id: str
    role_id: str
    locator: str
    source_export: IndependentSourceExport | None = None

    def __post_init__(self) -> None:
        if self.source_export is not None:
            if type(self.source_export) is not IndependentSourceExport:
                raise ValueError("post-hoc source requires its exact typed independent source export")
            if self.artifact_id != self.source_export.target_artifact.artifact_id or self.locator != self.source_export.target_relative_path:
                raise ValueError("post-hoc member differs from the exported current target artifact and path")


@dataclass(frozen=True, slots=True)
class ParentSourceDefinition:
    parent_id: str
    lane_id: str
    study_id: str
    target_id: str
    world_id: str
    world_kind: WorldKind
    independent_unit_id: str
    members: tuple[SourceMemberDefinition, ...]
    role_ids: tuple[str, ...]
    analysis_ids: tuple[str, ...]
    expected_unit_count: int | None = None
    eligibility_reason_codes: tuple[str, ...] = ()


def _member(artifact_id: str, role_id: str, locator: str) -> SourceMemberDefinition:
    return SourceMemberDefinition(artifact_id, role_id, locator)


def _identity_version(payload_version: object) -> tuple[str | None, str | None]:
    """Accept the exact current semantic version without historical conversion."""
    if not isinstance(payload_version, str):
        return None, None
    try:
        validate_semantic_version(payload_version)
    except ValueError:
        return None, None
    return payload_version, "EXACT"


def _parent(
    parent_id: str,
    lane_id: str,
    study_id: str,
    target_id: str,
    members: tuple[SourceMemberDefinition, ...],
    *,
    roles: tuple[str, ...],
    analyses: tuple[str, ...],
    units: int | None = None,
    world_kind: WorldKind = WorldKind.NUMERICAL_SIMULATOR,
    reasons: tuple[str, ...] = (),
) -> ParentSourceDefinition:
    return ParentSourceDefinition(
        parent_id=parent_id,
        lane_id=lane_id,
        study_id=study_id,
        target_id=target_id,
        world_id=f"world.{parent_id}",
        world_kind=world_kind,
        independent_unit_id=f"unit.{parent_id}",
        members=members,
        role_ids=tuple(sorted(roles)),
        analysis_ids=tuple(sorted(analyses)),
        expected_unit_count=units,
        eligibility_reason_codes=tuple(sorted(reasons)),
    )





def _custody_sidecar(path: Path) -> Path | None:
    for candidate in (
        path.with_suffix(path.suffix + ".manifest.json"),
        path.with_suffix(path.suffix + ".receipt.json"),
    ):
        if candidate.is_file():
            return candidate
    return None


def _current_sidecar_payload_identity(
    payload: bytes, export: IndependentSourceExport, *, manifest: bool,
) -> tuple[str, int] | None:
    """Decode current custody and match its logical/physical target semantics."""
    if manifest:
        record = decode_canonical_bytes(payload, ArtifactManifest, maximum_bytes=1024 * 1024)
        logical = record.logical
        materialization = record.materialization
        if record.publication is not None and record.publication.commit_marker_sha256 != export.target_publication_commit_sha256:
            return None
    else:
        record = decode_canonical_bytes(payload, CanonicalTaskReceipt, maximum_bytes=1024 * 1024)
        if record.receipt_id != export.target_task_receipt.object_id:
            return None
        logical_rows = tuple(value for value in record.output_logical_artifacts if value.logical_artifact_id == export.target_artifact.artifact_id)
        materialization_rows = tuple(value for value in record.output_materializations if value.logical_artifact_id == export.target_artifact.artifact_id)
        if len(logical_rows) != 1 or len(materialization_rows) != 1:
            return None
        logical, materialization = logical_rows[0], materialization_rows[0]
    if (
        logical.logical_artifact_id != export.target_artifact.artifact_id
        or logical.payload_schema != export.target_artifact.payload_schema
        or logical.content_sha256 != export.target_artifact.sha256
        or materialization.logical_artifact_id != export.target_artifact.artifact_id
        or materialization.relative_path != export.target_relative_path
        or materialization.physical_sha256 != export.target_artifact.sha256
        or materialization.size_bytes != export.target_artifact.size_bytes
    ):
        return None
    return materialization.physical_sha256, materialization.size_bytes


def qualify_parent_sources(
    external_root: Path, parent_sources: tuple[ParentSourceDefinition, ...]
) -> dict[str, object]:
    """Read only an explicit, caller-supplied frozen source roster."""

    if not parent_sources:
        raise ValueError("post-hoc source roster is empty")
    if len({parent.parent_id for parent in parent_sources}) != len(parent_sources):
        raise ValueError("post-hoc source roster duplicates a parent")
    root = external_root.resolve(strict=True)

    parent_rows: list[dict[str, object]] = []
    total_bytes = 0
    maximum_record_bytes = 0
    for parent in parent_sources:
        member_rows: list[dict[str, object]] = []
        parent_reasons = list(parent.eligibility_reason_codes)
        for member in parent.members:
            export = member.source_export
            if type(export) is not IndependentSourceExport:
                parent_reasons.append("CURRENT_SOURCE_EXPORT_CUSTODY_REQUIRED")
                member_rows.append({
                    "artifact_id": member.artifact_id, "role_id": member.role_id,
                    "relative_locator": member.locator, "status": "UNQUALIFIED",
                    "custody_disposition": "ORIGINAL_HISTORY_REQUIRES_EXPLICIT_VERIFIED_EXTERNAL_EXPORT_IMPORT",
                })
                continue
            validate_relative_locator(member.locator)
            path = root / member.locator
            if path.is_symlink() or not path.resolve(strict=False).is_relative_to(root):
                raise PermissionError("post-hoc source locator escapes external custody")
            if not path.is_file():
                parent_reasons.append("PARENT_ARTIFACT_ABSENT")
                member_rows.append(
                    {
                        "artifact_id": member.artifact_id,
                        "role_id": member.role_id,
                        "relative_locator": member.locator,
                        "status": "ABSENT",
                    }
                )
                continue
            payload = read_bounded_bytes(path, maximum_bytes=16 * 1024 * 1024)
            total_bytes += len(payload)
            maximum_record_bytes = max(maximum_record_bytes, len(payload))
            digest = sha256(payload).hexdigest()
            try:
                document = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError):
                parent_reasons.append("PARENT_SCHEMA_UNSUPPORTED")
                continue
            schema = document.get("schema") if isinstance(document, Mapping) else None
            version = document.get("version") if isinstance(document, Mapping) else None
            value = document.get("value") if isinstance(document, Mapping) else None
            identity_version, version_normalization = _identity_version(version)
            schema_supported = (
                isinstance(schema, str)
                and schema == export.target_artifact.payload_schema
                and (digest, len(payload)) == (export.target_artifact.sha256, export.target_artifact.size_bytes)
                and version == export.target_source.object_version
                and isinstance(version, str)
                and identity_version is not None
                and version_normalization is not None
                and isinstance(value, Mapping)
                and set(document) == {"schema", "value", "version"}
            )
            if not schema_supported:
                parent_reasons.append("CURRENT_SOURCE_SCHEMA_OR_BYTES_MISMATCH")
            sidecar_path = _custody_sidecar(path)
            custody_digest: str | None = None
            custody_status = "ABSENT"
            if sidecar_path is not None:
                if sidecar_path.is_symlink() or not sidecar_path.resolve(
                    strict=False
                ).is_relative_to(root):
                    raise PermissionError("post-hoc custody sidecar escapes external root")
                sidecar_bytes = read_bounded_bytes(
                    sidecar_path, maximum_bytes=1024 * 1024
                )
                custody_digest = sha256(sidecar_bytes).hexdigest()
                expected_custody_digest = (
                    export.target_manifest_sha256
                    if sidecar_path.name.endswith(".manifest.json")
                    else export.target_task_receipt.object_fingerprint
                )
                if custody_digest != expected_custody_digest:
                    parent_reasons.append("CURRENT_SOURCE_RECEIPT_CUSTODY_MISMATCH")
                try:
                    expected = _current_sidecar_payload_identity(
                        sidecar_bytes, export,
                        manifest=sidecar_path.name.endswith(".manifest.json"),
                    )
                except (ValueError, TypeError):
                    parent_reasons.append("CURRENT_SOURCE_RECEIPT_FORMAT_INVALID")
                    custody_status = "INVALID"
                else:
                    if expected is not None and expected != (digest, len(payload)):
                        parent_reasons.append("PARENT_HASH_OR_SIZE_MISMATCH")
                        custody_status = "MISMATCH"
                    else:
                        custody_status = "VERIFIED" if expected is not None else "PRESENT_UNBOUND"
                        if expected is None:
                            parent_reasons.append("PARENT_RECEIPT_UNBOUND")
            else:
                parent_reasons.append("PARENT_RECEIPT_ABSENT")
            member_rows.append(
                {
                    "artifact_id": member.artifact_id,
                    "content_sha256": digest,
                    "custody_sha256": custody_digest,
                    "custody_status": custody_status,
                    "identity_version": identity_version,
                    "payload_schema": schema,
                    "payload_version": version,
                    "relative_locator": member.locator,
                    "role_id": member.role_id,
                    "size_bytes": len(payload),
                    "status": "QUALIFIED" if schema_supported else "UNSUPPORTED",
                    "version_normalization": version_normalization,
                    "source_export_sha256": export.fingerprint(),
                    "source_export": export.to_document(),
                    "grants_authority": False,
                }
            )
        reasons = sorted(set(parent_reasons))
        qualified_members = sum(row.get("status") == "QUALIFIED" for row in member_rows)
        status = (
            "ELIGIBLE"
            if qualified_members == len(parent.members) and not reasons
            else "PARTIALLY_ELIGIBLE"
            if qualified_members
            else "INELIGIBLE"
        )
        parent_rows.append(
            {
                "analysis_ids": list(parent.analysis_ids),
                "expected_unit_count": parent.expected_unit_count,
                "independent_unit_id": parent.independent_unit_id,
                "lane_id": parent.lane_id,
                "members": member_rows,
                "parent_id": parent.parent_id,
                'study_id': parent.study_id,
                "reason_codes": reasons,
                "role_ids": list(parent.role_ids),
                "status": status,
                "target_id": parent.target_id,
                "world_id": parent.world_id,
                "world_kind": parent.world_kind.value,
            }
        )
    return {
        "schema": 'empirical-lawhood/methods/posthoc-response-composition/input-eligibility-assessment',
        "version": "1.0.0",
        "value": {
            "candidate_parent_count": len(parent_sources),
            "maximum_record_bytes": maximum_record_bytes,
            "nested_observations_count_as_units": False,
            "parents": parent_rows,
            "source_scan_bytes": total_bytes,
            "source_selection_after_outcome_access": False,
        },
    }


__all__ = [
    "ParentSourceDefinition",
    "SourceMemberDefinition",
    "qualify_parent_sources",
]
