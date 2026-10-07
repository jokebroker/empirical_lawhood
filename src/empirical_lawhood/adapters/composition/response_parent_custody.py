"""Bounded historical prepared response inventory replay after target authority resolution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, Protocol

from empirical_lawhood.adapters._bounded_files import (
    AdapterFileBoundError,
    read_bounded_contained,
    sha256_bounded_contained,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)

from .response_parent_input import ImportedResponseParentCustody

_MAX_MANIFEST_BYTES = 128 * 1024**2
_MAX_MEMBERS = 100_000
_MAX_MEMBER_BYTES = 300 * 1024**2
_MAX_TOTAL_BYTES = 8 * 1024**3
_MAX_RECORD_BYTES = 8 * 1024**2
_ROOT_TOKEN = re.compile(r"^(.+?\.r[0-9]{3})\.")


@dataclass(frozen=True, slots=True)
class ResponseSourceParentIdentity(CanonicalRecord):
    """Exact source-namespace parent bytes, before target record conversion."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/response-source-parent-identity'
    source_schema: str
    source_version: str
    physical_sha256: str

    def __post_init__(self) -> None:
        segments = self.source_schema.split("/")
        if len(self.source_schema) > 256 or not 3 <= len(segments) <= 8:
            raise ValueError("RESPONSE_PARENT_SOURCE_SCHEMA_INVALID")
        for segment in segments:
            validate_stable_id(segment, field_name="source_schema segment")
        validate_semantic_version(self.source_version)
        validate_sha256(self.physical_sha256)


@dataclass(frozen=True, slots=True)
class ResponseParentTargetGrant(CanonicalRecord):
    """A store-resolved target grant; source records cannot stand in for it."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/response-parent-target-grant'
    grant_id: str
    action: str
    route: str
    parent: ResponseSourceParentIdentity
    source_manifest_sha256: str
    source_project_relative: str
    terminal_relative: str
    parent_member_relative: str
    stage_envelope_relative: str
    source_storage_root_id: str
    original_root_ids: tuple[str, ...]
    evidence_role: str

    def __post_init__(self) -> None:
        validate_stable_id(self.grant_id, field_name="grant_id")
        if self.action not in ("CUSTODY", "REVEAL", "ANALYSIS"):
            raise ValueError("RESPONSE_PARENT_GRANT_ACTION_INVALID")
        if self.route not in ("dependent-refinement", "fresh-response-calibration", "response-composition", "finite-response-law-supplemental-development", "finite-response-law-informative-composition", "finite-response-law-calibration-method", "finite-response-law-prospective-parent", "finite-response-law-preparation-screening"):
            raise ValueError("RESPONSE_PARENT_GRANT_ROUTE_INVALID")
        if not isinstance(self.parent, ResponseSourceParentIdentity):
            raise TypeError("RESPONSE_PARENT_GRANT_PARENT_INVALID")
        validate_sha256(self.source_manifest_sha256)
        validate_relative_locator(self.source_project_relative)
        validate_relative_locator(self.terminal_relative)
        validate_relative_locator(self.parent_member_relative)
        validate_relative_locator(self.stage_envelope_relative)
        validate_stable_id(self.source_storage_root_id)
        require_sorted_unique_strings(
            self.original_root_ids, field_name="original_root_ids", allow_empty=False
        )
        for root_id in self.original_root_ids:
            validate_stable_id(root_id, field_name="original_root_id")
        role = (
            "EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE"
            if self.route == "response-composition"
            else "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
        )
        if self.evidence_role != role:
            raise ValueError("RESPONSE_PARENT_GRANT_ROLE_INVALID")


class ResponseParentTargetAuthorityStore(Protocol):
    """Target-owned resolver injected by the installed context, never by JSON."""

    def resolve_grant(self, locator: Path) -> ResponseParentTargetGrant | None: ...


def _read_member(root: Path, path: Path, *, maximum_bytes: int) -> bytes:
    try:
        raw = read_bounded_contained(root, path, maximum_bytes=maximum_bytes)
    except AdapterFileBoundError as error:
        raise ValueError("RESPONSE_PARENT_MEMBER_INVALID") from error
    if not raw:
        raise ValueError("RESPONSE_PARENT_MEMBER_INVALID")
    return raw


def _read_record(path: Path, root: Path, roster: dict[Path, tuple[int, str]]) -> dict:
    if path not in roster:
        raise ValueError("RESPONSE_PARENT_RECORD_MISSING_FROM_ROSTER")
    size, digest = roster[path]
    if size > _MAX_RECORD_BYTES:
        raise ValueError("RESPONSE_PARENT_RECORD_TOO_LARGE")
    raw = _read_member(root, path, maximum_bytes=min(size, _MAX_RECORD_BYTES))
    if len(raw) != size or sha256(raw).hexdigest() != digest:
        raise ValueError("RESPONSE_PARENT_RECORD_DIGEST_MISMATCH")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("RESPONSE_PARENT_RECORD_INVALID_JSON") from error
    value = _unwrap(value)
    if not isinstance(value, dict):
        raise TypeError("RESPONSE_PARENT_RECORD_SHAPE_INVALID")
    return value


def _manifest_publication(
    *,
    path: Path,
    project: Path,
    root: Path,
    roster: dict[Path, tuple[int, str]],
    expected_material: dict,
    expected_storage_root_id: str,
) -> Path:
    manifest_path = Path(str(path) + ".manifest.json")
    manifest = _read_record(manifest_path, root, roster)
    material = manifest.get("materialization")
    logical = manifest.get("logical")
    publication = manifest.get("publication")
    if not all(isinstance(x, dict) for x in (material, logical, publication)):
        raise ValueError("RESPONSE_PARENT_MATERIAL_MANIFEST_INVALID")
    if (
        material != expected_material
        or material.get("storage_root_id") != expected_storage_root_id
        or project / material.get("relative_path", "") != path
        or material.get("physical_sha256") != roster[path][1]
        or material.get("size_bytes") != roster[path][0]
        or logical.get("content_sha256") != roster[path][1]
    ):
        raise ValueError("RESPONSE_PARENT_MATERIAL_BINDING_MISMATCH")
    marker_relative = publication.get("commit_marker_relative_path")
    if not isinstance(marker_relative, str):
        raise TypeError("RESPONSE_PARENT_PUBLICATION_INVALID")
    validate_relative_locator(marker_relative)
    marker = project / marker_relative
    if (
        marker not in roster
        or roster[marker]
        != (
            publication.get("commit_marker_size_bytes"),
            publication.get("commit_marker_sha256"),
        )
        or not isinstance(publication.get("members"), list)
        or not any(
            isinstance(member, dict)
            and member.get("materialization_id") == material.get("materialization_id")
            and member.get("physical_sha256") == material.get("physical_sha256")
            for member in publication["members"]
        )
    ):
        raise ValueError("RESPONSE_PARENT_PUBLICATION_MISMATCH")
    return marker


def _resolve_grants(
    *,
    route: str,
    custody: Path,
    reveal_record: Path,
    analysis_record: Path,
    authority_store: ResponseParentTargetAuthorityStore | None,
) -> ResponseParentTargetGrant:
    if route not in ("dependent-refinement", "fresh-response-calibration", "response-composition", "finite-response-law-supplemental-development", "finite-response-law-informative-composition", "finite-response-law-calibration-method", "finite-response-law-prospective-parent", "finite-response-law-preparation-screening"):
        raise ValueError("RESPONSE_PARENT_ROUTE_INVALID")
    if authority_store is None or not callable(
        getattr(authority_store, "resolve_grant", None)
    ):
        raise ValueError("RESPONSE_PARENT_TARGET_AUTHORITY_STORE_REQUIRED")
    locators = (custody, reveal_record, analysis_record)
    if len(set(locators)) != 3 or any(
        not path.is_absolute()
        or ".." in path.parts
        or any(part.is_symlink() for part in (path, *path.parents))
        for path in locators
    ):
        raise ValueError("RESPONSE_PARENT_TARGET_GRANT_LOCATOR_INVALID")
    grants = tuple(authority_store.resolve_grant(path) for path in locators)
    if any(not isinstance(grant, ResponseParentTargetGrant) for grant in grants):
        raise ValueError("RESPONSE_PARENT_TARGET_GRANT_UNRESOLVED")
    custody_grant, reveal_grant, analysis_grant = grants
    assert (
        custody_grant is not None
        and reveal_grant is not None
        and analysis_grant is not None
    )
    if (
        tuple(grant.action for grant in grants) != ("CUSTODY", "REVEAL", "ANALYSIS")
        or len({grant.grant_id for grant in grants}) != 3
        or any(grant.route != route for grant in grants)
        or any(
            (
                grant.parent,
                grant.source_manifest_sha256,
                grant.source_project_relative,
                grant.terminal_relative,
                grant.parent_member_relative,
                grant.stage_envelope_relative,
                grant.source_storage_root_id,
                grant.original_root_ids,
                grant.evidence_role,
            )
            != (
                custody_grant.parent,
                custody_grant.source_manifest_sha256,
                custody_grant.source_project_relative,
                custody_grant.terminal_relative,
                custody_grant.parent_member_relative,
                custody_grant.stage_envelope_relative,
                custody_grant.source_storage_root_id,
                custody_grant.original_root_ids,
                custody_grant.evidence_role,
            )
            for grant in grants
        )
    ):
        raise ValueError("RESPONSE_PARENT_TARGET_GRANT_SCOPE_MISMATCH")
    return custody_grant


@dataclass(frozen=True)
class ReplayedResponseParent:
    """The exact bytes authenticated by replay, never a path to reopen later."""

    grant: ResponseParentTargetGrant
    custody: ImportedResponseParentCustody
    raw: bytes


def import_response_parent_custody(
    *,
    route: str,
    source_root: Path,
    parent_manifest: Path,
    custody: Path,
    reveal_record: Path,
    analysis_record: Path,
    authority_store: ResponseParentTargetAuthorityStore | None,
    source_schema: str,
) -> ImportedResponseParentCustody:
    """Retain the existing witness interface for custody-only consumers."""

    return replay_response_parent_custody(
        route=route,
        source_root=source_root,
        parent_manifest=parent_manifest,
        custody=custody,
        reveal_record=reveal_record,
        analysis_record=analysis_record,
        authority_store=authority_store,
        source_schema=source_schema,
    ).custody


def replay_response_parent_custody(
    *,
    route: str,
    source_root: Path,
    parent_manifest: Path,
    custody: Path,
    reveal_record: Path,
    analysis_record: Path,
    authority_store: ResponseParentTargetAuthorityStore | None,
    source_schema: str,
) -> ReplayedResponseParent:
    """Replay every referenced artifact, receipt and publication after grants.

    This authenticates historical development input bytes. It does not issue
    a parent, execute a native task, or interpret a scientific outcome.
    """

    grant = _resolve_grants(
        route=route,
        custody=custody,
        reveal_record=reveal_record,
        analysis_record=analysis_record,
        authority_store=authority_store,
    )
    if (
        not source_root.is_absolute()
        or ".." in source_root.parts
        or any(path.is_symlink() for path in (source_root, *source_root.parents))
        or not source_root.is_dir()
    ):
        raise ValueError("RESPONSE_PARENT_SOURCE_ROOT_INVALID")
    # Do not resolve a pathname after checking it: a replacement root must not
    # become a newly trusted outside path. All opens traverse these names using
    # no-follow directory descriptors.
    root = source_root
    manifest_raw = _read_member(
        root, parent_manifest, maximum_bytes=_MAX_MANIFEST_BYTES
    )
    if sha256(manifest_raw).hexdigest() != grant.source_manifest_sha256:
        raise ValueError("RESPONSE_PARENT_SOURCE_MANIFEST_DIGEST_MISMATCH")
    try:
        entries = json.loads(manifest_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("RESPONSE_PARENT_SOURCE_MANIFEST_INVALID_JSON") from error
    if (
        not isinstance(entries, list)
        or not 0 < len(entries) <= _MAX_MEMBERS
        or manifest_raw
        != json.dumps(
            entries, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ):
        raise ValueError("RESPONSE_PARENT_SOURCE_MANIFEST_INVALID")
    project = root / grant.source_project_relative
    terminal = project / grant.terminal_relative
    roster: dict[Path, tuple[int, str]] = {}
    total_bytes = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "bytes"}:
            raise ValueError("RESPONSE_PARENT_SOURCE_ENTRY_INVALID")
        raw_path, digest, size = entry["path"], entry["sha256"], entry["bytes"]
        if (
            not isinstance(raw_path, str)
            or not isinstance(size, int)
            or isinstance(size, bool)
        ):
            raise TypeError("RESPONSE_PARENT_SOURCE_ENTRY_INVALID")
        validate_sha256(digest)
        path = Path(raw_path)
        if str(path) != raw_path or path in roster or not path.is_relative_to(project):
            raise ValueError("RESPONSE_PARENT_SOURCE_ENTRY_PATH_INVALID")
        if not 0 < size <= _MAX_MEMBER_BYTES:
            raise ValueError("RESPONSE_PARENT_SOURCE_ENTRY_SIZE_INVALID")
        total_bytes += size
        if total_bytes > _MAX_TOTAL_BYTES:
            raise ValueError("RESPONSE_PARENT_SOURCE_INVENTORY_TOO_LARGE")
        roster[path] = (size, digest)
    if list(roster) != sorted(roster, key=str) or terminal not in roster:
        raise ValueError("RESPONSE_PARENT_SOURCE_ROSTER_INVALID")
    terminal_record = _read_record(terminal, root, roster)
    run_id = terminal_record.get("run_id")
    if terminal_record.get("operational_status") != "SUCCEEDED" or not isinstance(
        run_id, str
    ):
        raise ValueError("RESPONSE_PARENT_TERMINAL_INVALID")
    attempts = terminal_record.get("attempts")
    if not isinstance(attempts, list):
        raise TypeError("RESPONSE_PARENT_TERMINAL_ATTEMPTS_INVALID")
    run = project / "runs" / run_id
    if not terminal.is_relative_to(run):
        raise ValueError("RESPONSE_PARENT_TERMINAL_RUN_MISMATCH")
    allowed = {terminal}
    outputs: set[str] = set()
    receipts: set[str] = set()
    roots: set[str] = set()
    successful: set[str] = set()
    for raw_attempt in attempts:
        attempt = _unwrap(raw_attempt)
        if not isinstance(attempt, dict) or attempt.get("disposition") != "SUCCEEDED":
            continue
        task_id = attempt.get("task_id")
        receipt_id = attempt.get("receipt_id")
        if (
            not isinstance(task_id, str)
            or not isinstance(receipt_id, str)
            or not receipt_id.startswith("receipt.")
            or task_id in successful
        ):
            raise ValueError("RESPONSE_PARENT_SUCCESS_ROSTER_INVALID")
        validate_stable_id(task_id)
        validate_stable_id(receipt_id)
        successful.add(task_id)
        root_match = _ROOT_TOKEN.match(task_id)
        if root_match is not None:
            roots.add(root_match.group(1))
        receipt_path = (
            run / "receipts" / task_id / (receipt_id.removeprefix("receipt.") + ".json")
        )
        if receipt_path not in roster or roster[receipt_path][1] != attempt.get(
            "receipt_sha256"
        ):
            raise ValueError("RESPONSE_PARENT_RECEIPT_ROSTER_INCOMPLETE")
        receipt = _read_record(receipt_path, root, roster)
        if (
            receipt.get("task_id") != task_id
            or receipt.get("receipt_id") != receipt_id
            or receipt.get("run_id") != run_id
            or receipt.get("operational_status") != "SUCCEEDED"
        ):
            raise ValueError("RESPONSE_PARENT_RECEIPT_BINDING_MISMATCH")
        receipts.add(roster[receipt_path][1])
        allowed.add(receipt_path)
        allowed.add(Path(str(receipt_path) + ".manifest.json"))
        receipt_manifest = _read_record(
            Path(str(receipt_path) + ".manifest.json"), root, roster
        )
        receipt_material = receipt_manifest.get("materialization")
        if not isinstance(receipt_material, dict):
            raise TypeError("RESPONSE_PARENT_RECEIPT_MATERIAL_INVALID")
        allowed.add(
            _manifest_publication(
                path=receipt_path,
                project=project,
                root=root,
                roster=roster,
                expected_material=receipt_material,
                expected_storage_root_id=grant.source_storage_root_id,
            )
        )
        materials = receipt.get("output_materializations")
        if not isinstance(materials, list):
            raise TypeError("RESPONSE_PARENT_OUTPUT_MATERIALS_INVALID")
        for raw_material in materials:
            material = _unwrap(raw_material)
            if not isinstance(material, dict) or not isinstance(
                material.get("relative_path"), str
            ):
                raise TypeError("RESPONSE_PARENT_OUTPUT_MATERIAL_INVALID")
            validate_relative_locator(material["relative_path"])
            output = project / material["relative_path"]
            if output not in roster or not output.is_relative_to(run / "outputs"):
                raise ValueError("RESPONSE_PARENT_OUTPUT_ROSTER_INCOMPLETE")
            outputs.add(roster[output][1])
            allowed.add(output)
            allowed.add(Path(str(output) + ".manifest.json"))
            allowed.add(
                _manifest_publication(
                    path=output,
                    project=project,
                    root=root,
                    roster=roster,
                    expected_material=material,
                    expected_storage_root_id=grant.source_storage_root_id,
                )
            )
    if not successful or tuple(sorted(roots)) != grant.original_root_ids:
        raise ValueError("RESPONSE_PARENT_ORIGINAL_ROOTS_MISMATCH")
    # Prior recovery terminals are allowed only as an exact, bounded companion.
    for path in roster.keys() - allowed:
        if path.name == "resource-terminal.json" and path.is_relative_to(run / "recovery"):
            prior = _read_record(path, root, roster)
            if prior.get("run_id") != run_id:
                raise ValueError("RESPONSE_PARENT_PRIOR_TERMINAL_MISMATCH")
            allowed.add(path)
    if set(roster) != allowed:
        raise ValueError("RESPONSE_PARENT_SOURCE_ROSTER_INCOMPLETE_OR_EXTRA")
    parent_member = project / grant.parent_member_relative
    if parent_member not in allowed or not parent_member.is_relative_to(
        run / "outputs"
    ):
        raise ValueError("RESPONSE_PARENT_MEMBER_NOT_AUTHENTICATED_OUTPUT")
    if roster[parent_member][1] != grant.parent.physical_sha256:
        raise ValueError("RESPONSE_PARENT_IDENTITY_DIGEST_MISMATCH")
    parent_raw = _read_member(
        root, parent_member, maximum_bytes=_MAX_RECORD_BYTES
    )
    if (
        len(parent_raw) != roster[parent_member][0]
        or sha256(parent_raw).hexdigest() != grant.parent.physical_sha256
    ):
        raise ValueError("RESPONSE_PARENT_IDENTITY_BYTES_MISMATCH")
    try:
        parent_document = json.loads(parent_raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("RESPONSE_PARENT_IDENTITY_DOCUMENT_INVALID") from error
    if (
        not isinstance(parent_document, dict)
        or parent_document.get("schema") != grant.parent.source_schema
        or parent_document.get("version") != grant.parent.source_version
        or set(parent_document) != {"schema", "value", "version"}
    ):
        raise ValueError("RESPONSE_PARENT_IDENTITY_SCHEMA_MISMATCH")
    stage_path = project / grant.stage_envelope_relative
    if stage_path not in allowed or not stage_path.is_relative_to(run / "outputs"):
        raise ValueError("RESPONSE_PARENT_STAGE_ENVELOPE_NOT_AUTHENTICATED")
    stage = _read_record(stage_path, root, roster)
    product = stage.get("scientific_product")
    if (
        not isinstance(product, dict)
        or product.get("object_schema") != grant.parent.source_schema
        or product.get("object_version") != grant.parent.source_version
        or product.get("object_fingerprint") != grant.parent.physical_sha256
    ):
        raise ValueError("RESPONSE_PARENT_STAGE_PRODUCT_MISMATCH")
    for path, (size, expected) in roster.items():
        if path == parent_member:
            # The bounded bytes above already match the roster and granted
            # identity. Keep them; do not reopen the mutable source pathname.
            continue
        try:
            observed_size, digest = sha256_bounded_contained(
                root, path, maximum_bytes=min(size, _MAX_MEMBER_BYTES)
            )
        except AdapterFileBoundError as error:
            raise ValueError("RESPONSE_PARENT_MEMBER_INVALID") from error
        if observed_size != size:
            raise ValueError("RESPONSE_PARENT_SOURCE_SIZE_MISMATCH")
        if digest != expected:
            raise ValueError("RESPONSE_PARENT_SOURCE_DIGEST_MISMATCH")
    witness = ImportedResponseParentCustody(
        source_schema,
        grant.source_manifest_sha256,
        tuple(sorted(outputs)),
        grant.original_root_ids,
        tuple(sorted(receipts)),
        grant.evidence_role,
    )
    return ReplayedResponseParent(grant, witness, parent_raw)


def _unwrap(value: object) -> object:
    while isinstance(value, dict) and set(value) == {"schema", "value", "version"}:
        value = value["value"]
    if isinstance(value, dict):
        return {key: _unwrap(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_unwrap(item) for item in value]
    return value


__all__ = [
    'ResponseParentTargetAuthorityStore',
    'ResponseParentTargetGrant',
    'ResponseSourceParentIdentity',
    'import_response_parent_custody',
    'replay_response_parent_custody',
]
