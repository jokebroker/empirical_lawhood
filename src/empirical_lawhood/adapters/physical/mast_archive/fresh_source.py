"""Bounded, fresh FAIR-MAST REST and Zarr source custody.

SPDX-License-Identifier: MPL-2.0

This route selects one public shot and one Zarr array chunk. It does not reuse
the fixed metadata digests or issued identities of earlier MAST experiments.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import ClassVar, Protocol
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.study_issue import StudyOperationAuthorityStore
from empirical_lawhood.infrastructure.source_origin import (
    require_executing_target_source,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import InformationCutoff, parse_utc_timestamp
from empirical_lawhood.planning.study_authoring import DesignInputRecord, DesignInputRole
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ArtifactWriteRequest

REST_BASE = "https://mastapp.site/json"
ZARR_BASE = "https://s3.echo.stfc.ac.uk/mast/level2/shots"
PUBLISHER_PAGE = "https://www.ukaea.org/service/fair-mast/"
LICENCE = "CC-BY-SA-4.0"
_ARRAY_PATH = re.compile(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*\Z")
_MAX_CONFIG_BYTES = 64 * 1024
_MAX_CONTROL_BYTES = 1024 * 1024
SOURCE_GRANTEE = "operator.fair-mast-source-service"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class FairMastPublicSelection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-public-selection'

    source_id: str
    publisher: str
    release_id: str
    rest_base: str
    zarr_base: str
    publisher_page: str
    licence_id: str
    shot_id: int
    campaign_id: str
    array_path: str
    expected_unit: str
    expected_data_type: str
    zarr_format: int
    chunk_key: str
    maximum_rest_bytes: int
    maximum_zarr_metadata_bytes: int
    maximum_chunk_bytes: int
    maximum_total_bytes: int
    request_timeout_seconds: int
    external_custody_role: str
    relative_root: str

    def __post_init__(self) -> None:
        for value in (self.source_id, self.release_id, self.external_custody_role):
            validate_stable_id(value)
        if (
            self.publisher != "UKAEA FAIR-MAST"
            or self.rest_base != REST_BASE
            or self.zarr_base != ZARR_BASE
            or self.publisher_page != PUBLISHER_PAGE
            or self.licence_id != LICENCE
        ):
            raise ValueError(
                "FAIR-MAST public selection changed its publisher or endpoints"
            )
        if type(self.shot_id) is not int or not 1 <= self.shot_id <= 99999:
            raise ValueError("FAIR-MAST shot selection is outside its finite range")
        if not re.fullmatch(r"M[0-9]+", self.campaign_id):
            raise ValueError("FAIR-MAST campaign selector is invalid")
        if not _ARRAY_PATH.fullmatch(self.array_path):
            raise ValueError(
                "FAIR-MAST array path is not a bounded group/array selector"
            )
        if not self.expected_unit or len(self.expected_unit) > 64:
            raise ValueError("FAIR-MAST expected unit is absent or unbounded")
        if self.expected_data_type not in ("float32", "float64"):
            raise ValueError(
                "FAIR-MAST data type is outside the decoded source contract"
            )
        if self.zarr_format != 3 or self.chunk_key != "c/0":
            raise ValueError("this bounded gateway acquires one Zarr v3 first chunk")
        limits = (
            (self.maximum_rest_bytes, 1024 * 1024),
            (self.maximum_zarr_metadata_bytes, 256 * 1024),
            (self.maximum_chunk_bytes, 8 * 1024 * 1024),
        )
        if any(
            type(value) is not int or not 1 <= value <= ceiling
            for value, ceiling in limits
        ):
            raise ValueError("FAIR-MAST member byte bound is invalid")
        if self.maximum_total_bytes != sum(value for value, _ in limits):
            raise ValueError(
                "FAIR-MAST aggregate bound must equal its declared members"
            )
        if (
            type(self.request_timeout_seconds) is not int
            or not 1 <= self.request_timeout_seconds <= 90
        ):
            raise ValueError("FAIR-MAST request timeout is outside its finite range")
        validate_relative_locator(self.relative_root)
        if not self.relative_root.startswith("sources/fair-mast/"):
            raise ValueError(
                "FAIR-MAST custody lies outside the public source namespace"
            )

    @property
    def urls(self) -> tuple[str, str, str]:
        prefix = f"{self.zarr_base}/{self.shot_id}.zarr/{self.array_path}"
        return (
            f"{self.rest_base}/shots/{self.shot_id}",
            f"{prefix}/zarr.json",
            f"{prefix}/{self.chunk_key}",
        )


@dataclass(frozen=True, slots=True)
class FairMastSourceAttempt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-source-attempt'
    attempt_id: str
    selection: ObjectIdentity
    implementation_commit: str
    acquisition_authority: ObjectIdentity
    custody_authority: ObjectIdentity
    storage_root: ObjectIdentity
    started_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.attempt_id)
        if self.selection.object_schema != FairMastPublicSelection.SCHEMA:
            raise ValueError("source attempt binds another selection kind")
        if not re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit):
            raise ValueError("source attempt has no exact clean target commit")
        if (
            self.acquisition_authority.object_schema
            != StudyOperationAuthority.SCHEMA
            or self.custody_authority.object_schema
            != StudyOperationAuthority.SCHEMA
        ):
            raise ValueError("source attempt lacks separate typed authorities")
        parse_utc_timestamp(self.started_at_utc, field_name="started_at_utc")


@dataclass(frozen=True, slots=True)
class FairMastCustodyMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-custody-member'
    member_id: str
    request_url: str
    relative_locator: str
    media_type: str
    size_bytes: int
    physical_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id)
        validate_relative_locator(self.relative_locator)
        validate_sha256(self.physical_sha256)
        if self.size_bytes <= 0:
            raise ValueError("empty source member cannot be receipted")
        if urlsplit(self.request_url).scheme != "https":
            raise ValueError("source member is not an HTTPS selection")


@dataclass(frozen=True, slots=True)
class FairMastSourceReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive/fair-mast-source-receipt'
    receipt_id: str
    attempt: ObjectIdentity
    selection: ObjectIdentity
    storage_root: ObjectIdentity
    members: tuple[FairMastCustodyMember, ...]
    campaign_id: str
    observed_unit: str
    observed_shape: tuple[int, ...]
    coverage: str
    custodied_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id)
        if self.attempt.object_schema != FairMastSourceAttempt.SCHEMA:
            raise ValueError("FAIR-MAST receipt lacks its durable attempt")
        require_sorted_unique_ids(
            self.members, attribute="member_id", field_name="members"
        )
        if (
            len(self.members) != 3
            or self.coverage != "selected-shot-one-array-first-chunk"
        ):
            raise ValueError("FAIR-MAST receipt claims an unsupported source coverage")
        if (
            not self.observed_unit
            or not self.observed_shape
            or any(x <= 0 for x in self.observed_shape)
        ):
            raise ValueError("FAIR-MAST native metadata is incomplete")
        parse_utc_timestamp(self.custodied_at_utc, field_name="custodied_at_utc")


def load_fair_mast_selection(path: Path) -> FairMastPublicSelection:
    return decode_canonical_bytes(
        read_bounded_bytes(path, maximum_bytes=_MAX_CONFIG_BYTES),
        FairMastPublicSelection,
        maximum_bytes=_MAX_CONFIG_BYTES,
    )


class FairMastPublicGateway(Protocol):
    def fetch(self, url: str, *, maximum_bytes: int, timeout_seconds: int) -> bytes: ...


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        raise PermissionError(
            "FAIR-MAST redirected outside the exact endpoint selection"
        )


class BoundedFairMastHttpGateway:
    """Only the three URLs built from a decoded selection reach this transport."""

    def fetch(self, url: str, *, maximum_bytes: int, timeout_seconds: int) -> bytes:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.port not in (None, 443)
            or parsed.hostname not in ("mastapp.site", "s3.echo.stfc.ac.uk")
        ):
            raise PermissionError("FAIR-MAST request is outside the endpoint allowlist")
        opener = build_opener(ProxyHandler({}), _NoRedirect())
        request = Request(
            url,
            headers={
                "User-Agent": "empirical-lawhood-fair-mast-source/0.1.0",
                "Accept-Encoding": "identity",
            },
        )
        with opener.open(request, timeout=timeout_seconds) as response:
            if response.status != 200 or response.geturl() != url:
                raise ValueError("FAIR-MAST endpoint returned a different selection")
            if (
                response.headers.get("Content-Encoding", "identity").lower()
                != "identity"
            ):
                raise ValueError("FAIR-MAST response used implicit decompression")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > maximum_bytes:
                raise ValueError("FAIR-MAST member exceeds its declared byte bound")
            payload = bytes(response.read(maximum_bytes + 1))
        if not payload or len(payload) > maximum_bytes:
            raise ValueError("FAIR-MAST response is empty or exceeds its bound")
        return payload


def _shot_metadata(payload: bytes, selection: FairMastPublicSelection) -> None:
    document = json.loads(payload)
    if isinstance(document, dict) and isinstance(document.get("items"), list):
        items = document["items"]
        if len(items) != 1:
            raise ValueError("FAIR-MAST REST shot selection is not unique")
        document = items[0]
    if not isinstance(document, dict) or document.get("shot_id") != selection.shot_id:
        raise ValueError("FAIR-MAST REST shot identity drifted")
    if document.get("campaign") != selection.campaign_id:
        raise ValueError("FAIR-MAST REST campaign identity drifted")


def _zarr_metadata(
    payload: bytes, selection: FairMastPublicSelection
) -> tuple[str, tuple[int, ...]]:
    document = json.loads(payload)
    if not isinstance(document, dict) or (
        document.get("zarr_format") != selection.zarr_format
        or document.get("node_type") != "array"
        or document.get("data_type") != selection.expected_data_type
    ):
        raise ValueError("FAIR-MAST Zarr v3 array contract drifted")
    shape = document.get("shape")
    attributes = document.get("attributes")
    grid = document.get("chunk_grid")
    if (
        not isinstance(shape, list)
        or not shape
        or any(type(x) is not int or x <= 0 for x in shape)
        or not isinstance(attributes, dict)
        or not isinstance(grid, dict)
        or grid.get("name") != "regular"
        or not isinstance(grid.get("configuration"), dict)
        or not isinstance(grid["configuration"].get("chunk_shape"), list)
        or len(grid["configuration"]["chunk_shape"]) != len(shape)
        or any(
            type(x) is not int or x <= 0 for x in grid["configuration"]["chunk_shape"]
        )
        or not isinstance(document.get("codecs"), list)
        or not document["codecs"]
        or not isinstance(document["codecs"][-1], dict)
        or document["codecs"][-1].get("name") != "zstd"
    ):
        raise ValueError("FAIR-MAST Zarr shape or chunk grid is incomplete")
    unit = attributes.get("units")
    if unit != selection.expected_unit:
        raise ValueError("FAIR-MAST Zarr unit drifted")
    return unit, tuple(shape)


class FairMastPublicSourceService:
    def __init__(
        self,
        *,
        repo_root: Path | None = None,
        plane: ExternalArtifactPlane | None = None,
        authorities: StudyOperationAuthorityStore | None = None,
        gateway: FairMastPublicGateway | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.plane = plane
        self.authorities = authorities
        self.gateway = gateway or BoundedFairMastHttpGateway()

    def preview(self, selection: FairMastPublicSelection) -> dict[str, object]:
        return {
            "source_id": selection.source_id,
            "selection_sha256": selection.fingerprint(),
            "urls": selection.urls,
            "maximum_total_bytes": selection.maximum_total_bytes,
            "coverage": "selected-shot-one-array-first-chunk",
            "custody_role": selection.external_custody_role,
            "relative_root": selection.relative_root,
            "source_contacted": False,
        }

    def _record_path(self, selection: FairMastPublicSelection, name: str) -> str:
        return f"{selection.relative_root}/{name}.json"

    def _load_record(
        self, relative: str, record_type: type[CanonicalRecord]
    ) -> CanonicalRecord | None:
        assert self.plane is not None
        path = self.plane.root.resolve(relative, for_write=False)
        sidecar = self.plane.root.resolve(f"{relative}.manifest.json", for_write=False)
        if not path.exists() and not sidecar.exists():
            return None
        if not path.is_file() or not sidecar.is_file():
            raise ValueError("FAIR-MAST control record is partially published")
        from empirical_lawhood.infrastructure.task_receipts import (
            decode_artifact_manifest,
        )

        manifest = decode_artifact_manifest(
            read_bounded_bytes(sidecar, maximum_bytes=_MAX_CONTROL_BYTES)
        )
        self.plane.verify_manifest(manifest)
        payload = read_bounded_bytes(path, maximum_bytes=_MAX_CONTROL_BYTES)
        record = decode_canonical_bytes(
            payload, record_type, maximum_bytes=_MAX_CONTROL_BYTES
        )
        if record.fingerprint() != manifest.logical.content_sha256:
            raise ValueError("FAIR-MAST control record differs from its manifest")
        return record

    def _write(
        self,
        selection: FairMastPublicSelection,
        name: str,
        payload: bytes,
        schema: str,
        profile: ArtifactProfile,
        media: str,
        access: OutcomeAccess,
        visibility: VisibilityCeiling,
    ) -> None:
        assert self.plane is not None
        relative = f"{selection.relative_root}/{name}"
        self.plane.write(
            ArtifactWriteRequest(
                logical_artifact_id=f"{selection.source_id}.{name.replace('/', '.')}",
                relative_path=relative,
                payload_schema=schema,
                profile=profile,
                media_type=media,
                publication_scope_id=f"scope.{selection.source_id}",
                publication_scope_relative_root=selection.relative_root,
                payload=payload,
                visibility_ceiling=visibility,
                parent_visibility_ceilings=(),
                outcome_access=access,
                minimum_free_bytes=self.plane.root.contract.minimum_free_bytes,
            )
        )

    def _write_raw(
        self, selection: FairMastPublicSelection, name: str, payload: bytes
    ) -> str:
        """Use the guarded raw-byte path; the generic source media roster excludes Zarr."""

        assert self.plane is not None
        relative = f"{selection.relative_root}/raw/{name}"
        path = self.plane.root.resolve(
            relative,
            for_write=True,
            operation_minimum_free_bytes=max(
                len(payload), self.plane.root.contract.minimum_free_bytes
            ),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.is_symlink() or path.read_bytes() != payload:
                raise ValueError(
                    "FAIR-MAST raw source drifted before custody completion"
                )
            return relative
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o640)
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            if path.exists() and not path.is_symlink():
                path.unlink()
            raise
        return relative

    def _clean_head(self) -> str:
        if self.repo_root is None:
            raise PermissionError(
                "FAIR-MAST acquisition requires a target source checkout"
            )
        require_executing_target_source(self.repo_root)
        head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=self.repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        if status:
            raise PermissionError(
                "FAIR-MAST custody requires a clean target source checkout"
            )
        return head

    def _require_authorities(
        self, selection: FairMastPublicSelection, acquisition_id: str, custody_id: str
    ) -> tuple[StudyOperationAuthority, StudyOperationAuthority]:
        if self.plane is None or self.authorities is None:
            raise PermissionError(
                "FAIR-MAST custody requires an explicit operator storage profile"
            )
        self.plane.root.verify(for_write=True)
        source = self.authorities.load(acquisition_id)
        custody = self.authorities.load(custody_id)
        subject = ObjectIdentity.from_record(selection.source_id, selection)
        at_utc = _now()
        require_study_authority(
            source,
            kind=StudyAuthorityKind.SOURCE_ACQUISITION,
            subject=subject,
            prerequisite_authority=None,
            grantee_id=SOURCE_GRANTEE,
            at_utc=at_utc,
        )
        require_study_authority(
            custody,
            kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
            subject=subject,
            prerequisite_authority=None,
            grantee_id=SOURCE_GRANTEE,
            storage_root_id=self.plane.root.contract.storage_root_id,
            relative_root=selection.relative_root,
            at_utc=at_utc,
        )
        if source.issuer != custody.issuer:
            raise PermissionError(
                "FAIR-MAST source and custody authority issuers differ"
            )
        return source, custody

    def acquire(
        self,
        selection: FairMastPublicSelection,
        *,
        acquisition_authority_id: str,
        custody_authority_id: str,
        recover: bool = False,
    ) -> FairMastSourceReceipt:
        source, custody = self._require_authorities(
            selection, acquisition_authority_id, custody_authority_id
        )
        head = self._clean_head()
        assert self.plane is not None
        receipt_path = self._record_path(selection, "receipt")
        existing = self._load_record(receipt_path, FairMastSourceReceipt)
        if existing is not None:
            assert isinstance(existing, FairMastSourceReceipt)
            self.verify_receipt(selection, existing)
            return existing
        attempt_path = self._record_path(selection, "attempt")
        attempted = self._load_record(attempt_path, FairMastSourceAttempt)
        if attempted is not None and not recover:
            raise FileExistsError(
                "FAIR-MAST attempt already exists; use source recover"
            )
        if attempted is None and recover:
            raise FileNotFoundError("FAIR-MAST recovery has no durable attempt")
        expected = FairMastSourceAttempt(
            attempt_id=f"attempt.{selection.source_id}",
            selection=ObjectIdentity.from_record(selection.source_id, selection),
            implementation_commit=head,
            acquisition_authority=ObjectIdentity.from_record(
                source.authority_id, source
            ),
            custody_authority=ObjectIdentity.from_record(custody.authority_id, custody),
            storage_root=ObjectIdentity.from_record(
                self.plane.root.contract.storage_root_id, self.plane.root.contract
            ),
            started_at_utc=_now() if attempted is None else attempted.started_at_utc,
        )
        if attempted is not None and attempted != expected:
            raise PermissionError("FAIR-MAST recovery changed the frozen attempt")
        if attempted is None:
            self._write(
                selection,
                "attempt.json",
                expected.canonical_bytes(),
                expected.SCHEMA,
                ArtifactProfile.CANONICAL_JSON,
                "application/json",
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            )
        limits = (
            selection.maximum_rest_bytes,
            selection.maximum_zarr_metadata_bytes,
            selection.maximum_chunk_bytes,
        )
        payloads = tuple(
            self.gateway.fetch(
                url,
                maximum_bytes=limit,
                timeout_seconds=selection.request_timeout_seconds,
            )
            for url, limit in zip(selection.urls, limits, strict=True)
        )
        if sum(map(len, payloads)) > selection.maximum_total_bytes:
            raise ValueError("FAIR-MAST aggregate transfer exceeds its bound")
        _shot_metadata(payloads[0], selection)
        unit, shape = _zarr_metadata(payloads[1], selection)
        if not payloads[2].startswith(b"\x28\xb5\x2f\xfd"):
            raise ValueError("FAIR-MAST Zarr first chunk is not a Zstandard frame")
        specs = (
            ("rest-shot.json", "application/json", "metadata-document"),
            ("zarr-array.json", "application/json", "zarr-v3-array-metadata"),
            (
                "zarr-first-chunk.bin",
                "application/octet-stream",
                "zarr-v3-compressed-chunk",
            ),
        )
        members: list[FairMastCustodyMember] = []
        for index, (name, media, kind) in enumerate(specs):
            payload = payloads[index]
            relative = self._write_raw(selection, name, payload)
            members.append(
                FairMastCustodyMember(
                    member_id=f"{selection.source_id}.{kind}",
                    request_url=selection.urls[index],
                    relative_locator=relative,
                    media_type=media,
                    size_bytes=len(payload),
                    physical_sha256=sha256(payload).hexdigest(),
                )
            )
        receipt = FairMastSourceReceipt(
            receipt_id=f"receipt.{selection.source_id}",
            attempt=ObjectIdentity.from_record(expected.attempt_id, expected),
            selection=expected.selection,
            storage_root=expected.storage_root,
            members=tuple(sorted(members, key=lambda x: x.member_id)),
            campaign_id=selection.campaign_id,
            observed_unit=unit,
            observed_shape=shape,
            coverage="selected-shot-one-array-first-chunk",
            custodied_at_utc=_now(),
        )
        self._write(
            selection,
            "receipt.json",
            receipt.canonical_bytes(),
            receipt.SCHEMA,
            ArtifactProfile.CANONICAL_JSON,
            "application/json",
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        )
        return receipt

    def verify_receipt(
        self, selection: FairMastPublicSelection, receipt: FairMastSourceReceipt
    ) -> None:
        if self.plane is None:
            raise PermissionError("receipt verification requires external custody")
        if receipt.selection != ObjectIdentity.from_record(
            selection.source_id, selection
        ):
            raise ValueError("FAIR-MAST receipt binds another source selection")
        attempt = self._load_record(
            self._record_path(selection, "attempt"), FairMastSourceAttempt
        )
        if attempt is None or receipt.attempt != ObjectIdentity.from_record(
            attempt.attempt_id, attempt
        ):
            raise ValueError("FAIR-MAST attempt lineage differs")
        for member in receipt.members:
            if not member.relative_locator.startswith(
                f"{selection.relative_root}/raw/"
            ):
                raise ValueError("FAIR-MAST receipt escapes its custody root")
            path = self.plane.root.resolve(member.relative_locator, for_write=False)
            payload = read_bounded_bytes(
                path, maximum_bytes=selection.maximum_total_bytes
            )
            if (
                len(payload) != member.size_bytes
                or sha256(payload).hexdigest() != member.physical_sha256
            ):
                raise ValueError("FAIR-MAST custodied bytes drifted")
        if tuple(
            member.request_url
            for member in sorted(receipt.members, key=lambda m: m.request_url)
        ) != tuple(sorted(selection.urls)):
            raise ValueError("FAIR-MAST receipt URL roster drifted")

    def design_input(
        self,
        selection: FairMastPublicSelection,
        *,
        input_id: str,
        information_cutoff: InformationCutoff,
        operator_id: str,
        authoring_at_utc: str,
    ) -> DesignInputRecord:
        """Bind verified custody as predeclared design context, never native evidence."""

        authoring_at = parse_utc_timestamp(
            authoring_at_utc, field_name="authoring_at_utc"
        )
        receipt = self._load_record(
            self._record_path(selection, "receipt"), FairMastSourceReceipt
        )
        if receipt is None or not isinstance(receipt, FairMastSourceReceipt):
            raise FileNotFoundError("FAIR-MAST custody receipt is absent")
        self.verify_receipt(selection, receipt)
        if parse_utc_timestamp(receipt.custodied_at_utc) > authoring_at:
            raise ValueError(
                "FAIR-MAST receipt postdates the declared authoring cutoff"
            )
        return DesignInputRecord(
            input_id=input_id,
            object_identity=ObjectIdentity.from_record(receipt.receipt_id, receipt),
            materialization_sha256=receipt.fingerprint(),
            information_cutoff=information_cutoff,
            role=DesignInputRole.MOTIVATION,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            operator_id=operator_id,
        )
