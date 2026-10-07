"""Checksum-first acquisition for separately custodied FSM NumPy members."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from empirical_lawhood.kernel.authority import ResourceBudget, SourceAccessClass
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactMaterialization, validate_relative_locator
from empirical_lawhood.runtime.sources import (
    SourceCapabilityManifest,
    SourceMode,
    SourceRequest,
    SourceResult,
    validate_source_result,
)

_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_UPSTREAM = re.compile(r"^data/[uy]_(?:100|200|300)mV_(?:train|test)\.npy$")
_OFFICIAL_HOST = "raw.githubusercontent.com"
_REPOSITORY_PATH = "merijnfloren/fsm-benchmark-data"


class FineSteeringMirrorSourceError(RuntimeError):
    """A pinned source member failed retrieval, custody or integrity checks."""


class FineSteeringMirrorSourceRole(StrEnum):
    DEVELOPMENT_ACTION = "DEVELOPMENT_ACTION"
    DEVELOPMENT_RECEIVER = "DEVELOPMENT_RECEIVER"
    EVALUATION_ACTION = "EVALUATION_ACTION"
    EVALUATION_RECEIVER_SEALED = "EVALUATION_RECEIVER_SEALED"


_ROLE_ACCESS = {
    FineSteeringMirrorSourceRole.DEVELOPMENT_ACTION: OutcomeAccess.DEVELOPMENT_VISIBLE,
    FineSteeringMirrorSourceRole.DEVELOPMENT_RECEIVER: OutcomeAccess.DEVELOPMENT_VISIBLE,
    FineSteeringMirrorSourceRole.EVALUATION_ACTION: OutcomeAccess.OUTCOME_BLIND,
    FineSteeringMirrorSourceRole.EVALUATION_RECEIVER_SEALED: OutcomeAccess.EVALUATION_SEALED,
}


@dataclass(frozen=True, slots=True)
class FineSteeringMirrorSourceMemberSpec(CanonicalRecord):
    """One pinned upstream byte object and its exact custody destination."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/fine-steering-mirror/fine-steering-mirror-source-member-spec'

    source_id: str
    release_commit_sha1: str
    upstream_path: str
    role: FineSteeringMirrorSourceRole
    git_blob_sha1: str
    size_bytes: int
    destination_relative_path: str

    def __post_init__(self) -> None:
        validate_stable_id(self.source_id, field_name="source_id")
        if _COMMIT.fullmatch(self.release_commit_sha1) is None:
            raise ValueError("release commit must be a lowercase Git SHA-1")
        if _UPSTREAM.fullmatch(self.upstream_path) is None:
            raise ValueError("FSM source member must be a safe data/*.npy path")
        if _SHA1.fullmatch(self.git_blob_sha1) is None:
            raise ValueError("source member Git blob identity is malformed")
        if self.size_bytes <= 0 or self.size_bytes >= 4_294_967_295:
            raise ValueError("source member size must be positive")
        validate_relative_locator(self.destination_relative_path)
        expected_leaf = self.upstream_path.rsplit("/", maxsplit=1)[-1].casefold()
        if self.destination_relative_path.rsplit("/", maxsplit=1)[-1] != expected_leaf:
            raise ValueError("custody destination must preserve the case-folded source leaf name")

    @property
    def outcome_access(self) -> OutcomeAccess:
        return _ROLE_ACCESS[self.role]

    @property
    def url(self) -> str:
        return (
            f"https://{_OFFICIAL_HOST}/{_REPOSITORY_PATH}/"
            f"{self.release_commit_sha1}/{self.upstream_path}"
        )


@dataclass(frozen=True, slots=True)
class FetchedBytes:
    payload: bytes
    final_url: str


class SourceFetcher(Protocol):
    def fetch(self, url: str, *, maximum_bytes: int) -> FetchedBytes: ...


class SourceArtifactWriter(Protocol):
    """Persistence port implemented by the composition root, not this adapter."""

    def write_source(
        self,
        *,
        source_id: str,
        destination_relative_path: str,
        payload: bytes,
    ) -> ArtifactMaterialization: ...


class PinnedHttpsFetcher:
    """Retrieve one small pinned object without local temporary-file fallback."""

    def __init__(self, *, timeout_seconds: float = 60.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def fetch(self, url: str, *, maximum_bytes: int) -> FetchedBytes:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname != _OFFICIAL_HOST:
            raise FineSteeringMirrorSourceError("source URL is outside the official HTTPS host")
        request = Request(
            url,
            headers={
                "Accept-Encoding": "identity",
                "User-Agent": "empirical-lawhood-r10 pinned-source-acquisition",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                final_url = response.geturl()
                final = urlparse(final_url)
                if final.scheme != "https" or final.hostname != _OFFICIAL_HOST:
                    raise FineSteeringMirrorSourceError("source redirect escaped the official HTTPS host")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != maximum_bytes:
                    raise FineSteeringMirrorSourceError("source Content-Length differs from the frozen size")
                payload = response.read(maximum_bytes + 1)
        except FineSteeringMirrorSourceError:
            raise
        except Exception as error:
            raise FineSteeringMirrorSourceError(f"source acquisition failed: {error}") from error
        if len(payload) != maximum_bytes:
            raise FineSteeringMirrorSourceError("source payload size differs from the frozen size")
        return FetchedBytes(payload=payload, final_url=final_url)


def _git_blob_sha1(payload: bytes) -> str:
    prefix = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(prefix + payload).hexdigest()  # noqa: S324 - Git identity contract


class FineSteeringMirrorSourceAcquirer:
    """Acquire allowlisted source members and return canonical source results."""

    capability_key = "source.fine-steering-mirror"
    capability_version = "1.0.0"
    payload_schema = 'empirical-lawhood/source/fine-steering-mirror-npy'

    def __init__(
        self,
        *,
        members: tuple[FineSteeringMirrorSourceMemberSpec, ...],
        writer: SourceArtifactWriter,
        implementation_sha256: str,
        fetcher: SourceFetcher | None = None,
        storage_root_id: str = "external-primary",
    ) -> None:
        if len({member.source_id for member in members}) != len(members):
            raise ValueError("FSM source member IDs must be unique")
        if len({member.destination_relative_path for member in members}) != len(members):
            raise ValueError("FSM source destinations must be unique")
        validate_stable_id(storage_root_id, field_name="storage_root_id")
        self._members = {member.source_id: member for member in members}
        self._writer = writer
        self._fetcher = fetcher or PinnedHttpsFetcher()
        self._storage_root_id = storage_root_id
        self.manifest = SourceCapabilityManifest(
            capability_key=self.capability_key,
            capability_version=self.capability_version,
            mode=SourceMode.ARCHIVAL,
            source_access=SourceAccessClass.OFFICIAL_OPEN_PUBLIC,
            output_schema_ids=(self.payload_schema,),
            maximum_evidence=EvidenceCeiling.MEASUREMENT,
            maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            deterministic=True,
            implementation_sha256=implementation_sha256,
            resource_budget=ResourceBudget(
                cpu_cores=1,
                memory_bytes=16_000_000,
                gpu_devices=0,
                wall_time_seconds=120,
                source_scan_bytes=3_000_000,
                output_bytes=3_000_000,
            ),
        )

    def acquire(self, request: SourceRequest) -> SourceResult:
        try:
            member = self._members[request.source_id]
        except KeyError as error:
            raise FineSteeringMirrorSourceError(
                f"source member is not allowlisted: {request.source_id}"
            ) from error
        if request.licence_id != "CC-BY-4.0":
            raise FineSteeringMirrorSourceError("source request does not bind the frozen CC BY 4.0 license")
        if request.outcome_access is not member.outcome_access:
            raise FineSteeringMirrorSourceError("source request outcome access differs from its custody role")
        if request.expected_size_bytes != member.size_bytes:
            raise FineSteeringMirrorSourceError("source request size differs from the frozen member")
        fetched = self._fetcher.fetch(member.url, maximum_bytes=member.size_bytes)
        if fetched.final_url != member.url:
            raise FineSteeringMirrorSourceError("source final URL differs from the pinned immutable URL")
        if _git_blob_sha1(fetched.payload) != member.git_blob_sha1:
            raise FineSteeringMirrorSourceError("source payload differs from the frozen Git blob")
        materialization = self._writer.write_source(
            source_id=member.source_id,
            destination_relative_path=member.destination_relative_path,
            payload=fetched.payload,
        )
        if (
            materialization.storage_root_id != self._storage_root_id
            or materialization.relative_path != member.destination_relative_path
            or materialization.physical_sha256 != hashlib.sha256(fetched.payload).hexdigest()
            or materialization.size_bytes != len(fetched.payload)
        ):
            raise FineSteeringMirrorSourceError("source writer returned a mismatched materialization")
        result = SourceResult(
            request_id=request.request_id,
            source_id=request.source_id,
            payload_schema=self.payload_schema,
            materialization=materialization,
            observed_sha256=materialization.physical_sha256,
            observed_size_bytes=materialization.size_bytes,
            custody_status="verified",
        )
        validate_source_result(self.manifest, request, result)
        return result
