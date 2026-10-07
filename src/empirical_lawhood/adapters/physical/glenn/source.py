"""Bounded, checksum-first inspection of the exact Glenn safe-member selector."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
import stat
import struct
from types import MappingProxyType
from typing import BinaryIO, ClassVar, Mapping
import unicodedata
import zipfile
import zlib

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_nonempty,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)

from .contracts import (
    GlennAdapterError,
    GlennArchiveInput,
    GlennMemberExpectation,
    GlennMemberRole,
    GlennTransformProfile,
)


_EOCD_SIGNATURE = b"PK\x05\x06"
_CENTRAL_SIGNATURE = b"PK\x01\x02"
_EOCD_SIZE = 22
_MAX_ZIP_COMMENT_BYTES = 65_535
_ALLOWED_COMPRESSION_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})


def _require_nonnegative_integer(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class GlennMemberReceipt(CanonicalRecord):
    """Observed identity for one exact, CRC-checked selected member."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/glenn/glenn-member-receipt'

    member_id: str
    role: GlennMemberRole
    relative_locator: str
    physical_sha256: str
    compressed_size_bytes: int
    uncompressed_size_bytes: int
    crc32: str
    media_type: str
    format_profile_id: str
    inspection_policy_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        if not isinstance(self.role, GlennMemberRole):
            raise ValueError("role must be a GlennMemberRole")
        validate_relative_locator(self.relative_locator)
        validate_sha256(self.physical_sha256, field_name="physical_sha256")
        _require_nonnegative_integer(
            self.compressed_size_bytes,
            field_name="compressed_size_bytes",
        )
        _require_nonnegative_integer(
            self.uncompressed_size_bytes,
            field_name="uncompressed_size_bytes",
        )
        if self.uncompressed_size_bytes == 0:
            raise ValueError("selected Glenn members must not be empty")
        if re.fullmatch(r"[0-9a-f]{8}", self.crc32) is None:
            raise ValueError("crc32 must be exactly eight lowercase hexadecimal characters")
        validate_nonempty(self.media_type, field_name="media_type")
        if self.media_type != self.media_type.strip():
            raise ValueError("media_type must not contain surrounding whitespace")
        validate_stable_id(self.format_profile_id, field_name="format_profile_id")
        validate_stable_id(self.inspection_policy_id, field_name="inspection_policy_id")


@dataclass(frozen=True, slots=True)
class GlennArchiveAudit(CanonicalRecord):
    """Bounded archive inventory evidence; no unselected payload was opened."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/glenn/glenn-archive-audit'

    profile_id: str
    locator: str
    guard_evidence_id: str
    archive_size_bytes: int
    archive_sha256: str
    archive_member_count: int
    central_directory_bytes: int
    total_compressed_bytes: int
    total_uncompressed_bytes: int
    inventory_sha256: str
    selected_members: tuple[GlennMemberReceipt, ...]
    unselected_member_count: int
    predecode_and_postdecode_hashes_match: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_relative_locator(self.locator)
        validate_stable_id(self.guard_evidence_id, field_name="guard_evidence_id")
        for field_name, value in (
            ("archive_size_bytes", self.archive_size_bytes),
            ("archive_member_count", self.archive_member_count),
            ("central_directory_bytes", self.central_directory_bytes),
            ("total_compressed_bytes", self.total_compressed_bytes),
            ("total_uncompressed_bytes", self.total_uncompressed_bytes),
            ("unselected_member_count", self.unselected_member_count),
        ):
            _require_nonnegative_integer(value, field_name=field_name)
        if self.archive_size_bytes == 0 or self.archive_member_count == 0:
            raise ValueError("Glenn archive audit requires a nonempty archive and inventory")
        if self.central_directory_bytes == 0:
            raise ValueError("Glenn archive audit requires a nonempty central directory")
        if self.total_compressed_bytes > self.archive_size_bytes:
            raise ValueError("compressed inventory bytes exceed the archive size")
        validate_sha256(self.archive_sha256, field_name="archive_sha256")
        validate_sha256(self.inventory_sha256, field_name="inventory_sha256")
        if not isinstance(self.selected_members, tuple) or not self.selected_members:
            raise ValueError("selected_members must be a nonempty tuple")
        if any(not isinstance(value, GlennMemberReceipt) for value in self.selected_members):
            raise ValueError("selected_members contains another record type")
        require_sorted_unique_ids(
            self.selected_members,
            attribute="member_id",
            field_name="selected_members",
        )
        locators = tuple(value.relative_locator for value in self.selected_members)
        if len(set(locators)) != len(locators):
            raise ValueError("selected_members contains duplicate relative locators")
        roles = tuple(value.role for value in self.selected_members)
        if set(roles) != set(GlennMemberRole) or len(roles) != len(GlennMemberRole):
            raise ValueError("Glenn archive audit must retain exactly one member per safe role")
        if self.archive_member_count != len(self.selected_members) + self.unselected_member_count:
            raise ValueError("selected and unselected member counts do not close the inventory")
        if sum(value.compressed_size_bytes for value in self.selected_members) > (
            self.total_compressed_bytes
        ):
            raise ValueError("selected compressed bytes exceed the archive inventory")
        if sum(value.uncompressed_size_bytes for value in self.selected_members) > (
            self.total_uncompressed_bytes
        ):
            raise ValueError("selected uncompressed bytes exceed the archive inventory")
        if self.predecode_and_postdecode_hashes_match is not True:
            raise ValueError("Glenn archive audit requires matching pre/post decode hashes")


@dataclass(frozen=True, slots=True)
class GlennDecodedSource:
    """The four safe payloads after complete inventory and identity validation."""

    audit: GlennArchiveAudit
    observation_log: bytes
    model_log: bytes
    focal_radius_npy: bytes
    figure_notebook_json: bytes


@dataclass(frozen=True, slots=True)
class _EndRecord:
    entries: int
    central_directory_bytes: int
    central_directory_start: int


@dataclass(frozen=True, slots=True)
class _Inventory:
    infos: Mapping[str, zipfile.ZipInfo]
    inventory_sha256: str
    total_compressed_bytes: int
    total_uncompressed_bytes: int


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    payload = stream.read(size)
    if not isinstance(payload, bytes) or len(payload) != size:
        raise GlennAdapterError("archive stream was truncated or did not return bytes")
    return payload


def _stream_size(stream: BinaryIO) -> int:
    try:
        if not stream.readable() or not stream.seekable():
            raise GlennAdapterError("archive input must be readable and seekable")
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(0)
    except GlennAdapterError:
        raise
    except (AttributeError, OSError, ValueError) as error:
        raise GlennAdapterError("archive input cannot be positioned safely") from error
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise GlennAdapterError("archive input reported an invalid size")
    return size


def _hash_stream(stream: BinaryIO, *, size: int, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    try:
        stream.seek(0)
        remaining = size
        while remaining:
            chunk = stream.read(min(chunk_bytes, remaining))
            if not isinstance(chunk, bytes) or not chunk:
                raise GlennAdapterError("archive changed or truncated while hashing")
            if len(chunk) > remaining:
                raise GlennAdapterError("archive stream exceeded its measured size")
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1) != b"":
            raise GlennAdapterError("archive grew beyond its measured size")
        stream.seek(0)
    except GlennAdapterError:
        raise
    except (OSError, ValueError) as error:
        raise GlennAdapterError("archive cannot be hashed safely") from error
    return digest.hexdigest()


def _read_end_record(
    stream: BinaryIO,
    *,
    archive_size: int,
    maximum_members: int,
    maximum_central_directory_bytes: int,
) -> _EndRecord:
    tail_size = min(archive_size, _EOCD_SIZE + _MAX_ZIP_COMMENT_BYTES)
    try:
        stream.seek(archive_size - tail_size)
        tail = _read_exact(stream, tail_size)
    except (OSError, ValueError) as error:
        raise GlennAdapterError("cannot inspect ZIP end record") from error

    position = tail.rfind(_EOCD_SIGNATURE)
    record: tuple[bytes, int, int, int, int, int, int, int] | None = None
    while position >= 0:
        if position + _EOCD_SIZE <= len(tail):
            candidate = struct.unpack_from("<4s4H2IH", tail, position)
            comment_bytes = candidate[-1]
            if position + _EOCD_SIZE + comment_bytes == len(tail):
                record = candidate
                break
        position = tail.rfind(_EOCD_SIGNATURE, 0, position)
    if record is None:
        raise GlennAdapterError("ZIP end-of-central-directory record is missing or ambiguous")

    _, disk, central_disk, disk_entries, entries, central_bytes, central_offset, _ = record
    if disk != 0 or central_disk != 0 or disk_entries != entries:
        raise GlennAdapterError("multi-disk ZIP archives are forbidden")
    if entries in {0, 0xFFFF} or central_bytes == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
        raise GlennAdapterError("empty or ZIP64 Glenn archives are forbidden")
    if entries > maximum_members:
        raise GlennAdapterError("ZIP member count exceeds its pre-parser ceiling")
    if central_bytes > maximum_central_directory_bytes:
        raise GlennAdapterError("ZIP central directory exceeds its byte ceiling")
    eocd_absolute = archive_size - tail_size + position
    central_start = eocd_absolute - central_bytes
    if central_start < 0 or central_offset > central_start:
        raise GlennAdapterError("ZIP central-directory offsets are inconsistent")
    try:
        stream.seek(central_start)
        if _read_exact(stream, len(_CENTRAL_SIGNATURE)) != _CENTRAL_SIGNATURE:
            raise GlennAdapterError("ZIP central directory does not start at its declared boundary")
        stream.seek(0)
    except (OSError, ValueError) as error:
        raise GlennAdapterError("cannot verify ZIP central-directory boundary") from error
    return _EndRecord(
        entries=entries,
        central_directory_bytes=central_bytes,
        central_directory_start=central_start,
    )


def _safe_member_name(info: zipfile.ZipInfo) -> tuple[str, bool]:
    name = info.filename
    if not name or "\x00" in name or "\\" in name:
        raise GlennAdapterError("ZIP member has an empty, NUL, or backslash name")
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise GlennAdapterError(f"ZIP member uses an absolute path: {name!r}")
    is_directory = info.is_dir()
    raw = name[:-1] if is_directory and name.endswith("/") else name
    parts = raw.split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise GlennAdapterError(f"ZIP member has an unsafe path component: {name!r}")
    normalized = tuple(unicodedata.normalize("NFKC", part).casefold() for part in parts)
    if any(not part for part in normalized):
        raise GlennAdapterError(f"ZIP member normalizes to an empty path: {name!r}")
    return "/".join(normalized), is_directory


def _validate_member_type(info: zipfile.ZipInfo, *, is_directory: bool) -> None:
    if info.flag_bits & 0x1:
        raise GlennAdapterError(f"encrypted ZIP member is forbidden: {info.filename!r}")
    mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode):
        raise GlennAdapterError(f"ZIP symlink member is forbidden: {info.filename!r}")
    allowed = {0, stat.S_IFDIR if is_directory else stat.S_IFREG}
    if file_type not in allowed:
        raise GlennAdapterError(f"ZIP member is not a regular file: {info.filename!r}")
    if is_directory != info.filename.endswith("/"):
        raise GlennAdapterError(f"ZIP directory marker is inconsistent: {info.filename!r}")
    if info.compress_type not in _ALLOWED_COMPRESSION_METHODS:
        raise GlennAdapterError(f"ZIP compression method is not allowlisted: {info.filename!r}")


def _compression_ratio(info: zipfile.ZipInfo) -> float:
    if info.file_size == 0:
        return 1.0
    if info.compress_size <= 0:
        return math.inf
    return float(info.file_size) / float(info.compress_size)


def _canonical_inventory_sha256(rows: list[dict[str, object]]) -> str:
    payload = json.dumps(
        rows,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _audit_inventory(
    archive: zipfile.ZipFile,
    *,
    profile: GlennTransformProfile,
    end_record: _EndRecord,
) -> _Inventory:
    infos = archive.infolist()
    if len(infos) != end_record.entries:
        raise GlennAdapterError("ZIP parser count disagrees with the bounded end record")
    if archive.start_dir != end_record.central_directory_start:
        raise GlennAdapterError("ZIP parser central-directory boundary disagrees with preflight")

    exact: dict[str, zipfile.ZipInfo] = {}
    normalized_names: dict[str, str] = {}
    normalized_regular: set[str] = set()
    inventory_rows: list[dict[str, object]] = []
    total_compressed = 0
    total_uncompressed = 0
    for info in infos:
        if info.filename in exact:
            raise GlennAdapterError(f"ZIP contains a duplicate member name: {info.filename!r}")
        normalized, is_directory = _safe_member_name(info)
        prior = normalized_names.get(normalized)
        if prior is not None:
            raise GlennAdapterError(
                f"ZIP contains a Unicode/case-normalized collision: {prior!r}, {info.filename!r}"
            )
        normalized_names[normalized] = info.filename
        _validate_member_type(info, is_directory=is_directory)
        if info.file_size < 0 or info.compress_size < 0:
            raise GlennAdapterError("ZIP member sizes cannot be negative")
        if info.file_size > profile.limits.maximum_member_uncompressed_bytes:
            raise GlennAdapterError(f"ZIP member exceeds its expansion ceiling: {info.filename!r}")
        if _compression_ratio(info) > profile.limits.maximum_member_compression_ratio:
            raise GlennAdapterError(
                f"ZIP member exceeds its compression-ratio ceiling: {info.filename!r}"
            )
        total_compressed += info.compress_size
        total_uncompressed += info.file_size
        if total_uncompressed > profile.limits.maximum_total_uncompressed_bytes:
            raise GlennAdapterError("ZIP total expansion exceeds its byte ceiling")
        if not is_directory:
            normalized_regular.add(normalized)
        exact[info.filename] = info
        inventory_rows.append(
            {
                "compressed_size_bytes": info.compress_size,
                "compression_method": info.compress_type,
                "crc32": f"{info.CRC:08x}",
                "is_directory": is_directory,
                "normalized_locator": normalized,
                "relative_locator": info.filename,
                "uncompressed_size_bytes": info.file_size,
            }
        )

    for normalized, original in normalized_names.items():
        components = normalized.split("/")
        if any(
            "/".join(components[:depth]) in normalized_regular
            for depth in range(1, len(components))
        ):
            raise GlennAdapterError(f"ZIP regular member is an ancestor of {original!r}")
    total_ratio = (
        1.0
        if total_uncompressed == 0
        else math.inf
        if total_compressed <= 0
        else float(total_uncompressed) / float(total_compressed)
    )
    if total_ratio > profile.limits.maximum_total_compression_ratio:
        raise GlennAdapterError("ZIP total compression ratio exceeds its ceiling")

    expected_names = {member.relative_locator for member in profile.members}
    missing = sorted(expected_names.difference(exact))
    if missing:
        raise GlennAdapterError(f"ZIP lacks exact selected members: {missing}")
    for member in profile.members:
        info = exact[member.relative_locator]
        if info.is_dir():
            raise GlennAdapterError("selected Glenn member cannot be a directory")
        if (
            info.compress_size != member.expected_compressed_size_bytes
            or info.file_size != member.expected_uncompressed_size_bytes
            or f"{info.CRC:08x}" != member.expected_crc32
        ):
            raise GlennAdapterError(
                f"selected member central-directory identity changed: {member.relative_locator!r}"
            )
    return _Inventory(
        infos=MappingProxyType(exact),
        inventory_sha256=_canonical_inventory_sha256(inventory_rows),
        total_compressed_bytes=total_compressed,
        total_uncompressed_bytes=total_uncompressed,
    )


def _read_selected_member(
    archive: zipfile.ZipFile,
    *,
    info: zipfile.ZipInfo,
    expectation: GlennMemberExpectation,
    chunk_bytes: int,
) -> tuple[bytes, GlennMemberReceipt]:
    payload = bytearray()
    digest = hashlib.sha256()
    crc = 0
    try:
        with archive.open(info, mode="r") as member_stream:
            remaining = expectation.expected_uncompressed_size_bytes
            while remaining:
                chunk = member_stream.read(min(chunk_bytes, remaining))
                if not chunk:
                    raise GlennAdapterError("selected member truncated during streaming decode")
                if len(chunk) > remaining:
                    raise GlennAdapterError("selected member exceeded its expected size")
                payload.extend(chunk)
                digest.update(chunk)
                crc = zlib.crc32(chunk, crc)
                remaining -= len(chunk)
            if member_stream.read(1) != b"":
                raise GlennAdapterError("selected member grew beyond its expected size")
    except GlennAdapterError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as error:
        raise GlennAdapterError(
            f"cannot safely stream selected member: {expectation.relative_locator!r}"
        ) from error
    observed_sha256 = digest.hexdigest()
    observed_crc32 = f"{crc & 0xFFFFFFFF:08x}"
    if observed_sha256 != expectation.expected_physical_sha256:
        raise GlennAdapterError(
            f"selected member SHA-256 changed: {expectation.relative_locator!r}"
        )
    if observed_crc32 != expectation.expected_crc32:
        raise GlennAdapterError(f"selected member CRC-32 changed: {expectation.relative_locator!r}")
    return bytes(payload), GlennMemberReceipt(
        member_id=expectation.member_id,
        role=expectation.role,
        relative_locator=expectation.relative_locator,
        physical_sha256=observed_sha256,
        compressed_size_bytes=info.compress_size,
        uncompressed_size_bytes=len(payload),
        crc32=observed_crc32,
        media_type=expectation.media_type,
        format_profile_id=expectation.format_profile_id,
        inspection_policy_id=expectation.inspection_policy_id,
    )


def inspect_glenn_archive(
    source: GlennArchiveInput,
    *,
    profile: GlennTransformProfile,
) -> GlennDecodedSource:
    """Hash, inventory, and stream only the exact four-member selector.

    The complete central directory is bounded and validated before Python's ZIP
    parser is constructed.  The archive is hashed again after all four reads so
    a mutable preopened stream cannot silently substitute bytes during decode.
    """

    stream = source.stream
    archive_size = _stream_size(stream)
    if archive_size > profile.limits.maximum_archive_bytes:
        raise GlennAdapterError("archive exceeds its configured byte ceiling")
    if archive_size != profile.expected_archive_size_bytes:
        raise GlennAdapterError("archive size differs from the exact release expectation")
    before_sha256 = _hash_stream(
        stream,
        size=archive_size,
        chunk_bytes=profile.limits.stream_chunk_bytes,
    )
    if before_sha256 != profile.expected_archive_sha256:
        raise GlennAdapterError("archive SHA-256 differs from the exact release expectation")
    end_record = _read_end_record(
        stream,
        archive_size=archive_size,
        maximum_members=profile.limits.maximum_members,
        maximum_central_directory_bytes=profile.limits.maximum_central_directory_bytes,
    )

    selected_payloads: dict[GlennMemberRole, bytes] = {}
    receipts: list[GlennMemberReceipt] = []
    try:
        stream.seek(0)
        with zipfile.ZipFile(stream, mode="r") as archive:
            inventory = _audit_inventory(archive, profile=profile, end_record=end_record)
            for expectation in profile.members:
                payload, receipt = _read_selected_member(
                    archive,
                    info=inventory.infos[expectation.relative_locator],
                    expectation=expectation,
                    chunk_bytes=profile.limits.stream_chunk_bytes,
                )
                selected_payloads[expectation.role] = payload
                receipts.append(receipt)
    except GlennAdapterError:
        raise
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, NotImplementedError) as error:
        raise GlennAdapterError("cannot parse bounded Glenn ZIP archive") from error

    after_size = _stream_size(stream)
    if after_size != archive_size:
        raise GlennAdapterError("archive size changed during selected-member decode")
    after_sha256 = _hash_stream(
        stream,
        size=archive_size,
        chunk_bytes=profile.limits.stream_chunk_bytes,
    )
    if after_sha256 != before_sha256:
        raise GlennAdapterError("archive bytes changed during selected-member decode")
    audit = GlennArchiveAudit(
        profile_id=profile.profile_id,
        locator=source.locator,
        guard_evidence_id=source.guard_evidence_id,
        archive_size_bytes=archive_size,
        archive_sha256=after_sha256,
        archive_member_count=end_record.entries,
        central_directory_bytes=end_record.central_directory_bytes,
        total_compressed_bytes=inventory.total_compressed_bytes,
        total_uncompressed_bytes=inventory.total_uncompressed_bytes,
        inventory_sha256=inventory.inventory_sha256,
        selected_members=tuple(receipts),
        unselected_member_count=end_record.entries - len(receipts),
    )
    return GlennDecodedSource(
        audit=audit,
        observation_log=selected_payloads[GlennMemberRole.OBSERVATION_LOG],
        model_log=selected_payloads[GlennMemberRole.MODEL_LOG],
        focal_radius_npy=selected_payloads[GlennMemberRole.FOCAL_RADIUS_ARRAY],
        figure_notebook_json=selected_payloads[GlennMemberRole.FIGURE_NOTEBOOK],
    )


__all__ = [
    "GlennArchiveAudit",
    "GlennDecodedSource",
    "GlennMemberReceipt",
    "inspect_glenn_archive",
]
