"""Bounded stdlib-ZIP inventory for exact public MAST-U held objects."""

from __future__ import annotations

from hashlib import sha256
from pathlib import PurePosixPath, PureWindowsPath
import stat
from typing import BinaryIO
import unicodedata
import zipfile

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .contracts import MASTUPublicArchiveInventory, MASTUPublicArchiveLimits, MASTUPublicArchiveMemberInventory, MASTUPublicDataCiteMetadataReceipt, MASTUPublicMemberRole, MASTUPublicReleaseObjectPolicy, MASTUPublicReleaseRole, MASTUPublicSelectedMemberExpectation, MASTUPublicTrustedUseAcceptance


_ALLOWED_COMPRESSION = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})


class MASTUPublicArchiveError(ValueError):
    """An exact held release failed bounded, non-executing inventory."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def primary_release_policies() -> tuple[MASTUPublicReleaseObjectPolicy, ...]:
    """Return exact policies for the acquired primary 26M5 and JMPP objects."""

    trace_limits = MASTUPublicArchiveLimits(
        limits_id="limits.mastu-public.26m5-ey02",
        maximum_archive_bytes=170_000_000,
        maximum_member_count=100,
        maximum_member_uncompressed_bytes=100_000_000,
        maximum_total_uncompressed_bytes=500_000_000,
        maximum_compression_ratio=1_000,
        stream_chunk_bytes=1_048_576,
    )
    configuration_limits = MASTUPublicArchiveLimits(
        limits_id="limits.mastu-public.jmpp-4c57.1.0",
        maximum_archive_bytes=1_000_000,
        maximum_member_count=10,
        maximum_member_uncompressed_bytes=100_000,
        maximum_total_uncompressed_bytes=500_000,
        maximum_compression_ratio=1_000,
        stream_chunk_bytes=65_536,
    )
    trace = MASTUPublicReleaseObjectPolicy(
        release_object_id="release.26m5-ey02",
        release_role=MASTUPublicReleaseRole.DYNAMIC_TRACE_CORPUS,
        doi="10.14468/26M5-EY02",
        version_label="v1",
        object_name="MAST-U_validation_odr.zip",
        held_relative_locator=("sources/public/26m5-ey02/v1/held/MAST-U_validation_odr.zip"),
        datacite_metadata_relative_locator=(
            "sources/public/26m5-ey02/v1/custody/datacite-metadata.xml"
        ),
        datacite_metadata_size_bytes=15_111,
        datacite_metadata_sha256=(
            "2e0a7fae968143a2535609af74764b0c14d9f48387933fbd9678cc0e62af38cc"
        ),
        advertised_sha1="856a383a6cf6dc18097ff503dda44be8723f1e57",
        expected_archive_size_bytes=165_001_751,
        expected_archive_sha256=(
            "31c1ffacde84db094c91d143d027927746f2778bd5a2bb921140de5a07349bee"
        ),
        expected_member_count=83,
        licence_spdx_id="CC-BY-NC-SA-4.0",
        limits=trace_limits,
        selected_members=(
            MASTUPublicSelectedMemberExpectation(
                member_id="member.26m5.provenance-source",
                relative_locator=("MAST-U_validation_odr/freegsnke/MASTU_description_UDA.py"),
                role=MASTUPublicMemberRole.PROVENANCE_SOURCE,
                expected_uncompressed_size_bytes=12_603,
                expected_physical_sha256=(
                    "27a69788c861e4d1ec6f18a36f64c47f066705847f930ac3510b3b0bff4f8802"
                ),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            ),
            MASTUPublicSelectedMemberExpectation(
                member_id="member.26m5.shot-45292",
                relative_locator=("MAST-U_validation_odr/data/MAST-U_shot_45292.pickle"),
                role=MASTUPublicMemberRole.DEVELOPMENT_SHOT_TRACE,
                expected_uncompressed_size_bytes=76_967_478,
                expected_physical_sha256=(
                    "d2bc37eac5340bab0f23dc76e6ab824fec823e921b29b2e6996378ceda9fa5e2"
                ),
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            ),
            MASTUPublicSelectedMemberExpectation(
                member_id="member.26m5.shot-45425",
                relative_locator=("MAST-U_validation_odr/data/MAST-U_shot_45425.pickle"),
                role=MASTUPublicMemberRole.EVALUATION_SHOT_TRACE_SEALED,
                expected_uncompressed_size_bytes=86_888_862,
                expected_physical_sha256=(
                    "8768d670328f78ca8dc8722967c13e22c34bf383019571aa2f5843884a4abec4"
                ),
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
            ),
        ),
    )
    configuration_rows = (
        (
            "member.jmpp.active-coils",
            "MAST-U_active_coils.pickle",
            MASTUPublicMemberRole.ACTIVE_COILS,
            12_767,
            "5c5cc1da422fdf43f04639c85e97936dc4a68b00cfdda5a1078d426970e84c94",
        ),
        (
            "member.jmpp.active-coils-nonsym",
            "MAST-U_active_coils_nonsym.pickle",
            MASTUPublicMemberRole.NONSYMMETRIC_ACTIVE_COILS,
            12_971,
            "bc38cf8641e36e8bbba9a7349c0017a6042357aa586b8a7fcbede90846c88d40",
        ),
        (
            "member.jmpp.limiter",
            "MAST-U_limiter.pickle",
            MASTUPublicMemberRole.LIMITER,
            3_829,
            "e1903bf8b6931e31909b31c9a6ac8a69921b2b28c526cd28c45303a914e2af7f",
        ),
        (
            "member.jmpp.magnetic-probes",
            "MAST-U_magnetic_probes.pickle",
            MASTUPublicMemberRole.MAGNETIC_PROBES,
            58_775,
            "d7c1814d67ef94e7b194aa4bb50b43e1a572ee41208396b170b4aac7c72e9f76",
        ),
        (
            "member.jmpp.passive-coils",
            "MAST-U_passive_coils.pickle",
            MASTUPublicMemberRole.PASSIVE_COILS,
            34_357,
            "76ae3e588318e60fab2a8b42ef4337e92786507541667992495156bf0c28b85b",
        ),
        (
            "member.jmpp.wall",
            "MAST-U_wall.pickle",
            MASTUPublicMemberRole.WALL,
            3_829,
            "e1903bf8b6931e31909b31c9a6ac8a69921b2b28c526cd28c45303a914e2af7f",
        ),
    )
    configuration = MASTUPublicReleaseObjectPolicy(
        release_object_id="release.jmpp-4c57.1.0",
        release_role=MASTUPublicReleaseRole.MACHINE_CONFIGURATION_COMPARATOR,
        doi="10.14468/JMPP-4C57",
        version_label="1.0",
        object_name="MAST-U_freegs_data.zip",
        held_relative_locator=("sources/public/jmpp-4c57/1.0/held/MAST-U_freegs_data.zip"),
        datacite_metadata_relative_locator=(
            "sources/public/jmpp-4c57/1.0/custody/datacite-metadata.xml"
        ),
        datacite_metadata_size_bytes=3_099,
        datacite_metadata_sha256=(
            "b2de5d6c93bf64a3692c31f7d592f9d20eb1d23eaff70724fa6e158d8aada5f8"
        ),
        advertised_sha1="c50e7e9c5199737d53778bbf78726ab834d18b93",
        expected_archive_size_bytes=24_055,
        expected_archive_sha256=(
            "28294edad722037a3597e665ad4d1ae419a14b9422a2262689ddba7ac350710a"
        ),
        expected_member_count=6,
        licence_spdx_id="CC-BY-NC-SA-4.0",
        limits=configuration_limits,
        selected_members=tuple(
            MASTUPublicSelectedMemberExpectation(
                member_id=member_id,
                relative_locator=relative_locator,
                role=role,
                expected_uncompressed_size_bytes=size_bytes,
                expected_physical_sha256=physical_sha256,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
            for member_id, relative_locator, role, size_bytes, physical_sha256 in (
                configuration_rows
            )
        ),
    )
    return (trace, configuration)


def primary_trusted_use_acceptance() -> MASTUPublicTrustedUseAcceptance:
    """Bind the 2026-08-28 owner decision without granting publication/BQTS use."""

    policies = primary_release_policies()
    return MASTUPublicTrustedUseAcceptance(
        acceptance_id="acceptance.mastu-public.primary-trusted-local.2026-08-28",
        experiment_id='mastu-public-release-voltage-law-evidence',
        decision_date="2026-08-28",
        accepted_release_policies=tuple(
            ObjectIdentity.from_record(value.release_object_id, value) for value in policies
        ),
        licence_spdx_id="CC-BY-NC-SA-4.0",
        personal_project=True,
        noncommercial_use=True,
        attribution_sharealike_obligations_accepted=True,
        trusted_local_pickle_code_execution_risk_accepted=True,
        pinned_project_uv_environment_required=True,
        publication_authorized=False,
        upload_authorized=False,
        bqts_execution_authorized=False,
    )


def _stream_size(stream: BinaryIO) -> int:
    try:
        if not stream.readable() or not stream.seekable():
            raise MASTUPublicArchiveError(
                "ARCHIVE_STREAM_NOT_SEEKABLE",
                "archive stream must be readable and seekable",
            )
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(0)
    except MASTUPublicArchiveError:
        raise
    except (AttributeError, OSError, ValueError) as error:
        raise MASTUPublicArchiveError(
            "ARCHIVE_STREAM_POSITION_FAILURE",
            "archive stream cannot be positioned",
        ) from error
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise MASTUPublicArchiveError(
            "ARCHIVE_SIZE_INVALID",
            "archive stream reported an invalid size",
        )
    return size


def _hash_stream(stream: BinaryIO, *, size: int, chunk_bytes: int) -> str:
    digest = sha256()
    try:
        stream.seek(0)
        remaining = size
        while remaining:
            chunk = stream.read(min(remaining, chunk_bytes))
            if not isinstance(chunk, bytes) or not chunk:
                raise MASTUPublicArchiveError(
                    "ARCHIVE_TRUNCATED",
                    "archive changed or truncated while hashing",
                )
            if len(chunk) > remaining:
                raise MASTUPublicArchiveError(
                    "ARCHIVE_SIZE_CHANGED",
                    "archive exceeded its measured size",
                )
            digest.update(chunk)
            remaining -= len(chunk)
        if stream.read(1) != b"":
            raise MASTUPublicArchiveError(
                "ARCHIVE_SIZE_CHANGED",
                "archive grew beyond its measured size",
            )
        stream.seek(0)
    except MASTUPublicArchiveError:
        raise
    except (OSError, ValueError) as error:
        raise MASTUPublicArchiveError(
            "ARCHIVE_READ_FAILURE",
            "archive cannot be hashed",
        ) from error
    return digest.hexdigest()


def _safe_member(info: zipfile.ZipInfo) -> tuple[str, str, bool, bool]:
    archive_name = info.filename
    if not archive_name or "\x00" in archive_name or "\\" in archive_name:
        raise MASTUPublicArchiveError(
            "ARCHIVE_MEMBER_PATH_UNSAFE",
            "archive member has an empty, NUL, or backslash name",
        )
    is_directory = info.is_dir()
    if is_directory != archive_name.endswith("/"):
        raise MASTUPublicArchiveError(
            "ARCHIVE_MEMBER_TYPE_INCONSISTENT",
            "archive member directory marker is inconsistent",
        )
    raw = archive_name[:-1] if is_directory else archive_name
    posix = PurePosixPath(raw)
    windows = PureWindowsPath(raw)
    parts = raw.split("/")
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or not parts
        or any(value in {"", ".", ".."} for value in parts)
        or unicodedata.normalize("NFC", raw) != raw
    ):
        raise MASTUPublicArchiveError(
            "ARCHIVE_MEMBER_PATH_UNSAFE",
            "archive member path is not a normalized relative locator",
        )
    normalized_collision_key = "/".join(
        unicodedata.normalize("NFKC", value).casefold() for value in parts
    )
    mode = (info.external_attr >> 16) & 0xFFFF
    is_symlink = stat.S_ISLNK(mode)
    file_type = stat.S_IFMT(mode)
    allowed_types = {0, stat.S_IFDIR if is_directory else stat.S_IFREG}
    if file_type not in allowed_types or is_symlink:
        raise MASTUPublicArchiveError(
            "ARCHIVE_MEMBER_TYPE_FORBIDDEN",
            "archive member is not a regular file or directory",
        )
    if bool(info.flag_bits & 0x1):
        raise MASTUPublicArchiveError(
            "ARCHIVE_MEMBER_ENCRYPTED",
            "encrypted archive members are forbidden",
        )
    if info.compress_type not in _ALLOWED_COMPRESSION:
        raise MASTUPublicArchiveError(
            "ARCHIVE_COMPRESSION_FORBIDDEN",
            "archive compression method is not allowlisted",
        )
    return raw, normalized_collision_key, is_directory, is_symlink


def _selected_digest(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    expectation: MASTUPublicSelectedMemberExpectation,
    *,
    chunk_bytes: int,
) -> str:
    digest = sha256()
    observed_bytes = 0
    try:
        with archive.open(info, mode="r") as selected:
            while True:
                chunk = selected.read(chunk_bytes)
                if not chunk:
                    break
                observed_bytes += len(chunk)
                if observed_bytes > expectation.expected_uncompressed_size_bytes:
                    raise MASTUPublicArchiveError(
                        "SELECTED_MEMBER_SIZE_MISMATCH",
                        "selected member exceeded its exact size",
                    )
                digest.update(chunk)
    except MASTUPublicArchiveError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile, NotImplementedError) as error:
        raise MASTUPublicArchiveError(
            "SELECTED_MEMBER_READ_FAILURE",
            "selected archive member cannot be streamed",
        ) from error
    if observed_bytes != expectation.expected_uncompressed_size_bytes:
        raise MASTUPublicArchiveError(
            "SELECTED_MEMBER_SIZE_MISMATCH",
            "selected member differs from its exact size",
        )
    observed = digest.hexdigest()
    if observed != expectation.expected_physical_sha256:
        raise MASTUPublicArchiveError(
            "SELECTED_MEMBER_HASH_MISMATCH",
            "selected member differs from its exact SHA-256",
        )
    return observed


def inspect_release_archive(
    stream: BinaryIO,
    *,
    policy: MASTUPublicReleaseObjectPolicy,
) -> MASTUPublicArchiveInventory:
    """Inventory and hash exact selected members without extraction or unpickling."""

    size = _stream_size(stream)
    if size != policy.expected_archive_size_bytes or size > policy.limits.maximum_archive_bytes:
        raise MASTUPublicArchiveError(
            "ARCHIVE_SIZE_MISMATCH",
            "held archive size differs from its exact policy",
        )
    preinspection_sha256 = _hash_stream(
        stream,
        size=size,
        chunk_bytes=policy.limits.stream_chunk_bytes,
    )
    if preinspection_sha256 != policy.expected_archive_sha256:
        raise MASTUPublicArchiveError(
            "ARCHIVE_HASH_MISMATCH",
            "held archive SHA-256 differs from its exact policy",
        )
    expectations = {value.relative_locator: value for value in policy.selected_members}
    members: list[MASTUPublicArchiveMemberInventory] = []
    exact_names: set[str] = set()
    collision_keys: dict[str, str] = {}
    regular_collision_keys: set[str] = set()
    total_compressed = 0
    total_uncompressed = 0
    try:
        stream.seek(0)
        with zipfile.ZipFile(stream, mode="r") as archive:
            infos = archive.infolist()
            if len(infos) != policy.expected_member_count:
                raise MASTUPublicArchiveError(
                    "ARCHIVE_MEMBER_COUNT_MISMATCH",
                    "archive member count differs from its exact policy",
                )
            for index, info in enumerate(infos):
                if info.filename in exact_names:
                    raise MASTUPublicArchiveError(
                        "ARCHIVE_MEMBER_DUPLICATE",
                        "archive repeats an exact member name",
                    )
                exact_names.add(info.filename)
                normalized, collision_key, is_directory, is_symlink = _safe_member(info)
                if collision_key in collision_keys:
                    raise MASTUPublicArchiveError(
                        "ARCHIVE_MEMBER_NORMALIZED_COLLISION",
                        "archive member names collide after Unicode/case normalization",
                    )
                collision_keys[collision_key] = info.filename
                if not is_directory:
                    regular_collision_keys.add(collision_key)
                if info.file_size < 0 or info.compress_size < 0:
                    raise MASTUPublicArchiveError(
                        "ARCHIVE_MEMBER_SIZE_INVALID",
                        "archive member has a negative size",
                    )
                if info.file_size > policy.limits.maximum_member_uncompressed_bytes:
                    raise MASTUPublicArchiveError(
                        "ARCHIVE_MEMBER_EXPANSION_LIMIT",
                        "archive member exceeds its expanded-byte ceiling",
                    )
                if info.file_size and (
                    info.compress_size <= 0
                    or info.file_size > info.compress_size * policy.limits.maximum_compression_ratio
                ):
                    raise MASTUPublicArchiveError(
                        "ARCHIVE_MEMBER_COMPRESSION_RATIO_LIMIT",
                        "archive member exceeds its compression-ratio ceiling",
                    )
                total_compressed += info.compress_size
                total_uncompressed += info.file_size
                if total_uncompressed > policy.limits.maximum_total_uncompressed_bytes:
                    raise MASTUPublicArchiveError(
                        "ARCHIVE_TOTAL_EXPANSION_LIMIT",
                        "archive exceeds its total expanded-byte ceiling",
                    )
                expectation = expectations.get(info.filename)
                content_sha256 = (
                    None
                    if expectation is None
                    else _selected_digest(
                        archive,
                        info,
                        expectation,
                        chunk_bytes=policy.limits.stream_chunk_bytes,
                    )
                )
                members.append(
                    MASTUPublicArchiveMemberInventory(
                        member_id=(
                            expectation.member_id
                            if expectation is not None
                            else f"inventory-member.{policy.release_object_id}.{index:03d}"
                        ),
                        central_directory_index=index,
                        archive_name=info.filename,
                        normalized_relative_locator=normalized,
                        compressed_size_bytes=info.compress_size,
                        uncompressed_size_bytes=info.file_size,
                        compression_method=info.compress_type,
                        crc32=f"{info.CRC:08x}",
                        unix_mode=(info.external_attr >> 16) & 0xFFFF,
                        is_directory=is_directory,
                        encrypted=False,
                        is_symlink=is_symlink,
                        selected_role=None if expectation is None else expectation.role,
                        selected_content_sha256=content_sha256,
                        outcome_access=(
                            OutcomeAccess.OUTCOME_BLIND
                            if expectation is None
                            else expectation.outcome_access
                        ),
                    )
                )
            missing = set(expectations).difference(exact_names)
            if missing:
                raise MASTUPublicArchiveError(
                    "SELECTED_MEMBER_MISSING",
                    "archive lacks an exact selected member",
                )
    except MASTUPublicArchiveError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile, NotImplementedError) as error:
        raise MASTUPublicArchiveError(
            "ARCHIVE_ZIP_PARSE_FAILURE",
            "held object is not an accepted ZIP archive",
        ) from error
    for collision_key, original in collision_keys.items():
        components = collision_key.split("/")
        if any(
            "/".join(components[:depth]) in regular_collision_keys
            for depth in range(1, len(components))
        ):
            raise MASTUPublicArchiveError(
                "ARCHIVE_MEMBER_ANCESTOR_COLLISION",
                f"regular archive member is an ancestor of {original!r}",
            )
    postinspection_sha256 = _hash_stream(
        stream,
        size=size,
        chunk_bytes=policy.limits.stream_chunk_bytes,
    )
    if postinspection_sha256 != preinspection_sha256:
        raise MASTUPublicArchiveError(
            "ARCHIVE_CHANGED_DURING_INVENTORY",
            "held archive changed during inventory",
        )
    roster = tuple(members)
    return MASTUPublicArchiveInventory(
        inventory_id=f"inventory.{policy.release_object_id}",
        release_policy=ObjectIdentity.from_record(policy.release_object_id, policy),
        held_relative_locator=policy.held_relative_locator,
        archive_size_bytes=size,
        archive_sha256=preinspection_sha256,
        member_count=len(roster),
        total_compressed_bytes=total_compressed,
        total_uncompressed_bytes=total_uncompressed,
        member_roster_sha256=sha256(canonical_json_bytes(roster)).hexdigest(),
        members=roster,
        selected_member_ids=tuple(
            sorted(value.member_id for value in roster if value.selected_role is not None)
        ),
        source_mutated=False,
        member_extraction_performed=False,
        archive_preinspection_postinspection_hashes_match=True,
    )


def inspect_datacite_metadata(
    stream: BinaryIO,
    *,
    policy: MASTUPublicReleaseObjectPolicy,
) -> MASTUPublicDataCiteMetadataReceipt:
    """Authenticate one held DataCite XML object without interpreting source data."""

    size = _stream_size(stream)
    observed = _hash_stream(stream, size=size, chunk_bytes=65_536)
    if size != policy.datacite_metadata_size_bytes or observed != (policy.datacite_metadata_sha256):
        raise MASTUPublicArchiveError(
            "DATACITE_METADATA_IDENTITY_MISMATCH",
            "held DataCite metadata differs from its exact release policy",
        )
    return MASTUPublicDataCiteMetadataReceipt(
        receipt_id=f"datacite.{policy.release_object_id}",
        release_policy=ObjectIdentity.from_record(policy.release_object_id, policy),
        relative_locator=policy.datacite_metadata_relative_locator,
        size_bytes=size,
        physical_sha256=observed,
        source_mutated=False,
    )


__all__ = [
    "MASTUPublicArchiveError",
    "inspect_datacite_metadata",
    "inspect_release_archive",
    "primary_release_policies",
    "primary_trusted_use_acceptance",
]
