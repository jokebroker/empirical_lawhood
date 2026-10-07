"""Exact held-object and trusted-use contracts for public MAST-U releases.

These records describe already-held public objects.  They are deliberately not
FAIR-MAST query receipts and grant no DOI contact, publication, or scientific
qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
import re
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)


_DOI = re.compile(r"^10\.14468/[A-Z0-9]{4}-[A-Z0-9]{4}$")
_SHA1 = re.compile(r"^[0-9a-f]{40}$")
_CRC32 = re.compile(r"^[0-9a-f]{8}$")


def _positive_integer(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")


def _nonnegative_integer(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer")


class MASTUPublicReleaseRole(StrEnum):
    DYNAMIC_TRACE_CORPUS = "DYNAMIC_TRACE_CORPUS"
    MACHINE_CONFIGURATION_COMPARATOR = "MACHINE_CONFIGURATION_COMPARATOR"


class MASTUPublicMemberRole(StrEnum):
    PROVENANCE_SOURCE = "PROVENANCE_SOURCE"
    DEVELOPMENT_SHOT_TRACE = "DEVELOPMENT_SHOT_TRACE"
    EVALUATION_SHOT_TRACE_SEALED = "EVALUATION_SHOT_TRACE_SEALED"
    ACTIVE_COILS = "ACTIVE_COILS"
    NONSYMMETRIC_ACTIVE_COILS = "NONSYMMETRIC_ACTIVE_COILS"
    LIMITER = "LIMITER"
    MAGNETIC_PROBES = "MAGNETIC_PROBES"
    PASSIVE_COILS = "PASSIVE_COILS"
    WALL = "WALL"


@dataclass(frozen=True, slots=True)
class MASTUPublicArchiveLimits(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu-release-qualification/mastu-public-archive-limits'

    limits_id: str
    maximum_archive_bytes: int
    maximum_member_count: int
    maximum_member_uncompressed_bytes: int
    maximum_total_uncompressed_bytes: int
    maximum_compression_ratio: int
    stream_chunk_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.limits_id, field_name="limits_id")
        for field_name in (
            "maximum_archive_bytes",
            "maximum_member_count",
            "maximum_member_uncompressed_bytes",
            "maximum_total_uncompressed_bytes",
            "maximum_compression_ratio",
            "stream_chunk_bytes",
        ):
            _positive_integer(getattr(self, field_name), field_name=field_name)
        if self.maximum_member_uncompressed_bytes > self.maximum_total_uncompressed_bytes:
            raise ValueError("per-member expansion ceiling exceeds the archive ceiling")
        if self.stream_chunk_bytes > self.maximum_member_uncompressed_bytes:
            raise ValueError("stream chunk exceeds the per-member ceiling")


@dataclass(frozen=True, slots=True)
class MASTUPublicSelectedMemberExpectation(CanonicalRecord):
    """One exact serialized member selected from a hash-pinned release object."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu-release-qualification/mastu-public-selected-member-expectation'

    member_id: str
    relative_locator: str
    role: MASTUPublicMemberRole
    expected_uncompressed_size_bytes: int
    expected_physical_sha256: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        validate_relative_locator(self.relative_locator)
        if not isinstance(self.role, MASTUPublicMemberRole):
            raise TypeError("role must be a MASTUPublicMemberRole")
        _positive_integer(
            self.expected_uncompressed_size_bytes,
            field_name="expected_uncompressed_size_bytes",
        )
        validate_sha256(
            self.expected_physical_sha256,
            field_name="expected_physical_sha256",
        )
        if not isinstance(self.outcome_access, OutcomeAccess):
            raise TypeError("outcome_access must be an OutcomeAccess")
        expected_access = {
            MASTUPublicMemberRole.DEVELOPMENT_SHOT_TRACE: (OutcomeAccess.DEVELOPMENT_VISIBLE),
            MASTUPublicMemberRole.EVALUATION_SHOT_TRACE_SEALED: (OutcomeAccess.EVALUATION_SEALED),
        }.get(self.role, OutcomeAccess.OUTCOME_BLIND)
        if self.outcome_access is not expected_access:
            raise ValueError("selected member role and outcome access differ")


@dataclass(frozen=True, slots=True)
class MASTUPublicReleaseObjectPolicy(CanonicalRecord):
    """Exact policy for one already-finalized UKAEA public ZIP object."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu-release-qualification/mastu-public-release-object-policy'

    release_object_id: str
    release_role: MASTUPublicReleaseRole
    doi: str
    version_label: str
    object_name: str
    held_relative_locator: str
    datacite_metadata_relative_locator: str
    datacite_metadata_size_bytes: int
    datacite_metadata_sha256: str
    advertised_sha1: str
    expected_archive_size_bytes: int
    expected_archive_sha256: str
    expected_member_count: int
    licence_spdx_id: str
    limits: MASTUPublicArchiveLimits
    selected_members: tuple[MASTUPublicSelectedMemberExpectation, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.release_object_id, field_name="release_object_id")
        if not isinstance(self.release_role, MASTUPublicReleaseRole):
            raise TypeError("release_role must be a MASTUPublicReleaseRole")
        if not isinstance(self.doi, str) or _DOI.fullmatch(self.doi) is None:
            raise ValueError("doi must be the canonical UKAEA DOI spelling")
        validate_nonempty(self.version_label, field_name="version_label")
        validate_nonempty(self.object_name, field_name="object_name")
        if "/" in self.object_name or "\\" in self.object_name:
            raise ValueError("object_name must be one filename")
        validate_relative_locator(self.held_relative_locator)
        if not self.held_relative_locator.endswith(f"/{self.object_name}"):
            raise ValueError("held locator and public object name differ")
        validate_relative_locator(self.datacite_metadata_relative_locator)
        if not self.datacite_metadata_relative_locator.endswith("/custody/datacite-metadata.xml"):
            raise ValueError("DataCite metadata locator differs from its custody contract")
        _positive_integer(
            self.datacite_metadata_size_bytes,
            field_name="datacite_metadata_size_bytes",
        )
        validate_sha256(
            self.datacite_metadata_sha256,
            field_name="datacite_metadata_sha256",
        )
        if (
            not isinstance(self.advertised_sha1, str)
            or _SHA1.fullmatch(self.advertised_sha1) is None
        ):
            raise ValueError("advertised_sha1 must be a lowercase SHA-1 digest")
        _positive_integer(
            self.expected_archive_size_bytes,
            field_name="expected_archive_size_bytes",
        )
        validate_sha256(
            self.expected_archive_sha256,
            field_name="expected_archive_sha256",
        )
        _positive_integer(self.expected_member_count, field_name="expected_member_count")
        if self.expected_archive_size_bytes > self.limits.maximum_archive_bytes:
            raise ValueError("exact archive size exceeds its bounded policy")
        if self.expected_member_count > self.limits.maximum_member_count:
            raise ValueError("exact member count exceeds its bounded policy")
        if self.licence_spdx_id != "CC-BY-NC-SA-4.0":
            raise ValueError("public MAST-U policy must retain CC-BY-NC-SA-4.0")
        require_sorted_unique_ids(
            self.selected_members,
            attribute="member_id",
            field_name="selected_members",
        )
        if not self.selected_members:
            raise ValueError("release policy must select at least one exact member")
        locators = tuple(value.relative_locator for value in self.selected_members)
        if len(set(locators)) != len(locators):
            raise ValueError("selected member locators repeat")
        expected_roles = {
            MASTUPublicReleaseRole.DYNAMIC_TRACE_CORPUS: {
                MASTUPublicMemberRole.PROVENANCE_SOURCE,
                MASTUPublicMemberRole.DEVELOPMENT_SHOT_TRACE,
                MASTUPublicMemberRole.EVALUATION_SHOT_TRACE_SEALED,
            },
            MASTUPublicReleaseRole.MACHINE_CONFIGURATION_COMPARATOR: {
                MASTUPublicMemberRole.ACTIVE_COILS,
                MASTUPublicMemberRole.NONSYMMETRIC_ACTIVE_COILS,
                MASTUPublicMemberRole.LIMITER,
                MASTUPublicMemberRole.MAGNETIC_PROBES,
                MASTUPublicMemberRole.PASSIVE_COILS,
                MASTUPublicMemberRole.WALL,
            },
        }[self.release_role]
        if {value.role for value in self.selected_members} != expected_roles:
            raise ValueError("release policy selected-member roles are incomplete")


@dataclass(frozen=True, slots=True)
class MASTUPublicTrustedUseAcceptance(CanonicalRecord):
    """Owner's bounded personal-project acceptance of trusted local loading."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu-release-qualification/mastu-public-trusted-use-acceptance'

    acceptance_id: str
    experiment_id: str
    decision_date: str
    accepted_release_policies: tuple[ObjectIdentity, ...]
    licence_spdx_id: str
    personal_project: bool
    noncommercial_use: bool
    attribution_sharealike_obligations_accepted: bool
    trusted_local_pickle_code_execution_risk_accepted: bool
    pinned_project_uv_environment_required: bool
    publication_authorized: bool
    upload_authorized: bool
    bqts_execution_authorized: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.acceptance_id, field_name="acceptance_id")
        validate_stable_id(self.experiment_id, field_name="experiment_id")
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", self.decision_date):
            raise ValueError("decision_date must use YYYY-MM-DD")
        require_sorted_unique_ids(
            self.accepted_release_policies,
            attribute="object_id",
            field_name="accepted_release_policies",
        )
        if tuple(value.object_id for value in self.accepted_release_policies) != (
            "release.26m5-ey02",
            "release.jmpp-4c57.1.0",
        ):
            raise ValueError("trusted-use acceptance is limited to primary 26M5/JMPP objects")
        if any(
            value.object_schema != MASTUPublicReleaseObjectPolicy.SCHEMA
            for value in self.accepted_release_policies
        ):
            raise ValueError("trusted-use acceptance binds another record schema")
        if self.licence_spdx_id != "CC-BY-NC-SA-4.0":
            raise ValueError("trusted-use acceptance must retain the advertised licence")
        if not all(
            (
                self.personal_project,
                self.noncommercial_use,
                self.attribution_sharealike_obligations_accepted,
                self.trusted_local_pickle_code_execution_risk_accepted,
                self.pinned_project_uv_environment_required,
            )
        ):
            raise ValueError("trusted local use requires every bounded acceptance")
        if self.publication_authorized or self.upload_authorized:
            raise ValueError("trusted-use acceptance grants no publication or upload authority")
        if self.bqts_execution_authorized:
            raise ValueError("primary trusted-use acceptance grants no BQTS execution authority")


@dataclass(frozen=True, slots=True)
class MASTUPublicArchiveMemberInventory(CanonicalRecord):
    """One central-directory member; content is hashed only when selected."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu-release-qualification/mastu-public-archive-member-inventory'

    member_id: str
    central_directory_index: int
    archive_name: str
    normalized_relative_locator: str
    compressed_size_bytes: int
    uncompressed_size_bytes: int
    compression_method: int
    crc32: str
    unix_mode: int
    is_directory: bool
    encrypted: bool
    is_symlink: bool
    selected_role: MASTUPublicMemberRole | None
    selected_content_sha256: str | None
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        _nonnegative_integer(
            self.central_directory_index,
            field_name="central_directory_index",
        )
        validate_nonempty(self.archive_name, field_name="archive_name")
        validate_relative_locator(self.normalized_relative_locator)
        for field_name in (
            "compressed_size_bytes",
            "uncompressed_size_bytes",
            "compression_method",
            "unix_mode",
        ):
            _nonnegative_integer(getattr(self, field_name), field_name=field_name)
        if not isinstance(self.crc32, str) or _CRC32.fullmatch(self.crc32) is None:
            raise ValueError("crc32 must be eight lowercase hexadecimal characters")
        if self.encrypted or self.is_symlink:
            raise ValueError("accepted archive inventory cannot retain encryption or links")
        if self.is_directory:
            if self.selected_role is not None or self.selected_content_sha256 is not None:
                raise ValueError("archive directories cannot be selected payload members")
            if self.uncompressed_size_bytes != 0:
                raise ValueError("archive directory has nonzero expanded bytes")
        selected = self.selected_role is not None
        if selected != (self.selected_content_sha256 is not None):
            raise ValueError("selected member role and content hash must be present together")
        if self.selected_content_sha256 is not None:
            validate_sha256(
                self.selected_content_sha256,
                field_name="selected_content_sha256",
            )
        expected_access = (
            OutcomeAccess.OUTCOME_BLIND
            if self.selected_role is None
            else {
                MASTUPublicMemberRole.DEVELOPMENT_SHOT_TRACE: (OutcomeAccess.DEVELOPMENT_VISIBLE),
                MASTUPublicMemberRole.EVALUATION_SHOT_TRACE_SEALED: (
                    OutcomeAccess.EVALUATION_SEALED
                ),
            }.get(self.selected_role, OutcomeAccess.OUTCOME_BLIND)
        )
        if self.outcome_access is not expected_access:
            raise ValueError("archive member role and outcome access differ")


@dataclass(frozen=True, slots=True)
class MASTUPublicArchiveInventory(CanonicalRecord):
    """Exact non-executing inventory for one hash-pinned public ZIP."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu-release-qualification/mastu-public-archive-inventory'

    inventory_id: str
    release_policy: ObjectIdentity
    held_relative_locator: str
    archive_size_bytes: int
    archive_sha256: str
    member_count: int
    total_compressed_bytes: int
    total_uncompressed_bytes: int
    member_roster_sha256: str
    members: tuple[MASTUPublicArchiveMemberInventory, ...]
    selected_member_ids: tuple[str, ...]
    source_mutated: bool
    member_extraction_performed: bool
    archive_preinspection_postinspection_hashes_match: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.inventory_id, field_name="inventory_id")
        if (
            not isinstance(self.release_policy, ObjectIdentity)
            or self.release_policy.object_schema != MASTUPublicReleaseObjectPolicy.SCHEMA
        ):
            raise ValueError("inventory must bind one exact public release policy")
        validate_relative_locator(self.held_relative_locator)
        for field_name in (
            "archive_size_bytes",
            "member_count",
            "total_compressed_bytes",
            "total_uncompressed_bytes",
        ):
            _positive_integer(getattr(self, field_name), field_name=field_name)
        validate_sha256(self.archive_sha256, field_name="archive_sha256")
        validate_sha256(self.member_roster_sha256, field_name="member_roster_sha256")
        if len(self.members) != self.member_count:
            raise ValueError("archive member count differs from its exact roster")
        if tuple(value.central_directory_index for value in self.members) != tuple(
            range(self.member_count)
        ):
            raise ValueError("archive central-directory indexes are not contiguous")
        if len({value.archive_name for value in self.members}) != self.member_count:
            raise ValueError("archive roster repeats an exact member name")
        if self.total_compressed_bytes != sum(
            value.compressed_size_bytes for value in self.members
        ) or self.total_uncompressed_bytes != sum(
            value.uncompressed_size_bytes for value in self.members
        ):
            raise ValueError("archive byte totals differ from the exact member roster")
        if self.member_roster_sha256 != sha256(canonical_json_bytes(self.members)).hexdigest():
            raise ValueError("archive member-roster digest differs from its exact roster")
        require_sorted_unique_strings(
            self.selected_member_ids,
            field_name="selected_member_ids",
            allow_empty=False,
        )
        observed_selected = tuple(
            sorted(value.member_id for value in self.members if value.selected_role is not None)
        )
        if self.selected_member_ids != observed_selected:
            raise ValueError("selected member IDs differ from the inventory roster")
        if self.source_mutated or self.member_extraction_performed:
            raise ValueError("archive inventory must be read-only and non-extracting")
        if not self.archive_preinspection_postinspection_hashes_match:
            raise ValueError("archive changed during inventory")


@dataclass(frozen=True, slots=True)
class MASTUPublicDataCiteMetadataReceipt(CanonicalRecord):
    """Exact read-only identity of one held official DataCite XML record."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mastu-release-qualification/mastu-public-data-cite-metadata-receipt'

    receipt_id: str
    release_policy: ObjectIdentity
    relative_locator: str
    size_bytes: int
    physical_sha256: str
    source_mutated: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.release_policy.object_schema != MASTUPublicReleaseObjectPolicy.SCHEMA:
            raise ValueError("DataCite receipt binds another release-policy schema")
        validate_relative_locator(self.relative_locator)
        _positive_integer(self.size_bytes, field_name="size_bytes")
        validate_sha256(self.physical_sha256, field_name="physical_sha256")
        if self.source_mutated:
            raise ValueError("DataCite metadata verification must be read-only")


__all__ = [
    'MASTUPublicArchiveInventory',
    'MASTUPublicArchiveLimits',
    'MASTUPublicArchiveMemberInventory',
    'MASTUPublicDataCiteMetadataReceipt',
    'MASTUPublicMemberRole',
    'MASTUPublicReleaseObjectPolicy',
    'MASTUPublicReleaseRole',
    'MASTUPublicSelectedMemberExpectation',
    'MASTUPublicTrustedUseAcceptance',
]
