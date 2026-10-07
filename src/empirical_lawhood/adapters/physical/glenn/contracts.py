"""Frozen byte and table contracts for the current Glenn 2026 adapter.

The package deliberately accepts an already-open, seekable stream.  Opening a
scientific path and proving mount/containment authority belong to the caller's
infrastructure boundary, not to this byte decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
import re
from typing import BinaryIO


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CRC32 = re.compile(r"^[0-9a-f]{8}$")
_TWO_GIB = 2 * 1024**3
_SIXTY_FOUR_MIB = 64 * 1024**2
_SIXTEEN_MIB = 16 * 1024**2

ACTION_COLUMNS = (
    "focus",
    "astig0",
    "astig45",
    "coma0",
    "coma90",
    "spherical",
)
NUISANCE_COLUMNS = ("o2", "o3", "o4", "target_z")
OBSERVATION_COLUMNS = (
    "run",
    "burst",
    "o2",
    "o3",
    "o4",
    *ACTION_COLUMNS,
    "target_z",
    "fitness",
    "fitness_error",
)
MODEL_COLUMNS = (
    "run",
    "burst",
    "o2",
    "o3",
    "o4",
    *ACTION_COLUMNS,
    "target_z",
    "prediction",
    "error",
)
CANONICAL_COLUMNS = (
    *OBSERVATION_COLUMNS,
    "focal_r50_um",
    "acquisition_order",
    "prior_fitness",
    "prior_fitness_missing",
    "prior_focal_r50_um",
    "prior_focal_r50_missing",
)


class GlennAdapterError(ValueError):
    """The source bytes or requested operation violated the frozen contract."""


class GlennMemberRole(StrEnum):
    """The four non-executing members selected from the larger archive."""

    OBSERVATION_LOG = "OBSERVATION_LOG"
    MODEL_LOG = "MODEL_LOG"
    FOCAL_RADIUS_ARRAY = "FOCAL_RADIUS_ARRAY"
    FIGURE_NOTEBOOK = "FIGURE_NOTEBOOK"


class GlennColumnRole(StrEnum):
    """Operational table role; this is not a scientific-status label."""

    RUN_IDENTITY = "RUN_IDENTITY"
    INDEPENDENT_UNIT = "INDEPENDENT_UNIT"
    NUISANCE_COVARIATE = "NUISANCE_COVARIATE"
    ACTION = "ACTION"
    ONLINE_FITNESS = "ONLINE_FITNESS"
    LOGGED_UNCERTAINTY = "LOGGED_UNCERTAINTY"
    POST_COMMAND_MEDIATOR = "POST_COMMAND_MEDIATOR"
    ACQUISITION_ORDER = "ACQUISITION_ORDER"
    CAUSAL_HISTORY = "CAUSAL_HISTORY"
    HISTORY_MISSINGNESS = "HISTORY_MISSINGNESS"


class GlennExceptionCode(StrEnum):
    """Exact metadata limitations retained by the follow-up transform."""

    FOCAL_R50_GENERATION_METHOD_UNAVAILABLE = "FOCAL_R50_GENERATION_METHOD_UNAVAILABLE"
    FOCAL_R50_UNCERTAINTY_UNAVAILABLE = "FOCAL_R50_UNCERTAINTY_UNAVAILABLE"
    SOURCE_ACTION_LABEL_SEMANTICS_CAVEAT = "SOURCE_ACTION_LABEL_SEMANTICS_CAVEAT"


class GlennMemberMediaProfile(StrEnum):
    PLAIN_TEXT = "plain-text"
    NPY_NO_PICKLE = "numpy-npy-no-pickle"
    NOTEBOOK_JSON_NO_EXECUTION = "utf8-json-no-execution"


_ROLE_CONTRACT = {
    GlennMemberRole.OBSERVATION_LOG: (
        "text/plain",
        GlennMemberMediaProfile.PLAIN_TEXT.value,
        "exact-allowlist-plain-text",
    ),
    GlennMemberRole.MODEL_LOG: (
        "text/plain",
        GlennMemberMediaProfile.PLAIN_TEXT.value,
        "exact-allowlist-plain-text",
    ),
    GlennMemberRole.FOCAL_RADIUS_ARRAY: (
        "application/x-npy",
        GlennMemberMediaProfile.NPY_NO_PICKLE.value,
        "exact-allowlist-npy-allow-pickle-false",
    ),
    GlennMemberRole.FIGURE_NOTEBOOK: (
        "application/x-ipynb+json",
        GlennMemberMediaProfile.NOTEBOOK_JSON_NO_EXECUTION.value,
        "exact-allowlist-utf8-json-no-execution",
    ),
}


@dataclass(frozen=True, slots=True)
class GlennMemberExpectation:
    """Exact identity and decoder profile for one selected archive member."""

    member_id: str
    role: GlennMemberRole
    relative_locator: str
    expected_physical_sha256: str
    expected_compressed_size_bytes: int
    expected_uncompressed_size_bytes: int
    expected_crc32: str
    media_type: str
    format_profile_id: str
    inspection_policy_id: str

    def __post_init__(self) -> None:
        if not self.member_id or self.member_id != self.member_id.strip():
            raise ValueError("member_id must be nonblank and whitespace-exact")
        if not self.relative_locator or self.relative_locator != self.relative_locator.strip():
            raise ValueError("member locator must be nonblank and whitespace-exact")
        if "\x00" in self.relative_locator or "\\" in self.relative_locator:
            raise ValueError("member locator contains a forbidden character")
        if _SHA256.fullmatch(self.expected_physical_sha256) is None:
            raise ValueError("member SHA-256 must be lowercase hexadecimal")
        if self.expected_compressed_size_bytes <= 0:
            raise ValueError("selected member compressed size must be positive")
        if self.expected_uncompressed_size_bytes <= 0:
            raise ValueError("selected member uncompressed size must be positive")
        if _CRC32.fullmatch(self.expected_crc32) is None:
            raise ValueError("member CRC-32 must be eight lowercase hexadecimal digits")
        expected_media, expected_profile, expected_policy = _ROLE_CONTRACT[self.role]
        if (
            self.media_type,
            self.format_profile_id,
            self.inspection_policy_id,
        ) != (expected_media, expected_profile, expected_policy):
            raise ValueError("member role and media/inspection profile disagree")


@dataclass(frozen=True, slots=True)
class GlennArchiveLimits:
    """Fail-closed ZIP and decoder work ceilings with absolute upper bounds."""

    maximum_archive_bytes: int
    maximum_members: int
    maximum_central_directory_bytes: int
    maximum_member_uncompressed_bytes: int
    maximum_total_uncompressed_bytes: int
    maximum_member_compression_ratio: float
    maximum_total_compression_ratio: float
    maximum_selected_uncompressed_bytes: int
    stream_chunk_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        integer_bounds = (
            ("maximum_archive_bytes", self.maximum_archive_bytes, _TWO_GIB),
            ("maximum_members", self.maximum_members, 100_000),
            (
                "maximum_central_directory_bytes",
                self.maximum_central_directory_bytes,
                _SIXTY_FOUR_MIB,
            ),
            (
                "maximum_member_uncompressed_bytes",
                self.maximum_member_uncompressed_bytes,
                _TWO_GIB,
            ),
            (
                "maximum_total_uncompressed_bytes",
                self.maximum_total_uncompressed_bytes,
                _TWO_GIB,
            ),
            (
                "maximum_selected_uncompressed_bytes",
                self.maximum_selected_uncompressed_bytes,
                _SIXTEEN_MIB,
            ),
        )
        for name, value, absolute in integer_bounds:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= absolute:
                raise ValueError(f"{name} must be a positive integer at most {absolute}")
        if not 4096 <= self.stream_chunk_bytes <= _SIXTEEN_MIB:
            raise ValueError("stream_chunk_bytes must be in [4096, 16 MiB]")
        for name, ratio in (
            ("maximum_member_compression_ratio", self.maximum_member_compression_ratio),
            ("maximum_total_compression_ratio", self.maximum_total_compression_ratio),
        ):
            if not math.isfinite(ratio) or not 1.0 <= ratio <= 350.0:
                raise ValueError(f"{name} must be finite and in [1, 350]")
        if self.maximum_member_uncompressed_bytes > self.maximum_total_uncompressed_bytes:
            raise ValueError("member expansion ceiling cannot exceed total expansion ceiling")
        if self.maximum_selected_uncompressed_bytes > self.maximum_total_uncompressed_bytes:
            raise ValueError("selected expansion ceiling cannot exceed total expansion ceiling")


@dataclass(frozen=True, slots=True)
class GlennTransformProfile:
    """All byte, shape, semantic, and safe-decoder expectations for one release."""

    profile_id: str
    expected_archive_size_bytes: int
    expected_archive_sha256: str
    members: tuple[GlennMemberExpectation, ...]
    limits: GlennArchiveLimits
    expected_run: str
    expected_rows: int
    initial_exploration_rows: int
    focal_native_to_um: float
    required_notebook_markers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.profile_id or self.profile_id != self.profile_id.strip():
            raise ValueError("profile_id must be nonblank and whitespace-exact")
        if not 0 < self.expected_archive_size_bytes <= self.limits.maximum_archive_bytes:
            raise ValueError("expected archive size is outside its work ceiling")
        if _SHA256.fullmatch(self.expected_archive_sha256) is None:
            raise ValueError("archive SHA-256 must be lowercase hexadecimal")
        if len(self.members) != 4 or {member.role for member in self.members} != set(
            GlennMemberRole
        ):
            raise ValueError("Glenn transform must select exactly one member for each role")
        if len({member.member_id for member in self.members}) != 4:
            raise ValueError("selected member IDs must be unique")
        if len({member.relative_locator for member in self.members}) != 4:
            raise ValueError("selected member locators must be unique")
        selected_bytes = sum(member.expected_uncompressed_size_bytes for member in self.members)
        if selected_bytes > self.limits.maximum_selected_uncompressed_bytes:
            raise ValueError("selected member expectations exceed their expansion ceiling")
        if not self.expected_run or self.expected_run != self.expected_run.strip():
            raise ValueError("expected_run must be nonblank and whitespace-exact")
        if not 1 <= self.expected_rows <= 10_000:
            raise ValueError("expected_rows must be in [1, 10000]")
        if not 1 <= self.initial_exploration_rows <= self.expected_rows:
            raise ValueError("initial exploration rows must be within the table")
        if not math.isfinite(self.focal_native_to_um) or self.focal_native_to_um <= 0:
            raise ValueError("focal conversion must be finite and positive")
        if (
            not self.required_notebook_markers
            or len(set(self.required_notebook_markers)) != len(self.required_notebook_markers)
            or any(not marker for marker in self.required_notebook_markers)
        ):
            raise ValueError("notebook markers must be nonempty and unique")

    def member_for(self, role: GlennMemberRole) -> GlennMemberExpectation:
        return next(member for member in self.members if member.role is role)


@dataclass(slots=True)
class GlennArchiveInput:
    """A caller-opened stream plus its bounded locator/guard attestation.

    The adapter never interprets ``locator`` as a filesystem path.  Production
    callers must open the stream only after their storage guard has proved
    mount, containment, regular-file identity, and read authority.
    """

    stream: BinaryIO
    locator: str
    guard_evidence_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("locator", self.locator),
            ("guard_evidence_id", self.guard_evidence_id),
        ):
            if not value or value != value.strip() or "\x00" in value or len(value) > 4096:
                raise ValueError(f"{name} must be a bounded nonblank exact string")


_GLENN_MEMBERS = (
    GlennMemberExpectation(
        member_id="member.ionacc-model",
        role=GlennMemberRole.MODEL_LOG,
        relative_locator="Data/Automation/Outputs/IonAcc_GP_20210730_run14_model.txt",
        expected_physical_sha256=(
            "611213987a106c87e9a20e5849c48c9ffbd72d953321d5172f39386a2cd03030"
        ),
        expected_compressed_size_bytes=4_371,
        expected_uncompressed_size_bytes=17_133,
        expected_crc32="39b3e00e",
        media_type="text/plain",
        format_profile_id="plain-text",
        inspection_policy_id="exact-allowlist-plain-text",
    ),
    GlennMemberExpectation(
        member_id="member.ionacc-run",
        role=GlennMemberRole.OBSERVATION_LOG,
        relative_locator="Data/Automation/Outputs/IonAcc_GP_20210730_run14.txt",
        expected_physical_sha256=(
            "ce851cb455705af48295d6e159a783d3f302172f990e24596898415e1af543bc"
        ),
        expected_compressed_size_bytes=4_425,
        expected_uncompressed_size_bytes=17_122,
        expected_crc32="d6735be8",
        media_type="text/plain",
        format_profile_id="plain-text",
        inspection_policy_id="exact-allowlist-plain-text",
    ),
    GlennMemberExpectation(
        member_id="member.notebook",
        role=GlennMemberRole.FIGURE_NOTEBOOK,
        relative_locator="Figures/Figs_6-8.ipynb",
        expected_physical_sha256=(
            "6c040063077c7ba804189e323a557b0c39333b3eae82f0c9468df9cd35643694"
        ),
        expected_compressed_size_bytes=485_786,
        expected_uncompressed_size_bytes=684_618,
        expected_crc32="13979fac",
        media_type="application/x-ipynb+json",
        format_profile_id="utf8-json-no-execution",
        inspection_policy_id="exact-allowlist-utf8-json-no-execution",
    ),
    GlennMemberExpectation(
        member_id="member.spot-radius",
        role=GlennMemberRole.FOCAL_RADIUS_ARRAY,
        relative_locator=("Figures/Figs_6-8 Files/spot_radius_burst_20210730_run14.npy"),
        expected_physical_sha256=(
            "9f602bc17c7a10bc528cdcf89d8cc0aa562008be2778be07bd0908738ca71a9f"
        ),
        expected_compressed_size_bytes=429,
        expected_uncompressed_size_bytes=936,
        expected_crc32="a7e482f1",
        media_type="application/x-npy",
        format_profile_id="numpy-npy-no-pickle",
        inspection_policy_id="exact-allowlist-npy-allow-pickle-false",
    ),
)

GLENN_2026_PROFILE = GlennTransformProfile(
    profile_id="glenn-zenodo-17163053-g1-custodied-reproduction",
    expected_archive_size_bytes=549_154_725,
    expected_archive_sha256=("18af82987cfd5369ba670da6ab8413656d89ebaa1f0ed34b1034453ad2c8d92a"),
    members=_GLENN_MEMBERS,
    limits=GlennArchiveLimits(
        maximum_archive_bytes=600_000_000,
        maximum_members=20_000,
        maximum_central_directory_bytes=16 * 1024**2,
        maximum_member_uncompressed_bytes=1_000_000_000,
        maximum_total_uncompressed_bytes=2_000_000_000,
        maximum_member_compression_ratio=350.0,
        maximum_total_compression_ratio=350.0,
        maximum_selected_uncompressed_bytes=2 * 1024**2,
    ),
    expected_run="20210730/run14",
    expected_rows=101,
    initial_exploration_rows=20,
    focal_native_to_um=1_000.0,
    required_notebook_markers=(
        'opt_run_name = "20210730/run14"',
        'burst_nums = opt_df["burst"].values',
        "spot_radius_burst = np.load(",
        "spot_radius_burst * 1e3",
    ),
)


COLUMN_ROLES = {
    "run": GlennColumnRole.RUN_IDENTITY,
    "burst": GlennColumnRole.INDEPENDENT_UNIT,
    "o2": GlennColumnRole.NUISANCE_COVARIATE,
    "o3": GlennColumnRole.NUISANCE_COVARIATE,
    "o4": GlennColumnRole.NUISANCE_COVARIATE,
    "focus": GlennColumnRole.ACTION,
    "astig0": GlennColumnRole.ACTION,
    "astig45": GlennColumnRole.ACTION,
    "coma0": GlennColumnRole.ACTION,
    "coma90": GlennColumnRole.ACTION,
    "spherical": GlennColumnRole.ACTION,
    "target_z": GlennColumnRole.NUISANCE_COVARIATE,
    "fitness": GlennColumnRole.ONLINE_FITNESS,
    "fitness_error": GlennColumnRole.LOGGED_UNCERTAINTY,
    "focal_r50_um": GlennColumnRole.POST_COMMAND_MEDIATOR,
    "acquisition_order": GlennColumnRole.ACQUISITION_ORDER,
    "prior_fitness": GlennColumnRole.CAUSAL_HISTORY,
    "prior_fitness_missing": GlennColumnRole.HISTORY_MISSINGNESS,
    "prior_focal_r50_um": GlennColumnRole.CAUSAL_HISTORY,
    "prior_focal_r50_missing": GlennColumnRole.HISTORY_MISSINGNESS,
}

if tuple(COLUMN_ROLES) != CANONICAL_COLUMNS:  # pragma: no cover - import invariant
    raise RuntimeError("Glenn column-role registry drifted from canonical order")


__all__ = [
    "ACTION_COLUMNS",
    "CANONICAL_COLUMNS",
    "COLUMN_ROLES",
    "GLENN_2026_PROFILE",
    "MODEL_COLUMNS",
    "NUISANCE_COLUMNS",
    "OBSERVATION_COLUMNS",
    "GlennAdapterError",
    "GlennArchiveInput",
    "GlennArchiveLimits",
    "GlennColumnRole",
    "GlennExceptionCode",
    "GlennMemberExpectation",
    "GlennMemberRole",
    "GlennTransformProfile",
]
