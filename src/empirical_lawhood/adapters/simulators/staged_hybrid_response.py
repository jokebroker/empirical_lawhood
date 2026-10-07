"Shared compact records for staged hybrid response custody, gates and tranche synthesis."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp


@dataclass(frozen=True, slots=True)
class StagedHybridResponseSourceFile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/staged-hybrid-response-source-file'

    relative_path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            not self.relative_path
            or self.relative_path.startswith("/")
            or ".." in self.relative_path.split("/")
            or self.size_bytes <= 0
        ):
            raise ValueError("Staged hybrid response source file locator is invalid")
        validate_sha256(self.sha256, field_name="sha256")


@dataclass(frozen=True, slots=True)
class StagedHybridResponseImplementationManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/staged-hybrid-response-implementation-manifest'

    manifest_id: str
    source_files: tuple[StagedHybridResponseSourceFile, ...]
    installed_distribution_record_sha256: tuple[str, ...]
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        paths = tuple(value.relative_path for value in self.source_files)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("Staged hybrid response source manifest is not sorted and unique")
        for value in self.installed_distribution_record_sha256:
            validate_sha256(
                value,
                field_name="installed_distribution_record_sha256",
            )
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )


@dataclass(frozen=True, slots=True)
class StagedHybridResponseEpisodeArtifactReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/staged-hybrid-response-episode-artifact-receipt'

    receipt_id: str
    lane_id: str
    config: ObjectIdentity
    episode: ObjectIdentity
    cell_id: str
    view_id: str
    word_id: str
    branch_id: str
    relative_path: str
    physical_sha256: str
    size_bytes: int
    runtime_seconds: Decimal
    disposition: str
    committed_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("lane_id", self.lane_id),
            ("cell_id", self.cell_id),
            ("view_id", self.view_id),
            ("word_id", self.word_id),
            ("branch_id", self.branch_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(self.physical_sha256, field_name="physical_sha256")
        parse_utc_timestamp(self.committed_at_utc, field_name="committed_at_utc")
        if self.size_bytes <= 0 or self.relative_path.startswith("/"):
            raise ValueError("Staged hybrid response episode receipt is invalid")


@dataclass(frozen=True, slots=True)
class StagedHybridResponseAcquisitionIndex(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/staged-hybrid-response-acquisition-index'

    index_id: str
    lane_id: str
    config: ObjectIdentity
    episode_receipts: tuple[StagedHybridResponseEpisodeArtifactReceipt, ...]
    attempted_cell_ids: tuple[str, ...]
    effective_cell_ids: tuple[str, ...]
    expected_episode_count: int
    total_size_bytes: int
    maximum_episode_size_bytes: int
    terminal_acquisition: bool
    recovery_verified_without_reacquisition: bool
    completed_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.index_id, field_name="index_id")
        validate_stable_id(self.lane_id, field_name="lane_id")
        require_sorted_unique_strings(
            self.attempted_cell_ids,
            field_name="attempted_cell_ids",
        )
        require_sorted_unique_strings(
            self.effective_cell_ids,
            field_name="effective_cell_ids",
        )
        keys = tuple(
            (
                value.cell_id,
                value.view_id,
                value.word_id,
                value.branch_id,
            )
            for value in self.episode_receipts
        )
        if (
            keys != tuple(sorted(keys))
            or len(keys) != len(set(keys))
            or len(keys) != self.expected_episode_count
            or self.total_size_bytes
            != sum(value.size_bytes for value in self.episode_receipts)
            or self.maximum_episode_size_bytes
            != max(value.size_bytes for value in self.episode_receipts)
            or not self.terminal_acquisition
            or not self.recovery_verified_without_reacquisition
        ):
            raise ValueError("Staged hybrid response acquisition index is incomplete")
        parse_utc_timestamp(self.completed_at_utc, field_name="completed_at_utc")


@dataclass(frozen=True, slots=True)
class StagedHybridResponseGateRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/staged-hybrid-response-gate-record'

    record_id: str
    lane_id: str
    record_kind: str
    subject: ObjectIdentity
    prerequisite: ObjectIdentity | None
    owner_id: str
    authorization_basis: str
    fresh_unit_ids: tuple[str, ...]
    created_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.record_id, field_name="record_id")
        validate_stable_id(self.lane_id, field_name="lane_id")
        validate_stable_id(self.owner_id, field_name="owner_id")
        require_sorted_unique_strings(
            self.fresh_unit_ids,
            field_name="fresh_unit_ids",
        )
        parse_utc_timestamp(self.created_at_utc, field_name="created_at_utc")
        if self.record_kind not in {
            "SCIENTIFIC_APPROVAL",
            "ISSUE",
            "EXECUTION_AUTHORITY",
            "EXECUTION_RECEIPT",
            "REVEAL_AUTHORITY",
            "REVEAL_RECEIPT",
            "CUSTODY",
            "CONSTRUCTION_AUTHORITY",
        }:
            raise ValueError("Staged hybrid response gate record kind is invalid")


@dataclass(frozen=True, slots=True)
class StagedHybridResponseTrancheFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/staged-hybrid-response-tranche-freeze'

    freeze_id: str
    gym_design: ObjectIdentity
    gym_qualification_config: ObjectIdentity
    gym_evaluation_config: ObjectIdentity
    pybamm_design: ObjectIdentity
    pybamm_qualification_config: ObjectIdentity
    pybamm_development_config: ObjectIdentity
    pybamm_evaluation_config: ObjectIdentity
    boptest_conditional_design: ObjectIdentity | None
    formal_register: ObjectIdentity
    config_file_sha256: tuple[tuple[str, str], ...]
    no_cross_lane_learning: bool
    frozen_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        paths = tuple(value[0] for value in self.config_file_sha256)
        if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
            raise ValueError("Staged hybrid response freeze config files are not sorted and unique")
        for _, digest in self.config_file_sha256:
            validate_sha256(digest, field_name="config_file_sha256")
        parse_utc_timestamp(self.frozen_at_utc, field_name="frozen_at_utc")
        if (
            not self.no_cross_lane_learning
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("Staged hybrid response freeze permits cross-lane outcome tuning")
        gym_ids = {
            self.gym_design.object_id,
            self.gym_qualification_config.object_id,
            self.gym_evaluation_config.object_id,
        }
        pybamm_ids = {
            self.pybamm_design.object_id,
            self.pybamm_qualification_config.object_id,
            self.pybamm_development_config.object_id,
            self.pybamm_evaluation_config.object_id,
        }
        if len(gym_ids) != 3 or len(pybamm_ids) != 4 or gym_ids & pybamm_ids:
            raise ValueError("Staged hybrid response lane identities are not separate")

    @property
    def active_lanes(self) -> tuple[str, ...]:
        "Return Gym--TORAX and PyBaMM lanes, plus BOPTEST when its conditional design is present."

        return ("gym-torax", "pybamm") + (
            ("boptest",) if self.boptest_conditional_design else ()
        )


@dataclass(frozen=True, slots=True)
class StagedHybridResponseQualificationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/staged-hybrid-response-qualification-receipt'

    receipt_id: str
    lane_id: str
    config: ObjectIdentity
    implementation: ObjectIdentity
    acquisition_index: ObjectIdentity
    maximum_episode_size_bytes: int
    maximum_episode_runtime_seconds: Decimal
    maximum_observed_aggregate_worker_rss_bytes: int
    source_identity: tuple[tuple[str, str], ...]
    recovery_verified_without_reacquisition: bool
    complete_route: bool
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.lane_id, field_name="lane_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.maximum_episode_size_bytes <= 0
            or self.maximum_episode_runtime_seconds <= 0
            or self.maximum_observed_aggregate_worker_rss_bytes <= 0
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("Staged hybrid response qualification receipt is inconsistent")


@dataclass(frozen=True, slots=True)
class StagedHybridResponseFormalGapOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/staged-hybrid-response-formal-gap-outcome'

    gap_id: str
    lane_id: str
    disposition: str
    result_class: str
    estimator_id: str | None
    independent_unit_count: int
    numerical_view_ids: tuple[str, ...]
    estimate: Decimal | None
    native_unit: str | None
    multiplicity_family_id: str
    evidence_ids: tuple[str, ...]
    interpretation: str
    promotable: bool
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.gap_id, field_name="gap_id")
        validate_stable_id(self.lane_id, field_name="lane_id")
        require_sorted_unique_strings(
            self.numerical_view_ids,
            field_name="numerical_view_ids",
        )
        require_sorted_unique_strings(self.evidence_ids, field_name="evidence_ids")


@dataclass(frozen=True, slots=True)
class StagedHybridResponseLaneFormalPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/staged-hybrid-response-lane-formal-panel'

    panel_id: str
    lane_id: str
    formal_register: ObjectIdentity
    outcomes: tuple[StagedHybridResponseFormalGapOutcome, ...]
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        validate_stable_id(self.lane_id, field_name="lane_id")
        require_sorted_unique_ids(
            self.outcomes,
            attribute="gap_id",
            field_name="outcomes",
        )
        if len(self.outcomes) != 48:
            raise ValueError("Staged hybrid response lane formal panel must contain 48 gaps")


__all__ = [
    'StagedHybridResponseAcquisitionIndex',
    'StagedHybridResponseEpisodeArtifactReceipt',
    'StagedHybridResponseFormalGapOutcome',
    'StagedHybridResponseGateRecord',
    'StagedHybridResponseImplementationManifest',
    'StagedHybridResponseLaneFormalPanel',
    'StagedHybridResponseQualificationReceipt',
    'StagedHybridResponseSourceFile',
    'StagedHybridResponseTrancheFreeze',
]
