"""Outcome-blind FAIR-MAST metadata qualification contracts.

The source service may inspect source/action metadata needed to construct a
candidate roster, but it cannot inspect or carry a receiver endpoint.  Raw
Parquet bytes remain in external custody; these compact records bind their
physical and canonical table identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar
from urllib.parse import urlsplit

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)


FAIR_MAST_LEVEL2_INGESTION_COMMIT = "ab435c799d892956fb042d55391f7d1be0c950e6"
FAIR_MAST_LICENSE_ID = "cc-by-sa-4.0"
FAIR_MAST_METADATA_ORIGIN = "https://mastapp.site"
FAIR_MAST_PAYLOAD_ORIGIN = "https://s3.echo.stfc.ac.uk"
FAIR_MAST_MAPPING_ORIGIN = "https://raw.githubusercontent.com"
FAIR_MAST_LEVEL2_MAPPING_URL = (
    f"{FAIR_MAST_MAPPING_ORIGIN}/ukaea/fair-mast-ingestion/"
    f"{FAIR_MAST_LEVEL2_INGESTION_COMMIT}/mappings/level2/mast.yml"
)
FAIR_MAST_SOURCE_INSPECTION_ARRAY_PATHS = (
    "gas_injection/time",
    "gas_injection/valve_target_voltage",
    "gas_injection/valve_voltage",
    "pf_active/coil_current",
    "pf_active/coil_voltage",
    "pf_active/time",
    "pulse_schedule/i_plasma",
    "pulse_schedule/n_e_line",
    "pulse_schedule/time",
    "pulse_schedule/z_ref",
    "summary/power_nbi",
    "summary/time",
    "thomson_scattering/t_e_core",
    "thomson_scattering/time",
)
FAIR_MAST_SOURCE_INSPECTION_SHOT_IDS = (24809, 27582, 29643)
FAIR_MAST_REQUIRED_GROUPS = (
    "equilibrium",
    "gas_injection",
    "magnetics",
    "pf_active",
    "pulse_schedule",
    "summary",
    "thomson_scattering",
)


class FairMastSourceContractDecision(StrEnum):
    "Outcome-blind disposition of the archive/source interface."

    SOURCE_CONTRACT_EXECUTABLE = "SOURCE_CONTRACT_EXECUTABLE"
    SCIENTIFIC_AMENDMENT_REQUIRED = "SCIENTIFIC_AMENDMENT_REQUIRED"
    ARCHIVE_SOURCE_NOT_QUALIFIED = "ARCHIVE_SOURCE_NOT_QUALIFIED"


def _validate_exact_https_url(value: str, *, origin: str, field_name: str) -> None:
    validate_nonempty(value, field_name=field_name)
    parts = urlsplit(value)
    expected = urlsplit(origin)
    if (
        parts.scheme != "https"
        or parts.netloc != expected.netloc
        or parts.username is not None
        or parts.password is not None
        or parts.fragment
    ):
        raise ValueError(f"{field_name} lies outside {origin}")


@dataclass(frozen=True, slots=True)
class FairMastMetadataMember(CanonicalRecord):
    """One bounded upstream metadata object and its held-source comparison."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-metadata-member'

    member_id: str
    request_url: str
    final_url: str
    maximum_bytes: int
    size_bytes: int
    physical_sha256: str
    canonical_table_sha256: str
    historical_physical_sha256: str
    historical_canonical_table_sha256: str
    row_count: int
    arrow_field_count: int
    arrow_schema_sha256: str
    primary_key_fields: tuple[str, ...]
    external_relative_locator: str
    semantic_match_to_historical_snapshot: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        _validate_exact_https_url(
            self.request_url,
            origin=FAIR_MAST_METADATA_ORIGIN,
            field_name="request_url",
        )
        _validate_exact_https_url(
            self.final_url,
            origin=FAIR_MAST_METADATA_ORIGIN,
            field_name="final_url",
        )
        for name in (
            "physical_sha256",
            "canonical_table_sha256",
            "historical_physical_sha256",
            "historical_canonical_table_sha256",
            "arrow_schema_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.primary_key_fields,
            field_name="primary_key_fields",
            allow_empty=False,
        )
        validate_relative_locator(self.external_relative_locator)
        if (
            self.maximum_bytes <= 0
            or not 0 < self.size_bytes <= self.maximum_bytes
            or self.row_count <= 0
            or self.arrow_field_count <= 0
            or not self.semantic_match_to_historical_snapshot
            or self.canonical_table_sha256 != self.historical_canonical_table_sha256
        ):
            raise ValueError("FAIR-MAST metadata member is not semantically qualified")


@dataclass(frozen=True, slots=True)
class FairMastCampaignMetadataAvailability(CanonicalRecord):
    """Metadata-only candidate capacity; not an eligible-event count."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-campaign-metadata-availability'

    availability_id: str
    campaign_id: str
    total_shot_count: int
    unexposed_shot_count: int
    nbi_metadata_complete_count: int
    required_groups_complete_count: int
    outcome_blind_candidate_count: int
    preferred_total_needed_per_campaign: int
    minimum_total_needed_per_campaign: int
    preferred_metadata_capacity: bool
    minimum_metadata_capacity: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.availability_id, field_name="availability_id")
        if self.campaign_id not in {"M7", "M8", "M9"}:
            raise ValueError("metadata availability names an out-of-scope campaign")
        counts = (
            self.total_shot_count,
            self.unexposed_shot_count,
            self.nbi_metadata_complete_count,
            self.required_groups_complete_count,
            self.outcome_blind_candidate_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("metadata availability counts must be nonnegative")
        if (
            self.preferred_total_needed_per_campaign != 130
            or self.minimum_total_needed_per_campaign != 110
            or self.preferred_metadata_capacity != (self.outcome_blind_candidate_count >= 130)
            or self.minimum_metadata_capacity != (self.outcome_blind_candidate_count >= 110)
        ):
            raise ValueError("metadata capacity does not implement development30+construct20+protected80-or60 per campaign")


@dataclass(frozen=True, slots=True)
class FairMastMetadataCandidate(CanonicalRecord):
    """One outcome-blind, metadata-qualified shot candidate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-metadata-candidate'

    candidate_id: str
    shot_id: int
    campaign_id: str
    shot_endpoint_url: str
    nbi_start_time_s: Decimal
    nbi_end_time_s: Decimal
    nbi_max_power_w: Decimal
    required_source_group_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        if self.candidate_id != f"fair-mast-shot-{self.shot_id}" or self.shot_id <= 0:
            raise ValueError("FAIR-MAST candidate does not preserve shot identity")
        if self.campaign_id not in {"M7", "M8", "M9"}:
            raise ValueError("FAIR-MAST candidate lies outside M7--M9")
        _validate_exact_https_url(
            self.shot_endpoint_url,
            origin=FAIR_MAST_PAYLOAD_ORIGIN,
            field_name="shot_endpoint_url",
        )
        for name in ("nbi_start_time_s", "nbi_end_time_s", "nbi_max_power_w"):
            validate_decimal(getattr(self, name), field_name=name)
        if (
            self.nbi_start_time_s < 0
            or self.nbi_end_time_s - self.nbi_start_time_s <= Decimal("0.100")
            or self.nbi_max_power_w <= 0
            or self.required_source_group_ids != FAIR_MAST_REQUIRED_GROUPS
        ):
            raise ValueError("FAIR-MAST candidate fails the metadata-only predicate")


@dataclass(frozen=True, slots=True)
class FairMastMetadataCandidateIndex(CanonicalRecord):
    """Finite pre-event candidate index with no receiver outcome fields."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-metadata-candidate-index'

    index_id: str
    candidates: tuple[FairMastMetadataCandidate, ...]
    metadata_predicate_ids: tuple[str, ...]
    protected_ineligible_shot_ids_sha256: str
    receiver_field_ids_present: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.index_id, field_name="index_id")
        require_sorted_unique_ids(
            self.candidates,
            attribute="candidate_id",
            field_name="candidates",
        )
        require_sorted_unique_strings(
            self.metadata_predicate_ids,
            field_name="metadata_predicate_ids",
            allow_empty=False,
        )
        validate_sha256(
            self.protected_ineligible_shot_ids_sha256,
            field_name="protected_ineligible_shot_ids_sha256",
        )
        if (
            self.receiver_field_ids_present
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("metadata candidate index exposes a receiver outcome")


@dataclass(frozen=True, slots=True)
class FairMastMetadataQualification(CanonicalRecord):
    """F2 source qualification sufficient for candidate roster construction."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-metadata-qualification'

    qualification_id: str
    source_bundle: ObjectIdentity
    exposure_ledger: ObjectIdentity
    metadata_members: tuple[FairMastMetadataMember, ...]
    candidate_index: ObjectIdentity
    campaign_availability: tuple[FairMastCampaignMetadataAvailability, ...]
    level2_ingestion_commit: str
    license_id: str
    observed_source_group_ids: tuple[str, ...]
    required_source_group_ids: tuple[str, ...]
    observed_payload_https_origins: tuple[str, ...]
    metadata_semantics_unchanged: bool
    source_qualified_for_outcome_blind_roster_construction: bool
    payload_acquisition_requires_separate_origin_authority: bool
    receiver_outcome_values_projected: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        require_sorted_unique_ids(
            self.metadata_members,
            attribute="member_id",
            field_name="metadata_members",
        )
        require_sorted_unique_ids(
            self.campaign_availability,
            attribute="availability_id",
            field_name="campaign_availability",
        )
        for name in ("observed_source_group_ids", "required_source_group_ids"):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        require_sorted_unique_strings(
            self.observed_payload_https_origins,
            field_name="observed_payload_https_origins",
            allow_empty=False,
        )
        if tuple(value.campaign_id for value in self.campaign_availability) != (
            "M7",
            "M8",
            "M9",
        ):
            raise ValueError("FAIR-MAST metadata campaign roster differs")
        if (
            self.level2_ingestion_commit != FAIR_MAST_LEVEL2_INGESTION_COMMIT
            or self.license_id != FAIR_MAST_LICENSE_ID
        ):
            raise ValueError("FAIR-MAST ingestion or licence identity differs")
        if self.required_source_group_ids != FAIR_MAST_REQUIRED_GROUPS or not set(
            FAIR_MAST_REQUIRED_GROUPS
        ).issubset(self.observed_source_group_ids):
            raise ValueError("FAIR-MAST required source groups are incomplete")
        if self.observed_payload_https_origins != (FAIR_MAST_PAYLOAD_ORIGIN,):
            raise ValueError("FAIR-MAST payload origin roster differs")
        if not self.metadata_semantics_unchanged:
            raise ValueError("FAIR-MAST metadata semantics drifted")
        if not self.source_qualified_for_outcome_blind_roster_construction:
            raise ValueError("FAIR-MAST metadata was not qualified for roster construction")
        if not all(value.minimum_metadata_capacity for value in self.campaign_availability):
            raise ValueError("FAIR-MAST minimum metadata capacity is unavailable")
        if not self.payload_acquisition_requires_separate_origin_authority:
            raise ValueError("FAIR-MAST payload origin authority boundary was omitted")
        if self.receiver_outcome_values_projected:
            raise ValueError("FAIR-MAST metadata qualification projected receiver outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("FAIR-MAST metadata qualification is not outcome-blind")


@dataclass(frozen=True, slots=True)
class FairMastSourceInspectionScope(CanonicalRecord):
    "Finite source-inspection authority subject for payload-schema inspection only."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-source-inspection-scope'

    scope_id: str
    predecessor_source_bundle: ObjectIdentity
    allowed_https_origins: tuple[str, ...]
    level2_mapping_url: str
    representative_shot_ids: tuple[int, ...]
    zarr_array_paths: tuple[str, ...]
    permits_zarr_metadata: bool
    permits_value_chunks: bool
    permits_receiver_values: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.scope_id, field_name="scope_id")
        require_sorted_unique_strings(
            self.allowed_https_origins,
            field_name="allowed_https_origins",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.zarr_array_paths,
            field_name="zarr_array_paths",
            allow_empty=False,
        )
        _validate_exact_https_url(
            self.level2_mapping_url,
            origin=FAIR_MAST_MAPPING_ORIGIN,
            field_name="level2_mapping_url",
        )
        if (
            self.allowed_https_origins != (FAIR_MAST_MAPPING_ORIGIN, FAIR_MAST_PAYLOAD_ORIGIN)
            or self.level2_mapping_url != FAIR_MAST_LEVEL2_MAPPING_URL
            or self.representative_shot_ids != FAIR_MAST_SOURCE_INSPECTION_SHOT_IDS
            or self.zarr_array_paths != FAIR_MAST_SOURCE_INSPECTION_ARRAY_PATHS
            or not self.permits_zarr_metadata
            or self.permits_value_chunks
            or self.permits_receiver_values
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("FAIR-MAST source inspection exceeds its finite metadata scope")


@dataclass(frozen=True, slots=True)
class FairMastCampaignSourceContractAudit(CanonicalRecord):
    """Campaign-local source flags and timing contact on contaminated shots."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-campaign-source-contract-audit'

    campaign_audit_id: str
    campaign_id: str
    metadata_candidate_count: int
    pellets_true_count: int
    pellets_false_count: int
    pellets_null_count: int
    rmp_true_count: int
    rmp_false_count: int
    rmp_null_count: int
    contaminated_shot_count: int
    primary_unique_active_event_count: int
    primary_multiple_equal_event_count: int
    primary_hold_opportunity_count: int
    primary_hold_qualified_count: int
    primary_endpoint_contact_count: int
    primary_temporal_placebo_contact_count: int
    primary_joint_timing_contact_count: int
    half_cadence_joint_timing_contact_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.campaign_audit_id, field_name="campaign_audit_id")
        if self.campaign_id not in {"M7", "M8", "M9"}:
            raise ValueError("source-contract audit names an out-of-scope campaign")
        counts = (
            self.metadata_candidate_count,
            self.pellets_true_count,
            self.pellets_false_count,
            self.pellets_null_count,
            self.rmp_true_count,
            self.rmp_false_count,
            self.rmp_null_count,
            self.contaminated_shot_count,
            self.primary_unique_active_event_count,
            self.primary_multiple_equal_event_count,
            self.primary_hold_opportunity_count,
            self.primary_hold_qualified_count,
            self.primary_endpoint_contact_count,
            self.primary_temporal_placebo_contact_count,
            self.primary_joint_timing_contact_count,
            self.half_cadence_joint_timing_contact_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("source-contract audit counts must be nonnegative")
        if (
            self.pellets_true_count + self.pellets_false_count + self.pellets_null_count
            != self.metadata_candidate_count
            or self.rmp_true_count + self.rmp_false_count + self.rmp_null_count
            != self.metadata_candidate_count
        ):
            raise ValueError("source flag counts do not cover the metadata candidates")
        timing_counts = (
            self.primary_endpoint_contact_count,
            self.primary_temporal_placebo_contact_count,
            self.primary_joint_timing_contact_count,
            self.half_cadence_joint_timing_contact_count,
        )
        if (
            self.primary_unique_active_event_count
            + self.primary_multiple_equal_event_count
            + self.primary_hold_opportunity_count
            != self.contaminated_shot_count
            or self.primary_hold_qualified_count > self.primary_hold_opportunity_count
        ):
            raise ValueError("Event dispositions do not cover the contaminated shots")
        if any(value > self.primary_unique_active_event_count for value in timing_counts):
            raise ValueError("timing contact exceeds the unique-active-event denominator")
        if self.primary_joint_timing_contact_count > min(
            self.primary_endpoint_contact_count,
            self.primary_temporal_placebo_contact_count,
        ):
            raise ValueError("joint timing contact exceeds a component contact count")


@dataclass(frozen=True, slots=True)
class FairMastArraySourceContractAudit(CanonicalRecord):
    """Finite Zarr-array availability and declaration audit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-array-source-contract-audit'

    array_audit_id: str
    array_path: str
    source_role_id: str
    audited_shot_count: int
    present_count: int
    observed_unit_ids: tuple[str, ...]
    observed_leading_axis_counts: tuple[int, ...]
    quantization_attribute_present_count: int
    material_floor_attribute_present_count: int
    value_chunks_acquired: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.array_audit_id, field_name="array_audit_id")
        validate_nonempty(self.array_path, field_name="array_path")
        validate_stable_id(self.source_role_id, field_name="source_role_id")
        require_sorted_unique_strings(
            self.observed_unit_ids,
            field_name="observed_unit_ids",
            allow_empty=self.present_count == 0,
        )
        if tuple(sorted(set(self.observed_leading_axis_counts))) != (
            self.observed_leading_axis_counts
        ) or any(value <= 0 for value in self.observed_leading_axis_counts):
            raise ValueError("observed leading-axis counts must be sorted positive values")
        if (
            self.audited_shot_count <= 0
            or not 0 <= self.present_count <= self.audited_shot_count
            or not 0 <= self.quantization_attribute_present_count <= self.present_count
            or not 0 <= self.material_floor_attribute_present_count <= self.present_count
        ):
            raise ValueError("array audit availability counts are inconsistent")
        if self.value_chunks_acquired:
            raise ValueError("source-contract audit may not acquire array value chunks")


@dataclass(frozen=True, slots=True)
class FairMastSourceContractAuditSummary(CanonicalRecord):
    "Compact, outcome-blind decision on whether the archive source law can run."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/fair-mast-source-contract-audit-summary'

    audit_id: str
    source_inspection_scope: ObjectIdentity
    source_acquisition_authority: ObjectIdentity
    custody_publication_authority: ObjectIdentity
    archive_law_spec: ObjectIdentity
    metadata_qualification: ObjectIdentity
    candidate_index: ObjectIdentity
    level2_ingestion_commit: str
    level2_mapping_url: str
    level2_mapping_sha256: str
    stage0_selection_sha256: str
    stage0_observability_sha256: str
    contaminated_shot_ids_sha256: str
    detail_external_relative_locator: str
    detail_sha256: str
    campaign_audits: tuple[FairMastCampaignSourceContractAudit, ...]
    array_audits: tuple[FairMastArraySourceContractAudit, ...]
    actuator_inventory_bound_to_source_contract: bool
    source_declared_quantization_complete: bool
    source_declared_material_floors_complete: bool
    rmp_null_semantics_resolved: bool
    pellet_timing_surface_available: bool
    all_audited_arrays_present: bool
    new_source_value_chunks_acquired: bool
    new_receiver_value_chunks_acquired: bool
    authority_scope_satisfied: bool
    decision: FairMastSourceContractDecision
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        if self.level2_ingestion_commit != FAIR_MAST_LEVEL2_INGESTION_COMMIT:
            raise ValueError("source-contract audit uses a different ingestion commit")
        _validate_exact_https_url(
            self.level2_mapping_url,
            origin="https://raw.githubusercontent.com",
            field_name="level2_mapping_url",
        )
        for name in (
            "level2_mapping_sha256",
            "stage0_selection_sha256",
            "stage0_observability_sha256",
            "contaminated_shot_ids_sha256",
            "detail_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        validate_relative_locator(self.detail_external_relative_locator)
        require_sorted_unique_ids(
            self.campaign_audits,
            attribute="campaign_audit_id",
            field_name="campaign_audits",
        )
        require_sorted_unique_ids(
            self.array_audits,
            attribute="array_audit_id",
            field_name="array_audits",
        )
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=self.decision
            is FairMastSourceContractDecision.SOURCE_CONTRACT_EXECUTABLE,
        )
        if tuple(value.campaign_id for value in self.campaign_audits) != (
            "M7",
            "M8",
            "M9",
        ):
            raise ValueError("source-contract campaign audit roster differs")
        if self.new_source_value_chunks_acquired or self.new_receiver_value_chunks_acquired:
            raise ValueError("source-contract audit crossed its metadata-only access boundary")
        if not self.authority_scope_satisfied:
            raise ValueError("source-contract audit lacks exact acquisition/custody authority")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("source-contract audit must remain outcome-blind")
        source_ready = all(
            (
                self.actuator_inventory_bound_to_source_contract,
                self.source_declared_quantization_complete,
                self.source_declared_material_floors_complete,
                self.rmp_null_semantics_resolved,
                self.pellet_timing_surface_available,
                self.all_audited_arrays_present,
            )
        )
        full_primary_timing_contact = all(
            value.primary_unique_active_event_count > 0
            and value.primary_joint_timing_contact_count == value.primary_unique_active_event_count
            for value in self.campaign_audits
        )
        if self.decision is FairMastSourceContractDecision.SOURCE_CONTRACT_EXECUTABLE:
            if not source_ready or not full_primary_timing_contact or self.reason_codes:
                raise ValueError("Executable decision lacks complete source/timing evidence")
        elif self.decision is FairMastSourceContractDecision.SCIENTIFIC_AMENDMENT_REQUIRED:
            if source_ready and full_primary_timing_contact:
                raise ValueError("Amendment decision has no material source/timing obstruction")
        elif self.all_audited_arrays_present:
            raise ValueError("archive source was rejected despite complete audited array presence")


__all__ = [
    "FAIR_MAST_LEVEL2_INGESTION_COMMIT",
    "FAIR_MAST_LICENSE_ID",
    "FAIR_MAST_LEVEL2_MAPPING_URL",
    "FAIR_MAST_MAPPING_ORIGIN",
    "FAIR_MAST_METADATA_ORIGIN",
    "FAIR_MAST_PAYLOAD_ORIGIN",
    "FAIR_MAST_REQUIRED_GROUPS",
    "FAIR_MAST_SOURCE_INSPECTION_ARRAY_PATHS",
    "FAIR_MAST_SOURCE_INSPECTION_SHOT_IDS",
    'FairMastArraySourceContractAudit',
    'FairMastCampaignMetadataAvailability',
    'FairMastCampaignSourceContractAudit',
    'FairMastMetadataCandidateIndex',
    'FairMastMetadataCandidate',
    'FairMastMetadataMember',
    'FairMastMetadataQualification',
    'FairMastSourceContractAuditSummary',
    'FairMastSourceContractDecision',
    'FairMastSourceInspectionScope',
]
