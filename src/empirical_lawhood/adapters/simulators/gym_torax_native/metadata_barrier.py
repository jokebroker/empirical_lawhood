"""Outcome-blind native metadata canary adjudication for fresh Gym-TORAX evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .diagnostic_contracts import GymToraxDeliveryDisposition, GymToraxNumericalDisposition, GymToraxObservationDisposition, GymToraxSourceDisposition
from .field_metadata_contracts import GymToraxFieldMetadataNativeEpisode
from .field_metadata import GymToraxFieldMetadataManifest


class GymToraxNativeMetadataCanaryDisposition(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class GymToraxNativeMetadataCanaryReceipt(CanonicalRecord):
    """Nonpromotable receipt required before a fresh 96-cell Q freeze."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-metadata-canary-receipt'

    receipt_id: str
    episode: ObjectIdentity
    field_metadata_manifest: ObjectIdentity
    expected_source_field_count: int
    observed_source_field_count: int
    disposition: GymToraxNativeMetadataCanaryDisposition
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    evidence_ceiling: EvidenceCeiling
    grants_source_assessment_freeze_authority: bool
    grants_source_assessment_execution_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.episode.object_schema != GymToraxFieldMetadataNativeEpisode.SCHEMA:
            raise ValueError("metadata canary receipt requires a metadata-complete native episode")
        if self.field_metadata_manifest.object_schema != GymToraxFieldMetadataManifest.SCHEMA:
            raise ValueError("metadata canary receipt requires the frozen metadata manifest")
        if self.expected_source_field_count <= 0:
            raise ValueError("metadata canary expected field count must be positive")
        if self.observed_source_field_count < 0:
            raise ValueError("metadata canary observed field count must be nonnegative")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is GymToraxNativeMetadataCanaryDisposition.PASS:
            if (
                self.reason_codes
                or self.observed_source_field_count != self.expected_source_field_count
            ):
                raise ValueError("passing metadata canary requires the exact complete closure")
        elif not self.reason_codes:
            raise ValueError("failed metadata canary requires typed reasons")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("metadata canary must remain outcome-blind")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("metadata canary cannot be promotable evidence")
        if self.grants_source_assessment_freeze_authority or self.grants_source_assessment_execution_authority:
            raise ValueError("metadata canary cannot grant freeze or execution authority")


def adjudicate_gym_torax_native_metadata_canary(
    *,
    episode: GymToraxFieldMetadataNativeEpisode,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
) -> GymToraxNativeMetadataCanaryReceipt:
    """Check metadata/transport completeness without reading scientific values."""

    reasons: set[str] = set()
    expected_manifest_identity = ObjectIdentity.from_record(
        field_metadata_manifest.manifest_id,
        field_metadata_manifest,
    )
    if episode.field_metadata_manifest != expected_manifest_identity:
        reasons.add("CANARY_FIELD_METADATA_MANIFEST_DIVERGENCE")
    if episode.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
        reasons.add("CANARY_OUTCOME_ACCESS_NOT_BLIND")
    if episode.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
        reasons.add("CANARY_EVIDENCE_CEILING_NOT_NONPROMOTABLE")
    if episode.source_disposition is not GymToraxSourceDisposition.AVAILABLE:
        reasons.add("CANARY_SOURCE_UNAVAILABLE")
    if episode.delivery_disposition is not GymToraxDeliveryDisposition.COMPLETE:
        reasons.add("CANARY_DELIVERY_INCOMPLETE")
    if episode.numerical_disposition is not GymToraxNumericalDisposition.VALID:
        reasons.add("CANARY_NUMERICS_INVALID")
    if episode.observation_disposition is not GymToraxObservationDisposition.COMPLETE:
        reasons.add("CANARY_OBSERVATION_INCOMPLETE")
    if episode.missing_required_state_clocks or episode.state_clocks != tuple(range(121)):
        reasons.add("CANARY_STATE_CLOCKS_INCOMPLETE")
    if not episode.source_reset_attempted:
        reasons.add("CANARY_SOURCE_RESET_NOT_ATTEMPTED")

    expected = field_metadata_manifest.by_key()
    observed = {
        (block.category.removeprefix("source-"), block.native_field_id): block
        for block in episode.blocks
        if block.category.startswith("source-")
    }
    category_map = {"profile": "profiles", "scalar": "scalars", "numerics": "numerics"}
    normalized_observed = {
        (category_map.get(category, category), field_id): block
        for (category, field_id), block in observed.items()
    }
    if set(normalized_observed) != set(expected):
        reasons.add("CANARY_SOURCE_FIELD_ROSTER_DIVERGENCE")
    for key in sorted(set(normalized_observed) & set(expected)):
        block = normalized_observed[key]
        metadata = expected[key]
        if block.field_metadata_id != metadata.field_metadata_id:
            reasons.add("CANARY_FIELD_METADATA_ID_DIVERGENCE")
        if block.native_unit != metadata.native_unit:
            reasons.add("CANARY_NATIVE_UNIT_DIVERGENCE")
        if block.native_frame_id != metadata.native_frame_id:
            reasons.add("CANARY_NATIVE_FRAME_DIVERGENCE")
        if block.dimension_ids != ("state-clock", *metadata.native_dimension_ids):
            reasons.add("CANARY_NATIVE_DIMENSION_DIVERGENCE")
    if any(
        reason.startswith(("SOURCE_FIELD_", "SOURCE_COORDINATE_", "REQUIRED_FIELD_"))
        for reason in episode.reason_codes
    ):
        reasons.add("CANARY_SOURCE_METADATA_REASON_PRESENT")

    disposition = (
        GymToraxNativeMetadataCanaryDisposition.FAIL
        if reasons
        else GymToraxNativeMetadataCanaryDisposition.PASS
    )
    return GymToraxNativeMetadataCanaryReceipt(
        receipt_id=f"receipt.{episode.episode_id.removeprefix('episode.')}.metadata-canary",
        episode=ObjectIdentity.from_record(episode.episode_id, episode),
        field_metadata_manifest=expected_manifest_identity,
        expected_source_field_count=len(expected),
        observed_source_field_count=len(normalized_observed),
        disposition=disposition,
        reason_codes=tuple(sorted(reasons)),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        grants_source_assessment_freeze_authority=False,
        grants_source_assessment_execution_authority=False,
    )


__all__ = [
    'GymToraxNativeMetadataCanaryDisposition',
    'GymToraxNativeMetadataCanaryReceipt',
    'adjudicate_gym_torax_native_metadata_canary',
]
