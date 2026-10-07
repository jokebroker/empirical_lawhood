"""Outcome-blind source qualification after the real native metadata canary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .field_metadata_contracts import GymToraxFieldMetadataNativeEpisode
from .extraction_manifest import GymToraxBoundedExtractionManifest
from .field_metadata import GymToraxFieldMetadataManifest
from .metadata_barrier import GymToraxNativeMetadataCanaryDisposition, GymToraxNativeMetadataCanaryReceipt


class GymToraxNativeSourceQualificationDisposition(StrEnum):
    QUALIFIED = "QUALIFIED"
    NOT_QUALIFIED = "NOT_QUALIFIED"


@dataclass(frozen=True, slots=True)
class GymToraxNativeSourceQualificationReceipt(CanonicalRecord):
    """Mechanical source readiness only; never law or execution authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-source-qualification-receipt'

    receipt_id: str
    canary_receipt: ObjectIdentity
    canary_episode: ObjectIdentity
    extraction_manifest: ObjectIdentity
    field_metadata_manifest: ObjectIdentity
    disposition: GymToraxNativeSourceQualificationDisposition
    reason_codes: tuple[str, ...]
    source_resets_observed: int
    source_assessment_outcomes_read: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling
    grants_source_assessment_freeze_authority: bool
    grants_source_assessment_execution_authority: bool
    grants_scientific_promotion: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.canary_receipt.object_schema != GymToraxNativeMetadataCanaryReceipt.SCHEMA:
            raise ValueError("source qualification requires the typed metadata canary")
        if self.canary_episode.object_schema != GymToraxFieldMetadataNativeEpisode.SCHEMA:
            raise ValueError("source qualification requires the metadata-complete canary episode")
        if self.extraction_manifest.object_schema != GymToraxBoundedExtractionManifest.SCHEMA:
            raise ValueError("source qualification requires the surgical closure")
        if self.field_metadata_manifest.object_schema != GymToraxFieldMetadataManifest.SCHEMA:
            raise ValueError("source qualification requires the native metadata closure")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.source_resets_observed != 1:
            raise ValueError("source qualification must bind exactly one canary reset")
        if self.disposition is GymToraxNativeSourceQualificationDisposition.QUALIFIED:
            if self.reason_codes:
                raise ValueError("qualified source cannot retain barrier failures")
        elif not self.reason_codes:
            raise ValueError("unqualified source requires typed reasons")
        if self.source_assessment_outcomes_read:
            raise ValueError("source qualification cannot follow fresh Q outcome access")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("source qualification must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("source qualification must remain prospective")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("source qualification cannot promote scientific truth")
        if (
            self.grants_source_assessment_freeze_authority
            or self.grants_source_assessment_execution_authority
            or self.grants_scientific_promotion
        ):
            raise ValueError("source qualification cannot grant authority or promotion")


def qualify_gym_torax_native_source(
    *,
    episode: GymToraxFieldMetadataNativeEpisode,
    canary: GymToraxNativeMetadataCanaryReceipt,
    extraction_manifest: GymToraxBoundedExtractionManifest,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
) -> GymToraxNativeSourceQualificationReceipt:
    """Bind a passing real canary to both exact code-owned source manifests."""

    reasons: set[str] = set()
    episode_identity = ObjectIdentity.from_record(episode.episode_id, episode)
    extraction_identity = ObjectIdentity.from_record(
        extraction_manifest.manifest_id,
        extraction_manifest,
    )
    metadata_identity = ObjectIdentity.from_record(
        field_metadata_manifest.manifest_id,
        field_metadata_manifest,
    )
    if canary.episode != episode_identity:
        reasons.add("SOURCE_QUALIFICATION_CANARY_EPISODE_DIVERGENCE")
    if canary.disposition is not GymToraxNativeMetadataCanaryDisposition.PASS:
        reasons.add("SOURCE_QUALIFICATION_METADATA_CANARY_FAILED")
    if episode.extraction_manifest != extraction_identity:
        reasons.add("SOURCE_QUALIFICATION_EXTRACTION_MANIFEST_DIVERGENCE")
    if episode.field_metadata_manifest != metadata_identity:
        reasons.add("SOURCE_QUALIFICATION_FIELD_METADATA_DIVERGENCE")
    if not episode.source_reset_attempted:
        reasons.add("SOURCE_QUALIFICATION_RESET_NOT_ATTEMPTED")
    disposition = (
        GymToraxNativeSourceQualificationDisposition.NOT_QUALIFIED
        if reasons
        else GymToraxNativeSourceQualificationDisposition.QUALIFIED
    )
    return GymToraxNativeSourceQualificationReceipt(
        receipt_id='qualification.tokamak-control.native-source-readiness',
        canary_receipt=ObjectIdentity.from_record(canary.receipt_id, canary),
        canary_episode=episode_identity,
        extraction_manifest=extraction_identity,
        field_metadata_manifest=metadata_identity,
        disposition=disposition,
        reason_codes=tuple(sorted(reasons)),
        source_resets_observed=1,
        source_assessment_outcomes_read=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        grants_source_assessment_freeze_authority=False,
        grants_source_assessment_execution_authority=False,
        grants_scientific_promotion=False,
    )


__all__ = [
    'GymToraxNativeSourceQualificationDisposition',
    'GymToraxNativeSourceQualificationReceipt',
    'qualify_gym_torax_native_source',
]
