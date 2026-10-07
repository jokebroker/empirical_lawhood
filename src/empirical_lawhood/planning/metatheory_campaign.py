"""Closed, family-neutral executable-metatheory campaign profile."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.atlas_qualification import AtlasQualificationSpec
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeSpec
from empirical_lawhood.planning.decision_assurance import DecisionAssuranceSpec
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceSpec
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.metatheory import MetatheoryClaimKind
from empirical_lawhood.planning.metatheory_prediction import MetatheoryAdjudicationSpec, MetatheoryPredictionPackage
from empirical_lawhood.planning.study_authoring import CapabilitySelection
from empirical_lawhood.planning.property_survival import PropertySurvivalSpec
from empirical_lawhood.planning.scientific_source_qualification import ScientificSourceQualificationSpec
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile


class MetatheoryCampaignStageRole(StrEnum):
    SOURCE_PIPELINE = "SOURCE_PIPELINE"
    SCIENTIFIC_SOURCE_QUALIFICATION = "SCIENTIFIC_SOURCE_QUALIFICATION"
    COORDINATE_CONSTRUCTION = "COORDINATE_CONSTRUCTION"
    COORDINATE_EVALUATION = "COORDINATE_EVALUATION"
    LAW_QUALIFICATION_OPTIONAL = "LAW_QUALIFICATION_OPTIONAL"
    ATLAS_QUALIFICATION_OPTIONAL = "ATLAS_QUALIFICATION_OPTIONAL"
    DECISION_ASSURANCE_OPTIONAL = "DECISION_ASSURANCE_OPTIONAL"
    PROPERTY_SURVIVAL_OPTIONAL = "PROPERTY_SURVIVAL_OPTIONAL"
    PREDICTION_ISSUE = "PREDICTION_ISSUE"
    TARGET_ACQUISITION = "TARGET_ACQUISITION"
    REVEAL_AND_ADJUDICATION = "REVEAL_AND_ADJUDICATION"
    OBSTRUCTION_CLOSEOUT = "OBSTRUCTION_CLOSEOUT"
    REPORT = "REPORT"


@dataclass(frozen=True, slots=True)
class MetatheoryStageApplicability(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-stage-applicability'

    role: MetatheoryCampaignStageRole
    applicable: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.applicable == bool(self.reason_codes):
            raise ValueError("stage applicability and reasons disagree")


@dataclass(frozen=True, slots=True)
class MetatheoryCampaignRosters(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-campaign-rosters'

    excluded_method_power_unit_ids: tuple[str, ...]
    development_unit_ids: tuple[str, ...]
    protected_target_unit_ids: tuple[str, ...]
    reference_unit_ids: tuple[str, ...]
    reserve_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = (
            self.excluded_method_power_unit_ids,
            self.development_unit_ids,
            self.protected_target_unit_ids,
            self.reference_unit_ids,
            self.reserve_unit_ids,
        )
        for name, values in zip(
            (
                "excluded_method_power_unit_ids",
                "development_unit_ids",
                "protected_target_unit_ids",
                "reference_unit_ids",
                "reserve_unit_ids",
            ),
            groups,
            strict=True,
        ):
            require_sorted_unique_strings(values, field_name=name)
        flattened = tuple(value for group in groups for value in group)
        if len(flattened) != len(set(flattened)):
            raise ValueError("metatheory campaign roster roles overlap")
        if not self.development_unit_ids or not self.protected_target_unit_ids:
            raise ValueError("metatheory campaign requires development and protected targets")


@dataclass(frozen=True, slots=True)
class MetatheoryCampaignCapabilityBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-campaign-capability-binding'

    binding_id: str
    role: MetatheoryCampaignStageRole
    selection: CapabilitySelection
    config_id: str
    config_schema: str
    config_schema_sha256: str
    config_content_sha256: str
    config_artifact_id: str
    resource_budget: ResourceBudget
    resource_lock_ids: tuple[str, ...]
    maximum_attempts: int

    def __post_init__(self) -> None:
        for name in ("binding_id", "config_id", "config_artifact_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_schema(self.config_schema)
        validate_sha256(self.config_schema_sha256, field_name="config_schema_sha256")
        validate_sha256(self.config_content_sha256, field_name="config_content_sha256")
        require_sorted_unique_strings(self.resource_lock_ids, field_name="resource_lock_ids")
        if self.maximum_attempts < 1 or self.maximum_attempts > 8:
            raise ValueError("metatheory campaign attempt bound must be in [1, 8]")


@dataclass(frozen=True, slots=True)
class MetatheoryCampaignOwnerBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-campaign-owner-binding'

    binding_id: str
    role: MetatheoryCampaignStageRole
    owner: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")


@dataclass(frozen=True, slots=True)
class MetatheoryCampaignArtifactBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-campaign-artifact-binding'

    binding_id: str
    role: MetatheoryCampaignStageRole
    output_id: str
    payload_schema: str
    media_type: str
    filename_suffix: str
    custody_role_id: str
    maximum_bytes: int
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name in ("binding_id", "output_id", "custody_role_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_schema(self.payload_schema)
        if not self.media_type or not self.filename_suffix.startswith("."):
            raise ValueError("metatheory artifact media/suffix is invalid")
        if self.maximum_bytes < 1:
            raise ValueError("metatheory artifact requires positive bytes")
        expected = (
            VisibilityCeiling.PROSPECTIVE
            if self.outcome_access
            in {
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
            }
            else VisibilityCeiling.OUTCOME_VISIBLE
        )
        if self.visibility_ceiling is not expected:
            raise ValueError("metatheory artifact visibility differs from outcome access")


@dataclass(frozen=True, slots=True)
class MetatheoryCampaignProfile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-campaign-profile'

    profile_id: str
    profile_version: str
    claim_kind: MetatheoryClaimKind
    evidence_profile_selection: ObjectIdentity
    source_pipeline_profile: ObjectIdentity | None
    source_qualification_spec: ObjectIdentity | None
    coordinate_challenge_spec: ObjectIdentity | None
    law_qualification_parent: ObjectIdentity | None
    atlas_qualification_spec: ObjectIdentity | None
    decision_assurance_spec: ObjectIdentity | None
    dependence_spec: ObjectIdentity | None
    property_survival_spec: ObjectIdentity | None
    prediction_package: ObjectIdentity | None
    adjudication_spec: ObjectIdentity | None
    obstruction_profile: ObjectIdentity
    rosters: MetatheoryCampaignRosters
    capabilities: tuple[MetatheoryCampaignCapabilityBinding, ...]
    owners: tuple[MetatheoryCampaignOwnerBinding, ...]
    artifacts: tuple[MetatheoryCampaignArtifactBinding, ...]
    topology_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_semantic_version(self.profile_version)
        if self.evidence_profile_selection.object_schema != EvidenceProfileSelection.SCHEMA:
            raise ValueError("metatheory campaign requires an evidence-profile selection")
        expected_schemas = (
            (self.source_pipeline_profile, SourcePipelineProfile.SCHEMA),
            (self.source_qualification_spec, ScientificSourceQualificationSpec.SCHEMA),
            (self.coordinate_challenge_spec, CoordinateChallengeSpec.SCHEMA),
            (self.atlas_qualification_spec, AtlasQualificationSpec.SCHEMA),
            (self.decision_assurance_spec, DecisionAssuranceSpec.SCHEMA),
            (self.dependence_spec, EvidenceDependenceSpec.SCHEMA),
            (self.property_survival_spec, PropertySurvivalSpec.SCHEMA),
            (self.prediction_package, MetatheoryPredictionPackage.SCHEMA),
            (self.adjudication_spec, MetatheoryAdjudicationSpec.SCHEMA),
        )
        if any(
            value is not None and value.object_schema != schema
            for value, schema in expected_schemas
        ):
            raise ValueError("metatheory campaign priority spec schema differs")
        roles = set(MetatheoryCampaignStageRole)
        for name, values in (
            ("capabilities", self.capabilities),
            ("owners", self.owners),
            ("artifacts", self.artifacts),
        ):
            require_sorted_unique_ids(values, attribute="binding_id", field_name=name)
            if {value.role for value in values} != roles or len(values) != len(roles):
                raise ValueError(f"metatheory campaign {name} role roster is incomplete")
        validate_stable_id(self.topology_id, field_name="topology_id")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("metatheory campaign authoring must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("metatheory campaign authoring must remain prospective")
        self._validate_claim_requirements()

    def _validate_claim_requirements(self) -> None:
        if (self.source_pipeline_profile is None) != (self.source_qualification_spec is None):
            raise ValueError("source pipeline and scientific qualification enter together")
        if self.claim_kind is MetatheoryClaimKind.SOURCE_PREPARATION_QUALIFICATION:
            if self.source_qualification_spec is None:
                raise ValueError("source/preparation claim lacks its source contracts")
        elif self.claim_kind in {
            MetatheoryClaimKind.COORDINATE_CONSTRUCT_VALIDITY,
            MetatheoryClaimKind.RECEIVER_HISTORY_CLOSURE,
        }:
            if self.coordinate_challenge_spec is None:
                raise ValueError("coordinate claim lacks coordinate challenge")
        elif self.claim_kind is MetatheoryClaimKind.CHART_QUALIFICATION:
            if self.coordinate_challenge_spec is None or self.atlas_qualification_spec is None:
                raise ValueError("chart claim lacks coordinate/atlas qualification")
        elif self.claim_kind is MetatheoryClaimKind.DECISION_ASSURANCE:
            if self.decision_assurance_spec is None:
                raise ValueError("decision claim lacks decision assurance")
        elif self.claim_kind is MetatheoryClaimKind.PROPERTY_TRANSPORT:
            if self.dependence_spec is None or self.property_survival_spec is None:
                raise ValueError("property claim lacks dependence/property specs")
        elif self.claim_kind is MetatheoryClaimKind.STRUCTURAL_RECURRENCE:
            if self.prediction_package is None or self.adjudication_spec is None:
                raise ValueError("structural recurrence lacks prediction/adjudication")


__all__ = [
    'MetatheoryCampaignArtifactBinding',
    'MetatheoryCampaignCapabilityBinding',
    'MetatheoryCampaignOwnerBinding',
    'MetatheoryCampaignProfile',
    'MetatheoryCampaignRosters',
    'MetatheoryCampaignStageRole',
    'MetatheoryStageApplicability',
]
