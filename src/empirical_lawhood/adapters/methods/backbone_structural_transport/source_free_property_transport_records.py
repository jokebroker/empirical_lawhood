"""Issued, source-free records for the corrected executable-metatheory route.

The records contain scientific operands and source-free generator selections,
never an expected aggregate disposition or final report.  They grant no
operation, reveal, source, simulator, network, or actuation authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.atlas_qualification import AtlasQualificationSpec, ChartTransitionEvidence, SetValuedChartRegion
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeEvidence, CoordinateChallengeNomination, CoordinateChallengeSpec
from empirical_lawhood.planning.dataset_authority import DatasetOperationRequest
from empirical_lawhood.planning.decision_assurance import DecisionAssuranceSpec, DecisionComparisonEvidence
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceAssessment, EvidenceDependenceSpec
from empirical_lawhood.planning.metatheory import MetatheoryClaimKind, MetatheoryEvidenceCeiling, MetatheoryPredictiveLevel
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignStageRole
from empirical_lawhood.planning.metatheory_prediction import MetatheoryAdjudicationSpec, MetatheoryPredictionPackage, MetatheoryTargetOutcome
from empirical_lawhood.planning.obstruction_atlas import ObstructionMappingRegistry
from empirical_lawhood.planning.property_survival import PropertyPathCorrespondence, PropertySurvivalCellEvidence, PropertySurvivalSpec
from empirical_lawhood.planning.scientific_source_qualification import ScientificSourceGateEvidence, ScientificSourceQualificationSpec
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.datasets import DatasetCapabilityRegistry
from empirical_lawhood.runtime.source_pipelines import SourcePipelineExecutionEvidence
from empirical_lawhood.runtime.sources import SourceCapabilityManifest


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportPreparedMedium(CanonicalRecord):
    """Small source-free prepared-medium marker, never native source data."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-prepared-medium'

    medium_id: str
    physical_unit_ids: tuple[str, ...]
    source_free: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.medium_id, field_name="medium_id")
        if not self.physical_unit_ids:
            raise ValueError("prepared medium requires a physical-unit roster")
        for value in self.physical_unit_ids:
            validate_stable_id(value, field_name="physical_unit_ids")
        if not self.source_free or self.grants_authority:
            raise ValueError("prepared medium exceeds source-free conformance")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportSourcePipelineConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-source-pipeline-config'

    config_id: str
    prepared_medium: SourceFreePropertyTransportPreparedMedium
    profile: SourcePipelineProfile
    source_manifest: SourceCapabilityManifest
    capability_registry: CapabilityRegistry
    dataset_registry: DatasetCapabilityRegistry
    dataset_operation_request: DatasetOperationRequest | None
    execution_evidence: SourcePipelineExecutionEvidence | None
    decision_clock_id: str | None
    decision_time_utc: str | None
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        present = (
            self.dataset_operation_request is not None,
            self.execution_evidence is not None,
            self.decision_clock_id is not None,
            self.decision_time_utc is not None,
        )
        if any(present) and not all(present):
            raise ValueError("source-pipeline config has partial qualification inputs")
        if self.decision_clock_id is not None:
            validate_stable_id(self.decision_clock_id, field_name="decision_clock_id")
        if self.grants_authority:
            raise ValueError("source-pipeline config cannot grant authority")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportSourceQualificationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-source-qualification-config'

    config_id: str
    spec: ScientificSourceQualificationSpec
    capability_registry: CapabilityRegistry
    gate_evidence: tuple[ScientificSourceGateEvidence, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.grants_authority:
            raise ValueError("source-qualification config cannot grant authority")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportCoordinateConstructionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-coordinate-construction-config'

    config_id: str
    spec: CoordinateChallengeSpec
    capability_registry: CapabilityRegistry
    nominations: tuple[CoordinateChallengeNomination, ...]
    evidence: tuple[CoordinateChallengeEvidence, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.grants_authority:
            raise ValueError("coordinate-construction config cannot grant authority")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportCoordinateEvaluationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-coordinate-evaluation-config'

    config_id: str
    spec: CoordinateChallengeSpec
    capability_registry: CapabilityRegistry
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.grants_authority:
            raise ValueError("coordinate-evaluation config cannot grant authority")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportAtlasQualificationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-atlas-qualification-config'

    config_id: str
    spec_template: AtlasQualificationSpec
    capability_registry: CapabilityRegistry
    region: SetValuedChartRegion
    transition_evidence: tuple[ChartTransitionEvidence, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.grants_authority:
            raise ValueError("atlas-qualification config cannot grant authority")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportDecisionAssuranceConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-decision-assurance-config'

    config_id: str
    spec: DecisionAssuranceSpec
    capability_registry: CapabilityRegistry
    evidence: tuple[DecisionComparisonEvidence, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.grants_authority:
            raise ValueError("decision-assurance config cannot grant authority")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportPropertySurvivalConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-property-survival-config'

    config_id: str
    spec: PropertySurvivalSpec
    dependence_spec: EvidenceDependenceSpec
    capability_registry: CapabilityRegistry
    evidence: tuple[PropertySurvivalCellEvidence, ...]
    path_correspondences: tuple[PropertyPathCorrespondence, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.grants_authority:
            raise ValueError("property-survival config cannot grant authority")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportPredictionIssueConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-prediction-issue-config'

    config_id: str
    package_template: MetatheoryPredictionPackage
    issue_clock_id: str
    publication: ObjectIdentity
    recovery: ObjectIdentity
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("config_id", "issue_clock_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.grants_authority:
            raise ValueError("prediction-issue config cannot grant authority")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportSyntheticTargetGenerator(CanonicalRecord):
    """Opaque deterministic source-free generator selection, not an outcome."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-synthetic-target-generator'

    generator_id: str
    seed: str
    target_present: bool
    source_free: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("generator_id", "seed"):
            validate_stable_id(getattr(self, name), field_name=name)
        if not self.source_free or self.grants_authority:
            raise ValueError("target generator exceeds source-free conformance")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportTargetAcquisitionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-target-acquisition-config'

    config_id: str
    generator: SourceFreePropertyTransportSyntheticTargetGenerator
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.grants_authority:
            raise ValueError("target-acquisition config cannot grant authority")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportSealedOutcomeBundle(CanonicalRecord):
    """Target-task product kept sealed until the platform reveal barrier."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-sealed-outcome-bundle'

    bundle_id: str
    prediction_issue: ObjectIdentity
    target_id: str
    outcomes: tuple[MetatheoryTargetOutcome, ...]
    acquisition_complete: bool
    outcomes_exposed: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("bundle_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.outcomes_exposed or self.grants_authority:
            raise ValueError("sealed outcome bundle crosses its custody boundary")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportRevealAdjudicationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-reveal-adjudication-config'

    config_id: str
    spec_template: MetatheoryAdjudicationSpec
    dependence_assessment: EvidenceDependenceAssessment
    scoring_registry: CapabilityRegistry
    reveal_authority: ObjectIdentity
    revealed_publication_namespace: str
    revealed_recovery_namespace: str
    reveal_permitted: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "revealed_publication_namespace",
            "revealed_recovery_namespace",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.grants_authority:
            raise ValueError("reveal config cannot grant platform reveal authority")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportObstructionCloseoutConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-obstruction-closeout-config'

    config_id: str
    mapping_registry: ObstructionMappingRegistry
    target_id: str
    claim_kind: MetatheoryClaimKind
    predictive_level: MetatheoryPredictiveLevel
    affected_operand_ids: tuple[str, ...]
    affected_property_ids: tuple[str, ...]
    affected_coordinate_ids: tuple[str, ...]
    affected_action_ids: tuple[str, ...]
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("config_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.grants_authority:
            raise ValueError("obstruction config cannot grant authority")


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportReportConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-report-config'

    config_id: str
    fixture_scope_id: str
    plumbing_only: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("config_id", "fixture_scope_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if not self.plumbing_only or self.grants_authority:
            raise ValueError("report config exceeds contract-conformance scope")


SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE: dict[
    MetatheoryCampaignStageRole,
    type[CanonicalRecord],
] = {
    MetatheoryCampaignStageRole.SOURCE_PIPELINE: SourceFreePropertyTransportSourcePipelineConfig,
    MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION: (
        SourceFreePropertyTransportSourceQualificationConfig
    ),
    MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION: (
        SourceFreePropertyTransportCoordinateConstructionConfig
    ),
    MetatheoryCampaignStageRole.COORDINATE_EVALUATION: (SourceFreePropertyTransportCoordinateEvaluationConfig),
    MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL: (
        SourceFreePropertyTransportAtlasQualificationConfig
    ),
    MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL: (
        SourceFreePropertyTransportDecisionAssuranceConfig
    ),
    MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL: (
        SourceFreePropertyTransportPropertySurvivalConfig
    ),
    MetatheoryCampaignStageRole.PREDICTION_ISSUE: SourceFreePropertyTransportPredictionIssueConfig,
    MetatheoryCampaignStageRole.TARGET_ACQUISITION: SourceFreePropertyTransportTargetAcquisitionConfig,
    MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION: (SourceFreePropertyTransportRevealAdjudicationConfig),
    MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT: (SourceFreePropertyTransportObstructionCloseoutConfig),
    MetatheoryCampaignStageRole.REPORT: SourceFreePropertyTransportReportConfig,
}

SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPES = tuple(
    SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE[role]
    for role in MetatheoryCampaignStageRole
    if role in SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE
)


__all__ = [
    "SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE",
    "SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPES",
    'SourceFreePropertyTransportAtlasQualificationConfig',
    'SourceFreePropertyTransportCoordinateConstructionConfig',
    'SourceFreePropertyTransportCoordinateEvaluationConfig',
    'SourceFreePropertyTransportDecisionAssuranceConfig',
    'SourceFreePropertyTransportObstructionCloseoutConfig',
    'SourceFreePropertyTransportPredictionIssueConfig',
    'SourceFreePropertyTransportPreparedMedium',
    'SourceFreePropertyTransportPropertySurvivalConfig',
    'SourceFreePropertyTransportReportConfig',
    'SourceFreePropertyTransportRevealAdjudicationConfig',
    'SourceFreePropertyTransportSealedOutcomeBundle',
    'SourceFreePropertyTransportSourcePipelineConfig',
    'SourceFreePropertyTransportSourceQualificationConfig',
    'SourceFreePropertyTransportSyntheticTargetGenerator',
    'SourceFreePropertyTransportTargetAcquisitionConfig',
]
