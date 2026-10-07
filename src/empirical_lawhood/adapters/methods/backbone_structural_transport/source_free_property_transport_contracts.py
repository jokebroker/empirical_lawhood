"""Frozen corrected capability and scientific-owner matrix for EMPC-R1."""

from __future__ import annotations

from hashlib import sha256

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.atlas_qualification import AtlasQualificationResult, AtlasQualificationSpec, ChartTransitionEvidence, SetValuedChartRegion
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeEvidence, CoordinateChallengeNomination, CoordinateChallengeResult, CoordinateChallengeSpec, MetatheoryCoordinateConstructionProduct
from empirical_lawhood.planning.decision_assurance import DecisionAssuranceResult, DecisionAssuranceSpec, DecisionComparisonEvidence
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceAssessment, EvidenceDependenceSpec
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignProfile, MetatheoryCampaignStageRole
from empirical_lawhood.planning.metatheory_prediction import CustodyBoundMetatheoryAdjudicationResult, MetatheoryAdjudicationSpec, MetatheoryPredictionIssueReceipt, MetatheoryPredictionPackage, MetatheoryRevealAuthorizationBinding, MetatheorySealedTarget, MetatheoryTargetAcquisitionIndex, MetatheoryTargetOutcome, RevealedTargetOutcomeBinding
from empirical_lawhood.planning.obstruction_atlas import ObstructionAtlasSpec, ObstructionAtlas, ObstructionSourceBinding
from empirical_lawhood.planning.property_survival import PropertyPathCorrespondence, FacePairedPropertySurvivalAssessment, PropertySurvivalCellEvidence, PropertySurvivalSpec
from empirical_lawhood.planning.scientific_source_qualification import ScientificSourceGateEvidence, ScientificSourceQualificationResult, ScientificSourceQualificationSpec
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile, SourcePipelineQualificationReceipt
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.metatheory_campaigns import MetatheoryStageOwnerContract
from empirical_lawhood.runtime.metatheory_reporting import MetatheoryReportInput
from empirical_lawhood.runtime.source_pipelines import SourcePipelineExecutionEvidence

from .source_free_property_transport_records import SourceFreePropertyTransportPreparedMedium, SourceFreePropertyTransportSealedOutcomeBundle


SOURCE_FREE_PROPERTY_TRANSPORT_CAPABILITY_VERSION = "1.0.0"
SOURCE_FREE_PROPERTY_TRANSPORT_RESOURCE_BUDGET = ResourceBudget(
    cpu_cores=1,
    memory_bytes=2 * 1024 * 1024 * 1024,
    gpu_devices=0,
    wall_time_seconds=60,
    source_scan_bytes=1_000_000,
    output_bytes=1_000_000,
)
_MEDIA_SCHEMA = 'empirical-lawhood/runtime/metatheory-report-input'
_BASE_PERMISSIONS = (
    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
)


def _identity(owner_id: str) -> ObjectIdentity:
    schema = 'empirical-lawhood/runtime/scientific-owner'
    return ObjectIdentity(
        object_id=owner_id,
        object_schema=schema,
        object_version="1.0.0",
        object_fingerprint=sha256(f"{owner_id}:{schema}".encode()).hexdigest(),
    )


SOURCE_FREE_PROPERTY_TRANSPORT_EVIDENCE_DEPENDENCE_OWNER = _identity(
    "owner.runtime.evidence-dependence-service"
)

_ROLE_MATRIX: dict[
    MetatheoryCampaignStageRole,
    tuple[
        str,
        str,
        tuple[str, ...],
        str,
        str,
        ObjectIdentity,
        OutcomeAccess,
        CapabilityKind,
    ],
] = {
    MetatheoryCampaignStageRole.SOURCE_PIPELINE: (
        "executable-source-free-property-transport.source-pipeline",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-source-pipeline-config',
        (
            SourceFreePropertyTransportPreparedMedium.SCHEMA,
            SourcePipelineExecutionEvidence.SCHEMA,
            SourcePipelineProfile.SCHEMA,
        ),
        SourcePipelineQualificationReceipt.SCHEMA,
        "21787c7629fd4d6a5ae8e42a268e7d38c17dfec36793cae3c7b2da538a36831e",
        _identity("owner.runtime.compile-source-pipeline"),
        OutcomeAccess.OUTCOME_BLIND,
        CapabilityKind.SOURCE,
    ),
    MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION: (
        "executable-source-free-property-transport.scientific-source-qualification",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-source-qualification-config',
        (
            ScientificSourceGateEvidence.SCHEMA,
            ScientificSourceQualificationSpec.SCHEMA,
            SourcePipelineQualificationReceipt.SCHEMA,
        ),
        ScientificSourceQualificationResult.SCHEMA,
        "9aa956e9ddf1c89e099cdf10b8dcd2da2be3e80475c3533ddb97cbf6923c67f8",
        _identity("owner.runtime.scientific-source-qualification-service"),
        OutcomeAccess.OUTCOME_BLIND,
        CapabilityKind.NUMERICAL_QUALIFIER,
    ),
    MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION: (
        "executable-source-free-property-transport.coordinate-construction",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-coordinate-construction-config',
        (
            CoordinateChallengeEvidence.SCHEMA,
            CoordinateChallengeNomination.SCHEMA,
            CoordinateChallengeSpec.SCHEMA,
            ScientificSourceQualificationResult.SCHEMA,
        ),
        MetatheoryCoordinateConstructionProduct.SCHEMA,
        "b1e98df558aff85388cf10ac55ed6a2227a2e7e5c755c7972cbbb1dbe38f350f",
        _identity("owner.runtime.metatheory-coordinate-construction-service"),
        OutcomeAccess.OUTCOME_BLIND,
        CapabilityKind.PROSPECTIVE_NOMINATOR,
    ),
    MetatheoryCampaignStageRole.COORDINATE_EVALUATION: (
        "executable-source-free-property-transport.coordinate-evaluation",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-coordinate-evaluation-config',
        (
            CoordinateChallengeSpec.SCHEMA,
            MetatheoryCoordinateConstructionProduct.SCHEMA,
        ),
        CoordinateChallengeResult.SCHEMA,
        "b8be880511a87adb6570fc6cd67428def6e8b1569ad0f2fd1d89ac6babe8d5db",
        _identity("owner.runtime.coordinate-challenge-service"),
        OutcomeAccess.OUTCOME_BLIND,
        CapabilityKind.NUMERICAL_QUALIFIER,
    ),
    MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL: (
        "executable-source-free-property-transport.atlas-qualification",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-atlas-qualification-config',
        (
            AtlasQualificationSpec.SCHEMA,
            ChartTransitionEvidence.SCHEMA,
            CoordinateChallengeResult.SCHEMA,
            SetValuedChartRegion.SCHEMA,
        ),
        AtlasQualificationResult.SCHEMA,
        "97e5fb00f839489125c224dbebb1b5eada1e6b1d41390d4b76fa1e4aec569e67",
        _identity("owner.runtime.atlas-qualification-service"),
        OutcomeAccess.OUTCOME_BLIND,
        CapabilityKind.ATLAS_ASSEMBLER,
    ),
    MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL: (
        "executable-source-free-property-transport.decision-assurance",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-decision-assurance-config',
        (
            AtlasQualificationResult.SCHEMA,
            DecisionAssuranceSpec.SCHEMA,
            DecisionComparisonEvidence.SCHEMA,
        ),
        DecisionAssuranceResult.SCHEMA,
        "fb6f0871c337f105d0c8f5eded098b57ad3290a1ea1c7326a9241118addc54a8",
        _identity("owner.runtime.decision-assurance-service"),
        OutcomeAccess.OUTCOME_BLIND,
        CapabilityKind.ADMISSION_EVALUATOR,
    ),
    MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL: (
        "executable-source-free-property-transport.property-survival",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-property-survival-config',
        (
            DecisionAssuranceResult.SCHEMA,
            EvidenceDependenceSpec.SCHEMA,
            PropertyPathCorrespondence.SCHEMA,
            PropertySurvivalCellEvidence.SCHEMA,
            PropertySurvivalSpec.SCHEMA,
        ),
        FacePairedPropertySurvivalAssessment.SCHEMA,
        "a222b13aa785fdbd9582629b484116238fd01f0949287ea078e68016ae603711",
        _identity("owner.runtime.property-survival-service"),
        OutcomeAccess.OUTCOME_BLIND,
        CapabilityKind.TRANSPORT_TESTER,
    ),
    MetatheoryCampaignStageRole.PREDICTION_ISSUE: (
        "executable-source-free-property-transport.prediction-issue",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-prediction-issue-config',
        (MetatheoryPredictionPackage.SCHEMA, FacePairedPropertySurvivalAssessment.SCHEMA),
        MetatheoryPredictionIssueReceipt.SCHEMA,
        "f6c7fca2b4da365f095243e2109db23b4418f707438925d7b5e87c0d9083f6fd",
        _identity("owner.runtime.metatheory-prediction-issue-service"),
        OutcomeAccess.OUTCOME_BLIND,
        CapabilityKind.EXPERIMENT_DESIGNER,
    ),
    MetatheoryCampaignStageRole.TARGET_ACQUISITION: (
        "executable-source-free-property-transport.target-acquisition",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-target-acquisition-config',
        (
            MetatheoryPredictionIssueReceipt.SCHEMA,
            MetatheoryPredictionPackage.SCHEMA,
            MetatheorySealedTarget.SCHEMA,
        ),
        MetatheoryTargetAcquisitionIndex.SCHEMA,
        "f6c7fca2b4da365f095243e2109db23b4418f707438925d7b5e87c0d9083f6fd",
        _identity("owner.runtime.metatheory-target-acquisition-service"),
        OutcomeAccess.EVALUATION_SEALED,
        CapabilityKind.SOURCE,
    ),
    MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION: (
        "executable-source-free-property-transport.reveal-adjudication",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-reveal-adjudication-config',
        (
            EvidenceDependenceAssessment.SCHEMA,
            MetatheoryAdjudicationSpec.SCHEMA,
            MetatheoryPredictionIssueReceipt.SCHEMA,
            MetatheoryPredictionPackage.SCHEMA,
            MetatheoryRevealAuthorizationBinding.SCHEMA,
            MetatheoryTargetAcquisitionIndex.SCHEMA,
            MetatheoryTargetOutcome.SCHEMA,
            SourceFreePropertyTransportSealedOutcomeBundle.SCHEMA,
            RevealedTargetOutcomeBinding.SCHEMA,
        ),
        CustodyBoundMetatheoryAdjudicationResult.SCHEMA,
        "f6c7fca2b4da365f095243e2109db23b4418f707438925d7b5e87c0d9083f6fd",
        _identity("owner.runtime.metatheory-adjudication-service"),
        OutcomeAccess.EVALUATION_REVEALED,
        CapabilityKind.EVALUATOR,
    ),
    MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT: (
        "executable-source-free-property-transport.obstruction-closeout",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-obstruction-closeout-config',
        (
            AtlasQualificationResult.SCHEMA,
            CoordinateChallengeResult.SCHEMA,
            CustodyBoundMetatheoryAdjudicationResult.SCHEMA,
            ObstructionAtlasSpec.SCHEMA,
            ObstructionSourceBinding.SCHEMA,
            FacePairedPropertySurvivalAssessment.SCHEMA,
            ScientificSourceQualificationResult.SCHEMA,
        ),
        ObstructionAtlas.SCHEMA,
        "3afa51df64519ba0f67360b17e7bee8b184882c3f5509a35710424ef578d8066",
        _identity("owner.runtime.obstruction-atlas-builder"),
        OutcomeAccess.EVALUATION_REVEALED,
        CapabilityKind.ATLAS_ASSEMBLER,
    ),
    MetatheoryCampaignStageRole.REPORT: (
        "executable-source-free-property-transport.report",
        'empirical-lawhood/methods/backbone-structural-transport/source-free-property-transport-report-config',
        (MetatheoryReportInput.SCHEMA, ObstructionAtlas.SCHEMA),
        ScientificAdjudicationRecord.SCHEMA,
        "f456ff891b8f15f3e7bcedd1847e3669717fbebe1a38453b4b844bdf1890df33",
        _identity("owner.runtime.metatheory-report-service"),
        OutcomeAccess.EVALUATION_REVEALED,
        CapabilityKind.REPORTER,
    ),
}


def _permissions(
    role: MetatheoryCampaignStageRole,
) -> tuple[CapabilityPermission, ...]:
    values = set(_BASE_PERMISSIONS)
    if role is MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION:
        # The platform reveal barrier owns authority.  The evaluator receives
        # sealed dependencies only through that separately authorized barrier.
        values.add(CapabilityPermission.READ_OUTCOME_VISIBLE)
    elif role in {
        MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT,
        MetatheoryCampaignStageRole.REPORT,
    }:
        values.add(CapabilityPermission.READ_OUTCOME_VISIBLE)
    return tuple(sorted(values, key=lambda value: value.value))


def source_free_property_transport_capability_registry() -> CapabilityRegistry:
    manifests = []
    for role, value in _ROLE_MATRIX.items():
        key, config_schema, inputs, output, implementation, _, access, kind = value
        manifests.append(
            CapabilityManifest(
                capability_key=key,
                capability_version=SOURCE_FREE_PROPERTY_TRANSPORT_CAPABILITY_VERSION,
                kind=kind,
                config_schema=config_schema,
                config_schema_sha256=sha256(config_schema.encode()).hexdigest(),
                input_schema_ids=tuple(sorted(inputs)),
                output_schema_ids=tuple(
                    sorted(
                        (
                            output,
                            *(
                                (CoordinateChallengeSpec.SCHEMA,)
                                if role
                                is MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION
                                else ()
                            ),
                            *(
                                (MetatheoryPredictionPackage.SCHEMA,)
                                if role
                                is MetatheoryCampaignStageRole.PREDICTION_ISSUE
                                else ()
                            ),
                            *(
                                (SourceFreePropertyTransportSealedOutcomeBundle.SCHEMA,)
                                if role
                                is MetatheoryCampaignStageRole.TARGET_ACQUISITION
                                else ()
                            ),
                        )
                    )
                ),
                permissions=_permissions(role),
                maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                maximum_outcome_access=access,
                resource_ceiling=SOURCE_FREE_PROPERTY_TRANSPORT_RESOURCE_BUDGET,
                deterministic=True,
                seed_required=False,
                language_id="python",
                runtime_id="cpython",
                requires_clean_commit=True,
                requires_active_mount=False,
                requires_network=False,
                conformance_check_ids=(
                    f"conformance.source-free-property-transport.{role.value.lower()}",
                ),
                implementation_sha256=implementation,
            )
        )
    return CapabilityRegistry(
        registry_id="registry.executable-source-free-property-transport",
        capabilities=tuple(sorted(manifests, key=lambda item: item.registry_id)),
    )


def source_free_property_transport_owner_contracts(
    *,
    profile: MetatheoryCampaignProfile,
    registry: CapabilityRegistry,
) -> tuple[MetatheoryStageOwnerContract, ...]:
    applicable = {
        role
        for role in _ROLE_MATRIX
        if role is not MetatheoryCampaignStageRole.LAW_QUALIFICATION_OPTIONAL
        and {
            MetatheoryCampaignStageRole.SOURCE_PIPELINE: profile.source_pipeline_profile,
            MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION: (
                profile.source_qualification_spec
            ),
            MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION: (
                profile.coordinate_challenge_spec
            ),
            MetatheoryCampaignStageRole.COORDINATE_EVALUATION: (
                profile.coordinate_challenge_spec
            ),
            MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL: (
                profile.atlas_qualification_spec
            ),
            MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL: (
                profile.decision_assurance_spec
            ),
            MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL: (
                profile.property_survival_spec
            ),
            MetatheoryCampaignStageRole.PREDICTION_ISSUE: profile.prediction_package,
            MetatheoryCampaignStageRole.TARGET_ACQUISITION: profile.prediction_package,
            MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION: profile.prediction_package,
            MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT: profile.obstruction_profile,
            MetatheoryCampaignStageRole.REPORT: profile.obstruction_profile,
        }[role]
        is not None
    }
    contracts = []
    for role in applicable:
        key, _, inputs, output, _, owner, access, _ = _ROLE_MATRIX[role]
        manifest = registry.resolve(key, SOURCE_FREE_PROPERTY_TRANSPORT_CAPABILITY_VERSION)
        contracts.append(
            MetatheoryStageOwnerContract(
                contract_id=f"owner-contract.source-free-property-transport.{role.value.lower()}",
                role=role,
                capability_key=key,
                capability_version=SOURCE_FREE_PROPERTY_TRANSPORT_CAPABILITY_VERSION,
                config_schema=manifest.config_schema,
                input_schema_ids=tuple(sorted(inputs)),
                output_schema_id=output,
                canonical_owner=owner,
                supporting_owners=(
                    (SOURCE_FREE_PROPERTY_TRANSPORT_EVIDENCE_DEPENDENCE_OWNER,)
                    if role is MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL
                    else ()
                ),
                implementation_sha256=manifest.implementation_sha256,
                permissions=manifest.permissions,
                resource_budget=manifest.resource_ceiling,
                outcome_access=access,
                visibility_ceiling=(
                    VisibilityCeiling.OUTCOME_VISIBLE
                    if access is OutcomeAccess.EVALUATION_REVEALED
                    else VisibilityCeiling.PROSPECTIVE
                ),
            )
        )
    return tuple(sorted(contracts, key=lambda item: item.role.value))


def source_free_property_transport_owner_contract(
    *,
    role: MetatheoryCampaignStageRole,
    registry: CapabilityRegistry,
) -> MetatheoryStageOwnerContract:
    """Return the frozen contract for one installed corrected stage."""

    try:
        key, _, inputs, output, _, owner, access, _ = _ROLE_MATRIX[role]
    except KeyError as error:
        raise ValueError(f"SOURCE_FREE_PROPERTY_TRANSPORT_ROLE_NOT_EXECUTABLE:{role.value}") from error
    manifest = registry.resolve(key, SOURCE_FREE_PROPERTY_TRANSPORT_CAPABILITY_VERSION)
    return MetatheoryStageOwnerContract(
        contract_id=f"owner-contract.source-free-property-transport.{role.value.lower()}",
        role=role,
        capability_key=key,
        capability_version=SOURCE_FREE_PROPERTY_TRANSPORT_CAPABILITY_VERSION,
        config_schema=manifest.config_schema,
        input_schema_ids=tuple(sorted(inputs)),
        output_schema_id=output,
        canonical_owner=owner,
        supporting_owners=(
            (SOURCE_FREE_PROPERTY_TRANSPORT_EVIDENCE_DEPENDENCE_OWNER,)
            if role is MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL
            else ()
        ),
        implementation_sha256=manifest.implementation_sha256,
        permissions=manifest.permissions,
        resource_budget=manifest.resource_ceiling,
        outcome_access=access,
        visibility_ceiling=(
            VisibilityCeiling.OUTCOME_VISIBLE
            if access is OutcomeAccess.EVALUATION_REVEALED
            else VisibilityCeiling.PROSPECTIVE
        ),
    )


__all__ = [
    "SOURCE_FREE_PROPERTY_TRANSPORT_EVIDENCE_DEPENDENCE_OWNER",
    "SOURCE_FREE_PROPERTY_TRANSPORT_RESOURCE_BUDGET",
    "SOURCE_FREE_PROPERTY_TRANSPORT_CAPABILITY_VERSION",
    'source_free_property_transport_capability_registry',
    'source_free_property_transport_owner_contract',
    'source_free_property_transport_owner_contracts',
]
