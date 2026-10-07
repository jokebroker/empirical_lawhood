"""Closed source-free conformance inputs for the corrected metatheory route.

Case values are nonauthoritative suite labels.  Scientific differences live in
the typed operands below; no config stores an expected aggregate disposition or
report status.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.coordinate_challenges import CoordinateSamplingMode
from empirical_lawhood.planning.decision_assurance import DecisionDisposition
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceSpec
from empirical_lawhood.planning.metatheory import EvidenceDependenceClass, LegitimateNextAct, MetatheoryCellDisposition, MetatheoryClaimKind, MetatheoryEvidenceCeiling, MetatheoryObstructionKind, MetatheoryPredictiveLevel
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignProfile, MetatheoryCampaignStageRole
from empirical_lawhood.planning.metatheory_prediction import CustodyBoundMetatheoryAdjudicationResult, MetatheoryMissingTargetPolicy
from empirical_lawhood.planning.obstruction_atlas import ObstructionClaimEffect, ObstructionMappingRegistry, ObstructionMapping
from empirical_lawhood.planning.property_survival import FacePairedPropertySurvivalAssessment, PropertySurvivalKind, PropertySurvivalPath
from empirical_lawhood.planning.scientific_source_qualification import ScientificSourceGateKind, ScientificSourceQualificationResult
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeResult
from empirical_lawhood.planning.atlas_qualification import AtlasQualificationResult
from empirical_lawhood.runtime.candidate_compiler import CandidateScientificGraph
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.metatheory_campaigns import MetatheoryCampaignCompilation

from .source_free_property_transport_atlas_input_parts import _evidence as _atlas_evidence, _region as _atlas_region, _registry as _atlas_registry, _spec as _atlas_spec
from .source_free_property_transport_campaign import build_source_free_property_transport_graph, compile_source_free_property_transport_campaign, source_free_property_transport_profile_for_registry
from .source_free_property_transport_contracts import source_free_property_transport_capability_registry
from .source_free_property_transport_coordinate_input_parts import _inputs as _coordinate_inputs, _registry as _coordinate_registry, _spec as _coordinate_spec
from .source_free_property_transport_decision_input_parts import _comparison_spec as _decision_spec, _evidence as _decision_evidence, _registry as _decision_registry
from .source_free_property_transport_lineage_input_parts import _base_pair
from .source_free_property_transport_prediction_input_parts import _adjudication_spec, _dependence as _prediction_dependence, _package as _prediction_package, _registry as _prediction_registry
from .source_free_property_transport_property_input_parts import _correspondences as _property_correspondences, _evidence as _property_evidence, _registry as _property_registry, _spec as _property_spec
from .source_free_property_transport_records import SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE, SourceFreePropertyTransportAtlasQualificationConfig, SourceFreePropertyTransportCoordinateConstructionConfig, SourceFreePropertyTransportCoordinateEvaluationConfig, SourceFreePropertyTransportDecisionAssuranceConfig, SourceFreePropertyTransportObstructionCloseoutConfig, SourceFreePropertyTransportPredictionIssueConfig, SourceFreePropertyTransportPreparedMedium, SourceFreePropertyTransportPropertySurvivalConfig, SourceFreePropertyTransportReportConfig, SourceFreePropertyTransportRevealAdjudicationConfig, SourceFreePropertyTransportSourcePipelineConfig, SourceFreePropertyTransportSourceQualificationConfig, SourceFreePropertyTransportSyntheticTargetGenerator, SourceFreePropertyTransportTargetAcquisitionConfig
from .source_free_property_transport_source_input_parts import _compile as _compile_source, _execution_evidence as _source_execution_evidence, _profile as _source_profile
from .source_free_property_transport_source_qualification_input_parts import _evidence as _source_evidence, _registry as _source_registry, _spec as _source_spec


class SourceFreePropertyTransportConformanceCase(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED_PROPERTY_DECISION = "OPPOSED_PROPERTY_DECISION"
    COLLISION_ABSENT_UNEVALUABLE = "COLLISION_ABSENT_UNEVALUABLE"
    ATLAS_LOCALIZATION_TOO_WIDE = "ATLAS_LOCALIZATION_TOO_WIDE"
    MIXED_PREDICTIVE_LEVELS = "MIXED_PREDICTIVE_LEVELS"
    MISSING_PERMITTED_TARGET = "MISSING_PERMITTED_TARGET"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    AUTHORITY_STOP = "AUTHORITY_STOP"


_CASE_TOKEN = {
    SourceFreePropertyTransportConformanceCase.SUPPORTED: "c01",
    SourceFreePropertyTransportConformanceCase.OPPOSED_PROPERTY_DECISION: "c02",
    SourceFreePropertyTransportConformanceCase.COLLISION_ABSENT_UNEVALUABLE: "c03",
    SourceFreePropertyTransportConformanceCase.ATLAS_LOCALIZATION_TOO_WIDE: "c04",
    SourceFreePropertyTransportConformanceCase.MIXED_PREDICTIVE_LEVELS: "c05",
    SourceFreePropertyTransportConformanceCase.MISSING_PERMITTED_TARGET: "c06",
    SourceFreePropertyTransportConformanceCase.PREREQUISITE_NONATTEMPT: "c07",
    SourceFreePropertyTransportConformanceCase.AUTHORITY_STOP: "c08",
}


@dataclass(frozen=True, slots=True)
class SourceFreePropertyTransportCaseInputs:
    """Code-owned records and compiled topology for one source-free case."""

    case: SourceFreePropertyTransportConformanceCase
    registry: CapabilityRegistry
    configs: tuple[CanonicalRecord, ...]
    profile: MetatheoryCampaignProfile
    compilation: MetatheoryCampaignCompilation
    graph: CandidateScientificGraph


def _identity(object_id: str, schema: str) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema=schema,
        object_version="1.0.0",
        object_fingerprint=sha256(f"{object_id}:{schema}".encode()).hexdigest(),
    )


def _mapping(
    *,
    terminal_schema: str,
    reason: str,
    kind: MetatheoryObstructionKind,
    effect: ObstructionClaimEffect,
    next_act: LegitimateNextAct,
) -> ObstructionMapping:
    token = reason.lower().replace("_", "-").replace(".", "-")
    return ObstructionMapping(
        mapping_id=f"mapping.source-free-property-transport.{token}",
        terminal_schema=terminal_schema,
        reason_code=reason,
        obstruction_kind=kind,
        claim_effect=effect,
        allowed_next_acts=(next_act,),
    )


def _obstruction_registry() -> ObstructionMappingRegistry:
    mappings = (
        _mapping(
            terminal_schema=ScientificSourceQualificationResult.SCHEMA,
            reason="DECISIVE_FALSIFIER",
            kind=MetatheoryObstructionKind.PREREQUISITE_NONATTEMPT,
            effect=ObstructionClaimEffect.PREREQUISITE_NONATTEMPT,
            next_act=LegitimateNextAct.PREDECLARE_DISCRIMINATING_STUDY,
        ),
        _mapping(
            terminal_schema=CoordinateChallengeResult.SCHEMA,
            reason="NO_INFORMATIVE_COLLISION_ENTERED",
            kind=MetatheoryObstructionKind.OPERAND_ABSENT,
            effect=ObstructionClaimEffect.UNEVALUABLE,
            next_act=LegitimateNextAct.MEASURE_MISSING_OPERAND,
        ),
        _mapping(
            terminal_schema=AtlasQualificationResult.SCHEMA,
            reason="UNCERTAINTY_REGION_TOO_WIDE_OR_UNLOCALIZED",
            kind=MetatheoryObstructionKind.INSUFFICIENT_POWER,
            effect=ObstructionClaimEffect.UNEVALUABLE,
            next_act=LegitimateNextAct.NEW_POWERED_DESIGN,
        ),
        _mapping(
            terminal_schema=FacePairedPropertySurvivalAssessment.SCHEMA,
            reason="PROPERTY_SURVIVAL_FALSIFIED",
            kind=MetatheoryObstructionKind.OBSERVED_OPPOSITION,
            effect=ObstructionClaimEffect.OPPOSES,
            next_act=LegitimateNextAct.PREDECLARE_DISCRIMINATING_STUDY,
        ),
        _mapping(
            terminal_schema=CustodyBoundMetatheoryAdjudicationResult.SCHEMA,
            reason="DECISIVE_UNSAFE_CATEGORY_OBSERVED",
            kind=MetatheoryObstructionKind.OBSERVED_OPPOSITION,
            effect=ObstructionClaimEffect.OPPOSES,
            next_act=LegitimateNextAct.PREDECLARE_DISCRIMINATING_STUDY,
        ),
        _mapping(
            terminal_schema=CustodyBoundMetatheoryAdjudicationResult.SCHEMA,
            reason="FROZEN_FORECAST_OPPOSED",
            kind=MetatheoryObstructionKind.OBSERVED_OPPOSITION,
            effect=ObstructionClaimEffect.OPPOSES,
            next_act=LegitimateNextAct.PREDECLARE_DISCRIMINATING_STUDY,
        ),
        _mapping(
            terminal_schema=CustodyBoundMetatheoryAdjudicationResult.SCHEMA,
            reason="PERMITTED_TARGET_MISSING",
            kind=MetatheoryObstructionKind.OPERAND_ABSENT,
            effect=ObstructionClaimEffect.UNEVALUABLE,
            next_act=LegitimateNextAct.PRESERVE_UNEVALUABLE,
        ),
    )
    return ObstructionMappingRegistry(
        registry_id="registry.source-free-property-transport.obstruction-mappings",
        mappings=tuple(sorted(mappings, key=lambda value: value.mapping_id)),
    )


def _active_roles(case: SourceFreePropertyTransportConformanceCase) -> set[MetatheoryCampaignStageRole]:
    prefix = {
        MetatheoryCampaignStageRole.SOURCE_PIPELINE,
        MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION,
    }
    if case is SourceFreePropertyTransportConformanceCase.PREREQUISITE_NONATTEMPT:
        return prefix
    prefix.update(
        {
            MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION,
            MetatheoryCampaignStageRole.COORDINATE_EVALUATION,
        }
    )
    if case is SourceFreePropertyTransportConformanceCase.COLLISION_ABSENT_UNEVALUABLE:
        return prefix
    prefix.add(MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL)
    if case is SourceFreePropertyTransportConformanceCase.ATLAS_LOCALIZATION_TOO_WIDE:
        return prefix
    prefix.update(
        {
            MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL,
            MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL,
        }
    )
    if case is SourceFreePropertyTransportConformanceCase.OPPOSED_PROPERTY_DECISION:
        return prefix
    prefix.update(
        {
            MetatheoryCampaignStageRole.PREDICTION_ISSUE,
            MetatheoryCampaignStageRole.TARGET_ACQUISITION,
            MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION,
        }
    )
    return prefix


def _case_configs(
    case: SourceFreePropertyTransportConformanceCase,
) -> tuple[CanonicalRecord, ...]:
    token = _CASE_TOKEN[case]
    roles = _active_roles(case)

    source_profile, source_manifest, source_capabilities, dataset_registry = _source_profile(
        label="source-free-property-transport-source-free"
    )
    source_compilation = _compile_source(
        source_profile,
        source_manifest,
        source_capabilities,
        dataset_registry,
    )
    assert source_compilation.dataset_operation_request is not None
    source_config = SourceFreePropertyTransportSourcePipelineConfig(
        config_id=f"config.source-free-property-transport.{token}.source-pipeline",
        prepared_medium=SourceFreePropertyTransportPreparedMedium(
            medium_id=f"medium.source-free-property-transport.{token}",
            physical_unit_ids=("unit.synthetic.001", "unit.synthetic.002"),
            source_free=True,
            grants_authority=False,
        ),
        profile=source_profile,
        source_manifest=source_manifest,
        capability_registry=source_capabilities,
        dataset_registry=dataset_registry,
        dataset_operation_request=source_compilation.dataset_operation_request,
        execution_evidence=_source_execution_evidence(source_profile, source_compilation),
        decision_clock_id="source-pipeline-decision-clock",
        decision_time_utc="2026-08-23T01:00:00Z",
        grants_authority=False,
    )

    source_spec = _source_spec()
    source_qualification_config = SourceFreePropertyTransportSourceQualificationConfig(
        config_id=f"config.source-free-property-transport.{token}.source-qualification",
        spec=source_spec,
        capability_registry=_source_registry(),
        gate_evidence=_source_evidence(
            source_spec,
            opposed_gate=(
                ScientificSourceGateKind.SOURCE_IDENTITY
                if case is SourceFreePropertyTransportConformanceCase.PREREQUISITE_NONATTEMPT
                else None
            ),
        ),
        grants_authority=False,
    )

    coordinate_spec = _coordinate_spec(CoordinateSamplingMode.TARGETED)
    nominations, coordinate_evidence = _coordinate_inputs(
        coordinate_spec,
        entered=case is not SourceFreePropertyTransportConformanceCase.COLLISION_ABSENT_UNEVALUABLE,
    )
    coordinate_construction = SourceFreePropertyTransportCoordinateConstructionConfig(
        config_id=f"config.source-free-property-transport.{token}.coordinate-construction",
        spec=coordinate_spec,
        capability_registry=_coordinate_registry(),
        nominations=nominations,
        evidence=coordinate_evidence,
        grants_authority=False,
    )
    coordinate_evaluation = SourceFreePropertyTransportCoordinateEvaluationConfig(
        config_id=f"config.source-free-property-transport.{token}.coordinate-evaluation",
        spec=coordinate_spec,
        capability_registry=_coordinate_registry(),
        grants_authority=False,
    )

    levels = tuple(sorted(MetatheoryPredictiveLevel, key=lambda value: value.value))
    atlas_spec = _atlas_spec(levels=levels)
    atlas_region = _atlas_region()
    atlas = SourceFreePropertyTransportAtlasQualificationConfig(
        config_id=f"config.source-free-property-transport.{token}.atlas-qualification",
        spec_template=atlas_spec,
        capability_registry=_atlas_registry(),
        region=atlas_region,
        transition_evidence=_atlas_evidence(
            atlas_spec,
            atlas_region,
            metric_too_wide=(case is SourceFreePropertyTransportConformanceCase.ATLAS_LOCALIZATION_TOO_WIDE),
        ),
        grants_authority=False,
    )

    decision_spec = _decision_spec()
    decision = SourceFreePropertyTransportDecisionAssuranceConfig(
        config_id=f"config.source-free-property-transport.{token}.decision-assurance",
        spec=decision_spec,
        capability_registry=_decision_registry(),
        evidence=(
            _decision_evidence(
                decision_spec,
                decision_spec.targets[0],
                candidate=DecisionDisposition.ACTIVE_ACTION,
                reference=DecisionDisposition.ACTIVE_ACTION,
            ),
        ),
        grants_authority=False,
    )

    property_spec = _property_spec(
        {PropertySurvivalKind.COMPOSITION},
        composed={PropertySurvivalKind.COMPOSITION},
    )
    predecessor, target = _base_pair()
    dependence_spec = EvidenceDependenceSpec(
        spec_id='dependence.source-free-property-transport.synthetic',
        requested_class=EvidenceDependenceClass.SAME_IMPLEMENTATION_RESAMPLE,
        predecessor=predecessor,
        target=target,
        stronger_class_refusal_required=True,
    )
    property_spec = replace(
        property_spec,
        dependence_spec=ObjectIdentity.from_record(dependence_spec.spec_id, dependence_spec),
    )
    property_disposition = (
        MetatheoryCellDisposition.OPPOSED
        if case is SourceFreePropertyTransportConformanceCase.OPPOSED_PROPERTY_DECISION
        else MetatheoryCellDisposition.SUPPORTED
    )
    property_evidence = _property_evidence(
        property_spec,
        {
            (PropertySurvivalKind.COMPOSITION, PropertySurvivalPath.DIRECT): (
                property_disposition
            ),
            (PropertySurvivalKind.COMPOSITION, PropertySurvivalPath.COMPOSED): (
                property_disposition
            ),
        },
    )
    property_config = SourceFreePropertyTransportPropertySurvivalConfig(
        config_id=f"config.source-free-property-transport.{token}.property-survival",
        spec=property_spec,
        dependence_spec=dependence_spec,
        capability_registry=_property_registry(),
        evidence=property_evidence,
        path_correspondences=_property_correspondences(property_spec, property_evidence),
        grants_authority=False,
    )

    missing = case is SourceFreePropertyTransportConformanceCase.MISSING_PERMITTED_TARGET
    package = _prediction_package(
        MetatheoryMissingTargetPolicy.PERMIT_UNEVALUABLE
        if missing
        else MetatheoryMissingTargetPolicy.REQUIRE_COMPLETE
    )
    target_id = f"target.source-free-property-transport.{token}"
    package = replace(
        package,
        package_id=f"source-free-property-transport-prediction.{token}.synthetic",
        targets=(replace(package.targets[0], target_id=target_id),),
        forecast_cells=tuple(
            replace(value, target_id=target_id) for value in package.forecast_cells
        ),
        planned_issue_id=f"issue-clock.source-free-property-transport.{token}",
    )
    prediction = SourceFreePropertyTransportPredictionIssueConfig(
        config_id=f"config.source-free-property-transport.{token}.prediction-issue",
        package_template=package,
        issue_clock_id=package.planned_issue_id,
        publication=_identity(
            f"publication.source-free-property-transport.{token}.prediction",
            'empirical-lawhood/runtime/structural-transport/prediction-publication-reference',
        ),
        recovery=_identity(
            f"recovery.source-free-property-transport.{token}.prediction",
            'empirical-lawhood/runtime/run-recovery-reference',
        ),
        grants_authority=False,
    )
    target_config = SourceFreePropertyTransportTargetAcquisitionConfig(
        config_id=f"config.source-free-property-transport.{token}.target-acquisition",
        generator=SourceFreePropertyTransportSyntheticTargetGenerator(
            generator_id=f"generator.source-free-property-transport.{token}",
            seed=(
                "seed.metatheory-r1.000005"
                if case is SourceFreePropertyTransportConformanceCase.MIXED_PREDICTIVE_LEVELS
                else "seed.metatheory-r1.000007"
            ),
            target_present=not missing,
            source_free=True,
            grants_authority=False,
        ),
        grants_authority=False,
    )
    reveal = SourceFreePropertyTransportRevealAdjudicationConfig(
        config_id=f"config.source-free-property-transport.{token}.reveal-adjudication",
        spec_template=replace(
            _adjudication_spec(package),
            spec_id=f"adjudication-spec.source-free-property-transport.{token}",
        ),
        dependence_assessment=_prediction_dependence(package),
        scoring_registry=_prediction_registry(),
        reveal_authority=_identity(
            f"authority.source-free-property-transport.{token}.reveal",
            'empirical-lawhood/runtime/structural-transport/reveal-authority-reference',
        ),
        revealed_publication_namespace=f"revealed.source-free-property-transport.{token}",
        revealed_recovery_namespace=f"recovery.source-free-property-transport.{token}",
        reveal_permitted=True,
        grants_authority=False,
    )

    obstruction = SourceFreePropertyTransportObstructionCloseoutConfig(
        config_id=f"config.source-free-property-transport.{token}.obstruction-closeout",
        mapping_registry=_obstruction_registry(),
        target_id=target_id,
        claim_kind=MetatheoryClaimKind.STRUCTURAL_RECURRENCE,
        predictive_level=MetatheoryPredictiveLevel.CATEGORICAL,
        affected_operand_ids=("operand.source-free-property-transport.synthetic",),
        affected_property_ids=("property.composition",),
        affected_coordinate_ids=("coordinate.synthetic",),
        affected_action_ids=("action-fibre.synthetic",),
        maximum_structural_evidence_ceiling=(MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE),
        grants_authority=False,
    )
    report = SourceFreePropertyTransportReportConfig(
        config_id=f"config.source-free-property-transport.{token}.report",
        fixture_scope_id=f"fixture.source-free-property-transport.{token}",
        plumbing_only=True,
        grants_authority=False,
    )

    by_role: dict[MetatheoryCampaignStageRole, CanonicalRecord] = {
        MetatheoryCampaignStageRole.SOURCE_PIPELINE: source_config,
        MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION: (
            source_qualification_config
        ),
        MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION: coordinate_construction,
        MetatheoryCampaignStageRole.COORDINATE_EVALUATION: coordinate_evaluation,
        MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL: atlas,
        MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL: decision,
        MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL: property_config,
        MetatheoryCampaignStageRole.PREDICTION_ISSUE: prediction,
        MetatheoryCampaignStageRole.TARGET_ACQUISITION: target_config,
        MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION: reveal,
        MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT: obstruction,
        MetatheoryCampaignStageRole.REPORT: report,
    }
    selected = roles | {
        MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT,
        MetatheoryCampaignStageRole.REPORT,
    }
    return tuple(by_role[role] for role in MetatheoryCampaignStageRole if role in selected)


def build_source_free_property_transport_case(
    case: SourceFreePropertyTransportConformanceCase,
) -> SourceFreePropertyTransportCaseInputs:
    registry = source_free_property_transport_capability_registry()
    configs = _case_configs(case)
    if any(
        not isinstance(config, SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE[role])
        for role, config in zip(
            (
                role
                for role in MetatheoryCampaignStageRole
                if role in SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE
                and any(
                    isinstance(value, SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE[role]) for value in configs
                )
            ),
            configs,
            strict=True,
        )
    ):
        raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_ROSTER_MISMATCH")
    profile = source_free_property_transport_profile_for_registry(registry=registry, configs=configs)
    compilation = compile_source_free_property_transport_campaign(profile=profile, registry=registry)
    source = next(
        value for value in configs if isinstance(value, SourceFreePropertyTransportSourcePipelineConfig)
    )
    graph = build_source_free_property_transport_graph(
        compilation=compilation,
        registry=registry,
        prepared_medium=source.prepared_medium,
    )
    return SourceFreePropertyTransportCaseInputs(
        case=case,
        registry=registry,
        configs=configs,
        profile=profile,
        compilation=compilation,
        graph=graph,
    )


__all__ = [
    'SourceFreePropertyTransportCaseInputs',
    'SourceFreePropertyTransportConformanceCase',
    'build_source_free_property_transport_case',
]
