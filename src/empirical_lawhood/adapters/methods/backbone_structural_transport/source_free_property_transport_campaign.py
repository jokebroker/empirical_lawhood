"""Code-owned campaign/profile/graph assembly for corrected metatheory R1."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.metatheory import MetatheoryClaimKind
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignArtifactBinding, MetatheoryCampaignCapabilityBinding, MetatheoryCampaignOwnerBinding, MetatheoryCampaignProfile, MetatheoryCampaignRosters, MetatheoryCampaignStageRole
from empirical_lawhood.planning.study_authoring import CapabilitySelection
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.candidate_compiler import (
    CandidateGraphEdge,
    CandidateGraphExternalInput,
    CandidateGraphNode,
    CandidateScientificGraph,
    ContentIdentityPolicy,
    ScientificInputRole,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.metatheory_campaigns import MetatheoryCampaignCompilation, MetatheoryCampaignConditionEdge, compile_metatheory_campaign_profile
from empirical_lawhood.runtime.plans import OutputTemplate, ProtocolTemplate

from .source_free_property_transport_contracts import SOURCE_FREE_PROPERTY_TRANSPORT_RESOURCE_BUDGET, source_free_property_transport_owner_contract
from .source_free_property_transport_provider import SOURCE_FREE_PROPERTY_TRANSPORT_MEDIA_TYPE
from .source_free_property_transport_records import SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE, SourceFreePropertyTransportAtlasQualificationConfig, SourceFreePropertyTransportCoordinateConstructionConfig, SourceFreePropertyTransportDecisionAssuranceConfig, SourceFreePropertyTransportObstructionCloseoutConfig, SourceFreePropertyTransportPredictionIssueConfig, SourceFreePropertyTransportPreparedMedium, SourceFreePropertyTransportPropertySurvivalConfig, SourceFreePropertyTransportRevealAdjudicationConfig, SourceFreePropertyTransportSealedOutcomeBundle, SourceFreePropertyTransportSourcePipelineConfig, SourceFreePropertyTransportSourceQualificationConfig
from empirical_lawhood.planning.metatheory_prediction import MetatheoryPredictionPackage
from empirical_lawhood.planning.coordinate_challenges import CoordinateChallengeSpec


def _identity(object_id: str, schema: str) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema=schema,
        object_version="1.0.0",
        object_fingerprint=sha256(f"{object_id}:{schema}".encode()).hexdigest(),
    )


def _role_for_config(config: CanonicalRecord) -> MetatheoryCampaignStageRole:
    for role, record_type in SOURCE_FREE_PROPERTY_TRANSPORT_CONFIG_TYPE_BY_ROLE.items():
        if isinstance(config, record_type):
            return role
    raise TypeError("SOURCE_FREE_PROPERTY_TRANSPORT_UNKNOWN_CAMPAIGN_CONFIG")


def _config_by_role(
    configs: tuple[CanonicalRecord, ...],
) -> dict[MetatheoryCampaignStageRole, CanonicalRecord]:
    values = {_role_for_config(value): value for value in configs}
    if len(values) != len(configs):
        raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_REPEATED_ROLE_CONFIG")
    if not {
        MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT,
        MetatheoryCampaignStageRole.REPORT,
    }.issubset(values):
        raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_CAMPAIGN_LACKS_CLOSEOUT_CONFIGS")
    return values


def source_free_property_transport_profile_for_registry(
    *,
    registry: CapabilityRegistry,
    configs: tuple[CanonicalRecord, ...],
) -> MetatheoryCampaignProfile:
    """Bind a complete profile roster while preserving condition-false roles."""

    by_role = _config_by_role(configs)
    capabilities = []
    owners = []
    artifacts = []
    for role in MetatheoryCampaignStageRole:
        config = by_role.get(role)
        if role is MetatheoryCampaignStageRole.LAW_QUALIFICATION_OPTIONAL:
            key = "executable-source-free-property-transport.law-condition-false"
            version = "1.0.0"
            implementation = sha256(key.encode()).hexdigest()
            config_schema = 'empirical-lawhood/methods/structural-transport/optional-law-condition-false/config'
            config_sha = sha256(config_schema.encode()).hexdigest()
            config_content = sha256(b"condition-false").hexdigest()
            config_id = "config.source-free-property-transport.law-condition-false"
            output_schema = 'empirical-lawhood/methods/structural-transport/optional-law-condition-false/result'
            owner = _identity(
                "owner.source-free-property-transport.law-condition-false",
                'empirical-lawhood/runtime/scientific-owner',
            )
            access = OutcomeAccess.OUTCOME_BLIND
            visibility = VisibilityCeiling.PROSPECTIVE
        else:
            contract = source_free_property_transport_owner_contract(role=role, registry=registry)
            manifest = registry.resolve(contract.capability_key, contract.capability_version)
            key = manifest.capability_key
            version = manifest.capability_version
            implementation = manifest.implementation_sha256
            config_schema = manifest.config_schema
            config_sha = manifest.config_schema_sha256
            config_id = (
                getattr(config, "config_id")
                if config is not None
                else f"config.source-free-property-transport.condition-false.{role.value.lower()}"
            )
            config_content = (
                config.fingerprint()
                if config is not None
                else sha256(f"condition-false:{role.value}".encode()).hexdigest()
            )
            output_schema = contract.output_schema_id
            owner = contract.canonical_owner
            access = contract.outcome_access
            visibility = contract.visibility_ceiling
        capabilities.append(
            MetatheoryCampaignCapabilityBinding(
                binding_id=f"binding.source-free-property-transport.capability.{role.value.lower()}",
                role=role,
                selection=CapabilitySelection(
                    capability_key=key,
                    capability_version=version,
                    implementation_sha256=implementation,
                ),
                config_id=config_id,
                config_schema=config_schema,
                config_schema_sha256=config_sha,
                config_content_sha256=config_content,
                config_artifact_id=f"artifact.{config_id}",
                resource_budget=(
                    ResourceBudget(1, 1024, 0, 1, 0, 1024)
                    if role is MetatheoryCampaignStageRole.LAW_QUALIFICATION_OPTIONAL
                    else SOURCE_FREE_PROPERTY_TRANSPORT_RESOURCE_BUDGET
                ),
                resource_lock_ids=(),
                maximum_attempts=2,
            )
        )
        owners.append(
            MetatheoryCampaignOwnerBinding(
                binding_id=f"binding.source-free-property-transport.owner.{role.value.lower()}",
                role=role,
                owner=owner,
            )
        )
        artifacts.append(
            MetatheoryCampaignArtifactBinding(
                binding_id=f"binding.source-free-property-transport.artifact.{role.value.lower()}",
                role=role,
                output_id="result",
                payload_schema=output_schema,
                media_type=SOURCE_FREE_PROPERTY_TRANSPORT_MEDIA_TYPE,
                filename_suffix=".canonical.json",
                custody_role_id=f"custody.source-free-property-transport.{role.value.lower()}",
                maximum_bytes=1_000_000,
                outcome_access=access,
                visibility_ceiling=visibility,
            )
        )

    source = by_role.get(MetatheoryCampaignStageRole.SOURCE_PIPELINE)
    source_qualification = by_role.get(
        MetatheoryCampaignStageRole.SCIENTIFIC_SOURCE_QUALIFICATION
    )
    coordinate = by_role.get(MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION)
    atlas = by_role.get(MetatheoryCampaignStageRole.ATLAS_QUALIFICATION_OPTIONAL)
    decision = by_role.get(MetatheoryCampaignStageRole.DECISION_ASSURANCE_OPTIONAL)
    property_config = by_role.get(MetatheoryCampaignStageRole.PROPERTY_SURVIVAL_OPTIONAL)
    prediction = by_role.get(MetatheoryCampaignStageRole.PREDICTION_ISSUE)
    reveal = by_role.get(MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION)
    obstruction = by_role.get(MetatheoryCampaignStageRole.OBSTRUCTION_CLOSEOUT)
    assert source is None or isinstance(source, SourceFreePropertyTransportSourcePipelineConfig)
    assert source_qualification is None or isinstance(
        source_qualification,
        SourceFreePropertyTransportSourceQualificationConfig,
    )
    assert coordinate is None or isinstance(coordinate, SourceFreePropertyTransportCoordinateConstructionConfig)
    assert atlas is None or isinstance(atlas, SourceFreePropertyTransportAtlasQualificationConfig)
    assert decision is None or isinstance(decision, SourceFreePropertyTransportDecisionAssuranceConfig)
    assert property_config is None or isinstance(
        property_config,
        SourceFreePropertyTransportPropertySurvivalConfig,
    )
    assert prediction is None or isinstance(prediction, SourceFreePropertyTransportPredictionIssueConfig)
    assert reveal is None or isinstance(reveal, SourceFreePropertyTransportRevealAdjudicationConfig)
    assert obstruction is None or isinstance(obstruction, SourceFreePropertyTransportObstructionCloseoutConfig)
    if obstruction is None:
        raise ValueError("SOURCE_FREE_PROPERTY_TRANSPORT_CAMPAIGN_LACKS_OBSTRUCTION_CONFIG")
    profile_digest = sha256(
        b"".join(value.canonical_bytes() for value in sorted(configs, key=lambda item: item.SCHEMA))
    ).hexdigest()
    claim_kind = (
        MetatheoryClaimKind.STRUCTURAL_RECURRENCE
        if prediction is not None
        else MetatheoryClaimKind.PROPERTY_TRANSPORT
        if property_config is not None
        else MetatheoryClaimKind.DECISION_ASSURANCE
        if decision is not None
        else MetatheoryClaimKind.CHART_QUALIFICATION
        if atlas is not None
        else MetatheoryClaimKind.COORDINATE_CONSTRUCT_VALIDITY
        if coordinate is not None
        else MetatheoryClaimKind.SOURCE_PREPARATION_QUALIFICATION
    )
    return MetatheoryCampaignProfile(
        profile_id=f"source-free-property-transport-profile.{profile_digest[:24]}",
        profile_version="1.0.0",
        claim_kind=claim_kind,
        evidence_profile_selection=_identity(
            "evidence-selection.source-free-property-transport.contract-conformance",
            EvidenceProfileSelection.SCHEMA,
        ),
        source_pipeline_profile=(
            None
            if source is None
            else ObjectIdentity.from_record(source.profile.profile_id, source.profile)
        ),
        source_qualification_spec=(
            None
            if source_qualification is None
            else ObjectIdentity.from_record(
                source_qualification.spec.spec_id, source_qualification.spec
            )
        ),
        coordinate_challenge_spec=(
            None
            if coordinate is None
            else ObjectIdentity.from_record(coordinate.spec.spec_id, coordinate.spec)
        ),
        law_qualification_parent=None,
        atlas_qualification_spec=(
            None
            if atlas is None
            else ObjectIdentity.from_record(atlas.spec_template.spec_id, atlas.spec_template)
        ),
        decision_assurance_spec=(
            None
            if decision is None
            else ObjectIdentity.from_record(decision.spec.spec_id, decision.spec)
        ),
        dependence_spec=(
            None
            if property_config is None
            else ObjectIdentity.from_record(
                property_config.dependence_spec.spec_id,
                property_config.dependence_spec,
            )
        ),
        property_survival_spec=(
            None
            if property_config is None
            else ObjectIdentity.from_record(property_config.spec.spec_id, property_config.spec)
        ),
        prediction_package=(
            None
            if prediction is None
            else ObjectIdentity.from_record(
                prediction.package_template.package_id,
                prediction.package_template,
            )
        ),
        adjudication_spec=(
            None
            if reveal is None
            else ObjectIdentity.from_record(reveal.spec_template.spec_id, reveal.spec_template)
        ),
        obstruction_profile=ObjectIdentity.from_record(obstruction.config_id, obstruction),
        rosters=MetatheoryCampaignRosters(
            excluded_method_power_unit_ids=("unit.source-free-property-transport.excluded",),
            development_unit_ids=("unit.source-free-property-transport.development",),
            protected_target_unit_ids=("unit.source-free-property-transport.target",),
            reference_unit_ids=("unit.source-free-property-transport.reference",),
            reserve_unit_ids=("unit.source-free-property-transport.reserve",),
        ),
        capabilities=tuple(sorted(capabilities, key=lambda value: value.binding_id)),
        owners=tuple(sorted(owners, key=lambda value: value.binding_id)),
        artifacts=tuple(sorted(artifacts, key=lambda value: value.binding_id)),
        topology_id="topology.executable-source-free-property-transport.real-owner",
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def compile_source_free_property_transport_campaign(
    *,
    profile: MetatheoryCampaignProfile,
    registry: CapabilityRegistry,
) -> MetatheoryCampaignCompilation:
    """Add only the sealed-bundle and prediction-package dependency edges."""

    base = compile_metatheory_campaign_profile(
        profile=profile,
        capability_registry=registry,
    )
    by_role = {
        role: next(
            (
                step
                for step in base.protocol.steps
                if step.step_id == f"metatheory-{role.value.lower().replace('_', '-')}"
            ),
            None,
        )
        for role in MetatheoryCampaignStageRole
    }
    prediction = by_role[MetatheoryCampaignStageRole.PREDICTION_ISSUE]
    construction = by_role[MetatheoryCampaignStageRole.COORDINATE_CONSTRUCTION]
    target = by_role[MetatheoryCampaignStageRole.TARGET_ACQUISITION]
    reveal = by_role[MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION]
    replacements = {}
    if construction is not None:
        replacements[construction.step_id] = replace(
            construction,
            outputs=tuple(
                sorted(
                    (
                        *construction.outputs,
                        OutputTemplate(
                            output_id="spec",
                            payload_schema=CoordinateChallengeSpec.SCHEMA,
                            profile=ArtifactProfile.CANONICAL_JSON,
                            media_type=SOURCE_FREE_PROPERTY_TRANSPORT_MEDIA_TYPE,
                            filename_suffix=".canonical.json",
                        ),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
        )
    if prediction is not None:
        replacements[prediction.step_id] = replace(
            prediction,
            outputs=tuple(
                sorted(
                    (
                        *prediction.outputs,
                        OutputTemplate(
                            output_id="package",
                            payload_schema=MetatheoryPredictionPackage.SCHEMA,
                            profile=ArtifactProfile.CANONICAL_JSON,
                            media_type=SOURCE_FREE_PROPERTY_TRANSPORT_MEDIA_TYPE,
                            filename_suffix=".canonical.json",
                        ),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
        )
    if target is not None:
        replacements[target.step_id] = replace(
            target,
            outputs=tuple(
                sorted(
                    (
                        *target.outputs,
                        OutputTemplate(
                            output_id="sealed-outcomes",
                            payload_schema=SourceFreePropertyTransportSealedOutcomeBundle.SCHEMA,
                            profile=ArtifactProfile.CANONICAL_JSON,
                            media_type=SOURCE_FREE_PROPERTY_TRANSPORT_MEDIA_TYPE,
                            filename_suffix=".canonical.json",
                        ),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
        )
    if prediction is not None and target is not None and reveal is not None:
        replacements[reveal.step_id] = replace(
            reveal,
            dependency_step_ids=tuple(sorted((prediction.step_id, target.step_id))),
        )
    protocol = ProtocolTemplate(
        template_id=base.protocol.template_id,
        template_version=base.protocol.template_version,
        steps=tuple(
            sorted(
                (replacements.get(value.step_id, value) for value in base.protocol.steps),
                key=lambda value: value.step_id,
            )
        ),
        requires_model_set=base.protocol.requires_model_set,
        requests_controller=base.protocol.requests_controller,
        nonactuating=base.protocol.nonactuating,
    )
    condition_graph = list(base.condition_graph)
    if prediction is not None and reveal is not None:
        condition_graph.append(
            MetatheoryCampaignConditionEdge(
                edge_id="condition-edge.prediction-issue.reveal-and-adjudication",
                upstream_role=MetatheoryCampaignStageRole.PREDICTION_ISSUE,
                downstream_role=MetatheoryCampaignStageRole.REVEAL_AND_ADJUDICATION,
                condition_id="condition.exact-issued-package-and-sealed-target",
            )
        )
    digest = sha256(
        profile.canonical_bytes() + protocol.canonical_bytes() + registry.canonical_bytes()
    ).hexdigest()
    return replace(
        base,
        compilation_id=f"source-free-property-transport-compilation.{digest[:32]}",
        protocol=protocol,
        condition_graph=tuple(sorted(condition_graph, key=lambda value: value.edge_id)),
        predicted_terminal_schemas=tuple(
            sorted({output.payload_schema for step in protocol.steps for output in step.outputs})
        ),
    )


def build_source_free_property_transport_graph(
    *,
    compilation: MetatheoryCampaignCompilation,
    registry: CapabilityRegistry,
    prepared_medium: SourceFreePropertyTransportPreparedMedium,
) -> CandidateScientificGraph:
    """Build exact multi-output edges without a private DAG or case switch."""

    protocol = compilation.protocol
    nodes = tuple(
        CandidateGraphNode(
            node_id=step.step_id,
            stage=step.stage,
            capability_key=step.capability_key,
            capability_version=step.capability_version,
            implementation_sha256=registry.resolve(
                step.capability_key,
                step.capability_version,
            ).implementation_sha256,
            protocol_step_sha256=step.fingerprint(),
            obligation_ids=step.obligation_ids,
            outcome_access=step.requested_outcome_access,
            visibility_ceiling=step.visibility_ceiling,
            resource_budget=step.resource_budget,
        )
        for step in protocol.steps
    )
    steps = {value.step_id: value for value in protocol.steps}
    source_step = next(
        value
        for value in protocol.steps
        if value.capability_key == "executable-source-free-property-transport.source-pipeline"
    )
    external_input = CandidateGraphExternalInput(
        input_id=prepared_medium.medium_id,
        scientific_role=ScientificInputRole.PREPARED_MEDIUM,
        logical_artifact_id=prepared_medium.medium_id,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=prepared_medium.fingerprint(),
        payload_schema=prepared_medium.SCHEMA,
        media_type=SOURCE_FREE_PROPERTY_TRANSPORT_MEDIA_TYPE,
        maximum_size_bytes=1_000_000,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    edges = [
        CandidateGraphEdge(
            edge_id=f"edge.external.{prepared_medium.medium_id}.{source_step.step_id}",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=prepared_medium.medium_id,
            consumer_node_id=source_step.step_id,
            consumer_input_id=f"input.{prepared_medium.medium_id}",
            scientific_role=ScientificInputRole.PREPARED_MEDIUM,
            logical_artifact_id=external_input.logical_artifact_id,
            payload_schema=external_input.payload_schema,
            media_type=external_input.media_type,
            maximum_size_bytes=external_input.maximum_size_bytes,
            outcome_access=external_input.outcome_access,
            visibility_ceiling=external_input.visibility_ceiling,
            barrier=source_step.barrier,
        )
    ]
    for consumer in protocol.steps:
        for producer_id in consumer.dependency_step_ids:
            producer = steps[producer_id]
            for output in producer.outputs:
                edges.append(
                    CandidateGraphEdge(
                        edge_id=f"edge.{producer.step_id}.{output.output_id}.{consumer.step_id}",
                        producer_node_id=producer.step_id,
                        producer_output_id=output.output_id,
                        external_input_id=None,
                        consumer_node_id=consumer.step_id,
                        consumer_input_id=f"input.{producer.step_id}.{output.output_id}",
                        scientific_role=ScientificInputRole.PARENT_RECEIPT,
                        logical_artifact_id=f"artifact.{producer.step_id}.{output.output_id}",
                        payload_schema=output.payload_schema,
                        media_type=output.media_type,
                        maximum_size_bytes=producer.resource_budget.output_bytes,
                        outcome_access=producer.requested_outcome_access,
                        visibility_ceiling=producer.visibility_ceiling,
                        barrier=consumer.barrier,
                    )
                )
    return CandidateScientificGraph(
        graph_id=f"graph.{compilation.compilation_id}",
        external_inputs=(external_input,),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


__all__ = [
    'build_source_free_property_transport_graph',
    'compile_source_free_property_transport_campaign',
    'source_free_property_transport_profile_for_registry',
]
