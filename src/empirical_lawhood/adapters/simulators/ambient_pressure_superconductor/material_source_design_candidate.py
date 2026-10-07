'Static ambient pressure superconductor material source design candidate graph and capability catalog.'

from __future__ import annotations

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.plans import OutputTemplate, ProtocolStepTemplate, ProtocolTemplate

from .material_source_design_contracts import MATERIAL_SOURCE_DESIGN_MAXIMUM_CONFIG_BYTES, MaterialSourceDesignConfig, MaterialSourceDesignSourceQualification
from .material_source_design_runtime import DESIGN_STEP, EVALUATOR_STEP, SOURCE_CONTRACT_ARTIFACT_ID, SOURCE_STEP


def _step(protocol: ProtocolTemplate, step_id: str) -> ProtocolStepTemplate:
    return next(value for value in protocol.steps if value.step_id == step_id)


def _output(protocol: ProtocolTemplate, step_id: str, output_id: str) -> OutputTemplate:
    return next(value for value in _step(protocol, step_id).outputs if value.output_id == output_id)


def _edge(
    protocol: ProtocolTemplate,
    *,
    producer: str,
    output_id: str,
    consumer: str,
    consumer_input: str,
    role: ScientificInputRole,
    maximum_size_bytes: int,
) -> CandidateGraphEdge:
    output = _output(protocol, producer, output_id)
    source = _step(protocol, producer)
    target = _step(protocol, consumer)
    return CandidateGraphEdge(
        edge_id=f'edge.{producer}.{output_id}.{consumer}.{consumer_input}',
        producer_node_id=producer,
        producer_output_id=output_id,
        external_input_id=None,
        consumer_node_id=consumer,
        consumer_input_id=consumer_input,
        scientific_role=role,
        logical_artifact_id=f'artifact.{producer}.{output_id}',
        payload_schema=output.payload_schema,
        media_type=output.media_type,
        maximum_size_bytes=maximum_size_bytes,
        outcome_access=source.requested_outcome_access,
        visibility_ceiling=source.visibility_ceiling,
        barrier=target.barrier,
    )


def material_source_design_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    config: MaterialSourceDesignConfig,
    source_contract: MaterialSourceDesignSourceQualification,
) -> CandidateScientificGraph:
    external = CandidateGraphExternalInput(
        input_id=SOURCE_CONTRACT_ARTIFACT_ID,
        scientific_role=ScientificInputRole.MODEL,
        logical_artifact_id=SOURCE_CONTRACT_ARTIFACT_ID,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=source_contract.fingerprint(),
        payload_schema=MaterialSourceDesignSourceQualification.SCHEMA,
        media_type="application/json",
        maximum_size_bytes=1024 * 1024,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    manifests = {value.registry_id: value for value in registry.capabilities}
    nodes = tuple(
        CandidateGraphNode(
            node_id=step.step_id,
            stage=step.stage,
            capability_key=step.capability_key,
            capability_version=step.capability_version,
            implementation_sha256=manifests[
                f'{step.capability_key}@{step.capability_version}'
            ].implementation_sha256,
            protocol_step_sha256=step.fingerprint(),
            obligation_ids=step.obligation_ids,
            outcome_access=step.requested_outcome_access,
            visibility_ceiling=step.visibility_ceiling,
            resource_budget=step.resource_budget,
        )
        for step in protocol.steps
    )
    maximum = int(config.resource("output_bytes"))
    edges = [
        CandidateGraphEdge(
            edge_id=f'edge.external.{SOURCE_CONTRACT_ARTIFACT_ID}.{SOURCE_STEP}.contract',
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=SOURCE_CONTRACT_ARTIFACT_ID,
            consumer_node_id=SOURCE_STEP,
            consumer_input_id="expected-source-contract",
            scientific_role=ScientificInputRole.MODEL,
            logical_artifact_id=SOURCE_CONTRACT_ARTIFACT_ID,
            payload_schema=external.payload_schema,
            media_type=external.media_type,
            maximum_size_bytes=1024 * 1024,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=_step(protocol, SOURCE_STEP).barrier,
        ),
        _edge(
            protocol,
            producer=SOURCE_STEP,
            output_id="source-qualification",
            consumer=DESIGN_STEP,
            consumer_input="verified-source-qualification",
            role=ScientificInputRole.QUALIFICATION,
            maximum_size_bytes=maximum,
        ),
        _edge(
            protocol,
            producer=SOURCE_STEP,
            output_id="source-qualification",
            consumer=EVALUATOR_STEP,
            consumer_input="verified-source-qualification",
            role=ScientificInputRole.QUALIFICATION,
            maximum_size_bytes=maximum,
        ),
    ]
    for output_id, consumer_input, role in (
        ("design-basis-audit", "design-basis-audit", ScientificInputRole.QUALIFICATION),
        ("exploration-design-candidate", "exploration-design", ScientificInputRole.MODEL),
        ("material-roster-candidate", "material-roster", ScientificInputRole.DENOMINATOR),
        ("science-design-candidate", "science-design", ScientificInputRole.MODEL),
    ):
        edges.append(
            _edge(
                protocol,
                producer=DESIGN_STEP,
                output_id=output_id,
                consumer=EVALUATOR_STEP,
                consumer_input=consumer_input,
                role=role,
                maximum_size_bytes=maximum,
            )
        )
    return CandidateScientificGraph(
        graph_id='graph.ambient-pressure-superconductor-material-source-design-source-design-audit-audit',
        external_inputs=(external,),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def material_source_design_obligation_coverage(
    *, experiment: ExperimentSpec, protocol: ProtocolTemplate, graph: CandidateScientificGraph
) -> ObligationCoverage:
    incoming = {
        node.node_id: tuple(
            sorted(edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id)
        )
        for node in graph.nodes
    }
    owners = {
        obligation_id: (step.step_id, step.outputs[0].output_id)
        for step in protocol.steps
        for obligation_id in step.obligation_ids
    }
    fallback = (EVALUATOR_STEP, 'material-source-design-result')
    bindings = tuple(
        ObligationCoverageBinding(
            obligation_id=obligation_id,
            proof_owner_node_id=owners.get(obligation_id, fallback)[0],
            required_output_id=owners.get(obligation_id, fallback)[1],
            contributor_edge_ids=incoming[owners.get(obligation_id, fallback)[0]],
        )
        for obligation_id in required_candidate_obligation_ids(experiment, protocol)
    )
    return ObligationCoverage(
        coverage_id='obligation-coverage.ambient-pressure-superconductor-material-source-design-source-design-audit',
        bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
    )


def material_source_design_template(
    *,
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    config: MaterialSourceDesignConfig,
    source_contract: MaterialSourceDesignSourceQualification,
) -> StudyTemplate:
    graph = material_source_design_graph(
        protocol=protocol,
        registry=registry,
        config=config,
        source_contract=source_contract,
    )
    return StudyTemplate(
        template_key='ambient-pressure-superconductor.material-source-design-source-design-audit-audit',
        template_version="1.0.0",
        protocol=protocol,
        graph=graph,
        coverage=material_source_design_obligation_coverage(
            experiment=experiment,
            protocol=protocol,
            graph=graph,
        ),
    )


def material_source_design_candidate_catalog(
    *, registry: CapabilityRegistry, template: StudyTemplate
) -> CandidateCapabilityCatalog:
    registrations = tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=manifest.capability_key,
            provider_version=manifest.capability_version,
            config_media_type="application/json",
            maximum_config_bytes=MATERIAL_SOURCE_DESIGN_MAXIMUM_CONFIG_BYTES,
        )
        for manifest in registry.capabilities
    )
    return CandidateCapabilityCatalog(
        catalog_id='candidate-catalog.ambient-pressure-superconductor-material-source-design-source-design-audit',
        registrations=tuple(sorted(registrations, key=lambda value: value.manifest.registry_id)),
        templates=(template,),
    )


__all__ = [
    'material_source_design_candidate_catalog',
    'material_source_design_graph',
    'material_source_design_obligation_coverage',
    'material_source_design_template',
]
