"""Static candidate graph and capability catalog for uniform electron gas transverse receiver screen."""

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

from .contracts import MAXIMUM_CONFIG_BYTES, UniformElectronGasTransverseScreenConfig
from .runtime import CONFIG_SOURCE_ID, MEDIUM_ARTIFACT_ID


def _step(protocol: ProtocolTemplate, step_id: str) -> ProtocolStepTemplate:
    return next(value for value in protocol.steps if value.step_id == step_id)


def _output(protocol: ProtocolTemplate, step_id: str, output_id: str) -> OutputTemplate:
    return next(value for value in _step(protocol, step_id).outputs if value.output_id == output_id)


def _internal_edge(
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
        edge_id=f"edge.{producer}.{output_id}.{consumer}.{consumer_input}",
        producer_node_id=producer,
        producer_output_id=output_id,
        external_input_id=None,
        consumer_node_id=consumer,
        consumer_input_id=consumer_input,
        scientific_role=role,
        logical_artifact_id=f"artifact.{producer}.{output_id}",
        payload_schema=output.payload_schema,
        media_type=output.media_type,
        maximum_size_bytes=maximum_size_bytes,
        outcome_access=source.requested_outcome_access,
        visibility_ceiling=source.visibility_ceiling,
        barrier=target.barrier,
    )


def uniform_electron_gas_transverse_screen_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    config: UniformElectronGasTransverseScreenConfig,
) -> CandidateScientificGraph:
    external = CandidateGraphExternalInput(
        input_id=CONFIG_SOURCE_ID,
        scientific_role=ScientificInputRole.PREPARED_MEDIUM,
        logical_artifact_id=MEDIUM_ARTIFACT_ID,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=config.payload_sha256,
        payload_schema=protocol.steps[0].config.config_schema,
        media_type="application/json",
        maximum_size_bytes=MAXIMUM_CONFIG_BYTES,
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
                f"{step.capability_key}@{step.capability_version}"
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
            edge_id=f"edge.external.{CONFIG_SOURCE_ID}.truth-input-source.config",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=CONFIG_SOURCE_ID,
            consumer_node_id="truth-input-source",
            consumer_input_id="frozen-generator-config",
            scientific_role=ScientificInputRole.PREPARED_MEDIUM,
            logical_artifact_id=MEDIUM_ARTIFACT_ID,
            payload_schema=external.payload_schema,
            media_type=external.media_type,
            maximum_size_bytes=MAXIMUM_CONFIG_BYTES,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=_step(protocol, "truth-input-source").barrier,
        ),
        CandidateGraphEdge(
            edge_id=f"edge.external.{CONFIG_SOURCE_ID}.truth-oracle-custody.config",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=CONFIG_SOURCE_ID,
            consumer_node_id="truth-oracle-custody",
            consumer_input_id="frozen-generator-config",
            scientific_role=ScientificInputRole.PREPARED_MEDIUM,
            logical_artifact_id=MEDIUM_ARTIFACT_ID,
            payload_schema=external.payload_schema,
            media_type=external.media_type,
            maximum_size_bytes=MAXIMUM_CONFIG_BYTES,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=_step(protocol, "truth-oracle-custody").barrier,
        ),
        _internal_edge(
            protocol,
            producer="truth-input-source",
            output_id="truth-inputs",
            consumer="method-identify",
            consumer_input="truth-blind-inputs",
            role=ScientificInputRole.RECEIVER,
            maximum_size_bytes=maximum,
        ),
        _internal_edge(
            protocol,
            producer="truth-input-source",
            output_id="truth-inputs",
            consumer="admission-intersect",
            consumer_input="typed-branch-inputs",
            role=ScientificInputRole.RECEIVER,
            maximum_size_bytes=maximum,
        ),
        _internal_edge(
            protocol,
            producer="method-identify",
            output_id="truth-laws",
            consumer="admission-intersect",
            consumer_input="frozen-measurement-order-response-law-results",
            role=ScientificInputRole.MODEL,
            maximum_size_bytes=maximum,
        ),
        _internal_edge(
            protocol,
            producer="admission-intersect",
            output_id="truth-screens",
            consumer="evaluator-reveal",
            consumer_input="committed-measurement-order-response-law-admission-results",
            role=ScientificInputRole.QUALIFICATION,
            maximum_size_bytes=maximum,
        ),
        _internal_edge(
            protocol,
            producer="truth-oracle-custody",
            output_id="truth-oracles",
            consumer="evaluator-reveal",
            consumer_input="privileged-truth-oracles",
            role=ScientificInputRole.OUTCOME,
            maximum_size_bytes=maximum,
        ),
    ]
    return CandidateScientificGraph(
        graph_id="graph.uniform-electron-gas-transverse-screen-conformance",
        external_inputs=(external,),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def uniform_electron_gas_transverse_screen_obligation_coverage(
    *,
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
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
    fallback = ("evaluator-reveal", "method-conformance")
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
        coverage_id="obligation-coverage.uniform-electron-gas-transverse-screen-conformance",
        bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
    )


def uniform_electron_gas_transverse_screen_template(
    *,
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    config: UniformElectronGasTransverseScreenConfig,
) -> StudyTemplate:
    graph = uniform_electron_gas_transverse_screen_graph(protocol=protocol, registry=registry, config=config)
    return StudyTemplate(
        template_key="uniform-electron-gas.transverse-screen-conformance",
        template_version="1.0.0",
        protocol=protocol,
        graph=graph,
        coverage=uniform_electron_gas_transverse_screen_obligation_coverage(
            experiment=experiment,
            protocol=protocol,
            graph=graph,
        ),
    )


def uniform_electron_gas_transverse_screen_candidate_catalog(
    *, registry: CapabilityRegistry, template: StudyTemplate
) -> CandidateCapabilityCatalog:
    registrations = tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=manifest.capability_key,
            provider_version=manifest.capability_version,
            config_media_type="application/json",
            maximum_config_bytes=MAXIMUM_CONFIG_BYTES,
        )
        for manifest in registry.capabilities
    )
    return CandidateCapabilityCatalog(
        catalog_id="candidate-catalog.uniform-electron-gas-transverse-screen",
        registrations=tuple(sorted(registrations, key=lambda value: value.registration_id)),
        templates=(template,),
    )


__all__ = [
    "uniform_electron_gas_transverse_screen_candidate_catalog",
    "uniform_electron_gas_transverse_screen_graph",
    "uniform_electron_gas_transverse_screen_obligation_coverage",
    "uniform_electron_gas_transverse_screen_template",
]
