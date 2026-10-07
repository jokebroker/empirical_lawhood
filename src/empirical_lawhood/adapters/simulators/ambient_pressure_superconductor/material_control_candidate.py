'Static candidate graph for the fresh ambient pressure superconductor material control control act.'

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

from .material_control_contracts import MATERIAL_CONTROL_MAXIMUM_CONFIG_BYTES, MaterialControlConfig, MaterialControlControlBundleQualification
from .material_control_runtime import CLOSEOUT_STEP, METHOD_STEP, PANEL_TASKS, SOURCE_CONTRACT_ARTIFACT_ID, SOURCE_STEP, WORKFLOW_TASKS, CONSTRUCTIVE_SEARCH_CONFORMANCE_STEP


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


def material_control_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    config: MaterialControlConfig,
    source_contract: MaterialControlControlBundleQualification,
) -> CandidateScientificGraph:
    external = CandidateGraphExternalInput(
        input_id=SOURCE_CONTRACT_ARTIFACT_ID,
        scientific_role=ScientificInputRole.MODEL,
        logical_artifact_id=SOURCE_CONTRACT_ARTIFACT_ID,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=source_contract.fingerprint(),
        payload_schema=MaterialControlControlBundleQualification.SCHEMA,
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
    edges: list[CandidateGraphEdge] = [
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
        )
    ]
    for workflow_task in WORKFLOW_TASKS:
        edges.extend(
            (
                _edge(
                    protocol,
                    producer=SOURCE_STEP,
                    output_id="control-bundle",
                    consumer=workflow_task,
                    consumer_input="qualified-control-source",
                    role=ScientificInputRole.QUALIFICATION,
                    maximum_size_bytes=maximum,
                ),
                _edge(
                    protocol,
                    producer=METHOD_STEP,
                    output_id='multiband-gauge-covariant-response-bridge',
                    consumer=workflow_task,
                    consumer_input='qualified-gauge-covariant-response-method',
                    role=ScientificInputRole.QUALIFICATION,
                    maximum_size_bytes=maximum,
                ),
            )
        )
    for panel_task, structure_id in PANEL_TASKS.items():
        producers = tuple(
            task_id
            for task_id, profile in WORKFLOW_TASKS.items()
            if profile.structure_id == structure_id
            and profile.view_id in {"view.pbe-efficiency-base", "view.pbe-precision-refined"}
        )
        for index, producer in enumerate(sorted(producers), start=1):
            edges.append(
                _edge(
                    protocol,
                    producer=producer,
                    output_id="control-observation",
                    consumer=panel_task,
                    consumer_input=f'science-view-{index:02d}',
                    role=ScientificInputRole.OUTCOME,
                    maximum_size_bytes=maximum,
                )
            )

    final_edges = (
        (SOURCE_STEP, "control-bundle", "source", ScientificInputRole.QUALIFICATION),
        (METHOD_STEP, 'gauge-covariant-response-gauge-covariant-response-conformance', 'gauge-covariant-response', ScientificInputRole.QUALIFICATION),
        (METHOD_STEP, 'multiband-gauge-covariant-response-bridge', "bridge", ScientificInputRole.QUALIFICATION),
        (CONSTRUCTIVE_SEARCH_CONFORMANCE_STEP, 'constructive-search-conformance', 'constructive_search', ScientificInputRole.QUALIFICATION),
    )
    for producer, output_id, consumer_input, role in final_edges:
        edges.append(
            _edge(
                protocol,
                producer=producer,
                output_id=output_id,
                consumer=CLOSEOUT_STEP,
                consumer_input=consumer_input,
                role=role,
                maximum_size_bytes=maximum,
            )
        )
    for index in range(1, 10):
        edges.append(
            _edge(
                protocol,
                producer=METHOD_STEP,
                output_id=f'gauge-covariant-response-observation-{index:02d}',
                consumer=CLOSEOUT_STEP,
                consumer_input=f'gauge-covariant-response-calibration-{index:02d}',
                role=ScientificInputRole.QUALIFICATION,
                maximum_size_bytes=maximum,
            )
        )
    for task_id, profile in WORKFLOW_TASKS.items():
        if profile.source.value == "EPW_SUPERCONDUCTING_REFERENCE":
            output_id = "tutorial-reproduction"
            input_id = f'tutorial-{profile.prefix}'
        elif profile.structure_id in {'structure.calibration-pb-fcc', 'structure.calibration-mgb2-alb2'}:
            output_id = 'material-gauge-covariant-response-result'
            input_id = f"material-gauge-covariant-response-{profile.prefix}-{profile.view_id.removeprefix('view.')}"
        else:
            continue
        edges.append(
            _edge(
                protocol,
                producer=task_id,
                output_id=output_id,
                consumer=CLOSEOUT_STEP,
                consumer_input=input_id,
                role=ScientificInputRole.OUTCOME,
                maximum_size_bytes=maximum,
            )
        )
    for panel_task in PANEL_TASKS:
        edges.append(
            _edge(
                protocol,
                producer=panel_task,
                output_id="control-panel",
                consumer=CLOSEOUT_STEP,
                consumer_input=f"panel-{PANEL_TASKS[panel_task].removeprefix('structure.')}",
                role=ScientificInputRole.OUTCOME,
                maximum_size_bytes=maximum,
            )
        )
    return CandidateScientificGraph(
        graph_id='graph.ambient-pressure-superconductor-material-control-control-science-freeze',
        external_inputs=(external,),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def material_control_obligation_coverage(
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
    fallback = (CLOSEOUT_STEP, 'material-control-closeout')
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
        coverage_id='obligation-coverage.ambient-pressure-superconductor-material-control',
        bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
    )


def material_control_template(
    *,
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    config: MaterialControlConfig,
    source_contract: MaterialControlControlBundleQualification,
) -> StudyTemplate:
    graph = material_control_graph(
        protocol=protocol,
        registry=registry,
        config=config,
        source_contract=source_contract,
    )
    return StudyTemplate(
        template_key='ambient-pressure-superconductor.material-control-control-science-freeze',
        template_version="1.0.0",
        protocol=protocol,
        graph=graph,
        coverage=material_control_obligation_coverage(
            experiment=experiment,
            protocol=protocol,
            graph=graph,
        ),
    )


def material_control_candidate_catalog(
    *, registry: CapabilityRegistry, template: StudyTemplate
) -> CandidateCapabilityCatalog:
    registrations = tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=manifest.capability_key,
            provider_version=manifest.capability_version,
            config_media_type="application/json",
            maximum_config_bytes=MATERIAL_CONTROL_MAXIMUM_CONFIG_BYTES,
        )
        for manifest in registry.capabilities
    )
    return CandidateCapabilityCatalog(
        catalog_id='candidate-catalog.ambient-pressure-superconductor-material-control',
        registrations=tuple(sorted(registrations, key=lambda value: value.manifest.registry_id)),
        templates=(template,),
    )


__all__ = [
    'material_control_candidate_catalog',
    'material_control_graph',
    'material_control_obligation_coverage',
    'material_control_template',
]
