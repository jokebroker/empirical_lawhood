"""Adapter-owned authoring of the unfinished C graph using retained operands."""

from dataclasses import replace

from empirical_lawhood.adapters.methods.reactor_regime_response.continuation import RegimeCContinuation
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphExternalInput, ContentIdentityPolicy, StudyTemplate, required_candidate_obligation_ids


def bind_c_continuation(
    template: StudyTemplate, experiment: ExperimentSpec, continuation: RegimeCContinuation,
) -> StudyTemplate:
    """Remove completed producers and preserve each original consumer/barrier."""
    completed = set(continuation.completed_task_ids)
    originals = {(row.task_id, row.output_id): row for row in continuation.inputs}
    protocol = replace(template.protocol, steps=tuple(
        replace(step, dependency_step_ids=tuple(value for value in step.dependency_step_ids if value not in completed))
        for step in template.protocol.steps if step.step_id not in completed
    ))
    steps = {step.step_id: step for step in protocol.steps}
    if len(steps) != 132 or sum(key.startswith("regime.assay.") for key in steps) != 64:
        raise ValueError("retained C graph changed its 132 remaining tasks/64 qualification assays")
    external = {value.input_id: value for value in template.graph.external_inputs}
    edges = []
    for edge in template.graph.edges:
        if edge.consumer_node_id in completed:
            continue
        if edge.producer_node_id in completed:
            assert edge.producer_node_id is not None and edge.producer_output_id is not None
            row = originals[(edge.producer_node_id, edge.producer_output_id)]
            artifact, logical = row.artifact, row.manifest.logical
            if edge.payload_schema != artifact.payload_schema:
                raise ValueError("retained C graph changed an original output schema")
            external[artifact.artifact_id] = CandidateGraphExternalInput(
                artifact.artifact_id, edge.scientific_role, artifact.artifact_id,
                ContentIdentityPolicy.EXACT_SHA256, artifact.sha256, artifact.payload_schema,
                artifact.media_type, artifact.size_bytes, logical.outcome_access, logical.visibility_ceiling,
            )
            edge = replace(
                edge, producer_node_id=None, producer_output_id=None,
                external_input_id=artifact.artifact_id, logical_artifact_id=artifact.artifact_id,
                maximum_size_bytes=artifact.size_bytes, outcome_access=logical.outcome_access,
                visibility_ceiling=logical.visibility_ceiling,
            )
        edges.append(edge)
    used = {edge.external_input_id for edge in edges if edge.external_input_id is not None}
    if not {row.artifact.artifact_id for row in continuation.inputs}.issubset(used):
        raise ValueError("C continuation omitted a completed operand")
    graph = replace(
        template.graph,
        nodes=tuple(replace(node, protocol_step_sha256=steps[node.node_id].fingerprint())
                    for node in template.graph.nodes if node.node_id in steps),
        external_inputs=tuple(external[key] for key in sorted(used)),
        edges=tuple(edges),
    )
    required = set(required_candidate_obligation_ids(experiment, protocol))
    coverage = replace(template.coverage, bindings=tuple(
        replace(binding, contributor_edge_ids=tuple(
            edge.edge_id for edge in graph.edges if edge.consumer_node_id == binding.proof_owner_node_id
        )) for binding in template.coverage.bindings if binding.obligation_id in required
    ))
    if any(binding.proof_owner_node_id not in steps for binding in coverage.bindings):
        raise ValueError("retained C graph lost an active scientific obligation owner")
    return replace(template, protocol=protocol, graph=graph, coverage=coverage)
