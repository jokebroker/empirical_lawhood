"""Reusable formal-analysis augmentation for standard experiment DAGs."""

from __future__ import annotations

from dataclasses import replace

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.planning.formal_gaps import FormalDomain, FormalGapRegister
from empirical_lawhood.planning.formal_results import (
    FormalAnalysisInputManifest,
    FormalDomainAnalysisResult,
    FormalGapAdjudicationPanel,
)
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ObligationCoverage, ObligationCoverageBinding, StudyTemplate
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
    CapabilityPermission,
)
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificInputRole,
    ScientificStage,
)

from .formal_analysis import (
    FORMAL_DOMAIN_CAPABILITY_KEYS,
    FORMAL_PANEL_CAPABILITY_KEY,
)


def formal_gap_obligation_id(gap_id: str) -> str:
    return f"formal-obligation.{gap_id}"


def add_standard_formal_analysis(
    *,
    base: StudyTemplate,
    register: FormalGapRegister,
    manifests: tuple[CapabilityManifest, ...],
    domain_configs: tuple[tuple[FormalDomain, CapabilityConfigRef], ...],
    panel_config: CapabilityConfigRef,
    formal_input_step_id: str,
    report_step_id: str,
    sealed_evaluation_input: CandidateGraphExternalInput | None = None,
    sealed_evaluation_step_id: str | None = None,
    sealed_evaluation_output_id: str | None = None,
    sealed_evaluation_output_ids: tuple[tuple[FormalDomain, str], ...] = (),
    formal_panel_consumer_step_ids: tuple[str, ...] = (),
) -> StudyTemplate:
    """Add four development methods and one sole-reveal panel evaluator.

    The substrate-owned ``formal_input_step_id`` must emit one standard input
    manifest per domain.  A claim-bearing campaign supplies a receipt-bound
    sealed producer step; the external-input form is retained for compatibility
    fixtures.  Development methods cannot read either sealed form.
    """

    manifests_by_key = {value.capability_key: value for value in manifests}
    required_keys = {
        *FORMAL_DOMAIN_CAPABILITY_KEYS.values(),
        FORMAL_PANEL_CAPABILITY_KEY,
    }
    if not required_keys.issubset(manifests_by_key):
        raise ValueError("formal protocol augmentation lacks registered manifests")
    configs = dict(domain_configs)
    if set(configs) != set(FormalDomain):
        raise ValueError("formal protocol augmentation requires four domain configs")
    steps = {value.step_id: value for value in base.protocol.steps}
    nodes = {value.node_id: value for value in base.graph.nodes}
    input_step = steps.get(formal_input_step_id)
    report_step = steps.get(report_step_id)
    if input_step is None or report_step is None:
        raise ValueError("formal input/report step is absent from the base template")
    input_outputs = {value.output_id: value for value in input_step.outputs}
    expected_output_ids = {
        domain: f"formal-input-{domain.value.lower()}" for domain in FormalDomain
    }
    for domain, output_id in expected_output_ids.items():
        output = input_outputs.get(output_id)
        if output is None or output.payload_schema != FormalAnalysisInputManifest.SCHEMA:
            raise ValueError(f"formal input step lacks the {domain.value} standard manifest output")
    internal_sealed = sealed_evaluation_step_id is not None
    if sealed_evaluation_output_id is not None and sealed_evaluation_output_ids:
        raise ValueError("sealed formal producer output forms are mutually exclusive")
    has_internal_outputs = sealed_evaluation_output_id is not None or bool(
        sealed_evaluation_output_ids
    )
    if internal_sealed != has_internal_outputs:
        raise ValueError("sealed formal producer requires its step and output IDs")
    if internal_sealed == (sealed_evaluation_input is not None):
        raise ValueError("formal protocol requires exactly one sealed input form")
    sealed_step = None
    sealed_outputs: tuple[tuple[FormalDomain | None, OutputTemplate], ...] = ()
    sealed_bindings: tuple[
        tuple[
            str,
            str | None,
            str | None,
            str | None,
            str,
            str,
            str,
            str,
            int,
            OutcomeAccess,
            VisibilityCeiling,
        ],
        ...,
    ]
    if internal_sealed:
        sealed_step = steps.get(sealed_evaluation_step_id or "")
        if sealed_step is None:
            raise ValueError("sealed formal producer step is absent")
        requested_outputs = (
            ((None, sealed_evaluation_output_id),)
            if sealed_evaluation_output_id is not None
            else tuple(sealed_evaluation_output_ids)
        )
        if sealed_evaluation_output_ids and {
            domain for domain, _ in sealed_evaluation_output_ids
        } != set(FormalDomain):
            raise ValueError("sealed formal producer requires all four domains")
        outputs_by_id = {value.output_id: value for value in sealed_step.outputs}
        sealed_outputs = tuple(
            (domain, outputs_by_id[output_id])
            for domain, output_id in requested_outputs
            if output_id in outputs_by_id
        )
        if (
            len(sealed_outputs) != len(requested_outputs)
            or any(
                output.payload_schema != FormalAnalysisInputManifest.SCHEMA
                for _, output in sealed_outputs
            )
            or sealed_step.requested_outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("sealed formal producer has another schema or access")
    else:
        assert sealed_evaluation_input is not None
        if sealed_evaluation_input.payload_schema != FormalAnalysisInputManifest.SCHEMA:
            raise ValueError("sealed formal evaluation input has another schema")
        if sealed_evaluation_input.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("formal evaluation input must remain sealed")
        if sealed_evaluation_input.input_id in {
            value.input_id for value in base.graph.external_inputs
        }:
            raise ValueError("formal evaluation input duplicates a base external input")

    new_steps = list(base.protocol.steps)
    new_nodes = list(base.graph.nodes)
    new_edges = list(base.graph.edges)
    new_bindings = list(base.coverage.bindings)
    domain_step_ids: list[str] = []
    domain_edge_ids: list[str] = []
    for domain in FormalDomain:
        key = FORMAL_DOMAIN_CAPABILITY_KEYS[domain]
        manifest = manifests_by_key[key]
        step_id = f"formal-{domain.value.lower()}"
        output_id = f"formal-{domain.value.lower()}-result"
        domain_step_ids.append(step_id)
        step = ProtocolStepTemplate(
            step_id=step_id,
            stage=ScientificStage.DEVELOP,
            capability_key=manifest.capability_key,
            capability_version=manifest.capability_version,
            config=configs[domain],
            dependency_step_ids=(formal_input_step_id,),
            outputs=(
                OutputTemplate(
                    output_id=output_id,
                    payload_schema=FormalDomainAnalysisResult.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/json",
                    filename_suffix=".json",
                ),
            ),
            required_permissions=tuple(
                sorted(
                    (
                        CapabilityPermission.READ_DEVELOPMENT,
                        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                    ),
                    key=lambda value: value.value,
                )
            ),
            requested_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility_ceiling=input_step.visibility_ceiling,
            resource_budget=manifest.resource_ceiling,
            resource_lock_ids=("lock.formal-analysis",),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=(f"formal-domain-analysis.{domain.value.lower()}",),
        )
        new_steps.append(step)
        new_nodes.append(
            CandidateGraphNode(
                node_id=step_id,
                stage=step.stage,
                capability_key=step.capability_key,
                capability_version=step.capability_version,
                implementation_sha256=manifest.implementation_sha256,
                protocol_step_sha256=step.fingerprint(),
                obligation_ids=step.obligation_ids,
                outcome_access=step.requested_outcome_access,
                visibility_ceiling=step.visibility_ceiling,
                resource_budget=step.resource_budget,
            )
        )
        edge_id = f"edge.{formal_input_step_id}.{step_id}"
        domain_edge_ids.append(edge_id)
        new_edges.append(
            CandidateGraphEdge(
                edge_id=edge_id,
                producer_node_id=formal_input_step_id,
                producer_output_id=expected_output_ids[domain],
                external_input_id=None,
                consumer_node_id=step_id,
                consumer_input_id="formal-development-input",
                scientific_role=ScientificInputRole.OUTCOME,
                logical_artifact_id=f"artifact.{expected_output_ids[domain]}",
                payload_schema=FormalAnalysisInputManifest.SCHEMA,
                media_type="application/json",
                maximum_size_bytes=manifest.resource_ceiling.source_scan_bytes,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                visibility_ceiling=input_step.visibility_ceiling,
                barrier=BarrierKind.NONE,
            )
        )
        new_bindings.append(
            ObligationCoverageBinding(
                obligation_id=step.obligation_ids[0],
                proof_owner_node_id=step_id,
                required_output_id=output_id,
                contributor_edge_ids=(edge_id,),
            )
        )

    panel_manifest = manifests_by_key[FORMAL_PANEL_CAPABILITY_KEY]
    panel_step_id = "formal-panel-evaluator"
    panel_output_id = "formal-gap-panel"
    gap_obligations = tuple(
        sorted(formal_gap_obligation_id(value.gap_id) for value in register.gaps)
    )
    panel_step = ProtocolStepTemplate(
        step_id=panel_step_id,
        stage=ScientificStage.EVALUATE,
        capability_key=panel_manifest.capability_key,
        capability_version=panel_manifest.capability_version,
        config=panel_config,
        dependency_step_ids=tuple(
            sorted(
                (
                    *domain_step_ids,
                    *((sealed_evaluation_step_id,) if internal_sealed else ()),
                )
            )
        ),
        outputs=(
            OutputTemplate(
                output_id=panel_output_id,
                payload_schema=FormalGapAdjudicationPanel.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                filename_suffix=".json",
            ),
        ),
        required_permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.READ_SEALED_OUTCOMES,
                    CapabilityPermission.REVEAL_OUTCOMES,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                ),
                key=lambda value: value.value,
            )
        ),
        requested_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=input_step.visibility_ceiling,
        resource_budget=panel_manifest.resource_ceiling,
        resource_lock_ids=("lock.formal-panel-reveal",),
        barrier=BarrierKind.REVEAL,
        maximum_attempts=1,
        obligation_ids=tuple(sorted(("formal-panel-complete", *gap_obligations))),
    )
    new_steps.append(panel_step)
    new_nodes.append(
        CandidateGraphNode(
            node_id=panel_step_id,
            stage=panel_step.stage,
            capability_key=panel_step.capability_key,
            capability_version=panel_step.capability_version,
            implementation_sha256=panel_manifest.implementation_sha256,
            protocol_step_sha256=panel_step.fingerprint(),
            obligation_ids=panel_step.obligation_ids,
            outcome_access=panel_step.requested_outcome_access,
            visibility_ceiling=panel_step.visibility_ceiling,
            resource_budget=panel_step.resource_budget,
        )
    )
    panel_incoming: list[str] = []
    for domain in FormalDomain:
        edge_id = f"edge.formal-{domain.value.lower()}.{panel_step_id}"
        panel_incoming.append(edge_id)
        new_edges.append(
            CandidateGraphEdge(
                edge_id=edge_id,
                producer_node_id=f"formal-{domain.value.lower()}",
                producer_output_id=f"formal-{domain.value.lower()}-result",
                external_input_id=None,
                consumer_node_id=panel_step_id,
                consumer_input_id=f"formal-{domain.value.lower()}-result",
                scientific_role=ScientificInputRole.OUTCOME,
                logical_artifact_id=f"artifact.formal-{domain.value.lower()}-result",
                payload_schema=FormalDomainAnalysisResult.SCHEMA,
                media_type="application/json",
                maximum_size_bytes=panel_manifest.resource_ceiling.source_scan_bytes,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                visibility_ceiling=input_step.visibility_ceiling,
                barrier=BarrierKind.REVEAL,
            )
        )
    if internal_sealed:
        assert sealed_step is not None
        sealed_bindings = tuple(
            (
                (f"edge.{sealed_step.step_id}.{output.output_id}.{panel_step_id}"),
                sealed_step.step_id,
                output.output_id,
                None,
                f"formal-sealed-evaluation-{domain.value.lower()}"
                if domain is not None
                else "formal-sealed-evaluation-input",
                f"artifact.{sealed_step.step_id}.{output.output_id}",
                output.payload_schema,
                output.media_type,
                panel_manifest.resource_ceiling.source_scan_bytes,
                sealed_step.requested_outcome_access,
                sealed_step.visibility_ceiling,
            )
            for domain, output in sealed_outputs
        )
    else:
        assert sealed_evaluation_input is not None
        sealed_bindings = (
            (
                f"edge.{sealed_evaluation_input.input_id}.{panel_step_id}",
                None,
                None,
                sealed_evaluation_input.input_id,
                "formal-sealed-evaluation-input",
                sealed_evaluation_input.logical_artifact_id,
                sealed_evaluation_input.payload_schema,
                sealed_evaluation_input.media_type,
                sealed_evaluation_input.maximum_size_bytes,
                sealed_evaluation_input.outcome_access,
                sealed_evaluation_input.visibility_ceiling,
            ),
        )
    for (
        sealed_edge_id,
        sealed_producer_node_id,
        sealed_producer_output_id,
        sealed_external_input_id,
        sealed_consumer_input_id,
        sealed_logical_artifact_id,
        sealed_payload_schema,
        sealed_media_type,
        sealed_maximum_size_bytes,
        sealed_outcome_access,
        sealed_visibility_ceiling,
    ) in sealed_bindings:
        panel_incoming.append(sealed_edge_id)
        new_edges.append(
            CandidateGraphEdge(
                edge_id=sealed_edge_id,
                producer_node_id=sealed_producer_node_id,
                producer_output_id=sealed_producer_output_id,
                external_input_id=sealed_external_input_id,
                consumer_node_id=panel_step_id,
                consumer_input_id=sealed_consumer_input_id,
                scientific_role=ScientificInputRole.OUTCOME,
                logical_artifact_id=sealed_logical_artifact_id,
                payload_schema=sealed_payload_schema,
                media_type=sealed_media_type,
                maximum_size_bytes=sealed_maximum_size_bytes,
                outcome_access=sealed_outcome_access,
                visibility_ceiling=sealed_visibility_ceiling,
                barrier=BarrierKind.REVEAL,
            )
        )
    for obligation_id in panel_step.obligation_ids:
        new_bindings.append(
            ObligationCoverageBinding(
                obligation_id=obligation_id,
                proof_owner_node_id=panel_step_id,
                required_output_id=panel_output_id,
                contributor_edge_ids=tuple(sorted(panel_incoming)),
            )
        )

    updated_report = replace(
        report_step,
        dependency_step_ids=tuple(sorted({*report_step.dependency_step_ids, panel_step_id})),
    )
    new_steps = [
        updated_report if value.step_id == report_step_id else value for value in new_steps
    ]
    report_manifest_node = nodes[report_step_id]
    new_nodes = [
        (
            replace(
                value,
                protocol_step_sha256=updated_report.fingerprint(),
            )
            if value.node_id == report_step_id
            else value
        )
        for value in new_nodes
    ]
    del report_manifest_node
    report_edge_id = f"edge.{panel_step_id}.{report_step_id}"
    new_edges.append(
        CandidateGraphEdge(
            edge_id=report_edge_id,
            producer_node_id=panel_step_id,
            producer_output_id=panel_output_id,
            external_input_id=None,
            consumer_node_id=report_step_id,
            consumer_input_id=panel_output_id,
            scientific_role=ScientificInputRole.OUTCOME,
            logical_artifact_id="artifact.formal-gap-panel",
            payload_schema=FormalGapAdjudicationPanel.SCHEMA,
            media_type="application/json",
            maximum_size_bytes=panel_manifest.resource_ceiling.output_bytes,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=input_step.visibility_ceiling,
            barrier=BarrierKind.REVEAL,
        )
    )

    for consumer_step_id in formal_panel_consumer_step_ids:
        if consumer_step_id == report_step_id:
            raise ValueError("report is already a standard formal-panel consumer")
        consumer = next(
            (value for value in new_steps if value.step_id == consumer_step_id),
            None,
        )
        if consumer is None:
            raise ValueError("additional formal-panel consumer step is absent")
        updated_consumer = replace(
            consumer,
            dependency_step_ids=tuple(sorted({*consumer.dependency_step_ids, panel_step_id})),
        )
        new_steps = [
            updated_consumer if value.step_id == consumer_step_id else value for value in new_steps
        ]
        new_nodes = [
            (
                replace(
                    value,
                    protocol_step_sha256=updated_consumer.fingerprint(),
                )
                if value.node_id == consumer_step_id
                else value
            )
            for value in new_nodes
        ]
        new_edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{panel_step_id}.{consumer_step_id}",
                producer_node_id=panel_step_id,
                producer_output_id=panel_output_id,
                external_input_id=None,
                consumer_node_id=consumer_step_id,
                consumer_input_id=panel_output_id,
                scientific_role=ScientificInputRole.OUTCOME,
                logical_artifact_id="artifact.formal-gap-panel",
                payload_schema=FormalGapAdjudicationPanel.SCHEMA,
                media_type="application/json",
                maximum_size_bytes=panel_manifest.resource_ceiling.output_bytes,
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                visibility_ceiling=input_step.visibility_ceiling,
                barrier=consumer.barrier,
            )
        )

    protocol = ProtocolTemplate(
        template_id=f"{base.protocol.template_id}.formal-standard",
        template_version=base.protocol.template_version,
        steps=tuple(sorted(new_steps, key=lambda value: value.step_id)),
        requires_model_set=base.protocol.requires_model_set,
        requests_controller=base.protocol.requests_controller,
        nonactuating=base.protocol.nonactuating,
    )
    graph = CandidateScientificGraph(
        graph_id=f"{base.graph.graph_id}.formal-standard",
        external_inputs=tuple(
            sorted(
                (
                    *base.graph.external_inputs,
                    *((sealed_evaluation_input,) if not internal_sealed else ()),
                ),
                key=lambda value: value.input_id,
            )
        ),
        nodes=tuple(sorted(new_nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(new_edges, key=lambda value: value.edge_id)),
    )
    coverage = ObligationCoverage(
        coverage_id=f"{base.coverage.coverage_id}.formal-standard",
        bindings=tuple(sorted(new_bindings, key=lambda value: value.obligation_id)),
    )
    return StudyTemplate(
        template_key=f"{base.template_key}.formal-standard",
        template_version=base.template_version,
        protocol=protocol,
        graph=graph,
        coverage=coverage,
    )


__all__ = [
    "add_standard_formal_analysis",
    "formal_gap_obligation_id",
]
