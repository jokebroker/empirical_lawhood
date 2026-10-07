# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from empirical_lawhood.api import load_authoring
from empirical_lawhood.api.models import CampaignPackage
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling, inherited_visibility
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.campaigns import CampaignNode
from empirical_lawhood.planning.study_authoring import CapabilitySelection, ConditionalGateKind, ConditionalChildRequest, ConditionalTerminalDisposition, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole, UnresolvedDecision
from empirical_lawhood.runtime.candidate_compiler import AuthoringMaterializationIdentity, CandidateCompilationContext, CandidateCompilationDisposition, CandidateDiagnosticCode, CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole, compile_draft_candidate, required_candidate_obligation_ids
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.conditional_children import bind_frozen_parent_input, instantiate_conditional_child
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolTemplate
from empirical_lawhood.runtime.recovery import ProtocolRunRecoveryIndex, TaskRecoveryDisposition, ProtocolTaskRecoveryEvent


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PATH = REPO_ROOT / "tests/fixtures/reference-campaign.json"
SOURCE_ID = "source.reference-medium"


def _pending_components() -> tuple[CampaignPackage, object, object]:
    package = load_authoring(PACKAGE_PATH)
    assert isinstance(package, CampaignPackage)
    experiment = package.frozen_proposal.proposal.candidate_experiment
    identity = ObjectIdentity.from_record(experiment.experiment_id, experiment)
    nodes = tuple(
        replace(node, object_identity=identity)
        if node.object_identity.object_id == experiment.experiment_id
        else node
        for node in package.campaign.nodes
    )
    assert all(isinstance(node, CampaignNode) for node in nodes)
    campaign = replace(
        package.campaign,
        nodes=nodes,
        evidence_state=(identity,),
    )
    return package, experiment, campaign


def _graph_for(package: CampaignPackage) -> CandidateScientificGraph:
    manifests = {value.registry_id: value for value in package.registry.capabilities}
    nodes = tuple(
        sorted(
            (
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
                for step in package.protocol.steps
            ),
            key=lambda value: value.node_id,
        )
    )
    output_by_step = {step.step_id: step.outputs[0] for step in package.protocol.steps}
    edges: list[CandidateGraphEdge] = [
        CandidateGraphEdge(
            edge_id="edge.source.prepare",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=SOURCE_ID,
            consumer_node_id="prepare",
            consumer_input_id="source",
            scientific_role=ScientificInputRole.PREPARED_MEDIUM,
            logical_artifact_id="artifact.source-placeholder",
            payload_schema='empirical-lawhood/testing/fixtures/reference-source',
            media_type="application/json",
            maximum_size_bytes=10_000,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
        )
    ]
    for step in package.protocol.steps:
        for parent_id in step.dependency_step_ids:
            output = output_by_step[parent_id]
            edges.append(
                CandidateGraphEdge(
                    edge_id=f"edge.{parent_id}.{step.step_id}",
                    producer_node_id=parent_id,
                    producer_output_id=output.output_id,
                    external_input_id=None,
                    consumer_node_id=step.step_id,
                    consumer_input_id=f"input-{parent_id}",
                    scientific_role=ScientificInputRole.OUTCOME,
                    logical_artifact_id=f"artifact.{parent_id}.{output.output_id}",
                    payload_schema=output.payload_schema,
                    media_type=output.media_type,
                    maximum_size_bytes=10_000,
                    outcome_access=step.requested_outcome_access,
                    visibility_ceiling=inherited_visibility(
                        (step.visibility_ceiling,), step.requested_outcome_access
                    ),
                    barrier=step.barrier,
                )
            )
    return CandidateScientificGraph(
        graph_id="graph.reference-prospective",
        external_inputs=(
            CandidateGraphExternalInput(
                input_id=SOURCE_ID,
                scientific_role=ScientificInputRole.PREPARED_MEDIUM,
                logical_artifact_id="artifact.source-placeholder",
                content_identity_policy=ContentIdentityPolicy.RECEIPT_BOUND,
                expected_content_sha256=None,
                payload_schema='empirical-lawhood/testing/fixtures/reference-source',
                media_type="application/json",
                maximum_size_bytes=10_000,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            ),
        ),
        nodes=nodes,
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _coverage_for(
    experiment: object,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> ObligationCoverage:
    assert hasattr(experiment, "obligations")
    incoming = {
        node.node_id: tuple(
            sorted(edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id)
        )
        for node in graph.nodes
    }
    step_owners = {
        obligation_id: (step.step_id, step.outputs[0].output_id)
        for step in protocol.steps
        for obligation_id in step.obligation_ids
    }
    bindings = []
    for obligation_id in required_candidate_obligation_ids(experiment, protocol):
        node_id, output_id = step_owners.get(
            obligation_id,
            ("report", "report"),
        )
        bindings.append(
            ObligationCoverageBinding(
                obligation_id=obligation_id,
                proof_owner_node_id=node_id,
                required_output_id=output_id,
                contributor_edge_ids=incoming[node_id],
            )
        )
    return ObligationCoverage(
        coverage_id="coverage.reference-prospective",
        bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
    )


def _conditional_template(package: CampaignPackage) -> StudyTemplate:
    source_step = next(value for value in package.protocol.steps if value.step_id == "evaluate")
    step = replace(
        source_step,
        step_id="conditional-evaluate",
        dependency_step_ids=(),
        obligation_ids=("conditional-prospective-controller-evaluation-evaluation",),
    )
    manifest = package.registry.resolve(
        step.capability_key,
        step.capability_version,
    )
    graph = CandidateScientificGraph(
        graph_id="graph.reference-conditional",
        external_inputs=(
            CandidateGraphExternalInput(
                input_id="parent-controller-admission-receipt",
                scientific_role=ScientificInputRole.PARENT_RECEIPT,
                logical_artifact_id="artifact.parent-controller-admission-receipt",
                content_identity_policy=ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION,
                expected_content_sha256=None,
                payload_schema='empirical-lawhood/runtime/protocol-task-recovery-event',
                media_type="application/json",
                maximum_size_bytes=10_000,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            ),
        ),
        nodes=(
            CandidateGraphNode(
                node_id=step.step_id,
                stage=step.stage,
                capability_key=step.capability_key,
                capability_version=step.capability_version,
                implementation_sha256=manifest.implementation_sha256,
                protocol_step_sha256=step.fingerprint(),
                obligation_ids=step.obligation_ids,
                outcome_access=step.requested_outcome_access,
                visibility_ceiling=step.visibility_ceiling,
                resource_budget=step.resource_budget,
            ),
        ),
        edges=(
            CandidateGraphEdge(
                edge_id="edge.parent-controller-admission-receipt.conditional-evaluate",
                producer_node_id=None,
                producer_output_id=None,
                external_input_id="parent-controller-admission-receipt",
                consumer_node_id="conditional-evaluate",
                consumer_input_id="parent-controller-admission-receipt",
                scientific_role=ScientificInputRole.PARENT_RECEIPT,
                logical_artifact_id="artifact.parent-controller-admission-receipt",
                payload_schema='empirical-lawhood/runtime/protocol-task-recovery-event',
                media_type="application/json",
                maximum_size_bytes=10_000,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                barrier=BarrierKind.REVEAL,
            ),
        ),
    )
    return StudyTemplate(
        template_key='reference.conditional-controller-evaluation',
        template_version="1.0.0",
        protocol=ProtocolTemplate(
            template_id="protocol.reference-conditional",
            template_version="1.0.0",
            steps=(step,),
            requires_model_set=False,
            requests_controller=False,
            nonactuating=True,
        ),
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id="coverage.reference-conditional",
            bindings=(
                ObligationCoverageBinding(
                    obligation_id="conditional-prospective-controller-evaluation-evaluation",
                    proof_owner_node_id="conditional-evaluate",
                    required_output_id=step.outputs[0].output_id,
                    contributor_edge_ids=("edge.parent-controller-admission-receipt.conditional-evaluate",),
                ),
            ),
        ),
    )


def _fixture(
    *,
    conditional: bool = False,
) -> tuple[StudyDraft, CandidateCompilationContext]:
    package, experiment, campaign = _pending_components()
    assert hasattr(experiment, "information_cutoffs")
    graph = _graph_for(package)
    template = StudyTemplate(
        template_key="reference.prospective",
        template_version="1.0.0",
        protocol=package.protocol,
        graph=graph,
        coverage=_coverage_for(experiment, package.protocol, graph),
    )
    source_materialization = ObjectIdentity(
        object_id="materialization.reference-medium",
        object_schema='empirical-lawhood/testing/fixtures/reference-source-materialization',
        object_version="1.0.0",
        object_fingerprint="1" * 64,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id="qualification.reference-medium",
        source_id=SOURCE_ID,
        materialization=source_materialization,
        content_sha256="2" * 64,
        evidence_world_id=package.system.world.world_id,
        observation_operator=ObjectIdentity(
            object_id="observation.reference",
            object_schema='empirical-lawhood/testing/fixtures/reference-observation',
            object_version="1.0.0",
            object_fingerprint="3" * 64,
        ),
        numerical_view_ids=tuple(value.view_id for value in package.system.numerical_views),
        native_unit_ids=tuple(sorted({value.native_unit for value in package.system.quantities})),
        frame_ids=tuple(sorted({value.coordinate_frame for value in package.system.quantities})),
        clock_ids=tuple(value.clock_id for value in package.system.clocks),
        receiver_semantics_id=package.system.relation.relation_id,
        validity_contract_id=experiment.obligations.validity.validity_id,
        uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
        access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    design_input = DesignInputRecord(
        input_id="input.reference-theory",
        object_identity=ObjectIdentity.from_record(
            package.system.system_id,
            package.system,
        ),
        materialization_sha256="4" * 64,
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.owner",
    )
    templates = [template]
    successor = None
    if conditional:
        templates.append(_conditional_template(package))
        successor = ConditionalChildRequest(
            request_id='conditional.reference-controller-evaluation',
            parent_node_id="evaluate",
            parent_receipt_input_id="parent-controller-admission-receipt",
            gate_kind=ConditionalGateKind.ADMISSION_AND_REACHABILITY_PASSED,
            template_key='reference.conditional-controller-evaluation',
            evaluation_unit_ids=("unit.eval-01",),
            eligible_disposition=ConditionalTerminalDisposition.EXECUTION_ELIGIBLE,
            ineligible_disposition=(
                ConditionalTerminalDisposition.STOPPED_BY_FROZEN_SCIENTIFIC_GATE
            ),
        )
    candidate_capabilities = tuple(
        sorted(
            (
                replace(
                    value,
                    input_schema_ids=tuple(
                        sorted(
                            {
                                *value.input_schema_ids,
                                *(
                                    {'empirical-lawhood/testing/fixtures/reference-source'}
                                    if value.capability_key == "reference.prepare"
                                    else set()
                                ),
                                *(
                                    {'empirical-lawhood/runtime/protocol-task-recovery-event'}
                                    if value.capability_key == "reference.evaluate"
                                    else set()
                                ),
                            }
                        )
                    ),
                )
                for value in package.registry.capabilities
            ),
            key=lambda value: value.registry_id,
        )
    )
    context = CandidateCompilationContext(
        context_id="context.reference-candidate",
        registry=replace(package.registry, capabilities=candidate_capabilities),
        templates=tuple(sorted(templates, key=lambda value: value.template_key)),
        qualifications=(qualification,),
        known_design_inputs=(design_input,),
        implementation_sha256="5" * 64,
    )
    draft = StudyDraft(
        draft_id="draft.reference-candidate",
        lifecycle=StudyDraftLifecycle.DRAFT,
        question="Does the frozen reference protocol discriminate its alternatives?",
        alternative_ids=("alternative.hold", "alternative.response"),
        design_origin=DesignOrigin(
            origin_id="origin.reference-theory",
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=(design_input.input_id,),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=(design_input,),
        development_unit_ids=("unit.dev-01",),
        evaluation_unit_ids=("unit.eval-01",),
        development_seed_ids=("seed.dev-01",),
        evaluation_seed_ids=("seed.eval-01",),
        unresolved_decisions=(),
        system=package.system,
        experiment=experiment,
        campaign=campaign,
        dag_template_key=template.template_key,
        capability_selections=tuple(
            sorted(
                (
                    CapabilitySelection(
                        capability_key=value.capability_key,
                        capability_version=value.capability_version,
                        implementation_sha256=value.implementation_sha256,
                    )
                    for value in package.registry.capabilities
                ),
                key=lambda value: value.selection_id,
            )
        ),
        source_materializations=(
            SourceMaterializationRef(
                source_id=SOURCE_ID,
                role=SourceMaterializationRole.PREPARED_MEDIUM,
                evidence_world_id=package.system.world.world_id,
                materialization=source_materialization,
                content_sha256=qualification.content_sha256,
                source_config_sha256="6" * 64,
                observation_operator=qualification.observation_operator,
                numerical_view_ids=qualification.numerical_view_ids,
                qualification_receipt=ObjectIdentity.from_record(
                    qualification.receipt_id,
                    qualification,
                ),
                access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            ),
        ),
        resource_ceiling=package.campaign.budget,
        conditional_successor=successor,
    )
    return draft, context


def test_exact_frozen_model_source_receipt_may_stay_sealed() -> None:
    _, context = _fixture()
    receipt = context.qualifications[0]
    assert replace(receipt, outcome_access=OutcomeAccess.EVALUATION_SEALED).outcome_access is OutcomeAccess.EVALUATION_SEALED
    with pytest.raises(ValueError, match="cannot reveal evaluation outcomes"):
        replace(receipt, outcome_access=OutcomeAccess.EVALUATOR_REVEAL)


def _materialization(raw: str = "a") -> AuthoringMaterializationIdentity:
    return AuthoringMaterializationIdentity(
        media_type="application/yaml",
        byte_count=100,
        raw_materialization_sha256=raw * 64,
    )


def test_complete_draft_compiles_to_nonauthoritative_authority_pending_candidate() -> None:
    draft, context = _fixture(conditional=True)

    report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )

    assert report.disposition is CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING
    assert report.candidate is not None
    assert report.candidate.experiment.authorization_record_id is None
    assert report.candidate.conditional_successor is not None
    assert report.candidate.required_issue_inputs == (
        "accountable-human-proposer-attestation",
        "clean-commit-or-exact-source-closure",
        "custody-publication-authority",
        "exact-materialization-revalidation",
    )
    assert report.candidate.conditional_successor.allowed_instantiation_fields == (
        "authority_id",
        "issue_id",
        "parent_receipt",
        "run_id",
        "task_id",
    )

    assert CandidateDiagnosticCode.READY_TO_ISSUE in {value.code for value in report.diagnostics}
    assert {
        value.code
        for value in report.diagnostics
        if value.diagnostic_class.value == "AUTHORITY_GATE"
    } == {
        CandidateDiagnosticCode.CUSTODY_PUBLICATION_AUTHORITY_REQUIRED,
        CandidateDiagnosticCode.SCIENTIFIC_APPROVAL_AUTHORITY_REQUIRED,
        CandidateDiagnosticCode.EXPERIMENT_EXECUTION_AUTHORITY_REQUIRED,
        CandidateDiagnosticCode.OUTCOME_REVEAL_AUTHORITY_REQUIRED,
    }
    source_input = next(
        value
        for value in report.candidate.scientific_graph.external_inputs
        if value.input_id == SOURCE_ID
    )
    assert source_input.expected_content_sha256 == "2" * 64
    assert source_input.content_identity_policy is ContentIdentityPolicy.EXACT_SHA256

    assert draft.conditional_successor is not None
    foreign_parent = replace(
        draft,
        conditional_successor=replace(
            draft.conditional_successor,
            parent_node_id="foreign-parent",
        ),
    )
    refused = compile_draft_candidate(
        draft=foreign_parent,
        authoring_materialization=_materialization(),
        context=context,
    )
    assert refused.candidate is None
    assert CandidateDiagnosticCode.SCIENTIFIC_SPEC_INCOMPLETE in {
        value.code for value in refused.diagnostics
    }


def _conditional_parent_record() -> ProtocolTaskRecoveryEvent:
    return ProtocolTaskRecoveryEvent(
        event_id="event.parent-controller-admission-terminal",
        recovery_index=ObjectIdentity(
            object_id="recovery.parent-controller-admission",
            object_schema=ProtocolRunRecoveryIndex.SCHEMA,
            object_version="1.0.0",
            object_fingerprint="9" * 64,
        ),
        run_id="run.parent-controller-admission",
        task_id="task.parent-controller-admission-terminal",
        attempt_id=None,
        attempt_ordinal=None,
        disposition=TaskRecoveryDisposition.NOT_ATTEMPTED,
        receipt_id=None,
        receipt_relative_path=None,
        receipt_sha256=None,
        receipt_schema=None,
        output_materialization_ids=(),
        reason_codes=("parent-branch-record-fixture",),
    )


def test_conditional_child_instantiates_only_the_parent_record() -> None:
    draft, context = _fixture(conditional=True)
    parent = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )

    instantiation, child = instantiate_conditional_child(
        parent_compilation=parent,
        parent_record_id="parent-record.controller-admission-terminal",
        parent_record=_conditional_parent_record(),
        eligible=True,
    )

    assert child is not None
    assert child.candidate is not None
    assert parent.candidate is not None
    assert (
        instantiation.decode_parent(
            ProtocolTaskRecoveryEvent,
            maximum_bytes=10_000,
        )
        == _conditional_parent_record()
    )
    assert child.candidate.protocol == parent.candidate.conditional_successor.protocol
    assert (
        child.candidate.scientific_graph == parent.candidate.conditional_successor.scientific_graph
    )
    assert (
        child.candidate.obligation_coverage
        == parent.candidate.conditional_successor.obligation_coverage
    )
    assert child.candidate.system == parent.candidate.system
    assert child.candidate.experiment == parent.candidate.experiment
    assert child.candidate.source_locks == parent.candidate.source_locks
    assert child.candidate.conditional_successor is None
    assert child.provisional_protocol == child.candidate.protocol
    assert child.provisional_graph == child.candidate.scientific_graph
    assert child.provisional_coverage == child.candidate.obligation_coverage
    assert child.provisional_graph_sha256 == child.candidate.scientific_graph.fingerprint()
    assert child.provisional_coverage_sha256 == child.candidate.obligation_coverage.fingerprint()


def test_conditional_child_terminal_stop_creates_no_child() -> None:
    draft, context = _fixture(conditional=True)
    parent = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )

    instantiation, child = instantiate_conditional_child(
        parent_compilation=parent,
        parent_record_id="parent-record.controller-admission-terminal",
        parent_record=_conditional_parent_record(),
        eligible=False,
        reason_codes=("controller-admission-not-admitted-or-reachable",),
    )

    assert child is None
    assert instantiation.child_candidate is None
    assert (
        instantiation.disposition
        is ConditionalTerminalDisposition.STOPPED_BY_FROZEN_SCIENTIFIC_GATE
    )


def test_frozen_parent_input_binding_changes_no_candidate_science() -> None:
    draft, context = _fixture(conditional=True)
    report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )
    assert report.candidate is not None
    successor = report.candidate.conditional_successor
    assert successor is not None
    child = replace(
        report.candidate,
        candidate_id="candidate.predeclared-child",
        protocol=successor.protocol,
        scientific_graph=successor.scientific_graph,
        obligation_coverage=successor.obligation_coverage,
        conditional_successor=None,
    )

    binding = bind_frozen_parent_input(
        candidate=child,
        external_input_id="parent-controller-admission-receipt",
        parent_record_id="parent-record.controller-admission-terminal",
        parent_record=_conditional_parent_record(),
    )

    assert binding.candidate == ObjectIdentity.from_record(child.candidate_id, child)
    assert binding.scientific_graph_sha256 == child.scientific_graph.fingerprint()
    assert (
        binding.decode_parent(
            ProtocolTaskRecoveryEvent,
            maximum_bytes=10_000,
        )
        == _conditional_parent_record()
    )


def test_candidate_is_deterministic_but_raw_materialization_identity_is_distinct() -> None:
    draft, context = _fixture()

    first = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization("a"),
        context=context,
    )
    repeated = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization("a"),
        context=context,
    )
    reformatted = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization("b"),
        context=context,
    )

    assert first.canonical_bytes() == repeated.canonical_bytes()
    assert first.semantic_config_sha256 == reformatted.semantic_config_sha256
    assert first.provisional_graph_sha256 == reformatted.provisional_graph_sha256
    assert first.candidate is not None
    assert reformatted.candidate is not None
    assert (
        first.candidate.system.canonical_bytes() == reformatted.candidate.system.canonical_bytes()
    )
    assert first.candidate.candidate_id != reformatted.candidate.candidate_id


def test_semantic_source_and_implementation_changes_stale_prior_candidate() -> None:
    draft, context = _fixture()
    baseline = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )
    assert baseline.candidate is not None

    semantic = compile_draft_candidate(
        draft=replace(draft, question=f"{draft.question} Exactly once."),
        authoring_materialization=_materialization(),
        context=context,
    )
    assert semantic.candidate is not None
    assert semantic.semantic_config_sha256 != baseline.semantic_config_sha256
    assert semantic.candidate.candidate_id != baseline.candidate.candidate_id

    source = replace(
        draft.source_materializations[0],
        content_sha256="8" * 64,
    )
    stale_source = compile_draft_candidate(
        draft=replace(draft, source_materializations=(source,)),
        authoring_materialization=_materialization(),
        context=context,
    )
    assert stale_source.candidate is None
    assert CandidateDiagnosticCode.SOURCE_QUALIFICATION_REQUIRED in {
        value.code for value in stale_source.diagnostics
    }

    implementation = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=replace(context, implementation_sha256="8" * 64),
    )
    assert implementation.candidate is not None
    assert implementation.implementation_sha256 != baseline.implementation_sha256
    assert implementation.candidate.candidate_id != baseline.candidate.candidate_id


def test_unresolved_decision_returns_classified_fatal_without_candidate() -> None:
    draft, context = _fixture()
    draft = replace(
        draft,
        unresolved_decisions=(
            UnresolvedDecision(
                decision_id="decision.threshold",
                field_path="experiment.precision_goals",
                question="Which frozen precision threshold applies?",
                option_ids=("threshold.a", "threshold.b"),
            ),
        ),
    )

    report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )

    assert report.disposition is CandidateCompilationDisposition.INVALID_DRAFT
    assert report.candidate is None
    assert CandidateDiagnosticCode.SCIENTIFIC_SPEC_INCOMPLETE in {
        value.code for value in report.diagnostics
    }


@pytest.mark.parametrize("overlap", ["unit", "seed"])
def test_development_visible_unit_or_seed_reuse_fails_closed(overlap: str) -> None:
    draft, context = _fixture()
    if overlap == "unit":
        draft = replace(draft, evaluation_unit_ids=draft.development_unit_ids)
    else:
        draft = replace(draft, evaluation_seed_ids=draft.development_seed_ids)

    report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )

    assert report.candidate is None
    assert CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED in {
        value.code for value in report.diagnostics
    }


def test_missing_qualification_is_readiness_block_not_fatal_or_authority() -> None:
    draft, context = _fixture()
    context = replace(context, qualifications=())

    report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )

    assert report.disposition is CandidateCompilationDisposition.UNRESOLVED_READINESS
    assert report.candidate is None
    assert {value.code for value in report.diagnostics} == {
        CandidateDiagnosticCode.SOURCE_QUALIFICATION_REQUIRED
    }


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        (
            "observation_operator",
            ObjectIdentity(
                object_id="observation.substituted",
                object_schema='empirical-lawhood/testing/fixtures/reference-observation',
                object_version="1.0.0",
                object_fingerprint="8" * 64,
            ),
        ),
        ("numerical_view_ids", ("view.substituted",)),
        ("native_unit_ids", ("unit.substituted",)),
        ("frame_ids", ("frame.substituted",)),
        ("clock_ids", ("clock.substituted",)),
        ("receiver_semantics_id", "receiver-semantics.substituted"),
        ("validity_contract_id", "validity.substituted"),
        ("uncertainty_contract_id", "uncertainty.substituted"),
    ),
)
def test_materialization_qualification_semantic_substitution_is_refused(
    field_name: str,
    replacement: object,
) -> None:
    draft, context = _fixture()
    qualification = replace(
        context.qualifications[0],
        **{field_name: replacement},
    )
    source = replace(
        draft.source_materializations[0],
        qualification_receipt=ObjectIdentity.from_record(
            qualification.receipt_id,
            qualification,
        ),
    )

    report = compile_draft_candidate(
        draft=replace(draft, source_materializations=(source,)),
        authoring_materialization=_materialization(),
        context=replace(context, qualifications=(qualification,)),
    )

    assert report.candidate is None
    assert CandidateDiagnosticCode.SOURCE_QUALIFICATION_REQUIRED in {
        value.code for value in report.diagnostics
    }


def test_capability_substitution_and_obligation_omission_fail_closed() -> None:
    draft, context = _fixture()
    changed_selection = replace(
        draft.capability_selections[0],
        implementation_sha256="9" * 64,
    )
    substituted = replace(
        draft,
        capability_selections=tuple(
            sorted(
                (changed_selection, *draft.capability_selections[1:]),
                key=lambda value: value.selection_id,
            )
        ),
    )
    substitution_report = compile_draft_candidate(
        draft=substituted,
        authoring_materialization=_materialization(),
        context=context,
    )
    assert CandidateDiagnosticCode.CAPABILITY_CONTRACT_MISMATCH in {
        value.code for value in substitution_report.diagnostics
    }

    template = context.templates[0]
    protocol_obligations = {
        obligation_id for step in template.protocol.steps for obligation_id in step.obligation_ids
    }
    omitted = next(
        value
        for value in template.coverage.bindings
        if value.obligation_id not in protocol_obligations
    )
    reduced = replace(
        template,
        coverage=replace(
            template.coverage,
            bindings=tuple(value for value in template.coverage.bindings if value != omitted),
        ),
    )
    reduced_context = replace(
        context,
        templates=(reduced,),
    )
    coverage_report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=reduced_context,
    )
    assert CandidateDiagnosticCode.SCIENTIFIC_OBLIGATION_UNCOVERED in {
        value.code for value in coverage_report.diagnostics
    }


def _replace_candidate_experiment(
    draft: StudyDraft,
    experiment: object,
) -> StudyDraft:
    assert draft.campaign is not None
    assert hasattr(experiment, "experiment_id")
    identity = ObjectIdentity.from_record(experiment.experiment_id, experiment)
    campaign = replace(
        draft.campaign,
        nodes=tuple(
            replace(node, object_identity=identity)
            if node.object_identity.object_id == experiment.experiment_id
            else node
            for node in draft.campaign.nodes
        ),
        evidence_state=(identity,),
    )
    return replace(draft, experiment=experiment, campaign=campaign)


def test_nomination_origin_remains_visible_but_fresh_child_can_compile() -> None:
    draft, context = _fixture()
    assert draft.experiment is not None
    motivation = replace(
        draft.design_inputs[0],
        role=DesignInputRole.MOTIVATION,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        physical_unit_ids=("unit.historical-parent",),
    )
    origin = DesignOrigin(
        origin_id="origin.reference-nomination",
        kind=DesignOriginKind.PROSPECTIVE_NOMINATION,
        declared_input_ids=(motivation.input_id,),
        parent_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        nomination=ObjectIdentity(
            object_id="nomination.reference",
            object_schema='empirical-lawhood/planning/prospective-nomination',
            object_version="1.0.0",
            object_fingerprint="7" * 64,
        ),
        requests_fresh_child=True,
    )
    experiment = replace(
        draft.experiment,
        design_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    draft = _replace_candidate_experiment(draft, experiment)
    draft = replace(
        draft,
        design_origin=origin,
        design_inputs=(motivation,),
    )
    context = replace(context, known_design_inputs=(motivation,))

    report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )

    assert report.candidate is not None
    assert report.candidate.design_origin.parent_visibility_ceiling is (
        VisibilityCeiling.OUTCOME_VISIBLE
    )
    assert all(
        claim.visibility_ceiling is VisibilityCeiling.PROSPECTIVE
        for claim in report.candidate.experiment.claims
    )

    nonchild = replace(
        draft,
        design_origin=replace(origin, requests_fresh_child=False),
    )
    refused = compile_draft_candidate(
        draft=nonchild,
        authoring_materialization=_materialization(),
        context=context,
    )
    assert refused.candidate is None
    assert CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED in {
        value.code for value in refused.diagnostics
    }


@pytest.mark.parametrize("lineage_case", ["missing", "cycle"])
def test_incomplete_or_cyclic_transitive_design_lineage_is_refused(
    lineage_case: str,
) -> None:
    draft, context = _fixture()
    first = replace(
        draft.design_inputs[0],
        parent_input_ids=("input.parent",),
    )
    if lineage_case == "missing":
        inputs = (first,)
    else:
        parent = replace(
            draft.design_inputs[0],
            input_id="input.parent",
            parent_input_ids=(first.input_id,),
        )
        inputs = tuple(sorted((first, parent), key=lambda value: value.input_id))
    draft = replace(draft, design_inputs=inputs)
    context = replace(context, known_design_inputs=inputs)

    report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )

    assert report.candidate is None
    assert CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED in {
        value.code for value in report.diagnostics
    }


def test_multiple_design_input_cutoffs_are_refused_before_current_approval_route() -> None:
    draft, context = _fixture()
    first = draft.design_inputs[0]
    second = replace(
        first,
        input_id="input.other-theory",
        information_cutoff=replace(
            first.information_cutoff,
            cutoff_id="other-information-cutoff",
        ),
    )
    inputs = tuple(sorted((first, second), key=lambda value: value.input_id))
    draft = replace(
        draft,
        design_origin=replace(
            draft.design_origin,
            declared_input_ids=tuple(value.input_id for value in inputs),
        ),
        design_inputs=inputs,
    )
    context = replace(context, known_design_inputs=inputs)

    report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )

    assert report.candidate is None
    assert any(
        value.code is CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED
        and value.field_path == "design_inputs.information_cutoff"
        for value in report.diagnostics
    )


def test_revealed_claim_input_and_advisory_operator_are_refused() -> None:
    draft, context = _fixture()
    assert draft.experiment is not None
    exposed = replace(
        draft.design_inputs[0],
        role=DesignInputRole.CLAIM_DERIVATION,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        operator_id="codex.advisory",
        physical_unit_ids=draft.evaluation_unit_ids,
    )
    experiment = replace(
        draft.experiment,
        design_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    draft = _replace_candidate_experiment(draft, experiment)
    draft = replace(draft, design_inputs=(exposed,))
    context = replace(context, known_design_inputs=(exposed,))

    report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=context,
    )

    assert report.candidate is None
    assert {value.code for value in report.diagnostics} == {
        CandidateDiagnosticCode.OUTCOME_SEPARATION_REQUIRED
    }


def test_unbounded_resource_and_stage_role_mismatch_are_fatal() -> None:
    draft, context = _fixture()
    bounded = replace(
        draft,
        resource_ceiling=ResourceBudget(
            cpu_cores=1,
            memory_bytes=1,
            gpu_devices=0,
            wall_time_seconds=1,
            source_scan_bytes=0,
            output_bytes=0,
        ),
    )
    resource_report = compile_draft_candidate(
        draft=bounded,
        authoring_materialization=_materialization(),
        context=context,
    )
    assert CandidateDiagnosticCode.RESOURCE_UNBOUNDED in {
        value.code for value in resource_report.diagnostics
    }

    prepare = context.registry.resolve("reference.prepare", "1.0.0")
    changed = replace(prepare, kind=CapabilityKind.REPORTER)
    registry = replace(
        context.registry,
        capabilities=tuple(
            sorted(
                (
                    changed if value.registry_id == changed.registry_id else value
                    for value in context.registry.capabilities
                ),
                key=lambda value: value.registry_id,
            )
        ),
    )
    role_report = compile_draft_candidate(
        draft=draft,
        authoring_materialization=_materialization(),
        context=replace(context, registry=registry),
    )
    assert CandidateDiagnosticCode.CAPABILITY_CONTRACT_MISMATCH in {
        value.code for value in role_report.diagnostics
    }


def test_template_rejects_dependency_edge_and_output_drift() -> None:
    _, context = _fixture()
    template = context.templates[0]
    edge = next(value for value in template.graph.edges if value.producer_node_id is not None)
    without_edge = replace(
        template.graph,
        edges=tuple(value for value in template.graph.edges if value.edge_id != edge.edge_id),
    )
    with pytest.raises(ValueError, match="dependencies"):
        replace(template, graph=without_edge)

    wrong_output_edge = replace(
        edge,
        producer_output_id="output.substituted",
    )
    wrong_output_graph = replace(
        template.graph,
        edges=tuple(
            sorted(
                (
                    wrong_output_edge if value.edge_id == edge.edge_id else value
                    for value in template.graph.edges
                ),
                key=lambda value: value.edge_id,
            )
        ),
    )
    with pytest.raises(ValueError, match="producer output"):
        replace(template, graph=wrong_output_graph)
