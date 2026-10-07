"""Scientific protocol and exploration lowering into immutable execution DAGs."""

from __future__ import annotations

from empirical_lawhood.kernel.retrospective_prediction_experiments import RetrospectiveExperimentSpec
from .plans import RetrospectiveStudyRunPlan, ResourceBoundRetrospectiveStudyRunPlan

import hashlib
from typing import cast

from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.experiments import ExperimentSpec, validate_experiment_against_system
from empirical_lawhood.kernel.models import ViewModelSetSpec, ModelSetSpec
from empirical_lawhood.kernel.provenance import EvidenceSnapshot, ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.approval import (
    CompleteApprovalService,
    DurableAuthorizationRecord,
    FrozenApprovalRoot,
)
from empirical_lawhood.planning.campaigns import CampaignSpec
from empirical_lawhood.planning.exploration import (
    AnalysisProposal,
    AnalysisSpec,
    ExplorationPlan,
    ExploratoryFinding,
    HypothesisSet,
    ProposalDisposition,
    ProposalSelection,
)
from empirical_lawhood.planning.discovery import SkepticReport
from empirical_lawhood.planning.datasets import ExperimentDatasetBinding

from .adjudication import ScientificAdjudicationRecord
from .artifacts import ArtifactProfile, derived_outcome_access
from .capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
    CapabilityRequirement,
)
from .candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateScientificGraph, DraftStudyCandidate
from .execution_envelope import ExecutionEnvelopeSpec, ExecutionResourceEnvelopeSpec, JitGraphSignatureManifest, PredevelopmentJitSignatureCensus, validate_execution_jit_evidence
from .exploration import ExplorationWaveInput
from .dataset_binding import dataset_external_inputs_for_stage
from .plans import ArtifactOutputSpec, BarrierKind, ProtocolExecutionPlan, CandidateExecutionPlan, EnvelopeExecutionPlan, ExecutionPlan, ProtocolExecutionTask, ExecutionTask, ExternalInputSpec, PlanLane, ProtocolStepTemplate, ProtocolTemplate, ProtocolRunPlan, CandidateRunPlan, EnvelopeRunPlan, RunPlan, ProtocolPlanStep, RunPlanStep, ScientificInputSpec, ScientificStage, SnapshotVerification
from .study_issue import IssuedExtensionSet


class ProtocolCompilationError(ValueError):
    pass


_NONACTUATING_ACTIONS = {
    AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
    AuthorityAction.READ_ONLY_EXPLORATION,
    AuthorityAction.REFERENCE_WORLD_EXECUTION,
    AuthorityAction.SIMULATION_EXECUTION,
}


def _config_input(
    config: CapabilityConfigRef,
    *,
    visibility_ceiling: VisibilityCeiling | None = None,
    outcome_access: OutcomeAccess | None = None,
) -> ExternalInputSpec:
    return ExternalInputSpec(
        input_id="config",
        logical_artifact_id=config.artifact_id,
        expected_content_sha256=config.content_sha256,
        expected_payload_schema=config.config_schema,
        expected_media_type=None,
        expected_size_bytes=None,
        expected_visibility_ceiling=visibility_ceiling,
        expected_outcome_access=outcome_access,
        identity_scope_sha256=None,
    )


def _sealed_dependency_outcome_ids(
    *,
    template: ProtocolStepTemplate,
    compiled_steps: dict[str, ProtocolPlanStep],
    sealed_outcome_artifact_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve sealed outcomes already produced by direct receipt dependencies.

    Reveal barriers may bind either pre-existing external outcomes or outputs
    generated earlier in the same frozen DAG.  The latter remain dependency
    inputs; treating them as a second external source would break their receipt
    lineage and require an ad-hoc republishing act.
    """

    dependency_outputs = {
        output.logical_artifact_id: output
        for dependency_id in template.dependency_step_ids
        for output in compiled_steps[dependency_id].outputs
    }
    resolved = tuple(sorted(set(sealed_outcome_artifact_ids).intersection(dependency_outputs)))
    for artifact_id in resolved:
        output = dependency_outputs[artifact_id]
        if (
            output.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or not output.visibility_ceiling.is_at_least_as_restrictive_as(
                VisibilityCeiling.PROSPECTIVE
            )
        ):
            raise ProtocolCompilationError(
                "receipt-dependent reveal input is not prospectively sealed"
            )
    return resolved


def _aggregate_budget(steps: tuple[ProtocolStepTemplate, ...]) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=max(step.resource_budget.cpu_cores for step in steps),
        memory_bytes=max(step.resource_budget.memory_bytes for step in steps),
        gpu_devices=max(step.resource_budget.gpu_devices for step in steps),
        wall_time_seconds=sum(step.resource_budget.wall_time_seconds for step in steps),
        source_scan_bytes=sum(step.resource_budget.source_scan_bytes for step in steps),
        output_bytes=sum(step.resource_budget.output_bytes for step in steps),
    )


def _validate_campaign_bindings(
    campaign: CampaignSpec,
    system: SystemSpec,
    experiment: ExperimentSpec,
) -> None:
    if system.system_id not in campaign.system_ids:
        raise ProtocolCompilationError("campaign does not contain the prepared system")
    if system.world.world_id not in campaign.world_ids:
        raise ProtocolCompilationError("campaign does not contain the experiment world")
    experiment_claim_ids = {claim.claim_id for claim in experiment.claims}
    if not experiment_claim_ids.issubset(campaign.target_claim_ids):
        raise ProtocolCompilationError("campaign does not contain every experiment claim")
    expected_authority = ObjectIdentity.from_record(
        system.authority_policy.policy_id,
        system.authority_policy,
    )
    if campaign.authority_policy != expected_authority:
        raise ProtocolCompilationError("campaign and system authority policies differ")


def _validate_authorization(
    system: SystemSpec,
    experiment: ExperimentSpec,
    frozen_proposal: FrozenApprovalRoot,
    authorization: DurableAuthorizationRecord,
    template: ProtocolTemplate,
    implementation_commit: str,
    approval_service: CompleteApprovalService,
) -> None:
    if not isinstance(authorization, DurableAuthorizationRecord):
        raise ProtocolCompilationError("historical authorization is decode/reproduction-only")
    try:
        stored_experiment = approval_service.replay(
            system=system,
            frozen_proposal=ObjectIdentity.from_record(
                frozen_proposal.frozen_proposal_id,
                frozen_proposal,
            ),
            authorization=ObjectIdentity.from_record(
                authorization.authorization_id,
                authorization,
            ),
        )
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        raise ProtocolCompilationError(f"durable authorization replay failed: {error}") from error
    if stored_experiment != experiment:
        raise ProtocolCompilationError(
            "experiment differs from the completely replayed authorization"
        )
    if authorization.implementation_commit != implementation_commit:
        raise ProtocolCompilationError("authorization implementation commit drifted")
    if template.nonactuating and authorization.action not in _NONACTUATING_ACTIONS:
        raise ProtocolCompilationError("nonactuating template received broader authority")


def _validate_model_scope(
    system: SystemSpec,
    template: ProtocolTemplate,
    model_set: ViewModelSetSpec | ModelSetSpec | None,
) -> None:
    if template.requires_model_set and model_set is None:
        raise ProtocolCompilationError("robust protocol requires a frozen ModelSetSpec")
    if model_set is not None:
        if model_set.target_world_id != system.world.world_id:
            raise ProtocolCompilationError("model set targets another evidence world")
        known_views = {view.view_id for view in system.numerical_views}
        bound_views = (
            set(model_set.member_view_ids)
            if isinstance(model_set, ViewModelSetSpec)
            else {
                view_id for member in model_set.members for view_id in member.qualification_view_ids
            }
        )
        if not bound_views.issubset(known_views):
            raise ProtocolCompilationError("model set contains an unbound numerical view")
    if template.requests_controller and system.readiness is ReadinessStatus.COMPUTABILITY_BOUNDARY:
        raise ProtocolCompilationError("controller request exceeds the computability envelope")


def _validate_controller_promotion_containment(
    template: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> None:
    """Admit only explicitly qualified corrected controller implementations."""

    if not template.requests_controller:
        return
    controller_steps = tuple(
        value for value in template.steps if value.stage is ScientificStage.CONTROLLER
    )
    if not controller_steps:
        raise ProtocolCompilationError(
            "RWRR_CONTROLLER_PROMOTION_CONTAINED: controller template lacks a "
            "qualified current controller stage"
        )
    for step in controller_steps:
        manifest = registry.resolve(step.capability_key, step.capability_version)
        if (
            manifest.kind is not CapabilityKind.CONTROLLER_SYNTHESIZER
            or "corrected-current-controller-contract" not in manifest.conformance_check_ids
        ):
            raise ProtocolCompilationError(
                "RWRR_CONTROLLER_PROMOTION_CONTAINED: controller capability lacks "
                "the corrected current-path conformance contract"
            )


def _step_input_schemas(
    template: ProtocolStepTemplate,
    steps: dict[str, ProtocolPlanStep],
    incoming_edges: tuple[CandidateGraphEdge, ...] | None,
) -> tuple[str, ...]:
    if incoming_edges is not None:
        return tuple(sorted({edge.payload_schema for edge in incoming_edges}))
    return tuple(
        sorted(
            {
                output.payload_schema
                for dependency_id in template.dependency_step_ids
                for output in steps[dependency_id].outputs
            }
        )
    )


def _compile_step(
    *,
    run_plan_id: str,
    template: ProtocolStepTemplate,
    compiled_steps: dict[str, ProtocolPlanStep],
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
    dataset_bindings: tuple[ExperimentDatasetBinding, ...],
    scientific_graph: CandidateScientificGraph | None,
    incoming_edges: tuple[CandidateGraphEdge, ...],
    external_by_id: dict[str, CandidateGraphExternalInput],
) -> ProtocolPlanStep:
    input_schemas = _step_input_schemas(
        template,
        compiled_steps,
        None if scientific_graph is None else incoming_edges,
    )
    output_schemas = tuple(sorted({output.payload_schema for output in template.outputs}))
    manifest = registry.resolve(template.capability_key, template.capability_version)
    _validate_stage_capability(
        template, manifest.kind, historical=isinstance(experiment, RetrospectiveExperimentSpec)
    )
    requirement = CapabilityRequirement(
        capability_key=template.capability_key,
        capability_version=template.capability_version,
        kind=manifest.kind,
        config=template.config,
        required_input_schema_ids=input_schemas,
        required_output_schema_ids=output_schemas,
        required_permissions=template.required_permissions,
        requested_evidence_ceiling=EvidenceCeiling.lowest(
            *(claim.evidence_ceiling for claim in experiment.claims),
            manifest.maximum_evidence_ceiling,
        ),
        requested_outcome_access=template.requested_outcome_access,
        requested_resources=template.resource_budget,
    )
    registry.require(requirement)
    parent_visibility_ceilings = tuple(
        sorted(
            (
                (edge.visibility_ceiling for edge in incoming_edges)
                if scientific_graph is not None
                else (
                    output.visibility_ceiling
                    for dependency_id in template.dependency_step_ids
                    for output in compiled_steps[dependency_id].outputs
                )
            ),
            key=lambda ceiling: ceiling.value,
        )
    )
    dataset_inputs = dataset_external_inputs_for_stage(dataset_bindings, template.stage)
    parent_visibility_ceilings = tuple(
        sorted(
            (
                *parent_visibility_ceilings,
                *(
                    value.expected_visibility_ceiling
                    for value in dataset_inputs
                    if value.expected_visibility_ceiling is not None
                ),
            ),
            key=lambda ceiling: ceiling.value,
        )
    )
    if scientific_graph is None and template.stage in {
        ScientificStage.EVALUATE,
        ScientificStage.REVEAL,
    }:
        parent_visibility_ceilings = tuple(
            sorted(
                (*parent_visibility_ceilings, experiment.evaluation_visibility_ceiling),
                key=lambda ceiling: ceiling.value,
            )
        )
    # An explicitly prospective evaluator may commit evaluator-only operands
    # for a later qualification. Entering its protected read is not a public
    # release of those operands. REVEAL and exposed evaluation still publish
    # revealed outputs; inherited exposed lineage below can never be resealed.
    evaluator_only = (
        scientific_graph is not None
        and template.stage is ScientificStage.EVALUATE
        and template.requested_outcome_access is OutcomeAccess.EVALUATOR_REVEAL
        and template.visibility_ceiling is VisibilityCeiling.PROSPECTIVE
    )
    declared_output_access = (
        OutcomeAccess.EVALUATION_REVEALED
        if template.stage in {ScientificStage.EVALUATE, ScientificStage.REVEAL}
        and not evaluator_only
        else template.requested_outcome_access
    )
    parent_outcome_accesses = tuple(
        (edge.outcome_access for edge in incoming_edges)
        if scientific_graph is not None
        else (
            output.outcome_access
            for dependency_id in template.dependency_step_ids
            for output in compiled_steps[dependency_id].outputs
        )
    )
    output_access = derived_outcome_access(
        declared_output_access,
        *parent_outcome_accesses,
        *(
            value.expected_outcome_access
            for value in dataset_inputs
            if value.expected_outcome_access is not None
        ),
    )
    inherited_output_visibility = inherited_visibility(
        parent_visibility_ceilings,
        output_access,
    )
    output_visibility = VisibilityCeiling.most_restrictive(
        template.visibility_ceiling,
        inherited_output_visibility,
    )
    outputs = tuple(
        ArtifactOutputSpec(
            output_id=f"{template.step_id}.{output.output_id}",
            logical_artifact_id=(f"artifact.{run_plan_id}.{template.step_id}.{output.output_id}"),
            relative_path=(
                f"runs/{run_plan_id}/outputs/{template.step_id}/"
                f"{output.output_id}{output.filename_suffix}"
            ),
            payload_schema=output.payload_schema,
            profile=output.profile,
            media_type=output.media_type,
            # A terminal adjudication is the completed reveal, even when its
            # sibling scientific operands stay private to the next evaluator.
            # Its existing record contract forbids an in-reveal context.
            visibility_ceiling=(
                VisibilityCeiling.most_restrictive(
                    output_visibility, VisibilityCeiling.OUTCOME_VISIBLE
                )
                if evaluator_only and output.payload_schema == ScientificAdjudicationRecord.SCHEMA
                else output_visibility
            ),
            parent_visibility_ceilings=parent_visibility_ceilings,
            outcome_access=(
                derived_outcome_access(OutcomeAccess.EVALUATION_REVEALED, output_access)
                if evaluator_only and output.payload_schema == ScientificAdjudicationRecord.SCHEMA
                else output_access
            ),
        )
        for output in template.outputs
    )
    scientific_external_inputs = (
        ()
        if scientific_graph is None
        else _scientific_external_inputs_for_step(
            scientific_graph,
            incoming_edges=incoming_edges,
            external_by_id=external_by_id,
            exact_sizes=isinstance(experiment, RetrospectiveExperimentSpec),
        )
    )
    external_inputs = [
        _config_input(template.config),
        *dataset_inputs,
        *scientific_external_inputs,
    ]
    if scientific_graph is None and template.stage in {
        ScientificStage.EVALUATE,
        ScientificStage.REVEAL,
    }:
        sealed_dependency_ids = set(
            _sealed_dependency_outcome_ids(
                template=template,
                compiled_steps=compiled_steps,
                sealed_outcome_artifact_ids=(experiment.reveal_barrier.sealed_outcome_artifact_ids),
            )
        )
        external_inputs.extend(
            ExternalInputSpec(
                input_id=f"sealed.{artifact_id}",
                logical_artifact_id=artifact_id,
                expected_content_sha256=None,
                expected_payload_schema=None,
                expected_media_type=None,
                expected_size_bytes=None,
                expected_visibility_ceiling=experiment.evaluation_visibility_ceiling,
                expected_outcome_access=(experiment.reveal_barrier.evaluation_outcome_access),
                identity_scope_sha256=(experiment.reveal_barrier.evaluation_manifest_sha256),
            )
            for artifact_id in experiment.reveal_barrier.sealed_outcome_artifact_ids
            if artifact_id not in sealed_dependency_ids
        )
    ordered_external_inputs = tuple(sorted(external_inputs, key=lambda value: value.input_id))
    if scientific_graph is not None:
        return RunPlanStep(
            step_id=template.step_id,
            stage=template.stage,
            capability=requirement,
            dependency_step_ids=template.dependency_step_ids,
            external_inputs=ordered_external_inputs,
            outputs=outputs,
            resource_lock_ids=template.resource_lock_ids,
            barrier=template.barrier,
            maximum_attempts=template.maximum_attempts,
            obligation_ids=template.obligation_ids,
            scientific_inputs=_scientific_inputs_for_step(
                incoming_edges=incoming_edges,
                external_by_id=external_by_id,
                compiled_steps=compiled_steps,
            ),
        )
    return ProtocolPlanStep(
        step_id=template.step_id,
        stage=template.stage,
        capability=requirement,
        dependency_step_ids=template.dependency_step_ids,
        external_inputs=ordered_external_inputs,
        outputs=outputs,
        resource_lock_ids=template.resource_lock_ids,
        barrier=template.barrier,
        maximum_attempts=template.maximum_attempts,
        obligation_ids=template.obligation_ids,
    )


def _scientific_inputs_for_step(
    *,
    incoming_edges: tuple[CandidateGraphEdge, ...],
    external_by_id: dict[str, CandidateGraphExternalInput],
    compiled_steps: dict[str, ProtocolPlanStep],
) -> tuple[ScientificInputSpec, ...]:
    values: list[ScientificInputSpec] = []
    for edge in incoming_edges:
        operational_logical_artifact_id: str
        producer_output_id: str | None = None
        if edge.producer_node_id is not None:
            producer = compiled_steps.get(edge.producer_node_id)
            if producer is None:
                raise ProtocolCompilationError(
                    "scientific edge producer is not available in topological order"
                )
            expected_output_id = f"{edge.producer_node_id}.{edge.producer_output_id}"
            output = next(
                (value for value in producer.outputs if value.output_id == expected_output_id),
                None,
            )
            if output is None:
                raise ProtocolCompilationError(
                    "scientific edge names an unavailable producer output"
                )
            operational_logical_artifact_id = output.logical_artifact_id
            producer_output_id = output.output_id
        else:
            external = external_by_id.get(edge.external_input_id or "")
            if external is None:
                raise ProtocolCompilationError(
                    "scientific edge names an unavailable external input"
                )
            operational_logical_artifact_id = external.logical_artifact_id
        values.append(
            ScientificInputSpec(
                edge_id=edge.edge_id,
                consumer_input_id=edge.consumer_input_id,
                scientific_role=edge.scientific_role,
                producer_task_id=edge.producer_node_id,
                producer_output_id=producer_output_id,
                external_input_id=edge.external_input_id,
                logical_artifact_id=edge.logical_artifact_id,
                operational_logical_artifact_id=(operational_logical_artifact_id),
                payload_schema=edge.payload_schema,
                media_type=edge.media_type,
                maximum_size_bytes=edge.maximum_size_bytes,
                outcome_access=edge.outcome_access,
                visibility_ceiling=edge.visibility_ceiling,
                barrier=edge.barrier,
            )
        )
    return tuple(sorted(values, key=lambda value: value.edge_id))


def _scientific_external_inputs_for_step(
    graph: CandidateScientificGraph,
    *,
    incoming_edges: tuple[CandidateGraphEdge, ...],
    external_by_id: dict[str, CandidateGraphExternalInput],
    exact_sizes: bool = False,
) -> tuple[ExternalInputSpec, ...]:
    values: list[ExternalInputSpec] = []
    for edge in incoming_edges:
        if edge.external_input_id is None:
            continue
        external = external_by_id[edge.external_input_id]
        values.append(
            ExternalInputSpec(
                input_id=edge.consumer_input_id,
                logical_artifact_id=external.logical_artifact_id,
                expected_content_sha256=external.expected_content_sha256,
                expected_payload_schema=external.payload_schema,
                expected_media_type=external.media_type,
                # Historical carriers declare an exact ArtifactIdentity for every
                # source. Closure checks this byte count against that carrier.
                expected_size_bytes=external.maximum_size_bytes if exact_sizes else None,
                expected_visibility_ceiling=external.visibility_ceiling,
                expected_outcome_access=external.outcome_access,
                identity_scope_sha256=(
                    None if external.expected_content_sha256 is not None else graph.fingerprint()
                ),
            )
        )
    return tuple(sorted(values, key=lambda value: value.input_id))


def _validate_stage_capability(
    template: ProtocolStepTemplate,
    kind: CapabilityKind,
    *,
    historical: bool = False,
) -> None:
    permissions = set(template.required_permissions)
    sealed_read_permission = CapabilityPermission.READ_SEALED_OUTCOMES
    reveal_permission = CapabilityPermission.REVEAL_OUTCOMES
    evaluator_permissions = {sealed_read_permission, reveal_permission}
    if kind is CapabilityKind.APPROVAL_GATE:
        raise ProtocolCompilationError("approval gates cannot execute as scientific workers")
    if template.stage in {ScientificStage.EVALUATE, ScientificStage.REVEAL}:
        if kind is CapabilityKind.EVALUATOR:
            prospective_reveal = evaluator_permissions.issubset(permissions) and (
                template.requested_outcome_access is OutcomeAccess.EVALUATOR_REVEAL
            )
            post_reveal = (
                template.stage is ScientificStage.EVALUATE
                and not permissions.intersection(evaluator_permissions)
                and template.requested_outcome_access is OutcomeAccess.EVALUATION_REVEALED
            )
            historical_reveal = (
                historical
                and template.visibility_ceiling is VisibilityCeiling.OUTCOME_VISIBLE
                and template.requested_outcome_access is OutcomeAccess.EVALUATOR_REVEAL
                and {CapabilityPermission.READ_OUTCOME_VISIBLE, reveal_permission}.issubset(
                    permissions
                )
                and sealed_read_permission not in permissions
            )
            if not prospective_reveal and not post_reveal and not historical_reveal:
                raise ProtocolCompilationError(
                    "evaluator must either own sealed reveal or consume revealed summaries"
                )
        elif (
            template.stage is ScientificStage.EVALUATE
            and kind is CapabilityKind.HYPOTHESIS_ADJUDICATOR
        ):
            if permissions.intersection(evaluator_permissions):
                raise ProtocolCompilationError(
                    "post-reveal adjudicator requests sealed reveal authority"
                )
            if template.requested_outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
                raise ProtocolCompilationError(
                    "post-reveal adjudicator must use revealed summary access"
                )
        else:
            raise ProtocolCompilationError("evaluation stages require evaluator capability")
    elif reveal_permission in permissions:
        raise ProtocolCompilationError("non-evaluator stage requests reveal authority")
    elif (
        sealed_read_permission in permissions
        and template.requested_outcome_access is not OutcomeAccess.EVALUATION_SEALED
    ):
        raise ProtocolCompilationError("sealed-compute stage must retain prospective sealed access")


def _validate_resource_envelope_inputs(
    *,
    steps: tuple[ProtocolPlanStep, ...],
    issued_extension_set: IssuedExtensionSet,
    resource_envelope: ExecutionResourceEnvelopeSpec,
    jit_census: PredevelopmentJitSignatureCensus | None,
    jit_manifest: JitGraphSignatureManifest | None,
) -> None:
    extension_identity = ObjectIdentity.from_record(
        issued_extension_set.issued_extension_set_id,
        issued_extension_set,
    )
    if resource_envelope.issued_study_extensions != extension_identity:
        raise ProtocolCompilationError("resource envelope binds another issued extension set")
    try:
        validate_execution_jit_evidence(resource_envelope, jit_census, jit_manifest)
    except ValueError as error:
        raise ProtocolCompilationError(str(error)) from error
    cells = {value.task_id: value for value in resource_envelope.task_cells}
    if set(cells) != {value.step_id for value in steps}:
        raise ProtocolCompilationError(
            "resource envelope task topology differs from the compiled graph"
        )
    for step in steps:
        cell = cells[step.step_id]
        if cell.maximum_attempts != step.maximum_attempts:
            raise ProtocolCompilationError(
                "resource envelope attempt ceiling differs from the issued task"
            )
        budget = step.capability.requested_resources
        if (
            cell.resource_budget.cpu_cores != budget.cpu_cores
            or cell.resource_budget.memory_bytes != budget.memory_bytes
            or cell.resource_budget.output_bytes != budget.output_bytes
            or cell.resource_budget.source_byte_limit != budget.source_scan_bytes
        ):
            raise ProtocolCompilationError(
                "non-time resource cell differs from the issued task budget"
            )


def compile_run_plan(
    *,
    run_plan_id: str,
    campaign: CampaignSpec,
    system: SystemSpec,
    experiment: ExperimentSpec,
    frozen_proposal: FrozenApprovalRoot | None,
    authorization: DurableAuthorizationRecord | None,
    template: ProtocolTemplate,
    registry: CapabilityRegistry,
    implementation_commit: str,
    approval_service: CompleteApprovalService | None,
    model_set: ViewModelSetSpec | ModelSetSpec | None = None,
    dataset_bindings: tuple[ExperimentDatasetBinding, ...] = (),
    scientific_graph: CandidateScientificGraph | None = None,
    candidate: ObjectIdentity | None = None,
    issued_extension_set: IssuedExtensionSet | None = None,
    execution_envelope_spec: ExecutionEnvelopeSpec | None = None,
    execution_resource_envelope_spec: ExecutionResourceEnvelopeSpec | None = None,
    predevelopment_jit_signature_census: PredevelopmentJitSignatureCensus | None = None,
    jit_graph_signature_manifest: JitGraphSignatureManifest | None = None,
    preissue_candidate: DraftStudyCandidate | None = None,
) -> ProtocolRunPlan:
    """Compile an authorized plan or an explicitly nonauthorizing preissue projection."""

    if preissue_candidate is None:
        if experiment.readiness is not ReadinessStatus.READY:
            raise ProtocolCompilationError("only a ready authorized experiment can compile")
        if frozen_proposal is None or authorization is None or approval_service is None:
            raise ProtocolCompilationError("authorized compilation requires exact authority")
    elif (
        any(value is not None for value in (frozen_proposal, authorization, approval_service))
        or experiment.readiness is not ReadinessStatus.AUTHORITY_REQUIRED
        or (system, experiment, campaign, template, scientific_graph)
        != (
            preissue_candidate.system,
            preissue_candidate.experiment,
            preissue_candidate.campaign,
            preissue_candidate.protocol,
            preissue_candidate.scientific_graph,
        )
    ):
        raise ProtocolCompilationError(
            "preissue projection differs from its authority-pending candidate"
        )
    validate_experiment_against_system(experiment, system)
    _validate_campaign_bindings(campaign, system, experiment)
    if preissue_candidate is None:
        assert frozen_proposal is not None and authorization is not None
        assert approval_service is not None
        _validate_authorization(
            system,
            experiment,
            frozen_proposal,
            authorization,
            template,
            implementation_commit,
            approval_service,
        )
    _validate_model_scope(system, template, model_set)
    _validate_controller_promotion_containment(template, registry)
    if any(not isinstance(value, ExperimentDatasetBinding) for value in dataset_bindings):
        raise ProtocolCompilationError("dataset_bindings contains another record type")
    if (scientific_graph is None) != (candidate is None):
        raise ProtocolCompilationError(
            "exact candidate identity and scientific graph must be supplied together"
        )
    if execution_envelope_spec is not None and execution_resource_envelope_spec is not None:
        raise ProtocolCompilationError(
            "deadline-bearing and deadline-free envelopes are mutually exclusive"
        )
    envelope_supplied = (
        execution_envelope_spec is not None or execution_resource_envelope_spec is not None
    )
    if (issued_extension_set is None) != (not envelope_supplied):
        raise ProtocolCompilationError(
            "issued extensions and execution-envelope spec must be supplied together"
        )
    jit_values = (
        predevelopment_jit_signature_census,
        jit_graph_signature_manifest,
    )
    if execution_resource_envelope_spec is None and any(value is not None for value in jit_values):
        raise ProtocolCompilationError("JIT resource identities require a resource envelope")
    if issued_extension_set is not None and scientific_graph is None:
        raise ProtocolCompilationError("extension/envelope plans require the exact candidate graph")
    binding_ids = tuple(value.binding_id for value in dataset_bindings)
    if tuple(sorted(set(binding_ids))) != binding_ids:
        raise ProtocolCompilationError("dataset_bindings must be sorted and unique")
    experiment_identity = ObjectIdentity.from_record(experiment.experiment_id, experiment)
    if any(
        value.experiment != experiment_identity or value.binding_state.value != "VERIFIED"
        for value in dataset_bindings
    ):
        raise ProtocolCompilationError(
            "dataset bindings must be verified against the exact experiment"
        )
    aggregate = _aggregate_budget(template.steps)
    if not campaign.budget.contains(aggregate):
        raise ProtocolCompilationError("protocol exceeds the campaign resource budget")
    if not system.authority_policy.budget_ceiling.contains(aggregate):
        raise ProtocolCompilationError("protocol exceeds the authority resource budget")
    budget_authority = (
        preissue_candidate.resource_ceiling
        if preissue_candidate is not None
        else cast(DurableAuthorizationRecord, authorization).requested_budget
    )
    if not budget_authority.contains(aggregate):
        raise ProtocolCompilationError("protocol exceeds its authorized resource budget")

    compiled: dict[str, ProtocolPlanStep] = {}
    edges_by_consumer: dict[str, list[CandidateGraphEdge]] = {}
    external_by_id: dict[str, CandidateGraphExternalInput] = {}
    if scientific_graph is not None:
        for edge in scientific_graph.edges:
            edges_by_consumer.setdefault(edge.consumer_node_id, []).append(edge)
        external_by_id = {value.input_id: value for value in scientific_graph.external_inputs}
    templates_by_id = {step.step_id: step for step in template.steps}
    for step_id in _template_order(template):
        source = templates_by_id[step_id]
        compiled[step_id] = _compile_step(
            run_plan_id=run_plan_id,
            template=source,
            compiled_steps=compiled,
            experiment=experiment,
            registry=registry,
            dataset_bindings=dataset_bindings,
            scientific_graph=scientific_graph,
            incoming_edges=tuple(edges_by_consumer.get(step_id, ())),
            external_by_id=external_by_id,
        )
    numerical_view_ids = tuple(
        sorted({view_id for claim in experiment.claims for view_id in claim.numerical_view_ids})
    )
    steps = tuple(sorted(compiled.values(), key=lambda step: step.step_id))
    if issued_extension_set is not None and execution_envelope_spec is not None:
        expected_extension_identity = ObjectIdentity.from_record(
            issued_extension_set.issued_extension_set_id,
            issued_extension_set,
        )
        if execution_envelope_spec.issued_study_extensions != (expected_extension_identity):
            raise ProtocolCompilationError("execution envelope binds another issued extension set")
        envelope_task_ids = {value.task_id for value in execution_envelope_spec.cells}
        plan_task_ids = {value.step_id for value in steps}
        if envelope_task_ids != plan_task_ids:
            raise ProtocolCompilationError(
                "execution envelope task topology differs from the compiled graph"
            )
        attempts_by_task = {value.step_id: value.maximum_attempts for value in template.steps}
        if any(
            cell.maximum_physical_tokens > attempts_by_task[cell.task_id]
            for cell in execution_envelope_spec.cells
        ):
            raise ProtocolCompilationError(
                "execution envelope cell exceeds the issued task attempt limit"
            )
    if issued_extension_set is not None and execution_resource_envelope_spec is not None:
        _validate_resource_envelope_inputs(
            steps=steps,
            issued_extension_set=issued_extension_set,
            resource_envelope=execution_resource_envelope_spec,
            jit_census=predevelopment_jit_signature_census,
            jit_manifest=jit_graph_signature_manifest,
        )
    campaign_identity = ObjectIdentity.from_record(campaign.campaign_id, campaign)
    system_identity = ObjectIdentity.from_record(system.system_id, system)
    claim_identities = tuple(
        sorted(
            (ObjectIdentity.from_record(claim.claim_id, claim) for claim in experiment.claims),
            key=lambda identity: identity.object_id,
        )
    )
    experiment_identity = ObjectIdentity.from_record(
        experiment.experiment_id,
        experiment,
    )
    authorization_identity = (
        ObjectIdentity(
            f"preissue-boundary.{preissue_candidate.candidate_id}",
            'empirical-lawhood/runtime/preissue-authority-boundary',
            "1.0.0",
            hashlib.sha256(
                canonical_json_bytes(preissue_candidate.expected_authority_gates)
            ).hexdigest(),
        )
        if preissue_candidate is not None
        else ObjectIdentity.from_record(
            cast(DurableAuthorizationRecord, authorization).authorization_id,
            cast(DurableAuthorizationRecord, authorization),
        )
    )
    model_set_identity = (
        None if model_set is None else ObjectIdentity.from_record(model_set.model_set_id, model_set)
    )
    if scientific_graph is not None:
        if not all(isinstance(value, RunPlanStep) for value in steps):
            raise ProtocolCompilationError("exact-edge compilation produced an earlier plan-step format")
        assert candidate is not None
        exact_steps = cast(tuple[RunPlanStep, ...], steps)
        if (
            preissue_candidate is None
            and issued_extension_set is not None
            and execution_envelope_spec is not None
        ):
            return EnvelopeRunPlan(
                run_plan_id=run_plan_id,
                campaign=campaign_identity,
                system=system_identity,
                claims=claim_identities,
                experiment=experiment_identity,
                authorization=authorization_identity,
                model_set=model_set_identity,
                world_id=system.world.world_id,
                world_kind=system.world.kind,
                physical_preparation_ids=(
                    experiment.known_physical_independent_unit_ids
                    if isinstance(experiment, RetrospectiveExperimentSpec)
                    else (system.independent_unit.unit_id,)
                ),
                numerical_view_ids=numerical_view_ids,
                registry_sha256=registry.fingerprint(),
                implementation_commit=implementation_commit,
                steps=exact_steps,
                robust_model_set_required=template.requires_model_set,
                controller_requested=template.requests_controller,
                nonactuating=template.nonactuating,
                candidate=candidate,
                issued_extension_set=expected_extension_identity,
                execution_envelope_spec=ObjectIdentity.from_record(
                    execution_envelope_spec.envelope_spec_id,
                    execution_envelope_spec,
                ),
            )
        if (
            preissue_candidate is None
            and issued_extension_set is not None
            and execution_resource_envelope_spec is not None
        ):
            resource_plan_type: type[RunPlan] = (
                ResourceBoundRetrospectiveStudyRunPlan
                if isinstance(experiment, RetrospectiveExperimentSpec)
                else RunPlan
            )
            return resource_plan_type(
                run_plan_id=run_plan_id,
                campaign=campaign_identity,
                system=system_identity,
                claims=claim_identities,
                experiment=experiment_identity,
                authorization=authorization_identity,
                model_set=model_set_identity,
                world_id=system.world.world_id,
                world_kind=system.world.kind,
                physical_preparation_ids=(
                    experiment.known_physical_independent_unit_ids
                    if isinstance(experiment, RetrospectiveExperimentSpec)
                    else (system.independent_unit.unit_id,)
                ),
                numerical_view_ids=numerical_view_ids,
                registry_sha256=registry.fingerprint(),
                implementation_commit=implementation_commit,
                steps=exact_steps,
                robust_model_set_required=template.requires_model_set,
                controller_requested=template.requests_controller,
                nonactuating=template.nonactuating,
                candidate=candidate,
                issued_extension_set=ObjectIdentity.from_record(
                    issued_extension_set.issued_extension_set_id,
                    issued_extension_set,
                ),
                execution_resource_envelope_spec=ObjectIdentity.from_record(
                    execution_resource_envelope_spec.envelope_spec_id,
                    execution_resource_envelope_spec,
                ),
                predevelopment_jit_signature_census=None
                if predevelopment_jit_signature_census is None
                else ObjectIdentity.from_record(
                    predevelopment_jit_signature_census.census_id,
                    predevelopment_jit_signature_census,
                ),
                jit_graph_signature_manifest=None
                if jit_graph_signature_manifest is None
                else ObjectIdentity.from_record(
                    jit_graph_signature_manifest.manifest_id,
                    jit_graph_signature_manifest,
                ),
            )
        exact_plan_type: type[CandidateRunPlan] = (
            RetrospectiveStudyRunPlan if isinstance(experiment, RetrospectiveExperimentSpec) else CandidateRunPlan
        )
        return exact_plan_type(
            run_plan_id=run_plan_id,
            campaign=campaign_identity,
            system=system_identity,
            claims=claim_identities,
            experiment=experiment_identity,
            authorization=authorization_identity,
            model_set=model_set_identity,
            world_id=system.world.world_id,
            world_kind=system.world.kind,
            physical_preparation_ids=(
                experiment.known_physical_independent_unit_ids
                if isinstance(experiment, RetrospectiveExperimentSpec)
                else (system.independent_unit.unit_id,)
            ),
            numerical_view_ids=numerical_view_ids,
            registry_sha256=registry.fingerprint(),
            implementation_commit=implementation_commit,
            steps=exact_steps,
            robust_model_set_required=template.requires_model_set,
            controller_requested=template.requests_controller,
            nonactuating=template.nonactuating,
            candidate=candidate,
        )
    return ProtocolRunPlan(
        run_plan_id=run_plan_id,
        campaign=campaign_identity,
        system=system_identity,
        claims=claim_identities,
        experiment=experiment_identity,
        authorization=authorization_identity,
        model_set=model_set_identity,
        world_id=system.world.world_id,
        world_kind=system.world.kind,
        physical_preparation_ids=(
            experiment.known_physical_independent_unit_ids
            if isinstance(experiment, RetrospectiveExperimentSpec)
            else (system.independent_unit.unit_id,)
        ),
        numerical_view_ids=numerical_view_ids,
        registry_sha256=registry.fingerprint(),
        implementation_commit=implementation_commit,
        steps=steps,
        robust_model_set_required=template.requires_model_set,
        controller_requested=template.requests_controller,
        nonactuating=template.nonactuating,
    )


def compile_preissue_run_plan(
    *,
    run_plan_id: str,
    candidate_record: DraftStudyCandidate,
    candidate: ObjectIdentity,
    registry: CapabilityRegistry,
    implementation_commit: str,
    issued_extension_set: IssuedExtensionSet,
    resource_envelope: ExecutionResourceEnvelopeSpec,
    jit_census: PredevelopmentJitSignatureCensus | None,
    jit_manifest: JitGraphSignatureManifest | None,
    model_set: ViewModelSetSpec | ModelSetSpec | None = None,
) -> CandidateRunPlan:
    """Project exact future topology without approval, issue, or execution authority."""

    value = compile_run_plan(
        run_plan_id=run_plan_id,
        campaign=candidate_record.campaign,
        system=candidate_record.system,
        experiment=candidate_record.experiment,
        frozen_proposal=None,
        authorization=None,
        template=candidate_record.protocol,
        registry=registry,
        implementation_commit=implementation_commit,
        approval_service=None,
        scientific_graph=candidate_record.scientific_graph,
        candidate=candidate,
        issued_extension_set=issued_extension_set,
        execution_resource_envelope_spec=resource_envelope,
        predevelopment_jit_signature_census=jit_census,
        jit_graph_signature_manifest=jit_manifest,
        preissue_candidate=candidate_record,
        model_set=model_set,
    )
    if not isinstance(value, CandidateRunPlan) or isinstance(value, (EnvelopeRunPlan, RunPlan)):
        raise ProtocolCompilationError("preissue projection crossed its nonauthorizing boundary")
    return value


def _template_order(template: ProtocolTemplate) -> tuple[str, ...]:
    remaining = {step.step_id for step in template.steps}
    complete: set[str] = set()
    order: list[str] = []
    dependencies = {step.step_id: set(step.dependency_step_ids) for step in template.steps}
    while remaining:
        ready = sorted(step_id for step_id in remaining if dependencies[step_id].issubset(complete))
        if not ready:
            raise ProtocolCompilationError("protocol template contains a dependency cycle")
        for step_id in ready:
            remaining.remove(step_id)
            complete.add(step_id)
            order.append(step_id)
    return tuple(order)


def lower_run_plan(run_plan: ProtocolRunPlan, registry: CapabilityRegistry) -> ProtocolExecutionPlan:
    if run_plan.registry_sha256 != registry.fingerprint():
        raise ProtocolCompilationError("capability registry drifted after scientific freeze")
    tasks: list[ProtocolExecutionTask] = []
    for step in run_plan.steps:
        manifest = registry.require(step.capability)
        if isinstance(run_plan, CandidateRunPlan):
            if not isinstance(step, RunPlanStep):
                raise ProtocolCompilationError("candidate run plan contains a step outside its contract")
            tasks.append(
                ExecutionTask(
                    task_id=step.step_id,
                    stage=step.stage,
                    capability=step.capability,
                    capability_implementation_sha256=(manifest.implementation_sha256),
                    dependency_task_ids=step.dependency_step_ids,
                    external_inputs=step.external_inputs,
                    outputs=step.outputs,
                    resource_lock_ids=step.resource_lock_ids,
                    barrier=step.barrier,
                    maximum_attempts=step.maximum_attempts,
                    obligation_ids=step.obligation_ids,
                    scientific_inputs=step.scientific_inputs,
                )
            )
        else:
            tasks.append(
                ProtocolExecutionTask(
                    task_id=step.step_id,
                    stage=step.stage,
                    capability=step.capability,
                    capability_implementation_sha256=(manifest.implementation_sha256),
                    dependency_task_ids=step.dependency_step_ids,
                    external_inputs=step.external_inputs,
                    outputs=step.outputs,
                    resource_lock_ids=step.resource_lock_ids,
                    barrier=step.barrier,
                    maximum_attempts=step.maximum_attempts,
                    obligation_ids=step.obligation_ids,
                )
            )
    source_plan = ObjectIdentity.from_record(run_plan.run_plan_id, run_plan)
    if isinstance(run_plan, CandidateRunPlan):
        if not all(isinstance(value, ExecutionTask) for value in tasks):
            raise ProtocolCompilationError("candidate run plan lowering produced a task outside its execution contract")
        exact_tasks = cast(tuple[ExecutionTask, ...], tuple(tasks))
        if isinstance(run_plan, EnvelopeRunPlan):
            return EnvelopeExecutionPlan(
                execution_plan_id=f"execution.{run_plan.run_plan_id}",
                lane=PlanLane.PROSPECTIVE,
                source_plan=source_plan,
                registry_sha256=registry.fingerprint(),
                implementation_commit=run_plan.implementation_commit,
                tasks=exact_tasks,
                nonactuating=run_plan.nonactuating,
                candidate=run_plan.candidate,
                issued_extension_set=run_plan.issued_extension_set,
                execution_envelope_spec=run_plan.execution_envelope_spec,
            )
        if isinstance(run_plan, RunPlan):
            return ExecutionPlan(
                execution_plan_id=f"execution.{run_plan.run_plan_id}",
                lane=PlanLane.PROSPECTIVE,
                source_plan=source_plan,
                registry_sha256=registry.fingerprint(),
                implementation_commit=run_plan.implementation_commit,
                tasks=exact_tasks,
                nonactuating=run_plan.nonactuating,
                candidate=run_plan.candidate,
                issued_extension_set=run_plan.issued_extension_set,
                execution_resource_envelope_spec=(run_plan.execution_resource_envelope_spec),
                predevelopment_jit_signature_census=(run_plan.predevelopment_jit_signature_census),
                jit_graph_signature_manifest=run_plan.jit_graph_signature_manifest,
            )
        return CandidateExecutionPlan(
            execution_plan_id=f"execution.{run_plan.run_plan_id}",
            lane=PlanLane.PROSPECTIVE,
            source_plan=source_plan,
            registry_sha256=registry.fingerprint(),
            implementation_commit=run_plan.implementation_commit,
            tasks=exact_tasks,
            nonactuating=run_plan.nonactuating,
            candidate=run_plan.candidate,
        )
    return ProtocolExecutionPlan(
        execution_plan_id=f"execution.{run_plan.run_plan_id}",
        lane=PlanLane.PROSPECTIVE,
        source_plan=source_plan,
        registry_sha256=registry.fingerprint(),
        implementation_commit=run_plan.implementation_commit,
        tasks=tuple(tasks),
        nonactuating=run_plan.nonactuating,
    )


def execution_topology_sha256(
    execution_plan: ProtocolExecutionPlan,
    resource_envelope: ExecutionResourceEnvelopeSpec | None = None,
) -> str:
    """Fingerprint task edges, locators, attempts, and optional exact resource cells."""

    return hashlib.sha256(
        canonical_json_bytes(
            {
                "tasks": execution_plan.tasks,
                "resource_envelope": resource_envelope,
            }
        )
    ).hexdigest()


def exploration_search_family_sha256(
    proposals: tuple[AnalysisProposal, ...],
) -> str:
    """Fingerprint the complete registered family, including rejected proposals."""

    identities = tuple(
        ObjectIdentity.from_record(proposal.proposal_id, proposal)
        for proposal in sorted(proposals, key=lambda item: item.proposal_id)
    )
    return hashlib.sha256(canonical_json_bytes(identities)).hexdigest()


def schema_identity_sha256(schema: str) -> str:
    """Bind the exact canonical schema identity when no external schema file exists."""

    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def compile_exploration_plan(
    *,
    plan_id: str,
    snapshot: EvidenceSnapshot,
    snapshot_verification: SnapshotVerification,
    proposals: tuple[AnalysisProposal, ...],
    selections: tuple[ProposalSelection, ...],
    budget: ResourceBudget,
) -> ExplorationPlan:
    """Freeze a complete selected/rejected proposal family over one verified snapshot."""

    plan = ExplorationPlan(
        plan_id=plan_id,
        snapshot=ObjectIdentity.from_record(snapshot.snapshot_id, snapshot),
        snapshot_visibility_ceiling=snapshot.visibility_ceiling,
        proposals=proposals,
        selections=selections,
        information_cutoff=snapshot.information_cutoff,
        budget=budget,
        search_family_sha256=exploration_search_family_sha256(proposals),
        outcome_access=snapshot.outcome_access,
        visibility_ceiling=snapshot.visibility_ceiling,
    )
    _validate_exploration_bindings(plan, snapshot, snapshot_verification)
    return plan


def _validate_exploration_bindings(
    exploration_plan: ExplorationPlan,
    snapshot: EvidenceSnapshot,
    verification: SnapshotVerification,
) -> None:
    expected_snapshot = ObjectIdentity.from_record(snapshot.snapshot_id, snapshot)
    if exploration_plan.snapshot != expected_snapshot:
        raise ProtocolCompilationError("exploration plan binds another EvidenceSnapshot")
    if verification.snapshot != expected_snapshot:
        raise ProtocolCompilationError("snapshot verification binds another snapshot")
    artifact_ids = tuple(artifact.artifact_id for artifact in snapshot.artifacts)
    if verification.artifact_ids != artifact_ids:
        raise ProtocolCompilationError("snapshot verification does not cover every artifact")
    if exploration_plan.snapshot_visibility_ceiling != snapshot.visibility_ceiling:
        raise ProtocolCompilationError("exploration snapshot visibility drifted")
    if exploration_plan.visibility_ceiling.is_promotable:
        raise ProtocolCompilationError("exploration plan cannot receive promotable visibility")
    if exploration_plan.search_family_sha256 != exploration_search_family_sha256(
        exploration_plan.proposals
    ):
        raise ProtocolCompilationError("registered exploration search family is incomplete")
    _validate_exploration_analyses(exploration_plan, snapshot)


def _validate_exploration_analyses(
    exploration_plan: ExplorationPlan,
    snapshot: EvidenceSnapshot,
) -> None:
    for proposal in exploration_plan.proposals:
        analysis = proposal.analysis
        if analysis.snapshot_fingerprint != snapshot.fingerprint():
            raise ProtocolCompilationError("analysis snapshot fingerprint drifted")
        if not set(analysis.projection_ids).issubset(snapshot.projection_ids):
            raise ProtocolCompilationError("analysis requests an unverified snapshot projection")
        if analysis.parent_visibility_ceiling is not snapshot.visibility_ceiling:
            raise ProtocolCompilationError("analysis snapshot visibility differs")
        if analysis.outcome_access is not exploration_plan.outcome_access:
            raise ProtocolCompilationError("analysis outcome access differs from its plan")
        if not exploration_plan.visibility_ceiling.is_at_least_as_restrictive_as(
            analysis.visibility_ceiling
        ):
            raise ProtocolCompilationError("exploration output would lower analysis visibility")


def _require_exploration_manifest(
    manifest: CapabilityManifest,
    *,
    allowed_kinds: tuple[CapabilityKind, ...] = (CapabilityKind.ANALYSIS,),
) -> None:
    forbidden_permissions = {
        CapabilityPermission.APPROVE_NONACTUATING,
        CapabilityPermission.COMMAND_ACTUATOR,
        CapabilityPermission.READ_SEALED_OUTCOMES,
        CapabilityPermission.REVEAL_OUTCOMES,
        CapabilityPermission.WRITE_CATALOG,
    }
    if manifest.kind not in allowed_kinds:
        raise ProtocolCompilationError("exploration capability has the wrong registered kind")
    if manifest.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
        raise ProtocolCompilationError("exploration capability has scientific promotion authority")
    if forbidden_permissions.intersection(manifest.permissions):
        raise ProtocolCompilationError("exploration capability has forbidden authority")
    if manifest.requires_network:
        raise ProtocolCompilationError("exploration capability must use verified local artifacts")


def compile_exploration_execution_plan(
    *,
    exploration_plan: ExplorationPlan,
    snapshot: EvidenceSnapshot,
    snapshot_verification: SnapshotVerification,
    registry: CapabilityRegistry,
    implementation_commit: str,
    synthesis_capability_key: str,
    synthesis_capability_version: str,
    synthesis_budget: ResourceBudget,
) -> ProtocolExecutionPlan:
    """Lower a frozen read-only exploration wave with no promotion capability."""

    _validate_exploration_bindings(
        exploration_plan,
        snapshot,
        snapshot_verification,
    )
    selected_ids = {
        selection.proposal_id
        for selection in exploration_plan.selections
        if selection.disposition is ProposalDisposition.SELECTED
    }
    input_schemas = tuple(sorted({artifact.payload_schema for artifact in snapshot.artifacts}))
    tasks: list[ProtocolExecutionTask] = []
    for proposal in exploration_plan.proposals:
        if proposal.proposal_id not in selected_ids:
            continue
        analysis = proposal.analysis
        permissions = tuple(
            sorted(
                {
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.READ_OUTCOME_VISIBLE,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                }
            )
        )
        manifest = registry.resolve(
            analysis.registered_pipeline_key,
            analysis.registered_pipeline_version,
        )
        _require_exploration_manifest(manifest)
        requirement = CapabilityRequirement(
            capability_key=analysis.registered_pipeline_key,
            capability_version=analysis.registered_pipeline_version,
            kind=CapabilityKind.ANALYSIS,
            config=CapabilityConfigRef(
                config_id=f"config.{analysis.analysis_id}",
                config_schema=AnalysisSpec.SCHEMA,
                config_schema_sha256=schema_identity_sha256(AnalysisSpec.SCHEMA),
                content_sha256=analysis.fingerprint(),
                artifact_id=f"analysis-spec.{analysis.analysis_id}",
            ),
            required_input_schema_ids=input_schemas,
            required_output_schema_ids=(ExploratoryFinding.SCHEMA,),
            required_permissions=permissions,
            requested_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            requested_outcome_access=exploration_plan.outcome_access,
            requested_resources=analysis.budget,
        )
        registry.require(requirement)
        tasks.append(
            ProtocolExecutionTask(
                task_id=f"explore.{analysis.analysis_id}",
                stage=ScientificStage.EXPLORE,
                capability=requirement,
                capability_implementation_sha256=manifest.implementation_sha256,
                dependency_task_ids=(),
                external_inputs=tuple(
                    sorted(
                        (
                            _config_input(
                                requirement.config,
                                visibility_ceiling=exploration_plan.visibility_ceiling,
                                outcome_access=exploration_plan.outcome_access,
                            ),
                            *(
                                ExternalInputSpec(
                                    input_id=f"snapshot.{artifact.artifact_id}",
                                    logical_artifact_id=artifact.artifact_id,
                                    expected_content_sha256=artifact.sha256,
                                    expected_payload_schema=artifact.payload_schema,
                                    expected_media_type=artifact.media_type,
                                    expected_size_bytes=artifact.size_bytes,
                                    expected_visibility_ceiling=snapshot.visibility_ceiling,
                                    expected_outcome_access=snapshot.outcome_access,
                                    identity_scope_sha256=snapshot.fingerprint(),
                                )
                                for artifact in snapshot.artifacts
                            ),
                        ),
                        key=lambda value: value.input_id,
                    )
                ),
                outputs=(
                    ArtifactOutputSpec(
                        output_id=f"finding.{analysis.analysis_id}",
                        logical_artifact_id=f"finding.{exploration_plan.plan_id}.{analysis.analysis_id}",
                        relative_path=(
                            f"runs/{exploration_plan.plan_id}/exploration/"
                            f"{analysis.analysis_id}.json"
                        ),
                        payload_schema=ExploratoryFinding.SCHEMA,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/json",
                        visibility_ceiling=exploration_plan.visibility_ceiling,
                        parent_visibility_ceilings=(snapshot.visibility_ceiling,),
                        outcome_access=exploration_plan.outcome_access,
                    ),
                ),
                resource_lock_ids=("exploration-read-only",),
                barrier=BarrierKind.NONE,
                maximum_attempts=2,
                obligation_ids=("complete-search-family", "retain-null-attempts"),
            )
        )
    synthesis_manifest = registry.resolve(
        synthesis_capability_key,
        synthesis_capability_version,
    )
    _require_exploration_manifest(
        synthesis_manifest,
        allowed_kinds=(
            CapabilityKind.ANALYSIS,
            CapabilityKind.HYPOTHESIS_SYNTHESIZER,
        ),
    )
    synthesis_permissions = (
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.READ_OUTCOME_VISIBLE,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    )
    synthesis_requirement = CapabilityRequirement(
        capability_key=synthesis_capability_key,
        capability_version=synthesis_capability_version,
        kind=synthesis_manifest.kind,
        config=CapabilityConfigRef(
            config_id=f"config.{exploration_plan.plan_id}.synthesis",
            config_schema=ExplorationPlan.SCHEMA,
            config_schema_sha256=schema_identity_sha256(ExplorationPlan.SCHEMA),
            content_sha256=exploration_plan.fingerprint(),
            artifact_id=f"exploration-plan.{exploration_plan.plan_id}",
        ),
        required_input_schema_ids=(ExploratoryFinding.SCHEMA,),
        required_output_schema_ids=(HypothesisSet.SCHEMA,),
        required_permissions=synthesis_permissions,
        requested_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        requested_outcome_access=exploration_plan.outcome_access,
        requested_resources=synthesis_budget,
    )
    registry.require(synthesis_requirement)
    synthesis_parent_visibility = tuple(
        sorted(
            (
                snapshot.visibility_ceiling,
                *(output.visibility_ceiling for task in tasks for output in task.outputs),
            ),
            key=lambda ceiling: ceiling.value,
        )
    )
    tasks.append(
        ProtocolExecutionTask(
            task_id="exploration-synthesis",
            stage=ScientificStage.SYNTHESIZE,
            capability=synthesis_requirement,
            capability_implementation_sha256=(synthesis_manifest.implementation_sha256),
            dependency_task_ids=tuple(sorted(task.task_id for task in tasks)),
            external_inputs=(
                _config_input(
                    synthesis_requirement.config,
                    visibility_ceiling=exploration_plan.visibility_ceiling,
                    outcome_access=exploration_plan.outcome_access,
                ),
            ),
            outputs=(
                ArtifactOutputSpec(
                    output_id="hypothesis-set",
                    logical_artifact_id=f"hypothesis-set.{exploration_plan.plan_id}",
                    relative_path=(
                        f"runs/{exploration_plan.plan_id}/exploration/hypothesis-set.json"
                    ),
                    payload_schema=HypothesisSet.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/json",
                    visibility_ceiling=exploration_plan.visibility_ceiling,
                    parent_visibility_ceilings=synthesis_parent_visibility,
                    outcome_access=exploration_plan.outcome_access,
                ),
            ),
            resource_lock_ids=("exploration-read-only",),
            barrier=BarrierKind.NONE,
            maximum_attempts=1,
            obligation_ids=("retain-competing-hypotheses",),
        )
    )
    selected_budgets = tuple(
        proposal.analysis.budget
        for proposal in exploration_plan.proposals
        if proposal.proposal_id in selected_ids
    )
    aggregate_budget = _aggregate_resource_budgets((*selected_budgets, synthesis_budget))
    if not exploration_plan.budget.contains(aggregate_budget):
        raise ProtocolCompilationError("exploration execution exceeds its frozen budget")
    return ProtocolExecutionPlan(
        execution_plan_id=f"execution.{exploration_plan.plan_id}",
        lane=PlanLane.EXPLORATORY,
        source_plan=ObjectIdentity.from_record(exploration_plan.plan_id, exploration_plan),
        registry_sha256=registry.fingerprint(),
        implementation_commit=implementation_commit,
        tasks=tuple(sorted(tasks, key=lambda task: task.task_id)),
        nonactuating=True,
    )


def compile_exploration_wave_execution_plan(
    *,
    exploration_plan: ExplorationPlan,
    snapshot: EvidenceSnapshot,
    snapshot_verification: SnapshotVerification,
    wave_input: ExplorationWaveInput,
    registry: CapabilityRegistry,
    implementation_commit: str,
    skeptic_capability_key: str,
    skeptic_capability_version: str,
    skeptic_budget: ResourceBudget,
    synthesis_capability_key: str,
    synthesis_capability_version: str,
    synthesis_budget: ResourceBudget,
    wave_result_schema: str,
) -> ProtocolExecutionPlan:
    """Lower an input-only exploration wave into receipt-linked tasks.

    The inward runtime wave contract contains inputs only. Expected findings or
    hypotheses are not compiler inputs.
    """

    _validate_exploration_bindings(
        exploration_plan,
        snapshot,
        snapshot_verification,
    )
    if (
        wave_input.snapshot != snapshot
        or wave_input.snapshot_verification != snapshot_verification
        or wave_input.plan != exploration_plan
    ):
        raise ProtocolCompilationError("wave input differs from the exploration freeze")
    wave_input_id = wave_input.wave_input_id
    wave_input_spec = ExternalInputSpec(
        input_id="wave-input",
        logical_artifact_id=wave_input_id,
        expected_content_sha256=wave_input.fingerprint(),
        expected_payload_schema=wave_input.SCHEMA,
        expected_media_type="application/json",
        expected_size_bytes=len(wave_input.canonical_bytes()),
        expected_visibility_ceiling=exploration_plan.visibility_ceiling,
        expected_outcome_access=exploration_plan.outcome_access,
        identity_scope_sha256=exploration_plan.fingerprint(),
    )
    selected_ids = {
        selection.proposal_id
        for selection in exploration_plan.selections
        if selection.disposition is ProposalDisposition.SELECTED
    }
    snapshot_schemas = tuple(sorted({artifact.payload_schema for artifact in snapshot.artifacts}))
    analysis_tasks: list[ProtocolExecutionTask] = []
    analysis_permissions = tuple(
        sorted(
            (
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            )
        )
    )
    for proposal in exploration_plan.proposals:
        if proposal.proposal_id not in selected_ids:
            continue
        analysis = proposal.analysis
        manifest = registry.resolve(
            analysis.registered_pipeline_key,
            analysis.registered_pipeline_version,
        )
        _require_exploration_manifest(manifest)
        requirement = CapabilityRequirement(
            capability_key=analysis.registered_pipeline_key,
            capability_version=analysis.registered_pipeline_version,
            kind=CapabilityKind.ANALYSIS,
            config=CapabilityConfigRef(
                config_id=f"config.{analysis.analysis_id}",
                config_schema=AnalysisSpec.SCHEMA,
                config_schema_sha256=schema_identity_sha256(AnalysisSpec.SCHEMA),
                content_sha256=analysis.fingerprint(),
                artifact_id=f"analysis-spec.{analysis.analysis_id}",
            ),
            required_input_schema_ids=tuple(sorted({*snapshot_schemas, wave_input.SCHEMA})),
            required_output_schema_ids=(ExploratoryFinding.SCHEMA,),
            required_permissions=analysis_permissions,
            requested_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            requested_outcome_access=exploration_plan.outcome_access,
            requested_resources=analysis.budget,
        )
        registry.require(requirement)
        analysis_tasks.append(
            ProtocolExecutionTask(
                task_id=f"explore.{analysis.analysis_id}",
                stage=ScientificStage.EXPLORE,
                capability=requirement,
                capability_implementation_sha256=manifest.implementation_sha256,
                dependency_task_ids=(),
                external_inputs=tuple(
                    sorted(
                        (
                            _config_input(
                                requirement.config,
                                visibility_ceiling=exploration_plan.visibility_ceiling,
                                outcome_access=exploration_plan.outcome_access,
                            ),
                            wave_input_spec,
                            *(
                                ExternalInputSpec(
                                    input_id=f"snapshot.{artifact.artifact_id}",
                                    logical_artifact_id=artifact.artifact_id,
                                    expected_content_sha256=artifact.sha256,
                                    expected_payload_schema=artifact.payload_schema,
                                    expected_media_type=artifact.media_type,
                                    expected_size_bytes=artifact.size_bytes,
                                    expected_visibility_ceiling=snapshot.visibility_ceiling,
                                    expected_outcome_access=snapshot.outcome_access,
                                    identity_scope_sha256=snapshot.fingerprint(),
                                )
                                for artifact in snapshot.artifacts
                            ),
                        ),
                        key=lambda value: value.input_id,
                    )
                ),
                outputs=(
                    ArtifactOutputSpec(
                        output_id=f"finding.{analysis.analysis_id}",
                        logical_artifact_id=(
                            f"finding.{exploration_plan.plan_id}.{analysis.analysis_id}"
                        ),
                        relative_path=(
                            f"runs/{exploration_plan.plan_id}/exploration/"
                            f"{analysis.analysis_id}.json"
                        ),
                        payload_schema=ExploratoryFinding.SCHEMA,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/json",
                        visibility_ceiling=exploration_plan.visibility_ceiling,
                        parent_visibility_ceilings=(snapshot.visibility_ceiling,),
                        outcome_access=exploration_plan.outcome_access,
                    ),
                ),
                resource_lock_ids=("exploration-read-only",),
                barrier=BarrierKind.NONE,
                maximum_attempts=2,
                obligation_ids=(
                    "complete-search-family",
                    "retain-null-attempts",
                    "verified-wave-input",
                ),
            )
        )
    if not analysis_tasks:
        raise ProtocolCompilationError("exploration wave has no selected analysis")

    downstream_permissions = analysis_permissions
    analysis_task_ids = tuple(sorted(task.task_id for task in analysis_tasks))
    skeptic_manifest = registry.resolve(
        skeptic_capability_key,
        skeptic_capability_version,
    )
    _require_exploration_manifest(skeptic_manifest)
    skeptic_requirement = CapabilityRequirement(
        capability_key=skeptic_capability_key,
        capability_version=skeptic_capability_version,
        kind=CapabilityKind.ANALYSIS,
        config=CapabilityConfigRef(
            config_id=f"config.{exploration_plan.plan_id}.skeptic",
            config_schema=ExplorationPlan.SCHEMA,
            config_schema_sha256=schema_identity_sha256(ExplorationPlan.SCHEMA),
            content_sha256=exploration_plan.fingerprint(),
            artifact_id=f"exploration-plan.{exploration_plan.plan_id}",
        ),
        required_input_schema_ids=tuple(sorted((ExploratoryFinding.SCHEMA, wave_input.SCHEMA))),
        required_output_schema_ids=(SkepticReport.SCHEMA,),
        required_permissions=downstream_permissions,
        requested_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        requested_outcome_access=exploration_plan.outcome_access,
        requested_resources=skeptic_budget,
    )
    registry.require(skeptic_requirement)
    skeptic_task = ProtocolExecutionTask(
        task_id="exploration-skeptic",
        stage=ScientificStage.FALSIFY,
        capability=skeptic_requirement,
        capability_implementation_sha256=skeptic_manifest.implementation_sha256,
        dependency_task_ids=analysis_task_ids,
        external_inputs=tuple(
            sorted(
                (
                    _config_input(
                        skeptic_requirement.config,
                        visibility_ceiling=exploration_plan.visibility_ceiling,
                        outcome_access=exploration_plan.outcome_access,
                    ),
                    wave_input_spec,
                ),
                key=lambda value: value.input_id,
            )
        ),
        outputs=(
            ArtifactOutputSpec(
                output_id="skeptic-report",
                logical_artifact_id=f"skeptic-report.{exploration_plan.plan_id}",
                relative_path=(f"runs/{exploration_plan.plan_id}/exploration/skeptic-report.json"),
                payload_schema=SkepticReport.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                visibility_ceiling=exploration_plan.visibility_ceiling,
                parent_visibility_ceilings=tuple(
                    output.visibility_ceiling for task in analysis_tasks for output in task.outputs
                ),
                outcome_access=exploration_plan.outcome_access,
            ),
        ),
        resource_lock_ids=("exploration-read-only",),
        barrier=BarrierKind.NONE,
        maximum_attempts=1,
        obligation_ids=("complete-skeptic-family", "receipt-bound-findings"),
    )

    synthesis_manifest = registry.resolve(
        synthesis_capability_key,
        synthesis_capability_version,
    )
    _require_exploration_manifest(
        synthesis_manifest,
        allowed_kinds=(CapabilityKind.HYPOTHESIS_SYNTHESIZER,),
    )
    synthesis_requirement = CapabilityRequirement(
        capability_key=synthesis_capability_key,
        capability_version=synthesis_capability_version,
        kind=CapabilityKind.HYPOTHESIS_SYNTHESIZER,
        config=CapabilityConfigRef(
            config_id=f"config.{exploration_plan.plan_id}.synthesis",
            config_schema=ExplorationPlan.SCHEMA,
            config_schema_sha256=schema_identity_sha256(ExplorationPlan.SCHEMA),
            content_sha256=exploration_plan.fingerprint(),
            artifact_id=f"exploration-plan.{exploration_plan.plan_id}",
        ),
        required_input_schema_ids=tuple(
            sorted(
                (
                    ExploratoryFinding.SCHEMA,
                    SkepticReport.SCHEMA,
                    wave_input.SCHEMA,
                )
            )
        ),
        required_output_schema_ids=tuple(sorted((HypothesisSet.SCHEMA, wave_result_schema))),
        required_permissions=downstream_permissions,
        requested_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        requested_outcome_access=exploration_plan.outcome_access,
        requested_resources=synthesis_budget,
    )
    registry.require(synthesis_requirement)
    synthesis_task = ProtocolExecutionTask(
        task_id="exploration-synthesis",
        stage=ScientificStage.SYNTHESIZE,
        capability=synthesis_requirement,
        capability_implementation_sha256=synthesis_manifest.implementation_sha256,
        dependency_task_ids=tuple(sorted((*analysis_task_ids, skeptic_task.task_id))),
        external_inputs=tuple(
            sorted(
                (
                    _config_input(
                        synthesis_requirement.config,
                        visibility_ceiling=exploration_plan.visibility_ceiling,
                        outcome_access=exploration_plan.outcome_access,
                    ),
                    wave_input_spec,
                ),
                key=lambda value: value.input_id,
            )
        ),
        outputs=(
            ArtifactOutputSpec(
                output_id="hypothesis-set",
                logical_artifact_id=f"hypothesis-set.{exploration_plan.plan_id}",
                relative_path=(f"runs/{exploration_plan.plan_id}/exploration/hypothesis-set.json"),
                payload_schema=HypothesisSet.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                visibility_ceiling=exploration_plan.visibility_ceiling,
                parent_visibility_ceilings=(
                    exploration_plan.visibility_ceiling,
                    exploration_plan.visibility_ceiling,
                ),
                outcome_access=exploration_plan.outcome_access,
            ),
            ArtifactOutputSpec(
                output_id="wave-result",
                logical_artifact_id=f"wave-result.{exploration_plan.plan_id}",
                relative_path=(f"runs/{exploration_plan.plan_id}/exploration/wave-result.json"),
                payload_schema=wave_result_schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                visibility_ceiling=exploration_plan.visibility_ceiling,
                parent_visibility_ceilings=(
                    exploration_plan.visibility_ceiling,
                    exploration_plan.visibility_ceiling,
                ),
                outcome_access=exploration_plan.outcome_access,
            ),
        ),
        resource_lock_ids=("exploration-read-only",),
        barrier=BarrierKind.NONE,
        maximum_attempts=1,
        obligation_ids=(
            "receipt-bound-findings",
            "receipt-bound-skeptic-report",
            "retain-competing-hypotheses",
        ),
    )
    aggregate_budget = _aggregate_resource_budgets(
        tuple(
            (
                *(
                    proposal.analysis.budget
                    for proposal in exploration_plan.proposals
                    if proposal.proposal_id in selected_ids
                ),
                skeptic_budget,
                synthesis_budget,
            )
        )
    )
    if not exploration_plan.budget.contains(aggregate_budget):
        raise ProtocolCompilationError("exploration execution exceeds its frozen budget")
    return ProtocolExecutionPlan(
        execution_plan_id=f"execution.{exploration_plan.plan_id}",
        lane=PlanLane.EXPLORATORY,
        source_plan=ObjectIdentity.from_record(
            exploration_plan.plan_id,
            exploration_plan,
        ),
        registry_sha256=registry.fingerprint(),
        implementation_commit=implementation_commit,
        tasks=tuple(
            sorted(
                (*analysis_tasks, skeptic_task, synthesis_task),
                key=lambda task: task.task_id,
            )
        ),
        nonactuating=True,
    )


def _aggregate_resource_budgets(
    budgets: tuple[ResourceBudget, ...],
) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=max(budget.cpu_cores for budget in budgets),
        memory_bytes=max(budget.memory_bytes for budget in budgets),
        gpu_devices=max(budget.gpu_devices for budget in budgets),
        wall_time_seconds=sum(budget.wall_time_seconds for budget in budgets),
        source_scan_bytes=sum(budget.source_scan_bytes for budget in budgets),
        output_bytes=sum(budget.output_bytes for budget in budgets),
    )
