"""Shared public-route authoring assembly; adapters supply scientific declarations."""

from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import LifecycleStatus, ReadinessStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.worlds import EvidenceUnitScope
from empirical_lawhood.planning.campaigns import (
    CampaignLane,
    CampaignNode,
    CampaignNodeKind,
    CampaignSpec,
    DecisionRight,
)
from empirical_lawhood.planning.experiment_entry import ExperimentEntryChecklist, ExperimentEntryPackage, ExperimentEntryRequirement, ExperimentEntryRequirementBinding, ExperimentEntryTransition, ExperimentTerminalClass, StudyDefinition
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodBinding,
    FormalMethodCatalog,
    FormalMethodRole,
    derive_formal_gap_applicability,
)
from empirical_lawhood.planning.formal_gaps import (
    FormalDomain,
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
    FormalGapSpec,
)
from empirical_lawhood.planning.study_authoring import StudyDraft
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, StandardCandidateCompilationContext, required_candidate_obligation_ids
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolTemplate,
    ScientificInputRole,
    ScientificStage,
)
from empirical_lawhood.adapters.composition.protocol_helpers import CANONICAL_MEDIA_TYPE


def single_experiment_campaign(
    system: SystemSpec,
    experiment: ExperimentSpec,
    *,
    prefix: str,
    budget: ResourceBudget,
    objective: str,
    actions: tuple[tuple[str, AuthorityAction], ...],
) -> CampaignSpec:
    node = CampaignNode(
        f"{prefix}.experiment-node",
        CampaignNodeKind.EXPERIMENT_SPEC,
        CampaignLane.PROSPECTIVE,
        ObjectIdentity.from_record(experiment.experiment_id, experiment),
        (),
        LifecycleStatus.ACTIVE,
    )
    rights = tuple(
        DecisionRight(
            f"{prefix}.right.{label}",
            action,
            system.authority_policy.delegate_id,
            system.authority_policy.policy_id,
            True,
        )
        for label, action in actions
    )
    return CampaignSpec(
        f"{prefix}.campaign",
        objective,
        (system.system_id,),
        (system.world.world_id,),
        tuple(claim.claim_id for claim in experiment.claims),
        budget,
        ObjectIdentity.from_record(system.authority_policy.policy_id, system.authority_policy),
        rights,
        (node,),
        (node.node_id,),
        (node.node_id,),
        (),
        LifecycleStatus.ACTIVE,
    )


def study_template_from_protocol(
    protocol: ProtocolTemplate,
    source: CanonicalRecord,
    source_id: str,
    experiment: ExperimentSpec,
    registry: CapabilityRegistry,
    *,
    prefix: str,
    source_task_id: str,
    terminal_task_id: str,
    primary_output_ids: dict[ScientificStage, str],
) -> StudyTemplate:
    """Bind ordinary authoring graph/coverage metadata to adapter-owned steps."""
    steps = protocol.steps
    nodes = tuple(
        CandidateGraphNode(
            step.step_id,
            step.stage,
            step.capability_key,
            step.capability_version,
            registry.resolve(step.capability_key, step.capability_version).implementation_sha256,
            step.fingerprint(),
            step.obligation_ids,
            step.requested_outcome_access,
            step.visibility_ceiling,
            step.resource_budget,
        )
        for step in steps
    )
    external = CandidateGraphExternalInput(
        source_id,
        ScientificInputRole.MODEL,
        source_id,
        ContentIdentityPolicy.EXACT_SHA256,
        source.fingerprint(),
        source.SCHEMA,
        CANONICAL_MEDIA_TYPE,
        len(source.canonical_bytes()),
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
    )
    edges = [
        CandidateGraphEdge(
            f"{prefix}.edge.source-config",
            None,
            None,
            external.input_id,
            source_task_id,
            "source-config",
            ScientificInputRole.MODEL,
            external.logical_artifact_id,
            external.payload_schema,
            external.media_type,
            external.maximum_size_bytes,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            BarrierKind.NONE,
        )
    ]
    by_id = {step.step_id: step for step in steps}
    for child in steps:
        for parent_id in child.dependency_step_ids:
            parent = by_id[parent_id]
            for output in parent.outputs:
                edges.append(
                    CandidateGraphEdge(
                        f"edge.{parent_id}.{output.output_id}.{child.step_id}",
                        parent_id,
                        output.output_id,
                        None,
                        child.step_id,
                        f"input-{parent_id}-{output.output_id}",
                        ScientificInputRole.OUTCOME,
                        f"artifact.{parent_id}.{output.output_id}",
                        output.payload_schema,
                        output.media_type,
                        parent.resource_budget.output_bytes,
                        child.requested_outcome_access,
                        child.visibility_ceiling,
                        child.barrier,
                    )
                )
    graph = CandidateScientificGraph(
        f"{prefix}.graph", (external,), nodes, tuple(sorted(edges, key=lambda value: value.edge_id))
    )
    owners = {
        obligation: (
            step.step_id,
            primary_output_ids[step.stage],
        )
        for step in steps
        for obligation in step.obligation_ids
    }
    coverage = tuple(
        ObligationCoverageBinding(
            obligation,
            *owners.get(
                obligation, (terminal_task_id, primary_output_ids[ScientificStage.EVALUATE])
            ),
            tuple(
                sorted(
                    edge.edge_id
                    for edge in graph.edges
                    if edge.consumer_node_id
                    == owners.get(
                        obligation, (terminal_task_id, primary_output_ids[ScientificStage.EVALUATE])
                    )[0]
                )
            ),
        )
        for obligation in required_candidate_obligation_ids(experiment, protocol)
    )
    return StudyTemplate(
        f"{prefix}.template",
        "1.0.0",
        protocol,
        graph,
        ObligationCoverage(
            f"{prefix}.coverage", tuple(sorted(coverage, key=lambda value: value.obligation_id))
        ),
    )


def formal_entry(
    draft: StudyDraft,
    context: CandidateCompilationContext,
    science: ObjectIdentity,
    *,
    prefix: str,
    minimum_units: int,
    operand_description: str,
    estimator: str,
    uncertainty: str,
    ceiling: EvidenceCeiling,
    evaluator: CapabilityManifest,
    input_schema: str,
    output_schema: str,
    world: FormalGapEvidenceWorld,
    minimum_views: int,
    evidence_unit_ids: tuple[str, ...] | None = None,
) -> tuple[StudyDefinition, StandardCandidateCompilationContext]:
    assert draft.system is not None and draft.experiment is not None
    gaps = tuple(
        FormalGapSpec(
            f"{prefix}.gap.{domain.value.lower()}",
            domain,
            operand_description.format(domain=domain.value.lower()),
            (f"{prefix}.native-operands",),
            (world,),
            minimum_units,
            minimum_views,
            (f"{prefix}.{estimator}",),
            (draft.experiment.controls[0].control_id,),
            (f"{prefix}.falsifier",),
            (f"{prefix}.fixed-roster",),
            f"{prefix}.{uncertainty}",
            ceiling,
            (science.object_id,),
        )
        for domain in FormalDomain
    )
    register = FormalGapRegister(
        f"{prefix}.formal-register",
        "1.0.0",
        (science,),
        tuple(sorted(gaps, key=lambda value: value.gap_id)),
    )
    inventory = FormalGapSourceCapabilityInventory(
        f"{prefix}.source-inventory",
        draft.system.system_id,
        world,
        tuple(value.materialization for value in draft.source_materializations),
        (f"{prefix}.native-operands",),
        (f"{prefix}.fixed-roster",),
        draft.evaluation_unit_ids if evidence_unit_ids is None else evidence_unit_ids,
        EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        tuple(value.view_id for value in draft.system.numerical_views),
        (f"{prefix}.{estimator}",),
        (draft.experiment.controls[0].control_id,),
        (f"{prefix}.{uncertainty}",),
        (),
        (),
        ceiling,
        OutcomeAccess.OUTCOME_BLIND,
    )
    applicability = derive_formal_gap_applicability(register, inventory)
    template = context.template(draft.dag_template_key)
    assert template is not None
    terminal = next(
        value
        for value in template.coverage.bindings
        if value.obligation_id == f"{prefix}.single-terminal"
    )
    assignments = tuple(
        FormalGapCoverageAssignment(
            gap.gap_id,
            FormalGapCoverageDisposition.TEST_IN_THIS_ACT,
            None,
            (),
            f"{prefix}.{estimator}",
            (draft.experiment.controls[0].control_id,),
            f"{prefix}.{uncertainty}",
            (terminal.obligation_id,),
            (terminal.required_output_id,),
            (terminal.proof_owner_node_id,),
        )
        for gap in gaps
    )
    coverage = FormalGapCoverage(
        f"{prefix}.formal-coverage",
        ObjectIdentity.from_record(register.register_id, register),
        draft.system.system_id,
        draft.draft_id,
        applicability,
        tuple(sorted(assignments, key=lambda value: value.gap_id)),
        OutcomeAccess.OUTCOME_BLIND,
    )
    requirements = tuple(
        ExperimentEntryRequirementBinding(
            requirement,
            (f"{prefix}.entry.{requirement.value.lower()}",),
            ReadinessStatus.AUTHORITY_REQUIRED
            if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
            else ReadinessStatus.READY,
            ("EXECUTION_AND_REVEAL_AUTHORITY_SEPARATE",)
            if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
            else (),
        )
        for requirement in sorted(ExperimentEntryRequirement, key=lambda value: value.value)
    )
    checklist = ExperimentEntryChecklist(
        f"{prefix}.entry-checklist",
        ObjectIdentity.from_record(draft.draft_id, draft),
        ObjectIdentity.from_record(register.register_id, register),
        ObjectIdentity.from_record(coverage.coverage_id, coverage),
        world,
        f"{prefix}.public-execution-route",
        f"{prefix}.receipt-first-recovery",
        requirements,
        tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        tuple(sorted(ExperimentTerminalClass, key=lambda value: value.value)),
        OutcomeAccess.OUTCOME_BLIND,
    )
    authoring = StudyDefinition(
        f"{prefix}.authoring-base",
        draft,
        ExperimentEntryPackage(f"{prefix}.entry", register, coverage, checklist),
    )
    methods = tuple(
        FormalMethodBinding(
            key,
            role,
            evaluator.capability_key,
            evaluator.capability_version,
            evaluator.implementation_sha256,
            tuple(sorted(gap.gap_id for gap in gaps)),
            input_schema,
            output_schema,
            ceiling,
            OutcomeAccess.EVALUATOR_REVEAL,
        )
        for role, key in (
            (FormalMethodRole.ESTIMATOR, f"{prefix}.{estimator}"),
            (FormalMethodRole.MULTIPLICITY, f"{prefix}.{uncertainty}"),
        )
    )
    return authoring, StandardCandidateCompilationContext(
        f"{prefix}.standard-context",
        context,
        FormalMethodCatalog(
            f"{prefix}.formal-method-catalog",
            tuple(sorted(methods, key=lambda value: value.binding_id)),
        ),
        (inventory,),
    )
