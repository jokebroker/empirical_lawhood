"""Strict authoring, candidate catalog and exact scientific DAG for physical scale morphism."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.reference_worlds.builders import build_reference_system
from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
from empirical_lawhood.kernel.evidence import (
    ClaimSpec,
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.experiments import (
    AssignmentKind,
    AssignmentSpec,
    ControlKind,
    ControlSpec,
    ExperimentSpec,
    PrecisionGoal,
    RevealBarrierSpec,
)
from empirical_lawhood.kernel.obligations import (
    ClosureSpec,
    ComputabilityEvidence,
    FalsifierKind,
    FalsifierSpec,
    ObligationStatus,
    ScientificObligations,
    StructuralConvergenceSpec,
    SupportSpec,
    UncertaintySpec,
    ValiditySpec,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import QuantityBound
from empirical_lawhood.kernel.status import LifecycleStatus, ReadinessStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.kernel.worlds import EvidenceUnitScope
from empirical_lawhood.planning.campaigns import (
    CampaignLane,
    CampaignNode,
    CampaignNodeKind,
    CampaignSpec,
    DecisionRight,
)
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.planning.experiment_entry import ExperimentEntryChecklist, ExperimentEntryPackage, ExperimentEntryRequirement, ExperimentEntryRequirementBinding, ExperimentEntryTransition, ExperimentTerminalClass, StudyDefinition
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodBinding,
    FormalMethodCatalog,
    FormalMethodRole,
    derive_formal_gap_applicability,
)
from empirical_lawhood.planning.formal_gaps import (
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
)
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, StandardCandidateCompilationContext, ScientificInputRole, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .provider import PhysicalScaleMorphismStudyConfig, default_study_config
from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.provider import PhysicalScaleMorphismTruthCaseBatch, PhysicalScaleMorphismTruthObservationBatch
from .runtime_provider import PHYSICAL_SCALE_MORPHISM_SOURCE_ARTIFACT_ID, PHYSICAL_SCALE_MORPHISM_SOURCE_ID, PhysicalScaleMorphismCapabilityAdapter, MORPHISM_METHOD_KEY, physical_scale_morphism_protocol_template, physical_scale_morphism_registry


PHYSICAL_SCALE_MORPHISM_DEVELOPMENT_UNIT_IDS = tuple(
    f"case.{fixture}.deterministic"
    for fixture in (
        "boundary-topology",
        "composition-discrimination",
        "energy-mean-collision",
        "exact-receiver-composition",
        "future-closure",
        "guard-false-safe",
        "heterogeneous-local-support",
        "hidden-history",
        "hold-state-grammar",
        "leakage-firewall",
        "numerical-qualification",
        "power-and-precision",
        "saturation-coordinate",
        "scale-morphism",
        "unit-clock-grouping",
    )
)
PHYSICAL_SCALE_MORPHISM_EVALUATION_UNIT_IDS = tuple(
    value.replace(".deterministic", ".noisy-repeated-board") for value in PHYSICAL_SCALE_MORPHISM_DEVELOPMENT_UNIT_IDS
)
PHYSICAL_SCALE_MORPHISM_DEVELOPMENT_SEED_IDS = ("seed.physical-scale-morphism-development.42017",)
PHYSICAL_SCALE_MORPHISM_EVALUATION_SEED_IDS = ("seed.physical-scale-morphism-evaluation.90173",)
PHYSICAL_SCALE_MORPHISM_FORMAL_GAP_IDS = (
    "gap.algebra.cross-context-recurrence",
    "gap.algebra.falsifier-preservation",
    "gap.algebra.quotient-lumpability",
    "gap.algebra.restriction-gluing-transport",
    "gap.calculus.numerical-view-convergence",
    "gap.calculus.time-scaling",
    "gap.calculus.uncertainty-propagation",
    "gap.dynamics.causal-cones-clock-transport",
    "gap.dynamics.preservation-barriers",
    "gap.dynamics.state-closure-memory",
    "gap.geometry.admission-margins",
    "gap.geometry.boundary-strata",
    "gap.geometry.reachability-viability",
    "gap.geometry.receiver-fibers",
    "gap.geometry.topology-restriction",
)

_TASK_BUDGET = ResourceBudget(
    cpu_cores=2,
    memory_bytes=2 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=300,
    source_scan_bytes=16 * 1024**2,
    output_bytes=16 * 1024**2,
)
_PROGRAMME_BUDGET = ResourceBudget(
    cpu_cores=2,
    memory_bytes=2 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=7 * _TASK_BUDGET.wall_time_seconds,
    source_scan_bytes=7 * _TASK_BUDGET.source_scan_bytes,
    output_bytes=7 * _TASK_BUDGET.output_bytes,
)


def physical_scale_morphism_system() -> SystemSpec:
    system = build_reference_system(
        "physical-scale-morphism-synthetic-reference",
        history_dependent=True,
        numerical=False,
    )
    return replace(
        system,
        authority_policy=replace(
            system.authority_policy,
            policy_id="physical-scale-morphism-synthetic-authority-policy",
            scope_ids=("physical-scale-morphism-synthetic-ip1-ip5",),
            budget_ceiling=_PROGRAMME_BUDGET,
        ),
    )


def _obligations(system: SystemSpec) -> ScientificObligations:
    view_ids = tuple(value.view_id for value in system.numerical_views)
    return ScientificObligations(
        obligations_id="physical-scale-morphism-synthetic-obligations",
        support=SupportSpec(
            support_id="physical-scale-morphism-synthetic-support",
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=len(PHYSICAL_SCALE_MORPHISM_EVALUATION_UNIT_IDS),
            nested_numerical_view_count=len(view_ids),
            information_cutoff_id="physical-scale-morphism-pre-outcome-cutoff",
            chart_ids=("physical-scale-morphism-synthetic-action-chart",),
            denominator_cell_ids=("physical-scale-morphism-synthetic-truth-cell",),
            action_bounds=(
                QuantityBound(
                    bound_id="physical-scale-morphism-synthetic-action-bound",
                    quantity_id="action",
                    native_unit="1",
                    lower=Decimal("-1"),
                    upper=Decimal("1"),
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id="physical-scale-morphism-synthetic-validity",
            validity_domain_ids=("physical-scale-morphism-synthetic-truth-cell",),
            assumption_ids=("planted-truth-worlds-and-rc-denominator-exact",),
            exclusion_reason_codes=("NO_PHYSICAL_EVIDENCE",),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id="physical-scale-morphism-synthetic-uncertainty",
            method_key="physical-scale-morphism-frozen-deterministic-and-noisy-fixtures",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal("0.95"),
            interval_quantity_ids=("receiver",),
            limitation_codes=("ANALYTIC_AND_NUMERICAL_REFERENCE_ONLY",),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=(
            FalsifierSpec(
                falsifier_id="physical-scale-morphism-decisive-negative-controls",
                kind=FalsifierKind.WRONG_ACTION,
                capability_key="physical-scale-morphism.receiver-morphism-method",
                description=(
                    "Reject task saturation, future leakage, false-safe hold, "
                    "wrong numerical denominators and aggregate-only claims."
                ),
                decisive_rule="Every frozen negative fixture emits its exact opposition code.",
                status=ObligationStatus.REQUIRED,
            ),
        ),
        closure=ClosureSpec(
            closure_id="physical-scale-morphism-morphism-and-history-closure",
            recurrence_cell_ids=("physical-scale-morphism-synthetic-truth-cell",),
            exchange_factor_ids=system.relation.denominator_quantity_ids,
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id="physical-scale-morphism-numerical-and-composition-convergence",
            required_structure_ids=(
                "boundary-complex",
                "morphism-composition",
                "receiver-fibres",
            ),
            numerical_view_ids=view_ids,
            tolerances=(),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id="physical-scale-morphism-synthetic-computability",
            envelope_id=system.computability_envelopes[0].envelope_id,
            numerical_view_ids=view_ids,
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=("physical-scale-morphism-synthetic-resource-preflight",),
        ),
    )


def physical_scale_morphism_experiment(system: SystemSpec) -> ExperimentSpec:
    claim = ClaimSpec(
        claim_id="physical-scale-morphism-synthetic-method-conformance-claim",
        world_id=system.world.world_id,
        relation_id=system.relation.relation_id,
        proposition=(
            "The frozen method distinguishes the predicted survivors and failures "
            "of receiver, numerical and physical-scale morphisms on planted worlds."
        ),
        estimand="Exact recovery of all thirty frozen truth cases and numerical/source controls.",
        physical_independent_unit_id=system.independent_unit.unit_id,
        requested_rung=None,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        promotion_rule="No promotion; synthetic method/runtime qualification only.",
        assumption_ids=("planted-truth-worlds-and-rc-denominator-exact",),
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
    )
    cutoff = InformationCutoff(
        cutoff_id="physical-scale-morphism-pre-outcome-cutoff",
        clock_id=system.clocks[0].clock_id,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal(0),
    )
    return ExperimentSpec(
        experiment_id="physical-scale-morphism-synthetic-experiment",
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=(claim,),
        assignment=AssignmentSpec(
            assignment_id="physical-scale-morphism-synthetic-assignment",
            kind=AssignmentKind.SIMULATOR_INTERVENTION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism="Frozen analytic truth cases and in-memory RC reference computations.",
            support_restriction_ids=(),
        ),
        measurement_quantity_ids=("receiver", "sink"),
        controls=(
            ControlSpec(
                control_id="control.physical-scale-morphism-frozen-counterfeit-roster",
                kind=ControlKind.WRONG_ACTION,
                capability_key="physical-scale-morphism.receiver-morphism-method",
                target_quantity_ids=("receiver", "sink"),
                decisive_rule="All frozen counterfeit fixtures must be opposed exactly.",
            ),
            ControlSpec(
                control_id="control.negative-action",
                kind=ControlKind.NEGATIVE_ACTION,
                capability_key="physical-scale-morphism.receiver-morphism-method",
                target_quantity_ids=("receiver", "sink"),
                decisive_rule="Signed and false-safe controls must retain opposition.",
            ),
            ControlSpec(
                control_id="control.support-matched-comparator",
                kind=ControlKind.BASELINE_COMPARATOR,
                capability_key="physical-scale-morphism.receiver-morphism-method",
                target_quantity_ids=("receiver", "sink"),
                decisive_rule="Every comparison must remain on frozen matched support.",
            ),
        ),
        precision_goals=(
            PrecisionGoal(
                goal_id="physical-scale-morphism-exact-thirty-case-recovery",
                metric_id="exact-case-recovery",
                target_width=Decimal("0.000001"),
                native_unit="1",
                maximum_independent_units=30,
                stopping_rule="Stop after the frozen thirty-case roster; never retune.",
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id="physical-scale-morphism-truth-reveal",
            development_unit_ids=PHYSICAL_SCALE_MORPHISM_DEVELOPMENT_UNIT_IDS,
            evaluation_cohort_id="physical-scale-morphism-noisy-evaluation-cohort",
            evaluation_manifest_sha256=sha256(
                "\n".join(PHYSICAL_SCALE_MORPHISM_EVALUATION_UNIT_IDS).encode()
            ).hexdigest(),
            sealed_outcome_artifact_ids=("artifact.physical-scale-morphism-truth-labels",),
            evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            fresh_evidence=True,
        ),
        obligations=_obligations(system),
        design_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
    )


def physical_scale_morphism_study(system: SystemSpec, experiment: ExperimentSpec) -> CampaignSpec:
    node = CampaignNode(
        node_id="campaign-node.physical-scale-morphism-synthetic-experiment",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id="campaign.physical-scale-morphism-synthetic",
        objective=(
            "Qualify physical scale morphism method, numerical twin, strict source adapter and "
            "receipt-first runtime through IP-5 without physical execution."
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(value.claim_id for value in experiment.claims),
        budget=_PROGRAMME_BUDGET,
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id, system.authority_policy
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id="decision-right.physical-scale-morphism-synthetic-execution",
                action=AuthorityAction.REFERENCE_WORLD_EXECUTION,
                decision_maker_id=system.authority_policy.delegate_id,
                authority_policy_id=system.authority_policy.policy_id,
                delegated=True,
            ),
        ),
        nodes=(node,),
        root_node_ids=(node.node_id,),
        active_node_ids=(node.node_id,),
        evidence_state=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )


def _output(protocol: ProtocolTemplate, step_id: str, output_id: str):  # type: ignore[no-untyped-def]
    step = next(value for value in protocol.steps if value.step_id == step_id)
    return next(value for value in step.outputs if value.output_id == output_id)


def _internal_edge(
    protocol: ProtocolTemplate,
    *,
    producer: str,
    output_id: str,
    consumer: str,
    consumer_input: str,
    role: ScientificInputRole,
    outcome_access: OutcomeAccess | None = None,
) -> CandidateGraphEdge:
    output = _output(protocol, producer, output_id)
    producer_step = next(value for value in protocol.steps if value.step_id == producer)
    consumer_step = next(value for value in protocol.steps if value.step_id == consumer)
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
        maximum_size_bytes=16 * 1024**2,
        outcome_access=outcome_access or producer_step.requested_outcome_access,
        visibility_ceiling=producer_step.visibility_ceiling,
        barrier=consumer_step.barrier,
    )


def physical_scale_morphism_scientific_graph(
    *, protocol: ProtocolTemplate, registry: CapabilityRegistry
) -> CandidateScientificGraph:
    manifests = {value.registry_id: value for value in registry.capabilities}
    source = CandidateGraphExternalInput(
        input_id=PHYSICAL_SCALE_MORPHISM_SOURCE_ID,
        scientific_role=ScientificInputRole.PREPARED_MEDIUM,
        logical_artifact_id=PHYSICAL_SCALE_MORPHISM_SOURCE_ARTIFACT_ID,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=protocol.steps[0].config.content_sha256,
        payload_schema=PhysicalScaleMorphismStudyConfig.SCHEMA,
        media_type="application/json",
        maximum_size_bytes=16 * 1024**2,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
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
    edges = [
        _internal_edge(
            protocol,
            producer="truth-generate",
            output_id="truth-cases",
            consumer="run-morphism-method",
            consumer_input="truth-blind-cases",
            role=ScientificInputRole.PREPARED_MEDIUM,
        ),
        _internal_edge(
            protocol,
            producer="truth-generate",
            output_id="truth-cases",
            consumer="truth-evaluate",
            consumer_input="case-identities",
            role=ScientificInputRole.QUALIFICATION,
        ),
        _internal_edge(
            protocol,
            producer="run-morphism-method",
            output_id="truth-observations",
            consumer="freeze-method",
            consumer_input="truth-blind-observations",
            role=ScientificInputRole.MODEL,
        ),
        _internal_edge(
            protocol,
            producer="run-morphism-method",
            output_id="truth-observations",
            consumer="truth-evaluate",
            consumer_input="truth-blind-observations",
            role=ScientificInputRole.MODEL,
        ),
        _internal_edge(
            protocol,
            producer="freeze-method",
            output_id="method-freeze",
            consumer="truth-evaluate",
            consumer_input="method-freeze",
            role=ScientificInputRole.QUALIFICATION,
        ),
        _internal_edge(
            protocol,
            producer="truth-evaluate",
            output_id="truth-method-result",
            consumer="adjudicate-synthetic",
            consumer_input="truth-method-result",
            role=ScientificInputRole.QUALIFICATION,
            outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
        ),
        _internal_edge(
            protocol,
            producer="qualify-numerical",
            output_id="numerical-qualification",
            consumer="adjudicate-synthetic",
            consumer_input="numerical-qualification",
            role=ScientificInputRole.QUALIFICATION,
        ),
        _internal_edge(
            protocol,
            producer="qualify-source-adapter",
            output_id="source-qualification",
            consumer="adjudicate-synthetic",
            consumer_input="source-qualification",
            role=ScientificInputRole.QUALIFICATION,
        ),
    ]
    for consumer in ("qualify-numerical", "qualify-source-adapter", "truth-generate"):
        step = next(value for value in protocol.steps if value.step_id == consumer)
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.external.{PHYSICAL_SCALE_MORPHISM_SOURCE_ID}.{consumer}.programme-root",
                producer_node_id=None,
                producer_output_id=None,
                external_input_id=PHYSICAL_SCALE_MORPHISM_SOURCE_ID,
                consumer_node_id=consumer,
                consumer_input_id="programme-root",
                scientific_role=source.scientific_role,
                logical_artifact_id=source.logical_artifact_id,
                payload_schema=source.payload_schema,
                media_type=source.media_type,
                maximum_size_bytes=source.maximum_size_bytes,
                outcome_access=source.outcome_access,
                visibility_ceiling=source.visibility_ceiling,
                barrier=step.barrier,
            )
        )
    return CandidateScientificGraph(
        graph_id="graph.physical-scale-morphism-synthetic",
        external_inputs=(source,),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _coverage(
    experiment: ExperimentSpec,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> ObligationCoverage:
    protocol_owners = {
        obligation_id: (step.step_id, step.outputs[0].output_id)
        for step in protocol.steps
        for obligation_id in step.obligation_ids
    }
    terminal_edges = tuple(
        sorted(
            value.edge_id
            for value in graph.edges
            if value.consumer_node_id == "adjudicate-synthetic"
        )
    )
    bindings = []
    for obligation_id in required_candidate_obligation_ids(experiment, protocol):
        protocol_owner = protocol_owners.get(obligation_id)
        owner, output_id = protocol_owner or (
            "adjudicate-synthetic",
            "synthetic-readiness",
        )
        contributors = tuple(
            sorted(value.edge_id for value in graph.edges if value.consumer_node_id == owner)
        )
        bindings.append(
            ObligationCoverageBinding(
                obligation_id=obligation_id,
                proof_owner_node_id=owner,
                required_output_id=output_id,
                contributor_edge_ids=(
                    contributors if protocol_owner is not None else terminal_edges
                ),
            )
        )
    return ObligationCoverage(
        coverage_id="coverage.physical-scale-morphism-synthetic",
        bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
    )


def physical_scale_morphism_candidate_catalog(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    experiment: ExperimentSpec,
) -> CandidateCapabilityCatalog:
    graph = physical_scale_morphism_scientific_graph(protocol=protocol, registry=registry)
    template = StudyTemplate(
        template_key="physical-scale-morphism.synthetic",
        template_version="1.0.0",
        protocol=protocol,
        graph=graph,
        coverage=_coverage(experiment, protocol, graph),
    )
    registrations = tuple(
        CandidateCapabilityRegistration(
            manifest=value,
            provider_key=value.capability_key,
            provider_version=value.capability_version,
            config_media_type="application/json",
            maximum_config_bytes=16 * 1024**2,
        )
        for value in registry.capabilities
    )
    return CandidateCapabilityCatalog(
        catalog_id="catalog.physical-scale-morphism-synthetic",
        registrations=registrations,
        templates=(template,),
    )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismAuthoringBundle:
    config: PhysicalScaleMorphismStudyConfig
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    protocol: ProtocolTemplate
    registry: CapabilityRegistry
    catalog: CandidateCapabilityCatalog
    qualification: MaterializationQualificationReceipt
    design_input: DesignInputRecord
    draft: StudyDraft
    context: CandidateCompilationContext


def build_physical_scale_morphism_authoring_bundle(*, implementation_sha256: str) -> PhysicalScaleMorphismAuthoringBundle:
    config = default_study_config()
    system = physical_scale_morphism_system()
    experiment = physical_scale_morphism_experiment(system)
    campaign = physical_scale_morphism_study(system, experiment)
    registry = physical_scale_morphism_registry(implementation_sha256=implementation_sha256)
    protocol = physical_scale_morphism_protocol_template(registry=registry, config=config)
    catalog = physical_scale_morphism_candidate_catalog(protocol=protocol, registry=registry, experiment=experiment)
    observation_operator = ObjectIdentity(
        object_id="reference.physical-scale-morphism-synthetic-materializer",
        object_schema='empirical-lawhood/methods/physical-scale-morphism/synthetic-materializer',
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id="qualification.physical-scale-morphism-synthetic-root",
        source_id=PHYSICAL_SCALE_MORPHISM_SOURCE_ID,
        materialization=ObjectIdentity.from_record(config.config_id, config),
        content_sha256=config.fingerprint(),
        evidence_world_id=system.world.world_id,
        observation_operator=observation_operator,
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
        native_unit_ids=tuple(sorted({value.native_unit for value in system.quantities})),
        frame_ids=tuple(sorted({value.coordinate_frame for value in system.quantities})),
        clock_ids=tuple(value.clock_id for value in system.clocks),
        receiver_semantics_id=system.relation.relation_id,
        validity_contract_id=experiment.obligations.validity.validity_id,
        uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
        access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    design_input = DesignInputRecord(
        input_id="design-input.physical-scale-morphism-amended-tranche",
        object_identity=ObjectIdentity.from_record(system.system_id, system),
        materialization_sha256=system.fingerprint(),
        information_cutoff=experiment.information_cutoffs[0],
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    draft = StudyDraft(
        draft_id="draft.physical-scale-morphism-synthetic-implementation-rehearsal",
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            "Does the frozen physical scale morphism implementation recover its planted map, "
            "forecast, semantic, boundary and negative-control contract without "
            "crossing the independent physical gate?"
        ),
        alternative_ids=(
            "alternative.physical-scale-morphism-method-runtime-conforms",
            "alternative.physical-scale-morphism-method-runtime-fails",
        ),
        design_origin=DesignOrigin(
            origin_id="origin.physical-scale-morphism-amended-tranche",
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=(design_input.input_id,),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=(design_input,),
        development_unit_ids=PHYSICAL_SCALE_MORPHISM_DEVELOPMENT_UNIT_IDS,
        evaluation_unit_ids=PHYSICAL_SCALE_MORPHISM_EVALUATION_UNIT_IDS,
        development_seed_ids=PHYSICAL_SCALE_MORPHISM_DEVELOPMENT_SEED_IDS,
        evaluation_seed_ids=PHYSICAL_SCALE_MORPHISM_EVALUATION_SEED_IDS,
        unresolved_decisions=(),
        system=system,
        experiment=experiment,
        campaign=campaign,
        dag_template_key="physical-scale-morphism.synthetic",
        capability_selections=tuple(
            CapabilitySelection(
                capability_key=value.capability_key,
                capability_version=value.capability_version,
                implementation_sha256=value.implementation_sha256,
            )
            for value in registry.capabilities
        ),
        source_materializations=(
            SourceMaterializationRef(
                source_id=PHYSICAL_SCALE_MORPHISM_SOURCE_ID,
                role=SourceMaterializationRole.PREPARED_MEDIUM,
                evidence_world_id=system.world.world_id,
                materialization=ObjectIdentity.from_record(config.config_id, config),
                content_sha256=config.fingerprint(),
                source_config_sha256=config.fingerprint(),
                observation_operator=observation_operator,
                numerical_view_ids=qualification.numerical_view_ids,
                qualification_receipt=ObjectIdentity.from_record(
                    qualification.receipt_id, qualification
                ),
                access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            ),
        ),
        resource_ceiling=_PROGRAMME_BUDGET,
    )
    context = CandidateCompilationContext(
        context_id="context.physical-scale-morphism-synthetic",
        registry=registry,
        templates=catalog.templates,
        qualifications=(qualification,),
        known_design_inputs=(design_input,),
        implementation_sha256=implementation_sha256,
    )
    return PhysicalScaleMorphismAuthoringBundle(
        config=config,
        system=system,
        experiment=experiment,
        campaign=campaign,
        protocol=protocol,
        registry=registry,
        catalog=catalog,
        qualification=qualification,
        design_input=design_input,
        draft=draft,
        context=context,
    )


def physical_scale_morphism_candidate_config_decoders(
    catalog: CandidateCapabilityCatalog, *, config: PhysicalScaleMorphismStudyConfig
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    return tuple(PhysicalScaleMorphismCapabilityAdapter(value.manifest, config) for value in catalog.registrations)


def physical_scale_morphism_formal_source_inventory(
    *, base: PhysicalScaleMorphismAuthoringBundle, register: FormalGapRegister
) -> FormalGapSourceCapabilityInventory:
    selected = {
        value.gap_id: value for value in register.gaps if value.gap_id in PHYSICAL_SCALE_MORPHISM_FORMAL_GAP_IDS
    }
    if set(selected) != set(PHYSICAL_SCALE_MORPHISM_FORMAL_GAP_IDS):
        raise ValueError("current formal register lacks an physical scale morphism priority gap")
    all_gap_ids = {value.gap_id for value in register.gaps}
    return FormalGapSourceCapabilityInventory(
        inventory_id="formal-source-inventory.physical-scale-morphism-synthetic",
        denominator_id=base.system.system_id,
        evidence_world=FormalGapEvidenceWorld.ANALYTIC_REFERENCE,
        source_materializations=tuple(
            sorted(
                (value.materialization for value in base.draft.source_materializations),
                key=lambda value: value.object_id,
            )
        ),
        present_operand_ids=tuple(
            sorted({operand for gap in selected.values() for operand in gap.required_operand_ids})
        ),
        satisfied_prerequisite_ids=tuple(
            sorted(
                {
                    prerequisite
                    for gap in selected.values()
                    for prerequisite in gap.support_prerequisite_ids
                }
            )
        ),
        independent_unit_ids=tuple(
            sorted((*base.draft.development_unit_ids, *base.draft.evaluation_unit_ids))
        ),
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=tuple(value.view_id for value in base.system.numerical_views),
        available_estimator_family_ids=tuple(
            sorted({family for gap in selected.values() for family in gap.estimator_family_ids})
        ),
        available_control_ids=tuple(
            sorted({control for gap in selected.values() for control in gap.control_ids})
        ),
        multiplicity_family_ids=tuple(
            sorted({gap.multiplicity_family_id for gap in selected.values()})
        ),
        denominator_inapplicable_gap_ids=tuple(sorted(all_gap_ids - set(selected))),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def physical_scale_morphism_formal_coverage(
    *,
    base: PhysicalScaleMorphismAuthoringBundle,
    register: FormalGapRegister,
    inventory: FormalGapSourceCapabilityInventory,
) -> FormalGapCoverage:
    selected = set(PHYSICAL_SCALE_MORPHISM_FORMAL_GAP_IDS)
    assignments = []
    for gap in register.gaps:
        if gap.gap_id in selected:
            assignments.append(
                FormalGapCoverageAssignment(
                    gap_id=gap.gap_id,
                    disposition=FormalGapCoverageDisposition.TEST_IN_THIS_ACT,
                    readiness_reason=None,
                    reason_codes=(),
                    selected_estimator_family_id=gap.estimator_family_ids[0],
                    selected_control_ids=gap.control_ids,
                    selected_multiplicity_family_id=gap.multiplicity_family_id,
                    obligation_ids=("physical-scale-morphism-adjudicate-synthetic-contract",),
                    output_ids=("scientific-adjudication",),
                    adjudication_owner_ids=("adjudicate-synthetic",),
                )
            )
        else:
            assignments.append(
                FormalGapCoverageAssignment(
                    gap_id=gap.gap_id,
                    disposition=FormalGapCoverageDisposition.NOT_APPLICABLE_TO_DENOMINATOR,
                    readiness_reason=None,
                    reason_codes=("OUTSIDE_PHYSICAL_SCALE_MORPHISM_SYNTHETIC_DENOMINATOR",),
                    selected_estimator_family_id=None,
                    selected_control_ids=(),
                    selected_multiplicity_family_id=None,
                    obligation_ids=(),
                    output_ids=(),
                    adjudication_owner_ids=(),
                )
            )
    return FormalGapCoverage(
        coverage_id="formal-gap-coverage.physical-scale-morphism-synthetic",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=base.system.system_id,
        candidate_act_id=base.draft.draft_id,
        applicability=derive_formal_gap_applicability(register, inventory),
        assignments=tuple(assignments),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def physical_scale_morphism_formal_method_catalog(
    *,
    base: PhysicalScaleMorphismAuthoringBundle,
    register: FormalGapRegister,
) -> FormalMethodCatalog:
    selected = tuple(value for value in register.gaps if value.gap_id in PHYSICAL_SCALE_MORPHISM_FORMAL_GAP_IDS)
    manifest = base.registry.resolve(MORPHISM_METHOD_KEY, "1.0.0")
    estimator_families = sorted({family for gap in selected for family in gap.estimator_family_ids})
    multiplicity_families = sorted({gap.multiplicity_family_id for gap in selected})
    bindings = []
    for role, families in (
        (FormalMethodRole.ESTIMATOR, estimator_families),
        (FormalMethodRole.MULTIPLICITY, multiplicity_families),
    ):
        for family in families:
            supported = tuple(
                sorted(
                    gap.gap_id
                    for gap in selected
                    if (
                        family in gap.estimator_family_ids
                        if role is FormalMethodRole.ESTIMATOR
                        else family == gap.multiplicity_family_id
                    )
                )
            )
            bindings.append(
                FormalMethodBinding(
                    family_id=family,
                    role=role,
                    capability_key=manifest.capability_key,
                    capability_version=manifest.capability_version,
                    implementation_sha256=manifest.implementation_sha256,
                    supported_gap_ids=supported,
                    input_schema_id=PhysicalScaleMorphismTruthCaseBatch.SCHEMA,
                    output_schema_id=PhysicalScaleMorphismTruthObservationBatch.SCHEMA,
                    maximum_claim_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                    maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                )
            )
    return FormalMethodCatalog(
        catalog_id="formal-method-catalog.physical-scale-morphism-synthetic",
        bindings=tuple(sorted(bindings, key=lambda value: value.binding_id)),
    )


def _entry_package(
    *,
    draft: StudyDraft,
    register: FormalGapRegister,
    coverage: FormalGapCoverage,
) -> ExperimentEntryPackage:
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(f"binding.physical-scale-morphism.{requirement.value.lower()}",),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                ("SYNTHETIC_EXECUTION_AND_REVEAL_AUTHORITY_SEPARATE",)
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ()
            ),
        )
        for requirement in sorted(ExperimentEntryRequirement, key=lambda value: value.value)
    )
    checklist = ExperimentEntryChecklist(
        checklist_id="entry-checklist.physical-scale-morphism-synthetic",
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.ANALYTIC_REFERENCE,
        execution_route_id="route.physical-scale-morphism-synthetic",
        durability_disposition_id="durability.synthetic-receipt-first",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id="experiment-entry-package.physical-scale-morphism-synthetic",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismStandardAuthoringBundle:
    base: PhysicalScaleMorphismAuthoringBundle
    formal_coverage: FormalGapCoverage
    formal_methods: FormalMethodCatalog
    formal_source_inventory: FormalGapSourceCapabilityInventory
    entry_package: ExperimentEntryPackage
    authoring_package: StudyDefinition
    context: StandardCandidateCompilationContext


def build_physical_scale_morphism_standard_authoring_bundle(
    *, implementation_sha256: str, register: FormalGapRegister
) -> PhysicalScaleMorphismStandardAuthoringBundle:
    base = build_physical_scale_morphism_authoring_bundle(implementation_sha256=implementation_sha256)
    inventory = physical_scale_morphism_formal_source_inventory(base=base, register=register)
    coverage = physical_scale_morphism_formal_coverage(base=base, register=register, inventory=inventory)
    methods = physical_scale_morphism_formal_method_catalog(base=base, register=register)
    entry = _entry_package(draft=base.draft, register=register, coverage=coverage)
    package = StudyDefinition(
        package_id="programme-authoring-package.physical-scale-morphism-synthetic",
        draft=base.draft,
        entry_package=entry,
    )
    context = StandardCandidateCompilationContext(
        context_id="standard-context.physical-scale-morphism-synthetic",
        base=base.context,
        formal_methods=methods,
        source_inventories=(inventory,),
    )
    return PhysicalScaleMorphismStandardAuthoringBundle(
        base=base,
        formal_coverage=coverage,
        formal_methods=methods,
        formal_source_inventory=inventory,
        entry_package=entry,
        authoring_package=package,
        context=context,
    )


__all__ = [
    'PhysicalScaleMorphismAuthoringBundle',
    'PhysicalScaleMorphismStandardAuthoringBundle',
    "PHYSICAL_SCALE_MORPHISM_DEVELOPMENT_SEED_IDS",
    "PHYSICAL_SCALE_MORPHISM_DEVELOPMENT_UNIT_IDS",
    "PHYSICAL_SCALE_MORPHISM_EVALUATION_SEED_IDS",
    "PHYSICAL_SCALE_MORPHISM_EVALUATION_UNIT_IDS",
    "PHYSICAL_SCALE_MORPHISM_FORMAL_GAP_IDS",
    'build_physical_scale_morphism_authoring_bundle',
    'build_physical_scale_morphism_standard_authoring_bundle',
    'physical_scale_morphism_study',
    'physical_scale_morphism_candidate_catalog',
    'physical_scale_morphism_candidate_config_decoders',
    'physical_scale_morphism_experiment',
    'physical_scale_morphism_formal_coverage',
    'physical_scale_morphism_formal_method_catalog',
    'physical_scale_morphism_formal_source_inventory',
    'physical_scale_morphism_scientific_graph',
    'physical_scale_morphism_system',
]
