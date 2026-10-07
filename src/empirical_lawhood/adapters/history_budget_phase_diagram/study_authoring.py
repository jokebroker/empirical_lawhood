"""Standard ProgrammeDraft and formal-entry composition for history budget phase diagram phases."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.reference_worlds.builders import build_reference_system
from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
from empirical_lawhood.kernel.evidence import (
    ClaimSpec,
    EvidenceCeiling,
    EvidenceRung,
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
from empirical_lawhood.kernel.systems import IndependentUnitSpec, SystemSpec
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.kernel.worlds import (
    EvidenceUnitScope,
    NumericalCoordinateKind,
    RandomnessSemantics,
)
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
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
)
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.candidate_compiler import (
    CandidateCompilationContext,
    StandardCandidateCompilationContext,
)
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.plans import ProtocolTemplate, ScientificInputRole
from empirical_lawhood.runtime.source_resolution import (
    SourceMaterializationConfig,
    SourceReadMode,
)

from .authoring import HISTORY_KEY, HistoryBudgetPhaseDiagramExternalRecord, RECURRENCE_KEY, UNIT_ADJUDICATOR_KEY, VERSION, candidate_catalog, history_budget_phase_diagram_phase_registry, history_budget_phase_diagram_registry, study_template, protocol_template
from .contracts import HistoryBudgetPhaseDiagramConfig, HistoryBudgetPhaseDiagramDisorderFamily, HistoryBudgetPhaseDiagramPhase, HistoryBudgetPhaseDiagramRecurrenceResult
from .descriptors import development_unit_ids, evaluation_unit_ids
from .runtime_contracts import HistoryBudgetPhaseDiagramAdjudicationBundle, HistoryBudgetPhaseDiagramGeneratorBundle


HISTORY_BUDGET_PHASE_DIAGRAM_FORMAL_GAP_IDS = (
    "gap.algebra.cross-context-recurrence",
    "gap.algebra.quotient-lumpability",
    "gap.dynamics.state-closure-memory",
    "gap.geometry.boundary-strata",
)

_PROGRAMME_BUDGET = ResourceBudget(
    cpu_cores=4,
    memory_bytes=12 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=500 * 3600,
    source_scan_bytes=24 * 1024**3,
    output_bytes=32 * 1024**3,
)

_SOURCE_ROLE_BY_SCIENTIFIC_ROLE = {
    ScientificInputRole.DENOMINATOR: SourceMaterializationRole.PREPARED_MEDIUM,
    ScientificInputRole.MODEL: SourceMaterializationRole.NUMERICAL_CONFIGURATION,
    ScientificInputRole.PREPARED_MEDIUM: SourceMaterializationRole.PREPARED_MEDIUM,
    ScientificInputRole.QUALIFICATION: SourceMaterializationRole.CALIBRATION,
    ScientificInputRole.RECEIVER: SourceMaterializationRole.RECEIVER_DEFINITION,
    ScientificInputRole.SOURCE: SourceMaterializationRole.OBSERVATION_STREAM,
}


def history_budget_phase_diagram_system() -> SystemSpec:
    base = build_reference_system(
        "history-budget-phase-diagram-numerical-simulator",
        history_dependent=True,
        numerical=True,
        latency_seconds=Decimal("1800"),
        deadline_seconds=Decimal("129600"),
    )
    template_view = base.numerical_views[0]
    coordinate = template_view.coordinates[0]
    views = tuple(
        sorted(
            (
                replace(
                    template_view,
                    view_id=f"history-budget-phase-diagram-n{scale}-view",
                    physical_preparation_id="history-budget-phase-diagram-disorder-seed-block",
                    equations_id="history-budget-phase-diagram-rc-ladder-equations",
                    closure_ids=("history-budget-phase-diagram-finite-rc-closure",),
                    boundary_condition_ids=("history-budget-phase-diagram-source-and-termination",),
                    coordinates=(
                        replace(
                            coordinate,
                            coordinate_id=f"history-budget-phase-diagram-grid-n{scale}",
                            kind=NumericalCoordinateKind.SPATIAL_GRID,
                            value=Decimal(scale),
                            unit="cells",
                            refinement_level=index,
                        ),
                    ),
                    solver_id="history-budget-phase-diagram-independent-sparse-rc-generator",
                    solver_version=VERSION,
                    precision="float64",
                    runtime_id="cpython-numpy-scipy-history-budget-phase-diagram",
                    randomness=RandomnessSemantics.GENERATIVE_PREPARATION,
                    observation_operator_id="history-budget-phase-diagram-capacitance-weighted-r8",
                    computability_envelope_id="history-budget-phase-diagram-compute-envelope",
                )
                for index, scale in enumerate((16, 32, 64, 128, 256))
            ),
            key=lambda value: value.view_id,
        )
    )
    envelope = replace(
        base.computability_envelopes[0],
        envelope_id="history-budget-phase-diagram-compute-envelope",
        represented_effect_ids=("finite-linear-rc-response", "seeded-component-disorder"),
        unresolved_effect_ids=("continuum-limit", "physical-component-discrepancy"),
        required_structure_ids=(
            "boundary-decision-collisions",
            "history-observation-rank",
            "scale-disorder-recurrence",
        ),
        computable_structure_ids=(
            "boundary-decision-collisions",
            "history-observation-rank",
            "scale-disorder-recurrence",
        ),
        max_cpu_cores=_PROGRAMME_BUDGET.cpu_cores,
        max_memory_bytes=_PROGRAMME_BUDGET.memory_bytes,
        max_wall_time_seconds=_PROGRAMME_BUDGET.wall_time_seconds,
        max_output_bytes=_PROGRAMME_BUDGET.output_bytes,
        worst_case_latency_seconds=Decimal("129600"),
        deadline_seconds=Decimal("259200"),
    )
    world = replace(
        base.world,
        label="history budget phase diagram generated RC-ladder numerical simulator world",
        represented_physics=("finite-linear-rc-response", "seeded-component-disorder"),
        unrepresented_physics=(
            "continuum-limit",
            "independent-physical-substrate",
            "physical-component-discrepancy",
        ),
        maximum_evidence=EvidenceCeiling.LOCAL_LAW,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.EVALUATION_REVEALED,
            }
        ),
    )
    authority = replace(
        base.authority_policy,
        policy_id="history-budget-phase-diagram-simulator-authority-policy",
        delegator_id="human.project-owner",
        delegate_id="history-budget-phase-diagram-nonactuating-gate",
        scope_ids=("history-budget-phase-diagram-simulator-only",),
        budget_ceiling=_PROGRAMME_BUDGET,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    return replace(
        base,
        label="history budget phase diagram support-limited numerical RC response system",
        world=world,
        independent_unit=IndependentUnitSpec(
            unit_id="history-budget-phase-diagram-disorder-seed-block",
            label="One complete disorder-family seed block with coupled finite scale views",
            grouping_key="disorder-family-seed-block-id",
        ),
        authority_policy=authority,
        computability_envelopes=(envelope,),
        numerical_views=views,
    )


def _recurrence_cell_ids() -> tuple[str, ...]:
    return tuple(
        sorted(
            f"cell.{family.value}.n{scale}.{coordinate}.{endpoint}"
            for family in HistoryBudgetPhaseDiagramDisorderFamily
            for scale in (64, 128, 256)
            for coordinate in ("k4", "k8", "b0p5", "b1")
            for endpoint in ("decision-closure", "dynamical-closure")
        )
    )


def _obligations(system: SystemSpec) -> ScientificObligations:
    view_ids = tuple(value.view_id for value in system.numerical_views)
    return ScientificObligations(
        obligations_id="history-budget-phase-diagram-scientific-obligations",
        support=SupportSpec(
            support_id="history-budget-phase-diagram-frozen-support",
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=90,
            nested_numerical_view_count=3,
            information_cutoff_id="history-budget-phase-diagram-pre-outcome-cutoff",
            chart_ids=("history-budget-phase-diagram-action-panel-5x5",),
            denominator_cell_ids=_recurrence_cell_ids(),
            action_bounds=(
                QuantityBound(
                    bound_id="history-budget-phase-diagram-action-amplitude-bound",
                    quantity_id="action",
                    native_unit="u-star",
                    lower=Decimal("0.20"),
                    upper=Decimal("1.00"),
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id="history-budget-phase-diagram-simulator-validity",
            validity_domain_ids=_recurrence_cell_ids(),
            assumption_ids=(
                "finite-linear-rc-equations",
                "float64-numpy-scipy-runtime",
                "seeded-bounded-component-fields",
            ),
            exclusion_reason_codes=tuple(
                sorted(
                    (
                        "NO_CONTINUUM_CLAIM",
                        "NO_PHYSICAL_EVIDENCE",
                        "NO_ADMISSION_OR_CONTROLLER_USE_CLAIM",
                    )
                )
            ),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id="history-budget-phase-diagram-complete-unit-uncertainty",
            method_key="history-budget-phase-diagram-exact-binomial-and-whole-block-bootstrap",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal("0.95"),
            interval_quantity_ids=(
                "b-full",
                "k-full",
                "opposition-fraction",
                "rank-curve-distance",
            ),
            limitation_codes=(
                "FINITE_REQUESTED_SIMULATOR_POPULATION",
                "RIGHT_CENSORING_RETAINED",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=(
            FalsifierSpec(
                falsifier_id="history-budget-phase-diagram-boundary-counterexample",
                kind=FalsifierKind.RECEIVER_GATE,
                capability_key="history-budget-phase-diagram.unit-adjudicator",
                description="A verified coordinate collision changes target, sink or hold/admit disposition.",
                decisive_rule="One verified adverse collision opposes the corresponding sufficiency claim.",
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id="history-budget-phase-diagram-generator-observer-disagreement",
                kind=FalsifierKind.NEGATIVE_CONTROL,
                capability_key="history-budget-phase-diagram.unit-adjudicator",
                description="Independent generator realization disagrees with the frozen observer forecast.",
                decisive_rule="Wrong source, termination or action clock must be detected and exact category disagreement retained.",
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id="history-budget-phase-diagram-recurrence-conjunction",
                kind=FalsifierKind.WITHIN_CELL_RECURRENCE,
                capability_key=RECURRENCE_KEY,
                description="Cross-scale and cross-disorder recurrence is a noncompensating conjunction.",
                decisive_rule="Failure of any frozen family-scale cell prevents the across-family recurrence statement.",
                status=ObligationStatus.REQUIRED,
            ),
        ),
        closure=ClosureSpec(
            closure_id="history-budget-phase-diagram-history-decision-closure",
            recurrence_cell_ids=_recurrence_cell_ids(),
            exchange_factor_ids=("disorder-family", "finite-scale-view", "seed-block"),
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id="history-budget-phase-diagram-finite-scale-recurrence",
            required_structure_ids=(
                "boundary-decision-collisions",
                "history-rank-staircase",
                "independent-generator-agreement",
            ),
            numerical_view_ids=tuple(
                value
                for value in view_ids
                if value
                in {
                    "history-budget-phase-diagram-n64-view",
                    "history-budget-phase-diagram-n128-view",
                    "history-budget-phase-diagram-n256-view",
                }
            ),
            tolerances=(),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id="history-budget-phase-diagram-computability",
            envelope_id=system.computability_envelopes[0].envelope_id,
            numerical_view_ids=view_ids,
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=("history-budget-phase-diagram-excluded-resource-canary",),
        ),
    )


def history_budget_phase_diagram_experiment(system: SystemSpec) -> ExperimentSpec:
    view_ids = tuple(value.view_id for value in system.numerical_views)
    claim = ClaimSpec(
        claim_id="history-budget-phase-diagram-prospective-recurrence-claim",
        world_id=system.world.world_id,
        relation_id=system.relation.relation_id,
        proposition=(
            "Predeclared absolute-depth and normalized-budget closure states, rank/conditioning "
            "separation and untouched-preparation encounter states recur across the entered "
            "scales and disorder families under an independently implemented generator."
        ),
        estimand=(
            "Requested-denominator T phase states in 72 primary coordinate cells, paired K/B "
            "alignment, and U adverse/unsafe prevalence in the frozen family-scale cells."
        ),
        physical_independent_unit_id=system.independent_unit.unit_id,
        requested_rung=EvidenceRung.LOCAL_LAW,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        promotion_rule="No physical, continuum, controller admission, prospective controller evaluation or controller promotion is permitted.",
        assumption_ids=(
            "finite-linear-rc-equations",
            "independent-generator-observer-source-firewall",
            "seeded-bounded-component-fields",
        ),
        numerical_view_ids=view_ids,
    )
    cutoff = InformationCutoff(
        cutoff_id="history-budget-phase-diagram-pre-outcome-cutoff",
        clock_id=system.clocks[0].clock_id,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal(0),
    )
    sealed_artifacts = tuple(
        sorted(
            artifact_id
            for unit_id in evaluation_unit_ids()
            for artifact_id in (
                f"artifact.evaluation.evaluation-generator.{unit_id}.generator-arrays",
                f"artifact.evaluation.evaluation-generator.{unit_id}.generator-bundle",
            )
        )
    )
    return ExperimentSpec(
        experiment_id="history-budget-phase-diagram-simulator-recurrence-experiment",
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=(claim,),
        assignment=AssignmentSpec(
            assignment_id="history-budget-phase-diagram-simulator-intervention",
            kind=AssignmentKind.SIMULATOR_INTERVENTION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism=(
                "Frozen 5x5 requested/accepted/applied/realized actions on independently "
                "generated finite RC-ladder preparations."
            ),
            support_restriction_ids=("history-budget-phase-diagram-hidden-voltage-envelope",),
        ),
        measurement_quantity_ids=("receiver", "sink"),
        controls=tuple(
            sorted(
                (
                    ControlSpec(
                        control_id="control.history-budget-phase-diagram-duplicate-and-shuffled-history",
                        kind=ControlKind.BASELINE_COMPARATOR,
                        capability_key=HISTORY_KEY,
                        target_quantity_ids=("history", "receiver"),
                        decisive_rule="Duplicate-present and shuffled-lag controls retain their frozen rank behavior.",
                    ),
                    ControlSpec(
                        control_id="control.history-budget-phase-diagram-random-fibre",
                        kind=ControlKind.BASELINE_COMPARATOR,
                        capability_key="history-budget-phase-diagram.boundary-targeter",
                        target_quantity_ids=("receiver", "sink"),
                        decisive_rule="Random-fibre comparators remain separate from gate-targeted nominations.",
                    ),
                    ControlSpec(
                        control_id="control.negative-action",
                        kind=ControlKind.NEGATIVE_ACTION,
                        capability_key="history-budget-phase-diagram.unit-adjudicator",
                        target_quantity_ids=("receiver", "sink"),
                        decisive_rule="Wrong source, termination and action-clock controls must be opposed.",
                    ),
                    ControlSpec(
                        control_id="control.support-matched-comparator",
                        kind=ControlKind.BASELINE_COMPARATOR,
                        capability_key="history-budget-phase-diagram.boundary-targeter",
                        target_quantity_ids=("receiver", "sink"),
                        decisive_rule="All recurrence comparisons retain the frozen family-scale support cell.",
                    ),
                    ControlSpec(
                        control_id="control.history-budget-phase-diagram-wrong-source-termination-clock",
                        kind=ControlKind.WRONG_ACTION,
                        capability_key="history-budget-phase-diagram.unit-adjudicator",
                        target_quantity_ids=("receiver", "sink"),
                        decisive_rule="Wrong source impedance, termination impedance and action clock are detected.",
                    ),
                ),
                key=lambda value: value.control_id,
            )
        ),
        precision_goals=(
            PrecisionGoal(
                goal_id="history-budget-phase-diagram-frozen-90-unit-stop",
                metric_id="requested-unit-opposition-fraction",
                target_width=Decimal("1"),
                native_unit="fraction",
                maximum_independent_units=90,
                stopping_rule="Stop after all 90 requested seed blocks; never retune or extend after reveal.",
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id="history-budget-phase-diagram-sealed-generator-reveal",
            development_unit_ids=development_unit_ids(),
            evaluation_cohort_id="history-budget-phase-diagram-evaluation-seed-roster",
            evaluation_manifest_sha256=sha256(
                "\n".join(evaluation_unit_ids()).encode("ascii")
            ).hexdigest(),
            sealed_outcome_artifact_ids=sealed_artifacts,
            evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            fresh_evidence=True,
        ),
        obligations=_obligations(system),
        design_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
        predecessor_experiment_ids=("simulator-morphism-challenges",),
    )


def history_budget_phase_diagram_study(
    system: SystemSpec,
    experiment: ExperimentSpec,
    phase: HistoryBudgetPhaseDiagramPhase,
) -> CampaignSpec:
    node = CampaignNode(
        node_id=f"campaign-node.history-budget-phase-diagram.{phase.value.lower()}",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id=f"campaign.history-budget-phase-diagram.{phase.value.lower()}",
        objective=(
            f"Execute the separately issued {phase.value.lower()} phase of the history budget phase diagram "
            "history, boundary and recurrence simulator challenge."
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(value.claim_id for value in experiment.claims),
        budget=_PROGRAMME_BUDGET,
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id=f"decision-right.history-budget-phase-diagram.{phase.value.lower()}",
                action=AuthorityAction.SIMULATION_EXECUTION,
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


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramAuthoringBundle:
    config: HistoryBudgetPhaseDiagramConfig
    external_records: tuple[HistoryBudgetPhaseDiagramExternalRecord, ...]
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    protocol: ProtocolTemplate
    registry: CapabilityRegistry
    catalog: CandidateCapabilityCatalog
    source_configs: tuple[SourceMaterializationConfig, ...]
    qualifications: tuple[MaterializationQualificationReceipt, ...]
    design_input: DesignInputRecord
    draft: StudyDraft
    context: CandidateCompilationContext


def _all_external_records(
    config: HistoryBudgetPhaseDiagramConfig,
    external_records: tuple[HistoryBudgetPhaseDiagramExternalRecord, ...],
) -> tuple[HistoryBudgetPhaseDiagramExternalRecord, ...]:
    phase_record = HistoryBudgetPhaseDiagramExternalRecord(
        input_id=f"input.history-budget-phase-diagram.{config.phase.value.lower()}.config",
        record=config,
        role=ScientificInputRole.MODEL,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility=VisibilityCeiling.PROSPECTIVE,
    )
    values = tuple(sorted((phase_record, *external_records), key=lambda value: value.input_id))
    if tuple(value.input_id for value in values) != tuple(
        sorted({value.input_id for value in values})
    ):
        raise ValueError("history budget phase diagram authoring external input IDs repeat")
    return values


def build_history_budget_phase_diagram_authoring_bundle(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    implementation_sha256: str,
    external_records: tuple[HistoryBudgetPhaseDiagramExternalRecord, ...] = (),
) -> HistoryBudgetPhaseDiagramAuthoringBundle:
    system = history_budget_phase_diagram_system()
    experiment = history_budget_phase_diagram_experiment(system)
    campaign = history_budget_phase_diagram_study(system, experiment, config.phase)
    registry = history_budget_phase_diagram_phase_registry(
        implementation_sha256=implementation_sha256,
        config=config,
    )
    protocol = protocol_template(registry=registry, config=config)
    template = study_template(
        registry=registry,
        config=config,
        protocol=protocol,
        experiment=experiment,
        external_records=external_records,
    )
    catalog = candidate_catalog(templates=(template,), registry=registry)
    observation_operator = ObjectIdentity(
        object_id="reference.history-budget-phase-diagram-canonical-materializer",
        object_schema='empirical-lawhood/history-budget-phase-diagram/candidate-input-materializer',
        object_version=VERSION,
        object_fingerprint=implementation_sha256,
    )
    records = _all_external_records(config, external_records)
    external_specs = {value.input_id: value for value in template.graph.external_inputs}
    if set(external_specs) != {value.input_id for value in records}:
        raise ValueError("history budget phase diagram source records differ from the scientific graph")
    source_configs = tuple(
        SourceMaterializationConfig(
            config_id=f"source-config.{value.input_id}",
            source_id=value.input_id,
            role=_SOURCE_ROLE_BY_SCIENTIFIC_ROLE[value.role],
            content_sha256=value.record.fingerprint(),
            expected_size_bytes=len(value.record.canonical_bytes()),
            maximum_bytes=external_specs[value.input_id].maximum_size_bytes,
            payload_schema=value.record.SCHEMA,
            media_type=external_specs[value.input_id].media_type,
            read_mode=SourceReadMode.ORDINARY_BOUNDED,
            outcome_access=value.outcome_access,
            visibility_ceiling=value.visibility,
        )
        for value in records
    )
    source_config_by_source = {value.source_id: value for value in source_configs}
    view_ids = tuple(value.view_id for value in system.numerical_views)
    qualifications = tuple(
        MaterializationQualificationReceipt(
            receipt_id=f"qualification.{value.input_id}",
            source_id=value.input_id,
            materialization=ObjectIdentity.from_record(value.input_id, value.record),
            content_sha256=value.record.fingerprint(),
            evidence_world_id=system.world.world_id,
            observation_operator=observation_operator,
            numerical_view_ids=view_ids,
            native_unit_ids=tuple(sorted({quantity.native_unit for quantity in system.quantities})),
            frame_ids=tuple(sorted({quantity.coordinate_frame for quantity in system.quantities})),
            clock_ids=tuple(value.clock_id for value in system.clocks),
            receiver_semantics_id=system.relation.relation_id,
            validity_contract_id=experiment.obligations.validity.validity_id,
            uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
            access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            outcome_access=value.outcome_access,
            visibility_ceiling=value.visibility,
        )
        for value in records
    )
    qualification_by_source = {value.source_id: value for value in qualifications}
    materializations = tuple(
        SourceMaterializationRef(
            source_id=value.input_id,
            role=_SOURCE_ROLE_BY_SCIENTIFIC_ROLE[value.role],
            evidence_world_id=system.world.world_id,
            materialization=ObjectIdentity.from_record(value.input_id, value.record),
            content_sha256=value.record.fingerprint(),
            source_config_sha256=source_config_by_source[value.input_id].fingerprint(),
            observation_operator=observation_operator,
            numerical_view_ids=view_ids,
            qualification_receipt=ObjectIdentity.from_record(
                qualification_by_source[value.input_id].receipt_id,
                qualification_by_source[value.input_id],
            ),
            access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        )
        for value in records
    )
    cutoff = experiment.information_cutoffs[0]
    design_input = DesignInputRecord(
        input_id=f"design-input.history-budget-phase-diagram.{config.phase.value.lower()}",
        object_identity=ObjectIdentity.from_record(config.config_id, config),
        materialization_sha256=config.fingerprint(),
        information_cutoff=cutoff,
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
        physical_unit_ids=(),
    )
    draft = StudyDraft(
        draft_id=f"draft.history-budget-phase-diagram.{config.phase.value.lower()}",
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            "Do the frozen history budget phase diagram history, boundary and independent-generator "
            f"contracts survive the {config.phase.value.lower()} phase without leakage or rescue?"
        ),
        alternative_ids=(
            "alternative.history-budget-phase-diagram-opposed-or-mixed",
            "alternative.history-budget-phase-diagram-supported",
            "alternative.history-budget-phase-diagram-unevaluable-or-stopped",
        ),
        design_origin=DesignOrigin(
            origin_id=f"origin.history-budget-phase-diagram.{config.phase.value.lower()}",
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=(design_input.input_id,),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=(design_input,),
        development_unit_ids=development_unit_ids(),
        evaluation_unit_ids=evaluation_unit_ids(),
        development_seed_ids=tuple(
            f"seed.history-budget-phase-diagram-development.{unit_id}" for unit_id in development_unit_ids()
        ),
        evaluation_seed_ids=("seed-roster.history-budget-phase-diagram-evaluation-opaque",),
        unresolved_decisions=(),
        system=system,
        experiment=experiment,
        campaign=campaign,
        dag_template_key=template.template_key,
        capability_selections=tuple(
            CapabilitySelection(
                capability_key=value.capability_key,
                capability_version=value.capability_version,
                implementation_sha256=value.implementation_sha256,
            )
            for value in registry.capabilities
            if value.capability_key in {step.capability_key for step in protocol.steps}
        ),
        source_materializations=materializations,
        resource_ceiling=_PROGRAMME_BUDGET,
    )
    context = CandidateCompilationContext(
        context_id=f"context.history-budget-phase-diagram.{config.phase.value.lower()}",
        registry=registry,
        templates=catalog.templates,
        qualifications=qualifications,
        known_design_inputs=(design_input,),
        implementation_sha256=implementation_sha256,
    )
    return HistoryBudgetPhaseDiagramAuthoringBundle(
        config=config,
        external_records=external_records,
        system=system,
        experiment=experiment,
        campaign=campaign,
        protocol=protocol,
        registry=registry,
        catalog=catalog,
        source_configs=source_configs,
        qualifications=qualifications,
        design_input=design_input,
        draft=draft,
        context=context,
    )


def history_budget_phase_diagram_formal_source_inventory(
    *,
    base: HistoryBudgetPhaseDiagramAuthoringBundle,
    register: FormalGapRegister,
) -> FormalGapSourceCapabilityInventory:
    selected = {
        value.gap_id: value for value in register.gaps if value.gap_id in HISTORY_BUDGET_PHASE_DIAGRAM_FORMAL_GAP_IDS
    }
    if set(selected) != set(HISTORY_BUDGET_PHASE_DIAGRAM_FORMAL_GAP_IDS):
        raise ValueError("current formal register lacks an history budget phase diagram priority gap")
    return FormalGapSourceCapabilityInventory(
        inventory_id=f"formal-source-inventory.history-budget-phase-diagram.{base.config.phase.value.lower()}",
        denominator_id=base.system.system_id,
        evidence_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        source_materializations=tuple(
            sorted(
                (value.materialization for value in base.draft.source_materializations),
                key=lambda value: value.object_id,
            )
        ),
        present_operand_ids=tuple(
            sorted({operand for gap in selected.values() for operand in gap.required_operand_ids})
        ),
        satisfied_prerequisite_ids=(
            tuple(
                sorted(
                    {
                        prerequisite
                        for gap in selected.values()
                        for prerequisite in gap.support_prerequisite_ids
                    }
                )
            )
            if base.config.phase is HistoryBudgetPhaseDiagramPhase.EVALUATION
            else ()
        ),
        independent_unit_ids=evaluation_unit_ids(),
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
        # The remaining register questions are not declared denominator-
        # inapplicable.  This act simply does not provide their operands and
        # protocol prerequisites, which is a materially narrower statement.
        denominator_inapplicable_gap_ids=(),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def history_budget_phase_diagram_formal_coverage(
    *,
    base: HistoryBudgetPhaseDiagramAuthoringBundle,
    register: FormalGapRegister,
    inventory: FormalGapSourceCapabilityInventory,
) -> FormalGapCoverage:
    selected = set(HISTORY_BUDGET_PHASE_DIAGRAM_FORMAL_GAP_IDS)
    assignments = []
    evaluation_phase = base.config.phase is HistoryBudgetPhaseDiagramPhase.EVALUATION
    for gap in register.gaps:
        tested = evaluation_phase and gap.gap_id in selected
        deferred = not evaluation_phase and gap.gap_id in selected
        outside_protocol = gap.gap_id not in selected
        assignments.append(
            FormalGapCoverageAssignment(
                gap_id=gap.gap_id,
                disposition=(
                    FormalGapCoverageDisposition.TEST_IN_THIS_ACT
                    if tested
                    else FormalGapCoverageDisposition.DEFER_WITH_TYPED_PREREQUISITE
                    if deferred or outside_protocol
                    else FormalGapCoverageDisposition.NOT_APPLICABLE_TO_DENOMINATOR
                ),
                readiness_reason=(
                    ReadinessStatus.REQUIRES_PROTOCOL_FREEZE
                    if deferred
                    else ReadinessStatus.PREREQUISITE_NOT_MET
                    if outside_protocol
                    else None
                ),
                reason_codes=(
                    ()
                    if tested
                    else ("HISTORY_BUDGET_PHASE_DIAGRAM_PREREQUISITE_PHASE_ONLY",)
                    if deferred
                    else ("HISTORY_BUDGET_PHASE_DIAGRAM_OPERAND_OR_PROTOCOL_NOT_PROVIDED",)
                    if outside_protocol
                    else ("OUTSIDE_HISTORY_BUDGET_PHASE_DIAGRAM_SIMULATOR_DENOMINATOR",)
                ),
                selected_estimator_family_id=(gap.estimator_family_ids[0] if tested else None),
                selected_control_ids=gap.control_ids if tested else (),
                selected_multiplicity_family_id=(gap.multiplicity_family_id if tested else None),
                obligation_ids=(("history-budget-phase-diagram-recurrence-synthesize-contract",) if tested else ()),
                output_ids=(("recurrence-result",) if tested else ()),
                adjudication_owner_ids=(("recurrence-synthesize",) if tested else ()),
            )
        )
    return FormalGapCoverage(
        coverage_id=f"formal-gap-coverage.history-budget-phase-diagram.{base.config.phase.value.lower()}",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=base.system.system_id,
        candidate_act_id=base.draft.draft_id,
        applicability=derive_formal_gap_applicability(register, inventory),
        assignments=tuple(assignments),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def history_budget_phase_diagram_formal_method_catalog(
    *,
    base: HistoryBudgetPhaseDiagramAuthoringBundle,
    register: FormalGapRegister,
) -> FormalMethodCatalog:
    # Formal catalog identity spans the whole tranche, while the executable
    # candidate registry remains phase-local and least privilege.  Deferred
    # phases never resolve these bindings; the evaluation phase contains both
    # capabilities and is checked against them by the standard compiler.
    method_registry = history_budget_phase_diagram_registry(implementation_sha256=base.context.implementation_sha256)
    selected = tuple(value for value in register.gaps if value.gap_id in HISTORY_BUDGET_PHASE_DIAGRAM_FORMAL_GAP_IDS)
    estimator_families = sorted({family for gap in selected for family in gap.estimator_family_ids})
    multiplicity_families = sorted({gap.multiplicity_family_id for gap in selected})
    bindings = []
    for role, families, capability_key, input_schema, output_schema, access in (
        (
            FormalMethodRole.ESTIMATOR,
            estimator_families,
            UNIT_ADJUDICATOR_KEY,
            HistoryBudgetPhaseDiagramGeneratorBundle.SCHEMA,
            HistoryBudgetPhaseDiagramAdjudicationBundle.SCHEMA,
            OutcomeAccess.EVALUATOR_REVEAL,
        ),
        (
            FormalMethodRole.MULTIPLICITY,
            multiplicity_families,
            RECURRENCE_KEY,
            HistoryBudgetPhaseDiagramAdjudicationBundle.SCHEMA,
            HistoryBudgetPhaseDiagramRecurrenceResult.SCHEMA,
            OutcomeAccess.EVALUATOR_REVEAL,
        ),
    ):
        manifest = method_registry.resolve(capability_key, VERSION)
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
                    capability_key=capability_key,
                    capability_version=VERSION,
                    implementation_sha256=manifest.implementation_sha256,
                    supported_gap_ids=supported,
                    input_schema_id=input_schema,
                    output_schema_id=output_schema,
                    maximum_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
                    maximum_outcome_access=access,
                )
            )
    return FormalMethodCatalog(
        catalog_id=f"formal-method-catalog.history-budget-phase-diagram.{base.config.phase.value.lower()}",
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
            object_ids=(f"binding.history-budget-phase-diagram.{requirement.value.lower()}",),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                ("HISTORY_BUDGET_PHASE_DIAGRAM_ISSUE_EXECUTION_AND_REVEAL_AUTHORITIES_SEPARATE",)
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ()
            ),
        )
        for requirement in sorted(ExperimentEntryRequirement, key=lambda value: value.value)
    )
    checklist = ExperimentEntryChecklist(
        checklist_id=f"entry-checklist.{draft.draft_id}",
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        execution_route_id="route.history-budget-phase-diagram-receipt-first",
        durability_disposition_id="durability.history-budget-phase-diagram-external-receipt-first",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id=f"experiment-entry-package.{draft.draft_id}",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


@dataclass(frozen=True, slots=True)
class HistoryBudgetPhaseDiagramStandardAuthoringBundle:
    base: HistoryBudgetPhaseDiagramAuthoringBundle
    formal_coverage: FormalGapCoverage
    formal_methods: FormalMethodCatalog
    formal_source_inventory: FormalGapSourceCapabilityInventory
    entry_package: ExperimentEntryPackage
    authoring_package: StudyDefinition
    context: StandardCandidateCompilationContext


def build_history_budget_phase_diagram_standard_authoring_bundle(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    implementation_sha256: str,
    register: FormalGapRegister,
    external_records: tuple[HistoryBudgetPhaseDiagramExternalRecord, ...] = (),
) -> HistoryBudgetPhaseDiagramStandardAuthoringBundle:
    base = build_history_budget_phase_diagram_authoring_bundle(
        config=config,
        implementation_sha256=implementation_sha256,
        external_records=external_records,
    )
    inventory = history_budget_phase_diagram_formal_source_inventory(base=base, register=register)
    coverage = history_budget_phase_diagram_formal_coverage(
        base=base,
        register=register,
        inventory=inventory,
    )
    methods = history_budget_phase_diagram_formal_method_catalog(base=base, register=register)
    entry = _entry_package(draft=base.draft, register=register, coverage=coverage)
    package = StudyDefinition(
        package_id=f"programme-authoring-package.{base.draft.draft_id}",
        draft=base.draft,
        entry_package=entry,
    )
    context = StandardCandidateCompilationContext(
        context_id=f"standard-context.history-budget-phase-diagram.{config.phase.value.lower()}",
        base=base.context,
        formal_methods=methods,
        source_inventories=(inventory,),
    )
    return HistoryBudgetPhaseDiagramStandardAuthoringBundle(
        base=base,
        formal_coverage=coverage,
        formal_methods=methods,
        formal_source_inventory=inventory,
        entry_package=entry,
        authoring_package=package,
        context=context,
    )


__all__ = [
    "HISTORY_BUDGET_PHASE_DIAGRAM_FORMAL_GAP_IDS",
    'HistoryBudgetPhaseDiagramAuthoringBundle',
    'HistoryBudgetPhaseDiagramStandardAuthoringBundle',
    'build_history_budget_phase_diagram_authoring_bundle',
    'build_history_budget_phase_diagram_standard_authoring_bundle',
    'history_budget_phase_diagram_study',
    'history_budget_phase_diagram_experiment',
    'history_budget_phase_diagram_formal_coverage',
    'history_budget_phase_diagram_formal_method_catalog',
    'history_budget_phase_diagram_formal_source_inventory',
    'history_budget_phase_diagram_system',
]
