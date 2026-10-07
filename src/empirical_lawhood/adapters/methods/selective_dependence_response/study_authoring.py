"""Standard nonpromotable authoring root for the selective-response programme forecast gate.

The planning world in this module represents only the finite, deterministic
policy predicate evaluated by the programme gate.  The two target completion
envelopes remain development-visible design parents and graph
``PARENT_RECEIPT`` inputs; they are never relabelled as analytic target
evidence or formal source materializations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.methods.formal_analysis import standard_formal_method_catalog
from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
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
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.references import QuantityBound
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.kernel.status import LifecycleStatus, ReadinessStatus
from empirical_lawhood.kernel.systems import (
    BalanceRole,
    IndependentUnitSpec,
    PortDirection,
    PortSpec,
    RelationalIdentity,
    SystemBoundaryKind,
    SystemSpec,
)
from empirical_lawhood.kernel.time import (
    AvailabilitySpec,
    CausalPhase,
    ClockLabelSemantics,
    ClockSpec,
    HoldSemantics,
    HorizonSpec,
    InformationCutoff,
    SamplingSemantics,
)
from empirical_lawhood.kernel.worlds import (
    ComputabilityEnvelope,
    EvidenceUnitScope,
    NumericalCoordinateKind,
    NumericalCoordinateSpec,
    NumericalViewSpec,
    RandomnessSemantics,
    WorldKind,
    WorldSpec,
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
from empirical_lawhood.runtime.candidate_compiler import AuthoringMaterializationIdentity, CandidateCompilationContext, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, StandardCandidateCompilationContext, StudyCompilationReport, compile_study_candidate, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.source_resolution import SourceMaterializationConfig, SourceReadMode

from .composition import SelectiveDependenceResponseStageComposition, compose_study_forecast
from .contracts import SelectiveDependenceResponseMethodQuestionFreeze
from .development_completion import SelectiveDependenceResponseDevelopmentCompletionEnvelope


_PROGRAMME_BUDGET = ResourceBudget(
    cpu_cores=1,
    memory_bytes=256 * 1024**2,
    gpu_devices=0,
    wall_time_seconds=60,
    source_scan_bytes=64 * 1024**2,
    output_bytes=2 * 1024**2,
)
_DEVELOPMENT_DECISION_UNIT_IDS = (
    "unit-development.selective-dependence-response.programme.cantera-completion",
    "unit-development.selective-dependence-response.programme.fipy-completion",
)
_EVALUATION_DECISION_UNIT_IDS = ("unit-evaluation.selective-dependence-response.programme.exact-policy-predicate",)


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseStudyAuthoringAct:
    """Pure authority-pending programme-gate authoring result."""

    completions: tuple[
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
    ]
    question: SelectiveDependenceResponseMethodQuestionFreeze
    stage_composition: SelectiveDependenceResponseStageComposition
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    source_config: SourceMaterializationConfig
    qualification: MaterializationQualificationReceipt
    design_inputs: tuple[DesignInputRecord, ...]
    inventory: FormalGapSourceCapabilityInventory
    template: StudyTemplate
    catalog: CandidateCapabilityCatalog
    draft: StudyDraft
    package: StudyDefinition
    context: StandardCandidateCompilationContext
    compilation: StudyCompilationReport

    @property
    def candidate_input_payloads(self) -> tuple[bytes, ...]:
        """Return exact compact bytes required by file-backed compilation."""

        records: tuple[CanonicalRecord, ...] = (
            *self.stage_composition.configs,
            self.source_config,
            self.question,
            *self.completions,
            self.qualification,
        )
        return tuple(value.canonical_bytes() for value in records)

    @property
    def candidate_composition(self) -> SelectiveDependenceResponseStageComposition:
        return replace(
            self.stage_composition,
            catalog=self.catalog,
            known_design_inputs=self.design_inputs,
            formal_source_inventories=(self.inventory,),
        )


def _quantity(
    *,
    quantity_id: str,
    kind: QuantityKind,
    phase: CausalPhase,
    access: OutcomeAccess,
    direction: ResponseDirection = ResponseDirection.NOT_APPLICABLE,
) -> QuantitySpec:
    return QuantitySpec(
        quantity_id=quantity_id,
        label=quantity_id.replace(".", " "),
        kind=kind,
        dimension="finite-programme-decision",
        native_unit="decision-code",
        coordinate_frame="selective-dependence-response-programme-policy-predicate",
        clock_id="clock.selective-dependence-response.programme.predicate",
        availability=AvailabilitySpec(
            clock_id="clock.selective-dependence-response.programme.predicate",
            phase=phase,
            outcome_access=access,
        ),
        response_direction=direction,
    )


def build_study_system() -> SystemSpec:
    """Describe only the exact finite policy predicate, never target physics."""

    clock = ClockSpec(
        clock_id="clock.selective-dependence-response.programme.predicate",
        label="selective dependence response exact programme-policy predicate clock",
        time_unit="decision-step",
        coordinate_frame="selective-dependence-response-programme-policy-predicate",
        sampling=SamplingSemantics.EVENT_DRIVEN,
        hold=HoldSemantics.SOURCE_DEFINED,
        label_semantics=ClockLabelSemantics.EVENT,
        nominal_period=None,
        alignment_tolerance=Decimal(0),
    )
    action_rows = (
        ("action.accepted-evaluation-issue", CausalPhase.ACTION_REQUESTED),
        ("action.applied-policy-predicate", CausalPhase.ACTION_APPLIED),
        ("action.realized-programme-qualification", CausalPhase.ACTION_APPLIED),
        ("action.requested-evaluation-issue", CausalPhase.ACTION_REQUESTED),
    )
    quantities = tuple(
        sorted(
            (
                _quantity(
                    quantity_id="denominator.exact-two-target-roster",
                    kind=QuantityKind.DENOMINATOR,
                    phase=CausalPhase.PRE_ACTION,
                    access=OutcomeAccess.OUTCOME_BLIND,
                ),
                _quantity(
                    quantity_id="history.frozen-development-completions",
                    kind=QuantityKind.HISTORY,
                    phase=CausalPhase.PRE_ACTION,
                    access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                ),
                *(
                    _quantity(
                        quantity_id=quantity_id,
                        kind=QuantityKind.ACTION,
                        phase=phase,
                        access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                    )
                    for quantity_id, phase in action_rows
                ),
                _quantity(
                    quantity_id="receiver.policy-disposition-roster-complete",
                    kind=QuantityKind.RECEIVER,
                    phase=CausalPhase.RECEIVER,
                    access=OutcomeAccess.EVALUATION_SEALED,
                    direction=ResponseDirection.HIGHER_IS_BETTER,
                ),
                _quantity(
                    quantity_id="sink.evaluation-outcome-contact-count",
                    kind=QuantityKind.SINK,
                    phase=CausalPhase.RECEIVER,
                    access=OutcomeAccess.EVALUATION_SEALED,
                    direction=ResponseDirection.LOWER_IS_BETTER,
                ),
            ),
            key=lambda value: value.quantity_id,
        )
    )
    unit = IndependentUnitSpec(
        unit_id="unit.selective-dependence-response.programme.exact-two-completion-roster",
        label=(
            "one complete analytic decision preparation containing exactly one "
            "Cantera and one FiPy terminal development-completion envelope"
        ),
        grouping_key="group.selective-dependence-response.programme.exact-two-completion-roster",
    )
    relation = RelationalIdentity(
        relation_id="relation.selective-dependence-response.programme.finite-policy-predicate",
        denominator_quantity_ids=("denominator.exact-two-target-roster",),
        history_quantity_ids=("history.frozen-development-completions",),
        memoryless=False,
        action_quantity_ids=tuple(value[0] for value in action_rows),
        receiver_quantity_ids=("receiver.policy-disposition-roster-complete",),
        horizon=HorizonSpec(
            horizon_id="horizon.selective-dependence-response.programme.one-predicate",
            clock_id=clock.clock_id,
            duration=Decimal(1),
            time_unit="decision-step",
        ),
    )
    world = WorldSpec(
        world_id="world.selective-dependence-response.programme.analytic-decision",
        label="selective dependence response finite programme-policy decision world",
        kind=WorldKind.ANALYTIC_REFERENCE,
        represented_physics=("deterministic-finite-policy-roster-predicate",),
        unrepresented_physics=(
            "cantera-target-response",
            "fipy-target-response",
            "physical-substrate-response",
        ),
        privileged_truth_quantity_ids=(),
        maximum_evidence=EvidenceCeiling.NON_PROMOTABLE,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
            }
        ),
    )
    structures = (
        "exact-cantera-fipy-parent-roster",
        "exact-policy-disposition-union",
        "no-evaluation-outcome-contact",
        "no-native-cross-target-pooling",
    )
    envelope = ComputabilityEnvelope(
        envelope_id="compute.selective-dependence-response.programme.finite-predicate",
        represented_effect_ids=("deterministic-programme-policy-decision",),
        unresolved_effect_ids=(),
        required_structure_ids=structures,
        computable_structure_ids=structures,
        max_cpu_cores=_PROGRAMME_BUDGET.cpu_cores,
        max_memory_bytes=_PROGRAMME_BUDGET.memory_bytes,
        max_gpu_devices=0,
        max_wall_time_seconds=_PROGRAMME_BUDGET.wall_time_seconds,
        max_output_bytes=_PROGRAMME_BUDGET.output_bytes,
        worst_case_latency_seconds=Decimal(60),
        deadline_seconds=Decimal(_PROGRAMME_BUDGET.wall_time_seconds),
    )
    view = NumericalViewSpec(
        view_id="view.selective-dependence-response.programme.exact-predicate",
        world_id=world.world_id,
        physical_preparation_id=unit.unit_id,
        equations_id="equations.selective-dependence-response.programme.boolean-policy-predicate",
        closure_ids=("closure.selective-dependence-response.programme.exact-record-decoding",),
        boundary_condition_ids=("boundary.selective-dependence-response.programme.exact-two-targets",),
        coordinates=(
            NumericalCoordinateSpec(
                coordinate_id="coordinate.selective-dependence-response.programme.target-count",
                kind=NumericalCoordinateKind.SOLVER_REFINEMENT,
                value=Decimal(2),
                unit="target-completion",
                refinement_level=0,
            ),
        ),
        solver_id="solver.selective-dependence-response.programme.exact-predicate",
        solver_version="1.0.0",
        precision="exact-canonical-record-and-enum-comparison",
        device_class="cpu",
        runtime_id="cpython-3.11-selective-dependence-response-programme-forecast",
        randomness=RandomnessSemantics.DETERMINISTIC,
        observation_operator_id="observer.selective-dependence-response.programme.qualification",
        computability_envelope_id=envelope.envelope_id,
    )
    policy = AuthorityPolicy(
        policy_id="authority-policy.selective-dependence-response.programme-forecast",
        delegator_id="human.project-owner",
        delegate_id="gate.selective-dependence-response.programme-forecast",
        scope_ids=("campaign.selective-dependence-response.programme-forecast",),
        allowed_world_kinds=frozenset({WorldKind.ANALYTIC_REFERENCE}),
        allowed_actions=frozenset(
            {
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.REFERENCE_WORLD_EXECUTION,
            }
        ),
        allowed_source_classes=frozenset({SourceAccessClass.NONE}),
        required_gate_ids=(
            "clean-implementation",
            "exact-two-development-completions",
            "no-evaluation-outcome-contact",
            "programme-forecast-execution-authority",
        ),
        nondelegable_actions=frozenset(
            {
                AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
                AuthorityAction.HIL_ACTUATION,
                AuthorityAction.HUMAN_OR_ANIMAL_INTERVENTION,
                AuthorityAction.LIVE_ACTUATION,
                AuthorityAction.SAFETY_SIGNIFICANT_OPERATION,
            }
        ),
        budget_ceiling=_PROGRAMME_BUDGET,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    ports = tuple(
        sorted(
            (
                *(
                    PortSpec(
                        port_id=f"port.{quantity_id}.input",
                        quantity_id=quantity_id,
                        clock_id=clock.clock_id,
                        direction=PortDirection.INPUT,
                        balance_role=BalanceRole.COMMAND,
                        authority_action=AuthorityAction.REFERENCE_WORLD_EXECUTION,
                    )
                    for quantity_id, _ in action_rows
                ),
                PortSpec(
                    port_id="port.receiver.policy-disposition-roster-complete.output",
                    quantity_id="receiver.policy-disposition-roster-complete",
                    clock_id=clock.clock_id,
                    direction=PortDirection.OUTPUT,
                    balance_role=BalanceRole.OBSERVATION,
                ),
                PortSpec(
                    port_id="port.sink.evaluation-outcome-contact-count.output",
                    quantity_id="sink.evaluation-outcome-contact-count",
                    clock_id=clock.clock_id,
                    direction=PortDirection.OUTPUT,
                    balance_role=BalanceRole.INFORMATION,
                ),
            ),
            key=lambda value: value.port_id,
        )
    )
    return SystemSpec(
        system_id="system.selective-dependence-response.programme-forecast",
        label="selective dependence response exact-two-target programme forecast predicate",
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=quantities,
        independent_unit=unit,
        authority_policy=policy,
        ports=ports,
        computability_envelopes=(envelope,),
        numerical_views=(view,),
    )


def _study_obligations(
    system: SystemSpec,
    cutoff: InformationCutoff,
) -> ScientificObligations:
    action_bounds = tuple(
        QuantityBound(
            bound_id=f"bound.selective-dependence-response.programme.{quantity_id}",
            quantity_id=quantity_id,
            native_unit="decision-code",
            lower=Decimal(0),
            upper=Decimal(1),
        )
        for quantity_id in system.relation.action_quantity_ids
    )
    capability = "method.selective-dependence-response.programme-forecast.qualify-evaluation-issue"
    view_id = system.numerical_views[0].view_id
    return ScientificObligations(
        obligations_id="obligations.selective-dependence-response.programme-forecast",
        support=SupportSpec(
            support_id="support.selective-dependence-response.programme.exact-two-completions",
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=1,
            nested_numerical_view_count=1,
            information_cutoff_id=cutoff.cutoff_id,
            chart_ids=("chart.selective-dependence-response.programme.binary-issue-decision",),
            denominator_cell_ids=("cell.selective-dependence-response.programme.cantera-plus-fipy",),
            action_bounds=action_bounds,
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id="validity.selective-dependence-response.programme.exact-parent-predicate",
            validity_domain_ids=("domain.selective-dependence-response.programme.exact-two-targets",),
            assumption_ids=(
                "completion-custody-is-terminal",
                "method-question-is-exact",
                "predicate-does-not-reinterpret-target-evidence",
            ),
            exclusion_reason_codes=(
                "NO_ANALYTIC_RECLASSIFICATION_OF_SIMULATOR_EVIDENCE",
                "NO_NATIVE_CROSS_TARGET_POOLING",
                "NO_MEASUREMENT_TO_CONTROLLER_USE_PROMOTION",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id="uncertainty.selective-dependence-response.programme.exact-predicate",
            method_key="exact-finite-policy-roster-conjunction",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal("0.999999"),
            interval_quantity_ids=("sink.evaluation-outcome-contact-count",),
            limitation_codes=(
                "ANALYTIC_DECISION_ONLY",
                "DEVELOPMENT_VISIBLE_PARENTS",
                "NO_TARGET_LOCAL_CLAIM_PROMOTION",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=(
            FalsifierSpec(
                falsifier_id="falsifier.selective-dependence-response.programme.evaluation-contact",
                kind=FalsifierKind.RECEIVER_GATE,
                capability_key=capability,
                description="Any evaluation outcome contact invalidates the gate.",
                decisive_rule=(
                    "A nonzero evaluation-outcome count stops qualification; no field "
                    "or authority may rescue the programme roster."
                ),
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id="falsifier.selective-dependence-response.programme.parent-roster",
                kind=FalsifierKind.NEGATIVE_CONTROL,
                capability_key=capability,
                description="The exact parent roster is one Cantera and one FiPy completion.",
                decisive_rule=(
                    "A missing, duplicate, surplus, ineligible or implementation-mismatched "
                    "completion is a terminal programme-parent failure."
                ),
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id="falsifier.selective-dependence-response.programme.policy-diversity",
                kind=FalsifierKind.BASELINE_COMPARATOR,
                capability_key=capability,
                description="All three frozen policy dispositions must be represented.",
                decisive_rule=(
                    "Missing ACTION_AVAILABLE, HOLD_ONLY or NONATTEMPT yields the "
                    "predeclared negative qualification without method drift."
                ),
                status=ObligationStatus.REQUIRED,
            ),
        ),
        closure=ClosureSpec(
            closure_id="closure.selective-dependence-response.programme.exact-policy-predicate",
            recurrence_cell_ids=("cell.selective-dependence-response.programme.cantera-plus-fipy",),
            exchange_factor_ids=system.relation.denominator_quantity_ids,
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id="convergence.selective-dependence-response.programme.exact-records",
            required_structure_ids=(
                "exact-cantera-fipy-parent-roster",
                "exact-policy-disposition-union",
                "no-evaluation-outcome-contact",
                "no-native-cross-target-pooling",
            ),
            numerical_view_ids=(view_id,),
            tolerances=(),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id="computability.selective-dependence-response.programme.bounded",
            envelope_id=system.computability_envelopes[0].envelope_id,
            numerical_view_ids=(view_id,),
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=("selective-dependence-response.method-question-freeze",),
        ),
    )


def build_study_experiment(system: SystemSpec) -> ExperimentSpec:
    """Build the nonpromotable analytic decision specification."""

    cutoff = InformationCutoff(
        cutoff_id="cutoff.selective-dependence-response.programme.before-policy-predicate",
        clock_id=system.clocks[0].clock_id,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal(0),
    )
    return ExperimentSpec(
        experiment_id="experiment.selective-dependence-response.programme-forecast",
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=(
            ClaimSpec(
                claim_id="claim.selective-dependence-response.programme-policy-gate.nonpromotable",
                world_id=system.world.world_id,
                relation_id=system.relation.relation_id,
                proposition=(
                    "The exact two terminal development completions satisfy or fail the "
                    "predeclared outcome-free programme-policy diversity predicate."
                ),
                estimand=(
                    "Exact Cantera/FiPy roster, eligibility and implementation identity; "
                    "union of ACTION_AVAILABLE, HOLD_ONLY and NONATTEMPT; and literal "
                    "evaluation-outcome count zero."
                ),
                physical_independent_unit_id=system.independent_unit.unit_id,
                requested_rung=None,
                evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                promotion_rule=(
                    'No measurement-through-controller-use or target-local promotion. This decision may only gate separately authored evaluation packages.'
                ),
                assumption_ids=(
                    "completion-envelopes-are-terminal-custody-proofs",
                    "predicate-is-deterministic-and-target-neutral",
                ),
                numerical_view_ids=(system.numerical_views[0].view_id,),
            ),
        ),
        assignment=AssignmentSpec(
            assignment_id="assignment.selective-dependence-response.programme.shadow-issue-decision",
            kind=AssignmentKind.SHADOW_DECISION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism=(
                "A deterministic nonactuating predicate maps the exact two frozen "
                "completion records to a qualification or typed scientific stop."
            ),
            support_restriction_ids=("support.selective-dependence-response.programme.exact-two-completions",),
        ),
        measurement_quantity_ids=(
            "receiver.policy-disposition-roster-complete",
            "sink.evaluation-outcome-contact-count",
        ),
        controls=(
            ControlSpec(
                control_id="control.selective-dependence-response.programme.incomplete-policy-roster",
                kind=ControlKind.BASELINE_COMPARATOR,
                capability_key=("method.selective-dependence-response.programme-forecast.qualify-evaluation-issue"),
                target_quantity_ids=("receiver.policy-disposition-roster-complete",),
                decisive_rule=(
                    "Removing any one required disposition must produce a negative "
                    "qualification, never an operational failure or rescue."
                ),
            ),
        ),
        precision_goals=(
            PrecisionGoal(
                goal_id="precision.selective-dependence-response.programme.exact-one-predicate",
                metric_id="exact-programme-policy-mismatch-count",
                target_width=Decimal("0.000001"),
                native_unit="decision",
                maximum_independent_units=1,
                stopping_rule=(
                    "Evaluate the exact two-parent predicate once or emit its typed "
                    "stop; never add a target, disposition or rescue rule."
                ),
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id="reveal.selective-dependence-response.programme.no-target-evaluation",
            development_unit_ids=_DEVELOPMENT_DECISION_UNIT_IDS,
            evaluation_cohort_id="cohort.selective-dependence-response.programme.exact-policy-predicate",
            evaluation_manifest_sha256=sha256(
                b"selective-dependence-response:programme:exact-cantera-fipy-policy-predicate"
            ).hexdigest(),
            sealed_outcome_artifact_ids=("artifact.selective-dependence-response.programme.policy-predicate-result",),
            evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            fresh_evidence=True,
        ),
        obligations=_study_obligations(system, cutoff),
        design_visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
    )


def build_study_campaign(
    system: SystemSpec,
    experiment: ExperimentSpec,
) -> CampaignSpec:
    node = CampaignNode(
        node_id="campaign-node.selective-dependence-response.programme-forecast.experiment",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id="campaign.selective-dependence-response.programme-forecast",
        objective=(
            "Apply the exact nonpromotable two-target policy predicate without "
            "pooling, target rescue or evaluation outcome contact."
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
                decision_right_id="decision-right.selective-dependence-response.programme-forecast.execution",
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


def _complete_coverage(
    template: StudyTemplate,
    experiment: ExperimentSpec,
) -> StudyTemplate:
    bindings = {value.obligation_id: value for value in template.coverage.bindings}
    step = template.protocol.steps[0]
    incoming = tuple(
        sorted(
            value.edge_id
            for value in template.graph.edges
            if value.consumer_node_id == step.step_id
        )
    )
    for obligation_id in required_candidate_obligation_ids(
        experiment,
        template.protocol,
    ):
        bindings.setdefault(
            obligation_id,
            ObligationCoverageBinding(
                obligation_id=obligation_id,
                proof_owner_node_id=step.step_id,
                required_output_id=step.outputs[0].output_id,
                contributor_edge_ids=incoming,
            ),
        )
    return replace(
        template,
        coverage=ObligationCoverage(
            coverage_id="coverage.selective-dependence-response.programme-forecast.standard-authoring",
            bindings=tuple(sorted(bindings.values(), key=lambda value: value.obligation_id)),
        ),
    )


def _entry_package(
    *,
    draft: StudyDraft,
    register: FormalGapRegister,
    inventory: FormalGapSourceCapabilityInventory,
) -> ExperimentEntryPackage:
    applicability = derive_formal_gap_applicability(register, inventory)
    coverage = FormalGapCoverage(
        coverage_id="formal-gap-coverage.selective-dependence-response.programme-forecast",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=inventory.denominator_id,
        candidate_act_id=draft.draft_id,
        applicability=applicability,
        assignments=tuple(
            FormalGapCoverageAssignment(
                gap_id=value.gap_id,
                disposition=(FormalGapCoverageDisposition.NOT_APPLICABLE_TO_DENOMINATOR),
                readiness_reason=None,
                reason_codes=("FORMAL_LAW_GAP_NOT_APPLICABLE_TO_FINITE_PROGRAMME_PREDICATE",),
                selected_estimator_family_id=None,
                selected_control_ids=(),
                selected_multiplicity_family_id=None,
                obligation_ids=(),
                output_ids=(),
                adjudication_owner_ids=(),
            )
            for value in register.gaps
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(f"binding.selective-dependence-response.programme-forecast.{requirement.value.lower()}",),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                ("REFERENCE_WORLD_EXECUTION_AUTHORITY_REQUIRED",)
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ()
            ),
        )
        for requirement in sorted(ExperimentEntryRequirement, key=lambda value: value.value)
    )
    checklist = ExperimentEntryChecklist(
        checklist_id="entry-checklist.selective-dependence-response.programme-forecast",
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.ANALYTIC_REFERENCE,
        execution_route_id="route.standard-candidate-issue-compile-execute",
        durability_disposition_id="durability.external-receipt-first-seamios",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id="experiment-entry-package.selective-dependence-response.programme-forecast",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


def build_study_authoring_act(
    *,
    completions: tuple[
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
    ],
    method_question: SelectiveDependenceResponseMethodQuestionFreeze,
    implementation_sha256: str,
    register: FormalGapRegister,
) -> SelectiveDependenceResponseStudyAuthoringAct:
    """Compile the exact two-parent programme gate to authority-pending."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    if any(
        value.development_implementation_sha256 != implementation_sha256 for value in completions
    ):
        raise ValueError("programme completion implementation identity differs")
    stage = compose_study_forecast(
        completions,
        method_question=method_question,
        implementation_sha256=implementation_sha256,
    )
    system = build_study_system()
    experiment = build_study_experiment(system)
    campaign = build_study_campaign(system, experiment)
    template = _complete_coverage(stage.catalog.templates[0], experiment)
    catalog = CandidateCapabilityCatalog(
        catalog_id="selective-dependence-response-programme-forecast-authoring-catalog",
        registrations=stage.catalog.registrations,
        templates=(template,),
    )
    external = {value.input_id: value for value in template.graph.external_inputs}
    source_spec = external["input.selective-dependence-response.programme-method-question"]
    if (
        source_spec.logical_artifact_id != method_question.freeze_id
        or source_spec.expected_content_sha256 != method_question.fingerprint()
    ):
        raise ValueError("programme method-question source differs from graph")
    source_config = SourceMaterializationConfig(
        config_id="source-config.selective-dependence-response.programme-method-question",
        source_id=source_spec.input_id,
        role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        content_sha256=method_question.fingerprint(),
        expected_size_bytes=len(method_question.canonical_bytes()),
        maximum_bytes=source_spec.maximum_size_bytes,
        payload_schema=method_question.SCHEMA,
        media_type=source_spec.media_type,
        read_mode=SourceReadMode.ORDINARY_BOUNDED,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    observer = ObjectIdentity(
        object_id="observer.selective-dependence-response.programme-method-question.canonical",
        object_schema='empirical-lawhood/methods/selective-dependence-response/study-question-observer',
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id="qualification.selective-dependence-response.programme-method-question.authoring",
        source_id=source_spec.input_id,
        materialization=ObjectIdentity.from_record(
            method_question.freeze_id,
            method_question,
        ),
        content_sha256=method_question.fingerprint(),
        evidence_world_id=system.world.world_id,
        observation_operator=observer,
        numerical_view_ids=(system.numerical_views[0].view_id,),
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
    cutoff = experiment.information_cutoffs[0]
    completion_inputs = tuple(
        DesignInputRecord(
            input_id=f"design-input.selective-dependence-response.programme.{value.target_slug}-completion",
            object_identity=ObjectIdentity.from_record(value.envelope_id, value),
            materialization_sha256=value.fingerprint(),
            information_cutoff=cutoff,
            role=DesignInputRole.DEVELOPMENT_TUNING,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            operator_id="service.selective-dependence-response-development-custody",
            physical_unit_ids=(
                f"unit-development.selective-dependence-response.programme.{value.target_slug}-completion",
            ),
        )
        for value in sorted(completions, key=lambda item: item.target_slug)
    )
    question_input = DesignInputRecord(
        input_id="design-input.selective-dependence-response.programme.method-question",
        object_identity=ObjectIdentity.from_record(
            method_question.freeze_id,
            method_question,
        ),
        materialization_sha256=method_question.fingerprint(),
        information_cutoff=cutoff,
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    design_inputs = tuple(
        sorted((*completion_inputs, question_input), key=lambda value: value.input_id)
    )
    source_ref = SourceMaterializationRef(
        source_id=source_spec.input_id,
        role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        evidence_world_id=system.world.world_id,
        materialization=ObjectIdentity.from_record(
            method_question.freeze_id,
            method_question,
        ),
        content_sha256=method_question.fingerprint(),
        source_config_sha256=source_config.fingerprint(),
        observation_operator=observer,
        numerical_view_ids=(system.numerical_views[0].view_id,),
        qualification_receipt=ObjectIdentity.from_record(
            qualification.receipt_id,
            qualification,
        ),
        access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
    )
    inventory = FormalGapSourceCapabilityInventory(
        inventory_id="formal-source-inventory.selective-dependence-response.programme-forecast",
        denominator_id=system.system_id,
        evidence_world=FormalGapEvidenceWorld.ANALYTIC_REFERENCE,
        source_materializations=(source_ref.materialization,),
        present_operand_ids=(),
        satisfied_prerequisite_ids=(),
        independent_unit_ids=(),
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=(system.numerical_views[0].view_id,),
        available_estimator_family_ids=(),
        available_control_ids=(),
        multiplicity_family_ids=(),
        denominator_inapplicable_gap_ids=tuple(value.gap_id for value in register.gaps),
        resource_blocked_gap_ids=(),
        requested_claim_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    draft = StudyDraft(
        draft_id="draft.selective-dependence-response.programme-forecast",
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            "Do the exact terminal Cantera and FiPy development completions satisfy "
            "the frozen outcome-free programme-policy diversity predicate?"
        ),
        alternative_ids=(
            "alternative.selective-dependence-response.programme-qualified",
            "alternative.selective-dependence-response.programme-stopped",
        ),
        design_origin=DesignOrigin(
            origin_id="origin.selective-dependence-response.programme-policy.predeclared",
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=tuple(value.input_id for value in design_inputs),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=design_inputs,
        development_unit_ids=_DEVELOPMENT_DECISION_UNIT_IDS,
        evaluation_unit_ids=_EVALUATION_DECISION_UNIT_IDS,
        development_seed_ids=("seed.selective-dependence-response.programme.completion-identities",),
        evaluation_seed_ids=("seed.selective-dependence-response.programme.exact-predicate",),
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
            for value in stage.registry.capabilities
        ),
        source_materializations=(source_ref,),
        resource_ceiling=_PROGRAMME_BUDGET,
    )
    entry = _entry_package(draft=draft, register=register, inventory=inventory)
    package = StudyDefinition(
        package_id="programme-authoring-package.selective-dependence-response.programme-forecast",
        draft=draft,
        entry_package=entry,
    )
    base_context = CandidateCompilationContext(
        context_id="candidate-context.selective-dependence-response.programme-forecast",
        registry=stage.registry,
        templates=(template,),
        qualifications=(qualification,),
        known_design_inputs=design_inputs,
        implementation_sha256=implementation_sha256,
    )
    context = StandardCandidateCompilationContext(
        context_id="standard-context.selective-dependence-response.programme-forecast",
        base=base_context,
        formal_methods=standard_formal_method_catalog(register),
        source_inventories=(inventory,),
    )
    payload = package.canonical_bytes()
    compilation = compile_study_candidate(
        authoring_package=package,
        authoring_materialization=AuthoringMaterializationIdentity(
            media_type="application/json",
            byte_count=len(payload),
            raw_materialization_sha256=sha256(b"application/json\x00" + payload).hexdigest(),
        ),
        context=context,
    )
    ordered_completions = tuple(sorted(completions, key=lambda value: value.target_slug))
    return SelectiveDependenceResponseStudyAuthoringAct(
        completions=(ordered_completions[0], ordered_completions[1]),
        question=method_question,
        stage_composition=stage,
        system=system,
        experiment=experiment,
        campaign=campaign,
        source_config=source_config,
        qualification=qualification,
        design_inputs=design_inputs,
        inventory=inventory,
        template=template,
        catalog=catalog,
        draft=draft,
        package=package,
        context=context,
        compilation=compilation,
    )


__all__ = [
    'SelectiveDependenceResponseStudyAuthoringAct',
    'build_study_authoring_act',
    'build_study_campaign',
    'build_study_experiment',
    'build_study_system',
]
