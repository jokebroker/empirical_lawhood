"""Standard nonpromotable authoring root for selective dependence response cross-target adjudication.

This module authors only the finite deterministic adjudication act.  The two
target completion envelopes remain outcome-visible parent receipts from their
own simulator worlds; the analytic wrapper neither pools their native values
nor promotes them to analytic evidence.  The frozen method question is the
sole formal source materialization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

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
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_sha256,
    validate_stable_id,
)
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

from .completion import SelectiveDependenceResponseTargetCompletionEnvelope
from .composition import SelectiveDependenceResponseStageComposition, compose_cross_target
from .contracts import SelectiveDependenceResponseMethodQuestionFreeze


_CROSS_TARGET_BUDGET = ResourceBudget(
    cpu_cores=1,
    memory_bytes=256 * 1024**2,
    gpu_devices=0,
    wall_time_seconds=60,
    source_scan_bytes=8 * 1024**2,
    output_bytes=4 * 1024**2,
)
_PARENT_UNIT_IDS = (
    "unit-development.selective-dependence-response.cross-target.cantera-completion",
    "unit-development.selective-dependence-response.cross-target.fipy-completion",
)
_DECISION_UNIT_IDS = ("unit-evaluation.selective-dependence-response.cross-target.exact-adjudication",)
_TARGET_IDS = (
    "target.cantera-selective-dependence-response",
    "target.fipy-selective-dependence-response",
)


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseCrossTargetParentRoster(CanonicalRecord):
    """Symmetric compact nomination identity for the exact two terminal parents."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-cross-target-parent-roster'

    roster_id: str
    completion_envelopes: tuple[ObjectIdentity, ...]
    native_numeric_value_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        require_sorted_unique_ids(
            self.completion_envelopes,
            attribute="object_id",
            field_name="completion_envelopes",
        )
        if len(self.completion_envelopes) != 2:
            raise ValueError("cross-target parent roster requires exactly two completions")
        if any(
            value.object_schema != SelectiveDependenceResponseTargetCompletionEnvelope.SCHEMA
            for value in self.completion_envelopes
        ):
            raise ValueError("cross-target parent roster schema differs")
        if self.native_numeric_value_count:
            raise ValueError("cross-target parent roster cannot carry native numeric values")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("cross-target parent roster must remain outcome visible")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseCrossTargetAuthoringAct:
    """Pure authority-pending cross-target authoring result."""

    completions: tuple[
        SelectiveDependenceResponseTargetCompletionEnvelope,
        SelectiveDependenceResponseTargetCompletionEnvelope,
    ]
    parent_roster: SelectiveDependenceResponseCrossTargetParentRoster
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
        dimension="finite-cross-target-adjudication",
        native_unit="canonical-enum-code",
        coordinate_frame="selective-dependence-response-cross-target-nonpooling-adjudicator",
        clock_id="clock.selective-dependence-response.cross-target.adjudication",
        availability=AvailabilitySpec(
            clock_id="clock.selective-dependence-response.cross-target.adjudication",
            phase=phase,
            outcome_access=access,
        ),
        response_direction=direction,
    )


def build_cross_target_system() -> SystemSpec:
    """Describe the compact adjudicator, not either simulator response medium."""

    clock = ClockSpec(
        clock_id="clock.selective-dependence-response.cross-target.adjudication",
        label="selective dependence response exact cross-target adjudication clock",
        time_unit="decision-step",
        coordinate_frame="selective-dependence-response-cross-target-nonpooling-adjudicator",
        sampling=SamplingSemantics.EVENT_DRIVEN,
        hold=HoldSemantics.SOURCE_DEFINED,
        label_semantics=ClockLabelSemantics.EVENT,
        nominal_period=None,
        alignment_tolerance=Decimal(0),
    )
    action_rows = (
        ("action.accepted-cross-target-adjudication", CausalPhase.ACTION_REQUESTED),
        ("action.applied-frozen-precedence", CausalPhase.ACTION_APPLIED),
        ("action.realized-relation-state", CausalPhase.ACTION_APPLIED),
        ("action.requested-cross-target-adjudication", CausalPhase.ACTION_REQUESTED),
    )
    quantities = tuple(
        sorted(
            (
                _quantity(
                    quantity_id="denominator.exact-two-target-completion-roster",
                    kind=QuantityKind.DENOMINATOR,
                    phase=CausalPhase.PRE_ACTION,
                    access=OutcomeAccess.EVALUATION_REVEALED,
                ),
                _quantity(
                    quantity_id="history.frozen-compact-target-handoffs",
                    kind=QuantityKind.HISTORY,
                    phase=CausalPhase.PRE_ACTION,
                    access=OutcomeAccess.EVALUATION_REVEALED,
                ),
                *(
                    _quantity(
                        quantity_id=quantity_id,
                        kind=QuantityKind.ACTION,
                        phase=phase,
                        access=OutcomeAccess.EVALUATION_REVEALED,
                    )
                    for quantity_id, phase in action_rows
                ),
                _quantity(
                    quantity_id="receiver.frozen-relation-state-adjudicable",
                    kind=QuantityKind.RECEIVER,
                    phase=CausalPhase.RECEIVER,
                    access=OutcomeAccess.EVALUATION_REVEALED,
                    direction=ResponseDirection.HIGHER_IS_BETTER,
                ),
                _quantity(
                    quantity_id="sink.pooled-native-numeric-value-count",
                    kind=QuantityKind.SINK,
                    phase=CausalPhase.RECEIVER,
                    access=OutcomeAccess.EVALUATION_REVEALED,
                    direction=ResponseDirection.LOWER_IS_BETTER,
                ),
            ),
            key=lambda value: value.quantity_id,
        )
    )
    unit = IndependentUnitSpec(
        unit_id="unit.selective-dependence-response.cross-target.exact-two-completion-roster",
        label=(
            "one deterministic analytic adjudication over the complete Cantera and "
            "FiPy target-completion envelopes"
        ),
        grouping_key="group.selective-dependence-response.cross-target.exact-two-completion-roster",
    )
    relation = RelationalIdentity(
        relation_id="relation.selective-dependence-response.cross-target.frozen-four-component-precedence",
        denominator_quantity_ids=("denominator.exact-two-target-completion-roster",),
        history_quantity_ids=("history.frozen-compact-target-handoffs",),
        memoryless=False,
        action_quantity_ids=tuple(value[0] for value in action_rows),
        receiver_quantity_ids=("receiver.frozen-relation-state-adjudicable",),
        horizon=HorizonSpec(
            horizon_id="horizon.selective-dependence-response.cross-target.one-adjudication",
            clock_id=clock.clock_id,
            duration=Decimal(1),
            time_unit="decision-step",
        ),
    )
    world = WorldSpec(
        world_id="world.selective-dependence-response.cross-target.analytic-wrapper",
        label="selective dependence response finite nonpooling cross-target adjudication world",
        kind=WorldKind.ANALYTIC_REFERENCE,
        represented_physics=("deterministic-frozen-cross-target-precedence",),
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
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATION_REVEALED,
            }
        ),
    )
    structures = (
        "counterexample-precedes-support",
        "exact-two-terminal-target-envelopes",
        "no-native-cross-target-pooling",
        "unevaluable-precedes-counterexample-precedes-distinctiveness",
    )
    envelope = ComputabilityEnvelope(
        envelope_id="compute.selective-dependence-response.cross-target.finite-adjudicator",
        represented_effect_ids=("deterministic-cross-target-relation-decision",),
        unresolved_effect_ids=(),
        required_structure_ids=structures,
        computable_structure_ids=structures,
        max_cpu_cores=_CROSS_TARGET_BUDGET.cpu_cores,
        max_memory_bytes=_CROSS_TARGET_BUDGET.memory_bytes,
        max_gpu_devices=0,
        max_wall_time_seconds=_CROSS_TARGET_BUDGET.wall_time_seconds,
        max_output_bytes=_CROSS_TARGET_BUDGET.output_bytes,
        worst_case_latency_seconds=Decimal(60),
        deadline_seconds=Decimal(_CROSS_TARGET_BUDGET.wall_time_seconds),
    )
    view = NumericalViewSpec(
        view_id="view.selective-dependence-response.cross-target.exact-enum-adjudicator",
        world_id=world.world_id,
        physical_preparation_id=unit.unit_id,
        equations_id="equations.selective-dependence-response.cross-target.frozen-precedence",
        closure_ids=("closure.selective-dependence-response.cross-target.exact-record-decoding",),
        boundary_condition_ids=("boundary.selective-dependence-response.cross-target.exact-two-targets",),
        coordinates=(
            NumericalCoordinateSpec(
                coordinate_id="coordinate.selective-dependence-response.cross-target.target-count",
                kind=NumericalCoordinateKind.SOLVER_REFINEMENT,
                value=Decimal(2),
                unit="target-completion",
                refinement_level=0,
            ),
        ),
        solver_id="solver.selective-dependence-response.cross-target.exact-precedence",
        solver_version="1.0.0",
        precision="exact-canonical-record-and-enum-comparison",
        device_class="cpu",
        runtime_id="cpython-3.11-selective-dependence-response-cross-target",
        randomness=RandomnessSemantics.DETERMINISTIC,
        observation_operator_id="observer.selective-dependence-response.cross-target.adjudication",
        computability_envelope_id=envelope.envelope_id,
    )
    policy = AuthorityPolicy(
        policy_id="authority-policy.selective-dependence-response.cross-target",
        delegator_id="human.project-owner",
        delegate_id="gate.selective-dependence-response.cross-target",
        scope_ids=("campaign.selective-dependence-response.cross-target",),
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
            "cross-target-execution-authority",
            "exact-two-terminal-target-completions",
            "frozen-method-question",
            "no-native-value-pooling",
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
        budget_ceiling=_CROSS_TARGET_BUDGET,
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
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
                    port_id="port.receiver.frozen-relation-state-adjudicable.output",
                    quantity_id="receiver.frozen-relation-state-adjudicable",
                    clock_id=clock.clock_id,
                    direction=PortDirection.OUTPUT,
                    balance_role=BalanceRole.OBSERVATION,
                ),
                PortSpec(
                    port_id="port.sink.pooled-native-numeric-value-count.output",
                    quantity_id="sink.pooled-native-numeric-value-count",
                    clock_id=clock.clock_id,
                    direction=PortDirection.OUTPUT,
                    balance_role=BalanceRole.INFORMATION,
                ),
            ),
            key=lambda value: value.port_id,
        )
    )
    return SystemSpec(
        system_id="system.selective-dependence-response.cross-target",
        label="selective dependence response compact exact-two-target adjudicator",
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


def _cross_target_obligations(
    system: SystemSpec,
    cutoff: InformationCutoff,
) -> ScientificObligations:
    action_bounds = tuple(
        QuantityBound(
            bound_id=f"bound.selective-dependence-response.cross-target.{quantity_id}",
            quantity_id=quantity_id,
            native_unit="canonical-enum-code",
            lower=Decimal(0),
            upper=Decimal(1),
        )
        for quantity_id in system.relation.action_quantity_ids
    )
    capability = "method.selective-dependence-response.cross-target.adjudicate-relation"
    view_id = system.numerical_views[0].view_id
    return ScientificObligations(
        obligations_id="obligations.selective-dependence-response.cross-target",
        support=SupportSpec(
            support_id="support.selective-dependence-response.cross-target.exact-two-completions",
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=1,
            nested_numerical_view_count=1,
            information_cutoff_id=cutoff.cutoff_id,
            chart_ids=("chart.selective-dependence-response.cross-target.frozen-relation-state",),
            denominator_cell_ids=("cell.selective-dependence-response.cross-target.cantera-plus-fipy",),
            action_bounds=action_bounds,
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id="validity.selective-dependence-response.cross-target.compact-nonpooling",
            validity_domain_ids=("domain.selective-dependence-response.cross-target.exact-two-targets",),
            assumption_ids=(
                "completion-envelopes-are-terminal",
                "handoffs-preserve-target-local-ceilings",
                "method-question-is-exact",
            ),
            exclusion_reason_codes=(
                "NO_ANALYTIC_RECLASSIFICATION_OF_SIMULATOR_EVIDENCE",
                "NO_NATIVE_COEFFICIENT_THRESHOLD_OR_SAMPLE_POOLING",
                "NO_TARGET_SPECIFIC_RESCUE",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id="uncertainty.selective-dependence-response.cross-target.exact-precedence",
            method_key="exact-finite-enum-precedence",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal("0.999999"),
            interval_quantity_ids=("sink.pooled-native-numeric-value-count",),
            limitation_codes=(
                "ANALYTIC_WRAPPER_ONLY",
                "EXACTLY_TWO_OUTCOME_VISIBLE_SELECTED_SIMULATORS",
                "NO_EXTERNAL_REPLICATION_OR_PHYSICAL_PROMOTION",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=(
            FalsifierSpec(
                falsifier_id="falsifier.selective-dependence-response.cross-target.ineligible-parent",
                kind=FalsifierKind.RECEIVER_GATE,
                capability_key=capability,
                description="Both target-local handoffs must be independently eligible.",
                decisive_rule=(
                    "Any ineligible target makes the relation unevaluable; it cannot be "
                    "replaced, pooled or compensated by the other target."
                ),
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id="falsifier.selective-dependence-response.cross-target.native-pooling",
                kind=FalsifierKind.NEGATIVE_CONTROL,
                capability_key=capability,
                description="No native target number may cross the compact boundary.",
                decisive_rule=(
                    "Any native coefficient, threshold, sample or nonzero native-value "
                    "count invalidates cross-target adjudication."
                ),
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id="falsifier.selective-dependence-response.cross-target.precedence",
                kind=FalsifierKind.BASELINE_COMPARATOR,
                capability_key=capability,
                description="The frozen terminal-state precedence is exact.",
                decisive_rule=(
                    "UNEVALUABLE precedes counterexample OPPOSED, which precedes "
                    "NOT_DISTINGUISHED; support is available only after all gates close."
                ),
                status=ObligationStatus.REQUIRED,
            ),
        ),
        closure=ClosureSpec(
            closure_id="closure.selective-dependence-response.cross-target.exact-precedence",
            recurrence_cell_ids=("cell.selective-dependence-response.cross-target.cantera-plus-fipy",),
            exchange_factor_ids=system.relation.denominator_quantity_ids,
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id="convergence.selective-dependence-response.cross-target.exact-records",
            required_structure_ids=(
                "counterexample-precedes-support",
                "exact-two-terminal-target-envelopes",
                "no-native-cross-target-pooling",
                "unevaluable-precedes-counterexample-precedes-distinctiveness",
            ),
            numerical_view_ids=(view_id,),
            tolerances=(),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id="computability.selective-dependence-response.cross-target.bounded",
            envelope_id=system.computability_envelopes[0].envelope_id,
            numerical_view_ids=(view_id,),
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=("selective-dependence-response.method-question-freeze",),
        ),
    )


def build_cross_target_experiment(system: SystemSpec) -> ExperimentSpec:
    """Build the nonpromotable analytic wrapper over revealed compact parents."""

    cutoff = InformationCutoff(
        cutoff_id="cutoff.selective-dependence-response.cross-target.before-adjudication",
        clock_id=system.clocks[0].clock_id,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal(0),
    )
    capability = "method.selective-dependence-response.cross-target.adjudicate-relation"
    return ExperimentSpec(
        experiment_id="experiment.selective-dependence-response.cross-target",
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=(
            ClaimSpec(
                claim_id="claim.selective-dependence-response.cross-target-adjudication.nonpromotable",
                world_id=system.world.world_id,
                relation_id=system.relation.relation_id,
                proposition=(
                    "The exact two compact target handoffs map to one frozen cross-target "
                    "relation state under the predeclared nonpooling precedence."
                ),
                estimand=(
                    "Exact target eligibility, component enum states, counterexample "
                    "presence and predictive-distinctiveness states; no native values."
                ),
                physical_independent_unit_id=system.independent_unit.unit_id,
                requested_rung=None,
                evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                promotion_rule=(
                    "No analytic, physical or target-local promotion; retain the two "
                    "simulator-local ceilings and report selection dependence explicitly."
                ),
                assumption_ids=(
                    "completion-envelopes-are-terminal-custody-proofs",
                    "cross-target-rule-was-frozen-before-target-outcomes",
                ),
                numerical_view_ids=(system.numerical_views[0].view_id,),
            ),
        ),
        assignment=AssignmentSpec(
            assignment_id="assignment.selective-dependence-response.cross-target.shadow-adjudication",
            kind=AssignmentKind.SHADOW_DECISION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism=(
                "A deterministic nonactuating decoder applies the frozen state "
                "precedence once to exactly two terminal compact handoffs."
            ),
            support_restriction_ids=("support.selective-dependence-response.cross-target.exact-two-completions",),
        ),
        measurement_quantity_ids=(
            "receiver.frozen-relation-state-adjudicable",
            "sink.pooled-native-numeric-value-count",
        ),
        controls=(
            ControlSpec(
                control_id="control.selective-dependence-response.cross-target.precedence-truth-table",
                kind=ControlKind.BASELINE_COMPARATOR,
                capability_key=capability,
                target_quantity_ids=("receiver.frozen-relation-state-adjudicable",),
                decisive_rule=(
                    "Every frozen terminal-state truth-table row must reproduce its exact "
                    "state; a later target-specific rule is forbidden."
                ),
            ),
        ),
        precision_goals=(
            PrecisionGoal(
                goal_id="precision.selective-dependence-response.cross-target.exact-one-adjudication",
                metric_id="exact-cross-target-state-mismatch-count",
                target_width=Decimal("0.000001"),
                native_unit="decision",
                maximum_independent_units=1,
                stopping_rule=(
                    "Apply the exact two-parent relation once or emit the typed stop; "
                    "never add a target, alternate metric or rescue rule."
                ),
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id="reveal.selective-dependence-response.cross-target.no-additional-target-outcomes",
            development_unit_ids=_PARENT_UNIT_IDS,
            evaluation_cohort_id="cohort.selective-dependence-response.cross-target.exact-adjudication",
            evaluation_manifest_sha256=sha256(
                b"selective-dependence-response:cross-target:exact-cantera-fipy-adjudication"
            ).hexdigest(),
            sealed_outcome_artifact_ids=("artifact.selective-dependence-response.cross-target.adjudication",),
            evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            fresh_evidence=True,
        ),
        obligations=_cross_target_obligations(system, cutoff),
        design_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
    )


def build_cross_target_campaign(
    system: SystemSpec,
    experiment: ExperimentSpec,
) -> CampaignSpec:
    node = CampaignNode(
        node_id="campaign-node.selective-dependence-response.cross-target.experiment",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id="campaign.selective-dependence-response.cross-target",
        objective=(
            "Apply the exact compact nonpooling cross-target precedence without "
            "target rescue, native-value transport or evidence-world promotion."
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(value.claim_id for value in experiment.claims),
        budget=_CROSS_TARGET_BUDGET,
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id="decision-right.selective-dependence-response.cross-target.execution",
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
    step = next(
        value
        for value in template.protocol.steps
        if value.step_id == "adjudicate-cross-target-relation"
    )
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
            coverage_id="coverage.selective-dependence-response.cross-target.standard-authoring",
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
        coverage_id="formal-gap-coverage.selective-dependence-response.cross-target",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=inventory.denominator_id,
        candidate_act_id=draft.draft_id,
        applicability=applicability,
        assignments=tuple(
            FormalGapCoverageAssignment(
                gap_id=value.gap_id,
                disposition=FormalGapCoverageDisposition.NOT_APPLICABLE_TO_DENOMINATOR,
                readiness_reason=None,
                reason_codes=("FORMAL_LAW_GAP_NOT_APPLICABLE_TO_FINITE_CROSS_TARGET_ADJUDICATOR",),
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
            object_ids=(f"binding.selective-dependence-response.cross-target.{requirement.value.lower()}",),
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
        checklist_id="entry-checklist.selective-dependence-response.cross-target",
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
        package_id="experiment-entry-package.selective-dependence-response.cross-target",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


def _ordered_completions(
    completions: tuple[
        SelectiveDependenceResponseTargetCompletionEnvelope,
        SelectiveDependenceResponseTargetCompletionEnvelope,
    ],
) -> tuple[SelectiveDependenceResponseTargetCompletionEnvelope, SelectiveDependenceResponseTargetCompletionEnvelope]:
    ordered = tuple(sorted(completions, key=lambda value: value.target_id))
    if tuple(value.target_id for value in ordered) != _TARGET_IDS:
        raise ValueError("cross-target authoring requires exact Cantera/FiPy completions")
    if any(value.native_numeric_value_count for value in ordered):
        raise ValueError("cross-target authoring rejects native numeric values")
    return ordered[0], ordered[1]


def build_cross_target_authoring_act(
    *,
    completions: tuple[
        SelectiveDependenceResponseTargetCompletionEnvelope,
        SelectiveDependenceResponseTargetCompletionEnvelope,
    ],
    method_question: SelectiveDependenceResponseMethodQuestionFreeze,
    implementation_sha256: str,
    register: FormalGapRegister,
) -> SelectiveDependenceResponseCrossTargetAuthoringAct:
    """Compile the exact nonpooling cross-target root to authority-pending."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    ordered = _ordered_completions(completions)
    parent_roster = SelectiveDependenceResponseCrossTargetParentRoster(
        roster_id="roster.selective-dependence-response.cross-target.exact-cantera-fipy",
        completion_envelopes=tuple(
            sorted(
                (ObjectIdentity.from_record(value.envelope_id, value) for value in ordered),
                key=lambda value: value.object_id,
            )
        ),
        native_numeric_value_count=0,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    stage = compose_cross_target(
        ordered,
        method_question=method_question,
        implementation_sha256=implementation_sha256,
    )
    system = build_cross_target_system()
    experiment = build_cross_target_experiment(system)
    campaign = build_cross_target_campaign(system, experiment)
    template = _complete_coverage(stage.catalog.templates[0], experiment)
    catalog = CandidateCapabilityCatalog(
        catalog_id="selective-dependence-response-cross-target-authoring-catalog",
        registrations=stage.catalog.registrations,
        templates=(template,),
    )
    external = {value.input_id: value for value in template.graph.external_inputs}
    source_spec = external["input.selective-dependence-response.cross-target-method-question"]
    if (
        source_spec.logical_artifact_id != method_question.freeze_id
        or source_spec.expected_content_sha256 != method_question.fingerprint()
    ):
        raise ValueError("cross-target method-question source differs from graph")
    source_config = SourceMaterializationConfig(
        config_id="source-config.selective-dependence-response.cross-target-method-question",
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
        object_id="observer.selective-dependence-response.cross-target-method-question.canonical",
        object_schema='empirical-lawhood/methods/selective-dependence-response/cross-target-question-observer',
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id="qualification.selective-dependence-response.cross-target-method-question.authoring",
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
            input_id=f"design-input.selective-dependence-response.cross-target.{value.target_id}-completion",
            object_identity=ObjectIdentity.from_record(value.envelope_id, value),
            materialization_sha256=value.fingerprint(),
            information_cutoff=cutoff,
            role=DesignInputRole.MOTIVATION,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            operator_id="service.selective-dependence-response-target-custody",
            physical_unit_ids=(
                f"unit-evaluation.selective-dependence-response.cross-target.{value.target_id}-completion",
            ),
        )
        for value in ordered
    )
    question_input = DesignInputRecord(
        input_id="design-input.selective-dependence-response.cross-target.method-question",
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
        inventory_id="formal-source-inventory.selective-dependence-response.cross-target",
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
        draft_id="draft.selective-dependence-response.cross-target",
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            "What state does the frozen four-component selective-response relation take across the "
            "exact terminal Cantera and FiPy compact handoffs?"
        ),
        alternative_ids=(
            "alternative.selective-dependence-response.cross-target.mixed",
            "alternative.selective-dependence-response.cross-target.not-distinguished",
            "alternative.selective-dependence-response.cross-target.opposed",
            "alternative.selective-dependence-response.cross-target.supported",
            "alternative.selective-dependence-response.cross-target.unevaluable",
        ),
        design_origin=DesignOrigin(
            origin_id="origin.selective-dependence-response.cross-target.predeclared",
            kind=DesignOriginKind.PROSPECTIVE_NOMINATION,
            declared_input_ids=tuple(value.input_id for value in design_inputs),
            parent_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            nomination=ObjectIdentity.from_record(
                parent_roster.roster_id,
                parent_roster,
            ),
            requests_fresh_child=True,
        ),
        design_inputs=design_inputs,
        development_unit_ids=_PARENT_UNIT_IDS,
        evaluation_unit_ids=_DECISION_UNIT_IDS,
        development_seed_ids=("seed.selective-dependence-response.cross-target.completion-identities",),
        evaluation_seed_ids=("seed.selective-dependence-response.cross-target.exact-precedence",),
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
        resource_ceiling=_CROSS_TARGET_BUDGET,
    )
    entry = _entry_package(draft=draft, register=register, inventory=inventory)
    package = StudyDefinition(
        package_id="programme-authoring-package.selective-dependence-response.cross-target",
        draft=draft,
        entry_package=entry,
    )
    base_context = CandidateCompilationContext(
        context_id="candidate-context.selective-dependence-response.cross-target",
        registry=stage.registry,
        templates=(template,),
        qualifications=(qualification,),
        known_design_inputs=design_inputs,
        implementation_sha256=implementation_sha256,
    )
    context = StandardCandidateCompilationContext(
        context_id="standard-context.selective-dependence-response.cross-target",
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
    return SelectiveDependenceResponseCrossTargetAuthoringAct(
        completions=ordered,
        parent_roster=parent_roster,
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
    'SelectiveDependenceResponseCrossTargetAuthoringAct',
    'SelectiveDependenceResponseCrossTargetParentRoster',
    "build_cross_target_authoring_act",
    "build_cross_target_campaign",
    "build_cross_target_experiment",
    "build_cross_target_system",
]
