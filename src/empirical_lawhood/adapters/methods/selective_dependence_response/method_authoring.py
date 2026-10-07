"Standard candidate-authoring root for the no-target selective dependence response method act.\n\nThe method act is an analytic, truth-known software qualification.  It is not\na measurement through controller use claim-bearing target experiment.  It uses the accepted standard\nauthoring route with an explicit NON_PROMOTABLE source-capability inventory and\ntyped formal-lowering deferrals; it does not manufacture a local law claim.  Its only\nscientific source is the exact outcome-blind method-question freeze; the\ntruth-known result remains privileged inside the conformance task and may be\ndeclassified only by :mod:`method_completion` after terminal receipt checks.\n"

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
from empirical_lawhood.kernel.serialization import CanonicalRecord
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
from empirical_lawhood.runtime.capabilities import CapabilityConfigRef, CapabilityRegistry
from empirical_lawhood.runtime.candidate_compiler import AuthoringMaterializationIdentity, CandidateCompilationContext, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, StandardCandidateCompilationContext, StudyCompilationReport, compile_study_candidate, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.source_resolution import SourceMaterializationConfig, SourceReadMode

from .conformance import SELECTIVE_DEPENDENCE_RESPONSE_CONFORMANCE_CASE_IDS
from .contracts import SelectiveDependenceResponseMethodQuestionFreeze
from .method import method_question_freeze
from .provider import SelectiveDependenceResponseMethodOperation, SelectiveDependenceResponseMethodRuntimeConfig, build_method_protocol, selective_dependence_response_method_catalog, selective_dependence_response_method_registry, method_runtime_config


SELECTIVE_DEPENDENCE_RESPONSE_METHOD_DEVELOPMENT_UNIT_IDS = (
    "case-development.action-stage-counterfeit",
    "case-development.complete-unit-counterfeit",
    "case-development.receiver-gate-counterfeit",
    "case-development.support-counterfeit",
)
SELECTIVE_DEPENDENCE_RESPONSE_METHOD_EVALUATION_UNIT_IDS = tuple(
    f"unit-evaluation.{value.removeprefix('case.')}" for value in SELECTIVE_DEPENDENCE_RESPONSE_CONFORMANCE_CASE_IDS
)

_METHOD_BUDGET = ResourceBudget(
    cpu_cores=1,
    memory_bytes=256 * 1024**2,
    gpu_devices=0,
    wall_time_seconds=20,
    source_scan_bytes=4 * 1024**2,
    output_bytes=4 * 1024**2,
)


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseMethodAuthoringAct:
    """Pure authority-pending authoring result for method qualification."""

    question: SelectiveDependenceResponseMethodQuestionFreeze
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    registry: CapabilityRegistry
    template: StudyTemplate
    catalog: CandidateCapabilityCatalog
    runtime_configs: tuple[SelectiveDependenceResponseMethodRuntimeConfig, ...]
    source_config: SourceMaterializationConfig
    qualification: MaterializationQualificationReceipt
    design_input: DesignInputRecord
    inventory: FormalGapSourceCapabilityInventory
    draft: StudyDraft
    package: StudyDefinition
    context: StandardCandidateCompilationContext
    compilation: StudyCompilationReport

    @property
    def candidate_input_payloads(self) -> tuple[bytes, ...]:
        """Return every compact content-addressed byte string needed to compile."""

        records: tuple[CanonicalRecord, ...] = (
            *self.runtime_configs,
            self.source_config,
            self.question,
            self.qualification,
        )
        return tuple(value.canonical_bytes() for value in records)


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
        dimension="finite-symbolic-category",
        native_unit="category-code",
        coordinate_frame="selective-dependence-response-method-truth-case",
        clock_id="clock.selective-dependence-response.method.case",
        availability=AvailabilitySpec(
            clock_id="clock.selective-dependence-response.method.case",
            phase=phase,
            outcome_access=access,
        ),
        response_direction=direction,
    )


def build_method_system() -> SystemSpec:
    """Bind method qualification to a finite analytic L(D,H,A,R,tau) world."""

    clock = ClockSpec(
        clock_id="clock.selective-dependence-response.method.case",
        label="selective dependence response analytic truth-case clock",
        time_unit="case-step",
        coordinate_frame="selective-dependence-response-method-case-order",
        sampling=SamplingSemantics.EVENT_DRIVEN,
        hold=HoldSemantics.SOURCE_DEFINED,
        label_semantics=ClockLabelSemantics.EVENT,
        nominal_period=None,
        alignment_tolerance=Decimal(0),
    )
    action_rows = (
        ("action.accepted", CausalPhase.ACTION_REQUESTED),
        ("action.applied", CausalPhase.ACTION_APPLIED),
        ("action.realized", CausalPhase.ACTION_APPLIED),
        ("action.requested", CausalPhase.ACTION_REQUESTED),
    )
    quantities = tuple(
        sorted(
            (
                _quantity(
                    quantity_id="denominator.case-family",
                    kind=QuantityKind.DENOMINATOR,
                    phase=CausalPhase.PRE_ACTION,
                    access=OutcomeAccess.OUTCOME_BLIND,
                ),
                _quantity(
                    quantity_id="history.truth-fixture-context",
                    kind=QuantityKind.HISTORY,
                    phase=CausalPhase.PRE_ACTION,
                    access=OutcomeAccess.OUTCOME_BLIND,
                ),
                *(
                    _quantity(
                        quantity_id=quantity_id,
                        kind=QuantityKind.ACTION,
                        phase=phase,
                        access=OutcomeAccess.OUTCOME_BLIND,
                    )
                    for quantity_id, phase in action_rows
                ),
                _quantity(
                    quantity_id="receiver.expected-classification",
                    kind=QuantityKind.OBSERVATION,
                    phase=CausalPhase.RECEIVER,
                    access=OutcomeAccess.PRIVILEGED_TRUTH,
                ),
                _quantity(
                    quantity_id="receiver.observed-classification",
                    kind=QuantityKind.OBSERVATION,
                    phase=CausalPhase.RECEIVER,
                    access=OutcomeAccess.PRIVILEGED_TRUTH,
                ),
                _quantity(
                    quantity_id="receiver.exact-case-pass",
                    kind=QuantityKind.RECEIVER,
                    phase=CausalPhase.RECEIVER,
                    access=OutcomeAccess.PRIVILEGED_TRUTH,
                    direction=ResponseDirection.HIGHER_IS_BETTER,
                ),
                _quantity(
                    quantity_id="sink.exact-mismatch",
                    kind=QuantityKind.SINK,
                    phase=CausalPhase.RECEIVER,
                    access=OutcomeAccess.PRIVILEGED_TRUTH,
                    direction=ResponseDirection.LOWER_IS_BETTER,
                ),
            ),
            key=lambda value: value.quantity_id,
        )
    )
    unit = IndependentUnitSpec(
        unit_id="unit.selective-dependence-response.method.complete-truth-case",
        label=(
            "one complete analytic truth case; expected state, method state and all "
            "nested checks remain one independent preparation"
        ),
        grouping_key="group.selective-dependence-response.method.complete-truth-case",
    )
    relation = RelationalIdentity(
        relation_id="relation.selective-dependence-response.method.finite-truth-case",
        denominator_quantity_ids=("denominator.case-family",),
        history_quantity_ids=("history.truth-fixture-context",),
        memoryless=False,
        action_quantity_ids=tuple(value[0] for value in action_rows),
        receiver_quantity_ids=("receiver.exact-case-pass",),
        horizon=HorizonSpec(
            horizon_id="horizon.selective-dependence-response.method.one-case",
            clock_id=clock.clock_id,
            duration=Decimal(1),
            time_unit="case-step",
        ),
    )
    world = WorldSpec(
        world_id="world.selective-dependence-response.method.analytic-reference",
        label="selective dependence response finite truth-known method world",
        kind=WorldKind.ANALYTIC_REFERENCE,
        represented_physics=("analytic-selective-dependence-and-action-fibre-grammar",),
        unrepresented_physics=(
            "cantera-target-response",
            "fipy-target-response",
            "physical-substrate-response",
        ),
        privileged_truth_quantity_ids=("receiver.expected-classification",),
        maximum_evidence=EvidenceCeiling.NON_PROMOTABLE,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.PRIVILEGED_TRUTH,
            }
        ),
    )
    envelope = ComputabilityEnvelope(
        envelope_id="compute.selective-dependence-response.method.finite-enumeration",
        represented_effect_ids=("exact-finite-truth-case-enumeration",),
        unresolved_effect_ids=(),
        required_structure_ids=(
            "action-stage-distinction",
            "active-and-invariant-exchange-distinction",
            "complete-unit-preservation",
            "negative-and-unevaluable-dispositions",
            "noncompensating-admission",
            "zero-target-contact",
        ),
        computable_structure_ids=(
            "action-stage-distinction",
            "active-and-invariant-exchange-distinction",
            "complete-unit-preservation",
            "negative-and-unevaluable-dispositions",
            "noncompensating-admission",
            "zero-target-contact",
        ),
        max_cpu_cores=_METHOD_BUDGET.cpu_cores,
        max_memory_bytes=_METHOD_BUDGET.memory_bytes,
        max_gpu_devices=0,
        max_wall_time_seconds=_METHOD_BUDGET.wall_time_seconds,
        max_output_bytes=_METHOD_BUDGET.output_bytes,
        worst_case_latency_seconds=Decimal(10),
        deadline_seconds=Decimal(_METHOD_BUDGET.wall_time_seconds),
    )
    view = NumericalViewSpec(
        view_id="view.selective-dependence-response.method.exact-finite-interpreter",
        world_id=world.world_id,
        physical_preparation_id=unit.unit_id,
        equations_id="equations.selective-dependence-response.method.inference-functions",
        closure_ids=("closure.selective-dependence-response.method.exact-python-decimal",),
        boundary_condition_ids=("boundary.selective-dependence-response.method.fourteen-case-roster",),
        coordinates=(
            NumericalCoordinateSpec(
                coordinate_id="coordinate.selective-dependence-response.method.case-count",
                kind=NumericalCoordinateKind.SOLVER_REFINEMENT,
                value=Decimal(len(SELECTIVE_DEPENDENCE_RESPONSE_CONFORMANCE_CASE_IDS)),
                unit="truth-case",
                refinement_level=0,
            ),
        ),
        solver_id="solver.selective-dependence-response.method.exact-finite-enumerator",
        solver_version="1.0.0",
        precision="python-decimal-exact-for-declared-literals",
        device_class="cpu",
        runtime_id="cpython-3.11-selective-dependence-response-method",
        randomness=RandomnessSemantics.DETERMINISTIC,
        observation_operator_id="observer.selective-dependence-response.method.expected-observed-exact-match",
        computability_envelope_id=envelope.envelope_id,
    )
    policy = AuthorityPolicy(
        policy_id="authority-policy.selective-dependence-response.method-reference",
        delegator_id="human.project-owner",
        delegate_id="gate.selective-dependence-response.method-reference",
        scope_ids=("campaign.selective-dependence-response.method-reference",),
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
            "exact-method-question",
            "reference-world-execution-authority",
            "zero-target-contact",
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
        budget_ceiling=_METHOD_BUDGET,
        maximum_outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
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
                    port_id="port.receiver.exact-case-pass.output",
                    quantity_id="receiver.exact-case-pass",
                    clock_id=clock.clock_id,
                    direction=PortDirection.OUTPUT,
                    balance_role=BalanceRole.OBSERVATION,
                ),
                PortSpec(
                    port_id="port.sink.exact-mismatch.output",
                    quantity_id="sink.exact-mismatch",
                    clock_id=clock.clock_id,
                    direction=PortDirection.OUTPUT,
                    balance_role=BalanceRole.INFORMATION,
                ),
            ),
            key=lambda value: value.port_id,
        )
    )
    return SystemSpec(
        system_id="system.selective-dependence-response.method-reference",
        label="selective dependence response no-target analytic method qualification",
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


def _method_obligations(
    system: SystemSpec,
    cutoff: InformationCutoff,
) -> ScientificObligations:
    view_id = system.numerical_views[0].view_id
    action_bounds = tuple(
        QuantityBound(
            bound_id=f"bound.selective-dependence-response.method.{quantity_id}",
            quantity_id=quantity_id,
            native_unit="category-code",
            lower=Decimal(-1),
            upper=Decimal(1),
        )
        for quantity_id in system.relation.action_quantity_ids
    )
    return ScientificObligations(
        obligations_id="obligations.selective-dependence-response.method-reference",
        support=SupportSpec(
            support_id="support.selective-dependence-response.method.exact-case-roster",
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=len(SELECTIVE_DEPENDENCE_RESPONSE_METHOD_EVALUATION_UNIT_IDS),
            nested_numerical_view_count=1,
            information_cutoff_id=cutoff.cutoff_id,
            chart_ids=("chart.selective-dependence-response.method.four-stage-action",),
            denominator_cell_ids=tuple(
                f"cell.selective-dependence-response.method.{value.removeprefix('case.')}"
                for value in SELECTIVE_DEPENDENCE_RESPONSE_CONFORMANCE_CASE_IDS
            ),
            action_bounds=action_bounds,
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id="validity.selective-dependence-response.method.exact-truth-known",
            validity_domain_ids=("domain.selective-dependence-response.method.fourteen-cases",),
            assumption_ids=(
                "declared-truth-oracle-is-exact",
                "method-question-precedes-truth-evaluation",
                "target-response-count-remains-zero",
            ),
            exclusion_reason_codes=(
                "NO_MEASUREMENT_TO_CONTROLLER_USE_PROMOTION",
                "NO_TARGET_CONSTRUCT_EVIDENCE",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id="uncertainty.selective-dependence-response.method.exact-enumeration",
            method_key="exact-finite-case-enumeration",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal("0.999999"),
            interval_quantity_ids=("sink.exact-mismatch",),
            limitation_codes=(
                "ANALYTIC_REFERENCE_ONLY",
                "FINITE_CASE_ROSTER_ONLY",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=(
            FalsifierSpec(
                falsifier_id="falsifier.selective-dependence-response.method.exact-counterexample",
                kind=FalsifierKind.NEGATIVE_CONTROL,
                capability_key="method.selective-dependence-response.verify-conformance",
                description="Any observed state differing from exact truth is decisive.",
                decisive_rule=(
                    "One failed case, missing case, extra case or target response fails "
                    "method completion; aggregate pass rate cannot compensate."
                ),
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id="falsifier.selective-dependence-response.method.noncompensating-gate",
                kind=FalsifierKind.RECEIVER_GATE,
                capability_key="method.selective-dependence-response.verify-conformance",
                description="A favourable target margin cannot compensate a failed sink.",
                decisive_rule=(
                    "The target-positive/sink-failed and unmeasured-hold cases must "
                    "remain NONATTEMPT and UNEVALUABLE respectively."
                ),
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id="falsifier.selective-dependence-response.method.selective-exchange",
                kind=FalsifierKind.ONE_FACTOR_EXCHANGE,
                capability_key="method.selective-dependence-response.verify-conformance",
                description="Active and invariant exchanges require different decisions.",
                decisive_rule=(
                    "Active-null and invariant-broken cases are opposed even when "
                    "other truth cases pass."
                ),
                status=ObligationStatus.REQUIRED,
            ),
        ),
        closure=ClosureSpec(
            closure_id="closure.selective-dependence-response.method.finite-case-roster",
            recurrence_cell_ids=("cell.selective-dependence-response.method.truth-known",),
            exchange_factor_ids=system.relation.denominator_quantity_ids,
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id="convergence.selective-dependence-response.method.exact-interpreter",
            required_structure_ids=(
                "exact-case-roster",
                "exact-expected-observed-comparison",
                "zero-target-contact",
            ),
            numerical_view_ids=(view_id,),
            tolerances=(),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id="computability.selective-dependence-response.method.bounded",
            envelope_id=system.computability_envelopes[0].envelope_id,
            numerical_view_ids=(view_id,),
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=("selective-dependence-response.method-question-freeze",),
        ),
    )


def build_method_experiment(system: SystemSpec) -> ExperimentSpec:
    """Build a non-promotable prospective analytic conformance experiment."""

    cutoff = InformationCutoff(
        cutoff_id="cutoff.selective-dependence-response.method.pre-truth",
        clock_id=system.clocks[0].clock_id,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal(0),
    )
    case_manifest = "\n".join(SELECTIVE_DEPENDENCE_RESPONSE_METHOD_EVALUATION_UNIT_IDS).encode("ascii")
    return ExperimentSpec(
        experiment_id="experiment.selective-dependence-response.method-reference",
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=(
            ClaimSpec(
                claim_id="claim.selective-dependence-response.method-conformance.nonpromotable",
                world_id=system.world.world_id,
                relation_id=system.relation.relation_id,
                proposition=(
                    "The exact implementation distinguishes all fourteen predeclared "
                    "truth-known action, exchange, response and admission cases while "
                    "contacting zero target responses."
                ),
                estimand=(
                    "Exact per-case expected/observed equality, complete case roster, "
                    "all-passed conjunction and target-response count zero."
                ),
                physical_independent_unit_id=system.independent_unit.unit_id,
                requested_rung=None,
                evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                promotion_rule=(
                    'No measurement-through-controller-use promotion. Only a terminal outcome-blind method-completion envelope may gate later target work.'
                ),
                assumption_ids=(
                    "analytic-truth-oracle-is-exact",
                    "truth-cases-do-not-contain-target-responses",
                ),
                numerical_view_ids=(system.numerical_views[0].view_id,),
            ),
        ),
        assignment=AssignmentSpec(
            assignment_id="assignment.selective-dependence-response.method.finite-truth-cases",
            kind=AssignmentKind.SHADOW_DECISION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism=(
                "Deterministic finite truth fixtures exercise the four-stage action "
                "semantics without target, simulator or physical actuation."
            ),
            support_restriction_ids=("support.selective-dependence-response.method.exact-case-roster",),
        ),
        measurement_quantity_ids=(
            "receiver.exact-case-pass",
            "receiver.expected-classification",
            "receiver.observed-classification",
            "sink.exact-mismatch",
        ),
        controls=(
            ControlSpec(
                control_id="control.selective-dependence-response.method.deliberate-counterexamples",
                kind=ControlKind.WRONG_ACTION,
                capability_key="method.selective-dependence-response.verify-conformance",
                target_quantity_ids=("receiver.exact-case-pass",),
                decisive_rule=(
                    "The active-null, invariant-broken, target-positive/sink-failed, "
                    "underpowered and unmeasured-hold cases retain their negative states."
                ),
            ),
        ),
        precision_goals=(
            PrecisionGoal(
                goal_id="precision.selective-dependence-response.method.exact-roster",
                metric_id="exact-case-mismatch-count",
                target_width=Decimal("0.000001"),
                native_unit="case",
                maximum_independent_units=len(SELECTIVE_DEPENDENCE_RESPONSE_METHOD_EVALUATION_UNIT_IDS),
                stopping_rule=(
                    "Evaluate the exact fourteen-case roster once or emit a typed stop; "
                    "never add, remove or retune cases after truth access."
                ),
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id="reveal.selective-dependence-response.method.truth-known",
            development_unit_ids=SELECTIVE_DEPENDENCE_RESPONSE_METHOD_DEVELOPMENT_UNIT_IDS,
            evaluation_cohort_id="cohort.selective-dependence-response.method.fourteen-truth-cases",
            evaluation_manifest_sha256=sha256(case_manifest).hexdigest(),
            sealed_outcome_artifact_ids=("artifact.selective-dependence-response.method.privileged-conformance-report",),
            evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            fresh_evidence=True,
        ),
        obligations=_method_obligations(system, cutoff),
        design_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
    )


def build_method_campaign(system: SystemSpec, experiment: ExperimentSpec) -> CampaignSpec:
    node = CampaignNode(
        node_id="campaign-node.selective-dependence-response.method-reference.experiment",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id="campaign.selective-dependence-response.method-reference",
        objective=(
            "Qualify the frozen selective dependence response finite method grammar in a no-target "
            "truth-known analytic act without scientific claim promotion."
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(value.claim_id for value in experiment.claims),
        budget=_METHOD_BUDGET,
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id="decision-right.selective-dependence-response.method-reference.execution",
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


def _config_refs(
    registry: CapabilityRegistry,
    configs: tuple[SelectiveDependenceResponseMethodRuntimeConfig, ...],
) -> dict[str, CapabilityConfigRef]:
    by_operation = {value.operation: value for value in configs}
    if set(by_operation) != set(SelectiveDependenceResponseMethodOperation):
        raise ValueError("method authoring runtime config roster differs")
    values: dict[str, CapabilityConfigRef] = {}
    for step_id, operation in (
        ("freeze-method-question", SelectiveDependenceResponseMethodOperation.FREEZE_QUESTION),
        ("verify-method-conformance", SelectiveDependenceResponseMethodOperation.VERIFY_CONFORMANCE),
    ):
        config = by_operation[operation]
        manifest = registry.resolve(config.capability_key, config.capability_version)
        values[step_id] = CapabilityConfigRef(
            config_id=f"config.selective-dependence-response.method.{step_id}",
            config_schema=config.SCHEMA,
            config_schema_sha256=manifest.config_schema_sha256,
            content_sha256=config.fingerprint(),
            artifact_id=f"config-artifact.selective-dependence-response.method.{step_id}",
        )
    return values


def _complete_coverage(
    template: StudyTemplate,
    experiment: ExperimentSpec,
) -> StudyTemplate:
    bindings = {value.obligation_id: value for value in template.coverage.bindings}
    step = next(
        value for value in template.protocol.steps if value.step_id == "verify-method-conformance"
    )
    incoming = tuple(
        sorted(
            value.edge_id
            for value in template.graph.edges
            if value.consumer_node_id == step.step_id
        )
    )
    for obligation_id in required_candidate_obligation_ids(experiment, template.protocol):
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
            coverage_id="coverage.selective-dependence-response.no-target-method.standard-authoring",
            bindings=tuple(sorted(bindings.values(), key=lambda value: value.obligation_id)),
        ),
    )


def _formal_inventory(
    *,
    register: FormalGapRegister,
    system: SystemSpec,
    experiment: ExperimentSpec,
    source_ref: SourceMaterializationRef,
) -> FormalGapSourceCapabilityInventory:
    return FormalGapSourceCapabilityInventory(
        inventory_id="formal-source-inventory.selective-dependence-response.method-reference",
        denominator_id=system.system_id,
        evidence_world=FormalGapEvidenceWorld.ANALYTIC_REFERENCE,
        source_materializations=(source_ref.materialization,),
        present_operand_ids=(),
        satisfied_prerequisite_ids=(),
        independent_unit_ids=tuple(
            sorted(
                (
                    *SELECTIVE_DEPENDENCE_RESPONSE_METHOD_DEVELOPMENT_UNIT_IDS,
                    *SELECTIVE_DEPENDENCE_RESPONSE_METHOD_EVALUATION_UNIT_IDS,
                )
            )
        ),
        independent_unit_scope=system.independent_unit.scope,
        numerical_view_ids=(system.numerical_views[0].view_id,),
        available_estimator_family_ids=(),
        available_control_ids=tuple(value.control_id for value in experiment.controls),
        multiplicity_family_ids=(),
        denominator_inapplicable_gap_ids=(),
        resource_blocked_gap_ids=tuple(value.gap_id for value in register.gaps),
        requested_claim_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _entry_package(
    *,
    draft: StudyDraft,
    register: FormalGapRegister,
    inventory: FormalGapSourceCapabilityInventory,
) -> ExperimentEntryPackage:
    applicability = derive_formal_gap_applicability(register, inventory)
    coverage = FormalGapCoverage(
        coverage_id="formal-gap-coverage.selective-dependence-response.method-reference",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=inventory.denominator_id,
        candidate_act_id=draft.draft_id,
        applicability=applicability,
        assignments=tuple(
            FormalGapCoverageAssignment(
                gap_id=value.gap_id,
                disposition=(FormalGapCoverageDisposition.DEFER_WITH_TYPED_PREREQUISITE),
                readiness_reason=ReadinessStatus.COMPUTABILITY_BOUNDARY,
                reason_codes=("FORMAL_LOWERING_NOT_APPLICABLE_TO_NONPROMOTABLE_METHOD_ACT",),
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
            object_ids=(f"binding.selective-dependence-response.method-reference.{requirement.value.lower()}",),
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
        checklist_id="entry-checklist.selective-dependence-response.method-reference",
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
        package_id="experiment-entry-package.selective-dependence-response.method-reference",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


def build_method_authoring_act(
    *,
    implementation_sha256: str,
    register: FormalGapRegister,
) -> SelectiveDependenceResponseMethodAuthoringAct:
    """Compile the exact no-target method package to authority-pending, without I/O."""

    question = method_question_freeze()
    system = build_method_system()
    experiment = build_method_experiment(system)
    campaign = build_method_campaign(system, experiment)
    registry = selective_dependence_response_method_registry(implementation_sha256=implementation_sha256)
    runtime_configs = tuple(method_runtime_config(value) for value in SelectiveDependenceResponseMethodOperation)
    protocol = build_method_protocol(
        registry=registry,
        config_by_step_id=_config_refs(registry, runtime_configs),
    )
    base_catalog = selective_dependence_response_method_catalog(protocol=protocol, registry=registry)
    template = _complete_coverage(base_catalog.templates[0], experiment)
    catalog = CandidateCapabilityCatalog(
        catalog_id="selective-dependence-response-method-authoring-catalog",
        registrations=base_catalog.registrations,
        templates=(template,),
    )
    source_spec = template.graph.external_inputs[0]
    if (
        source_spec.input_id != "input.method-question"
        or source_spec.logical_artifact_id != question.freeze_id
        or source_spec.expected_content_sha256 != question.fingerprint()
    ):
        raise ValueError("method authoring source differs from the exact graph")
    source_config = SourceMaterializationConfig(
        config_id="source-config.selective-dependence-response.method-question",
        source_id=source_spec.input_id,
        role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        content_sha256=question.fingerprint(),
        expected_size_bytes=len(question.canonical_bytes()),
        maximum_bytes=source_spec.maximum_size_bytes,
        payload_schema=question.SCHEMA,
        media_type=source_spec.media_type,
        read_mode=SourceReadMode.ORDINARY_BOUNDED,
        outcome_access=source_spec.outcome_access,
        visibility_ceiling=source_spec.visibility_ceiling,
    )
    observer = ObjectIdentity(
        object_id="observer.selective-dependence-response.method-question.canonical",
        object_schema='empirical-lawhood/methods/selective-dependence-response/method-question-observer',
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )
    qualification = MaterializationQualificationReceipt(
        receipt_id="qualification.selective-dependence-response.method-question.authoring",
        source_id=source_spec.input_id,
        materialization=ObjectIdentity.from_record(question.freeze_id, question),
        content_sha256=question.fingerprint(),
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
    design_input = DesignInputRecord(
        input_id="design-input.selective-dependence-response.method-question",
        object_identity=ObjectIdentity.from_record(question.freeze_id, question),
        materialization_sha256=question.fingerprint(),
        information_cutoff=cutoff,
        role=DesignInputRole.READINESS_METADATA,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        operator_id="human.project-owner",
    )
    source_ref = SourceMaterializationRef(
        source_id=source_spec.input_id,
        role=SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        evidence_world_id=system.world.world_id,
        materialization=ObjectIdentity.from_record(question.freeze_id, question),
        content_sha256=question.fingerprint(),
        source_config_sha256=source_config.fingerprint(),
        observation_operator=observer,
        numerical_view_ids=(system.numerical_views[0].view_id,),
        qualification_receipt=ObjectIdentity.from_record(
            qualification.receipt_id,
            qualification,
        ),
        access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
    )
    draft = StudyDraft(
        draft_id="draft.selective-dependence-response.method-reference",
        lifecycle=StudyDraftLifecycle.DRAFT,
        question=(
            "Does the frozen selective dependence response implementation recover every exact truth-known "
            "distinction with zero target response access?"
        ),
        alternative_ids=(
            "alternative.selective-dependence-response.method-conforms",
            "alternative.selective-dependence-response.method-fails",
            "alternative.selective-dependence-response.method-unevaluable",
        ),
        design_origin=DesignOrigin(
            origin_id="origin.selective-dependence-response.method-question.predeclared",
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=(design_input.input_id,),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=(design_input,),
        development_unit_ids=SELECTIVE_DEPENDENCE_RESPONSE_METHOD_DEVELOPMENT_UNIT_IDS,
        evaluation_unit_ids=SELECTIVE_DEPENDENCE_RESPONSE_METHOD_EVALUATION_UNIT_IDS,
        development_seed_ids=("seed.selective-dependence-response.method.development.static",),
        evaluation_seed_ids=("seed.selective-dependence-response.method.evaluation.deterministic",),
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
        ),
        source_materializations=(source_ref,),
        resource_ceiling=_METHOD_BUDGET,
    )
    inventory = _formal_inventory(
        register=register,
        system=system,
        experiment=experiment,
        source_ref=source_ref,
    )
    entry = _entry_package(
        draft=draft,
        register=register,
        inventory=inventory,
    )
    package = StudyDefinition(
        package_id="programme-authoring-package.selective-dependence-response.method-reference",
        draft=draft,
        entry_package=entry,
    )
    base_context = CandidateCompilationContext(
        context_id="context.selective-dependence-response.method-reference",
        registry=registry,
        templates=(template,),
        qualifications=(qualification,),
        known_design_inputs=(design_input,),
        implementation_sha256=implementation_sha256,
    )
    context = StandardCandidateCompilationContext(
        context_id="standard-context.selective-dependence-response.method-reference",
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
    return SelectiveDependenceResponseMethodAuthoringAct(
        question=question,
        system=system,
        experiment=experiment,
        campaign=campaign,
        registry=registry,
        template=template,
        catalog=catalog,
        runtime_configs=runtime_configs,
        source_config=source_config,
        qualification=qualification,
        design_input=design_input,
        inventory=inventory,
        draft=draft,
        package=package,
        context=context,
        compilation=compilation,
    )


__all__ = [
    "SELECTIVE_DEPENDENCE_RESPONSE_METHOD_DEVELOPMENT_UNIT_IDS",
    "SELECTIVE_DEPENDENCE_RESPONSE_METHOD_EVALUATION_UNIT_IDS",
    'SelectiveDependenceResponseMethodAuthoringAct',
    "build_method_authoring_act",
    "build_method_campaign",
    "build_method_experiment",
    "build_method_system",
]
