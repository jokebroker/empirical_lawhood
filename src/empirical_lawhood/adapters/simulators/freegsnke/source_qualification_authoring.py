"""Prospective authoring root for the independent substrate grounding FreeGSNKE source qualification."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.methods.independent_substrate_design_protocol import IndependentSubstrateTargetAuthoringManifest
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
from empirical_lawhood.kernel.quantities import (
    QuantityKind,
    QuantitySpec,
    ResponseDirection,
)
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
from empirical_lawhood.planning.formal_gaps import (
    FormalGapApplicability,
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
)
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext, StudyTemplate
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_resolution import (
    SourceMaterializationConfig,
    SourceReadMode,
)

from .contracts import FreeGsnkePhase, FreeGsnkeProcessRequest, FreeGsnkeSavedPreparation, FreeGsnkeSourceBinding
from .generation_protocol import FREEGSNKE_SOURCE_ARTIFACT_ID
from .registry import FREEGSNKE_CAPABILITY_VERSION, FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY
from .source_qualification_protocol import FREEGSNKE_SOURCE_QUALIFICATION_CONFIG_ARTIFACT_ID, FREEGSNKE_SOURCE_QUALIFICATION_EVALUATOR_KEY, FREEGSNKE_SOURCE_QUALIFICATION_REDUCER_KEY, build_freegsnke_source_qualification_protocol, freegsnke_source_qualification_candidate_registrations, freegsnke_source_qualification_study_template, freegsnke_source_qualification_registry, freegsnke_source_qualification_scientific_graph
from .source_qualification import FreeGsnkeSourceQualificationConfig


FREEGSNKE_SOURCE_QUALIFICATION_CAMPAIGN_ID = (
    "campaign.independent-substrate-grounding.freegsnke-source-action-qualification"
)
FREEGSNKE_SOURCE_QUALIFICATION_SYSTEM_ID = (
    "system.independent-substrate-grounding.freegsnke-source-action-qualification"
)
FREEGSNKE_SOURCE_QUALIFICATION_EXPERIMENT_ID = (
    "experiment.independent-substrate-grounding.freegsnke-source-action-qualification"
)
FREEGSNKE_SOURCE_QUALIFICATION_DEVELOPMENT_UNIT_ID = (
    "freegsnke-source-scout-c280964-preparation-01"
)


def freegsnke_source_qualification_study_budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=5,
        memory_bytes=20 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=8 * 12 * 60 * 60 + 5 * 60,
        source_scan_bytes=8 * 2 * 1024**3 + 256 * 1024**2,
        output_bytes=8 * 256 * 1024**2 + 64 * 1024**2,
    )


def _quantity(
    *,
    quantity_id: str,
    kind: QuantityKind,
    unit: str,
    frame: str,
    clock_id: str,
    phase: CausalPhase,
    access: OutcomeAccess,
    direction: ResponseDirection = ResponseDirection.NOT_APPLICABLE,
) -> QuantitySpec:
    return QuantitySpec(
        quantity_id=quantity_id,
        label=quantity_id.replace(".", " "),
        kind=kind,
        dimension=quantity_id.split(".", 1)[0],
        native_unit=unit,
        coordinate_frame=frame,
        clock_id=clock_id,
        availability=AvailabilitySpec(
            clock_id=clock_id,
            phase=phase,
            outcome_access=access,
        ),
        response_direction=direction,
    )


def freegsnke_source_qualification_system(
    *,
    source_binding: FreeGsnkeSourceBinding,
    saved_preparation: FreeGsnkeSavedPreparation,
) -> SystemSpec:
    clock = ClockSpec(
        clock_id="clock.independent-substrate-grounding.freegsnke-solver",
        label="FreeGSNKE evolutive solver clock",
        time_unit="s",
        coordinate_frame="freegsnke-solver-time",
        sampling=SamplingSemantics.REGULAR,
        hold=HoldSemantics.ZERO_ORDER,
        label_semantics=ClockLabelSemantics.INSTANT,
        nominal_period=Decimal("0.0005"),
        alignment_tolerance=Decimal("1e-12"),
    )
    denominator_rows = (
        ("denominator.alpha-m", "1"),
        ("denominator.alpha-n", "1"),
        ("denominator.elongation-target", "1"),
        ("denominator.fvac", "T m"),
        ("denominator.plasma-current-target", "A"),
        ("denominator.pressure-axis", "Pa"),
    )
    receiver_rows = (
        "receiver.action-clock-ledger-status",
        "receiver.dynamic-gs-convergence-status",
        "receiver.linearization-domain-status",
    )
    quantities = [
        *(
            _quantity(
                quantity_id=quantity_id,
                kind=QuantityKind.DENOMINATOR,
                unit=unit,
                frame="freegsnke-preparation",
                clock_id=clock.clock_id,
                phase=CausalPhase.PRE_ACTION,
                access=OutcomeAccess.OUTCOME_BLIND,
            )
            for quantity_id, unit in denominator_rows
        ),
        _quantity(
            quantity_id="history.initial-causal-state",
            kind=QuantityKind.HISTORY,
            unit="1",
            frame="freegsnke-plasma-metal-state",
            clock_id=clock.clock_id,
            phase=CausalPhase.PRE_ACTION,
            access=OutcomeAccess.OUTCOME_BLIND,
        ),
        _quantity(
            quantity_id="boundary.source-declared-linearization-domain",
            kind=QuantityKind.BOUNDARY,
            unit="1",
            frame="freegsnke-source-qualification",
            clock_id=clock.clock_id,
            phase=CausalPhase.RECEIVER,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        ),
        *(
            _quantity(
                quantity_id=f"action.{port}-voltage",
                kind=QuantityKind.ACTION,
                unit="V",
                frame=f"freegsnke-{port}-circuit",
                clock_id=clock.clock_id,
                phase=CausalPhase.ACTION_APPLIED,
                access=OutcomeAccess.OUTCOME_BLIND,
            )
            for port in ("p4", "p5")
        ),
        *(
            _quantity(
                quantity_id=quantity_id,
                kind=QuantityKind.RECEIVER,
                unit="1",
                frame="freegsnke-source-qualification",
                clock_id=clock.clock_id,
                phase=CausalPhase.RECEIVER,
                access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                direction=ResponseDirection.TARGET_BAND,
            )
            for quantity_id in receiver_rows
        ),
    ]
    quantities_tuple = tuple(sorted(quantities, key=lambda value: value.quantity_id))
    unit_id = "unit.independent-substrate-grounding.freegsnke-excluded-preparation"
    relation = RelationalIdentity(
        relation_id="relation.independent-substrate-grounding.freegsnke-source-action-support",
        denominator_quantity_ids=tuple(value[0] for value in denominator_rows),
        history_quantity_ids=("history.initial-causal-state",),
        memoryless=False,
        action_quantity_ids=("action.p4-voltage", "action.p5-voltage"),
        receiver_quantity_ids=receiver_rows,
        horizon=HorizonSpec(
            horizon_id="horizon.independent-substrate-grounding.freegsnke-source-qualification",
            clock_id=clock.clock_id,
            duration=Decimal("4.498"),
            time_unit="s",
        ),
    )
    world = WorldSpec(
        world_id="world.independent-substrate-grounding.freegsnke-numerical-simulator",
        label="FreeGSNKE 3.0.1 linear evolutive numerical response world",
        kind=WorldKind.NUMERICAL_SIMULATOR,
        represented_physics=tuple(
            sorted(
                (
                    "free-boundary-grad-shafranov-equilibrium",
                    "linearized-plasma-metal-circuit-response",
                    "mastu-like-passive-structure-coupling",
                )
            )
        ),
        unrepresented_physics=tuple(
            sorted(("facility-action-discrepancy", "nonlinear-post-domain-dynamics"))
        ),
        privileged_truth_quantity_ids=(),
        maximum_evidence=EvidenceCeiling.NON_PROMOTABLE,
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
    budget = freegsnke_source_qualification_study_budget()
    envelope = ComputabilityEnvelope(
        envelope_id="compute.independent-substrate-grounding.freegsnke-source-qualification",
        represented_effect_ids=world.represented_physics,
        unresolved_effect_ids=("post-linearization-domain-response",),
        required_structure_ids=(
            "all-issued-branches-accounted",
            "source-domain-stop-retained",
        ),
        computable_structure_ids=(
            "all-issued-branches-accounted",
            "source-domain-stop-retained",
        ),
        max_cpu_cores=budget.cpu_cores,
        max_memory_bytes=budget.memory_bytes,
        max_gpu_devices=0,
        max_wall_time_seconds=budget.wall_time_seconds,
        max_output_bytes=budget.output_bytes,
        worst_case_latency_seconds=Decimal(budget.wall_time_seconds),
        deadline_seconds=None,
    )
    view_identity = saved_preparation.numerical_view
    numerical_view = NumericalViewSpec(
        view_id=view_identity.object_id,
        world_id=world.world_id,
        physical_preparation_id=unit_id,
        equations_id="equations.freegsnke-linear-evolutive-v3-0-1",
        closure_ids=(
            "closure.freegs4e-v0-13-1",
            f"closure.machine.{source_binding.machine_id}",
        ),
        boundary_condition_ids=("boundary.mastu-like-lower-single-null",),
        coordinates=(
            NumericalCoordinateSpec(
                coordinate_id="coordinate.freegsnke-grid-points",
                kind=NumericalCoordinateKind.SPATIAL_GRID,
                value=Decimal(65 * 129),
                unit="grid-points",
                refinement_level=0,
            ),
            NumericalCoordinateSpec(
                coordinate_id="coordinate.freegsnke-timestep",
                kind=NumericalCoordinateKind.TIMESTEP,
                value=Decimal("0.0005"),
                unit="s",
                refinement_level=0,
            ),
        ),
        solver_id="solver.freegsnke-linear-evolutive",
        solver_version=source_binding.freegsnke_version,
        precision="float64",
        device_class="cpu",
        runtime_id=f"runtime.freegsnke.{source_binding.runtime_tree_sha256[:16]}",
        randomness=RandomnessSemantics.DETERMINISTIC,
        observation_operator_id="observer.freegsnke-source-domain-and-action-ledger",
        computability_envelope_id=envelope.envelope_id,
    )
    authority = AuthorityPolicy(
        policy_id="authority-policy.independent-substrate-grounding.freegsnke-source-qualification",
        delegator_id="human.project-owner",
        delegate_id="gate.independent-substrate-grounding.freegsnke-source-qualification",
        scope_ids=(FREEGSNKE_SOURCE_QUALIFICATION_CAMPAIGN_ID,),
        allowed_world_kinds=frozenset({WorldKind.NUMERICAL_SIMULATOR}),
        allowed_actions=frozenset(
            {
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.SIMULATION_EXECUTION,
                AuthorityAction.EVALUATOR_REVEAL,
            }
        ),
        allowed_source_classes=frozenset({SourceAccessClass.OFFICIAL_OPEN_PUBLIC}),
        required_gate_ids=(
            "clean-implementation",
            "exact-source-closure",
            "resource-envelope",
        ),
        nondelegable_actions=frozenset(
            {
                AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
                AuthorityAction.HIL_ACTUATION,
                AuthorityAction.LIVE_ACTUATION,
                AuthorityAction.SAFETY_SIGNIFICANT_OPERATION,
            }
        ),
        budget_ceiling=budget,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
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
                        authority_action=AuthorityAction.SIMULATION_EXECUTION,
                    )
                    for quantity_id in relation.action_quantity_ids
                ),
                *(
                    PortSpec(
                        port_id=f"port.{quantity_id}.output",
                        quantity_id=quantity_id,
                        clock_id=clock.clock_id,
                        direction=PortDirection.OUTPUT,
                        balance_role=BalanceRole.OBSERVATION,
                    )
                    for quantity_id in relation.receiver_quantity_ids
                ),
            ),
            key=lambda value: value.port_id,
        )
    )
    return SystemSpec(
        system_id=FREEGSNKE_SOURCE_QUALIFICATION_SYSTEM_ID,
        label="independent substrate grounding excluded FreeGSNKE source/action qualification",
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=quantities_tuple,
        independent_unit=IndependentUnitSpec(
            unit_id=unit_id,
            label="one excluded restored FreeGSNKE preparation",
            grouping_key="group.independent-substrate-grounding.freegsnke-excluded-preparation",
            scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        ),
        authority_policy=authority,
        ports=ports,
        computability_envelopes=(envelope,),
        numerical_views=(numerical_view,),
    )


def _obligations(
    system: SystemSpec, cutoff: InformationCutoff
) -> ScientificObligations:
    return ScientificObligations(
        obligations_id="obligations.independent-substrate-grounding.freegsnke-source-qualification",
        support=SupportSpec(
            support_id="support.independent-substrate-grounding.freegsnke-source-qualification",
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=0,
            nested_numerical_view_count=8,
            information_cutoff_id=cutoff.cutoff_id,
            chart_ids=("chart.freegsnke-p4-p5-native-voltage",),
            denominator_cell_ids=("cell.freegsnke-restored-lower-single-null",),
            action_bounds=(),
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id="validity.independent-substrate-grounding.freegsnke-source-qualification",
            validity_domain_ids=("domain.freegsnke-source-declared-linearization",),
            assumption_ids=("exact-held-source-runtime-machine-and-view",),
            exclusion_reason_codes=(
                "FREEGSNKE_DYNAMIC_GS_CONVERGENCE_STOP",
                "FREEGSNKE_LINEARIZATION_DOMAIN_STOP",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id="uncertainty.independent-substrate-grounding.freegsnke-source-qualification",
            method_key="complete-issued-branch-accounting-no-inference",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal("0.95"),
            interval_quantity_ids=system.relation.receiver_quantity_ids,
            limitation_codes=("EXCLUDED_NONPROMOTABLE_SOURCE_QUALIFICATION",),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=(
            FalsifierSpec(
                falsifier_id="falsifier.independent-substrate-grounding.freegsnke-linearization-domain",
                kind=FalsifierKind.STRUCTURAL_CONVERGENCE,
                capability_key=FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
                description="The source-declared linearization domain departs before a horizon.",
                decisive_rule="Retain any domain departure as an adverse qualification stop.",
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id="falsifier.independent-substrate-grounding.freegsnke-operational-omission",
                kind=FalsifierKind.NEGATIVE_CONTROL,
                capability_key=FREEGSNKE_SOURCE_QUALIFICATION_REDUCER_KEY,
                description="An issued branch is omitted or converted into nonentry.",
                decisive_rule="Every issued branch must contribute a typed terminal response.",
                status=ObligationStatus.REQUIRED,
            ),
        ),
        closure=ClosureSpec(
            closure_id="closure.independent-substrate-grounding.freegsnke-source-qualification",
            recurrence_cell_ids=("cell.freegsnke-restored-lower-single-null",),
            exchange_factor_ids=system.relation.denominator_quantity_ids,
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id="convergence.independent-substrate-grounding.freegsnke-source-qualification",
            required_structure_ids=(
                "exact-restored-causal-state",
                "source-linearization-domain",
            ),
            numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
            tolerances=(),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id="computability.independent-substrate-grounding.freegsnke-source-qualification",
            envelope_id=system.computability_envelopes[0].envelope_id,
            numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=("held-freegsnke-v3-0-1-runtime",),
        ),
    )


def freegsnke_source_qualification_experiment(system: SystemSpec) -> ExperimentSpec:
    cutoff = InformationCutoff(
        cutoff_id="cutoff.independent-substrate-grounding.freegsnke-before-source-qualification",
        clock_id=system.clocks[0].clock_id,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal(0),
    )
    claim = ClaimSpec(
        claim_id="claim.independent-substrate-grounding.freegsnke-source-action-support",
        world_id=system.world.world_id,
        relation_id=system.relation.relation_id,
        proposition=(
            "The exact held FreeGSNKE source either supports the predeclared P4/P5 "
            "action branches through their horizons or returns a typed source stop."
        ),
        estimand="Complete eight-branch typed source/action qualification disposition.",
        physical_independent_unit_id=system.independent_unit.unit_id,
        requested_rung=None,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        promotion_rule="No promotion; only a compact excluded qualification gate may be consumed.",
        assumption_ids=(
            "exact-held-source-runtime-machine-view-and-restored-preparation",
        ),
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
    )
    return ExperimentSpec(
        experiment_id=FREEGSNKE_SOURCE_QUALIFICATION_EXPERIMENT_ID,
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=(claim,),
        assignment=AssignmentSpec(
            assignment_id="assignment.independent-substrate-grounding.freegsnke-eight-source-branches",
            kind=AssignmentKind.SIMULATOR_INTERVENTION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism="Eight exact hold/signed/repeat branches restored from one excluded state.",
            support_restriction_ids=(
                "support.independent-substrate-grounding.freegsnke-source-qualification",
            ),
        ),
        measurement_quantity_ids=system.relation.receiver_quantity_ids,
        controls=(
            ControlSpec(
                control_id="control.independent-substrate-grounding.freegsnke-exact-hold-repeat",
                kind=ControlKind.PERSISTENCE,
                capability_key=FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
                target_quantity_ids=system.relation.receiver_quantity_ids,
                decisive_rule="Both holds and exact repeats must be retained in the issued roster.",
            ),
        ),
        precision_goals=(
            PrecisionGoal(
                goal_id="precision.independent-substrate-grounding.freegsnke-complete-branch-accounting",
                metric_id="issued-branch-accounting-fraction",
                target_width=Decimal("1e-12"),
                native_unit="1",
                maximum_independent_units=1,
                stopping_rule="Stop after all eight issued branches have typed terminal responses.",
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id="reveal.independent-substrate-grounding.freegsnke-source-qualification",
            development_unit_ids=(FREEGSNKE_SOURCE_QUALIFICATION_DEVELOPMENT_UNIT_ID,),
            evaluation_cohort_id="cohort.independent-substrate-grounding.freegsnke-excluded-preparation",
            evaluation_manifest_sha256=sha256(
                b"independent-substrate-grounding-freegsnke-source-action-qualification-eight-branch-roster"
            ).hexdigest(),
            sealed_outcome_artifact_ids=(
                "artifact.independent-substrate-grounding.freegsnke.source-qualification-result",
            ),
            evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            fresh_evidence=True,
        ),
        obligations=_obligations(system, cutoff),
        design_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
    )


def freegsnke_source_qualification_campaign(
    system: SystemSpec,
    experiment: ExperimentSpec,
) -> CampaignSpec:
    node = CampaignNode(
        node_id="campaign-node.independent-substrate-grounding.freegsnke-source-qualification",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(
            experiment.experiment_id, experiment
        ),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id=FREEGSNKE_SOURCE_QUALIFICATION_CAMPAIGN_ID,
        objective="Adjudicate the exact FreeGSNKE source/action support boundary without promotion.",
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(value.claim_id for value in experiment.claims),
        budget=freegsnke_source_qualification_study_budget(),
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id="decision-right.independent-substrate-grounding.freegsnke-source-execution",
                action=AuthorityAction.SIMULATION_EXECUTION,
                decision_maker_id=system.authority_policy.delegate_id,
                authority_policy_id=system.authority_policy.policy_id,
                delegated=True,
            ),
            DecisionRight(
                decision_right_id="decision-right.independent-substrate-grounding.freegsnke-source-reveal",
                action=AuthorityAction.EVALUATOR_REVEAL,
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


def _formal_coverage(
    *,
    register: FormalGapRegister,
    system: SystemSpec,
    draft_id: str,
) -> FormalGapCoverage:
    applicability = tuple(
        FormalGapApplicability(
            gap_id=gap.gap_id,
            evidence_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
            present_operand_ids=(),
            satisfied_prerequisite_ids=(),
            independent_unit_ids=(),
            independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
            numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
            available_estimator_family_ids=(),
            available_control_ids=(),
            multiplicity_family_ids=(),
            requested_claim_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            denominator_applicable=True,
            resource_envelope_satisfied=True,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        for gap in register.gaps
    )
    assignments = tuple(
        FormalGapCoverageAssignment(
            gap_id=gap.gap_id,
            disposition=FormalGapCoverageDisposition.DEFER_WITH_TYPED_PREREQUISITE,
            readiness_reason=ReadinessStatus.SOURCE_PREREQUISITE_NOT_MET,
            reason_codes=("FREEGSNKE_SOURCE_ACTION_QUALIFICATION_PENDING",),
            selected_estimator_family_id=None,
            selected_control_ids=(),
            selected_multiplicity_family_id=None,
            obligation_ids=(),
            output_ids=(),
            adjudication_owner_ids=(),
        )
        for gap in register.gaps
    )
    return FormalGapCoverage(
        coverage_id="formal-gap-coverage.independent-substrate-grounding.freegsnke-source-qualification",
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=system.system_id,
        candidate_act_id=draft_id,
        applicability=applicability,
        assignments=assignments,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
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
            object_ids=(
                f"binding.independent-substrate-grounding.freegsnke-source-{requirement.value.lower().replace('_', '-')}",
            ),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement
                is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                ("NONACTUATING_SIMULATION_EXECUTION_AUTHORITY_REQUIRED",)
                if requirement
                is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ()
            ),
        )
        for requirement in sorted(
            ExperimentEntryRequirement, key=lambda value: value.value
        )
    )
    checklist = ExperimentEntryChecklist(
        checklist_id="entry-checklist.independent-substrate-grounding.freegsnke-source-qualification",
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        execution_route_id="route.independent-substrate-grounding.freegsnke-source-qualification",
        durability_disposition_id="durability.external-receipt-first-seamios",
        bindings=bindings,
        transitions=tuple(
            sorted(ExperimentEntryTransition, key=lambda value: value.value)
        ),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id="experiment-entry-package.independent-substrate-grounding.freegsnke-source-qualification",
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


def _record_id(record: CanonicalRecord) -> str:
    for attribute in ("request_id", "config_id", "binding_id", "saved_state_id"):
        value = getattr(record, attribute, None)
        if isinstance(value, str):
            return value
    raise TypeError("unknown FreeGSNKE source qualification record")


@dataclass(frozen=True, slots=True)
class FreeGsnkeSourceQualificationAuthoringBundle:
    requests: tuple[FreeGsnkeProcessRequest, ...]
    qualification_config: FreeGsnkeSourceQualificationConfig
    source_binding: FreeGsnkeSourceBinding
    saved_preparation: FreeGsnkeSavedPreparation
    source_configs: tuple[SourceMaterializationConfig, ...]
    request_config_refs: tuple[CapabilityConfigRef, ...]
    qualification_config_ref: CapabilityConfigRef
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    registry: CapabilityRegistry
    protocol: ProtocolTemplate
    template: StudyTemplate
    catalog: CandidateCapabilityCatalog
    qualifications: tuple[MaterializationQualificationReceipt, ...]
    formal_coverage: FormalGapCoverage
    entry_package: ExperimentEntryPackage
    package: StudyDefinition
    draft: StudyDraft
    context: CandidateCompilationContext

    @property
    def candidate_input_payloads(self) -> tuple[bytes, ...]:
        records: tuple[CanonicalRecord, ...] = (
            *self.requests,
            self.qualification_config,
            self.source_binding,
            self.saved_preparation,
            *self.source_configs,
            *self.qualifications,
        )
        values = {record.fingerprint(): record.canonical_bytes() for record in records}
        return tuple(values[key] for key in sorted(values))


def build_freegsnke_source_qualification_authoring_bundle(
    *,
    target_authoring_manifest: IndependentSubstrateTargetAuthoringManifest,
    requests: tuple[FreeGsnkeProcessRequest, ...],
    qualification_config: FreeGsnkeSourceQualificationConfig,
    source_binding: FreeGsnkeSourceBinding,
    saved_preparation: FreeGsnkeSavedPreparation,
    register: FormalGapRegister,
    implementation_sha256: str,
) -> FreeGsnkeSourceQualificationAuthoringBundle:
    ordered_requests = tuple(sorted(requests, key=lambda value: value.request_id))
    if {value.phase for value in ordered_requests} != {
        FreeGsnkePhase.EVALUATION
    } or saved_preparation.phase is not FreeGsnkePhase.EVALUATION:
        raise ValueError(
            "prospective FreeGSNKE source qualification requires EVALUATION identities"
        )
    system = freegsnke_source_qualification_system(
        source_binding=source_binding,
        saved_preparation=saved_preparation,
    )
    experiment = freegsnke_source_qualification_experiment(system)
    campaign = freegsnke_source_qualification_campaign(system, experiment)
    registry = freegsnke_source_qualification_registry(
        implementation_sha256=implementation_sha256
    )
    generation_manifest = registry.resolve(
        FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
        FREEGSNKE_CAPABILITY_VERSION,
    )
    request_refs = tuple(
        CapabilityConfigRef(
            config_id=request.request_id,
            config_schema=request.SCHEMA,
            config_schema_sha256=generation_manifest.config_schema_sha256,
            content_sha256=request.fingerprint(),
            artifact_id=f"config-artifact.{request.request_id}",
        )
        for request in ordered_requests
    )
    reducer = registry.resolve(
        FREEGSNKE_SOURCE_QUALIFICATION_EVALUATOR_KEY,
        FREEGSNKE_CAPABILITY_VERSION,
    )
    qualification_ref = CapabilityConfigRef(
        config_id=qualification_config.config_id,
        config_schema=qualification_config.SCHEMA,
        config_schema_sha256=reducer.config_schema_sha256,
        content_sha256=qualification_config.fingerprint(),
        artifact_id=FREEGSNKE_SOURCE_QUALIFICATION_CONFIG_ARTIFACT_ID,
    )
    protocol = build_freegsnke_source_qualification_protocol(
        registry=registry,
        requests=ordered_requests,
        request_config_refs=request_refs,
        qualification_config=qualification_config,
        qualification_config_ref=qualification_ref,
    )
    graph = freegsnke_source_qualification_scientific_graph(
        protocol=protocol,
        registry=registry,
        requests=ordered_requests,
        source_binding=source_binding,
        saved_preparation=saved_preparation,
    )
    template = freegsnke_source_qualification_study_template(
        protocol=protocol,
        graph=graph,
        experiment=experiment,
    )
    catalog = CandidateCapabilityCatalog(
        catalog_id="candidate-catalog.independent-substrate-grounding.freegsnke-source-qualification",
        registrations=freegsnke_source_qualification_candidate_registrations(
            implementation_sha256=implementation_sha256
        ),
        templates=(template,),
    )
    records_by_source_id: dict[
        str,
        tuple[CanonicalRecord, str, SourceMaterializationRole],
    ] = {
        "input.independent-substrate-grounding.freegsnke.source-binding": (
            source_binding,
            FREEGSNKE_SOURCE_ARTIFACT_ID,
            SourceMaterializationRole.PREPARED_MEDIUM,
        ),
        (
            f"input.independent-substrate-grounding.freegsnke.saved-preparation.{saved_preparation.preparation_id}"
        ): (
            saved_preparation,
            saved_preparation.saved_state_id,
            SourceMaterializationRole.PREPARED_MEDIUM,
        ),
    }
    observation_operator = ObjectIdentity(
        object_id="observer.independent-substrate-grounding.freegsnke-source-qualification",
        object_schema='empirical-lawhood/simulators/freegsnke/source-qualification-observer',
        object_version="1.0.0",
        object_fingerprint=implementation_sha256,
    )
    source_configs = tuple(
        SourceMaterializationConfig(
            config_id=f"source-config.{source_id}",
            source_id=source_id,
            role=role,
            content_sha256=record.fingerprint(),
            expected_size_bytes=len(record.canonical_bytes()),
            maximum_bytes=4 * 1024**2,
            payload_schema=record.SCHEMA,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            read_mode=SourceReadMode.ORDINARY_BOUNDED,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        )
        for source_id, (record, _artifact_id, role) in sorted(
            records_by_source_id.items()
        )
    )
    qualifications = tuple(
        MaterializationQualificationReceipt(
            receipt_id=f"qualification.{source_id}",
            source_id=source_id,
            materialization=ObjectIdentity(
                object_id=artifact_id,
                object_schema=record.SCHEMA,
                object_version=record.VERSION,
                object_fingerprint=record.fingerprint(),
            ),
            content_sha256=record.fingerprint(),
            evidence_world_id=system.world.world_id,
            observation_operator=observation_operator,
            numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
            native_unit_ids=tuple(
                sorted({value.native_unit for value in system.quantities})
            ),
            frame_ids=tuple(
                sorted({value.coordinate_frame for value in system.quantities})
            ),
            clock_ids=tuple(value.clock_id for value in system.clocks),
            receiver_semantics_id=system.relation.relation_id,
            validity_contract_id=experiment.obligations.validity.validity_id,
            uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
            access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        )
        for source_id, (record, artifact_id, _role) in sorted(
            records_by_source_id.items()
        )
    )
    source_config_by_id = {value.source_id: value for value in source_configs}
    qualification_by_id = {value.source_id: value for value in qualifications}
    cutoff = experiment.information_cutoffs[0]
    design_records: tuple[CanonicalRecord, ...] = (
        source_binding,
        saved_preparation,
        *ordered_requests,
        qualification_config,
    )
    design_inputs = (
        DesignInputRecord(
            input_id="design-input.independent-substrate-grounding.freegsnke-source-qualification-manifest",
            object_identity=ObjectIdentity(
                object_id=target_authoring_manifest.manifest_id,
                object_schema=IndependentSubstrateTargetAuthoringManifest.SCHEMA,
                object_version="1.0.0",
                object_fingerprint=target_authoring_manifest.fingerprint(),
            ),
            materialization_sha256=target_authoring_manifest.fingerprint(),
            information_cutoff=cutoff,
            role=DesignInputRole.READINESS_METADATA,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            operator_id="human.project-owner",
        ),
        *(
            DesignInputRecord(
                input_id=f"design-input.independent-substrate-grounding.freegsnke-source-{_record_id(record)}",
                object_identity=ObjectIdentity.from_record(_record_id(record), record),
                materialization_sha256=record.fingerprint(),
                information_cutoff=cutoff,
                role=DesignInputRole.READINESS_METADATA,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                operator_id="human.project-owner",
            )
            for record in design_records
        ),
    )
    design_inputs = tuple(sorted(design_inputs, key=lambda value: value.input_id))
    draft_id = "draft.independent-substrate-grounding.freegsnke-source-action-qualification"
    source_materializations = tuple(
        SourceMaterializationRef(
            source_id=source_id,
            role=role,
            evidence_world_id=system.world.world_id,
            materialization=qualification_by_id[source_id].materialization,
            content_sha256=record.fingerprint(),
            source_config_sha256=source_config_by_id[source_id].fingerprint(),
            observation_operator=observation_operator,
            numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
            qualification_receipt=ObjectIdentity.from_record(
                qualification_by_id[source_id].receipt_id,
                qualification_by_id[source_id],
            ),
            access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        )
        for source_id, (record, _artifact_id, role) in sorted(
            records_by_source_id.items()
        )
    )
    draft = StudyDraft(
        draft_id=draft_id,
        lifecycle=StudyDraftLifecycle.DRAFT,
        question="Does the exact held FreeGSNKE source support the frozen eight-branch action act?",
        alternative_ids=(
            "alternative.freegsnke-source-action-pass",
            "alternative.freegsnke-source-action-stop",
            "alternative.freegsnke-source-operational-unevaluable",
        ),
        design_origin=DesignOrigin(
            origin_id="origin.independent-substrate-grounding.freegsnke-source-owner-predeclared",
            kind=DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            declared_input_ids=tuple(value.input_id for value in design_inputs),
            parent_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        ),
        design_inputs=design_inputs,
        development_unit_ids=(FREEGSNKE_SOURCE_QUALIFICATION_DEVELOPMENT_UNIT_ID,),
        evaluation_unit_ids=(saved_preparation.preparation_id,),
        development_seed_ids=(),
        evaluation_seed_ids=(),
        unresolved_decisions=(),
        system=system,
        experiment=experiment,
        campaign=campaign,
        dag_template_key=template.template_key,
        capability_selections=tuple(
            CapabilitySelection(
                capability_key=manifest.capability_key,
                capability_version=manifest.capability_version,
                implementation_sha256=manifest.implementation_sha256,
            )
            for manifest in tuple(
                registry.resolve(key, version)
                for key, version in sorted(
                    {
                        (step.capability_key, step.capability_version)
                        for step in protocol.steps
                    }
                )
            )
        ),
        source_materializations=source_materializations,
        resource_ceiling=freegsnke_source_qualification_study_budget(),
    )
    coverage = _formal_coverage(
        register=register, system=system, draft_id=draft.draft_id
    )
    entry = _entry_package(draft=draft, register=register, coverage=coverage)
    package = StudyDefinition(
        package_id="programme-authoring-package.independent-substrate-grounding.freegsnke-source-qualification",
        draft=draft,
        entry_package=entry,
    )
    context = CandidateCompilationContext(
        context_id="context.independent-substrate-grounding.freegsnke-source-qualification",
        registry=registry,
        templates=(template,),
        qualifications=qualifications,
        known_design_inputs=design_inputs,
        implementation_sha256=implementation_sha256,
    )
    return FreeGsnkeSourceQualificationAuthoringBundle(
        requests=ordered_requests,
        qualification_config=qualification_config,
        source_binding=source_binding,
        saved_preparation=saved_preparation,
        source_configs=source_configs,
        request_config_refs=request_refs,
        qualification_config_ref=qualification_ref,
        system=system,
        experiment=experiment,
        campaign=campaign,
        registry=registry,
        protocol=protocol,
        template=template,
        catalog=catalog,
        qualifications=qualifications,
        formal_coverage=coverage,
        entry_package=entry,
        package=package,
        draft=draft,
        context=context,
    )


__all__ = [
    "FREEGSNKE_SOURCE_QUALIFICATION_CAMPAIGN_ID",
    "FREEGSNKE_SOURCE_QUALIFICATION_EXPERIMENT_ID",
    "FREEGSNKE_SOURCE_QUALIFICATION_SYSTEM_ID",
    'FreeGsnkeSourceQualificationAuthoringBundle',
    "build_freegsnke_source_qualification_authoring_bundle",
    "freegsnke_source_qualification_campaign",
    "freegsnke_source_qualification_experiment",
    'freegsnke_source_qualification_study_budget',
    "freegsnke_source_qualification_system",
]
