'Typed excluded solver control Pb control system, evidence world and experiment specification.'

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import ClaimSpec, EvidenceCeiling, OutcomeAccess, VisibilityCeiling
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
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.references import NamedDecimal, QuantityBound
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.systems import (
    IndependentUnitSpec,
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

from .contracts import ExcludedSolverControlConfig, SOLVER_CAPABILITY_KEY


SYSTEM_ID = 'system.ambient-pressure-superconductor-excluded-solver-control-pb-solver-smoke'
WORLD_ID = 'world.ambient-pressure-superconductor-excluded-solver-control-qe76-numerical-control'
RELATION_ID = 'relation.ambient-pressure-superconductor-excluded-solver-control-preparation-solver-response'
EXPERIMENT_ID = 'experiment.ambient-pressure-superconductor-excluded-solver-control-pb-solver-smoke'
INDEPENDENT_UNIT_ID = 'unit.ambient-pressure-superconductor-excluded-solver-control-one-held-pb-solver-acquisition'
CLOCK_ID = 'clock.ambient-pressure-superconductor-excluded-solver-control-solver-stage'
POLICY_ID = 'policy.ambient-pressure-superconductor-excluded-solver-control-public-local-simulation'
SCOPE_ID = 'scope.ambient-pressure-superconductor-excluded-solver-control-excluded-control'
ENVELOPE_ID = 'compute.ambient-pressure-superconductor-excluded-solver-control-qe76-pb'
CONTROL_UNIT_IDS = ('unit-instance.ambient-pressure-superconductor-excluded-solver-control-pb-control-001',)


def task_budget(config: ExcludedSolverControlConfig) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=int(config.resource("cpu_cores")),
        memory_bytes=int(config.resource("memory_bytes")),
        gpu_devices=int(config.resource("gpu_devices")),
        wall_time_seconds=int(config.resource("wall_time_seconds")),
        source_scan_bytes=int(config.resource("maximum_all_inputs_bytes")),
        output_bytes=int(config.resource("output_bytes")),
    )


def study_budget(config: ExcludedSolverControlConfig) -> ResourceBudget:
    one = task_budget(config)
    return ResourceBudget(
        cpu_cores=one.cpu_cores,
        memory_bytes=one.memory_bytes,
        gpu_devices=one.gpu_devices,
        wall_time_seconds=3 * one.wall_time_seconds,
        source_scan_bytes=3 * one.source_scan_bytes,
        output_bytes=3 * one.output_bytes,
    )


def _quantity(
    quantity_id: str,
    label: str,
    kind: QuantityKind,
    phase: CausalPhase,
    access: OutcomeAccess,
    *,
    dimension: str,
    unit: str,
    frame: str,
    direction: ResponseDirection = ResponseDirection.NOT_APPLICABLE,
) -> QuantitySpec:
    return QuantitySpec(
        quantity_id=quantity_id,
        label=label,
        kind=kind,
        dimension=dimension,
        native_unit=unit,
        coordinate_frame=frame,
        clock_id=CLOCK_ID,
        availability=AvailabilitySpec(clock_id=CLOCK_ID, phase=phase, outcome_access=access),
        response_direction=direction,
    )


def excluded_solver_control_system(config: ExcludedSolverControlConfig) -> SystemSpec:
    budget = study_budget(config)
    clock = ClockSpec(
        clock_id=CLOCK_ID,
        label="Prepared-input through converged-SCF stage",
        time_unit="stage",
        coordinate_frame="requested-accepted-applied-realized-solver-stage",
        sampling=SamplingSemantics.EVENT_DRIVEN,
        hold=HoldSemantics.SOURCE_DEFINED,
        label_semantics=ClockLabelSemantics.EVENT,
        alignment_tolerance=Decimal("0"),
    )
    quantities = tuple(
        sorted(
            (
                _quantity(
                    'ambient-pressure-superconductor-action-held-solver-control',
                    "Requested through realized held/no-op control action",
                    QuantityKind.ACTION,
                    CausalPhase.ACTION_REQUESTED,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="material-control-action",
                    unit="1",
                    frame="held-source-pb-scf",
                ),
                _quantity(
                    'ambient-pressure-superconductor-denominator-pb-preparation',
                    "Held EPW-distributed fcc Pb preparation",
                    QuantityKind.DENOMINATOR,
                    CausalPhase.PREPARATION,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="prepared-material-identity",
                    unit="1",
                    frame="fcc-primitive-cell",
                ),
                _quantity(
                    'ambient-pressure-superconductor-history-source-build-closure',
                    "Source, executable, input and pseudopotential closure",
                    QuantityKind.HISTORY,
                    CausalPhase.PRE_ACTION,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="solver-provenance-history",
                    unit="1",
                    frame="qe76-epw61-fixed-profile",
                ),
                _quantity(
                    'ambient-pressure-superconductor-receiver-scf-accuracy',
                    "Final self-consistent-field estimated accuracy",
                    QuantityKind.RECEIVER,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.EVALUATOR_REVEAL,
                    dimension="energy",
                    unit="Ry",
                    frame="qe-total-energy-closure",
                    direction=ResponseDirection.LOWER_IS_BETTER,
                ),
                _quantity(
                    'ambient-pressure-superconductor-receiver-total-energy',
                    "Converged Pb total energy",
                    QuantityKind.RECEIVER,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.EVALUATOR_REVEAL,
                    dimension="energy",
                    unit="Ry",
                    frame="qe-total-energy-closure",
                    direction=ResponseDirection.SIGNED_VECTOR,
                ),
                _quantity(
                    'ambient-pressure-superconductor-sink-compute-resource',
                    "Bounded compute and scratch expenditure",
                    QuantityKind.SINK,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.EVALUATOR_REVEAL,
                    dimension="computational-effort",
                    unit="cpu*s",
                    frame="trusted-local-fixed-profile",
                    direction=ResponseDirection.LOWER_IS_BETTER,
                ),
                _quantity(
                    'ambient-pressure-superconductor-truth-workflow-control',
                    "Privileged expected Pb workflow-control identity",
                    QuantityKind.OBSERVATION,
                    CausalPhase.POST_OUTCOME,
                    OutcomeAccess.PRIVILEGED_TRUTH,
                    dimension="truth-known-control-label",
                    unit="1",
                    frame="excluded-workflow-control",
                ),
            ),
            key=lambda value: value.quantity_id,
        )
    )
    relation = RelationalIdentity(
        relation_id=RELATION_ID,
        denominator_quantity_ids=('ambient-pressure-superconductor-denominator-pb-preparation',),
        history_quantity_ids=('ambient-pressure-superconductor-history-source-build-closure',),
        memoryless=False,
        action_quantity_ids=('ambient-pressure-superconductor-action-held-solver-control',),
        receiver_quantity_ids=(
            'ambient-pressure-superconductor-receiver-scf-accuracy',
            'ambient-pressure-superconductor-receiver-total-energy',
        ),
        horizon=HorizonSpec(
            horizon_id='horizon.ambient-pressure-superconductor-excluded-solver-control-one-converged-scf',
            clock_id=CLOCK_ID,
            duration=Decimal("2"),
            time_unit="stage",
        ),
    )
    authority = AuthorityPolicy(
        policy_id=POLICY_ID,
        delegator_id="human.project-owner",
        delegate_id="role.outcome-blind-scientific-approver",
        scope_ids=(SCOPE_ID,),
        allowed_world_kinds=frozenset({WorldKind.NUMERICAL_SIMULATOR}),
        allowed_actions=frozenset(
            {
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.PUBLIC_SOURCE_ACQUISITION,
                AuthorityAction.REPOSITORY_IMPLEMENTATION,
                AuthorityAction.SIMULATION_EXECUTION,
            }
        ),
        allowed_source_classes=frozenset(
            {SourceAccessClass.NONE, SourceAccessClass.OFFICIAL_OPEN_PUBLIC}
        ),
        required_gate_ids=(
            "clean-implementation",
            "exact-public-source-closure",
            "external-storage-active",
            "network-disabled-execution",
        ),
        nondelegable_actions=frozenset(
            {
                AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
                AuthorityAction.HIL_ACTUATION,
                AuthorityAction.HUMAN_OR_ANIMAL_INTERVENTION,
                AuthorityAction.LIVE_ACTUATION,
                AuthorityAction.PAID_OR_EXTERNALLY_BILLED_RESOURCE,
                AuthorityAction.SAFETY_SIGNIFICANT_OPERATION,
            }
        ),
        budget_ceiling=budget,
        maximum_outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
    )
    world = WorldSpec(
        world_id=WORLD_ID,
        label="QE 7.6 Pb excluded numerical workflow control",
        kind=WorldKind.NUMERICAL_SIMULATOR,
        represented_physics=(
            "plane-wave-pseudopotential-ground-state-dft",
            "self-consistent-kohn-sham-pz-lda",
            "solver-completion-and-convergence-semantics",
        ),
        unrepresented_physics=(
            "electron-phonon-coupling",
            "finite-temperature-superconducting-gap",
            "gauge-closed-transverse-response",
            "physical-material-realization",
            "room-temperature-superconductivity",
        ),
        privileged_truth_quantity_ids=('ambient-pressure-superconductor-truth-workflow-control',),
        maximum_evidence=EvidenceCeiling.NON_PROMOTABLE,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.PRIVILEGED_TRUTH,
            }
        ),
    )
    envelope = ComputabilityEnvelope(
        envelope_id=ENVELOPE_ID,
        represented_effect_ids=("pb-scf-ground-state", "qe-output-extraction"),
        unresolved_effect_ids=(
            "aggregate-descendant-resource-enforcement",
            "strict-network-namespace-evidence",
        ),
        required_structure_ids=(
            "exact-source-and-binary-binding",
            "finite-total-energy",
            "scf-convergence-marker",
        ),
        computable_structure_ids=(
            "exact-source-and-binary-binding",
            "finite-total-energy",
            "scf-convergence-marker",
        ),
        max_cpu_cores=budget.cpu_cores,
        max_memory_bytes=budget.memory_bytes,
        max_gpu_devices=0,
        max_wall_time_seconds=budget.wall_time_seconds,
        max_output_bytes=budget.output_bytes,
        worst_case_latency_seconds=Decimal(config.resource("wall_time_seconds")),
        deadline_seconds=Decimal(config.resource("wall_time_seconds")),
    )
    views = tuple(
        NumericalViewSpec(
            view_id=view_id,
            world_id=WORLD_ID,
            physical_preparation_id=INDEPENDENT_UNIT_ID,
            equations_id='equations.kohn-sham-pz-lda-plane-wave',
            closure_ids=(
                "closure.epw-bundled-pb-pseudopotential",
                "closure.qe76-scf-default-mixing",
            ),
            boundary_condition_ids=("boundary.fcc-pb-periodic-primitive-cell",),
            coordinates=(
                NumericalCoordinateSpec(
                    coordinate_id=coordinate_id,
                    kind=NumericalCoordinateKind.SOLVER_REFINEMENT,
                    value=Decimal(config.k_mesh[0]),
                    unit="points-per-reciprocal-axis",
                    refinement_level=level,
                ),
            ),
            solver_id="solver.quantum-espresso-pw",
            solver_version="7.6",
            precision="double",
            device_class="cpu",
            runtime_id='runtime.ambient-pressure-superconductor-qe76-epw61-local-ext4',
            randomness=RandomnessSemantics.DETERMINISTIC,
            observation_operator_id=observer_id,
            computability_envelope_id=ENVELOPE_ID,
        )
        for level, view_id, coordinate_id, observer_id in (
            (
                0,
                'view.ambient-pressure-superconductor-excluded-solver-control-qe76-pb-scf',
                'coordinate.ambient-pressure-superconductor-excluded-solver-control-kmesh-primary',
                'observer.ambient-pressure-superconductor-qe-pw-text-extractor',
            ),
            (
                1,
                'view.ambient-pressure-superconductor-excluded-solver-control-qe76-pb-scf-semantic-reparse',
                'coordinate.ambient-pressure-superconductor-excluded-solver-control-kmesh-semantic-reparse',
                'observer.ambient-pressure-superconductor-qe-pw-semantic-contract',
            ),
        )
    )
    return SystemSpec(
        system_id=SYSTEM_ID,
        label='ambient pressure superconductor excluded Pb solver-smoke system',
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=quantities,
        independent_unit=IndependentUnitSpec(
            unit_id=INDEPENDENT_UNIT_ID,
            label="One complete held Pb solver acquisition",
            grouping_key="solver-acquisition-id",
        ),
        authority_policy=authority,
        ports=(),
        computability_envelopes=(envelope,),
        numerical_views=views,
    )


def _obligations(system: SystemSpec, config: ExcludedSolverControlConfig) -> ScientificObligations:
    return ScientificObligations(
        obligations_id='obligations.ambient-pressure-superconductor-excluded-solver-control-pb-solver-smoke',
        support=SupportSpec(
            support_id='support.ambient-pressure-superconductor-excluded-solver-control-single-pb-control',
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=1,
            nested_numerical_view_count=1,
            information_cutoff_id='cutoff.ambient-pressure-superconductor-excluded-solver-control-pre-solver',
            chart_ids=('chart.ambient-pressure-superconductor-excluded-solver-control-hold-only',),
            denominator_cell_ids=('cell.ambient-pressure-superconductor-excluded-solver-control-pb-control',),
            action_bounds=(
                QuantityBound(
                    bound_id='bound.ambient-pressure-superconductor-excluded-solver-control-hold-action',
                    quantity_id='ambient-pressure-superconductor-action-held-solver-control',
                    native_unit="1",
                    lower=Decimal("0"),
                    upper=Decimal("0"),
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id='validity.ambient-pressure-superconductor-excluded-solver-control-qe-pb-control',
            validity_domain_ids=('cell.ambient-pressure-superconductor-excluded-solver-control-pb-control',),
            assumption_ids=(
                "bundled-pseudopotential-used-exactly",
                "excluded-truth-known-workflow-control",
                "no-superconductivity-inference",
            ),
            exclusion_reason_codes=(
                "no-alpha2f-operand",
                "no-gap-operand",
                "no-phonon-operand",
                "no-transverse-response-operand",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id='uncertainty.ambient-pressure-superconductor-excluded-solver-control-deterministic-output',
            method_key="deterministic-solver-output-no-statistical-interval",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal("0.999999"),
            interval_quantity_ids=('ambient-pressure-superconductor-receiver-scf-accuracy',),
            limitation_codes=(
                "one-numerical-view-only",
                "truth-known-workflow-control-only",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=(
            FalsifierSpec(
                falsifier_id='falsifier.ambient-pressure-superconductor-excluded-solver-control-exact-binding',
                kind=FalsifierKind.WRONG_ACTION,
                capability_key=SOLVER_CAPABILITY_KEY,
                description="Reject a changed source, input, pseudo, executable or profile.",
                decisive_rule='Any digest/profile mismatch fails excluded solver control before interpretation.',
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id='falsifier.ambient-pressure-superconductor-excluded-solver-control-solver-completion',
                kind=FalsifierKind.NEGATIVE_CONTROL,
                capability_key=SOLVER_CAPABILITY_KEY,
                description="Reject missing completion, convergence or expected Pb operands.",
                decisive_rule='Any missing or inconsistent required operand fails excluded solver control.',
                status=ObligationStatus.REQUIRED,
            ),
        ),
        closure=ClosureSpec(
            closure_id='closure.ambient-pressure-superconductor-excluded-solver-control-held-source-build-input',
            recurrence_cell_ids=('cell.ambient-pressure-superconductor-excluded-solver-control-pb-control',),
            exchange_factor_ids=('ambient-pressure-superconductor-denominator-pb-preparation',),
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id='convergence.ambient-pressure-superconductor-excluded-solver-control-single-view-smoke',
            required_structure_ids=(
                "finite-total-energy",
                "qe-completion-marker",
                "scf-convergence-marker",
            ),
            numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
            tolerances=(
                NamedDecimal(
                    value_id='tolerance.ambient-pressure-superconductor-excluded-solver-control-scf-accuracy',
                    value=config.convergence_threshold_Ry,
                    unit="Ry",
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id='computability.ambient-pressure-superconductor-excluded-solver-control-qe-profile',
            envelope_id=ENVELOPE_ID,
            numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
            readiness=(
                ReadinessStatus.READY
                if config.qe_binary_sha256
                else ReadinessStatus.PREREQUISITE_NOT_MET
            ),
            unresolved_reason_codes=(
                () if config.qe_binary_sha256 else ("qualified-qe-binary-required",)
            ),
            evidence_link_ids=('preflight.ambient-pressure-superconductor-resource-preflight-resource-and-loop-volume',),
        ),
    )


def excluded_solver_control_experiment(system: SystemSpec, config: ExcludedSolverControlConfig) -> ExperimentSpec:
    cutoff = InformationCutoff(
        cutoff_id='cutoff.ambient-pressure-superconductor-excluded-solver-control-pre-solver',
        clock_id=CLOCK_ID,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal("0"),
    )
    claim = ClaimSpec(
        claim_id='claim.ambient-pressure-superconductor-excluded-solver-control-qe-pb-solver-route',
        world_id=system.world.world_id,
        relation_id=system.relation.relation_id,
        proposition=(
            "The exact frozen QE profile completes the excluded Pb SCF control and "
            "emits every predeclared typed convergence operand."
        ),
        estimand="Exact completion, convergence and operand recovery for one Pb control.",
        physical_independent_unit_id=system.independent_unit.unit_id,
        requested_rung=None,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        promotion_rule='No promotion; excluded solver control is excluded method evidence only.',
        assumption_ids=("truth-known-pb-control",),
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
    )
    controls = (
        ControlSpec(
            control_id='control.ambient-pressure-superconductor-excluded-solver-control-exact-source-binary',
            kind=ControlKind.BASELINE_COMPARATOR,
            capability_key=SOLVER_CAPABILITY_KEY,
            target_quantity_ids=(
                'ambient-pressure-superconductor-receiver-scf-accuracy',
                'ambient-pressure-superconductor-receiver-total-energy',
            ),
            decisive_rule="Every source and executable digest must match before evaluation.",
        ),
        ControlSpec(
            control_id='control.ambient-pressure-superconductor-excluded-solver-control-hold-only',
            kind=ControlKind.NEGATIVE_ACTION,
            capability_key=SOLVER_CAPABILITY_KEY,
            target_quantity_ids=('ambient-pressure-superconductor-receiver-total-energy',),
            decisive_rule='No material transformation or target contact is permitted in excluded solver control.',
        ),
    )
    return ExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=(claim,),
        assignment=AssignmentSpec(
            assignment_id='assignment.ambient-pressure-superconductor-excluded-solver-control-held-pb-control',
            kind=AssignmentKind.SIMULATOR_INTERVENTION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism="Deterministic held/no-op execution of the solver-distributed Pb SCF input.",
            support_restriction_ids=('chart.ambient-pressure-superconductor-excluded-solver-control-hold-only',),
        ),
        measurement_quantity_ids=(
            'ambient-pressure-superconductor-receiver-scf-accuracy',
            'ambient-pressure-superconductor-receiver-total-energy',
            'ambient-pressure-superconductor-truth-workflow-control',
        ),
        controls=controls,
        precision_goals=(
            PrecisionGoal(
                goal_id='precision.ambient-pressure-superconductor-excluded-solver-control-exact-control-completion',
                metric_id="required-operand-recovery-count",
                target_width=Decimal("0.000001"),
                native_unit="1",
                maximum_independent_units=1,
                stopping_rule="Stop after the one frozen Pb acquisition; never retune.",
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id='barrier.ambient-pressure-superconductor-excluded-solver-control-workflow-control',
            development_unit_ids=CONTROL_UNIT_IDS,
            evaluation_cohort_id='cohort.ambient-pressure-superconductor-excluded-solver-control-pb-control',
            evaluation_manifest_sha256=sha256("\n".join(CONTROL_UNIT_IDS).encode()).hexdigest(),
            sealed_outcome_artifact_ids=('artifact.ambient-pressure-superconductor-excluded-solver-control-solver-observation',),
            evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            fresh_evidence=True,
        ),
        obligations=_obligations(system, config),
        design_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
    )


__all__ = [
    "CLOCK_ID",
    "CONTROL_UNIT_IDS",
    "ENVELOPE_ID",
    "EXPERIMENT_ID",
    "INDEPENDENT_UNIT_ID",
    "POLICY_ID",
    "RELATION_ID",
    "SCOPE_ID",
    "SYSTEM_ID",
    "WORLD_ID",
    'excluded_solver_control_experiment',
    'excluded_solver_control_system',
    'study_budget',
    "task_budget",
]
