"""Typed truth-known conformance system and experiment identity for uniform electron gas transverse receiver screen."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

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

from .contracts import METHOD_CAPABILITY_KEY, UniformElectronGasTransverseScreenConfig


SYSTEM_ID = "system.uniform-electron-gas-transverse-screen-conformance"
WORLD_ID = "world.uniform-electron-gas-transverse-screen-analytic-reference"
RELATION_ID = "relation.uniform-electron-gas-response-transverse-current"
EXPERIMENT_ID = "experiment.uniform-electron-gas-transverse-screen-conformance"
INDEPENDENT_UNIT_ID = "unit.uniform-electron-gas-response-complete-response-acquisition"
CLOCK_ID = "clock.uniform-electron-gas-response-static-acquisition"
POLICY_ID = "policy.uniform-electron-gas-transverse-screen-conformance"
SCOPE_ID = "scope.uniform-electron-gas-transverse-screen-conformance"
ENVELOPE_ID = "compute.uniform-electron-gas-transverse-screen-conformance"

TRUTH_UNIT_IDS = tuple(f"truth-case-{index:03d}" for index in range(1, 10))
TRUTH_SEED_IDS = ("seed.uniform-electron-gas-transverse-screen-conformance-deterministic",)


def task_budget(config: UniformElectronGasTransverseScreenConfig) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=int(config.resource("cpu_cores")),
        memory_bytes=int(config.resource("memory_bytes")),
        gpu_devices=int(config.resource("gpu_devices")),
        wall_time_seconds=int(config.resource("wall_time_seconds")),
        source_scan_bytes=int(config.resource("maximum_all_inputs_bytes")),
        output_bytes=int(config.resource("output_bytes")),
    )


def study_budget(config: UniformElectronGasTransverseScreenConfig) -> ResourceBudget:
    one = task_budget(config)
    task_count = 5
    return ResourceBudget(
        cpu_cores=one.cpu_cores,
        memory_bytes=one.memory_bytes,
        gpu_devices=one.gpu_devices,
        wall_time_seconds=task_count * one.wall_time_seconds,
        source_scan_bytes=task_count * one.source_scan_bytes,
        output_bytes=task_count * one.output_bytes,
    )


def _availability(phase: CausalPhase, access: OutcomeAccess) -> AvailabilitySpec:
    return AvailabilitySpec(
        clock_id=CLOCK_ID,
        phase=phase,
        outcome_access=access,
    )


def _quantity(
    quantity_id: str,
    label: str,
    kind: QuantityKind,
    phase: CausalPhase,
    access: OutcomeAccess,
    *,
    dimension: str,
    native_unit: str,
    frame: str,
    direction: ResponseDirection = ResponseDirection.NOT_APPLICABLE,
) -> QuantitySpec:
    return QuantitySpec(
        quantity_id=quantity_id,
        label=label,
        kind=kind,
        dimension=dimension,
        native_unit=native_unit,
        coordinate_frame=frame,
        clock_id=CLOCK_ID,
        availability=_availability(phase, access),
        response_direction=direction,
    )


def uniform_electron_gas_transverse_screen_system(config: UniformElectronGasTransverseScreenConfig) -> SystemSpec:
    clock = ClockSpec(
        clock_id=CLOCK_ID,
        label="Static source acquisition stage",
        time_unit="iteration",
        coordinate_frame="requested-accepted-applied-receiver-stage",
        sampling=SamplingSemantics.EVENT_DRIVEN,
        hold=HoldSemantics.SOURCE_DEFINED,
        label_semantics=ClockLabelSemantics.EVENT,
        alignment_tolerance=Decimal("0"),
    )
    quantities = tuple(
        sorted(
            (
                _quantity(
                    "uniform-electron-gas-response-action-transverse-vector-potential",
                    "Requested through realized transverse vector-potential action",
                    QuantityKind.ACTION,
                    CausalPhase.ACTION_REQUESTED,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="magnetic-vector-potential",
                    native_unit="T*m",
                    frame="q-x-polarization-y",
                ),
                _quantity(
                    "uniform-electron-gas-response-denominator-rs4-300k-jellium",
                    "Three-dimensional rs=4, 300 K neutral jellium denominator",
                    QuantityKind.DENOMINATOR,
                    CausalPhase.PREPARATION,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="prepared-medium-identity",
                    native_unit="1",
                    frame="three-dimensional-uniform-electron-gas",
                ),
                _quantity(
                    "uniform-electron-gas-response-history-static-convergence",
                    "Static solver convergence, normal control and order-of-limits history",
                    QuantityKind.HISTORY,
                    CausalPhase.PRE_ACTION,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="convergence-history",
                    native_unit="1",
                    frame="finite-amplitude-static-finite-q",
                ),
                _quantity(
                    "uniform-electron-gas-response-receiver-gauge-controls",
                    "Normal cancellation and Ward gauge-control receiver",
                    QuantityKind.RECEIVER,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    dimension="normalized-gauge-residual",
                    native_unit="1",
                    frame="producer-owned-static-gauge",
                    direction=ResponseDirection.LOWER_IS_BETTER,
                ),
                _quantity(
                    "uniform-electron-gas-response-receiver-order",
                    "Source-defined order opportunity operand",
                    QuantityKind.RECEIVER,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    dimension="normalized-order-operand",
                    native_unit="1",
                    frame="source-defined-order-frame",
                    direction=ResponseDirection.HIGHER_IS_BETTER,
                ),
                _quantity(
                    "uniform-electron-gas-response-receiver-transverse-current",
                    "Signed transverse current-density receiver",
                    QuantityKind.RECEIVER,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    dimension="electric-current-density",
                    native_unit="A/m^2",
                    frame="q-x-polarization-y",
                    direction=ResponseDirection.SIGNED_VECTOR,
                ),
                _quantity(
                    "uniform-electron-gas-response-sink-order-preservation",
                    "Order preservation under the maximum accepted probe",
                    QuantityKind.SINK,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    dimension="normalized-preservation-fraction",
                    native_unit="1",
                    frame="source-defined-order-frame",
                    direction=ResponseDirection.HIGHER_IS_BETTER,
                ),
                _quantity(
                    "uniform-electron-gas-response-truth-oracle",
                    "Privileged truth-family and expected-disposition oracle",
                    QuantityKind.OBSERVATION,
                    CausalPhase.POST_OUTCOME,
                    OutcomeAccess.PRIVILEGED_TRUTH,
                    dimension="truth-known-class-label",
                    native_unit="1",
                    frame="analytic-reference-oracle",
                ),
            ),
            key=lambda value: value.quantity_id,
        )
    )
    relation = RelationalIdentity(
        relation_id=RELATION_ID,
        denominator_quantity_ids=("uniform-electron-gas-response-denominator-rs4-300k-jellium",),
        history_quantity_ids=("uniform-electron-gas-response-history-static-convergence",),
        memoryless=False,
        action_quantity_ids=("uniform-electron-gas-response-action-transverse-vector-potential",),
        receiver_quantity_ids=(
            "uniform-electron-gas-response-receiver-gauge-controls",
            "uniform-electron-gas-response-receiver-order",
            "uniform-electron-gas-response-receiver-transverse-current",
        ),
        horizon=HorizonSpec(
            horizon_id="horizon.uniform-electron-gas-response-converged-static-receiver",
            clock_id=CLOCK_ID,
            duration=Decimal(config.receiver_clock - config.requested_clock),
            time_unit="iteration",
        ),
    )
    budget = study_budget(config)
    authority = AuthorityPolicy(
        policy_id=POLICY_ID,
        delegator_id="human.project-owner",
        delegate_id="role.outcome-blind-scientific-approver",
        scope_ids=(SCOPE_ID,),
        allowed_world_kinds=frozenset({WorldKind.ANALYTIC_REFERENCE}),
        allowed_actions=frozenset(
            {
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.REFERENCE_WORLD_EXECUTION,
                AuthorityAction.REPOSITORY_IMPLEMENTATION,
            }
        ),
        allowed_source_classes=frozenset({SourceAccessClass.NONE}),
        required_gate_ids=(
            "clean-implementation",
            "exact-source-closure",
            "separate-execution-and-reveal-authority",
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
        label="uniform electron gas transverse receiver truth-known analytic reference",
        kind=WorldKind.ANALYTIC_REFERENCE,
        represented_physics=(
            "analytic-london-slab-receiver",
            "deterministic-signed-transverse-current-fixtures",
            "normal-cancellation-and-ward-controls",
            "order-preservation-fixtures",
        ),
        unrepresented_physics=(
            "physical-material-realization",
            "target-paper-transverse-current-response",
            "time-dependent-field-expulsion",
            "vortices-and-critical-fields",
        ),
        privileged_truth_quantity_ids=("uniform-electron-gas-response-truth-oracle",),
        maximum_evidence=EvidenceCeiling.NON_PROMOTABLE,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.PRIVILEGED_TRUTH,
            }
        ),
    )
    envelope = ComputabilityEnvelope(
        envelope_id=ENVELOPE_ID,
        represented_effect_ids=(
            "deterministic-central-response",
            "fixed-even-q-intercept",
            "noncompensating-admission-intersection",
            "stable-analytic-slab-map",
        ),
        unresolved_effect_ids=(),
        required_structure_ids=(
            "nine-truth-family-classification",
            "per-view-local-law",
            "privileged-oracle-separation",
        ),
        computable_structure_ids=(
            "nine-truth-family-classification",
            "per-view-local-law",
            "privileged-oracle-separation",
        ),
        max_cpu_cores=budget.cpu_cores,
        max_memory_bytes=budget.memory_bytes,
        max_gpu_devices=budget.gpu_devices,
        max_wall_time_seconds=budget.wall_time_seconds,
        max_output_bytes=budget.output_bytes,
        worst_case_latency_seconds=Decimal("600"),
        deadline_seconds=Decimal(budget.wall_time_seconds),
    )
    views = tuple(
        NumericalViewSpec(
            view_id=view_id,
            world_id=WORLD_ID,
            physical_preparation_id=INDEPENDENT_UNIT_ID,
            equations_id="equations.uniform-electron-gas-response-planted-transverse-response",
            closure_ids=("closure.uniform-electron-gas-response-truth-known-deterministic",),
            boundary_condition_ids=("boundary.uniform-electron-gas-response-static-transverse-plane-wave",),
            coordinates=(
                NumericalCoordinateSpec(
                    coordinate_id=f"coordinate.uniform-electron-gas-response-{view_id}-refinement",
                    kind=NumericalCoordinateKind.SOLVER_REFINEMENT,
                    value=Decimal(level + 1),
                    unit="level",
                    refinement_level=level,
                ),
            ),
            solver_id="solver.uniform-electron-gas-response-truth-known-generator",
            solver_version="1.0.0",
            precision="decimal-with-codata-float-boundary",
            device_class="cpu",
            runtime_id="cpython-uniform-electron-gas-transverse-screen",
            randomness=RandomnessSemantics.DETERMINISTIC,
            observation_operator_id="observer.uniform-electron-gas-response-transverse-current",
            computability_envelope_id=ENVELOPE_ID,
        )
        for level, view_id in enumerate(config.views)
    )
    return SystemSpec(
        system_id=SYSTEM_ID,
        label="uniform electron gas transverse response/admission truth-known system",
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=quantities,
        independent_unit=IndependentUnitSpec(
            unit_id=INDEPENDENT_UNIT_ID,
            label="One complete deterministic response-panel acquisition",
            grouping_key="independent-unit-id",
        ),
        authority_policy=authority,
        ports=(),
        computability_envelopes=(envelope,),
        numerical_views=views,
    )


def _obligations(system: SystemSpec, config: UniformElectronGasTransverseScreenConfig) -> ScientificObligations:
    return ScientificObligations(
        obligations_id="obligations.uniform-electron-gas-transverse-screen-conformance",
        support=SupportSpec(
            support_id="support.uniform-electron-gas-transverse-screen-conformance",
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=len(TRUTH_UNIT_IDS),
            nested_numerical_view_count=len(config.views),
            information_cutoff_id="cutoff.uniform-electron-gas-transverse-screen-pre-action",
            chart_ids=("chart.uniform-electron-gas-response-u-q-transverse",),
            denominator_cell_ids=("cell.uniform-electron-gas-response-rs4-300k-truth-known",),
            action_bounds=(
                QuantityBound(
                    bound_id="bound.uniform-electron-gas-response-normalized-u",
                    quantity_id="uniform-electron-gas-response-action-transverse-vector-potential",
                    native_unit="1",
                    lower=-config.u0,
                    upper=config.u0,
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id="validity.uniform-electron-gas-transverse-screen-conformance",
            validity_domain_ids=("cell.uniform-electron-gas-response-rs4-300k-truth-known",),
            assumption_ids=(
                "analytic-reference-only",
                "deterministic-error-bounds-producer-certified",
                "static-transverse-limit-only",
            ),
            exclusion_reason_codes=("no-target-transverse-source",),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id="uncertainty.uniform-electron-gas-response-deterministic-bounds",
            method_key="uniform-electron-gas-response-deterministic-bound-propagation",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal("0.999999"),
            interval_quantity_ids=("uniform-electron-gas-response-receiver-transverse-current",),
            limitation_codes=(
                "deterministic-reference-no-frequentist-replication",
                "nested-q-action-view-rows-not-independent",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=tuple(
            sorted(
                (
                    FalsifierSpec(
                        falsifier_id="falsifier.uniform-electron-gas-response-action-clock",
                        kind=FalsifierKind.WRONG_ACTION,
                        capability_key=METHOD_CAPABILITY_KEY,
                        description="Reject incorrect action realization, frame or clock order.",
                        decisive_rule="Any required invalid row prevents transverse response/local law qualification and forces hold.",
                        status=ObligationStatus.REQUIRED,
                    ),
                    FalsifierSpec(
                        falsifier_id="falsifier.uniform-electron-gas-response-gauge-normal",
                        kind=FalsifierKind.NEGATIVE_CONTROL,
                        capability_key=METHOD_CAPABILITY_KEY,
                        description="Recover planted normal-cancellation and Ward failures.",
                        decisive_rule="Either failed control prevents gauge-closed local law qualification/receiver admission.",
                        status=ObligationStatus.REQUIRED,
                    ),
                    FalsifierSpec(
                        falsifier_id="falsifier.uniform-electron-gas-response-nonlinear-q-view",
                        kind=FalsifierKind.STRUCTURAL_CONVERGENCE,
                        capability_key=METHOD_CAPABILITY_KEY,
                        description="Recover amplitude, q-limit and view-intersection failures.",
                        decisive_rule="Each planted defect terminates at its frozen rung.",
                        status=ObligationStatus.REQUIRED,
                    ),
                    FalsifierSpec(
                        falsifier_id="falsifier.uniform-electron-gas-response-preservation",
                        kind=FalsifierKind.RECEIVER_GATE,
                        capability_key=METHOD_CAPABILITY_KEY,
                        description="Keep missing physical-sink operands distinct from validity.",
                        decisive_rule="Missing preservation retains local law qualification and forces receiver admission hold.",
                        status=ObligationStatus.REQUIRED,
                    ),
                ),
                key=lambda value: value.falsifier_id,
            )
        ),
        closure=ClosureSpec(
            closure_id="closure.uniform-electron-gas-transverse-screen-conformance",
            recurrence_cell_ids=("cell.uniform-electron-gas-response-rs4-300k-truth-known",),
            exchange_factor_ids=("uniform-electron-gas-response-denominator-rs4-300k-jellium",),
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id="convergence.uniform-electron-gas-response-base-refined",
            required_structure_ids=(
                "exact-view-decision-intersection",
                "fixed-q2-intercept",
                "signed-amplitude-locality",
            ),
            numerical_view_ids=config.views,
            tolerances=(
                NamedDecimal(
                    value_id="tolerance.uniform-electron-gas-response-view-k0",
                    value=config.threshold("view_k0_agreement_relative"),
                    unit="1",
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id="computability.uniform-electron-gas-transverse-screen-conformance",
            envelope_id=ENVELOPE_ID,
            numerical_view_ids=config.views,
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=("preflight.uniform-electron-gas-transverse-screen-conformance-resource-envelope",),
        ),
    )


def uniform_electron_gas_transverse_screen_experiment(system: SystemSpec, config: UniformElectronGasTransverseScreenConfig) -> ExperimentSpec:
    cutoff = InformationCutoff(
        cutoff_id="cutoff.uniform-electron-gas-transverse-screen-pre-action",
        clock_id=CLOCK_ID,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal(config.requested_clock),
    )
    claim = ClaimSpec(
        claim_id="claim.uniform-electron-gas-transverse-screen-conformance-method-conformance",
        world_id=system.world.world_id,
        relation_id=system.relation.relation_id,
        proposition=(
            "The frozen transverse receiver method exactly recovers all nine "
            "truth-known dispositions while keeping closure inputs outside receiver admission."
        ),
        estimand="Exact transverse response/local law qualification/receiver admission class and decisive-falsifier recovery over nine cases.",
        physical_independent_unit_id=system.independent_unit.unit_id,
        requested_rung=None,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        promotion_rule="No promotion; Truth-known conformance is analytic-reference method qualification only.",
        assumption_ids=("analytic-reference-fixtures-exact",),
        numerical_view_ids=config.views,
    )
    controls = tuple(
        ControlSpec(
            control_id=control_id,
            kind=kind,
            capability_key=METHOD_CAPABILITY_KEY,
            target_quantity_ids=targets,
            decisive_rule=rule,
        )
        for control_id, kind, targets, rule in (
            (
                "control.support-matched-comparator",
                ControlKind.BASELINE_COMPARATOR,
                (
                    "uniform-electron-gas-response-receiver-gauge-controls",
                    "uniform-electron-gas-response-receiver-order",
                    "uniform-electron-gas-response-receiver-transverse-current",
                ),
                "Normal cancellation or Ward failure blocks gauge-closed promotion.",
            ),
            (
                "control.negative-action",
                ControlKind.NEGATIVE_ACTION,
                ("uniform-electron-gas-response-receiver-transverse-current",),
                "Both signs, both amplitudes and exact zero are mandatory.",
            ),
            (
                "control.uniform-electron-gas-response-wrong-action-clock",
                ControlKind.WRONG_ACTION,
                ("uniform-electron-gas-response-receiver-transverse-current",),
                "Wrong action, frame, polarization or clock makes the panel incomplete.",
            ),
        )
    )
    return ExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=(claim,),
        assignment=AssignmentSpec(
            assignment_id="assignment.uniform-electron-gas-response-truth-known-fixtures",
            kind=AssignmentKind.SIMULATOR_INTERVENTION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism="Deterministic predeclared analytic reference generator.",
            support_restriction_ids=("chart.uniform-electron-gas-response-u-q-transverse",),
        ),
        measurement_quantity_ids=(
            "uniform-electron-gas-response-receiver-gauge-controls",
            "uniform-electron-gas-response-receiver-order",
            "uniform-electron-gas-response-receiver-transverse-current",
            "uniform-electron-gas-response-sink-order-preservation",
            "uniform-electron-gas-response-truth-oracle",
        ),
        controls=tuple(sorted(controls, key=lambda value: value.control_id)),
        precision_goals=(
            PrecisionGoal(
                goal_id="precision.uniform-electron-gas-response-exact-nine-case-recovery",
                metric_id="exact-case-classification-count",
                target_width=Decimal("0.000001"),
                native_unit="1",
                maximum_independent_units=len(TRUTH_UNIT_IDS),
                stopping_rule="Stop after the frozen nine-case roster; never retune.",
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id="barrier.uniform-electron-gas-response-truth-oracle",
            development_unit_ids=TRUTH_UNIT_IDS,
            evaluation_cohort_id="cohort.uniform-electron-gas-response-truth-oracles",
            evaluation_manifest_sha256=sha256("\n".join(TRUTH_UNIT_IDS).encode()).hexdigest(),
            sealed_outcome_artifact_ids=("artifact.uniform-electron-gas-response-truth-oracles",),
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
    "ENVELOPE_ID",
    "EXPERIMENT_ID",
    "INDEPENDENT_UNIT_ID",
    "POLICY_ID",
    "RELATION_ID",
    "SCOPE_ID",
    "SYSTEM_ID",
    "TRUTH_SEED_IDS",
    "TRUTH_UNIT_IDS",
    "WORLD_ID",
    "uniform_electron_gas_transverse_screen_experiment",
    "uniform_electron_gas_transverse_screen_system",
    'study_budget',
    "task_budget",
]
