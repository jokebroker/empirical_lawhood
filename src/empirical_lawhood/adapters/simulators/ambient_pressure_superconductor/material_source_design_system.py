'ambient pressure superconductor material source design source/design evidence world and experiment specification.'

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

from .material_source_design_contracts import MaterialSourceDesignConfig, MATERIAL_SOURCE_DESIGN_DESIGN_CAPABILITY_KEY, MATERIAL_SOURCE_DESIGN_SOURCE_CAPABILITY_KEY
from .material_source_design_design import build_material_roster, build_science_design


SYSTEM_ID = 'system.ambient-pressure-superconductor-material-source-design-design-basis'
WORLD_ID = 'world.ambient-pressure-superconductor-material-source-design-outcome-blind-design'
RELATION_ID = 'relation.ambient-pressure-superconductor-material-source-design-source-design-freeze'
EXPERIMENT_ID = 'experiment.ambient-pressure-superconductor-material-source-design-design-basis'
INDEPENDENT_UNIT_ID = 'unit.ambient-pressure-superconductor-material-source-design-one-design-freeze-act'
CLOCK_ID = 'clock.ambient-pressure-superconductor-material-source-design-source-design-stage'
POLICY_ID = 'policy.ambient-pressure-superconductor-material-source-design-public-local-nonactuating'
SCOPE_ID = 'scope.ambient-pressure-superconductor-material-source-design-source-and-design'
ENVELOPE_ID = 'compute.ambient-pressure-superconductor-material-source-design-source-design'
UNIT_IDS = ('unit-instance.ambient-pressure-superconductor-material-source-design-design-freeze-001',)


def task_budget(config: MaterialSourceDesignConfig) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=int(config.resource("cpu_cores")),
        memory_bytes=int(config.resource("memory_bytes")),
        gpu_devices=int(config.resource("gpu_devices")),
        wall_time_seconds=int(config.resource("wall_time_seconds")),
        source_scan_bytes=int(config.resource("scratch_bytes")),
        output_bytes=int(config.resource("output_bytes")),
    )


def study_budget(config: MaterialSourceDesignConfig) -> ResourceBudget:
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


def material_source_design_system(config: MaterialSourceDesignConfig) -> SystemSpec:
    budget = study_budget(config)
    science = build_science_design()
    clock = ClockSpec(
        clock_id=CLOCK_ID,
        label='Public-source qualification through material source design design-basis freeze',
        time_unit="stage",
        coordinate_frame="source-qualified-design-frozen-evaluated",
        sampling=SamplingSemantics.EVENT_DRIVEN,
        hold=HoldSemantics.SOURCE_DEFINED,
        label_semantics=ClockLabelSemantics.EVENT,
        alignment_tolerance=Decimal("0"),
    )
    quantities = tuple(
        sorted(
            (
                _quantity(
                    'ambient-pressure-superconductor-material-source-design-action-finite-chart-freeze',
                    "Outcome-blind finite candidate-and-route chart freeze",
                    QuantityKind.ACTION,
                    CausalPhase.ACTION_REQUESTED,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="nonactuating-design-action",
                    unit="action-count",
                    frame="canonical-material-family-chart",
                ),
                _quantity(
                    'ambient-pressure-superconductor-material-source-design-denominator-public-source-roster',
                    "Exact public source and pseudopotential roster",
                    QuantityKind.DENOMINATOR,
                    CausalPhase.PREPARATION,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="source-custody-identity",
                    unit="source-count",
                    frame="semi-os-external-content-addressed",
                ),
                _quantity(
                    'ambient-pressure-superconductor-material-source-design-history-excluded-solver-control-qualified-route',
                    'Retained excluded solver control corrective solver control route qualification and no-target-contact history',
                    QuantityKind.HISTORY,
                    CausalPhase.PRE_ACTION,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="method-provenance-history",
                    unit="1",
                    frame='excluded-solver-control-corrective-solver-control-to-material-source-design-cutoff',
                ),
                _quantity(
                    'ambient-pressure-superconductor-material-source-design-receiver-design-completeness',
                    "Source-qualified and internally closed design-basis disposition",
                    QuantityKind.RECEIVER,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.EVALUATOR_REVEAL,
                    dimension="design-completeness",
                    unit="1",
                    frame='material-source-design-noncompensating-freeze',
                    direction=ResponseDirection.HIGHER_IS_BETTER,
                ),
                _quantity(
                    'ambient-pressure-superconductor-material-source-design-sink-source-scan-effort',
                    "Bounded physical source verification effort",
                    QuantityKind.SINK,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.EVALUATOR_REVEAL,
                    dimension="computational-effort",
                    unit="byte",
                    frame="external-read-only-source-scan",
                    direction=ResponseDirection.LOWER_IS_BETTER,
                ),
                _quantity(
                    'ambient-pressure-superconductor-material-source-design-truth-target-contact-count',
                    'Privileged audit of development atlas/sealed prospective target outcome contact',
                    QuantityKind.OBSERVATION,
                    CausalPhase.POST_OUTCOME,
                    OutcomeAccess.PRIVILEGED_TRUTH,
                    dimension="target-contact-count",
                    unit="1",
                    frame="protected-outcome-audit",
                ),
            ),
            key=lambda value: value.quantity_id,
        )
    )
    relation = RelationalIdentity(
        relation_id=RELATION_ID,
        denominator_quantity_ids=('ambient-pressure-superconductor-material-source-design-denominator-public-source-roster',),
        history_quantity_ids=('ambient-pressure-superconductor-material-source-design-history-excluded-solver-control-qualified-route',),
        memoryless=False,
        action_quantity_ids=('ambient-pressure-superconductor-material-source-design-action-finite-chart-freeze',),
        receiver_quantity_ids=('ambient-pressure-superconductor-material-source-design-receiver-design-completeness',),
        horizon=HorizonSpec(
            horizon_id='horizon.ambient-pressure-superconductor-material-source-design-one-freeze-act',
            clock_id=CLOCK_ID,
            duration=Decimal("3"),
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
            "target-contact-count-zero",
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
        label='ambient pressure superconductor outcome-blind public-source and design-freeze world',
        kind=WorldKind.NUMERICAL_SIMULATOR,
        represented_physics=(
            "finite-material-and-route-chart-identity",
            "solver-view-and-gate-design-semantics",
            "source-custody-and-method-reference-reproduction",
        ),
        unrepresented_physics=(
            "candidate-electron-phonon-outcomes",
            "candidate-transverse-response",
            "physical-material-realization",
            "room-temperature-superconductivity",
        ),
        privileged_truth_quantity_ids=('ambient-pressure-superconductor-material-source-design-truth-target-contact-count',),
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
        represented_effect_ids=(
            "canonical-design-component-construction",
            "physical-source-byte-verification",
            "truth-world-hash-locking",
        ),
        unresolved_effect_ids=('material-control-transport-nomination-material-solver-environment-closure',),
        required_structure_ids=tuple(
            sorted(
                (
                    "exact-source-locks",
                    "finite-family-partition",
                    "six-truth-world-locks",
                    "three-matched-policy-locks",
                    "sixteen-gate-lowering",
                )
            )
        ),
        computable_structure_ids=tuple(
            sorted(
                (
                    "exact-source-locks",
                    "finite-family-partition",
                    "six-truth-world-locks",
                    "three-matched-policy-locks",
                    "sixteen-gate-lowering",
                )
            )
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
            view_id=view.view_id,
            world_id=WORLD_ID,
            physical_preparation_id=INDEPENDENT_UNIT_ID,
            equations_id='equations.conventional-electron-phonon-design',
            closure_ids=view.model_closure_ids,
            boundary_condition_ids=("boundary.periodic-ordered-crystal",),
            coordinates=(
                NumericalCoordinateSpec(
                    coordinate_id=f"coordinate.{view.view_id.removeprefix('view.')}.kmesh",
                    kind=NumericalCoordinateKind.SOLVER_REFINEMENT,
                    value=Decimal(view.k_mesh[0]),
                    unit="points-per-reciprocal-axis",
                    refinement_level=index,
                ),
            ),
            solver_id=view.solver_id,
            solver_version=view.solver_version,
            precision=view.precision,
            device_class="cpu",
            runtime_id='runtime.ambient-pressure-superconductor-material-source-design-design-only',
            randomness=RandomnessSemantics.DETERMINISTIC,
            observation_operator_id='observer.ambient-pressure-superconductor-material-source-design-design-contract',
            computability_envelope_id=ENVELOPE_ID,
        )
        for index, view in enumerate(science.solver_views)
    )
    return SystemSpec(
        system_id=SYSTEM_ID,
        label='ambient pressure superconductor material source design source qualification and design-basis freeze',
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=quantities,
        independent_unit=IndependentUnitSpec(
            unit_id=INDEPENDENT_UNIT_ID,
            label='One complete source-qualified material source design design-freeze act',
            grouping_key="design-freeze-id",
        ),
        authority_policy=authority,
        ports=(),
        computability_envelopes=(envelope,),
        numerical_views=views,
    )


def _obligations(system: SystemSpec, config: MaterialSourceDesignConfig) -> ScientificObligations:
    roster = build_material_roster()
    return ScientificObligations(
        obligations_id='obligations.ambient-pressure-superconductor-material-source-design-design-basis',
        support=SupportSpec(
            support_id='support.ambient-pressure-superconductor-material-source-design-finite-design',
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=1,
            nested_numerical_view_count=len(system.numerical_views),
            information_cutoff_id='cutoff.ambient-pressure-superconductor-material-source-design-before-target-contact',
            chart_ids=('chart.ambient-pressure-superconductor-material-source-design-finite-material-actions',),
            denominator_cell_ids=('cell.ambient-pressure-superconductor-material-source-design-public-design',),
            action_bounds=(
                QuantityBound(
                    bound_id='bound.ambient-pressure-superconductor-material-source-design-action-count',
                    quantity_id='ambient-pressure-superconductor-material-source-design-action-finite-chart-freeze',
                    native_unit="action-count",
                    lower=Decimal("0"),
                    upper=Decimal(len(roster.actions)),
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id='validity.ambient-pressure-superconductor-material-source-design-outcome-blind-design',
            validity_domain_ids=('cell.ambient-pressure-superconductor-material-source-design-public-design',),
            assumption_ids=(
                'excluded-solver-control-corrective-solver-control-local-scf-route-only',
                "conventional-electron-phonon-lane-only",
                "no-candidate-target-values-used",
            ),
            exclusion_reason_codes=(
                'material-control-material-environment-not-yet-closed',
                'no-material-outcome-at-material-source-design',
                "no-physical-realization-claim",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id='uncertainty.ambient-pressure-superconductor-material-source-design-canonical-freeze',
            method_key="exact-canonical-fingerprint-and-source-byte-verification",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal("0.999999"),
            interval_quantity_ids=('ambient-pressure-superconductor-material-source-design-receiver-design-completeness',),
            limitation_codes=('scientific-constants-remain-calibration-derived-at-material-source-design',),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=(
            FalsifierSpec(
                falsifier_id='falsifier.ambient-pressure-superconductor-material-source-design-source-custody',
                kind=FalsifierKind.NEGATIVE_CONTROL,
                capability_key=MATERIAL_SOURCE_DESIGN_SOURCE_CAPABILITY_KEY,
                description="Reject any changed source, archive member, licence or pseudo lock.",
                decisive_rule="Any exact custody mismatch stops design freeze.",
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id='falsifier.ambient-pressure-superconductor-material-source-design-target-contact',
                kind=FalsifierKind.WRONG_ACTION,
                capability_key=MATERIAL_SOURCE_DESIGN_DESIGN_CAPABILITY_KEY,
                description='Reject any development atlas/sealed prospective target-value use during roster or design construction.',
                decisive_rule='Any target contact or ranking use invalidates material source design.',
                status=ObligationStatus.REQUIRED,
            ),
        ),
        closure=ClosureSpec(
            closure_id='closure.ambient-pressure-superconductor-material-source-design-source-roster-design',
            recurrence_cell_ids=('cell.ambient-pressure-superconductor-material-source-design-public-design',),
            exchange_factor_ids=('ambient-pressure-superconductor-material-source-design-denominator-public-source-roster',),
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id='convergence.ambient-pressure-superconductor-material-source-design-exact-design-components',
            required_structure_ids=tuple(
                sorted(
                    (
                        "source-qualification-fingerprint",
                        "material-roster-fingerprint",
                        "exploration-design-fingerprint",
                        "science-design-fingerprint",
                    )
                )
            ),
            numerical_view_ids=tuple(view.view_id for view in system.numerical_views),
            tolerances=(
                NamedDecimal(
                    value_id='tolerance.ambient-pressure-superconductor-material-source-design-target-contact-count',
                    value=Decimal("0"),
                    unit="1",
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id='computability.ambient-pressure-superconductor-material-source-design-local-source-design',
            envelope_id=ENVELOPE_ID,
            numerical_view_ids=tuple(view.view_id for view in system.numerical_views),
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=('result.ambient-pressure-superconductor-excluded-solver-control-corrective-solver-control-solver-smoke',),
        ),
    )


def material_source_design_experiment(system: SystemSpec, config: MaterialSourceDesignConfig) -> ExperimentSpec:
    cutoff = InformationCutoff(
        cutoff_id='cutoff.ambient-pressure-superconductor-material-source-design-before-target-contact',
        clock_id=CLOCK_ID,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal("0"),
    )
    claim = ClaimSpec(
        claim_id='claim.ambient-pressure-superconductor-material-source-design-source-qualified-design-frozen',
        world_id=system.world.world_id,
        relation_id=system.relation.relation_id,
        proposition=(
            'The exact public source roster qualifies and the finite outcome-blind ambient pressure superconductor '
            'design basis closes before any development atlas/sealed prospective target outcome is contacted.'
        ),
        estimand="Exact source-custody and design-component closure with target contact count zero.",
        physical_independent_unit_id=system.independent_unit.unit_id,
        requested_rung=None,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        promotion_rule="At most source-design/method readiness; no material or superconductivity claim.",
        assumption_ids=("public-source-roster-exact", "target-contact-count-zero"),
        numerical_view_ids=tuple(view.view_id for view in system.numerical_views),
    )
    controls = (
        ControlSpec(
            control_id='control.ambient-pressure-superconductor-material-source-design-exact-source-locks',
            kind=ControlKind.BASELINE_COMPARATOR,
            capability_key=MATERIAL_SOURCE_DESIGN_SOURCE_CAPABILITY_KEY,
            target_quantity_ids=('ambient-pressure-superconductor-material-source-design-receiver-design-completeness',),
            decisive_rule="Every physical source and nested member must match its exact lock.",
        ),
        ControlSpec(
            control_id='control.ambient-pressure-superconductor-material-source-design-zero-target-contact',
            kind=ControlKind.NEGATIVE_ACTION,
            capability_key=MATERIAL_SOURCE_DESIGN_DESIGN_CAPABILITY_KEY,
            target_quantity_ids=('ambient-pressure-superconductor-material-source-design-truth-target-contact-count',),
            decisive_rule='development atlas/sealed prospective target values may not construct or rank the design.',
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
            assignment_id='assignment.ambient-pressure-superconductor-material-source-design-one-design-freeze',
            kind=AssignmentKind.SIMULATOR_INTERVENTION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism="Deterministic source verification and canonical design-basis construction.",
            support_restriction_ids=('chart.ambient-pressure-superconductor-material-source-design-finite-material-actions',),
        ),
        measurement_quantity_ids=(
            'ambient-pressure-superconductor-material-source-design-receiver-design-completeness',
            'ambient-pressure-superconductor-material-source-design-sink-source-scan-effort',
            'ambient-pressure-superconductor-material-source-design-truth-target-contact-count',
        ),
        controls=controls,
        precision_goals=(
            PrecisionGoal(
                goal_id='precision.ambient-pressure-superconductor-material-source-design-exact-canonical-closure',
                metric_id="component-fingerprint-mismatch-count",
                target_width=Decimal("0.000001"),
                native_unit="1",
                maximum_independent_units=1,
                stopping_rule="Stop after one exact frozen act; never repair after target contact.",
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id='barrier.ambient-pressure-superconductor-material-source-design-design-freeze',
            development_unit_ids=UNIT_IDS,
            evaluation_cohort_id='cohort.ambient-pressure-superconductor-material-source-design-design',
            evaluation_manifest_sha256=sha256("\n".join(UNIT_IDS).encode()).hexdigest(),
            sealed_outcome_artifact_ids=('artifact.ambient-pressure-superconductor-material-source-design-design-basis-freeze',),
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
    "UNIT_IDS",
    "WORLD_ID",
    'material_source_design_experiment',
    'material_source_design_system',
    'study_budget',
    "task_budget",
]
