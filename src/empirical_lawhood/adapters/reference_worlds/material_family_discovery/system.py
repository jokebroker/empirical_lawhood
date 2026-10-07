"Typed retrospective worlds for material-family discovery."

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
from empirical_lawhood.kernel.references import QuantityBound
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

from .contracts import MaterialFamilyConfig, MaterialFamilyDiscoveryAdjudicationConfig, MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION
from .registration import (
    TRUTH_CONTROL_CAPABILITY_KEY,
    WORLD_EVALUATOR_CAPABILITY_KEY,
    policy_capability_key,
)
from empirical_lawhood.adapters.methods.budgeted_first_discovery.contracts import PolicyKind


SYSTEM_ID = f"system.material-family-discovery-historical-material-first-discovery-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}"
WORLD_ID = 'world.material-family-discovery-nims-supercon-220808'
RELATION_ID = 'relation.material-family-discovery-budgeted-first-discovery'
EXPERIMENT_ID = f"experiment.material-family-discovery-evaluation-held-family-first-discovery-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}"
INDEPENDENT_UNIT_ID = 'unit.material-family-discovery-held-material-family-world'
CLOCK_ID = 'clock.material-family-discovery-query-round'
NUMERICAL_VIEW_ID = 'view.material-family-discovery-cpu-policy-computation'
POLICY_ID = f"policy.material-family-discovery-nonactuating-benchmark-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}"
SCOPE_ID = f"scope.material-family-discovery-evaluation-{MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION}"

DENOMINATOR_ID = 'material-family-discovery-denominator-frozen-candidate-universe'
HISTORY_ID = 'material-family-discovery-history-committed-query-prefix'
ACTION_ID = 'material-family-discovery-action-candidate-query-batch'
RECEIVER_ID = 'material-family-discovery-receiver-critical-temperature-observation'
SINK_ID = 'material-family-discovery-sink-invalid-or-unsupported-query'


def study_budget() -> ResourceBudget:
    "Conservative aggregate ceiling for the complete 20-world discovery graph."

    return ResourceBudget(
        cpu_cores=2,
        memory_bytes=8 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=600_000,
        source_scan_bytes=192 * 1024**3,
        output_bytes=16 * 1024**3,
    )


def _quantity(
    *,
    quantity_id: str,
    label: str,
    kind: QuantityKind,
    phase: CausalPhase,
    access: OutcomeAccess,
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
        availability=AvailabilitySpec(
            clock_id=CLOCK_ID,
            phase=phase,
            outcome_access=access,
        ),
        response_direction=direction,
    )


def material_family_system() -> SystemSpec:
    budget = study_budget()
    clock = ClockSpec(
        clock_id=CLOCK_ID,
        label="Atomic committed candidate-query round",
        time_unit="query-round",
        coordinate_frame="frozen-held-family-search-world",
        sampling=SamplingSemantics.EVENT_DRIVEN,
        hold=HoldSemantics.SOURCE_DEFINED,
        label_semantics=ClockLabelSemantics.EVENT,
        alignment_tolerance=Decimal(0),
    )
    quantities = tuple(
        sorted(
            (
                _quantity(
                    quantity_id=ACTION_ID,
                    label="Committed candidate query batch",
                    kind=QuantityKind.ACTION,
                    phase=CausalPhase.ACTION_REQUESTED,
                    access=OutcomeAccess.OUTCOME_BLIND,
                    dimension="material-candidate-identity-batch",
                    unit="candidate-query",
                    frame="nims-canonical-formula-registry",
                ),
                _quantity(
                    quantity_id=DENOMINATOR_ID,
                    label="Frozen held-family candidate universe",
                    kind=QuantityKind.DENOMINATOR,
                    phase=CausalPhase.PREPARATION,
                    access=OutcomeAccess.OUTCOME_BLIND,
                    dimension="prepared-historical-material-search-world",
                    unit="world-identity",
                    frame="nims-supercon-220808",
                ),
                _quantity(
                    quantity_id=HISTORY_ID,
                    label="Prior committed queries and returned labels",
                    kind=QuantityKind.HISTORY,
                    phase=CausalPhase.PRE_ACTION,
                    access=OutcomeAccess.EVALUATION_SEALED,
                    dimension="causal-query-prefix",
                    unit="query-observation-record",
                    frame="policy-private-history",
                ),
                _quantity(
                    quantity_id=RECEIVER_ID,
                    label="Historical critical-temperature receiver observation",
                    kind=QuantityKind.RECEIVER,
                    phase=CausalPhase.RECEIVER,
                    access=OutcomeAccess.EVALUATION_SEALED,
                    dimension="thermodynamic-temperature",
                    unit="K",
                    frame="nims-reported-critical-temperature",
                    direction=ResponseDirection.HIGHER_IS_BETTER,
                ),
                _quantity(
                    quantity_id=SINK_ID,
                    label="Invalid, unsupported or unlabelled query state",
                    kind=QuantityKind.SINK,
                    phase=CausalPhase.RECEIVER,
                    access=OutcomeAccess.EVALUATION_SEALED,
                    dimension="query-validity-state",
                    unit="indicator",
                    frame='material-family-discovery-receiver-validity',
                    direction=ResponseDirection.LOWER_IS_BETTER,
                ),
            ),
            key=lambda value: value.quantity_id,
        )
    )
    relation = RelationalIdentity(
        relation_id=RELATION_ID,
        denominator_quantity_ids=(DENOMINATOR_ID,),
        history_quantity_ids=(HISTORY_ID,),
        memoryless=False,
        action_quantity_ids=(ACTION_ID,),
        receiver_quantity_ids=(RECEIVER_ID,),
        horizon=HorizonSpec(
            horizon_id='horizon.material-family-discovery-100-query-budget',
            clock_id=CLOCK_ID,
            duration=Decimal(10),
            time_unit="query-round",
        ),
    )
    authority = AuthorityPolicy(
        policy_id=POLICY_ID,
        delegator_id="human.project-owner",
        delegate_id="role.outcome-blind-scientific-approver",
        scope_ids=(SCOPE_ID,),
        allowed_world_kinds=frozenset({WorldKind.PHYSICAL_EXPERIMENT}),
        allowed_actions=frozenset(
            {
                AuthorityAction.DATASET_BINDING,
                AuthorityAction.DATASET_REGISTRATION,
                AuthorityAction.DATASET_TRANSFORMATION,
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.PUBLIC_SOURCE_ACQUISITION,
                AuthorityAction.REFERENCE_WORLD_EXECUTION,
                AuthorityAction.REPOSITORY_IMPLEMENTATION,
            }
        ),
        allowed_source_classes=frozenset(
            {SourceAccessClass.NONE, SourceAccessClass.OFFICIAL_OPEN_PUBLIC}
        ),
        required_gate_ids=(
            "external-storage-active",
            "method-conformance-complete",
            "source-custody-byte-closure",
            "world-policy-outcome-isolation",
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
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    world = WorldSpec(
        world_id=WORLD_ID,
        label="Retrospective NIMS SuperCon v220808 material-family search world",
        kind=WorldKind.PHYSICAL_EXPERIMENT,
        represented_physics=(
            "historical-reported-composition",
            "historical-reported-critical-temperature",
            "nims-reported-structure-family-label",
        ),
        unrepresented_physics=(
            "ambient-pressure-ambient-temperature-superconductivity",
            "prospective-material-synthesis",
            "synthesizability-and-phase-stability",
            "transport-measurement-conditions",
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
    envelope = ComputabilityEnvelope(
        envelope_id='compute.material-family-discovery-local-cpu',
        represented_effect_ids=(
            "bounded-discrete-botorch-ucb",
            "deterministic-material-family-world-construction",
            "family-level-paired-bootstrap",
            "local-law-boundary-selection",
        ),
        unresolved_effect_ids=("exact-pekala-roost-reproduction",),
        required_structure_ids=(
            "causal-batch-commitment",
            "matched-candidate-universe",
            "policy-private-query-prefix",
        ),
        computable_structure_ids=(
            "causal-batch-commitment",
            "matched-candidate-universe",
            "policy-private-query-prefix",
        ),
        max_cpu_cores=budget.cpu_cores,
        max_memory_bytes=budget.memory_bytes,
        max_gpu_devices=0,
        max_wall_time_seconds=budget.wall_time_seconds,
        max_output_bytes=budget.output_bytes,
        worst_case_latency_seconds=Decimal(budget.wall_time_seconds),
        deadline_seconds=Decimal(budget.wall_time_seconds),
    )
    view = NumericalViewSpec(
        view_id=NUMERICAL_VIEW_ID,
        world_id=WORLD_ID,
        physical_preparation_id=INDEPENDENT_UNIT_ID,
        equations_id='equations.material-family-discovery-policy-specific-statistical-models',
        closure_ids=('closure.material-family-discovery-frozen-candidate-universe',),
        boundary_condition_ids=('boundary.material-family-discovery-held-family-query-budget',),
        coordinates=(
            NumericalCoordinateSpec(
                coordinate_id='coordinate.material-family-discovery-query-round',
                kind=NumericalCoordinateKind.TIMESTEP,
                value=Decimal(1),
                unit="query-round",
                refinement_level=0,
            ),
        ),
        solver_id='solver.material-family-discovery-cpu-policy-roster',
        solver_version="1.0.0",
        precision="float64-model-float32-feature-wire",
        device_class="cpu",
        runtime_id="cpython-3.11",
        randomness=RandomnessSemantics.DETERMINISTIC,
        observation_operator_id="observer.nims-supercon-historical-query-replay",
        computability_envelope_id=envelope.envelope_id,
    )
    return SystemSpec(
        system_id=SYSTEM_ID,
        label="SDCB superconductor held-family first-discovery benchmark",
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=quantities,
        independent_unit=IndependentUnitSpec(
            unit_id=INDEPENDENT_UNIT_ID,
            label="One independently held material-family search world",
            grouping_key="target-family-id",
        ),
        authority_policy=authority,
        ports=(),
        computability_envelopes=(envelope,),
        numerical_views=(view,),
    )


def _obligations(
    *,
    system: SystemSpec,
    family_config: MaterialFamilyConfig,
    adjudication_config: MaterialFamilyDiscoveryAdjudicationConfig,
) -> ScientificObligations:
    return ScientificObligations(
        obligations_id='obligations.material-family-discovery-evaluation',
        support=SupportSpec(
            support_id='support.material-family-discovery-held-family-worlds',
            relation_id=system.relation.relation_id,
            independent_unit_id=INDEPENDENT_UNIT_ID,
            physical_unit_count=len(family_config.evaluation_family_ids),
            nested_numerical_view_count=1,
            information_cutoff_id='cutoff.material-family-discovery-pre-query-outcome',
            chart_ids=('chart.material-family-discovery-canonical-candidate-query',),
            denominator_cell_ids=('cell.material-family-discovery-frozen-family-search-world',),
            action_bounds=(
                QuantityBound(
                    bound_id='bound.material-family-discovery-query-budget',
                    quantity_id=ACTION_ID,
                    native_unit="candidate-query",
                    lower=Decimal(0),
                    upper=Decimal(adjudication_config.query_budget),
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id='validity.material-family-discovery-historical-material',
            validity_domain_ids=("nims-supercon-220808",),
            assumption_ids=(
                "assumption.family-label-is-benchmark-stratum-not-physical-law",
                "assumption.historical-tc-is-query-receiver-only",
                "assumption.no-absence-as-negative",
            ),
            exclusion_reason_codes=("ambient-condition-and-synthesis-validity-not-established",),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id='uncertainty.material-family-discovery-paired-family-bootstrap',
            method_key=adjudication_config.interval_method_id,
            independent_unit_id=INDEPENDENT_UNIT_ID,
            confidence_level=adjudication_config.confidence_level,
            interval_quantity_ids=(RECEIVER_ID,),
            limitation_codes=(
                "finite-family-roster",
                "historical-material-evidence-only",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=tuple(
            sorted(
                (
                    FalsifierSpec(
                        falsifier_id='falsifier.material-family-discovery-causal-query-order',
                        kind=FalsifierKind.WRONG_ACTION,
                        capability_key=WORLD_EVALUATOR_CAPABILITY_KEY,
                        description="Reject outcome access before immutable query commitment.",
                        decisive_rule="Any query response preceding its exact decision artifact fails SC.",
                        status=ObligationStatus.REQUIRED,
                    ),
                    FalsifierSpec(
                        falsifier_id='falsifier.material-family-discovery-family-duplicate-leakage',
                        kind=FalsifierKind.STRUCTURAL_CONVERGENCE,
                        capability_key=policy_capability_key(PolicyKind.LOCAL_LAW_BOUNDARY),
                        description="Reject target-family or duplicate identity in policy-visible bytes.",
                        decisive_rule="Any target-family identity in selector inputs fails SC.",
                        status=ObligationStatus.REQUIRED,
                    ),
                    FalsifierSpec(
                        falsifier_id='falsifier.material-family-discovery-truth-control-hold',
                        kind=FalsifierKind.NEGATIVE_CONTROL,
                        capability_key=TRUTH_CONTROL_CAPABILITY_KEY,
                        description="Require HOLD on empty and exhausted truth-known support.",
                        decisive_rule="Any false admission or missed mandatory HOLD blocks B2.",
                        status=ObligationStatus.REQUIRED,
                    ),
                ),
                key=lambda value: value.falsifier_id,
            )
        ),
        closure=ClosureSpec(
            closure_id='closure.material-family-discovery-query-history',
            recurrence_cell_ids=('cell.material-family-discovery-frozen-family-search-world',),
            exchange_factor_ids=(DENOMINATOR_ID,),
            retained_history_ids=(HISTORY_ID,),
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id='convergence.material-family-discovery-matched-policy-world',
            required_structure_ids=(
                "causal-batch-commitment",
                "family-independent-uncertainty",
                "matched-candidate-universe",
            ),
            numerical_view_ids=(NUMERICAL_VIEW_ID,),
            tolerances=(),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id='computability.material-family-discovery-local-cpu',
            envelope_id=system.computability_envelopes[0].envelope_id,
            numerical_view_ids=(NUMERICAL_VIEW_ID,),
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=('canary.material-family-discovery-botorch-cpu',),
        ),
    )


def material_family_experiment(
    *,
    system: SystemSpec,
    family_config: MaterialFamilyConfig,
    adjudication_config: MaterialFamilyDiscoveryAdjudicationConfig,
) -> ExperimentSpec:
    cutoff = InformationCutoff(
        cutoff_id='cutoff.material-family-discovery-pre-query-outcome',
        clock_id=CLOCK_ID,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal(0),
    )
    evaluation_ids = family_config.evaluation_family_ids
    return ExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=INDEPENDENT_UNIT_ID,
        claims=(
            ClaimSpec(
                claim_id='claim.material-family-discovery-evaluation-first-discovery-advantage',
                world_id=system.world.world_id,
                relation_id=system.relation.relation_id,
                proposition=(
                    "Across the frozen held-family worlds, the local-law explorer has "
                    "a predeclared restricted-query or query/false-promotion Pareto "
                    "advantage over pinned constrained discrete BoTorch UCB."
                ),
                estimand=(
                    "Paired family-level restricted mean candidate queries to first "
                    "discovery and false-promotion count at a 100-query budget."
                ),
                physical_independent_unit_id=INDEPENDENT_UNIT_ID,
                requested_rung=None,
                evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                promotion_rule=(
                    'Comparison only under the frozen Bayesian-optimization superiority or Pareto rule plus zero executed truth-control false admissions. No measurement-through-controller-use promotion.'
                ),
                assumption_ids=(
                    "frozen-family-ontology",
                    "matched-policy-information-and-budget",
                ),
                numerical_view_ids=(NUMERICAL_VIEW_ID,),
            ),
        ),
        assignment=AssignmentSpec(
            assignment_id='assignment.material-family-discovery-historical-query-replay',
            kind=AssignmentKind.LOGGED_INTERVENTION,
            independent_unit_id=INDEPENDENT_UNIT_ID,
            action_quantity_ids=(ACTION_ID,),
            mechanism=(
                "A policy commits candidate identities before the bounded historical "
                "receiver returns only those candidates' archived observations."
            ),
            support_restriction_ids=(
                "restriction.frozen-4000-candidate-pool",
                "restriction.no-unlabelled-as-negative",
            ),
        ),
        measurement_quantity_ids=(RECEIVER_ID, SINK_ID),
        controls=(
            ControlSpec(
                control_id='control.material-family-discovery-primary-botorch',
                kind=ControlKind.BASELINE_COMPARATOR,
                capability_key=policy_capability_key(PolicyKind.BOTORCH_DISCRETE_UCB),
                target_quantity_ids=(RECEIVER_ID, SINK_ID),
                decisive_rule="The frozen B2 rule is evaluated directly against pinned BoTorch.",
            ),
            ControlSpec(
                control_id='control.material-family-discovery-truth-known-hold',
                kind=ControlKind.NEGATIVE_ACTION,
                capability_key=TRUTH_CONTROL_CAPABILITY_KEY,
                target_quantity_ids=(SINK_ID,),
                decisive_rule="Every empty/exhausted control must HOLD with zero admissions.",
            ),
        ),
        precision_goals=(
            PrecisionGoal(
                goal_id='precision.material-family-discovery-frozen-family-roster',
                metric_id="paired-restricted-query-advantage",
                target_width=Decimal(20),
                native_unit="candidate-query",
                maximum_independent_units=len(evaluation_ids),
                stopping_rule=(
                    "Use every frozen evaluation family; never stop or retune from interim outcomes."
                ),
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id='reveal.material-family-discovery-evaluation',
            development_unit_ids=family_config.development_family_ids,
            evaluation_cohort_id='cohort.material-family-discovery-evaluation-families',
            evaluation_manifest_sha256=sha256("\n".join(evaluation_ids).encode()).hexdigest(),
            sealed_outcome_artifact_ids=('artifact.material-family-discovery-evaluation-world-truth',),
            evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            fresh_evidence=True,
        ),
        obligations=_obligations(
            system=system,
            family_config=family_config,
            adjudication_config=adjudication_config,
        ),
        design_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
    )


__all__ = [
    "EXPERIMENT_ID",
    "INDEPENDENT_UNIT_ID",
    "NUMERICAL_VIEW_ID",
    "SCOPE_ID",
    "WORLD_ID",
    'study_budget',
    'material_family_experiment',
    'material_family_system',
]
