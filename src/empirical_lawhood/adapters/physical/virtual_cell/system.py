"Typed 2025 Virtual Cell source world and benchmark experiment.\n\nThe source is a retrospective physical endpoint screen.  The experiment below\nis a newly frozen, non-actuating prediction/evaluation act over still-sealed\ntest responses; it is deliberately non-promotable as biological measurement through controller use\nevidence.  A separate numerical empirical-replay world owns any later replay\ncontroller programme.\n"

from __future__ import annotations

from decimal import Decimal

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
from empirical_lawhood.kernel.quantities import (
    QuantityKind,
    QuantitySpec,
    ResponseDirection,
)
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

from .config import (
    FALSIFIER_CAPABILITY_KEY,
    MODEL_CAPABILITY_KEY,
    SOURCE_CAPABILITY_KEY,
)
from .contracts import ProvenanceBoundVirtualCellPipelineConfig, VirtualCellSourceObject


SYSTEM_ID = "system.virtual-cell-2025-h1-retrospective-benchmark"
WORLD_ID = "world.virtual-cell-2025-h1-retrospective-dataset"
RELATION_ID = "relation.virtual-cell-2025-h1-endpoint-response"
EXPERIMENT_ID = "experiment.virtual-cell-2025-tier-l0-sealed-benchmark"
INDEPENDENT_UNIT_ID = "unit.virtual-cell-2025-observed-batch-ceiling"
CLOCK_ID = "clock.virtual-cell-2025-endpoint-stage"
POLICY_ID = "policy.virtual-cell-2025-nonactuating-benchmark"
SCOPE_ID = "scope.virtual-cell-2025-tier-l0-sealed-benchmark"
ENVELOPE_ID = "compute.virtual-cell-2025-tier-l0-local"
NUMERICAL_VIEW_ID = "view.virtual-cell-2025-tier-l0-computation"

DENOMINATOR_QUANTITY_ID = "vcc-denominator-h1-endpoint-screen"
HISTORY_QUANTITY_ID = "vcc-history-prefix-target-batch"
ACTION_QUANTITY_ID = "vcc-action-nominal-dual-guide-target"
RECEIVER_QUANTITY_ID = "vcc-receiver-endpoint-transcript-counts"
SINK_QUANTITY_ID = "vcc-sink-assay-invalidity"

DEVELOPMENT_UNIT_IDS = (
    "cohort.virtual-cell-2025-training-targets",
    "cohort.virtual-cell-2025-validation-targets",
)
EVALUATION_UNIT_IDS = ("cohort.virtual-cell-2025-test-targets",)


def study_budget() -> ResourceBudget:
    """Conservative aggregate ceiling for all nine Tier-L0 tasks."""

    return ResourceBudget(
        cpu_cores=8,
        memory_bytes=30 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=32 * 60 * 60,
        # Sum of the nine per-step scan ceilings is 111.25 GiB.  The
        # campaign contract is aggregate for scan/output/wall-time (unlike
        # CPU and memory, which are maxima), so retain a small exact margin.
        source_scan_bytes=112 * 1024**3,
        output_bytes=12 * 1024**3,
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
        availability=AvailabilitySpec(
            clock_id=CLOCK_ID,
            phase=phase,
            outcome_access=access,
        ),
        response_direction=direction,
    )


def virtual_cell_2025_system() -> SystemSpec:
    budget = study_budget()
    clock = ClockSpec(
        clock_id=CLOCK_ID,
        label="Preparation, logged CRISPRi action and destructive endpoint assay stage",
        time_unit="endpoint-stage",
        coordinate_frame="source-declared-h1-crispri-endpoint-stage",
        sampling=SamplingSemantics.EVENT_DRIVEN,
        hold=HoldSemantics.SOURCE_DEFINED,
        label_semantics=ClockLabelSemantics.EVENT,
        alignment_tolerance=Decimal("0"),
    )
    quantities = tuple(
        sorted(
            (
                _quantity(
                    ACTION_QUANTITY_ID,
                    "Nominal dual-guide CRISPRi target label",
                    QuantityKind.ACTION,
                    CausalPhase.ACTION_REQUESTED,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="nominal-genetic-action",
                    unit="target-identity",
                    frame="official-2025-target-registry",
                ),
                _quantity(
                    DENOMINATOR_QUANTITY_ID,
                    "Archived H1 endpoint-screen denominator",
                    QuantityKind.DENOMINATOR,
                    CausalPhase.PREPARATION,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="prepared-cell-context",
                    unit="context-identity",
                    frame="h1-dual-guide-crispri-screen",
                ),
                _quantity(
                    HISTORY_QUANTITY_ID,
                    "Outcome-blind target roster, advertised cell count and batch prefix",
                    QuantityKind.HISTORY,
                    CausalPhase.PRE_ACTION,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="retained-prefix-metadata",
                    unit="registry-record",
                    frame="official-2025-split-prefix",
                ),
                _quantity(
                    RECEIVER_QUANTITY_ID,
                    "Endpoint single-cell transcript count vector",
                    QuantityKind.RECEIVER,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.EVALUATION_SEALED,
                    dimension="transcript-count-vector",
                    unit="umi-count",
                    frame="official-18080-gene-order",
                    direction=ResponseDirection.SIGNED_VECTOR,
                ),
                _quantity(
                    SINK_QUANTITY_ID,
                    "Unresolved assay invalidity and physical sink indicator",
                    QuantityKind.SINK,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.EVALUATION_SEALED,
                    dimension="assay-invalidity",
                    unit="indicator",
                    frame="source-validity-ceiling",
                    direction=ResponseDirection.LOWER_IS_BETTER,
                ),
            ),
            key=lambda value: value.quantity_id,
        )
    )
    relation = RelationalIdentity(
        relation_id=RELATION_ID,
        denominator_quantity_ids=(DENOMINATOR_QUANTITY_ID,),
        history_quantity_ids=(HISTORY_QUANTITY_ID,),
        memoryless=False,
        action_quantity_ids=(ACTION_QUANTITY_ID,),
        receiver_quantity_ids=(RECEIVER_QUANTITY_ID,),
        horizon=HorizonSpec(
            horizon_id="horizon.virtual-cell-2025-destructive-endpoint",
            clock_id=CLOCK_ID,
            duration=Decimal("1"),
            time_unit="endpoint-stage",
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
                AuthorityAction.REPOSITORY_IMPLEMENTATION,
            }
        ),
        allowed_source_classes=frozenset(
            {SourceAccessClass.NONE, SourceAccessClass.OFFICIAL_OPEN_PUBLIC}
        ),
        required_gate_ids=(
            "clean-implementation-or-exact-source-closure",
            "external-storage-active",
            "source-custody-byte-closure",
            "test-outcome-lineage-empty",
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
        label="Retrospective 2025 H1 CRISPRi endpoint dataset",
        kind=WorldKind.PHYSICAL_EXPERIMENT,
        represented_physics=(
            "archived-endpoint-transcript-counts",
            "dual-guide-crispri-target-labels",
            "observed-batch-labels",
        ),
        unrepresented_physics=(
            "dose-and-delivery-stage-semantics",
            "physical-preparation-identity",
            "prospective-controller-caused-intervention",
            "time-resolved-dynamics-and-feedback",
        ),
        privileged_truth_quantity_ids=(),
        maximum_evidence=EvidenceCeiling.NON_PROMOTABLE,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_REVEALED,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.OUTCOME_BLIND,
            }
        ),
    )
    envelope = ComputabilityEnvelope(
        envelope_id=ENVELOPE_ID,
        represented_effect_ids=(
            "exact-cell-eval-0-6-6-scoring",
            "sparse-target-batch-summary",
            "tier-l0-low-rank-prediction",
        ),
        unresolved_effect_ids=(),
        required_structure_ids=(
            "bounded-dense-prediction-envelope",
            "evaluator-only-test-open",
            "exact-source-gene-order",
        ),
        computable_structure_ids=(
            "bounded-dense-prediction-envelope",
            "evaluator-only-test-open",
            "exact-source-gene-order",
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
        equations_id="equations.virtual-cell-tier-l0-summary-predictor",
        closure_ids=(
            "closure.fixed-total-log1p",
            "closure.training-only-target-feature-model",
        ),
        boundary_condition_ids=("boundary.official-2025-split-and-gene-order",),
        coordinates=(
            NumericalCoordinateSpec(
                coordinate_id="coordinate.virtual-cell-tier-l0-fidelity",
                kind=NumericalCoordinateKind.SURROGATE_FIDELITY,
                value=Decimal("1"),
                unit="tier-l0-profile",
                refinement_level=0,
            ),
        ),
        solver_id="solver.virtual-cell-tier-l0-sklearn-cell-eval",
        solver_version="1.0.0",
        precision="float64-analysis-float32-prediction",
        device_class="cpu",
        runtime_id="runtime.cpython-3-11-virtual-cell",
        randomness=RandomnessSemantics.DETERMINISTIC,
        observation_operator_id="observer.virtual-cell-official-h5ad",
        computability_envelope_id=ENVELOPE_ID,
    )
    return SystemSpec(
        system_id=SYSTEM_ID,
        label="2025 Virtual Cell retrospective physical-source benchmark",
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=quantities,
        independent_unit=IndependentUnitSpec(
            unit_id=INDEPENDENT_UNIT_ID,
            label=(
                "Observed batch-only ceiling; physical preparation identity is unavailable"
            ),
            grouping_key="observed-batch-id",
        ),
        authority_policy=authority,
        ports=(),
        computability_envelopes=(envelope,),
        numerical_views=(view,),
    )


def _obligations(
    system: SystemSpec, config: ProvenanceBoundVirtualCellPipelineConfig
) -> ScientificObligations:
    return ScientificObligations(
        obligations_id="obligations.virtual-cell-2025-tier-l0-benchmark",
        support=SupportSpec(
            support_id="support.virtual-cell-2025-observed-batch-ceiling",
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=0,
            nested_numerical_view_count=1,
            information_cutoff_id="cutoff.virtual-cell-2025-final-target-release",
            chart_ids=("chart.virtual-cell-2025-nominal-target-identity",),
            denominator_cell_ids=("cell.virtual-cell-2025-h1-endpoint-context",),
            action_bounds=(
                QuantityBound(
                    bound_id="bound.virtual-cell-2025-one-nominal-target",
                    quantity_id=ACTION_QUANTITY_ID,
                    native_unit="target-identity",
                    lower=Decimal("0"),
                    upper=Decimal("1"),
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id="validity.virtual-cell-2025-tier-l0",
            validity_domain_ids=("cell.virtual-cell-2025-h1-endpoint-context",),
            assumption_ids=(
                "official-processed-release-retains-challenge-semantics",
                "target-generalization-is-the-only-estimand",
                "test-target-prefix-is-legal-model-input",
            ),
            exclusion_reason_codes=(
                "action-stage-realization-unavailable",
                "physical-preparation-identity-unavailable",
                "time-resolved-dynamics-unavailable",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id="uncertainty.virtual-cell-2025-observed-batch-ceiling",
            method_key="between-observed-batch-target-bootstrap",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=config.model_selection.confidence_level,
            interval_quantity_ids=(RECEIVER_QUANTITY_ID,),
            limitation_codes=(
                "cells-and-guides-are-nested-views",
                "physical-preparation-count-unknown",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=(
            FalsifierSpec(
                falsifier_id="falsifier.virtual-cell-2025-label-and-gene-permutation",
                kind=FalsifierKind.WRONG_ACTION,
                capability_key=FALSIFIER_CAPABILITY_KEY,
                description="Target-label and gene-label permutation must destroy transferable signal.",
                decisive_rule=(
                    "Failure rejects receiver-admission support but does not replace the "
                    "official-validation competition winner."
                ),
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id="falsifier.virtual-cell-2025-official-baseline",
                kind=FalsifierKind.BASELINE_COMPARATOR,
                capability_key=MODEL_CAPABILITY_KEY,
                description="No-change, weighted common-response, target-feature ridge and reduced-rank target-feature ridge models remain mandatory capacity-matched comparators.",
                decisive_rule="Selection is validation-only under the frozen tournament rule.",
                status=ObligationStatus.REQUIRED,
            ),
            FalsifierSpec(
                falsifier_id="falsifier.virtual-cell-2025-outcome-lineage",
                kind=FalsifierKind.TEMPORAL_SUPPORT,
                capability_key=SOURCE_CAPABILITY_KEY,
                description="Test response and final leaderboard lineage must remain empty at freeze.",
                decisive_rule="Any protected parent makes the prediction commitment ineligible.",
                status=ObligationStatus.REQUIRED,
            ),
        ),
        closure=ClosureSpec(
            closure_id="closure.virtual-cell-2025-prefix-and-summary",
            recurrence_cell_ids=("cell.virtual-cell-2025-h1-endpoint-context",),
            exchange_factor_ids=(DENOMINATOR_QUANTITY_ID,),
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id="convergence.virtual-cell-2025-tier-l0",
            required_structure_ids=(
                "exact-gene-order",
                "grouped-target-cv",
                "prediction-mean-preservation",
            ),
            numerical_view_ids=(NUMERICAL_VIEW_ID,),
            tolerances=(
                NamedDecimal(
                    value_id="tolerance.virtual-cell-prediction-mean",
                    value=config.prediction_compiler.mean_tolerance,
                    unit="log1p-fixed-total",
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id="computability.virtual-cell-2025-tier-l0-local",
            envelope_id=ENVELOPE_ID,
            numerical_view_ids=(NUMERICAL_VIEW_ID,),
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=(
                "evidence.virtual-cell-local-resource-audit-2026-08-08",
            ),
        ),
    )


def virtual_cell_2025_experiment(
    system: SystemSpec,
    config: ProvenanceBoundVirtualCellPipelineConfig,
    *,
    test_source: VirtualCellSourceObject,
) -> ExperimentSpec:
    cutoff = InformationCutoff(
        cutoff_id="cutoff.virtual-cell-2025-final-target-release",
        clock_id=CLOCK_ID,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal("0"),
    )
    claim = ClaimSpec(
        claim_id="claim.virtual-cell-2025-frozen-competition-prediction",
        world_id=system.world.world_id,
        relation_id=system.relation.relation_id,
        proposition=(
            "The validation-selected frozen Tier-L0 method predicts the official 2025 test "
            "responses and receives one exact official score."
        ),
        estimand="Official cell-eval 0.6.6 aggregate score on the 100 held target units.",
        physical_independent_unit_id=system.independent_unit.unit_id,
        requested_rung=None,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        promotion_rule=(
            "Competition prediction only; never promote to a physical response-law or "
            "controller claim."
        ),
        assumption_ids=(
            "historical-cutoff-source-allowlist",
            "official-scorer-0-6-6",
            "test-response-unopened-before-freeze",
        ),
        numerical_view_ids=(NUMERICAL_VIEW_ID,),
    )
    return ExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=(claim,),
        assignment=AssignmentSpec(
            assignment_id="assignment.virtual-cell-2025-logged-crispri",
            kind=AssignmentKind.LOGGED_INTERVENTION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism=(
                "Retrospectively logged nominal dual-guide CRISPRi target; requested, "
                "accepted, applied and realized physical stages are not inferred."
            ),
            support_restriction_ids=("support.official-2025-150-50-100-target-split",),
        ),
        measurement_quantity_ids=(RECEIVER_QUANTITY_ID, SINK_QUANTITY_ID),
        controls=(
            ControlSpec(
                control_id="control.virtual-cell-2025-mandatory-comparator-models",
                kind=ControlKind.BASELINE_COMPARATOR,
                capability_key=MODEL_CAPABILITY_KEY,
                target_quantity_ids=(RECEIVER_QUANTITY_ID,),
                decisive_rule="The no-change, weighted common-response, target-feature ridge and reduced-rank target-feature ridge comparators all participate in the frozen validation tournament.",
            ),
            ControlSpec(
                control_id="control.virtual-cell-2025-non-targeting",
                kind=ControlKind.NEGATIVE_ACTION,
                capability_key=SOURCE_CAPABILITY_KEY,
                target_quantity_ids=(RECEIVER_QUANTITY_ID,),
                decisive_rule="Use exact source non-targeting controls without treating cells as replicates.",
            ),
            ControlSpec(
                control_id="control.virtual-cell-2025-wrong-label",
                kind=ControlKind.WRONG_ACTION,
                capability_key=FALSIFIER_CAPABILITY_KEY,
                target_quantity_ids=(RECEIVER_QUANTITY_ID,),
                decisive_rule="Permuted action identity must not retain the selected transfer signal.",
            ),
        ),
        precision_goals=(
            PrecisionGoal(
                goal_id="precision.virtual-cell-2025-official-score",
                metric_id="official-aggregate-score",
                target_width=config.metric_contract.numerical_tolerance,
                native_unit="fraction",
                maximum_independent_units=100,
                stopping_rule="One frozen evaluation only; never retune after test reveal.",
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id="barrier.virtual-cell-2025-test-response",
            development_unit_ids=DEVELOPMENT_UNIT_IDS,
            evaluation_cohort_id=EVALUATION_UNIT_IDS[0],
            evaluation_manifest_sha256=test_source.fingerprint(),
            sealed_outcome_artifact_ids=(test_source.object_id,),
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
    "DEVELOPMENT_UNIT_IDS",
    "EVALUATION_UNIT_IDS",
    "EXPERIMENT_ID",
    "INDEPENDENT_UNIT_ID",
    "NUMERICAL_VIEW_ID",
    "POLICY_ID",
    "RELATION_ID",
    "SCOPE_ID",
    "SYSTEM_ID",
    "WORLD_ID",
    'study_budget',
    "virtual_cell_2025_experiment",
    "virtual_cell_2025_system",
]
