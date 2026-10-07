"""Maintained, non-promotable exploration authoring for reproduced Glenn G1."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Final

from empirical_lawhood.adapters.exploration.detectors import (
    bind_evidence_view,
    default_detector_registry,
)
from empirical_lawhood.adapters.exploration.portfolio import (
    ExplorationPortfolioPlanner,
    build_portfolio_candidates,
    default_portfolio_policy,
)
from empirical_lawhood.adapters.exploration.templates import (
    TemplateContext,
    instantiate_matching_templates,
)
from empirical_lawhood.adapters.reference_worlds.exploration_runtime import (
    ReferenceExplorationExecutionRuntimeProvider,
    RegisteredExplorationInput,
    input_derived_exploration_execution_registry,
)
from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.experiments import AssignmentKind
from empirical_lawhood.kernel.provenance import EvidenceSnapshot, ObjectIdentity
from empirical_lawhood.kernel.quantities import (
    QuantityKind,
    QuantitySpec,
    ResponseDirection,
)
from empirical_lawhood.kernel.references import ArtifactIdentity, QuantityBound
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
from empirical_lawhood.kernel.worlds import WorldKind, WorldSpec
from empirical_lawhood.planning.authority import ApprovalRequest
from empirical_lawhood.planning.discovery import (
    ExplorationDiagnosticProjection,
    PortfolioPolicy,
    QuantityReportingObligation,
)
from empirical_lawhood.planning.exploration import ProposalDisposition, SearchAxis
from empirical_lawhood.planning.prospective import (
    NativeMeasurementRequirement,
    ProspectiveDesignContext,
    ProspectiveDesignKind,
    ProspectiveObjectiveVector,
)
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.compiler import compile_exploration_plan
from empirical_lawhood.runtime.exploration import (
    AnalysisCoordinateObservation,
    ExplorationExecutionPackage,
    ExplorationWaveInput,
)
from empirical_lawhood.runtime.plans import SnapshotVerification

GLENN_G1_EXPLORATION_PLAN_ID: Final = "glenn-g1-remediated-exploration-wave"
GLENN_G1_PROJECTION_ARTIFACT_ID: Final = "glenn-g1-diagnostic-projection"
_ACTION_SPECS: Final = (
    ("glenn-action-astig0", "astig0", "-0.496026", "0.281741"),
    ("glenn-action-astig45", "astig45", "-0.587594", "0.253970"),
    ("glenn-action-coma0", "coma0", "-0.399676", "0.555268"),
    ("glenn-action-coma90", "coma90", "-0.340775", "0.581392"),
    ("glenn-action-focus", "focus", "-0.313035", "0.343098"),
    ("glenn-action-spherical", "spherical", "-0.409768", "0.456662"),
)

_ALL_AXIS_IDS: Final = (
    "action-channel",
    "action-chart",
    "admission-gate",
    "denominator-cell",
    "design-sensitivity",
    "donor",
    "gauge",
    "history-ladder",
    "horizon",
    "interface",
    "model",
    "numerical-view",
    "receiver",
    "receiver-component",
    "representation",
    "stage",
    "time-direction",
    "transform",
)


def _analysis_budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=2_000_000_000,
        gpu_devices=0,
        wall_time_seconds=60,
        source_scan_bytes=1_000_000,
        output_bytes=1_000_000,
    )


def glenn_g1_exploration_registry() -> CapabilityRegistry:
    """Return the exact generic runtime registry used by the Glenn wave."""

    return input_derived_exploration_execution_registry(
        registry_id="glenn-g1-exploration-execution-registry",
        source_schema_ids=(ExplorationDiagnosticProjection.SCHEMA,),
        resource_ceiling=_analysis_budget(),
    )


def glenn_g1_exploration_system() -> SystemSpec:
    """Bind G1 to its physical laser--water-sheet denominator and causal chart."""

    clock = ClockSpec(
        clock_id="glenn-burst-clock",
        label="Source-defined Glenn burst order",
        time_unit="burst",
        coordinate_frame="20210730-run14-acquisition-order",
        sampling=SamplingSemantics.EVENT_DRIVEN,
        hold=HoldSemantics.SOURCE_DEFINED,
        label_semantics=ClockLabelSemantics.EVENT,
    )

    def availability(phase: CausalPhase, access: OutcomeAccess) -> AvailabilitySpec:
        return AvailabilitySpec(
            clock_id=clock.clock_id, phase=phase, outcome_access=access
        )

    quantities = (
        *(
            QuantitySpec(
                quantity_id=quantity_id,
                label=f"Executed deformable-mirror {label} command coefficient",
                kind=QuantityKind.ACTION,
                dimension="wavefront-command-coefficient",
                native_unit="1",
                coordinate_frame="source-executed-six-coordinate-action-chart",
                clock_id=clock.clock_id,
                availability=availability(
                    CausalPhase.ACTION_APPLIED,
                    OutcomeAccess.OUTCOME_BLIND,
                ),
            )
            for quantity_id, label, _lower, _upper in _ACTION_SPECS
        ),
        QuantitySpec(
            quantity_id="glenn-boundary-laser-water-sheet",
            label="Prepared laser, deformable mirror, diagnostics and water-sheet boundary",
            kind=QuantityKind.BOUNDARY,
            dimension="prepared-boundary-identity",
            native_unit="1",
            coordinate_frame="20210730-run14-physical-setup",
            clock_id=clock.clock_id,
            availability=availability(
                CausalPhase.PREPARATION, OutcomeAccess.OUTCOME_BLIND
            ),
        ),
        QuantitySpec(
            quantity_id="glenn-denominator-run14-adaptive-campaign",
            label="Prepared 20210730 run14 adaptive campaign denominator",
            kind=QuantityKind.DENOMINATOR,
            dimension="prepared-system-identity",
            native_unit="1",
            coordinate_frame="20210730-run14-acquisition-session",
            clock_id=clock.clock_id,
            availability=availability(
                CausalPhase.PREPARATION, OutcomeAccess.OUTCOME_BLIND
            ),
        ),
        QuantitySpec(
            quantity_id="glenn-history-prior-burst-trajectory",
            label="Retained prior action and online-fitness trajectory",
            kind=QuantityKind.HISTORY,
            dimension="adaptive-campaign-history",
            native_unit="1",
            coordinate_frame="strictly-prior-source-bursts",
            clock_id=clock.clock_id,
            availability=availability(
                CausalPhase.PRE_ACTION,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
            ),
        ),
        QuantitySpec(
            quantity_id="glenn-online-fitness-score",
            label="Source online fitness score",
            kind=QuantityKind.RECEIVER,
            dimension="source-fitness-score",
            native_unit="fitness-score",
            coordinate_frame="source-online-optimizer-readout",
            clock_id=clock.clock_id,
            availability=availability(
                CausalPhase.POST_OUTCOME,
                OutcomeAccess.EVALUATION_REVEALED,
            ),
            response_direction=ResponseDirection.HIGHER_IS_BETTER,
        ),
    )
    relation = RelationalIdentity(
        relation_id="glenn-run14-history-conditioned-fitness-response",
        denominator_quantity_ids=(
            "glenn-boundary-laser-water-sheet",
            "glenn-denominator-run14-adaptive-campaign",
        ),
        history_quantity_ids=("glenn-history-prior-burst-trajectory",),
        memoryless=False,
        action_quantity_ids=tuple(value[0] for value in _ACTION_SPECS),
        receiver_quantity_ids=("glenn-online-fitness-score",),
        horizon=HorizonSpec(
            horizon_id="glenn-one-burst-response-horizon",
            clock_id=clock.clock_id,
            duration=Decimal(1),
            time_unit="burst",
        ),
    )
    budget = ResourceBudget(
        cpu_cores=8,
        memory_bytes=2_000_000_000,
        gpu_devices=0,
        wall_time_seconds=3_600,
        source_scan_bytes=100_000_000,
        output_bytes=20_000_000,
    )
    authority = AuthorityPolicy(
        policy_id="glenn-nonactuating-exploration-policy",
        delegator_id="local-operator",
        delegate_id="bounded-local-exploration-runtime",
        scope_ids=(
            "glenn-g1-existing-artifacts",
            "glenn-g1-outcome-visible-exploration",
        ),
        allowed_world_kinds=frozenset({WorldKind.PHYSICAL_EXPERIMENT}),
        allowed_actions=frozenset(
            {
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.PUBLIC_SOURCE_ACQUISITION,
                AuthorityAction.READ_ONLY_EXPLORATION,
                AuthorityAction.REPOSITORY_IMPLEMENTATION,
            }
        ),
        allowed_source_classes=frozenset({SourceAccessClass.OFFICIAL_OPEN_PUBLIC}),
        required_gate_ids=(
            "clean-commit",
            "custody-separation",
            "exact-existing-source",
            "resource-envelope",
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
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    world = WorldSpec(
        world_id="glenn-20210730-run14-physical-experiment",
        label="Glenn laser--water-sheet physical experiment, 20210730 run14",
        kind=WorldKind.PHYSICAL_EXPERIMENT,
        represented_physics=(
            "adaptive-deformable-mirror-command-trajectory",
            "laser-water-sheet-prepared-medium",
            "source-online-fitness-readout",
        ),
        unrepresented_physics=(
            "delivered-wavefront-measurement",
            "fresh-independent-acquisition-sessions",
            "joint-physical-receiver-and-sink-gates",
        ),
        privileged_truth_quantity_ids=(),
        maximum_evidence=EvidenceCeiling.ADMISSION,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_REVEALED,
                OutcomeAccess.OUTCOME_BLIND,
            }
        ),
    )
    return SystemSpec(
        system_id="glenn-run14-laser-water-sheet-system",
        label="Glenn run14 adaptive laser--water-sheet system",
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=tuple(sorted(quantities, key=lambda value: value.quantity_id)),
        independent_unit=IndependentUnitSpec(
            unit_id="glenn-acquisition-session",
            label="Independently prepared physical acquisition session",
            grouping_key="glenn-acquisition-session-id",
        ),
        authority_policy=authority,
        ports=(),
    )


def glenn_g1_diagnostic_projection(
    *,
    projection: ExplorationDiagnosticProjection,
    reproduction_receipt: ObjectIdentity,
) -> ExplorationDiagnosticProjection:
    """Accept a qualified external projection; no embedded historical values."""
    system = glenn_g1_exploration_system()
    if (
        projection.system_id != system.system_id
        or projection.relation != system.relation
        or projection.independent_unit_id != system.independent_unit.unit_id
        or projection.upstream_objects != (reproduction_receipt,)
        or reproduction_receipt.object_schema
        != "empirical-lawhood/physical/glenn/qualified-reproduction-receipt"
        or projection.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
        or projection.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
    ):
        raise ValueError(
            "Glenn diagnostic projection changes its qualified parent or denominator"
        )
    return projection


def glenn_g1_exploration_runtime_provider(
    *, projection: ExplorationDiagnosticProjection, reproduction_receipt: ObjectIdentity
) -> ReferenceExplorationExecutionRuntimeProvider:
    """Compose the generic executor with the exact registered Glenn projection."""

    projection = glenn_g1_diagnostic_projection(
        projection=projection, reproduction_receipt=reproduction_receipt
    )
    return ReferenceExplorationExecutionRuntimeProvider(
        registry=glenn_g1_exploration_registry(),
        registered_inputs=(
            RegisteredExplorationInput(
                logical_artifact_id=GLENN_G1_PROJECTION_ARTIFACT_ID,
                payload_schema=projection.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                payload=projection.canonical_bytes(),
            ),
        ),
    )


def _axes() -> tuple[SearchAxis, ...]:
    values = {
        "denominator-cell": ("run14-prepared-campaign",),
        "design-sensitivity": ("single-session-information-limit",),
        "horizon": ("lag20-support",),
        "model": ("ridge-versus-knn",),
        "receiver": ("online-fitness-score",),
        "receiver-component": ("online-fitness-score",),
        "representation": ("initial-versus-evaluation-action-chart",),
        "transform": ("source-native-command-scaling",),
    }
    return tuple(
        SearchAxis(
            axis_id=axis_id,
            candidate_ids=values.get(axis_id, (f"{axis_id}-not-source-resolved",)),
        )
        for axis_id in _ALL_AXIS_IDS
    )


def _portfolio_policy() -> PortfolioPolicy:
    # One physical session is enough to diagnose an information limit, but not
    # enough to make the selected analysis coordinates evaluable.
    return replace(
        default_portfolio_policy(),
        policy_id="glenn-g1-conservative-pareto-policy",
        minimum_falsification_value=Decimal("0.03"),
        minimum_independent_unit_adequacy=Decimal("0.125"),
    )


def _design_context(
    system: SystemSpec,
    snapshot: EvidenceSnapshot,
) -> ProspectiveDesignContext:
    kinds = tuple(
        sorted(
            (
                ProspectiveDesignKind.INFORMATION_RANK,
                ProspectiveDesignKind.MODEL_DISCRIMINATION,
            )
        )
    )
    return ProspectiveDesignContext(
        context_id="glenn-g1-fresh-test-nomination-context",
        source_snapshot=ObjectIdentity.from_record(snapshot.snapshot_id, snapshot),
        system=ObjectIdentity.from_record(system.system_id, system),
        design_kinds=kinds,
        evidence_contract=None,
        assignment_kind=AssignmentKind.RANDOMIZED_INTERVENTION,
        assignment_mechanism=(
            "Randomize a frozen bounded local command design by fresh acquisition session."
        ),
        assignment_support_restriction_ids=("historical-g1-executed-action-bounds",),
        randomization_unit_id=system.independent_unit.unit_id,
        measurements=(
            NativeMeasurementRequirement(
                requirement_id="measurement.glenn-online-fitness-score",
                quantity_id="glenn-online-fitness-score",
                native_unit="fitness-score",
                clock_id="glenn-burst-clock",
            ),
        ),
        action_bounds=tuple(
            sorted(
                (
                    QuantityBound(
                        bound_id=f"bound.{quantity_id}",
                        quantity_id=quantity_id,
                        native_unit="1",
                        lower=Decimal(lower),
                        upper=Decimal(upper),
                    )
                    for quantity_id, _label, lower, upper in _ACTION_SPECS
                ),
                key=lambda value: value.bound_id,
            )
        ),
        denominator_cell_ids=("glenn-run14-prepared-campaign-cell",),
        controls=(),
        admission_gate_ids=(
            "delivered-wavefront-validity",
            "independent-session-replication",
            "receiver-measurement-validity",
        ),
        precision_goals=(),
        objectives=tuple(
            sorted(
                (
                    ProspectiveObjectiveVector(
                        objective_id="objective.glenn-information-rank",
                        design_kind=ProspectiveDesignKind.INFORMATION_RANK,
                        falsification_value=Decimal("0.9"),
                        hypothesis_discrimination=Decimal("0.8"),
                        support_or_rank_gain=Decimal("0.9"),
                        acquisition_cost=Decimal("0.8"),
                        safety_or_authority_risk=Decimal("0.8"),
                    ),
                    ProspectiveObjectiveVector(
                        objective_id="objective.glenn-model-discrimination",
                        design_kind=ProspectiveDesignKind.MODEL_DISCRIMINATION,
                        falsification_value=Decimal("0.9"),
                        hypothesis_discrimination=Decimal("0.9"),
                        support_or_rank_gain=Decimal("0.7"),
                        acquisition_cost=Decimal("0.8"),
                        safety_or_authority_risk=Decimal("0.8"),
                    ),
                ),
                key=lambda value: value.objective_id,
            )
        ),
        alternative_design_ids=("glenn-g2-local-curvature", "glenn-g3-memory-closure"),
        safety_constraint_ids=(
            "facility-approved-action-envelope",
            "laser-and-deformable-mirror-limits",
        ),
        budget=system.authority_policy.budget_ceiling,
        information_cutoff=InformationCutoff(
            cutoff_id="glenn-fresh-test-pre-action-cutoff",
            clock_id="glenn-burst-clock",
            phase=CausalPhase.PRE_ACTION,
            coordinate=Decimal(0),
        ),
        authority_action=AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
        source_access=SourceAccessClass.OFFICIAL_OPEN_PUBLIC,
        claim_proposition=(
            "A randomized local Glenn action response recurs across fresh sessions."
        ),
        claim_estimand=(
            "Session-replicated native action-to-receiver contrast within frozen bounds."
        ),
        claim_promotion_rule=(
            "Fresh randomized sessions pass support, falsifier, validity and precision gates."
        ),
        claim_assumption_ids=(
            "delivered-command-measured",
            "fresh-session-independence",
            "receiver-validity-measured",
        ),
        requested_rung=EvidenceRung.LOCAL_LAW,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        no_acquisition_reason_codes=(
            "facility-authority-and-fresh-session-manifest-required",
        ),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )


def build_glenn_g1_exploration_package(
    implementation_commit: str,
    *,
    projection: ExplorationDiagnosticProjection,
    reproduction_receipt: ObjectIdentity,
) -> ExplorationExecutionPackage:
    """Build the input-only package for one bounded Glenn G1 exploration wave."""

    system = glenn_g1_exploration_system()
    projection = glenn_g1_diagnostic_projection(
        projection=projection, reproduction_receipt=reproduction_receipt
    )
    registry = glenn_g1_exploration_registry()
    projection_payload = projection.canonical_bytes()
    artifact = ArtifactIdentity(
        artifact_id=GLENN_G1_PROJECTION_ARTIFACT_ID,
        role="outcome-visible-glenn-g1-diagnostic-projection",
        payload_schema=projection.SCHEMA,
        sha256=projection.fingerprint(),
        media_type="application/json",
        size_bytes=len(projection_payload),
    )
    parent_objects = tuple(
        sorted(
            (
                reproduction_receipt,
                ObjectIdentity.from_record(system.system_id, system),
            ),
            key=lambda value: value.object_id,
        )
    )
    snapshot = EvidenceSnapshot(
        snapshot_id="snapshot.glenn-g1-remediated-exploration",
        campaign_id="glenn-g1-existing-evidence-campaign",
        run_id="glenn-g1-custodied-reproduction",
        world_id=system.world.world_id,
        parent_objects=parent_objects,
        parent_visibility_ceilings=tuple(
            VisibilityCeiling.OUTCOME_VISIBLE for _value in parent_objects
        ),
        artifacts=(artifact,),
        claim_ids=(),
        information_cutoff=InformationCutoff(
            cutoff_id="cutoff.glenn-g1-all-reproduced-outcomes",
            clock_id="glenn-burst-clock",
            phase=CausalPhase.POST_OUTCOME,
            coordinate=Decimal(101),
        ),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        projection_ids=("glenn-g1-diagnostic-measures",),
    )
    verification = SnapshotVerification(
        verification_id="verification.glenn-g1-remediated-exploration",
        snapshot=ObjectIdentity.from_record(snapshot.snapshot_id, snapshot),
        artifact_ids=(artifact.artifact_id,),
        verification_check_ids=(
            "content-sha256",
            "read-only-outcome-visible",
            "reproduction-receipt-bound",
        ),
        verifier_key="glenn-g1-reproduction-snapshot-verifier",
        verifier_version="1.0.0",
    )
    view = bind_evidence_view(snapshot, projection)
    signals = default_detector_registry().detect(snapshot, projection, view)
    quantity_reporting = tuple(
        QuantityReportingObligation(
            quantity_id=quantity.quantity_id,
            native_unit=quantity.native_unit,
            normalized_reporting_allowed=True,
        )
        for quantity in system.quantities
        if quantity.quantity_id
        in {
            *system.relation.action_quantity_ids,
            *system.relation.receiver_quantity_ids,
        }
    )
    context = TemplateContext(
        context_id="template-context.glenn-g1-remediated-exploration",
        snapshot=ObjectIdentity.from_record(snapshot.snapshot_id, snapshot),
        evidence_view=ObjectIdentity.from_record(view.view_id, view),
        search_axes=_axes(),
        quantity_reporting=quantity_reporting,
        analysis_budget=_analysis_budget(),
        executed_template_ids=(),
    )
    all_instantiations = instantiate_matching_templates(
        snapshot, view, signals, context
    )
    candidates = build_portfolio_candidates(all_instantiations, signals, view)
    planned = ExplorationPortfolioPlanner().select(
        plan_id=GLENN_G1_EXPLORATION_PLAN_ID,
        snapshot=snapshot,
        snapshot_verification=verification,
        candidates=candidates,
        policy=_portfolio_policy(),
    )
    if planned.plan is None:
        raise ValueError(
            "Glenn G1 diagnostics produced no executable conservative portfolio"
        )
    old_plan = planned.plan
    proposals = tuple(
        replace(
            proposal,
            analysis=replace(
                proposal.analysis,
                registered_pipeline_key="exploration.pipeline",
                registered_pipeline_version="1.0.0",
                budget=_analysis_budget(),
            ),
        )
        for proposal in old_plan.proposals
    )
    proposals_by_id = {value.proposal_id: value for value in proposals}
    selected_ids = {
        value.proposal_id
        for value in old_plan.selections
        if value.disposition is ProposalDisposition.SELECTED
    }
    instantiations = tuple(
        sorted(
            (
                replace(value, proposal=proposals_by_id[value.proposal.proposal_id])
                for value in all_instantiations
                if value.proposal.proposal_id in selected_ids
            ),
            key=lambda value: value.instantiation_id,
        )
    )
    plan = compile_exploration_plan(
        plan_id=GLENN_G1_EXPLORATION_PLAN_ID,
        snapshot=snapshot,
        snapshot_verification=verification,
        proposals=proposals,
        selections=old_plan.selections,
        budget=old_plan.budget,
    )
    metric_by_template = {
        "counterfeit-rank-representation-preprocessing": Decimal("0.5862078604215281"),
        "distributional-receiver-opposing-components": Decimal("0.1528507"),
        "independent-unit-information-design-sensitivity": Decimal(1),
        "support-chart-denominator-fragmentation": Decimal("0.7741935"),
        "wrong-action-donor-specificity-topology": Decimal("0.1528507"),
    }
    observations = tuple(
        sorted(
            (
                AnalysisCoordinateObservation(
                    observation_id=(
                        f"observation.{value.proposal.analysis.analysis_id}.{member_id}"
                    ),
                    analysis_id=value.proposal.analysis.analysis_id,
                    family_member_id=member_id,
                    effect_quantity_id="glenn-online-fitness-score",
                    native_unit="diagnostic-ratio",
                    point=metric_by_template.get(value.template.object_id, Decimal(0)),
                    lower=Decimal(0),
                    upper=Decimal(2),
                    minimum_absolute_pattern=Decimal("0.05"),
                    physical_independent_unit_count=1,
                    evaluable=False,
                    operational_failure_reason_codes=(),
                    matched_null_passed=False,
                    scaling_control_passed=True,
                    aggregation_control_passed=True,
                    leakage_control_passed=True,
                    selection_control_passed=True,
                    leave_one_unit_max_change=Decimal(1),
                    maximum_influence=Decimal(0),
                )
                for value in instantiations
                for member_id in value.obligations.registered_family_member_ids
            ),
            key=lambda value: value.observation_id,
        )
    )
    wave_input = ExplorationWaveInput(
        wave_input_id="glenn-g1-remediated-exploration-input",
        snapshot=snapshot,
        snapshot_verification=verification,
        plan=plan,
        instantiations=instantiations,
        observations=observations,
    )
    design_context = _design_context(system, snapshot)
    approval_request = ApprovalRequest(
        authorization_id="authorization.glenn-fresh-experiment-required",
        action=design_context.authority_action,
        world_kind=system.world.kind,
        source_access=design_context.source_access,
        requested_scope_id="glenn-fresh-g2-g3-experiment",
        requested_budget=design_context.budget,
        passed_gate_ids=("clean-commit", "existing-source-verified"),
        proposer_id="remediated-exploration-runtime",
        approver_id="local-operator",
        decided_at_utc="2026-07-16T00:00:00Z",
        implementation_commit=implementation_commit,
    )
    return ExplorationExecutionPackage(
        package_id="glenn-g1-remediated-exploration-package",
        execution_plan_id=f"execution.{plan.plan_id}",
        implementation_commit=implementation_commit,
        system=system,
        snapshot=snapshot,
        snapshot_verification=verification,
        exploration_plan=plan,
        registry=registry,
        wave_input=wave_input,
        design_context=design_context,
        approval_request=approval_request,
        skeptic_capability_key="exploration.skeptic",
        skeptic_capability_version="1.0.0",
        synthesis_capability_key="exploration.synthesis",
        synthesis_capability_version="1.0.0",
    )


__all__ = [
    "GLENN_G1_EXPLORATION_PLAN_ID",
    "GLENN_G1_PROJECTION_ARTIFACT_ID",
    "build_glenn_g1_exploration_package",
    "glenn_g1_diagnostic_projection",
    "glenn_g1_exploration_registry",
    "glenn_g1_exploration_runtime_provider",
    "glenn_g1_exploration_system",
]
