"""Canonical Exploration conformance over truth-known exploration reference preparations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.adapters.reference_worlds import (
    ReferenceEvaluation,
    ReferenceWorldKind,
    ReferenceWorldSpec,
    evaluate_reference_world,
    get_reference_world,
)
from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceSnapshot, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import LifecycleStatus
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.campaigns import (
    CampaignLane,
    CampaignNode,
    CampaignNodeKind,
    CampaignSpec,
    DecisionRight,
)
from empirical_lawhood.planning.discovery import (
    CampaignGraph,
    DiagnosticMeasure,
    DiagnosticMeasureKind,
    DiscoveryEdge,
    DiscoveryEdgeKind,
    DiscoveryLane,
    DiscoveryNode,
    DiscoveryNodeKind,
    ExplorationDiagnosticProjection,
    ExplorationEvidenceView,
    ExplorationLifecycleLedger,
    PortfolioDecisionKind,
    QuantityReportingObligation,
    TemplateInstantiation,
    ThresholdDirection,
)
from empirical_lawhood.planning.exploration import (
    AnalysisAttemptStatus,
    AnomalySignal,
    ExplorationPlan,
    ProposalDisposition,
    SearchAxis,
)
from empirical_lawhood.runtime.plans import SnapshotVerification

from .detectors import bind_evidence_view, default_detector_registry
from .execution import (
    AnalysisCoordinateObservation,
    ExplorationWaveResult,
    execute_wave,
)
from .portfolio import (
    ExplorationPortfolioPlanner,
    PlannedPortfolio,
    build_portfolio_candidates,
    default_portfolio_policy,
)
from .templates import (
    TemplateContext,
    instantiate_matching_templates,
)


@dataclass(frozen=True, slots=True)
class NamedCount(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/exploration/named-count'

    count_id: str
    count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.count_id, field_name="count_id")
        if self.count < 0:
            raise ValueError("named count must be nonnegative")


@dataclass(frozen=True, slots=True)
class ExplorationConformanceOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/exploration/exploration-conformance-outcome'

    case_id: str
    reference_world_id: str
    signal_kind_ids: tuple[str, ...]
    decision_kind: PortfolioDecisionKind
    disposition_counts: tuple[NamedCount, ...]
    attempt_counts: tuple[NamedCount, ...]
    finding_status_ids: tuple[str, ...]
    selected_template_ids: tuple[str, ...]
    hypothesis_disposition_ids: tuple[str, ...]
    succeeded_family_member_ids: tuple[str, ...]
    non_succeeded_family_member_ids: tuple[str, ...]
    skeptic_passed: bool | None
    parent_fingerprint_preserved: bool
    result_fingerprint: str

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        validate_stable_id(self.reference_world_id, field_name="reference_world_id")
        for field_name, values in (
            ("signal_kind_ids", self.signal_kind_ids),
            ("finding_status_ids", self.finding_status_ids),
            ("selected_template_ids", self.selected_template_ids),
            ("hypothesis_disposition_ids", self.hypothesis_disposition_ids),
            ("succeeded_family_member_ids", self.succeeded_family_member_ids),
            ("non_succeeded_family_member_ids", self.non_succeeded_family_member_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name)
        require_sorted_unique_ids(
            self.disposition_counts,
            attribute="count_id",
            field_name="disposition_counts",
        )
        require_sorted_unique_ids(
            self.attempt_counts,
            attribute="count_id",
            field_name="attempt_counts",
        )
        validate_sha256(self.result_fingerprint, field_name="result_fingerprint")
        if not self.parent_fingerprint_preserved:
            raise ValueError(
                "Exploration conformance cannot mutate its parent reference result"
            )


@dataclass(frozen=True, slots=True)
class ExplorationReferenceConformanceReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/exploration/exploration-reference-conformance-report'

    report_id: str
    registry_fingerprint: str
    outcomes: tuple[ExplorationConformanceOutcome, ...]
    campaign_graph: CampaignGraph
    wave_ledgers: tuple[ExplorationLifecycleLedger, ...]
    physical_observation_to_controller_use_execution: str
    status: str

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        validate_sha256(self.registry_fingerprint, field_name="registry_fingerprint")
        require_sorted_unique_ids(
            self.outcomes, attribute="case_id", field_name="outcomes"
        )
        require_sorted_unique_ids(
            self.wave_ledgers,
            attribute="ledger_id",
            field_name="wave_ledgers",
        )
        validate_nonempty(
            self.physical_observation_to_controller_use_execution, field_name="physical_observation_to_controller_use_execution"
        )
        validate_nonempty(self.status, field_name="status")
        if self.physical_observation_to_controller_use_execution != "NONE":
            raise ValueError(
                "Exploration reference conformance cannot claim physical measurement through controller use execution"
            )
        if self.status != "PASS":
            raise ValueError("Exploration canonical conformance report must pass")


@dataclass(frozen=True, slots=True)
class _PreparedExploration:
    world: ReferenceWorldSpec
    evaluation: ReferenceEvaluation
    projection: ExplorationDiagnosticProjection
    snapshot: EvidenceSnapshot
    view: ExplorationEvidenceView
    verification: SnapshotVerification
    signals: tuple[AnomalySignal, ...]
    instantiations: tuple[TemplateInstantiation, ...]
    planned: PlannedPortfolio
    parent_fingerprint_before: str


_ALL_AXIS_IDS = (
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


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=1_000_000_000,
        gpu_devices=0,
        wall_time_seconds=600,
        source_scan_bytes=100_000_000,
        output_bytes=10_000_000,
    )


def _axes(
    overrides: dict[str, tuple[str, ...]] | None = None,
) -> tuple[SearchAxis, ...]:
    selected = overrides or {}
    return tuple(
        SearchAxis(
            axis_id=axis_id,
            candidate_ids=selected.get(axis_id, (f"{axis_id}-default",)),
        )
        for axis_id in _ALL_AXIS_IDS
    )


def _measure(
    world: ReferenceWorldSpec,
    measure_id: str,
    kind: DiagnosticMeasureKind,
    value: str,
    threshold: str,
    direction: ThresholdDirection,
    physical_units: int,
) -> DiagnosticMeasure:
    return DiagnosticMeasure(
        measure_id=measure_id,
        kind=kind,
        relation_id=world.system.relation.relation_id,
        affected_quantity_ids=world.system.relation.receiver_quantity_ids,
        value=Decimal(value),
        threshold=Decimal(threshold),
        threshold_direction=direction,
        native_unit="1",
        physical_independent_unit_count=physical_units,
        evidence_link_ids=(f"reference-evidence.{world.reference_id}.{measure_id}",),
    )


def _prepare(
    *,
    case_id: str,
    kind: ReferenceWorldKind,
    measures: tuple[DiagnosticMeasure, ...] | None = None,
    axis_overrides: dict[str, tuple[str, ...]] | None = None,
    executed_template_ids: tuple[str, ...] = (),
) -> _PreparedExploration:
    world = get_reference_world(kind)
    evaluation = evaluate_reference_world(world)
    selected_measures = measures or ()
    if any(
        measure.relation_id != world.system.relation.relation_id
        for measure in selected_measures
    ):
        raise ValueError("reference preparation received a foreign diagnostic measure")
    projection = ExplorationDiagnosticProjection(
        projection_id=f"projection.exploration.{case_id}",
        system_id=world.system.system_id,
        relation=world.system.relation,
        independent_unit_id=world.system.independent_unit.unit_id,
        grouping_clock_ids=(world.system.clocks[0].clock_id,),
        available_role_ids=(
            "action",
            "applied-action",
            "constraint",
            "denominator",
            "history",
            "independent-unit",
            "numerical-view",
            "observer",
            "receiver",
            "selector",
            "support",
            "time",
            "uncertainty",
        ),
        available_upstream_schema_ids=(ReferenceEvaluation.SCHEMA,),
        upstream_objects=(
            ObjectIdentity.from_record(evaluation.evaluation_id, evaluation),
        ),
        measures=tuple(
            sorted(selected_measures, key=lambda measure: measure.measure_id)
        ),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    payload = projection.canonical_bytes()
    artifact = ArtifactIdentity(
        artifact_id=f"artifact.{projection.projection_id}",
        role="outcome-visible-diagnostic-projection",
        payload_schema=projection.SCHEMA,
        sha256=projection.fingerprint(),
        media_type="application/json",
        size_bytes=len(payload),
    )
    cutoff = InformationCutoff(
        cutoff_id=f"cutoff.exploration.{case_id}",
        clock_id=world.system.clocks[0].clock_id,
        phase=CausalPhase.POST_OUTCOME,
        coordinate=Decimal(100),
    )
    snapshot = EvidenceSnapshot(
        snapshot_id=f"snapshot.exploration.{case_id}",
        campaign_id="exploration-reference-campaign",
        run_id=f"exploration-parent.{case_id}",
        world_id=world.reference_id,
        parent_objects=tuple(
            sorted(
                (
                    ObjectIdentity.from_record(evaluation.evaluation_id, evaluation),
                    ObjectIdentity.from_record(world.reference_id, world),
                ),
                key=lambda identity: identity.object_id,
            )
        ),
        parent_visibility_ceilings=(
            VisibilityCeiling.OUTCOME_VISIBLE,
            VisibilityCeiling.OUTCOME_VISIBLE,
        ),
        artifacts=(artifact,),
        claim_ids=(),
        information_cutoff=cutoff,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        projection_ids=("diagnostic-measures", "reference-evaluation"),
    )
    view = bind_evidence_view(snapshot, projection)
    verification = SnapshotVerification(
        verification_id=f"verification.exploration.{case_id}",
        snapshot=ObjectIdentity.from_record(snapshot.snapshot_id, snapshot),
        artifact_ids=(artifact.artifact_id,),
        verification_check_ids=("content-sha256", "read-only-outcome-visible"),
        verifier_key="exploration-reference-snapshot-verifier",
        verifier_version="1.0.0",
    )
    signals = default_detector_registry().detect(snapshot, projection, view)
    quantity_reporting = tuple(
        QuantityReportingObligation(
            quantity_id=quantity.quantity_id,
            native_unit=quantity.native_unit,
            normalized_reporting_allowed=True,
        )
        for quantity in world.system.quantities
        if quantity.quantity_id
        in {
            *world.system.relation.action_quantity_ids,
            *world.system.relation.receiver_quantity_ids,
        }
    )
    context = TemplateContext(
        context_id=f"template-context.exploration.{case_id}",
        snapshot=ObjectIdentity.from_record(snapshot.snapshot_id, snapshot),
        evidence_view=ObjectIdentity.from_record(view.view_id, view),
        search_axes=_axes(axis_overrides),
        quantity_reporting=quantity_reporting,
        analysis_budget=_budget(),
        executed_template_ids=executed_template_ids,
    )
    instantiations = instantiate_matching_templates(snapshot, view, signals, context)
    candidates = build_portfolio_candidates(
        instantiations,
        signals,
        view,
        executed_template_ids=executed_template_ids,
    )
    planned = ExplorationPortfolioPlanner().select(
        plan_id=f"exploration-{case_id}-wave",
        snapshot=snapshot,
        snapshot_verification=verification,
        candidates=candidates,
        policy=default_portfolio_policy(),
    )
    return _PreparedExploration(
        world=world,
        evaluation=evaluation,
        projection=projection,
        snapshot=snapshot,
        view=view,
        verification=verification,
        signals=signals,
        instantiations=instantiations,
        planned=planned,
        parent_fingerprint_before=evaluation.fingerprint(),
    )


def _selected_instantiations(
    prepared: _PreparedExploration,
) -> tuple[TemplateInstantiation, ...]:
    plan = prepared.planned.plan
    if plan is None:
        return ()
    selected = {
        selection.proposal_id
        for selection in plan.selections
        if selection.disposition is ProposalDisposition.SELECTED
    }
    return tuple(
        value
        for value in prepared.instantiations
        if value.proposal.proposal_id in selected
    )


def _observation(
    instantiation: TemplateInstantiation,
    member_id: str,
    *,
    point: str,
    lower: str,
    upper: str,
    physical_units: int,
    evaluable: bool = True,
    operational_failure: bool = False,
) -> AnalysisCoordinateObservation:
    return AnalysisCoordinateObservation(
        observation_id=f"observation.{instantiation.proposal.analysis.analysis_id}.{member_id}",
        analysis_id=instantiation.proposal.analysis.analysis_id,
        family_member_id=member_id,
        effect_quantity_id=instantiation.proposal.analysis.receiver_quantity_ids[0],
        native_unit=instantiation.obligations.quantity_reporting[-1].native_unit,
        point=Decimal(point),
        lower=Decimal(lower),
        upper=Decimal(upper),
        minimum_absolute_pattern=Decimal("0.2"),
        physical_independent_unit_count=physical_units,
        evaluable=evaluable and not operational_failure,
        operational_failure_reason_codes=(
            ("injected-registered-method-failure",) if operational_failure else ()
        ),
        matched_null_passed=True,
        scaling_control_passed=True,
        aggregation_control_passed=True,
        leakage_control_passed=True,
        selection_control_passed=True,
        leave_one_unit_max_change=Decimal("0.05"),
        maximum_influence=Decimal("0.2"),
    )


def _execute_planted(prepared: _PreparedExploration) -> ExplorationWaveResult:
    plan = prepared.planned.plan
    if plan is None:
        raise ValueError("planted reference portfolio unexpectedly stopped")
    observations = []
    for instantiation in _selected_instantiations(prepared):
        for member in instantiation.obligations.registered_family_member_ids:
            if "planted-relation" in member:
                point, lower, upper = "0.8", "0.6", "1.0"
                evaluable = True
            elif "attractive-decoy" in member:
                point, lower, upper = "1.2", "1.0", "1.4"
                evaluable = False
            else:
                point, lower, upper = "0", "-0.1", "0.1"
                evaluable = True
            observations.append(
                _observation(
                    instantiation,
                    member,
                    point=point,
                    lower=lower,
                    upper=upper,
                    physical_units=12,
                    evaluable=evaluable,
                )
            )
    return execute_wave(
        snapshot=prepared.snapshot,
        plan=plan,
        instantiations=prepared.instantiations,
        observations=tuple(
            sorted(observations, key=lambda value: value.observation_id)
        ),
    )


def _execute_uniform(
    prepared: _PreparedExploration,
    *,
    evaluable: bool,
    operational_failure_template_id: str | None = None,
) -> ExplorationWaveResult:
    plan = prepared.planned.plan
    if plan is None:
        raise ValueError("reference portfolio unexpectedly stopped")
    observations = []
    for instantiation in _selected_instantiations(prepared):
        for index, member in enumerate(
            instantiation.obligations.registered_family_member_ids
        ):
            failed = (
                operational_failure_template_id == instantiation.template.object_id
                and index == 0
            )
            observations.append(
                _observation(
                    instantiation,
                    member,
                    point="0",
                    lower="-0.1",
                    upper="0.1",
                    physical_units=min(
                        measure.physical_independent_unit_count
                        for measure in prepared.view.measures
                    ),
                    evaluable=evaluable,
                    operational_failure=failed,
                )
            )
    return execute_wave(
        snapshot=prepared.snapshot,
        plan=plan,
        instantiations=prepared.instantiations,
        observations=tuple(
            sorted(observations, key=lambda value: value.observation_id)
        ),
    )


def _counts(values: tuple[str, ...]) -> tuple[NamedCount, ...]:
    return tuple(
        NamedCount(count_id=value, count=values.count(value))
        for value in sorted(set(values))
    )


def _outcome(
    case_id: str,
    prepared: _PreparedExploration,
    result: ExplorationWaveResult | None,
) -> ExplorationConformanceOutcome:
    decision = prepared.planned.decision
    selected_templates = tuple(
        sorted(value.template.object_id for value in _selected_instantiations(prepared))
    )
    if result is None:
        attempt_ids: tuple[str, ...] = ()
        finding_ids: tuple[str, ...] = ()
        hypothesis_ids: tuple[str, ...] = ()
        succeeded_members: tuple[str, ...] = ()
        non_succeeded_members: tuple[str, ...] = ()
        skeptic_passed = None
        result_fingerprint = decision.fingerprint()
    else:
        attempt_ids = tuple(attempt.status.value.lower() for attempt in result.attempts)
        finding_ids = tuple(
            sorted({finding.status.value.lower() for finding in result.findings})
        )
        hypothesis_ids = tuple(
            sorted(
                {
                    hypothesis.disposition.value.lower()
                    for hypothesis in result.hypothesis_synthesis.hypothesis_set.hypotheses
                }
            )
        )
        skeptic_passed = result.skeptic_report.passed
        succeeded_members = tuple(
            sorted(
                attempt.family_coordinate_ids[0]
                for attempt in result.attempts
                if attempt.status is AnalysisAttemptStatus.SUCCEEDED
            )
        )
        non_succeeded_members = tuple(
            sorted(
                attempt.family_coordinate_ids[0]
                for attempt in result.attempts
                if attempt.status is not AnalysisAttemptStatus.SUCCEEDED
            )
        )
        result_fingerprint = result.fingerprint()
    dispositions = tuple(
        selection.disposition.value.lower() for selection in decision.selections
    )
    return ExplorationConformanceOutcome(
        case_id=case_id,
        reference_world_id=prepared.world.reference_id,
        signal_kind_ids=tuple(
            sorted({signal.kind.value.lower() for signal in prepared.signals})
        ),
        decision_kind=decision.kind,
        disposition_counts=_counts(dispositions),
        attempt_counts=_counts(attempt_ids),
        finding_status_ids=finding_ids,
        selected_template_ids=selected_templates,
        hypothesis_disposition_ids=hypothesis_ids,
        succeeded_family_member_ids=succeeded_members,
        non_succeeded_family_member_ids=non_succeeded_members,
        skeptic_passed=skeptic_passed,
        parent_fingerprint_preserved=(
            prepared.evaluation.fingerprint() == prepared.parent_fingerprint_before
        ),
        result_fingerprint=result_fingerprint,
    )


def _second_wave(
    first: _PreparedExploration,
    first_result: ExplorationWaveResult,
) -> _PreparedExploration:
    measures = (
        _measure(
            first.world,
            "adaptive-coordinate-instability",
            DiagnosticMeasureKind.COORDINATE_INSTABILITY,
            "0.4",
            "0.2",
            ThresholdDirection.ABOVE_MAXIMUM,
            16,
        ),
        _measure(
            first.world,
            "adaptive-residual-memory",
            DiagnosticMeasureKind.RESIDUAL_MEMORY,
            "0.6",
            "0.2",
            ThresholdDirection.ABOVE_MAXIMUM,
            16,
        ),
    )
    prepared = _prepare(
        case_id="adaptive-second-wave",
        kind=ReferenceWorldKind.PLANTED_RELATIONAL_ANOMALY,
        measures=measures,
        axis_overrides={
            "history-ladder": ("current-state",),
            "horizon": ("short",),
            "representation": ("native",),
            "time-direction": ("forward",),
            "transform": ("native",),
        },
        executed_template_ids=("counterfeit-rank-representation-preprocessing",),
    )
    parent_objects = tuple(
        sorted(
            (
                *prepared.snapshot.parent_objects,
                ObjectIdentity.from_record(
                    first_result.hypothesis_synthesis.hypothesis_set.hypothesis_set_id,
                    first_result.hypothesis_synthesis.hypothesis_set,
                ),
                *(
                    ObjectIdentity.from_record(value.finding_id, value)
                    for value in first_result.findings
                ),
            ),
            key=lambda identity: identity.object_id,
        )
    )
    snapshot = EvidenceSnapshot(
        snapshot_id=prepared.snapshot.snapshot_id,
        campaign_id=prepared.snapshot.campaign_id,
        run_id=prepared.snapshot.run_id,
        world_id=prepared.snapshot.world_id,
        parent_objects=parent_objects,
        parent_visibility_ceilings=tuple(
            VisibilityCeiling.OUTCOME_VISIBLE for _ in parent_objects
        ),
        artifacts=prepared.snapshot.artifacts,
        claim_ids=prepared.snapshot.claim_ids,
        information_cutoff=prepared.snapshot.information_cutoff,
        outcome_access=prepared.snapshot.outcome_access,
        visibility_ceiling=prepared.snapshot.visibility_ceiling,
        projection_ids=prepared.snapshot.projection_ids,
    )
    view = bind_evidence_view(snapshot, prepared.projection)
    verification = SnapshotVerification(
        verification_id=prepared.verification.verification_id,
        snapshot=ObjectIdentity.from_record(snapshot.snapshot_id, snapshot),
        artifact_ids=prepared.verification.artifact_ids,
        verification_check_ids=prepared.verification.verification_check_ids,
        verifier_key=prepared.verification.verifier_key,
        verifier_version=prepared.verification.verifier_version,
    )
    context = TemplateContext(
        context_id="template-context.exploration.adaptive-second-wave",
        snapshot=ObjectIdentity.from_record(snapshot.snapshot_id, snapshot),
        evidence_view=ObjectIdentity.from_record(view.view_id, view),
        search_axes=_axes(
            {
                "history-ladder": ("current-state",),
                "horizon": ("short",),
                "representation": ("native",),
                "time-direction": ("forward",),
                "transform": ("native",),
            }
        ),
        quantity_reporting=tuple(
            QuantityReportingObligation(
                quantity_id=quantity.quantity_id,
                native_unit=quantity.native_unit,
                normalized_reporting_allowed=True,
            )
            for quantity in prepared.world.system.quantities
            if quantity.quantity_id
            in {
                *prepared.world.system.relation.action_quantity_ids,
                *prepared.world.system.relation.receiver_quantity_ids,
            }
        ),
        analysis_budget=_budget(),
        executed_template_ids=("counterfeit-rank-representation-preprocessing",),
    )
    signals = default_detector_registry().detect(snapshot, prepared.projection, view)
    instantiations = instantiate_matching_templates(snapshot, view, signals, context)
    candidates = build_portfolio_candidates(
        instantiations,
        signals,
        view,
        executed_template_ids=context.executed_template_ids,
    )
    planned = ExplorationPortfolioPlanner().select(
        plan_id="exploration-adaptive-second-wave",
        snapshot=snapshot,
        snapshot_verification=verification,
        candidates=candidates,
        policy=default_portfolio_policy(),
    )
    return _PreparedExploration(
        world=prepared.world,
        evaluation=prepared.evaluation,
        projection=prepared.projection,
        snapshot=snapshot,
        view=view,
        verification=verification,
        signals=signals,
        instantiations=instantiations,
        planned=planned,
        parent_fingerprint_before=prepared.parent_fingerprint_before,
    )


def _ledger(
    wave_index: int,
    prepared: _PreparedExploration,
    result: ExplorationWaveResult,
    parent: str | None,
) -> ExplorationLifecycleLedger:
    plan = prepared.planned.plan
    if plan is None:
        raise ValueError("executed wave ledger requires a plan")
    return ExplorationLifecycleLedger(
        ledger_id=f"ledger.{plan.plan_id}",
        wave_index=wave_index,
        parent_wave_ledger_id=parent,
        snapshot=ObjectIdentity.from_record(
            prepared.snapshot.snapshot_id, prepared.snapshot
        ),
        plan=ObjectIdentity.from_record(plan.plan_id, plan),
        decision=ObjectIdentity.from_record(
            prepared.planned.decision.decision_id,
            prepared.planned.decision,
        ),
        selections=plan.selections,
        attempts=result.attempts,
        findings=result.findings,
        hypothesis_syntheses=(result.hypothesis_synthesis,),
        outcome_access=plan.outcome_access,
        visibility_ceiling=plan.visibility_ceiling,
    )


def _reference_campaign(first: _PreparedExploration) -> CampaignSpec:
    root_node = CampaignNode(
        node_id="exploration-campaign-root-snapshot",
        kind=CampaignNodeKind.EVIDENCE_SNAPSHOT,
        lane=CampaignLane.EXPLORATORY,
        object_identity=ObjectIdentity.from_record(
            first.snapshot.snapshot_id, first.snapshot
        ),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.TERMINAL,
    )
    return CampaignSpec(
        campaign_id="exploration-reference-campaign",
        objective="Automate bounded outcome-visible discovery without scientific promotion.",
        system_ids=(first.world.system.system_id,),
        world_ids=(first.world.reference_id,),
        target_claim_ids=("exploration-reference-question",),
        budget=default_portfolio_policy().budget,
        authority_policy=ObjectIdentity.from_record(
            first.world.system.authority_policy.policy_id,
            first.world.system.authority_policy,
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id="exploration-read-only-exploration-right",
                action=AuthorityAction.READ_ONLY_EXPLORATION,
                decision_maker_id=first.world.system.authority_policy.delegate_id,
                authority_policy_id=first.world.system.authority_policy.policy_id,
                delegated=True,
            ),
        ),
        nodes=(root_node,),
        root_node_ids=(root_node.node_id,),
        active_node_ids=(root_node.node_id,),
        evidence_state=(root_node.object_identity,),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )


def _append_node(
    nodes: list[DiscoveryNode],
    node_id: str,
    kind: DiscoveryNodeKind,
    value: CanonicalRecord,
    object_id: str,
) -> None:
    nodes.append(
        DiscoveryNode(
            node_id=node_id,
            kind=kind,
            lane=DiscoveryLane.EXPLORATORY,
            object_identity=ObjectIdentity.from_record(object_id, value),
            lifecycle_status=LifecycleStatus.TERMINAL,
        )
    )


def _append_edge(
    edges: list[DiscoveryEdge],
    edge_id: str,
    kind: DiscoveryEdgeKind,
    source: str,
    target: str,
    reasons: tuple[str, ...] = (),
) -> None:
    edges.append(
        DiscoveryEdge(
            edge_id=edge_id,
            kind=kind,
            source_node_id=source,
            target_node_id=target,
            reason_codes=tuple(sorted(reasons)),
        )
    )


def _append_wave(
    nodes: list[DiscoveryNode],
    edges: list[DiscoveryEdge],
    prefix: str,
    prepared: _PreparedExploration,
    result: ExplorationWaveResult,
    sequential_parent: str | None,
) -> str:
    snapshot_node = f"{prefix}.snapshot"
    _append_node(
        nodes,
        snapshot_node,
        DiscoveryNodeKind.EVIDENCE_SNAPSHOT,
        prepared.snapshot,
        prepared.snapshot.snapshot_id,
    )
    if sequential_parent is not None:
        _append_edge(
            edges,
            f"{prefix}.edge.sequential-parent",
            DiscoveryEdgeKind.SEQUENTIAL_PARENT,
            sequential_parent,
            snapshot_node,
        )
    signal_nodes = {}
    for signal in prepared.signals:
        signal_node = f"{prefix}.signal.{signal.signal_id}"
        signal_nodes[signal.signal_id] = signal_node
        _append_node(
            nodes,
            signal_node,
            DiscoveryNodeKind.ANOMALY_SIGNAL,
            signal,
            signal.signal_id,
        )
        _append_edge(
            edges,
            f"{prefix}.edge.snapshot-to-{signal.signal_id}",
            DiscoveryEdgeKind.MOTIVATED_BY,
            snapshot_node,
            signal_node,
        )
    plan = prepared.planned.plan
    if plan is None:
        raise ValueError("campaign graph wave unexpectedly stopped")
    plan_node = f"{prefix}.plan"
    _append_node(
        nodes, plan_node, DiscoveryNodeKind.EXPLORATION_PLAN, plan, plan.plan_id
    )
    _append_proposals(nodes, edges, prefix, plan, plan_node, signal_nodes)
    finding_nodes = _append_findings(nodes, edges, prefix, plan_node, result)
    hypothesis_set = result.hypothesis_synthesis.hypothesis_set
    hypothesis_node = f"{prefix}.hypothesis-set"
    _append_node(
        nodes,
        hypothesis_node,
        DiscoveryNodeKind.HYPOTHESIS_SET,
        hypothesis_set,
        hypothesis_set.hypothesis_set_id,
    )
    for finding_node in finding_nodes:
        _append_edge(
            edges,
            f"{prefix}.edge.{finding_node}-to-hypotheses",
            DiscoveryEdgeKind.MOTIVATED_BY,
            finding_node,
            hypothesis_node,
        )
    return hypothesis_node


def _append_proposals(
    nodes: list[DiscoveryNode],
    edges: list[DiscoveryEdge],
    prefix: str,
    plan: ExplorationPlan,
    plan_node: str,
    signal_nodes: dict[str, str],
) -> None:
    selection_by_id = {value.proposal_id: value for value in plan.selections}
    disposition_edges = {
        ProposalDisposition.SELECTED: DiscoveryEdgeKind.SELECTED,
        ProposalDisposition.REJECTED: DiscoveryEdgeKind.REJECTED,
        ProposalDisposition.BLOCKED: DiscoveryEdgeKind.BLOCKED,
        ProposalDisposition.SUPERSEDED: DiscoveryEdgeKind.SUPERSEDED,
    }
    for proposal in plan.proposals:
        proposal_node = f"{prefix}.proposal.{proposal.proposal_id}"
        _append_node(
            nodes,
            proposal_node,
            DiscoveryNodeKind.ANALYSIS_PROPOSAL,
            proposal,
            proposal.proposal_id,
        )
        for signal_id in proposal.anomaly_signal_ids:
            _append_edge(
                edges,
                f"{prefix}.edge.{signal_id}-to-{proposal.proposal_id}",
                DiscoveryEdgeKind.MOTIVATED_BY,
                signal_nodes[signal_id],
                proposal_node,
            )
        selection = selection_by_id[proposal.proposal_id]
        _append_edge(
            edges,
            f"{prefix}.edge.{proposal.proposal_id}-disposition",
            disposition_edges[selection.disposition],
            proposal_node,
            plan_node,
            selection.reason_codes,
        )


def _append_findings(
    nodes: list[DiscoveryNode],
    edges: list[DiscoveryEdge],
    prefix: str,
    plan_node: str,
    result: ExplorationWaveResult,
) -> tuple[str, ...]:
    finding_nodes = []
    for finding in result.findings:
        finding_node = f"{prefix}.finding.{finding.finding_id}"
        finding_nodes.append(finding_node)
        _append_node(
            nodes,
            finding_node,
            DiscoveryNodeKind.EXPLORATORY_FINDING,
            finding,
            finding.finding_id,
        )
        _append_edge(
            edges,
            f"{prefix}.edge.plan-to-{finding.finding_id}",
            DiscoveryEdgeKind.EXECUTED,
            plan_node,
            finding_node,
        )
    return tuple(finding_nodes)


def _campaign_graph(
    first: _PreparedExploration,
    first_result: ExplorationWaveResult,
    second: _PreparedExploration,
    second_result: ExplorationWaveResult,
) -> CampaignGraph:
    campaign = _reference_campaign(first)
    nodes: list[DiscoveryNode] = []
    edges: list[DiscoveryEdge] = []
    first_hypothesis = _append_wave(nodes, edges, "wave-1", first, first_result, None)
    second_hypothesis = _append_wave(
        nodes,
        edges,
        "wave-2",
        second,
        second_result,
        first_hypothesis,
    )
    return CampaignGraph(
        graph_id="exploration-reference-campaign-graph",
        campaign=ObjectIdentity.from_record(campaign.campaign_id, campaign),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
        root_node_ids=("wave-1.snapshot",),
        active_node_ids=(second_hypothesis,),
    )


def run_exploration_reference_conformance(
    registry_fingerprint: str,
) -> ExplorationReferenceConformanceReport:
    validate_sha256(registry_fingerprint, field_name="registry_fingerprint")
    planted_world = get_reference_world(ReferenceWorldKind.PLANTED_RELATIONAL_ANOMALY)
    planted = _prepare(
        case_id="planted-decoy",
        kind=ReferenceWorldKind.PLANTED_RELATIONAL_ANOMALY,
        measures=(
            _measure(
                planted_world,
                "coordinate-instability",
                DiagnosticMeasureKind.COORDINATE_INSTABILITY,
                "0.7",
                "0.2",
                ThresholdDirection.ABOVE_MAXIMUM,
                12,
            ),
            _measure(
                planted_world,
                "rank-instability",
                DiagnosticMeasureKind.RANK_INSTABILITY,
                "0.6",
                "0.2",
                ThresholdDirection.ABOVE_MAXIMUM,
                12,
            ),
        ),
        axis_overrides={
            "representation": (
                "attractive-decoy",
                "null-candidate",
                "planted-relation",
            ),
            "transform": ("native",),
        },
    )
    planted_result = _execute_planted(planted)

    null_world = get_reference_world(ReferenceWorldKind.NULL_SEARCH_FAMILY)
    broad_null = _prepare(
        case_id="broad-null",
        kind=ReferenceWorldKind.NULL_SEARCH_FAMILY,
        measures=(
            _measure(
                null_world,
                "broad-null-information",
                DiagnosticMeasureKind.EFFECTIVE_INDEPENDENT_UNITS,
                "12",
                "20",
                ThresholdDirection.BELOW_MINIMUM,
                12,
            ),
        ),
        axis_overrides={
            "design-sensitivity": tuple(
                f"candidate-{index:02d}" for index in range(1, 13)
            ),
            "receiver": ("receiver",),
        },
    )
    broad_null_result = _execute_uniform(broad_null, evaluable=True)

    information_world = get_reference_world(ReferenceWorldKind.LATENCY_BOUNDARY)
    information_limit = _prepare(
        case_id="information-limit",
        kind=ReferenceWorldKind.LATENCY_BOUNDARY,
        measures=(
            _measure(
                information_world,
                "independent-unit-information",
                DiagnosticMeasureKind.EFFECTIVE_INDEPENDENT_UNITS,
                "2",
                "8",
                ThresholdDirection.BELOW_MINIMUM,
                2,
            ),
        ),
        axis_overrides={
            "design-sensitivity": ("current-design", "fresh-eight-units"),
            "receiver": ("receiver",),
        },
    )
    information_result = _execute_uniform(information_limit, evaluable=False)

    stable_world = get_reference_world(ReferenceWorldKind.STABLE_LINEAR)
    stop = _prepare(
        case_id="stop-no-analysis",
        kind=ReferenceWorldKind.STABLE_LINEAR,
        measures=(
            _measure(
                stable_world,
                "adequate-support",
                DiagnosticMeasureKind.SUPPORT_COVERAGE,
                "1",
                "0.8",
                ThresholdDirection.BELOW_MINIMUM,
                16,
            ),
        ),
    )
    second = _second_wave(planted, planted_result)
    second_result = _execute_uniform(
        second,
        evaluable=True,
        operational_failure_template_id="relaxation-propagator-semigroup-arrow",
    )
    first_ledger = _ledger(1, planted, planted_result, None)
    second_ledger = _ledger(2, second, second_result, first_ledger.ledger_id)
    graph = _campaign_graph(planted, planted_result, second, second_result)
    outcomes = (
        _outcome("adaptive-second-wave", second, second_result),
        _outcome("broad-null", broad_null, broad_null_result),
        _outcome("information-limit", information_limit, information_result),
        _outcome("planted-decoy", planted, planted_result),
        _outcome("stop-no-analysis", stop, None),
    )
    return ExplorationReferenceConformanceReport(
        report_id="platform-remediation-exploration-reference-conformance",
        registry_fingerprint=registry_fingerprint,
        outcomes=outcomes,
        campaign_graph=graph,
        wave_ledgers=tuple(
            sorted((first_ledger, second_ledger), key=lambda ledger: ledger.ledger_id)
        ),
        physical_observation_to_controller_use_execution="NONE",
        status="PASS",
    )
