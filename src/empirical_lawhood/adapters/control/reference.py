"""Disk-independent truth-known world for sole-route controller conformance."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.adapters.control.evidence_services import AtlasAdmissionDeriver, AtlasReachabilityDeriver
from empirical_lawhood.adapters.control.composition import ControllerStudyComposition, controller_capability_registry
from empirical_lawhood.adapters.geometry.conformance import _atlas, _model_set
from empirical_lawhood.kernel.action_contracts import ActionChannelBinding, ActionDeliveryStage, ActionOccurrence, ActionOccurrenceGroup, ActionOccurrenceOrder, ActionStageEvent, ActionStageQuantityBinding, OccurrenceActionWord, ActionWordMode, ActionWordSupportStatus, ObservedActionOccurrence
from empirical_lawhood.kernel.admission import AdmissionGateKind, GateStatus
from empirical_lawhood.kernel.atlases import ResponseAtlas
from empirical_lawhood.kernel.causal_contracts import ActionStageReceiverTransport, CausalCompositionDisposition, CausalConeExistence, CausalPrefixAssessment, ActionStageCausalSupportAssessment, DeliveredActionValidity, EvidenceEvaluability, NumericalViewAgreement, PredicateDirection, PrefixSupportStatus, ReceiverInterval, ResponseDirectionStatus, TemporalPredicateAssessment, TemporalPredicateSemantics
from empirical_lawhood.kernel.control import OperationalDeliveryState, ScientificCommitmentKind
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.models import ViewModelSetSpec
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import (
    CausalPhase,
    ClockCoordinate,
    ClockTransport,
    ClockTransportAvailability,
    ClockTransportKind,
    ClockTransportMonotonicity,
    CoordinateOrigin,
    InformationCutoff,
)
from empirical_lawhood.planning.controller_study import AtlasActionCandidate, CandidateGateReceipt, CandidatePriorityOrientation, CandidateReachabilityReceipt, CandidateUtilityDefinition, CandidateUtilityReceipt, AtlasControllerStudy, AdmissionControllerStudy, AtlasControllerSynthesisPlan, DeliveryEquivalenceSpec, EvaluationStratumSpec, EvaluatorBoundarySpec, AtlasActionCandidateChart, ImplementationBinding, ImplementationRole, AtlasMeasuredHoldFibre, OnlineSupportMonitorSpec, ControllerActionBinding, ProspectiveControllerEvaluationPlan, ReidentificationTriggerSpec, UtilityDirection
from empirical_lawhood.planning.evidence_geometry import BaselinePreservationCompatibility, DirectionConstraintReceipt, AtlasAdmissionSpec, AtlasReachabilitySpec, AtlasGateReceipt, AdmissionCoordinateGateReceipt, GatePredicateKind, GatePredicateSpec, PreservationCompatibilityRule, ReachabilityCellDisposition, AtlasReachabilityReceipt, ReachabilityRepresentationStatus, derive_atlas_admission_comparison, derive_admission_gate_observation, derive_atlas_reachability_comparison
from empirical_lawhood.planning.geometry import AdmissionCandidateCell, AdmissionComparison
from empirical_lawhood.runtime.controller_compiler import CompiledAtlasControllerStudy
from empirical_lawhood.runtime.controller_evaluation import (
    ControllerUseEvaluator,
    ProspectiveControllerBundle,
    RevealedBranchOutcome,
    RevealedProspectiveControllerBundle,
    SealedOutcomeLocator,
)
from empirical_lawhood.runtime.controller_runtime import AtlasControllerTickReceipt, DeliveryPortResult, AtlasLiveGateReceipt, AdmissionLiveGateReceipt, ObserverDisposition, ObserverEvaluation, ExactActionDeliveryTrace, AtlasLiveGateEvaluation, AdmissionLiveGateEvaluation, RuntimeObservation


REFERENCE_DECISION_CELL_ID = "reference-decision-cell"
ACTIVE_A_CELL_ID = "reference-active-a-cell"
ACTIVE_B_CELL_ID = "reference-active-b-cell"
HOLD_CELL_ID = "reference-hold-cell"
ACTIVE_A_BINDING_ID = "action-binding.reference-active-a"
ACTIVE_B_BINDING_ID = "action-binding.reference-active-b"
HOLD_BINDING_ID = "action-binding.reference-hold"
HOLD_FIBRE_ID = "measured-hold.reference"


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _implementations(reference: ExecutableReference) -> tuple[ImplementationBinding, ...]:
    return tuple(
        sorted(
            (
                ImplementationBinding(
                    binding_id=f"implementation.{role.value.lower().replace('_', '-')}",
                    role=role,
                    reference=replace(
                        reference,
                        reference_id=f"reference.{role.value.lower().replace('_', '-')}",
                        capability_key=f"control.{role.value.lower().replace('_', '-')}",
                        evaluator_key=f"evaluator.{role.value.lower().replace('_', '-')}",
                    ),
                    config_sha256=_digest(f"current-reference-config:{role.value}"),
                    implementation_sha256=_digest(f"current-reference-code:{role.value}"),
                )
                for role in ImplementationRole
            ),
            key=lambda value: value.binding_id,
        )
    )


def _coordinate(value: str) -> ClockCoordinate:
    return ClockCoordinate(
        clock_id="reference-clock",
        coordinate=Decimal(value),
        time_unit="s",
        coordinate_frame="reference-time",
        origin=CoordinateOrigin.EPISODE_RELATIVE,
    )


def _interval() -> ReceiverInterval:
    return ReceiverInterval(start=_coordinate("0"), end=_coordinate("3"))


def _transport(
    evaluator: ExecutableReference,
    *,
    transport_id: str,
    offset: str = "0",
    tolerance: str = "0",
) -> ClockTransport:
    exact = Decimal(offset) == 0 and Decimal(tolerance) == 0
    return ClockTransport(
        transport_id=transport_id,
        source_clock_id="reference-clock",
        target_clock_id="reference-clock",
        source_time_unit="s",
        target_time_unit="s",
        source_coordinate_frame="reference-time",
        target_coordinate_frame="reference-time",
        source_origin=CoordinateOrigin.EPISODE_RELATIVE,
        target_origin=CoordinateOrigin.EPISODE_RELATIVE,
        kind=ClockTransportKind.IDENTITY if exact else ClockTransportKind.AFFINE_BOUNDED,
        availability=ClockTransportAvailability.AVAILABLE,
        scale=Decimal(1),
        offset=Decimal(offset),
        tolerance=Decimal(tolerance),
        valid_source_lower=Decimal(0),
        valid_source_upper=Decimal(10),
        monotonicity=ClockTransportMonotonicity.STRICTLY_INCREASING,
        evaluator=evaluator,
        reason_codes=(),
    )


def _evidence(*, atlas: ResponseAtlas, evaluator: ExecutableReference, suffix: str) -> EvidenceLink:
    return EvidenceLink(
        link_id=f"evidence.current-reference.{suffix}",
        relation=EvidenceRelation.DERIVED_FROM,
        source=ObjectIdentity.from_record(atlas.atlas_id, atlas),
        target=ObjectIdentity.from_record(
            atlas.laws[0].relation.relation_id, atlas.laws[0].relation
        ),
        artifact_ids=(evaluator.payload.artifact_id,),
        world_id=atlas.world_id,
        information_cutoff_id=f"cutoff.current-reference.{suffix}",
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        reason="Truth-known synthetic bytes support controller conformance only.",
    )


def _temporal_obligation(
    evaluator: ExecutableReference, evidence: EvidenceLink, *, suffix: str
) -> TemporalPredicateAssessment:
    return TemporalPredicateAssessment(
        obligation_id=f"reference-receiver-preservation.{suffix}",
        semantics=TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH,
        receiver_id="receiver",
        direction=PredicateDirection.AT_LEAST,
        criterion=NamedDecimal(
            value_id=f"reference-receiver-minimum.{suffix}",
            value=Decimal(0),
            unit="1",
        ),
        protected_domain=_interval(),
        evaluated_domain=_interval(),
        status=ObligationStatus.SATISFIED,
        first_failure=None,
        satisfaction_coordinate=None,
        required_numerical_view_ids=("fine-view",),
        evaluator=evaluator,
        physical_independent_unit_id="reference-preparation",
        physical_independent_unit_count=8,
        evidence_links=(evidence,),
        reason_codes=(),
    )


def _predicate(
    evaluator: ExecutableReference,
    *,
    cell_id: str,
    kind: AdmissionGateKind,
    obligation: TemporalPredicateAssessment,
) -> GatePredicateSpec:
    lower = (
        obligation.criterion
        if kind is AdmissionGateKind.BASELINE_PRESERVATION
        else NamedDecimal(
            value_id=f"{cell_id}.{kind.value.lower()}-minimum",
            value=Decimal(0),
            unit="1",
        )
    )
    return GatePredicateSpec(
        predicate_id=f"predicate.{cell_id}.{kind.value.lower()}",
        gate_kind=kind,
        receiver_id="receiver",
        direction=PredicateDirection.AT_LEAST,
        quantity_id=f"quantity.{cell_id}.{kind.value.lower()}",
        protected_interval=obligation.protected_domain,
        predicate_kind=GatePredicateKind.SCALAR_AT_LEAST,
        lower=lower,
        upper=None,
        expected_boolean=None,
        expected_identity=None,
        temporal_semantics=(
            obligation.semantics if kind is AdmissionGateKind.BASELINE_PRESERVATION else None
        ),
        constraint_ids=(f"constraint-{kind.value.lower()}",),
        evaluator=evaluator,
    )


def _gate_receipt(
    evaluator: ExecutableReference,
    evidence: EvidenceLink,
    obligation: TemporalPredicateAssessment,
    *,
    cell_id: str,
    member: str,
    kind: AdmissionGateKind,
    passed: bool = True,
    suffix: str = "",
) -> AtlasGateReceipt:
    predicate = _predicate(evaluator, cell_id=cell_id, kind=kind, obligation=obligation)
    compatibility = (
        BaselinePreservationCompatibility(
            compatibility_id=f"compatibility.{cell_id}.{member}",
            gate_predicate=predicate,
            obligation=obligation,
            rule=PreservationCompatibilityRule.EXACT_SEMANTIC_EQUALITY,
            evaluator=evaluator,
            evidence_link_ids=(evidence.link_id,),
            compatible=True,
            reason_codes=(),
        )
        if kind is AdmissionGateKind.BASELINE_PRESERVATION
        else None
    )
    observed = Decimal(1) if passed else Decimal(-1)
    if predicate.lower is None:
        raise AssertionError("reference predicate lost lower bound")
    receipt_id = f"receipt.{cell_id}.{member}.{kind.value.lower()}{suffix}"
    margin = observed - predicate.lower.value
    return AtlasGateReceipt(
        receipt_id=receipt_id,
        cell_id=cell_id,
        model_member_id=member,
        numerical_view_id=member,
        predicate=predicate,
        observed_scalar=NamedDecimal(value_id=f"observed.{receipt_id}", value=observed, unit="1"),
        observed_boolean=None,
        observed_identity=None,
        information_cutoff=InformationCutoff(
            cutoff_id=evidence.information_cutoff_id,
            clock_id="reference-clock",
            phase=CausalPhase.PRE_ACTION,
            coordinate=Decimal("0.5"),
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        evaluator=evaluator,
        input_artifacts=(evaluator.payload,),
        evidence_links=(evidence,),
        baseline_compatibility=compatibility,
        status=GateStatus.PASS if passed else GateStatus.FAIL,
        margin=NamedDecimal(value_id=f"margin.{receipt_id}", value=margin, unit="1"),
        reason_codes=() if passed else ("GATE_PREDICATE_FAILED",),
    )


def _admission_spec(
    *,
    atlas: ResponseAtlas,
    model_set: ViewModelSetSpec,
    evaluator: ExecutableReference,
    evidence: EvidenceLink,
    obligation: TemporalPredicateAssessment,
    fail_high_utility_sink: bool,
) -> AtlasAdmissionSpec:
    cells = (
        AdmissionCandidateCell(
            cell_id=ACTIVE_A_CELL_ID,
            denominator_cell_id="response-method-cell-a",
            chart_id="response-method-local-chart",
            action_bound_ids=("bound.active-a",),
        ),
        AdmissionCandidateCell(
            cell_id=ACTIVE_B_CELL_ID,
            denominator_cell_id="response-method-cell-a",
            chart_id="response-method-local-chart",
            action_bound_ids=("bound.active-b",),
        ),
        AdmissionCandidateCell(
            cell_id=HOLD_CELL_ID,
            denominator_cell_id="response-method-cell-a",
            chart_id="response-method-local-chart",
            action_bound_ids=("bound.hold",),
        ),
    )
    receipts = tuple(
        sorted(
            (
                _gate_receipt(
                    evaluator,
                    evidence,
                    obligation,
                    cell_id=cell.cell_id,
                    member=member,
                    kind=kind,
                    passed=not (
                        fail_high_utility_sink
                        and cell.cell_id == ACTIVE_A_CELL_ID
                        and kind is AdmissionGateKind.PHYSICAL_SINK
                    ),
                )
                for cell in cells
                for member in model_set.member_view_ids
                for kind in AdmissionGateKind
            ),
            key=lambda value: value.receipt_id,
        )
    )
    return AtlasAdmissionSpec(
        evaluation_id="admission.current-reference-controller",
        atlas=atlas,
        model_set=model_set,
        nominal_model_member_id="coarse-view",
        receiver_quantity_ids=("receiver",),
        candidate_cells=cells,
        receipts=receipts,
        evidence_links=(evidence,),
        evidence_ceiling=EvidenceCeiling.ADMISSION,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _reachability_spec(
    *,
    admission: AdmissionComparison,
    model_set: ViewModelSetSpec,
    evaluator: ExecutableReference,
    evidence: EvidenceLink,
    obligation: TemporalPredicateAssessment,
    law: ResponseLaw,
) -> AtlasReachabilitySpec:
    receipts: list[AtlasReachabilityReceipt] = []
    for cell in admission.robust.cells:
        for member in model_set.member_view_ids:
            gate = _gate_receipt(
                evaluator,
                evidence,
                obligation,
                cell_id=cell.cell_id,
                member=member,
                kind=AdmissionGateKind.REACHABILITY,
                suffix="-direction",
            )
            direction = DirectionConstraintReceipt(
                direction_id="positive-action-direction", gate_receipts=(gate,)
            )
            receipts.append(
                AtlasReachabilityReceipt(
                    receipt_id=f"reachability.{cell.cell_id}.{member}",
                    admission_cell_id=cell.cell_id,
                    model_member_id=member,
                    numerical_view_id=member,
                    admission=ObjectIdentity.from_record(admission.comparison_id, admission),
                    initial_set_id="reference-initial-set",
                    action_chart_ids=("response-method-local-chart",),
                    dynamics_law_ids=(law.law_id,),
                    horizon=law.relation.horizon,
                    constraint_ids=("constraint-reachability",),
                    structural_convergence=law.obligations.structural_convergence,
                    computability=law.obligations.computability,
                    method=evaluator,
                    information_cutoff=gate.information_cutoff,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    representation_status=ReachabilityRepresentationStatus.AVAILABLE,
                    candidate_direction_ids=(direction.direction_id,),
                    direction_receipts=(direction,),
                    independent_basis_direction_ids=(direction.direction_id,),
                    input_artifacts=(evaluator.payload,),
                    evidence_links=(evidence,),
                    status=ReachabilityCellDisposition.REACHABLE,
                    viable_direction_rank=1,
                    reason_codes=(),
                )
            )
    return AtlasReachabilitySpec(
        evaluation_id="reachability.current-reference-controller",
        admission=admission,
        model_set=model_set,
        nominal_model_member_id="coarse-view",
        initial_set_id="reference-initial-set",
        action_chart_ids=("response-method-local-chart",),
        dynamics_law_ids=(law.law_id,),
        horizon=law.relation.horizon,
        constraint_ids=("constraint-reachability",),
        numerical_view_ids=model_set.member_view_ids,
        structural_convergence=law.obligations.structural_convergence,
        computability=law.obligations.computability,
        method=evaluator,
        receipts=tuple(sorted(receipts, key=lambda value: value.receipt_id)),
        evidence_links=(evidence,),
        evidence_ceiling=EvidenceCeiling.ADMISSION,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def _action_word(
    evaluator: ExecutableReference,
    *,
    word_id: str,
    occurrence_id: str,
    value: Decimal,
) -> OccurrenceActionWord:
    channel = ActionChannelBinding(
        binding_id=f"channel.{word_id}",
        port_id="action-in",
        controller_quantity_id="action",
        stages=tuple(
            ActionStageQuantityBinding(
                stage=stage,
                quantity_id=f"action-{stage.value.lower()}",
                clock_id="reference-clock",
                time_unit="s",
                clock_coordinate_frame="reference-time",
                clock_origin=CoordinateOrigin.EPISODE_RELATIVE,
            )
            for stage in ActionDeliveryStage
        ),
        native_unit="1",
        native_action_frame="reference-action-frame",
        native_direction="positive-action",
        support_contract_id=f"support.{word_id}",
        delivery_contract_id=f"delivery.{word_id}",
    )

    def event(stage: ActionDeliveryStage, coordinate: str) -> ActionStageEvent:
        return ActionStageEvent(
            stage=stage,
            quantity_id=f"action-{stage.value.lower()}",
            value=value,
            native_unit="1",
            native_action_frame="reference-action-frame",
            native_direction="positive-action",
            coordinate=_coordinate(coordinate),
        )

    occurrence = ActionOccurrence(
        occurrence_id=occurrence_id,
        channel=channel,
        requested=event(ActionDeliveryStage.REQUESTED, "1"),
        accepted=event(ActionDeliveryStage.ACCEPTED, "1.1"),
        applied=event(ActionDeliveryStage.APPLIED, "1.2"),
        realized=event(ActionDeliveryStage.REALIZED, "1.3"),
        requested_to_accepted=_transport(
            evaluator,
            transport_id=f"transport.{word_id}.requested-accepted",
            offset="0.1",
            tolerance="0.001",
        ),
        accepted_to_applied=_transport(
            evaluator,
            transport_id=f"transport.{word_id}.accepted-applied",
            offset="0.1",
            tolerance="0.001",
        ),
        applied_to_realized=_transport(
            evaluator,
            transport_id=f"transport.{word_id}.applied-realized",
            offset="0.1",
            tolerance="0.001",
        ),
        duration=Decimal("0.5"),
        duration_unit="s",
    )
    return OccurrenceActionWord(
        word_id=word_id,
        mode=ActionWordMode.SEQUENTIAL,
        occurrences=(occurrence,),
        groups=(
            ActionOccurrenceGroup(
                group_id=f"group.{word_id}",
                members=(
                    ActionOccurrenceOrder(
                        occurrence_id=occurrence_id,
                        transport=_transport(
                            evaluator, transport_id=f"transport.{word_id}.applied-order"
                        ),
                    ),
                ),
            ),
        ),
        ordering_clock_id="reference-clock",
        ordering_time_unit="s",
        ordering_coordinate_frame="reference-time",
        ordering_origin=CoordinateOrigin.EPISODE_RELATIVE,
        denominator_id="response-method-cell-a",
        retained_history_id="history-none",
        receiver_id="receiver",
        horizon_id="reference-horizon",
        prefix_support_ids=("prefix-identity", f"prefix.{word_id}"),
        support_status=ActionWordSupportStatus.SUPPORTED,
        reason_codes=(),
    )


def _action_binding(
    *,
    binding_id: str,
    word: OccurrenceActionWord,
    system: SystemSpec,
    evidence: EvidenceLink,
    obligation: TemporalPredicateAssessment,
) -> ControllerActionBinding:
    support = ActionStageCausalSupportAssessment(
        assessment_id=f"causal-support.{binding_id}",
        relation=system.relation,
        world_id=system.world.world_id,
        action_word=word,
        receiver_id="receiver",
        receiver_clock_id="reference-clock",
        receiver_time_unit="s",
        receiver_coordinate_frame="reference-time",
        receiver_origin=CoordinateOrigin.EPISODE_RELATIVE,
        stage_transports=tuple(
            ActionStageReceiverTransport(
                occurrence_id=word.occurrences[0].occurrence_id,
                stage=stage,
                transport=_transport(
                    word.occurrences[0].requested_to_accepted.evaluator,
                    transport_id=f"transport.{binding_id}.{stage.value.lower()}-receiver",
                ),
            )
            for stage in ActionDeliveryStage
        ),
        earliest_influence=_coordinate("2"),
        latest_influence=_coordinate("5"),
        evidence_evaluability=EvidenceEvaluability.EVALUABLE,
        delivered_action_validity=DeliveredActionValidity.VALID,
        causal_cone=CausalConeExistence.PRESENT,
        response_direction=ResponseDirectionStatus.SUPPORTED,
        physical_independent_unit_id="reference-preparation",
        physical_independent_unit_count=8,
        numerical_view_ids=("coarse-view", "fine-view"),
        evidence_links=(evidence,),
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        reason_codes=(),
    )
    prefix = CausalPrefixAssessment(
        assessment_id=f"causal-prefix.{binding_id}",
        causal_support=support,
        prefix_id=f"prepared-prefix.{binding_id}",
        prefix_cutoff=_coordinate("0.5"),
        prefix_support=PrefixSupportStatus.SUPPORTED,
        obligations=(obligation,),
        required_numerical_view_ids=("coarse-view", "fine-view"),
        numerical_view_agreement=NumericalViewAgreement.AGREED,
        obstructions=(),
        disposition=CausalCompositionDisposition.DEFINED,
        reason_codes=(),
    )
    return ControllerActionBinding(
        action_binding_id=binding_id, action_word=word, causal_prefix=prefix
    )


def _candidate_gate_receipts(
    *,
    candidate_id: str,
    action_binding_id: str,
    cell_id: str,
    admission: AtlasAdmissionSpec,
) -> tuple[CandidateGateReceipt, ...]:
    return tuple(
        sorted(
            (
                CandidateGateReceipt(
                    candidate_id=candidate_id,
                    action_binding_id=action_binding_id,
                    receipt=value,
                )
                for value in admission.receipts
                if value.cell_id == cell_id
            ),
            key=lambda value: value.receipt.receipt_id,
        )
    )


def _candidate_reachability_receipts(
    *,
    candidate_id: str,
    action_binding_id: str,
    cell_id: str,
    reachability: AtlasReachabilitySpec,
) -> tuple[CandidateReachabilityReceipt, ...]:
    return tuple(
        sorted(
            (
                CandidateReachabilityReceipt(
                    candidate_id=candidate_id,
                    action_binding_id=action_binding_id,
                    receipt=value,
                )
                for value in reachability.receipts
                if value.admission_cell_id == cell_id
            ),
            key=lambda value: value.receipt.receipt_id,
        )
    )


def _utility_receipts(
    *,
    candidate_id: str,
    action_binding_id: str,
    values: dict[str, Decimal],
    definition: CandidateUtilityDefinition,
    evidence: EvidenceLink,
) -> tuple[CandidateUtilityReceipt, ...]:
    return tuple(
        sorted(
            (
                CandidateUtilityReceipt(
                    receipt_id=f"utility.{candidate_id}.{member}",
                    candidate_id=candidate_id,
                    action_binding_id=action_binding_id,
                    model_member_id=member,
                    utility_definition=ObjectIdentity.from_record(
                        definition.definition_id, definition
                    ),
                    predicted_response=NamedDecimal(
                        value_id=f"predicted.{candidate_id}.{member}",
                        value=value,
                        unit="1",
                    ),
                    reference_response=NamedDecimal(
                        value_id=f"reference.{candidate_id}.{member}",
                        value=Decimal(0),
                        unit="1",
                    ),
                    uncertainty_allowance=NamedDecimal(
                        value_id=f"uncertainty.{candidate_id}.{member}",
                        value=Decimal(0),
                        unit="1",
                    ),
                    derived_utility=NamedDecimal(
                        value_id=f"utility-value.{candidate_id}.{member}",
                        value=value,
                        unit="1",
                    ),
                    information_cutoff=InformationCutoff(
                        cutoff_id=f"cutoff.utility.{candidate_id}.{member}",
                        clock_id="reference-clock",
                        phase=CausalPhase.PRE_ACTION,
                        coordinate=Decimal("0.5"),
                    ),
                    evaluator=definition.evaluator,
                    input_artifacts=(definition.evaluator.payload,),
                    evidence_links=(evidence,),
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                )
                for member, value in values.items()
            ),
            key=lambda value: value.receipt_id,
        )
    )


def build_current_reference_controller_study(
    *, fail_high_utility_sink: bool = False
) -> AtlasControllerStudy:
    system, laws, _links, atlas_result = _atlas()
    law = next(value for value in laws if value.chart_id == "response-method-local-chart")
    model_set = _model_set(laws)
    implementations = _implementations(law.evaluator)
    by_role = {value.role: value for value in implementations}
    admission_evidence = _evidence(
        atlas=atlas_result.atlas,
        evaluator=by_role[ImplementationRole.ADMISSION_DERIVER].reference,
        suffix="admission",
    )
    admission_obligation = _temporal_obligation(
        by_role[ImplementationRole.ADMISSION_DERIVER].reference,
        admission_evidence,
        suffix="admission",
    )
    admission_spec = _admission_spec(
        atlas=atlas_result.atlas,
        model_set=model_set,
        evaluator=by_role[ImplementationRole.ADMISSION_DERIVER].reference,
        evidence=admission_evidence,
        obligation=admission_obligation,
        fail_high_utility_sink=fail_high_utility_sink,
    )
    admission = derive_atlas_admission_comparison(admission_spec)
    reach_evidence = _evidence(
        atlas=atlas_result.atlas,
        evaluator=by_role[ImplementationRole.REACHABILITY_DERIVER].reference,
        suffix="reachability",
    )
    reach_obligation = _temporal_obligation(
        by_role[ImplementationRole.REACHABILITY_DERIVER].reference,
        reach_evidence,
        suffix="reachability",
    )
    reachability_spec = _reachability_spec(
        admission=admission,
        model_set=model_set,
        evaluator=by_role[ImplementationRole.REACHABILITY_DERIVER].reference,
        evidence=reach_evidence,
        obligation=reach_obligation,
        law=law,
    )
    reachability = derive_atlas_reachability_comparison(reachability_spec)
    causal_evidence = _evidence(atlas=atlas_result.atlas, evaluator=law.evaluator, suffix="causal")
    causal_obligation = _temporal_obligation(law.evaluator, causal_evidence, suffix="causal")
    words = {
        ACTIVE_A_BINDING_ID: _action_word(
            law.evaluator,
            word_id="word.reference-active-a",
            occurrence_id="occurrence.reference-active-a",
            value=Decimal("0.8"),
        ),
        ACTIVE_B_BINDING_ID: _action_word(
            law.evaluator,
            word_id="word.reference-active-b",
            occurrence_id="occurrence.reference-active-b",
            value=Decimal("0.5"),
        ),
        HOLD_BINDING_ID: _action_word(
            law.evaluator,
            word_id="word.reference-hold",
            occurrence_id="occurrence.reference-hold",
            value=Decimal(0),
        ),
    }
    action_bindings = tuple(
        sorted(
            (
                _action_binding(
                    binding_id=binding_id,
                    word=word,
                    system=system,
                    evidence=causal_evidence,
                    obligation=causal_obligation,
                )
                for binding_id, word in words.items()
            ),
            key=lambda value: value.action_binding_id,
        )
    )
    utility_evidence = _evidence(
        atlas=atlas_result.atlas,
        evaluator=by_role[ImplementationRole.SYNTHESIZER].reference,
        suffix="utility",
    )
    utility_definition = CandidateUtilityDefinition(
        definition_id="utility-definition.reference-controller",
        receiver_quantity_id="receiver",
        direction=UtilityDirection.HIGHER_IS_BETTER,
        reference_id="reference.zero-response",
        horizon_id=law.relation.horizon.horizon_id,
        native_unit="1",
        uncertainty_rule_id="uncertainty.exact-truth-known",
        minimum_robust_utility=NamedDecimal(
            value_id="minimum-robust-utility.reference", value=Decimal(0), unit="1"
        ),
        evaluator=by_role[ImplementationRole.SYNTHESIZER].reference,
    )
    candidate_a = AtlasActionCandidate(
        candidate_id="candidate.reference-active-a",
        decision_cell_id=REFERENCE_DECISION_CELL_ID,
        admission_cell_id=ACTIVE_A_CELL_ID,
        action_binding_id=ACTIVE_A_BINDING_ID,
        action_bound_ids=("bound.active-a",),
        model_member_ids=model_set.member_view_ids,
        gate_receipts=_candidate_gate_receipts(
            candidate_id="candidate.reference-active-a",
            action_binding_id=ACTIVE_A_BINDING_ID,
            cell_id=ACTIVE_A_CELL_ID,
            admission=admission_spec,
        ),
        reachability_receipts=_candidate_reachability_receipts(
            candidate_id="candidate.reference-active-a",
            action_binding_id=ACTIVE_A_BINDING_ID,
            cell_id=ACTIVE_A_CELL_ID,
            reachability=reachability_spec,
        ),
        utility_receipts=_utility_receipts(
            candidate_id="candidate.reference-active-a",
            action_binding_id=ACTIVE_A_BINDING_ID,
            values={"coarse-view": Decimal(5), "fine-view": Decimal(1)},
            definition=utility_definition,
            evidence=utility_evidence,
        ),
        priority_rank=0,
    )
    candidate_b = AtlasActionCandidate(
        candidate_id="candidate.reference-active-b",
        decision_cell_id=REFERENCE_DECISION_CELL_ID,
        admission_cell_id=ACTIVE_B_CELL_ID,
        action_binding_id=ACTIVE_B_BINDING_ID,
        action_bound_ids=("bound.active-b",),
        model_member_ids=model_set.member_view_ids,
        gate_receipts=_candidate_gate_receipts(
            candidate_id="candidate.reference-active-b",
            action_binding_id=ACTIVE_B_BINDING_ID,
            cell_id=ACTIVE_B_CELL_ID,
            admission=admission_spec,
        ),
        reachability_receipts=_candidate_reachability_receipts(
            candidate_id="candidate.reference-active-b",
            action_binding_id=ACTIVE_B_BINDING_ID,
            cell_id=ACTIVE_B_CELL_ID,
            reachability=reachability_spec,
        ),
        utility_receipts=_utility_receipts(
            candidate_id="candidate.reference-active-b",
            action_binding_id=ACTIVE_B_BINDING_ID,
            values={"coarse-view": Decimal(3), "fine-view": Decimal(3)},
            definition=utility_definition,
            evidence=utility_evidence,
        ),
        priority_rank=1,
    )
    chart = AtlasActionCandidateChart(
        chart_id="candidate-chart.reference-controller",
        model_member_ids=model_set.member_view_ids,
        nominal_member_id="coarse-view",
        utility_definition=utility_definition,
        priority_orientation=CandidatePriorityOrientation.EARLIER_WINS,
        candidates=tuple(sorted((candidate_a, candidate_b), key=lambda value: value.candidate_id)),
    )
    hold = AtlasMeasuredHoldFibre(
        hold_fibre_id=HOLD_FIBRE_ID,
        decision_cell_id=REFERENCE_DECISION_CELL_ID,
        admission_cell_id=HOLD_CELL_ID,
        action_binding_id=HOLD_BINDING_ID,
        model_member_ids=model_set.member_view_ids,
        gate_receipts=_candidate_gate_receipts(
            candidate_id=HOLD_FIBRE_ID,
            action_binding_id=HOLD_BINDING_ID,
            cell_id=HOLD_CELL_ID,
            admission=admission_spec,
        ),
        reachability_receipts=_candidate_reachability_receipts(
            candidate_id=HOLD_FIBRE_ID,
            action_binding_id=HOLD_BINDING_ID,
            cell_id=HOLD_CELL_ID,
            reachability=reachability_spec,
        ),
        viable_direction_id="positive-action-direction",
        calibration_id="calibration.reference-native-hold",
        null_tolerance=NamedDecimal(
            value_id="hold-null-tolerance.reference", value=Decimal("0.01"), unit="1"
        ),
        delivery_equivalence=DeliveryEquivalenceSpec(
            equivalence_id="delivery-equivalence.reference-hold",
            require_exact_word_identity=True,
            require_all_delivery_stages=True,
            stage_value_tolerance=NamedDecimal(
                value_id="hold-stage-tolerance.reference", value=Decimal(0), unit="1"
            ),
        ),
    )
    synthesis = AtlasControllerSynthesisPlan(
        synthesis_id="synthesis.current-reference-controller",
        atlas=ObjectIdentity.from_record(atlas_result.atlas.atlas_id, atlas_result.atlas),
        admission_comparison=ObjectIdentity.from_record(admission.comparison_id, admission),
        reachability_comparison=ObjectIdentity.from_record(
            reachability.comparison_id, reachability
        ),
        model_set=ObjectIdentity.from_record(model_set.model_set_id, model_set),
        candidate_chart=chart,
        admission_deriver_binding_id=by_role[ImplementationRole.ADMISSION_DERIVER].binding_id,
        reachability_deriver_binding_id=by_role[ImplementationRole.REACHABILITY_DERIVER].binding_id,
        synthesizer_binding_id=by_role[ImplementationRole.SYNTHESIZER].binding_id,
        observer_binding_id=by_role[ImplementationRole.OBSERVER].binding_id,
        online_gate_evaluator_binding_id=by_role[
            ImplementationRole.ONLINE_GATE_EVALUATOR
        ].binding_id,
        delivery_binding_id=by_role[ImplementationRole.DELIVERY].binding_id,
        observation_quantity_ids=("receiver",),
        support_monitors=(
            OnlineSupportMonitorSpec(
                monitor_id="monitor.current-reference-controller",
                support_ids=(law.obligations.support.support_id,),
                check_rule="Invalidate unless a declared decision or HOLD cell resolves.",
                invalidation_reason_code="outside-supported-reference-cell",
            ),
        ),
        reidentification_triggers=(
            ReidentificationTriggerSpec(
                trigger_id="trigger.current-reference-controller",
                condition="Observed state exits the declared decision chart.",
                invalidates_chart=True,
                invalidation_reason_code="reference-reidentification-required",
            ),
        ),
        authority_policy_id=system.authority_policy.policy_id,
        worst_case_latency_seconds=Decimal("0.01"),
        deadline_seconds=Decimal("0.1"),
        evidence_ceiling=EvidenceCeiling.ADMISSION,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    outcome_evaluator = by_role[ImplementationRole.OUTCOME_EVALUATOR]
    outcome_predicate = replace(
        _predicate(
            outcome_evaluator.reference,
            cell_id='reference-controller-prospective-evaluation',
            kind=AdmissionGateKind.TARGET,
            obligation=admission_obligation,
        ),
        predicate_id='predicate.reference-controller-prospective-evaluation-safety',
        quantity_id="receiver",
        lower=NamedDecimal(value_id='reference-controller-prospective-evaluation-safety-minimum', value=Decimal("-100"), unit="1"),
    )
    independent_units = tuple(f"reference-unit-{index:03d}" for index in range(1, 5))
    evaluation = ProspectiveControllerEvaluationPlan(
        evaluation_plan_id='evaluation-plan.reference-controller-prospective-evaluation',
        evaluator_boundary=EvaluatorBoundarySpec(
            boundary_id='boundary.reference-controller-prospective-evaluation',
            outcome_evaluator_binding_id=outcome_evaluator.binding_id,
            outcome_custodian_role_id="role.reference-outcome-custodian",
            reveal_authority_role_id="role.reference-reveal-authority",
            evaluator_role_id="role.reference-outcome-evaluator",
            sealed_outcome_schema='empirical-lawhood/runtime/sealed-outcome-locator',
            revealed_outcome_schema='empirical-lawhood/runtime/revealed-branch-outcome',
            raw_outcome_quantity_ids=("receiver",),
            outcome_predicates=(outcome_predicate,),
        ),
        independent_unit_type_id="reference-preparation",
        independent_unit_ids=independent_units,
        model_member_ids=model_set.member_view_ids,
        controller_branch_id="branch.reference-controller",
        reference_branch_id="branch.reference-open-loop",
        hold_calibration_branch_id="branch.reference-hold-calibration",
        causal_cutoff_id='cutoff.reference-controller-prospective-evaluation',
        commitment_point_id='commitment.reference-controller-prospective-evaluation',
        post_cutoff_action_window_id='window.reference-controller-prospective-evaluation',
        effect_quantity_id="receiver",
        effect_native_unit="1",
        favorable_direction=UtilityDirection.HIGHER_IS_BETTER,
        robust_reducer_id="memberwise-minimum",
        hold_null_tolerance=NamedDecimal(
            value_id='reference-controller-hold-null-tolerance.reference',
            value=Decimal("0.01"),
            unit="1",
        ),
        strata=(
            EvaluationStratumSpec(
                stratum_id="stratum.reference-all",
                independent_unit_ids=independent_units,
                minimum_attempted_units=4,
            ),
        ),
        minimum_attempted_units=4,
        alpha=Decimal("0.05"),
        one_sided_critical_value=Decimal("1.64485362695147"),
        materiality=NamedDecimal(
            value_id='reference-controller-materiality.reference', value=Decimal("0.5"), unit="1"
        ),
        interval_method_id="one-sided-predeclared-critical-value",
        multiplicity_family_id="single-primary-effect",
        maximum_claim_ceiling="truth-known synthetic controller conformance only",
        intent_to_treat=True,
        result_hierarchy=(
            "CONTROLLER_USE_UNSAFE",
            "CONTROLLER_USE_DELIVERY_INVALID",
            "CONTROLLER_USE_TECHNICAL_FAILURE",
            "CONTROLLER_USE_UNEVALUABLE",
            "CONTROLLER_USE_PARTIAL_OR_HETEROGENEOUS",
            "CONTROLLER_USE_HOLD_DOMINANT",
            "CONTROLLER_USE_VALIDATED",
            "CONTROLLER_USE_POSITIVE_BUT_BELOW_MATERIALITY",
            "CONTROLLER_USE_NEGATIVE",
        ),
    )
    return AtlasControllerStudy(
        study_id="programme.current-reference-controller",
        compiler_release_id='compiler.reference-controller-programme',
        system=system,
        law=law,
        action_bindings=action_bindings,
        admission=admission_spec,
        reachability=reachability_spec,
        synthesis=synthesis,
        implementations=implementations,
        measured_hold_fibre=hold,
        prospective_evaluation=evaluation,
        compilation_ceiling=EvidenceCeiling.ADMISSION,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def compile_current_reference_controller(
    study: AtlasControllerStudy | None = None,
    *,
    fail_high_utility_sink: bool = False,
) -> CompiledAtlasControllerStudy:
    if study is None:
        study = build_current_reference_controller_study(
            fail_high_utility_sink=fail_high_utility_sink
        )
    elif fail_high_utility_sink:
        raise ValueError("cannot combine an explicit programme with fixture mutation")
    return build_current_reference_controller_composition(study).compile(study)


def build_current_reference_controller_composition(
    study: AtlasControllerStudy,
    *,
    observer_disposition: ObserverDisposition = ObserverDisposition.EXACT,
    partial_delivery: bool = False,
) -> ControllerStudyComposition:
    by_role = {value.role: value for value in study.implementations}
    evaluator_binding = by_role.get(ImplementationRole.OUTCOME_EVALUATOR)
    return ControllerStudyComposition(
        registry=controller_capability_registry(study.implementations),
        admission_deriver=AtlasAdmissionDeriver(by_role[ImplementationRole.ADMISSION_DERIVER]),
        reachability_deriver=AtlasReachabilityDeriver(
            by_role[ImplementationRole.REACHABILITY_DERIVER]
        ),
        observer=ReferenceObserver(
            by_role[ImplementationRole.OBSERVER], disposition=observer_disposition
        ),
        online_gate_evaluator=AtlasReferenceLiveGateEvaluator(
            by_role[ImplementationRole.ONLINE_GATE_EVALUATOR]
        ),
        native_delivery=ReferenceNativeDelivery(
            by_role[ImplementationRole.DELIVERY], partial=partial_delivery
        ),
        outcome_evaluator=(
            None if evaluator_binding is None else ControllerUseEvaluator(evaluator_binding)
        ),
    )


class ReferenceObserver:
    def __init__(
        self,
        implementation_binding: ImplementationBinding,
        *,
        disposition: ObserverDisposition = ObserverDisposition.EXACT,
        decision_cell_id: str = REFERENCE_DECISION_CELL_ID,
    ) -> None:
        self.implementation_binding = implementation_binding
        self.disposition = disposition
        self.decision_cell_id = decision_cell_id

    def observe(self, observation: RuntimeObservation) -> ObserverEvaluation:
        exact = self.disposition is ObserverDisposition.EXACT
        return ObserverEvaluation(
            evaluation_id=f"observer-evaluation.{observation.observation_id}",
            observation=ObjectIdentity.from_record(observation.observation_id, observation),
            observer=self.implementation_binding,
            disposition=self.disposition,
            resolved_decision_cell_id=self.decision_cell_id if exact else None,
            candidate_cell_ids=(self.decision_cell_id,) if exact else (),
            receiver_candidate_pair_ids=(),
            receiver_certificate=None,
            reason_codes=() if exact else (f"REFERENCE_{self.disposition.value}",),
        )


class AtlasReferenceLiveGateEvaluator:
    def __init__(self, implementation_binding: ImplementationBinding) -> None:
        self.implementation_binding = implementation_binding

    def evaluate(
        self,
        *,
        study: AtlasControllerStudy,
        observation: RuntimeObservation,
        decision_cell_id: str,
        admission_cell_id: str,
        candidate_id: str,
        action_binding_id: str,
        model_member_ids: tuple[str, ...],
        offline_gate_receipts: tuple[CandidateGateReceipt, ...],
    ) -> AtlasLiveGateEvaluation:
        values = {value.value_id: value for value in observation.values}
        observation_identity = ObjectIdentity.from_record(observation.observation_id, observation)
        live: list[AtlasLiveGateReceipt] = []
        for offline in offline_gate_receipts:
            source = offline.receipt
            live_receipt_id = f"live-receipt.{candidate_id}.{source.receipt_id}"
            predicate = replace(
                source.predicate,
                predicate_id=f"live.{candidate_id}.{source.predicate.predicate_id}",
                evaluator=self.implementation_binding.reference,
            )
            evidence = EvidenceLink(
                link_id=f"live-evidence.{candidate_id}.{source.receipt_id}",
                relation=EvidenceRelation.DERIVED_FROM,
                source=observation_identity,
                target=ObjectIdentity.from_record(predicate.predicate_id, predicate),
                artifact_ids=tuple(
                    sorted(value.artifact_id for value in observation.input_artifacts)
                ),
                world_id=study.system.world.world_id,
                information_cutoff_id=f"live-cutoff.{observation.observation_id}",
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=study.visibility_ceiling,
                parent_visibility_ceilings=(study.visibility_ceiling,),
                reason="Live observation derives this pre-action gate.",
            )
            observed = values.get(predicate.quantity_id)
            reasons: tuple[str, ...]
            if observed is None:
                status = GateStatus.UNEVALUABLE
                margin = None
                reasons = ("GATE_OBSERVATION_UNAVAILABLE",)
            else:
                if predicate.lower is None:
                    raise AssertionError("reference live predicate lost lower bound")
                margin_value = observed.value - predicate.lower.value
                status = GateStatus.PASS if margin_value >= 0 else GateStatus.FAIL
                margin = NamedDecimal(
                    value_id=f"margin.{live_receipt_id}",
                    value=margin_value,
                    unit=observed.unit,
                )
                reasons = () if status is GateStatus.PASS else ("GATE_PREDICATE_FAILED",)
            compatibility = (
                replace(
                    source.baseline_compatibility,
                    compatibility_id=f"live-compatibility.{candidate_id}.{source.receipt_id}",
                    gate_predicate=predicate,
                    evaluator=self.implementation_binding.reference,
                )
                if source.baseline_compatibility is not None
                else None
            )
            inherited_evidence = (
                source.baseline_compatibility.obligation.evidence_links
                if source.baseline_compatibility is not None
                else ()
            )
            receipt_evidence = tuple(
                sorted((*inherited_evidence, evidence), key=lambda value: value.link_id)
            )
            receipt_artifacts_by_id = {
                value.artifact_id: value
                for value in (*observation.input_artifacts, *source.input_artifacts)
            }
            receipt_artifacts = tuple(
                receipt_artifacts_by_id[value] for value in sorted(receipt_artifacts_by_id)
            )
            receipt = AtlasGateReceipt(
                receipt_id=live_receipt_id,
                cell_id=admission_cell_id,
                model_member_id=source.model_member_id,
                numerical_view_id=source.model_member_id,
                predicate=predicate,
                observed_scalar=observed,
                observed_boolean=None,
                observed_identity=None,
                information_cutoff=InformationCutoff(
                    cutoff_id=f"live-cutoff.{observation.observation_id}",
                    clock_id=observation.coordinate.clock_id,
                    phase=CausalPhase.PRE_ACTION,
                    coordinate=observation.coordinate.coordinate,
                ),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                evaluator=self.implementation_binding.reference,
                input_artifacts=receipt_artifacts,
                evidence_links=receipt_evidence,
                baseline_compatibility=compatibility,
                status=status,
                margin=margin,
                reason_codes=reasons,
            )
            live.append(
                AtlasLiveGateReceipt(
                    candidate_id=candidate_id,
                    action_binding_id=action_binding_id,
                    observation=observation_identity,
                    receipt=receipt,
                )
            )
        return AtlasLiveGateEvaluation(
            evaluation_id=f"online-gates.{observation.observation_id}.{candidate_id}",
            observation=observation,
            decision_cell_id=decision_cell_id,
            admission_cell_id=admission_cell_id,
            candidate_id=candidate_id,
            action_binding_id=action_binding_id,
            model_member_ids=model_member_ids,
            evaluator=self.implementation_binding,
            receipts=tuple(sorted(live, key=lambda value: value.receipt.receipt_id)),
        )


class AdmissionReferenceLiveGateEvaluator:
    "Truth-known scalar live gates over every exact raw admission constituent."

    def __init__(self, implementation_binding: ImplementationBinding) -> None:
        self.implementation_binding = implementation_binding

    def evaluate_admission(
        self,
        *,
        study: AdmissionControllerStudy,
        observation: RuntimeObservation,
        decision_cell_id: str,
        admission_candidate_cell_id: str,
        candidate_id: str,
        action_binding_id: str,
        planned_coordinate_ids: tuple[str, ...],
        offline_gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...],
    ) -> AdmissionLiveGateEvaluation:
        values = {value.value_id: value for value in observation.values}
        observation_identity = ObjectIdentity.from_record(
            observation.observation_id,
            observation,
        )
        live: list[AdmissionLiveGateReceipt] = []
        scalar_kinds = {
            GatePredicateKind.SCALAR_AT_LEAST,
            GatePredicateKind.SCALAR_AT_MOST,
            GatePredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL,
        }
        for offline in offline_gate_receipts:
            receipt_id = f"live-receipt.{candidate_id}.{offline.receipt_id}"
            predicate = replace(
                offline.predicate,
                predicate_id=f"live.{candidate_id}.{offline.predicate.predicate_id}",
                evaluator=self.implementation_binding.reference,
            )
            observed = (
                values.get(predicate.quantity_id)
                if predicate.predicate_kind in scalar_kinds
                else None
            )
            reasons: tuple[str, ...]
            if observed is None:
                status = GateStatus.UNEVALUABLE
                margin = None
                reasons = ("GATE_OBSERVATION_UNAVAILABLE",)
            else:
                status, margin, reasons = derive_admission_gate_observation(
                    receipt_id=receipt_id,
                    predicate=predicate,
                    observed_scalar=observed,
                    observed_boolean=None,
                    observed_identity=None,
                )
            cutoff = InformationCutoff(
                cutoff_id=f"live-cutoff.{observation.observation_id}",
                clock_id=observation.coordinate.clock_id,
                phase=CausalPhase.PRE_ACTION,
                coordinate=observation.coordinate.coordinate,
            )
            evidence = EvidenceLink(
                link_id=f"live-evidence.{candidate_id}.{offline.receipt_id}",
                relation=EvidenceRelation.DERIVED_FROM,
                source=observation_identity,
                target=ObjectIdentity.from_record(predicate.predicate_id, predicate),
                artifact_ids=tuple(
                    sorted(value.artifact_id for value in observation.input_artifacts)
                ),
                world_id=study.system.world.world_id,
                information_cutoff_id=cutoff.cutoff_id,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=study.visibility_ceiling,
                parent_visibility_ceilings=(study.visibility_ceiling,),
                reason="Fresh truth-known pre-action observation derives this live gate.",
            )
            artifacts = {
                value.artifact_id: value
                for value in (*offline.input_artifacts, *observation.input_artifacts)
            }
            links = {value.link_id: value for value in (*offline.evidence_links, evidence)}
            live.append(
                AdmissionLiveGateReceipt(
                    receipt_id=receipt_id,
                    candidate_id=candidate_id,
                    action_binding_id=action_binding_id,
                    observation=observation_identity,
                    offline_gate_receipt=ObjectIdentity.from_record(
                        offline.receipt_id,
                        offline,
                    ),
                    planned_coordinate=offline.planned_coordinate,
                    law_evaluation_binding=offline.law_evaluation_binding,
                    predicate=predicate,
                    observed_scalar=observed,
                    observed_boolean=None,
                    observed_identity=None,
                    information_cutoff=cutoff,
                    evaluator=self.implementation_binding,
                    input_artifacts=tuple(artifacts[key] for key in sorted(artifacts)),
                    evidence_links=tuple(links[key] for key in sorted(links)),
                    status=status,
                    margin=margin,
                    reason_codes=reasons,
                )
            )
        return AdmissionLiveGateEvaluation(
            evaluation_id=f"online-gates.{observation.observation_id}.{candidate_id}",
            observation=observation,
            decision_cell_id=decision_cell_id,
            admission_candidate_cell_id=admission_candidate_cell_id,
            candidate_id=candidate_id,
            action_binding_id=action_binding_id,
            planned_coordinate_ids=planned_coordinate_ids,
            evaluator=self.implementation_binding,
            receipts=tuple(sorted(live, key=lambda value: value.receipt_id)),
        )


class ReferenceNativeDelivery:
    def __init__(
        self,
        implementation_binding: ImplementationBinding,
        *,
        partial: bool = False,
    ) -> None:
        self.implementation_binding = implementation_binding
        self.partial = partial

    def deliver(
        self,
        *,
        action_word: OccurrenceActionWord,
        commitment_kind: ScientificCommitmentKind,
    ) -> DeliveryPortResult:
        del commitment_kind
        observed = tuple(
            ObservedActionOccurrence(
                observation_id=f"observed.{value.occurrence_id}",
                expected_occurrence_id=value.occurrence_id,
                requested=value.requested,
                accepted=value.accepted,
                applied=value.applied,
                realized=None if self.partial else value.realized,
                reason_codes=("REALIZED_STAGE_NOT_OBSERVED",) if self.partial else (),
            )
            for value in action_word.occurrences
        )
        return DeliveryPortResult(
            result_id=f"delivery-result.{action_word.word_id}",
            observed_occurrences=observed,
            failure_state=(OperationalDeliveryState.DELIVERY_RECOVERY if self.partial else None),
        )


def build_reference_runtime_observation(
    compiled: CompiledAtlasControllerStudy,
    *,
    observation_id: str = "reference-observation",
    independent_unit_id: str | None = None,
    fail_active_gate: AdmissionGateKind | None = None,
    fail_hold_gate: AdmissionGateKind | None = None,
) -> RuntimeObservation:
    evaluator = compiled.implementation(ImplementationRole.ONLINE_GATE_EVALUATOR)
    values = tuple(
        sorted(
            (
                NamedDecimal(
                    value_id=f"quantity.{cell_id}.{kind.value.lower()}",
                    value=(
                        Decimal(-1)
                        if (
                            (cell_id != HOLD_CELL_ID and kind is fail_active_gate)
                            or (cell_id == HOLD_CELL_ID and kind is fail_hold_gate)
                        )
                        else Decimal(1)
                    ),
                    unit="1",
                )
                for cell_id in (ACTIVE_A_CELL_ID, ACTIVE_B_CELL_ID, HOLD_CELL_ID)
                for kind in AdmissionGateKind
            ),
            key=lambda value: value.value_id,
        )
    )
    return RuntimeObservation(
        observation_id=observation_id,
        independent_unit_id=independent_unit_id or f"unit.{observation_id}",
        values=values,
        coordinate=_coordinate("0.5"),
        input_artifacts=(evaluator.reference.payload,),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def run_current_reference_tick(
    compiled: CompiledAtlasControllerStudy,
    *,
    observation: RuntimeObservation | None = None,
    observer_disposition: ObserverDisposition = ObserverDisposition.EXACT,
    partial_delivery: bool = False,
    commitment_coordinate: ClockCoordinate | None = None,
) -> AtlasControllerTickReceipt:
    observation = observation or build_reference_runtime_observation(compiled)
    composition = build_current_reference_controller_composition(
        compiled.study,
        observer_disposition=observer_disposition,
        partial_delivery=partial_delivery,
    )
    return composition.tick(
        compiled,
        observation,
        commitment_coordinate=commitment_coordinate or _coordinate("0.51"),
    )


def _reference_delivery_trace(
    compiled: CompiledAtlasControllerStudy,
    tick: AtlasControllerTickReceipt,
    *,
    trace_id: str,
    action_word: OccurrenceActionWord,
) -> ExactActionDeliveryTrace:
    delivery = ReferenceNativeDelivery(
        compiled.implementation(ImplementationRole.DELIVERY)
    ).deliver(
        action_word=action_word,
        commitment_kind=ScientificCommitmentKind.ACTION,
    )
    return ExactActionDeliveryTrace(
        trace_id=trace_id,
        commitment=ObjectIdentity.from_record(tick.commitment.commitment_id, tick.commitment),
        delivery=compiled.implementation(ImplementationRole.DELIVERY),
        commitment_kind=ScientificCommitmentKind.ACTION,
        expected_action_word=action_word,
        observed_occurrences=delivery.observed_occurrences,
        operational_state=OperationalDeliveryState.DELIVERED,
        reason_codes=(),
    )


def build_reference_revealed_bundle(
    compiled: CompiledAtlasControllerStudy,
    tick: AtlasControllerTickReceipt,
    *,
    independent_unit_id: str,
    controller_effects: dict[str, Decimal],
    reference_effects: dict[str, Decimal] | None = None,
    calibration_effects: dict[str, Decimal] | None = None,
    prefix_mismatch: bool = False,
    controller_technical_reason_codes: tuple[str, ...] = (),
) -> RevealedProspectiveControllerBundle:
    """Build an in-memory truth-known reveal; never an external scientific result."""

    plan = compiled.study.prospective_evaluation
    if plan is None:
        raise ValueError("reference reveal requires a controller-use-declared programme")
    if independent_unit_id not in plan.independent_unit_ids:
        raise ValueError("reference reveal unit is absent from the frozen roster")
    if set(controller_effects) != set(plan.model_member_ids):
        raise ValueError("controller effect map differs from the member roster")
    reference_effects = reference_effects or {
        member: Decimal(0) for member in plan.model_member_ids
    }
    calibration_effects = calibration_effects or {
        member: controller_effects[member] for member in plan.model_member_ids
    }
    if set(reference_effects) != set(plan.model_member_ids) or set(calibration_effects) != set(
        plan.model_member_ids
    ):
        raise ValueError("reference/calibration effect map differs from member roster")

    action_by_id = {
        binding.action_binding_id: binding.action_word
        for binding in compiled.study.action_bindings
    }
    expected_controller_word = (
        tick.commitment.action_binding.action_word
        if tick.commitment.action_binding is not None
        else None
    )
    expected_reference_word = action_by_id[ACTIVE_A_BINDING_ID]
    initial_fingerprint = _digest(f"initial:{independent_unit_id}")
    prefix_fingerprint = _digest(f"prefix:{independent_unit_id}")
    branches = [plan.controller_branch_id, plan.reference_branch_id]
    if plan.hold_calibration_branch_id is not None:
        branches.append(plan.hold_calibration_branch_id)
    locators = tuple(
        sorted(
            (
                SealedOutcomeLocator(
                    locator_id=f"locator.{independent_unit_id}.{branch}.{member}",
                    branch_id=branch,
                    model_member_id=member,
                    artifact=ArtifactIdentity(
                        artifact_id=f"artifact.{independent_unit_id}.{branch}.{member}",
                        role="truth-known-controller-outcome",
                        payload_schema=plan.evaluator_boundary.sealed_outcome_schema,
                        sha256=_digest(f"sealed:{independent_unit_id}:{branch}:{member}"),
                        media_type="application/json",
                        size_bytes=1,
                    ),
                    outcome_access=OutcomeAccess.EVALUATION_SEALED,
                )
                for branch in branches
                for member in plan.model_member_ids
            ),
            key=lambda value: value.locator_id,
        )
    )
    sealed = ProspectiveControllerBundle(
        bundle_id=f"bundle.{independent_unit_id}",
        evaluation_plan=ObjectIdentity.from_record(plan.evaluation_plan_id, plan),
        independent_unit_id=independent_unit_id,
        cohort_id='cohort.reference-controller-prospective-evaluation',
        common_initial_state_fingerprint=initial_fingerprint,
        common_prefix_fingerprint=prefix_fingerprint,
        compiled_study=ObjectIdentity.from_record(compiled.compiled_study_id, compiled),
        tick=ObjectIdentity.from_record(tick.tick_id, tick),
        model_member_ids=plan.model_member_ids,
        controller_branch_id=plan.controller_branch_id,
        reference_branch_id=plan.reference_branch_id,
        hold_calibration_branch_id=plan.hold_calibration_branch_id,
        expected_controller_word=expected_controller_word,
        expected_reference_word=expected_reference_word,
        sealed_outcome_locators=locators,
    )
    reference_trace = _reference_delivery_trace(
        compiled,
        tick,
        trace_id=f"reference-trace.{independent_unit_id}",
        action_word=expected_reference_word,
    )
    calibration_word = expected_controller_word or expected_reference_word
    calibration_trace = _reference_delivery_trace(
        compiled,
        tick,
        trace_id=f"calibration-trace.{independent_unit_id}",
        action_word=calibration_word,
    )
    locator_by_coordinate = {
        (locator.branch_id, locator.model_member_id): locator for locator in locators
    }
    outcomes: list[RevealedBranchOutcome] = []
    for branch in branches:
        for member in plan.model_member_ids:
            if branch == plan.controller_branch_id:
                action_word = expected_controller_word
                delivery_trace = tick.delivery_trace
                value = controller_effects[member]
                technical = controller_technical_reason_codes
            elif branch == plan.reference_branch_id:
                action_word = expected_reference_word
                delivery_trace = reference_trace
                value = reference_effects[member]
                technical = ()
            else:
                action_word = calibration_word
                delivery_trace = calibration_trace
                value = calibration_effects[member]
                technical = ()
            locator = locator_by_coordinate[(branch, member)]
            outcomes.append(
                RevealedBranchOutcome(
                    outcome_id=f"outcome.{independent_unit_id}.{branch}.{member}",
                    sealed_locator=ObjectIdentity.from_record(locator.locator_id, locator),
                    branch_id=branch,
                    model_member_id=member,
                    initial_state_fingerprint=initial_fingerprint,
                    prefix_fingerprint=(
                        _digest(f"mismatched-prefix:{independent_unit_id}")
                        if prefix_mismatch and branch == plan.controller_branch_id
                        else prefix_fingerprint
                    ),
                    action_word=action_word,
                    delivery_trace=delivery_trace,
                    raw_outcomes=(
                        ()
                        if action_word is None
                        else (
                            NamedDecimal(
                                value_id=plan.effect_quantity_id,
                                value=value,
                                unit=plan.effect_native_unit,
                            ),
                        )
                    ),
                    technical_reason_codes=technical,
                    outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                )
            )
    return RevealedProspectiveControllerBundle(
        reveal_id=f"reveal.{independent_unit_id}",
        sealed_bundle=sealed,
        reveal_authorization=ObjectIdentity(
            object_id=f"truth-known-reveal-authorization.{independent_unit_id}",
            object_schema='empirical-lawhood/testing/truth-known-reveal-authorization',
            object_version="1.0.0",
            object_fingerprint=_digest(f"authorization:{independent_unit_id}"),
        ),
        outcomes=tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
    )


__all__ = [
    "ACTIVE_A_BINDING_ID",
    "ACTIVE_A_CELL_ID",
    "ACTIVE_B_BINDING_ID",
    "ACTIVE_B_CELL_ID",
    "HOLD_BINDING_ID",
    "HOLD_CELL_ID",
    "HOLD_FIBRE_ID",
    "REFERENCE_DECISION_CELL_ID",
    "ReferenceNativeDelivery",
    "ReferenceObserver",
    'AtlasReferenceLiveGateEvaluator',
    'AdmissionReferenceLiveGateEvaluator',
    'build_current_reference_controller_study',
    "build_current_reference_controller_composition",
    "build_reference_runtime_observation",
    "build_reference_revealed_bundle",
    "compile_current_reference_controller",
    "run_current_reference_tick",
]
