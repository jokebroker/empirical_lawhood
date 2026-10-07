"""Disk-independent receiver-ambiguity conformance through the sole route."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.adapters.control.reference import ACTIVE_B_BINDING_ID, ACTIVE_B_CELL_ID, REFERENCE_DECISION_CELL_ID, AtlasReferenceLiveGateEvaluator, build_current_reference_controller_composition, build_current_reference_controller_study, build_reference_runtime_observation
from empirical_lawhood.adapters.methods.control_quotient import (
    _assessment,
    _candidate,
    _decision,
    _fiber,
    verified_candidate_action_evidence,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.receiver_geometry_control import (
    CandidateActionEvidence,
    CandidateDecisionDisposition,
    EquivalenceDisposition,
    ReceiverCandidateView,
    ReceiverFiberDisposition,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.controller_study import AtlasControllerStudy, ImplementationRole
from empirical_lawhood.planning.controller_study import ControllerActionBinding
from empirical_lawhood.planning.receiver_geometry import ReceiverQuotientControlPlan
from empirical_lawhood.runtime.controller_compiler import CompiledAtlasControllerStudy
from empirical_lawhood.runtime.controller_runtime import AtlasControllerTickReceipt, ObserverDisposition, ObserverEvaluation, RuntimeObservation


class GeneratedReceiverStratum(StrEnum):
    SHARED_ACTION_AMBIGUOUS = "SHARED_ACTION_AMBIGUOUS"
    DECISION_DISTINCT_AMBIGUOUS = "DECISION_DISTINCT_AMBIGUOUS"
    SAFE_ABSTENTION_EQUIVALENT = "SAFE_ABSTENTION_EQUIVALENT"
    EXACT_OBSERVATION = "EXACT_OBSERVATION"
    OUTSIDE_SUPPORT_BOUNDARY = "OUTSIDE_SUPPORT_BOUNDARY"


@dataclass(frozen=True, slots=True)
class ReferenceTerminationContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/reference-termination-contract'

    contract_id: str
    rule: str

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        if self.rule != "terminate-before-unsafe-prefix":
            raise ValueError("reference termination rule changes")


def _termination_identity(case_id: str) -> ObjectIdentity:
    contract = ReferenceTerminationContract(
        contract_id=f"termination.{case_id}", rule="terminate-before-unsafe-prefix"
    )
    return ObjectIdentity.from_record(contract.contract_id, contract)


def _action_binding(study: AtlasControllerStudy) -> ControllerActionBinding:
    return next(
        value
        for value in study.action_bindings
        if value.action_binding_id == ACTIVE_B_BINDING_ID
    )


def _evidence(
    study: AtlasControllerStudy,
    *,
    case_id: str,
    candidate: ReceiverCandidateView,
) -> CandidateActionEvidence:
    binding = _action_binding(study)
    gate_receipts = tuple(
        value
        for value in study.admission.receipts
        if value.cell_id == ACTIVE_B_CELL_ID
        and value.model_member_id == candidate.numerical_view_id
    )
    reachability = next(
        value
        for value in study.reachability.receipts
        if value.admission_cell_id == ACTIVE_B_CELL_ID
        and value.model_member_id == candidate.numerical_view_id
    )
    return verified_candidate_action_evidence(
        evidence_id=f"evidence.{case_id}.{candidate.pair_id}",
        candidate=candidate,
        action_word=binding.action_word,
        response_law=study.law,
        response_law_id=study.law.law_id,
        causal_prefix=binding.causal_prefix,
        gate_receipts=gate_receipts,
        reachability_receipt=reachability,
    )


def build_generated_receiver_study(
    stratum: GeneratedReceiverStratum,
    *,
    preparation_id: str,
) -> AtlasControllerStudy:
    validate_stable_id(preparation_id, field_name="preparation_id")
    base = build_current_reference_controller_study()
    case_id = f"generated-{stratum.value.lower().replace('_', '-')}-{preparation_id}"
    action = _action_binding(base).action_word
    if base.measured_hold_fibre is None:
        raise AssertionError("reference programme lost measured HOLD")
    hold = next(
        value.action_word
        for value in base.action_bindings
        if value.action_binding_id == base.measured_hold_fibre.action_binding_id
    )
    views = (
        ("coarse-view",)
        if stratum is GeneratedReceiverStratum.EXACT_OBSERVATION
        else ("coarse-view", "fine-view")
    )
    candidates = tuple(
        _candidate(case_id, f"candidate-{index:02d}", numerical_view_id=view, action=action)
        for index, view in enumerate(views, start=1)
    )
    fiber = _fiber(
        case_id,
        action,
        candidates,
        disposition=(
            ReceiverFiberDisposition.INJECTIVE_ON_TESTED_DOMAIN
            if stratum is GeneratedReceiverStratum.EXACT_OBSERVATION
            else (
                ReceiverFiberDisposition.MIXTURE_OR_UNRESOLVED
                if stratum is GeneratedReceiverStratum.OUTSIDE_SUPPORT_BOUNDARY
                else ReceiverFiberDisposition.FINITE_MULTIPLE
            )
        ),
        world_id=base.system.world.world_id,
    )
    fallback = None
    termination = None
    if stratum in {
        GeneratedReceiverStratum.SHARED_ACTION_AMBIGUOUS,
        GeneratedReceiverStratum.EXACT_OBSERVATION,
    }:
        dispositions = tuple(CandidateDecisionDisposition.ACTION_SET_AVAILABLE for _ in candidates)
        response = admission = EquivalenceDisposition.EQUIVALENT
    elif stratum is GeneratedReceiverStratum.DECISION_DISTINCT_AMBIGUOUS:
        dispositions = (
            CandidateDecisionDisposition.ACTION_SET_AVAILABLE,
            CandidateDecisionDisposition.QUALIFIED_HOLD,
        )
        response = admission = EquivalenceDisposition.DISTINCT
        fallback = hold
    elif stratum is GeneratedReceiverStratum.SAFE_ABSTENTION_EQUIVALENT:
        dispositions = tuple(CandidateDecisionDisposition.QUALIFIED_HOLD for _ in candidates)
        response = EquivalenceDisposition.DISTINCT
        admission = EquivalenceDisposition.EQUIVALENT
        fallback = hold
    else:
        dispositions = tuple(CandidateDecisionDisposition.UNEVALUABLE for _ in candidates)
        response = admission = EquivalenceDisposition.UNEVALUABLE
        termination = _termination_identity(case_id)
    decisions = tuple(
        _decision(
            case_id,
            candidate,
            disposition,
            action_evidence=(
                _evidence(base, case_id=case_id, candidate=candidate)
                if disposition is CandidateDecisionDisposition.ACTION_SET_AVAILABLE
                else None
            ),
            hold=hold,
        )
        for candidate, disposition in zip(candidates, dispositions, strict=True)
    )
    assessment = _assessment(
        case_id,
        fiber,
        decisions,
        observational=(
            EquivalenceDisposition.EQUIVALENT
            if admission is not EquivalenceDisposition.UNEVALUABLE
            else EquivalenceDisposition.UNEVALUABLE
        ),
        response=response,
        admission=admission,
    )
    observer = next(
        value for value in base.implementations if value.role is ImplementationRole.OBSERVER
    )
    quotient = ReceiverQuotientControlPlan(
        plan_id=f"plan.{case_id}",
        observation_experiment=fiber.observation_experiment,
        fiber_assessment=fiber,
        equivalence_assessment=assessment,
        observer_binding_id=observer.binding_id,
        fallback_word=fallback,
        termination_contract=termination,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return replace(base, study_id=f"programme.{case_id}", receiver_quotient=quotient)


def compile_generated_receiver_study(
    study: AtlasControllerStudy,
) -> CompiledAtlasControllerStudy:
    return build_current_reference_controller_composition(study).compile(study)


class ReferenceReceiverQuotientObserver:
    def __init__(self, compiled: CompiledAtlasControllerStudy) -> None:
        self.implementation_binding = compiled.implementation(ImplementationRole.OBSERVER)
        certificate = compiled.ambiguity_certificate
        if certificate is None:
            raise ValueError("receiver quotient observer requires a certificate")
        self._certificate = certificate

    def observe(self, observation: RuntimeObservation) -> ObserverEvaluation:
        candidates = self._certificate.assessment.fiber_assessment.candidates
        exact = len(candidates) == 1
        return ObserverEvaluation(
            evaluation_id=f"observer-evaluation.{observation.observation_id}",
            observation=ObjectIdentity.from_record(observation.observation_id, observation),
            observer=self.implementation_binding,
            disposition=(ObserverDisposition.EXACT if exact else ObserverDisposition.AMBIGUOUS),
            resolved_decision_cell_id=REFERENCE_DECISION_CELL_ID,
            candidate_cell_ids=(REFERENCE_DECISION_CELL_ID,),
            receiver_candidate_pair_ids=tuple(sorted(value.pair_id for value in candidates)),
            receiver_certificate=ObjectIdentity.from_record(
                self._certificate.certificate_id, self._certificate
            ),
            reason_codes=(() if exact else ("OBSERVER_BOUNDED_AMBIGUITY",)),
        )


def run_generated_receiver_tick(
    stratum: GeneratedReceiverStratum,
    *,
    preparation_id: str,
) -> tuple[CompiledAtlasControllerStudy, AtlasControllerTickReceipt]:
    programme = build_generated_receiver_study(stratum, preparation_id=preparation_id)
    compiled = compile_generated_receiver_study(programme)
    composition = replace(
        build_current_reference_controller_composition(programme),
        observer=ReferenceReceiverQuotientObserver(compiled),
        online_gate_evaluator=AtlasReferenceLiveGateEvaluator(
            compiled.implementation(ImplementationRole.ONLINE_GATE_EVALUATOR)
        ),
    )
    observation = build_reference_runtime_observation(
        compiled,
        observation_id=f"observation.{preparation_id}",
        independent_unit_id=preparation_id,
    )
    return compiled, composition.tick(
        compiled,
        observation,
        commitment_coordinate=replace(
            observation.coordinate,
            coordinate=observation.coordinate.coordinate + Decimal("0.05"),
        ),
    )


__all__ = [
    "GeneratedReceiverStratum",
    "ReferenceReceiverQuotientObserver",
    "ReferenceTerminationContract",
    'build_generated_receiver_study',
    'compile_generated_receiver_study',
    "run_generated_receiver_tick",
]
