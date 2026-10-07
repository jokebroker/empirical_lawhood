"""Bounded observation and live-gate replay for a frozen finite child instance."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import ClassVar

from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import validate_stable_id
from empirical_lawhood.planning.controller_study import DeliveryControllerStudy, ControllerInstanceBinding, ImplementationBinding, ImplementationRole
from empirical_lawhood.planning.evidence_geometry import AdmissionCoordinateGateReceipt, derive_admission_gate_observation
from empirical_lawhood.runtime.controller_runtime import AdmissionLiveGateReceipt, ObserverDisposition, ObserverEvaluation, NumericalViewLiveGateEvaluation, RuntimeObservation


@dataclass(frozen=True, slots=True)
class FiniteInstanceObserver:
    """Resolve only the exact handoff observation used to compile this child."""

    capability_key: ClassVar[str] = "control.finite-instance-observer"
    capability_version: ClassVar[str] = "1.0.0"

    implementation_binding: ImplementationBinding
    instance: ControllerInstanceBinding
    decision_cell_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_cell_id, field_name="decision_cell_id")
        if self.implementation_binding.role is not ImplementationRole.OBSERVER:
            raise ValueError("finite instance observer requires the observer role")

    def observe(self, observation: RuntimeObservation) -> ObserverEvaluation:
        identity = ObjectIdentity.from_record(observation.observation_id, observation)
        exact = identity == self.instance.bound_observation
        return ObserverEvaluation(
            evaluation_id=f"finite-observer.{observation.observation_id}",
            observation=identity,
            observer=self.implementation_binding,
            disposition=ObserverDisposition.EXACT if exact else ObserverDisposition.OUTSIDE_SUPPORT,
            resolved_decision_cell_id=self.decision_cell_id if exact else None,
            candidate_cell_ids=(self.decision_cell_id,) if exact else (),
            receiver_candidate_pair_ids=(),
            receiver_certificate=None,
            reason_codes=() if exact else ("FINITE_INSTANCE_OBSERVATION_CHANGED",),
        )


@dataclass(frozen=True, slots=True)
class FiniteInstanceOnlineGateEvaluator:
    """Replay the same raw native margins only while the exact handoff remains bound.

    There is no predictor refit or live policy search. Delivery validity is
    checked separately against all four native stages by ControllerRuntime.
    """

    capability_key: ClassVar[str] = "control.finite-instance-online-gates"
    capability_version: ClassVar[str] = "1.0.0"

    implementation_binding: ImplementationBinding
    instance: ControllerInstanceBinding

    def __post_init__(self) -> None:
        if self.implementation_binding.role is not ImplementationRole.ONLINE_GATE_EVALUATOR:
            raise ValueError("finite online gate evaluator requires its own exact role")

    def evaluate_numerical_view_live_gate(
        self,
        *,
        study: DeliveryControllerStudy,
        observation: RuntimeObservation,
        decision_cell_id: str,
        admission_candidate_cell_id: str,
        candidate_id: str,
        action_binding_id: str,
        planned_coordinate_ids: tuple[str, ...],
        offline_gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...],
    ) -> NumericalViewLiveGateEvaluation:
        observation_identity = ObjectIdentity.from_record(observation.observation_id, observation)
        if (
            study.instance_binding != self.instance
            or observation_identity != self.instance.bound_observation
        ):
            raise ValueError("finite live gate cannot reuse a changed or unbound handoff")
        expected = tuple(
            g
            for g in study.admission.corpus.gate_receipts
            if g.planned_coordinate.coordinate_id in planned_coordinate_ids
        )
        if offline_gate_receipts != expected:
            raise ValueError("finite live gate replaces or omits frozen raw operands")
        receipts = []
        for offline in offline_gate_receipts:
            artifacts = {a.artifact_id: a for a in offline.input_artifacts}
            if any(artifacts.get(a.artifact_id) != a for a in observation.input_artifacts):
                raise ValueError("finite live gate lacks the exact public observation artifacts")
            predicate = replace(
                offline.predicate,
                predicate_id=f"live.{offline.predicate.predicate_id}",
                evaluator=self.implementation_binding.reference,
            )
            receipt_id = f"live.{candidate_id}.{offline.receipt_id}"
            supplied = any(
                v is not None
                for v in (
                    offline.observed_scalar,
                    offline.observed_boolean,
                    offline.observed_identity,
                )
            )
            status, margin, reasons = (
                derive_admission_gate_observation(
                    receipt_id=receipt_id,
                    predicate=predicate,
                    observed_scalar=offline.observed_scalar,
                    observed_boolean=offline.observed_boolean,
                    observed_identity=offline.observed_identity,
                )
                if supplied
                else (GateStatus.UNEVALUABLE, None, ("GATE_OBSERVATION_UNAVAILABLE",))
            )
            receipts.append(
                AdmissionLiveGateReceipt(
                    receipt_id=receipt_id,
                    candidate_id=candidate_id,
                    action_binding_id=action_binding_id,
                    observation=observation_identity,
                    offline_gate_receipt=ObjectIdentity.from_record(offline.receipt_id, offline),
                    planned_coordinate=offline.planned_coordinate,
                    law_evaluation_binding=offline.law_evaluation_binding,
                    predicate=predicate,
                    observed_scalar=offline.observed_scalar,
                    observed_boolean=offline.observed_boolean,
                    observed_identity=offline.observed_identity,
                    information_cutoff=offline.information_cutoff,
                    evaluator=self.implementation_binding,
                    input_artifacts=offline.input_artifacts,
                    evidence_links=offline.evidence_links,
                    status=status,
                    margin=margin,
                    reason_codes=reasons,
                )
            )
        return NumericalViewLiveGateEvaluation(
            evaluation_id=f"finite-live.{observation.observation_id}.{candidate_id}",
            observation=observation,
            decision_cell_id=decision_cell_id,
            admission_candidate_cell_id=admission_candidate_cell_id,
            candidate_id=candidate_id,
            action_binding_id=action_binding_id,
            planned_coordinate_ids=planned_coordinate_ids,
            evaluator=self.implementation_binding,
            receipts=tuple(sorted(receipts, key=lambda receipt: receipt.receipt_id)),
        )
