"""Finite native-word certification over complete calibrated response sets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionReceiptProductionPlan
from empirical_lawhood.planning.finite_response_geometry import FiniteReachabilityMethodReceipt, FiniteResponseSetReachabilityRequest, FiniteResponseSetReachabilityResult, finite_certificate_status, finite_constraint_margins


@dataclass(frozen=True, slots=True)
class FiniteResponseSetReachabilityMethod:
    """No map, rank, baseline subtraction, favorable-view selection or mean-only test."""

    capability_key: ClassVar[str] = "geometry.finite-response-set-reachability"
    capability_version: ClassVar[str] = "1.0.0"

    def evaluate(
        self,
        request: FiniteResponseSetReachabilityRequest,
        *,
        evaluator_implementation: ObjectIdentity,
    ) -> FiniteResponseSetReachabilityResult:
        return FiniteResponseSetReachabilityResult(
            result_id=f"finite-reachability.{request.request_id}",
            request=request,
            evaluator_implementation=evaluator_implementation,
            margins=finite_constraint_margins(request),
            status=finite_certificate_status(request),
        )


@dataclass(frozen=True, slots=True)
class FiniteReachabilityAdmissionReceiptProducer:
    "Bind mechanically derived finite truth to the frozen admission production plan."

    implementation: ObjectIdentity

    capability_key: ClassVar[str] = "geometry.finite-admission-reachability-receipt-producer"
    capability_version: ClassVar[str] = "1.0.0"

    def produce(
        self,
        *,
        plan: ReceiptAdmissionReceiptProductionPlan,
        result: FiniteResponseSetReachabilityResult,
        method: ExecutableReference,
        admission_direction_gate_receipt_ids: tuple[str, ...],
        input_artifacts: tuple[ArtifactIdentity, ...],
        evidence_links: tuple[EvidenceLink, ...],
    ) -> FiniteReachabilityMethodReceipt:
        coordinate = result.request.response_set.planned_coordinate
        if (
            result.evaluator_implementation
            != ObjectIdentity.from_record(method.reference_id, method)
            or coordinate not in plan.coordinates
            or result.request.action_word != plan.action_fibre(coordinate.action_fibre).action_word
            or result.request.information_cutoff != plan.information_cutoff
            or self.implementation != plan.reachability_receipt_producer
        ):
            raise ValueError("finite admission result/producer differs from its frozen plan")
        return FiniteReachabilityMethodReceipt(
            receipt_id=f"receipt.finite-admission.{plan.plan_id}.{result.result_id}",
            request=result.request,
            raw_result=ObjectIdentity.from_record(result.result_id, result),
            admission_direction_gate_receipt_ids=admission_direction_gate_receipt_ids,
            horizon=plan.horizon,
            constraint_ids=plan.reachability_constraint_ids,
            evidence_domain=plan.evidence_domain,
            authority_boundary=plan.authority_boundary,
            producer=self.implementation,
            resource_envelope=plan.reachability_resource_envelope,
            method=method,
            input_artifacts=input_artifacts,
            evidence_links=evidence_links,
            margins=result.margins,
            status=result.status,
        )
