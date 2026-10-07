"""Native interval operands enter the unchanged finite reachability/gate owners."""

from decimal import Decimal as D

from empirical_lawhood.adapters.geometry.finite_response_reachability import (
    FiniteReachabilityAdmissionReceiptProducer,
    FiniteResponseSetReachabilityMethod,
)
from empirical_lawhood.adapters.geometry.admission_receipts import LawMemberEvaluationBinder, AdmissionGateRawInput, AdmissionUtilityRawInput, RawAdmissionGateReceiptProducer, RawUtilityAdmissionReceiptProducer
from empirical_lawhood.adapters.methods.law_evaluation import LawEvaluationDisposition
from dataclasses import replace
from empirical_lawhood.kernel.admission import AdmissionGateKind as G
from empirical_lawhood.kernel.causal_contracts import ReceiverInterval, TemporalPredicateAssessment
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.time import AvailabilitySpec, CausalPhase, ClockCoordinate
from empirical_lawhood.planning.controller_study import FrozenFeedbackConsumer
from empirical_lawhood.planning.evidence_geometry import BaselinePreservationCompatibility, ReceiptAdmissionRawDisposition, ReceiptAdmissionUtilityDirection, PreservationCompatibilityRule
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionReceiptCorpus
from empirical_lawhood.planning.finite_response_geometry import FiniteJointCalibration, FiniteResponseSet, FiniteResponseSetReachabilityRequest, FiniteSetDisposition, FiniteResponseBound, FiniteTaskFunctionalSpec, FiniteTaskFunctionalKind, FiniteTargetBox, FiniteTargetInterval
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.methods.law_evaluation import LawEvaluationRequest, LawEvaluationResult
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionReceiptProductionPlan
from empirical_lawhood.planning.finite_response_geometry import FiniteResponseCoordinate


def finite_interval_corpus(
    *,
    plan: ReceiptAdmissionReceiptProductionPlan,
    law: ResponseLaw,
    table_record: CanonicalRecord,
    table_id: str,
    requests: tuple[LawEvaluationRequest, ...],
    results: tuple[LawEvaluationResult, ...],
    word: OccurrenceActionWord,
    qualification: ObjectIdentity,
    qualified: bool,
    root: str,
    causal_valid: bool,
    effort_admissible: bool,
    projected_cost: NamedDecimal,
    response_receiver: str,
    safety_receiver: str,
    coordinates: tuple[FiniteResponseCoordinate, ...],
    clock: ObjectIdentity,
    target_intervals: tuple[FiniteTargetInterval, ...],
    consumer: FrozenFeedbackConsumer,
    calibration_artifact: ArtifactIdentity,
    source_artifacts: tuple[ArtifactIdentity, ...],
    evidence: tuple[EvidenceLink, ...],
    prediction_unavailable_reason: str,
) -> FiniteCertificateAdmissionReceiptCorpus:
    if (
        len({c.unit for c in coordinates}) != 1
        or projected_cost.unit != "1"
        or calibration_artifact.sha256 != qualification.object_fingerprint
        or any(a not in source_artifacts for r in requests for a in r.input_artifacts)
        or not evidence
        or any(
            e.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or e.information_cutoff_id != plan.information_cutoff.cutoff_id
            for e in evidence
        )
    ):
        raise ValueError("finite admission changed causal custody or action-specific calibration")
    method, cell = plan.gate_predicates[0].evaluator, plan.coordinates[0]
    time = plan.information_cutoff.coordinate
    assert time is not None
    native_start = plan.gate_predicates[0].protected_interval.start

    def coordinate(value: D) -> ClockCoordinate:
        return replace(native_start, coordinate=value)

    available = AvailabilitySpec(
        native_start.clock_id, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, time
    )
    word_id = ObjectIdentity.from_record(word.word_id, word)
    joint = FiniteJointCalibration(
        f"{plan.plan_id}.calibration",
        qualification,
        consumer.recipe,
        (word_id,),
        coordinates,
        cell.qualification_view_ids,
        D(".90"),
        calibration_artifact,
    )
    box = FiniteTargetBox(f"{plan.plan_id}.box", target_intervals)
    task = FiniteTaskFunctionalSpec(
        f"{plan.plan_id}.task",
        FiniteTaskFunctionalKind.WINDOW_MAXIMUM_BOX,
        consumer.recipe,
        consumer.recipe,
        available,
        (box,),
        (box.box_id,),
    )
    bindings = tuple(
        LawMemberEvaluationBinder().bind(
            plan=plan, coordinate_id=cell.coordinate_id, request=req, result=res
        )
        for req, res in zip(requests, results, strict=True)
    )
    present = all(r.disposition is LawEvaluationDisposition.SUPPORTED for r in results)
    outside = any(r.disposition is LawEvaluationDisposition.OUTSIDE_SUPPORT for r in results)
    disposition = (
        FiniteSetDisposition.AVAILABLE
        if present
        else FiniteSetDisposition.OUTSIDE_SUPPORT
        if outside
        else FiniteSetDisposition.UNAVAILABLE
    )
    raw_disposition = (
        ReceiptAdmissionRawDisposition.EVALUATED
        if present
        else ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT
        if outside
        else ReceiptAdmissionRawDisposition.UNEVALUABLE
    )
    bounds = []
    if present:
        for result in results:
            values = {
                v.value_id: v.value for v in (*result.response_values, *result.uncertainty_values)
            }
            for c in coordinates:
                bounds.append(
                    FiniteResponseBound(
                        f"{cell.coordinate_id}.{result.qualification_view_id}.{c.quantity_id}",
                        result.qualification_view_id,
                        c,
                        D(0),
                        values[f"interval-centre.{c.quantity_id}"],
                        D(0),
                        values[f"lower.{c.quantity_id}"],
                        values[f"upper.{c.quantity_id}"],
                        D(0),
                    )
                )
    table_identity = ObjectIdentity.from_record(table_id, table_record)
    response = FiniteResponseSet(
        f"{cell.coordinate_id}.set",
        cell,
        law.evaluator.payload,
        bindings,
        joint,
        table_identity,
        clock,
        available,
        plan.information_cutoff,
        OutcomeAccess.OUTCOME_BLIND,
        table_identity,
        table_identity,
        word_id,
        table_identity,
        disposition,
        tuple(sorted(bounds, key=lambda b: b.bound_id)),
        source_artifacts,
        () if present else (prediction_unavailable_reason,),
    )
    request = FiniteResponseSetReachabilityRequest(
        f"{cell.coordinate_id}.reach",
        response,
        task,
        word,
        tuple(ObjectIdentity.from_record(o.occurrence_id, o) for o in word.occurrences),
        plan.information_cutoff,
    )
    reach = FiniteResponseSetReachabilityMethod().evaluate(
        request, evaluator_implementation=ObjectIdentity.from_record(method.reference_id, method)
    )
    artifacts = tuple(
        sorted(
            {
                a.artifact_id: a
                for a in (*source_artifacts, law.evaluator.payload, calibration_artifact)
            }.values(),
            key=lambda a: a.artifact_id,
        )
    )
    gate_owner, utility_owner = (
        RawAdmissionGateReceiptProducer(plan.gate_receipt_producer),
        RawUtilityAdmissionReceiptProducer(plan.utility_receipt_producer),
    )
    gates, utilities = [], []
    for binding in bindings:
        view = binding.qualification_view_id
        by_quantity = {
            b.coordinate.quantity_id: b for b in bounds if b.qualification_view_id == view
        }
        upper = by_quantity[safety_receiver].upper if present else None
        lower = by_quantity[response_receiver].lower if present else None
        target = (
            min(
                min(m.lower_face_margin, m.upper_face_margin)
                for m in reach.margins
                if m.qualification_view_id == view
            )
            if present
            else None
        )
        for predicate in plan.gate_predicates:
            kind, compatibility = predicate.gate_kind, None
            if kind is G.BASELINE_PRESERVATION:
                assert predicate.upper is not None and predicate.temporal_semantics is not None
                future = TemporalPredicateAssessment(
                    f"{cell.coordinate_id}.{view}.future",
                    predicate.temporal_semantics,
                    predicate.receiver_id,
                    predicate.direction,
                    predicate.upper,
                    predicate.protected_interval,
                    ReceiverInterval(coordinate(time), coordinate(time)),
                    ObligationStatus.UNEVALUABLE,
                    None,
                    None,
                    cell.qualification_view_ids,
                    method,
                    root,
                    0,
                    evidence,
                    ("OBLIGATION_FUTURE_NOT_OBSERVED",),
                )
                compatibility = BaselinePreservationCompatibility(
                    f"{cell.coordinate_id}.{view}.semantics",
                    predicate,
                    future,
                    PreservationCompatibilityRule.EXACT_SEMANTIC_EQUALITY,
                    method,
                    tuple(e.link_id for e in evidence),
                    True,
                    (),
                )
            scalar = {
                G.TARGET: target,
                G.REACHABILITY: lower,
                G.PHYSICAL_SINK: upper,
                G.BASELINE_PRESERVATION: upper,
            }.get(kind)
            boolean = {
                G.UNCERTAINTY: qualified,
                G.EFFORT: effort_admissible,
                G.OBSERVATION_VALIDITY: causal_valid,
                G.AUTHORITY: plan.authority_boundary is not None,
                G.DYNAMICS: plan.information_cutoff.phase is CausalPhase.PRE_ACTION,
            }.get(kind)
            gates.append(
                gate_owner.produce(
                    plan=plan,
                    raw=AdmissionGateRawInput(
                        f"{cell.coordinate_id}.{view}.{kind.value.lower()}",
                        cell.coordinate_id,
                        kind,
                        raw_disposition,
                        NamedDecimal("native-operand", scalar, coordinates[0].unit)
                        if present and scalar is not None
                        else None,
                        boolean if present and scalar is None else None,
                        None,
                        binding,
                        artifacts,
                        evidence,
                        compatibility,
                    ),
                )
            )
        utilities.append(
            utility_owner.produce(
                plan=plan,
                raw=AdmissionUtilityRawInput(
                    f"{cell.coordinate_id}.{view}.utility",
                    cell.coordinate_id,
                    binding,
                    consumer.selector,
                    ReceiptAdmissionUtilityDirection.LOWER_IS_BETTER,
                    NamedDecimal("finite-utility-floor", D("-1.7976931348623157e308"), "1"),
                    projected_cost if present else None,
                    NamedDecimal("effort-origin", D(0), "1") if present else None,
                    NamedDecimal("exact-actuator-cost", D(0), "1") if present else None,
                    raw_disposition,
                    method,
                    artifacts,
                    evidence,
                ),
            )
        )
    receipt = FiniteReachabilityAdmissionReceiptProducer(plan.reachability_receipt_producer).produce(
        plan=plan,
        result=reach,
        method=method,
        admission_direction_gate_receipt_ids=tuple(
            sorted(g.receipt_id for g in gates if g.predicate.gate_kind is G.REACHABILITY)
        ),
        input_artifacts=artifacts,
        evidence_links=evidence,
    )
    return FiniteCertificateAdmissionReceiptCorpus(
        f"{plan.plan_id}.corpus",
        plan,
        tuple(sorted(gates, key=lambda g: g.receipt_id)),
        (receipt,),
        tuple(sorted(utilities, key=lambda u: u.receipt_id)),
        artifacts,
        evidence,
    )
