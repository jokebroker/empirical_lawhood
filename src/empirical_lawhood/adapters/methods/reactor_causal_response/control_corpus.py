"""Empirical operands enter existing finite gate, reachability and utility owners."""

from decimal import Decimal as D
from empirical_lawhood.kernel.admission import AdmissionGateKind as G
from empirical_lawhood.kernel.causal_contracts import ReceiverInterval, TemporalPredicateAssessment
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.time import AvailabilitySpec, CausalPhase
from empirical_lawhood.planning.controller_study import FrozenFeedbackConsumer
from empirical_lawhood.planning.evidence_geometry import BaselinePreservationCompatibility, ReceiptAdmissionRawDisposition, ReceiptAdmissionUtilityDirection, PreservationCompatibilityRule
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionReceiptCorpus
from empirical_lawhood.planning.finite_response_geometry import FinitePolicyPathCalibration, FinitePolicyPathResponseSet, FiniteResponseSetReachabilityRequest, FiniteSetDisposition, FiniteResponseCoordinate, FiniteReadoutKind, FiniteBinary64ResponseBound, FiniteTaskFunctionalSpec, FiniteTaskFunctionalKind, FiniteTargetBox, FiniteTargetInterval
from empirical_lawhood.adapters.geometry.finite_response_reachability import (
    FiniteReachabilityAdmissionReceiptProducer,
    FiniteResponseSetReachabilityMethod,
)
from empirical_lawhood.adapters.geometry.admission_receipts import LawMemberEvaluationBinder, AdmissionGateRawInput, AdmissionUtilityRawInput, RawAdmissionGateReceiptProducer, RawUtilityAdmissionReceiptProducer
from empirical_lawhood.adapters.methods.law_evaluation import LawEvaluationDisposition
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import coordinate
from .control_plan import EmpiricalControlLawContext
from .control_prediction import EmpiricalCallbackInput, EmpiricalPredictionTable
from .science import CLOCK, RECEIVERS, UNITS


def control_corpus(
    *,
    context: EmpiricalControlLawContext,
    causal: EmpiricalCallbackInput,
    table: EmpiricalPredictionTable,
    consumer: FrozenFeedbackConsumer,
    calibration_artifact: ArtifactIdentity,
    source_artifacts: tuple[ArtifactIdentity, ...],
    evidence_links: tuple[EvidenceLink, ...],
) -> FiniteCertificateAdmissionReceiptCorpus:
    plan = context.plan
    if (
        table.causal_input != ObjectIdentity.from_record(causal.input_id, causal)
        or not source_artifacts
        or not evidence_links
        or any(
            e.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or e.information_cutoff_id != plan.information_cutoff.cutoff_id
            for e in evidence_links
        )
    ):
        raise ValueError("Admission causal input/evidence custody differs")
    law = context.qualification.response_law
    assert law is not None
    method = plan.gate_predicates[0].evaluator
    table_id = ObjectIdentity.from_record(table.table_id, table)
    time = causal.observation[0]
    available = AvailabilitySpec(CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, time)
    coordinates = tuple(
        sorted(
            (
                FiniteResponseCoordinate(
                    key,
                    key,
                    key,
                    consumer.command_expansion,
                    FiniteReadoutKind.WINDOW_MAXIMUM if i == 0 else FiniteReadoutKind.ENDPOINT,
                    unit,
                    "reactor-native",
                    CLOCK,
                    time if i == 0 else time + D(10),
                    time + D(10),
                )
                for i, (key, unit) in enumerate(zip(RECEIVERS, UNITS, strict=True))
            ),
            key=lambda c: c.coordinate_id,
        )
    )
    joint = FinitePolicyPathCalibration(
        f"{causal.input_id}.calibration",
        consumer.calibration,
        consumer.recipe,
        tuple(
            sorted((f.action_word_identity for f in plan.action_fibres), key=lambda w: w.object_id)
        ),
        coordinates,
        plan.coordinates[0].qualification_view_ids,
        D(".95"),
        calibration_artifact,
        ObjectIdentity.from_record(consumer.consumer_id, consumer),
    )
    temperature = next(c for c in coordinates if c.quantity_id == RECEIVERS[0])
    # This is the binary64 finite lower endpoint, not an added physical target.
    box = FiniteTargetBox(
        f"{causal.input_id}.safe",
        (FiniteTargetInterval(temperature, D("-1.7976931348623157e308"), D("356.2")),),
    )
    task = FiniteTaskFunctionalSpec(
        f"{causal.input_id}.task",
        FiniteTaskFunctionalKind.WINDOW_MAXIMUM_BOX,
        consumer.recipe,
        consumer.recipe,
        available,
        (box,),
        (box.box_id,),
    )
    items = (*source_artifacts, law.evaluator.payload, calibration_artifact)
    artifacts = tuple(
        sorted({a.artifact_id: a for a in items}.values(), key=lambda a: a.artifact_id)
    )
    if any(next(a for a in artifacts if a.artifact_id == x.artifact_id) != x for x in items):
        raise ValueError("conflicting admission input artifact identity")
    gate_owner, utility_owner = (
        RawAdmissionGateReceiptProducer(plan.gate_receipt_producer),
        RawUtilityAdmissionReceiptProducer(plan.utility_receipt_producer),
    )
    gates, reaches, utilities = [], [], []
    for i, cell in enumerate(plan.coordinates):
        mapping = context.word_maps[i]
        pairs = tuple(
            (req, res)
            for req, res in zip(table.requests, table.results, strict=True)
            if req.action_word == mapping.word
        )
        bindings = tuple(
            LawMemberEvaluationBinder().bind(
                plan=plan, coordinate_id=cell.coordinate_id, request=req, result=res
            )
            for req, res in pairs
        )
        dispositions = {res.disposition for _, res in pairs}
        present = dispositions == {LawEvaluationDisposition.SUPPORTED}
        outside = LawEvaluationDisposition.OUTSIDE_SUPPORT in dispositions
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
        for _, res in pairs:
            if not present:
                continue
            means = {v.quantity_id: float(v.value) for v in res.response_values}
            widths = {
                v.quantity_id: float(v.value)
                for v in res.uncertainty_values
                if v.value_id.startswith("halfwidth.")
            }
            for c in coordinates:
                mean, width = means[c.quantity_id], widths[c.quantity_id]
                bounds.append(
                    FiniteBinary64ResponseBound(
                        f"{cell.coordinate_id}.{res.qualification_view_id}.{c.quantity_id}",
                        res.qualification_view_id,
                        c,
                        D(0),
                        D(repr(mean)),
                        D(0),
                        D(repr(mean - width)),
                        D(repr(mean + width)),
                        D(0),
                        D(repr(width)),
                    )
                )
        source = FinitePolicyPathResponseSet(
            f"{cell.coordinate_id}.set",
            cell,
            law.evaluator.payload,
            bindings,
            joint,
            table_id,
            table_id,
            available,
            plan.information_cutoff,
            OutcomeAccess.OUTCOME_BLIND,
            table_id,
            table_id,
            ObjectIdentity.from_record(mapping.word.word_id, mapping.word),
            table_id,
            disposition,
            tuple(sorted(bounds, key=lambda b: b.bound_id)),
            source_artifacts,
            () if present else ("EMPIRICAL_PREDICTION_UNAVAILABLE",),
        )
        request = FiniteResponseSetReachabilityRequest(
            f"{cell.coordinate_id}.reach",
            source,
            task,
            mapping.word,
            tuple(ObjectIdentity.from_record(o.occurrence_id, o) for o in mapping.word.occurrences),
            plan.information_cutoff,
        )
        result = FiniteResponseSetReachabilityMethod().evaluate(
            request,
            evaluator_implementation=ObjectIdentity.from_record(method.reference_id, method),
        )
        for binding, (_, prediction) in zip(bindings, pairs, strict=True):
            view = binding.qualification_view_id
            upper = next(
                (
                    b.upper
                    for b in bounds
                    if b.qualification_view_id == view and b.coordinate.quantity_id == RECEIVERS[0]
                ),
                D(0),
            )
            safety = D("356.2") - upper
            for predicate in plan.gate_predicates:
                kind = predicate.gate_kind
                assert predicate.native_unit is not None
                compatibility = None
                if kind is G.BASELINE_PRESERVATION:
                    assert predicate.lower is not None and predicate.temporal_semantics is not None
                    future = TemporalPredicateAssessment(
                        f"{cell.coordinate_id}.{view}.future",
                        predicate.temporal_semantics,
                        predicate.receiver_id,
                        predicate.direction,
                        predicate.lower,
                        predicate.protected_interval,
                        ReceiverInterval(coordinate(time), coordinate(time)),
                        ObligationStatus.UNEVALUABLE,
                        None,
                        None,
                        cell.qualification_view_ids,
                        method,
                        causal.root,
                        0,
                        evidence_links,
                        ("OBLIGATION_FUTURE_NOT_OBSERVED",),
                    )
                    compatibility = BaselinePreservationCompatibility(
                        f"{cell.coordinate_id}.{view}.semantics",
                        predicate,
                        future,
                        PreservationCompatibilityRule.EXACT_SEMANTIC_EQUALITY,
                        method,
                        tuple(e.link_id for e in evidence_links),
                        True,
                        (),
                    )
                margin = (
                    safety
                    if kind in (G.TARGET, G.PHYSICAL_SINK, G.BASELINE_PRESERVATION, G.REACHABILITY)
                    else D(0)
                )
                if kind is G.UNCERTAINTY and present:
                    widths = {
                        v.quantity_id: float(v.value)
                        for v in prediction.uncertainty_values
                        if v.value_id.startswith("halfwidth.")
                    }
                    margin = min(
                        D(repr(0.26 - widths[RECEIVERS[0]])) / D(".26"),
                        D(repr(0.0102 - widths[RECEIVERS[1]])) / D(".0102"),
                    )
                # Causal input validation, pinned profile construction and the
                # effect-adjacent authority/resource port are prerequisites.
                gates.append(
                    gate_owner.produce(
                        plan=plan,
                        raw=AdmissionGateRawInput(
                            f"{cell.coordinate_id}.{view}.{kind.value.lower()}",
                            cell.coordinate_id,
                            kind,
                            raw_disposition,
                            NamedDecimal("native-margin", margin, predicate.native_unit)
                            if present
                            else None,
                            None,
                            None,
                            binding,
                            artifacts,
                            evidence_links,
                            compatibility,
                        ),
                    )
                )
            nominal, allowance = None, None
            if present:
                point = next(
                    float(v.value)
                    for v in prediction.response_values
                    if v.quantity_id == RECEIVERS[1]
                )
                width = next(
                    float(v.value)
                    for v in prediction.uncertainty_values
                    if v.value_id == f"halfwidth.{RECEIVERS[1]}"
                )
                dose = float(causal.observation[3])
                for _, dt, feed, _ in mapping.exposure:
                    dose += float(dt) * float(feed)
                nominal_value, robust_value = point + dose / 287.3, point - width + dose / 287.3
                nominal = NamedDecimal("nominal-progress", D(repr(nominal_value)), "1")
                allowance = NamedDecimal(
                    "binary64-calibrated-progress-width", nominal.value - D(repr(robust_value)), "1"
                )
            utilities.append(
                utility_owner.produce(
                    plan=plan,
                    raw=AdmissionUtilityRawInput(
                        f"{cell.coordinate_id}.{view}.utility",
                        cell.coordinate_id,
                        binding,
                        consumer.selector,
                        ReceiptAdmissionUtilityDirection.HIGHER_IS_BETTER,
                        NamedDecimal("finite-utility-floor", D("-1.7976931348623157e308"), "1"),
                        nominal,
                        NamedDecimal("progress-origin", D(0), "1") if present else None,
                        allowance,
                        raw_disposition,
                        method,
                        artifacts,
                        evidence_links,
                    ),
                )
            )
        reaches.append(
            FiniteReachabilityAdmissionReceiptProducer(plan.reachability_receipt_producer).produce(
                plan=plan,
                result=result,
                method=method,
                admission_direction_gate_receipt_ids=tuple(
                    sorted(
                        g.receipt_id
                        for g in gates
                        if g.planned_coordinate == cell and g.predicate.gate_kind is G.REACHABILITY
                    )
                ),
                input_artifacts=artifacts,
                evidence_links=evidence_links,
            )
        )
    return FiniteCertificateAdmissionReceiptCorpus(
        f"{causal.input_id}.corpus",
        plan,
        tuple(sorted(gates, key=lambda r: r.receipt_id)),
        tuple(sorted(reaches, key=lambda r: r.receipt_id)),
        tuple(sorted(utilities, key=lambda r: r.receipt_id)),
        artifacts,
        evidence_links,
    )
