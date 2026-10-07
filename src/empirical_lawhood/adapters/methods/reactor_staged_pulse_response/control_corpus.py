"Strict selected-law and external-task boxes for shared finite admission production."

from decimal import Decimal as D

from empirical_lawhood.adapters.control.finite_response_corpus import finite_interval_corpus
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.planning.controller_study import FrozenFeedbackConsumer
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionReceiptCorpus
from empirical_lawhood.planning.finite_response_geometry import FiniteTargetInterval
from .control_plan import ClassicalControlContext
from .control_prediction import ClassicalPredictionTable
from .control_records import ClassicalClockMap, response_coordinates


def control_corpus(
    *,
    context: ClassicalControlContext,
    table: ClassicalPredictionTable,
    consumer: FrozenFeedbackConsumer,
    calibration_artifact: ArtifactIdentity,
    source_artifacts: tuple[ArtifactIdentity, ...],
    evidence: tuple[EvidenceLink, ...],
) -> FiniteCertificateAdmissionReceiptCorpus:
    ca, report, plan = context.causal, context.report, context.plan
    assert ca.callback is not None
    clock = ClassicalClockMap(
        ca.root,
        ca.callback,
        ca.first_callback if ca.first_callback is not None else ca.callback,
        context.readout,
    )
    if (
        table.plan != ObjectIdentity.from_record(plan.plan_id, plan)
        or table.causal != ObjectIdentity.from_record(ca.record_id, ca)
        or not any(
            a.sha256 == clock.fingerprint() and a.payload_schema == clock.SCHEMA
            for a in source_artifacts
        )
    ):
        raise ValueError("Admission lost its exact causal clock/readout or prediction table")
    law = report.qualification.response_law
    assert law is not None
    coordinates = response_coordinates(context.readout)
    intervals = []
    for coordinate in coordinates:
        key = coordinate.quantity_id
        lower, upper = context.predicted.interval(key)
        if key.startswith("s"):
            upper = min(upper, D("356.2"))
        elif key in ("c", "c1"):
            lower = max(lower, context.request.required_K)
        elif key == "c2":
            assert context.request.second_K is not None
            lower = max(lower, context.request.second_K)
        elif key == "g":
            lower = max(lower, D(0))
        # A request stronger than this law stays a well-formed target outside
        # its predicted set, so the existing owner can produce NONATTEMPT.
        intervals.append(FiniteTargetInterval(coordinate, lower, max(lower, upper)))
    primary = context.word.word.receiver_id
    mass = context.predicted.projected_mass_kg
    assert mass is not None
    return finite_interval_corpus(
        plan=plan,
        law=law,
        table_record=table,
        table_id=table.table_id,
        requests=table.requests,
        results=table.results,
        word=context.word.word,
        qualification=report.payload.qualification,
        qualified=report.operands.qualifies,
        root=ca.root,
        causal_valid=not ca.reasons,
        effort_admissible=mass <= context.request.budget_kg + D("1e-10"),
        projected_cost=NamedDecimal("projected-applied-mass-normalized", mass / D("2.10"), "1"),
        response_receiver=primary,
        safety_receiver="s",
        coordinates=coordinates,
        clock=ObjectIdentity.from_record(clock.map_id, clock),
        target_intervals=tuple(intervals),
        consumer=consumer,
        calibration_artifact=calibration_artifact,
        source_artifacts=source_artifacts,
        evidence=evidence,
        prediction_unavailable_reason="CLASSICAL_LAW_PREDICTION_UNAVAILABLE",
    )
