"""Native interval operands enter the unchanged finite reachability/gate owners."""

from decimal import Decimal as D

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.planning.controller_study import FrozenFeedbackConsumer
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionReceiptCorpus
from empirical_lawhood.planning.finite_response_geometry import FiniteTargetBox, FiniteTargetInterval
from .control_plan import FrontierControlContext
from .control_prediction import FrontierPredictionTable
from .control_records import FrontierClockMap, FrontierReadout, response_coordinates
from .science import RECEIVERS


from empirical_lawhood.adapters.control.finite_response_corpus import finite_interval_corpus


def control_corpus(
    *,
    context: FrontierControlContext,
    table: FrontierPredictionTable,
    consumer: FrozenFeedbackConsumer,
    calibration_artifact: ArtifactIdentity,
    source_artifacts: tuple[ArtifactIdentity, ...],
    evidence: tuple[EvidenceLink, ...],
) -> FiniteCertificateAdmissionReceiptCorpus:
    plan, ca, report = context.plan, context.causal, context.report
    assert ca.callback is not None
    if (
        table.plan != ObjectIdentity.from_record(plan.plan_id, plan)
        or table.causal != ObjectIdentity.from_record(f"{ca.root}.{ca.context}.causal", ca)
        or calibration_artifact.sha256 != report.payload.qualification.object_fingerprint
        or any(a not in source_artifacts for r in table.requests for a in r.input_artifacts)
        or not evidence
        or any(
            e.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or e.information_cutoff_id != plan.information_cutoff.cutoff_id
            for e in evidence
        )
    ):
        raise ValueError("finite admission changed its causal evidence or action-specific calibration")
    law = report.qualification.response_law
    assert law is not None
    readout = FrontierReadout(ca.context, context.request.horizon_s)
    clock = FrontierClockMap(ca.root, ca.callback, readout)
    if not any(
        a.sha256 == clock.fingerprint() and a.payload_schema == clock.SCHEMA
        for a in source_artifacts
    ):
        raise ValueError("Admission lacks its persisted inner-window/guard clock map")
    coordinates = response_coordinates(readout)
    box = FiniteTargetBox(
        f"{plan.plan_id}.box",
        tuple(
            sorted(
                (
                    FiniteTargetInterval(
                        c,
                        D(0) if c.quantity_id == RECEIVERS[0] else context.request.request_K,
                        D("356.2")
                        if c.quantity_id == RECEIVERS[0]
                        else D("1.7976931348623157e308"),
                    )
                    for c in coordinates
                ),
                key=lambda v: v.coordinate.coordinate_id,
            )
        ),
    )
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
        effort_admissible=context.word.applied_mass_kg <= context.request.budget_kg + D("1e-10"),
        projected_cost=NamedDecimal(
            "projected-applied-mass-normalized", context.word.applied_mass_kg / D("2.10"), "1"
        ),
        response_receiver=RECEIVERS[1],
        safety_receiver=RECEIVERS[0],
        coordinates=coordinates,
        clock=ObjectIdentity.from_record(clock.map_id, clock),
        target_intervals=box.intervals,
        consumer=consumer,
        calibration_artifact=calibration_artifact,
        source_artifacts=source_artifacts,
        evidence=evidence,
        prediction_unavailable_reason="FRONTIER_LAW_PREDICTION_UNAVAILABLE",
    )
