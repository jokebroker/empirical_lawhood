"Finite response-law operands for existing finite-admission owners, before any parent or future.\n\nThe baseline compatibility record checks the meaning of a future preservation\nobligation. Its realized assessment is explicitly UNEVALUABLE at decision\ntime. Only the calibrated prediction supplies the prospective gate margin.\nActual work, source failures and preservation remain later controller-use observations.\n"

from decimal import Decimal as D

from empirical_lawhood.kernel.admission import AdmissionGateKind as G
from empirical_lawhood.kernel.causal_contracts import ReceiverInterval, TemporalPredicateAssessment
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.time import AvailabilitySpec, CausalPhase
from empirical_lawhood.planning.evidence_geometry import BaselinePreservationCompatibility, ReceiptAdmissionRawDisposition, ReceiptAdmissionUtilityDirection, PreservationCompatibilityRule
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionReceiptCorpus
from empirical_lawhood.planning.finite_response_geometry import FiniteJointCalibration, FiniteResponseSetReachabilityRequest, FiniteResponseSet, FiniteSetDisposition
from empirical_lawhood.adapters.geometry.finite_response_reachability import (
    FiniteReachabilityAdmissionReceiptProducer,
    FiniteResponseSetReachabilityMethod,
)
from empirical_lawhood.adapters.geometry.admission_receipts import LawMemberEvaluationBinder, AdmissionGateRawInput, AdmissionUtilityRawInput, RawAdmissionGateReceiptProducer, RawUtilityAdmissionReceiptProducer
from empirical_lawhood.adapters.methods.law_evaluation import LawEvaluationDisposition
from .consumer import FiniteResponseLawConsumerRequest, FiniteResponseLawNativeUnitReadoutMap, consumer_task, finite_bound_operands
from .control_plan import FiniteResponseLawControlLawContext, clock
from .control_prediction import FiniteResponseLawControlPredictionTable
from .method_records import FiniteResponseLawQualificationReport


def control_corpus(
    *,
    context: FiniteResponseLawControlLawContext,
    report: FiniteResponseLawQualificationReport,
    table: FiniteResponseLawControlPredictionTable,
    consumer: FiniteResponseLawConsumerRequest,
    readout_map: FiniteResponseLawNativeUnitReadoutMap,
    calibration_artifact: ArtifactIdentity,
    source_artifacts: tuple[ArtifactIdentity, ...],
    evidence_links: tuple[EvidenceLink, ...],
    operational_inputs: tuple[AdmissionGateRawInput, ...],
) -> FiniteCertificateAdmissionReceiptCorpus:
    """Derive response gates; runtime must authenticate the supplied input custody.

    This stage accepts a complete causal law table. Prefix/source inability to
    construct that table must be represented by a separate unavailable input,
    never by inventing features or a successful controller receipt.
    """
    plan = context.plan
    operational_kinds = {G.PHYSICAL_SINK, G.OBSERVATION_VALIDITY, G.DYNAMICS, G.AUTHORITY}
    operational = {
        (raw.coordinate_id, raw.law_evaluation_binding.qualification_view_id, raw.gate_kind): raw
        for raw in operational_inputs
    }
    expected_operational = {
        (c.coordinate_id, v, k)
        for c in plan.coordinates
        for v in c.qualification_view_ids
        for k in operational_kinds
    }
    if set(operational) != expected_operational or len(operational) != len(operational_inputs):
        raise ValueError("Finite response-law admission requires every actual pre-parent operational operand")
    if (
        consumer.root_id != table.root_id
        or table.qualified_report != ObjectIdentity.from_record(report.report_id, report)
        or not report.eligible_for_prospective_evaluation
        or not source_artifacts
        or not evidence_links
        or any(
            e.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or e.information_cutoff_id != plan.information_cutoff.cutoff_id
            for e in evidence_links
        )
    ):
        raise ValueError("Finite response-law admission loses its qualified boundary, root or causal input custody")
    boundary = tuple(b.boundary for b in report.calibration.boundaries).index(table.boundary)
    if report.qualifications[boundary] != context.qualification:
        raise ValueError("Finite response-law admission substitutes another qualified law")
    calibration = report.calibration.boundaries[boundary]
    method = plan.gate_predicates[0].evaluator
    law = context.qualification.response_law
    assert law is not None
    task = readout_map.task(consumer_task(consumer, readout_map.native_coordinates))
    joint = FiniteJointCalibration(
        f"{calibration.calibration_id}.finite-chart",
        calibration.identity,
        calibration.native_evaluation,
        tuple(
            sorted((f.action_word_identity for f in plan.action_fibres), key=lambda w: w.object_id)
        ),
        readout_map.coordinates,
        plan.coordinates[0].qualification_view_ids,
        D(".90"),
        calibration_artifact,
    )
    artifacts_by_id = {
        a.artifact_id: a for a in (*source_artifacts, law.evaluator.payload, calibration_artifact)
    }
    if any(
        artifacts_by_id[a.artifact_id] != a
        for a in (*source_artifacts, law.evaluator.payload, calibration_artifact)
    ):
        raise ValueError("Finite response-law admission artifact ID has conflicting identities")
    artifacts = tuple(sorted(artifacts_by_id.values(), key=lambda a: a.artifact_id))
    available = AvailabilitySpec(
        plan.information_cutoff.clock_id,
        CausalPhase.PRE_ACTION,
        OutcomeAccess.OUTCOME_BLIND,
        D(4096),
    )
    table_id = ObjectIdentity.from_record(table.table_id, table)
    gate_owner = RawAdmissionGateReceiptProducer(plan.gate_receipt_producer)
    utility_owner = RawUtilityAdmissionReceiptProducer(plan.utility_receipt_producer)
    gates, reaches, utilities = [], [], []
    for coordinate in plan.coordinates:
        word = plan.action_fibre(coordinate.action_fibre).action_word
        pairs = tuple(
            (request, result)
            for request, result in zip(table.requests, table.results, strict=True)
            if request.action_word == word
        )
        bindings = tuple(
            LawMemberEvaluationBinder().bind(
                plan=plan, coordinate_id=coordinate.coordinate_id, request=req, result=result
            )
            for req, result in pairs
        )
        dispositions = {r.disposition for _, r in pairs}
        if dispositions == {LawEvaluationDisposition.SUPPORTED}:
            disposition, raw_disposition = (
                FiniteSetDisposition.AVAILABLE,
                ReceiptAdmissionRawDisposition.EVALUATED,
            )
        elif LawEvaluationDisposition.OUTSIDE_SUPPORT in dispositions:
            disposition, raw_disposition = (
                FiniteSetDisposition.OUTSIDE_SUPPORT,
                ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT,
            )
        else:
            disposition, raw_disposition = (
                FiniteSetDisposition.UNAVAILABLE,
                ReceiptAdmissionRawDisposition.UNEVALUABLE,
            )
        operands = (
            tuple(
                finite_bound_operands(result, readout_map.native_coordinates) for _, result in pairs
            )
            if disposition is FiniteSetDisposition.AVAILABLE
            else ()
        )
        bounds = (
            readout_map.bounds(tuple(b for value in operands for b in value.bounds))
            if operands
            else ()
        )
        source = FiniteResponseSet(
            f"{consumer.request_id}.{table.boundary}.{coordinate.coordinate_id}.set",
            coordinate,
            law.evaluator.payload,
            bindings,
            joint,
            table_id,
            readout_map.identity,
            available,
            plan.information_cutoff,
            OutcomeAccess.OUTCOME_BLIND,
            table_id,
            table_id,
            ObjectIdentity.from_record(word.word_id, word),
            table_id,
            disposition,
            tuple(sorted(bounds, key=lambda b: b.bound_id)),
            source_artifacts,
            () if operands else ("FLH_LAW_RESPONSE_UNAVAILABLE",),
        )
        request = FiniteResponseSetReachabilityRequest(
            f"{source.set_id}.reach",
            source,
            task,
            word,
            tuple(ObjectIdentity.from_record(o.occurrence_id, o) for o in word.occurrences),
            plan.information_cutoff,
        )
        result = FiniteResponseSetReachabilityMethod().evaluate(
            request,
            evaluator_implementation=ObjectIdentity.from_record(method.reference_id, method),
        )
        for view_index, binding in enumerate(bindings):
            view = binding.qualification_view_id
            local = tuple(m for m in result.margins if m.qualification_view_id == view)
            containment = min(
                (min(m.lower_face_margin, m.upper_face_margin) for m in local), default=D(0)
            )
            preservation = min(
                (
                    m.upper_face_margin
                    for m in local
                    if "preservation" in m.coordinate.coordinate_id
                ),
                default=D(0),
            )
            for predicate in plan.gate_predicates:
                kind = predicate.gate_kind
                if kind in operational_kinds:
                    raw = operational[(coordinate.coordinate_id, view, kind)]
                    if (
                        raw.law_evaluation_binding != binding
                        or raw.input_artifacts != artifacts
                        or raw.evidence_links != evidence_links
                    ):
                        raise ValueError("Finite response-law operational gate substitutes its input custody")
                    gates.append(gate_owner.produce(plan=plan, raw=raw))
                    continue
                compatibility = None
                if kind is G.BASELINE_PRESERVATION:
                    assert predicate.lower is not None and predicate.temporal_semantics is not None
                    obligation = TemporalPredicateAssessment(
                        f"{source.set_id}.future-preservation-unobserved",
                        predicate.temporal_semantics,
                        predicate.receiver_id,
                        predicate.direction,
                        predicate.lower,
                        predicate.protected_interval,
                        ReceiverInterval(clock(4096), clock(4096)),
                        ObligationStatus.UNEVALUABLE,
                        None,
                        None,
                        coordinate.qualification_view_ids,
                        method,
                        table.root_id,
                        0,
                        evidence_links,
                        ("OBLIGATION_FUTURE_NOT_OBSERVED",),
                    )
                    compatibility = BaselinePreservationCompatibility(
                        f"{source.set_id}.preservation-semantics",
                        predicate,
                        obligation,
                        PreservationCompatibilityRule.EXACT_SEMANTIC_EQUALITY,
                        method,
                        tuple(e.link_id for e in evidence_links),
                        True,
                        (),
                    )
                # Effort is the frozen native-menu bound, not response utility.
                magnitude = max(abs(o.requested.value) for o in word.occurrences)
                value = (
                    containment
                    if kind in (G.TARGET, G.REACHABILITY)
                    else preservation
                    if kind is G.BASELINE_PRESERVATION
                    else operands[view_index].precision_margin
                    if kind is G.UNCERTAINTY and operands
                    else D(16) - magnitude
                    if kind is G.EFFORT
                    else D(0)
                )
                gates.append(
                    gate_owner.produce(
                        plan=plan,
                        raw=AdmissionGateRawInput(
                            f"{source.set_id}.{view}.{kind.value.lower()}",
                            coordinate.coordinate_id,
                            kind,
                            raw_disposition,
                            NamedDecimal("native-unit-margin", value, "1") if operands else None,
                            None,
                            None,
                            binding,
                            artifacts,
                            evidence_links,
                            compatibility,
                        ),
                    )
                )
            longitudinal = f"future-1.response-{'x' if consumer.direction // 2 == 0 else 'y'}"
            bound = next(
                (
                    b
                    for b in bounds
                    if b.qualification_view_id == view
                    and b.coordinate.coordinate_id == longitudinal
                ),
                None,
            )
            sign = D(1 if consumer.direction % 2 == 0 else -1)
            utilities.append(
                utility_owner.produce(
                    plan=plan,
                    raw=AdmissionUtilityRawInput(
                        f"{source.set_id}.{view}.utility",
                        coordinate.coordinate_id,
                        binding,
                        ObjectIdentity.from_record(consumer.request_id, consumer),
                        ReceiptAdmissionUtilityDirection.HIGHER_IS_BETTER,
                        NamedDecimal("minimum", D(0), "1"),
                        NamedDecimal("signed-paired-mean", sign * bound.response_delta, "1")
                        if bound
                        else None,
                        NamedDecimal("paired-origin", D(0), "1") if bound else None,
                        NamedDecimal(
                            "statistical-and-numerical",
                            (bound.upper - bound.lower) / 2 + bound.numerical_floor,
                            "1",
                        )
                        if bound
                        else None,
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
                        if g.planned_coordinate == coordinate
                        and g.predicate.gate_kind is G.REACHABILITY
                    )
                ),
                input_artifacts=artifacts,
                evidence_links=evidence_links,
            )
        )
    return FiniteCertificateAdmissionReceiptCorpus(
        f"{consumer.request_id}.{table.boundary}.corpus",
        plan,
        tuple(sorted(gates, key=lambda r: r.receipt_id)),
        tuple(sorted(reaches, key=lambda r: r.receipt_id)),
        tuple(sorted(utilities, key=lambda r: r.receipt_id)),
        artifacts,
        evidence_links,
    )
