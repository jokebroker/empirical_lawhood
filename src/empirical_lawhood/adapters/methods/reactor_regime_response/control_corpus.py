"Reactor cooling operands enter the installed finite admission receipt owners."

from decimal import Decimal as D
from math import isfinite
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
from empirical_lawhood.planning.finite_response_geometry import FiniteJointCalibration, FiniteResponseSet, FiniteResponseSetReachabilityRequest, FiniteSetDisposition, FiniteResponseCoordinate, FiniteReadoutKind, FiniteBinary64ResponseBound, FiniteTaskFunctionalSpec, FiniteTaskFunctionalKind, FiniteTargetBox, FiniteTargetInterval
from empirical_lawhood.adapters.geometry.finite_response_reachability import (
    FiniteReachabilityAdmissionReceiptProducer,
    FiniteResponseSetReachabilityMethod,
)
from empirical_lawhood.adapters.geometry.admission_receipts import LawMemberEvaluationBinder, AdmissionGateRawInput, AdmissionUtilityRawInput, RawAdmissionGateReceiptProducer, RawUtilityAdmissionReceiptProducer
from empirical_lawhood.adapters.methods.law_evaluation import LawEvaluationDisposition
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import coordinate
from .control_plan import RegimeControlLawContext
from .control_prediction import RegimeControlPredictionTable
from .control_clock import LOCAL_CLOCK, LOCAL_FRAME, ReactorLocalClockMap
from .science import RECEIVERS

CLOCK = "reactor-clock"
UNITS = ("K", "K")


def control_corpus(
    *,
    context: RegimeControlLawContext,
    table: RegimeControlPredictionTable,
    consumer: FrozenFeedbackConsumer,
    calibration_artifact: ArtifactIdentity,
    source_artifacts: tuple[ArtifactIdentity, ...],
    evidence_links: tuple[EvidenceLink, ...],
) -> FiniteCertificateAdmissionReceiptCorpus:
    plan = context.plan
    if (
        table.causal_preparation
        != ObjectIdentity.from_record(f"{context.causal.root}.causal-preparation", context.causal)
        or table.prospective_decision
        != ObjectIdentity.from_record(context.decision.decision_id, context.decision)
        or not source_artifacts
        or not evidence_links
        or any(
            e.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or e.information_cutoff_id != plan.information_cutoff.cutoff_id
            for e in evidence_links
        )
    ):
        raise ValueError("Admission causal input/evidence custody differs")
    law = context.report.qualification.response_law
    assert law is not None
    method = plan.gate_predicates[0].evaluator
    table_id = ObjectIdentity.from_record(table.table_id, table)
    time = D(context.decision.callback * 10)
    clock_map = ReactorLocalClockMap(
        f"{context.causal.root}.callback-{context.decision.callback:04d}.local-clock",
        context.causal.root,
        context.decision.callback,
        time,
    )
    if not any(
        artifact.payload_schema == clock_map.SCHEMA
        and artifact.sha256 == clock_map.fingerprint()
        for artifact in source_artifacts
    ):
        raise ValueError("Admission lacks its persisted local-to-native clock map")
    prefix = "exploration_unshifted" if context.decision.route == "prepared_t0" else context.decision.route[0]
    observed = context.causal.arrays.unpack()[f"{prefix}_v0_observations"][context.decision.callback]
    observed_dose = float(observed[3])
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
                    LOCAL_FRAME,
                    LOCAL_CLOCK,
                    D(0) if i == 0 else D(10),
                    D(10),
                )
                for i, (key, unit) in enumerate(zip(RECEIVERS, UNITS, strict=True))
            ),
            key=lambda c: c.coordinate_id,
        )
    )
    joint = FiniteJointCalibration(
        f"{plan.plan_id}.calibration",
        consumer.calibration,
        consumer.recipe,
        tuple(
            sorted((f.action_word_identity for f in plan.action_fibres), key=lambda w: w.object_id)
        ),
        coordinates,
        plan.coordinates[0].qualification_view_ids,
        D(".90"),
        calibration_artifact,
    )
    temperature = next(c for c in coordinates if c.quantity_id == RECEIVERS[0])
    cooling = next(c for c in coordinates if c.quantity_id == RECEIVERS[1])
    # This is the binary64 finite lower endpoint, not an added physical target.
    box = FiniteTargetBox(
        f"{plan.plan_id}.task-box",
        tuple(sorted((
            FiniteTargetInterval(temperature, D("-1.7976931348623157e308"), D("356.2")),
            FiniteTargetInterval(cooling, context.request_K, D("1.7976931348623157e308")),
        ), key=lambda value: value.coordinate.coordinate_id)),
    )
    task = FiniteTaskFunctionalSpec(
        f"{plan.plan_id}.task",
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
    for i, cell in enumerate(plan.coordinates, start=1):
        mapping = context.word_maps[i]
        pairs = tuple(
            (req, res)
            for req, res in zip(table.requests, table.results, strict=True)
            if req.action_word == mapping.scalar_word
        )
        if len(pairs) != len(cell.qualification_view_ids):
            raise ValueError("Admission lacks an installed law result for every numerical view")
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
        source = FiniteResponseSet(
            f"{cell.coordinate_id}.set",
            cell,
            law.evaluator.payload,
            bindings,
            joint,
            table_id,
            ObjectIdentity.from_record(clock_map.map_id, clock_map),
            available,
            plan.information_cutoff,
            OutcomeAccess.OUTCOME_BLIND,
            table_id,
            table_id,
            ObjectIdentity.from_record(mapping.scalar_word.word_id, mapping.scalar_word),
            table_id,
            disposition,
            tuple(sorted(bounds, key=lambda b: b.bound_id)),
            source_artifacts,
            () if present else ("REACTOR_JOINT_LAW_PREDICTION_UNAVAILABLE",),
        )
        request = FiniteResponseSetReachabilityRequest(
            f"{cell.coordinate_id}.reach",
            source,
            task,
            mapping.scalar_word,
            tuple(
                ObjectIdentity.from_record(o.occurrence_id, o)
                for o in mapping.scalar_word.occurrences
            ),
            plan.information_cutoff,
        )
        result = FiniteResponseSetReachabilityMethod().evaluate(
            request,
            evaluator_implementation=ObjectIdentity.from_record(method.reference_id, method),
        )
        for binding, (_, prediction) in zip(bindings, pairs, strict=True):
            view = binding.qualification_view_id
            upper = next(
                b.upper for b in bounds
                if b.qualification_view_id == view
                and b.coordinate.quantity_id == RECEIVERS[0]
            ) if present else None
            lower_cooling = next(
                b.lower for b in bounds
                if b.qualification_view_id == view
                and b.coordinate.quantity_id == RECEIVERS[1]
            ) if present else None
            target_margin = next(
                min(m.lower_face_margin, m.upper_face_margin)
                for m in result.margins
                if m.qualification_view_id == view
                and m.coordinate.quantity_id == RECEIVERS[0]
            ) if present else None
            if present:
                assert target_margin is not None
                target_margin = min(
                    target_margin,
                    next(
                        min(m.lower_face_margin, m.upper_face_margin)
                        for m in result.margins
                        if m.qualification_view_id == view
                        and m.coordinate.quantity_id == RECEIVERS[1]
                    ),
                )
            q_widths = {
                value.quantity_id: value.value
                for value in prediction.uncertainty_values
                if value.value_id.startswith("halfwidth.")
            }
            q_t = context.report.payload.q_temperature
            q_c = context.report.payload.q_cooling
            uncertainty_ok = (
                present
                and q_t is not None and q_t <= 1
                and q_c is not None and q_c <= 1
                and set(q_widths) == set(RECEIVERS)
                and all(value >= 0 for value in q_widths.values())
            )
            for predicate in plan.gate_predicates:
                kind = predicate.gate_kind
                compatibility = None
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
                        context.causal.root,
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
                scalar = {
                    G.TARGET: target_margin,
                    G.REACHABILITY: lower_cooling,
                    G.PHYSICAL_SINK: upper,
                    G.BASELINE_PRESERVATION: upper,
                }.get(kind)
                boolean = {
                    G.UNCERTAINTY: uncertainty_ok,
                    G.EFFORT: (
                        bool(mapping.full_native.exposure)
                        and context.decision.primary_words[i].projection_valid
                        and context.decision.primary_words[i].delivered_mass_kg > 0
                        and 600 <= float(time) <= 25200
                        and observed_dose + float(context.decision.primary_words[i].delivered_mass_kg) <= 287.3
                    ),
                    G.OBSERVATION_VALIDITY: (
                        context.decision.causal_preparation_valid
                        and isfinite(observed_dose)
                        and context.decision.primary_words[i].projection_valid
                    ),
                    G.AUTHORITY: bool(evidence_links),
                    G.DYNAMICS: (
                        plan.information_cutoff.phase is CausalPhase.PRE_ACTION
                        and context.decision.primary_words[i].supported
                    ),
                }.get(kind)
                gates.append(
                    gate_owner.produce(
                        plan=plan,
                        raw=AdmissionGateRawInput(
                            f"{cell.coordinate_id}.{view}.{kind.value.lower()}",
                            cell.coordinate_id,
                            kind,
                            raw_disposition,
                            NamedDecimal("native-observation", scalar, "K")
                            if present and scalar is not None else None,
                            boolean if present and scalar is None else None,
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
                nominal_value = sum(
                    float(duration) * float(feed)
                    for _, duration, feed, _ in mapping.full_native.exposure
                )
                nominal = NamedDecimal("projected-delivered-mass", D(repr(nominal_value)), "kg")
                allowance = NamedDecimal(
                    "exact-projected-mass-uncertainty", D(0), "kg"
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
                        NamedDecimal("finite-utility-floor", D("-1.7976931348623157e308"), "kg"),
                        nominal,
                        NamedDecimal("mass-origin", D(0), "kg") if present else None,
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
        f"{plan.plan_id}.corpus",
        plan,
        tuple(sorted(gates, key=lambda r: r.receipt_id)),
        tuple(sorted(reaches, key=lambda r: r.receipt_id)),
        tuple(sorted(utilities, key=lambda r: r.receipt_id)),
        artifacts,
        evidence_links,
    )
