"""Qualified word-to-receiver support, retaining every native delivery stage."""

from decimal import Decimal as D

from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import coordinate, clock_transport, FRAME, ORIGIN
from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage
from empirical_lawhood.kernel.causal_contracts import ActionStageReceiverTransport, CausalCompositionDisposition, CausalConeExistence, CausalPrefixAssessment, StreamingCausalSupportAssessment, DeliveredActionValidity, EvidenceEvaluability, NumericalViewAgreement, PredicateDirection, PrefixSupportStatus, ReceiverInterval, ResponseDirectionStatus, TemporalPredicateAssessment, TemporalPredicateSemantics
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.controller_study import ControllerActionBinding
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionReceiptProductionPlan


def finite_feed_causal_binding(
    *,
    plan: ReceiptAdmissionReceiptProductionPlan,
    law: ResponseLaw,
    word: OccurrenceActionWord,
    native_mapping: ObjectIdentity,
    views: tuple[str, ...],
    unit: str,
    guard_s: int,
    evidence: tuple[EvidenceLink, ...],
) -> ControllerActionBinding:
    if not evidence or guard_s not in (120, 240):
        raise ValueError("finite word support lacks its evidence or declared native guard")
    fibre = plan.action_fibres[0]
    stem, time = fibre.action_binding_id, plan.information_cutoff.coordinate
    assert time is not None
    interval = ReceiverInterval(coordinate(time), coordinate(time + guard_s))
    predicate = TemporalPredicateAssessment(
        f"{stem}.prior-finite-receiver",
        TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH,
        word.receiver_id,
        PredicateDirection.AT_LEAST,
        NamedDecimal("finite-native-receiver", D("-1.7976931348623157e308"), "K"),
        interval,
        interval,
        ObligationStatus.SATISFIED,
        None,
        None,
        views,
        law.evaluator,
        unit,
        96,
        evidence,
        (),
    )
    support = StreamingCausalSupportAssessment(
        f"{stem}.causal",
        law.relation,
        law.world_id,
        word,
        word.receiver_id,
        "reactor-clock",
        "s",
        FRAME,
        ORIGIN,
        tuple(
            ActionStageReceiverTransport(
                o.occurrence_id, s, clock_transport(f"{o.occurrence_id}.{s.value.lower()}.receiver")
            )
            for o in word.occurrences
            for s in ActionDeliveryStage
        ),
        coordinate(time + 1),
        coordinate(time + guard_s),
        EvidenceEvaluability.EVALUABLE,
        DeliveredActionValidity.VALID,
        CausalConeExistence.PRESENT,
        ResponseDirectionStatus.SUPPORTED,
        unit,
        96,
        views,
        evidence,
        EvidenceCeiling.LOCAL_LAW,
        OutcomeAccess.OUTCOME_BLIND,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
        (),
        native_mapping,
    )
    prefix = CausalPrefixAssessment(
        f"{stem}.prefix",
        support,
        f"{stem}.prepared-prefix",
        coordinate(time),
        PrefixSupportStatus.SUPPORTED,
        (predicate,),
        views,
        NumericalViewAgreement.AGREED,
        (),
        CausalCompositionDisposition.DEFINED,
        (),
    )
    return ControllerActionBinding(stem, word, prefix)
