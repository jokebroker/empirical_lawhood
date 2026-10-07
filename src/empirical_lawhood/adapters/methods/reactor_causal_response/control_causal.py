"Map the prior qualified policy domain to the streaming native exposure chart.\n\nThe prefix obligation records finite receiver validity in the prior qualified\ncohort. It is not a claim that a future trajectory is safe or that individual\nsubsteps have independently identified response effects. Safety remains admission.\n"

from decimal import Decimal as D
from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage
from empirical_lawhood.kernel.causal_contracts import ActionStageReceiverTransport, CausalCompositionDisposition, CausalConeExistence, CausalPrefixAssessment, StreamingCausalSupportAssessment, DeliveredActionValidity, EvidenceEvaluability, NumericalViewAgreement, PredicateDirection, PrefixSupportStatus, ReceiverInterval, ResponseDirectionStatus, TemporalPredicateAssessment, TemporalPredicateSemantics
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.controller_study import ControllerActionBinding
from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import coordinate, clock_transport, FRAME, ORIGIN
from .control_plan import EmpiricalControlLawContext
from .terminal import EmpiricalQualificationResult
from .science import CLOCK, UNIT


def causal_bindings(
    context: EmpiricalControlLawContext,
    report: EmpiricalQualificationResult,
    evidence: tuple[EvidenceLink, ...],
) -> tuple[ControllerActionBinding, ...]:
    law = report.qualification.response_law
    if (
        law is None
        or not all(report.operands.adequacy)
        or len(report.operands.adequacy) != 32
        or not evidence
    ):
        raise ValueError("streaming causal map requires the complete qualified prior cohort")
    result = []
    time = context.plan.information_cutoff.coordinate
    assert time is not None
    for fibre, mapping in zip(context.plan.action_fibres, context.word_maps, strict=True):
        word = fibre.action_word
        stem = fibre.action_binding_id
        views = context.axes.qualification_view_ids
        interval = ReceiverInterval(coordinate(time), coordinate(time + D(10)))
        obligation = TemporalPredicateAssessment(
            f"{stem}.prior-finite-receiver",
            TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH,
            word.receiver_id,
            PredicateDirection.AT_LEAST,
            NamedDecimal("finite-binary64-receiver", D("-1.7976931348623157e308"), "K"),
            interval,
            interval,
            ObligationStatus.SATISFIED,
            None,
            None,
            views,
            law.evaluator,
            UNIT,
            32,
            evidence,
            (),
        )
        support = StreamingCausalSupportAssessment(
            f"{stem}.causal",
            law.relation,
            law.world_id,
            word,
            word.receiver_id,
            CLOCK,
            "s",
            FRAME,
            ORIGIN,
            tuple(
                ActionStageReceiverTransport(
                    o.occurrence_id,
                    s,
                    clock_transport(f"{o.occurrence_id}.{s.value.lower()}.receiver"),
                )
                for o in word.occurrences
                for s in ActionDeliveryStage
            ),
            coordinate(time + D(1)),
            coordinate(time + D(10)),
            EvidenceEvaluability.EVALUABLE,
            DeliveredActionValidity.VALID,
            CausalConeExistence.PRESENT,
            ResponseDirectionStatus.SUPPORTED,
            UNIT,
            32,
            views,
            evidence,
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.OUTCOME_BLIND,
            (VisibilityCeiling.PROSPECTIVE,),
            VisibilityCeiling.PROSPECTIVE,
            (),
            ObjectIdentity.from_record(mapping.command.decision_id, mapping),
        )
        prefix = CausalPrefixAssessment(
            f"{stem}.prefix",
            support,
            f"{stem}.prepared-prefix",
            coordinate(time),
            PrefixSupportStatus.SUPPORTED,
            (obligation,),
            views,
            NumericalViewAgreement.AGREED,
            (),
            CausalCompositionDisposition.DEFINED,
            (),
        )
        result.append(ControllerActionBinding(stem, word, prefix))
    return tuple(result)
