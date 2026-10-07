"Prior finite response-law qualification causal witnesses for the frozen native chart, not future finite response-law evaluation truth.\n\nProgramme action bindings are qualification-level evidence shared by instances.\nTheir unit census is the retained calibration cohort. They must not describe a\nprospective root's unobserved parent, delivery or preservation as completed.\nThe terminal signed-response check uses the already frozen consumer-A upper\nbound. Its failure is retained: a terminal task failure is not absence of causal\ninfluence before that endpoint. Instance admission remains the separate admission owner.\n"

from decimal import Decimal as D

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage
from empirical_lawhood.kernel.causal_contracts import ActionStageReceiverTransport, CausalCompositionDisposition, CausalConeExistence, CausalPrefixAssessment, ActionStageCausalSupportAssessment, DeliveredActionValidity, EvidenceEvaluability, NumericalViewAgreement, PredicateDirection, PrefixSupportStatus, ReceiverInterval, ResponseDirectionStatus, TemporalPredicateAssessment, TemporalPredicateSemantics
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.time import CoordinateOrigin
from empirical_lawhood.planning.controller_study import ControllerActionBinding
from empirical_lawhood.adapters.simulators.finite_response_law.roster import Q_CLOCK, Q_FRAME
from .control_plan import FiniteResponseLawControlLawContext, clock
from .law_binding import NATIVE_WORDS, output_quantities
from .method_records import FiniteResponseLawQualificationReport
from .native_records import FiniteResponseLawCalibrationNativeEvaluation
from .paired_assay import paired_native_assay
from .science import FiniteResponseLawScienceSpec


def prior_causal_bindings(
    *,
    context: FiniteResponseLawControlLawContext,
    report: FiniteResponseLawQualificationReport,
    native: FiniteResponseLawCalibrationNativeEvaluation,
    evidence_links: tuple[EvidenceLink, ...],
) -> tuple[ControllerActionBinding, ...]:
    "Authenticate all 32 prior roots; retain their actual terminal failures.\n\n    This bounded map requires complete source/numerical witnesses in the\n    retained finite response-law qualification cohort. If they are absent it reports a binding obstruction;\n    it cannot filter roots, repair qualification, or impose a new finite response-law evaluation outcome\n    gate. No finite response-law evaluation native observation is an input.\n    "
    native_id = ObjectIdentity.from_record(native.evaluation_id, native)
    if (
        not report.eligible_for_prospective_evaluation
        or any(b.native_evaluation != native_id for b in report.calibration.boundaries)
        or not evidence_links
        or any(e.outcome_access is not OutcomeAccess.OUTCOME_BLIND for e in evidence_links)
        or len(native.config.projection.native_spec.roots) != 32
    ):
        raise ValueError("Finite response-law causal binding requires exact prior qualification custody and scope")
    law = context.qualification.response_law
    assert law is not None
    source = native.config.projection.native_spec
    views = context.plan.coordinates[0].qualification_view_ids
    science = FiniteResponseLawScienceSpec()
    result = []
    for fibre in context.plan.action_fibres:
        word = fibre.action_word
        index = tuple(m.controller_word for m in context.word_maps).index(word)
        force = NATIVE_WORDS[index]
        actual: list[D | None] = []
        for root in source.roots:
            root_views = tuple(v for v in native.views if v.root == root)
            assay = paired_native_assay(root_views, parent=root.assigned_parent, word=force)
            if any(not r.source_valid for r in assay.readouts):
                raise ValueError("Prior finite response-law qualification causal witness has unavailable actual delivery")
            for purpose in ("future-1", "future-2"):
                pair = tuple(r for r in assay.readouts if r.purpose == purpose)
                if tuple(r.refinement for r in pair) != (1, 2):
                    raise ValueError("Prior causal witness loses both numerical views")
                for j, (a, b) in enumerate(zip(pair[0].values, pair[1].values, strict=True)):
                    if a is None or b is None or abs(a - b) > science.delta[j] / 8:
                        raise ValueError("Prior finite response-law qualification causal/numerical compatibility is unproved")
            actual.extend(r.values[0] for r in assay.readouts)
        # Both orientations are represented as separate native words, so the
        # two corresponding upper checks retain both signed response faces.
        passed = all(v is not None and v <= science.upper[0] for v in actual)
        stem = f"{fibre.action_binding_id}.prior-law-qualification"
        domain = ReceiverInterval(clock(4096), clock(4560))
        obligation = TemporalPredicateAssessment(
            f"{stem}.terminal-response",
            TemporalPredicateSemantics.TERMINAL,
            word.receiver_id,
            PredicateDirection.AT_MOST,
            NamedDecimal(
                "frozen-consumer-a-upper", science.upper[0], output_quantities()[0].native_unit
            ),
            domain,
            domain,
            ObligationStatus.SATISFIED if passed else ObligationStatus.FAILED,
            None if passed else clock(4560),
            None,
            views,
            law.evaluator,
            "finite-response-law.calibration",
            32,
            evidence_links,
            () if passed else ("OBLIGATION_TERMINAL_RESPONSE_BOUND_FAILED",),
        )
        occurrence = word.occurrences[0]
        support = ActionStageCausalSupportAssessment(
            f"{stem}.support",
            law.relation,
            law.world_id,
            word,
            word.receiver_id,
            Q_CLOCK,
            "reference-tick",
            Q_FRAME,
            CoordinateOrigin.ABSOLUTE,
            tuple(
                ActionStageReceiverTransport(
                    occurrence.occurrence_id, stage, occurrence.requested_to_accepted
                )
                for stage in ActionDeliveryStage
            ),
            # First post-action reference boundary through the native marcher;
            # no claim about an unsampled continuous-time response onset.
            clock(4369),
            clock(4560),
            EvidenceEvaluability.EVALUABLE,
            DeliveredActionValidity.VALID,
            CausalConeExistence.PRESENT,
            ResponseDirectionStatus.SUPPORTED,
            "finite-response-law.calibration",
            32,
            views,
            evidence_links,
            EvidenceCeiling.LOCAL_LAW,
            OutcomeAccess.OUTCOME_BLIND,
            (VisibilityCeiling.PROSPECTIVE,),
            VisibilityCeiling.PROSPECTIVE,
            (),
        )
        prefix = CausalPrefixAssessment(
            f"{stem}.prefix",
            support,
            "finite-response-law.calibration.prior-prefix",
            clock(4096),
            PrefixSupportStatus.SUPPORTED,
            (obligation,),
            views,
            NumericalViewAgreement.AGREED,
            (),
            CausalCompositionDisposition.DEFINED,
            (),
        )
        result.append(ControllerActionBinding(fibre.action_binding_id, word, prefix))
    return tuple(result)
