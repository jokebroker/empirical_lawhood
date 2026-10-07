"The installed evaluator supplies exact decimal interval endpoints to admission."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar, cast

from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationKind,
    LawEvaluationMode,
    LawEvaluationRequest,
    LawEvaluationResult,
    LawEvaluationService,
    LawEvaluationValue,
    LawEvaluatorImplementation,
    LawEvaluatorRegistry,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .causal import causal_operands
from .config import CONTEXTS
from .control_plan import FrontierControlContext
from .law_evaluator import FrontierLawEvaluator, registration
from .science import HORIZON, frontier_system


@dataclass(frozen=True, slots=True)
class FrontierPredictionTable(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-prediction-table'
    table_id: str
    plan: ObjectIdentity
    causal: ObjectIdentity
    requests: tuple[LawEvaluationRequest, ...]
    results: tuple[LawEvaluationResult, ...]

    def __post_init__(self) -> None:
        if (
            len(self.requests) != 2
            or len(self.results) != 2
            or len({r.qualification_view_id for r in self.requests}) != 2
            or any(
                res.request != ObjectIdentity.from_record(req.request_id, req)
                for req, res in zip(self.requests, self.results, strict=True)
            )
        ):
            raise ValueError("Admission table changed the exact two-view law evaluation")


def predict(
    context: FrontierControlContext,
    reader: CandidatePayloadReader,
    causal_artifact: ArtifactIdentity,
) -> FrontierPredictionTable:
    report = context.report
    law = report.qualification.response_law
    assert law is not None
    inputs = causal_operands(context.causal, report.payload.prepared_domain)
    assert inputs is not None
    publication = report.candidate.payload_publication
    evaluator = FrontierLawEvaluator(registration(publication.implementation))
    service = LawEvaluationService(
        reader, LawEvaluatorRegistry((cast(LawEvaluatorImplementation, evaluator),))
    )
    o, p = inputs.observation, inputs.previous
    values = {
        f"domain-feature-{i:02d}": (D(repr(v)), "1") for i, v in enumerate(inputs.domain_features)
    }
    values.update(
        {
            "callback-time": (D(repr(o.time)), "s"),
            "observed-dose": (D(repr(o.dose)), "kg"),
            "observed-temperature": (D(repr(o.temperature)), "K"),
            "observed-jacket": (D(repr(o.jacket)), "K"),
            "previous-applied-feed": (D(repr(p[0])), "kg/s"),
            "previous-applied-jacket": (D(repr(p[1])), "K"),
            "frontier-context": (D(CONTEXTS.index(context.causal.context)), "1"),
        }
    )
    numbers = tuple(
        LawEvaluationValue(k, k, v, u, "reactor-native", "reactor-clock")
        for k, (v, u) in sorted(values.items())
    )
    ca = context.causal
    axis = report.family.axis_map.bindings[0]
    requests, results = [], []
    for view in axis.qualification_view_ids:
        request = LawEvaluationRequest(
            f"{context.plan.plan_id}.{view}",
            ObjectIdentity.from_record(law.law_id, law),
            ObjectIdentity.from_record(report.qualification.result_id, report.qualification),
            ObjectIdentity.from_record(report.family.axis_map.axis_map_id, report.family.axis_map),
            publication.implementation,
            ObjectIdentity.from_record(publication.receipt_id, publication),
            LawEvaluationKind.POINT_NAMED_VALUES,
            LawEvaluationMode.ONLINE,
            evaluator.registration.online_resource_envelope_id,
            None,
            frontier_system().system_id,
            law.world_id,
            law.relation.relation_id,
            law.chart_id,
            context.word.word.denominator_id,
            axis.denominator_member_id,
            axis.candidate_version_member_id,
            view,
            law.relation.history_quantity_ids,
            law.relation.receiver_quantity_ids,
            ObjectIdentity.from_record(HORIZON, law.relation.horizon),
            axis.denominator_member_id,
            context.word.word,
            numbers,
            (causal_artifact,),
            ("response-values", "uncertainty-values"),
        )
        requests.append(request)
        results.append(
            service.evaluate(
                frontier_system(),
                law,
                report.qualification,
                report.family.axis_map,
                request,
                publication,
            )
        )
    return FrontierPredictionTable(
        f"{context.plan.plan_id}.table",
        ObjectIdentity.from_record(context.plan.plan_id, context.plan),
        ObjectIdentity.from_record(f"{ca.root}.{ca.context}.causal", ca),
        tuple(requests),
        tuple(results),
    )
