"""Two-view offline/online correspondence through the shared law-evaluation service."""

from dataclasses import dataclass
from typing import ClassVar, cast

from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationKind,
    LawEvaluationMode,
    LawEvaluationRequest,
    LawEvaluationResult,
    LawEvaluationService,
    LawEvaluatorImplementation,
    LawEvaluatorRegistry,
)
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .control_plan import ClassicalControlContext
from .law_evaluator import ClassicalLawEvaluator, input_values, registration
from .prediction import HORIZON
from .science import staged_pulse_reactor_system


@dataclass(frozen=True, slots=True)
class ClassicalPredictionTable(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-prediction-table'
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
                result.request != ObjectIdentity.from_record(request.request_id, request)
                for request, result in zip(self.requests, self.results, strict=True)
            )
        ):
            raise ValueError("prediction table changes the exact native-view evaluation census")


def predict_control(
    context: ClassicalControlContext,
    source: ReactorBatchSource,
    reader: CandidatePayloadReader,
    causal_artifact: ArtifactIdentity,
) -> ClassicalPredictionTable:
    report = context.report
    law = report.qualification.response_law
    assert law is not None
    publication = report.candidate.payload_publication
    evaluator = ClassicalLawEvaluator(
        registration(publication.implementation), source, context.causal, report.payload
    )
    if evaluator.prediction != context.predicted:
        raise ValueError("Admission recomputation differs from the pre-response prediction seal")
    service = LawEvaluationService(
        reader, LawEvaluatorRegistry((cast(LawEvaluatorImplementation, evaluator),))
    )
    system = staged_pulse_reactor_system(report.payload.recipe.bound.coordinate)
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
            system.system_id,
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
            input_values(context.causal, report.payload),
            (causal_artifact,),
            ("response-values", "uncertainty-values"),
        )
        requests.append(request)
        results.append(
            service.evaluate(
                system, law, report.qualification, report.family.axis_map, request, publication
            )
        )
    return ClassicalPredictionTable(
        f"{context.plan.plan_id}.table",
        ObjectIdentity.from_record(context.plan.plan_id, context.plan),
        ObjectIdentity.from_record(context.causal.record_id, context.causal),
        tuple(requests),
        tuple(results),
    )
