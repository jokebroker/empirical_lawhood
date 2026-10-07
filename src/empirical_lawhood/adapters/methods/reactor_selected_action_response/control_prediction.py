"""Run the installed bound evaluator before prospective native delivery."""

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
from .causal import inputs
from .control_plan import ClassicalControlContext
from .law_evaluator import ClassicalLawEvaluator, registration
from .law_terminal import RESULT_ID
from .science import HORIZON, classical_system


@dataclass(frozen=True, slots=True)
class ClassicalPredictionTable(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-prediction-table'
    table_id: str
    qualified_report: ObjectIdentity
    causal_preparation: ObjectIdentity
    prospective_decision: ObjectIdentity
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
            raise ValueError("classical prediction lost its selected word or two views")


def predict_control_table(
    context: ClassicalControlContext, reader: CandidatePayloadReader
) -> ClassicalPredictionTable:
    report = context.report
    qualification = report.qualification
    law = qualification.response_law
    assert law is not None
    publication = report.candidate.payload_publication
    reg = registration(publication.implementation)
    service = LawEvaluationService(
        reader,
        LawEvaluatorRegistry((cast(LawEvaluatorImplementation, ClassicalLawEvaluator(reg)),)),
    )
    _, features, observation, previous = inputs(context.causal)
    numbers = {f"domain-feature-{i:02d}": (D(repr(float(x))), "1") for i, x in enumerate(features)}
    numbers.update(
        {
            "callback-time": (D(repr(observation.time)), "s"),
            "observed-dose": (D(repr(observation.dose)), "kg"),
            "observed-temperature": (D(repr(observation.temperature)), "K"),
            "observed-jacket": (D(repr(observation.jacket)), "K"),
            "previous-applied-feed": (D(repr(previous[0])), "kg/s"),
            "previous-applied-jacket": (D(repr(previous[1])), "K"),
        }
    )
    values = tuple(
        LawEvaluationValue(key, key, value, unit, "reactor-native", "reactor-clock")
        for key, (value, unit) in sorted(numbers.items())
    )
    artifact = ArtifactIdentity(
        f"{context.causal.root}.causal-preparation",
        "causal-native-prefix",
        context.causal.SCHEMA,
        context.causal.fingerprint(),
        "application/json",
        len(context.causal.canonical_bytes()),
    )
    axis = report.family.axis_map.bindings[0]
    mapping = context.word_maps[1]
    requests, results = [], []
    for view in axis.qualification_view_ids:
        request = LawEvaluationRequest(
            f"{context.plan.plan_id}.word-1.{view}",
            ObjectIdentity.from_record(law.law_id, law),
            ObjectIdentity.from_record(qualification.result_id, qualification),
            ObjectIdentity.from_record(report.family.axis_map.axis_map_id, report.family.axis_map),
            publication.implementation,
            ObjectIdentity.from_record(publication.receipt_id, publication),
            LawEvaluationKind.POINT_NAMED_VALUES,
            LawEvaluationMode.ONLINE,
            reg.online_resource_envelope_id,
            None,
            law.system_id,
            law.world_id,
            law.relation.relation_id,
            law.chart_id,
            mapping.scalar_word.denominator_id,
            axis.denominator_member_id,
            axis.candidate_version_member_id,
            view,
            law.relation.history_quantity_ids,
            law.relation.receiver_quantity_ids,
            ObjectIdentity.from_record(HORIZON, law.relation.horizon),
            axis.denominator_member_id,
            mapping.scalar_word,
            values,
            (artifact,),
            ("response-values", "uncertainty-values"),
        )
        requests.append(request)
        results.append(
            service.evaluate(
                classical_system(), law, qualification, report.family.axis_map, request, publication
            )
        )
    return ClassicalPredictionTable(
        f"{context.plan.plan_id}.table",
        ObjectIdentity.from_record(RESULT_ID, report),
        ObjectIdentity.from_record(f"{context.causal.root}.causal-preparation", context.causal),
        ObjectIdentity.from_record(context.decision.decision_id, context.decision),
        tuple(requests),
        tuple(results),
    )
