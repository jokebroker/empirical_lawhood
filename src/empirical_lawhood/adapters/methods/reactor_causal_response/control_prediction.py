"""Causal callback measurements and the complete ordinary law-service query table."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar, cast
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationRequest,
    LawEvaluationResult,
    LawEvaluationService,
    LawEvaluatorRegistry,
    LawEvaluatorImplementation,
    LawEvaluationMode,
    LawEvaluationKind,
    LawEvaluationValue,
)
from .control_plan import EmpiricalControlLawContext
from .evaluator import EmpiricalEvaluator, registration
from .terminal import EmpiricalQualificationResult
from .science import CLOCK


@dataclass(frozen=True, slots=True)
class EmpiricalCallbackInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-callback-input'
    input_id: str
    root: str
    callback: int
    observation: tuple[D, D, D, D]
    previous_applied: tuple[D, D]
    history_sha256: str
    previous_tick: ObjectIdentity | None
    features: tuple[tuple[D, ...], ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        validate_stable_id(self.root, field_name="root")
        validate_sha256(self.history_sha256, field_name="history_sha256")
        if (
            self.callback not in range(2880)
            or self.observation[0] != self.callback * 10
            or len(self.features) != 9
            or any(len(x) != 23 for x in self.features)
        ):
            raise ValueError("callback causal census differs")
        if (self.callback == 0) != (self.previous_tick is None):
            raise ValueError("callback loses its previous actual owner tick")


@dataclass(frozen=True, slots=True)
class EmpiricalPredictionTable(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-causal-response/empirical-prediction-table'
    table_id: str
    causal_input: ObjectIdentity
    qualified_parent: ObjectIdentity
    requests: tuple[LawEvaluationRequest, ...]
    results: tuple[LawEvaluationResult, ...]

    def __post_init__(self) -> None:
        if len(self.requests) != 18 or len(self.results) != 18:
            raise ValueError("prediction table requires all nine actions and both views")
        for request, result in zip(self.requests, self.results, strict=True):
            if result.request != ObjectIdentity.from_record(request.request_id, request):
                raise ValueError("prediction result substitutes its request")


def predict_table(
    context: EmpiricalControlLawContext,
    report: EmpiricalQualificationResult,
    causal: EmpiricalCallbackInput,
    reader: CandidatePayloadReader,
) -> EmpiricalPredictionTable:
    law = context.qualification.response_law
    assert law is not None
    publication = context.publication
    reg = registration(publication.implementation)
    service = LawEvaluationService(
        reader, LawEvaluatorRegistry((cast(LawEvaluatorImplementation, EmpiricalEvaluator(reg)),))
    )
    axis = context.axes.bindings[0]
    requests, results = [], []
    for i, mapping in enumerate(context.word_maps):
        numbers = {f"feature-{j:02d}": (x, "1") for j, x in enumerate(causal.features[i])}
        numbers.update(
            {
                "callback-time": (causal.observation[0], "s"),
                "action-id": (D(i), "1"),
                "previous-applied-feed": (causal.previous_applied[0], "kg/s"),
                "previous-applied-jacket": (causal.previous_applied[1], "K"),
                "observed-dose": (causal.observation[3], "kg"),
            }
        )
        values = tuple(
            LawEvaluationValue(key, key, value, unit, "reactor-native", CLOCK)
            for key, (value, unit) in sorted(numbers.items())
        )
        for view in axis.qualification_view_ids:
            request = LawEvaluationRequest(
                f"{causal.input_id}.{i}.{view}",
                ObjectIdentity.from_record(law.law_id, law),
                ObjectIdentity.from_record(context.qualification.result_id, context.qualification),
                ObjectIdentity.from_record(context.axes.axis_map_id, context.axes),
                publication.implementation,
                ObjectIdentity.from_record(publication.receipt_id, publication),
                LawEvaluationKind.POINT_NAMED_VALUES,
                LawEvaluationMode.ONLINE,
                reg.online_resource_envelope_id,
                None,
                context.system.system_id,
                context.system.world.world_id,
                law.relation.relation_id,
                law.chart_id,
                law.obligations.support.denominator_cell_ids[0],
                axis.denominator_member_id,
                axis.candidate_version_member_id,
                view,
                law.relation.history_quantity_ids,
                law.relation.receiver_quantity_ids,
                ObjectIdentity.from_record(law.relation.horizon.horizon_id, law.relation.horizon),
                axis.denominator_member_id,
                mapping.word,
                values,
                (),
                ("response-values", "uncertainty-values"),
            )
            requests.append(request)
            results.append(
                service.evaluate(
                    context.system, law, context.qualification, context.axes, request, publication
                )
            )
    return EmpiricalPredictionTable(
        f"{causal.input_id}.table",
        ObjectIdentity.from_record(causal.input_id, causal),
        ObjectIdentity.from_record("reactor-empirical-qualification", report),
        tuple(requests),
        tuple(results),
    )
