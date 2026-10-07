"""Use the installed law service for every D scalar word and numerical view."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar, cast

from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationKind,
    LawEvaluationDisposition,
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

from .causal_contexts import causal_contexts
from .control_plan import RegimeControlLawContext
from .law_evaluator import RegimeJointLawEvaluator, registration
from .law_terminal import RESULT_ID, RegimeJointLawResult
from .science import HORIZON, regime_system


@dataclass(frozen=True, slots=True)
class RegimeControlPredictionTable(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-control-prediction-table'

    table_id: str
    qualified_report: ObjectIdentity
    causal_preparation: ObjectIdentity
    prospective_decision: ObjectIdentity
    requests: tuple[LawEvaluationRequest, ...]
    results: tuple[LawEvaluationResult, ...]

    def __post_init__(self) -> None:
        if len(self.requests) != 6 or len(self.results) != 6:
            raise ValueError("D law query lost three native words or two numerical views")
        seen = set()
        for request, result in zip(self.requests, self.results, strict=True):
            if (
                request.action_word is None
                or result.request != ObjectIdentity.from_record(request.request_id, request)
                or result.action_word
                != ObjectIdentity.from_record(request.action_word.word_id, request.action_word)
            ):
                raise ValueError("D installed law result changed its actual scalar word")
            seen.add((request.action_word.word_id, request.qualification_view_id))
        if len(seen) != 6 or len({item[0] for item in seen}) != 3 or len({item[1] for item in seen}) != 2:
            raise ValueError("D law query lost a distinct word or view")


def predict_control_table(
    context: RegimeControlLawContext,
    reader: CandidatePayloadReader,
) -> RegimeControlPredictionTable:
    report: RegimeJointLawResult = context.report
    qualification = report.qualification
    law = qualification.response_law
    assert law is not None
    system = context.plan.model_set
    if system.target_world_id != law.world_id:
        raise ValueError("D law query changed its frozen evidence world")
    publication = report.candidate.payload_publication
    reg = registration(publication.implementation)
    service = LawEvaluationService(
        reader,
        LawEvaluatorRegistry((cast(
            LawEvaluatorImplementation, RegimeJointLawEvaluator(reg)
        ),)),
    )
    selected = next(
        value for value in causal_contexts(context.causal)
        if value.name == context.decision.route
    )
    if (
        selected.callback != context.decision.callback
        or selected.history is None
        or selected.input_sha256 is None
    ):
        raise ValueError("D law query changed the sealed causal feature prefix")
    callback = selected.callback
    assert callback is not None
    native = context.causal.arrays.unpack()
    stem = "exploration_unshifted" if context.decision.route == "prepared_t0" else context.decision.route[0]
    observation = native[f"{stem}_v0_observations"][callback]
    previous = native[f"{stem}_v0_stages"][callback - 1]
    numbers = {
        f"feature-{index:02d}": (D(repr(value)), "1")
        for index, value in enumerate(selected.history)
    }
    numbers.update({
        "callback-time": (D(callback * 10), "s"),
        "observed-dose": (D(repr(float(observation[3]))), "kg"),
        "previous-applied-feed": (D(repr(float(previous[2]))), "kg/s"),
        "previous-applied-jacket": (D(repr(float(previous[3]))), "K"),
        "reactor-fixed-jacket": (context.word_maps[0].fixed_jacket_K, "K"),
        "reactor-regime-preparation-policy-version": (
            D({"prepared_t0": 0, "c_q": 1, "p_q": 2}[context.decision.route]), "1"
        ),
    })
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
    requests = []
    results = []
    for index, mapping in enumerate(context.word_maps):
        for view in axis.qualification_view_ids:
            request = LawEvaluationRequest(
                f"{context.plan.plan_id}.word-{index}.{view}",
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
            results.append(service.evaluate(
                regime_system(), law, qualification, report.family.axis_map,
                request, publication,
            ))
    for index, forecast in enumerate(context.decision.primary_words):
        for result in (results[2 * index], results[2 * index + 1]):
            if not forecast.supported or not forecast.projection_valid:
                if result.disposition is LawEvaluationDisposition.SUPPORTED:
                    raise ValueError("D installed law admitted a word outside its sealed support")
                continue
            if result.disposition is not LawEvaluationDisposition.SUPPORTED:
                raise ValueError("D installed law refused a sealed supported word")
            means = {value.quantity_id: value.value for value in result.response_values}
            widths = {
                value.quantity_id: value.value
                for value in result.uncertainty_values
                if value.value_id.startswith("halfwidth.")
            }
            # The seal is a binary64 native forecast; the installed law uses
            # canonical Decimal operands.  Round-trip arithmetic can differ
            # by a few final bits without changing any physical bound.
            def same_native_value(observed: D | None, sealed: D) -> bool:
                return observed is not None and abs(observed - sealed) <= D("1e-12")

            if (
                not same_native_value(means.get("reactor-peak-temperature"), forecast.predicted_peak_K)
                or not same_native_value(means.get("reactor-feed-cooling"), forecast.predicted_cooling_K)
                or not same_native_value(widths.get("reactor-peak-temperature"), forecast.temperature_halfwidth_K)
                or not same_native_value(widths.get("reactor-feed-cooling"), forecast.cooling_halfwidth_K)
            ):
                raise ValueError("D installed law changed the pre-outcome sealed action intervals")
    return RegimeControlPredictionTable(
        f"{context.plan.plan_id}.law-prediction-table",
        ObjectIdentity.from_record(RESULT_ID, report),
        ObjectIdentity.from_record(f"{context.causal.root}.causal-preparation", context.causal),
        ObjectIdentity.from_record(context.decision.decision_id, context.decision),
        tuple(requests),
        tuple(results),
    )
