"""Full pre-parent native tables through the existing qualified-law service."""

from dataclasses import dataclass
from decimal import Decimal as D
from hashlib import sha256
from typing import ClassVar, cast

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
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
from empirical_lawhood.adapters.simulators.finite_response_law.randomness import _assigned_stage_unit
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedEvaluationTaskResult, FiniteResponseLawEvaluationTaskResult
from .control_plan import FiniteResponseLawControlLawContext
from .law_binding import HORIZON, feature_quantities, parent_quantity
from .law_evaluator import FiniteResponseLawLawEvaluator, law_registration, PRODUCTS
from .method_records import FiniteResponseLawAssignedQualificationReport, FiniteResponseLawQualificationReport
from .law_payloads import FiniteResponseLawConditionalPayload, MAXIMUM_LAW_BYTES


@dataclass(frozen=True, slots=True)
class FiniteResponseLawControlPredictionTable(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-control-prediction-table'
    table_id: str
    root_id: str
    boundary: str
    qualified_report: ObjectIdentity
    prefix_source: ObjectIdentity
    prefix_artifacts: tuple[ArtifactIdentity, ...]
    predicted_interface: tuple[D, ...] | None
    requests: tuple[LawEvaluationRequest, ...]
    results: tuple[LawEvaluationResult, ...]
    # Unqualified point arithmetic is retained even outside admission support.
    # Information loss must not become a support-filtered complete-case result.
    point_mean: tuple[D, ...] | None

    def __post_init__(self) -> None:
        validate_stable_id(self.table_id, field_name="table_id")
        validate_stable_id(self.root_id, field_name="root_id")
        assigned = _assigned_stage_unit(self.root_id, "prospective-evaluation", 64)
        if (
            not assigned
            and self.root_id not in {f"prospective-evaluation.r{i:03d}" for i in range(64)}
            or self.boundary not in ("composed", "cached", "direct")
            or len(self.requests) != 16
            or len(self.results) != 16
        ):
            raise ValueError("Finite response-law prediction must retain every native word and both views")
        if self.qualified_report.object_schema != (
            FiniteResponseLawAssignedQualificationReport if assigned else FiniteResponseLawQualificationReport
        ).SCHEMA:
            raise ValueError("Finite response-law control table loses its qualification source")
        if self.point_mean is not None and (
            len(self.point_mean) != 32
            or any(type(v) is not D or not v.is_finite() for v in self.point_mean)
        ):
            raise ValueError("Finite response-law evaluation point snapshot changes its four-pair/eight-output census")
        if not self.prefix_artifacts or (
            self.predicted_interface is not None
            and (
                self.boundary != "composed"
                or len(self.predicted_interface) != 24
                or any(type(v) is not D or not v.is_finite() for v in self.predicted_interface)
            )
        ):
            raise ValueError("Finite response-law table loses causal input custody or changes forecast coordinates")
        if (
            self.prefix_source.object_schema
            != (FiniteResponseLawAssignedEvaluationTaskResult if assigned else FiniteResponseLawEvaluationTaskResult).SCHEMA
            or self.prefix_source.object_id
            != f"finite-response-law.{self.root_id}.prefix.native.result"
        ):
            raise ValueError("Finite response-law evaluation forecasts require this root's actual prefix result")
        keys = set()
        for request, result in zip(self.requests, self.results, strict=True):
            if (
                request.action_word is None
                or result.request != ObjectIdentity.from_record(request.request_id, request)
                or result.action_word
                != ObjectIdentity.from_record(request.action_word.word_id, request.action_word)
            ):
                raise ValueError("Finite response-law table substitutes its native request or result")
            keys.add((request.action_word.word_id, request.qualification_view_id))
        if len(keys) != 16 or len({k[0] for k in keys}) != 8 or len({k[1] for k in keys}) != 2:
            raise ValueError("Finite response-law table loses exact full-menu/view coverage")


def predict_control_table(
    *,
    context: FiniteResponseLawControlLawContext,
    report: FiniteResponseLawQualificationReport,
    boundary: str,
    root_id: str,
    prefix: tuple[D, ...] | None,
    prefix_source: ObjectIdentity,
    prefix_artifacts: tuple[ArtifactIdentity, ...],
    parent_index: int,
    payload_reader: CandidatePayloadReader,
) -> FiniteResponseLawControlPredictionTable:
    # Frozen design: one primary-input table tested against both views/futures.
    # Refined observations assess numerical agreement; they cannot change the
    # primary prediction, support, scale or decision.
    if prefix is not None and (
        len(prefix) != 24 or any(type(v) is not D or not v.is_finite() for v in prefix)
    ):
        raise ValueError("Finite response-law pre-parent prediction requires exactly 24 causal finite features")
    if type(parent_index) is not int or parent_index not in range(5):
        raise ValueError("Finite response-law prediction requires its independently assigned old parent")
    index = tuple(b.boundary for b in report.calibration.boundaries).index(boundary)
    if (
        report.qualifications[index] != context.qualification
        or report.publications[index] != context.publication
    ):
        raise ValueError("Finite response-law prediction substitutes its frozen qualified boundary")
    law = context.qualification.response_law
    assert law is not None
    registration = law_registration(boundary, context.publication.implementation)
    evaluator = FiniteResponseLawLawEvaluator(
        registration,
        boundary,
        payload_reader if boundary == "composed" else None,
        report.publications[0] if boundary == "composed" else None,
    )
    service = LawEvaluationService(
        payload_reader, LawEvaluatorRegistry((cast(LawEvaluatorImplementation, evaluator),))
    )
    values = [
        LawEvaluationValue(
            q.quantity_id, q.quantity_id, v, q.native_unit, q.coordinate_frame, q.clock_id
        )
        for q, v in zip(feature_quantities(boundary), prefix or ())
    ]
    q = parent_quantity()
    values.append(
        LawEvaluationValue(
            q.quantity_id,
            q.quantity_id,
            D(parent_index),
            q.native_unit,
            q.coordinate_frame,
            q.clock_id,
        )
    )
    calibration = report.calibration.boundaries[index]
    inputs = tuple(
        sorted(
            (calibration.development_manifest, calibration.frozen_coefficients),
            key=lambda a: a.artifact_id,
        )
    )
    requests = []
    results = []
    for coordinate in context.plan.coordinates:
        word = context.plan.action_fibre(coordinate.action_fibre).action_word
        for view in coordinate.qualification_view_ids:
            request = LawEvaluationRequest(
                f"{root_id}.{boundary}.{coordinate.coordinate_id.rsplit('.', 2)[-2]}.{view}",
                coordinate.response_law,
                coordinate.qualification_result,
                ObjectIdentity.from_record(context.axes.axis_map_id, context.axes),
                registration.implementation,
                ObjectIdentity.from_record(context.publication.receipt_id, context.publication),
                LawEvaluationKind.POINT_NAMED_VALUES,
                LawEvaluationMode.OFFLINE,
                registration.offline_resource_envelope_id,
                None,
                law.system_id,
                law.world_id,
                law.relation.relation_id,
                law.chart_id,
                word.denominator_id,
                coordinate.denominator_member_id,
                coordinate.candidate_version_id,
                view,
                law.relation.history_quantity_ids,
                law.relation.receiver_quantity_ids,
                ObjectIdentity.from_record(HORIZON.horizon_id, HORIZON),
                context.plan.support_cells[0].denominator_cell_id,
                word,
                tuple(sorted(values, key=lambda v: v.value_id)),
                inputs,
                PRODUCTS,
            )
            requests.append(request)
            results.append(
                service.evaluate(
                    context.system,
                    law,
                    context.qualification,
                    context.axes,
                    request,
                    context.publication,
                )
            )
    forecast, point_mean = None, None
    if boundary == "cached" or prefix is not None:
        raw = payload_reader.read_candidate_payload(context.publication)
        if sha256(raw).hexdigest() != context.publication.content_sha256:
            raise ValueError("Finite response-law U forecast substitutes qualified payload bytes")
        payload = evaluator.decode(raw, maximum_bytes=MAXIMUM_LAW_BYTES)
        if not isinstance(payload, FiniteResponseLawConditionalPayload):
            raise ValueError("Finite response-law U forecast requires its frozen composition")
        x = (
            np.zeros((1, 24), dtype=np.float64)
            if prefix is None
            else np.asarray([[float(v) for v in prefix]], dtype=np.float64)
        )
        parents = np.asarray([parent_index], dtype=np.int64)
        lower = evaluator._lower(payload) if boundary == "composed" else None
        prediction = payload.predict(x, parents, lower=lower)
        if np.isfinite(prediction.mean).all():
            point_mean = tuple(D(format(float(v), ".17g")) for v in prediction.mean.reshape(-1))
        if boundary == "composed":
            values_u = payload.forecast_handoff(x, parents)[0]
            if np.isfinite(values_u).all():
                forecast = tuple(D(format(float(v), ".17g")) for v in values_u)
    return FiniteResponseLawControlPredictionTable(
        f"{root_id}.{boundary}.table",
        root_id,
        boundary,
        ObjectIdentity.from_record(report.report_id, report),
        prefix_source,
        prefix_artifacts,
        forecast,
        tuple(requests),
        tuple(results),
        point_mean,
    )
