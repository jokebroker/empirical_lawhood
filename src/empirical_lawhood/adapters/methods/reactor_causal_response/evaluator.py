"""Uncertainty-bearing empirical evaluator behind LawEvaluationService."""

from dataclasses import dataclass
from decimal import Decimal as D
import json
from functools import lru_cache
import numpy as np
from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluatorRegistration,
    LawEvaluationKind,
    LawEvaluationRequest,
    LawEvaluationResult,
    LawEvaluationDisposition,
    LawEvaluationValue,
    _refusal,
)
from empirical_lawhood.kernel.laws import LawRepresentationKind, ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec
from .payload import DevelopmentBoundReactorResponsePayload, DevelopmentBoundReactorResponseDecoder
from .serialization import read_fit
from .numerical import FrozenFit
from .science import PREFIX, CLOCK, RECEIVERS, METHOD


@lru_cache(maxsize=16)
def _decoded_payload(payload: bytes, maximum_bytes: int) -> CanonicalRecord:
    return DevelopmentBoundReactorResponseDecoder().decode(payload, maximum_bytes=maximum_bytes)


@lru_cache(maxsize=16)
def _frozen_model(model_json: str) -> FrozenFit:
    return read_fit(json.loads(model_json))


def registration(implementation: ObjectIdentity) -> LawEvaluatorRegistration:
    decoder = DevelopmentBoundReactorResponseDecoder()
    return LawEvaluatorRegistration(
        f"{PREFIX}.evaluator-registration",
        METHOD,
        LawEvaluationKind.POINT_NAMED_VALUES,
        LawRepresentationKind.FINITE_ACTION_OPERATOR,
        decoder.payload_schema,
        decoder.extension_namespace,
        decoder.decoder_schema,
        decoder.decoder_version,
        LawEvaluationRequest.SCHEMA,
        LawEvaluationResult.SCHEMA,
        implementation,
        True,
        f"{PREFIX}.offline",
        f"{PREFIX}.online",
        2 * 1024**2,
    )


@dataclass(frozen=True, slots=True)
class EmpiricalEvaluator:
    registration: LawEvaluatorRegistration

    def __post_init__(self) -> None:
        if self.registration != registration(self.registration.implementation):
            raise ValueError("empirical evaluator registration differs")

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return _decoded_payload(payload, maximum_bytes)

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, DevelopmentBoundReactorResponsePayload):
            raise TypeError("empirical evaluator received another payload")

        def refuse(
            reason: str,
            disposition: LawEvaluationDisposition = LawEvaluationDisposition.OPERAND_ABSENT,
        ) -> LawEvaluationResult:
            return _refusal(request, self.registration.implementation, disposition, reason)

        if request.requested_product_ids != ("response-values", "uncertainty-values"):
            return refuse("empirical-products-differ", LawEvaluationDisposition.PRODUCT_NOT_CLAIMED)
        if payload.q is None:
            return refuse("empirical-calibration-unavailable")
        quantities = {q.quantity_id: q for q in system.quantities}
        values = {v.quantity_id: v for v in request.input_values}
        expected = {f"feature-{i:02d}" for i in range(23)} | {"callback-time", "action-id"}
        native_units = {
            "previous-applied-feed": "kg/s",
            "previous-applied-jacket": "K",
            "observed-dose": "kg",
        }
        if request.action_word is not None:
            expected.update(native_units)
        if set(values) != expected or len(values) != len(request.input_values):
            return refuse("empirical-causal-input-census")
        for key, v in values.items():
            if (
                key not in quantities
                or v.native_unit != ("s" if key == "callback-time" else native_units.get(key, "1"))
                or v.native_frame_id != "reactor-native"
                or v.clock_id != CLOCK
            ):
                return refuse("empirical-unit-frame-clock-differs")
        if (
            law.relation.horizon.duration != D(10)
            or request.horizon
            != ObjectIdentity.from_record(law.relation.horizon.horizon_id, law.relation.horizon)
            or law.relation.receiver_quantity_ids != tuple(sorted(RECEIVERS))
        ):
            return refuse("empirical-receiver-horizon-differs")
        time, action = float(values["callback-time"].value), float(values["action-id"].value)
        x = np.array([float(values[f"feature-{i:02d}"].value) for i in range(23)])
        if time % 10 or not 0 <= time < 28800 or x[0] != time / 28800 or action not in range(9):
            return refuse("empirical-cutoff-or-action-differs")
        if request.action_word is not None:
            from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
            from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import ReactorExposureWordMap
            from .numerical import Observation

            previous = (
                float(values["previous-applied-feed"].value),
                float(values["previous-applied-jacket"].value),
            )
            dose = float(values["observed-dose"].value)
            projection = Actuator().project(Observation(time, 0.0, 0.0, dose), previous)[
                int(action)
            ]
            mapping = ReactorExposureWordMap.from_projection(
                projection,
                decision_id=request.action_word.word_id.removesuffix(".word"),
                history_id=request.action_word.retained_history_id,
                horizon_id=request.horizon.object_id,
            )
            if (
                mapping.word != request.action_word
                or x[1] != dose / 287.3
                or x[5] != projection.mean_feed / 0.032
                or x[6] != (projection.applied[1] - 316.0) / 40.0
            ):
                return refuse("empirical-native-profile-or-causal-dose-differs")
        model = _frozen_model(payload.model_json)
        if not model.supported(x, time, int(action)):
            return refuse(
                "empirical-outside-training-support", LawEvaluationDisposition.OUTSIDE_SUPPORT
            )
        point = model.predict(x)
        width = (float(payload.q) * 0.25 + 0.01, float(payload.q) * 0.01 + 0.0002)

        def bound(
            role: str, numbers: tuple[float, ...] | np.ndarray
        ) -> tuple[LawEvaluationValue, ...]:
            return tuple(
                sorted(
                    (
                        LawEvaluationValue(
                            f"{role}.{key}",
                            key,
                            D(repr(float(value))),
                            unit,
                            "reactor-native",
                            CLOCK,
                        )
                        for key, value, unit in zip(RECEIVERS, numbers, ("K", "1"), strict=True)
                    ),
                    key=lambda v: v.value_id,
                )
            )

        return LawEvaluationResult(
            f"evaluation.{request.request_id}",
            ObjectIdentity.from_record(request.request_id, request),
            self.registration.implementation,
            request.denominator_member_id,
            request.candidate_version_id,
            request.qualification_view_id,
            None
            if request.action_word is None
            else ObjectIdentity.from_record(request.action_word.word_id, request.action_word),
            LawEvaluationDisposition.SUPPORTED,
            bound("mean", point),
            (),
            (),
            tuple(
                sorted(
                    (*bound("halfwidth", width), *bound("numerical-allowance", (0.01, 0.0002))),
                    key=lambda v: v.value_id,
                )
            ),
            (),
            (),
        )
