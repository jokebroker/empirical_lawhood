"""Support-limited evaluation of one qualified local receiver law."""

from dataclasses import dataclass
from decimal import Decimal as D
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
from .payload import LocalPayload, LocalDecoder
from .science import PREFIX, CLOCK, RECEIVERS, UNITS, METHOD


def registration(implementation: ObjectIdentity) -> LawEvaluatorRegistration:
    decoder = LocalDecoder()
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
        65536,
    )


@dataclass(frozen=True, slots=True)
class LocalEvaluator:
    registration: LawEvaluatorRegistration

    def __post_init__(self) -> None:
        if self.registration != registration(self.registration.implementation):
            raise ValueError("local evaluator registration differs")

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return LocalDecoder().decode(payload, maximum_bytes=maximum_bytes)

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, LocalPayload):
            raise TypeError("local evaluator received another payload")

        def refuse(
            reason: str,
            disposition: LawEvaluationDisposition = LawEvaluationDisposition.OPERAND_ABSENT,
        ) -> LawEvaluationResult:
            return _refusal(request, self.registration.implementation, disposition, reason)

        if request.requested_product_ids != ("response-values", "uncertainty-values"):
            return refuse("local-products-differ", LawEvaluationDisposition.PRODUCT_NOT_CLAIMED)
        if payload.halfwidth is None:
            return refuse("local-calibration-unavailable")
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
            return refuse("local-causal-input-census")
        for key, value in values.items():
            if (
                value.native_unit != ("s" if key == "callback-time" else native_units.get(key, "1"))
                or value.native_frame_id != "reactor-native"
                or value.clock_id != CLOCK
            ):
                return refuse("local-native-units-frame-clock-differ")
        if (
            law.relation.horizon.duration != 10
            or request.horizon
            != ObjectIdentity.from_record(law.relation.horizon.horizon_id, law.relation.horizon)
            or law.relation.receiver_quantity_ids != (RECEIVERS[payload.receiver],)
        ):
            return refuse("local-receiver-horizon-differs")
        time, action = float(values["callback-time"].value), float(values["action-id"].value)
        x = np.array([float(values[f"feature-{i:02d}"].value) for i in range(23)])
        if (
            not np.isfinite(x).all()
            or time % 10
            or not 0 <= time < 28800
            or x[0] != time / 28800
            or action not in range(9)
        ):
            return refuse("local-causal-cutoff-or-action-differs")
        if request.action_word is not None:
            from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
            from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import ReactorExposureWordMap
            from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation

            previous = (
                float(values["previous-applied-feed"].value),
                float(values["previous-applied-jacket"].value),
            )
            dose = float(values["observed-dose"].value)
            projection = Actuator().project(Observation(time, 0, 0, dose), previous)[int(action)]
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
                or x[6] != (projection.applied[1] - 316) / 40
            ):
                return refuse("local-native-delivery-or-dose-differs")
        if not payload.domain.proposed_support(x[None], np.array([int(action)]))[0]:
            return refuse(
                "local-outside-frozen-domain-support", LawEvaluationDisposition.OUTSIDE_SUPPORT
            )
        point = payload.domain.predict(x[None])[0, payload.receiver]
        if not np.isfinite(point):
            return refuse("local-nonfinite-prediction")

        def output_value(role: str, number: D) -> LawEvaluationValue:
            return LawEvaluationValue(
                f"{role}.{RECEIVERS[payload.receiver]}",
                RECEIVERS[payload.receiver],
                number,
                UNITS[payload.receiver],
                "reactor-native",
                CLOCK,
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
            (output_value("mean", D(repr(float(point)))),),
            (),
            (),
            (
                output_value("halfwidth", payload.halfwidth),
                output_value("numerical-allowance", (D(".01"), D(".0002"))[payload.receiver]),
            ),
            (),
            (),
        )
