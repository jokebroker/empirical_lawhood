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
from .payload import FeedPayload, FeedDecoder
from .science import PREFIX, CLOCK, RECEIVERS, UNITS, METHOD


def registration(implementation: ObjectIdentity) -> LawEvaluatorRegistration:
    decoder = FeedDecoder()
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
class FeedEvaluator:
    registration: LawEvaluatorRegistration

    def __post_init__(self) -> None:
        if self.registration != registration(self.registration.implementation):
            raise ValueError("local evaluator registration differs")

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return FeedDecoder().decode(payload, maximum_bytes=maximum_bytes)

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, FeedPayload):
            raise TypeError("local evaluator received another payload")

        def refuse(
            reason: str,
            disposition: LawEvaluationDisposition = LawEvaluationDisposition.OPERAND_ABSENT,
        ) -> LawEvaluationResult:
            return _refusal(request, self.registration.implementation, disposition, reason)

        if request.requested_product_ids != ("response-values", "uncertainty-values"):
            return refuse("local-products-differ", LawEvaluationDisposition.PRODUCT_NOT_CLAIMED)
        if any(v is None for v in (*payload.halfwidths, *payload.coefficients)):
            return refuse("local-calibration-unavailable")
        values = {v.quantity_id: v for v in request.input_values}
        expected = {f"feature-{i:02d}" for i in range(23)} | {"callback-time", "action-id"}
        native_units = {
            "previous-applied-feed": "kg/s",
            "previous-applied-jacket": "K",
            "observed-dose": "kg",
        }
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
            or law.relation.receiver_quantity_ids != tuple(sorted(RECEIVERS))
        ):
            return refuse("local-receiver-horizon-differs")
        time, action = float(values["callback-time"].value), float(values["action-id"].value)
        x = np.array([float(values[f"feature-{i:02d}"].value) for i in range(23)])
        if (
            not np.isfinite(x).all()
            or time % 10
            or not 600 <= time <= 6000
            or x[0] != time / 28800
            or action not in (1, 4, 7)
        ):
            return refuse("local-causal-cutoff-or-action-differs")
        from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
        from empirical_lawhood.adapters.simulators.reactor_causal_response.delivery import ReactorExposureWordMap
        from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation

        previous = (
            float(values["previous-applied-feed"].value),
            float(values["previous-applied-jacket"].value),
        )
        dose = float(values["observed-dose"].value)
        projections = Actuator().project(Observation(time, 0, 0, dose), previous)
        projection = projections[int(action)]
        if (
            previous[0] > 0.016
            or projections[1].mean_feed != 0
            or len({projections[a].delivery_key for a in (1, 4, 7)}) != 3
            or projection.applied[1] != previous[1]
            or x[1] != dose / 287.3
            or x[5] != projection.mean_feed / 0.032
            or x[6] != (projection.applied[1] - 316) / 40
        ):
            return refuse("feed-native-delivery-or-dose-differs")
        if request.action_word is not None:
            mapping = ReactorExposureWordMap.from_projection(
                projection,
                decision_id=request.action_word.word_id.removesuffix(".word"),
                history_id=request.action_word.retained_history_id,
                horizon_id=request.horizon.object_id,
            )
            if mapping.word != request.action_word:
                return refuse("feed-native-word-differs")
        if not payload.domain.proposed_support(x[None], np.array([int(action)]))[0]:
            return refuse(
                "local-outside-frozen-domain-support", LawEvaluationDisposition.OUTSIDE_SUPPORT
            )
        point = payload.domain.predict(x[None])[0]
        if not np.isfinite(point).all():
            return refuse("feed-nonfinite-prediction")
        absolute_width, maximum_response_width = payload.halfwidths
        response_coefficient = payload.coefficients[1]
        assert (
            absolute_width is not None
            and maximum_response_width is not None
            and response_coefficient is not None
        )
        widths = (
            absolute_width,
            response_coefficient * (D(".00005") + D(".5") * abs(D(repr(float(point[1])))))
            + D(".000001"),
        )
        if widths[1] > maximum_response_width:
            return refuse("feed-response-envelope-exceeds-qualified-cap")

        def output_value(role: str, receiver: int, number: D) -> LawEvaluationValue:
            return LawEvaluationValue(
                f"{role}.{RECEIVERS[receiver]}",
                RECEIVERS[receiver],
                number,
                UNITS[receiver],
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
            tuple(
                sorted(
                    (output_value("mean", r, D(repr(float(point[r])))) for r in (0, 1)),
                    key=lambda v: v.value_id,
                )
            ),
            (),
            (),
            tuple(
                sorted(
                    (
                        *(output_value("halfwidth", r, widths[r]) for r in (0, 1)),
                        *(
                            output_value("numerical-allowance", r, (D(".01"), D(".000001"))[r])
                            for r in (0, 1)
                        ),
                    ),
                    key=lambda v: v.value_id,
                )
            ),
            (),
            (),
        )
