"""Executable selected joint response law on the scalar native feed chart."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D

import numpy as np

from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationDisposition,
    LawEvaluationKind,
    LawEvaluationRequest,
    LawEvaluationResult,
    LawEvaluationValue,
    LawEvaluatorRegistration,
    _refusal,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
from empirical_lawhood.adapters.simulators.reactor_regime_response.feed_word import ReactorFeedOnlyWordMap
from empirical_lawhood.kernel.action_contracts import ActionWordSupportStatus
from empirical_lawhood.kernel.laws import LawRepresentationKind, ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec

from .config import PREFIX
from .law_payload import RegimeJointLawDecoder, RegimeJointLawPayload
from .science import HORIZON, METHOD, RECEIVERS, regime_system

CLOCK = "reactor-clock"
FEEDS = (0.0, 0.016, 0.032)
PRODUCTS = ("response-values", "uncertainty-values")


def registration(implementation: ObjectIdentity) -> LawEvaluatorRegistration:
    decoder = RegimeJointLawDecoder()
    return LawEvaluatorRegistration(
        f"{PREFIX}.joint-evaluator-registration",
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
        4 * 1024**2,
    )


def _number(value: float) -> D:
    return D(repr(float(value)))


@dataclass(frozen=True, slots=True)
class RegimeJointLawEvaluator:
    registration: LawEvaluatorRegistration

    def __post_init__(self) -> None:
        if self.registration != registration(self.registration.implementation):
            raise ValueError("reactor joint-law evaluator registration differs")

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return RegimeJointLawDecoder().decode(payload, maximum_bytes=maximum_bytes)

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, RegimeJointLawPayload):
            raise TypeError("reactor joint evaluator received another payload")

        def refuse(
            reason: str,
            disposition: LawEvaluationDisposition = LawEvaluationDisposition.OPERAND_ABSENT,
        ) -> LawEvaluationResult:
            return _refusal(request, self.registration.implementation, disposition, reason)

        if system != regime_system() or law.relation != system.relation:
            return refuse("joint-law-relation-differs")
        if (
            payload.route is None or payload.coefficient is None or payload.absolute is None
            or payload.q_temperature is None or payload.q_cooling is None
        ):
            return refuse("joint-law-frozen-model-or-calibration-unavailable")
        if request.requested_product_ids != PRODUCTS:
            return refuse("joint-law-products-differ", LawEvaluationDisposition.PRODUCT_NOT_CLAIMED)
        if request.action_word is None:
            return refuse("joint-law-native-word-absent")
        if (
            request.horizon != ObjectIdentity.from_record(HORIZON, law.relation.horizon)
            or request.receiver_quantity_ids != tuple(sorted(RECEIVERS))
        ):
            return refuse("joint-law-receiver-horizon-differs")
        values = {item.quantity_id: item for item in request.input_values}
        expected = {f"feature-{index:02d}" for index in range(18)} | {
            "callback-time", "observed-dose", "previous-applied-feed",
            "previous-applied-jacket", "reactor-fixed-jacket",
            "reactor-regime-preparation-policy-version",
        }
        if set(values) != expected or len(values) != len(request.input_values):
            return refuse("joint-law-causal-input-census")
        units = {
            "callback-time": "s", "observed-dose": "kg",
            "previous-applied-feed": "kg/s", "previous-applied-jacket": "K",
            "reactor-fixed-jacket": "K",
        }
        for key, value in values.items():
            if (
                value.native_unit != units.get(key, "1")
                or value.native_frame_id != "reactor-native"
                or value.clock_id != CLOCK
            ):
                return refuse("joint-law-native-units-frame-clock-differ")
        feature = np.asarray(
            [float(values[f"feature-{index:02d}"].value) for index in range(18)],
            dtype=np.float64,
        )
        time = float(values["callback-time"].value)
        dose = float(values["observed-dose"].value)
        previous_feed = float(values["previous-applied-feed"].value)
        previous_jacket = float(values["previous-applied-jacket"].value)
        fixed_jacket = float(values["reactor-fixed-jacket"].value)
        route_code = float(values["reactor-regime-preparation-policy-version"].value)
        if (
            not np.isfinite(feature).all()
            or time % 10
            or not 0 < time < 28800
            or abs(feature[0] - time / 28800) > 1e-9
            or abs(feature[1] - dose / 287.3) > 1e-9
            or abs(feature[4] - previous_feed / 0.032) > 1e-9
            or abs(feature[5] - (previous_jacket - 316) / 40) > 1e-9
            or previous_jacket != fixed_jacket
            or route_code != {"prepared_t0": 0, "c_q": 1, "p_q": 2}[payload.route]
        ):
            return refuse("joint-law-causal-clock-dose-or-fixed-jacket-differs")
        feed = float(request.action_word.occurrences[0].requested.value)
        if feed not in FEEDS:
            return refuse("joint-law-feed-word-outside-chart", LawEvaluationDisposition.OUTSIDE_SUPPORT)
        word = FEEDS.index(feed)
        observed = Observation(
            time, 318.4 + 40 * feature[2], 316 + 40 * feature[3], dose
        )
        try:
            projection = Actuator().project_request(
                observed, (previous_feed, fixed_jacket), (feed, fixed_jacket), 1 + 3 * word
            )
            mapping = ReactorFeedOnlyWordMap.from_projection(
                projection,
                decision_id=request.action_word.word_id.removesuffix(".feed.word"),
                history_id=request.action_word.retained_history_id,
            )
        except (ValueError, IndexError):
            return refuse("joint-law-native-projection-invalid")
        if mapping.scalar_word != request.action_word:
            return refuse("joint-law-actual-scalar-word-differs")
        x_coefficient = feature[:len(payload.coefficient.mean)][None, :]
        x_absolute = feature[:len(payload.absolute.mean)][None, :]
        if (
            not payload.coefficient.contains(x_coefficient)[0]
            or not payload.absolute.contains(x_absolute)[0]
            or mapping.full_native.word.support_status is not ActionWordSupportStatus.SUPPORTED
        ):
            return refuse("joint-law-outside-frozen-support", LawEvaluationDisposition.OUTSIDE_SUPPORT)
        coefficient = float(payload.coefficient.to_fit().predict(x_coefficient)[0])
        baseline = float(payload.absolute.to_fit().predict(x_absolute)[0])
        mass = float(sum(dt * realized for _, dt, realized, _ in mapping.full_native.exposure))
        cooling = coefficient * mass
        temperature = baseline - cooling
        if (
            not np.isfinite((coefficient, baseline, mass, cooling, temperature)).all()
            or abs(cooling) > 0.01
            or payload.q_temperature > 1
            or payload.q_cooling > 1
        ):
            return refuse("joint-law-precision-or-native-chart-failed")
        widths = (
            _number(float(payload.q_temperature) * 0.25 + 0.01),
            _number(float(payload.q_cooling) * (0.00005 + 0.5 * abs(cooling)) + 0.000001),
        )

        def output(kind: str, receiver: int, number: D) -> LawEvaluationValue:
            return LawEvaluationValue(
                f"{kind}.{RECEIVERS[receiver]}", RECEIVERS[receiver], number,
                "K", "reactor-native", CLOCK,
            )

        return LawEvaluationResult(
            f"evaluation.{request.request_id}",
            ObjectIdentity.from_record(request.request_id, request),
            self.registration.implementation,
            request.denominator_member_id,
            request.candidate_version_id,
            request.qualification_view_id,
            ObjectIdentity.from_record(request.action_word.word_id, request.action_word),
            LawEvaluationDisposition.SUPPORTED,
            tuple(sorted((
                output("mean", 0, _number(temperature)),
                output("mean", 1, _number(cooling)),
            ), key=lambda value: value.value_id)),
            (), (),
            tuple(sorted((
                output("halfwidth", 0, widths[0]),
                output("halfwidth", 1, widths[1]),
                output("numerical-allowance", 0, D(".01")),
                output("numerical-allowance", 1, D(".000001")),
            ), key=lambda value: value.value_id)),
            (), (),
        )
