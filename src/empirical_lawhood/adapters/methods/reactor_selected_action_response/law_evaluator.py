"""Evaluate the single qualified finite interval; never estimate hidden kappa."""

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
from empirical_lawhood.kernel.laws import LawRepresentationKind, ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec
from .config import PREFIX
from .law_payload import ClassicalLawDecoder, ClassicalLawPayload
from .science import HORIZON, METHOD, RECEIVERS, classical_system


def registration(implementation: ObjectIdentity) -> LawEvaluatorRegistration:
    decoder = ClassicalLawDecoder()
    return LawEvaluatorRegistration(
        f"{PREFIX}.evaluator",
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


@dataclass(frozen=True, slots=True)
class ClassicalLawEvaluator:
    registration: LawEvaluatorRegistration

    def __post_init__(self) -> None:
        if self.registration != registration(self.registration.implementation):
            raise ValueError("classical evaluator registration differs")

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return ClassicalLawDecoder().decode(payload, maximum_bytes=maximum_bytes)

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, ClassicalLawPayload):
            raise TypeError("classical law received another payload")

        def refuse(
            reason: str,
            disposition: LawEvaluationDisposition = LawEvaluationDisposition.OPERAND_ABSENT,
        ) -> LawEvaluationResult:
            return _refusal(request, self.registration.implementation, disposition, reason)

        if system != classical_system() or law.relation != system.relation:
            return refuse("classical-law-relation-differs")
        if (
            request.requested_product_ids != ("response-values", "uncertainty-values")
            or request.action_word is None
            or request.horizon != ObjectIdentity.from_record(HORIZON, law.relation.horizon)
            or request.receiver_quantity_ids != tuple(sorted(RECEIVERS))
        ):
            return refuse("classical-law-action-receiver-horizon-products-differ")
        values = {value.quantity_id: value for value in request.input_values}
        scalar_units = {
            "callback-time": "s",
            "observed-dose": "kg",
            "observed-temperature": "K",
            "observed-jacket": "K",
            "previous-applied-feed": "kg/s",
            "previous-applied-jacket": "K",
        }
        expected = set(scalar_units) | {f"domain-feature-{i:02d}" for i in range(23)}
        if set(values) != expected or len(values) != len(request.input_values):
            return refuse("classical-law-causal-input-census")
        if any(
            v.native_unit != scalar_units.get(key, "1")
            or v.native_frame_id != "reactor-native"
            or v.clock_id != "reactor-clock"
            for key, v in values.items()
        ):
            return refuse("classical-law-native-units-frame-clock")
        raw = {key: float(value.value) for key, value in values.items()}
        feature = np.asarray([raw[f"domain-feature-{i:02d}"] for i in range(23)])
        u = (raw["observed-temperature"] - 318.4) / 40
        v = (raw["observed-jacket"] - 316) / 40
        j = (raw["previous-applied-jacket"] - 316) / 40
        coherent = {
            0: raw["callback-time"] / 28800,
            1: raw["observed-dose"] / 287.3,
            2: float(600 <= raw["callback-time"] < 25200),
            4: j,
            5: 0.5,
            6: j,
            7: u,
            8: v,
            9: u * u,
            10: u * v,
            11: u * 0.5,
            12: u * j,
            21: feature[13] * 0.5,
            22: feature[13] * j,
        }
        if (
            not all(np.isfinite(v) for v in raw.values())
            or any(abs(feature[i] - value) > 1e-12 for i, value in coherent.items())
            or raw["previous-applied-feed"] < 0
        ):
            return refuse("classical-law-inconsistent-causal-coordinates")
        if (
            not 600 <= raw["callback-time"] <= 6000
            or raw["previous-applied-feed"] > 0.016
            or not payload.prepared_domain.proposed_support(
                feature[None, :], np.asarray((4,))
            ).all()
        ):
            return refuse(
                "classical-law-outside-causal-domain", LawEvaluationDisposition.OUTSIDE_SUPPORT
            )
        try:
            observation = Observation(
                raw["callback-time"],
                raw["observed-temperature"],
                raw["observed-jacket"],
                raw["observed-dose"],
            )
            previous = raw["previous-applied-feed"], raw["previous-applied-jacket"]
            projection = Actuator().project_request(observation, previous, (0.016, previous[1]), 4)
            mapping = ReactorFeedOnlyWordMap.from_projection(
                projection,
                decision_id=request.action_word.word_id.removesuffix(".feed.word"),
                history_id=request.action_word.retained_history_id,
            )
        except (ValueError, IndexError):
            return refuse("classical-law-native-projection-invalid")
        if (
            mapping.scalar_word != request.action_word
            or abs(sum(dt * feed for _, dt, feed, _ in mapping.full_native.exposure) - D(".16"))
            > 1e-12
        ):
            return refuse(
                "classical-law-word-outside-selected-support",
                LawEvaluationDisposition.OUTSIDE_SUPPORT,
            )
        design = payload.design
        centre = (design.response_lower_K + design.response_upper_K) / 2
        width = (design.response_upper_K - design.response_lower_K) / 2

        def output(kind: str, receiver: int, value: D) -> LawEvaluationValue:
            return LawEvaluationValue(
                f"{kind}.{RECEIVERS[receiver]}",
                RECEIVERS[receiver],
                value,
                "K",
                "reactor-native",
                "reactor-clock",
            )

        return LawEvaluationResult(
            f"evaluation.{request.request_id}",
            ObjectIdentity.from_record(request.request_id, request),
            self.registration.implementation,
            request.denominator_member_id,
            request.candidate_version_id,
            request.qualification_view_id,
            ObjectIdentity.from_record(mapping.scalar_word.word_id, mapping.scalar_word),
            LawEvaluationDisposition.SUPPORTED,
            tuple(
                sorted(
                    (
                        output("interval-centre", 0, values["observed-temperature"].value),
                        output("interval-centre", 1, centre),
                    ),
                    key=lambda v: v.value_id,
                )
            ),
            (),
            (),
            tuple(
                sorted(
                    (
                        output("halfwidth", 0, design.causal_temperature_halfwidth_K),
                        output("halfwidth", 1, width),
                        output("numerical-allowance", 0, design.numerical_peak_tolerance_K),
                        output("numerical-allowance", 1, design.numerical_cooling_tolerance_K),
                    ),
                    key=lambda v: v.value_id,
                )
            ),
            (),
            (),
        )
