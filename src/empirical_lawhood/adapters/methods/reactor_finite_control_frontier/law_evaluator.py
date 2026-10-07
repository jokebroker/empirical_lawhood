"""Evaluate the exact finite pulse bound on permitted native coordinates."""

from dataclasses import dataclass
from decimal import Decimal as D

import numpy as np

from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationDisposition as Disposition,
    LawEvaluationKind,
    LawEvaluationRequest,
    LawEvaluationResult,
    LawEvaluationValue,
    LawEvaluatorRegistration,
    _refusal,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.words import project_pulse
from empirical_lawhood.kernel.laws import LawRepresentationKind, ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec
from .config import CONTEXTS, PREFIX
from .law_payload import FrontierLawDecoder, FrontierLawPayload
from .science import HORIZON, METHOD, RECEIVERS, frontier_system


def registration(implementation: ObjectIdentity) -> LawEvaluatorRegistration:
    decoder = FrontierLawDecoder()
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
class FrontierLawEvaluator:
    registration: LawEvaluatorRegistration

    def __post_init__(self) -> None:
        if self.registration != registration(self.registration.implementation):
            raise ValueError("frontier evaluator registration differs")

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return FrontierLawDecoder().decode(payload, maximum_bytes=maximum_bytes)

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, FrontierLawPayload):
            raise TypeError("frontier law requires its exact finite payload")

        def refuse(
            reason: str, disposition: Disposition = Disposition.OPERAND_ABSENT
        ) -> LawEvaluationResult:
            return _refusal(request, self.registration.implementation, disposition, reason)

        word, bound = request.action_word, payload.bound
        if (
            system != frontier_system()
            or law.relation != system.relation
            or word is None
            or request.requested_product_ids != ("response-values", "uncertainty-values")
            or request.horizon != ObjectIdentity.from_record(HORIZON, law.relation.horizon)
            or request.receiver_quantity_ids != tuple(sorted(RECEIVERS))
        ):
            return refuse("frontier-law-relation-word-receiver-or-window-differs")
        values = {v.quantity_id: v for v in request.input_values}
        units = {
            "callback-time": "s",
            "observed-dose": "kg",
            "observed-temperature": "K",
            "observed-jacket": "K",
            "previous-applied-feed": "kg/s",
            "previous-applied-jacket": "K",
            "frontier-context": "1",
        }
        if (
            set(values) != set(units) | {f"domain-feature-{i:02d}" for i in range(23)}
            or len(values) != len(request.input_values)
            or any(
                v.native_unit != units.get(key, "1")
                or v.native_frame_id != "reactor-native"
                or v.clock_id != "reactor-clock"
                for key, v in values.items()
            )
        ):
            return refuse("frontier-law-causal-operand-census")
        raw = {k: float(v.value) for k, v in values.items()}
        if not all(np.isfinite(v) for v in raw.values()):
            return refuse("frontier-law-nonfinite-causal-operands")
        c = bound.coordinate
        low, high = {
            "early": (600, 1200),
            "prepared_t0": (600, 6000),
            "middle": (7200, 9000),
            "late": (14400, 16200),
        }[c.context]
        if (
            not low <= raw["callback-time"] <= high
            or raw["frontier-context"] != CONTEXTS.index(c.context)
            or not 0 <= raw["previous-applied-feed"] <= 0.016
        ):
            return refuse("frontier-law-outside-causal-context", Disposition.OUTSIDE_SUPPORT)
        observation = Observation(
            raw["callback-time"],
            raw["observed-temperature"],
            raw["observed-jacket"],
            raw["observed-dose"],
        )
        previous = raw["previous-applied-feed"], raw["previous-applied-jacket"]
        projections = tuple(
            Actuator().project_request(observation, previous, (feed, previous[1]), i)
            for i, feed in enumerate((0.0, 0.016, 0.032))
        )
        if projections[0].mean_feed != 0 or len({p.delivery_key for p in projections}) != 3:
            return refuse("frontier-law-native-chart-collapsed", Disposition.OUTSIDE_SUPPORT)
        feature = np.asarray([raw[f"domain-feature-{i:02d}"] for i in range(23)])
        u, v, j, f = (
            (observation.temperature - 318.4) / 40,
            (observation.jacket - 316) / 40,
            (previous[1] - 316) / 40,
            projections[1].mean_feed / 0.032,
        )
        coherent = {
            0: observation.time / 28800,
            1: observation.dose / 287.3,
            2: float(600 <= observation.time < 25200),
            4: j,
            5: f,
            6: j,
            7: u,
            8: v,
            9: u * u,
            10: u * v,
            11: u * f,
            12: u * j,
            21: feature[13] * f,
            22: feature[13] * j,
        }
        if any(abs(feature[i] - value) > 1e-12 for i, value in coherent.items()):
            return refuse("frontier-law-inconsistent-causal-coordinates")
        if (
            c.context == "prepared_t0"
            and not payload.prepared_domain.proposed_support(
                feature[None, :], np.asarray((4,))
            ).all()
        ):
            return refuse(
                "frontier-law-outside-retained-prepared-domain", Disposition.OUTSIDE_SUPPORT
            )
        mapping = project_pulse(
            observation,
            previous,
            c.pulse,
            decision_id=word.word_id.removesuffix(".word"),
            history_id=word.retained_history_id,
            horizon_id=word.horizon_id,
        )
        if mapping.word != word:
            return refuse("frontier-law-another-native-pulse", Disposition.OUTSIDE_SUPPORT)
        if (
            not bound.nominated
            or bound.lower_K is None
            or bound.upper_K is None
            or bound.thermal_allowance_K is None
        ):
            return refuse("frontier-law-unavailable-bound")
        # Kelvin has a native zero floor. Q is an upper allowance, never a
        # fabricated symmetric lower-temperature guarantee around the sensor.
        intervals = (
            (D(0), values["observed-temperature"].value + bound.thermal_allowance_K),
            (bound.lower_K, bound.upper_K),
        )

        def output(kind: str, key: str, value: D) -> LawEvaluationValue:
            return LawEvaluationValue(
                f"{kind}.{key}", key, value, "K", "reactor-native", "reactor-clock"
            )

        response = tuple(
            sorted(
                (
                    output("interval-centre", key, (lo + hi) / 2)
                    for key, (lo, hi) in zip(RECEIVERS, intervals, strict=True)
                ),
                key=lambda v: v.value_id,
            )
        )
        uncertainty = tuple(
            sorted(
                (
                    output(kind, key, value)
                    for key, (lo, hi) in zip(RECEIVERS, intervals, strict=True)
                    for kind, value in (("lower", lo), ("upper", hi), ("halfwidth", (hi - lo) / 2))
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
            ObjectIdentity.from_record(word.word_id, word),
            Disposition.SUPPORTED,
            response,
            (),
            (),
            uncertainty,
            (),
            (),
        )
