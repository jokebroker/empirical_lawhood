"""Evaluate qualified finite maps on authenticated permitted causal prefixes."""

from dataclasses import dataclass
from decimal import Decimal as D
from functools import cached_property

from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationDisposition as Disposition,
    LawEvaluationKind,
    LawEvaluationRequest,
    LawEvaluationResult,
    LawEvaluationValue,
    LawEvaluatorRegistration,
    _refusal,
)
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.causal import causal_operands
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.words import project_pulse
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import BATCH_SOURCE_SHA256, ReactorBatchSource
from empirical_lawhood.kernel.laws import LawRepresentationKind, ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec
from .config import CONTEXTS, FIRST, PREFIX
from .law_payload import ClassicalLawDecoder, ClassicalLawPayload
from .measurement import decimal, receiver_ids
from .prediction import HORIZON, ClassicalPredictedBound, causal_point, predict
from .records import CAUSAL_FIELDS, ClassicalContext
from .science import METHOD, staged_pulse_reactor_system


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


def first_context(causal: ClassicalContext) -> ClassicalContext:
    if causal.context != "induced":
        return causal
    k = causal.first_callback
    assert k is not None
    arrays = causal.arrays.unpack()
    return ClassicalContext(
        causal.root,
        "early",
        k,
        LocalArrayPayload.pack(
            {name: arrays[name][: k + int(name == "observations")] for name in CAUSAL_FIELDS}
        ),
        (),
    )


def input_values(
    causal: ClassicalContext, payload: ClassicalLawPayload
) -> tuple[LawEvaluationValue, ...]:
    operands = causal_operands(causal, payload.prepared_domain)
    if operands is None:
        return ()
    original = first_context(causal)
    original_data = original.arrays.unpack()["observations"][-1]
    o, p = operands.observation, operands.previous
    values = {
        f"domain-feature-{i:02d}": (D(repr(v)), "1") for i, v in enumerate(operands.domain_features)
    }
    values.update(
        {
            "callback-time": (decimal(o.time), "s"),
            "observed-dose": (decimal(o.dose), "kg"),
            "observed-temperature": (decimal(o.temperature), "K"),
            "observed-jacket": (decimal(o.jacket), "K"),
            "previous-applied-feed": (decimal(p[0]), "kg/s"),
            "previous-applied-jacket": (decimal(p[1]), "K"),
            "classical-context": (D((*CONTEXTS, "induced").index(causal.context)), "1"),
            "initial-temperature": (decimal(original_data[1]), "K"),
            "first-callback-time": (decimal(original_data[0]), "s"),
        }
    )
    return tuple(
        LawEvaluationValue(k, k, value, unit, "reactor-native", "reactor-clock")
        for k, (value, unit) in sorted(values.items())
    )


@dataclass(frozen=True)
class ClassicalLawEvaluator:
    registration: LawEvaluatorRegistration
    source: ReactorBatchSource
    causal: ClassicalContext
    payload: ClassicalLawPayload

    def __post_init__(self) -> None:
        if (
            self.registration != registration(self.registration.implementation)
            or self.source.fingerprint() != BATCH_SOURCE_SHA256
        ):
            raise ValueError(
                "law evaluator substitutes its installed implementation or native source"
            )

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return ClassicalLawDecoder().decode(payload, maximum_bytes=maximum_bytes)

    @cached_property
    def prediction(self) -> ClassicalPredictedBound:
        payload, ca = self.payload, self.causal
        point = causal_point(
            self.source,
            ca,
            payload.prepared_domain,
            block=payload.block,
            mechanistic=payload.recipe.arm == "MECH_SAFE",
        )
        original = first_context(ca)
        prior = None
        mass = D(0)
        if ca.context == "induced":
            prior = causal_point(
                self.source, original, payload.prepared_domain, block="staged-sequence-comparison", mechanistic=False
            )
            mass = next((m for w, m, _ in prior.masses if w == FIRST), D(0))
        return predict(payload.recipe, point, first_point=prior, first_mass=mass)

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        def refuse(
            reason: str, disposition: Disposition = Disposition.OPERAND_ABSENT
        ) -> LawEvaluationResult:
            return _refusal(request, self.registration.implementation, disposition, reason)

        if payload != self.payload:
            return refuse("staged-pulse-law-payload-substitution")
        c, ca, word = self.payload.recipe.bound.coordinate, self.causal, request.action_word
        if (
            system != staged_pulse_reactor_system(c)
            or law.relation != system.relation
            or word is None
            or request.requested_product_ids != ("response-values", "uncertainty-values")
            or request.receiver_quantity_ids != tuple(sorted(receiver_ids(c)))
            or request.horizon != ObjectIdentity.from_record(HORIZON, law.relation.horizon)
        ):
            return refuse("staged-pulse-law-relation-word-receiver-or-horizon-differs")
        if ca.context != c.context or (
            c.kind == "joint" and (ca.predecessor is None or ca.first_callback is None)
        ):
            return refuse(
                "staged-pulse-law-missing-actual-induced-owner-history", Disposition.OUTSIDE_SUPPORT
            )
        if not any(
            a.sha256 == ca.fingerprint() and a.payload_schema == ca.SCHEMA
            for a in request.input_artifacts
        ) or request.input_values != input_values(ca, self.payload):
            return refuse("staged-pulse-law-causal-operand-or-prefix-substitution")
        inputs = causal_operands(ca, self.payload.prepared_domain)
        if inputs is None:
            return refuse("staged-pulse-law-outside-constructed-support", Disposition.OUTSIDE_SUPPORT)
        expected = project_pulse(
            inputs.observation,
            inputs.previous,
            c.pulse,
            decision_id=word.word_id.removesuffix(".word"),
            history_id=ca.record_id,
            horizon_id=HORIZON,
            guard_s=240 if c.kind == "baseline" else 120,
            receiver_id="c2"
            if c.kind == "joint"
            else "c1"
            if c.kind in ("first", "baseline")
            else "c",
        )
        if word != expected.word:
            return refuse("staged-pulse-law-another-native-word-or-history", Disposition.OUTSIDE_SUPPORT)
        predicted = self.prediction
        if not predicted.available:
            return refuse("staged-pulse-law-unavailable-frozen-bound")

        def value(kind: str, key: str, number: D) -> LawEvaluationValue:
            return LawEvaluationValue(
                f"{kind}.{key}", key, number, "K", "reactor-native", "reactor-clock"
            )

        intervals = tuple(
            (lo.value_id, lo.value, hi.value)
            for lo, hi in zip(predicted.lower, predicted.upper, strict=True)
        )
        response = tuple(
            sorted(
                (value("interval-centre", key, (lo + hi) / 2) for key, lo, hi in intervals),
                key=lambda v: v.value_id,
            )
        )
        uncertainty = tuple(
            sorted(
                (
                    value(kind, key, number)
                    for key, lo, hi in intervals
                    for kind, number in (("lower", lo), ("upper", hi), ("halfwidth", (hi - lo) / 2))
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
