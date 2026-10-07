"Frozen finite response-law prediction behind the shared qualified-law evaluation service."

from dataclasses import dataclass, replace
from decimal import Decimal as D
from hashlib import sha256

import numpy as np

from empirical_lawhood.kernel.action_contracts import ActionWordSupportStatus
from empirical_lawhood.kernel.laws import LawRepresentationKind, ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadReader,
    CandidatePayloadReadError,
)
from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationDisposition,
    LawEvaluationKind,
    LawEvaluationMode,
    LawEvaluationRequest,
    LawEvaluationResult,
    LawEvaluationValue,
    LawEvaluatorRegistration,
    _refusal,
)
from .control_word import controller_word_map
from .intervals import oriented_interval
from .law_binding import HORIZON, NATIVE_WORDS, PARENT_QUANTITY, feature_quantities, native_action_word, output_quantities, parent_quantity
from .law_payloads import FiniteResponseLawConditionalDecoder, FiniteResponseLawConditionalPayload, FiniteResponseLawLowerDecoder, FiniteResponseLawLowerPayload, MAXIMUM_LAW_BYTES
from .science import PROGRAMME

PRODUCTS = ("response-values", "uncertainty-values")


def law_registration(boundary: str, implementation: ObjectIdentity) -> LawEvaluatorRegistration:
    if boundary not in ("lower", "composed", "cached", "direct"):
        raise ValueError("Unknown finite response law boundary")
    decoder = FiniteResponseLawLowerDecoder() if boundary == "lower" else FiniteResponseLawConditionalDecoder()
    key = f"{PROGRAMME}.{boundary}-law"
    return LawEvaluatorRegistration(
        f"{key}.registration",
        key,
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
        f"{PROGRAMME}.law-offline",
        f"{PROGRAMME}.law-online-unqualified",
        MAXIMUM_LAW_BYTES,
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawLawEvaluator:
    registration: LawEvaluatorRegistration
    boundary: str
    payload_reader: CandidatePayloadReader | None = None
    lower_publication: CandidatePayloadPublicationReceipt | None = None

    def __post_init__(self) -> None:
        if self.registration != law_registration(self.boundary, self.registration.implementation):
            raise ValueError("Finite response-law evaluator registration changes its frozen boundary")
        if self.boundary != "composed" and (
            self.payload_reader is not None or self.lower_publication is not None
        ):
            raise ValueError("Only composition may bind the lower publication")

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        decoder = FiniteResponseLawLowerDecoder() if self.boundary == "lower" else FiniteResponseLawConditionalDecoder()
        decoded = decoder.decode(payload, maximum_bytes=maximum_bytes)
        if isinstance(decoded, FiniteResponseLawConditionalPayload) and decoded.boundary != self.boundary:
            raise ValueError("Finite response-law evaluator cannot substitute another conditional boundary")
        return decoded

    def _lower(self, payload: FiniteResponseLawConditionalPayload) -> FiniteResponseLawLowerPayload | None:
        publication = self.lower_publication
        if self.payload_reader is None or publication is None:
            return None
        decoder = FiniteResponseLawLowerDecoder()
        if (
            publication.candidate_evaluator.payload.payload_schema != decoder.payload_schema
            or publication.decoder_schema != decoder.decoder_schema
            or publication.decoder_version != decoder.decoder_version
            or publication.maximum_decode_bytes > MAXIMUM_LAW_BYTES
        ):
            raise CandidatePayloadReadError("FLH_LOWER_PUBLICATION_DIFFERS")
        try:
            raw = self.payload_reader.read_candidate_payload(publication)
        except Exception as error:
            raise CandidatePayloadReadError("FLH_LOWER_PAYLOAD_UNREADABLE") from error
        if (
            len(raw) > publication.maximum_decode_bytes
            or sha256(raw).hexdigest() != publication.content_sha256
        ):
            raise CandidatePayloadReadError("FLH_LOWER_PAYLOAD_DIGEST_OR_BOUND_DIFFERS")
        lower = decoder.decode(raw, maximum_bytes=publication.maximum_decode_bytes)
        if not isinstance(lower, FiniteResponseLawLowerPayload) or lower.identity != payload.lower:
            raise CandidatePayloadReadError("FLH_LOWER_PAYLOAD_IDENTITY_DIFFERS")
        return lower

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, FiniteResponseLawLowerPayload | FiniteResponseLawConditionalPayload) or (
            (self.boundary == "lower") != isinstance(payload, FiniteResponseLawLowerPayload)
            or isinstance(payload, FiniteResponseLawConditionalPayload)
            and payload.boundary != self.boundary
        ):
            raise TypeError("Finite response-law evaluator received another boundary payload")

        def refuse(disposition: LawEvaluationDisposition, reason: str) -> LawEvaluationResult:
            return _refusal(request, self.registration.implementation, disposition, reason)

        if (
            request.evaluation_mode is not LawEvaluationMode.OFFLINE
            or request.requested_product_ids != PRODUCTS
        ):
            return refuse(
                LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                "flh-offline-paired-mean-and-interval-only",
            )
        outputs = output_quantities()
        inputs = feature_quantities(self.boundary)
        if self.boundary != "lower":
            inputs = (*inputs, parent_quantity())
        known = {q.quantity_id: q for q in system.quantities}
        # The shared service binds values to SystemSpec. Here bind that system
        # to the frozen instrument, causal cutoff and paired receiver protocol.
        if any(known.get(q.quantity_id) != q for q in (*inputs, *outputs)) or (
            law.relation.horizon != HORIZON
            or law.relation.receiver_quantity_ids != tuple(sorted(q.quantity_id for q in outputs))
            or not {q.quantity_id for q in inputs}.issubset(law.interface_input_quantity_ids)
        ):
            raise ValueError(
                "Finite response law changes frozen quantities, units, frame, cutoff or paired horizon"
            )
        if payload.q is None:
            return refuse(LawEvaluationDisposition.OPERAND_ABSENT, "flh-calibration-unavailable")
        expected_artifacts = tuple(
            sorted(
                (payload.development_manifest, payload.frozen_coefficients),
                key=lambda a: a.artifact_id,
            )
        )
        values = {value.quantity_id: value.value for value in request.input_values}
        if (
            request.input_artifacts != expected_artifacts
            or len(values) != len(request.input_values)
            or set(values) != {q.quantity_id for q in inputs}
        ):
            return refuse(
                LawEvaluationDisposition.OPERAND_ABSENT,
                "flh-frozen-provenance-or-causal-inputs-absent",
            )
        action = request.action_word
        if action is None or action.support_status is not ActionWordSupportStatus.SUPPORTED:
            return refuse(LawEvaluationDisposition.OUTSIDE_SUPPORT, "flh-native-action-unsupported")
        # Compare the complete declaration, including every stage, transport,
        # waveform/duration and receiver. Support alone is assigned by its owner.
        matches = [
            i
            for i, word in enumerate(NATIVE_WORDS)
            if action
            in (
                original := replace(
                    native_action_word(
                        word,
                        denominator_id=request.prepared_denominator_id,
                        history_id=action.retained_history_id,
                    ),
                    support_status=action.support_status,
                    reason_codes=action.reason_codes,
                ),
                controller_word_map(original).controller_word,
            )
        ]
        if len(matches) != 1:
            return refuse(
                LawEvaluationDisposition.OUTSIDE_SUPPORT,
                "flh-native-pulse-or-paired-receiver-differs",
            )
        features = feature_quantities(self.boundary)
        x = np.asarray([[float(values[q.quantity_id]) for q in features]], dtype=np.float64)
        if isinstance(payload, FiniteResponseLawLowerPayload):
            prediction = payload.predict(x)
        else:
            parent = values[PARENT_QUANTITY]
            if parent not in tuple(map(D, range(5))):
                return refuse(LawEvaluationDisposition.MEMBER_ABSENT, "flh-native-parent-absent")
            lower = self._lower(payload) if self.boundary == "composed" else None
            if self.boundary == "composed" and (lower is None or lower.q is None):
                return refuse(
                    LawEvaluationDisposition.OPERAND_ABSENT,
                    "flh-exact-lower-publication-unavailable",
                )
            # Cached ignores the current interface. The padding only satisfies
            # its shared arithmetic shape and is never an observed feature.
            prediction = payload.predict(
                np.zeros((1, 24), dtype=np.float64) if self.boundary == "cached" else x,
                np.array([int(parent)], dtype=np.int64),
                lower=lower,
            )
        if not prediction.supported[0]:
            return refuse(
                LawEvaluationDisposition.OUTSIDE_SUPPORT, "flh-frozen-feature-support-exceeded"
            )
        width = prediction.sigma * float(payload.q)
        if not np.isfinite(prediction.mean).all() or not np.isfinite(width).all():
            return refuse(
                LawEvaluationDisposition.OPERAND_ABSENT, "flh-nonfinite-prediction-or-width"
            )
        mean, _ = oriented_interval(prediction.mean, prediction.mean, matches[0])
        _, halfwidth = oriented_interval(-width, width, matches[0])

        def bound(role: str, numbers: np.ndarray) -> tuple[LawEvaluationValue, ...]:
            return tuple(
                sorted(
                    (
                        LawEvaluationValue(
                            f"{role}.{q.quantity_id}",
                            q.quantity_id,
                            D(format(float(number), ".17g")),
                            q.native_unit,
                            q.coordinate_frame,
                            q.clock_id,
                        )
                        for q, number in zip(outputs, numbers, strict=True)
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
            ObjectIdentity.from_record(action.word_id, action),
            LawEvaluationDisposition.SUPPORTED,
            bound("mean", mean[0]),
            (),
            (),
            tuple(
                sorted(
                    (
                        *bound("halfwidth", halfwidth[0]),
                        *bound(
                            "numerical-allowance",
                            np.array([float(d / 8) for d in payload.chart.science.delta]),
                        ),
                    ),
                    key=lambda v: v.value_id,
                )
            ),
            (),
            (),
        )
