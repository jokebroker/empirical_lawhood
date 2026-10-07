"""Public finite predictions for the existing law-evaluation service.

Publication and terminal qualification remain with their existing owners.
This payload excludes dependent refinement roots, scenarios, residuals and native checkpoints.
It does not decide support, admission, action selection or forecast success.
"""

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationDisposition,
    LawEvaluationRequest,
    LawEvaluationResult,
    LawEvaluationValue,
    LawEvaluatorRegistration,
    _refusal,
)
from empirical_lawhood.adapters.simulators.prepared_response.development_roster import prepared_response_development_action_word, prepared_response_development_clock_evaluator
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.adapters.simulators.prepared_response.contracts import READOUTS

from .calibration_records import PreparedResponseCalibrationCalibratedLibrary, PreparedResponseCalibrationContextCalibration
from .coefficients import PreparedBilinearPredictorCoefficients

MAXIMUM_PUBLIC_LAW_BYTES = 4 * 1024**2


@dataclass(frozen=True, slots=True)
class PreparedFiniteLawPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-finite-law-payload'
    coefficients: PreparedBilinearPredictorCoefficients
    calibration: ObjectIdentity
    calibrated_multiplier: Decimal
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_ids: tuple[str, ...]
    # Native word order is HOLD followed by the eight signed port directions.
    action_words: tuple[OccurrenceActionWord, ...]
    history_quantity_ids: tuple[str, ...]
    sketch_quantity_ids: tuple[str, ...]
    output_quantity_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for value in (self.denominator_member_id, self.candidate_version_id):
            validate_stable_id(value, field_name="law axis")
        if (
            self.calibration.object_schema != PreparedResponseCalibrationContextCalibration.SCHEMA
            or type(self.calibrated_multiplier) is not Decimal
            or not self.calibrated_multiplier.is_finite()
            or self.calibrated_multiplier < 0
            or len(self.action_words) != 9
            or len({word.word_id for word in self.action_words}) != 9
            or len({(word.denominator_id, word.retained_history_id, word.horizon_id)
                    for word in self.action_words}) != 1
        ):
            raise ValueError("public prepared response law changes its calibration or finite action chart")
        expected_sketch = 8 if self.coefficients.structure == "mechanism-i1" else 0
        for values, count in (
            (self.qualification_view_ids, 2), (self.history_quantity_ids, 12),
            (self.sketch_quantity_ids, expected_sketch), (self.output_quantity_ids, 7),
        ):
            if len(values) != count or len(set(values)) != count:
                raise ValueError("public prepared response law changes its view or scalar interface roster")
            for value in values:
                validate_stable_id(value, field_name="scalar interface coordinate")
        if set(self.history_quantity_ids) & set(self.sketch_quantity_ids):
            raise ValueError("prepared response history and mechanism coordinates must remain distinct")
        if len(self.canonical_bytes()) > MAXIMUM_PUBLIC_LAW_BYTES:
            raise ValueError("public prepared response law exceeds its bounded payload")

    @property
    def input_coordinates(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (f"history.s{sample:02d}.c{channel:02d}", quantity)
            for sample in range(16)
            for channel, quantity in enumerate(self.history_quantity_ids)
        ) + tuple(
            (f"sketch.c{channel:02d}", quantity)
            for channel, quantity in enumerate(self.sketch_quantity_ids)
        )


@dataclass(frozen=True, slots=True)
class PreparedFiniteLawDecoder:
    "The existing qualification registry's decoder for the prepared response payload."

    extension_namespace: ClassVar[str] = "prepared-response-finite-law"
    decoder_schema: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/finite-law-decoder'
    decoder_version: ClassVar[str] = "1.0.0"
    payload_schema: ClassVar[str] = PreparedFiniteLawPayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> PreparedFiniteLawPayload:
        return decode_canonical_bytes(
            payload, PreparedFiniteLawPayload,
            maximum_bytes=min(maximum_bytes, MAXIMUM_PUBLIC_LAW_BYTES),
        )


def prepared_finite_law_payload(
    calibrated: PreparedResponseCalibrationCalibratedLibrary,
    *,
    context: str,
    denominator_member_id: str,
    candidate_version_id: str,
    action_words: tuple[OccurrenceActionWord, ...],
    history_quantity_ids: tuple[str, ...],
    sketch_quantity_ids: tuple[str, ...],
    output_quantity_ids: tuple[str, ...],
) -> PreparedFiniteLawPayload:
    "Strip private training/calibration rows while binding the actual fresh calibration product."
    if calibrated.disposition != "CALIBRATED_FOR_C_R":
        raise ValueError("an uncalibratable fresh calibration product cannot publish finite prediction sets")
    # Reuse the native declaration: the predictor's nine array indices cannot
    # acquire a different action meaning through caller-supplied tuple order.
    expected_words = tuple(
        prepared_response_development_action_word(word, prepared_response_development_clock_evaluator())
        for word in calibrated.config.projection.native_spec.words
    )
    if len(action_words) != len(expected_words) or any(
        replace(actual, support_status=expected.support_status, reason_codes=expected.reason_codes)
        != expected
        for actual, expected in zip(action_words, expected_words, strict=True)
    ):
        raise ValueError("public prepared response law changes its frozen native action declaration")
    library = calibrated.config.projection.development_library
    matches = [
        (coefficient, calibration)
        for coefficient, calibration in zip(library.selected_coefficients, calibrated.contexts, strict=True)
        if calibration.context == context
    ]
    if len(matches) != 1:
        raise ValueError("public prepared response law requires one exact calibrated context")
    coefficients, calibration = matches[0]
    if (
        coefficients.model_spec.context != context
        or calibration.coefficient_product != ObjectIdentity.from_record(
            f"prepared-response.dependent-refinement.coefficients.{context}", coefficients,
        )
        or calibration.calibrated_multiplier is None
    ):
        raise ValueError("public prepared response law calibration is detached from its frozen dependent refinement coefficients")
    return PreparedFiniteLawPayload(
        coefficients, ObjectIdentity.from_record(calibration.calibration_id, calibration),
        calibration.calibrated_multiplier, denominator_member_id, candidate_version_id,
        tuple(view.view_id for view in calibrated.config.projection.native_spec.numerical_views),
        action_words, history_quantity_ids, sketch_quantity_ids, output_quantity_ids,
    )


@dataclass(frozen=True, slots=True)
class PreparedFiniteLawEvaluator:
    """Adapter implementation supplied to the existing LawEvaluatorRegistry."""

    registration: LawEvaluatorRegistration

    def decode(self, payload: bytes, *, maximum_bytes: int) -> PreparedFiniteLawPayload:
        return PreparedFiniteLawDecoder().decode(payload, maximum_bytes=maximum_bytes)

    def evaluate(
        self, system: SystemSpec, law: ResponseLaw,
        request: LawEvaluationRequest, payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if type(payload) is not PreparedFiniteLawPayload:
            raise TypeError("prepared response finite evaluator received another payload type")
        implementation = self.registration.implementation
        if (
            request.denominator_member_id != payload.denominator_member_id
            or request.candidate_version_id != payload.candidate_version_id
            or request.qualification_view_id not in payload.qualification_view_ids
        ):
            return _refusal(request, implementation, LawEvaluationDisposition.MEMBER_ABSENT,
                            "prepared-response-qualified-axis-absent")
        if request.action_word not in payload.action_words:
            return _refusal(request, implementation, LawEvaluationDisposition.OUTSIDE_SUPPORT,
                            "prepared-response-action-outside-frozen-menu")
        values = {value.value_id: value for value in request.input_values}
        coordinates = payload.input_coordinates
        if set(values) != {name for name, _ in coordinates}:
            return _refusal(request, implementation, LawEvaluationDisposition.OPERAND_ABSENT,
                            "prepared-response-causal-instrument-roster-incomplete")
        if any(values[name].quantity_id != quantity for name, quantity in coordinates):
            raise ValueError("prepared response input coordinate changes its native scalar quantity")
        if not {quantity for _, quantity in coordinates} <= set(law.interface_input_quantity_ids):
            raise ValueError("prepared response finite inputs exceed their qualified interface")
        products = set(request.requested_product_ids)
        if "response-values" not in products or not products <= {
            "response-values", "sink-values", "effort-values", "uncertainty-values",
        }:
            return _refusal(request, implementation, LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                            "prepared-response-finite-product-not-claimed")
        vector = np.asarray([float(values[name].value) for name, _ in coordinates])
        if not np.isfinite(vector).all():
            return _refusal(request, implementation, LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                            "prepared-response-finite-input-numerically-unresolved")
        history = vector[:192].reshape(1, 16, 12)
        sketch = vector[192:].reshape(1, 8) if payload.sketch_quantity_ids else None
        model, scale = payload.coefficients.predictors()
        word = payload.action_words.index(request.action_word)
        means = model.predict(history, sketch)[0, word]
        widths = float(payload.calibrated_multiplier) * scale.predict(history, sketch)[0, word]
        if not np.isfinite(widths).all():
            return _refusal(request, implementation, LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                            "prepared-response-finite-uncertainty-unresolved")
        quantities = {quantity.quantity_id: quantity for quantity in system.quantities}

        def result_values(channels: range, uncertainty: bool = False) -> tuple[LawEvaluationValue, ...]:
            result = []
            for readout, tick in enumerate(READOUTS):
                for channel in channels:
                    quantity_id = payload.output_quantity_ids[channel]
                    if quantity_id not in law.interface_output_quantity_ids:
                        raise ValueError("prepared response finite output exceeds its qualified interface")
                    quantity = quantities[quantity_id]
                    value = widths[readout, channel] if uncertainty else means[readout, channel]
                    result.append(LawEvaluationValue(
                        f"{'halfwidth' if uncertainty else 'mean'}.t{tick:03d}.c{channel:02d}",
                        quantity_id, Decimal(format(float(value), ".17g")),
                        quantity.native_unit, quantity.coordinate_frame, quantity.clock_id,
                    ))
            return tuple(result)

        return LawEvaluationResult(
            f"evaluation.{request.request_id}", ObjectIdentity.from_record(request.request_id, request),
            implementation, request.denominator_member_id, request.candidate_version_id,
            request.qualification_view_id,
            ObjectIdentity.from_record(request.action_word.word_id, request.action_word),
            LawEvaluationDisposition.SUPPORTED,
            result_values(range(2)),
            result_values(range(2, 5)) if "sink-values" in products else (),
            result_values(range(5, 7)) if "effort-values" in products else (),
            result_values(range(7), True) if "uncertainty-values" in products else (),
            (), (),
        )
