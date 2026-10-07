"""development's complete affine forecast payload behind the existing law-evaluation port.

The fitted feature map remains bound to its published fit artifact. The input
is its causal invocation state, not a future HOLD trajectory. This evaluator
supplies model-conditional endpoint forecasts offline; development does not admit an
online stopping policy or a joint response/preservation distribution.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ActionWordMode
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.laws import ResponseLaw, LawRepresentationKind
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import CoordinateOrigin
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
from empirical_lawhood.adapters.simulators.response_geometry_prospective.development_actions import DEVELOPMENT_ACTION_KNOTS, DEVELOPMENT_CLOCK, DEVELOPMENT_EPISODE_FRAME, DEVELOPMENT_FORCE_DIRECTION, DEVELOPMENT_FORCE_FRAME, DEVELOPMENT_FORCE_UNIT
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import PARENTS

from .development_models import REPRESENTATIONS, ResponseGeometryDevelopmentAffineModel, ResponseGeometryDevelopmentModelGroup, FloatArray, propagate_response_geometry_development_latent
from .development_records import DEVELOPMENT_FIT_SCHEMA, ResponseGeometryDevelopmentCalibrationResult, ResponseGeometryDevelopmentFitResult


DEVELOPMENT_LAW_KEY = "response-geometry-development.complete-affine-law"
DEVELOPMENT_LAW_DECODER_SCHEMA = 'empirical-lawhood/methods/response-geometry-prospective/development-affine-law-decoder'
DEVELOPMENT_LAW_MAXIMUM_BYTES = 1024 * 1024
DEVELOPMENT_LATENT_QUANTITIES = tuple(f"response-geometry-development.latent.{i:02d}" for i in range(8))
DEVELOPMENT_INVOCATION_QUANTITY = "response-geometry-development.invocation-offset"


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentAffineCoefficients(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-affine-coefficients'
    parent: str
    dimension: int
    phases: int
    drift: tuple[Decimal, ...]
    diffusion: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if self.parent not in PARENTS or self.dimension not in (4, 8) or self.phases not in (1, 2):
            raise ValueError("development law coefficients change their declared chart")
        if (
            len(self.drift) != self.phases * (self.dimension + 2) * self.dimension
            or len(self.diffusion) != self.phases * self.dimension**2
        ):
            raise ValueError("development law omits affine or innovation coefficients")
        if any(not v.is_finite() for v in (*self.drift, *self.diffusion)):
            raise ValueError("development law coefficients must be finite")
        drift, diffusion = self.arrays()
        if not np.isfinite(drift).all() or not np.isfinite(diffusion).all():
            raise ValueError("development law coefficients exceed float64")
        for covariance in diffusion:
            if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14) or (
                np.min(np.linalg.eigvalsh(covariance))
                < -1e-12 * max(1.0, float(np.linalg.norm(covariance)))
            ):
                raise ValueError("development innovation covariance must be symmetric positive semidefinite")
        if self.phases == 2 and np.any(drift[1, -2] != 0):
            raise ValueError("development law cannot invent a post-force input coefficient")

    def arrays(self) -> tuple[FloatArray, FloatArray]:
        return (
            np.asarray(self.drift, dtype=np.float64).reshape(
                self.phases, self.dimension + 2, self.dimension
            ),
            np.asarray(self.diffusion, dtype=np.float64).reshape(
                self.phases, self.dimension, self.dimension
            ),
        )

    @classmethod
    def from_model(cls, model: ResponseGeometryDevelopmentAffineModel) -> 'ResponseGeometryDevelopmentAffineCoefficients':
        return cls(
            model.parent,
            model.dimension,
            len(model.drift),
            tuple(Decimal(format(float(v), ".17g")) for v in model.drift.ravel()),
            tuple(Decimal(format(float(v), ".17g")) for v in model.diffusion.ravel()),
        )


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentAffineLawPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-geometry-prospective/response-geometry-development-affine-law-payload'
    payload_id: str
    context: str
    representation: str
    fit_result: ObjectIdentity
    fit_artifact: ArtifactIdentity
    calibration_result: ObjectIdentity
    calibration_quantile: Decimal | None
    members: tuple[ResponseGeometryDevelopmentAffineCoefficients, ...]
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.payload_id, field_name="payload_id")
        if (
            self.context not in ("assembling", "prepared")
            or self.representation not in REPRESENTATIONS
        ):
            raise ValueError("development law changes its selected context/representation")
        if (
            self.fit_result.object_schema != ResponseGeometryDevelopmentFitResult.SCHEMA
            or self.fit_artifact.payload_schema != DEVELOPMENT_FIT_SCHEMA
            or self.calibration_result.object_schema != ResponseGeometryDevelopmentCalibrationResult.SCHEMA
        ):
            raise ValueError("development law loses its fit/calibration lineage")
        if self.calibration_quantile is not None and (
            not self.calibration_quantile.is_finite() or self.calibration_quantile < 0
        ):
            raise ValueError("development law calibration must be finite nonnegative or unavailable")
        if self.members:
            if (
                tuple(m.parent for m in self.members) != PARENTS
                or len({(m.dimension, m.phases) for m in self.members}) != 1
                or self.unavailable_reason is not None
            ):
                raise ValueError("development law changes its complete selected five-parent family")
        elif not self.unavailable_reason:
            raise ValueError("development unavailable law must retain its model refusal")


def build_response_geometry_development_affine_payload(
    group: ResponseGeometryDevelopmentModelGroup,
    fit: ResponseGeometryDevelopmentFitResult,
    fit_artifact: ArtifactIdentity,
    calibration: ResponseGeometryDevelopmentCalibrationResult,
) -> ResponseGeometryDevelopmentAffineLawPayload:
    fit_identity = ObjectIdentity.from_record(fit.result_id, fit)
    if (
        group.context != fit.context
        or calibration.fit_result != fit_identity
        or calibration.calibration.context != fit.context
        or calibration.config != fit.config
        or fit_artifact.sha256 != fit.data_sha256
        or bool(group.models) != (group.representation in fit.available_representations)
    ):
        raise ValueError("development law publication changes its authenticated fit/calibration family")
    return ResponseGeometryDevelopmentAffineLawPayload(
        f"development.affine.{group.context}.{group.representation}",
        group.context,
        group.representation,
        fit_identity,
        fit_artifact,
        ObjectIdentity.from_record(calibration.result_id, calibration),
        calibration.calibration.quantile,
        tuple(ResponseGeometryDevelopmentAffineCoefficients.from_model(model) for model in group.models),
        group.reason,
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentAffineLawDecoder:
    extension_namespace: str = "response-geometry-development-affine-law"
    decoder_schema: str = DEVELOPMENT_LAW_DECODER_SCHEMA
    decoder_version: str = "1.0.0"
    payload_schema: str = ResponseGeometryDevelopmentAffineLawPayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, ResponseGeometryDevelopmentAffineLawPayload, maximum_bytes=min(maximum_bytes, DEVELOPMENT_LAW_MAXIMUM_BYTES)
        )


def response_geometry_development_law_registration(implementation: ObjectIdentity) -> LawEvaluatorRegistration:
    """Declare the existing named-value interface for development's complete forecast."""
    decoder = ResponseGeometryDevelopmentAffineLawDecoder()
    return LawEvaluatorRegistration(
        f"{DEVELOPMENT_LAW_KEY}.registration",
        DEVELOPMENT_LAW_KEY,
        LawEvaluationKind.POINT_NAMED_VALUES,
        LawRepresentationKind.LOCAL_STATE_SPACE,
        decoder.payload_schema,
        decoder.extension_namespace,
        decoder.decoder_schema,
        decoder.decoder_version,
        LawEvaluationRequest.SCHEMA,
        LawEvaluationResult.SCHEMA,
        implementation,
        True,
        "development-affine-offline",
        "development-affine-online-unqualified",
        DEVELOPMENT_LAW_MAXIMUM_BYTES,
    )


def response_geometry_development_word_sign(word: OccurrenceActionWord) -> int | None:
    """Accept only the actual finite short-pulse-response force shape; stage expectations must agree."""
    if (
        word.mode is not ActionWordMode.SEQUENTIAL
        or len(word.occurrences) != 1
        or word.ordering_origin is not CoordinateOrigin.EPISODE_RELATIVE
        or word.ordering_time_unit != "reference-tick"
        or word.ordering_clock_id != DEVELOPMENT_CLOCK
        or word.ordering_coordinate_frame != DEVELOPMENT_EPISODE_FRAME
    ):
        return None
    start = word.occurrences[0].applied.coordinate.coordinate
    force = word.occurrences[0].applied.value
    if force not in (Decimal(-8), Decimal(0), Decimal(8)):
        return None
    for index, occurrence in enumerate(word.occurrences):
        if (
            occurrence.duration != DEVELOPMENT_ACTION_KNOTS[index + 1] - DEVELOPMENT_ACTION_KNOTS[index]
            or occurrence.duration_unit != "reference-tick"
            or occurrence.channel.port_id != "response-geometry.x-force"
            or occurrence.channel.controller_quantity_id != "response-geometry.x-force-command"
        ):
            return None
        for event in (
            occurrence.requested,
            occurrence.accepted,
            occurrence.applied,
            occurrence.realized,
        ):
            if (
                event.value != force
                or event.coordinate.coordinate != start + DEVELOPMENT_ACTION_KNOTS[index]
                or event.coordinate.time_unit != "reference-tick"
                or event.coordinate.clock_id != DEVELOPMENT_CLOCK
                or event.coordinate.coordinate_frame != DEVELOPMENT_EPISODE_FRAME
                or event.coordinate.origin is not CoordinateOrigin.EPISODE_RELATIVE
                or event.native_unit != DEVELOPMENT_FORCE_UNIT
                or event.native_action_frame != DEVELOPMENT_FORCE_FRAME
                or event.native_direction != DEVELOPMENT_FORCE_DIRECTION
            ):
                return None
    return int(force / 8)


@dataclass(frozen=True, slots=True)
class ResponseGeometryDevelopmentAffineLawEvaluator:
    registration: LawEvaluatorRegistration

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return ResponseGeometryDevelopmentAffineLawDecoder().decode(payload, maximum_bytes=maximum_bytes)

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, ResponseGeometryDevelopmentAffineLawPayload):
            raise TypeError("development affine evaluator received another payload")

        def refuse(disposition: LawEvaluationDisposition, reason: str) -> LawEvaluationResult:
            return _refusal(request, self.registration.implementation, disposition, reason)

        if (
            request.evaluation_mode is not LawEvaluationMode.OFFLINE
            or request.requested_product_ids
            != (
                "endpoint-mean",
                "endpoint-model-conditional-halfwidth",
            )
        ):
            return refuse(
                LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                "development-online-or-joint-product-unqualified",
            )
        member = next(
            (
                m
                for m in payload.members
                if request.denominator_member_id == f"development.{payload.context}.{m.parent}"
            ),
            None,
        )
        if member is None:
            return refuse(LawEvaluationDisposition.MEMBER_ABSENT, "development-selected-parent-model-absent")
        if payload.calibration_quantile is None:
            return refuse(LawEvaluationDisposition.OPERAND_ABSENT, "development-calibration-unavailable")
        if request.input_artifacts != (payload.fit_artifact,):
            return refuse(
                LawEvaluationDisposition.OPERAND_ABSENT, "development-frozen-feature-map-binding-absent"
            )
        inputs = {v.quantity_id: float(v.value) for v in request.input_values}
        expected = {*DEVELOPMENT_LATENT_QUANTITIES[: member.dimension], DEVELOPMENT_INVOCATION_QUANTITY}
        if len(inputs) != len(request.input_values) or set(inputs) != expected:
            return refuse(
                LawEvaluationDisposition.OPERAND_ABSENT, "development-causal-initial-coordinates-absent"
            )
        if inputs[DEVELOPMENT_INVOCATION_QUANTITY] not in range(384, 513, 16):
            return refuse(
                LawEvaluationDisposition.OUTSIDE_SUPPORT, "development-invocation-outside-sampled-design"
            )
        sign = None if request.action_word is None else response_geometry_development_word_sign(request.action_word)
        if sign is None or request.action_word is None:
            return refuse(
                LawEvaluationDisposition.OUTSIDE_SUPPORT, "development-action-outside-short-pulse-response-pulse-chart"
            )
        if request.action_word is not None:
            expected_origin = (1024 if payload.context == "assembling" else 4096) + inputs[
                DEVELOPMENT_INVOCATION_QUANTITY
            ]
            if request.action_word.occurrences[0].applied.coordinate.coordinate != Decimal(
                str(expected_origin)
            ):
                return refuse(
                    LawEvaluationDisposition.OUTSIDE_SUPPORT, "development-action-invocation-clock-differs"
                )
        initial = np.asarray([inputs[q] for q in DEVELOPMENT_LATENT_QUANTITIES[: member.dimension]])
        try:
            means, sigma = propagate_response_geometry_development_latent(initial, *member.arrays())
            halfwidth = float(payload.calibration_quantile) * sigma[-1]
        except (ValueError, FloatingPointError):
            return refuse(LawEvaluationDisposition.OPERAND_ABSENT, "development-nonfinite-model-propagation")
        if not np.isfinite(halfwidth):
            return refuse(LawEvaluationDisposition.OPERAND_ABSENT, "development-nonfinite-calibrated-bound")
        if len(law.relation.receiver_quantity_ids) != 1:
            raise ValueError("development law requires its single signed native receiver")
        receiver = next(
            q for q in system.quantities if q.quantity_id == law.relation.receiver_quantity_ids[0]
        )

        def value(value_id: str, number: float) -> LawEvaluationValue:
            return LawEvaluationValue(
                value_id,
                receiver.quantity_id,
                Decimal(format(number, ".17g")),
                receiver.native_unit,
                receiver.coordinate_frame,
                receiver.clock_id,
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
            (value("endpoint-mean", float(means[sign + 1, -1])),),
            (),
            (),
            (value("endpoint-model-conditional-halfwidth", halfwidth),),
            (),
            (),
        )
