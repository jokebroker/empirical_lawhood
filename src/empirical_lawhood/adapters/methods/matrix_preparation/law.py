"""Published finite action/readout predictors on the existing law-evaluation port."""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.laws import LawRepresentationKind, ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.systems import SystemSpec
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
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_law import response_geometry_development_word_sign
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import CONTEXTS, DEVELOPMENT, PARENTS, READOUTS
from .contracts import CANDIDATES, FEATURES, FIT_SCHEMA
from .data import Array
from .models import ObservablePredictor, ResidualScaleModel
from .records import PreparationFitReport


LAW_KEY = f"{DEVELOPMENT}.finite-observable-response"
LAW_MAXIMUM_BYTES = 1024**2
FEATURE_QUANTITIES = tuple(f"{DEVELOPMENT}.observable.{name.lower()}" for name in FEATURES)
FEATURE_UNITS = (
    "hilbert-schmidt-native",
    "native-hs-momentum",
    "native-hs-position-squared",
    "native-hs-position-squared",
    "native-hs-momentum-squared",
    "native-hs-momentum-squared",
    "native-hs-position-squared-per-native-time",
    "native-hs-position-per-native-time-squared",
    "native-time",
)
MODEL_ARRAY_KEYS = (
    "mean",
    "scale",
    "operator",
    "constant_odd_even",
    "penalty",
    "scale_means",
    "scale_scales",
    "scale_operators",
    "reference_rms_scales",
)


def coefficient_shapes(candidate: str) -> dict[str, tuple[int, ...]]:
    if candidate not in CANDIDATES:
        raise ValueError("finite response law adds an undeclared observable candidate")
    return {
        "mean": (9,),
        "scale": (9,),
        "operator": (10, 5)
        if candidate == "constant_gain"
        else (55 if candidate == "direct_quadratic" else 10, 15),
        "constant_odd_even": (2, 5) if candidate == "constant_gain" else (0, 5),
        "penalty": (1,),
        "scale_means": (2, 9),
        "scale_scales": (2, 9),
        "scale_operators": (2, 10, 15),
        "reference_rms_scales": (2, 3, 5),
        "support_lower": (5, 2, 9),
        "support_upper": (5, 2, 9),
    }


@dataclass(frozen=True, slots=True)
class PreparationLawPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-law-payload'
    payload_id: str
    context: str
    candidate: str
    fit: ObjectIdentity
    fit_artifact: ArtifactIdentity
    coefficients: tuple[tuple[str, tuple[Decimal, ...]], ...]
    calibration_quantile: Decimal | None
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if (
            self.context not in CONTEXTS
            or self.candidate not in CANDIDATES
            or self.payload_id != f"{LAW_KEY}.{self.context}.{self.candidate}"
            or self.fit.object_schema != PreparationFitReport.SCHEMA
            or self.fit_artifact.payload_schema != FIT_SCHEMA
            or (
                self.calibration_quantile is not None
                and (
                    type(self.calibration_quantile) is not Decimal
                    or not self.calibration_quantile.is_finite()
                    or self.calibration_quantile < 0
                )
            )
        ):
            raise ValueError("finite law changes its exact fitted predictor/calibration lineage")
        shapes = coefficient_shapes(self.candidate)
        if not self.coefficients:
            if not self.unavailable_reason or self.calibration_quantile is not None:
                raise ValueError("unavailable finite law must preserve its refusal")
            return
        if self.unavailable_reason is not None or tuple(k for k, _ in self.coefficients) != tuple(
            sorted(shapes)
        ):
            raise ValueError("finite law omits or adds a predictor coefficient block")
        for key, values in self.coefficients:
            if len(values) != int(np.prod(shapes[key])) or any(
                type(v) is not Decimal or not v.is_finite() for v in values
            ):
                raise ValueError("finite law changes a coefficient chart or uses nonfinite values")
        arrays = self.arrays()
        if (
            any(not np.isfinite(v).all() for v in arrays.values())
            or np.any(arrays["scale"] <= 0)
            or np.any(arrays["scale_scales"] <= 0)
            or np.any(arrays["support_lower"] > arrays["support_upper"])
            or np.any(arrays["reference_rms_scales"] < 1 / 128)
            or arrays["penalty"][0] not in (0.01, 0.1, 1.0, 10.0, 100.0)
        ):
            raise ValueError("finite law coefficient numeric support is unresolved")

    def arrays(self) -> dict[str, Array]:
        shapes = coefficient_shapes(self.candidate)
        return {
            key: np.asarray(values, dtype=float).reshape(shapes[key])
            for key, values in self.coefficients
        }

    def predictors(self) -> tuple[ObservablePredictor, ResidualScaleModel]:
        if not self.coefficients:
            raise ValueError("unavailable finite law has no predictor")
        a = self.arrays()
        return ObservablePredictor(
            self.candidate,
            float(a["penalty"][0]),
            a["mean"],
            a["scale"],
            a["operator"],
            a["constant_odd_even"] if self.candidate == "constant_gain" else None,
            32,
        ), ResidualScaleModel(
            a["scale_means"], a["scale_scales"], a["scale_operators"], a["reference_rms_scales"], 32
        )


def build_law_payload(
    candidate: str,
    fit: PreparationFitReport,
    fit_artifact: ArtifactIdentity,
    arrays: Mapping[str, Array] | None,
    training_features: Array,
) -> PreparationLawPayload:
    if fit_artifact.sha256 != fit.data_sha256 or training_features.shape != (32, 5, 2, 9):
        raise ValueError("finite law changes its authenticated training/payload inputs")
    coefficients: tuple[tuple[str, tuple[Decimal, ...]], ...] = ()
    quantile = None
    if arrays is not None:
        values = {key: arrays[key] for key in MODEL_ARRAY_KEYS}
        values.update(
            support_lower=training_features.min(axis=0), support_upper=training_features.max(axis=0)
        )
        coefficients = tuple(
            (key, tuple(Decimal(format(float(v), ".17g")) for v in values[key].ravel()))
            for key in sorted(values)
        )
        raw = float(arrays["quantiles"][1])
        quantile = Decimal(format(raw, ".17g")) if np.isfinite(raw) else None
    return PreparationLawPayload(
        f"{LAW_KEY}.{fit.context}.{candidate}",
        fit.context,
        candidate,
        ObjectIdentity.from_record(fit.report_id, fit),
        fit_artifact,
        coefficients,
        quantile,
        None if arrays is not None else dict(fit.unavailable_candidates)[candidate],
    )


@dataclass(frozen=True, slots=True)
class PreparationLawDecoder:
    extension_namespace: str = "matrix-preparation-finite-observable-law"
    decoder_schema: str = 'empirical-lawhood/methods/matrix-preparation/finite-observable-law-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = PreparationLawPayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, PreparationLawPayload, maximum_bytes=min(maximum_bytes, LAW_MAXIMUM_BYTES)
        )


def law_registration(implementation: ObjectIdentity) -> LawEvaluatorRegistration:
    d = PreparationLawDecoder()
    return LawEvaluatorRegistration(
        f"{LAW_KEY}.registration",
        LAW_KEY,
        LawEvaluationKind.POINT_NAMED_VALUES,
        LawRepresentationKind.FINITE_ACTION_OPERATOR,
        d.payload_schema,
        d.extension_namespace,
        d.decoder_schema,
        d.decoder_version,
        LawEvaluationRequest.SCHEMA,
        LawEvaluationResult.SCHEMA,
        implementation,
        True,
        "matrix-preparation-finite-offline",
        "matrix-preparation-online-unqualified",
        LAW_MAXIMUM_BYTES,
    )


@dataclass(frozen=True, slots=True)
class PreparationLawEvaluator:
    registration: LawEvaluatorRegistration

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return PreparationLawDecoder().decode(payload, maximum_bytes=maximum_bytes)

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, PreparationLawPayload):
            raise TypeError("preparation evaluator received another scientific payload")

        def refuse(kind: LawEvaluationDisposition, reason: str) -> LawEvaluationResult:
            return _refusal(request, self.registration.implementation, kind, reason)

        products = tuple(
            sorted(f"{role}-{j:03d}" for role in ("mean", "halfwidth") for j in READOUTS)
        )
        if (
            request.evaluation_mode is not LawEvaluationMode.OFFLINE
            or request.requested_product_ids != products
        ):
            return refuse(
                LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                "finite-observable-law-products-or-online-mode-unqualified",
            )
        if (
            request.input_artifacts != (payload.fit_artifact,)
            or not payload.coefficients
            or payload.calibration_quantile is None
        ):
            return refuse(
                LawEvaluationDisposition.OPERAND_ABSENT,
                "finite-predictor-or-calibration-unavailable",
            )
        parent = next(
            (
                i
                for i, p in enumerate(PARENTS)
                if request.denominator_member_id == f"{DEVELOPMENT}.{payload.context}.{p}"
            ),
            None,
        )
        view = next(
            (
                i
                for i, v in enumerate(system.numerical_views)
                if v.view_id == request.qualification_view_id
            ),
            None,
        )
        if parent is None or view not in (0, 1):
            return refuse(
                LawEvaluationDisposition.MEMBER_ABSENT, "unentered-parent-or-numerical-view"
            )
        values = {v.quantity_id: float(v.value) for v in request.input_values}
        if set(values) != set(FEATURE_QUANTITIES) or len(values) != len(request.input_values):
            return refuse(
                LawEvaluationDisposition.OPERAND_ABSENT, "causal-nine-observable-inputs-required"
            )
        x = np.asarray([values[k] for k in FEATURE_QUANTITIES])
        arrays = payload.arrays()
        if (
            not np.isfinite(x).all()
            or np.any(x < arrays["support_lower"][parent, view])
            or np.any(x > arrays["support_upper"][parent, view])
        ):
            return refuse(
                LawEvaluationDisposition.OUTSIDE_SUPPORT, "outside-training-observer-support"
            )
        sign = None if request.action_word is None else response_geometry_development_word_sign(request.action_word)
        origin = (1024 if payload.context == "assembling" else 4096) + 144
        if (
            sign is None
            or request.action_word is None
            or request.action_word.occurrences[0].applied.coordinate.coordinate != Decimal(origin)
        ):
            return refuse(
                LawEvaluationDisposition.OUTSIDE_SUPPORT, "unentered-force-pulse-or-handoff-clock"
            )
        model, scale_model = payload.predictors()
        features = np.broadcast_to(x, (1, 5, 2, 9))
        means = model.predict(features)[0, parent, view, sign + 1]
        widths = scale_model.predict(features)[0, parent, view, sign + 1] * float(
            payload.calibration_quantile
        )
        if not np.isfinite(means).all() or not np.isfinite(widths).all():
            return refuse(
                LawEvaluationDisposition.OPERAND_ABSENT, "nonfinite-finite-chart-prediction"
            )
        receiver = next(
            q for q in system.quantities if q.quantity_id == law.relation.receiver_quantity_ids[0]
        )

        def output(role: str, j: int, value: float) -> LawEvaluationValue:
            return LawEvaluationValue(
                f"{role}-{j:03d}",
                receiver.quantity_id,
                Decimal(format(float(value), ".17g")),
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
            tuple(output("mean", j, v) for j, v in zip(READOUTS, means, strict=True)),
            (),
            (),
            tuple(output("halfwidth", j, v) for j, v in zip(READOUTS, widths, strict=True)),
            (),
            (),
        )
