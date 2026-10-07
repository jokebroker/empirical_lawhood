"""Strict portable numeric products, distinct from a qualified response law.

The artifact port publishes these bytes with their producing task's lineage.
Law qualification/publication must separately bind that artifact, its fit and
calibration reports, support and method receipts. This record grants none of
those roles. It contains no native state, root seed, training label or runner.
"""

import base64
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, TypeVar

import numpy as np

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .models import Array, PreparedFinitePredictor, PreparedFittedModel, PreparedAffineModelSpec, PreparedBilinearModelSpec, _model_shapes, prepared_affine_model_shapes, prepared_bilinear_model_shapes
from .calibration import PreparedConditionalScale, PreparedResidualScale

MAXIMUM_COEFFICIENT_BYTES = 2 * 1024 * 1024


def prepared_affine_coefficient_shapes(structure: str) -> dict[str, tuple[int, ...]]:
    result = {f"model.{name}": shape for name, shape in prepared_affine_model_shapes(structure).items()}
    count = 200 if structure == "mechanism-i1" else 192
    result.update(
        {
            "scale.feature_mean": (count,),
            "scale.feature_scale": (count,),
            "scale.logvariance_operator": (count + 1, 315),
        }
    )
    return dict(sorted(result.items()))


def prepared_bilinear_coefficient_shapes(structure: str) -> dict[str, tuple[int, ...]]:
    shapes = prepared_affine_coefficient_shapes(structure)
    shapes.update({f"model.{name}": shape for name, shape in prepared_bilinear_model_shapes(structure).items()})
    return shapes


@dataclass(frozen=True, slots=True)
class PreparedAffinePredictorCoefficients(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/prepared-response/prepared-affine-predictor-coefficients'
    MODEL_SPEC: ClassVar[type[PreparedAffineModelSpec]] = PreparedAffineModelSpec
    model_spec: PreparedAffineModelSpec
    structure: str
    ridge: Decimal
    scale_positive_floor: Decimal
    blocks: tuple[tuple[str, str], ...]
    encoding: str = "BASE64_LITTLE_ENDIAN_FLOAT64_C_ORDER_EXACT_FIXED_SHAPES"

    def __post_init__(self) -> None:
        shapes = self._shapes()
        if (
            type(self.model_spec) is not self.MODEL_SPEC
            or self.encoding != "BASE64_LITTLE_ENDIAN_FLOAT64_C_ORDER_EXACT_FIXED_SHAPES"
            or self.ridge not in self.model_spec.ridges
            or not isinstance(self.ridge, Decimal)
            or not isinstance(self.scale_positive_floor, Decimal)
            or not self.scale_positive_floor.is_finite()
            or self.scale_positive_floor <= 0
            or type(self.blocks) is not tuple
            or any(type(block) is not tuple or len(block) != 2 for block in self.blocks)
            or tuple(name for name, _ in self.blocks) != tuple(shapes)
        ):
            raise ValueError(
                "coefficient product changes its numeric family, floor or block roster"
            )
        # This also checks every finite array and each strictly positive scale.
        self.predictors()
        if len(self.canonical_bytes()) > MAXIMUM_COEFFICIENT_BYTES:
            raise ValueError("coefficient product exceeds its complete two-MiB artifact bound")

    def _shapes(self) -> dict[str, tuple[int, ...]]:
        return (
            prepared_bilinear_coefficient_shapes(self.structure)
            if self.MODEL_SPEC is PreparedBilinearModelSpec
            else prepared_affine_coefficient_shapes(self.structure)
        )

    def arrays(self) -> dict[str, Array]:
        shapes = self._shapes()
        result: dict[str, Array] = {}
        for name, encoded in self.blocks:
            expected_bytes = int(np.prod(shapes[name])) * 8
            if not isinstance(encoded, str) or len(encoded) != 4 * ((expected_bytes + 2) // 3):
                raise ValueError("coefficient block has an unexpected encoded byte bound")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ValueError("coefficient block has a noncanonical base64 encoding") from exc
            if len(raw) != expected_bytes or base64.b64encode(raw).decode("ascii") != encoded:
                raise ValueError("coefficient block changes its exact bounded float encoding")
            result[name] = np.frombuffer(raw, dtype="<f8").reshape(shapes[name])
        return result

    def predictors(self) -> tuple[PreparedFinitePredictor, PreparedConditionalScale]:
        arrays = self.arrays()
        model = PreparedFinitePredictor(
            self.model_spec,
            self.structure,
            self.ridge,
            **{name: arrays[f"model.{name}"] for name in _model_shapes(self.model_spec, self.structure)},
        )
        scale = PreparedConditionalScale(
            float(self.scale_positive_floor),
            "I1" if self.structure == "mechanism-i1" else "I0",
            arrays["scale.feature_mean"],
            arrays["scale.feature_scale"],
            arrays["scale.logvariance_operator"],
        )
        return model, scale


@dataclass(frozen=True, slots=True)
class PreparedBilinearPredictorCoefficients(PreparedAffinePredictorCoefficients):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/prepared-response/prepared-bilinear-predictor-coefficients'
    MODEL_SPEC: ClassVar[type[PreparedAffineModelSpec]] = PreparedBilinearModelSpec
    model_spec: PreparedBilinearModelSpec


CoefficientT = TypeVar("CoefficientT", bound=PreparedAffinePredictorCoefficients)


def _coefficient_product(
    model: PreparedFittedModel, scale: PreparedResidualScale, record_type: type[CoefficientT],
) -> CoefficientT:
    if model.assigned_training_roots != scale.training_roots or scale.instrument_tier != (
        "I1" if model.structure == "mechanism-i1" else "I0"
    ):
        raise ValueError(
            "portable coefficients mix different dependent refinement training populations or input tiers"
        )
    predictor, conditional = model.deployment_predictor(), scale.deployment_scale()
    arrays = {
        f"model.{name}": getattr(predictor, name)
        for name in _model_shapes(model.spec, model.structure)
    }
    arrays.update(
        {
            f"scale.{name}": getattr(conditional, name)
            for name in ("feature_mean", "feature_scale", "logvariance_operator")
        }
    )
    blocks = tuple(
        (
            name,
            base64.b64encode(np.asarray(arrays[name], dtype="<f8").tobytes(order="C")).decode(
                "ascii"
            ),
        )
        for name in prepared_affine_coefficient_shapes(model.structure)
    )
    return record_type(
        model.spec,
        model.structure,
        model.ridge,
        Decimal(format(scale.positive_floor, ".17g")),
        blocks,
    )


def prepared_affine_predictor_coefficients(
    model: PreparedFittedModel, scale: PreparedResidualScale,
) -> PreparedAffinePredictorCoefficients:
    return _coefficient_product(model, scale, PreparedAffinePredictorCoefficients)


def prepared_bilinear_predictor_coefficients(
    model: PreparedFittedModel, scale: PreparedResidualScale,
) -> PreparedBilinearPredictorCoefficients:
    return _coefficient_product(model, scale, PreparedBilinearPredictorCoefficients)


def decode_prepared_affine_predictor_coefficients(payload: bytes) -> PreparedAffinePredictorCoefficients:
    return decode_canonical_bytes(
        payload, PreparedAffinePredictorCoefficients, maximum_bytes=MAXIMUM_COEFFICIENT_BYTES
    )


def decode_prepared_bilinear_predictor_coefficients(payload: bytes) -> PreparedBilinearPredictorCoefficients:
    return decode_canonical_bytes(
        payload, PreparedBilinearPredictorCoefficients, maximum_bytes=MAXIMUM_COEFFICIENT_BYTES
    )
