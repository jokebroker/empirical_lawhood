"""Authentic-input calibration operands, without terminal qualification truth."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt

from .assigned_native_records import FiniteResponseLawAssignedCalibrationNativeEvaluation
from .assigned_prediction import AssignedPrediction
from .calibration_analysis import CalibrationScores
from .calibration_panel import FIXED_CALIBRATION_ROOT_IDS, valid_calibration_root_ids
from .native_records import FiniteResponseLawCalibrationNativeEvaluation

SCORE_REASONS = (
    "MEASUREMENT_UNAVAILABLE",
    "NONFINITE_RESIDUAL",
    "NUMERICAL_DISCREPANCY",
    "OUTSIDE_SUPPORT",
)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawBoundaryCalibration(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-boundary-calibration'
    )
    NATIVE_EVALUATION_TYPE: ClassVar[type[CanonicalRecord]] = (
        FiniteResponseLawCalibrationNativeEvaluation
    )
    ASSIGNED_ROOTS: ClassVar[bool] = False
    calibration_id: str
    boundary: str
    development_manifest: ArtifactIdentity
    frozen_coefficients: ArtifactIdentity
    native_evaluation: ObjectIdentity
    native_task_receipt: ObjectIdentity
    prediction_artifact: ArtifactIdentity
    root_ids: tuple[str, ...]
    scores: tuple[Decimal | None, ...]
    score_reasons: tuple[tuple[str, ...], ...]
    order_index: int
    q: Decimal | None

    def __post_init__(self) -> None:
        validate_stable_id(self.calibration_id, field_name="calibration_id")
        if (
            self.boundary not in ("lower", "composed", "cached", "direct")
            or self.native_evaluation.object_schema
            != self.NATIVE_EVALUATION_TYPE.SCHEMA
            or self.native_task_receipt.object_schema != CanonicalTaskReceipt.SCHEMA
            or (
                not valid_calibration_root_ids(self.root_ids)
                or (self.root_ids == FIXED_CALIBRATION_ROOT_IDS) == self.ASSIGNED_ROOTS
            )
            or len(self.scores) != 32
            or len(self.score_reasons) != 32
            or type(self.order_index) is not int
            or self.order_index != 30
        ):
            raise ValueError(
                "Finite response-law calibration changes its fresh independent-root census or source"
            )
        for value, reasons in zip(self.scores, self.score_reasons, strict=True):
            if (
                (value is None) != bool(reasons)
                or tuple(sorted(set(reasons))) != reasons
                or not set(reasons) <= set(SCORE_REASONS)
                or value is not None
                and (type(value) is not Decimal or not value.is_finite() or value < 0)
            ):
                raise ValueError(
                    "Finite response-law calibration score must be finite or explicitly unavailable"
                )
        finite = sorted(v for v in self.scores if v is not None)
        expected = finite[29] if len(finite) >= 30 else None
        if self.q != expected or (self.q is not None and type(self.q) is not Decimal):
            raise ValueError(
                "Finite response-law calibration multiplier differs from rank 30 including infinities"
            )

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.calibration_id, self)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedBoundaryCalibration(FiniteResponseLawBoundaryCalibration):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-boundary-calibration'
    )
    VERSION: ClassVar[str] = '1.0.0'
    NATIVE_EVALUATION_TYPE: ClassVar[type[CanonicalRecord]] = (
        FiniteResponseLawAssignedCalibrationNativeEvaluation
    )
    ASSIGNED_ROOTS: ClassVar[bool] = True


def boundary_calibration(
    values: CalibrationScores,
    prediction: AssignedPrediction,
    *,
    calibration_id: str,
    boundary: str,
    development_manifest: ArtifactIdentity,
    frozen_coefficients: ArtifactIdentity,
    native_evaluation: ObjectIdentity,
    native_task_receipt: ObjectIdentity,
    prediction_artifact: ArtifactIdentity,
    root_ids: tuple[str, ...] = FIXED_CALIBRATION_ROOT_IDS,
) -> FiniteResponseLawBoundaryCalibration:
    if values.root_scores.shape != (32,) or prediction.supported.shape != (32,):
        raise ValueError("Finite response-law calibration cannot discard or replace an assigned root")
    reasons = []
    for r, value in enumerate(values.root_scores):
        reason = []
        if not np.isfinite(value):
            if not values.measured[r]:
                reason.append("MEASUREMENT_UNAVAILABLE")
            if not values.numerical[r]:
                reason.append("NUMERICAL_DISCREPANCY")
            if not prediction.supported[r]:
                reason.append("OUTSIDE_SUPPORT")
            if not reason:
                reason.append("NONFINITE_RESIDUAL")
        reasons.append(tuple(sorted(reason)))
    if (
        native_evaluation.object_schema
        == FiniteResponseLawAssignedCalibrationNativeEvaluation.SCHEMA
    ):
        record_type = FiniteResponseLawAssignedBoundaryCalibration
    elif native_evaluation.object_schema == FiniteResponseLawCalibrationNativeEvaluation.SCHEMA:
        record_type = FiniteResponseLawBoundaryCalibration
    else:
        raise ValueError("Finite response-law calibration has an undeclared native completion")
    return record_type(
        calibration_id,
        boundary,
        development_manifest,
        frozen_coefficients,
        native_evaluation,
        native_task_receipt,
        prediction_artifact,
        root_ids,
        tuple(
            Decimal(format(float(v), ".17g")) if np.isfinite(v) else None
            for v in values.root_scores
        ),
        tuple(reasons),
        values.rank,
        Decimal(format(values.q, ".17g")) if np.isfinite(values.q) else None,
    )
