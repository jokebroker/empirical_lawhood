"""Immutable C root maxima from the nominated law's sealed forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .config import ROOTS
from .confirmation_pipeline import ConfirmationRow, selected_operand, validate_confirmation_rows
from .model_records import RegimeFitPackage
from .nomination_records import RegimeNominationPackage
from .qualification_math import Calibration, RootPrecision, calibrate, root_precision


def _number(value: float | None) -> D | None:
    return None if value is None else D(repr(value))


@dataclass(frozen=True, slots=True)
class CalibrationRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/calibration-root'

    root: str
    contacted_valid: bool
    temperature_score: D | None
    cooling_score: D | None
    maximum_absolute_error_K: D | None
    maximum_contrast_error_K: D | None
    reasons: tuple[str, ...]

    @classmethod
    def from_score(cls, value: RootPrecision) -> CalibrationRoot:
        return cls(
            value.root,
            value.contacted_valid,
            _number(value.temperature_score),
            _number(value.cooling_score),
            _number(value.maximum_absolute_error_K),
            _number(value.maximum_contrast_error_K),
            value.reasons,
        )


@dataclass(frozen=True, slots=True)
class RegimeCalibrationPackage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-calibration-package'

    package_id: str
    fit_package: ObjectIdentity
    nomination: ObjectIdentity
    calibration_causal: tuple[ObjectIdentity, ...]
    calibration_assays: tuple[ObjectIdentity, ...]
    roots: tuple[CalibrationRoot, ...]
    contacted_roots: tuple[str, ...]
    q_temperature: D | None
    q_cooling: D | None
    precision_pass: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = tuple(root for root, role, _, _ in ROOTS if role == "calibration")
        if (
            self.package_id != "reactor-regime-response-calibration-package"
            or self.fit_package.object_schema != RegimeFitPackage.SCHEMA
            or self.nomination.object_schema != RegimeNominationPackage.SCHEMA
            or tuple(value.object_id for value in self.calibration_causal)
            != tuple(f"{root}.causal-preparation" for root in expected)
            or tuple(value.object_id for value in self.calibration_assays)
            != tuple(f"{root}.assay-panel" for root in expected)
            or tuple(row.root for row in self.roots) != expected
            or any(root not in expected for root in self.contacted_roots)
            or (self.precision_pass and (self.q_temperature is None or self.q_cooling is None))
            or any(
                value is not None and (not value.is_finite() or value < 0)
                for value in (self.q_temperature, self.q_cooling)
            )
        ):
            raise ValueError("calibration changed its all-assigned sealed C census")

    def numerical_calibration(self) -> Calibration:
        return Calibration(
            self.contacted_roots,
            None if self.q_temperature is None else float(self.q_temperature),
            None if self.q_cooling is None else float(self.q_cooling),
            self.precision_pass,
            self.reasons,
        )


def build_calibration_package(
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    rows: tuple[ConfirmationRow, ...],
) -> RegimeCalibrationPackage:
    validate_confirmation_rows("calibration", fit, nomination, rows)
    scores = []
    for row in rows:
        operand = selected_operand(row, nomination, q_temperature=0, q_cooling=0)
        scores.append(
            RootPrecision(
                row[0].root, False, None, None, None, None,
                ("NO_NOMINATED_PRIMARY_LAW",),
            )
            if operand is None else root_precision(operand)
        )
    reduced = calibrate(tuple(scores))
    reasons = tuple(sorted({*reduced.reasons, *(
        ("NO_NOMINATED_PRIMARY_LAW",) if nomination.selected_route is None else ()
    )}))
    return RegimeCalibrationPackage(
        "reactor-regime-response-calibration-package",
        ObjectIdentity.from_record(fit.package_id, fit),
        ObjectIdentity.from_record(nomination.package_id, nomination),
        tuple(
            ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal)
            for causal, _, _, _ in rows
        ),
        tuple(
            ObjectIdentity.from_record(f"{assay.root}.assay-panel", assay)
            for _, _, _, assay in rows
        ),
        tuple(CalibrationRoot.from_score(value) for value in scores),
        reduced.contacted_roots,
        None if reduced.q_temperature is None else D(repr(reduced.q_temperature)),
        None if reduced.q_cooling is None else D(repr(reduced.q_cooling)),
        reduced.precision_pass and nomination.selected_route is not None,
        reasons,
    )
