"The immutable selected joint-law operands for generic local law and online use."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .calibration_pipeline import RegimeCalibrationPackage
from .frozen_models import FrozenFit
from .model_records import RegimeFitPackage
from .nomination_records import RegimeNominationPackage
from .qualification_pipeline import RegimeQualificationPackage
from .route_nomination import ROUTES


@dataclass(frozen=True, slots=True)
class RegimeJointLawPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-joint-law-payload'

    fit_package: ObjectIdentity
    nomination: ObjectIdentity
    calibration: ObjectIdentity
    qualification: ObjectIdentity
    route: str | None
    coefficient_model_id: str | None
    absolute_model_id: str | None
    coefficient: FrozenFit | None
    absolute: FrozenFit | None
    q_temperature: D | None
    q_cooling: D | None

    def __post_init__(self) -> None:
        if (
            self.route not in (*ROUTES, None)
            or (self.route is None and any(value is not None for value in (
                self.coefficient_model_id, self.absolute_model_id, self.coefficient, self.absolute
            )))
            or self.fit_package.object_schema != RegimeFitPackage.SCHEMA
            or self.nomination.object_schema != RegimeNominationPackage.SCHEMA
            or self.calibration.object_schema != RegimeCalibrationPackage.SCHEMA
            or self.qualification.object_schema != RegimeQualificationPackage.SCHEMA
            or (self.coefficient is None) != (self.coefficient_model_id is None)
            or (self.absolute is None) != (self.absolute_model_id is None)
            or (self.q_temperature is None) != (self.q_cooling is None)
            or (self.coefficient is not None and self.coefficient_model_id is not None
                and self.coefficient.fit_id != self.coefficient_model_id.lower())
            or (self.absolute is not None and self.absolute_model_id is not None
                and self.absolute.fit_id != self.absolute_model_id.lower())
            or any(value is not None and (not value.is_finite() or value < 0) for value in (
                self.q_temperature, self.q_cooling
            ))
        ):
            raise ValueError("joint law payload changed selected route, model or bounds")


def joint_law_payload(
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    calibration: RegimeCalibrationPackage,
    qualification: RegimeQualificationPackage,
) -> RegimeJointLawPayload:
    if (
        nomination.fit_package != ObjectIdentity.from_record(fit.package_id, fit)
        or calibration.fit_package != nomination.fit_package
        or calibration.nomination != ObjectIdentity.from_record(nomination.package_id, nomination)
        or qualification.fit_package != nomination.fit_package
        or qualification.nomination != calibration.nomination
        or qualification.calibration != ObjectIdentity.from_record(calibration.package_id, calibration)
    ):
        raise ValueError("joint law payload changed its B/C causal ancestry")
    route = nomination.selected_route
    selected = None if route is None else next(value for value in nomination.routes if value.route == route)
    coefficients = {value.model_id: value for value in fit.coefficients}
    absolute = {value.model_id: value for value in fit.absolute}
    coefficient_id = None if selected is None else selected.coefficient_model_id
    absolute_id = None if selected is None else selected.absolute_model_id
    coefficient = None if coefficient_id is None else coefficients[coefficient_id].fitted
    baseline = None if absolute_id is None else absolute[absolute_id].fitted
    if (coefficient_id is not None and coefficient is None) or (absolute_id is not None and baseline is None):
        raise ValueError("nominated joint law lost its frozen fit")
    return RegimeJointLawPayload(
        ObjectIdentity.from_record(fit.package_id, fit),
        ObjectIdentity.from_record(nomination.package_id, nomination),
        ObjectIdentity.from_record(calibration.package_id, calibration),
        ObjectIdentity.from_record(qualification.package_id, qualification),
        route,
        coefficient_id,
        absolute_id,
        coefficient,
        baseline,
        calibration.q_temperature,
        calibration.q_cooling,
    )


@dataclass(frozen=True, slots=True)
class RegimeJointLawDecoder:
    extension_namespace: str = "tbs-reactor-regime-joint-response"
    decoder_schema: str = 'empirical-lawhood/methods/reactor-regime-response/joint-law-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = RegimeJointLawPayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, RegimeJointLawPayload, maximum_bytes=min(maximum_bytes, 4 * 1024**2)
        )
