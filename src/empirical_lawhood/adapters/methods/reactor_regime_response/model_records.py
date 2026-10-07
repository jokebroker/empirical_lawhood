"""Canonical fit-root package sealed before any nomination assay."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .config import ROOTS, ReactorRegimeResponseDesign
from .frozen_models import FrozenFit
from .information_models import FittedInformationSpec, MatchedSmoothSpec
from .model_selection import Candidate


@dataclass(frozen=True, slots=True)
class FittedCandidateRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/fitted-candidate-record'

    model_id: str
    mask: str
    family: str
    penalty: D
    multiplier: D | None
    fitted: FrozenFit | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.mask not in ("current", "history")
            or self.family not in ("K", "affine", "rbf", "local")
            or not self.penalty.is_finite()
            or self.penalty < 0
            or (self.multiplier is not None and (not self.multiplier.is_finite() or self.multiplier <= 0))
            or (self.fitted is None and not self.reasons)
            or (self.fitted is not None and self.fitted.kind != self.family)
            or tuple(sorted(set(self.reasons))) != self.reasons
        ):
            raise ValueError("fitted candidate structure or failure census differs")

    @classmethod
    def from_candidate(cls, candidate: Candidate) -> FittedCandidateRecord:
        fitted = None
        if candidate.fitted is not None:
            if candidate.support_lower is None or candidate.support_upper is None:
                raise ValueError("fitted candidate lacks its frozen support box")
            fitted = FrozenFit.from_fit(
                candidate.model_id.lower(),
                candidate.fitted,
                candidate.support_lower,
                candidate.support_upper,
            )
        return cls(
            candidate.model_id,
            candidate.mask,
            candidate.family,
            D(repr(candidate.penalty)),
            None if candidate.multiplier is None else D(repr(candidate.multiplier)),
            fitted,
            candidate.reasons,
        )

    def to_candidate(self) -> Candidate:
        return Candidate(
            self.model_id,
            self.mask,
            self.family,
            float(self.penalty),
            None if self.multiplier is None else float(self.multiplier),
            None if self.fitted is None else self.fitted.to_fit(),
            None if self.fitted is None else np.asarray(tuple(float(x) for x in self.fitted.support_lower)),
            None if self.fitted is None else np.asarray(tuple(float(x) for x in self.fitted.support_upper)),
            {},
            None,
            (),
            self.reasons,
        )


@dataclass(frozen=True, slots=True)
class FittedInformationSpecRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/fitted-information-spec-record'

    pair_id: str
    family: str
    penalty: D
    multiplier: D | None
    arms: tuple[str, ...]
    fits: tuple[FrozenFit, ...]

    def __post_init__(self) -> None:
        if (
            self.pair_id not in ("S_H", "S_I", "S_Q")
            or self.family not in ("affine", "rbf")
            or len(self.arms) != len(self.fits)
            or not self.fits
            or any(fit.kind != self.family for fit in self.fits)
        ):
            raise ValueError("matched information fit record differs")

    @classmethod
    def from_fitted(cls, item: FittedInformationSpec) -> FittedInformationSpecRecord:
        return cls(
            item.pair_id,
            item.spec.family,
            D(repr(item.spec.penalty)),
            None if item.spec.multiplier is None else D(repr(item.spec.multiplier)),
            item.arms,
            tuple(
                FrozenFit.from_fit(
                    f"{item.pair_id.lower()}.{item.spec.spec_id}.{arm}".replace("_", "-"),
                    fit,
                    lower,
                    upper,
                )
                for arm, fit, lower, upper in zip(
                    item.arms, item.fits, item.support_lower, item.support_upper, strict=True
                )
            ),
        )

    def to_fitted(self) -> FittedInformationSpec:
        return FittedInformationSpec(
            self.pair_id,
            MatchedSmoothSpec(
                self.family,
                float(self.penalty),
                None if self.multiplier is None else float(self.multiplier),
            ),
            self.arms,
            tuple(fit.to_fit() for fit in self.fits),
            tuple(np.asarray(tuple(float(x) for x in fit.support_lower)) for fit in self.fits),
            tuple(np.asarray(tuple(float(x) for x in fit.support_upper)) for fit in self.fits),
        )


@dataclass(frozen=True, slots=True)
class RegimeFitPackage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-fit-package'

    package_id: str
    design: ObjectIdentity
    fit_causal: tuple[ObjectIdentity, ...]
    fit_assays: tuple[ObjectIdentity, ...]
    coefficients: tuple[FittedCandidateRecord, ...]
    absolute: tuple[FittedCandidateRecord, ...]
    information: tuple[FittedInformationSpecRecord, ...]
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.package_id, field_name="package_id")
        design = ReactorRegimeResponseDesign()
        roots = tuple(root for root, role, _, _ in ROOTS if role == "fit")
        if (
            self.design != ObjectIdentity.from_record(design.config_id, design)
            or tuple(value.object_id for value in self.fit_causal)
            != tuple(f"{root}.causal-preparation" for root in roots)
            or tuple(value.object_id for value in self.fit_assays)
            != tuple(f"{root}.assay-panel" for root in roots)
            or len(self.coefficients) != 32
            or len(self.absolute) != 24
            or len({record.model_id for record in self.coefficients}) != 32
            or len({record.model_id for record in self.absolute}) != 24
            or tuple(sorted(set(self.failures))) != self.failures
        ):
            raise ValueError("fit package changed its complete fit-only roster or evidence")

    @classmethod
    def build(
        cls,
        causal: tuple[ObjectIdentity, ...],
        assays: tuple[ObjectIdentity, ...],
        coefficients: tuple[Candidate, ...],
        absolute: tuple[Candidate, ...],
        information: tuple[FittedInformationSpec, ...],
        failures: tuple[str, ...],
    ) -> RegimeFitPackage:
        design = ReactorRegimeResponseDesign()
        return cls(
            "reactor-regime-response-fit-package",
            ObjectIdentity.from_record(design.config_id, design),
            causal,
            assays,
            tuple(FittedCandidateRecord.from_candidate(item) for item in coefficients),
            tuple(FittedCandidateRecord.from_candidate(item) for item in absolute),
            tuple(FittedInformationSpecRecord.from_fitted(item) for item in information),
            tuple(sorted(set(failures))),
        )
