"""All-assigned C law/preparation and five-comparison reduction records."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .calibration_pipeline import RegimeCalibrationPackage
from .config import ROOTS
from .confirmation_pipeline import ConfirmationRow, selected_operand, validate_confirmation_rows
from .information_math import CONTRASTS, ContrastResult, RootComparisonLosses, reduce_contrasts
from .model_records import RegimeFitPackage
from .nomination_records import RegimeNominationPackage
from .qualification_comparisons import root_comparison_losses
from .qualification_math import RootCoverage, Qualification, qualification_root, qualify


def _number(value: float | None) -> D | None:
    return None if value is None else D(repr(value))


@dataclass(frozen=True, slots=True)
class QualificationRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/qualification-root'

    root: str
    contacted_valid: bool
    interval_covered: bool
    law_adequate: bool
    preparation_safe: bool
    joined: bool
    reasons: tuple[str, ...]

    @classmethod
    def from_reduction(cls, value: RootCoverage) -> QualificationRoot:
        return cls(
            value.root, value.contacted_valid, value.interval_covered,
            value.law_adequate, value.preparation_safe, value.joined, value.reasons,
        )


@dataclass(frozen=True, slots=True)
class ComparisonRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/comparison-root'

    root: str
    contacts: tuple[bool, bool, bool, bool]
    source_validity: tuple[bool, bool, bool, bool, bool]
    sealed_before_labels: bool
    losses_K2: tuple[D | None, ...]

    @classmethod
    def from_reduction(cls, value: RootComparisonLosses) -> ComparisonRoot:
        return cls(
            value.root,
            (
                value.complete_baseline_contact,
                value.common_anchor_preparation_pair_contact,
                value.p_at_q_contact,
                bool(value.probe_preparation_common_anchor_contact),
            ),
            value.source_valid_by_contrast or (value.source_valid,) * 5,
            value.sealed_before_labels,
            tuple(
                _number(getattr(value, name)) for name in (
                    "r_rival_K2", "r_local_K2", "h_current_K2", "h_history_K2",
                    "constant_preparation_action_only_common_anchor_loss_K2", "constant_preparation_full_readout_common_anchor_loss_K2", "probe_preparation_action_only_common_anchor_loss_K2", "probe_preparation_full_readout_common_anchor_loss_K2",
                    "p_masked_q_K2", "p_full_q_K2",
                )
            ),
        )

    def __post_init__(self) -> None:
        if len(self.losses_K2) != 10 or any(
            value is not None and (not value.is_finite() or value < 0)
            for value in self.losses_K2
        ):
            raise ValueError("C comparison root lost a finite ten-arm loss census")


@dataclass(frozen=True, slots=True)
class ReactorRegimeContrastResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/reactor-regime-contrast-result'

    contrast_id: str
    assigned_roots: int
    complete_contact_roots: int
    model_failure_roots: int
    mean_advantage_K2: D | None
    lower_99_one_sided_K2: D | None
    upper_99_one_sided_K2: D | None
    arm_rmse_K: tuple[D, ...]
    arm_loss_ratio: D | None
    verdict: str
    reasons: tuple[str, ...]

    @classmethod
    def from_reduction(cls, value: ContrastResult) -> ReactorRegimeContrastResult:
        return cls(
            value.contrast_id,
            value.assigned_roots,
            value.complete_contact_roots,
            value.model_failure_roots,
            _number(value.mean_advantage_K2),
            _number(value.lower_99_one_sided_K2),
            _number(value.upper_99_one_sided_K2),
            tuple(D(repr(x)) for x in value.arm_rmse_K),
            _number(value.arm_loss_ratio),
            value.verdict,
            value.reasons,
        )


@dataclass(frozen=True, slots=True)
class RegimeQualificationPackage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-qualification-package'

    package_id: str
    fit_package: ObjectIdentity
    nomination: ObjectIdentity
    calibration: ObjectIdentity
    qualification_causal: tuple[ObjectIdentity, ...]
    qualification_assays: tuple[ObjectIdentity, ...]
    roots: tuple[QualificationRoot, ...]
    comparison_roots: tuple[ComparisonRoot, ...]
    contrasts: tuple[ReactorRegimeContrastResult, ...]
    law_successes: int
    joint_successes: int
    law_lower_95: D
    joint_lower_95: D
    law_pass: bool
    release_D: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = tuple(root for root, role, _, _ in ROOTS if role == "qualification")
        if (
            self.package_id != "reactor-regime-response-qualification-package"
            or self.fit_package.object_schema != RegimeFitPackage.SCHEMA
            or self.nomination.object_schema != RegimeNominationPackage.SCHEMA
            or self.calibration.object_schema != RegimeCalibrationPackage.SCHEMA
            or tuple(value.object_id for value in self.qualification_causal)
            != tuple(f"{root}.causal-preparation" for root in expected)
            or tuple(value.object_id for value in self.qualification_assays)
            != tuple(f"{root}.assay-panel" for root in expected)
            or tuple(value.root for value in self.roots) != expected
            or tuple(value.root for value in self.comparison_roots) != expected
            or tuple(value.contrast_id for value in self.contrasts) != CONTRASTS
            or self.law_successes != sum(value.law_adequate for value in self.roots)
            or self.joint_successes != sum(value.joined for value in self.roots)
            or not self.law_lower_95.is_finite()
            or not self.joint_lower_95.is_finite()
        ):
            raise ValueError("qualification lost its 64-root intersection or five contrasts")


def build_qualification_package(
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    calibration: RegimeCalibrationPackage,
    rows: tuple[ConfirmationRow, ...],
) -> RegimeQualificationPackage:
    validate_confirmation_rows("qualification", fit, nomination, rows)
    if (
        calibration.fit_package != ObjectIdentity.from_record(fit.package_id, fit)
        or calibration.nomination != ObjectIdentity.from_record(nomination.package_id, nomination)
    ):
        raise ValueError("qualification changed its frozen calibration ancestry")
    q_t = calibration.q_temperature
    q_c = calibration.q_cooling
    coverage = []
    comparisons = []
    for row in rows:
        if q_t is None or q_c is None or nomination.selected_route is None:
            coverage.append(
                RootCoverage(
                    row[0].root, False, False, False, False, False,
                    ("NO_FROZEN_CALIBRATED_PRIMARY_LAW",),
                )
            )
        else:
            operand = selected_operand(
                row, nomination, q_temperature=float(q_t), q_cooling=float(q_c)
            )
            if operand is None:
                raise ValueError("qualification lost its selected C route")
            coverage.append(qualification_root(operand, calibration.numerical_calibration()))
        comparisons.append(root_comparison_losses(row, fit, nomination))
    root_values = tuple(coverage)
    comparison_values = tuple(comparisons)
    law: Qualification = qualify(root_values, calibration.numerical_calibration())
    contrast_values = reduce_contrasts(comparison_values)
    return RegimeQualificationPackage(
        "reactor-regime-response-qualification-package",
        ObjectIdentity.from_record(fit.package_id, fit),
        ObjectIdentity.from_record(nomination.package_id, nomination),
        ObjectIdentity.from_record(calibration.package_id, calibration),
        tuple(
            ObjectIdentity.from_record(f"{causal.root}.causal-preparation", causal)
            for causal, _, _, _ in rows
        ),
        tuple(
            ObjectIdentity.from_record(f"{assay.root}.assay-panel", assay)
            for _, _, _, assay in rows
        ),
        tuple(QualificationRoot.from_reduction(value) for value in root_values),
        tuple(ComparisonRoot.from_reduction(value) for value in comparison_values),
        tuple(ReactorRegimeContrastResult.from_reduction(value) for value in contrast_values),
        law.law_successes,
        law.joint_successes,
        D(repr(law.law_lower_95)),
        D(repr(law.joint_lower_95)),
        law.law_pass,
        law.release_D,
        law.reasons,
    )
