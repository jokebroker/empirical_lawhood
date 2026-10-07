"Joint predictive uncertainty without an asserted latent decomposition.\n\nThis additive value does not reinterpret UncertaintyDecomposition. A registered\nqualification profile must explicitly choose this basis for its terminal gate.\n"

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from .obligations import ObligationStatus
from .provenance import ObjectIdentity
from .references import ArtifactIdentity, NamedDecimal
from .serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class JointPredictiveUncertainty(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/joint-predictive-uncertainty'

    assessment_id: str
    method_key: str
    calibration: ObjectIdentity | None
    scale_artifact: ArtifactIdentity
    independent_unit_id: str
    calibration_unit_ids: tuple[str, ...]
    simultaneous_scope_id: str
    interval_quantity_ids: tuple[str, ...]
    support_id: str
    confidence_level: Decimal
    bounds: tuple[NamedDecimal, ...]
    numerical_allowances: tuple[NamedDecimal, ...]
    assumption_ids: tuple[str, ...]
    status: ObligationStatus
    reason_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]
    coverage_scope: str = "marginal-independent-unit"
    scope_limitations: tuple[str, ...] = (
        "NO_AUTOMATIC_JOINT_COVERAGE_ACROSS_SEPARATE_BOUNDARIES",
        "NO_CONDITIONAL_COVERAGE_AT_EACH_INTERFACE",
        "NO_LATENT_UNCERTAINTY_COMPONENT_IDENTIFICATION",
        "NO_POST_SELECTION_OR_USABILITY_CONDITIONAL_GUARANTEE",
    )

    def __post_init__(self) -> None:
        for name in (
            "assessment_id",
            "method_key",
            "independent_unit_id",
            "simultaneous_scope_id",
            "support_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("calibration_unit_ids", "interval_quantity_ids", "assumption_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        for name in ("reason_codes", "evidence_link_ids", "scope_limitations"):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        for name in ("bounds", "numerical_allowances"):
            values = getattr(self, name)
            require_sorted_unique_ids(values, attribute="value_id", field_name=name)
            if any(v.value < 0 for v in values):
                raise ValueError("joint predictive uncertainty requires nonnegative bounds")
        if (
            type(self.confidence_level) is not Decimal
            or not self.confidence_level.is_finite()
            or not Decimal(0) < self.confidence_level < Decimal(1)
            or self.coverage_scope != "marginal-independent-unit"
            or self.scope_limitations
            != type(self).__dataclass_fields__["scope_limitations"].default
            or tuple(v.value_id for v in self.numerical_allowances) != self.interval_quantity_ids
        ):
            raise ValueError("joint predictive uncertainty changes its coverage or quantity scope")
        if self.status not in (
            ObligationStatus.SATISFIED,
            ObligationStatus.FAILED,
            ObligationStatus.UNEVALUABLE,
        ):
            raise ValueError("joint predictive uncertainty requires an assessed disposition")
        if self.status is ObligationStatus.SATISFIED:
            if (
                self.calibration is None
                or not self.bounds
                or not self.evidence_link_ids
                or self.reason_codes
            ):
                raise ValueError(
                    "satisfied joint uncertainty requires calibration and bounded evidence"
                )
        elif not self.reason_codes:
            raise ValueError("unresolved joint uncertainty requires explicit reasons")
        if self.status is ObligationStatus.FAILED and self.calibration is None:
            raise ValueError(
                "failed joint calibration requires an authenticated calibration record"
            )

    @property
    def claim_ready(self) -> bool:
        return self.status is ObligationStatus.SATISFIED
