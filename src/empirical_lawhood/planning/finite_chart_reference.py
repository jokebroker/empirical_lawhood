"""Sealed finite-chart disposition-reference authoring."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class NativeMaterialEquivalenceSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-material-equivalence-spec'

    equivalence_id: str
    quantity_id: str
    native_unit: str
    absolute_tolerance: NamedDecimal
    relative_tolerance: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.equivalence_id, field_name="equivalence_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        if self.absolute_tolerance.unit != self.native_unit:
            raise ValueError("material-equivalence absolute tolerance uses another unit")
        validate_decimal(
            self.absolute_tolerance.value,
            field_name="absolute_tolerance",
            minimum=Decimal(0),
        )
        validate_decimal(
            self.relative_tolerance,
            field_name="relative_tolerance",
            minimum=Decimal(0),
        )


class ReferenceComparisonKind(StrEnum):
    TARGET = "TARGET"
    EFFORT = "EFFORT"
    UNCERTAINTY = "UNCERTAINTY"
    EXACT_CLASS = "EXACT_CLASS"


class ReferenceComparisonDirection(StrEnum):
    MAXIMIZE = "MAXIMIZE"
    MINIMIZE = "MINIMIZE"


class ReferenceComparisonReducer(StrEnum):
    MINIMUM = "MINIMUM"
    MAXIMUM = "MAXIMUM"


class ReferenceMissingnessRule(StrEnum):
    EXCLUDE_ACTION = "EXCLUDE_ACTION"


@dataclass(frozen=True, slots=True)
class ReferenceComparisonDimensionSpec(CanonicalRecord):
    """One ordered native-unit dimension of a set-valued reference class."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/reference-comparison-dimension-spec'

    dimension_id: str
    order: int
    kind: ReferenceComparisonKind
    native_unit: str
    direction: ReferenceComparisonDirection
    reducer: ReferenceComparisonReducer
    native_scale: NamedDecimal
    absolute_tolerance: NamedDecimal
    relative_tolerance: Decimal
    missingness_rule: ReferenceMissingnessRule

    def __post_init__(self) -> None:
        validate_stable_id(self.dimension_id, field_name="dimension_id")
        if self.order < 1:
            raise ValueError("reference comparison order must be positive")
        if not self.native_unit:
            raise ValueError("reference comparison native unit is empty")
        if (
            self.native_scale.unit != self.native_unit
            or self.absolute_tolerance.unit != self.native_unit
        ):
            raise ValueError("reference comparison scale/tolerance uses another unit")
        validate_decimal(
            self.native_scale.value,
            field_name="native_scale",
            minimum=Decimal(0),
        )
        if self.native_scale.value <= 0:
            raise ValueError("reference comparison native scale must be positive")
        validate_decimal(
            self.absolute_tolerance.value,
            field_name="absolute_tolerance",
            minimum=Decimal(0),
        )
        validate_decimal(
            self.relative_tolerance,
            field_name="relative_tolerance",
            minimum=Decimal(0),
        )
        if self.kind is ReferenceComparisonKind.EXACT_CLASS and (
            self.absolute_tolerance.value != 0 or self.relative_tolerance != 0
        ):
            raise ValueError("exact reference classes cannot use tolerance")


@dataclass(frozen=True, slots=True)
class FiniteChartReferenceDesign(CanonicalRecord):
    """Controller-independent complete finite-chart reference authoring."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-chart-reference-design'

    reference_design_id: str
    independent_unit_id: str
    action_words: tuple[OccurrenceActionWord, ...]
    model_member_ids: tuple[str, ...]
    reference_seed_ids: tuple[str, ...]
    comparison_dimensions: tuple[ReferenceComparisonDimensionSpec, ...]
    required_gate_roles: tuple[AdmissionGateKind, ...]
    action_priority_ids: tuple[str, ...]
    measured_hold_word_id: str | None
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.reference_design_id, field_name="reference_design_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        require_sorted_unique_strings(
            self.model_member_ids,
            field_name="model_member_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.reference_seed_ids,
            field_name="reference_seed_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.comparison_dimensions,
            attribute="dimension_id",
            field_name="comparison_dimensions",
        )
        if not self.action_words or not self.comparison_dimensions:
            raise ValueError("prospective finite-chart reference requires actions and comparison dimensions")
        orders = tuple(value.order for value in self.comparison_dimensions)
        if tuple(sorted(orders)) != tuple(range(1, len(orders) + 1)):
            raise ValueError("reference comparison dimension order is not contiguous")
        expected_gates = tuple(sorted(AdmissionGateKind, key=lambda value: value.value))
        if self.required_gate_roles != expected_gates:
            raise ValueError("member-local reference must require the exact nine admission gate roles")
        action_ids = {value.word_id for value in self.action_words}
        if (
            len(self.action_priority_ids) != len(action_ids)
            or set(self.action_priority_ids) != action_ids
        ):
            raise ValueError("reference action priority must cover the exact chart")
        if self.measured_hold_word_id is not None:
            validate_stable_id(
                self.measured_hold_word_id,
                field_name="measured_hold_word_id",
            )
            if self.measured_hold_word_id not in action_ids:
                raise ValueError("reference measured HOLD is outside the chart")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("prospective finite-chart reference must remain evaluation-sealed")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("prospective finite-chart reference must remain prospective")


@dataclass(frozen=True, slots=True)
class FiniteChartPanelCoordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-chart-panel-coordinate'

    panel_id: str
    independent_unit_id: str
    seed_id: str
    model_member_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("panel_id", self.panel_id),
            ("independent_unit_id", self.independent_unit_id),
            ("seed_id", self.seed_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class FiniteChartReferencePlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-chart-reference-plan'

    reference_plan_id: str
    controller_study: ObjectIdentity
    action_words: tuple[OccurrenceActionWord, ...]
    panel: tuple[FiniteChartPanelCoordinate, ...]
    material_equivalence: tuple[NativeMaterialEquivalenceSpec, ...]
    canonical_disposition_priority: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.reference_plan_id, field_name="reference_plan_id")
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        require_sorted_unique_ids(self.panel, attribute="panel_id", field_name="panel")
        require_sorted_unique_ids(
            self.material_equivalence,
            attribute="equivalence_id",
            field_name="material_equivalence",
        )
        if not self.canonical_disposition_priority or len(
            set(self.canonical_disposition_priority)
        ) != len(self.canonical_disposition_priority):
            raise ValueError("finite-chart disposition priority must be unique")
        if not self.action_words or not self.panel:
            raise ValueError("finite-chart reference requires actions and a panel")
        if self.canonical_disposition_priority != (
            "UNSAFE",
            "UNEVALUABLE",
            "ACTION_COMMITTED",
            "MEASURED_HOLD_COMMITTED",
            "NONATTEMPT",
        ):
            raise ValueError("finite-chart reference priority differs from the frozen order")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("finite-chart reference must remain sealed")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("finite-chart reference cannot be outcome-visible")


__all__ = [
    'FiniteChartReferenceDesign',
    "FiniteChartPanelCoordinate",
    "FiniteChartReferencePlan",
    "NativeMaterialEquivalenceSpec",
    "ReferenceComparisonDimensionSpec",
    "ReferenceComparisonDirection",
    "ReferenceComparisonKind",
    "ReferenceComparisonReducer",
    "ReferenceMissingnessRule",
]
