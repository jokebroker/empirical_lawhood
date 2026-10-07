"Prospective controller use for one continuous feedback trajectory per independent root."

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
    validate_sha256,
    validate_semantic_version,
)
from .controller_study import ImplementationBinding, ImplementationRole


@dataclass(frozen=True, slots=True)
class TrajectoryReducerRegistration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/trajectory-reducer-registration'
    registration_id: str
    capability_key: str
    capability_version: str
    config_sha256: str
    implementation_sha256: str
    reduction_order: tuple[str, ...] = ("CALLBACK", "NUMERICAL_VIEW", "INDEPENDENT_ROOT")

    def __post_init__(self) -> None:
        validate_stable_id(self.registration_id, field_name="registration_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(self.config_sha256, field_name="config_sha256")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.reduction_order != ("CALLBACK", "NUMERICAL_VIEW", "INDEPENDENT_ROOT"):
            raise ValueError("trajectory reducer changed its noncompensating root reduction")


@dataclass(frozen=True, slots=True)
class TrajectoryControllerEvaluationPlan(CanonicalRecord):
    """No repeat/preparation/HOLD semantics are borrowed for callback children."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/trajectory-controller-evaluation-plan'
    evaluation_plan_id: str
    frozen_consumer: ObjectIdentity
    roots: tuple[str, ...]
    callback_count: int
    callback_seconds: Decimal
    numerical_views: tuple[str, ...]
    numerical_tolerances: tuple[NamedDecimal, ...]
    maximum_halfwidths: tuple[NamedDecimal, ...]
    path_upper: NamedDecimal
    final_lower: tuple[NamedDecimal, ...]
    maximum_completion_ratio: Decimal
    panel_mean_completion_ratio: Decimal
    evaluator: ImplementationBinding

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_plan_id, field_name="evaluation_plan_id")
        require_sorted_unique_strings(self.roots, field_name="roots", allow_empty=False)
        require_sorted_unique_strings(
            self.numerical_views, field_name="numerical_views", allow_empty=False
        )
        if self.frozen_consumer.object_schema != 'empirical-lawhood/planning/frozen-feedback-consumer':
            raise ValueError("trajectory controller use requires the frozen policy parent")
        if (
            type(self.callback_count) is not int
            or not 1 <= self.callback_count <= 10000
            or self.callback_seconds <= 0
        ):
            raise ValueError("trajectory callback census/clock is invalid")
        if (
            len(self.numerical_views) != 2
            or self.evaluator.role is not ImplementationRole.OUTCOME_EVALUATOR
        ):
            raise ValueError("trajectory requires paired numerical views and existing controller-use owner")
        for values in (self.numerical_tolerances, self.maximum_halfwidths, self.final_lower):
            if not values or len({v.value_id for v in values}) != len(values):
                raise ValueError("trajectory receiver predicates must be complete and unique")
        if any(v.value < 0 for v in (*self.numerical_tolerances, *self.maximum_halfwidths)):
            raise ValueError("negative numerical or sharpness tolerance")
        if not 0 < self.panel_mean_completion_ratio <= self.maximum_completion_ratio:
            raise ValueError("invalid trajectory completion-ratio thresholds")
