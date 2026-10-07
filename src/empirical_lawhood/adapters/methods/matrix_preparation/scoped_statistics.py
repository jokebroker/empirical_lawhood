"""The explicitly allocated eight-bound family for matrix-preparation-adequacy.

The frozen 64-slot family is unchanged. This scoped contract reuses its
bounded-mean and exact-binomial constructions with the new explicit alpha.
"""

from dataclasses import dataclass
from decimal import Decimal
from math import log, sqrt
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord
from .statistics import exact_binomial_upper


@dataclass(frozen=True, slots=True)
class PreparationEightBoundStatistics(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-eight-bound-statistics'
    family_id: str = "matrix-preparation-adequacy.eight-bounds"
    slots: tuple[str, ...] = ("adequacy-benefit-against-hold", "adequacy-benefit-against-best-fixed-parent", "task-success-benefit-against-hold", "task-success-benefit-against-best-fixed-parent", "full-menu-interval-coverage", "certified-parent-unusable-rate", "all-root-selected-parent-usability", "paired-union-preservation-harm")
    alpha: Decimal = Decimal("0.05")
    alpha_per_slot: Decimal = Decimal("0.00625")
    context: str = "assembling"
    calibration_roots: int = 128
    efficacy_roots: int = 1280
    minimum_certified_roots: int = 640
    calibration_order_statistic: int = 123
    minimum_benefit: Decimal = Decimal("0.05")
    minimum_coverage: Decimal = Decimal("0.90")
    maximum_false_certification: Decimal = Decimal("0.10")
    maximum_union_preservation_harm: Decimal = Decimal("0.01")
    independent_unit: str = "FRESH_ROOT"
    preservation_rule: str = (
        "ANY_HARM_OR_UNRESOLVED_PAIR_ACROSS_COMPARATOR_ROLES_AUDITS_TASKS_AND_VIEWS"
    )
    grants_authority: bool = False

    def __post_init__(self) -> None:
        expected = {
            "family_id": "matrix-preparation-adequacy.eight-bounds",
            "slots": ("adequacy-benefit-against-hold", "adequacy-benefit-against-best-fixed-parent", "task-success-benefit-against-hold", "task-success-benefit-against-best-fixed-parent", "full-menu-interval-coverage", "certified-parent-unusable-rate", "all-root-selected-parent-usability", "paired-union-preservation-harm"),
            "alpha": Decimal("0.05"),
            "alpha_per_slot": Decimal("0.00625"),
            "context": "assembling",
            "calibration_roots": 128,
            "efficacy_roots": 1280,
            "minimum_certified_roots": 640,
            "calibration_order_statistic": 123,
            "minimum_benefit": Decimal("0.05"),
            "minimum_coverage": Decimal("0.90"),
            "maximum_false_certification": Decimal("0.10"),
            "maximum_union_preservation_harm": Decimal("0.01"),
            "independent_unit": "FRESH_ROOT",
            "preservation_rule": "ANY_HARM_OR_UNRESOLVED_PAIR_ACROSS_COMPARATOR_ROLES_AUDITS_TASKS_AND_VIEWS",
            "grants_authority": False,
        }
        if any(
            type(getattr(self, key)) is not type(value) or getattr(self, key) != value
            for key, value in expected.items()
        ):
            raise ValueError(
                "preparation eight-bound family changes its fixed estimands, alpha or population"
            )


def plan_eight_bound_preparation_precision() -> dict[str, object]:
    family = PreparationEightBoundStatistics()
    alpha = float(family.alpha_per_slot)
    rate_deduction = sqrt(log(1 / alpha) / (2 * family.efficacy_roots))
    return {
        "family": family.to_document(),
        "family_sha256": family.fingerprint(),
        "selected_roots_per_context": family.efficacy_roots,
        "paired_lower_bound_deduction": 2 * rate_deduction,
        "all_root_rate_deduction": rate_deduction,
        "minimum_certified_rate_deduction": sqrt(
            log(1 / alpha) / (2 * family.minimum_certified_roots)
        ),
        "union_harm_upper_zero": exact_binomial_upper(0, family.efficacy_roots, alpha),
        "union_harm_upper_four": exact_binomial_upper(4, family.efficacy_roots, alpha),
        "union_harm_upper_five": exact_binomial_upper(5, family.efficacy_roots, alpha),
        "joint_power_assurance": None,
        "scientific_evidence": False,
    }
