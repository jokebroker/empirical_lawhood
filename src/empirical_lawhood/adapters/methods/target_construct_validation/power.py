"""Target-neutral complete-unit power and simultaneous-family contract."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)


class TargetConstructValidationCompleteUnitMethod(StrEnum):
    EXACT_FINITE_PANEL = "EXACT_FINITE_PANEL"
    CLUSTER_BOOTSTRAP = "CLUSTER_BOOTSTRAP"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationPowerFreeze(CanonicalRecord):
    """Power/design record frozen before evaluation issue."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-power-freeze'

    freeze_id: str
    target_id: str
    complete_unit_role: str
    resampling_unit_role: str
    primary_estimand_ids: tuple[str, ...]
    decisive_falsifier_ids: tuple[str, ...]
    simultaneous_family_ids: tuple[str, ...]
    smallest_effect_worth_distinguishing: Decimal
    expected_adverse_or_missing_rate: Decimal
    dependence_block_ids: tuple[str, ...]
    cluster_hierarchy_ids: tuple[str, ...]
    method: TargetConstructValidationCompleteUnitMethod
    target_local_estimator_id: str
    simultaneous_error_control: str
    development_complete_unit_ids: tuple[str, ...]
    evaluation_complete_unit_ids: tuple[str, ...]
    reserve_complete_unit_ids: tuple[str, ...]
    joint_precision_passed: bool
    panel_limited_units_retained: bool
    nested_rows_or_views_inflate_replication: bool
    cross_target_pooling_allowed: bool
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("freeze_id", "target_id", "target_local_estimator_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "complete_unit_role",
            "resampling_unit_role",
            "simultaneous_error_control",
        ):
            validate_nonempty(getattr(self, name), field_name=name)
        for name in (
            "primary_estimand_ids",
            "decisive_falsifier_ids",
            "simultaneous_family_ids",
            "dependence_block_ids",
            "cluster_hierarchy_ids",
            "development_complete_unit_ids",
            "evaluation_complete_unit_ids",
            "reserve_complete_unit_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name
                in {
                    "reserve_complete_unit_ids",
                    "dependence_block_ids",
                    "cluster_hierarchy_ids",
                },
            )
        validate_decimal(
            self.smallest_effect_worth_distinguishing,
            field_name="smallest_effect_worth_distinguishing",
            minimum=Decimal("0"),
        )
        validate_decimal(
            self.expected_adverse_or_missing_rate,
            field_name="expected_adverse_or_missing_rate",
            minimum=Decimal("0"),
        )
        if self.smallest_effect_worth_distinguishing == 0:
            raise ValueError("the smallest effect worth distinguishing must be positive")
        if self.expected_adverse_or_missing_rate > 1:
            raise ValueError("expected adverse/missing rate cannot exceed one")
        development = set(self.development_complete_unit_ids)
        evaluation = set(self.evaluation_complete_unit_ids)
        reserve = set(self.reserve_complete_unit_ids)
        if development & evaluation or development & reserve or evaluation & reserve:
            raise ValueError("development/evaluation/reserve units must be disjoint")
        if not self.panel_limited_units_retained:
            raise ValueError("panel-limited units must remain in the issued denominator")
        if self.nested_rows_or_views_inflate_replication:
            raise ValueError("nested rows/views cannot inflate complete-unit replication")
        if self.cross_target_pooling_allowed:
            raise ValueError("target power cannot pool across substrates")
        if self.evaluation_outcome_count:
            raise ValueError("power freeze cannot use evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("power confirmation must remain development-only")


def require_adequate_power(power: TargetConstructValidationPowerFreeze) -> TargetConstructValidationPowerFreeze:
    if not power.joint_precision_passed:
        raise ValueError("POWER_INADEQUATE")
    return power
