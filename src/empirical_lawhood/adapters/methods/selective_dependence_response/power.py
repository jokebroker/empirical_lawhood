"""Estimand-specific complete-unit design qualification for selective dependence response.

The design does not turn seven unlike questions into one standardized normal
effect.  Continuous minimum-effect, equivalence, paired-comparator and
zero-failure error-rate requirements retain their own decision rule and native
scale.  Bonferroni allocation is over the actual finite requirement roster,
including each separately declared active or invariant exchange.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from enum import StrEnum
from math import log
from statistics import NormalDist
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)

from .contracts import digest_ids


SELECTIVE_DEPENDENCE_RESPONSE_PRIMARY_FAMILY_IDS = (
    "active-exchange",
    "comparator-separation",
    "false-safe-hold",
    "invariant-exchange",
    "law-calibration",
    "support-boundary",
    "unsafe-false-admission",
)


class SelectiveDependenceResponsePowerMethod(StrEnum):
    ONE_SIDED_MINIMUM_MEAN = "ONE_SIDED_MINIMUM_MEAN"
    TWO_ONE_SIDED_EQUIVALENCE = "TWO_ONE_SIDED_EQUIVALENCE"
    TWO_SIDED_MINIMUM_ABSOLUTE_MEAN = "TWO_SIDED_MINIMUM_ABSOLUTE_MEAN"
    ZERO_FAILURE_UPPER_BOUND = "ZERO_FAILURE_UPPER_BOUND"


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponsePowerOperand(CanonicalRecord):
    """One development-derived operand for a frozen primary requirement."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-power-operand'

    requirement_id: str
    family_id: str
    method: SelectiveDependenceResponsePowerMethod
    expected_location: Decimal
    decision_boundary: Decimal
    development_standard_deviation: Decimal | None
    maximum_adverse_rate: Decimal | None
    development_complete_unit_count: int
    development_unit_ids_sha256: str
    native_unit: str

    def __post_init__(self) -> None:
        for name in ("requirement_id", "family_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.family_id not in SELECTIVE_DEPENDENCE_RESPONSE_PRIMARY_FAMILY_IDS:
            raise ValueError("power operand has an unknown primary family")
        for name in ("expected_location", "decision_boundary"):
            validate_decimal(getattr(self, name), field_name=name)
        if self.development_standard_deviation is not None:
            validate_decimal(
                self.development_standard_deviation,
                field_name="development_standard_deviation",
                minimum=Decimal(0),
            )
        if self.maximum_adverse_rate is not None:
            validate_decimal(
                self.maximum_adverse_rate,
                field_name="maximum_adverse_rate",
                minimum=Decimal(0),
            )
            if self.maximum_adverse_rate >= 1:
                raise ValueError("maximum adverse rate must be below one")
        if self.development_complete_unit_count < 0:
            raise ValueError("development complete-unit count must be nonnegative")
        if len(self.development_unit_ids_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.development_unit_ids_sha256
        ):
            raise ValueError("development unit roster digest must be lowercase SHA-256")
        validate_nonempty(self.native_unit, field_name="native_unit")
        zero_failure = self.method is SelectiveDependenceResponsePowerMethod.ZERO_FAILURE_UPPER_BOUND
        if zero_failure != (self.maximum_adverse_rate is not None):
            raise ValueError("only zero-failure requirements carry an adverse-rate bound")
        if zero_failure:
            if self.development_standard_deviation is not None:
                raise ValueError("zero-failure requirements do not use a normal standard deviation")
            if not Decimal(0) <= self.expected_location <= Decimal(1):
                raise ValueError("zero-failure development error rate must lie inside [0, 1]")
            if self.decision_boundary <= 0:
                raise ValueError("zero-failure design requires a positive error-rate bound")
            if self.maximum_adverse_rate != self.decision_boundary:
                raise ValueError("zero-failure boundary and adverse-rate bound differ")
        elif self.development_standard_deviation is None:
            raise ValueError("continuous power requirements need a development standard deviation")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponsePowerFamilyRequirement(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-power-family-requirement'

    requirement_id: str
    family_id: str
    method: SelectiveDependenceResponsePowerMethod
    expected_location: Decimal
    decision_boundary: Decimal
    favorable_gap: Decimal
    development_standard_deviation: Decimal | None
    maximum_adverse_rate: Decimal | None
    development_complete_unit_count: int
    development_unit_ids_sha256: str
    bonferroni_alpha: Decimal
    required_evaluable_complete_unit_count: int | None
    required_issued_complete_unit_count: int | None
    native_unit: str

    def __post_init__(self) -> None:
        for name in ("requirement_id", "family_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.family_id not in SELECTIVE_DEPENDENCE_RESPONSE_PRIMARY_FAMILY_IDS:
            raise ValueError("power requirement has an unknown primary family")
        for name in (
            "expected_location",
            "decision_boundary",
            "favorable_gap",
            "bonferroni_alpha",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if not Decimal(0) < self.bonferroni_alpha < Decimal(1):
            raise ValueError("per-requirement alpha must lie inside (0, 1)")
        if self.development_standard_deviation is not None:
            validate_decimal(
                self.development_standard_deviation,
                field_name="development_standard_deviation",
                minimum=Decimal(0),
            )
        if self.maximum_adverse_rate is not None:
            validate_decimal(
                self.maximum_adverse_rate,
                field_name="maximum_adverse_rate",
                minimum=Decimal(0),
            )
        if self.development_complete_unit_count < 0:
            raise ValueError("development complete-unit count must be nonnegative")
        if len(self.development_unit_ids_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.development_unit_ids_sha256
        ):
            raise ValueError("development unit roster digest must be lowercase SHA-256")
        if (self.required_evaluable_complete_unit_count is None) != (
            self.required_issued_complete_unit_count is None
        ):
            raise ValueError("evaluable and issued power counts must be jointly present")
        if self.required_evaluable_complete_unit_count is not None:
            if self.required_evaluable_complete_unit_count < 2:
                raise ValueError("power cannot qualify with fewer than two complete units")
            assert self.required_issued_complete_unit_count is not None
            if (
                self.required_issued_complete_unit_count
                < self.required_evaluable_complete_unit_count
            ):
                raise ValueError("issued power count cannot be below evaluable count")
            if self.favorable_gap <= 0:
                raise ValueError("an attainable requirement must have a positive favorable gap")
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponsePowerDesignQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-power-design-qualification'

    qualification_id: str
    target_id: str
    candidate_panel_sizes: tuple[int, ...]
    family_requirements: tuple[SelectiveDependenceResponsePowerFamilyRequirement, ...]
    family_ids: tuple[str, ...]
    familywise_alpha: Decimal
    target_power: Decimal
    maximum_nonevaluable_rate: Decimal
    selected_evaluation_unit_count: int | None
    attainable: bool
    resampling_unit: str
    nested_conditions_count_as_units: bool
    cross_target_pooling_allowed: bool
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("qualification_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if tuple(sorted(set(self.candidate_panel_sizes))) != self.candidate_panel_sizes:
            raise ValueError("candidate panel sizes must be sorted and unique")
        if not self.candidate_panel_sizes or self.candidate_panel_sizes[0] < 2:
            raise ValueError("candidate panel roster is invalid")
        require_sorted_unique_ids(
            self.family_requirements,
            attribute="requirement_id",
            field_name="family_requirements",
        )
        require_sorted_unique_strings(self.family_ids, field_name="family_ids", allow_empty=False)
        if self.family_ids != SELECTIVE_DEPENDENCE_RESPONSE_PRIMARY_FAMILY_IDS:
            raise ValueError("power qualification lacks the exact simultaneous family")
        if {value.family_id for value in self.family_requirements} != set(self.family_ids):
            raise ValueError("power requirements do not cover the exact family roster")
        for name in ("familywise_alpha", "target_power", "maximum_nonevaluable_rate"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if not Decimal(0) < self.familywise_alpha < Decimal(1):
            raise ValueError("familywise alpha must be inside (0, 1)")
        if not Decimal("0.5") < self.target_power < Decimal(1):
            raise ValueError("target power must be inside (0.5, 1)")
        if self.maximum_nonevaluable_rate >= 1:
            raise ValueError("nonevaluable rate must be below one")
        required_counts = tuple(
            value.required_issued_complete_unit_count for value in self.family_requirements
        )
        expected_selection = None
        if all(value is not None for value in required_counts):
            largest = max(int(value) for value in required_counts if value is not None)
            expected_selection = next(
                (value for value in self.candidate_panel_sizes if value >= largest),
                None,
            )
        if self.selected_evaluation_unit_count != expected_selection:
            raise ValueError("selected panel is not the smallest attainable roster member")
        if self.attainable != (expected_selection is not None):
            raise ValueError("power attainability is not computed from the finite roster")
        validate_nonempty(self.resampling_unit, field_name="resampling_unit")
        if self.nested_conditions_count_as_units or self.cross_target_pooling_allowed:
            raise ValueError("power must use target-local complete preparation units")
        if self.evaluation_outcome_count:
            raise ValueError("power qualification cannot use evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("power qualification must remain development visible")


def _favorable_gap(operand: SelectiveDependenceResponsePowerOperand) -> Decimal:
    if operand.method is SelectiveDependenceResponsePowerMethod.TWO_SIDED_MINIMUM_ABSOLUTE_MEAN:
        return abs(operand.expected_location) - operand.decision_boundary
    if operand.method is SelectiveDependenceResponsePowerMethod.TWO_ONE_SIDED_EQUIVALENCE:
        return operand.decision_boundary - abs(operand.expected_location)
    if operand.method is SelectiveDependenceResponsePowerMethod.ONE_SIDED_MINIMUM_MEAN:
        return operand.expected_location - operand.decision_boundary
    return operand.decision_boundary - operand.expected_location


def continuous_power_operand(
    *,
    requirement_id: str,
    family_id: str,
    method: SelectiveDependenceResponsePowerMethod,
    observations: tuple[NamedDecimal, ...],
    decision_boundary: Decimal,
    native_unit: str,
) -> SelectiveDependenceResponsePowerOperand:
    """Build one auditable mean/equivalence operand from complete-unit values."""

    if method is SelectiveDependenceResponsePowerMethod.ZERO_FAILURE_UPPER_BOUND:
        raise ValueError("continuous operand cannot use the zero-failure method")
    require_sorted_unique_ids(
        observations,
        attribute="value_id",
        field_name="observations",
    )
    if len(observations) < 2:
        raise ValueError("continuous power operand needs at least two complete units")
    values = tuple(value.value for value in observations)
    expected = sum(values, Decimal(0)) / Decimal(len(values))
    numeric = tuple(float(value) for value in values)
    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1)
    standard_deviation = Decimal(format(variance**0.5, ".12g"))
    ids = tuple(value.value_id for value in observations)
    return SelectiveDependenceResponsePowerOperand(
        requirement_id=requirement_id,
        family_id=family_id,
        method=method,
        expected_location=expected,
        decision_boundary=decision_boundary,
        development_standard_deviation=standard_deviation,
        maximum_adverse_rate=None,
        development_complete_unit_count=len(observations),
        development_unit_ids_sha256=digest_ids(ids),
        native_unit=native_unit,
    )


def zero_failure_power_operand(
    *,
    requirement_id: str,
    family_id: str,
    complete_unit_ids: tuple[str, ...],
    adverse_complete_unit_ids: tuple[str, ...],
    maximum_adverse_rate: Decimal,
) -> SelectiveDependenceResponsePowerOperand:
    """Build a unit-level zero-failure upper-bound requirement."""

    require_sorted_unique_strings(
        complete_unit_ids, field_name="complete_unit_ids", allow_empty=False
    )
    require_sorted_unique_strings(
        adverse_complete_unit_ids,
        field_name="adverse_complete_unit_ids",
    )
    if not set(adverse_complete_unit_ids).issubset(complete_unit_ids):
        raise ValueError("adverse power units lie outside the development roster")
    observed_rate = Decimal(len(adverse_complete_unit_ids)) / Decimal(len(complete_unit_ids))
    return SelectiveDependenceResponsePowerOperand(
        requirement_id=requirement_id,
        family_id=family_id,
        method=SelectiveDependenceResponsePowerMethod.ZERO_FAILURE_UPPER_BOUND,
        expected_location=observed_rate,
        decision_boundary=maximum_adverse_rate,
        development_standard_deviation=None,
        maximum_adverse_rate=maximum_adverse_rate,
        development_complete_unit_count=len(complete_unit_ids),
        development_unit_ids_sha256=digest_ids(complete_unit_ids),
        native_unit="complete-unit-adverse-rate",
    )


def _required_evaluable_count(
    *,
    operand: SelectiveDependenceResponsePowerOperand,
    alpha_each: float,
    target_power: float,
) -> int | None:
    gap = _favorable_gap(operand)
    if gap <= 0:
        return None
    if operand.method is SelectiveDependenceResponsePowerMethod.ZERO_FAILURE_UPPER_BOUND:
        if operand.expected_location != 0:
            return None
        assert operand.maximum_adverse_rate is not None
        raw_zero = log(alpha_each) / log(1 - float(operand.maximum_adverse_rate))
        return max(2, int(Decimal(str(raw_zero)).to_integral_value(rounding=ROUND_CEILING)))
    if operand.development_complete_unit_count < 2:
        return None
    assert operand.development_standard_deviation is not None
    if operand.development_standard_deviation == 0:
        return 2
    tail_alpha = (
        alpha_each / 2
        if operand.method is SelectiveDependenceResponsePowerMethod.TWO_SIDED_MINIMUM_ABSOLUTE_MEAN
        else alpha_each
    )
    z_alpha = NormalDist().inv_cdf(1 - tail_alpha)
    z_power = NormalDist().inv_cdf(target_power)
    raw_normal = (
        Decimal(str(z_alpha + z_power)) * operand.development_standard_deviation / gap
    ) ** 2
    return max(2, int(raw_normal.to_integral_value(rounding=ROUND_CEILING)))


def qualify_power_design(
    *,
    target_id: str,
    candidate_panel_sizes: tuple[int, ...],
    operands: tuple[SelectiveDependenceResponsePowerOperand, ...],
    familywise_alpha: Decimal = Decimal("0.05"),
    target_power: Decimal = Decimal("0.80"),
    maximum_nonevaluable_rate: Decimal = Decimal("0.10"),
) -> SelectiveDependenceResponsePowerDesignQualification:
    """Qualify the smallest finite panel across heterogeneous primary tests."""

    require_sorted_unique_ids(operands, attribute="requirement_id", field_name="operands")
    if {value.family_id for value in operands} != set(SELECTIVE_DEPENDENCE_RESPONSE_PRIMARY_FAMILY_IDS):
        raise ValueError("power operands do not cover the exact simultaneous family")
    validate_decimal(maximum_nonevaluable_rate, field_name="maximum_nonevaluable_rate")
    if not Decimal(0) <= maximum_nonevaluable_rate < Decimal(1):
        raise ValueError("maximum nonevaluable rate must lie inside [0, 1)")
    alpha_each_decimal = familywise_alpha / Decimal(len(operands))
    alpha_each = float(alpha_each_decimal)
    requirements = []
    for operand in operands:
        required_evaluable = _required_evaluable_count(
            operand=operand,
            alpha_each=alpha_each,
            target_power=float(target_power),
        )
        required_issued = (
            None
            if required_evaluable is None
            else int(
                (
                    Decimal(required_evaluable) / (Decimal(1) - maximum_nonevaluable_rate)
                ).to_integral_value(rounding=ROUND_CEILING)
            )
        )
        requirements.append(
            SelectiveDependenceResponsePowerFamilyRequirement(
                requirement_id=operand.requirement_id,
                family_id=operand.family_id,
                method=operand.method,
                expected_location=operand.expected_location,
                decision_boundary=operand.decision_boundary,
                favorable_gap=_favorable_gap(operand),
                development_standard_deviation=operand.development_standard_deviation,
                maximum_adverse_rate=operand.maximum_adverse_rate,
                development_complete_unit_count=operand.development_complete_unit_count,
                development_unit_ids_sha256=operand.development_unit_ids_sha256,
                bonferroni_alpha=alpha_each_decimal,
                required_evaluable_complete_unit_count=required_evaluable,
                required_issued_complete_unit_count=required_issued,
                native_unit=operand.native_unit,
            )
        )
    ordered = tuple(sorted(requirements, key=lambda value: value.requirement_id))
    required_counts = tuple(value.required_issued_complete_unit_count for value in ordered)
    selected = None
    if all(value is not None for value in required_counts):
        largest = max(int(value) for value in required_counts if value is not None)
        candidates = tuple(sorted(set(candidate_panel_sizes)))
        selected = next((value for value in candidates if value >= largest), None)
    else:
        candidates = tuple(sorted(set(candidate_panel_sizes)))
    return SelectiveDependenceResponsePowerDesignQualification(
        qualification_id=f"qualification.{target_id}.power",
        target_id=target_id,
        candidate_panel_sizes=candidates,
        family_requirements=ordered,
        family_ids=SELECTIVE_DEPENDENCE_RESPONSE_PRIMARY_FAMILY_IDS,
        familywise_alpha=familywise_alpha,
        target_power=target_power,
        maximum_nonevaluable_rate=maximum_nonevaluable_rate,
        selected_evaluation_unit_count=selected,
        attainable=selected is not None,
        resampling_unit="complete material preparation",
        nested_conditions_count_as_units=False,
        cross_target_pooling_allowed=False,
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


__all__ = [
    "SELECTIVE_DEPENDENCE_RESPONSE_PRIMARY_FAMILY_IDS",
    'SelectiveDependenceResponsePowerDesignQualification',
    'SelectiveDependenceResponsePowerFamilyRequirement',
    'SelectiveDependenceResponsePowerMethod',
    'SelectiveDependenceResponsePowerOperand',
    "continuous_power_operand",
    "qualify_power_design",
    "zero_failure_power_operand",
]
