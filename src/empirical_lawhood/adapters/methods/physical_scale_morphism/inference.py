"""Board-cluster simultaneous inference and structural-attainability gate."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from enum import StrEnum
from math import log
from statistics import NormalDist
from typing import ClassVar

from scipy.stats import beta

from empirical_lawhood.adapters.methods.complete_unit_inference import CompleteUnitInferenceConfig, CompleteUnitInferenceResult, CompleteUnitVector, complete_unit_simultaneous_inference
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_stable_id,
)


PHYSICAL_SCALE_MORPHISM_PANEL_SIZES = (8, 12, 16, 24)


class PhysicalScaleMorphismPowerMethod(StrEnum):
    CONTINUOUS_EQUIVALENCE = "CONTINUOUS_EQUIVALENCE"
    CONTINUOUS_MINIMUM_EFFECT = "CONTINUOUS_MINIMUM_EFFECT"
    ZERO_ADVERSE_RATE = "ZERO_ADVERSE_RATE"


class PhysicalScaleMorphismPowerTerminal(StrEnum):
    DEVELOPMENT_EFFECT_ON_WRONG_SIDE = "DEVELOPMENT_EFFECT_ON_WRONG_SIDE"
    METHOD_POWER_QUALIFIED = "METHOD_POWER_QUALIFIED"
    NO_FINITE_COUNT_UNDER_OBSERVED_RATE = "NO_FINITE_COUNT_UNDER_OBSERVED_RATE"
    PRECISION_LIMITED_AT_RESOURCE_CEILING = "PRECISION_LIMITED_AT_RESOURCE_CEILING"


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismPowerRequirement(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-power-requirement'

    requirement_id: str
    family_id: str
    method: PhysicalScaleMorphismPowerMethod
    expected_location: Decimal
    decision_boundary: Decimal
    development_standard_deviation: Decimal | None
    observed_adverse_count: int | None
    development_board_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.requirement_id, field_name="requirement_id")
        validate_stable_id(self.family_id, field_name="family_id")
        validate_decimal(self.expected_location, field_name="expected_location")
        validate_decimal(self.decision_boundary, field_name="decision_boundary")
        if self.development_standard_deviation is not None:
            validate_decimal(
                self.development_standard_deviation,
                field_name="development_standard_deviation",
                minimum=Decimal(0),
            )
        if self.development_board_count < 2:
            raise ValueError("power requirement needs at least two development boards")
        zero_rate = self.method is PhysicalScaleMorphismPowerMethod.ZERO_ADVERSE_RATE
        if zero_rate:
            if self.observed_adverse_count is None:
                raise ValueError("zero-adverse requirement needs the observed count")
            if not 0 <= self.observed_adverse_count <= self.development_board_count:
                raise ValueError("observed adverse count lies outside its board roster")
            if self.development_standard_deviation is not None:
                raise ValueError("zero-adverse requirement does not use a standard deviation")
            if not Decimal(0) < self.decision_boundary < Decimal(1):
                raise ValueError("maximum adverse rate must lie inside (0, 1)")
            observed_rate = Decimal(self.observed_adverse_count) / Decimal(
                self.development_board_count
            )
            if self.expected_location != observed_rate:
                raise ValueError(
                    "zero-adverse expected location must equal the observed board rate"
                )
        elif self.observed_adverse_count is not None or self.development_standard_deviation is None:
            raise ValueError("continuous power requirement has inconsistent operands")
        elif self.method is PhysicalScaleMorphismPowerMethod.CONTINUOUS_EQUIVALENCE and self.decision_boundary <= 0:
            raise ValueError("continuous equivalence requires a positive symmetric margin")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismPowerRequirementResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-power-requirement-result'

    result_id: str
    requirement_id: str
    method: PhysicalScaleMorphismPowerMethod
    favorable_side: bool
    required_board_count: int | None
    terminal: PhysicalScaleMorphismPowerTerminal

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.requirement_id, field_name="requirement_id")
        if self.required_board_count is not None and self.required_board_count < 2:
            raise ValueError("power requirement cannot use fewer than two boards")
        if self.terminal is PhysicalScaleMorphismPowerTerminal.METHOD_POWER_QUALIFIED:
            if (
                not self.favorable_side
                or self.required_board_count is None
                or self.required_board_count > PHYSICAL_SCALE_MORPHISM_PANEL_SIZES[-1]
            ):
                raise ValueError("qualified power requirement needs a finite favorable design")
        if self.terminal is PhysicalScaleMorphismPowerTerminal.NO_FINITE_COUNT_UNDER_OBSERVED_RATE:
            if (
                self.method is not PhysicalScaleMorphismPowerMethod.ZERO_ADVERSE_RATE
                or self.favorable_side
                or self.required_board_count is not None
            ):
                raise ValueError("wrong-side adverse rate cannot have a finite rescue count")
        if self.terminal is PhysicalScaleMorphismPowerTerminal.DEVELOPMENT_EFFECT_ON_WRONG_SIDE:
            if (
                self.method is PhysicalScaleMorphismPowerMethod.ZERO_ADVERSE_RATE
                or self.favorable_side
                or self.required_board_count is not None
            ):
                raise ValueError("wrong-side continuous effect cannot have a finite rescue count")
        if self.terminal is PhysicalScaleMorphismPowerTerminal.PRECISION_LIMITED_AT_RESOURCE_CEILING:
            if (
                not self.favorable_side
                or self.required_board_count is None
                or self.required_board_count <= PHYSICAL_SCALE_MORPHISM_PANEL_SIZES[-1]
            ):
                raise ValueError("precision terminal requires a qualifying-side oversized design")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismPowerQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-power-qualification'

    qualification_id: str
    requirement_results: tuple[PhysicalScaleMorphismPowerRequirementResult, ...]
    candidate_panel_sizes: tuple[int, ...]
    selected_board_count_per_scale: int | None
    terminal: PhysicalScaleMorphismPowerTerminal
    board_is_resampling_unit: bool
    nested_rows_count_as_units: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        require_sorted_unique_ids(
            self.requirement_results,
            attribute="result_id",
            field_name="requirement_results",
        )
        if self.candidate_panel_sizes != PHYSICAL_SCALE_MORPHISM_PANEL_SIZES:
            raise ValueError("physical scale morphism power must use the frozen finite count ladder")
        if not self.board_is_resampling_unit or self.nested_rows_count_as_units:
            raise ValueError("physical scale morphism power must preserve the physical board unit")
        if self.terminal is PhysicalScaleMorphismPowerTerminal.METHOD_POWER_QUALIFIED:
            if self.selected_board_count_per_scale not in self.candidate_panel_sizes:
                raise ValueError("qualified power must select a frozen panel size")
        elif self.selected_board_count_per_scale is not None:
            raise ValueError("failed power gate cannot select an evaluation board count")


def _required_count(
    requirement: PhysicalScaleMorphismPowerRequirement,
    *,
    alpha: float,
    target_power: float,
) -> tuple[bool, int | None, PhysicalScaleMorphismPowerTerminal]:
    if requirement.method is PhysicalScaleMorphismPowerMethod.ZERO_ADVERSE_RATE:
        assert requirement.observed_adverse_count is not None
        observed_rate = Decimal(requirement.observed_adverse_count) / Decimal(
            requirement.development_board_count
        )
        if observed_rate >= requirement.decision_boundary:
            return False, None, PhysicalScaleMorphismPowerTerminal.NO_FINITE_COUNT_UNDER_OBSERVED_RATE
        if not requirement.observed_adverse_count:
            raw_zero = log(alpha) / log(1 - float(requirement.decision_boundary))
            required = max(
                2,
                int(Decimal(str(raw_zero)).to_integral_value(rounding=ROUND_CEILING)),
            )
        else:
            required = _required_binomial_upper_count(
                observed_rate=observed_rate,
                maximum_rate=requirement.decision_boundary,
                alpha=alpha,
            )
        terminal = (
            PhysicalScaleMorphismPowerTerminal.METHOD_POWER_QUALIFIED
            if required <= PHYSICAL_SCALE_MORPHISM_PANEL_SIZES[-1]
            else PhysicalScaleMorphismPowerTerminal.PRECISION_LIMITED_AT_RESOURCE_CEILING
        )
        return True, required, terminal
    if requirement.method is PhysicalScaleMorphismPowerMethod.CONTINUOUS_EQUIVALENCE:
        gap = requirement.decision_boundary - abs(requirement.expected_location)
    elif requirement.decision_boundary < 0:
        gap = requirement.decision_boundary - requirement.expected_location
    else:
        gap = requirement.expected_location - requirement.decision_boundary
    if gap <= 0:
        return False, None, PhysicalScaleMorphismPowerTerminal.DEVELOPMENT_EFFECT_ON_WRONG_SIDE
    assert requirement.development_standard_deviation is not None
    if requirement.development_standard_deviation == 0:
        return True, 2, PhysicalScaleMorphismPowerTerminal.METHOD_POWER_QUALIFIED
    z_alpha = NormalDist().inv_cdf(1 - alpha)
    z_power = NormalDist().inv_cdf(target_power)
    raw_normal = (
        Decimal(str(z_alpha + z_power)) * requirement.development_standard_deviation / gap
    ) ** 2
    required = max(2, int(raw_normal.to_integral_value(rounding=ROUND_CEILING)))
    terminal = (
        PhysicalScaleMorphismPowerTerminal.METHOD_POWER_QUALIFIED
        if required <= PHYSICAL_SCALE_MORPHISM_PANEL_SIZES[-1]
        else PhysicalScaleMorphismPowerTerminal.PRECISION_LIMITED_AT_RESOURCE_CEILING
    )
    return True, required, terminal


def _required_binomial_upper_count(
    *,
    observed_rate: Decimal,
    maximum_rate: Decimal,
    alpha: float,
) -> int:
    """First count whose conservative projected Clopper--Pearson upper bound passes."""

    for count in range(2, 1_000_001):
        projected_adverse = int(
            (observed_rate * Decimal(count)).to_integral_value(rounding=ROUND_CEILING)
        )
        if projected_adverse >= count:
            upper = 1.0
        else:
            upper = float(beta.ppf(1 - alpha, projected_adverse + 1, count - projected_adverse))
        if upper <= float(maximum_rate):
            return count
    raise ValueError("qualifying-side adverse rate requires more than one million boards")


def qualify_method_power(
    *,
    qualification_id: str,
    requirements: tuple[PhysicalScaleMorphismPowerRequirement, ...],
    familywise_alpha: Decimal = Decimal("0.05"),
    target_power: Decimal = Decimal("0.80"),
) -> PhysicalScaleMorphismPowerQualification:
    require_sorted_unique_ids(requirements, attribute="requirement_id", field_name="requirements")
    if not requirements:
        raise ValueError("power gate requires at least one primary requirement")
    validate_decimal(familywise_alpha, field_name="familywise_alpha", minimum=Decimal(0))
    validate_decimal(target_power, field_name="target_power", minimum=Decimal(0))
    if not Decimal(0) < familywise_alpha < Decimal(1):
        raise ValueError("familywise alpha must lie inside (0, 1)")
    if not Decimal("0.5") < target_power < Decimal(1):
        raise ValueError("target power must lie inside (0.5, 1)")
    alpha_each = float(familywise_alpha / Decimal(len(requirements)))
    results = []
    for requirement in requirements:
        favorable, count, terminal = _required_count(
            requirement,
            alpha=alpha_each,
            target_power=float(target_power),
        )
        results.append(
            PhysicalScaleMorphismPowerRequirementResult(
                result_id=f"{qualification_id}.{requirement.requirement_id}",
                requirement_id=requirement.requirement_id,
                method=requirement.method,
                favorable_side=favorable,
                required_board_count=count,
                terminal=terminal,
            )
        )
    ordered = tuple(sorted(results, key=lambda value: value.result_id))
    terminals = {value.terminal for value in ordered}
    if PhysicalScaleMorphismPowerTerminal.NO_FINITE_COUNT_UNDER_OBSERVED_RATE in terminals:
        terminal = PhysicalScaleMorphismPowerTerminal.NO_FINITE_COUNT_UNDER_OBSERVED_RATE
        selected = None
    elif PhysicalScaleMorphismPowerTerminal.DEVELOPMENT_EFFECT_ON_WRONG_SIDE in terminals:
        terminal = PhysicalScaleMorphismPowerTerminal.DEVELOPMENT_EFFECT_ON_WRONG_SIDE
        selected = None
    else:
        largest = max(value.required_board_count or 0 for value in ordered)
        selected = next((value for value in PHYSICAL_SCALE_MORPHISM_PANEL_SIZES if value >= largest), None)
        terminal = (
            PhysicalScaleMorphismPowerTerminal.METHOD_POWER_QUALIFIED
            if selected is not None
            else PhysicalScaleMorphismPowerTerminal.PRECISION_LIMITED_AT_RESOURCE_CEILING
        )
    return PhysicalScaleMorphismPowerQualification(
        qualification_id=qualification_id,
        requirement_results=ordered,
        candidate_panel_sizes=PHYSICAL_SCALE_MORPHISM_PANEL_SIZES,
        selected_board_count_per_scale=selected,
        terminal=terminal,
        board_is_resampling_unit=True,
        nested_rows_count_as_units=False,
    )


def board_level_simultaneous_inference(
    *,
    result_id: str,
    units: tuple[CompleteUnitVector, ...],
    seed: int,
    bootstrap_replicates: int = 2_000,
) -> CompleteUnitInferenceResult:
    """Reuse the current complete-unit max-t implementation with board strata."""

    config = CompleteUnitInferenceConfig(
        config_id=f"{result_id}.config",
        family_id="physical-scale-morphism-primary-board-family",
        confidence_level=Decimal("0.95"),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
        chunk_size=min(200, bootstrap_replicates),
        minimum_complete_units=2,
        resampling_unit="COMPLETE_UNIT",
        stratification_frozen=True,
        simultaneous_family_frozen=True,
        frozen_before_outcomes=True,
        protected_outcome_access_count=0,
    )
    return complete_unit_simultaneous_inference(result_id=result_id, config=config, units=units)


__all__ = [
    "PHYSICAL_SCALE_MORPHISM_PANEL_SIZES",
    'PhysicalScaleMorphismPowerMethod',
    'PhysicalScaleMorphismPowerQualification',
    'PhysicalScaleMorphismPowerRequirement',
    'PhysicalScaleMorphismPowerRequirementResult',
    'PhysicalScaleMorphismPowerTerminal',
    "board_level_simultaneous_inference",
    "qualify_method_power",
]
