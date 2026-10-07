"""Prospective, outcome-blind design arithmetic for quantum receiver response."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import NormalDist
from typing import Final

from .contracts import QuantumReceiverResponseConfig


OBSERVATION_ORDER_FAMILY_SIZE: Final = 8
OBSERVATION_ORDER_DESIGN_ALTERNATIVES: Final = {
    "chi_h20": 0.50,
    "chi_h4": 0.20,
    "weak_separation_h20": 0.30,
    "proxy_association_h4": 0.30,
    "chi_total": 0.0,
    "cancellation": 0.0,
}


@dataclass(frozen=True, slots=True)
class DesignRequirement:
    contrast_id: str
    alternative: float
    boundary: float
    gate_margin: float
    maximum_halfwidth: float
    required_for_power: int
    required_for_halfwidth: int


@dataclass(frozen=True, slots=True)
class ObservationOrderDesignCalculation:
    family_size: int
    family_alpha: float
    joint_power_target: float
    design_critical_value: float
    requirements: tuple[DesignRequirement, ...]
    unrounded_required_units: int
    selected_units: int
    activated_expansion_blocks: int
    maximum_units: int
    projected_joint_power_lower_bound: float
    passed: bool

    def to_document(self) -> dict[str, object]:
        return {
            "schema": 'empirical-lawhood/planning/quantum-receiver-response/prospective-design-calculation',
            "version": "1.0.0",
            "value": asdict(self),
        }


def _fano_standard_error(fano: float, units: int) -> float:
    return math.sqrt(2.0 * fano * fano / (units - 1))


def _correlation_standard_error(correlation: float, units: int) -> float:
    # Delta-method conversion from Fisher-z back to correlation.
    return (1.0 - correlation * correlation) / math.sqrt(units - 3)


def _required_units(
    *,
    standard_error_numerator: float,
    gate_margin: float,
    maximum_halfwidth: float,
    critical: float,
    directional_power_z: float,
    offset: int = 1,
) -> tuple[int, int]:
    power = offset + math.ceil(
        (standard_error_numerator * (critical + directional_power_z) / gate_margin) ** 2
    )
    precision = offset + math.ceil((standard_error_numerator * critical / maximum_halfwidth) ** 2)
    return power, precision


def observation_order_design_calculation(config: QuantumReceiverResponseConfig) -> ObservationOrderDesignCalculation:
    """Return the conservative block-rounded observation-order design fixed before source contact.

    The design uses a Bonferroni critical value for eight scalar contrasts.
    Adjudication still uses the plan's empirical max-statistic bootstrap.
    """

    alpha = float(config.observation_order_family_alpha)
    joint_power = float(config.observation_order_joint_design_power)
    normal = NormalDist()
    critical = normal.inv_cdf(1.0 - alpha / (2.0 * OBSERVATION_ORDER_FAMILY_SIZE))
    # Requiring every contrast to have this marginal power gives a conservative
    # Bonferroni lower bound of the requested joint power.
    marginal_failure = (1.0 - joint_power) / OBSERVATION_ORDER_FAMILY_SIZE
    directional_power_z = normal.inv_cdf(1.0 - marginal_failure)

    requirements: list[DesignRequirement] = []

    def add_fano(
        contrast_id: str,
        *,
        fano: float,
        boundary: float,
        alternative: float,
    ) -> None:
        margin = alternative - boundary
        maximum_halfwidth = 0.5 * margin
        numerator = math.sqrt(2.0) * fano
        power, precision = _required_units(
            standard_error_numerator=numerator,
            gate_margin=margin,
            maximum_halfwidth=maximum_halfwidth,
            critical=critical,
            directional_power_z=directional_power_z,
        )
        requirements.append(
            DesignRequirement(
                contrast_id=contrast_id,
                alternative=alternative,
                boundary=boundary,
                gate_margin=margin,
                maximum_halfwidth=maximum_halfwidth,
                required_for_power=power,
                required_for_halfwidth=precision,
            )
        )

    add_fano(
        "chi_h20",
        fano=1.50,
        boundary=float(config.observation_order_chi_h20_gate),
        alternative=OBSERVATION_ORDER_DESIGN_ALTERNATIVES["chi_h20"],
    )
    add_fano(
        "chi_h4",
        fano=1.20,
        boundary=float(config.observation_order_chi_h4_gate),
        alternative=OBSERVATION_ORDER_DESIGN_ALTERNATIVES["chi_h4"],
    )

    # Difference of independently streamed strong and weak Fano estimates.
    weak_numerator = math.sqrt(2.0 * (1.50**2 + 1.20**2))
    weak_margin = OBSERVATION_ORDER_DESIGN_ALTERNATIVES["weak_separation_h20"] - float(config.observation_order_weak_separation)
    weak_power, weak_precision = _required_units(
        standard_error_numerator=weak_numerator,
        gate_margin=weak_margin,
        maximum_halfwidth=0.5 * weak_margin,
        critical=critical,
        directional_power_z=directional_power_z,
    )
    requirements.append(
        DesignRequirement(
            contrast_id="weak_separation_h20",
            alternative=OBSERVATION_ORDER_DESIGN_ALTERNATIVES["weak_separation_h20"],
            boundary=float(config.observation_order_weak_separation),
            gate_margin=weak_margin,
            maximum_halfwidth=0.5 * weak_margin,
            required_for_power=weak_power,
            required_for_halfwidth=weak_precision,
        )
    )

    association_alternative = OBSERVATION_ORDER_DESIGN_ALTERNATIVES["proxy_association_h4"]
    association_boundary = float(config.observation_order_proxy_association)
    association_margin = association_alternative - association_boundary
    fisher_margin = math.atanh(association_alternative) - math.atanh(association_boundary)
    association_power = 3 + math.ceil(((critical + directional_power_z) / fisher_margin) ** 2)
    association_halfwidth = 0.5 * association_margin
    association_numerator = 1.0 - association_alternative**2
    _, association_precision = _required_units(
        standard_error_numerator=association_numerator,
        gate_margin=association_margin,
        maximum_halfwidth=association_halfwidth,
        critical=critical,
        directional_power_z=directional_power_z,
        offset=3,
    )
    requirements.append(
        DesignRequirement(
            contrast_id="proxy_association_h4",
            alternative=association_alternative,
            boundary=association_boundary,
            gate_margin=association_margin,
            maximum_halfwidth=association_halfwidth,
            required_for_power=association_power,
            required_for_halfwidth=association_precision,
        )
    )

    # The exact fixed-hazard null has unit Fano. Use its conservative Fano
    # standard error for both global-null and cancellation equivalence roles.
    for contrast_id, boundary in (
        ("chi_total_h20_h4", float(config.observation_order_global_equivalence)),
        ("cancellation_h20_h4", float(config.observation_order_cancellation_equivalence)),
    ):
        power, precision = _required_units(
            standard_error_numerator=math.sqrt(2.0),
            gate_margin=boundary,
            maximum_halfwidth=0.5 * boundary,
            critical=critical,
            directional_power_z=directional_power_z,
        )
        requirements.append(
            DesignRequirement(
                contrast_id=contrast_id,
                alternative=0.0,
                boundary=boundary,
                gate_margin=boundary,
                maximum_halfwidth=0.5 * boundary,
                required_for_power=power,
                required_for_halfwidth=precision,
            )
        )

    unrounded = max(max(row.required_for_power, row.required_for_halfwidth) for row in requirements)
    selected = config.observation_order_base_units
    while selected < unrounded and selected < config.observation_order_max_units:
        selected += config.observation_order_expansion_block_units
    selected = min(selected, config.observation_order_max_units)

    failure_probabilities = []
    for row in requirements:
        if row.contrast_id == "proxy_association_h4":
            standardized_margin = (
                math.atanh(row.alternative) - math.atanh(row.boundary)
            ) * math.sqrt(selected - 3)
        elif row.contrast_id == "weak_separation_h20":
            standardized_margin = row.gate_margin * math.sqrt(selected) / weak_numerator
        elif row.contrast_id.startswith("chi_"):
            fano = 1.50 if row.contrast_id == "chi_h20" else 1.20
            standardized_margin = row.gate_margin / _fano_standard_error(
                fano,
                selected,
            )
        else:
            standardized_margin = row.gate_margin * math.sqrt(selected - 1) / math.sqrt(2.0)
        failure_probabilities.append(normal.cdf(critical - standardized_margin))
    projected_joint_lower = max(0.0, 1.0 - sum(failure_probabilities))
    passed = (
        selected >= unrounded
        and selected <= config.observation_order_max_units
        and projected_joint_lower >= joint_power
    )
    return ObservationOrderDesignCalculation(
        family_size=OBSERVATION_ORDER_FAMILY_SIZE,
        family_alpha=alpha,
        joint_power_target=joint_power,
        design_critical_value=critical,
        requirements=tuple(requirements),
        unrounded_required_units=unrounded,
        selected_units=selected,
        activated_expansion_blocks=(
            (selected - config.observation_order_base_units) // config.observation_order_expansion_block_units
        ),
        maximum_units=config.observation_order_max_units,
        projected_joint_power_lower_bound=projected_joint_lower,
        passed=passed,
    )


__all__ = [
    "DesignRequirement",
    "OBSERVATION_ORDER_DESIGN_ALTERNATIVES",
    "OBSERVATION_ORDER_FAMILY_SIZE",
    "ObservationOrderDesignCalculation",
    "observation_order_design_calculation",
]
