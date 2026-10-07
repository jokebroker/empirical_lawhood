"""Frozen finite design and material preparation distribution for Cantera."""

from __future__ import annotations

from decimal import Decimal

from empirical_lawhood.adapters.methods.selective_dependence_response.analysis_design import SelectiveDependenceResponseExchangeDesign, SelectiveDependenceResponseExchangeReductionKind, SelectiveDependenceResponseTargetAnalysisFreeze, expected_primary_cell_ids, expected_support_boundary_cell_ids, validate_target_analysis_freeze
from empirical_lawhood.adapters.methods.selective_dependence_response.comparators import SelectiveDependenceResponseComparatorKind
from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponseExchangeExpectation, SelectiveDependenceResponsePreparationDistributionFreeze, digest_ids
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import CanteraReactionResponseCanteraDesign


def _unit_ids(phase: str, count: int) -> tuple[str, ...]:
    return tuple(f"cantera.selective-dependence-response.{phase}.unit-{index:03d}" for index in range(count))


def cantera_preparation_freeze() -> SelectiveDependenceResponsePreparationDistributionFreeze:
    canary = _unit_ids("canary", 4)
    development = _unit_ids("development", 24)
    evaluation = _unit_ids("evaluation", 64)
    reserve = _unit_ids("reserve", 8)
    all_ids = tuple(sorted((*canary, *development, *evaluation, *reserve)))
    return SelectiveDependenceResponsePreparationDistributionFreeze(
        freeze_id="cantera.preparation-distribution",
        target_id="target.cantera-selective-dependence-response",
        variable_ids=(
            "bath-temperature",
            "cold-checkpoint-temperature",
            "equivalence-ratio",
            "heat-transfer-scale",
            "hot-checkpoint-seed-temperature",
            "inlet-temperature",
            "reactor-volume",
        ),
        distribution_statement=(
            "Independent SHA-256/PCG64 draws vary inlet temperature and equivalence ratio, "
            "hot and cold checkpoint temperatures, bath temperature, heat-transfer scale "
            "and reactor volume over frozen bounded uniform ranges."
        ),
        canary_unit_ids=canary,
        development_unit_ids=development,
        evaluation_unit_ids=evaluation,
        reserve_unit_ids=reserve,
        all_unit_ids_sha256=digest_ids(all_ids),
        nested_conditions_count_as_units=False,
        materially_varying_preparations=True,
        seed_family="explicit-full-seed-pcg64",
        development_response_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def cantera_design() -> CanteraReactionResponseCanteraDesign:
    return CanteraReactionResponseCanteraDesign(
        design_id="cantera.nonisothermal-cstr.design",
        target_id="target.cantera-selective-dependence-response",
        cantera_version="3.2.0",
        mechanism_id="gri30",
        phase_name="gri30",
        reactor_model_id="ideal-gas-reactor-energy-on",
        denominator_ids=("heat-loss-high", "heat-loss-low"),
        history_ids=("cold-checkpoint", "hot-checkpoint"),
        action_ids=("flow-high", "flow-hold", "flow-low", "flow-outside"),
        horizon_ids=("long", "short"),
        receiver_ids=(
            "carbon-monoxide",
            "element-error",
            "methane-conversion",
            "peak-temperature",
            "temperature",
        ),
        action_multipliers=(Decimal("0.75"), Decimal("1.00"), Decimal("1.25")),
        horizon_seconds=(Decimal("0.05"), Decimal("0.60")),
        base_residence_seconds=Decimal("0.12"),
        heat_transfer_coefficients=(Decimal("0.14"), Decimal("0.26")),
        target_temperature_lower=Decimal("1810"),
        target_temperature_upper=Decimal("1830"),
        minimum_conversion=Decimal("0.95"),
        maximum_co_mole_fraction=Decimal("0.00050"),
        maximum_peak_temperature=Decimal("2700"),
        maximum_element_error=Decimal("0.000001"),
        outside_action_multiplier=Decimal("1.60"),
        maximum_solver_steps=20000,
        maximum_wall_seconds_per_unit=Decimal("30"),
        construct_validation_numeric_identity_reused=False,
        development_outcome_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def cantera_analysis_freeze() -> SelectiveDependenceResponseTargetAnalysisFreeze:
    """Return the exact outcome-blind Cantera analysis and adjudication design."""

    design = cantera_design()
    preparation = cantera_preparation_freeze()
    scored_receivers = ("carbon-monoxide", "methane-conversion", "temperature")
    exchanges = (
        SelectiveDependenceResponseExchangeDesign(
            exchange_id="exchange.cantera.active-denominator",
            target_id=design.target_id,
            expectation=SelectiveDependenceResponseExchangeExpectation.ACTIVE,
            coordinate_role_id="D",
            left_cell_id=("cell.heat-loss-high.hot-checkpoint.flow-high.long.receiver-temperature"),
            right_cell_id=("cell.heat-loss-low.hot-checkpoint.flow-high.long.receiver-temperature"),
            estimand_id="active-denominator",
            equivalence_margin=Decimal("1"),
            minimum_active_difference=Decimal("5"),
            native_unit="kelvin",
            reduction_kind=SelectiveDependenceResponseExchangeReductionKind.DIFFERENCE_OF_ACTION_MINUS_HOLD,
        ),
        SelectiveDependenceResponseExchangeDesign(
            exchange_id="exchange.cantera.active-history",
            target_id=design.target_id,
            expectation=SelectiveDependenceResponseExchangeExpectation.ACTIVE,
            coordinate_role_id="H",
            left_cell_id=("cell.heat-loss-high.hot-checkpoint.flow-high.long.receiver-temperature"),
            right_cell_id=(
                "cell.heat-loss-high.cold-checkpoint.flow-high.long.receiver-temperature"
            ),
            estimand_id="active-history",
            equivalence_margin=Decimal("1"),
            minimum_active_difference=Decimal("5"),
            native_unit="kelvin",
            reduction_kind=SelectiveDependenceResponseExchangeReductionKind.DIFFERENCE_OF_ACTION_MINUS_HOLD,
        ),
        SelectiveDependenceResponseExchangeDesign(
            exchange_id="exchange.cantera.active-horizon",
            target_id=design.target_id,
            expectation=SelectiveDependenceResponseExchangeExpectation.ACTIVE,
            coordinate_role_id="tau",
            left_cell_id=(
                "cell.heat-loss-high.hot-checkpoint.flow-high.short.receiver-temperature"
            ),
            right_cell_id=(
                "cell.heat-loss-high.hot-checkpoint.flow-high.long.receiver-temperature"
            ),
            estimand_id="active-horizon",
            equivalence_margin=Decimal("1"),
            minimum_active_difference=Decimal("20"),
            native_unit="kelvin",
            reduction_kind=SelectiveDependenceResponseExchangeReductionKind.DIFFERENCE_OF_ACTION_MINUS_HOLD,
        ),
        SelectiveDependenceResponseExchangeDesign(
            exchange_id="exchange.cantera.invariant-cold-denominator",
            target_id=design.target_id,
            expectation=SelectiveDependenceResponseExchangeExpectation.INVARIANT,
            coordinate_role_id="D",
            left_cell_id=(
                "cell.heat-loss-high.cold-checkpoint.flow-high.long.receiver-methane-conversion"
            ),
            right_cell_id=(
                "cell.heat-loss-low.cold-checkpoint.flow-high.long.receiver-methane-conversion"
            ),
            estimand_id="invariant-cold-denominator",
            equivalence_margin=Decimal("0.0001"),
            minimum_active_difference=Decimal("0.001"),
            native_unit="dimensionless-fraction",
            reduction_kind=SelectiveDependenceResponseExchangeReductionKind.DIFFERENCE_OF_ACTION_MINUS_HOLD,
        ),
        SelectiveDependenceResponseExchangeDesign(
            exchange_id="exchange.cantera.support-boundary",
            target_id=design.target_id,
            expectation=SelectiveDependenceResponseExchangeExpectation.SUPPORT_BOUNDARY,
            coordinate_role_id="A",
            left_cell_id=(
                "cell.heat-loss-high.hot-checkpoint.flow-outside.long.receiver-temperature"
            ),
            right_cell_id=(
                "cell.heat-loss-high.hot-checkpoint.flow-hold.long.receiver-temperature"
            ),
            estimand_id="support-boundary",
            equivalence_margin=Decimal("0"),
            minimum_active_difference=Decimal("1"),
            native_unit="support-class",
            reduction_kind=SelectiveDependenceResponseExchangeReductionKind.SUPPORT_REFUSAL,
        ),
    )
    freeze = SelectiveDependenceResponseTargetAnalysisFreeze(
        freeze_id="cantera.target-analysis-freeze",
        target_id=design.target_id,
        design=ObjectIdentity.from_record(design.design_id, design),
        preparation_freeze=ObjectIdentity.from_record(preparation.freeze_id, preparation),
        signature_id="signature.cantera.selective-dependence",
        hold_action_id="flow-hold",
        scored_receiver_ids=scored_receivers,
        primary_cell_ids=expected_primary_cell_ids(
            denominator_ids=design.denominator_ids,
            history_ids=design.history_ids,
            action_ids=design.action_ids,
            horizon_ids=design.horizon_ids,
            receiver_ids=scored_receivers,
        ),
        support_boundary_cell_ids=expected_support_boundary_cell_ids(
            denominator_ids=design.denominator_ids,
            history_ids=design.history_ids,
            outside_action_ids=("flow-outside",),
            horizon_ids=design.horizon_ids,
        ),
        neutral_margin=Decimal("0.0001"),
        minimum_law_accuracy=Decimal("0.70"),
        exchange_familywise_alpha=Decimal("0.05"),
        exchange_interval_method="bonferroni-complete-unit-student-t",
        exchanges=exchanges,
        denominator_omitted_role_id="H",
        denominator_omitted_comparator_kind=SelectiveDependenceResponseComparatorKind.HISTORY_BLIND,
        denominator_split_variable_id="heat-transfer-scale",
        denominator_split_threshold=Decimal("1.0"),
        candidate_evaluation_panel_sizes=(32, 48, 64),
        power_familywise_alpha=Decimal("0.05"),
        target_power=Decimal("0.80"),
        maximum_nonevaluable_rate=Decimal("0.10"),
        maximum_adverse_rate=Decimal("0.10"),
        minimum_comparator_advantage=Decimal("0"),
        target_margin_id="target-temperature-band",
        sink_margin_ids=(
            "co-ceiling",
            "element-closure",
            "minimum-conversion",
            "peak-temperature-ceiling",
        ),
        development_outcome_count=0,
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    validate_target_analysis_freeze(
        freeze,
        design=design,
        design_id=design.design_id,
        preparation=preparation,
        denominator_ids=design.denominator_ids,
        history_ids=design.history_ids,
        action_ids=design.action_ids,
        horizon_ids=design.horizon_ids,
        receiver_ids=design.receiver_ids,
    )
    return freeze


__all__ = ["cantera_analysis_freeze", "cantera_design", "cantera_preparation_freeze"]
