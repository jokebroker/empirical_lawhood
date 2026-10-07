"""Frozen finite design and independent field preparations for FiPy."""

from __future__ import annotations

from decimal import Decimal

from empirical_lawhood.adapters.methods.selective_dependence_response.analysis_design import SelectiveDependenceResponseExchangeDesign, SelectiveDependenceResponseExchangeReductionKind, SelectiveDependenceResponseTargetAnalysisFreeze, expected_primary_cell_ids, expected_support_boundary_cell_ids, validate_target_analysis_freeze
from empirical_lawhood.adapters.methods.selective_dependence_response.comparators import SelectiveDependenceResponseComparatorKind
from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponseExchangeExpectation, SelectiveDependenceResponsePreparationDistributionFreeze, digest_ids
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import FipyReactionDiffusionResponseFiPyDesign


def _unit_ids(phase: str, count: int) -> tuple[str, ...]:
    return tuple(f"fipy.selective-dependence-response.{phase}.unit-{index:03d}" for index in range(count))


def fipy_preparation_freeze() -> SelectiveDependenceResponsePreparationDistributionFreeze:
    canary = _unit_ids("canary", 4)
    development = _unit_ids("development", 32)
    evaluation = _unit_ids("evaluation", 64)
    reserve = _unit_ids("reserve", 8)
    all_ids = tuple(sorted((*canary, *development, *evaluation, *reserve)))
    return SelectiveDependenceResponsePreparationDistributionFreeze(
        freeze_id="fipy.preparation-distribution",
        target_id="target.fipy-selective-dependence-response",
        variable_ids=(
            "background-amplitude",
            "residual-amplitude",
            "residual-centre",
            "residual-width",
        ),
        distribution_statement=(
            "Independent SHA-256/PCG64 draws vary the pre-action residual plume "
            "amplitude, centre, width and bounded background amplitude; cleared and "
            "residual checkpoints are nested branches of the same preparation."
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


def fipy_design() -> FipyReactionDiffusionResponseFiPyDesign:
    return FipyReactionDiffusionResponseFiPyDesign(
        design_id="fipy.signed-reaction-diffusion.design",
        target_id="target.fipy-selective-dependence-response",
        fipy_version="4.0.3",
        equation_id="transient-diffusion-linear-decay-signed-source",
        denominator_ids=("diffusivity-high", "diffusivity-low"),
        history_ids=("cleared-checkpoint", "residual-checkpoint"),
        action_ids=("source-hold", "source-inject", "source-outside", "source-remove"),
        horizon_ids=("long", "short"),
        receiver_ids=(
            "balance-error",
            "boundary-flux",
            "downstream-mean",
            "field-mass",
            "field-maximum",
            "field-minimum",
        ),
        diffusivities=(Decimal("0.01"), Decimal("0.08")),
        action_rates=(Decimal("-0.25"), Decimal("0"), Decimal("0.25")),
        horizon_seconds=(Decimal("0.10"), Decimal("0.50")),
        decay_rate=Decimal("0.15"),
        action_duration_seconds=Decimal("0.10"),
        mesh_cells=60,
        domain_length=Decimal("1.0"),
        timestep_seconds=Decimal("0.010"),
        target_downstream_lower=Decimal("0.00070"),
        target_downstream_upper=Decimal("0.00120"),
        maximum_peak_field=Decimal("0.014"),
        maximum_boundary_flux=Decimal("0.020"),
        maximum_balance_error=Decimal("0.0020"),
        nonnegativity_tolerance=Decimal("0.00010"),
        outside_action_rate=Decimal("-0.50"),
        maximum_solver_steps=500,
        construct_validation_numeric_identity_reused=False,
        development_outcome_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def fipy_analysis_freeze() -> SelectiveDependenceResponseTargetAnalysisFreeze:
    """Return the exact outcome-blind FiPy analysis and adjudication design."""

    design = fipy_design()
    preparation = fipy_preparation_freeze()
    scored_receivers = ("downstream-mean", "field-minimum")
    specs = (
        (
            "exchange.fipy.active-action",
            SelectiveDependenceResponseExchangeExpectation.ACTIVE,
            "A",
            "cell.diffusivity-high.residual-checkpoint.source-inject.long.receiver-downstream-mean",
            "cell.diffusivity-high.residual-checkpoint.source-remove.long.receiver-downstream-mean",
            "active-action-orientation",
            Decimal("0.000001"),
            Decimal("0.00005"),
            SelectiveDependenceResponseExchangeReductionKind.DIFFERENCE_OF_ACTION_MINUS_HOLD,
        ),
        (
            "exchange.fipy.active-denominator",
            SelectiveDependenceResponseExchangeExpectation.ACTIVE,
            "D",
            "cell.diffusivity-high.residual-checkpoint.source-inject.long.receiver-downstream-mean",
            "cell.diffusivity-low.residual-checkpoint.source-inject.long.receiver-downstream-mean",
            "active-denominator",
            Decimal("0.000001"),
            Decimal("0.00005"),
            SelectiveDependenceResponseExchangeReductionKind.DIFFERENCE_OF_ACTION_MINUS_HOLD,
        ),
        (
            "exchange.fipy.active-history-admission",
            SelectiveDependenceResponseExchangeExpectation.ACTIVE,
            "H",
            "cell.diffusivity-high.residual-checkpoint.source-hold.long.receiver-downstream-mean",
            "cell.diffusivity-high.cleared-checkpoint.source-hold.long.receiver-downstream-mean",
            "active-history-admission",
            Decimal("0.000001"),
            Decimal("0.00030"),
            SelectiveDependenceResponseExchangeReductionKind.DIRECT_RECEIVER_DIFFERENCE,
        ),
        (
            "exchange.fipy.active-horizon",
            SelectiveDependenceResponseExchangeExpectation.ACTIVE,
            "tau",
            "cell.diffusivity-high.residual-checkpoint.source-inject.long.receiver-downstream-mean",
            "cell.diffusivity-high.residual-checkpoint.source-inject.short.receiver-downstream-mean",
            "active-horizon",
            Decimal("0.000001"),
            Decimal("0.00005"),
            SelectiveDependenceResponseExchangeReductionKind.DIFFERENCE_OF_ACTION_MINUS_HOLD,
        ),
        (
            "exchange.fipy.invariant-history-response",
            SelectiveDependenceResponseExchangeExpectation.INVARIANT,
            "H",
            "cell.diffusivity-high.residual-checkpoint.source-inject.long.receiver-downstream-mean",
            "cell.diffusivity-high.cleared-checkpoint.source-inject.long.receiver-downstream-mean",
            "invariant-history-response",
            Decimal("0.00000001"),
            Decimal("0.00005"),
            SelectiveDependenceResponseExchangeReductionKind.DIFFERENCE_OF_ACTION_MINUS_HOLD,
        ),
        (
            "exchange.fipy.support-boundary",
            SelectiveDependenceResponseExchangeExpectation.SUPPORT_BOUNDARY,
            "A",
            "cell.diffusivity-high.residual-checkpoint.source-outside.long.receiver-downstream-mean",
            "cell.diffusivity-high.residual-checkpoint.source-hold.long.receiver-downstream-mean",
            "support-boundary",
            Decimal("0"),
            Decimal("0.00005"),
            SelectiveDependenceResponseExchangeReductionKind.SUPPORT_REFUSAL,
        ),
    )
    exchanges = tuple(
        SelectiveDependenceResponseExchangeDesign(
            exchange_id=exchange_id,
            target_id=design.target_id,
            expectation=expectation,
            coordinate_role_id=role,
            left_cell_id=left,
            right_cell_id=right,
            estimand_id=estimand,
            equivalence_margin=margin,
            minimum_active_difference=difference,
            native_unit=(
                "support-class"
                if expectation is SelectiveDependenceResponseExchangeExpectation.SUPPORT_BOUNDARY
                else "field-amplitude"
            ),
            reduction_kind=reduction_kind,
        )
        for (
            exchange_id,
            expectation,
            role,
            left,
            right,
            estimand,
            margin,
            difference,
            reduction_kind,
        ) in specs
    )
    freeze = SelectiveDependenceResponseTargetAnalysisFreeze(
        freeze_id="fipy.target-analysis-freeze",
        target_id=design.target_id,
        design=ObjectIdentity.from_record(design.design_id, design),
        preparation_freeze=ObjectIdentity.from_record(preparation.freeze_id, preparation),
        signature_id="signature.fipy.selective-dependence",
        hold_action_id="source-hold",
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
            outside_action_ids=("source-outside",),
            horizon_ids=design.horizon_ids,
        ),
        neutral_margin=Decimal("0.0000001"),
        minimum_law_accuracy=Decimal("0.70"),
        exchange_familywise_alpha=Decimal("0.05"),
        exchange_interval_method="bonferroni-complete-unit-student-t",
        exchanges=exchanges,
        denominator_omitted_role_id="H",
        denominator_omitted_comparator_kind=SelectiveDependenceResponseComparatorKind.HISTORY_BLIND,
        denominator_split_variable_id="residual-amplitude",
        denominator_split_threshold=Decimal("0.040"),
        candidate_evaluation_panel_sizes=(32, 48, 64),
        power_familywise_alpha=Decimal("0.05"),
        target_power=Decimal("0.80"),
        maximum_nonevaluable_rate=Decimal("0.10"),
        maximum_adverse_rate=Decimal("0.10"),
        minimum_comparator_advantage=Decimal("0"),
        target_margin_id="downstream-target-band",
        sink_margin_ids=(
            "balance-error-ceiling",
            "boundary-flux-ceiling",
            "nonnegativity-tolerance",
            "peak-field-ceiling",
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


__all__ = ["fipy_analysis_freeze", "fipy_design", "fipy_preparation_freeze"]
