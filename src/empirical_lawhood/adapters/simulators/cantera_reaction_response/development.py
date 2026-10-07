"""Cantera development-only law, comparator, power and forecast freeze."""

from __future__ import annotations

from empirical_lawhood.adapters.methods.selective_dependence_response.analysis_design import SelectiveDependenceResponseTargetAnalysisFreeze
from empirical_lawhood.adapters.methods.selective_dependence_response.comparators import SelectiveDependenceResponseComparatorEncoding, SelectiveDependenceResponseForecastSeparationReport, compare_forecast_to_comparators, comparator_relevant_cells, comparator_relevant_exchanges, fit_comparator_family
from empirical_lawhood.adapters.methods.selective_dependence_response.denominator import SelectiveDependenceResponseDenominatorSelection, assess_denominator_alternatives
from empirical_lawhood.adapters.methods.selective_dependence_response.analysis import SelectiveDependenceResponseFiniteLawCalibration, SelectiveDependenceResponseSelectiveDependenceSignature, development_prediction_metrics, derive_context_decisions_from_predictions, target_sink_conflict_case_ids
from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponseCaseState, SelectiveDependenceResponseCellForecast, SelectiveDependenceResponseConstructReviewAttestation, SelectiveDependenceResponseContaminationLedger, SelectiveDependenceResponseDisposition, SelectiveDependenceResponseExchangeExpectation, SelectiveDependenceResponseMethodQuestionFreeze, SelectiveDependenceResponsePhase, SelectiveDependenceResponsePreparationDistributionFreeze, SelectiveDependenceResponseSelectiveLawForecast, SelectiveDependenceResponseTargetPanel, SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID, digest_ids
from empirical_lawhood.adapters.methods.selective_dependence_response.forecast import SelectiveDependenceResponseDevelopmentParent, SelectiveDependenceResponseDevelopmentLineage, SelectiveDependenceResponseForecastChallengeQualification, freeze_development_parent, qualify_forecast_challenge
from empirical_lawhood.adapters.methods.selective_dependence_response.power import SelectiveDependenceResponsePowerDesignQualification, SelectiveDependenceResponsePowerMethod, SelectiveDependenceResponsePowerOperand, continuous_power_operand, qualify_power_design, zero_failure_power_operand
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal

from .contracts import CanteraReactionResponseCanteraDesign
from .reduction import analyze_cantera_development, cantera_exchange_contrasts


def _mechanistic(
    denominator_id: str,
    history_id: str,
    action_id: str,
    horizon_id: str,
    receiver_id: str,
) -> tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition]:
    if action_id == "flow-outside":
        return (
            SelectiveDependenceResponseCaseState.OUTSIDE_SUPPORT,
            False,
            SelectiveDependenceResponseDisposition.NONATTEMPT,
        )
    if action_id == "flow-hold":
        state = SelectiveDependenceResponseCaseState.NEUTRAL
    elif action_id == "flow-high":
        state = SelectiveDependenceResponseCaseState.HIGH
    else:
        state = SelectiveDependenceResponseCaseState.LOW
    if receiver_id == "carbon-monoxide" and action_id == "flow-low":
        state = SelectiveDependenceResponseCaseState.LOW
    hold_viable = (
        denominator_id == "heat-loss-low"
        and history_id == "hot-checkpoint"
        and horizon_id == "long"
    )
    admitted = hold_viable and action_id in {"flow-high", "flow-low"}
    fibre_admitted = hold_viable if action_id == "flow-hold" else admitted
    disposition = (
        SelectiveDependenceResponseDisposition.ACTION_AVAILABLE
        if admitted
        else (SelectiveDependenceResponseDisposition.HOLD_ONLY if hold_viable else SelectiveDependenceResponseDisposition.NONATTEMPT)
    )
    return state, fibre_admitted, disposition


def _parse_cell(cell_id: str) -> tuple[str, str, str, str, str]:
    parts = cell_id.split(".")
    if len(parts) != 6 or not parts[5].startswith("receiver-"):
        raise ValueError("Cantera forecast cell identity differs")
    return parts[1], parts[2], parts[3], parts[4], parts[5].removeprefix("receiver-")


def _power_operands(
    *,
    panel: SelectiveDependenceResponseTargetPanel,
    law: SelectiveDependenceResponseFiniteLawCalibration,
    comparators: tuple[SelectiveDependenceResponseComparatorEncoding, ...],
    separation: SelectiveDependenceResponseForecastSeparationReport,
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
) -> tuple[SelectiveDependenceResponsePowerOperand, ...]:
    exchange_specs = {value.exchange_id: value for value in analysis_freeze.exchanges}
    operands: list[SelectiveDependenceResponsePowerOperand] = []
    for exchange_id, contrasts in sorted(
        cantera_exchange_contrasts(panel, analysis_freeze).items()
    ):
        observations = tuple(
            NamedDecimal(
                value_id=value.complete_unit_id,
                value=value.value,
                unit=value.native_unit,
            )
            for value in contrasts
            if value.value is not None
        )
        spec = exchange_specs[exchange_id]
        invariant = spec.expectation is SelectiveDependenceResponseExchangeExpectation.INVARIANT
        operands.append(
            continuous_power_operand(
                requirement_id=f"power.cantera.{spec.estimand_id}",
                family_id="invariant-exchange" if invariant else "active-exchange",
                method=(
                    SelectiveDependenceResponsePowerMethod.TWO_ONE_SIDED_EQUIVALENCE
                    if invariant
                    else SelectiveDependenceResponsePowerMethod.TWO_SIDED_MINIMUM_ABSOLUTE_MEAN
                ),
                observations=observations,
                decision_boundary=(
                    spec.equivalence_margin if invariant else spec.minimum_active_difference
                ),
                native_unit=spec.native_unit,
            )
        )
    primary = development_prediction_metrics(
        panel,
        predictions=law.predictions,
        complete_unit_ids=law.calibration_complete_unit_ids,
        hold_action_id=analysis_freeze.hold_action_id,
        receiver_ids=analysis_freeze.scored_receiver_ids,
        neutral_margin=analysis_freeze.neutral_margin,
    )
    operands.append(
        continuous_power_operand(
            requirement_id="power.cantera.law-calibration",
            family_id="law-calibration",
            method=SelectiveDependenceResponsePowerMethod.ONE_SIDED_MINIMUM_MEAN,
            observations=primary.complete_unit_accuracies,
            decision_boundary=law.minimum_required_accuracy,
            native_unit="complete-unit-fraction-correct",
        )
    )
    cell_required_comparator_ids = {
        value.comparator_id for value in separation.separations if not value.differing_exchange_ids
    }
    comparator_scores = []
    for comparator in comparators:
        if comparator.encoding_id not in cell_required_comparator_ids:
            continue
        metrics = development_prediction_metrics(
            panel,
            predictions=comparator.predictions,
            complete_unit_ids=law.calibration_complete_unit_ids,
            hold_action_id=analysis_freeze.hold_action_id,
            receiver_ids=analysis_freeze.scored_receiver_ids,
            neutral_margin=analysis_freeze.neutral_margin,
        )
        comparator_scores.append(
            {value.value_id: value.value for value in metrics.complete_unit_accuracies}
        )
    paired_advantages = tuple(
        NamedDecimal(
            value_id=value.value_id,
            value=value.value - max(score[value.value_id] for score in comparator_scores),
            unit="paired-fraction-correct-advantage",
        )
        for value in primary.complete_unit_accuracies
    )
    operands.append(
        continuous_power_operand(
            requirement_id="power.cantera.comparator-separation",
            family_id="comparator-separation",
            method=SelectiveDependenceResponsePowerMethod.ONE_SIDED_MINIMUM_MEAN,
            observations=paired_advantages,
            decision_boundary=analysis_freeze.minimum_comparator_advantage,
            native_unit="paired-complete-unit-accuracy-advantage",
        )
    )
    metric_unit_ids = tuple(value.value_id for value in primary.complete_unit_accuracies)
    for requirement_id, family_id, adverse_ids in (
        (
            "power.cantera.support-boundary",
            "support-boundary",
            primary.support_boundary_error_unit_ids,
        ),
        (
            "power.cantera.unsafe-false-admission",
            "unsafe-false-admission",
            primary.unsafe_false_admission_unit_ids,
        ),
        (
            "power.cantera.false-safe-hold",
            "false-safe-hold",
            primary.false_safe_hold_unit_ids,
        ),
    ):
        operands.append(
            zero_failure_power_operand(
                requirement_id=requirement_id,
                family_id=family_id,
                complete_unit_ids=metric_unit_ids,
                adverse_complete_unit_ids=adverse_ids,
                maximum_adverse_rate=analysis_freeze.maximum_adverse_rate,
            )
        )
    return tuple(sorted(operands, key=lambda value: value.requirement_id))


def freeze_cantera_development(
    *,
    panel: SelectiveDependenceResponseTargetPanel,
    design: CanteraReactionResponseCanteraDesign,
    preparation: SelectiveDependenceResponsePreparationDistributionFreeze,
    question: SelectiveDependenceResponseMethodQuestionFreeze,
    contamination: SelectiveDependenceResponseContaminationLedger,
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
    lineage: SelectiveDependenceResponseDevelopmentLineage,
) -> tuple[
    SelectiveDependenceResponseDevelopmentParent,
    SelectiveDependenceResponseSelectiveLawForecast | None,
    SelectiveDependenceResponseFiniteLawCalibration,
    SelectiveDependenceResponseSelectiveDependenceSignature,
    tuple[SelectiveDependenceResponseComparatorEncoding, ...],
    SelectiveDependenceResponsePowerDesignQualification,
    SelectiveDependenceResponseForecastSeparationReport,
    SelectiveDependenceResponseForecastChallengeQualification,
    SelectiveDependenceResponseDenominatorSelection,
]:
    if (
        panel.phase is not SelectiveDependenceResponsePhase.DEVELOPMENT
        or panel.target_id != design.target_id
        or panel.expected_complete_unit_ids != preparation.development_unit_ids
    ):
        raise ValueError("Cantera development freeze requires the exact issued development panel")
    if (
        analysis_freeze.target_id != design.target_id
        or analysis_freeze.design != ObjectIdentity.from_record(design.design_id, design)
        or analysis_freeze.preparation_freeze
        != ObjectIdentity.from_record(preparation.freeze_id, preparation)
    ):
        raise ValueError("Cantera analysis freeze differs from the issued target design")
    law, signature = analyze_cantera_development(panel, analysis_freeze)
    exchange_forecasts = analysis_freeze.exchange_forecasts
    comparators = fit_comparator_family(
        target_id=design.target_id,
        forecast=law.predictions,
        exchange_forecast=exchange_forecasts,
        development_unit_ids_sha256=signature.complete_unit_ids_sha256,
        hold_action_id=analysis_freeze.hold_action_id,
        mechanistic_prediction=_mechanistic,
    )
    denominator = assess_denominator_alternatives(
        panel=panel,
        law=law,
        signature=signature,
        comparators=comparators,
        exchange_forecasts=exchange_forecasts,
        hold_action_id=analysis_freeze.hold_action_id,
        receiver_ids=analysis_freeze.scored_receiver_ids,
        neutral_margin=analysis_freeze.neutral_margin,
        omitted_role_id=analysis_freeze.denominator_omitted_role_id,
        omitted_comparator_kind=analysis_freeze.denominator_omitted_comparator_kind,
        split_variable_id=analysis_freeze.denominator_split_variable_id,
        split_threshold=analysis_freeze.denominator_split_threshold,
    )
    relevant = comparator_relevant_cells(
        forecast=law.predictions,
        exchange_forecasts=exchange_forecasts,
        hold_action_id=analysis_freeze.hold_action_id,
    )
    separation = compare_forecast_to_comparators(
        target_id=design.target_id,
        forecast=law.predictions,
        exchange_forecast=exchange_forecasts,
        comparators=comparators,
        relevant_cells_by_kind=relevant,
        relevant_exchanges_by_kind=comparator_relevant_exchanges(exchange_forecasts),
    )
    power = qualify_power_design(
        target_id=design.target_id,
        candidate_panel_sizes=analysis_freeze.candidate_evaluation_panel_sizes,
        operands=_power_operands(
            panel=panel,
            law=law,
            comparators=comparators,
            separation=separation,
            analysis_freeze=analysis_freeze,
        ),
        familywise_alpha=analysis_freeze.power_familywise_alpha,
        target_power=analysis_freeze.target_power,
        maximum_nonevaluable_rate=analysis_freeze.maximum_nonevaluable_rate,
    )
    context_forecasts = derive_context_decisions_from_predictions(
        law.predictions,
        hold_action_id=analysis_freeze.hold_action_id,
        decision_id_prefix="forecast-decision.cantera",
    )
    challenge = qualify_forecast_challenge(
        target_id=design.target_id,
        exchange_forecasts=exchange_forecasts,
        context_forecasts=context_forecasts,
        target_sink_conflict_case_ids=target_sink_conflict_case_ids(
            panel,
            hold_action_id=analysis_freeze.hold_action_id,
            target_margin_id=analysis_freeze.target_margin_id,
            sink_margin_ids=analysis_freeze.sink_margin_ids,
        ),
    )
    evaluation_ids = preparation.evaluation_unit_ids[: power.selected_evaluation_unit_count or 0]
    parent = freeze_development_parent(
        target_id=design.target_id,
        design=design,
        design_id=design.design_id,
        method_question=question,
        method_question_id=question.freeze_id,
        contamination_ledger=contamination,
        contamination_ledger_id=contamination.ledger_id,
        preparation_freeze=preparation,
        preparation_freeze_id=preparation.freeze_id,
        analysis_freeze=analysis_freeze,
        analysis_freeze_id=analysis_freeze.freeze_id,
        construct_review=construct_review,
        construct_review_id=construct_review.attestation_id,
        selected_role_map=construct_review.role_map,
        role_map_selection=construct_review.role_map_selection,
        development_panel=panel,
        development_panel_id=panel.panel_id,
        lineage=lineage,
        law=law,
        signature=signature,
        comparators=comparators,
        power=power,
        separation=separation,
        challenge=challenge,
        denominator=denominator,
        evaluation_complete_unit_ids=evaluation_ids,
        reserve_complete_unit_ids=preparation.reserve_unit_ids,
    )
    cell_forecasts = []
    for prediction in law.predictions:
        denominator_id, history_id, action_id, horizon_id, receiver_id = _parse_cell(
            prediction.cell_id
        )
        cell_forecasts.append(
            SelectiveDependenceResponseCellForecast(
                cell_id=prediction.cell_id,
                denominator_id=denominator_id,
                history_id=history_id,
                action_id=action_id,
                receiver_id=receiver_id,
                horizon_id=horizon_id,
                expected_state=prediction.response_state,
                expected_fibre_admitted=prediction.fibre_admitted,
                expected_disposition=prediction.disposition,
            )
        )
    if not parent.eligible_for_evaluation:
        return (
            parent,
            None,
            law,
            signature,
            comparators,
            power,
            separation,
            challenge,
            denominator,
        )
    forecast = SelectiveDependenceResponseSelectiveLawForecast(
        forecast_id="forecast.cantera.selective-law",
        target_id=design.target_id,
        method_question=ObjectIdentity.from_record(question.freeze_id, question),
        contamination_ledger=ObjectIdentity.from_record(contamination.ledger_id, contamination),
        construct_review=ObjectIdentity.from_record(
            construct_review.attestation_id, construct_review
        ),
        selected_role_map=construct_review.role_map,
        role_map_selection=construct_review.role_map_selection,
        preparation_freeze=ObjectIdentity.from_record(preparation.freeze_id, preparation),
        analysis_freeze=ObjectIdentity.from_record(analysis_freeze.freeze_id, analysis_freeze),
        design=ObjectIdentity.from_record(design.design_id, design),
        development_result=ObjectIdentity.from_record(parent.parent_id, parent),
        denominator_selection=ObjectIdentity.from_record(denominator.selection_id, denominator),
        target_relation_id=SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID,
        exchange_familywise_alpha=power.familywise_alpha,
        exchange_interval_method=analysis_freeze.exchange_interval_method,
        exchange_forecasts=exchange_forecasts,
        cell_forecasts=tuple(sorted(cell_forecasts, key=lambda value: value.cell_id)),
        context_forecasts=context_forecasts,
        comparator_ids=tuple(sorted(value.encoding_id for value in comparators)),
        evaluation_unit_ids_sha256=digest_ids(evaluation_ids),
        predictively_distinguishing=separation.all_claim_relevant_comparators_separated,
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    return (
        parent,
        forecast,
        law,
        signature,
        comparators,
        power,
        separation,
        challenge,
        denominator,
    )


__all__ = ["freeze_cantera_development"]
