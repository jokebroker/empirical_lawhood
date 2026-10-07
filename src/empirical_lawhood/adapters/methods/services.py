"""Method-independent adequacy, falsification and numerical qualification."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations

import numpy as np

from empirical_lawhood.kernel.identification import (
    AdequacyCheckKind,
    AdequacyCheckResult,
    StructuralConvergenceResult,
    StructuralSignature,
    UncertaintyClass,
    UncertaintyComponent,
    UncertaintyDecomposition,
)
from empirical_lawhood.kernel.obligations import ObligationStatus
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.status import ReadinessStatus, ScientificStatus
from empirical_lawhood.kernel.systems import SystemSpec

from .contracts import (
    CandidateFit,
    DataSplit,
    IdentificationDataset,
    LawIdentificationConfig,
    LawObservation,
    ObservationRole,
)


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("scientific check produced a non-finite value")
    return Decimal(f"{value:.15g}")


def _metric(value_id: str, value: Decimal, unit: str = "1") -> NamedDecimal:
    return NamedDecimal(value_id=value_id, value=value, unit=unit)


def _check(
    *,
    check_id: str,
    kind: AdequacyCheckKind,
    passed: bool,
    reason_code: str,
    evidence_link_id: str,
    metrics: tuple[NamedDecimal, ...] = (),
    decisive: bool = False,
) -> AdequacyCheckResult:
    return AdequacyCheckResult(
        check_id=check_id,
        kind=kind,
        status=ObligationStatus.SATISFIED if passed else ObligationStatus.FAILED,
        decisive=decisive,
        metrics=tuple(sorted(metrics, key=lambda value: value.value_id)),
        reason_codes=() if passed else (reason_code,),
        evidence_link_ids=(evidence_link_id,),
    )


def _values(observation: LawObservation, field: str) -> dict[str, Decimal]:
    records = getattr(observation, field)
    return {record.value_id: record.value for record in records}


def _response_slopes(
    observations: tuple[LawObservation, ...],
    action_ids: tuple[str, ...],
    history_ids: tuple[str, ...],
    receiver_ids: tuple[str, ...],
) -> dict[tuple[str, str], Decimal]:
    feature_ids = (*action_ids, *history_ids)
    design = np.asarray(
        tuple(
            (1.0, *(float(observation.value(value_id)) for value_id in feature_ids))
            for observation in observations
        ),
        dtype=np.float64,
    )
    targets = np.asarray(
        tuple(
            tuple(float(observation.value(value_id)) for value_id in receiver_ids)
            for observation in observations
        ),
        dtype=np.float64,
    )
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(design, targets, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("response slope comparison requires full-rank action/history support")
    return {
        (action_id, receiver_id): _decimal(
            float(coefficients[1 + action_ids.index(action_id), receiver_ids.index(receiver_id)])
        )
        for action_id in action_ids
        for receiver_id in receiver_ids
    }


def _prediction_map(fit: CandidateFit, *, held_out: bool) -> dict[str, dict[str, Decimal]]:
    predictions = fit.held_out_predictions if held_out else fit.calibration_predictions
    return {
        prediction.observation_id: {
            value.value_id: value.value for value in prediction.predicted_receiver_values
        }
        for prediction in predictions
    }


class CoordinateAdequacyService:
    def evaluate(
        self,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        *,
        evidence_link_id: str,
    ) -> AdequacyCheckResult:
        required_denominator = set(dataset.relation.denominator_quantity_ids)
        required_history = set(dataset.relation.history_quantity_ids)
        required_actions = set(dataset.relation.action_quantity_ids)
        required_receivers = set(config.receiver_quantity_ids)
        complete = True
        feature_coordinates = {
            *dataset.relation.denominator_quantity_ids,
            *dataset.relation.history_quantity_ids,
            *dataset.relation.action_quantity_ids,
        }
        complete = complete and set(config.feature_quantity_ids).issubset(feature_coordinates)
        complete = complete and set(dataset.relation.action_quantity_ids).issubset(
            config.feature_quantity_ids
        )
        criteria = {
            criterion.receiver_quantity_id: criterion for criterion in config.receiver_criteria
        }
        action_units = {bound.quantity_id: bound.native_unit for bound in config.action_bounds}
        observed_units: dict[str, str] = {}
        for observation in dataset.selected(chart_id=config.chart_id):
            complete = complete and required_denominator.issubset(
                _values(observation, "denominator_values")
            )
            complete = complete and required_history.issubset(
                _values(observation, "history_values")
            )
            complete = complete and required_actions.issubset(_values(observation, "action_values"))
            complete = complete and required_receivers.issubset(
                _values(observation, "receiver_values")
            )
            for value in (
                *observation.denominator_values,
                *observation.history_values,
                *observation.action_values,
                *observation.receiver_values,
            ):
                previous = observed_units.setdefault(value.value_id, value.unit)
                complete = complete and previous == value.unit
            complete = complete and all(
                action_units.get(value.value_id) == value.unit
                for value in observation.action_values
                if value.value_id in required_actions
            )
            complete = complete and all(
                criteria[value.value_id].native_unit == value.unit
                for value in observation.receiver_values
                if value.value_id in required_receivers
            )
        return _check(
            check_id="coordinate-adequacy",
            kind=AdequacyCheckKind.COORDINATE_ADEQUACY,
            passed=complete,
            reason_code="declared-coordinate-missing",
            evidence_link_id=evidence_link_id,
        )


class RecurrenceService:
    def evaluate(
        self,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        *,
        evidence_link_id: str,
    ) -> AdequacyCheckResult:
        groups: dict[tuple[object, ...], set[str]] = defaultdict(set)
        for observation in dataset.selected(
            chart_id=config.chart_id,
            role=ObservationRole.PRIMARY,
        ):
            key = (
                observation.split,
                observation.denominator_cell_id,
                observation.numerical_view_id,
                tuple((value.value_id, value.value) for value in observation.action_values),
                tuple((value.value_id, value.value) for value in observation.history_values),
            )
            groups[key].add(observation.physical_unit_instance_id)
        minimum = min((len(unit_ids) for unit_ids in groups.values()), default=0)
        return _check(
            check_id="within-cell-recurrence",
            kind=AdequacyCheckKind.WITHIN_CELL_RECURRENCE,
            passed=minimum >= config.minimum_recurrence_units,
            reason_code="within-cell-recurrence-insufficient",
            evidence_link_id=evidence_link_id,
            metrics=(_metric("minimum-recurrence-units", Decimal(minimum)),),
            decisive=True,
        )


class OneFactorExchangeService:
    def evaluate(
        self,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        *,
        evidence_link_id: str,
    ) -> AdequacyCheckResult:
        primary = dataset.selected(chart_id=config.chart_id, role=ObservationRole.PRIMARY)
        expected_pairs = {
            (action_id, receiver_id)
            for action_id in dataset.relation.action_quantity_ids
            for receiver_id in config.receiver_quantity_ids
        }
        difference_metrics: list[NamedDecimal] = []
        valid_coordinate_exchange = True
        for exchange in dataset.one_factor_exchanges:
            tolerances = {
                (value.action_quantity_id, value.receiver_quantity_id): value
                for value in exchange.slope_tolerances
            }
            if set(tolerances) != expected_pairs:
                valid_coordinate_exchange = False
            left = tuple(
                observation
                for observation in primary
                if observation.denominator_cell_id == exchange.left_denominator_cell_id
            )
            right = tuple(
                observation
                for observation in primary
                if observation.denominator_cell_id == exchange.right_denominator_cell_id
            )
            if not left or not right:
                valid_coordinate_exchange = False
                continue
            left_values = _values(left[0], "denominator_values")
            right_values = _values(right[0], "denominator_values")
            changed = {
                quantity_id
                for quantity_id in set(left_values) | set(right_values)
                if left_values.get(quantity_id) != right_values.get(quantity_id)
            }
            valid_coordinate_exchange = valid_coordinate_exchange and changed == {
                exchange.exchanged_factor_id
            }
            views = sorted({observation.numerical_view_id for observation in (*left, *right)})
            for split in DataSplit:
                for view_id in views:
                    left_view = tuple(
                        observation
                        for observation in left
                        if observation.split is split and observation.numerical_view_id == view_id
                    )
                    right_view = tuple(
                        observation
                        for observation in right
                        if observation.split is split and observation.numerical_view_id == view_id
                    )
                    if not left_view or not right_view:
                        valid_coordinate_exchange = False
                        continue
                    for action_id, receiver_id in sorted(expected_pairs):
                        tolerance = tolerances.get((action_id, receiver_id))
                        if tolerance is None:
                            continue
                        try:
                            left_slopes = _response_slopes(
                                left_view,
                                dataset.relation.action_quantity_ids,
                                dataset.relation.history_quantity_ids,
                                config.receiver_quantity_ids,
                            )
                            right_slopes = _response_slopes(
                                right_view,
                                dataset.relation.action_quantity_ids,
                                dataset.relation.history_quantity_ids,
                                config.receiver_quantity_ids,
                            )
                        except ValueError:
                            valid_coordinate_exchange = False
                            continue
                        difference = abs(
                            left_slopes[(action_id, receiver_id)]
                            - right_slopes[(action_id, receiver_id)]
                        )
                        difference_metrics.append(
                            _metric(
                                (
                                    f"exchange-slope-difference.{exchange.exchange_id}."
                                    f"{split.value.lower()}.{view_id}.{tolerance.tolerance_id}"
                                ),
                                difference,
                                tolerance.native_unit,
                            )
                        )
                        valid_coordinate_exchange = (
                            valid_coordinate_exchange
                            and difference <= tolerance.maximum_absolute_difference
                        )
        return _check(
            check_id="one-factor-exchange",
            kind=AdequacyCheckKind.ONE_FACTOR_EXCHANGE,
            passed=valid_coordinate_exchange and bool(difference_metrics),
            reason_code="one-factor-exchange-failed",
            evidence_link_id=evidence_link_id,
            metrics=(
                _metric("exchange-comparison-count", Decimal(len(difference_metrics))),
                *difference_metrics,
            ),
            decisive=True,
        )


class SupportService:
    def evaluate(
        self,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        *,
        evidence_link_id: str,
    ) -> AdequacyCheckResult:
        calibration = dataset.selected(
            chart_id=config.chart_id,
            split=DataSplit.CALIBRATION,
            role=ObservationRole.PRIMARY,
        )
        held_out = dataset.selected(
            chart_id=config.chart_id,
            split=DataSplit.HELD_OUT,
            role=ObservationRole.PRIMARY,
        )
        bound_by_quantity = {bound.quantity_id: bound for bound in config.action_bounds}
        levels = {
            tuple(
                observation.value(action_id) for action_id in dataset.relation.action_quantity_ids
            )
            for observation in calibration
        }
        supported = len(levels) >= config.minimum_action_levels
        calibration_ranges: dict[str, tuple[Decimal, Decimal]] = {}
        for action_id in dataset.relation.action_quantity_ids:
            values = tuple(observation.value(action_id) for observation in calibration)
            if values:
                calibration_ranges[action_id] = (min(values), max(values))
        for observation in (*calibration, *held_out):
            for action_id in dataset.relation.action_quantity_ids:
                bound = bound_by_quantity.get(action_id)
                if bound is None:
                    supported = False
                    continue
                value = observation.value(action_id)
                supported = supported and (bound.lower is None or value >= bound.lower)
                supported = supported and (bound.upper is None or value <= bound.upper)
                if observation.split is DataSplit.HELD_OUT:
                    calibration_range = calibration_ranges.get(action_id)
                    supported = supported and calibration_range is not None
                    if calibration_range is not None:
                        supported = (
                            supported and calibration_range[0] <= value <= calibration_range[1]
                        )
        return _check(
            check_id="action-support",
            kind=AdequacyCheckKind.SUPPORT,
            passed=supported,
            reason_code="held-out-action-outside-support",
            evidence_link_id=evidence_link_id,
            metrics=(_metric("calibration-action-levels", Decimal(len(levels))),),
            decisive=True,
        )


class ClosureMemoryService:
    def evaluate(
        self,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        fit: CandidateFit,
        *,
        evidence_link_id: str,
    ) -> AdequacyCheckResult:
        held_out = dataset.selected(
            chart_id=config.chart_id,
            split=DataSplit.HELD_OUT,
            role=ObservationRole.PRIMARY,
            numerical_view_id=fit.model.numerical_view_id,
        )
        predicted = _prediction_map(fit, held_out=True)
        correlations: list[float] = []
        for history_id in dataset.relation.history_quantity_ids:
            history = np.asarray(tuple(float(value.value(history_id)) for value in held_out))
            for receiver_id in config.receiver_quantity_ids:
                residual = np.asarray(
                    tuple(
                        float(
                            value.value(receiver_id) - predicted[value.observation_id][receiver_id]
                        )
                        for value in held_out
                    )
                )
                if np.std(history) == 0 or np.std(residual) == 0:
                    correlations.append(0.0)
                else:
                    correlations.append(abs(float(np.corrcoef(history, residual)[0, 1])))
        maximum = max(correlations, default=0.0)
        passed = maximum <= float(config.maximum_residual_history_correlation)
        if set(dataset.relation.history_quantity_ids).difference(config.feature_quantity_ids):
            passed = False
        return _check(
            check_id="closure-memory-adequacy",
            kind=AdequacyCheckKind.CLOSURE_MEMORY,
            passed=passed,
            reason_code="residual-history-retained",
            evidence_link_id=evidence_link_id,
            metrics=(_metric("maximum-residual-history-correlation", _decimal(maximum)),),
            decisive=True,
        )


class CalibrationService:
    def evaluate(
        self,
        config: LawIdentificationConfig,
        fit: CandidateFit,
        *,
        evidence_link_id: str,
    ) -> AdequacyCheckResult:
        criteria = {
            criterion.receiver_quantity_id: criterion for criterion in config.receiver_criteria
        }
        rmse_metrics = tuple(
            metric for metric in fit.metrics if metric.value_id.startswith("held-out-rmse.")
        )
        passed = all(
            metric.value
            <= criteria[metric.value_id.removeprefix("held-out-rmse.")].maximum_held_out_rmse
            for metric in rmse_metrics
        ) and len(rmse_metrics) == len(criteria)
        return _check(
            check_id="held-out-calibration",
            kind=AdequacyCheckKind.HELD_OUT_CALIBRATION,
            passed=passed,
            reason_code="held-out-calibration-failed",
            evidence_link_id=evidence_link_id,
            metrics=rmse_metrics,
            decisive=True,
        )


class DecisiveFalsifierService:
    def evaluate(
        self,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        fit: CandidateFit,
        *,
        evidence_link_id: str,
    ) -> AdequacyCheckResult:
        wrong_action = dataset.selected(
            chart_id=config.chart_id,
            split=DataSplit.HELD_OUT,
            role=ObservationRole.WRONG_ACTION,
            numerical_view_id=fit.model.numerical_view_id,
        )
        held_out = dataset.selected(
            chart_id=config.chart_id,
            split=DataSplit.HELD_OUT,
            role=ObservationRole.PRIMARY,
            numerical_view_id=fit.model.numerical_view_id,
        )
        calibration = dataset.selected(
            chart_id=config.chart_id,
            split=DataSplit.CALIBRATION,
            role=ObservationRole.PRIMARY,
            numerical_view_id=fit.model.numerical_view_id,
        )
        criteria = {
            criterion.receiver_quantity_id: criterion for criterion in config.receiver_criteria
        }
        means = {
            receiver_id: sum(
                (observation.value(receiver_id) for observation in calibration),
                Decimal(0),
            )
            / Decimal(len(calibration))
            for receiver_id in config.receiver_quantity_ids
        }
        metrics: list[NamedDecimal] = [_metric("wrong-action-count", Decimal(len(wrong_action)))]
        passed = bool(wrong_action)
        fit_metrics = {metric.value_id: metric for metric in fit.metrics}
        for receiver_id in config.receiver_quantity_ids:
            criterion = criteria[receiver_id]
            wrong_effect = max(
                (abs(observation.value(receiver_id)) for observation in wrong_action),
                default=Decimal(0),
            )
            baseline_squared = tuple(
                float(observation.value(receiver_id) - means[receiver_id]) ** 2
                for observation in held_out
            )
            baseline_rmse = _decimal(float(np.sqrt(np.mean(baseline_squared))))
            held_out_rmse = fit_metrics[f"held-out-rmse.{receiver_id}"].value
            improvement = baseline_rmse - held_out_rmse
            passed = (
                passed
                and wrong_effect <= criterion.maximum_wrong_action_effect
                and improvement >= criterion.minimum_baseline_improvement
            )
            metrics.extend(
                (
                    _metric(
                        f"baseline-improvement.{receiver_id}",
                        improvement,
                        criterion.native_unit,
                    ),
                    _metric(
                        f"wrong-action-effect.{receiver_id}",
                        wrong_effect,
                        criterion.native_unit,
                    ),
                )
            )
        return _check(
            check_id="decisive-falsifier",
            kind=AdequacyCheckKind.DECISIVE_FALSIFIER,
            passed=passed,
            reason_code="wrong-action-or-baseline-falsifier-failed",
            evidence_link_id=evidence_link_id,
            metrics=tuple(metrics),
            decisive=True,
        )


class NumericalQualifier:
    def qualify(
        self,
        system: SystemSpec,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        fits: tuple[CandidateFit, ...],
        *,
        evidence_links: tuple[EvidenceLink, ...],
    ) -> StructuralConvergenceResult:
        if len(fits) < 2:
            raise ValueError("numerical qualifier requires at least two fitted views")
        expected_views = {view.view_id for view in system.numerical_views}
        observed_views = {fit.model.numerical_view_id for fit in fits}
        if observed_views != expected_views or len(observed_views) != len(fits):
            raise ValueError("numerical qualifier requires exactly every declared numerical view")
        signatures = tuple(
            sorted(
                (self._signature(system, config, fit) for fit in fits),
                key=lambda value: value.signature_id,
            )
        )
        coefficient_maps = {
            fit.model.numerical_view_id: {
                (coefficient.receiver_quantity_id, coefficient.term_id): coefficient
                for coefficient in fit.model.coefficients
            }
            for fit in fits
        }
        tolerances = {
            (value.receiver_quantity_id, value.term_id): value
            for value in config.structural_tolerances
        }
        comparable = all(set(values) == set(tolerances) for values in coefficient_maps.values())
        difference_metrics: list[NamedDecimal] = []
        differences_pass = comparable
        if comparable:
            for (left_view, left), (right_view, right) in combinations(coefficient_maps.items(), 2):
                for key, tolerance in tolerances.items():
                    if (
                        left[key].native_unit != tolerance.native_unit
                        or right[key].native_unit != tolerance.native_unit
                    ):
                        differences_pass = False
                        continue
                    difference = abs(left[key].value - right[key].value)
                    difference_metrics.append(
                        _metric(
                            (
                                f"coefficient-difference.{left_view}.{right_view}."
                                f"{tolerance.tolerance_id}"
                            ),
                            difference,
                            tolerance.native_unit,
                        )
                    )
                    differences_pass = (
                        differences_pass and difference <= tolerance.maximum_refinement_difference
                    )
        numerical_metrics, aligned = self._prediction_disagreement(dataset, config, fits)
        signature_shapes = {
            (
                signature.response_rank,
                signature.response_direction_ids,
                signature.curved_term_ids,
                signature.retained_history_ids,
            )
            for signature in signatures
        }
        stable = comparable and differences_pass and aligned and len(signature_shapes) == 1
        unstable_ids = () if stable else ("response-structure",)
        stable_ids = (
            ("curvature", "memory", "response-direction", "response-rank") if stable else ()
        )
        return StructuralConvergenceResult(
            result_id=f"convergence.{config.config_id}",
            system_id=system.system_id,
            relation_id=system.relation.relation_id,
            signatures=signatures,
            stable_structure_ids=stable_ids,
            unstable_structure_ids=unstable_ids,
            metrics=tuple(
                sorted((*difference_metrics, *numerical_metrics), key=lambda value: value.value_id)
            ),
            status=(ScientificStatus.SUPPORTED if stable else ScientificStatus.NOT_SUPPORTED),
            evidence_links=tuple(sorted(evidence_links, key=lambda value: value.link_id)),
            reason_codes=() if stable else ("structure-changed-under-refinement",),
        )

    @staticmethod
    def _prediction_disagreement(
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        fits: tuple[CandidateFit, ...],
    ) -> tuple[tuple[NamedDecimal, ...], bool]:
        values_by_view: dict[str, dict[tuple[object, ...], dict[str, Decimal]]] = {}
        for fit in fits:
            observations = dataset.selected(
                chart_id=config.chart_id,
                split=DataSplit.HELD_OUT,
                role=ObservationRole.PRIMARY,
                numerical_view_id=fit.model.numerical_view_id,
            )
            predictions = _prediction_map(fit, held_out=True)
            view_values: dict[tuple[object, ...], dict[str, Decimal]] = {}
            for observation in observations:
                key = (
                    observation.physical_unit_instance_id,
                    observation.denominator_cell_id,
                    tuple(
                        (value.value_id, value.value) for value in observation.denominator_values
                    ),
                    tuple((value.value_id, value.value) for value in observation.history_values),
                    tuple((value.value_id, value.value) for value in observation.action_values),
                )
                if key in view_values:
                    raise ValueError("numerical view contains duplicate aligned held-out inputs")
                view_values[key] = predictions[observation.observation_id]
            values_by_view[fit.model.numerical_view_id] = view_values
        key_sets = tuple(set(values) for values in values_by_view.values())
        aligned = bool(key_sets) and all(keys == key_sets[0] for keys in key_sets[1:])
        criteria = {
            criterion.receiver_quantity_id: criterion for criterion in config.receiver_criteria
        }
        metrics: list[NamedDecimal] = []
        for receiver_id in config.receiver_quantity_ids:
            maximum = Decimal(0)
            if aligned:
                for aligned_key in key_sets[0]:
                    predicted_values = tuple(
                        values[aligned_key][receiver_id] for values in values_by_view.values()
                    )
                    maximum = max(
                        maximum,
                        *(abs(left - right) for left, right in combinations(predicted_values, 2)),
                    )
            metrics.append(
                _metric(
                    f"numerical-bound.{receiver_id}",
                    maximum,
                    criteria[receiver_id].native_unit,
                )
            )
        return tuple(metrics), aligned

    @staticmethod
    def _signature(
        system: SystemSpec,
        config: LawIdentificationConfig,
        fit: CandidateFit,
    ) -> StructuralSignature:
        action_ids = {bound.quantity_id for bound in config.action_bounds}
        tolerances = {
            (value.receiver_quantity_id, value.term_id): value
            for value in config.structural_tolerances
        }
        linear = tuple(
            coefficient
            for coefficient in fit.model.coefficients
            if coefficient.term_id.startswith("linear.")
            and coefficient.term_id.removeprefix("linear.") in action_ids
            and abs(coefficient.value)
            > tolerances[
                (coefficient.receiver_quantity_id, coefficient.term_id)
            ].zero_absolute_tolerance
        )
        directions = tuple(
            sorted(
                f"{value.receiver_quantity_id}:{value.term_id}:"
                f"{'positive' if value.value > 0 else 'negative'}"
                for value in linear
            )
        )
        curved = tuple(
            sorted(
                f"{coefficient.receiver_quantity_id}:{coefficient.term_id}"
                for coefficient in fit.model.coefficients
                if coefficient.term_id.startswith("quadratic.")
                and abs(coefficient.value)
                > tolerances[
                    (coefficient.receiver_quantity_id, coefficient.term_id)
                ].zero_absolute_tolerance
            )
        )
        history = tuple(
            sorted(
                history_id
                for history_id in system.relation.history_quantity_ids
                if any(
                    coefficient.term_id == f"linear.{history_id}"
                    and abs(coefficient.value)
                    > tolerances[
                        (coefficient.receiver_quantity_id, coefficient.term_id)
                    ].zero_absolute_tolerance
                    for coefficient in fit.model.coefficients
                )
            )
        )
        action_matrix = np.asarray(
            tuple(
                tuple(
                    float(
                        next(
                            (
                                (
                                    coefficient.value
                                    if abs(coefficient.value)
                                    > tolerances[
                                        (
                                            coefficient.receiver_quantity_id,
                                            coefficient.term_id,
                                        )
                                    ].zero_absolute_tolerance
                                    else Decimal(0)
                                )
                                for coefficient in fit.model.coefficients
                                if coefficient.receiver_quantity_id == receiver_id
                                and coefficient.term_id == f"linear.{action_id}"
                            ),
                            Decimal(0),
                        )
                    )
                    for action_id in sorted(action_ids)
                )
                for receiver_id in config.receiver_quantity_ids
            ),
            dtype=np.float64,
        )
        column_scale = np.max(np.abs(action_matrix), axis=0)
        column_scale[column_scale == 0] = 1
        normalized = action_matrix / column_scale
        row_scale = np.max(np.abs(normalized), axis=1)
        row_scale[row_scale == 0] = 1
        normalized = normalized / row_scale[:, np.newaxis]
        response_rank = int(np.linalg.matrix_rank(normalized))
        return StructuralSignature(
            signature_id=f"signature.{fit.model.numerical_view_id}",
            numerical_view_id=fit.model.numerical_view_id,
            response_rank=response_rank,
            response_direction_ids=directions,
            curved_term_ids=curved,
            retained_history_ids=history,
        )


class UncertaintyService:
    def decompose(
        self,
        config: LawIdentificationConfig,
        fit: CandidateFit,
        convergence: StructuralConvergenceResult,
        *,
        evidence_link_id: str,
        transport_bounds: tuple[NamedDecimal, ...] | None = None,
    ) -> UncertaintyDecomposition:
        numerical_bounds = tuple(
            metric
            for metric in convergence.metrics
            if metric.value_id.startswith("numerical-bound.")
        )
        aleatoric_bounds = tuple(
            _metric(
                f"aleatoric-bound.{criterion.receiver_quantity_id}",
                criterion.aleatoric_uncertainty_bound,
                criterion.native_unit,
            )
            for criterion in config.receiver_criteria
        )
        epistemic_bounds = tuple(
            _metric(
                f"epistemic-bound.{metric.value_id.removeprefix('held-out-rmse.')}",
                metric.value,
                metric.unit,
            )
            for metric in fit.metrics
            if metric.value_id.startswith("held-out-rmse.")
        )
        observation_bounds = tuple(
            _metric(
                f"observation-bound.{criterion.receiver_quantity_id}",
                criterion.observation_uncertainty_bound,
                criterion.native_unit,
            )
            for criterion in config.receiver_criteria
        )
        components = (
            self._bounded(
                UncertaintyClass.NUMERICAL,
                numerical_bounds,
                evidence_link_id,
                resolved=convergence.status is ScientificStatus.SUPPORTED,
            ),
            self._bounded(
                UncertaintyClass.ALEATORIC,
                aleatoric_bounds,
                evidence_link_id,
                resolved=True,
            ),
            self._bounded(
                UncertaintyClass.EPISTEMIC,
                epistemic_bounds,
                evidence_link_id,
                resolved=True,
            ),
            self._optional(
                UncertaintyClass.TRANSPORT,
                transport_bounds,
                evidence_link_id,
            ),
            self._bounded(
                UncertaintyClass.OBSERVATION,
                observation_bounds,
                evidence_link_id,
                resolved=True,
            ),
        )
        return UncertaintyDecomposition(
            decomposition_id=f"uncertainty.{fit.fit_id}",
            components=tuple(sorted(components, key=lambda value: value.component_id)),
        )

    @staticmethod
    def _bounded(
        uncertainty_class: UncertaintyClass,
        bounds: tuple[NamedDecimal, ...],
        evidence_link_id: str,
        *,
        resolved: bool,
    ) -> UncertaintyComponent:
        return UncertaintyComponent(
            component_id=f"uncertainty.{uncertainty_class.value.lower()}",
            uncertainty_class=uncertainty_class,
            status=(ObligationStatus.SATISFIED if resolved else ObligationStatus.FAILED),
            bounds=tuple(sorted(bounds, key=lambda value: value.value_id)),
            limitation_codes=() if resolved else ("structural-instability",),
            evidence_link_ids=(evidence_link_id,),
        )

    @classmethod
    def _optional(
        cls,
        uncertainty_class: UncertaintyClass,
        bounds: tuple[NamedDecimal, ...] | None,
        evidence_link_id: str,
    ) -> UncertaintyComponent:
        if bounds is None:
            return UncertaintyComponent(
                component_id=f"uncertainty.{uncertainty_class.value.lower()}",
                uncertainty_class=uncertainty_class,
                status=ObligationStatus.NOT_APPLICABLE,
                bounds=(),
                limitation_codes=(),
                evidence_link_ids=(),
            )
        return cls._bounded(
            uncertainty_class,
            bounds,
            evidence_link_id,
            resolved=True,
        )


@dataclass(frozen=True, slots=True)
class AdequacyServices:
    coordinates: CoordinateAdequacyService = CoordinateAdequacyService()
    recurrence: RecurrenceService = RecurrenceService()
    exchanges: OneFactorExchangeService = OneFactorExchangeService()
    support: SupportService = SupportService()
    closure: ClosureMemoryService = ClosureMemoryService()
    calibration: CalibrationService = CalibrationService()
    falsifier: DecisiveFalsifierService = DecisiveFalsifierService()

    def evaluate(
        self,
        system: SystemSpec,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        fit: CandidateFit,
        convergence: StructuralConvergenceResult,
        *,
        evidence_link_id: str,
    ) -> tuple[AdequacyCheckResult, ...]:
        structural = _check(
            check_id="structural-convergence",
            kind=AdequacyCheckKind.STRUCTURAL_CONVERGENCE,
            passed=convergence.status is ScientificStatus.SUPPORTED,
            reason_code="structure-changed-under-refinement",
            evidence_link_id=evidence_link_id,
            metrics=convergence.metrics,
            decisive=True,
        )
        computability = _check(
            check_id="computability-envelope",
            kind=AdequacyCheckKind.COMPUTABILITY,
            passed=(
                convergence.system_id == system.system_id
                and dataset.system == ObjectIdentity.from_record(system.system_id, system)
                and system.readiness is ReadinessStatus.READY
                and len(
                    {
                        view.computability_envelope_id
                        for view in system.numerical_views
                        if view.view_id
                        in {signature.numerical_view_id for signature in convergence.signatures}
                    }
                )
                == 1
            ),
            reason_code="computability-envelope-not-ready",
            evidence_link_id=evidence_link_id,
            decisive=True,
        )
        visibility = _check(
            check_id="evidence-visibility",
            kind=AdequacyCheckKind.EVIDENCE_VISIBILITY,
            passed=dataset.visibility_ceiling.is_promotable,
            reason_code="outcome-visible-evidence-non-promotable",
            evidence_link_id=evidence_link_id,
            decisive=True,
        )
        checks = (
            self.support.evaluate(dataset, config, evidence_link_id=evidence_link_id),
            self.calibration.evaluate(config, fit, evidence_link_id=evidence_link_id),
            self.closure.evaluate(
                dataset,
                config,
                fit,
                evidence_link_id=evidence_link_id,
            ),
            self.coordinates.evaluate(dataset, config, evidence_link_id=evidence_link_id),
            computability,
            self.falsifier.evaluate(
                dataset,
                config,
                fit,
                evidence_link_id=evidence_link_id,
            ),
            self.exchanges.evaluate(dataset, config, evidence_link_id=evidence_link_id),
            structural,
            self.recurrence.evaluate(dataset, config, evidence_link_id=evidence_link_id),
            visibility,
        )
        return tuple(sorted(checks, key=lambda value: value.check_id))
