"""Small transparent baseline law families behind one typed fit contract."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.references import NamedDecimal

from .contracts import (
    CandidateFit,
    DataSplit,
    IdentificationDataset,
    LawIdentificationConfig,
    LawIdentifier,
    LawObservation,
    ModelCoefficient,
    ObservationRole,
    ParametricLawModel,
    PredictionRecord,
)


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("law method produced a non-finite numeric value")
    return Decimal(f"{value:.15g}")


def _term_ids(config: LawIdentificationConfig) -> tuple[str, ...]:
    terms = ["intercept"]
    terms.extend(f"linear.{feature_id}" for feature_id in config.feature_quantity_ids)
    if config.polynomial_degree == 2:
        terms.extend(f"quadratic.{feature_id}" for feature_id in config.feature_quantity_ids)
    return tuple(sorted(terms))


def _design_row(
    observation: LawObservation,
    term_ids: tuple[str, ...],
) -> tuple[float, ...]:
    values = []
    for term_id in term_ids:
        if term_id == "intercept":
            values.append(1.0)
        elif term_id.startswith("linear."):
            values.append(float(observation.value(term_id.removeprefix("linear."))))
        elif term_id.startswith("quadratic."):
            value = float(observation.value(term_id.removeprefix("quadratic.")))
            values.append(value * value)
        else:
            raise ValueError(f"unsupported parametric term: {term_id}")
    return tuple(values)


def _matrix(
    observations: tuple[LawObservation, ...],
    term_ids: tuple[str, ...],
) -> npt.NDArray[np.float64]:
    return np.asarray(
        tuple(_design_row(observation, term_ids) for observation in observations),
        dtype=np.float64,
    )


def _targets(
    observations: tuple[LawObservation, ...],
    receiver_ids: tuple[str, ...],
) -> npt.NDArray[np.float64]:
    return np.asarray(
        tuple(
            tuple(float(observation.value(receiver_id)) for receiver_id in receiver_ids)
            for observation in observations
        ),
        dtype=np.float64,
    )


def _coefficient_matrix(model: ParametricLawModel) -> npt.NDArray[np.float64]:
    values = {
        (coefficient.receiver_quantity_id, coefficient.term_id): float(coefficient.value)
        for coefficient in model.coefficients
    }
    return np.asarray(
        tuple(
            tuple(values[(receiver_id, term_id)] for receiver_id in model.receiver_quantity_ids)
            for term_id in model.term_ids
        ),
        dtype=np.float64,
    )


def predict_model(
    model: ParametricLawModel,
    observations: tuple[LawObservation, ...],
) -> tuple[PredictionRecord, ...]:
    design = _matrix(observations, model.term_ids)
    predictions = design @ _coefficient_matrix(model)
    units = {
        value.value_id: value.unit
        for value in observations[0].receiver_values
        if value.value_id in model.receiver_quantity_ids
    }
    return tuple(
        PredictionRecord(
            observation_id=observation.observation_id,
            predicted_receiver_values=tuple(
                NamedDecimal(
                    value_id=receiver_id,
                    value=_decimal(float(predictions[row_index, column_index])),
                    unit=units[receiver_id],
                )
                for column_index, receiver_id in enumerate(model.receiver_quantity_ids)
            ),
        )
        for row_index, observation in enumerate(observations)
    )


def _rmse_by_receiver(
    observations: tuple[LawObservation, ...],
    predictions: tuple[PredictionRecord, ...],
    receiver_ids: tuple[str, ...],
) -> dict[str, Decimal]:
    predicted = {
        prediction.observation_id: {
            value.value_id: value.value for value in prediction.predicted_receiver_values
        }
        for prediction in predictions
    }
    return {
        receiver_id: _decimal(
            float(
                np.sqrt(
                    np.mean(
                        np.asarray(
                            tuple(
                                float(
                                    observation.value(receiver_id)
                                    - predicted[observation.observation_id][receiver_id]
                                )
                                ** 2
                                for observation in observations
                            ),
                            dtype=np.float64,
                        )
                    )
                )
            )
        )
        for receiver_id in receiver_ids
    }


def _coefficient_unit(
    receiver_unit: str,
    term_id: str,
    feature_units: dict[str, str],
) -> str:
    if term_id == "intercept":
        return receiver_unit
    if term_id.startswith("linear."):
        return f"{receiver_unit}/({feature_units[term_id.removeprefix('linear.')]})"
    if term_id.startswith("quadratic."):
        return f"{receiver_unit}/({feature_units[term_id.removeprefix('quadratic.')]}^2)"
    raise ValueError(f"unsupported coefficient term: {term_id}")


class _ParametricIdentifier:
    method_key: str
    method_version = "1.0.0"
    method_kind: LawMethodKind
    required_degree: int

    def _validate_config(
        self,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
    ) -> None:
        if (
            config.method_key != self.method_key
            or config.method_version != self.method_version
            or config.method_kind is not self.method_kind
        ):
            raise ValueError("law method configuration selects another registered method")
        if config.polynomial_degree != self.required_degree:
            raise ValueError("law method configuration uses an unsupported degree")
        if not set(dataset.relation.action_quantity_ids).issubset(config.feature_quantity_ids):
            raise ValueError("law method features omit the native action chart")
        if not set(config.receiver_quantity_ids).issubset(dataset.relation.receiver_quantity_ids):
            raise ValueError("law method requests a receiver outside the relation")

    def fit(
        self,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
        *,
        numerical_view_id: str,
    ) -> CandidateFit:
        self._validate_config(dataset, config)
        calibration = dataset.selected(
            chart_id=config.chart_id,
            split=DataSplit.CALIBRATION,
            role=ObservationRole.PRIMARY,
            numerical_view_id=numerical_view_id,
        )
        held_out = dataset.selected(
            chart_id=config.chart_id,
            split=DataSplit.HELD_OUT,
            role=ObservationRole.PRIMARY,
            numerical_view_id=numerical_view_id,
        )
        if not calibration or not held_out:
            raise ValueError("law method requires primary calibration and held-out observations")
        term_ids = _term_ids(config)
        design = _matrix(calibration, term_ids)
        targets = _targets(calibration, config.receiver_quantity_ids)
        feature_units = {
            feature_id: next(
                value.unit
                for value in (
                    *calibration[0].denominator_values,
                    *calibration[0].history_values,
                    *calibration[0].action_values,
                )
                if value.value_id == feature_id
            )
            for feature_id in config.feature_quantity_ids
        }
        receiver_units = {
            criterion.receiver_quantity_id: criterion.native_unit
            for criterion in config.receiver_criteria
        }
        observed_receiver_units = {
            value.value_id: value.unit for value in calibration[0].receiver_values
        }
        if any(
            observed_receiver_units.get(receiver_id) != receiver_units[receiver_id]
            for receiver_id in config.receiver_quantity_ids
        ):
            raise ValueError("receiver criterion unit differs from observed native unit")
        coefficients, _residuals, rank, _singular = np.linalg.lstsq(
            design,
            targets,
            rcond=None,
        )
        if rank < len(term_ids):
            raise ValueError("law method design matrix is rank deficient")
        coefficient_records = tuple(
            sorted(
                (
                    ModelCoefficient(
                        coefficient_id=f"coef.{receiver_id}.{term_id}",
                        receiver_quantity_id=receiver_id,
                        term_id=term_id,
                        value=_decimal(float(coefficients[term_index, receiver_index])),
                        native_unit=_coefficient_unit(
                            receiver_units[receiver_id],
                            term_id,
                            feature_units,
                        ),
                    )
                    for term_index, term_id in enumerate(term_ids)
                    for receiver_index, receiver_id in enumerate(config.receiver_quantity_ids)
                ),
                key=lambda value: value.coefficient_id,
            )
        )
        model = ParametricLawModel(
            model_id=(
                f"model.{dataset.dataset_id}.{config.chart_id}."
                f"{numerical_view_id}.{self.method_kind.value.lower()}"
            ),
            method_key=self.method_key,
            method_version=self.method_version,
            method_kind=self.method_kind,
            chart_id=config.chart_id,
            feature_quantity_ids=config.feature_quantity_ids,
            receiver_quantity_ids=config.receiver_quantity_ids,
            term_ids=term_ids,
            coefficients=coefficient_records,
            numerical_view_id=numerical_view_id,
        )
        calibration_predictions = predict_model(model, calibration)
        held_out_predictions = predict_model(model, held_out)
        calibration_rmse = _rmse_by_receiver(
            calibration,
            calibration_predictions,
            config.receiver_quantity_ids,
        )
        held_out_rmse = _rmse_by_receiver(
            held_out,
            held_out_predictions,
            config.receiver_quantity_ids,
        )
        return CandidateFit(
            fit_id=f"fit.{model.model_id}",
            model=model,
            calibration_predictions=calibration_predictions,
            held_out_predictions=held_out_predictions,
            metrics=tuple(
                sorted(
                    (
                        NamedDecimal(
                            value_id=f"calibration-rmse.{receiver_id}",
                            value=calibration_rmse[receiver_id],
                            unit=receiver_units[receiver_id],
                        )
                        for receiver_id in config.receiver_quantity_ids
                    ),
                    key=lambda value: value.value_id,
                )
            )
            + (
                NamedDecimal(
                    value_id="design-rank",
                    value=Decimal(int(rank)),
                    unit="1",
                ),
            )
            + tuple(
                sorted(
                    (
                        NamedDecimal(
                            value_id=f"held-out-rmse.{receiver_id}",
                            value=held_out_rmse[receiver_id],
                            unit=receiver_units[receiver_id],
                        )
                        for receiver_id in config.receiver_quantity_ids
                    ),
                    key=lambda value: value.value_id,
                )
            ),
        )


class LocalLinearIdentifier(_ParametricIdentifier):
    method_key = "baseline.local-linear"
    method_kind = LawMethodKind.LOCAL_LINEAR
    required_degree = 1


class LocalStateSpaceIdentifier(_ParametricIdentifier):
    method_key = "baseline.local-state-space"
    method_kind = LawMethodKind.LOCAL_STATE_SPACE
    required_degree = 1

    def _validate_config(
        self,
        dataset: IdentificationDataset,
        config: LawIdentificationConfig,
    ) -> None:
        super()._validate_config(dataset, config)
        if not dataset.relation.history_quantity_ids:
            raise ValueError("state-space identification requires retained history")
        if not set(dataset.relation.history_quantity_ids).issubset(config.feature_quantity_ids):
            raise ValueError("state-space features omit declared retained history")


class NonlinearLocalIdentifier(_ParametricIdentifier):
    method_key = "baseline.nonlinear-local"
    method_kind = LawMethodKind.NONLINEAR_LOCAL
    required_degree = 2


def baseline_law_identifiers() -> tuple[LawIdentifier, ...]:
    return (
        LocalLinearIdentifier(),
        LocalStateSpaceIdentifier(),
        NonlinearLocalIdentifier(),
    )
