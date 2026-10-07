"""Safe physical observation operator for orthogonal FSM block experiments."""

from __future__ import annotations

import hashlib
from decimal import Decimal
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .contracts import FineSteeringMirrorBlockResponse, FineSteeringMirrorDevelopmentModel, FineSteeringMirrorObservationOperatorSpec

FloatArray = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]


class FineSteeringMirrorObservationError(ValueError):
    """Source bytes or derived observations violate the frozen operator."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_safe_array(
    path: Path,
    *,
    expected_realizations: int,
    operator: FineSteeringMirrorObservationOperatorSpec,
) -> FloatArray:
    """Load one exact numeric source array without executable deserialization."""

    loaded = np.load(path, allow_pickle=False)
    if loaded.dtype != np.dtype(np.float64):
        raise FineSteeringMirrorObservationError(f"{path.name} must have exact float64 dtype")
    expected_shape = (
        operator.sample_count,
        operator.input_channels,
        expected_realizations,
        operator.periods_per_realization,
    )
    if loaded.shape != expected_shape:
        raise FineSteeringMirrorObservationError(
            f"{path.name} shape {loaded.shape!r} differs from {expected_shape!r}"
        )
    if not np.isfinite(loaded).all():
        raise FineSteeringMirrorObservationError(f"{path.name} contains non-finite values")
    return np.asarray(loaded, dtype=np.float64)


def _decimal(value: object) -> Decimal:
    return Decimal(str(value))


def _triple(values: FloatArray) -> tuple[Decimal, Decimal, Decimal]:
    flattened = np.ravel(values)
    if flattened.size != 3:
        raise FineSteeringMirrorObservationError("receiver vector must have exactly three coordinates")
    return (_decimal(flattened[0]), _decimal(flattened[1]), _decimal(flattened[2]))


def reduce_block(
    *,
    block_id: str,
    action_level_volts: Decimal,
    actions: FloatArray,
    receivers: FloatArray,
    realization_start: int,
    operator: FineSteeringMirrorObservationOperatorSpec,
) -> FineSteeringMirrorBlockResponse:
    """Estimate the 3x3 FRM and reduce it to three band-local peak frequencies."""

    stop = realization_start + operator.realizations_per_block
    if stop > actions.shape[2] or actions.shape != receivers.shape:
        raise FineSteeringMirrorObservationError("block arrays do not contain the required matched realizations")
    period_frms: list[ComplexArray] = []
    maximum_condition = 0.0
    for period_index in range(operator.periods_per_realization):
        action_fft = np.asarray(
            np.fft.rfft(
                actions[:, :, realization_start:stop, period_index],
                axis=0,
            ),
            dtype=np.complex128,
        )
        receiver_fft = np.asarray(
            np.fft.rfft(
                receivers[:, :, realization_start:stop, period_index],
                axis=0,
            ),
            dtype=np.complex128,
        )
        inverse_action = np.linalg.inv(action_fft)
        period_frms.append(np.asarray(receiver_fft @ inverse_action, dtype=np.complex128))
        frequencies = np.fft.rfftfreq(
            operator.sample_count,
            d=1.0 / float(operator.sampling_frequency_hz),
        )
        lower, upper = (float(value) for value in operator.frequency_band_hz)
        band = (frequencies >= lower) & (frequencies <= upper)
        maximum_condition = max(
            maximum_condition,
            float(np.max(np.linalg.cond(action_fft[band]))),
        )
    mean_frm = np.mean(np.stack(period_frms, axis=0), axis=0)
    band_frequencies = frequencies[band]
    peak_hz = tuple(
        _decimal(
            band_frequencies[
                int(np.argmax(np.linalg.norm(mean_frm[band, output_index, :], axis=1)))
            ]
        )
        for output_index in range(operator.output_channels)
    )
    if len(period_frms) != 2:
        raise FineSteeringMirrorObservationError("fine-steering mirror period recurrence requires exactly two periods")
    period_difference = float(
        np.linalg.norm((period_frms[0] - period_frms[1])[band]) / np.linalg.norm(mean_frm[band])
    )
    return FineSteeringMirrorBlockResponse(
        block_id=block_id,
        action_level_volts=action_level_volts,
        receiver_peak_hz=(peak_hz[0], peak_hz[1], peak_hz[2]),
        maximum_input_matrix_condition=_decimal(maximum_condition),
        period_relative_difference=_decimal(period_difference),
    )


def fit_development_model(
    blocks: tuple[FineSteeringMirrorBlockResponse, ...],
    action_levels: tuple[Decimal, ...],
) -> FineSteeringMirrorDevelopmentModel:
    """Fit the frozen scalar-action linear susceptibility and its dev diagnostics."""

    if len(blocks) != 2 * len(action_levels):
        raise FineSteeringMirrorObservationError("development fit requires exactly two blocks per action level")
    action = np.asarray([float(block.action_level_volts) for block in blocks], dtype=np.float64)
    response = np.asarray(
        [[float(value) for value in block.receiver_peak_hz] for block in blocks],
        dtype=np.float64,
    )
    design = np.column_stack((np.ones(action.shape[0], dtype=np.float64), action))
    coefficients, _, _, _ = np.linalg.lstsq(design, response, rcond=None)
    fitted = design @ coefficients
    training_rmse = np.sqrt(np.mean(np.square(response - fitted), axis=0))
    leave_one_out_errors: list[FloatArray] = []
    for index in range(len(blocks)):
        keep = np.arange(len(blocks)) != index
        fold_coefficients, _, _, _ = np.linalg.lstsq(
            design[keep],
            response[keep],
            rcond=None,
        )
        predicted = design[index] @ fold_coefficients
        leave_one_out_errors.append(np.abs(response[index] - predicted))
    reverse = {level: action_levels[-1 - index] for index, level in enumerate(action_levels)}
    reversed_design = np.column_stack(
        (
            np.ones(action.shape[0], dtype=np.float64),
            np.asarray(
                [float(reverse[block.action_level_volts]) for block in blocks],
                dtype=np.float64,
            ),
        )
    )
    reversed_rmse = np.sqrt(np.mean(np.square(response - reversed_design @ coefficients), axis=0))
    prediction = tuple(
        _triple(np.asarray([1.0, float(level)]) @ coefficients) for level in action_levels
    )
    return FineSteeringMirrorDevelopmentModel(
        model_id="fsm-development-linear-susceptibility",
        intercept_hz=_triple(coefficients[0]),
        slope_hz_per_v=_triple(coefficients[1]),
        training_rmse_hz=_triple(training_rmse),
        leave_one_block_max_error_hz=_triple(np.max(np.stack(leave_one_out_errors), axis=0)),
        reversed_action_rmse_hz=_triple(reversed_rmse),
        prediction_hz_by_action=(prediction[0], prediction[1], prediction[2]),
        fit_block_ids=tuple(sorted(block.block_id for block in blocks)),
    )


def receiver_errors(
    model: FineSteeringMirrorDevelopmentModel,
    evaluation_blocks: tuple[FineSteeringMirrorBlockResponse, ...],
    action_levels: tuple[Decimal, ...],
) -> tuple[tuple[tuple[Decimal, Decimal, Decimal], ...], Decimal, Decimal]:
    """Return absolute prediction errors and correct/reversed primary RMSEs."""

    if tuple(block.action_level_volts for block in evaluation_blocks) != action_levels:
        raise FineSteeringMirrorObservationError("evaluation blocks must follow the frozen action chart")
    observed = np.asarray(
        [[float(value) for value in block.receiver_peak_hz] for block in evaluation_blocks],
        dtype=np.float64,
    )
    predicted = np.asarray(
        [[float(value) for value in row] for row in model.prediction_hz_by_action],
        dtype=np.float64,
    )
    errors = np.abs(observed - predicted)
    reversed_predictions = predicted[::-1]
    primary = 1
    correct_rmse = float(np.sqrt(np.mean(np.square(observed[:, primary] - predicted[:, primary]))))
    reversed_rmse = float(
        np.sqrt(np.mean(np.square(observed[:, primary] - reversed_predictions[:, primary])))
    )
    error_rows = tuple(_triple(row) for row in errors)
    return (
        (error_rows[0], error_rows[1], error_rows[2]),
        _decimal(correct_rmse),
        _decimal(reversed_rmse),
    )
