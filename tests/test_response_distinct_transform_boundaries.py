"""Independent arithmetic and leakage checks for retained prepared response consumers.

These are bounded development checks. They do not create qualified parent data.
"""

from math import ceil

from tests.prepared_seed_fixtures import fixture_root_randomness

import numpy as np
import pytest

from empirical_lawhood.adapters.methods.causal_response.models import central_gain, primary_contrast
from empirical_lawhood.adapters.methods.finite_response_law.intervals import oriented_interval, quantile
from empirical_lawhood.adapters.methods.information_response.models import features
from empirical_lawhood.adapters.methods.prepared_response.calibration import prepared_joint_root_losses, prepared_root_folds
from empirical_lawhood.adapters.methods.response_composition.development import folds, paired_response, parent_contrast, root_error, split_training
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import REFERENCE_INSTRUMENT, reference_sketch
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedRoot
from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame
from empirical_lawhood.adapters.simulators.six_matrix_response.gradients import analytic_gradient_terms
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ideal_state


def test_information_masks_keep_only_causal_snapshot_columns() -> None:
    history = np.arange(16 * 12, dtype=np.float64).reshape(1, 16, 12)
    sketch = np.arange(8, dtype=np.float64).reshape(1, 8)
    snapshot = (0, 1, 2, 3, 4, 5, 9, 10)
    np.testing.assert_array_equal(
        features(history, sketch, "snapshot")[0], history[0, -1, snapshot]
    )
    np.testing.assert_array_equal(
        features(history, sketch, "history")[0], history[0][:, list(snapshot)].reshape(-1)
    )
    assert features(history, sketch, "snapshot-with-mechanism").shape == (1, 16)
    assert features(history, sketch, "history-with-mechanism").shape == (1, 136)
    hidden = history.copy()
    hidden[:, :, (6, 7, 8, 11)] = 1e9
    np.testing.assert_array_equal(
        features(hidden, sketch, "history"), features(history, sketch, "history")
    )
    earlier = history.copy()
    earlier[:, 0, 0] = -123
    np.testing.assert_array_equal(
        features(earlier, sketch, "snapshot"), features(history, sketch, "snapshot")
    )
    assert not np.array_equal(
        features(earlier, sketch, "history"), features(history, sketch, "history")
    )
    with pytest.raises(ValueError, match="scalar chart"):
        features(history[:, :-1], sketch, "history")


def test_causal_finite_difference_and_parent_contrast() -> None:
    observed = np.zeros((5, 9, 5, 2), dtype=np.float64)
    for parent in range(5):
        observed[parent, 1, :, 0] = -4 * parent
        observed[parent, 2, :, 0] = 4 * parent
        observed[parent, 3, :, 1] = -2 * parent
        observed[parent, 4, :, 1] = 2 * parent
    gain = central_gain(observed)
    np.testing.assert_array_equal(gain[:, 0, 0, 0], np.arange(5) * 4)
    np.testing.assert_array_equal(gain[:, 0, 1, 1], np.arange(5) * 2)
    contrast = primary_contrast(gain)
    assert contrast.shape == (4, 2, 2, 2)
    assert contrast[1, 0, 0, 0] == 8  # parent 2 minus the parent-0 hold gain
    assert contrast[3, 1, 1, 1] == 8  # parent 4 minus the parent-0 hold gain
    with pytest.raises(ValueError, match="nine-word"):
        central_gain(observed[:, :-1])


def test_finite_lawhood_root_rank_and_signed_word_orientation() -> None:
    scores = np.arange(16, dtype=np.float64) / 10
    rank, threshold = quantile(scores)
    assert rank == ceil(17 * 0.9) == 16
    assert threshold == 1.5
    assert quantile(np.arange(8, dtype=np.float64))[1] == float("inf")
    lower = np.arange(8, dtype=np.float64).reshape(1, 1, 8).repeat(4, axis=1)
    upper = lower + 0.25
    negative_low, negative_high = oriented_interval(lower, upper, 0)
    assert negative_low[0, 0] == -upper[0, 0, 0]
    assert negative_high[0, 0] == -lower[0, 0, 0]
    with pytest.raises(ValueError, match="Unknown native word"):
        oriented_interval(lower, upper, 8)


def test_response_composition_posthoc_pairing_whole_root_folds_and_two_view_score() -> None:
    values = np.zeros((2, 5, 9, 5, 2), dtype=np.float64)
    values[..., 0, :, :] = 7
    values[:, 2, 1, 2, 0] = 10
    values[:, 2, 2, 2, 0] = 14
    response = paired_response(values)
    assert response[:, 2, 0, 2, 0].tolist() == [0, 0]
    assert response[:, 2, 2, 2, 0].tolist() == [7, 7]
    contrast = parent_contrast(response)
    assert contrast[:, 2, 2, 2, 0].tolist() == [14, 14]
    root_folds = folds(tuple(range(16)), 4)
    assert root_folds == tuple(tuple(range(k, 16, 4)) for k in range(4))
    assert split_training(tuple(range(16)), root_folds[0]) == tuple(
        i for i in range(16) if i % 4 != 0
    )
    prediction = np.zeros((1, 2, 5, 9, 5, 2), dtype=np.float64)
    truth = prediction.copy()
    truth[0, 1, 4, 2, 2, 0] = 0.03
    assert root_error(prediction, truth).tolist() == [0.03]
    with pytest.raises(ValueError, match="whole-root"):
        folds((0, 0, 1), 2)
    with pytest.raises(ValueError, match="complete nine-word"):
        paired_response(values[..., :8, :, :])


def test_prepared_development_uses_whole_root_folds_and_joint_maximum_loss() -> None:
    roots = tuple(
        PreparedRoot('development', "assembling", index, "a" * 64, fixture_root_randomness('development', 'a' * 64, 'assembling', index)) for index in (3, 0, 2, 1)
    )
    assert prepared_root_folds(roots) == (3, 0, 2, 1)
    observed = np.zeros((4, 5, 2, 9, 5, 7))
    predicted = observed.copy()
    predicted[2, 4, 1, 8, 4, 6] = 0.5
    predicted[2, 0, 0, 0, 0, 0] = 0.25
    scale = np.array((0.25, 1, 1, 1, 1, 1, 0.5))
    np.testing.assert_array_equal(
        prepared_joint_root_losses(observed, predicted, scale),
        np.array((0, 0, 1, 0)),
    )
    predicted[0, 0, 0, 0, 0, 0] = np.nan
    assert np.isinf(prepared_joint_root_losses(observed, predicted, scale)[0])
    with pytest.raises(ValueError, match="full output roster"):
        prepared_joint_root_losses(observed[:, :, :1], predicted[:, :, :1], scale)


def test_finite_reference_hessian_matches_independent_central_difference() -> None:
    modes = np.zeros((2, 3, 4, 4), dtype=np.complex128)
    traceless = np.diag([1, -1, 0, 0]) / np.sqrt(2)
    modes[0, 0] = traceless
    modes[1, 1] = traceless
    frame = PreparedPortFrame(4096, modes)
    positions = ideal_state(
        q=2, alpha_tilde_x=2 / 3, alpha_tilde_y=22 / 3, constitution="11"
    ).positions
    sketch = reference_sketch(positions, frame)
    observed_hessian = sketch[2:6].reshape(2, 2)
    expected = np.empty((2, 2))
    step = 1e-5
    for column in range(2):
        direction = np.zeros_like(positions)
        direction[0] = modes[column]
        plus = analytic_gradient_terms(
            positions + step * direction, REFERENCE_INSTRUMENT.parameters
        ).total[0]
        minus = analytic_gradient_terms(
            positions - step * direction, REFERENCE_INSTRUMENT.parameters
        ).total[0]
        derivative = (plus - minus) / (2 * step)
        for row in range(2):
            expected[row, column] = np.vdot(modes[row], derivative).real
    np.testing.assert_allclose(observed_hessian, expected, rtol=1e-8, atol=1e-8)
    assert np.max(np.abs(observed_hessian - observed_hessian.T)) < 1e-12
    invalid = positions.copy()
    invalid[0, 0, 0, 1] = 1j
    with pytest.raises(ValueError, match="finite Hermitian"):
        reference_sketch(invalid, frame)
