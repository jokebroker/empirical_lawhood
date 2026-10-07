"""dependent refinement constant-gain control with the existing learned absolute baseline.

Reuse the native nine-word chart, whole-root folds and fitted lower models.
Only the two-port gain is replaced. These are point-prediction diagnostics;
the original model's calibration cannot be attached to changed predictions.
"""

from decimal import Decimal

import numpy as np

from empirical_lawhood.adapters.methods.response_formalization import fit_affine_operator

from .calibration import PreparedCrossfit
from .models import Array, PreparedTrainingPanel, PreparedFinitePredictor, _freeze, _word_controls


def fit_prepared_constant_gain(panel: PreparedTrainingPanel, amplitude: Decimal) -> Array:
    """Unpenalized, zero-baseline port gain from training-only paired odd responses.

    All four directions, parents and numerical views of the supplied training
    roots enter the same two-port least-squares fit. There is no hyperparameter
    search. The symmetric signed design fixes the intercept to zero; baseline
    motion remains the existing model's HOLD prediction.
    """
    if amplitude not in (Decimal(8), Decimal(16)):
        raise ValueError("constant-gain benchmark requires the frozen source qualification amplitude")
    if not panel.valid.any():
        raise ValueError("constant-gain benchmark has no complete training rows")
    controls = _word_controls(amplitude)[1:]
    observed = panel.observed[panel.valid, ..., :2]
    odd = (observed[:, 2::2] - observed[:, 1::2]) / 2
    signed = np.stack((-odd, odd), axis=2).reshape(-1, 8, 5, 2)
    design = np.broadcast_to(controls, (len(signed), 8, 2)).reshape(-1, 2)
    # The common solver adds an unpenalized intercept. Signed pairs cancel it;
    # only its two slope rows describe the intervention contrast.
    operator = fit_affine_operator(design, signed.reshape(-1, 10), ridge=0.0)
    gain = operator[:2].reshape(2, 5, 2)
    if not np.isfinite(gain).all():
        raise ValueError("constant-gain benchmark is numerically unresolved")
    return _freeze(gain)


def predict_prepared_constant_gain(
    baseline: PreparedFinitePredictor,
    gain: Array,
    history: Array,
    sketch: Array | None = None,
) -> Array:
    """Both absolute receivers at all nine words and five readouts, without refit."""
    if gain.shape != (2, 5, 2) or not np.isfinite(gain).all():
        raise ValueError("constant-gain benchmark requires the complete two-port gain")
    hold = baseline.predict(history, sketch)[:, :1, :, :2]
    contrast = np.einsum("wp,ptr->wtr", _word_controls(baseline.spec.amplitude), gain)
    return _freeze(hold + contrast[None])


def crossfit_prepared_constant_gain(
    panel: PreparedTrainingPanel, baseline: PreparedCrossfit,
) -> tuple[Array, tuple[Array, ...]]:
    """Reuse four existing outer models; retain every assigned root and missing row.

    Returns held-root receiver predictions and the four training-only gains.
    The same gains can serve every baseline structure on this exact fold roster;
    no baseline or residual-scale fit is repeated by this calculation.
    """
    if panel.roots != baseline.roots:
        raise ValueError("constant-gain benchmark changes the baseline's whole-root census")
    predictions = np.full((*panel.observed.shape[:-1], 2), np.nan)
    gains = []
    for fold, model in enumerate(baseline.models):
        training = panel.subset(tuple(i for i, f in enumerate(baseline.folds) if f != fold))
        gain = fit_prepared_constant_gain(training, model.spec.amplitude)
        gains.append(gain)
        for index, assignment in enumerate(baseline.folds):
            if assignment != fold:
                continue
            valid = panel.valid[index].copy()
            if model.structure == "mechanism-i1":
                valid &= np.isfinite(panel.sketch[index]).all(axis=-1)
            if valid.any():
                hold = baseline.predictions[index, valid, :1, :, :2]
                contrast = np.einsum("wp,ptr->wtr", _word_controls(model.spec.amplitude), gain)
                predictions[index, valid] = hold + contrast[None]
    return _freeze(predictions), tuple(gains)
