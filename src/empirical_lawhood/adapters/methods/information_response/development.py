"Whole-root ridge selection on the exposed source qualification panel; no native acquisition."

from typing import Any
from decimal import Decimal

import numpy as np

from empirical_lawhood.adapters.methods.causal_response.models import CausalResponsePanel
from empirical_lawhood.adapters.methods.prepared_response.models import PreparedBilinearModelSpec
from .models import Array, InformationResponseContextPredictor, InformationResponsePredictor, FITTED_IDS, MODEL_IDS, PROGRAMME, RIDGES, central_gain, fit, primary_contrast


def predict_panel(model: InformationResponsePredictor, panel: CausalResponsePanel, view: int = 0) -> Array:
    return model.predict(
        panel.history[:, :, view].reshape(-1, 16, 12),
        panel.sketch[:, :, view].reshape(-1, 8),
        tuple(range(5)) * len(panel.roots),
    ).reshape(len(panel.roots), 5, 5, 2, 2)


def select(panel: CausalResponsePanel, arm: str, folds: int) -> tuple[Decimal | None, list[float], Array]:
    candidates = RIDGES if arm in FITTED_IDS else (None,)
    predictions = np.empty((len(candidates), len(panel.roots), 5, 5, 2, 2))
    for ci, ridge in enumerate(candidates):
        for fold in range(folds):
            train = tuple(i for i in range(len(panel.roots)) if i % folds != fold)
            test = tuple(i for i in range(len(panel.roots)) if i % folds == fold)
            model = fit(panel.subset(train), arm, ridge)
            predictions[ci, list(test)] = predict_panel(model, panel.subset(test))
    observed = central_gain(panel.observed[:, :, 0])
    errors = predictions - observed
    losses = np.mean(errors[..., (2, 4), :, :] ** 2, axis=(1, 2, 3, 4, 5))
    if not np.isfinite(predictions).all() or not np.isfinite(losses).all():
        raise ValueError("information response prediction required development fit is nonfinite; no arm substitution")
    return candidates[int(np.argmin(losses))], losses.tolist(), predictions


def develop(panel: CausalResponsePanel) -> tuple[InformationResponseContextPredictor, dict[str, Any], dict[str, Array]]:
    if tuple(r.index for r in panel.roots) != tuple(range(16)):
        raise ValueError("information response prediction development requires all original 16 roots in order")
    models, selections, arrays = [], {}, {}
    outer = np.empty((16, 2, 7, 5, 5, 2, 2))
    for mi, arm in enumerate(MODEL_IDS):
        ridge, losses, cv = select(panel, arm, 4)
        models.append(fit(panel, arm, ridge))
        selections[arm] = {"ridge": None if ridge is None else str(ridge), "cv_mse": losses}
        arrays[f"{arm}_candidate_cv_gain"] = cv
        for fold in range(4):
            train = tuple(i for i in range(16) if i % 4 != fold)
            test = tuple(i for i in range(16) if i % 4 == fold)
            training, testing = panel.subset(train), panel.subset(test)
            inner, _, _ = select(training, arm, 3)
            model = fit(training, arm, inner)
            for view in range(2):
                outer[list(test), view, mi] = predict_panel(model, testing, view)
    observed = central_gain(np.swapaxes(panel.observed, 1, 2))
    full = np.mean((outer - observed[:, :, None])[..., (2, 4), :, :] ** 2, axis=(3, 4, 5, 6))
    contrast = np.mean(
        (primary_contrast(outer) - primary_contrast(observed)[:, :, None]) ** 2,
        axis=(3, 4, 5, 6),
    )
    arrays.update(outer_gain=outer, outer_full_mse=full, outer_contrast_mse=contrast)
    context = InformationResponseContextPredictor(
        PreparedBilinearModelSpec(
            f"{PROGRAMME}.{panel.roots[0].context}.recipe", panel.roots[0].context, Decimal(16)
        ),
        panel.roots,
        tuple(models),
    )
    report = {
        "context": panel.roots[0].context,
        "training_roots": [r.root_id for r in panel.roots],
        "selections": selections,
        "nested_outer_full_mse_by_view_model": full.mean(axis=0).tolist(),
        "nested_outer_contrast_mse_by_view_model": contrast.mean(axis=0).tolist(),
        "scientific_ceiling": "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
        "fresh_entry": "REQUIRED_IF_OPERATIONALLY_ELIGIBLE_IRRESPECTIVE_OF_ACCURACY",
    }
    return context, report, arrays
