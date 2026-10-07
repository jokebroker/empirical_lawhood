"Frozen whole-root comparison and selection on exposed source qualification observations."

from typing import Any

import numpy as np

from empirical_lawhood.adapters.methods.prepared_response.models import PreparedBilinearModelSpec, prepared_bilinear_model_shapes

from .models import Array, CANDIDATES, CausalResponseCoefficientBlock, CausalResponseContextPredictor, CausalResponsePanel, central_gain, fit, predict_gain, primary_contrast


def cross_validate(panel: CausalResponsePanel, folds: int) -> tuple[int, Array, list[dict[str, Any]]]:
    n = len(panel.roots)
    predictions = np.full((len(CANDIDATES), n, 5, 5, 2, 2), np.nan)
    fits: list[dict[str, Any]] = []
    for ci, candidate in enumerate(CANDIDATES):
        for fold in range(folds):
            train = tuple(i for i in range(n) if i % folds != fold)
            test = tuple(i for i in range(n) if i % folds == fold)
            entry: dict[str, Any] = {
                "candidate": [candidate[0], str(candidate[1])],
                "fold": fold,
                "training_roots": [panel.roots[i].root_id for i in train],
                "validation_roots": [panel.roots[i].root_id for i in test],
            }
            try:
                model = fit(panel.subset(train), candidate)
                predictions[ci, list(test)] = predict_gain(model, panel.subset(test))
                entry["status"] = "FINITE"
            except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
                entry.update(status="FAILED", reason=str(error))
            fits.append(entry)
    observed = primary_contrast(central_gain(panel.observed[:, :, 0]))
    losses = np.mean((primary_contrast(predictions) - observed) ** 2, axis=(1, 2, 3, 4, 5))
    losses[~np.isfinite(losses)] = np.inf
    if not np.isfinite(losses).any():
        raise ValueError(
            "causal response prediction no finite candidate; retain all failed fits before a prerequisite stop"
        )
    selected = int(np.argmin(losses))
    return selected, predictions, fits


def develop(panel: CausalResponsePanel) -> tuple[CausalResponseContextPredictor, dict[str, Any], dict[str, Array]]:
    if len(panel.roots) != 16 or tuple(r.index for r in panel.roots) != tuple(range(16)):
        raise ValueError("causal response prediction context development requires its full original 16-root assignment")
    selected, cv_predictions, cv_fits = cross_validate(panel, 4)
    observed = primary_contrast(central_gain(panel.observed[:, :, 0]))
    cv_loss = np.mean((primary_contrast(cv_predictions) - observed) ** 2, axis=(1, 2, 3, 4, 5))
    outer = np.full((16, 2, 5, 5, 2, 2), np.nan)
    baseline = np.empty_like(outer)
    outer_fits: list[dict[str, Any]] = []
    arrays = {"candidate_cv_gain": cv_predictions}
    for fold in range(4):
        train = tuple(i for i in range(16) if i % 4 != fold)
        test = tuple(i for i in range(16) if i % 4 == fold)
        training, testing = panel.subset(train), panel.subset(test)
        inner_selected, inner_predictions, inner_fits = cross_validate(training, 3)
        arrays[f"outer_{fold}_inner_gain"] = inner_predictions
        model = fit(training, CANDIDATES[inner_selected])
        for view in range(2):
            outer[list(test), view] = predict_gain(model, testing, view)
            baseline[list(test), view] = central_gain(training.observed[:, :, 0]).mean(axis=0)
        outer_fits.append(
            {
                "fold": fold,
                "selected_index": inner_selected,
                "inner_fits": inner_fits,
                "training_roots": [r.root_id for r in training.roots],
                "validation_roots": [r.root_id for r in testing.roots],
            }
        )
    fitted = []
    for ci, candidate in enumerate(CANDIDATES):
        entry = {"candidate_index": ci, "status": "FINITE"}
        try:
            model = fit(panel, candidate)
            for name in prepared_bilinear_model_shapes(model.structure):
                arrays[f"refit_{ci}_{name}"] = getattr(model, name)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as error:
            entry.update(status="FAILED", reason=str(error))
        fitted.append(entry)
    model = fit(panel, CANDIDATES[selected])
    blocks = {name: getattr(model, name) for name in prepared_bilinear_model_shapes(model.structure)}
    blocks["baseline_gain"] = central_gain(panel.observed[:, :, 0]).mean(axis=0)
    assert isinstance(model.spec, PreparedBilinearModelSpec)
    context = CausalResponseContextPredictor(
        model.spec,
        model.structure,
        model.ridge,
        panel.roots,
        tuple(
            CausalResponseCoefficientBlock.from_array(name, value) for name, value in sorted(blocks.items())
        ),
    )
    measured = primary_contrast(central_gain(np.swapaxes(panel.observed, 1, 2)))
    outer_loss = np.mean((primary_contrast(outer) - measured) ** 2, axis=(2, 3, 4, 5))
    baseline_loss = np.mean((primary_contrast(baseline) - measured) ** 2, axis=(2, 3, 4, 5))
    arrays.update(
        outer_gain=outer,
        outer_baseline_gain=baseline,
        outer_loss=outer_loss,
        outer_baseline_loss=baseline_loss,
        candidate_cv_loss=cv_loss,
    )
    report = {
        "context": panel.roots[0].context,
        "assigned_source_qualification_roots": [r.root_id for r in panel.roots],
        "selected_index": selected,
        "selected_structure": model.structure,
        "selected_ridge": str(model.ridge),
        "candidates": [[s, str(r)] for s, r in CANDIDATES],
        "candidate_cv_losses": [float(x) if np.isfinite(x) else None for x in cv_loss],
        "cv_fits": cv_fits,
        "outer_fits": outer_fits,
        "full_refits": fitted,
        "nested_outer_mse": outer_loss.mean(axis=0).tolist(),
        "nested_outer_baseline_mse": baseline_loss.mean(axis=0).tolist(),
        "fresh_entry": "REQUIRED_IF_PRODUCTION_ELIGIBLE_IRRESPECTIVE_OF_DEVELOPMENT_ACCURACY",
        "scientific_ceiling": "EXPOSED_DEVELOPMENT_ONLY_NONPROMOTABLE",
    }
    return context, report, arrays
