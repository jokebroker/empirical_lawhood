"""Whole-root exploratory transfer, uncertainty and signed error accounting.

The radial/all outer fits are authenticated saved I2R operands. They are
re-evaluated, never fitted again. The other fixed regimes and the cohort-balanced
inner folds use the same current kernel and ridge owners. Realized HOLD is used
only in the explicitly named evaluator diagnostic.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.adapters.methods.finite_response_law.fitting import DELTA, Normalizer
from empirical_lawhood.adapters.methods.finite_response_law.original_f import OriginalFiniteResponseLaw
from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge import (
    BridgeFit, RADIAL_STIFFNESS, fit_bridge, fit_hold, mechanical_kernel,
)
from empirical_lawhood.adapters.methods.response_formalization import affine_prediction

REGIMES = {
    "all": (tuple(range(1, 9)), tuple(range(2, 26))),
    "early_time": (tuple(range(1, 9)), tuple(range(2, 18))),
    "duration_transfer": ((1, 2, 5, 6), tuple(range(2, 26))),
    "recovery_transfer": ((1, 3, 5, 7), tuple(range(2, 26))),
}
TARGET_SCHEDULES = {
    "all": tuple(range(1, 9)), "early_time": tuple(range(1, 9)),
    "duration_transfer": (3, 4, 7, 8), "recovery_transfer": (2, 4, 6, 8),
}
Array = NDArray[np.float64]


def _close(actual, expected, message: str, *, atol=1e-12) -> None:
    if not np.allclose(actual, expected, rtol=0, atol=atol):
        raise ValueError(message)


def _required(arrays, name, shape, *, kind="float64"):
    value = arrays.get(name)
    if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != np.dtype(kind):
        raise ValueError(f"Preparation analysis changes required {name} axes/dtype")
    return value


def validate_operands(arrays: Mapping[str, NDArray], lower: OriginalFiniteResponseLaw) -> tuple[int, ...]:
    shapes = {"x": (24, 24, 2), "z": (24, 9, 24, 2), "y": (24, 9, 4, 8, 2, 2),
              "features": (24, 9, 2, 26, 24), "upper_z": (24, 9, 24),
              "upper_mean": (24, 9, 4, 8)}
    missing = set()
    for name, shape in shapes.items():
        value = _required(arrays, name, shape)
        missing.update(np.flatnonzero(~np.isfinite(value).reshape(24, -1).all(axis=1)).tolist())
    folds = _required(arrays, "folds", (24,), kind="int64")
    if not np.array_equal(folds, np.arange(24) % 4):
        raise ValueError("Transient analysis changes whole-root modulo-four outer folds")
    for name in ("center", "scale"):
        _close(_required(arrays, name, (24,)), np.asarray(getattr(lower, name), dtype=np.float64), "Original F numerical operand differs", atol=0)
    _close(_required(arrays, "operator", (25, 32)), np.asarray(lower.operator, dtype=np.float64).reshape(25, 32), "Original F operator differs", atol=0)
    if missing:
        return tuple(sorted(missing))
    _close(arrays["features"][:, :, :, -1].transpose(0, 1, 3, 2), arrays["z"], "Trajectory endpoint differs from current native handoff")
    _close(arrays["features"][:, :, 0, 0], arrays["x"][:, None, :, 0], "Trajectory changes same-root pre-parent input")
    _close(lower.predict(arrays["upper_z"].reshape(-1, 24)).mean.reshape(24, 9, 4, 8), arrays["upper_mean"], "Saved current U2 differs from Original F composition")
    for fold in range(4):
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        key = f"u2.fold.{fold}."
        if not np.array_equal(_required(arrays, key + "train", (18,), kind="int64"), train) or not np.array_equal(_required(arrays, key + "excluded", (6,), kind="int64"), test):
            raise ValueError("Saved current U2 changes the complete excluded-root census")
        center = _required(arrays, key + "center", (24,))
        scale = _required(arrays, key + "scale", (24,))
        operators = _required(arrays, key + "operator", (9, 25, 24))
        normalizer = Normalizer.fit(arrays["x"][train, :, 0])
        _close(center, normalizer.center, "Saved U2 normalizer changes training roots")
        _close(scale, normalizer.scale, "Saved U2 normalizer changes training roots")
        checked = np.stack([affine_prediction(operator, (arrays["x"][test, :, 0] - center) / scale) for operator in operators], axis=1)
        _close(checked, arrays["upper_z"][test], "Saved current U2 forecast disagrees with excluded-root operators")
    return ()


def _saved_radial_fit(arrays, fold: int, kernel: Array) -> BridgeFit:
    key = f"radial.all.fold.{fold}."
    train, test = np.flatnonzero(np.arange(24) % 4 != fold), np.flatnonzero(np.arange(24) % 4 == fold)
    for name, expected in (("training_roots", train), ("test_roots", test)):
        if not np.array_equal(_required(arrays, key + name, expected.shape, kind="int64"), expected):
            raise ValueError("Saved radial fit changes its exact excluded-root census")
    center = _required(arrays, key + "normalizer_center", (24,))
    scale = _required(arrays, key + "normalizer_scale", (24,))
    kernel_scale = _required(arrays, key + "kernel_scale", (4,))
    operator = _required(arrays, key + "operator", (25, 96))
    rank = _required(arrays, key + "kernel_rank", (), kind="int64")
    residual = _required(arrays, key + "training_modal_projection_mse", ())
    if not all(np.isfinite(v).all() for v in (center, scale, kernel_scale, operator, residual)) or np.any(scale <= 0) or np.any(kernel_scale <= 0) or rank != 4 or residual < 0:
        raise ValueError("Saved radial fit has invalid numerical coefficients")
    # These checks authenticate recipe coordinates, without another fit.
    expected = Normalizer.fit(arrays["x"][train, :, 0])
    _close(center, expected.center, "Saved radial normalizer changes training roots")
    _close(scale, expected.scale, "Saved radial normalizer changes training roots")
    _close(kernel_scale, np.maximum(np.sqrt(np.mean(kernel[1:, 2:] ** 2, axis=(0, 1))), 1e-12), "Saved radial modal scale changes training cells")
    fit = BridgeFit(Normalizer(center, scale), kernel_scale, operator, int(rank), float(residual))
    _close(fit.predict(arrays["x"][test, :, 0], kernel), arrays["radial.all.forecast_delta"][test], "Saved radial predictions disagree with exact coefficients")
    return fit


def _assessment(native: Array, arrays, lower, key: str, out: dict) -> None:
    predicted = lower.predict(native.reshape(-1, 24))
    mean = predicted.mean.reshape(24, 9, 4, 8)
    sigma = predicted.sigma.reshape(24, 9, 4, 8)
    normalized = lower.normalize(native.reshape(-1, 24)).reshape(24, 9, 24)
    error = mean[..., None, None] - arrays["y"]
    width = float(lower.q) * sigma + DELTA / 8
    out[key + ".native"] = native
    out[key + ".mean"] = mean
    out[key + ".faces"] = 6 - np.abs(normalized)
    out[key + ".point_support"] = predicted.supported.reshape(24, 9)
    out[key + ".response_mse_by_root"] = np.mean(error[:, 1:, :, :2] ** 2, axis=(1, 2, 3, 4, 5))
    out[key + ".feature_mse_by_root"] = np.mean(((native - arrays["z"][..., 0]) / arrays["scale"]) ** 2, axis=(1, 2))
    out[key + ".frozen_lower_width"] = width
    out[key + ".frozen_lower_joint_coverage"] = np.all(np.abs(error) <= width[..., None, None], axis=(2, 3, 4, 5))


def transient_analysis(arrays: Mapping[str, NDArray], lower: OriginalFiniteResponseLaw) -> dict[str, NDArray]:
    """Complete fixed 32 outer / 24 inner fit census; no native acquisition."""
    incomplete = validate_operands(arrays, lower)
    if incomplete:
        raise ValueError("UNEVALUABLE: complete current roots are required before fitting")
    prefix, scale = arrays["x"][..., 0], arrays["scale"]
    delta = (arrays["features"][:, :, 0] - arrays["features"][:, :1, 0]) / scale
    truth_delta = (arrays["z"][..., 0] - arrays["z"][:, :1, :, 0]) / scale
    folds = np.arange(24, dtype=np.int64) % 4
    out = {name: np.array(arrays[name], copy=True) for name in ("x", "z", "y", "upper_z", "upper_mean", "center", "scale", "operator")}
    out.update(folds=folds, truth_delta=truth_delta)
    hold = np.repeat(arrays["upper_z"][:, :1], 9, axis=1)
    _assessment(arrays["z"][..., 0], arrays, lower, "actual_handoff_f", out)
    _assessment(arrays["upper_z"], arrays, lower, "current_u2", out)
    _assessment(hold, arrays, lower, "predicted_hold_only", out)
    for model, stiffness in (("radial", RADIAL_STIFFNESS), ("nonrestoring", 0.0)):
        kernel, waveform = mechanical_kernel(stiffness)
        if model == "radial":
            _close(kernel, arrays["radial.kernel"], "Saved radial kernel differs from fixed recipe", atol=0)
        out[model + ".kernel"], out[model + ".waveform"] = kernel, waveform
        for regime, (schedules, times) in REGIMES.items():
            key = model + "." + regime
            forecast = np.empty((24, 9, 26, 24), dtype=np.float64)
            fits = []
            response_q = np.empty(24, dtype=np.float64)
            feature_radius = np.empty((24, 24), dtype=np.float64)
            inner_masks = np.zeros((4, 3, 3, 24), dtype=np.bool_)
            inner_scores = np.empty((4, 18), dtype=np.float64)
            inner_fits = []
            inner_native_all = np.empty((4, 18, 9, 24), dtype=np.float64)
            for fold in range(4):
                train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
                fit = _saved_radial_fit(arrays, fold, kernel) if key == "radial.all" else fit_bridge(prefix[train], delta[train], kernel, schedules=schedules, times=times)
                fits.append(fit)
                forecast[test] = fit.predict(prefix[test], kernel)
                if regime != "all":
                    continue
                _close(fit_hold(prefix[train], arrays["z"][train, 0, :, 0], prefix[test]), arrays["upper_z"][test, 0], "Saved HOLD anchor changes its excluded-root recipe", atol=1e-10)
                labels = np.empty(18, dtype=np.int64)
                for cohort in (train < 8, train >= 8):
                    positions = np.flatnonzero(cohort)
                    labels[positions] = np.arange(len(positions)) % 3
                inner_native = np.empty((18, 9, 24), dtype=np.float64)
                for inner in range(3):
                    itraining = train[labels != inner]
                    positions = np.flatnonzero(labels == inner)
                    scored = train[positions]
                    inner_masks[fold, inner, 0, itraining] = True
                    inner_masks[fold, inner, 1, scored] = True
                    inner_masks[fold, inner, 2, test] = True
                    if set(itraining) & (set(scored) | set(test)) or set(scored) & set(test):
                        raise ValueError("Whole-root inner/outer leakage")
                    inner_fit = fit_bridge(prefix[itraining], delta[itraining], kernel)
                    inner_fits.append(inner_fit)
                    shift = inner_fit.predict(prefix[scored], kernel)[:, :, -1] * scale
                    inner_hold = fit_hold(prefix[itraining], arrays["z"][itraining, 0, :, 0], prefix[scored])
                    inner_native[positions] = inner_hold[:, None] + shift
                inner_native_all[fold] = inner_native
                mean = lower.predict(inner_native.reshape(-1, 24)).mean.reshape(18, 9, 4, 8)
                scores = np.max(np.abs(mean[..., None, None] - arrays["y"][train]) / DELTA[None, None, None, :, None, None], axis=(1, 2, 3, 4, 5))
                inner_scores[fold] = scores
                response_q[test] = scores.max()
                feature_radius[test] = np.max(np.abs(inner_native - arrays["z"][train, :, :, 0]) / scale, axis=(0, 1))
            for name in ("operator", "kernel_scale"):
                out[key + ".fits." + name] = np.stack([getattr(fit, name) for fit in fits])
            for name in ("center", "scale"):
                out[key + ".fits." + name] = np.stack([getattr(fit.normalizer, name) for fit in fits])
            out[key + ".fits.rank"] = np.asarray([fit.rank for fit in fits], dtype=np.int64)
            out[key + ".fits.residual_mse"] = np.asarray([fit.residual_mse for fit in fits], dtype=np.float64)
            out[key + ".forecast_delta"] = forecast
            shift = forecast[:, :, -1]
            native = hold + shift * scale
            _assessment(native, arrays, lower, key, out)
            # This diagnostic never supplies a predictor input or a training target.
            _assessment(arrays["z"][:, :1, :, 0] + shift * scale, arrays, lower, key + ".measured_hold_diagnostic", out)
            for suffix in ("native", "feature_mse_by_root", "frozen_lower_width", "frozen_lower_joint_coverage"):
                out.pop(key + ".measured_hold_diagnostic." + suffix)
            out[key + ".increment_mse_by_root"] = np.mean((shift[:, 1:] - truth_delta[:, 1:]) ** 2, axis=(1, 2))
            selected = np.asarray(TARGET_SCHEDULES[regime])
            out[key + ".transfer_mse_by_root"] = np.mean((shift[:, selected] - truth_delta[:, selected]) ** 2, axis=(1, 2))
            out[key + ".trajectory_mse_by_root"] = np.mean((forecast[:, 1:, 2:] - delta[:, 1:, 2:]) ** 2, axis=(1, 2, 3))
            out[key + ".late_mse_by_root"] = np.mean((forecast[:, 1:, 18:] - delta[:, 1:, 18:]) ** 2, axis=(1, 2, 3))
            if regime == "all":
                out[key + ".inner_masks"] = inner_masks
                out[key + ".inner_scores"] = inner_scores
                out[key + ".inner_native"] = inner_native_all
                out[key + ".inner_operator"] = np.stack([fit.operator for fit in inner_fits]).reshape(4, 3, 25, 96)
                out[key + ".inner_center"] = np.stack([fit.normalizer.center for fit in inner_fits]).reshape(4, 3, 24)
                out[key + ".inner_scale"] = np.stack([fit.normalizer.scale for fit in inner_fits]).reshape(4, 3, 24)
                out[key + ".inner_kernel_scale"] = np.stack([fit.kernel_scale for fit in inner_fits]).reshape(4, 3, 4)
                out[key + ".response_q"] = response_q
                out[key + ".feature_radius"] = feature_radius
                error = np.abs(out[key + ".mean"][..., None, None] - arrays["y"])
                box = response_q[:, None, None, None] * DELTA
                out[key + ".exploratory_response_covered"] = np.all(error <= box[..., None, None], axis=(1, 2, 3, 4, 5))
                out[key + ".enclosed_support"] = np.all(out[key + ".faces"] >= feature_radius[:, None], axis=-1)
                out[key + ".feature_enclosed"] = np.all(np.abs(native - arrays["z"][..., 0]) / scale <= feature_radius[:, None], axis=(1, 2))
    return out


def verify_saved_forecasts(arrays) -> None:
    """Re-evaluate every saved coefficient and disjoint mask without fitting."""
    folds = _required(arrays, "folds", (24,), kind="int64")
    if not np.array_equal(folds, np.arange(24) % 4):
        raise ValueError("Saved report changes its outer-root census")
    prefix = _required(arrays, "x", (24, 24, 2))[..., 0]
    for model in ("radial", "nonrestoring"):
        kernel = _required(arrays, model + ".kernel", (9, 26, 4))
        _close(kernel, mechanical_kernel(RADIAL_STIFFNESS if model == "radial" else 0.0)[0], "Saved report changes its mechanical kernel", atol=0)
        for regime in REGIMES:
            key = model + "." + regime
            forecast = _required(arrays, key + ".forecast_delta", (24, 9, 26, 24))
            operators = _required(arrays, key + ".fits.operator", (4, 25, 96))
            centers = _required(arrays, key + ".fits.center", (4, 24))
            scales = _required(arrays, key + ".fits.scale", (4, 24))
            modal = _required(arrays, key + ".fits.kernel_scale", (4, 4))
            for fold in range(4):
                test = np.flatnonzero(folds == fold)
                coefficients = affine_prediction(operators[fold], (prefix[test] - centers[fold]) / scales[fold]).reshape(6, 4, 24)
                checked = np.einsum("stk,rkj->rstj", kernel / modal[fold], coefficients)
                _close(checked, forecast[test], "Saved coefficient/forecast disagreement")
            if regime == "all":
                masks = _required(arrays, key + ".inner_masks", (4, 3, 3, 24), kind="bool")
                for fold in range(4):
                    train, test = folds != fold, folds == fold
                    score_census = np.zeros(24, dtype=np.int64)
                    for inner in range(3):
                        a, b, c = masks[fold, inner]
                        if np.any(a & (b | c)) or np.any(b & c) or not np.array_equal(c, test) or not np.array_equal(a | b, train):
                            raise ValueError("Saved inner mask changes complete disjoint root census")
                        positions = np.flatnonzero(train)
                        expected = np.empty(18, dtype=np.int64)
                        for cohort in (positions < 8, positions >= 8):
                            p = np.flatnonzero(cohort)
                            expected[p] = np.arange(len(p)) % 3
                        if not np.array_equal(np.flatnonzero(b), positions[expected == inner]):
                            raise ValueError("Saved inner split changes cohort-balanced modulo-three recipe")
                        score_census += b
                    if not np.array_equal(score_census, train.astype(np.int64)):
                        raise ValueError("Saved inner scoring drops or repeats an independent root")


def baseline_support_analysis(arrays, lower: OriginalFiniteResponseLaw, *, model="radial.all") -> tuple[dict[str, NDArray], dict[str, float]]:
    """Outcome-visible decomposition, inclusive support and complete failures."""
    verify_saved_forecasts(arrays)
    scale = np.asarray(lower.scale, dtype=np.float64)
    center = np.asarray(lower.center, dtype=np.float64)
    truth = arrays["z"][..., 0]
    shift = arrays[model + ".forecast_delta"][:, :, -1]
    native = arrays[model + ".native"]
    baseline = (arrays["upper_z"][:, :1] - truth[:, :1]) / scale
    transient = shift - arrays["truth_delta"]
    full = (native - truth) / scale
    _close(full, baseline + transient, "Signed feature error components do not reconstruct forecast")
    operator = np.asarray(lower.operator, dtype=np.float64).reshape(25, 4, 8)[:-1, :, :2]
    baseline_response = np.einsum("rsj,jko->rsko", baseline, operator)[..., None, None]
    transient_response = np.einsum("rsj,jko->rsko", transient, operator)[..., None, None]
    lower_error = arrays["actual_handoff_f.mean"][:, :, :, :2, None, None] - arrays["y"][:, :, :, :2]
    full_error = arrays[model + ".mean"][:, :, :, :2, None, None] - arrays["y"][:, :, :, :2]
    _close(full_error, lower_error + baseline_response + transient_response, "Signed response components do not reconstruct forecast")
    square = lambda value: float(np.mean(value ** 2))
    metrics = {"feature.baseline_mse": square(baseline), "feature.transient_mse": square(transient[:, 1:]),
               "feature.cross_term": 2 * float(np.mean(baseline * transient[:, 1:])), "feature.total_mse": square(full[:, 1:])}
    components = (lower_error[:, 1:], np.broadcast_to(baseline_response, full_error.shape)[:, 1:], transient_response[:, 1:])
    labels = ("lower", "baseline", "transient")
    for label, value in zip(labels, components, strict=True):
        metrics["response." + label + "_mse"] = square(value)
    for i in range(3):
        for j in range(i + 1, 3):
            metrics[f"response.cross_{labels[i]}_{labels[j]}"] = 2 * float(np.mean(components[i] * components[j]))
    metrics["response.total_mse"] = square(full_error[:, 1:])
    _close(metrics["feature.total_mse"], sum(value for key, value in metrics.items() if key.startswith("feature.") and not key.endswith("total_mse")), "Feature signed cross terms fail reconstruction")
    _close(metrics["response.total_mse"], sum(value for key, value in metrics.items() if key.startswith("response.") and not key.endswith("total_mse")), "Response signed cross terms fail reconstruction")
    normalized = (truth - center) / scale
    actual = np.isfinite(normalized).all(axis=-1) & (np.max(np.abs(normalized), axis=-1) <= 6)
    predicted = lower.predict(native.reshape(-1, 24)).supported.reshape(24, 9)
    false_supported, false_unsupported = predicted & ~actual, ~predicted & actual
    known_failure = np.argwhere(~actual).astype(np.int64)
    metrics.update({"support.true_supported": float(np.sum(predicted & actual)),
                    "support.true_unsupported": float(np.sum(~predicted & ~actual)),
                    "support.false_supported": float(false_supported.sum()),
                    "support.false_unsupported": float(false_unsupported.sum()),
                    "support.total_cells": 216.0, "support.failure_roots": float(np.any(~actual, axis=1).sum())})
    out = {"feature_baseline": baseline, "feature_transient": transient, "feature_total": full,
           "response_lower": components[0], "response_baseline": components[1], "response_transient": components[2],
           "actual_support": actual, "predicted_support": predicted, "false_supported": false_supported,
           "false_unsupported": false_unsupported, "known_failure_indices": known_failure,
           "actual_faces": 6 - np.abs(normalized), "predicted_faces": arrays[model + ".faces"],
           "missing_roots": np.zeros(24, dtype=np.bool_)}
    return out, metrics
