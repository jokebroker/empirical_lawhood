"""Observable-only fit producer and authenticated frozen-model decoding."""

from hashlib import sha256
from collections.abc import Mapping

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from .contracts import CANDIDATES, FIT_SCHEMA, PreparationMethodConfig
from .data import Array, ProjectedContext, read_method_arrays, write_method_arrays
from .models import ObservableFit, ObservablePredictor, ResidualScaleModel, fit_candidate, root_roles
from .records import PreparationFitReport, PreparationNumericalSemanticsReport
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT


def fit_arrays(fit: ObservableFit) -> dict[str, Array]:
    model, scales = fit.model, fit.scale_model
    return {
        "mean": model.mean,
        "scale": model.scale,
        "operator": model.operator,
        "constant_odd_even": np.empty((0, 5))
        if model.constant_odd_even is None
        else model.constant_odd_even,
        "penalty": np.array([model.penalty]),
        "training_roots": np.array([model.training_roots]),
        "penalty_losses": fit.penalty_losses,
        "fit_fold_predictions": fit.fit_fold_predictions,
        "all_root_fold_predictions": fit.all_root_fold_predictions,
        "predictions": fit.predictions,
        "scale_means": scales.means,
        "scale_scales": scales.scales,
        "scale_operators": scales.operators,
        "reference_rms_scales": scales.reference_rms,
        "residual_scales": fit.residual_scales,
        "quantiles": np.array([fit.provisional_quantile, fit.development_quantile]),
        "provisional_scores": fit.provisional_scores,
        "development_scores": fit.development_scores,
    }


def model_from_fit_arrays(
    candidate: str, arrays: Mapping[str, Array]
) -> tuple[ObservablePredictor, ResidualScaleModel]:
    shapes = {
        "mean": (9,),
        "scale": (9,),
        "operator": (10, 5)
        if candidate == "constant_gain"
        else (55 if candidate == "direct_quadratic" else 10, 15),
        "constant_odd_even": (2, 5) if candidate == "constant_gain" else (0, 5),
        "penalty": (1,),
        "training_roots": (1,),
        "penalty_losses": (5,),
        "fit_fold_predictions": (32, 5, 2, 3, 5),
        "all_root_fold_predictions": (64, 5, 2, 3, 5),
        "predictions": (64, 5, 2, 3, 5),
        "scale_means": (2, 9),
        "scale_scales": (2, 9),
        "scale_operators": (2, 10, 15),
        "reference_rms_scales": (2, 3, 5),
        "residual_scales": (64, 5, 2, 3, 5),
        "quantiles": (2,),
        "provisional_scores": (16,),
        "development_scores": (64,),
    }
    if (
        candidate not in CANDIDATES
        or set(arrays) != set(shapes)
        or any(arrays[k].shape != shape for k, shape in shapes.items())
    ):
        raise ValueError("frozen observable model changes its exact coefficient/array chart")
    required = (
        "mean",
        "scale",
        "operator",
        "constant_odd_even",
        "penalty",
        "training_roots",
        "scale_means",
        "scale_scales",
        "scale_operators",
        "reference_rms_scales",
    )
    if (
        any(not np.isfinite(arrays[k]).all() for k in required)
        or np.any(arrays["scale"] <= 0)
        or np.any(arrays["scale_scales"] <= 0)
        or arrays["training_roots"][0] != 32
        or arrays["quantiles"][0] != np.inf
        or np.any(arrays["reference_rms_scales"] < 1 / 128)
    ):
        raise ValueError("frozen observable model has unresolved or changed fitted operands")
    predictor = ObservablePredictor(
        candidate,
        float(arrays["penalty"][0]),
        arrays["mean"],
        arrays["scale"],
        arrays["operator"],
        arrays["constant_odd_even"] if candidate == "constant_gain" else None,
        32,
    )
    scale_model = ResidualScaleModel(
        arrays["scale_means"],
        arrays["scale_scales"],
        arrays["scale_operators"],
        arrays["reference_rms_scales"],
        32,
    )
    return predictor, scale_model


def fit_preparation_context(
    config: PreparationMethodConfig,
    observed: ProjectedContext,
    numerical_semantics: PreparationNumericalSemanticsReport,
) -> tuple[PreparationFitReport, bytes]:
    if (
        observed.role != "observable"
        or numerical_semantics.context != observed.context
        or numerical_semantics.input_reports != observed.identities
        or any(r.projection_config != config.projection_config for r in observed.reports)
    ):
        raise ValueError(
            "observable fitting received privileged/task data or changed measurement lineage"
        )
    x, y = observed.arrays["handoff_features"], observed.arrays["response"]
    train = root_roles(observed.roots) == 0
    available, missing, groups = [], [], {}
    for candidate in CANDIDATES:
        if not np.isfinite(x[train]).all() or not np.isfinite(y[train]).all():
            missing.append((candidate, "FIT_INPUT_UNRESOLVED"))
            continue
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                fit = fit_candidate(observed.roots, x, y, candidate)
        except (FloatingPointError, np.linalg.LinAlgError):
            missing.append((candidate, "NUMERICAL_FIT_UNRESOLVED"))
            continue
        groups[candidate] = fit_arrays(fit)
        available.append(candidate)
    identity = ObjectIdentity.from_record(config.config_id, config)
    payload = write_method_arrays(FIT_SCHEMA, observed.context, identity, groups)
    report = PreparationFitReport(
        f"{DEVELOPMENT}.fit.{observed.context}",
        observed.context,
        identity,
        observed.identities,
        ObjectIdentity.from_record(numerical_semantics.report_id, numerical_semantics),
        tuple(available),
        tuple(missing),
        sha256(payload).hexdigest(),
    )
    return report, payload


def read_fit(report: PreparationFitReport, payload: bytes) -> dict[str, dict[str, Array]]:
    groups = read_method_arrays(
        payload,
        schema=FIT_SCHEMA,
        context=report.context,
        config=report.config,
        expected_sha256=report.data_sha256,
    )
    if set(groups) != set(report.available_candidates):
        raise ValueError("fit artifact changes its attempted candidate roster")
    for candidate, arrays in groups.items():
        model_from_fit_arrays(candidate, arrays)
    return groups
