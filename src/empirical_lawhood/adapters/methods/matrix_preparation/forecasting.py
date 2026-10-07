"""Observable joint-event fitting with held-root forecast diagnostics."""

from hashlib import sha256

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT
from .contracts import CANDIDATES, PreparationMethodConfig
from .data import Array, FORECAST_SCHEMA, ProjectedContext, read_method_arrays, write_method_arrays
from .description import observable_assessment_arrays
from .fitting import read_fit
from .forecasts import EVENTS, EventPredictor, brier_roots, fit_forecast, forecast_contact
from .models import root_roles
from .records import PreparationDescriptionReport, PreparationFitReport, PreparationForecastReport, PreparationGate
from .numerical_semantics import finite_metrics
from .statistics import coarse_reliability


def event_model_arrays(model: EventPredictor) -> dict[str, Array]:
    return {
        "mean": model.mean,
        "scale": model.scale,
        "coefficients": model.coefficients,
        "constant": np.array([np.nan if model.constant is None else model.constant]),
        "lower": model.lower,
        "upper": model.upper,
        "penalty": np.array([model.penalty]),
        "training_roots": np.array([model.training_roots], dtype=float),
    }


def forecast_preparation_context(
    config: PreparationMethodConfig,
    observed: ProjectedContext,
    fit: PreparationFitReport,
    fit_payload: bytes,
    description: PreparationDescriptionReport,
) -> tuple[PreparationForecastReport, bytes]:
    fit_ref = ObjectIdentity.from_record(fit.report_id, fit)
    if (
        observed.role != "observable"
        or description.context != observed.context
        or fit.context != observed.context
        or description.fit != fit_ref
        or fit.input_reports != observed.identities
        or description.input_reports != observed.identities
        or fit.config != ObjectIdentity.from_record(config.config_id, config)
    ):
        raise ValueError("event forecasts received another model or forbidden information role")
    fitted = read_fit(fit, fit_payload)
    roles = root_roles(observed.roots)
    train, screen = roles == 0, roles == 2
    pre, post = observed.arrays["preparent_features"][:, 0], observed.arrays["handoff_features"]
    groups, available, missing, candidate_metrics, passes = {}, [], [], [], {}
    for candidate in CANDIDATES:
        if candidate not in fitted:
            missing.append((candidate, "POINT_MODEL_UNAVAILABLE"))
            continue
        if not np.isfinite(pre[train]).all() or not np.isfinite(post[train]).all():
            missing.append((candidate, "TRAINING_OBSERVER_UNRESOLVED"))
            continue
        assessment = observable_assessment_arrays(fitted[candidate], observed)
        # These labels describe this exact frozen predictor and interval recipe.
        # Held screen roots fitted neither the point model nor the forecasters.
        arrays = {f"label_{event}": assessment[event] for event in EVENTS}
        metrics: dict[str, tuple[float | int, str]] = {}
        candidate_passes = True
        try:
            for event in EVENTS:
                labels = assessment[event]
                forecast = fit_forecast(observed.roots, pre, post, labels)
                arrays[f"{event}_context_intercept"] = np.array([forecast.context_intercept])
                arrays[f"{event}_parent_intercepts"] = forecast.parent_intercepts
                arrays[f"{event}_penalty_losses"] = forecast.penalty_losses
                for stage, model, probabilities, support in (
                    (
                        "preparent",
                        forecast.preparent_model,
                        forecast.preparent_predictions,
                        forecast.preparent_support,
                    ),
                    (
                        "handoff",
                        forecast.handoff_model,
                        forecast.handoff_predictions,
                        forecast.handoff_support,
                    ),
                ):
                    prefix = f"{event}_{stage}"
                    arrays[prefix + "_probability"] = probabilities
                    arrays[prefix + "_support"] = support.astype(float)
                    arrays.update(
                        {prefix + "_" + k: v for k, v in event_model_arrays(model).items()}
                    )
                    contact = forecast_contact(
                        probabilities[screen],
                        labels[screen],
                        forecast.context_intercept,
                        forecast.parent_intercepts,
                    )
                    if event != "preservation":
                        candidate_passes &= bool(contact["passes"])
                    metrics.update(
                        {f"{prefix}-{key}": (float(value), "1") for key, value in contact.items()}
                    )
                    metrics[prefix + "-brier"] = (
                        float(brier_roots(probabilities[screen], labels[screen]).mean()),
                        "1",
                    )
                    metrics[prefix + "-calibration-in-large"] = (
                        float(np.mean(probabilities[screen][..., None] - labels[screen])),
                        "1",
                    )
                    reliability = coarse_reliability(probabilities[screen], labels[screen])
                    columns = (
                        "lower_edge",
                        "upper_edge",
                        "root_count",
                        "parent_cells",
                        "mean_probability",
                        "event_fraction",
                        "calibration_error_lower",
                        "calibration_error_upper",
                    )
                    arrays[prefix + "_coarse_bins"] = np.array(
                        [
                            [np.nan if row[key] is None else row[key] for key in columns]
                            for row in reliability
                        ],
                        dtype=float,
                    )
        except (FloatingPointError, np.linalg.LinAlgError):
            missing.append((candidate, "NUMERICAL_FORECAST_FIT_UNRESOLVED"))
            continue
        groups[candidate], passes[candidate] = arrays, candidate_passes
        available.append(candidate)
        candidate_metrics.append((candidate, finite_metrics(metrics)))
    chosen = description.selected_candidate
    reasons = []
    if chosen is None:
        reasons.append("NO_QUALIFIED_OBSERVABLE_DESCRIPTION")
    elif chosen not in passes:
        reasons.append("SELECTED_DESCRIPTION_FORECAST_UNAVAILABLE")
    elif not passes[chosen]:
        reasons.append("FORECAST_UNCONTACTED_OR_NO_HELD_BRIER_IMPROVEMENT")
        if np.all(groups[chosen]["label_point_adequacy"] == 1):
            reasons.append("UNIVERSAL_DEVELOPMENT_POINT_ADEQUACY")
        if not np.any(groups[chosen]["label_usable"]):
            reasons.append("NO_DEVELOPMENT_USABLE_EVENTS")
            if not np.any(observable_assessment_arrays(fitted[chosen], observed)["width_pass"]):
                reasons.append("WIDTH_FORCES_UNUSABILITY")
    identity = ObjectIdentity.from_record(config.config_id, config)
    payload = write_method_arrays(FORECAST_SCHEMA, observed.context, identity, groups)
    return PreparationForecastReport(
        f"{DEVELOPMENT}.forecast.{observed.context}",
        observed.context,
        identity,
        fit_ref,
        ObjectIdentity.from_record(description.report_id, description),
        observed.identities,
        tuple(available),
        tuple(missing),
        tuple(candidate_metrics),
        PreparationGate(
            f"{DEVELOPMENT}.gate.preparation-window-forecast.{observed.context}",
            observed.context,
            "preparation-window-forecast",
            not reasons,
            tuple(sorted(reasons)),
        ),
        sha256(payload).hexdigest(),
    ), payload


def read_forecasts(
    report: PreparationForecastReport, payload: bytes
) -> dict[str, dict[str, Array]]:
    groups = read_method_arrays(
        payload,
        schema=FORECAST_SCHEMA,
        context=report.context,
        config=report.config,
        expected_sha256=report.data_sha256,
    )
    expected: dict[str, tuple[int, ...]] = {}
    for event in EVENTS:
        expected[f"label_{event}"] = (64, 5, 4)
        expected[f"{event}_context_intercept"] = (1,)
        expected[f"{event}_parent_intercepts"] = (5,)
        expected[f"{event}_penalty_losses"] = (2, 5)
        for stage, features in (("preparent", 14), ("handoff", 9)):
            prefix = f"{event}_{stage}_"
            shapes = {
                "probability": (64, 5),
                "support": (64, 5),
                "mean": (features,),
                "scale": (features,),
                "coefficients": (features + 1,),
                "constant": (1,),
                "lower": (5, features),
                "upper": (5, features),
                "penalty": (1,),
                "training_roots": (1,),
                "coarse_bins": (4, 8),
            }
            expected.update({prefix + k: v for k, v in shapes.items()})
    if set(groups) != set(report.available_candidates) or any(
        set(a) != set(expected) or any(a[k].shape != s for k, s in expected.items())
        for a in groups.values()
    ):
        raise ValueError("frozen event models changed their exact coefficient/observation roster")
    return groups
