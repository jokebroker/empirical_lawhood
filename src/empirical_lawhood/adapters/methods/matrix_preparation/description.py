"""Held development description checks and equally point-calibrated controls."""

from collections.abc import Mapping
from hashlib import sha256

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT
from .contracts import CANDIDATES
from .data import Array, BENCHMARK_SCHEMA, DESCRIPTION_SCHEMA, ProjectedContext, write_method_arrays
from .fitting import read_fit
from .models import adequacy_events, calibrate_gain, chart_components, privileged_chart, root_roles
from .records import PreparationAssessmentConfig, PreparationBenchmarkReport, PreparationCandidateScreen, PreparationDescriptionReport, PreparationFitReport, PreparationNumericalSemanticsReport
from .numerical_semantics import finite_metrics

npt_bool = np.ndarray[tuple[int], np.dtype[np.bool_]]


def prediction_errors(prediction: Array, response: Array) -> dict[str, Array]:
    if response.shape != (*prediction.shape[:-2], 4, 3, 5):
        raise ValueError("description error changes its whole root/chart/audit axes")
    errors = prediction[..., None, :, :] - response
    components = chart_components(prediction)[..., None, :, :] - chart_components(response)
    return {
        "absolute_errors": errors,
        "baseline_errors": components[..., 0, :],
        "odd_errors": components[..., 1, :],
        "even_errors": components[..., 2, :],
    }


def error_metrics(errors: Mapping[str, Array], screen: npt_bool) -> tuple[NamedDecimal, ...]:
    values: dict[str, tuple[float | int, str]] = {}
    for name, error in errors.items():
        if not name.endswith("_errors"):
            continue
        for view in range(2):
            values[f"{name.removesuffix('_errors')}-rmse-view{view + 1}"] = (
                float(np.sqrt(np.mean(error[screen, :, view] ** 2))),
                "hilbert-schmidt-native",
            )
        values[f"{name.removesuffix('_errors')}-rmse"] = (
            float(np.sqrt(np.mean(error[screen] ** 2))),
            "hilbert-schmidt-native",
        )
    return finite_metrics(values)


def observable_assessment_arrays(
    fitted: Mapping[str, Array], observable: ProjectedContext
) -> dict[str, Array]:
    arrays = observable.arrays
    prediction = fitted["predictions"]
    with np.errstate(invalid="ignore"):
        halfwidth = fitted["residual_scales"] * fitted["quantiles"][1]
    events = adequacy_events(
        prediction,
        arrays["response"],
        halfwidth,
        arrays["delivered"],
        arrays["contact"],
        arrays["preservation"],
    )
    result = prediction_errors(prediction, arrays["response"])
    result.update(
        {key: np.asarray(getattr(events, key), dtype=float) for key in events.__dataclass_fields__}
    )
    result["predictions"], result["halfwidths"] = prediction, halfwidth
    result["readout_path_rmse"] = np.sqrt(np.mean(result["absolute_errors"] ** 2, axis=(2, 3, 4)))
    return result


def assess_description(
    config: PreparationAssessmentConfig,
    observable: ProjectedContext,
    privileged: ProjectedContext,
    fit: PreparationFitReport,
    fit_payload: bytes,
    numerical_semantics: PreparationNumericalSemanticsReport,
) -> tuple[PreparationDescriptionReport, bytes, PreparationBenchmarkReport, bytes]:
    if (
        observable.role != "observable"
        or privileged.role != "privileged"
        or observable.context != privileged.context
        or fit.context != observable.context
        or fit.input_reports != observable.identities
        or numerical_semantics.input_reports != observable.identities
        or numerical_semantics.privileged_reports != privileged.identities
        or fit.config != ObjectIdentity.from_record(config.method.config_id, config.method)
        or fit.numerical_semantics != ObjectIdentity.from_record(numerical_semantics.report_id, numerical_semantics)
    ):
        raise ValueError("description assessment changes authenticated development operands")
    fitted = read_fit(fit, fit_payload)
    roles = root_roles(observable.roots)
    train, screen = roles == 0, roles == 2
    groups, screens = {}, []
    for candidate in CANDIDATES:
        reasons = []
        if candidate not in fitted:
            screens.append(
                PreparationCandidateScreen(candidate, (), False, ("POINT_MODEL_UNAVAILABLE",))
            )
            continue
        assessed = observable_assessment_arrays(fitted[candidate], observable)
        groups[candidate] = assessed
        abs_rmse = float(np.sqrt(np.mean(assessed["absolute_errors"][screen] ** 2)))
        odd_rmse = float(np.sqrt(np.mean(assessed["odd_errors"][screen] ** 2)))
        useful_roots = int(
            np.any(
                (assessed["point_adequacy"][screen].mean(axis=2) >= 0.5)
                & (assessed["width_pass"][screen] == 1),
                axis=1,
            ).sum()
        )
        simultaneous = float(
            np.mean(np.all(assessed["simultaneous_coverage"][:, :, 0] == 1, axis=1))
        )
        if not np.isfinite(abs_rmse) or abs_rmse > 0.125:
            reasons.append("ABSOLUTE_PREDICTION_TOLERANCE_FAILED_OR_UNRESOLVED")
        if not np.isfinite(odd_rmse) or odd_rmse > 1 / 64:
            reasons.append("ODD_RESPONSE_TOLERANCE_FAILED_OR_UNRESOLVED")
        if useful_roots < 8:
            reasons.append("FEWER_THAN_EIGHT_SAME_HANDOFF_ADEQUATE_SHARP_SCREEN_ROOTS")
        if simultaneous < 0.9:
            reasons.append("DEVELOPMENT_SIMULTANEOUS_COVERAGE_BELOW_FLOOR")
        if not numerical_semantics.numerical_semantics_qualified:
            reasons.append("NATIVE_NUMERICAL_SEMANTICS_UNQUALIFIED")
        metrics = list(error_metrics(assessed, screen))
        metrics.extend(
            finite_metrics(
                {
                    "same-handoff-adequate-sharp-screen-roots": (useful_roots, "1"),
                    "development-simultaneous-coverage": (simultaneous, "1"),
                    "screen-point-adequacy": (
                        float(assessed["point_adequacy"][screen].mean()),
                        "1",
                    ),
                    "screen-usable-window": (float(assessed["usable"][screen].mean()), "1"),
                    "screen-width-pass": (float(assessed["width_pass"][screen].mean()), "1"),
                    "screen-independent-roots": (16, "1"),
                }
            )
        )
        # Frozen wrong-action comparison is a falsifier, not a new fitting loss.
        y = observable.arrays["response"]
        wrong = prediction_errors(fitted[candidate]["predictions"][..., ::-1, :], y)
        groups[candidate]["wrong_sign_loss_increase"] = np.mean(
            (wrong["odd_errors"] / (1 / 64)) ** 2 - (assessed["odd_errors"] / (1 / 64)) ** 2,
            axis=(1, 2, 3, 4),
        )
        informative = np.any(
            np.abs(chart_components(y)[screen, ..., 1, -1]) >= 1 / 32, axis=(1, 2, 3)
        )
        metrics.extend(
            finite_metrics(
                {
                    "informative-action-roots": (int(informative.sum()), "1"),
                    "wrong-sign-loss-increase": (
                        float(groups[candidate]["wrong_sign_loss_increase"][screen].mean()),
                        "1",
                    ),
                }
            )
        )
        screens.append(
            PreparationCandidateScreen(
                candidate,
                tuple(sorted(metrics, key=lambda m: m.value_id)),
                not reasons,
                tuple(sorted(reasons)),
            )
        )

    benchmark_metrics, missing = [], []
    gain = privileged.arrays["privileged_gain"]
    truth = chart_components(observable.arrays["response"]).mean(axis=3)[..., 1, :]
    for index, model in enumerate(("scalar", "full_hessian", "microscopic")):
        raw = gain[..., index, :]
        for calibrated in (False, True):
            name = f"{model}_{'calibrated' if calibrated else 'raw'}"
            if "direct_linear" not in fitted:
                missing.append((name, "CAUSAL_OBSERVABLE_BASELINE_EVEN_COMPANION_UNAVAILABLE"))
                continue
            predicted_gain = raw.copy()
            parameters: dict[str, Array] = {"raw_gain": raw}
            if calibrated:
                if not np.isfinite(raw[train]).all() or not np.isfinite(truth[train]).all():
                    missing.append((name, "GAIN_CALIBRATION_TRAINING_OPERAND_UNRESOLVED"))
                    continue
                calibration = calibrate_gain(raw[train], truth[train])
                parameters.update(
                    {
                        "intercept": calibration.intercept,
                        "slope": calibration.slope,
                        "training_minimum": calibration.minimum,
                        "training_maximum": calibration.maximum,
                        "eligible": calibration.eligible.astype(float),
                    }
                )
                predicted_gain = calibration.predict(raw)
                parameters["calibrated_gain"] = predicted_gain
                if not calibration.eligible.all():
                    groups[name] = parameters
                    missing.append((name, "POSITIVE_SLOPE_OR_IDENTIFIABLE_GAIN_CALIBRATION_FAILED"))
                    continue
            prediction = privileged_chart(fitted["direct_linear"]["predictions"], predicted_gain)
            errors = prediction_errors(prediction, observable.arrays["response"])
            groups[name] = {**parameters, **errors, "predictions": prediction}
            if not all(np.isfinite(error[screen]).all() for error in errors.values()):
                missing.append((name, "BENCHMARK_SCREEN_OPERAND_UNRESOLVED"))
                continue
            benchmark_metrics.append((name, error_metrics(errors, screen)))
    identity = ObjectIdentity.from_record(config.config_id, config)
    payload = write_method_arrays(
        DESCRIPTION_SCHEMA,
        observable.context,
        identity,
        {k: v for k, v in groups.items() if k in CANDIDATES},
    )
    benchmark_data = write_method_arrays(
        BENCHMARK_SCHEMA,
        observable.context,
        identity,
        {k: v for k, v in groups.items() if k not in CANDIDATES},
    )
    report = PreparationDescriptionReport(
        f"{DEVELOPMENT}.description.{observable.context}",
        observable.context,
        identity,
        ObjectIdentity.from_record(fit.report_id, fit),
        ObjectIdentity.from_record(numerical_semantics.report_id, numerical_semantics),
        observable.identities,
        tuple(screens),
        sha256(payload).hexdigest(),
    )
    benchmark = PreparationBenchmarkReport(
        f"{DEVELOPMENT}.benchmarks.{observable.context}",
        observable.context,
        identity,
        ObjectIdentity.from_record(fit.report_id, fit),
        ObjectIdentity.from_record(report.report_id, report),
        privileged.identities,
        tuple(benchmark_metrics),
        tuple(missing),
        sha256(benchmark_data).hexdigest(),
    )
    return report, payload, benchmark, benchmark_data
