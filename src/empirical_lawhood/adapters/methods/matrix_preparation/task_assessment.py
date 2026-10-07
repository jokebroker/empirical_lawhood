"""Prediction-locked decisions and independently revealed task scoring."""

from hashlib import sha256

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT, PARENTS, preparation_roots
from empirical_lawhood.adapters.simulators.six_matrix_response.response_assay import parent_schedule
from .contracts import CANDIDATES, PreparationMethodConfig
from .data import Array, DECISION_SCHEMA, TASK_ASSESSMENT_SCHEMA, ProjectedContext, read_method_arrays, write_method_arrays
from .decisions import TARGET_SEED_SHA256, development_targets, select_commands, select_parents
from .description import observable_assessment_arrays
from .fitting import read_fit
from .forecasting import read_forecasts
from .forecasts import brier_roots
from .models import root_roles
from .records import PreparationDecisionReport, PreparationDescriptionReport, PreparationFitReport, PreparationForecastReport, PreparationGate, PreparationTaskAssessmentReport
from .numerical_semantics import finite_metrics, native_view_errors
from .statistics import coarse_reliability, paired_adequacy_bounds, preservation_noninferiority


def declared_parent_costs() -> Array:
    return np.asarray(
        [
            float(
                np.abs(
                    np.diff(
                        np.vstack(
                            ([2 / 3, 22 / 3], parent_schedule(parent=p, refinement=1)[:144])
                        ),
                        axis=0,
                    )
                ).sum()
            )
            for p in PARENTS
        ]
    )


def lock_development_decisions(
    config: PreparationMethodConfig,
    fit: PreparationFitReport,
    fit_payload: bytes,
    description: PreparationDescriptionReport,
    forecast: PreparationForecastReport,
    forecast_payload: bytes,
) -> tuple[PreparationDecisionReport, bytes]:
    fit_ref, description_ref = (
        ObjectIdentity.from_record(fit.report_id, fit),
        ObjectIdentity.from_record(description.report_id, description),
    )
    if (
        description.fit != fit_ref
        or forecast.fit != fit_ref
        or forecast.description != description_ref
        or fit.config != ObjectIdentity.from_record(config.config_id, config)
        or forecast.config != fit.config
        or not fit.context == description.context == forecast.context
    ):
        raise ValueError("decisions changed the previously persisted prediction/event locks")
    fitted, forecasts = read_fit(fit, fit_payload), read_forecasts(forecast, forecast_payload)
    roots = tuple(r for r in preparation_roots() if r.context == fit.context)
    groups, costs = {}, declared_parent_costs()
    for candidate in CANDIDATES:
        group = {
            "parent_indices": np.zeros(64),
            "parent_refused": np.ones(64),
            "parent_eligible": np.zeros((64, 5)),
            "command_indices": np.full((64, 5), -1.0),
            "command_reasons": np.ones((64, 5)),
        }
        if candidate in fitted:
            group["predictions"] = fitted[candidate]["predictions"]
            with np.errstate(invalid="ignore"):
                group["halfwidths"] = (
                    fitted[candidate]["residual_scales"] * fitted[candidate]["quantiles"][1]
                )
        if candidate in forecasts:
            f = forecasts[candidate]
            supported = (f["usable_preparent_support"] == 1) & (
                f["preservation_preparent_support"] == 1
            )
            selected = select_parents(
                f["usable_preparent_probability"],
                f["preservation_preparent_probability"],
                supported,
                costs,
            )
            group["parent_indices"] = selected.parent_indices.astype(float)
            group["parent_refused"] = selected.refused.astype(float)
            group["parent_eligible"] = selected.eligible.astype(float)
        groups[candidate] = group
    # This function is a downstream task: both input artifacts already have
    # immutable runtime receipts. Targets become visible only after those locks.
    targets = development_targets(roots)
    groups["assignment"] = {"targets": targets, "parent_costs": costs}
    for candidate in CANDIDATES:
        if candidate not in forecasts or candidate not in fitted:
            continue
        f, group = forecasts[candidate], groups[candidate]
        commands = select_commands(
            group["predictions"],
            group["halfwidths"],
            targets,
            f["preservation_handoff_probability"],
            f["preservation_handoff_support"] == 1,
            description_qualified=candidate in description.qualified_candidates,
        )
        group["command_indices"], group["command_reasons"] = (
            commands.action_indices.astype(float),
            commands.reasons.astype(float),
        )
    identity = ObjectIdentity.from_record(config.config_id, config)
    payload = write_method_arrays(DECISION_SCHEMA, fit.context, identity, groups)
    return PreparationDecisionReport(
        f"{DEVELOPMENT}.decision.{fit.context}",
        fit.context,
        identity,
        fit_ref,
        description_ref,
        ObjectIdentity.from_record(forecast.report_id, forecast),
        TARGET_SEED_SHA256,
        sha256(payload).hexdigest(),
    ), payload


def read_decisions(
    report: PreparationDecisionReport, payload: bytes
) -> dict[str, dict[str, Array]]:
    groups = read_method_arrays(
        payload,
        schema=DECISION_SCHEMA,
        context=report.context,
        config=report.config,
        expected_sha256=report.data_sha256,
    )
    if set(groups) != {*CANDIDATES, "assignment"} or set(groups["assignment"]) != {
        "targets",
        "parent_costs",
    }:
        raise ValueError("decision artifact changes its locked candidate/assignment roster")
    if groups["assignment"]["targets"].shape != (64,) or groups["assignment"][
        "parent_costs"
    ].shape != (5,):
        raise ValueError("decision assignment changes its full root/parent census")
    for candidate in CANDIDATES:
        g = groups[candidate]
        expected = {
            "parent_indices": (64,),
            "parent_refused": (64,),
            "parent_eligible": (64, 5),
            "command_indices": (64, 5),
            "command_reasons": (64, 5),
        }
        if "predictions" in g:
            expected.update({"predictions": (64, 5, 2, 3, 5), "halfwidths": (64, 5, 2, 3, 5)})
        if set(g) != set(expected) or any(g[k].shape != s for k, s in expected.items()):
            raise ValueError("decision artifact changes its coefficient or finite command chart")
        if (
            not np.isin(g["parent_indices"], range(5)).all()
            or not np.isin(g["command_indices"], (-1, 0, 1, 2)).all()
        ):
            raise ValueError("decision artifact contains an unentered parent/action")
    return groups


def native_task_chart(task: ProjectedContext, targets: Array) -> dict[str, Array]:
    if task.role != "prospective-task" or targets.shape != (64,):
        raise ValueError(
            "native task scoring requires independent task outcomes and assigned targets"
        )
    a = task.arrays
    response = a["response"]
    if response.shape != (64, 5, 2, 3, 5):
        raise ValueError("native task outcomes dropped a root/parent/view/action/readout")
    absolute, odd = native_view_errors(response[..., None, :, :])
    numerical = (absolute[..., 0] <= 1 / 128) & (odd[..., 0] <= 1 / 256)
    delivered = np.all(a["delivered"] == 1, axis=2)
    preserved = np.all(a["preservation"] == 1, axis=2)
    contact = np.all(a["contact"] == 1, axis=2)
    attained = np.all(
        np.isfinite(response[..., -1])
        & (np.abs(response[..., -1] - targets[:, None, None, None]) <= 0.125),
        axis=2,
    )
    success = attained & delivered & preserved & contact[..., None] & numerical[..., None]
    return {
        "endpoint": response[..., -1],
        "raw_attainment": attained.astype(float),
        "delivered": delivered.astype(float),
        "preserved": preserved.astype(float),
        "contact": contact.astype(float),
        "numerical": numerical.astype(float),
        "physical_task_success": success.astype(float),
        "oracle_task_success": np.any(success, axis=2).astype(float),
        "unique_successful_command": np.where(
            success.sum(axis=2) == 1, np.argmax(success, axis=2), -1
        ).astype(float),
    }


def assess_development_tasks(
    config: PreparationMethodConfig,
    observed: ProjectedContext,
    task: ProjectedContext,
    fit: PreparationFitReport,
    fit_payload: bytes,
    description: PreparationDescriptionReport,
    forecast: PreparationForecastReport,
    forecast_payload: bytes,
    decision: PreparationDecisionReport,
    decision_payload: bytes,
) -> tuple[PreparationTaskAssessmentReport, bytes]:
    if (
        observed.role != "observable"
        or task.role != "prospective-task"
        or observed.roots != task.roots
        or not observed.context
        == fit.context
        == description.context
        == forecast.context
        == decision.context
        or fit.input_reports != observed.identities
        or forecast.input_reports != observed.identities
        or decision.prediction_lock != ObjectIdentity.from_record(fit.report_id, fit)
        or decision.description != ObjectIdentity.from_record(description.report_id, description)
        or decision.forecast != ObjectIdentity.from_record(forecast.report_id, forecast)
        or decision.config != ObjectIdentity.from_record(config.config_id, config)
        or any(
            a.native_result != b.native_result
            for a, b in zip(observed.reports, task.reports, strict=True)
        )
    ):
        raise ValueError("task assessment changes its independent futures or prior decision locks")
    fitted = read_fit(fit, fit_payload)
    forecasts = read_forecasts(forecast, forecast_payload)
    decisions = read_decisions(decision, decision_payload)
    native = native_task_chart(task, decisions["assignment"]["targets"])
    groups: dict[str, dict[str, Array]] = {"native_task": native}
    roles = root_roles(observed.roots)
    train, screen = roles == 0, roles == 2
    row = np.arange(64)
    metrics: dict[str, tuple[float | int, str]] = {}
    for candidate in CANDIDATES:
        d = decisions[candidate]
        indices = d["command_indices"].astype(int)
        commanded = indices >= 0
        taken = np.take_along_axis(
            native["physical_task_success"], np.maximum(indices, 0)[..., None], axis=2
        )[..., 0]
        success = taken * commanded
        parents = d["parent_indices"].astype(int)
        groups[candidate] = {
            "task_success": success,
            "refused": (~commanded).astype(float),
            "selected_task_success": success[row, parents],
            "selected_task_refused": (~commanded[row, parents]).astype(float),
        }
        if candidate in fitted:
            assessed = observable_assessment_arrays(fitted[candidate], observed)
            for key in (
                "point_adequacy",
                "usable",
                "width_pass",
                "simultaneous_coverage",
                "numerical",
                "contact",
                "preservation",
                "delivery",
            ):
                groups[candidate][key] = assessed[key]
                groups[candidate]["selected_" + key] = assessed[key][row, parents]
            known = np.isfinite(assessed["absolute_maximum"]) & np.isfinite(assessed["odd_maximum"])
            groups[candidate]["adequacy_known"] = known.astype(float)
            preservation_known = np.all(
                np.isin(observed.arrays["preservation"], (0, 1)), axis=(2, 4)
            )
            groups[candidate]["preservation_known"] = preservation_known.astype(float)
            groups[candidate]["selected_preservation_known"] = preservation_known[
                row, parents
            ].astype(float)
        if candidate in forecasts:
            f = forecasts[candidate]
            for event in ("point_adequacy", "usable"):
                labels = f["label_" + event][row, parents]
                for stage in ("preparent", "handoff"):
                    p = f[f"{event}_{stage}_probability"][row, parents]
                    groups[candidate][f"selected_{event}_{stage}_brier_roots"] = brier_roots(
                        p, labels
                    )
                    metrics[f"{candidate}-{event}-{stage}-selected-screen-brier"] = (
                        float(brier_roots(p[screen], labels[screen]).mean()),
                        "1",
                    )
                    metrics[f"{candidate}-{event}-{stage}-selected-calibration-in-large"] = (
                        float(np.mean(p[screen, None] - labels[screen])),
                        "1",
                    )
                    for kind, baseline in (
                        ("context", np.full(64, f[event + "_context_intercept"][0])),
                        ("parent", f[event + "_parent_intercepts"][parents]),
                    ):
                        groups[candidate][
                            f"selected_{event}_{stage}_{kind}_brier_improvement_roots"
                        ] = brier_roots(baseline, labels) - brier_roots(p, labels)
                    bins = coarse_reliability(p[screen], labels[screen])
                    groups[candidate][f"selected_{event}_{stage}_reliability"] = np.asarray(
                        [
                            [
                                np.nan if b[k] is None else b[k]
                                for k in (
                                    "root_count",
                                    "parent_cells",
                                    "mean_probability",
                                    "event_fraction",
                                    "calibration_error_lower",
                                    "calibration_error_upper",
                                )
                            ]
                            for b in bins
                        ],
                        dtype=float,
                    )
            certified = d["parent_refused"] == 0
            metrics[f"{candidate}-certified-screen-roots"] = (int(certified[screen].sum()), "1")
            if certified[screen].any():
                metrics[f"{candidate}-false-certification-screen"] = (
                    float(
                        (1 - groups[candidate]["selected_usable"][screen][certified[screen]]).mean()
                    ),
                    "1",
                )
            metrics[f"{candidate}-selected-useful-coverage"] = (
                float(groups[candidate]["selected_usable"][screen].mean()),
                "1",
            )

    selected = description.selected_candidate
    fixed_a = fixed_s = reference = None
    reasons = []
    primary_parents = (
        np.zeros(64, dtype=int)
        if selected is None
        else decisions[selected]["parent_indices"].astype(int)
    )
    if selected is None:
        reasons.append("NO_QUALIFIED_OBSERVABLE_DESCRIPTION")
    else:
        primary = groups[selected]
        costs = decisions["assignment"]["parent_costs"]
        fixed_a = min(
            range(5),
            key=lambda p: (-primary["point_adequacy"][train, p].mean(), costs[p], PARENTS[p]),
        )
        fixed_s = min(
            range(5),
            key=lambda p: (-primary["task_success"][train, p].mean(), costs[p], PARENTS[p]),
        )
        reference = min(
            description.qualified_candidates,
            key=lambda c: (
                -groups[c]["task_success"][row, primary_parents][train].mean(),
                CANDIDATES.index(c),
            ),
        )
        for label, comparator in (
            ("hold", 0),
            ("adequacy_best_fixed", fixed_a),
            ("task_best_fixed", fixed_s),
        ):
            contrast = primary["selected_point_adequacy"] - primary["point_adequacy"][:, comparator]
            success_difference = (
                primary["selected_task_success"] - primary["task_success"][:, comparator]
            )
            primary[label + "_adequacy_difference_roots"] = contrast.mean(axis=1)
            primary[label + "_task_difference_roots"] = success_difference
            known = primary["adequacy_known"] == 1
            bounds = paired_adequacy_bounds(
                primary["selected_point_adequacy"][screen],
                primary["point_adequacy"][screen, comparator],
                known[row, primary_parents][screen],
                known[screen, comparator],
            )
            metrics[f"{label}-adequacy-difference"] = (bounds[0], "1")
            metrics[f"{label}-adequacy-conservative-lower"] = (bounds[1], "1")
            metrics[f"{label}-task-difference"] = (float(success_difference[screen].mean()), "1")
            harms = preservation_noninferiority(
                primary["selected_preservation"][screen] == 1,
                primary["preservation"][screen, comparator] == 1,
                (primary["selected_delivery"][screen] == 1)
                & (primary["delivery"][screen, comparator] == 1)
                & (primary["selected_preservation_known"][screen] == 1)
                & (primary["preservation_known"][screen, comparator] == 1),
            )
            metrics[f"{label}-preservation-harm-upper"] = (harms[2], "1")
        increment = (
            primary["selected_task_success"]
            - groups[reference]["task_success"][row, primary_parents]
        )
        primary["reference_controller_task_difference_roots"] = increment
        metrics["reference-controller-task-difference"] = (float(increment[screen].mean()), "1")
        if metrics["hold-adequacy-difference"][0] < 0.05:
            reasons.append("ADEQUACY_PREPARATION_EFFECT_BELOW_DEVELOPMENT_MINIMUM")
        if metrics["hold-task-difference"][0] < 0.05:
            reasons.append("QUALIFIED_TASK_EFFECT_BELOW_DEVELOPMENT_MINIMUM")
        if not np.any(decisions[selected]["parent_eligible"][screen, 1:] == 1):
            reasons.append("NO_ELIGIBLE_NONHOLD_PREPARATION")

    selected_native = native["physical_task_success"][row, primary_parents]
    oracle = np.max(selected_native, axis=1)
    fixed_command = min(
        range(3), key=lambda s: (-selected_native[train, s].mean(), 0 if s == 1 else 1, s)
    )
    oracle_difference = float(np.mean(oracle[screen] - selected_native[screen, fixed_command]))
    unique = native["unique_successful_command"][row, primary_parents]
    unique_counts = [int(np.sum(unique[screen] == s)) for s in range(3)]
    metrics["native-oracle-versus-development-fixed-command"] = (oracle_difference, "1")
    for s, count in enumerate(unique_counts):
        metrics[f"native-unique-command-{s}-screen-roots"] = (count, "1")
    if oracle_difference < 0.05 or sum(c >= 4 for c in unique_counts) < 2:
        reasons.append("NATIVE_TARGET_ACTION_CONTACT_UNRESOLVED_OR_INSUFFICIENT")
    for p, parent in enumerate(PARENTS):
        metrics[f"native-{parent}-oracle-task-attainment"] = (
            float(native["oracle_task_success"][screen, p].mean()),
            "1",
        )
        for s in range(3):
            metrics[f"native-{parent}-raw-command-{s}-attainment"] = (
                float(native["raw_attainment"][screen, p, s].mean()),
                "1",
            )
    identity = ObjectIdentity.from_record(config.config_id, config)
    payload = write_method_arrays(TASK_ASSESSMENT_SCHEMA, observed.context, identity, groups)
    return PreparationTaskAssessmentReport(
        f"{DEVELOPMENT}.task-assessment.{observed.context}",
        observed.context,
        identity,
        ObjectIdentity.from_record(decision.report_id, decision),
        ObjectIdentity.from_record(description.report_id, description),
        ObjectIdentity.from_record(forecast.report_id, forecast),
        observed.identities,
        task.identities,
        finite_metrics(metrics),
        None if fixed_a is None else PARENTS[fixed_a],
        None if fixed_s is None else PARENTS[fixed_s],
        reference,
        PreparationGate(
            f"{DEVELOPMENT}.gate.prospective-task-feasibility.{observed.context}",
            observed.context,
            "prospective-task-feasibility",
            not reasons,
            tuple(sorted(reasons)),
        ),
        sha256(payload).hexdigest(),
    ), payload
