"Reveal raw whole-root operands, call the existing controller-use owner, then paired analysis."

from dataclasses import asdict, replace
from decimal import Decimal as D
from typing import Any
import json
import numpy as np
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.runtime.controller_evaluation_trajectory import RevealedTrajectoryBundle, TrajectoryCallbackOutcome
from empirical_lawhood.runtime.controller_evaluation_nested import NestedControllerUseEvaluator
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode, paired_operands
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator, measured_labels
from empirical_lawhood.adapters.simulators.reactor_causal_response.campaign import ControlCustodyPort, prerequisite
from .campaign_records import TimedReactorConfirmationEnvelope, EmpiricalRootAnalysis, EmpiricalContributionResult, EmpiricalComparatorSources
from .experiment_records import EmpiricalQualificationTerminal, EmpiricalDiscoveryEnvelope
from .control_owner import trajectory_plan
from .control_services import trajectory_reducer
from .comparison import EpisodeEndpoints, endpoints, common_history_errors, exposure_contact, paired_analysis
from .calibration import root_score
from .numerical import CausalFeatures, Observation, select
from .comparators import published_controller
from .config import ARMS, CONFIRMATION_ROOT_COUNT, COMPARATIVE_SCOPE, NATIVE_BENCHMARK_ARMS
from .serialization import read_fit, json_bytes


def _completion(e: NativeEpisode) -> float | None:
    if not e.complete or not np.isfinite(e.grid).all():
        return None
    completed = np.flatnonzero((e.grid[:, 3] >= 0.999 * 287.3) & (e.grid[:, 4] >= 0.98))
    return float(e.grid[completed[0], 0]) if len(completed) else None


def _labels(e: NativeEpisode) -> np.ndarray:
    return (
        measured_labels(e.grid[:, 0], e.grid[:, 1], e.grid[:, 4], e.grid[:, 3], e.dt)
        if e.complete and np.isfinite(e.grid).all()
        else np.empty((0, 3))
    )


def timing_summary(
    raw: TimedReactorConfirmationEnvelope, nominal: dict[str, NativeEpisode]
) -> dict[str, Any]:
    result = {}
    for timing in raw.telemetry:
        wall = np.asarray(timing.callback_wall_seconds, dtype=float)
        if len(wall) != len(nominal[timing.arm].callback_cpu):
            raise ValueError("callback wall/CPU measurement census differs")
        result[timing.arm] = dict(
            measured_callbacks=len(wall),
            callback_wall_seconds_sum=float(wall.sum()),
            callback_wall_seconds_max=float(wall.max()) if len(wall) else None,
            callback_wall_seconds_median=float(np.median(wall)) if len(wall) else None,
            callback_wall_seconds_p95=float(np.quantile(wall, 0.95)) if len(wall) else None,
            episode_wall_seconds=float(timing.episode_wall_seconds),
            worker_peak_rss_bytes=timing.worker_peak_rss_bytes,
        )
    return result


def _contact(
    nominal: dict[str, NativeEpisode],
    discovery: dict[str, Any],
    q: float,
    comparators: EmpiricalComparatorSources,
    params: dict[str, Any],
) -> dict[str, Any]:
    model = read_fit(discovery["fits"][discovery["selected"]])
    rivals = {
        arm: read_fit(discovery["fits"][discovery["rivals"][i]])
        for i, arm in enumerate(("F0", "F1"))
    }
    result: dict[str, Any] = {}
    primary = nominal["EL"]
    for arm in ARMS[1:]:
        other = nominal[arm]
        result[arm] = dict(
            applied=exposure_contact(
                primary.requests, other.requests, primary.exposure, other.exposure
            )
            if primary.complete and other.complete
            else None,
            missing_whole_episode=not (primary.complete and other.complete),
            comparable_forecast=arm in ("F0", "F1", "MARGIN", "FIXED"),
        )
    state, previous = CausalFeatures(), (0.0, 316.0)
    fixed = published_controller("SCHEDULED_BACKOFF_HALF", comparators.source("SCHEDULED_BACKOFF_HALF"))
    fixed.reset(params)
    counts = {
        arm: dict(
            queries=0,
            different_points=0,
            different_intervals=0,
            different_support=0,
            different_eligible_sets=0,
            different_shadow_choices=0,
        )
        for arm in ("F0", "F1", "MARGIN", "FIXED")
    }
    for k, values in enumerate(primary.observations):
        observation = Observation(*map(float, values))
        state.append(observation)
        projections = Actuator().project(observation, previous)
        left = select((observation,), previous[1], projections, model, q, causal_state=state)
        for arm in counts:
            right = select(
                (observation,),
                previous[1],
                projections,
                rivals.get(arm, model),
                q,
                causal_state=state,
                temperature_halfwidth=0.25 if arm == "MARGIN" else None,
            )
            choice = right.selected
            if arm == "FIXED":
                feed, jacket = fixed.step(
                    values[0],
                    dict(t_reactor_k=values[1], t_jacket_k=values[2], dosed_kg=values[3]),
                    10.0,
                )
                eligible = [
                    c
                    for c in right.candidates
                    if not c.reasons and c.projection.action == min(c.aliases)
                ]
                choice = (
                    min(
                        eligible,
                        key=lambda c: (
                            abs(c.projection.requested[0] - feed) / 0.032
                            + abs(c.projection.requested[1] - jacket) / 68.3,
                            c.projection.action,
                        ),
                    ).projection.action
                    if eligible
                    else None
                )
            count = counts[arm]
            count["queries"] += 9
            count["different_points"] += sum(
                a.point != b.point for a, b in zip(left.candidates, right.candidates, strict=True)
            )
            count["different_intervals"] += sum(
                (a.point, a.halfwidth) != (b.point, b.halfwidth)
                for a, b in zip(left.candidates, right.candidates, strict=True)
            )
            count["different_support"] += sum(
                ("OUTSIDE_SUPPORT" in a.reasons) != ("OUTSIDE_SUPPORT" in b.reasons)
                for a, b in zip(left.candidates, right.candidates, strict=True)
            )
            count["different_eligible_sets"] += tuple(
                not c.reasons for c in left.candidates
            ) != tuple(not c.reasons for c in right.candidates)
            count["different_shadow_choices"] += left.selected != choice
        if k < len(primary.stages):
            previous = (float(primary.stages[k, 2]), float(primary.stages[k, 3]))
    for arm, count in counts.items():
        result[arm]["common_history"] = count
    # REF's own observed histories and exact requests, never a quantized proxy.
    reference = nominal["REF"]
    labels = _labels(reference)
    audit = dict(
        assigned=2880,
        observed_proposals=len(reference.requests),
        exact_menu=0,
        supported=0,
        admitted=0,
        false_admissions=0,
        safe_refusals=0,
        unavailable=0,
    )
    state, previous = CausalFeatures(), (0.0, 316.0)
    for k, values in enumerate(reference.observations[: len(reference.requests)]):
        o = Observation(*map(float, values))
        state.append(o)
        projections = Actuator().project(o, previous)
        ids = [p.action for p in projections if p.requested == tuple(reference.requests[k])]
        if not ids or k >= len(labels):
            audit["unavailable"] += 1
        else:
            audit["exact_menu"] += 1
            decision = select((o,), previous[1], projections, model, q, causal_state=state)
            candidate = decision.candidates[min(ids)]
            supported = "OUTSIDE_SUPPORT" not in candidate.reasons
            admitted = not candidate.reasons
            audit["supported"] += supported
            audit["admitted"] += admitted
            audit["false_admissions"] += admitted and labels[k, 0] > 356.2
            audit["safe_refusals"] += supported and not admitted and labels[k, 0] <= 356.2
        previous = (float(reference.stages[k, 2]), float(reference.stages[k, 3]))
    audit["unavailable"] += 2880 - len(reference.requests)
    result["reference_exact_proposal_audit"] = audit
    return result


def analyze_root(
    raw: TimedReactorConfirmationEnvelope,
    qualification: EmpiricalQualificationTerminal,
    discovery: EmpiricalDiscoveryEnvelope,
    comparators: EmpiricalComparatorSources,
    params: dict[str, Any],
    custody: ControlCustodyPort,
) -> EmpiricalRootAnalysis:
    if (
        raw.qualification
        != ObjectIdentity.from_record("reactor-empirical-qualification", qualification)
        or raw.discovery != ObjectIdentity.from_record("reactor-empirical-discovery", discovery)
        or raw.prerequisite != prerequisite(qualification)
    ):
        raise ValueError("analysis substitutes its prospective qualification/discovery parents")
    identity = ObjectIdentity.from_record(raw.root, raw)
    if raw.prerequisite is not None:
        return EmpiricalRootAnalysis(raw.root, identity, None, "[]", "{}", raw.prerequisite)
    report = qualification.result
    assert report is not None and raw.acquisition is not None and report.payload.q is not None
    episodes = tuple(NativeEpisode.from_envelope(e) for e in raw.acquisition.episodes)
    nominal = {e.episode: e for e in episodes if e.dt == 1}
    refined = next(e for e in episodes if e.dt == 0.5)
    primary, reference = nominal["EL"], nominal["REF"]
    plan = trajectory_plan(report)
    model = read_fit(json.loads(report.payload.model_json))
    q = float(report.payload.q)
    causal = paired_operands(primary, refined, model, q)
    measured = (_labels(primary), _labels(refined))
    callbacks = tuple(
        TrajectoryCallbackOutcome(
            k,
            compiled,
            tick,
            link,
            tuple(
                tuple(
                    NamedDecimal(key, D(repr(float(value))), unit)
                    for key, value, unit in zip(
                        ("reactor-peak-temperature", "reactor-end-conversion", "reactor-dose"),
                        view[k],
                        ("K", "1", "kg"),
                        strict=True,
                    )
                )
                if k < len(view)
                else ()
                for view in measured
            ),
        )
        for k, (compiled, tick, link) in enumerate(raw.children)
    )
    time, reference_time = _completion(primary), _completion(reference)
    bundle = RevealedTrajectoryBundle(
        f"{raw.root}.revealed",
        raw.root,
        ObjectIdentity.from_record(plan.evaluation_plan_id, plan),
        callbacks,
        primary.complete,
        None if time is None else D(repr(time)),
        None if reference_time is None else D(repr(reference_time)),
        (identity,),
        root_score(causal, model, consumer_q=q).value is not None,
    )
    owner = NestedControllerUseEvaluator(plan.evaluator, trajectory_reducer(plan.evaluator))
    prospective_evaluation = owner.evaluate_trajectory_unit(
        plan=plan, bundle=bundle, reader=custody.open_control_store(raw.root)
    )
    saved = json.loads(discovery.result_json)
    rows = []
    coverage = {}
    for arm in ARMS:
        e = nominal[arm]
        row = endpoints(
            raw.root,
            arm,
            e.grid,
            float(e.callback_cpu.sum()) if len(e.callback_cpu) else None,
            reference_time,
            completed=e.complete,
        )
        if arm in ("EL", "F0", "F1"):
            fitted = (
                model if arm == "EL" else read_fit(saved["fits"][saved["rivals"][int(arm[-1])]])
            )
            errors, support = common_history_errors(
                causal.features, measured[0], fitted, causal.clocks, causal.actions
            )
            coverage[arm] = support
            if errors is not None:
                row = replace(row, temperature_mae=errors[0], conversion_mae=errors[1])
        rows.append(row)
    if rows[0].task_success != prospective_evaluation.J:
        raise ValueError(
            "raw endpoint reducer disagrees with the existing controller-use owner's nominal task predicate"
        )
    contact = _contact(nominal, saved, q, comparators, params)
    contact["common_history_support_fraction"] = coverage
    contact["cost"] = {
        "native_batches": len(episodes),
        "native_steps": sum(len(e.exposure) * int(10 / e.dt) for e in episodes),
        "owner_cpu_seconds": float(sum(raw.owner_cpu_seconds)),
        "online_cpu_seconds": {
            e.episode: float(e.callback_cpu.sum()) for e in episodes if e.dt == 1
        },
        "independent_roots": 1,
        "timing": timing_summary(raw, nominal),
    }
    return EmpiricalRootAnalysis(
        raw.root,
        identity,
        prospective_evaluation,
        json_bytes([asdict(r) for r in rows]).decode(),
        json_bytes(contact).decode(),
        None,
    )


def analyze_cohort(
    roots: tuple[EmpiricalRootAnalysis, ...], qualification: EmpiricalQualificationTerminal
) -> EmpiricalContributionResult:
    roots = tuple(sorted(roots, key=lambda r: r.root))
    stop = prerequisite(qualification)
    matrix: dict[str, Any] = {
        key: {
            "status": "UNENTERED" if stop else "DESCRIPTIVE",
            "analysis_scope": COMPARATIVE_SCOPE,
            "confirmatory_superiority_claim": False,
            "overall_superiority_claim": False,
        }
        for key in (
            "completion",
            "safety-productivity",
            "prediction",
            "measured-cost",
            "qualification",
            "history-law-information",
            "action-selection",
            "reference-policy-admission",
        )
    }
    matrix["prediction"]["published_forecasts"] = "UNAVAILABLE_NO_COMPARABLE_CONTRACT"
    matrix["measured-cost"]["historical_development_cost"] = "UNAVAILABLE"
    matrix["completion"].update(
        native_benchmark_arms=NATIVE_BENCHMARK_ARMS,
        deferred_native_benchmark_arms=tuple(a for a in ARMS if a not in NATIVE_BENCHMARK_ARMS),
    )
    if stop:
        return EmpiricalContributionResult(roots, None, "[]", json_bytes(matrix).decode(), stop)
    report = qualification.result
    assert report is not None
    plan = trajectory_plan(report)
    owner = NestedControllerUseEvaluator(plan.evaluator, trajectory_reducer(plan.evaluator))
    cohort = owner.reduce_trajectory_cohort(
        plan=plan, units=tuple(r.prospective_evaluation for r in roots if r.prospective_evaluation is not None)
    )
    rows = tuple(EpisodeEndpoints(**e) for r in roots for e in json.loads(r.endpoints_json))
    contrasts = paired_analysis(rows)
    for key, endpoints_ in (
        ("completion", ("J",)),
        ("safety-productivity", ("unsafe-or-unobserved", "restricted-completion-time")),
        ("prediction", ("temperature-mae", "conversion-mae")),
        ("measured-cost", ("log-online-cpu-ratio",)),
    ):
        matrix[key]["contrasts"] = [asdict(c) for c in contrasts if c.endpoint in endpoints_]
    contacts = [json.loads(r.contact_json) for r in roots]
    matrix["qualification"].update(
        status="DESCRIPTIVE_PRIMARY_ONLY",
        law_status=report.qualification.scientific_status.value,
        assigned=CONFIRMATION_ROOT_COUNT,
        A=cohort.A_count,
        J=cohort.J_count,
        C=cohort.C_count,
        panel_mean_ratio=None if cohort.panel_mean_ratio is None else str(cohort.panel_mean_ratio),
        panel_mean_passed=cohort.panel_mean_passed,
        comparator_qualification="NOT_INHERITED",
    )
    for key, arms in (
        ("history-law-information", ("F0", "F1")),
        ("action-selection", ("MARGIN", "FIXED")),
    ):
        matrix[key]["treatments"] = {}
        for arm in arms:
            common = {
                name: sum(c[arm]["common_history"][name] for c in contacts)
                for name in contacts[0][arm]["common_history"]
            }
            applied = [c[arm]["applied"] for c in contacts]
            matrix[key]["treatments"][arm] = dict(
                status="CONTACTED"
                if common["different_shadow_choices"]
                else "UNCONTACTED_SELECTION",
                common_history=common,
                paired_whole_episodes=sum(v is not None for v in applied),
                missing_whole_episodes=sum(v is None for v in applied),
                root_applied_contacts=applied,
                contrasts=[asdict(c) for c in contrasts if c.comparator == arm],
            )
    reference = {
        name: sum(c["reference_exact_proposal_audit"][name] for c in contacts)
        for name in contacts[0]["reference_exact_proposal_audit"]
    }
    matrix["reference-policy-admission"].update(
        status="DESCRIPTIVE_EXACT_PROPOSALS_ONLY"
        if reference["exact_menu"]
        else "UNAVAILABLE_NO_EXACT_PROPOSALS",
        counts=reference,
    )
    matrix["measured-cost"]["confirmation"] = dict(
        native_batches=sum(c["cost"]["native_batches"] for c in contacts),
        native_steps=sum(c["cost"]["native_steps"] for c in contacts),
        owner_and_delivery_cpu_seconds=sum(c["cost"]["owner_cpu_seconds"] for c in contacts),
        independent_roots=CONFIRMATION_ROOT_COUNT,
        full_study_cost_source="RETAINED_TASK_COST_RECORDS_AND_NATIVE_CONTAINER_TELEMETRY",
        online_cpu_contrast_scope="NUMERICAL_POLICY_CALLBACK_ONLY",
        owner_and_delivery_cpu_scope="SEPARATE_OWNER_PERSISTENCE_AND_NATIVE_DELIVERY_PROCESS_CPU",
        whole_procedure_cost_superiority_claim=False,
        timing_scope="CALLBACK_NUMERICAL_POLICY_WALL_TIME; EPISODE_WALL_INCLUDES_NATIVE_AND_OWNER; RSS_IS_CUMULATIVE_WORKER_PEAK",
        root_timing_source="RETAINED_ROOT_ANALYSES_CONTACT_COST_TIMING",
        episode_wall_seconds_by_arm={
            arm: sum(c["cost"]["timing"][arm]["episode_wall_seconds"] for c in contacts)
            for arm in ARMS
        },
        maximum_worker_rss_bytes=max(
            c["cost"]["timing"][arm]["worker_peak_rss_bytes"] for c in contacts for arm in ARMS
        ),
    )
    return EmpiricalContributionResult(
        roots,
        cohort,
        json_bytes([asdict(c) for c in contrasts]).decode(),
        json_bytes(matrix).decode(),
        None,
    )
