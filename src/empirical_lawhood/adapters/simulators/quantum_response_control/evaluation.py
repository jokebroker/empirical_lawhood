"""Parent-level inference and frozen adjudication for quantum response control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta

from ..quantum_scientific_seeds import ScientificSeedCensus, validate_scientific_seed, validate_scientific_seed_census
from .contracts import Action, Branch, PreparationUnit, QuantumResponseControlConfig, Verdict
from .receivers import ActionReceiver, ReceiverOperator, excess_variance_contributions, prefix_receivers, receiver_operator, trajectory_receiver, weighted_spearman
from .response import PredictionRow, ThresholdFamilies
from .source import EventRecord, PrefixCheckpoint


OBSERVATION_ORDER_ESTIMANDS = (
    "chi_a_h20",
    "chi_a_h4",
    "abs_chi_total_max",
    "abs_cancellation_max",
    "rho_burst_excess",
    "chi_thinning_advantage_h20",
    "chi_thinning_advantage_h4",
    "rho_history_advantage",
    "rho_wrong_time_advantage",
)

ScientificControlSeedCensus: TypeAlias = tuple[tuple[str, int, int, int], ...]


def validate_scientific_control_seeds(
    seeds: ScientificControlSeedCensus,
    *,
    unit_ids: tuple[str, ...],
    replicates: int,
) -> None:
    expected = tuple((unit_id, replicate) for replicate in (-1, *range(replicates)) for unit_id in unit_ids)
    if (
        type(seeds) is not tuple
        or len(seeds) != len(expected)
        or any(type(row) is not tuple or len(row) != 4 for row in seeds)
        or any(type(unit_id) is not str or type(replicate) is not int for unit_id, replicate, _, _ in seeds)
        or tuple((unit_id, replicate) for unit_id, replicate, _, _ in seeds) != expected
    ):
        raise ValueError("scientific control seed census differs from the complete ordered parent/replicate roster")
    allocated = []
    for _, _, thinning, history in seeds:
        for seed in (thinning, history):
            validate_scientific_seed(seed, bits=128)
            allocated.append(seed)
    if len(set(allocated)) != len(allocated):
        raise ValueError("scientific control seeds repeat independent allocations")


@dataclass(frozen=True, slots=True)
class SimultaneousInterval:
    estimand_id: str
    point: float
    lower: float
    upper: float
    standard_error: float
    critical_value: float


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryObservationOrderParent:
    unit: PreparationUnit
    prefix: PrefixCheckpoint


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryObservationOrderAnalysis:
    verdict: Verdict
    intervals: tuple[SimultaneousInterval, ...]
    observed_h20: ReceiverOperator
    observed_h4: ReceiverOperator
    reason_codes: tuple[str, ...]
    parent_count: int
    bootstrap_replicates: int
    scientific_bootstrap_seeds: ScientificSeedCensus
    scientific_control_seeds: ScientificControlSeedCensus


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryResponseQualificationEvidence:
    action_receivers: Mapping[Action, ActionReceiver]
    target_intervals: Mapping[Action, SimultaneousInterval]
    receiver_intervals: Mapping[str, SimultaneousInterval]
    prediction_rows: tuple[PredictionRow, ...]
    thresholds: ThresholdFamilies
    source_delivery_valid: bool
    future_state_leakage: bool


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryResponseQualificationAdjudication:
    verdict: Verdict
    material_actions: tuple[Action, ...]
    target_reducing_actions: tuple[Action, ...]
    proxy_status: Mapping[Action, str]
    support_fraction: float
    support_fraction_lower: float
    primary_adequacy_fraction: float
    primary_adequacy_fraction_lower: float
    full_adequacy_fraction: float
    full_adequacy_fraction_lower: float
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryProspectiveBranchSummary:
    branch: Branch
    mean_burst_a: float
    v_anom_a: float
    fano_a: float
    mean_burst_b: float
    mean_q_a: float
    mean_energy: float
    mean_work_abs: float
    mean_transport_absolute: float
    action_rate: float
    false_actions: int
    delivery_mismatches: int
    validity_passed: bool


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryProspectiveEvidence:
    summaries: Mapping[Branch, QuantumTrajectoryProspectiveBranchSummary]
    target_intervals: Mapping[str, SimultaneousInterval]
    receiver_interval: SimultaneousInterval
    inherited_law_admission_passed: bool
    preservation_passed: bool
    commitment_preceded_outcomes: bool
    coverage_passed: bool
    all_holds: bool
    precision_resolved: bool


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryProspectiveAdjudication:
    verdict: Verdict
    reason_codes: tuple[str, ...]


def _rng(scientific_seed: int) -> np.random.Generator:
    validate_scientific_seed(scientific_seed, bits=128)
    return np.random.Generator(np.random.Philox(scientific_seed))


def _stratified_bootstrap_indices(
    k_values: NDArray[np.int64],
    *,
    scientific_seed: int,
) -> NDArray[np.int64]:
    rng = _rng(scientific_seed)
    blocks = []
    for k_left in range(7):
        indices = np.flatnonzero(k_values == k_left)
        if len(indices) < 2:
            raise ValueError("bootstrap stratum contains fewer than two parents")
        blocks.append(rng.choice(indices, size=len(indices), replace=True))
    return np.concatenate(blocks)


def simultaneous_intervals(
    point: Mapping[str, float],
    bootstrap: NDArray[np.float64],
    *,
    alpha: float,
) -> tuple[SimultaneousInterval, ...]:
    names = tuple(point)
    if bootstrap.ndim != 2 or bootstrap.shape[1] != len(names):
        raise ValueError("bootstrap matrix differs from the estimand vector")
    center = np.asarray([point[name] for name in names], dtype=np.float64)
    standard_errors = np.std(bootstrap, axis=0, ddof=1)
    scale = np.maximum(standard_errors, 1e-12)
    maximum = np.max(np.abs((bootstrap - center) / scale), axis=1)
    critical = float(np.quantile(maximum, 1.0 - alpha, method="higher"))
    return tuple(
        SimultaneousInterval(
            estimand_id=name,
            point=float(center[index]),
            lower=float(center[index] - critical * scale[index]),
            upper=float(center[index] + critical * scale[index]),
            standard_error=float(standard_errors[index]),
            critical_value=critical,
        )
        for index, name in enumerate(names)
    )


def _crossfit_site_probabilities(
    parents: Sequence[QuantumTrajectoryObservationOrderParent],
) -> NDArray[np.float64]:
    site_counts = np.asarray(
        [
            prefix_receivers(
                parent.prefix,
                gamma=1.0,
                particles=6,
                l_sites=12,
            )["h20"].site_counts
            for parent in parents
        ],
        dtype=np.float64,
    )
    strata = np.asarray([parent.unit.k_left for parent in parents], dtype=np.int64)
    probabilities = np.empty_like(site_counts)
    for k_left in range(7):
        indices = np.flatnonzero(strata == k_left)
        total = site_counts[indices].sum(axis=0)
        for index in indices:
            donor = total - site_counts[index]
            donor_total = float(donor.sum())
            if donor_total <= 0:
                raise ValueError("thinning-null cross-fit donor activity is empty")
            probabilities[index] = donor / donor_total
    return probabilities


def _control_events(
    parent: QuantumTrajectoryObservationOrderParent,
    probabilities: NDArray[np.float64],
    *,
    scientific_seed: int,
    kind: str,
) -> tuple[EventRecord, ...]:
    rng = _rng(scientific_seed)
    events = parent.prefix.history_events
    if kind == "thinning":
        sites = rng.choice(12, size=len(events), p=probabilities)
    elif kind == "history":
        sites = np.asarray([event.site for event in events], dtype=np.int64)
        sites = sites[rng.permutation(len(sites))]
    else:
        raise ValueError("unknown OBSERVATION_ORDER control")
    return tuple(
        EventRecord(event.event_index, event.event_time, int(site))
        for event, site in zip(events, sites, strict=True)
    )


def _observation_metric_vector(
    parents: Sequence[QuantumTrajectoryObservationOrderParent],
    probabilities: NDArray[np.float64],
    *,
    scientific_control_seeds: Mapping[str, tuple[int, int]],
) -> tuple[dict[str, float], ReceiverOperator, ReceiverOperator]:
    k_values = [parent.unit.k_left for parent in parents]
    observed = [
        prefix_receivers(
            parent.prefix,
            gamma=1.0,
            particles=6,
            l_sites=12,
        )
        for parent in parents
    ]
    thinning = []
    shuffled_bursts = []
    for index, parent in enumerate(parents):
        thinning_events = _control_events(
            parent,
            probabilities[index],
            scientific_seed=scientific_control_seeds[parent.unit.unit_id][0],
            kind="thinning",
        )
        history_events = _control_events(
            parent,
            probabilities[index],
            scientific_seed=scientific_control_seeds[parent.unit.unit_id][1],
            kind="history",
        )
        thinning.append(
            {
                "h20": trajectory_receiver(
                    thinning_events,
                    start_time=180.0,
                    end_time=200.0,
                    l_sites=12,
                    particles=6,
                    gamma=1.0,
                ),
                "h4": trajectory_receiver(
                    thinning_events,
                    start_time=196.0,
                    end_time=200.0,
                    l_sites=12,
                    particles=6,
                    gamma=1.0,
                ),
            }
        )
        shuffled_bursts.append(
            trajectory_receiver(
                history_events,
                start_time=196.0,
                end_time=200.0,
                l_sites=12,
                particles=6,
                gamma=1.0,
            ).burst_a
        )
    operators = {
        horizon: receiver_operator(
            np.asarray([row[horizon].site_counts for row in observed]),
            k_values,
            l_sites=12,
            particles=6,
        )
        for horizon in ("h20", "h4")
    }
    thinning_operators = {
        horizon: receiver_operator(
            np.asarray([row[horizon].site_counts for row in thinning]),
            k_values,
            l_sites=12,
            particles=6,
        )
        for horizon in ("h20", "h4")
    }
    z_values = excess_variance_contributions(
        [row["h4"].n_a for row in observed],
        k_values,
    )
    rho = weighted_spearman(
        [row["h4"].burst_a for row in observed],
        z_values.tolist(),
        k_values,
    )
    rho_history = weighted_spearman(shuffled_bursts, z_values.tolist(), k_values)
    rho_wrong = weighted_spearman(
        [row["h4_previous"].burst_a for row in observed],
        z_values.tolist(),
        k_values,
    )
    metrics = {
        "chi_a_h20": operators["h20"].chi_a,
        "chi_a_h4": operators["h4"].chi_a,
        "abs_chi_total_max": max(
            abs(operators["h20"].chi_total),
            abs(operators["h4"].chi_total),
        ),
        "abs_cancellation_max": max(
            abs(operators["h20"].cancellation_residual),
            abs(operators["h4"].cancellation_residual),
        ),
        "rho_burst_excess": rho,
        "chi_thinning_advantage_h20": (operators["h20"].chi_a - thinning_operators["h20"].chi_a),
        "chi_thinning_advantage_h4": (operators["h4"].chi_a - thinning_operators["h4"].chi_a),
        "rho_history_advantage": rho - rho_history,
        "rho_wrong_time_advantage": rho - rho_wrong,
    }
    return metrics, operators["h20"], operators["h4"]


def _interval_map(
    intervals: Sequence[SimultaneousInterval],
) -> dict[str, SimultaneousInterval]:
    return {interval.estimand_id: interval for interval in intervals}


def analyze_observation(
    parents: Sequence[QuantumTrajectoryObservationOrderParent],
    config: QuantumResponseControlConfig,
    *,
    scientific_bootstrap_seeds: ScientificSeedCensus,
    scientific_control_seeds: ScientificControlSeedCensus,
    bootstrap_replicates: int | None = None,
) -> QuantumTrajectoryObservationOrderAnalysis:
    if len(parents) < config.observation_order_base_parents:
        raise ValueError("OBSERVATION_ORDER analysis is below the activated base roster")
    parent_ids = [parent.unit.unit_id for parent in parents]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("OBSERVATION_ORDER independent parent repeats")
    count = config.bootstrap_replicates if bootstrap_replicates is None else bootstrap_replicates
    if type(count) is not int or count < 2:
        raise ValueError("observation bootstrap requires at least two declared replicates")
    validate_scientific_seed_census(scientific_bootstrap_seeds, indices=tuple(range(count)), bits=128)
    validate_scientific_control_seeds(scientific_control_seeds, unit_ids=tuple(parent_ids), replicates=count)
    if {seed for _, seed in scientific_bootstrap_seeds} & {seed for _, _, thinning, history in scientific_control_seeds for seed in (thinning, history)}:
        raise ValueError("bootstrap and control scientific allocations overlap")
    bootstrap_seeds = dict(scientific_bootstrap_seeds)
    control_seeds = {
        replicate: {unit_id: (thinning, history) for unit_id, row_replicate, thinning, history in scientific_control_seeds if row_replicate == replicate}
        for replicate in (-1, *range(count))
    }
    if any(parent.prefix.cutoff_time != 200.0 for parent in parents):
        raise ValueError("OBSERVATION_ORDER prefix cutoff differs from trajectory preparation qualification")
    probabilities = _crossfit_site_probabilities(parents)
    point, operator_h20, operator_h4 = _observation_metric_vector(
        parents,
        probabilities,
        scientific_control_seeds=control_seeds[-1],
    )
    bootstrap = np.empty((count, len(OBSERVATION_ORDER_ESTIMANDS)), dtype=np.float64)
    k_values = np.asarray([parent.unit.k_left for parent in parents], dtype=np.int64)
    for replicate in range(count):
        indices = _stratified_bootstrap_indices(k_values, scientific_seed=bootstrap_seeds[replicate])
        selected = [parents[int(index)] for index in indices]
        selected_probabilities = probabilities[indices]
        metrics, _operator20, _operator4 = _observation_metric_vector(
            selected,
            selected_probabilities,
            scientific_control_seeds=control_seeds[replicate],
        )
        bootstrap[replicate] = [metrics[name] for name in OBSERVATION_ORDER_ESTIMANDS]
    intervals = simultaneous_intervals(
        {name: point[name] for name in OBSERVATION_ORDER_ESTIMANDS},
        bootstrap,
        alpha=float(config.family_alpha),
    )
    by_id = _interval_map(intervals)
    reasons = []
    if by_id["chi_a_h20"].lower < float(config.observation_order_chi_h20_gate):
        reasons.append("h20-excess-gate-failed")
    if by_id["chi_a_h4"].lower < float(config.observation_order_chi_h4_gate):
        reasons.append("h4-excess-gate-failed")
    if by_id["abs_chi_total_max"].upper > float(config.observation_order_global_equivalence):
        reasons.append("global-null-gate-failed")
    if by_id["abs_cancellation_max"].upper > float(config.observation_order_cancellation_equivalence):
        reasons.append("half-complement-cancellation-failed")
    if by_id["rho_burst_excess"].lower < float(config.observation_order_proxy_association):
        reasons.append("burst-excess-bridge-failed")
    if by_id["chi_thinning_advantage_h20"].lower < float(config.observation_order_thinning_h20_advantage):
        reasons.append("h20-thinning-control-failed")
    if by_id["chi_thinning_advantage_h4"].lower < float(config.observation_order_thinning_h4_advantage):
        reasons.append("h4-thinning-control-failed")
    if by_id["rho_history_advantage"].lower < float(config.observation_order_history_advantage):
        reasons.append("history-specificity-failed")
    if by_id["rho_wrong_time_advantage"].lower < float(config.observation_order_wrong_time_advantage):
        reasons.append("timing-specificity-failed")
    validity = all(
        parent.prefix.maximum_norm_error <= 1e-11
        and parent.prefix.maximum_particle_number_error <= 1e-12
        for parent in parents
    )
    if not validity or not all(
        math.isfinite(interval.lower) and math.isfinite(interval.upper) for interval in intervals
    ):
        verdict = Verdict.OBSERVATION_ORDER_UNEVALUABLE
        reasons.append("source-validity-or-precision-unresolved")
    elif any(reason == "global-null-gate-failed" for reason in reasons):
        verdict = Verdict.OBSERVATION_ORDER_GLOBAL_NULL_OPPOSED
    elif any(
        reason
        in {
            "burst-excess-bridge-failed",
            "history-specificity-failed",
            "timing-specificity-failed",
        }
        for reason in reasons
    ) and not any(
        reason in {"h20-excess-gate-failed", "h4-excess-gate-failed"} for reason in reasons
    ):
        verdict = Verdict.OBSERVATION_ORDER_RECEIVER_PROXY_STOP
    elif any(reason in {"h20-excess-gate-failed", "h4-excess-gate-failed"} for reason in reasons):
        verdict = Verdict.OBSERVATION_ORDER_NO_RECEIVER_OPPORTUNITY
    elif reasons:
        verdict = Verdict.OBSERVATION_ORDER_RECEIVER_ORDER_MIXED
    else:
        verdict = Verdict.OBSERVATION_ORDER_RECEIVER_OPPORTUNITY_SUPPORTED
    return QuantumTrajectoryObservationOrderAnalysis(
        verdict=verdict,
        intervals=intervals,
        observed_h20=operator_h20,
        observed_h4=operator_h4,
        reason_codes=tuple(sorted(set(reasons))),
        parent_count=len(parents),
        bootstrap_replicates=count,
        scientific_bootstrap_seeds=scientific_bootstrap_seeds,
        scientific_control_seeds=scientific_control_seeds,
    )


def one_sided_clopper_pearson_lower(
    successes: int,
    trials: int,
    *,
    alpha: float,
) -> float:
    """Return the exact one-sided binomial lower confidence bound.

    Population promotion is based on the bound, never on the observed
    fraction.  The all-success boundary is deliberately not special-cased:
    ``Beta(alpha; n, 1)`` is below one and therefore preserves finite-sample
    uncertainty.
    """

    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if successes == 0 or trials == 0:
        return 0.0
    return float(beta.ppf(alpha, successes, trials - successes + 1))


def adjudicate_response_and_local_law(
    evidence: QuantumTrajectoryResponseQualificationEvidence,
    config: QuantumResponseControlConfig,
) -> QuantumTrajectoryResponseQualificationAdjudication:
    if set(evidence.action_receivers) != set(Action):
        raise ValueError("response-law-evaluation evidence omits an action receiver")
    material: list[Action] = []
    reducing: list[Action] = []
    for action in (Action.MINUS, Action.PLUS):
        interval = evidence.target_intervals[action]
        delta = evidence.thresholds.response_effect.values["burst_a"]
        if interval.upper <= -delta:
            material.append(action)
            reducing.append(action)
        elif interval.lower >= delta:
            material.append(action)
    proxy_status: dict[Action, str] = {}
    for action in reducing:
        v_interval = evidence.receiver_intervals[f"v_anom_a.{action.value}"]
        f_interval = evidence.receiver_intervals[f"fano_a.{action.value}"]
        v_margin = evidence.thresholds.response_effect.values["z_a_ref"]
        f_margin = 0.10
        improves_v = v_interval.upper <= -v_margin
        improves_f = f_interval.upper <= -f_margin
        opposed_v = v_interval.lower >= v_margin
        opposed_f = f_interval.lower >= f_margin
        proxy_status[action] = (
            "PROXY_RECEIVER_FAITHFUL"
            if (improves_v or improves_f) and not (opposed_v or opposed_f)
            else "PROXY_RECEIVER_UNFAITHFUL"
            if opposed_v or opposed_f
            else "PROXY_RECEIVER_PARTIAL"
        )
    # The prepared parent is the independent unit.  Prediction rows may contain
    # multiple actions for the same parent and must not inflate binomial
    # replication.  A parent clears a vector gate only when every entered
    # action row clears it.
    rows_by_parent: dict[str, list[PredictionRow]] = {}
    for row in evidence.prediction_rows:
        rows_by_parent.setdefault(row.target_unit_id, []).append(row)
    total_count = len(rows_by_parent)
    supported_parent_ids = {
        parent_id
        for parent_id, rows in rows_by_parent.items()
        if rows and all(row.inside_support for row in rows)
    }
    supported = [
        row for row in evidence.prediction_rows if row.target_unit_id in supported_parent_ids
    ]
    supported_count = len(supported_parent_ids)
    primary_count = sum(
        all(row.primary_adequate is True for row in rows_by_parent[parent_id])
        for parent_id in supported_parent_ids
    )
    full_count = sum(
        all(row.adequate is True for row in rows_by_parent[parent_id])
        for parent_id in supported_parent_ids
    )
    support_fraction = supported_count / total_count if total_count else 0.0
    primary = primary_count / supported_count if supported_count else 0.0
    full = full_count / supported_count if supported_count else 0.0
    alpha = float(config.family_alpha)
    support_lower = one_sided_clopper_pearson_lower(
        supported_count,
        total_count,
        alpha=alpha,
    )
    primary_lower = one_sided_clopper_pearson_lower(
        primary_count,
        supported_count,
        alpha=alpha,
    )
    full_lower = one_sided_clopper_pearson_lower(
        full_count,
        supported_count,
        alpha=alpha,
    )
    reasons = []
    if not evidence.source_delivery_valid:
        verdict = Verdict.RESPONSE_LAW_SOURCE_OR_DELIVERY_STOP
        reasons.append("source-delivery-observation-invalid")
    elif evidence.future_state_leakage:
        verdict = Verdict.RESPONSE_LAW_SOURCE_OR_DELIVERY_STOP
        reasons.append("future-or-state-leakage")
    elif not material:
        verdict = Verdict.NO_CAUSAL_RESPONSE
        reasons.append("no-material-signed-response")
    elif reducing and not any(
        status == "PROXY_RECEIVER_FAITHFUL" for status in proxy_status.values()
    ):
        verdict = Verdict.PROXY_CAUSAL_MISMATCH
        reasons.append("target-response-not-receiver-faithful")
    elif any(not receiver.transport_valid for receiver in evidence.action_receivers.values()):
        verdict = Verdict.RESPONSE_LAW_TRANSPORT_OR_SINK_STOP
        reasons.append("transport-continuity-invalid")
    elif (
        support_lower >= float(config.chart_law_support_fraction)
        and primary_lower >= float(config.chart_law_primary_adequacy)
        and full_lower >= float(config.chart_law_full_adequacy)
        and any(status == "PROXY_RECEIVER_FAITHFUL" for status in proxy_status.values())
    ):
        verdict = Verdict.RECORD_CHART_SUPPORTED
    elif supported:
        verdict = Verdict.DENOMINATOR_LOCAL_MIXED
        reasons.append("uniform-chart_law-population-gate-failed")
    else:
        verdict = Verdict.COMPLETE_STATE_RESPONSE_ONLY
        reasons.append("record-chart-outside-support-or-inadequate")
    return QuantumTrajectoryResponseQualificationAdjudication(
        verdict=verdict,
        material_actions=tuple(material),
        target_reducing_actions=tuple(reducing),
        proxy_status=proxy_status,
        support_fraction=support_fraction,
        support_fraction_lower=support_lower,
        primary_adequacy_fraction=primary,
        primary_adequacy_fraction_lower=primary_lower,
        full_adequacy_fraction=full,
        full_adequacy_fraction_lower=full_lower,
        reason_codes=tuple(sorted(reasons)),
    )


def adjudicate_prospective(evidence: QuantumTrajectoryProspectiveEvidence) -> QuantumTrajectoryProspectiveAdjudication:
    reasons: list[str] = []
    policy = evidence.summaries[Branch.POLICY]
    if not evidence.coverage_passed:
        return QuantumTrajectoryProspectiveAdjudication(
            Verdict.PROSPECTIVE_COVERAGE_NONATTEMPT,
            ("prefix-only-coverage-failed",),
        )
    if evidence.all_holds:
        return QuantumTrajectoryProspectiveAdjudication(Verdict.PROSPECTIVE_SAFE_HOLD_ONLY, ("policy-is-hold-only",))
    if not evidence.precision_resolved:
        return QuantumTrajectoryProspectiveAdjudication(Verdict.PROSPECTIVE_UNEVALUABLE, ("precision-unresolved",))
    if not evidence.commitment_preceded_outcomes:
        return QuantumTrajectoryProspectiveAdjudication(
            Verdict.PROSPECTIVE_SOURCE_OR_DELIVERY_STOP,
            ("outcome-preceded-policy-commitment",),
        )
    if policy.delivery_mismatches or not policy.validity_passed:
        return QuantumTrajectoryProspectiveAdjudication(
            Verdict.PROSPECTIVE_SOURCE_OR_DELIVERY_STOP,
            ("source-delivery-observation-invalid",),
        )
    if policy.false_actions:
        return QuantumTrajectoryProspectiveAdjudication(
            Verdict.PROSPECTIVE_REJECTED_FALSE_ACTION,
            ("action-outside-admitted-reachable-support",),
        )
    policy_hold = evidence.target_intervals["POLICY-HOLD"]
    if policy_hold.upper >= 0:
        reasons.append("policy-does-not-improve-target")
        return QuantumTrajectoryProspectiveAdjudication(Verdict.PROSPECTIVE_REJECTED_TARGET, tuple(reasons))
    if (
        not evidence.inherited_law_admission_passed
        or not evidence.preservation_passed
        or evidence.receiver_interval.upper >= 0
    ):
        reasons.append("sink-preservation-or-receiver-gate-failed")
        return QuantumTrajectoryProspectiveAdjudication(
            Verdict.PROSPECTIVE_REJECTED_SINK_OR_PRESERVATION,
            tuple(reasons),
        )
    specificity_ids = (
        "POLICY-OPEN_LOOP_MATCHED",
        "POLICY-HISTORY_SHUFFLED",
        "POLICY-WRONG_RECEIVER",
    )
    failed_specificity = [
        estimand_id
        for estimand_id in specificity_ids
        if evidence.target_intervals[estimand_id].upper >= 0
    ]
    if failed_specificity:
        if "POLICY-OPEN_LOOP_MATCHED" in failed_specificity:
            verdict = Verdict.PROSPECTIVE_NO_INCREMENTAL_CHART_VALUE
        else:
            verdict = Verdict.PROSPECTIVE_EFFECTIVE_NOT_RECEIVER_SPECIFIC
        return QuantumTrajectoryProspectiveAdjudication(
            verdict,
            tuple(f"specificity-failed:{value}" for value in failed_specificity),
        )
    return QuantumTrajectoryProspectiveAdjudication(Verdict.PROSPECTIVE_VALIDATED_RECEIVER_ADMITTED_LOCAL, ())


__all__ = [
    'QuantumTrajectoryObservationOrderAnalysis',
    'QuantumTrajectoryObservationOrderParent',
    'QuantumTrajectoryResponseQualificationAdjudication',
    'QuantumTrajectoryResponseQualificationEvidence',
    'QuantumTrajectoryProspectiveAdjudication',
    'QuantumTrajectoryProspectiveBranchSummary',
    'QuantumTrajectoryProspectiveEvidence',
    "SimultaneousInterval",
    "adjudicate_response_and_local_law",
    'adjudicate_prospective',
    'analyze_observation',
    "simultaneous_intervals",
]
