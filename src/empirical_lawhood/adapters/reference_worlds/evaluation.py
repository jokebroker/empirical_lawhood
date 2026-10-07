"""Deterministic recovery and falsification for the R3 reference worlds."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from decimal import Decimal

from empirical_lawhood.kernel.evidence import (
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.references import ControllerDecisionKind, NamedDecimal
from empirical_lawhood.kernel.status import AdmissionStatus, ReadinessStatus, ScientificStatus
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.kernel.worlds import EvidenceUnitScope

from .contracts import (
    ReferenceCase,
    ReferenceCheck,
    ReferenceEvaluation,
    ReferenceWorldKind,
    ReferenceWorldSpec,
    assert_matches_oracle,
)


@dataclass(frozen=True)
class _Recovered:
    checks: tuple[ReferenceCheck, ...]
    metrics: tuple[NamedDecimal, ...]
    recovered: tuple[str, ...]
    rejected: tuple[str, ...]
    admission: AdmissionStatus
    controller: ControllerDecisionKind
    nominated: tuple[str, ...] = ()
    supported: tuple[str, ...] = ()
    status: ScientificStatus = ScientificStatus.SUPPORTED


def _metric(value_id: str, value: Decimal, unit: str = "1") -> NamedDecimal:
    return NamedDecimal(value_id=value_id, value=value, unit=unit)


def _metrics(*values: NamedDecimal) -> tuple[NamedDecimal, ...]:
    return tuple(sorted(values, key=lambda item: item.value_id))


def _check(check_id: str, condition: bool, reason: str) -> ReferenceCheck:
    return ReferenceCheck(
        check_id=check_id,
        passed=condition,
        reason_codes=() if condition else (reason,),
    )


def _quantitative(
    check_id: str,
    observed: Decimal,
    expected: Decimal,
    *,
    unit: str = "1",
    tolerance: Decimal = Decimal(0),
) -> ReferenceCheck:
    passed = abs(observed - expected) <= tolerance
    return ReferenceCheck(
        check_id=check_id,
        passed=passed,
        reason_codes=() if passed else ("truth-oracle-mismatch",),
        observed=observed,
        expected=expected,
        tolerance=tolerance,
        unit=unit,
    )


def _mean(values: Iterable[Decimal]) -> Decimal:
    collected = tuple(values)
    if not collected:
        raise ValueError("cannot average an empty reference series")
    return sum(collected, Decimal(0)) / Decimal(len(collected))


def _linear_fit(cases: tuple[ReferenceCase, ...], x_id: str, y_id: str) -> tuple[Decimal, Decimal]:
    count = Decimal(len(cases))
    x_values = tuple(case.value(x_id) for case in cases)
    y_values = tuple(case.value(y_id) for case in cases)
    denominator = (
        count * sum((value * value for value in x_values), Decimal(0))
        - sum(x_values, Decimal(0)) ** 2
    )
    if denominator == 0:
        raise ValueError("reference fit has zero action variation")
    slope = (
        count * sum((x * y for x, y in zip(x_values, y_values, strict=True)), Decimal(0))
        - sum(x_values, Decimal(0)) * sum(y_values, Decimal(0))
    ) / denominator
    intercept = (sum(y_values, Decimal(0)) - slope * sum(x_values, Decimal(0))) / count
    return slope, intercept


def _common_controls(world: ReferenceWorldSpec) -> tuple[ReferenceCheck, ...]:
    controls = world.controls
    receiver = next(
        quantity for quantity in world.system.quantities if quantity.quantity_id == "receiver"
    )
    cutoff = InformationCutoff(
        cutoff_id="reference-pre-action-cutoff",
        clock_id=receiver.clock_id,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal(0),
    )
    unsupported_decision = (
        ControllerDecisionKind.ACTION
        if controls.supported_action_lower
        <= controls.unsupported_action_value
        <= controls.supported_action_upper
        else ControllerDecisionKind.HOLD
    )
    robust_decision = (
        ControllerDecisionKind.ACTION
        if controls.nominal_model_safe and controls.discrepant_model_safe
        else ControllerDecisionKind.HOLD
    )
    outside_decision = (
        ControllerDecisionKind.ACTION
        if all(controls.outside_admission_gate_passes)
        else ControllerDecisionKind.HOLD
    )
    views = world.system.numerical_views
    checks = (
        _check(
            "control-controller-exploitation",
            robust_decision is ControllerDecisionKind.HOLD,
            "nominal-controller-was-not-held",
        ),
        _check(
            "control-exploration-to-confirmation",
            controls.parent_visibility is VisibilityCeiling.OUTCOME_VISIBLE
            and controls.fresh_visibility is VisibilityCeiling.PROSPECTIVE,
            "fresh-evidence-boundary-lost",
        ),
        _check(
            "control-leakage",
            inherited_visibility((controls.parent_visibility,), OutcomeAccess.EVALUATION_REVEALED)
            is VisibilityCeiling.OUTCOME_VISIBLE,
            "outcome-visibility-was-laundered",
        ),
        _check(
            "control-model-discrepancy",
            controls.nominal_model_safe and not controls.discrepant_model_safe,
            "model-discrepancy-control-did-not-separate",
        ),
        _check(
            "control-multiplicity",
            controls.raw_signal_p > controls.multiplicity_cutoff,
            "chance-signal-cleared-multiplicity",
        ),
        _check(
            "control-negative",
            not world.accepts_identity(world.reference_id, "0" * 64),
            "incorrect-content-identity-was-accepted",
        ),
        _check(
            "control-no-admission",
            outside_decision is ControllerDecisionKind.HOLD,
            "failed-gate-was-compensated",
        ),
        _check(
            "control-numerical-refinement",
            len(views) >= 2
            and all(
                view.evidence_scope is EvidenceUnitScope.NESTED_NUMERICAL_VIEW
                and view.physical_preparation_id == world.system.independent_unit.unit_id
                for view in views
            ),
            "numerical-view-inflated-replication",
        ),
        _check(
            "control-positive",
            world.accepts_identity(world.reference_id, world.fingerprint()),
            "correct-content-identity-was-rejected",
        ),
        _check(
            "control-search-family-completeness",
            controls.registered_analysis_ids == controls.executed_analysis_ids,
            "registered-analysis-family-was-incomplete",
        ),
        _check(
            "control-unsupported-action",
            unsupported_decision is ControllerDecisionKind.HOLD,
            "unsupported-action-was-issued",
        ),
        _check(
            "control-wrong-action",
            controls.wrong_action_effect < controls.minimum_response_effect,
            "wrong-action-falsifier-retained-response",
        ),
        _check(
            "control-wrong-time",
            not cutoff.allows(receiver.availability),
            "receiver-was-visible-before-action",
        ),
    )
    return tuple(sorted(checks, key=lambda check: check.check_id))


def _stable_linear(world: ReferenceWorldSpec) -> _Recovered:
    slope, intercept = _linear_fit(world.cases, "action", "receiver")
    residual = max(
        abs(case.value("receiver") - (intercept + slope * case.value("action")))
        for case in world.cases
    )
    supported = slope == Decimal(2) and intercept == Decimal(1) and residual == 0
    return _Recovered(
        checks=(
            _quantitative("linear-intercept", intercept, Decimal(1)),
            _quantitative("linear-residual", residual, Decimal(0)),
            _quantitative("linear-slope", slope, Decimal(2)),
        ),
        metrics=_metrics(
            _metric("intercept", intercept),
            _metric("max-residual", residual),
            _metric("slope", slope),
        ),
        recovered=("finite-horizon-response", "rank-one-response") if supported else (),
        rejected=("zero-response-law",) if supported else (),
        admission=AdmissionStatus.ADMITTED if supported else AdmissionStatus.UNEVALUABLE,
        controller=ControllerDecisionKind.ACTION if supported else ControllerDecisionKind.HOLD,
        status=ScientificStatus.SUPPORTED if supported else ScientificStatus.NOT_SUPPORTED,
    )


def _nonlinear_charted(world: ReferenceWorldSpec) -> _Recovered:
    charts = sorted({case.denominator_cell_id for case in world.cases})
    curvatures: list[Decimal] = []
    chart_actions: dict[str, tuple[Decimal, ...]] = {}
    for chart in charts:
        cases = tuple(
            sorted(
                (case for case in world.cases if case.denominator_cell_id == chart),
                key=lambda case: case.value("action"),
            )
        )
        actions = tuple(case.value("action") for case in cases)
        responses = tuple(case.value("receiver") for case in cases)
        step = actions[1] - actions[0]
        curvatures.append((responses[2] - Decimal(2) * responses[1] + responses[0]) / step**2)
        chart_actions[chart] = actions
    curvature = _mean(curvatures)
    gap_width = min(chart_actions["right-chart"]) - max(chart_actions["left-chart"])
    supported = len(charts) == 2 and curvature == Decimal(2) and gap_width > 0
    return _Recovered(
        checks=(
            _check("atlas-gap", gap_width > 0, "atlas-gap-not-detected"),
            _quantitative("chart-count", Decimal(len(charts)), Decimal(2)),
            _quantitative("local-curvature", curvature, Decimal(2)),
        ),
        metrics=_metrics(
            _metric("chart-count", Decimal(len(charts))),
            _metric("curvature", curvature),
            _metric("gap-width", gap_width),
        ),
        recovered=("atlas-boundary", "charted-curvature") if supported else (),
        rejected=("global-smooth-law",) if supported else (),
        admission=AdmissionStatus.PARTIAL if supported else AdmissionStatus.UNEVALUABLE,
        controller=ControllerDecisionKind.ACTION if supported else ControllerDecisionKind.HOLD,
        status=ScientificStatus.SUPPORTED if supported else ScientificStatus.NOT_SUPPORTED,
    )


def _hysteretic_memory(world: ReferenceWorldSpec) -> _Recovered:
    coefficient, _ = _linear_fit(world.cases, "history", "receiver")
    response_values = tuple(case.value("receiver") for case in world.cases)
    memoryless_range = max(response_values) - min(response_values)
    supported = coefficient == Decimal("1.5") and memoryless_range == Decimal(3)
    return _Recovered(
        checks=(
            _quantitative("history-coefficient", coefficient, Decimal("1.5")),
            _check(
                "memoryless-recurrence",
                memoryless_range > 0,
                "memoryless-law-falsifier-did-not-displace",
            ),
        ),
        metrics=_metrics(
            _metric("history-coefficient", coefficient),
            _metric("memoryless-range", memoryless_range),
        ),
        recovered=("history-dependent-response",) if supported else (),
        rejected=("memoryless-law",) if supported else (),
        admission=AdmissionStatus.ADMITTED if supported else AdmissionStatus.UNEVALUABLE,
        controller=ControllerDecisionKind.ACTION if supported else ControllerDecisionKind.HOLD,
        status=ScientificStatus.SUPPORTED if supported else ScientificStatus.NOT_SUPPORTED,
    )


def _multirate_delayed(world: ReferenceWorldSpec) -> _Recovered:
    applied_delay = _mean(
        case.value("applied-time") - case.value("requested-time") for case in world.cases
    )
    receiver_delay = _mean(
        case.value("receiver-time") - case.value("applied-time") for case in world.cases
    )
    supported = applied_delay == Decimal(2) and receiver_delay == Decimal(3)
    return _Recovered(
        checks=(
            _quantitative("applied-delay", applied_delay, Decimal(2), unit="s"),
            _quantitative("receiver-delay", receiver_delay, Decimal(3), unit="s"),
            _check(
                "same-label-rejection",
                applied_delay != 0 and receiver_delay != 0,
                "nominal-row-labels-were-treated-as-one-clock",
            ),
        ),
        metrics=_metrics(
            _metric("applied-delay", applied_delay, "s"),
            _metric("receiver-delay", receiver_delay, "s"),
        ),
        recovered=("applied-clock-delay", "receiver-clock-delay") if supported else (),
        rejected=("nominal-label-clock-law",) if supported else (),
        admission=AdmissionStatus.ADMITTED if supported else AdmissionStatus.UNEVALUABLE,
        controller=ControllerDecisionKind.ACTION if supported else ControllerDecisionKind.HOLD,
        status=ScientificStatus.SUPPORTED if supported else ScientificStatus.NOT_SUPPORTED,
    )


def _hybrid_partial(world: ReferenceWorldSpec) -> _Recovered:
    mode_zero = tuple(case for case in world.cases if case.value("mode") == 0)
    mode_one = tuple(case for case in world.cases if case.value("mode") == 1)
    threshold = (
        max(case.value("state") for case in mode_zero)
        + min(case.value("state") for case in mode_one)
    ) / Decimal(2)
    output_states: dict[Decimal, set[Decimal]] = {}
    for case in world.cases:
        output_states.setdefault(case.value("receiver"), set()).add(case.value("state"))
    ambiguous = sum(len(states) > 1 for states in output_states.values())
    supported = threshold == Decimal(1) and ambiguous == 2
    return _Recovered(
        checks=(
            _quantitative("event-threshold", threshold, Decimal(1)),
            _quantitative("partial-observation", Decimal(ambiguous), Decimal(2)),
        ),
        metrics=_metrics(
            _metric("ambiguous-output-count", Decimal(ambiguous)),
            _metric("event-threshold", threshold),
        ),
        recovered=("hybrid-event", "partial-observation") if supported else (),
        rejected=("continuous-fully-observed-law",) if supported else (),
        admission=AdmissionStatus.PARTIAL if supported else AdmissionStatus.UNEVALUABLE,
        controller=ControllerDecisionKind.HOLD,
        status=ScientificStatus.SUPPORTED if supported else ScientificStatus.NOT_SUPPORTED,
    )


def _strong_empty_admission(world: ReferenceWorldSpec) -> _Recovered:
    ordered = tuple(sorted(world.cases, key=lambda case: case.value("action")))
    response_gain = ordered[-1].value("receiver") - ordered[0].value("receiver")
    worst_sink = max(case.value("sink") for case in ordered)
    sink_limit = min(case.value("sink-limit") for case in ordered)
    strong = response_gain >= Decimal(10)
    empty = worst_sink > sink_limit
    return _Recovered(
        checks=(
            _check(
                "admission-intersection",
                empty,
                "failed-sink-did-not-empty-admission",
            ),
            _quantitative("strong-response", response_gain, Decimal(10)),
        ),
        metrics=_metrics(
            _metric("response-gain", response_gain),
            _metric("worst-sink", worst_sink),
        ),
        recovered=("strong-response",) if strong else (),
        rejected=("scalar-reward-admission",) if strong and empty else (),
        admission=AdmissionStatus.EMPTY if empty else AdmissionStatus.ADMITTED,
        controller=ControllerDecisionKind.HOLD if empty else ControllerDecisionKind.ACTION,
        status=ScientificStatus.SUPPORTED if strong else ScientificStatus.NOT_SUPPORTED,
    )


def _coupled_interfaces(world: ReferenceWorldSpec) -> _Recovered:
    quantities = {quantity.quantity_id: quantity for quantity in world.system.quantities}
    valid_errors = quantities["coupling-output"].compatibility_errors(quantities["coupling-input"])
    invalid_target = replace(quantities["coupling-input"], native_unit="mV")
    invalid_errors = quantities["coupling-output"].compatibility_errors(invalid_target)
    valid_count = len(world.system.interfaces) if not valid_errors else 0
    supported = valid_count == 1 and invalid_errors == ("NATIVE_UNIT_MISMATCH",)
    return _Recovered(
        checks=(
            _quantitative("invalid-interface", Decimal(len(invalid_errors)), Decimal(1)),
            _quantitative("valid-interface", Decimal(valid_count), Decimal(1)),
        ),
        metrics=_metrics(
            _metric("invalid-interface-errors", Decimal(len(invalid_errors))),
            _metric("valid-interface-count", Decimal(valid_count)),
        ),
        recovered=("composable-interface",) if supported else (),
        rejected=("unit-mismatched-interface",) if supported else (),
        admission=AdmissionStatus.ADMITTED if supported else AdmissionStatus.UNEVALUABLE,
        controller=ControllerDecisionKind.ACTION if supported else ControllerDecisionKind.HOLD,
        status=ScientificStatus.SUPPORTED if supported else ScientificStatus.NOT_SUPPORTED,
    )


def _drifting_degrading(world: ReferenceWorldSpec) -> _Recovered:
    epochs = {
        epoch: tuple(case for case in world.cases if case.value("epoch") == epoch)
        for epoch in (Decimal(0), Decimal(1))
    }
    pre_slope, _ = _linear_fit(epochs[Decimal(0)], "action", "receiver")
    post_slope, _ = _linear_fit(epochs[Decimal(1)], "action", "receiver")
    change = abs(pre_slope - post_slope)
    drift = change > Decimal("0.25")
    return _Recovered(
        checks=(
            _check(
                "chart-invalidation",
                drift,
                "drift-did-not-invalidate-chart",
            ),
            _quantitative("drift-detection", change, Decimal("1.5")),
        ),
        metrics=_metrics(
            _metric("post-drift-slope", post_slope),
            _metric("pre-drift-slope", pre_slope),
            _metric("slope-change", change),
        ),
        recovered=("chart-invalidation", "response-drift") if drift else (),
        rejected=("stationary-law",) if drift else (),
        admission=AdmissionStatus.EMPTY if drift else AdmissionStatus.ADMITTED,
        controller=ControllerDecisionKind.HOLD if drift else ControllerDecisionKind.ACTION,
        status=ScientificStatus.MIXED if drift else ScientificStatus.NOT_SUPPORTED,
    )


def _phase_model_difference(cases: tuple[ReferenceCase, ...], phase: str) -> Decimal:
    responses = tuple(case.value("receiver") for case in cases if phase in case.tags)
    return max(responses) - min(responses)


def _observational_equivalence(world: ReferenceWorldSpec) -> _Recovered:
    observational = _phase_model_difference(world.cases, "observational")
    intervention = _phase_model_difference(world.cases, "intervention")
    supported = observational == 0 and intervention == Decimal(1)
    return _Recovered(
        checks=(
            _quantitative("intervention-divergence", intervention, Decimal(1)),
            _quantitative("observational-equivalence", observational, Decimal(0)),
        ),
        metrics=_metrics(
            _metric("intervention-difference", intervention),
            _metric("observational-difference", observational),
        ),
        recovered=("intervention-discrimination",) if supported else (),
        rejected=("observational-identification",) if supported else (),
        admission=AdmissionStatus.NOT_EVALUATED,
        controller=ControllerDecisionKind.HOLD,
        status=ScientificStatus.SUPPORTED if supported else ScientificStatus.NOT_SUPPORTED,
    )


def _matrix_rank_2x2(matrix: tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal]]) -> int:
    flattened = (*matrix[0], *matrix[1])
    if all(value == 0 for value in flattened):
        return 0
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return 2 if determinant != 0 else 1


def _matrix_for_view(
    world: ReferenceWorldSpec, view_id: str
) -> tuple[tuple[Decimal, Decimal], tuple[Decimal, Decimal]]:
    values: dict[tuple[int, int], Decimal] = {}
    for case in world.cases:
        if case.numerical_view_id == view_id:
            key = (int(case.value("row")), int(case.value("column")))
            values[key] = case.value("receiver")
    return ((values[(0, 0)], values[(0, 1)]), (values[(1, 0)], values[(1, 1)]))


def _numerical_false_structure(world: ReferenceWorldSpec) -> _Recovered:
    coarse = _matrix_rank_2x2(_matrix_for_view(world, "coarse-view"))
    fine = _matrix_rank_2x2(_matrix_for_view(world, "fine-view"))
    change = abs(coarse - fine)
    unstable = change > 0
    return _Recovered(
        checks=(
            _quantitative("rank-instability", Decimal(change), Decimal(1)),
            _check(
                "refined-basin-rejection",
                unstable and fine < coarse,
                "coarse-admission-basin-survived-refinement",
            ),
        ),
        metrics=_metrics(
            _metric("coarse-rank", Decimal(coarse)),
            _metric("fine-rank", Decimal(fine)),
            _metric("rank-change", Decimal(change)),
        ),
        recovered=("numerical-structural-instability",) if unstable else (),
        rejected=("coarse-admission-basin", "coarse-rank-law") if unstable else (),
        admission=AdmissionStatus.EMPTY if unstable else AdmissionStatus.ADMITTED,
        controller=ControllerDecisionKind.HOLD if unstable else ControllerDecisionKind.ACTION,
        status=ScientificStatus.NOT_SUPPORTED if unstable else ScientificStatus.SUPPORTED,
    )


def _rare_decisive_sink(world: ReferenceWorldSpec) -> _Recovered:
    mean_receiver = _mean(case.value("receiver") for case in world.cases)
    mean_sink = _mean(case.value("sink") for case in world.cases)
    worst_sink = max(case.value("sink") for case in world.cases)
    sink_limit = min(case.value("sink-limit") for case in world.cases)
    hidden = mean_sink <= sink_limit < worst_sink
    return _Recovered(
        checks=(
            _check(
                "average-performance-decoy",
                mean_receiver > 0 and mean_sink <= sink_limit,
                "average-decoy-was-not-favourable",
            ),
            _check(
                "rare-sink-falsifier",
                hidden,
                "rare-decisive-sink-was-not-detected",
            ),
        ),
        metrics=_metrics(
            _metric("mean-receiver", mean_receiver),
            _metric("mean-sink", mean_sink),
            _metric("worst-sink", worst_sink),
        ),
        recovered=("rare-decisive-sink",) if hidden else (),
        rejected=("mean-score-admission",) if hidden else (),
        admission=AdmissionStatus.EMPTY if hidden else AdmissionStatus.ADMITTED,
        controller=ControllerDecisionKind.HOLD if hidden else ControllerDecisionKind.ACTION,
        status=ScientificStatus.SUPPORTED if hidden else ScientificStatus.NOT_SUPPORTED,
    )


def _discrepancy_exploitation(world: ReferenceWorldSpec) -> _Recovered:
    sinks = {case.model_id: case.value("sink") for case in world.cases}
    nominal = sinks["nominal-model"]
    worst = max(sinks.values())
    limit = min(case.value("sink-limit") for case in world.cases)
    exploited = nominal <= limit < worst
    return _Recovered(
        checks=(
            _check(
                "nominal-exploitation",
                exploited,
                "nominal-controller-was-robust",
            ),
            _check(
                "robust-model-set",
                worst > limit,
                "model-set-did-not-force-hold",
            ),
        ),
        metrics=_metrics(
            _metric("model-set-worst-sink", worst),
            _metric("nominal-sink", nominal),
        ),
        recovered=("discrepancy-exploitation",) if exploited else (),
        rejected=("nominal-controller",) if exploited else (),
        admission=AdmissionStatus.EMPTY if exploited else AdmissionStatus.ADMITTED,
        controller=ControllerDecisionKind.HOLD if exploited else ControllerDecisionKind.ACTION,
        status=ScientificStatus.SUPPORTED if exploited else ScientificStatus.NOT_SUPPORTED,
    )


def _latency_boundary(world: ReferenceWorldSpec) -> _Recovered:
    case = world.cases[0]
    latency = case.value("latency")
    cadence = case.value("cadence")
    error = case.value("prediction-error")
    overrun = latency - cadence
    boundary = world.system.readiness is ReadinessStatus.COMPUTABILITY_BOUNDARY
    return _Recovered(
        checks=(
            _check(
                "accuracy-is-insufficient",
                error == 0 and boundary,
                "offline-accuracy-overrode-compute-boundary",
            ),
            _quantitative("latency-deadline", overrun, Decimal("0.1"), unit="s"),
        ),
        metrics=_metrics(
            _metric("latency", latency, "s"),
            _metric("latency-overrun", overrun, "s"),
            _metric("prediction-error", error),
        ),
        recovered=("compute-envelope-boundary",) if boundary else (),
        rejected=("offline-accuracy-controller",) if boundary else (),
        admission=AdmissionStatus.EMPTY if boundary else AdmissionStatus.ADMITTED,
        controller=ControllerDecisionKind.HOLD if boundary else ControllerDecisionKind.ACTION,
        status=ScientificStatus.SUPPORTED if boundary else ScientificStatus.NOT_SUPPORTED,
    )


def _planted_relational_anomaly(world: ReferenceWorldSpec) -> _Recovered:
    eligible = tuple(
        case
        for case in world.cases
        if all(
            case.value(value_id) == 1
            for value_id in (
                "denominator-match",
                "gauge-match",
                "horizon-match",
                "recurrence",
            )
        )
        and case.value("effect") >= Decimal("0.5")
    )
    selected = max(eligible, key=lambda case: case.value("effect")) if eligible else None
    selected_effect = selected.value("effect") if selected is not None else Decimal(0)
    largest_raw = max(case.value("effect") for case in world.cases)
    recovered = selected is not None and selected.model_id == "planted-relation"
    return _Recovered(
        checks=(
            _check(
                "decoy-rejection",
                recovered and selected_effect < largest_raw,
                "attractive-decoy-was-selected",
            ),
            _check(
                "planted-anomaly-recovery",
                recovered,
                "planted-relational-anomaly-was-missed",
            ),
        ),
        metrics=_metrics(
            _metric("largest-raw-effect", largest_raw),
            _metric("selected-effect", selected_effect),
        ),
        recovered=("planted-relational-anomaly",) if recovered else (),
        rejected=("attractive-decoy",) if recovered else (),
        admission=AdmissionStatus.NOT_EVALUATED,
        controller=ControllerDecisionKind.HOLD,
        nominated=("planted-relation",) if recovered else (),
        status=ScientificStatus.PARTIAL if recovered else ScientificStatus.NOT_SUPPORTED,
    )


def _null_search_family(world: ReferenceWorldSpec) -> _Recovered:
    probabilities = tuple(case.value("p-value") for case in world.cases)
    minimum = min(probabilities)
    corrected_cutoff = Decimal("0.004")
    selected_count = sum(value <= corrected_cutoff for value in probabilities)
    honest_null = selected_count == 0
    return _Recovered(
        checks=(
            _check(
                "complete-null-family",
                len(probabilities) == 12,
                "search-family-was-incomplete",
            ),
            _check(
                "multiplicity-null",
                honest_null and minimum < Decimal("0.05"),
                "chance-effect-became-preferred",
            ),
        ),
        metrics=_metrics(
            _metric("corrected-selection-count", Decimal(selected_count)),
            _metric("minimum-p", minimum),
        ),
        recovered=("honest-null-search",) if honest_null else (),
        rejected=("preferred-hypothesis",) if honest_null else (),
        admission=AdmissionStatus.NOT_EVALUATED,
        controller=ControllerDecisionKind.HOLD,
        status=ScientificStatus.NOT_SUPPORTED if honest_null else ScientificStatus.MIXED,
    )


def _retrospective_defeated(world: ReferenceWorldSpec) -> _Recovered:
    retrospective = next(case for case in world.cases if "outcome-visible" in case.tags)
    fresh = next(case for case in world.cases if "fresh-evidence" in case.tags)
    retrospective_effect = retrospective.value("effect")
    fresh_effect = fresh.value("effect")
    defeated = (
        retrospective_effect > 0 and fresh_effect == 0 and fresh.value("p-value") > Decimal("0.05")
    )
    return _Recovered(
        checks=(
            _check(
                "fresh-defeat",
                defeated,
                "fresh-evidence-did-not-defeat-mechanism",
            ),
            _check(
                "parent-claim-immutability",
                world.controls.parent_visibility is VisibilityCeiling.OUTCOME_VISIBLE,
                "outcome-visible-parent-was-promoted",
            ),
        ),
        metrics=_metrics(
            _metric("fresh-effect", fresh_effect),
            _metric("retrospective-effect", retrospective_effect),
        ),
        recovered=("prospective-falsification",) if defeated else (),
        rejected=("retrospective-mechanism",) if defeated else (),
        admission=AdmissionStatus.NOT_EVALUATED,
        controller=ControllerDecisionKind.HOLD,
        nominated=("retrospective-mechanism",),
        status=ScientificStatus.NOT_SUPPORTED if defeated else ScientificStatus.MIXED,
    )


_EVALUATORS: dict[ReferenceWorldKind, Callable[[ReferenceWorldSpec], _Recovered]] = {
    ReferenceWorldKind.STABLE_LINEAR: _stable_linear,
    ReferenceWorldKind.NONLINEAR_CHARTED: _nonlinear_charted,
    ReferenceWorldKind.HYSTERETIC_MEMORY: _hysteretic_memory,
    ReferenceWorldKind.MULTIRATE_DELAYED: _multirate_delayed,
    ReferenceWorldKind.HYBRID_PARTIAL: _hybrid_partial,
    ReferenceWorldKind.STRONG_EMPTY_ADMISSION: _strong_empty_admission,
    ReferenceWorldKind.COUPLED_INTERFACES: _coupled_interfaces,
    ReferenceWorldKind.DRIFTING_DEGRADING: _drifting_degrading,
    ReferenceWorldKind.OBSERVATIONAL_EQUIVALENCE: _observational_equivalence,
    ReferenceWorldKind.NUMERICAL_FALSE_STRUCTURE: _numerical_false_structure,
    ReferenceWorldKind.RARE_DECISIVE_SINK: _rare_decisive_sink,
    ReferenceWorldKind.DISCREPANCY_EXPLOITATION: _discrepancy_exploitation,
    ReferenceWorldKind.LATENCY_BOUNDARY: _latency_boundary,
    ReferenceWorldKind.PLANTED_RELATIONAL_ANOMALY: _planted_relational_anomaly,
    ReferenceWorldKind.NULL_SEARCH_FAMILY: _null_search_family,
    ReferenceWorldKind.RETROSPECTIVE_DEFEATED: _retrospective_defeated,
}


def _evaluate_unchecked(world: ReferenceWorldSpec) -> ReferenceEvaluation:
    recovered = _EVALUATORS[world.kind](world)
    parent_fingerprint = hashlib.sha256(f"{world.reference_id}:parent-claim".encode()).hexdigest()
    checks = tuple(
        sorted((*_common_controls(world), *recovered.checks), key=lambda item: item.check_id)
    )
    return ReferenceEvaluation(
        evaluation_id=f"{world.reference_id}-evaluation",
        reference_id=world.reference_id,
        reference_fingerprint=world.fingerprint(),
        checks=checks,
        metrics=recovered.metrics,
        recovered_structure_ids=tuple(sorted(recovered.recovered)),
        rejected_law_ids=tuple(sorted(recovered.rejected)),
        admission_status=recovered.admission,
        controller_decision=recovered.controller,
        nominated_hypothesis_ids=tuple(sorted(recovered.nominated)),
        supported_hypothesis_ids=tuple(sorted(recovered.supported)),
        scientific_status=recovered.status,
        parent_claim_fingerprint_before=parent_fingerprint,
        parent_claim_fingerprint_after=parent_fingerprint,
        exploration_visibility=world.controls.parent_visibility,
        fresh_evidence_visibility=world.controls.fresh_visibility,
    )


def evaluate_reference_world(world: ReferenceWorldSpec) -> ReferenceEvaluation:
    evaluation = _evaluate_unchecked(world)
    assert_matches_oracle(world, evaluation)
    return evaluation


def evaluate_reference_worlds(
    worlds: tuple[ReferenceWorldSpec, ...],
) -> tuple[ReferenceEvaluation, ...]:
    evaluations = tuple(evaluate_reference_world(world) for world in worlds)
    return tuple(sorted(evaluations, key=lambda result: result.reference_id))
