"""History-clustered geometry, interval, transition, and terminal primitives.

These records are action-free and source-agnostic.  They retain primitive
geometry overlap and observation failures separately, while treating a whole
preparation history as the independent unit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from random import Random
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


class GeometryObservationStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


class GeometryState(StrEnum):
    NEITHER = "NEITHER"
    FULL_ONLY = "FULL_ONLY"
    ROBUST_00_ONLY = "ROBUST_00_ONLY"
    BOTH = "BOTH"


class ScientificView(StrEnum):
    PRIMARY = "PRIMARY"
    FINE = "FINE"


class GeometryEventKind(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    HISTORY_QUALIFIED_REENTRY = "HISTORY_QUALIFIED_REENTRY"
    POST_CUTOFF_EXIT_REENTRY = "POST_CUTOFF_EXIT_REENTRY"


class IntervalDisposition(StrEnum):
    EVENT = "EVENT"
    RIGHT_CENSORED = "RIGHT_CENSORED"
    UNKNOWN = "UNKNOWN"
    NOT_AT_RISK = "NOT_AT_RISK"


class ViewDecision(StrEnum):
    QUALITATIVE_PASS = "QUALITATIVE_PASS"
    NOT_CONFIRMED = "NOT_CONFIRMED"
    STRATUM_UNEVALUABLE = "STRATUM_UNEVALUABLE"
    NUMERICAL_UNEVALUABLE = "NUMERICAL_UNEVALUABLE"
    OPERATIONALLY_UNEVALUABLE = "OPERATIONALLY_UNEVALUABLE"


class TwoViewTerminal(StrEnum):
    CONFIRMED = "CONFIRMED"
    MIXED_VIEWS = "MIXED_VIEWS"
    NOT_CONFIRMED = "NOT_CONFIRMED"
    STRATUM_UNEVALUABLE = "STRATUM_UNEVALUABLE"
    NUMERICAL_UNEVALUABLE = "NUMERICAL_UNEVALUABLE"
    OPERATIONALLY_UNEVALUABLE = "OPERATIONALLY_UNEVALUABLE"
    PLATFORM_OBSTRUCTION = "PLATFORM_OBSTRUCTION"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class GeometryTick(CanonicalRecord):
    """One receiver-lattice observation; availability is not geometry state."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/geometry-tick'

    tick_id: str
    history_id: str
    preparation_family_id: str
    acquisition_group_id: str
    scientific_view: ScientificView
    receiver_tick: int
    receiver_time: Decimal
    time_unit: str
    frame_id: str
    receiver_direction: str
    required_window_start_tick: int
    status: GeometryObservationStatus
    full_intersection_pass: bool | None
    robust_00_pass: bool | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("tick_id", "history_id", "acquisition_group_id", "frame_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_stable_id(self.preparation_family_id, field_name="preparation_family_id")
        if (
            not isinstance(self.receiver_tick, int)
            or isinstance(self.receiver_tick, bool)
            or self.receiver_tick < 0
            or not isinstance(self.required_window_start_tick, int)
            or isinstance(self.required_window_start_tick, bool)
            or self.required_window_start_tick < 0
            or self.required_window_start_tick > self.receiver_tick
        ):
            raise ValueError("geometry tick receiver lattice differs")
        validate_decimal(self.receiver_time, field_name="receiver_time", minimum=Decimal(0))
        if not self.time_unit or not self.receiver_direction:
            raise ValueError("geometry tick units or receiver direction are absent")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        flags = (self.full_intersection_pass, self.robust_00_pass)
        if self.status is GeometryObservationStatus.AVAILABLE:
            if any(not isinstance(value, bool) for value in flags) or self.reason_codes:
                raise ValueError("available geometry tick lacks primitive flags")
        elif any(value is not None for value in flags) or not self.reason_codes:
            raise ValueError("unavailable or invalid geometry tick carries geometry state")

    @property
    def geometry_state(self) -> GeometryState | None:
        if self.status is not GeometryObservationStatus.AVAILABLE:
            return None
        if self.full_intersection_pass and self.robust_00_pass:
            return GeometryState.BOTH
        if self.full_intersection_pass:
            return GeometryState.FULL_ONLY
        if self.robust_00_pass:
            return GeometryState.ROBUST_00_ONLY
        return GeometryState.NEITHER


@dataclass(frozen=True, slots=True)
class HistoryViewTrace(CanonicalRecord):
    """One dense view of one physical history on a frozen receiver lattice."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/history-view-trace'

    trace_id: str
    history_id: str
    preparation_family_id: str
    acquisition_group_id: str
    scientific_view: ScientificView
    branch_id: str
    receiver_lattice: tuple[int, ...]
    ticks: tuple[GeometryTick, ...]

    def __post_init__(self) -> None:
        for name in ("trace_id", "history_id", "acquisition_group_id", "branch_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_stable_id(self.preparation_family_id, field_name="preparation_family_id")
        if (
            not self.receiver_lattice
            or self.receiver_lattice != tuple(sorted(set(self.receiver_lattice)))
            or self.receiver_lattice != tuple(value.receiver_tick for value in self.ticks)
        ):
            raise ValueError("history trace must retain every frozen receiver tick")
        for tick in self.ticks:
            if (
                tick.history_id != self.history_id
                or tick.preparation_family_id != self.preparation_family_id
                or tick.acquisition_group_id != self.acquisition_group_id
                or tick.scientific_view is not self.scientific_view
            ):
                raise ValueError("history trace lineage differs")


@dataclass(frozen=True, slots=True)
class CutoffMembership(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/cutoff-membership'

    cutoff_tick: int
    available: bool
    full_intersection: bool
    robust_00: bool
    overlap: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.cutoff_tick < 0:
            raise ValueError("cutoff tick cannot be negative")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.overlap != (self.full_intersection and self.robust_00):
            raise ValueError("cutoff overlap differs from primitive strata")
        if self.available:
            if self.reason_codes:
                raise ValueError("available cutoff has failure reasons")
        elif self.full_intersection or self.robust_00 or self.overlap or not self.reason_codes:
            raise ValueError("unavailable cutoff carries stratum membership")


def cutoff_membership(trace: HistoryViewTrace, cutoff_tick: int) -> CutoffMembership:
    by_tick = {value.receiver_tick: value for value in trace.ticks}
    tick = by_tick.get(cutoff_tick)
    if tick is None:
        raise ValueError("cutoff is outside the frozen receiver lattice")
    if tick.status is not GeometryObservationStatus.AVAILABLE:
        return CutoffMembership(
            cutoff_tick=cutoff_tick,
            available=False,
            full_intersection=False,
            robust_00=False,
            overlap=False,
            reason_codes=(f"CUTOFF_{tick.status.value}",),
        )
    full = bool(tick.full_intersection_pass)
    robust = bool(tick.robust_00_pass)
    return CutoffMembership(cutoff_tick, True, full, robust, full and robust, ())


@dataclass(frozen=True, slots=True)
class EventInterval(CanonicalRecord):
    """One first-event interval on the observed receiver lattice."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/event-interval'

    history_id: str
    acquisition_group_id: str
    scientific_view: ScientificView
    branch_id: str
    event_kind: GeometryEventKind
    cutoff_tick: int
    horizon_tick: int
    disposition: IntervalDisposition
    lower_tick: int | None
    upper_tick: int | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("history_id", "acquisition_group_id", "branch_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if not 0 <= self.cutoff_tick < self.horizon_tick:
            raise ValueError("event interval horizon differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is IntervalDisposition.EVENT:
            if (
                self.lower_tick is None
                or self.upper_tick is None
                or not self.cutoff_tick <= self.lower_tick < self.upper_tick <= self.horizon_tick
                or self.reason_codes
            ):
                raise ValueError("observed event bracket differs")
        elif self.disposition is IntervalDisposition.RIGHT_CENSORED:
            if (
                self.lower_tick != self.horizon_tick
                or self.upper_tick is not None
                or self.reason_codes
            ):
                raise ValueError("right-censor record differs")
        elif self.disposition is IntervalDisposition.UNKNOWN:
            if (
                self.lower_tick is None
                or self.upper_tick is None
                or not self.cutoff_tick <= self.lower_tick < self.upper_tick <= self.horizon_tick
                or not self.reason_codes
            ):
                raise ValueError("unknown interval localization differs")
        elif self.lower_tick is not None or self.upper_tick is not None or not self.reason_codes:
            raise ValueError("not-at-risk interval differs")


@dataclass(frozen=True, slots=True)
class HistoryEventSet(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/history-event-set'

    entry: EventInterval
    exit: EventInterval
    history_qualified_reentry: EventInterval
    post_cutoff_exit_reentry: EventInterval
    retained_through_horizon: bool | None

    def __post_init__(self) -> None:
        values = (
            self.entry,
            self.exit,
            self.history_qualified_reentry,
            self.post_cutoff_exit_reentry,
        )
        identities = {
            (value.history_id, value.acquisition_group_id, value.scientific_view, value.branch_id)
            for value in values
        }
        if len(identities) != 1:
            raise ValueError("history event set lineage differs")
        expected = (
            True
            if self.exit.disposition is IntervalDisposition.RIGHT_CENSORED
            else False
            if self.exit.disposition is IntervalDisposition.EVENT
            else None
        )
        if self.retained_through_horizon is not expected:
            raise ValueError("retention differs from first-exit interval")


def _event(
    trace: HistoryViewTrace,
    kind: GeometryEventKind,
    cutoff: int,
    horizon: int,
    disposition: IntervalDisposition,
    lower: int | None = None,
    upper: int | None = None,
    reasons: tuple[str, ...] = (),
) -> EventInterval:
    return EventInterval(
        history_id=trace.history_id,
        acquisition_group_id=trace.acquisition_group_id,
        scientific_view=trace.scientific_view,
        branch_id=trace.branch_id,
        event_kind=kind,
        cutoff_tick=cutoff,
        horizon_tick=horizon,
        disposition=disposition,
        lower_tick=lower,
        upper_tick=upper,
        reason_codes=reasons,
    )


def _first_change(
    trace: HistoryViewTrace,
    *,
    kind: GeometryEventKind,
    cutoff_tick: int,
    horizon_tick: int,
    start_tick: int,
    from_full: bool,
    to_full: bool,
) -> EventInterval:
    ticks = tuple(
        value for value in trace.ticks if start_tick <= value.receiver_tick <= horizon_tick
    )
    if not ticks or ticks[0].receiver_tick != start_tick or ticks[-1].receiver_tick != horizon_tick:
        raise ValueError("event bounds are outside the frozen receiver lattice")
    for left, right in zip(ticks, ticks[1:]):
        if (
            left.status is not GeometryObservationStatus.AVAILABLE
            or right.status is not GeometryObservationStatus.AVAILABLE
        ):
            return _event(
                trace,
                kind,
                cutoff_tick,
                horizon_tick,
                IntervalDisposition.UNKNOWN,
                left.receiver_tick,
                right.receiver_tick,
                ("OBSERVATION_GAP",),
            )
        if left.full_intersection_pass is from_full and right.full_intersection_pass is to_full:
            return _event(
                trace,
                kind,
                cutoff_tick,
                horizon_tick,
                IntervalDisposition.EVENT,
                left.receiver_tick,
                right.receiver_tick,
            )
    return _event(
        trace,
        kind,
        cutoff_tick,
        horizon_tick,
        IntervalDisposition.RIGHT_CENSORED,
        horizon_tick,
    )


def _history_reentry_eligible(trace: HistoryViewTrace, cutoff_tick: int) -> bool:
    candidate = False
    prior = tuple(value for value in trace.ticks if value.receiver_tick <= cutoff_tick)
    for left, right in zip(prior, prior[1:]):
        if (
            left.status is not GeometryObservationStatus.AVAILABLE
            or right.status is not GeometryObservationStatus.AVAILABLE
        ):
            candidate = False
        elif left.full_intersection_pass is True and right.full_intersection_pass is False:
            candidate = True
    return candidate


def derive_history_events(
    trace: HistoryViewTrace,
    *,
    cutoff_tick: int,
    horizon_tick: int,
) -> HistoryEventSet:
    """Derive entry/exit/re-entry without right-endpoint time imputation."""

    membership = cutoff_membership(trace, cutoff_tick)
    if horizon_tick not in trace.receiver_lattice or horizon_tick <= cutoff_tick:
        raise ValueError("event horizon is outside the frozen receiver lattice")
    if not membership.available:
        unknowns = tuple(
            _event(
                trace,
                kind,
                cutoff_tick,
                horizon_tick,
                IntervalDisposition.UNKNOWN,
                cutoff_tick,
                trace.receiver_lattice[trace.receiver_lattice.index(cutoff_tick) + 1],
                membership.reason_codes,
            )
            for kind in GeometryEventKind
        )
        return HistoryEventSet(
            entry=unknowns[0],
            exit=unknowns[1],
            history_qualified_reentry=unknowns[2],
            post_cutoff_exit_reentry=unknowns[3],
            retained_through_horizon=None,
        )

    entry_risk = membership.robust_00 and not membership.full_intersection
    exit_risk = membership.full_intersection
    entry = (
        _first_change(
            trace,
            kind=GeometryEventKind.ENTRY,
            cutoff_tick=cutoff_tick,
            horizon_tick=horizon_tick,
            start_tick=cutoff_tick,
            from_full=False,
            to_full=True,
        )
        if entry_risk
        else _event(
            trace,
            GeometryEventKind.ENTRY,
            cutoff_tick,
            horizon_tick,
            IntervalDisposition.NOT_AT_RISK,
            reasons=("NOT_IN_ENTRY_STRATUM",),
        )
    )
    exit_interval = (
        _first_change(
            trace,
            kind=GeometryEventKind.EXIT,
            cutoff_tick=cutoff_tick,
            horizon_tick=horizon_tick,
            start_tick=cutoff_tick,
            from_full=True,
            to_full=False,
        )
        if exit_risk
        else _event(
            trace,
            GeometryEventKind.EXIT,
            cutoff_tick,
            horizon_tick,
            IntervalDisposition.NOT_AT_RISK,
            reasons=("NOT_IN_RETENTION_STRATUM",),
        )
    )
    history_reentry = (
        replace(entry, event_kind=GeometryEventKind.HISTORY_QUALIFIED_REENTRY)
        if entry_risk and _history_reentry_eligible(trace, cutoff_tick)
        else _event(
            trace,
            GeometryEventKind.HISTORY_QUALIFIED_REENTRY,
            cutoff_tick,
            horizon_tick,
            IntervalDisposition.NOT_AT_RISK,
            reasons=("HISTORY_REENTRY_CONDITION_FALSE",),
        )
    )
    if exit_interval.disposition is IntervalDisposition.EVENT:
        assert exit_interval.upper_tick is not None
        post_reentry = _first_change(
            trace,
            kind=GeometryEventKind.POST_CUTOFF_EXIT_REENTRY,
            cutoff_tick=cutoff_tick,
            horizon_tick=horizon_tick,
            start_tick=exit_interval.upper_tick,
            from_full=False,
            to_full=True,
        )
    elif exit_interval.disposition is IntervalDisposition.UNKNOWN:
        post_reentry = replace(
            exit_interval,
            event_kind=GeometryEventKind.POST_CUTOFF_EXIT_REENTRY,
        )
    else:
        post_reentry = _event(
            trace,
            GeometryEventKind.POST_CUTOFF_EXIT_REENTRY,
            cutoff_tick,
            horizon_tick,
            IntervalDisposition.NOT_AT_RISK,
            reasons=("NO_OBSERVED_POST_CUTOFF_EXIT",),
        )
    retained = (
        True
        if exit_interval.disposition is IntervalDisposition.RIGHT_CENSORED
        else False
        if exit_interval.disposition is IntervalDisposition.EVENT
        else None
    )
    return HistoryEventSet(entry, exit_interval, history_reentry, post_reentry, retained)


@dataclass(frozen=True, slots=True)
class TransitionCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/transition-cell'

    from_state: GeometryState
    to_state: GeometryState
    numerator: int
    at_risk_denominator: int
    probability: Decimal | None

    def __post_init__(self) -> None:
        if not 0 <= self.numerator <= self.at_risk_denominator:
            raise ValueError("transition counts differ")
        expected = (
            Decimal(self.numerator) / Decimal(self.at_risk_denominator)
            if self.at_risk_denominator
            else None
        )
        if self.probability != expected:
            raise ValueError("transition probability differs from its risk denominator")


@dataclass(frozen=True, slots=True)
class TransitionReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/transition-report'

    preparation_family_id: str
    scientific_view: ScientificView
    receiver_lag: int
    physical_history_count: int
    cells: tuple[TransitionCell, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.preparation_family_id, field_name="preparation_family_id")
        if self.receiver_lag <= 0 or self.physical_history_count < 0:
            raise ValueError("transition lag or history count differs")
        pairs = tuple((value.from_state, value.to_state) for value in self.cells)
        expected = tuple((left, right) for left in GeometryState for right in GeometryState)
        if pairs != expected:
            raise ValueError("transition matrix cell order differs")
        for state in GeometryState:
            row = tuple(value for value in self.cells if value.from_state is state)
            denominator = row[0].at_risk_denominator
            if any(value.at_risk_denominator != denominator for value in row):
                raise ValueError("transition row denominators differ")
            if sum(value.numerator for value in row) != denominator:
                raise ValueError("transition row does not normalize")


def transition_report(
    traces: tuple[HistoryViewTrace, ...],
    *,
    preparation_family_id: str,
    scientific_view: ScientificView,
    receiver_lag: int,
) -> TransitionReport:
    selected = tuple(
        value
        for value in traces
        if value.preparation_family_id == preparation_family_id
        and value.scientific_view is scientific_view
    )
    counts = {(left, right): 0 for left in GeometryState for right in GeometryState}
    denominators = {state: 0 for state in GeometryState}
    for trace in selected:
        by_tick = {value.receiver_tick: value for value in trace.ticks}
        for left in trace.ticks:
            right = by_tick.get(left.receiver_tick + receiver_lag)
            if right is None or left.geometry_state is None or right.geometry_state is None:
                continue
            counts[(left.geometry_state, right.geometry_state)] += 1
            denominators[left.geometry_state] += 1
    cells = tuple(
        TransitionCell(
            left,
            right,
            counts[(left, right)],
            denominators[left],
            (
                Decimal(counts[(left, right)]) / Decimal(denominators[left])
                if denominators[left]
                else None
            ),
        )
        for left in GeometryState
        for right in GeometryState
    )
    return TransitionReport(
        preparation_family_id,
        scientific_view,
        receiver_lag,
        len({value.history_id for value in selected}),
        cells,
    )


def physical_history_support(
    traces: tuple[HistoryViewTrace, ...],
    *,
    preparation_family_ids: tuple[str, ...],
) -> tuple[int, tuple[tuple[str, int], ...]]:
    """Count histories once regardless of ticks, branches, or nested views."""

    require_sorted_unique_strings(
        preparation_family_ids,
        field_name="preparation_family_ids",
        allow_empty=False,
    )
    lineage: dict[str, tuple[str, str]] = {}
    for trace in traces:
        observed = (trace.preparation_family_id, trace.acquisition_group_id)
        previous = lineage.setdefault(trace.history_id, observed)
        if previous != observed:
            raise ValueError("physical history has conflicting lineage")
    if {value[0] for value in lineage.values()} - set(preparation_family_ids):
        raise ValueError("history names a family outside the supplied roster")
    per_family = tuple(
        (family, sum(value[0] == family for value in lineage.values()))
        for family in preparation_family_ids
    )
    return len(lineage), per_family


@dataclass(frozen=True, slots=True)
class ClusterOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/cluster-outcome'

    history_id: str
    preparation_family_id: str
    value: bool | None

    def __post_init__(self) -> None:
        validate_stable_id(self.history_id, field_name="history_id")
        validate_stable_id(self.preparation_family_id, field_name="preparation_family_id")
        if self.value is not None and not isinstance(self.value, bool):
            raise ValueError("cluster outcome must be Boolean or unknown")


@dataclass(frozen=True, slots=True)
class ClusteredProbabilityInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/clustered-probability-interval'

    physical_history_count: int
    known_risk_count: int
    estimate: Decimal | None
    lower: Decimal | None
    upper: Decimal | None
    confidence_level: Decimal
    defined_replicates: int
    undefined_replicates: int
    support_sufficient: bool

    def __post_init__(self) -> None:
        if not 0 <= self.known_risk_count <= self.physical_history_count:
            raise ValueError("clustered interval support differs")
        validate_decimal(self.confidence_level, field_name="confidence_level")
        if not Decimal(0) < self.confidence_level < Decimal(1):
            raise ValueError("clustered confidence level must lie in (0,1)")
        values = (self.estimate, self.lower, self.upper)
        if self.support_sufficient != all(value is not None for value in values):
            raise ValueError("clustered interval evaluability differs")
        for value in values:
            if value is not None and not Decimal(0) <= value <= Decimal(1):
                raise ValueError("clustered probability lies outside [0,1]")
        if self.defined_replicates < 0 or self.undefined_replicates < 0:
            raise ValueError("clustered replicate counts cannot be negative")


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def clustered_probability_interval(
    outcomes: tuple[ClusterOutcome, ...],
    *,
    seed_sha256: str,
    replicates: int,
    confidence_level: Decimal,
    minimum_overall_risk: int,
    minimum_per_family_risk: int,
    preparation_family_ids: tuple[str, ...],
) -> ClusteredProbabilityInterval:
    """Family-stratified bootstrap over complete histories, retaining undefined draws."""

    validate_sha256(seed_sha256, field_name="seed_sha256")
    validate_decimal(confidence_level, field_name="confidence_level")
    if (
        replicates <= 0
        or minimum_overall_risk <= 0
        or minimum_per_family_risk <= 0
        or not Decimal(0) < confidence_level < Decimal(1)
    ):
        raise ValueError("clustered inference contract differs")
    by_id: dict[str, ClusterOutcome] = {}
    for outcome in outcomes:
        previous = by_id.setdefault(outcome.history_id, outcome)
        if previous != outcome:
            raise ValueError("history cluster outcome conflicts across repeated views")
    families = preparation_family_ids
    if not isinstance(families, tuple) or not families or len(set(families)) != len(families):
        raise ValueError("clustered inference requires the complete original scientific family acquisition order")
    for family in families:
        validate_stable_id(family, field_name="preparation_family_id")
    if {value.preparation_family_id for value in by_id.values()} - set(families):
        raise ValueError("cluster outcome names a family outside the supplied roster")
    groups = {
        family: tuple(value for value in by_id.values() if value.preparation_family_id == family)
        for family in families
    }
    known_by_family = {
        family: tuple(value for value in group if value.value is not None)
        for family, group in groups.items()
    }
    known_count = sum(len(value) for value in known_by_family.values())
    support = known_count >= minimum_overall_risk and all(
        len(known_by_family[family]) >= minimum_per_family_risk for family in families
    )
    if not support:
        return ClusteredProbabilityInterval(
            len(by_id), known_count, None, None, None, confidence_level, 0, replicates, False
        )
    family_estimates = [
        sum(value.value is True for value in known_by_family[family]) / len(known_by_family[family])
        for family in families
    ]
    estimate = sum(family_estimates) / len(family_estimates)
    rng = Random(int(seed_sha256, 16))
    draws: list[float] = []
    undefined = 0
    for _ in range(replicates):
        estimates: list[float] = []
        for family in families:
            group = groups[family]
            sampled = tuple(group[rng.randrange(len(group))] for _ in group)
            known = tuple(value for value in sampled if value.value is not None)
            if not known:
                break
            estimates.append(sum(value.value is True for value in known) / len(known))
        if len(estimates) != len(families):
            undefined += 1
        else:
            draws.append(sum(estimates) / len(estimates))
    if not draws:
        return ClusteredProbabilityInterval(
            len(by_id), known_count, None, None, None, confidence_level, 0, undefined, False
        )
    alpha = (1.0 - float(confidence_level)) / 2.0
    return ClusteredProbabilityInterval(
        physical_history_count=len(by_id),
        known_risk_count=known_count,
        estimate=Decimal(repr(estimate)),
        lower=Decimal(repr(_quantile(draws, alpha))),
        upper=Decimal(repr(_quantile(draws, 1.0 - alpha))),
        confidence_level=confidence_level,
        defined_replicates=len(draws),
        undefined_replicates=undefined,
        support_sufficient=True,
    )


@dataclass(frozen=True, slots=True)
class ViewQualitativeDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/view-qualitative-decision'

    scientific_view: ScientificView
    operational_complete: bool
    numerical_valid: bool
    support_sufficient: bool
    retention_pass: bool
    reentry_pass: bool
    decision: ViewDecision

    def __post_init__(self) -> None:
        expected = (
            ViewDecision.OPERATIONALLY_UNEVALUABLE
            if not self.operational_complete
            else ViewDecision.NUMERICAL_UNEVALUABLE
            if not self.numerical_valid
            else ViewDecision.STRATUM_UNEVALUABLE
            if not self.support_sufficient
            else ViewDecision.QUALITATIVE_PASS
            if self.retention_pass and self.reentry_pass
            else ViewDecision.NOT_CONFIRMED
        )
        if self.decision is not expected:
            raise ValueError("view decision differs from noncompensating predicates")


def adjudicate_two_views(
    primary: ViewQualitativeDecision,
    fine: ViewQualitativeDecision,
    *,
    platform_obstruction: bool = False,
    stopped: bool = False,
) -> TwoViewTerminal:
    if (
        primary.scientific_view is not ScientificView.PRIMARY
        or fine.scientific_view is not ScientificView.FINE
    ):
        raise ValueError("two-view adjudication requires separate primary and fine reports")
    if platform_obstruction:
        return TwoViewTerminal.PLATFORM_OBSTRUCTION
    if stopped:
        return TwoViewTerminal.STOPPED
    decisions = (primary.decision, fine.decision)
    for value, terminal in (
        (ViewDecision.OPERATIONALLY_UNEVALUABLE, TwoViewTerminal.OPERATIONALLY_UNEVALUABLE),
        (ViewDecision.NUMERICAL_UNEVALUABLE, TwoViewTerminal.NUMERICAL_UNEVALUABLE),
        (ViewDecision.STRATUM_UNEVALUABLE, TwoViewTerminal.STRATUM_UNEVALUABLE),
    ):
        if value in decisions:
            return terminal
    passed = sum(value is ViewDecision.QUALITATIVE_PASS for value in decisions)
    if passed == 2:
        return TwoViewTerminal.CONFIRMED
    if passed == 1:
        return TwoViewTerminal.MIXED_VIEWS
    return TwoViewTerminal.NOT_CONFIRMED


__all__ = [
    'ClusterOutcome',
    'ClusteredProbabilityInterval',
    'CutoffMembership',
    'EventInterval',
    'GeometryEventKind',
    'GeometryObservationStatus',
    'GeometryState',
    'GeometryTick',
    'HistoryEventSet',
    'HistoryViewTrace',
    'IntervalDisposition',
    'ScientificView',
    'TransitionCell',
    'TransitionReport',
    'TwoViewTerminal',
    'ViewDecision',
    'ViewQualitativeDecision',
    'adjudicate_two_views',
    'clustered_probability_interval',
    'cutoff_membership',
    'derive_history_events',
    'physical_history_support',
    'transition_report',
]
