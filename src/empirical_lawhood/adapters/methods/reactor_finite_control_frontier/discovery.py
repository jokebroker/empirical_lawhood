"Fixed whole-root development reduction; these are nominations, not local law."

from dataclasses import dataclass
from decimal import Decimal as D, ROUND_CEILING, ROUND_FLOOR
from typing import ClassVar, TYPE_CHECKING

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import BUDGETS, COORDINATES, CONTEXTS, HORIZONS, REQUESTS, ROOTS, Coordinate, Pulse, ZERO
from .measurement import FrontierGuard, FrontierMeasuredRoot, FrontierObservation
from .mechanistic import FrontierForecast

if TYPE_CHECKING:
    from .selection import FrontierPredictionSeal


DEVELOPMENT_ROOTS = tuple(r for r, role, _, _ in ROOTS if role == "development")


def round_down(value: D, quantum: str = ".00001") -> D:
    return value.quantize(D(quantum), rounding=ROUND_FLOOR)


def round_up(value: D, quantum: str = ".01") -> D:
    return value.quantize(D(quantum), rounding=ROUND_CEILING)


def valid(row: FrontierObservation) -> bool:
    return row.contact and row.evaluable and row.delivery_valid and row.numerical_valid


def safe_service(row: FrontierObservation, request: D, budget: D) -> bool:
    return (
        valid(row)
        and row.prefix_safe
        and row.action_safe
        and min(row.effects_K) >= request
        and max(row.applied_masses_kg) <= budget + D("1e-10")
    )


@dataclass(frozen=True, slots=True)
class FrontierBound(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-bound'
    coordinate: Coordinate
    valid_roots: tuple[str, ...]
    unsafe_roots: tuple[str, ...]
    lower_K: D | None
    upper_K: D | None
    mean_K: D | None
    minimum_K: D | None
    maximum_K: D | None
    thermal_allowance_K: D | None
    zero_thermal_allowance_K: D | None
    mechanistic_roots: tuple[str, ...]
    mechanistic_cooling_error_K: D | None
    mechanistic_thermal_error_K: D | None
    mechanistic_zero_thermal_error_K: D | None
    nominated: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not set(self.valid_roots) <= set(DEVELOPMENT_ROOTS)
            or len(set(self.valid_roots)) != len(self.valid_roots)
            or any(
                v is not None and (type(v) is not D or not v.is_finite())
                for v in (
                    self.lower_K,
                    self.upper_K,
                    self.mean_K,
                    self.minimum_K,
                    self.maximum_K,
                    self.thermal_allowance_K,
                    self.zero_thermal_allowance_K,
                    self.mechanistic_cooling_error_K,
                    self.mechanistic_thermal_error_K,
                    self.mechanistic_zero_thermal_error_K,
                )
            )
            or self.nominated
            != (
                len(self.valid_roots) >= 29
                and not self.unsafe_roots
                and self.thermal_allowance_K is not None
                and self.zero_thermal_allowance_K is not None
            )
            or (self.nominated and self.reasons)
            or (not self.nominated and not self.reasons)
        ):
            raise ValueError("development row changes its fixed nomination predicate")


@dataclass(frozen=True, slots=True)
class FrontierDevelopment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-development'
    record_id: str
    measurements: tuple[ObjectIdentity, ...]
    forecasts: tuple[ObjectIdentity, ...]
    prediction_seals: tuple[ObjectIdentity, ...]
    bounds: tuple[FrontierBound, ...]
    fixed_tuned: tuple[tuple[str, int, Pulse], ...]
    fixed_tuned_scores: tuple[tuple[Coordinate, int, D], ...]

    def __post_init__(self) -> None:
        if (
            len(self.measurements) != 32
            or len(self.forecasts) != 128
            or len(self.prediction_seals) != 32
            or tuple(r.coordinate for r in self.bounds) != COORDINATES
            or tuple((c, t) for c, t, _ in self.fixed_tuned)
            != tuple((c, t) for c in CONTEXTS for t in HORIZONS)
            or tuple(c for c, _, _ in self.fixed_tuned_scores) != COORDINATES
        ):
            raise ValueError("development changed the fixed whole-root or finite-chart denominator")


def discover(
    measurements: tuple[FrontierMeasuredRoot, ...],
    forecasts: tuple[FrontierForecast, ...],
    seals: tuple['FrontierPredictionSeal', ...],
) -> FrontierDevelopment:
    if tuple(r.root for r in measurements) != DEVELOPMENT_ROOTS or tuple(
        f.root for f in forecasts
    ) != tuple(r for r in DEVELOPMENT_ROOTS for _ in CONTEXTS):
        raise ValueError("discovery requires exactly the retained 32 development roots")
    if tuple(s.root for s in seals) != DEVELOPMENT_ROOTS or any(
        s.development is not None for s in seals
    ):
        raise ValueError("development requires the exact outcome-blind mass/forecast seals")
    projected = {(s.root, c, w): mass for s in seals for c, w, mass, _ in s.masses}
    guards = {(m.root, g.context, g.pulse): g for m in measurements for g in m.guards}
    predicted = {
        (f.root, c): (cooling, peak, zero) for f in forecasts for c, cooling, peak, zero in f.values
    }
    rows = {(root.root, row.coordinate): row for root in measurements for row in root.observations}
    bounds, scores = [], []
    for c in COORDINATES:
        assigned = tuple((r, rows[r, c]) for r in DEVELOPMENT_ROOTS)
        good = tuple((r, o) for r, o in assigned if valid(o))
        guard = tuple(
            guards[r, c.context, c.pulse]
            for r in DEVELOPMENT_ROOTS
            if guards[r, c.context, c.pulse].valid
        )
        reference_guard = tuple(
            guards[r, c.context, ZERO]
            for r in DEVELOPMENT_ROOTS
            if guards[r, c.context, ZERO].valid
        )
        unsafe = tuple(
            r
            for r, o in assigned
            if guards[r, c.context, c.pulse].unsafe
            or (o.contact and o.evaluable and not o.prefix_safe)
        )
        low = high = mean = minimum = maximum = None
        if good:
            values = tuple(v for _, row in good for v in row.effects_K)
            minimum, maximum = min(values), max(values)
            low = round_down(minimum - D(".1") * abs(minimum) - c.epsilon_K)
            high = round_up(maximum + D(".1") * abs(maximum) + c.epsilon_K, ".00001")
            mean = sum((sum(o.effects_K, D(0)) / 2 for _, o in good), D(0)) / len(good)

        def allowance(branches: tuple[FrontierGuard, ...]) -> D | None:
            if not branches:
                return None
            z = max(
                v - g.observed_temperature_K
                for g in branches
                for v in g.peaks_K
                if g.observed_temperature_K is not None
            )
            return round_up(max(D(1), z + D(".1") * abs(z) + D(".1")))

        q, q0 = allowance(guard), allowance(reference_guard)
        mech = tuple((r, o) for r, o in good if (r, c) in predicted)
        a = v = v0 = None
        if len(mech) >= 29:
            a = max(D(0), *(predicted[r, c][0] - actual for r, o in mech for actual in o.effects_K))
            v = max(
                D(0), *(actual - predicted[r, c][1] for r, o in mech for actual in o.action_peaks_K)
            )
            v0 = max(
                D(0),
                *(actual - predicted[r, c][2] for r, o in mech for actual in o.reference_peaks_K),
            )
        reasons = tuple(
            reason
            for fail, reason in (
                (len(good) < 29, "INSUFFICIENT_DEVELOPMENT_CONTACT"),
                (bool(unsafe), "OBSERVED_NATIVE_UNSAFE"),
                (q is None or q0 is None, "UNAVAILABLE_THERMAL_ENVELOPE"),
            )
            if fail
        )
        bound = FrontierBound(
            c,
            tuple(r for r, _ in good),
            unsafe,
            low,
            high,
            mean,
            minimum,
            maximum,
            q,
            q0,
            tuple(r for r, _ in mech),
            a,
            v,
            v0,
            not reasons,
            reasons,
        )
        bounds.append(bound)
        successes = 0
        for _, o in assigned:
            for b in REQUESTS:
                for budget in BUDGETS:
                    admitted = (
                        valid(o)
                        and mean is not None
                        and mean >= b
                        and q is not None
                        and o.observed_temperature_K is not None
                        and o.observed_temperature_K + q <= D("356.2")
                        and o.applied_masses_kg[0] <= budget + D("1e-10")
                    )
                    successes += int(admitted and safe_service(o, b, budget))
        # Tie breaking uses all causally projected contact masses, never an outcome-selected subset.
        mass_values = tuple(
            projected[r, c.context, c.pulse]
            for r in DEVELOPMENT_ROOTS
            if (r, c.context, c.pulse) in projected
        )
        average_mass = sum(mass_values, D(0)) / len(mass_values) if mass_values else D("2.10")
        scores.append((c, successes, average_mass))
    tuned = tuple(
        (
            context,
            horizon,
            min(
                (
                    row
                    for row in scores
                    if row[0].context == context and row[0].horizon_s == horizon
                ),
                key=lambda row: (
                    -row[1],
                    row[2],
                    row[0].pulse.duration_s,
                    row[0].pulse.rate_kg_s,
                    row[0].pulse.word_id,
                ),
            )[0].pulse,
        )
        for context in CONTEXTS
        for horizon in HORIZONS
    )
    return FrontierDevelopment(
        "reactor-finite-control-frontier.development",
        tuple(ObjectIdentity.from_record(f"{m.root}.measurement", m) for m in measurements),
        tuple(
            ObjectIdentity.from_record(f"{f.root}.forecast-{i % 4}", f)
            for i, f in enumerate(forecasts)
        ),
        tuple(ObjectIdentity.from_record(s.record_id, s) for s in seals),
        tuple(bounds),
        tuned,
        tuple(scores),
    )
