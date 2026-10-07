"Fixed empirical reductions; nominations and calibration are never local law results."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.discovery import FrontierDevelopment, round_down, round_up
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import ARMS, DEVELOPMENT_ROOTS, GAMMAS, JOINT, LADDER, LOCAL_BASE, LOCAL_EXPANDED, PAIR_REQUESTS, Coordinate, Pulse
from .measurement import ClassicalMeasuredRoot, ClassicalObservation, epsilon, receiver_ids


@dataclass(frozen=True, slots=True)
class EmpiricalRange(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/empirical-range'
    receiver: str
    lower: D
    upper: D
    mean: D

    def __post_init__(self) -> None:
        if (
            self.receiver not in ("c", "c1", "c2", "g")
            or any(
                type(x) is not D or not x.is_finite() for x in (self.lower, self.upper, self.mean)
            )
            or self.lower > self.upper
        ):
            raise ValueError("empirical range loses its native finite response")


@dataclass(frozen=True, slots=True)
class ClassicalBound(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-bound'
    coordinate: Coordinate
    valid_roots: tuple[str, ...]
    unsafe_roots: tuple[str, ...]
    responses: tuple[EmpiricalRange, ...]
    thermal_K: D | None
    suffix_thermal_K: D | None
    mechanistic_error_K: D | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not set(self.valid_roots) <= set(DEVELOPMENT_ROOTS)
            or len(set(self.valid_roots)) != len(self.valid_roots)
            or not set(self.unsafe_roots) <= set(DEVELOPMENT_ROOTS)
            or (
                self.responses
                and tuple(r.receiver for r in self.responses)
                != tuple(k for k in receiver_ids(self.coordinate) if not k.startswith("s"))
            )
            or any(
                x is not None and (type(x) is not D or not x.is_finite() or x < 0)
                for x in (self.thermal_K, self.suffix_thermal_K, self.mechanistic_error_K)
            )
            or (
                not self.reasons
                and (
                    len(self.valid_roots) < 29
                    or self.unsafe_roots
                    or not self.responses
                    or self.thermal_K is None
                    or (self.coordinate.kind == "joint" and self.suffix_thermal_K is None)
                )
            )
        ):
            raise ValueError("nomination changes its whole-root empirical operands")

    @property
    def nominated(self) -> bool:
        return not self.reasons

    def response(self, receiver: str) -> EmpiricalRange:
        return next(r for r in self.responses if r.receiver == receiver)


def retained_bounds(
    old: FrontierDevelopment, *, first_only: bool = False
) -> tuple[ClassicalBound, ...]:
    result = []
    for c in (JOINT[0],) if first_only else LOCAL_BASE:
        b = next(
            b
            for b in old.bounds
            if (
                b.coordinate.context,
                b.coordinate.horizon_s,
                b.coordinate.pulse.rate_kg_s,
                b.coordinate.pulse.duration_s,
            )
            == (c.context, c.horizon_s, c.pulse.rate_kg_s, c.pulse.duration_s)
        )
        reasons = tuple(
            r
            for failed, r in (
                (len(b.valid_roots) < 29, "INSUFFICIENT_DEVELOPMENT_CONTACT"),
                (bool(b.unsafe_roots), "OBSERVED_NATIVE_UNSAFE"),
                (b.thermal_allowance_K is None, "UNAVAILABLE_ACTION_THERMAL_ENVELOPE"),
            )
            if failed
        )
        ranges = (
            ()
            if b.lower_K is None or b.upper_K is None or b.mean_K is None
            else (EmpiricalRange("c1" if first_only else "c", b.lower_K, b.upper_K, b.mean_K),)
        )
        result.append(
            ClassicalBound(
                c,
                b.valid_roots,
                b.unsafe_roots,
                ranges,
                b.thermal_allowance_K,
                None,
                b.mechanistic_cooling_error_K,
                reasons,
            )
        )
    return tuple(result)


def discover_bound(
    c: Coordinate,
    measured: tuple[ClassicalMeasuredRoot, ...],
    *,
    first: ClassicalBound | None = None,
) -> ClassicalBound:
    if tuple(m.root for m in measured) != DEVELOPMENT_ROOTS:
        raise ValueError("discovery cannot omit, resample or replace a development root")
    rows = tuple((m.root, next(o for o in m.observations if o.coordinate == c)) for m in measured)
    good = tuple((root, row) for root, row in rows if row.valid)
    unsafe = tuple(root for root, row in rows if row.unsafe)
    ranges: list[EmpiricalRange] = []
    for key in receiver_ids(c):
        if key.startswith("s"):
            continue
        if key == "c1":
            if first is None or first.coordinate != JOINT[0]:
                raise ValueError("every IV row must share the unreselected retained first bound")
            ranges.extend(first.responses)
        elif good:
            raw = tuple(v for _, row in good for v in row.pair(key))
            low, high = min(raw), max(raw)
            ranges.append(
                EmpiricalRange(
                    key,
                    round_down(low - D(".1") * abs(low) - epsilon(c, key)),
                    round_up(high + D(".1") * abs(high) + epsilon(c, key), ".00001"),
                    sum((sum(row.pair(key), D(0)) / 2 for _, row in good), D(0)) / len(good),
                )
            )

    def allowance(key: str) -> D | None:
        values = tuple(
            v - t
            for _, row in good
            for t in ((row.second_temperature_K if key == "s2" else row.observed_temperature_K),)
            if t is not None
            for v in row.pair(key)
        )
        if not values:
            return None
        z = max(values)
        return round_up(max(D(1), z + D(".1") * abs(z) + D(".1")))

    thermal = allowance("s")
    suffix = allowance("s2") if c.kind == "joint" else None
    reasons = tuple(
        r
        for fail, r in (
            (len(good) < 29, "INSUFFICIENT_DEVELOPMENT_CONTACT"),
            (bool(unsafe), "OBSERVED_NATIVE_UNSAFE"),
            (
                thermal is None or (c.kind == "joint" and suffix is None),
                "UNAVAILABLE_THERMAL_ENVELOPE",
            ),
            (
                c.kind != "local" and (first is None or not first.nominated),
                "FIRST_RELATION_NONENTRY",
            ),
        )
        if fail
    )
    return ClassicalBound(
        c, tuple(root for root, _ in good), unsafe, tuple(ranges), thermal, suffix, None, reasons
    )


def positive_scale(value: D, gamma: D) -> D:
    if gamma not in GAMMAS:
        raise ValueError("undeclared calibration factor")
    return round_down(value * gamma) if value > 0 else value


def ladder(value: D) -> D | None:
    return next((q for q in reversed(LADDER) if q <= value), None)


@dataclass(frozen=True, slots=True)
class ClassicalRecipe(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-recipe'
    arm: str
    bound: ClassicalBound
    gamma: D | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.arm not in (*ARMS, "EL_SERVICE_MENU", "EL_SEQUENCE_RELATION")
            or (self.gamma is not None and self.gamma not in GAMMAS)
            or (self.gamma is None and not self.reasons)
        ):
            raise ValueError("recipe changes its fixed treatment or calibration disposition")
        if self.arm in ("EL_SERVICE_MENU", "EL_SEQUENCE_RELATION") and self.gamma != 1:
            raise ValueError("independent menu/sequence recipe changes its fixed gamma")

    @property
    def recipe_id(self) -> str:
        return f"{self.arm.lower()}.{self.bound.coordinate.coordinate_id}"

    @property
    def entered(self) -> bool:
        return self.gamma is not None and not self.reasons and self.bound.nominated

    def lower(self, receiver: str, *, predicted_C: D | None = None) -> D | None:
        if not self.entered:
            return None
        assert self.gamma is not None
        raw = self.bound.response(receiver)
        value = raw.lower
        if self.arm == "MECH_SAFE":
            if predicted_C is None or self.bound.mechanistic_error_K is None:
                return None
            value = round_down(
                predicted_C
                - D("1.1") * self.bound.mechanistic_error_K
                - self.bound.coordinate.epsilon_K
            )
        elif self.arm == "DIRECT_SAFE":
            value = raw.mean
        value = positive_scale(value, self.gamma)
        return (
            ladder(value)
            if self.arm in ("EL_SERVICE", "EL_SERVICE_MENU", "EL_SEQUENCE_RELATION")
            and receiver != "g"
            else value
        )


def recipe(arm: str, bound: ClassicalBound, gamma: D | None) -> ClassicalRecipe:
    return ClassicalRecipe(
        arm, bound, gamma, (*bound.reasons, *(("CALIBRATION_NONENTRY",) if gamma is None else ()))
    )


def physical_service(
    row: ClassicalObservation, required: D, budget: D, *, second: D | None = None
) -> bool:
    if not row.valid or row.unsafe or max(row.applied_masses_kg) > budget + D("1e-10"):
        return False
    first = "c" if row.coordinate.kind == "local" else "c1"
    return min(row.pair(first)) >= required and (
        second is None
        or (
            row.coordinate.kind == "joint"
            and min(row.pair("c2")) >= second
            and min(row.pair("g")) >= 0
        )
    )


def fixed_second_words(
    measured: tuple[ClassicalMeasuredRoot, ...],
) -> tuple[tuple[str, Pulse], ...]:
    if tuple(m.root for m in measured) != DEVELOPMENT_ROOTS:
        raise ValueError("fixed-policy tuning requires the entire development census")
    result = []
    for request in PAIR_REQUESTS:
        scores = []
        for c in JOINT[2:]:
            rows = tuple(next(o for o in m.observations if o.coordinate == c) for m in measured)
            success = sum(
                physical_service(o, request.required_K, request.budget_kg, second=request.second_K)
                for o in rows
            )
            masses = tuple(
                o.applied_masses_kg[0] for o in rows if o.contact and o.applied_masses_kg
            )
            mean = sum(masses, D(0)) / len(masses) if masses else D("Infinity")
            scores.append((-success, mean, c.pulse.rate_kg_s, c.pulse.word_id, c.pulse))
        result.append((request.request_id, min(scores, key=lambda x: x[:-1])[-1]))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ClassicalNomination(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-nomination'
    block: str
    parents: tuple[ObjectIdentity, ...]
    recipes: tuple[ClassicalRecipe, ...]
    fixed_seconds: tuple[tuple[str, Pulse], ...]
    entry_opportunities: tuple[tuple[str, str, str], ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = (
            tuple((arm, c) for arm in ARMS for c in LOCAL_BASE)
            if self.block == "base-menu-comparison"
            else tuple(("EL_SERVICE_MENU", c) for c in LOCAL_EXPANDED)
            if self.block == "expanded-menu-comparison"
            else tuple(("EL_SEQUENCE_RELATION", c) for c in JOINT)
            if self.block == "staged-sequence-comparison"
            else ()
        )
        if (
            not expected
            or tuple((r.arm, r.bound.coordinate) for r in self.recipes) != expected
            or not self.parents
            or (self.block == "staged-sequence-comparison" and tuple(k for k, _ in self.fixed_seconds) != ("LOW", "HIGH"))
            or (self.block != "staged-sequence-comparison" and self.fixed_seconds)
            or (not self.entry_opportunities and not self.reasons)
        ):
            raise ValueError("nomination lost its closed recipe census, lineage or entry screen")
        if self.block == "staged-sequence-comparison" and any(
            r.bound.responses and r.bound.response("c1") != self.recipes[0].bound.response("c1")
            for r in self.recipes
        ):
            raise ValueError("sequence suffix evidence refits the unconditional first bound")

    @property
    def record_id(self) -> str:
        return f"reactor-staged-pulse-response.{self.block}.nomination"

    @property
    def entered(self) -> bool:
        return bool(self.entry_opportunities) and not self.reasons
