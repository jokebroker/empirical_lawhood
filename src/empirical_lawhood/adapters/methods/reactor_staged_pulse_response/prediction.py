"""Causal projections and frozen receiver maps, before local response visibility."""

from dataclasses import dataclass
from decimal import Decimal as D
from functools import lru_cache
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.causal import causal_operands
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.measurement import FrontierMeasuredRoot
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.selection import FrontierPredictionSeal
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.words import project_pulse
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.adapters.simulators.reactor_prefix_response.causal_feed_forecast import causal_feed_forecast
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import BASE, CONTEXTS, EXPANDED, FIRST, LOCAL_BASE, LOCAL_EXPANDED, SECONDS, ZERO, Coordinate, Pulse, Request, assignment
from .discovery import ClassicalNomination, ClassicalRecipe
from .measurement import ClassicalObservation, decimal
from .records import ClassicalContext, ClassicalPreparation

HORIZON = "reactor-staged-pulse-response-local-guard"


@dataclass(frozen=True, slots=True)
class ClassicalCausalPoint(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-causal-point'
    root: str
    context: str
    causal: ObjectIdentity
    temperature_K: D | None
    callback: int | None
    masses: tuple[tuple[Pulse, D, D], ...]
    forecasts: tuple[tuple[Coordinate, D], ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        assignment(self.root)
        historical = assignment(self.root)[1] == "calibration"
        if (
            self.context not in (*CONTEXTS, "induced")
            or len({w for w, _, _ in self.masses}) != len(self.masses)
            or len({c for c, _ in self.forecasts}) != len(self.forecasts)
            or (
                self.callback is None
                and not historical
                and (self.temperature_K is not None or self.masses)
            )
            or (self.temperature_K is None and not self.reasons)
        ):
            raise ValueError("prediction changed its causal prefix or native projection census")
        if any(not x.is_finite() for _, *values in self.masses for x in values) or any(
            not value.is_finite() for _, value in self.forecasts
        ):
            raise ValueError("nonfinite causal prediction")


@lru_cache(maxsize=16)
def causal_point(
    source: ReactorBatchSource,
    context: ClassicalContext,
    domain: FeedDomain,
    *,
    block: str,
    mechanistic: bool,
) -> ClassicalCausalPoint:
    # Several laws/requests share this exact outcome-blind prefix. Reuse its
    # deterministic projection, keyed by the complete source, history, domain
    # and chart; a new root or induced predecessor cannot inherit the result.
    words = (
        SECONDS
        if context.context == "induced"
        else (FIRST,)
        if block == "staged-sequence-comparison"
        else BASE
        if block == "base-menu-comparison"
        else EXPANDED
    )
    inputs = causal_operands(context, domain)
    parent = ObjectIdentity.from_record(context.record_id, context)
    if inputs is None:
        return ClassicalCausalPoint(
            context.root, context.context, parent, None, None, (), (), ("NO_VALID_CAUSAL_CONTACT",)
        )
    masses = []
    for word in words:
        mapping = project_pulse(
            inputs.observation,
            inputs.previous,
            word,
            decision_id=f"{context.root}.{context.context}.{word.word_id}.projection",
            history_id=context.record_id,
            horizon_id=HORIZON,
        )
        masses.append((word, mapping.applied_mass_kg, mapping.realized_mass_kg))
    forecasts: tuple[tuple[Coordinate, D], ...] = ()
    reasons: tuple[str, ...] = ()
    if mechanistic:
        grids, reasons = causal_feed_forecast(
            source, context.root, context, tuple((w.word_id, w.rates()) for w in (ZERO, *words))
        )
        if not reasons:
            forecasts = tuple(
                (
                    c,
                    decimal(
                        grids[ZERO.word_id][: c.horizon_s + 1, 1].max()
                        - grids[c.pulse.word_id][: c.horizon_s + 1, 1].max()
                    ),
                )
                for c in (LOCAL_BASE if block == "base-menu-comparison" else LOCAL_EXPANDED)
                if c.context == context.context
            )
    return ClassicalCausalPoint(
        context.root,
        context.context,
        parent,
        decimal(inputs.observation.temperature),
        context.callback,
        tuple(masses),
        forecasts,
        reasons,
    )


def retained_points(
    seal: FrontierPredictionSeal, measured: FrontierMeasuredRoot
) -> tuple[ClassicalCausalPoint, ...]:
    if seal.root != measured.root or seal.preparation != measured.preparation:
        raise ValueError("historical prediction and measurement replace their common causal parent")
    result = []
    for context in CONTEXTS:
        observed = {
            o.observed_temperature_K
            for o in measured.observations
            if o.coordinate.context == context
        }
        if len(observed) != 1:
            raise ValueError("historical causal temperature is inconsistent within one cutoff")
        forecast = (
            next(f for f in seal.forecasts if f.values and f.values[0][0].context == context)
            if any(f.values and f.values[0][0].context == context for f in seal.forecasts)
            else None
        )
        source = forecast.context if forecast is not None else seal.preparation
        values = tuple(
            (c, value)
            for c in LOCAL_BASE
            if c.context == context
            for old_c, value, _, _ in (() if forecast is None else forecast.values)
            if (old_c.context, old_c.horizon_s, old_c.pulse.rate_kg_s, old_c.pulse.duration_s)
            == (c.context, c.horizon_s, c.pulse.rate_kg_s, c.pulse.duration_s)
        )
        masses = tuple(
            (w, applied, realized)
            for w in BASE
            for ctx, old_w, applied, realized in seal.masses
            if ctx == context and (w.rate_kg_s, w.duration_s) == (old_w.rate_kg_s, old_w.duration_s)
        )
        temperature = next(iter(observed))
        # This compatibility point is only historical calibration. Its clock is
        # not available in the preceding seal and must never instantiate a current owner.
        result.append(
            ClassicalCausalPoint(
                seal.root,
                context,
                source,
                temperature,
                None,
                masses,
                values,
                () if temperature is not None else ("NO_CAUSAL_CONTACT",),
            )
        )
    return tuple(result)


@dataclass(frozen=True, slots=True)
class ClassicalPredictedBound(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-predicted-bound'
    recipe_id: str
    coordinate: Coordinate
    causal: ObjectIdentity
    lower: tuple[NamedDecimal, ...]
    upper: tuple[NamedDecimal, ...]
    saturated_receivers: tuple[str, ...]
    projected_mass_kg: D | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            tuple(v.value_id for v in self.lower) != tuple(v.value_id for v in self.upper)
            or any(
                lo.unit != "K" or hi.unit != "K" or lo.value > hi.value
                for lo, hi in zip(self.lower, self.upper, strict=True)
            )
            or not set(self.saturated_receivers) <= {v.value_id for v in self.lower}
            or (not self.lower and not self.reasons)
        ):
            raise ValueError("predicted receiver map changes its finite bounds or native unit")
        for lo, hi in zip(self.lower, self.upper, strict=True):
            if lo.value_id in self.saturated_receivers and lo.value != hi.value:
                raise ValueError("saturated lower-bound receiver must use its exact point carrier")

    @property
    def available(self) -> bool:
        return not self.reasons

    def interval(self, receiver: str) -> tuple[D, D]:
        return (
            next(v.value for v in self.lower if v.value_id == receiver),
            next(v.value for v in self.upper if v.value_id == receiver),
        )


def predict(
    recipe: ClassicalRecipe,
    point: ClassicalCausalPoint,
    *,
    first_point: ClassicalCausalPoint | None = None,
    first_mass: D = D(0),
) -> ClassicalPredictedBound:
    c, b = recipe.bound.coordinate, recipe.bound
    reasons = list(recipe.reasons)
    expected_context = "induced" if c.kind == "joint" else c.context
    if point.context != expected_context or (
        first_point is not None and first_point.root != point.root
    ):
        raise ValueError("recipe was predicted on another causal history")
    masses = {w: mass for w, mass, _ in point.masses}
    mass = masses.get(c.pulse)
    if point.temperature_K is None or mass is None:
        reasons.append("NO_CAUSAL_CONTACT_OR_PROJECTION")
    low, high, mapped = [], [], []
    mech = dict(point.forecasts).get(c)
    for empirical in b.responses:
        q = recipe.lower(empirical.receiver, predicted_C=mech)
        if q is None:
            reasons.append("NO_DECLARED_POSITIVE_BOUND_OR_FORECAST")
            continue
        interval = recipe.arm == "EL_INTERVAL"
        low.append(NamedDecimal(empirical.receiver, q, "K"))
        high.append(NamedDecimal(empirical.receiver, empirical.upper if interval else q, "K"))
        if not interval:
            mapped.append(empirical.receiver)
    initial = point if c.kind != "joint" else first_point
    if initial is not None and initial.temperature_K is not None and b.thermal_K is not None:
        low.append(NamedDecimal("s", D(0), "K"))
        high.append(NamedDecimal("s", initial.temperature_K + b.thermal_K, "K"))
    else:
        reasons.append("MISSING_FIRST_THERMAL_PREDICTION")
    if c.kind == "joint":
        if point.temperature_K is not None and b.suffix_thermal_K is not None:
            low.append(NamedDecimal("s2", D(0), "K"))
            high.append(NamedDecimal("s2", point.temperature_K + b.suffix_thermal_K, "K"))
        else:
            reasons.append("MISSING_SUFFIX_THERMAL_PREDICTION")
        if mass is not None:
            mass += first_mass
    return ClassicalPredictedBound(
        recipe.recipe_id,
        c,
        point.causal,
        tuple(low),
        tuple(high),
        tuple(mapped),
        mass,
        tuple(sorted(set(reasons))),
    )


def predicted_event(
    prediction: ClassicalPredictedBound, observation: ClassicalObservation
) -> tuple[bool, tuple[str, ...]]:
    if prediction.coordinate != observation.coordinate:
        raise ValueError("law event changes its declared action/history/receiver")
    reasons = set(prediction.reasons)
    if not observation.valid:
        reasons.add("INVALID_CONTACT_DELIVERY_OR_RAW_NUMERICS")
    if observation.unsafe:
        reasons.add("NATIVE_UNSAFE")
    if observation.values:
        for lo, hi in zip(prediction.lower, prediction.upper, strict=True):
            for raw in observation.pair(lo.value_id):
                mapped = (
                    min(raw, lo.value) if lo.value_id in prediction.saturated_receivers else raw
                )
                if not lo.value <= mapped <= hi.value:
                    reasons.add(f"{lo.value_id}_DECLARED_BOUND_MISS")
                if lo.value_id in prediction.saturated_receivers and raw < lo.value:
                    reasons.add(f"{lo.value_id}_RAW_LOWER_BOUND_MISS")
        if prediction.projected_mass_kg is None or any(
            abs(m - prediction.projected_mass_kg) > D("1e-12")
            for m in observation.applied_masses_kg
        ):
            reasons.add("PROJECTED_ACTUAL_MASS_MISMATCH")
    return not reasons, tuple(sorted(reasons))


def eligible(
    prediction: ClassicalPredictedBound, request: Request, *, first_gate_only: bool = False
) -> bool:
    if (
        not prediction.available
        or prediction.projected_mass_kg is None
        or prediction.projected_mass_kg > request.budget_kg + D("1e-10")
    ):
        return False
    c = prediction.coordinate
    if c.kind == "local" and (c.context, c.horizon_s) != (request.context, request.horizon_s):
        return False
    key = "c" if c.kind == "local" else "c1"
    if prediction.interval(key)[0] < request.required_K or prediction.interval("s")[1] > D("356.2"):
        return False
    if c.kind == "joint":
        return (
            request.second_K is not None
            and prediction.interval("c2")[0] >= request.second_K
            and prediction.interval("g")[0] >= 0
            and (first_gate_only or prediction.interval("s2")[1] <= D("356.2"))
        )
    return True


def choose(
    predictions: tuple[ClassicalPredictedBound, ...],
    request: Request,
    *,
    allowed: tuple[str, ...] | None = None,
) -> ClassicalPredictedBound | None:
    candidates = [
        p
        for p in predictions
        if (allowed is None or p.recipe_id in allowed) and eligible(p, request)
    ]
    return (
        min(
            candidates,
            key=lambda p: (
                p.projected_mass_kg,
                p.coordinate.pulse.duration_s,
                p.coordinate.pulse.rate_kg_s,
                p.coordinate.pulse.word_id,
            ),
        )
        if candidates
        else None
    )


@dataclass(frozen=True, slots=True)
class ClassicalPredictionSeal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-prediction-seal'
    root: str
    preparation: ObjectIdentity
    nomination: ObjectIdentity | None
    points: tuple[ClassicalCausalPoint, ...]
    predictions: tuple[ClassicalPredictedBound, ...]

    def __post_init__(self) -> None:
        assignment(self.root)
        if (
            any(p.root != self.root for p in self.points)
            or len({p.context for p in self.points}) != len(self.points)
            or len({p.recipe_id for p in self.predictions}) != len(self.predictions)
            or (self.nomination is None and self.predictions)
        ):
            raise ValueError("seal mixes roots, recipes or causal cutoffs")

    @property
    def record_id(self) -> str:
        return f"{self.root}.prediction-seal"


def seal_predictions(
    source: ReactorBatchSource,
    preparation: ClassicalPreparation,
    domain: FeedDomain,
    nomination: ClassicalNomination | None,
) -> ClassicalPredictionSeal:
    points = tuple(
        causal_point(
            source, context, domain, block=preparation.block, mechanistic=preparation.block == "base-menu-comparison"
        )
        for context in preparation.contexts
    )
    predictions = (
        ()
        if nomination is None
        else tuple(
            predict(r, next(p for p in points if p.context == r.bound.coordinate.context))
            for r in nomination.recipes
            if r.bound.coordinate.kind != "joint"
        )
    )
    return ClassicalPredictionSeal(
        preparation.root,
        ObjectIdentity.from_record(preparation.record_id, preparation),
        None
        if nomination is None
        else ObjectIdentity.from_record(nomination.record_id, nomination),
        points,
        predictions,
    )


@dataclass(frozen=True, slots=True)
class ClassicalInducedPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-induced-prediction'
    first_seal: ObjectIdentity
    context: ClassicalContext
    point: ClassicalCausalPoint
    predictions: tuple[ClassicalPredictedBound, ...]

    def __post_init__(self) -> None:
        if (
            self.first_seal.object_schema != ClassicalPredictionSeal.SCHEMA
            or self.context.context != "induced"
            or self.point.causal != ObjectIdentity.from_record(self.context.record_id, self.context)
            or any(
                p.coordinate.kind != "joint" or p.causal != self.point.causal
                for p in self.predictions
            )
        ):
            raise ValueError("conditional prediction substitutes its actual causal cutoff")

    @property
    def record_id(self) -> str:
        return f"{self.context.root}.induced-prediction-seal"


def induced_prediction(
    source: ReactorBatchSource,
    context: ClassicalContext,
    domain: FeedDomain,
    first: ClassicalPredictionSeal,
    nomination: ClassicalNomination | None,
) -> ClassicalInducedPrediction:
    if context.root != first.root or (
        nomination is not None
        and first.nomination != ObjectIdentity.from_record(nomination.record_id, nomination)
    ):
        raise ValueError("second cutoff changed the pre-t0 recipe")
    point = causal_point(source, context, domain, block="staged-sequence-comparison", mechanistic=False)
    original = next(p for p in first.points if p.context == "early")
    masses = {w: mass for w, mass, _ in original.masses}
    predictions = (
        ()
        if nomination is None
        else tuple(
            predict(r, point, first_point=original, first_mass=masses.get(FIRST, D(0)))
            for r in nomination.recipes
            if r.bound.coordinate.kind == "joint"
        )
    )
    return ClassicalInducedPrediction(
        ObjectIdentity.from_record(first.record_id, first), context, point, predictions
    )
