"""Causal finite-chart selections sealed before any local branch measurement."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.words import project_pulse
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchSource
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .causal import causal_operands
from .config import BUDGETS, CONTEXTS, HORIZONS, POLICIES, PULSES, REQUESTS, Pulse, assignment
from .discovery import FrontierBound, FrontierDevelopment, round_down, round_up
from .mechanistic import FrontierForecast, forecast_context
from .records import FrontierPreparation

HORIZON_ID = "reactor-frontier-delivery-guard-120s"


def pulse_tie_key(pulse: Pulse) -> str:
    "The declared duration/rate/word tie order shared by sealing and admission ranking."
    return f"d{pulse.duration_s:03d}.f{int(pulse.rate_kg_s * 1000):03d}.{pulse.word_id}"


def worst_mass(pulse: Pulse) -> D:
    """Exact maximum over previous feed [0,.016] and allowed observed dose.

    First applied feed min(f, previous+.020) is monotone; subsequent positive
    commands reach f. Shutdown costs max(f-.020,0)*10. Dose capping can only
    reduce subsequent applied stages. Thus the upper endpoint is attained at
    previous=.016 with sufficient remaining dose, including the .032 ramp.
    """
    return (
        10 * min(pulse.rate_kg_s, D(".036"))
        + (pulse.duration_s - 10) * pulse.rate_kg_s
        + 10 * max(D(0), pulse.rate_kg_s - D(".020"))
    )


@dataclass(frozen=True, slots=True)
class FrontierUseRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-use-request'
    context: str
    horizon_s: int
    kind: str
    request_K: D
    budget_kg: D

    def __post_init__(self) -> None:
        if (
            self.context not in CONTEXTS
            or self.horizon_s not in HORIZONS
            or self.kind not in ("low", "high")
            or self.request_K <= 0
            or self.budget_kg not in BUDGETS
        ):
            raise ValueError("primary request changed its declared context, service or budget")

    @property
    def request_id(self) -> str:
        return f"{self.context}.t{self.horizon_s:03d}.{self.kind}"


def primary_requests(
    development: FrontierDevelopment, qualified: tuple[str, ...]
) -> tuple[FrontierUseRequest, ...]:
    result: list[FrontierUseRequest] = []
    for context in CONTEXTS:
        for horizon in HORIZONS:
            rows = tuple(
                b
                for b in development.bounds
                if b.coordinate.context == context
                and b.coordinate.horizon_s == horizon
                and b.coordinate.coordinate_id in qualified
                and b.lower_K is not None
                and b.lower_K >= REQUESTS[0]
                and worst_mass(b.coordinate.pulse) <= BUDGETS[-1]
            )
            if not rows:
                continue

            def order(b: FrontierBound) -> tuple[D, int, D, str]:
                w = b.coordinate.pulse
                return worst_mass(w), w.duration_s, w.rate_kg_s, w.word_id

            low = min(rows, key=order)
            high = min(rows, key=lambda b: (-b.lower_K, *order(b)))  # type: ignore[operator]
            for kind, b, row in (("low", REQUESTS[0], low), ("high", high.lower_K, high)):
                assert b is not None
                budget = next(B for B in BUDGETS if B >= worst_mass(row.coordinate.pulse))
                if kind == "high" and result[-1].request_K == b and result[-1].budget_kg == budget:
                    continue  # Explicit single-request case; never duplicate its success.
                result.append(FrontierUseRequest(context, horizon, kind, b, budget))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class FrontierPolicyGrid(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-policy-grid'
    context: str
    horizon_s: int
    policy: str
    choices: tuple[Pulse | None, ...]

    def __post_init__(self) -> None:
        if (
            self.context not in CONTEXTS
            or self.horizon_s not in HORIZONS
            or self.policy not in POLICIES
            or len(self.choices) != 84
            or any(
                w is not None and (w not in PULSES or w.duration_s > self.horizon_s)
                for w in self.choices
            )
        ):
            raise ValueError("policy grid changes the fixed request denominator")


@dataclass(frozen=True, slots=True)
class FrontierPredictionSeal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-prediction-seal'
    root: str
    preparation: ObjectIdentity
    development: ObjectIdentity | None
    parent_laws: ObjectIdentity | None
    forecasts: tuple[FrontierForecast, ...]
    masses: tuple[tuple[str, Pulse, D, D], ...]
    grids: tuple[FrontierPolicyGrid, ...]
    primary: tuple[tuple[FrontierUseRequest, Pulse | None], ...]

    def __post_init__(self) -> None:
        role, _ = assignment(self.root)
        if (
            self.preparation.object_schema != FrontierPreparation.SCHEMA
            or len(self.forecasts) != 4
            or any(f.root != self.root for f in self.forecasts)
            or (
                role == "development"
                and (self.development is not None or self.grids or self.primary)
            )
            or (role != "development" and (self.development is None or len(self.grids) != 60))
            or (role == "prospective" and self.parent_laws is None)
            or (role != "prospective" and (self.primary or self.parent_laws is not None))
        ):
            raise ValueError("prediction seal changed its causal role or qualification parent")

    @property
    def record_id(self) -> str:
        return f"{self.root}.frontier-prediction-seal"


def eligible_el_pulses(
    *,
    bounds: tuple[FrontierBound, ...],
    context: str,
    horizon: int,
    request: D,
    budget: D,
    observed_temperature: D | None,
    masses: dict[Pulse, D],
    qualified: tuple[str, ...],
) -> tuple[Pulse, ...]:
    if observed_temperature is None:
        return ()
    admitted = (
        b.coordinate.pulse
        for b in bounds
        if b.coordinate.context == context
        and b.coordinate.horizon_s == horizon
        and b.coordinate.coordinate_id in qualified
        and b.coordinate.pulse in masses
        and masses[b.coordinate.pulse] <= budget + D("1e-10")
        and b.lower_K is not None
        and b.lower_K >= request
        and b.thermal_allowance_K is not None
        and observed_temperature + b.thermal_allowance_K <= D("356.2")
    )
    return tuple(sorted(admitted, key=lambda w: (masses[w], pulse_tie_key(w))))


def eligible_pulses(
    *,
    development: FrontierDevelopment,
    context: str,
    horizon: int,
    policy: str,
    request: D,
    budget: D,
    observed_temperature: D | None,
    masses: dict[Pulse, D],
    forecast: FrontierForecast,
    qualified: tuple[str, ...],
) -> tuple[Pulse, ...]:
    if policy == "EL_BOUND":
        return eligible_el_pulses(
            bounds=development.bounds,
            context=context,
            horizon=horizon,
            request=request,
            budget=budget,
            observed_temperature=observed_temperature,
            masses=masses,
            qualified=qualified,
        )
    if observed_temperature is None:
        return ()
    predicted = {c: (effect, peak, zero) for c, effect, peak, zero in forecast.values}
    tuned = next(w for c, t, w in development.fixed_tuned if c == context and t == horizon)
    admitted = []
    for row in development.bounds:
        c, w = row.coordinate, row.coordinate.pulse
        if (
            c.context != context
            or c.horizon_s != horizon
            or w not in masses
            or masses[w] > budget + D("1e-10")
        ):
            continue
        lower, upper = (
            row.mean_K,
            None
            if row.thermal_allowance_K is None
            else observed_temperature + row.thermal_allowance_K,
        )
        if policy == "FIXED_SELECTED_ACTION" and w != Pulse(D(".016"), 10):
            continue
        elif policy == "FIXED_TUNED" and w != tuned:
            continue
        elif policy == "MECH_BOUND":
            if (
                c not in predicted
                or row.mechanistic_cooling_error_K is None
                or row.mechanistic_thermal_error_K is None
            ):
                continue
            lower = round_down(
                predicted[c][0] - D("1.1") * row.mechanistic_cooling_error_K - c.epsilon_K
            )
            upper = round_up(predicted[c][1] + D("1.1") * row.mechanistic_thermal_error_K + D(".1"))
        elif policy not in POLICIES:
            raise ValueError("undeclared policy")
        if lower is not None and upper is not None and lower >= request and upper <= D("356.2"):
            admitted.append(w)
    return tuple(sorted(admitted, key=lambda w: (masses[w], pulse_tie_key(w))))


def choose(
    *,
    development: FrontierDevelopment,
    context: str,
    horizon: int,
    policy: str,
    request: D,
    budget: D,
    observed_temperature: D | None,
    masses: dict[Pulse, D],
    forecast: FrontierForecast,
    qualified: tuple[str, ...],
) -> Pulse | None:
    candidates = eligible_pulses(
        development=development,
        context=context,
        horizon=horizon,
        policy=policy,
        request=request,
        budget=budget,
        observed_temperature=observed_temperature,
        masses=masses,
        forecast=forecast,
        qualified=qualified,
    )
    return next(iter(candidates), None)


def seal_predictions(
    *,
    source: ReactorBatchSource,
    preparation: FrontierPreparation,
    domain: FeedDomain,
    development: FrontierDevelopment | None,
    qualified: tuple[str, ...] = (),
    parent_laws: ObjectIdentity | None = None,
    requests: tuple[FrontierUseRequest, ...] = (),
) -> FrontierPredictionSeal:
    forecasts: list[FrontierForecast] = []
    amounts: list[tuple[str, Pulse, D, D]] = []
    grids: list[FrontierPolicyGrid] = []
    primary: list[tuple[FrontierUseRequest, Pulse | None]] = []
    for context in preparation.contexts:
        forecast = forecast_context(source, context)
        forecasts.append(forecast)
        causal = causal_operands(context, domain)
        masses = {}
        if causal is not None:
            for pulse in PULSES:
                projected = project_pulse(
                    causal.observation,
                    causal.previous,
                    pulse,
                    decision_id=f"{preparation.root}.{context.context}.{pulse.word_id}.projection",
                    history_id=f"{preparation.root}.{context.context}.causal",
                    horizon_id=HORIZON_ID,
                )
                masses[pulse] = projected.applied_mass_kg
                amounts.append(
                    (context.context, pulse, projected.applied_mass_kg, projected.realized_mass_kg)
                )
        temperature = None if causal is None else D(repr(causal.observation.temperature))
        if development is not None:

            def select(policy: str, horizon: int, request: D, budget: D) -> Pulse | None:
                return choose(
                    development=development,
                    context=context.context,
                    horizon=horizon,
                    policy=policy,
                    request=request,
                    budget=budget,
                    observed_temperature=temperature,
                    masses=masses,
                    forecast=forecast,
                    qualified=qualified,
                )

            grids.extend(
                FrontierPolicyGrid(
                    context.context,
                    horizon,
                    policy,
                    tuple(
                        select(policy, horizon, b, budget) for b in REQUESTS for budget in BUDGETS
                    ),
                )
                for horizon in HORIZONS
                for policy in POLICIES
            )
            primary.extend(
                (r, select("EL_BOUND", r.horizon_s, r.request_K, r.budget_kg))
                for r in requests
                if r.context == context.context
            )
    return FrontierPredictionSeal(
        preparation.root,
        ObjectIdentity.from_record(preparation.record_id, preparation),
        None
        if development is None
        else ObjectIdentity.from_record(development.record_id, development),
        parent_laws,
        tuple(forecasts),
        tuple(amounts),
        tuple(grids),
        tuple(primary),
    )
