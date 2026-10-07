"""Development-only, service-aware selection of one fixed local route.

All candidate fits and support boxes come from fit roots.  These operations
see nomination assays only and produce no calibrated or qualified law.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .control_math import REQUESTS_K, forecast_chart, select_word
from .config import ROOTS
from .measured_panel import MeasuredContext
from .model_selection import Candidate

ROUTES = ("prepared_t0", "c_q", "p_q")


@dataclass(frozen=True)
class NominationRoot:
    root: str
    route: str
    chart: MeasuredContext
    projected_masses_kg: tuple[float, float, float] | None
    projection_valid: tuple[bool, bool, bool]
    native_word_valid: tuple[bool, bool, bool]
    preparation_prefix_valid: bool
    preparation_nominal_max_K: float | None
    preparation_refined_max_K: float | None

    def __post_init__(self) -> None:
        if self.route not in ROUTES or self.chart.root != self.root or self.chart.context != self.route:
            raise ValueError("nomination root/route context differs")
        if len(self.projection_valid) != 3 or len(self.native_word_valid) != 3:
            raise ValueError("nomination lacks its complete native word validity")
        masses = self.projected_masses_kg
        if masses is not None and (
            len(masses) != 3
            or not all(isfinite(mass) for mass in masses)
            or not masses[0] == 0 < masses[1] < masses[2]
        ):
            raise ValueError("nomination projected action chart differs")

    @property
    def preparation_safe(self) -> bool:
        return (
            self.preparation_prefix_valid
            and self.preparation_nominal_max_K is not None
            and self.preparation_refined_max_K is not None
            and isfinite(self.preparation_nominal_max_K)
            and isfinite(self.preparation_refined_max_K)
            and max(self.preparation_nominal_max_K, self.preparation_refined_max_K) <= 356.2
        )

    @property
    def projection_matches_measured(self) -> bool:
        masses = self.projected_masses_kg
        return bool(
            masses is not None
            and self.chart.nominal_mass_kg is not None
            and self.chart.refined_mass_kg is not None
            and all(
                abs(projected - actual) <= 1e-6
                for actuals in (self.chart.nominal_mass_kg, self.chart.refined_mass_kg)
                for projected, actual in zip(masses, actuals, strict=True)
            )
        )


@dataclass(frozen=True)
class NominationOpportunity:
    root: str
    route: str
    candidate_id: str
    q_temperature: float | None
    q_cooling: float | None
    choices: tuple[int | None, int | None, int | None, int | None]
    joined_service: bool
    chart_valid: bool
    covered: bool
    support_valid: bool
    preparation_safe: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RouteCandidate:
    route: str
    coefficient: Candidate
    absolute: Candidate
    direct: Candidate | None
    provisional_q_temperature: float | None
    provisional_q_cooling: float | None
    mean_contrast_loss_K2: float | None
    joined_service_roots: int
    opportunities: tuple[NominationOpportunity, ...]

    @property
    def max_precision_score(self) -> float:
        if self.provisional_q_temperature is None or self.provisional_q_cooling is None:
            return float("inf")
        return max(self.provisional_q_temperature, self.provisional_q_cooling)


def _x(chart: MeasuredContext, mask: str) -> np.ndarray | None:
    features = chart.causal_current if mask == "current" else chart.causal_history
    return None if features is None else np.asarray(features, dtype=np.float64)[None, :]


def _predict(candidate: Candidate, chart: MeasuredContext) -> tuple[float, bool] | None:
    if candidate.fitted is None:
        return None
    features = _x(chart, candidate.mask)
    if features is None:
        return None
    result = float(candidate.fitted.predict(features)[0])
    return (result, bool(candidate.contains(features)[0])) if isfinite(result) else None


def _rank(candidates: tuple[Candidate, ...], roots: tuple[NominationRoot, ...]) -> tuple[Candidate, float] | None:
    scored = []
    for candidate in candidates:
        if not candidate.eligible:
            continue
        losses = []
        for root in roots:
            chart = root.chart
            prediction = _predict(candidate, chart)
            if (
                not chart.valid
                or chart.nominal_mass_kg is None
                or chart.nominal_cooling_K is None
                or prediction is None
            ):
                continue
            predicted, _ = prediction
            masses = np.asarray(chart.nominal_mass_kg[1:])
            cooling = np.asarray(chart.nominal_cooling_K[1:])
            losses.append(float(np.mean((predicted * masses - cooling) ** 2)))
        if losses:
            scored.append((float(np.mean(losses)), candidate))
    if not scored:
        return None
    minimum = min(loss for loss, _ in scored)
    tied = [(loss, candidate) for loss, candidate in scored if loss <= 1.01 * minimum]
    loss, candidate = min(
        tied,
        key=lambda pair: (
            pair[1].leaf_count,
            pair[1].parameter_count,
            -pair[1].penalty,
            pair[1].model_id,
        ),
    )
    return candidate, loss


def _absolute_for_mask(candidates: tuple[Candidate, ...], mask: str) -> Candidate:
    eligible = [candidate for candidate in candidates if candidate.eligible and candidate.mask == mask]
    if not eligible:
        raise ValueError("selected route has no fitted empirical absolute baseline")
    minimum = min(float(candidate.nomination_mean_loss_K2) for candidate in eligible if candidate.nomination_mean_loss_K2 is not None)
    tied = [
        candidate for candidate in eligible
        if candidate.nomination_mean_loss_K2 is not None
        and candidate.nomination_mean_loss_K2 <= 1.01 * minimum
    ]
    return min(tied, key=lambda candidate: (candidate.parameter_count, -candidate.penalty, candidate.model_id))


def _errors(
    root: NominationRoot, coefficient: Candidate, absolute: Candidate
) -> tuple[float, float] | None:
    chart = root.chart
    k = _predict(coefficient, chart)
    p0 = _predict(absolute, chart)
    if (
        not chart.valid
        or k is None
        or p0 is None
        or root.projected_masses_kg is None
        or chart.nominal_peak_K is None
        or chart.refined_peak_K is None
        or chart.nominal_cooling_K is None
        or chart.refined_cooling_K is None
        or not (k[1] and p0[1])
        or not all(root.projection_valid)
        or not root.projection_matches_measured
    ):
        return None
    masses = root.projected_masses_kg
    q_t = q_c = 0.0
    for peak, cooling in (
        (chart.nominal_peak_K, chart.nominal_cooling_K),
        (chart.refined_peak_K, chart.refined_cooling_K),
    ):
        for word, mass in enumerate(masses):
            c_hat = k[0] * mass
            p_hat = p0[0] - c_hat
            q_t = max(q_t, abs(p_hat - peak[word]) / 0.25)
            q_c = max(
                q_c,
                abs(c_hat - cooling[word]) / (0.00005 + 0.5 * abs(c_hat)),
            )
    return q_t, q_c


def _opportunity(
    root: NominationRoot,
    coefficient: Candidate,
    absolute: Candidate,
    q_t: float | None,
    q_c: float | None,
) -> NominationOpportunity:
    chart = root.chart
    k, p0 = _predict(coefficient, chart), _predict(absolute, chart)
    supported = bool(
        k is not None
        and p0 is not None
        and k[1]
        and p0[1]
        and root.projection_matches_measured
    )
    reasons = set(chart.reasons)
    if not supported:
        reasons.add("OUTSIDE_FROZEN_FIT_SUPPORT_OR_NO_PREDICTION")
    if not root.preparation_safe:
        reasons.add("PREPARATION_UNSAFE_OR_INVALID")
    if q_t is None or q_c is None:
        reasons.add("NO_FINITE_PROVISIONAL_PRECISION")
    choices: tuple[int | None, int | None, int | None, int | None] = (None,) * 4
    covered = False
    if k is not None and p0 is not None and root.projected_masses_kg is not None and q_t is not None and q_c is not None:
        forecasts = forecast_chart(
            p0[0],
            k[0],
            root.projected_masses_kg,
            q_t,
            q_c,
            (supported,) * 3,
            root.projection_valid,
        )
        choices = tuple(
            select_word(
                request,
                forecasts,
                law_valid=q_t <= 1 and q_c <= 1,
                preparation_safe=root.preparation_safe,
                clock_valid=chart.callback is not None,
                authority_valid=True,
                numerical_valid=chart.valid,
            ).choice
            for request in REQUESTS_K
        )  # type: ignore[assignment]
        covered = bool(
            chart.valid
            and chart.nominal_peak_K is not None
            and chart.refined_peak_K is not None
            and chart.nominal_cooling_K is not None
            and chart.refined_cooling_K is not None
            and all(
                abs(word.predicted_peak_K - peak[word.word]) <= word.temperature_halfwidth_K
                and abs(word.predicted_cooling_K - cooling[word.word]) <= word.cooling_halfwidth_K
                for peak, cooling in (
                    (chart.nominal_peak_K, chart.nominal_cooling_K),
                    (chart.refined_peak_K, chart.refined_cooling_K),
                )
                for word in forecasts
            )
        )
    joined = (
        chart.valid
        and supported
        and covered
        and root.preparation_safe
        and all(choice is not None for choice in choices)
        and chart.nominal_peak_K is not None
        and chart.refined_peak_K is not None
        and chart.nominal_cooling_K is not None
        and chart.refined_cooling_K is not None
        and all(
            choice is not None
            and root.native_word_valid[choice]
            and chart.nominal_cooling_K[choice] >= request
            and chart.refined_cooling_K[choice] >= request
            and chart.nominal_peak_K[choice] <= 356.2
            and chart.refined_peak_K[choice] <= 356.2
            for request, choice in zip(REQUESTS_K, choices, strict=True)
        )
    )
    return NominationOpportunity(
        root.root,
        root.route,
        coefficient.model_id,
        q_t,
        q_c,
        choices,
        bool(joined),
        chart.valid,
        covered,
        supported,
        root.preparation_safe,
        tuple(sorted(reasons)),
    )


def nominate_route(
    roots: tuple[NominationRoot, ...],
    coefficient_candidates: tuple[Candidate, ...],
    absolute_candidates: tuple[Candidate, ...],
) -> RouteCandidate:
    if (
        len(roots) != 16
        or len({root.root for root in roots}) != 16
        or len({root.route for root in roots}) != 1
        or tuple(root.root for root in roots)
        != tuple(name for name, role, _, _ in ROOTS if role == "nomination")
    ):
        raise ValueError("route nomination requires its exact 16 independent development roots")
    ranked = _rank(coefficient_candidates, roots)
    if ranked is None:
        raise ValueError("route has no finite nomination candidate")
    coefficient, loss = ranked
    absolute = _absolute_for_mask(absolute_candidates, coefficient.mask)
    direct_ranked = _rank(
        tuple(candidate for candidate in coefficient_candidates if candidate.mask == coefficient.mask and candidate.family in ("K", "affine", "rbf")),
        roots,
    )
    direct = None if direct_ranked is None else direct_ranked[0]
    scores = tuple(_errors(root, coefficient, absolute) for root in roots)
    contacted = tuple(score for score in scores if score is not None)
    q_t = max(score[0] for score in contacted) if contacted else None
    q_c = max(score[1] for score in contacted) if contacted else None
    opportunities = tuple(_opportunity(root, coefficient, absolute, q_t, q_c) for root in roots)
    return RouteCandidate(
        roots[0].route,
        coefficient,
        absolute,
        direct,
        q_t,
        q_c,
        loss,
        sum(row.joined_service for row in opportunities),
        opportunities,
    )


def select_route(
    routes: tuple[RouteCandidate | None, RouteCandidate | None, RouteCandidate | None],
) -> RouteCandidate | None:
    if any(route is not None and route.route != expected for route, expected in zip(routes, ROUTES, strict=True)):
        raise ValueError("selected route requires prepared, C and P in the frozen order")
    finite = [route for route in routes if route is not None]
    if not finite:
        return None
    eligible = [route for route in finite if route.max_precision_score <= 1]
    if eligible:
        maximum = max(route.joined_service_roots for route in eligible)
        tied = [route for route in eligible if route.joined_service_roots == maximum]
        score = min(route.max_precision_score for route in tied)
        tied = [route for route in tied if route.max_precision_score <= 1.01 * score]
    else:
        score = min(route.max_precision_score for route in finite)
        tied = [route for route in finite if route.max_precision_score <= 1.01 * score]
    loss = min(route.mean_contrast_loss_K2 if route.mean_contrast_loss_K2 is not None else float("inf") for route in tied)
    tied = [
        route for route in tied
        if route.mean_contrast_loss_K2 is not None
        and route.mean_contrast_loss_K2 <= 1.01 * loss
    ]
    return min(tied, key=lambda route: ROUTES.index(route.route))
