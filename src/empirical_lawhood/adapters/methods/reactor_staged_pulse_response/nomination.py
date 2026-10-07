"""Independent menu and sequence development screens; no fresh-Q adaptation."""

from decimal import Decimal as D

from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.discovery import FrontierDevelopment
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.measurement import FrontierMeasuredRoot
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.records import FrontierPreparation
from empirical_lawhood.kernel.provenance import ObjectIdentity
from .config import DEVELOPMENT_ROOTS, FIRST, JOINT, LOCAL_BASE, LOCAL_EXPANDED, LOCAL_REQUESTS, PAIR_REQUESTS
from .discovery import ClassicalBound, ClassicalNomination, discover_bound, fixed_second_words, ladder, recipe, retained_bounds
from .measurement import ClassicalMeasuredRoot, retained_observation
from .prediction import ClassicalCausalPoint, ClassicalInducedPrediction, ClassicalPredictionSeal, eligible, predict


def first_compatibility(
    development: FrontierDevelopment,
    preparations: tuple[FrontierPreparation, ...],
    measurements: tuple[FrontierMeasuredRoot, ...],
) -> ClassicalBound:
    if (
        tuple(p.root for p in preparations) != DEVELOPMENT_ROOTS
        or tuple(m.root for m in measurements) != DEVELOPMENT_ROOTS
    ):
        raise ValueError("first-word compatibility requires the exact retained B root census")
    bound = retained_bounds(development, first_only=True)[0]
    valid, unsafe = [], []
    values: list[D] = []
    from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.discovery import round_down, round_up

    for p, m in zip(preparations, measurements, strict=True):
        if m.preparation != ObjectIdentity.from_record(p.record_id, p):
            raise ValueError("retained first response replaces its original preparation")
        row = retained_observation(m, JOINT[0])
        if row.valid:
            valid.append(m.root)
            values.extend(row.pair("c1"))
        if row.unsafe:
            unsafe.append(m.root)
    if tuple(valid) != bound.valid_roots or tuple(unsafe) != bound.unsafe_roots:
        raise ValueError("retained first bound changes its historical contact/unsafe denominator")
    if values:
        low, high = min(values), max(values)
        old = bound.response("c1")
        if old.lower != round_down(
            low - D(".1") * abs(low) - D(".000001")
        ) or old.upper != round_up(high + D(".1") * abs(high) + D(".000001"), ".00001"):
            raise ValueError("first-word compatibility changes its retained empirical extrema")
    return bound


def first_opportunity(bound: ClassicalBound) -> bool:
    return (
        bound.nominated
        and bool(bound.responses)
        and (ladder(bound.response("c1").lower) or D(0)) >= D(".0012")
    )


def menu_nomination(
    development: FrontierDevelopment,
    measurements: tuple[ClassicalMeasuredRoot, ...],
    seals: tuple[ClassicalPredictionSeal, ...],
) -> ClassicalNomination:
    if (
        tuple(m.root for m in measurements) != DEVELOPMENT_ROOTS
        or tuple(s.root for s in seals) != DEVELOPMENT_ROOTS
        or any(
            s.nomination is not None or s.preparation != m.preparation
            for s, m in zip(seals, measurements, strict=True)
        )
    ):
        raise ValueError("menu development changed its exact pre-response B preparation census")
    old = {b.coordinate: b for b in retained_bounds(development)}
    bounds = tuple(
        old[c] if c in LOCAL_BASE else discover_bound(c, measurements) for c in LOCAL_EXPANDED
    )
    recipes = tuple(recipe("EL_SERVICE_MENU", b, D(1)) for b in bounds)
    opportunities = []
    for seal in seals:
        for r in recipes:
            point = next(p for p in seal.points if p.context == r.bound.coordinate.context)
            prediction = predict(r, point)
            for request in LOCAL_REQUESTS:
                if eligible(prediction, request):
                    opportunities.append((seal.root, request.request_id, r.recipe_id))
    parents = (
        ObjectIdentity.from_record(development.record_id, development),
        *(ObjectIdentity.from_record(m.record_id, m) for m in measurements),
        *(ObjectIdentity.from_record(s.record_id, s) for s in seals),
    )
    return ClassicalNomination(
        "expanded-menu-comparison",
        parents,
        recipes,
        (),
        tuple(opportunities),
        () if opportunities else ("NO_DEVELOPMENT_TASK_OPPORTUNITY",),
    )


def sequence_nomination(
    first: ClassicalBound,
    measurements: tuple[ClassicalMeasuredRoot, ...],
    seals: tuple[ClassicalPredictionSeal, ...],
    induced: tuple[ClassicalInducedPrediction, ...],
    *,
    historical_parent: ObjectIdentity,
) -> ClassicalNomination:
    if (
        tuple(m.root for m in measurements) != DEVELOPMENT_ROOTS
        or tuple(s.root for s in seals) != DEVELOPMENT_ROOTS
        or any(
            s.nomination is not None or s.preparation != m.preparation
            for s, m in zip(seals, measurements, strict=True)
        )
        or len({s.context.root for s in induced}) != len(induced)
    ):
        raise ValueError("sequence development replaced or dropped a fixed root/cutoff")
    bounds = (first, *(discover_bound(c, measurements, first=first) for c in JOINT[1:]))
    recipes = tuple(recipe("EL_SEQUENCE_RELATION", b, D(1)) for b in bounds)
    opportunities = []
    for seal in seals:
        second = next((s for s in induced if s.context.root == seal.root), None)
        if second is None or second.first_seal != ObjectIdentity.from_record(seal.record_id, seal):
            continue
        initial = next(p for p in seal.points if p.context == "early")
        first_mass = next((mass for w, mass, _ in initial.masses if w == FIRST), None)
        if first_mass is None:
            continue
        one = predict(recipes[0], initial)
        for r in recipes[2:]:
            prediction = predict(r, second.point, first_point=initial, first_mass=first_mass)
            for request in PAIR_REQUESTS:
                if eligible(one, request) and eligible(prediction, request):
                    opportunities.append((seal.root, request.request_id, r.recipe_id))
    parents = (
        historical_parent,
        *(ObjectIdentity.from_record(m.record_id, m) for m in measurements),
        *(ObjectIdentity.from_record(s.record_id, s) for s in seals),
        *(ObjectIdentity.from_record(s.record_id, s) for s in induced),
    )
    return ClassicalNomination(
        "staged-sequence-comparison",
        parents,
        recipes,
        fixed_second_words(measurements),
        tuple(opportunities),
        () if opportunities else ("NO_JOINT_DEVELOPMENT_TASK_OPPORTUNITY",),
    )


def initial_pair_opportunities(
    nomination: ClassicalNomination,
    first_point: ClassicalCausalPoint,
    qualified: tuple[str, ...],
    request_id: str,
    projected_second_masses: dict[str, D],
    *,
    fixed: bool,
) -> tuple[str, ...]:
    """Only t0-known gates; the actual t1 thermal gate remains prospective."""
    request = next(r for r in PAIR_REQUESTS if r.request_id == request_id)
    first = nomination.recipes[0]
    if first.recipe_id not in qualified or not eligible(predict(first, first_point), request):
        return ()
    initial_mass = next((m for w, m, _ in first_point.masses if w == FIRST), None)
    selected = dict(nomination.fixed_seconds)[request_id]
    accepted = []
    for r in nomination.recipes[2:]:
        c, b = r.bound.coordinate, r.bound
        mass = projected_second_masses.get(c.pulse.word_id)
        if (
            r.recipe_id not in qualified
            or (fixed and c.pulse != selected)
            or initial_mass is None
            or mass is None
            or first_point.temperature_K is None
            or b.thermal_K is None
        ):
            continue
        global_lower = r.lower("g")
        if (
            (r.lower("c1") or D("-1")) >= request.required_K
            and (r.lower("c2") or D("-1")) >= (request.second_K or D(0))
            and global_lower is not None
            and global_lower >= 0
            and initial_mass + mass <= request.budget_kg + D("1e-10")
            and first_point.temperature_K + b.thermal_K <= D("356.2")
        ):
            accepted.append(r.recipe_id)
    return tuple(accepted)
