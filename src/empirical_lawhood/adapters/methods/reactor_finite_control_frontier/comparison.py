"""Whole-root finite-chart comparisons; request rows never become independent n."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar
import numpy as np
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.methods.whole_root_bootstrap import root_indices, nearest_rank_interval
from .config import BUDGETS, COORDINATES, POLICIES, REQUESTS, ROOTS, FrontierDesign
from .discovery import safe_service
from .measurement import FrontierMeasuredRoot
from .selection import FrontierPredictionSeal


@dataclass(frozen=True, slots=True)
class FrontierPolicyScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-policy-score'
    root: str
    policy: str
    evaluable: bool
    assigned: int
    admitted: int
    served: int
    false_admissions: int
    unknown: int
    served_mass_kg: D
    served_cooling_K: D
    common_with_el: int
    common_el_mass_kg: D
    common_policy_mass_kg: D


@dataclass(frozen=True, slots=True)
class FrontierContrast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-contrast'
    comparator: str
    evaluable: bool
    service_gain: D
    interval_95: tuple[D, D]
    interval_family: tuple[D, D]
    conclusion: str
    common_requests: int
    conditional_cost_ratio: D | None
    conditional_cost_interval_95: tuple[D, D] | None


@dataclass(frozen=True, slots=True)
class FrontierComparison(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-comparison'
    role: str
    measurements: tuple[ObjectIdentity, ...]
    seals: tuple[ObjectIdentity, ...]
    scores: tuple[FrontierPolicyScore, ...]
    contrasts: tuple[FrontierContrast, ...]
    oracle_served: tuple[tuple[str, int], ...]

    @property
    def record_id(self) -> str:
        return f"reactor-finite-control-frontier.{self.role}-comparison"


def comparison(
    measurements: tuple[FrontierMeasuredRoot, ...],
    seals: tuple[FrontierPredictionSeal, ...],
    *,
    role: str,
) -> FrontierComparison:
    expected = tuple(r for r, phase, _, _ in ROOTS if phase == role)
    if (
        role not in ("qualification", "prospective")
        or tuple(m.root for m in measurements) != expected
        or tuple(s.root for s in seals) != expected
    ):
        raise ValueError("comparison changed the whole assigned root roster")
    scores, oracle = [], []
    for m, seal in zip(measurements, seals, strict=True):
        if m.preparation != seal.preparation:
            raise ValueError("comparison substituted a prediction made after source contact")
        rows = {o.coordinate: o for o in m.observations}
        choices = {(g.context, g.horizon_s, g.policy): g.choices for g in seal.grids}
        outcomes = {}
        for policy in POLICIES:
            actual = []
            for context, horizon, p in choices:
                if p != policy:
                    continue
                for (b, budget), pulse in zip(
                    ((b, B) for b in REQUESTS for B in BUDGETS),
                    choices[context, horizon, policy],
                    strict=True,
                ):
                    row = (
                        None
                        if pulse is None
                        else next(
                            rows[c]
                            for c in COORDINATES
                            if (c.context, c.horizon_s, c.pulse) == (context, horizon, pulse)
                        )
                    )
                    served = row is not None and safe_service(row, b, budget)
                    known = row is None or row.evaluable
                    actual.append(
                        (
                            row is not None,
                            known,
                            served,
                            D(0) if row is None or not row.evaluable else row.applied_masses_kg[0],
                            D(0) if row is None or not row.evaluable else min(row.effects_K),
                        )
                    )
            if len(actual) != 1008:
                raise ValueError("comparison omitted a fixed context/window/grid coordinate")
            outcomes[policy] = actual
        for policy in POLICIES:
            actual = outcomes[policy]
            shared = [
                (el, other)
                for el, other in zip(outcomes[POLICIES[0]], actual, strict=True)
                if el[2] and other[2]
            ]
            scores.append(
                FrontierPolicyScore(
                    m.root,
                    policy,
                    all(a[1] for a in actual),
                    1008,
                    sum(a[0] for a in actual),
                    sum(a[2] for a in actual),
                    sum(a[0] and a[1] and not a[2] for a in actual),
                    sum(not a[1] for a in actual),
                    sum((a[3] for a in actual if a[2]), D(0)),
                    sum((a[4] for a in actual if a[2]), D(0)),
                    len(shared),
                    sum((a[3] for a, _ in shared), D(0)),
                    sum((b[3] for _, b in shared), D(0)),
                )
            )
        oracle.append(
            (
                m.root,
                sum(
                    any(
                        safe_service(o, b, budget)
                        for o in m.observations
                        if (o.coordinate.context, o.coordinate.horizon_s) == (c, t)
                    )
                    for c, t, p in choices
                    if p == POLICIES[0]
                    for b in REQUESTS
                    for budget in BUDGETS
                ),
            )
        )
    contrasts = []
    if role == "prospective":
        design = FrontierDesign()
        indices = root_indices(roots=64, draws=design.bootstrap_draws, seed=design.bootstrap_seed)
        interval = nearest_rank_interval

        el = tuple(s for s in scores if s.policy == POLICIES[0])
        for policy in POLICIES[1:]:
            other = tuple(s for s in scores if s.policy == policy)
            gains = np.array([(a.served - b.served) / 1008 for a, b in zip(el, other, strict=True)])
            sampled = gains[indices].mean(axis=1)
            ordinary, adjusted = interval(sampled, 0.025), interval(sampled, 0.05 / 8)
            gain = D(repr(float(gains.mean())))
            evaluable = all(s.evaluable for s in (*el, *other))
            conclusion = (
                "UNEVALUABLE"
                if not evaluable
                else "SUPERIOR"
                if gain >= D(".05") and adjusted[0] > 0
                else "OPPOSED"
                if adjusted[1] < 0
                else "UNRESOLVED"
            )
            numerator = np.array([float(s.common_el_mass_kg) for s in other])
            denominator = np.array([float(s.common_policy_mass_kg) for s in other])
            ratio = band = None
            draws = denominator[indices].sum(axis=1)
            # An undefined root-resampled ratio remains undefined, never silently discarded.
            if denominator.sum() > 0 and (draws > 0).all():
                ratio = D(repr(float(numerator.sum() / denominator.sum())))
                band = interval(numerator[indices].sum(axis=1) / draws, 0.025)
            contrasts.append(
                FrontierContrast(
                    policy,
                    evaluable,
                    gain,
                    ordinary,
                    adjusted,
                    conclusion,
                    sum(s.common_with_el for s in other),
                    ratio,
                    band,
                )
            )
    return FrontierComparison(
        role,
        tuple(ObjectIdentity.from_record(f"{m.root}.measurement", m) for m in measurements),
        tuple(ObjectIdentity.from_record(s.record_id, s) for s in seals),
        tuple(scores),
        tuple(contrasts),
        tuple(oracle),
    )
