"""Small finite-bound compatibility witness; no controller programme construction."""

from typing import Any

from dataclasses import replace
from decimal import Decimal
import numpy as np
from ..consumer import FiniteResponseLawConsumerRequest, consumer_task, response_coordinates, SPEC
from ..law_binding import output_quantities
from ..fitting import DELTA
from ..intervals import oriented_interval
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.finite_response_geometry import FiniteResponseBound


def compatibility(
    mean: Any,
    width: Any,
    entry: Any,
    directions: Any,
    requirements: Any,
    expected: Any,
    *,
    sigma: Any = None,
    q: Any = None,
) -> Any:
    """Compare every eligibility cell with native finite target-box containment.

    The installed coordinate/target/bound owners supply templates. Only the
    longitudinal request face varies over the request census. The independent
    precision and scalar-entry gates remain conjuncts, not response coordinates.
    """
    old = response_coordinates(ObjectIdentity.from_record("flh-science", SPEC))
    coordinates = tuple(replace(c, start=Decimal(4688), end=Decimal(4688)) for c in old)
    quantities = {q.quantity_id: j for j, q in enumerate(output_quantities())}
    templates = {}
    for c in range(2):
        for d in range(4):
            task = consumer_task(
                FiniteResponseLawConsumerRequest(f"retained-preparation-policy.constitution{c}.d{d}", "retained-preparation-policy.root", c, d, Decimal(".02")), old
            )
            intervals = tuple(
                replace(v, coordinate=coordinates[old.index(v.coordinate)])
                for v in task.boxes[0].intervals
            )
            mapped_task = replace(task, boxes=(replace(task.boxes[0], intervals=intervals),))
            templates[c, d] = mapped_task.boxes[0].intervals
    result = np.zeros_like(expected)
    n, ns = mean.shape[:2]
    for word in range(8):
        lo, hi = oriented_interval(mean - width, mean + width, word)
        mu, _ = oriented_interval(mean, mean, word)
        lo[..., 2:] = np.maximum(lo[..., 2:], 0)
        hi[..., 2:] = np.maximum(hi[..., 2:], 0)
        precise = np.isfinite(width[:, :, word // 2]).all(axis=-1) & (
            width[:, :, word // 2] <= DELTA
        ).all(axis=-1)
        # Authenticate finite bound representation once per root/schedule/word.
        for r in range(n):
            for s in range(ns):
                if not np.isfinite(lo[r, s]).all() or not np.isfinite(hi[r, s]).all():
                    continue
                for coordinate in coordinates[:8]:
                    j = quantities[coordinate.quantity_id]
                    b = FiniteResponseBound(
                        "retained-preparation-policy.bound",
                        "retained-preparation-policy.exposed-view",
                        coordinate,
                        Decimal(0),
                        Decimal(str(float(mu[r, s, j]))),
                        Decimal(0),
                        Decimal(str(float(lo[r, s, j]))),
                        Decimal(str(float(hi[r, s, j]))),
                        Decimal(0),
                    )
                    if tuple(map(float, b.expanded_interval)) != (lo[r, s, j], hi[r, s, j]):
                        raise ValueError("Finite owner changes native bounds")
        for c in range(2):
            for d in range(4):
                eligible = np.broadcast_to((entry & precise)[:, :, None], (n, ns, 256)).copy()
                for target in templates[c, d][:8]:
                    j = quantities[target.coordinate.quantity_id]
                    lower = np.full((n, 1, 256), float(target.lower))
                    upper = np.full((n, 1, 256), float(target.upper))
                    if j == d // 2:
                        if d % 2 == 0:
                            lower = requirements[:, None, :, c]
                        else:
                            upper = -requirements[:, None, :, c]
                    eligible &= (lo[:, :, None, j] >= lower) & (hi[:, :, None, j] <= upper)
                mask = directions[:, None, :, c] == d
                result[..., c, word] |= mask & eligible
    from ..preparation_policy_screen import _single_policy_choices

    expected_choice = np.where(expected.any(axis=-1), expected.argmax(axis=-1), -1)
    quantiles = np.ones(n) if q is None else np.asarray(q)
    for schedule in range(ns):
        for quantile in np.unique(quantiles):
            rows = np.flatnonzero(quantiles == quantile)
            inherited = _single_policy_choices(
                mean[rows, schedule],
                width[rows, schedule] - DELTA / 8 if sigma is None else sigma[rows, schedule],
                float(quantile),
                entry[rows, schedule],
                directions[rows],
                requirements[rows],
            )
            if not np.array_equal(inherited, expected_choice[rows, schedule]):
                raise ValueError("FINITE_DECISION_CHOICE_COMPATIBILITY_OBSTRUCTION")
    if not np.array_equal(result, expected):
        raise ValueError("FINITE_DECISION_COMPATIBILITY_OBSTRUCTION")
    return {
        "mapped": True,
        "eligibility_cells": int(result.size),
        "coordinate_terminal_tick": 4688,
        "causal_observation_tick": 4096,
        "handoff_tick": 4496,
        "entry_predicate": "finite-q AND L_m>=0 AND valid-causal-observation",
        "owners": [
            "FiniteResponseBound.expanded_interval",
            'FiniteTaskFunctionalSpec',
            "preparation_policy_screen._single_policy_choices",
            "consumer_task",
        ],
        "controller_programmes_constructed": 0,
    }
