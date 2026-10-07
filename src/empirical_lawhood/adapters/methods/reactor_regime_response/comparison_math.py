"Frozen six-policy root-paired local task comparisons after controller-use reveal."

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.stats import binom

from .control_math import REQUESTS_K, WordForecast

POLICIES = ("EL", "POINT", "DIRECT", "MECH", "FIXED_FEED_SIXTEEN_GRAMS_PER_SECOND", "FIXED_FEED_THIRTY_TWO_GRAMS_PER_SECOND")


def point_choice(
    request_K: float,
    chart: tuple[WordForecast, WordForecast, WordForecast],
) -> int | None:
    if request_K not in REQUESTS_K or tuple(word.word for word in chart) != (0, 1, 2):
        raise ValueError("point comparator request/chart differs")
    eligible = (
        word
        for word in chart[1:]
        if word.supported
        and word.projection_valid
        and word.predicted_cooling_K >= request_K
        and word.predicted_peak_K <= 356.2
    )
    chosen = min(
        eligible,
        key=lambda word: (word.delivered_mass_kg, word.requested_feed_kg_s, word.word),
        default=None,
    )
    return None if chosen is None else chosen.word


def fixed_choice(word: int, native_available: bool) -> int | None:
    if word not in (1, 2):
        raise ValueError("fixed comparator requires one declared nonzero native feed word")
    return word if native_available else None


@dataclass(frozen=True)
class PolicyRootOutcome:
    root: str
    policy: str
    source: str
    choices: tuple[int | None, int | None, int | None, int | None]
    success: tuple[bool, bool, bool, bool]
    delivered_mass_kg: tuple[float | None, float | None, float | None, float | None]
    refusal: tuple[bool, bool, bool, bool]

    def __post_init__(self) -> None:
        if self.policy not in POLICIES or self.source != ("OWNER" if self.policy == "EL" else "SHADOW_CHART"):
            raise ValueError("policy source/provenance differs")
        for choice, success, mass, refusal in zip(
            self.choices, self.success, self.delivered_mass_kg, self.refusal, strict=True
        ):
            if (
                choice not in (None, 1, 2)
                or refusal != (choice is None)
                or (success and (choice is None or mass is None))
                or (mass is not None and (not isfinite(mass) or not 0 <= mass <= 0.32))
            ):
                raise ValueError("policy action, success, refusal or native mass differs")

    @property
    def all_four(self) -> bool:
        return all(self.success)

    @property
    def mean_delivered_mass_kg(self) -> float:
        return sum(mass or 0.0 for mass in self.delivered_mass_kg) / 4

    @property
    def failure_penalized_feed_loss_kg(self) -> float:
        return sum(
            mass if success and mass is not None else 0.32
            for success, mass in zip(self.success, self.delivered_mass_kg, strict=True)
        ) / 4


@dataclass(frozen=True)
class DirectComparison:
    evaluable: bool
    el_success_roots: int
    direct_success_roots: int
    el_only: int
    direct_only: int
    paired_exact_one_sided_p: float | None
    improvement_fraction: float | None
    added_service_supported: bool
    reason: str | None


def compare_direct(
    el: tuple[PolicyRootOutcome, ...],
    direct: tuple[PolicyRootOutcome, ...] | None,
    *,
    primary_prospective_evaluation_passed: bool,
) -> DirectComparison:
    if (
        len(el) != 64
        or len({row.root for row in el}) != 64
        or any(row.policy != "EL" for row in el)
    ):
        raise ValueError("EL comparator requires all 64 actual-owner roots")
    if direct is None or len(direct) != 64 or tuple(row.root for row in direct) != tuple(row.root for row in el):
        return DirectComparison(False, sum(row.all_four for row in el), 0, 0, 0, None, None, False, "DIRECT_COMPARISON_MISSING")
    if any(row.policy != "DIRECT" for row in direct):
        raise ValueError("DIRECT comparator changed its frozen policy identity")
    el_only = sum(a.all_four and not b.all_four for a, b in zip(el, direct, strict=True))
    direct_only = sum(b.all_four and not a.all_four for a, b in zip(el, direct, strict=True))
    discordant = el_only + direct_only
    p = 1.0 if discordant == 0 else float(binom.sf(el_only - 1, discordant, 0.5))
    improvement = (sum(a.all_four for a in el) - sum(b.all_four for b in direct)) / 64
    return DirectComparison(
        True,
        sum(row.all_four for row in el),
        sum(row.all_four for row in direct),
        el_only,
        direct_only,
        p,
        improvement,
        bool(primary_prospective_evaluation_passed and p <= 0.05 and improvement >= 0.05),
        None,
    )


def paired_descriptive_effects(
    el: tuple[PolicyRootOutcome, ...],
    rival: tuple[PolicyRootOutcome, ...],
) -> dict[str, object]:
    if (
        len(el) != 64
        or len(rival) != 64
        or tuple(row.root for row in el) != tuple(row.root for row in rival)
        or any(row.policy != "EL" for row in el)
        or len({row.policy for row in rival}) != 1
        or rival[0].policy == "EL"
    ):
        raise ValueError("descriptive comparison requires all 64 paired roots")
    measures = {
        "all_four_success_fraction": np.asarray(
            [float(a.all_four) - float(b.all_four) for a, b in zip(el, rival, strict=True)]
        ),
        "mean_delivered_mass_kg": np.asarray(
            [a.mean_delivered_mass_kg - b.mean_delivered_mass_kg for a, b in zip(el, rival, strict=True)]
        ),
        "refusal_fraction": np.asarray(
            [sum(a.refusal) / 4 - sum(b.refusal) / 4 for a, b in zip(el, rival, strict=True)]
        ),
        "failure_penalized_feed_loss_kg": np.asarray(
            [a.failure_penalized_feed_loss_kg - b.failure_penalized_feed_loss_kg for a, b in zip(el, rival, strict=True)]
        ),
    }
    rng = np.random.default_rng(20260925)
    indices = rng.integers(0, 64, size=(20000, 64))
    return {
        "policy": rival[0].policy,
        "roots": 64,
        "draws": 20000,
        "seed": 20260925,
        "effects_EL_minus_rival": {
            name: {
                "mean": float(np.mean(values)),
                "percentile_95": (
                    float(np.quantile(np.mean(values[indices], axis=1), 0.025)),
                    float(np.quantile(np.mean(values[indices], axis=1), 0.975)),
                ),
            }
            for name, values in measures.items()
        },
    }
