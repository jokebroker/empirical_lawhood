"Corrected synthetic propulsion discovery comparator for immutable attempt 2.\n\nSynthetic propulsion attempt 1 used a truth-aware one-step greedy comparator named\n``oracle_upper_bound``.  Because that heuristic could exhaust its cost budget,\nit was not an upper bound.  This additive module preserves attempt 1 and\nreplaces only that arm with an exhaustive minimum-cost plan that reaches the\ntruth's full observable equivalence class whenever such a plan exists within\nthe frozen budget.\n"

from __future__ import annotations

from functools import lru_cache
from itertools import combinations
from typing import Any, Sequence

from . import synthetic_propulsion_discovery as base


_HYPOTHESES = base.closure_hypotheses()
_ALL_HYPOTHESIS_MASK = (1 << len(_HYPOTHESES)) - 1
_OUTCOME_MASKS: dict[str, dict[str, int]] = {}
for _experiment in base.EXPERIMENTS:
    _partitions: dict[str, int] = {}
    for _index, _hypothesis in enumerate(_HYPOTHESES):
        _outcome = base.predicted_outcome(_hypothesis, _experiment)
        _partitions[_outcome] = _partitions.get(_outcome, 0) | (1 << _index)
    _OUTCOME_MASKS[_experiment.experiment_id] = _partitions


def _survivor_mask(
    truth: base.ClosureHypothesis,
    selected: Sequence[base.InterfaceExperiment],
) -> int:
    mask = _ALL_HYPOTHESIS_MASK
    for experiment in selected:
        outcome = base.predicted_outcome(truth, experiment)
        mask &= _OUTCOME_MASKS[experiment.experiment_id][outcome]
    return mask


@lru_cache(maxsize=1024)
def minimum_cost_oracle_plan(
    truth: base.ClosureHypothesis,
    *,
    maximum_acts: int = base.MAXIMUM_ACTS,
    maximum_cost: int = base.MAXIMUM_COST,
) -> tuple[str, ...]:
    """Return the cheapest exact observable-class separating experiment set."""

    equivalence_mask = _survivor_mask(truth, base.EXPERIMENTS)
    candidates: list[tuple[int, int, tuple[str, ...]]] = []
    for act_count in range(maximum_acts + 1):
        for selected in combinations(base.EXPERIMENTS, act_count):
            cost = sum(value.cost for value in selected)
            if cost > maximum_cost:
                continue
            if _survivor_mask(truth, selected) == equivalence_mask:
                experiment_ids = tuple(value.experiment_id for value in selected)
                candidates.append((cost, act_count, experiment_ids))
    if not candidates:
        raise ValueError("Synthetic propulsion corrected oracle cannot resolve within the frozen budget")
    return min(candidates)[2]


def run_discovery_method(
    *,
    world: base.ClosureWorld,
    method_id: str,
    maximum_acts: int = base.MAXIMUM_ACTS,
    maximum_cost: int = base.MAXIMUM_COST,
) -> dict[str, Any]:
    if method_id != "oracle_upper_bound":
        return base.run_discovery_method(
            world=world,
            method_id=method_id,
            maximum_acts=maximum_acts,
            maximum_cost=maximum_cost,
        )
    hypotheses = base.closure_hypotheses()
    surviving = list(hypotheses)
    equivalence_ids = set(base.observable_equivalence_class(world.truth, hypotheses))
    plan = minimum_cost_oracle_plan(
        world.truth,
        maximum_acts=maximum_acts,
        maximum_cost=maximum_cost,
    )
    decisions: list[dict[str, Any]] = []
    cost_used = 0
    for experiment_id in plan:
        selected = base.EXPERIMENT_BY_ID[experiment_id]
        observed = base.predicted_outcome(world.truth, selected)
        before = len(surviving)
        surviving = [
            value for value in surviving if base.predicted_outcome(value, selected) == observed
        ]
        cost_used += selected.cost
        decisions.append(
            {
                "act_index": len(decisions) + 1,
                "experiment_id": selected.experiment_id,
                "interface_id": selected.interface_id,
                "observed_outcome": observed,
                "version_size_before": before,
                "version_size_after": len(surviving),
                "requested_experiment": selected.experiment_id,
                "accepted_experiment": selected.experiment_id,
                "applied_experiment": selected.experiment_id,
                "realized_experiment": selected.experiment_id,
                "observation_cost": selected.cost,
            }
        )
    surviving_ids = {value.hypothesis_id for value in surviving}
    equivalence_resolved = surviving_ids == equivalence_ids
    if not equivalence_resolved:
        raise ValueError("Synthetic propulsion corrected oracle plan failed its exact-resolution contract")
    stop_reason = (
        "STRUCTURAL_CLASS_RESOLVED"
        if len(equivalence_ids) == 1
        else "OBSERVATIONALLY_NONIDENTIFIABLE"
    )
    return {
        "world_id": world.world_id,
        "split": world.split,
        "method_id": method_id,
        "truth_hypothesis_id": world.truth.hypothesis_id,
        "true_observable_equivalence_size": len(equivalence_ids),
        "true_structurally_identifiable": len(equivalence_ids) == 1,
        "acts_used": len(decisions),
        "cost_used": cost_used,
        "version_size_final": len(surviving),
        "version_collapse_fraction": 1.0 - len(surviving) / len(hypotheses),
        "equivalence_class_resolved": True,
        "exact_truth_resolved": len(equivalence_ids) == 1,
        "truth_retained": world.truth.hypothesis_id in surviving_ids,
        "false_promotion": world.truth.hypothesis_id not in surviving_ids,
        "stop_reason": stop_reason,
        "surviving_hypothesis_ids": tuple(value.hypothesis_id for value in surviving),
        "decisions": tuple(decisions),
        "oracle_deployable": False,
        "oracle_objective": "MINIMUM_TOTAL_OBSERVATION_COST_THEN_ACT_COUNT",
        "oracle_plan_proved_by_exhaustive_subset_search": True,
    }


def run_discovery_benchmark(
    worlds: Sequence[base.ClosureWorld],
) -> tuple[dict[str, Any], ...]:
    return tuple(
        run_discovery_method(world=world, method_id=method_id)
        for world in worlds
        for method_id in base.METHOD_IDS
    )
