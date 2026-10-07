"Truth-known closed-loop response discovery benchmark.\n\nThe scientific unit is a complete generated world.  Algorithm trajectories,\nqueries, posterior alternatives, and timing samples are nested observations and\nare never treated as independent replication.  Deployable selectors receive\nonly the declarative experiment library, the current truth-blind posterior,\nand prior observations.  Privileged truth is used solely by the generated\nobservation boundary and the explicitly non-deployable oracle comparator.\n"

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
import math
from time import perf_counter_ns
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .response_formalization import bootstrap_mean_interval


FloatArray = npt.NDArray[np.float64]

AXIS_IDS = (
    "delivery_observed",
    "receiver_closed",
    "action_rank_full",
    "regime_changes",
    "topology_periodic",
    "hidden_reservoir",
    "topology_factorization",
    "interface_obstruction",
)
METHOD_IDS = (
    "response_version_space",
    "random_space_filling",
    "uncertainty_sampling",
    "bayesian_optimization",
    "active_system_identification",
    "topology_blind_response",
    "oracle_upper_bound",
)
DEPLOYABLE_METHOD_IDS = tuple(value for value in METHOD_IDS if value != "oracle_upper_bound")
MAXIMUM_ACTS = 12
MAXIMUM_OBSERVATION_COST = 24
MAXIMUM_SCORE_EVALUATIONS = 100_000
ADEQUATE_ADMITTED_SET_JACCARD = 0.80
FAILED_OUTCOME = "MEASUREMENT_FAILED"


@dataclass(frozen=True, slots=True)
class StructuralHypothesis:
    """One member of the frozen eight-axis structural version space."""

    delivery_observed: int
    receiver_closed: int
    action_rank_full: int
    regime_changes: int
    topology_periodic: int
    hidden_reservoir: int
    topology_factorization: int
    interface_obstruction: int

    def __post_init__(self) -> None:
        if any(value not in (0, 1) for value in self.bits):
            raise ValueError("structural axes must be binary")

    @property
    def bits(self) -> tuple[int, ...]:
        return (
            self.delivery_observed,
            self.receiver_closed,
            self.action_rank_full,
            self.regime_changes,
            self.topology_periodic,
            self.hidden_reservoir,
            self.topology_factorization,
            self.interface_obstruction,
        )

    @property
    def hypothesis_id(self) -> str:
        encoded = "".join(str(value) for value in self.bits)
        return f"hypothesis.structural-eight-axis.{encoded}"

    def to_document(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "axes": tuple((axis_id, value) for axis_id, value in zip(AXIS_IDS, self.bits)),
        }


@dataclass(frozen=True, slots=True)
class DiscoveryWorld:
    """One independently prepared truth-known structural world."""

    world_id: str
    split: str
    truth: StructuralHypothesis
    wrong_prior: bool
    prior_preferred_hypothesis_id: str | None
    topology_shift: bool
    failed_experiment_id: str | None
    preparation_seed: int
    nuisance_scale: float

    def __post_init__(self) -> None:
        _frozen_world_randomization_coordinate(self.world_id)
        if self.split not in {"DEVELOPMENT", "EVALUATION"} or not self.world_id.startswith(
            "world.structural-eight-axis." + self.split.lower() + "-"
        ):
            raise ValueError("response discovery world identity differs from its split")

    def to_document(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "split": self.split,
            "truth": self.truth.to_document(),
            "wrong_prior": self.wrong_prior,
            "prior_preferred_hypothesis_id": self.prior_preferred_hypothesis_id,
            "topology_shift": self.topology_shift,
            "failed_experiment_id": self.failed_experiment_id,
            "preparation_seed": self.preparation_seed,
            "nuisance_scale": self.nuisance_scale,
            "independent_unit": "complete-generated-open-system-world",
        }


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """A bounded declarative experiment; no executable action is embedded."""

    experiment_id: str
    measurement_family: str
    cost_units: int
    measured_axes: tuple[str, ...]
    requires_observed_delivery: bool = False
    requires_closed_receiver: bool = False
    requires_absent_reservoir: bool = False
    topology_sensitive: bool = False
    optimization_utility: float = 0.0

    def to_document(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "measurement_family": self.measurement_family,
            "cost_units": self.cost_units,
            "measured_axes": self.measured_axes,
            "requires_observed_delivery": self.requires_observed_delivery,
            "requires_closed_receiver": self.requires_closed_receiver,
            "requires_absent_reservoir": self.requires_absent_reservoir,
            "topology_sensitive": self.topology_sensitive,
            "optimization_utility": self.optimization_utility,
            "authority_class": "NON_ACTUATING_TRUTH_KNOWN_GENERATED_WORLD",
            "arbitrary_code_or_action_permitted": False,
        }


EXPERIMENT_LIBRARY = (
    ExperimentSpec(
        "experiment.delivery-audit",
        "delivery",
        1,
        ("delivery_observed",),
        optimization_utility=0.5,
    ),
    ExperimentSpec(
        "experiment.closure-reservoir-decay",
        "passive-state",
        2,
        ("receiver_closed", "hidden_reservoir"),
        optimization_utility=1.0,
    ),
    ExperimentSpec(
        "experiment.ambient-clock-exchange",
        "clock",
        1,
        ("regime_changes",),
        optimization_utility=1.5,
    ),
    ExperimentSpec(
        "experiment.passive-boundary-spectrum",
        "topology",
        3,
        ("topology_periodic",),
        topology_sensitive=True,
        optimization_utility=1.0,
    ),
    ExperimentSpec(
        "experiment.passive-interface-decay",
        "transport",
        2,
        ("interface_obstruction",),
        topology_sensitive=True,
        optimization_utility=1.0,
    ),
    ExperimentSpec(
        "experiment.signed-port-image",
        "action-image",
        2,
        ("action_rank_full",),
        requires_observed_delivery=True,
        optimization_utility=3.0,
    ),
    ExperimentSpec(
        "experiment.action-topology-exchange",
        "factorization",
        2,
        ("topology_periodic", "topology_factorization"),
        requires_observed_delivery=True,
        requires_closed_receiver=True,
        topology_sensitive=True,
        optimization_utility=2.0,
    ),
    ExperimentSpec(
        "experiment.commutator-transport",
        "composition",
        3,
        ("regime_changes", "topology_factorization", "interface_obstruction"),
        requires_observed_delivery=True,
        requires_closed_receiver=True,
        topology_sensitive=True,
        optimization_utility=2.5,
    ),
    ExperimentSpec(
        "experiment.admission-scan",
        "admission",
        3,
        ("action_rank_full", "hidden_reservoir", "interface_obstruction"),
        requires_observed_delivery=True,
        optimization_utility=5.0,
    ),
    ExperimentSpec(
        "experiment.low-load-reservoir",
        "action-image",
        2,
        ("hidden_reservoir", "action_rank_full"),
        requires_observed_delivery=True,
        optimization_utility=4.0,
    ),
    ExperimentSpec(
        "experiment.high-load-interface",
        "stress",
        4,
        ("topology_periodic", "interface_obstruction", "action_rank_full"),
        requires_observed_delivery=True,
        requires_closed_receiver=True,
        requires_absent_reservoir=True,
        topology_sensitive=True,
        optimization_utility=6.0,
    ),
    ExperimentSpec(
        "experiment.cut-glue-exchange",
        "transport",
        3,
        ("topology_periodic", "interface_obstruction"),
        requires_observed_delivery=True,
        requires_closed_receiver=True,
        requires_absent_reservoir=True,
        topology_sensitive=True,
        optimization_utility=2.0,
    ),
)
EXPERIMENT_BY_ID = {value.experiment_id: value for value in EXPERIMENT_LIBRARY}


def structural_hypotheses() -> tuple[StructuralHypothesis, ...]:
    """Return the frozen 2**8 candidate family in stable identifier order."""

    values = tuple(StructuralHypothesis(*bits) for bits in product((0, 1), repeat=8))
    return tuple(sorted(values, key=lambda value: value.hypothesis_id))


def hypothesis_from_document(value: Mapping[str, Any]) -> StructuralHypothesis:
    axes = value.get("axes")
    if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
        raise ValueError("Response discovery hypothesis axes are absent")
    decoded = {str(row[0]): int(row[1]) for row in axes}
    if tuple(sorted(decoded)) != tuple(sorted(AXIS_IDS)):
        raise ValueError("Response discovery hypothesis axes differ")
    result = StructuralHypothesis(*(decoded[axis_id] for axis_id in AXIS_IDS))
    if value.get("hypothesis_id") != result.hypothesis_id:
        raise ValueError("Response discovery hypothesis identifier differs from axes")
    return result


def world_from_document(value: Mapping[str, Any]) -> DiscoveryWorld:
    truth = value.get("truth")
    if not isinstance(truth, Mapping):
        raise ValueError("Response discovery world truth is absent")
    failed = value.get("failed_experiment_id")
    preferred = value.get("prior_preferred_hypothesis_id")
    return DiscoveryWorld(
        world_id=str(value["world_id"]),
        split=str(value["split"]),
        truth=hypothesis_from_document(truth),
        wrong_prior=bool(value["wrong_prior"]),
        prior_preferred_hypothesis_id=None if preferred is None else str(preferred),
        topology_shift=bool(value["topology_shift"]),
        failed_experiment_id=None if failed is None else str(failed),
        preparation_seed=int(value["preparation_seed"]),
        nuisance_scale=float(value["nuisance_scale"]),
    )


def _exact_flags(count: int, ones: int, rng: np.random.Generator) -> list[int]:
    if not 0 <= ones <= count:
        raise ValueError("invalid exact binary allocation")
    values = np.asarray([1] * ones + [0] * (count - ones), dtype=np.int64)
    rng.shuffle(values)
    return [int(value) for value in values]


def generate_discovery_worlds(
    *,
    split: str,
    independent_world_count: int,
    seed: int,
    wrong_prior_fraction: float,
    hidden_delivery_fraction: float,
    receiver_coarsening_fraction: float,
    topology_periodic_fraction: float,
) -> tuple[DiscoveryWorld, ...]:
    """Generate disjoint complete preparations with exact robustness fractions."""

    if split not in {"DEVELOPMENT", "EVALUATION"}:
        raise ValueError("Response discovery split differs")
    if independent_world_count < 8:
        raise ValueError("Response discovery needs at least eight independent worlds")
    rng = np.random.default_rng(seed)
    hidden_count = round(independent_world_count * hidden_delivery_fraction)
    coarse_count = round(independent_world_count * receiver_coarsening_fraction)
    periodic_count = round(independent_world_count * topology_periodic_fraction)
    wrong_count = round(independent_world_count * wrong_prior_fraction)
    delivery = [1 - value for value in _exact_flags(independent_world_count, hidden_count, rng)]
    receiver = [1 - value for value in _exact_flags(independent_world_count, coarse_count, rng)]
    topology = _exact_flags(independent_world_count, periodic_count, rng)
    wrong = _exact_flags(independent_world_count, wrong_count, rng)
    failed_flags = _exact_flags(independent_world_count, round(independent_world_count / 6), rng)
    remaining_patterns = tuple(product((0, 1), repeat=5))
    used: set[tuple[int, ...]] = set()
    truths: list[StructuralHypothesis] = []
    for index in range(independent_world_count):
        order = rng.permutation(len(remaining_patterns))
        chosen: tuple[int, ...] | None = None
        for raw_pattern_index in order:
            pattern = remaining_patterns[int(raw_pattern_index)]
            bits = (
                delivery[index],
                receiver[index],
                pattern[0],
                pattern[1],
                topology[index],
                pattern[2],
                pattern[3],
                pattern[4],
            )
            if bits not in used:
                chosen = bits
                break
        if chosen is None:
            raise ValueError("Response discovery could not allocate a unique truth-known class")
        used.add(chosen)
        truths.append(StructuralHypothesis(*chosen))
    failure_candidates = (
        "experiment.ambient-clock-exchange",
        "experiment.closure-reservoir-decay",
        "experiment.delivery-audit",
        "experiment.passive-boundary-spectrum",
        "experiment.passive-interface-decay",
    )
    worlds = []
    for index, truth in enumerate(truths):
        opposite = StructuralHypothesis(*(1 - value for value in truth.bits))
        failed_id = (
            failure_candidates[index % len(failure_candidates)] if failed_flags[index] else None
        )
        worlds.append(
            DiscoveryWorld(
                world_id=f"world.structural-eight-axis.{split.lower()}-{index + 1:03d}",
                split=split,
                truth=truth,
                wrong_prior=bool(wrong[index]),
                prior_preferred_hypothesis_id=(
                    opposite.hypothesis_id if wrong[index] else None
                ),
                topology_shift=bool(truth.topology_periodic),
                failed_experiment_id=failed_id,
                preparation_seed=seed * 10_000 + index + 1,
                nuisance_scale=float(rng.uniform(0.85, 1.15)),
            )
        )
    return tuple(worlds)


def admitted_actions(hypothesis: StructuralHypothesis) -> tuple[str, ...]:
    """Truth-known receiver-admissible action set for one structural class."""

    actions = ["hold"]
    if not hypothesis.hidden_reservoir or not hypothesis.regime_changes:
        actions.append("port-a")
    if hypothesis.action_rank_full and hypothesis.receiver_closed:
        actions.append("port-b")
    if (
        hypothesis.action_rank_full
        and not hypothesis.hidden_reservoir
        and not hypothesis.interface_obstruction
        and not hypothesis.regime_changes
    ):
        actions.append("joint-ab")
    if (
        hypothesis.topology_periodic
        and hypothesis.topology_factorization
        and not hypothesis.hidden_reservoir
    ):
        actions.append("boundary-loop")
    return tuple(sorted(actions))


def obstruction_location(hypothesis: StructuralHypothesis) -> str:
    if not hypothesis.interface_obstruction:
        return "NONE"
    if hypothesis.topology_periodic:
        return "PERIODIC_SEAM_07_00"
    return "OPEN_INTERFACE_03_04"


def predicted_outcome(
    hypothesis: StructuralHypothesis,
    experiment: ExperimentSpec,
) -> str:
    """Exact truth-known response symbol under the declared observation chart."""

    experiment_id = experiment.experiment_id
    if experiment_id == "experiment.delivery-audit":
        return "DELIVERY_OBSERVED" if hypothesis.delivery_observed else "DELIVERY_HIDDEN"
    if experiment_id == "experiment.closure-reservoir-decay":
        receiver = "CLOSED" if hypothesis.receiver_closed else "COARSE"
        reservoir = "RESERVOIR" if hypothesis.hidden_reservoir else "NO_RESERVOIR"
        return f"{receiver}|{reservoir}"
    if experiment_id == "experiment.ambient-clock-exchange":
        return "REGIME_CHANGE" if hypothesis.regime_changes else "STATIONARY"
    if experiment_id == "experiment.passive-boundary-spectrum":
        if not hypothesis.receiver_closed:
            return "RECEIVER_ALIASED"
        return "PERIODIC" if hypothesis.topology_periodic else "OPEN"
    if experiment_id == "experiment.passive-interface-decay":
        if not hypothesis.receiver_closed:
            return "RECEIVER_ALIASED"
        return "OBSTRUCTION" if hypothesis.interface_obstruction else "GLUES"
    if experiment_id == "experiment.signed-port-image":
        return "RANK_TWO" if hypothesis.action_rank_full else "RANK_ONE_KERNEL_B"
    if experiment_id == "experiment.action-topology-exchange":
        topology = "PERIODIC" if hypothesis.topology_periodic else "OPEN"
        factor = "TOPOLOGY" if hypothesis.topology_factorization else "ACTION"
        return f"{topology}|{factor}"
    if experiment_id == "experiment.commutator-transport":
        regime = "CHANGING" if hypothesis.regime_changes else "STATIONARY"
        factor = "TOPOLOGY" if hypothesis.topology_factorization else "ACTION"
        gluing = "OBSTRUCTED" if hypothesis.interface_obstruction else "GLUES"
        return f"{regime}|{factor}|{gluing}"
    if experiment_id == "experiment.admission-scan":
        return "ADMITTED:" + ",".join(admitted_actions(hypothesis))
    if experiment_id == "experiment.low-load-reservoir":
        reservoir = "RESERVOIR" if hypothesis.hidden_reservoir else "NO_RESERVOIR"
        rank = "RANK_TWO" if hypothesis.action_rank_full else "RANK_ONE"
        return f"{reservoir}|{rank}"
    if experiment_id == "experiment.high-load-interface":
        topology = "PERIODIC" if hypothesis.topology_periodic else "OPEN"
        gluing = "OBSTRUCTED" if hypothesis.interface_obstruction else "GLUES"
        rank = "RANK_TWO" if hypothesis.action_rank_full else "RANK_ONE"
        return f"{topology}|{gluing}|{rank}"
    if experiment_id == "experiment.cut-glue-exchange":
        topology = "PERIODIC" if hypothesis.topology_periodic else "OPEN"
        gluing = "OBSTRUCTED" if hypothesis.interface_obstruction else "GLUES"
        return f"{topology}|{gluing}"
    raise ValueError(f'unknown response discovery experiment {experiment_id}')


def initial_posterior(
    hypotheses: Sequence[StructuralHypothesis],
    world: DiscoveryWorld,
) -> FloatArray:
    """Return a full-support prior, adversarially misspecified where declared."""

    if not world.wrong_prior:
        return np.full(len(hypotheses), 1.0 / len(hypotheses), dtype=np.float64)
    if world.prior_preferred_hypothesis_id is None:
        raise ValueError("wrong-prior world lacks a declared preferred alternative")
    preferred = next(
        value for value in hypotheses if value.hypothesis_id == world.prior_preferred_hypothesis_id
    )
    weights = np.ones(len(hypotheses), dtype=np.float64)
    for index, hypothesis in enumerate(hypotheses):
        weight = 1.0
        for actual, preferred_value in zip(hypothesis.bits, preferred.bits):
            weight *= 0.75 if actual == preferred_value else 0.25
        weights[index] = weight
    weights /= np.sum(weights)
    return weights


def posterior_entropy(weights: FloatArray) -> float:
    positive = weights[weights > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def update_posterior(
    hypotheses: Sequence[StructuralHypothesis],
    weights: FloatArray,
    experiment: ExperimentSpec,
    observed_outcome: str,
) -> FloatArray:
    """Apply one exact symbolic observation; failed acts carry no evidence."""

    if observed_outcome == FAILED_OUTCOME:
        return weights.copy()
    updated = np.asarray(
        [
            weight if predicted_outcome(hypothesis, experiment) == observed_outcome else 0.0
            for hypothesis, weight in zip(hypotheses, weights)
        ],
        dtype=np.float64,
    )
    total = float(np.sum(updated))
    if total <= 0.0:
        raise ValueError("Response discovery observation eliminated the complete version space")
    return updated / total


def _support(hypotheses: Sequence[StructuralHypothesis], weights: FloatArray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.flatnonzero(weights > 1e-15))


def _partition_probabilities(
    hypotheses: Sequence[StructuralHypothesis],
    weights: FloatArray,
    experiment: ExperimentSpec,
) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for hypothesis, weight in zip(hypotheses, weights):
        if weight <= 0.0:
            continue
        outcome = predicted_outcome(hypothesis, experiment)
        probabilities[outcome] = probabilities.get(outcome, 0.0) + float(weight)
    return dict(sorted(probabilities.items()))


def expected_information_gain(
    hypotheses: Sequence[StructuralHypothesis],
    weights: FloatArray,
    experiment: ExperimentSpec,
) -> float:
    probabilities = np.asarray(
        tuple(_partition_probabilities(hypotheses, weights, experiment).values()),
        dtype=np.float64,
    )
    positive = probabilities[probabilities > 0.0]
    return float(-np.sum(positive * np.log2(positive)))


def _requirement_status(
    experiment: ExperimentSpec,
    hypotheses: Sequence[StructuralHypothesis],
    support: Sequence[int],
) -> str | None:
    candidates = tuple(hypotheses[index] for index in support)
    if experiment.requires_observed_delivery and any(
        not value.delivery_observed for value in candidates
    ):
        return "DELIVERY_NOT_ESTABLISHED_FOR_ALL_ALTERNATIVES"
    if experiment.requires_closed_receiver and any(
        not value.receiver_closed for value in candidates
    ):
        return "RECEIVER_CLOSURE_NOT_ESTABLISHED_FOR_ALL_ALTERNATIVES"
    if experiment.requires_absent_reservoir and any(
        value.hidden_reservoir for value in candidates
    ):
        return "RESERVOIR_ABSENCE_NOT_ESTABLISHED_FOR_ALL_ALTERNATIVES"
    return None


def eligible_experiments(
    *,
    method_id: str,
    hypotheses: Sequence[StructuralHypothesis],
    weights: FloatArray,
    executed_experiment_ids: Sequence[str],
    remaining_cost: int,
) -> tuple[tuple[ExperimentSpec, ...], tuple[dict[str, str], ...]]:
    """Apply noncompensating authority, safety, support, and validity filters."""

    if method_id not in METHOD_IDS:
        raise ValueError("unknown response discovery method")
    support = _support(hypotheses, weights)
    executed = set(executed_experiment_ids)
    eligible = []
    rejected = []
    for experiment in EXPERIMENT_LIBRARY:
        reason: str | None = None
        if experiment.experiment_id in executed:
            reason = "ALREADY_EXECUTED"
        elif experiment.cost_units > remaining_cost:
            reason = "OBSERVATION_COST_BUDGET"
        elif method_id == "topology_blind_response" and experiment.topology_sensitive:
            reason = "METHOD_DECLARED_TOPOLOGY_BLIND"
        else:
            reason = _requirement_status(experiment, hypotheses, support)
        if reason is None:
            outcomes = {
                predicted_outcome(hypotheses[index], experiment) for index in support
            }
            if len(outcomes) <= 1:
                reason = "NO_REMAINING_DISCRIMINATION"
        if reason is None:
            eligible.append(experiment)
        else:
            rejected.append({"experiment_id": experiment.experiment_id, "reason_code": reason})
    return tuple(eligible), tuple(rejected)


def _frozen_world_randomization_coordinate(world_id: str) -> str:
    """Recover the original scientific KDF coordinate from a current world ID."""

    prefix = "world.structural-eight-axis."
    if not world_id.startswith(prefix):
        raise ValueError("response discovery randomization needs a current world identity")
    coordinate = world_id.removeprefix(prefix)
    split, separator, ordinal = coordinate.partition("-")
    if (
        split not in {"development", "evaluation"}
        or separator != "-"
        or not ordinal
        or any(value not in "0123456789" for value in ordinal)
        or int(ordinal) < 1
        or format(int(ordinal), "03d") != ordinal
    ):
        raise ValueError("response discovery world identity changes its split or ordinal")
    # Exact original selected scientific hash coordinate; never a current alias.
    return "world.rf8." + coordinate


def _stable_random_score(method_id: str, world_id: str, step: int, experiment_id: str) -> float:
    scientific_world_id = _frozen_world_randomization_coordinate(world_id)
    payload = f"{method_id}|{scientific_world_id}|{step}|{experiment_id}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big") / float(2**64)


def _selection_scores(
    *,
    method_id: str,
    world: DiscoveryWorld,
    hypotheses: Sequence[StructuralHypothesis],
    weights: FloatArray,
    eligible: Sequence[ExperimentSpec],
    executed_experiment_ids: Sequence[str],
    step: int,
) -> tuple[dict[str, float], int]:
    if method_id == "oracle_upper_bound":
        scores = {}
        entropy_before = posterior_entropy(weights)
        for experiment in eligible:
            observed = (
                FAILED_OUTCOME
                if experiment.experiment_id == world.failed_experiment_id
                else predicted_outcome(world.truth, experiment)
            )
            after = update_posterior(hypotheses, weights, experiment, observed)
            scores[experiment.experiment_id] = entropy_before - posterior_entropy(after)
        return scores, len(eligible) * len(hypotheses)
    if method_id in {"response_version_space", "topology_blind_response"}:
        return (
            {
                experiment.experiment_id: expected_information_gain(
                    hypotheses, weights, experiment
                )
                for experiment in eligible
            },
            len(eligible) * len(hypotheses),
        )
    if method_id == "uncertainty_sampling":
        scores = {}
        for experiment in eligible:
            probabilities = _partition_probabilities(hypotheses, weights, experiment)
            largest = max(probabilities.values())
            scores[experiment.experiment_id] = largest * (1.0 - largest)
        return scores, len(eligible) * len(hypotheses)
    if method_id == "bayesian_optimization":
        return (
            {
                experiment.experiment_id: experiment.optimization_utility
                / experiment.cost_units
                for experiment in eligible
            },
            len(eligible),
        )
    if method_id == "active_system_identification":
        family_priority = {
            "delivery": 20.0,
            "action-image": 12.0,
            "clock": 11.0,
            "factorization": 10.0,
            "passive-state": 9.0,
            "composition": 8.0,
            "topology": 7.0,
            "transport": 6.0,
            "admission": 3.0,
            "stress": 2.0,
        }
        return (
            {
                experiment.experiment_id: family_priority[experiment.measurement_family]
                + 0.01 * expected_information_gain(hypotheses, weights, experiment)
                for experiment in eligible
            },
            len(eligible) * len(hypotheses),
        )
    if method_id == "random_space_filling":
        family_counts: dict[str, int] = {}
        for experiment_id in executed_experiment_ids:
            family = EXPERIMENT_BY_ID[experiment_id].measurement_family
            family_counts[family] = family_counts.get(family, 0) + 1
        return (
            {
                experiment.experiment_id: -float(
                    family_counts.get(experiment.measurement_family, 0)
                )
                + 0.25
                * _stable_random_score(method_id, world.world_id, step, experiment.experiment_id)
                for experiment in eligible
            },
            len(eligible),
        )
    raise ValueError("unknown response discovery selection method")


def _consensus_value(values: Sequence[tuple[str, ...] | str]) -> tuple[str, ...] | str | None:
    if not values:
        return None
    first = values[0]
    return first if all(value == first for value in values[1:]) else None


def _posterior_document(
    hypotheses: Sequence[StructuralHypothesis],
    weights: FloatArray,
) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "hypothesis_id": hypothesis.hypothesis_id,
            "probability": float(weight),
        }
        for hypothesis, weight in zip(hypotheses, weights)
        if weight > 1e-15
    )


def run_discovery_method(
    *,
    world: DiscoveryWorld,
    method_id: str,
    maximum_acts: int = MAXIMUM_ACTS,
    maximum_observation_cost: int = MAXIMUM_OBSERVATION_COST,
    maximum_score_evaluations: int = MAXIMUM_SCORE_EVALUATIONS,
) -> dict[str, Any]:
    """Run one method in one world while preserving every selection decision."""

    if method_id not in METHOD_IDS:
        raise ValueError("unknown response discovery method")
    hypotheses = structural_hypotheses()
    weights = initial_posterior(hypotheses, world)
    executed: list[str] = []
    decisions: list[dict[str, Any]] = []
    cumulative_cost = 0
    score_evaluations = 0
    failed_acts = 0
    invalid_queries = 0
    class_act: int | None = None
    admission_act: int | None = None
    obstruction_act: int | None = None
    stop_reason = "ACT_BUDGET_EXHAUSTED"
    for step in range(1, maximum_acts + 1):
        support_before = _support(hypotheses, weights)
        if len(support_before) == 1:
            stop_reason = "STRUCTURAL_CLASS_RESOLVED"
            break
        remaining_cost = maximum_observation_cost - cumulative_cost
        eligible, rejected = eligible_experiments(
            method_id=method_id,
            hypotheses=hypotheses,
            weights=weights,
            executed_experiment_ids=executed,
            remaining_cost=remaining_cost,
        )
        if not eligible:
            stop_reason = "OBSERVATIONALLY_NONIDENTIFIABLE"
            break
        started = perf_counter_ns()
        scores, evaluations = _selection_scores(
            method_id=method_id,
            world=world,
            hypotheses=hypotheses,
            weights=weights,
            eligible=eligible,
            executed_experiment_ids=executed,
            step=step,
        )
        elapsed = perf_counter_ns() - started
        score_evaluations += evaluations
        if score_evaluations > maximum_score_evaluations:
            stop_reason = "COMPUTE_BUDGET_EXHAUSTED"
            break
        maximum_score = max(scores.values())
        tied = tuple(
            sorted(
                experiment_id
                for experiment_id, score in scores.items()
                if math.isclose(score, maximum_score, rel_tol=0.0, abs_tol=1e-12)
            )
        )
        selected_id = tied[0]
        experiment = EXPERIMENT_BY_ID[selected_id]
        requirement = _requirement_status(experiment, hypotheses, support_before)
        if requirement is not None:
            invalid_queries += 1
            raise ValueError("Response discovery hard filters admitted an invalid experiment")
        observed = (
            FAILED_OUTCOME
            if selected_id == world.failed_experiment_id
            else predicted_outcome(world.truth, experiment)
        )
        if observed == FAILED_OUTCOME:
            failed_acts += 1
        weights_after = update_posterior(hypotheses, weights, experiment, observed)
        support_after = _support(hypotheses, weights_after)
        cumulative_cost += experiment.cost_units
        executed.append(selected_id)
        admissions = [admitted_actions(hypotheses[index]) for index in support_after]
        obstructions = [obstruction_location(hypotheses[index]) for index in support_after]
        admission_consensus = _consensus_value(admissions)
        obstruction_consensus = _consensus_value(obstructions)
        if len(support_after) == 1 and class_act is None:
            class_act = step
        if admission_consensus is not None and admission_act is None:
            admission_act = step
        if obstruction_consensus is not None and obstruction_act is None:
            obstruction_act = step
        decisions.append(
            {
                "step": step,
                "experiment_id": selected_id,
                "measurement_family": experiment.measurement_family,
                "requested_experiment_id": selected_id,
                "accepted_experiment_id": selected_id,
                "applied_experiment_id": selected_id,
                "realized_experiment_id": selected_id,
                "delivery_chain_exact": True,
                "observed_outcome": observed,
                "measurement_failed": observed == FAILED_OUTCOME,
                "cost_units": experiment.cost_units,
                "cumulative_cost_units": cumulative_cost,
                "version_size_before": len(support_before),
                "version_size_after": len(support_after),
                "posterior_entropy_before": posterior_entropy(weights),
                "posterior_entropy_after": posterior_entropy(weights_after),
                "predictive_outcome_probabilities": tuple(
                    _partition_probabilities(hypotheses, weights, experiment).items()
                ),
                "candidate_scores": tuple(sorted(scores.items())),
                "tie_candidate_ids": tied,
                "tie_rule": "LEXICOGRAPHIC_EXPERIMENT_ID",
                "rejected_candidates": rejected,
                "selection_compute_nanoseconds": elapsed,
                "selection_score_evaluations": evaluations,
                "admitted_set_consensus": admission_consensus,
                "obstruction_consensus": obstruction_consensus,
            }
        )
        weights = weights_after
    else:
        if len(_support(hypotheses, weights)) == 1:
            stop_reason = "STRUCTURAL_CLASS_RESOLVED"

    support_final = _support(hypotheses, weights)
    if len(support_final) == 1:
        stop_reason = "STRUCTURAL_CLASS_RESOLVED"
    final_eligible, final_rejected = eligible_experiments(
        method_id="response_version_space",
        hypotheses=hypotheses,
        weights=weights,
        executed_experiment_ids=executed,
        remaining_cost=maximum_observation_cost - cumulative_cost,
    )
    next_falsifier: str | None = None
    if final_eligible:
        final_scores, _ = _selection_scores(
            method_id="response_version_space",
            world=world,
            hypotheses=hypotheses,
            weights=weights,
            eligible=final_eligible,
            executed_experiment_ids=executed,
            step=len(executed) + 1,
        )
        best = max(final_scores.values())
        next_falsifier = min(
            key for key, value in final_scores.items() if math.isclose(value, best, abs_tol=1e-12)
        )
    maximum_probability = float(np.max(weights))
    map_id = min(
        hypothesis.hypothesis_id
        for hypothesis, weight in zip(hypotheses, weights)
        if math.isclose(float(weight), maximum_probability, rel_tol=0.0, abs_tol=1e-15)
    )
    final_admissions = [admitted_actions(hypotheses[index]) for index in support_final]
    final_obstructions = [obstruction_location(hypotheses[index]) for index in support_final]
    return {
        "world_id": world.world_id,
        "method_id": method_id,
        "deployable": method_id != "oracle_upper_bound",
        "oracle_truth_access": method_id == "oracle_upper_bound",
        "maximum_acts": maximum_acts,
        "maximum_observation_cost": maximum_observation_cost,
        "maximum_score_evaluations": maximum_score_evaluations,
        "acts_used": len(executed),
        "observation_cost_used": cumulative_cost,
        "score_evaluations_used": score_evaluations,
        "selection_compute_nanoseconds": sum(
            int(value["selection_compute_nanoseconds"]) for value in decisions
        ),
        "failed_acts": failed_acts,
        "invalid_queries": invalid_queries,
        "stop_reason": stop_reason,
        "version_size_final": len(support_final),
        "map_hypothesis_id": map_id,
        "maximum_posterior_probability": maximum_probability,
        "acts_to_structural_class": class_act,
        "acts_to_admitted_set_consensus": admission_act,
        "acts_to_obstruction_consensus": obstruction_act,
        "admitted_set_consensus": _consensus_value(final_admissions),
        "obstruction_consensus": _consensus_value(final_obstructions),
        "next_falsifying_experiment_id": next_falsifier,
        "final_rejected_candidates": final_rejected,
        "final_posterior": _posterior_document(hypotheses, weights),
        "decisions": tuple(decisions),
    }


def run_discovery_benchmark(
    worlds: Sequence[DiscoveryWorld],
    *,
    methods: Sequence[str] = METHOD_IDS,
    maximum_acts: int = MAXIMUM_ACTS,
    maximum_observation_cost: int = MAXIMUM_OBSERVATION_COST,
    maximum_score_evaluations: int = MAXIMUM_SCORE_EVALUATIONS,
) -> tuple[dict[str, Any], ...]:
    if len({world.world_id for world in worlds}) != len(worlds):
        raise ValueError("Response discovery independent world identifiers are not unique")
    if len(set(methods)) != len(methods) or any(method not in METHOD_IDS for method in methods):
        raise ValueError("Response discovery method family differs")
    return tuple(
        run_discovery_method(
            world=world,
            method_id=method,
            maximum_acts=maximum_acts,
            maximum_observation_cost=maximum_observation_cost,
            maximum_score_evaluations=maximum_score_evaluations,
        )
        for world in worlds
        for method in methods
    )


def observable_equivalence_class(world: DiscoveryWorld) -> tuple[str, ...]:
    """Return the class left by every valid feasible nonfailed experiment."""

    hypotheses = structural_hypotheses()
    surviving = list(hypotheses)
    truth = world.truth
    for experiment in EXPERIMENT_LIBRARY:
        if experiment.requires_observed_delivery and not truth.delivery_observed:
            continue
        if experiment.requires_closed_receiver and not truth.receiver_closed:
            continue
        if experiment.requires_absent_reservoir and truth.hidden_reservoir:
            continue
        observed = predicted_outcome(truth, experiment)
        surviving = [
            hypothesis
            for hypothesis in surviving
            if predicted_outcome(hypothesis, experiment) == observed
        ]
    return tuple(value.hypothesis_id for value in surviving)


def _interval(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("Response discovery interval requires finite independent-world values")
    estimate, lower, upper = bootstrap_mean_interval(
        array,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {"estimate": estimate, "lower": lower, "upper": upper}


def _jaccard(left: Sequence[str], right: Sequence[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    return len(left_set & right_set) / len(left_set | right_set)


def _credible_contains_true(posterior: Sequence[Mapping[str, Any]], truth_id: str) -> bool:
    ordered = sorted(
        ((str(row["hypothesis_id"]), float(row["probability"])) for row in posterior),
        key=lambda row: (-row[1], row[0]),
    )
    cumulative = 0.0
    selected = set()
    for hypothesis_id, probability in ordered:
        selected.add(hypothesis_id)
        cumulative += probability
        if cumulative >= 0.90:
            break
    return truth_id in selected


def _world_metric_row(
    world: DiscoveryWorld,
    transcript: Mapping[str, Any],
    maximum_acts: int,
) -> dict[str, Any]:
    truth_id = world.truth.hypothesis_id
    posterior = transcript["final_posterior"]
    if not isinstance(posterior, Sequence):
        raise ValueError("Response discovery transcript posterior differs")
    probability_by_id = {
        str(row["hypothesis_id"]): float(row["probability"])
        for row in posterior
        if isinstance(row, Mapping)
    }
    true_probability = probability_by_id.get(truth_id, 0.0)
    squared_sum = sum(value * value for value in probability_by_id.values())
    brier = squared_sum - 2.0 * true_probability + 1.0
    map_id = str(transcript["map_hypothesis_id"])
    map_hypothesis = next(
        value for value in structural_hypotheses() if value.hypothesis_id == map_id
    )
    truth_admitted = admitted_actions(world.truth)
    map_admitted = admitted_actions(map_hypothesis)
    class_correct = (
        transcript["stop_reason"] == "STRUCTURAL_CLASS_RESOLVED" and map_id == truth_id
    )
    identifiable = len(observable_equivalence_class(world)) == 1
    admission_act = transcript["acts_to_admitted_set_consensus"]
    consensus = transcript["admitted_set_consensus"]
    consensus_correct = consensus is not None and tuple(consensus) == truth_admitted
    return {
        "world_id": world.world_id,
        "wrong_prior": world.wrong_prior,
        "receiver_coarsened": not bool(world.truth.receiver_closed),
        "topology_shift": world.topology_shift,
        "hidden_delivery": not bool(world.truth.delivery_observed),
        "theoretically_identifiable": identifiable,
        "class_resolved_correct": class_correct,
        "point_class_correct": map_id == truth_id,
        "false_promotion": (
            transcript["stop_reason"] == "STRUCTURAL_CLASS_RESOLVED" and map_id != truth_id
        ),
        "correct_typed_nonidentifiability": (
            not identifiable and transcript["stop_reason"] == "OBSERVATIONALLY_NONIDENTIFIABLE"
        ),
        "acts_to_class_censored": (
            int(transcript["acts_to_structural_class"])
            if class_correct
            else maximum_acts + 1
        ),
        "acts_to_admitted_censored": (
            int(admission_act) if admission_act is not None and consensus_correct else maximum_acts + 1
        ),
        "admitted_set_jaccard": _jaccard(map_admitted, truth_admitted),
        "admitted_set_adequate": _jaccard(map_admitted, truth_admitted)
        >= ADEQUATE_ADMITTED_SET_JACCARD,
        "obstruction_localized": obstruction_location(map_hypothesis)
        == obstruction_location(world.truth),
        "invalid_queries": int(transcript["invalid_queries"]),
        "failed_acts": int(transcript["failed_acts"]),
        "acts_used": int(transcript["acts_used"]),
        "observation_cost_used": int(transcript["observation_cost_used"]),
        "score_evaluations_used": int(transcript["score_evaluations_used"]),
        "selection_compute_nanoseconds": int(transcript["selection_compute_nanoseconds"]),
        "version_contraction_bits_per_act": (
            (8.0 - math.log2(int(transcript["version_size_final"])))
            / max(int(transcript["acts_used"]), 1)
        ),
        "true_posterior_probability": true_probability,
        "multiclass_brier_score": brier,
        "negative_log_probability": -math.log(max(true_probability, 1e-300)),
        "credible_90_contains_truth": _credible_contains_true(posterior, truth_id),
    }


def _summarize_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, Any]:
    metric_fields = (
        "class_resolved_correct",
        "point_class_correct",
        "false_promotion",
        "correct_typed_nonidentifiability",
        "acts_to_class_censored",
        "acts_to_admitted_censored",
        "admitted_set_jaccard",
        "admitted_set_adequate",
        "obstruction_localized",
        "invalid_queries",
        "failed_acts",
        "acts_used",
        "observation_cost_used",
        "score_evaluations_used",
        "selection_compute_nanoseconds",
        "version_contraction_bits_per_act",
        "true_posterior_probability",
        "multiclass_brier_score",
        "negative_log_probability",
        "credible_90_contains_truth",
    )
    return {
        "independent_world_count": len(rows),
        "identifiable_world_count": sum(bool(row["theoretically_identifiable"]) for row in rows),
        "nonidentifiable_world_count": sum(
            not bool(row["theoretically_identifiable"]) for row in rows
        ),
        "metrics": {
            field: _interval(
                [float(row[field]) for row in rows],
                seed=seed + index,
                resamples=resamples,
                confidence_level=confidence_level,
            )
            for index, field in enumerate(metric_fields)
        },
    }


def analyse_discovery_benchmark(
    *,
    worlds: Sequence[DiscoveryWorld],
    transcripts: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    maximum_acts: int,
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    """Adjudicate efficiency and correctness over independent worlds only."""

    world_by_id = {world.world_id: world for world in worlds}
    expected_pairs = {(world.world_id, method) for world in worlds for method in methods}
    actual_pairs = {
        (str(row["world_id"]), str(row["method_id"])) for row in transcripts
    }
    if actual_pairs != expected_pairs or len(transcripts) != len(expected_pairs):
        raise ValueError("Response discovery trajectory panel is incomplete or duplicated")
    rows_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    for transcript in transcripts:
        world_id = str(transcript["world_id"])
        method_id = str(transcript["method_id"])
        rows_by_method[method_id].append(
            _world_metric_row(world_by_id[world_id], transcript, maximum_acts)
        )
    for rows in rows_by_method.values():
        rows.sort(key=lambda row: str(row["world_id"]))
    method_results = {
        method: _summarize_rows(
            rows,
            seed=seed + 1000 * index,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        )
        for index, (method, rows) in enumerate(rows_by_method.items())
    }
    response_rows = rows_by_method["response_version_space"]
    comparators = tuple(
        method
        for method in methods
        if method not in {"response_version_space", "oracle_upper_bound"}
    )
    simultaneous_confidence = 1.0 - (1.0 - confidence_level) / max(len(comparators), 1)
    pairwise: list[dict[str, Any]] = []
    for index, method in enumerate(comparators):
        comparator_rows = rows_by_method[method]
        acts_difference = [
            float(comparator["acts_to_class_censored"])
            - float(response["acts_to_class_censored"])
            for response, comparator in zip(response_rows, comparator_rows)
        ]
        cost_difference = [
            float(comparator["observation_cost_used"])
            - float(response["observation_cost_used"])
            for response, comparator in zip(response_rows, comparator_rows)
        ]
        resolution_difference = [
            float(response["class_resolved_correct"])
            - float(comparator["class_resolved_correct"])
            for response, comparator in zip(response_rows, comparator_rows)
        ]
        pairwise.append(
            {
                "comparator_method_id": method,
                "censored_acts_saved_by_response": _interval(
                    acts_difference,
                    seed=seed + 20_000 + index,
                    resamples=bootstrap_resamples,
                    confidence_level=simultaneous_confidence,
                ),
                "observation_cost_saved_by_response": _interval(
                    cost_difference,
                    seed=seed + 21_000 + index,
                    resamples=bootstrap_resamples,
                    confidence_level=simultaneous_confidence,
                ),
                "correct_resolution_rate_difference": _interval(
                    resolution_difference,
                    seed=seed + 22_000 + index,
                    resamples=bootstrap_resamples,
                    confidence_level=simultaneous_confidence,
                ),
            }
        )
    oracle_rows = rows_by_method["oracle_upper_bound"]
    oracle_gap = _interval(
        [
            float(response["acts_to_class_censored"]) - float(oracle["acts_to_class_censored"])
            for response, oracle in zip(response_rows, oracle_rows)
        ],
        seed=seed + 30_000,
        resamples=bootstrap_resamples,
        confidence_level=confidence_level,
    )
    robustness = {}
    for index, flag in enumerate(
        ("wrong_prior", "receiver_coarsened", "topology_shift", "hidden_delivery")
    ):
        selected = [row for row in response_rows if bool(row[flag])]
        complement = [row for row in response_rows if not bool(row[flag])]
        robustness[flag] = {
            "exposed": _summarize_rows(
                selected,
                seed=seed + 40_000 + index * 100,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
            ),
            "unexposed": _summarize_rows(
                complement,
                seed=seed + 41_000 + index * 100,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
            ),
        }
    pair_by_id = {row["comparator_method_id"]: row for row in pairwise}
    primary_advantage = all(
        float(pair_by_id[method]["censored_acts_saved_by_response"]["lower"]) > 0.0
        for method in (
            "random_space_filling",
            "bayesian_optimization",
            "topology_blind_response",
        )
    )
    response_metrics = method_results["response_version_space"]["metrics"]
    structured_noninferiority = all(
        float(pair_by_id[method]["correct_resolution_rate_difference"]["lower"]) >= -0.05
        for method in ("uncertainty_sampling", "active_system_identification")
    )
    correctness_supported = (
        float(response_metrics["false_promotion"]["upper"]) == 0.0
        and float(response_metrics["invalid_queries"]["upper"]) == 0.0
        and all(
            bool(row["correct_typed_nonidentifiability"])
            for row in response_rows
            if not bool(row["theoretically_identifiable"])
        )
    )
    efficiency_supported = primary_advantage and structured_noninferiority
    return {
        "independent_evaluation_world_count": len(worlds),
        "nested_method_trajectory_count": len(transcripts),
        "independent_unit": "complete-generated-open-system-world",
        "algorithm_seeds_or_queries_treated_as_replicates": False,
        "candidate_structural_class_count": len(structural_hypotheses()),
        "declarative_experiment_count": len(EXPERIMENT_LIBRARY),
        "method_results": method_results,
        "pairwise_response_comparisons": tuple(pairwise),
        "simultaneous_pairwise_confidence_level": simultaneous_confidence,
        "oracle_gap_censored_acts": oracle_gap,
        "oracle_upper_bound_deployable": False,
        "robustness": robustness,
        "scientific_correctness_grade": "SUPPORTED" if correctness_supported else "MIXED",
        "discovery_efficiency_grade": "SUPPORTED" if efficiency_supported else "MIXED",
        "efficiency_supported": efficiency_supported,
        "correctness_supported": correctness_supported,
        "grading_axes_compensated": False,
        "claim_ceiling": "TRUTH_KNOWN_GENERATED_OPEN_SYSTEM_DISCOVERY_BENCHMARK",
        "reason_codes": tuple(
            reason
            for condition, reason in (
                (not primary_advantage, "PRIMARY_BASELINE_ADVANTAGE_NOT_SIMULTANEOUSLY_RESOLVED"),
                (not structured_noninferiority, "STRUCTURED_BASELINE_NONINFERIORITY_NOT_SHOWN"),
                (not correctness_supported, "SCIENTIFIC_CORRECTNESS_CRITERIA_NOT_ALL_MET"),
            )
            if condition
        ),
    }


def discovery_method_specification() -> dict[str, Any]:
    "Return the complete pre-freeze semantics for the response discovery child."

    return {
        "method_family_id": "method.response-version-space",
        "candidate_axes": AXIS_IDS,
        "candidate_structural_class_count": len(structural_hypotheses()),
        "experiment_library": tuple(value.to_document() for value in EXPERIMENT_LIBRARY),
        "method_ids": METHOD_IDS,
        "deployable_method_ids": DEPLOYABLE_METHOD_IDS,
        "oracle_method_id": "oracle_upper_bound",
        "oracle_deployable": False,
        "partition_score": "EXPECTED_POSTERIOR_INFORMATION_GAIN_BITS",
        "hard_filter_order": (
            "authority",
            "safety",
            "support",
            "validity",
            "remaining-discrimination",
        ),
        "maximum_acts": MAXIMUM_ACTS,
        "maximum_observation_cost": MAXIMUM_OBSERVATION_COST,
        "maximum_score_evaluations": MAXIMUM_SCORE_EVALUATIONS,
        "stopping_rules": (
            "STRUCTURAL_CLASS_RESOLVED",
            "OBSERVATIONALLY_NONIDENTIFIABLE",
            "ACT_BUDGET_EXHAUSTED",
            "OBSERVATION_COST_BUDGET_EXHAUSTED",
            "COMPUTE_BUDGET_EXHAUSTED",
        ),
        "tie_rule": "LEXICOGRAPHIC_EXPERIMENT_ID",
        "failed_act_rule": "CONSUME_ACT_AND_COST_WITH_NO_POSTERIOR_CONTRACTION",
        "multiplicity_rule": "BONFERRONI_SIMULTANEOUS_PAIRED_METHOD_INTERVALS",
        "adequate_admitted_set_jaccard": ADEQUATE_ADMITTED_SET_JACCARD,
        "primary_advantage_comparators": (
            "random_space_filling",
            "bayesian_optimization",
            "topology_blind_response",
        ),
        "structured_noninferiority_comparators": (
            "uncertainty_sampling",
            "active_system_identification",
        ),
        "independent_unit": "complete-generated-open-system-world",
        "nested_algorithm_seeds_are_replication": False,
        "arbitrary_generated_code_or_actions": False,
    }


__all__ = [
    "ADEQUATE_ADMITTED_SET_JACCARD",
    "AXIS_IDS",
    "DEPLOYABLE_METHOD_IDS",
    "DiscoveryWorld",
    "EXPERIMENT_LIBRARY",
    "FAILED_OUTCOME",
    "MAXIMUM_ACTS",
    "MAXIMUM_OBSERVATION_COST",
    "MAXIMUM_SCORE_EVALUATIONS",
    "METHOD_IDS",
    "StructuralHypothesis",
    "admitted_actions",
    "analyse_discovery_benchmark",
    "discovery_method_specification",
    "eligible_experiments",
    "expected_information_gain",
    "generate_discovery_worlds",
    "hypothesis_from_document",
    "initial_posterior",
    "observable_equivalence_class",
    "obstruction_location",
    "predicted_outcome",
    "run_discovery_benchmark",
    "run_discovery_method",
    "structural_hypotheses",
    "update_posterior",
    "world_from_document",
]
