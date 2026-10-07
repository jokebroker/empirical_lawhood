'Truth-known ICF-to-propulsion closure shadow for synthetic propulsion reference world.\n\nThe module does not simulate inertial-confinement fusion.  It implements a\nsmall, evaluator-known dependency/version-space world used to test whether an\nexperiment selector can expose confounded interfaces and repeated-cycle\nreliability requirements.  One complete generated target-and-machine\npreparation is the independent unit; experiments and cycle observations are\nnested views of that preparation.\n'

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .response_formalization import bootstrap_mean_interval


AXIS_NAMES = (
    "fabrication_defect",
    "delivery_bias",
    "coupling_asymmetry",
    "mix_memory",
    "diagnostic_bias",
    "weak_thermal_sink",
    "cycle_degradation",
    "repair_effective",
)
METHOD_IDS = (
    "response_version_space",
    "random_space_filling",
    "aggregate_yield_first",
    "oracle_upper_bound",
)
MAXIMUM_ACTS = 8
MAXIMUM_COST = 12
FAULT_CYCLE = 8
REPAIR_CYCLE = 13
CYCLE_COUNT = 24


@dataclass(frozen=True, slots=True)
class ClosureHypothesis:
    'One binary mechanism assignment in the synthetic propulsion reference world truth-known shadow.'

    fabrication_defect: int
    delivery_bias: int
    coupling_asymmetry: int
    mix_memory: int
    diagnostic_bias: int
    weak_thermal_sink: int
    cycle_degradation: int
    repair_effective: int

    def __post_init__(self) -> None:
        if any(value not in {0, 1} for value in self.values):
            raise ValueError('synthetic propulsion reference world hypothesis axes must be binary')

    @property
    def values(self) -> tuple[int, ...]:
        return tuple(int(getattr(self, name)) for name in AXIS_NAMES)

    @property
    def hypothesis_id(self) -> str:
        return "closure." + "".join(str(value) for value in self.values)

    def to_document(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            **{name: int(getattr(self, name)) for name in AXIS_NAMES},
        }


@dataclass(frozen=True, slots=True)
class InterfaceExperiment:
    experiment_id: str
    interface_id: str
    name: str
    cost: int
    evidence_role: str
    aggregate_endpoint: bool

    def __post_init__(self) -> None:
        if self.cost <= 0:
            raise ValueError('synthetic propulsion reference world experiment cost must be positive')

    def to_document(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "interface_id": self.interface_id,
            "name": self.name,
            "cost": self.cost,
            "evidence_role": self.evidence_role,
            "aggregate_endpoint": self.aggregate_endpoint,
        }


EXPERIMENTS = (
    InterfaceExperiment(
        "experiment.target-surface-metrology",
        "interface.fabrication-to-target-quality",
        "independent target-surface and defect metrology",
        1,
        "LOCAL_INTERFACE",
        False,
    ),
    InterfaceExperiment(
        "experiment.delivered-waveform-audit",
        "interface.driver-request-to-delivery",
        "requested-versus-delivered waveform audit",
        1,
        "DELIVERY_MAP",
        False,
    ),
    InterfaceExperiment(
        "experiment.diagnostic-crosscheck",
        "interface.state-to-diagnostic-receiver",
        "independent diagnostic validity cross-check",
        1,
        "RECEIVER_VALIDITY",
        False,
    ),
    InterfaceExperiment(
        "experiment.symmetry-scan",
        "interface.delivery-target-to-implosion",
        "delivery-conditioned symmetry scan",
        2,
        "LOCAL_INTERFACE",
        False,
    ),
    InterfaceExperiment(
        "experiment.pulse-pair-history",
        "interface.implosion-history-to-burn",
        "order-reversed pulse-pair history test",
        2,
        "MEMORY_AND_SINK",
        False,
    ),
    InterfaceExperiment(
        "experiment.accelerated-cycle-panel",
        "interface.repetition-to-degradation",
        "independent manufactured preparations under repeated cycles",
        3,
        "RELIABILITY_MAP",
        False,
    ),
    InterfaceExperiment(
        "experiment.fault-repair-requalification",
        "interface.fault-repair-to-preservation",
        "fault injection, repair, and independent requalification",
        3,
        "REPAIR_AND_PRESERVATION",
        False,
    ),
    InterfaceExperiment(
        "experiment.chamber-energy-balance",
        "interface.burn-to-chamber-sink",
        "chamber energy and thermal-sink balance",
        3,
        "SINK_MAP",
        False,
    ),
    InterfaceExperiment(
        "experiment.integrated-yield",
        "interface.target-driver-to-yield",
        "integrated single-shot yield",
        4,
        "AGGREGATE_ENDPOINT",
        True,
    ),
    InterfaceExperiment(
        "experiment.propulsion-output-proxy",
        "interface.chamber-to-propulsion-output",
        "integrated momentum-conversion proxy",
        5,
        "AGGREGATE_ENDPOINT",
        True,
    ),
)
EXPERIMENT_BY_ID = {value.experiment_id: value for value in EXPERIMENTS}


@dataclass(frozen=True, slots=True)
class ClosureWorld:
    """One independent generated target, machine, receiver, and repair preparation."""

    world_id: str
    split: str
    preparation_index: int
    preparation_seed: int
    truth: ClosureHypothesis
    initial_integrity: float
    degradation_rate: float
    fault_severity: float
    thermal_retention: float
    receiver_noise_sd: float

    def __post_init__(self) -> None:
        if self.split not in {"DEVELOPMENT", "EVALUATION"}:
            raise ValueError('synthetic propulsion reference world split differs')
        if not 0.7 <= self.initial_integrity <= 1.0:
            raise ValueError('synthetic propulsion reference world initial integrity is outside its generated support')
        if not 0.0 < self.degradation_rate < 0.04:
            raise ValueError('synthetic propulsion reference world degradation rate is outside its generated support')
        if not 0.05 < self.fault_severity < 0.25:
            raise ValueError('synthetic propulsion reference world fault severity is outside its generated support')
        if not 0.3 < self.thermal_retention < 1.0:
            raise ValueError('synthetic propulsion reference world thermal retention is outside its generated support')
        if not 0.0 < self.receiver_noise_sd < 0.02:
            raise ValueError('synthetic propulsion reference world receiver noise is outside its generated support')

    def to_document(self) -> dict[str, Any]:
        return {
            "world_id": self.world_id,
            "split": self.split,
            "preparation_index": self.preparation_index,
            "preparation_seed": self.preparation_seed,
            "truth": self.truth.to_document(),
            "initial_integrity": self.initial_integrity,
            "degradation_rate": self.degradation_rate,
            "fault_severity": self.fault_severity,
            "thermal_retention": self.thermal_retention,
            "receiver_noise_sd": self.receiver_noise_sd,
            "independent_unit": "complete-generated-target-machine-preparation",
            "cycles_and_experiment_views_are_nested": True,
        }


def hypothesis_from_document(value: Mapping[str, Any]) -> ClosureHypothesis:
    return ClosureHypothesis(**{name: int(value[name]) for name in AXIS_NAMES})


def world_from_document(value: Mapping[str, Any]) -> ClosureWorld:
    truth = value.get("truth")
    if not isinstance(truth, Mapping):
        raise ValueError('synthetic propulsion reference world world truth document differs')
    return ClosureWorld(
        world_id=str(value["world_id"]),
        split=str(value["split"]),
        preparation_index=int(value["preparation_index"]),
        preparation_seed=int(value["preparation_seed"]),
        truth=hypothesis_from_document(truth),
        initial_integrity=float(value["initial_integrity"]),
        degradation_rate=float(value["degradation_rate"]),
        fault_severity=float(value["fault_severity"]),
        thermal_retention=float(value["thermal_retention"]),
        receiver_noise_sd=float(value["receiver_noise_sd"]),
    )


def closure_hypotheses() -> tuple[ClosureHypothesis, ...]:
    return tuple(ClosureHypothesis(*values) for values in product((0, 1), repeat=8))


def predicted_outcome(
    hypothesis: ClosureHypothesis,
    experiment: InterfaceExperiment,
) -> str:
    """Return the exact evaluator-known outcome category for one interface test."""

    f, d, a, m, q, s, g, r = hypothesis.values
    experiment_id = experiment.experiment_id
    if experiment_id == "experiment.target-surface-metrology":
        value = f
    elif experiment_id == "experiment.delivered-waveform-audit":
        value = d
    elif experiment_id == "experiment.diagnostic-crosscheck":
        value = q
    elif experiment_id == "experiment.symmetry-scan":
        value = 2 * a + ((d + q) % 2)
    elif experiment_id == "experiment.pulse-pair-history":
        value = 2 * m + s
    elif experiment_id == "experiment.accelerated-cycle-panel":
        value = 4 * g + 2 * s + f
    elif experiment_id == "experiment.fault-repair-requalification":
        value = 0 if g == 0 else 1 + r
    elif experiment_id == "experiment.chamber-energy-balance":
        value = 2 * s + int(bool(d or m))
    elif experiment_id == "experiment.integrated-yield":
        value = min(3, f + d + a + m + q)
    elif experiment_id == "experiment.propulsion-output-proxy":
        upstream_loss = int((f + d + a + m + q) >= 2)
        reliability_loss = int(bool(s or g))
        repair_credit = int(bool(g and r))
        value = 2 * upstream_loss + reliability_loss - repair_credit
    else:
        raise ValueError(f"unknown synthetic propulsion reference world experiment: {experiment_id}")
    return f"outcome.{value}"


def observable_equivalence_class(
    truth: ClosureHypothesis,
    hypotheses: Sequence[ClosureHypothesis] | None = None,
) -> tuple[str, ...]:
    candidates = tuple(hypotheses or closure_hypotheses())
    truth_signature = tuple(predicted_outcome(truth, value) for value in EXPERIMENTS)
    return tuple(
        candidate.hypothesis_id
        for candidate in candidates
        if tuple(predicted_outcome(candidate, value) for value in EXPERIMENTS) == truth_signature
    )


def _partition_entropy(
    hypotheses: Sequence[ClosureHypothesis],
    experiment: InterfaceExperiment,
) -> float:
    if not hypotheses:
        raise ValueError('synthetic propulsion reference world partition requires hypotheses')
    counts: dict[str, int] = {}
    for hypothesis in hypotheses:
        outcome = predicted_outcome(hypothesis, experiment)
        counts[outcome] = counts.get(outcome, 0) + 1
    total = float(len(hypotheses))
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def rank_interfaces() -> tuple[dict[str, Any], ...]:
    """Rank tests only by expected structural collapse per declared cost."""

    hypotheses = closure_hypotheses()
    rows = []
    for experiment in EXPERIMENTS:
        bits = _partition_entropy(hypotheses, experiment)
        rows.append(
            {
                **experiment.to_document(),
                "prior_version_count": len(hypotheses),
                "expected_information_bits": bits,
                "information_bits_per_cost": bits / experiment.cost,
                "ranking_uses_nominal_yield": False,
                "ranking_uses_publicity": False,
            }
        )
    ordered = sorted(
        rows,
        key=lambda row: (
            -float(row["information_bits_per_cost"]),
            -float(row["expected_information_bits"]),
            str(row["experiment_id"]),
        ),
    )
    return tuple({**row, "rank": index + 1} for index, row in enumerate(ordered))


def generate_closure_worlds(
    *,
    split: str,
    independent_world_count: int,
    seed: int,
    world_id_prefix: str = 'world.synthetic-propulsion',
) -> tuple[ClosureWorld, ...]:
    if split not in {"DEVELOPMENT", "EVALUATION"}:
        raise ValueError('synthetic propulsion reference world split differs')
    if independent_world_count < 16 or independent_world_count > 256:
        raise ValueError('synthetic propulsion reference world world count is outside [16, 256]')
    if not world_id_prefix.startswith("world.") or not all(
        character.islower() or character.isdigit() or character in ".-"
        for character in world_id_prefix
    ):
        raise ValueError('synthetic propulsion reference world world ID prefix differs')
    rng = np.random.default_rng(seed)
    hypotheses = closure_hypotheses()
    selected = rng.choice(len(hypotheses), size=independent_world_count, replace=False)
    worlds = []
    for index, hypothesis_index in enumerate(selected):
        truth = hypotheses[int(hypothesis_index)]
        preparation_seed = seed * 10_000 + index
        initial = float(rng.uniform(0.88, 0.97) - 0.035 * truth.fabrication_defect)
        degradation = float(
            rng.uniform(0.010, 0.016) if truth.cycle_degradation else rng.uniform(0.001, 0.003)
        )
        retention = float(
            rng.uniform(0.88, 0.94) if truth.weak_thermal_sink else rng.uniform(0.48, 0.62)
        )
        worlds.append(
            ClosureWorld(
                world_id=f"{world_id_prefix}.{split.lower()}-{index + 1:03d}",
                split=split,
                preparation_index=index,
                preparation_seed=preparation_seed,
                truth=truth,
                initial_integrity=initial,
                degradation_rate=degradation,
                fault_severity=float(rng.uniform(0.10, 0.16)),
                thermal_retention=retention,
                receiver_noise_sd=float(rng.uniform(0.003, 0.008)),
            )
        )
    return tuple(worlds)


def _resolved(
    surviving: Sequence[ClosureHypothesis],
    equivalence_ids: set[str],
) -> bool:
    return {value.hypothesis_id for value in surviving} == equivalence_ids


def _select_experiment(
    *,
    method_id: str,
    surviving: Sequence[ClosureHypothesis],
    remaining: Sequence[InterfaceExperiment],
    truth: ClosureHypothesis,
    rng: np.random.Generator,
) -> InterfaceExperiment:
    if method_id == "random_space_filling":
        return remaining[int(rng.integers(0, len(remaining)))]
    if method_id == "aggregate_yield_first":
        order = (
            "experiment.integrated-yield",
            "experiment.propulsion-output-proxy",
            "experiment.chamber-energy-balance",
            "experiment.symmetry-scan",
            "experiment.target-surface-metrology",
            "experiment.delivered-waveform-audit",
            "experiment.diagnostic-crosscheck",
            "experiment.pulse-pair-history",
            "experiment.accelerated-cycle-panel",
            "experiment.fault-repair-requalification",
        )
        by_id = {value.experiment_id: value for value in remaining}
        return next(by_id[value] for value in order if value in by_id)

    def response_score(experiment: InterfaceExperiment) -> tuple[float, float, str]:
        information = _partition_entropy(surviving, experiment)
        return information / experiment.cost, information, experiment.experiment_id

    if method_id == "response_version_space":
        return max(remaining, key=response_score)
    if method_id == "oracle_upper_bound":
        scored = []
        for experiment in remaining:
            truth_outcome = predicted_outcome(truth, experiment)
            survivors = sum(
                predicted_outcome(value, experiment) == truth_outcome for value in surviving
            )
            scored.append(
                (
                    survivors / len(surviving),
                    experiment.cost,
                    experiment.experiment_id,
                    experiment,
                )
            )
        return min(scored, key=lambda value: value[:3])[3]
    raise ValueError(f"unknown synthetic propulsion reference world method: {method_id}")


def run_discovery_method(
    *,
    world: ClosureWorld,
    method_id: str,
    maximum_acts: int = MAXIMUM_ACTS,
    maximum_cost: int = MAXIMUM_COST,
) -> dict[str, Any]:
    if method_id not in METHOD_IDS:
        raise ValueError('synthetic propulsion reference world discovery method differs')
    hypotheses = closure_hypotheses()
    surviving = list(hypotheses)
    equivalence_ids = set(observable_equivalence_class(world.truth, hypotheses))
    executed: list[str] = []
    decisions = []
    cost_used = 0
    rng = np.random.default_rng(world.preparation_seed + METHOD_IDS.index(method_id) * 997)
    while len(executed) < maximum_acts and not _resolved(surviving, equivalence_ids):
        remaining = tuple(
            value
            for value in EXPERIMENTS
            if value.experiment_id not in executed and value.cost <= maximum_cost - cost_used
        )
        if not remaining:
            break
        selected = _select_experiment(
            method_id=method_id,
            surviving=surviving,
            remaining=remaining,
            truth=world.truth,
            rng=rng,
        )
        observed = predicted_outcome(world.truth, selected)
        before = len(surviving)
        surviving = [value for value in surviving if predicted_outcome(value, selected) == observed]
        cost_used += selected.cost
        executed.append(selected.experiment_id)
        decisions.append(
            {
                "act_index": len(executed),
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
    equivalence_resolved = _resolved(surviving, equivalence_ids)
    if equivalence_resolved and len(equivalence_ids) == 1:
        stop_reason = "STRUCTURAL_CLASS_RESOLVED"
    elif equivalence_resolved:
        stop_reason = "OBSERVATIONALLY_NONIDENTIFIABLE"
    elif cost_used >= maximum_cost:
        stop_reason = "COST_CENSORED"
    else:
        stop_reason = "ACT_CENSORED"
    truth_present = world.truth.hypothesis_id in {value.hypothesis_id for value in surviving}
    return {
        "world_id": world.world_id,
        "split": world.split,
        "method_id": method_id,
        "truth_hypothesis_id": world.truth.hypothesis_id,
        "true_observable_equivalence_size": len(equivalence_ids),
        "true_structurally_identifiable": len(equivalence_ids) == 1,
        "acts_used": len(executed),
        "cost_used": cost_used,
        "version_size_final": len(surviving),
        "version_collapse_fraction": 1.0 - len(surviving) / len(hypotheses),
        "equivalence_class_resolved": equivalence_resolved,
        "exact_truth_resolved": equivalence_resolved and len(equivalence_ids) == 1,
        "truth_retained": truth_present,
        "false_promotion": not truth_present,
        "stop_reason": stop_reason,
        "surviving_hypothesis_ids": tuple(value.hypothesis_id for value in surviving),
        "decisions": tuple(decisions),
        "oracle_deployable": method_id != "oracle_upper_bound",
    }


def run_discovery_benchmark(worlds: Sequence[ClosureWorld]) -> tuple[dict[str, Any], ...]:
    return tuple(
        run_discovery_method(world=world, method_id=method_id)
        for world in worlds
        for method_id in METHOD_IDS
    )


def simulate_reliability_trajectory(world: ClosureWorld) -> dict[str, Any]:
    """Generate nested repeated-cycle observations for one preparation."""

    rng = np.random.default_rng(world.preparation_seed + 71_003)
    truth = world.truth
    integrity = world.initial_integrity
    thermal_load = 0.0
    previous_observed_integrity = integrity
    rows = []
    repair_demanded = bool(truth.cycle_degradation)
    for cycle in range(CYCLE_COUNT):
        requested = 1.0
        accepted = float(previous_observed_integrity >= 0.64 and thermal_load <= 1.10)
        applied = accepted
        realized = applied * (0.92 if truth.delivery_bias else 1.0)
        realized *= float(1.0 + rng.normal(0.0, 0.003))
        if applied > 0.0:
            integrity -= world.degradation_rate
        fault_applied = cycle == FAULT_CYCLE
        if fault_applied:
            integrity -= world.fault_severity
        repair_requested = cycle == REPAIR_CYCLE and repair_demanded
        repair_accepted = repair_requested
        repair_applied = repair_accepted
        repair_realized = repair_applied
        if repair_realized and truth.repair_effective:
            integrity += 0.72 * world.fault_severity + 0.025
        integrity = float(np.clip(integrity, 0.0, 1.0))
        thermal_load = (
            world.thermal_retention * thermal_load
            + realized * (0.16 if truth.weak_thermal_sink else 0.10)
            - (1.0 - applied) * 0.08
        )
        thermal_load = float(max(0.0, thermal_load))
        coupling_factor = 1.0 - 0.08 * truth.coupling_asymmetry
        memory_factor = 1.0 - 0.05 * truth.mix_memory * int(cycle > 0)
        true_output = float(max(0.0, realized * integrity * coupling_factor * memory_factor))
        observed_integrity = float(
            integrity + 0.055 * truth.diagnostic_bias + rng.normal(0.0, world.receiver_noise_sd)
        )
        receiver_valid = not bool(truth.diagnostic_bias)
        preserved = bool(integrity >= 0.62 and thermal_load <= 1.25)
        rows.append(
            {
                "cycle": cycle,
                "requested_drive": requested,
                "accepted_drive": accepted,
                "applied_drive": applied,
                "realized_drive": realized,
                "fault_requested": fault_applied,
                "fault_accepted": fault_applied,
                "fault_applied": fault_applied,
                "fault_realized": fault_applied,
                "repair_requested": repair_requested,
                "repair_accepted": repair_accepted,
                "repair_applied": repair_applied,
                "repair_realized": repair_realized,
                "true_integrity": integrity,
                "observed_integrity": observed_integrity,
                "receiver_valid": receiver_valid,
                "thermal_load": thermal_load,
                "true_output": true_output,
                "true_preserved": preserved,
                "observed_continue": bool(accepted),
            }
        )
        previous_observed_integrity = observed_integrity
    pre_fault = float(rows[FAULT_CYCLE - 1]["true_integrity"])
    post_fault = float(rows[FAULT_CYCLE]["true_integrity"])
    pre_repair = float(rows[REPAIR_CYCLE - 1]["true_integrity"])
    post_repair = float(rows[REPAIR_CYCLE]["true_integrity"])
    early = np.asarray([float(row["true_integrity"]) for row in rows[:FAULT_CYCLE]])
    slope = float(np.polyfit(np.arange(early.size, dtype=np.float64), early, 1)[0])
    degradation_detected = slope < -0.006
    repair_detected = post_repair - pre_repair > 0.04
    repair_axis_evaluable = bool(truth.cycle_degradation)
    one_shot_success = bool(rows[0]["true_output"] >= 0.70 and rows[0]["true_preserved"])
    final_preserved = bool(rows[-1]["true_preserved"])
    false_continue_count = sum(
        bool(row["observed_continue"] and not row["true_preserved"]) for row in rows
    )
    return {
        "world_id": world.world_id,
        "split": world.split,
        "truth_hypothesis_id": truth.hypothesis_id,
        "cycle_count": CYCLE_COUNT,
        "cycles_are_independent_units": False,
        "one_shot_success": one_shot_success,
        "final_preserved": final_preserved,
        "any_preservation_failure": any(not bool(row["true_preserved"]) for row in rows),
        "actuated_cycle_count": sum(float(row["applied_drive"]) > 0.0 for row in rows),
        "hold_cycle_count": sum(float(row["accepted_drive"]) == 0.0 for row in rows),
        "fault_drop": pre_fault - post_fault,
        "repair_gain": post_repair - pre_repair,
        "degradation_detected": degradation_detected,
        "degradation_axis_correct": degradation_detected == bool(truth.cycle_degradation),
        "repair_axis_evaluable": repair_axis_evaluable,
        "repair_detected": repair_detected,
        "repair_axis_correct": (
            repair_detected == bool(truth.repair_effective) if repair_axis_evaluable else None
        ),
        "false_continue_count": false_continue_count,
        "requested_accepted_applied_realized_kept_distinct": True,
        "trajectory": tuple(rows),
    }


def simulate_reliability_panel(
    worlds: Sequence[ClosureWorld],
) -> tuple[dict[str, Any], ...]:
    return tuple(simulate_reliability_trajectory(world) for world in worlds)


def _interval(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError('synthetic propulsion reference world interval requires finite independent-preparation values')
    estimate, lower, upper = bootstrap_mean_interval(
        array,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {"estimate": estimate, "lower": lower, "upper": upper}


def analyse_closure_shadow(
    *,
    worlds: Sequence[ClosureWorld],
    transcripts: Sequence[Mapping[str, Any]],
    reliability: Sequence[Mapping[str, Any]],
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    world_ids = tuple(world.world_id for world in worlds)
    if len(world_ids) < 16 or len(set(world_ids)) != len(world_ids):
        raise ValueError('synthetic propulsion reference world analysis requires unique independent preparations')
    expected_transcripts = len(worlds) * len(METHOD_IDS)
    if len(transcripts) != expected_transcripts or len(reliability) != len(worlds):
        raise ValueError('synthetic propulsion reference world nested panel dimensions differ')
    transcript_by_key = {(str(row["world_id"]), str(row["method_id"])): row for row in transcripts}
    reliability_by_id = {str(row["world_id"]): row for row in reliability}
    if len(transcript_by_key) != expected_transcripts or set(reliability_by_id) != set(world_ids):
        raise ValueError('synthetic propulsion reference world panel identities differ')
    method_results: dict[str, Any] = {}
    for method_index, method_id in enumerate(METHOD_IDS):
        rows = [transcript_by_key[(world_id, method_id)] for world_id in world_ids]
        method_results[method_id] = {
            "equivalence_resolution_rate": _interval(
                [float(bool(row["equivalence_class_resolved"])) for row in rows],
                seed=seed + method_index * 20 + 1,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
            ),
            "exact_resolution_rate": _interval(
                [float(bool(row["exact_truth_resolved"])) for row in rows],
                seed=seed + method_index * 20 + 2,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
            ),
            "mean_acts": _interval(
                [float(row["acts_used"]) for row in rows],
                seed=seed + method_index * 20 + 3,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
            ),
            "mean_cost": _interval(
                [float(row["cost_used"]) for row in rows],
                seed=seed + method_index * 20 + 4,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
            ),
            "version_collapse_fraction": _interval(
                [float(row["version_collapse_fraction"]) for row in rows],
                seed=seed + method_index * 20 + 5,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
            ),
            "false_promotion_count": sum(bool(row["false_promotion"]) for row in rows),
            "typed_nonidentifiability_count": sum(
                row["stop_reason"] == "OBSERVATIONALLY_NONIDENTIFIABLE" for row in rows
            ),
        }
    response_rows = [
        transcript_by_key[(world_id, "response_version_space")] for world_id in world_ids
    ]
    contrasts = {}
    contrast_confidence = 1.0 - (1.0 - confidence_level) / 2.0
    for contrast_index, baseline in enumerate(("random_space_filling", "aggregate_yield_first")):
        baseline_rows = [transcript_by_key[(world_id, baseline)] for world_id in world_ids]
        contrasts[baseline] = {
            "paired_acts_saved_response_minus_baseline": _interval(
                [
                    float(base["acts_used"]) - float(response["acts_used"])
                    for response, base in zip(response_rows, baseline_rows, strict=True)
                ],
                seed=seed + 101 + contrast_index,
                resamples=bootstrap_resamples,
                confidence_level=contrast_confidence,
            ),
            "paired_cost_saved_response_minus_baseline": _interval(
                [
                    float(base["cost_used"]) - float(response["cost_used"])
                    for response, base in zip(response_rows, baseline_rows, strict=True)
                ],
                seed=seed + 111 + contrast_index,
                resamples=bootstrap_resamples,
                confidence_level=contrast_confidence,
            ),
            "simultaneous_confidence_level": contrast_confidence,
        }
    reliability_rows = [reliability_by_id[world_id] for world_id in world_ids]
    repair_evaluable = [row for row in reliability_rows if bool(row["repair_axis_evaluable"])]
    one_shot = [float(bool(row["one_shot_success"])) for row in reliability_rows]
    final = [float(bool(row["final_preserved"])) for row in reliability_rows]
    one_shot_minus_sustained = [left - right for left, right in zip(one_shot, final, strict=True)]
    reliability_result = {
        "one_shot_success_rate": _interval(
            one_shot,
            seed=seed + 201,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        ),
        "final_cycle_preservation_rate": _interval(
            final,
            seed=seed + 202,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        ),
        "one_shot_overstatement": _interval(
            one_shot_minus_sustained,
            seed=seed + 203,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        ),
        "degradation_axis_accuracy": _interval(
            [float(bool(row["degradation_axis_correct"])) for row in reliability_rows],
            seed=seed + 204,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        ),
        "repair_axis_accuracy_when_evaluable": _interval(
            [float(bool(row["repair_axis_correct"])) for row in repair_evaluable],
            seed=seed + 205,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        ),
        "repair_evaluable_preparation_count": len(repair_evaluable),
        "mean_false_continue_cycles": _interval(
            [float(row["false_continue_count"]) for row in reliability_rows],
            seed=seed + 206,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        ),
        "cycles_treated_as_independent_units": False,
    }
    response = method_results["response_version_space"]
    random = method_results["random_space_filling"]
    yield_first = method_results["aggregate_yield_first"]
    selector_supported = bool(
        float(response["equivalence_resolution_rate"]["estimate"])
        >= float(random["equivalence_resolution_rate"]["estimate"])
        and float(response["equivalence_resolution_rate"]["estimate"])
        >= float(yield_first["equivalence_resolution_rate"]["estimate"])
        and int(response["false_promotion_count"]) == 0
    )
    nonidentifiable_world_count = sum(
        len(observable_equivalence_class(world.truth)) > 1 for world in worlds
    )
    return {
        "scientific_grade": "SUPPORTED" if selector_supported else "MIXED",
        "selector_supported_in_truth_known_shadow": selector_supported,
        "independent_preparation_count": len(worlds),
        "nested_discovery_trajectory_count": len(transcripts),
        "nested_cycle_observation_count": len(worlds) * CYCLE_COUNT,
        "algorithm_trajectories_or_cycles_treated_as_replicates": False,
        "true_nonidentifiable_preparation_count": nonidentifiable_world_count,
        "method_results": method_results,
        "paired_contrasts": contrasts,
        "reliability": reliability_result,
        "claim_ceiling": "TRUTH_KNOWN_ICF_LIKE_CLOSURE_SHADOW_ONLY",
        "icf_physics_validated": False,
        "propulsion_system_validated": False,
        "facility_or_live_actuation_performed": False,
    }


def closure_dependency_graph() -> dict[str, Any]:
    """Return the typed non-actuating ICF-to-propulsion dependency graph."""

    nodes = (
        ("node.target-material", "TARGET", "target material and microstructure"),
        ("node.target-geometry", "TARGET", "capsule/hohlraum geometry"),
        ("node.fabrication", "PREPARATION", "fabrication tolerance, process, and lot"),
        ("node.target-quality", "RECEIVER", "surface, pit, thickness, and defect state"),
        ("node.driver-request", "REQUESTED_ACTION", "requested waveform, symmetry, and timing"),
        ("node.driver-delivery", "REALIZED_ACTION", "delivered waveform, symmetry, and timing"),
        ("node.coupling", "STATE", "capsule/hohlraum and plasma coupling"),
        ("node.implosion", "STATE", "implosion trajectory and instability"),
        ("node.mix", "STATE", "mix and retained implosion history"),
        ("node.stagnation", "STATE", "stagnation state"),
        ("node.burn", "RECEIVER", "burn observables and yield"),
        ("node.diagnostics", "OBSERVATION_OPERATOR", "diagnostic validity and latency"),
        ("node.chamber", "SINK", "debris, radiation, clearing, and chamber heat"),
        ("node.conversion", "RECEIVER", "energy/momentum conversion output"),
        ("node.cadence", "DYNAMICS", "target feed, recharge, repetition, and thermal rejection"),
        ("node.structural-health", "PRESERVATION", "fatigue, erosion, and radiation damage"),
        ("node.manufacturing-variation", "DISTURBANCE", "lot variation and component faults"),
        ("node.controller", "CONTROL", "observer, controller, abort, and hold"),
        (
            "node.maintenance",
            "RELIABILITY",
            "maintenance, redundancy, repair, and mission reliability",
        ),
        (
            "node.authority",
            "AUTHORITY",
            "facility, safety, export, licence, compute, and actuation authority",
        ),
    )
    edges = (
        (
            "edge.fabrication-quality",
            "node.fabrication",
            "node.target-quality",
            "LOCAL_RESPONSE",
            "ICF_HELD_PAPER_DATA_ABSENT",
            "SOURCE",
        ),
        (
            "edge.material-quality",
            "node.target-material",
            "node.target-quality",
            "PREPARATION_MAP",
            "PARTIAL_HELD_LITERATURE",
            "STATE",
        ),
        (
            "edge.geometry-quality",
            "node.target-geometry",
            "node.target-quality",
            "PREPARATION_MAP",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "SCALE",
        ),
        (
            "edge.variation-quality",
            "node.manufacturing-variation",
            "node.target-quality",
            "DISTURBANCE_MAP",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "RELIABILITY",
        ),
        (
            "edge.request-delivery",
            "node.driver-request",
            "node.driver-delivery",
            "DELIVERY_MAP",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "DELIVERY",
        ),
        (
            "edge.delivery-coupling",
            "node.driver-delivery",
            "node.coupling",
            "LOCAL_RESPONSE",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "STATE",
        ),
        (
            "edge.target-coupling",
            "node.target-quality",
            "node.coupling",
            "LOCAL_RESPONSE",
            "NO_IDENTIFIED_CAUSAL_BRIDGE",
            "STATE",
        ),
        (
            "edge.geometry-coupling",
            "node.target-geometry",
            "node.coupling",
            "LOCAL_RESPONSE",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "SCALE",
        ),
        (
            "edge.coupling-implosion",
            "node.coupling",
            "node.implosion",
            "LOCAL_RESPONSE",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "STATE",
        ),
        (
            "edge.implosion-mix",
            "node.implosion",
            "node.mix",
            "HISTORY_MAP",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "STATE",
        ),
        (
            "edge.mix-stagnation",
            "node.mix",
            "node.stagnation",
            "LOCAL_RESPONSE",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "STATE",
        ),
        (
            "edge.stagnation-burn",
            "node.stagnation",
            "node.burn",
            "LOCAL_RESPONSE",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "STATE",
        ),
        (
            "edge.state-diagnostics",
            "node.implosion",
            "node.diagnostics",
            "OBSERVATION_MAP",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "STATE",
        ),
        (
            "edge.burn-diagnostics",
            "node.burn",
            "node.diagnostics",
            "OBSERVATION_MAP",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "STATE",
        ),
        (
            "edge.burn-chamber",
            "node.burn",
            "node.chamber",
            "SINK_MAP",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "SINK",
        ),
        (
            "edge.chamber-conversion",
            "node.chamber",
            "node.conversion",
            "CONVERSION_MAP",
            "PROPULSION_ASSUMPTION_ONLY",
            "SCALE",
        ),
        (
            "edge.cadence-delivery",
            "node.cadence",
            "node.driver-delivery",
            "RECOVERY_MAP",
            "PROPULSION_ASSUMPTION_ONLY",
            "SCALE",
        ),
        (
            "edge.cadence-chamber",
            "node.cadence",
            "node.chamber",
            "ACCUMULATION_MAP",
            "PROPULSION_ASSUMPTION_ONLY",
            "SINK",
        ),
        (
            "edge.cadence-health",
            "node.cadence",
            "node.structural-health",
            "DEGRADATION_MAP",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "RELIABILITY",
        ),
        (
            "edge.chamber-health",
            "node.chamber",
            "node.structural-health",
            "DAMAGE_MAP",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "RELIABILITY",
        ),
        (
            "edge.health-maintenance",
            "node.structural-health",
            "node.maintenance",
            "REPAIR_MAP",
            "PROPULSION_ASSUMPTION_ONLY",
            "RELIABILITY",
        ),
        (
            "edge.variation-maintenance",
            "node.manufacturing-variation",
            "node.maintenance",
            "FAULT_MAP",
            "PROPULSION_ASSUMPTION_ONLY",
            "RELIABILITY",
        ),
        (
            "edge.diagnostics-controller",
            "node.diagnostics",
            "node.controller",
            "OBSERVER_MAP",
            "NO_QUALIFIED_LOCAL_SOURCE",
            "STATE",
        ),
        (
            "edge.controller-request",
            "node.controller",
            "node.driver-request",
            "CONTROL_MAP",
            "NO_ACTUATION_AUTHORITY",
            "DELIVERY",
        ),
        (
            "edge.controller-maintenance",
            "node.controller",
            "node.maintenance",
            "ABORT_HOLD_MAP",
            "NO_ACTUATION_AUTHORITY",
            "RELIABILITY",
        ),
        (
            "edge.authority-controller",
            "node.authority",
            "node.controller",
            "AUTHORITY_GATE",
            "AUTHORITY_REQUIRED",
            "AUTHORITY",
        ),
    )
    node_rows = tuple(
        {"node_id": node_id, "role": role, "description": description}
        for node_id, role, description in nodes
    )
    edge_rows = tuple(
        {
            "edge_id": edge_id,
            "source_node_id": source,
            "target_node_id": target,
            "map_type": map_type,
            "evidence_status": evidence,
            "gap_class": gap,
        }
        for edge_id, source, target, map_type, evidence, gap in edges
    )
    node_ids = {row["node_id"] for row in node_rows}
    if any(
        row["source_node_id"] not in node_ids or row["target_node_id"] not in node_ids
        for row in edge_rows
    ):
        raise ValueError('synthetic propulsion reference world dependency graph contains a dangling edge')
    gap_counts = {
        gap: sum(row["gap_class"] == gap for row in edge_rows)
        for gap in ("DELIVERY", "STATE", "SINK", "SCALE", "RELIABILITY", "AUTHORITY", "SOURCE")
    }
    return {
        "graph_id": "closure-graph.icf-to-propulsion-shadow",
        "nodes": node_rows,
        "edges": edge_rows,
        "node_count": len(node_rows),
        "edge_count": len(edge_rows),
        "unclosed_gap_counts": gap_counts,
        "magnetic_confinement_evidence_used_as_icf_evidence": False,
        "laser_plasma_evidence_used_as_icf_evidence": False,
        "propulsion_assumptions_used_as_icf_evidence": False,
        "facility_or_propulsion_validation_claimed": False,
    }


def source_feasibility_disposition(source_audit: Mapping[str, Any]) -> dict[str, Any]:
    qualified = int(source_audit.get("qualified_icf_materialization_count", -1))
    if qualified != 0:
        raise ValueError('synthetic propulsion reference world source audit unexpectedly reports a qualified ICF source')
    ranking = rank_interfaces()
    source_candidate = next(
        row for row in ranking if row["interface_id"] == "interface.fabrication-to-target-quality"
    )
    return {
        "selected_interface_id": source_candidate["interface_id"],
        "selected_experiment_id": source_candidate["experiment_id"],
        "version_space_rank": source_candidate["rank"],
        "selection_reason": "ONLY_INTERFACE_WITH_HELD_ICF_PAPER_LINEAGE_AND_LOCAL_RESPONSE_SCOPE",
        "qualified_local_dataset_or_simulator_exists": False,
        "source_child_specification_issued": False,
        "source_child_status": "SOURCE_REQUIRED",
        "computability_status": "COMPUTABILITY_REQUIRED",
        "missing_source_objects": (
            "byte-verified HDC polishing dataset with preparation/lot identities",
            "measured capsule-quality receiver with uncertainty and sampling coverage",
            "qualified ICF interface simulator or joint target-quality/implosion evidence",
        ),
        "next_bounded_experiment": (
            "register and verify the released HDC polishing materialization, then issue "
            'a measurement and observable-coordinate study for fabrication-to-measured-target-quality response; '
            "do not connect it to implosion or propulsion until a separate bridge exists"
        ),
    }
