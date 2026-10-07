"Target-owned synthetic propulsion truth-known development reference over the retained world."

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from . import synthetic_propulsion_discovery as base
from . import synthetic_propulsion_exact_comparator as corrected


@dataclass(frozen=True, slots=True)
class SyntheticPropulsionDevelopmentInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/synthetic-propulsion-development-input'

    config_id: str
    world_id_prefix: str
    split: str
    independent_world_count: int
    generation_seed: int
    method_ids: tuple[str, ...]
    maximum_acts: int
    maximum_cost: int
    cycle_count: int
    fault_cycle: int
    repair_cycle: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        validate_stable_id(self.world_id_prefix)
        if not self.config_id.startswith(
            "empirical-lawhood-"
        ) or not self.world_id_prefix.startswith("world.empirical-lawhood-"):
            raise ValueError("Synthetic propulsion quick start needs target-owned identities")
        if self.split != "DEVELOPMENT" or not 16 <= self.independent_world_count <= 256:
            raise ValueError("Synthetic propulsion quick start accepts only bounded development worlds")
        if self.generation_seed < 0 or self.generation_seed > 2**31 - 1:
            raise ValueError(
                "Synthetic propulsion generation seed is outside its bounded integer range"
            )
        if self.method_ids != base.METHOD_IDS:
            raise ValueError("Synthetic propulsion method chart differs from the source contract")
        if (self.maximum_acts, self.maximum_cost) != (
            base.MAXIMUM_ACTS,
            base.MAXIMUM_COST,
        ):
            raise ValueError("Synthetic propulsion act/cost budget differs from the source contract")
        if (self.cycle_count, self.fault_cycle, self.repair_cycle) != (
            base.CYCLE_COUNT,
            base.FAULT_CYCLE,
            base.REPAIR_CYCLE,
        ):
            raise ValueError("Synthetic propulsion reliability clock differs from the source contract")


def run_native_development_check(request: SyntheticPropulsionDevelopmentInput) -> dict[str, object]:
    """Run exact generated worlds and report one bounded native action trace."""

    worlds = base.generate_closure_worlds(
        split=request.split,
        independent_world_count=request.independent_world_count,
        seed=request.generation_seed,
        world_id_prefix=request.world_id_prefix,
    )
    decisions = corrected.run_discovery_benchmark(worlds)
    reliability = base.simulate_reliability_panel(worlds)
    first_world = worlds[0]
    first_oracle = next(
        row
        for row in decisions
        if row["world_id"] == first_world.world_id
        and row["method_id"] == "oracle_upper_bound"
    )
    first_reliability = reliability[0]
    return {
        "config_id": request.config_id,
        "development_only": True,
        "truth_known_method_reference": True,
        "physical_icf_claim": False,
        "independent_generated_preparations": len(worlds),
        "unique_truth_hypotheses": len({world.truth.hypothesis_id for world in worlds}),
        "nested_decision_transcripts": len(decisions),
        "nested_cycle_observations": sum(row["cycle_count"] for row in reliability),
        "false_promotion_count": sum(bool(row["false_promotion"]) for row in decisions),
        "corrected_oracle_unresolved_count": sum(
            not bool(row["equivalence_class_resolved"])
            for row in decisions
            if row["method_id"] == "oracle_upper_bound"
        ),
        "oracle_deployable": False,
        "first_world_id": first_world.world_id,
        "first_world_truth": first_world.truth.hypothesis_id,
        "first_world_oracle_decisions": first_oracle["decisions"],
        "first_world_cycles": first_reliability["trajectory"],
        "candidate_compiled": False,
        "campaign_issued": False,
    }


__all__ = ['SyntheticPropulsionDevelopmentInput', "run_native_development_check"]
