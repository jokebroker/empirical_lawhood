'Synthetic propulsion truth-known world, causal cycle and corrected-oracle checks.'

from __future__ import annotations

from dataclasses import replace
from math import isclose
from pathlib import Path

import pytest

from empirical_lawhood.adapters.methods import synthetic_propulsion_discovery as base
from empirical_lawhood.adapters.methods import synthetic_propulsion_exact_comparator as corrected
from empirical_lawhood.adapters.methods.synthetic_propulsion_development_input import SyntheticPropulsionDevelopmentInput, run_native_development_check
from empirical_lawhood.api.codecs import load_registered_authoring

STARTER = Path("experiments/propulsion-reliability-reference/config.json")


def _config() -> SyntheticPropulsionDevelopmentInput:
    return load_registered_authoring(
        STARTER,
        root_schemas={SyntheticPropulsionDevelopmentInput.SCHEMA: SyntheticPropulsionDevelopmentInput},
        maximum_bytes=16 * 1024,
    )


def test_shipped_synthetic_propulsion_world_and_causal_reliability_clock() -> None:
    request = _config()
    report = run_native_development_check(request)
    assert report["independent_generated_preparations"] == 16
    assert report["unique_truth_hypotheses"] == 16
    assert report["nested_decision_transcripts"] == 16 * 4
    assert report["nested_cycle_observations"] == 16 * 24
    assert report["false_promotion_count"] == 0
    assert report["corrected_oracle_unresolved_count"] == 0
    assert report["oracle_deployable"] is False
    assert report["first_world_id"].startswith(request.world_id_prefix + ".")
    assert report["candidate_compiled"] is False

    world = base.generate_closure_worlds(
        split=request.split,
        independent_world_count=request.independent_world_count,
        seed=request.generation_seed,
        world_id_prefix=request.world_id_prefix,
    )[0]

    decisions = report["first_world_oracle_decisions"]
    assert decisions
    assert all(
        row["requested_experiment"]
        == row["accepted_experiment"]
        == row["applied_experiment"]
        == row["realized_experiment"]
        for row in decisions
    )
    assert sum(row["observation_cost"] for row in decisions) <= request.maximum_cost
    cycles = report["first_world_cycles"]
    assert len(cycles) == request.cycle_count
    assert [row["cycle"] for row in cycles] == list(range(request.cycle_count))
    assert [row["fault_applied"] for row in cycles if row["fault_applied"]] == [True]
    for index, row in enumerate(cycles):
        prior_integrity = 1.0 if index == 0 else cycles[index - 1]["observed_integrity"]
        prior_thermal = 0.0 if index == 0 else cycles[index - 1]["thermal_load"]
        # At cycle zero the generator uses world.initial_integrity, so check
        # the causal acceptance cutoff only from the second cycle onward.
        if index > 0:
            assert row["accepted_drive"] == float(
                prior_integrity >= 0.64 and prior_thermal <= 1.10
            )
        assert row["applied_drive"] == row["accepted_drive"]
        assert row["fault_requested"] == (index == request.fault_cycle)
        assert row["repair_requested"] is False or index == request.repair_cycle
    before_fault = cycles[request.fault_cycle - 1]["true_integrity"]
    fault = cycles[request.fault_cycle]
    expected_after_fault = (
        before_fault
        - world.degradation_rate * fault["applied_drive"]
        - world.fault_severity
    )
    assert isclose(fault["true_integrity"], expected_after_fault, abs_tol=1e-12)


def test_independent_interface_equation_and_corrected_oracle_counterexample() -> None:
    truth = base.ClosureHypothesis(0, 1, 1, 1, 1, 0, 1, 0)
    symmetry = base.EXPERIMENT_BY_ID["experiment.symmetry-scan"]
    # The interface's two-bit asymmetry component is separate from the
    # parity of delivery and diagnostic bias: 2*a + ((d+q) mod 2).
    assert base.predicted_outcome(truth, symmetry) == "outcome.2"
    world = base.ClosureWorld(
        world_id='world.synthetic-propulsion-oracle-counterexample',
        split="DEVELOPMENT",
        preparation_index=0,
        preparation_seed=5011,
        truth=truth,
        initial_integrity=0.9,
        degradation_rate=0.012,
        fault_severity=0.12,
        thermal_retention=0.9,
        receiver_noise_sd=0.005,
    )
    old = base.run_discovery_method(world=world, method_id="oracle_upper_bound")
    new = corrected.run_discovery_method(world=world, method_id="oracle_upper_bound")
    assert old["cost_used"] == 12 and not old["equivalence_class_resolved"]
    assert new["cost_used"] == 9 and new["equivalence_class_resolved"]
    assert new["truth_retained"] and not new["false_promotion"]
    assert len(new["decisions"]) == 5 and new["oracle_deployable"] is False


def test_synthetic_propulsion_evaluation_or_changed_budget_refuses_before_generation() -> None:
    request = _config()
    with pytest.raises(ValueError, match="development worlds"):
        replace(request, split="EVALUATION")
    with pytest.raises(ValueError, match="act/cost budget"):
        replace(request, maximum_cost=request.maximum_cost + 1)
