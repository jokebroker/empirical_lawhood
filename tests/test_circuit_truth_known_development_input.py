"""Physical scale morphism method reference over a new truth-known development seed pair."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from empirical_lawhood.adapters.methods.physical_scale_morphism.conformance import run_truth_blind_case
from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.contracts import PhysicalScaleMorphismTruthSuiteConfig, truth_input_sha256
from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.native_quickstart import run_truth_method_development_check
from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.oracle import oracle_for_case, score_truth_observation
from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.worlds import generate_truth_cases
from empirical_lawhood.api.codecs import load_registered_authoring

STARTER = Path("experiments/response-method-reference/config.json")


def _config() -> PhysicalScaleMorphismTruthSuiteConfig:
    return load_registered_authoring(
        STARTER,
        root_schemas={PhysicalScaleMorphismTruthSuiteConfig.SCHEMA: PhysicalScaleMorphismTruthSuiteConfig},
        maximum_bytes=16 * 1024,
    )


def test_shipped_truth_reference_counts_and_claim_ceiling() -> None:
    config = _config()
    report = run_truth_method_development_check(config)
    assert config.validation_seeds == (20260930, 20261001)
    assert report["independent_truth_cases"] == 30
    assert report["fixture_count"] == 15
    assert report["nested_variants_per_fixture"] == 2
    assert report["passed_case_count"] == 30
    assert report["failed_case_ids"] == ()
    assert report["method_qualified_on_exposed_reference"] is True
    assert report["scientific_ceiling"] == "NON_PROMOTABLE"
    assert report["claim_promotion_allowed"] is False
    assert report["first_case_id"].startswith(f"case.{config.config_id}.")
    assert report["physical_board_claim"] is False
    assert report["campaign_candidate_compiled"] is False


def test_independent_energy_collision_and_guard_counterexample() -> None:
    config = _config()
    cases = generate_truth_cases(config, case_id_prefix=f"case.{config.config_id}")
    energy = next(
        case
        for case in cases
        if case.fixture_id == "energy-mean-collision" and case.variant.value == "DETERMINISTIC"
    )
    data = {value.datum_id: value.numeric_value for value in energy.input_data}
    left = [data[f"left-voltage-{index:03d}"] for index in range(16)]
    right = [data[f"right-voltage-{index:03d}"] for index in range(16)]
    assert sum(left) / 16 == sum(right) / 16 == 1
    assert sum(value * value for value in left) == 32
    assert sum(value * value for value in right) == 16
    assert run_truth_blind_case(energy).observed_code_ids == (
        "mean-collision-energy-separated",
    )

    guard = next(
        case
        for case in cases
        if case.fixture_id == "guard-false-safe" and case.variant.value == "DETERMINISTIC"
    )
    changed_data = tuple(
        replace(value, categorical_value="PASS")
        if value.datum_id == "fine-gate-0"
        else value
        for value in guard.input_data
    )
    changed = replace(
        guard,
        input_data=changed_data,
        input_data_sha256=truth_input_sha256(changed_data),
    )
    observation = run_truth_blind_case(changed)
    score = score_truth_observation(observation, oracle_for_case(changed))
    assert observation.truth_input_sha256 == changed.input_data_sha256
    assert observation.privileged_label_access_count == 0
    assert observation.observed_code_ids == ("false-safe-missed",)
    assert score.passed is False
    assert score.missing_code_ids == ("mean-false-safe-rejected",)


def test_historical_input_and_physical_promotion_refuse() -> None:
    config = _config()
    with pytest.raises(ValueError, match="historical validation seeds"):
        run_truth_method_development_check(
            replace(config, validation_seeds=(42017, 90173))
        )
    with pytest.raises(ValueError, match="target-owned config ID"):
        run_truth_method_development_check(
            replace(config, config_id="physical-scale-morphism-truth-suite")
        )
    with pytest.raises(ValueError, match="physical outcomes"):
        replace(config, protected_physical_outcome_count=1)
