"Shipped prepared response authoring inputs, bounded prior census and selected source qualification binding."

import json
from dataclasses import replace
from pathlib import Path

import pytest

from empirical_lawhood.adapters.composition.prepared_response.native_authoring import PreparedResponseNativeAuthoringInput, _prior_census
from empirical_lawhood.api.native_authoring import author_matrix_response
from empirical_lawhood.api.codecs import load_registered_authoring

ROOT = Path(__file__).parents[1]
CONFIGS = {
    "source-qualification": ROOT / "experiments/prepared-response/prepared-source-qualification-author.json",
    "information-prediction": ROOT / "experiments/information-response/information-author.json",
    "causal-response-prediction": ROOT / "experiments/causal-response/causal-author.json",
}


@pytest.mark.parametrize("route", ("source-qualification", "information-prediction", "causal-response-prediction"))
def test_shipped_authoring_inputs_refuse_before_held_source_contact(route: str) -> None:
    config = load_registered_authoring(
        CONFIGS[route],
        root_schemas={PreparedResponseNativeAuthoringInput.SCHEMA: PreparedResponseNativeAuthoringInput},
        maximum_bytes=16 * 1024,
    )
    assert config.route == route
    assert config.evidence_role == "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
    with pytest.raises(ValueError, match="SIX_MATRIX_RESPONSE_SOURCE_ROOT_REQUIRED"):
        author_matrix_response(
            config,
            repo_root=ROOT,
            source_root=None,
            plan=None,
            design_packet=None,
            prior_exposure=None,
            model_bank=None,
        )
    with pytest.raises(ValueError, match="target development identity"):
        replace(config, target_prefix="empirical-lawhood.unrelated-source-authoring")
    with pytest.raises(ValueError, match="target development identity"):
        replace(config, evidence_role="PROSPECTIVE")


def test_prior_census_requires_sorted_complete_units_and_streams() -> None:
    value = {
        "schema": 'empirical-lawhood/composition/prepared-response/prepared-exposure-inspection',
        "value": {
            "excluded_unit_ids": ["unit.a"],
            "proposed_unit_ids": ["unit.b"],
            "excluded_seed_ids": ["seed.a"],
            "proposed_seed_ids": ["seed.b"],
        },
    }
    assert _prior_census(json.dumps(value).encode()) == (
        ("unit.a", "unit.b"),
        ("seed.a", "seed.b"),
    )
    value["value"]["excluded_seed_ids"] = ["seed.z", "seed.a"]
    with pytest.raises(ValueError, match="excluded_seed_ids"):
        _prior_census(json.dumps(value).encode())


def test_synthetic_prior_compiles_target_source_qualification_candidate_without_promotion(
    tmp_path: Path,
) -> None:
    # This deliberately fabricated census verifies wiring only. The result is
    # nonpromotable and cannot attest any real prior campaign or native output.
    config = load_registered_authoring(
        CONFIGS["source-qualification"],
        root_schemas={PreparedResponseNativeAuthoringInput.SCHEMA: PreparedResponseNativeAuthoringInput},
        maximum_bytes=16 * 1024,
    )
    held = tmp_path / "held"
    held.mkdir()
    prior = held / "inspection.json"
    prior.write_text(
        json.dumps(
            {
                "schema": 'empirical-lawhood/composition/prepared-response/prepared-exposure-inspection',
                "value": {
                    "excluded_unit_ids": [
                        "unit.synthetic-excluded-a",
                        "unit.synthetic-excluded-b",
                    ],
                    "proposed_unit_ids": ["unit.synthetic-prior"],
                    "excluded_seed_ids": [
                        "seed.synthetic-excluded-a",
                        "seed.synthetic-excluded-b",
                    ],
                    "proposed_seed_ids": ["seed.synthetic-prior"],
                },
            }
        )
    )
    result = author_matrix_response(
        config,
        repo_root=ROOT,
        source_root=held,
        plan=ROOT / "experiments/prepared-response/guide.md",
        design_packet=ROOT / "experiments/prepared-response/guide.md",
        prior_exposure=prior,
        model_bank=None,
        output_dir=tmp_path / "candidate",
    )
    assert result["campaign_candidate_compiled"] is True
    assert result["native_source_binding_selected"] == ("binding.prepared-response.source-qualification.source")
    assert result["native_source_provider_built"] is True
    assert result["native_source_runner_selected"] == 'PreparedStaticSourceTask'
    assert result["native_contact"] is False
    assert result["experiment_id"] == config.target_prefix
    assert result["independent_units"] == 32
    assert result["nested_views_per_unit"] == 2
    assert result["prior_custody_authenticated"] is False
    assert result["prospective_issue_eligible"] is False
    assert result["campaign_issued"] is False
    assert result["native_tasks_executed"] == 0
    assert (tmp_path / "candidate/candidate.json").is_file()
    exposure = json.loads(
        (tmp_path / "candidate/exposure-inspection.json").read_text()
    )["value"]
    assert exposure["excluded_unit_ids"] == [
        "unit.synthetic-excluded-a",
        "unit.synthetic-excluded-b",
        "unit.synthetic-prior",
    ]
    assert exposure["excluded_seed_ids"] == [
        "seed.synthetic-excluded-a",
        "seed.synthetic-excluded-b",
        "seed.synthetic-prior",
    ]
