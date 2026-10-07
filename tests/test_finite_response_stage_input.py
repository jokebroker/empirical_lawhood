"""Finite stage starts must stop before old parent or unbound native effects."""

from tests.finite_response_seed_fixtures import ASSIGNED_SEEDS
import os
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from empirical_lawhood.adapters.composition.finite_response_law.assignment import FiniteResponseLawCohortAssignment, assigned_native_seed_ids
from empirical_lawhood.adapters.composition.finite_response_law import stage_input
from empirical_lawhood.adapters.composition.finite_response_law.stage_input import STAGE_BINDINGS, FiniteResponseLawStageInput, check_finite_stage_input
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.api.codecs import load_registered_authoring

ROOT = Path(__file__).parents[1] / "experiments/finite-response-law"
FILES = {
    "informative-composition": 'informative-composition-input.json',
    "calibration": "calibration-stage-input.json",
    "supplemental-development": "supplemental-development-input.json",
    "calibration-method": 'calibration-input.json',
    "prospective-evaluation": 'prospective-evaluation-input.json',
    "prospective-continuation": 'prospective-continuation-input.json',
    "preparation-screening": 'preparation-screen-input.json',
}


def _input(stage: str) -> FiniteResponseLawStageInput:
    return load_registered_authoring(
        ROOT / FILES[stage],
        root_schemas={FiniteResponseLawStageInput.SCHEMA: FiniteResponseLawStageInput},
        maximum_bytes=16 * 1024,
    )


def _assignment(stage: str) -> FiniteResponseLawCohortAssignment:
    return FiniteResponseLawCohortAssignment(
        stage,
        f"empirical-lawhood.finite-response-law.{stage}.development",
        FiniteResponseLawScienceSpec().plan_sha256,
        32 if stage == "calibration" else 64,
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
    scientific_seeds=ASSIGNED_SEEDS[f"empirical-lawhood.finite-response-law.{stage}.development"],
    )


def test_all_seven_checked_stage_selections_are_distinct_and_nonpromotable() -> None:
    assert set(FILES) == set(STAGE_BINDINGS)
    loaded = {stage: _input(stage) for stage in FILES}
    assert len({record.config_id for record in loaded.values()}) == 7
    assert len({record.target_prefix for record in loaded.values()}) == 7
    assert {record.evidence_role for record in loaded.values()} == {
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
    }
    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_STAGE_INPUT_INVALID"):
        replace(loaded["calibration"], evidence_role="QUALIFICATION")


def test_stage_start_refuses_without_held_input_before_any_parent_read() -> None:
    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_SOURCE_ROOT_REQUIRED"):
        check_finite_stage_input(
            _input("preparation-screening"),
            source_root=None,
            plan=None,
            prior_exposure=None,
            assignment=None,
            parent_manifest=Path("/missing-parent.json"),
            custody=None,
            reveal_record=None,
            analysis_record=None,
        )


def test_continuation_reuses_original_census_while_fresh_evaluation_refuses_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_bytes = b"synthetic-development-plan"
    plan_sha = FiniteResponseLawScienceSpec().plan_sha256
    monkeypatch.setattr(stage_input, "_read_plan", lambda _: plan_bytes)
    monkeypatch.setattr(
        stage_input, "sha256", lambda _: SimpleNamespace(hexdigest=lambda: plan_sha)
    )
    assignment = FiniteResponseLawCohortAssignment(
        "prospective-evaluation",
        "empirical-lawhood.finite-response-law.prospective-evaluation.synthetic-retained",
        plan_sha,
        64,
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
    scientific_seeds=ASSIGNED_SEEDS["empirical-lawhood.finite-response-law.prospective-evaluation.synthetic-retained"],
    )
    seeds = assigned_native_seed_ids(assignment)
    prior = tmp_path / "prior.json"

    def write_prior(selected_seeds: tuple[str, ...]) -> None:
        prior.write_text(
            json.dumps(
                {
                    "schema": 'empirical-lawhood/methods/finite-response-law/native-exposure-metadata',
                    "excluded_unit_ids": list(assignment.physical_unit_ids),
                    "proposed_unit_ids": [],
                    "excluded_seed_ids": list(selected_seeds),
                    "proposed_seed_ids": [],
                }
            )
        )

    common = {
        "source_root": tmp_path,
        "plan": tmp_path / "plan.md",
        "prior_exposure": prior,
        "assignment": assignment,
        "parent_manifest": None,
        "custody": None,
        "reveal_record": None,
        "analysis_record": None,
    }
    write_prior(seeds)
    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_PARENT_MANIFEST_REQUIRED"):
        check_finite_stage_input(_input("prospective-continuation"), **common)
    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_EXPOSED_ASSIGNED_ROSTER"):
        check_finite_stage_input(_input("prospective-evaluation"), **common)
    write_prior(seeds[:-1])
    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_RETAINED_ROSTER_REQUIRED"):
        check_finite_stage_input(_input("prospective-continuation"), **common)


@pytest.mark.held
def test_authentic_census_selects_calibration_factory_and_refuses_untrusted_parent() -> None:
    frozen = os.environ.get("EL_FINITE_RESPONSE_LAW_FROZEN_PLAN")
    prior = os.environ.get("EL_FINITE_RESPONSE_LAW_PRIOR_EXPOSURE")
    if frozen is None or prior is None:
        pytest.skip("authentic held finite source supplied only for local check")
    prior_path = Path(prior)
    common = {
        "source_root": prior_path.parent,
        "plan": Path(frozen),
        "prior_exposure": prior_path,
        "parent_manifest": None,
        "custody": None,
        "reveal_record": None,
        "analysis_record": None,
    }
    calibration = check_finite_stage_input(
        _input("calibration"), assignment=_assignment("calibration"), **common
    )
    assert calibration["binding_selected"] == 'FiniteResponseLawCalibrationSourceFactory'
    assert calibration["independent_units"] == 32
    assert calibration["streams_reserved"] == 576
    assert calibration["native_tasks_planned"] == 640
    assert calibration["native_tasks_executed"] == 0
    assert calibration["provider_built"] is True
    assert calibration["projection_provider_built"] is True
    assert calibration["evaluation_provider_built"] is True
    assert calibration["future_streams_per_unit"] == 2
    assert calibration["campaign_candidate_compiled"] is False
    assert calibration["status"] == "FINITE_RESPONSE_LAW_DEVELOPMENT_BINDING_READY"

    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_PARENT_MANIFEST_REQUIRED"):
        check_finite_stage_input(
            _input("prospective-evaluation"), assignment=_assignment("prospective-evaluation"), **common
        )
    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_RETAINED_ROSTER_REQUIRED"):
        check_finite_stage_input(
            _input("prospective-continuation"), assignment=_assignment("prospective-evaluation"), **common
        )
    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_PARENT_MANIFEST_REQUIRED"):
        check_finite_stage_input(_input("supplemental-development"), assignment=None, **common)
    forged = {
        **common,
        "parent_manifest": Path("/untrusted-parent.json"),
        "custody": Path("/self-declared-custody.json"),
        "reveal_record": Path("/self-declared-reveal.json"),
        "analysis_record": Path("/self-declared-analysis.json"),
    }
    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_TARGET_CUSTODY_STORE_REQUIRED"):
        check_finite_stage_input(_input("supplemental-development"), assignment=None, **forged)
