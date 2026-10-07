"""Finite-lawhood starter selection and typed exposure refusal."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from empirical_lawhood.adapters.composition.finite_response_law.native_input import FiniteResponseLawCalibrationInput, _census, check_finite_calibration_input
from empirical_lawhood.api.codecs import load_registered_authoring

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "experiments/finite-response-law/finite-calibration-input.json"


def test_shipped_finite_input_refuses_missing_source_and_changed_role() -> None:
    config = load_registered_authoring(
        CONFIG,
        root_schemas={FiniteResponseLawCalibrationInput.SCHEMA: FiniteResponseLawCalibrationInput},
        maximum_bytes=16 * 1024,
    )
    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_SOURCE_ROOT_REQUIRED"):
        check_finite_calibration_input(
            config, source_root=None, plan=None, prior_exposure=None
        )
    with pytest.raises(ValueError, match="development role"):
        replace(config, evidence_role="PROSPECTIVE")
    with pytest.raises(ValueError, match="development role"):
        replace(config, stage="prospective-evaluation")


def test_finite_exposure_requires_all_four_sorted_identity_arrays() -> None:
    document = {
        "schema": 'empirical-lawhood/methods/finite-response-law/native-exposure-metadata',
        "excluded_unit_ids": ["unit.a"],
        "proposed_unit_ids": ["unit.b"],
        "excluded_seed_ids": ["seed.a"],
        "proposed_seed_ids": ["seed.b"],
    }
    assert _census(json.dumps(document).encode()) == (
        ("unit.a", "unit.b"),
        ("seed.a", "seed.b"),
    )
    del document["proposed_seed_ids"]
    with pytest.raises(TypeError, match="proposed_seed_ids"):
        _census(json.dumps(document).encode())
    document["proposed_seed_ids"] = ["seed.z", "seed.b"]
    with pytest.raises(ValueError, match="proposed_seed_ids"):
        _census(json.dumps(document).encode())


def test_current_local_guide_cannot_substitute_frozen_finite_plan(
    tmp_path: Path,
) -> None:
    config = load_registered_authoring(
        CONFIG,
        root_schemas={FiniteResponseLawCalibrationInput.SCHEMA: FiniteResponseLawCalibrationInput},
        maximum_bytes=16 * 1024,
    )
    held = tmp_path / "held"
    held.mkdir()
    prior = held / "prior.json"
    prior.write_bytes(b"{}")
    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_PLAN_MISMATCH"):
        check_finite_calibration_input(
            config,
            source_root=held,
            plan=ROOT / "experiments/prepared-response/guide.md",
            prior_exposure=prior,
        )
