"""Finite cohort reservation checks; public assignments never issue native work."""

from tests.finite_response_seed_fixtures import ASSIGNED_SEEDS
import json
import os
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from empirical_lawhood.adapters.composition.finite_response_law.assignment import FiniteResponseLawCohortAssignment, assigned_native_seed_ids
from empirical_lawhood.adapters.composition.finite_response_law.native_input import FiniteResponseLawCalibrationInput, check_finite_calibration_input
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.adapters.simulators.finite_response_law.randomness import FiniteResponseLawCalibrationRNGStream, FiniteResponseLawEvaluationRNGStream, native_rng
from empirical_lawhood.api.codecs import load_registered_authoring

ROOT = Path(__file__).parents[1]
CONFIG = ROOT / "experiments/finite-response-law/finite-calibration-input.json"


def _assignment(stage: str) -> FiniteResponseLawCohortAssignment:
    return FiniteResponseLawCohortAssignment(
        stage,
        f"empirical-lawhood.finite-response-law.{stage}.development",
        FiniteResponseLawScienceSpec().plan_sha256,
        32 if stage == "calibration" else 64,
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
    scientific_seeds=ASSIGNED_SEEDS[f"empirical-lawhood.finite-response-law.{stage}.development"],
    )


def test_assignment_reserves_every_independent_root_and_stream() -> None:
    calibration = _assignment("calibration")
    evaluation = _assignment("prospective-evaluation")
    assert (
        len(set(calibration.stage_units))
        == len(set(calibration.physical_unit_ids))
        == 32
    )
    assert (
        len(set(evaluation.stage_units)) == len(set(evaluation.physical_unit_ids)) == 64
    )
    assert set(calibration.stage_units).isdisjoint(evaluation.stage_units)
    assert not any(
        unit.startswith(("calibration.r", "prospective-evaluation.r")) for unit in calibration.stage_units
    )
    assert len(assigned_native_seed_ids(calibration)) == 576
    assert len(assigned_native_seed_ids(evaluation)) == 1153
    assert set(assigned_native_seed_ids(calibration)).isdisjoint(
        assigned_native_seed_ids(evaluation)
    )
    assert f"seed.pcg64.{calibration.seed_for('prefix', 0):032x}" in assigned_native_seed_ids(
        calibration
    )
    assert (
        f"seed.pcg64.{evaluation.seed_for('bootstrap'):032x}"
        in assigned_native_seed_ids(evaluation)
    )
    for changed in (
        {"cohort_namespace": "calibration"},
        {"cohort_namespace": "empirical-lawhood.finite-response-law.calibration." + "x" * 128},
        {"root_count": 31},
        {"science_plan_sha256": "0" * 64},
        {"evidence_role": "QUALIFICATION"},
    ):
        with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_ASSIGNMENT_INVALID"):
            replace(calibration, **changed)


def test_assigned_native_rng_matches_independent_jumped_state_and_rejects_bad_units() -> (
    None
):
    for stage, stream_type in (
        ("calibration", FiniteResponseLawCalibrationRNGStream),
        ("prospective-evaluation", FiniteResponseLawEvaluationRNGStream),
    ):
        assignment = _assignment(stage)
        reserved = set(assigned_native_seed_ids(assignment))
        for unit in (assignment.stage_units[0], assignment.stage_units[-1]):
            for purpose, substream, jump in (
                ("prefix", "coarse", 0),
                ("prefix", "passive-probes", 4),
                ("future-2", "bridge", 1),
            ):
                expected_seed = assignment.seed_for(purpose, assignment.stage_units.index(unit))
                record = native_rng(unit, purpose, substream, committed_seed=expected_seed)
                assert type(record) is stream_type
                expected_state = sha256(
                    json.dumps(
                        np.random.PCG64(expected_seed).jumped(jump).state,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                assert record.committed_seed_decimal == str(expected_seed)
                assert record.initial_state_sha256 == expected_state
                assert f"seed-state.pcg64.{expected_state}" in reserved
        for bad in (
            f"{assignment.cohort_namespace}.r{assignment.root_count:03d}",
            f"{assignment.cohort_namespace}.r0",
            f"empirical-lawhood.finite-response-law.{stage}.r000",
            f"{assignment.cohort_namespace}.r000.r001",
        ):
            with pytest.raises(ValueError, match="Finite response-law native RNG"):
                native_rng(bad, "prefix", "coarse", committed_seed=assignment.seed_for("prefix", 0))
        with pytest.raises(ValueError, match="Finite response-law native RNG"):
            native_rng(assignment.stage_units[0], "parent", "passive-probes", committed_seed=assignment.seed_for("parent", 0))


@pytest.mark.held
def test_assigned_calibration_preflight_refuses_a_seed_collision(
    tmp_path: Path,
) -> None:
    frozen = os.environ.get("EL_FINITE_RESPONSE_LAW_FROZEN_PLAN")
    if frozen is None:
        pytest.skip("frozen source plan supplied only for held-input checks")
    assignment = _assignment("calibration")
    held = tmp_path / "held"
    held.mkdir()
    prior = held / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "schema": 'empirical-lawhood/methods/finite-response-law/native-exposure-metadata',
                "excluded_unit_ids": ["unit.old"],
                "proposed_unit_ids": ["unit.other"],
                "excluded_seed_ids": [assigned_native_seed_ids(assignment)[0]],
                "proposed_seed_ids": ["seed.old"],
            }
        )
    )
    config = load_registered_authoring(
        CONFIG,
        root_schemas={FiniteResponseLawCalibrationInput.SCHEMA: FiniteResponseLawCalibrationInput},
        maximum_bytes=16 * 1024,
    )
    with pytest.raises(
        ValueError, match="FINITE_RESPONSE_LAW_EXPOSED_ASSIGNED_ROSTER: 0 units, 1 streams"
    ):
        check_finite_calibration_input(
            config,
            source_root=held,
            plan=Path(frozen),
            prior_exposure=prior,
            assignment=assignment,
        )


@pytest.mark.held
def test_authentic_exposed_roster_refuses_but_assigned_preflight_has_no_tasks() -> None:
    frozen = os.environ.get("EL_FINITE_RESPONSE_LAW_FROZEN_PLAN")
    prior = os.environ.get("EL_FINITE_RESPONSE_LAW_PRIOR_EXPOSURE")
    if frozen is None or prior is None:
        pytest.skip("authentic held finite source supplied only for local check")
    config = load_registered_authoring(
        CONFIG,
        root_schemas={FiniteResponseLawCalibrationInput.SCHEMA: FiniteResponseLawCalibrationInput},
        maximum_bytes=16 * 1024,
    )
    prior_path = Path(prior)
    source_root = prior_path.parent
    with pytest.raises(
        ValueError, match="FINITE_RESPONSE_LAW_EXPOSED_CALIBRATION_ROSTER: 32 units, 576 streams"
    ):
        check_finite_calibration_input(
            config,
            source_root=source_root,
            plan=Path(frozen),
            prior_exposure=prior_path,
        )
    report = check_finite_calibration_input(
        config,
        source_root=source_root,
        plan=Path(frozen),
        prior_exposure=prior_path,
        assignment=_assignment("calibration"),
    )
    assert report["independent_units"] == 32
    assert report["streams_reserved"] == 576
    assert report["provider_built"] is False
    assert report["native_tasks_planned"] == report["native_tasks_executed"] == 0
    assert report["status"] == "FINITE_RESPONSE_LAW_ASSIGNMENT_RESERVED_USE_STAGE_SELECTOR"
