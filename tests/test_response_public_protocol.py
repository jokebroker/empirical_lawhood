# SPDX-License-Identifier: MPL-2.0

"""The installed finite start binds public bytes and rejects altered recipes."""

import json
from hashlib import sha256
from pathlib import Path

import pytest

from empirical_lawhood.adapters.composition.finite_response_law import stage_input
from empirical_lawhood.adapters.composition.finite_response_law.assignment import (
    FiniteResponseLawCohortAssignment,
    proposed_scientific_seeds,
)
from empirical_lawhood.adapters.methods.finite_response_law import science
from empirical_lawhood.api.codecs import load_registered_authoring

ROOT = Path(__file__).parents[1]


def test_packaged_protocol_drives_real_stage_preflight_and_tamper_stop(tmp_path):
    protocol = Path(science.__file__).with_name('specification.md')
    assert (
        sha256(protocol.read_bytes()).hexdigest()
        == science.FiniteResponseLawScienceSpec().plan_sha256
    )
    config = load_registered_authoring(
        ROOT / "experiments/finite-response-law/calibration-stage-input.json",
        root_schemas={
            stage_input.FiniteResponseLawStageInput.SCHEMA: stage_input.FiniteResponseLawStageInput
        },
    )
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "schema": 'empirical-lawhood/methods/finite-response-law/native-exposure-metadata',
                "excluded_unit_ids": ["synthetic.exposed.unit"],
                "proposed_unit_ids": [],
                "excluded_seed_ids": ["synthetic.exposed.seed"],
                "proposed_seed_ids": [],
            }
        )
    )
    assignment = FiniteResponseLawCohortAssignment(
        "calibration",
        "empirical-lawhood.finite-response-law.calibration.synthetic-public-protocol",
        science.FiniteResponseLawScienceSpec().plan_sha256,
        32,
        "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
        scientific_seeds=proposed_scientific_seeds("calibration", 173),
    )
    args = {
        "source_root": tmp_path,
        "prior_exposure": prior,
        "assignment": assignment,
        "parent_manifest": None,
        "custody": None,
        "reveal_record": None,
        "analysis_record": None,
    }
    report = stage_input.check_finite_stage_input(config, plan=protocol, **args)
    assert report["provider_built"] and report["independent_units"] == 32
    assert report["native_tasks_planned"] == 640
    assert report["native_tasks_executed"] == 0 and not report["campaign_issued"]
    changed = tmp_path / "changed-protocol.md"
    changed.write_bytes(protocol.read_bytes().replace(b"0.80", b"0.79", 1))
    assert changed.read_bytes() != protocol.read_bytes()
    with pytest.raises(ValueError, match="FINITE_RESPONSE_LAW_PLAN_MISMATCH"):
        stage_input.check_finite_stage_input(config, plan=changed, **args)
