# SPDX-License-Identifier: MPL-2.0
"""Public record examples retain explicit exposure and exact source identities."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import fresh_assignment_status, load_fresh_reactor_profile
from empirical_lawhood.adapters.composition.prepared_response.native_authoring import _prior_census
from empirical_lawhood.adapters.composition.finite_response_law.native_input import _census
from scripts.generate_operator_examples import examples

DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/reactor-response/exposed-inputs"


def test_assigned_example_is_not_a_prospective_roster_and_inventory_is_exact():
    for name, raw in examples().items():
        assert (DIRECTORY / name).read_bytes() == raw
    profile = load_fresh_reactor_profile(DIRECTORY / "reactor-assigned-exposed.json")
    assert fresh_assignment_status(profile)["prospective_issue_eligible"] is False
    assert profile.assignment.evidence_role == "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
    raw = (DIRECTORY / "exposed-inventory.json").read_bytes()
    assert profile.prior_census.source_inventory.object_fingerprint == sha256(raw).hexdigest()
    document = json.loads(raw)["value"]
    assert document["history_complete"] is False
    for row in document["members"]:
        assert sha256((DIRECTORY / row["locator"]).read_bytes()).hexdigest() == row["sha256"]
    for decoder, name in ((_prior_census, 'prepared-response-prior-exposure.json'),
                          (_census, 'finite-response-prior-exposure.json')):
        assert decoder((DIRECTORY / name).read_bytes()) == (
            profile.prior_census.prior_unit_ids, profile.prior_census.prior_seed_ids,
        )
    with pytest.raises(ValueError, match="COLLISION"):
        revised = replace(profile.prior_census, prior_seed_ids=profile.assignment.seed_ids)
        from empirical_lawhood.kernel.provenance import ObjectIdentity
        replace(profile.assignment, prior_census=ObjectIdentity.from_record(revised.census_id, revised)).check_prior_census(revised)
