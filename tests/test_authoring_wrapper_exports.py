"""Public wrapper products retain frozen science and reviewed provenance identities.

SPDX-License-Identifier: MPL-2.0
"""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from tests.authoring_wrapper_support import FAMILIES, MATRIX_BASELINE_LOCK_SHA256, ROOT, export_wrapper


BASELINE = Path(__file__).parent / "fixtures/authoring-wrapper-exports.json"


# The immutable baseline used the protocol-composition digest as the matrix
# implementation digest. Current composition preserves the caller's true source
# digest; only these hash-derived candidate/report identities change.
MATRIX_IDENTITY_CROSSWALK = {
    "historical_fixture_sha256": "e6002beaf41b688ef18f67aaca3de210551e42206cf640dc3f2af3db0090e5cd",
    "candidate_id": (
        "executable-study-candidate.c2f9157998600c2a6e90d95f9a16efba",
        "executable-study-candidate.1934a1dab1c9cd3f17064caace0e509f",
    ),
    "member_sha256": {
        "candidate.json": (
            "6d604e6029458cacb3f253a0a85075ccfa211f7bb33c24589dbf0e3b79fc6b28",
            "90971ec2e2d6a72cdc9bd727f7a7a6ed0da50a1e3eb0f25ccde02ccadfdf4261",
        ),
        "candidate-report.json": (
            "ef4fa455480e09c24f3f539a5f46bbfe8cc8d7cd3d0cb63344d1b0f7bcc9059d",
            "f6d74708770bc6ea69328231f8057d31ba645dcea4b39d470043f87f91ec1868",
        ),
    },
}


def _matrix_export_with_reviewed_provenance_identity(historical):
    assert sha256(BASELINE.read_bytes()).hexdigest() == MATRIX_IDENTITY_CROSSWALK["historical_fixture_sha256"]
    old_candidate_id, new_candidate_id = MATRIX_IDENTITY_CROSSWALK["candidate_id"]
    assert historical["summary"]["candidate_id"] == old_candidate_id
    old_candidate_sha256, new_candidate_sha256 = MATRIX_IDENTITY_CROSSWALK["member_sha256"]["candidate.json"]
    assert historical["summary"]["candidate_sha256"] == old_candidate_sha256
    members = []
    for member in historical["files"]:
        crosswalk = MATRIX_IDENTITY_CROSSWALK["member_sha256"].get(member["name"])
        if crosswalk is None:
            members.append(member)
        else:
            old_sha256, new_sha256 = crosswalk
            assert member["sha256"] == old_sha256
            members.append({**member, "sha256": new_sha256})
    return {
        "summary": {
            **historical["summary"],
            "candidate_id": new_candidate_id,
            "candidate_sha256": new_candidate_sha256,
        },
        "files": members,
    }


@pytest.mark.parametrize("family", FAMILIES)
def test_public_authoring_wrapper_exports_match_frozen_products_and_reviewed_identities(tmp_path, monkeypatch, family):
    expected = json.loads(BASELINE.read_text())[family]
    if family == "matrix":
        from empirical_lawhood.runtime import candidate_compiler

        compile_candidate = candidate_compiler.compile_draft_candidate
        compiled_implementations = []

        def compile_with_preserved_implementation(*args, **kwargs):
            report = compile_candidate(*args, **kwargs)
            assert report.candidate is not None
            assert report.implementation_sha256 == report.candidate.implementation_sha256 == "b" * 64
            compiled_implementations.append(report.implementation_sha256)
            return report

        monkeypatch.setattr(
            candidate_compiler,
            "compile_draft_candidate",
            compile_with_preserved_implementation,
        )
        expected = _matrix_export_with_reviewed_provenance_identity(expected)
    actual = export_wrapper(family, tmp_path, monkeypatch, frozen_matrix_lock=family == "matrix")
    assert actual == expected
    assert actual["summary"]["native_tasks_executed"] == 0
    if family == "matrix":
        assert compiled_implementations and set(compiled_implementations) == {"b" * 64}


def test_current_matrix_export_binds_actual_lock_without_changing_scientific_roots_or_seeds(tmp_path, monkeypatch):
    from empirical_lawhood.adapters.composition.prepared_response.exposure import PreparedExposureInspection, prepared_seed_ids
    from empirical_lawhood.adapters.composition.prepared_response.native_authoring import PreparedResponseNativeAuthoringInput
    from empirical_lawhood.api.codecs import load_registered_authoring
    from empirical_lawhood.kernel.decoding import decode_canonical_bytes

    actual = export_wrapper("matrix", tmp_path, monkeypatch)
    exposure = decode_canonical_bytes(
        (tmp_path / "export/exposure-inspection.json").read_bytes(),
        PreparedExposureInspection,
        maximum_bytes=1024 * 1024,
    )
    config = load_registered_authoring(
        ROOT / "experiments/prepared-response/prepared-source-qualification-author.json",
        root_schemas={PreparedResponseNativeAuthoringInput.SCHEMA: PreparedResponseNativeAuthoringInput},
        maximum_bytes=16 * 1024,
    )
    source = exposure.source_config
    assert source.dependency_lock_sha256 == sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    assert source.seed_sha256 == config.source_seed_sha256
    assert source.root_seed_census == config.root_seed_census
    historical_identity = replace(source, dependency_lock_sha256=MATRIX_BASELINE_LOCK_SHA256)
    assert source.roots == historical_identity.roots
    assert exposure.proposed_seed_ids == prepared_seed_ids(source) == prepared_seed_ids(historical_identity)
    assert actual["summary"]["native_tasks_executed"] == 0
    assert actual["summary"]["campaign_issued"] is False
