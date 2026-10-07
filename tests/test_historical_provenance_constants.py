"""Readable historical values retain exposure, custody and partition identities.

SPDX-License-Identifier: MPL-2.0
"""

import ast
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from empirical_lawhood.adapters.physical.mast_archive_response_qualification import development
from empirical_lawhood.api import rc_authoring
from empirical_lawhood.infrastructure.campaign_elapsed_budgets import (
    CampaignElapsedBudgetConflict,
    ExternalCampaignElapsedBudgetStore,
)


def _partition_leaf():
    # Full deferred-family import requires optional scikit-learn. Execute its
    # exact pure leaf with its real enum/constants, without selecting a model.
    root = Path(__file__).resolve().parents[1] / "src/empirical_lawhood/adapters/reference_worlds/material_family_discovery"
    contract = ast.parse((root / "contracts.py").read_text())
    source = ast.parse((root / "worlds.py").read_text())
    selected = [node for node in contract.body if isinstance(node, ast.ClassDef) and node.name == "MaterialFamilyDiscoveryPhase" or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in {"UNPARTITIONED_CANDIDATE_ALGORITHM_ID", "PHASE_DISJOINT_CANDIDATE_ALGORITHM_ID"} for target in node.targets)]
    selected += [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_phase_candidates" or isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_HISTORICAL_PHASE_PARTITION_DOMAIN" for target in node.targets)]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    namespace = {"StrEnum": StrEnum, "sha256": sha256}
    exec(compile(ast.fix_missing_locations(module), str(root / "worlds.py"), "exec"), namespace)
    return SimpleNamespace(**namespace)


def test_historical_values_preserve_complete_types_and_bytes():
    assert rc_authoring._HISTORICAL_RC_DEVELOPMENT_STUDY_ID == "ipsmc-rc-development-study-v1"
    assert isinstance(rc_authoring._HISTORICAL_RC_DEVELOPMENT_STUDY_ID, str)
    assert development._HISTORICAL_KDF_ROOT == "el-ec-rcj-io-p5-mw-v4"
    assert development._HISTORICAL_KDF_CHILD_GRAMMAR == "archive/v5/development-screen/{campaign_id}/{shot_id}"
    assert development._HISTORICAL_KDF_CHILD_GRAMMAR.format(campaign_id="M8", shot_id=1001) == "archive/v5/development-screen/M8/1001"
    assert _partition_leaf()._HISTORICAL_PHASE_PARTITION_DOMAIN == b"phase-partition-dev75-v2\x00"


def test_original_elapsed_journal_refuses_default_without_reset(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    observed = []

    def resolve(locator, *, for_write):
        observed.append((locator, for_write))
        return original

    plane = SimpleNamespace(root=SimpleNamespace(resolve=resolve))
    with pytest.raises(CampaignElapsedBudgetConflict, match="ORIGINAL_JOURNAL_REQUIRES_VERIFIED_MIGRATION"):
        ExternalCampaignElapsedBudgetStore(plane)
    assert observed == [("control/campaign-elapsed-budgets/v1", False)]
    assert list(original.iterdir()) == []
    ExternalCampaignElapsedBudgetStore(plane, state_root_relative_path="control/explicit-journal")
    assert len(observed) == 1


@pytest.mark.parametrize(("study_id", "exposed"), (
    ("ipsmc-rc-development-study-v1", True),
    ("rc-ladder-development-study", True),
    ("synthetic.unexposed-study", False),
))
def test_rc_exposure_membership_preserves_both_original_and_public_study_ids(study_id, exposed):
    source = ast.parse(Path(rc_authoring.__file__).read_text())
    expression = next(value for node in ast.walk(source) if isinstance(node, ast.Dict) for key, value in zip(node.keys, node.values) if isinstance(key, ast.Constant) and key.value == "public_study_exposed")
    actual = eval(compile(ast.Expression(expression), rc_authoring.__file__, "eval"), {**vars(rc_authoring), "study": SimpleNamespace(study_id=study_id)})
    assert actual is exposed


@pytest.mark.parametrize("seed", (0, 17, 2147483647))
def test_material_phase_partition_preserves_independent_golden_membership(seed):
    worlds = _partition_leaf()
    candidates = tuple(SimpleNamespace(candidate_id=f"material.synthetic-{index}", family_id=None if index % 2 else f"family.synthetic-{index}") for index in range(12))
    corpus = SimpleNamespace(candidates=candidates)
    config = SimpleNamespace(
        phase=worlds.MaterialFamilyDiscoveryPhase.DEVELOPMENT,
        canonical_candidate_algorithm_id=worlds.PHASE_DISJOINT_CANDIDATE_ALGORITHM_ID,
        development_family_ids=(), evaluation_family_ids=(), world_seed=seed,
    )
    # The literal domain is independent of the production constant. These
    # inputs preserve the original order, digest prefix, endianness and cutoff.
    expected = tuple(candidate for candidate in candidates if int.from_bytes(sha256(b"phase-partition-dev75-v2\x00" + str(seed).encode("ascii") + b"\x00" + (candidate.family_id or candidate.candidate_id).encode("ascii")).digest()[:8], "big") < 3 * 2**62)
    assert worlds._phase_candidates(corpus, config) == expected
    config.phase = worlds.MaterialFamilyDiscoveryPhase.EVALUATION
    assert worlds._phase_candidates(corpus, config) == tuple(candidate for candidate in candidates if candidate not in expected)
