# SPDX-License-Identifier: MPL-2.0

"""Target authority requires a complete immutable external publication."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.test_response_parent_custody import _fixture

from empirical_lawhood.adapters.composition.response_parent_custody import import_response_parent_custody
from empirical_lawhood.adapters.composition.response_parent_store import ResponseExternalParentAuthorityStore
from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.runtime.artifacts import ExternalRootContract
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile


def test_real_target_store_replay_and_forged_locator_refusal(tmp_path: Path) -> None:
    args, details = _fixture(tmp_path)
    target = tmp_path / "target-store"
    target.mkdir()
    contract = ExternalRootContract(
        "synthetic-target",
        "synthetic target",
        str(target),
        "/",
        OperatorStorageProfile.SCHEMA,
        1,
        None,
        None,
        (),
    )
    store = ResponseExternalParentAuthorityStore(
        ExternalArtifactPlane(GuardedExternalRoot(contract))
    )
    for field in ("custody", "reveal_record", "analysis_record"):
        grant = details["grants"][args[field]]
        identity = store.persist(grant)
        assert identity.object_fingerprint == grant.fingerprint()
        store.persist(grant)
        args[field] = (
            target / "authority/prepared-response-parent-grants" / (grant.grant_id + ".json")
        )
        assert store.resolve_grant(args[field]) == grant
    args["authority_store"] = store
    witness = import_response_parent_custody(**args)
    assert witness.original_root_ids == ('synthetic.source-qualification.prepared.r000',)
    with pytest.raises(ValueError, match="OUTSIDE_TRUSTED_STORE"):
        store.resolve_grant(tmp_path / "target-custody.json")
    original = store.resolve_grant(args["custody"])
    with pytest.raises(FileExistsError, match="IMMUTABLE"):
        store.persist(replace(original, source_manifest_sha256="a" * 64))
    args["reveal_record"].write_bytes(b"{}")
    with pytest.raises((ValueError, RuntimeError)):
        import_response_parent_custody(**args)
