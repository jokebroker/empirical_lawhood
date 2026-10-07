# SPDX-License-Identifier: MPL-2.0

"""Published operator contexts, never selector JSON, supply live reactor ports."""

from dataclasses import replace
from pathlib import Path

import pytest

from empirical_lawhood.adapters.composition.reactor_prefix_response.port_store import ReactorExternalPortStore, ReactorPortContext
from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.study_issue import ExternalStudyOperationAuthorityStore
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ArtifactWriteRequest,
    ExternalRootContract,
)
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile


def test_store_requires_publication_exact_context_and_independent_authority(
    tmp_path: Path,
) -> None:
    """Synthetic development fixture; no issue, native call or scientific grant."""
    root = tmp_path / "store"
    root.mkdir()
    plane = ExternalArtifactPlane(
        GuardedExternalRoot(
            ExternalRootContract(
                "synthetic-store",
                "synthetic store",
                str(root),
                "/",
                OperatorStorageProfile.SCHEMA,
                1,
                None,
                None,
                (),
            )
        )
    )

    def identity(name: str) -> ObjectIdentity:
        return ObjectIdentity(
            name, 'empirical-lawhood/test/synthetic', "1.0.0", "a" * 64
        )

    issued, approval = identity("synthetic.issue"), identity("synthetic.approval")
    grant = StudyOperationAuthority(
        "synthetic.execution",
        StudyAuthorityKind.EXPERIMENT_EXECUTION,
        issued,
        approval,
        identity("synthetic.owner"),
        "operator.execution-service",
        "synthetic.scope",
        None,
        None,
        False,
        False,
        True,
        False,
        False,
        "2026-09-30T00:00:00Z",
        None,
        OutcomeAccess.EVALUATION_SEALED,
    )
    authority = ObjectIdentity.from_record(grant.authority_id, grant)
    record = ReactorPortContext(
        "synthetic.local-custody",
        "local",
        "reactor-local-native-dependency-reader",
        "b" * 64,
        "synthetic-run",
        ArtifactLineageParent(
            issued, VisibilityCeiling.DEVELOPMENT_ONLY, OutcomeAccess.OUTCOME_BLIND
        ),
        authority,
        identity("synthetic.resources"),
        issued,
        (),
        approval=approval,
    )
    selected = ObjectIdentity.from_record(record.context_id, record)
    relative = f"operator/reactor-ports/{record.context_id}.json"
    store = ReactorExternalPortStore(plane)
    locator = root / relative
    locator.parent.mkdir(parents=True)
    locator.write_bytes(record.canonical_bytes())
    with pytest.raises(KeyError):
        store.resolve_port(selected)
    locator.unlink()  # Disposable unpublished synthetic fixture only.
    plane.write(
        ArtifactWriteRequest(
            logical_artifact_id=record.context_id,
            relative_path=relative,
            payload_schema=record.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/json",
            publication_scope_id="synthetic.context",
            publication_scope_relative_root="operator/reactor-ports",
            payload=record.canonical_bytes(),
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            parent_visibility_ceilings=(),
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
    )
    with pytest.raises(KeyError):
        store.resolve_port(selected)
    ExternalStudyOperationAuthorityStore(plane).persist(grant)
    port = store.resolve_port(selected)
    assert port.identity == selected
    assert callable(port.port.read_dependency)
    store.validate_binding(
        route="local",
        native_config_sha256="b" * 64,
        key=record.port_key,
        identity=selected,
    )
    with pytest.raises(ValueError, match="BINDING_CONTEXT_MISMATCH"):
        store.validate_binding(
            route="feed",
            native_config_sha256="b" * 64,
            key=record.port_key,
            identity=selected,
        )
    with pytest.raises(ValueError, match="PUBLICATION_MISMATCH"):
        store.resolve_port(replace(selected, object_fingerprint="c" * 64))
    locator.write_bytes(b"{}")
    with pytest.raises((ValueError, RuntimeError)):
        store.resolve_port(selected)
