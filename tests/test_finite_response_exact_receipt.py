# SPDX-License-Identifier: MPL-2.0
"""Exact receipt custody in disposable stores, without execution or retry grants."""

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import pytest

from tests.test_doctor_profile import _profile
from empirical_lawhood.adapters.methods.finite_response_law.runtime_ports import (
    FiniteResponseLawControlRuntimeBinding,
    FiniteResponseLawPreparedStorePort,
)
from empirical_lawhood.infrastructure.artifacts import (
    ArtifactIdentityConflict,
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.planning.study_issue import StudyOperationAuthority
from empirical_lawhood.runtime.artifacts import (
    ArtifactProfile,
    ArtifactWriteRequest,
    CanonicalTaskReceipt,
)
from empirical_lawhood.runtime.execution import DependencyReceiptBinding
from empirical_lawhood.runtime.operator_profile import resolve_external_root_contract


RUN = "synthetic-run"
TASK = "synthetic.pure-projection"


@dataclass(frozen=True)
class SyntheticOperand(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/tests/exact-receipt-operand"
    value: str


@pytest.fixture
def custody():
    with TemporaryDirectory(prefix="exact-finite-receipt-", dir="/dev/shm") as directory:
        root = Path(directory)
        (root / "artifacts").mkdir()
        profile = _profile(root)
        repo = Path(__file__).resolve().parents[1]
        guarded = GuardedExternalRoot(
            resolve_external_root_contract(profile, repo_root=repo, home_root=Path.home())
        )
        plane = ExternalArtifactPlane(guarded)
        identity = ObjectIdentity(
            "synthetic.identity", "empirical-lawhood/tests/synthetic", "1.0.0", "a" * 64
        )
        authority = replace(identity, object_schema=StudyOperationAuthority.SCHEMA)
        runtime = FiniteResponseLawControlRuntimeBinding(
            guarded,
            RUN,
            FiniteResponseLawPreparedStorePort(guarded, "synthetic/prepared", 1),
            identity,
            identity,
            authority,
            "operator.execution-service",
            "synthetic.compiler",
        )
        yield plane, ExternalTaskReceiptStore(plane), runtime


def publish_receipt(plane, store, attempt):
    writes = []
    for member in ("a", "b"):
        record = SyntheticOperand(attempt + member)
        writes.append(
            plane.write(
                ArtifactWriteRequest(
                    logical_artifact_id=f"synthetic.{attempt}.{member}",
                    relative_path=f"synthetic/{attempt}/{member}.json",
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/json",
                    publication_scope_id=f"synthetic.{attempt}",
                    publication_scope_relative_root=f"synthetic/{attempt}",
                    payload=record.canonical_bytes(),
                    visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
                    parent_visibility_ceilings=(),
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    minimum_free_bytes=1,
                )
            )
        )
    writes.sort(key=lambda item: item.materialization.materialization_id)
    receipt = CanonicalTaskReceipt(
        f"receipt.{attempt}", RUN, TASK, attempt, "0" * 40, (),
        tuple(item.materialization for item in writes),
        tuple(item.logical for item in writes), (), OperationalStatus.SUCCEEDED, (),
    )
    store.commit(
        receipt,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return receipt


def binding(receipt, *, subset=False):
    ids = tuple(item.materialization_id for item in receipt.output_materializations)
    return DependencyReceiptBinding(receipt.receipt_id, TASK, ids[:1] if subset else ids)


def test_exact_repair_receipt_selected_without_earlier_attempt_fallback(custody):
    plane, store, runtime = custody
    earlier = publish_receipt(plane, store, f"{RUN}.{TASK}.attempt-001")
    exact = publish_receipt(plane, store, f"{RUN}.{TASK}.repair-0123456789ab.attempt-002")
    assert runtime.dependency_receipt(RUN, binding(exact)) == exact
    assert runtime.dependency_receipt(RUN, binding(exact, subset=True)) == exact
    missing = replace(binding(earlier), receipt_id=f"receipt.{RUN}.{TASK}.attempt-003")
    with pytest.raises(ValueError, match="exact committed task receipt"):
        runtime.dependency_receipt(RUN, missing)


def test_exact_receipt_rejects_run_task_and_materialization_substitution(custody):
    plane, store, runtime = custody
    exact = publish_receipt(plane, store, f"{RUN}.{TASK}.attempt-001")
    with pytest.raises(ValueError, match="different run"):
        runtime.dependency_receipt("synthetic-other-run", binding(exact))
    with pytest.raises(ArtifactIdentityConflict, match="run or task"):
        runtime.dependency_receipt(RUN, replace(binding(exact), task_id="synthetic.other-task"))
    with pytest.raises(ValueError, match="exact committed task receipt"):
        runtime.dependency_receipt(
            RUN, replace(binding(exact), output_materialization_ids=("synthetic.unknown",))
        )


def test_exact_receipt_requires_authenticated_original_publication(custody):
    plane, store, runtime = custody
    exact = publish_receipt(plane, store, f"{RUN}.{TASK}.attempt-001")
    path = plane.root.resolve(
        f"runs/{RUN}/receipts/{TASK}/{exact.attempt_id}.json", for_write=False
    )
    path.write_bytes(exact.canonical_bytes() + b" ")
    with pytest.raises(ArtifactIdentityConflict):
        runtime.dependency_receipt(RUN, binding(exact))
