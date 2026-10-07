from __future__ import annotations

import json
import subprocess
from hashlib import sha256

import pytest

from empirical_lawhood.adapters.methods.effective_law_comparison.analysis import PosthocDocumentSet, execute_analysis
from empirical_lawhood.adapters.methods.effective_law_comparison.authoring import build_authorized_package, build_freeze
from empirical_lawhood.adapters.methods.effective_law_comparison.sources import ParentSourceDefinition, SourceMemberDefinition, qualify_parent_sources
from empirical_lawhood.adapters.independent_source_exports import IndependentSourceExport, OriginalObjectReference, OriginalArtifactReference
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.runtime.artifacts import ArtifactGenericValidation, ArtifactProfile, LogicalArtifactIdentity, ArtifactMaterialization, ArtifactManifest
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.worlds import WorldKind
from empirical_lawhood.planning.posthoc import POSTHOC_ANALYSIS_IDS, PosthocExecutionAuthorization


def _parent(source_export: IndependentSourceExport | None = None) -> ParentSourceDefinition:
    return ParentSourceDefinition(
        parent_id="sample-parent",
        lane_id="sample-lane",
        study_id="sample-programme",
        target_id="sample-target",
        world_id="sample-world",
        world_kind=WorldKind.NUMERICAL_SIMULATOR,
        independent_unit_id="sample-unit",
        members=(
            SourceMemberDefinition(
                "sample-artifact", "terminal", "inputs/sample.json", source_export
            ),
        ),
        role_ids=("terminal",),
        analysis_ids=("analysis.obstruction-signatures",),
    )


def _current_source(tmp_path, *, write_manifest: bool = True):
    path = tmp_path / "inputs" / "sample.json"
    path.parent.mkdir(exist_ok=True)
    schema = "empirical-lawhood/methods/effective-law-comparison/source-contract-fixture"
    payload = json.dumps({"schema": schema, "version": "1.0.0", "value": {"terminal": "action_response"}}).encode()
    path.write_bytes(payload)
    digest = sha256(payload).hexdigest()
    validator = ArtifactGenericValidation("validator.canonical-json", "1.0.0", "3" * 64, schema, ArtifactProfile.CANONICAL_JSON)
    logical = LogicalArtifactIdentity("sample-artifact", digest, schema, ArtifactProfile.CANONICAL_JSON, "application/json", VisibilityCeiling.OUTCOME_VISIBLE, (), OutcomeAccess.EVALUATION_REVEALED, validator)
    physical = ArtifactMaterialization("sample-materialization", "sample-artifact", "sample-root", "inputs/sample.json", digest, len(payload), "none")
    manifest = ArtifactManifest(logical, physical)
    if write_manifest:
        path.with_suffix(".json.manifest.json").write_bytes(manifest.canonical_bytes())
    original_schema = "icf-yolo/results/qtsq-closeout/v4"
    export = IndependentSourceExport(
        original_source=OriginalObjectReference("original.sample", original_schema, "1", "a" * 64),
        original_payload_version="1",
        original_artifact=OriginalArtifactReference("original.sample-artifact", "terminal", original_schema, "a" * 64, "application/json", 123),
        original_relative_path="original/sample.json",
        original_task_receipt=OriginalObjectReference("original.sample-receipt", "icf-yolo/runtime/task-receipt/v2", "2", "b" * 64),
        original_manifest_sha256="c" * 64,
        original_publication_commit_sha256="d" * 64,
        original_source_commit="e" * 40,
        interpretation_sha256="f" * 64,
        verified_export=ObjectIdentity("verified.sample-export", "empirical-lawhood/migration/verified-source-export", "1.0.0", "1" * 64),
        target_source=ObjectIdentity("current.sample", schema, "1.0.0", digest),
        target_artifact=ArtifactIdentity("sample-artifact", "terminal", schema, digest, "application/json", len(payload)),
        target_relative_path="inputs/sample.json",
        target_task_receipt=ObjectIdentity("current.sample-receipt", "empirical-lawhood/runtime/canonical-task-receipt", "1.0.0", "2" * 64),
        target_manifest_sha256=manifest.fingerprint(),
        target_publication_commit_sha256="4" * 64,
        original_physical_unit_ids=("original.sample-unit",),
        target_physical_unit_ids=("sample-unit",),
    )
    return path, payload, export, manifest


def test_explicit_posthoc_source_roster_requires_bound_external_bytes(tmp_path) -> None:
    path, payload, export, manifest = _current_source(tmp_path, write_manifest=False)
    absent = qualify_parent_sources(tmp_path, (_parent(export),))["value"]["parents"][0]
    assert absent["status"] == "PARTIALLY_ELIGIBLE"
    assert "PARENT_RECEIPT_ABSENT" in absent["reason_codes"]

    sidecar = path.with_suffix(".json.manifest.json")
    sidecar.write_bytes(manifest.canonical_bytes())
    qualified = qualify_parent_sources(tmp_path, (_parent(export),))["value"]["parents"][0]
    assert qualified["status"] == "ELIGIBLE"
    assert qualified["members"][0]["content_sha256"] == sha256(payload).hexdigest()

    sidecar.write_bytes(manifest.canonical_bytes() + b" ")
    drifted = qualify_parent_sources(tmp_path, (_parent(export),))["value"]["parents"][0]
    assert drifted["status"] != "ELIGIBLE"
    assert "CURRENT_SOURCE_RECEIPT_CUSTODY_MISMATCH" in drifted["reason_codes"]


def test_partition_sensitive_analysis_requires_explicit_mapping() -> None:
    document = {
        "schema": "empirical-lawhood/methods/effective-law-comparison/source-contract-fixture",
        "version": "1.0.0",
        "value": {"reason_codes": ["POWER_STOP"]},
    }
    with pytest.raises(ValueError, match="parent partition map"):
        execute_analysis("analysis.obstruction-signatures", {"sample-artifact": document}, ("sample-parent",))
    mapped = PosthocDocumentSet(
        {"sample-artifact": document}, {"sample-artifact": "sample-parent"}
    )
    result = execute_analysis("analysis.obstruction-signatures", mapped, ("sample-parent",))
    assert result["value"]["parent_ids"] == ["sample-parent"]


def test_posthoc_freeze_uses_explicit_roster_and_clean_source(tmp_path, monkeypatch) -> None:
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    _, _, export, _ = _current_source(tmp_path)
    from empirical_lawhood.adapters.methods.effective_law_comparison import provider
    monkeypatch.setattr(
        provider,
        "SUPPORTED_INPUT_SCHEMAS",
        tuple(sorted((*provider.SUPPORTED_INPUT_SCHEMAS, export.target_source.object_schema))),
    )
    (tmp_path / "method.py").write_text("METHOD = 'bounded'\n")
    subprocess.run(("git", "add", "."), cwd=tmp_path, check=True)
    subprocess.run(
        ("git", "-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "freeze"),
        cwd=tmp_path,
        check=True,
    )
    parent = _parent(export)
    parent = ParentSourceDefinition(
        parent.parent_id,
        parent.lane_id,
        parent.study_id,
        parent.target_id,
        parent.world_id,
        parent.world_kind,
        parent.independent_unit_id,
        parent.members,
        parent.role_ids,
        tuple(sorted(POSTHOC_ANALYSIS_IDS)),
    )
    qualification = qualify_parent_sources(tmp_path, (parent,))
    freeze, closure = build_freeze(
        repository_root=tmp_path,
        qualification=qualification,
        frozen_at_utc="2026-09-28T00:00:00Z",
        parent_sources=(parent,),
        implementation_files=("method.py",),
        plan_sha256=sha256(b"one explicit nonpromotable plan").hexdigest(),
        freeze_id="sample-posthoc.freeze",
    )
    assert len(freeze.analyses) == 9
    assert len(freeze.parents) == 1
    assert closure["value"]["base_commit"] == freeze.implementation_commit
    authority = PosthocExecutionAuthorization(
        authorization_id="sample-posthoc.authorization",
        freeze=ObjectIdentity.from_record(freeze.freeze_id, freeze),
        requested_scope_id="sample-posthoc",
        external_run_root="runs/sample-posthoc",
        requested_budget=ResourceBudget(1, 1024**3, 0, 3600, 1024**2, 1024**2),
        proposer_id="sample-human-proposer",
        approver_id="sample-human-approver",
        authorized_at_utc="2026-09-28T00:00:01Z",
        implementation_commit=freeze.implementation_commit,
        passed_gate_ids=("bounded-source-roster",),
        plan_mutated=False,
        grants_claim_promotion=False,
        nonactuating=True,
    )
    plan, package, registry = build_authorized_package(
        freeze,
        authorization=authority,
        run_root="runs/sample-posthoc",
        package_id="sample-posthoc.package",
    )
    assert len(plan.tasks) == 21
    assert package.authorization == authority
    for task in plan.tasks:
        registry.require(task.capability)
    (tmp_path / "method.py").write_text("METHOD = 'changed'\n")
    with pytest.raises(PermissionError, match="clean source commit"):
        build_freeze(
            repository_root=tmp_path,
            qualification=qualification,
            frozen_at_utc="2026-09-28T00:00:00Z",
            parent_sources=(parent,),
            implementation_files=("method.py",),
            plan_sha256=sha256(b"one explicit nonpromotable plan").hexdigest(),
            freeze_id="sample-posthoc.freeze",
        )
