# SPDX-License-Identifier: MPL-2.0
"""Synthetic historical inventories with real target-store publication.

Canonical scientific records are fabricated conformance data. The source receipt
envelopes model the historical import format, not a qualified native execution.
Only disposable stores may consume these grants; they confer no real authority.
"""

from dataclasses import replace
from pathlib import Path

from tests.test_response_parent_custody import _canonical, _put, _wrapped
from empirical_lawhood.adapters.composition.response_parent_custody import ResponseParentTargetGrant, ResponseSourceParentIdentity
from empirical_lawhood.adapters.composition.response_parent_store import ResponseExternalParentAuthorityStore
from empirical_lawhood.infrastructure.artifacts import (
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.runtime.operator_profile import resolve_external_root_contract


def publish_parent(
    root: Path,
    *,
    profile,
    repo,
    route,
    parent,
    roots,
    members=(),
    source_name="held",
    artifact_members=(),
    parent_artifact=None,
):
    source = root / source_name
    project = source / "project"
    run = project / "runs" / "synthetic"
    roster, attempts = [], []

    def publish(path, raw, material_id):
        entry = _put(path, raw)
        roster.append(entry)
        material = {
            "materialization_id": material_id,
            "relative_path": str(path.relative_to(project)),
            "physical_sha256": entry["sha256"],
            "size_bytes": entry["bytes"],
            "storage_root_id": "synthetic-source",
        }
        marker_path = run / ".publication-batches" / f"{material_id}.commit.json"
        marker = _put(marker_path, _canonical({"member": material_id}))
        roster.append(marker)
        roster.append(
            _put(
                Path(str(path) + ".manifest.json"),
                _canonical(
                    {
                        "materialization": _wrapped(material, "material"),
                        "logical": {"content_sha256": entry["sha256"]},
                        "publication": {
                            "commit_marker_relative_path": str(
                                marker_path.relative_to(project)
                            ),
                            "commit_marker_sha256": marker["sha256"],
                            "commit_marker_size_bytes": marker["bytes"],
                            "members": [
                                {
                                    "materialization_id": material_id,
                                    "physical_sha256": entry["sha256"],
                                }
                            ],
                        },
                    }
                ),
            )
        )
        return material

    def task(task_id, outputs):
        receipt_id = f"receipt.synthetic.{task_id}.attempt-001"
        path = (
            run / "receipts" / task_id / f"{receipt_id.removeprefix('receipt.')}.json"
        )
        material = publish(
            path,
            _canonical(
                _wrapped(
                    {
                        "task_id": task_id,
                        "receipt_id": receipt_id,
                        "run_id": "synthetic",
                        "operational_status": "SUCCEEDED",
                        "output_materializations": [
                            _wrapped(m, "material") for m in outputs
                        ],
                    },
                    "receipt",
                )
            ),
            receipt_id,
        )
        attempts.append(
            _wrapped(
                {
                    "task_id": task_id,
                    "disposition": "SUCCEEDED",
                    "receipt_id": receipt_id,
                    "receipt_sha256": material["physical_sha256"],
                },
                "attempt",
            )
        )

    # Every independent root is explicitly accounted for even when its record
    # is aggregated into a parent fit. Numerical views never add root IDs.
    for root_id in roots:
        task(root_id + ".synthetic-accounting", ())
    outputs = []
    if artifact_members or parent_artifact is not None:
        from hashlib import sha256
        from empirical_lawhood.infrastructure.task_receipts import (
            decode_artifact_manifest,
        )
        from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
        from empirical_lawhood.runtime.artifacts import (
            ArtifactProfile,
            ArtifactWriteRequest,
        )

        project.mkdir(parents=True, exist_ok=True)
        contract = resolve_external_root_contract(
            profile, repo_root=repo, home_root=Path.home()
        )
        source_plane = ExternalArtifactPlane(
            GuardedExternalRoot(
                replace(
                    contract,
                    storage_root_id="synthetic-source",
                    canonical_path=str(project),
                )
            )
        )

        def publish_full(artifact, raw, relative):
            assert (sha256(raw).hexdigest(), len(raw)) == (
                artifact.sha256,
                artifact.size_bytes,
            )
            written = source_plane.write(
                ArtifactWriteRequest(
                    logical_artifact_id=artifact.artifact_id,
                    relative_path=relative,
                    payload_schema=artifact.payload_schema,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type=artifact.media_type,
                    publication_scope_id="synthetic.fixture-inputs",
                    publication_scope_relative_root="runs/synthetic/outputs",
                    payload=raw,
                    visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
                    parent_visibility_ceilings=(),
                    outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                    minimum_free_bytes=1,
                )
            )
            sidecar = Path(str(project / relative) + ".manifest.json")
            manifest = decode_artifact_manifest(sidecar.read_bytes())
            marker = project / manifest.publication.commit_marker_relative_path
            for path in (project / relative, sidecar, marker):
                data = path.read_bytes()
                roster.append(
                    {
                        "path": str(path),
                        "bytes": len(data),
                        "sha256": sha256(data).hexdigest(),
                    }
                )
            return written.materialization.to_document()["value"]

        for artifact, raw in artifact_members:
            outputs.append(
                publish_full(
                    artifact,
                    raw,
                    f"runs/synthetic/outputs/{artifact.artifact_id}.fixture",
                )
            )
    for index, record in enumerate(members):
        outputs.append(
            publish(
                run / "outputs" / f"member-{index}.json",
                record.canonical_bytes(),
                f"member-{index}",
            )
        )
    parent_path = run / "outputs" / "parent.json"
    outputs.append(
        publish(parent_path, parent.canonical_bytes(), "parent")
        if parent_artifact is None
        else publish_full(
            parent_artifact,
            parent.canonical_bytes(),
            str(parent_path.relative_to(project)),
        )
    )
    stage_path = run / "outputs" / "stage.json"
    outputs.append(
        publish(
            stage_path,
            _canonical(
                _wrapped(
                    {
                        "scientific_product": _wrapped(
                            {
                                "object_schema": parent.SCHEMA,
                                "object_version": parent.VERSION,
                                "object_fingerprint": parent.fingerprint(),
                            },
                            "identity",
                        ),
                    },
                    "stage",
                )
            ),
            "stage",
        )
    )
    task("synthetic.analysis", outputs)
    terminal_path = run / "recovery" / "resource-terminal.json"
    roster.append(
        _put(
            terminal_path,
            _canonical(
                _wrapped(
                    {
                        "run_id": "synthetic",
                        "operational_status": "SUCCEEDED",
                        "attempts": attempts,
                    },
                    "terminal",
                )
            ),
        )
    )
    manifest_path = source / "analysis" / "input-manifest.json"
    manifest = _put(manifest_path, _canonical(sorted(roster, key=lambda e: e["path"])))
    contract = resolve_external_root_contract(
        profile, repo_root=repo, home_root=Path.home()
    )
    store = ResponseExternalParentAuthorityStore(
        ExternalArtifactPlane(GuardedExternalRoot(contract))
    )
    grant = ResponseParentTargetGrant(
        f"synthetic.{source_name}.custody",
        "CUSTODY",
        route,
        ResponseSourceParentIdentity(parent.SCHEMA, parent.VERSION, parent.fingerprint()),
        manifest["sha256"],
        "project",
        str(terminal_path.relative_to(project)),
        str(parent_path.relative_to(project)),
        str(stage_path.relative_to(project)),
        "synthetic-source",
        roots,
        "EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE"
        if route == "response-composition"
        else "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
    )
    args = {
        "source_root": source,
        "parent_manifest": manifest_path,
        "authority_store": store,
    }
    for field, action in (
        ("custody", "CUSTODY"),
        ("reveal_record", "REVEAL"),
        ("analysis_record", "ANALYSIS"),
    ):
        selected = replace(
            grant, grant_id=f"synthetic.{source_name}.{action.lower()}", action=action
        )
        store.persist(selected)
        args[field] = (
            Path(contract.canonical_path)
            / "authority/prepared-response-parent-grants"
            / f"{selected.grant_id}.json"
        )
    return args, parent_path
