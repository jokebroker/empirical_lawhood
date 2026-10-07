"""Synthetic complete-inventory replay and pre-read authority refusals."""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from empirical_lawhood.adapters.composition import response_parent_custody
from empirical_lawhood.adapters.composition.response_parent_custody import ResponseParentTargetGrant, ResponseSourceParentIdentity, import_response_parent_custody


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _wrapped(value: object, name: str) -> dict:
    return {"schema": f"empirical-lawhood/testing/fixtures/response-parent-custody/{name}", "value": value, "version": "1.0.0"}


def _put(path: Path, payload: bytes) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path),
        "sha256": sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _fixture(tmp_path: Path, *, omit_receipt: bool = False) -> tuple[dict, dict]:
    root = tmp_path / "held"
    project = root / "project"
    run = project / "runs" / "run1"
    task = "synthetic.source-qualification.prepared.r000.prefix.native"
    receipt_id = f"receipt.run1.{task}.attempt-001"
    receipt_path = (
        run / "receipts" / task / (receipt_id.removeprefix("receipt.") + ".json")
    )
    output_path = run / "outputs" / task / "scalar.json"
    stage_path = run / "outputs" / task / "stage-envelope.json"
    roster: list[dict] = []
    output = _put(output_path, _canonical(_wrapped({"scalar": 0}, "output")))
    roster.append(output)
    output_material = {
        "materialization_id": "material.output",
        "relative_path": str(output_path.relative_to(project)),
        "physical_sha256": output["sha256"],
        "size_bytes": output["bytes"],
        "storage_root_id": "synthetic-root",
    }
    stage = _put(
        stage_path,
        _canonical(
            _wrapped(
                {
                    "scientific_product": _wrapped(
                        {
                            "object_schema": "empirical-lawhood/testing/fixtures/response-parent-custody/output",
                            "object_version": "1.0.0",
                            "object_fingerprint": output["sha256"],
                        },
                        "identity",
                    )
                },
                "stage-envelope",
            )
        ),
    )
    roster.append(stage)
    stage_material = {
        "materialization_id": "material.stage",
        "relative_path": str(stage_path.relative_to(project)),
        "physical_sha256": stage["sha256"],
        "size_bytes": stage["bytes"],
        "storage_root_id": "synthetic-root",
    }
    receipt = _put(
        receipt_path,
        _canonical(
            _wrapped(
                {
                    "task_id": task,
                    "receipt_id": receipt_id,
                    "run_id": "run1",
                    "operational_status": "SUCCEEDED",
                    "output_materializations": [
                        _wrapped(output_material, "material"),
                        _wrapped(stage_material, "material"),
                    ],
                },
                "receipt",
            )
        ),
    )
    if not omit_receipt:
        roster.append(receipt)
    receipt_material = {
        "materialization_id": "material.receipt",
        "relative_path": str(receipt_path.relative_to(project)),
        "physical_sha256": receipt["sha256"],
        "size_bytes": receipt["bytes"],
        "storage_root_id": "synthetic-root",
    }
    for path, content, material, kind in (
        (output_path, output, output_material, "output"),
        (stage_path, stage, stage_material, "stage"),
        (receipt_path, receipt, receipt_material, "receipt"),
    ):
        marker_path = (
            run
            / (
                ".publication-batches"
                if kind == "output"
                else "receipts/.publication-batches"
            )
            / f"{kind}.commit.json"
        )
        marker = _put(
            marker_path, _canonical({"member": material["materialization_id"]})
        )
        roster.append(marker)
        publication = {
            "commit_marker_relative_path": str(marker_path.relative_to(project)),
            "commit_marker_sha256": marker["sha256"],
            "commit_marker_size_bytes": marker["bytes"],
            "members": [
                {
                    "materialization_id": material["materialization_id"],
                    "physical_sha256": material["physical_sha256"],
                }
            ],
        }
        material_manifest = {
            "materialization": _wrapped(material, "material"),
            "logical": {"content_sha256": content["sha256"]},
            "publication": publication,
        }
        roster.append(
            _put(Path(str(path) + ".manifest.json"), _canonical(material_manifest))
        )
    terminal_path = run / "recovery" / "wave" / "resource-terminal.json"
    terminal = {
        "run_id": "run1",
        "operational_status": "SUCCEEDED",
        "attempts": [
            _wrapped(
                {
                    "task_id": task,
                    "disposition": "SUCCEEDED",
                    "receipt_id": receipt_id,
                    "receipt_sha256": receipt["sha256"],
                },
                "attempt",
            )
        ],
    }
    roster.append(_put(terminal_path, _canonical(_wrapped(terminal, "terminal"))))
    manifest_path = root / "analysis" / "input-manifest.json"
    _put(manifest_path, _canonical(sorted(roster, key=lambda item: item["path"])))
    parent = ResponseSourceParentIdentity(
        "empirical-lawhood/testing/fixtures/response-parent-custody/output",
        "1.0.0",
        output["sha256"],
    )
    base = ResponseParentTargetGrant(
        "target-custody",
        "CUSTODY",
        "response-composition",
        parent,
        sha256(manifest_path.read_bytes()).hexdigest(),
        "project",
        str(terminal_path.relative_to(project)),
        str(output_path.relative_to(project)),
        str(stage_path.relative_to(project)),
        "synthetic-root",
        ("synthetic.source-qualification.prepared.r000",),
        "EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE",
    )
    locators = tuple(tmp_path / name for name in ("custody", "reveal", "analysis"))
    grants = {
        locators[0]: base,
        locators[1]: replace(base, grant_id="target-reveal", action="REVEAL"),
        locators[2]: replace(base, grant_id="target-analysis", action="ANALYSIS"),
    }

    class Store:
        def resolve_grant(self, locator: Path) -> ResponseParentTargetGrant | None:
            return grants.get(locator)

    kwargs = {
        "route": "response-composition",
        "source_root": root,
        "parent_manifest": manifest_path,
        "custody": locators[0],
        "reveal_record": locators[1],
        "analysis_record": locators[2],
        "authority_store": Store(),
        "source_schema": "empirical-lawhood/testing/fixtures/response-parent-custody/native-input-manifest",
    }
    return kwargs, {
        "grants": grants,
        "receipt_path": receipt_path,
        "output_path": output_path,
        "stage_path": stage_path,
    }


def test_synthetic_complete_inventory_replays_to_nonpromotable_witness(
    tmp_path: Path,
) -> None:
    kwargs, details = _fixture(tmp_path)
    witness = import_response_parent_custody(**kwargs)
    assert (
        witness.source_manifest_sha256
        == sha256(kwargs["parent_manifest"].read_bytes()).hexdigest()
    )
    assert witness.original_root_ids == ("synthetic.source-qualification.prepared.r000",)
    assert set(witness.artifact_sha256s) == {
        sha256(details["output_path"].read_bytes()).hexdigest(),
        sha256(details["stage_path"].read_bytes()).hexdigest(),
    }
    assert witness.receipt_sha256s == (
        sha256(details["receipt_path"].read_bytes()).hexdigest(),
    )
    assert witness.evidence_role == "EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE"


def test_missing_store_wrong_scope_and_unresolved_grant_refuse_before_manifest_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs, details = _fixture(tmp_path)
    def guarded_read(*args, **kwargs) -> bytes:
        raise AssertionError("source read before target grants")

    monkeypatch.setattr(response_parent_custody, "read_bounded_contained", guarded_read)
    with pytest.raises(ValueError, match="TARGET_AUTHORITY_STORE_REQUIRED"):
        import_response_parent_custody(**(kwargs | {"authority_store": None}))
    details["grants"][kwargs["reveal_record"]] = None
    with pytest.raises(ValueError, match="TARGET_GRANT_UNRESOLVED"):
        import_response_parent_custody(**kwargs)
    details["grants"][kwargs["reveal_record"]] = replace(
        details["grants"][kwargs["custody"]],
        grant_id="target-reveal",
        action="REVEAL",
        source_manifest_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="TARGET_GRANT_SCOPE_MISMATCH"):
        import_response_parent_custody(**kwargs)


def test_missing_receipt_and_changed_bytes_refuse(tmp_path: Path) -> None:
    kwargs, _ = _fixture(tmp_path / "missing", omit_receipt=True)
    with pytest.raises(ValueError, match="RECEIPT_ROSTER_INCOMPLETE"):
        import_response_parent_custody(**kwargs)
    kwargs, details = _fixture(tmp_path / "changed")
    details["output_path"].write_bytes(b"changed outcome")
    with pytest.raises(ValueError, match="IDENTITY_BYTES_MISMATCH"):
        import_response_parent_custody(**kwargs)


def test_changed_original_root_and_reused_grant_refuse(tmp_path: Path) -> None:
    kwargs, details = _fixture(tmp_path)
    custody = details["grants"][kwargs["custody"]]
    for locator in details["grants"]:
        details["grants"][locator] = replace(
            details["grants"][locator], original_root_ids=("synthetic.source-qualification.prepared.r001",)
        )
    with pytest.raises(ValueError, match="ORIGINAL_ROOTS_MISMATCH"):
        import_response_parent_custody(**kwargs)
    details["grants"][kwargs["custody"]] = custody
    details["grants"][kwargs["reveal_record"]] = replace(custody, action="REVEAL")
    with pytest.raises(ValueError, match="TARGET_GRANT_SCOPE_MISMATCH"):
        import_response_parent_custody(**kwargs)


def test_parent_identity_and_symlink_member_refuse(tmp_path: Path) -> None:
    kwargs, details = _fixture(tmp_path / "identity")
    for locator, grant in tuple(details["grants"].items()):
        details["grants"][locator] = replace(
            grant,
            parent=replace(grant.parent, physical_sha256="b" * 64),
        )
    with pytest.raises(ValueError, match="IDENTITY_DIGEST_MISMATCH"):
        import_response_parent_custody(**kwargs)

    kwargs, details = _fixture(tmp_path / "symlink")
    output = details["output_path"]
    outside = tmp_path / "outside.json"
    outside.write_bytes(output.read_bytes())
    output.unlink()
    output.symlink_to(outside)
    with pytest.raises(ValueError, match="PARENT_MEMBER_INVALID"):
        import_response_parent_custody(**kwargs)


def test_stage_product_must_bind_the_parent(tmp_path: Path) -> None:
    kwargs, details = _fixture(tmp_path)
    wrong_stage = str(
        details["output_path"].relative_to(kwargs["source_root"] / "project")
    )
    for locator, grant in tuple(details["grants"].items()):
        details["grants"][locator] = replace(grant, stage_envelope_relative=wrong_stage)
    with pytest.raises(ValueError, match="STAGE_PRODUCT_MISMATCH"):
        import_response_parent_custody(**kwargs)
