"""Guarded saved-current readiness; no native acquisition or scientific authority."""

import json
from collections import Counter
from math import isfinite
from pathlib import Path

import numpy as np

from empirical_lawhood.api.authoring_handoff import write_exclusive_record
from empirical_lawhood.api.authoring_output import report_incomplete_output
from empirical_lawhood.api.finite_operands import saved_matrix_arrays_import, saved_matrix_arrays_export, preparation_bridge_fits
from empirical_lawhood.adapters.methods.finite_response_law.retained_entry_readiness.current_input import current_readiness
from empirical_lawhood.kernel.provenance import ObjectIdentity


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        # Exact nonfinite numerical operands remain in the authenticated arrays.
        return {"nonfinite": "nan" if np.isnan(value) else "positive-infinity" if value > 0 else "negative-infinity"}
    return value


def current_preparation_readiness(*, manifest_path: Path, arrays_path: Path,
    allocation_path: Path, original_f, output_dir: Path, report_id: str, artifact_writer, project_root: Path) -> dict[str, object]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("readiness output must use a fresh directory")
    from empirical_lawhood.api.matrix_numerical_provenance import capture_matrix_provenance, preflight_matrix_output_directory
    from empirical_lawhood.infrastructure.retained_analysis import publish_retained_analysis
    from empirical_lawhood.kernel.references import ArtifactIdentity
    from empirical_lawhood.runtime.retained_analysis import RetainedAnalysisMember
    from hashlib import sha256
    preflight_matrix_output_directory(directory=output_dir, artifact_writer=artifact_writer, project_root=project_root)
    provenance = capture_matrix_provenance(project_root=project_root)
    manifest, allocation, arrays = saved_matrix_arrays_import(manifest_path, arrays_path=arrays_path, allocation_path=allocation_path, artifact_writer=artifact_writer)
    report, store, result, witnesses, checked = current_readiness(manifest=manifest,
        allocation=allocation, arrays=arrays, lower=original_f, bridge_fits=preparation_bridge_fits(arrays), report_id=report_id)
    output_dir.mkdir(parents=True, exist_ok=False)
    provenance_raw = provenance.canonical_bytes()
    with (output_dir / "numerical-provenance.canonical.json").open("xb") as handle:
        handle.write(provenance_raw)
    members = [RetainedAnalysisMember("numerical-provenance.canonical.json", ArtifactIdentity(
        "numerical.producing-provenance", "readiness-numerical-provenance", provenance.SCHEMA,
        sha256(provenance_raw).hexdigest(), "application/json", len(provenance_raw)))]
    producer = ObjectIdentity.from_record(report_id, report)
    with report_incomplete_output(output_dir) as progress:
        progress.stage = "saved readiness operands"
        names = sorted(store.arrays)
        shards = []
        for start in range(0, len(names), 256):
            shard = output_dir / f"arrays-{start//256:02d}"
            shard.mkdir()
            array_members = {name: np.asarray(store.arrays[name]) for name in names[start:start+256]}
            exported = saved_matrix_arrays_export(shard, operand_id=f"{report_id}.arrays-{start//256:02d}",
                allocation=allocation, arrays=array_members, producers=(producer,),
                source_artifacts=(manifest.array_artifact,), source_receipts=manifest.source_receipts,
                original_f=original_f.identity,
                nonfinite_members=tuple(name for name, value in array_members.items() if not np.isfinite(value).all()))
            shards.append(exported)
            prefix = f"arrays-{start//256:02d}/"
            members.append(RetainedAnalysisMember(prefix + "arrays.npz", exported.array_artifact))
            for name, record in (("arrays.canonical.json", exported), ("allocation.canonical.json", allocation)):
                raw = record.canonical_bytes()
                members.append(RetainedAnalysisMember(prefix + name, ArtifactIdentity(
                    f"{report_id}.shard-{start//256:02d}.{name}", "readiness-saved-member", record.SCHEMA,
                    sha256(raw).hexdigest(), "application/json", len(raw))))
        progress.stage = "readiness report"
        document = {"result": result, "fits": store.records, "fit_counts": dict(Counter(row["kind"] for row in store.records)), "compatibility": witnesses,
                    "verification": checked, "array_shards": [value.fingerprint() for value in shards]}
        with (output_dir / "readiness-detail.json").open("xb") as handle:
            handle.write((json.dumps(_json_value(document), sort_keys=True, allow_nan=False)+"\n").encode())
        # Completion marker follows every required saved operand and detail.
        write_exclusive_record(output_dir, "readiness-report.canonical.json", report)
        for name, schema in (("readiness-report.canonical.json", report.SCHEMA),
                             ("readiness-detail.json", "empirical-lawhood/preparation-readiness/detail")):
            raw = (output_dir / name).read_bytes()
            members.append(RetainedAnalysisMember(name, ArtifactIdentity(
                f"{report_id}.{name}", "readiness-result-member", schema,
                sha256(raw).hexdigest(), "application/json", len(raw))))
        publish_retained_analysis(directory=output_dir, artifact_writer=artifact_writer,
            completion_id=f"{report_id}.completion", members=tuple(members))
    return {"report_sha256": report.fingerprint(), "nomination": report.nomination,
        "known_failure_cells": report.known_failure_cells, "follow_on_status": "UNENTERED",
        "output_dir": str(output_dir), "native_calls": 0, "issue_calls": 0}
