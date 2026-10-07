"""Ordinary32D/64E panel exports and their two exposed development screens."""

import json
from pathlib import Path
import numpy as np

from empirical_lawhood.api.finite_operands import saved_matrix_arrays_export, saved_matrix_arrays_import, _read_selected
from empirical_lawhood.api.authoring_output import report_incomplete_output
from empirical_lawhood.adapters.methods.preparation_applicability.measurement import reduce_panel, validity, service, requests
from empirical_lawhood.adapters.methods.preparation_applicability.root_measurement import measure_root
from empirical_lawhood.adapters.methods.preparation_applicability.readout import report
from empirical_lawhood.adapters.methods.preparation_applicability.headroom import opportunity, prefix_forecast
from empirical_lawhood.adapters.methods.preparation_applicability.config import PreparationApplicabilityStage, preparation_run_id
from empirical_lawhood.adapters.methods.preparation_applicability.readout import PreparationApplicabilityScreenReport
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.task_records import artifact_identity
from empirical_lawhood.api.preparation_result_access import authenticate_preparation_result_access


def ordinary_preparation_operands_export(*, stage, original_f, panels, destination: Path, artifact_writer):
    """Authenticate and reduce all assigned ordinary native panels without fitting."""
    if not isinstance(artifact_writer,ExternalArtifactPlane):
        raise TypeError("ordinary panel export requires the actual guarded custody plane")
    if ".." in destination.parts or any(path.is_symlink() for path in (destination,*destination.parents)):
        raise ValueError("ordinary panel export refuses parent traversal and symbolic links")
    destination=destination.absolute()
    guarded=Path(artifact_writer.root.contract.canonical_path)
    if not destination.is_relative_to(guarded):
        raise ValueError("ordinary panel export must stay under its selected guarded custody root")
    artifact_writer.root.resolve(destination.relative_to(guarded).as_posix(),for_write=True)
    if not destination.is_dir() or any(destination.iterdir()):
        raise FileExistsError("ordinary operands require an explicitly selected empty directory")
    if tuple(value.prefix.root_id for value in panels)!=stage.root_ids:
        raise ValueError("ordinary panel export changes its complete assigned independent-root roster")
    if stage.upstream[0].artifact.sha256!=original_f.fingerprint():
        raise ValueError("ordinary panels substitute the original unchanged current F wrapper")
    arrays={"prefix":[],"handoff":[],"maxima":[],"work":[]}
    if stage.phase=="E":
        arrays["success"]=[]
    measured=[]
    for index,value in enumerate(panels):
        value.authenticate(artifact_writer,ExternalTaskReceiptStore(artifact_writer),result_access_authenticator=authenticate_preparation_result_access)
        allocation=stage.allocation.roots[index]
        if (any(phase.allocation!=allocation or phase.source_sha256!=stage.source.object_fingerprint for phase in (*value.prefix.phases,*value.panel.phases))
            or any(receipt.run_id!=preparation_run_id(stage) for receipt in value.receipts)):
            raise ValueError("ordinary panel substitutes its actual source/allocation/run")
        reduced=reduce_panel(value.prefix,value.panel)
        if reduced is None:
            raise ValueError("UNEVALUABLE: ordinary screen retains an incomplete assigned panel; inspect its receipts")
        z,y,work=reduced
        row=measure_root(value.prefix,value.panel,original_f,stage.phase,allocation.seed_for("requests"),value.seal)
        if row!=value.measurement:
            raise ValueError("ordinary screen substitutes its independent readout")
        measured.append(row)
        maxima,_,_,mean,width,support=validity(original_f,z,y,work)
        arrays["prefix"].append(np.asarray(value.prefix.features,dtype=float).T)
        arrays["handoff"].append(z); arrays["maxima"].append(maxima); arrays["work"].append(work)
        if stage.phase=="E":
            arrays["success"].append(service(mean,width,support,y,work,*requests(allocation.seed_for("requests")))[1])
    summary=report(stage,tuple(measured))
    with report_incomplete_output(destination):
        for name,record in (("stage.canonical.json",stage),("screen-report.canonical.json",summary)):
            with (destination/name).open("xb") as handle:
                handle.write(record.canonical_bytes())
        result=saved_matrix_arrays_export(destination,operand_id=f"{stage.config_id}.screen-operands",
            allocation=stage.allocation,arrays={name:np.stack(rows) for name,rows in arrays.items()},
            producers=(ObjectIdentity.from_record(stage.config_id,stage),ObjectIdentity.from_record(f"{stage.config_id}.report",summary)),
            source_artifacts=tuple(artifact_identity(manifest,role="ordinary-native-panel") for value in panels for manifest in value.manifests),
            source_receipts=tuple(ObjectIdentity.from_record(receipt.receipt_id,receipt) for value in panels for receipt in value.receipts),
            original_f=original_f.identity)
    return result


def ordinary_preparation_screens(*, development_directory: Path,evaluation_directory: Path,
                                  original_f,output_dir: Path):
    """Run finite-menu headroom on E64 and whole-root LOO nomination on D32+E64."""
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("ordinary screen output must use a fresh directory")
    retained=[]
    for directory,count,phase in ((development_directory,32,"D"),(evaluation_directory,64,"E")):
        manifest,allocation,arrays=saved_matrix_arrays_import(directory/"arrays.canonical.json",arrays_path=directory/"arrays.npz",allocation_path=directory/"allocation.canonical.json")
        stage=decode_canonical_bytes(_read_selected(directory/"stage.canonical.json",maximum_bytes=4*1024**2),PreparationApplicabilityStage,maximum_bytes=4*1024**2)
        summary=decode_canonical_bytes(_read_selected(directory/"screen-report.canonical.json",maximum_bytes=4*1024**2),PreparationApplicabilityScreenReport,maximum_bytes=4*1024**2)
        if (manifest.original_f!=original_f.identity or len(allocation.roots)!=count
            or stage.phase!=phase or stage.allocation!=allocation or summary.phase!=phase
            or not summary.complete or summary.root_ids!=stage.root_ids
            or summary.allocation_sha256!=allocation.fingerprint()
            or summary.source_sha256!=stage.source.object_fingerprint
            or summary.lower_sha256!=original_f.fingerprint()
            or ObjectIdentity.from_record(stage.config_id,stage) not in manifest.producers
            or ObjectIdentity.from_record(f"{stage.config_id}.report",summary) not in manifest.producers
            or any(root.cohort!="q2" for root in allocation.roots)
            or arrays["prefix"].shape!=(count,24,2) or arrays["handoff"].shape!=(count,3,24,2)
            or arrays["maxima"].shape!=(count,3,7) or arrays["work"].shape!=(count,3,2)
            or set(arrays)!={"prefix","handoff","maxima","work",*(('success',) if phase=='E' else ())}
            or not np.array_equal(np.asarray(summary.handoff,dtype=float).reshape(count,3,24,2),arrays["handoff"])
            or summary.full_valid_counts!=tuple(map(int,(arrays["maxima"]<=1).all(axis=2).sum(axis=0)))
            or summary.support_counts!=tuple(map(int,(arrays["maxima"][:,:,0]<=1).sum(axis=0)))):
            raise ValueError("ordinary screen changes D32/E64 roles, source or full panel census")
        if phase=="E":
            success=arrays["success"]
            if success.shape!=(64,3,256,2) or success.dtype!=np.bool_:
                raise ValueError("ordinary evaluation changes its complete paired request census")
            joint=success.all(axis=3).sum(axis=2)
            if (summary.joint_counts!=tuple(map(int,joint.sum(axis=0)))
                or summary.covered_counts!=tuple(map(int,(joint*(arrays["maxima"]<=1).all(axis=2)).sum(axis=0)))):
                raise ValueError("ordinary screen changes its actual joint/covered service counts")
        retained.append((manifest,allocation,arrays))
    development,evaluation=retained
    if set(root.root_id for root in development[1].roots)&set(root.root_id for root in evaluation[1].roots):
        raise ValueError("ordinary D/E screens repeat independent roots")
    from empirical_lawhood.adapters.methods.preparation_applicability.exposure import effective_seed_ids
    if set(effective_seed_ids(development[1]))&set(effective_seed_ids(evaluation[1])):
        raise ValueError("ordinary D/E screens repeat effective source streams")
    d,e=development[2],evaluation[2]
    face=float(original_f.center[17]+6*original_f.scale[17])
    prefix=np.concatenate((d["prefix"][...,0],e["prefix"][...,0]))
    handoff=np.concatenate((d["handoff"][:,0,17,0],e["handoff"][:,0,17,0]))
    result={"ordinary_independent_roots":96,"development_roots":32,"evaluation_roots":64,
        "original_f_sha256":original_f.fingerprint(),"support_screen":{
            "development_support_counts":(d["maxima"][:,:,0]<=1).sum(axis=0).tolist(),
            "evaluation_support_counts":(e["maxima"][:,:,0]<=1).sum(axis=0).tolist()},
        "headroom":opportunity(e["maxima"],e["success"]),
        "causal_prefix_forecast":prefix_forecast(prefix,handoff,face),
        "input_manifest_sha256s":[value[0].fingerprint() for value in retained],
        "ceiling":"EXPOSED_DEVELOPMENT_NONPROMOTABLE","follow_on_status":"UNENTERED"}
    output_dir.mkdir(parents=True,exist_ok=False)
    with report_incomplete_output(output_dir):
        with (output_dir/"ordinary-screens.json").open("xb") as handle:
            handle.write((json.dumps(result,sort_keys=True,allow_nan=False)+"\n").encode())
    return result
