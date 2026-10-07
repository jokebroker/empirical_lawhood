"""Current input import publications and exact retained result bindings."""

from empirical_lawhood.adapters.methods.preparation_applicability.issued_inputs import PreparationApplicabilityImportReceipt, PreparationApplicabilityPublishedInput
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.records import ConstructedPreparationReport
from empirical_lawhood.adapters.methods.preparation_applicability.panel_inputs import PreparationApplicabilityPanelPublication
from empirical_lawhood.adapters.simulators.preparation_applicability.records import PreparationApplicabilityPrefix, PreparationApplicabilityPanel
from empirical_lawhood.adapters.methods.preparation_applicability.records import PreparationApplicabilityMeasuredRoot
from empirical_lawhood.adapters.methods.preparation_applicability.seals import PreparationApplicabilityLowerSeal
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes, MAX_ARTIFACT_MANIFEST_BYTES
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore, decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import validate_relative_locator, validate_stable_id
from empirical_lawhood.runtime.artifacts import ArtifactWriteRequest, ArtifactProfile
from empirical_lawhood.api.preparation_result_access import authenticate_preparation_result_access
from empirical_lawhood.adapters.methods.preparation_applicability.result_access import PREPARATION_PROTECTED_OUTCOMES


def publish_preparation_input(*, record, key: str, logical_artifact_id: str, relative_root: str,
                              receipt_id: str, artifact_writer: ExternalArtifactPlane):
    """Import original-law/exposure bytes; creates no scheduler or qualification receipt."""
    if not isinstance(artifact_writer, ExternalArtifactPlane):
        raise TypeError("input import requires the actual guarded external artifact plane")
    validate_stable_id(logical_artifact_id); validate_stable_id(receipt_id)
    validate_relative_locator(relative_root)
    if key not in ("lower", "exposure", "conformance"):
        raise ValueError("Q transfer cannot use an import publication")
    from empirical_lawhood.adapters.methods.finite_response_law.original_f import OriginalFiniteResponseLaw
    from empirical_lawhood.adapters.methods.preparation_applicability.issued_inputs import PreparationApplicabilityExposureInspection
    from empirical_lawhood.adapters.methods.preparation_applicability.conformance import PreparationApplicabilityNativeConformance
    if not isinstance(record, {"lower": OriginalFiniteResponseLaw,"exposure": PreparationApplicabilityExposureInspection,"conformance": PreparationApplicabilityNativeConformance}[key]):
        raise TypeError("input import changed its code-owned record type")
    def publish(value, name, logical_id):
        result=artifact_writer.write(ArtifactWriteRequest(logical_artifact_id=logical_id,
            relative_path=f"{relative_root}/{name}",payload_schema=value.SCHEMA,profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/json",publication_scope_id=f"{receipt_id}.import-publication",
            publication_scope_relative_root=relative_root,payload=value.canonical_bytes(),
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,parent_visibility_ceilings=(),
            outcome_access=OutcomeAccess.OUTCOME_BLIND))
        path=artifact_writer.root.resolve(result.manifest_materialization.relative_path,for_write=False)
        manifest=decode_artifact_manifest(read_bounded_bytes(path,maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES))
        artifact_writer.verify_manifest(manifest)
        return manifest
    manifest=publish(record,"input.canonical.json",logical_artifact_id)
    receipt=PreparationApplicabilityImportReceipt(receipt_id,key,manifest)
    receipt_manifest=publish(receipt,"import-receipt.canonical.json",receipt_id)
    return PreparationApplicabilityPublishedInput(key,record,manifest,receipt,receipt_manifest)


def _retained_output(*, plane, receipt, kind, result_access,
                     access_authenticator=authenticate_preparation_result_access):
    outputs = tuple(value for value in receipt.output_logical_artifacts if value.payload_schema == kind.SCHEMA)
    if len(outputs) != 1:
        raise ValueError("retained task omits its exact single scientific output")
    if outputs[0].outcome_access not in PREPARATION_PROTECTED_OUTCOMES:
        raise PermissionError("retained preparation output changes its protected issued outcome class")
    access_authenticator(context=result_access,receipt=receipt,logical=outputs[0],writer=plane)
    manifests=[]
    for materialization in receipt.output_materializations:
        path=plane.root.resolve(f"{materialization.relative_path}.manifest.json",for_write=False)
        manifest=decode_artifact_manifest(read_bounded_bytes(path,maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES))
        if manifest.logical.payload_schema==kind.SCHEMA:
            manifests.append(manifest)
    if len(manifests)!=1:
        raise ValueError("retained task omits its exact single scientific output")
    manifest=manifests[0]
    if manifest.logical not in receipt.output_logical_artifacts or manifest.materialization not in receipt.output_materializations:
        raise ValueError("retained scientific output substitutes its committed task receipt")
    plane.verify_manifest(manifest)
    path=plane.root.resolve(manifest.materialization.relative_path,for_write=False)
    record=decode_canonical_bytes(read_bounded_bytes(path,maximum_bytes=128*1024**2),kind,maximum_bytes=128*1024**2)
    if record.fingerprint()!=manifest.materialization.physical_sha256:
        raise ValueError("retained scientific output bytes differ")
    return record,manifest


def bind_constructed_preparation_q(*, run_id: str, receipt_id: str, artifact_writer: ExternalArtifactPlane, result_access):
    """Select the exact current Q report receipt; never search a latest result."""
    receipt=ExternalTaskReceiptStore(artifact_writer).read_by_receipt_id(run_id,"ap.report",receipt_id)
    if receipt is None:
        raise ValueError("selected Q report receipt is absent")
    record,manifest=_retained_output(plane=artifact_writer,receipt=receipt,kind=ConstructedPreparationReport,result_access=result_access)
    if record.phase!="Q":
        raise ValueError("selected prerequisite is not a Q constructor report")
    return PreparationApplicabilityPublishedInput("qualification",record,manifest,receipt,result_access=result_access)


def bind_ordinary_preparation_panel(*, run_id: str, root_id: str,
    receipt_ids: tuple[str,...], artifact_writer: ExternalArtifactPlane, result_access):
    """Bind prefix/assay/lower/measurement in that exact order for one root."""
    if len(receipt_ids)!=4:
        raise ValueError("ordinary panel binding requires its four exact selected task receipts")
    records=[]; manifests=[]; receipts=[]
    for task,receipt_id,kind in zip(("prefix","assay","lower","measure"),receipt_ids,
            (PreparationApplicabilityPrefix,PreparationApplicabilityPanel,PreparationApplicabilityLowerSeal,PreparationApplicabilityMeasuredRoot),strict=True):
        receipt=ExternalTaskReceiptStore(artifact_writer).read_by_receipt_id(run_id,f"ap.{task}.{root_id}",receipt_id)
        if receipt is None:
            raise ValueError("selected ordinary panel receipt is absent")
        record,manifest=_retained_output(plane=artifact_writer,receipt=receipt,kind=kind,result_access=result_access)
        records.append(record);manifests.append(manifest);receipts.append(receipt)
    return PreparationApplicabilityPanelPublication(*records,tuple(manifests),tuple(receipts),result_access)


def bind_preparation_report(*,run_id: str,receipt_id: str,constructed: bool,
    artifact_writer: ExternalArtifactPlane,result_access):
    """Read the selected ordinary or constructed terminal scientific report."""
    from empirical_lawhood.adapters.methods.preparation_applicability.readout import PreparationApplicabilityScreenReport
    from empirical_lawhood.adapters.methods.preparation_applicability.result_access import PreparationApplicabilityRetainedResult
    receipt=ExternalTaskReceiptStore(artifact_writer).read_by_receipt_id(run_id,"ap.report",receipt_id)
    if receipt is None:
        raise ValueError("selected preparation terminal report receipt is absent")
    kind=ConstructedPreparationReport if constructed else PreparationApplicabilityScreenReport
    record,manifest=_retained_output(plane=artifact_writer,receipt=receipt,kind=kind,result_access=result_access)
    return PreparationApplicabilityRetainedResult(record.canonical_bytes().decode("utf-8"),manifest,receipt,result_access)


def read_preparation_report(retained,*,artifact_writer):
    """Recheck current grants/custody before presenting full denominators/stops."""
    from empirical_lawhood.adapters.methods.preparation_applicability.readout import PreparationApplicabilityScreenReport
    kind={ConstructedPreparationReport.SCHEMA:ConstructedPreparationReport,
          PreparationApplicabilityScreenReport.SCHEMA:PreparationApplicabilityScreenReport}.get(retained.manifest.logical.payload_schema)
    if kind is None:
        raise ValueError("retained preparation result has an unsupported current report schema")
    authenticate_preparation_result_access(context=retained.result_access,receipt=retained.receipt,
        logical=retained.manifest.logical,writer=artifact_writer)
    actual=ExternalTaskReceiptStore(artifact_writer).read_by_receipt_id(retained.receipt.run_id,retained.receipt.task_id,retained.receipt.receipt_id)
    if actual!=retained.receipt:
        raise ValueError("retained preparation report changes its committed task receipt")
    artifact_writer.verify_manifest(retained.manifest)
    record=decode_canonical_bytes(retained.record_payload.encode("utf-8"),kind,maximum_bytes=16*1024**2)
    return {"phase":record.phase,"independent_roots":len(record.root_ids),"complete":record.complete,
        "disposition":record.disposition,"full_valid_counts":record.full_valid_counts,
        "support_counts":record.support_counts,"joint_counts":record.joint_counts,
        "covered_counts":record.covered_counts,"false_admissions":record.false_admissions,
        "report_sha256":record.fingerprint(),"scientific_execution_performed_by_reader":False}


def create_preparation_selection(*, root, config_id: str, phase: str, allocation,
    lower, exposure, conformance, qualification=None, constructed: bool=False):
    """Build editable current stage/source from actual published inputs and explicit seeds.

    Counts and purpose rosters belong to the fixed family. This supplies no seed,
    conformance result, exposure inspection, Q result or execution authority.
    """
    from hashlib import sha256
    from empirical_lawhood.kernel.provenance import ObjectIdentity
    from empirical_lawhood.runtime.task_records import artifact_identity
    from empirical_lawhood.adapters.methods.preparation_applicability.config import PreparationApplicabilityUpstream
    from empirical_lawhood.adapters.methods.preparation_applicability.exposure import PreparationApplicabilityExposure
    if constructed:
        from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import ConstructedPreparationStage as Stage,ConstructedPreparationSource as Source
        from empirical_lawhood.adapters.methods.constructed_preparation_applicability.records import ConstructedPreparationDesign as Design
        from empirical_lawhood.adapters.composition.constructed_preparation_applicability.selection import ConstructedPreparationApplicabilitySelection as Selection
        from empirical_lawhood.adapters.composition.constructed_preparation_applicability.authoring import SPECIFICATION_PATH
    else:
        from empirical_lawhood.adapters.methods.preparation_applicability.config import PreparationApplicabilityStage as Stage,PreparationApplicabilitySource as Source
        from empirical_lawhood.adapters.methods.preparation_applicability.records import PreparationApplicabilityDesign as Design
        from empirical_lawhood.adapters.composition.preparation_applicability.selection import PreparationApplicabilitySelection as Selection
        from empirical_lawhood.adapters.composition.preparation_applicability.authoring import SPECIFICATION_PATH
    if (lower.key!="lower" or exposure.key!="exposure" or conformance.key!="conformance"
        or bool(qualification is not None)!=(constructed and phase=="E")):
        raise ValueError("selection requires its actual closed phase input publications")
    specification=read_bounded_bytes(root/SPECIFICATION_PATH,maximum_bytes=2*1024**2)
    lock=read_bounded_bytes(root/"uv.lock",maximum_bytes=4*1024**2)
    source=Source(sha256(specification).hexdigest(),sha256(lock).hexdigest(),conformance.record.implementation,
        ObjectIdentity.from_record(conformance.receipt.receipt_id,conformance.receipt),
        python_version=conformance.record.python_version,numpy_version=conformance.record.numpy_version)
    published=(lower,) if qualification is None else (lower,qualification)
    upstream=tuple(PreparationApplicabilityUpstream(value.key,artifact_identity(value.manifest,role="current-preparation-input"),
        ObjectIdentity.from_record(value.receipt.receipt_id,value.receipt),
        value.receipt.run_id if hasattr(value.receipt,"run_id") else value.receipt.receipt_id) for value in published)
    inspection=exposure.record
    checked=PreparationApplicabilityExposure(ObjectIdentity.from_record(inspection.inspection_id,inspection),
        ObjectIdentity.from_record(exposure.receipt.receipt_id,exposure.receipt),inspection.excluded_unit_ids,inspection.excluded_seed_ids)
    stage=Stage(config_id,Design(),phase,allocation,upstream,checked,ObjectIdentity.from_record(f"{config_id}.source",source))
    return Selection(f"{config_id}.selection",stage,source,tuple(sorted((*published,exposure,conformance),key=lambda value:value.key)))


def inspect_preparation_exposure(*, prior_inputs, inspection_id: str, original_f, artifact_writer=None):
    """Inspect exact supplied saved-current banks plus the known public seed recipes.

    prior_inputs is an ordered tuple of (manifest_path,arrays_path,allocation_path).
    Paths stay operational. Exact allocation and source identities enter the new
    inspection. An unavailable declared bank must be supplied, not relabelled.
    """
    from empirical_lawhood.api.finite_operands import saved_matrix_arrays_import
    from empirical_lawhood.adapters.methods.preparation_applicability.issued_inputs import PreparationApplicabilityExposureInspection
    from empirical_lawhood.adapters.methods.preparation_applicability.exposure import public_historical_exposure,effective_seed_ids
    known_units,known_seeds=public_historical_exposure()
    units=set(known_units); seeds=set(known_seeds); allocations=[]
    sources=[original_f.original_payload]
    for manifest_path,arrays_path,allocation_path in prior_inputs:
        manifest,allocation,_=saved_matrix_arrays_import(manifest_path,arrays_path=arrays_path,allocation_path=allocation_path,artifact_writer=artifact_writer)
        allocations.append(allocation); sources.append(manifest.array_artifact)
        units.update(root.root_id for root in allocation.roots);seeds.update(effective_seed_ids(allocation))
    return PreparationApplicabilityExposureInspection(inspection_id,tuple(sorted(units)),tuple(sorted(seeds)),tuple(sources),tuple(allocations))


def prepare_preparation_allocation(*, allocation_id: str, cohort_namespace: str,
    phase: str, master_seed: int, constructed: bool=False, proposed_unrun: bool=False):
    """Derive the closed full-size explicit seed roster; assign no freshness claim.

    Recipe v1 uses SHA256 of canonical tagged tuples. The full digest is the
    ordinary passive seed; other purposes and bootstrap consume its first128
    bits. Constructed probes are fixed conditioning, never independent draws.
    """
    from hashlib import sha256
    from empirical_lawhood.kernel.matrix_inputs import MATRIX_NATIVE_PURPOSES,MatrixAllocation,MatrixRootAllocation,MatrixPurposeSeed
    from empirical_lawhood.kernel.serialization import canonical_json_bytes
    from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import FIXED_PROBE_SEED
    validate_stable_id(allocation_id);validate_stable_id(cohort_namespace)
    if (type(master_seed) is not int or not 0<=master_seed<2**128
        or type(constructed) is not bool or type(proposed_unrun) is not bool
        or phase not in (("Q","E") if constructed else ("D","E"))):
        raise ValueError("allocation requires its explicit128-bit master and closed family phase")
    count=(8 if phase=="Q" else 32) if constructed else (32 if phase=="D" else 64)
    def seed(index,purpose):
        return int.from_bytes(sha256(canonical_json_bytes(("empirical-lawhood/preparation-applicability/allocation-v1",master_seed,cohort_namespace,phase,index,purpose))).digest(),"big")
    roots=[]
    for index in range(count):
        values=[MatrixPurposeSeed(purpose,seed(index,purpose)>>128) for purpose in MATRIX_NATIVE_PURPOSES]
        conditioning=()
        if constructed:
            conditioning=(MatrixPurposeSeed("passive-probes",FIXED_PROBE_SEED,"PCG64DXSM"),)
        else:
            values.append(MatrixPurposeSeed("passive-probes",seed(index,"passive-probes"),"PCG64DXSM"))
        roots.append(MatrixRootAllocation(f"{cohort_namespace}.{phase.lower()}.r{index:03d}",
            "constructed" if constructed else "q2",tuple(sorted(values,key=lambda value:value.purpose_id)),conditioning_seeds=conditioning))
    return MatrixAllocation(allocation_id,tuple(roots),bootstrap_seed=seed(-1,"bootstrap")>>128,
        exposure="PROPOSED_UNRUN" if proposed_unrun else "EXPOSED_DEVELOPMENT_NONPROMOTABLE")
