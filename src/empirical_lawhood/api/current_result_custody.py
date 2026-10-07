"""Current issued-plan and complete receipt joins before scientific output reads."""

from dataclasses import fields
from datetime import UTC, datetime
from hashlib import sha256
from typing import Literal

from empirical_lawhood.api.codecs import MAX_CAMPAIGN_PACKAGE_BYTES, loads_campaign_package_authoring
from empirical_lawhood.api.models import EnvelopeExperimentPackage, ExperimentPackage
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.infrastructure.bounded_io import MAX_ARTIFACT_MANIFEST_BYTES, MAX_RUNTIME_PLAN_JSON_BYTES, read_bounded_bytes
from empirical_lawhood.infrastructure.study_issue import ExternalIssuedStudyPublisher, ExternalStudyOperationAuthorityStore, PROGRAMME_ISSUE_GRANTEE_ID, PROGRAMME_EXECUTION_GRANTEE_ID, PROGRAMME_REVEAL_GRANTEE_ID
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore, decode_artifact_manifest
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, require_study_authority
from empirical_lawhood.runtime.artifacts import ArtifactProfile, CanonicalTaskReceipt, LogicalArtifactIdentity
from empirical_lawhood.runtime.compiler import compile_run_plan, lower_run_plan
from empirical_lawhood.runtime.plans import EnvelopeExecutionPlan, EnvelopeRunPlan, ExecutionPlan, RunPlan


def _authenticate_protected_current_result(*, issued_study: ObjectIdentity,
        execution_authority: ObjectIdentity, reveal_authority: ObjectIdentity,
        receipt: CanonicalTaskReceipt, logical: LogicalArtifactIdentity,
        writer: ExternalArtifactPlane, family: Literal['rc', 'preparation']
        ) -> tuple[ExecutionPlan | EnvelopeExecutionPlan, ExperimentPackage | EnvelopeExperimentPackage]:
    """Share current package/grant/source checks; reuse no authorization across calls.

    Family wrappers own protected-role selection, their context/run checks and
    public return values. These two closed adaptations preserve their refusal
    codes while keeping the actual authentication sequence in one owner.
    """
    if family == 'rc':
        prefixes = ('simulator-morphism-challenges.',)
        codes = ('RC_RESULT_CURRENT_PACKAGE_REQUIRED', 'RC_RESULT_ISSUED_RUN_IDENTITY_MISMATCH',
                 'RC_RESULT_AUTHORITY_IDENTITY_MISMATCH', 'RC_RESULT_CURRENT_ISSUED_SOURCE_MISMATCH',
                 'RC_RESULT_CURRENT_ISSUED_TASK_OUTPUT_MISMATCH')
    elif family == 'preparation':
        prefixes = ('preparation-applicability.', 'constructed-preparation-applicability.')
        codes = ('PREPARATION_RESULT_CURRENT_PACKAGE_REQUIRED', 'PREPARATION_RESULT_ISSUED_RUN_IDENTITY_MISMATCH',
                 'PREPARATION_RESULT_STORED_GRANT_MISMATCH', 'PREPARATION_RESULT_CURRENT_ISSUED_SOURCE_MISMATCH',
                 'PREPARATION_RESULT_ISSUED_TASK_OUTPUT_MISMATCH')
    else:
        raise ValueError('Unsupported protected current result family')
    relative = f'runs/{receipt.run_id}/plans/campaign-package.json'
    manifest = decode_artifact_manifest(read_bounded_bytes(writer.root.resolve(relative + '.manifest.json', for_write=False),
        maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES))
    if (manifest.materialization.relative_path != relative
        or manifest.materialization.size_bytes > MAX_CAMPAIGN_PACKAGE_BYTES
        or manifest.logical.logical_artifact_id != f'campaign-package.{receipt.run_id}'
        or manifest.logical.payload_schema not in (ExperimentPackage.SCHEMA, EnvelopeExperimentPackage.SCHEMA)
        or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
        or manifest.logical.media_type != 'application/json'):
        raise PermissionError(codes[0])
    writer.verify_manifest(manifest)
    raw = read_bounded_bytes(writer.root.resolve(relative, for_write=False), maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES)
    package = loads_campaign_package_authoring(raw.decode('utf-8'), media_type='application/json')
    if (not isinstance(package, (ExperimentPackage, EnvelopeExperimentPackage))
        or package.run_plan_id != receipt.run_id or package.canonical_bytes() != raw
        or package.SCHEMA != manifest.logical.payload_schema
        or package.fingerprint() != manifest.logical.content_sha256
        or ObjectIdentity.from_record(package.issued_study.issue_id, package.issued_study) != issued_study
        or ObjectIdentity.from_record(package.execution_authority.authority_id, package.execution_authority) != execution_authority):
        raise PermissionError(codes[1])
    store = ExternalStudyOperationAuthorityStore(writer)
    execution = store.load(execution_authority.object_id)
    reveal = store.load(reveal_authority.object_id)
    if execution != package.execution_authority or ObjectIdentity.from_record(reveal.authority_id, reveal) != reveal_authority:
        raise PermissionError(codes[2])
    now = datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    require_study_authority(execution, kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=issued_study, prerequisite_authority=ObjectIdentity.from_record(package.scientific_approval.authorization_id, package.scientific_approval),
        grantee_id=PROGRAMME_EXECUTION_GRANTEE_ID, at_utc=now)
    require_study_authority(reveal, kind=StudyAuthorityKind.OUTCOME_REVEAL,
        subject=issued_study, prerequisite_authority=execution_authority,
        grantee_id=PROGRAMME_REVEAL_GRANTEE_ID, at_utc=now)
    published = ExternalIssuedStudyPublisher(artifact_plane=writer, authority_store=store,
        grantee_id=PROGRAMME_ISSUE_GRANTEE_ID).load_executable_study(issued_study.object_id)
    issued = published.manifest
    if (issued != package.issued_study or published.publication_receipt != package.publication_receipt
        or issued.source_closure.implementation_commit != receipt.implementation_commit):
        raise PermissionError(codes[3])
    steps = {step.step_id: step for step in issued.candidate.base_candidate.base_candidate.protocol.steps}
    step = steps.get(receipt.task_id)
    if (step is None or not step.capability_key.startswith(prefixes)
        or not any(f'artifact.{receipt.run_id}.{step.step_id}.{output.output_id}' == logical.logical_artifact_id
                   and output.payload_schema == logical.payload_schema for output in step.outputs)):
        raise PermissionError(codes[4])
    plan = authenticate_current_result_plan_outputs(package=package, issued=issued, receipt=receipt,
        logical=logical, writer=writer, capability_prefixes=prefixes)
    return plan, package


def current_result_package_present(*, writer, run_id):
    """Inspect only the exact selected public control pair, without payload I/O."""
    if not isinstance(writer, ExternalArtifactPlane):
        return False
    relative = f'runs/{run_id}/plans/campaign-package.json'
    return any(writer.root.resolve(path, for_write=False).exists()
               or writer.root.resolve(path, for_write=False).is_symlink()
               for path in (relative, relative + '.manifest.json'))


def _current_control(*, writer, run_id, name, record_type, maximum_bytes):
    relative = f'runs/{run_id}/plans/{name}.json'
    manifest = decode_artifact_manifest(read_bounded_bytes(
        writer.root.resolve(relative + '.manifest.json', for_write=False),
        maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES))
    if (manifest.materialization.relative_path != relative
        or manifest.materialization.storage_root_id != writer.root.contract.storage_root_id
        or manifest.materialization.size_bytes > maximum_bytes
        or manifest.materialization.compression != 'none'
        or manifest.materialization.partition_selector is not None
        or manifest.logical.logical_artifact_id != f'{name}.{run_id}'
        or manifest.logical.payload_schema != record_type.SCHEMA
        or manifest.logical.profile is not ArtifactProfile.CANONICAL_JSON
        or manifest.logical.media_type != 'application/json'):
        raise PermissionError('CURRENT_RESULT_EXACT_PUBLISHED_CONTROL_REQUIRED')
    writer.verify_manifest(manifest)
    raw = read_bounded_bytes(writer.root.resolve(relative, for_write=False), maximum_bytes=maximum_bytes)
    record = decode_canonical_bytes(raw, record_type, maximum_bytes=maximum_bytes)
    if (record.canonical_bytes() != raw
        or len(raw) != manifest.materialization.size_bytes
        or record.fingerprint() != manifest.logical.content_sha256
        or sha256(raw).hexdigest() != manifest.materialization.physical_sha256):
        raise PermissionError('CURRENT_RESULT_PUBLISHED_CONTROL_BYTES_MISMATCH')
    return record


def require_current_result_receipt_outputs(*, receipt, logical, execution_plan, capability_prefixes):
    """Compare the complete output census with the actual typed execution task."""
    tasks = tuple(task for task in execution_plan.tasks if task.task_id == receipt.task_id)
    if (len(tasks) != 1 or receipt.run_id != execution_plan.source_plan.object_id
        or receipt.implementation_commit != execution_plan.implementation_commit):
        raise PermissionError('CURRENT_RESULT_EXECUTION_TASK_MISMATCH')
    task = tasks[0]
    if not task.capability.capability_key.startswith(capability_prefixes):
        raise PermissionError('CURRENT_RESULT_EXECUTION_CAPABILITY_MISMATCH')
    expected = {output.logical_artifact_id: output for output in task.outputs}
    actual = {output.logical_artifact_id: output for output in receipt.output_logical_artifacts}
    physical = {output.logical_artifact_id: output for output in receipt.output_materializations}
    if (set(expected) != set(actual) or set(expected) != set(physical)
        or len(actual) != len(receipt.output_logical_artifacts)
        or len(physical) != len(receipt.output_materializations)
        or actual.get(logical.logical_artifact_id) != logical):
        raise PermissionError('CURRENT_RESULT_COMPLETE_EXECUTION_OUTPUT_CENSUS_REQUIRED')
    for artifact_id, output in expected.items():
        observed = actual[artifact_id]
        materialization = physical[artifact_id]
        # Concrete lineage records include resolved config/dependency parents
        # in identity order. Their validated tuple need not equal the compiler's
        # edge-ceiling tuple; exact frozen visibility and access still apply.
        if ((observed.payload_schema, observed.profile, observed.media_type,
             observed.visibility_ceiling, observed.outcome_access,
             materialization.relative_path)
            != (output.payload_schema, output.profile, output.media_type,
                output.visibility_ceiling, output.outcome_access,
                output.relative_path)
            or materialization.compression != 'none' or materialization.partition_selector is not None):
            raise PermissionError('CURRENT_RESULT_EXECUTION_OUTPUT_CONTRACT_MISMATCH')
    if sum(value.size_bytes for value in physical.values()) > task.capability.requested_resources.output_bytes:
        raise PermissionError('CURRENT_RESULT_EXECUTION_OUTPUT_BYTE_BUDGET_EXCEEDED')
    return task


def authenticate_current_result_plan_outputs(*, package, issued, receipt, logical, writer, capability_prefixes):
    """Recompile exact public controls; create neither authority nor task effects."""
    if not isinstance(writer, ExternalArtifactPlane) or package.issued_study != issued:
        raise PermissionError('CURRENT_RESULT_ISSUED_PACKAGE_REQUIRED')
    modern = isinstance(package, ExperimentPackage)
    if not modern and not isinstance(package, EnvelopeExperimentPackage):
        raise PermissionError('CURRENT_RESULT_UNSUPPORTED_PACKAGE')
    run_type, execution_type = (RunPlan, ExecutionPlan) if modern else (EnvelopeRunPlan, EnvelopeExecutionPlan)
    run = _current_control(writer=writer, run_id=receipt.run_id, name='run-plan',
                           record_type=run_type, maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES)
    execution = _current_control(writer=writer, run_id=receipt.run_id, name='execution-plan',
                                 record_type=execution_type, maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES)
    if (package.run_plan_id != receipt.run_id or run.run_plan_id != receipt.run_id
        or execution.execution_plan_id != package.execution_plan_id
        or execution.source_plan != ObjectIdentity.from_record(run.run_plan_id, run)):
        raise PermissionError('CURRENT_RESULT_PACKAGE_PLAN_JOIN_MISMATCH')
    candidate = issued.candidate
    base = candidate.base_candidate.base_candidate
    projected = compile_run_plan(run_plan_id=receipt.run_id, campaign=base.campaign,
        system=base.system, experiment=base.experiment, frozen_proposal=None,
        authorization=None, template=base.protocol, registry=package.registry,
        implementation_commit=issued.source_closure.implementation_commit,
        approval_service=None, model_set=package.model_set,
        scientific_graph=base.scientific_graph,
        candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
        issued_extension_set=issued.issued_extensions,
        execution_envelope_spec=None if modern else package.execution_envelope_spec,
        execution_resource_envelope_spec=package.execution_resource_envelope_spec if modern else None,
        predevelopment_jit_signature_census=package.predevelopment_jit_signature_census if modern else None,
        jit_graph_signature_manifest=package.jit_graph_signature_manifest if modern else None,
        preissue_candidate=base)
    values = {item.name: getattr(projected, item.name) for item in fields(projected)}
    values.update(experiment=ObjectIdentity.from_record(package.experiment.experiment_id, package.experiment),
        authorization=ObjectIdentity.from_record(package.scientific_approval.authorization_id, package.scientific_approval),
        issued_extension_set=ObjectIdentity.from_record(issued.issued_extensions.issued_extension_set_id, issued.issued_extensions))
    if modern:
        values.update(execution_resource_envelope_spec=ObjectIdentity.from_record(package.execution_resource_envelope_spec.envelope_spec_id, package.execution_resource_envelope_spec),
            predevelopment_jit_signature_census=None if package.predevelopment_jit_signature_census is None else ObjectIdentity.from_record(package.predevelopment_jit_signature_census.census_id, package.predevelopment_jit_signature_census),
            jit_graph_signature_manifest=None if package.jit_graph_signature_manifest is None else ObjectIdentity.from_record(package.jit_graph_signature_manifest.manifest_id, package.jit_graph_signature_manifest))
    else:
        values['execution_envelope_spec'] = ObjectIdentity.from_record(package.execution_envelope_spec.envelope_spec_id, package.execution_envelope_spec)
    expected = run_type(**values)
    if run != expected or execution != lower_run_plan(expected, package.registry):
        raise PermissionError('CURRENT_RESULT_DETERMINISTIC_ISSUED_PLAN_MISMATCH')
    require_current_result_receipt_outputs(receipt=receipt, logical=logical,
        execution_plan=execution, capability_prefixes=capability_prefixes)
    if ExternalTaskReceiptStore(writer).read_by_receipt_id(receipt.run_id, receipt.task_id, receipt.receipt_id) != receipt:
        raise PermissionError('CURRENT_RESULT_EXACT_STORED_RECEIPT_REQUIRED')
    if any(value.storage_root_id != writer.root.contract.storage_root_id for value in receipt.output_materializations):
        raise PermissionError('CURRENT_RESULT_OUTPUT_STORAGE_ROOT_MISMATCH')
    # An operation-local batch reader can reuse this authenticated immutable
    # projection while checking every exact stored receipt/output contract.
    # This is not a persistent cache or a substitute for current authority.
    return execution


def authenticate_unprotected_current_result_if_issued(*, writer, receipt, logical, capability_prefixes):
    """Preserve exposed unissued imports; current runs must match their frozen plans."""
    if not current_result_package_present(writer=writer, run_id=receipt.run_id):
        # Missing controls cannot prove that a result was never issued. Only
        # the existing explicitly exposed software-import contract may omit
        # them, and its supplied metadata must still equal actual stored custody.
        if (not isinstance(writer, ExternalArtifactPlane)
            or logical.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
            or logical.outcome_access not in (OutcomeAccess.DEVELOPMENT_VISIBLE, OutcomeAccess.OUTCOME_BLIND)):
            raise PermissionError('CURRENT_RESULT_MISSING_ISSUED_CONTROLS')
        actual = ExternalTaskReceiptStore(writer).read_by_receipt_id(receipt.run_id, receipt.task_id, receipt.receipt_id)
        if (actual != receipt or logical not in actual.output_logical_artifacts
            or any(value.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
                   or value.outcome_access not in (OutcomeAccess.DEVELOPMENT_VISIBLE, OutcomeAccess.OUTCOME_BLIND)
                   for value in actual.output_logical_artifacts)):
            raise PermissionError('CURRENT_RESULT_EXACT_EXPOSED_IMPORT_RECEIPT_REQUIRED')
        return
    # The package is public control metadata. Its closed schema selects only the
    # two existing current package owners; it cannot select executable modules.
    relative = f'runs/{receipt.run_id}/plans/campaign-package.json'
    manifest = decode_artifact_manifest(read_bounded_bytes(
        writer.root.resolve(relative + '.manifest.json', for_write=False), maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES))
    record_type = {ExperimentPackage.SCHEMA: ExperimentPackage,
                   EnvelopeExperimentPackage.SCHEMA: EnvelopeExperimentPackage}.get(manifest.logical.payload_schema)
    if record_type is None:
        raise PermissionError('CURRENT_RESULT_UNSUPPORTED_PACKAGE')
    package = _current_control(writer=writer, run_id=receipt.run_id, name='campaign-package',
                               record_type=record_type, maximum_bytes=MAX_CAMPAIGN_PACKAGE_BYTES)
    # Keep the shared strict package parser's closed current package validation.
    if loads_campaign_package_authoring(package.canonical_bytes().decode(), media_type='application/json') != package:
        raise PermissionError('CURRENT_RESULT_PACKAGE_BYTES_MISMATCH')
    store = ExternalStudyOperationAuthorityStore(writer)
    issued = ExternalIssuedStudyPublisher(artifact_plane=writer, authority_store=store,
        grantee_id=PROGRAMME_ISSUE_GRANTEE_ID).load_executable_study(package.issued_study.issue_id).manifest
    authenticate_current_result_plan_outputs(package=package, issued=issued, receipt=receipt,
        logical=logical, writer=writer, capability_prefixes=capability_prefixes)
