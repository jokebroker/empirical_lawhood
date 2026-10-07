"""Independent typed-plan output denials; no task execution or scientific grant."""

from dataclasses import replace
from pathlib import Path

import pytest

from empirical_lawhood.api import current_result_custody, preparation_result_access, rc_challenge_results
from empirical_lawhood.infrastructure.artifacts import ArtifactProfileValidatorRegistry, ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.runtime.artifacts import ArtifactMaterialization, ArtifactProfile, ArtifactWriteRequest, CanonicalTaskReceipt, ExternalRootContract, LogicalArtifactIdentity, ReceiptCheck
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile
from empirical_lawhood.runtime.plans import CandidateExecutionPlan


@pytest.fixture
def frozen_outputs():
    plan = decode_canonical_bytes((Path(__file__).parent / 'fixtures/authoring-execution-plan.json').read_bytes(), CandidateExecutionPlan, maximum_bytes=1024**2)
    task = next(value for value in plan.tasks if value.task_id.endswith('.numerical-evaluate'))
    validators = ArtifactProfileValidatorRegistry()
    logical = tuple(LogicalArtifactIdentity(output.logical_artifact_id, 'a' * 64,
        output.payload_schema, output.profile, output.media_type, output.visibility_ceiling,
        output.parent_visibility_ceilings, output.outcome_access,
        validators.generic_validation(profile=output.profile, payload_schema=output.payload_schema))
        for output in task.outputs)
    physical = tuple(ArtifactMaterialization(f'synthetic.output-{index}', output.logical_artifact_id,
        'synthetic.result-store', output.relative_path, 'a' * 64, 1, 'none')
        for index, output in enumerate(task.outputs))
    attempt = f'{plan.source_plan.object_id}.{task.task_id}.attempt-001'
    receipt = CanonicalTaskReceipt(f'receipt.{attempt}', plan.source_plan.object_id, task.task_id,
        attempt, plan.implementation_commit, (), physical, logical,
        (ReceiptCheck('metadata-contract-only', True, ()),), OperationalStatus.SUCCEEDED, ())
    return plan, receipt, logical[0]


def _join(plan, receipt, logical):
    return current_result_custody.require_current_result_receipt_outputs(
        receipt=receipt, logical=logical, execution_plan=plan,
        capability_prefixes=('rc-ladder-response.',))


def test_complete_typed_output_contract_accepts_without_creating_authority(frozen_outputs):
    plan, receipt, logical = frozen_outputs
    task = _join(plan, receipt, logical)
    assert task.task_id == receipt.task_id and len(task.outputs) == 2


@pytest.mark.parametrize('mutation', ('access', 'visibility', 'media', 'path', 'over-budget'))
def test_supplied_metadata_cannot_replace_frozen_output_contract(frozen_outputs, mutation):
    plan, receipt, logical = frozen_outputs
    if mutation == 'access':
        logical = replace(logical, outcome_access=OutcomeAccess.OUTCOME_BLIND)
    elif mutation == 'visibility':
        logical = replace(logical, visibility_ceiling=VisibilityCeiling.UNBOUND_HISTORICAL)
    elif mutation == 'media':
        logical = replace(logical, media_type='application/json')
    if mutation in ('access', 'visibility', 'media'):
        receipt = replace(receipt, output_logical_artifacts=(logical, *receipt.output_logical_artifacts[1:]))
    elif mutation == 'path':
        receipt = replace(receipt, output_materializations=(replace(receipt.output_materializations[0], relative_path='synthetic/foreign-output.json'), *receipt.output_materializations[1:]))
    else:
        task = next(value for value in plan.tasks if value.task_id == receipt.task_id)
        receipt = replace(receipt, output_materializations=(replace(receipt.output_materializations[0], size_bytes=task.capability.requested_resources.output_bytes + 1), *receipt.output_materializations[1:]))
    with pytest.raises(PermissionError, match='OUTPUT_(CONTRACT_MISMATCH|BYTE_BUDGET_EXCEEDED)'):
        _join(plan, receipt, logical)


@pytest.mark.parametrize('mutation', ('missing-output', 'foreign-output', 'foreign-task', 'foreign-run', 'foreign-implementation'))
def test_wrong_complete_census_or_task_identity_is_refused(frozen_outputs, mutation):
    plan, receipt, logical = frozen_outputs
    if mutation == 'missing-output':
        receipt = replace(receipt, output_logical_artifacts=receipt.output_logical_artifacts[:1], output_materializations=receipt.output_materializations[:1])
    elif mutation == 'foreign-output':
        logical = replace(logical, logical_artifact_id=logical.logical_artifact_id + '.foreign')
        physical = replace(receipt.output_materializations[0], logical_artifact_id=logical.logical_artifact_id)
        receipt = replace(receipt, output_logical_artifacts=tuple(sorted((logical, *receipt.output_logical_artifacts[1:]), key=lambda value:value.logical_artifact_id)), output_materializations=(physical, *receipt.output_materializations[1:]))
    else:
        receipt = replace(receipt, **{'task_id':'synthetic.foreign-task'} if mutation == 'foreign-task' else {'run_id':'synthetic.foreign-run'} if mutation == 'foreign-run' else {'implementation_commit':'b' * 40})
    with pytest.raises(PermissionError, match='(CENSUS_REQUIRED|TASK_MISMATCH)'):
        _join(plan, receipt, logical)


@pytest.mark.parametrize('owner', ('rc', 'preparation'))
def test_unprotected_downgrade_cannot_skip_present_current_control(tmp_path, frozen_outputs, monkeypatch, owner):
    plan, receipt, logical = frozen_outputs
    logical = replace(logical, outcome_access=OutcomeAccess.OUTCOME_BLIND)
    receipt = replace(receipt, output_logical_artifacts=(logical, *receipt.output_logical_artifacts[1:]))
    plane = ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract(
        'synthetic.result-store', 'software-only result denial', str(tmp_path), '/',
        OperatorStorageProfile.SCHEMA, 1, None, None, (),)))
    # A current control pair exists but its schema is deliberately foreign. The
    # supplied access downgrade must reach that refusal rather than return early.
    plane.write(ArtifactWriteRequest(logical_artifact_id=f'campaign-package.{receipt.run_id}',
        relative_path=f'runs/{receipt.run_id}/plans/campaign-package.json',
        payload_schema=ObjectIdentity.SCHEMA, profile=ArtifactProfile.CANONICAL_JSON,
        media_type='application/json', publication_scope_id='synthetic.foreign-controls',
        publication_scope_relative_root=f'runs/{receipt.run_id}/plans',
        payload=ObjectIdentity('synthetic.foreign-package', ObjectIdentity.SCHEMA, '1.0.0', 'a' * 64).canonical_bytes(),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE, parent_visibility_ceilings=(),
        outcome_access=OutcomeAccess.OUTCOME_BLIND, minimum_free_bytes=1))

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid current control reached protected payload verification')

    monkeypatch.setattr(plane, 'verify_manifest', forbidden)
    guard = rc_challenge_results._authenticate_result_access if owner == 'rc' else preparation_result_access.authenticate_preparation_result_access
    with pytest.raises(PermissionError, match='UNSUPPORTED_PACKAGE'):
        guard(context=None, receipt=receipt, logical=logical, writer=plane)


def test_exposed_unissued_output_preserves_existing_inspection_contract(tmp_path, frozen_outputs):
    from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
    _, receipt, logical = frozen_outputs
    rows = tuple(replace(value, outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY, parent_visibility_ceilings=())
        for value in receipt.output_logical_artifacts)
    logical = rows[0]
    receipt = replace(receipt, output_logical_artifacts=rows)
    plane = ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract(
        'synthetic.result-store', 'exposed unissued software output', str(tmp_path), '/',
        OperatorStorageProfile.SCHEMA, 1, None, None, (),)))
    ExternalTaskReceiptStore(plane, minimum_free_bytes=1).commit(receipt,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY, outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE)
    rc_challenge_results._authenticate_result_access(context=None, receipt=receipt, logical=logical, writer=plane)
    assert not (tmp_path / 'runs' / receipt.run_id / 'plans').exists()


@pytest.mark.parametrize('owner', ('rc', 'preparation'))
def test_missing_current_controls_do_not_authorize_supplied_access_downgrade(tmp_path, frozen_outputs, owner):
    _, receipt, logical = frozen_outputs
    logical = replace(logical, outcome_access=OutcomeAccess.OUTCOME_BLIND)
    plane = ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract(
        'synthetic.result-store', 'missing current controls denial', str(tmp_path), '/',
        OperatorStorageProfile.SCHEMA, 1, None, None, (),)))
    guard = rc_challenge_results._authenticate_result_access if owner == 'rc' else preparation_result_access.authenticate_preparation_result_access
    with pytest.raises(PermissionError, match='MISSING_ISSUED_CONTROLS'):
        guard(context=None, receipt=receipt, logical=logical, writer=plane)


def test_concrete_lineage_census_does_not_replace_frozen_access_contract(frozen_outputs):
    plan, receipt, logical = frozen_outputs
    # Concrete runtime parents and compiler scientific-edge parents have
    # distinct order/census contracts. Both retain the same frozen access.
    logical = replace(logical, parent_visibility_ceilings=())
    receipt = replace(receipt, output_logical_artifacts=(logical, *receipt.output_logical_artifacts[1:]))
    assert _join(plan, receipt, logical).task_id == receipt.task_id


def test_exposed_label_cannot_replace_actual_stored_protected_receipt(tmp_path, frozen_outputs, monkeypatch):
    from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
    _, receipt, logical = frozen_outputs
    plane = ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract(
        'synthetic.result-store', 'stored protected receipt downgrade denial', str(tmp_path), '/',
        OperatorStorageProfile.SCHEMA, 1, None, None, (),)))
    ExternalTaskReceiptStore(plane, minimum_free_bytes=1).commit(receipt,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE, outcome_access=OutcomeAccess.EVALUATION_REVEALED)
    rows = tuple(replace(value, outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY, parent_visibility_ceilings=())
        for value in receipt.output_logical_artifacts)
    claimed = replace(receipt, output_logical_artifacts=rows)

    original_verify = plane.verify_manifests

    def metadata_only(manifests):
        if any(manifest.logical.payload_schema != CanonicalTaskReceipt.SCHEMA for manifest in manifests):
            raise AssertionError('downgraded receipt reached scientific output verification')
        return original_verify(manifests)

    monkeypatch.setattr(plane, 'verify_manifests', metadata_only)
    with pytest.raises(PermissionError, match='EXACT_EXPOSED_IMPORT_RECEIPT_REQUIRED'):
        rc_challenge_results._authenticate_result_access(context=None, receipt=claimed, logical=rows[0], writer=plane)
