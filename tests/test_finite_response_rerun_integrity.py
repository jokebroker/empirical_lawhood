"""Current input integrity and exact software custody; no scientific execution."""

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from empirical_lawhood.adapters.composition.finite_response_law.assignment import proposed_scientific_seeds
from empirical_lawhood.adapters.composition.finite_response_law.rerun_input import (
    PUBLIC_CALIBRATION_MASTER_SEED, PUBLIC_EVALUATION_MASTER_SEED,
    FiniteResponseLawCurrentAllocation, FiniteResponseLawCurrentExposure,
    FiniteResponseLawCurrentPublication, FiniteResponseLawRerunInput,
    original_development_exposure,
)
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.api import finite_rerun_authoring as application
from empirical_lawhood.api.finite_operands import current_nomination_operand_export
from empirical_lawhood.infrastructure.artifacts import ArtifactIdentityConflict, ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.infrastructure.task_receipts import ExternalTaskReceiptStore
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.planning.study_issue import StudyOperationAuthority
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ArtifactWriteRequest, CanonicalTaskReceipt, ExternalRootContract, ReceiptCheck
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile

ROOT = Path(__file__).resolve().parents[1]


def allocation(stage='calibration', master=202610080032, role='PROPOSED_UNRUN'):
    return FiniteResponseLawCurrentAllocation(stage,
        f'empirical-lawhood.finite-response-law.{stage}.integrity-{master}',
        FiniteResponseLawScienceSpec().plan_sha256, 32 if stage == 'calibration' else 64,
        role, proposed_scientific_seeds(stage, master))


@pytest.fixture
def nomination(tmp_path):
    directory = tmp_path / 'fixed-public-nomination'
    directory.mkdir()
    return current_nomination_operand_export(directory, nomination_id='synthetic.integrity-nomination'), directory


def test_original_development_census_preserves_authenticated_archive_roots_and_effective_streams():
    units, seeds = original_development_exposure()
    assert len(units) == 48 and len(seeds) == 912
    assert sum('.q.prepared.' in unit for unit in units) == 16
    assert sum('.e.prepared.' in unit for unit in units) == 32
    assert sum(seed.startswith('seed.pcg64dxsm.') for seed in seeds) == 864
    assert sum(seed.startswith('seed.pcg64.') for seed in seeds) == 48
    # Golden census pins accompany the independently checked archived exposure,
    # not another implementation of the derivation recipe.
    assert sha256('\n'.join(units).encode()).hexdigest() == '9a62e0218c7a955fab653d9b3442ca2564112e318e9ff1135aa43abfa4429c63'
    assert sha256('\n'.join(seeds).encode()).hexdigest() == 'f8ccd7f07455e3f5fad697ec3c98efa4261d4c9aa4eeb18442115fc1a9b17bb3'


def test_original_prior_census_cannot_be_omitted(nomination):
    operand, _ = nomination
    units, seeds = original_development_exposure()
    with pytest.raises(ValueError, match='ORIGINAL_48_ROOT_EFFECTIVE_EXPOSURE_REQUIRED'):
        FiniteResponseLawRerunInput('synthetic.incomplete-prior', 'calibration', allocation(), operand, units[:-1], seeds)
    with pytest.raises(ValueError, match='ORIGINAL_48_ROOT_EFFECTIVE_EXPOSURE_REQUIRED'):
        FiniteResponseLawRerunInput('synthetic.incomplete-prior', 'calibration', allocation(), operand, units, seeds[:-1])


def test_public_sample_streams_remain_exposed_after_namespace_and_role_change(nomination):
    operand, _ = nomination
    units, seeds = original_development_exposure()
    sample = allocation(master=PUBLIC_CALIBRATION_MASTER_SEED, role='EXPOSED_DEVELOPMENT_NONPROMOTABLE')
    packet = FiniteResponseLawRerunInput('synthetic.public-sample', 'calibration', sample, operand, units, seeds)
    assert packet.evidence_role == 'EXPOSED_DEVELOPMENT_NONPROMOTABLE'
    renamed = replace(sample, cohort_namespace='empirical-lawhood.finite-response-law.calibration.changed-label', evidence_role='PROPOSED_UNRUN')
    with pytest.raises(ValueError, match='PUBLIC_SAMPLE_CANNOT_BECOME_UNEXPOSED'):
        replace(packet, allocation=renamed)
    assert PUBLIC_EVALUATION_MASTER_SEED != PUBLIC_CALIBRATION_MASTER_SEED


def test_effective_probe_aliases_do_not_count_as_independent_roots():
    original = allocation()
    rows = original.scientific_seeds
    first = next(seed for purpose, index, seed in rows if (purpose, index) == ('passive-probes', 0))
    altered = tuple((purpose, index, first ^ 1 if (purpose, index) == ('passive-probes', 1) else seed) for purpose, index, seed in rows)
    with pytest.raises(ValueError, match='EFFECTIVE_PROBE_STREAM_COLLISION'):
        replace(original, scientific_seeds=altered)
    # Full scientific commitments differ, while the actual consumed bits agree.
    assert first != first ^ 1 and first >> 128 == (first ^ 1) >> 128


def test_allocation_helper_preserves_numeric_census_and_exact_independent_root_counts():
    for stage, count, master in (('calibration', 32, 2032032), ('prospective-evaluation', 64, 2032064)):
        original = allocation(stage=stage, master=master)
        prepared = application.prepare_finite_response_allocation(stage=stage, cohort_namespace=original.cohort_namespace, master_seed=master)
        assert prepared == original and prepared.root_count == count
        renamed = application.prepare_finite_response_allocation(stage=stage, cohort_namespace=original.cohort_namespace + '.new-label', master_seed=master)
        assert renamed.scientific_seeds == prepared.scientific_seeds
        assert renamed.fingerprint() != prepared.fingerprint()


def test_historical_or_handcrafted_issue_identity_is_not_current_authority():
    grant = ObjectIdentity('synthetic.grant', StudyOperationAuthority.SCHEMA, '1.0.0', 'a' * 64)
    old = ObjectIdentity('synthetic.old-issued', 'empirical-lawhood/tests/old-development-issue', '1.0.0', 'b' * 64)
    with pytest.raises(ValueError, match='ACTUAL_CURRENT_ISSUE_AND_AUTHORITY_IDENTITIES_REQUIRED'):
        FiniteResponseLawCurrentPublication('synthetic.run', old, grant, grant)


def test_result_selector_rejects_unknown_kind_before_opening_any_store():
    with pytest.raises(ValueError, match='UNSUPPORTED_KIND'):
        application.current_result_input_from_receipt(config_id='synthetic.result', run_id='synthetic.run', issued_study_id='synthetic.issued', execution_authority_id='synthetic.execute', reveal_authority_id='synthetic.reveal', task_id='synthetic.task', receipt_id='receipt.synthetic.run.synthetic.task.attempt-001', result_kind='new-qualification', artifact_writer=None)


def test_packet_helper_authenticates_public_nomination_and_keeps_current_exposure(nomination):
    operand, directory = nomination
    units, seeds = original_development_exposure()
    exposure = FiniteResponseLawCurrentExposure('synthetic.exposure', units, seeds)
    packet = application.prepare_finite_response_rerun(config_id='synthetic.prepared-current', stage='calibration', allocation=allocation(), exposure=exposure, nomination=operand, nomination_directory=directory, artifact_writer=None)
    assert packet.original_prior_unit_ids == units
    assert packet.nomination == operand and packet.method is packet.control is None
    changed = directory / 'coefficients.transport.json'
    changed.write_bytes(changed.read_bytes() + b' ')
    with pytest.raises(ValueError):
        application.prepare_finite_response_rerun(config_id='synthetic.changed-nomination', stage='calibration', allocation=allocation(), exposure=exposure, nomination=operand, nomination_directory=directory, artifact_writer=None)


def test_exact_receipt_custody_rejects_wrong_run_task_commit_and_incomplete_outputs(tmp_path):
    """Exercise the real receipt store; this unit supplies no scientific grants."""
    plane = ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract('synthetic.integrity-store', 'software-only receipt store', str(tmp_path), '/', OperatorStorageProfile.SCHEMA, 1, None, None, ())))
    run_id, task_id = 'synthetic.current-run', 'synthetic.current-task'
    output = SimpleNamespace(output_id='report', payload_schema=ObjectIdentity.SCHEMA, profile=ArtifactProfile.CANONICAL_JSON, media_type='application/json', filename_suffix='.json')
    logical_id = f'artifact.{run_id}.{task_id}.report'
    retained = plane.write(ArtifactWriteRequest(logical_artifact_id=logical_id, relative_path=f'runs/{run_id}/outputs/{task_id}/report.json', payload_schema=output.payload_schema, profile=output.profile, media_type=output.media_type, publication_scope_id='synthetic.current-outputs', publication_scope_relative_root=f'runs/{run_id}/outputs', payload=ObjectIdentity('synthetic.fixture', ObjectIdentity.SCHEMA, '1.0.0', 'c' * 64).canonical_bytes(), visibility_ceiling=VisibilityCeiling.PROSPECTIVE, parent_visibility_ceilings=(), outcome_access=OutcomeAccess.EVALUATOR_REVEAL, minimum_free_bytes=1))
    step = SimpleNamespace(step_id=task_id, outputs=(output,), resource_budget=SimpleNamespace(output_bytes=1024**2))
    issued = SimpleNamespace(candidate=SimpleNamespace(base_candidate=SimpleNamespace(base_candidate=SimpleNamespace(protocol=SimpleNamespace(steps=(step,))))), source_closure=SimpleNamespace(implementation_commit='0' * 40))
    store = ExternalTaskReceiptStore(plane, minimum_free_bytes=1)
    receipts = []
    for number in (1, 2):
        attempt = f'{run_id}.{task_id}.attempt-{number:03d}'
        receipt = CanonicalTaskReceipt(f'receipt.{attempt}', run_id, task_id, attempt, '0' * 40, (), (retained.materialization,), (retained.logical,), (ReceiptCheck('software-only-contract', True, ()),), OperationalStatus.SUCCEEDED, ())
        store.commit(receipt, visibility_ceiling=VisibilityCeiling.PROSPECTIVE, outcome_access=OutcomeAccess.EVALUATOR_REVEAL)
        receipts.append(receipt)
    assert application._exact_current_receipt(plane, issued, run_id, task_id, receipts[1].receipt_id) == receipts[1]
    assert application._exact_current_receipt(plane, issued, run_id, task_id, receipts[0].receipt_id) == receipts[0]
    for wrong_run, wrong_task in (('synthetic.other-run', task_id), (run_id, 'synthetic.other-task')):
        with pytest.raises(ArtifactIdentityConflict, match='substitutes its run or task'):
            application._exact_current_receipt(plane, issued, wrong_run, wrong_task, receipts[0].receipt_id)
    issued.source_closure.implementation_commit = '1' * 40
    with pytest.raises(ValueError, match='SUCCESSFUL_EXACT_ISSUED_RECEIPT_REQUIRED'):
        application._exact_current_receipt(plane, issued, run_id, task_id, receipts[0].receipt_id)
    issued.source_closure.implementation_commit = '0' * 40
    step.resource_budget.output_bytes = retained.materialization.size_bytes - 1
    with pytest.raises(ValueError, match='ISSUED_OUTPUT_BYTE_BUDGET_EXCEEDED'):
        application._exact_current_receipt(plane, issued, run_id, task_id, receipts[0].receipt_id)
    step.resource_budget.output_bytes = 1024**2
    step.outputs += (SimpleNamespace(output_id='missing-report'),)
    with pytest.raises(ValueError, match='COMPLETE_ISSUED_TASK_OUTPUTS_REQUIRED'):
        application._exact_current_receipt(plane, issued, run_id, task_id, receipts[0].receipt_id)


@pytest.mark.parametrize('destination', ('outside', 'traversal', 'parent-symlink', 'nonempty', 'parent-file'))
def test_authoring_destination_is_refused_before_closure_or_protected_reads(tmp_path, monkeypatch, destination):
    store = tmp_path / 'guarded'
    store.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    plane = ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract('synthetic.preflight-store',
        'software-only output preflight', str(store), '/', OperatorStorageProfile.SCHEMA, 1, None, None, ())))
    if destination == 'outside':
        output = outside / 'new'
    elif destination == 'traversal':
        output = store / '..' / 'outside' / 'new'
    elif destination == 'parent-symlink':
        (store / 'link').symlink_to(outside, target_is_directory=True)
        output = store / 'link' / 'new'
    elif destination == 'nonempty':
        output = store / 'retained'
        output.mkdir()
        (output / 'preserved.txt').write_text('existing evidence')
    else:
        (store / 'file').write_text('existing file')
        output = store / 'file' / 'new'
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid destination reached source or protected input')
    for name in ('capture_clean_target_closure', 'current_nomination_operand_import', '_validate_parent'):
        monkeypatch.setattr(application, name, forbidden)
    with pytest.raises((OSError, ValueError)):
        application.author_finite_response_rerun(root=ROOT, packet=None, exposure=None,
            nomination_directory=outside, output_dir=output, artifact_writer=plane)
    assert not (outside / 'new').exists()
    if destination == 'nonempty':
        assert (output / 'preserved.txt').read_text() == 'existing evidence'


def test_authoring_accepts_cli_created_empty_directory_before_closure(tmp_path, monkeypatch):
    output = tmp_path / 'empty'
    output.mkdir()
    plane = ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract('synthetic.empty-preflight-store',
        'software-only empty directory', str(tmp_path), '/', OperatorStorageProfile.SCHEMA, 1, None, None, ())))
    def stop(*args, **kwargs):
        raise AssertionError('empty output passed preflight')
    monkeypatch.setattr(application, 'capture_clean_target_closure', stop)
    with pytest.raises(AssertionError, match='empty output passed preflight'):
        application.author_finite_response_rerun(root=ROOT, packet=SimpleNamespace(config_id='synthetic.packet'),
            exposure=None, nomination_directory=tmp_path, output_dir=output, artifact_writer=plane)
    assert not tuple(output.iterdir())


def test_native_result_summary_fits_report_bound_and_preserves_full_failed_census():
    from empirical_lawhood.infrastructure.development_attempts import MAX_REPORT_BYTES
    from tests.finite_response_custody_fixtures import synthetic_native
    record = synthetic_native()
    view = record.views[0]
    # A declared native numerical failure remains within the exact typed full
    # roster; no observation or independent root is removed from the result.
    failed = replace(view, accounting=((view.accounting[0][0], 0, 'NUMERICAL_FAILURE'), *view.accounting[1:]))
    record = replace(record, views=(failed, *record.views[1:]))
    raw = record.canonical_bytes()
    assert len(raw) > MAX_REPORT_BYTES
    logical = SimpleNamespace(logical_artifact_id='synthetic.native-result', payload_schema=record.SCHEMA,
        content_sha256=sha256(raw).hexdigest(), visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED)
    physical = SimpleNamespace(relative_path='runs/synthetic.native/outputs/native/result.json',
        physical_sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
    config = SimpleNamespace(config_id='synthetic.selected-native', manifests=(SimpleNamespace(logical=logical,
        materialization=physical),), result_artifact_id=logical.logical_artifact_id,
        receipt_binding=('synthetic.native', 'native', 'receipt.synthetic.native.attempt-002'))
    summary = application.finite_response_current_result_summary(config=config, record=record)
    assert len(json.dumps(summary).encode()) < MAX_REPORT_BYTES
    assert summary['independent_roots'] == len(summary['root_ids']) == 32
    assert summary['nested_numerical_views'] == len(summary['views']) == 64
    assert summary['native_cell_dispositions']['NUMERICAL_FAILURE'] == 1
    assert sum(summary['native_cell_dispositions'].values()) == summary['assigned_native_cells']
    assert summary['scientific_status'] == 'UNEVALUABLE'
    assert summary['result_reference']['receipt_id'].endswith('attempt-002')
    assert summary['result_reference']['physical_sha256'] == sha256(raw).hexdigest()
    assert 'eligible_for_prospective_evaluation' not in summary
    logical.content_sha256 = 'f' * 64
    with pytest.raises(ValueError, match='SUMMARY_PUBLICATION_MISMATCH'):
        application.finite_response_current_result_summary(config=config, record=record)
