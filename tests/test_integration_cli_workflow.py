# SPDX-License-Identifier: MPL-2.0
"""Public input journeys preserve immutable evidence and editable allocation joins."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from empirical_lawhood.cli.app import app

ROOT = Path(__file__).resolve().parents[1]
RUNNER = CliRunner()


def invoke(*args):
    result = RUNNER.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_original_f_public_export_authenticate_and_refuse_evidence_preparation(tmp_path):
    directory = tmp_path / 'original'
    result = invoke('campaign', 'original-f-export', '--output-dir', str(directory), '--payload-id', 'study.original-f')
    assert result['scientific_execution_performed'] is False
    operand = directory / 'original-f.canonical.json'
    checked = invoke('campaign', 'original-f-check', '--operand', str(operand), '--source-directory', str(directory))
    assert checked['identity'] == result['identity']
    assert checked['qualification_performed'] is False
    output = tmp_path / 'replacement.json'
    refused = RUNNER.invoke(app, ['config', 'prepare', '--config', str(operand), '--consumer', 'original-f', '--output-file', str(output)])
    assert refused.exit_code == 2 and not output.exists()
    assert 'retained' in refused.stderr or 'preparation' in refused.stderr.lower()
    provenance = directory / 'qualification-report.json'
    raw = provenance.read_bytes()
    provenance.write_bytes(b'[' + raw[1:])
    refused = RUNNER.invoke(app, ['campaign', 'original-f-check', '--operand', str(operand), '--source-directory', str(directory)])
    assert refused.exit_code == 2 and not refused.stdout


def test_meaningful_allocation_edit_prepares_actual_current_source_without_native_contact(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('configuration contacted native acquisition')
    monkeypatch.setattr('empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source.acquire_current_preparation_root', forbidden)
    document = json.loads((ROOT / 'experiments/matrix-inputs/allocation.json').read_bytes())
    seeds = document['value']['roots'][0]['value']['scientific_seeds']
    seeds[0]['value']['seed'] += 1
    edited = tmp_path / 'allocation.editable.json'
    edited.write_text(json.dumps(document, indent=2))
    ready = invoke('config', 'validate', '--config', str(edited), '--consumer', 'matrix-allocation', '--format', 'json')
    assert ready['canonical_byte_ready'] is False
    canonical = tmp_path / 'allocation.canonical.json'
    invoke('config', 'prepare', '--config', str(edited), '--consumer', 'matrix-allocation', '--output-file', str(canonical), '--format', 'json')
    source = tmp_path / 'source.json'
    attempt = tmp_path / 'attempt'
    prepared = invoke('campaign', 'matrix-source-config', '--config-id', 'study.edited-source', '--allocation', str(canonical), '--output', str(source), '--attempt-dir', str(attempt))
    assert prepared['native_contact'] == 'none'
    from empirical_lawhood.api.configuration import load_configuration_record
    allocation = load_configuration_record(canonical, consumer='matrix-allocation')
    bound = load_configuration_record(source, consumer='matrix-source')
    assert bound.allocation == allocation.identity
    assert edited.read_text() == json.dumps(document, indent=2)
    observed = json.loads((attempt / 'invocation.json').read_bytes())
    assert observed['state'] == 'SUCCEEDED'


def test_current_finite_allocation_is_a_numeric_proposal_and_missing_storage_refuses_before_work(tmp_path):
    allocation = tmp_path / 'allocation.json'
    proposed = invoke('campaign', 'finite-response-allocation', '--stage', 'calibration', '--cohort-namespace', 'empirical-lawhood.finite-response-law.calibration.study', '--master-seed', '9234567', '--output', str(allocation))
    assert proposed['scientific_freshness_established'] is False
    assert proposed['evidence_role'] == 'PROPOSED_UNRUN'
    ready = invoke('config', 'validate', '--config', str(allocation), '--consumer', 'finite-current-allocation', '--format', 'json')
    assert ready['canonical_byte_ready'] is True
    output = tmp_path / 'native'
    refused = RUNNER.invoke(app, ['campaign', 'matrix-source-export', '--config', str(ROOT / 'experiments/matrix-inputs/matrix-source.json'), '--allocation', str(ROOT / 'experiments/matrix-inputs/allocation.json'), '--output-dir', str(output)])
    assert refused.exit_code == 2 and '--operator-profile' in refused.stderr
    assert not output.exists()


@pytest.mark.parametrize('destination_kind', ('missing-parent', 'parent-file', 'parent-symlink', 'traversal', 'existing', 'filesystem-root'))
def test_rc_source_output_refused_before_custody_or_publication(tmp_path, monkeypatch, destination_kind):
    real = tmp_path / 'real'
    real.mkdir()
    if destination_kind == 'missing-parent':
        output = tmp_path / 'missing' / 'numeric.json'
    elif destination_kind == 'parent-file':
        parent = tmp_path / 'file'
        parent.write_bytes(b'preserved')
        output = parent / 'numeric.json'
    elif destination_kind == 'parent-symlink':
        parent = tmp_path / 'link'
        parent.symlink_to(real, target_is_directory=True)
        output = parent / 'numeric.json'
    elif destination_kind == 'traversal':
        output = real / '..' / 'numeric.json'
    elif destination_kind == 'filesystem-root':
        output = Path(tmp_path.anchor)
    else:
        output = real / 'numeric.json'
        output.write_bytes(b'preserved')

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid output contacted custody or publication')

    monkeypatch.setattr('empirical_lawhood.cli.integrations._storage', forbidden)
    monkeypatch.setattr('empirical_lawhood.api.rc_challenge_inputs.publish_rc_source_export', forbidden)
    refused = RUNNER.invoke(app, ['campaign', 'rc-challenge-source-export', '--config', str(ROOT / 'experiments/rc-challenges/source-export.json'), '--output', str(output)])
    assert refused.exit_code == 2, refused.output
    assert not refused.stdout
    assert 'output' in refused.stderr.lower() or 'traverse' in refused.stderr.lower()
    assert not (tmp_path / 'numeric.json').exists()
    assert not (real / 'numeric.json').exists() or (real / 'numeric.json').read_bytes() == b'preserved'


def test_output_writer_rechecks_parent_after_noncreating_preflight(tmp_path):
    from empirical_lawhood.api.configuration import ConfigurationError, _write_new_file, preflight_new_configuration_output

    parent = tmp_path / 'parent'
    parent.mkdir()
    output = parent / 'result.json'
    assert preflight_new_configuration_output(output) == output
    assert not output.exists()
    moved = tmp_path / 'moved'
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ConfigurationError, match='symlinks'):
        _write_new_file(output, b'never-written')
    assert not (moved / 'result.json').exists()


def _rc_cli_plane(tmp_path, monkeypatch):
    from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
    from empirical_lawhood.runtime.artifacts import ExternalRootContract
    from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile

    store = tmp_path / 'synthetic-rc-cli-store'
    store.mkdir()
    plane = ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract(
        'synthetic.rc-cli-store', 'exposed RC CLI software fixture', str(store), '/',
        OperatorStorageProfile.SCHEMA, 1, None, None, (),
    )))
    monkeypatch.setattr('empirical_lawhood.cli.integrations._storage', lambda: (ROOT, None, plane))
    return plane, store


def test_rc_canary_cli_authors_new_directory_and_loads_public_handoff(tmp_path, monkeypatch):
    from empirical_lawhood.api import rc_challenge_authoring, integration_handoffs
    from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind

    plane, store = _rc_cli_plane(tmp_path, monkeypatch)
    closure = ImplementationSourceClosure('synthetic.rc-cli-source', SourceClosureKind.CLEAN_GIT_COMMIT,
        '0' * 40, 'd' * 64, 'e' * 64, True)
    monkeypatch.setattr(rc_challenge_authoring, 'capture_clean_target_closure', lambda *args: closure)
    monkeypatch.setattr(rc_challenge_authoring, 'resolve_authoring_directory',
        lambda directory, **kwargs: (directory, plane.root.contract))
    monkeypatch.setattr(integration_handoffs, 'resolve_authoring_directory',
        lambda directory, **kwargs: (directory, plane.root.contract))
    destination = store / 'canary'
    report = invoke('campaign', 'rc-challenge-author', '--config',
        str(ROOT / 'experiments/rc-challenges/canary.json'), '--output-dir', str(destination))
    assert report['scientific_execution_performed'] is False
    handoff = integration_handoffs.load_integration_handoff(directory=destination, root=ROOT,
        storage_profile=None, artifact_writer=plane)
    proof = integration_handoffs.prove_integration_handoff(handoff)
    assert handoff.family == 'rc-challenges'
    assert proof['task_count'] == report['task_count'] == 7
    assert not proof['scientific_execution_performed']


@pytest.mark.parametrize('destination_kind', ('existing', 'outside-root'))
def test_rc_cli_author_refuses_output_before_protected_prerequisite_contact(tmp_path, monkeypatch, destination_kind):
    _, store = _rc_cli_plane(tmp_path, monkeypatch)
    destination = store / 'occupied' if destination_kind == 'existing' else tmp_path / 'outside'
    if destination_kind == 'existing':
        destination.mkdir()

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid authoring destination contacted protected input')

    monkeypatch.setattr('empirical_lawhood.cli.integrations._configuration', forbidden)
    monkeypatch.setattr('empirical_lawhood.api.rc_challenge_inputs.read_rc_challenge_numeric_input', forbidden)
    monkeypatch.setattr('empirical_lawhood.api.rc_challenge_results.read_rc_challenge_result', forbidden)
    refused = RUNNER.invoke(app, ['campaign', 'rc-challenge-author', '--config',
        str(ROOT / 'experiments/rc-challenges/canary.json'), '--output-dir', str(destination),
        '--numeric-input', str(tmp_path / 'not-contacted.json')])
    assert refused.exit_code == 2, refused.output
    assert not refused.stdout
    assert not destination.exists() or not tuple(destination.iterdir())


@pytest.mark.parametrize('command', ('preparation-conformance', 'preparation-panel-export'))
def test_preparation_directory_refuses_file_ancestor_before_input_or_native_contact(tmp_path, monkeypatch, command):
    _, store = _rc_cli_plane(tmp_path, monkeypatch)
    parent = store / 'preserved-file'
    parent.write_bytes(b'preserved')
    output = parent / 'nested' / 'result'

    def forbidden(*args, **kwargs):
        raise AssertionError('invalid directory contacted input or native computation')

    monkeypatch.setattr('empirical_lawhood.cli.integrations._configuration', forbidden)
    monkeypatch.setattr('empirical_lawhood.api.preparation_conformance.preparation_native_conformance', forbidden)
    monkeypatch.setattr('empirical_lawhood.api.preparation_screens.ordinary_preparation_operands_export', forbidden)
    source = str(ROOT / 'experiments/preparation-applicability/conformance-allocation.json')
    options = ['--allocation', source, '--conformance-id', 'synthetic.refused-conformance'] if command == 'preparation-conformance' else [
        '--stage', source, '--original-f', source, '--panel', source,
    ]
    refused = RUNNER.invoke(app, ['campaign', command, *options, '--output', str(output)])
    assert refused.exit_code == 2, refused.output
    assert 'non-directory parent' in refused.stderr
    assert not refused.stdout
    assert parent.read_bytes() == b'preserved'
