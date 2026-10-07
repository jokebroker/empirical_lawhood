# SPDX-License-Identifier: MPL-2.0
"""Focused software-only integrity checks for avoiding repeated control work."""

from dataclasses import dataclass, replace
from decimal import Decimal
import os
from types import SimpleNamespace
from typing import ClassVar

import pytest

from empirical_lawhood.api import finite_rerun_authoring as application
from empirical_lawhood.api.finite_operands import current_nomination_operand_export
from empirical_lawhood.adapters.composition.finite_response_law.consumer_ports import FiniteResponseLawRuntimeContext
from empirical_lawhood.adapters.composition.finite_response_law.rerun_input import FiniteResponseLawCurrentExposure, FiniteResponseLawRerunInput, original_development_exposure
from empirical_lawhood.infrastructure import bounded_io
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from tests.test_finite_response_rerun_integrity import allocation


@dataclass(frozen=True)
class ExpectedProduct(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/tests/expected-product'
    quantity: Decimal
    labels: tuple[str, ...]


@pytest.mark.parametrize('filename', ('authoring.json', 'base-authoring.json', 'candidate-report.json',
    'candidate.json', 'base-candidate.json', 'resources.json', 'preissue-run-plan.json',
    'preissue-execution-plan.json', 'payload-00.json', 'decoder-00.json'))
def test_expected_projection_reads_exact_bytes_without_typed_reconstruction(tmp_path, monkeypatch, filename):
    record = ExpectedProduct(Decimal('1.25'), ('first', 'second'))
    (tmp_path / filename).write_bytes(record.canonical_bytes())
    monkeypatch.setattr(application, 'decode_canonical_bytes', lambda *args, **kwargs: pytest.fail('redundant typed decode'))
    application._assert_expected_record(tmp_path, filename, record, 'exact-refusal')
    # A later independent operation must inspect changed actual bytes afresh.
    (tmp_path / filename).write_bytes(record.canonical_bytes() + b' ')
    with pytest.raises(ValueError, match='exact-refusal'):
        application._assert_expected_record(tmp_path, filename, record, 'exact-refusal')


@pytest.mark.parametrize('mutation', ('malformed', 'duplicate', 'schema', 'version', 'decimal',
    'negative-zero', 'extra-field', 'missing-field', 'changed-value', 'reordered', 'whitespace'))
def test_expected_projection_refuses_every_nonexact_canonical_product(tmp_path, mutation):
    record = ExpectedProduct(Decimal('1.25'), ('first', 'second'))
    raw = record.canonical_bytes()
    altered = {
        'malformed': b'{',
        'duplicate': raw.replace(b'"quantity":', b'"quantity":{"decimal":"1.25"},"quantity":'),
        'schema': raw.replace(b'expected-product', b'other-product'),
        'version': raw.replace(b'"version":"1.0.0"', b'"version":"1.0.1"'),
        'decimal': raw.replace(b'"1.25"', b'"1.25e0"'),
        'negative-zero': raw.replace(b'"1.25"', b'"-0"'),
        'extra-field': raw.replace(b'"labels":', b'"extra":null,"labels":'),
        'missing-field': raw.replace(b'"quantity":{"decimal":"1.25"}', b'"absent":null'),
        'changed-value': raw.replace(b'"1.25"', b'"1.26"'),
        'reordered': raw.replace(b'["first","second"]', b'["second","first"]'),
        'whitespace': raw + b' ',
    }[mutation]
    (tmp_path / 'candidate.json').write_bytes(altered)
    with pytest.raises(ValueError, match='same-refusal'):
        application._assert_expected_record(tmp_path, 'candidate.json', record, 'same-refusal')


@pytest.mark.parametrize('kind', ('missing', 'symlink', 'fifo', 'oversized'))
def test_expected_projection_retains_descriptor_and_consumer_bounds(tmp_path, monkeypatch, kind):
    record = ExpectedProduct(Decimal('1.25'), ('first', 'second'))
    path = tmp_path / 'preissue-execution-plan.json'
    if kind == 'symlink':
        target = tmp_path / 'outside'
        target.write_bytes(record.canonical_bytes())
        path.symlink_to(target)
    elif kind == 'fifo':
        os.mkfifo(path)
    elif kind == 'oversized':
        monkeypatch.setattr(application, 'MAX_RUNTIME_PLAN_JSON_BYTES', 256)
        with path.open('wb') as handle:
            handle.truncate(257)
    with pytest.raises((OSError, bounded_io.BoundedFileIOError)):
        application._assert_expected_record(tmp_path, path.name, record, 'same-refusal')


def _packet_and_nomination(tmp_path):
    directory = tmp_path / 'nomination'
    directory.mkdir()
    nomination = current_nomination_operand_export(directory, nomination_id='synthetic.latency-nomination')
    units, seeds = original_development_exposure()
    return FiniteResponseLawRerunInput('synthetic.latency-calibration', 'calibration', allocation(), nomination, units, seeds), directory


def _runtime_context(packet):
    identity = ObjectIdentity('synthetic.software-only', ObjectIdentity.SCHEMA, '1.0.0', 'a' * 64)
    return FiniteResponseLawRuntimeContext('synthetic.latency-context', application._packet_run_id(packet),
        packet.source.fingerprint(), identity, identity, identity, 'operator.execution-service', 'synthetic.compiler')


@pytest.mark.parametrize('invalid', ('no-control', 'wrong-source', 'wrong-run', 'exposed-role'))
def test_invalid_runtime_packet_refuses_before_large_plan_read(tmp_path, monkeypatch, invalid):
    from tests.test_finite_response_assigned_evaluation_binding import _assigned_prospective_evaluation
    packet, _ = _packet_and_nomination(tmp_path)
    if invalid != 'no-control':
        _, source, control = _assigned_prospective_evaluation()
        selected = allocation(stage='prospective-evaluation', master=202610090064,
            role='EXPOSED_DEVELOPMENT_NONPROMOTABLE' if invalid == 'exposed-role' else 'PROPOSED_UNRUN')
        source = replace(source, cohort_namespace=selected.cohort_namespace,
            assignment_sha256=selected.fingerprint(), scientific_seeds=selected.scientific_seeds)
        control = replace(control, source=source)
        packet = FiniteResponseLawRerunInput('synthetic.latency-evaluation', 'prospective-evaluation', selected,
            packet.nomination, packet.original_prior_unit_ids, packet.original_prior_seed_ids,
            control.qualification.sha256, control=control)
    context = _runtime_context(packet)
    if invalid == 'wrong-source':
        context = replace(context, source_sha256='0' * 64)
    elif invalid == 'wrong-run':
        context = replace(context, run_id='synthetic.wrong-run')
    (tmp_path / 'rerun-input.json').write_bytes(packet.canonical_bytes())
    actual = application._read_record

    def read(directory, filename, record_type):
        if filename != 'rerun-input.json':
            pytest.fail('invalid packet reached the large plan')
        return actual(directory, filename, record_type)

    monkeypatch.setattr(application, '_read_record', read)
    monkeypatch.setattr(application, '_load_current_issued', lambda *args, **kwargs: pytest.fail('invalid packet reached protected custody'))
    with pytest.raises(ValueError, match='PROPOSED_SOURCE_OR_RUN_MISMATCH'):
        application.bind_finite_response_runtime(directory=tmp_path, context=context, artifact_writer=None)


def test_operational_run_identity_retains_exact_packet_commitment(tmp_path):
    packet, _ = _packet_and_nomination(tmp_path)
    assert application._packet_run_id(packet) == f'{packet.config_id}.run-{packet.fingerprint()[:32]}'
    edited = replace(packet, allocation=allocation(master=202610090032))
    assert edited.config_id == packet.config_id
    assert application._packet_run_id(edited) != application._packet_run_id(packet)


def test_changed_actual_source_refuses_before_parent_or_reconstruction(tmp_path, monkeypatch):
    from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind
    packet, _ = _packet_and_nomination(tmp_path)
    units, seeds = original_development_exposure()
    exposure = FiniteResponseLawCurrentExposure('synthetic.latency-source-exposure', units, seeds)
    closure = ImplementationSourceClosure('synthetic.latency-closure', SourceClosureKind.CLEAN_GIT_COMMIT,
        implementation_commit='0' * 40, implementation_sha256='d' * 64,
        source_tree_sha256='e' * 64, clean_worktree=True)
    for filename, record in (('rerun-input.json', packet), ('exposure.json', exposure), ('source-closure.json', closure)):
        (tmp_path / filename).write_bytes(record.canonical_bytes())
    monkeypatch.setattr(application, 'resolve_authoring_directory', lambda *args, **kwargs: (tmp_path, None))
    monkeypatch.setattr(application, 'capture_clean_target_closure', lambda *args: replace(closure, implementation_sha256='a' * 64))
    for name in ('_validate_parent', '_bundle', 'create_authoring_api'):
        monkeypatch.setattr(application, name, lambda *args, **kwargs: pytest.fail('changed source reached downstream inputs'))
    with pytest.raises(ValueError, match='ACTUAL_SOURCE_CLOSURE_CHANGED'):
        application.load_finite_response_rerun_handoff(directory=tmp_path, root=tmp_path,
            storage_profile=None, artifact_writer=None)


def test_matching_runtime_packet_still_contacts_exact_current_issue(tmp_path, monkeypatch):
    from tests.test_finite_response_assigned_evaluation_binding import _assigned_prospective_evaluation
    packet, _ = _packet_and_nomination(tmp_path)
    _, source, control = _assigned_prospective_evaluation()
    selected = allocation(stage='prospective-evaluation', master=202610090064)
    source = replace(source, cohort_namespace=selected.cohort_namespace,
        assignment_sha256=selected.fingerprint(), scientific_seeds=selected.scientific_seeds)
    control = replace(control, source=source)
    packet = FiniteResponseLawRerunInput('synthetic.latency-evaluation', 'prospective-evaluation', selected,
        packet.nomination, packet.original_prior_unit_ids, packet.original_prior_seed_ids,
        control.qualification.sha256, control=control)
    context = _runtime_context(packet)
    projection = SimpleNamespace(source_plan=SimpleNamespace(object_id=application._packet_run_id(packet)),
        candidate=ObjectIdentity('synthetic.current-candidate', ObjectIdentity.SCHEMA, '1.0.0', 'b' * 64))
    observed = []

    def deny(selected_context, plane, *, expected_run, expected_candidate):
        observed.append((selected_context.issued_study, expected_run, expected_candidate))
        raise PermissionError('actual-current-issue-refusal')

    monkeypatch.setattr(application, '_load_current_issued', deny)
    with pytest.raises(PermissionError, match='actual-current-issue-refusal'):
        application._authenticate_runtime_context(context, packet, projection, None)
    assert observed == [(context.issued_study, context.run_id, projection.candidate)]


def test_preparation_authenticates_parent_once_then_preserves_real_operand_joins(tmp_path, monkeypatch):
    """Reuse is local; the actual native decoder/receipt/source joins still run.

    Only the outer issue/grant authenticator is a software sentinel here. This
    fixture publishes fictional data and asserts no genuine study authority.
    """
    from empirical_lawhood.adapters.composition.finite_response_law.exposure import native_seed_ids
    from empirical_lawhood.adapters.composition.finite_response_law.rerun_input import FiniteResponseLawCurrentParent
    from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
    from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
    from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
    from empirical_lawhood.kernel.status import OperationalStatus
    from empirical_lawhood.planning.study_issue import StudyOperationAuthority
    from empirical_lawhood.runtime.artifacts import ArtifactProfile, ArtifactWriteRequest, CanonicalTaskReceipt, ExternalRootContract, ReceiptCheck
    from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile
    from empirical_lawhood.runtime.study_issue import IssuedExecutableStudyManifest
    from tests.finite_response_custody_fixtures import synthetic_native

    packet, directory = _packet_and_nomination(tmp_path)
    source = packet.source
    native = synthetic_native(source)
    raw = native.canonical_bytes()
    plane = ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract('synthetic.latency-store',
        'software-only operand test', str(tmp_path), '/', OperatorStorageProfile.SCHEMA, 1, None, None, ())))
    written = plane.write(ArtifactWriteRequest(logical_artifact_id='synthetic.latency-native',
        relative_path='fictional/native.json', payload_schema=native.SCHEMA, profile=ArtifactProfile.CANONICAL_JSON,
        media_type='application/json', publication_scope_id='synthetic.latency-native', publication_scope_relative_root='fictional',
        payload=raw, visibility_ceiling=VisibilityCeiling.PROSPECTIVE, parent_visibility_ceilings=(),
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL, minimum_free_bytes=1))
    manifest = decode_artifact_manifest(plane.root.resolve(written.manifest_materialization.relative_path,
        for_write=False).read_bytes())
    run, task = 'synthetic.latency-run', 'finite-response-law.native-evaluate'
    attempt = f'{run}.{task}.attempt-001'
    receipt = CanonicalTaskReceipt(f'receipt.{attempt}', run, task, attempt, '0' * 40, (),
        (written.materialization,), (written.logical,), (ReceiptCheck('fictional-software-only', True, ()),),
        OperationalStatus.SUCCEEDED, ())
    issued = ObjectIdentity('synthetic.latency-issued', IssuedExecutableStudyManifest.SCHEMA, '1.0.0', 'a' * 64)
    grant = ObjectIdentity('synthetic.latency-grant', StudyOperationAuthority.SCHEMA, '1.0.0', 'b' * 64)
    parent = FiniteResponseLawCurrentParent('synthetic.latency-parent', issued, grant, grant, source,
        (manifest,), ((run, task, receipt.receipt_id),))
    units, seeds = original_development_exposure()
    exposure = FiniteResponseLawCurrentExposure('synthetic.latency-exposure',
        tuple(sorted((*units, *packet.allocation.physical_unit_ids))), tuple(sorted(set((*seeds, *native_seed_ids(source))))))
    data = {written.logical.logical_artifact_id: raw, receipt.receipt_id: receipt.canonical_bytes()}
    authenticated = []

    def authenticate(selected, selected_plane, *, nomination):
        assert selected is parent and selected_plane is plane and nomination is packet.nomination
        authenticated.append((selected, nomination))
        return dict(data)

    monkeypatch.setattr(application, 'authenticate_finite_current_parent', authenticate)
    prepared = application.prepare_finite_response_rerun(config_id='synthetic.latency-method',
        stage='calibration-method', allocation=packet.allocation, exposure=exposure,
        nomination=packet.nomination, nomination_directory=directory, artifact_writer=plane, parent=parent)
    assert len(authenticated) == 1
    assert prepared.source == source
    assert prepared.method.expected_native_receipt == ObjectIdentity.from_record(receipt.receipt_id, receipt)
    assert prepared.method.native_evaluation.sha256 == written.logical.content_sha256
    assert prepared.nomination == packet.nomination
    # The next call must authenticate again, and corruption cannot enter the
    # method just because these operands were accepted by an earlier call.
    data[written.logical.logical_artifact_id] = raw + b' '
    with pytest.raises(ValueError):
        application.prepare_finite_response_rerun(config_id='synthetic.latency-corrupt-method',
            stage='calibration-method', allocation=packet.allocation, exposure=exposure,
            nomination=packet.nomination, nomination_directory=directory, artifact_writer=plane, parent=parent)
    assert len(authenticated) == 2
