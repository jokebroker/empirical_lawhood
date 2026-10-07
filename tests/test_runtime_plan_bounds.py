"""Typed plan envelopes preserve compact-control limits and custody refusal."""

from dataclasses import replace
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from empirical_lawhood.api import authoring_handoff, codecs, finite_rerun_authoring, preparation_applicability
from empirical_lawhood.api.execution import (
    _CatalogSemanticReplayBudget, CampaignCapabilityError,
    CampaignExecutionService, CampaignIdentityError,
)
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.infrastructure.bounded_io import (
    BoundedFileIOError, MAX_CONTROL_PLANE_JSON_BYTES, MAX_RUNTIME_PLAN_JSON_BYTES,
)
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.artifacts import (
    ArtifactManifest, ArtifactProfile, ArtifactWriteRequest, ExternalRootContract,
    artifact_publication_batch_id, artifact_publication_commit_relative_path,
    artifact_publication_member,
)
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile
from empirical_lawhood.runtime.plans import CandidateExecutionPlan


FIXTURE = Path(__file__).parent / 'fixtures/authoring-execution-plan.json'


def _sparse(path, size):
    with path.open('xb') as handle:
        handle.truncate(size)


@pytest.mark.parametrize('owner', ('shared', 'finite', 'preparation'))
def test_oversize_typed_plan_is_refused_before_decode(tmp_path, monkeypatch, owner):
    """Descriptor size refusal happens before hostile JSON allocation/decoding."""
    _sparse(tmp_path / 'preissue-execution-plan.json', MAX_RUNTIME_PLAN_JSON_BYTES + 1)

    def forbidden(*args, **kwargs):
        raise AssertionError('oversize plan reached its decoder')

    selected = {'shared': authoring_handoff, 'finite': finite_rerun_authoring, 'preparation': preparation_applicability}[owner]
    monkeypatch.setattr(selected, 'decode_canonical_bytes', forbidden)
    with pytest.raises(BoundedFileIOError, match='exceeds its byte limit'):
        if owner == 'shared':
            authoring_handoff.load_authoring_execution_projection(tmp_path)
        elif owner == 'finite':
            finite_rerun_authoring._read_record(tmp_path, 'preissue-execution-plan.json', CandidateExecutionPlan)
        else:
            preparation_applicability._read_handoff_record(tmp_path, 'preissue-execution-plan.json', CandidateExecutionPlan)


@pytest.mark.parametrize('owner', ('finite', 'preparation'))
def test_scientific_control_retains_compact_limit(tmp_path, monkeypatch, owner):
    _sparse(tmp_path / 'source-closure.json', MAX_CONTROL_PLANE_JSON_BYTES + 1)

    def forbidden(*args, **kwargs):
        raise AssertionError('oversize compact control reached its decoder')

    selected = finite_rerun_authoring if owner == 'finite' else preparation_applicability
    monkeypatch.setattr(selected, 'decode_canonical_bytes', forbidden)
    with pytest.raises(BoundedFileIOError, match='exceeds its byte limit'):
        if owner == 'finite':
            selected._read_record(tmp_path, 'source-closure.json', ObjectIdentity)
        else:
            selected._read_handoff_record(tmp_path, 'source-closure.json', ObjectIdentity)


def test_closed_plan_codec_refuses_sparse_oversize_before_decode(tmp_path, monkeypatch):
    path = tmp_path / 'plan.json'
    _sparse(path, MAX_RUNTIME_PLAN_JSON_BYTES + 1)

    def forbidden(*args, **kwargs):
        raise AssertionError('oversize closed plan root reached its decoder')

    monkeypatch.setattr(codecs, 'decode_record', forbidden)
    with pytest.raises(codecs.AuthoringCodecError, match='byte limit'):
        codecs.load_registered_authoring(path, root_schemas={CandidateExecutionPlan.SCHEMA: CandidateExecutionPlan})


def test_mixed_root_codec_applies_typed_plan_bound_after_closed_selection(monkeypatch):
    raw = FIXTURE.read_text()
    monkeypatch.setattr(codecs, 'MAX_RUNTIME_PLAN_JSON_BYTES', len(raw.encode()) - 1)
    registry = {CandidateExecutionPlan.SCHEMA: CandidateExecutionPlan, ObjectIdentity.SCHEMA: ObjectIdentity}
    with pytest.raises(codecs.AuthoringCodecError, match='runtime plan exceeds'):
        codecs.loads_registered_authoring(raw, media_type='application/json', root_schemas=registry)
    # The same mixed registry retains its ordinary authoring envelope for roots
    # outside the plan hierarchy, instead of lowering every input's bound.
    identity = ObjectIdentity('synthetic.large-nonplan-root', ObjectIdentity.SCHEMA, '1.0.0', 'a' * 64)
    monkeypatch.setattr(codecs, 'MAX_RUNTIME_PLAN_JSON_BYTES', 1)
    assert codecs.loads_registered_authoring(identity.canonical_bytes().decode(), media_type='application/json', root_schemas=registry) == identity


def _stored_plan(tmp_path):
    plane = ExternalArtifactPlane(GuardedExternalRoot(ExternalRootContract(
        'synthetic.plan-bound-store', 'software-only bounded plan custody', str(tmp_path),
        '/', OperatorStorageProfile.SCHEMA, 1, None, None, (),
    )))
    (tmp_path / 'preissue-execution-plan.json').write_bytes(FIXTURE.read_bytes())
    plan = authoring_handoff.load_authoring_execution_projection(tmp_path)
    run_id = plan.source_plan.object_id
    retained = plane.write(ArtifactWriteRequest(
        logical_artifact_id=f'execution-plan.{run_id}',
        relative_path=f'runs/{run_id}/plans/execution-plan.json',
        payload_schema=plan.SCHEMA, profile=ArtifactProfile.CANONICAL_JSON,
        media_type='application/json', publication_scope_id='synthetic.plan-controls',
        publication_scope_relative_root=f'runs/{run_id}/plans',
        payload=plan.canonical_bytes(), visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        parent_visibility_ceilings=(), outcome_access=OutcomeAccess.OUTCOME_BLIND,
        minimum_free_bytes=1,
    ))
    service = SimpleNamespace(artifact_plane=plane)
    for name in ('_verified_rebuild_control_payload', '_rebuild_control_pair_present'):
        setattr(service, name, MethodType(getattr(CampaignExecutionService, name), service))
    return plane, service, plan, retained


def test_status_reads_canonical_stored_plan_and_refuses_tampered_bytes(tmp_path):
    plane, service, plan, retained = _stored_plan(tmp_path)
    assert CampaignExecutionService._load_status_execution_plan(service, plan.source_plan.object_id) == plan
    path = plane.root.resolve(retained.materialization.relative_path, for_write=False)
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(CampaignIdentityError, match='custody is invalid'):
        CampaignExecutionService._load_status_execution_plan(service, plan.source_plan.object_id)


@pytest.mark.parametrize('missing_bytes', (0, 1))
def test_replay_aggregate_charges_both_manifest_and_plan_before_open(tmp_path, monkeypatch, missing_bytes):
    plane, service, plan, retained = _stored_plan(tmp_path)
    manifest_path = plane.root.resolve(retained.materialization.relative_path + '.manifest.json', for_write=False)
    aggregate = len(manifest_path.read_bytes()) + retained.materialization.size_bytes
    budget = _CatalogSemanticReplayBudget(remaining_bytes=aggregate - missing_bytes)
    original_open = plane.open
    opens = []

    def observed(*args, **kwargs):
        opens.append(kwargs['maximum_bytes'])
        return original_open(*args, **kwargs)

    monkeypatch.setattr(plane, 'open', observed)
    arguments = dict(logical_artifact_id=retained.logical.logical_artifact_id,
        relative_path=retained.materialization.relative_path,
        payload_schemas=(plan.SCHEMA,), maximum_bytes=MAX_RUNTIME_PLAN_JSON_BYTES,
        budget=budget)
    if missing_bytes:
        with pytest.raises(CampaignCapabilityError, match='aggregate byte limit'):
            service._verified_rebuild_control_payload(**arguments)
        assert opens == []
    else:
        payload, fingerprint, schema = service._verified_rebuild_control_payload(**arguments)
        assert payload == plan.canonical_bytes() and fingerprint == plan.fingerprint() and schema == plan.SCHEMA
        assert opens == [retained.materialization.size_bytes] and budget.remaining_bytes == 0


@pytest.mark.parametrize('purpose', ('plan', 'compact-provider-receipt'))
def test_declared_oversize_control_is_refused_before_protected_open(tmp_path, monkeypatch, purpose):
    plane, service, plan, retained = _stored_plan(tmp_path)
    maximum = MAX_RUNTIME_PLAN_JSON_BYTES if purpose == 'plan' else MAX_CONTROL_PLANE_JSON_BYTES
    # The manifest's declaration is itself sufficient to refuse a protected
    # payload read. This intentionally unauthenticated tamper claims no issue.
    manifest_path = plane.root.resolve(retained.materialization.relative_path + '.manifest.json', for_write=False)
    original = decode_artifact_manifest(manifest_path.read_bytes())
    materialization = replace(retained.materialization, size_bytes=maximum + 1)
    members = (artifact_publication_member(retained.logical, materialization),)
    scope = original.publication.publication_scope
    batch_id = artifact_publication_batch_id(scope, members)
    publication = replace(original.publication, members=members,
        publication_batch_id=batch_id,
        commit_marker_relative_path=artifact_publication_commit_relative_path(scope, batch_id))
    altered = ArtifactManifest(retained.logical, materialization, publication)
    manifest_path.write_bytes(altered.canonical_bytes())

    def forbidden(*args, **kwargs):
        raise AssertionError('oversize declared control reached protected open')

    monkeypatch.setattr(plane, 'open', forbidden)
    if purpose == 'plan':
        with pytest.raises(CampaignIdentityError, match='frozen identity'):
            CampaignExecutionService._load_status_execution_plan(service, plan.source_plan.object_id)
    else:
        with pytest.raises(CampaignIdentityError, match='frozen identity'):
            service._verified_rebuild_control_payload(
                logical_artifact_id=retained.logical.logical_artifact_id,
                relative_path=retained.materialization.relative_path,
                payload_schemas=(plan.SCHEMA,), maximum_bytes=maximum,
                budget=_CatalogSemanticReplayBudget(),
            )
