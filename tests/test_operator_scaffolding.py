# SPDX-License-Identifier: MPL-2.0
"""Inert constructor checks: none of these records are real qualification inputs."""
from dataclasses import replace
from pathlib import Path

import pytest

from scripts.operator_records import export_record, make_assigned_reactor_profile, make_storage_profile, read_record
from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import AssignedReactorAuthoringProfile, fresh_assignment_status
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.operator_profile import OperatorStorageAccessMode, OperatorStorageProfile


def profile(**changes):
    arguments = dict(
        profile_id='fixture.profile', experiment_id='fixture.experiment',
        public_source_sha256='a' * 64, census_id='fixture.census',
        prior_unit_ids=('unit.fixture.previous',), prior_seed_ids=('seed.native-reactor.1',),
        source_inventory=ObjectIdentity('fixture.inventory', 'empirical-lawhood/testing/fixtures/inventory', '1.0.0', 'b' * 64),
        noise_seeds=(101, 102, 103, 104, 105), evidence_role='EXPOSED_DEVELOPMENT_NONPROMOTABLE',
    )
    return make_assigned_reactor_profile(**(arguments | changes))


def test_constructor_exports_exact_records_and_keeps_exposure(tmp_path):
    value = profile()
    path = export_record(tmp_path / 'profile.json', value)
    assert read_record(path, AssignedReactorAuthoringProfile) == value
    assert fresh_assignment_status(value)['prospective_issue_eligible'] is False
    assert export_record(path, value) == path
    with pytest.raises(FileExistsError, match='differs'):
        export_record(path, replace(value, profile_id='fixture.other'))


@pytest.mark.parametrize('changes', [
    {'noise_seeds': (1, 102, 103, 104, 105)},
    {'prior_seed_ids': ('101',)},
    {'prior_unit_ids': ('unit.fixture.experiment.cohort.nominal',)},
    {'noise_seeds': (101, 101, 103, 104, 105)},
])
def test_constructor_refuses_alias_collisions(changes):
    with pytest.raises(ValueError, match='COLLISION'):
        profile(**changes)


def test_constructor_reports_missing_inputs_and_bad_count():
    with pytest.raises(TypeError, match='required keyword-only'):
        make_assigned_reactor_profile(profile_id='fixture.missing')
    with pytest.raises(ValueError, match='five explicit'):
        profile(noise_seeds=(2,))
    with pytest.raises(ValueError):
        profile(prior_unit_ids=())


def test_storage_constructor_is_structural_and_has_no_authority(tmp_path):
    value = make_storage_profile(
        profile_id='fixture.storage', external_root=Path('/mnt/research/run'),
        required_mount=Path('/mnt/research'), expected_mount_source=None,
        expected_volume_identity='owner-volume-id', filesystem_types=('ext4',),
        minimum_free_bytes=1_000_000, access_mode=OperatorStorageAccessMode.READ_ONLY,
        maximum_parallel_tasks=1,
    )
    assert value.authority_granted is False
    assert value.containment_policy_key == 'strict-mount-contained-no-symlink'
    assert read_record(export_record(tmp_path / 'storage.json', value), OperatorStorageProfile) == value
