"""Material control calibration held-input roster and native HDF5 envelope checks."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import h5py
import pytest

from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.material_control_input_preflight import MaterialControlInputPreflight, inspect_material_control_input_roster, inspect_material_control_raw_input, inspect_material_control_tar_input
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.material_control_raw import encode_raw_archive_hdf5
from empirical_lawhood.api.codecs import load_registered_authoring

STARTER = Path("experiments/material-control-inputs/control-inputs.json")


def test_material_control_builder_selects_solver_views_before_explicit_source_refusal(monkeypatch) -> None:
    from empirical_lawhood._required_inputs import ExternalInputRequired
    from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.material_control_inputs import build_material_control_input_files

    monkeypatch.delenv("EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT", raising=False)
    with pytest.raises(ExternalInputRequired, match="EMPIRICAL_LAWHOOD_AP_EXTERNAL_ROOT"):
        build_material_control_input_files()


def _config() -> MaterialControlInputPreflight:
    return load_registered_authoring(
        STARTER,
        root_schemas={MaterialControlInputPreflight.SCHEMA: MaterialControlInputPreflight},
        maximum_bytes=32 * 1024,
    )


def test_shipped_control_roster_counts_structures_and_refuses_missing_bytes(
    tmp_path: Path,
) -> None:
    request = _config()
    assert len(request.workflow_profile_ids) == 10
    assert (
        len(
            {
                profile.removeprefix("workflow.material-control-").split("-calibration-")[1]
                for profile in request.workflow_profile_ids
            }
        )
        == 5
    )
    with pytest.raises(ValueError, match="raw workflow is absent"):
        inspect_material_control_input_roster(request, source_root=tmp_path)


def test_material_control_envelope_rejects_wrong_profile_and_changed_logical_bytes(
    tmp_path: Path,
) -> None:
    profile_id = _config().workflow_profile_ids[0]
    payload = b"native-output-placeholder\n"
    archive = tmp_path / "raw-workflow.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("workflow/scf.stdout")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    from hashlib import sha256

    wrapped = tmp_path / "raw-workflow.h5"
    encode_raw_archive_hdf5(
        archive_path=archive,
        output_path=wrapped,
        profile_id=profile_id,
        archive_sha256=sha256(archive.read_bytes()).hexdigest(),
    )
    result = inspect_material_control_raw_input(wrapped, profile_id=profile_id)
    assert result["logical_archive_sha256"] == sha256(archive.read_bytes()).hexdigest()
    assert result["structure_id"] == "structure.calibration-c-diamond"
    with pytest.raises(ValueError, match="profile differs"):
        inspect_material_control_raw_input(wrapped, profile_id=_config().workflow_profile_ids[1])
    with h5py.File(wrapped, "r+") as handle:
        handle["archive_bytes"][0] ^= 1
    with pytest.raises(ValueError, match="logical archive digest differs"):
        inspect_material_control_raw_input(wrapped, profile_id=profile_id)


def test_positive_native_tar_requires_separate_same_gauge_position(
    tmp_path: Path,
) -> None:
    profile_id = next(
        value
        for value in _config().workflow_profile_ids
        if value.endswith("calibration-mgb2-alb2")
    )
    archive = tmp_path / "mgb2.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = b"hamiltonian\n"
        member = tarfile.TarInfo("workflow/mgb2_hr.dat")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="exactly one mgb2_r.dat"):
        inspect_material_control_tar_input(archive, profile_id=profile_id)
