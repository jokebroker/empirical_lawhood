"""Optional scientific/refusal probe on an authenticated EPW tutorial archive.

The supplied HDF5 file is external, outcome-visible reference input.  It is
never a fresh candidate unit or material multiband-response admission.
"""

from __future__ import annotations

import io
import os
import tarfile
from decimal import Decimal
from pathlib import Path

import h5py
import numpy as np
import pytest

pytestmark = pytest.mark.held

from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.material_control_material_operands import PAIRING_PROJECTION_LIMIT, _anisotropic_slices, parse_epw_egnv, parse_wannier_hr, reduce_material_operands
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.material_control_solver import parse_epw_a2f_integrated_lambda, resolve_material_control_workflow_profile


class _DatasetReader(io.RawIOBase):
    def __init__(self, dataset: h5py.Dataset) -> None:
        self._dataset = dataset
        self._position = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        count = min(len(buffer), len(self._dataset) - self._position)
        if count <= 0:
            return 0
        buffer[:count] = self._dataset[self._position : self._position + count].tobytes()
        self._position += count
        return count


def test_authentic_mgb2_two_gap_reference_stops_before_material_multiband_strong_coupling_response(tmp_path: Path) -> None:
    raw_path = os.environ.get("EMPIRICAL_LAWHOOD_MATERIAL_MGB2_REFERENCE_H5")
    if raw_path is None:
        pytest.skip("supply an authenticated, external EPW MgB2 raw-workflow HDF5")
    path = Path(raw_path)
    assert path.is_file() and not path.is_symlink()
    assert 0 < path.stat().st_size <= 600_000_000

    selected: dict[str, bytes] = {}
    position_member_present = False
    with h5py.File(path) as handle:
        archive_bytes = handle["archive_bytes"]
        assert archive_bytes.dtype == np.dtype("u1")
        with tarfile.open(
            fileobj=io.BufferedReader(
                _DatasetReader(archive_bytes), buffer_size=8 * 1024 * 1024
            ),
            mode="r|gz",
        ) as archive:
            for member in archive:
                if not member.isfile() or member.issym() or member.islnk():
                    continue
                name = member.name.rsplit("/", 1)[-1]
                if name == "mgb2_r.dat":
                    position_member_present = True
                if name not in {"mgb2_hr.dat", "egnv", "mgb2.imag_aniso_020.00", "mgb2.a2f"}:
                    continue
                assert name not in selected and member.size <= 8 * 1024 * 1024
                stream = archive.extractfile(member)
                assert stream is not None
                selected[name] = stream.read(member.size + 1)
                assert len(selected[name]) == member.size

    assert set(selected) == {"mgb2_hr.dat", "egnv", "mgb2.imag_aniso_020.00", "mgb2.a2f"}
    # R11's old extractor reported the peak of a spectral column (0.8923162)
    # as coupling.  The native table's first cumulative integral is 0.5729519.
    assert str(parse_epw_a2f_integrated_lambda(selected["mgb2.a2f"])) == "0.5729519"
    hamiltonian = parse_wannier_hr(selected["mgb2_hr.dat"])
    shell = parse_epw_egnv(selected["egnv"])
    slices, projection_residual, alignment_eV = _anisotropic_slices(
        selected["mgb2.imag_aniso_020.00"], hamiltonian=hamiltonian, shell=shell
    )
    assert hamiltonian.orbital_count == 5
    assert len(shell) > 100 and len(slices) >= 4
    assert alignment_eV < 1.0e-4

    # EPW reports delta in eV.  The independent physical reference is MgB2's
    # two-gap structure: the lower and upper quartiles straddle 2 and 5 meV.
    rows = np.loadtxt(io.BytesIO(selected["mgb2.imag_aniso_020.00"]), comments="#")
    first_frequency = rows[rows[:, 0] == rows[0, 0], 3]
    assert np.quantile(first_frequency, 0.25) * 1000 < 2.0
    assert np.quantile(first_frequency, 0.75) * 1000 > 5.0

    # The archived calculation is real native output, but it is not a valid
    # same-gauge material-R2 input.  Both gates must remain a refusal.
    assert not position_member_present
    assert projection_residual > float(PAIRING_PROJECTION_LIMIT)

    # Rebind only the authenticated native members in a bounded local archive
    # and exercise the actual reducer's first precontact stop.  The missing
    # same-gauge position matrix must be reported before any R2 value exists.
    diagnostic = tmp_path / "mgb2-reference.tar.gz"
    with tarfile.open(diagnostic, "w:gz") as archive:
        for name in ("mgb2_hr.dat", "egnv", "mgb2.imag_aniso_020.00"):
            payload = selected[name]
            member = tarfile.TarInfo(f"source/{name}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    with pytest.raises(ValueError, match="exactly one mgb2_r.dat member"):
        reduce_material_operands(
            raw_archive_path=diagnostic,
            profile=resolve_material_control_workflow_profile("workflow.material-control-tutorial04-mgb2"),
            fermi_energy_eV=Decimal("9.233971637106672"),
            electron_phonon_lambda=Decimal("0.5729519"),
        )
