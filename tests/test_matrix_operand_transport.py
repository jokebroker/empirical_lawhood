"""Consumed streams and bounded exact array custody, without native execution."""

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import zipfile

import numpy as np
import pytest

from empirical_lawhood.api.finite_operands import saved_matrix_arrays_export, saved_matrix_arrays_import
from empirical_lawhood.infrastructure.matrix_array_io import decode_matrix_arrays
from empirical_lawhood.kernel.matrix_inputs import MATRIX_NATIVE_PURPOSES, MatrixAllocation, MatrixArrayMember, MatrixPurposeSeed, MatrixRootAllocation
from empirical_lawhood.kernel.provenance import ObjectIdentity


def allocation(count=2):
    roots = tuple(MatrixRootAllocation(f"test.root.{index}", "q2", tuple(sorted((*(MatrixPurposeSeed(purpose, 100 + index * 100 + j) for j, purpose in enumerate(MATRIX_NATIVE_PURPOSES)), MatrixPurposeSeed("passive-probes", (1000 + index) << 128, "PCG64DXSM")), key=lambda row: row.purpose_id))) for index in range(count))
    return MatrixAllocation("test.allocation", roots, bootstrap_seed=9000)


def test_allocation_excludes_effective_probe_aliases():
    current = allocation()
    root = current.roots[1]
    original = current.roots[0].seed_for("passive-probes")
    seeds = tuple(replace(seed, seed=original + 1) if seed.purpose_id == "passive-probes" else seed for seed in root.scientific_seeds)
    with pytest.raises(ValueError, match="collide"):
        replace(current, roots=(current.roots[0], replace(root, scientific_seeds=seeds)))
    assert replace(current, allocation_id="test.renamed").roots == current.roots


def test_allocation_requires_complete_purposes_and_generator():
    current = allocation()
    with pytest.raises(ValueError, match="census"):
        replace(current.roots[0], scientific_seeds=current.roots[0].scientific_seeds[:-1])
    with pytest.raises(ValueError, match="generator"):
        MatrixPurposeSeed("passive-probes", 1000)
    proposed = replace(current, exposure="PROPOSED_UNRUN")
    assert proposed.exposure == "PROPOSED_UNRUN"
    with pytest.raises(ValueError, match="ceiling"):
        replace(current, exposure="QUALIFIED")


def test_native_probe_draws_follow_numeric_seed_and_not_allocation_labels():
    from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import derive_probe_roster
    root = allocation().roots[0]
    renamed = replace(root, root_id="test.renamed-root")
    def draw(selected):
        return derive_probe_roster(config_fingerprint=selected.fingerprint(),
            rule_id="test.passive-probes", scientific_seed=selected.seed_for("passive-probes"))
    first, other_label = draw(root), draw(renamed)
    np.testing.assert_array_equal(first.fields, other_label.fields)
    changed = replace(root, scientific_seeds=tuple(replace(seed, seed=seed.seed + (1 << 128))
        if seed.purpose_id == "passive-probes" else seed for seed in root.scientific_seeds))
    assert not np.array_equal(first.fields, draw(changed).fields)


def test_array_public_roundtrip_preserves_missingness_and_fortran_values(tmp_path):
    current = allocation()
    values = {"features": np.asfortranarray(np.arange(12, dtype=np.float64).reshape(3, 4)), "observed": np.array([True, False]), "outcomes": np.array([1.0, np.nan]), "count": np.array(2, dtype=np.int64), "known_failure_indices": np.empty((0, 2), dtype=np.int64)}
    producer = ObjectIdentity("test.producer", "empirical-lawhood/kernel/matrix-allocation", "1.0.0", "a" * 64)
    manifest = saved_matrix_arrays_export(tmp_path, operand_id="test.arrays", allocation=current, arrays=values, producers=(producer,), nonfinite_members=("outcomes",))
    imported, imported_allocation, arrays = saved_matrix_arrays_import(tmp_path / "arrays.canonical.json", arrays_path=tmp_path / "arrays.npz", allocation_path=tmp_path / "allocation.canonical.json")
    assert imported == manifest
    assert imported_allocation == current
    for key, value in values.items():
        np.testing.assert_array_equal(arrays[key], value)
        assert not arrays[key].flags.writeable
    with pytest.raises(TypeError):
        arrays["new"] = np.zeros(1)


def test_array_transport_refuses_undeclared_nonfinite_and_objects(tmp_path):
    producer = ObjectIdentity("test.producer", MatrixAllocation.SCHEMA, "1.0.0", "a" * 64)
    for arrays in ({"outcomes": np.array([np.nan])}, {"objects": np.array([object()], dtype=object)}):
        with pytest.raises(ValueError):
            saved_matrix_arrays_export(tmp_path, operand_id="test.arrays", allocation=allocation(), arrays=arrays, producers=(producer,))
    assert not tuple(tmp_path.iterdir())


def test_array_manifest_authenticates_each_member_before_loading(tmp_path):
    producer = ObjectIdentity("test.producer", MatrixAllocation.SCHEMA, "1.0.0", "a" * 64)
    manifest = saved_matrix_arrays_export(tmp_path, operand_id="test.arrays", allocation=allocation(), arrays={"features": np.zeros((2, 24))}, producers=(producer,))
    raw = (tmp_path / "arrays.npz").read_bytes()
    altered = replace(manifest, members=(replace(manifest.members[0], shape=(1, 48)),))
    with pytest.raises(ValueError, match="axes"):
        decode_matrix_arrays(raw, altered)
    altered = replace(manifest, members=(replace(manifest.members[0], sha256="0" * 64),))
    with pytest.raises(ValueError, match="values"):
        decode_matrix_arrays(raw, altered)
    with pytest.raises(ValueError, match="artifact"):
        decode_matrix_arrays(raw + b" ", manifest)


def test_huge_header_refused_before_npy_allocation(tmp_path):
    producer = ObjectIdentity("test.producer", MatrixAllocation.SCHEMA, "1.0.0", "a" * 64)
    manifest = saved_matrix_arrays_export(tmp_path, operand_id="test.arrays", allocation=allocation(), arrays={"features": np.zeros((2, 24))}, producers=(producer,))
    header = BytesIO()
    np.lib.format.write_array_header_1_0(header, {"descr": "<f8", "fortran_order": False, "shape": (2**60,)})
    archive = BytesIO()
    with zipfile.ZipFile(archive, "w") as payload:
        payload.writestr("features.npy", header.getvalue())
    raw = archive.getvalue()
    altered = replace(manifest, array_artifact=replace(manifest.array_artifact, sha256=sha256(raw).hexdigest(), size_bytes=len(raw)))
    with pytest.raises(ValueError, match="axes"):
        decode_matrix_arrays(raw, altered)


def test_array_member_enforces_byte_and_dtype_limits():
    with pytest.raises(ValueError, match="contract"):
        MatrixArrayMember("values", "<f8", (1000000, 1000000), "0" * 64)
    with pytest.raises(ValueError, match="contract"):
        MatrixArrayMember("values", "|O", (1,), "0" * 64)
