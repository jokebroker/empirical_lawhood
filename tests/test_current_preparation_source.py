"""Current source failure/custody and complete saved-array production seams.

Synthetic saved arrays are software fixtures, not native or qualification
evidence. Native acquisition is never called by this file's tests.
"""

from dataclasses import replace
from types import SimpleNamespace
from hashlib import sha256
from pathlib import Path
from empirical_lawhood.api.matrix_numerical_provenance import selected_dependency_lock_sha256

ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pytest

from tests.test_matrix_operand_transport import allocation as allocated
from tests.test_matrix_history_public_api import plane as plane
from tests.numerical_provenance_fixtures import numerical_plane
from empirical_lawhood.api.finite_operands import (
    current_preparation_code_sources_sha256, current_preparation_source_export,
    matrix_preparation_source_export, original_f_operand_export,
    preparation_bridge_fits, preparation_operands_export, saved_matrix_arrays_export,
    saved_matrix_arrays_import,
)
from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge import mechanical_kernel
from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source import acquire_current_preparation_root
from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source_records import (
    MatrixPreparationSourceConfig, MatrixPreparationSourceProtocol,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity


def source_config(allocation):
    return MatrixPreparationSourceConfig("test.matrix-source", allocation.identity, MatrixPreparationSourceProtocol(), current_preparation_code_sources_sha256(), dependency_lock_sha256=selected_dependency_lock_sha256(ROOT))


def test_failed_prefix_retains_two_failures_and_complete_unentered_denominator(monkeypatch):
    current = allocated()
    phases = tuple(SimpleNamespace(refinement=view, disposition="NUMERICAL_FAILURE", reason="NATIVE_NUMERICAL_FAILURE") for view in (1, 2))
    monkeypatch.setattr("empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source.acquire_prefix", lambda *args: SimpleNamespace(phases=phases, frame_base64=None, features=()))
    retained = []
    def retain(cell_id, record):
        retained.append(cell_id)
        return ArtifactIdentity(cell_id, "native-test-fixture", "empirical-lawhood/preparation-applicability/native-phase", "a" * 64, "application/json", 100)
    result = acquire_current_preparation_root(config=source_config(current), root=current.roots[0], retain=retain)
    assert len(result.report.cells) == 344
    assert len(retained) == 2
    assert sum(cell.disposition == "UNENTERED" for cell in result.report.cells) == 342
    assert result.report.disposition == "UNEVALUABLE"
    assert not result.arrays["response_observed"].any()
    assert np.isnan(result.arrays["y"]).all()


def test_interruption_keeps_native_member_and_has_no_completed_manifest(tmp_path, monkeypatch, plane):
    tmp_path = Path(plane.root.contract.canonical_path) / "source"
    tmp_path.mkdir()
    current = allocated()
    def interrupt(**kwargs):
        record = ObjectIdentity("test.partial", "empirical-lawhood/kernel/matrix-root-allocation", "1.0.0", "a" * 64)
        kwargs["retain"]("test.partial-native-cell", record)
        raise KeyboardInterrupt
    monkeypatch.setattr("empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source.acquire_current_preparation_root", interrupt)
    with pytest.raises(KeyboardInterrupt):
        matrix_preparation_source_export(tmp_path, config=source_config(current), allocation=current, project_root=ROOT, artifact_writer=plane)
    assert (tmp_path / "test.partial-native-cell.canonical.json").is_file()
    assert not (tmp_path / "arrays.canonical.json").exists()
    assert not (tmp_path / "source-report.canonical.json").exists()


def test_source_code_and_actual_roster_refuse_before_effects(tmp_path, monkeypatch):
    current = allocated()
    def forbidden(**kwargs):
        raise AssertionError("native acquisition entered")
    monkeypatch.setattr("empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source.acquire_current_preparation_root", forbidden)
    with pytest.raises(ValueError, match="code"):
        matrix_preparation_source_export(tmp_path, config=replace(source_config(current), code_sources_sha256="0" * 64), allocation=current, project_root=ROOT, artifact_writer=None)
    assert not tuple(tmp_path.iterdir())
    with pytest.raises(ValueError, match="source"):
        current_preparation_source_export(tmp_path, config=source_config(current), allocation=current, original_f=SimpleNamespace(identity=ObjectIdentity("test.lower", "empirical-lawhood/methods/finite-response-law/original-finite-response-law", "1.0.0", "a" * 64)), project_root=ROOT, artifact_writer=None)


def test_complete_saved_native_to_bridge_readiness_export(tmp_path):
    writer = numerical_plane(tmp_path)
    source_dir = tmp_path / "native"
    output_dir = tmp_path / "derived"
    lower_dir = tmp_path / "lower"
    for directory in (source_dir, output_dir, lower_dir):
        directory.mkdir()
    lower = original_f_operand_export(lower_dir, payload_id="test.lower")
    current = allocated(24)
    current = replace(current, roots=tuple(replace(root, cohort="q2" if index < 8 else "cir1") for index, root in enumerate(current.roots)))
    rng = np.random.default_rng(2026)
    center, scale = np.asarray(lower.center, dtype=np.float64), np.asarray(lower.scale, dtype=np.float64)
    primary = center + 0.01 * rng.normal(size=(24, 24)) * scale
    hold = center + 0.01 * rng.normal(size=(24, 24)) * scale
    kernel, _ = mechanical_kernel()
    shift = np.einsum("stk,rkj->rstj", kernel, rng.normal(size=(24, 4, 24))) * scale
    baseline = primary[:, None, None] + np.linspace(0, 1, 26)[None, None, :, None] * (hold - primary)[:, None, None]
    features = np.repeat((baseline + shift)[:, :, None], 2, axis=2)
    z = features[:, :, :, -1].transpose(0, 1, 3, 2).copy()
    mean = lower.predict(z[..., 0].reshape(-1, 24)).mean.reshape(24, 9, 4, 8)
    arrays = {"x": np.repeat(primary[..., None], 2, axis=-1), "z": z, "features": features, "y": np.broadcast_to(mean[..., None, None], (24, 9, 4, 8, 2, 2)).copy(), "work": np.zeros((24, 9, 2)), "positions": np.zeros((24, 9, 2, 26, 2, 3, 4, 4), dtype=np.complex128), "momenta": np.zeros((24, 9, 2, 26, 2, 3, 4, 4), dtype=np.complex128)}
    producer = ObjectIdentity("test.synthetic-current-source", MatrixPreparationSourceConfig.SCHEMA, "1.0.0", sha256(b"explicit software fixture; no native evidence").hexdigest())
    saved_matrix_arrays_export(source_dir, operand_id="test.native", allocation=current, arrays=arrays, producers=(producer,), original_f=lower.identity)
    result = preparation_operands_export(output_dir, native_manifest_path=source_dir / "arrays.canonical.json", native_arrays_path=source_dir / "arrays.npz", allocation_path=source_dir / "allocation.canonical.json", original_f=lower, operand_id="test.readiness", artifact_writer=writer, project_root=ROOT)
    imported, _, output = saved_matrix_arrays_import(output_dir / "arrays.canonical.json", arrays_path=output_dir / "arrays.npz", allocation_path=output_dir / "allocation.canonical.json", artifact_writer=writer)
    assert imported == result
    fits = preparation_bridge_fits(output)
    assert len(fits) == 4
    assert all(row["kernel_rank"] == 4 for row in fits)
    assert output["conjuncts"].shape == (24, 9, 7)
    assert output["upper_mean"].shape == (24, 9, 4, 8)
    np.testing.assert_array_equal(output["operator"], np.asarray(lower.operator, dtype=np.float64).reshape(25, 32))
    assert result.original_f == lower.identity
    assert (output_dir / "bridge-input.canonical.json").is_file()
    assert (output_dir / "bridge-spec.canonical.json").is_file()
