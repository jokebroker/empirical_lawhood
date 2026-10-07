# SPDX-License-Identifier: MPL-2.0
"""Independent precontact drift refusals; no native campaign is performed."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from empirical_lawhood.api import finite_operands, matrix_history_analysis
from empirical_lawhood.api import matrix_numerical_provenance as provenance
from empirical_lawhood.kernel.provenance import ObjectIdentity
from tests.test_matrix_operand_transport import allocation
from tests.test_matrix_history_public_api import plane as plane

ROOT = Path(__file__).resolve().parents[1]


class ProducerReached(Exception):
    pass


def invocation(route, destination, writer, monkeypatch):
    def producer(*args, **kwargs):
        raise ProducerReached
    if route == "history":
        inputs = matrix_history_analysis.prepare_matrix_history_allocation(
            allocation_id="test.runtime.allocation", namespace="test.runtime", master_seed=865)
        config = matrix_history_analysis.prepare_matrix_history_source_configuration(
            config_id="test.runtime.history", allocation=inputs, repo_root=ROOT)
        monkeypatch.setattr("empirical_lawhood.adapters.simulators.six_matrix_response.history_source.acquire_history", producer)
        return lambda: matrix_history_analysis.produce_matrix_histories(config=config, allocation=inputs,
            writer=writer, relative_root="source", repo_root=ROOT, root_indices=(0,))
    inputs = allocation(24 if route == "fixed24" else 2)
    if route == "fixed24":
        inputs = replace(inputs, roots=tuple(replace(root, cohort="q2" if i < 8 else "cir1")
                                             for i, root in enumerate(inputs.roots)))
    lower = SimpleNamespace(identity=ObjectIdentity("test.lower",
        "empirical-lawhood/methods/finite-response-law/original-finite-response-law", "1.0.0", "a" * 64))
    config = finite_operands.prepare_matrix_source_configuration(config_id="test.runtime.source",
        allocation=inputs, original_f=lower if route == "fixed24" else None, project_root=ROOT)
    monkeypatch.setattr("empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source.acquire_current_preparation_root", producer)
    if route == "fixed24":
        return lambda: finite_operands.current_preparation_source_export(destination, config=config,
            allocation=inputs, original_f=lower, project_root=ROOT, artifact_writer=writer)
    return lambda: finite_operands.matrix_preparation_source_export(destination, config=config,
        allocation=inputs, project_root=ROOT, artifact_writer=writer)


@pytest.mark.parametrize("route", ("generic", "fixed24", "history"))
@pytest.mark.parametrize("drift", ("python", "numpy", "scipy", "threadpool", "lock", "gradient"))
def test_drift_refuses_before_native_and_output(route, drift, tmp_path, monkeypatch, plane):
    destination = Path(plane.root.contract.canonical_path) / "new-source"
    destination.mkdir()
    invoke = invocation(route, destination, plane, monkeypatch)
    if drift == "python":
        monkeypatch.setattr("platform.python_version", lambda: "3.11.99")
    elif drift == "numpy":
        monkeypatch.setattr(np, "__version__", "0.0.0-review-drift")
    elif drift == "scipy":
        monkeypatch.setattr("scipy.__version__", "0.0.0-review-drift")
    elif drift == "threadpool":
        monkeypatch.setattr("threadpoolctl.threadpool_info", lambda: [{"num_threads": 2}])
    elif drift == "lock":
        monkeypatch.setattr(provenance, "selected_dependency_lock_sha256", lambda root: "0" * 64)
    else:
        original = provenance.read_bounded_bytes
        def changed(path, **kwargs):
            raw = original(path, **kwargs)
            return raw + b"\n# synthetic dependency drift\n" if Path(path).name == "gradients.py" else raw
        monkeypatch.setattr(provenance, "read_bounded_bytes", changed)
    with pytest.raises(ValueError):
        invoke()
    assert not tuple(destination.iterdir())
    assert not tuple(Path(plane.root.contract.canonical_path).rglob("*.json"))


@pytest.mark.parametrize("route", ("generic", "fixed24", "history"))
def test_supported_runtime_reaches_only_mocked_producer(route, tmp_path, monkeypatch, plane):
    destination = Path(plane.root.contract.canonical_path) / "new-source"
    destination.mkdir()
    invoke = invocation(route, destination, plane, monkeypatch)
    with pytest.raises(ProducerReached):
        invoke()


def test_complete_producing_manifest_and_reader_environment_are_distinct(monkeypatch):
    record = provenance.capture_matrix_provenance(project_root=ROOT)
    names = {name for name, _ in record.package_sources}
    assert {
        "adapters/simulators/six_matrix_response/gradients.py",
        "adapters/simulators/six_matrix_response/model.py",
        "adapters/simulators/six_matrix_response/response_hessian.py",
        "adapters/methods/prepared_response/projection.py",
    } <= names
    assert record.code_sources_sha256 == provenance.numerical_package_code_sha256()
    import subprocess
    clean = not bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT))
    assert record.observed_checkout_clean == clean
    monkeypatch.setattr(np, "__version__", "reader-environment-is-not-producer")
    assert provenance.authenticate_recorded_matrix_provenance(record) == record


def test_native_completion_authenticates_provenance_without_reacquisition(tmp_path, monkeypatch, plane):
    from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source_records import (
        CurrentPreparationNativeCell, CurrentPreparationRootReport, current_preparation_cell_ids,
    )
    inputs = allocation(2)
    config = finite_operands.prepare_matrix_source_configuration(config_id="test.completed.source",
        allocation=inputs, project_root=ROOT)
    def exposed_mock_producer(*, root, **kwargs):
        cells = tuple(CurrentPreparationNativeCell(name, "UNENTERED", None, "SYNTHETIC_SOFTWARE_FIXTURE")
                      for name in current_preparation_cell_ids(root.root_id))
        report = CurrentPreparationRootReport(root.root_id, ObjectIdentity.from_record(root.root_id, root),
            config.identity, cells, "UNEVALUABLE", "SYNTHETIC_SOFTWARE_FIXTURE")
        return SimpleNamespace(report=report, arrays={"x": np.zeros((24, 2))})
    monkeypatch.setattr("empirical_lawhood.adapters.methods.finite_response_law.transient_bridge_source.acquire_current_preparation_root", exposed_mock_producer)
    directory = Path(plane.root.contract.canonical_path) / "completed"
    directory.mkdir()
    manifest = finite_operands.matrix_preparation_source_export(directory, config=config, allocation=inputs,
        project_root=ROOT, artifact_writer=plane)
    def read():
        return finite_operands.saved_matrix_arrays_import(directory / "arrays.canonical.json",
            arrays_path=directory / "arrays.npz", allocation_path=directory / "allocation.canonical.json",
            artifact_writer=plane)
    monkeypatch.setattr(np, "__version__", "reader-does-not-reproduce-old-runtime")
    assert read()[0] == manifest
    original = (directory / "numerical-provenance.canonical.json").read_bytes()
    (directory / "numerical-provenance.canonical.json").write_bytes(original.replace(b"3.11.14", b"3.11.99"))
    with pytest.raises(ValueError, match="completion"):
        read()


@pytest.mark.parametrize("route", ("generic", "fit", "readiness"))
@pytest.mark.parametrize("escape", ("traversal", "outside-symlink", "inside-symlink"))
def test_output_preflight_refuses_before_producer_or_writes(route, escape, tmp_path, monkeypatch, plane):
    from empirical_lawhood.api import preparation_readiness
    root = Path(plane.root.contract.canonical_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    if escape == "traversal":
        destination = root / ".." / "outside" / "new"
    else:
        target = outside if escape == "outside-symlink" else root / "real"
        target.mkdir(exist_ok=True)
        (root / "link").symlink_to(target, target_is_directory=True)
        destination = root / "link" / "new"
    if route != "readiness":
        destination.mkdir()
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    def forbidden(*args, **kwargs):
        raise AssertionError("numerical producer or input reader entered")
    if route == "generic":
        invoke = invocation(route, destination, plane, monkeypatch)
    elif route == "fit":
        monkeypatch.setattr(finite_operands, "saved_matrix_arrays_import", forbidden)
        invoke = lambda: finite_operands.preparation_operands_export(destination,
            native_manifest_path=tmp_path / "absent", native_arrays_path=tmp_path / "absent",
            allocation_path=tmp_path / "absent", original_f=None, operand_id="test.refused.fit",
            artifact_writer=plane, project_root=ROOT)
    else:
        monkeypatch.setattr(preparation_readiness, "saved_matrix_arrays_import", forbidden)
        monkeypatch.setattr(preparation_readiness, "current_readiness", forbidden)
        invoke = lambda: preparation_readiness.current_preparation_readiness(
            manifest_path=tmp_path / "absent", arrays_path=tmp_path / "absent",
            allocation_path=tmp_path / "absent", original_f=None, output_dir=destination,
            report_id="test.refused.readiness", artifact_writer=plane, project_root=ROOT)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        invoke()
    assert {path.relative_to(tmp_path) for path in tmp_path.rglob("*")} == before
