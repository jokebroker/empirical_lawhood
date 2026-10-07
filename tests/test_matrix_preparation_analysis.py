"""Exposed software fixtures for complete current preparation analyses.

No native campaign or scientific qualification is executed by these tests.
"""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import json

import numpy as np
import pytest

from tests.numerical_provenance_fixtures import PROJECT_ROOT, numerical_plane

from empirical_lawhood.api.finite_operands import original_f_operand_export, preparation_operands_export, saved_matrix_arrays_export, saved_matrix_arrays_import
from empirical_lawhood.api.matrix_preparation_analysis import (
    baseline_support_analysis, preparation_analysis_read, tangent_preparation_configuration,
    transient_response_analysis,
)
from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge import mechanical_kernel
from empirical_lawhood.adapters.methods.matrix_preparation_analysis.records import BaselineSupportConfig, TransientAnalysisConfig
from empirical_lawhood.adapters.methods.matrix_preparation_analysis.transient import baseline_support_analysis as evaluate_baseline, transient_analysis, verify_saved_forecasts
from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent_records import TangentPreparationConfig
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from tests.test_matrix_operand_transport import allocation


@pytest.fixture
def operands(tmp_path):
    native, bridge, original = (tmp_path / name for name in ("native", "bridge", "original"))
    for path in (native, bridge, original):
        path.mkdir()
    lower = original_f_operand_export(original, payload_id="test.analysis.original-f")
    current = allocation(24)
    current = replace(current, roots=tuple(replace(root, cohort="q2" if index < 8 else "cir1") for index, root in enumerate(current.roots)))
    rng = np.random.default_rng(20261007)
    center, scale = np.asarray(lower.center, dtype=np.float64), np.asarray(lower.scale, dtype=np.float64)
    prefix = center + 0.02 * rng.normal(size=(24, 24)) * scale
    hold = center + 0.02 * rng.normal(size=(24, 24)) * scale
    kernel, _ = mechanical_kernel()
    shift = np.einsum("stk,rkj->rstj", kernel, rng.normal(size=(24, 4, 24))) * scale
    baseline = prefix[:, None, None] + np.linspace(0, 1, 26)[None, None, :, None] * (hold - prefix)[:, None, None]
    features = np.repeat((baseline + shift)[:, :, None], 2, axis=2)
    handoff = features[:, :, :, -1].transpose(0, 1, 3, 2).copy()
    mean = lower.predict(handoff[..., 0].reshape(-1, 24)).mean.reshape(24, 9, 4, 8)
    truth = np.broadcast_to(mean[..., None, None], (24, 9, 4, 8, 2, 2)).copy()
    truth += rng.normal(size=truth.shape) * 0.0001
    arrays = {"x": np.repeat(prefix[..., None], 2, axis=-1), "z": handoff,
              "features": features, "y": truth, "work": np.zeros((24, 9, 2)),
              "positions": np.zeros((24, 9, 2, 26, 2, 3, 4, 4), dtype=np.complex128),
              "momenta": np.zeros((24, 9, 2, 26, 2, 3, 4, 4), dtype=np.complex128),
              "requested_ticks": np.tile(np.arange(4096, 4497, 16, dtype=np.int64), (24, 1)),
              "parent_complete": np.ones((24, 9, 2), dtype=np.bool_),
              "response_observed": np.ones((24, 9, 4, 8, 2, 2), dtype=np.bool_)}
    producer = ObjectIdentity("test.explicit-software-fixture", "empirical-lawhood/methods/finite-response-law/current-preparation-source-config", "1.0.0", sha256(b"software fixture only").hexdigest())
    saved_matrix_arrays_export(native, operand_id="test.analysis.native", allocation=current, arrays=arrays, producers=(producer,), original_f=lower.identity)
    preparation_operands_export(bridge, native_manifest_path=native / "arrays.canonical.json", native_arrays_path=native / "arrays.npz", allocation_path=native / "allocation.canonical.json", original_f=lower, operand_id="test.analysis.bridge", artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)
    _, _, derived = saved_matrix_arrays_import(bridge / "arrays.canonical.json", arrays_path=bridge / "arrays.npz", allocation_path=bridge / "allocation.canonical.json", artifact_writer=numerical_plane(tmp_path))
    return native, bridge, original, lower, current, arrays, derived


def test_public_complete_transfer_baseline_read_and_recovery(tmp_path, operands):
    native, bridge, original, _, _, _, _ = operands
    output, baseline = tmp_path / "transient-result", tmp_path / "baseline-result"
    report = transient_response_analysis(output, config=TransientAnalysisConfig("test.transient"), native_directory=native, bridge_directory=bridge, original_f_directory=original, artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)
    assert report.disposition == "COMPLETE" and report.qualification == "NONE"
    values = {metric.name: metric.value for metric in report.metrics}
    assert values["outer_fits"] == 32 and values["new_outer_fits"] == 28 and values["inner_fits"] == 24
    retained, roots, arrays = preparation_analysis_read(output, artifact_writer=numerical_plane(tmp_path))
    assert retained == report and len(roots.roots) == 24
    assert len(arrays) <= 256
    verify_saved_forecasts(arrays)
    assert arrays["radial.all.inner_masks"].shape == (4, 3, 3, 24)
    assert arrays["radial.all.response_q"].shape == (24,)
    assert np.isfinite(arrays["radial.all.inner_scores"]).all()
    _assert_report_only_mutations_refuse(output, retained, numerical_plane(tmp_path))
    result = baseline_support_analysis(baseline, config=BaselineSupportConfig("test.baseline"), transient_directory=output, original_f_directory=original, artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)
    assert result.qualification == "NONE" and result.disposition == "COMPLETE"
    metrics = {metric.name: float(metric.value) for metric in result.metrics if metric.value is not None}
    assert metrics["support.total_cells"] == 216
    assert sum(metrics[key] for key in ("support.true_supported", "support.true_unsupported", "support.false_supported", "support.false_unsupported")) == 216
    assert metrics["feature.total_mse"] == pytest.approx(metrics["feature.baseline_mse"] + metrics["feature.transient_mse"] + metrics["feature.cross_term"], abs=1e-12)
    assert metrics["response.total_mse"] == pytest.approx(sum(value for key, value in metrics.items() if key.startswith("response.") and key != "response.total_mse"), abs=1e-12)
    _, _, saved = preparation_analysis_read(baseline, artifact_writer=numerical_plane(tmp_path))
    _assert_report_only_mutations_refuse(baseline, result, numerical_plane(tmp_path))
    assert saved["known_failure_indices"].shape == (0, 2)
    (baseline / "arrays.npz").write_bytes((baseline / "arrays.npz").read_bytes() + b"altered")
    with pytest.raises((ValueError, RuntimeError), match="(artifact|committed|byte limit)"):
        preparation_analysis_read(baseline, artifact_writer=numerical_plane(tmp_path))
    with pytest.raises(FileExistsError):
        transient_response_analysis(output, config=TransientAnalysisConfig("test.another"), native_directory=native, bridge_directory=bridge, original_f_directory=original, artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)


def test_whole_root_future_poison_cannot_change_held_out_fit_or_width(operands):
    _, _, _, lower, _, _, derived = operands
    first = transient_analysis(derived, lower)
    poisoned = dict(derived)
    poisoned["y"] = np.array(derived["y"], copy=True)
    held = np.arange(24) % 4 == 0
    poisoned["y"][held] += 1e6
    other = transient_analysis(poisoned, lower)
    for model in ("radial", "nonrestoring"):
        key = model + ".all"
        np.testing.assert_array_equal(first[key + ".fits.operator"][0], other[key + ".fits.operator"][0])
        np.testing.assert_array_equal(first[key + ".forecast_delta"][held], other[key + ".forecast_delta"][held])
        np.testing.assert_array_equal(first[key + ".response_q"][held], other[key + ".response_q"][held])
        np.testing.assert_array_equal(first[key + ".inner_scores"][0], other[key + ".inner_scores"][0])


def test_saved_coefficients_and_inner_exclusion_are_falsifiable(operands):
    _, _, _, lower, _, _, derived = operands
    out = transient_analysis(derived, lower)
    altered = dict(out)
    altered["radial.duration_transfer.fits.operator"] = np.array(out["radial.duration_transfer.fits.operator"], copy=True)
    altered["radial.duration_transfer.fits.operator"][0, -1, 0] += 1
    with pytest.raises(ValueError, match="coefficient"):
        verify_saved_forecasts(altered)


def test_support_is_inclusive_and_failure_count_is_a_result(operands):
    _, _, _, lower, _, _, derived = operands
    out = transient_analysis(derived, lower)
    center, scale = np.asarray(lower.center, dtype=np.float64), np.asarray(lower.scale, dtype=np.float64)
    truth = np.tile(center, (24, 9, 1))
    truth[:, :, 23] += scale[23] * 6
    # Exact equality is inside, including the last (24th) feature.
    changed = dict(out)
    changed["z"] = np.repeat(truth[..., None], 2, axis=-1)
    changed["truth_delta"] = np.zeros((24, 9, 24), dtype=np.float64)
    changed["actual_handoff_f.mean"] = lower.predict(truth.reshape(-1, 24)).mean.reshape(24, 9, 4, 8)
    saved, metrics = evaluate_baseline(changed, lower)
    assert saved["actual_support"].all()
    assert metrics["support.total_cells"] == 216
    truth[:, :, 23] = np.nextafter(truth[:, :, 23], np.inf)
    changed["z"] = np.repeat(truth[..., None], 2, axis=-1)
    changed["actual_handoff_f.mean"] = lower.predict(truth.reshape(-1, 24)).mean.reshape(24, 9, 4, 8)
    saved, metrics = evaluate_baseline(changed, lower)
    assert not saved["actual_support"].any()
    assert saved["known_failure_indices"].shape == (216, 2)
    assert metrics["support.failure_roots"] == 24
    altered = dict(out)
    altered["radial.all.inner_masks"] = np.array(out["radial.all.inner_masks"], copy=True)
    altered["radial.all.inner_masks"][0, 0, 0, 0] = True
    with pytest.raises(ValueError, match="disjoint"):
        verify_saved_forecasts(altered)


def test_unevaluable_native_census_retains_all_roots(tmp_path, operands):
    native, bridge, original, lower, current, arrays, _ = operands
    broken = tmp_path / "missing-native"
    broken.mkdir()
    changed = dict(arrays)
    changed["response_observed"] = arrays["response_observed"].copy()
    changed["response_observed"][7, 2, 0, 0, 1, 1] = False
    saved_matrix_arrays_export(broken, operand_id="test.missing-native", allocation=current, arrays=changed,
                               producers=(ObjectIdentity("test.source", TransientAnalysisConfig.SCHEMA, "1.0.0", "c" * 64),), original_f=lower.identity)
    broken_bridge = tmp_path / "missing-bridge"
    broken_bridge.mkdir()
    preparation_operands_export(broken_bridge, native_manifest_path=broken / "arrays.canonical.json", native_arrays_path=broken / "arrays.npz", allocation_path=broken / "allocation.canonical.json", original_f=lower, operand_id="test.missing-bridge", artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)
    result = transient_response_analysis(tmp_path / "missing-result", config=TransientAnalysisConfig("test.missing"), native_directory=broken, bridge_directory=broken_bridge, original_f_directory=original, artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)
    assert result.disposition == "UNEVALUABLE" and result.incomplete_roots == (current.roots[7].root_id,)
    assert len(result.root_ids) == 24
    assert {metric.name: metric.value for metric in result.metrics}["outer_fits"] == 0
    _assert_report_only_mutations_refuse(tmp_path / "missing-result", result, numerical_plane(tmp_path))


def _assert_report_only_mutations_refuse(directory, report, plane):
    path = directory / "analysis-report.canonical.json"
    original = path.read_bytes()
    metric = report.metrics[0]
    changed = replace(report, metrics=(replace(metric, value=Decimal(-999)), *report.metrics[1:]))
    documents = [changed.to_document()]
    for field, value in (("implementation_sources_sha256", "f" * 64), ("disposition", "COMPLETE"), ("incomplete_roots", [])):
        document = json.loads(original)
        document["value"][field] = value
        if _document_bytes(document) != original:
            documents.append(document)
    wrong_schema = json.loads(original)
    wrong_schema["schema"] = "empirical-lawhood/review/foreign-report"
    documents.append(wrong_schema)
    wrong_role = json.loads(original)
    wrong_role["value"]["metrics"][0]["value"]["name"] = "foreign.metric"
    documents.append(wrong_role)
    wrong_runtime = json.loads(original)
    wrong_runtime["value"]["provenance"]["value"]["runtime_observation_json"] = "{}"
    documents.append(wrong_runtime)
    try:
        for document in documents:
            path.write_bytes(_document_bytes(document))
            with pytest.raises((ValueError, RuntimeError), match="(committed|byte limit)"):
                preparation_analysis_read(directory, artifact_writer=plane)
    finally:
        path.write_bytes(original)
    assert preparation_analysis_read(directory, artifact_writer=plane)[0] == report


def _document_bytes(document):
    return json.dumps(document, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def test_missing_input_manifest_or_completion_refuses_without_fit(tmp_path, operands, monkeypatch):
    native, bridge, original, _, _, _, _ = operands
    directory = tmp_path / "retained"
    plane = numerical_plane(tmp_path)
    result = transient_response_analysis(directory, config=TransientAnalysisConfig("test.custody"), native_directory=native,
        bridge_directory=bridge, original_f_directory=original, artifact_writer=plane, project_root=PROJECT_ROOT)
    def forbidden(*args, **kwargs):
        raise AssertionError("retained read repeated numerical/native work")
    monkeypatch.setattr("empirical_lawhood.adapters.methods.matrix_preparation_analysis.transient.transient_analysis", forbidden)
    monkeypatch.setattr("empirical_lawhood.api.matrix_preparation_analysis.capture_matrix_provenance", forbidden)
    assert preparation_analysis_read(directory, artifact_writer=numerical_plane(tmp_path))[0] == result
    for name in ("analysis-inputs.canonical.json", "analysis-completion.canonical.json.manifest.json"):
        path = directory / name
        raw = path.read_bytes()
        path.unlink()
        with pytest.raises((OSError, ValueError, RuntimeError)):
            preparation_analysis_read(directory, artifact_writer=plane)
        path.write_bytes(raw)
    path = directory / "analysis-inputs.canonical.json"
    original = path.read_bytes()
    path.write_bytes(original.replace(b"native-arrays", b"foreign-array"))
    with pytest.raises((ValueError, RuntimeError)):
        preparation_analysis_read(directory, artifact_writer=plane)
    path.write_bytes(original)


def test_independently_custodied_wrong_roles_state_schema_census_and_environment_refuse(tmp_path, operands):
    """Generic byte custody alone cannot satisfy the scientific reader contract."""
    from empirical_lawhood.infrastructure.retained_analysis import publish_retained_analysis, read_retained_analysis
    from empirical_lawhood.runtime.retained_analysis import RetainedAnalysisMember

    native, bridge, original, _, _, _, _ = operands
    plane = numerical_plane(tmp_path)
    directory = tmp_path / "honest"
    report = transient_response_analysis(directory, config=TransientAnalysisConfig("test.closed"), native_directory=native,
        bridge_directory=bridge, original_f_directory=original, artifact_writer=plane, project_root=PROJECT_ROOT)
    completion = read_retained_analysis(directory=directory, artifact_writer=plane)
    for mutation in ("metric-role", "missing-state", "input-role", "schema", "member-census", "environment"):
        selected = tmp_path / mutation
        selected.mkdir()
        documents = {member.relative_path: (directory / member.relative_path).read_bytes() for member in completion.members}
        if mutation in ("metric-role", "missing-state", "environment"):
            value = json.loads(documents["analysis-report.canonical.json"])
            if mutation == "metric-role":
                value["value"]["metrics"][0]["value"]["name"] = "foreign.metric"
            elif mutation == "missing-state":
                changed = replace(report, disposition="UNEVALUABLE", incomplete_roots=(report.root_ids[0],),
                    metrics=tuple(replace(report.metrics[0], name=name, value=Decimal(number)) for name, number in
                                  (("complete_roots", 23), ("independent_roots", 24), ("inner_fits", 0), ("outer_fits", 0))))
                value = changed.to_document()
            else:
                runtime = json.loads(value["value"]["provenance"]["value"]["runtime_observation_json"])
                runtime["numpy"] = "0.0.0-review-drift"
                value["value"]["provenance"]["value"]["runtime_observation_json"] = json.dumps(runtime, sort_keys=True, separators=(",", ":"))
            documents["analysis-report.canonical.json"] = _document_bytes(value)
        if mutation == "input-role":
            value = json.loads(documents["analysis-inputs.canonical.json"])
            value["value"]["roles"][0] = "foreign-role"
            documents["analysis-inputs.canonical.json"] = _document_bytes(value)
        members = []
        for member in completion.members:
            if mutation == "member-census" and member.relative_path == "allocation.canonical.json":
                continue
            raw = documents[member.relative_path]
            (selected / member.relative_path).write_bytes(raw)
            artifact = replace(member.artifact, sha256=sha256(raw).hexdigest(), size_bytes=len(raw))
            if mutation == "schema" and member.relative_path == "analysis-report.canonical.json":
                artifact = replace(artifact, payload_schema="empirical-lawhood/test/foreign-report")
            members.append(RetainedAnalysisMember(member.relative_path, artifact))
        publish_retained_analysis(directory=selected, artifact_writer=plane, completion_id=completion.completion_id, members=tuple(members))
        with pytest.raises((ValueError, RuntimeError)):
            preparation_analysis_read(selected, artifact_writer=plane)


def test_console_actual_transient_baseline_and_authenticated_readout(tmp_path, operands, monkeypatch):
    from typer.testing import CliRunner
    from empirical_lawhood.cli.app import app

    native, bridge, original, _, _, _, _ = operands
    plane = numerical_plane(tmp_path)
    monkeypatch.setattr("empirical_lawhood.cli.integrations._storage", lambda: (PROJECT_ROOT, None, plane))
    # The dirty test checkout has new files not yet staged. Numerical provenance
    # still checks every actual loaded origin and all producing bytes itself.
    monkeypatch.setattr("empirical_lawhood.cli.paper_analysis_integrations._project", lambda: PROJECT_ROOT)
    runner = CliRunner()
    outputs = {}
    for kind, record in (("transient", TransientAnalysisConfig("test.console.transient")), ("baseline", BaselineSupportConfig("test.console.baseline"))):
        config_path = tmp_path / (kind + ".json")
        config_path.write_bytes(record.canonical_bytes())
        output = tmp_path / (kind + "-result")
        outputs[kind] = output
        args = ["campaign", "matrix-" + kind + "-analyze", "--input", str(config_path), "--original-f-dir", str(original), "--output", str(output)]
        if kind == "transient":
            args += ["--native-dir", str(native), "--bridge-dir", str(bridge)]
        else:
            args += ["--transient-dir", str(outputs["transient"])]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, (result.output, result.exception)
        assert json.loads(result.stdout)["status"] == "COMPLETE"
        assert (output / "analysis-completion.canonical.json.manifest.json").is_file()
    def forbidden(*args, **kwargs):
        raise AssertionError("console readout recalculated producing science")
    monkeypatch.setattr("empirical_lawhood.api.matrix_preparation_analysis.capture_matrix_provenance", forbidden)
    monkeypatch.setattr("empirical_lawhood.adapters.methods.matrix_preparation_analysis.transient.transient_analysis", forbidden)
    for output in outputs.values():
        result = runner.invoke(app, ["campaign", "matrix-analysis-result", "--directory", str(output)])
        assert result.exit_code == 0, (result.output, result.exception)
        assert json.loads(result.stdout)["status"] == "COMPLETE"
    changed = outputs["transient"] / "analysis-report.canonical.json"
    value = json.loads(changed.read_bytes())
    value["value"]["metrics"][0]["value"]["value"] = {"decimal": "-999"}
    changed.write_bytes(_document_bytes(value))
    refused = runner.invoke(app, ["campaign", "matrix-analysis-result", "--directory", str(outputs["transient"])])
    assert refused.exit_code == 2 and not refused.stdout


@pytest.mark.parametrize("operation", ("source-export", "analyze"))
def test_console_tangent_forwards_exact_plane_and_project_before_native_contact(tmp_path, monkeypatch, operation):
    from typer.testing import CliRunner
    from empirical_lawhood.cli.app import app
    config = tangent_preparation_configuration(config_id="test.console.forward", namespace="test.console.forward", master_seed=97231, project_root=PROJECT_ROOT)
    path = tmp_path / "tangent.json"
    path.write_bytes(config.canonical_bytes())
    plane = numerical_plane(tmp_path)
    monkeypatch.setattr("empirical_lawhood.cli.integrations._storage", lambda: (PROJECT_ROOT, None, plane))
    monkeypatch.setattr("empirical_lawhood.cli.paper_analysis_integrations._project", lambda: PROJECT_ROOT)
    calls = []
    def selected(*args, **kwargs):
        assert kwargs["artifact_writer"] is plane and kwargs["project_root"] == PROJECT_ROOT
        assert kwargs["config"] == config
        calls.append((args, kwargs))
        raise ValueError("explicit pre-native forward sentinel")
    function = "tangent_preparation_source_export" if operation == "source-export" else "tangent_response_analysis"
    monkeypatch.setattr("empirical_lawhood.api.matrix_preparation_analysis." + function, selected)
    args = ["campaign", "matrix-tangent-" + operation, "--input", str(path), "--output", str(tmp_path / "result")]
    if operation == "analyze":
        source = tmp_path / "source"
        source.mkdir()
        args += ["--source-dir", str(source)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2 and len(calls) == 1 and "forward sentinel" in result.output
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("bad", ("existing", "traversal", "symlink", "file-parent"))
def test_invalid_output_refused_before_original_input_contact(tmp_path, monkeypatch, bad):
    def contact(*args, **kwargs):
        raise AssertionError("input contacted")
    monkeypatch.setattr("empirical_lawhood.api.matrix_preparation_analysis._original", contact)
    if bad == "existing":
        output = tmp_path
    elif bad == "traversal":
        output = tmp_path / "a" / ".." / "output"
    elif bad == "symlink":
        (tmp_path / "link").symlink_to(tmp_path, target_is_directory=True)
        output = tmp_path / "link" / "output"
    else:
        (tmp_path / "file").write_text("file")
        output = tmp_path / "file" / "output"
    with pytest.raises((ValueError, FileExistsError, NotADirectoryError)):
        transient_response_analysis(output, config=TransientAnalysisConfig("test.invalid"), native_directory=tmp_path, bridge_directory=tmp_path, original_f_directory=tmp_path, artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)


def test_tangent_allocation_numeric_inputs_and_preoutcome_ranks():
    config = tangent_preparation_configuration(config_id="test.tangent", namespace="test.current", master_seed=2026100709, project_root=PROJECT_ROOT)
    renamed = tangent_preparation_configuration(config_id="test.renamed", namespace="test.another-label", master_seed=2026100709, project_root=PROJECT_ROOT)
    changed = tangent_preparation_configuration(config_id="test.tangent", namespace="test.current", master_seed=2026100710, project_root=PROJECT_ROOT)
    assert len(config.roots) == 128
    assert sum(root.numerical_semantics for root in config.roots) == 32
    for context in ("assembling", "prepared"):
        roots = [root for root in config.roots if root.context == context]
        assert [sum(root.development_role == role for root in roots) for role in ("fit", "interval", "screen")] == [32, 16, 16]
    assert config.seed_root_sha256 == renamed.seed_root_sha256
    assert config.seed_root_sha256 != changed.seed_root_sha256
    root = config.roots[1]
    alias = root.prefix_seed_sha256[0][:32] + "0" * 32
    replacement = replace(root, prefix_seed_sha256=(config.roots[0].prefix_seed_sha256[0][:32] + alias[32:], *root.prefix_seed_sha256[1:]))
    with pytest.raises(ValueError, match="effective"):
        replace(config, roots=(config.roots[0], replacement, *config.roots[2:]))
    assert decode_canonical_bytes(config.canonical_bytes(), TangentPreparationConfig, maximum_bytes=1024**2) == config
