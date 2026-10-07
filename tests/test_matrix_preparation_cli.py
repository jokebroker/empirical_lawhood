"""Actual CLI/configuration/result routes with exposed saved software operands."""

import json

from typer.testing import CliRunner

from empirical_lawhood.cli.app import app
from tests.numerical_provenance_fixtures import PROJECT_ROOT, numerical_plane
from tests.test_integration_cli_workflow import _rc_cli_plane
from tests.test_matrix_preparation_analysis import operands  # noqa: F401
from tests.test_matrix_tangent_public import _software_source


def _invoke(*arguments):
    result = CliRunner().invoke(app, ["--project-root", str(PROJECT_ROOT), *arguments])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_public_copy_edit_prepare_transfer_baseline_result_journey(tmp_path, monkeypatch, operands):  # noqa: F811 - pytest resolves the imported fixture
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.records import BaselineSupportConfig, TransientAnalysisConfig
    native, bridge, original, _, _, _, _ = operands
    plane = numerical_plane(tmp_path)
    monkeypatch.setattr("empirical_lawhood.cli.integrations._storage", lambda: (PROJECT_ROOT, None, plane))
    store = tmp_path
    document = TransientAnalysisConfig("test.cli-transient").to_document()
    document["value"]["summary_population"] = "cir1"
    editable = tmp_path / "transient.editable.json"
    editable.write_text(json.dumps(document, indent=2))
    canonical = tmp_path / "transient.canonical.json"
    result = _invoke("config", "validate", "--config", str(editable), "--consumer", "matrix-transient-analysis", "--format", "json")
    assert not result["canonical_byte_ready"]
    _invoke("config", "prepare", "--config", str(editable), "--consumer", "matrix-transient-analysis", "--output-file", str(canonical), "--format", "json")
    transient = store / "transient"
    result = _invoke("campaign", "matrix-transient-analyze", "--input", str(canonical), "--native-dir", str(native),
                     "--bridge-dir", str(bridge), "--original-f-dir", str(original), "--output", str(transient), "--attempt-dir", str(tmp_path / "transfer-attempt"))
    assert result["status"] == "COMPLETE"
    baseline_config = tmp_path / "baseline.json"
    baseline_config.write_bytes(BaselineSupportConfig("test.cli-baseline", "nonrestoring.all").canonical_bytes())
    baseline = store / "baseline"
    result = _invoke("campaign", "matrix-baseline-analyze", "--input", str(baseline_config), "--transient-dir", str(transient),
                     "--original-f-dir", str(original), "--output", str(baseline), "--attempt-dir", str(tmp_path / "baseline-attempt"))
    assert result["status"] == "COMPLETE"
    result = _invoke("campaign", "matrix-analysis-result", "--directory", str(baseline), "--attempt-dir", str(tmp_path / "read-attempt"))
    assert result["status"] == "COMPLETE" and float(result["metrics"]["support.total_cells"]) == 216
    assert json.loads((tmp_path / "transfer-attempt" / "invocation.json").read_bytes())["state"] == "SUCCEEDED"


def test_public_full_tangent_configuration_source_evaluator_result(tmp_path, monkeypatch):
    _, store = _rc_cli_plane(tmp_path, monkeypatch)
    config = tmp_path / "tangent.json"
    allocated = _invoke("campaign", "matrix-tangent-config", "--config-id", "test.cli-tangent", "--namespace", "test.cli-roots",
                        "--master-seed", "90232", "--output", str(config), "--attempt-dir", str(tmp_path / "allocation-attempt"))
    assert allocated["status"] == "PREPARED"
    _invoke("config", "validate", "--config", str(config), "--consumer", "matrix-tangent", "--format", "json")
    monkeypatch.setattr("empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent.acquire_tangent_root", _software_source)
    source, analysis = store / "source", store / "analysis"
    result = _invoke("campaign", "matrix-tangent-source-export", "--input", str(config), "--output", str(source), "--attempt-dir", str(tmp_path / "source-attempt"))
    assert result["status"] == "EXPOSED_TANGENT_SOURCE"
    result = _invoke("campaign", "matrix-tangent-analyze", "--input", str(config), "--source-dir", str(source),
                     "--output", str(analysis), "--attempt-dir", str(tmp_path / "analysis-attempt"))
    assert result["status"] == "COMPLETE"
    result = _invoke("campaign", "matrix-analysis-result", "--directory", str(analysis), "--attempt-dir", str(tmp_path / "read-attempt"))
    assert result["status"] == "COMPLETE" and float(result["metrics"]["independent_source_roots"]) == 128
    assert float(result["metrics"]["nested_mechanism_comparisons"]) == 6400


def test_invalid_cli_destination_precedes_current_input_contact(tmp_path, monkeypatch):
    from empirical_lawhood.adapters.methods.matrix_preparation_analysis.records import TransientAnalysisConfig
    _, store = _rc_cli_plane(tmp_path, monkeypatch)
    config = tmp_path / "input.json"
    config.write_bytes(TransientAnalysisConfig("test.denied").canonical_bytes())
    parent = store / "file"
    parent.write_bytes(b"preserved")
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid output contacted inputs")
    monkeypatch.setattr("empirical_lawhood.cli.integrations._configuration", forbidden)
    monkeypatch.setattr("empirical_lawhood.api.matrix_preparation_analysis._original", forbidden)
    result = CliRunner().invoke(app, ["campaign", "matrix-transient-analyze", "--input", str(config),
                                     "--native-dir", str(tmp_path), "--bridge-dir", str(tmp_path), "--original-f-dir", str(tmp_path),
                                     "--output", str(parent / "output")])
    assert result.exit_code == 2 and "non-directory parent" in result.stderr
    assert parent.read_bytes() == b"preserved"
