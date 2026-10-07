"""Authoring failures preserve diagnosis, refusal exits and fresh-path recovery.

SPDX-License-Identifier: MPL-2.0
"""

import errno
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from empirical_lawhood.api.authoring_output import report_incomplete_output
from empirical_lawhood.cli import platform
from tests.authoring_wrapper_support import FAMILIES, MODULES, ROOT, export_wrapper


@pytest.mark.parametrize("family", FAMILIES)
def test_candidate_refusal_preserves_failed_inputs_and_fresh_path_recovery(tmp_path, monkeypatch, family):
    module = importlib.import_module(MODULES[family])
    original_api = module.create_authoring_api
    refusal = SimpleNamespace(succeeded=False, payload=None, reason_codes=("SYNTHETIC_CANDIDATE_REFUSAL",), errors=())
    monkeypatch.setattr(module, "create_authoring_api", lambda **kwargs: SimpleNamespace(compile_candidate=lambda request: refusal))
    selected = tmp_path / "refused"
    selected.mkdir()
    with pytest.raises(RuntimeError) as failure:
        export_wrapper(family, selected, monkeypatch)
    output = selected / "export"
    assert "SYNTHETIC_CANDIDATE_REFUSAL" in str(failure.value)
    if family == "matrix":
        assert not output.exists()
        assert "incomplete" not in str(failure.value)
    else:
        assert (output / "authoring.json").is_file()
        assert not (output / "candidate.json").exists()
        assert str(output) in str(failure.value)
        assert "candidate compilation" in str(failure.value)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with pytest.raises(FileExistsError) as conflict:
            export_wrapper(family, selected, monkeypatch)
        assert "incomplete" not in str(conflict.value)
        assert before == {path.name: path.read_bytes() for path in output.iterdir()}
    monkeypatch.setattr(module, "create_authoring_api", original_api)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert export_wrapper(family, fresh, monkeypatch)["summary"]["native_tasks_executed"] == 0


@pytest.mark.parametrize("family", FAMILIES)
def test_late_export_retains_exception_type_errno_cause_and_incomplete_products(tmp_path, monkeypatch, family):
    module = importlib.import_module(MODULES[family])
    output = tmp_path / "export"
    failure = PermissionError(errno.EACCES, "synthetic late export", str(output / "candidate.json"))
    cause = ValueError("original synthetic cause")
    failure.__cause__ = cause
    if family == "matrix":
        original = Path.write_bytes

        def save(path, payload):
            if path == output / "candidate.json":
                raise failure
            return original(path, payload)

        monkeypatch.setattr(Path, "write_bytes", save)
    else:
        original = module._save_new

        def save(directory, name, record):
            if name == "candidate.json":
                raise failure
            return original(directory, name, record)

        monkeypatch.setattr(module, "_save_new", save)
    with pytest.raises(PermissionError) as observed:
        export_wrapper(family, tmp_path, monkeypatch)
    assert observed.value is failure
    assert observed.value.errno == errno.EACCES
    assert observed.value.filename == str(output / "candidate.json")
    assert observed.value.__cause__ is cause
    assert "product export" in str(observed.value)
    assert str(output) in str(observed.value)
    assert (output / "authoring.json").is_file()
    assert not (output / "candidate.json").exists()


@pytest.mark.parametrize("family", FAMILIES)
def test_existing_empty_directory_policy_remains_family_specific(tmp_path, monkeypatch, family):
    output = tmp_path / "export"
    output.mkdir()
    if family == "matrix":
        with pytest.raises(FileExistsError, match="new directory") as failure:
            export_wrapper(family, tmp_path, monkeypatch)
        assert "incomplete" not in str(failure.value)
        assert list(output.iterdir()) == []
    else:
        assert export_wrapper(family, tmp_path, monkeypatch)["summary"]["native_tasks_executed"] == 0


def test_matrix_optional_output_preserves_temporary_compilation_only(tmp_path, monkeypatch):
    result = export_wrapper("matrix", tmp_path, monkeypatch, persist=False)
    assert result["summary"]["authoring_dir"] is None
    assert result["summary"]["native_tasks_executed"] == 0
    assert result["files"] == []
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize(("command", "entry", "family", "expected_exit"), (
    ("circuit-author", "author_fresh_rc", "rc", 5),
    ("electron-gas-author", "author_uniform_electron_gas_analytic_reference", "electron-gas", 5),
    ("synthetic-material-author", "author_synthetic_material_response", "material", 5),
    ("matrix-response-author", "author_matrix_response", "matrix", 3),
))
def test_actual_cli_handlers_render_incomplete_path_stage_and_original_refusal(tmp_path, monkeypatch, command, entry, family, expected_exit):
    selected = tmp_path / "selected"
    selected.mkdir()
    try:
        with report_incomplete_output(selected) as progress:
            progress.stage = "candidate compilation"
            raise RuntimeError("synthetic original refusal")
    except RuntimeError as error:
        failure = error

    def refuse(*args, **kwargs):
        raise failure

    class Profile:
        SCHEMA = "testing/operator-profile"

    profile = tmp_path / "profile.json"
    profile.write_text("{}")
    monkeypatch.setattr(platform, "CLI_PROJECT_ROOT", ROOT)
    monkeypatch.setattr(platform, "CLI_OPERATOR_PROFILE", profile)
    monkeypatch.setattr(platform, "OperatorStorageProfile", Profile)
    monkeypatch.setattr(platform, "load_registered_authoring", lambda *args, **kwargs: Profile())
    monkeypatch.setattr(platform, "resolve_external_root_contract", lambda *args, **kwargs: SimpleNamespace(canonical_path=str(tmp_path)))
    monkeypatch.setattr(platform, "GuardedExternalRoot", lambda contract: SimpleNamespace(verify=lambda **kwargs: None, resolve=lambda *args, **kwargs: selected))
    if family == "matrix":
        monkeypatch.setattr(importlib.import_module(MODULES[family]), entry, refuse)
        arguments = [command, "--config", str(ROOT / "experiments/prepared-response/prepared-source-qualification-author.json")]
    else:
        monkeypatch.setattr(platform, entry, refuse)
        loader = {"rc": "load_rc_study", "electron-gas": "load_uniform_electron_gas_analytic_config", "material": "load_synthetic_material_response_config"}[family]
        monkeypatch.setattr(platform, loader, lambda *args: None)
        arguments = [command, "--study" if family == "rc" else "--config", str(profile), "--experiment-id", "synthetic.cli-refusal", "--output-dir", str(selected)]
    result = CliRunner().invoke(platform.campaign_app, arguments)
    assert result.exit_code == expected_exit, result.output
    assert result.stdout == ""
    assert "synthetic original refusal" in result.stderr
    assert str(selected) in result.stderr
    assert "candidate compilation" in result.stderr
    assert list(selected.iterdir()) == []
