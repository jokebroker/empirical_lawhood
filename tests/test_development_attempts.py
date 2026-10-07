"""Operational retention preserves scientific invocations and failure boundaries.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from empirical_lawhood.cli import attempts
from empirical_lawhood.cli.app import app
from empirical_lawhood.cli.metadata import RETAINED_ATTEMPT_OPTIONS
from empirical_lawhood.infrastructure import development_attempts as storage


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "experiments/rc-ladder-response/model.json"
RUNNER = CliRunner()


def invoke(config, destination=None, *, before=(), after=(), **kwargs):
    arguments = [*before, "campaign", "rc-ladder-native-check", "--config", str(config)]
    if destination is not None:
        arguments.extend(["--output-dir", str(destination)])
    return RUNNER.invoke(app, [*arguments, *after], **kwargs)


def record(destination):
    return json.loads((destination / "invocation.json").read_text())


@pytest.fixture
def native(monkeypatch):
    from empirical_lawhood.adapters.simulators.rc_ladder_response import native_quickstart

    calls = []
    report = {"independent_units": 1, "receiver_unit": "V", "verdict": "UNEVALUABLE", "campaign_issued": False}

    def compute(path):
        calls.append(path.read_bytes())
        return report

    monkeypatch.setattr(native_quickstart, "check_native_model", compute)
    return native_quickstart, calls, report


def test_declared_roster_matches_actual_click_options_and_effect_metadata():
    participants = attempts.participation()
    assert {name: spec["option"] for name, spec in participants.items()} == RETAINED_ATTEMPT_OPTIONS
    click = get_command(app)
    for name, spec in participants.items():
        current = click
        for word in name.split():
            current = current.commands[word]
        assert any(spec["option"] in parameter.opts for parameter in current.params), name


def test_retained_runtime_identifies_declared_torax_and_grid_distribution_names(tmp_path, monkeypatch):
    installed = {"gymtorax": "1.1.1", "torax": "1.4.2", "grid2op": "1.12.5", "lightsim2grid": "0.13.1"}
    queried = []

    def installed_version(name):
        queried.append(name)
        if name not in installed:
            raise storage.PackageNotFoundError(name)
        return installed[name]

    monkeypatch.setattr(storage, "version", installed_version)
    monkeypatch.setattr(storage, "__file__", str(tmp_path / "site-packages/empirical_lawhood/infrastructure/development_attempts.py"))
    destination = tmp_path / "attempt"
    attempt = storage.DevelopmentAttempt(destination, "campaign grid2op-native-check", native=True)
    try:
        runtime = record(destination)["runtime"]
        versions = runtime["installed_distribution_versions"]
        assert {name: versions[name] for name in installed} == installed
        assert "gym-torax" not in queried
        assert versions["brian2"] is None
        assert runtime["qualification"] == "NOT_ESTABLISHED"
        assert record(destination)["native_contact"] == "unknown"
    finally:
        attempt.close()


def test_success_retains_exact_consumed_input_and_unchanged_report_once(tmp_path, native):
    _, calls, returned = native
    destination = tmp_path / "attempt"
    result = invoke(MODEL, destination)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == returned
    assert (destination / "report.json").read_text() == result.stdout
    assert calls == [MODEL.read_bytes()]
    invocation = record(destination)
    assert invocation["state"] == "SUCCEEDED"
    assert invocation["exit_code"] == 0
    assert invocation["operation_completed"] is True
    assert invocation["scientific_adjudication"] == "NOT_PERFORMED"
    assert invocation["native_contact"] == "operation_reported_complete"
    assert len(invocation["inputs"]) == 1
    selected = invocation["inputs"][0]
    assert selected["sha256"] == hashlib.sha256(calls[0]).hexdigest()
    assert selected["bytes"] == len(calls[0])
    assert selected["used_snapshot"] is True
    assert selected["accepted"] is True
    assert (destination / "input.json").read_bytes() == calls[0]
    assert not (destination / "failure.json").exists()
    assert "retained at" in result.stderr
    assert attempts._current.get() is None


def test_retention_needs_no_hard_links(tmp_path, native, monkeypatch):
    def unsupported(*args, **kwargs):
        raise PermissionError("FAT does not support hard links")

    monkeypatch.setattr(storage.os, "link", unsupported)
    destination = tmp_path / "fat-attempt"
    result = invoke(MODEL, destination)
    assert result.exit_code == 0, result.output
    assert record(destination)["state"] == "SUCCEEDED"
    assert json.loads((destination / "report.json").read_text()) == native[2]
    assert len(native[1]) == 1
    assert not list(destination.glob(".write-*"))


@pytest.mark.parametrize("symlink", [False, True])
def test_atomic_attempt_publication_refuses_a_racing_writer(tmp_path, monkeypatch, symlink):
    attempt = storage.DevelopmentAttempt(tmp_path / "attempt", "controlled")
    protected = tmp_path / "protected"
    protected.write_bytes(b"other writer's bytes")
    original = storage.rename_noreplace

    def raced(source, destination, **kwargs):
        target = attempt.path / destination
        if symlink:
            target.symlink_to(protected)
        else:
            target.write_bytes(protected.read_bytes())
        return original(source, destination, **kwargs)

    monkeypatch.setattr(storage, "rename_noreplace", raced)
    try:
        with pytest.raises(FileExistsError):
            attempt._write("report.json", b"our report", 1024)
        assert (attempt.path / "report.json").read_bytes() == protected.read_bytes()
        assert protected.read_bytes() == b"other writer's bytes"
        assert not list(attempt.path.glob(".write-*"))
    finally:
        attempt.close()


def test_multiple_consumed_inputs_keep_distinct_owned_snapshots(tmp_path):
    first, second = tmp_path / 'first.json', tmp_path / 'second.json'
    first.write_bytes(b'{"first":1}\n')
    second.write_bytes(b'{"second":2}\n')
    attempt = storage.DevelopmentAttempt(tmp_path / 'attempt', 'test.multi-input')
    try:
        first_copy = attempt.capture_input('config', first, maximum_bytes=1024, copy=True)
        second_copy = attempt.capture_input('config', second, maximum_bytes=1024, copy=True)
        assert first_copy != second_copy
        assert first_copy.read_bytes() == first.read_bytes()
        assert second_copy.read_bytes() == second.read_bytes()
        assert attempt.capture_input('config', first, maximum_bytes=1024, copy=True) == first_copy
        rows = record(attempt.path)['inputs']
        assert len(rows) == 2
        for row in rows:
            assert hashlib.sha256((attempt.path / row['retained_file']).read_bytes()).hexdigest() == row['sha256']
    finally:
        attempt.close()


def test_no_export_preserves_stdout_stderr_and_no_files(tmp_path, native, monkeypatch):
    _, calls, returned = native
    monkeypatch.chdir(tmp_path)
    result = invoke(MODEL)
    assert result.exit_code == 0
    assert result.stdout == json.dumps(returned, sort_keys=True, indent=2) + "\n"
    assert result.stderr == ""
    assert calls == [MODEL.read_bytes()]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("arguments", [
    ["campaign", "rc-ladder-native-check"],
    ["campaign", "rc-ladder-native-check", "--config", "/missing/config.json"],
    ["campaign", "rc-ladder-native-check", "unexpected", "--config", str(MODEL)],
    ["--project-root", "/missing/checkout", "campaign", "rc-ladder-native-check", "--config", str(MODEL)],
    ["campaign", "rc-ladder-native-check", "--unknown", "--config", str(MODEL)],
])
def test_early_click_errors_retain_usable_destination_without_contact(tmp_path, native, arguments):
    _, calls, _ = native
    destination = tmp_path / "early"
    result = RUNNER.invoke(app, [*arguments, "--output-dir", str(destination)])
    assert result.exit_code == 2, result.output
    assert calls == []
    invocation = record(destination)
    assert invocation["state"] == "FAILED"
    assert invocation["exit_code"] == 2
    assert invocation["operation_completed"] is False
    assert (destination / "failure.json").is_file()
    assert not (destination / "report.json").exists()
    assert attempts._current.get() is None


def test_invalid_config_retains_hash_but_not_arbitrary_private_payload(tmp_path, native):
    _, calls, _ = native
    document = json.loads(MODEL.read_text())
    document["value"]["unknown"] = "private-sealed-payload-do-not-copy"
    config = tmp_path / "bad.json"
    config.write_text(json.dumps(document))
    destination = tmp_path / "invalid"
    result = invoke(config, destination)
    assert result.exit_code == 2
    assert calls == []
    invocation = record(destination)
    assert invocation["inputs"][0]["sha256"] == hashlib.sha256(config.read_bytes()).hexdigest()
    assert len(invocation["inputs"]) == 1
    assert not (destination / "input.json").exists()
    assert invocation["native_contact"] == "none"
    assert not (destination / "report.json").exists()
    assert "private-sealed-payload" not in "".join(path.read_text() for path in destination.iterdir())


def test_pretty_value_valid_input_keeps_real_exact_byte_refusal(tmp_path):
    config = tmp_path / "pretty.json"
    config.write_text(json.dumps(json.loads(MODEL.read_text()), indent=2))
    destination = tmp_path / "pretty-attempt"
    result = invoke(config, destination)
    assert result.exit_code == 2
    assert "not canonical" in result.stderr
    assert (destination / "input.json").read_bytes() == config.read_bytes()
    assert record(destination)["input_validation"]["consumer_byte_ready"] is False
    assert not (destination / "report.json").exists()


def test_caller_mutation_after_snapshot_cannot_change_consumed_config(tmp_path, native, monkeypatch):
    module, calls, returned = native
    config = tmp_path / "caller.json"
    config.write_bytes(MODEL.read_bytes())
    original = config.read_bytes()

    def compute(path):
        config.write_bytes(b"caller changed after validation")
        calls.append(path.read_bytes())
        return returned

    monkeypatch.setattr(module, "check_native_model", compute)
    destination = tmp_path / "stable"
    result = invoke(config, destination)
    assert result.exit_code == 0
    assert calls == [original]
    assert (destination / "input.json").read_bytes() == original
    assert config.read_bytes() != original


@pytest.mark.parametrize("kind, expected", [(ValueError, 2), (RuntimeError, 1), (KeyboardInterrupt, 130)])
def test_handled_unexpected_and_interrupted_worker_preserve_failure(tmp_path, native, monkeypatch, kind, expected):
    module, calls, _ = native

    def fail(path):
        calls.append(path)
        raise kind("secret-exception-locals-never-retain")

    monkeypatch.setattr(module, "check_native_model", fail)
    destination = tmp_path / "worker"
    result = invoke(MODEL, destination)
    assert result.exit_code == expected, result.output
    assert len(calls) == 1
    invocation = record(destination)
    assert invocation["exit_code"] == expected
    assert invocation["state"] == "FAILED"
    assert invocation["native_contact"] == "unknown"
    assert kind.__name__ in str(invocation["failure"])
    assert "secret-exception" not in "".join(path.read_text() for path in destination.iterdir())
    assert not (destination / "report.json").exists()
    assert attempts._current.get() is None


@pytest.mark.parametrize("name, kind, expected", [("report.json", PermissionError, 1), ("report.md", PermissionError, 1), ("report.json", KeyboardInterrupt, 130)])
def test_export_failure_after_computation_does_not_rerun_or_hide_completed_work(tmp_path, native, monkeypatch, name, kind, expected):
    _, calls, returned = native
    original = storage.DevelopmentAttempt._write

    def write(attempt, member, payload, maximum):
        if member == name:
            raise kind("secondary private error")
        return original(attempt, member, payload, maximum)

    monkeypatch.setattr(storage.DevelopmentAttempt, "_write", write)
    destination = tmp_path / "export"
    result = invoke(MODEL, destination)
    assert result.exit_code == expected, result.output
    assert json.loads(result.stdout) == returned
    assert len(calls) == 1
    invocation = record(destination)
    assert invocation["state"] == "EXPORT_FAILED"
    assert invocation["operation_completed"] is True
    assert invocation["exit_code"] == expected
    assert kind.__name__ in str(invocation["export_failure"])
    assert "Work was not rerun" in result.stderr
    assert not list(destination.glob(".write-*"))


def test_secondary_failure_keeps_original_classification_and_exit(tmp_path, native, monkeypatch):
    module, calls, _ = native

    def fail(path):
        calls.append(path)
        raise ValueError("original refusal")

    monkeypatch.setattr(module, "check_native_model", fail)
    original = storage.DevelopmentAttempt._write

    def write(attempt, member, payload, maximum):
        if member == "failure.json":
            raise PermissionError("cannot retain failure sidecar")
        return original(attempt, member, payload, maximum)

    monkeypatch.setattr(storage.DevelopmentAttempt, "_write", write)
    destination = tmp_path / "secondary"
    result = invoke(MODEL, destination)
    assert result.exit_code == 2
    invocation = record(destination)
    assert "ValueError" in str(invocation["failure"])
    assert "PermissionError" in str(invocation["export_failure"])
    assert invocation["exit_code"] == 2
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["existing-empty", "existing-file", "symlink", "missing-parent"])
def test_unusable_destination_never_overwrites_or_runs(tmp_path, native, kind):
    _, calls, _ = native
    destination = tmp_path / "bad-destination"
    before = None
    if kind == "existing-empty":
        destination.mkdir()
    elif kind == "existing-file":
        before = b"protected user output"
        destination.write_bytes(before)
    elif kind == "symlink":
        protected = tmp_path / "protected"
        protected.mkdir()
        destination.symlink_to(protected, target_is_directory=True)
    else:
        destination = destination / "absent-parent" / "attempt"
    result = invoke(MODEL, destination)
    assert result.exit_code == 2
    assert calls == []
    assert "No durable development attempt" in result.stderr
    if before is not None:
        assert destination.read_bytes() == before
    if kind == "existing-empty":
        assert list(destination.iterdir()) == []


def test_start_record_write_failure_stops_before_contact(tmp_path, native, monkeypatch):
    _, calls, _ = native
    monkeypatch.setattr(storage.DevelopmentAttempt, "_write", lambda *args: (_ for _ in ()).throw(PermissionError()))
    result = invoke(MODEL, tmp_path / "failed-start")
    assert result.exit_code == 2
    assert calls == []
    assert "No durable development attempt" in result.stderr


def test_report_and_event_limits_are_bounded_and_not_success(tmp_path, native, monkeypatch):
    module, calls, _ = native

    def large(path):
        calls.append(path)
        return {"large": "x" * (storage.MAX_REPORT_BYTES + 1)}

    monkeypatch.setattr(module, "check_native_model", large)
    destination = tmp_path / "bounded-report"
    result = invoke(MODEL, destination)
    assert result.exit_code == 1
    assert len(calls) == 1
    assert record(destination)["state"] == "EXPORT_FAILED"
    assert not (destination / "report.json").exists()
    attempt = storage.DevelopmentAttempt(tmp_path / "events", "controlled")
    try:
        for _ in range(100):
            attempt.event("controlled")
        attempt.finish(0)
        assert (attempt.path / "diagnostics.jsonl").stat().st_size <= storage.MAX_DIAGNOSTICS_BYTES
        assert len((attempt.path / "diagnostics.jsonl").read_text().splitlines()) == storage.MAX_EVENTS
        assert record(attempt.path)["events_dropped"] > 0
    finally:
        attempt.close()


def test_help_remains_read_only_even_when_output_option_is_present(tmp_path, native):
    _, calls, _ = native
    destination = tmp_path / "help"
    result = invoke(MODEL, destination, after=["--help"])
    assert result.exit_code == 0
    assert not destination.exists()
    assert calls == []


def test_closed_authoring_early_failure_does_not_precreate_scientific_output(tmp_path):
    scientific, operational = tmp_path / "science", tmp_path / "operational"
    result = RUNNER.invoke(app, ["campaign", "circuit-author", "--study", str(ROOT / "experiments/rc-ladder-response/study.json"), "--experiment-id", "development-test", "--output-dir", str(scientific), "--attempt-dir", str(operational)])
    assert result.exit_code == 2
    assert not scientific.exists()
    assert record(operational)["state"] == "FAILED"
    assert not (operational / "input.json").exists()


def test_closed_destinations_cannot_be_nested_even_before_argument_validation(tmp_path):
    science = tmp_path / "science"
    result = RUNNER.invoke(app, ["campaign", "circuit-author", "--output-dir", str(science), "--attempt-dir", str(science / "logs")])
    assert result.exit_code == 2
    assert not science.exists()
    assert "No durable development attempt" in result.stderr


def test_nonstandalone_dispatch_preserves_refusal_code_and_cause(tmp_path, native, monkeypatch):
    module, calls, _ = native

    def fail(path):
        calls.append(path)
        raise ValueError("typed worker refusal")

    monkeypatch.setattr(module, "check_native_model", fail)
    destination = tmp_path / "library-dispatch"
    result = get_command(app).main(["campaign", "rc-ladder-native-check", "--config", str(MODEL), "--output-dir", str(destination)], standalone_mode=False)
    assert result == 2
    assert "ValueError" in str(record(destination)["failure"])
    assert len(calls) == 1
    assert attempts._current.get() is None


def test_identity_drift_after_validation_refuses_before_copy_or_contact(tmp_path, native, monkeypatch):
    from empirical_lawhood.api import configuration

    _, calls, _ = native
    path = tmp_path / "changing.json"
    path.write_bytes(MODEL.read_bytes())
    initial = path.read_bytes()
    validate = configuration.validate_configuration

    def change(selected, **kwargs):
        validated = validate(selected, **kwargs)
        selected.write_bytes(b"unvalidated-private-payload")
        return validated

    monkeypatch.setattr(configuration, "validate_configuration", change)
    destination = tmp_path / "drift"
    result = invoke(path, destination)
    assert result.exit_code == 2
    assert calls == []
    assert record(destination)["inputs"][0]["sha256"] == hashlib.sha256(initial).hexdigest()
    assert not (destination / "input.json").exists()


@pytest.mark.parametrize("kind", ["oversized", "symlink"])
def test_snapshot_input_read_boundaries_stop_before_contact(tmp_path, native, kind):
    _, calls, _ = native
    config = tmp_path / "input.json"
    if kind == "symlink":
        config.symlink_to(MODEL)
    else:
        config.write_bytes(MODEL.read_bytes() + b" " * (64 * 1024))
    destination = tmp_path / "input-boundary"
    result = invoke(config, destination)
    assert result.exit_code == 2
    assert calls == []
    assert record(destination)["native_contact"] == "none"
    assert not (destination / "input.json").exists()


def test_external_member_collision_is_never_overwritten(tmp_path, native, monkeypatch):
    module, calls, returned = native
    destination = tmp_path / "member-collision"
    protected = b"concurrent user file"

    def compute(path):
        calls.append(path)
        (destination / "report.json").write_bytes(protected)
        return returned

    monkeypatch.setattr(module, "check_native_model", compute)
    result = invoke(MODEL, destination)
    assert result.exit_code == 1
    assert len(calls) == 1
    assert (destination / "report.json").read_bytes() == protected
    invocation = record(destination)
    assert invocation["state"] == "EXPORT_FAILED"
    assert invocation["products"] == []


def test_unfinalized_start_is_explicitly_incomplete(tmp_path):
    attempt = storage.DevelopmentAttempt(tmp_path / "lost-process", "controlled")
    attempt.close()
    invocation = record(attempt.path)
    assert invocation["state"] == "RUNNING"
    assert invocation["exit_code"] is None
    assert invocation["ended_at_utc"] is None
    assert invocation["operation_completed"] is False
    assert not (attempt.path / "report.json").exists()


def test_matrix_operational_export_does_not_add_members_to_closed_products(tmp_path, monkeypatch):
    from empirical_lawhood.api import native_authoring

    science, operational = tmp_path / "science", tmp_path / "operational"
    members = {"authoring.json": b"original authoring bytes", "candidate.json": b"original candidate bytes"}
    calls = []

    def author(config, **kwargs):
        calls.append(config)
        directory = kwargs["output_dir"]
        directory.mkdir()
        for name, data in members.items():
            (directory / name).write_bytes(data)
        return {"authoring_dir": str(directory), "campaign_candidate_compiled": True, "campaign_issued": False}

    monkeypatch.setattr(native_authoring, "author_matrix_response", author)
    result = RUNNER.invoke(app, ["--project-root", str(ROOT), "campaign", "matrix-response-author", "--config", str(ROOT / "experiments/prepared-response/prepared-source-qualification-author.json"), "--output-dir", str(science), "--attempt-dir", str(operational)])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert {path.name: path.read_bytes() for path in science.iterdir()} == members
    assert record(operational)["inputs"][0]["used_snapshot"] is False
    assert not (operational / "input.json").exists()
    assert (operational / "report.json").read_text() == result.stdout


def test_demo_scientific_products_and_legacy_stdout_remain_separate(tmp_path, monkeypatch):
    from empirical_lawhood.examples import reactor_prefix

    science, operational = tmp_path / "demo", tmp_path / "logs"
    members = {"panel.json": b"typed original panel", "report.json": b"scientific original report", "report.md": b"scientific readable report"}
    calls = []

    def example(directory):
        calls.append(directory)
        directory.mkdir()
        for name, payload in members.items():
            (directory / name).write_bytes(payload)
        return {"label": "EXPOSED_TEST", "complete_branches": 2, "observed_deliveries": 4}

    monkeypatch.setattr(reactor_prefix, "run_reactor_prefix", example)
    result = RUNNER.invoke(app, ["example", "reactor-prefix", "--output-dir", str(science), "--attempt-dir", str(operational)])
    assert result.exit_code == 0, result.output
    assert result.stdout == f"EXPOSED_TEST: 2 branches, 4 deliveries; {science / 'report.json'}\n"
    assert {path.name: path.read_bytes() for path in science.iterdir()} == members
    assert calls == [science]
    assert record(operational)["state"] == "SUCCEEDED"


def test_broken_stdout_after_computation_preserves_returned_report(tmp_path, native, monkeypatch):
    import errno

    _, calls, returned = native
    original = attempts.typer.echo

    def echo(message=None, **kwargs):
        if not kwargs.get("err") and isinstance(message, str) and message.startswith("{"):
            raise BrokenPipeError(errno.EPIPE, "closed report consumer")
        return original(message, **kwargs)

    monkeypatch.setattr(attempts.typer, "echo", echo)
    destination = tmp_path / "broken-stdout"
    result = invoke(MODEL, destination)
    assert result.exit_code == 1
    assert len(calls) == 1
    invocation = record(destination)
    assert invocation["operation_completed"] is True
    assert invocation["state"] == "FAILED"
    assert json.loads((destination / "report.json").read_text()) == returned
    assert "BrokenPipeError" in str(invocation["failure"])


def test_nested_cli_invocations_restore_exact_context_and_do_not_log_default_calls(tmp_path, native):
    outer = storage.DevelopmentAttempt(tmp_path / "outer", "controlled")
    token = attempts._current.set(outer)
    initial = (outer.path / "invocation.json").read_bytes()
    try:
        result = invoke(MODEL)
        assert result.exit_code == 0
        assert attempts._current.get() is outer
        assert outer.record["operation_completed"] is False
        assert (outer.path / "invocation.json").read_bytes() == initial
        selected = tmp_path / "nested-early"
        early = RUNNER.invoke(app, ["campaign", "rc-ladder-native-check", "--output-dir", str(selected)])
        assert early.exit_code == 2
        assert record(selected)["state"] == "FAILED"
        assert attempts._current.get() is outer
        assert (outer.path / "invocation.json").read_bytes() == initial
    finally:
        attempts._current.reset(token)
        outer.close()


def test_report_serialization_failure_retains_known_completed_work(tmp_path, native, monkeypatch):
    module, calls, _ = native

    def compute(path):
        calls.append(path)
        return {"not_json": object()}

    monkeypatch.setattr(module, "check_native_model", compute)
    destination = tmp_path / "serialization"
    result = invoke(MODEL, destination)
    assert result.exit_code == 1
    assert len(calls) == 1
    invocation = record(destination)
    assert invocation["operation_completed"] is True
    assert invocation["failure"]["stage"] == "result_serialization"
    assert "TypeError" in str(invocation["failure"])
    assert not (destination / "report.json").exists()
