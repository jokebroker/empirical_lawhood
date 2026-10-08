# SPDX-License-Identifier: MPL-2.0
"""Synthetic release failures retain terminal, independently hashed evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import threading

import pytest


@pytest.fixture
def release_runner(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "scripts/release_check.py"
    spec = importlib.util.spec_from_file_location("release_runner_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    identity = {"commit": "synthetic", "tree_object": "synthetic-tree",
                "tree_inventory_sha256": "synthetic-inventory", "clean": True}
    monkeypatch.setattr(module, "source_identity", lambda root: identity)
    monkeypatch.setattr(module, "output", lambda command, **kwargs: "synthetic uv")
    return module


def invoke(runner, monkeypatch, packet, child):
    monkeypatch.setattr(runner.subprocess, "run", child)
    monkeypatch.setattr(runner.sys, "argv", ["release_check.py", "--profile", "native-open",
                                            "--output-dir", str(packet), "--offline"])
    runner.main()


def read_packet(packet):
    manifest_bytes = (packet / "release-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    status = json.loads((packet / "release-status.json").read_bytes())
    assert status["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert status["software_status"] == manifest["status"]
    assert status["checks"] == manifest["checks"]
    assert status["scientific_qualification"]["status"] == "NOT_PERFORMED"
    for check in manifest["checks"]:
        if check.get("log") is not None:
            payload = (packet / check["log"]).read_bytes()
            assert check["sha256"] == hashlib.sha256(payload).hexdigest()
    return manifest


@pytest.mark.parametrize("error", [RuntimeError("child defect"), SystemExit(7), KeyboardInterrupt()])
def test_catchable_failure_is_terminal_and_propagates(release_runner, monkeypatch, tmp_path, error):
    count = 0

    def child(command, **kwargs):
        nonlocal count
        count += 1
        kwargs["stdout"].write(f"attempt {count}\n")
        kwargs["stdout"].flush()
        if count == 2:
            raise error
        return SimpleNamespace(returncode=0)

    packet = tmp_path / "packet"
    with pytest.raises(type(error)) as raised:
        invoke(release_runner, monkeypatch, packet, child)
    assert raised.value is error
    manifest = read_packet(packet)
    assert manifest["status"] == "FAILED"
    assert manifest["failure_type"] == type(error).__name__
    assert len(manifest["checks"]) == 2
    completed, attempted = manifest["checks"]
    assert completed["exit_code"] == 0
    assert attempted["disposition"] == "INTERRUPTED"
    assert attempted["exit_code"] is None
    assert (packet / attempted["log"]).read_bytes() == b"attempt 2\n"


def test_nonzero_child_exit_retains_observed_exit(release_runner, monkeypatch, tmp_path):
    def child(command, **kwargs):
        kwargs["stdout"].write("reported child failure\n")
        return SimpleNamespace(returncode=9)

    packet = tmp_path / "packet"
    with pytest.raises(RuntimeError, match="exit 9"):
        invoke(release_runner, monkeypatch, packet, child)
    manifest = read_packet(packet)
    assert manifest["status"] == "FAILED"
    assert manifest["checks"][0]["exit_code"] == 9
    assert manifest["checks"][0]["disposition"] == "FAILED"


def test_final_identity_refusal_cannot_pass(release_runner, monkeypatch, tmp_path):
    count = 0
    original = release_runner.source_identity

    def identity(root):
        nonlocal count
        count += 1
        if count == 3:
            raise SystemExit("final source changed")
        return original(root)

    monkeypatch.setattr(release_runner, "source_identity", identity)

    def child(command, **kwargs):
        kwargs["stdout"].write("completed synthetic child\n")
        return SimpleNamespace(returncode=0)

    packet = tmp_path / "packet"
    with pytest.raises(SystemExit, match="final source changed"):
        invoke(release_runner, monkeypatch, packet, child)
    manifest = read_packet(packet)
    assert manifest["status"] == "FAILED"
    assert all(check["exit_code"] == 0 for check in manifest["checks"])


def test_interrupted_snapshot_survives_inherited_writer(release_runner, monkeypatch, tmp_path):
    allow_write = threading.Event()
    writer_done = threading.Event()
    thread = None

    def child(command, **kwargs):
        nonlocal thread
        stream = kwargs["stdout"]
        stream.write("observed prefix\n")
        stream.flush()
        descriptor = os.dup(stream.fileno())

        def writer():
            try:
                assert allow_write.wait(10)
                os.write(descriptor, b"later inherited output\n")
            finally:
                os.close(descriptor)
                writer_done.set()

        thread = threading.Thread(target=writer)
        thread.start()
        raise KeyboardInterrupt()

    packet = tmp_path / "packet"
    try:
        with pytest.raises(KeyboardInterrupt):
            invoke(release_runner, monkeypatch, packet, child)
        manifest = read_packet(packet)
        attempted = manifest["checks"][0]
        frozen = (packet / attempted["log"]).read_bytes()
        assert frozen == b"observed prefix\n"
        allow_write.set()
        assert writer_done.wait(10)
        assert (packet / attempted["log"]).read_bytes() == frozen
        assert (packet / attempted["unbound_live_log"]).read_bytes() == frozen + b"later inherited output\n"
        read_packet(packet)
    finally:
        allow_write.set()
        if thread is not None:
            thread.join(timeout=10)
            assert not thread.is_alive()


def test_initial_dirty_refusal_creates_no_packet(release_runner, monkeypatch, tmp_path):
    def dirty(root):
        raise SystemExit("dirty source")

    monkeypatch.setattr(release_runner, "source_identity", dirty)
    packet = tmp_path / "packet"
    with pytest.raises(SystemExit, match="dirty source"):
        invoke(release_runner, monkeypatch, packet, lambda *args, **kwargs: pytest.fail("unexpected child"))
    assert not packet.exists()


def test_uv_prerequisite_interrupt_creates_no_unfinalized_packet(release_runner, monkeypatch, tmp_path):
    def unavailable(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(release_runner, "output", unavailable)
    packet = tmp_path / "packet"
    with pytest.raises(KeyboardInterrupt):
        invoke(release_runner, monkeypatch, packet, lambda *args, **kwargs: pytest.fail("unexpected child"))
    assert not packet.exists()


@pytest.mark.parametrize(("script", "check_name"), (
    ("scripts/generate_integration_examples.py", "generated-integration_examples"),
    ("scripts/check_frozen_rc_inputs.py", "frozen-rc-numerical-inputs"),
))
def test_source_drift_refuses_before_portable_suite(release_runner, monkeypatch, tmp_path, script, check_name):
    packet = tmp_path / "packet"
    monkeypatch.setattr(release_runner.sys, "argv", ["release_check.py", "--profile", "portable",
                                                    "--output-dir", str(packet)])

    def child(command, **kwargs):
        assert "pytest" not in command, "source drift must refuse before the expensive suite"
        drifted = script in command
        kwargs["stdout"].write("source input drift\n" if drifted else "synthetic preflight passed\n")
        return SimpleNamespace(returncode=1 if drifted else 0)

    monkeypatch.setattr(release_runner.subprocess, "run", child)
    with pytest.raises(RuntimeError, match=f"{check_name} failed"):
        release_runner.main()
    manifest = read_packet(packet)
    assert manifest["status"] == "FAILED"
    failed = manifest["checks"][-1]
    assert failed["name"] == check_name
    assert failed["exit_code"] == 1
    assert (packet / failed["log"]).read_text() == "source input drift\n"
    assert not any(check["name"] == "tests" for check in manifest["checks"])


@pytest.mark.parametrize("stage", ["packet", "logs", "temporary-files"])
@pytest.mark.parametrize("error", [RuntimeError("setup defect"), SystemExit(7), KeyboardInterrupt()])
def test_setup_failure_finalizes_created_packet(release_runner, monkeypatch, tmp_path, stage, error):
    packet = tmp_path / "packet"
    environment_root = tmp_path / "environments"
    original_mkdir = Path.mkdir

    def mkdir(path, *args, **kwargs):
        result = original_mkdir(path, *args, **kwargs)
        if path.name == stage:
            raise error
        return result

    monkeypatch.setattr(Path, "mkdir", mkdir)
    monkeypatch.setattr(release_runner.sys, "argv", ["release_check.py", "--profile", "native-open",
                                                    "--output-dir", str(packet),
                                                    "--environment-root", str(environment_root)])
    monkeypatch.setattr(release_runner.subprocess, "run", lambda *args, **kwargs: pytest.fail("unexpected child"))
    with pytest.raises(type(error)) as raised:
        release_runner.main()
    assert raised.value is error
    manifest = read_packet(packet)
    assert manifest["status"] == "FAILED"
    assert manifest["failure_type"] == type(error).__name__
    assert manifest["checks"] == []


@pytest.mark.parametrize("exit_code", [0, 9])
def test_completed_snapshot_survives_inherited_writer(release_runner, monkeypatch, tmp_path, exit_code):
    allow_write = threading.Event()
    thread = None
    count = 0

    def child(command, **kwargs):
        nonlocal thread, count
        count += 1
        stream = kwargs["stdout"]
        stream.write("observed completion\n")
        stream.flush()
        if count == 1:
            descriptor = os.dup(stream.fileno())

            def writer():
                try:
                    assert allow_write.wait(10)
                    os.write(descriptor, b"late descendant output\n")
                finally:
                    os.close(descriptor)

            thread = threading.Thread(target=writer)
            thread.start()
            return SimpleNamespace(returncode=exit_code)
        return SimpleNamespace(returncode=0)

    packet = tmp_path / "packet"
    try:
        if exit_code:
            with pytest.raises(RuntimeError, match="exit 9"):
                invoke(release_runner, monkeypatch, packet, child)
        else:
            invoke(release_runner, monkeypatch, packet, child)
        manifest = read_packet(packet)
        assert manifest["status"] == ("FAILED" if exit_code else "PASSED")
        completed = manifest["checks"][0]
        assert completed["exit_code"] == exit_code
        frozen = (packet / completed["log"]).read_bytes()
        assert frozen == b"observed completion\n"
        allow_write.set()
        thread.join(timeout=10)
        assert not thread.is_alive()
        assert (packet / completed["unbound_live_log"]).read_bytes() == frozen + b"late descendant output\n"
        read_packet(packet)
    finally:
        allow_write.set()
        if thread is not None:
            thread.join(timeout=10)
            assert not thread.is_alive()


def test_project_environment_checks_real_origin_and_version(release_runner, monkeypatch, tmp_path):
    real_run = subprocess.run
    environment_command = None

    def child(command, **kwargs):
        nonlocal environment_command
        if command[:4] == ["uv", "run", "--no-sync", "python"] and command[4] == "-c":
            environment_command = command[5]
        kwargs["stdout"].write("synthetic release orchestration\n")
        return SimpleNamespace(returncode=0)

    packet = tmp_path / "packet"
    invoke(release_runner, monkeypatch, packet, child)
    assert read_packet(packet)["status"] == "PASSED"
    assert environment_command is not None
    root = Path(__file__).resolve().parents[1]
    good = real_run([sys.executable, "-c", environment_command], cwd=root,
                    capture_output=True, text=True, timeout=30)
    assert good.returncode == 0, good.stderr
    evidence = json.loads(good.stdout)
    assert Path(evidence["package_origin"]).resolve() == root / "src/empirical_lawhood/__init__.py"
    assert evidence["selected_version"] == "0.2.0"
    wrong_version = environment_command.replace("selected='0.2.0'", "selected='0.1.0'")
    assert wrong_version != environment_command
    bad = real_run([sys.executable, "-c", wrong_version], cwd=root,
                   capture_output=True, text=True, timeout=30)
    assert bad.returncode != 0
    assert "source/module/distribution version mismatch" in bad.stderr


@pytest.mark.parametrize("target", ["packet", "environment"])
@pytest.mark.parametrize("alias", ["symlink", "traversal"])
def test_release_destinations_refuse_aliases_before_child_effects(release_runner, monkeypatch, tmp_path, target, alias):
    if alias == "symlink":
        parent = tmp_path / "alias"
        parent.symlink_to(release_runner.ROOT, target_is_directory=True)
        selected = parent / "must-not-exist"
    else:
        selected = tmp_path / "never-created" / ".." / "must-not-exist"
    packet = selected if target == "packet" else tmp_path / "ordinary-packet"
    args = ["release_check.py", "--profile", "native-open", "--output-dir", str(packet)]
    if target == "environment":
        args += ["--environment-root", str(selected)]
    monkeypatch.setattr(release_runner.sys, "argv", args)
    monkeypatch.setattr(release_runner.subprocess, "run", lambda *a, **k: pytest.fail("invalid destination contacted a child"))
    with pytest.raises(SystemExit, match="outside"):
        release_runner.main()
    assert not packet.exists() and not (tmp_path / "never-created").exists()
