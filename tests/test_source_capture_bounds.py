# SPDX-License-Identifier: MPL-2.0
"""Bounded Git capture preserves exact bytes and managed process-group cleanup."""

from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from empirical_lawhood.infrastructure import bounded_process, source_closure
from tests.test_source_origin import build_source_repo


@pytest.mark.parametrize("size", (8, 9))
def test_stdout_ceiling_is_enforced_while_capturing(size):
    command = [sys.executable, "-c", f"import os; os.write(1,b'x'*{size})"]
    if size == 8:
        result = bounded_process.run_bounded_command(command, maximum_stdout_bytes=8)
        assert result.stdout == b"x" * 8
    else:
        with pytest.raises(bounded_process.BoundedProcessError, match="stdout.*byte ceiling"):
            bounded_process.run_bounded_command(command, maximum_stdout_bytes=8)


def test_stderr_ceiling_and_nonzero_exit_are_separate():
    command = [sys.executable, "-c", "import os; os.write(2,b'error'); raise SystemExit(7)"]
    result = bounded_process.run_bounded_command(command, maximum_stderr_bytes=5)
    assert result.returncode == 7 and result.stderr == b"error"
    with pytest.raises(bounded_process.BoundedProcessError, match="stderr.*byte ceiling"):
        bounded_process.run_bounded_command(command, maximum_stderr_bytes=4)


def _still_running(pid):
    try:
        status = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    # A terminated orphan may remain a zombie until its host init reaps it.
    return status[status.rfind(")") + 2:].split()[0] != "Z"


@pytest.mark.parametrize("failure", ("timeout", "overflow", "interrupt", "success"))
def test_managed_group_descendants_stop_and_direct_child_is_reaped(tmp_path, monkeypatch, failure):
    pid_file = tmp_path / "descendant.pid"
    program = '''
import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                         stdout=subprocess.DEVNULL if sys.argv[2] == "success" else None,
                         stderr=subprocess.DEVNULL if sys.argv[2] == "success" else None)
Path(sys.argv[1]).write_text(str(child.pid))
os.write(1, b"ready")
if sys.argv[2] == "overflow":
    os.write(1, b"x" * 1024)
if sys.argv[2] == "success":
    sys.exit(0)
time.sleep(30)
'''
    children = []
    real_popen = subprocess.Popen

    def popen(*args, **kwargs):
        child = real_popen(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(bounded_process.subprocess, "Popen", popen)
    if failure == "interrupt":
        real_read = os.read

        def interrupted_read(fd, size):
            if pid_file.exists():
                raise KeyboardInterrupt("synthetic capture interruption")
            return real_read(fd, size)

        monkeypatch.setattr(bounded_process.os, "read", interrupted_read)
    command = [sys.executable, "-c", program, str(pid_file), failure]
    try:
        if failure == "success":
            result = bounded_process.run_bounded_command(command, timeout_seconds=5, maximum_stdout_bytes=8)
            assert result.returncode == 0 and result.stdout == b"ready"
        else:
            expected = KeyboardInterrupt if failure == "interrupt" else bounded_process.BoundedProcessError
            with pytest.raises(expected):
                bounded_process.run_bounded_command(command, timeout_seconds=1, maximum_stdout_bytes=8)
        assert len(children) == 1 and children[0].returncode is not None
        assert all(stream.closed for stream in (children[0].stdin, children[0].stdout, children[0].stderr))
        assert pid_file.exists(), "descendant must actually start before the failure"
        pid = int(pid_file.read_text())
        deadline = time.monotonic() + 1
        while _still_running(pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _still_running(pid)
    finally:
        # Cleanup is limited to the disposable group created by this test,
        # including on a future regression in the production helper.
        for child in children:
            bounded_process._kill_process_tree(child)


def test_git_operations_match_original_bytes_for_same_commit(tmp_path):
    repo = tmp_path / "small-source"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    (repo / "member.txt").write_bytes(b"exact original bytes\n")
    subprocess.run(["git", "-C", str(repo), "add", "member.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=Synthetic fixture", "-c",
                    "user.email=fixture@example.invalid", "commit", "--quiet", "-m", "Synthetic source"], check=True)
    for arguments in (("rev-parse", "HEAD"), ("status", "--porcelain=v1", "--untracked-files=all"),
                      ("ls-tree", "-r", "-z", "--full-tree", "HEAD"), ("archive", "HEAD")):
        original = subprocess.run(["git", *arguments], cwd=repo, check=True, capture_output=True, timeout=60).stdout
        assert source_closure._git(repo, *arguments) == original


def test_git_capture_bound_failures_are_handled_source_refusals(tmp_path, monkeypatch):
    calls = []

    def bounded(argv, **kwargs):
        calls.append((argv, kwargs))
        raise bounded_process.BoundedProcessError("bounded local command stdout exceeds its byte ceiling")

    monkeypatch.setattr(source_closure, "run_bounded_command", bounded)
    with pytest.raises(PermissionError, match="target source capture failed.*byte ceiling"):
        source_closure._git(tmp_path, "archive", "HEAD")
    assert calls[0][0] == ("git", "archive", "HEAD")
    assert calls[0][1]["maximum_stdout_bytes"] == 256 * 1024**2
    assert calls[0][1]["maximum_stderr_bytes"] == 64 * 1024
    assert calls[0][1]["timeout_seconds"] == 60


def test_nonzero_git_exit_is_handled_source_refusal(tmp_path):
    with pytest.raises(PermissionError, match="target source capture failed"):
        source_closure._git(tmp_path, "archive", "HEAD")


def test_complete_clean_capture_checks_actual_executing_package(tmp_path):
    repo, commit, tree = build_source_repo(tmp_path)
    original_archive = subprocess.check_output(["git", "-C", str(repo), "archive", "HEAD"])
    program = '''
from pathlib import Path
import json, sys
from empirical_lawhood.infrastructure.source_closure import capture_clean_target_closure
from empirical_lawhood.infrastructure.source_origin import require_executing_target_source
root = Path(sys.argv[1]); require_executing_target_source(root)
closure = capture_clean_target_closure(root, "synthetic.capture")
print(json.dumps({"commit":closure.implementation_commit,
                  "archive":closure.implementation_sha256,"tree":closure.source_tree_sha256}))
'''
    result = subprocess.run([sys.executable, "-c", program, str(repo)], cwd=tmp_path,
                            env=dict(os.environ, PYTHONPATH=str(repo / "src"), PYTHONDONTWRITEBYTECODE="1"),
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"commit": commit, "archive": sha256(original_archive).hexdigest(), "tree": tree}
