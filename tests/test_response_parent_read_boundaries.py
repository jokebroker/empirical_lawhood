# SPDX-License-Identifier: MPL-2.0
"""Race and work bounds at the actual custody replay and consuming parent reader."""

import json
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

import pytest

from empirical_lawhood.adapters.composition import response_parent_custody, response_parent_reader
from tests.test_response_parent_custody import _fixture


def _read_parent(kwargs):
    return response_parent_reader.read_authenticated_parent(
        **{key: value for key, value in kwargs.items() if key != "source_schema"}
    )


@pytest.mark.parametrize("replacement", ("member", "ancestor", "root"))
def test_parent_replay_rejects_replacement_before_open(tmp_path, monkeypatch, replacement):
    kwargs, details = _fixture(tmp_path)
    output = details["output_path"]
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / output.name).write_bytes(output.read_bytes())
    original_open = os.open
    replaced = False

    def replace_before_open(path, flags, *args, **options):
        nonlocal replaced
        trigger = {"member": output.name, "ancestor": "outputs", "root": "/"}[replacement]
        if not replaced and str(path) == trigger:
            replaced = True
            if replacement == "member":
                output.unlink()
                output.symlink_to(outside / output.name)
            else:
                selected = kwargs["source_root"] if replacement == "root" else output.parent.parent
                selected.rename(tmp_path / "original-tree")
                selected.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **options)

    monkeypatch.setattr(os, "open", replace_before_open)
    with pytest.raises(ValueError, match="PARENT_MEMBER_INVALID"):
        _read_parent(kwargs)
    assert replaced


@pytest.mark.parametrize("member", ("parent", "streamed-marker"))
@pytest.mark.parametrize("change", ("grow", "truncate", "replace-open-file"))
def test_parent_replay_bounds_and_rejects_concurrent_changes(tmp_path, monkeypatch, member, change):
    kwargs, details = _fixture(tmp_path)
    if member == "parent":
        target = details["output_path"]
    else:
        entries = json.loads(kwargs["parent_manifest"].read_bytes())
        target = next(Path(e["path"]) for e in entries if e["path"].endswith(".commit.json"))
    before = target.stat()
    outside = tmp_path / "outside.json"
    outside.write_bytes(target.read_bytes())
    original_read = os.read
    requests, returned = [], []

    def change_before_read(fd, count):
        if os.fstat(fd).st_ino == before.st_ino:
            if not requests:
                if change == "replace-open-file":
                    target.unlink()
                    target.symlink_to(outside)
                else:
                    target.write_bytes(b"x" * (before.st_size + 1_000_000) if change == "grow" else b"x")
            requests.append(count)
            value = original_read(fd, count)
            returned.append(len(value))
            return value
        return original_read(fd, count)

    monkeypatch.setattr(os, "read", change_before_read)
    with pytest.raises(ValueError, match="PARENT_MEMBER_INVALID"):
        _read_parent(kwargs)
    assert requests and max(requests) <= before.st_size
    assert sum(returned) <= before.st_size + 1


@pytest.mark.parametrize("member", ("parent", "streamed-marker"))
def test_parent_replay_rejects_unlink_with_equal_timestamps(tmp_path, monkeypatch, member):
    kwargs, details = _fixture(tmp_path)
    if member == "parent":
        target = details["output_path"]
    else:
        entries = json.loads(kwargs["parent_manifest"].read_bytes())
        target = next(Path(e["path"]) for e in entries if e["path"].endswith(".commit.json"))
    original_bytes = target.read_bytes()
    before = target.stat()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"unauthenticated replacement bytes must never be read")
    original_fstat, original_read = os.fstat, os.read
    link_counts, observed_times, chunks = [], [], []

    class EqualTimestamps:
        # Preserve the actual descriptor identity, size, type and link count.
        # Only clock observations are fixed; substitution is a real unlink.
        st_mtime_ns = before.st_mtime_ns
        st_ctime_ns = before.st_ctime_ns

        def __init__(self, value):
            self.value = value

        def __getattr__(self, name):
            return getattr(self.value, name)

    def frozen_fstat(fd):
        value = original_fstat(fd)
        if value.st_ino == before.st_ino and value.st_dev == before.st_dev:
            value = EqualTimestamps(value)
            link_counts.append(value.st_nlink)
            observed_times.append((value.st_mtime_ns, value.st_ctime_ns))
        return value

    def unlink_before_read(fd, count):
        value = original_fstat(fd)
        if value.st_ino == before.st_ino and value.st_dev == before.st_dev:
            if not chunks:
                target.unlink()
                target.symlink_to(outside)
            chunk = original_read(fd, count)
            chunks.append(chunk)
            return chunk
        return original_read(fd, count)

    monkeypatch.setattr(os, "fstat", frozen_fstat)
    monkeypatch.setattr(os, "read", unlink_before_read)
    with pytest.raises(ValueError, match="PARENT_MEMBER_INVALID"):
        _read_parent(kwargs)
    assert link_counts[0] == 1 and link_counts[-1] == 0
    assert len(set(observed_times)) == 1
    assert b"".join(chunks) == original_bytes


@pytest.mark.parametrize("kind", ("fifo", "directory"))
def test_parent_replay_rejects_nonregular_files_without_blocking(tmp_path, kind):
    # A separate process makes a regression to blocking open a bounded failure.
    program = "\nimport os, sys\nfrom pathlib import Path\nfrom tests.test_response_parent_custody import _fixture\nfrom empirical_lawhood.adapters.composition.response_parent_reader import read_authenticated_parent\nargs, details = _fixture(Path(sys.argv[1]))\npath = details['output_path']\npath.unlink()\nos.mkfifo(path) if sys.argv[2] == 'fifo' else path.mkdir()\ntry:\n    read_authenticated_parent(**{k: v for k, v in args.items() if k != 'source_schema'})\nexcept ValueError as error:\n    assert str(error) == 'RESPONSE_PARENT_MEMBER_INVALID', str(error)\nelse:\n    raise AssertionError('nonregular parent accepted')\n"
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), kind],
        cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_parent_replay_accepts_exact_limits_and_refuses_one_byte_less(tmp_path, monkeypatch):
    kwargs, _ = _fixture(tmp_path)
    raw = kwargs["parent_manifest"].read_bytes()
    entries = json.loads(raw)
    maximum_member = max(e["bytes"] for e in entries)
    monkeypatch.setattr(response_parent_custody, "_MAX_MANIFEST_BYTES", len(raw))
    monkeypatch.setattr(response_parent_custody, "_MAX_MEMBER_BYTES", maximum_member)
    monkeypatch.setattr(response_parent_custody, "_MAX_RECORD_BYTES", maximum_member)
    monkeypatch.setattr(response_parent_custody, "_MAX_TOTAL_BYTES", sum(e["bytes"] for e in entries))
    assert _read_parent(kwargs).custody.original_root_ids == ('synthetic.source-qualification.prepared.r000',)
    monkeypatch.setattr(response_parent_custody, "_MAX_MANIFEST_BYTES", len(raw) - 1)
    with pytest.raises(ValueError, match="PARENT_MEMBER_INVALID"):
        _read_parent(kwargs)


def test_parent_reader_returns_replayed_bytes_without_reopening(tmp_path, monkeypatch):
    kwargs, details = _fixture(tmp_path)
    original = details["output_path"].read_bytes()
    replay = response_parent_reader.replay_response_parent_custody
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"untrusted replacement after successful replay")

    def replace_after_replay(**args):
        verified = replay(**args)
        details["output_path"].unlink()
        details["output_path"].symlink_to(outside)
        return verified

    monkeypatch.setattr(response_parent_reader, 'replay_response_parent_custody', replace_after_replay)
    result = _read_parent(kwargs)
    assert result.raw == original
    assert result.grant.parent.physical_sha256 == sha256(original).hexdigest()
