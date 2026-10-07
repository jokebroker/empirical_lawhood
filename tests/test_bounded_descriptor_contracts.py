# SPDX-License-Identifier: MPL-2.0
"""Independent expected contracts for bounded reads, hashing and early stops."""

from hashlib import sha256
import os
import stat

import pytest

from empirical_lawhood.adapters import _bounded_files as adapter
from empirical_lawhood.infrastructure import bounded_io as control


OPERATIONS = ("control-read", "control-hash", "control-match", "control-search",
              "adapter-read", "adapter-hash", "contained-read", "contained-hash")


def operate(name, root, path, maximum=64, expected=b"abcdefghij", needles=(b"ij",)):
    if name == "control-read":
        return control.read_bounded_bytes(path, maximum_bytes=maximum)
    if name == "control-hash":
        return control.bounded_file_sha256(path, maximum_bytes=maximum)
    if name == "control-match":
        return control.file_matches_bytes(path, expected)
    if name == "control-search":
        return control.bounded_file_contains_any(path, needles, maximum_bytes=maximum)
    if name == "adapter-read":
        return adapter.read_bounded_regular(path, maximum_bytes=maximum)
    if name == "adapter-hash":
        return adapter.sha256_bounded_regular(path, maximum_bytes=maximum)
    if name == "contained-read":
        return adapter.read_bounded_contained(root, path, maximum_bytes=maximum)
    return adapter.sha256_bounded_contained(root, path, maximum_bytes=maximum)


def observe_io(monkeypatch):
    original_open, original_close, original_read = os.open, os.close, os.read
    current = {}
    acquisitions = []
    requests = []

    def opened(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        entry = [fd, 0]
        acquisitions.append(entry)
        current[fd] = entry
        return fd

    def closed(fd):
        if fd in current:
            current.pop(fd)[1] += 1
        return original_close(fd)

    def read(fd, size):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            requests.append(size)
        return original_read(fd, size)

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "close", closed)
    monkeypatch.setattr(os, "read", read)
    monkeypatch.setattr(control, "CONTROL_PLANE_CHUNK_BYTES", 4)
    monkeypatch.setattr(adapter, "_CHUNK_BYTES", 4)
    return acquisitions, requests, original_read


@pytest.mark.parametrize("name", OPERATIONS)
@pytest.mark.parametrize("raw", (b"", b"abcdefghij"))
def test_valid_bytes_digest_and_exactly_once_close(tmp_path, monkeypatch, name, raw):
    path = tmp_path / "member"
    path.write_bytes(raw)
    acquired, _, _ = observe_io(monkeypatch)
    result = operate(name, tmp_path, path, maximum=len(raw), expected=raw, needles=(b"ij",))
    if name.endswith("read"):
        assert result == raw
    elif name == "control-hash":
        assert result == (len(raw), sha256(raw).hexdigest())
    elif name.endswith("hash"):
        digest = result[1] if name == "contained-hash" else result
        assert digest == sha256(raw).hexdigest()
    elif name == "control-match":
        assert result is True
    else:
        assert result is bool(raw)
    assert acquired and all(count == 1 for _, count in acquired)


@pytest.mark.parametrize("name", OPERATIONS)
def test_initial_type_refusal_and_interruption_close_exactly_once(tmp_path, monkeypatch, name):
    path = tmp_path / "member"
    path.write_bytes(b"abcdefghij")
    acquired, _, _ = observe_io(monkeypatch)
    original_fstat = os.fstat

    def interrupted(fd):
        observed = original_fstat(fd)
        if stat.S_ISREG(observed.st_mode):
            raise KeyboardInterrupt("synthetic descriptor validation interruption")
        return observed

    monkeypatch.setattr(os, "fstat", interrupted)
    with pytest.raises(KeyboardInterrupt):
        operate(name, tmp_path, path)
    assert acquired and all(count == 1 for _, count in acquired)


@pytest.mark.parametrize("name", OPERATIONS)
@pytest.mark.parametrize("failure", ("read-error", "interrupted-read"))
def test_read_error_translation_and_close_are_preserved(tmp_path, monkeypatch, name, failure):
    path = tmp_path / "member"
    path.write_bytes(b"abcdefghij")
    acquired, _, _ = observe_io(monkeypatch)

    def failed(fd, size):
        if failure == "interrupted-read":
            raise SystemExit("synthetic read interruption")
        raise OSError("synthetic read error")

    monkeypatch.setattr(os, "read", failed)
    expected = SystemExit if failure == "interrupted-read" else (
        adapter.AdapterFileBoundError if name.startswith("contained") else OSError
    )
    with pytest.raises(expected):
        operate(name, tmp_path, path)
    assert acquired and all(count == 1 for _, count in acquired)


@pytest.mark.parametrize("name", OPERATIONS)
def test_initial_fstat_oserror_preserves_open_vs_validation_policy(tmp_path, monkeypatch, name):
    path = tmp_path / "member"
    path.write_bytes(b"abcdefghij")
    acquired, _, _ = observe_io(monkeypatch)
    original_fstat = os.fstat

    def failed(fd):
        observed = original_fstat(fd)
        if stat.S_ISREG(observed.st_mode):
            raise OSError("synthetic initial fstat error")
        return observed

    monkeypatch.setattr(os, "fstat", failed)
    expected = adapter.AdapterFileBoundError if name.startswith("contained") else OSError
    with pytest.raises(expected):
        operate(name, tmp_path, path)
    assert acquired and all(count == 1 for _, count in acquired)


@pytest.mark.parametrize("name", OPERATIONS)
def test_missing_member_error_category_is_preserved(tmp_path, name):
    error = FileNotFoundError if name.startswith("control") else adapter.AdapterFileBoundError
    with pytest.raises(error):
        operate(name, tmp_path, tmp_path / "missing")


@pytest.mark.parametrize("name", ("control-read", "control-hash", "adapter-read", "adapter-hash", "contained-read", "contained-hash"))
def test_opened_size_vs_declared_maximum_remain_distinct(tmp_path, monkeypatch, name):
    path = tmp_path / "member"
    path.write_bytes(b"abcdefghij")
    acquired, requests, original_read = observe_io(monkeypatch)
    changed = False

    def read(fd, size):
        nonlocal changed
        if not changed:
            changed = True
            path.write_bytes(b"abcdefghijklmnop")
        requests.append(size)
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", read)
    expected = control.BoundedFileIOError if name.startswith("control") else adapter.AdapterFileBoundError
    message = "changed during" if name.startswith("control") else "grew beyond"
    with pytest.raises(expected, match=message):
        operate(name, tmp_path, path, maximum=20)
    assert requests == ([4, 4, 4, 4, 4] if name.startswith("control") else [4, 4, 2, 1])
    assert acquired and all(count == 1 for _, count in acquired)


@pytest.mark.parametrize(("name", "kwargs", "expected_requests"), (
    ("control-match", {"expected": b"Xbcdefghij"}, [4]),
    ("control-match", {"expected": b"abcdefghij-extra"}, []),
    ("control-search", {"needles": (b"bc",)}, [4]),
    ("control-search", {"needles": (b"def",)}, [4, 4]),
    ("control-search", {"needles": ()}, [4, 4, 4, 4]),
))
def test_comparison_and_search_keep_early_stop_or_full_hash_contract(tmp_path, monkeypatch, name, kwargs, expected_requests):
    path = tmp_path / "member"
    path.write_bytes(b"abcdefghij")
    acquired, requests, _ = observe_io(monkeypatch)
    result = operate(name, tmp_path, path, **kwargs)
    assert result is (name == "control-search" and bool(kwargs.get("needles")))
    assert requests == expected_requests
    assert acquired and all(count == 1 for _, count in acquired)


@pytest.mark.parametrize("name", ("control-read", "control-hash", "control-search", "adapter-read", "adapter-hash", "contained-read", "contained-hash"))
def test_negative_bounds_refuse_before_open(tmp_path, monkeypatch, name):
    def forbidden(*args, **kwargs):
        pytest.fail("negative maximum reached descriptor opening")

    monkeypatch.setattr(os, "open", forbidden)
    with pytest.raises(ValueError, match="nonnegative"):
        operate(name, tmp_path, tmp_path / "missing", maximum=-1)


@pytest.mark.parametrize("name", OPERATIONS)
def test_directory_type_refusal_never_reads_and_closes_exactly_once(tmp_path, monkeypatch, name):
    member = tmp_path / "directory"
    member.mkdir()
    acquired, requests, _ = observe_io(monkeypatch)
    expected = control.BoundedFileIOError if name.startswith("control") else adapter.AdapterFileBoundError
    with pytest.raises(expected, match="not a regular file"):
        operate(name, tmp_path, member)
    assert requests == []
    assert acquired and all(count == 1 for _, count in acquired)


@pytest.mark.parametrize("name", OPERATIONS)
def test_final_symlink_refusal_closes_every_acquired_directory(tmp_path, monkeypatch, name):
    outside = tmp_path / "outside"
    outside.write_bytes(b"abcdefghij")
    member = tmp_path / "member"
    member.symlink_to(outside)
    acquired, requests, _ = observe_io(monkeypatch)
    expected = OSError if name.startswith("control") else adapter.AdapterFileBoundError
    with pytest.raises(expected):
        operate(name, tmp_path, member)
    assert requests == []
    assert all(count == 1 for _, count in acquired)


def test_search_needle_ceiling_refuses_before_descriptor_open(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "MAX_CONTROL_PLANE_JSON_BYTES", 2)

    def forbidden(*args, **kwargs):
        pytest.fail("oversized needle reached descriptor opening")

    monkeypatch.setattr(os, "open", forbidden)
    with pytest.raises(control.BoundedFileIOError, match="forbidden-payload probe exceeds"):
        control.bounded_file_contains_any(tmp_path / "missing", (b"abc",), maximum_bytes=64)


def test_explicit_comparison_bound_preserves_exact_stable_bytes(tmp_path, monkeypatch):
    path = tmp_path / "member"
    path.write_bytes(b"abcdefghij")
    acquired, requests, _ = observe_io(monkeypatch)
    assert control.file_matches_bytes(path, b"abcdefghij", maximum_bytes=20)
    assert requests == [4, 4, 4, 4]
    assert acquired and all(count == 1 for _, count in acquired)
    assert not control.file_matches_bytes(path, b"short", maximum_bytes=20)
    assert not control.file_matches_bytes(path, b"longer than actual", maximum_bytes=20)
    with pytest.raises(control.BoundedFileIOError, match="byte limit"):
        control.file_matches_bytes(path, b"abcdefghij", maximum_bytes=9)


def test_oversized_comparison_expectation_refuses_before_open(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "open", lambda *args, **kwargs: pytest.fail("expectation reached open"))
    with pytest.raises(control.BoundedFileIOError, match="expected input exceeds"):
        control.file_matches_bytes(tmp_path / "missing", b"abc", maximum_bytes=2)


def test_explicit_comparison_detects_same_length_change_during_read(tmp_path, monkeypatch):
    path = tmp_path / "member"
    path.write_bytes(b"abcdefghij")
    before = path.stat()
    acquired, _, actual_read = observe_io(monkeypatch)
    changed = False

    def read(fd, count):
        nonlocal changed
        chunk = actual_read(fd, count)
        if not changed:
            changed = True
            path.write_bytes(b"Xbcdefghij")
            # Make identity drift independent of the filesystem timestamp
            # resolution when both operations occur in one clock tick.
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        return chunk

    monkeypatch.setattr(os, "read", read)
    with pytest.raises(control.BoundedFileIOError, match="changed during comparison"):
        control.file_matches_bytes(path, b"abcdefghij", maximum_bytes=20)
    assert acquired and all(count == 1 for _, count in acquired)
