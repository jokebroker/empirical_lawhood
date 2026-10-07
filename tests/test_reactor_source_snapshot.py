# SPDX-License-Identifier: MPL-2.0
"""Synthetic source plumbing tests confer no authentic reactor source identity."""

from hashlib import sha256
import os

import pytest

from empirical_lawhood.adapters.simulators.reactor_prefix_response import batch_input
from empirical_lawhood.adapters._bounded_files import AdapterFileBoundError


@pytest.mark.parametrize("members", (("z.txt", "a.txt", "sub/b.txt"), ("empty",)))
def test_census_order_and_digests_match_captured_original_bytes(tmp_path, monkeypatch, members):
    expected = []
    for name in members:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = b"" if name == "empty" else name.encode()
        path.write_bytes(raw)
        expected.append((name, sha256(raw).hexdigest()))
    snapshot = batch_input._ReactorSourceSnapshot(tmp_path)
    assert snapshot.census() == tuple(sorted(expected))

    def no_reopen(*args, **kwargs):
        pytest.fail("captured comparator bytes were reopened")

    monkeypatch.setattr(batch_input, "read_bounded_contained", no_reopen)
    for name in members:
        (tmp_path / name).write_bytes(b"later replacement")
        assert snapshot.read_member(name, 100) == (b"" if name == "empty" else name.encode())


def test_capture_cache_preserves_tighter_member_bound(tmp_path):
    path = tmp_path / "member"
    path.write_bytes(b"1234")
    snapshot = batch_input._ReactorSourceSnapshot(tmp_path)
    assert snapshot.read_member(path.name, 4) == b"1234"
    with pytest.raises(AdapterFileBoundError, match="byte limit"):
        snapshot.read_member(path.name, 3)


@pytest.mark.parametrize("tree", ("broad", "deep"))
def test_empty_directories_cannot_bypass_traversal_work_bound(tmp_path, monkeypatch, tree):
    # The same production work owner counts breadth and reopened depth. Small
    # fixtures exercise its boundary without constructing thousands of inodes.
    monkeypatch.setattr(batch_input, "_MAX_SOURCE_TRAVERSAL_WORK", 8)
    if tree == "broad":
        for i in range(9):
            (tmp_path / f"dir-{i}").mkdir()
    else:
        path = tmp_path
        for i in range(5):
            path /= f"dir-{i}"
            path.mkdir()
    with pytest.raises(ValueError, match="SOURCE_TRAVERSAL_TOO_LARGE"):
        batch_input._ReactorSourceSnapshot(tmp_path).census()


def test_file_count_refuses_before_reading_unrestricted_members(tmp_path, monkeypatch):
    for i in range(129):
        (tmp_path / f"member-{i}").touch()

    def no_read(*args, **kwargs):
        pytest.fail("over-count census reached member reads")

    monkeypatch.setattr(batch_input, "read_bounded_contained", no_read)
    with pytest.raises(ValueError, match="SOURCE_CENSUS_TOO_LARGE"):
        batch_input._ReactorSourceSnapshot(tmp_path).census()


def test_total_capture_bytes_are_bounded_during_reads(tmp_path, monkeypatch):
    monkeypatch.setattr(batch_input, "_MAX_SOURCE_BYTES", 5)
    (tmp_path / "a").write_bytes(b"123")
    (tmp_path / "b").write_bytes(b"456")
    snapshot = batch_input._ReactorSourceSnapshot(tmp_path)
    with pytest.raises(ValueError, match="SOURCE_CENSUS_INVALID.*byte limit"):
        snapshot.census()
    assert snapshot.payloads == {"a": b"123"}
    assert snapshot.total_bytes == 3


@pytest.mark.parametrize("linked", ("member", "directory", "root-ancestor"))
def test_snapshot_refuses_links_before_traversal_or_read(tmp_path, linked):
    root = tmp_path / "source"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret").write_bytes(b"outside input")
    if linked == "member":
        (root / "linked").symlink_to(outside / "secret")
    elif linked == "directory":
        (root / "linked").symlink_to(outside, target_is_directory=True)
    else:
        linked_root = tmp_path / "linked"
        linked_root.symlink_to(root, target_is_directory=True)
        root = linked_root
    with pytest.raises(ValueError, match="SOURCE_CENSUS_ESCAPES_ROOT"):
        batch_input._ReactorSourceSnapshot(root).census()


@pytest.mark.parametrize("change", ("grow", "truncate", "replace"))
def test_census_member_changes_are_refused(tmp_path, monkeypatch, change):
    path = tmp_path / "comparator.py"
    path.write_bytes(b"original")
    real_read = os.read
    changed = False

    def read(fd, maximum):
        nonlocal changed
        if not changed:
            changed = True
            if change == "replace":
                path.unlink()
            path.write_bytes(b"growing content" if change == "grow" else b"x")
        return real_read(fd, maximum)

    monkeypatch.setattr(os, "read", read)
    with pytest.raises(ValueError, match="SOURCE_CENSUS_INVALID"):
        batch_input._ReactorSourceSnapshot(tmp_path).census()
    assert changed


def test_special_member_never_waits_for_writer(tmp_path):
    # O_NONBLOCK is characterized by the reader child tests; this census must
    # send a FIFO through that same contained descriptor type check.
    os.mkfifo(tmp_path / "fifo")
    with pytest.raises(ValueError, match="SOURCE_CENSUS_INVALID.*regular file"):
        batch_input._ReactorSourceSnapshot(tmp_path).census()


def test_prepared_handoff_reuses_authenticated_comparator_capture(tmp_path, monkeypatch):
    """Synthetic handoff plumbing; it does not select an authentic source pin."""
    from types import SimpleNamespace
    from empirical_lawhood.adapters.simulators.reactor_prefix_response import prepared_input

    member = tmp_path / "comparator.py"
    original = b"synthetic original comparator\n"
    member.write_bytes(original)
    snapshot_seen = []
    read_count = 0
    original_read = batch_input.read_bounded_contained

    def read(*args, **kwargs):
        nonlocal read_count
        read_count += 1
        return original_read(*args, **kwargs)

    monkeypatch.setattr(batch_input, "read_bounded_contained", read)
    monkeypatch.setattr(prepared_input, "COMPARATOR_SOURCES", (
        ("REF", member.name, sha256(original).hexdigest()),
    ))

    def load(root, *, _snapshot):
        snapshot_seen.append(_snapshot)
        return SimpleNamespace(fingerprint=lambda: "a" * 64)

    def inspect(*args, _snapshot, **kwargs):
        assert _snapshot is snapshot_seen[0]
        member.write_bytes(b"later replacement")
        assert _snapshot.census() == ((member.name, sha256(original).hexdigest()),)
        assert _snapshot.read_member(member.name, 1024**2) == original
        return {"native_contact": False, "provider_built": False}

    monkeypatch.setattr(prepared_input, "load_batch_source", load)
    monkeypatch.setattr(prepared_input, "inspect_reactor_binding", inspect)
    config = prepared_input.ReactorPreparedInput(
        "empirical-lawhood-reactor-causal-response-study-input", "causal-response-study",
    )
    report = prepared_input.check_prepared_input(config, source_root=tmp_path, discovery_file=None)
    assert report["comparator_sources_authenticated"] is True
    assert report["campaign_issued"] is False and report["native_tasks_executed"] == 0
    assert read_count == 1
