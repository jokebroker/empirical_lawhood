# SPDX-License-Identifier: MPL-2.0
"""Live unlink refusal does not depend on filesystem timestamp resolution."""

import os

import pytest

from empirical_lawhood.adapters import _bounded_files as adapter
from empirical_lawhood.infrastructure import bounded_io as control
from empirical_lawhood.infrastructure import repository_identity as repository


@pytest.mark.parametrize("operation", (
    "adapter-read", "adapter-hash", "contained-read", "contained-hash",
    "control-read", "control-hash", "control-match", "control-search", "repository-read",
))
def test_live_unlink_with_equal_timestamps_is_refused(tmp_path, monkeypatch, operation):
    path = tmp_path / "input.json"
    expected = b'authenticated original bytes'
    path.write_bytes(expected)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b'forbidden replacement bytes')
    before = path.stat()
    real_read, real_fstat = os.read, os.fstat
    chunks, link_counts, timestamps = [], [], []

    class EqualTimestamps:
        st_mtime_ns = before.st_mtime_ns
        st_ctime_ns = before.st_ctime_ns

        def __init__(self, observed):
            self.observed = observed

        def __getattr__(self, name):
            return getattr(self.observed, name)

    def fstat(fd):
        observed = real_fstat(fd)
        if (observed.st_dev, observed.st_ino) == (before.st_dev, before.st_ino):
            observed = EqualTimestamps(observed)
            link_counts.append(observed.st_nlink)
            timestamps.append((observed.st_mtime_ns, observed.st_ctime_ns))
        return observed

    def read(fd, maximum):
        observed = real_fstat(fd)
        if (observed.st_dev, observed.st_ino) == (before.st_dev, before.st_ino):
            if not chunks:
                path.unlink()
                path.symlink_to(outside)
            chunk = real_read(fd, maximum)
            chunks.append(chunk)
            return chunk
        return real_read(fd, maximum)

    operations = {
        "adapter-read": lambda: adapter.read_bounded_regular(path, maximum_bytes=len(expected)),
        "adapter-hash": lambda: adapter.sha256_bounded_regular(path, maximum_bytes=len(expected)),
        "contained-read": lambda: adapter.read_bounded_contained(tmp_path, path, maximum_bytes=len(expected)),
        "contained-hash": lambda: adapter.sha256_bounded_contained(tmp_path, path, maximum_bytes=len(expected)),
        "control-read": lambda: control.read_bounded_bytes(path, maximum_bytes=len(expected)),
        "control-hash": lambda: control.bounded_file_sha256(path, maximum_bytes=len(expected)),
        "control-match": lambda: control.file_matches_bytes(path, expected),
        "control-search": lambda: control.bounded_file_contains_any(path, (b'original',), maximum_bytes=len(expected)),
        "repository-read": lambda: repository._read_exact_file(tmp_path, repository.RepositoryArtifactSpec(
            relative_path=path.name, artifact_id="test.input", role="test.input",
            payload_schema='empirical-lawhood/testing/live-read-stability-input', media_type="application/json", maximum_bytes=len(expected),
        )),
    }
    monkeypatch.setattr(os, "fstat", fstat)
    monkeypatch.setattr(os, "read", read)
    with pytest.raises((adapter.AdapterFileBoundError, control.BoundedFileIOError,
                        repository.RepositoryIdentityError), match="changed during"):
        operations[operation]()
    assert link_counts[0] == 1 and link_counts[-1] == 0
    assert len(set(timestamps)) == 1
    assert b''.join(chunks) == expected
