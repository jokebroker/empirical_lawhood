# SPDX-License-Identifier: MPL-2.0
"""Nonblocking and interruption-safe reads preserve authenticated input bytes."""

from hashlib import sha256
import os
import stat
import subprocess
import sys

import pytest

from empirical_lawhood.api import codecs
from empirical_lawhood.infrastructure import bounded_io, dataset_io, repository_identity
from empirical_lawhood.infrastructure.artifacts import GuardedExternalRoot
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.dataset_authority import DatasetStorageScope
from empirical_lawhood.runtime.artifacts import ExternalRootContract
from empirical_lawhood.runtime.operator_profile import OperatorStorageProfile


EXPECTED = b"authenticated original bytes\n"


def read_input(owner, root, path):
    if owner == "authoring":
        return codecs._read_bounded_authoring_bytes(path, maximum_bytes=len(EXPECTED))
    if owner == "repository":
        return repository_identity._read_exact_file(root, repository_identity.RepositoryArtifactSpec(
            relative_path=path.name, artifact_id="test.input", role="test.input",
            payload_schema="empirical-lawhood/testing/input-boundary", media_type="application/octet-stream",
            maximum_bytes=len(EXPECTED),
        ))
    if owner == "bounded":
        return bounded_io.read_bounded_bytes(path, maximum_bytes=len(EXPECTED))
    guarded_root = GuardedExternalRoot(ExternalRootContract(
        "synthetic-input-root", "synthetic input", str(root), "/", OperatorStorageProfile.SCHEMA,
        1, None, None, (),
    ))
    scope = DatasetStorageScope("synthetic.input-scope", ObjectIdentity.from_record(
        guarded_root.contract.storage_root_id, guarded_root.contract,
    ), path.name)
    with dataset_io.ExternalDatasetSourceOpener(guarded_root).open_exact(
        source_scope=scope, expected_size_bytes=len(EXPECTED), expected_sha256=sha256(EXPECTED).hexdigest(),
        maximum_bytes=len(EXPECTED), trusted_at_utc="2026-10-06T00:00:00Z",
    ) as source:
        return source.stream.read()


@pytest.mark.parametrize("owner", ("authoring", "repository", "dataset", "bounded"))
def test_reader_original_regular_bytes_are_unchanged(tmp_path, owner):
    path = tmp_path / "input.bin"
    path.write_bytes(EXPECTED)
    assert read_input(owner, tmp_path, path) == EXPECTED


@pytest.mark.parametrize("owner", ("authoring", "repository", "dataset"))
def test_fifo_substituted_at_final_open_refuses_without_writer(tmp_path, owner):
    # Execute the affected real reader in a child so a blocking regression cannot
    # hang the suite. Replacement occurs after the ordinary pathname prechecks.
    program = '''
import os, sys
from pathlib import Path
from tests.test_input_reader_boundaries import EXPECTED, read_input
root = Path(sys.argv[1]); path = root / "input.bin"; path.write_bytes(EXPECTED)
original_open = os.open
replaced = False
def open_member(member, flags, *args, **kwargs):
    global replaced
    if not replaced and Path(member).name == path.name:
        replaced = True
        path.unlink(); os.mkfifo(path)
    return original_open(member, flags, *args, **kwargs)
os.open = open_member
try:
    read_input(sys.argv[2], root, path)
except (OSError, ValueError, RuntimeError):
    assert replaced
else:
    raise AssertionError("FIFO accepted")
'''
    result = subprocess.run([sys.executable, "-c", program, str(tmp_path), owner],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("owner", ("authoring", "repository", "dataset", "bounded"))
@pytest.mark.parametrize("change", ("grow", "truncate", "replace", "symlink"))
def test_reader_refuses_mutation_during_read(tmp_path, monkeypatch, owner, change):
    path = tmp_path / "input.bin"
    path.write_bytes(EXPECTED)
    original_read = os.read
    changed = False

    def read(fd, size):
        nonlocal changed
        if not changed and stat.S_ISREG(os.fstat(fd).st_mode):
            changed = True
            if change == "grow":
                path.write_bytes(EXPECTED + b"extra")
            elif change == "truncate":
                path.write_bytes(b"x")
            else:
                path.unlink()
                if change == "replace":
                    path.write_bytes(EXPECTED)
                else:
                    outside = tmp_path / "outside.bin"
                    outside.write_bytes(b"outside")
                    path.symlink_to(outside)
        return original_read(fd, size)

    monkeypatch.setattr(os, "read", read)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        read_input(owner, tmp_path, path)
    assert changed


@pytest.mark.parametrize("owner", ("authoring", "repository", "dataset", "bounded"))
@pytest.mark.parametrize("interruption", (KeyboardInterrupt, SystemExit))
def test_initial_fstat_interruption_closes_every_opened_descriptor(tmp_path, monkeypatch, owner, interruption):
    path = tmp_path / "input.bin"
    path.write_bytes(EXPECTED)
    original_open, original_fstat = os.open, os.fstat
    opened = []

    def open_tracked(*args, **kwargs):
        fd = original_open(*args, **kwargs)
        opened.append(fd)
        return fd

    def fstat(fd):
        observed = original_fstat(fd)
        if stat.S_ISREG(observed.st_mode):
            raise interruption("synthetic initial-fstat interruption")
        return observed

    monkeypatch.setattr(os, "open", open_tracked)
    monkeypatch.setattr(os, "fstat", fstat)
    with pytest.raises(interruption):
        read_input(owner, tmp_path, path)
    assert opened
    for fd in set(opened):
        with pytest.raises(OSError):
            original_fstat(fd)
