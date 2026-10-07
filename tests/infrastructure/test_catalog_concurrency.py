# SPDX-License-Identifier: MPL-2.0
# Adapted from icf-yolo: synthetic shared-core contract regressions.
from pathlib import Path
import subprocess
import sys
import pytest
from empirical_lawhood.infrastructure import (
    CatalogMutationLockUnavailable, CatalogRebuildError,
    production_catalog_mutation_lock, production_catalog_mutation_lock_path,
    rebuild_production_catalog,
)
from empirical_lawhood.infrastructure.catalog_projection import encode_catalog_snapshot
from empirical_lawhood.runtime.catalog import CatalogSnapshot

def test_production_catalog_mutation_lock_refuses_same_and_second_process(
    tmp_path: Path,
) -> None:
    (tmp_path / ".git").mkdir()
    projection = encode_catalog_snapshot(CatalogSnapshot.empty())
    child_program = """
from pathlib import Path
import sys
from empirical_lawhood.infrastructure import (
    CatalogMutationLockUnavailable,
    production_catalog_mutation_lock,
)
try:
    with production_catalog_mutation_lock(Path(sys.argv[1]), timeout_seconds=0):
        raise SystemExit(7)
except CatalogMutationLockUnavailable:
    raise SystemExit(0)
"""
    with production_catalog_mutation_lock(tmp_path, timeout_seconds=0) as held:
        assert held.lock_path == production_catalog_mutation_lock_path(tmp_path)
        assert held.lock_path.is_file()
        assert not held.lock_path.is_symlink()
        with pytest.raises(CatalogMutationLockUnavailable, match="remained held"):
            with production_catalog_mutation_lock(tmp_path, timeout_seconds=0):
                pass
        child = subprocess.run(
            [sys.executable, "-c", child_program, str(tmp_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert child.returncode == 0, child.stderr
        with pytest.raises(CatalogRebuildError, match="mutation lock"):
            rebuild_production_catalog(
                tmp_path,
                projection,
                mutation_lock_timeout_seconds=0,
            )

    result = rebuild_production_catalog(
        tmp_path,
        projection,
        mutation_lock_timeout_seconds=0,
    )
    assert result.snapshot_sha256 == CatalogSnapshot.empty().fingerprint()
