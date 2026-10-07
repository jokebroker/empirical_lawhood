"""Scientific checks on an optional, authentic public Glenn archive."""

from __future__ import annotations

import os
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

pytestmark = pytest.mark.held

from empirical_lawhood.adapters.physical.glenn.contracts import (
    ACTION_COLUMNS,
    GLENN_2026_PROFILE,
    GlennAdapterError,
    GlennArchiveInput,
    GlennMemberRole,
)
from empirical_lawhood.adapters.physical.glenn.source import inspect_glenn_archive
from empirical_lawhood.adapters.physical.glenn.transform import transform_glenn_archive


class _CountingStream:
    def __init__(self, source: object) -> None:
        self.source = source
        self.read_count = 0

    def readable(self) -> bool:
        return self.source.readable()  # type: ignore[attr-defined]

    def seekable(self) -> bool:
        return self.source.seekable()  # type: ignore[attr-defined]

    def seek(self, *args: object) -> int:
        return self.source.seek(*args)  # type: ignore[attr-defined]

    def tell(self) -> int:
        return self.source.tell()  # type: ignore[attr-defined]

    def read(self, *args: object) -> bytes:
        self.read_count += 1
        return self.source.read(*args)  # type: ignore[attr-defined]


def test_official_glenn_archive_causal_history_units_and_precontact_refusal() -> None:
    value = os.environ.get("GLENN_PUBLIC_ARCHIVE")
    if value is None:
        pytest.skip("set GLENN_PUBLIC_ARCHIVE to a lawfully held Zenodo 17163053 ZIP")
    path = Path(value)
    if not path.is_file():
        pytest.fail("GLENN_PUBLIC_ARCHIVE does not name a regular file")

    with path.open("rb") as source:
        counted = _CountingStream(source)
        with pytest.raises(GlennAdapterError, match="archive size differs"):
            inspect_glenn_archive(
                GlennArchiveInput(counted, "glenn-public-reference", "glenn-reference-guard"),
                profile=replace(
                    GLENN_2026_PROFILE,
                    expected_archive_size_bytes=GLENN_2026_PROFILE.expected_archive_size_bytes - 1,
                ),
            )
        assert counted.read_count == 0

    with path.open("rb") as source:
        result = transform_glenn_archive(
            GlennArchiveInput(source, "glenn-public-reference", "glenn-reference-guard"),
            profile=GLENN_2026_PROFILE,
        )
    rows = result.table.to_pydict()
    bursts = rows["burst"]
    assert len(bursts) > 50
    assert len(set(bursts)) == len(bursts)
    assert rows["acquisition_order"] == list(range(1, len(bursts) + 1))
    assert all(abs(rows[name][0]) < 1e-12 for name in ACTION_COLUMNS)
    assert rows["prior_fitness_missing"] == [True, *([False] * (len(bursts) - 1))]
    np.testing.assert_allclose(
        rows["prior_fitness"][1:], rows["fitness"][:-1], rtol=0, atol=1e-12
    )

    member = GLENN_2026_PROFILE.member_for(GlennMemberRole.FOCAL_RADIUS_ARRAY)
    with ZipFile(path) as archive:
        native_focal = np.load(
            BytesIO(archive.read(member.relative_locator)), allow_pickle=False
        )
    np.testing.assert_allclose(
        rows["focal_r50_um"],
        native_focal * 1_000,  # deposited figure notebook: spot_radius_burst * 1e3
        rtol=0,
        atol=1e-10,
    )
