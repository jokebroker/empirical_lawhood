"""Strict Glenn selector and optional authentic one-CLI input transform."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from empirical_lawhood.adapters.physical.glenn.quickstart import GlennInputQuickstart, run_local_import_check
from empirical_lawhood.api.codecs import load_registered_authoring

STARTER = Path("experiments/laser-archive-inspection/config.json")


def _config() -> GlennInputQuickstart:
    return load_registered_authoring(
        STARTER,
        root_schemas={GlennInputQuickstart.SCHEMA: GlennInputQuickstart},
        maximum_bytes=16 * 1024,
    )


def test_shipped_glenn_selection_refuses_before_archive_contact(tmp_path: Path) -> None:
    request = _config()
    assert request.archive_relative_path == "GDGlenn_PRR_2026.zip"
    with pytest.raises(ValueError, match="held ZIP is absent"):
        run_local_import_check(request, source_root=tmp_path)
    with pytest.raises(ValueError, match="cannot be marked prospective"):
        replace(request, source_role="PROSPECTIVE_OUTCOME_BLIND")
    with pytest.raises(ValueError, match="bounded relative ZIP"):
        replace(request, archive_relative_path="../other.zip")


@pytest.mark.held
def test_authentic_glenn_release_through_target_selector_when_supplied() -> None:
    value = os.environ.get("GLENN_PUBLIC_ARCHIVE")
    if value is None:
        pytest.skip("set GLENN_PUBLIC_ARCHIVE to the lawful public release ZIP")
    archive = Path(value)
    assert archive.is_file() and not archive.is_symlink()
    request = replace(_config(), archive_relative_path=archive.name)
    report = run_local_import_check(request, source_root=archive.parent.resolve())
    assert report["independent_bursts"] == 101
    assert report["column_count"] == 20 and report["selected_members"] == 4
    assert report["first_prior_fitness_missing"] is True
    assert report["second_prior_fitness"] == report["first_fitness"]
    assert set(report["first_action_values"].values()) == {0.0}
    assert report["focal_native_to_um"] == 1000.0
    assert report["action_stage_journal_available"] is False
    assert report["storage_custody_qualified"] is False
    assert report["candidate_compiled"] is False
