"""Read-only, unqualified local inspection of the exact Glenn public release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .contracts import ACTION_COLUMNS, GLENN_2026_PROFILE, GlennArchiveInput
from .transform import transform_glenn_archive


@dataclass(frozen=True, slots=True)
class GlennInputQuickstart(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/glenn/glenn-input-quickstart'

    config_id: str
    profile_id: str
    archive_relative_path: str
    source_role: str

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        if not self.config_id.startswith("empirical-lawhood-"):
            raise ValueError("Glenn quick start needs a target-owned config identity")
        if self.profile_id != GLENN_2026_PROFILE.profile_id:
            raise ValueError("Glenn quick start selects another source release")
        if self.source_role != "RETROSPECTIVE_OUTCOME_VISIBLE":
            raise ValueError("Glenn historical outcomes cannot be marked prospective")
        path = PurePosixPath(self.archive_relative_path)
        if (
            not self.archive_relative_path
            or path.is_absolute()
            or "\\" in self.archive_relative_path
            or any(
                part in {"", ".", ".."}
                for part in self.archive_relative_path.split("/")
            )
            or path.suffix != ".zip"
        ):
            raise ValueError("Glenn source path must be a bounded relative ZIP path")


def run_local_import_check(
    config: GlennInputQuickstart, *, source_root: Path
) -> dict[str, object]:
    """Exercise the retained source decoder without claiming storage custody."""

    if (
        not source_root.is_absolute()
        or not source_root.is_dir()
        or source_root.is_symlink()
    ):
        raise ValueError("Glenn held source root is absent, relative or a symlink")
    root = source_root.resolve()
    path = root.joinpath(*PurePosixPath(config.archive_relative_path).parts)
    if (
        not path.resolve().is_relative_to(root)
        or not path.is_file()
        or path.is_symlink()
    ):
        raise ValueError("Glenn held ZIP is absent or escapes its source root")
    with path.open("rb") as stream:
        result = transform_glenn_archive(
            GlennArchiveInput(
                stream,
                "source.empirical-lawhood-glenn-local-inspection",
                "inspection.unqualified-local-no-custody",
            ),
            profile=GLENN_2026_PROFILE,
        )
    rows = result.table.to_pydict()
    return {
        "config_id": config.config_id,
        "source_role": config.source_role,
        "archive_sha256": result.archive_audit.archive_sha256,
        "selected_members": len(result.archive_audit.selected_members),
        "logical_table_sha256": result.logical_sha256,
        "independent_bursts": result.table.num_rows,
        "column_count": result.table.num_columns,
        "nested_shots_count_as_units": False,
        "first_action_values": {name: rows[name][0] for name in ACTION_COLUMNS},
        "first_prior_fitness_missing": rows["prior_fitness_missing"][0],
        "second_prior_fitness": rows["prior_fitness"][1],
        "first_fitness": rows["fitness"][0],
        "focal_native_to_um": result.focal_native_to_um,
        "action_stage_journal_available": False,
        "focal_is_post_command_mediator": result.focal_is_post_command_mediator,
        "source_exceptions": tuple(code.value for code in result.exception_codes),
        "storage_custody_qualified": False,
        "candidate_compiled": False,
        "campaign_issued": False,
    }


def preview_public_release(config: GlennInputQuickstart) -> dict[str, object]:
    """Decode the exact pinned public source without network or outcome contact."""

    return {
        "config_id": config.config_id,
        "source_role": config.source_role,
        "upstream_record": "https://zenodo.org/records/17163053",
        "upstream_download": "https://zenodo.org/records/17163053/files/GDGlenn_PRR_2026.zip?download=1",
        "archive_relative_path": config.archive_relative_path,
        "expected_archive_size_bytes": GLENN_2026_PROFILE.expected_archive_size_bytes,
        "expected_archive_sha256": GLENN_2026_PROFILE.expected_archive_sha256,
        "selected_member_roles": tuple(
            member.role.value for member in GLENN_2026_PROFILE.members
        ),
        "profile_id": config.profile_id,
        "independent_unit": "logged-burst",
        "network_contacted": False,
        "outcome_contacted": False,
        "storage_custody_qualified": False,
    }


__all__ = ['GlennInputQuickstart', "preview_public_release", "run_local_import_check"]
