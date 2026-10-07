'Target-owned read-only preflight for separately custodied material control calibration outputs.\n\nThis checks an accepted native input envelope. It does not execute QE/EPW,\nqualify a control, reduce a material response, or freeze material control science.\n'

from __future__ import annotations

import tarfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import ClassVar

import h5py
import numpy as np

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .material_control_raw import RAW_ARCHIVE_DATASET, RAW_ARCHIVE_SCHEMA, RAW_ARCHIVE_VERSION
from .material_control_solver import MATERIAL_CONTROL_WORKFLOW_PROFILES, WorkflowKind, WorkflowSource, resolve_material_control_workflow_profile

MAXIMUM_RAW_ARCHIVE_BYTES = 64 * 1024**3
MAXIMUM_TAR_ARCHIVE_BYTES = 32 * 1024**3
CHUNK_BYTES = 8 * 1024**2


@dataclass(frozen=True, slots=True)
class MaterialControlInputPreflight(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-control-input-preflight'

    config_id: str
    workflow_profile_ids: tuple[str, ...]
    archive_relative_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        if not self.config_id.startswith("empirical-lawhood-"):
            raise ValueError('material control input config requires a target-owned identity')
        required = tuple(
            profile.profile_id
            for profile in MATERIAL_CONTROL_WORKFLOW_PROFILES
            if profile.source is WorkflowSource.SSSP_CONTROL
        )
        if self.workflow_profile_ids != required:
            raise ValueError(
                'material control calibration input requires all ten fixed control/view profiles'
            )
        if len(self.archive_relative_paths) != len(required) or len(
            set(self.archive_relative_paths)
        ) != len(required):
            raise ValueError(
                'material control calibration raw archive paths must be one-to-one with profiles'
            )
        for profile_id, value in zip(
            required, self.archive_relative_paths, strict=True
        ):
            path = PurePosixPath(value)
            if (
                not value
                or path.is_absolute()
                or "\\" in value
                or any(part in {"", ".", ".."} for part in value.split("/"))
                or path.suffix != ".h5"
            ):
                raise ValueError(
                    f'material control raw archive path is not a bounded relative HDF5 path: {profile_id}'
                )


def _attribute(handle: h5py.File, key: str) -> str:
    value = handle.attrs.get(key)
    if not isinstance(value, (bytes, np.bytes_)):
        raise TypeError(f'material control raw archive lacks byte attribute {key}')
    return bytes(value).decode("utf-8")


def inspect_material_control_tar_input(path: Path, *, profile_id: str) -> dict[str, object]:
    'Bound a researcher-supplied native tar before wrapping it as material control raw HDF5.'

    profile = resolve_material_control_workflow_profile(profile_id)
    if profile.source is not WorkflowSource.SSSP_CONTROL:
        raise ValueError('material control calibration input cannot substitute a tutorial profile')
    if not path.is_file() or path.is_symlink():
        raise ValueError('material control raw tar is absent or is not a regular file')
    size = path.stat().st_size
    if size <= 0 or size > MAXIMUM_TAR_ARCHIVE_BYTES:
        raise ValueError('material control raw tar exceeds its source output budget')
    names: set[str] = set()
    expanded_bytes = 0
    with tarfile.open(path, mode="r|gz") as archive:
        for member in archive:
            member_path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or member_path.is_absolute()
                or ".." in member_path.parts
                or member.name in names
                or member.size > 2 * 1024**3
            ):
                raise ValueError('material control raw tar contains an unsafe or duplicate member')
            names.add(member.name)
            expanded_bytes += member.size
            if len(names) > 1_000_000 or expanded_bytes > MAXIMUM_RAW_ARCHIVE_BYTES:
                raise ValueError('material control raw tar expanded inventory exceeds its bound')
    if not names:
        raise ValueError('material control raw tar has no native output members')
    if profile.kind is WorkflowKind.POSITIVE:
        for suffix in (f'{profile.prefix}_hr.dat', f'{profile.prefix}_r.dat'):
            if sum(name.endswith(suffix) for name in names) != 1:
                raise ValueError(
                    f'material control positive control requires exactly one {suffix} member'
                )
    digest = sha256()
    with path.open("rb") as stream:
        while block := stream.read(CHUNK_BYTES):
            digest.update(block)
    return {
        "profile_id": profile_id,
        "logical_archive_sha256": digest.hexdigest(),
        "logical_archive_bytes": size,
        "regular_member_count": len(names),
        "expanded_member_bytes": expanded_bytes,
        "material_operands_qualified": False,
    }


def inspect_material_control_raw_input(path: Path, *, profile_id: str) -> dict[str, object]:
    'Check the native material control HDF5 envelope and its logical tar digest.'

    profile = resolve_material_control_workflow_profile(profile_id)
    if profile.source is not WorkflowSource.SSSP_CONTROL:
        raise ValueError('material control calibration input cannot substitute a tutorial profile')
    if not path.is_file() or path.is_symlink():
        raise ValueError(f'material control raw workflow is absent or is not a regular file: {path}')
    if path.stat().st_size <= 0 or path.stat().st_size > MAXIMUM_RAW_ARCHIVE_BYTES:
        raise ValueError('material control raw HDF5 size exceeds the source execution envelope')
    with h5py.File(path, "r") as handle:
        if (
            _attribute(handle, "schema") != RAW_ARCHIVE_SCHEMA
            or _attribute(handle, "version") != RAW_ARCHIVE_VERSION
        ):
            raise ValueError('material control raw HDF5 schema/version differs')
        if _attribute(handle, "profile_id") != profile_id:
            raise ValueError(
                'material control raw HDF5 profile differs from the selected control/view'
            )
        expected_digest = _attribute(handle, "archive_sha256")
        if len(expected_digest) != 64 or any(
            char not in "0123456789abcdef" for char in expected_digest
        ):
            raise ValueError('material control raw HDF5 logical digest is malformed')
        if set(handle.keys()) != {RAW_ARCHIVE_DATASET}:
            raise ValueError('material control raw HDF5 dataset inventory differs')
        dataset = handle[RAW_ARCHIVE_DATASET]
        if (
            not isinstance(dataset, h5py.Dataset)
            or dataset.dtype != np.dtype("u1")
            or len(dataset.shape) != 1
        ):
            raise ValueError('material control raw HDF5 byte stream differs')
        count = dataset.shape[0]
        if count <= 0 or count > MAXIMUM_RAW_ARCHIVE_BYTES:
            raise ValueError('material control raw logical archive exceeds its bound')
        observed = sha256()
        for start in range(0, count, CHUNK_BYTES):
            observed.update(dataset[start : min(count, start + CHUNK_BYTES)].tobytes())
        if observed.hexdigest() != expected_digest:
            raise ValueError('material control raw logical archive digest differs')
    return {
        "profile_id": profile_id,
        "structure_id": profile.structure_id,
        "view_id": profile.view_id,
        "control_class": profile.control_class.value,
        "workflow_kind": profile.kind.value,
        "logical_archive_sha256": expected_digest,
        "logical_archive_bytes": count,
    }


def inspect_material_control_input_roster(
    config: MaterialControlInputPreflight, *, source_root: Path
) -> dict[str, object]:
    """Require ten separately supplied native outputs before any material claim."""

    if (
        not source_root.is_absolute()
        or not source_root.is_dir()
        or source_root.is_symlink()
    ):
        raise ValueError('material control held source root is absent, relative or a symlink')
    root = source_root.resolve()
    paths: list[Path] = []
    for relative in config.archive_relative_paths:
        path = root.joinpath(*PurePosixPath(relative).parts)
        if (
            not path.resolve().is_relative_to(root)
            or not path.is_file()
            or path.is_symlink()
        ):
            raise ValueError(
                f'material control held raw workflow is absent or escapes its source root: {relative}'
            )
        paths.append(path)
    profiles = tuple(
        inspect_material_control_raw_input(path, profile_id=profile_id)
        for profile_id, path in zip(config.workflow_profile_ids, paths, strict=True)
    )
    return {
        "config_id": config.config_id,
        "independent_control_structures": len(
            {row["structure_id"] for row in profiles}
        ),
        "nested_numerical_views": len(profiles),
        "input_envelopes_checked": len(profiles),
        "material_operands_qualified": False,
        "science_frozen": False,
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
        "profiles": profiles,
    }


__all__ = [
    'MaterialControlInputPreflight',
    'inspect_material_control_input_roster',
    'inspect_material_control_raw_input',
    'inspect_material_control_tar_input',
]
