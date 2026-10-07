"""Version-bound native metadata for the retained Gym-TORAX Gym--TORAX fields.

TORAX 1.4.2 documents units in its code but does not emit them as xarray
variable attributes, and GymTORAX 1.1.1 discards dimensions while building its
observation dictionary.  This module makes that missing contract explicit.  It
does not infer units or frames from observed values.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
from importlib.util import find_spec
from pathlib import Path
import re
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_nonempty,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)


GYM_TORAX_FIELD_METADATA_MANIFEST_ID = 'field-metadata.tokamak-control.gymtorax-1-1-1.torax-1-4-2'


@dataclass(frozen=True, slots=True)
class GymToraxMetadataSourceFile(CanonicalRecord):
    """One installed-distribution source file that defines field semantics."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-metadata-source-file'

    source_file_id: str
    distribution: str
    distribution_version: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.source_file_id, field_name="source_file_id")
        validate_nonempty(self.distribution, field_name="distribution")
        validate_nonempty(self.distribution_version, field_name="distribution_version")
        validate_relative_locator(self.relative_path)
        validate_sha256(self.sha256, field_name="sha256")


@dataclass(frozen=True, slots=True)
class GymToraxNativeFieldMetadata(CanonicalRecord):
    """Code-owned semantics for one retained raw field."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-field-metadata'

    field_metadata_id: str
    source_category: str
    native_field_id: str
    native_unit: str
    native_dimension_ids: tuple[str, ...]
    native_frame_id: str
    semantic_source_file_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.field_metadata_id, field_name="field_metadata_id")
        if self.source_category not in {"profiles", "scalars", "numerics"}:
            raise ValueError("unsupported Gym--TORAX source category")
        validate_nonempty(self.native_field_id, field_name="native_field_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.native_unit == "source-unit-unspecified":
            raise ValueError("unspecified units are prohibited by the Gym-TORAX metadata barrier")
        if not self.native_dimension_ids or any(not value for value in self.native_dimension_ids):
            raise ValueError("native dimensions must be explicit and ordered")
        if len(set(self.native_dimension_ids)) != len(self.native_dimension_ids):
            raise ValueError("native dimensions must be unique")
        validate_stable_id(self.native_frame_id, field_name="native_frame_id")
        if not self.semantic_source_file_ids:
            raise ValueError("field metadata requires an installed-source definition")
        if self.semantic_source_file_ids != tuple(sorted(set(self.semantic_source_file_ids))):
            raise ValueError("semantic source-file IDs must be sorted and unique")


@dataclass(frozen=True, slots=True)
class GymToraxFieldMetadataManifest(CanonicalRecord):
    """Closed retained-field contract for the exact installed source versions."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-field-metadata-manifest'

    manifest_id: str
    gymtorax_version: str
    torax_version: str
    xarray_version: str
    source_files: tuple[GymToraxMetadataSourceFile, ...]
    fields: tuple[GymToraxNativeFieldMetadata, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        if self.manifest_id != GYM_TORAX_FIELD_METADATA_MANIFEST_ID:
            raise ValueError("unexpected Gym--TORAX field metadata manifest identity")
        if (
            self.gymtorax_version != "1.1.1"
            or self.torax_version != "1.4.2"
            or self.xarray_version != "2026.7.0"
        ):
            raise ValueError("field metadata manifest binds another source denominator")
        require_sorted_unique_ids(
            self.source_files,
            attribute="source_file_id",
            field_name="source_files",
        )
        field_ids = tuple(value.field_metadata_id for value in self.fields)
        if len(set(field_ids)) != len(field_ids):
            raise ValueError("field metadata IDs must be unique")
        keys = tuple((value.source_category, value.native_field_id) for value in self.fields)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("field metadata keys must be sorted and unique")
        source_ids = {value.source_file_id for value in self.source_files}
        if any(
            source_id not in source_ids
            for field in self.fields
            for source_id in field.semantic_source_file_ids
        ):
            raise ValueError("field metadata refers to a foreign source file")
        if keys != _RETAINED_FIELD_KEYS:
            raise ValueError("field metadata manifest is not the exact retained-field closure")

    def by_key(self) -> dict[tuple[str, str], GymToraxNativeFieldMetadata]:
        return {(value.source_category, value.native_field_id): value for value in self.fields}


_SOURCE_FILES = (
    (
        "metadata-source.gymtorax-1-1-1.observation-handler",
        "gymtorax",
        "1.1.1",
        "observation_handler.py",
        "9dcf3491b343bd027acc7c044962c52c636d5c6c0f345df913936d314ed450e0",
    ),
    (
        "metadata-source.gymtorax-1-1-1.torax-app",
        "gymtorax",
        "1.1.1",
        "torax_wrapper/torax_app.py",
        "96f0c4b2b1b075a16ec41de8dcd7456e1884df34d935ea81f5113571272fd1b6",
    ),
    (
        "metadata-source.torax-1-4-2.output",
        "torax",
        "1.4.2",
        "_src/output_tools/output.py",
        "108bc9830cfbf6ce92c2b4f906b954ebc53762e50f74b550a60f1bd879ccbcc1",
    ),
    (
        "metadata-source.torax-1-4-2.post-processing",
        "torax",
        "1.4.2",
        "_src/output_tools/post_processing.py",
        "347f0a8045870c94e32c7be9bde04905e03fe97cea8b7ab93216a3216b5f3190",
    ),
    (
        "metadata-source.torax-1-4-2.state",
        "torax",
        "1.4.2",
        "_src/state.py",
        "bf18418febc25166c656c217e437baf4b1619f2141c04f5c70e0bb779baf95f6",
    ),
)

_PROFILE_SOURCE_IDS = (
    "metadata-source.gymtorax-1-1-1.observation-handler",
    "metadata-source.torax-1-4-2.output",
    "metadata-source.torax-1-4-2.state",
)
_SCALAR_SOURCE_IDS = (
    "metadata-source.gymtorax-1-1-1.observation-handler",
    "metadata-source.torax-1-4-2.output",
    "metadata-source.torax-1-4-2.post-processing",
)
_NUMERICS_SOURCE_IDS = (
    "metadata-source.torax-1-4-2.output",
    "metadata-source.torax-1-4-2.state",
)

_FIELD_ROWS = (
    (
        "numerics",
        "inner_solver_iterations",
        "1",
        ("value",),
        "numerics-scalar",
        _NUMERICS_SOURCE_IDS,
    ),
    (
        "numerics",
        "outer_solver_iterations",
        "1",
        ("value",),
        "numerics-scalar",
        _NUMERICS_SOURCE_IDS,
    ),
    ("numerics", "sawtooth_crash", "1", ("value",), "numerics-scalar", _NUMERICS_SOURCE_IDS),
    ("numerics", "solver_error_state", "1", ("value",), "numerics-scalar", _NUMERICS_SOURCE_IDS),
    ("profiles", "T_e", "keV", ("rho_norm",), "rho-norm", _PROFILE_SOURCE_IDS),
    ("profiles", "T_i", "keV", ("rho_norm",), "rho-norm", _PROFILE_SOURCE_IDS),
    ("profiles", "n_e", "m^-3", ("rho_norm",), "rho-norm", _PROFILE_SOURCE_IDS),
    ("profiles", "psi", "Wb", ("rho_norm",), "rho-norm", _PROFILE_SOURCE_IDS),
    ("profiles", "q", "1", ("rho_face_norm",), "rho-face-norm", _PROFILE_SOURCE_IDS),
    ("scalars", "H98", "1", ("value",), "scalar", _SCALAR_SOURCE_IDS),
    ("scalars", "P_heat_total", "W", ("value",), "scalar", _SCALAR_SOURCE_IDS),
    ("scalars", "P_radiation_e", "W", ("value",), "scalar", _SCALAR_SOURCE_IDS),
    ("scalars", "Q_fusion", "1", ("value",), "scalar", _SCALAR_SOURCE_IDS),
    ("scalars", "beta_N", "1", ("value",), "scalar", _SCALAR_SOURCE_IDS),
    ("scalars", "fgw_n_e_volume_avg", "1", ("value",), "scalar", _SCALAR_SOURCE_IDS),
    ("scalars", "q95", "1", ("value",), "scalar", _SCALAR_SOURCE_IDS),
    ("scalars", "q_min", "1", ("value",), "scalar", _SCALAR_SOURCE_IDS),
)

_RETAINED_FIELD_KEYS = tuple(sorted((row[0], row[1]) for row in _FIELD_ROWS))


def build_gym_torax_field_metadata_manifest() -> GymToraxFieldMetadataManifest:
    """Build the frozen 17-field metadata contract without inspecting outcomes."""

    source_files = tuple(
        GymToraxMetadataSourceFile(
            source_file_id=source_file_id,
            distribution=distribution,
            distribution_version=version,
            relative_path=relative_path,
            sha256=digest,
        )
        for source_file_id, distribution, version, relative_path, digest in _SOURCE_FILES
    )
    fields = tuple(
        sorted(
            (
                GymToraxNativeFieldMetadata(
                    field_metadata_id=(
                        f'field-metadata.tokamak-control.{category}.'
                        f"{re.sub(r'[^a-z0-9]+', '-', field_id.lower()).strip('-')}"
                    ),
                    source_category=category,
                    native_field_id=field_id,
                    native_unit=unit,
                    native_dimension_ids=dimensions,
                    native_frame_id=f'frame.tokamak-control.{frame_slug}',
                    semantic_source_file_ids=tuple(sorted(source_ids)),
                )
                for category, field_id, unit, dimensions, frame_slug, source_ids in _FIELD_ROWS
            ),
            key=lambda value: (value.source_category, value.native_field_id),
        )
    )
    return GymToraxFieldMetadataManifest(
        manifest_id=GYM_TORAX_FIELD_METADATA_MANIFEST_ID,
        gymtorax_version="1.1.1",
        torax_version="1.4.2",
        xarray_version="2026.7.0",
        source_files=source_files,
        fields=fields,
    )


def verify_installed_gym_torax_metadata_sources(
    manifest: GymToraxFieldMetadataManifest,
) -> None:
    """Fail closed if installed source versions or defining bytes have drifted."""

    for distribution, expected in (
        ("gymtorax", manifest.gymtorax_version),
        ("torax", manifest.torax_version),
        ("xarray", manifest.xarray_version),
    ):
        if importlib_metadata.version(distribution) != expected:
            raise RuntimeError(f"GYM_TORAX_METADATA_DISTRIBUTION_DRIFT:{distribution}")
    for source_file in manifest.source_files:
        spec = find_spec(source_file.distribution)
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(f"GYM_TORAX_METADATA_DISTRIBUTION_UNAVAILABLE:{source_file.distribution}")
        package_root = Path(next(iter(spec.submodule_search_locations))).resolve(strict=True)
        source_path = (package_root / source_file.relative_path).resolve(strict=True)
        if not source_path.is_relative_to(package_root):
            raise RuntimeError("GYM_TORAX_METADATA_SOURCE_PATH_ESCAPE")
        digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if digest != source_file.sha256:
            raise RuntimeError(f"GYM_TORAX_METADATA_SOURCE_DRIFT:{source_file.source_file_id}")


__all__ = [
    "GYM_TORAX_FIELD_METADATA_MANIFEST_ID",
    'GymToraxFieldMetadataManifest',
    'GymToraxMetadataSourceFile',
    'GymToraxNativeFieldMetadata',
    'build_gym_torax_field_metadata_manifest',
    'verify_installed_gym_torax_metadata_sources',
]
