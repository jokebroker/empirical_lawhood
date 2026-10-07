"""Pure Six-matrix response artifact-profile compilation for outer infrastructure composition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from .contracts import SixMatrixResponseArtifactProfileConfig


MATRIX_RESPONSE_PREPARATION_SCHEDULING_MEASUREMENT_HDF5_SCHEMA = (
    'empirical-lawhood/simulators/six-matrix-response/preparation-scheduling-trajectory-measurement-hdf5'
)
MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_TRAJECTORY_DECODED_DATASET_BYTES = 5_339_160


@dataclass(frozen=True, slots=True)
class SixMatrixResponseHDF5SemanticBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-hdf5-semantic-binding'
    VERSION: ClassVar[str] = "1.0.0"

    binding_id: str
    dataset_path: str
    native_unit: str
    frame_id: str
    clock_id: str
    key_role_id: str

    def __post_init__(self) -> None:
        for name in ("binding_id", "frame_id", "clock_id", "key_role_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        path = PurePosixPath(self.dataset_path)
        if not self.dataset_path.startswith("/") or str(path) != self.dataset_path:
            raise ValueError("Six-matrix response HDF5 semantic binding path is not normalized")
        if not self.native_unit:
            raise ValueError("Six-matrix response HDF5 semantic binding requires a native unit")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseCompiledHDF5Inventory(CanonicalRecord):
    """Infrastructure-neutral exact inventory and deterministic writer contract."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-compiled-hdf5-inventory'
    VERSION: ClassVar[str] = "1.0.0"

    inventory_id: str
    profile: SixMatrixResponseArtifactProfileConfig
    profile_identity: ObjectIdentity
    group_paths: tuple[str, ...]
    semantic_bindings: tuple[SixMatrixResponseHDF5SemanticBinding, ...]
    maximum_name_bytes: int
    maximum_attribute_bytes: int
    relative_output_root_rule_id: str
    filename_suffix: str
    catalog_loss_reconstructible: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("inventory_id", "relative_output_root_rule_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.profile_identity != ObjectIdentity.from_record(
            self.profile.config_id,
            self.profile,
        ):
            raise ValueError("Six-matrix response compiled HDF5 inventory changes its exact profile")
        require_sorted_unique_strings(
            self.group_paths,
            field_name="group_paths",
            allow_empty=False,
        )
        if not self.group_paths or self.group_paths[0] != "/":
            raise ValueError("Six-matrix response HDF5 group inventory must begin at root")
        require_sorted_unique_ids(
            self.semantic_bindings,
            attribute="binding_id",
            field_name="semantic_bindings",
        )
        dataset_paths = tuple(value.path for value in self.profile.datasets)
        if tuple(value.dataset_path for value in self.semantic_bindings) != dataset_paths:
            raise ValueError("Six-matrix response HDF5 semantic bindings differ from its datasets")
        expected_groups = {"/"}
        for dataset_path in dataset_paths:
            parent = PurePosixPath(dataset_path).parent
            while str(parent) != ".":
                expected_groups.add(str(parent))
                if str(parent) == "/":
                    break
                parent = parent.parent
        if self.group_paths != tuple(sorted(expected_groups)):
            raise ValueError("Six-matrix response HDF5 group inventory differs from dataset ancestry")
        if min(self.maximum_name_bytes, self.maximum_attribute_bytes) < 1:
            raise ValueError("Six-matrix response HDF5 decoded metadata bounds must be positive")
        if self.filename_suffix != ".h5":
            raise ValueError("Six-matrix response HDF5 output uses another filename suffix")
        if not self.catalog_loss_reconstructible or self.grants_authority:
            raise ValueError("Six-matrix response HDF5 inventory must reconstruct without granting authority")


@dataclass(frozen=True, slots=True)
class SixMatrixResponseArtifactOutputReservation(CanonicalRecord):
    """One exact issue-time output path; never a dynamic writer choice."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-artifact-output-reservation'
    VERSION: ClassVar[str] = "1.0.0"

    reservation_id: str
    issued_campaign_id: str
    task_id: str
    panel_id: str
    shard_id: str
    artifact_profile_identity: ObjectIdentity
    relative_path: str
    maximum_shard_bytes: int
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "reservation_id",
            "issued_campaign_id",
            "task_id",
            "panel_id",
            "shard_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        expected = (
            "experiments/Six-matrix response-GIF/"
            f"{self.issued_campaign_id}/{self.task_id}/{self.panel_id}/{self.shard_id}.h5"
        )
        path = PurePosixPath(self.relative_path)
        if (
            str(path) != self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or self.relative_path != expected
        ):
            raise ValueError("Six-matrix response artifact output differs from its issued reservation")
        if self.maximum_shard_bytes < 1 or self.maximum_shard_bytes > 536_870_912:
            raise ValueError("Six-matrix response artifact reservation exceeds the vfat shard ceiling")
        if self.grants_authority:
            raise ValueError("Six-matrix response artifact output reservation cannot grant authority")


@dataclass(frozen=True, slots=True)
class SixMatrixResponsePreparationWindowSchedulingTrajectoryArtifactReservation(CanonicalRecord):
    """One exact issue-time preparation scheduling trajectory payload path and byte ceiling."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-trajectory-artifact-reservation'

    reservation_id: str
    attempt_id: str
    task_id: str
    root_block_id: str
    output_id: str
    payload_schema: str
    relative_path: str
    maximum_bytes: int
    maximum_decoded_bytes: int
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in (
            "reservation_id",
            "attempt_id",
            "task_id",
            "root_block_id",
            "output_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        expected = (
            f"experiments/matrix-preparation-scheduling/{self.attempt_id}/"
            f"{self.task_id}/{self.root_block_id}.h5"
        )
        path = PurePosixPath(self.relative_path)
        if (
            str(path) != self.relative_path
            or path.is_absolute()
            or ".." in path.parts
            or self.relative_path != expected
        ):
            raise ValueError("six-matrix preparation scheduling artifact differs from its tranche reservation")
        if self.payload_schema != MATRIX_RESPONSE_PREPARATION_SCHEDULING_MEASUREMENT_HDF5_SCHEMA:
            raise ValueError("six-matrix preparation scheduling tranche reservation has another payload schema")
        if not 1 <= self.maximum_bytes < 4_000_000_000:
            raise ValueError("six-matrix preparation scheduling tranche reservation exceeds its shard ceiling")
        if not 1 <= self.maximum_decoded_bytes < 4_000_000_000:
            raise ValueError("six-matrix preparation scheduling tranche decoded ceiling is invalid")
        if self.grants_authority:
            raise ValueError("six-matrix preparation scheduling tranche reservation cannot grant authority")


def reserve_six_matrix_response_preparation_window_scheduling_trajectory_artifact(
    *,
    attempt_id: str,
    task_id: str,
    root_block_id: str,
    output_id: str,
    maximum_bytes: int,
    maximum_decoded_bytes: int,
) -> SixMatrixResponsePreparationWindowSchedulingTrajectoryArtifactReservation:
    """Reserve the sole typed full-path payload for one preparation scheduling trajectory."""

    return SixMatrixResponsePreparationWindowSchedulingTrajectoryArtifactReservation(
        reservation_id=f"matrix-preparation-scheduling.reservation.{attempt_id}.{task_id}.{root_block_id}",
        attempt_id=attempt_id,
        task_id=task_id,
        root_block_id=root_block_id,
        output_id=output_id,
        payload_schema=MATRIX_RESPONSE_PREPARATION_SCHEDULING_MEASUREMENT_HDF5_SCHEMA,
        relative_path=(
            f"experiments/matrix-preparation-scheduling/{attempt_id}/{task_id}/{root_block_id}.h5"
        ),
        maximum_bytes=maximum_bytes,
        maximum_decoded_bytes=maximum_decoded_bytes,
        grants_authority=False,
    )


def reserve_six_matrix_response_hdf5_output(
    *,
    issued_campaign_id: str,
    task_id: str,
    panel_id: str,
    shard_id: str,
    profile: SixMatrixResponseArtifactProfileConfig,
) -> SixMatrixResponseArtifactOutputReservation:
    profile_identity = ObjectIdentity.from_record(profile.config_id, profile)
    return SixMatrixResponseArtifactOutputReservation(
        reservation_id=(
            f"six-matrix-response.artifact-reservation.{issued_campaign_id}.{task_id}.{panel_id}.{shard_id}"
        ),
        issued_campaign_id=issued_campaign_id,
        task_id=task_id,
        panel_id=panel_id,
        shard_id=shard_id,
        artifact_profile_identity=profile_identity,
        relative_path=(
            f"experiments/Six-matrix response-GIF/{issued_campaign_id}/{task_id}/{panel_id}/{shard_id}.h5"
        ),
        maximum_shard_bytes=profile.maximum_shard_bytes,
        grants_authority=False,
    )


def compile_six_matrix_response_hdf5_inventory(
    profile: SixMatrixResponseArtifactProfileConfig,
) -> SixMatrixResponseCompiledHDF5Inventory:
    groups = {"/"}
    semantic_bindings = []
    for dataset in profile.datasets:
        parent = PurePosixPath(dataset.path).parent
        while str(parent) != ".":
            groups.add(str(parent))
            if str(parent) == "/":
                break
            parent = parent.parent
        semantic_bindings.append(
            SixMatrixResponseHDF5SemanticBinding(
                binding_id=f"six-matrix-response.hdf5-semantic.{dataset.dataset_id}",
                dataset_path=dataset.path,
                native_unit=dataset.native_unit,
                frame_id=dataset.frame_id,
                clock_id=dataset.clock_id,
                key_role_id=dataset.key_role_id,
            )
        )
    return SixMatrixResponseCompiledHDF5Inventory(
        inventory_id=f"six-matrix-response.compiled-hdf5-inventory.{profile.config_id}",
        profile=profile,
        profile_identity=ObjectIdentity.from_record(profile.config_id, profile),
        group_paths=tuple(sorted(groups)),
        semantic_bindings=tuple(sorted(semantic_bindings, key=lambda value: value.dataset_path)),
        maximum_name_bytes=65_536,
        maximum_attribute_bytes=65_536,
        relative_output_root_rule_id="six-matrix-response.output.issued-task-panel-contained",
        filename_suffix=".h5",
        catalog_loss_reconstructible=True,
        grants_authority=False,
    )


__all__ = [
    "MATRIX_RESPONSE_PREPARATION_SCHEDULING_MAXIMUM_TRAJECTORY_DECODED_DATASET_BYTES",
    "MATRIX_RESPONSE_PREPARATION_SCHEDULING_MEASUREMENT_HDF5_SCHEMA",
    'SixMatrixResponseArtifactOutputReservation',
    'SixMatrixResponseCompiledHDF5Inventory',
    'SixMatrixResponseHDF5SemanticBinding',
    'SixMatrixResponsePreparationWindowSchedulingTrajectoryArtifactReservation',
    'compile_six_matrix_response_hdf5_inventory',
    'reserve_six_matrix_response_hdf5_output',
    'reserve_six_matrix_response_preparation_window_scheduling_trajectory_artifact',
]
