"""Outcome-visible imported evidence identities and independently replayed arithmetic."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar
import re

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_relative_locator,
    validate_sha256,
)

from .contracts import CONTEXTS, PARENTS, RetainedPreparationPrefix, PreparationPrefixBundleRef, preparation_roots


@dataclass(frozen=True, slots=True)
class PreparationImportedFile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-imported-file'
    relative_path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        validate_relative_locator(self.relative_path)
        validate_sha256(self.sha256, field_name="sha256")
        if type(self.size_bytes) is not int or not 0 < self.size_bytes <= 64 * 1024**2:
            raise ValueError("imported file exceeds its finite bound")


@dataclass(frozen=True, slots=True)
class PreparationImportedParent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-imported-parent'
    parent: str
    refinement: int
    original_384_tick_schedule_sha256: str
    preparation_144_tick_schedule_sha256: str
    last_perturbed_reference_tick: Decimal | None
    returned_x: Decimal
    returned_y: Decimal
    maximum_excursion: Decimal
    maximum_increment: Decimal

    def __post_init__(self) -> None:
        for name in ("original_384_tick_schedule_sha256", "preparation_144_tick_schedule_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            self.parent not in PARENTS
            or type(self.refinement) is not int
            or self.refinement not in (1, 2)
        ):
            raise ValueError("imported parent changes the exact five-schedule/two-view menu")
        if (
            abs(self.returned_x - Decimal(2) / 3) > Decimal("1e-12")
            or abs(self.returned_y - Decimal(22) / 3) > Decimal("1e-12")
            or not Decimal(0) <= self.maximum_excursion <= Decimal("0.125000000001")
            or not Decimal(0)
            <= self.maximum_increment
            <= Decimal(1) / (384 * self.refinement) + Decimal("1e-12")
            or (self.parent == "hold") != (self.last_perturbed_reference_tick is None)
            or (
                self.last_perturbed_reference_tick is not None
                and not Decimal(0) < self.last_perturbed_reference_tick <= 128
            )
        ):
            raise ValueError("imported parent fails common-endpoint preparation-only semantics")


@dataclass(frozen=True, slots=True)
class PreparationCalibrationReplay(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-calibration-replay'
    context: str
    model: str
    validation_roots: int
    rmse: Decimal
    bias: Decimal
    radius: Decimal
    covered_roots: int
    recomputed_summary_cells: int
    maximum_recorded_summary_discrepancy: Decimal
    maximum_calibration_transform_discrepancy: Decimal

    def __post_init__(self) -> None:
        if (
            self.context not in CONTEXTS
            or self.model
            not in (
                "constant",
                "gaussian",
                "gaussian_offset",
                "gaussian_scale",
                "gaussian_affine",
                "frozen",
                "frozen_affine",
                "coupled",
                "coupled_affine",
                "direct_observable",
            )
            or type(self.validation_roots) is not int
            or self.validation_roots != 32
            or type(self.covered_roots) is not int
            or not 0 <= self.covered_roots <= 32
            or type(self.recomputed_summary_cells) is not int
            or self.recomputed_summary_cells != 9
            or any(
                not v.is_finite()
                for v in (
                    self.rmse,
                    self.bias,
                    self.radius,
                    self.maximum_recorded_summary_discrepancy,
                    self.maximum_calibration_transform_discrepancy,
                )
            )
            or self.rmse < 0
            or self.radius < 0
            or not 0 <= self.maximum_recorded_summary_discrepancy <= Decimal("1e-10")
            or not 0 <= self.maximum_calibration_transform_discrepancy <= Decimal("1e-10")
        ):
            raise ValueError("calibration import does not reproduce the original split/arithmetic")


@dataclass(frozen=True, slots=True)
class PreparationSourceImportInventory(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/matrix-preparation/preparation-source-import-inventory'
    inventory_id: str
    original_source_config: ObjectIdentity
    prefix_source_config: ObjectIdentity
    original_adjudication_sha256: str
    original_source_commit: str
    prefixes: tuple[RetainedPreparationPrefix, ...]
    prefix_bundles: tuple[PreparationPrefixBundleRef, ...]
    parents: tuple[PreparationImportedParent, ...]
    calibration_manifest_sha256: str
    calibration_files: tuple[PreparationImportedFile, ...]
    calibration_replay: tuple[PreparationCalibrationReplay, ...]
    native_contract_source_files: tuple[tuple[str, str], ...]
    calibration_recipe: str = "FIT_ROOTS_0_15_CONTEXT_OLS_GAIN_A_PLUS_B_RAW;CAL_ROOTS_16_31_MAX_OVER_TWO_VIEWS;VALIDATION_ROOTS_32_63;NOMINAL_0.90"
    exposed_development_roots: int = 128
    new_native_updates: int = 0
    grants_authority: bool = False

    def __post_init__(self) -> None:
        validate_sha256(self.original_adjudication_sha256, field_name="original_adjudication_sha256")
        validate_sha256(self.calibration_manifest_sha256, field_name="calibration_manifest_sha256")
        if re.fullmatch(r"[0-9a-f]{40}", self.original_source_commit) is None:
            raise ValueError("source inventory requires an explicit source commit")
        require_sorted_unique_ids(self.prefixes, attribute="root_id", field_name="prefixes")
        require_sorted_unique_ids(
            self.calibration_files, attribute="relative_path", field_name="calibration_files"
        )
        if (
            self.prefix_source_config.object_schema != 'empirical-lawhood/simulators/six-matrix-response/response-geometry-development-native-config'
            or any(row.source_config != self.prefix_source_config or row.source_export.original_source != self.original_source_config for row in self.prefixes)
            or tuple(p.root for p in self.prefixes) != preparation_roots()
            or tuple((b.context, b.group_index) for b in self.prefix_bundles)
            != tuple((c, i) for c in CONTEXTS for i in range(16))
            or tuple((p.parent, p.refinement) for p in self.parents)
            != tuple((p, r) for p in PARENTS for r in (1, 2))
            or len(self.calibration_files) != 23
            or len(self.calibration_replay) != 20
            or len({(r.context, r.model) for r in self.calibration_replay}) != 20
            or self.exposed_development_roots != 128
            or self.new_native_updates != 0
            or self.grants_authority is not False
        ):
            raise ValueError(
                "source inventory changes imported identities, denominator or authority"
            )
        for path, digest in self.native_contract_source_files:
            validate_relative_locator(path)
            validate_sha256(digest, field_name="native_contract_source_files")
