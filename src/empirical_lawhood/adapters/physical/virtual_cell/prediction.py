"""Deterministic, bounded construction of 2025 prediction H5AD payloads."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
import hashlib
import io
import math
from pathlib import Path
from typing import ClassVar

import h5py  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .contracts import VirtualCellContractError
from .dataset import ControlReservoirArrays


MAXIMUM_PERTURBATION_ROSTER_BYTES = 1024**2
CELL_EVAL_LOG1P_MAXIMUM = 14.999


@dataclass(frozen=True, slots=True)
class PredictionTargetAllocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/prediction-target-allocation'

    target_id: str
    recommended_cells: int
    median_umi_per_cell: Decimal
    allocated_cells: int

    def __post_init__(self) -> None:
        if not self.target_id:
            raise ValueError("prediction target ID must be nonempty")
        for count_name, count_value in (
            ("recommended_cells", self.recommended_cells),
            ("allocated_cells", self.allocated_cells),
        ):
            if isinstance(count_value, bool) or count_value <= 0:
                raise ValueError(f"{count_name} must be positive")
        validate_decimal(
            self.median_umi_per_cell,
            field_name="median_umi_per_cell",
            minimum=Decimal("0"),
        )
        if self.median_umi_per_cell == 0:
            raise ValueError("median UMI per cell must be positive")


@dataclass(frozen=True, slots=True)
class PredictionRoster(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/virtual-cell/prediction-roster'

    roster_id: str
    prefix_source_sha256: str
    targets: tuple[PredictionTargetAllocation, ...]
    control_label: str
    control_cells: int
    total_cell_limit: int
    allocation_rule: str

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        validate_sha256(self.prefix_source_sha256, field_name="prefix_source_sha256")
        require_sorted_unique_ids(self.targets, attribute="target_id", field_name="targets")
        if not self.targets or not self.control_label:
            raise ValueError("prediction roster requires targets and a control label")
        if isinstance(self.control_cells, bool) or self.control_cells <= 0:
            raise ValueError("prediction roster requires positive control cells")
        if isinstance(self.total_cell_limit, bool) or self.total_cell_limit <= 0:
            raise ValueError("prediction roster total-cell limit must be positive")
        if self.total_cells != self.total_cell_limit:
            raise ValueError("prediction allocation must exactly fill its frozen cell limit")
        if not self.allocation_rule:
            raise ValueError("prediction allocation rule must be nonempty")

    @property
    def total_cells(self) -> int:
        return self.control_cells + sum(value.allocated_cells for value in self.targets)


@dataclass(frozen=True, slots=True)
class PredictionMeanCompilation:
    target_ids: tuple[str, ...]
    absolute_means: NDArray[np.float64]
    control_mean: NDArray[np.float64]
    clipped_value_count: int
    clipped_value_fraction: float

    def __post_init__(self) -> None:
        if not self.target_ids or len(set(self.target_ids)) != len(self.target_ids):
            raise ValueError("prediction means require unique target IDs")
        if self.absolute_means.ndim != 2 or self.absolute_means.shape[0] != len(self.target_ids):
            raise ValueError("absolute prediction means are not target aligned")
        if self.control_mean.shape != (self.absolute_means.shape[1],):
            raise ValueError("control mean is not gene aligned")
        if not np.all(np.isfinite(self.absolute_means)) or np.any(self.absolute_means < 0):
            raise ValueError("absolute prediction means must be finite and nonnegative")
        if float(np.max(self.absolute_means)) > CELL_EVAL_LOG1P_MAXIMUM:
            raise ValueError("absolute prediction means exceed cell-eval's log1p range")
        if self.clipped_value_count < 0:
            raise ValueError("clipped value count cannot be negative")
        if not 0 <= self.clipped_value_fraction <= 1:
            raise ValueError("clipped value fraction must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class PredictionH5ADBuildReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/physical/virtual-cell/prediction-h5-ad-build-receipt'

    receipt_id: str
    roster_sha256: str
    predicted_mean_sha256: str
    control_reservoir_sha256: str
    gene_order_sha256: str
    h5ad_sha256: str
    row_count: int
    gene_count: int
    target_count: int
    control_cell_count: int
    seed: int
    maximum_mean_error: Decimal
    maximum_value: Decimal
    output_dtype: str
    outcome_read: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        for digest_name, digest_value in (
            ("roster_sha256", self.roster_sha256),
            ("predicted_mean_sha256", self.predicted_mean_sha256),
            ("control_reservoir_sha256", self.control_reservoir_sha256),
            ("gene_order_sha256", self.gene_order_sha256),
            ("h5ad_sha256", self.h5ad_sha256),
        ):
            validate_sha256(digest_value, field_name=digest_name)
        for count_name, count_value in (
            ("row_count", self.row_count),
            ("gene_count", self.gene_count),
            ("target_count", self.target_count),
            ("control_cell_count", self.control_cell_count),
        ):
            if isinstance(count_value, bool) or count_value <= 0:
                raise ValueError(f"{count_name} must be positive")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("prediction seed must be a nonnegative integer")
        validate_decimal(
            self.maximum_mean_error,
            field_name="maximum_mean_error",
            minimum=Decimal("0"),
        )
        validate_decimal(self.maximum_value, field_name="maximum_value", minimum=Decimal("0"))
        if self.output_dtype != "float32":
            raise ValueError("the 2025 prediction H5AD must use float32 X")
        if self.outcome_read:
            raise ValueError("prediction H5AD construction cannot read test outcomes")


def _strict_roster_rows(payload: bytes) -> tuple[tuple[str, int, Decimal], ...]:
    if len(payload) > MAXIMUM_PERTURBATION_ROSTER_BYTES:
        raise VirtualCellContractError("perturbation roster exceeds its byte bound")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise VirtualCellContractError("perturbation roster is not UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    expected = ("target_gene", "n_cells", "median_umi_per_cell")
    if tuple(reader.fieldnames or ()) != expected:
        raise VirtualCellContractError("perturbation roster header differs")
    rows: list[tuple[str, int, Decimal]] = []
    try:
        for row in reader:
            if set(row) != set(expected):
                raise VirtualCellContractError("perturbation roster row shape differs")
            target = row["target_gene"]
            if not target or target.strip() != target:
                raise VirtualCellContractError("perturbation target is invalid")
            count = int(row["n_cells"])
            median = Decimal(row["median_umi_per_cell"])
            if count <= 0 or not median.is_finite() or median <= 0:
                raise VirtualCellContractError("perturbation roster values are invalid")
            rows.append((target, count, median))
    except (ArithmeticError, TypeError, ValueError) as error:
        raise VirtualCellContractError("perturbation roster numeric value is invalid") from error
    if not rows or len({value[0] for value in rows}) != len(rows):
        raise VirtualCellContractError("perturbation roster is empty or duplicated")
    return tuple(sorted(rows))


def build_prediction_roster(
    payload: bytes,
    *,
    roster_id: str,
    expected_target_count: int,
    total_cell_limit: int,
    control_cells: int,
    minimum_cells_per_target: int,
    control_label: str = "non-targeting",
) -> PredictionRoster:
    """Allocate the exact H5AD row budget by deterministic largest remainder."""

    rows = _strict_roster_rows(payload)
    if len(rows) != expected_target_count:
        raise VirtualCellContractError("perturbation roster target count differs")
    if any(
        isinstance(value, bool) or value <= 0
        for value in (total_cell_limit, control_cells, minimum_cells_per_target)
    ):
        raise ValueError("prediction cell budgets must be positive integers")
    perturbation_budget = total_cell_limit - control_cells
    floor_budget = minimum_cells_per_target * len(rows)
    if perturbation_budget < floor_budget:
        raise ValueError("prediction budget cannot satisfy the per-target floor")
    weights = np.asarray([value[1] for value in rows], dtype=np.float64)
    remaining = perturbation_budget - floor_budget
    shares = weights / float(np.sum(weights)) * remaining
    extras = np.floor(shares).astype(np.int64)
    unallocated = remaining - int(np.sum(extras))
    fractions = shares - extras
    order = sorted(
        range(len(rows)),
        key=lambda index: (-float(fractions[index]), rows[index][0]),
    )
    for index in order[:unallocated]:
        extras[index] += 1
    source_sha256 = hashlib.sha256(payload).hexdigest()
    return PredictionRoster(
        roster_id=roster_id,
        prefix_source_sha256=source_sha256,
        targets=tuple(
            PredictionTargetAllocation(
                target_id=target,
                recommended_cells=recommended,
                median_umi_per_cell=median,
                allocated_cells=minimum_cells_per_target + int(extras[index]),
            )
            for index, (target, recommended, median) in enumerate(rows)
        ),
        control_label=control_label,
        control_cells=control_cells,
        total_cell_limit=total_cell_limit,
        allocation_rule=(
            "per-target floor then recommended-cell-weighted deterministic largest remainder"
        ),
    )


def compile_absolute_prediction_means(
    *,
    target_ids: tuple[str, ...],
    predicted_deltas: NDArray[np.float64],
    control_reservoir: ControlReservoirArrays,
    maximum_value: float = CELL_EVAL_LOG1P_MAXIMUM,
) -> PredictionMeanCompilation:
    if predicted_deltas.shape != (len(target_ids), len(control_reservoir.gene_ids)):
        raise ValueError("predicted response deltas are not target/gene aligned")
    if not np.all(np.isfinite(predicted_deltas)):
        raise ValueError("predicted response deltas contain nonfinite values")
    if not math.isfinite(maximum_value) or maximum_value <= 0 or maximum_value >= 15:
        raise ValueError("prediction maximum must lie strictly between zero and 15")
    control_mean = np.asarray(
        np.mean(control_reservoir.values, axis=0, dtype=np.float64),
        dtype=np.float64,
    )
    raw = control_mean[None, :] + predicted_deltas
    clipped = np.clip(raw, 0.0, maximum_value)
    clipped_count = int(np.count_nonzero(clipped != raw))
    return PredictionMeanCompilation(
        target_ids=target_ids,
        absolute_means=np.asarray(clipped, dtype=np.float64),
        control_mean=np.asarray(control_mean, dtype=np.float64),
        clipped_value_count=clipped_count,
        clipped_value_fraction=clipped_count / clipped.size,
    )


def _registry_sha256(values: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _array_sha256(values: NDArray[np.float64]) -> str:
    data = np.ascontiguousarray(values, dtype="<f8")
    digest = hashlib.sha256()
    digest.update(str(data.shape).encode("ascii"))
    digest.update(data.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _write_string_array(group: h5py.Group, name: str, values: tuple[str, ...]) -> None:
    dtype = h5py.string_dtype(encoding="utf-8")
    dataset = group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=dtype)
    dataset.attrs["encoding-type"] = "string-array"
    dataset.attrs["encoding-version"] = "0.2.0"


def _write_dataframe(
    group: h5py.Group,
    *,
    index_values: tuple[str, ...],
    columns: tuple[tuple[str, tuple[str, ...]], ...],
) -> None:
    group.attrs["encoding-type"] = "dataframe"
    group.attrs["encoding-version"] = "0.2.0"
    group.attrs["_index"] = "_index"
    group.attrs["column-order"] = np.asarray(
        [name for name, _values in columns],
        dtype=h5py.string_dtype("utf-8"),
    )
    _write_string_array(group, "_index", index_values)
    for name, values in columns:
        _write_string_array(group, name, values)


def _mean_preserving_cells(
    *,
    residual_source: NDArray[np.float64],
    requested_mean: NDArray[np.float64],
    cell_count: int,
    rng: np.random.Generator,
    maximum_value: float,
) -> NDArray[np.float32]:
    sampled = np.asarray(
        residual_source[rng.integers(0, residual_source.shape[0], size=cell_count, endpoint=False)],
        dtype=np.float64,
    )
    sampled -= np.mean(sampled, axis=0, keepdims=True)
    positive = np.max(sampled, axis=0)
    negative = -np.min(sampled, axis=0)
    upper_scale = np.divide(
        maximum_value - requested_mean,
        positive,
        out=np.full_like(requested_mean, np.inf),
        where=positive > 0,
    )
    lower_scale = np.divide(
        requested_mean,
        negative,
        out=np.full_like(requested_mean, np.inf),
        where=negative > 0,
    )
    scale = np.minimum(1.0, np.minimum(upper_scale, lower_scale))
    scale = np.maximum(scale, 0.0)
    cells = requested_mean[None, :] + sampled * scale[None, :]
    if np.any(cells < -1e-10) or np.any(cells > maximum_value + 1e-10):
        raise VirtualCellContractError("residual compiler violated its bounded log1p range")
    return np.asarray(np.clip(cells, 0.0, maximum_value), dtype=np.float32)


def write_prediction_h5ad(
    *,
    path: Path,
    roster: PredictionRoster,
    gene_ids: tuple[str, ...],
    predicted_means: PredictionMeanCompilation,
    control_reservoir: ControlReservoirArrays,
    control_reservoir_sha256: str,
    seed: int,
    mean_tolerance: float,
    maximum_value: float = CELL_EVAL_LOG1P_MAXIMUM,
) -> PredictionH5ADBuildReceipt:
    """Stream an AnnData-compatible dense float32 H5AD without test truth."""

    if path.exists() or path.is_symlink():
        raise VirtualCellContractError("prediction H5AD destination must not already exist")
    validate_sha256(control_reservoir_sha256, field_name="control_reservoir_sha256")
    if gene_ids != control_reservoir.gene_ids:
        raise VirtualCellContractError("prediction and control gene registries differ")
    roster_targets = tuple(value.target_id for value in roster.targets)
    if roster_targets != predicted_means.target_ids:
        raise VirtualCellContractError("prediction means and roster targets differ")
    if predicted_means.absolute_means.shape[1] != len(gene_ids):
        raise VirtualCellContractError("prediction mean gene dimension differs")
    if isinstance(seed, bool) or seed < 0:
        raise ValueError("prediction seed must be a nonnegative integer")
    if not math.isfinite(mean_tolerance) or mean_tolerance < 0:
        raise ValueError("prediction mean tolerance must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    source = np.asarray(control_reservoir.values, dtype=np.float64)
    source_mean = np.mean(source, axis=0, keepdims=True)
    residual_source = source - source_mean
    cell_target_ids: list[str] = []
    maximum_error = 0.0
    try:
        with h5py.File(path, "x") as handle:
            handle.attrs["encoding-type"] = "anndata"
            handle.attrs["encoding-version"] = "0.1.0"
            matrix = handle.create_dataset(
                "X",
                shape=(roster.total_cells, len(gene_ids)),
                dtype=np.float32,
                chunks=(min(256, roster.total_cells), min(2048, len(gene_ids))),
                compression="lzf",
            )
            matrix.attrs["encoding-type"] = "array"
            matrix.attrs["encoding-version"] = "0.2.0"
            offset = 0
            control_indices = rng.integers(
                0,
                source.shape[0],
                size=roster.control_cells,
                endpoint=False,
            )
            control_values = np.asarray(source[control_indices], dtype=np.float32)
            matrix[offset : offset + roster.control_cells, :] = control_values
            cell_target_ids.extend([roster.control_label] * roster.control_cells)
            offset += roster.control_cells
            for target_index, allocation in enumerate(roster.targets):
                values = _mean_preserving_cells(
                    residual_source=residual_source,
                    requested_mean=predicted_means.absolute_means[target_index],
                    cell_count=allocation.allocated_cells,
                    rng=rng,
                    maximum_value=maximum_value,
                )
                observed = np.mean(values, axis=0, dtype=np.float64)
                error = float(
                    np.max(np.abs(observed - predicted_means.absolute_means[target_index]))
                )
                maximum_error = max(maximum_error, error)
                matrix[offset : offset + allocation.allocated_cells, :] = values
                cell_target_ids.extend([allocation.target_id] * allocation.allocated_cells)
                offset += allocation.allocated_cells
            if offset != roster.total_cells or maximum_error > mean_tolerance:
                raise VirtualCellContractError(
                    "prediction compiler failed row closure or mean preservation"
                )
            obs_index = tuple(f"prediction-cell-{index:06d}" for index in range(offset))
            _write_dataframe(
                handle.create_group("obs"),
                index_values=obs_index,
                columns=(("target_gene", tuple(cell_target_ids)),),
            )
            _write_dataframe(
                handle.create_group("var"),
                index_values=gene_ids,
                columns=(),
            )
            for name in ("layers", "obsm", "obsp", "uns", "varm", "varp"):
                group = handle.create_group(name)
                group.attrs["encoding-type"] = "dict"
                group.attrs["encoding-version"] = "0.1.0"
    except VirtualCellContractError:
        raise
    except (OSError, TypeError, ValueError) as error:
        raise VirtualCellContractError("prediction H5AD construction failed closed") from error
    h5ad_sha256 = _file_sha256(path)
    return PredictionH5ADBuildReceipt(
        receipt_id="prediction-h5ad.virtual-cell-2025-tier-l0",
        roster_sha256=roster.fingerprint(),
        predicted_mean_sha256=_array_sha256(predicted_means.absolute_means),
        control_reservoir_sha256=control_reservoir_sha256,
        gene_order_sha256=_registry_sha256(gene_ids),
        h5ad_sha256=h5ad_sha256,
        row_count=roster.total_cells,
        gene_count=len(gene_ids),
        target_count=len(roster.targets),
        control_cell_count=roster.control_cells,
        seed=seed,
        maximum_mean_error=Decimal(str(maximum_error)),
        maximum_value=Decimal(str(maximum_value)),
        output_dtype="float32",
        outcome_read=False,
    )


__all__ = [
    "CELL_EVAL_LOG1P_MAXIMUM",
    "MAXIMUM_PERTURBATION_ROSTER_BYTES",
    "PredictionH5ADBuildReceipt",
    "PredictionMeanCompilation",
    "PredictionRoster",
    "PredictionTargetAllocation",
    "build_prediction_roster",
    "compile_absolute_prediction_means",
    "write_prediction_h5ad",
]
