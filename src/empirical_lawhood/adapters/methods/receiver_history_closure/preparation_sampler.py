"""Outcome-blind scale-coupled untouched-preparation sampler for receiver-history.

The canonical coarse views preserve denominator-local normalized charge using
descriptor capacitances.  Invalid requested preparations are retained through
an explicit validity mask and raw IEEE-754 bit arrays; they are never redrawn
or silently converted into valid observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from hashlib import sha256
from typing import Mapping

import numpy as np
import numpy.typing as npt
from scipy.sparse.linalg import expm_multiply

from empirical_lawhood.adapters.receiver_history.contracts import (
    ReceiverHistoryConfig,
    ReceiverHistoryCoordinateLabel,
    ReceiverHistoryDenominatorDescriptor,
    ReceiverHistoryPreparationMapManifest,
)

from .history import assemble_dense_operator, coordinate_labels


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

_SCALE_ORDER = (64, 128, 256)
_MAP_RESIDUAL_LIMIT = 1e-13


@dataclass(frozen=True, slots=True)
class UntouchedPreparationExecution:
    manifest: ReceiverHistoryPreparationMapManifest
    coordinates: tuple[ReceiverHistoryCoordinateLabel, ...]
    float_arrays: Mapping[str, FloatArray]
    int_arrays: Mapping[str, IntArray]


def _domain_rng(seed: bytes, unit_id: str) -> np.random.Generator:
    if len(seed) != 32:
        raise ValueError("receiver-history preparation seed must contain exactly 32 bytes")
    digest = sha256(
        seed + b"\0receiver-history-untouched-preparations\0" + unit_id.encode()
    ).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))


def _decimal_weights(descriptor: ReceiverHistoryDenominatorDescriptor) -> tuple[Decimal, ...]:
    denominator = Decimal(descriptor.scale_cells) * descriptor.capacitance_bar_farads
    return tuple(value / denominator for value in descriptor.capacitances_farads)


def _float_weights(descriptor: ReceiverHistoryDenominatorDescriptor) -> npt.NDArray[np.longdouble]:
    return np.asarray([str(value) for value in _decimal_weights(descriptor)], dtype=np.longdouble)


def _direct_charge_map(
    fine: FloatArray,
    *,
    fine_descriptor: ReceiverHistoryDenominatorDescriptor,
    coarse_descriptor: ReceiverHistoryDenominatorDescriptor,
) -> FloatArray:
    """Apply the direct N256-to-N map using Decimal-derived coefficients."""

    coarse_cells = coarse_descriptor.scale_cells
    if fine_descriptor.scale_cells != 256 or coarse_cells not in {64, 128}:
        raise ValueError("receiver-history charge map requires N256 to N64/N128")
    block = 256 // coarse_cells
    fine_weights = _decimal_weights(fine_descriptor)
    coarse_weights = _decimal_weights(coarse_descriptor)
    coefficients = np.empty((coarse_cells, block), dtype=np.longdouble)
    with localcontext() as context:
        context.prec = 80
        for coarse_index in range(coarse_cells):
            start = coarse_index * block
            for offset in range(block):
                coefficients[coarse_index, offset] = np.longdouble(
                    str(fine_weights[start + offset] / coarse_weights[coarse_index])
                )
    grouped = np.asarray(fine, dtype=np.longdouble).reshape(fine.shape[0], coarse_cells, block)
    mapped = np.sum(grouped * coefficients[None, :, :], axis=2, dtype=np.longdouble)
    return np.asarray(mapped, dtype=np.float64)


def _pre_cast_charge_preservation_error(
    fine: FloatArray,
    descriptors: Mapping[int, ReceiverHistoryDenominatorDescriptor],
) -> float:
    """Measure the extended-precision map identity before binary64 storage."""

    fine_extended = np.asarray(fine, dtype=np.longdouble)
    fine_charge = fine_extended @ _float_weights(descriptors[256])
    maximum = 0.0
    for scale in (64, 128):
        coarse_cells = scale
        block = 256 // coarse_cells
        fine_weights = _decimal_weights(descriptors[256])
        coarse_weights = _decimal_weights(descriptors[scale])
        coefficients = np.empty((coarse_cells, block), dtype=np.longdouble)
        with localcontext() as context:
            context.prec = 80
            for coarse_index in range(coarse_cells):
                start = coarse_index * block
                for offset in range(block):
                    coefficients[coarse_index, offset] = np.longdouble(
                        str(fine_weights[start + offset] / coarse_weights[coarse_index])
                    )
        mapped = np.sum(
            fine_extended.reshape(fine.shape[0], coarse_cells, block) * coefficients[None, :, :],
            axis=2,
            dtype=np.longdouble,
        )
        residual = np.abs(mapped @ _float_weights(descriptors[scale]) - fine_charge)
        maximum = max(maximum, float(np.max(residual, initial=0.0)))
    return maximum


def scale_coupled_initial_states(
    *,
    seed: bytes,
    unit_id: str,
    descriptors: Mapping[int, ReceiverHistoryDenominatorDescriptor],
    count: int = 512,
) -> dict[int, FloatArray]:
    """Draw N256 Beta(2,2) states and direct capacitance-weighted views."""

    if count != 512:
        raise ValueError("receiver-history requires exactly 512 untouched preparations")
    if tuple(sorted(descriptors)) != _SCALE_ORDER:
        raise ValueError("receiver-history charge map requires N64/N128/N256 descriptors")
    rng = _domain_rng(seed, unit_id)
    fine = np.asarray(0.15 + 0.70 * rng.beta(2.0, 2.0, size=(count, 256)), dtype=np.float64)
    values = {
        64: _direct_charge_map(
            fine,
            fine_descriptor=descriptors[256],
            coarse_descriptor=descriptors[64],
        ),
        128: _direct_charge_map(
            fine,
            fine_descriptor=descriptors[256],
            coarse_descriptor=descriptors[128],
        ),
        256: fine,
    }
    if any(array.shape != (count, scale) for scale, array in values.items()):
        raise ValueError("receiver-history scale-coupled preparation shape is invalid")
    return values


def charge_preservation_errors(
    states: Mapping[int, FloatArray],
    descriptors: Mapping[int, ReceiverHistoryDenominatorDescriptor],
) -> dict[int, FloatArray]:
    """Return per-preparation normalized-charge residuals in extended precision."""

    fine = np.asarray(states[256], dtype=np.longdouble)
    fine_charge = fine @ _float_weights(descriptors[256])
    result: dict[int, FloatArray] = {}
    for scale in (64, 128):
        coarse = np.asarray(states[scale], dtype=np.longdouble)
        residual = np.abs(coarse @ _float_weights(descriptors[scale]) - fine_charge)
        result[scale] = np.asarray(residual, dtype=np.float64)
    return result


def charge_preservation_error(
    states: Mapping[int, FloatArray],
    descriptors: Mapping[int, ReceiverHistoryDenominatorDescriptor],
) -> float:
    errors = charge_preservation_errors(states, descriptors)
    finite = [value[np.isfinite(value)] for value in errors.values()]
    if any(value.size != next(iter(errors.values())).size for value in finite):
        return float("inf")
    return max(float(np.max(value, initial=0.0)) for value in finite)


def _receiver_history(
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    earliest_states: FloatArray,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    """Propagate all 32 history times in one Krylov invocation."""

    operator = assemble_dense_operator(descriptor)
    lag_seconds = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    timeline = expm_multiply(
        operator.state_matrix,
        earliest_states.T,
        start=0.0,
        stop=lag_seconds * config.history_max_depth,
        num=config.history_max_depth + 1,
        endpoint=True,
        traceA=float(np.trace(operator.state_matrix)),
    )
    # expm_multiply returns time x cell x preparation.
    timeline_by_preparation = np.asarray(np.transpose(timeline, (2, 0, 1)), dtype=np.float64)
    receiver_forward = np.einsum(
        "bti,ri->btr", timeline_by_preparation, operator.receiver, optimize=True
    )
    receiver_by_lag = np.ascontiguousarray(receiver_forward[:, ::-1, :])
    present = np.ascontiguousarray(timeline_by_preparation[:, -1, :])
    finite = np.all(np.isfinite(timeline_by_preparation), axis=(1, 2))
    in_envelope = np.all(
        (timeline_by_preparation >= -1e-12) & (timeline_by_preparation <= 1.0 + 1e-12),
        axis=(1, 2),
    )
    return present, receiver_by_lag, np.asarray(finite & in_envelope, dtype=np.bool_)


def _collision_edges(
    histories: FloatArray,
    coordinates: tuple[ReceiverHistoryCoordinateLabel, ...],
    validity: npt.NDArray[np.bool_],
) -> IntArray:
    """Build all requested graphs from one cumulative pair-distance pass."""

    count = histories.shape[0]
    left, right = np.triu_indices(count, k=1)
    valid_pairs = validity[left] & validity[right]
    cumulative = np.zeros(left.size, dtype=np.float64)
    coordinates_by_depth: dict[int, list[tuple[int, ReceiverHistoryCoordinateLabel]]] = {}
    for coordinate_index, coordinate in enumerate(coordinates):
        coordinates_by_depth.setdefault(coordinate.depth, []).append((coordinate_index, coordinate))
    edges: list[npt.NDArray[np.int64]] = []
    for depth in range(histories.shape[1]):
        distance = np.max(np.abs(histories[left, depth] - histories[right, depth]), axis=1)
        np.maximum(cumulative, distance, out=cumulative)
        for coordinate_index, coordinate in coordinates_by_depth.get(depth, ()):
            selected = valid_pairs & (cumulative <= float(coordinate.resolution_epsilon))
            if np.any(selected):
                coordinate_column = np.full(
                    np.count_nonzero(selected), coordinate_index, dtype=np.int64
                )
                edges.append(
                    np.column_stack((coordinate_column, left[selected], right[selected])).astype(
                        np.int64, copy=False
                    )
                )
    if not edges:
        return np.empty((0, 3), dtype=np.int64)
    return np.ascontiguousarray(np.vstack(edges), dtype=np.int64)


def _increment(reasons: dict[str, int], key: str, mask: npt.NDArray[np.bool_]) -> None:
    count = int(np.count_nonzero(mask))
    if count:
        reasons[key] = reasons.get(key, 0) + count


def _raw_state_bits(values: FloatArray) -> IntArray:
    """Retain every IEEE-754 payload bit, including nonfinite invalid states."""

    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return np.ascontiguousarray(contiguous.view(np.int64))


def sample_untouched_preparations(
    *,
    config: ReceiverHistoryConfig,
    unit_id: str,
    descriptors: Mapping[int, ReceiverHistoryDenominatorDescriptor],
    seed: bytes,
) -> UntouchedPreparationExecution:
    if tuple(sorted(descriptors)) != _SCALE_ORDER:
        raise ValueError("receiver-history untouched sampler requires N64/N128/N256 descriptors")
    if any(value.unit_id != unit_id for value in descriptors.values()):
        raise ValueError("receiver-history untouched descriptors do not share one unit")
    raw_initial = scale_coupled_initial_states(seed=seed, unit_id=unit_id, descriptors=descriptors)
    pre_cast_error = _pre_cast_charge_preservation_error(raw_initial[256], descriptors)
    residuals = charge_preservation_errors(raw_initial, descriptors)
    reasons: dict[str, int] = {}
    float_arrays: dict[str, FloatArray] = {}
    int_arrays: dict[str, IntArray] = {}
    valid_counts: list[int] = []
    all_coordinates: list[ReceiverHistoryCoordinateLabel] = []

    for scale in _SCALE_ORDER:
        raw = raw_initial[scale]
        finite = np.all(np.isfinite(raw), axis=1)
        in_envelope = np.all((raw >= 0.0) & (raw <= 1.0), axis=1)
        valid = finite & in_envelope
        if scale == 256:
            _increment(reasons, "NONFINITE_SOURCE_STATE", ~finite)
        else:
            _increment(reasons, "CHARGE_MAP_NONFINITE", ~finite)
            _increment(reasons, "MAPPED_STATE_OUTSIDE_ENVELOPE", finite & ~in_envelope)
            charge_valid = np.isfinite(residuals[scale]) & (residuals[scale] <= _MAP_RESIDUAL_LIMIT)
            _increment(
                reasons,
                "CHARGE_MAP_RESIDUAL_EXCEEDED",
                finite & ~charge_valid,
            )
            valid &= charge_valid

        # Invalid binary values are retained in raw-bits.  The finite working
        # buffer is explicitly governed by validity-mask and cannot contribute
        # to any graph or adjudication.
        working = np.where(valid[:, None], raw, 0.0).astype(np.float64, copy=False)
        present, histories, propagation_valid = _receiver_history(
            config, descriptors[scale], working
        )
        propagation_finite = np.all(np.isfinite(present), axis=1) & np.all(
            np.isfinite(histories), axis=(1, 2)
        )
        _increment(
            reasons,
            "ZERO_INPUT_PROPAGATION_NONFINITE",
            valid & ~propagation_finite,
        )
        _increment(
            reasons,
            "ZERO_INPUT_PROPAGATION_OUTSIDE_ENVELOPE",
            valid & propagation_finite & ~propagation_valid,
        )
        valid &= propagation_valid
        coordinates = coordinate_labels(config, scale)
        edges = _collision_edges(histories, coordinates, valid)
        float_arrays[f"initial-states-n{scale}"] = working
        float_arrays[f"present-states-n{scale}"] = present
        float_arrays[f"receiver-history-n{scale}"] = histories
        int_arrays[f"raw-bits-n{scale}"] = _raw_state_bits(raw)
        int_arrays[f"validity-mask-n{scale}"] = np.asarray(valid, dtype=np.int64)
        int_arrays[f"collision-edges-n{scale}"] = edges
        valid_counts.append(int(np.count_nonzero(valid)))
        all_coordinates.extend(coordinates)

    realized_error = charge_preservation_error(raw_initial, descriptors)
    fine_digest = sha256(raw_initial[256].tobytes(order="C")).hexdigest()
    mapped_digest = sha256(
        raw_initial[64].tobytes(order="C") + raw_initial[128].tobytes(order="C")
    ).hexdigest()
    collision_digest = sha256(
        b"".join(
            int_arrays[key].tobytes(order="C")
            for key in sorted(int_arrays)
            if key.startswith("collision-edges-")
        )
    ).hexdigest()
    raw_bits_digest = sha256(
        b"".join(
            int_arrays[key].tobytes(order="C")
            for key in sorted(int_arrays)
            if key.startswith("raw-bits-")
        )
    ).hexdigest()
    validity_digest = sha256(
        b"".join(
            int_arrays[key].tobytes(order="C")
            for key in sorted(int_arrays)
            if key.startswith("validity-mask-")
        )
    ).hexdigest()
    manifest = ReceiverHistoryPreparationMapManifest(
        manifest_id=f"preparation-map.{unit_id}",
        unit_id=unit_id,
        preparation_count=config.untouched_preparation_count,
        fine_scale_cells=256,
        coarse_scale_cells=(64, 128),
        fine_states_sha256=fine_digest,
        mapped_states_sha256=mapped_digest,
        collision_edges_sha256=collision_digest,
        prefix_counts=config.untouched_prefix_counts,
        valid_preparation_counts=tuple(valid_counts),
        invalid_reason_counts=tuple(sorted(reasons.items())),
        pre_cast_charge_preservation_max_error=Decimal(str(pre_cast_error)),
        charge_preservation_max_error=Decimal(str(realized_error)),
        invalid_raw_bits_sha256=raw_bits_digest,
        validity_mask_sha256=validity_digest,
        outcome_count_at_freeze=0,
        algorithm_id=config.preparation_sampler_algorithm_id,
    )
    return UntouchedPreparationExecution(
        manifest=manifest,
        coordinates=tuple(sorted(all_coordinates, key=lambda value: value.coordinate_id)),
        float_arrays=float_arrays,
        int_arrays=int_arrays,
    )


__all__ = [
    "UntouchedPreparationExecution",
    "charge_preservation_error",
    "charge_preservation_errors",
    "sample_untouched_preparations",
    "scale_coupled_initial_states",
]
