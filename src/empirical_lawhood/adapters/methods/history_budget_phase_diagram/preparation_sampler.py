"""Outcome-blind scale-coupled untouched-preparation sampler for history budget phase diagram."""

from __future__ import annotations

from empirical_lawhood.adapters.history_budget_scientific_inputs import HistoryBudgetPreparationScientificInput

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import Mapping

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from empirical_lawhood.adapters.history_budget_phase_diagram.contracts import HistoryBudgetPhaseDiagramConfig, HistoryBudgetPhaseDiagramCoordinateLabel, HistoryBudgetPhaseDiagramDenominatorDescriptor, HistoryBudgetPhaseDiagramPreparationMapManifest

from .history import assemble_dense_operator, coordinate_labels


FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class UntouchedPreparationExecution:
    manifest: HistoryBudgetPhaseDiagramPreparationMapManifest
    coordinates: tuple[HistoryBudgetPhaseDiagramCoordinateLabel, ...]
    float_arrays: Mapping[str, FloatArray]
    int_arrays: Mapping[str, IntArray]


def _domain_rng(seed: bytes, unit_id: str, scientific_input: HistoryBudgetPreparationScientificInput | None) -> np.random.Generator:
    if len(seed) != 32:
        raise ValueError("history budget phase diagram preparation seed must contain exactly 32 bytes")
    if not isinstance(scientific_input, HistoryBudgetPreparationScientificInput):
        raise ValueError("untouched sampler requires authenticated original full numeric seed input before draws")
    scientific_input.require_current_binding(programme_ordinal=1, unit_id=unit_id, seed=seed)
    digest = bytes.fromhex(scientific_input.sampler_full_seed_sha256)
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))


def scale_coupled_initial_states(
    *,
    seed: bytes,
    scientific_input: HistoryBudgetPreparationScientificInput | None = None,
    unit_id: str,
    count: int = 512,
) -> dict[int, FloatArray]:
    """Draw N256 Beta(2,2) states and exact dyadic block-average views."""

    if count != 512:
        raise ValueError("history budget phase diagram requires exactly 512 untouched preparations")
    rng = _domain_rng(seed, unit_id, scientific_input)
    fine = np.asarray(0.15 + 0.70 * rng.beta(2.0, 2.0, size=(count, 256)), dtype=np.float64)
    n128 = np.asarray(fine.reshape(count, 128, 2).mean(axis=2), dtype=np.float64)
    n64 = np.asarray(fine.reshape(count, 64, 4).mean(axis=2), dtype=np.float64)
    values = {64: n64, 128: n128, 256: fine}
    if any(
        array.shape != (count, scale) or not np.all(np.isfinite(array))
        for scale, array in values.items()
    ):
        raise ValueError("history budget phase diagram scale-coupled preparation shape is invalid")
    if any(np.min(array) < 0.15 or np.max(array) > 0.85 for array in values.values()):
        raise ValueError("history budget phase diagram untouched preparation escaped its frozen envelope")
    return values


def charge_preservation_error(states: Mapping[int, FloatArray]) -> float:
    fine_mean = np.mean(states[256], axis=1)
    errors = [
        float(np.max(np.abs(np.mean(states[scale], axis=1) - fine_mean))) for scale in (64, 128)
    ]
    return max(errors)


def _receiver_history(
    config: HistoryBudgetPhaseDiagramConfig,
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    earliest_states: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    operator = assemble_dense_operator(descriptor)
    lag_seconds = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    transition = expm(operator.state_matrix * lag_seconds)
    state = earliest_states.copy()
    state_timeline = np.empty(
        (earliest_states.shape[0], config.history_max_depth + 1, descriptor.scale_cells),
        dtype=np.float64,
    )
    state_timeline[:, 0, :] = state
    for index in range(1, config.history_max_depth + 1):
        state = state @ transition.T
        state_timeline[:, index, :] = state
    if not np.all(np.isfinite(state_timeline)):
        raise ValueError("history budget phase diagram untouched zero-input propagation is nonfinite")
    if np.min(state_timeline) < -1e-12 or np.max(state_timeline) > 1.0 + 1e-12:
        raise ValueError("history budget phase diagram untouched propagation escaped the hidden-state envelope")
    receiver_forward = np.einsum("bti,ri->btr", state_timeline, operator.receiver)
    receiver_by_lag = np.asarray(receiver_forward[:, ::-1, :], dtype=np.float64)
    return state_timeline, receiver_by_lag


def _collision_edges(
    histories: FloatArray,
    coordinates: tuple[HistoryBudgetPhaseDiagramCoordinateLabel, ...],
) -> IntArray:
    count = histories.shape[0]
    edges: list[tuple[int, int, int]] = []
    for coordinate_index, coordinate in enumerate(coordinates):
        view = histories[:, : coordinate.depth + 1, :]
        epsilon = float(coordinate.resolution_epsilon)
        for left in range(count - 1):
            distances = np.max(np.abs(view[left + 1 :] - view[left]), axis=(1, 2))
            for relative_right in np.flatnonzero(distances <= epsilon):
                edges.append((coordinate_index, left, left + 1 + int(relative_right)))
    return np.asarray(edges, dtype=np.int64).reshape((-1, 3))


def sample_untouched_preparations(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    unit_id: str,
    descriptors: Mapping[int, HistoryBudgetPhaseDiagramDenominatorDescriptor],
    seed: bytes,
    scientific_input: HistoryBudgetPreparationScientificInput | None = None,
) -> UntouchedPreparationExecution:
    if not isinstance(scientific_input, HistoryBudgetPreparationScientificInput):
        raise ValueError("untouched sampler requires authenticated original numeric input and current descriptor custody")
    scientific_input.require_current_descriptors(tuple(descriptors[scale].fingerprint() for scale in sorted(descriptors)))
    if tuple(sorted(descriptors)) != (64, 128, 256):
        raise ValueError("history budget phase diagram untouched sampler requires N64/N128/N256 descriptors")
    if any(value.unit_id != unit_id for value in descriptors.values()):
        raise ValueError("history budget phase diagram untouched descriptors do not share one unit")
    initial = scale_coupled_initial_states(seed=seed, unit_id=unit_id, scientific_input=scientific_input)
    preservation_error = charge_preservation_error(initial)
    if preservation_error > 1e-15:
        raise ValueError("history budget phase diagram fine-to-coarse charge preservation failed")
    float_arrays: dict[str, FloatArray] = {}
    int_arrays: dict[str, IntArray] = {}
    all_coordinates: list[HistoryBudgetPhaseDiagramCoordinateLabel] = []
    for scale in (64, 128, 256):
        timeline, histories = _receiver_history(config, descriptors[scale], initial[scale])
        coordinates = coordinate_labels(config, scale)
        edges = _collision_edges(histories, coordinates)
        float_arrays[f"initial-states-n{scale}"] = initial[scale]
        float_arrays[f"state-timeline-n{scale}"] = timeline
        float_arrays[f"receiver-history-n{scale}"] = histories
        int_arrays[f"collision-edges-n{scale}"] = edges
        all_coordinates.extend(coordinates)
    fine_digest = sha256(float_arrays["initial-states-n256"].tobytes(order="C")).hexdigest()
    mapped_digest = sha256(
        float_arrays["initial-states-n64"].tobytes(order="C")
        + float_arrays["initial-states-n128"].tobytes(order="C")
    ).hexdigest()
    collision_digest = sha256(
        b"".join(int_arrays[key].tobytes(order="C") for key in sorted(int_arrays))
    ).hexdigest()
    manifest = HistoryBudgetPhaseDiagramPreparationMapManifest(
        manifest_id=f"preparation-map.{unit_id}",
        unit_id=unit_id,
        preparation_count=config.untouched_preparation_count,
        fine_scale_cells=256,
        coarse_scale_cells=(64, 128),
        fine_states_sha256=fine_digest,
        mapped_states_sha256=mapped_digest,
        collision_edges_sha256=collision_digest,
        prefix_counts=config.untouched_prefix_counts,
        charge_preservation_max_error=Decimal(str(preservation_error)),
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
    "sample_untouched_preparations",
    "scale_coupled_initial_states",
]
