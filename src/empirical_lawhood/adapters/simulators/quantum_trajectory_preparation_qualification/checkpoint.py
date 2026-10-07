"""Safe NPZ trajectory/checkpoint serialization and exact restart capsules."""

from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
import json
from typing import Mapping, cast

import numpy as np

from .schemas import stable_json_bytes
from .types import Action, Denominator, EventRecord, JumpDiagnostic, TrajectoryCheckpoint, TrajectoryResult, View


DIAGNOSTIC_COLUMNS = (
    "event_index",
    "event_time",
    "site",
    "propagator_norm_residual",
    "projected_mass_expectation",
    "projected_vector_norm_squared",
    "projected_mass_identity_residual",
    "selected_mark_probability",
    "mass_sum_residual",
    "mark_probability_sum_residual",
    "post_jump_norm_residual",
    "particle_number_residual",
)


def _event_arrays(events: tuple[EventRecord, ...]) -> dict[str, np.ndarray]:
    return {
        "event_index": np.asarray([event.event_index for event in events], dtype="<i8"),
        "event_time": np.asarray([event.event_time for event in events], dtype="<f8"),
        "event_site": np.asarray([event.site for event in events], dtype="<i2"),
    }


def _diagnostic_array(diagnostics: tuple[JumpDiagnostic, ...]) -> np.ndarray:
    return np.asarray(
        [
            (
                row.event_index,
                row.event_time,
                row.site,
                row.propagator_norm_residual,
                row.projected_mass_expectation,
                row.projected_vector_norm_squared,
                row.projected_mass_identity_residual,
                row.selected_mark_probability,
                row.mass_sum_residual,
                row.mark_probability_sum_residual,
                row.post_jump_norm_residual,
                row.particle_number_residual,
            )
            for row in diagnostics
        ],
        dtype="<f8",
    ).reshape((-1, len(DIAGNOSTIC_COLUMNS)))


def _decode_events(data: Mapping[str, np.ndarray]) -> tuple[EventRecord, ...]:
    indices = data["event_index"]
    times = data["event_time"]
    sites = data["event_site"]
    if not (len(indices) == len(times) == len(sites)):
        raise ValueError("event arrays differ in length")
    return tuple(
        EventRecord(int(index), float(time), int(site))
        for index, time, site in zip(indices, times, sites, strict=True)
    )


def _decode_diagnostics(array: np.ndarray) -> tuple[JumpDiagnostic, ...]:
    if array.ndim != 2 or array.shape[1] != len(DIAGNOSTIC_COLUMNS):
        raise ValueError("jump diagnostic array shape is invalid")
    return tuple(
        JumpDiagnostic(
            event_index=int(row[0]),
            event_time=float(row[1]),
            site=int(row[2]),
            propagator_norm_residual=float(row[3]),
            projected_mass_expectation=float(row[4]),
            projected_vector_norm_squared=float(row[5]),
            projected_mass_identity_residual=float(row[6]),
            selected_mark_probability=float(row[7]),
            mass_sum_residual=float(row[8]),
            mark_probability_sum_residual=float(row[9]),
            post_jump_norm_residual=float(row[10]),
            particle_number_residual=float(row[11]),
        )
        for row in array
    )


def trajectory_to_bytes(result: TrajectoryResult) -> bytes:
    endpoints = np.asarray(sorted(result.snapshots), dtype="<f8")
    states = np.stack(
        [np.asarray(result.snapshots[float(endpoint)], dtype="<c16") for endpoint in endpoints]
    )
    metadata = {
        "schema": 'empirical-lawhood/simulators/quantum-trajectory-reference-validation/trajectory',
        "version": '1.0.0',
        "unit_id": result.unit_id,
        "denominator": result.denominator.value,
        "action": result.action.value,
        "view": result.view.value,
        "l_sites": result.l_sites,
        "particles": result.particles,
        "initial_state": result.initial_state,
        "preparation_seed": result.preparation_seed,
        "end_time": result.end_time,
        "checkpoint_rng_states": {
            format(clock, ".17g"): state for clock, state in result.checkpoint_rng_states.items()
        },
        "checkpoint_rng_draw_counts": {
            format(clock, ".17g"): value
            for clock, value in result.checkpoint_rng_draw_counts.items()
        },
        "maximum_propagator_norm_residual": (result.maximum_propagator_norm_residual),
        "maximum_post_jump_norm_residual": result.maximum_post_jump_norm_residual,
        "maximum_projected_mass_identity_residual": (
            result.maximum_projected_mass_identity_residual
        ),
        "maximum_mark_probability_sum_residual": (result.maximum_mark_probability_sum_residual),
        "maximum_particle_number_residual": result.maximum_particle_number_residual,
    }
    output = BytesIO()
    np.savez_compressed(
        output,
        **_event_arrays(result.events),  # type: ignore[arg-type]
        diagnostics=_diagnostic_array(result.diagnostics),
        snapshot_endpoints=endpoints,
        snapshot_states=states,
        terminal_state=np.asarray(result.terminal_state, dtype="<c16"),
        metadata=np.frombuffer(stable_json_bytes(metadata), dtype=np.uint8),
    )
    return output.getvalue()


def trajectory_from_bytes(payload: bytes) -> TrajectoryResult:
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = cast(
        dict[str, object],
        json.loads(bytes(arrays["metadata"].astype(np.uint8)).decode()),
    )
    if metadata.get("schema") != 'empirical-lawhood/simulators/quantum-trajectory-reference-validation/trajectory':
        raise ValueError("trajectory schema differs from quantum trajectory preparation qualification")
    endpoints = np.asarray(arrays["snapshot_endpoints"], dtype=np.float64)
    states = np.asarray(arrays["snapshot_states"], dtype=np.complex128)
    if states.ndim != 2 or states.shape[0] != len(endpoints):
        raise ValueError("snapshot array shape is invalid")
    rng_states_raw = cast(
        Mapping[str, Mapping[str, object]],
        metadata["checkpoint_rng_states"],
    )
    rng_counts_raw = cast(Mapping[str, int], metadata["checkpoint_rng_draw_counts"])
    return TrajectoryResult(
        unit_id=str(metadata["unit_id"]),
        denominator=Denominator(str(metadata["denominator"])),
        action=Action(str(metadata["action"])),
        view=View(str(metadata["view"])),
        l_sites=cast(int, metadata["l_sites"]),
        particles=cast(int, metadata["particles"]),
        initial_state=cast(int, metadata["initial_state"]),
        preparation_seed=cast(int, metadata["preparation_seed"]),
        end_time=cast(float, metadata["end_time"]),
        terminal_state=np.asarray(arrays["terminal_state"], dtype=np.complex128),
        events=_decode_events(arrays),
        diagnostics=_decode_diagnostics(arrays["diagnostics"]),
        snapshots={
            float(endpoint): np.asarray(state, dtype=np.complex128)
            for endpoint, state in zip(endpoints, states, strict=True)
        },
        checkpoint_rng_states={float(clock): state for clock, state in rng_states_raw.items()},
        checkpoint_rng_draw_counts={
            float(clock): int(value) for clock, value in rng_counts_raw.items()
        },
        maximum_propagator_norm_residual=cast(
            float,
            metadata["maximum_propagator_norm_residual"],
        ),
        maximum_post_jump_norm_residual=cast(
            float,
            metadata["maximum_post_jump_norm_residual"],
        ),
        maximum_projected_mass_identity_residual=cast(
            float,
            metadata["maximum_projected_mass_identity_residual"],
        ),
        maximum_mark_probability_sum_residual=cast(
            float,
            metadata["maximum_mark_probability_sum_residual"],
        ),
        maximum_particle_number_residual=cast(
            float,
            metadata["maximum_particle_number_residual"],
        ),
    )


def checkpoint_from_result(
    result: TrajectoryResult,
    clock: float,
) -> TrajectoryCheckpoint:
    if clock not in result.snapshots:
        raise ValueError("requested checkpoint clock is absent")
    events = tuple(event for event in result.events if event.event_time < clock)
    diagnostics = tuple(row for row in result.diagnostics if row.event_time < clock)
    return TrajectoryCheckpoint(
        unit_id=result.unit_id,
        denominator=result.denominator,
        action=result.action,
        l_sites=result.l_sites,
        particles=result.particles,
        clock=clock,
        state=np.asarray(result.snapshots[clock], dtype=np.complex128),
        events=events,
        diagnostics=diagnostics,
        rng_state=result.checkpoint_rng_states[clock],
        rng_draw_count=result.checkpoint_rng_draw_counts[clock],
        maximum_propagator_norm_residual=max(
            (row.propagator_norm_residual for row in diagnostics),
            default=0.0,
        ),
        maximum_post_jump_norm_residual=max(
            (row.post_jump_norm_residual for row in diagnostics),
            default=0.0,
        ),
        maximum_projected_mass_identity_residual=max(
            (row.projected_mass_identity_residual for row in diagnostics),
            default=0.0,
        ),
        maximum_mark_probability_sum_residual=max(
            (row.mark_probability_sum_residual for row in diagnostics),
            default=0.0,
        ),
        maximum_particle_number_residual=max(
            (row.particle_number_residual for row in diagnostics),
            default=0.0,
        ),
    )


def checkpoint_to_bytes(checkpoint: TrajectoryCheckpoint) -> bytes:
    metadata = {
        "schema": 'empirical-lawhood/simulators/quantum-trajectory/reference-compatible-exact-restart-checkpoint',
        "version": '1.0.0',
        **{
            key: value
            for key, value in asdict(checkpoint).items()
            if key not in {"state", "events", "diagnostics"}
        },
        "denominator": checkpoint.denominator.value,
        "action": checkpoint.action.value,
    }
    output = BytesIO()
    np.savez_compressed(
        output,
        **_event_arrays(checkpoint.events),  # type: ignore[arg-type]
        diagnostics=_diagnostic_array(checkpoint.diagnostics),
        state=np.asarray(checkpoint.state, dtype="<c16"),
        metadata=np.frombuffer(stable_json_bytes(metadata), dtype=np.uint8),
    )
    return output.getvalue()


def checkpoint_from_bytes(payload: bytes) -> TrajectoryCheckpoint:
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    metadata = cast(
        dict[str, object],
        json.loads(bytes(arrays["metadata"].astype(np.uint8)).decode()),
    )
    if metadata.get("schema") != 'empirical-lawhood/simulators/quantum-trajectory/reference-compatible-exact-restart-checkpoint':
        raise ValueError("checkpoint schema differs from quantum trajectory preparation qualification")
    return TrajectoryCheckpoint(
        unit_id=str(metadata["unit_id"]),
        denominator=Denominator(str(metadata["denominator"])),
        action=Action(str(metadata["action"])),
        l_sites=cast(int, metadata["l_sites"]),
        particles=cast(int, metadata["particles"]),
        clock=cast(float, metadata["clock"]),
        state=np.asarray(arrays["state"], dtype=np.complex128),
        events=_decode_events(arrays),
        diagnostics=_decode_diagnostics(arrays["diagnostics"]),
        rng_state=cast(Mapping[str, object], metadata["rng_state"]),
        rng_draw_count=cast(int, metadata["rng_draw_count"]),
        maximum_propagator_norm_residual=cast(
            float,
            metadata["maximum_propagator_norm_residual"],
        ),
        maximum_post_jump_norm_residual=cast(
            float,
            metadata["maximum_post_jump_norm_residual"],
        ),
        maximum_projected_mass_identity_residual=cast(
            float,
            metadata["maximum_projected_mass_identity_residual"],
        ),
        maximum_mark_probability_sum_residual=cast(
            float,
            metadata["maximum_mark_probability_sum_residual"],
        ),
        maximum_particle_number_residual=cast(
            float,
            metadata["maximum_particle_number_residual"],
        ),
    )


__all__ = [
    "DIAGNOSTIC_COLUMNS",
    "checkpoint_from_bytes",
    "checkpoint_from_result",
    "checkpoint_to_bytes",
    "trajectory_from_bytes",
    "trajectory_to_bytes",
]
