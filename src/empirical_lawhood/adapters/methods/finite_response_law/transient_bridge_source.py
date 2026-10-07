"""Current nine-menu native acquisition using the shared owned phase primitive.

The caller binds allocation/protocol/code before effects and supplies a custody
sink. This module owns no files, store, scheduler, qualification or private path.
Every accepted failure and every unentered denominator cell remains explicit.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.kernel.matrix_inputs import MatrixRootAllocation
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.methods.preparation_applicability.measurement import response_arrays
from empirical_lawhood.adapters.methods.prepared_response.projection import reduce_native_response_arrays
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import REFERENCE_INSTRUMENT, reference_sketch
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import PREPARATION_POLICY_SCHEDULES
from empirical_lawhood.adapters.simulators.preparation_applicability.contracts import WORDS
from empirical_lawhood.adapters.simulators.preparation_applicability.native import acquire_phase, acquire_prefix
from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame, _observation_window
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from .transient_bridge_source_records import (
    CurrentPreparationNativeCell, CurrentPreparationRootReport, CurrentPreparationSourceConfig, MatrixPreparationSourceConfig,
    current_preparation_cell_ids,
)


@dataclass(frozen=True, slots=True)
class CurrentPreparationRootArrays:
    report: CurrentPreparationRootReport
    arrays: Mapping[str, NDArray[np.generic]]


def trajectory_features(
    *, frame: PreparedPortFrame, prefix, parent
) -> tuple[NDArray[np.float64], NDArray[np.complex128], NDArray[np.complex128]]:
    """Reduce 26 actual observations, with each t-32 rate strictly backward."""
    if prefix.disposition != "COMPLETE" or parent.disposition != "COMPLETE" or parent.ticks != tuple(range(4096, 4497, 16)):
        raise ValueError("UNEVALUABLE: current parent trajectory is incomplete")
    positions = _decode(parent.positions_base64, (26, 2, 3, 4, 4))
    momenta = _decode(parent.momenta_base64, (26, 2, 3, 4, 4))
    past_positions = _decode(prefix.history_positions_base64, (31, 2, 3, 4, 4))
    past_momenta = _decode(prefix.history_momenta_base64, (31, 2, 3, 4, 4))
    if not np.array_equal(past_positions[-1], positions[0]) or not np.array_equal(past_momenta[-1], momenta[0]):
        raise ValueError("Current trajectory substitutes its pre-parent state")
    ticks = prefix.history_ticks + parent.ticks[1:]
    all_positions = np.concatenate((past_positions, positions[1:]))
    all_momenta = np.concatenate((past_momenta, momenta[1:]))
    features = np.empty((26, 24), dtype=np.float64)
    for sample in range(26):
        window = slice(sample, sample + 31)
        x, p = all_positions[window], all_momenta[window]
        snapshot = _observation_window(frame, ticks[window], x, p, last_only=True)[0]
        current = reference_sketch(x[-1], frame)
        previous = reference_sketch(x[-3], frame)
        features[sample] = np.r_[snapshot[list(REFERENCE_INSTRUMENT.snapshot_channels)], current, (current - previous) / 0.032]
    return features, positions, momenta


def acquire_current_preparation_root(
    *,
    config: CurrentPreparationSourceConfig | MatrixPreparationSourceConfig,
    root: MatrixRootAllocation,
    retain: Callable[[str, CanonicalRecord], ArtifactIdentity],
    progress: Callable[[int], None] | None = None,
) -> CurrentPreparationRootArrays:
    """Acquire one actual Q2/CIR1 root and retain the full 344-cell denominator."""
    if root.cohort not in ("q2", "cir1") or root.source_prefix is not None:
        raise ValueError("Current readiness producer requires its actual allocated Q2/CIR1 prefix construction")
    cells = {cell_id: CurrentPreparationNativeCell(cell_id, "UNENTERED", None, "DEPENDENCY_UNEVALUABLE") for cell_id in current_preparation_cell_ids(root.root_id)}
    arrays = {
        "x": np.full((24, 2), np.nan), "z": np.full((9, 24, 2), np.nan),
        "y": np.full((9, 4, 8, 2, 2), np.nan), "work": np.full((9, 2), np.nan),
        "features": np.full((9, 2, 26, 24), np.nan),
        "positions": np.full((9, 2, 26, 2, 3, 4, 4), complex(np.nan, np.nan), dtype=np.complex128),
        "momenta": np.full((9, 2, 26, 2, 3, 4, 4), complex(np.nan, np.nan), dtype=np.complex128),
        "response_observed": np.zeros((9, 4, 8, 2, 2), dtype=bool),
        "parent_complete": np.zeros((9, 2), dtype=bool),
        "frame": np.full((2, 3, 4, 4), complex(np.nan, np.nan), dtype=np.complex128),
        "prefix_positions": np.full((2, 31, 2, 3, 4, 4), complex(np.nan, np.nan), dtype=np.complex128),
        "prefix_momenta": np.full((2, 31, 2, 3, 4, 4), complex(np.nan, np.nan), dtype=np.complex128),
        "requested_ticks": np.arange(4096, 4497, 16, dtype=np.int64),
    }
    source_sha = config.fingerprint()
    prefix = acquire_prefix(root, source_sha, progress)
    for phase in prefix.phases:
        cell_id = f"{root.root_id}.prefix.r{phase.refinement}"
        cells[cell_id] = CurrentPreparationNativeCell(cell_id, phase.disposition, retain(cell_id, phase), phase.reason)
    reason = None
    if prefix.frame_base64 is None or len(prefix.features) != 2:
        reason = "PREFIX_OR_FRAME_UNRESOLVED"
    else:
        arrays["x"][:] = np.asarray(prefix.features, dtype=np.float64).T
        frame = PreparedPortFrame(4096, _decode(prefix.frame_base64, (2, 3, 4, 4)))
        arrays["frame"][:] = frame.modes
        for phase in prefix.phases:
            arrays["prefix_positions"][phase.refinement - 1] = _decode(phase.history_positions_base64, (31, 2, 3, 4, 4))
            arrays["prefix_momenta"][phase.refinement - 1] = _decode(phase.history_momenta_base64, (31, 2, 3, 4, 4))
        for schedule_index, schedule in enumerate(PREPARATION_POLICY_SCHEDULES):
            for refinement in (1, 2):
                cell_id = f"{root.root_id}.parent.s{schedule_index}.r{refinement}"
                parent = acquire_phase(phase_name="parent", allocation=root, refinement=refinement, source_sha256=source_sha, incoming=prefix.phases[refinement - 1], schedule_index=schedule_index, frame=frame, progress=progress, preparation_schedule=schedule)
                cells[cell_id] = CurrentPreparationNativeCell(cell_id, parent.disposition, retain(cell_id, parent), parent.reason)
                if parent.disposition != "COMPLETE":
                    reason = "INCOMPLETE_NATIVE_PREPARATION"
                    continue
                try:
                    features, positions, momenta = trajectory_features(frame=frame, prefix=prefix.phases[refinement - 1], parent=parent)
                except ValueError:
                    reason = "UNRESOLVED_TRAJECTORY_INSTRUMENT"
                else:
                    arrays["features"][schedule_index, refinement - 1] = features
                    arrays["positions"][schedule_index, refinement - 1] = positions
                    arrays["momenta"][schedule_index, refinement - 1] = momenta
                    arrays["z"][schedule_index, :, refinement - 1] = features[-1]
                    arrays["parent_complete"][schedule_index, refinement - 1] = True
                arrays["work"][schedule_index, refinement - 1] = float(parent.parent_work)
                for future_index in range(2):
                    branches = []
                    for word_index in range(9):
                        cell_id = f"{root.root_id}.future.s{schedule_index}.r{refinement}.f{future_index}.w{word_index}"
                        phase = acquire_phase(phase_name="future", allocation=root, refinement=refinement, source_sha256=source_sha, incoming=parent, schedule_index=schedule_index, future_index=future_index, word_index=word_index, frame=frame, progress=progress)
                        cells[cell_id] = CurrentPreparationNativeCell(cell_id, phase.disposition, retain(cell_id, phase), phase.reason)
                        branches.append(phase)
                    hold = branches[0]
                    for pair in range(4):
                        rows = []
                        for word_index in (2 * pair + 1, 2 * pair + 2):
                            phase = branches[word_index]
                            if phase.disposition != "COMPLETE" or hold.disposition != "COMPLETE":
                                rows = []
                                break
                            if phase.incoming_sha256 != parent.fingerprint() or phase.streams != hold.streams or phase.innovation_sha256 != hold.innovation_sha256:
                                raise ValueError("Current response changes its actual same-purpose paired HOLD")
                            observed, known = reduce_native_response_arrays(frame=frame, word=WORDS[word_index], handoff_tick=4496, readouts=(192,), future=response_arrays(phase), paired_hold=response_arrays(hold))
                            if not known[0, :5].all():
                                rows = []
                                break
                            rows.append(observed[0, :5])
                        if len(rows) != 2:
                            reason = "INCOMPLETE_PAIRED_RESPONSE"
                            continue
                        minus, plus = rows
                        value = np.r_[(plus[:2] - minus[:2]) / 2, plus[2:], minus[2:]]
                        arrays["y"][schedule_index, pair, :, future_index, refinement - 1] = value
                        arrays["response_observed"][schedule_index, pair, :, future_index, refinement - 1] = np.isfinite(value)
    complete = all(np.isfinite(value).all() for name, value in arrays.items() if name not in ("response_observed", "parent_complete")) and all(cell.disposition == "COMPLETE" for cell in cells.values())
    if not complete and reason is None:
        reason = "INCOMPLETE_REQUIRED_CURRENT_ARRAYS"
    report = CurrentPreparationRootReport(root.root_id, ObjectIdentity.from_record(root.root_id, root), config.identity, tuple(cells[cell_id] for cell_id in sorted(cells)), "COMPLETE" if complete else "UNEVALUABLE", None if complete else reason)
    for value in arrays.values():
        value.flags.writeable = False
    return CurrentPreparationRootArrays(report, MappingProxyType(arrays))
