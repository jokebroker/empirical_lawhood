"""Native reduction to finite scalar operands, with explicit paired-HOLD lineage.

Only this instrument/projection boundary sees native arrays. Method learners
consume the scalar output; qualification and outcome adjudication stay at
their separate scientific owners. No acquisition or task-HOLD roster is added.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.prepared_response.contracts import DIRECTIONS, READOUTS, PreparedForceWord, PreparedRoot
from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame, prepared_mechanism_sketch, prepared_observation_history, prepared_receiver, prepared_training_observation
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedCommonStart, PreparedNativeHandoff, PreparedNativePhaseData
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import state_from_checkpoint

Array = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def _freeze(value: Array) -> Array:
    return np.frombuffer(np.asarray(value, dtype="<f8").tobytes(), dtype="<f8").reshape(value.shape)


def _handoff_lineage(common: PreparedCommonStart, handoff: PreparedNativeHandoff) -> None:
    d = handoff.delivery
    if (
        d.common_start != ObjectIdentity.from_record(common.common_start_id, common)
        or d.root != common.root
        or d.source_spec != common.checkpoints[d.refinement - 1].source_spec
        or d.incoming_checkpoint
        != ObjectIdentity.from_record(
            common.checkpoints[d.refinement - 1].checkpoint_id,
            common.checkpoints[d.refinement - 1],
        )
        or common.frame is None
    ):
        raise ValueError("projection handoff is detached from its resolved common start")


def _future_lineage(
    common: PreparedCommonStart,
    handoff: PreparedNativeHandoff,
    future: PreparedNativePhaseData,
) -> None:
    _handoff_lineage(common, handoff)
    d, parent = future.delivery, handoff.delivery
    initial, _ = state_from_checkpoint(handoff.checkpoint.native)
    if (
        d.phase != "future"
        or d.root != parent.root
        or d.parent != parent.parent
        or d.refinement != parent.refinement
        or d.source_spec != parent.source_spec
        or d.common_start != parent.common_start
        or d.incoming_checkpoint
        != ObjectIdentity.from_record(handoff.checkpoint.checkpoint_id, handoff.checkpoint)
        or not np.array_equal(future.positions[0], initial.positions)
        or not np.array_equal(future.momenta[0], initial.momenta)
    ):
        raise ValueError("projection future changes its exact parent handoff/source/view lineage")


@dataclass(frozen=True, slots=True)
class PreparedHandoffObservation:
    root: PreparedRoot
    common_start: ObjectIdentity
    handoff: ObjectIdentity
    refinement: int
    instrument_tier: str
    history: Array
    sketch: Array | None
    preparent_displacement: Array
    parent_absolute_density_work: float

    def __post_init__(self) -> None:
        if (
            self.common_start.object_schema != PreparedCommonStart.SCHEMA
            or self.handoff.object_schema != PreparedNativeHandoff.SCHEMA
            or type(self.refinement) is not int
            or self.refinement not in (1, 2)
            or not np.isfinite(self.parent_absolute_density_work)
            or self.parent_absolute_density_work < 0
            or self.instrument_tier not in ("I0", "I1")
            or (self.sketch is None) != (self.instrument_tier == "I0")
        ):
            raise ValueError("handoff observation loses its native lineage or measured parent work")
        for name, shape in (("history", (16, 12)), ("preparent_displacement", (2,))):
            value = getattr(self, name)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("handoff instrument has an unresolved scalar channel")
            object.__setattr__(self, name, _freeze(value))
        if self.sketch is not None:
            if self.sketch.shape != (8,) or not np.isfinite(self.sketch).all():
                raise ValueError("I1 handoff instrument has an unresolved scalar sketch")
            object.__setattr__(self, "sketch", _freeze(self.sketch))


def project_prepared_handoff(
    common: PreparedCommonStart, handoff: PreparedNativeHandoff, *, instrument_tier: str
) -> PreparedHandoffObservation:
    if instrument_tier not in ("I0", "I1"):
        raise ValueError("handoff projection requires its predeclared I0 or I1 instrument tier")
    _handoff_lineage(common, handoff)
    frame, checkpoint = common.frame, handoff.checkpoint
    assert frame is not None
    positions = _decode(checkpoint.history_positions_base64, (31, 2, 3, 4, 4))
    momenta = _decode(checkpoint.history_momenta_base64, (31, 2, 3, 4, 4))
    native, _ = state_from_checkpoint(checkpoint.native)
    origin, _ = state_from_checkpoint(common.checkpoints[handoff.delivery.refinement - 1].native)
    return PreparedHandoffObservation(
        common.root,
        ObjectIdentity.from_record(common.common_start_id, common),
        ObjectIdentity.from_record(handoff.handoff_id, handoff),
        handoff.delivery.refinement,
        instrument_tier,
        prepared_observation_history(
            frame=frame,
            ticks=checkpoint.history_ticks,
            positions=positions,
            momenta=momenta,
            cutoff_tick=common.root.handoff,
        ),
        prepared_mechanism_sketch(native, frame) if instrument_tier == "I1" else None,
        prepared_receiver(frame, native.positions[0] - origin.positions[0]),
        float(handoff.delivery.parent_absolute_density_work),
    )


@dataclass(frozen=True, slots=True)
class PreparedFiniteObservation:
    root: PreparedRoot
    common_start: ObjectIdentity
    handoff: ObjectIdentity
    delivery: ObjectIdentity
    matched_hold_delivery: ObjectIdentity | None
    word: PreparedForceWord
    refinement: int
    outputs: Array  # readout, receiver two / preservation three / signed and absolute work
    known: BoolArray
    delivery_complete: bool

    def __post_init__(self) -> None:
        if (
            self.outputs.shape != (5, 7)
            or self.known.shape != (5, 7)
            or self.known.dtype != np.dtype("bool")
            or not np.isfinite(self.outputs[self.known]).all()
            or not np.isnan(self.outputs[~self.known]).all()
            or type(self.delivery_complete) is not bool
            or type(self.refinement) is not int
            or self.refinement not in (1, 2)
            or self.common_start.object_schema != PreparedCommonStart.SCHEMA
            or self.handoff.object_schema != PreparedNativeHandoff.SCHEMA
            or self.delivery.object_schema != 'empirical-lawhood/simulators/prepared-response/prepared-native-delivery'
            or (self.matched_hold_delivery is None and self.known[:, 2:5].any())
        ):
            raise ValueError("finite native observation changes its missingness or finite chart")
        if (
            self.matched_hold_delivery is not None
            and self.matched_hold_delivery.object_schema != self.delivery.object_schema
        ):
            raise ValueError("preservation reference is not an authenticated native delivery")
        object.__setattr__(self, "outputs", _freeze(self.outputs))
        object.__setattr__(
            self, "known", np.frombuffer(self.known.tobytes(), dtype="bool").reshape(5, 7)
        )


def project_prepared_future(
    common: PreparedCommonStart,
    handoff: PreparedNativeHandoff,
    future: PreparedNativePhaseData,
    *,
    matched_hold: PreparedNativePhaseData | None,
) -> PreparedFiniteObservation:
    _future_lineage(common, handoff, future)
    frame, word = common.frame, future.delivery.word
    assert frame is not None and word is not None
    paired = False
    if matched_hold is not None:
        _future_lineage(common, handoff, matched_hold)
        hold = matched_hold.delivery
        if hold.word is None or hold.word.sign != 0:
            raise ValueError("preservation reference must be an actual native HOLD")
        if hold.innovation_streams != future.delivery.innovation_streams:
            raise ValueError("preservation cannot substitute an independent-noise audit HOLD")
        paired = hold.disposition == future.delivery.disposition == "COMPLETE"
        if paired and hold.innovation_sha256 != future.delivery.innovation_sha256:
            raise ValueError("matched HOLD and action disagree on their realized innovations")
    outputs, known = reduce_native_response_arrays(
        frame=frame,
        word=word,
        handoff_tick=common.root.handoff,
        readouts=READOUTS,
        future=NativeResponseArrays(
            future.ticks,
            future.positions,
            future.transfer,
            future.transfer_known,
            float(future.delivery.signed_force_work),
            float(future.delivery.absolute_force_work),
        ),
        paired_hold=None
        if not paired or matched_hold is None
        else NativeResponseArrays(
            matched_hold.ticks,
            matched_hold.positions,
            matched_hold.transfer,
            matched_hold.transfer_known,
            float(matched_hold.delivery.signed_force_work),
            float(matched_hold.delivery.absolute_force_work),
        ),
    )
    return PreparedFiniteObservation(
        common.root,
        ObjectIdentity.from_record(common.common_start_id, common),
        ObjectIdentity.from_record(handoff.handoff_id, handoff),
        ObjectIdentity.from_record(future.delivery.occurrence_id, future.delivery),
        None
        if matched_hold is None
        else ObjectIdentity.from_record(matched_hold.delivery.occurrence_id, matched_hold.delivery),
        word,
        future.delivery.refinement,
        outputs,
        known,
        future.delivery.disposition == "COMPLETE",
    )


@dataclass(frozen=True, slots=True)
class NativeResponseArrays:
    """Already authenticated native operands; callers retain all pairing authority."""

    ticks: npt.NDArray[np.int64]
    positions: npt.NDArray[np.complex128]
    transfer: Array
    transfer_known: BoolArray
    signed_force_work: float
    absolute_force_work: float


def reduce_native_response_arrays(
    *,
    frame: PreparedPortFrame,
    word: PreparedForceWord,
    handoff_tick: int,
    readouts: tuple[int, ...],
    future: NativeResponseArrays,
    paired_hold: NativeResponseArrays | None,
) -> tuple[Array, BoolArray]:
    """Single owner of native displacement, preservation and work reduction.

    Versioned callers must prove source/root/frame, handoff and same-purpose
    innovation joins first. Supplying a HOLD here is not such a proof.
    """
    if (
        type(handoff_tick) is not int
        or handoff_tick < 0
        or not readouts
        or any(type(t) is not int or t < 64 or t % 16 for t in readouts)
        or tuple(sorted(set(readouts))) != readouts
    ):
        raise ValueError("native response reduction changes its post-pulse readout clocks")
    if paired_hold is not None and not np.array_equal(future.ticks, paired_hold.ticks):
        raise ValueError("native paired response arrays change the shared observation clock")
    mode = frame.modes[0]
    if word.sign:
        direction = np.array(DIRECTIONS[word.direction_index], dtype=float)
        mode = np.einsum("p,pabc->abc", direction / np.linalg.norm(direction), frame.modes)
    outputs = np.full((len(readouts), 7), np.nan)
    known = np.zeros((len(readouts), 7), dtype=bool)
    for row, tick in enumerate(readouts):
        indices = np.flatnonzero(future.ticks == handoff_tick + tick)
        if len(indices) != 1:
            continue
        end = int(indices[0])
        outputs[row, :2] = prepared_receiver(
            frame, future.positions[end, 0] - future.positions[0, 0]
        )
        # All readouts are after the 64-tick force pulse, so retained force-work
        # totals are already final even if later force-off dynamics failed.
        outputs[row, 5:] = (
            future.signed_force_work,
            future.absolute_force_work,
        )
        known[row, (0, 1, 5, 6)] = True
        if paired_hold is None:
            continue
        delta = future.positions[: end + 1] - paired_hold.positions[: end + 1]
        along = np.einsum("abc,sabc->s", mode.conj(), delta[:, 0]).real
        off = delta[:, 0] - along[:, None, None, None] * mode
        outputs[row, 2] = np.max(np.linalg.norm(off.reshape(end + 1, -1), axis=1))
        outputs[row, 3] = np.max(np.linalg.norm(delta[:, 1].reshape(end + 1, -1), axis=1)) / max(
            1.0, float(np.linalg.norm(paired_hold.positions[0, 1]))
        )
        known[row, 2:4] = True
        if (
            future.transfer_known[2 : end + 1].all()
            and paired_hold.transfer_known[2 : end + 1].all()
        ):
            differences = future.transfer[2 : end + 1] - paired_hold.transfer[2 : end + 1]
            outputs[row, 4] = np.max(np.linalg.norm(differences, ord=2, axis=(1, 2)))
            known[row, 4] = True
    return outputs, known


def project_prepared_response_development_transitions(
    common: PreparedCommonStart,
    handoff: PreparedNativeHandoff,
    future: PreparedNativePhaseData,
) -> Array:
    _future_lineage(common, handoff, future)
    if common.root.stage != 'development':
        raise ValueError("future scalar transition labels are available only from dependent refinement")
    frame, checkpoint = common.frame, handoff.checkpoint
    assert frame is not None
    positions = np.concatenate(
        (_decode(checkpoint.history_positions_base64, (31, 2, 3, 4, 4)), future.positions[1:])
    )
    momenta = np.concatenate(
        (_decode(checkpoint.history_momenta_base64, (31, 2, 3, 4, 4)), future.momenta[1:])
    )
    ticks = (*checkpoint.history_ticks, *(int(tick) for tick in future.ticks[1:]))
    result = np.full((21, 12), np.nan)
    for row in range(len(future.ticks)):
        result[row] = prepared_training_observation(
            root=common.root,
            frame=frame,
            ticks=ticks[row : row + 31],
            positions=positions[row : row + 31],
            momenta=momenta[row : row + 31],
            cutoff_tick=int(future.ticks[row]),
        )
    return _freeze(result)
