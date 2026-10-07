"""Native phase mechanics for a registered prepared-response source provider.

No filesystem, scheduler, authority, model fitting, command selection or
scientific adjudication is owned here. The provider must reserve each declared
effect before calling these mechanics and publish its external task receipt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from typing import ClassVar, Generic, TypeVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView
from empirical_lawhood.adapters.simulators.six_matrix_response.gradients import SixMatrixParameters, coupling_derivative_density
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ComplexArray, SixMatrixState, ideal_state
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import derive_probe_roster
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import ResponseGeometryNativePassiveObserver, ResponseGeometryNativeProbeCheckpoint, PassiveReadout, _decode, _encode
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import brownian_bridge_split
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import BAOABGradientCache, SixMatrixResponseCheckpoint, SixMatrixResponseRNGStreamReceipt, baoab_step_with_hermitian_noise, checkpoint_from_phase_state, derive_rng_stream, hermitian_noise, state_from_checkpoint

from .contracts import CAMPAIGN, PARENTS, RNG_RULE, PreparedForceWord, PreparedNativeSpec, PreparedRoot, validate_prepared_future_role
from .instruments import PreparedPortFrame, prepared_force_components, prepared_force_step, prepared_parent_schedule, prepared_receiver, select_prepared_ports

Array = npt.NDArray[np.float64]
StreamRecord = TypeVar("StreamRecord", bound=CanonicalRecord)
NativeStreamSwitch = tuple[
    int, np.random.Generator, StreamRecord, np.random.Generator, StreamRecord
]
PURPOSES = (
    "initial-ramp",
    "initial-post-ramp",
    "parent",
    'common-response',
    'independent-response-1',
    'independent-response-2',
    'independent-response-3',
    'prospective-task',
    "passive-probes",
)


def prepared_rng(
    root: PreparedRoot, purpose: str, *, bridge: bool = False
) -> tuple[np.random.Generator, SixMatrixResponseRNGStreamReceipt]:
    if (
        purpose not in PURPOSES
        or type(bridge) is not bool
        or (purpose == "passive-probes" and bridge)
    ):
        raise ValueError("prepared source RNG purpose is outside its frozen separation rule")
    return derive_rng_stream(
        seed_root_id=f"{CAMPAIGN}.seed.{root.seed_sha256}",
        purpose_id=f"{root.physical_unit_id}.{purpose}{'.bridge' if bridge else ''}",
        stream_index=0,
        derivation_rule_id=RNG_RULE,
        scientific_seed_sha256=root.randomness.stream_seed(purpose, bridge),
    )


@dataclass(frozen=True, slots=True)
class PreparedNativeCheckpoint(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-native-checkpoint'
    checkpoint_id: str
    source_spec: ObjectIdentity
    root: PreparedRoot
    native: SixMatrixResponseCheckpoint
    passive: ResponseGeometryNativeProbeCheckpoint
    history_ticks: tuple[int, ...]
    history_positions_base64: str
    history_momenta_base64: str
    bridge_rng_state_json: str

    def __post_init__(self) -> None:
        validate_stable_id(self.checkpoint_id, field_name="checkpoint_id")
        if self.source_spec.object_schema != PreparedNativeSpec.SCHEMA:
            raise ValueError("prepared checkpoint must bind its new native specification")
        refinement = self.passive.refinement
        tick, remainder = divmod(self.native.step_index, refinement)
        if (
            refinement not in (1, 2)
            or remainder
            or self.native.step_index != self.passive.native_step
            or self.native.q != 2
            or self.native.receivers
            or tick > self.root.handoff + 320
        ):
            raise ValueError("prepared checkpoint must retain a complete native/observer interval")
        latest = tick // 16 * 16
        expected = tuple(range(max(0, latest - 480), latest + 1, 16))
        if self.history_ticks != expected or any(type(t) is not int for t in self.history_ticks):
            raise ValueError(
                "prepared checkpoint lost its thirty-one-sample causal instrument history"
            )
        positions = _decode(self.history_positions_base64, (len(expected), 2, 3, 4, 4))
        momenta = _decode(self.history_momenta_base64, (len(expected), 2, 3, 4, 4))
        state, _ = state_from_checkpoint(self.native)
        if not state.finite or not np.array_equal(
            state.positions[1], _decode(self.passive.y_base64, (3, 4, 4))
        ):
            raise ValueError(
                "prepared observer checkpoint is detached from its finite native state"
            )
        if tick == latest and (
            not np.array_equal(positions[-1], state.positions)
            or not np.array_equal(momenta[-1], state.momenta)
        ):
            raise ValueError("prepared sampled state differs from its exact checkpoint boundary")
        if len(self.bridge_rng_state_json) > 1024:
            raise ValueError("prepared numerical bridge RNG state exceeds its bound")
        document = json.loads(self.bridge_rng_state_json)
        if (
            not isinstance(document, dict)
            or document.get("bit_generator") != "PCG64DXSM"
            or json.dumps(document, sort_keys=True, separators=(",", ":"))
            != self.bridge_rng_state_json
        ):
            raise ValueError("prepared numerical bridge requires canonical PCG64DXSM state")
        generator = np.random.Generator(np.random.PCG64DXSM(0))
        generator.bit_generator.state = document
        if len(self.canonical_bytes()) > 512 * 1024:
            raise ValueError("prepared checkpoint exceeds its 512-KiB bound")


@dataclass(frozen=True, slots=True)
class PreparedCommonStart(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-common-start'
    common_start_id: str
    root: PreparedRoot
    checkpoints: tuple[PreparedNativeCheckpoint, ...]
    frozen_ports_base64: str | None
    mode_disposition: str

    def __post_init__(self) -> None:
        validate_stable_id(self.common_start_id, field_name="common_start_id")
        if (
            len(self.checkpoints) != 2
            or tuple(c.passive.refinement for c in self.checkpoints) != (1, 2)
            or any(
                c.root != self.root
                or c.native.step_index != self.root.landmark * c.passive.refinement
                for c in self.checkpoints
            )
            or self.checkpoints[0].source_spec != self.checkpoints[1].source_spec
            or self.mode_disposition not in ("RESOLVED", "NONATTEMPT_UNRESOLVED_PORT_FRAME")
            or (self.frozen_ports_base64 is None) != (self.mode_disposition != "RESOLVED")
        ):
            raise ValueError(
                "prepared common start changes its paired pre-parent checkpoint/frame topology"
            )
        primary = self.checkpoints[0]
        history = _decode(primary.history_positions_base64, (31, 2, 3, 4, 4))
        expected = select_prepared_ports(
            ticks=primary.history_ticks[-16:],
            observations=history[-16:, 0],
            cutoff_tick=self.root.landmark,
        )
        if (expected is None) != (self.frozen_ports_base64 is None):
            raise ValueError(
                "prepared common-start mode disposition differs from the causal primary instrument"
            )
        if expected is not None and self.frozen_ports_base64 != _encode(expected.modes):
            raise ValueError(
                "prepared common-start ports differ from the frozen primary nomination"
            )

    @property
    def frame(self) -> PreparedPortFrame | None:
        return (
            None
            if self.frozen_ports_base64 is None
            else PreparedPortFrame(
                self.root.landmark, _decode(self.frozen_ports_base64, (2, 3, 4, 4))
            )
        )


def bind_prepared_common_start(
    primary: PreparedNativeCheckpoint, half: PreparedNativeCheckpoint
) -> PreparedCommonStart:
    history = _decode(primary.history_positions_base64, (31, 2, 3, 4, 4))
    frame = select_prepared_ports(
        ticks=primary.history_ticks[-16:],
        observations=history[-16:, 0],
        cutoff_tick=primary.root.landmark,
    )
    return PreparedCommonStart(
        f"{primary.root.root_id}.common-start",
        primary.root,
        (primary, half),
        None if frame is None else _encode(frame.modes),
        "NONATTEMPT_UNRESOLVED_PORT_FRAME" if frame is None else "RESOLVED",
    )


@dataclass(frozen=True, slots=True)
class PreparedNativeDelivery(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-native-delivery'
    occurrence_id: str
    source_spec: ObjectIdentity
    common_start: ObjectIdentity | None
    incoming_checkpoint: ObjectIdentity | None
    root: PreparedRoot
    phase: str
    parent: str | None
    word: PreparedForceWord | None
    refinement: int
    start_tick: int
    requested_end_tick: int
    accepted: bool
    completed_intervals: int
    applied_force_kicks: int
    nonzero_force_intervals: int
    realized_impulse: tuple[Decimal, ...]
    signed_force_work: Decimal
    absolute_force_work: Decimal
    squared_input_integral: Decimal
    parent_absolute_density_work: Decimal
    realized_trace_sha256: str
    innovation_sha256: str
    innovation_streams: tuple[SixMatrixResponseRNGStreamReceipt, ...]
    disposition: str
    reason: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.occurrence_id, field_name="occurrence_id")
        if self.source_spec.object_schema != PreparedNativeSpec.SCHEMA:
            raise ValueError("prepared delivery lacks its exact source specification")
        if self.phase == "prefix":
            if self.common_start is not None or self.incoming_checkpoint is not None:
                raise ValueError("a fresh prefix cannot import a prepared state")
        elif (
            self.common_start is None
            or self.common_start.object_schema != PreparedCommonStart.SCHEMA
            or self.incoming_checkpoint is None
            or self.incoming_checkpoint.object_schema != PreparedNativeCheckpoint.SCHEMA
        ):
            raise ValueError(
                "prepared continuation lacks its common-start and incoming checkpoint identities"
            )
        for name in ("realized_trace_sha256", "innovation_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            self.phase not in ("prefix", "parent", "future")
            or type(self.refinement) is not int
            or self.refinement not in (1, 2)
        ):
            raise ValueError("prepared delivery phase/view is outside its native route")
        start, end = {
            "prefix": (0, self.root.landmark),
            "parent": (self.root.landmark, self.root.handoff),
            "future": (self.root.handoff, self.root.handoff + 320),
        }[self.phase]
        if (
            self.start_tick != start
            or self.requested_end_tick != end
            or any(
                type(v) is not int
                for v in (
                    self.start_tick,
                    self.requested_end_tick,
                    self.completed_intervals,
                    self.applied_force_kicks,
                    self.nonzero_force_intervals,
                )
            )
            or not 0 <= self.completed_intervals <= (end - start) * self.refinement
            or self.applied_force_kicks
            != (2 * self.completed_intervals if self.phase == "future" else 0)
            or not 0
            <= self.nonzero_force_intervals
            <= min(self.completed_intervals, 64 * self.refinement)
            or type(self.accepted) is not bool
            or self.phase == "prefix"
            and (self.parent is not None or self.word is not None)
            or self.phase == "parent"
            and (self.parent not in PARENTS or self.word is not None)
            or self.phase == "future"
            and (self.parent not in PARENTS or self.word is None)
        ):
            raise ValueError(
                "prepared delivery changes requested/accepted/applied interval accounting"
            )
        if len(self.realized_impulse) != 2:
            raise ValueError("prepared delivery must retain both native force impulses")
        for value in (
            *self.realized_impulse,
            self.signed_force_work,
            self.absolute_force_work,
            self.squared_input_integral,
            self.parent_absolute_density_work,
        ):
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError("prepared delivery requires finite raw effort operands")
        if min(
            self.absolute_force_work, self.squared_input_integral, self.parent_absolute_density_work
        ) < 0 or self.absolute_force_work < abs(self.signed_force_work):
            raise ValueError("prepared signed and absolute work accounting is inconsistent")
        if self.disposition not in ("COMPLETE", "NUMERICAL_FAILURE", "OBSERVATION_FAILURE") or (
            self.disposition == "COMPLETE"
        ) != (self.reason is None):
            raise ValueError("prepared native delivery must retain the actual terminal reason")
        if self.disposition == "COMPLETE" and (
            not self.accepted or self.completed_intervals != (end - start) * self.refinement
        ):
            raise ValueError("prepared COMPLETE delivery has missing native intervals")


@dataclass(frozen=True, slots=True)
class PreparedNativePhaseData:
    """Bounded in-memory native output, serialized by the source artifact owner."""

    delivery: PreparedNativeDelivery
    checkpoint: PreparedNativeCheckpoint | None
    ticks: npt.NDArray[np.int64]
    positions: ComplexArray
    momenta: ComplexArray
    couplings: Array
    transfer: Array
    transfer_known: npt.NDArray[np.bool_]
    realized_trace: Array

    def __post_init__(self) -> None:
        delivery = self.delivery
        count = delivery.completed_intervals
        expected_ticks = np.arange(
            delivery.start_tick,
            delivery.start_tick + count // delivery.refinement + 1,
            16,
            dtype=np.int64,
        )
        # A failed observer may omit the just-completed boundary sample.
        if not (
            np.array_equal(self.ticks, expected_ticks)
            or delivery.disposition == "OBSERVATION_FAILURE"
            and np.array_equal(self.ticks, expected_ticks[:-1])
        ):
            raise ValueError("prepared native data loses or changes its observed reference clock")
        rows = len(self.ticks)
        shapes = {
            "positions": (rows, 2, 3, 4, 4),
            "momenta": (rows, 2, 3, 4, 4),
            "couplings": (rows, 2),
            "transfer": (rows, 15, 15),
            "transfer_known": (rows,),
            "realized_trace": (count, 9),
        }
        for name, shape in shapes.items():
            value = getattr(self, name)
            dtype = (
                "complex128"
                if name in ("positions", "momenta")
                else "bool"
                if name == "transfer_known"
                else "float64"
            )
            if value.shape != shape or value.dtype != np.dtype(dtype):
                raise ValueError(f"prepared native {name} has another bounded data geometry")
            if name != "transfer" and not np.isfinite(value).all():
                raise ValueError(f"prepared native {name} contains a nonfinite retained operand")
            object.__setattr__(
                self, name, np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(shape)
            )
        if not np.isfinite(self.transfer[self.transfer_known]).all():
            raise ValueError("prepared native resolved transfer contains nonfinite values")
        if (
            self.ticks.dtype != np.dtype("int64")
            or sha256(self.realized_trace.tobytes()).hexdigest() != delivery.realized_trace_sha256
        ):
            raise ValueError("prepared native delivery trace bytes differ from its receipt")
        object.__setattr__(self, "ticks", np.frombuffer(self.ticks.tobytes(), dtype=np.int64))
        if (
            self.checkpoint is not None
            and self.checkpoint.native.request
            != ObjectIdentity.from_record(delivery.occurrence_id, delivery)
        ):
            raise ValueError("prepared native data checkpoint is detached from its delivery")
        if (self.checkpoint is not None) != (delivery.disposition == "COMPLETE"):
            raise ValueError("only a complete native phase can retain a reusable checkpoint")
        if self.checkpoint is not None:
            checkpoint = self.checkpoint
            state, _ = state_from_checkpoint(checkpoint.native)
            if (
                checkpoint.root != delivery.root
                or checkpoint.source_spec != delivery.source_spec
                or checkpoint.passive.refinement != delivery.refinement
                or state.step_index != delivery.requested_end_tick * delivery.refinement
                or not np.array_equal(state.positions, self.positions[-1])
                or not np.array_equal(state.momenta, self.momenta[-1])
                or not np.allclose(
                    self.couplings[-1],
                    (state.alpha_tilde_x, state.alpha_tilde_y),
                    rtol=0,
                    atol=1e-14,
                )
            ):
                raise ValueError("native phase checkpoint disagrees with its final sampled state")


@dataclass(frozen=True, slots=True)
class PreparedNativeHandoff(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-native-handoff'
    handoff_id: str
    delivery: PreparedNativeDelivery
    checkpoint: PreparedNativeCheckpoint

    def __post_init__(self) -> None:
        validate_stable_id(self.handoff_id, field_name="handoff_id")
        delivery, checkpoint = self.delivery, self.checkpoint
        if (
            delivery.phase != "parent"
            or delivery.disposition != "COMPLETE"
            or checkpoint.native.request
            != ObjectIdentity.from_record(delivery.occurrence_id, delivery)
            or checkpoint.source_spec != delivery.source_spec
            or checkpoint.root != delivery.root
            or checkpoint.passive.refinement != delivery.refinement
            or checkpoint.native.step_index != delivery.root.handoff * delivery.refinement
            or abs(float(checkpoint.native.alpha_tilde_x) - 2 / 3) > 1e-12
            or abs(float(checkpoint.native.alpha_tilde_y) - 22 / 3) > 1e-12
        ):
            raise ValueError(
                "prepared handoff is not the returned exact parent delivery/checkpoint"
            )


def bind_prepared_native_handoff(result: PreparedNativePhaseData) -> PreparedNativeHandoff:
    if result.checkpoint is None:
        raise ValueError("a failed native parent has no completed handoff")
    return PreparedNativeHandoff(
        f"{result.delivery.occurrence_id}.handoff", result.delivery, result.checkpoint
    )


@dataclass
class _Branch:
    state: SixMatrixState
    observer: ResponseGeometryNativePassiveObserver
    ticks: list[int]
    positions: list[ComplexArray]
    momenta: list[ComplexArray]

    @classmethod
    def restore(cls, checkpoint: PreparedNativeCheckpoint) -> _Branch:
        state, _ = state_from_checkpoint(checkpoint.native)
        count = len(checkpoint.history_ticks)
        return cls(
            state,
            ResponseGeometryNativePassiveObserver.restore(checkpoint.passive),
            list(checkpoint.history_ticks),
            list(_decode(checkpoint.history_positions_base64, (count, 2, 3, 4, 4))),
            list(_decode(checkpoint.history_momenta_base64, (count, 2, 3, 4, 4))),
        )


def _initial_branch(root: PreparedRoot, refinement: int) -> _Branch:
    coupling = 8.0 if root.context == "prepared" else 0.0
    state = ideal_state(
        q=2,
        alpha_tilde_x=coupling,
        alpha_tilde_y=coupling,
        constitution="11" if root.context == "prepared" else "00",
    )
    _, stream = prepared_rng(root, "passive-probes")
    roster = derive_probe_roster(
        config_fingerprint=stream.derived_seed_sha256, rule_id=f"{CAMPAIGN}.passive-probes",
        scientific_seed=int(root.randomness.passive_probe_seed_sha256, 16),
        field_id_prefix=f"{CAMPAIGN}.passive-probe.field",
    )
    observer = ResponseGeometryNativePassiveObserver(
        roster=roster, y=state.positions[1], y_velocity=state.momenta[1], refinement=refinement
    )
    return _Branch(state, observer, [0], [state.positions], [state.momenta])


def _source_join(spec: PreparedNativeSpec, root: PreparedRoot, refinement: int) -> None:
    if (
        root.stage != spec.stage
        or root.seed_sha256 != spec.seed_sha256
        or root.randomness != spec.root_seed_census[
            root.index + (0 if root.context == "assembling" else len(spec.root_seed_census) // 2)
        ]
        or type(refinement) is not int
        or refinement not in (1, 2)
    ):
        raise ValueError(
            "prepared source request differs from its issued stage/root/view declaration"
        )


def prepare_native_prefix(
    spec: PreparedNativeSpec,
    root: PreparedRoot,
    *,
    refinement: int,
    progress: Callable[[int], None] | None = None,
) -> PreparedNativePhaseData:
    _source_join(spec, root, refinement)
    return _advance(
        spec,
        root,
        refinement,
        _initial_branch(root, refinement),
        phase="prefix",
        parent=None,
        word=None,
        frame=None,
        purpose="initial-ramp",
        common_start=None,
        incoming_checkpoint=None,
        progress=progress,
    )


def execute_native_parent(
    spec: PreparedNativeSpec,
    common: PreparedCommonStart,
    *,
    parent: str,
    refinement: int,
    progress: Callable[[int], None] | None = None,
) -> PreparedNativePhaseData:
    _source_join(spec, common.root, refinement)
    checkpoint = common.checkpoints[refinement - 1]
    if (
        checkpoint.source_spec != ObjectIdentity.from_record(spec.spec_id, spec)
        or parent not in PARENTS
        or common.frame is None
    ):
        raise ValueError(
            "prepared parent requires the frozen source/common-start and resolved pre-parent frame"
        )
    return _advance(
        spec,
        common.root,
        refinement,
        _Branch.restore(checkpoint),
        phase="parent",
        parent=parent,
        word=None,
        frame=common.frame,
        purpose="parent",
        common_start=ObjectIdentity.from_record(common.common_start_id, common),
        incoming_checkpoint=ObjectIdentity.from_record(checkpoint.checkpoint_id, checkpoint),
        progress=progress,
    )


def execute_native_future(
    spec: PreparedNativeSpec,
    common: PreparedCommonStart,
    handoff: PreparedNativeHandoff,
    *,
    parent: str,
    word: PreparedForceWord,
    purpose: str,
    progress: Callable[[int], None] | None = None,
) -> PreparedNativePhaseData:
    checkpoint = handoff.checkpoint
    refinement = checkpoint.passive.refinement
    _source_join(spec, common.root, refinement)
    if (
        checkpoint.source_spec != ObjectIdentity.from_record(spec.spec_id, spec)
        or checkpoint.root != common.root
        or handoff.delivery.common_start
        != ObjectIdentity.from_record(common.common_start_id, common)
        or handoff.delivery.parent != parent
        or handoff.delivery.incoming_checkpoint
        != ObjectIdentity.from_record(
            common.checkpoints[refinement - 1].checkpoint_id, common.checkpoints[refinement - 1]
        )
        or parent not in PARENTS
        or word not in spec.words
        or purpose not in ('common-response', 'independent-response-1', 'independent-response-2', 'independent-response-3', 'prospective-task')
        or common.frame is None
    ):
        raise ValueError(
            "prepared future differs from its fixed source/handoff/chart/future assignment"
        )
    validate_prepared_future_role(spec.stage, word, purpose)
    return _advance(
        spec,
        common.root,
        refinement,
        _Branch.restore(checkpoint),
        phase="future",
        parent=parent,
        word=word,
        frame=common.frame,
        purpose=purpose,
        common_start=ObjectIdentity.from_record(common.common_start_id, common),
        incoming_checkpoint=ObjectIdentity.from_record(checkpoint.checkpoint_id, checkpoint),
        progress=progress,
    )


@dataclass(frozen=True, slots=True)
class NativeMarchResult(Generic[StreamRecord]):
    """Raw mechanics only; versioned wrappers own identities and checkpoints."""

    branch: _Branch
    rng: np.random.Generator
    stream: StreamRecord
    bridge: np.random.Generator
    streams: tuple[StreamRecord, ...]
    completed: int
    nonzero: int
    impulse: Array
    signed_work: float
    absolute_work: float
    squared_input: float
    parent_work: float
    innovation_sha256: str
    disposition: str
    reason: str | None
    ticks: npt.NDArray[np.int64]
    positions: ComplexArray
    momenta: ComplexArray
    couplings: Array
    transfer: Array
    transfer_known: npt.NDArray[np.bool_]
    realized_trace: Array


def march_native_intervals(
    *,
    branch: _Branch,
    start: int,
    end: int,
    refinement: int,
    member: SixMatrixResponseModelFamilyMember,
    view: SixMatrixResponseNumericalView,
    schedule: Array | None,
    word: PreparedForceWord | None,
    frame: PreparedPortFrame | None,
    rng: np.random.Generator,
    stream: StreamRecord,
    bridge: np.random.Generator,
    bridge_stream: StreamRecord,
    stream_switches: tuple[NativeStreamSwitch[StreamRecord], ...] = (),
    accumulate_parent_work: bool = False,
    force_parent_ticks: int = 272,
    progress: Callable[[int], None] | None = None,
) -> NativeMarchResult[StreamRecord]:
    """Single native marcher with explicit clocks, schedules and noise streams.

    No root census, source identity, checkpoint schema or execution authority is
    implied here. Installed providers and versioned wrappers retain those duties.
    """
    if (
        any(type(t) is not int for t in (start, end, refinement))
        or refinement not in (1, 2)
        or start < 0
        or end <= start
        or start % 16
        or end % 16
        or branch.state.step_index != start * refinement
        or float(view.timestep) != 0.001 / refinement
        or type(accumulate_parent_work) is not bool
        or type(force_parent_ticks) is not int
        or force_parent_ticks <= 0
    ):
        raise ValueError("native march cannot reset or shift its incoming native/view clock")
    if word is None:
        if (
            schedule is None
            or schedule.shape != ((end - start) * refinement, 2)
            or schedule.dtype != np.dtype("float64")
            or not np.isfinite(schedule).all()
        ):
            raise ValueError("native unforced march requires its complete finite coupling schedule")
    elif frame is None or schedule is not None or accumulate_parent_work:
        raise ValueError("native forced march requires its frozen frame and no parent schedule")
    switches = {row[0]: row[1:] for row in stream_switches}
    if len(switches) != len(stream_switches) or any(
        type(t) is not int or not start < t < end for t in switches
    ):
        raise ValueError("native noise switches must be unique interior reference ticks")
    streams = [stream, bridge_stream]
    cache = BAOABGradientCache()
    rows: list[tuple[int, ComplexArray, ComplexArray, Array, Array, bool]] = []
    trace: list[tuple[float, ...]] = []
    innovation_digest = sha256()
    impulse = np.zeros(2)
    signed_work = absolute_work = squared_input = parent_work = 0.0
    completed = nonzero = 0
    disposition, reason = "COMPLETE", None

    def sample(readout: PassiveReadout | None) -> None:
        state = branch.state
        transfer = (
            np.full((15, 15), np.nan)
            if readout is None
            else _decode(readout.propagator_base64, (15, 15), real=True)
        )
        known = (
            readout is not None
            and readout.transfer_resolved
            and readout.origin_tick == state.step_index // refinement - 32
            and readout.evidence_ready_tick == state.step_index // refinement
        )
        rows.append(
            (
                state.step_index // refinement,
                state.positions,
                state.momenta,
                np.array([state.alpha_tilde_x, state.alpha_tilde_y]),
                transfer,
                known,
            )
        )

    sample(None)
    for tick in range(start, end):
        if tick in switches:
            rng, stream, bridge, bridge_stream = switches[tick]
            streams.extend((stream, bridge_stream))
        coarse = hermitian_noise(rng=rng, q=2)
        extra = hermitian_noise(rng=bridge, q=2)
        halves = brownian_bridge_split(
            coarse_noise=coarse, bridge_noise=extra, half_decay=float(np.exp(-0.0005))
        )
        noises = (coarse,) if refinement == 1 else halves
        for noise in noises:
            before = branch.state
            components = np.zeros(2)
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    if word is not None:
                        assert word is not None and frame is not None
                        components = prepared_force_components(
                            word,
                            native_step=before.step_index,
                            invocation_tick=start,
                            refinement=refinement,
                        )
                        after = prepared_force_step(
                            before,
                            word=word,
                            frame=frame,
                            invocation_tick=start,
                            refinement=refinement,
                            member=member,
                            numerical_view=view,
                            standardized_noise=noise,
                            gradient_cache=cache,
                            expected_parent_ticks=force_parent_ticks,
                        )
                        work_increment = float(
                            np.dot(
                                components,
                                prepared_receiver(
                                    frame, after.positions[0] - before.positions[0]
                                ),
                            )
                        )
                        next_signed, next_absolute = (
                            signed_work + work_increment,
                            absolute_work + abs(work_increment),
                        )
                        next_parent = parent_work
                    else:
                        assert schedule is not None
                        alpha = schedule[before.step_index - start * refinement]
                        after = baoab_step_with_hermitian_noise(
                            before,
                            member=member,
                            numerical_view=view,
                            next_alpha_tilde_x=float(alpha[0]),
                            next_alpha_tilde_y=float(alpha[1]),
                            standardized_noise=noise,
                            gradient_cache=cache,
                        )
                        next_signed, next_absolute, next_parent = (
                            signed_work,
                            absolute_work,
                            parent_work,
                        )
                        if accumulate_parent_work:
                            derivatives = coupling_derivative_density(
                                before.positions,
                                SixMatrixParameters(
                                    2, 0.5, 0.5, 1, before.alpha_tilde_x, before.alpha_tilde_y
                                ),
                            )
                            next_parent += abs(
                                derivatives[0] * (after.alpha_tilde_x - before.alpha_tilde_x)
                            ) + abs(derivatives[1] * (after.alpha_tilde_y - before.alpha_tilde_y))
                    if (
                        not after.finite
                        or not np.isfinite((next_signed, next_absolute, next_parent)).all()
                    ):
                        raise FloatingPointError("nonfinite native state or work")
                    branch.state = after
                    signed_work, absolute_work, parent_work = (
                        next_signed,
                        next_absolute,
                        next_parent,
                    )
                    impulse += components * float(view.timestep)
                    squared_input += float(np.dot(components, components)) * float(view.timestep)
                    nonzero += int(bool(np.any(components)))
                    completed += 1
                    innovation_digest.update(noise.tobytes())
                    trace.append(
                        (
                            float(before.step_index),
                            before.alpha_tilde_x,
                            before.alpha_tilde_y,
                            after.alpha_tilde_x,
                            after.alpha_tilde_y,
                            float(components[0]),
                            float(components[1]),
                            float(components[0]),
                            float(components[1]),
                        )
                    )
            except (FloatingPointError, np.linalg.LinAlgError):
                disposition, reason = "NUMERICAL_FAILURE", "NATIVE_STATE_OR_WORK_UNRESOLVED"
                break
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    readout = branch.observer.advance(
                        y=after.positions[1],
                        native_step=after.step_index,
                        y_velocity=after.momenta[1],
                    )
                if after.step_index % (16 * refinement) == 0:
                    branch.ticks.append(after.step_index // refinement)
                    branch.positions.append(after.positions)
                    branch.momenta.append(after.momenta)
                    branch.ticks, branch.positions, branch.momenta = (
                        branch.ticks[-31:],
                        branch.positions[-31:],
                        branch.momenta[-31:],
                    )
                    sample(readout)
            except (FloatingPointError, np.linalg.LinAlgError):
                disposition, reason = "OBSERVATION_FAILURE", "NATIVE_PASSIVE_OBSERVER_UNRESOLVED"
                break
        if progress is not None and completed % 256 == 0:
            progress(completed)
        if disposition != "COMPLETE":
            break
    raw_trace = np.asarray(trace, dtype="<f8").reshape((-1, 9))
    if progress is not None:
        progress(completed)
    return NativeMarchResult(
        branch,
        rng,
        stream,
        bridge,
        tuple(streams),
        completed,
        nonzero,
        impulse,
        signed_work,
        absolute_work,
        squared_input,
        parent_work,
        innovation_digest.hexdigest(),
        disposition,
        reason,
        np.asarray([r[0] for r in rows], dtype=np.int64),
        np.asarray([r[1] for r in rows]),
        np.asarray([r[2] for r in rows]),
        np.asarray([r[3] for r in rows]),
        np.asarray([r[4] for r in rows]),
        np.asarray([r[5] for r in rows], dtype=np.bool_),
        raw_trace,
    )


def _advance(
    spec: PreparedNativeSpec,
    root: PreparedRoot,
    refinement: int,
    branch: _Branch,
    *,
    phase: str,
    parent: str | None,
    word: PreparedForceWord | None,
    frame: PreparedPortFrame | None,
    purpose: str,
    common_start: ObjectIdentity | None,
    incoming_checkpoint: ObjectIdentity | None,
    progress: Callable[[int], None] | None,
) -> PreparedNativePhaseData:
    start, end = {
        "prefix": (0, root.landmark),
        "parent": (root.landmark, root.handoff),
        "future": (root.handoff, root.handoff + 320),
    }[phase]
    suffix = (
        phase
        if phase == "prefix"
        else f"{parent}.{phase}"
        if phase == "parent"
        else f"{parent}.{purpose}.{word.word_id if word is not None else 'missing'}"
    )
    occurrence = f"{root.root_id}.{suffix}.r{refinement}"
    rng, stream = prepared_rng(root, purpose)
    bridge, bridge_stream = prepared_rng(root, purpose, bridge=True)
    switches: tuple[NativeStreamSwitch[SixMatrixResponseRNGStreamReceipt], ...] = ()
    schedule = None
    if phase == "parent" and parent is not None:
        schedule = prepared_parent_schedule(parent=parent, refinement=refinement)
    elif phase == "prefix":
        initial = 8.0 if root.context == "prepared" else 0.0
        schedule = np.asarray(
            [
                [
                    initial + min(1.0, (step + 1) / (256 * refinement)) * (2 / 3 - initial),
                    initial + min(1.0, (step + 1) / (256 * refinement)) * (22 / 3 - initial),
                ]
                for step in range(start * refinement, end * refinement)
            ]
        )
        later_rng, later_stream = prepared_rng(root, "initial-post-ramp")
        later_bridge, later_bridge_stream = prepared_rng(root, "initial-post-ramp", bridge=True)
        switches = ((256, later_rng, later_stream, later_bridge, later_bridge_stream),)
    raw = march_native_intervals(
        branch=branch,
        start=start,
        end=end,
        refinement=refinement,
        member=spec.member,
        view=spec.numerical_views[refinement - 1],
        schedule=schedule,
        word=word,
        frame=frame,
        rng=rng,
        stream=stream,
        bridge=bridge,
        bridge_stream=bridge_stream,
        stream_switches=switches,
        accumulate_parent_work=phase == "parent",
        progress=progress,
    )
    delivery = PreparedNativeDelivery(
        occurrence,
        ObjectIdentity.from_record(spec.spec_id, spec),
        common_start,
        incoming_checkpoint,
        root,
        phase,
        parent,
        word,
        refinement,
        start,
        end,
        True,
        raw.completed,
        2 * raw.completed if phase == "future" else 0,
        raw.nonzero,
        tuple(Decimal(str(float(v))) for v in raw.impulse),
        Decimal(str(raw.signed_work)),
        Decimal(str(raw.absolute_work)),
        Decimal(str(raw.squared_input)),
        Decimal(str(raw.parent_work)),
        sha256(raw.realized_trace.tobytes()).hexdigest(),
        raw.innovation_sha256,
        raw.streams,
        raw.disposition,
        raw.reason,
    )
    checkpoint = None
    if raw.disposition == "COMPLETE":
        native = checkpoint_from_phase_state(
            checkpoint_id=f"{occurrence}.native-checkpoint",
            request=ObjectIdentity.from_record(occurrence, delivery),
            total_steps=(root.handoff + 320) * refinement,
            state=branch.state,
            rng=raw.rng,
            rng_stream=raw.stream,
            receivers=(),
        )
        checkpoint = PreparedNativeCheckpoint(
            f"{occurrence}.checkpoint",
            ObjectIdentity.from_record(spec.spec_id, spec),
            root,
            native,
            branch.observer.checkpoint(),
            tuple(branch.ticks),
            _encode(np.asarray(branch.positions, dtype="<c16")),
            _encode(np.asarray(branch.momenta, dtype="<c16")),
            json.dumps(raw.bridge.bit_generator.state, sort_keys=True, separators=(",", ":")),
        )
    return PreparedNativePhaseData(
        delivery,
        checkpoint,
        raw.ticks,
        raw.positions,
        raw.momenta,
        raw.couplings,
        raw.transfer,
        raw.transfer_known,
        raw.realized_trace,
    )
