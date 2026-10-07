"Finite response-law clocks and checkpoints around the owned prepared-response marcher.\n\nThese mechanics grant no authority. The installed task provider must authenticate\nall inputs and reserve the declared effect before calling a native phase.\n"

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.simulators.prepared_response.contracts import prepared_native_member, prepared_numerical_view
from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame, prepared_parent_schedule, select_prepared_ports
from empirical_lawhood.adapters.simulators.prepared_response.source import NativeStreamSwitch, PreparedNativeCheckpoint, _Branch, march_native_intervals
from empirical_lawhood.adapters.simulators.six_matrix_response.model import SixMatrixState, ideal_state
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import derive_probe_roster
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import ResponseGeometryNativePassiveObserver, ResponseGeometryNativeProbeCheckpoint, _decode, _encode
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.adapters.methods.finite_response_law.science import passive_probe_seed_for

from .assigned_contracts import FiniteResponseLawAssignedCalibrationInvocation, FiniteResponseLawAssignedEvaluationInvocation
from .contracts import FiniteResponseLawNativeConfig, FiniteResponseLawNativeInvocation, native_invocations
from .evaluation_contracts import FiniteResponseLawEvaluationInvocation
from .fresh_contracts import FiniteResponseLawCalibrationInvocation
from .randomness import FiniteResponseLawCalibrationRNGStream, FiniteResponseLawEvaluationRNGStream, FiniteResponseLawRNGStream, canonical_rng_state, native_rng, restore_rng


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeDelivery(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-native-delivery'
    INVOCATION_TYPE: ClassVar[type[FiniteResponseLawNativeInvocation]] = FiniteResponseLawNativeInvocation
    invocation: FiniteResponseLawNativeInvocation
    refinement: int
    incoming_checkpoint: ObjectIdentity | None
    frozen_frame_sha256: str | None
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
    innovation_streams: tuple[FiniteResponseLawRNGStream, ...]
    disposition: str
    reason: str | None
    accepted: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        if type(self.invocation) is not self.INVOCATION_TYPE:
            raise ValueError("Finite response-law delivery requires its exact versioned invocation")
        start, end = self.invocation.clocks
        future = self.invocation.phase == "future"
        if self.accepted is not True:
            raise ValueError(
                "Finite response-law nonaccepted invocation must remain explicitly unentered"
            )
        if type(self.refinement) is not int or self.refinement not in (1, 2):
            raise ValueError("Finite response-law native delivery has an undeclared numerical view")
        if self.invocation.phase == "prefix":
            if (
                self.incoming_checkpoint is not None
                or self.frozen_frame_sha256 is not None
            ):
                raise ValueError("Finite response-law fresh prefix cannot import a checkpoint/frame")
        else:
            expected = (
                PreparedNativeCheckpoint.SCHEMA
                if self.invocation.root.cohort == "retained-prepared-response"
                else FiniteResponseLawAssignedEvaluationCheckpoint.SCHEMA
                if type(self.invocation) is FiniteResponseLawAssignedEvaluationInvocation
                else FiniteResponseLawAssignedCalibrationCheckpoint.SCHEMA
                if type(self.invocation) is FiniteResponseLawAssignedCalibrationInvocation
                else FiniteResponseLawEvaluationCheckpoint.SCHEMA
                if type(self.invocation) is FiniteResponseLawEvaluationInvocation
                else FiniteResponseLawCalibrationCheckpoint.SCHEMA
                if type(self.invocation) is FiniteResponseLawCalibrationInvocation
                else FiniteResponseLawNativeCheckpoint.SCHEMA
            )
            if (
                self.incoming_checkpoint is None
                or self.incoming_checkpoint.object_schema != expected
            ):
                raise ValueError(
                    "Finite response-law continuation lacks its authentic native predecessor"
                )
            if self.frozen_frame_sha256 is None:
                raise ValueError("Finite response-law continuation lacks its primary pre-parent frame")
            validate_sha256(self.frozen_frame_sha256, field_name="frozen_frame_sha256")
        for name in ("realized_trace_sha256", "innovation_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            any(
                type(v) is not int
                for v in (
                    self.completed_intervals,
                    self.applied_force_kicks,
                    self.nonzero_force_intervals,
                )
            )
            or not 0 <= self.completed_intervals <= (end - start) * self.refinement
            or self.applied_force_kicks
            != (2 * self.completed_intervals if future else 0)
            or not 0
            <= self.nonzero_force_intervals
            <= (min(self.completed_intervals, 64 * self.refinement) if future else 0)
            or len(self.realized_impulse) != 2
            or any(
                not isinstance(v, Decimal) or not v.is_finite()
                for v in (
                    *self.realized_impulse,
                    self.signed_force_work,
                    self.absolute_force_work,
                    self.squared_input_integral,
                    self.parent_absolute_density_work,
                )
            )
            or min(
                self.absolute_force_work,
                self.squared_input_integral,
                self.parent_absolute_density_work,
            )
            < 0
            or self.absolute_force_work < abs(self.signed_force_work)
        ):
            raise ValueError(
                "Finite response-law native delivery changes applied clock/action/work accounting"
            )
        if self.disposition not in (
            "COMPLETE",
            "NUMERICAL_FAILURE",
            "OBSERVATION_FAILURE",
        ) or (self.disposition == "COMPLETE") != (self.reason is None):
            raise ValueError(
                "Finite response-law native delivery loses its explicit terminal disposition"
            )
        if (
            self.disposition == "COMPLETE"
            and self.completed_intervals != (end - start) * self.refinement
        ):
            raise ValueError("Finite response-law complete delivery omits native updates")
        expected_streams = streams_for(self.invocation)
        allowed_streams: tuple[tuple[FiniteResponseLawRNGStream, ...], ...] = (expected_streams,)
        if self.invocation.phase == "prefix":
            boundary = 256 * self.refinement
            if self.completed_intervals < boundary:
                allowed_streams = (expected_streams[:2],)
            elif self.completed_intervals == boundary:
                allowed_streams = (expected_streams[:2], expected_streams)
        if self.innovation_streams not in allowed_streams:
            raise ValueError(
                "Finite response-law native delivery changes committed innovations or numerical coupling"
            )

    @property
    def occurrence_id(self) -> str:
        return f"{self.invocation.task_id}.r{self.refinement}"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeCheckpoint(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-native-checkpoint'
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawNativeDelivery]] = FiniteResponseLawNativeDelivery
    delivery: FiniteResponseLawNativeDelivery
    passive: ResponseGeometryNativeProbeCheckpoint
    history_ticks: tuple[int, ...]
    history_positions_base64: str
    history_momenta_base64: str
    alpha_x: Decimal
    alpha_y: Decimal
    rng_state_json: str
    bridge_rng_state_json: str

    def __post_init__(self) -> None:
        delivery = self.delivery
        if type(delivery) is not self.DELIVERY_TYPE:
            raise ValueError("Finite response-law checkpoint requires its exact versioned delivery")
        end = delivery.invocation.clocks[1]
        expected = tuple(range(end - 480, end + 1, 16))
        if (
            delivery.disposition != "COMPLETE"
            or self.passive.refinement != delivery.refinement
            or self.passive.native_step != end * delivery.refinement
            or self.history_ticks != expected
            or any(type(t) is not int for t in self.history_ticks)
            or not all(
                isinstance(v, Decimal) and v.is_finite()
                for v in (self.alpha_x, self.alpha_y)
            )
            or abs(float(self.alpha_x) - 2 / 3) > 1e-12
            or abs(float(self.alpha_y) - 22 / 3) > 1e-12
        ):
            raise ValueError(
                "Finite response-law checkpoint changes its completed native clock, baseline or history"
            )
        positions = _decode(self.history_positions_base64, (31, 2, 3, 4, 4))
        momenta = _decode(self.history_momenta_base64, (31, 2, 3, 4, 4))
        if (
            not np.isfinite(positions).all()
            or not np.isfinite(momenta).all()
            or not np.array_equal(
                positions[-1, 1], _decode(self.passive.y_base64, (3, 4, 4))
            )
        ):
            raise ValueError("Finite response-law native and passive checkpoints disagree")
        self.restore()
        restore_rng(self.rng_state_json)
        restore_rng(self.bridge_rng_state_json)
        if len(self.canonical_bytes()) > 512 * 1024:
            raise ValueError("Finite response-law native checkpoint exceeds its declared bound")

    @property
    def checkpoint_id(self) -> str:
        return f"{self.delivery.occurrence_id}.checkpoint"

    def restore(self) -> _Branch:
        positions = _decode(self.history_positions_base64, (31, 2, 3, 4, 4))
        momenta = _decode(self.history_momenta_base64, (31, 2, 3, 4, 4))
        state = SixMatrixState(
            2,
            positions[-1],
            momenta[-1],
            self.passive.native_step,
            float(self.alpha_x),
            float(self.alpha_y),
        )
        return _Branch(
            state,
            ResponseGeometryNativePassiveObserver.restore(self.passive),
            list(self.history_ticks),
            list(positions),
            list(momenta),
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationDelivery(FiniteResponseLawNativeDelivery):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-calibration-delivery'
    )
    VERSION: ClassVar[str] = '1.0.0'
    INVOCATION_TYPE: ClassVar[type[FiniteResponseLawNativeInvocation]] = FiniteResponseLawCalibrationInvocation
    invocation: FiniteResponseLawCalibrationInvocation
    innovation_streams: tuple[FiniteResponseLawCalibrationRNGStream, ...]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationCheckpoint(FiniteResponseLawNativeCheckpoint):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-calibration-checkpoint'
    )
    VERSION: ClassVar[str] = '1.0.0'
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawNativeDelivery]] = FiniteResponseLawCalibrationDelivery
    delivery: FiniteResponseLawCalibrationDelivery


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationDelivery(FiniteResponseLawNativeDelivery):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-evaluation-delivery'
    )
    INVOCATION_TYPE: ClassVar[type[FiniteResponseLawNativeInvocation]] = FiniteResponseLawEvaluationInvocation
    invocation: FiniteResponseLawEvaluationInvocation
    innovation_streams: tuple[FiniteResponseLawEvaluationRNGStream, ...]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationCheckpoint(FiniteResponseLawNativeCheckpoint):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-evaluation-checkpoint'
    )
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawNativeDelivery]] = FiniteResponseLawEvaluationDelivery
    delivery: FiniteResponseLawEvaluationDelivery


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationDelivery(FiniteResponseLawNativeDelivery):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-calibration-delivery'
    )
    VERSION: ClassVar[str] = '1.0.0'
    INVOCATION_TYPE: ClassVar[type[FiniteResponseLawNativeInvocation]] = (
        FiniteResponseLawAssignedCalibrationInvocation
    )
    invocation: FiniteResponseLawAssignedCalibrationInvocation
    innovation_streams: tuple[FiniteResponseLawCalibrationRNGStream, ...]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationCheckpoint(FiniteResponseLawNativeCheckpoint):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-calibration-checkpoint'
    )
    VERSION: ClassVar[str] = '1.0.0'
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawNativeDelivery]] = (
        FiniteResponseLawAssignedCalibrationDelivery
    )
    delivery: FiniteResponseLawAssignedCalibrationDelivery


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationDelivery(FiniteResponseLawNativeDelivery):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-evaluation-delivery'
    )
    VERSION: ClassVar[str] = '1.0.0'
    INVOCATION_TYPE: ClassVar[type[FiniteResponseLawNativeInvocation]] = (
        FiniteResponseLawAssignedEvaluationInvocation
    )
    invocation: FiniteResponseLawAssignedEvaluationInvocation
    innovation_streams: tuple[FiniteResponseLawEvaluationRNGStream, ...]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationCheckpoint(FiniteResponseLawNativeCheckpoint):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-evaluation-checkpoint'
    )
    VERSION: ClassVar[str] = '1.0.0'
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawNativeDelivery]] = FiniteResponseLawAssignedEvaluationDelivery
    delivery: FiniteResponseLawAssignedEvaluationDelivery


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativePhaseData:
    delivery: FiniteResponseLawNativeDelivery
    checkpoint: FiniteResponseLawNativeCheckpoint | None
    ticks: npt.NDArray[np.int64]
    positions: npt.NDArray[np.complex128]
    momenta: npt.NDArray[np.complex128]
    couplings: npt.NDArray[np.float64]
    transfer: npt.NDArray[np.float64]
    transfer_known: npt.NDArray[np.bool_]
    realized_trace: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        d = self.delivery
        start, end = d.invocation.clocks
        expected = np.arange(
            start, start + d.completed_intervals // d.refinement + 1, 16, dtype=np.int64
        )
        if not (
            np.array_equal(self.ticks, expected)
            or d.disposition == "OBSERVATION_FAILURE"
            and np.array_equal(self.ticks, expected[:-1])
        ):
            raise ValueError("Finite response-law observations change the native reference clock")
        rows = len(self.ticks)
        shapes = {
            "ticks": ((rows,), "<i8"),
            "positions": ((rows, 2, 3, 4, 4), "<c16"),
            "momenta": ((rows, 2, 3, 4, 4), "<c16"),
            "couplings": ((rows, 2), "<f8"),
            "transfer": ((rows, 15, 15), "<f8"),
            "transfer_known": ((rows,), "bool"),
            "realized_trace": ((d.completed_intervals, 9), "<f8"),
        }
        for name, (shape, dtype) in shapes.items():
            value = getattr(self, name)
            if (
                value.shape != shape
                or value.dtype != np.dtype(dtype)
                or (name != "transfer" and not np.isfinite(value).all())
            ):
                raise ValueError(
                    f'Finite response-law native {name} changes bounded geometry or finite operands'
                )
            object.__setattr__(
                self, name, np.frombuffer(value.tobytes(), dtype=dtype).reshape(shape)
            )
        if (
            not np.isfinite(self.transfer[self.transfer_known]).all()
            or sha256(self.realized_trace.tobytes()).hexdigest()
            != d.realized_trace_sha256
        ):
            raise ValueError(
                "Finite response-law known transfer or realized trace differs from its delivery"
            )
        if (self.checkpoint is not None) != (d.disposition == "COMPLETE"):
            raise ValueError("Finite response-law failed phase cannot supply a continuation checkpoint")
        if self.checkpoint is not None:
            state = self.checkpoint.restore().state
            if (
                self.checkpoint.delivery != d
                or self.ticks[-1] != end
                or not np.array_equal(state.positions, self.positions[-1])
                or not np.array_equal(state.momenta, self.momenta[-1])
                or not np.allclose(
                    self.couplings[-1],
                    (state.alpha_tilde_x, state.alpha_tilde_y),
                    rtol=0,
                    atol=1e-14,
                )
            ):
                raise ValueError(
                    "Finite response-law checkpoint differs from its delivered terminal observation"
                )


def streams_for(invocation: FiniteResponseLawNativeInvocation) -> tuple[FiniteResponseLawRNGStream, ...]:
    key = (
        invocation.root.stage_unit
        if isinstance(invocation, FiniteResponseLawCalibrationInvocation)
        else "canary.r000"
        if invocation.root.cohort == "native-canary"
        else f"supplemental-development.r{invocation.root.index:03d}"
    )
    names = (
        ("coarse", "bridge", "post-ramp-coarse", "post-ramp-bridge")
        if invocation.phase == "prefix"
        else ("coarse", "bridge")
    )
    committed_seed = dict(getattr(invocation.root, "scientific_seeds", ())).get(invocation.purpose)
    return tuple(native_rng(key, invocation.purpose, name, committed_seed=committed_seed) for name in names)


def frozen_prefix_frame(
    checkpoint: FiniteResponseLawNativeCheckpoint,
) -> PreparedPortFrame | None:
    if (
        checkpoint.delivery.invocation.phase != "prefix"
        or checkpoint.delivery.refinement != 1
    ):
        raise ValueError("Finite response-law frame must be selected from the actual primary prefix")
    history = _decode(checkpoint.history_positions_base64, (31, 2, 3, 4, 4))
    return select_prepared_ports(
        ticks=checkpoint.history_ticks[-16:],
        observations=history[-16:, 0],
        cutoff_tick=4096,
    )


def execute_native_phase(
    config: FiniteResponseLawNativeConfig,
    invocation: FiniteResponseLawNativeInvocation,
    refinement: int,
    *,
    incoming: FiniteResponseLawNativeCheckpoint | PreparedNativeCheckpoint | None,
    frame: PreparedPortFrame | None,
    progress: Callable[[int], None] | None = None,
) -> FiniteResponseLawNativePhaseData:
    """Execute only an authenticated, effect-reserved native invocation.

    The provider owns exact retained receipt and common-start/frame joins. This
    wrapper repeats root, clock, source and branch joins before native updates.
    """
    if (
        refinement not in (1, 2)
        or type(refinement) is not int
        or invocation not in native_invocations(config)
    ):
        raise ValueError("Finite response-law native entry differs from its complete frozen census")
    start, end = invocation.clocks
    streams = streams_for(invocation)
    if invocation.phase == "prefix":
        if incoming is not None or frame is not None:
            raise ValueError("Finite response-law fresh canary cannot import a state or frame")
        state = ideal_state(
            q=2, alpha_tilde_x=8.0, alpha_tilde_y=8.0, constitution="11"
        )
        probes = native_rng(streams[0].stage_unit, "prefix", "passive-probes", committed_seed=int(streams[0].committed_seed_decimal))
        roster = derive_probe_roster(
            config_fingerprint=probes.initial_state_sha256,
            rule_id="finite-response-law.passive-probes",
            scientific_seed=passive_probe_seed_for(streams[0].stage_unit, committed_seed=dict(getattr(invocation.root, "scientific_seeds", ())).get("passive-probes")),
        )
        branch = _Branch(
            state,
            ResponseGeometryNativePassiveObserver(
                roster=roster,
                y=state.positions[1],
                y_velocity=state.momenta[1],
                refinement=refinement,
            ),
            [0],
            [state.positions],
            [state.momenta],
        )
    elif invocation.root.cohort == "retained-prepared-response":
        if (
            not isinstance(incoming, PreparedNativeCheckpoint)
            or config.retained_source is None
            or incoming.root != invocation.root.retained_root
            or incoming.source_spec
            != ObjectIdentity.from_record(
                config.retained_source.spec_id, config.retained_source
            )
            or incoming.passive.refinement != refinement
            or incoming.native.step_index != start * refinement
            or incoming.checkpoint_id
            != f"{incoming.root.root_id}.{invocation.parent}.parent.r{refinement}.checkpoint"
        ):
            raise ValueError(
                "Finite response-law continuation changes its retained prepared-response root/parent/native clock"
            )
        branch = _Branch.restore(incoming)
    else:
        if (
            not isinstance(incoming, FiniteResponseLawNativeCheckpoint)
            or incoming.delivery.invocation.source != invocation.source
            or incoming.delivery.invocation.root != invocation.root
            or incoming.delivery.refinement != refinement
            or incoming.delivery.invocation.phase
            != ("prefix" if invocation.phase == "parent" else "parent")
            or incoming.passive.native_step != start * refinement
            or invocation.phase == "future"
            and incoming.delivery.invocation.parent != invocation.parent
        ):
            raise ValueError("Finite response-law canary continuation changes its native predecessor")
        branch = incoming.restore()
    if start and (frame is None or frame.cutoff_tick != 4096):
        raise ValueError("Finite response-law continuation requires its exact pre-parent frame")
    schedule = None
    switches: tuple[NativeStreamSwitch[FiniteResponseLawRNGStream], ...] = ()
    if invocation.phase == "prefix":
        schedule = np.asarray(
            [
                [
                    8.0 + min(1.0, (step + 1) / (256 * refinement)) * (target - 8.0)
                    for target in (2 / 3, 22 / 3)
                ]
                for step in range(end * refinement)
            ]
        )
        switches = (
            (
                256,
                streams[2].generator_instance(),
                streams[2],
                streams[3].generator_instance(),
                streams[3],
            ),
        )
    elif invocation.phase == "parent":
        assert invocation.parent is not None
        schedule = prepared_parent_schedule(
            parent=invocation.parent, refinement=refinement
        )
    raw = march_native_intervals(
        branch=branch,
        start=start,
        end=end,
        refinement=refinement,
        member=prepared_native_member(),
        view=prepared_numerical_view(refinement),
        schedule=schedule,
        word=invocation.word,
        frame=frame,
        rng=streams[0].generator_instance(),
        stream=streams[0],
        bridge=streams[1].generator_instance(),
        bridge_stream=streams[1],
        stream_switches=switches,
        accumulate_parent_work=invocation.phase == "parent",
        progress=progress,
    )
    delivery_type: type[FiniteResponseLawNativeDelivery] = (
        FiniteResponseLawAssignedEvaluationDelivery
        if type(invocation) is FiniteResponseLawAssignedEvaluationInvocation
        else FiniteResponseLawAssignedCalibrationDelivery
        if type(invocation) is FiniteResponseLawAssignedCalibrationInvocation
        else FiniteResponseLawEvaluationDelivery
        if type(invocation) is FiniteResponseLawEvaluationInvocation
        else FiniteResponseLawCalibrationDelivery
        if type(invocation) is FiniteResponseLawCalibrationInvocation
        else FiniteResponseLawNativeDelivery
    )
    delivery = delivery_type(
        invocation,
        refinement,
        None
        if incoming is None
        else ObjectIdentity.from_record(incoming.checkpoint_id, incoming),
        None if frame is None else sha256(frame.modes.tobytes()).hexdigest(),
        raw.completed,
        2 * raw.completed if invocation.phase == "future" else 0,
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
        checkpoint_type: type[FiniteResponseLawNativeCheckpoint] = (
            FiniteResponseLawAssignedEvaluationCheckpoint
            if type(delivery) is FiniteResponseLawAssignedEvaluationDelivery
            else FiniteResponseLawAssignedCalibrationCheckpoint
            if type(delivery) is FiniteResponseLawAssignedCalibrationDelivery
            else FiniteResponseLawEvaluationCheckpoint
            if type(delivery) is FiniteResponseLawEvaluationDelivery
            else FiniteResponseLawCalibrationCheckpoint
            if type(delivery) is FiniteResponseLawCalibrationDelivery
            else FiniteResponseLawNativeCheckpoint
        )
        checkpoint = checkpoint_type(
            delivery,
            branch.observer.checkpoint(),
            tuple(branch.ticks),
            _encode(np.asarray(branch.positions, dtype="<c16")),
            _encode(np.asarray(branch.momenta, dtype="<c16")),
            Decimal(str(branch.state.alpha_tilde_x)),
            Decimal(str(branch.state.alpha_tilde_y)),
            canonical_rng_state(raw.rng),
            canonical_rng_state(raw.bridge),
        )
    return FiniteResponseLawNativePhaseData(
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
