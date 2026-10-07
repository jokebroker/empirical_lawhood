"""preparation-policy native wrappers around the single prepared prepared response marcher."""

from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.simulators.prepared_response.contracts import prepared_native_member, prepared_numerical_view
from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedNativeCheckpoint, _Branch, march_native_intervals
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256

from .randomness import FiniteResponseLawPreparationPolicyRandomStream, canonical_rng_state, native_rng
from .source import FiniteResponseLawNativeCheckpoint, FiniteResponseLawNativePhaseData
from .preparation_policy_contracts import FiniteResponseLawPreparationPolicyNativeConfig, FiniteResponseLawPreparationPolicyNativeInvocation, preparation_policy_native_invocations
from .preparation_policy_instruments import preparation_policy_preparation_schedule


def preparation_policy_streams(
    invocation: FiniteResponseLawPreparationPolicyNativeInvocation,
) -> tuple[FiniteResponseLawPreparationPolicyRandomStream, FiniteResponseLawPreparationPolicyRandomStream]:
    values = tuple(
        native_rng(invocation.root.stage_unit, invocation.purpose, name)
        for name in ("coarse", "bridge")
    )
    if not all(type(value) is FiniteResponseLawPreparationPolicyRandomStream for value in values):
        raise ValueError("preparation-policy invocation resolved a non-preparation-policy random stream")
    return values  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyNativeDelivery(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-preparation-policy-native-delivery'
    invocation: FiniteResponseLawPreparationPolicyNativeInvocation
    refinement: int
    incoming_checkpoint: ObjectIdentity
    frozen_frame_sha256: str
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
    innovation_streams: tuple[FiniteResponseLawPreparationPolicyRandomStream, ...]
    disposition: str
    reason: str | None
    accepted: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        start, end = self.invocation.clocks
        future = self.invocation.phase == "future"
        expected_schema = (
            FiniteResponseLawPreparationPolicyNativeCheckpoint.SCHEMA
            if future
            else PreparedNativeCheckpoint.SCHEMA
        )
        if (
            self.accepted is not True
            or type(self.refinement) is not int
            or self.refinement not in (1, 2)
            or self.incoming_checkpoint.object_schema != expected_schema
        ):
            raise ValueError("preparation-policy delivery changes entry, view or predecessor schema")
        validate_sha256(self.frozen_frame_sha256, field_name="frozen_frame_sha256")
        validate_sha256(self.realized_trace_sha256, field_name="realized_trace_sha256")
        validate_sha256(self.innovation_sha256, field_name="innovation_sha256")
        values = (
            *self.realized_impulse,
            self.signed_force_work,
            self.absolute_force_work,
            self.squared_input_integral,
            self.parent_absolute_density_work,
        )
        if (
            len(self.realized_impulse) != 2
            or any(not isinstance(value, Decimal) or not value.is_finite() for value in values)
            or min(
                self.absolute_force_work,
                self.squared_input_integral,
                self.parent_absolute_density_work,
            )
            < 0
            or self.absolute_force_work < abs(self.signed_force_work)
            or self.completed_intervals != (end - start) * self.refinement
            and self.disposition == "COMPLETE"
            or not 0 <= self.completed_intervals <= (end - start) * self.refinement
            or self.applied_force_kicks != (2 * self.completed_intervals if future else 0)
            or not 0
            <= self.nonzero_force_intervals
            <= (min(self.completed_intervals, 64 * self.refinement) if future else 0)
            or self.innovation_streams != preparation_policy_streams(self.invocation)
            or self.disposition
            not in ("COMPLETE", "NUMERICAL_FAILURE", "OBSERVATION_FAILURE")
            or (self.disposition == "COMPLETE") != (self.reason is None)
        ):
            raise ValueError("preparation-policy delivery changes its clock/action/work accounting")

    @property
    def occurrence_id(self) -> str:
        return f"{self.invocation.task_id}.r{self.refinement}"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyNativeCheckpoint(FiniteResponseLawNativeCheckpoint):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-preparation-policy-native-checkpoint'
    DELIVERY_TYPE: ClassVar[type[FiniteResponseLawPreparationPolicyNativeDelivery]] = FiniteResponseLawPreparationPolicyNativeDelivery  # type: ignore[assignment]
    delivery: FiniteResponseLawPreparationPolicyNativeDelivery  # type: ignore[assignment]


def execute_preparation_policy_native_phase(
    config: FiniteResponseLawPreparationPolicyNativeConfig,
    invocation: FiniteResponseLawPreparationPolicyNativeInvocation,
    refinement: int,
    *,
    incoming: PreparedNativeCheckpoint | FiniteResponseLawPreparationPolicyNativeCheckpoint,
    frame: PreparedPortFrame,
    progress: Callable[[int], None] | None = None,
) -> FiniteResponseLawNativePhaseData:
    if (
        type(refinement) is not int
        or refinement not in (1, 2)
        or invocation not in preparation_policy_native_invocations(config)
        or frame.cutoff_tick != 4096
    ):
        raise ValueError("preparation-policy native entry differs from its frozen census/frame")
    start, _ = invocation.clocks
    if invocation.phase == "preparation":
        if (
            type(incoming) is not PreparedNativeCheckpoint
            or invocation.root.retained_prefix is None
            or incoming.root.root_id != invocation.root.root_id
            or incoming.source_spec != invocation.root.retained_prefix.source_spec
            or incoming.passive.refinement != refinement
            or incoming.native.step_index != start * refinement
        ):
            raise ValueError("preparation-policy preparation changes its retained prefix/view/clock")
        branch = _Branch.restore(incoming)
        assert invocation.schedule is not None
        schedule = preparation_policy_preparation_schedule(invocation.schedule, refinement=refinement)
    else:
        if (
            type(incoming) is not FiniteResponseLawPreparationPolicyNativeCheckpoint
            or incoming.delivery.invocation.source != invocation.source
            or incoming.delivery.invocation.root != invocation.root
            or incoming.delivery.invocation.schedule != invocation.schedule
            or incoming.delivery.invocation.phase != "preparation"
            or incoming.delivery.refinement != refinement
            or incoming.passive.native_step != start * refinement
        ):
            raise ValueError("preparation-policy future changes its exact preparation predecessor")
        branch = incoming.restore()
        schedule = None
    streams = preparation_policy_streams(invocation)
    raw = march_native_intervals(
        branch=branch,
        start=invocation.clocks[0],
        end=invocation.clocks[1],
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
        accumulate_parent_work=invocation.phase == "preparation",
        force_parent_ticks=400,
        progress=progress,
    )
    delivery = FiniteResponseLawPreparationPolicyNativeDelivery(
        invocation,
        refinement,
        ObjectIdentity.from_record(incoming.checkpoint_id, incoming),
        sha256(frame.modes.tobytes()).hexdigest(),
        raw.completed,
        2 * raw.completed if invocation.phase == "future" else 0,
        raw.nonzero,
        tuple(Decimal(str(float(value))) for value in raw.impulse),
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
        from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _encode

        checkpoint = FiniteResponseLawPreparationPolicyNativeCheckpoint(
            delivery,
            branch.observer.checkpoint(),
            tuple(branch.ticks),
            _encode(np.asarray(raw.branch.positions, dtype="<c16")),
            _encode(np.asarray(raw.branch.momenta, dtype="<c16")),
            Decimal(str(branch.state.alpha_tilde_x)),
            Decimal(str(branch.state.alpha_tilde_y)),
            canonical_rng_state(raw.rng),
            canonical_rng_state(raw.bridge),
        )
    return FiniteResponseLawNativePhaseData(
        delivery,  # type: ignore[arg-type]
        checkpoint,
        raw.ticks,
        raw.positions,
        raw.momenta,
        raw.couplings,
        raw.transfer,
        raw.transfer_known,
        raw.realized_trace,
    )
