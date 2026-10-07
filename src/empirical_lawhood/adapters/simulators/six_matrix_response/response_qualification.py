"""Strict outcome-blind Response geometry assay roster and complete-interval segment requests.

These records select native work behind production ports. Construction never
acquires a trajectory, grants authority or supplies an empirical assay terminal.
"""

from dataclasses import dataclass
from decimal import Decimal
from functools import lru_cache
import base64
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .response_checkpoint import ResponseGeometryNativeCheckpoint
from .response_assay import ResponseGeometryNativeForcePulse
from .response_scientific_inputs import ResponseGeometryScientificInputs, require_response_scientific_inputs


PARENTS = ("hold", "x-negative", "x-positive", "y-negative", "y-positive")
INNER_SIGNS = (-1, 0, 1)


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayNativeRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-assay-native-root'

    context: str
    landmark_tick: int
    index: int

    def __post_init__(self) -> None:
        if self.context not in {"prepared", "assembling"}:
            raise ValueError("assay context differs from the two declared preparations")
        if type(self.landmark_tick) is not int or self.landmark_tick not in {1024, 4096}:
            raise ValueError("assay landmark differs from the two frozen landmarks")
        if type(self.index) is not int or not 0 <= self.index < 8:
            raise ValueError("assay cell must contain precisely roots zero through seven")

    @property
    def root_id(self) -> str:
        return f"response-geometry-assay.{self.context}.t{self.landmark_tick}.r{self.index}"

    @property
    def assay(self) -> str:
        return "short-pulse-response" if self.index % 2 == 0 else "extended-pulse-response"

    @property
    def refinements(self) -> tuple[int, ...]:
        return (1, 2, 4) if self.index < 2 else (1, 2)

    @property
    def restart(self) -> bool:
        return self.index in {4, 5}

    @property
    def covariance(self) -> bool:
        return self.index in {6, 7}

    @property
    def pulse_ticks(self) -> int:
        return 64 if self.assay == "short-pulse-response" else 128

    @property
    def horizon_ticks(self) -> int:
        return 320 if self.assay == "short-pulse-response" else 640

    @property
    def invocation_offset(self) -> int:
        return 384


@lru_cache(maxsize=1)
def assay_roots() -> tuple[ResponseGeometryAssayNativeRoot, ...]:
    return tuple(
        ResponseGeometryAssayNativeRoot(context, landmark, index)
        for context in ("assembling", "prepared")
        for landmark in (1024, 4096)
        for index in range(8)
    )


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayNativeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-assay-native-config'

    config_id: str
    design_packet_sha256: str
    seed_root_sha256: str
    roots: tuple[ResponseGeometryAssayNativeRoot, ...]
    scientific_inputs: ResponseGeometryScientificInputs

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(self.design_packet_sha256, field_name="design_packet_sha256")
        validate_sha256(self.seed_root_sha256, field_name="seed_root_sha256")
        require_response_scientific_inputs(
            self.scientific_inputs, config_id=self.config_id,
            design_packet_sha256=self.design_packet_sha256,
            seed_root_sha256=self.seed_root_sha256, roots=self.roots,
        )
        if self.roots != assay_roots():
            raise ValueError("assay config differs from the entire fixed 32-root roster")

    def physical_unit_id(self, root: ResponseGeometryAssayNativeRoot) -> str:
        from .response_panel import assay_physical_unit_id

        return assay_physical_unit_id(self, root)

    @property
    def observation_schema(self) -> str:
        return 'empirical-lawhood/simulators/six-matrix-response/assay-native-observations-hdf5'

    @property
    def result_type(self) -> type['ResponseGeometryAssayNativeSegmentResult']:
        return ResponseGeometryAssayNativeSegmentResult


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayNativeSegment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-assay-native-segment'

    root: ResponseGeometryAssayNativeRoot
    phase: str
    parent: str | None
    sign: int | None

    def __post_init__(self) -> None:
        if self.phase not in {"prefix", "parent", "inner", "resume"}:
            raise ValueError("unknown assay segment phase")
        if self.phase == "prefix":
            if self.parent is not None or self.sign is not None:
                raise ValueError("assay prefix cannot select a parent or pulse")
        elif self.parent not in PARENTS:
            raise ValueError("assay segment has another parent")
        if self.phase == "parent" and self.sign is not None:
            raise ValueError("assay parent cannot consume its future pulse sign")
        if self.phase in {"inner", "resume"}:
            if type(self.sign) is not int or self.sign not in INNER_SIGNS:
                raise ValueError("assay inner sign differs from NEG/HOLD/POS")
        if self.phase == "resume" and not self.root.restart:
            raise ValueError("assay resume is restricted to the predetermined restart subset")

    @property
    def task_id(self) -> str:
        suffix = self.phase
        if self.parent is not None:
            suffix += f".{self.parent}"
        if self.sign is not None:
            suffix += "." + {-1: "neg", 0: "zero", 1: "pos"}[self.sign]
        return f"{self.root.root_id}.{suffix}"

    @property
    def predecessor(self) -> 'ResponseGeometryAssayNativeSegment | None':
        if self.phase == "prefix":
            return None
        if self.phase == "parent":
            return type(self)(self.root, "prefix", None, None)
        if self.phase == "inner":
            return type(self)(self.root, "parent", self.parent, None)
        return type(self)(self.root, "inner", self.parent, self.sign)

    @property
    def start_tick(self) -> int:
        if self.phase == "prefix":
            return 0
        if self.phase == "parent":
            return self.root.landmark_tick
        origin = self.root.landmark_tick + self.root.invocation_offset
        return origin + (self.root.pulse_ticks // 2 if self.phase == "resume" else 0)

    @property
    def end_tick(self) -> int:
        if self.phase == "prefix":
            return self.root.landmark_tick
        if self.phase == "parent":
            return self.root.landmark_tick + self.root.invocation_offset
        duration = (
            self.root.pulse_ticks // 2
            if self.phase == "inner" and self.root.restart
            else self.root.horizon_ticks
        )
        return self.root.landmark_tick + self.root.invocation_offset + duration

    @property
    def native_updates(self) -> int:
        return (self.end_tick - self.start_tick) * sum(self.root.refinements)


@lru_cache(maxsize=1)
def assay_segments() -> tuple[ResponseGeometryAssayNativeSegment, ...]:
    records: list[ResponseGeometryAssayNativeSegment] = []
    for root in assay_roots():
        records.append(ResponseGeometryAssayNativeSegment(root, "prefix", None, None))
        for parent in PARENTS:
            records.append(ResponseGeometryAssayNativeSegment(root, "parent", parent, None))
            for sign in INNER_SIGNS:
                records.append(ResponseGeometryAssayNativeSegment(root, "inner", parent, sign))
                if root.restart:
                    records.append(ResponseGeometryAssayNativeSegment(root, "resume", parent, sign))
    return tuple(sorted(records, key=lambda record: record.task_id))


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayNativeDelivery(CanonicalRecord):
    """Applied complete intervals; realized states remain in native observations.

    The segment supplies requested clocks/parent/sign. Acceptance occurs before
    stepping; the digest commits each delivered coupling endpoint and both force
    evaluations, including zero-force intervals. A failure never implies delivery
    of the remaining request.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-assay-native-delivery'

    accepted: bool
    completed_intervals: int
    force_evaluations: int
    nonzero_force_intervals: int
    signed_impulse: Decimal
    interval_trace_sha256: str
    frozen_mode_sha256: str | None

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise ValueError("assay delivery acceptance must be explicit")
        for value in (
            self.completed_intervals,
            self.force_evaluations,
            self.nonzero_force_intervals,
        ):
            if type(value) is not int or value < 0:
                raise ValueError("assay applied interval counts must be nonnegative integers")
        if self.force_evaluations != 2 * self.completed_intervals:
            raise ValueError("assay complete interval must account for both force evaluations")
        if self.nonzero_force_intervals > self.completed_intervals:
            raise ValueError("assay nonzero force exceeds delivered intervals")
        if not self.accepted and self.completed_intervals:
            raise ValueError("assay unaccepted segment cannot have applied intervals")
        if not self.signed_impulse.is_finite():
            raise ValueError("assay delivered impulse must be finite")
        if not self.nonzero_force_intervals and self.signed_impulse:
            raise ValueError("assay zero force cannot deliver impulse")
        validate_sha256(self.interval_trace_sha256, field_name="interval_trace_sha256")
        if self.frozen_mode_sha256 is not None:
            validate_sha256(self.frozen_mode_sha256, field_name="frozen_mode_sha256")


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayNativeViewSegment(CanonicalRecord):
    """Native delivery only; source failure cannot become a scientific negative."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-assay-native-view-segment'

    refinement: int
    disposition: str
    last_completed_native_step: int
    checkpoint: ResponseGeometryNativeCheckpoint | None
    reason: str | None
    force_work: Decimal
    parent_absolute_density_work: Decimal
    delivery: ResponseGeometryAssayNativeDelivery

    def __post_init__(self) -> None:
        if type(self.refinement) is not int or self.refinement not in {1, 2, 4}:
            raise ValueError("assay segment numerical refinement differs")
        if self.disposition not in {
            "COMPLETE",
            "UNENTERED",
            "NUMERICAL_FAILURE",
            "OBSERVATION_FAILURE",
        }:
            raise ValueError("assay native segment disposition differs")
        if type(self.last_completed_native_step) is not int or self.last_completed_native_step < 0:
            raise ValueError("assay completed native step must be a nonnegative integer")
        if (self.disposition == "COMPLETE") != (self.reason is None):
            raise ValueError("assay source completion and reason disagree")
        if self.checkpoint is not None and (
            self.checkpoint.passive.refinement != self.refinement
            or self.checkpoint.native.step_index != self.last_completed_native_step
        ):
            raise ValueError("assay segment checkpoint differs from its delivered view/clock")
        if self.disposition == "COMPLETE" and self.checkpoint is None:
            raise ValueError("assay complete native segment lacks its continuation checkpoint")
        if self.disposition != "COMPLETE" and self.checkpoint is not None:
            raise ValueError("assay failed segment cannot supply a continuation checkpoint")
        if self.delivery.accepted != (self.disposition != "UNENTERED"):
            raise ValueError("assay disposition and accepted delivery disagree")
        if not all(
            value.is_finite() for value in (self.force_work, self.parent_absolute_density_work)
        ):
            raise ValueError("assay native work must remain finite")
        if self.parent_absolute_density_work < 0:
            raise ValueError("assay absolute parent density work cannot be negative")


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssayNativeSegmentResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/response-geometry-assay-native-segment-result'
    SOURCE_CONFIG_SCHEMA: ClassVar[str] = ResponseGeometryAssayNativeConfig.SCHEMA
    OBSERVATION_SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/six-matrix-response/assay-native-observations-hdf5'

    result_id: str
    source_config: ObjectIdentity
    segment: ResponseGeometryAssayNativeSegment
    predecessor_result: ObjectIdentity | None
    views: tuple[ResponseGeometryAssayNativeViewSegment, ...]
    observations_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.result_id != f"result.{self.segment.task_id}":
            raise ValueError("assay result identity differs from its issued segment")
        if self.source_config.object_schema != self.SOURCE_CONFIG_SCHEMA:
            raise ValueError("assay native result requires its exact source config")
        if (self.segment.predecessor is None) != (self.predecessor_result is None):
            raise ValueError("assay native result predecessor roster differs")
        if self.predecessor_result is not None:
            predecessor = self.segment.predecessor
            assert predecessor is not None
            if (
                self.predecessor_result.object_schema != self.SCHEMA
                or self.predecessor_result.object_id != f"result.{predecessor.task_id}"
            ):
                raise ValueError("assay native result names another predecessor")
        if tuple(view.refinement for view in self.views) != self.segment.root.refinements:
            raise ValueError("assay native result drops or adds a numerical view")
        for view in self.views:
            start = self.segment.start_tick * view.refinement
            end = self.segment.end_tick * view.refinement
            delivered = view.delivery
            if view.disposition == "UNENTERED":
                if view.last_completed_native_step > start:
                    raise ValueError("assay unentered segment advances its native clock")
            elif not start <= view.last_completed_native_step <= end:
                raise ValueError("assay delivered clock is outside its declared interval")
            elif delivered.completed_intervals != view.last_completed_native_step - start:
                raise ValueError("assay applied intervals differ from its delivered native clock")
            if view.disposition == "COMPLETE" and (view.last_completed_native_step != end):
                raise ValueError("assay complete view stopped before its declared endpoint")
            pulse = None
            if self.segment.phase in {"inner", "resume"}:
                inner = self.segment if self.segment.phase == "inner" else self.segment.predecessor
                assert inner is not None and inner.sign is not None
                pulse = ResponseGeometryNativeForcePulse(
                    f"action.{inner.task_id}", inner.root.assay, inner.sign, inner.start_tick
                )
            force = 0 if pulse is None else pulse.sign * int(pulse.amplitude)
            active_end = (
                start
                if pulse is None
                else (pulse.invocation_tick + pulse.pulse_ticks) * view.refinement
            )
            expected_nonzero = (
                max(0, min(view.last_completed_native_step, active_end) - start)
                if delivered.accepted and force
                else 0
            )
            if delivered.nonzero_force_intervals != expected_nonzero or (
                delivered.signed_impulse
                != Decimal(force * expected_nonzero) * Decimal("0.001") / view.refinement
            ):
                raise ValueError("assay applied force/impulse differs from its exact pulse clock")
            checkpoint = view.checkpoint
            if checkpoint is not None and (
                checkpoint.native.request
                != ObjectIdentity.from_record(self.segment.task_id, self.segment)
                or checkpoint.native.total_steps
                != (
                    self.segment.root.landmark_tick
                    + self.segment.root.invocation_offset
                    + self.segment.root.horizon_ticks
                )
                * view.refinement
                or checkpoint.parent_id != self.segment.parent
                or checkpoint.parent_origin_tick
                != (None if self.segment.parent is None else self.segment.root.landmark_tick)
                or checkpoint.pulse != pulse
            ):
                raise ValueError("assay continuation checkpoint differs from its issued native request")
            if (
                checkpoint is not None
                and self.segment.phase != "prefix"
                and (
                    checkpoint.frozen_mode_base64 is None
                    or delivered.frozen_mode_sha256
                    != sha256(
                        base64.b64decode(checkpoint.frozen_mode_base64, validate=True)
                    ).hexdigest()
                )
            ):
                raise ValueError("assay delivered force mode differs from its frozen continuation")
        validate_sha256(self.observations_sha256, field_name="observations_sha256")
