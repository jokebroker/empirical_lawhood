"""Finite native task products with explicit failed and unentered branches."""

import io
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedCommonStart
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode, _encode
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_text, response_hdf5_writer
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .assigned_contracts import FiniteResponseLawAssignedCalibrationInvocation, FiniteResponseLawAssignedEvaluationInvocation
from .contracts import FiniteResponseLawNativeInvocation
from .evaluation_contracts import FiniteResponseLawEvaluationInvocation
from .fresh_contracts import FiniteResponseLawCalibrationInvocation
from .native_artifact import METADATA, FiniteResponseLawAssignedCalibrationPairArtifact, FiniteResponseLawAssignedEvaluationPairArtifact, FiniteResponseLawCalibrationPairArtifact, FiniteResponseLawEvaluationPairArtifact, FiniteResponseLawNativePairArtifact, decode_native_pair
from .source import FiniteResponseLawNativePhaseData, frozen_prefix_frame


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeTaskResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-native-task-result'
    INVOCATION_TYPE: ClassVar[type[FiniteResponseLawNativeInvocation]] = FiniteResponseLawNativeInvocation
    PAIR_TYPE: ClassVar[type[FiniteResponseLawNativePairArtifact]] = FiniteResponseLawNativePairArtifact
    invocation: FiniteResponseLawNativeInvocation
    predecessors: tuple[ObjectIdentity, ...]
    native_pair: FiniteResponseLawNativePairArtifact | None
    common_start: ObjectIdentity | None
    frozen_ports_base64: str | None
    unentered_reason: str | None

    def __post_init__(self) -> None:
        task = self.invocation
        if type(task) is not self.INVOCATION_TYPE or (
            self.native_pair is not None
            and type(self.native_pair) is not self.PAIR_TYPE
        ):
            raise ValueError(
                "Finite response-law native result changes its versioned invocation/artifact binding"
            )
        predecessor = task.predecessor_segment_id
        retained = task.root.cohort == "retained-prepared-response"
        expected_schema = PreparedNativeTaskResult.SCHEMA if retained else self.SCHEMA
        if tuple(p.object_id for p in self.predecessors) != (
            () if predecessor is None else (f"{predecessor}.result",)
        ) or any(p.object_schema != expected_schema for p in self.predecessors):
            raise ValueError("Finite response-law native result changes its exact predecessor identity")
        pair = self.native_pair
        if pair is not None:
            if self.unentered_reason is not None or any(
                d.invocation != task for d in pair.deliveries
            ):
                raise ValueError(
                    "Finite response-law native result differs from its declared invocation"
                )
        elif (
            task.phase == "prefix"
            or retained
            or self.unentered_reason
            not in (
                "PREFIX_UNAVAILABLE",
                "PORT_FRAME_UNRESOLVED",
                "HANDOFF_UNAVAILABLE",
            )
        ):
            raise ValueError("Finite response-law nonentry lacks an unavailable new predecessor")
        if task.phase == "prefix":
            if self.common_start is not None or (
                not self.native_complete and self.frame is not None
            ):
                raise ValueError(
                    "Finite response-law prefix cannot import a common start or nominate from failed views"
                )
        elif self.common_start is not None:
            schema = PreparedCommonStart.SCHEMA if retained else self.SCHEMA
            expected_id = (
                f"{task.root.root_id}.common-start"
                if retained
                else f"finite-response-law.{task.root.stage_unit}.prefix.native.result"
                if isinstance(
                    task,
                    (
                        FiniteResponseLawCalibrationInvocation,
                        FiniteResponseLawAssignedEvaluationInvocation,
                    ),
                )
                else "finite-response-law.native-canary.r000.prefix.native.result"
            )
            if (
                self.common_start.object_schema != schema
                or self.common_start.object_id != expected_id
            ):
                raise ValueError("Finite response-law continuation changes its common-start identity")
        if task.phase != "prefix" and pair is not None:
            if self.common_start is None or self.frame is None:
                raise ValueError(
                    "Finite response-law entered continuation lacks its resolved pre-parent frame"
                )
            digest = sha256(self.frame.modes.tobytes()).hexdigest()
            if any(d.frozen_frame_sha256 != digest for d in pair.deliveries):
                raise ValueError("Finite response-law continuation delivers a different frozen frame")
        if (
            self.unentered_reason == "PREFIX_UNAVAILABLE"
            and (self.common_start is not None or self.frame is not None)
            or self.unentered_reason == "PORT_FRAME_UNRESOLVED"
            and (self.common_start is None or self.frame is not None)
            or self.unentered_reason == "HANDOFF_UNAVAILABLE"
            and (
                task.phase != "future"
                or self.common_start is None
                or self.frame is None
            )
        ):
            raise ValueError(
                "Finite response-law nonentry reason disagrees with its actual predecessor/frame"
            )

    @property
    def result_id(self) -> str:
        return f"{self.invocation.task_id}.result"

    @property
    def native_complete(self) -> bool:
        return self.native_pair is not None and all(
            d.disposition == "COMPLETE" for d in self.native_pair.deliveries
        )

    @property
    def completed_native_updates(self) -> int:
        return (
            0
            if self.native_pair is None
            else sum(d.completed_intervals for d in self.native_pair.deliveries)
        )

    @property
    def frame(self) -> PreparedPortFrame | None:
        return (
            None
            if self.frozen_ports_base64 is None
            else PreparedPortFrame(
                4096, _decode(self.frozen_ports_base64, (2, 3, 4, 4))
            )
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationTaskResult(FiniteResponseLawNativeTaskResult):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-calibration-task-result'
    )
    VERSION: ClassVar[str] = '1.0.0'
    INVOCATION_TYPE: ClassVar[type[FiniteResponseLawNativeInvocation]] = FiniteResponseLawCalibrationInvocation
    PAIR_TYPE: ClassVar[type[FiniteResponseLawNativePairArtifact]] = FiniteResponseLawCalibrationPairArtifact
    invocation: FiniteResponseLawCalibrationInvocation
    native_pair: FiniteResponseLawCalibrationPairArtifact | None


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationTaskResult(FiniteResponseLawNativeTaskResult):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-evaluation-task-result'
    )
    INVOCATION_TYPE: ClassVar[type[FiniteResponseLawNativeInvocation]] = FiniteResponseLawEvaluationInvocation
    PAIR_TYPE: ClassVar[type[FiniteResponseLawNativePairArtifact]] = FiniteResponseLawEvaluationPairArtifact
    invocation: FiniteResponseLawEvaluationInvocation
    native_pair: FiniteResponseLawEvaluationPairArtifact | None


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationTaskResult(FiniteResponseLawNativeTaskResult):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-calibration-task-result'
    )
    VERSION: ClassVar[str] = '1.0.0'
    INVOCATION_TYPE: ClassVar[type[FiniteResponseLawNativeInvocation]] = (
        FiniteResponseLawAssignedCalibrationInvocation
    )
    PAIR_TYPE: ClassVar[type[FiniteResponseLawNativePairArtifact]] = (
        FiniteResponseLawAssignedCalibrationPairArtifact
    )
    invocation: FiniteResponseLawAssignedCalibrationInvocation
    native_pair: FiniteResponseLawAssignedCalibrationPairArtifact | None


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationTaskResult(FiniteResponseLawNativeTaskResult):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/finite-response-law/finite-response-law-assigned-evaluation-task-result'
    )
    VERSION: ClassVar[str] = '1.0.0'
    INVOCATION_TYPE: ClassVar[type[FiniteResponseLawNativeInvocation]] = (
        FiniteResponseLawAssignedEvaluationInvocation
    )
    PAIR_TYPE: ClassVar[type[FiniteResponseLawNativePairArtifact]] = (
        FiniteResponseLawAssignedEvaluationPairArtifact
    )
    invocation: FiniteResponseLawAssignedEvaluationInvocation
    native_pair: FiniteResponseLawAssignedEvaluationPairArtifact | None


@lru_cache(maxsize=1)
def unentered_native_bytes() -> bytes:
    stream = io.BytesIO()
    with response_hdf5_writer(stream) as artifact:
        for name, value in sorted(METADATA.items()):
            response_hdf5_text(artifact, name, value)
        response_hdf5_text(artifact, "native_phase_disposition", "NOT_ENTERED")
    return stream.getvalue()


def decode_task_native(
    result: FiniteResponseLawNativeTaskResult, payload: bytes
) -> tuple[FiniteResponseLawNativePhaseData, FiniteResponseLawNativePhaseData] | None:
    if result.native_pair is None:
        if type(payload) is not bytes or payload != unentered_native_bytes():
            raise ValueError("Finite response-law unentered task cannot carry a native observation")
        return None
    pair = decode_native_pair(result.native_pair, payload)
    if result.invocation.phase == "prefix" and result.native_complete:
        assert pair[0].checkpoint is not None
        frame = frozen_prefix_frame(pair[0].checkpoint)
        if result.frozen_ports_base64 != (
            None if frame is None else _encode(frame.modes)
        ):
            raise ValueError(
                "Finite response-law frozen frame differs from its actual causal primary prefix"
            )
    return pair
