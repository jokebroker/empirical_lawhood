"""preparation-policy native task products with explicit retained-prefix lineage."""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .native_artifact import decode_native_pair
from .source import FiniteResponseLawNativePhaseData
from .preparation_policy_contracts import FiniteResponseLawPreparationPolicyNativeInvocation
from .preparation_policy_native_artifact import FiniteResponseLawPreparationPolicyNativePairArtifact
from .preparation_policy_native_artifact import unentered_preparation_policy_native_bytes


@dataclass(frozen=True, slots=True)
class FiniteResponseLawPreparationPolicyNativeTaskResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/finite-response-law/finite-response-law-preparation-policy-native-task-result'
    invocation: FiniteResponseLawPreparationPolicyNativeInvocation
    predecessor: ObjectIdentity
    native_pair: FiniteResponseLawPreparationPolicyNativePairArtifact | None
    retained_common_start: ObjectIdentity
    frozen_ports_base64: str
    unentered_reason: str | None

    def __post_init__(self) -> None:
        task = self.invocation
        expected_predecessor = task.predecessor_segment_id
        expected_schema = (
            'empirical-lawhood/simulators/prepared-response/prepared-native-task-result'
            if task.phase == "preparation"
            else self.SCHEMA
        )
        if (
            expected_predecessor is None
            or self.predecessor.object_id != f"{expected_predecessor}.result"
            or self.predecessor.object_schema != expected_schema
            or self.retained_common_start.object_schema
            != 'empirical-lawhood/simulators/prepared-response/prepared-common-start'
            or self.retained_common_start.object_id != f"{task.root.root_id}.common-start"
        ):
            raise ValueError("preparation-policy result changes predecessor/common-start identity")
        frame = self.frame
        if frame.cutoff_tick != 4096:
            raise ValueError("preparation-policy result changes its frozen retained-prefix frame")
        if self.native_pair is None:
            if self.unentered_reason not in ("PREFIX_UNAVAILABLE", "HANDOFF_UNAVAILABLE"):
                raise ValueError("preparation-policy nonentry lacks its exact predecessor disposition")
        elif self.unentered_reason is not None or any(
            delivery.invocation != task for delivery in self.native_pair.deliveries
        ):
            raise ValueError("preparation-policy result differs from its declared native invocation")
        elif any(
            delivery.frozen_frame_sha256 != sha256(frame.modes.tobytes()).hexdigest()
            for delivery in self.native_pair.deliveries
        ):
            raise ValueError("preparation-policy result delivers a substituted port frame")

    @property
    def result_id(self) -> str:
        return f"{self.invocation.task_id}.result"

    @property
    def native_complete(self) -> bool:
        return self.native_pair is not None and all(
            delivery.disposition == "COMPLETE" for delivery in self.native_pair.deliveries
        )

    @property
    def completed_native_updates(self) -> int:
        return (
            0
            if self.native_pair is None
            else sum(delivery.completed_intervals for delivery in self.native_pair.deliveries)
        )

    @property
    def frame(self) -> PreparedPortFrame:
        return PreparedPortFrame(4096, _decode(self.frozen_ports_base64, (2, 3, 4, 4)))


def decode_preparation_policy_task_native(
    result: FiniteResponseLawPreparationPolicyNativeTaskResult, payload: bytes
) -> tuple[FiniteResponseLawNativePhaseData, FiniteResponseLawNativePhaseData] | None:
    if result.native_pair is None:
        if payload != unentered_preparation_policy_native_bytes():
            raise ValueError("preparation-policy unentered result carries native observations")
        return None
    return decode_native_pair(result.native_pair, payload)
