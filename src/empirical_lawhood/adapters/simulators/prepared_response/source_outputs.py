"""Source task products distinguish native delivery from an unentered phase."""

from dataclasses import dataclass
from functools import lru_cache
import io
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import response_hdf5_text, response_hdf5_writer

from .native_pair import PAIR_METADATA, PreparedNativePairArtifact, decode_prepared_native_pair
from .native_tasks import PreparedNativeInvocation
from .source import PreparedCommonStart, PreparedNativePhaseData


@dataclass(frozen=True, slots=True)
class PreparedNativeTaskResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-native-task-result'
    invocation: PreparedNativeInvocation
    predecessors: tuple[ObjectIdentity, ...]
    native_pair: PreparedNativePairArtifact | None
    common_start: PreparedCommonStart | None
    unentered_reason: str | None

    def __post_init__(self) -> None:
        if (
            type(self.predecessors) is not tuple
            or tuple(value.object_id for value in self.predecessors)
            != tuple(f"{task}.result" for task in self.invocation.dependency_task_ids)
            or any(value.object_schema != self.SCHEMA for value in self.predecessors)
        ):
            raise ValueError("prepared native result changes its exact predecessor receipt census")
        if self.native_pair is not None:
            if self.unentered_reason is not None or self.native_pair.invocation != self.invocation:
                raise ValueError("prepared native delivery differs from its declared invocation")
        elif self.invocation.phase == "prefix" or self.unentered_reason not in (
            ("PREFIX_UNAVAILABLE", "PORT_FRAME_UNRESOLVED")
            if self.invocation.phase == "parent"
            else ("PREFIX_UNAVAILABLE", "PORT_FRAME_UNRESOLVED", "HANDOFF_UNAVAILABLE")
        ):
            raise ValueError("prepared unentered phase lacks a declared unavailable predecessor")
        common = self.common_start
        if common is not None and (
            common.root != self.invocation.root
            or common.common_start_id != f"{self.invocation.root.root_id}.common-start"
            or any(c.source_spec != self.invocation.source_spec for c in common.checkpoints)
        ):
            raise ValueError("prepared result carries another root/source/common-start binding")
        if self.invocation.phase == "prefix":
            if (common is not None) != self.native_complete:
                raise ValueError("only a complete prefix pair can carry a common start")
            if (
                common is not None
                and self.native_pair is not None
                and tuple(
                    ObjectIdentity.from_record(c.checkpoint_id, c) for c in common.checkpoints
                )
                != tuple(v.checkpoint for v in self.native_pair.views)
            ):
                raise ValueError("prepared prefix common start changes its exact checkpoint pair")
        elif self.native_pair is not None:
            if common is None or common.mode_disposition != "RESOLVED":
                raise ValueError(
                    "an entered native continuation requires its resolved common start"
                )
            identity = ObjectIdentity.from_record(common.common_start_id, common)
            if any(v.delivery.common_start != identity for v in self.native_pair.views):
                raise ValueError("prepared continuation changes the carried common-start identity")
            if self.invocation.phase == "parent" and tuple(
                v.delivery.incoming_checkpoint for v in self.native_pair.views
            ) != tuple(ObjectIdentity.from_record(c.checkpoint_id, c) for c in common.checkpoints):
                raise ValueError("prepared parent changes an incoming common-start checkpoint")
        elif (
            self.unentered_reason == "PREFIX_UNAVAILABLE"
            and common is not None
            or self.unentered_reason == "PORT_FRAME_UNRESOLVED"
            and (common is None or common.mode_disposition != "NONATTEMPT_UNRESOLVED_PORT_FRAME")
            or self.unentered_reason == "HANDOFF_UNAVAILABLE"
            and (common is None or common.mode_disposition != "RESOLVED")
        ):
            raise ValueError(
                "prepared nonentry reason disagrees with its actual common-start state"
            )

    @property
    def result_id(self) -> str:
        return f"{self.invocation.task_id}.result"

    @property
    def completed_native_updates(self) -> int:
        return (
            0
            if self.native_pair is None
            else sum(view.delivery.completed_intervals for view in self.native_pair.views)
        )

    @property
    def native_complete(self) -> bool:
        return self.native_pair is not None and all(
            view.delivery.disposition == "COMPLETE" for view in self.native_pair.views
        )


@lru_cache(maxsize=1)
def unentered_prepared_native_bytes() -> bytes:
    """Explicit empty inventory for a task whose scientific phase did not enter.

    The result record supplies the reason and authenticated predecessor IDs.
    There is no delivery, checkpoint, observation, or implied zero response.
    """
    stream = io.BytesIO()
    with response_hdf5_writer(stream) as artifact:
        for name, value in sorted(PAIR_METADATA.items()):
            response_hdf5_text(artifact, name, value)
        response_hdf5_text(artifact, "native_phase_disposition", "NOT_ENTERED")
        for name in ("primary", "half"):
            artifact.create_dataset(name, data=np.empty(0, dtype=np.uint8), track_times=False)
    return stream.getvalue()


def decode_prepared_task_native(
    result: PreparedNativeTaskResult, payload: bytes
) -> tuple[PreparedNativePhaseData, PreparedNativePhaseData] | None:
    if result.native_pair is None:
        if type(payload) is not bytes or payload != unentered_prepared_native_bytes():
            raise ValueError("an unentered native phase cannot carry undeclared scientific bytes")
        return None
    return decode_prepared_native_pair(result.native_pair, payload)
