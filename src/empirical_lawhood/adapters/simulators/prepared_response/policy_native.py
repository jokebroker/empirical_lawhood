"""Policy-keyed fresh calibration native task records with explicit realized-parent lineage.

The production graph cannot know an adaptive parent before the pre-parent
instrument exists.  These records keep the operational policy task identity
separate from the realized native parent invocation while preserving the same
common start and paired numerical views.
"""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .contracts import PreparedForceWord, PreparedNativeSpec, PreparedRoot
from .native_pair import PreparedNativePairArtifact, decode_prepared_native_pair
from .native_tasks import PreparedNativeInvocation
from .source import PreparedCommonStart, PreparedNativePhaseData
from .source_outputs import unentered_prepared_native_bytes


CALIBRATION_POLICIES = ("primary", "hold", "best-fixed", "conventional")
PARENT_DECISION_SCHEMA = 'empirical-lawhood/methods/prepared-response/prepared-parent-decision'


def _native_task_id(root: PreparedRoot, suffix: str) -> str:
    # Operational names are repeated on every graph edge; science stays in the invocation.
    digest = sha256(f"{root.root_id}.{suffix}.native".encode()).hexdigest()
    return f"prepared-response.calibration.native.{digest[:24]}"


@dataclass(frozen=True, slots=True)
class PreparedResponseCalibrationNativeInvocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-response-calibration-native-invocation'
    source_spec: ObjectIdentity
    root: PreparedRoot
    phase: str
    policy_id: str | None
    word: PreparedForceWord | None
    purpose: str

    def __post_init__(self) -> None:
        if (
            self.source_spec.object_schema != PreparedNativeSpec.SCHEMA
            or self.root.stage != 'calibration'
            or self.phase not in ("prefix", "parent", "future")
            or self.phase == "prefix"
            and (
                self.policy_id is not None
                or self.word is not None
                or self.purpose != "initial-ramp"
            )
            or self.phase == "parent"
            and (
                self.policy_id not in CALIBRATION_POLICIES
                or self.word is not None
                or self.purpose != "parent"
            )
            or self.phase == "future"
            and (
                self.policy_id not in CALIBRATION_POLICIES
                or self.word is None
                or self.purpose != 'common-response'
            )
        ):
            raise ValueError("prepared fresh calibration invocation changes its policy/source phase roster")

    @property
    def task_id(self) -> str:
        if self.phase == "prefix":
            suffix = "prefix"
        elif self.phase == "parent":
            suffix = f"policy.{self.policy_id}.parent"
        else:
            assert self.word is not None
            suffix = f"policy.{self.policy_id}.common-response.{self.word.word_id}"
        return _native_task_id(self.root, suffix)

    @property
    def prefix_task_id(self) -> str:
        return _native_task_id(self.root, "prefix")

    @property
    def decision_task_id(self) -> str:
        if self.policy_id is None:
            raise ValueError("a prefix has no parent-policy decision")
        return f"{self.root.root_id}.policy.{self.policy_id}.decision"

    @property
    def dependency_task_ids(self) -> tuple[str, ...]:
        if self.phase == "prefix":
            return ()
        if self.phase == "parent":
            return tuple(sorted((self.prefix_task_id, self.decision_task_id)))
        return (_native_task_id(self.root, f"policy.{self.policy_id}.parent"),)

    @property
    def maximum_native_updates(self) -> int:
        return 3 * (
            self.root.landmark
            if self.phase == "prefix"
            else 272
            if self.phase == "parent"
            else 320
        )


def prepared_response_calibration_native_invocations(
    spec: PreparedNativeSpec,
    *,
    root: PreparedRoot | None = None,
) -> tuple[PreparedResponseCalibrationNativeInvocation, ...]:
    """The fixed policy census, or one authenticated root from that census."""
    if spec.stage != 'calibration':
        raise ValueError("prepared fresh calibration acquisition cannot reinterpret another stage")
    roots = spec.roots
    if root is not None:
        if root not in roots:
            raise ValueError("prepared fresh calibration invocation root is outside its frozen source census")
        roots = (root,)
    source = ObjectIdentity.from_record(spec.spec_id, spec)
    words = spec.words
    values = []
    for root in roots:
        values.append(
            PreparedResponseCalibrationNativeInvocation(source, root, "prefix", None, None, "initial-ramp")
        )
        for policy in CALIBRATION_POLICIES:
            values.append(
                PreparedResponseCalibrationNativeInvocation(source, root, "parent", policy, None, "parent")
            )
            values.extend(
                PreparedResponseCalibrationNativeInvocation(source, root, "future", policy, word, 'common-response')
                for word in words
            )
    return tuple(sorted(values, key=lambda value: value.task_id))


@dataclass(frozen=True, slots=True)
class PreparedResponseCalibrationNativeTaskResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-response-calibration-native-task-result'
    invocation: PreparedResponseCalibrationNativeInvocation
    predecessors: tuple[ObjectIdentity, ...]
    realized_parent: str | None
    native_pair: PreparedNativePairArtifact | None
    common_start: PreparedCommonStart | None
    unentered_reason: str | None

    def __post_init__(self) -> None:
        expected_ids = tuple(f"{task_id}.result" for task_id in self.invocation.dependency_task_ids)
        if tuple(value.object_id for value in self.predecessors) != expected_ids:
            raise ValueError("prepared fresh calibration result changes its operational predecessor census")
        if self.invocation.phase == "parent":
            if {value.object_schema for value in self.predecessors} != {
                self.SCHEMA,
                PARENT_DECISION_SCHEMA,
            }:
                raise ValueError("prepared fresh calibration parent lacks its prefix and frozen decision")
        elif any(value.object_schema != self.SCHEMA for value in self.predecessors):
            raise ValueError("prepared fresh calibration native lineage uses another result schema")
        if self.invocation.phase == "prefix":
            if self.realized_parent is not None:
                raise ValueError("prepared fresh calibration prefix cannot fabricate a realized parent")
        elif self.realized_parent is None:
            if self.native_pair is not None or self.unentered_reason is None:
                raise ValueError("prepared fresh calibration nonentry cannot carry native parent/future bytes")
        if self.native_pair is not None:
            if self.unentered_reason is not None or (
                self.invocation.phase != "prefix" and self.realized_parent is None
            ):
                raise ValueError("prepared fresh calibration native pair conflicts with its disposition")
            actual = self.native_pair.invocation
            if (
                actual.source_spec != self.invocation.source_spec
                or actual.root != self.invocation.root
                or actual.phase != self.invocation.phase
                or actual.parent != self.realized_parent
                or actual.word != self.invocation.word
                or actual.purpose != self.invocation.purpose
            ):
                raise ValueError("prepared fresh calibration native pair changes its realized policy invocation")
        elif self.invocation.phase == "prefix" or self.unentered_reason not in {
            "PREFIX_UNAVAILABLE",
            "PORT_FRAME_UNRESOLVED",
            "POLICY_NONATTEMPT",
            "HANDOFF_UNAVAILABLE",
        }:
            raise ValueError("prepared fresh calibration unentered task has no valid predecessor disposition")
        if self.common_start is not None and (
            self.common_start.root != self.invocation.root
            or any(
                checkpoint.source_spec != self.invocation.source_spec
                for checkpoint in self.common_start.checkpoints
            )
        ):
            raise ValueError("prepared fresh calibration result carries another common start")
        if self.invocation.phase == "prefix":
            if (self.common_start is not None) != self.native_complete:
                raise ValueError("only a complete fresh calibration prefix can carry a common start")
        elif self.native_pair is not None and (
            self.common_start is None or self.common_start.mode_disposition != "RESOLVED"
        ):
            raise ValueError("entered fresh calibration continuation requires its resolved common start")

    @property
    def result_id(self) -> str:
        return f"{self.invocation.task_id}.result"

    @property
    def completed_native_updates(self) -> int:
        return (
            0
            if self.native_pair is None
            else sum(value.delivery.completed_intervals for value in self.native_pair.views)
        )

    @property
    def native_complete(self) -> bool:
        return self.native_pair is not None and all(
            value.delivery.disposition == "COMPLETE" for value in self.native_pair.views
        )


def decode_prepared_response_calibration_task_native(
    result: PreparedResponseCalibrationNativeTaskResult, payload: bytes
) -> tuple[PreparedNativePhaseData, PreparedNativePhaseData] | None:
    if result.native_pair is None:
        if payload != unentered_prepared_native_bytes():
            raise ValueError("unentered prepared fresh calibration task cannot carry native evidence")
        return None
    return decode_prepared_native_pair(result.native_pair, payload)


def actual_prepared_invocation(
    invocation: PreparedResponseCalibrationNativeInvocation, realized_parent: str
) -> PreparedNativeInvocation:
    return PreparedNativeInvocation(
        invocation.source_spec,
        invocation.root,
        invocation.phase,
        None if invocation.phase == "prefix" else realized_parent,
        invocation.word,
        invocation.purpose,
    )
