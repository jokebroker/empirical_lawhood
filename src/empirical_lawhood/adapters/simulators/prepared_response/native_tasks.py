"""Exact native invocations and the outcome-blind source qualification/dependent refinement acquisition census.

These are source declarations, not an execution DAG or permission records.
The installed provider expands them through the shared protocol compiler.
Later policy-selected stages must bind their frozen decision before producing
an invocation; this module cannot expand them to a full exploration menu.
"""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .contracts import PARENTS, PreparedForceWord, PreparedNativeSpec, PreparedRoot, validate_prepared_future_role


@dataclass(frozen=True, slots=True)
class PreparedNativeInvocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/prepared-response/prepared-native-invocation'
    source_spec: ObjectIdentity
    root: PreparedRoot
    phase: str
    parent: str | None
    word: PreparedForceWord | None
    purpose: str

    def __post_init__(self) -> None:
        if (
            self.source_spec.object_schema != PreparedNativeSpec.SCHEMA
            or self.phase not in ("prefix", "parent", "future")
            or self.phase == "prefix"
            and (self.parent is not None or self.word is not None or self.purpose != "initial-ramp")
            or self.phase == "parent"
            and (self.parent not in PARENTS or self.word is not None or self.purpose != "parent")
            or self.phase == "future"
            and (self.parent not in PARENTS or self.word is None)
        ):
            raise ValueError(
                "prepared invocation changes its exact source/phase/action/purpose roles"
            )
        if self.phase == "future":
            assert self.word is not None
            validate_prepared_future_role(self.root.stage, self.word, self.purpose)

    def validate_spec(self, spec: PreparedNativeSpec) -> None:
        if (
            self.source_spec != ObjectIdentity.from_record(spec.spec_id, spec)
            or self.root not in spec.roots
            or self.word is not None
            and self.word not in spec.words
        ):
            raise ValueError("prepared invocation differs from its frozen source/root/word census")

    @property
    def task_id(self) -> str:
        suffix = (
            "prefix"
            if self.phase == "prefix"
            else f"{self.parent}.parent"
            if self.phase == "parent"
            else f"{self.parent}.{self.purpose}.{self.word.word_id}"
            if self.word is not None
            else "missing"
        )
        return f"{self.root.root_id}.{suffix}.native"

    @property
    def native_occurrence_ids(self) -> tuple[str, str]:
        stem = self.task_id.removesuffix(".native")
        return f"{stem}.r1", f"{stem}.r2"

    @property
    def dependency_task_ids(self) -> tuple[str, ...]:
        if self.phase == "prefix":
            return ()
        prefix = f"{self.root.root_id}.prefix.native"
        return (
            (prefix,)
            if self.phase == "parent"
            else (f"{self.root.root_id}.{self.parent}.parent.native",)
        )

    @property
    def maximum_native_updates(self) -> int:
        return 3 * (
            self.root.landmark if self.phase == "prefix" else 272 if self.phase == "parent" else 320
        )


def prepared_static_native_invocations(
    spec: PreparedNativeSpec,
    *,
    root: PreparedRoot | None = None,
) -> tuple[PreparedNativeInvocation, ...]:
    "The fixed census, or one authenticated root, including dependent refinement's extra HOLDs."
    if spec.stage not in ('qualification', 'development', 'excluded-canary'):
        raise ValueError("policy-selected stages cannot acquire a static full parent/action menu")
    roots = spec.roots
    if root is not None:
        if root not in roots:
            raise ValueError("prepared invocation root is outside its frozen source census")
        roots = (root,)
    identity = ObjectIdentity.from_record(spec.spec_id, spec)
    words = spec.words
    values = []
    for root in roots:
        values.append(
            PreparedNativeInvocation(identity, root, "prefix", None, None, "initial-ramp")
        )
        for parent in PARENTS:
            values.append(
                PreparedNativeInvocation(identity, root, "parent", parent, None, "parent")
            )
            values.extend(
                PreparedNativeInvocation(identity, root, "future", parent, word, 'common-response')
                for word in words
            )
            if spec.stage == 'development':
                values.extend(
                    PreparedNativeInvocation(
                        identity, root, "future", parent, words[0], purpose
                    )
                    for purpose in ('independent-response-1', 'independent-response-2', 'independent-response-3')
                )
    return tuple(sorted(values, key=lambda value: value.task_id))
