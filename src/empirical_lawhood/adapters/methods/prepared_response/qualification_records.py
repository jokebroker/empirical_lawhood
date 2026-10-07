"""Frozen source qualification projection and task-charter nomination operands.

source qualification is source/response qualification. These records cannot emit a law, confer
control authority, or supply protected evaluation observations.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PARENTS, PreparedForceWord, PreparedNativeSpec, PreparedRoot, prepared_words
from empirical_lawhood.adapters.simulators.prepared_response.native_tasks import PreparedNativeInvocation
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult


SOURCE_QUALIFICATION_OUTPUT_ORDER = (
    "receiver-m1-handoff-increment",
    "receiver-m2-handoff-increment",
    "cumulative-off-force-x-hs",
    "cumulative-relative-y-hs",
    "cumulative-transfer-operator-difference",
    "signed-force-work",
    "absolute-force-work",
)


@dataclass(frozen=True, slots=True)
class PreparedResponseSourceQualificationProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-source-qualification-projection-config'
    config_id: str
    native_spec: PreparedNativeSpec
    instrument_tier: str = "I0"
    task_velocity_view: int = 1
    output_order: tuple[str, ...] = SOURCE_QUALIFICATION_OUTPUT_ORDER

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.native_spec.stage != 'qualification'
            or self.instrument_tier != "I0"
            or type(self.task_velocity_view) is not int
            or self.task_velocity_view != 1
            or self.output_order != SOURCE_QUALIFICATION_OUTPUT_ORDER
        ):
            raise ValueError(
                "prepared source qualification projection changes its fixed source/instrument/output contract"
            )


def _optional_decimal(value: Decimal | None) -> None:
    if value is not None and (not isinstance(value, Decimal) or not value.is_finite()):
        raise ValueError("prepared source qualification requires finite native Decimal operands or explicit unknowns")


@dataclass(frozen=True, slots=True)
class PreparedResponseSourceQualificationWordObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-source-qualification-word-observation'
    parent: str
    word: PreparedForceWord
    source_result: ObjectIdentity
    delivery_complete: bool
    force_component_error: Decimal | None
    outputs: tuple[tuple[Decimal | None, ...], ...]

    def __post_init__(self) -> None:
        if (
            self.parent not in PARENTS
            or self.word not in prepared_words()
            or self.source_result.object_schema != PreparedNativeTaskResult.SCHEMA
            or type(self.delivery_complete) is not bool
            or type(self.outputs) is not tuple
            or len(self.outputs) != 5
            or any(type(row) is not tuple or len(row) != 7 for row in self.outputs)
        ):
            raise ValueError("prepared source qualification word observation changes its native product census")
        _optional_decimal(self.force_component_error)
        if self.force_component_error is not None and self.force_component_error < 0:
            raise ValueError("prepared source qualification force discrepancy cannot be negative")
        for row in self.outputs:
            for value in row:
                _optional_decimal(value)
            if any(value is not None and value < 0 for value in (row[2], row[3], row[4], row[6])):
                raise ValueError("prepared source qualification magnitude operands cannot be negative")


@dataclass(frozen=True, slots=True)
class PreparedResponseSourceQualificationViewObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prepared-response/prepared-response-source-qualification-view-observation'
    projection_config: ObjectIdentity
    source_spec: ObjectIdentity
    root: PreparedRoot
    refinement: int
    source_results: tuple[ObjectIdentity, ...]
    common_start: ObjectIdentity | None
    mode_disposition: str
    preparent_velocity: tuple[Decimal, ...] | None
    handoff_displacements: tuple[tuple[Decimal, ...] | None, ...]
    parent_density_work: tuple[Decimal | None, ...]
    words: tuple[PreparedResponseSourceQualificationWordObservation, ...]

    def __post_init__(self) -> None:
        if (
            self.projection_config.object_schema != PreparedResponseSourceQualificationProjectionConfig.SCHEMA
            or self.source_spec.object_schema != PreparedNativeSpec.SCHEMA
            or self.root.stage != 'qualification'
            or type(self.refinement) is not int
            or self.refinement not in (1, 2)
            or type(self.words) is not tuple
            or tuple((value.parent, value.word) for value in self.words)
            != tuple((parent, word) for parent in PARENTS for word in prepared_words())
            or type(self.handoff_displacements) is not tuple
            or len(self.handoff_displacements) != 5
            or type(self.parent_density_work) is not tuple
            or len(self.parent_density_work) != 5
            or self.mode_disposition
            not in ("PREFIX_UNAVAILABLE", "NONATTEMPT_UNRESOLVED_PORT_FRAME", "RESOLVED")
            or (self.common_start is None) != (self.mode_disposition == "PREFIX_UNAVAILABLE")
            or (self.preparent_velocity is None) != (self.mode_disposition != "RESOLVED")
        ):
            raise ValueError("prepared source qualification view changes its exact source/parent/word/view census")
        if (
            self.common_start is not None
            and self.common_start.object_schema != 'empirical-lawhood/simulators/prepared-response/prepared-common-start'
        ):
            raise ValueError("prepared source qualification observation loses its pre-parent frame identity")
        invocations = (
            PreparedNativeInvocation(
                self.source_spec, self.root, "prefix", None, None, "initial-ramp"
            ),
            *(
                PreparedNativeInvocation(
                    self.source_spec, self.root, "parent", parent, None, "parent"
                )
                for parent in PARENTS
            ),
            *(
                PreparedNativeInvocation(
                    self.source_spec, self.root, "future", value.parent, value.word, 'common-response'
                )
                for value in self.words
            ),
        )
        if (
            type(self.source_results) is not tuple
            or tuple(value.object_id for value in self.source_results)
            != tuple(sorted(f"{task.task_id}.result" for task in invocations))
            or any(
                value.object_schema != PreparedNativeTaskResult.SCHEMA
                for value in self.source_results
            )
        ):
            raise ValueError("prepared source qualification view must retain every assigned native source result")
        by_id = {value.object_id: value for value in self.source_results}
        for value in self.words:
            task = PreparedNativeInvocation(
                self.source_spec, self.root, "future", value.parent, value.word, 'common-response'
            )
            if value.source_result != by_id[f"{task.task_id}.result"]:
                raise ValueError("prepared source qualification word is detached from its exact native invocation")
        for vector in (self.preparent_velocity, *self.handoff_displacements):
            if vector is not None and (
                type(vector) is not tuple
                or len(vector) != 2
                or any(not isinstance(v, Decimal) or not v.is_finite() for v in vector)
            ):
                raise ValueError("prepared source qualification frame operands require two finite native coordinates")
        for work in self.parent_density_work:
            _optional_decimal(work)
            if work is not None and work < 0:
                raise ValueError("prepared source qualification parent density work cannot be negative")
        if self.mode_disposition != "RESOLVED" and (
            any(v is not None for v in (*self.handoff_displacements, *self.parent_density_work))
            or any(
                v.delivery_complete or any(x is not None for row in v.outputs for x in row)
                for v in self.words
            )
        ):
            raise ValueError("unavailable source qualification source cannot acquire a resolved native response")

    @property
    def report_id(self) -> str:
        return f"{self.root.root_id}.project.r{self.refinement}"
