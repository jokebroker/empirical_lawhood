"""Exact retained inputs for completion after the description spawn defect."""

from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar, Protocol
import re

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.adapters.simulators.matrix_preparation.retained_exports import PreparationRetainedSourceExport
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_relative_locator,
    validate_sha256,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    lineage_parent_sort_key,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    ExternalInputPayload,
    ExternalInputSource,
    MAX_EXTERNAL_INPUT_CHUNK_BYTES,
)
from empirical_lawhood.adapters.simulators.matrix_preparation.continuation import CONTINUATION_RUN
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import CONTEXTS, DEVELOPMENT, preparation_roots
from .records import PreparationAssessmentConfig
from .topology import METHOD_CONFIG_TYPES, PreparationTaskDeclaration, expected_method_inputs, preparation_task_declarations, task_output_schemas

METHOD_CONTINUATION_RUN = "matrix-preparation-adequacy.method-completion"
METHOD_COMPACT_RUN = "matrix-preparation-adequacy.compact-method-completion"
METHOD_COMPLETION_RUNS = (METHOD_CONTINUATION_RUN, METHOD_COMPACT_RUN)
METHOD_CONTINUATION_ROLES = ("description", "forecast", "decision", "task-assessment", "evaluation")
RETAINED_METHOD_PORT = "matrix-preparation-retained-method-inputs"


def unfinished_method_tasks() -> tuple[PreparationTaskDeclaration, ...]:
    return tuple(
        t for t in preparation_task_declarations() if t.role in METHOD_CONTINUATION_ROLES
    )


def retained_method_slots(task_id: str) -> dict[str, str]:
    task = next(t for t in unfinished_method_tasks() if t.task_id == task_id)
    unfinished = {t.task_id for t in unfinished_method_tasks()}
    return {
        slot: schema
        for slot, schema in expected_method_inputs(
            task, "unused", METHOD_CONFIG_TYPES[task.role].SCHEMA
        ).items()
        if not slot.startswith("config-artifact.") and slot.rsplit(".", 1)[0] not in unfinished
    }


def _retained_slot_coordinate(slot: str) -> tuple[int, int, int, int]:
    """Declared producer/output coordinates preserve the frozen compact roster."""
    task_id, output = slot.rsplit(".", 1)
    output_order = (
        "data", "privileged-data", "privileged-report", "report", "stage",
        "prospective-task-data", "prospective-task-report",
    )
    producers = {
        f"{root.root_id}.project.r{refinement}": (
            0 if root.context == "assembling" else 2, root.index, refinement
        )
        for root in preparation_roots()
        for refinement in (1, 2)
    }
    producers.update({
        f"{DEVELOPMENT}.{role}.{context}": (group, CONTEXTS.index(context), 0)
        for role, group in (("fit", 1), ("numerical-semantics", 3))
        for context in CONTEXTS
    })
    if task_id not in producers or output not in output_order:
        raise ValueError("compact retained method slot lacks its declared producer/output coordinate")
    return (*producers[task_id], output_order.index(output))


@lru_cache(maxsize=1)
def _retained_slot_ordinals() -> dict[str, int]:
    return {
        slot: i
        for i, slot in enumerate(
            sorted(
                {
                    s
                    for t in unfinished_method_tasks()
                    for s in retained_method_slots(t.task_id)
                },
                key=_retained_slot_coordinate,
            )
        )
    }


def retained_method_artifact_id(run_id: str, role: str, slot: str) -> str:
    if run_id == METHOD_COMPACT_RUN:
        # Outcome-blind ordinal of the closed scientific slot roster. The input
        # spec still binds the complete original artifact hash, schema and custody.
        return (
            f"retained.{role}.{_retained_slot_ordinals()[slot]:04d}"
        )
    return f"artifact.{run_id}.retained.{role}.{slot}"


@dataclass(frozen=True, slots=True)
class PreparationMethodCustodyInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-method-custody-input'
    task_id: str
    output_id: str
    artifact: ArtifactIdentity
    relative_path: str
    task_receipt: ObjectIdentity
    manifest_sha256: str
    publication_commit_sha256: str

    source_export: PreparationRetainedSourceExport | None = None

    def __post_init__(self) -> None:
        if type(self.source_export) is not PreparationRetainedSourceExport:
            raise ValueError("retained completion input requires a separately verified original-source export and current target custody before work")
        validate_relative_locator(self.relative_path)
        validate_sha256(self.manifest_sha256, field_name="manifest_sha256")
        validate_sha256(self.publication_commit_sha256, field_name="publication_commit_sha256")
        tasks = {
            t.task_id: t
            for t in preparation_task_declarations()
            if t.role in ("projection", "numerical-semantics", "fit")
        }
        if self.task_id not in tasks:
            raise ValueError("retained method input changes its completed producer roster")
        schema = task_output_schemas(tasks[self.task_id].role).get(self.output_id)
        selected_task = tasks[self.task_id]
        physical_units = tuple(sorted(
            root.physical_unit_id for root in preparation_roots()
            if (
                selected_task.role == "projection"
                and self.task_id in tuple(f"{root.root_id}.project.r{r}" for r in (1, 2))
            ) or (
                selected_task.role != "projection"
                and self.task_id == f"{DEVELOPMENT}.{selected_task.role}.{root.context}"
            )
        ))
        self.source_export.validate_target(
            self.artifact, self.relative_path, self.task_receipt,
            self.manifest_sha256, self.publication_commit_sha256, physical_units,
        )
        binary = self.output_id in ("data", "privileged-data", "prospective-task-data")
        suffix = ".h5" if binary else ".canonical.json"
        if (
            self.artifact.artifact_id != f"artifact.{CONTINUATION_RUN}.{self.slot_id}"
            or self.artifact.payload_schema != schema
            or self.artifact.media_type
            != ("application/x-hdf5" if binary else "application/vnd.empirical-lawhood.canonical+json")
            or self.artifact.role != "retained-development-method-output"
            or self.artifact.extensions
            or not 0 < self.artifact.size_bytes <= 18 * 1024**2
            or self.relative_path
            != f"runs/{CONTINUATION_RUN}/outputs/{self.task_id}/{self.output_id}{suffix}"
            or self.task_receipt.object_id
            != f"receipt.{CONTINUATION_RUN}.{self.task_id}.attempt-001"
            or self.task_receipt.object_schema != 'empirical-lawhood/runtime/canonical-task-receipt'
        ):
            raise ValueError("retained method input substitutes its original custody")

    @property
    def slot_id(self) -> str:
        return f"{self.task_id}.{self.output_id}"


@dataclass(frozen=True, slots=True)
class PreparationMethodContinuation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-method-continuation'
    config_id: str
    assessment: ObjectIdentity
    source_authoring: ObjectIdentity
    source_execution_plan: ObjectIdentity
    owner_amendment: ObjectIdentity
    inputs: tuple[PreparationMethodCustodyInput, ...]
    source_commit: str
    run_id: str = METHOD_CONTINUATION_RUN
    source_run_id: str = CONTINUATION_RUN
    grants_authority: bool = False
    new_native_tasks: int = 0
    new_projection_tasks: int = 0
    new_fit_tasks: int = 0

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{40}", self.source_commit) is None:
            raise ValueError("method continuation requires an explicit source commit")
        require_sorted_unique_ids(self.inputs, attribute="slot_id", field_name="inputs")
        expected = {
            slot: schema
            for t in unfinished_method_tasks()
            for slot, schema in retained_method_slots(t.task_id).items()
        }
        if (
            self.config_id != f"{self.run_id}.continuation"
            or self.assessment.object_schema != PreparationAssessmentConfig.SCHEMA
            or self.source_authoring.object_schema
            != 'empirical-lawhood/planning/executable-study-definition'
            or self.source_authoring.object_id != f"{CONTINUATION_RUN}.authoring"
            or self.source_authoring.object_version != "1.0.0"
            or self.source_execution_plan.object_schema != 'empirical-lawhood/runtime/execution-plan'
            or self.source_execution_plan.object_version != "1.0.0"
            or self.owner_amendment.object_schema != 'empirical-lawhood/document/markdown'
            or self.run_id not in METHOD_COMPLETION_RUNS
            or self.source_run_id != CONTINUATION_RUN
            or self.grants_authority is not False
            or (self.new_native_tasks, self.new_projection_tasks, self.new_fit_tasks) != (0, 0, 0)
            or {r.slot_id: r.artifact.payload_schema for r in self.inputs} != expected
        ):
            raise ValueError("method continuation changes its exact unfinished route or science")
        for task_id in {r.task_id for r in self.inputs}:
            if len({r.task_receipt for r in self.inputs if r.task_id == task_id}) != 1:
                raise ValueError("completed producer outputs disagree on their receipt")

    def inputs_for_task(self, task_id: str) -> tuple[PreparationMethodCustodyInput, ...]:
        wanted = retained_method_slots(task_id)
        return tuple(r for r in self.inputs if r.slot_id in wanted)


class PreparationRetainedMethodPort(Protocol):
    @property
    def continuation(self) -> PreparationMethodContinuation | None: ...
    def create_source(
        self, declaration: PreparationMethodCustodyInput
    ) -> ExternalInputSource: ...


@dataclass(frozen=True, slots=True)
class NoRetainedMethodInputs:
    continuation: None = None

    def create_source(self, declaration: PreparationMethodCustodyInput) -> ExternalInputSource:
        raise ValueError("full development has no retained method imports")


def retained_method_payloads(
    plan: ProtocolExecutionPlan, sources: PreparationRetainedMethodPort, role: str
) -> tuple[ExternalInputPayload, ...]:
    continuation = sources.continuation
    if continuation is None:
        return ()
    if plan.source_plan.object_id != continuation.run_id:
        raise ValueError("retained method inputs change their issued continuation run")
    wanted = {
        r.slot_id
        for t in unfinished_method_tasks()
        if t.role == role
        for r in continuation.inputs_for_task(t.task_id)
    }
    values = []
    for row in continuation.inputs:
        if row.slot_id not in wanted:
            continue
        artifact = row.artifact
        artifact_id = retained_method_artifact_id(continuation.run_id, role, row.slot_id)
        consumers = [
            task
            for task in plan.tasks
            if any(p.logical_artifact_id == artifact_id for p in task.external_inputs)
        ]
        expected = {
            t.task_id
            for t in unfinished_method_tasks()
            if t.role == role and row.slot_id in retained_method_slots(t.task_id)
        }
        if {t.task_id for t in consumers} != expected:
            raise ValueError("retained method import changes its exact consumers")
        for task in consumers:
            spec = next(p for p in task.external_inputs if p.logical_artifact_id == artifact_id)
            if (
                spec.expected_payload_schema != artifact.payload_schema
                or spec.expected_content_sha256 != artifact.sha256
                or spec.expected_media_type != artifact.media_type
                or spec.expected_visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
                or spec.expected_outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            ):
                raise ValueError("retained method import changes content or exposure")
        parents = tuple(
            sorted(
                (
                    ArtifactLineageParent(
                        identity,
                        VisibilityCeiling.OUTCOME_VISIBLE,
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                    )
                    for identity in (
                        ObjectIdentity.from_record(artifact.artifact_id, artifact),
                        row.task_receipt,
                        *row.source_export.lineage_identities,
                    )
                ),
                key=lineage_parent_sort_key,
            )
        )
        values.append(
            ExternalInputPayload(
                artifact_id,
                artifact.payload_schema,
                ArtifactProfile.AUDITED_HDF5
                if artifact.media_type == "application/x-hdf5"
                else ArtifactProfile.CANONICAL_JSON,
                artifact.media_type,
                sources.create_source(row),
                artifact.size_bytes,
                artifact.sha256,
                artifact.size_bytes,
                min(artifact.size_bytes, MAX_EXTERNAL_INPUT_CHUNK_BYTES),
                VisibilityCeiling.OUTCOME_VISIBLE,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                tuple(p.visibility_ceiling for p in parents),
                parents,
                artifact.sha256,
            )
        )
    return tuple(sorted(values, key=lambda p: p.logical_artifact_id))


@dataclass(frozen=True, slots=True)
class _PreparationMethodImport(CanonicalRecord):
    ROLE: ClassVar[str]
    config_id: str
    continuation: ObjectIdentity
    scientific_config: ObjectIdentity

    def __post_init__(self) -> None:
        run_id = self.continuation.object_id.removesuffix(".continuation")
        if (
            run_id not in METHOD_COMPLETION_RUNS
            or self.config_id != f"{run_id}.retained-{self.ROLE}"
            or self.continuation.object_id != f"{run_id}.continuation"
            or self.continuation.object_schema != PreparationMethodContinuation.SCHEMA
            or self.scientific_config.object_schema != METHOD_CONFIG_TYPES[self.ROLE].SCHEMA
        ):
            raise ValueError("method import binding changes its exact role or configuration")


@dataclass(frozen=True, slots=True)
class PreparationForecastImport(_PreparationMethodImport):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-forecast-import'
    ROLE: ClassVar[str] = "forecast"


@dataclass(frozen=True, slots=True)
class PreparationDecisionImport(_PreparationMethodImport):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-decision-import'
    ROLE: ClassVar[str] = "decision"


@dataclass(frozen=True, slots=True)
class PreparationTaskAssessmentImport(_PreparationMethodImport):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-task-assessment-import'
    ROLE: ClassVar[str] = "task-assessment"


@dataclass(frozen=True, slots=True)
class PreparationEvaluationImport(_PreparationMethodImport):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-evaluation-import'
    ROLE: ClassVar[str] = "evaluation"


METHOD_IMPORT_TYPES: dict[str, type[_PreparationMethodImport]] = {
    "forecast": PreparationForecastImport,
    "decision": PreparationDecisionImport,
    "task-assessment": PreparationTaskAssessmentImport,
    "evaluation": PreparationEvaluationImport,
}


def method_continuation_type(role: str) -> type[CanonicalRecord]:
    return PreparationMethodContinuation if role == "description" else METHOD_IMPORT_TYPES[role]


def retained_method_port_key(role: str) -> str:
    if role not in METHOD_CONTINUATION_ROLES:
        raise ValueError("only unfinished methods have a retained input port")
    return f"{RETAINED_METHOD_PORT}.{role}"
