"""Immutable scientific run plans and lowered typed execution DAGs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling, inherited_visibility
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.worlds import WorldKind

from .artifacts import ArtifactProfile
from .capabilities import (
    CapabilityConfigRef,
    CapabilityPermission,
    CapabilityRequirement,
)
from .execution_envelope import validate_optional_jit_identities


class PlanLane(StrEnum):
    PROSPECTIVE = "PROSPECTIVE"
    EXPLORATORY = "EXPLORATORY"


class ScientificStage(StrEnum):
    PREPARE = "PREPARE"
    ACQUIRE = "ACQUIRE"
    TRANSFORM = "TRANSFORM"
    DEVELOP = "DEVELOP"
    FALSIFY = "FALSIFY"
    QUALIFY = "QUALIFY"
    FREEZE = "FREEZE"
    EVALUATE = "EVALUATE"
    REVEAL = "REVEAL"
    SYNTHESIZE = "SYNTHESIZE"
    ADMISSION = "ADMISSION"
    CONTROLLER = "CONTROLLER"
    REPORT = "REPORT"
    EXPLORE = "EXPLORE"


class ScientificInputRole(StrEnum):
    SOURCE = "SOURCE"
    PREPARED_MEDIUM = "PREPARED_MEDIUM"
    DENOMINATOR = "DENOMINATOR"
    HISTORY = "HISTORY"
    ACTION = "ACTION"
    RECEIVER = "RECEIVER"
    MODEL = "MODEL"
    QUALIFICATION = "QUALIFICATION"
    AUTHORITY = "AUTHORITY"
    OUTCOME = "OUTCOME"
    PARENT_RECEIPT = "PARENT_RECEIPT"


class BarrierKind(StrEnum):
    NONE = "NONE"
    FREEZE = "FREEZE"
    REVEAL = "REVEAL"
    AUTHORITY = "AUTHORITY"


@dataclass(frozen=True, slots=True)
class OutputTemplate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/output-template'

    output_id: str
    payload_schema: str
    profile: ArtifactProfile
    media_type: str
    filename_suffix: str

    def __post_init__(self) -> None:
        validate_stable_id(self.output_id, field_name="output_id")
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        validate_nonempty(self.filename_suffix, field_name="filename_suffix")
        if not self.filename_suffix.startswith(".") or "/" in self.filename_suffix:
            raise ValueError("output filename suffix must be a simple dotted suffix")


@dataclass(frozen=True, slots=True)
class ProtocolStepTemplate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/protocol-step-template'

    step_id: str
    stage: ScientificStage
    capability_key: str
    capability_version: str
    config: CapabilityConfigRef
    dependency_step_ids: tuple[str, ...]
    outputs: tuple[OutputTemplate, ...]
    required_permissions: tuple[CapabilityPermission, ...]
    requested_outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    resource_budget: ResourceBudget
    resource_lock_ids: tuple[str, ...]
    barrier: BarrierKind
    maximum_attempts: int
    obligation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("step_id", self.step_id),
            ("capability_key", self.capability_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.capability_version)
        require_sorted_unique_strings(
            self.dependency_step_ids,
            field_name="dependency_step_ids",
        )
        require_sorted_unique_ids(self.outputs, attribute="output_id", field_name="outputs")
        if not self.outputs:
            raise ValueError("protocol step requires at least one output")
        if self.resource_budget.output_bytes == 0:
            raise ValueError("protocol step outputs require a positive output-byte budget")
        require_sorted_unique_strings(
            self.required_permissions,
            field_name="required_permissions",
        )
        require_sorted_unique_strings(
            self.resource_lock_ids,
            field_name="resource_lock_ids",
        )
        require_sorted_unique_strings(
            self.obligation_ids,
            field_name="obligation_ids",
            allow_empty=False,
        )
        if self.maximum_attempts <= 0:
            raise ValueError("protocol step maximum attempts must be positive")
        if CapabilityPermission.APPROVE_NONACTUATING in self.required_permissions:
            raise ValueError("protocol workers cannot request approval authority")
        if self.stage is ScientificStage.FREEZE and self.barrier is not BarrierKind.FREEZE:
            raise ValueError("freeze stage requires a freeze barrier")
        if self.stage in {ScientificStage.EVALUATE, ScientificStage.REVEAL} and (
            self.barrier is not BarrierKind.REVEAL
        ):
            raise ValueError("evaluation/reveal stages require a reveal barrier")


@dataclass(frozen=True, slots=True)
class ProtocolTemplate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/protocol-template'

    template_id: str
    template_version: str
    steps: tuple[ProtocolStepTemplate, ...]
    requires_model_set: bool
    requests_controller: bool
    nonactuating: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.template_id, field_name="template_id")
        validate_semantic_version(self.template_version)
        require_sorted_unique_ids(self.steps, attribute="step_id", field_name="steps")
        if not self.steps:
            raise ValueError("protocol template requires steps")
        _validate_dependencies(
            tuple(step.step_id for step in self.steps),
            {step.step_id: step.dependency_step_ids for step in self.steps},
        )
        if self.requests_controller and not any(
            step.stage is ScientificStage.CONTROLLER for step in self.steps
        ):
            raise ValueError("controller template lacks a controller stage")


@dataclass(frozen=True, slots=True)
class SnapshotVerification(CanonicalRecord):
    """Content-identity attestation required before read-only exploration compiles."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/snapshot-verification'

    verification_id: str
    snapshot: ObjectIdentity
    artifact_ids: tuple[str, ...]
    verification_check_ids: tuple[str, ...]
    verifier_key: str
    verifier_version: str
    verified: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("verification_id", self.verification_id),
            ("verifier_key", self.verifier_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.verifier_version)
        require_sorted_unique_strings(
            self.artifact_ids,
            field_name="artifact_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.verification_check_ids,
            field_name="verification_check_ids",
            allow_empty=False,
        )
        if not self.verified:
            raise ValueError("snapshot verification must be affirmative")


@dataclass(frozen=True, slots=True)
class ArtifactOutputSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-output-spec'

    output_id: str
    logical_artifact_id: str
    relative_path: str
    payload_schema: str
    profile: ArtifactProfile
    media_type: str
    visibility_ceiling: VisibilityCeiling
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.output_id, field_name="output_id")
        validate_stable_id(self.logical_artifact_id, field_name="logical_artifact_id")
        validate_relative_locator(self.relative_path)
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("task output visibility cannot be lowered")


@dataclass(frozen=True, slots=True)
class ExternalInputSpec(CanonicalRecord):
    """Frozen logical identity expected from an external input resolver."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/external-input-spec'

    input_id: str
    logical_artifact_id: str
    expected_content_sha256: str | None
    expected_payload_schema: str | None
    expected_media_type: str | None
    expected_size_bytes: int | None
    expected_visibility_ceiling: VisibilityCeiling | None
    expected_outcome_access: OutcomeAccess | None
    identity_scope_sha256: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        validate_stable_id(self.logical_artifact_id, field_name="logical_artifact_id")
        if self.expected_content_sha256 is not None:
            validate_sha256(
                self.expected_content_sha256,
                field_name="expected_content_sha256",
            )
        if self.expected_payload_schema is not None:
            validate_schema(self.expected_payload_schema)
        if self.expected_media_type is not None:
            validate_nonempty(self.expected_media_type, field_name="expected_media_type")
        if self.expected_size_bytes is not None and self.expected_size_bytes < 0:
            raise ValueError("expected external input size must be nonnegative")
        if self.identity_scope_sha256 is not None:
            validate_sha256(
                self.identity_scope_sha256,
                field_name="identity_scope_sha256",
            )
        if self.expected_content_sha256 is None and self.identity_scope_sha256 is None:
            raise ValueError("external input requires content or enclosing-scope identity")


@dataclass(frozen=True, slots=True)
class ScientificInputSpec(CanonicalRecord):
    """Exact scientific edge retained alongside operational input resolution."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/scientific-input-spec'

    edge_id: str
    consumer_input_id: str
    scientific_role: ScientificInputRole
    producer_task_id: str | None
    producer_output_id: str | None
    external_input_id: str | None
    logical_artifact_id: str
    operational_logical_artifact_id: str
    payload_schema: str
    media_type: str
    maximum_size_bytes: int
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    barrier: BarrierKind

    def __post_init__(self) -> None:
        for name, value in (
            ("edge_id", self.edge_id),
            ("consumer_input_id", self.consumer_input_id),
            ("logical_artifact_id", self.logical_artifact_id),
            (
                "operational_logical_artifact_id",
                self.operational_logical_artifact_id,
            ),
        ):
            validate_stable_id(value, field_name=name)
        internal = self.producer_task_id is not None or self.producer_output_id is not None
        external = self.external_input_id is not None
        if internal == external:
            raise ValueError("scientific input must bind one internal or external producer")
        if internal:
            if self.producer_task_id is None or self.producer_output_id is None:
                raise ValueError("internal scientific input requires task and output IDs")
            validate_stable_id(self.producer_task_id, field_name="producer_task_id")
            validate_stable_id(self.producer_output_id, field_name="producer_output_id")
        else:
            assert self.external_input_id is not None
            validate_stable_id(self.external_input_id, field_name="external_input_id")
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        if self.maximum_size_bytes <= 0:
            raise ValueError("scientific input requires a positive size bound")


@dataclass(frozen=True, slots=True)
class ProtocolPlanStep(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/protocol-plan-step'

    step_id: str
    stage: ScientificStage
    capability: CapabilityRequirement
    dependency_step_ids: tuple[str, ...]
    external_inputs: tuple[ExternalInputSpec, ...]
    outputs: tuple[ArtifactOutputSpec, ...]
    resource_lock_ids: tuple[str, ...]
    barrier: BarrierKind
    maximum_attempts: int
    obligation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.step_id, field_name="step_id")
        require_sorted_unique_strings(
            self.dependency_step_ids,
            field_name="dependency_step_ids",
        )
        require_sorted_unique_ids(
            self.external_inputs,
            attribute="input_id",
            field_name="external_inputs",
        )
        logical_ids = tuple(value.logical_artifact_id for value in self.external_inputs)
        require_sorted_unique_strings(tuple(sorted(logical_ids)), field_name="external logical IDs")
        require_sorted_unique_ids(self.outputs, attribute="output_id", field_name="outputs")
        if not self.outputs:
            raise ValueError("run-plan step requires at least one output")
        require_sorted_unique_strings(
            self.resource_lock_ids,
            field_name="resource_lock_ids",
        )
        require_sorted_unique_strings(
            self.obligation_ids,
            field_name="obligation_ids",
            allow_empty=False,
        )
        if self.maximum_attempts <= 0:
            raise ValueError("run-plan step maximum attempts must be positive")
        if CapabilityPermission.APPROVE_NONACTUATING in self.capability.required_permissions:
            raise ValueError("run-plan workers cannot request approval authority")
        output_schemas = tuple(sorted({output.payload_schema for output in self.outputs}))
        if output_schemas != self.capability.required_output_schema_ids:
            raise ValueError("run-plan output schemas differ from capability requirement")

    @property
    def external_input_artifact_ids(self) -> tuple[str, ...]:
        return tuple(sorted(value.logical_artifact_id for value in self.external_inputs))


@dataclass(frozen=True, slots=True)
class RunPlanStep(ProtocolPlanStep):
    "Exact-edge step extending ProtocolPlanStep."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-plan-step'

    scientific_inputs: tuple[ScientificInputSpec, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        super(RunPlanStep, self).__post_init__()
        require_sorted_unique_ids(
            self.scientific_inputs,
            attribute="edge_id",
            field_name="scientific_inputs",
        )
        consumer_input_ids = tuple(value.consumer_input_id for value in self.scientific_inputs)
        if len(set(consumer_input_ids)) != len(consumer_input_ids):
            raise ValueError("scientific consumer inputs must be unique")
        producers = tuple(
            sorted(
                {
                    value.producer_task_id
                    for value in self.scientific_inputs
                    if value.producer_task_id is not None
                }
            )
        )
        if producers != self.dependency_step_ids:
            raise ValueError("exact scientific inputs differ from the declared step dependencies")
        external_by_input = {value.input_id: value for value in self.external_inputs}
        for value in self.scientific_inputs:
            if value.external_input_id is None:
                continue
            operational = external_by_input.get(value.consumer_input_id)
            if (
                operational is None
                or operational.logical_artifact_id != value.operational_logical_artifact_id
                or operational.expected_payload_schema != value.payload_schema
                or operational.expected_media_type != value.media_type
                or operational.expected_visibility_ceiling is not value.visibility_ceiling
                or operational.expected_outcome_access is not value.outcome_access
            ):
                raise ValueError("scientific external input differs from operational resolution")


@dataclass(frozen=True, slots=True)
class ProtocolRunPlan(CanonicalRecord):
    """Frozen scientific protocol before infrastructure lowering."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/protocol-run-plan'

    run_plan_id: str
    campaign: ObjectIdentity
    system: ObjectIdentity
    claims: tuple[ObjectIdentity, ...]
    experiment: ObjectIdentity
    authorization: ObjectIdentity
    model_set: ObjectIdentity | None
    world_id: str
    world_kind: WorldKind
    physical_preparation_ids: tuple[str, ...]
    numerical_view_ids: tuple[str, ...]
    registry_sha256: str
    implementation_commit: str
    steps: tuple[ProtocolPlanStep, ...]
    robust_model_set_required: bool
    controller_requested: bool
    nonactuating: bool
    frozen: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.run_plan_id, field_name="run_plan_id")
        validate_stable_id(self.world_id, field_name="world_id")
        require_sorted_unique_ids(self.claims, attribute="object_id", field_name="claims")
        if not self.claims:
            raise ValueError("run plan requires claims")
        self._validate_physical_preparations()
        require_sorted_unique_strings(
            self.numerical_view_ids,
            field_name="numerical_view_ids",
        )
        validate_sha256(self.registry_sha256, field_name="registry_sha256")
        _validate_git_commit(self.implementation_commit)
        require_sorted_unique_ids(self.steps, attribute="step_id", field_name="steps")
        if not self.steps:
            raise ValueError("run plan requires steps")
        _validate_step_graph(self.steps)
        _validate_output_uniqueness(self.steps)
        _validate_freeze_reveal(self.steps)
        if self.robust_model_set_required and self.model_set is None:
            raise ValueError("robust run plan requires a frozen model set")
        if self.controller_requested:
            if self.model_set is None:
                raise ValueError("controller run plan requires a frozen model set")
            if not any(step.stage is ScientificStage.CONTROLLER for step in self.steps):
                raise ValueError("controller run plan lacks a controller step")
        if self.nonactuating and any(
            CapabilityPermission.COMMAND_ACTUATOR in step.capability.required_permissions
            for step in self.steps
        ):
            raise ValueError("nonactuating run plan requests actuator authority")
        if set(self.physical_preparation_ids).intersection(self.numerical_view_ids):
            raise ValueError("numerical views cannot be counted as physical preparations")
        if not self.frozen:
            raise ValueError("RunPlan must be frozen")

    def _validate_physical_preparations(self) -> None:
        if self.experiment.object_schema == 'empirical-lawhood/kernel/retrospective-experiment-spec':
            raise ValueError("historical experiment requires its explicit run-plan schema")
        require_sorted_unique_strings(
            self.physical_preparation_ids, field_name="physical_preparation_ids", allow_empty=False
        )

    def topological_step_ids(self) -> tuple[str, ...]:
        return _topological_order(
            tuple(step.step_id for step in self.steps),
            {step.step_id: step.dependency_step_ids for step in self.steps},
        )


@dataclass(frozen=True, slots=True)
class CandidateRunPlan(ProtocolRunPlan):
    """Current issued run plan with an exact candidate and edge binding."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-run-plan'

    steps: tuple[RunPlanStep, ...]
    candidate: ObjectIdentity = field(kw_only=True)

    def __post_init__(self) -> None:
        super(CandidateRunPlan, self).__post_init__()
        if not all(isinstance(value, RunPlanStep) for value in self.steps):
            raise ValueError("candidate run plan requires only exact-edge steps")
        outputs_by_task = {
            step.step_id: {output.output_id: output for output in step.outputs}
            for step in self.steps
        }
        for step in self.steps:
            for value in step.scientific_inputs:
                if value.producer_task_id is None:
                    continue
                output = outputs_by_task[value.producer_task_id].get(value.producer_output_id or "")
                if (
                    output is None
                    or output.logical_artifact_id != value.operational_logical_artifact_id
                    or output.payload_schema != value.payload_schema
                    or output.media_type != value.media_type
                ):
                    raise ValueError(
                        "run-plan scientific dependency differs from its producer output"
                    )


@dataclass(frozen=True, slots=True)
class EnvelopeRunPlan(CandidateRunPlan):
    "Run plan retaining the exact candidate graph and issued extension/envelope identities."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/envelope-run-plan'

    issued_extension_set: ObjectIdentity = field(kw_only=True)
    execution_envelope_spec: ObjectIdentity = field(kw_only=True)

    def __post_init__(self) -> None:
        super(EnvelopeRunPlan, self).__post_init__()
        if (
            self.issued_extension_set.object_schema
            != 'empirical-lawhood/runtime/issued-extension-set'
            or self.execution_envelope_spec.object_schema
            != 'empirical-lawhood/runtime/execution-envelope-spec'
        ):
            raise ValueError("envelope run-plan extension/envelope identity is incompatible")


@dataclass(frozen=True, slots=True)
class RunPlan(CandidateRunPlan):
    """Current deadline-free plan with compilation evidence when applicable."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/run-plan'

    issued_extension_set: ObjectIdentity = field(kw_only=True)
    execution_resource_envelope_spec: ObjectIdentity = field(kw_only=True)
    predevelopment_jit_signature_census: ObjectIdentity | None = field(kw_only=True)
    jit_graph_signature_manifest: ObjectIdentity | None = field(kw_only=True)

    def __post_init__(self) -> None:
        super(RunPlan, self).__post_init__()
        expected = (
            'empirical-lawhood/runtime/issued-extension-set',
            'empirical-lawhood/runtime/execution-resource-envelope-spec',
        )
        observed = (
            self.issued_extension_set.object_schema,
            self.execution_resource_envelope_spec.object_schema,
        )
        if observed != expected:
            raise ValueError("deadline-free run-plan resource identity is incompatible")
        validate_optional_jit_identities(
            self.predevelopment_jit_signature_census, self.jit_graph_signature_manifest
        )


@dataclass(frozen=True, slots=True)
class ProtocolExecutionTask(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/protocol-execution-task'

    task_id: str
    stage: ScientificStage
    capability: CapabilityRequirement
    capability_implementation_sha256: str
    dependency_task_ids: tuple[str, ...]
    external_inputs: tuple[ExternalInputSpec, ...]
    outputs: tuple[ArtifactOutputSpec, ...]
    resource_lock_ids: tuple[str, ...]
    barrier: BarrierKind
    maximum_attempts: int
    obligation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.task_id, field_name="task_id")
        validate_sha256(
            self.capability_implementation_sha256,
            field_name="capability_implementation_sha256",
        )
        require_sorted_unique_strings(
            self.dependency_task_ids,
            field_name="dependency_task_ids",
        )
        require_sorted_unique_ids(
            self.external_inputs,
            attribute="input_id",
            field_name="external_inputs",
        )
        logical_ids = tuple(value.logical_artifact_id for value in self.external_inputs)
        require_sorted_unique_strings(tuple(sorted(logical_ids)), field_name="external logical IDs")
        require_sorted_unique_ids(self.outputs, attribute="output_id", field_name="outputs")
        if not self.outputs:
            raise ValueError("execution task requires at least one output")
        require_sorted_unique_strings(
            self.resource_lock_ids,
            field_name="resource_lock_ids",
        )
        require_sorted_unique_strings(
            self.obligation_ids,
            field_name="obligation_ids",
            allow_empty=False,
        )
        if self.maximum_attempts <= 0:
            raise ValueError("execution task maximum attempts must be positive")
        if CapabilityPermission.APPROVE_NONACTUATING in self.capability.required_permissions:
            raise ValueError("an execution worker cannot impersonate the approval gate")

    @property
    def external_input_artifact_ids(self) -> tuple[str, ...]:
        return tuple(sorted(value.logical_artifact_id for value in self.external_inputs))


@dataclass(frozen=True, slots=True)
class ExecutionTask(ProtocolExecutionTask):
    """Current execution task retaining the exact scientific input edge set."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/execution-task'

    scientific_inputs: tuple[ScientificInputSpec, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        super(ExecutionTask, self).__post_init__()
        require_sorted_unique_ids(
            self.scientific_inputs,
            attribute="edge_id",
            field_name="scientific_inputs",
        )
        consumer_input_ids = tuple(value.consumer_input_id for value in self.scientific_inputs)
        if len(set(consumer_input_ids)) != len(consumer_input_ids):
            raise ValueError("scientific consumer inputs must be unique")
        producers = tuple(
            sorted(
                {
                    value.producer_task_id
                    for value in self.scientific_inputs
                    if value.producer_task_id is not None
                }
            )
        )
        if producers != self.dependency_task_ids:
            raise ValueError("exact scientific inputs differ from the declared task dependencies")


@dataclass(frozen=True, slots=True)
class ProtocolExecutionPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/protocol-execution-plan'

    execution_plan_id: str
    lane: PlanLane
    source_plan: ObjectIdentity
    registry_sha256: str
    implementation_commit: str
    tasks: tuple[ProtocolExecutionTask, ...]
    nonactuating: bool
    frozen: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.execution_plan_id, field_name="execution_plan_id")
        validate_sha256(self.registry_sha256, field_name="registry_sha256")
        _validate_git_commit(self.implementation_commit)
        require_sorted_unique_ids(self.tasks, attribute="task_id", field_name="tasks")
        if not self.tasks:
            raise ValueError("execution plan requires tasks")
        task_ids = tuple(task.task_id for task in self.tasks)
        dependencies = {task.task_id: task.dependency_task_ids for task in self.tasks}
        _validate_dependencies(task_ids, dependencies)
        _validate_output_uniqueness(self.tasks)
        _validate_freeze_reveal(self.tasks)
        if self.nonactuating and any(
            CapabilityPermission.COMMAND_ACTUATOR in task.capability.required_permissions
            for task in self.tasks
        ):
            raise ValueError("nonactuating execution plan requests actuator authority")
        if not self.frozen:
            raise ValueError("ExecutionPlan must be frozen")

    def topological_task_ids(self) -> tuple[str, ...]:
        return _topological_order(
            tuple(task.task_id for task in self.tasks),
            {task.task_id: task.dependency_task_ids for task in self.tasks},
        )

    @property
    def minimum_free_bytes(self) -> int:
        """Frozen operation-specific floor from the complete output allocation."""

        return sum(task.capability.requested_resources.output_bytes for task in self.tasks)

    def parallel_ready_groups(self) -> tuple[tuple[str, ...], ...]:
        dependencies = {task.task_id: set(task.dependency_task_ids) for task in self.tasks}
        completed: set[str] = set()
        groups: list[tuple[str, ...]] = []
        while len(completed) < len(self.tasks):
            ready = tuple(
                sorted(
                    task_id
                    for task_id, required in dependencies.items()
                    if task_id not in completed and required.issubset(completed)
                )
            )
            if not ready:
                raise ValueError("execution plan graph has no ready task")
            groups.append(ready)
            completed.update(ready)
        return tuple(groups)


@dataclass(frozen=True, slots=True)
class CandidateExecutionPlan(ProtocolExecutionPlan):
    """Current issued execution plan with exact-edge tasks."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-execution-plan'

    tasks: tuple[ExecutionTask, ...]
    candidate: ObjectIdentity = field(kw_only=True)

    def __post_init__(self) -> None:
        super(CandidateExecutionPlan, self).__post_init__()
        if not all(isinstance(value, ExecutionTask) for value in self.tasks):
            raise ValueError("candidate execution plan requires only exact-edge tasks")


@dataclass(frozen=True, slots=True)
class EnvelopeExecutionPlan(CandidateExecutionPlan):
    "Operational lowering retaining the exact issued extension/envelope roots."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/envelope-execution-plan'

    issued_extension_set: ObjectIdentity = field(kw_only=True)
    execution_envelope_spec: ObjectIdentity = field(kw_only=True)

    def __post_init__(self) -> None:
        super(EnvelopeExecutionPlan, self).__post_init__()
        if (
            self.issued_extension_set.object_schema
            != 'empirical-lawhood/runtime/issued-extension-set'
            or self.execution_envelope_spec.object_schema
            != 'empirical-lawhood/runtime/execution-envelope-spec'
        ):
            raise ValueError("envelope execution-plan extension/envelope identity is incompatible")


@dataclass(frozen=True, slots=True)
class ExecutionPlan(CandidateExecutionPlan):
    "Deadline-free execution plan retaining exact resource/JIT identities."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/execution-plan'

    issued_extension_set: ObjectIdentity = field(kw_only=True)
    execution_resource_envelope_spec: ObjectIdentity = field(kw_only=True)
    predevelopment_jit_signature_census: ObjectIdentity | None = field(kw_only=True)
    jit_graph_signature_manifest: ObjectIdentity | None = field(kw_only=True)

    def __post_init__(self) -> None:
        super(ExecutionPlan, self).__post_init__()
        expected = (
            'empirical-lawhood/runtime/issued-extension-set',
            'empirical-lawhood/runtime/execution-resource-envelope-spec',
        )
        observed = (
            self.issued_extension_set.object_schema,
            self.execution_resource_envelope_spec.object_schema,
        )
        if observed != expected:
            raise ValueError("deadline-free execution-plan resource identity is incompatible")
        validate_optional_jit_identities(
            self.predevelopment_jit_signature_census, self.jit_graph_signature_manifest
        )


@dataclass(frozen=True, slots=True)
class FrozenCompilationExecutionPlan(CandidateExecutionPlan):
    """Decode-only frozen declaration; current compilers and packages never emit it."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/frozen-compilation-execution-plan'

    issued_extension_set: ObjectIdentity = field(kw_only=True)
    execution_resource_envelope_spec: ObjectIdentity = field(kw_only=True)
    predevelopment_jit_signature_census: ObjectIdentity = field(kw_only=True)
    jit_graph_signature_manifest: ObjectIdentity = field(kw_only=True)

    def __post_init__(self) -> None:
        super(FrozenCompilationExecutionPlan, self).__post_init__()
        expected = (
            'empirical-lawhood/runtime/issued-extension-set',
            'empirical-lawhood/runtime/compiled-resource-envelope-reference',
            'empirical-lawhood/runtime/predevelopment-jit-signature-census',
            'empirical-lawhood/runtime/jit-graph-signature-manifest',
        )
        observed = (
            self.issued_extension_set.object_schema,
            self.execution_resource_envelope_spec.object_schema,
            self.predevelopment_jit_signature_census.object_schema,
            self.jit_graph_signature_manifest.object_schema,
        )
        if observed != expected:
            raise ValueError("frozen-compilation execution-plan resource/JIT identity is incompatible")


def _validate_git_commit(value: str) -> None:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError("implementation commit must be a lowercase Git SHA-1")


def _validate_step_graph(steps: tuple[ProtocolPlanStep, ...]) -> None:
    _validate_dependencies(
        tuple(step.step_id for step in steps),
        {step.step_id: step.dependency_step_ids for step in steps},
    )


def _validate_dependencies(
    identifiers: tuple[str, ...],
    dependencies: dict[str, tuple[str, ...]],
) -> None:
    known = set(identifiers)
    for identifier, required in dependencies.items():
        unknown = set(required) - known
        if unknown:
            raise ValueError(f"{identifier!r} has unknown dependencies: {sorted(unknown)}")
        if identifier in required:
            raise ValueError("task cannot depend on itself")
    _topological_order(identifiers, dependencies)


def _topological_order(
    identifiers: tuple[str, ...],
    dependencies: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    remaining = set(identifiers)
    completed: set[str] = set()
    ordered: list[str] = []
    while remaining:
        ready = sorted(
            identifier
            for identifier in remaining
            if set(dependencies[identifier]).issubset(completed)
        )
        if not ready:
            raise ValueError("task dependency graph contains a cycle")
        for identifier in ready:
            ordered.append(identifier)
            completed.add(identifier)
            remaining.remove(identifier)
    return tuple(ordered)


def _validate_output_uniqueness(
    steps: tuple[ProtocolPlanStep, ...] | tuple[ProtocolExecutionTask, ...],
) -> None:
    output_ids = [output.output_id for step in steps for output in step.outputs]
    logical_ids = [output.logical_artifact_id for step in steps for output in step.outputs]
    paths = [output.relative_path for step in steps for output in step.outputs]
    if any(len(values) != len(set(values)) for values in (output_ids, logical_ids, paths)):
        raise ValueError("plan output IDs, logical identities and paths must be unique")


def _validate_freeze_reveal(
    steps: tuple[ProtocolPlanStep, ...] | tuple[ProtocolExecutionTask, ...],
) -> None:
    identifiers = {getattr(step, "step_id", getattr(step, "task_id", "")): step for step in steps}
    dependencies = {
        identifier: getattr(step, "dependency_step_ids", getattr(step, "dependency_task_ids", ()))
        for identifier, step in identifiers.items()
    }
    freeze_ids = {
        identifier for identifier, step in identifiers.items() if step.barrier is BarrierKind.FREEZE
    }
    for identifier, step in identifiers.items():
        if step.stage not in {ScientificStage.EVALUATE, ScientificStage.REVEAL}:
            continue
        ancestors = _ancestors(identifier, dependencies)
        if not freeze_ids.intersection(ancestors):
            raise ValueError("evaluation/reveal task lacks an upstream freeze barrier")


def _ancestors(identifier: str, dependencies: dict[str, tuple[str, ...]]) -> set[str]:
    pending = list(dependencies[identifier])
    observed: set[str] = set()
    while pending:
        current = pending.pop()
        if current in observed:
            continue
        observed.add(current)
        pending.extend(dependencies[current])
    return observed


@dataclass(frozen=True, slots=True)
class RetrospectiveStudyRunPlan(CandidateRunPlan):
    """Exact-edge historical preissue plan; unknown physical roster stays empty."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retrospective-study-run-plan'

    def _validate_physical_preparations(self) -> None:
        _validate_historical_preparations(self)


@dataclass(frozen=True, slots=True)
class ResourceBoundRetrospectiveStudyRunPlan(RunPlan):
    "Resource-bound historical run using deadline-free execution lowering."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/resource-bound-retrospective-study-run-plan'

    def _validate_physical_preparations(self) -> None:
        _validate_historical_preparations(self)


def _validate_historical_preparations(plan: ProtocolRunPlan) -> None:
    if plan.experiment.object_schema != 'empirical-lawhood/kernel/retrospective-experiment-spec':
        raise ValueError("historical run plan requires an exact historical experiment identity")
    require_sorted_unique_strings(
        plan.physical_preparation_ids, field_name="physical_preparation_ids"
    )
    if plan.controller_requested or plan.robust_model_set_required or not plan.nonactuating:
        raise ValueError("historical reanalysis is nonactuating and cannot qualify a controller")
