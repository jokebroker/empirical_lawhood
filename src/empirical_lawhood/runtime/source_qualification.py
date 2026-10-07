"""Noncontact topology and substrate joins for source/response qualification."""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_stable_id,
    require_sorted_unique_strings,
)
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment, PredecessorBoundSourceQualificationExperiment, QualifiedSourceUseExperiment, ProspectiveRetainedSourceUse, SourceQualificationRetainedPredecessor

from .artifacts import CanonicalTaskReceipt
from .response_experiment_ports import NativeInteractionKind, ResponseSubstrateBinding
from .plans import ProtocolExecutionPlan, ExecutionTask, ScientificInputRole, ScientificStage


@dataclass(frozen=True, slots=True)
class FreshSourceQualificationSubstrateBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/fresh-source-qualification-substrate-binding'
    CARRIER_SCHEMA: ClassVar[str] = FreshSourceQualificationExperiment.SCHEMA

    binding_id: str
    qualification_carrier: ObjectIdentity
    substrate: ResponseSubstrateBinding
    source_config: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if self.qualification_carrier.object_schema != self.CARRIER_SCHEMA:
            raise ValueError("qualification substrate binding requires its exact carrier")
        if (
            self.substrate.interaction_kind is not NativeInteractionKind.INTERACTIVE_EXECUTION
            or self.substrate.action_contract is None
            or self.substrate.native_config != self.source_config
        ):
            raise ValueError("qualification substrate binding changes interactive native semantics")


@dataclass(frozen=True, slots=True)
class PredecessorBoundSourceQualificationSubstrateBinding(FreshSourceQualificationSubstrateBinding):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/predecessor-bound-source-qualification-substrate-binding'
    CARRIER_SCHEMA: ClassVar[str] = PredecessorBoundSourceQualificationExperiment.SCHEMA


@dataclass(frozen=True, slots=True)
class QualifiedSourceUseSubstrateBinding(PredecessorBoundSourceQualificationSubstrateBinding):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/qualified-source-use-substrate-binding'
    CARRIER_SCHEMA: ClassVar[str] = QualifiedSourceUseExperiment.SCHEMA


@dataclass(frozen=True, slots=True)
class ProspectiveRetainedSourceUseBinding(QualifiedSourceUseSubstrateBinding):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prospective-retained-source-use-binding'
    CARRIER_SCHEMA: ClassVar[str] = ProspectiveRetainedSourceUse.SCHEMA


@dataclass(frozen=True, slots=True)
class SourceQualificationStage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-qualification-stage'

    task_id: str
    stage: ScientificStage
    dependency_task_ids: tuple[str, ...]
    owner: ObjectIdentity
    config: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.task_id, field_name="task_id")
        require_sorted_unique_strings(self.dependency_task_ids, field_name="dependency_task_ids")
        if self.stage not in {
            ScientificStage.PREPARE,
            ScientificStage.TRANSFORM,
            ScientificStage.EVALUATE,
        }:
            raise ValueError("qualification topology cannot add a law or controller stage")
        if (
            self.task_id in self.dependency_task_ids
            or (self.stage is ScientificStage.PREPARE and len(self.dependency_task_ids) > 1)
            or (self.stage is not ScientificStage.PREPARE and not self.dependency_task_ids)
        ):
            raise ValueError("qualification stage changes its continuation/projection join")
        if self.owner.object_schema != 'empirical-lawhood/runtime/capability-manifest':
            raise ValueError("qualification stage requires an exact capability owner")


@dataclass(frozen=True, slots=True)
class CompiledFreshSourceQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-fresh-source-qualification'
    CARRIER_SCHEMA: ClassVar[str] = FreshSourceQualificationExperiment.SCHEMA

    extension_set: ObjectIdentity
    stages: tuple[SourceQualificationStage, ...]

    def __post_init__(self) -> None:
        if self.extension_set.object_schema != self.CARRIER_SCHEMA:
            raise ValueError("qualification topology names another carrier schema")
        ids = tuple(stage.task_id for stage in self.stages)
        if not 3 <= len(ids) <= 8193 or ids != tuple(sorted(set(ids))):
            raise ValueError("qualification topology requires a bounded unique ordered roster")
        if sum(stage.stage is ScientificStage.EVALUATE for stage in self.stages) != 1 or any(
            not set(stage.dependency_task_ids) <= set(ids) for stage in self.stages
        ):
            raise ValueError("qualification topology loses its evaluator or declared predecessor")


@dataclass(frozen=True, slots=True)
class CompiledPredecessorBoundSourceQualification(CompiledFreshSourceQualification):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-predecessor-bound-source-qualification'
    CARRIER_SCHEMA: ClassVar[str] = PredecessorBoundSourceQualificationExperiment.SCHEMA


@dataclass(frozen=True, slots=True)
class QualifiedSourceUseStage(SourceQualificationStage):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/qualified-source-use-stage'

    def __post_init__(self) -> None:
        from empirical_lawhood.planning.source_qualification import QualifiedSourceUseStage as Binding

        Binding(self.task_id, self.stage.value, self.dependency_task_ids, self.owner, self.config)


@dataclass(frozen=True, slots=True)
class CompiledQualifiedSourceUse(CompiledFreshSourceQualification):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-qualified-source-use'
    CARRIER_SCHEMA: ClassVar[str] = QualifiedSourceUseExperiment.SCHEMA
    stages: tuple[QualifiedSourceUseStage, ...]

    def __post_init__(self) -> None:
        ids = tuple(s.task_id for s in self.stages)
        if (
            self.extension_set.object_schema != self.CARRIER_SCHEMA
            or not 3 <= len(ids) <= 8193
            or ids != tuple(sorted(set(ids)))
        ):
            raise ValueError("source-use compilation changes its carrier or finite census")
        if not any(s.stage is ScientificStage.EVALUATE for s in self.stages) or any(
            not set(s.dependency_task_ids) <= set(ids) for s in self.stages
        ):
            raise ValueError("source-use compilation loses evaluation or a predecessor")


@dataclass(frozen=True, slots=True)
class CompiledProspectiveRetainedSourceUse(CompiledQualifiedSourceUse):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/compiled-prospective-retained-source-use'
    CARRIER_SCHEMA: ClassVar[str] = ProspectiveRetainedSourceUse.SCHEMA


def derive_source_qualification_topology(
    extension: FreshSourceQualificationExperiment,
) -> CompiledFreshSourceQualification:
    """Project declared continuation/view joins; never build an execution plan."""
    if isinstance(extension, QualifiedSourceUseExperiment):
        compiled = (
            CompiledProspectiveRetainedSourceUse
            if isinstance(extension, ProspectiveRetainedSourceUse)
            else CompiledQualifiedSourceUse
        )
        return compiled(
            ObjectIdentity.from_record(extension.extension_set_id, extension),
            tuple(
                QualifiedSourceUseStage(
                    s.task_id, ScientificStage(s.stage), s.dependency_task_ids, s.owner, s.config
                )
                for s in extension.stages
            ),
        )
    new_segments = {segment.segment_id for segment in extension.segments}
    stages = [
        SourceQualificationStage(
            segment.segment_id,
            ScientificStage.PREPARE,
            (segment.predecessor_segment_id,)
            if segment.predecessor_segment_id is not None
            and segment.predecessor_segment_id in new_segments
            else (),
            extension.source_capability,
            extension.source_config,
        )
        for segment in extension.segments
    ]
    stages.extend(
        SourceQualificationStage(
            view.view_id,
            ScientificStage.TRANSFORM,
            view.segment_ids,
            extension.projection_capability,
            extension.projection_config,
        )
        for view in extension.views
    )
    stages.append(
        SourceQualificationStage(
            extension.evaluator_task_id,
            ScientificStage.EVALUATE,
            tuple(view.view_id for view in extension.views),
            extension.evaluator_capability,
            extension.evaluator_config,
        )
    )
    compiled_type = (
        CompiledPredecessorBoundSourceQualification
        if isinstance(extension, PredecessorBoundSourceQualificationExperiment)
        else CompiledFreshSourceQualification
    )
    return compiled_type(
        ObjectIdentity.from_record(extension.extension_set_id, extension),
        tuple(sorted(stages, key=lambda stage: stage.task_id)),
    )


def retained_qualification_input_reasons(
    extension: PredecessorBoundSourceQualificationExperiment, plan: ProtocolExecutionPlan
) -> tuple[str, ...]:
    """Check exact external artifact/receipt edges in preview and execution.

    Byte authentication remains with normal custody/input resolution. Native
    state and receipt-content correspondence must also pass before source effects.
    """
    retained = {r.segment_id: r for r in extension.retained_predecessors}
    tasks = {t.task_id: t for t in plan.tasks}
    reasons = []
    for segment in extension.segments:
        prior = retained.get(segment.predecessor_segment_id or "")
        if prior is None:
            continue
        imported = isinstance(extension, ProspectiveRetainedSourceUse)
        task = tasks.get(prior.segment_id if imported else segment.segment_id)
        reason = f"QUALIFICATION_RETAINED_INPUT_DIFFERS:{segment.segment_id}"
        if not isinstance(task, ExecutionTask):
            reasons.append(reason)
            continue
        expected = tuple((a, ScientificInputRole.PREPARED_MEDIUM) for a in prior.artifacts) + (
            (prior.task_receipt, ScientificInputRole.PARENT_RECEIPT),
        )
        scientific = tuple(
            edge
            for edge in task.scientific_inputs
            if edge.scientific_role
            in (ScientificInputRole.PREPARED_MEDIUM, ScientificInputRole.PARENT_RECEIPT)
        )
        if len(scientific) != len(expected) or task.dependency_task_ids:
            reasons.append(reason)
        for artifact, role in expected:
            physical = tuple(
                e for e in task.external_inputs if e.logical_artifact_id == artifact.artifact_id
            )
            edges = tuple(e for e in scientific if e.logical_artifact_id == artifact.artifact_id)
            if len(physical) != 1 or len(edges) != 1:
                reasons.append(reason)
                continue
            physical_input, edge = physical[0], edges[0]
            if (
                physical_input.expected_content_sha256 != artifact.sha256
                or physical_input.expected_payload_schema != artifact.payload_schema
                or physical_input.expected_media_type != artifact.media_type
                # Candidate lowering carries the bound on the scientific edge
                # and leaves this optional operational size unspecified. Keep
                # the exact edge bound/hash; authenticate actual byte length at
                # the retained source/receipt boundary before native effects.
                or physical_input.expected_size_bytes not in (None, artifact.size_bytes)
                or physical_input.expected_outcome_access is not prior.outcome_access
                or physical_input.expected_visibility_ceiling is not prior.visibility_ceiling
                or edge.scientific_role is not role
                or edge.external_input_id is None
                or edge.producer_task_id is not None
                or edge.operational_logical_artifact_id != artifact.artifact_id
                or edge.payload_schema != artifact.payload_schema
                or edge.media_type != artifact.media_type
                or edge.maximum_size_bytes != artifact.size_bytes
                or edge.outcome_access is not prior.outcome_access
                or edge.visibility_ceiling is not prior.visibility_ceiling
            ):
                reasons.append(reason)
        if task.capability.requested_resources.source_scan_bytes < sum(
            a.size_bytes for a, _ in expected
        ):
            reasons.append(f"QUALIFICATION_RETAINED_SCAN_UNDERBOUND:{segment.segment_id}")
    return tuple(sorted(set(reasons)))


def validate_retained_qualification_receipt(
    prior: SourceQualificationRetainedPredecessor, receipt: CanonicalTaskReceipt
) -> None:
    """Bind authenticated predecessor bytes to their successful native receipt.

    This pure join grants neither access nor exposure authority. Call it after
    ordinary input authentication and before the adapter's native continuation.
    """
    raw = receipt.canonical_bytes()
    if (
        type(receipt) is not CanonicalTaskReceipt
        or receipt.task_id != prior.segment_id
        or receipt.operational_status is not OperationalStatus.SUCCEEDED
        or sha256(raw).hexdigest() != prior.task_receipt.sha256
        or len(raw) != prior.task_receipt.size_bytes
    ):
        raise ValueError("retained predecessor receipt identity, producer or terminal differs")
    for artifact in prior.artifacts:
        physical = tuple(
            m
            for m in receipt.output_materializations
            if m.logical_artifact_id == artifact.artifact_id
        )
        logical = tuple(
            a
            for a in receipt.output_logical_artifacts
            if a.logical_artifact_id == artifact.artifact_id
        )
        if len(physical) != 1 or len(logical) != 1:
            raise ValueError("retained artifact is not a sole receipted native output")
        if (
            physical[0].physical_sha256 != artifact.sha256
            or physical[0].size_bytes != artifact.size_bytes
            or logical[0].payload_schema != artifact.payload_schema
            or logical[0].media_type != artifact.media_type
        ):
            raise ValueError("retained artifact content/schema differs from its native receipt")
