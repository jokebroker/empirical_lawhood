"""Projection-owned imports of the exactly declared retained native observations."""

from dataclasses import dataclass
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
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
from empirical_lawhood.adapters.simulators.matrix_preparation.continuation import PreparationDevelopmentContinuation, PreparationNativeCustodyInput
from .contracts import PreparationProjectionConfig


RETAINED_NATIVE_PORT = "matrix-preparation-retained-native-observations"


@dataclass(frozen=True, slots=True)
class PreparationProjectionContinuation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-projection-continuation'
    config_id: str
    projection_config: ObjectIdentity
    continuation: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.projection_config.object_schema != PreparationProjectionConfig.SCHEMA
            or self.continuation.object_schema != PreparationDevelopmentContinuation.SCHEMA
        ):
            raise ValueError("retained projection inputs require their exact configuration schemas")


class PreparationRetainedNativePort(Protocol):
    @property
    def continuation(self) -> PreparationDevelopmentContinuation | None: ...

    def create_source(
        self, declaration: PreparationNativeCustodyInput
    ) -> ExternalInputSource: ...


@dataclass(frozen=True, slots=True)
class NoRetainedNativeInputs:
    continuation: None = None

    def create_source(self, declaration: PreparationNativeCustodyInput) -> ExternalInputSource:
        raise ValueError("the complete native panel has no retained observation inputs")


def preparation_retained_native_payloads(
    plan: ProtocolExecutionPlan,
    sources: PreparationRetainedNativePort,
) -> tuple[ExternalInputPayload, ...]:
    continuation = sources.continuation
    if continuation is None:
        return ()
    if plan.source_plan.object_id != continuation.run_id:
        raise ValueError("retained native imports require their declared continuation run")
    values = []
    for retained in continuation.native_inputs:
        artifact = retained.artifact
        consumers = tuple(
            task
            for task in plan.tasks
            if any(
                p.logical_artifact_id == retained.imported_artifact_id for p in task.external_inputs
            )
        )
        expected_consumers = {f"{retained.root.root_id}.project.r{view}" for view in (1, 2)}
        if {task.task_id for task in consumers} != expected_consumers:
            raise ValueError("retained native slot changed its exact projection consumers")
        for task in consumers:
            spec = next(
                p
                for p in task.external_inputs
                if p.logical_artifact_id == retained.imported_artifact_id
            )
            if (
                spec.expected_payload_schema != artifact.payload_schema
                or spec.expected_content_sha256 != artifact.sha256
                or spec.expected_media_type != artifact.media_type
                or spec.expected_visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
                or spec.expected_outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            ):
                raise ValueError("retained native slot changed its content or exposure")
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
                        retained.task_receipt,
                        *retained.source_export.lineage_identities,
                    )
                ),
                key=lineage_parent_sort_key,
            )
        )
        values.append(
            ExternalInputPayload(
                retained.imported_artifact_id,
                artifact.payload_schema,
                ArtifactProfile.AUDITED_HDF5
                if retained.output_id == "native-observations"
                else ArtifactProfile.CANONICAL_JSON,
                artifact.media_type,
                sources.create_source(retained),
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
    return tuple(sorted(values, key=lambda row: row.logical_artifact_id))
