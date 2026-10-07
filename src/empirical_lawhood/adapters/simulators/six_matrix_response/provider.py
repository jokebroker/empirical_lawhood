"""Installed one-request/one-result Six-matrix response runtime provider."""

from __future__ import annotations

from dataclasses import fields

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
)
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .contracts import SixMatrixResponseSixMatrixSourceConfig
from .simulation import MATRIX_RESPONSE_CANONICAL_MEDIA_TYPE, MATRIX_RESPONSE_MAXIMUM_REQUEST_BYTES, SixMatrixResponseEpisodeRequest, SixMatrixResponseEpisodeResult, SixMatrixResponseSixMatrixEpisodeEngine


class SixMatrixResponseSixMatrixEpisodeTask:
    """Consume one bounded canonical request and emit one native episode result."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        engine: SixMatrixResponseSixMatrixEpisodeEngine,
    ) -> None:
        if (
            manifest.kind is not CapabilityKind.SIMULATOR
            or manifest.input_schema_ids != (SixMatrixResponseEpisodeRequest.SCHEMA,)
            or manifest.output_schema_ids != (SixMatrixResponseEpisodeResult.SCHEMA,)
        ):
            raise ValueError("Six-matrix response task manifest has another native contract")
        self.manifest = manifest
        self._engine = engine

    def execute(self, context: TaskContext) -> RunnerResult:
        if len(context.input_ports) != 1 or len(context.output_ports) != 1:
            raise ValueError("Six-matrix response episode task requires one input and one output")
        input_port = context.input_ports[0]
        output_port = context.output_ports[0]
        if (
            input_port.payload_schema != SixMatrixResponseEpisodeRequest.SCHEMA
            or input_port.media_type != MATRIX_RESPONSE_CANONICAL_MEDIA_TYPE
            or input_port.size_bytes > MATRIX_RESPONSE_MAXIMUM_REQUEST_BYTES
        ):
            raise ValueError("Six-matrix response episode request port differs")
        try:
            payload = input_port.read(input_port.size_bytes + 1)
            if len(payload) != input_port.size_bytes or input_port.read(1):
                raise ValueError("Six-matrix response episode request size differs")
        finally:
            input_port.close()
        request = decode_canonical_bytes(
            payload,
            SixMatrixResponseEpisodeRequest,
            maximum_bytes=MATRIX_RESPONSE_MAXIMUM_REQUEST_BYTES,
        )
        if request.task_id != context.task_id:
            raise ValueError("Six-matrix response request task identity differs from worker context")
        if (
            output_port.output_id != request.output_id
            or output_port.payload_schema != SixMatrixResponseEpisodeResult.SCHEMA
            or output_port.profile is not ArtifactProfile.CANONICAL_JSON
            or output_port.media_type != MATRIX_RESPONSE_CANONICAL_MEDIA_TYPE
        ):
            raise ValueError("Six-matrix response episode output port differs")
        result = self._engine.run(request)
        return RunnerResult(
            outputs=(TaskOutputPayload(request.output_id, result.canonical_bytes()),),
            checks=(
                ReceiptCheck("exact-checkpoint-or-typed-terminal", True, ()),
                ReceiptCheck("four-stage-action-captured", True, ()),
                ReceiptCheck("no-scientific-verdict-constructed", True, ()),
            ),
        )


class SixMatrixResponseSixMatrixCampaignRuntimeProvider(CampaignRuntimeProvider):
    """Source-free installed provider; construction performs no simulation or I/O."""

    issued_source_schema_ids = (SixMatrixResponseEpisodeRequest.SCHEMA,)

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        source_config: SixMatrixResponseSixMatrixSourceConfig,
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("Six-matrix response provider manifest is not the installed capability")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.manifest = manifest
        self.source_config = source_config
        self._runner = SixMatrixResponseSixMatrixEpisodeTask(
            manifest=manifest,
            engine=SixMatrixResponseSixMatrixEpisodeEngine(source_config),
        )

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or any(
            not isinstance(value, SixMatrixResponseEpisodeRequest) for value in source_records
        ):
            raise ValueError("Six-matrix response provider registry/source record types differ")
        return (self._runner,)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or any(
            not isinstance(value, SixMatrixResponseEpisodeRequest) for value in source_records
        ):
            raise ValueError("Six-matrix response provider plan/source record types differ")
        requests = {
            value.task_id: value
            for value in source_records
            if isinstance(value, SixMatrixResponseEpisodeRequest)
        }
        if len(requests) != len(source_records):
            raise ValueError("Six-matrix response requests require unique task identities")
        tasks = {
            task.task_id: task
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
            and task.capability.capability_version == self.manifest.capability_version
        }
        if set(tasks) != set(requests):
            raise ValueError("Six-matrix response source request roster differs from execution tasks")
        payloads = []
        for task_id in sorted(tasks):
            task = tasks[task_id]
            request = requests[task_id]
            if (
                task.capability_implementation_sha256 != self.manifest.implementation_sha256
                or len(task.external_inputs) != 1
                or task.external_inputs[0].logical_artifact_id != request.request_id
                or len(task.outputs) != 1
                or task.outputs[0].output_id != request.output_id
                or task.outputs[0].payload_schema != SixMatrixResponseEpisodeResult.SCHEMA
            ):
                raise ValueError("Six-matrix response execution task differs from its native request")
            identity = ObjectIdentity.from_record(request.request_id, request)
            parent = ArtifactLineageParent(
                identity=identity,
                visibility_ceiling=request.visibility_ceiling,
                outcome_access=request.outcome_access,
            )
            payloads.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=request.request_id,
                    payload_schema=SixMatrixResponseEpisodeRequest.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type=MATRIX_RESPONSE_CANONICAL_MEDIA_TYPE,
                    payload=request.canonical_bytes(),
                    visibility_ceiling=request.visibility_ceiling,
                    outcome_access=request.outcome_access,
                    parent_visibility_ceilings=(request.visibility_ceiling,),
                    lineage_parents=(parent,),
                    logical_content_sha256=task.external_inputs[0].expected_content_sha256,
                )
            )
        return tuple(payloads)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("Six-matrix response provider semantic registry differs")
        del execution_plan
        return (
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=SixMatrixResponseEpisodeResult.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(value.name for value in fields(SixMatrixResponseEpisodeResult))),
                task_id_field="task_id",
                output_id_field="output_id",
            ),
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("Six-matrix response provider adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    'SixMatrixResponseSixMatrixCampaignRuntimeProvider',
    'SixMatrixResponseSixMatrixEpisodeTask',
]
