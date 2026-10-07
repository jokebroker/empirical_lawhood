"Finite response-law native tasks over authenticated runtime ports; no private execution route."

from collections.abc import Callable
from decimal import Decimal
from hashlib import sha256
from typing import Protocol, cast

from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA as RETAINED_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedNativeCheckpoint
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import decode_port, envelope, output_result, read_port, semantic_contracts
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _encode
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    CanonicalTaskReceipt,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskProgressEmitter,
    TaskRunner,
    WorkerInputKind,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    MAX_EXTERNAL_INPUT_CHUNK_BYTES,
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
    ExternalInputSource,
)
from empirical_lawhood.runtime.source_qualification import retained_qualification_input_reasons

from .assigned_contracts import FiniteResponseLawAssignedCalibrationConfig, FiniteResponseLawAssignedEvaluationConfig
from .contracts import FiniteResponseLawNativeConfig, FiniteResponseLawNativeInvocation, native_invocations
from .evaluation_contracts import FiniteResponseLawEvaluationConfig
from .fresh_contracts import FiniteResponseLawCalibrationConfig
from .native_artifact import MAXIMUM_PAIR_BYTES, NATIVE_PAIR_SCHEMA, encode_native_pair
from .retained import authenticate_retained_handoff
from .source import FiniteResponseLawNativeCheckpoint, execute_native_phase, frozen_prefix_frame
from .source_outputs import FiniteResponseLawAssignedCalibrationTaskResult, FiniteResponseLawAssignedEvaluationTaskResult, FiniteResponseLawCalibrationTaskResult, FiniteResponseLawEvaluationTaskResult, FiniteResponseLawNativeTaskResult, decode_task_native, unentered_native_bytes


def native_result_type(config: FiniteResponseLawNativeConfig) -> type[FiniteResponseLawNativeTaskResult]:
    return (
        FiniteResponseLawAssignedEvaluationTaskResult
        if type(config) is FiniteResponseLawAssignedEvaluationConfig
        else FiniteResponseLawAssignedCalibrationTaskResult
        if type(config) is FiniteResponseLawAssignedCalibrationConfig
        else FiniteResponseLawEvaluationTaskResult
        if type(config) is FiniteResponseLawEvaluationConfig
        else FiniteResponseLawCalibrationTaskResult
        if type(config) is FiniteResponseLawCalibrationConfig
        else FiniteResponseLawNativeTaskResult
    )


RETAINED_SOURCE_PORT = "finite-response-law-retained-handoff-sources"
CANONICAL_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"


class FiniteResponseLawRetainedSources(Protocol):
    @property
    def artifacts(self) -> tuple[ArtifactIdentity, ...]: ...

    def create_source(self, artifact: ArtifactIdentity) -> ExternalInputSource: ...


def native_stage(result: FiniteResponseLawNativeTaskResult) -> LinkedCampaignStageEnvelope:
    reasons = (
        (result.unentered_reason,)
        if result.unentered_reason is not None
        else tuple(
            sorted(
                {
                    d.reason
                    for d in result.native_pair.deliveries
                    if d.reason is not None
                }
            )
        )
        if result.native_pair is not None
        else ()
    )
    return envelope(
        result.invocation.task_id,
        result,
        result.result_id,
        LinkedCampaignStageRole.SOURCE_MATERIALIZATION,
        result.native_complete,
        reasons,
    )


class FiniteResponseLawSourceTask:
    def __init__(self, manifest: CapabilityManifest, config: FiniteResponseLawNativeConfig) -> None:
        if (
            manifest.config_schema != config.SCHEMA
            or config.native_owner
            != ObjectIdentity.from_record(manifest.capability_key, manifest)
        ):
            raise ValueError(
                "Finite response-law native task changes its installed source/config identity"
            )
        self.manifest, self.config = manifest, config
        self.result_type = native_result_type(config)
        self.invocations = {t.task_id: t for t in native_invocations(config)}
        self.retained = {p.segment_id: p for p in config.retained_predecessors}

    def execute(self, context: TaskContext) -> RunnerResult:
        return self._execute(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        return self._execute(context, lambda count: emitter.advance(Decimal(count)))

    def validate_predecessor(
        self, task: FiniteResponseLawNativeInvocation, previous: FiniteResponseLawNativeTaskResult
    ) -> None:
        "Finite response-law evaluation's registered source adds its immutable control guard at this seam."

    def _execute(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        try:
            task = self.invocations.get(context.task_id)
            config = self.config
            if task is None:
                raise ValueError("Finite response-law native task is outside its frozen census")
            config_ids = {f"config-artifact.{config.spec_id}"}
            if not task.dependency_task_ids:
                config_ids.add(config.spec_id)
            prior = self.retained.get(task.predecessor_segment_id or "")
            retained_artifacts = (
                () if prior is None else (*prior.artifacts, prior.task_receipt)
            )
            expected_external = {
                **{key: config.SCHEMA for key in config_ids},
                **{a.artifact_id: a.payload_schema for a in retained_artifacts},
            }
            external = tuple(
                p for p in context.input_ports if p.kind is WorkerInputKind.EXTERNAL
            )
            if (
                len(external) != len(expected_external)
                or {p.artifact_id: p.payload_schema for p in external}
                != expected_external
                or context.config.config_id != config.spec_id
                or context.config.config_schema != config.SCHEMA
                or context.config.config_schema_sha256
                != self.manifest.config_schema_sha256
                or context.config.content_sha256 != config.fingerprint()
                or context.config.artifact_id != f"config-artifact.{config.spec_id}"
                or tuple(sorted(r.task_id for r in context.dependency_receipts))
                != task.dependency_task_ids
            ):
                raise ValueError(
                    "Finite response-law source changes exact config, retained input or dependency bindings"
                )
            for port in external:
                if (
                    port.artifact_id in config_ids
                    and decode_port(port, type(config)) != config
                ):
                    raise ValueError("Finite response-law source received another native configuration")
            predecessors: tuple[ObjectIdentity, ...] = ()
            incoming: tuple[
                FiniteResponseLawNativeCheckpoint | PreparedNativeCheckpoint | None, ...
            ] = (
                None,
                None,
            )
            common = None
            frame = None
            unentered = None
            dependency_ports = tuple(
                p for p in context.input_ports if p.kind is not WorkerInputKind.EXTERNAL
            )
            if prior is not None:
                if dependency_ports or config.retained_source is None:
                    raise ValueError(
                        "Finite response-law retained continuation cannot acquire a replacement predecessor"
                    )
                by_id = {p.artifact_id: p for p in external}
                payloads = {}
                for artifact in retained_artifacts:
                    port = by_id[artifact.artifact_id]
                    if (
                        port.outcome_access is not prior.outcome_access
                        or port.visibility_ceiling is not prior.visibility_ceiling
                    ):
                        raise ValueError("Finite response-law retained input changes declared exposure")
                    raw = read_port(port, artifact.size_bytes)
                    if (
                        len(raw) != artifact.size_bytes
                        or sha256(raw).hexdigest() != artifact.sha256
                    ):
                        raise ValueError(
                            "Finite response-law retained input differs from its authenticated bytes"
                        )
                    payloads[artifact.payload_schema] = raw
                from empirical_lawhood.kernel.decoding import decode_canonical_bytes

                receipt = decode_canonical_bytes(
                    payloads[CanonicalTaskReceipt.SCHEMA],
                    CanonicalTaskReceipt,
                    maximum_bytes=prior.task_receipt.size_bytes,
                )
                record_artifact = next(
                    a
                    for a in prior.artifacts
                    if a.payload_schema == PreparedNativeTaskResult.SCHEMA
                )
                record = decode_canonical_bytes(
                    payloads[PreparedNativeTaskResult.SCHEMA],
                    PreparedNativeTaskResult,
                    maximum_bytes=record_artifact.size_bytes,
                )
                pair = authenticate_retained_handoff(
                    declaration=prior,
                    source=config.retained_source,
                    receipt=receipt,
                    record=record,
                    payload=payloads[RETAINED_PAIR_SCHEMA],
                )
                predecessors = (ObjectIdentity.from_record(record.result_id, record),)
                incoming = tuple(v.checkpoint for v in pair)
                assert record.common_start is not None
                common = ObjectIdentity.from_record(
                    record.common_start.common_start_id, record.common_start
                )
                frame = record.common_start.frame
            elif task.dependency_task_ids:
                receipt_binding = context.dependency_receipts[0]
                if any(
                    p.materialization_id
                    not in receipt_binding.output_materialization_ids
                    for p in dependency_ports
                ):
                    raise ValueError(
                        "Finite response-law dependency port is detached from its native receipt"
                    )
                schemas = {p.payload_schema: p for p in dependency_ports}
                if len(dependency_ports) != 3 or set(schemas) != {
                    self.result_type.SCHEMA,
                    LinkedCampaignStageEnvelope.SCHEMA,
                    NATIVE_PAIR_SCHEMA,
                }:
                    raise ValueError(
                        "Finite response-law native dependency changes its full output roster"
                    )
                previous = decode_port(
                    schemas[self.result_type.SCHEMA], self.result_type
                )
                stage = decode_port(
                    schemas[LinkedCampaignStageEnvelope.SCHEMA],
                    LinkedCampaignStageEnvelope,
                )
                if previous.invocation != self.invocations[
                    task.dependency_task_ids[0]
                ] or stage != native_stage(previous):
                    raise ValueError(
                        "Finite response-law native dependency changes its invocation or stage receipt"
                    )
                self.validate_predecessor(task, previous)
                pair_new = decode_task_native(
                    previous, read_port(schemas[NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES)
                )
                identity = ObjectIdentity.from_record(previous.result_id, previous)
                predecessors = (identity,)
                frame = previous.frame
                common = (
                    (identity if previous.native_complete else None)
                    if task.phase == "parent"
                    else previous.common_start
                )
                if common is None:
                    unentered = "PREFIX_UNAVAILABLE"
                elif frame is None:
                    unentered = "PORT_FRAME_UNRESOLVED"
                elif pair_new is None or not previous.native_complete:
                    unentered = "HANDOFF_UNAVAILABLE"
                else:
                    incoming = tuple(v.checkpoint for v in pair_new)
            elif dependency_ports:
                raise ValueError(
                    "Finite response-law fresh prefix cannot import dependent scientific data"
                )
            native = None
            if unentered is None:
                values = []
                offset = emitted = 0

                def advance(count: int) -> None:
                    nonlocal emitted
                    total = offset + count
                    if progress is not None and total > emitted:
                        progress(total)
                        emitted = total

                for refinement in (1, 2):
                    value = execute_native_phase(
                        config,
                        task,
                        refinement,
                        incoming=incoming[refinement - 1],
                        frame=frame,
                        progress=advance,
                    )
                    values.append(value)
                    offset += value.delivery.completed_intervals
                native, payload = encode_native_pair(values[0], values[1])
                if task.phase == "prefix" and all(
                    v.checkpoint is not None for v in values
                ):
                    assert values[0].checkpoint is not None
                    frame = frozen_prefix_frame(values[0].checkpoint)
            else:
                payload = unentered_native_bytes()
            result = self.result_type(
                task,
                predecessors,
                native,
                common,
                None if frame is None else _encode(frame.modes),
                unentered,
            )
            stage = native_stage(result)
            return output_result(
                context,
                {
                    result.SCHEMA: result.canonical_bytes(),
                    NATIVE_PAIR_SCHEMA: payload,
                    stage.SCHEMA: stage.canonical_bytes(),
                },
                (
                    "exact-flh-native-phase-census",
                    "authenticated-native-predecessor",
                    "paired-view-delivery-accounting",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class FiniteResponseLawSourceProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: FiniteResponseLawNativeConfig,
        carrier: PredecessorBoundSourceQualificationExperiment,
        sources: FiniteResponseLawRetainedSources | None,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version)
            != manifest
            or config.native_owner
            != ObjectIdentity.from_record(manifest.capability_key, manifest)
            or carrier.retained_predecessors != config.retained_predecessors
            or carrier.source_config
            != ObjectIdentity.from_record(config.spec_id, config)
        ):
            raise ValueError(
                "Finite response-law provider changes its installed source/carrier identity"
            )
        expected = tuple(
            sorted(
                (
                    a
                    for p in config.retained_predecessors
                    for a in (*p.artifacts, p.task_receipt)
                ),
                key=lambda a: a.artifact_id,
            )
        )
        if (
            (sources is None) != (not expected)
            or sources is not None
            and sources.artifacts != expected
        ):
            raise ValueError(
                "Finite response-law provider lacks its exact retained-artifact source port"
            )
        self.registry, self.manifest, self.config, self.carrier, self.sources = (
            registry,
            manifest,
            config,
            carrier,
            sources,
        )
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        self.invocations = {t.task_id: t for t in native_invocations(config)}

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("Finite response-law native runner registry/records differ")
        return (cast(TaskRunner, FiniteResponseLawSourceTask(self.manifest, self.config)),)

    def expected_dependencies(
        self, invocation: FiniteResponseLawNativeInvocation
    ) -> tuple[str, ...]:
        "Exact physical predecessors, plus registered finite response-law evaluation causal guards when applicable."
        return invocation.dependency_task_ids

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if (
            plan.registry_sha256 != self.registry_sha256
            or source_records
            or retained_qualification_input_reasons(self.carrier, plan)
        ):
            raise ValueError(
                "Finite response-law native execution changes exact retained graph/registry bindings"
            )
        tasks = tuple(
            t
            for t in plan.tasks
            if t.capability.capability_key == self.manifest.capability_key
        )
        if {t.task_id for t in tasks} != set(self.invocations):
            raise ValueError("Finite response-law native execution changes its full finite task census")
        config = self.config
        config_bytes = config.canonical_bytes()
        config_inputs = set()
        retained = {p.segment_id: p for p in config.retained_predecessors}
        all_tasks = {t.task_id: t for t in plan.tasks}
        for task in tasks:
            invocation = self.invocations[task.task_id]
            keys = {f"config-artifact.{config.spec_id}"}
            if not invocation.dependency_task_ids:
                keys.add(config.spec_id)
            prior = retained.get(invocation.predecessor_segment_id or "")
            artifacts = () if prior is None else (*prior.artifacts, prior.task_receipt)
            expected = keys | {a.artifact_id for a in artifacts}
            specs = task.external_inputs
            if (
                task.dependency_task_ids != self.expected_dependencies(invocation)
                or task.maximum_attempts != 1
                or len(specs) != len(expected)
                or {v.logical_artifact_id for v in specs} != expected
                or any(
                    v.expected_payload_schema != config.SCHEMA
                    or v.expected_content_sha256 != config.fingerprint()
                    for v in specs
                    if v.logical_artifact_id in keys
                )
            ):
                raise ValueError(
                    "Finite response-law native execution changes config or predecessor census"
                )
            scan = (
                len(config_bytes) * len(keys)
                + sum(a.size_bytes for a in artifacts)
                + sum(
                    all_tasks[d].capability.requested_resources.output_bytes
                    for d in self.expected_dependencies(invocation)
                )
            )
            if scan > task.capability.requested_resources.source_scan_bytes:
                raise ValueError("Finite response-law source scan budget omits exact input bytes")
            config_inputs.update(keys)
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(config.spec_id, config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        values = [
            ExternalInputPayload.from_bytes(
                logical_artifact_id=key,
                payload_schema=config.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type=CANONICAL_MEDIA_TYPE,
                payload=config_bytes,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                lineage_parents=(parent,),
                logical_content_sha256=config.fingerprint(),
            )
            for key in sorted(config_inputs)
        ]
        if self.sources is not None:
            for artifact in self.sources.artifacts:
                lineage = ArtifactLineageParent(
                    ObjectIdentity.from_record(artifact.artifact_id, artifact),
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                )
                values.append(
                    ExternalInputPayload(
                        artifact.artifact_id,
                        artifact.payload_schema,
                        ArtifactProfile.AUDITED_HDF5
                        if artifact.payload_schema == RETAINED_PAIR_SCHEMA
                        else ArtifactProfile.CANONICAL_JSON,
                        artifact.media_type,
                        self.sources.create_source(artifact),
                        artifact.size_bytes,
                        artifact.sha256,
                        artifact.size_bytes,
                        min(artifact.size_bytes, MAX_EXTERNAL_INPUT_CHUNK_BYTES),
                        VisibilityCeiling.OUTCOME_VISIBLE,
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                        (lineage.visibility_ceiling,),
                        (lineage,),
                        artifact.sha256,
                    )
                )
        return tuple(sorted(values, key=lambda v: v.logical_artifact_id))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("Finite response-law native semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (native_result_type(self.config), LinkedCampaignStageEnvelope),
            NATIVE_PAIR_SCHEMA,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("Finite response-law source adjudication registry differs")
