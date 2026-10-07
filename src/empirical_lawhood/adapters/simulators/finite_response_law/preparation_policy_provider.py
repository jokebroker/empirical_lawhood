"""Installed preparation-policy source provider over authenticated retained prefixes."""

from collections.abc import Callable
from decimal import Decimal
from hashlib import sha256
from typing import Protocol, cast

from empirical_lawhood.adapters.simulators.prepared_response.native_pair import NATIVE_PAIR_SCHEMA as RETAINED_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult, decode_prepared_task_native
from empirical_lawhood.adapters.simulators.prepared_response.source import PreparedNativeCheckpoint, PreparedNativePhaseData
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _encode
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import decode_port, envelope, output_result, read_port, semantic_contracts
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
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
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
    ExternalInputSource,
    MAX_EXTERNAL_INPUT_CHUNK_BYTES,
)
from empirical_lawhood.runtime.source_qualification import retained_qualification_input_reasons

from .native_artifact import MAXIMUM_PAIR_BYTES, encode_native_pair
from .preparation_policy_contracts import FiniteResponseLawPreparationPolicyNativeConfig, FiniteResponseLawPreparationPolicyRetainedPrefix, preparation_policy_native_invocations
from .preparation_policy_native_artifact import FiniteResponseLawPreparationPolicyNativePairArtifact, PREPARATION_POLICY_NATIVE_PAIR_SCHEMA, unentered_preparation_policy_native_bytes
from .preparation_policy_source import FiniteResponseLawPreparationPolicyNativeCheckpoint, execute_preparation_policy_native_phase
from .preparation_policy_source_outputs import FiniteResponseLawPreparationPolicyNativeTaskResult, decode_preparation_policy_task_native

CANONICAL_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
PREPARATION_POLICY_RETAINED_SOURCE_PORT = "finite-response-law-preparation-policy-retained-prefix-sources"


class FiniteResponseLawPreparationPolicyRetainedSources(Protocol):
    @property
    def artifacts(self) -> tuple[ArtifactIdentity, ...]: ...

    def create_source(self, artifact: ArtifactIdentity) -> ExternalInputSource: ...


def preparation_policy_native_stage(result: FiniteResponseLawPreparationPolicyNativeTaskResult) -> LinkedCampaignStageEnvelope:
    reasons = (
        (result.unentered_reason,)
        if result.unentered_reason is not None
        else tuple(
            sorted(
                {
                    delivery.reason
                    for delivery in result.native_pair.deliveries
                    if delivery.reason is not None
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


def _retained_prefix(
    prior: FiniteResponseLawPreparationPolicyRetainedPrefix,
    ports: dict[str, object],
) -> tuple[
    PreparedNativeTaskResult,
    tuple[PreparedNativePhaseData, PreparedNativePhaseData],
]:
    declaration = prior.declaration
    artifacts = (*declaration.artifacts, declaration.task_receipt)
    raw: dict[str, bytes] = {}
    for artifact in artifacts:
        port = ports[artifact.artifact_id]
        if (
            port.outcome_access is not declaration.outcome_access  # type: ignore[attr-defined]
            or port.visibility_ceiling is not declaration.visibility_ceiling  # type: ignore[attr-defined]
        ):
            raise ValueError("preparation-policy retained prefix changes its declared exposure")
        value = read_port(port, artifact.size_bytes)  # type: ignore[arg-type]
        if len(value) != artifact.size_bytes or sha256(value).hexdigest() != artifact.sha256:
            raise ValueError("preparation-policy retained prefix differs from authenticated bytes")
        raw[artifact.payload_schema] = value
    receipt = decode_canonical_bytes(
        raw[CanonicalTaskReceipt.SCHEMA],
        CanonicalTaskReceipt,
        maximum_bytes=declaration.task_receipt.size_bytes,
    )
    record_artifact = next(
        artifact
        for artifact in declaration.artifacts
        if artifact.payload_schema == PreparedNativeTaskResult.SCHEMA
    )
    record = decode_canonical_bytes(
        raw[PreparedNativeTaskResult.SCHEMA],
        PreparedNativeTaskResult,
        maximum_bytes=record_artifact.size_bytes,
    )
    if (
        receipt.task_id != declaration.segment_id
        or receipt.operational_status.value != "SUCCEEDED"
        or ObjectIdentity.from_record(record.result_id, record) != prior.native_result
        or record.invocation.task_id != declaration.segment_id
        or record.invocation.source_spec != prior.source_spec
        or not record.native_complete
        or record.common_start is None
        or record.common_start.frame is None
    ):
        raise ValueError("preparation-policy retained prefix receipt/source/result join differs")
    pair = decode_prepared_task_native(record, raw[RETAINED_PAIR_SCHEMA])
    if pair is None or any(value.checkpoint is None for value in pair):
        raise ValueError("preparation-policy retained prefix lacks its complete paired checkpoints")
    return record, pair


class FiniteResponseLawPreparationPolicySourceTask:
    def __init__(self, manifest: CapabilityManifest, config: FiniteResponseLawPreparationPolicyNativeConfig) -> None:
        if (
            manifest.config_schema != config.SCHEMA
            or config.native_owner != ObjectIdentity.from_record(manifest.capability_key, manifest)
        ):
            raise ValueError("preparation-policy source task changes installed config identity")
        self.manifest, self.config = manifest, config
        self.invocations = {row.task_id: row for row in preparation_policy_native_invocations(config)}
        self.retained = {
            row.declaration.segment_id: row for row in config.retained_prefixes
        }

    def execute(self, context: TaskContext) -> RunnerResult:
        return self._execute(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        return self._execute(context, lambda count: emitter.advance(Decimal(count)))

    def _execute(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        raise ValueError("FINITE_RESPONSE_LAW_CURRENT_PREPARATION_ELIGIBILITY_EXPORT_REQUIRED")
        try:
            task = self.invocations.get(context.task_id)
            config = self.config
            if task is None:
                raise ValueError("preparation-policy task is outside the frozen source census")
            config_ids = {f"config-artifact.{config.spec_id}"}
            if not task.dependency_task_ids:
                config_ids.add(config.spec_id)
            prior = self.retained.get(task.predecessor_segment_id or "")
            retained_artifacts = (
                ()
                if prior is None
                else (*prior.declaration.artifacts, prior.declaration.task_receipt)
            )
            expected_external = {
                **{key: config.SCHEMA for key in config_ids},
                **{artifact.artifact_id: artifact.payload_schema for artifact in retained_artifacts},
            }
            external = tuple(
                port for port in context.input_ports if port.kind is WorkerInputKind.EXTERNAL
            )
            if (
                {port.artifact_id: port.payload_schema for port in external}
                != expected_external
                or context.config.config_id != config.spec_id
                or context.config.config_schema != config.SCHEMA
                or context.config.config_schema_sha256 != self.manifest.config_schema_sha256
                or context.config.content_sha256 != config.fingerprint()
                or context.config.artifact_id != f"config-artifact.{config.spec_id}"
                or tuple(sorted(row.task_id for row in context.dependency_receipts))
                != task.dependency_task_ids
            ):
                raise ValueError("preparation-policy task changes config/input/dependency binding")
            for port in external:
                if port.artifact_id in config_ids and decode_port(port, type(config)) != config:
                    raise ValueError("preparation-policy source received another configuration")
            dependency_ports = tuple(
                port for port in context.input_ports if port.kind is not WorkerInputKind.EXTERNAL
            )
            incoming: tuple[
                PreparedNativeCheckpoint | FiniteResponseLawPreparationPolicyNativeCheckpoint | None,
                PreparedNativeCheckpoint | FiniteResponseLawPreparationPolicyNativeCheckpoint | None,
            ]
            if prior is not None:
                if dependency_ports:
                    raise ValueError("preparation-policy retained preparation imported a replacement prefix")
                record, pair = _retained_prefix(prior, {port.artifact_id: port for port in external})
                predecessor = ObjectIdentity.from_record(record.result_id, record)
                assert pair[0].checkpoint is not None and pair[1].checkpoint is not None
                incoming = (pair[0].checkpoint, pair[1].checkpoint)
                assert record.common_start is not None and record.common_start.frame is not None
                common = ObjectIdentity.from_record(
                    record.common_start.common_start_id, record.common_start
                )
                frame = record.common_start.frame
                unentered = None
            elif task.dependency_task_ids:
                binding = context.dependency_receipts[0]
                if any(
                    port.materialization_id not in binding.output_materialization_ids
                    for port in dependency_ports
                ):
                    raise ValueError("preparation-policy dependency is detached from its receipt")
                schemas = {port.payload_schema: port for port in dependency_ports}
                if set(schemas) != {
                    FiniteResponseLawPreparationPolicyNativeTaskResult.SCHEMA,
                    LinkedCampaignStageEnvelope.SCHEMA,
                    PREPARATION_POLICY_NATIVE_PAIR_SCHEMA,
                }:
                    raise ValueError("preparation-policy dependency changes its complete output roster")
                previous = decode_port(
                    schemas[FiniteResponseLawPreparationPolicyNativeTaskResult.SCHEMA],
                    FiniteResponseLawPreparationPolicyNativeTaskResult,
                )
                stage = decode_port(
                    schemas[LinkedCampaignStageEnvelope.SCHEMA],
                    LinkedCampaignStageEnvelope,
                )
                if (
                    previous.invocation != self.invocations[task.dependency_task_ids[0]]
                    or stage != preparation_policy_native_stage(previous)
                ):
                    raise ValueError("preparation-policy dependency changes invocation/stage identity")
                decoded = decode_preparation_policy_task_native(
                    previous, read_port(schemas[PREPARATION_POLICY_NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES)
                )
                predecessor = ObjectIdentity.from_record(previous.result_id, previous)
                common, frame = previous.retained_common_start, previous.frame
                if decoded is None or not previous.native_complete:
                    incoming = (None, None)
                    unentered = "HANDOFF_UNAVAILABLE"
                else:
                    incoming = cast(
                        tuple[
                            FiniteResponseLawPreparationPolicyNativeCheckpoint, FiniteResponseLawPreparationPolicyNativeCheckpoint
                        ],
                        tuple(value.checkpoint for value in decoded),
                    )
                    unentered = None
            else:
                raise ValueError("preparation-policy preparation lacks its retained prefix")
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
                    value = execute_preparation_policy_native_phase(
                        config,
                        task,
                        refinement,
                        incoming=incoming[refinement - 1],  # type: ignore[arg-type]
                        frame=frame,
                        progress=advance,
                    )
                    values.append(value)
                    offset += value.delivery.completed_intervals
                native, payload = encode_native_pair(
                    values[0], values[1], record_type=FiniteResponseLawPreparationPolicyNativePairArtifact
                )
            else:
                payload = unentered_preparation_policy_native_bytes()
            result = FiniteResponseLawPreparationPolicyNativeTaskResult(
                task,
                predecessor,
                cast(FiniteResponseLawPreparationPolicyNativePairArtifact | None, native),
                common,
                _encode(frame.modes),
                unentered,
            )
            stage = preparation_policy_native_stage(result)
            return output_result(
                context,
                {
                    result.SCHEMA: result.canonical_bytes(),
                    PREPARATION_POLICY_NATIVE_PAIR_SCHEMA: payload,
                    stage.SCHEMA: stage.canonical_bytes(),
                },
                (
                    "exact-preparation-policy-native-phase-census",
                    "authenticated-retained-prefix",
                    "paired-view-delivery-accounting",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class FiniteResponseLawPreparationPolicySourceProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: FiniteResponseLawPreparationPolicyNativeConfig,
        carrier: PredecessorBoundSourceQualificationExperiment,
        sources: FiniteResponseLawPreparationPolicyRetainedSources,
    ) -> None:
        declarations = tuple(row.declaration for row in config.retained_prefixes)
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or config.native_owner != ObjectIdentity.from_record(manifest.capability_key, manifest)
            or carrier.retained_predecessors != declarations
            or carrier.source_config != ObjectIdentity.from_record(config.spec_id, config)
        ):
            raise ValueError("preparation-policy provider changes source/carrier identity")
        expected = tuple(
            sorted(
                (
                    artifact
                    for row in declarations
                    for artifact in (*row.artifacts, row.task_receipt)
                ),
                key=lambda artifact: artifact.artifact_id,
            )
        )
        if sources.artifacts != expected:
            raise ValueError("preparation-policy provider lacks exact retained-prefix ports")
        self.registry, self.manifest, self.config = registry, manifest, config
        self.carrier, self.sources = carrier, sources
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.invocations = {row.task_id: row for row in preparation_policy_native_invocations(config)}

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("preparation-policy runner registry/records differ")
        return (cast(TaskRunner, FiniteResponseLawPreparationPolicySourceTask(self.manifest, self.config)),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        raise ValueError("FINITE_RESPONSE_LAW_CURRENT_PREPARATION_ELIGIBILITY_EXPORT_REQUIRED")
        if (
            plan.registry_sha256 != self.registry_sha256
            or source_records
            or retained_qualification_input_reasons(self.carrier, plan)
        ):
            raise ValueError("preparation-policy execution changes retained graph/registry binding")
        tasks = tuple(
            task
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
        )
        if {task.task_id for task in tasks} != set(self.invocations):
            raise ValueError("preparation-policy execution changes its full finite task census")
        config_bytes = self.config.canonical_bytes()
        config_inputs: set[str] = set()
        retained = {
            row.declaration.segment_id: row.declaration
            for row in self.config.retained_prefixes
        }
        all_tasks = {task.task_id: task for task in plan.tasks}
        for task in tasks:
            invocation = self.invocations[task.task_id]
            keys = {f"config-artifact.{self.config.spec_id}"}
            if not invocation.dependency_task_ids:
                keys.add(self.config.spec_id)
            prior = retained.get(invocation.predecessor_segment_id or "")
            artifacts = () if prior is None else (*prior.artifacts, prior.task_receipt)
            expected = keys | {artifact.artifact_id for artifact in artifacts}
            specs = task.external_inputs
            scan = (
                len(config_bytes) * len(keys)
                + sum(artifact.size_bytes for artifact in artifacts)
                + sum(
                    all_tasks[dependency].capability.requested_resources.output_bytes
                    for dependency in invocation.dependency_task_ids
                )
            )
            if (
                task.dependency_task_ids != invocation.dependency_task_ids
                or task.maximum_attempts != 1
                or {value.logical_artifact_id for value in specs} != expected
                or scan > task.capability.requested_resources.source_scan_bytes
            ):
                raise ValueError("preparation-policy execution changes input/resource census")
            config_inputs.update(keys)
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.config.spec_id, self.config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        values = [
            ExternalInputPayload.from_bytes(
                logical_artifact_id=key,
                payload_schema=self.config.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type=CANONICAL_MEDIA_TYPE,
                payload=config_bytes,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                lineage_parents=(parent,),
                logical_content_sha256=self.config.fingerprint(),
            )
            for key in sorted(config_inputs)
        ]
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
        return tuple(sorted(values, key=lambda value: value.logical_artifact_id))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("preparation-policy semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (FiniteResponseLawPreparationPolicyNativeTaskResult, LinkedCampaignStageEnvelope),
            PREPARATION_POLICY_NATIVE_PAIR_SCHEMA,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("preparation-policy adjudication registry differs")
        return None
