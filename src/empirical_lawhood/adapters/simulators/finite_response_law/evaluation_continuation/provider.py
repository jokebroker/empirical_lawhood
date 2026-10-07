"""Restore donor bytes through ordinary custody; reuse the guarded native runner."""

from hashlib import sha256
from typing import cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import ProspectiveRetainedSourceUse
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import TaskContext, RunnerResult, TaskRunner, TaskProgressEmitter
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    ExternalInputPayload,
    CapabilityOutputSemanticContract,
    MAX_EXTERNAL_INPUT_CHUNK_BYTES,
)
from empirical_lawhood.runtime.source_resolution import (
    ContentAddressedInputRequirement,
    ContentAddressedInputKind,
    ContentAddressedInputResolver,
)
from empirical_lawhood.adapters.methods.finite_response_law.method_provider import _ResolvedInputSource
from empirical_lawhood.adapters.methods.finite_response_law.control_ports import FiniteResponseLawControlRuntimePort
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, read_port, output_result, semantic_contracts
from ..evaluation.discovery import SOURCE_CAPABILITY as ORIGINAL_SOURCE
from ..assigned_contracts import FiniteResponseLawAssignedEvaluationConfig
from ..contracts import native_invocations
from ..evaluation_provider import FiniteResponseLawEvaluationSourceProvider, FiniteResponseLawEvaluationSourceTask
from ..evaluation_retention import FiniteResponseLawEvaluationRetention
from ..native_artifact import NATIVE_PAIR_SCHEMA
from ..source_outputs import FiniteResponseLawAssignedEvaluationTaskResult
from ..provider import native_stage
from .discovery import SOURCE_CAPABILITY, IMPORT_CAPABILITY


class _ContinuationTask:
    manifest = SOURCE_CAPABILITY

    def __init__(self, source: FiniteResponseLawAssignedEvaluationConfig, runtime: FiniteResponseLawControlRuntimePort):
        self.source = source
        self.delegate = FiniteResponseLawEvaluationSourceTask(ORIGINAL_SOURCE, source, runtime.prepared_store)
        self.native_ids = {t.task_id for t in native_invocations(source) if t.phase != "prefix"}

    def execute(self, context: TaskContext) -> RunnerResult:
        if context.task_id not in self.native_ids:
            raise ValueError("Finite response-law evaluation continuation cannot reacquire a prefix")
        return self.delegate.execute(context)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        if context.task_id not in self.native_ids:
            raise ValueError("Finite response-law evaluation continuation cannot reacquire a prefix")
        return self.delegate.execute_with_progress(context, emitter)


class FiniteResponseLawContinuationSourceProvider(FiniteResponseLawEvaluationSourceProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        source: FiniteResponseLawAssignedEvaluationConfig,
        carrier: ProspectiveRetainedSourceUse,
        runtime: FiniteResponseLawControlRuntimePort,
    ):
        if (
            registry.resolve(SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY.capability_version)
            != SOURCE_CAPABILITY
            or source.native_owner
            != ObjectIdentity.from_record(ORIGINAL_SOURCE.capability_key, ORIGINAL_SOURCE)
            or carrier.source_capability
            != ObjectIdentity.from_record(SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY)
            or carrier.source_config != ObjectIdentity.from_record(source.spec_id, source)
        ):
            raise ValueError("Finite response-law evaluation continuation changes its explicit original-source compatibility")
        self.registry, self.manifest, self.config, self.carrier, self.sources = (
            registry,
            SOURCE_CAPABILITY,
            source,
            carrier,
            None,
        )
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        self.invocations = {
            t.task_id: t for t in native_invocations(source) if t.phase != "prefix"
        }
        self.source, self.runtime = source, runtime

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("Finite response-law evaluation continuation runner binding differs")
        return (cast(TaskRunner, _ContinuationTask(self.source, self.runtime)),)


class _PrefixImportTask:
    manifest = IMPORT_CAPABILITY

    def __init__(self, config: FiniteResponseLawEvaluationRetention):
        self.config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            prior = next(p for p in self.config.prefixes if p.segment_id == context.task_id)
            config_id = f"config-artifact.{self.config.config_id}"
            artifacts = {a.artifact_id: a for a in (*prior.artifacts, prior.task_receipt)}
            ports = {p.artifact_id: p for p in context.input_ports}
            if (
                context.dependency_receipts
                or len(ports) != len(context.input_ports)
                or set(ports) != {*artifacts, config_id}
                or context.config.config_id != self.config.config_id
                or context.config.config_schema != self.config.SCHEMA
                or context.config.content_sha256 != self.config.fingerprint()
                or context.config.config_schema_sha256 != self.manifest.config_schema_sha256
                or decode_port(ports[config_id], FiniteResponseLawEvaluationRetention) != self.config
            ):
                raise ValueError("Finite response-law evaluation import changes its exact configuration or receipt census")
            payloads = {}
            for key, artifact in artifacts.items():
                port = ports[key]
                raw = read_port(port, artifact.size_bytes)
                if (
                    len(raw),
                    sha256(raw).hexdigest(),
                    port.payload_schema,
                    port.outcome_access,
                    port.visibility_ceiling,
                ) != (
                    artifact.size_bytes,
                    artifact.sha256,
                    artifact.payload_schema,
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                ):
                    raise ValueError("Finite response-law evaluation retained bytes or exposure differ")
                payloads[artifact.payload_schema] = raw
            receipt = decode_canonical_bytes(
                payloads[CanonicalTaskReceipt.SCHEMA],
                CanonicalTaskReceipt,
                maximum_bytes=prior.task_receipt.size_bytes,
            )
            record = decode_canonical_bytes(
                payloads[FiniteResponseLawAssignedEvaluationTaskResult.SCHEMA],
                FiniteResponseLawAssignedEvaluationTaskResult,
                maximum_bytes=16 * 1024**2,
            )
            self.config.authenticate(prior, receipt, record)
            if (
                native_stage(record).canonical_bytes()
                != payloads[LinkedCampaignStageEnvelope.SCHEMA]
            ):
                raise ValueError("Finite response-law evaluation retained native stage differs")
            return output_result(
                context,
                {k: v for k, v in payloads.items() if k != CanonicalTaskReceipt.SCHEMA},
                (
                    "byte-identical-receipted-prefix",
                    "zero-native-updates",
                    "prospective-retained-origin",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class FiniteResponseLawPrefixImportProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        config: FiniteResponseLawEvaluationRetention,
        resolver: ContentAddressedInputResolver,
    ):
        if (
            registry.resolve(IMPORT_CAPABILITY.capability_key, IMPORT_CAPABILITY.capability_version)
            != IMPORT_CAPABILITY
        ):
            raise ValueError("Finite response-law evaluation import registry differs")
        self.registry, self.config, self.resolver, self.manifest = (
            registry,
            config,
            resolver,
            IMPORT_CAPABILITY,
        )
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("Finite response-law evaluation import runner registry differs")
        return (cast(TaskRunner, _PrefixImportTask(self.config)),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("Finite response-law evaluation import plan registry differs")
        values = list(
            config_payloads(
                plan,
                self.manifest,
                self.config,
                self.config.config_id,
                {p.segment_id for p in self.config.prefixes},
            )
        )
        tasks = {t.task_id: t for t in plan.tasks}
        for prior in self.config.prefixes:
            task = tasks[prior.segment_id]
            expected = {
                a.artifact_id: (a.payload_schema, a.sha256)
                for a in (*prior.artifacts, prior.task_receipt)
            }
            expected[f"config-artifact.{self.config.config_id}"] = (
                self.config.SCHEMA,
                self.config.fingerprint(),
            )
            if (
                task.dependency_task_ids
                or task.maximum_attempts != 1
                or len(task.external_inputs) != len(expected)
                or {
                    p.logical_artifact_id: (p.expected_payload_schema, p.expected_content_sha256)
                    for p in task.external_inputs
                }
                != expected
            ):
                raise ValueError("Finite response-law evaluation prefix import adds or drops a retained input")
            for artifact in (*prior.artifacts, prior.task_receipt):
                requirement = ContentAddressedInputRequirement(
                    f"input.{artifact.artifact_id}",
                    ContentAddressedInputKind.SOURCE_MATERIALIZATION,
                    artifact.sha256,
                    artifact.payload_schema,
                    artifact.media_type,
                    artifact.size_bytes,
                    artifact.size_bytes,
                    OutcomeAccess.OUTCOME_BLIND,
                    VisibilityCeiling.PROSPECTIVE,
                )
                lineage = ArtifactLineageParent(
                    ObjectIdentity.from_record(self.config.config_id, self.config),
                    VisibilityCeiling.PROSPECTIVE,
                    OutcomeAccess.OUTCOME_BLIND,
                )
                values.append(
                    ExternalInputPayload(
                        artifact.artifact_id,
                        artifact.payload_schema,
                        ArtifactProfile.AUDITED_HDF5
                        if artifact.payload_schema == NATIVE_PAIR_SCHEMA
                        else ArtifactProfile.CANONICAL_JSON,
                        artifact.media_type,
                        _ResolvedInputSource(self.resolver, requirement),
                        artifact.size_bytes,
                        artifact.sha256,
                        artifact.size_bytes,
                        min(artifact.size_bytes, MAX_EXTERNAL_INPUT_CHUNK_BYTES),
                        VisibilityCeiling.PROSPECTIVE,
                        OutcomeAccess.OUTCOME_BLIND,
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
            raise ValueError("Finite response-law evaluation import semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (FiniteResponseLawAssignedEvaluationTaskResult, LinkedCampaignStageEnvelope),
            NATIVE_PAIR_SCHEMA,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("Finite response-law evaluation import adjudication registry differs")
        return None
