"Registered information response prediction native continuation with immutable handoff predictions."

from typing import cast

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskRunner, WorkerInputKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.adapters.methods.information_response.prediction import predict_parent
from empirical_lawhood.adapters.methods.information_response.records import InformationResponseCommittedPrediction
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import decode_port, output_result, read_port, semantic_contracts
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import MAXIMUM_PAIR_BYTES, NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.native_tasks import PreparedNativeInvocation
from empirical_lawhood.adapters.simulators.prepared_response.provider import PreparedStaticSourceTask, Predecessors, prepared_native_stage_envelope
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult, decode_prepared_task_native
from .contracts import InformationResponseNativeConfig, native_invocations


def validate_prediction(
    config: InformationResponseNativeConfig,
    result: PreparedNativeTaskResult,
    prediction: InformationResponseCommittedPrediction,
) -> None:
    identity = ObjectIdentity.from_record
    if (
        prediction.native_config != identity(config.spec_id, config)
        or prediction.native_result != identity(result.result_id, result)
        or prediction.invocation != result.invocation
        or prediction.model_bank != identity(config.model_bank.bank_id, config.model_bank)
    ):
        raise ValueError("information response prediction forecast is detached from its exact source/config/bank receipt")


class InformationResponseSourceTask(PreparedStaticSourceTask):
    def __init__(self, manifest: CapabilityManifest, config: InformationResponseNativeConfig) -> None:
        if manifest.config_schema != config.SCHEMA:
            raise ValueError("information response prediction source changes its installed config schema")
        self.manifest, self.config, self.spec = manifest, config, config.recipe
        self.invocations = {t.task_id: t for t in native_invocations(config)}

    def _read_inputs(self, context: TaskContext, task: PreparedNativeInvocation) -> Predecessors:
        config = self.config
        external = tuple(p for p in context.input_ports if p.kind is WorkerInputKind.EXTERNAL)
        expected = {f"config-artifact.{config.spec_id}"}
        if task.phase == "prefix":
            expected.add(config.spec_id)
        if (
            {p.artifact_id for p in external} != expected
            or len(external) != len(expected)
            or any(p.payload_schema != config.SCHEMA for p in external)
            or context.config.config_id != config.spec_id
            or context.config.config_schema != config.SCHEMA
            or context.config.config_schema_sha256 != self.manifest.config_schema_sha256
            or context.config.content_sha256 != config.fingerprint()
            or context.config.artifact_id != f"config-artifact.{config.spec_id}"
            or tuple(sorted(r.task_id for r in context.dependency_receipts))
            != task.dependency_task_ids
        ):
            raise ValueError("information response prediction native task changes its exact config or dependencies")
        for port in external:
            if decode_port(port, InformationResponseNativeConfig) != config:
                raise ValueError("information response prediction source received another frozen configuration")
        predecessors: Predecessors = {}
        for receipt in context.dependency_receipts:
            ports = tuple(
                p
                for p in context.input_ports
                if p.materialization_id in receipt.output_materialization_ids
            )
            schemas = {p.payload_schema: p for p in ports}
            if len(ports) != 4 or set(schemas) != {
                PreparedNativeTaskResult.SCHEMA,
                NATIVE_PAIR_SCHEMA,
                LinkedCampaignStageEnvelope.SCHEMA,
                InformationResponseCommittedPrediction.SCHEMA,
            }:
                raise ValueError("information response prediction native dependency omits its committed prediction product")
            result = decode_port(
                schemas[PreparedNativeTaskResult.SCHEMA], PreparedNativeTaskResult
            )
            stage = decode_port(
                schemas[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
            )
            prediction = decode_port(
                schemas[InformationResponseCommittedPrediction.SCHEMA], InformationResponseCommittedPrediction
            )
            if result.invocation != self.invocations[
                receipt.task_id
            ] or stage != prepared_native_stage_envelope(result):
                raise ValueError("information response prediction native predecessor changes its exact invocation/envelope")
            validate_prediction(config, result, prediction)
            payload = read_port(schemas[NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES)
            predecessors[receipt.task_id] = result, decode_prepared_task_native(result, payload)
        return predecessors

    def _output_result(
        self, context: TaskContext, result: PreparedNativeTaskResult, payload: bytes
    ) -> RunnerResult:
        stage = prepared_native_stage_envelope(result)
        prediction = predict_parent(self.config, result, payload)
        return output_result(
            context,
            {
                result.SCHEMA: result.canonical_bytes(),
                NATIVE_PAIR_SCHEMA: payload,
                stage.SCHEMA: stage.canonical_bytes(),
                prediction.SCHEMA: prediction.canonical_bytes(),
            },
            (
                "exact-information-response-prediction-native-phase-census",
                "paired-view-delivery-accounting",
                "forecast-committed-before-dependent-futures",
            ),
        )


class InformationResponseSourceProvider(CampaignRuntimeProvider):
    def __init__(
        self, registry: CapabilityRegistry, manifest: CapabilityManifest, config: InformationResponseNativeConfig
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("information response prediction source changes its exact registry/configuration")
        self.registry, self.manifest, self.config = registry, manifest, config
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        self.invocations = {t.task_id: t for t in native_invocations(config)}

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("information response prediction native runner registry/records differ")
        return (cast(TaskRunner, InformationResponseSourceTask(self.manifest, self.config)),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("information response prediction native execution registry/records differ")
        tasks = tuple(
            t for t in plan.tasks if t.capability.capability_key == self.manifest.capability_key
        )
        if {t.task_id for t in tasks} != set(self.invocations):
            raise ValueError("information response prediction source execution changes its finite census")
        config = self.config
        payload = config.canonical_bytes()
        all_tasks = {t.task_id: t for t in plan.tasks}
        specs = {}
        for task in tasks:
            invocation = self.invocations[task.task_id]
            expected = {f"config-artifact.{config.spec_id}"}
            if invocation.phase == "prefix":
                expected.add(config.spec_id)
            values = task.external_inputs
            if (
                task.dependency_task_ids != invocation.dependency_task_ids
                or task.maximum_attempts != 1
                or len(values) != len(expected)
                or {v.logical_artifact_id for v in values} != expected
                or any(
                    v.expected_payload_schema != config.SCHEMA
                    or v.expected_content_sha256 != config.fingerprint()
                    for v in values
                )
            ):
                raise ValueError("information response prediction native plan changes config/dependency bindings")
            scan = len(payload) * len(values) + sum(
                all_tasks[d].capability.requested_resources.output_bytes
                for d in invocation.dependency_task_ids
            )
            if scan > task.capability.requested_resources.source_scan_bytes:
                raise ValueError("information response prediction native scan budget omits predecessor outputs")
            specs.update({v.logical_artifact_id: v for v in values})
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(config.spec_id, config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        return tuple(
            ExternalInputPayload.from_bytes(
                logical_artifact_id=key,
                payload_schema=config.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                payload=payload,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                lineage_parents=(parent,),
                logical_content_sha256=config.fingerprint(),
            )
            for key in sorted(specs)
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("information response prediction native semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (PreparedNativeTaskResult, InformationResponseCommittedPrediction, LinkedCampaignStageEnvelope),
            NATIVE_PAIR_SCHEMA,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("information response prediction native adjudication registry differs")
        return None
