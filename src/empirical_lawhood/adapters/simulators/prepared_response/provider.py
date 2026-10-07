"Native source qualification/dependent refinement task provider over authenticated, path-free runtime ports."

from collections.abc import Callable
from decimal import Decimal
from typing import cast

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile
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
)
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import decode_port, envelope, output_result, read_port, semantic_contracts

from .contracts import PreparedNativeSpec
from .native_pair import MAXIMUM_PAIR_BYTES, NATIVE_PAIR_SCHEMA, encode_prepared_native_pair
from .native_tasks import PreparedNativeInvocation, prepared_static_native_invocations
from .source import PreparedNativePhaseData, bind_prepared_common_start, bind_prepared_native_handoff, execute_native_future, execute_native_parent, prepare_native_prefix
from .source_outputs import PreparedNativeTaskResult, decode_prepared_task_native, unentered_prepared_native_bytes


CANONICAL_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
NativePair = tuple[PreparedNativePhaseData, PreparedNativePhaseData]
Predecessors = dict[str, tuple[PreparedNativeTaskResult, NativePair | None]]


def prepared_native_stage_envelope(
    result: PreparedNativeTaskResult,
) -> LinkedCampaignStageEnvelope:
    reasons = (
        (result.unentered_reason,)
        if result.unentered_reason is not None
        else tuple(
            sorted(
                {
                    view.delivery.reason
                    for view in result.native_pair.views
                    if view.delivery.reason is not None
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


class PreparedStaticSourceTask:
    """One declared paired phase per shared-runtime task, with no native retry."""

    def __init__(self, manifest: CapabilityManifest, spec: PreparedNativeSpec) -> None:
        if manifest.config_schema != spec.SCHEMA:
            raise ValueError("prepared source task changes its installed configuration schema")
        self.manifest, self.spec = manifest, spec
        self.invocations = {v.task_id: v for v in prepared_static_native_invocations(spec)}

    def execute(self, context: TaskContext) -> RunnerResult:
        return self._execute(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        return self._execute(context, lambda value: emitter.advance(Decimal(value)))

    def _read_inputs(self, context: TaskContext, task: PreparedNativeInvocation) -> Predecessors:
        external = tuple(p for p in context.input_ports if p.kind is WorkerInputKind.EXTERNAL)
        expected = {f"config-artifact.{self.spec.spec_id}"}
        if task.phase == "prefix":
            expected.add(self.spec.spec_id)
        if (
            {p.artifact_id for p in external} != expected
            or len(external) != len(expected)
            or any(p.payload_schema != self.spec.SCHEMA for p in external)
            or context.config.config_id != self.spec.spec_id
            or context.config.config_schema != self.spec.SCHEMA
            or context.config.config_schema_sha256 != self.manifest.config_schema_sha256
            or context.config.content_sha256 != self.spec.fingerprint()
            or context.config.artifact_id != f"config-artifact.{self.spec.spec_id}"
            or tuple(sorted(r.task_id for r in context.dependency_receipts))
            != task.dependency_task_ids
        ):
            raise ValueError("prepared native task changes its exact config/dependency roster")
        for port in external:
            if decode_port(port, PreparedNativeSpec) != self.spec:
                raise ValueError("prepared native task received another source configuration")
        predecessors: Predecessors = {}
        for receipt in context.dependency_receipts:
            ports = tuple(
                p
                for p in context.input_ports
                if p.materialization_id in receipt.output_materialization_ids
            )
            schemas = {p.payload_schema: p for p in ports}
            if len(ports) != 3 or set(schemas) != {
                PreparedNativeTaskResult.SCHEMA,
                LinkedCampaignStageEnvelope.SCHEMA,
                NATIVE_PAIR_SCHEMA,
            }:
                raise ValueError("prepared native dependency changes its complete output census")
            result = decode_port(
                schemas[PreparedNativeTaskResult.SCHEMA], PreparedNativeTaskResult
            )
            stage = decode_port(
                schemas[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
            )
            if result.invocation != self.invocations[
                receipt.task_id
            ] or stage != prepared_native_stage_envelope(result):
                raise ValueError("prepared native predecessor differs from its declared receipt")
            payload = read_port(schemas[NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES)
            predecessors[receipt.task_id] = result, decode_prepared_task_native(result, payload)
        return predecessors

    def _execute(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        try:
            task = self.invocations.get(context.task_id)
            if task is None:
                raise ValueError("prepared native task is outside its frozen source qualification/dependent refinement census")
            predecessors = self._read_inputs(context, task)
            identities = tuple(
                ObjectIdentity.from_record(predecessors[key][0].result_id, predecessors[key][0])
                for key in task.dependency_task_ids
            )
            common = None
            handoffs = None
            unentered = None
            if task.phase != "prefix":
                predecessor = predecessors[task.dependency_task_ids[0]][0]
                common = predecessor.common_start
                if common is None:
                    unentered = "PREFIX_UNAVAILABLE"
                elif common.frame is None:
                    unentered = "PORT_FRAME_UNRESOLVED"
            if task.phase == "future" and unentered is None:
                parent = predecessors[f"{task.root.root_id}.{task.parent}.parent.native"][1]
                if parent is None or any(v.checkpoint is None for v in parent):
                    unentered = "HANDOFF_UNAVAILABLE"
                else:
                    handoffs = tuple(bind_prepared_native_handoff(v) for v in parent)
            native = None
            if unentered is None:
                values = []
                offset, emitted = 0, 0

                def advance(count: int) -> None:
                    nonlocal emitted
                    total = offset + count
                    if progress is not None and total > emitted:
                        progress(total)
                        emitted = total

                for refinement in (1, 2):
                    if task.phase == "prefix":
                        value = prepare_native_prefix(
                            self.spec, task.root, refinement=refinement, progress=advance
                        )
                    elif task.phase == "parent":
                        assert common is not None and task.parent is not None
                        value = execute_native_parent(
                            self.spec,
                            common,
                            parent=task.parent,
                            refinement=refinement,
                            progress=advance,
                        )
                    else:
                        assert (
                            common is not None
                            and handoffs is not None
                            and task.parent is not None
                            and task.word is not None
                        )
                        value = execute_native_future(
                            self.spec,
                            common,
                            handoffs[refinement - 1],
                            parent=task.parent,
                            word=task.word,
                            purpose=task.purpose,
                            progress=advance,
                        )
                    values.append(value)
                    offset += value.delivery.completed_intervals
                native, payload = encode_prepared_native_pair(
                    values[0], values[1], purpose=task.purpose
                )
                if (
                    task.phase == "prefix"
                    and values[0].checkpoint is not None
                    and values[1].checkpoint is not None
                ):
                    common = bind_prepared_common_start(
                        values[0].checkpoint, values[1].checkpoint
                    )
            else:
                payload = unentered_prepared_native_bytes()
            result = PreparedNativeTaskResult(task, identities, native, common, unentered)
            return self._output_result(context, result, payload)
        finally:
            for port in context.input_ports:
                port.close()

    def _output_result(
        self, context: TaskContext, result: PreparedNativeTaskResult, payload: bytes
    ) -> RunnerResult:
        """Native products; a follow-up binding may attach a causal method product."""
        stage = prepared_native_stage_envelope(result)
        return output_result(
            context,
            {
                result.SCHEMA: result.canonical_bytes(),
                NATIVE_PAIR_SCHEMA: payload,
                stage.SCHEMA: stage.canonical_bytes(),
            },
            (
                "exact-native-phase-census",
                "paired-view-delivery-accounting",
                "authenticated-predecessor-checkpoints",
            ),
        )


class PreparedStaticSourceProvider(CampaignRuntimeProvider):
    def __init__(
        self, registry: CapabilityRegistry, manifest: CapabilityManifest, spec: PreparedNativeSpec
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != spec.SCHEMA
        ):
            raise ValueError("prepared native provider changes its exact installed registry/config")
        self.registry, self.manifest, self.spec = registry, manifest, spec
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        self.invocations = {v.task_id: v for v in prepared_static_native_invocations(spec)}

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("prepared source runner registry/records differ")
        return (cast(TaskRunner, PreparedStaticSourceTask(self.manifest, self.spec)),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("prepared source execution registry/records differ")
        tasks = tuple(
            t for t in plan.tasks if t.capability.capability_key == self.manifest.capability_key
        )
        if {t.task_id for t in tasks} != set(self.invocations):
            raise ValueError("prepared source execution changes its exact native phase census")
        payload = self.spec.canonical_bytes()
        digest = self.spec.fingerprint()
        all_tasks = {t.task_id: t for t in plan.tasks}
        specs = {}
        for task in tasks:
            invocation = self.invocations[task.task_id]
            expected = {f"config-artifact.{self.spec.spec_id}"}
            if invocation.phase == "prefix":
                expected.add(self.spec.spec_id)
            values = task.external_inputs
            if (
                task.dependency_task_ids != invocation.dependency_task_ids
                or task.maximum_attempts != 1
                or len(values) != len(expected)
                or {v.logical_artifact_id for v in values} != expected
                or any(
                    v.expected_payload_schema != self.spec.SCHEMA
                    or v.expected_content_sha256 != digest
                    for v in values
                )
            ):
                raise ValueError(
                    "prepared native plan changes its exact config/dependency bindings"
                )
            scan = len(payload) * len(values) + sum(
                all_tasks[d].capability.requested_resources.output_bytes
                for d in invocation.dependency_task_ids
            )
            if scan > task.capability.requested_resources.source_scan_bytes:
                raise ValueError("prepared native scan budget omits required predecessor outputs")
            specs.update({v.logical_artifact_id: v for v in values})
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.spec.spec_id, self.spec),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        return tuple(
            ExternalInputPayload.from_bytes(
                logical_artifact_id=key,
                payload_schema=self.spec.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type=CANONICAL_MEDIA_TYPE,
                payload=payload,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                lineage_parents=(parent,),
                logical_content_sha256=digest,
            )
            for key in sorted(specs)
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("prepared source semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (PreparedNativeTaskResult, LinkedCampaignStageEnvelope),
            NATIVE_PAIR_SCHEMA,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("prepared source adjudication registry differs")
        return None
