"fresh calibration native provider that executes dependent refinement-frozen policy decisions."

from collections.abc import Callable
from decimal import Decimal
from typing import cast

from empirical_lawhood.adapters.methods.prepared_response.policy_decision import PreparedParentDecision
from empirical_lawhood.adapters.methods.prepared_response.stage_envelopes import prepared_response_calibration_native_stage_envelope, prepared_parent_decision_envelope
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import decode_port, output_result, read_port, semantic_contracts
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskProgressEmitter,
    TaskRunner,
    WorkerInputKind,
    WorkerInputPort,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .contracts import PreparedNativeSpec
from .native_pair import MAXIMUM_PAIR_BYTES, NATIVE_PAIR_SCHEMA, encode_prepared_native_pair
from .policy_native import PreparedResponseCalibrationNativeInvocation, PreparedResponseCalibrationNativeTaskResult, decode_prepared_response_calibration_task_native, prepared_response_calibration_native_invocations
from .source import PreparedNativePhaseData, bind_prepared_common_start, bind_prepared_native_handoff, execute_native_future, execute_native_parent, prepare_native_prefix
from .source_outputs import unentered_prepared_native_bytes


CANONICAL_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
NativePair = tuple[PreparedNativePhaseData, PreparedNativePhaseData]


def _ports(
    context: TaskContext, receipt_id: str, materializations: tuple[str, ...]
) -> dict[str, WorkerInputPort]:
    values = tuple(
        port for port in context.input_ports if port.materialization_id in materializations
    )
    if not values or len({value.payload_schema for value in values}) != len(values):
        raise ValueError(f"prepared fresh calibration dependency {receipt_id} has no materialized outputs")
    return {value.payload_schema: value for value in values}


class PreparedResponseCalibrationSourceTask:
    def __init__(self, manifest: CapabilityManifest, spec: PreparedNativeSpec) -> None:
        if manifest.config_schema != spec.SCHEMA or spec.stage != 'calibration':
            raise ValueError("prepared fresh calibration source task changes its installed configuration")
        self.manifest, self.spec = manifest, spec
        self.invocations = {value.task_id: value for value in prepared_response_calibration_native_invocations(spec)}

    def execute(self, context: TaskContext) -> RunnerResult:
        return self._execute(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        return self._execute(context, lambda value: emitter.advance(Decimal(value)))

    def _external_config(self, context: TaskContext, invocation: PreparedResponseCalibrationNativeInvocation) -> None:
        external = tuple(
            value for value in context.input_ports if value.kind is WorkerInputKind.EXTERNAL
        )
        expected = {f"config-artifact.{self.spec.spec_id}"}
        if invocation.phase == "prefix":
            expected.add(self.spec.spec_id)
        if (
            {value.artifact_id for value in external} != expected
            or len(external) != len(expected)
            or any(
                value.payload_schema != self.spec.SCHEMA
                or decode_port(value, PreparedNativeSpec) != self.spec
                for value in external
            )
            or context.config.config_id != self.spec.spec_id
            or context.config.config_schema != self.spec.SCHEMA
            or context.config.config_schema_sha256 != self.manifest.config_schema_sha256
            or context.config.content_sha256 != self.spec.fingerprint()
        ):
            raise ValueError("prepared fresh calibration source changes its exact external configuration")

    def _read_source_result(
        self, context: TaskContext, task_id: str
    ) -> tuple[PreparedResponseCalibrationNativeTaskResult, NativePair | None]:
        receipt = next(value for value in context.dependency_receipts if value.task_id == task_id)
        ports = _ports(context, receipt.receipt_id, receipt.output_materialization_ids)
        if set(ports) != {
            PreparedResponseCalibrationNativeTaskResult.SCHEMA,
            NATIVE_PAIR_SCHEMA,
            LinkedCampaignStageEnvelope.SCHEMA,
        }:
            raise ValueError("prepared fresh calibration source predecessor changes its output schemas")
        result = decode_port(
            ports[PreparedResponseCalibrationNativeTaskResult.SCHEMA], PreparedResponseCalibrationNativeTaskResult
        )
        stage = decode_port(
            ports[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
        )
        if result.result_id != f"{task_id}.result" or stage != prepared_response_calibration_native_stage_envelope(
            result
        ):
            raise ValueError("prepared fresh calibration native predecessor is detached from its receipt")
        payload = read_port(ports[NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES)
        return result, decode_prepared_response_calibration_task_native(result, payload)

    def _read_decision(
        self, context: TaskContext, task_id: str
    ) -> PreparedParentDecision:
        receipt = next(value for value in context.dependency_receipts if value.task_id == task_id)
        ports = _ports(context, receipt.receipt_id, receipt.output_materialization_ids)
        if set(ports) != {
            PreparedParentDecision.SCHEMA,
            LinkedCampaignStageEnvelope.SCHEMA,
        }:
            raise ValueError("prepared fresh calibration parent decision changes its output schemas")
        result = decode_port(ports[PreparedParentDecision.SCHEMA], PreparedParentDecision)
        stage = decode_port(
            ports[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
        )
        if result.decision_id != f"{task_id}.result" or stage != prepared_parent_decision_envelope(
            result
        ):
            raise ValueError("prepared fresh calibration parent decision is detached from its receipt")
        return result

    def _execute(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        try:
            invocation = self.invocations.get(context.task_id)
            if invocation is None:
                raise ValueError("prepared fresh calibration source task is outside its fixed task census")
            self._external_config(context, invocation)
            if tuple(sorted(value.task_id for value in context.dependency_receipts)) != (
                invocation.dependency_task_ids
            ):
                raise ValueError("prepared fresh calibration source changes its dependency task census")
            common = None
            realized_parent = None
            unentered = None
            predecessor_records: dict[str, CanonicalRecord] = {}
            parent_pair = None
            if invocation.phase == "parent":
                prefix, _ = self._read_source_result(
                    context, invocation.prefix_task_id
                )
                decision = self._read_decision(context, invocation.decision_task_id)
                predecessor_records[prefix.invocation.task_id] = prefix
                predecessor_records[invocation.decision_task_id] = decision
                common = prefix.common_start
                if (
                    decision.root != invocation.root
                    or decision.policy_id != invocation.policy_id
                    or decision.common_start
                    != (
                        None
                        if common is None
                        else ObjectIdentity.from_record(common.common_start_id, common)
                    )
                ):
                    raise ValueError("prepared fresh calibration parent decision changes root or common start")
                if common is None:
                    unentered = "PREFIX_UNAVAILABLE"
                elif common.frame is None:
                    unentered = "PORT_FRAME_UNRESOLVED"
                elif decision.disposition != "DECIDED":
                    unentered = "POLICY_NONATTEMPT"
                else:
                    realized_parent = decision.selected_parent
            elif invocation.phase == "future":
                parent_id = invocation.dependency_task_ids[0]
                parent, parent_pair = self._read_source_result(context, parent_id)
                predecessor_records[parent_id] = parent
                common = parent.common_start
                realized_parent = parent.realized_parent
                if (
                    parent_pair is None
                    or realized_parent is None
                    or any(value.checkpoint is None for value in parent_pair)
                ):
                    unentered = "HANDOFF_UNAVAILABLE"
            native = None
            if unentered is None:
                values: list[PreparedNativePhaseData] = []
                offset = 0

                def advance(count: int) -> None:
                    total = offset + count
                    if progress is not None:
                        progress(total)

                for refinement in (1, 2):
                    if invocation.phase == "prefix":
                        value = prepare_native_prefix(
                            self.spec,
                            invocation.root,
                            refinement=refinement,
                            progress=advance,
                        )
                    elif invocation.phase == "parent":
                        assert common is not None and realized_parent is not None
                        value = execute_native_parent(
                            self.spec,
                            common,
                            parent=realized_parent,
                            refinement=refinement,
                            progress=advance,
                        )
                    else:
                        assert (
                            common is not None
                            and realized_parent is not None
                            and parent_pair is not None
                            and invocation.word is not None
                        )
                        native_parent = parent_pair[refinement - 1]
                        if native_parent.checkpoint is None:
                            unentered = "HANDOFF_UNAVAILABLE"
                            values = []
                            break
                        value = execute_native_future(
                            self.spec,
                            common,
                            bind_prepared_native_handoff(native_parent),
                            parent=realized_parent,
                            word=invocation.word,
                            purpose=invocation.purpose,
                            progress=advance,
                        )
                    values.append(value)
                    offset += value.delivery.completed_intervals
                if values:
                    native, payload = encode_prepared_native_pair(
                        values[0], values[1], purpose=invocation.purpose
                    )
                    if (
                        invocation.phase == "prefix"
                        and values[0].checkpoint is not None
                        and values[1].checkpoint is not None
                    ):
                        common = bind_prepared_common_start(
                            values[0].checkpoint, values[1].checkpoint
                        )
            if native is None:
                payload = unentered_prepared_native_bytes()
            predecessors = tuple(
                ObjectIdentity.from_record(
                    getattr(predecessor_records[task_id], "result_id", None)
                    or getattr(predecessor_records[task_id], "decision_id"),
                    predecessor_records[task_id],
                )
                for task_id in invocation.dependency_task_ids
            )
            result = PreparedResponseCalibrationNativeTaskResult(
                invocation,
                predecessors,
                realized_parent,
                native,
                common,
                unentered,
            )
            stage = prepared_response_calibration_native_stage_envelope(result)
            return output_result(
                context,
                {
                    result.SCHEMA: result.canonical_bytes(),
                    NATIVE_PAIR_SCHEMA: payload,
                    stage.SCHEMA: stage.canonical_bytes(),
                },
                (
                    "policy-keyed-realized-parent-lineage",
                    "paired-view-single-acquisition",
                    "no-target-visible-parent-selection",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class PreparedResponseCalibrationSourceProvider(CampaignRuntimeProvider):
    def __init__(
        self, registry: CapabilityRegistry, manifest: CapabilityManifest, spec: PreparedNativeSpec
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != spec.SCHEMA
            or spec.stage != 'calibration'
        ):
            raise ValueError("prepared fresh calibration source provider changes registry or source stage")
        self.registry, self.manifest, self.spec = registry, manifest, spec
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        self.invocations = {
            value.task_id: value for value in prepared_response_calibration_native_invocations(spec)
        }

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("prepared fresh calibration source runner registry/records differ")
        return (cast(TaskRunner, PreparedResponseCalibrationSourceTask(self.manifest, self.spec)),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("prepared fresh calibration source execution registry/records differ")
        tasks = tuple(
            value
            for value in plan.tasks
            if value.capability.capability_key == self.manifest.capability_key
        )
        if {value.task_id for value in tasks} != set(self.invocations):
            raise ValueError("prepared fresh calibration plan changes its exact native task census")
        payload = self.spec.canonical_bytes()
        digest = self.spec.fingerprint()
        specs = {}
        for task in tasks:
            invocation = self.invocations[task.task_id]
            expected = {f"config-artifact.{self.spec.spec_id}"}
            if invocation.phase == "prefix":
                expected.add(self.spec.spec_id)
            if (
                task.dependency_task_ids != invocation.dependency_task_ids
                or task.maximum_attempts != 1
                or {value.logical_artifact_id for value in task.external_inputs} != expected
            ):
                raise ValueError("prepared fresh calibration task changes source inputs or dependencies")
            specs.update({value.logical_artifact_id: value for value in task.external_inputs})
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
            raise ValueError("prepared fresh calibration source semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (PreparedResponseCalibrationNativeTaskResult, LinkedCampaignStageEnvelope),
            NATIVE_PAIR_SCHEMA,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("prepared fresh calibration source adjudication registry differs")
        return None
