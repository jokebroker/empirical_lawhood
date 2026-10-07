"""Projection and sealed evaluation providers for the observation order observation assay."""

from __future__ import annotations

from dataclasses import fields
from functools import lru_cache
from typing import cast

from empirical_lawhood.adapters.simulators.six_matrix_response.observation_order import MatrixObservationOrderAcquisitionDisposition, MatrixObservationOrderPairedHistoryResult, OBSERVATION_ORDER_CANONICAL_MEDIA_TYPE, OBSERVATION_ORDER_HDF5_SCHEMA, OBSERVATION_ORDER_MAXIMUM_HDF5_BYTES
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
    WorkerInputPort,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .history_conditioned_geometry import HistoryViewTrace, ScientificView, TwoViewTerminal
from .observation_order import MatrixObservationOrderEvaluationConfig, MatrixObservationOrderEvaluation, MatrixObservationOrderProjectionConfig, MatrixObservationOrderViewReport, evaluate_observation_order, project_observation_order_history_view


OBSERVATION_ORDER_PROJECTION_TASK_PREFIX = "matrix-observation-order.project"
OBSERVATION_ORDER_EVALUATION_TASK_ID = "matrix-observation-order.evaluate"
OBSERVATION_ORDER_PROJECTION_CAPABILITY_KEY = "matrix-observation-order.geometry-projection"
OBSERVATION_ORDER_EVALUATION_CAPABILITY_KEY = "matrix-observation-order.sealed-evaluation"
OBSERVATION_ORDER_METHOD_CAPABILITY_VERSION = "1.0.0"
_MAXIMUM_CANONICAL_BYTES = 32 * 1024 * 1024


def _read(port: object, maximum: int) -> bytes:
    size = cast(int, getattr(port, "size_bytes"))
    if size < 1 or size > maximum:
        raise ValueError("observation order method input exceeds its byte bound")
    try:
        payload = cast(bytes, getattr(port, "read")(size + 1))
        if len(payload) != size or getattr(port, "read")(1):
            raise ValueError("observation order method input size differs")
        return payload
    finally:
        getattr(port, "close")()


def _decode(port: object, record_type: type[CanonicalRecord]) -> CanonicalRecord:
    payload = _read(port, _MAXIMUM_CANONICAL_BYTES)
    # Reuse only immutable, bounded preparation records. Every task still reads
    # and authenticates its own input capability before this pure decode cache.
    if len(payload) <= 1024 * 1024 and record_type in (
        MatrixObservationOrderProjectionConfig,
        MatrixObservationOrderPairedHistoryResult,
        LinkedCampaignStageEnvelope,
    ):
        return _decode_preparation(payload, record_type)
    return decode_canonical_bytes(
        payload,
        record_type,
        maximum_bytes=_MAXIMUM_CANONICAL_BYTES,
    )


@lru_cache(maxsize=4)
def _decode_preparation(payload: bytes, record_type: type[CanonicalRecord]) -> CanonicalRecord:
    return decode_canonical_bytes(payload, record_type, maximum_bytes=1024 * 1024)


def _local_output_id(task_id: str, output_id: str) -> str:
    prefix = f"{task_id}."
    return output_id.removeprefix(prefix) if output_id.startswith(prefix) else output_id


def _stage_envelope(
    *,
    task_id: str,
    role: LinkedCampaignStageRole,
    product_id: str,
    product: CanonicalRecord,
    disposition: LinkedCampaignDisposition,
    reasons: tuple[str, ...],
) -> LinkedCampaignStageEnvelope:
    return LinkedCampaignStageEnvelope(
        envelope_id=f"stage-envelope.{task_id}",
        role=role,
        disposition=disposition,
        scientific_product=ObjectIdentity.from_record(product_id, product),
        reason_codes=reasons,
    )


class MatrixObservationOrderProjectionTask:
    # Pure projection; paired views may share one bounded interpreter chunk.
    # Source acquisition and sealed evaluation deliberately do not opt in.
    worker_chunk_limit = 2

    def __init__(self, *, manifest: CapabilityManifest, config: MatrixObservationOrderProjectionConfig) -> None:
        self.manifest = manifest
        self._config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        by_schema: dict[str, list[WorkerInputPort]] = {}
        for port in context.input_ports:
            by_schema.setdefault(port.payload_schema, []).append(port)
        required = {
            MatrixObservationOrderProjectionConfig.SCHEMA,
            MatrixObservationOrderPairedHistoryResult.SCHEMA,
            OBSERVATION_ORDER_HDF5_SCHEMA,
            LinkedCampaignStageEnvelope.SCHEMA,
        }
        if set(by_schema) != required or any(len(by_schema[value]) != 1 for value in required):
            for ports in by_schema.values():
                for port in ports:
                    port.close()
            raise ValueError("observation order projection input roster differs")
        config = cast(
            MatrixObservationOrderProjectionConfig,
            _decode(by_schema[MatrixObservationOrderProjectionConfig.SCHEMA][0], MatrixObservationOrderProjectionConfig),
        )
        result = cast(
            MatrixObservationOrderPairedHistoryResult,
            _decode(by_schema[MatrixObservationOrderPairedHistoryResult.SCHEMA][0], MatrixObservationOrderPairedHistoryResult),
        )
        upstream = cast(
            LinkedCampaignStageEnvelope,
            _decode(
                by_schema[LinkedCampaignStageEnvelope.SCHEMA][0], LinkedCampaignStageEnvelope
            ),
        )
        hdf5 = _read(by_schema[OBSERVATION_ORDER_HDF5_SCHEMA][0], OBSERVATION_ORDER_MAXIMUM_HDF5_BYTES)
        if (
            config != self._config
            or upstream.scientific_product != ObjectIdentity.from_record(result.result_id, result)
            or not context.task_id.startswith(f"{OBSERVATION_ORDER_PROJECTION_TASK_PREFIX}.")
        ):
            raise ValueError("observation order projection config, custody or task identity differs")
        view_id = context.task_id.removeprefix(f"{OBSERVATION_ORDER_PROJECTION_TASK_PREFIX}.")
        if view_id == result.primary_view_id:
            view = ScientificView.PRIMARY
        elif view_id == result.fine_view_id:
            view = ScientificView.FINE
        else:
            raise ValueError("observation order projection task names neither nested view")
        trace = project_observation_order_history_view(
            payload=hdf5,
            result=result,
            config=self._config,
            view=view,
        )
        complete = result.disposition is MatrixObservationOrderAcquisitionDisposition.COMPLETE
        envelope = _stage_envelope(
            task_id=context.task_id,
            role=LinkedCampaignStageRole.EVIDENCE_PROJECTION,
            product_id=trace.trace_id,
            product=trace,
            disposition=(
                LinkedCampaignDisposition.SUPPORTED
                if complete
                else LinkedCampaignDisposition.UNEVALUABLE
            ),
            reasons=() if complete else result.reason_codes,
        )
        payloads = {
            HistoryViewTrace.SCHEMA: trace.canonical_bytes(),
            LinkedCampaignStageEnvelope.SCHEMA: envelope.canonical_bytes(),
        }
        if {value.payload_schema for value in context.output_ports} != set(payloads):
            raise ValueError("observation order projection output roster differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(port.output_id, payloads[port.payload_schema])
                for port in sorted(context.output_ports, key=lambda value: value.output_id)
            ),
            checks=(
                ReceiptCheck("one-view-per-source-artifact", True, ()),
                ReceiptCheck("observation-state-separate-from-availability", True, ()),
            ),
        )


def _scientific_adjudication(
    context: TaskContext,
    evaluation: MatrixObservationOrderEvaluation,
) -> ScientificAdjudicationRecord:
    adjudication = context.scientific_adjudication_context
    if adjudication is None:
        raise ValueError("observation order sealed evaluator lacks its adjudication context")
    status = {
        TwoViewTerminal.CONFIRMED: ScientificStatus.SUPPORTED,
        TwoViewTerminal.MIXED_VIEWS: ScientificStatus.MIXED,
        TwoViewTerminal.NOT_CONFIRMED: ScientificStatus.NOT_SUPPORTED,
    }.get(evaluation.terminal, ScientificStatus.UNEVALUABLE)
    evaluable = status is not ScientificStatus.UNEVALUABLE
    if evaluation.qualification_fixture_id != adjudication.fixture_scope_id:
        raise ValueError("observation order evaluation fixture scope differs from adjudication context")
    return ScientificAdjudicationRecord(
        adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
        run_id=context.run_id,
        adjudication_task_id=context.task_id,
        execution_plan=adjudication.execution_plan,
        input_materialization_ids=context.input_materialization_ids,
        output_logical_artifact_ids=tuple(
            sorted(
                value.logical_artifact_id
                for value in context.output_ports
                if value.logical_artifact_id is not None
            )
        ),
        required_receipt_ids=context.dependency_receipt_ids,
        evidence_world_id=adjudication.evidence_world_id,
        evidence_world_kind=adjudication.evidence_world_kind,
        relation=adjudication.relation,
        independent_unit_id=adjudication.independent_unit_id,
        information_cutoffs=adjudication.information_cutoffs,
        visibility_ceiling=adjudication.visibility_ceiling,
        outcome_access=adjudication.outcome_access,
        evaluability=(
            AdjudicationEvaluability.EVALUABLE
            if evaluable
            else AdjudicationEvaluability.UNEVALUABLE
        ),
        scientific_status=status,
        admission_status=(
            AdmissionStatus.NOT_EVALUATED if evaluable else AdmissionStatus.UNEVALUABLE
        ),
        reason_codes=evaluation.reason_codes,
        fixture_scope_id=adjudication.fixture_scope_id,
        plumbing_only=adjudication.plumbing_only,
    )


class MatrixObservationOrderEvaluationTask:
    def __init__(self, *, manifest: CapabilityManifest, config: MatrixObservationOrderEvaluationConfig) -> None:
        self.manifest = manifest
        self._config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        configs: list[MatrixObservationOrderEvaluationConfig] = []
        traces: list[HistoryViewTrace] = []
        envelopes: list[LinkedCampaignStageEnvelope] = []
        record_types: dict[str, type[CanonicalRecord]] = {
            MatrixObservationOrderEvaluationConfig.SCHEMA: MatrixObservationOrderEvaluationConfig,
            HistoryViewTrace.SCHEMA: HistoryViewTrace,
            LinkedCampaignStageEnvelope.SCHEMA: LinkedCampaignStageEnvelope,
        }
        for port in context.input_ports:
            record_type = record_types.get(port.payload_schema)
            if record_type is None:
                port.close()
                raise ValueError("observation order evaluator received an unknown input schema")
            value = _decode(port, record_type)
            if isinstance(value, MatrixObservationOrderEvaluationConfig):
                configs.append(value)
            elif isinstance(value, HistoryViewTrace):
                traces.append(value)
            else:
                envelopes.append(cast(LinkedCampaignStageEnvelope, value))
        if (
            context.task_id != OBSERVATION_ORDER_EVALUATION_TASK_ID
            or configs != [self._config]
            or len(traces) != self._config.expected_traces
            or len(envelopes) != self._config.expected_traces
        ):
            raise ValueError("observation order evaluator lacks its exact config and paired projection roster")
        trace_identities = {ObjectIdentity.from_record(value.trace_id, value) for value in traces}
        if (
            len(trace_identities) != len(traces)
            or {value.scientific_product for value in envelopes} != trace_identities
            or any(
                value.role is not LinkedCampaignStageRole.EVIDENCE_PROJECTION for value in envelopes
            )
        ):
            raise ValueError("observation order evaluator projection custody roster differs")
        primary, fine, evaluation = evaluate_observation_order(
            traces=tuple(traces),
            config=self._config,
            analysis_identity=context.run_id,
        )
        disposition = (
            LinkedCampaignDisposition.SUPPORTED
            if evaluation.terminal is TwoViewTerminal.CONFIRMED
            else LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE
            if evaluation.terminal
            in {
                TwoViewTerminal.MIXED_VIEWS,
                TwoViewTerminal.NOT_CONFIRMED,
            }
            else LinkedCampaignDisposition.UNEVALUABLE
        )
        envelope = _stage_envelope(
            task_id=context.task_id,
            role=LinkedCampaignStageRole.METHOD_IDENTIFICATION,
            product_id=evaluation.evaluation_id,
            product=evaluation,
            disposition=disposition,
            reasons=()
            if disposition is LinkedCampaignDisposition.SUPPORTED
            else evaluation.reason_codes,
        )
        values = {
            "primary-report": primary.canonical_bytes(),
            "fine-report": fine.canonical_bytes(),
            "evaluation": evaluation.canonical_bytes(),
            "stage-envelope": envelope.canonical_bytes(),
            "scientific-adjudication": _scientific_adjudication(
                context, evaluation
            ).canonical_bytes(),
        }
        observed = {
            _local_output_id(context.task_id, value.output_id) for value in context.output_ports
        }
        if observed != set(values):
            raise ValueError("observation order evaluator output roster differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    port.output_id,
                    values[_local_output_id(context.task_id, port.output_id)],
                )
                for port in sorted(context.output_ports, key=lambda value: value.output_id)
            ),
            checks=(
                ReceiptCheck("physical-history-clustered-inference", True, ()),
                ReceiptCheck("paired-views-noncompensating", True, ()),
                ReceiptCheck("single-terminal-adjudication", True, ()),
            ),
        )


def _config_payload(
    plan: ProtocolExecutionPlan,
    manifest: CapabilityManifest,
    config: MatrixObservationOrderProjectionConfig | MatrixObservationOrderEvaluationConfig,
) -> ExternalInputPayload:
    specs = {
        item.logical_artifact_id: item
        for task in plan.tasks
        if task.capability.capability_key == manifest.capability_key
        for item in task.external_inputs
        if item.expected_payload_schema == config.SCHEMA
    }
    if (
        len(specs) != 1
        or next(iter(specs.values())).expected_content_sha256 != config.fingerprint()
    ):
        raise ValueError("observation order method plan lacks its exact external config")
    identity = ObjectIdentity.from_record(config.config_id, config)
    parent = ArtifactLineageParent(
        identity,
        VisibilityCeiling.PROSPECTIVE,
        OutcomeAccess.OUTCOME_BLIND,
    )
    return ExternalInputPayload.from_bytes(
        logical_artifact_id=next(iter(specs)),
        payload_schema=config.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type=OBSERVATION_ORDER_CANONICAL_MEDIA_TYPE,
        payload=config.canonical_bytes(),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        lineage_parents=(parent,),
        logical_content_sha256=config.fingerprint(),
    )


class _MatrixObservationOrderMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: MatrixObservationOrderProjectionConfig | MatrixObservationOrderEvaluationConfig,
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("observation order method manifest differs from the selected registry")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.manifest = manifest
        self.config = config

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("observation order method execution plan differs")
        return (_config_payload(plan, self.manifest, self.config),)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("observation order method semantic registry differs")
        del execution_plan
        record_types = (
            HistoryViewTrace,
            MatrixObservationOrderViewReport,
            MatrixObservationOrderEvaluation,
            LinkedCampaignStageEnvelope,
            ScientificAdjudicationRecord,
        )
        return tuple(
            sorted(
                (
                    CapabilityOutputSemanticContract.from_manifest(
                        self.manifest,
                        payload_schema=value.SCHEMA,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(sorted(field.name for field in fields(value))),
                    )
                    for value in record_types
                    if value.SCHEMA in self.manifest.output_schema_ids
                ),
                key=lambda value: value.key,
            )
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("observation order method adjudication registry differs")
        del execution_plan
        if self.manifest.capability_key != OBSERVATION_ORDER_EVALUATION_CAPABILITY_KEY:
            return None
        fixture = (
            self.config.qualification_fixture_id
            if isinstance(self.config, MatrixObservationOrderEvaluationConfig)
            else None
        )
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=f"{OBSERVATION_ORDER_EVALUATION_TASK_ID}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            fixture_scope_id=fixture,
            plumbing_only=fixture is not None,
        )


class MatrixObservationOrderProjectionProvider(_MatrixObservationOrderMethodProvider):
    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if (
            registry != self.registry
            or source_records
            or not isinstance(self.config, MatrixObservationOrderProjectionConfig)
        ):
            raise ValueError("observation order projection runner inputs differ")
        return (cast(TaskRunner, MatrixObservationOrderProjectionTask(manifest=self.manifest, config=self.config)),)


class MatrixObservationOrderEvaluationProvider(_MatrixObservationOrderMethodProvider):
    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if (
            registry != self.registry
            or source_records
            or not isinstance(self.config, MatrixObservationOrderEvaluationConfig)
        ):
            raise ValueError("observation order evaluation runner inputs differ")
        return (cast(TaskRunner, MatrixObservationOrderEvaluationTask(manifest=self.manifest, config=self.config)),)


__all__ = [
    "OBSERVATION_ORDER_EVALUATION_CAPABILITY_KEY",
    "OBSERVATION_ORDER_EVALUATION_TASK_ID",
    "OBSERVATION_ORDER_METHOD_CAPABILITY_VERSION",
    "OBSERVATION_ORDER_PROJECTION_CAPABILITY_KEY",
    "OBSERVATION_ORDER_PROJECTION_TASK_PREFIX",
    'MatrixObservationOrderEvaluationProvider',
    'MatrixObservationOrderEvaluationTask',
    'MatrixObservationOrderProjectionProvider',
    'MatrixObservationOrderProjectionTask',
]
