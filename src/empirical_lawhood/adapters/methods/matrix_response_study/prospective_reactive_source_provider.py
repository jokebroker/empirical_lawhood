"""Projection and sole finalization providers for matrix response prospective reactive source law."""

from __future__ import annotations

from dataclasses import fields
from typing import cast

from empirical_lawhood.adapters.simulators.six_matrix_response.prospective_reactive_source import SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANONICAL_MEDIA_TYPE, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA, SixMatrixResponseProspectiveReactiveSourceLawHistoryResult, SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, decode_six_matrix_response_reactive_source_hdf5
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

from .prospective_reactive_source_law import MatrixResponseProspectiveReactiveSourceLawAggregate, MatrixResponseProspectiveReactiveSourceLawMethodConfig, MatrixResponseProspectiveReactiveSourceLawProjectionTerminal, MatrixResponseProspectiveReactiveSourceLawProjection, MatrixResponseProspectiveReactiveSourceLawTerminal, project_matrix_response_study_reactive_source_history, reduce_matrix_response_study_reactive_source


PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_TASK_PREFIX = "matrix-response-prospective-reactive-source-law-project"
PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY_KEY = "matrix-response-prospective-reactive-source-law.causal-source-projection"
PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY_KEY = "matrix-response-prospective-reactive-source-law.source-law-finalization"
PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CAPABILITY_VERSION = "1.0.0"
_MAXIMUM_CANONICAL_BYTES = 16 * 1024 * 1024
_MAXIMUM_HDF5_BYTES = 1024 * 1024 * 1024


def _read(port: object, maximum: int) -> bytes:
    size = cast(int, getattr(port, "size_bytes"))
    if size > maximum:
        raise ValueError("matrix response prospective reactive source law method input exceeds its bound")
    try:
        payload = cast(bytes, getattr(port, "read")(size + 1))
        if len(payload) != size or getattr(port, "read")(1):
            raise ValueError("matrix response prospective reactive source law method input size differs")
        return payload
    finally:
        getattr(port, "close")()


def _decode(port: object, record_type: type[CanonicalRecord]) -> CanonicalRecord:
    return decode_canonical_bytes(
        _read(port, _MAXIMUM_CANONICAL_BYTES),
        record_type,
        maximum_bytes=_MAXIMUM_CANONICAL_BYTES,
    )


def _emit(context: TaskContext, records: tuple[CanonicalRecord, ...]) -> RunnerResult:
    by_schema = {value.SCHEMA: value for value in records}
    if set(by_schema) != {value.payload_schema for value in context.output_ports}:
        raise ValueError("matrix response prospective reactive source law method output roster differs")
    return RunnerResult(
        outputs=tuple(
            TaskOutputPayload(value.output_id, by_schema[value.payload_schema].canonical_bytes())
            for value in sorted(context.output_ports, key=lambda item: item.output_id)
        ),
        checks=(
            ReceiptCheck("artifact-plane-inputs-only", True, ()),
            ReceiptCheck("one-frozen-reducer", True, ()),
        ),
    )


def _envelope(task_id: str, role: LinkedCampaignStageRole, product: CanonicalRecord, disposition: LinkedCampaignDisposition, reasons: tuple[str, ...]) -> LinkedCampaignStageEnvelope:
    object_id = cast(str, getattr(product, "projection_id", getattr(product, "aggregate_id", "")))
    return LinkedCampaignStageEnvelope(
        envelope_id=f"stage-envelope.{task_id}", role=role, disposition=disposition,
        scientific_product=ObjectIdentity.from_record(object_id, product), reason_codes=reasons,
    )


class MatrixResponseProspectiveReactiveSourceLawProjectionTask:
    def __init__(self, *, manifest: CapabilityManifest, source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, method: MatrixResponseProspectiveReactiveSourceLawMethodConfig) -> None:
        self.manifest = manifest
        self._source = source
        self._method = method

    def execute(self, context: TaskContext) -> RunnerResult:
        by_schema: dict[str, list[WorkerInputPort]] = {}
        for port in context.input_ports:
            by_schema.setdefault(port.payload_schema, []).append(port)
        required = {
            MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA,
            SixMatrixResponseProspectiveReactiveSourceLawHistoryResult.SCHEMA,
            SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA,
            LinkedCampaignStageEnvelope.SCHEMA,
        }
        if set(by_schema) != required or any(len(by_schema[value]) != 1 for value in required):
            for values in by_schema.values():
                for port in values:
                    port.close()
            raise ValueError("matrix response prospective reactive source law projection input roster differs")
        method = cast(MatrixResponseProspectiveReactiveSourceLawMethodConfig, _decode(by_schema[MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA][0], MatrixResponseProspectiveReactiveSourceLawMethodConfig))
        result = cast(SixMatrixResponseProspectiveReactiveSourceLawHistoryResult, _decode(by_schema[SixMatrixResponseProspectiveReactiveSourceLawHistoryResult.SCHEMA][0], SixMatrixResponseProspectiveReactiveSourceLawHistoryResult))
        upstream = cast(LinkedCampaignStageEnvelope, _decode(by_schema[LinkedCampaignStageEnvelope.SCHEMA][0], LinkedCampaignStageEnvelope))
        payload = _read(by_schema[SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA][0], _MAXIMUM_HDF5_BYTES)
        if method != self._method or upstream.scientific_product != ObjectIdentity.from_record(result.result_id, result):
            raise ValueError("matrix response prospective reactive source law projection custody differs")
        prefix = f"{self._method.projection_task_prefix}."
        if not context.task_id.startswith(prefix) or result.scientific_view_id != context.task_id.removeprefix(prefix):
            raise ValueError("matrix response prospective reactive source law projection view differs")
        persisted = decode_six_matrix_response_reactive_source_hdf5(payload, result)
        projection = project_matrix_response_study_reactive_source_history(persisted=persisted, source=self._source, method=self._method)
        complete = projection.terminal is MatrixResponseProspectiveReactiveSourceLawProjectionTerminal.COMPLETE
        envelope = _envelope(
            context.task_id, LinkedCampaignStageRole.EVIDENCE_PROJECTION, projection,
            LinkedCampaignDisposition.SUPPORTED if complete else LinkedCampaignDisposition.UNEVALUABLE,
            projection.reason_codes,
        )
        return _emit(context, (projection, envelope))


def _adjudication(context: TaskContext, aggregate: MatrixResponseProspectiveReactiveSourceLawAggregate) -> ScientificAdjudicationRecord:
    adjudication = context.scientific_adjudication_context
    if adjudication is None:
        raise ValueError("matrix response prospective reactive source law adjudication context is absent")
    supported = aggregate.terminal in {
        MatrixResponseProspectiveReactiveSourceLawTerminal.DEVELOPMENT_SCORE_FROZEN,
        MatrixResponseProspectiveReactiveSourceLawTerminal.SUPPORTED,
    }
    negative = aggregate.terminal in {
        MatrixResponseProspectiveReactiveSourceLawTerminal.NO_DEVELOPMENT_SOURCE_RULE,
        MatrixResponseProspectiveReactiveSourceLawTerminal.CONDITIONAL,
        MatrixResponseProspectiveReactiveSourceLawTerminal.NO_SOURCE,
    }
    return ScientificAdjudicationRecord(
        adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
        run_id=context.run_id, adjudication_task_id=context.task_id,
        execution_plan=adjudication.execution_plan,
        input_materialization_ids=context.input_materialization_ids,
        output_logical_artifact_ids=tuple(sorted(
            value.logical_artifact_id for value in context.output_ports if value.logical_artifact_id is not None
        )),
        required_receipt_ids=context.dependency_receipt_ids,
        evidence_world_id=adjudication.evidence_world_id,
        evidence_world_kind=adjudication.evidence_world_kind,
        relation=adjudication.relation,
        independent_unit_id=adjudication.independent_unit_id,
        information_cutoffs=adjudication.information_cutoffs,
        visibility_ceiling=adjudication.visibility_ceiling,
        outcome_access=adjudication.outcome_access,
        evaluability=AdjudicationEvaluability.EVALUABLE if supported or negative else AdjudicationEvaluability.UNEVALUABLE,
        scientific_status=(ScientificStatus.SUPPORTED if supported else ScientificStatus.NOT_SUPPORTED if negative else ScientificStatus.UNEVALUABLE),
        admission_status=AdmissionStatus.NOT_EVALUATED if supported or negative else AdmissionStatus.UNEVALUABLE,
        reason_codes=aggregate.reason_codes or (aggregate.terminal.value,),
    )


class MatrixResponseProspectiveReactiveSourceLawFinalizationTask:
    def __init__(self, *, manifest: CapabilityManifest, source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, method: MatrixResponseProspectiveReactiveSourceLawMethodConfig) -> None:
        self.manifest = manifest
        self._source = source
        self._method = method

    def execute(self, context: TaskContext) -> RunnerResult:
        methods: list[MatrixResponseProspectiveReactiveSourceLawMethodConfig] = []
        projections: list[MatrixResponseProspectiveReactiveSourceLawProjection] = []
        envelopes: list[LinkedCampaignStageEnvelope] = []
        types: dict[str, type[CanonicalRecord]] = {
            MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA: MatrixResponseProspectiveReactiveSourceLawMethodConfig,
            MatrixResponseProspectiveReactiveSourceLawProjection.SCHEMA: MatrixResponseProspectiveReactiveSourceLawProjection,
            LinkedCampaignStageEnvelope.SCHEMA: LinkedCampaignStageEnvelope,
        }
        for port in context.input_ports:
            record_type = types.get(port.payload_schema)
            if record_type is None:
                port.close()
                raise ValueError("matrix response prospective reactive source law finalizer received an unknown input")
            value = _decode(port, record_type)
            if isinstance(value, MatrixResponseProspectiveReactiveSourceLawMethodConfig):
                methods.append(value)
            elif isinstance(value, MatrixResponseProspectiveReactiveSourceLawProjection):
                projections.append(value)
            else:
                envelopes.append(cast(LinkedCampaignStageEnvelope, value))
        if methods != [self._method] or len(projections) != 320 or len(envelopes) != 320:
            raise ValueError("matrix response prospective reactive source law finalizer lacks the complete 320-view roster")
        aggregate = reduce_matrix_response_study_reactive_source(
            source=self._source, method=self._method, projections=tuple(projections),
            analysis_identity=context.run_id,
        )
        disposition = (
            LinkedCampaignDisposition.SUPPORTED
            if aggregate.terminal in {MatrixResponseProspectiveReactiveSourceLawTerminal.DEVELOPMENT_SCORE_FROZEN, MatrixResponseProspectiveReactiveSourceLawTerminal.SUPPORTED}
            else LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE
            if aggregate.terminal in {MatrixResponseProspectiveReactiveSourceLawTerminal.NO_DEVELOPMENT_SOURCE_RULE, MatrixResponseProspectiveReactiveSourceLawTerminal.CONDITIONAL, MatrixResponseProspectiveReactiveSourceLawTerminal.NO_SOURCE}
            else LinkedCampaignDisposition.UNEVALUABLE
        )
        records: tuple[CanonicalRecord, ...] = (
            aggregate,
            _envelope(context.task_id, LinkedCampaignStageRole.METHOD_IDENTIFICATION, aggregate, disposition, aggregate.reason_codes),
        )
        if any(value.payload_schema == ScientificAdjudicationRecord.SCHEMA for value in context.output_ports):
            records = (*records, _adjudication(context, aggregate))
        return _emit(context, records)


def _config_payload(plan: ProtocolExecutionPlan, manifest: CapabilityManifest, config: MatrixResponseProspectiveReactiveSourceLawMethodConfig) -> ExternalInputPayload:
    specs = {
        item.logical_artifact_id: item
        for task in plan.tasks
        if task.capability.capability_key == manifest.capability_key
        for item in task.external_inputs
        if item.expected_payload_schema == config.SCHEMA
    }
    if len(specs) != 1 or next(iter(specs.values())).expected_content_sha256 != config.fingerprint():
        raise ValueError("matrix response prospective reactive source law method plan lacks its exact config")
    identity = ObjectIdentity.from_record(config.config_id, config)
    parent = ArtifactLineageParent(identity, VisibilityCeiling.PROSPECTIVE, OutcomeAccess.OUTCOME_BLIND)
    return ExternalInputPayload.from_bytes(
        logical_artifact_id=next(iter(specs)), payload_schema=config.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON, media_type=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_CANONICAL_MEDIA_TYPE,
        payload=config.canonical_bytes(), visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,), lineage_parents=(parent,),
        logical_content_sha256=config.fingerprint(),
    )


class _Provider(CampaignRuntimeProvider):
    def __init__(self, *, registry: CapabilityRegistry, manifest: CapabilityManifest, source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, method: MatrixResponseProspectiveReactiveSourceLawMethodConfig) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("matrix response prospective reactive source law method manifest differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.manifest = manifest
        self.capability_count = 1
        self.source = source
        self.method = method

    def external_inputs(self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("matrix response prospective reactive source law method plan differs")
        return (_config_payload(plan, self.manifest, self.method),)

    def output_semantic_contracts(self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("matrix response prospective reactive source law method semantic registry differs")
        del execution_plan
        records = (MatrixResponseProspectiveReactiveSourceLawProjection, MatrixResponseProspectiveReactiveSourceLawAggregate, LinkedCampaignStageEnvelope, ScientificAdjudicationRecord)
        return tuple(sorted((
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest, payload_schema=value.SCHEMA, profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(field.name for field in fields(value))),
            )
            for value in records if value.SCHEMA in self.manifest.output_schema_ids
        ), key=lambda value: value.key))

    def scientific_adjudication_contract(self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("matrix response prospective reactive source law adjudication registry differs")
        del execution_plan
        if self.manifest.capability_key != PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY_KEY:
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=f"{self.method.aggregate_task_id}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            maximum_bytes=128 * 1024,
        )


class MatrixResponseProspectiveReactiveSourceLawProjectionProvider(_Provider):
    def runners(self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("matrix response prospective reactive source law projection runner inputs differ")
        return (cast(TaskRunner, MatrixResponseProspectiveReactiveSourceLawProjectionTask(manifest=self.manifest, source=self.source, method=self.method)),)


class MatrixResponseProspectiveReactiveSourceLawFinalizationProvider(_Provider):
    def runners(self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("matrix response prospective reactive source law finalizer runner inputs differ")
        return (cast(TaskRunner, MatrixResponseProspectiveReactiveSourceLawFinalizationTask(manifest=self.manifest, source=self.source, method=self.method)),)


__all__ = [
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY_KEY", "PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CAPABILITY_VERSION",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY_KEY", "PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_TASK_PREFIX",
    'MatrixResponseProspectiveReactiveSourceLawFinalizationProvider', 'MatrixResponseProspectiveReactiveSourceLawProjectionProvider',
]
