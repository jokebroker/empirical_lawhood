"""Consolidated projection and sole reactive entrance finalization providers.

The provider reads only artifact-plane inputs produced by the issued campaign.
It applies the already-frozen method and emits one aggregate terminal; it does
not fit a law or open a reactive-nesting follow-on.
"""

from __future__ import annotations

from dataclasses import fields
from typing import cast

from empirical_lawhood.adapters.simulators.six_matrix_response.reactive_entrance import REACTIVE_ENTRANCE_CANONICAL_MEDIA_TYPE, REACTIVE_ENTRANCE_HDF5_SCHEMA, SixMatrixResponseReactiveEntrancePrecursorResult, SixMatrixResponseReactiveEntranceSourceConfig, decode_six_matrix_response_reactive_entrance_hdf5
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
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

from .reactive_entrance_source import MatrixResponseReactiveEntranceAggregate, MatrixResponseReactiveEntranceMethodConfig, MatrixResponseReactiveEntranceProjectionTerminal, MatrixResponseReactiveEntranceProjection, MatrixResponseReactiveEntranceTerminal, project_matrix_response_study_reactive_entrance_precursor, reduce_matrix_response_study_reactive_entrance_source


REACTIVE_ENTRANCE_PROJECTION_TASK_PREFIX = "matrix-response-reactive-entrance-project"
REACTIVE_ENTRANCE_PROJECTION_CAPABILITY_KEY = "matrix-response-reactive-entrance.reactive-entrance-projection"
REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY_KEY = "matrix-response-reactive-entrance.source-finalization"
REACTIVE_ENTRANCE_METHOD_CAPABILITY_VERSION = "1.0.0"
REACTIVE_ENTRANCE_LAW_CONDITION_FALSE_TASK_ID = "linked-law-qualification"
REACTIVE_ENTRANCE_ATLAS_CONDITION_FALSE_TASK_ID = "linked-atlas-assembly"
_MAXIMUM_CANONICAL_INPUT_BYTES = 4 * 1024 * 1024
_MAXIMUM_HDF5_INPUT_BYTES = 32 * 1024 * 1024


def _read(port: object, *, maximum_bytes: int) -> bytes:
    size = cast(int, getattr(port, "size_bytes"))
    if size > maximum_bytes:
        raise ValueError("matrix response reactive entrance task input exceeds its bound")
    try:
        payload = cast(bytes, getattr(port, "read")(size + 1))
        if len(payload) != size or getattr(port, "read")(1):
            raise ValueError("matrix response reactive entrance task input size differs")
        return payload
    finally:
        getattr(port, "close")()


def _decode(port: object, record_type: type[CanonicalRecord]) -> CanonicalRecord:
    return decode_canonical_bytes(
        _read(port, maximum_bytes=_MAXIMUM_CANONICAL_INPUT_BYTES),
        record_type,
        maximum_bytes=_MAXIMUM_CANONICAL_INPUT_BYTES,
    )


def _emit(context: TaskContext, records: tuple[CanonicalRecord, ...]) -> RunnerResult:
    by_schema = {value.SCHEMA: value for value in records}
    if set(by_schema) != {value.payload_schema for value in context.output_ports}:
        raise ValueError("matrix response reactive entrance method output roster differs")
    values = {
        output.output_id: by_schema[output.payload_schema].canonical_bytes()
        for output in context.output_ports
    }
    return RunnerResult(
        outputs=tuple(
            TaskOutputPayload(output_id, values[output_id]) for output_id in sorted(values)
        ),
        checks=(
            ReceiptCheck("artifact-plane-inputs-only", True, ()),
            ReceiptCheck("single-reactive-entrance-finalizer", True, ()),
        ),
    )


def _projection_envelope(
    task_id: str,
    projection: MatrixResponseReactiveEntranceProjection,
) -> LinkedCampaignStageEnvelope:
    complete = projection.terminal is MatrixResponseReactiveEntranceProjectionTerminal.COMPLETE
    return LinkedCampaignStageEnvelope(
        envelope_id=f"stage-envelope.{task_id}",
        role=LinkedCampaignStageRole.EVIDENCE_PROJECTION,
        disposition=(
            LinkedCampaignDisposition.SUPPORTED
            if complete
            else LinkedCampaignDisposition.UNEVALUABLE
        ),
        scientific_product=ObjectIdentity.from_record(
            projection.projection_id,
            projection,
        ),
        reason_codes=() if complete else projection.reason_codes,
    )


def _aggregate_envelope(
    task_id: str,
    aggregate: MatrixResponseReactiveEntranceAggregate,
) -> LinkedCampaignStageEnvelope:
    disposition = {
        MatrixResponseReactiveEntranceTerminal.QUALIFIED: LinkedCampaignDisposition.SUPPORTED,
        MatrixResponseReactiveEntranceTerminal.TOO_SPARSE: LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE,
        MatrixResponseReactiveEntranceTerminal.NUMERICAL_INVALID: LinkedCampaignDisposition.UNEVALUABLE,
        MatrixResponseReactiveEntranceTerminal.CUSTODY_INVALID: LinkedCampaignDisposition.UNEVALUABLE,
        MatrixResponseReactiveEntranceTerminal.PREREQUISITE_NONATTEMPT: LinkedCampaignDisposition.UNEVALUABLE,
    }[aggregate.terminal]
    return LinkedCampaignStageEnvelope(
        envelope_id=f"stage-envelope.{task_id}",
        role=LinkedCampaignStageRole.METHOD_IDENTIFICATION,
        disposition=disposition,
        scientific_product=ObjectIdentity.from_record(aggregate.aggregate_id, aggregate),
        reason_codes=aggregate.reason_codes,
    )


def _scientific_adjudication(
    context: TaskContext,
    aggregate: MatrixResponseReactiveEntranceAggregate,
) -> ScientificAdjudicationRecord:
    adjudication = context.scientific_adjudication_context
    if adjudication is None:
        raise ValueError("matrix response reactive entrance scientific adjudication context is absent")
    evaluable = aggregate.terminal in {
        MatrixResponseReactiveEntranceTerminal.QUALIFIED,
        MatrixResponseReactiveEntranceTerminal.TOO_SPARSE,
    }
    scientific_status = {
        MatrixResponseReactiveEntranceTerminal.QUALIFIED: ScientificStatus.SUPPORTED,
        MatrixResponseReactiveEntranceTerminal.TOO_SPARSE: ScientificStatus.NOT_SUPPORTED,
        MatrixResponseReactiveEntranceTerminal.NUMERICAL_INVALID: ScientificStatus.UNEVALUABLE,
        MatrixResponseReactiveEntranceTerminal.CUSTODY_INVALID: ScientificStatus.UNEVALUABLE,
        MatrixResponseReactiveEntranceTerminal.PREREQUISITE_NONATTEMPT: ScientificStatus.UNEVALUABLE,
    }[aggregate.terminal]
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
        scientific_status=scientific_status,
        admission_status=(
            AdmissionStatus.NOT_EVALUATED if evaluable else AdmissionStatus.UNEVALUABLE
        ),
        reason_codes=aggregate.reason_codes
        or (f"REACTIVE_ENTRANCE_{aggregate.terminal.value}",),
    )


def _condition_false_envelope(
    *, task_id: str, role: LinkedCampaignStageRole, reason: str
) -> LinkedCampaignStageEnvelope:
    return LinkedCampaignStageEnvelope(
        envelope_id=f"stage-envelope.{task_id}",
        role=role,
        disposition=LinkedCampaignDisposition.CONDITION_FALSE,
        scientific_product=None,
        reason_codes=(reason,),
    )


class MatrixResponseReactiveEntranceProjectionTask:
    """Decode one persisted precursor and apply the frozen entrance reducer."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        source_config: SixMatrixResponseReactiveEntranceSourceConfig,
        method_config: MatrixResponseReactiveEntranceMethodConfig,
    ) -> None:
        self.manifest = manifest
        self._source = source_config
        self._method = method_config

    def execute(self, context: TaskContext) -> RunnerResult:
        by_schema: dict[str, list[WorkerInputPort]] = {}
        for port in context.input_ports:
            by_schema.setdefault(port.payload_schema, []).append(port)
        required = {
            MatrixResponseReactiveEntranceMethodConfig.SCHEMA,
            SixMatrixResponseReactiveEntrancePrecursorResult.SCHEMA,
            REACTIVE_ENTRANCE_HDF5_SCHEMA,
            LinkedCampaignStageEnvelope.SCHEMA,
        }
        if set(by_schema) != required or any(len(by_schema[value]) != 1 for value in required):
            for values in by_schema.values():
                for port in values:
                    getattr(port, "close")()
            raise ValueError("matrix response reactive entrance projection input roster differs")
        method = cast(
            MatrixResponseReactiveEntranceMethodConfig,
            _decode(by_schema[MatrixResponseReactiveEntranceMethodConfig.SCHEMA][0], MatrixResponseReactiveEntranceMethodConfig),
        )
        result = cast(
            SixMatrixResponseReactiveEntrancePrecursorResult,
            _decode(by_schema[SixMatrixResponseReactiveEntrancePrecursorResult.SCHEMA][0], SixMatrixResponseReactiveEntrancePrecursorResult),
        )
        envelope = cast(
            LinkedCampaignStageEnvelope,
            _decode(
                by_schema[LinkedCampaignStageEnvelope.SCHEMA][0],
                LinkedCampaignStageEnvelope,
            ),
        )
        hdf5 = _read(
            by_schema[REACTIVE_ENTRANCE_HDF5_SCHEMA][0],
            maximum_bytes=_MAXIMUM_HDF5_INPUT_BYTES,
        )
        if method != self._method:
            raise ValueError("matrix response reactive entrance projection received a substituted method config")
        if (
            envelope.role is not LinkedCampaignStageRole.SOURCE_MATERIALIZATION
            or envelope.disposition is not LinkedCampaignDisposition.SUPPORTED
            or envelope.scientific_product
            != ObjectIdentity.from_record(result.result_id, result)
        ):
            raise ValueError("matrix response reactive entrance projection source custody envelope differs")
        prefix = f"{self._method.projection_task_prefix}."
        if not context.task_id.startswith(prefix):
            raise ValueError("matrix response reactive entrance projection task identity differs")
        view_id = context.task_id[len(prefix) :]
        if result.scientific_view_id != view_id:
            raise ValueError("matrix response reactive entrance projection changes its issued scientific view")
        persisted = decode_six_matrix_response_reactive_entrance_hdf5(payload=hdf5, result=result)
        projection = project_matrix_response_study_reactive_entrance_precursor(
            persisted=persisted,
            source_config=self._source,
            method_config=self._method,
        )
        return _emit(context, (projection, _projection_envelope(context.task_id, projection)))


class MatrixResponseReactiveEntranceFinalizationTask:
    """Reduce the complete projection roster or close fixed reactive entrance descendants."""

    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        source_config: SixMatrixResponseReactiveEntranceSourceConfig,
        method_config: MatrixResponseReactiveEntranceMethodConfig,
    ) -> None:
        self.manifest = manifest
        self._source = source_config
        self._method = method_config

    def execute(self, context: TaskContext) -> RunnerResult:
        method_values: list[MatrixResponseReactiveEntranceMethodConfig] = []
        projections: list[MatrixResponseReactiveEntranceProjection] = []
        aggregates: list[MatrixResponseReactiveEntranceAggregate] = []
        envelopes: list[LinkedCampaignStageEnvelope] = []
        types: dict[str, type[CanonicalRecord]] = {
            MatrixResponseReactiveEntranceMethodConfig.SCHEMA: MatrixResponseReactiveEntranceMethodConfig,
            MatrixResponseReactiveEntranceProjection.SCHEMA: MatrixResponseReactiveEntranceProjection,
            MatrixResponseReactiveEntranceAggregate.SCHEMA: MatrixResponseReactiveEntranceAggregate,
            LinkedCampaignStageEnvelope.SCHEMA: LinkedCampaignStageEnvelope,
        }
        for port in context.input_ports:
            record_type = types.get(port.payload_schema)
            if record_type is None:
                port.close()
                raise ValueError("matrix response reactive entrance finalization received an unknown input schema")
            value = _decode(port, record_type)
            if isinstance(value, MatrixResponseReactiveEntranceMethodConfig):
                method_values.append(value)
            elif isinstance(value, MatrixResponseReactiveEntranceProjection):
                projections.append(value)
            elif isinstance(value, MatrixResponseReactiveEntranceAggregate):
                aggregates.append(value)
            else:
                envelopes.append(cast(LinkedCampaignStageEnvelope, value))
        if method_values != [self._method]:
            raise ValueError("matrix response reactive entrance finalization lacks its exact method config")
        if context.task_id == self._method.aggregate_task_id:
            if aggregates or len(projections) != 640 or len(envelopes) != 640:
                raise ValueError("matrix response reactive entrance finalization lacks the exact 640-view roster")
            if {value.role for value in envelopes} != {
                LinkedCampaignStageRole.EVIDENCE_PROJECTION
            }:
                raise ValueError("matrix response reactive entrance projection envelopes differ")
            aggregate = reduce_matrix_response_study_reactive_entrance_source(
                source_config=self._source,
                method_config=self._method,
                projections=tuple(projections),
            )
            records: tuple[CanonicalRecord, ...] = (
                aggregate,
                _aggregate_envelope(context.task_id, aggregate),
            )
            if any(
                value.payload_schema == ScientificAdjudicationRecord.SCHEMA
                for value in context.output_ports
            ):
                records = (*records, _scientific_adjudication(context, aggregate))
            return _emit(context, records)
        if context.task_id == REACTIVE_ENTRANCE_LAW_CONDITION_FALSE_TASK_ID:
            if len(aggregates) != 1 or projections or len(envelopes) != 1:
                raise ValueError("matrix response reactive entrance law nonattempt parent differs")
            return _emit(
                context,
                (
                    _condition_false_envelope(
                        task_id=context.task_id,
                        role=LinkedCampaignStageRole.LAW_QUALIFICATION,
                        reason="REACTIVE_NESTING_REQUIRES_SEPARATE_POSITIVE_ENTRANCE_ISSUE",
                    ),
                ),
            )
        if context.task_id == REACTIVE_ENTRANCE_ATLAS_CONDITION_FALSE_TASK_ID:
            if aggregates or projections or len(envelopes) != 1 or (
                envelopes[0].role is not LinkedCampaignStageRole.LAW_QUALIFICATION
                or envelopes[0].disposition is not LinkedCampaignDisposition.CONDITION_FALSE
            ):
                raise ValueError("matrix response reactive entrance atlas nonattempt parent differs")
            return _emit(
                context,
                (
                    _condition_false_envelope(
                        task_id=context.task_id,
                        role=LinkedCampaignStageRole.ATLAS_ASSEMBLY,
                        reason="UPSTREAM_REACTIVE_NESTING_NOT_ISSUED_CONDITION_FALSE",
                    ),
                ),
            )
        raise ValueError("matrix response reactive entrance finalizer task identity differs")


def _config_payload(
    *, plan: ProtocolExecutionPlan, manifest: CapabilityManifest, config: MatrixResponseReactiveEntranceMethodConfig
) -> ExternalInputPayload:
    tasks = tuple(
        value
        for value in plan.tasks
        if value.capability.capability_key == manifest.capability_key
    )
    specs = {
        item.logical_artifact_id: item
        for task in tasks
        for item in task.external_inputs
        if item.expected_payload_schema == config.SCHEMA
    }
    if (
        len(specs) != 1
        or next(iter(specs.values())).expected_content_sha256 != config.fingerprint()
    ):
        raise ValueError("matrix response reactive entrance method tasks lack their exact issued config")
    parent = ArtifactLineageParent(
        ObjectIdentity.from_record(config.config_id, config),
        VisibilityCeiling.PROSPECTIVE,
        OutcomeAccess.OUTCOME_BLIND,
    )
    return ExternalInputPayload.from_bytes(
        logical_artifact_id=next(iter(specs)),
        payload_schema=config.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type=REACTIVE_ENTRANCE_CANONICAL_MEDIA_TYPE,
        payload=config.canonical_bytes(),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        lineage_parents=(parent,),
        logical_content_sha256=config.fingerprint(),
    )


class _MatrixReactiveEntranceMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        source_config: SixMatrixResponseReactiveEntranceSourceConfig,
        method_config: MatrixResponseReactiveEntranceMethodConfig,
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("matrix response reactive entrance method manifest differs from installed registry")
        if method_config.source_config != ObjectIdentity.from_record(
            source_config.config_id, source_config
        ):
            raise ValueError("matrix response reactive entrance method provider changes its source")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.manifest = manifest
        self.capability_count = 1
        self.source_config = source_config
        self.method_config = method_config

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("matrix response reactive entrance method plan/input registry differs")
        return (_config_payload(plan=plan, manifest=self.manifest, config=self.method_config),)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("matrix response reactive entrance method semantic registry differs")
        del execution_plan
        record_types = {
            MatrixResponseReactiveEntranceProjection.SCHEMA: MatrixResponseReactiveEntranceProjection,
            MatrixResponseReactiveEntranceAggregate.SCHEMA: MatrixResponseReactiveEntranceAggregate,
            LinkedCampaignStageEnvelope.SCHEMA: LinkedCampaignStageEnvelope,
            ScientificAdjudicationRecord.SCHEMA: ScientificAdjudicationRecord,
        }
        return tuple(
            sorted(
                (
                    CapabilityOutputSemanticContract.from_manifest(
                        self.manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(sorted(value.name for value in fields(record_type))),
                    )
                    for schema, record_type in record_types.items()
                    if schema in self.manifest.output_schema_ids
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
            raise ValueError("matrix response reactive entrance method adjudication registry differs")
        del execution_plan
        if self.manifest.capability_key != REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY_KEY:
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=f"{self.method_config.aggregate_task_id}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            maximum_bytes=128 * 1024,
        )


class MatrixResponseReactiveEntranceProjectionProvider(_MatrixReactiveEntranceMethodProvider):
    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("matrix response reactive entrance projection runner inputs differ")
        return (
            cast(
                TaskRunner,
                MatrixResponseReactiveEntranceProjectionTask(
                    manifest=self.manifest,
                    source_config=self.source_config,
                    method_config=self.method_config,
                ),
            ),
        )


class MatrixResponseReactiveEntranceFinalizationProvider(_MatrixReactiveEntranceMethodProvider):
    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("matrix response reactive entrance finalization runner inputs differ")
        return (
            cast(
                TaskRunner,
                MatrixResponseReactiveEntranceFinalizationTask(
                    manifest=self.manifest,
                    source_config=self.source_config,
                    method_config=self.method_config,
                ),
            ),
        )


__all__ = [
    "REACTIVE_ENTRANCE_ATLAS_CONDITION_FALSE_TASK_ID",
    "REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY_KEY",
    "REACTIVE_ENTRANCE_LAW_CONDITION_FALSE_TASK_ID",
    "REACTIVE_ENTRANCE_METHOD_CAPABILITY_VERSION",
    "REACTIVE_ENTRANCE_PROJECTION_CAPABILITY_KEY",
    "REACTIVE_ENTRANCE_PROJECTION_TASK_PREFIX",
    'MatrixResponseReactiveEntranceFinalizationProvider',
    'MatrixResponseReactiveEntranceFinalizationTask',
    'MatrixResponseReactiveEntranceProjectionProvider',
    'MatrixResponseReactiveEntranceProjectionTask',
]
