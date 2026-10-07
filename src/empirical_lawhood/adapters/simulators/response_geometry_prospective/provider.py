"""assay source task at the ordinary runtime ports and immutable artifact boundary."""

from dataclasses import fields
from decimal import Decimal
from hashlib import sha256
from typing import Any, Callable, TypeVar, cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.planning.observation_order import ObservationPhysicalUnit
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskProgressEmitter,
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
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeConfig, ResponseGeometryAssayNativeSegment, ResponseGeometryAssayNativeSegmentResult, assay_segments
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import ASSAY_HDF5_SCHEMA, execute_assay_segment

from .discovery import CANONICAL_MEDIA_TYPE


Record = TypeVar("Record", bound=CanonicalRecord)


def validate_response_geometry_assay_native_roster(
    config: ResponseGeometryAssayNativeConfig, carrier: FreshSourceQualificationExperiment
) -> None:
    """Bind the neutral carrier's unit, view, clock and action roles to assay."""
    from empirical_lawhood.adapters.simulators.six_matrix_response.response_panel import assay_physical_unit_id

    prefix = "response-geometry-assay"
    expected_units = tuple(
        ObservationPhysicalUnit(
            assay_physical_unit_id(config, root),
            f"{prefix}.coordinate.{root.context}.t{root.landmark_tick}",
            f"{assay_physical_unit_id(config, root)}.instance",
            f"{prefix}.family.{root.context}",
            root.fingerprint(),
            None,
        )
        for root in config.roots
    )
    if carrier.physical_units != expected_units or (
        carrier.receiver_ids != (f"{prefix}.receiver",)
        or carrier.clock_ids != (f"{prefix}.reference-clock",)
    ):
        raise ValueError("assay source carrier changes its preparation/receiver/clock roles")
    native = {segment.task_id: segment for segment in assay_segments()}
    if {value.segment_id for value in carrier.segments} != set(native):
        raise ValueError("assay source carrier changes the fixed native segment roster")
    for segment in carrier.segments:
        exact = native[segment.segment_id]
        occurrence = exact.predecessor if exact.phase == "resume" else exact
        assert occurrence is not None
        action_ids = () if exact.phase == "prefix" else (f"action.{occurrence.task_id}",)
        if (
            segment.physical_independent_unit_id != assay_physical_unit_id(config, exact.root)
            or segment.acquisition_group_id != f"group.{exact.task_id}"
            or segment.start != exact.start_tick
            or segment.end != exact.end_tick
            or segment.predecessor_segment_id
            != (None if exact.predecessor is None else exact.predecessor.task_id)
            or segment.native_clock_id != f"{prefix}.reference-clock"
            or segment.action_occurrence_ids != action_ids
            or segment.view_ids
            != tuple(f"{exact.root.root_id}.project.r{r}" for r in exact.root.refinements)
        ):
            raise ValueError(
                "assay source carrier changes native unit/view/action/continuation semantics"
            )
    expected_views = tuple(
        (
            f"{root.root_id}.project.r{r}",
            assay_physical_unit_id(config, root),
            f"response-geometry.view.r{r}",
        )
        for root in config.roots
        for r in root.refinements
    )
    if (
        tuple(
            (view.view_id, view.physical_independent_unit_id, view.numerical_view_id)
            for view in carrier.views
        )
        != expected_views
    ):
        raise ValueError("assay source carrier changes its diagnostic/numerical view mapping")


def read_port(port: WorkerInputPort, maximum: int) -> bytes:
    try:
        if not 0 < port.size_bytes <= maximum:
            raise ValueError("assay input exceeds its declared bound")
        payload = port.read(port.size_bytes + 1)
        if len(payload) != port.size_bytes or port.read(1):
            raise ValueError("assay input length differs from its immutable materialization")
        return payload
    finally:
        port.close()


def decode_port(
    port: WorkerInputPort, record_type: type[Record], *, maximum: int = 4 * 1024**2
) -> Record:
    return decode_canonical_bytes(
        read_port(port, maximum), record_type, maximum_bytes=maximum
    )


def envelope(
    task_id: str,
    product: CanonicalRecord,
    product_id: str,
    role: LinkedCampaignStageRole,
    complete: bool,
    reasons: tuple[str, ...],
) -> LinkedCampaignStageEnvelope:
    return LinkedCampaignStageEnvelope(
        f"stage-envelope.{task_id}",
        role,
        LinkedCampaignDisposition.SUPPORTED if complete else LinkedCampaignDisposition.UNEVALUABLE,
        ObjectIdentity.from_record(product_id, product),
        () if complete else reasons,
    )


def source_stage_envelope(result: ResponseGeometryAssayNativeSegmentResult) -> LinkedCampaignStageEnvelope:
    return envelope(
        result.segment.task_id,
        result,
        result.result_id,
        LinkedCampaignStageRole.SOURCE_MATERIALIZATION,
        all(view.disposition == "COMPLETE" for view in result.views),
        tuple(sorted({view.reason for view in result.views if view.reason is not None})),
    )


def output_result(
    context: TaskContext, values: dict[str, bytes], checks: tuple[str, ...]
) -> RunnerResult:
    if len(context.output_ports) != len(values) or {
        port.payload_schema for port in context.output_ports
    } != set(values):
        raise ValueError("assay task output roster differs from its installed contract")
    return RunnerResult(
        tuple(
            TaskOutputPayload(port.output_id, values[port.payload_schema])
            for port in sorted(context.output_ports, key=lambda value: value.output_id)
        ),
        tuple(ReceiptCheck(check, True, ()) for check in sorted(checks)),
    )


def config_payloads(
    plan: ProtocolExecutionPlan,
    manifest: CapabilityManifest,
    config: CanonicalRecord,
    config_id: str,
    expected_tasks: set[str],
) -> tuple[ExternalInputPayload, ...]:
    tasks = tuple(
        task for task in plan.tasks if task.capability.capability_key == manifest.capability_key
    )
    if {task.task_id for task in tasks} != expected_tasks:
        raise ValueError("assay provider task roster differs from its frozen complete topology")
    payload = config.canonical_bytes()
    expected_config_ids = {f"config-artifact.{config_id}"}
    if isinstance(config, ResponseGeometryAssayNativeConfig):
        expected_config_ids.add(config_id)
    by_task = {task.task_id: task for task in plan.tasks}
    for task in tasks:
        task_config_ids = {f"config-artifact.{config_id}"}
        if isinstance(config, ResponseGeometryAssayNativeConfig) and not task.dependency_task_ids:
            task_config_ids.add(config_id)
        config_inputs = tuple(
            value
            for value in task.external_inputs
            if value.expected_payload_schema == config.SCHEMA
        )
        if (
            len(config_inputs) != len(task_config_ids)
            or {value.logical_artifact_id for value in config_inputs} != task_config_ids
        ):
            raise ValueError("assay task configuration roles differ from its installed input contract")
        required_scan = len(config_inputs) * len(payload) + sum(
            by_task[dependency].capability.requested_resources.output_bytes
            for dependency in task.dependency_task_ids
        )
        if required_scan > task.capability.requested_resources.source_scan_bytes:
            raise ValueError("assay task input scan budget does not cover its complete input roster")
    specs = {
        item.logical_artifact_id: item
        for task in tasks
        for item in task.external_inputs
        if item.expected_payload_schema == config.SCHEMA
    }
    if (
        not specs
        or set(specs) != expected_config_ids
        or any(spec.expected_content_sha256 != config.fingerprint() for spec in specs.values())
    ):
        raise ValueError("assay provider lacks the exact external configuration")
    parent = ArtifactLineageParent(
        ObjectIdentity.from_record(config_id, config),
        VisibilityCeiling.PROSPECTIVE,
        OutcomeAccess.OUTCOME_BLIND,
    )
    return tuple(
        ExternalInputPayload.from_bytes(
            logical_artifact_id=key,
            payload_schema=config.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type=CANONICAL_MEDIA_TYPE,
            payload=payload,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
            lineage_parents=(parent,),
            logical_content_sha256=config.fingerprint(),
        )
        for key in sorted(specs)
    )


def semantic_contracts(
    manifest: CapabilityManifest,
    records: tuple[type[CanonicalRecord], ...],
    hdf5_schema: str | None,
) -> tuple[CapabilityOutputSemanticContract, ...]:
    contracts = [
        CapabilityOutputSemanticContract.from_manifest(
            manifest,
            payload_schema=record.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            top_level_keys=("schema", "value", "version"),
            value_keys=tuple(sorted(field.name for field in fields(cast(Any, record)))),
            record_version=record.VERSION,
        )
        for record in records
    ]
    if hdf5_schema is not None:
        contracts.append(
            CapabilityOutputSemanticContract.from_manifest(
                manifest, payload_schema=hdf5_schema, profile=ArtifactProfile.AUDITED_HDF5
            )
        )
    return tuple(sorted(contracts, key=lambda value: value.key))


class ResponseGeometryAssaySourceTask:
    def __init__(self, manifest: CapabilityManifest, config: ResponseGeometryAssayNativeConfig) -> None:
        self.manifest, self.config = manifest, config

    def segments(self) -> tuple[ResponseGeometryAssayNativeSegment, ...]:
        return assay_segments()

    def execute_native(
        self,
        segment: ResponseGeometryAssayNativeSegment,
        previous: ResponseGeometryAssayNativeSegmentResult | None,
        progress: Callable[[int], None] | None,
    ) -> tuple[ResponseGeometryAssayNativeSegmentResult, bytes]:
        return execute_assay_segment(self.config, segment, previous, progress=progress)

    def execute(self, context: TaskContext) -> RunnerResult:
        return self._execute(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        return self._execute(context, lambda completed: emitter.advance(Decimal(completed)))

    def validate_config_inputs(
        self, segment: ResponseGeometryAssayNativeSegment, ports: tuple[WorkerInputPort, ...]
    ) -> None:
        """Read the frozen control roles without performing native work."""
        expected = {f"config-artifact.{self.config.config_id}"}
        if segment.predecessor is None:
            expected.add(self.config.config_id)
        if (
            len(ports) != len(expected)
            or {port.artifact_id for port in ports} != expected
            or any(port.payload_schema != self.config.SCHEMA for port in ports)
        ):
            for port in ports:
                port.close()
            raise ValueError("assay requires its exact capability-config and MODEL-config inputs")
        try:
            for port in ports:
                if decode_port(port, type(self.config)) != self.config:
                    raise ValueError("assay source input config differs from the issued native config")
        finally:
            for port in ports:
                port.close()

    def _execute(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        segment = next(
            (value for value in self.segments() if value.task_id == context.task_id), None
        )
        if segment is None:
            raise ValueError("assay source task is outside its exact issued roster")
        by_schema: dict[str, list[WorkerInputPort]] = {}
        for port in context.input_ports:
            by_schema.setdefault(port.payload_schema, []).append(port)
        config_schema = self.config.SCHEMA
        result_type = self.config.result_type
        data_schema = self.config.observation_schema
        expected = {config_schema}
        if segment.predecessor is not None:
            expected.update((result_type.SCHEMA, data_schema, LinkedCampaignStageEnvelope.SCHEMA))
        if set(by_schema) != expected or any(
            len(values) != (2 if schema == config_schema and segment.predecessor is None else 1)
            for schema, values in by_schema.items()
        ):
            for port in context.input_ports:
                port.close()
            raise ValueError("assay source input roster differs from its sole predecessor/config")
        self.validate_config_inputs(segment, tuple(by_schema[config_schema]))
        previous = None
        if segment.predecessor is not None:
            previous = decode_port(by_schema[result_type.SCHEMA][0], result_type)
            upstream = decode_port(
                by_schema[LinkedCampaignStageEnvelope.SCHEMA][0], LinkedCampaignStageEnvelope
            )
            data = read_port(by_schema[data_schema][0], 16 * 1024**2)
            if (
                upstream != source_stage_envelope(previous)
                or sha256(data).hexdigest() != previous.observations_sha256
            ):
                raise ValueError("assay source predecessor custody differs")
        result, data = self.execute_native(segment, previous, progress)
        stage = source_stage_envelope(result)
        return output_result(
            context,
            {
                result.SCHEMA: result.canonical_bytes(),
                data_schema: data,
                stage.SCHEMA: stage.canonical_bytes(),
            },
            (
                "complete-interval-delivery-accounted",
                "same-root-refined-noise",
                "no-source-scientific-terminal",
            ),
        )


class ResponseGeometryAssaySourceProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: ResponseGeometryAssayNativeConfig,
        carrier: FreshSourceQualificationExperiment,
        association: FreshSourceQualificationSubstrateBinding,
        installed_binding: ObjectIdentity,
    ) -> None:
        if registry.resolve(
            manifest.capability_key, manifest.capability_version
        ) != manifest or carrier.source_capability != ObjectIdentity.from_record(
            manifest.capability_key, manifest
        ):
            raise ValueError("assay source capability differs from its registry/carrier")
        if (
            association.qualification_carrier
            != ObjectIdentity.from_record(carrier.extension_set_id, carrier)
            or carrier.source_config != ObjectIdentity.from_record(config.config_id, config)
            or association.source_config != carrier.source_config
            or association.substrate.installed_executable_binding != installed_binding
            or association.substrate.provider_key != manifest.capability_key
            or association.substrate.provider_implementation_sha256
            != manifest.implementation_sha256
        ):
            raise ValueError("assay native provider association differs")
        validate_response_geometry_assay_native_roster(config, carrier)
        self.registry, self.manifest, self.config = registry, manifest, config
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("assay source runner registry/records differ")
        return (cast(TaskRunner, ResponseGeometryAssaySourceTask(self.manifest, self.config)),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("assay source execution registry/records differ")
        return config_payloads(
            plan,
            self.manifest,
            self.config,
            self.config.config_id,
            {value.task_id for value in assay_segments()},
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("assay source semantic registry differs")
        return semantic_contracts(
            self.manifest, (ResponseGeometryAssayNativeSegmentResult, LinkedCampaignStageEnvelope), ASSAY_HDF5_SCHEMA
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("assay source adjudication registry differs")
        return None
