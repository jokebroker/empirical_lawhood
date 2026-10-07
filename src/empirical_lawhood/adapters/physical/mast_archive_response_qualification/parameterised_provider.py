"Issued acquisition, custody, projection and compact-manifest providers."

from __future__ import annotations

from dataclasses import dataclass, fields
from threading import Lock
from typing import ClassVar, Protocol, cast, runtime_checkable

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    StreamingOutputEmitter,
    TaskContext,
    TaskRunner,
    WorkerInputPort,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.response_experiment_binding import materialise_native_acquisition_request
from empirical_lawhood.runtime.response_experiment_ports import NativeAcquisitionRequest, ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .archive_receiver_projection import MastArchiveReceiverProjectionConfig, MastArchiveReceiverProjectionDisposition, MastArchiveReceiverProjectorPort, MastArchiveReceiverScientificProjection
from .query_source import MASTUArchiveAcquisitionDisposition, MASTUArchiveAcquisitionResult, MASTUArchiveAcquisitionUnitManifest, MASTUArchiveCompactManifestResult, MASTUArchiveCustodyPort, MASTUArchiveObjectReceipt, MASTUArchiveQueryClient, MASTUArchiveQueryManifest, MASTUArchiveQuerySourceConfig, MASTUArchiveQuery, reduce_mastu_archive_acquisition_results, retrieve_mastu_archive_query


MASTU_ARCHIVE_CANONICAL_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
MASTU_ARCHIVE_ACQUISITION_PROVIDER_KEY = 'mastu-archive.parameterised-acquisition'
MASTU_ARCHIVE_PROJECTION_PROVIDER_KEY = "mastu-archive.pf-view-projection"
MASTU_ARCHIVE_COMPACT_PROVIDER_KEY = "mastu-archive.compact-custody-reduction"
MASTU_ARCHIVE_PROVIDER_VERSION = "1.0.0"
_MAXIMUM_CANONICAL_INPUT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class MASTUArchiveSourceCustodyAuthority(CanonicalRecord):
    """Exact independently issued source and custody grants for one query manifest."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-source-custody-authority'

    authority_id: str
    subject: ObjectIdentity
    source_authority: StudyOperationAuthority
    custody_authority: StudyOperationAuthority
    grantee_id: str
    storage_root_id: str
    relative_root: str

    def __post_init__(self) -> None:
        validate_stable_id(self.authority_id, field_name="authority_id")
        validate_stable_id(self.grantee_id, field_name="grantee_id")
        if self.subject.object_schema != MASTUArchiveQueryManifest.SCHEMA:
            raise ValueError("MAST-U authorities bind another source subject")
        require_study_authority(
            self.source_authority,
            kind=StudyAuthorityKind.SOURCE_ACQUISITION,
            subject=self.subject,
            prerequisite_authority=None,
            grantee_id=self.grantee_id,
        )
        require_study_authority(
            self.custody_authority,
            kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
            subject=self.subject,
            prerequisite_authority=None,
            grantee_id=self.grantee_id,
            storage_root_id=self.storage_root_id,
            relative_root=self.relative_root,
        )


@dataclass(frozen=True, slots=True)
class MASTUArchiveParameterisedProviderConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-parameterised-provider-config'

    config_id: str
    experiment_extension_set: ObjectIdentity
    substrate_binding: ObjectIdentity
    query_manifest: ObjectIdentity
    query_source_config: ObjectIdentity
    authority: MASTUArchiveSourceCustodyAuthority
    causal_cutoff_id: str
    source_task_prefix: str
    grants_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.causal_cutoff_id, field_name="causal_cutoff_id")
        validate_stable_id(self.source_task_prefix, field_name="source_task_prefix")
        if self.experiment_extension_set.object_schema != ResponseExperimentExtensionSet.SCHEMA:
            raise ValueError("MAST-U provider config binds another extension set")
        if self.substrate_binding.object_schema != ResponseSubstrateBinding.SCHEMA:
            raise ValueError("MAST-U provider config binds another substrate")
        if self.query_manifest.object_schema != MASTUArchiveQueryManifest.SCHEMA:
            raise ValueError("MAST-U provider config binds another query manifest")
        if self.query_source_config.object_schema != MASTUArchiveQuerySourceConfig.SCHEMA:
            raise ValueError("MAST-U provider config binds another query-source config")
        if self.authority.subject != self.query_manifest:
            raise ValueError("MAST-U provider authority binds another query manifest")
        if self.grants_authority or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("MAST-U provider config cannot grant authority")


@dataclass(frozen=True, slots=True)
class MASTUArchiveCompactProviderConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-compact-provider-config'

    config_id: str
    query_manifest: MASTUArchiveQueryManifest
    task_id: str
    grants_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.task_id, field_name="task_id")
        if self.grants_authority or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("MAST-U compact provider config cannot grant authority")


@runtime_checkable
class MASTUArchiveAcquisitionSessionPort(Protocol):
    """One serialized, cumulative-bound session; implementations never retry."""

    def acquire(
        self,
        query: MASTUArchiveQuery,
    ) -> tuple[MASTUArchiveObjectReceipt, MASTUArchiveAcquisitionUnitManifest]: ...


class MASTUArchiveAcquisitionSession:
    """Default synchronized session over injected UDA and external-custody ports."""

    def __init__(
        self,
        *,
        manifest: MASTUArchiveQueryManifest,
        config: MASTUArchiveQuerySourceConfig,
        client: MASTUArchiveQueryClient,
        custody: MASTUArchiveCustodyPort,
        prior_units: tuple[MASTUArchiveAcquisitionUnitManifest, ...] = (),
    ) -> None:
        self._manifest = manifest
        self._config = config
        self._client = client
        self._custody = custody
        self._lock = Lock()
        self._attempted = {value.query.object_id for value in prior_units}
        if len(self._attempted) != len(prior_units):
            raise ValueError("MAST-U recovery supplies duplicate prior acquisition units")
        manifest_query_ids = {value.query_id for value in manifest.queries}
        if not self._attempted <= manifest_query_ids:
            raise ValueError("MAST-U recovery supplies an acquisition unit outside the manifest")
        manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
        if any(value.query_manifest != manifest_identity for value in prior_units):
            raise ValueError("MAST-U recovery substitutes its frozen query manifest")
        self._total_response_bytes = sum(
            value.observed_total_response_bytes for value in prior_units
        )
        if self._total_response_bytes > manifest.maximum_total_response_bytes:
            raise ValueError("MAST-U prior custody already exceeds the response bound")

    def acquire(
        self,
        query: MASTUArchiveQuery,
    ) -> tuple[MASTUArchiveObjectReceipt, MASTUArchiveAcquisitionUnitManifest]:
        with self._lock:
            if query.query_id in self._attempted:
                raise RuntimeError("MASTU_QUERY_ALREADY_ATTEMPTED_NO_REACQUISITION")
            self._attempted.add(query.query_id)
            remaining = self._manifest.maximum_total_response_bytes - self._total_response_bytes
            receipt, unit = retrieve_mastu_archive_query(
                manifest=self._manifest,
                config=self._config,
                query=query,
                client=self._client,
                custody=self._custody,
                unit_manifest_id=f"unit-manifest.{query.query_id}",
                maximum_remaining_response_bytes=remaining,
            )
            self._total_response_bytes += unit.observed_total_response_bytes
            return receipt, unit


class _MASTUArchiveAcquisitionTask:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        extension_set: ResponseExperimentExtensionSet,
        substrate_binding: ResponseSubstrateBinding,
        provider_config: MASTUArchiveParameterisedProviderConfig,
        query_manifest: MASTUArchiveQueryManifest,
        session: MASTUArchiveAcquisitionSessionPort,
    ) -> None:
        self.manifest = manifest
        self._extension_set = extension_set
        self._substrate_binding = substrate_binding
        self._provider_config = provider_config
        self._queries = {value.acquisition_group_id: value for value in query_manifest.queries}
        self._session = session

    def execute_streaming(
        self,
        context: TaskContext,
        emitter: StreamingOutputEmitter,
    ) -> tuple[ReceiptCheck, ...]:
        decoded_config = _decode_one(context, MASTUArchiveParameterisedProviderConfig)
        if decoded_config != self._provider_config:
            raise ValueError("MAST-U task received a substituted provider config")
        prefix = f"{self._provider_config.source_task_prefix}."
        if not context.task_id.startswith(prefix):
            raise ValueError("MAST-U acquisition task identity differs from its issued prefix")
        group_id = context.task_id[len(prefix) :]
        query = self._queries.get(group_id)
        identification_config = self._extension_set.identification_config
        groups = {value.acquisition_group_id: value for value in identification_config.acquisition_groups}
        units = {value.physical_independent_unit_id: value for value in identification_config.physical_units}
        if query is None or group_id not in groups:
            raise ValueError("MAST-U acquisition task lies outside the frozen query roster")
        group = groups[group_id]
        request = materialise_native_acquisition_request(
            request_id=f"request.mastu.{group_id}",
            extension_set=self._extension_set,
            binding=self._substrate_binding,
            installed_executable_binding=self._substrate_binding.installed_executable_binding,
            stage_id='stage.measurement-source-materialization',
            causal_cutoff_id=self._provider_config.causal_cutoff_id,
            physical_unit=units[group.physical_independent_unit_id],
            acquisition_group=group,
        )
        if query is None or (
            query.preparation_instance_id != request.preparation_instance_id
            or query.physical_independent_unit_id != request.physical_independent_unit_id
        ):
            raise ValueError("MAST-U request changes frozen query lineage")
        request_identity = ObjectIdentity.from_record(request.request_id, request)
        try:
            receipt, unit = self._session.acquire(query)
        except (ValueError, RuntimeError) as error:
            result = MASTUArchiveAcquisitionResult(
                result_id=f"acquisition-result.{query.query_id}",
                request=request_identity,
                query=ObjectIdentity.from_record(query.query_id, query),
                disposition=MASTUArchiveAcquisitionDisposition.INVALID_RESPONSE,
                object_receipt=None,
                unit_manifest=None,
                reason_codes=(type(error).__name__.upper(),),
                source_contacted=True,
                object_custodied=False,
                scientific_verdict_constructed=False,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
            )
        except Exception as error:  # provider failures are retained, never retried here
            result = MASTUArchiveAcquisitionResult(
                result_id=f"acquisition-result.{query.query_id}",
                request=request_identity,
                query=ObjectIdentity.from_record(query.query_id, query),
                disposition=MASTUArchiveAcquisitionDisposition.PROVIDER_ERROR,
                object_receipt=None,
                unit_manifest=None,
                reason_codes=(type(error).__name__.upper(),),
                source_contacted=True,
                object_custodied=False,
                scientific_verdict_constructed=False,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
            )
        else:
            result = MASTUArchiveAcquisitionResult(
                result_id=f"acquisition-result.{query.query_id}",
                request=request_identity,
                query=ObjectIdentity.from_record(query.query_id, query),
                disposition=MASTUArchiveAcquisitionDisposition.COMPLETE,
                object_receipt=receipt,
                unit_manifest=unit,
                reason_codes=(),
                source_contacted=True,
                object_custodied=True,
                scientific_verdict_constructed=False,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
            )
        envelope = _stage_envelope(context.task_id, result)
        _emit_records(context, emitter, (result, envelope))
        return (
            ReceiptCheck("one-query-attempt", True, ()),
            ReceiptCheck("no-adapter-scientific-verdict", True, ()),
            ReceiptCheck(
                "receipt-before-result" if result.object_custodied else "typed-source-stop",
                True,
                (),
            ),
        )


class _MASTUPFProjectionTask:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: MastArchiveReceiverProjectionConfig,
        projector: MastArchiveReceiverProjectorPort,
    ) -> None:
        self.manifest = manifest
        self._config = config
        self._projector = projector

    def execute_streaming(
        self,
        context: TaskContext,
        emitter: StreamingOutputEmitter,
    ) -> tuple[ReceiptCheck, ...]:
        records = _decode_inputs(
            context,
            (
                MastArchiveReceiverProjectionConfig,
                MASTUArchiveAcquisitionResult,
                LinkedCampaignStageEnvelope,
            ),
        )
        decoded_config = cast(
            MastArchiveReceiverProjectionConfig,
            records[MastArchiveReceiverProjectionConfig.SCHEMA],
        )
        if decoded_config != self._config:
            raise ValueError("MAST-U projection task received a substituted config")
        prefix = f"{self._config.projection_task_prefix}."
        if not context.task_id.startswith(prefix):
            raise ValueError("MAST-U projection task identity differs from its issued prefix")
        view = self._config.view(context.task_id[len(prefix) :])
        acquisition = cast(
            MASTUArchiveAcquisitionResult, records[MASTUArchiveAcquisitionResult.SCHEMA]
        )
        query_id = acquisition.query.object_id
        if acquisition.disposition is not MASTUArchiveAcquisitionDisposition.COMPLETE:
            disposition = MastArchiveReceiverProjectionDisposition.INVALID
            artifact = None
            reasons = tuple(sorted({*acquisition.reason_codes, "UPSTREAM_ACQUISITION_INCOMPLETE"}))
            receipt_identity = None
            unit_identity = None
        else:
            assert acquisition.object_receipt is not None
            assert acquisition.unit_manifest is not None
            if (
                acquisition.unit_manifest.acquisition_group_id != view.acquisition_group_id
                or acquisition.unit_manifest.physical_independent_unit_id
                != view.physical_independent_unit_id
                or view.view_id not in acquisition.unit_manifest.view_ids
            ):
                raise ValueError("MAST-U projection view changes acquisition lineage")
            receipt_identity = ObjectIdentity.from_record(
                acquisition.object_receipt.receipt_id,
                acquisition.object_receipt,
            )
            unit_identity = ObjectIdentity.from_record(
                acquisition.unit_manifest.manifest_id,
                acquisition.unit_manifest,
            )
            try:
                artifact = self._projector.project(
                    receipt=acquisition.object_receipt,
                    view=view,
                )
            except Exception as error:
                disposition = MastArchiveReceiverProjectionDisposition.PROVIDER_ERROR
                artifact = None
                reasons = (type(error).__name__.upper(),)
                receipt_identity = None
                unit_identity = None
            else:
                disposition = MastArchiveReceiverProjectionDisposition.COMPLETE
                reasons = ()
        projection = MastArchiveReceiverScientificProjection(
            projection_id=f"pf-projection.{view.view_id}.{query_id}",
            view=ObjectIdentity.from_record(view.view_id, view),
            object_receipt=receipt_identity,
            acquisition_unit_manifest=unit_identity,
            projected_artifact=artifact,
            receiver_ids=view.receiver_ids,
            clock_ids=view.clock_ids,
            disposition=disposition,
            reason_codes=reasons,
            preserves_native_units_frames_directions=True,
            preserves_causal_cutoff=True,
            scientific_verdict_constructed=False,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )
        envelope = _stage_envelope(context.task_id, projection)
        _emit_records(context, emitter, (projection, envelope))
        return (
            ReceiptCheck("custody-only-projection", True, ()),
            ReceiptCheck("one-view-no-reacquisition", True, ()),
        )


class _MASTUArchiveCompactTask:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        config: MASTUArchiveCompactProviderConfig,
    ) -> None:
        self.manifest = manifest
        self._config = config

    def execute_streaming(
        self,
        context: TaskContext,
        emitter: StreamingOutputEmitter,
    ) -> tuple[ReceiptCheck, ...]:
        decoded = _decode_all_by_schema(context)
        config_values = decoded.get(MASTUArchiveCompactProviderConfig.SCHEMA, ())
        acquisitions = decoded.get(MASTUArchiveAcquisitionResult.SCHEMA, ())
        if len(config_values) != 1 or config_values[0] != self._config or not acquisitions:
            raise ValueError("MAST-U compact task lacks manifest/acquisition terminals")
        if context.task_id != self._config.task_id:
            raise ValueError("MAST-U compact task identity differs from its config")
        manifest = self._config.query_manifest
        result = reduce_mastu_archive_acquisition_results(
            manifest=manifest,
            results=cast(tuple[MASTUArchiveAcquisitionResult, ...], acquisitions),
            result_id=f"compact-result.{manifest.manifest_id}",
            compact_manifest_id=f"compact-manifest.{manifest.manifest_id}",
        )
        envelope = _stage_envelope(context.task_id, result)
        _emit_records(context, emitter, (result, envelope))
        return (
            ReceiptCheck("all-query-terminals-reduced", True, ()),
            ReceiptCheck(
                "compact-after-custody", result.compact_manifest is not None, result.reason_codes
            ),
        )


def _read_port(port: WorkerInputPort) -> bytes:
    size = port.size_bytes
    if size > _MAXIMUM_CANONICAL_INPUT_BYTES:
        raise ValueError("MAST-U canonical task input exceeds its bound")
    try:
        payload = port.read(size + 1)
        if len(payload) != size or port.read(1):
            raise ValueError("MAST-U canonical task input size differs")
        return payload
    finally:
        port.close()


def _decode_one(context: TaskContext, record_type: type[CanonicalRecord]) -> CanonicalRecord:
    matches = tuple(
        value for value in context.input_ports if value.payload_schema == record_type.SCHEMA
    )
    if len(matches) != 1:
        raise ValueError("MAST-U task requires exactly one typed config input")
    for value in context.input_ports:
        if value is not matches[0]:
            value.close()
    return decode_canonical_bytes(
        _read_port(matches[0]),
        record_type,
        maximum_bytes=_MAXIMUM_CANONICAL_INPUT_BYTES,
    )


def _decode_inputs(
    context: TaskContext,
    record_types: tuple[type[CanonicalRecord], ...],
) -> dict[str, CanonicalRecord]:
    by_schema = {value.SCHEMA: value for value in record_types}
    decoded: dict[str, CanonicalRecord] = {}
    for port in context.input_ports:
        record_type = by_schema.get(port.payload_schema)
        if record_type is None:
            port.close()
            continue
        value = decode_canonical_bytes(
            _read_port(port),
            record_type,
            maximum_bytes=_MAXIMUM_CANONICAL_INPUT_BYTES,
        )
        if value.SCHEMA in decoded:
            raise ValueError("MAST-U task received duplicate typed input")
        decoded[value.SCHEMA] = value
    if set(decoded) != set(by_schema):
        raise ValueError("MAST-U task input roster differs")
    return decoded


def _decode_all_by_schema(context: TaskContext) -> dict[str, tuple[CanonicalRecord, ...]]:
    types = {
        MASTUArchiveCompactProviderConfig.SCHEMA: MASTUArchiveCompactProviderConfig,
        MASTUArchiveAcquisitionResult.SCHEMA: MASTUArchiveAcquisitionResult,
        LinkedCampaignStageEnvelope.SCHEMA: LinkedCampaignStageEnvelope,
    }
    values: dict[str, list[CanonicalRecord]] = {}
    for port in context.input_ports:
        record_type = types.get(port.payload_schema)
        if record_type is None:
            port.close()
            continue
        values.setdefault(port.payload_schema, []).append(
            decode_canonical_bytes(
                _read_port(port),
                record_type,
                maximum_bytes=_MAXIMUM_CANONICAL_INPUT_BYTES,
            )
        )
    return {key: tuple(value) for key, value in values.items()}


def _stage_envelope(task_id: str, value: CanonicalRecord) -> LinkedCampaignStageEnvelope:
    record_id = next(
        getattr(value, name) for name in ("result_id", "projection_id") if hasattr(value, name)
    )
    reason_codes = cast(tuple[str, ...], getattr(value, "reason_codes"))
    return LinkedCampaignStageEnvelope(
        envelope_id=f"stage-envelope.{task_id}",
        role=(
            LinkedCampaignStageRole.SOURCE_MATERIALIZATION
            if isinstance(value, MASTUArchiveAcquisitionResult)
            else LinkedCampaignStageRole.EVIDENCE_PROJECTION
        ),
        disposition=(
            LinkedCampaignDisposition.SUPPORTED
            if not reason_codes
            else LinkedCampaignDisposition.UNEVALUABLE
        ),
        scientific_product=ObjectIdentity.from_record(record_id, value),
        reason_codes=reason_codes,
    )


def _emit_records(
    context: TaskContext,
    emitter: StreamingOutputEmitter,
    records: tuple[CanonicalRecord, ...],
) -> None:
    by_schema = {value.SCHEMA: value for value in records}
    if set(by_schema) != {value.payload_schema for value in context.output_ports}:
        raise ValueError("MAST-U task output contract differs")
    for output in context.output_ports:
        if (
            output.profile is not ArtifactProfile.CANONICAL_JSON
            or output.payload_schema not in by_schema
        ):
            raise ValueError("MAST-U output port differs")
        emitter.write(output.output_id, by_schema[output.payload_schema].canonical_bytes())


def _canonical_payload(record: CanonicalRecord, logical_artifact_id: str) -> ExternalInputPayload:
    parent_id = next(
        getattr(record, name)
        for name in (
            "request_id",
            "view_id",
            "manifest_id",
            "config_id",
        )
        if hasattr(record, name)
    )
    parent = ArtifactLineageParent(
        identity=ObjectIdentity.from_record(parent_id, record),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExternalInputPayload.from_bytes(
        logical_artifact_id=logical_artifact_id,
        payload_schema=record.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type=MASTU_ARCHIVE_CANONICAL_MEDIA_TYPE,
        payload=record.canonical_bytes(),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(parent.visibility_ceiling,),
        lineage_parents=(parent,),
        logical_content_sha256=record.fingerprint(),
    )


class _OneRunnerProvider(CampaignRuntimeProvider):
    def __init__(self, registry: CapabilityRegistry, manifest: CapabilityManifest) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("MAST-U provider manifest differs from installed registry")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.manifest = manifest
        self.capability_count = 1

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("MAST-U semantic registry differs")
        del execution_plan
        types = {
            MASTUArchiveAcquisitionResult.SCHEMA: MASTUArchiveAcquisitionResult,
            MastArchiveReceiverScientificProjection.SCHEMA: MastArchiveReceiverScientificProjection,
            MASTUArchiveCompactManifestResult.SCHEMA: MASTUArchiveCompactManifestResult,
            LinkedCampaignStageEnvelope.SCHEMA: LinkedCampaignStageEnvelope,
        }
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(value.name for value in fields(types[schema]))),
            )
            for schema in self.manifest.output_schema_ids
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("MAST-U adjudication registry differs")
        del execution_plan
        return None


class MASTUArchiveAcquisitionProvider(_OneRunnerProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        extension_set: ResponseExperimentExtensionSet,
        substrate_binding: ResponseSubstrateBinding,
        query_manifest: MASTUArchiveQueryManifest,
        query_config: MASTUArchiveQuerySourceConfig,
        provider_config: MASTUArchiveParameterisedProviderConfig,
        session: MASTUArchiveAcquisitionSessionPort,
    ) -> None:
        super().__init__(registry, manifest)
        self.extension_set = extension_set
        self.substrate_binding = substrate_binding
        self.query_manifest = query_manifest
        self.query_config = query_config
        self.provider_config = provider_config
        self._session = session
        _validate_source_records(self)

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("MAST-U acquisition runner inputs differ")
        return (
            cast(
                TaskRunner,
                _MASTUArchiveAcquisitionTask(
                    manifest=self.manifest,
                    extension_set=self.extension_set,
                    substrate_binding=self.substrate_binding,
                    provider_config=self.provider_config,
                    query_manifest=self.query_manifest,
                    session=self._session,
                ),
            ),
        )

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("MAST-U acquisition plan/source inputs differ")
        tasks = tuple(
            task
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
        )
        expected_task_ids = {
            f"{self.provider_config.source_task_prefix}.{value.acquisition_group_id}"
            for value in self.query_manifest.queries
        }
        if {value.task_id for value in tasks} != expected_task_ids:
            raise ValueError("MAST-U acquisition task cardinality differs from queries")
        specs = {
            value.logical_artifact_id: value
            for task in tasks
            for value in task.external_inputs
            if value.expected_payload_schema == MASTUArchiveParameterisedProviderConfig.SCHEMA
        }
        if (
            len(specs) != 1
            or next(iter(specs.values())).expected_content_sha256
            != self.provider_config.fingerprint()
        ):
            raise ValueError("MAST-U acquisition tasks lack the exact provider config")
        artifact_id = next(iter(specs))
        return (_canonical_payload(self.provider_config, artifact_id),)


class MastArchiveReceiverProjectionProvider(_OneRunnerProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: MastArchiveReceiverProjectionConfig,
        projector: MastArchiveReceiverProjectorPort,
    ) -> None:
        super().__init__(registry, manifest)
        self.config = config
        self._projector = projector

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("MAST-U projection runner inputs differ")
        return (
            cast(
                TaskRunner,
                _MASTUPFProjectionTask(
                    manifest=self.manifest,
                    config=self.config,
                    projector=self._projector,
                ),
            ),
        )

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("MAST-U projection plan/source inputs differ")
        tasks = tuple(
            task
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
        )
        expected_task_ids = {
            f"{self.config.projection_task_prefix}.{value.view_id}" for value in self.config.views
        }
        if {value.task_id for value in tasks} != expected_task_ids:
            raise ValueError("MAST-U projection task cardinality differs from views")
        specs = {
            value.logical_artifact_id: value
            for task in tasks
            for value in task.external_inputs
            if value.expected_payload_schema == MastArchiveReceiverProjectionConfig.SCHEMA
        }
        if (
            len(specs) != 1
            or next(iter(specs.values())).expected_content_sha256 != self.config.fingerprint()
        ):
            raise ValueError("MAST-U projection tasks lack the exact projection config")
        artifact_id = next(iter(specs))
        return (_canonical_payload(self.config, artifact_id),)


class MASTUArchiveCompactProvider(_OneRunnerProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: MASTUArchiveCompactProviderConfig,
    ) -> None:
        super().__init__(registry, manifest)
        self.config = config

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("MAST-U compact runner inputs differ")
        return (
            cast(
                TaskRunner,
                _MASTUArchiveCompactTask(
                    manifest=self.manifest,
                    config=self.config,
                ),
            ),
        )

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("MAST-U compact plan/source inputs differ")
        tasks = tuple(
            task
            for task in plan.tasks
            if task.capability.capability_key == self.manifest.capability_key
        )
        if len(tasks) != 1 or tasks[0].task_id != self.config.task_id:
            raise ValueError("MAST-U compact plan changes its exact task")
        specs = [
            value
            for value in tasks[0].external_inputs
            if value.expected_payload_schema == MASTUArchiveCompactProviderConfig.SCHEMA
        ]
        if len(specs) != 1 or specs[0].expected_content_sha256 != self.config.fingerprint():
            raise ValueError("MAST-U compact task lacks its exact query manifest config")
        return (_canonical_payload(self.config, specs[0].logical_artifact_id),)


def _validate_source_records(provider: MASTUArchiveAcquisitionProvider) -> None:
    manifest_identity = ObjectIdentity.from_record(
        provider.query_manifest.manifest_id, provider.query_manifest
    )
    extension_identity = ObjectIdentity.from_record(
        provider.extension_set.extension_set_id, provider.extension_set
    )
    binding_identity = ObjectIdentity.from_record(
        provider.substrate_binding.binding_id, provider.substrate_binding
    )
    config_identity = ObjectIdentity.from_record(
        provider.query_config.config_id, provider.query_config
    )
    if (
        provider.provider_config.experiment_extension_set != extension_identity
        or provider.provider_config.substrate_binding != binding_identity
        or provider.provider_config.query_manifest != manifest_identity
        or provider.provider_config.query_source_config != config_identity
        or provider.query_config.query_manifest != manifest_identity
        or provider.substrate_binding.native_config != config_identity
        or provider.substrate_binding.native_request_schema != NativeAcquisitionRequest.SCHEMA
        or provider.substrate_binding.native_object_schema
        != provider.query_manifest.response_object_schema
        or provider.substrate_binding.action_contract is not None
    ):
        raise ValueError("MAST-U acquisition provider changes issued archive semantics")
    queries = {value.acquisition_group_id: value for value in provider.query_manifest.queries}
    groups = {
        value.acquisition_group_id: value
        for value in provider.extension_set.identification_config.acquisition_groups
    }
    if set(queries) != set(groups) or any(
        queries[key].physical_independent_unit_id != groups[key].physical_independent_unit_id
        or queries[key].preparation_instance_id != groups[key].preparation_instance_id
        or queries[key].view_ids != groups[key].view_ids
        for key in queries
    ):
        raise ValueError("MAST-U query roster differs from neutral acquisition groups")


__all__ = [
    "MASTU_ARCHIVE_ACQUISITION_PROVIDER_KEY",
    "MASTU_ARCHIVE_CANONICAL_MEDIA_TYPE",
    "MASTU_ARCHIVE_COMPACT_PROVIDER_KEY",
    "MASTU_ARCHIVE_PROJECTION_PROVIDER_KEY",
    "MASTU_ARCHIVE_PROVIDER_VERSION",
    'MASTUArchiveAcquisitionProvider',
    'MASTUArchiveAcquisitionSessionPort',
    'MASTUArchiveAcquisitionSession',
    'MASTUArchiveCompactProvider',
    'MASTUArchiveCompactProviderConfig',
    'MASTUArchiveParameterisedProviderConfig',
    'MASTUArchiveSourceCustodyAuthority',
    'MastArchiveReceiverProjectionProvider',
]
