"""Generated parameterised provider over the existing FreeGSNKE executor seam."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from typing import ClassVar, Protocol, cast, runtime_checkable

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.response_experiment_binding import materialise_native_execution_request
from empirical_lawhood.runtime.response_experiment_ports import NativeExecutionRequest, ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .contracts import FreeGsnkeProcessRequest, FreeGsnkeSavedPreparation, FreeGsnkeSourceBinding, FreeGsnkeTargetEpisodeStatus, FreeGsnkeTargetProcessResponse, FreeGsnkeTargetWorkerInput


FREEGSNKE_PARAMETERISED_PROVIDER_KEY = "freegsnke.parameterised-response-acquisition"
FREEGSNKE_PARAMETERISED_PROJECTION_KEY = "freegsnke.parameterised-view-projection"
FREEGSNKE_PARAMETERISED_VERSION = "1.0.0"
FREEGSNKE_PARAMETERISED_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
_MAXIMUM_CONFIG_BYTES = 64 * 1024 * 1024


class FreeGsnkeParameterisedDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    INVALID = "INVALID"
    NUMERICAL_FAILURE = "NUMERICAL_FAILURE"
    PROVIDER_ERROR = "PROVIDER_ERROR"


@dataclass(frozen=True, slots=True)
class FreeGsnkeParameterisedAcquisitionBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-parameterised-acquisition-binding'

    binding_id: str
    acquisition_group_id: str
    preparation_instance_id: str
    physical_independent_unit_id: str
    view_ids: tuple[str, ...]
    action_word_id: str
    numerical_member_id: str
    native_request: FreeGsnkeProcessRequest
    saved_preparation: FreeGsnkeSavedPreparation

    def __post_init__(self) -> None:
        for name in (
            "binding_id",
            "acquisition_group_id",
            "preparation_instance_id",
            "physical_independent_unit_id",
            "action_word_id",
            "numerical_member_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.view_ids, field_name="view_ids", allow_empty=False)
        if self.native_request.preparation.preparation_id != self.preparation_instance_id:
            raise ValueError("FreeGSNKE request changes preparation-instance identity")
        # Constructing the exact worker input is the shared, scientific join:
        # it proves that the excluded qualification state, request recipe,
        # source, numerical view and action chart are identical before issue.
        FreeGsnkeTargetWorkerInput(
            input_id=f"worker-input.{self.binding_id}",
            request=self.native_request,
            saved_preparation=self.saved_preparation,
        )


@dataclass(frozen=True, slots=True)
class FreeGsnkeParameterisedProviderConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-parameterised-provider-config'

    config_id: str
    experiment_extension_set: ObjectIdentity
    substrate_binding: ObjectIdentity
    source_binding: FreeGsnkeSourceBinding
    acquisition_bindings: tuple[FreeGsnkeParameterisedAcquisitionBinding, ...]
    source_task_prefix: str
    causal_cutoff_id: str
    maximum_episode_bytes: int
    grants_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.source_task_prefix, field_name="source_task_prefix")
        validate_stable_id(self.causal_cutoff_id, field_name="causal_cutoff_id")
        if self.experiment_extension_set.object_schema != ResponseExperimentExtensionSet.SCHEMA:
            raise ValueError("FreeGSNKE provider config binds another extension set")
        if self.substrate_binding.object_schema != ResponseSubstrateBinding.SCHEMA:
            raise ValueError("FreeGSNKE provider config binds another substrate")
        require_sorted_unique_ids(
            self.acquisition_bindings,
            attribute="acquisition_group_id",
            field_name="acquisition_bindings",
        )
        if not self.acquisition_bindings:
            raise ValueError("FreeGSNKE provider config requires acquisition bindings")
        if any(
            value.native_request.source_binding != self.source_binding
            for value in self.acquisition_bindings
        ):
            raise ValueError("FreeGSNKE acquisition requests change the source binding")
        if not 0 < self.maximum_episode_bytes <= 2 * 1024**3:
            raise ValueError("FreeGSNKE episode bound is outside 2 GiB")
        if self.grants_authority or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("FreeGSNKE provider config cannot grant authority")

    def acquisition(self, group_id: str) -> FreeGsnkeParameterisedAcquisitionBinding:
        try:
            return next(
                value
                for value in self.acquisition_bindings
                if value.acquisition_group_id == group_id
            )
        except StopIteration as error:
            raise KeyError(group_id) from error


@dataclass(frozen=True, slots=True)
class FreeGsnkeParameterisedEpisodeResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-parameterised-episode-result'

    result_id: str
    neutral_request: ObjectIdentity
    native_request: ObjectIdentity
    native_response: FreeGsnkeTargetProcessResponse | None
    disposition: FreeGsnkeParameterisedDisposition
    reason_codes: tuple[str, ...]
    requested_accepted_applied_realized_preserved: bool
    numerical_validity_preserved: bool
    scientific_verdict_constructed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.neutral_request.object_schema != NativeExecutionRequest.SCHEMA:
            raise ValueError("FreeGSNKE result binds another neutral request")
        if self.native_request.object_schema != FreeGsnkeProcessRequest.SCHEMA:
            raise ValueError("FreeGSNKE result binds another native request")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        complete = self.disposition is FreeGsnkeParameterisedDisposition.COMPLETE
        provider_error = self.disposition is FreeGsnkeParameterisedDisposition.PROVIDER_ERROR
        if complete == bool(self.reason_codes) or provider_error == (
            self.native_response is not None
        ):
            raise ValueError("FreeGSNKE episode disposition/response/reasons disagree")
        if self.native_response is not None and self.native_response.request != self.native_request:
            raise ValueError("FreeGSNKE native response binds another request")
        if (
            not self.requested_accepted_applied_realized_preserved
            or not self.numerical_validity_preserved
            or self.scientific_verdict_constructed
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("FreeGSNKE adapter loses semantics or exceeds authority")


@dataclass(frozen=True, slots=True)
class FreeGsnkeParameterisedProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-parameterised-projection-config'

    config_id: str
    parameterised_provider_config: ObjectIdentity
    projection_task_prefix: str
    view_ids: tuple[str, ...]
    receiver_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    grants_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.projection_task_prefix, field_name="projection_task_prefix")
        if (
            self.parameterised_provider_config.object_schema
            != FreeGsnkeParameterisedProviderConfig.SCHEMA
        ):
            raise ValueError("FreeGSNKE projection binds another provider config")
        for name in ("view_ids", "receiver_ids", "clock_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        if self.grants_authority or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("FreeGSNKE projection config cannot grant authority")


@dataclass(frozen=True, slots=True)
class FreeGsnkeParameterisedProjection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-parameterised-projection'

    projection_id: str
    view_id: str
    episode_result: ObjectIdentity
    receiver_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    disposition: FreeGsnkeParameterisedDisposition
    reason_codes: tuple[str, ...]
    scientific_verdict_constructed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.projection_id, field_name="projection_id")
        validate_stable_id(self.view_id, field_name="view_id")
        if self.episode_result.object_schema != FreeGsnkeParameterisedEpisodeResult.SCHEMA:
            raise ValueError("FreeGSNKE projection binds another episode result")
        require_sorted_unique_strings(
            self.receiver_ids, field_name="receiver_ids", allow_empty=False
        )
        require_sorted_unique_strings(self.clock_ids, field_name="clock_ids", allow_empty=False)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (self.disposition is FreeGsnkeParameterisedDisposition.COMPLETE) == bool(
            self.reason_codes
        ):
            raise ValueError("FreeGSNKE projection disposition/reasons disagree")
        if (
            self.scientific_verdict_constructed
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("FreeGSNKE projection exceeds adapter authority")


@runtime_checkable
class FreeGsnkeParameterisedExecutorPort(Protocol):
    def execute(
        self,
        request: FreeGsnkeProcessRequest,
        saved_preparation: FreeGsnkeSavedPreparation,
    ) -> FreeGsnkeTargetProcessResponse: ...


def validate_freegsnke_parameterised_config(
    *,
    config: FreeGsnkeParameterisedProviderConfig,
    extension_set: ResponseExperimentExtensionSet,
    substrate_binding: ResponseSubstrateBinding,
) -> None:
    if config.experiment_extension_set != ObjectIdentity.from_record(
        extension_set.extension_set_id, extension_set
    ) or config.substrate_binding != ObjectIdentity.from_record(
        substrate_binding.binding_id, substrate_binding
    ):
        raise ValueError("FreeGSNKE config changes neutral parents")
    identification_config = extension_set.identification_config
    groups = {value.acquisition_group_id: value for value in identification_config.acquisition_groups}
    views = {value.view_id: value for value in identification_config.nested_views}
    if set(groups) != {value.acquisition_group_id for value in config.acquisition_bindings}:
        raise ValueError("FreeGSNKE acquisition roster differs from neutral groups")
    for binding in config.acquisition_bindings:
        group = groups[binding.acquisition_group_id]
        request_keys = {
            (views[view_id].action_word_id, views[view_id].numerical_member_id)
            for view_id in group.view_ids
        }
        if (
            group.preparation_instance_id != binding.preparation_instance_id
            or group.physical_independent_unit_id != binding.physical_independent_unit_id
            or group.view_ids != binding.view_ids
            or request_keys != {(binding.action_word_id, binding.numerical_member_id)}
        ):
            raise ValueError("FreeGSNKE acquisition binding changes group request identity")
    if (
        substrate_binding.native_request_schema != FreeGsnkeProcessRequest.SCHEMA
        or substrate_binding.native_episode_schema != FreeGsnkeTargetProcessResponse.SCHEMA
        or substrate_binding.native_projection_schema != FreeGsnkeParameterisedProjection.SCHEMA
        or substrate_binding.action_contract is None
    ):
        raise ValueError("FreeGSNKE substrate binding changes native semantics")


class _FreeGsnkeAcquisitionRunner:
    def __init__(
        self,
        *,
        manifest: CapabilityManifest,
        extension_set: ResponseExperimentExtensionSet,
        binding: ResponseSubstrateBinding,
        config: FreeGsnkeParameterisedProviderConfig,
        executor: FreeGsnkeParameterisedExecutorPort,
    ) -> None:
        self.manifest = manifest
        self._extension = extension_set
        self._binding = binding
        self._config = config
        self._executor = executor

    def execute(self, context: TaskContext) -> RunnerResult:
        config = _decode_exact_config(context, FreeGsnkeParameterisedProviderConfig)
        if config != self._config:
            raise ValueError("FreeGSNKE task received a substituted config")
        prefix = f"{config.source_task_prefix}."
        if not context.task_id.startswith(prefix):
            raise ValueError("FreeGSNKE acquisition task identity differs")
        group_id = context.task_id[len(prefix) :]
        adapter_binding = config.acquisition(group_id)
        identification_config = self._extension.identification_config
        group = next(
            value for value in identification_config.acquisition_groups if value.acquisition_group_id == group_id
        )
        unit = next(
            value
            for value in identification_config.physical_units
            if value.physical_independent_unit_id == group.physical_independent_unit_id
        )
        view = next(value for value in identification_config.nested_views if value.view_id == group.view_ids[0])
        neutral = materialise_native_execution_request(
            request_id=f"neutral-request.freegsnke.{group_id}",
            extension_set=self._extension,
            binding=self._binding,
            installed_executable_binding=self._binding.installed_executable_binding,
            stage_id='stage.measurement-source-materialization',
            causal_cutoff_id=config.causal_cutoff_id,
            physical_unit=unit,
            acquisition_group=group,
            nested_view=view,
        )
        native_identity = ObjectIdentity.from_record(
            adapter_binding.native_request.request_id,
            adapter_binding.native_request,
        )
        try:
            response = self._executor.execute(
                adapter_binding.native_request,
                adapter_binding.saved_preparation,
            )
        except Exception as error:
            result = FreeGsnkeParameterisedEpisodeResult(
                result_id=f"episode-result.freegsnke.{group_id}",
                neutral_request=ObjectIdentity.from_record(neutral.request_id, neutral),
                native_request=native_identity,
                native_response=None,
                disposition=FreeGsnkeParameterisedDisposition.PROVIDER_ERROR,
                reason_codes=(type(error).__name__.upper(),),
                requested_accepted_applied_realized_preserved=True,
                numerical_validity_preserved=True,
                scientific_verdict_constructed=False,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
            )
        else:
            if response.request != native_identity:
                raise ValueError("FreeGSNKE executor substituted its native request")
            observed = response.episode.status is FreeGsnkeTargetEpisodeStatus.OBSERVED_COMPLETE
            result = FreeGsnkeParameterisedEpisodeResult(
                result_id=f"episode-result.freegsnke.{group_id}",
                neutral_request=ObjectIdentity.from_record(neutral.request_id, neutral),
                native_request=native_identity,
                native_response=response,
                disposition=(
                    FreeGsnkeParameterisedDisposition.COMPLETE
                    if observed
                    else FreeGsnkeParameterisedDisposition.NUMERICAL_FAILURE
                ),
                reason_codes=()
                if observed
                else (response.episode.error_code or "NUMERICAL_FAILURE",),
                requested_accepted_applied_realized_preserved=True,
                numerical_validity_preserved=True,
                scientific_verdict_constructed=False,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
            )
        envelope = _envelope(
            context.task_id, LinkedCampaignStageRole.SOURCE_MATERIALIZATION, result
        )
        return _result(context, (result, envelope))


class _FreeGsnkeProjectionRunner:
    def __init__(
        self, *, manifest: CapabilityManifest, config: FreeGsnkeParameterisedProjectionConfig
    ) -> None:
        self.manifest = manifest
        self._config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        records = _decode_inputs(
            context,
            (
                FreeGsnkeParameterisedProjectionConfig,
                FreeGsnkeParameterisedEpisodeResult,
                LinkedCampaignStageEnvelope,
            ),
        )
        config = cast(
            FreeGsnkeParameterisedProjectionConfig,
            records[FreeGsnkeParameterisedProjectionConfig.SCHEMA],
        )
        episode = cast(
            FreeGsnkeParameterisedEpisodeResult,
            records[FreeGsnkeParameterisedEpisodeResult.SCHEMA],
        )
        if config != self._config:
            raise ValueError("FreeGSNKE projection received a substituted config")
        prefix = f"{config.projection_task_prefix}."
        if not context.task_id.startswith(prefix):
            raise ValueError("FreeGSNKE projection task identity differs")
        view_id = context.task_id[len(prefix) :]
        if view_id not in config.view_ids:
            raise ValueError("FreeGSNKE projection view lies outside the roster")
        projection = FreeGsnkeParameterisedProjection(
            projection_id=f"projection.freegsnke.{view_id}",
            view_id=view_id,
            episode_result=ObjectIdentity.from_record(episode.result_id, episode),
            receiver_ids=config.receiver_ids,
            clock_ids=config.clock_ids,
            disposition=episode.disposition,
            reason_codes=episode.reason_codes,
            scientific_verdict_constructed=False,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )
        envelope = _envelope(
            context.task_id, LinkedCampaignStageRole.EVIDENCE_PROJECTION, projection
        )
        return _result(context, (projection, envelope))


def _decode_exact_config(
    context: TaskContext, record_type: type[CanonicalRecord]
) -> CanonicalRecord:
    records = _decode_inputs(context, (record_type,))
    return records[record_type.SCHEMA]


def _decode_inputs(
    context: TaskContext,
    record_types: tuple[type[CanonicalRecord], ...],
) -> dict[str, CanonicalRecord]:
    by_schema = {value.SCHEMA: value for value in record_types}
    values: dict[str, CanonicalRecord] = {}
    for port in context.input_ports:
        record_type = by_schema.get(port.payload_schema)
        try:
            if record_type is None:
                continue
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes or port.read(1):
                raise ValueError("FreeGSNKE input size differs")
            record = decode_canonical_bytes(
                payload,
                record_type,
                maximum_bytes=_MAXIMUM_CONFIG_BYTES,
            )
            if record.SCHEMA in values:
                raise ValueError("FreeGSNKE task received duplicate typed input")
            values[record.SCHEMA] = record
        finally:
            port.close()
    if set(values) != set(by_schema):
        raise ValueError("FreeGSNKE task input roster differs")
    return values


def _envelope(
    task_id: str,
    role: LinkedCampaignStageRole,
    record: CanonicalRecord,
) -> LinkedCampaignStageEnvelope:
    record_id = next(
        getattr(record, value) for value in ("result_id", "projection_id") if hasattr(record, value)
    )
    reasons = cast(tuple[str, ...], getattr(record, "reason_codes"))
    return LinkedCampaignStageEnvelope(
        envelope_id=f"stage-envelope.{task_id}",
        role=role,
        disposition=(
            LinkedCampaignDisposition.SUPPORTED
            if not reasons
            else LinkedCampaignDisposition.UNEVALUABLE
        ),
        scientific_product=ObjectIdentity.from_record(record_id, record),
        reason_codes=reasons,
    )


def _result(context: TaskContext, records: tuple[CanonicalRecord, ...]) -> RunnerResult:
    by_schema = {value.SCHEMA: value for value in records}
    if set(by_schema) != {value.payload_schema for value in context.output_ports}:
        raise ValueError("FreeGSNKE output contract differs")
    return RunnerResult(
        outputs=tuple(
            TaskOutputPayload(port.output_id, by_schema[port.payload_schema].canonical_bytes())
            for port in context.output_ports
        ),
        checks=(
            ReceiptCheck("native-action-chain-preserved", True, ()),
            ReceiptCheck("no-adapter-scientific-verdict", True, ()),
            ReceiptCheck("numerical-status-preserved", True, ()),
        ),
    )


def _payload(record: CanonicalRecord, artifact_id: str) -> ExternalInputPayload:
    record_id = next(getattr(record, value) for value in ("config_id",) if hasattr(record, value))
    parent = ArtifactLineageParent(
        ObjectIdentity.from_record(record_id, record),
        VisibilityCeiling.PROSPECTIVE,
        OutcomeAccess.OUTCOME_BLIND,
    )
    return ExternalInputPayload.from_bytes(
        logical_artifact_id=artifact_id,
        payload_schema=record.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type=FREEGSNKE_PARAMETERISED_MEDIA_TYPE,
        payload=record.canonical_bytes(),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(parent.visibility_ceiling,),
        lineage_parents=(parent,),
        logical_content_sha256=record.fingerprint(),
    )


class _Provider(CampaignRuntimeProvider):
    def __init__(self, registry: CapabilityRegistry, manifest: CapabilityManifest) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest:
            raise ValueError("FreeGSNKE manifest differs from registry")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.manifest = manifest
        self.capability_count = 1

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("FreeGSNKE semantic registry differs")
        del execution_plan
        types = {
            FreeGsnkeParameterisedEpisodeResult.SCHEMA: FreeGsnkeParameterisedEpisodeResult,
            FreeGsnkeParameterisedProjection.SCHEMA: FreeGsnkeParameterisedProjection,
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
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("FreeGSNKE adjudication registry differs")
        del execution_plan
        return None


class FreeGsnkeParameterisedAcquisitionProvider(_Provider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        extension_set: ResponseExperimentExtensionSet,
        substrate_binding: ResponseSubstrateBinding,
        config: FreeGsnkeParameterisedProviderConfig,
        executor: FreeGsnkeParameterisedExecutorPort,
    ) -> None:
        super().__init__(registry, manifest)
        validate_freegsnke_parameterised_config(
            config=config, extension_set=extension_set, substrate_binding=substrate_binding
        )
        self.extension_set = extension_set
        self.substrate_binding = substrate_binding
        self.config = config
        self._executor = executor

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("FreeGSNKE acquisition runner inputs differ")
        return (
            cast(
                TaskRunner,
                _FreeGsnkeAcquisitionRunner(
                    manifest=self.manifest,
                    extension_set=self.extension_set,
                    binding=self.substrate_binding,
                    config=self.config,
                    executor=self._executor,
                ),
            ),
        )

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        return _config_inputs(
            plan,
            source_records,
            self,
            self.config,
            {
                f"{self.config.source_task_prefix}.{value.acquisition_group_id}"
                for value in self.config.acquisition_bindings
            },
        )


class FreeGsnkeParameterisedProjectionProvider(_Provider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: FreeGsnkeParameterisedProjectionConfig,
    ) -> None:
        super().__init__(registry, manifest)
        self.config = config

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("FreeGSNKE projection runner inputs differ")
        return (
            cast(
                TaskRunner, _FreeGsnkeProjectionRunner(manifest=self.manifest, config=self.config)
            ),
        )

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        return _config_inputs(
            plan,
            source_records,
            self,
            self.config,
            {f"{self.config.projection_task_prefix}.{value}" for value in self.config.view_ids},
        )


def _config_inputs(
    plan: ProtocolExecutionPlan,
    source_records: tuple[CanonicalRecord, ...],
    provider: _Provider,
    config: CanonicalRecord,
    expected_task_ids: set[str],
) -> tuple[ExternalInputPayload, ...]:
    if plan.registry_sha256 != provider.registry_sha256 or source_records:
        raise ValueError("FreeGSNKE plan/source inputs differ")
    tasks = tuple(
        value
        for value in plan.tasks
        if value.capability.capability_key == provider.manifest.capability_key
    )
    if {value.task_id for value in tasks} != expected_task_ids:
        raise ValueError("FreeGSNKE task roster differs from config")
    specs = {
        value.logical_artifact_id: value
        for task in tasks
        for value in task.external_inputs
        if value.expected_payload_schema == config.SCHEMA
    }
    if (
        len(specs) != 1
        or next(iter(specs.values())).expected_content_sha256 != config.fingerprint()
    ):
        raise ValueError("FreeGSNKE tasks lack exact shared config")
    return (_payload(config, next(iter(specs))),)


__all__ = [
    "FREEGSNKE_PARAMETERISED_MEDIA_TYPE",
    "FREEGSNKE_PARAMETERISED_PROJECTION_KEY",
    "FREEGSNKE_PARAMETERISED_PROVIDER_KEY",
    "FREEGSNKE_PARAMETERISED_VERSION",
    'FreeGsnkeParameterisedAcquisitionBinding',
    'FreeGsnkeParameterisedAcquisitionProvider',
    'FreeGsnkeParameterisedDisposition',
    'FreeGsnkeParameterisedEpisodeResult',
    'FreeGsnkeParameterisedExecutorPort',
    'FreeGsnkeParameterisedProjectionConfig',
    'FreeGsnkeParameterisedProjectionProvider',
    'FreeGsnkeParameterisedProjection',
    'FreeGsnkeParameterisedProviderConfig',
    'validate_freegsnke_parameterised_config',
]
