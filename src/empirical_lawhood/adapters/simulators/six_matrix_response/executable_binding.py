"""Issued-record reconstruction bindings for Six-matrix response simulator-owned configs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
from typing import cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableBindingRole, ExecutableCapabilityBinding, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.extension_bundles import ExtensionComponentRegistration
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.planning.observation_order import ObservationOrderOwnerRole, ObservationOrderExperimentExtension
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import CapabilityConfigRef, CapabilityManifest
from empirical_lawhood.runtime.adjudication import (
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.execution import TaskRunner
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope, expand_acquisition_views, expand_linked_study_acquisition_views
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.runtime.plans import BarrierKind, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage, ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .contracts import SixMatrixResponseArtifactProfileConfig, SixMatrixResponseSixMatrixSourceConfig
from .artifact_inventory import SixMatrixResponseCompiledHDF5Inventory, compile_six_matrix_response_hdf5_inventory
from .extension_bundle import OBSERVATION_ORDER_HDF5_VALIDATOR, OBSERVATION_ORDER_SOURCE_CAPABILITY, OBSERVATION_ORDER_SOURCE_CONFIG_DECODER, OBSERVATION_ORDER_SOURCE_PROVIDER, MATRIX_RESPONSE_ARTIFACT_CONFIG_DECODER, MATRIX_RESPONSE_ARTIFACT_CONFIG_RECONSTRUCTOR, MATRIX_RESPONSE_EPISODE_VALIDATOR, MATRIX_RESPONSE_RUNTIME_PROVIDER, SIX_MATRIX_RESPONSE_CAPABILITY, MATRIX_RESPONSE_SOURCE_CONFIG_DECODER, REACTIVE_ENTRANCE_HDF5_VALIDATOR, REACTIVE_ENTRANCE_SOURCE_CAPABILITY, REACTIVE_ENTRANCE_SOURCE_CONFIG_DECODER, REACTIVE_ENTRANCE_SOURCE_PROVIDER, PROSPECTIVE_REACTIVE_SOURCE_LAW_HDF5_VALIDATOR, PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY, PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CONFIG_DECODER, PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_PROVIDER, _IMPLEMENTATION_SHA256
from .observation_order import MatrixObservationOrderPairedHistoryResult, MatrixObservationOrderPairedHistorySourceConfig, OBSERVATION_ORDER_HDF5_MEDIA_TYPE, OBSERVATION_ORDER_HDF5_SCHEMA
from .observation_order_provider import OBSERVATION_ORDER_SOURCE_TASK_PREFIX, MatrixObservationOrderPairedHistoryProvider
from .provider import SixMatrixResponseSixMatrixCampaignRuntimeProvider
from .reactive_entrance import REACTIVE_ENTRANCE_HDF5_MEDIA_TYPE, REACTIVE_ENTRANCE_HDF5_SCHEMA, SixMatrixResponseReactiveEntrancePrecursorResult, SixMatrixResponseReactiveEntranceSourceConfig
from .reactive_entrance_provider import SixMatrixResponseReactiveEntrancePrecursorProvider
from .prospective_reactive_source import SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_MEDIA_TYPE, SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA, SixMatrixResponseProspectiveReactiveSourceLawHistoryResult, SixMatrixResponseProspectiveReactiveSourceLawHistorySlot, SixMatrixResponseProspectiveReactiveSourceLawNumericalView, SixMatrixResponseProspectiveReactiveSourceLawRouting, SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, SixMatrixResponseProspectiveReactiveSourceLawTranche, ProspectiveReactiveSourceLawRoutingService
from .model import SixMatrixState
from .prospective_reactive_source_provider import SixMatrixResponseProspectiveReactiveSourceLawHistoryProvider
from .simulation import SixMatrixResponseEpisodeRequest, SixMatrixResponseEpisodeResult
from empirical_lawhood.adapters.methods.matrix_response_study.extension_bundle import OBSERVATION_ORDER_EVALUATION_CAPABILITY, OBSERVATION_ORDER_PROJECTION_CAPABILITY, REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY, REACTIVE_ENTRANCE_PROJECTION_CAPABILITY, PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY, PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY
from empirical_lawhood.adapters.methods.matrix_response_study.observation_order import MatrixObservationOrderEvaluationConfig, MatrixObservationOrderEvaluation, MatrixObservationOrderProjectionConfig, MatrixObservationOrderViewReport
from empirical_lawhood.adapters.methods.matrix_response_study.history_conditioned_geometry import HistoryViewTrace
from empirical_lawhood.adapters.control.backbone_linked_campaign.extension_bundle import (
    OBSERVATION_ORDER_CONFIG_DECODER,
    OBSERVATION_ORDER_SUBSTRATE_DECODER,
)
from empirical_lawhood.runtime.observation_order import ObservationOrderSubstrateBinding
from empirical_lawhood.adapters.methods.matrix_response_study.reactive_entrance_source import MatrixResponseReactiveEntranceAggregate, MatrixResponseReactiveEntranceMethodConfig, MatrixResponseReactiveEntranceProjection
from empirical_lawhood.adapters.methods.matrix_response_study.reactive_entrance_provider import MatrixResponseReactiveEntranceFinalizationProvider, MatrixResponseReactiveEntranceProjectionProvider
from empirical_lawhood.adapters.methods.matrix_response_study.prospective_reactive_source_law import MatrixResponseProspectiveReactiveSourceLawAggregate, MatrixResponseProspectiveReactiveSourceLawMethodConfig, MatrixResponseProspectiveReactiveSourceLawProjection, route_matrix_response_study_reactive_source_evaluation
from empirical_lawhood.adapters.methods.matrix_response_study.prospective_reactive_source_provider import MatrixResponseProspectiveReactiveSourceLawFinalizationProvider, MatrixResponseProspectiveReactiveSourceLawProjectionProvider


_MAXIMUM_PAYLOAD_BYTES = 4 * 1024 * 1024


class SixMatrixResponseRuntimeProvider(CampaignRuntimeProvider):
    """Exact three-role Six-matrix response acquisition/projection/finalization composition."""

    def __init__(
        self,
        providers: tuple[CampaignRuntimeProvider, ...],
    ) -> None:
        if len(providers) != 3 or len({value.registry_sha256 for value in providers}) != 1:
            raise ValueError("Six-matrix response runtime requires its exact three provider roles")
        self._providers = providers
        self.registry_sha256 = providers[0].registry_sha256
        self.capability_count = sum(value.capability_count for value in providers)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        return tuple(
            runner
            for provider in self._providers
            for runner in provider.runners(registry, source_records)
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        by_id: dict[str, ExternalInputPayload] = {}
        for provider in self._providers:
            for value in provider.external_inputs(plan, source_records):
                existing = by_id.get(value.logical_artifact_id)
                if existing is not None and existing != value:
                    raise ValueError("Six-matrix response providers disagree on an external input")
                by_id[value.logical_artifact_id] = value
        return tuple(by_id[value] for value in sorted(by_id))

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        by_key: dict[tuple[str, str, str, str], CapabilityOutputSemanticContract] = {}
        for provider in self._providers:
            for value in provider.output_semantic_contracts(registry, execution_plan):
                existing = by_key.get(value.key)
                if existing is not None and existing != value:
                    raise ValueError("Six-matrix response providers disagree on output semantics")
                by_key[value.key] = value
        return tuple(by_key[value] for value in sorted(by_key))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract | None:
        values = tuple(
            value
            for provider in self._providers
            if (
                value := provider.scientific_adjudication_contract(
                    registry,
                    execution_plan,
                )
            )
            is not None
        )
        if len(values) != 1:
            raise ValueError("Six-matrix response requires one scientific adjudication contract")
        return values[0]




def _decoder(
    *,
    registration_id: str,
    component: ExtensionComponentRegistration,
    record_type: type[CanonicalRecord],
) -> StudyExtensionDecoderRegistration:
    key = component.component_key
    version = component.component_version
    return StudyExtensionDecoderRegistration(
        registration_id=registration_id,
        decoder_key=key,
        decoder_version=version,
        payload_schema=record_type.SCHEMA,
        payload_version=record_type.VERSION,
        config_sha256=sha256(
            canonical_json_bytes(
                {
                    "decoder_key": key,
                    "decoder_version": version,
                    "mode": "exact-canonical-record",
                    "payload_schema": record_type.SCHEMA,
                }
            )
        ).hexdigest(),
        implementation_sha256=component.implementation_sha256,
        maximum_payload_bytes=_MAXIMUM_PAYLOAD_BYTES,
    )


MATRIX_RESPONSE_ARTIFACT_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.six-matrix-response-artifact-profile-config",
    component=MATRIX_RESPONSE_ARTIFACT_CONFIG_DECODER,
    record_type=SixMatrixResponseArtifactProfileConfig,
)
MATRIX_RESPONSE_SOURCE_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.six-matrix-response-source-config",
    component=MATRIX_RESPONSE_SOURCE_CONFIG_DECODER,
    record_type=SixMatrixResponseSixMatrixSourceConfig,
)
REACTIVE_ENTRANCE_SOURCE_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-response-reactive-entrance-source-config",
    component=REACTIVE_ENTRANCE_SOURCE_CONFIG_DECODER,
    record_type=SixMatrixResponseReactiveEntranceSourceConfig,
)
PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-response-prospective-reactive-source-law-source-config",
    component=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CONFIG_DECODER,
    record_type=SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
)
OBSERVATION_ORDER_CARRIER_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-observation-order-observation-carrier",
    component=OBSERVATION_ORDER_CONFIG_DECODER,
    record_type=ObservationOrderExperimentExtension,
)
OBSERVATION_ORDER_SOURCE_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-observation-order-source-config",
    component=OBSERVATION_ORDER_SOURCE_CONFIG_DECODER,
    record_type=MatrixObservationOrderPairedHistorySourceConfig,
)
OBSERVATION_ORDER_SUBSTRATE_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-observation-order-substrate-binding",
    component=OBSERVATION_ORDER_SUBSTRATE_DECODER,
    record_type=ObservationOrderSubstrateBinding,
)


def _binding(
    *,
    binding_id: str,
    decoder_component: ExtensionComponentRegistration,
    reconstructor_component: ExtensionComponentRegistration,
    decoder: StudyExtensionDecoderRegistration,
    record_type: type[CanonicalRecord],
    output_schema: str,
) -> ExecutableCapabilityBinding:
    decoder_identity = ObjectIdentity.from_record(
        decoder_component.registration_id, decoder_component
    )
    return ExecutableCapabilityBinding(
        binding_id=binding_id,
        capability_key=reconstructor_component.component_key,
        capability_version=reconstructor_component.component_version,
        capability_implementation_sha256=_IMPLEMENTATION_SHA256,
        role=ExecutableBindingRole.PROFILE_COMPILER,
        provider_key=reconstructor_component.component_key,
        provider_version=reconstructor_component.component_version,
        provider_implementation_sha256=_IMPLEMENTATION_SHA256,
        capability_backed=False,
        discovery_components=tuple(
            sorted(
                (decoder_component, reconstructor_component),
                key=lambda value: value.registration_id,
            )
        ),
        accepted_profile_types=(),
        accepted_config_types=(
            ExecutableRecordTypeBinding(record_type.SCHEMA, record_type.VERSION),
        ),
        required_issued_payload_schemas=(record_type.SCHEMA,),
        codec_registration_identities=(decoder_identity,),
        issued_decoder_registrations=(decoder,),
        input_schema_ids=(record_type.SCHEMA,),
        output_schema_ids=(output_schema,),
        artifact_validator_identities=(),
        required_platform_port_keys=(),
        may_require_active_mount=False,
        may_require_source_qualification=False,
        may_require_network=False,
        may_require_authority=False,
    )


MATRIX_RESPONSE_ARTIFACT_RECONSTRUCTION_BINDING = _binding(
    binding_id="binding.six-matrix-response-artifact-profile-config-reconstructor",
    decoder_component=MATRIX_RESPONSE_ARTIFACT_CONFIG_DECODER,
    reconstructor_component=MATRIX_RESPONSE_ARTIFACT_CONFIG_RECONSTRUCTOR,
    decoder=MATRIX_RESPONSE_ARTIFACT_DECODER_REGISTRATION,
    record_type=SixMatrixResponseArtifactProfileConfig,
    output_schema=SixMatrixResponseCompiledHDF5Inventory.SCHEMA,
)
MATRIX_RESPONSE_SIMULATOR_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.six-matrix-response-runtime-provider",
    capability_key=SIX_MATRIX_RESPONSE_CAPABILITY.capability_key,
    capability_version=SIX_MATRIX_RESPONSE_CAPABILITY.capability_version,
    capability_implementation_sha256=SIX_MATRIX_RESPONSE_CAPABILITY.implementation_sha256,
    role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
    provider_key=MATRIX_RESPONSE_RUNTIME_PROVIDER.component_key,
    provider_version=MATRIX_RESPONSE_RUNTIME_PROVIDER.component_version,
    provider_implementation_sha256=MATRIX_RESPONSE_RUNTIME_PROVIDER.implementation_sha256,
    capability_backed=True,
    discovery_components=tuple(
        sorted(
            (MATRIX_RESPONSE_EPISODE_VALIDATOR, MATRIX_RESPONSE_RUNTIME_PROVIDER, MATRIX_RESPONSE_SOURCE_CONFIG_DECODER),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_profile_types=(),
    accepted_config_types=(
        ExecutableRecordTypeBinding(
            SixMatrixResponseSixMatrixSourceConfig.SCHEMA,
            SixMatrixResponseSixMatrixSourceConfig.VERSION,
        ),
    ),
    required_issued_payload_schemas=(SixMatrixResponseSixMatrixSourceConfig.SCHEMA,),
    codec_registration_identities=(
        ObjectIdentity.from_record(
            MATRIX_RESPONSE_SOURCE_CONFIG_DECODER.registration_id,
            MATRIX_RESPONSE_SOURCE_CONFIG_DECODER,
        ),
    ),
    issued_decoder_registrations=(MATRIX_RESPONSE_SOURCE_DECODER_REGISTRATION,),
    input_schema_ids=(SixMatrixResponseEpisodeRequest.SCHEMA,),
    output_schema_ids=(SixMatrixResponseEpisodeResult.SCHEMA,),
    artifact_validator_identities=(
        ObjectIdentity.from_record(MATRIX_RESPONSE_EPISODE_VALIDATOR.registration_id, MATRIX_RESPONSE_EPISODE_VALIDATOR),
    ),
    required_platform_port_keys=(),
    may_require_active_mount=False,
    may_require_source_qualification=False,
    may_require_network=False,
    may_require_authority=False,
)
REACTIVE_ENTRANCE_SOURCE_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.matrix-response-reactive-entrance-reactive-precursor",
    capability_key=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.capability_key,
    capability_version=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.capability_version,
    capability_implementation_sha256=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.implementation_sha256,
    role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
    provider_key=REACTIVE_ENTRANCE_SOURCE_PROVIDER.component_key,
    provider_version=REACTIVE_ENTRANCE_SOURCE_PROVIDER.component_version,
    provider_implementation_sha256=REACTIVE_ENTRANCE_SOURCE_PROVIDER.implementation_sha256,
    capability_backed=True,
    discovery_components=tuple(
        sorted(
            (
                REACTIVE_ENTRANCE_HDF5_VALIDATOR,
                REACTIVE_ENTRANCE_SOURCE_CONFIG_DECODER,
                REACTIVE_ENTRANCE_SOURCE_PROVIDER,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_profile_types=(),
    accepted_config_types=tuple(
        sorted(
            (
                ExecutableRecordTypeBinding(MatrixResponseReactiveEntranceMethodConfig.SCHEMA, "1.0.0"),
                ExecutableRecordTypeBinding(SixMatrixResponseReactiveEntranceSourceConfig.SCHEMA, "1.0.0"),
                ExecutableRecordTypeBinding(
                    ResponseExperimentExtensionSet.SCHEMA,
                    ResponseExperimentExtensionSet.VERSION,
                ),
                ExecutableRecordTypeBinding(
                    ResponseSubstrateBinding.SCHEMA,
                    ResponseSubstrateBinding.VERSION,
                ),
            ),
            key=lambda value: value.record_schema,
        )
    ),
    required_issued_payload_schemas=(SixMatrixResponseReactiveEntranceSourceConfig.SCHEMA,),
    required_authenticated_record_schemas=tuple(
        sorted(
            (
                MatrixResponseReactiveEntranceMethodConfig.SCHEMA,
                ResponseExperimentExtensionSet.SCHEMA,
                ResponseSubstrateBinding.SCHEMA,
            )
        )
    ),
    codec_registration_identities=(
        ObjectIdentity.from_record(
            REACTIVE_ENTRANCE_SOURCE_CONFIG_DECODER.registration_id,
            REACTIVE_ENTRANCE_SOURCE_CONFIG_DECODER,
        ),
    ),
    issued_decoder_registrations=(REACTIVE_ENTRANCE_SOURCE_DECODER_REGISTRATION,),
    input_schema_ids=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.input_schema_ids,
    output_schema_ids=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.output_schema_ids,
    artifact_validator_identities=(
        ObjectIdentity.from_record(
            REACTIVE_ENTRANCE_HDF5_VALIDATOR.registration_id,
            REACTIVE_ENTRANCE_HDF5_VALIDATOR,
        ),
    ),
    required_platform_port_keys=(),
    may_require_active_mount=True,
    may_require_source_qualification=True,
    may_require_network=False,
    may_require_authority=True,
)
PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.matrix-response-prospective-reactive-source-law-complete-history-acquisition",
    capability_key=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.capability_key,
    capability_version=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.capability_version,
    capability_implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.implementation_sha256,
    role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
    provider_key=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_PROVIDER.component_key,
    provider_version=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_PROVIDER.component_version,
    provider_implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_PROVIDER.implementation_sha256,
    capability_backed=True,
    discovery_components=tuple(sorted((PROSPECTIVE_REACTIVE_SOURCE_LAW_HDF5_VALIDATOR, PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CONFIG_DECODER, PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_PROVIDER), key=lambda value: value.registration_id)),
    accepted_profile_types=(),
    accepted_config_types=tuple(sorted((
        ExecutableRecordTypeBinding(MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA, MatrixResponseProspectiveReactiveSourceLawMethodConfig.VERSION),
        ExecutableRecordTypeBinding(SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.SCHEMA, SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.VERSION),
        ExecutableRecordTypeBinding(ResponseExperimentExtensionSet.SCHEMA, ResponseExperimentExtensionSet.VERSION),
        ExecutableRecordTypeBinding(ResponseSubstrateBinding.SCHEMA, ResponseSubstrateBinding.VERSION),
    ), key=lambda value: value.record_schema)),
    required_issued_payload_schemas=(SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.SCHEMA,),
    required_authenticated_record_schemas=tuple(sorted((MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA, ResponseExperimentExtensionSet.SCHEMA, ResponseSubstrateBinding.SCHEMA))),
    codec_registration_identities=(ObjectIdentity.from_record(PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CONFIG_DECODER.registration_id, PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CONFIG_DECODER),),
    issued_decoder_registrations=(PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_DECODER_REGISTRATION,),
    input_schema_ids=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.input_schema_ids,
    output_schema_ids=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.output_schema_ids,
    artifact_validator_identities=(ObjectIdentity.from_record(PROSPECTIVE_REACTIVE_SOURCE_LAW_HDF5_VALIDATOR.registration_id, PROSPECTIVE_REACTIVE_SOURCE_LAW_HDF5_VALIDATOR),),
    required_platform_port_keys=(), may_require_active_mount=True,
    may_require_source_qualification=True, may_require_network=False, may_require_authority=True,
)

OBSERVATION_ORDER_SOURCE_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.matrix-observation-order-paired-history-acquisition",
    capability_key=OBSERVATION_ORDER_SOURCE_CAPABILITY.capability_key,
    capability_version=OBSERVATION_ORDER_SOURCE_CAPABILITY.capability_version,
    capability_implementation_sha256=OBSERVATION_ORDER_SOURCE_CAPABILITY.implementation_sha256,
    role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
    provider_key=OBSERVATION_ORDER_SOURCE_PROVIDER.component_key,
    provider_version=OBSERVATION_ORDER_SOURCE_PROVIDER.component_version,
    provider_implementation_sha256=OBSERVATION_ORDER_SOURCE_PROVIDER.implementation_sha256,
    capability_backed=True,
    discovery_components=tuple(
        sorted(
            (
                OBSERVATION_ORDER_HDF5_VALIDATOR,
                OBSERVATION_ORDER_SOURCE_CONFIG_DECODER,
                OBSERVATION_ORDER_SOURCE_PROVIDER,
                OBSERVATION_ORDER_CONFIG_DECODER,
                OBSERVATION_ORDER_SUBSTRATE_DECODER,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_profile_types=(),
    accepted_config_types=tuple(
        sorted(
            (
                ExecutableRecordTypeBinding(
                    MatrixObservationOrderPairedHistorySourceConfig.SCHEMA,
                    MatrixObservationOrderPairedHistorySourceConfig.VERSION,
                ),
                ExecutableRecordTypeBinding(
                    ObservationOrderExperimentExtension.SCHEMA,
                    ObservationOrderExperimentExtension.VERSION,
                ),
                ExecutableRecordTypeBinding(
                    ObservationOrderSubstrateBinding.SCHEMA,
                    ObservationOrderSubstrateBinding.VERSION,
                ),
            ),
            key=lambda value: value.record_schema,
        )
    ),
    required_issued_payload_schemas=tuple(
        sorted(
            (
                MatrixObservationOrderPairedHistorySourceConfig.SCHEMA,
                ObservationOrderExperimentExtension.SCHEMA,
                ObservationOrderSubstrateBinding.SCHEMA,
            )
        )
    ),
    codec_registration_identities=tuple(
        sorted(
            (
                ObjectIdentity.from_record(value.registration_id, value)
                for value in (
                    OBSERVATION_ORDER_SOURCE_CONFIG_DECODER,
                    OBSERVATION_ORDER_CONFIG_DECODER,
                    OBSERVATION_ORDER_SUBSTRATE_DECODER,
                )
            ),
            key=lambda value: value.object_id,
        )
    ),
    issued_decoder_registrations=tuple(
        sorted(
            (
                OBSERVATION_ORDER_CARRIER_DECODER_REGISTRATION,
                OBSERVATION_ORDER_SOURCE_DECODER_REGISTRATION,
                OBSERVATION_ORDER_SUBSTRATE_DECODER_REGISTRATION,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    input_schema_ids=OBSERVATION_ORDER_SOURCE_CAPABILITY.input_schema_ids,
    output_schema_ids=OBSERVATION_ORDER_SOURCE_CAPABILITY.output_schema_ids,
    artifact_validator_identities=(
        ObjectIdentity.from_record(OBSERVATION_ORDER_HDF5_VALIDATOR.registration_id, OBSERVATION_ORDER_HDF5_VALIDATOR),
    ),
    required_platform_port_keys=(),
    may_require_active_mount=True,
    may_require_source_qualification=False,
    may_require_network=False,
    may_require_authority=True,
)


@dataclass(frozen=True, slots=True)
class SixMatrixResponseSixMatrixProviderFactory:
    """Construct the source-free provider without executing or contacting a source."""

    binding: ExecutableCapabilityBinding = MATRIX_RESPONSE_SIMULATOR_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> SixMatrixResponseSixMatrixCampaignRuntimeProvider:
        if (
            platform_ports
            or len(records) != 1
            or not isinstance(records[0], SixMatrixResponseSixMatrixSourceConfig)
        ):
            raise ValueError("Six-matrix response provider requires one exact issued source config and no ports")
        return SixMatrixResponseSixMatrixCampaignRuntimeProvider(
            registry=registry,
            manifest=SIX_MATRIX_RESPONSE_CAPABILITY,
            source_config=records[0],
        )


@dataclass(frozen=True, slots=True)
class SixMatrixResponseArtifactInventoryCompilerFactory:
    binding: ExecutableCapabilityBinding = MATRIX_RESPONSE_ARTIFACT_RECONSTRUCTION_BINDING

    def build_compiler(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> SixMatrixResponseCompiledHDF5Inventory:
        if (
            platform_ports
            or len(records) != 1
            or not isinstance(records[0], SixMatrixResponseArtifactProfileConfig)
        ):
            raise ValueError("Six-matrix response artifact compiler requires one exact issued profile")
        return compile_six_matrix_response_hdf5_inventory(records[0])


@dataclass(frozen=True, slots=True)
class SixMatrixResponseReactiveEntrancePrecursorProviderFactory:
    """Bind the exact 640-view reactive entrance topology to installed source/method owners."""

    binding: ExecutableCapabilityBinding = REACTIVE_ENTRANCE_SOURCE_EXECUTABLE_BINDING

    @staticmethod
    def _config_ref(record: CanonicalRecord, manifest: CapabilityManifest) -> CapabilityConfigRef:
        config_id = getattr(record, "config_id")
        if not isinstance(config_id, str):
            raise ValueError("matrix response reactive entrance config lacks a stable identity")
        return CapabilityConfigRef(
            config_id=config_id,
            config_schema=record.SCHEMA,
            config_schema_sha256=manifest.config_schema_sha256,
            content_sha256=record.fingerprint(),
            artifact_id=f"config-artifact.{config_id}",
        )

    @staticmethod
    def _json_output(output_id: str, schema: str) -> OutputTemplate:
        return OutputTemplate(
            output_id=output_id,
            payload_schema=schema,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            filename_suffix=".canonical.json",
        )

    def expand_parameterised_protocol(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        template: ProtocolTemplate,
    ) -> ProtocolTemplate:
        by_type = {type(value): value for value in records}
        expected = {
            MatrixResponseReactiveEntranceMethodConfig,
            SixMatrixResponseReactiveEntranceSourceConfig,
            ResponseExperimentExtensionSet,
            ResponseSubstrateBinding,
        }
        if len(by_type) != len(records) or set(by_type) != expected:
            raise ValueError("matrix response reactive entrance topology expansion requires its exact record roster")
        source_config = cast(SixMatrixResponseReactiveEntranceSourceConfig, by_type[SixMatrixResponseReactiveEntranceSourceConfig])
        method_config = cast(MatrixResponseReactiveEntranceMethodConfig, by_type[MatrixResponseReactiveEntranceMethodConfig])
        extension = cast(
            ResponseExperimentExtensionSet,
            by_type[ResponseExperimentExtensionSet],
        )
        substrate = cast(ResponseSubstrateBinding, by_type[ResponseSubstrateBinding])
        if substrate.installed_executable_binding != ObjectIdentity.from_record(
            self.binding.binding_id, self.binding
        ):
            raise ValueError("matrix response reactive entrance substrate selects another executable binding")
        identification_config = extension.identification_config
        if (
            len(identification_config.physical_units) != 512
            or len(identification_config.acquisition_groups) != 640
            or len(identification_config.nested_views) != 640
            or set(identification_config.development_unit_ids)
            != {value.physical_independent_unit_id for value in source_config.slots[:256]}
            or set(identification_config.evaluation_unit_ids)
            != {value.physical_independent_unit_id for value in source_config.slots[256:]}
        ):
            raise ValueError("matrix response reactive entrance topology changes its 512-unit/640-view roster")

        source = ProtocolStepTemplate(
            step_id="matrix-response-reactive-entrance-source-prototype",
            stage=ScientificStage.PREPARE,
            capability_key=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.capability_key,
            capability_version=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.capability_version,
            config=self._config_ref(source_config, REACTIVE_ENTRANCE_SOURCE_CAPABILITY),
            dependency_step_ids=(),
            outputs=tuple(
                sorted(
                    (
                        self._json_output(
                            "precursor-result",
                            SixMatrixResponseReactiveEntrancePrecursorResult.SCHEMA,
                        ),
                        OutputTemplate(
                            output_id="precursor-trajectory",
                            payload_schema=REACTIVE_ENTRANCE_HDF5_SCHEMA,
                            profile=ArtifactProfile.AUDITED_HDF5,
                            media_type=REACTIVE_ENTRANCE_HDF5_MEDIA_TYPE,
                            filename_suffix=".h5",
                        ),
                        self._json_output(
                            "stage-envelope",
                            LinkedCampaignStageEnvelope.SCHEMA,
                        ),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.resource_ceiling,
            resource_lock_ids=(),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("matrix-response-reactive-entrance-complete-precursor-custody",),
        )
        projection = ProtocolStepTemplate(
            step_id="matrix-response-reactive-entrance-projection-prototype",
            stage=ScientificStage.TRANSFORM,
            capability_key=REACTIVE_ENTRANCE_PROJECTION_CAPABILITY.capability_key,
            capability_version=REACTIVE_ENTRANCE_PROJECTION_CAPABILITY.capability_version,
            config=self._config_ref(method_config, REACTIVE_ENTRANCE_PROJECTION_CAPABILITY),
            dependency_step_ids=(),
            outputs=tuple(
                sorted(
                    (
                        self._json_output("projection", MatrixResponseReactiveEntranceProjection.SCHEMA),
                        self._json_output(
                            "stage-envelope",
                            LinkedCampaignStageEnvelope.SCHEMA,
                        ),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=REACTIVE_ENTRANCE_PROJECTION_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=REACTIVE_ENTRANCE_PROJECTION_CAPABILITY.resource_ceiling,
            resource_lock_ids=(),
            barrier=BarrierKind.FREEZE,
            maximum_attempts=2,
            obligation_ids=("matrix-response-reactive-entrance-causal-entrance-projection",),
        )
        expanded = expand_linked_study_acquisition_views(
            template=template,
            identification_config=identification_config,
            source_task_prefix="matrix-response-reactive-entrance-acquire",
            projection_task_prefix=method_config.projection_task_prefix,
            source_step_template=source,
            projection_step_template=projection,
        )
        finalizer_config = self._config_ref(method_config, REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY)
        steps = []
        for step in expanded.steps:
            if step.step_id == method_config.aggregate_task_id:
                steps.append(
                    replace(
                        step,
                        stage=ScientificStage.EVALUATE,
                        capability_key=REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY.capability_key,
                        capability_version=REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY.capability_version,
                        config=finalizer_config,
                        outputs=tuple(
                            sorted(
                                (
                                    self._json_output(
                                        "source-aggregate",
                                        MatrixResponseReactiveEntranceAggregate.SCHEMA,
                                    ),
                                    self._json_output(
                                        "stage-envelope",
                                        LinkedCampaignStageEnvelope.SCHEMA,
                                    ),
                                    self._json_output(
                                        "scientific-adjudication",
                                        ScientificAdjudicationRecord.SCHEMA,
                                    ),
                                ),
                                key=lambda value: value.output_id,
                            )
                        ),
                        required_permissions=REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY.permissions,
                        requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                        resource_budget=REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY.resource_ceiling,
                        barrier=BarrierKind.REVEAL,
                        obligation_ids=("matrix-response-reactive-entrance-single-terminal",),
                    )
                )
            elif step.step_id in {
                "linked-law-qualification",
                "linked-atlas-assembly",
                "linked-programme-admission-evidence",
                "linked-admission",
                "linked-reachability",
                "linked-programme-authoring",
            }:
                continue
            else:
                steps.append(step)
        return replace(
            expanded,
            steps=tuple(sorted(steps, key=lambda value: value.step_id)),
            requests_controller=False,
            nonactuating=True,
        )

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> SixMatrixResponseRuntimeProvider:
        by_type = {type(value): value for value in records}
        expected = {
            MatrixResponseReactiveEntranceMethodConfig,
            SixMatrixResponseReactiveEntranceSourceConfig,
            ResponseExperimentExtensionSet,
            ResponseSubstrateBinding,
        }
        if platform_ports or len(by_type) != len(records) or set(by_type) != expected:
            raise ValueError("matrix response reactive entrance source provider requires its exact issued records")
        substrate = cast(ResponseSubstrateBinding, by_type[ResponseSubstrateBinding])
        if substrate.installed_executable_binding != ObjectIdentity.from_record(
            self.binding.binding_id, self.binding
        ):
            raise ValueError("matrix response reactive entrance source provider binding differs")
        source_config = cast(SixMatrixResponseReactiveEntranceSourceConfig, by_type[SixMatrixResponseReactiveEntranceSourceConfig])
        method_config = cast(MatrixResponseReactiveEntranceMethodConfig, by_type[MatrixResponseReactiveEntranceMethodConfig])
        source_provider = SixMatrixResponseReactiveEntrancePrecursorProvider(
            registry=registry,
            manifest=REACTIVE_ENTRANCE_SOURCE_CAPABILITY,
            extension_set=cast(
                ResponseExperimentExtensionSet,
                by_type[ResponseExperimentExtensionSet],
            ),
            substrate_binding=substrate,
            source_config=source_config,
            installed_executable_binding=ObjectIdentity.from_record(
                self.binding.binding_id,
                self.binding,
            ),
        )
        projection_provider = MatrixResponseReactiveEntranceProjectionProvider(
            registry=registry,
            manifest=REACTIVE_ENTRANCE_PROJECTION_CAPABILITY,
            source_config=source_config,
            method_config=method_config,
        )
        finalization_provider = MatrixResponseReactiveEntranceFinalizationProvider(
            registry=registry,
            manifest=REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY,
            source_config=source_config,
            method_config=method_config,
        )
        return SixMatrixResponseRuntimeProvider(
            (source_provider, projection_provider, finalization_provider)
        )


@dataclass(frozen=True, slots=True)
class SixMatrixResponseProspectiveReactiveSourceLawHistoryProviderFactory:
    """Bind the exact 256-history/320-view PSL topology."""

    binding: ExecutableCapabilityBinding = PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_EXECUTABLE_BINDING

    @staticmethod
    def _config_ref(record: CanonicalRecord, manifest: CapabilityManifest) -> CapabilityConfigRef:
        return SixMatrixResponseReactiveEntrancePrecursorProviderFactory._config_ref(record, manifest)

    @staticmethod
    def _json_output(output_id: str, schema: str) -> OutputTemplate:
        return SixMatrixResponseReactiveEntrancePrecursorProviderFactory._json_output(output_id, schema)

    def _records(self, records: tuple[CanonicalRecord, ...]) -> tuple[SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, MatrixResponseProspectiveReactiveSourceLawMethodConfig, ResponseExperimentExtensionSet, ResponseSubstrateBinding]:
        by_type = {type(value): value for value in records}
        expected = {SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, MatrixResponseProspectiveReactiveSourceLawMethodConfig, ResponseExperimentExtensionSet, ResponseSubstrateBinding}
        if len(by_type) != len(records) or set(by_type) != expected:
            raise ValueError("matrix response prospective reactive source law binding requires its exact record roster")
        return (
            cast(SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, by_type[SixMatrixResponseProspectiveReactiveSourceLawSourceConfig]),
            cast(MatrixResponseProspectiveReactiveSourceLawMethodConfig, by_type[MatrixResponseProspectiveReactiveSourceLawMethodConfig]),
            cast(ResponseExperimentExtensionSet, by_type[ResponseExperimentExtensionSet]),
            cast(ResponseSubstrateBinding, by_type[ResponseSubstrateBinding]),
        )

    def expand_parameterised_protocol(self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate) -> ProtocolTemplate:
        source_config, method_config, extension, substrate = self._records(records)
        if substrate.installed_executable_binding != ObjectIdentity.from_record(self.binding.binding_id, self.binding):
            raise ValueError("matrix response prospective reactive source law substrate selects another binding")
        identification_config = extension.identification_config
        if (
            len(identification_config.physical_units) != 256
            or len(identification_config.acquisition_groups) != 320
            or len(identification_config.nested_views) != 320
            or len(identification_config.development_unit_ids) != 128
            or len(identification_config.evaluation_unit_ids) != 128
        ):
            raise ValueError("matrix response prospective reactive source law topology changes its 256-unit/320-view roster")
        source = ProtocolStepTemplate(
            step_id="matrix-response-prospective-reactive-source-law-source-prototype", stage=ScientificStage.PREPARE,
            capability_key=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.capability_key,
            capability_version=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.capability_version,
            config=self._config_ref(source_config, PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY), dependency_step_ids=(),
            outputs=tuple(sorted((
                self._json_output("history-result", SixMatrixResponseProspectiveReactiveSourceLawHistoryResult.SCHEMA),
                OutputTemplate(
                    output_id="history-panel",
                    payload_schema=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA,
                    profile=ArtifactProfile.AUDITED_HDF5,
                    media_type=SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_MEDIA_TYPE,
                    filename_suffix=".h5",
                ),
                self._json_output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            ), key=lambda value: value.output_id)),
            required_permissions=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.resource_ceiling,
            resource_lock_ids=(), barrier=BarrierKind.NONE, maximum_attempts=2,
            obligation_ids=("matrix-response-prospective-reactive-source-law-routing-before-future-custody",),
        )
        projection = ProtocolStepTemplate(
            step_id="matrix-response-prospective-reactive-source-law-projection-prototype", stage=ScientificStage.TRANSFORM,
            capability_key=PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY.capability_key,
            capability_version=PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY.capability_version,
            config=self._config_ref(method_config, PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY), dependency_step_ids=(),
            outputs=tuple(sorted((
                self._json_output("projection", MatrixResponseProspectiveReactiveSourceLawProjection.SCHEMA),
                self._json_output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            ), key=lambda value: value.output_id)),
            required_permissions=PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY.resource_ceiling,
            resource_lock_ids=(), barrier=BarrierKind.FREEZE, maximum_attempts=2,
            obligation_ids=("matrix-response-prospective-reactive-source-law-prefix-and-outcome-projection",),
        )
        expanded = expand_linked_study_acquisition_views(
            template=template, identification_config=identification_config,
            source_task_prefix="matrix-response-prospective-reactive-source-law-acquire",
            projection_task_prefix=method_config.projection_task_prefix,
            source_step_template=source, projection_step_template=projection,
        )
        final_ref = self._config_ref(method_config, PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY)
        steps = []
        for step in expanded.steps:
            if step.step_id == method_config.aggregate_task_id:
                steps.append(replace(
                    step, stage=ScientificStage.EVALUATE,
                    capability_key=PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY.capability_key,
                    capability_version=PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY.capability_version,
                    config=final_ref,
                    outputs=tuple(sorted((
                        self._json_output("source-aggregate", MatrixResponseProspectiveReactiveSourceLawAggregate.SCHEMA),
                        self._json_output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                        self._json_output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                    ), key=lambda value: value.output_id)),
                    required_permissions=PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY.permissions,
                    requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                    resource_budget=PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY.resource_ceiling,
                    barrier=BarrierKind.REVEAL, obligation_ids=("matrix-response-prospective-reactive-source-law-single-terminal",),
                ))
            elif step.step_id in {
                "linked-law-qualification", "linked-atlas-assembly", "linked-programme-admission-evidence",
                "linked-admission", "linked-reachability", "linked-programme-authoring",
            }:
                continue
            else:
                steps.append(step)
        return replace(expanded, steps=tuple(sorted(steps, key=lambda value: value.step_id)), requests_controller=False, nonactuating=True)

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> SixMatrixResponseRuntimeProvider:
        if platform_ports:
            raise ValueError("matrix response prospective reactive source law provider accepts no platform ports")
        source_config, method_config, extension, substrate = self._records(records)
        binding_identity = ObjectIdentity.from_record(self.binding.binding_id, self.binding)
        if substrate.installed_executable_binding != binding_identity:
            raise ValueError("matrix response prospective reactive source law provider binding differs")
        routing_service: ProspectiveReactiveSourceLawRoutingService | None = None
        if source_config.tranche is SixMatrixResponseProspectiveReactiveSourceLawTranche.PROSPECTIVE_EVALUATION:
            def route(
                source: SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
                slot: SixMatrixResponseProspectiveReactiveSourceLawHistorySlot,
                numerical: SixMatrixResponseProspectiveReactiveSourceLawNumericalView,
                states: tuple[SixMatrixState, ...],
            ) -> SixMatrixResponseProspectiveReactiveSourceLawRouting:
                return route_matrix_response_study_reactive_source_evaluation(
                    source=source, method=method_config, slot=slot,
                    numerical_view=numerical, routing_states=states,
                )
            routing_service = route
        source_provider = SixMatrixResponseProspectiveReactiveSourceLawHistoryProvider(
            registry=registry, manifest=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY,
            extension_set=extension, substrate_binding=substrate,
            source_config=source_config, installed_executable_binding=binding_identity,
            routing_service=routing_service,
        )
        projection_provider = MatrixResponseProspectiveReactiveSourceLawProjectionProvider(
            registry=registry, manifest=PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY,
            source=source_config, method=method_config,
        )
        finalization_provider = MatrixResponseProspectiveReactiveSourceLawFinalizationProvider(
            registry=registry, manifest=PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY,
            source=source_config, method=method_config,
        )
        return SixMatrixResponseRuntimeProvider((source_provider, projection_provider, finalization_provider))


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderPairedHistoryProviderFactory:
    "Bind the exact observation-order carrier to the three installed observation order owners."

    binding: ExecutableCapabilityBinding = OBSERVATION_ORDER_SOURCE_EXECUTABLE_BINDING

    @staticmethod
    def _config_ref(record: CanonicalRecord, manifest: CapabilityManifest) -> CapabilityConfigRef:
        return SixMatrixResponseReactiveEntrancePrecursorProviderFactory._config_ref(record, manifest)

    @staticmethod
    def _json_output(output_id: str, schema: str) -> OutputTemplate:
        return SixMatrixResponseReactiveEntrancePrecursorProviderFactory._json_output(output_id, schema)

    @staticmethod
    def _expansion_records(
        records: tuple[CanonicalRecord, ...],
    ) -> tuple[
        ObservationOrderExperimentExtension,
        ObservationOrderSubstrateBinding,
        MatrixObservationOrderPairedHistorySourceConfig,
        MatrixObservationOrderProjectionConfig,
        MatrixObservationOrderEvaluationConfig,
    ]:
        by_type = {type(value): value for value in records}
        expected = {
            ObservationOrderExperimentExtension,
            ObservationOrderSubstrateBinding,
            MatrixObservationOrderPairedHistorySourceConfig,
            MatrixObservationOrderProjectionConfig,
            MatrixObservationOrderEvaluationConfig,
        }
        if len(by_type) != len(records) or set(by_type) != expected:
            raise ValueError("observation order expansion requires its exact five issued records")
        return (
            cast(ObservationOrderExperimentExtension, by_type[ObservationOrderExperimentExtension]),
            cast(
                ObservationOrderSubstrateBinding,
                by_type[ObservationOrderSubstrateBinding],
            ),
            cast(MatrixObservationOrderPairedHistorySourceConfig, by_type[MatrixObservationOrderPairedHistorySourceConfig]),
            cast(MatrixObservationOrderProjectionConfig, by_type[MatrixObservationOrderProjectionConfig]),
            cast(MatrixObservationOrderEvaluationConfig, by_type[MatrixObservationOrderEvaluationConfig]),
        )

    def expand_parameterised_protocol(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        template: ProtocolTemplate,
    ) -> ProtocolTemplate:
        carrier, association, source_config, projection_config, evaluation_config = (
            self._expansion_records(records)
        )
        substrate = association.substrate
        binding_identity = ObjectIdentity.from_record(self.binding.binding_id, self.binding)
        config_by_role = {value.role: value.config for value in carrier.owners}
        source_identity = ObjectIdentity.from_record(source_config.config_id, source_config)
        projection_identity = ObjectIdentity.from_record(
            projection_config.config_id,
            projection_config,
        )
        evaluation_identity = ObjectIdentity.from_record(
            evaluation_config.config_id,
            evaluation_config,
        )
        if (
            association.observation_carrier
            != ObjectIdentity.from_record(carrier.extension_set_id, carrier)
            or association.source_config != source_identity
            or substrate.installed_executable_binding != binding_identity
            or substrate.native_config != source_identity
            or config_by_role[ObservationOrderOwnerRole.SOURCE] != source_identity
            or config_by_role[ObservationOrderOwnerRole.EVIDENCE_PROJECTION]
            != projection_identity
            or config_by_role[ObservationOrderOwnerRole.METHOD] != projection_identity
            or config_by_role[ObservationOrderOwnerRole.SEALED_EVALUATOR]
            != evaluation_identity
            or projection_config.source_config != source_identity
            or evaluation_config.projection_config != projection_identity
        ):
            raise ValueError("observation order owner/config lineage differs")
        if (
            len(carrier.physical_units) != 256
            or len(carrier.acquisition_groups) != 256
            or len(carrier.nested_views) != 512
            or {value.physical_independent_unit_id for value in source_config.slots}
            != {value.physical_independent_unit_id for value in carrier.physical_units}
            or {value.acquisition_group_id for value in source_config.slots}
            != {value.acquisition_group_id for value in carrier.acquisition_groups}
        ):
            raise ValueError("observation order production topology differs from its 256/512 roster")
        source = ProtocolStepTemplate(
            step_id="matrix-observation-order-source-prototype",
            stage=ScientificStage.PREPARE,
            capability_key=OBSERVATION_ORDER_SOURCE_CAPABILITY.capability_key,
            capability_version=OBSERVATION_ORDER_SOURCE_CAPABILITY.capability_version,
            config=self._config_ref(source_config, OBSERVATION_ORDER_SOURCE_CAPABILITY),
            dependency_step_ids=(),
            outputs=tuple(
                sorted(
                    (
                        self._json_output("paired-history-result", MatrixObservationOrderPairedHistoryResult.SCHEMA),
                        OutputTemplate(
                            output_id="paired-native-history",
                            payload_schema=OBSERVATION_ORDER_HDF5_SCHEMA,
                            profile=ArtifactProfile.AUDITED_HDF5,
                            media_type=OBSERVATION_ORDER_HDF5_MEDIA_TYPE,
                            filename_suffix=".h5",
                        ),
                        self._json_output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=OBSERVATION_ORDER_SOURCE_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=replace(
                OBSERVATION_ORDER_SOURCE_CAPABILITY.resource_ceiling,
                wall_time_seconds=80,
            ),
            resource_lock_ids=(),
            barrier=BarrierKind.NONE,
            maximum_attempts=1,
            obligation_ids=("matrix-observation-order-paired-history-custody",),
        )
        projection = ProtocolStepTemplate(
            step_id="matrix-observation-order-projection-prototype",
            stage=ScientificStage.TRANSFORM,
            capability_key=OBSERVATION_ORDER_PROJECTION_CAPABILITY.capability_key,
            capability_version=OBSERVATION_ORDER_PROJECTION_CAPABILITY.capability_version,
            config=self._config_ref(projection_config, OBSERVATION_ORDER_PROJECTION_CAPABILITY),
            dependency_step_ids=(),
            outputs=tuple(
                sorted(
                    (
                        self._json_output("geometry-trace", HistoryViewTrace.SCHEMA),
                        self._json_output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=OBSERVATION_ORDER_PROJECTION_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=replace(
                OBSERVATION_ORDER_PROJECTION_CAPABILITY.resource_ceiling,
                wall_time_seconds=10,
            ),
            resource_lock_ids=(),
            barrier=BarrierKind.FREEZE,
            maximum_attempts=1,
            obligation_ids=("matrix-observation-order-dense-geometry-projection",),
        )
        expanded = expand_acquisition_views(
            template=template,
            carrier_fingerprint=carrier.fingerprint(),
            acquisition_groups=carrier.acquisition_groups,
            nested_views=carrier.nested_views,
            source_task_prefix=OBSERVATION_ORDER_SOURCE_TASK_PREFIX,
            projection_task_prefix=carrier.projection_task_prefix,
            source_step_template=source,
            projection_step_template=projection,
        )
        evaluation_ref = self._config_ref(evaluation_config, OBSERVATION_ORDER_EVALUATION_CAPABILITY)
        steps = tuple(
            replace(
                step,
                stage=ScientificStage.EVALUATE,
                capability_key=OBSERVATION_ORDER_EVALUATION_CAPABILITY.capability_key,
                capability_version=OBSERVATION_ORDER_EVALUATION_CAPABILITY.capability_version,
                config=evaluation_ref,
                outputs=tuple(
                    sorted(
                        (
                            self._json_output("evaluation", MatrixObservationOrderEvaluation.SCHEMA),
                            self._json_output("fine-report", MatrixObservationOrderViewReport.SCHEMA),
                            self._json_output("primary-report", MatrixObservationOrderViewReport.SCHEMA),
                            self._json_output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                            self._json_output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                        ),
                        key=lambda value: value.output_id,
                    )
                ),
                required_permissions=OBSERVATION_ORDER_EVALUATION_CAPABILITY.permissions,
                requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                resource_budget=replace(
                    OBSERVATION_ORDER_EVALUATION_CAPABILITY.resource_ceiling,
                    wall_time_seconds=17_600,
                ),
                barrier=BarrierKind.REVEAL,
                maximum_attempts=1,
                obligation_ids=("matrix-observation-order-single-terminal",),
            )
            if step.step_id == carrier.sealed_evaluation_task_id
            else step
            for step in expanded.steps
        )
        expected_ids = {
            *(f"{OBSERVATION_ORDER_SOURCE_TASK_PREFIX}.{value.acquisition_group_id}" for value in carrier.acquisition_groups),
            *(f"{carrier.projection_task_prefix}.{value.view_id}" for value in carrier.nested_views),
            carrier.sealed_evaluation_task_id,
        }
        if {value.step_id for value in steps} != expected_ids:
            raise ValueError("observation order expanded protocol contains an unexpected stage")
        return replace(
            expanded,
            steps=tuple(sorted(steps, key=lambda value: value.step_id)),
            requests_controller=False,
            nonactuating=True,
        )

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> MatrixObservationOrderPairedHistoryProvider:
        by_type = {type(value): value for value in records}
        expected = {
            ObservationOrderExperimentExtension,
            ObservationOrderSubstrateBinding,
            MatrixObservationOrderPairedHistorySourceConfig,
        }
        if platform_ports or len(by_type) != len(records) or set(by_type) != expected:
            raise ValueError("observation order source provider requires its exact three issued records")
        return MatrixObservationOrderPairedHistoryProvider(
            registry=registry,
            manifest=OBSERVATION_ORDER_SOURCE_CAPABILITY,
            carrier=cast(ObservationOrderExperimentExtension, by_type[ObservationOrderExperimentExtension]),
            substrate=cast(
                ObservationOrderSubstrateBinding,
                by_type[ObservationOrderSubstrateBinding],
            ).substrate,
            source=cast(MatrixObservationOrderPairedHistorySourceConfig, by_type[MatrixObservationOrderPairedHistorySourceConfig]),
            installed_binding=ObjectIdentity.from_record(self.binding.binding_id, self.binding),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    contribution_id="executable-contribution.six-matrix-response-config",
    contribution_version="1.0.0",
    bindings=tuple(
        sorted(
            (
                MATRIX_RESPONSE_ARTIFACT_RECONSTRUCTION_BINDING,
                OBSERVATION_ORDER_SOURCE_EXECUTABLE_BINDING,
                PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_EXECUTABLE_BINDING,
                REACTIVE_ENTRANCE_SOURCE_EXECUTABLE_BINDING,
                MATRIX_RESPONSE_SIMULATOR_EXECUTABLE_BINDING,
            ),
            key=lambda value: value.binding_id,
        )
    ),
)
EXECUTABLE_BINDING_FACTORIES = (
    SixMatrixResponseArtifactInventoryCompilerFactory(),
    MatrixObservationOrderPairedHistoryProviderFactory(),
    SixMatrixResponseProspectiveReactiveSourceLawHistoryProviderFactory(),
    SixMatrixResponseReactiveEntrancePrecursorProviderFactory(),
    SixMatrixResponseSixMatrixProviderFactory(),
)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = tuple(
    sorted(
        (
            MatrixObservationOrderPairedHistorySourceConfig,
            ObservationOrderSubstrateBinding,
            ObservationOrderExperimentExtension,
            SixMatrixResponseArtifactProfileConfig,
            SixMatrixResponseProspectiveReactiveSourceLawSourceConfig,
            SixMatrixResponseReactiveEntranceSourceConfig,
            SixMatrixResponseSixMatrixSourceConfig,
        ),
        key=lambda value: value.SCHEMA,
    )
)


__all__ = [
    "OBSERVATION_ORDER_CARRIER_DECODER_REGISTRATION",
    "OBSERVATION_ORDER_SOURCE_DECODER_REGISTRATION",
    "OBSERVATION_ORDER_SOURCE_EXECUTABLE_BINDING",
    "OBSERVATION_ORDER_SUBSTRATE_DECODER_REGISTRATION",
    'MatrixObservationOrderPairedHistoryProviderFactory',
    "MATRIX_RESPONSE_ARTIFACT_DECODER_REGISTRATION",
    "MATRIX_RESPONSE_ARTIFACT_RECONSTRUCTION_BINDING",
    "MATRIX_RESPONSE_SOURCE_DECODER_REGISTRATION",
    "REACTIVE_ENTRANCE_SOURCE_DECODER_REGISTRATION",
    "REACTIVE_ENTRANCE_SOURCE_EXECUTABLE_BINDING",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_DECODER_REGISTRATION",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_EXECUTABLE_BINDING",
    "MATRIX_RESPONSE_SIMULATOR_EXECUTABLE_BINDING",
    'SixMatrixResponseArtifactInventoryCompilerFactory',
    'SixMatrixResponseReactiveEntrancePrecursorProviderFactory',
    'SixMatrixResponseProspectiveReactiveSourceLawHistoryProviderFactory',
    'SixMatrixResponseSixMatrixProviderFactory',
    "EXECUTABLE_BINDING_CONTRIBUTION",
    "EXECUTABLE_BINDING_FACTORIES",
    "EXECUTABLE_RECORD_TYPES",
]
