"Installed Gym--TORAX metadata-complete campaign-provider factory and strict codecs."

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import cast

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.planning.response_experiment import ResponseExperimentExtensionSet
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableBindingRole, ExecutableCapabilityBinding, ExecutableFactory, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.response_experiment_ports import ResponseSubstrateBinding
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.providers import CampaignRuntimeProvider
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)

from .extension_bundle import GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER, GYM_TORAX_PARAMETERISED_CAPABILITY, GYM_TORAX_PARAMETERISED_CONFIG_DECODER, GYM_TORAX_PARAMETERISED_EPISODE_VALIDATOR, GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY, GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER, GYM_TORAX_PARAMETERISED_PROJECTION_RUNTIME_PROVIDER, GYM_TORAX_PARAMETERISED_RUNTIME_PROVIDER
from .field_metadata_contracts import GymToraxFieldMetadataNativeEpisode
from .projection import GymToraxNativeProjection, GymToraxParameterisedProjectionConfig, GymToraxParameterisedProjectionProvider, validate_gym_torax_projection_config
from .provider import GYM_TORAX_ACQUISITION_TASK_PREFIX, GymToraxParameterisedCampaignRuntimeProvider
from .source import GymToraxParameterisedAcquisitionManifest, GymToraxParameterisedSourceConfig, validate_gym_torax_parameterised_acquisition_manifest, validate_gym_torax_parameterised_source_config


_MAXIMUM_SOURCE_CONFIG_BYTES = 32 * 1024 * 1024
_MAXIMUM_PROJECTION_CONFIG_BYTES = 16 * 1024 * 1024
_DECODER_CONFIG_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "decoder_key": GYM_TORAX_PARAMETERISED_CONFIG_DECODER.component_key,
            "decoder_version": GYM_TORAX_PARAMETERISED_CONFIG_DECODER.component_version,
            "mode": "exact-canonical-record",
            "payload_schema": GymToraxParameterisedSourceConfig.SCHEMA,
        }
    )
).hexdigest()

GYM_TORAX_PARAMETERISED_CONFIG_DECODER_REGISTRATION = StudyExtensionDecoderRegistration(
    registration_id='decoder-registration.gym-torax-parameterised-source-config',
    decoder_key=GYM_TORAX_PARAMETERISED_CONFIG_DECODER.component_key,
    decoder_version=GYM_TORAX_PARAMETERISED_CONFIG_DECODER.component_version,
    payload_schema=GymToraxParameterisedSourceConfig.SCHEMA,
    payload_version=GymToraxParameterisedSourceConfig.VERSION,
    config_sha256=_DECODER_CONFIG_SHA256,
    implementation_sha256=GYM_TORAX_PARAMETERISED_CONFIG_DECODER.implementation_sha256,
    maximum_payload_bytes=_MAXIMUM_SOURCE_CONFIG_BYTES,
)

_ACQUISITION_MANIFEST_DECODER_CONFIG_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "decoder_key": GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER.component_key,
            "decoder_version": (
                GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER.component_version
            ),
            "mode": "exact-canonical-record",
            "payload_schema": GymToraxParameterisedAcquisitionManifest.SCHEMA,
        }
    )
).hexdigest()

GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER_REGISTRATION = (
    StudyExtensionDecoderRegistration(
        registration_id=('decoder-registration.gym-torax-parameterised-acquisition-manifest'),
        decoder_key=GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER.component_key,
        decoder_version=GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER.component_version,
        payload_schema=GymToraxParameterisedAcquisitionManifest.SCHEMA,
        payload_version=GymToraxParameterisedAcquisitionManifest.VERSION,
        config_sha256=_ACQUISITION_MANIFEST_DECODER_CONFIG_SHA256,
        implementation_sha256=(
            GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER.implementation_sha256
        ),
        maximum_payload_bytes=_MAXIMUM_SOURCE_CONFIG_BYTES,
    )
)

_PROJECTION_DECODER_CONFIG_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "decoder_key": GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER.component_key,
            "decoder_version": (
                GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER.component_version
            ),
            "mode": "exact-canonical-record",
            "payload_schema": GymToraxParameterisedProjectionConfig.SCHEMA,
        }
    )
).hexdigest()

GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER_REGISTRATION = (
    StudyExtensionDecoderRegistration(
        registration_id=('decoder-registration.gym-torax-parameterised-projection-config'),
        decoder_key=GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER.component_key,
        decoder_version=GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER.component_version,
        payload_schema=GymToraxParameterisedProjectionConfig.SCHEMA,
        payload_version=GymToraxParameterisedProjectionConfig.VERSION,
        config_sha256=_PROJECTION_DECODER_CONFIG_SHA256,
        implementation_sha256=(
            GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER.implementation_sha256
        ),
        maximum_payload_bytes=_MAXIMUM_PROJECTION_CONFIG_BYTES,
    )
)

GYM_TORAX_PARAMETERISED_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id='binding.gym-torax-parameterised-runtime-provider',
    capability_key=GYM_TORAX_PARAMETERISED_CAPABILITY.capability_key,
    capability_version=GYM_TORAX_PARAMETERISED_CAPABILITY.capability_version,
    capability_implementation_sha256=GYM_TORAX_PARAMETERISED_CAPABILITY.implementation_sha256,
    role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
    provider_key=GYM_TORAX_PARAMETERISED_RUNTIME_PROVIDER.component_key,
    provider_version=GYM_TORAX_PARAMETERISED_RUNTIME_PROVIDER.component_version,
    provider_implementation_sha256=(GYM_TORAX_PARAMETERISED_RUNTIME_PROVIDER.implementation_sha256),
    capability_backed=True,
    discovery_components=tuple(
        sorted(
            (
                GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER,
                GYM_TORAX_PARAMETERISED_CONFIG_DECODER,
                GYM_TORAX_PARAMETERISED_EPISODE_VALIDATOR,
                GYM_TORAX_PARAMETERISED_RUNTIME_PROVIDER,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_profile_types=(),
    accepted_config_types=tuple(
        sorted(
            (
                ExecutableRecordTypeBinding(
                    ResponseExperimentExtensionSet.SCHEMA,
                    ResponseExperimentExtensionSet.VERSION,
                ),
                ExecutableRecordTypeBinding(
                    ResponseSubstrateBinding.SCHEMA,
                    ResponseSubstrateBinding.VERSION,
                ),
                ExecutableRecordTypeBinding(
                    GymToraxParameterisedAcquisitionManifest.SCHEMA,
                    GymToraxParameterisedAcquisitionManifest.VERSION,
                ),
                ExecutableRecordTypeBinding(
                    GymToraxParameterisedSourceConfig.SCHEMA,
                    GymToraxParameterisedSourceConfig.VERSION,
                ),
            ),
            key=lambda value: value.type_id,
        )
    ),
    required_issued_payload_schemas=tuple(
        sorted(
            (
                GymToraxParameterisedAcquisitionManifest.SCHEMA,
                GymToraxParameterisedSourceConfig.SCHEMA,
            )
        )
    ),
    required_authenticated_record_schemas=tuple(
        sorted((ResponseExperimentExtensionSet.SCHEMA, ResponseSubstrateBinding.SCHEMA))
    ),
    codec_registration_identities=tuple(
        ObjectIdentity.from_record(value.registration_id, value)
        for value in (
            GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER,
            GYM_TORAX_PARAMETERISED_CONFIG_DECODER,
        )
    ),
    issued_decoder_registrations=tuple(
        sorted(
            (
                GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER_REGISTRATION,
                GYM_TORAX_PARAMETERISED_CONFIG_DECODER_REGISTRATION,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    input_schema_ids=GYM_TORAX_PARAMETERISED_CAPABILITY.input_schema_ids,
    output_schema_ids=GYM_TORAX_PARAMETERISED_CAPABILITY.output_schema_ids,
    artifact_validator_identities=(
        ObjectIdentity.from_record(
            GYM_TORAX_PARAMETERISED_EPISODE_VALIDATOR.registration_id,
            GYM_TORAX_PARAMETERISED_EPISODE_VALIDATOR,
        ),
    ),
    required_platform_port_keys=("repository-source-root",),
    may_require_active_mount=False,
    may_require_source_qualification=True,
    may_require_network=False,
    may_require_authority=True,
)

GYM_TORAX_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id='binding.gym-torax-parameterised-projection',
    capability_key=GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY.capability_key,
    capability_version=GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY.capability_version,
    capability_implementation_sha256=(
        GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY.implementation_sha256
    ),
    role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
    provider_key=GYM_TORAX_PARAMETERISED_PROJECTION_RUNTIME_PROVIDER.component_key,
    provider_version=GYM_TORAX_PARAMETERISED_PROJECTION_RUNTIME_PROVIDER.component_version,
    provider_implementation_sha256=(
        GYM_TORAX_PARAMETERISED_PROJECTION_RUNTIME_PROVIDER.implementation_sha256
    ),
    capability_backed=True,
    discovery_components=tuple(
        sorted(
            (
                GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER,
                GYM_TORAX_PARAMETERISED_PROJECTION_RUNTIME_PROVIDER,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_profile_types=(),
    accepted_config_types=(
        ExecutableRecordTypeBinding(
            GymToraxParameterisedProjectionConfig.SCHEMA,
            GymToraxParameterisedProjectionConfig.VERSION,
        ),
    ),
    required_issued_payload_schemas=(GymToraxParameterisedProjectionConfig.SCHEMA,),
    required_authenticated_record_schemas=(),
    codec_registration_identities=(
        ObjectIdentity.from_record(
            GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER.registration_id,
            GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER,
        ),
    ),
    issued_decoder_registrations=(GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER_REGISTRATION,),
    input_schema_ids=GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY.input_schema_ids,
    output_schema_ids=GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY.output_schema_ids,
    artifact_validator_identities=(),
    required_platform_port_keys=(),
    may_require_active_mount=False,
    may_require_source_qualification=False,
    may_require_network=False,
    may_require_authority=True,
)


@dataclass(frozen=True, slots=True)
class GymToraxParameterisedProviderFactory:
    binding: ExecutableCapabilityBinding = GYM_TORAX_PARAMETERISED_EXECUTABLE_BINDING

    def expand_parameterised_protocol(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        template: ProtocolTemplate,
    ) -> ProtocolTemplate:
        """Supply complete Gym episode/view contracts to neutral fan-out."""

        from empirical_lawhood.runtime.linked_campaigns import expand_linked_study_acquisition_views

        extension = next(
            (value for value in records if isinstance(value, ResponseExperimentExtensionSet)),
            None,
        )
        substrate = next(
            (value for value in records if isinstance(value, ResponseSubstrateBinding)),
            None,
        )
        source_config = next(
            (value for value in records if isinstance(value, GymToraxParameterisedSourceConfig)),
            None,
        )
        acquisition_manifest = next(
            (
                value
                for value in records
                if isinstance(value, GymToraxParameterisedAcquisitionManifest)
            ),
            None,
        )
        projection_config = next(
            (
                value
                for value in records
                if isinstance(value, GymToraxParameterisedProjectionConfig)
            ),
            None,
        )
        if not all(
            value is not None
            for value in (
                extension,
                substrate,
                source_config,
                acquisition_manifest,
                projection_config,
            )
        ):
            raise ValueError("Gym protocol expansion lacks its exact adapter records")
        assert isinstance(extension, ResponseExperimentExtensionSet)
        assert isinstance(substrate, ResponseSubstrateBinding)
        assert isinstance(source_config, GymToraxParameterisedSourceConfig)
        assert isinstance(
            acquisition_manifest,
            GymToraxParameterisedAcquisitionManifest,
        )
        assert isinstance(projection_config, GymToraxParameterisedProjectionConfig)
        if substrate.installed_executable_binding != ObjectIdentity.from_record(
            self.binding.binding_id,
            self.binding,
        ):
            raise ValueError("Gym protocol substrate selects another executable binding")
        validate_gym_torax_projection_config(
            config=projection_config,
            extension_set=extension,
            substrate_binding=substrate,
            source_config=source_config,
        )
        validate_gym_torax_parameterised_source_config(
            config=source_config,
            extension_set=extension,
            substrate_binding=substrate,
        )
        validate_gym_torax_parameterised_acquisition_manifest(
            manifest=acquisition_manifest,
            extension_set=extension,
            source_config=source_config,
        )

        def config_ref(
            record: CanonicalRecord,
            manifest: CapabilityManifest,
        ) -> CapabilityConfigRef:
            config_id = cast(str, getattr(record, "config_id"))
            return CapabilityConfigRef(
                config_id=config_id,
                config_schema=record.SCHEMA,
                config_schema_sha256=manifest.config_schema_sha256,
                content_sha256=record.fingerprint(),
                artifact_id=f"config-artifact.{config_id}",
            )

        def output(output_id: str, schema: str) -> OutputTemplate:
            return OutputTemplate(
                output_id=output_id,
                payload_schema=schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                filename_suffix=".canonical.json",
            )

        source = ProtocolStepTemplate(
            step_id="gym-torax-source-prototype",
            stage=ScientificStage.PREPARE,
            capability_key=GYM_TORAX_PARAMETERISED_CAPABILITY.capability_key,
            capability_version=GYM_TORAX_PARAMETERISED_CAPABILITY.capability_version,
            config=config_ref(source_config, GYM_TORAX_PARAMETERISED_CAPABILITY),
            dependency_step_ids=(),
            outputs=tuple(
                sorted(
                    (
                        output("native-episode", GymToraxFieldMetadataNativeEpisode.SCHEMA),
                        output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=GYM_TORAX_PARAMETERISED_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=GYM_TORAX_PARAMETERISED_CAPABILITY.resource_ceiling,
            resource_lock_ids=("lock.gym-torax-runtime",),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("gym-torax-native-episode",),
        )
        projection = ProtocolStepTemplate(
            step_id="gym-torax-projection-prototype",
            stage=ScientificStage.TRANSFORM,
            capability_key=GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY.capability_key,
            capability_version=GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY.capability_version,
            config=config_ref(
                projection_config,
                GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY,
            ),
            dependency_step_ids=(),
            outputs=tuple(
                sorted(
                    (
                        output("native-projection", GymToraxNativeProjection.SCHEMA),
                        output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY.resource_ceiling,
            resource_lock_ids=(),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("gym-torax-native-view-projection",),
        )
        return expand_linked_study_acquisition_views(
            template=template,
            identification_config=extension.identification_config,
            source_task_prefix=GYM_TORAX_ACQUISITION_TASK_PREFIX,
            projection_task_prefix=projection_config.projection_task_prefix,
            source_step_template=source,
            projection_step_template=projection,
        )

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> GymToraxParameterisedCampaignRuntimeProvider:
        by_type = {type(value): value for value in records}
        if len(by_type) != 4 or set(by_type) != {
            ResponseExperimentExtensionSet,
            ResponseSubstrateBinding,
            GymToraxParameterisedAcquisitionManifest,
            GymToraxParameterisedSourceConfig,
        }:
            raise ValueError("Gym provider factory requires exactly four authenticated records")
        if len(platform_ports) != 1 or platform_ports[0].port_key != "repository-source-root":
            raise ValueError("Gym provider factory requires the repository source root port")
        repository_root = platform_ports[0].port
        if not isinstance(repository_root, Path):
            raise TypeError("repository source root port must be a pathlib.Path")
        extension_set = by_type[ResponseExperimentExtensionSet]
        substrate_binding = by_type[ResponseSubstrateBinding]
        source_config = by_type[GymToraxParameterisedSourceConfig]
        acquisition_manifest = by_type[GymToraxParameterisedAcquisitionManifest]
        assert isinstance(extension_set, ResponseExperimentExtensionSet)
        assert isinstance(substrate_binding, ResponseSubstrateBinding)
        assert isinstance(source_config, GymToraxParameterisedSourceConfig)
        assert isinstance(
            acquisition_manifest,
            GymToraxParameterisedAcquisitionManifest,
        )
        if substrate_binding.installed_executable_binding != ObjectIdentity.from_record(
            self.binding.binding_id,
            self.binding,
        ):
            raise ValueError("Gym substrate binding differs from installed executable binding")
        return GymToraxParameterisedCampaignRuntimeProvider(
            registry=registry,
            manifest=GYM_TORAX_PARAMETERISED_CAPABILITY,
            extension_set=extension_set,
            substrate_binding=substrate_binding,
            source_config=source_config,
            acquisition_manifest=acquisition_manifest,
            repository_root=repository_root,
        )


@dataclass(frozen=True, slots=True)
class GymToraxParameterisedProjectionProviderFactory:
    binding: ExecutableCapabilityBinding = GYM_TORAX_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if (
            len(records) != 1
            or not isinstance(records[0], GymToraxParameterisedProjectionConfig)
            or platform_ports
        ):
            raise ValueError("Gym projection factory requires one exact config and no ports")
        return GymToraxParameterisedProjectionProvider(
            registry=registry,
            manifest=GYM_TORAX_PARAMETERISED_PROJECTION_CAPABILITY,
            config=records[0],
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    contribution_id='executable-contribution.gym-torax-parameterised',
    contribution_version="1.0.0",
    bindings=tuple(
        sorted(
            (
                GYM_TORAX_PARAMETERISED_EXECUTABLE_BINDING,
                GYM_TORAX_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING,
            ),
            key=lambda value: value.binding_id,
        )
    ),
)
EXECUTABLE_BINDING_FACTORIES: tuple[ExecutableFactory, ...] = tuple(
    sorted(
        cast(
            tuple[ExecutableFactory, ...],
            (
                GymToraxParameterisedProviderFactory(),
                GymToraxParameterisedProjectionProviderFactory(),
            ),
        ),
        key=lambda value: value.binding.binding_id,
    )
)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    GymToraxParameterisedAcquisitionManifest,
    GymToraxParameterisedProjectionConfig,
    GymToraxParameterisedSourceConfig,
)


__all__ = [
    "EXECUTABLE_BINDING_CONTRIBUTION",
    "EXECUTABLE_BINDING_FACTORIES",
    "EXECUTABLE_RECORD_TYPES",
    "GYM_TORAX_PARAMETERISED_ACQUISITION_MANIFEST_DECODER_REGISTRATION",
    "GYM_TORAX_PARAMETERISED_CONFIG_DECODER_REGISTRATION",
    "GYM_TORAX_PARAMETERISED_EXECUTABLE_BINDING",
    "GYM_TORAX_PARAMETERISED_PROJECTION_CONFIG_DECODER_REGISTRATION",
    "GYM_TORAX_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING",
    'GymToraxParameterisedProjectionProviderFactory',
    'GymToraxParameterisedProviderFactory',
]
