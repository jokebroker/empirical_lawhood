"""Generated factories and strict codecs for parameterised FreeGSNKE."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
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
from empirical_lawhood.runtime.extension_bundles import ExtensionComponentRegistration
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

from .extension_bundle import (
    FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY,
    FREEGSNKE_PARAMETERISED_ACQUISITION_PROVIDER,
    FREEGSNKE_PARAMETERISED_CONFIG_DECODER,
    FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY,
    FREEGSNKE_PARAMETERISED_PROJECTION_CONFIG_DECODER,
    FREEGSNKE_PARAMETERISED_PROJECTION_PROVIDER,
)
from .parameterised_provider import FreeGsnkeParameterisedAcquisitionProvider, FreeGsnkeParameterisedEpisodeResult, FreeGsnkeParameterisedExecutorPort, FreeGsnkeParameterisedProjectionConfig, FreeGsnkeParameterisedProjection, FreeGsnkeParameterisedProjectionProvider, FreeGsnkeParameterisedProviderConfig


def _decoder_registration(
    component: ExtensionComponentRegistration,
    record_type: type[CanonicalRecord],
) -> StudyExtensionDecoderRegistration:
    return StudyExtensionDecoderRegistration(
        registration_id=f"decoder-registration.{component.component_key}",
        decoder_key=component.component_key,
        decoder_version=component.component_version,
        payload_schema=record_type.SCHEMA,
        payload_version=record_type.VERSION,
        config_sha256=hashlib.sha256(
            canonical_json_bytes(
                {
                    "decoder_key": component.component_key,
                    "decoder_version": component.component_version,
                    "mode": "exact-canonical-record",
                    "payload_schema": record_type.SCHEMA,
                }
            )
        ).hexdigest(),
        implementation_sha256=component.implementation_sha256,
        maximum_payload_bytes=64 * 1024 * 1024,
    )


FREEGSNKE_PARAMETERISED_CONFIG_DECODER_REGISTRATION = _decoder_registration(
    FREEGSNKE_PARAMETERISED_CONFIG_DECODER,
    FreeGsnkeParameterisedProviderConfig,
)
FREEGSNKE_PARAMETERISED_PROJECTION_CONFIG_DECODER_REGISTRATION = _decoder_registration(
    FREEGSNKE_PARAMETERISED_PROJECTION_CONFIG_DECODER,
    FreeGsnkeParameterisedProjectionConfig,
)


def _binding(
    *,
    binding_id: str,
    capability: CapabilityManifest,
    provider: ExtensionComponentRegistration,
    config_type: type[CanonicalRecord],
    decoder_component: ExtensionComponentRegistration,
    decoder_registration: StudyExtensionDecoderRegistration,
    authenticated_types: tuple[type[CanonicalRecord], ...] = (),
    port_keys: tuple[str, ...] = (),
) -> ExecutableCapabilityBinding:
    return ExecutableCapabilityBinding(
        binding_id=binding_id,
        capability_key=capability.capability_key,
        capability_version=capability.capability_version,
        capability_implementation_sha256=capability.implementation_sha256,
        role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
        provider_key=provider.component_key,
        provider_version=provider.component_version,
        provider_implementation_sha256=provider.implementation_sha256,
        capability_backed=True,
        discovery_components=tuple(
            sorted(
                (decoder_component, provider),
                key=lambda value: value.registration_id,
            )
        ),
        accepted_profile_types=(),
        accepted_config_types=tuple(
            sorted(
                (
                    ExecutableRecordTypeBinding(value.SCHEMA, value.VERSION)
                    for value in (config_type, *authenticated_types)
                ),
                key=lambda value: value.record_schema,
            )
        ),
        required_issued_payload_schemas=(config_type.SCHEMA,),
        required_authenticated_record_schemas=tuple(
            sorted(value.SCHEMA for value in authenticated_types)
        ),
        codec_registration_identities=(
            ObjectIdentity.from_record(decoder_component.registration_id, decoder_component),
        ),
        issued_decoder_registrations=(decoder_registration,),
        input_schema_ids=capability.input_schema_ids,
        output_schema_ids=capability.output_schema_ids,
        artifact_validator_identities=(),
        required_platform_port_keys=port_keys,
        may_require_active_mount=True,
        may_require_source_qualification=True,
        may_require_network=False,
        may_require_authority=True,
    )


FREEGSNKE_PARAMETERISED_ACQUISITION_EXECUTABLE_BINDING = _binding(
    binding_id="binding.freegsnke-parameterised-acquisition",
    capability=FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY,
    provider=FREEGSNKE_PARAMETERISED_ACQUISITION_PROVIDER,
    config_type=FreeGsnkeParameterisedProviderConfig,
    decoder_component=FREEGSNKE_PARAMETERISED_CONFIG_DECODER,
    decoder_registration=FREEGSNKE_PARAMETERISED_CONFIG_DECODER_REGISTRATION,
    authenticated_types=(ResponseExperimentExtensionSet, ResponseSubstrateBinding),
    port_keys=("freegsnke-parameterised-executor",),
)
FREEGSNKE_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING = _binding(
    binding_id="binding.freegsnke-parameterised-projection",
    capability=FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY,
    provider=FREEGSNKE_PARAMETERISED_PROJECTION_PROVIDER,
    config_type=FreeGsnkeParameterisedProjectionConfig,
    decoder_component=FREEGSNKE_PARAMETERISED_PROJECTION_CONFIG_DECODER,
    decoder_registration=FREEGSNKE_PARAMETERISED_PROJECTION_CONFIG_DECODER_REGISTRATION,
)


@dataclass(frozen=True, slots=True)
class FreeGsnkeParameterisedAcquisitionProviderFactory:
    binding: ExecutableCapabilityBinding = FREEGSNKE_PARAMETERISED_ACQUISITION_EXECUTABLE_BINDING

    def expand_parameterised_protocol(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        template: ProtocolTemplate,
    ) -> ProtocolTemplate:
        """Supply FreeGSNKE episode/view contracts to neutral fan-out."""

        from empirical_lawhood.runtime.linked_campaigns import expand_linked_study_acquisition_views

        extension = next(
            (value for value in records if isinstance(value, ResponseExperimentExtensionSet)),
            None,
        )
        substrate = next(
            (value for value in records if isinstance(value, ResponseSubstrateBinding)),
            None,
        )
        provider_config = next(
            (
                value
                for value in records
                if isinstance(value, FreeGsnkeParameterisedProviderConfig)
            ),
            None,
        )
        projection_config = next(
            (
                value
                for value in records
                if isinstance(value, FreeGsnkeParameterisedProjectionConfig)
            ),
            None,
        )
        if not all(
            value is not None
            for value in (extension, substrate, provider_config, projection_config)
        ):
            raise ValueError("FreeGSNKE protocol expansion lacks its exact adapter records")
        assert isinstance(extension, ResponseExperimentExtensionSet)
        assert isinstance(substrate, ResponseSubstrateBinding)
        assert isinstance(provider_config, FreeGsnkeParameterisedProviderConfig)
        assert isinstance(projection_config, FreeGsnkeParameterisedProjectionConfig)
        if substrate.installed_executable_binding != ObjectIdentity.from_record(
            self.binding.binding_id,
            self.binding,
        ):
            raise ValueError("FreeGSNKE protocol substrate selects another executable binding")

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
                media_type="application/vnd.empirical-lawhood.canonical+json",
                filename_suffix=".canonical.json",
            )

        source = ProtocolStepTemplate(
            step_id="freegsnke-source-prototype",
            stage=ScientificStage.PREPARE,
            capability_key=FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY.capability_key,
            capability_version=FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY.capability_version,
            config=config_ref(provider_config, FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY),
            dependency_step_ids=(),
            outputs=tuple(
                sorted(
                    (
                        output(
                            "episode-result",
                            FreeGsnkeParameterisedEpisodeResult.SCHEMA,
                        ),
                        output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY.resource_ceiling,
            resource_lock_ids=("lock.freegsnke-executor",),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("freegsnke-native-episode",),
        )
        projection = ProtocolStepTemplate(
            step_id="freegsnke-projection-prototype",
            stage=ScientificStage.TRANSFORM,
            capability_key=FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY.capability_key,
            capability_version=FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY.capability_version,
            config=config_ref(projection_config, FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY),
            dependency_step_ids=(),
            outputs=tuple(
                sorted(
                    (
                        output("projection", FreeGsnkeParameterisedProjection.SCHEMA),
                        output("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY.permissions,
            requested_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY.resource_ceiling,
            resource_lock_ids=(),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("freegsnke-view-projection",),
        )
        return expand_linked_study_acquisition_views(
            template=template,
            identification_config=extension.identification_config,
            source_task_prefix=provider_config.source_task_prefix,
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
    ) -> CampaignRuntimeProvider:
        by_type = {type(value): value for value in records}
        if len(by_type) != 3 or set(by_type) != {
            ResponseExperimentExtensionSet,
            ResponseSubstrateBinding,
            FreeGsnkeParameterisedProviderConfig,
        }:
            raise ValueError("FreeGSNKE acquisition factory requires exact records")
        if (
            len(platform_ports) != 1
            or platform_ports[0].port_key != "freegsnke-parameterised-executor"
            or not isinstance(platform_ports[0].port, FreeGsnkeParameterisedExecutorPort)
        ):
            raise ValueError("FreeGSNKE acquisition factory requires one executor port")
        substrate = cast(ResponseSubstrateBinding, by_type[ResponseSubstrateBinding])
        if substrate.installed_executable_binding != ObjectIdentity.from_record(
            self.binding.binding_id,
            self.binding,
        ):
            raise ValueError("FreeGSNKE substrate differs from installed acquisition")
        return FreeGsnkeParameterisedAcquisitionProvider(
            registry=registry,
            manifest=FREEGSNKE_PARAMETERISED_ACQUISITION_CAPABILITY,
            extension_set=cast(
                ResponseExperimentExtensionSet,
                by_type[ResponseExperimentExtensionSet],
            ),
            substrate_binding=substrate,
            config=cast(
                FreeGsnkeParameterisedProviderConfig,
                by_type[FreeGsnkeParameterisedProviderConfig],
            ),
            executor=platform_ports[0].port,
        )


@dataclass(frozen=True, slots=True)
class FreeGsnkeParameterisedProjectionProviderFactory:
    binding: ExecutableCapabilityBinding = FREEGSNKE_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        if (
            len(records) != 1
            or not isinstance(records[0], FreeGsnkeParameterisedProjectionConfig)
            or platform_ports
        ):
            raise ValueError("FreeGSNKE projection factory requires one config")
        return FreeGsnkeParameterisedProjectionProvider(
            registry=registry,
            manifest=FREEGSNKE_PARAMETERISED_PROJECTION_CAPABILITY,
            config=records[0],
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    contribution_id="executable-contribution.freegsnke-parameterised",
    contribution_version="1.0.0",
    bindings=tuple(
        sorted(
            (
                FREEGSNKE_PARAMETERISED_ACQUISITION_EXECUTABLE_BINDING,
                FREEGSNKE_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING,
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
                FreeGsnkeParameterisedAcquisitionProviderFactory(),
                FreeGsnkeParameterisedProjectionProviderFactory(),
            ),
        ),
        key=lambda value: value.binding.binding_id,
    )
)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    FreeGsnkeParameterisedProjectionConfig,
    FreeGsnkeParameterisedProviderConfig,
)


__all__ = [
    "EXECUTABLE_BINDING_CONTRIBUTION",
    "EXECUTABLE_BINDING_FACTORIES",
    "EXECUTABLE_RECORD_TYPES",
    "FREEGSNKE_PARAMETERISED_ACQUISITION_EXECUTABLE_BINDING",
    "FREEGSNKE_PARAMETERISED_CONFIG_DECODER_REGISTRATION",
    "FREEGSNKE_PARAMETERISED_PROJECTION_CONFIG_DECODER_REGISTRATION",
    "FREEGSNKE_PARAMETERISED_PROJECTION_EXECUTABLE_BINDING",
    'FreeGsnkeParameterisedAcquisitionProviderFactory',
    'FreeGsnkeParameterisedProjectionProviderFactory',
]
