"""Executable factory for the installed confirmatory finite-action runner."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

from empirical_lawhood.adapters.methods.confirmatory_finite_action import ConfirmatoryFiniteActionConfig
from empirical_lawhood.adapters.methods.confirmatory_finite_action_registration import (
    CONFIRMATORY_FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
    ConfirmatoryFiniteActionCampaignRuntimeProvider,
    compose_confirmatory_finite_action,
)
from empirical_lawhood.adapters.methods.finite_action_identification import FiniteActionCandidateScaffold
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.planning.identification_evidence import IdentificationEvidenceProjection
from empirical_lawhood.planning.identification_evidence_extensions import IdentificationEvidenceProjectionExtension
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableBindingRole, ExecutableCapabilityBinding, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.providers import CampaignRuntimeProvider

from .extension_bundle import (
    CONFIRMATORY_FINITE_ACTION_ARTIFACT_VALIDATOR,
    CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER,
    CONFIRMATORY_FINITE_ACTION_MANIFEST,
    CONFIRMATORY_FINITE_ACTION_METHOD,
    CONFIRMATORY_FINITE_ACTION_RUNTIME_PROVIDER,
)


_PUBLISHER_PORT_KEY = "candidate-payload-publisher"
_DECODER_CONFIG_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "decoder_key": CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER.component_key,
            "decoder_version": (CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER.component_version),
            "mode": "exact-canonical-record",
            "payload_schema": ConfirmatoryFiniteActionConfig.SCHEMA,
        }
    )
).hexdigest()

CONFIRMATORY_FINITE_ACTION_DECODER_REGISTRATION = StudyExtensionDecoderRegistration(
    registration_id="decoder-registration.confirmatory-finite-action",
    decoder_key=CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER.component_key,
    decoder_version=CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER.component_version,
    payload_schema=ConfirmatoryFiniteActionConfig.SCHEMA,
    payload_version=ConfirmatoryFiniteActionConfig.VERSION,
    config_sha256=_DECODER_CONFIG_SHA256,
    implementation_sha256=(CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER.implementation_sha256),
    maximum_payload_bytes=CONFIRMATORY_FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
)

CONFIRMATORY_FINITE_ACTION_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.confirmatory-finite-action",
    capability_key=CONFIRMATORY_FINITE_ACTION_MANIFEST.capability_key,
    capability_version=CONFIRMATORY_FINITE_ACTION_MANIFEST.capability_version,
    capability_implementation_sha256=(CONFIRMATORY_FINITE_ACTION_MANIFEST.implementation_sha256),
    role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
    provider_key=CONFIRMATORY_FINITE_ACTION_RUNTIME_PROVIDER.component_key,
    provider_version=CONFIRMATORY_FINITE_ACTION_RUNTIME_PROVIDER.component_version,
    provider_implementation_sha256=(
        CONFIRMATORY_FINITE_ACTION_RUNTIME_PROVIDER.implementation_sha256
    ),
    capability_backed=True,
    discovery_components=tuple(
        sorted(
            (
                CONFIRMATORY_FINITE_ACTION_ARTIFACT_VALIDATOR,
                CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER,
                CONFIRMATORY_FINITE_ACTION_METHOD,
                CONFIRMATORY_FINITE_ACTION_RUNTIME_PROVIDER,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_profile_types=(),
    accepted_config_types=(
        ExecutableRecordTypeBinding(
            record_schema=ConfirmatoryFiniteActionConfig.SCHEMA,
            record_version=ConfirmatoryFiniteActionConfig.VERSION,
        ),
    ),
    required_issued_payload_schemas=(ConfirmatoryFiniteActionConfig.SCHEMA,),
    codec_registration_identities=(
        ObjectIdentity.from_record(
            CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER.registration_id,
            CONFIRMATORY_FINITE_ACTION_CONFIG_DECODER,
        ),
    ),
    issued_decoder_registrations=(CONFIRMATORY_FINITE_ACTION_DECODER_REGISTRATION,),
    input_schema_ids=CONFIRMATORY_FINITE_ACTION_MANIFEST.input_schema_ids,
    output_schema_ids=CONFIRMATORY_FINITE_ACTION_MANIFEST.output_schema_ids,
    artifact_validator_identities=(
        ObjectIdentity.from_record(
            CONFIRMATORY_FINITE_ACTION_ARTIFACT_VALIDATOR.registration_id,
            CONFIRMATORY_FINITE_ACTION_ARTIFACT_VALIDATOR,
        ),
    ),
    required_platform_port_keys=(_PUBLISHER_PORT_KEY,),
    may_require_active_mount=True,
    may_require_source_qualification=True,
    may_require_network=False,
    may_require_authority=True,
    required_authenticated_record_schemas=CONFIRMATORY_FINITE_ACTION_MANIFEST.input_schema_ids,
)


def _records_by_schema(
    records: tuple[CanonicalRecord, ...],
) -> dict[str, CanonicalRecord]:
    by_schema = {value.SCHEMA: value for value in records}
    expected = {
        ConfirmatoryFiniteActionConfig.SCHEMA,
        FiniteActionCandidateScaffold.SCHEMA,
        IdentificationEvidenceProjection.SCHEMA,
        IdentificationEvidenceProjectionExtension.SCHEMA,
    }
    if len(by_schema) != len(records) or set(by_schema) != expected:
        raise ValueError(
            "confirmatory factory requires exact config/projection/companion/scaffold records"
        )
    return by_schema


def _payload_publisher(platform_ports: tuple[ExecutablePlatformPort, ...]) -> object:
    if tuple(value.port_key for value in platform_ports) != (_PUBLISHER_PORT_KEY,):
        raise ValueError("confirmatory factory requires its exact publisher port")
    publisher = platform_ports[0].port
    if not callable(getattr(publisher, "publish_candidate_payload", None)):
        raise TypeError("confirmatory publisher port does not satisfy its protocol")
    return publisher


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionProviderFactory:
    binding: ExecutableCapabilityBinding = CONFIRMATORY_FINITE_ACTION_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CampaignRuntimeProvider:
        named = tuple(
            value
            for value in registry.capabilities
            if value.capability_key == self.binding.capability_key
            and value.capability_version == self.binding.capability_version
            and value.implementation_sha256 == self.binding.capability_implementation_sha256
        )
        if len(named) != 1:
            raise ValueError(
                "confirmatory factory requires one exact manifest in the complete registry"
            )
        values = _records_by_schema(records)
        config = values[ConfirmatoryFiniteActionConfig.SCHEMA]
        projection = values[IdentificationEvidenceProjection.SCHEMA]
        extension = values[IdentificationEvidenceProjectionExtension.SCHEMA]
        scaffold = values[FiniteActionCandidateScaffold.SCHEMA]
        if not isinstance(config, ConfirmatoryFiniteActionConfig):
            raise TypeError("confirmatory factory config has another record type")
        if not isinstance(projection, IdentificationEvidenceProjection):
            raise TypeError("confirmatory factory projection has another record type")
        if not isinstance(extension, IdentificationEvidenceProjectionExtension):
            raise TypeError("confirmatory factory companion has another record type")
        if not isinstance(scaffold, FiniteActionCandidateScaffold):
            raise TypeError("confirmatory factory scaffold has another record type")
        if (
            config.method_key != self.binding.capability_key
            or config.method_version != self.binding.capability_version
            or config.implementation_sha256 != self.binding.capability_implementation_sha256
        ):
            raise ValueError("confirmatory issued config differs from the installed binding")
        components = compose_confirmatory_finite_action(
            config=config,
            payload_publisher=_payload_publisher(platform_ports),  # type: ignore[arg-type]
        )
        components = replace(components, capability_registry=registry)
        return ConfirmatoryFiniteActionCampaignRuntimeProvider(
            components=components,
            config=config,
            projection=projection,
            extension=extension,
            scaffold=scaffold,
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    contribution_id="executable-contribution.confirmatory-replication",
    contribution_version="1.0.0",
    bindings=(CONFIRMATORY_FINITE_ACTION_EXECUTABLE_BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (ConfirmatoryFiniteActionProviderFactory(),)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (ConfirmatoryFiniteActionConfig,)


__all__ = [
    "CONFIRMATORY_FINITE_ACTION_DECODER_REGISTRATION",
    "CONFIRMATORY_FINITE_ACTION_EXECUTABLE_BINDING",
    'ConfirmatoryFiniteActionProviderFactory',
    "EXECUTABLE_BINDING_CONTRIBUTION",
    "EXECUTABLE_BINDING_FACTORIES",
    "EXECUTABLE_RECORD_TYPES",
]
