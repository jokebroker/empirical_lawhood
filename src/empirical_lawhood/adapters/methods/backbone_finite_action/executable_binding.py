"Executable construction binding for baseline finite-action identification.\n\nThe installed implementation is a profile/config compiler and candidate-evidence\nproducer composition.  It is deliberately not advertised as a\n``CampaignRuntimeProvider`` because no such baseline task-runner implementation\nexists in the current adapter today.\n"

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from empirical_lawhood.adapters.methods.finite_action_identification import FiniteActionIdentificationConfig
from empirical_lawhood.adapters.methods.finite_action_registration import (
    FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
    FINITE_ACTION_PROVIDER_KEY,
    FiniteActionIdentificationComponents,
    compose_finite_action_identification,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableBindingRole, ExecutableCapabilityBinding, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration

from .extension_bundle import (
    FINITE_ACTION_ARTIFACT_VALIDATOR,
    FINITE_ACTION_CONFIG_DECODER,
    FINITE_ACTION_MANIFEST,
    FINITE_ACTION_METHOD,
)


_PUBLISHER_PORT_KEY = "candidate-payload-publisher"
_DECODER_CONFIG_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "decoder_key": FINITE_ACTION_CONFIG_DECODER.component_key,
            "decoder_version": FINITE_ACTION_CONFIG_DECODER.component_version,
            "mode": "exact-canonical-record",
            "payload_schema": FiniteActionIdentificationConfig.SCHEMA,
        }
    )
).hexdigest()

FINITE_ACTION_DECODER_REGISTRATION = StudyExtensionDecoderRegistration(
    registration_id="decoder-registration.finite-action-identification",
    decoder_key=FINITE_ACTION_CONFIG_DECODER.component_key,
    decoder_version=FINITE_ACTION_CONFIG_DECODER.component_version,
    payload_schema=FiniteActionIdentificationConfig.SCHEMA,
    payload_version=FiniteActionIdentificationConfig.VERSION,
    config_sha256=_DECODER_CONFIG_SHA256,
    implementation_sha256=FINITE_ACTION_CONFIG_DECODER.implementation_sha256,
    maximum_payload_bytes=FINITE_ACTION_MAXIMUM_CONFIG_BYTES,
)

FINITE_ACTION_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.finite-action-identification",
    capability_key=FINITE_ACTION_MANIFEST.capability_key,
    capability_version=FINITE_ACTION_MANIFEST.capability_version,
    capability_implementation_sha256=FINITE_ACTION_MANIFEST.implementation_sha256,
    role=ExecutableBindingRole.PROFILE_COMPILER,
    provider_key=FINITE_ACTION_PROVIDER_KEY,
    provider_version=FINITE_ACTION_MANIFEST.capability_version,
    provider_implementation_sha256=FINITE_ACTION_CONFIG_DECODER.implementation_sha256,
    capability_backed=True,
    discovery_components=tuple(
        sorted(
            (
                FINITE_ACTION_ARTIFACT_VALIDATOR,
                FINITE_ACTION_CONFIG_DECODER,
                FINITE_ACTION_METHOD,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_profile_types=(),
    accepted_config_types=(
        ExecutableRecordTypeBinding(
            record_schema=FiniteActionIdentificationConfig.SCHEMA,
            record_version=FiniteActionIdentificationConfig.VERSION,
        ),
    ),
    required_issued_payload_schemas=(FiniteActionIdentificationConfig.SCHEMA,),
    codec_registration_identities=(
        ObjectIdentity.from_record(
            FINITE_ACTION_CONFIG_DECODER.registration_id,
            FINITE_ACTION_CONFIG_DECODER,
        ),
    ),
    issued_decoder_registrations=(FINITE_ACTION_DECODER_REGISTRATION,),
    input_schema_ids=FINITE_ACTION_MANIFEST.input_schema_ids,
    output_schema_ids=FINITE_ACTION_MANIFEST.output_schema_ids,
    artifact_validator_identities=(
        ObjectIdentity.from_record(
            FINITE_ACTION_ARTIFACT_VALIDATOR.registration_id,
            FINITE_ACTION_ARTIFACT_VALIDATOR,
        ),
    ),
    required_platform_port_keys=(_PUBLISHER_PORT_KEY,),
    may_require_active_mount=True,
    may_require_source_qualification=True,
    may_require_network=False,
    may_require_authority=True,
)


def _exact_config(records: tuple[CanonicalRecord, ...]) -> FiniteActionIdentificationConfig:
    if len(records) != 1 or not isinstance(records[0], FiniteActionIdentificationConfig):
        raise ValueError("finite-action factory requires one exact issued config")
    config = records[0]
    if (
        config.method_key != FINITE_ACTION_EXECUTABLE_BINDING.capability_key
        or config.method_version != FINITE_ACTION_EXECUTABLE_BINDING.capability_version
        or config.implementation_sha256
        != FINITE_ACTION_EXECUTABLE_BINDING.capability_implementation_sha256
    ):
        raise ValueError("finite-action issued config differs from the installed binding")
    return config


def _payload_publisher(platform_ports: tuple[ExecutablePlatformPort, ...]) -> object:
    if tuple(value.port_key for value in platform_ports) != (_PUBLISHER_PORT_KEY,):
        raise ValueError("finite-action factory requires its exact publisher port")
    publisher = platform_ports[0].port
    if not callable(getattr(publisher, "publish_candidate_payload", None)):
        raise TypeError("finite-action publisher port does not satisfy its protocol")
    return publisher


@dataclass(frozen=True, slots=True)
class FiniteActionProfileCompilerFactory:
    """Construct the existing adapter composition without claiming a runner."""

    binding: ExecutableCapabilityBinding = FINITE_ACTION_EXECUTABLE_BINDING

    def build_compiler(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FiniteActionIdentificationComponents:
        config = _exact_config(records)
        publisher = _payload_publisher(platform_ports)
        return compose_finite_action_identification(
            config=config,
            payload_publisher=publisher,  # type: ignore[arg-type]
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    contribution_id="executable-contribution.finite-action",
    contribution_version="1.0.0",
    bindings=(FINITE_ACTION_EXECUTABLE_BINDING,),
)
EXECUTABLE_BINDING_FACTORIES = (FiniteActionProfileCompilerFactory(),)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (FiniteActionIdentificationConfig,)


__all__ = [
    "EXECUTABLE_BINDING_CONTRIBUTION",
    "EXECUTABLE_BINDING_FACTORIES",
    "EXECUTABLE_RECORD_TYPES",
    "FINITE_ACTION_DECODER_REGISTRATION",
    "FINITE_ACTION_EXECUTABLE_BINDING",
    'FiniteActionProfileCompilerFactory',
]
