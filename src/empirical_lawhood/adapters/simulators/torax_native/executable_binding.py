"""Non-acquiring executable binding for the registered source-transform seam.

This descriptor can authenticate and select the generic TORAX-native transform
registration from a source-pipeline profile.  It does not implement a source
adapter, open a source, run TORAX, or claim a campaign task runner; those remain
typed readiness requirements for a later source composition.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import cast

from empirical_lawhood.kernel.authority import ResourceBudget, SourceAccessClass
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile
from empirical_lawhood.runtime.datasets import DatasetCapabilityRegistry, DatasetTransformRegistration
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableBindingRole, ExecutableCapabilityBinding, ExecutableFactory, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.profile_compilation import SourceProfileCompilerBinding
from empirical_lawhood.runtime.sources import SourceCapabilityManifest, SourceMode

from .extension_bundle import SOURCE_ARTIFACT_VALIDATOR, SOURCE_CONFIG_DECODER, SOURCE_DATASET_CAPABILITY_REGISTRY, SOURCE_TRANSFORM_MANIFEST, TORAX_SIMULATED_SOURCE_MANIFEST, TORAX_SIMULATED_SOURCE_PROVIDER, TORAX_SOURCE_PROFILE_COMPILER


_MAXIMUM_SOURCE_PROFILE_BYTES = 4 * 1024 * 1024
_DECODER_CONFIG_SHA256 = hashlib.sha256(
    canonical_json_bytes(
        {
            "decoder_key": SOURCE_CONFIG_DECODER.component_key,
            "decoder_version": SOURCE_CONFIG_DECODER.component_version,
            "mode": "exact-canonical-record",
            "payload_schema": SourcePipelineProfile.SCHEMA,
        }
    )
).hexdigest()

SOURCE_PIPELINE_PROFILE_DECODER_REGISTRATION = StudyExtensionDecoderRegistration(
    registration_id="decoder-registration.source-pipeline-profile",
    decoder_key=SOURCE_CONFIG_DECODER.component_key,
    decoder_version=SOURCE_CONFIG_DECODER.component_version,
    payload_schema=SourcePipelineProfile.SCHEMA,
    payload_version=SourcePipelineProfile.VERSION,
    config_sha256=_DECODER_CONFIG_SHA256,
    implementation_sha256=SOURCE_CONFIG_DECODER.implementation_sha256,
    maximum_payload_bytes=_MAXIMUM_SOURCE_PROFILE_BYTES,
)

SOURCE_TRANSFORM_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.source-pipeline-registered-transform",
    capability_key=SOURCE_TRANSFORM_MANIFEST.capability_key,
    capability_version=SOURCE_TRANSFORM_MANIFEST.capability_version,
    capability_implementation_sha256=SOURCE_TRANSFORM_MANIFEST.implementation_sha256,
    role=ExecutableBindingRole.SOURCE_PROVIDER,
    provider_key="source-pipeline.registered-transform-selection",
    provider_version="1.0.0",
    provider_implementation_sha256=SOURCE_TRANSFORM_MANIFEST.implementation_sha256,
    capability_backed=True,
    discovery_components=tuple(
        sorted(
            (SOURCE_ARTIFACT_VALIDATOR, SOURCE_CONFIG_DECODER),
            key=lambda value: value.registration_id,
        )
    ),
    accepted_profile_types=(
        ExecutableRecordTypeBinding(
            record_schema=SourcePipelineProfile.SCHEMA,
            record_version=SourcePipelineProfile.VERSION,
        ),
    ),
    accepted_config_types=(
        ExecutableRecordTypeBinding(
            record_schema=SOURCE_TRANSFORM_MANIFEST.config_schema,
            record_version="1.0.0",
        ),
    ),
    required_issued_payload_schemas=(SourcePipelineProfile.SCHEMA,),
    codec_registration_identities=(
        ObjectIdentity.from_record(
            SOURCE_CONFIG_DECODER.registration_id,
            SOURCE_CONFIG_DECODER,
        ),
    ),
    issued_decoder_registrations=(SOURCE_PIPELINE_PROFILE_DECODER_REGISTRATION,),
    input_schema_ids=SOURCE_TRANSFORM_MANIFEST.input_schema_ids,
    output_schema_ids=SOURCE_TRANSFORM_MANIFEST.output_schema_ids,
    artifact_validator_identities=(
        ObjectIdentity.from_record(
            SOURCE_ARTIFACT_VALIDATOR.registration_id,
            SOURCE_ARTIFACT_VALIDATOR,
        ),
    ),
    required_platform_port_keys=(),
    may_require_active_mount=True,
    may_require_source_qualification=True,
    may_require_network=False,
    may_require_authority=True,
)

TORAX_SIMULATED_SOURCE_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.torax-simulated-source",
    capability_key=TORAX_SIMULATED_SOURCE_MANIFEST.capability_key,
    capability_version=TORAX_SIMULATED_SOURCE_MANIFEST.capability_version,
    capability_implementation_sha256=TORAX_SIMULATED_SOURCE_MANIFEST.implementation_sha256,
    role=ExecutableBindingRole.SOURCE_PROVIDER,
    provider_key=TORAX_SIMULATED_SOURCE_PROVIDER.component_key,
    provider_version=TORAX_SIMULATED_SOURCE_PROVIDER.component_version,
    provider_implementation_sha256=TORAX_SIMULATED_SOURCE_PROVIDER.implementation_sha256,
    capability_backed=True,
    discovery_components=(TORAX_SIMULATED_SOURCE_PROVIDER,),
    accepted_profile_types=(
        ExecutableRecordTypeBinding(
            SourcePipelineProfile.SCHEMA,
            SourcePipelineProfile.VERSION,
        ),
    ),
    accepted_config_types=(
        ExecutableRecordTypeBinding(
            TORAX_SIMULATED_SOURCE_MANIFEST.config_schema,
            "1.0.0",
        ),
    ),
    required_issued_payload_schemas=(),
    codec_registration_identities=(),
    issued_decoder_registrations=(),
    input_schema_ids=TORAX_SIMULATED_SOURCE_MANIFEST.input_schema_ids,
    output_schema_ids=TORAX_SIMULATED_SOURCE_MANIFEST.output_schema_ids,
    artifact_validator_identities=(),
    required_platform_port_keys=(),
    may_require_active_mount=False,
    may_require_source_qualification=True,
    may_require_network=False,
    may_require_authority=True,
)

TORAX_SOURCE_PROFILE_COMPILER_EXECUTABLE_BINDING = ExecutableCapabilityBinding(
    binding_id="binding.torax-source-profile-compiler",
    capability_key=TORAX_SOURCE_PROFILE_COMPILER.component_key,
    capability_version=TORAX_SOURCE_PROFILE_COMPILER.component_version,
    capability_implementation_sha256=TORAX_SOURCE_PROFILE_COMPILER.implementation_sha256,
    role=ExecutableBindingRole.PROFILE_COMPILER,
    provider_key=TORAX_SOURCE_PROFILE_COMPILER.component_key,
    provider_version=TORAX_SOURCE_PROFILE_COMPILER.component_version,
    provider_implementation_sha256=TORAX_SOURCE_PROFILE_COMPILER.implementation_sha256,
    capability_backed=False,
    discovery_components=(TORAX_SOURCE_PROFILE_COMPILER,),
    accepted_profile_types=(),
    accepted_config_types=(),
    required_issued_payload_schemas=(),
    codec_registration_identities=(),
    issued_decoder_registrations=(),
    input_schema_ids=(),
    output_schema_ids=TORAX_SOURCE_PROFILE_COMPILER.output_schema_ids,
    artifact_validator_identities=(),
    required_platform_port_keys=(),
    may_require_active_mount=False,
    may_require_source_qualification=False,
    may_require_network=False,
    may_require_authority=False,
)


@dataclass(frozen=True, slots=True)
class RegisteredSourceTransformSelection:
    """Exact static transform selection; deliberately not a source adapter."""

    profile: SourcePipelineProfile
    transform: DatasetTransformRegistration
    dataset_registry: DatasetCapabilityRegistry
    source_read: bool = False
    executable_runner_installed: bool = False

    def __post_init__(self) -> None:
        if self.source_read or self.executable_runner_installed:
            raise ValueError("static source-transform selection overclaims execution")


@dataclass(frozen=True, slots=True)
class RegisteredSourceTransformFactory:
    """Authenticate profile selection without acquiring or transforming bytes."""

    binding: ExecutableCapabilityBinding = SOURCE_TRANSFORM_EXECUTABLE_BINDING

    def build_source_provider(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> RegisteredSourceTransformSelection:
        if platform_ports:
            raise ValueError("registered source-transform selection accepts no ports")
        if len(records) != 1 or not isinstance(records[0], SourcePipelineProfile):
            raise ValueError("source-transform factory requires one exact source profile")
        profile = records[0]
        transform = SOURCE_DATASET_CAPABILITY_REGISTRY.transforms[0]
        matching_edges = tuple(
            edge
            for edge in profile.transform_edges
            if edge.selection.capability_key == self.binding.capability_key
            and edge.selection.capability_version == self.binding.capability_version
            and edge.selection.implementation_sha256
            == self.binding.capability_implementation_sha256
            and edge.dataset_transform_registry_id == transform.registry_id
            and edge.config.object_schema == SOURCE_TRANSFORM_MANIFEST.config_schema
        )
        if not matching_edges:
            raise ValueError("source profile does not select the installed transform")
        return RegisteredSourceTransformSelection(
            profile=profile,
            transform=transform,
            dataset_registry=SOURCE_DATASET_CAPABILITY_REGISTRY,
        )


@dataclass(frozen=True, slots=True)
class RegisteredToraxSimulatedSourceSelection:
    profile: SourcePipelineProfile
    source_contacted: bool = False


@dataclass(frozen=True, slots=True)
class ToraxSimulatedSourceFactory:
    binding: ExecutableCapabilityBinding = TORAX_SIMULATED_SOURCE_EXECUTABLE_BINDING

    def build_source_provider(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> RegisteredToraxSimulatedSourceSelection:
        if (
            platform_ports
            or len(records) != 1
            or not isinstance(
                records[0],
                SourcePipelineProfile,
            )
        ):
            raise ValueError("TORAX source selection requires one exact profile and no ports")
        profile = records[0]
        selection = profile.source_selection
        if (
            profile.mode.value != "SIMULATED"
            or selection.capability_key != self.binding.capability_key
            or selection.capability_version != self.binding.capability_version
            or selection.implementation_sha256 != self.binding.capability_implementation_sha256
        ):
            raise ValueError("TORAX source profile changes the installed simulation binding")
        return RegisteredToraxSimulatedSourceSelection(profile)


@dataclass(frozen=True, slots=True)
class ToraxSourceProfileCompilerFactory:
    binding: ExecutableCapabilityBinding = TORAX_SOURCE_PROFILE_COMPILER_EXECUTABLE_BINDING

    def build_compiler(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> SourceProfileCompilerBinding:
        if records or platform_ports:
            raise ValueError("TORAX source compiler context is input-free")
        source = SourceCapabilityManifest(
            capability_key=TORAX_SIMULATED_SOURCE_MANIFEST.capability_key,
            capability_version=TORAX_SIMULATED_SOURCE_MANIFEST.capability_version,
            mode=SourceMode.SIMULATED,
            source_access=SourceAccessClass.NONE,
            output_schema_ids=TORAX_SIMULATED_SOURCE_MANIFEST.output_schema_ids,
            maximum_evidence=EvidenceCeiling.MEASUREMENT,
            maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
            deterministic=True,
            implementation_sha256=TORAX_SIMULATED_SOURCE_MANIFEST.implementation_sha256,
            resource_budget=ResourceBudget(4, 8 * 1024**3, 0, 3600, 8 * 1024**3, 8 * 1024**3),
        )
        capabilities = CapabilityRegistry(
            registry_id="capability-registry.torax-source-context",
            capabilities=tuple(
                sorted(
                    (SOURCE_TRANSFORM_MANIFEST, TORAX_SIMULATED_SOURCE_MANIFEST),
                    key=lambda value: value.registry_id,
                )
            ),
        )
        return SourceProfileCompilerBinding(
            binding_id="source-profile-compiler.torax-simulated-source",
            source_manifest=source,
            capability_registry=capabilities,
            dataset_registry=SOURCE_DATASET_CAPABILITY_REGISTRY,
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    contribution_id="executable-contribution.torax-native-source",
    contribution_version="1.0.0",
    bindings=tuple(
        sorted(
            (
                SOURCE_TRANSFORM_EXECUTABLE_BINDING,
                TORAX_SIMULATED_SOURCE_EXECUTABLE_BINDING,
                TORAX_SOURCE_PROFILE_COMPILER_EXECUTABLE_BINDING,
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
                RegisteredSourceTransformFactory(),
                ToraxSimulatedSourceFactory(),
                ToraxSourceProfileCompilerFactory(),
            ),
        ),
        key=lambda value: value.binding.binding_id,
    )
)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (SourcePipelineProfile,)


__all__ = [
    "EXECUTABLE_BINDING_CONTRIBUTION",
    "EXECUTABLE_BINDING_FACTORIES",
    "EXECUTABLE_RECORD_TYPES",
    'RegisteredSourceTransformFactory',
    'RegisteredSourceTransformSelection',
    "SOURCE_PIPELINE_PROFILE_DECODER_REGISTRATION",
    "SOURCE_TRANSFORM_EXECUTABLE_BINDING",
    "TORAX_SIMULATED_SOURCE_EXECUTABLE_BINDING",
    "TORAX_SOURCE_PROFILE_COMPILER_EXECUTABLE_BINDING",
]
