"""Nonexecuting descriptor assembly for statically installed adapter bindings."""

from hashlib import sha256
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityPermission
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityRegistration,
)
from empirical_lawhood.planning.evidence_profiles import ExperimentObjectiveProfile
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentRegistration, ExtensionComponentKind, ExtensionContributionKind, ExtensionProfileRegistration, ExtensionProfileKind
from empirical_lawhood.runtime.executable_bindings import ExecutableCapabilityBinding, ExecutableBindingRole, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.adapters.composition.protocol_helpers import CANONICAL_MEDIA_TYPE


def component(
    key: str,
    kind: ExtensionComponentKind,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    implementation: str,
    namespace: str,
) -> ExtensionComponentRegistration:
    return ExtensionComponentRegistration(
        f"component.{namespace}.{key}",
        f"{namespace}.{key}",
        "1.0.0",
        kind,
        tuple(sorted(inputs)),
        tuple(sorted(outputs)),
        implementation,
    )


def bundle(
    role: str,
    kind: ExtensionContributionKind,
    capability: CapabilityManifest,
    records: tuple[type[CanonicalRecord], ...],
    hdf5_schema: str | None,
    *,
    namespace: str,
    maximum_config_bytes: int = 4 * 1024**2,
) -> tuple[ExtensionBundleContribution, tuple[ExtensionComponentRegistration, ...]]:
    decoders = tuple(
        component(
            f"{role}.decoder.{index}",
            ExtensionComponentKind.CONFIG_DECODER,
            (record.SCHEMA,),
            (record.SCHEMA,),
            capability.implementation_sha256,
            namespace,
        )
        for index, record in enumerate(records)
    )
    provider = component(
        f"{role}.provider",
        ExtensionComponentKind.RUNTIME_PROVIDER,
        capability.input_schema_ids,
        capability.output_schema_ids,
        capability.implementation_sha256,
        namespace,
    )
    validators = (
        ()
        if hdf5_schema is None
        else (
            component(
                f"{role}.hdf5-validator",
                ExtensionComponentKind.ARTIFACT_VALIDATOR,
                (hdf5_schema,),
                ('empirical-lawhood/runtime/artifact-generic-validation',),
                capability.implementation_sha256,
                namespace,
            ),
        )
    )

    def identity(value: ExtensionComponentRegistration) -> ObjectIdentity:
        return ObjectIdentity.from_record(value.registration_id, value)

    contribution = ExtensionBundleContribution(
        contribution_id=f"extension-contribution.{namespace}.{role}",
        contribution_version="1.0.0",
        kind=kind,
        evidence_profile_registries=(),
        capability_manifests=(capability,),
        candidate_capability_registrations=(
            CandidateCapabilityRegistration(
                capability,
                provider.component_key,
                provider.component_version,
                CANONICAL_MEDIA_TYPE,
                maximum_config_bytes,
            ),
        ),
        dataset_capability_registries=(),
        profile_registrations=(
            ExtensionProfileRegistration(
                f"profile-registration.{namespace}.{role}",
                f"{namespace}.{role}",
                "1.0.0",
                ExtensionProfileKind.OBJECTIVE_TERMINAL,
                ExperimentObjectiveProfile.SCHEMA,
                sha256(ExperimentObjectiveProfile.SCHEMA.encode()).hexdigest(),
                capability.implementation_sha256,
            ),
        ),
        method_registrations=(),
        config_decoders=tuple(
            sorted(
                (identity(value) for value in decoders),
                key=lambda value: value.object_id,
            )
        ),
        artifact_validators=tuple(identity(value) for value in validators),
        runtime_providers=(identity(provider),),
        study_authors=(),
        grants_authority=False,
        embeds_scientific_payload=False,
    )
    return contribution, (*decoders, provider, *validators)


def executable(
    capability: CapabilityManifest,
    components: tuple[ExtensionComponentRegistration, ...],
    records: tuple[type[CanonicalRecord], ...],
) -> ExecutableCapabilityBinding:
    decoders = tuple(
        value
        for value in components
        if value.kind is ExtensionComponentKind.CONFIG_DECODER
    )
    provider = next(
        value
        for value in components
        if value.kind is ExtensionComponentKind.RUNTIME_PROVIDER
    )
    registrations = tuple(
        StudyExtensionDecoderRegistration(
            registration_id=f"decoder-registration.{decoder.component_key}",
            decoder_key=decoder.component_key,
            decoder_version=decoder.component_version,
            payload_schema=record.SCHEMA,
            payload_version=record.VERSION,
            config_sha256=sha256(
                canonical_json_bytes(
                    {
                        "decoder_key": decoder.component_key,
                        "decoder_version": decoder.component_version,
                        "mode": "exact-canonical-record",
                        "payload_schema": record.SCHEMA,
                    }
                )
            ).hexdigest(),
            implementation_sha256=decoder.implementation_sha256,
            maximum_payload_bytes=8 * 1024**2,
        )
        for decoder, record in zip(decoders, records, strict=True)
    )
    return ExecutableCapabilityBinding(
        binding_id=f"binding.{capability.capability_key}",
        capability_key=capability.capability_key,
        capability_version=capability.capability_version,
        capability_implementation_sha256=capability.implementation_sha256,
        role=ExecutableBindingRole.CAMPAIGN_RUNTIME_PROVIDER,
        provider_key=provider.component_key,
        provider_version=provider.component_version,
        provider_implementation_sha256=provider.implementation_sha256,
        capability_backed=True,
        discovery_components=tuple(
            sorted(components, key=lambda value: value.registration_id)
        ),
        accepted_profile_types=(),
        accepted_config_types=tuple(
            sorted(
                (
                    ExecutableRecordTypeBinding(record.SCHEMA, record.VERSION)
                    for record in records
                ),
                key=lambda value: value.record_schema,
            )
        ),
        required_issued_payload_schemas=tuple(
            sorted(record.SCHEMA for record in records)
        ),
        codec_registration_identities=tuple(
            sorted(
                (
                    ObjectIdentity.from_record(value.registration_id, value)
                    for value in decoders
                ),
                key=lambda value: value.object_id,
            )
        ),
        issued_decoder_registrations=tuple(
            sorted(registrations, key=lambda value: value.registration_id)
        ),
        input_schema_ids=capability.input_schema_ids,
        output_schema_ids=capability.output_schema_ids,
        artifact_validator_identities=tuple(
            ObjectIdentity.from_record(value.registration_id, value)
            for value in components
            if value.kind is ExtensionComponentKind.ARTIFACT_VALIDATOR
        ),
        required_platform_port_keys=(),
        may_require_active_mount=True,
        may_require_source_qualification=False,
        may_require_network=False,
        may_require_authority=True,
    )




def manifest(
    role: str,
    config: type[CanonicalRecord],
    kind: CapabilityKind,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    ceiling: EvidenceCeiling,
    access: OutcomeAccess,
    wall_seconds: int,
    *,
    namespace: str,
    resources: ResourceBudget,
    read_permission: CapabilityPermission,
    implementation_id: str,
    conformance_ids: tuple[str, ...],
) -> CapabilityManifest:
    """Assemble a descriptor from explicit family resources and read permissions."""
    permissions = [
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ]
    permissions.append(read_permission)
    return CapabilityManifest(
        capability_key=f"{namespace}.{role}",
        capability_version="1.0.0",
        kind=kind,
        config_schema=config.SCHEMA,
        config_schema_sha256=sha256(config.SCHEMA.encode()).hexdigest(),
        input_schema_ids=tuple(sorted(inputs)),
        output_schema_ids=tuple(sorted(outputs)),
        permissions=tuple(sorted(permissions)),
        maximum_evidence_ceiling=ceiling,
        maximum_outcome_access=access,
        resource_ceiling=resources,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id=f"{namespace}.{role}",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=conformance_ids,
        implementation_sha256=sha256(
            f"{namespace}.{role}.{implementation_id}".encode()
        ).hexdigest(),
    )
