"""Import-safe Six-matrix response simulator/config discovery; no simulator is constructed here."""

from __future__ import annotations

from hashlib import sha256
from typing import TypeVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.evidence_profiles import ExperimentObjectiveProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
)
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.extension_bundles import ExtensionBundleContribution, ExtensionComponentKind, ExtensionComponentRegistration, ExtensionContributionKind, ExtensionProfileKind, ExtensionProfileRegistration

from .artifact_inventory import SixMatrixResponseCompiledHDF5Inventory
from .contracts import SixMatrixResponseArtifactProfileConfig, SixMatrixResponseSixMatrixSourceConfig
from .simulation import SixMatrixResponseEpisodeRequest, SixMatrixResponseEpisodeResult
from .reactive_entrance import REACTIVE_ENTRANCE_HDF5_SCHEMA, SixMatrixResponseReactiveEntrancePrecursorResult, SixMatrixResponseReactiveEntranceSourceConfig
from .reactive_entrance_provider import REACTIVE_ENTRANCE_SOURCE_CAPABILITY_KEY, REACTIVE_ENTRANCE_SOURCE_CAPABILITY_VERSION
from .prospective_reactive_source import SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA, SixMatrixResponseProspectiveReactiveSourceLawHistoryResult, SixMatrixResponseProspectiveReactiveSourceLawSourceConfig
from .prospective_reactive_source_provider import PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY_KEY, PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY_VERSION
from .observation_order import MatrixObservationOrderPairedHistoryResult, MatrixObservationOrderPairedHistorySourceConfig, OBSERVATION_ORDER_HDF5_SCHEMA
from .observation_order_provider import OBSERVATION_ORDER_CAPABILITY_VERSION, OBSERVATION_ORDER_SOURCE_CAPABILITY_KEY
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope


_IMPLEMENTATION_SHA256 = sha256(b"six-matrix-response:config-reconstruction").hexdigest()
MATRIX_RESPONSE_SIMULATOR_IMPLEMENTATION_SHA256 = sha256(
    b"six-matrix-response-action-gradient-baoab-checkpoint-receiver"
).hexdigest()
REACTIVE_ENTRANCE_SOURCE_IMPLEMENTATION_SHA256 = sha256(
    b"matrix-response-reactive-entrance-step-816-neutral-precursor-provider"
).hexdigest()
PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_IMPLEMENTATION_SHA256 = sha256(
    b"matrix-response-prospective-reactive-source-law-complete-history-routing-before-future-provider"
).hexdigest()
OBSERVATION_ORDER_SOURCE_IMPLEMENTATION_SHA256 = sha256(
    b"matrix-observation-order-paired-same-driver-history-provider"
).hexdigest()
_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


SIX_MATRIX_RESPONSE_CAPABILITY = CapabilityManifest(
    capability_key="six-matrix-response-episode",
    capability_version="1.0.0",
    kind=CapabilityKind.SIMULATOR,
    config_schema=SixMatrixResponseSixMatrixSourceConfig.SCHEMA,
    config_schema_sha256=sha256(SixMatrixResponseSixMatrixSourceConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(SixMatrixResponseEpisodeRequest.SCHEMA,),
    output_schema_ids=(SixMatrixResponseEpisodeResult.SCHEMA,),
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(
        cpu_cores=1,
        memory_bytes=4 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=86_400,
        source_scan_bytes=0,
        output_bytes=512 * 1024**2,
    ),
    deterministic=False,
    seed_required=True,
    language_id="python",
    runtime_id="six-matrix-response-native",
    requires_clean_commit=True,
    requires_active_mount=False,
    requires_network=False,
    conformance_check_ids=(
        "explicit-original-full-seed-pcg64dxsm-stream",
        "four-stage-native-action-ledger",
        "hermitian-complex128-state",
        "no-scientific-verdict",
        "state-action-rng-exact-restart",
    ),
    implementation_sha256=MATRIX_RESPONSE_SIMULATOR_IMPLEMENTATION_SHA256,
)

REACTIVE_ENTRANCE_SOURCE_CAPABILITY = CapabilityManifest(
    capability_key=REACTIVE_ENTRANCE_SOURCE_CAPABILITY_KEY,
    capability_version=REACTIVE_ENTRANCE_SOURCE_CAPABILITY_VERSION,
    kind=CapabilityKind.SIMULATOR,
    config_schema=SixMatrixResponseReactiveEntranceSourceConfig.SCHEMA,
    config_schema_sha256=sha256(SixMatrixResponseReactiveEntranceSourceConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(SixMatrixResponseReactiveEntranceSourceConfig.SCHEMA,),
    output_schema_ids=tuple(
        sorted(
            (
                REACTIVE_ENTRANCE_HDF5_SCHEMA,
                SixMatrixResponseReactiveEntrancePrecursorResult.SCHEMA,
                LinkedCampaignStageEnvelope.SCHEMA,
            )
        )
    ),
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(
        cpu_cores=1,
        memory_bytes=4 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=7_200,
        source_scan_bytes=32 * 1024**2,
        output_bytes=32 * 1024**2,
    ),
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id="matrix-response-reactive-entrance-six-matrix-native",
    requires_clean_commit=True,
    requires_active_mount=True,
    requires_network=False,
    conformance_check_ids=(
        "complete-phase-space-and-innovation-custody",
        "exact-step-816-source",
        "neutral-native-acquisition-request",
        "same-driver-brownian-bridge-subset",
        "zero-scientific-verdict",
    ),
    implementation_sha256=REACTIVE_ENTRANCE_SOURCE_IMPLEMENTATION_SHA256,
)

PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY = CapabilityManifest(
    capability_key=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY_KEY,
    capability_version=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY_VERSION,
    kind=CapabilityKind.SIMULATOR,
    config_schema=SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.SCHEMA,
    config_schema_sha256=sha256(SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.SCHEMA,),
    output_schema_ids=tuple(sorted((SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA, SixMatrixResponseProspectiveReactiveSourceLawHistoryResult.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA))),
    permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS, CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS),
    maximum_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(cpu_cores=1, memory_bytes=8 * 1024**3, gpu_devices=0, wall_time_seconds=14_400, source_scan_bytes=16 * 1024**2, output_bytes=1024**3),
    deterministic=True, seed_required=False, language_id="python", runtime_id="matrix-response-prospective-reactive-source-law-six-matrix",
    requires_clean_commit=True, requires_active_mount=True, requires_network=False,
    conformance_check_ids=("complete-independent-history", "real-neutral-acquisition-request", "routing-digest-before-future-rng", "same-driver-fine-subset", "zero-scientific-terminal"),
    implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_IMPLEMENTATION_SHA256,
)

OBSERVATION_ORDER_SOURCE_CAPABILITY = CapabilityManifest(
    capability_key=OBSERVATION_ORDER_SOURCE_CAPABILITY_KEY,
    capability_version=OBSERVATION_ORDER_CAPABILITY_VERSION,
    kind=CapabilityKind.SIMULATOR,
    config_schema=MatrixObservationOrderPairedHistorySourceConfig.SCHEMA,
    config_schema_sha256=sha256(MatrixObservationOrderPairedHistorySourceConfig.SCHEMA.encode()).hexdigest(),
    input_schema_ids=(MatrixObservationOrderPairedHistorySourceConfig.SCHEMA,),
    output_schema_ids=tuple(
        sorted(
            (
                OBSERVATION_ORDER_HDF5_SCHEMA,
                MatrixObservationOrderPairedHistoryResult.SCHEMA,
                LinkedCampaignStageEnvelope.SCHEMA,
            )
        )
    ),
    permissions=(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    ),
    maximum_evidence_ceiling=EvidenceCeiling.ORDER_RELATION,
    maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    resource_ceiling=ResourceBudget(
        cpu_cores=1,
        memory_bytes=2 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=43_200,
        source_scan_bytes=0,
        output_bytes=16 * 1024**2,
    ),
    deterministic=True,
    seed_required=False,
    language_id="python",
    runtime_id="matrix-observation-order-six-matrix-native",
    requires_clean_commit=True,
    requires_active_mount=True,
    requires_network=False,
    conformance_check_ids=(
        "complete-four-family-history",
        "one-acquisition-group-paired-views",
        "same-driver-brownian-bridge",
        "zero-scientific-verdict",
    ),
    implementation_sha256=OBSERVATION_ORDER_SOURCE_IMPLEMENTATION_SHA256,
)


def _component(
    record_type: type[_RecordT],
    suffix: str,
    kind: ExtensionComponentKind,
) -> ExtensionComponentRegistration:
    return ExtensionComponentRegistration(
        registration_id=f"component.six-matrix-response.{suffix}",
        component_key=f"six-matrix-response-{suffix}",
        component_version="1.0.0",
        kind=kind,
        input_schema_ids=(record_type.SCHEMA,),
        output_schema_ids=(record_type.SCHEMA,),
        implementation_sha256=_IMPLEMENTATION_SHA256,
    )


MATRIX_RESPONSE_ARTIFACT_CONFIG_DECODER = _component(
    SixMatrixResponseArtifactProfileConfig,
    "artifact-profile-config-decoder",
    ExtensionComponentKind.CONFIG_DECODER,
)
MATRIX_RESPONSE_ARTIFACT_CONFIG_RECONSTRUCTOR = ExtensionComponentRegistration(
    registration_id="component.six-matrix-response.artifact-profile-config-reconstructor",
    component_key="six-matrix-response-artifact-profile-config-reconstructor",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=(SixMatrixResponseArtifactProfileConfig.SCHEMA,),
    output_schema_ids=(SixMatrixResponseCompiledHDF5Inventory.SCHEMA,),
    implementation_sha256=_IMPLEMENTATION_SHA256,
)
MATRIX_RESPONSE_SOURCE_CONFIG_DECODER = _component(
    SixMatrixResponseSixMatrixSourceConfig,
    "six-matrix-source-config-decoder",
    ExtensionComponentKind.CONFIG_DECODER,
)
REACTIVE_ENTRANCE_SOURCE_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id="component.matrix-response-reactive-entrance.source-config-decoder",
    component_key="matrix-response-reactive-entrance-source-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(SixMatrixResponseReactiveEntranceSourceConfig.SCHEMA,),
    output_schema_ids=(SixMatrixResponseReactiveEntranceSourceConfig.SCHEMA,),
    implementation_sha256=REACTIVE_ENTRANCE_SOURCE_IMPLEMENTATION_SHA256,
)
MATRIX_RESPONSE_EPISODE_VALIDATOR = ExtensionComponentRegistration(
    registration_id="component.six-matrix-response.six-matrix-episode-result-validator",
    component_key="six-matrix-response-episode-result-validator",
    component_version="1.0.0",
    kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
    input_schema_ids=(SixMatrixResponseEpisodeResult.SCHEMA,),
    output_schema_ids=('empirical-lawhood/runtime/artifact-generic-validation',),
    implementation_sha256=MATRIX_RESPONSE_SIMULATOR_IMPLEMENTATION_SHA256,
)
MATRIX_RESPONSE_RUNTIME_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.six-matrix-response.six-matrix-runtime-provider",
    component_key="six-matrix-response-runtime-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=(SixMatrixResponseEpisodeRequest.SCHEMA,),
    output_schema_ids=(SixMatrixResponseEpisodeResult.SCHEMA,),
    implementation_sha256=MATRIX_RESPONSE_SIMULATOR_IMPLEMENTATION_SHA256,
)
REACTIVE_ENTRANCE_HDF5_VALIDATOR = ExtensionComponentRegistration(
    registration_id="component.matrix-response-reactive-entrance.precursor-hdf5-validator",
    component_key="matrix-response-reactive-entrance-precursor-hdf5-validator",
    component_version="1.0.0",
    kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
    input_schema_ids=(REACTIVE_ENTRANCE_HDF5_SCHEMA,),
    output_schema_ids=('empirical-lawhood/runtime/artifact-generic-validation',),
    implementation_sha256=REACTIVE_ENTRANCE_SOURCE_IMPLEMENTATION_SHA256,
)
REACTIVE_ENTRANCE_SOURCE_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.matrix-response-reactive-entrance.precursor-provider",
    component_key="matrix-response-reactive-entrance-precursor-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.input_schema_ids,
    output_schema_ids=REACTIVE_ENTRANCE_SOURCE_CAPABILITY.output_schema_ids,
    implementation_sha256=REACTIVE_ENTRANCE_SOURCE_IMPLEMENTATION_SHA256,
)
PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id="component.matrix-response-prospective-reactive-source-law.source-config-decoder",
    component_key="matrix-response-prospective-reactive-source-law-source-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.SCHEMA,),
    output_schema_ids=(SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.SCHEMA,),
    implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_IMPLEMENTATION_SHA256,
)
PROSPECTIVE_REACTIVE_SOURCE_LAW_HDF5_VALIDATOR = ExtensionComponentRegistration(
    registration_id="component.matrix-response-prospective-reactive-source-law.history-hdf5-validator",
    component_key="matrix-response-prospective-reactive-source-law-history-hdf5-validator",
    component_version="1.0.0",
    kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
    input_schema_ids=(SIX_MATRIX_PROSPECTIVE_SOURCE_LAW_HDF5_SCHEMA,),
    output_schema_ids=('empirical-lawhood/runtime/artifact-generic-validation',),
    implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_IMPLEMENTATION_SHA256,
)
PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.matrix-response-prospective-reactive-source-law.history-provider",
    component_key="matrix-response-prospective-reactive-source-law-history-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.input_schema_ids,
    output_schema_ids=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY.output_schema_ids,
    implementation_sha256=PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_IMPLEMENTATION_SHA256,
)
OBSERVATION_ORDER_SOURCE_CONFIG_DECODER = ExtensionComponentRegistration(
    registration_id="component.matrix-observation-order.source-config-decoder",
    component_key="matrix-observation-order-source-config-decoder",
    component_version="1.0.0",
    kind=ExtensionComponentKind.CONFIG_DECODER,
    input_schema_ids=(MatrixObservationOrderPairedHistorySourceConfig.SCHEMA,),
    output_schema_ids=(MatrixObservationOrderPairedHistorySourceConfig.SCHEMA,),
    implementation_sha256=OBSERVATION_ORDER_SOURCE_IMPLEMENTATION_SHA256,
)
OBSERVATION_ORDER_HDF5_VALIDATOR = ExtensionComponentRegistration(
    registration_id="component.matrix-observation-order.paired-history-hdf5-validator",
    component_key="matrix-observation-order-paired-history-hdf5-validator",
    component_version="1.0.0",
    kind=ExtensionComponentKind.ARTIFACT_VALIDATOR,
    input_schema_ids=(OBSERVATION_ORDER_HDF5_SCHEMA,),
    output_schema_ids=('empirical-lawhood/runtime/artifact-generic-validation',),
    implementation_sha256=OBSERVATION_ORDER_SOURCE_IMPLEMENTATION_SHA256,
)
OBSERVATION_ORDER_SOURCE_PROVIDER = ExtensionComponentRegistration(
    registration_id="component.matrix-observation-order.paired-history-provider",
    component_key="matrix-observation-order-paired-history-provider",
    component_version="1.0.0",
    kind=ExtensionComponentKind.RUNTIME_PROVIDER,
    input_schema_ids=OBSERVATION_ORDER_SOURCE_CAPABILITY.input_schema_ids,
    output_schema_ids=OBSERVATION_ORDER_SOURCE_CAPABILITY.output_schema_ids,
    implementation_sha256=OBSERVATION_ORDER_SOURCE_IMPLEMENTATION_SHA256,
)
OBSERVATION_ORDER_SOURCE_CANDIDATE_REGISTRATION = CandidateCapabilityRegistration(
    OBSERVATION_ORDER_SOURCE_CAPABILITY,
    OBSERVATION_ORDER_SOURCE_PROVIDER.component_key,
    OBSERVATION_ORDER_SOURCE_PROVIDER.component_version,
    "application/vnd.empirical-lawhood.canonical+json",
    4 * 1024**2,
)
OBSERVATION_ORDER_OBSERVATION_SOURCE_PROFILE_REGISTRATION = ExtensionProfileRegistration(
    "profile-registration.matrix-observation-order-observation-source",
    "matrix-observation-order-observation-source",
    "1.0.0",
    ExtensionProfileKind.OBJECTIVE_TERMINAL,
    ExperimentObjectiveProfile.SCHEMA,
    sha256(ExperimentObjectiveProfile.SCHEMA.encode()).hexdigest(),
    OBSERVATION_ORDER_SOURCE_IMPLEMENTATION_SHA256,
)


EXTENSION_BUNDLE_CONTRIBUTION = ExtensionBundleContribution(
    contribution_id="extension-contribution.six-matrix-response-config",
    contribution_version="1.0.0",
    kind=ExtensionContributionKind.SOURCE,
    evidence_profile_registries=(),
    capability_manifests=tuple(
        sorted(
            (
                OBSERVATION_ORDER_SOURCE_CAPABILITY,
                PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY,
                REACTIVE_ENTRANCE_SOURCE_CAPABILITY,
                SIX_MATRIX_RESPONSE_CAPABILITY,
            ),
            key=lambda value: value.registry_id,
        )
    ),
    candidate_capability_registrations=(OBSERVATION_ORDER_SOURCE_CANDIDATE_REGISTRATION,),
    dataset_capability_registries=(),
    profile_registrations=(OBSERVATION_ORDER_OBSERVATION_SOURCE_PROFILE_REGISTRATION,),
    method_registrations=(),
    config_decoders=tuple(
        ObjectIdentity.from_record(value.registration_id, value)
        for value in sorted(
            (
                MATRIX_RESPONSE_ARTIFACT_CONFIG_DECODER,
                OBSERVATION_ORDER_SOURCE_CONFIG_DECODER,
                PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CONFIG_DECODER,
                REACTIVE_ENTRANCE_SOURCE_CONFIG_DECODER,
                MATRIX_RESPONSE_SOURCE_CONFIG_DECODER,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    artifact_validators=tuple(
        ObjectIdentity.from_record(value.registration_id, value)
        for value in sorted(
            (
                OBSERVATION_ORDER_HDF5_VALIDATOR,
                MATRIX_RESPONSE_EPISODE_VALIDATOR,
                PROSPECTIVE_REACTIVE_SOURCE_LAW_HDF5_VALIDATOR,
                REACTIVE_ENTRANCE_HDF5_VALIDATOR,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    runtime_providers=tuple(
        ObjectIdentity.from_record(value.registration_id, value)
        for value in sorted(
            (
                MATRIX_RESPONSE_ARTIFACT_CONFIG_RECONSTRUCTOR,
                OBSERVATION_ORDER_SOURCE_PROVIDER,
                PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_PROVIDER,
                REACTIVE_ENTRANCE_SOURCE_PROVIDER,
                MATRIX_RESPONSE_RUNTIME_PROVIDER,
            ),
            key=lambda value: value.registration_id,
        )
    ),
    study_authors=(),
    grants_authority=False,
    embeds_scientific_payload=False,
)


__all__ = [
    "OBSERVATION_ORDER_HDF5_VALIDATOR",
    "OBSERVATION_ORDER_SOURCE_CAPABILITY",
    "OBSERVATION_ORDER_SOURCE_CONFIG_DECODER",
    "OBSERVATION_ORDER_SOURCE_IMPLEMENTATION_SHA256",
    "OBSERVATION_ORDER_SOURCE_PROVIDER",
    "MATRIX_RESPONSE_ARTIFACT_CONFIG_DECODER",
    "MATRIX_RESPONSE_ARTIFACT_CONFIG_RECONSTRUCTOR",
    "MATRIX_RESPONSE_EPISODE_VALIDATOR",
    "MATRIX_RESPONSE_RUNTIME_PROVIDER",
    "REACTIVE_ENTRANCE_HDF5_VALIDATOR",
    "REACTIVE_ENTRANCE_SOURCE_CAPABILITY",
    "REACTIVE_ENTRANCE_SOURCE_CONFIG_DECODER",
    "REACTIVE_ENTRANCE_SOURCE_IMPLEMENTATION_SHA256",
    "REACTIVE_ENTRANCE_SOURCE_PROVIDER",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_HDF5_VALIDATOR",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CAPABILITY",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_CONFIG_DECODER",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_IMPLEMENTATION_SHA256",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_SOURCE_PROVIDER",
    "MATRIX_RESPONSE_SIMULATOR_IMPLEMENTATION_SHA256",
    "SIX_MATRIX_RESPONSE_CAPABILITY",
    "MATRIX_RESPONSE_SOURCE_CONFIG_DECODER",
    "EXTENSION_BUNDLE_CONTRIBUTION",
]
