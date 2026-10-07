"""Issued-input bindings for Six-matrix response authoring, prospective evaluation coordination and field access."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableBindingRole, ExecutableCapabilityBinding, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.extension_bundles import ExtensionComponentRegistration
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration

from .contracts import MatrixResponseOuterStudyAuthoringConfig, MatrixResponseProspectiveEvaluationTopology, ProtectedObservableAccessManifest
from .extension_bundle import MATRIX_RESPONSE_PROGRAMME_AUTHOR, MATRIX_RESPONSE_PROGRAMME_CONFIG_DECODER, MATRIX_RESPONSE_PROSPECTIVE_CONFIG_DECODER, MATRIX_RESPONSE_PROSPECTIVE_COORDINATOR, MATRIX_RESPONSE_PROTECTED_CONFIG_DECODER, MATRIX_RESPONSE_PROTECTED_FIELD_PROJECTOR, _IMPLEMENTATION_SHA256
from .study_authoring import MatrixResponseArbitraryChartStudyAuthor
from .prospective import MatrixResponseOuterProspectiveUmbrella, MatrixResponseProspectiveEvaluationCoordinator
from .protected_access import MatrixResponseProtectedFieldProjection, MatrixResponseProtectedFieldProjector, build_matrix_response_study_procedural_blindness_projector


_MAXIMUM_PAYLOAD_BYTES = 4 * 1024 * 1024


def _decoder(
    *,
    registration_id: str,
    component: ExtensionComponentRegistration,
    record_type: type[CanonicalRecord],
) -> StudyExtensionDecoderRegistration:
    return StudyExtensionDecoderRegistration(
        registration_id=registration_id,
        decoder_key=component.component_key,
        decoder_version=component.component_version,
        payload_schema=record_type.SCHEMA,
        payload_version=record_type.VERSION,
        config_sha256=sha256(
            canonical_json_bytes(
                {
                    "decoder_key": component.component_key,
                    "decoder_version": component.component_version,
                    "mode": "exact-canonical-record",
                    "payload_schema": record_type.SCHEMA,
                }
            )
        ).hexdigest(),
        implementation_sha256=_IMPLEMENTATION_SHA256,
        maximum_payload_bytes=_MAXIMUM_PAYLOAD_BYTES,
    )


MATRIX_RESPONSE_PROGRAMME_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.six-matrix-response-outer-programme-authoring-config",
    component=MATRIX_RESPONSE_PROGRAMME_CONFIG_DECODER,
    record_type=MatrixResponseOuterStudyAuthoringConfig,
)
MATRIX_RESPONSE_PROSPECTIVE_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.six-matrix-response-prospective-evaluation-topology",
    component=MATRIX_RESPONSE_PROSPECTIVE_CONFIG_DECODER,
    record_type=MatrixResponseProspectiveEvaluationTopology,
)
MATRIX_RESPONSE_PROTECTED_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.six-matrix-response-protected-observable-access-manifest",
    component=MATRIX_RESPONSE_PROTECTED_CONFIG_DECODER,
    record_type=ProtectedObservableAccessManifest,
)


def _binding(
    *,
    binding_id: str,
    role: ExecutableBindingRole,
    decoder_component: ExtensionComponentRegistration,
    product_component: ExtensionComponentRegistration,
    decoder: StudyExtensionDecoderRegistration,
    record_type: type[CanonicalRecord],
    output_schema: str,
) -> ExecutableCapabilityBinding:
    return ExecutableCapabilityBinding(
        binding_id=binding_id,
        capability_key=product_component.component_key,
        capability_version=product_component.component_version,
        capability_implementation_sha256=_IMPLEMENTATION_SHA256,
        role=role,
        provider_key=product_component.component_key,
        provider_version=product_component.component_version,
        provider_implementation_sha256=_IMPLEMENTATION_SHA256,
        capability_backed=False,
        discovery_components=tuple(
            sorted(
                (decoder_component, product_component),
                key=lambda value: value.registration_id,
            )
        ),
        accepted_profile_types=(),
        accepted_config_types=(
            ExecutableRecordTypeBinding(record_type.SCHEMA, record_type.VERSION),
        ),
        required_issued_payload_schemas=(record_type.SCHEMA,),
        codec_registration_identities=(
            ObjectIdentity.from_record(decoder_component.registration_id, decoder_component),
        ),
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


MATRIX_RESPONSE_PROGRAMME_AUTHOR_BINDING = _binding(
    binding_id="binding.six-matrix-response-arbitrary-chart-programme-author",
    role=ExecutableBindingRole.PROGRAMME_AUTHOR,
    decoder_component=MATRIX_RESPONSE_PROGRAMME_CONFIG_DECODER,
    product_component=MATRIX_RESPONSE_PROGRAMME_AUTHOR,
    decoder=MATRIX_RESPONSE_PROGRAMME_DECODER_REGISTRATION,
    record_type=MatrixResponseOuterStudyAuthoringConfig,
    output_schema=AdmissionControllerStudy.SCHEMA,
)
MATRIX_RESPONSE_PROSPECTIVE_COORDINATOR_BINDING = _binding(
    binding_id="binding.six-matrix-response-prospective-evaluation-coordinator",
    role=ExecutableBindingRole.LINKED_CAMPAIGN_COORDINATOR,
    decoder_component=MATRIX_RESPONSE_PROSPECTIVE_CONFIG_DECODER,
    product_component=MATRIX_RESPONSE_PROSPECTIVE_COORDINATOR,
    decoder=MATRIX_RESPONSE_PROSPECTIVE_DECODER_REGISTRATION,
    record_type=MatrixResponseProspectiveEvaluationTopology,
    output_schema=MatrixResponseOuterProspectiveUmbrella.SCHEMA,
)
MATRIX_RESPONSE_PROTECTED_PROJECTOR_BINDING = _binding(
    binding_id="binding.six-matrix-response-protected-field-projector",
    role=ExecutableBindingRole.PROFILE_COMPILER,
    decoder_component=MATRIX_RESPONSE_PROTECTED_CONFIG_DECODER,
    product_component=MATRIX_RESPONSE_PROTECTED_FIELD_PROJECTOR,
    decoder=MATRIX_RESPONSE_PROTECTED_DECODER_REGISTRATION,
    record_type=ProtectedObservableAccessManifest,
    output_schema=MatrixResponseProtectedFieldProjection.SCHEMA,
)


def _one(
    records: tuple[CanonicalRecord, ...],
    record_type: type[CanonicalRecord],
    platform_ports: tuple[ExecutablePlatformPort, ...],
) -> CanonicalRecord:
    if platform_ports or len(records) != 1 or not isinstance(records[0], record_type):
        raise ValueError("Six-matrix response control factory requires one exact issued record and no ports")
    return records[0]


@dataclass(frozen=True, slots=True)
class MatrixResponseStudyAuthorFactory:
    binding: ExecutableCapabilityBinding = MATRIX_RESPONSE_PROGRAMME_AUTHOR_BINDING

    def build_study_author(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> MatrixResponseArbitraryChartStudyAuthor:
        config = _one(records, MatrixResponseOuterStudyAuthoringConfig, platform_ports)
        assert isinstance(config, MatrixResponseOuterStudyAuthoringConfig)
        return MatrixResponseArbitraryChartStudyAuthor(config)


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveCoordinatorFactory:
    binding: ExecutableCapabilityBinding = MATRIX_RESPONSE_PROSPECTIVE_COORDINATOR_BINDING

    def build_coordinator(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> MatrixResponseProspectiveEvaluationCoordinator:
        topology = _one(records, MatrixResponseProspectiveEvaluationTopology, platform_ports)
        assert isinstance(topology, MatrixResponseProspectiveEvaluationTopology)
        return MatrixResponseProspectiveEvaluationCoordinator(topology)


@dataclass(frozen=True, slots=True)
class MatrixResponseProtectedProjectorFactory:
    binding: ExecutableCapabilityBinding = MATRIX_RESPONSE_PROTECTED_PROJECTOR_BINDING

    def build_compiler(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> MatrixResponseProtectedFieldProjector:
        manifest = _one(records, ProtectedObservableAccessManifest, platform_ports)
        assert isinstance(manifest, ProtectedObservableAccessManifest)
        return build_matrix_response_study_procedural_blindness_projector(manifest)


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    contribution_id="executable-contribution.six-matrix-response-control-compatibility",
    contribution_version="1.0.0",
    bindings=tuple(
        sorted(
            (
                MATRIX_RESPONSE_PROGRAMME_AUTHOR_BINDING,
                MATRIX_RESPONSE_PROSPECTIVE_COORDINATOR_BINDING,
                MATRIX_RESPONSE_PROTECTED_PROJECTOR_BINDING,
            ),
            key=lambda value: value.binding_id,
        )
    ),
)
EXECUTABLE_BINDING_FACTORIES = (
    MatrixResponseStudyAuthorFactory(),
    MatrixResponseProspectiveCoordinatorFactory(),
    MatrixResponseProtectedProjectorFactory(),
)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = tuple(
    sorted(
        (
            MatrixResponseOuterStudyAuthoringConfig,
            MatrixResponseProspectiveEvaluationTopology,
            ProtectedObservableAccessManifest,
        ),
        key=lambda value: value.SCHEMA,
    )
)


__all__ = [
    "MATRIX_RESPONSE_PROGRAMME_AUTHOR_BINDING",
    "MATRIX_RESPONSE_PROGRAMME_DECODER_REGISTRATION",
    "MATRIX_RESPONSE_PROSPECTIVE_COORDINATOR_BINDING",
    "MATRIX_RESPONSE_PROSPECTIVE_DECODER_REGISTRATION",
    "MATRIX_RESPONSE_PROTECTED_DECODER_REGISTRATION",
    "MATRIX_RESPONSE_PROTECTED_PROJECTOR_BINDING",
    "EXECUTABLE_BINDING_CONTRIBUTION",
    "EXECUTABLE_BINDING_FACTORIES",
    "EXECUTABLE_RECORD_TYPES",
]
