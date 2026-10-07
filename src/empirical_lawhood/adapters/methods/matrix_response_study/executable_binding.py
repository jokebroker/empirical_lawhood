"""Issued-record reconstruction bindings for Six-matrix response method-owned configs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableBindingRole, ExecutableCapabilityBinding, ExecutableFactory, ExecutablePlatformPort, ExecutableRecordTypeBinding
from empirical_lawhood.runtime.extension_bundles import ExtensionComponentRegistration
from empirical_lawhood.runtime.study_issue import StudyExtensionDecoderRegistration
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.adapters.composition.matrix_response_study.causal_intersection_residence_design import MatrixResponseCausalIntersectionResidenceStudyConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.reactive_entrance import SixMatrixResponseReactiveEntranceSourceConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.prospective_reactive_source import SixMatrixResponseProspectiveReactiveSourceLawSourceConfig

from .contracts import MatrixResponseLawMethodConfig, MatrixResponsePairedPanelReducerConfig, MatrixResponseRoleEquivarianceConfig, MatrixResponseStructuralFacePlan
from .extension_bundle import OBSERVATION_ORDER_EVALUATION_CAPABILITY, OBSERVATION_ORDER_EVALUATION_CONFIG_DECODER, OBSERVATION_ORDER_EVALUATION_PROVIDER, OBSERVATION_ORDER_PROJECTION_CAPABILITY, OBSERVATION_ORDER_PROJECTION_CONFIG_DECODER, OBSERVATION_ORDER_PROJECTION_PROVIDER, MATRIX_RESPONSE_LAW_CONFIG_DECODER, MATRIX_RESPONSE_LAW_CONFIG_RECONSTRUCTOR, MATRIX_RESPONSE_REDUCER_CONFIG_DECODER, MATRIX_RESPONSE_REDUCER_CONFIG_RECONSTRUCTOR, MATRIX_RESPONSE_ROLE_CONFIG_DECODER, MATRIX_RESPONSE_ROLE_CONFIG_RECONSTRUCTOR, MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_CONFIG_DECODER, MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_CONFIG_RECONSTRUCTOR, MATRIX_RESPONSE_STRUCTURAL_CONFIG_DECODER, MATRIX_RESPONSE_STRUCTURAL_CONFIG_RECONSTRUCTOR, REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY, REACTIVE_ENTRANCE_FINALIZATION_PROVIDER, REACTIVE_ENTRANCE_METHOD_CONFIG_DECODER, REACTIVE_ENTRANCE_PROJECTION_CAPABILITY, REACTIVE_ENTRANCE_PROJECTION_PROVIDER, PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY, PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_PROVIDER, PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CONFIG_DECODER, PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY, PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_PROVIDER, _IMPLEMENTATION_SHA256
from .observation_order import MatrixObservationOrderEvaluationConfig, MatrixObservationOrderProjectionConfig
from .observation_order_provider import MatrixObservationOrderEvaluationProvider, MatrixObservationOrderProjectionProvider
from .reactive_entrance_provider import MatrixResponseReactiveEntranceFinalizationProvider, MatrixResponseReactiveEntranceProjectionProvider
from .reactive_entrance_source import MatrixResponseReactiveEntranceMethodConfig
from .prospective_reactive_source_law import MatrixResponseProspectiveReactiveSourceLawMethodConfig
from .prospective_reactive_source_provider import MatrixResponseProspectiveReactiveSourceLawFinalizationProvider, MatrixResponseProspectiveReactiveSourceLawProjectionProvider


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
        implementation_sha256=component.implementation_sha256,
        maximum_payload_bytes=_MAXIMUM_PAYLOAD_BYTES,
    )


MATRIX_RESPONSE_LAW_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-response-study-law-method-config",
    component=MATRIX_RESPONSE_LAW_CONFIG_DECODER,
    record_type=MatrixResponseLawMethodConfig,
)
MATRIX_RESPONSE_REDUCER_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-response-study-paired-panel-reducer-config",
    component=MATRIX_RESPONSE_REDUCER_CONFIG_DECODER,
    record_type=MatrixResponsePairedPanelReducerConfig,
)
MATRIX_RESPONSE_ROLE_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-response-study-role-equivariance-config",
    component=MATRIX_RESPONSE_ROLE_CONFIG_DECODER,
    record_type=MatrixResponseRoleEquivarianceConfig,
)
MATRIX_RESPONSE_STRUCTURAL_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-response-study-structural-face-plan",
    component=MATRIX_RESPONSE_STRUCTURAL_CONFIG_DECODER,
    record_type=MatrixResponseStructuralFacePlan,
)
MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-response-study-causal-intersection-residence-study-config",
    component=MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_CONFIG_DECODER,
    record_type=MatrixResponseCausalIntersectionResidenceStudyConfig,
)
REACTIVE_ENTRANCE_METHOD_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-response-reactive-entrance-method-config",
    component=REACTIVE_ENTRANCE_METHOD_CONFIG_DECODER,
    record_type=MatrixResponseReactiveEntranceMethodConfig,
)
PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-response-prospective-reactive-source-law-method-config",
    component=PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CONFIG_DECODER,
    record_type=MatrixResponseProspectiveReactiveSourceLawMethodConfig,
)
OBSERVATION_ORDER_PROJECTION_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-observation-order-projection-config",
    component=OBSERVATION_ORDER_PROJECTION_CONFIG_DECODER,
    record_type=MatrixObservationOrderProjectionConfig,
)
OBSERVATION_ORDER_EVALUATION_DECODER_REGISTRATION = _decoder(
    registration_id="decoder-registration.matrix-observation-order-evaluation-config",
    component=OBSERVATION_ORDER_EVALUATION_CONFIG_DECODER,
    record_type=MatrixObservationOrderEvaluationConfig,
)


def _rn0_runtime_binding(
    *,
    binding_id: str,
    capability: CapabilityManifest,
    provider: ExtensionComponentRegistration,
    owns_issued_method_config: bool,
) -> ExecutableCapabilityBinding:
    discovery_components = (
        (REACTIVE_ENTRANCE_METHOD_CONFIG_DECODER, provider)
        if owns_issued_method_config
        else (provider,)
    )
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
                discovery_components,
                key=lambda value: value.registration_id,
            )
        ),
        accepted_profile_types=(),
        accepted_config_types=tuple(
            sorted(
                (
                    ExecutableRecordTypeBinding(
                        MatrixResponseReactiveEntranceMethodConfig.SCHEMA,
                        MatrixResponseReactiveEntranceMethodConfig.VERSION,
                    ),
                    ExecutableRecordTypeBinding(
                        SixMatrixResponseReactiveEntranceSourceConfig.SCHEMA,
                        SixMatrixResponseReactiveEntranceSourceConfig.VERSION,
                    ),
                ),
                key=lambda value: value.record_schema,
            )
        ),
        required_issued_payload_schemas=(
            (MatrixResponseReactiveEntranceMethodConfig.SCHEMA,) if owns_issued_method_config else ()
        ),
        required_authenticated_record_schemas=tuple(
            sorted(
                (
                    SixMatrixResponseReactiveEntranceSourceConfig.SCHEMA,
                    *((MatrixResponseReactiveEntranceMethodConfig.SCHEMA,) if not owns_issued_method_config else ()),
                )
            )
        ),
        codec_registration_identities=(
            (
                ObjectIdentity.from_record(
                    REACTIVE_ENTRANCE_METHOD_CONFIG_DECODER.registration_id,
                    REACTIVE_ENTRANCE_METHOD_CONFIG_DECODER,
                ),
            )
            if owns_issued_method_config
            else ()
        ),
        issued_decoder_registrations=(
            (REACTIVE_ENTRANCE_METHOD_DECODER_REGISTRATION,) if owns_issued_method_config else ()
        ),
        input_schema_ids=capability.input_schema_ids,
        output_schema_ids=capability.output_schema_ids,
        artifact_validator_identities=(),
        required_platform_port_keys=(),
        may_require_active_mount=True,
        may_require_source_qualification=True,
        may_require_network=False,
        may_require_authority=True,
    )


REACTIVE_ENTRANCE_PROJECTION_EXECUTABLE_BINDING = _rn0_runtime_binding(
    binding_id="binding.matrix-response-reactive-entrance-reactive-entrance-projection",
    capability=REACTIVE_ENTRANCE_PROJECTION_CAPABILITY,
    provider=REACTIVE_ENTRANCE_PROJECTION_PROVIDER,
    owns_issued_method_config=True,
)
REACTIVE_ENTRANCE_FINALIZATION_EXECUTABLE_BINDING = _rn0_runtime_binding(
    binding_id="binding.matrix-response-reactive-entrance-source-finalization",
    capability=REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY,
    provider=REACTIVE_ENTRANCE_FINALIZATION_PROVIDER,
    owns_issued_method_config=False,
)


def _psl_runtime_binding(
    *,
    binding_id: str,
    capability: CapabilityManifest,
    provider: ExtensionComponentRegistration,
    owns_issued_method_config: bool,
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
        discovery_components=tuple(sorted(
            ((PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CONFIG_DECODER, provider) if owns_issued_method_config else (provider,)),
            key=lambda value: value.registration_id,
        )),
        accepted_profile_types=(),
        accepted_config_types=tuple(sorted((
            ExecutableRecordTypeBinding(MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA, MatrixResponseProspectiveReactiveSourceLawMethodConfig.VERSION),
            ExecutableRecordTypeBinding(SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.SCHEMA, SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.VERSION),
        ), key=lambda value: value.record_schema)),
        required_issued_payload_schemas=((MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA,) if owns_issued_method_config else ()),
        required_authenticated_record_schemas=tuple(sorted((
            SixMatrixResponseProspectiveReactiveSourceLawSourceConfig.SCHEMA,
            *((MatrixResponseProspectiveReactiveSourceLawMethodConfig.SCHEMA,) if not owns_issued_method_config else ()),
        ))),
        codec_registration_identities=((ObjectIdentity.from_record(PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CONFIG_DECODER.registration_id, PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_CONFIG_DECODER),) if owns_issued_method_config else ()),
        issued_decoder_registrations=((PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_DECODER_REGISTRATION,) if owns_issued_method_config else ()),
        input_schema_ids=capability.input_schema_ids,
        output_schema_ids=capability.output_schema_ids,
        artifact_validator_identities=(), required_platform_port_keys=(),
        may_require_active_mount=True, may_require_source_qualification=True,
        may_require_network=False, may_require_authority=True,
    )


PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_EXECUTABLE_BINDING = _psl_runtime_binding(
    binding_id="binding.matrix-response-prospective-reactive-source-law-causal-source-projection",
    capability=PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY,
    provider=PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_PROVIDER,
    owns_issued_method_config=True,
)
PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_EXECUTABLE_BINDING = _psl_runtime_binding(
    binding_id="binding.matrix-response-prospective-reactive-source-law-source-law-finalization",
    capability=PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY,
    provider=PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_PROVIDER,
    owns_issued_method_config=False,
)


def _observation_order_runtime_binding(
    *,
    binding_id: str,
    capability: CapabilityManifest,
    decoder_component: ExtensionComponentRegistration,
    provider: ExtensionComponentRegistration,
    decoder: StudyExtensionDecoderRegistration,
    record_type: type[CanonicalRecord],
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
            sorted((decoder_component, provider), key=lambda value: value.registration_id)
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
        input_schema_ids=capability.input_schema_ids,
        output_schema_ids=capability.output_schema_ids,
        artifact_validator_identities=(),
        required_platform_port_keys=(),
        may_require_active_mount=True,
        may_require_source_qualification=False,
        may_require_network=False,
        may_require_authority=True,
    )


OBSERVATION_ORDER_PROJECTION_EXECUTABLE_BINDING = _observation_order_runtime_binding(
    binding_id="binding.matrix-observation-order-geometry-projection",
    capability=OBSERVATION_ORDER_PROJECTION_CAPABILITY,
    decoder_component=OBSERVATION_ORDER_PROJECTION_CONFIG_DECODER,
    provider=OBSERVATION_ORDER_PROJECTION_PROVIDER,
    decoder=OBSERVATION_ORDER_PROJECTION_DECODER_REGISTRATION,
    record_type=MatrixObservationOrderProjectionConfig,
)
OBSERVATION_ORDER_EVALUATION_EXECUTABLE_BINDING = _observation_order_runtime_binding(
    binding_id="binding.matrix-observation-order-sealed-evaluation",
    capability=OBSERVATION_ORDER_EVALUATION_CAPABILITY,
    decoder_component=OBSERVATION_ORDER_EVALUATION_CONFIG_DECODER,
    provider=OBSERVATION_ORDER_EVALUATION_PROVIDER,
    decoder=OBSERVATION_ORDER_EVALUATION_DECODER_REGISTRATION,
    record_type=MatrixObservationOrderEvaluationConfig,
)


def _binding(
    *,
    binding_id: str,
    decoder_component: ExtensionComponentRegistration,
    reconstructor_component: ExtensionComponentRegistration,
    decoder: StudyExtensionDecoderRegistration,
    record_type: type[CanonicalRecord],
) -> ExecutableCapabilityBinding:
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
        codec_registration_identities=(
            ObjectIdentity.from_record(decoder_component.registration_id, decoder_component),
        ),
        issued_decoder_registrations=(decoder,),
        input_schema_ids=(record_type.SCHEMA,),
        output_schema_ids=(record_type.SCHEMA,),
        artifact_validator_identities=(),
        required_platform_port_keys=(),
        may_require_active_mount=False,
        may_require_source_qualification=False,
        may_require_network=False,
        may_require_authority=False,
    )


_BINDING_INPUTS = (
    (
        "binding.matrix-response-study-law-method-config-reconstructor",
        MATRIX_RESPONSE_LAW_CONFIG_DECODER,
        MATRIX_RESPONSE_LAW_CONFIG_RECONSTRUCTOR,
        MATRIX_RESPONSE_LAW_DECODER_REGISTRATION,
        MatrixResponseLawMethodConfig,
    ),
    (
        "binding.matrix-response-study-paired-panel-reducer-config-reconstructor",
        MATRIX_RESPONSE_REDUCER_CONFIG_DECODER,
        MATRIX_RESPONSE_REDUCER_CONFIG_RECONSTRUCTOR,
        MATRIX_RESPONSE_REDUCER_DECODER_REGISTRATION,
        MatrixResponsePairedPanelReducerConfig,
    ),
    (
        "binding.matrix-response-study-role-equivariance-config-reconstructor",
        MATRIX_RESPONSE_ROLE_CONFIG_DECODER,
        MATRIX_RESPONSE_ROLE_CONFIG_RECONSTRUCTOR,
        MATRIX_RESPONSE_ROLE_DECODER_REGISTRATION,
        MatrixResponseRoleEquivarianceConfig,
    ),
    (
        "binding.matrix-response-study-causal-intersection-residence-study-config-reconstructor",
        MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_CONFIG_DECODER,
        MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_CONFIG_RECONSTRUCTOR,
        MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_DECODER_REGISTRATION,
        MatrixResponseCausalIntersectionResidenceStudyConfig,
    ),
    (
        "binding.matrix-response-study-structural-face-plan-reconstructor",
        MATRIX_RESPONSE_STRUCTURAL_CONFIG_DECODER,
        MATRIX_RESPONSE_STRUCTURAL_CONFIG_RECONSTRUCTOR,
        MATRIX_RESPONSE_STRUCTURAL_DECODER_REGISTRATION,
        MatrixResponseStructuralFacePlan,
    ),
)
_BINDINGS = tuple(
    sorted(
        (
            _binding(
                binding_id=binding_id,
                decoder_component=decoder_component,
                reconstructor_component=reconstructor_component,
                decoder=decoder,
                record_type=record_type,
            )
            for (
                binding_id,
                decoder_component,
                reconstructor_component,
                decoder,
                record_type,
            ) in _BINDING_INPUTS
        ),
        key=lambda value: value.binding_id,
    )
)


@dataclass(frozen=True, slots=True)
class MatrixResponseExactRecordReconstructionFactory:
    binding: ExecutableCapabilityBinding
    record_type: type[CanonicalRecord]

    def build_compiler(
        self,
        *,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> CanonicalRecord:
        if platform_ports or len(records) != 1 or not isinstance(records[0], self.record_type):
            raise ValueError("Six-matrix response reconstruction requires one exact issued record and no ports")
        return records[0]


@dataclass(frozen=True, slots=True)
class MatrixResponseReactiveEntranceProjectionProviderFactory:
    binding: ExecutableCapabilityBinding = REACTIVE_ENTRANCE_PROJECTION_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> MatrixResponseReactiveEntranceProjectionProvider:
        by_type = {type(value): value for value in records}
        if (
            platform_ports
            or len(by_type) != len(records)
            or set(by_type) != {MatrixResponseReactiveEntranceMethodConfig, SixMatrixResponseReactiveEntranceSourceConfig}
        ):
            raise ValueError("matrix response reactive entrance projection factory requires exact source/method records")
        return MatrixResponseReactiveEntranceProjectionProvider(
            registry=registry,
            manifest=REACTIVE_ENTRANCE_PROJECTION_CAPABILITY,
            source_config=cast(SixMatrixResponseReactiveEntranceSourceConfig, by_type[SixMatrixResponseReactiveEntranceSourceConfig]),
            method_config=cast(MatrixResponseReactiveEntranceMethodConfig, by_type[MatrixResponseReactiveEntranceMethodConfig]),
        )


@dataclass(frozen=True, slots=True)
class MatrixResponseReactiveEntranceFinalizationProviderFactory:
    binding: ExecutableCapabilityBinding = REACTIVE_ENTRANCE_FINALIZATION_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> MatrixResponseReactiveEntranceFinalizationProvider:
        by_type = {type(value): value for value in records}
        if (
            platform_ports
            or len(by_type) != len(records)
            or set(by_type) != {MatrixResponseReactiveEntranceMethodConfig, SixMatrixResponseReactiveEntranceSourceConfig}
        ):
            raise ValueError("matrix response reactive entrance finalizer factory requires exact source/method records")
        return MatrixResponseReactiveEntranceFinalizationProvider(
            registry=registry,
            manifest=REACTIVE_ENTRANCE_FINALIZATION_CAPABILITY,
            source_config=cast(SixMatrixResponseReactiveEntranceSourceConfig, by_type[SixMatrixResponseReactiveEntranceSourceConfig]),
            method_config=cast(MatrixResponseReactiveEntranceMethodConfig, by_type[MatrixResponseReactiveEntranceMethodConfig]),
        )


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveReactiveSourceLawProjectionProviderFactory:
    binding: ExecutableCapabilityBinding = PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> MatrixResponseProspectiveReactiveSourceLawProjectionProvider:
        by_type = {type(value): value for value in records}
        if platform_ports or len(by_type) != len(records) or set(by_type) != {MatrixResponseProspectiveReactiveSourceLawMethodConfig, SixMatrixResponseProspectiveReactiveSourceLawSourceConfig}:
            raise ValueError("matrix response prospective reactive source law projection factory requires exact source/method records")
        return MatrixResponseProspectiveReactiveSourceLawProjectionProvider(
            registry=registry, manifest=PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_CAPABILITY,
            source=cast(SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, by_type[SixMatrixResponseProspectiveReactiveSourceLawSourceConfig]),
            method=cast(MatrixResponseProspectiveReactiveSourceLawMethodConfig, by_type[MatrixResponseProspectiveReactiveSourceLawMethodConfig]),
        )


@dataclass(frozen=True, slots=True)
class MatrixResponseProspectiveReactiveSourceLawFinalizationProviderFactory:
    binding: ExecutableCapabilityBinding = PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> MatrixResponseProspectiveReactiveSourceLawFinalizationProvider:
        by_type = {type(value): value for value in records}
        if platform_ports or len(by_type) != len(records) or set(by_type) != {MatrixResponseProspectiveReactiveSourceLawMethodConfig, SixMatrixResponseProspectiveReactiveSourceLawSourceConfig}:
            raise ValueError("matrix response prospective reactive source law finalizer factory requires exact source/method records")
        return MatrixResponseProspectiveReactiveSourceLawFinalizationProvider(
            registry=registry, manifest=PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_CAPABILITY,
            source=cast(SixMatrixResponseProspectiveReactiveSourceLawSourceConfig, by_type[SixMatrixResponseProspectiveReactiveSourceLawSourceConfig]),
            method=cast(MatrixResponseProspectiveReactiveSourceLawMethodConfig, by_type[MatrixResponseProspectiveReactiveSourceLawMethodConfig]),
        )


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderProjectionProviderFactory:
    binding: ExecutableCapabilityBinding = OBSERVATION_ORDER_PROJECTION_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> MatrixObservationOrderProjectionProvider:
        if platform_ports or len(records) != 1 or not isinstance(
            records[0], MatrixObservationOrderProjectionConfig
        ):
            raise ValueError("observation order projection provider requires its exact issued config")
        return MatrixObservationOrderProjectionProvider(
            registry=registry,
            manifest=OBSERVATION_ORDER_PROJECTION_CAPABILITY,
            config=records[0],
        )


@dataclass(frozen=True, slots=True)
class MatrixObservationOrderEvaluationProviderFactory:
    binding: ExecutableCapabilityBinding = OBSERVATION_ORDER_EVALUATION_EXECUTABLE_BINDING

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> MatrixObservationOrderEvaluationProvider:
        if platform_ports or len(records) != 1 or not isinstance(
            records[0], MatrixObservationOrderEvaluationConfig
        ):
            raise ValueError("observation order evaluator provider requires its exact issued config")
        return MatrixObservationOrderEvaluationProvider(
            registry=registry,
            manifest=OBSERVATION_ORDER_EVALUATION_CAPABILITY,
            config=records[0],
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    contribution_id="executable-contribution.matrix-response-study-method-config",
    contribution_version="1.0.0",
    bindings=tuple(
        sorted(
            (
                *_BINDINGS,
                OBSERVATION_ORDER_EVALUATION_EXECUTABLE_BINDING,
                OBSERVATION_ORDER_PROJECTION_EXECUTABLE_BINDING,
                REACTIVE_ENTRANCE_FINALIZATION_EXECUTABLE_BINDING,
                REACTIVE_ENTRANCE_PROJECTION_EXECUTABLE_BINDING,
                PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_EXECUTABLE_BINDING,
                PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_EXECUTABLE_BINDING,
            ),
            key=lambda value: value.binding_id,
        )
    ),
)
EXECUTABLE_BINDING_FACTORIES: tuple[ExecutableFactory, ...] = tuple(
    sorted(
        (
            *(
                cast(
                    ExecutableFactory,
                    MatrixResponseExactRecordReconstructionFactory(
                        binding,
                        next(
                            record_type
                            for binding_id, _, _, _, record_type in _BINDING_INPUTS
                            if binding_id == binding.binding_id
                        ),
                    ),
                )
                for binding in _BINDINGS
            ),
            cast(ExecutableFactory, MatrixResponseReactiveEntranceFinalizationProviderFactory()),
            cast(ExecutableFactory, MatrixResponseReactiveEntranceProjectionProviderFactory()),
            cast(ExecutableFactory, MatrixResponseProspectiveReactiveSourceLawFinalizationProviderFactory()),
            cast(ExecutableFactory, MatrixResponseProspectiveReactiveSourceLawProjectionProviderFactory()),
            cast(ExecutableFactory, MatrixObservationOrderEvaluationProviderFactory()),
            cast(ExecutableFactory, MatrixObservationOrderProjectionProviderFactory()),
        ),
        key=lambda value: value.binding.binding_id,
    )
)
EXECUTABLE_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = tuple(
    sorted(
        (
            MatrixResponseLawMethodConfig,
            MatrixObservationOrderEvaluationConfig,
            MatrixObservationOrderProjectionConfig,
            MatrixResponseReactiveEntranceMethodConfig,
            MatrixResponseProspectiveReactiveSourceLawMethodConfig,
            MatrixResponsePairedPanelReducerConfig,
            MatrixResponseRoleEquivarianceConfig,
            MatrixResponseCausalIntersectionResidenceStudyConfig,
            MatrixResponseStructuralFacePlan,
        ),
        key=lambda value: value.SCHEMA,
    )
)


__all__ = [
    "OBSERVATION_ORDER_EVALUATION_DECODER_REGISTRATION",
    "OBSERVATION_ORDER_EVALUATION_EXECUTABLE_BINDING",
    "OBSERVATION_ORDER_PROJECTION_DECODER_REGISTRATION",
    "OBSERVATION_ORDER_PROJECTION_EXECUTABLE_BINDING",
    'MatrixObservationOrderEvaluationProviderFactory',
    'MatrixObservationOrderProjectionProviderFactory',
    "MATRIX_RESPONSE_LAW_DECODER_REGISTRATION",
    "REACTIVE_ENTRANCE_FINALIZATION_EXECUTABLE_BINDING",
    "REACTIVE_ENTRANCE_METHOD_DECODER_REGISTRATION",
    "REACTIVE_ENTRANCE_PROJECTION_EXECUTABLE_BINDING",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_FINALIZATION_EXECUTABLE_BINDING",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_METHOD_DECODER_REGISTRATION",
    "PROSPECTIVE_REACTIVE_SOURCE_LAW_PROJECTION_EXECUTABLE_BINDING",
    "MATRIX_RESPONSE_REDUCER_DECODER_REGISTRATION",
    "MATRIX_RESPONSE_ROLE_DECODER_REGISTRATION",
    "MATRIX_RESPONSE_CAUSAL_INTERSECTION_RESIDENCE_DECODER_REGISTRATION",
    "MATRIX_RESPONSE_STRUCTURAL_DECODER_REGISTRATION",
    'MatrixResponseReactiveEntranceFinalizationProviderFactory',
    'MatrixResponseReactiveEntranceProjectionProviderFactory',
    "EXECUTABLE_BINDING_CONTRIBUTION",
    "EXECUTABLE_BINDING_FACTORIES",
    "EXECUTABLE_RECORD_TYPES",
]
