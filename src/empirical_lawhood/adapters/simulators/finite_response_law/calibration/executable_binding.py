"""Exact fresh-calibration factories over the existing source and method providers."""

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.extension_bundles import ExtensionComponentKind
from empirical_lawhood.runtime.source_qualification import PredecessorBoundSourceQualificationSubstrateBinding
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.methods.finite_response_law.native_provider import FiniteResponseLawNativeMethodProvider
from empirical_lawhood.adapters.methods.finite_response_law.assigned_native_records import FiniteResponseLawAssignedCalibrationProjectionConfig, FiniteResponseLawAssignedCalibrationEvaluationConfig
from ..assigned_contracts import FiniteResponseLawAssignedCalibrationConfig
from ..protocol import validate_source_records, native_protocol_steps
from ..provider import FiniteResponseLawSourceProvider
from ..executable_binding import SOURCE_BINDING as COMMON_SOURCE_BINDING
from .discovery import NAMESPACE, SOURCE_RECORDS, SOURCE_COMPONENTS, SOURCE_CAPABILITY, PROJECTION_CAPABILITY, PROJECTION_COMPONENTS, EVALUATION_CAPABILITY, EVALUATION_COMPONENTS

_new_source = executable(
    SOURCE_CAPABILITY, SOURCE_COMPONENTS, (FiniteResponseLawAssignedCalibrationConfig,)
)
_shared_schemas = {
    PredecessorBoundSourceQualificationExperiment.SCHEMA,
    PredecessorBoundSourceQualificationSubstrateBinding.SCHEMA,
}
_shared_components = tuple(
    c
    for c in COMMON_SOURCE_BINDING.discovery_components
    if c.kind is ExtensionComponentKind.CONFIG_DECODER
    and set(c.input_schema_ids) <= _shared_schemas
)
SOURCE_BINDING = replace(
    _new_source,
    discovery_components=tuple(
        sorted(
            (*_new_source.discovery_components, *_shared_components),
            key=lambda c: c.registration_id,
        )
    ),
    accepted_config_types=tuple(
        sorted(
            (
                *_new_source.accepted_config_types,
                *(
                    t
                    for t in COMMON_SOURCE_BINDING.accepted_config_types
                    if t.record_schema in _shared_schemas
                ),
            ),
            key=lambda t: t.record_schema,
        )
    ),
    required_issued_payload_schemas=tuple(
        sorted((*_new_source.required_issued_payload_schemas, *_shared_schemas))
    ),
    codec_registration_identities=tuple(
        sorted(
            (
                *_new_source.codec_registration_identities,
                *(
                    ObjectIdentity.from_record(c.registration_id, c)
                    for c in _shared_components
                ),
            ),
            key=lambda i: i.object_id,
        )
    ),
    issued_decoder_registrations=tuple(
        sorted(
            (
                *_new_source.issued_decoder_registrations,
                *(
                    d
                    for d in COMMON_SOURCE_BINDING.issued_decoder_registrations
                    if d.payload_schema in _shared_schemas
                ),
            ),
            key=lambda d: d.registration_id,
        )
    ),
)
PROJECTION_BINDING = executable(
    PROJECTION_CAPABILITY,
    PROJECTION_COMPONENTS,
    (FiniteResponseLawAssignedCalibrationProjectionConfig,),
)
EVALUATION_BINDING = executable(
    EVALUATION_CAPABILITY,
    EVALUATION_COMPONENTS,
    (FiniteResponseLawAssignedCalibrationEvaluationConfig,),
)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationSourceFactory:
    binding: ExecutableCapabilityBinding = SOURCE_BINDING

    def _source_records(
        self, records: tuple[CanonicalRecord, ...]
    ) -> tuple[
        FiniteResponseLawAssignedCalibrationConfig,
        PredecessorBoundSourceQualificationExperiment,
        PredecessorBoundSourceQualificationSubstrateBinding,
    ]:
        by_type = {type(r): r for r in records}
        if (
            self.binding != SOURCE_BINDING
            or len(by_type) != len(records)
            or not set(SOURCE_RECORDS) <= set(by_type)
        ):
            raise ValueError(
                "Fresh calibration source requires its exact decoded records"
            )
        config = cast(
            FiniteResponseLawAssignedCalibrationConfig, by_type[FiniteResponseLawAssignedCalibrationConfig]
        )
        carrier = cast(
            PredecessorBoundSourceQualificationExperiment, by_type[PredecessorBoundSourceQualificationExperiment]
        )
        association = cast(
            PredecessorBoundSourceQualificationSubstrateBinding,
            by_type[PredecessorBoundSourceQualificationSubstrateBinding],
        )
        validate_source_records(
            config,
            carrier,
            association,
            ObjectIdentity.from_record(SOURCE_BINDING.binding_id, SOURCE_BINDING),
        )
        return config, carrier, association

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FiniteResponseLawSourceProvider:
        if platform_ports or len(records) != 3:
            raise ValueError(
                "Fresh calibration has no retained source port or additional inputs"
            )
        config, carrier, _ = self._source_records(records)
        return FiniteResponseLawSourceProvider(registry, SOURCE_CAPABILITY, config, carrier, None)

    def expand_parameterised_protocol(
        self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate
    ) -> ProtocolTemplate:
        if len(records) != 5 or {type(r) for r in records} != {
            *SOURCE_RECORDS,
            FiniteResponseLawAssignedCalibrationProjectionConfig,
            FiniteResponseLawAssignedCalibrationEvaluationConfig,
        }:
            raise ValueError(
                "Fresh calibration protocol requires exact source and method records"
            )
        config, carrier, _ = self._source_records(records)
        projection = next(
            r for r in records if type(r) is FiniteResponseLawAssignedCalibrationProjectionConfig
        )
        evaluation = next(
            r for r in records if type(r) is FiniteResponseLawAssignedCalibrationEvaluationConfig
        )
        return replace(
            template,
            template_id=f"{template.template_id}.{carrier.fingerprint()[:16]}",
            steps=native_protocol_steps(config, carrier, projection, evaluation),
            requires_model_set=False,
            requests_controller=False,
            nonactuating=True,
        )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationMethodFactory:
    binding: ExecutableCapabilityBinding

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> FiniteResponseLawNativeMethodProvider:
        if platform_ports or len(records) != 1:
            raise ValueError(
                "Fresh calibration method requires one exact issued configuration"
            )
        config = records[0]
        if (
            self.binding == PROJECTION_BINDING
            and type(config) is FiniteResponseLawAssignedCalibrationProjectionConfig
        ):
            return FiniteResponseLawNativeMethodProvider(registry, PROJECTION_CAPABILITY, config)
        if (
            self.binding == EVALUATION_BINDING
            and type(config) is FiniteResponseLawAssignedCalibrationEvaluationConfig
        ):
            return FiniteResponseLawNativeMethodProvider(registry, EVALUATION_CAPABILITY, config)
        raise ValueError("Fresh calibration method changes its registered binding")


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    f"executable-contribution.{NAMESPACE}",
    "1.0.0",
    tuple(
        sorted(
            (SOURCE_BINDING, PROJECTION_BINDING, EVALUATION_BINDING),
            key=lambda b: b.binding_id,
        )
    ),
)
EXECUTABLE_BINDING_FACTORIES = (
    FiniteResponseLawCalibrationSourceFactory(),
    FiniteResponseLawCalibrationMethodFactory(PROJECTION_BINDING),
    FiniteResponseLawCalibrationMethodFactory(EVALUATION_BINDING),
)
_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    FiniteResponseLawAssignedCalibrationConfig,
    FiniteResponseLawAssignedCalibrationProjectionConfig,
    FiniteResponseLawAssignedCalibrationEvaluationConfig,
)
EXECUTABLE_RECORD_TYPES = tuple(
    sorted(
        _RECORD_TYPES,
        key=lambda t: t.SCHEMA,
    )
)
