"Finite response-law source factory with exact predecessor-bound carrier codecs and a bounded retained-input port."

from dataclasses import dataclass, replace
from typing import cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.source_qualification import PredecessorBoundSourceQualificationSubstrateBinding
from empirical_lawhood.adapters.composition.discovery import executable
from empirical_lawhood.adapters.methods.finite_response_law.native_records import FiniteResponseLawProjectionConfig, FiniteResponseLawNativeEvaluationConfig
from .contracts import FiniteResponseLawNativeConfig
from .discovery import SOURCE_CAPABILITY, SOURCE_COMPONENTS, SOURCE_RECORDS
from .provider import FiniteResponseLawSourceProvider, FiniteResponseLawRetainedSources, RETAINED_SOURCE_PORT
from .protocol import validate_source_records, native_protocol_steps

SOURCE_BINDING = replace(
    executable(SOURCE_CAPABILITY, SOURCE_COMPONENTS, SOURCE_RECORDS),
    required_platform_port_keys=(RETAINED_SOURCE_PORT,),
)


def native_bindings(
    config: FiniteResponseLawNativeConfig,
) -> tuple[
    ExecutableCapabilityBinding,
    ExecutableCapabilityBinding,
    ExecutableCapabilityBinding,
]:
    from .fresh_contracts import FiniteResponseLawCalibrationConfig
    from .assigned_contracts import FiniteResponseLawAssignedCalibrationConfig, FiniteResponseLawAssignedEvaluationConfig
    from .evaluation_contracts import FiniteResponseLawEvaluationConfig

    if type(config) is FiniteResponseLawAssignedEvaluationConfig:
        from .evaluation import executable_binding as evaluation

        return (
            evaluation.SOURCE_BINDING,
            evaluation.PROJECTION_BINDING,
            evaluation.EVALUATION_BINDING,
        )

    if type(config) is FiniteResponseLawAssignedCalibrationConfig:
        from .calibration import executable_binding as fresh

        return fresh.SOURCE_BINDING, fresh.PROJECTION_BINDING, fresh.EVALUATION_BINDING
    if type(config) is FiniteResponseLawCalibrationConfig:
        raise ValueError(
            "Exposed fixed calibration roots have no installed source binding"
        )
    if type(config) is FiniteResponseLawEvaluationConfig:
        raise ValueError("Exposed fixed evaluation roots have no installed source binding")
    from empirical_lawhood.adapters.methods.finite_response_law.executable_binding import PROJECTION_BINDING, EVALUATION_BINDING

    return SOURCE_BINDING, PROJECTION_BINDING, EVALUATION_BINDING


@dataclass(frozen=True, slots=True)
class FiniteResponseLawSourceFactory:
    binding: ExecutableCapabilityBinding = SOURCE_BINDING

    def _source_records(
        self, records: tuple[CanonicalRecord, ...]
    ) -> tuple[
        FiniteResponseLawNativeConfig,
        PredecessorBoundSourceQualificationExperiment,
        PredecessorBoundSourceQualificationSubstrateBinding,
    ]:
        by_type = {type(r): r for r in records}
        if (
            self.binding != SOURCE_BINDING
            or not set(SOURCE_RECORDS) <= set(by_type)
            or len(by_type) != len(records)
        ):
            raise ValueError(
                "Finite response-law source factory requires its exact decoded record roster"
            )
        config = cast(FiniteResponseLawNativeConfig, by_type[FiniteResponseLawNativeConfig])
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
        if len(records) != 3 or tuple(p.port_key for p in platform_ports) != (
            RETAINED_SOURCE_PORT,
        ):
            raise ValueError(
                "Finite response-law source factory lacks its exact retained source port or issued records"
            )
        config, carrier, _ = self._source_records(records)
        sources = cast(FiniteResponseLawRetainedSources, platform_ports[0].port)
        expected = tuple(
            sorted(
                (
                    a
                    for r in config.retained_predecessors
                    for a in (*r.artifacts, r.task_receipt)
                ),
                key=lambda a: a.artifact_id,
            )
        )
        if sources.artifacts != expected:
            raise ValueError(
                "Finite response-law source port changes its full retained-artifact inventory"
            )
        return FiniteResponseLawSourceProvider(
            registry, SOURCE_CAPABILITY, config, carrier, sources if expected else None
        )

    def expand_parameterised_protocol(
        self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate
    ) -> ProtocolTemplate:
        if len(records) != 5 or {type(r) for r in records} != {
            *SOURCE_RECORDS,
            FiniteResponseLawProjectionConfig,
            FiniteResponseLawNativeEvaluationConfig,
        }:
            raise ValueError(
                "Finite response-law expansion requires its exact source/carrier/method records"
            )
        config, carrier, _ = self._source_records(records)
        projection = next(r for r in records if isinstance(r, FiniteResponseLawProjectionConfig))
        evaluation = next(
            r for r in records if isinstance(r, FiniteResponseLawNativeEvaluationConfig)
        )
        return replace(
            template,
            template_id=f"{template.template_id}.{carrier.fingerprint()[:16]}",
            steps=native_protocol_steps(config, carrier, projection, evaluation),
            requires_model_set=False,
            requests_controller=False,
            nonactuating=True,
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.finite-response-law.source", "1.0.0", (SOURCE_BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (FiniteResponseLawSourceFactory(),)
EXECUTABLE_RECORD_TYPES = tuple(sorted(SOURCE_RECORDS, key=lambda t: t.SCHEMA))
