"Installed source qualification method factories with exact issued configuration decoders."

from dataclasses import dataclass, replace

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.adapters.composition.discovery import executable
from .extension_bundle import DEVELOPMENT_FIT_CAPABILITY, DEVELOPMENT_FIT_COMPONENTS, DEVELOPMENT_PROJECTION_CAPABILITY, DEVELOPMENT_PROJECTION_COMPONENTS, DEVELOPMENT_SELECTION_CAPABILITY, DEVELOPMENT_SELECTION_COMPONENTS, CALIBRATION_POLICY_CAPABILITY, CALIBRATION_POLICY_COMPONENTS, CALIBRATION_PROJECTION_CAPABILITY, CALIBRATION_PROJECTION_COMPONENTS, CALIBRATION_CALIBRATION_CAPABILITY, CALIBRATION_CALIBRATION_COMPONENTS, SOURCE_QUALIFICATION_PROJECTION_CAPABILITY, SOURCE_QUALIFICATION_PROJECTION_COMPONENTS, SOURCE_QUALIFICATION_EVALUATION_CAPABILITY, SOURCE_QUALIFICATION_EVALUATION_COMPONENTS
from .development_fit import PreparedResponseDevelopmentFitConfig
from .development_projection import PreparedResponseDevelopmentProjectionConfig
from .development_provider import PreparedResponseDevelopmentMethodProvider
from .development_selection import PreparedResponseDevelopmentSelectionConfig
from .policy_decision import PreparedPolicyDecisionConfig
from .policy_provider import PreparedPolicyDecisionProvider
from .calibration_records import PreparedResponseCalibrationCalibrationConfig, PreparedResponseCalibrationProjectionConfig
from .calibration_provider import PreparedResponseCalibrationMethodProvider
from .qualification_records import PreparedResponseSourceQualificationProjectionConfig
from .qualification import PreparedResponseSourceQualificationEvaluationConfig
from .qualification_provider import PreparedResponseSourceQualificationMethodProvider


SOURCE_QUALIFICATION_PROJECTION_BINDING = executable(
    SOURCE_QUALIFICATION_PROJECTION_CAPABILITY, SOURCE_QUALIFICATION_PROJECTION_COMPONENTS, (PreparedResponseSourceQualificationProjectionConfig,)
)
SOURCE_QUALIFICATION_EVALUATION_BINDING = executable(
    SOURCE_QUALIFICATION_EVALUATION_CAPABILITY, SOURCE_QUALIFICATION_EVALUATION_COMPONENTS, (PreparedResponseSourceQualificationEvaluationConfig,)
)
DEVELOPMENT_PROJECTION_BINDING = executable(
    DEVELOPMENT_PROJECTION_CAPABILITY, DEVELOPMENT_PROJECTION_COMPONENTS, (PreparedResponseDevelopmentProjectionConfig,)
)
DEVELOPMENT_FIT_BINDING = executable(DEVELOPMENT_FIT_CAPABILITY, DEVELOPMENT_FIT_COMPONENTS, (PreparedResponseDevelopmentFitConfig,))
DEVELOPMENT_SELECTION_BINDING = executable(
    DEVELOPMENT_SELECTION_CAPABILITY, DEVELOPMENT_SELECTION_COMPONENTS, (PreparedResponseDevelopmentSelectionConfig,)
)
CALIBRATION_POLICY_BINDING = executable(
    CALIBRATION_POLICY_CAPABILITY, CALIBRATION_POLICY_COMPONENTS, (PreparedPolicyDecisionConfig,)
)
CALIBRATION_PROJECTION_BINDING = executable(
    CALIBRATION_PROJECTION_CAPABILITY, CALIBRATION_PROJECTION_COMPONENTS, (PreparedResponseCalibrationProjectionConfig,)
)
CALIBRATION_CALIBRATION_BINDING = executable(
    CALIBRATION_CALIBRATION_CAPABILITY, CALIBRATION_CALIBRATION_COMPONENTS, (PreparedResponseCalibrationCalibrationConfig,)
)
CALIBRATION_POLICY_BINDING, CALIBRATION_PROJECTION_BINDING, CALIBRATION_CALIBRATION_BINDING = tuple(
    replace(
        binding,
        issued_decoder_registrations=tuple(
            replace(decoder, maximum_payload_bytes=16 * 1024**2)
            for decoder in binding.issued_decoder_registrations
        ),
    )
    for binding in (CALIBRATION_POLICY_BINDING, CALIBRATION_PROJECTION_BINDING, CALIBRATION_CALIBRATION_BINDING)
)


@dataclass(frozen=True, slots=True)
class PreparedResponseSourceQualificationMethodFactory:
    binding: ExecutableCapabilityBinding

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> PreparedResponseSourceQualificationMethodProvider:
        if platform_ports or len(records) != 1:
            raise ValueError(
                "prepared source qualification method factory requires one exact issued config and no source port"
            )
        config = records[0]
        if self.binding == SOURCE_QUALIFICATION_PROJECTION_BINDING and type(config) is PreparedResponseSourceQualificationProjectionConfig:
            return PreparedResponseSourceQualificationMethodProvider(registry, SOURCE_QUALIFICATION_PROJECTION_CAPABILITY, config)
        if self.binding == SOURCE_QUALIFICATION_EVALUATION_BINDING and type(config) is PreparedResponseSourceQualificationEvaluationConfig:
            return PreparedResponseSourceQualificationMethodProvider(registry, SOURCE_QUALIFICATION_EVALUATION_CAPABILITY, config)
        raise ValueError(
            "prepared source qualification method factory changed its installed binding/configuration type"
        )


@dataclass(frozen=True, slots=True)
class PreparedResponseDevelopmentMethodFactory:
    binding: ExecutableCapabilityBinding

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> PreparedResponseDevelopmentMethodProvider:
        if platform_ports or len(records) != 1:
            raise ValueError(
                "prepared dependent refinement method factory requires one exact issued config and no source port"
            )
        config = records[0]
        if (
            self.binding == DEVELOPMENT_PROJECTION_BINDING
            and type(config) is PreparedResponseDevelopmentProjectionConfig
        ) or (
            self.binding == DEVELOPMENT_FIT_BINDING and type(config) is PreparedResponseDevelopmentFitConfig
        ) or (
            self.binding == DEVELOPMENT_SELECTION_BINDING and type(config) is PreparedResponseDevelopmentSelectionConfig
        ):
            return PreparedResponseDevelopmentMethodProvider(
                registry,
                DEVELOPMENT_PROJECTION_CAPABILITY
                if type(config) is PreparedResponseDevelopmentProjectionConfig
                else DEVELOPMENT_FIT_CAPABILITY
                if type(config) is PreparedResponseDevelopmentFitConfig
                else DEVELOPMENT_SELECTION_CAPABILITY,
                config,
            )
        raise ValueError(
            "prepared dependent refinement method factory changed its installed binding/configuration type"
        )


@dataclass(frozen=True, slots=True)
class PreparedResponseCalibrationMethodFactory:
    binding: ExecutableCapabilityBinding

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> PreparedPolicyDecisionProvider | PreparedResponseCalibrationMethodProvider:
        if platform_ports or len(records) != 1:
            raise ValueError(
                "prepared fresh calibration method factory requires one exact issued config and no source port"
            )
        config = records[0]
        if self.binding == CALIBRATION_POLICY_BINDING and type(config) is PreparedPolicyDecisionConfig:
            return PreparedPolicyDecisionProvider(registry, CALIBRATION_POLICY_CAPABILITY, config)
        if self.binding == CALIBRATION_PROJECTION_BINDING and type(config) is PreparedResponseCalibrationProjectionConfig:
            return PreparedResponseCalibrationMethodProvider(registry, CALIBRATION_PROJECTION_CAPABILITY, config)
        if self.binding == CALIBRATION_CALIBRATION_BINDING and type(config) is PreparedResponseCalibrationCalibrationConfig:
            return PreparedResponseCalibrationMethodProvider(registry, CALIBRATION_CALIBRATION_CAPABILITY, config)
        raise ValueError(
            "prepared fresh calibration method factory changed its installed binding/configuration type"
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.prepared-response.source-qualification-dependent-refinement.methods",
    "1.0.0",
    tuple(
        sorted(
            (
                SOURCE_QUALIFICATION_PROJECTION_BINDING,
                SOURCE_QUALIFICATION_EVALUATION_BINDING,
                DEVELOPMENT_PROJECTION_BINDING,
                DEVELOPMENT_FIT_BINDING,
                DEVELOPMENT_SELECTION_BINDING,
                CALIBRATION_POLICY_BINDING,
                CALIBRATION_PROJECTION_BINDING,
                CALIBRATION_CALIBRATION_BINDING,
            ),
            key=lambda value: value.binding_id,
        )
    ),
)
EXECUTABLE_BINDING_FACTORIES = tuple(
    PreparedResponseSourceQualificationMethodFactory(binding)
    if binding in (SOURCE_QUALIFICATION_PROJECTION_BINDING, SOURCE_QUALIFICATION_EVALUATION_BINDING)
    else PreparedResponseDevelopmentMethodFactory(binding)
    if binding in (DEVELOPMENT_PROJECTION_BINDING, DEVELOPMENT_FIT_BINDING, DEVELOPMENT_SELECTION_BINDING)
    else PreparedResponseCalibrationMethodFactory(binding)
    for binding in EXECUTABLE_BINDING_CONTRIBUTION.bindings
)
_RECORD_TYPES: tuple[type[CanonicalRecord], ...] = (
    PreparedResponseSourceQualificationProjectionConfig,
    PreparedResponseSourceQualificationEvaluationConfig,
    PreparedResponseDevelopmentProjectionConfig,
    PreparedResponseDevelopmentFitConfig,
    PreparedResponseDevelopmentSelectionConfig,
    PreparedPolicyDecisionConfig,
    PreparedResponseCalibrationProjectionConfig,
    PreparedResponseCalibrationCalibrationConfig,
)
EXECUTABLE_RECORD_TYPES = tuple(sorted(_RECORD_TYPES, key=lambda value: value.SCHEMA))
