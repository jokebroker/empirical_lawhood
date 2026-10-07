"""Versioned projections for externally assigned finite Tier 1 roots."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedCalibrationConfig, FiniteResponseLawAssignedCalibrationInvocation, FiniteResponseLawAssignedCalibrationRoot, FiniteResponseLawAssignedEvaluationConfig, FiniteResponseLawAssignedEvaluationInvocation, FiniteResponseLawAssignedEvaluationRoot
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import FiniteResponseLawNativeConfig
from empirical_lawhood.adapters.simulators.finite_response_law.source import FiniteResponseLawAssignedCalibrationCheckpoint, FiniteResponseLawAssignedEvaluationCheckpoint, FiniteResponseLawNativeCheckpoint
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedCalibrationTaskResult, FiniteResponseLawAssignedEvaluationTaskResult, FiniteResponseLawNativeTaskResult

from .evaluation_native_records import FiniteResponseLawEvaluationCompletionConfig, FiniteResponseLawEvaluationInterface, FiniteResponseLawEvaluationProjectionConfig, FiniteResponseLawEvaluationViewObservation, FiniteResponseLawEvaluationWordObservation, FiniteResponseLawEvaluationNativeCompletion
from .native_records import FiniteResponseLawCalibrationEvaluationConfig, FiniteResponseLawCalibrationInterface, FiniteResponseLawCalibrationNativeEvaluation, FiniteResponseLawCalibrationProjectionConfig, FiniteResponseLawCalibrationViewObservation, FiniteResponseLawCalibrationWordObservation, FiniteResponseLawNativeEvaluationConfig, FiniteResponseLawNativeViewObservation, FiniteResponseLawProjectionConfig


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationProjectionConfig(FiniteResponseLawCalibrationProjectionConfig):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-calibration-projection-config'
    )
    VERSION: ClassVar[str] = '1.0.0'
    SOURCE_TYPE: ClassVar[type[FiniteResponseLawNativeConfig]] = FiniteResponseLawAssignedCalibrationConfig
    native_spec: FiniteResponseLawAssignedCalibrationConfig


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationEvaluationConfig(FiniteResponseLawCalibrationEvaluationConfig):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-calibration-evaluation-config'
    )
    VERSION: ClassVar[str] = '1.0.0'
    PROJECTION_TYPE: ClassVar[type[FiniteResponseLawProjectionConfig]] = (
        FiniteResponseLawAssignedCalibrationProjectionConfig
    )
    projection: FiniteResponseLawAssignedCalibrationProjectionConfig


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationWordObservation(FiniteResponseLawCalibrationWordObservation):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-calibration-word-observation'
    )
    VERSION: ClassVar[str] = '1.0.0'
    RESULT_TYPE: ClassVar[type[FiniteResponseLawNativeTaskResult]] = (
        FiniteResponseLawAssignedCalibrationTaskResult
    )
    invocation: FiniteResponseLawAssignedCalibrationInvocation


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationInterface(FiniteResponseLawCalibrationInterface):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-calibration-interface'
    )
    VERSION: ClassVar[str] = '1.0.0'
    CHECKPOINT_TYPE: ClassVar[type[FiniteResponseLawNativeCheckpoint]] = (
        FiniteResponseLawAssignedCalibrationCheckpoint
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationViewObservation(FiniteResponseLawCalibrationViewObservation):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-calibration-view-observation'
    )
    VERSION: ClassVar[str] = '1.0.0'
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawProjectionConfig]] = (
        FiniteResponseLawAssignedCalibrationProjectionConfig
    )
    RESULT_TYPE: ClassVar[type[FiniteResponseLawNativeTaskResult]] = (
        FiniteResponseLawAssignedCalibrationTaskResult
    )
    root: FiniteResponseLawAssignedCalibrationRoot
    words: tuple[FiniteResponseLawAssignedCalibrationWordObservation, ...]
    prefix_interface: FiniteResponseLawAssignedCalibrationInterface | None
    handoff_interface: FiniteResponseLawAssignedCalibrationInterface | None


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedCalibrationNativeEvaluation(FiniteResponseLawCalibrationNativeEvaluation):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-calibration-native-evaluation'
    )
    VERSION: ClassVar[str] = '1.0.0'
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawNativeEvaluationConfig]] = (
        FiniteResponseLawAssignedCalibrationEvaluationConfig
    )
    VIEW_TYPE: ClassVar[type[FiniteResponseLawNativeViewObservation]] = (
        FiniteResponseLawAssignedCalibrationViewObservation
    )
    config: FiniteResponseLawAssignedCalibrationEvaluationConfig
    views: tuple[FiniteResponseLawAssignedCalibrationViewObservation, ...]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationProjectionConfig(FiniteResponseLawEvaluationProjectionConfig):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-evaluation-projection-config'
    )
    VERSION: ClassVar[str] = '1.0.0'
    SOURCE_TYPE: ClassVar[type[FiniteResponseLawNativeConfig]] = FiniteResponseLawAssignedEvaluationConfig
    native_spec: FiniteResponseLawAssignedEvaluationConfig


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationCompletionConfig(FiniteResponseLawEvaluationCompletionConfig):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-evaluation-completion-config'
    )
    VERSION: ClassVar[str] = '1.0.0'
    PROJECTION_TYPE: ClassVar[type[FiniteResponseLawProjectionConfig]] = (
        FiniteResponseLawAssignedEvaluationProjectionConfig
    )
    projection: FiniteResponseLawAssignedEvaluationProjectionConfig


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationWordObservation(FiniteResponseLawEvaluationWordObservation):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-evaluation-word-observation'
    )
    VERSION: ClassVar[str] = '1.0.0'
    RESULT_TYPE: ClassVar[type[FiniteResponseLawNativeTaskResult]] = (
        FiniteResponseLawAssignedEvaluationTaskResult
    )
    invocation: FiniteResponseLawAssignedEvaluationInvocation


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationInterface(FiniteResponseLawEvaluationInterface):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-evaluation-interface'
    )
    VERSION: ClassVar[str] = '1.0.0'
    CHECKPOINT_TYPE: ClassVar[type[FiniteResponseLawNativeCheckpoint]] = (
        FiniteResponseLawAssignedEvaluationCheckpoint
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationViewObservation(FiniteResponseLawEvaluationViewObservation):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-evaluation-view-observation'
    )
    VERSION: ClassVar[str] = '1.0.0'
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawProjectionConfig]] = (
        FiniteResponseLawAssignedEvaluationProjectionConfig
    )
    RESULT_TYPE: ClassVar[type[FiniteResponseLawNativeTaskResult]] = (
        FiniteResponseLawAssignedEvaluationTaskResult
    )
    root: FiniteResponseLawAssignedEvaluationRoot
    words: tuple[FiniteResponseLawAssignedEvaluationWordObservation, ...]
    prefix_interface: FiniteResponseLawAssignedEvaluationInterface | None
    handoff_interface: FiniteResponseLawAssignedEvaluationInterface | None


@dataclass(frozen=True, slots=True)
class FiniteResponseLawAssignedEvaluationNativeCompletion(FiniteResponseLawEvaluationNativeCompletion):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-assigned-evaluation-native-completion'
    )
    VERSION: ClassVar[str] = '1.0.0'
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawNativeEvaluationConfig]] = (
        FiniteResponseLawAssignedEvaluationCompletionConfig
    )
    VIEW_TYPE: ClassVar[type[FiniteResponseLawNativeViewObservation]] = (
        FiniteResponseLawAssignedEvaluationViewObservation
    )
    config: FiniteResponseLawAssignedEvaluationCompletionConfig
    views: tuple[FiniteResponseLawAssignedEvaluationViewObservation, ...]
