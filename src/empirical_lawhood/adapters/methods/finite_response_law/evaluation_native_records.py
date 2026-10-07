"""Prospective native records reusing the closed projection/reducer contracts."""

from dataclasses import dataclass
from typing import ClassVar
from empirical_lawhood.kernel.action_contracts import ObservedActionOccurrence

from empirical_lawhood.adapters.simulators.finite_response_law.contracts import FiniteResponseLawNativeConfig
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_contracts import FiniteResponseLawEvaluationConfig, FiniteResponseLawEvaluationInvocation, FiniteResponseLawEvaluationRoot
from empirical_lawhood.adapters.simulators.finite_response_law.source import FiniteResponseLawNativeCheckpoint, FiniteResponseLawEvaluationCheckpoint
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawNativeTaskResult, FiniteResponseLawEvaluationTaskResult
from .native_records import FiniteResponseLawProjectionConfig, FiniteResponseLawNativeEvaluationConfig, FiniteResponseLawNativeViewObservation, FiniteResponseLawCalibrationInterface, FiniteResponseLawCalibrationViewObservation, FiniteResponseLawCalibrationWordObservation, FiniteResponseLawNativeEvaluation


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationProjectionConfig(FiniteResponseLawProjectionConfig):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-evaluation-projection-config'
    SOURCE_TYPE: ClassVar[type[FiniteResponseLawNativeConfig]] = FiniteResponseLawEvaluationConfig
    native_spec: FiniteResponseLawEvaluationConfig


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationCompletionConfig(FiniteResponseLawNativeEvaluationConfig):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-evaluation-completion-config'
    PROJECTION_TYPE: ClassVar[type[FiniteResponseLawProjectionConfig]] = FiniteResponseLawEvaluationProjectionConfig
    projection: FiniteResponseLawEvaluationProjectionConfig


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationWordObservation(FiniteResponseLawCalibrationWordObservation):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-evaluation-word-observation'
    VERSION: ClassVar[str] = "1.0.0"
    RESULT_TYPE: ClassVar[type[FiniteResponseLawNativeTaskResult]] = FiniteResponseLawEvaluationTaskResult
    invocation: FiniteResponseLawEvaluationInvocation


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationInterface(FiniteResponseLawCalibrationInterface):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-evaluation-interface'
    VERSION: ClassVar[str] = "1.0.0"
    CHECKPOINT_TYPE: ClassVar[type[FiniteResponseLawNativeCheckpoint]] = FiniteResponseLawEvaluationCheckpoint


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationViewObservation(FiniteResponseLawCalibrationViewObservation):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-evaluation-view-observation'
    VERSION: ClassVar[str] = "1.0.0"
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawProjectionConfig]] = FiniteResponseLawEvaluationProjectionConfig
    RESULT_TYPE: ClassVar[type[FiniteResponseLawNativeTaskResult]] = FiniteResponseLawEvaluationTaskResult
    root: FiniteResponseLawEvaluationRoot
    words: tuple[FiniteResponseLawEvaluationWordObservation, ...]
    prefix_interface: FiniteResponseLawEvaluationInterface | None
    handoff_interface: FiniteResponseLawEvaluationInterface | None
    delivery_observations: tuple[ObservedActionOccurrence, ...]

    def __post_init__(self) -> None:
        super(FiniteResponseLawEvaluationViewObservation, self).__post_init__()
        from .control_delivery import observation_words

        words = observation_words(self.root.root_id)
        expected = tuple(sorted(w.occurrences[0].occurrence_id for w in words))
        if tuple(o.expected_occurrence_id for o in self.delivery_observations) != expected:
            raise ValueError("Finite response-law evaluation view loses the full nine-word actual delivery observation census")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationNativeCompletion(FiniteResponseLawNativeEvaluation):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-evaluation-native-completion'
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawNativeEvaluationConfig]] = FiniteResponseLawEvaluationCompletionConfig
    VIEW_TYPE: ClassVar[type[FiniteResponseLawNativeViewObservation]] = FiniteResponseLawEvaluationViewObservation
    config: FiniteResponseLawEvaluationCompletionConfig
    views: tuple[FiniteResponseLawEvaluationViewObservation, ...]
