"Native projection/completion records; no fitted law or consumer verdict."

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
)
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import FiniteResponseLawNativeConfig, FiniteResponseLawNativeInvocation, FiniteResponseLawNativeRoot, native_invocations
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawNativeTaskResult
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawCalibrationTaskResult
from empirical_lawhood.adapters.simulators.finite_response_law.fresh_contracts import FiniteResponseLawCalibrationConfig, FiniteResponseLawCalibrationRoot, FiniteResponseLawCalibrationInvocation
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import REFERENCE_INSTRUMENT
from empirical_lawhood.adapters.simulators.finite_response_law.source import FiniteResponseLawCalibrationCheckpoint
from empirical_lawhood.adapters.simulators.finite_response_law.source import FiniteResponseLawNativeCheckpoint


@dataclass(frozen=True, slots=True)
class FiniteResponseLawProjectionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-projection-config'
    )
    SOURCE_TYPE: ClassVar[type[FiniteResponseLawNativeConfig]] = FiniteResponseLawNativeConfig
    native_spec: FiniteResponseLawNativeConfig

    def __post_init__(self) -> None:
        if type(self.native_spec) is not self.SOURCE_TYPE:
            raise ValueError("Finite response-law projection requires its exact versioned source")

    @property
    def config_id(self) -> str:
        return f"{self.native_spec.spec_id}.projection"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeEvaluationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-native-evaluation-config'
    )
    PROJECTION_TYPE: ClassVar[type[FiniteResponseLawProjectionConfig]] = FiniteResponseLawProjectionConfig
    projection: FiniteResponseLawProjectionConfig

    def __post_init__(self) -> None:
        if type(self.projection) is not self.PROJECTION_TYPE:
            raise ValueError("Finite response-law evaluation requires its exact versioned projection")

    @property
    def config_id(self) -> str:
        return f"{self.projection.native_spec.spec_id}.evaluation"


@dataclass(frozen=True, slots=True)
class FiniteResponseLawWordObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-word-observation'
    )
    RESULT_TYPE: ClassVar[type[FiniteResponseLawNativeTaskResult]] = FiniteResponseLawNativeTaskResult
    invocation: FiniteResponseLawNativeInvocation
    native_result: ObjectIdentity
    matched_hold_result: ObjectIdentity
    complete: bool
    maximum_force_error: Decimal | None
    outputs: tuple[Decimal | None, ...]

    def __post_init__(self) -> None:
        if (
            self.invocation.phase != "future"
            or self.native_result.object_id != f"{self.invocation.task_id}.result"
            or self.native_result.object_schema != self.RESULT_TYPE.SCHEMA
            or self.matched_hold_result.object_schema != self.RESULT_TYPE.SCHEMA
            or self.matched_hold_result.object_id
            != f"{replace(self.invocation, word=PreparedForceWord(Decimal(0), 0, 0)).task_id}.result"
            or type(self.complete) is not bool
            or len(self.outputs) != 7
            or any(
                v is not None and (not isinstance(v, Decimal) or not v.is_finite())
                for v in (*self.outputs, self.maximum_force_error)
            )
            or self.maximum_force_error is not None
            and self.maximum_force_error < 0
        ):
            raise ValueError(
                "Finite response-law word projection changes finite operands or native lineage"
            )

    @property
    def observation_id(self) -> str:
        return self.invocation.task_id


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeViewObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-native-view-observation'
    )
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawProjectionConfig]] = FiniteResponseLawProjectionConfig
    RESULT_TYPE: ClassVar[type[FiniteResponseLawNativeTaskResult]] = FiniteResponseLawNativeTaskResult
    config: ObjectIdentity
    root: FiniteResponseLawNativeRoot
    refinement: int
    native_results: tuple[ObjectIdentity, ...]
    accounting: tuple[tuple[str, int, str], ...]
    words: tuple[FiniteResponseLawWordObservation, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_ids(
            self.native_results, attribute="object_id", field_name="native_results"
        )
        require_sorted_unique_ids(
            self.words, attribute="observation_id", field_name="words"
        )
        if (
            self.config.object_schema != self.CONFIG_TYPE.SCHEMA
            or type(self.refinement) is not int
            or self.refinement not in (1, 2)
            or len(self.native_results) != len(self.accounting)
            or any(
                r.object_schema != self.RESULT_TYPE.SCHEMA for r in self.native_results
            )
            or tuple(f"{row[0]}.result" for row in self.accounting)
            != tuple(r.object_id for r in self.native_results)
            or any(
                type(n) is not int
                or n < 0
                or status
                not in (
                    "COMPLETE",
                    "NUMERICAL_FAILURE",
                    "OBSERVATION_FAILURE",
                    "PREFIX_UNAVAILABLE",
                    "PORT_FRAME_UNRESOLVED",
                    "HANDOFF_UNAVAILABLE",
                )
                for _, n, status in self.accounting
            )
            or any(
                w.invocation.root != self.root
                or w.native_result not in self.native_results
                or w.matched_hold_result not in self.native_results
                for w in self.words
            )
        ):
            raise ValueError("Finite response-law native projection changes root/view/phase accounting")

    @property
    def report_id(self) -> str:
        return f"{self.root.root_id}.flh-project.r{self.refinement}"

    @property
    def reasons(self) -> tuple[str, ...]:
        values = {status for _, _, status in self.accounting if status != "COMPLETE"}
        if any(
            w.maximum_force_error is None or w.maximum_force_error != 0
            for w in self.words
        ):
            values.add("REQUESTED_APPLIED_FORCE_UNRESOLVED")
        if any(not w.complete or any(v is None for v in w.outputs) for w in self.words):
            values.add("PAIRED_NATIVE_OBSERVATION_INCOMPLETE")
        return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class FiniteResponseLawNativeEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-native-evaluation'
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawNativeEvaluationConfig]] = (
        FiniteResponseLawNativeEvaluationConfig
    )
    VIEW_TYPE: ClassVar[type[FiniteResponseLawNativeViewObservation]] = FiniteResponseLawNativeViewObservation
    config: FiniteResponseLawNativeEvaluationConfig
    views: tuple[FiniteResponseLawNativeViewObservation, ...]

    def __post_init__(self) -> None:
        if type(self.config) is not self.CONFIG_TYPE or any(
            type(v) is not self.VIEW_TYPE for v in self.views
        ):
            raise ValueError(
                "Finite response-law completion changes its versioned configuration or views"
            )
        require_sorted_unique_ids(self.views, attribute="report_id", field_name="views")
        source = self.config.projection.native_spec
        expected = tuple((root, r) for root in source.roots for r in (1, 2))
        if tuple((v.root, v.refinement) for v in self.views) != expected:
            raise ValueError(
                "Finite response-law completion omits an assigned physical root or numerical view"
            )
        tasks = native_invocations(source)
        identity = ObjectIdentity.from_record(
            self.config.projection.config_id, self.config.projection
        )
        for view in self.views:
            local = tuple(t for t in tasks if t.root == view.root)
            if (
                view.config != identity
                or tuple(w.invocation for w in view.words)
                != tuple(t for t in local if t.phase == "future")
                or tuple(r[0] for r in view.accounting)
                != tuple(t.task_id for t in local)
                or any(
                    n > (t.clocks[1] - t.clocks[0]) * view.refinement
                    or status == "COMPLETE"
                    and n != (t.clocks[1] - t.clocks[0]) * view.refinement
                    for t, (_, n, status) in zip(local, view.accounting, strict=True)
                )
            ):
                raise ValueError(
                    "Finite response-law completion changes its complete native task/update roster"
                )

    @property
    def evaluation_id(self) -> str:
        return f"{self.config.config_id}.result"

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(sorted({r for v in self.views for r in v.reasons}))

    @property
    def scientific_status(self) -> ScientificStatus:
        # This is completion of declared native measurement, not the joined measurement oracle,
        # interface information, predictive lawhood or a controller-use claim.
        return (
            ScientificStatus.UNEVALUABLE if self.reasons else ScientificStatus.SUPPORTED
        )

    @property
    def completed_native_updates(self) -> int:
        return sum(n for v in self.views for _, n, _ in v.accounting)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationProjectionConfig(FiniteResponseLawProjectionConfig):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-calibration-projection-config'
    )
    VERSION: ClassVar[str] = '1.0.0'
    SOURCE_TYPE: ClassVar[type[FiniteResponseLawNativeConfig]] = FiniteResponseLawCalibrationConfig
    native_spec: FiniteResponseLawCalibrationConfig


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationEvaluationConfig(FiniteResponseLawNativeEvaluationConfig):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-calibration-evaluation-config'
    )
    VERSION: ClassVar[str] = '1.0.0'
    PROJECTION_TYPE: ClassVar[type[FiniteResponseLawProjectionConfig]] = (
        FiniteResponseLawCalibrationProjectionConfig
    )
    projection: FiniteResponseLawCalibrationProjectionConfig


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationWordObservation(FiniteResponseLawWordObservation):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-calibration-word-observation'
    )
    VERSION: ClassVar[str] = '1.0.0'
    RESULT_TYPE: ClassVar[type[FiniteResponseLawNativeTaskResult]] = FiniteResponseLawCalibrationTaskResult
    invocation: FiniteResponseLawCalibrationInvocation


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationInterface(CanonicalRecord):
    """Frozen causal instrument output; never a fitted or forecast interface."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-calibration-interface'
    )
    VERSION: ClassVar[str] = '1.0.0'
    CHECKPOINT_TYPE: ClassVar[type[FiniteResponseLawNativeCheckpoint]] = FiniteResponseLawCalibrationCheckpoint
    checkpoint: ObjectIdentity
    instrument: ObjectIdentity
    cutoff_tick: int
    values: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        if (
            self.checkpoint.object_schema != self.CHECKPOINT_TYPE.SCHEMA
            or self.instrument != REFERENCE_INSTRUMENT.identity
            or type(self.cutoff_tick) is not int
            or self.cutoff_tick not in (4096, 4368)
            or len(self.values) != 24
            or any(not isinstance(v, Decimal) or not v.is_finite() for v in self.values)
        ):
            raise ValueError("Finite response-law calibration interface changes its causal instrument")


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationViewObservation(FiniteResponseLawNativeViewObservation):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-calibration-view-observation'
    )
    VERSION: ClassVar[str] = '1.0.0'
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawProjectionConfig]] = (
        FiniteResponseLawCalibrationProjectionConfig
    )
    RESULT_TYPE: ClassVar[type[FiniteResponseLawNativeTaskResult]] = FiniteResponseLawCalibrationTaskResult
    root: FiniteResponseLawCalibrationRoot
    words: tuple[FiniteResponseLawCalibrationWordObservation, ...]
    prefix_interface: FiniteResponseLawCalibrationInterface | None
    handoff_interface: FiniteResponseLawCalibrationInterface | None
    parent_absolute_density_work: Decimal | None

    def __post_init__(self) -> None:
        super(FiniteResponseLawCalibrationViewObservation, self).__post_init__()
        for phase, cutoff, interface in (
            ("prefix", 4096, self.prefix_interface),
            ("parent", 4368, self.handoff_interface),
        ):
            task = next(
                key for key, _, _ in self.accounting if key.endswith(f".{phase}.native")
            )
            if interface is not None and (
                interface.cutoff_tick != cutoff
                or interface.checkpoint.object_id
                != f"{task}.r{self.refinement}.checkpoint"
            ):
                raise ValueError(
                    "Finite response-law calibration interface substitutes root, view or cutoff"
                )
        work = self.parent_absolute_density_work
        if work is not None and (
            not isinstance(work, Decimal) or not work.is_finite() or work < 0
        ):
            raise ValueError(
                "Finite response-law calibration parent work must be finite and nonnegative"
            )

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons = set(super(FiniteResponseLawCalibrationViewObservation, self).reasons)
        if self.prefix_interface is None or self.handoff_interface is None:
            reasons.add("CAUSAL_INTERFACE_UNAVAILABLE")
        if self.parent_absolute_density_work is None:
            reasons.add("PARENT_WORK_UNAVAILABLE")
        return tuple(sorted(reasons))


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCalibrationNativeEvaluation(FiniteResponseLawNativeEvaluation):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/finite-response-law/finite-response-law-calibration-native-evaluation'
    )
    VERSION: ClassVar[str] = '1.0.0'
    CONFIG_TYPE: ClassVar[type[FiniteResponseLawNativeEvaluationConfig]] = (
        FiniteResponseLawCalibrationEvaluationConfig
    )
    VIEW_TYPE: ClassVar[type[FiniteResponseLawNativeViewObservation]] = (
        FiniteResponseLawCalibrationViewObservation
    )
    config: FiniteResponseLawCalibrationEvaluationConfig
    views: tuple[FiniteResponseLawCalibrationViewObservation, ...]


def native_method_types(
    config: FiniteResponseLawNativeConfig,
) -> tuple[
    type[FiniteResponseLawProjectionConfig],
    type[FiniteResponseLawNativeEvaluationConfig],
    type[FiniteResponseLawWordObservation],
    type[FiniteResponseLawNativeViewObservation],
    type[FiniteResponseLawNativeEvaluation],
]:
    """Explicit compatibility map; the old native record schemas remain closed."""
    from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_contracts import FiniteResponseLawEvaluationConfig
    from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedCalibrationConfig, FiniteResponseLawAssignedEvaluationConfig

    if type(config) is FiniteResponseLawAssignedCalibrationConfig:
        from .assigned_native_records import FiniteResponseLawAssignedCalibrationProjectionConfig, FiniteResponseLawAssignedCalibrationEvaluationConfig, FiniteResponseLawAssignedCalibrationWordObservation, FiniteResponseLawAssignedCalibrationViewObservation, FiniteResponseLawAssignedCalibrationNativeEvaluation

        return (
            FiniteResponseLawAssignedCalibrationProjectionConfig,
            FiniteResponseLawAssignedCalibrationEvaluationConfig,
            FiniteResponseLawAssignedCalibrationWordObservation,
            FiniteResponseLawAssignedCalibrationViewObservation,
            FiniteResponseLawAssignedCalibrationNativeEvaluation,
        )
    if type(config) is FiniteResponseLawAssignedEvaluationConfig:
        from .assigned_native_records import FiniteResponseLawAssignedEvaluationProjectionConfig, FiniteResponseLawAssignedEvaluationCompletionConfig, FiniteResponseLawAssignedEvaluationWordObservation, FiniteResponseLawAssignedEvaluationViewObservation, FiniteResponseLawAssignedEvaluationNativeCompletion

        return (
            FiniteResponseLawAssignedEvaluationProjectionConfig,
            FiniteResponseLawAssignedEvaluationCompletionConfig,
            FiniteResponseLawAssignedEvaluationWordObservation,
            FiniteResponseLawAssignedEvaluationViewObservation,
            FiniteResponseLawAssignedEvaluationNativeCompletion,
        )

    if type(config) is FiniteResponseLawEvaluationConfig:
        from .evaluation_native_records import FiniteResponseLawEvaluationProjectionConfig, FiniteResponseLawEvaluationCompletionConfig, FiniteResponseLawEvaluationWordObservation, FiniteResponseLawEvaluationViewObservation, FiniteResponseLawEvaluationNativeCompletion

        return (
            FiniteResponseLawEvaluationProjectionConfig,
            FiniteResponseLawEvaluationCompletionConfig,
            FiniteResponseLawEvaluationWordObservation,
            FiniteResponseLawEvaluationViewObservation,
            FiniteResponseLawEvaluationNativeCompletion,
        )
    if type(config) is FiniteResponseLawCalibrationConfig:
        return (
            FiniteResponseLawCalibrationProjectionConfig,
            FiniteResponseLawCalibrationEvaluationConfig,
            FiniteResponseLawCalibrationWordObservation,
            FiniteResponseLawCalibrationViewObservation,
            FiniteResponseLawCalibrationNativeEvaluation,
        )
    if type(config) is FiniteResponseLawNativeConfig:
        return (
            FiniteResponseLawProjectionConfig,
            FiniteResponseLawNativeEvaluationConfig,
            FiniteResponseLawWordObservation,
            FiniteResponseLawNativeViewObservation,
            FiniteResponseLawNativeEvaluation,
        )
    raise ValueError("Undeclared finite response-law native method source type")
