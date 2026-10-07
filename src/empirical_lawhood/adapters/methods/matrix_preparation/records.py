"""Exact development custody and method records; no self-granted authority."""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
)
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import CONTEXTS, DEVELOPMENT, PreparationSourceConfig, preparation_roots
from .contracts import CANDIDATES, PreparationMethodConfig, PreparationProjectionConfig, PreparationProjectionReport, PreparationPrivilegedReport


def _projection_inputs(
    context: str, reports: tuple[ObjectIdentity, ...], *, privileged: bool = False
) -> None:
    if context not in CONTEXTS:
        raise ValueError("preparation method cannot pool or add a context")
    require_sorted_unique_ids(reports, attribute="object_id", field_name="input_reports")
    expected = {
        f"report.{root.root_id}{'.privileged' if privileged else ''}.r{r}"
        for root in preparation_roots()
        if root.context == context
        for r in (1, 2)
    }
    if (
        {v.object_id for v in reports} != expected
        or len(reports) != 128
        or any(
            v.object_schema
            != (
                PreparationPrivilegedReport.SCHEMA
                if privileged
                else PreparationProjectionReport.SCHEMA
            )
            for v in reports
        )
    ):
        raise ValueError("preparation method omitted or changed an exposed root/numerical view")


@dataclass(frozen=True, slots=True)
class PreparationNumericalSemanticsConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-numerical-semantics-config'
    config_id: str
    projection_config: ObjectIdentity
    absolute_view_tolerance: Decimal = Decimal("0.0078125")
    odd_view_tolerance: Decimal = Decimal("0.00390625")
    secant_absolute_tolerance: Decimal = Decimal("1e-8")
    secant_relative_tolerance: Decimal = Decimal("1e-5")

    def __post_init__(self) -> None:
        if (
            self.config_id != f"{DEVELOPMENT}.numerical-semantics-config"
            or self.projection_config.object_schema != PreparationProjectionConfig.SCHEMA
            or (
                self.absolute_view_tolerance,
                self.odd_view_tolerance,
                self.secant_absolute_tolerance,
                self.secant_relative_tolerance,
            )
            != (Decimal("0.0078125"), Decimal("0.00390625"), Decimal("1e-8"), Decimal("1e-5"))
            or any(
                type(v) is not Decimal
                for v in (
                    self.absolute_view_tolerance,
                    self.odd_view_tolerance,
                    self.secant_absolute_tolerance,
                    self.secant_relative_tolerance,
                )
            )
        ):
            raise ValueError("Numerical semantics changes a frozen numerical tolerance or source binding")


@dataclass(frozen=True, slots=True)
class _PreparationMethodUseConfig(CanonicalRecord):
    config_id: str
    method: PreparationMethodConfig
    ROLE: ClassVar[str]

    def __post_init__(self) -> None:
        if (
            self.config_id != f"{DEVELOPMENT}.{self.ROLE}-config"
            or type(self.method) is not PreparationMethodConfig
        ):
            raise ValueError("preparation method use changes its exact role binding")


@dataclass(frozen=True, slots=True)
class PreparationForecastConfig(_PreparationMethodUseConfig):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-forecast-config'
    ROLE: ClassVar[str] = "forecast"


@dataclass(frozen=True, slots=True)
class PreparationDecisionConfig(_PreparationMethodUseConfig):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-decision-config'
    ROLE: ClassVar[str] = "decision"


@dataclass(frozen=True, slots=True)
class PreparationTaskAssessmentConfig(_PreparationMethodUseConfig):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-task-assessment-config'
    ROLE: ClassVar[str] = "task-assessment"


@dataclass(frozen=True, slots=True)
class PreparationNumericalSemanticsReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-numerical-semantics-report'
    report_id: str
    context: str
    config: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    privileged_reports: tuple[ObjectIdentity, ...]
    metrics: tuple[NamedDecimal, ...]
    data_sha256: str
    numerical_semantics_qualified: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _projection_inputs(self.context, self.input_reports)
        _projection_inputs(self.context, self.privileged_reports, privileged=True)
        validate_sha256(self.data_sha256, field_name="data_sha256")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.reasons, field_name="reasons")
        if (
            self.report_id != f"{DEVELOPMENT}.numerical-semantics.{self.context}"
            or self.config.object_schema != PreparationNumericalSemanticsConfig.SCHEMA
            or type(self.numerical_semantics_qualified) is not bool
            or self.numerical_semantics_qualified != (not self.reasons)
        ):
            raise ValueError("Numerical semantics changes its exact measurement qualification disposition")


@dataclass(frozen=True, slots=True)
class PreparationFitReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-fit-report'
    report_id: str
    context: str
    config: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    numerical_semantics: ObjectIdentity
    available_candidates: tuple[str, ...]
    unavailable_candidates: tuple[tuple[str, str], ...]
    data_sha256: str

    def __post_init__(self) -> None:
        _projection_inputs(self.context, self.input_reports)
        validate_sha256(self.data_sha256, field_name="data_sha256")
        if (
            self.report_id != f"{DEVELOPMENT}.fit.{self.context}"
            or self.config.object_schema != PreparationMethodConfig.SCHEMA
            or self.numerical_semantics.object_schema != PreparationNumericalSemanticsReport.SCHEMA
            or self.available_candidates
            != tuple(c for c in CANDIDATES if c in self.available_candidates)
            or tuple(c for c, _ in self.unavailable_candidates)
            != tuple(c for c in CANDIDATES if c not in self.available_candidates)
            or any(
                reason not in ("FIT_INPUT_UNRESOLVED", "NUMERICAL_FIT_UNRESOLVED")
                for _, reason in self.unavailable_candidates
            )
        ):
            raise ValueError("preparation fit changes its complete candidate/root/cutoff roster")


@dataclass(frozen=True, slots=True)
class PreparationAssessmentConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-assessment-config'
    config_id: str
    system: SystemSpec
    source_config: ObjectIdentity
    projection: PreparationProjectionConfig
    numerical_semantics: PreparationNumericalSemanticsConfig
    method: PreparationMethodConfig

    def __post_init__(self) -> None:
        projection = ObjectIdentity.from_record(self.projection.config_id, self.projection)
        if (
            self.config_id != f"{DEVELOPMENT}.assessment-config"
            or self.source_config.object_schema != PreparationSourceConfig.SCHEMA
            or self.projection.source_config != self.source_config
            or self.numerical_semantics.projection_config != projection
            or self.method.projection_config != projection
            or self.system.system_id != f"{DEVELOPMENT}.system"
        ):
            raise ValueError(
                "description assessment changes its scientific source/observer/method lineage"
            )


@dataclass(frozen=True, slots=True)
class PreparationGate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-gate'
    gate_id: str
    context: str
    role: str
    passes: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_strings(self.reasons, field_name="reasons")
        if (
            self.context not in CONTEXTS
            or self.role not in ("numerical-semantics", "observable-description", "preparation-window-forecast", "prospective-task-feasibility")
            or self.gate_id != f"{DEVELOPMENT}.gate.{self.role}.{self.context}"
            or type(self.passes) is not bool
            or self.passes != (not self.reasons)
        ):
            raise ValueError("development gate changes its declared context or honest terminal")


@dataclass(frozen=True, slots=True)
class PreparationCandidateScreen(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-candidate-screen'
    candidate: str
    metrics: tuple[NamedDecimal, ...]
    passes: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.reasons, field_name="reasons")
        if (
            self.candidate not in CANDIDATES
            or type(self.passes) is not bool
            or self.passes != (not self.reasons)
        ):
            raise ValueError("description screen changes its closed candidate or terminal")


@dataclass(frozen=True, slots=True)
class PreparationDescriptionReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-description-report'
    report_id: str
    context: str
    config: ObjectIdentity
    fit: ObjectIdentity
    numerical_semantics: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    screens: tuple[PreparationCandidateScreen, ...]
    data_sha256: str

    def __post_init__(self) -> None:
        _projection_inputs(self.context, self.input_reports)
        validate_sha256(self.data_sha256, field_name="data_sha256")
        if (
            self.report_id != f"{DEVELOPMENT}.description.{self.context}"
            or self.config.object_schema != PreparationAssessmentConfig.SCHEMA
            or self.fit.object_schema != PreparationFitReport.SCHEMA
            or self.numerical_semantics.object_schema != PreparationNumericalSemanticsReport.SCHEMA
            or tuple(s.candidate for s in self.screens) != CANDIDATES
        ):
            raise ValueError("description assessment changes its complete model/input lineage")

    @property
    def qualified_candidates(self) -> tuple[str, ...]:
        return tuple(s.candidate for s in self.screens if s.passes)

    @property
    def selected_candidate(self) -> str | None:
        return next(iter(self.qualified_candidates), None)


@dataclass(frozen=True, slots=True)
class PreparationBenchmarkReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-benchmark-report'
    report_id: str
    context: str
    config: ObjectIdentity
    fit: ObjectIdentity
    description: ObjectIdentity
    privileged_reports: tuple[ObjectIdentity, ...]
    benchmark_metrics: tuple[tuple[str, tuple[NamedDecimal, ...]], ...]
    unavailable_benchmarks: tuple[tuple[str, str], ...]
    data_sha256: str

    def __post_init__(self) -> None:
        _projection_inputs(self.context, self.privileged_reports, privileged=True)
        validate_sha256(self.data_sha256, field_name="data_sha256")
        if (
            self.report_id != f"{DEVELOPMENT}.benchmarks.{self.context}"
            or self.config.object_schema != PreparationAssessmentConfig.SCHEMA
            or self.fit.object_schema != PreparationFitReport.SCHEMA
            or self.description.object_schema != PreparationDescriptionReport.SCHEMA
        ):
            raise ValueError("privileged benchmarks changed their measured/fitted lineage")
        expected = tuple(
            f"{m}_{c}"
            for m in ("scalar", "full_hessian", "microscopic")
            for c in ("raw", "calibrated")
        )
        present = tuple(k for k, _ in self.benchmark_metrics)
        missing = tuple(k for k, _ in self.unavailable_benchmarks)
        if present != tuple(k for k in expected if k in present) or missing != tuple(
            k for k in expected if k not in present
        ):
            raise ValueError("benchmark assessment drops an attempted privileged control")


@dataclass(frozen=True, slots=True)
class PreparationForecastReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-forecast-report'
    report_id: str
    context: str
    config: ObjectIdentity
    fit: ObjectIdentity
    description: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    available_candidates: tuple[str, ...]
    unavailable_candidates: tuple[tuple[str, str], ...]
    candidate_metrics: tuple[tuple[str, tuple[NamedDecimal, ...]], ...]
    gate: PreparationGate
    data_sha256: str

    def __post_init__(self) -> None:
        _projection_inputs(self.context, self.input_reports)
        validate_sha256(self.data_sha256, field_name="data_sha256")
        if (
            self.report_id != f"{DEVELOPMENT}.forecast.{self.context}"
            or self.config.object_schema != PreparationMethodConfig.SCHEMA
            or self.fit.object_schema != PreparationFitReport.SCHEMA
            or self.description.object_schema != PreparationDescriptionReport.SCHEMA
            or self.gate.role != "preparation-window-forecast"
            or self.gate.context != self.context
            or self.available_candidates
            != tuple(c for c in CANDIDATES if c in self.available_candidates)
            or tuple(c for c, _ in self.unavailable_candidates)
            != tuple(c for c in CANDIDATES if c not in self.available_candidates)
            or tuple(c for c, _ in self.candidate_metrics) != self.available_candidates
        ):
            raise ValueError("forecast report changes its observable cutoff or candidate census")


@dataclass(frozen=True, slots=True)
class PreparationDecisionReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-decision-report'
    report_id: str
    context: str
    config: ObjectIdentity
    prediction_lock: ObjectIdentity
    description: ObjectIdentity
    forecast: ObjectIdentity
    target_seed_sha256: str
    data_sha256: str

    def __post_init__(self) -> None:
        for field in ("data_sha256", "target_seed_sha256"):
            validate_sha256(getattr(self, field), field_name=field)
        if (
            self.context not in CONTEXTS
            or self.report_id != f"{DEVELOPMENT}.decision.{self.context}"
            or self.config.object_schema != PreparationMethodConfig.SCHEMA
            or self.prediction_lock.object_schema != PreparationFitReport.SCHEMA
            or self.description.object_schema != PreparationDescriptionReport.SCHEMA
            or self.forecast.object_schema != PreparationForecastReport.SCHEMA
        ):
            raise ValueError(
                "decision report changes its pretarget lock or causal information inputs"
            )


@dataclass(frozen=True, slots=True)
class PreparationTaskAssessmentReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-task-assessment-report'
    report_id: str
    context: str
    config: ObjectIdentity
    decision: ObjectIdentity
    description: ObjectIdentity
    forecast: ObjectIdentity
    input_reports: tuple[ObjectIdentity, ...]
    prospective_task_reports: tuple[ObjectIdentity, ...]
    metrics: tuple[NamedDecimal, ...]
    adequacy_fixed_parent: str | None
    task_fixed_parent: str | None
    reference_candidate: str | None
    gate: PreparationGate
    data_sha256: str

    def __post_init__(self) -> None:
        from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import PARENTS
        from .contracts import PreparationProspectiveTaskReport

        _projection_inputs(self.context, self.input_reports)
        require_sorted_unique_ids(
            self.prospective_task_reports, attribute="object_id", field_name="prospective_task_reports"
        )
        validate_sha256(self.data_sha256, field_name="data_sha256")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        expected = {
            f"report.{r.root_id}.prospective-task.r{v}"
            for r in preparation_roots()
            if r.context == self.context
            for v in (1, 2)
        }
        if (
            self.report_id != f"{DEVELOPMENT}.task-assessment.{self.context}"
            or self.config.object_schema != PreparationMethodConfig.SCHEMA
            or self.decision.object_schema != PreparationDecisionReport.SCHEMA
            or self.description.object_schema != PreparationDescriptionReport.SCHEMA
            or self.forecast.object_schema != PreparationForecastReport.SCHEMA
            or {r.object_id for r in self.prospective_task_reports} != expected
            or any(r.object_schema != PreparationProspectiveTaskReport.SCHEMA for r in self.prospective_task_reports)
            or self.gate.role != "prospective-task-feasibility"
            or self.gate.context != self.context
            or self.adequacy_fixed_parent not in (*PARENTS, None)
            or self.task_fixed_parent not in (*PARENTS, None)
            or self.reference_candidate not in (*CANDIDATES, None)
        ):
            raise ValueError("task assessment changes its frozen decision or full root census")
