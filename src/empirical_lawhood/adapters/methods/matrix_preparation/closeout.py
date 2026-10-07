"Authenticated development terminals with explicit conditional continuation scope."

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
)
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import CONTEXTS, DEVELOPMENT
from .contracts import CANDIDATES
from .records import PreparationAssessmentConfig, PreparationBenchmarkReport, PreparationDecisionReport, PreparationDescriptionReport, PreparationForecastReport, PreparationGate, PreparationNumericalSemanticsReport, PreparationTaskAssessmentReport
from .terminal import PreparationLawReport


@dataclass(frozen=True, slots=True)
class PreparationCloseoutConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-closeout-config'
    config_id: str
    assessment: PreparationAssessmentConfig

    def __post_init__(self) -> None:
        if self.config_id != f"{DEVELOPMENT}.closeout-config":
            raise ValueError("preparation closeout has another campaign identity")


@dataclass(frozen=True, slots=True)
class PreparationContextDisposition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-context-disposition'
    context: str
    input_reports: tuple[ObjectIdentity, ...]
    gates: tuple[PreparationGate, ...]
    selected_observable_candidate: str | None
    law_status: ScientificStatus
    fresh_branch_disposition: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_ids(
            self.input_reports, attribute="object_id", field_name="input_reports"
        )
        require_sorted_unique_strings(self.reasons, field_name="reasons")
        if (
            self.context not in CONTEXTS
            or tuple(g.role for g in self.gates) != ("numerical-semantics", "observable-description", "preparation-window-forecast", "prospective-task-feasibility")
            or any(g.context != self.context for g in self.gates)
            or len(self.input_reports) != 7
            or self.selected_observable_candidate not in (*CANDIDATES, None)
            or self.fresh_branch_disposition
            != (
                "CONDITIONAL_FRESH_FREEZE_REQUIRED"
                if all(g.passes for g in self.gates)
                else "NOT_ENTERED_CONDITION_FALSE"
            )
        ):
            raise ValueError("development closeout changes a context/gate/conditional continuation disposition")


def aggregate_law_status(
    contexts: tuple[PreparationContextDisposition, ...],
) -> ScientificStatus:
    statuses = {c.law_status for c in contexts}
    if len(statuses) == 1:
        return next(iter(statuses))
    return (
        ScientificStatus.MIXED
        if ScientificStatus.SUPPORTED in statuses
        else ScientificStatus.UNEVALUABLE
        if ScientificStatus.UNEVALUABLE in statuses
        else ScientificStatus.NOT_SUPPORTED
    )


@dataclass(frozen=True, slots=True)
class PreparationDevelopmentResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-preparation/preparation-development-result'
    result_id: str
    config: ObjectIdentity
    contexts: tuple[PreparationContextDisposition, ...]
    scientific_status: ScientificStatus
    original_exposed_roots: int = 128
    independent_new_efficacy_roots: int = 0
    reasons: tuple[str, ...] = (
        "DEVELOPMENT_GATES_AND_FORMAL_LAW_STATUS_REMAIN_SEPARATE",
        "FRESH_CONFIRMATION_NOT_EXECUTED_BY_THIS_PACKAGE",
        "ORIGINAL_EXPOSED_ROOTS_NEVER_RELABELED_FRESH",
    )

    def __post_init__(self) -> None:
        if (
            self.result_id != f"{DEVELOPMENT}.evaluate"
            or self.config.object_schema != PreparationCloseoutConfig.SCHEMA
            or tuple(c.context for c in self.contexts) != CONTEXTS
            or self.scientific_status is not aggregate_law_status(self.contexts)
            or type(self.original_exposed_roots) is not int
            or self.original_exposed_roots != 128
            or type(self.independent_new_efficacy_roots) is not int
            or self.independent_new_efficacy_roots != 0
        ):
            raise ValueError(
                "development terminal changes scientific-owner status or its root census"
            )
        require_sorted_unique_strings(self.reasons, field_name="reasons", allow_empty=False)


def context_disposition(
    config: PreparationAssessmentConfig,
    numerical_semantics: PreparationNumericalSemanticsReport,
    description: PreparationDescriptionReport,
    benchmark: PreparationBenchmarkReport,
    law: PreparationLawReport,
    forecast: PreparationForecastReport,
    decision: PreparationDecisionReport,
    task: PreparationTaskAssessmentReport,
) -> PreparationContextDisposition:
    context = numerical_semantics.context
    records = (numerical_semantics, description, benchmark, law, forecast, decision, task)
    description_ref = ObjectIdentity.from_record(description.report_id, description)
    forecast_ref = ObjectIdentity.from_record(forecast.report_id, forecast)
    if (
        any(r.context != context for r in records)
        or description.numerical_semantics != ObjectIdentity.from_record(numerical_semantics.report_id, numerical_semantics)
        or any(r.description != description_ref for r in (benchmark, law, forecast, decision, task))
        or any(r.forecast != forecast_ref for r in (decision, task))
        or task.decision != ObjectIdentity.from_record(decision.report_id, decision)
        or description.config != ObjectIdentity.from_record(config.config_id, config)
        or law.config != description.config
        or numerical_semantics.config != ObjectIdentity.from_record(config.numerical_semantics.config_id, config.numerical_semantics)
        or forecast.config != ObjectIdentity.from_record(config.method.config_id, config.method)
    ):
        raise ValueError(
            "development terminal changes its actual input receipts or scientific lineage"
        )
    numerical_gate = PreparationGate(
        f"{DEVELOPMENT}.gate.numerical-semantics.{context}",
        context,
        "numerical-semantics",
        numerical_semantics.numerical_semantics_qualified,
        numerical_semantics.reasons,
    )
    description_gate = PreparationGate(
        f"{DEVELOPMENT}.gate.observable-description.{context}",
        context,
        "observable-description",
        bool(description.qualified_candidates),
        ()
        if description.qualified_candidates
        else ("NO_OBSERVABLE_CANDIDATE_PASSES_FROZEN_DESCRIPTION_SCREEN",),
    )
    gates = (numerical_gate, description_gate, forecast.gate, task.gate)
    passes = all(g.passes for g in gates)
    reasons = tuple(sorted({reason for g in gates for reason in g.reasons}))
    return PreparationContextDisposition(
        context,
        tuple(
            sorted(
                (ObjectIdentity.from_record(r.report_id, r) for r in records),
                key=lambda r: r.object_id,
            )
        ),
        gates,
        description.selected_candidate,
        law.qualification.scientific_status,
        "CONDITIONAL_FRESH_FREEZE_REQUIRED" if passes else "NOT_ENTERED_CONDITION_FALSE",
        reasons,
    )


def close_preparation_development(
    config: PreparationCloseoutConfig, contexts: tuple[PreparationContextDisposition, ...]
) -> PreparationDevelopmentResult:
    return PreparationDevelopmentResult(
        f"{DEVELOPMENT}.evaluate",
        ObjectIdentity.from_record(config.config_id, config),
        contexts,
        aggregate_law_status(contexts),
    )
