"""Closed development task/port declarations shared by authoring and workers."""

from dataclasses import dataclass

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import CONTEXTS, DEVELOPMENT, NATIVE_SCHEMA, PreparationNativeResult, preparation_roots
from .closeout import PreparationCloseoutConfig, PreparationDevelopmentResult
from .contracts import FIT_SCHEMA, PRIVILEGED_SCHEMA, PROJECTION_SCHEMA, PROSPECTIVE_TASK_SCHEMA, PreparationMethodConfig, PreparationPrivilegedReport, PreparationProjectionConfig, PreparationProjectionReport, PreparationProspectiveTaskReport
from .data import BENCHMARK_SCHEMA, DECISION_SCHEMA, DESCRIPTION_SCHEMA, FORECAST_SCHEMA, NUMERICAL_SEMANTICS_SCHEMA, TASK_ASSESSMENT_SCHEMA
from .records import PreparationForecastConfig, PreparationDecisionConfig, PreparationTaskAssessmentConfig, PreparationAssessmentConfig, PreparationBenchmarkReport, PreparationDecisionReport, PreparationDescriptionReport, PreparationFitReport, PreparationForecastReport, PreparationNumericalSemanticsConfig, PreparationNumericalSemanticsReport, PreparationTaskAssessmentReport
from .terminal import PreparationLawReport


STAGE_SCHEMA = LinkedCampaignStageEnvelope.SCHEMA
METHOD_ROLES = (
    "projection",
    "numerical-semantics",
    "fit",
    "description",
    "forecast",
    "decision",
    "task-assessment",
    "evaluation",
)
METHOD_CONFIG_TYPES: dict[str, type[CanonicalRecord]] = {
    "projection": PreparationProjectionConfig,
    "numerical-semantics": PreparationNumericalSemanticsConfig,
    "fit": PreparationMethodConfig,
    "description": PreparationAssessmentConfig,
    "forecast": PreparationForecastConfig,
    "decision": PreparationDecisionConfig,
    "task-assessment": PreparationTaskAssessmentConfig,
    "evaluation": PreparationCloseoutConfig,
}
RECORD_OUTPUTS: dict[str, tuple[tuple[str, type[CanonicalRecord]], ...]] = {
    "source": (("native-result", PreparationNativeResult),),
    "projection": (
        ("report", PreparationProjectionReport),
        ("privileged-report", PreparationPrivilegedReport),
        ("prospective-task-report", PreparationProspectiveTaskReport),
    ),
    "numerical-semantics": (("report", PreparationNumericalSemanticsReport),),
    "fit": (("report", PreparationFitReport),),
    "description": (
        ("report", PreparationDescriptionReport),
        ("benchmark-report", PreparationBenchmarkReport),
        ("law-report", PreparationLawReport),
    ),
    "forecast": (("report", PreparationForecastReport),),
    "decision": (("report", PreparationDecisionReport),),
    "task-assessment": (("report", PreparationTaskAssessmentReport),),
    "evaluation": (
        ("report", PreparationDevelopmentResult),
        ("scientific-adjudication", ScientificAdjudicationRecord),
    ),
}
DATA_OUTPUTS: dict[str, tuple[tuple[str, str], ...]] = {
    "source": (("native-observations", NATIVE_SCHEMA),),
    "projection": (
        ("data", PROJECTION_SCHEMA),
        ("privileged-data", PRIVILEGED_SCHEMA),
        ("prospective-task-data", PROSPECTIVE_TASK_SCHEMA),
    ),
    "numerical-semantics": (("data", NUMERICAL_SEMANTICS_SCHEMA),),
    "fit": (("data", FIT_SCHEMA),),
    "description": (("data", DESCRIPTION_SCHEMA), ("benchmark-data", BENCHMARK_SCHEMA)),
    "forecast": (("data", FORECAST_SCHEMA),),
    "decision": (("data", DECISION_SCHEMA),),
    "task-assessment": (("data", TASK_ASSESSMENT_SCHEMA),),
    "evaluation": (),
}
INPUT_SCHEMAS: dict[str, tuple[str, ...]] = {
    "projection": (PreparationNativeResult.SCHEMA, NATIVE_SCHEMA),
    "numerical-semantics": (
        PreparationProjectionReport.SCHEMA,
        PROJECTION_SCHEMA,
        PreparationPrivilegedReport.SCHEMA,
        PRIVILEGED_SCHEMA,
    ),
    "fit": (
        PreparationProjectionReport.SCHEMA,
        PROJECTION_SCHEMA,
        PreparationNumericalSemanticsReport.SCHEMA,
    ),
    "description": (
        PreparationProjectionReport.SCHEMA,
        PROJECTION_SCHEMA,
        PreparationPrivilegedReport.SCHEMA,
        PRIVILEGED_SCHEMA,
        PreparationFitReport.SCHEMA,
        FIT_SCHEMA,
        PreparationNumericalSemanticsReport.SCHEMA,
    ),
    "forecast": (
        PreparationProjectionReport.SCHEMA,
        PROJECTION_SCHEMA,
        PreparationFitReport.SCHEMA,
        FIT_SCHEMA,
        PreparationDescriptionReport.SCHEMA,
    ),
    "decision": (
        PreparationFitReport.SCHEMA,
        FIT_SCHEMA,
        PreparationDescriptionReport.SCHEMA,
        PreparationForecastReport.SCHEMA,
        FORECAST_SCHEMA,
    ),
    "task-assessment": (
        PreparationProjectionReport.SCHEMA,
        PROJECTION_SCHEMA,
        PreparationProspectiveTaskReport.SCHEMA,
        PROSPECTIVE_TASK_SCHEMA,
        PreparationFitReport.SCHEMA,
        FIT_SCHEMA,
        PreparationDescriptionReport.SCHEMA,
        PreparationForecastReport.SCHEMA,
        FORECAST_SCHEMA,
        PreparationDecisionReport.SCHEMA,
        DECISION_SCHEMA,
    ),
    "evaluation": (
        PreparationNumericalSemanticsReport.SCHEMA,
        NUMERICAL_SEMANTICS_SCHEMA,
        PreparationDescriptionReport.SCHEMA,
        DESCRIPTION_SCHEMA,
        PreparationBenchmarkReport.SCHEMA,
        BENCHMARK_SCHEMA,
        PreparationLawReport.SCHEMA,
        PreparationForecastReport.SCHEMA,
        FORECAST_SCHEMA,
        PreparationDecisionReport.SCHEMA,
        DECISION_SCHEMA,
        PreparationTaskAssessmentReport.SCHEMA,
        TASK_ASSESSMENT_SCHEMA,
    ),
}
OUTPUT_MIB = {
    "source": 98,
    "projection": 8,
    "numerical-semantics": 18,
    "fit": 18,
    "description": 52,
    "forecast": 18,
    "decision": 18,
    "task-assessment": 18,
    "evaluation": 4,
}
WALL_SECONDS = {
    "source": 3600,
    "projection": 1800,
    "numerical-semantics": 120,
    "fit": 900,
    "description": 900,
    "forecast": 900,
    "decision": 120,
    "task-assessment": 120,
    "evaluation": 120,
}


@dataclass(frozen=True, slots=True)
class PreparationTaskDeclaration:
    task_id: str
    role: str
    dependencies: tuple[str, ...]


def preparation_task_declarations() -> tuple[PreparationTaskDeclaration, ...]:
    tasks = [
        PreparationTaskDeclaration(f"{r.root_id}.native", "source", ())
        for r in preparation_roots()
    ]
    tasks.extend(
        PreparationTaskDeclaration(
            f"{r.root_id}.project.r{v}", "projection", (f"{r.root_id}.native",)
        )
        for r in preparation_roots()
        for v in (1, 2)
    )
    for context in CONTEXTS:
        projected = tuple(
            f"{r.root_id}.project.r{v}"
            for r in preparation_roots()
            if r.context == context
            for v in (1, 2)
        )
        ids = {
            role: f"{DEVELOPMENT}.{role}.{context}"
            for role in METHOD_ROLES
            if role not in ("projection", "evaluation")
        }
        dependencies = {
            "numerical-semantics": projected,
            "fit": (*projected, ids["numerical-semantics"]),
            "description": (*projected, ids["numerical-semantics"], ids["fit"]),
            "forecast": (*projected, ids["fit"], ids["description"]),
            "decision": (ids["fit"], ids["description"], ids["forecast"]),
            "task-assessment": (
                *projected,
                ids["fit"],
                ids["description"],
                ids["forecast"],
                ids["decision"],
            ),
        }
        tasks.extend(
            PreparationTaskDeclaration(ids[role], role, tuple(sorted(dependencies[role])))
            for role in dependencies
        )
    tasks.append(
        PreparationTaskDeclaration(
            f"{DEVELOPMENT}.evaluate",
            "evaluation",
            tuple(
                sorted(
                    f"{DEVELOPMENT}.{role}.{c}"
                    for c in CONTEXTS
                    for role in ("numerical-semantics", "description", "forecast", "decision", "task-assessment")
                )
            ),
        )
    )
    if len(tasks) != 397 or len({t.task_id for t in tasks}) != 397:
        raise ValueError("development topology changes its fixed 397-task census")
    return tuple(tasks)


def task_output_schemas(role: str) -> dict[str, str]:
    return {
        **{name: kind.SCHEMA for name, kind in RECORD_OUTPUTS[role]},
        **dict(DATA_OUTPUTS[role]),
        "stage-envelope" if role == "source" else "stage": STAGE_SCHEMA,
    }


def expected_method_inputs(
    task: PreparationTaskDeclaration, config_id: str, config_schema: str
) -> dict[str, str]:
    if task.role not in METHOD_ROLES:
        raise ValueError("method input declaration cannot create a native source port")
    by_id = {d.task_id: d for d in preparation_task_declarations()}
    if by_id.get(task.task_id) != task:
        raise ValueError("method input declaration changes its fixed task dependencies")
    accepted = {*INPUT_SCHEMAS[task.role], STAGE_SCHEMA}
    result = {f"config-artifact.{config_id}": config_schema}
    for dependency in task.dependencies:
        result.update(
            {
                f"{dependency}.{name}": schema
                for name, schema in task_output_schemas(by_id[dependency].role).items()
                if schema in accepted
            }
        )
    return result
