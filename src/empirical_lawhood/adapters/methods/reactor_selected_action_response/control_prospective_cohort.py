"All-assigned 32-root controller-use result, after the existing prepared unit owners."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from scipy.stats import beta, t

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.runtime.controller_evaluation import controller_prospective_student_summary
from empirical_lawhood.runtime.controller_evaluation_nested import NestedControllerUseEvaluator, PreparedControllerCohortEvaluation
from .config import ROOTS, ClassicalDesign
from .control_prospective_plan import ReactorSelectedActionResponseProspectivePlan
from .control_prospective_reveal import ClassicalRevealResult


@dataclass(frozen=True, slots=True)
class ReactorSelectedActionResponseProspectiveCohort(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/reactor-selected-action-response-prospective-cohort'
    plan: ObjectIdentity
    roots: tuple[ObjectIdentity, ...]
    generic_owner: PreparedControllerCohortEvaluation | None
    assigned: int
    successes: int
    lower_95: D
    mean_cooling_K: D | None
    mean_lower_95_K: D | None
    scientific_status: ScientificStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.assigned != 32 or len(self.roots) != 32 or not 0 <= self.successes <= 32:
            raise ValueError("classical controller use cohort lost its assigned independent units")


def reduce_prospective(
    plan: ReactorSelectedActionResponseProspectivePlan, roots: tuple[ClassicalRevealResult, ...]
) -> ReactorSelectedActionResponseProspectiveCohort:
    expected = tuple(root for root, role, _, _ in ROOTS if role == "prospective")
    if tuple(row.root for row in roots) != expected:
        raise ValueError("classical controller-use root denominator changed")
    generic = None
    if plan.plan is not None:
        units = tuple(unit for row in roots for unit in row.units)
        if len(units) != len(plan.plan.roots):
            raise ValueError("classical controller use lacks an installed unit result for every entered root")
        evaluator = NestedControllerUseEvaluator(
            next(row for row in roots if row.units).revealed[0].sealed.design.evaluator_binding,
            plan.plan.reducer,
        )
        generic = evaluator.reduce_prepared_cohort(plan=plan.plan, units=units)
    successes = sum(row.success for row in roots)
    lower = D(0) if not successes else D(repr(float(beta.ppf(0.05, successes, 33 - successes))))
    mean = mean_lower = None
    evaluable = all(row.evaluable and row.robust_cooling_K is not None for row in roots)
    if evaluable:
        mean, _, mean_lower = controller_prospective_student_summary(
            tuple(row.robust_cooling_K for row in roots if row.robust_cooling_K is not None),
            one_sided_critical_value=D(repr(float(t.ppf(0.95, 31)))),
        )
    supported = (
        successes == 32 and mean_lower is not None and mean_lower > ClassicalDesign().request_K
    )
    status = (
        ScientificStatus.SUPPORTED
        if supported
        else ScientificStatus.NOT_SUPPORTED
        if evaluable or any(row.evaluable and not row.success for row in roots)
        else ScientificStatus.UNEVALUABLE
    )
    return ReactorSelectedActionResponseProspectiveCohort(
        ObjectIdentity.from_record("classical.prospective-evaluation-plan", plan),
        tuple(ObjectIdentity.from_record(f"{row.root}.prospective-evaluation-reveal", row) for row in roots),
        generic,
        32,
        successes,
        lower,
        mean,
        mean_lower,
        status,
        ()
        if supported
        else ("CLASSICAL_CONTROLLER_USE_PREREQUISITE_NONENTRY",)
        if plan.plan is None
        else ("CLASSICAL_CONTROLLER_USE_ALL_ASSIGNED_VALIDATED_USE_FAILED",),
    )
