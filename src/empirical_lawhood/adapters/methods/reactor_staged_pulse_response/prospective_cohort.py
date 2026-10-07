"Fixed all-assigned 64-root controller use tests over existing owner evaluations."

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.qualification import cp
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from empirical_lawhood.runtime.controller_evaluation_nested import NestedControllerUseEvaluator, PreparedPolicyUnitEvaluation
from .config import ARMS, LOCAL_REQUESTS, MENUS, PAIR_REQUESTS, SEQUENCES, roots
from .control_services import implementation_payloads
from .prospective_plan import ReactorStagedPulseResponseProspectivePlan
from .prospective_reveal import ClassicalRevealedRoot
from empirical_lawhood.planning.controller_study import ImplementationRole


@dataclass(frozen=True)
class ClassicalCohortUse:
    """Evaluated operands only; full child archives remain in root custody."""

    policy: str
    request_id: str
    evaluable: bool
    strict_success: bool
    common_service: bool
    false_admission: bool | None
    unsafe: bool
    attempted_mass_kg: D | None
    units: tuple[PreparedPolicyUnitEvaluation | None, ...]


@dataclass(frozen=True)
class ClassicalCohortRoot:
    root: str
    identity: ObjectIdentity
    uses: tuple[ClassicalCohortUse, ...]


def cohort_operand(root: ClassicalRevealedRoot) -> ClassicalCohortRoot:
    return ClassicalCohortRoot(
        root.root,
        ObjectIdentity.from_record(root.record_id, root),
        tuple(
            ClassicalCohortUse(
                u.policy,
                u.request_id,
                u.evaluable,
                u.strict_success,
                u.common_service,
                u.false_admission,
                u.unsafe,
                u.attempted_mass_kg,
                tuple(s.unit for s in u.stages),
            )
            for u in root.uses
        ),
    )


@dataclass(frozen=True, slots=True)
class ReactorStagedPulseResponsePolicyProspective(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/reactor-staged-pulse-response-policy-prospective'
    policy: str
    prerequisite_entered: bool
    roots: tuple[tuple[str, bool, bool, bool | None, bool], ...]
    family_size: int
    successes: int
    false_admissions: int
    unsafe_roots: int
    success_lower: D
    false_upper: D
    status: ScientificStatus
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.roots) != 64
            or len({r[0] for r in self.roots}) != 64
            or self.family_size not in (8, 4, 6)
            or self.successes != sum(r[2] for r in self.roots)
            or self.false_admissions != sum(r[3] is True for r in self.roots)
            or self.unsafe_roots != sum(r[4] for r in self.roots)
        ):
            raise ValueError("policy controller-use result changes its independent root census")


@dataclass(frozen=True, slots=True)
class ReactorStagedPulseResponseProspectiveCohort(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/reactor-staged-pulse-response-prospective-cohort'
    block: str
    plan: ObjectIdentity
    roots: tuple[ObjectIdentity, ...]
    owners: tuple[CanonicalRecordArchive, ...]
    policies: tuple[ReactorStagedPulseResponsePolicyProspective, ...]
    uninvoked_child_units: tuple[tuple[ObjectIdentity, tuple[str, ...]], ...] = ()

    @property
    def record_id(self) -> str:
        return f"reactor-staged-pulse-response.{self.block}.prospective-evaluation-cohort"


def reduce_cohort(
    plan: ReactorStagedPulseResponseProspectivePlan, revealed: tuple[ClassicalRevealedRoot, ...]
) -> ReactorStagedPulseResponseProspectiveCohort:
    return reduce_cohort_operands(plan, tuple(cohort_operand(r) for r in revealed))


def reduce_cohort_operands(
    plan: ReactorStagedPulseResponseProspectivePlan, revealed: tuple[ClassicalCohortRoot, ...]
) -> ReactorStagedPulseResponseProspectiveCohort:
    assigned = roots(plan.block, "prospective")
    policies = ARMS if plan.block == "base-menu-comparison" else MENUS if plan.block == "expanded-menu-comparison" else SEQUENCES
    requests = LOCAL_REQUESTS if plan.block != "staged-sequence-comparison" else PAIR_REQUESTS
    expected = tuple((p, r.request_id) for p in policies for r in requests)
    if tuple(r.root for r in revealed) != assigned or any(
        tuple((u.policy, u.request_id) for u in r.uses) != expected for r in revealed
    ):
        raise ValueError("Controller-use reduction drops or substitutes an assigned owner/root/request")
    evaluator = next(
        b for b, _ in implementation_payloads() if b.role is ImplementationRole.OUTCOME_EVALUATOR
    )
    owners = []
    uninvoked = []
    for policy in policies:
        for request in requests:
            kinds = (
                ("local",)
                if plan.block != "staged-sequence-comparison"
                else ("baseline",)
                if policy == "ONE_PULSE"
                else ("first", "joint")
            )
            for kind in kinds:
                use = plan.use(policy, request, kind)
                if use.plan is None:
                    continue
                units = tuple(
                    unit
                    for r in revealed
                    for u in r.uses
                    if (u.policy, u.request_id) == (policy, request.request_id)
                    for i, unit in enumerate(u.units)
                    if kinds[i] == kind and unit is not None
                )
                expected_roots = tuple(r.root_id for r in use.plan.roots)
                actual_roots = tuple(u.root_id for u in units)
                if len(set(actual_roots)) != len(actual_roots) or not set(actual_roots) <= set(
                    expected_roots
                ):
                    raise ValueError(
                        "child census duplicates or substitutes an assigned prepared root"
                    )
                missing = tuple(r for r in expected_roots if r not in actual_roots)
                if missing:
                    # No second invocation after an early stop is not a new
                    # prepared execution. Keep its exact missing role; the
                    # bounded sequence owner and full outer census below
                    # determine whether the episode is known or unknown.
                    uninvoked.append(
                        (ObjectIdentity.from_record(use.plan.evaluation_plan_id, use.plan), missing)
                    )
                    continue
                cohort = NestedControllerUseEvaluator(
                    evaluator, use.plan.reducer
                ).reduce_prepared_cohort(plan=use.plan, units=units)
                owners.append(CanonicalRecordArchive.pack(cohort.evaluation_id, cohort))
    results = []
    family = {"base-menu-comparison": 8, "expanded-menu-comparison": 4, "staged-sequence-comparison": 6}[plan.block]
    for policy in policies:
        entries = []
        for r in revealed:
            uses = tuple(u for u in r.uses if u.policy == policy)
            known = all(u.evaluable for u in uses)
            unsafe = any(u.unsafe for u in uses)
            false = (
                True
                if any(u.false_admission is True for u in uses)
                else False
                if all(u.false_admission is False for u in uses)
                else None
            )
            success = (
                known
                and sum(u.strict_success for u in uses) >= (3 if plan.block != "staged-sequence-comparison" else 2)
                and false is False
                and not unsafe
            )
            entries.append((r.root, known, success, false, unsafe))
        entered = any(
            plan.use(
                policy,
                r,
                "local" if plan.block != "staged-sequence-comparison" else "baseline" if policy == "ONE_PULSE" else "first",
            ).prerequisite_entered
            for r in requests
        )
        n_j, n_f, n_unsafe = (
            sum(r[2] for r in entries),
            sum(r[3] is True for r in entries),
            sum(r[4] for r in entries),
        )
        lower, upper = (
            cp(n_j, 64, D(".05") / family, lower=True),
            cp(n_f, 64, D(".05") / family, lower=False),
        )
        complete = all(r[1] and r[3] is not None for r in entries)
        reasons = tuple(
            reason
            for fail, reason in (
                (not entered, "LOCAL_LAW_PREREQUISITE_NONENTRY"),
                (not complete, "MANDATORY_EVIDENCE_UNKNOWN"),
                (n_j < 56 or lower < D(".70"), "WHOLE_ROOT_SERVICE_GATE_FAILED"),
                (n_f != 0 or upper > D(".10"), "OWN_LAW_FALSE_ADMISSION_GATE_FAILED"),
                (n_unsafe != 0, "NATIVE_SAFETY_GATE_FAILED"),
            )
            if fail
        )
        status = (
            ScientificStatus.UNEVALUABLE
            if not entered or not complete
            else ScientificStatus.NOT_SUPPORTED
            if reasons
            else ScientificStatus.SUPPORTED
        )
        results.append(
            ReactorStagedPulseResponsePolicyProspective(
                policy,
                entered,
                tuple(entries),
                family,
                n_j,
                n_f,
                n_unsafe,
                lower,
                upper,
                status,
                reasons,
            )
        )
    return ReactorStagedPulseResponseProspectiveCohort(
        plan.block,
        ObjectIdentity.from_record(plan.record_id, plan),
        tuple(r.identity for r in revealed),
        tuple(owners),
        tuple(results),
        tuple(uninvoked),
    )
