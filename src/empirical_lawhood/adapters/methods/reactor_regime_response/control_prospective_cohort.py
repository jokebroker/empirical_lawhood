"All-assigned reactor D controller-use reduction and frozen policy comparisons."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.controller_evaluation_nested import NestedControllerUseEvaluator, PreparedControllerCohortEvaluation

from .comparison_math import POLICIES, compare_direct, paired_descriptive_effects
from .config import ROOTS
from .control_math import one_sided_cp
from .control_prospective_plan import ReactorRegimeResponsePreparedProspectivePlanBundle
from .control_prospective_reveal import RegimeDRootRevealResult


@dataclass(frozen=True, slots=True)
class RegimeDDirectComparison(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-d-direct-comparison'

    evaluable: bool
    el_success_roots: int
    direct_success_roots: int
    el_only: int
    direct_only: int
    exact_one_sided_p: D | None
    improvement_fraction: D | None
    added_service_supported: bool
    reason: str | None

    def __post_init__(self) -> None:
        if (
            (self.evaluable and (
                self.exact_one_sided_p is None
                or self.improvement_fraction is None
                or self.reason is not None
            ))
            or (not self.evaluable and (
                self.exact_one_sided_p is not None
                or self.improvement_fraction is not None
                or self.reason is None
                or self.added_service_supported
            ))
            or not all(0 <= count <= 64 for count in (
                self.el_success_roots, self.direct_success_roots,
                self.el_only, self.direct_only,
            ))
        ):
            raise ValueError("D DIRECT comparison lost its all-assigned paired contrast")


@dataclass(frozen=True, slots=True)
class RegimeDDescriptiveEffect(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-d-descriptive-effect'

    rival: str
    metric: str
    mean_EL_minus_rival: D
    lower_95: D
    upper_95: D
    draws: int
    seed: int

    def __post_init__(self) -> None:
        if (
            self.rival not in POLICIES[1:]
            or self.metric not in (
                "all_four_success_fraction", "mean_delivered_mass_kg",
                "refusal_fraction", "failure_penalized_feed_loss_kg",
            )
            or self.draws != 20000 or self.seed != 20260925
            or self.lower_95 > self.upper_95
        ):
            raise ValueError("D descriptive bootstrap changed its frozen root unit")


@dataclass(frozen=True, slots=True)
class RegimeDLeaveOneOut(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/regime-d-leave-one-out'

    removed_root: str
    remaining_joined: int
    remaining_false_admission: int
    joined_lower_95: D
    false_admission_upper_95: D
    adequate_use: bool


@dataclass(frozen=True, slots=True)
class ReactorRegimeResponseCohortProspective(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/reactor-regime-response-cohort-prospective'

    plan: ObjectIdentity
    roots: tuple[ObjectIdentity, ...]
    generic_owner: PreparedControllerCohortEvaluation | None
    assigned_count: int
    A_count: int
    J_count: int
    C_count: int
    F_count: int
    admitted_requests: int
    admitted_roots: int
    joined_lower_95: D
    false_admission_upper_95: D
    adequate_use: bool
    direct: RegimeDDirectComparison
    descriptive: tuple[RegimeDDescriptiveEffect, ...]
    unevaluable_policies: tuple[str, ...]
    leave_one_out: tuple[RegimeDLeaveOneOut, ...]
    outlier_roots: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            self.plan.object_schema != ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA
            or self.assigned_count != 64
            or len(self.roots) != 64
            or len({value.object_id for value in self.roots}) != 64
            or len(self.leave_one_out) != 64
            or not all(0 <= count <= 64 for count in (
                self.A_count, self.J_count, self.C_count, self.F_count,
            ))
            or not 0 <= self.admitted_requests <= 256
            or not 0 <= self.admitted_roots <= 64
            or self.C_count > min(self.A_count, self.J_count)
            or self.adequate_use != (
                self.joined_lower_95 >= D(".70")
                and self.false_admission_upper_95 <= D(".10")
            )
        ):
            raise ValueError("D controller use cohort changes its exact 64-root A/J/C/F census")


def reduce_d_cohort(
    *,
    plan_bundle: ReactorRegimeResponsePreparedProspectivePlanBundle,
    roots: tuple[RegimeDRootRevealResult, ...],
) -> ReactorRegimeResponseCohortProspective:
    expected = tuple(root for root, role, _, _ in ROOTS if role == "prospective")
    if (
        tuple(row.root for row in roots) != expected
        or tuple(row.root for row in plan_bundle.assigned_roots) != expected
    ):
        raise ValueError("D controller use cohort lost or replaced an assigned independent root")
    generic = None
    if plan_bundle.plan is not None:
        units = tuple(unit for row in roots for unit in row.units)
        if len(units) != 4 * len(plan_bundle.plan.roots):
            raise ValueError("D controller use cohort lacks an installed unit for each eligible request")
        evaluator = NestedControllerUseEvaluator(
            roots[next(index for index, row in enumerate(roots) if row.units)]
            .revealed[0].sealed.design.evaluator_binding,
            plan_bundle.plan.reducer,
        )
        generic = evaluator.reduce_prepared_cohort(
            plan=plan_bundle.plan, units=units
        )
    counts = (
        sum(row.A for row in roots), sum(row.J for row in roots),
        sum(row.C for row in roots), sum(row.F for row in roots),
    )
    lower = one_sided_cp(counts[2], 64, lower=True)
    upper = one_sided_cp(counts[3], 64, lower=False)
    adequate = lower >= .70 and upper <= .10
    by_policy = {
        policy: tuple(row.policies[index] for row in roots)
        for index, policy in enumerate(POLICIES)
    }
    el = tuple(row.to_comparison() for row in by_policy["EL"])
    direct_rows = by_policy["DIRECT"]
    direct = None if not all(row.evaluable for row in direct_rows) else tuple(
        row.to_comparison() for row in direct_rows
    )
    result = compare_direct(el, direct, primary_prospective_evaluation_passed=adequate)
    direct_record = RegimeDDirectComparison(
        result.evaluable, result.el_success_roots, result.direct_success_roots,
        result.el_only, result.direct_only,
        None if result.paired_exact_one_sided_p is None else D(repr(result.paired_exact_one_sided_p)),
        None if result.improvement_fraction is None else D(repr(result.improvement_fraction)),
        result.added_service_supported, result.reason,
    )
    descriptive = []
    unevaluable = []
    for policy in POLICIES[1:]:
        rows = by_policy[policy]
        if not all(row.evaluable for row in rows):
            unevaluable.append(policy)
            continue
        effects = paired_descriptive_effects(
            el, tuple(row.to_comparison() for row in rows)
        )
        raw = effects["effects_EL_minus_rival"]
        assert isinstance(raw, dict)
        for metric, value in sorted(raw.items()):
            assert isinstance(value, dict)
            bounds = value["percentile_95"]
            assert isinstance(bounds, tuple)
            descriptive.append(RegimeDDescriptiveEffect(
                policy, metric, D(repr(value["mean"])),
                D(repr(bounds[0])), D(repr(bounds[1])),
                20000, 20260925,
            ))
    sensitivity = []
    for row in roots:
        remaining_c = counts[2] - int(row.C)
        remaining_f = counts[3] - int(row.F)
        local_lower = one_sided_cp(remaining_c, 63, lower=True)
        local_upper = one_sided_cp(remaining_f, 63, lower=False)
        sensitivity.append(RegimeDLeaveOneOut(
            row.root, remaining_c, remaining_f,
            D(repr(local_lower)), D(repr(local_upper)),
            local_lower >= .70 and local_upper <= .10,
        ))
    outliers = tuple(
        row.root for row in roots
        if not row.C or row.F or row.reasons
    )
    return ReactorRegimeResponseCohortProspective(
        ObjectIdentity.from_record("regime.d-plan", plan_bundle),
        tuple(ObjectIdentity.from_record(f"{row.root}.d-reveal", row) for row in roots),
        generic, 64, *counts,
        sum(unit.admitted for row in roots for unit in row.units),
        sum(any(unit.admitted for unit in row.units) for row in roots),
        D(repr(lower)), D(repr(upper)), adequate,
        direct_record,
        tuple(descriptive), tuple(sorted(unevaluable)),
        tuple(sensitivity), outliers,
    )
