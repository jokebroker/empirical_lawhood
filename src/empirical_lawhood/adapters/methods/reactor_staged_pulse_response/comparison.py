"""Predeclared paired whole-root service and feed-cost endpoint families."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar
from enum import StrEnum
import numpy as np

from empirical_lawhood.adapters.methods.whole_root_bootstrap import root_indices, nearest_rank_interval
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from .config import roots
from .prospective_cohort import ReactorStagedPulseResponseProspectiveCohort, ClassicalCohortRoot, cohort_operand
from .prospective_reveal import ClassicalRevealedRoot


class ContrastDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    UNRESOLVED = "UNRESOLVED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ClassicalContrast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-contrast'
    phase: str
    treatment: str
    comparator: str
    service_gain: D
    service_interval: tuple[D, D]
    service_status: ContrastDisposition
    common_requests: int
    cost_ratio: D | None
    cost_interval: tuple[D, D] | None
    cost_status: ContrastDisposition | None
    seed: int
    draws: int
    tail_probability: D
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ClassicalComparison(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-comparison'
    block: str
    cohort: ObjectIdentity
    roots: tuple[ObjectIdentity, ...]
    contrasts: tuple[ClassicalContrast, ...]

    @property
    def record_id(self) -> str:
        return f"reactor-staged-pulse-response.{self.block}.comparison"


def comparison(
    cohort: ReactorStagedPulseResponseProspectiveCohort, revealed: tuple[ClassicalRevealedRoot, ...]
) -> ClassicalComparison:
    return compare_operands(cohort, tuple(cohort_operand(r) for r in revealed))


def compare_operands(
    cohort: ReactorStagedPulseResponseProspectiveCohort, revealed: tuple[ClassicalCohortRoot, ...]
) -> ClassicalComparison:
    identities = tuple(r.identity for r in revealed)
    if identities != cohort.roots or tuple(r.root for r in revealed) != roots(
        cohort.block, "prospective"
    ):
        raise ValueError("comparison changes the fixed root census or owned controller-use result")
    specifications = (
        (
            ("I", "EL_SERVICE", "EL_INTERVAL", 0.025, False),
            ("II", "EL_SERVICE", "DIRECT_SAFE", 0.05 / 8, True),
            ("II", "EL_SERVICE", "MECH_SAFE", 0.05 / 8, True),
        )
        if cohort.block == "base-menu-comparison"
        else (("III", "EXPANDED", "BASE", 0.05 / 4, True),)
        if cohort.block == "expanded-menu-comparison"
        else (("IV", "EL_SEQUENCE", "FIXED_SEQUENCE", 0.05 / 4, True),)
    )
    seed = {"base-menu-comparison": 20260927, "expanded-menu-comparison": 20260928, "staged-sequence-comparison": 20260929}[cohort.block]
    indices = root_indices(roots=64, draws=20000, seed=seed)
    results = []
    for phase, treatment, comparator, tail, cost_endpoint in specifications:
        gain, numerators, denominators, common = [], [], [], 0
        complete = True
        cost_panel = ("EARLY_TEN_SECOND_REQUEST", "EARLY_THIRTY_SECOND_REQUEST") if cohort.block != "staged-sequence-comparison" else ("LOW", "HIGH")
        for root in revealed:
            el = tuple(u for u in root.uses if u.policy == treatment)
            other = tuple(u for u in root.uses if u.policy == comparator)
            if len(el) != (4 if cohort.block != "staged-sequence-comparison" else 2) or tuple(
                u.request_id for u in el
            ) != tuple(u.request_id for u in other):
                raise ValueError("comparison drops an assigned task or changes paired order")
            complete &= all(u.evaluable for u in (*el, *other))
            gain.append(
                sum(
                    int(a.common_service) - int(b.common_service)
                    for a, b in zip(el, other, strict=True)
                )
                / len(el)
            )
            shared = tuple(
                (a, b)
                for a, b in zip(el, other, strict=True)
                if a.request_id in cost_panel and a.common_service and b.common_service
            )
            if any(a.attempted_mass_kg is None or b.attempted_mass_kg is None for a, b in shared):
                raise ValueError("common served task lacks its actual delivered feed mass")
            common += len(shared)
            numerators.append(
                sum(
                    (a.attempted_mass_kg for a, _ in shared if a.attempted_mass_kg is not None),
                    D(0),
                )
            )
            denominators.append(
                sum(
                    (b.attempted_mass_kg for _, b in shared if b.attempted_mass_kg is not None),
                    D(0),
                )
            )
        a = next(p for p in cohort.policies if p.policy == treatment)
        b = next(p for p in cohort.policies if p.policy == comparator)
        prerequisite = a.status is ScientificStatus.SUPPORTED and (
            b.status is ScientificStatus.SUPPORTED
            if phase != "I"
            else b.prerequisite_entered
            and all(r[1] and r[3] is False for r in b.roots)
            and b.false_upper <= D(".10")
            and b.unsafe_roots == 0
        )
        gains = np.asarray(gain)
        point = D(repr(float(gains.mean())))
        service_band = nearest_rank_interval(gains[indices].mean(axis=1), tail)
        service_pass = prerequisite and complete and point >= D(".05") and service_band[0] > 0
        unavailable = not complete or not a.prerequisite_entered or not b.prerequisite_entered
        service_status = (
            ContrastDisposition.UNEVALUABLE
            if unavailable
            else ContrastDisposition.NOT_SUPPORTED
            if not prerequisite
            else ContrastDisposition.SUPPORTED
            if service_pass
            else ContrastDisposition.OPPOSED
            if service_band[1] < 0
            else ContrastDisposition.UNRESOLVED
        )
        reasons = []
        if not prerequisite:
            reasons.append("CONTROLLER_USE_OR_RISK_PREREQUISITE_FAILED")
        if not complete:
            reasons.append("MANDATORY_ASSIGNED_OUTCOME_UNKNOWN")
        ratio, cost_band, cost_status = None, None, None
        if cost_endpoint:
            numerator = np.asarray(numerators, dtype=float)
            denominator = np.asarray(denominators, dtype=float)
            if denominator.sum() > 0:
                ratio = D(repr(float(numerator.sum() / denominator.sum())))
            sampled_denominator = denominator[indices].sum(axis=1)
            if (sampled_denominator > 0).all():
                cost_band = nearest_rank_interval(
                    numerator[indices].sum(axis=1) / sampled_denominator, tail
                )
            else:
                reasons.append("UNDEFINED_COST_RESAMPLE_RETAINED")
            passed = (
                prerequisite
                and complete
                and common >= 116
                and service_band[0] >= D("-.05")
                and ratio is not None
                and ratio <= D(".95")
                and cost_band is not None
                and cost_band[1] < 1
            )
            cost_status = (
                ContrastDisposition.UNEVALUABLE
                if unavailable or cost_band is None
                else ContrastDisposition.NOT_SUPPORTED
                if not prerequisite or common < 116 or service_band[0] < D("-.05")
                else ContrastDisposition.SUPPORTED
                if passed
                else ContrastDisposition.OPPOSED
                if cost_band[0] > 1
                else ContrastDisposition.UNRESOLVED
            )
            if common < 116:
                reasons.append("COMMON_SERVICE_COVERAGE_BELOW_116_OF_128")
            if service_band[0] < D("-.05"):
                reasons.append("ALL_TASK_SERVICE_NONINFERIORITY_FAILED")
        results.append(
            ClassicalContrast(
                phase,
                treatment,
                comparator,
                point,
                service_band,
                service_status,
                common,
                ratio,
                cost_band,
                cost_status,
                seed,
                20000,
                D(str(tail)),
                tuple(reasons),
            )
        )
    return ClassicalComparison(
        cohort.block,
        ObjectIdentity.from_record(cohort.record_id, cohort),
        identities,
        tuple(results),
    )
