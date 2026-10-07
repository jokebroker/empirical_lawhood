"""Shared request owners plus the declared twelve independent 64-root joins."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar
from collections.abc import Iterator
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.runtime.controller_evaluation_nested import NestedControllerUseEvaluator, PreparedControllerCohortEvaluation, PreparedPolicyUnitEvaluation
from empirical_lawhood.planning.controller_study import ImplementationBinding
from empirical_lawhood.runtime.canonical_record_archive import CanonicalRecordArchive
from .config import CONTEXTS, HORIZONS, FrontierDesign
from .prospective_plan import ReactorFiniteControlFrontierProspectivePlan, PROSPECTIVE_ROOTS
from .prospective_reveal import FrontierRevealedRoot
from .qualification import cp
from .selection import FrontierUseRequest


@dataclass(frozen=True)
class FrontierCohortUse:
    """Already evaluated unit operands; archives remain in authenticated root records."""

    request: FrontierUseRequest
    evaluable: bool
    success: bool
    false_admission: bool
    unsafe: bool
    unit: PreparedPolicyUnitEvaluation | None


@dataclass(frozen=True)
class FrontierCohortRoot:
    root: str
    identity: ObjectIdentity
    uses: tuple[FrontierCohortUse, ...]


@dataclass(frozen=True, slots=True)
class ReactorFiniteControlFrontierProspectivePair(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/reactor-finite-control-frontier-prospective-pair'
    context: str
    horizon_s: int
    active_requests: tuple[str, ...]
    roots: tuple[tuple[str, bool, bool, bool, bool], ...]
    successes: int
    false_admissions: int
    unsafe_roots: int
    success_lower: D
    false_upper: D
    scientific_status: ScientificStatus
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            tuple(r[0] for r in self.roots) != PROSPECTIVE_ROOTS
            or self.successes != sum(r[2] for r in self.roots)
            or self.false_admissions != sum(r[3] for r in self.roots)
            or self.unsafe_roots != sum(r[4] for r in self.roots)
        ):
            raise ValueError("Controller-use pair changed its physical 64-root denominator")


@dataclass(frozen=True, slots=True)
class ReactorFrontierCohort(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-frontier/reactor-frontier-cohort'
    plan: ObjectIdentity
    roots: tuple[ObjectIdentity, ...]
    owners: tuple[tuple[str, PreparedControllerCohortEvaluation], ...]
    pairs: tuple[ReactorFiniteControlFrontierProspectivePair, ...]

    @property
    def record_id(self) -> str:
        return "reactor-finite-control-frontier.prospective-evaluation-cohort"

    def __post_init__(self) -> None:
        _validate_census(self.roots, self.pairs)


def _validate_census(
    roots: tuple[ObjectIdentity, ...], pairs: tuple[ReactorFiniteControlFrontierProspectivePair, ...]
) -> None:
    if len(roots) != 64 or tuple((p.context, p.horizon_s) for p in pairs) != tuple(
        (c, t) for c in CONTEXTS for t in HORIZONS
    ):
        raise ValueError("Controller use changed its fixed twelve-context/window family")


@dataclass(frozen=True, slots=True)
class ArchivedReactorFrontierCohort(CanonicalRecord):
    "Lossless archive transport of finite-control frontier owners; the statistical operands are unchanged."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-frontier/archived-reactor-frontier-cohort'
    plan: ObjectIdentity
    roots: tuple[ObjectIdentity, ...]
    owner_archives: tuple[tuple[str, CanonicalRecordArchive], ...]
    pairs: tuple[ReactorFiniteControlFrontierProspectivePair, ...]

    @property
    def owners(self) -> Iterator[tuple[str, PreparedControllerCohortEvaluation]]:
        for request, archive in self.owner_archives:
            owner = decode_canonical_bytes(
                archive.unpack(),
                PreparedControllerCohortEvaluation,
                maximum_bytes=archive.decoded_bytes,
            )
            if archive.subject != ObjectIdentity.from_record(owner.evaluation_id, owner):
                raise ValueError("Controller use cohort archive substituted its exact owner")
            yield request, owner

    @property
    def record_id(self) -> str:
        return "reactor-finite-control-frontier.prospective-evaluation-cohort"

    def __post_init__(self) -> None:
        _validate_census(self.roots, self.pairs)
        if len({r for r, _ in self.owner_archives}) != len(self.owner_archives) or any(
            a.subject.object_schema != PreparedControllerCohortEvaluation.SCHEMA
            for _, a in self.owner_archives
        ):
            raise ValueError("Controller use cohort archive changed its owner census or schema")

    def as_unarchived(self) -> ReactorFrontierCohort:
        """Explicit representation compatibility, without scientific promotion."""
        return ReactorFrontierCohort(self.plan, self.roots, tuple(self.owners), self.pairs)


def reduce_prospective(
    plan: ReactorFiniteControlFrontierProspectivePlan, roots: tuple[FrontierRevealedRoot, ...]
) -> ArchivedReactorFrontierCohort:
    evaluators = {}
    for root in roots:
        for use in root.uses:
            if use.unit is not None and use.request.request_id not in evaluators:
                revealed = use.revealed
                assert revealed is not None
                evaluators[use.request.request_id] = revealed.sealed.design.evaluator_binding
    return reduce_prospective_operands(
        plan,
        tuple(
            FrontierCohortRoot(
                r.root,
                ObjectIdentity.from_record(r.record_id, r),
                tuple(
                    FrontierCohortUse(
                        u.request, u.evaluable, u.success, u.false_admission, u.unsafe, u.unit
                    )
                    for u in r.uses
                ),
            )
            for r in roots
        ),
        evaluators,
    )


def reduce_prospective_operands(
    plan: ReactorFiniteControlFrontierProspectivePlan,
    roots: tuple[FrontierCohortRoot, ...],
    evaluators: dict[str, ImplementationBinding],
) -> ArchivedReactorFrontierCohort:
    """The same full-cohort join for authenticated production or labelled software operands."""
    if tuple(r.root for r in roots) != PROSPECTIVE_ROOTS:
        raise ValueError("Controller use cannot drop an assigned physical root")
    owners = []
    for archived_request in plan.requests:
        request_plan = archived_request.unpack()
        evaluation = request_plan.plan
        if evaluation is None:
            continue
        rows = tuple(
            u
            for r in roots
            for u in r.uses
            if u.request == request_plan.request and u.unit is not None
        )
        units = tuple(r.unit for r in rows if r.unit is not None)
        if len(units) != len(evaluation.roots):
            raise ValueError("missing installed controller-use owner for an entered root/request")
        owner = NestedControllerUseEvaluator(
            evaluators[request_plan.request.request_id], evaluation.reducer
        ).reduce_prepared_cohort(plan=evaluation, units=units)
        owners.append(
            (
                request_plan.request.request_id,
                CanonicalRecordArchive.pack(owner.evaluation_id, owner),
            )
        )
    pairs = []
    design = FrontierDesign()
    for context in CONTEXTS:
        for horizon in HORIZONS:
            requests = tuple(
                p.request
                for p in plan.requests
                if (p.request.context, p.request.horizon_s) == (context, horizon)
            )
            census = []
            for root in roots:
                uses = tuple(u for u in root.uses if u.request in requests)
                if tuple(u.request for u in uses) != requests:
                    raise ValueError("Controller-use root changed its low/high request join")
                census.append(
                    (
                        root.root,
                        bool(uses) and all(u.evaluable for u in uses),
                        bool(uses) and all(u.success for u in uses),
                        any(u.false_admission for u in uses),
                        any(u.unsafe for u in uses),
                    )
                )
            success, false, unsafe = (
                sum(r[2] for r in census),
                sum(r[3] for r in census),
                sum(r[4] for r in census),
            )
            lower = cp(success, 64, design.prospective_evaluation_alpha, lower=True)
            upper = cp(false, 64, design.prospective_evaluation_alpha, lower=False)
            reasons = []
            if not requests:
                reasons.append("LOCAL_LAW_REQUEST_PREREQUISITE_NONENTRY")
            elif not all(r[1] for r in census):
                reasons.append("MISSING_MANDATORY_EVIDENCE")
            if requests and lower < design.prospective_evaluation_joined_probability:
                reasons.append("JOINED_SUCCESS_BOUND_FAILED")
            if requests and upper > design.prospective_evaluation_false_admission_ceiling:
                reasons.append("FALSE_ADMISSION_BOUND_FAILED")
            if unsafe:
                reasons.append("NATIVE_UNSAFE_VETO")
            status = (
                ScientificStatus.UNEVALUABLE
                if not requests or not all(r[1] for r in census)
                else ScientificStatus.NOT_SUPPORTED
                if reasons
                else ScientificStatus.SUPPORTED
            )
            pairs.append(
                ReactorFiniteControlFrontierProspectivePair(
                    context,
                    horizon,
                    tuple(r.request_id for r in requests),
                    tuple(census),
                    success,
                    false,
                    unsafe,
                    lower,
                    upper,
                    status,
                    tuple(reasons),
                )
            )
    return ArchivedReactorFrontierCohort(
        ObjectIdentity.from_record(plan.record_id, plan),
        tuple(r.identity for r in roots),
        tuple(owners),
        tuple(pairs),
    )
