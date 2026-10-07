"""Independent arithmetic check of saved D operands after separate reveal.

This module imports the typed operand shapes, but not the producer's coverage,
root reduction or cohort reduction functions. Custody and causal sealing must
be checked separately against the persisted production receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from math import isfinite
from typing import cast

from scipy.stats import beta

from .control_math import Admission, MeasuredWord, RootControllerUseEvaluation, WordForecast
from .calibration_pipeline import RegimeCalibrationPackage
from .nomination_records import RegimeNominationPackage
from .prospective_decision import CausalValidityRegimeDecision, ProspectiveWordForecast
from .records import RegimePredictionSeal
from collections.abc import Mapping
from typing import TypeVar
from .config import ROOTS
from .control_math import REQUESTS_K
from .control_prospective_cohort import ReactorRegimeResponseCohortProspective
from .control_prospective_plan import ReactorRegimeResponsePreparedProspectivePlanBundle
from .control_prospective_reveal import RegimeDRootRevealResult
from .prospective_decision import CausalValidityRegimeAssignment
from .records import RegimeCausalPreparation, RegimePrivatePreparation
from .preparation_evidence import preparation_evidence
from empirical_lawhood.adapters.simulators.reactor_regime_response.prospective_records import RegimeDNativeRoot
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.controller_runtime import TickDisposition
from empirical_lawhood.runtime.plans import CandidateExecutionPlan, ExecutionPlan

RecordT = TypeVar("RecordT", bound=CanonicalRecord)



_CONTEXTS = ("early", "middle", "late", "prepared_t0", "c_q", "c_q600", "p_q", "p_q600")
_REQUESTS = (0.0004, 0.0008, 0.0012, 0.0016)


def _saved_chart(
    seal: RegimePredictionSeal,
    context: int,
    coefficient_id: str,
    absolute_id: str,
    q_temperature: float,
    q_cooling: float,
) -> tuple[ProspectiveWordForecast, ...] | None:
    """Reconstruct the three native words from the prediction seal without producer scoring."""
    arrays = seal.predictions.unpack()
    positions = {key: index for index, key in enumerate(seal.candidate_ids)}
    coefficient = positions[f"coefficient.{coefficient_id}".lower()]
    absolute = positions[f"absolute.{absolute_id}".lower()]
    if not (arrays["present"][coefficient, context] and arrays["present"][absolute, context]):
        return None
    p0 = float(arrays["forecast"][absolute, context])
    kappa = float(arrays["forecast"][coefficient, context])
    masses = tuple(float(value) for value in arrays["masses"][context])
    if not masses[0] == 0 < masses[1] < masses[2]:
        return None
    supported = bool(
        arrays["support"][coefficient, context]
        and arrays["support"][absolute, context]
    )
    words = []
    for index, mass in enumerate(masses):
        cooling = kappa * mass
        words.append(ProspectiveWordForecast(
            index, D(repr((0.0, 0.016, 0.032)[index])), D(repr(mass)),
            D(repr(p0 - cooling)), D(repr(cooling)),
            D(repr(q_temperature * .25 + .01)),
            D(repr(q_cooling * (.00005 + .5 * abs(cooling)) + .000001)),
            supported and abs(cooling) <= .01,
            bool(arrays["projection_valid"][context, index]),
        ))
    return tuple(words)


def _choice(
    words: tuple[ProspectiveWordForecast, ...], request: float, *, interval: bool
) -> tuple[int | None, tuple[str, ...]]:
    eligible: list[ProspectiveWordForecast] = []
    blockers: set[str] = set()
    for word in words[1:]:
        if not word.supported:
            blockers.add("OUTSIDE_SUPPORT")
        elif not word.projection_valid:
            blockers.add("INVALID_NATIVE_PROJECTION")
        elif float(word.predicted_cooling_K) - (
            float(word.cooling_halfwidth_K) if interval else 0
        ) < request:
            blockers.add("COOLING_LOWER_BELOW_REQUEST")
        elif float(word.predicted_peak_K) + (
            float(word.temperature_halfwidth_K) if interval else 0
        ) > 356.2:
            blockers.add("TEMPERATURE_UPPER_ABOVE_LIMIT")
        else:
            eligible.append(word)
    if not eligible:
        return None, tuple(sorted(blockers))
    selected = min(
        eligible,
        key=lambda word: (
            word.delivered_mass_kg, word.requested_feed_kg_s, word.word
        ),
    )
    return selected.word, ()


def check_saved_decision(
    *,
    seal: RegimePredictionSeal,
    nomination: RegimeNominationPackage,
    calibration: RegimeCalibrationPackage,
    decision: CausalValidityRegimeDecision,
) -> None:
    """Check the sealed primary intervals and all six choices before reading any D outcome."""
    if (
        seal.root != decision.root
        or decision.route != nomination.selected_route
        or decision.route not in _CONTEXTS
        or calibration.q_temperature is None
        or calibration.q_cooling is None
    ):
        raise ValueError("saved D decision lacks its exact C route and calibration")
    index = _CONTEXTS.index(decision.route)
    arrays = seal.predictions.unpack()
    if (
        int(arrays["context_callback"][index]) != decision.callback
        or seal.context_sha256[index] is None
    ):
        raise ValueError("saved D decision changed its causal callback")
    route = next(row for row in nomination.routes if row.route == decision.route)
    if route.coefficient_model_id is None or route.absolute_model_id is None:
        raise ValueError("saved D decision lacks the nominated primary fit")
    primary = _saved_chart(
        seal, index, route.coefficient_model_id, route.absolute_model_id,
        float(calibration.q_temperature), float(calibration.q_cooling),
    )
    direct = None if route.direct_model_id is None else _saved_chart(
        seal, index, route.direct_model_id, route.absolute_model_id, 0, 0,
    )
    if primary is None or tuple(decision.primary_words) != primary:
        raise ValueError("saved D primary intervals differ from the frozen prediction seal")
    if tuple(decision.direct_words) != (() if direct is None else direct):
        raise ValueError("saved D direct comparator differs from its frozen fit")
    if tuple(policy.policy for policy in decision.policies) != (
        "EL", "POINT", "DIRECT", "MECH", "FIXED_FEED_SIXTEEN_GRAMS_PER_SECOND", "FIXED_FEED_THIRTY_TWO_GRAMS_PER_SECOND"
    ):
        raise ValueError("saved D decision changed its six-policy roster")
    for request_index, request in enumerate(_REQUESTS):
        expected, reasons = _choice(primary, request, interval=True)
        if not decision.causal_preparation_valid:
            expected, reasons = None, ("PREPARATION_UNSAFE",)
        if (
            decision.policies[0].choices[request_index] != expected
            or decision.refusal_reasons[request_index] != reasons
            or decision.policies[1].choices[request_index]
            != _choice(primary, request, interval=False)[0]
            or decision.policies[2].choices[request_index]
            != (None if direct is None else _choice(direct, request, interval=False)[0])
            or decision.policies[3].choices[request_index]
            != (None if not decision.mechanistic_words else _choice(decision.mechanistic_words, request, interval=False)[0])
            or decision.policies[4].choices[request_index]
            != (1 if primary[1].projection_valid else None)
            or decision.policies[5].choices[request_index]
            != (2 if primary[2].projection_valid else None)
        ):
            raise ValueError("saved D decision differs from its frozen six-policy chart")


@dataclass(frozen=True)
class IndependentControllerUseRootCheck:
    root: str
    A: bool
    J: bool
    C: bool
    F: bool
    request_J: tuple[bool, bool, bool, bool]
    chart_complete: bool
    reasons: tuple[str, ...]


def _within_primary_interval(forecast: WordForecast, observed: MeasuredWord) -> bool:
    return bool(
        forecast.word == observed.word
        and observed.receipt_valid
        and observed.observation_valid
        and observed.dose_valid
        and observed.window_valid
        and abs(observed.measured_peak_K - forecast.predicted_peak_K)
        <= forecast.temperature_halfwidth_K
        and abs(observed.measured_cooling_K - forecast.predicted_cooling_K)
        <= forecast.cooling_halfwidth_K
    )


def check_saved_root(
    *,
    root: str,
    primary_chart: tuple[WordForecast, ...],
    evaluator_chart: tuple[MeasuredWord, ...],
    sealed_decisions: tuple[Admission, ...],
    selected_preparation_safe: bool,
    numerical_valid: bool,
    owner_delivered_words: tuple[int | None, ...],
    owner_outcomes: tuple[tuple[MeasuredWord, MeasuredWord] | None, ...],
    produced: RootControllerUseEvaluation | None,
) -> IndependentControllerUseRootCheck:
    """Recalculate A/J/C/F using only the saved primary intervals and native views."""
    if (
        tuple(word.word for word in primary_chart) != (0, 1, 2)
        or tuple(decision.request_K for decision in sealed_decisions)
        != (0.0004, 0.0008, 0.0012, 0.0016)
        or len(owner_delivered_words) != 4
        or len(owner_outcomes) != 4
    ):
        raise ValueError("independent D check lost its exact three-word/four-request roster")
    by_key = {(word.word, word.view): word for word in evaluator_chart}
    complete = len(evaluator_chart) == 6 and set(by_key) == {
        (word, view) for word in (0, 1, 2) for view in (0, 1)
    }
    reasons: set[str] = set()
    if not complete:
        reasons.add("MISSING_COMPLETE_PAIRED_CHART")
    if not numerical_valid:
        reasons.add("NUMERICAL_INVALID")
    if not selected_preparation_safe:
        reasons.add("PREPARATION_UNSAFE")
    all_chart_covered = complete and all(
        _within_primary_interval(forecast, by_key[(forecast.word, view)])
        for forecast in primary_chart for view in (0, 1)
    )
    selected_covered = True
    request_success = []
    false_admission = False
    for decision, delivered, outcomes in zip(
        sealed_decisions, owner_delivered_words, owner_outcomes, strict=True
    ):
        choice = decision.choice
        if choice is None:
            request_success.append(False)
            continue
        if choice not in (1, 2):
            raise ValueError("positive D request borrowed the zero-feed reference")
        forecast = primary_chart[choice]
        paired = outcomes if outcomes is not None else ()
        both = outcomes is not None and tuple(item.view for item in outcomes) == (0, 1)
        covered = both and all(
            _within_primary_interval(forecast, item) for item in paired
        )
        selected_covered = selected_covered and bool(
            delivered == choice and forecast.supported and forecast.projection_valid and covered
        )
        actual = bool(
            both
            and delivered == choice
            and selected_preparation_safe
            and numerical_valid
            and forecast.supported
            and forecast.projection_valid
            and all(
                item.word == choice
                and item.receipt_valid
                and item.observation_valid
                and item.dose_valid
                and item.window_valid
                and isfinite(item.measured_cooling_K)
                and item.measured_cooling_K >= decision.request_K
                and max(item.grid_temperature_K) <= 356.2
                for item in paired
            )
        )
        request_success.append(actual)
        if not actual or not covered:
            false_admission = True
            reasons.add("ADMITTED_DELIVERY_UNSUPPORTED_OR_UNVALIDATED")
    A = bool(
        all_chart_covered
        and selected_covered
        and selected_preparation_safe
        and numerical_valid
        and all(word.supported and word.projection_valid for word in primary_chart)
    )
    request_J = cast(tuple[bool, bool, bool, bool], tuple(request_success))
    J = bool(all(request_J))
    C = A and J
    independent = IndependentControllerUseRootCheck(
        root, A, J, C, false_admission,
        request_J, complete, tuple(sorted(reasons)),
    )
    if produced is not None and (
        produced.adequate != A
        or produced.successes != request_J
        or produced.joined != C
        or produced.false_admission != false_admission
        or produced.evaluability_reasons != independent.reasons
    ):
        raise ValueError("saved D operands disagree with the independent A/J/C/F reconstruction")
    return independent


def check_saved_cohort(
    rows: tuple[IndependentControllerUseRootCheck, ...], produced: dict[str, object]
) -> dict[str, object]:
    if len(rows) != 64 or len({row.root for row in rows}) != 64:
        raise ValueError("independent D check requires all 64 assigned roots once")
    A = sum(row.A for row in rows)
    J = sum(row.J for row in rows)
    C = sum(row.C for row in rows)
    F = sum(row.F for row in rows)
    lower = 0.0 if C == 0 else float(beta.ppf(.05, C, 65 - C))
    upper = 1.0 if F == 64 else float(beta.ppf(.95, F + 1, 64 - F))
    result: dict[str, object] = {
        "assignments": 64,
        "A_roots": A,
        "J_roots": J,
        "joined_successes": C,
        "false_admission_roots": F,
        "joined_lower_95": lower,
        "false_admission_upper_95": upper,
        "adequate_use": lower >= .70 and upper <= .10,
    }
    for key in (
        "assignments", "joined_successes", "false_admission_roots",
        "joined_lower_95", "false_admission_upper_95", "adequate_use",
    ):
        if produced.get(key) != result[key]:
            raise ValueError(f"producer D cohort disagrees with independent {key}")
    return result


def verify_saved_service(
    *,
    plan: CandidateExecutionPlan | ExecutionPlan,
    nomination: RegimeNominationPackage,
    calibration: RegimeCalibrationPackage,
    records: Mapping[tuple[str, str], CanonicalRecord],
) -> dict[str, object]:
    """Recheck all original D physical and installed-owner intersections.

    Before entry authenticate supplied assignment/seal/action/reveal/preparation
    and plan/cohort operands against their compiled output, canonical bytes,
    atomic publication and bounded successful receipt census. Authenticate the
    frozen C nomination/calibration and separate reveal/analysis authority.
    This exposed-data verifier preserves every nonentry, paired native view,
    exact owner delivery, installed probe gate and all-assigned C/F bound. It
    performs no native action, issue, reveal or scientific adjudication.
    """
    tasks = {task.task_id: task for task in plan.tasks}
    expected_roots = tuple(name for name, role, _, _ in ROOTS if role == "prospective")
    if len(expected_roots) != 64 or len(tasks) != 387:
        raise ValueError("D verification changed its full assigned task denominator")
    checked: set[str] = set()
    run_ids = {
        output.relative_path.split("/")[1]
        for task in plan.tasks
        for output in task.outputs
        if output.relative_path.startswith("runs/")
    }
    if len(run_ids) != 1:
        raise ValueError("D verification mixes run namespaces")

    def read(task_id: str, kind: type[RecordT]) -> RecordT:
        task = tasks[task_id]
        outputs = [
            output for output in task.outputs if output.payload_schema == kind.SCHEMA
        ]
        if len(outputs) != 1:
            raise ValueError(f"{task_id} lacks one declared {kind.SCHEMA} output")
        record = records[(task_id, kind.SCHEMA)]
        if not isinstance(record, kind):
            raise ValueError(f"{task_id} changed canonical saved operand type")
        if (
            len(record.canonical_bytes())
            > task.capability.requested_resources.output_bytes
        ):
            raise ValueError(f"{task_id} exceeds its compiled output budget")
        checked.add(task_id)
        return record

    plan_bundle = read("regime.d-plan", ReactorRegimeResponsePreparedProspectivePlanBundle)
    cohort = read("regime.d-cohort", ReactorRegimeResponseCohortProspective)
    if cohort.plan != ObjectIdentity.from_record("regime.d-plan", plan_bundle):
        raise ValueError("D cohort substituted its saved controller-use plan")
    independent: list[IndependentControllerUseRootCheck] = []
    eligible = 0
    for name in expected_roots:
        assignment = read(f"regime.assignment.{name}", CausalValidityRegimeAssignment)
        seal = read(f"regime.seal.{name}", RegimePredictionSeal)
        native = read(f"regime.action.{name}", RegimeDNativeRoot)
        revealed = read(f"regime.d-reveal.{name}", RegimeDRootRevealResult)
        if (
            assignment.prediction_seal
            != ObjectIdentity.from_record(f"{name}.prediction-seal", seal)
            or revealed.assignment
            != ObjectIdentity.from_record(assignment.assignment_id, assignment)
            or revealed.native
            != ObjectIdentity.from_record(f"{name}.native-action", native)
            or (name != assignment.root)
            or (name != revealed.root)
        ):
            raise ValueError(f"{name} changed its sealed causal/native identity")
        decision = assignment.decision
        if decision is None:
            if (
                native.native_calls
                or native.locks
                or revealed.units
                or revealed.revealed
                or revealed.A
                or revealed.J
                or revealed.C
                or revealed.F
            ):
                raise ValueError(f'{name} converted a nonentry into controller use service')
            independent.append(
                IndependentControllerUseRootCheck(
                    name,
                    False,
                    False,
                    False,
                    False,
                    (False,) * 4,
                    False,
                    assignment.nonentry_reasons,
                )
            )
            continue
        eligible += 1
        if (
            decision.nomination
            != ObjectIdentity.from_record(nomination.package_id, nomination)
            or decision.calibration
            != ObjectIdentity.from_record(calibration.package_id, calibration)
            or len(revealed.units) != 4
            or (len(native.locks) != 4)
            or (plan_bundle.plan is None)
            or (cohort.generic_owner is None)
        ):
            raise ValueError(f'{name} lacks its qualified C or installed controller use binding')
        check_saved_decision(
            seal=seal, nomination=nomination, calibration=calibration, decision=decision
        )
        choices = decision.policies[0].choices
        admissions = tuple(
            (
                Admission(
                    REQUESTS_K[index],
                    choice,
                    () if choice is None else (choice,),
                    decision.refusal_reasons[index],
                )
                for index, choice in enumerate(choices)
            )
        )
        measured = revealed.measured
        physical = check_saved_root(
            root=name,
            primary_chart=tuple((word.to_word() for word in decision.primary_words)),
            evaluator_chart=tuple(
                (row.to_measured() for row in measured.chart if row is not None)
            ),
            sealed_decisions=admissions,
            selected_preparation_safe=preparation_evidence(
                read(f"regime.prepare.{name}", RegimeCausalPreparation),
                read(f"regime.prepare.{name}", RegimePrivatePreparation),
                decision.route,
            ).safe,
            numerical_valid=not any(
                (reason == "NUMERICAL_INVALID" for reason in measured.reasons)
            ),
            owner_delivered_words=tuple(
                (
                    choice
                    if tick is not None
                    and tick.disposition is TickDisposition.ACTION_DELIVERED
                    and tick.delivery_trace.exact
                    else None
                    for choice, tick in zip(choices, native.ticks, strict=True)
                )
            ),
            owner_outcomes=tuple(
                (
                    None
                    if pair is None
                    else (pair[0].to_measured(), pair[1].to_measured())
                    for pair in measured.owner
                )
            ),
            produced=None,
        )
        installed_A = bool(
            physical.A
            and all(
                (
                    not physical.request_J[index] or unit.task_success
                    for index, unit in enumerate(revealed.units)
                )
            )
            and all(
                (
                    unit.probe_adequacy is GateStatus.PASS
                    and unit.probe_coverage is GateStatus.PASS
                    for unit in revealed.units
                )
            )
        )
        installed_F = physical.F or any(
            (unit.admitted and (not unit.task_success) for unit in revealed.units)
        )
        if (revealed.physical_A, revealed.physical_request_J, revealed.physical_F) != (
            physical.A,
            physical.request_J,
            physical.F,
        ) or (revealed.A, revealed.request_J, revealed.J, revealed.C, revealed.F) != (
            installed_A,
            physical.request_J,
            physical.J,
            installed_A and physical.J,
            installed_F,
        ):
            raise ValueError(
                f'{name} differs from independent physical/installed controller use intersection'
            )
        independent.append(
            IndependentControllerUseRootCheck(
                name,
                installed_A,
                physical.J,
                installed_A and physical.J,
                installed_F,
                physical.request_J,
                physical.chart_complete,
                physical.reasons,
            )
        )
    if (plan_bundle.plan is None) != (eligible == 0):
        raise ValueError("D controller-use plan changed eligible-root entry")
    checked_cohort = check_saved_cohort(
        tuple(independent),
        {
            "assignments": cohort.assigned_count,
            "joined_successes": cohort.C_count,
            "false_admission_roots": cohort.F_count,
            "joined_lower_95": float(cohort.joined_lower_95),
            "false_admission_upper_95": float(cohort.false_admission_upper_95),
            "adequate_use": cohort.adequate_use,
        },
    )
    if (cohort.A_count, cohort.J_count) != (
        checked_cohort["A_roots"],
        checked_cohort["J_roots"],
    ):
        raise ValueError("D cohort A/J counts differ from saved independent roots")
    return {
        "scope": "RECEIPT_BACKED_SAVED_OPERAND_CHECK_NOT_SCIENTIFIC_ADJUDICATION",
        "execution_plan_sha256": plan.fingerprint(),
        "receipt_backed_tasks_checked": len(checked),
        "assigned_roots": len(independent),
        "eligible_roots": eligible,
        **checked_cohort,
        "shared_helpers": (
            "canonical decoding and typed word conversions",
            "saved publication marker and receipt field formats",
            "installed unit fields",
        ),
        "independent_operations": (
            "sealed primary interval and six-policy decision reconstruction",
            "paired-view physical A/J/F and installed-owner intersection",
            "all-assigned C/F binomial bounds",
        ),
    }
