"""Truth-known conformance cases for the selective dependence response method."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_nonempty,
    validate_stable_id,
)

from .contracts import SelectiveDependenceResponseCaseState, SelectiveDependenceResponseDisposition, SelectiveDependenceResponseExchangeExpectation, SelectiveDependenceResponseMethodQuestionFreeze
from .inference import SelectiveDependenceResponseExchangeState, assess_exchange, classify_response, noncompensating_disposition


SELECTIVE_DEPENDENCE_RESPONSE_CONFORMANCE_CASE_IDS = (
    "case.action-available",
    "case.active-opposed",
    "case.active-supported",
    "case.high-response",
    "case.hold-only",
    "case.invariant-opposed",
    "case.invariant-supported",
    "case.low-response",
    "case.neutral-response",
    "case.nonattempt",
    "case.target-positive-sink-failed",
    "case.underpowered-active",
    "case.unmeasured-hold",
    "case.zero-gate-admitted",
)


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseConformanceCaseResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-conformance-case-result'

    case_id: str
    expected: str
    observed: str
    passed: bool
    oracle_origin: str

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        validate_nonempty(self.expected, field_name="expected")
        validate_nonempty(self.observed, field_name="observed")
        validate_stable_id(self.oracle_origin, field_name="oracle_origin")
        if self.passed != (self.expected == self.observed):
            raise ValueError("conformance pass flag differs from exact comparison")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseMethodConformanceReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-method-conformance-report'

    report_id: str
    method_question: ObjectIdentity
    case_results: tuple[SelectiveDependenceResponseConformanceCaseResult, ...]
    all_passed: bool
    target_response_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        require_sorted_unique_ids(self.case_results, attribute="case_id", field_name="case_results")
        if tuple(value.case_id for value in self.case_results) != (SELECTIVE_DEPENDENCE_RESPONSE_CONFORMANCE_CASE_IDS):
            raise ValueError("selective dependence response conformance requires the exact fourteen cases")
        if self.all_passed != all(value.passed for value in self.case_results):
            raise ValueError("conformance aggregate differs")
        if self.target_response_count:
            raise ValueError("truth-world conformance cannot contact targets")
        if self.outcome_access is not OutcomeAccess.PRIVILEGED_TRUTH:
            raise ValueError("truth-world report must retain privileged-truth ceiling")


def _case(case_id: str, expected: object, observed: object) -> SelectiveDependenceResponseConformanceCaseResult:
    expected_text = expected.value if hasattr(expected, "value") else str(expected)
    observed_text = observed.value if hasattr(observed, "value") else str(observed)
    return SelectiveDependenceResponseConformanceCaseResult(
        case_id=case_id,
        expected=expected_text,
        observed=observed_text,
        passed=expected_text == observed_text,
        oracle_origin="oracle.analytic-truth-world",
    )


def execute_truth_known_conformance(
    question: SelectiveDependenceResponseMethodQuestionFreeze,
) -> SelectiveDependenceResponseMethodConformanceReport:
    """Exercise the decisive method distinctions without target access."""

    active = assess_exchange(
        assessment_id="assessment.truth.active",
        exchange_id="exchange.truth.active",
        expectation=SelectiveDependenceResponseExchangeExpectation.ACTIVE,
        point=Decimal("2"),
        lower=Decimal("1.5"),
        upper=Decimal("2.5"),
        equivalence_margin=Decimal("0.1"),
        minimum_active_difference=Decimal("1"),
        complete_unit_count=16,
    )
    active_null = assess_exchange(
        assessment_id="assessment.truth.active-null",
        exchange_id="exchange.truth.active-null",
        expectation=SelectiveDependenceResponseExchangeExpectation.ACTIVE,
        point=Decimal("0"),
        lower=Decimal("-0.05"),
        upper=Decimal("0.05"),
        equivalence_margin=Decimal("0.1"),
        minimum_active_difference=Decimal("1"),
        complete_unit_count=16,
    )
    invariant = assess_exchange(
        assessment_id="assessment.truth.invariant",
        exchange_id="exchange.truth.invariant",
        expectation=SelectiveDependenceResponseExchangeExpectation.INVARIANT,
        point=Decimal("0"),
        lower=Decimal("-0.05"),
        upper=Decimal("0.05"),
        equivalence_margin=Decimal("0.1"),
        minimum_active_difference=Decimal("1"),
        complete_unit_count=16,
    )
    invariant_broken = assess_exchange(
        assessment_id="assessment.truth.invariant-broken",
        exchange_id="exchange.truth.invariant-broken",
        expectation=SelectiveDependenceResponseExchangeExpectation.INVARIANT,
        point=Decimal("0.5"),
        lower=Decimal("0.3"),
        upper=Decimal("0.7"),
        equivalence_margin=Decimal("0.1"),
        minimum_active_difference=Decimal("1"),
        complete_unit_count=16,
    )
    cases = (
        _case(
            "case.action-available",
            SelectiveDependenceResponseDisposition.ACTION_AVAILABLE,
            noncompensating_disposition(
                active_action_gate_margins={"action.plus": (Decimal("1"), Decimal("0"))},
                hold_gate_margins=(Decimal("1"),),
            ),
        ),
        _case("case.active-opposed", SelectiveDependenceResponseExchangeState.OPPOSED, active_null.state),
        _case("case.active-supported", SelectiveDependenceResponseExchangeState.SUPPORTED, active.state),
        _case(
            "case.high-response",
            SelectiveDependenceResponseCaseState.HIGH,
            classify_response(Decimal("2"), neutral_margin=Decimal("0.1")),
        ),
        _case(
            "case.hold-only",
            SelectiveDependenceResponseDisposition.HOLD_ONLY,
            noncompensating_disposition(
                active_action_gate_margins={"action.plus": (Decimal("1"), Decimal("-0.1"))},
                hold_gate_margins=(Decimal("0"), Decimal("1")),
            ),
        ),
        _case("case.invariant-opposed", SelectiveDependenceResponseExchangeState.OPPOSED, invariant_broken.state),
        _case("case.invariant-supported", SelectiveDependenceResponseExchangeState.SUPPORTED, invariant.state),
        _case(
            "case.low-response",
            SelectiveDependenceResponseCaseState.LOW,
            classify_response(Decimal("-2"), neutral_margin=Decimal("0.1")),
        ),
        _case(
            "case.neutral-response",
            SelectiveDependenceResponseCaseState.NEUTRAL,
            classify_response(Decimal("0.05"), neutral_margin=Decimal("0.1")),
        ),
        _case(
            "case.nonattempt",
            SelectiveDependenceResponseDisposition.NONATTEMPT,
            noncompensating_disposition(
                active_action_gate_margins={"action.plus": (Decimal("1"), Decimal("-0.1"))},
                hold_gate_margins=(Decimal("1"), Decimal("-0.1")),
            ),
        ),
        _case(
            "case.target-positive-sink-failed",
            SelectiveDependenceResponseDisposition.NONATTEMPT,
            noncompensating_disposition(
                active_action_gate_margins={"action.plus": (Decimal("10"), Decimal("-1"))},
                hold_gate_margins=(Decimal("-1"), Decimal("1")),
            ),
        ),
        _case(
            "case.unmeasured-hold",
            SelectiveDependenceResponseDisposition.UNEVALUABLE,
            noncompensating_disposition(active_action_gate_margins={}, hold_gate_margins=None),
        ),
        _case(
            "case.underpowered-active",
            SelectiveDependenceResponseExchangeState.UNEVALUABLE,
            assess_exchange(
                assessment_id="assessment.truth.underpowered",
                exchange_id="exchange.truth.underpowered",
                expectation=SelectiveDependenceResponseExchangeExpectation.ACTIVE,
                point=Decimal("2"),
                lower=Decimal("1"),
                upper=Decimal("3"),
                equivalence_margin=Decimal("0.1"),
                minimum_active_difference=Decimal("1"),
                complete_unit_count=1,
            ).state,
        ),
        _case(
            "case.zero-gate-admitted",
            SelectiveDependenceResponseDisposition.ACTION_AVAILABLE,
            noncompensating_disposition(
                active_action_gate_margins={"action.plus": (Decimal("0"),)},
                hold_gate_margins=(Decimal("-1"),),
            ),
        ),
    )
    ordered = tuple(sorted(cases, key=lambda value: value.case_id))
    return SelectiveDependenceResponseMethodConformanceReport(
        report_id="selective-dependence-response.method-conformance-report",
        method_question=ObjectIdentity.from_record(question.freeze_id, question),
        case_results=ordered,
        all_passed=all(value.passed for value in ordered),
        target_response_count=0,
        outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
    )


__all__ = [
    "SELECTIVE_DEPENDENCE_RESPONSE_CONFORMANCE_CASE_IDS",
    'SelectiveDependenceResponseConformanceCaseResult',
    'SelectiveDependenceResponseMethodConformanceReport',
    "execute_truth_known_conformance",
]
