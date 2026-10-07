"""Pure complete-roster evaluation of a sealed finite action chart."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.admission import AdmissionGateKind, GateStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.finite_chart_reference import FiniteChartReferencePlan
from empirical_lawhood.planning.controller_study import LeastMagnitudeControllerStudy
from empirical_lawhood.planning.finite_chart_reference import FiniteChartReferenceDesign, ReferenceComparisonDimensionSpec, ReferenceComparisonDirection, ReferenceComparisonKind, ReferenceComparisonReducer
from empirical_lawhood.runtime.controller_runtime import CommitmentDisposition


class FiniteChartActionDisposition(StrEnum):
    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class FiniteChartActionAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/finite-chart-action-assessment'

    assessment_id: str
    panel_id: str
    action_word_id: str
    disposition: FiniteChartActionDisposition
    materially_equivalent: bool | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("panel_id", self.panel_id),
            ("action_word_id", self.action_word_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is FiniteChartActionDisposition.SAFE:
            if self.materially_equivalent is None or self.reason_codes:
                raise ValueError("safe finite-chart action assessment is incomplete")
        elif self.materially_equivalent is not None or not self.reason_codes:
            raise ValueError("unsafe/unevaluable action assessment is malformed")


@dataclass(frozen=True, slots=True)
class FiniteChartDispositionReferenceReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/finite-chart-disposition-reference-receipt'

    receipt_id: str
    plan: ObjectIdentity
    assessments: tuple[FiniteChartActionAssessment, ...]
    expected_commitment: CommitmentDisposition | None
    terminal_disposition: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_ids(
            self.assessments,
            attribute="assessment_id",
            field_name="assessments",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.terminal_disposition in {"UNSAFE", "UNEVALUABLE"}:
            if self.expected_commitment is not None or not self.reason_codes:
                raise ValueError("nonevaluable finite-chart receipt is malformed")
        elif self.expected_commitment is None or self.reason_codes:
            raise ValueError("finite-chart receipt lacks one expected commitment")


def evaluate_finite_chart_disposition(
    *,
    plan: FiniteChartReferencePlan,
    assessments: tuple[FiniteChartActionAssessment, ...],
    active_action_word_id: str,
    hold_action_word_id: str | None,
) -> FiniteChartDispositionReferenceReceipt:
    """Apply complete-product accounting and the frozen disposition priority."""

    validate_stable_id(active_action_word_id, field_name="active_action_word_id")
    require_sorted_unique_ids(
        assessments,
        attribute="assessment_id",
        field_name="assessments",
    )
    expected = {
        (panel.panel_id, word.word_id) for panel in plan.panel for word in plan.action_words
    }
    observed = {(value.panel_id, value.action_word_id) for value in assessments}
    if observed != expected or len(assessments) != len(expected):
        raise ValueError("finite-chart assessment product is incomplete or duplicated")
    active = tuple(value for value in assessments if value.action_word_id == active_action_word_id)
    if not active:
        raise ValueError("finite-chart active action is outside the frozen chart")
    reasons: set[str] = set()
    if any(value.disposition is FiniteChartActionDisposition.UNSAFE for value in active):
        terminal = "UNSAFE"
        expected_commitment = None
        reasons.add("ACTIVE_ACTION_UNSAFE_IN_FINITE_CHART")
    elif any(value.disposition is FiniteChartActionDisposition.UNEVALUABLE for value in active):
        terminal = "UNEVALUABLE"
        expected_commitment = None
        reasons.add("ACTIVE_ACTION_UNEVALUABLE_IN_FINITE_CHART")
    elif all(value.materially_equivalent is False for value in active):
        terminal = "ACTION_COMMITTED"
        expected_commitment = CommitmentDisposition.ACTION_COMMITTED
    elif hold_action_word_id is not None:
        hold = tuple(value for value in assessments if value.action_word_id == hold_action_word_id)
        if hold and all(
            value.disposition is FiniteChartActionDisposition.SAFE
            and value.materially_equivalent is True
            for value in hold
        ):
            terminal = "MEASURED_HOLD_COMMITTED"
            expected_commitment = CommitmentDisposition.MEASURED_HOLD_COMMITTED
        else:
            terminal = "NONATTEMPT"
            expected_commitment = CommitmentDisposition.NONATTEMPT
    else:
        terminal = "NONATTEMPT"
        expected_commitment = CommitmentDisposition.NONATTEMPT
    return FiniteChartDispositionReferenceReceipt(
        receipt_id=f"finite-chart-reference.{plan.reference_plan_id}",
        plan=ObjectIdentity.from_record(plan.reference_plan_id, plan),
        assessments=assessments,
        expected_commitment=expected_commitment,
        terminal_disposition=terminal,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class ReferenceGateAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reference-gate-assessment'

    assessment_id: str
    gate_kind: AdmissionGateKind
    status: GateStatus

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")


@dataclass(frozen=True, slots=True)
class ReferenceCellAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reference-cell-assessment'

    cell_id: str
    independent_unit_id: str
    action_word_id: str
    model_member_id: str
    reference_seed_id: str
    gates: tuple[ReferenceGateAssessment, ...]
    dimension_values: tuple[NamedDecimal, ...]
    delivery_valid: bool
    supported: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("independent_unit_id", self.independent_unit_id),
            ("action_word_id", self.action_word_id),
            ("model_member_id", self.model_member_id),
            ("reference_seed_id", self.reference_seed_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(self.gates, attribute="assessment_id", field_name="gates")
        require_sorted_unique_ids(
            self.dimension_values,
            attribute="value_id",
            field_name="dimension_values",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        observed_gates = tuple(
            sorted((value.gate_kind for value in self.gates), key=lambda x: x.value)
        )
        if observed_gates != tuple(sorted(AdmissionGateKind, key=lambda x: x.value)):
            raise ValueError("reference cell does not contain the exact nine gate roles")
        if len({value.gate_kind for value in self.gates}) != len(self.gates):
            raise ValueError("reference cell duplicates a gate role")
        if (
            self.delivery_valid
            and self.supported
            and all(value.status is GateStatus.PASS for value in self.gates)
        ):
            if self.reason_codes:
                raise ValueError("valid reference cell cannot carry failure reasons")
        elif not self.reason_codes:
            raise ValueError("invalid reference cell requires a typed reason")


class ReferenceActionDisposition(StrEnum):
    ADMITTED = "ADMITTED"
    UNSAFE = "UNSAFE"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ReferenceActionAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reference-action-assessment'

    assessment_id: str
    reference_design: ObjectIdentity
    action_word_id: str
    cell_ids: tuple[str, ...]
    disposition: ReferenceActionDisposition
    reduced_values: tuple[NamedDecimal, ...]
    leading_class_member: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.action_word_id, field_name="action_word_id")
        require_sorted_unique_strings(self.cell_ids, field_name="cell_ids", allow_empty=False)
        require_sorted_unique_ids(
            self.reduced_values,
            attribute="value_id",
            field_name="reduced_values",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.reference_design.object_schema != FiniteChartReferenceDesign.SCHEMA:
            raise ValueError("reference action assessment binds another design schema")
        if self.disposition is ReferenceActionDisposition.ADMITTED:
            if self.reason_codes or not self.reduced_values:
                raise ValueError("admitted reference action is incomplete")
        elif self.leading_class_member or not self.reason_codes:
            raise ValueError("invalid reference action assessment is malformed")


class ReferenceDispositionMemberKind(StrEnum):
    ACTIVE = "ACTIVE"
    MEASURED_HOLD = "MEASURED_HOLD"
    NONATTEMPT = "NONATTEMPT"


@dataclass(frozen=True, slots=True)
class ReferenceDispositionMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reference-disposition-member'

    member_id: str
    kind: ReferenceDispositionMemberKind
    action_word_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        if self.kind is ReferenceDispositionMemberKind.ACTIVE:
            if self.action_word_id is None:
                raise ValueError("active reference member requires an action")
            validate_stable_id(self.action_word_id, field_name="action_word_id")
        elif self.action_word_id is not None:
            raise ValueError("HOLD/NONATTEMPT reference member cannot carry an action")


@dataclass(frozen=True, slots=True)
class ReferenceDispositionClass(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reference-disposition-class'

    class_id: str
    reference_design: ObjectIdentity
    action_assessments: tuple[ReferenceActionAssessment, ...]
    members: tuple[ReferenceDispositionMember, ...]
    canonical_action_word_id: str | None
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.class_id, field_name="class_id")
        require_sorted_unique_ids(
            self.action_assessments,
            attribute="assessment_id",
            field_name="action_assessments",
        )
        require_sorted_unique_ids(self.members, attribute="member_id", field_name="members")
        if self.reference_design.object_schema != FiniteChartReferenceDesign.SCHEMA:
            raise ValueError("reference class binds another design schema")
        if not self.members:
            raise ValueError("reference disposition class cannot be empty")
        active_ids = {
            value.action_word_id
            for value in self.members
            if value.kind is ReferenceDispositionMemberKind.ACTIVE
        }
        if self.canonical_action_word_id is not None:
            validate_stable_id(
                self.canonical_action_word_id,
                field_name="canonical_action_word_id",
            )
            if self.canonical_action_word_id not in active_ids:
                raise ValueError("canonical action is outside the active reference class")
        elif active_ids:
            raise ValueError("active reference class requires a canonical action")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("reference disposition class must remain sealed")


class ReferenceMembershipStatus(StrEnum):
    MEMBER = "MEMBER"
    NONMEMBER = "NONMEMBER"


@dataclass(frozen=True, slots=True)
class ReferenceClassMembershipReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/reference-class-membership-receipt'

    receipt_id: str
    reference_class: ObjectIdentity
    commitment_disposition: CommitmentDisposition
    committed_action_word_id: str | None
    status: ReferenceMembershipStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.reference_class.object_schema != ReferenceDispositionClass.SCHEMA:
            raise ValueError("membership receipt binds another reference-class schema")
        if self.committed_action_word_id is not None:
            validate_stable_id(
                self.committed_action_word_id,
                field_name="committed_action_word_id",
            )
        if self.status is ReferenceMembershipStatus.MEMBER:
            if self.reason_codes:
                raise ValueError("successful class membership cannot carry reasons")
        elif not self.reason_codes:
            raise ValueError("reference nonmembership requires a decisive reason")


def _reduce_dimension(
    *,
    values: tuple[Decimal, ...],
    reducer: ReferenceComparisonReducer,
) -> Decimal:
    if not values:
        raise ValueError("cannot reduce an empty reference dimension")
    return min(values) if reducer is ReferenceComparisonReducer.MINIMUM else max(values)


def _within_material_equivalence(
    *,
    value: Decimal,
    best: Decimal,
    dimension: ReferenceComparisonDimensionSpec,
) -> bool:
    loss = (
        best - value
        if dimension.direction is ReferenceComparisonDirection.MAXIMIZE
        else value - best
    )
    allowance = dimension.absolute_tolerance.value + dimension.relative_tolerance * max(
        abs(best), dimension.native_scale.value
    )
    return loss <= allowance


def evaluate_native_magnitude_reference(
    *,
    study: LeastMagnitudeControllerStudy,
    design: FiniteChartReferenceDesign,
    cells: tuple[ReferenceCellAssessment, ...],
) -> ReferenceDispositionClass:
    """Authenticate native-cost dimensions, then reuse the independent reducer.

    Gate/support/delivery assessments remain independently supplied. Neither
    compiled choices nor response utility can supply reference eligibility.
    """
    dimensions = sorted(design.comparison_dimensions, key=lambda d: d.order)
    if len(dimensions) != 2:
        raise ValueError("native ordering reference requires magnitude and exact word priority")
    dimension, tie = dimensions
    unit = study.native_magnitudes[0].magnitude.unit
    if (
        dimension.kind is not ReferenceComparisonKind.EFFORT
        or dimension.order != 1
        or dimension.direction is not ReferenceComparisonDirection.MINIMIZE
        or dimension.reducer is not ReferenceComparisonReducer.MAXIMUM
        or dimension.native_unit != unit
        or dimension.absolute_tolerance.value != 0
        or dimension.relative_tolerance != 0
        or tie.order != 2
        or tie.kind is not ReferenceComparisonKind.EXACT_CLASS
        or tie.direction is not ReferenceComparisonDirection.MINIMIZE
        or tie.reducer is not ReferenceComparisonReducer.MAXIMUM
        or tie.native_unit != "1"
        or tie.absolute_tolerance.value != 0
        or tie.relative_tolerance != 0
    ):
        raise ValueError("reference comparison changes exact native magnitude ordering")
    words = {a.action_binding_id: a.action_word for a in study.action_bindings}
    if {w.word_id: w for w in design.action_words} != {w.word_id: w for w in words.values()}:
        raise ValueError("native ordering reference changes exact action words")
    chart = study.synthesis.candidate_chart
    if len({c.decision_cell_id for c in chart.candidates}) != 1:
        raise ValueError("native ordering reference requires one concrete decision cell")
    admission_candidate_cells_by_id = {c.candidate_cell_id: c for c in study.admission.candidate_cells}
    ordered_candidates = sorted(chart.candidates, key=lambda c: c.priority_rank)
    priority = tuple(
        words[admission_candidate_cells_by_id[c.admission_candidate_cell.object_id].action_fibre.object_id].word_id
        for c in ordered_candidates
    )
    if design.action_priority_ids != priority:
        raise ValueError("reference priority differs from frozen native word order")
    magnitudes = {m.action_word.object_id: m.magnitude.value for m in study.native_magnitudes}
    ranks = {word: c.priority_rank for word, c in zip(priority, ordered_candidates, strict=True)}
    for cell in cells:
        if cell.action_word_id not in magnitudes or cell.dimension_values != tuple(
            sorted(
                (
                    NamedDecimal(dimension.dimension_id, magnitudes[cell.action_word_id], unit),
                    NamedDecimal(tie.dimension_id, Decimal(ranks[cell.action_word_id]), "1"),
                ),
                key=lambda v: v.value_id,
            )
        ):
            raise ValueError("reference magnitude differs from requested native action")
    result = evaluate_finite_reference_class(design=design, cells=cells)
    admitted = {
        a.action_word_id
        for a in result.action_assessments
        if a.disposition is ReferenceActionDisposition.ADMITTED
    }
    if design.measured_hold_word_id in admitted and len(admitted) > 1:
        raise ValueError("fallback-HOLD reference cannot certify HOLD competing with active words")
    return result


def evaluate_finite_reference_class(
    *,
    design: FiniteChartReferenceDesign,
    cells: tuple[ReferenceCellAssessment, ...],
) -> ReferenceDispositionClass:
    """Reduce one complete panel into a sealed set-valued disposition class."""

    require_sorted_unique_ids(cells, attribute="cell_id", field_name="cells")
    expected = {
        (design.independent_unit_id, action.word_id, member_id, seed_id)
        for action in design.action_words
        for member_id in design.model_member_ids
        for seed_id in design.reference_seed_ids
    }
    observed = {
        (
            cell.independent_unit_id,
            cell.action_word_id,
            cell.model_member_id,
            cell.reference_seed_id,
        )
        for cell in cells
    }
    if observed != expected or len(cells) != len(expected):
        raise ValueError("reference panel is incomplete or duplicated")
    dimension_ids = {value.dimension_id for value in design.comparison_dimensions}
    for cell in cells:
        cell_value_map = {value.value_id: value for value in cell.dimension_values}
        if set(cell_value_map) != dimension_ids:
            raise ValueError("reference cell dimension roster differs from design")
        for dimension in design.comparison_dimensions:
            if cell_value_map[dimension.dimension_id].unit != dimension.native_unit:
                raise ValueError("reference cell dimension uses another native unit")

    design_identity = ObjectIdentity.from_record(design.reference_design_id, design)
    provisional: list[ReferenceActionAssessment] = []
    reduced_by_action: dict[str, dict[str, Decimal]] = {}
    for action in design.action_words:
        action_cells = tuple(value for value in cells if value.action_word_id == action.word_id)
        statuses = {gate.status for cell in action_cells for gate in cell.gates}
        reasons = {reason for cell in action_cells for reason in cell.reason_codes}
        if GateStatus.FAIL in statuses or any(not value.delivery_valid for value in action_cells):
            disposition = ReferenceActionDisposition.UNSAFE
            reasons.add("REFERENCE_ACTION_NONCOMPENSATING_FAILURE")
            reduced: tuple[NamedDecimal, ...] = ()
        elif GateStatus.UNEVALUABLE in statuses or any(
            not value.supported for value in action_cells
        ):
            disposition = ReferenceActionDisposition.UNEVALUABLE
            reasons.add("REFERENCE_ACTION_UNEVALUABLE")
            reduced = ()
        else:
            disposition = ReferenceActionDisposition.ADMITTED
            reduced_values: list[NamedDecimal] = []
            reduced_by_action[action.word_id] = {}
            for dimension in design.comparison_dimensions:
                raw = tuple(
                    next(
                        item.value
                        for item in cell.dimension_values
                        if item.value_id == dimension.dimension_id
                    )
                    for cell in action_cells
                )
                value = _reduce_dimension(values=raw, reducer=dimension.reducer)
                reduced_by_action[action.word_id][dimension.dimension_id] = value
                reduced_values.append(
                    NamedDecimal(
                        value_id=dimension.dimension_id,
                        value=value,
                        unit=dimension.native_unit,
                    )
                )
            reduced = tuple(sorted(reduced_values, key=lambda value: value.value_id))
        provisional.append(
            ReferenceActionAssessment(
                assessment_id=(
                    f"reference-action-assessment.{design.reference_design_id}.{action.word_id}"
                ),
                reference_design=design_identity,
                action_word_id=action.word_id,
                cell_ids=tuple(sorted(value.cell_id for value in action_cells)),
                disposition=disposition,
                reduced_values=reduced,
                leading_class_member=False,
                reason_codes=tuple(sorted(reasons)),
            )
        )

    admitted_active = {
        value.action_word_id
        for value in provisional
        if value.disposition is ReferenceActionDisposition.ADMITTED
        and value.action_word_id != design.measured_hold_word_id
    }
    retained = set(admitted_active)
    ordered_dimensions = tuple(sorted(design.comparison_dimensions, key=lambda value: value.order))
    for kind in (
        ReferenceComparisonKind.TARGET,
        ReferenceComparisonKind.EFFORT,
        ReferenceComparisonKind.UNCERTAINTY,
        ReferenceComparisonKind.EXACT_CLASS,
    ):
        for dimension in (value for value in ordered_dimensions if value.kind is kind):
            if not retained:
                break
            candidate_values = {
                action_id: reduced_by_action[action_id][dimension.dimension_id]
                for action_id in retained
            }
            best = (
                max(candidate_values.values())
                if dimension.direction is ReferenceComparisonDirection.MAXIMIZE
                else min(candidate_values.values())
            )
            retained = {
                action_id
                for action_id, value in candidate_values.items()
                if _within_material_equivalence(
                    value=value,
                    best=best,
                    dimension=dimension,
                )
            }

    assessments = tuple(
        sorted(
            (
                ReferenceActionAssessment(
                    assessment_id=value.assessment_id,
                    reference_design=value.reference_design,
                    action_word_id=value.action_word_id,
                    cell_ids=value.cell_ids,
                    disposition=value.disposition,
                    reduced_values=value.reduced_values,
                    leading_class_member=value.action_word_id in retained,
                    reason_codes=value.reason_codes,
                )
                for value in provisional
            ),
            key=lambda value: value.assessment_id,
        )
    )

    members: list[ReferenceDispositionMember] = []
    canonical_action: str | None = None
    if retained:
        priority = {value: index for index, value in enumerate(design.action_priority_ids)}

        def canonical_key(action_id: str) -> tuple[Decimal | int, ...]:
            numeric: list[Decimal | int] = []
            for dimension in ordered_dimensions:
                value = reduced_by_action[action_id][dimension.dimension_id]
                numeric.append(
                    -value
                    if dimension.direction is ReferenceComparisonDirection.MAXIMIZE
                    else value
                )
            return (*numeric, priority[action_id])

        canonical_action = min(retained, key=canonical_key)
        members.extend(
            ReferenceDispositionMember(
                member_id=f"reference-member.active.{action_id}",
                kind=ReferenceDispositionMemberKind.ACTIVE,
                action_word_id=action_id,
            )
            for action_id in sorted(retained)
        )
    else:
        hold = next(
            (
                value
                for value in assessments
                if value.action_word_id == design.measured_hold_word_id
            ),
            None,
        )
        if hold is not None and hold.disposition is ReferenceActionDisposition.ADMITTED:
            members.append(
                ReferenceDispositionMember(
                    member_id="reference-member.measured-hold",
                    kind=ReferenceDispositionMemberKind.MEASURED_HOLD,
                    action_word_id=None,
                )
            )
        else:
            members.append(
                ReferenceDispositionMember(
                    member_id="reference-member.nonattempt",
                    kind=ReferenceDispositionMemberKind.NONATTEMPT,
                    action_word_id=None,
                )
            )
    return ReferenceDispositionClass(
        class_id=f"reference-disposition-class.{design.reference_design_id}",
        reference_design=design_identity,
        action_assessments=assessments,
        members=tuple(sorted(members, key=lambda value: value.member_id)),
        canonical_action_word_id=canonical_action,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )


def evaluate_reference_class_membership(
    *,
    reference_class: ReferenceDispositionClass,
    commitment_disposition: CommitmentDisposition,
    committed_action_word_id: str | None,
    receipt_id: str,
) -> ReferenceClassMembershipReceipt:
    """Compare a commitment with the exact action-aware class membership."""

    expected: tuple[ReferenceDispositionMemberKind, str | None]
    if commitment_disposition is CommitmentDisposition.ACTION_COMMITTED:
        expected = (ReferenceDispositionMemberKind.ACTIVE, committed_action_word_id)
    elif commitment_disposition is CommitmentDisposition.MEASURED_HOLD_COMMITTED:
        expected = (ReferenceDispositionMemberKind.MEASURED_HOLD, None)
    else:
        expected = (ReferenceDispositionMemberKind.NONATTEMPT, None)
    matched = any(
        value.kind is expected[0] and value.action_word_id == expected[1]
        for value in reference_class.members
    )
    return ReferenceClassMembershipReceipt(
        receipt_id=receipt_id,
        reference_class=ObjectIdentity.from_record(reference_class.class_id, reference_class),
        commitment_disposition=commitment_disposition,
        committed_action_word_id=committed_action_word_id,
        status=(
            ReferenceMembershipStatus.MEMBER if matched else ReferenceMembershipStatus.NONMEMBER
        ),
        reason_codes=() if matched else ("COMMITMENT_OUTSIDE_REFERENCE_CLASS",),
    )


__all__ = [
    "FiniteChartActionAssessment",
    "FiniteChartActionDisposition",
    "FiniteChartDispositionReferenceReceipt",
    'ReferenceActionAssessment',
    'ReferenceActionDisposition',
    'ReferenceCellAssessment',
    'ReferenceClassMembershipReceipt',
    'ReferenceDispositionClass',
    "ReferenceDispositionMemberKind",
    'ReferenceDispositionMember',
    'ReferenceGateAssessment',
    "ReferenceMembershipStatus",
    'evaluate_finite_chart_disposition',
    'evaluate_finite_reference_class',
    "evaluate_reference_class_membership",
]
