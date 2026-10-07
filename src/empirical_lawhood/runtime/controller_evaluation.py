"Programme-bound matched prospective controller use evaluation for controller evidence."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.control import ScientificCommitmentKind
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.controller_study import ImplementationBinding, ImplementationRole, ProspectiveControllerEvaluationPlan, UtilityDirection
from empirical_lawhood.planning.evidence_geometry import GatePredicateKind, GatePredicateSpec
from empirical_lawhood.runtime.controller_compiler import CompiledAtlasControllerStudy, ProspectiveBindingDisposition
from empirical_lawhood.runtime.controller_runtime import CommitmentDisposition, AtlasControllerTickReceipt, ExactActionDeliveryTrace


class ControllerUnitDisposition(StrEnum):
    ATTEMPTED = "ATTEMPTED"
    QUALIFIED_HOLD = "QUALIFIED_HOLD"
    NONATTEMPT = "NONATTEMPT"
    UNSAFE = "UNSAFE"
    DELIVERY_INVALID = "DELIVERY_INVALID"
    TECHNICAL_INVALID = "TECHNICAL_INVALID"
    UNEVALUABLE = "UNEVALUABLE"


class ControllerUseResult(StrEnum):
    CONTROLLER_USE_VALIDATED = "CONTROLLER_USE_VALIDATED"
    CONTROLLER_USE_POSITIVE_BUT_BELOW_MATERIALITY = "CONTROLLER_USE_POSITIVE_BUT_BELOW_MATERIALITY"
    CONTROLLER_USE_HOLD_DOMINANT = "CONTROLLER_USE_HOLD_DOMINANT"
    CONTROLLER_USE_UNSAFE = "CONTROLLER_USE_UNSAFE"
    CONTROLLER_USE_DELIVERY_INVALID = "CONTROLLER_USE_DELIVERY_INVALID"
    CONTROLLER_USE_PARTIAL_OR_HETEROGENEOUS = "CONTROLLER_USE_PARTIAL_OR_HETEROGENEOUS"
    CONTROLLER_USE_NEGATIVE = "CONTROLLER_USE_NEGATIVE"
    CONTROLLER_USE_TECHNICAL_FAILURE = "CONTROLLER_USE_TECHNICAL_FAILURE"
    CONTROLLER_USE_UNEVALUABLE = "CONTROLLER_USE_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class SealedOutcomeLocator(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/sealed-outcome-locator'

    locator_id: str
    branch_id: str
    model_member_id: str
    artifact: ArtifactIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("locator_id", self.locator_id),
            ("branch_id", self.branch_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("sealed outcome locator must remain evaluation-sealed")


@dataclass(frozen=True, slots=True)
class ProspectiveControllerBundle(CanonicalRecord):
    """One sealed matched preparation; nested members never become samples."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/prospective-controller-bundle'

    bundle_id: str
    evaluation_plan: ObjectIdentity
    independent_unit_id: str
    cohort_id: str
    common_initial_state_fingerprint: str
    common_prefix_fingerprint: str
    compiled_study: ObjectIdentity
    tick: ObjectIdentity
    model_member_ids: tuple[str, ...]
    controller_branch_id: str
    reference_branch_id: str
    hold_calibration_branch_id: str | None
    expected_controller_word: OccurrenceActionWord | None
    expected_reference_word: OccurrenceActionWord
    sealed_outcome_locators: tuple[SealedOutcomeLocator, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("bundle_id", self.bundle_id),
            ("independent_unit_id", self.independent_unit_id),
            ("cohort_id", self.cohort_id),
            ("controller_branch_id", self.controller_branch_id),
            ("reference_branch_id", self.reference_branch_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.hold_calibration_branch_id is not None:
            validate_stable_id(
                self.hold_calibration_branch_id, field_name="hold_calibration_branch_id"
            )
        validate_sha256(
            self.common_initial_state_fingerprint,
            field_name="common_initial_state_fingerprint",
        )
        validate_sha256(self.common_prefix_fingerprint, field_name="common_prefix_fingerprint")
        require_sorted_unique_strings(
            self.model_member_ids, field_name="model_member_ids", allow_empty=False
        )
        require_sorted_unique_ids(
            self.sealed_outcome_locators,
            attribute="locator_id",
            field_name="sealed_outcome_locators",
        )
        branches = {self.controller_branch_id, self.reference_branch_id}
        if self.hold_calibration_branch_id is not None:
            branches.add(self.hold_calibration_branch_id)
        expected = {(branch, member) for branch in branches for member in self.model_member_ids}
        observed = {
            (value.branch_id, value.model_member_id) for value in self.sealed_outcome_locators
        }
        if expected != observed or len(expected) != len(self.sealed_outcome_locators):
            raise ValueError("sealed bundle outcome locator grid is incomplete")


@dataclass(frozen=True, slots=True)
class RevealedBranchOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-branch-outcome'

    outcome_id: str
    sealed_locator: ObjectIdentity
    branch_id: str
    model_member_id: str
    initial_state_fingerprint: str
    prefix_fingerprint: str
    action_word: OccurrenceActionWord | None
    delivery_trace: ExactActionDeliveryTrace
    raw_outcomes: tuple[NamedDecimal, ...]
    technical_reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("outcome_id", self.outcome_id),
            ("branch_id", self.branch_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(self.initial_state_fingerprint, field_name="initial_state_fingerprint")
        validate_sha256(self.prefix_fingerprint, field_name="prefix_fingerprint")
        require_sorted_unique_ids(
            self.raw_outcomes, attribute="value_id", field_name="raw_outcomes"
        )
        require_sorted_unique_strings(
            self.technical_reason_codes, field_name="technical_reason_codes"
        )
        if self.outcome_access not in {
            OutcomeAccess.EVALUATOR_REVEAL,
            OutcomeAccess.EVALUATION_REVEALED,
        }:
            raise ValueError("revealed branch outcome lacks evaluator reveal access")
        if self.delivery_trace.commitment_kind is ScientificCommitmentKind.NONATTEMPT:
            if self.action_word is not None or self.raw_outcomes:
                raise ValueError("NONATTEMPT outcome fabricates action or response evidence")
        elif self.action_word != self.delivery_trace.expected_action_word:
            raise ValueError("revealed action differs from its observed delivery trace")


@dataclass(frozen=True, slots=True)
class RevealedProspectiveControllerBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/revealed-prospective-controller-bundle'

    reveal_id: str
    sealed_bundle: ProspectiveControllerBundle
    reveal_authorization: ObjectIdentity
    outcomes: tuple[RevealedBranchOutcome, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.reveal_id, field_name="reveal_id")
        require_sorted_unique_ids(self.outcomes, attribute="outcome_id", field_name="outcomes")
        locators = {value.locator_id: value for value in self.sealed_bundle.sealed_outcome_locators}
        expected = {(value.branch_id, value.model_member_id) for value in locators.values()}
        observed = {(value.branch_id, value.model_member_id) for value in self.outcomes}
        if expected != observed or len(expected) != len(self.outcomes):
            raise ValueError("revealed outcome grid differs from sealed bundle")
        for outcome in self.outcomes:
            locator = next(
                (
                    value
                    for value in locators.values()
                    if value.branch_id == outcome.branch_id
                    and value.model_member_id == outcome.model_member_id
                ),
                None,
            )
            if locator is None or outcome.sealed_locator != ObjectIdentity.from_record(
                locator.locator_id, locator
            ):
                raise ValueError("revealed outcome binds another sealed locator")


@dataclass(frozen=True, slots=True)
class OutcomePredicateEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/outcome-predicate-evaluation'

    evaluation_id: str
    predicate_id: str
    status: GateStatus
    margin: NamedDecimal | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        validate_stable_id(self.predicate_id, field_name="predicate_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is GateStatus.PASS:
            if self.reason_codes:
                raise ValueError("passing outcome predicate cannot carry reasons")
        elif not self.reason_codes:
            raise ValueError("nonpassing outcome predicate requires reasons")


@dataclass(frozen=True, slots=True)
class MemberControllerEffect(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/member-controller-effect'

    model_member_id: str
    effect: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.model_member_id, field_name="model_member_id")


@dataclass(frozen=True, slots=True)
class ControllerUnitEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/controller-unit-evaluation'

    unit_evaluation_id: str
    independent_unit_id: str
    evaluation_plan: ObjectIdentity
    compiled_study: ObjectIdentity
    bundle: ObjectIdentity
    disposition: ControllerUnitDisposition
    member_effects: tuple[MemberControllerEffect, ...]
    robust_effect: NamedDecimal | None
    predicate_evaluations: tuple[OutcomePredicateEvaluation, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_evaluation_id, field_name="unit_evaluation_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        require_sorted_unique_ids(
            self.member_effects,
            attribute="model_member_id",
            field_name="member_effects",
        )
        require_sorted_unique_ids(
            self.predicate_evaluations,
            attribute="evaluation_id",
            field_name="predicate_evaluations",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        effect_dispositions = {
            ControllerUnitDisposition.ATTEMPTED,
            ControllerUnitDisposition.QUALIFIED_HOLD,
        }
        if self.disposition in effect_dispositions:
            if not self.member_effects or self.robust_effect is None:
                raise ValueError("attempted controller use unit requires member and robust effects")
            if self.robust_effect.value != min(value.effect.value for value in self.member_effects):
                raise ValueError("unit robust effect is not the memberwise minimum")
        elif self.member_effects or self.robust_effect is not None:
            raise ValueError("invalid/nonattempt controller use unit cannot carry an effect")


@dataclass(frozen=True, slots=True)
class ControllerCohortAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/controller-cohort-adjudication'

    adjudication_id: str
    evaluation_plan: ObjectIdentity
    compiled_study: ObjectIdentity
    units: tuple[ControllerUnitEvaluation, ...]
    rostered_unit_count: int
    attempted_action_count: int
    qualified_hold_count: int
    nonattempt_count: int
    unsafe_count: int
    delivery_invalid_count: int
    technical_invalid_count: int
    unevaluable_count: int
    mean_effect: NamedDecimal | None
    one_sided_lower_bound: NamedDecimal | None
    active_coverage: Decimal
    result: ControllerUseResult
    claim_ceiling: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(self.units, attribute="independent_unit_id", field_name="units")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if any(unit.compiled_study != self.compiled_study for unit in self.units):
            raise ValueError("cohort units bind different compiled programmes")
        counts = (
            self.attempted_action_count,
            self.qualified_hold_count,
            self.nonattempt_count,
            self.unsafe_count,
            self.delivery_invalid_count,
            self.technical_invalid_count,
            self.unevaluable_count,
        )
        if any(value < 0 for value in counts) or sum(counts) != self.rostered_unit_count:
            raise ValueError("cohort disposition counts do not partition the roster")
        validate_decimal(
            self.active_coverage,
            field_name="active_coverage",
            minimum=Decimal(0),
        )
        if self.active_coverage > 1:
            raise ValueError("active_coverage cannot exceed one")
        attempted = self.attempted_action_count + self.qualified_hold_count
        if attempted:
            if self.mean_effect is None or self.one_sided_lower_bound is None:
                raise ValueError("attempted cohort requires effect summaries")
        elif self.mean_effect is not None or self.one_sided_lower_bound is not None:
            raise ValueError("unattempted cohort cannot carry effect summaries")


def evaluate_controller_outcome_predicate(
    *,
    evaluation_id: str,
    predicate: GatePredicateSpec,
    raw: dict[str, NamedDecimal],
) -> OutcomePredicateEvaluation:
    observed = raw.get(predicate.quantity_id)
    if observed is None:
        return OutcomePredicateEvaluation(
            evaluation_id=evaluation_id,
            predicate_id=predicate.predicate_id,
            status=GateStatus.UNEVALUABLE,
            margin=None,
            reason_codes=("OUTCOME_QUANTITY_MISSING",),
        )
    if predicate.native_unit != observed.unit:
        return OutcomePredicateEvaluation(
            evaluation_id=evaluation_id,
            predicate_id=predicate.predicate_id,
            status=GateStatus.UNEVALUABLE,
            margin=None,
            reason_codes=("OUTCOME_UNIT_MISMATCH",),
        )
    if predicate.predicate_kind is GatePredicateKind.SCALAR_AT_LEAST:
        if predicate.lower is None:
            raise AssertionError("at-least outcome predicate lost lower bound")
        margin_value = observed.value - predicate.lower.value
    elif predicate.predicate_kind is GatePredicateKind.SCALAR_AT_MOST:
        if predicate.upper is None:
            raise AssertionError("at-most outcome predicate lost upper bound")
        margin_value = predicate.upper.value - observed.value
    elif predicate.predicate_kind is GatePredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL:
        if predicate.lower is None or predicate.upper is None:
            raise AssertionError("interval outcome predicate lost bound")
        margin_value = min(
            observed.value - predicate.lower.value,
            predicate.upper.value - observed.value,
        )
    else:
        raise ValueError("controller use outcome predicate must be scalar")
    margin = NamedDecimal(
        value_id=f"margin.{evaluation_id}", value=margin_value, unit=observed.unit
    )
    return OutcomePredicateEvaluation(
        evaluation_id=evaluation_id,
        predicate_id=predicate.predicate_id,
        status=GateStatus.PASS if margin_value >= 0 else GateStatus.FAIL,
        margin=margin,
        reason_codes=() if margin_value >= 0 else ("OUTCOME_PREDICATE_FAILED",),
    )


def controller_prospective_student_summary(
    values: tuple[Decimal, ...],
    *,
    one_sided_critical_value: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    """Return mean, sample SD and the registered one-sided Student lower bound."""

    if not values:
        raise ValueError("controller use Student summary requires at least one unit")
    mean = sum(values, Decimal(0)) / Decimal(len(values))
    variance = (
        Decimal(0)
        if len(values) == 1
        else sum(((value - mean) ** 2 for value in values), Decimal(0)) / Decimal(len(values) - 1)
    )
    sample_sd = variance.sqrt()
    standard_error = (variance / Decimal(len(values))).sqrt()
    return (
        mean,
        sample_sd,
        mean - one_sided_critical_value * standard_error,
    )


class ControllerUseEvaluator:
    """Outcome-visible evaluator that cannot select or invoke controller actions."""

    def __init__(self, implementation_binding: ImplementationBinding) -> None:
        if implementation_binding.role is not ImplementationRole.OUTCOME_EVALUATOR:
            raise ValueError("controller use evaluator uses another implementation role")
        self.implementation_binding = implementation_binding

    def evaluate_unit(
        self,
        *,
        compiled: CompiledAtlasControllerStudy,
        tick: AtlasControllerTickReceipt,
        revealed: RevealedProspectiveControllerBundle,
    ) -> ControllerUnitEvaluation:
        if compiled.prospective_evaluation_binding.disposition is not ProspectiveBindingDisposition.BOUND:
            raise ValueError("compiled programme did not declare controller use")
        plan = compiled.study.prospective_evaluation
        if plan is None:
            raise AssertionError("bound controller use programme lost evaluation plan")
        if (
            compiled.implementation(ImplementationRole.OUTCOME_EVALUATOR)
            != self.implementation_binding
        ):
            raise ValueError("controller use evaluator implementation differs from compiled programme")
        sealed = revealed.sealed_bundle
        if (
            sealed.evaluation_plan != ObjectIdentity.from_record(plan.evaluation_plan_id, plan)
            or sealed.compiled_study
            != ObjectIdentity.from_record(compiled.compiled_study_id, compiled)
            or sealed.tick != ObjectIdentity.from_record(tick.tick_id, tick)
        ):
            raise ValueError("sealed bundle identities differ from programme/tick")
        if (
            sealed.independent_unit_id not in plan.independent_unit_ids
            or tick.commitment.observation.independent_unit_id != sealed.independent_unit_id
            or sealed.model_member_ids != plan.model_member_ids
            or sealed.controller_branch_id != plan.controller_branch_id
            or sealed.reference_branch_id != plan.reference_branch_id
            or sealed.hold_calibration_branch_id != plan.hold_calibration_branch_id
        ):
            raise ValueError("sealed bundle design differs from controller use plan")
        bundle_identity = ObjectIdentity.from_record(sealed.bundle_id, sealed)
        compiled_identity = ObjectIdentity.from_record(compiled.compiled_study_id, compiled)
        if tick.commitment.disposition is CommitmentDisposition.NONATTEMPT:
            return ControllerUnitEvaluation(
                unit_evaluation_id=f"unit-evaluation.{sealed.independent_unit_id}",
                independent_unit_id=sealed.independent_unit_id,
                evaluation_plan=ObjectIdentity.from_record(plan.evaluation_plan_id, plan),
                compiled_study=compiled_identity,
                bundle=bundle_identity,
                disposition=ControllerUnitDisposition.NONATTEMPT,
                member_effects=(),
                robust_effect=None,
                predicate_evaluations=(),
                reason_codes=tick.reason_codes,
            )
        action_binding = tick.commitment.action_binding
        if action_binding is None or sealed.expected_controller_word != action_binding.action_word:
            raise ValueError("sealed controller word differs from frozen commitment")
        outcomes = {(value.branch_id, value.model_member_id): value for value in revealed.outcomes}
        predicate_evaluations: list[OutcomePredicateEvaluation] = []
        member_effects: list[MemberControllerEffect] = []
        unit_reasons: set[str] = set()
        disposition = ControllerUnitDisposition.ATTEMPTED
        failure_dispositions: set[ControllerUnitDisposition] = set()
        for member in plan.model_member_ids:
            controller = outcomes[(plan.controller_branch_id, member)]
            reference = outcomes[(plan.reference_branch_id, member)]
            if (
                controller.initial_state_fingerprint != sealed.common_initial_state_fingerprint
                or reference.initial_state_fingerprint != sealed.common_initial_state_fingerprint
                or controller.prefix_fingerprint != sealed.common_prefix_fingerprint
                or reference.prefix_fingerprint != sealed.common_prefix_fingerprint
            ):
                failure_dispositions.add(ControllerUnitDisposition.UNEVALUABLE)
                unit_reasons.add("MATCHED_PREFIX_OR_PREPARATION_MISMATCH")
                continue
            if (
                controller.action_word != sealed.expected_controller_word
                or reference.action_word != sealed.expected_reference_word
                or controller.delivery_trace != tick.delivery_trace
                or not controller.delivery_trace.exact
                or not reference.delivery_trace.exact
                or reference.delivery_trace.expected_action_word != sealed.expected_reference_word
            ):
                failure_dispositions.add(ControllerUnitDisposition.DELIVERY_INVALID)
                unit_reasons.add("REQUESTED_ACCEPTED_APPLIED_REALIZED_MISMATCH")
                continue
            if controller.technical_reason_codes or reference.technical_reason_codes:
                failure_dispositions.add(ControllerUnitDisposition.TECHNICAL_INVALID)
                unit_reasons.update(controller.technical_reason_codes)
                unit_reasons.update(reference.technical_reason_codes)
                continue
            controller_raw = {value.value_id: value for value in controller.raw_outcomes}
            reference_raw = {value.value_id: value for value in reference.raw_outcomes}
            evaluations = tuple(
                evaluate_controller_outcome_predicate(
                    evaluation_id=(
                        f"outcome-gate.{sealed.independent_unit_id}.{member}.{predicate.predicate_id}"
                    ),
                    predicate=predicate,
                    raw=controller_raw,
                )
                for predicate in plan.evaluator_boundary.outcome_predicates
            )
            predicate_evaluations.extend(evaluations)
            if any(value.status is GateStatus.FAIL for value in evaluations):
                failure_dispositions.add(ControllerUnitDisposition.UNSAFE)
                unit_reasons.add("CONTROLLER_OUTCOME_UNSAFE")
                continue
            if any(value.status is GateStatus.UNEVALUABLE for value in evaluations):
                failure_dispositions.add(ControllerUnitDisposition.UNEVALUABLE)
                unit_reasons.add("CONTROLLER_OUTCOME_UNEVALUABLE")
                continue
            controller_effect = controller_raw.get(plan.effect_quantity_id)
            reference_effect = reference_raw.get(plan.effect_quantity_id)
            if (
                controller_effect is None
                or reference_effect is None
                or controller_effect.unit != plan.effect_native_unit
                or reference_effect.unit != plan.effect_native_unit
            ):
                failure_dispositions.add(ControllerUnitDisposition.UNEVALUABLE)
                unit_reasons.add("EFFECT_QUANTITY_MISSING_OR_WRONG_UNIT")
                continue
            if tick.commitment.disposition is CommitmentDisposition.MEASURED_HOLD_COMMITTED:
                difference = abs(controller_effect.value - reference_effect.value)
                if difference > plan.hold_null_tolerance.value:
                    failure_dispositions.add(ControllerUnitDisposition.DELIVERY_INVALID)
                    unit_reasons.add("HOLD_NULL_TOLERANCE_FAILED")
                    continue
                calibration_id = plan.hold_calibration_branch_id
                if calibration_id is not None:
                    calibration = outcomes[(calibration_id, member)]
                    calibration_raw = {value.value_id: value for value in calibration.raw_outcomes}
                    calibration_effect = calibration_raw.get(plan.effect_quantity_id)
                    if (
                        calibration_effect is None
                        or calibration_effect.unit != plan.effect_native_unit
                        or calibration.initial_state_fingerprint
                        != sealed.common_initial_state_fingerprint
                        or calibration.prefix_fingerprint != sealed.common_prefix_fingerprint
                        or calibration.action_word != sealed.expected_controller_word
                        or calibration.delivery_trace.expected_action_word
                        != sealed.expected_controller_word
                        or abs(calibration_effect.value - controller_effect.value)
                        > plan.hold_null_tolerance.value
                        or not calibration.delivery_trace.exact
                        or calibration.technical_reason_codes
                    ):
                        failure_dispositions.add(ControllerUnitDisposition.DELIVERY_INVALID)
                        unit_reasons.add("HOLD_CALIBRATION_FAILED")
                        continue
                value = Decimal(0)
            else:
                value = (
                    controller_effect.value - reference_effect.value
                    if plan.favorable_direction is UtilityDirection.HIGHER_IS_BETTER
                    else reference_effect.value - controller_effect.value
                )
            member_effects.append(
                MemberControllerEffect(
                    model_member_id=member,
                    effect=NamedDecimal(
                        value_id=f"effect.{sealed.independent_unit_id}.{member}",
                        value=value,
                        unit=plan.effect_native_unit,
                    ),
                )
            )
        if len(member_effects) != len(plan.model_member_ids):
            member_effects = []
        if member_effects:
            robust = NamedDecimal(
                value_id=f"robust-effect.{sealed.independent_unit_id}",
                value=min(value.effect.value for value in member_effects),
                unit=plan.effect_native_unit,
            )
            if tick.commitment.disposition is CommitmentDisposition.MEASURED_HOLD_COMMITTED:
                disposition = ControllerUnitDisposition.QUALIFIED_HOLD
                if robust.value != 0:
                    raise AssertionError("qualified HOLD effect must be exact zero")
            else:
                disposition = ControllerUnitDisposition.ATTEMPTED
            reasons: tuple[str, ...] = ()
        else:
            robust = None
            disposition = next(
                value
                for value in (
                    ControllerUnitDisposition.UNSAFE,
                    ControllerUnitDisposition.DELIVERY_INVALID,
                    ControllerUnitDisposition.TECHNICAL_INVALID,
                    ControllerUnitDisposition.UNEVALUABLE,
                )
                if value in failure_dispositions
            )
            reasons = tuple(sorted(unit_reasons))
        return ControllerUnitEvaluation(
            unit_evaluation_id=f"unit-evaluation.{sealed.independent_unit_id}",
            independent_unit_id=sealed.independent_unit_id,
            evaluation_plan=ObjectIdentity.from_record(plan.evaluation_plan_id, plan),
            compiled_study=compiled_identity,
            bundle=bundle_identity,
            disposition=disposition,
            member_effects=tuple(member_effects),
            robust_effect=robust,
            predicate_evaluations=tuple(
                sorted(predicate_evaluations, key=lambda value: value.evaluation_id)
            ),
            reason_codes=reasons,
        )

    def adjudicate_cohort(
        self,
        *,
        compiled: CompiledAtlasControllerStudy,
        plan: ProspectiveControllerEvaluationPlan,
        units: tuple[ControllerUnitEvaluation, ...],
    ) -> ControllerCohortAdjudication:
        plan_identity = ObjectIdentity.from_record(plan.evaluation_plan_id, plan)
        compiled_identity = ObjectIdentity.from_record(compiled.compiled_study_id, compiled)
        if (
            compiled.study.prospective_evaluation != plan
            or compiled.implementation(ImplementationRole.OUTCOME_EVALUATOR)
            != self.implementation_binding
            or compiled.prospective_evaluation_binding.disposition is not ProspectiveBindingDisposition.BOUND
            or plan.evaluator_boundary.outcome_evaluator_binding_id
            != self.implementation_binding.binding_id
            or any(unit.evaluation_plan != plan_identity for unit in units)
        ):
            raise ValueError("controller use cohort units/evaluator differ from the frozen plan")
        if any(unit.compiled_study != compiled_identity for unit in units):
            raise ValueError("controller use cohort units differ from the compiled programme")
        if tuple(value.independent_unit_id for value in units) != plan.independent_unit_ids:
            raise ValueError("controller use cohort units differ from frozen independent-unit roster")
        counts = {value: 0 for value in ControllerUnitDisposition}
        for unit in units:
            counts[unit.disposition] += 1
        effects = tuple(
            unit.robust_effect.value
            for unit in units
            if unit.robust_effect is not None
            and unit.disposition
            in {
                ControllerUnitDisposition.ATTEMPTED,
                ControllerUnitDisposition.QUALIFIED_HOLD,
            }
        )
        mean: NamedDecimal | None = None
        lower: NamedDecimal | None = None
        if effects:
            mean_value, _, lower_value = controller_prospective_student_summary(
                effects,
                one_sided_critical_value=plan.one_sided_critical_value,
            )
            mean = NamedDecimal(
                value_id=f"mean-effect.{plan.evaluation_plan_id}",
                value=mean_value,
                unit=plan.effect_native_unit,
            )
            lower = NamedDecimal(
                value_id=f"lower-bound.{plan.evaluation_plan_id}",
                value=lower_value,
                unit=plan.effect_native_unit,
            )
        attempted_action = counts[ControllerUnitDisposition.ATTEMPTED]
        qualified_hold = counts[ControllerUnitDisposition.QUALIFIED_HOLD]
        attempted_total = attempted_action + qualified_hold
        stratum_failure = any(
            sum(
                unit.disposition
                in {
                    ControllerUnitDisposition.ATTEMPTED,
                    ControllerUnitDisposition.QUALIFIED_HOLD,
                }
                for unit in units
                if unit.independent_unit_id in stratum.independent_unit_ids
            )
            < stratum.minimum_attempted_units
            for stratum in plan.strata
        )
        reasons: set[str] = set()
        if counts[ControllerUnitDisposition.UNSAFE]:
            result = ControllerUseResult.CONTROLLER_USE_UNSAFE
            reasons.add("UNSAFE_UNIT_PRESENT")
        elif counts[ControllerUnitDisposition.DELIVERY_INVALID]:
            result = ControllerUseResult.CONTROLLER_USE_DELIVERY_INVALID
            reasons.add("DELIVERY_INVALID_UNIT_PRESENT")
        elif counts[ControllerUnitDisposition.TECHNICAL_INVALID]:
            result = ControllerUseResult.CONTROLLER_USE_TECHNICAL_FAILURE
            reasons.add("TECHNICAL_INVALID_UNIT_PRESENT")
        elif attempted_total < plan.minimum_attempted_units:
            result = ControllerUseResult.CONTROLLER_USE_UNEVALUABLE
            reasons.add("MINIMUM_ATTEMPTED_UNITS_NOT_MET")
        elif stratum_failure:
            result = ControllerUseResult.CONTROLLER_USE_PARTIAL_OR_HETEROGENEOUS
            reasons.add("STRATUM_MINIMUM_NOT_MET")
        elif qualified_hold > attempted_action:
            result = ControllerUseResult.CONTROLLER_USE_HOLD_DOMINANT
            reasons.add("QUALIFIED_HOLD_DOMINATES_ACTIVE_ACTION")
        elif lower is not None and lower.value > plan.materiality.value:
            result = ControllerUseResult.CONTROLLER_USE_VALIDATED
        elif mean is not None and mean.value > 0:
            result = ControllerUseResult.CONTROLLER_USE_POSITIVE_BUT_BELOW_MATERIALITY
            reasons.add("LOWER_BOUND_DOES_NOT_EXCEED_MATERIALITY")
        elif effects:
            result = ControllerUseResult.CONTROLLER_USE_NEGATIVE
            reasons.add("NONPOSITIVE_MEAN_EFFECT")
        else:
            result = ControllerUseResult.CONTROLLER_USE_UNEVALUABLE
            reasons.add("NO_EVALUABLE_UNIT_EFFECTS")
        coverage = Decimal(attempted_action) / Decimal(len(units))
        return ControllerCohortAdjudication(
            adjudication_id=f"cohort-adjudication.{plan.evaluation_plan_id}",
            evaluation_plan=ObjectIdentity.from_record(plan.evaluation_plan_id, plan),
            compiled_study=compiled_identity,
            units=units,
            rostered_unit_count=len(units),
            attempted_action_count=attempted_action,
            qualified_hold_count=qualified_hold,
            nonattempt_count=counts[ControllerUnitDisposition.NONATTEMPT],
            unsafe_count=counts[ControllerUnitDisposition.UNSAFE],
            delivery_invalid_count=counts[ControllerUnitDisposition.DELIVERY_INVALID],
            technical_invalid_count=counts[ControllerUnitDisposition.TECHNICAL_INVALID],
            unevaluable_count=counts[ControllerUnitDisposition.UNEVALUABLE],
            mean_effect=mean,
            one_sided_lower_bound=lower,
            active_coverage=coverage,
            result=result,
            claim_ceiling=plan.maximum_claim_ceiling,
            reason_codes=tuple(sorted(reasons)),
        )


__all__ = [
    "ControllerCohortAdjudication",
    "ControllerUseEvaluator",
    "ControllerUseResult",
    "ControllerUnitDisposition",
    "ControllerUnitEvaluation",
    "MemberControllerEffect",
    "OutcomePredicateEvaluation",
    "ProspectiveControllerBundle",
    "RevealedBranchOutcome",
    "RevealedProspectiveControllerBundle",
    "SealedOutcomeLocator",
    "evaluate_controller_outcome_predicate",
    'controller_prospective_student_summary',
]
