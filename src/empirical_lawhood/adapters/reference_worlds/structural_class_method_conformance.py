"Truth-known adversarial inputs and evaluator for structural-class method assessment.\n\nExpected dispositions live only in :class:`MethodCaseTruth`.  Execution\nfunctions consume typed method inputs and cannot receive that truth record.\nOpaque case identities avoid embedding expected outcomes in branch names.\n"

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.adapters.methods.finite_cohomology import CochainComplex, compare_refinement
from empirical_lawhood.adapters.methods.exact_assignment_cohomology import AssignmentRepresentation, CochainAssignment, ExactClassStatus, assess_assignment, compare_state_enrichment
from empirical_lawhood.adapters.methods.action_word_descent import MapDisposition, TransformationAssessment
from empirical_lawhood.adapters.methods.witnessed_action_word_descent import ClosureCompatibilityMap, ClosurePrerequisiteEvidence, assess_closure
from empirical_lawhood.adapters.methods.structural_response_classes import ISDG_AXIS_IDS
from empirical_lawhood.adapters.methods.observed_structural_classes import AblationSemantics, DevelopmentSubsetSelection, EvidenceWorld, ObservedStructuralClassAxisEvidence, ObservedStructuralClassObservedEvidence, InterfaceSubsetScore, RestorationSemantics, evaluate_isdg
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)


class MethodOperation(StrEnum):
    COHOMOLOGY = "COHOMOLOGY"
    INVALID_RESTRICTION_REFUSAL = "INVALID_RESTRICTION_REFUSAL"
    STATE_ENRICHMENT = "STATE_ENRICHMENT"
    REFINEMENT = "REFINEMENT"
    CLOSURE = "CLOSURE"
    ISDG = "ISDG"


@dataclass(frozen=True, slots=True)
class CohomologyMethodInput:
    case_id: str
    complex: CochainComplex
    assignment: CochainAssignment
    consistency_floor: Decimal
    materiality_floor: Decimal


@dataclass(frozen=True, slots=True)
class InvalidRestrictionInput:
    case_id: str
    complex: CochainComplex
    assignment: CochainAssignment


@dataclass(frozen=True, slots=True)
class StateEnrichmentInput:
    case_id: str
    omitted_complex: CochainComplex
    omitted_assignment: CochainAssignment
    enriched_complex: CochainComplex
    enriched_assignment: CochainAssignment


@dataclass(frozen=True, slots=True)
class RefinementInput:
    case_id: str
    complex: CochainComplex
    assignment: CochainAssignment


@dataclass(frozen=True, slots=True)
class ClosureInput:
    case_id: str
    assessments: tuple[TransformationAssessment, ...]
    prerequisites: ClosurePrerequisiteEvidence
    compatibility: ClosureCompatibilityMap


@dataclass(frozen=True, slots=True)
class ObservedStructuralClassInput:
    case_id: str
    evidence: ObservedStructuralClassObservedEvidence


StructuralClassMethodInput = (
    CohomologyMethodInput
    | InvalidRestrictionInput
    | StateEnrichmentInput
    | RefinementInput
    | ClosureInput
    | ObservedStructuralClassInput
)


@dataclass(frozen=True, slots=True)
class MethodCaseTruth:
    case_id: str
    operation: MethodOperation
    expected_primary: str
    expected_secondary: str
    independent_exact_rule_id: str | None = None


@dataclass(frozen=True, slots=True)
class MethodPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/method-prediction'

    case_id: str
    operation: MethodOperation
    primary: str
    secondary: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        validate_nonempty(self.primary, field_name="primary")
        validate_nonempty(self.secondary, field_name="secondary")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=True,
        )


@dataclass(frozen=True, slots=True)
class MethodScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/method-score'

    case_id: str
    operation: MethodOperation
    primary_correct: bool
    secondary_correct: bool
    independent_exact_check_passed: bool | None
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=True,
        )
        expected = (
            self.primary_correct
            and self.secondary_correct
            and (self.independent_exact_check_passed is not False)
        )
        if self.passed != expected:
            raise ValueError("method score conjunction differs")


def _cycle(identifier: str) -> CochainComplex:
    return CochainComplex(
        complex_id=identifier,
        coefficient_system_id="constant-rational-additive",
        c0_dimension=3,
        c1_dimension=3,
        c2_dimension=0,
        delta0=((-1, 1, 0), (0, -1, 1), (1, 0, -1)),
        delta1=(),
        restriction_system_valid=True,
        rank_floor=Decimal("1e-12"),
        evidence_world="TRUTH_KNOWN_GENERATED",
    )


def _assignment(
    case_id: str,
    complex_id: str,
    values: tuple[str, ...],
    representation: AssignmentRepresentation = (AssignmentRepresentation.EXACT_FINITE_DECIMAL),
) -> CochainAssignment:
    return CochainAssignment(
        assignment_id=f"assignment.{case_id}",
        complex_id=complex_id,
        coefficients=tuple(Decimal(value) for value in values),
        representation=representation,
        native_unit="dimensionless",
        evidence_world='TRUTH_KNOWN_GENERATED',
    )


def _transformation(case_id: str, word_id: str) -> TransformationAssessment:
    return TransformationAssessment(
        assessment_id=f"assessment.{case_id}",
        word_id=word_id,
        disposition=MapDisposition.EXACT,
        functional_defect=Decimal("0"),
        uncertainty_floor=Decimal("1e-9"),
        materiality_floor=Decimal("1e-6"),
        falsifier_preserved=True,
        identity_valid=True,
        composition_valid=True,
        associativity_valid=True,
        reason_codes=(),
    )


def _selection(case_id: str) -> DevelopmentSubsetSelection:
    return DevelopmentSubsetSelection(
        selection_id=f"selection.{case_id}",
        development_partition_id="partition.method-development",
        candidate_coordinate_ids=("interface.temperature", "receiver.current"),
        causally_available_coordinate_ids=(
            "interface.temperature",
            "receiver.current",
        ),
        scores=(
            InterfaceSubsetScore(
                subset_id=f"subset.{case_id}.empty",
                coordinate_ids=(),
                heldout_development_error=Decimal("0.30"),
                eligible=True,
                reason_codes=(),
            ),
            InterfaceSubsetScore(
                subset_id=f"subset.{case_id}.interface",
                coordinate_ids=("interface.temperature",),
                heldout_development_error=Decimal("0.08"),
                eligible=True,
                reason_codes=(),
            ),
            InterfaceSubsetScore(
                subset_id=f"subset.{case_id}.interface-extra",
                coordinate_ids=("interface.temperature", "receiver.current"),
                heldout_development_error=Decimal("0.075"),
                eligible=True,
                reason_codes=(),
            ),
        ),
        selected_subset_id=f"subset.{case_id}.interface",
        sufficient_error_allowance=Decimal("0.10"),
        parsimony_margin=Decimal("0.01"),
        independent_development_unit_count=24,
        evaluation_outcomes_accessed=False,
    )


def _isdg_evidence(
    case_id: str,
    *,
    failed_axis: str | None = None,
    archive: bool = False,
) -> ObservedStructuralClassObservedEvidence:
    axes = tuple(
        ObservedStructuralClassAxisEvidence(
            axis_id=axis_id,
            supported=axis_id != failed_axis,
            estimate=Decimal("1"),
            threshold=Decimal("0.5"),
            independent_unit_count=48,
            preparation_ids=("preparation.method-evaluation",),
            lot_ids=("lot.generated-a", "lot.generated-b") if not archive else (),
            reason_codes=() if axis_id != failed_axis else ("axis-opposed",),
        )
        for axis_id in ISDG_AXIS_IDS
    )
    return ObservedStructuralClassObservedEvidence(
        evidence_id=f"evidence.{case_id}",
        context_id=f"context.{case_id}",
        evidence_world=(
            EvidenceWorld.RETROSPECTIVE_PHYSICAL_ARCHIVE
            if archive
            else EvidenceWorld.TRUTH_KNOWN_GENERATED
        ),
        development_selection=_selection(case_id),
        axes=axes,
        ablation_semantics=(
            AblationSemantics.RECEIVER_CHANNEL_WITHHOLDING
            if archive
            else AblationSemantics.SIMULATED_STATE_INTERVENTION
        ),
        restoration_semantics=(
            RestorationSemantics.PREDICTIVE_CHANNEL_REINTRODUCTION
            if archive
            else RestorationSemantics.SIMULATED_STATE_RESTORATION
        ),
        source_backed_action=not archive,
        causal_localization_supported=not archive,
        lot_identity_available=not archive,
        cohomology_disposition="SEPARATELY_ASSESSED",
    )


def generate_method_conformance_suite() -> tuple[
    tuple[StructuralClassMethodInput, ...], tuple[MethodCaseTruth, ...]
]:
    """Return method inputs and evaluator-only truth as separate tuples."""

    cycle = _cycle("complex.case-cycle")
    noncocycle_complex = CochainComplex(
        complex_id="complex.case-noncocycle",
        coefficient_system_id="constant-rational-additive",
        c0_dimension=1,
        c1_dimension=2,
        c2_dimension=1,
        delta0=((1,), (0,)),
        delta1=((0, 1),),
        restriction_system_valid=True,
        rank_floor=Decimal("1e-12"),
        evidence_world='TRUTH_KNOWN_GENERATED',
    )
    invalid_complex = replace(
        noncocycle_complex,
        complex_id="complex.case-invalid",
        delta0=((0,), (1,)),
        restriction_system_valid=False,
    )
    enriched_complex = CochainComplex(
        complex_id="complex.case-enriched",
        coefficient_system_id="constant-rational-additive",
        c0_dimension=3,
        c1_dimension=3,
        c2_dimension=0,
        delta0=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        delta1=(),
        restriction_system_valid=True,
        rank_floor=Decimal("1e-12"),
        evidence_world='TRUTH_KNOWN_GENERATED',
    )
    common = {"consistency_floor": Decimal("1e-9"), "materiality_floor": Decimal("1e-6")}
    inputs: list[StructuralClassMethodInput] = [
        CohomologyMethodInput(
            'case.cycle-balanced-coefficients',
            cycle,
            _assignment('case.cycle-balanced-coefficients', cycle.complex_id, ("1", "1", "-2")),
            **common,
        ),
        CohomologyMethodInput(
            'case.cycle-uniform-coefficients',
            cycle,
            _assignment('case.cycle-uniform-coefficients', cycle.complex_id, ("1", "1", "1")),
            **common,
        ),
        CohomologyMethodInput(
            'case.cycle-small-exact-coefficients',
            cycle,
            _assignment('case.cycle-small-exact-coefficients', cycle.complex_id, ("1e-12", "1e-12", "1e-12")),
            consistency_floor=Decimal("1e-6"),
            materiality_floor=Decimal("1e-3"),
        ),
        CohomologyMethodInput(
            'case.two-coordinate-small-second-coefficient',
            noncocycle_complex,
            _assignment('case.two-coordinate-small-second-coefficient', noncocycle_complex.complex_id, ("0", "1e-12")),
            consistency_floor=Decimal("1e-6"),
            materiality_floor=Decimal("1e-3"),
        ),
        CohomologyMethodInput(
            'case.cycle-empirical-small-coefficients',
            cycle,
            _assignment(
                'case.cycle-empirical-small-coefficients',
                cycle.complex_id,
                ("1e-10", "1e-10", "1e-10"),
                AssignmentRepresentation.EMPIRICAL_DECIMAL_OBSERVATION,
            ),
            consistency_floor=Decimal("1e-6"),
            materiality_floor=Decimal("1e-3"),
        ),
        CohomologyMethodInput(
            'case.cycle-zero-coefficients',
            cycle,
            _assignment('case.cycle-zero-coefficients', cycle.complex_id, ("0", "0", "0")),
            **common,
        ),
        InvalidRestrictionInput(
            'case.changed-restriction-matrices',
            invalid_complex,
            _assignment('case.changed-restriction-matrices', invalid_complex.complex_id, ("0", "1")),
        ),
        StateEnrichmentInput(
            'case.identity-map-state-enrichment',
            cycle,
            _assignment('case.state-enrichment-source-cycle', cycle.complex_id, ("1", "1", "1")),
            enriched_complex,
            _assignment('case.state-enrichment-identity-map', enriched_complex.complex_id, ("1", "1", "1")),
        ),
        RefinementInput(
            'case.uniform-cycle-refinement',
            cycle,
            _assignment('case.uniform-cycle-refinement', cycle.complex_id, ("1", "1", "1")),
        ),
    ]
    sparse_assessment = _transformation('case.mixed-square-only-witnesses', 'word.mixed-square-only-witness')
    sparse_prerequisites = ClosurePrerequisiteEvidence(
        evidence_id='prerequisite.mixed-square-only-witnesses',
        typed_arrows_tested=True,
        explicit_undefined_set_complete=True,
        identity_witness_ids=(),
        composition_witness_ids=(),
        associativity_witness_ids=(),
        additional_structure_witness_ids=(),
        horizontal_direction_witness_ids=("word.horizontal",),
        vertical_direction_witness_ids=("word.vertical",),
        eligible_mixed_square_ids=("square.mixed",),
        passed_mixed_square_ids=("square.mixed",),
        declared_square_prerequisite_ids=("square.interface",),
        satisfied_square_prerequisite_ids=("square.interface",),
    )
    sparse_compatibility = ClosureCompatibilityMap(
        map_id='compatibility.mixed-square-only-witnesses',
        source_schema=TransformationAssessment.SCHEMA,
        source_assessment_ids=(sparse_assessment.assessment_id,),
        target_capability_key='witnessed-action-word-closure',
        scientific_meaning_preserved=True,
    )
    identity = _transformation('case.category-arrow-identity-witness', "word.identity")
    composite = _transformation('case.category-composition-witness', "word.composite")
    full_prerequisites = ClosurePrerequisiteEvidence(
        evidence_id='prerequisite.complete-category-witnesses',
        typed_arrows_tested=True,
        explicit_undefined_set_complete=True,
        identity_witness_ids=("word.identity",),
        composition_witness_ids=("word.composite",),
        associativity_witness_ids=("word.composite",),
        additional_structure_witness_ids=(),
        horizontal_direction_witness_ids=("word.horizontal",),
        vertical_direction_witness_ids=("word.vertical",),
        eligible_mixed_square_ids=("square.mixed",),
        passed_mixed_square_ids=("square.mixed",),
        declared_square_prerequisite_ids=("square.interface",),
        satisfied_square_prerequisite_ids=("square.interface",),
    )
    full_compatibility = ClosureCompatibilityMap(
        map_id='compatibility.complete-category-witnesses',
        source_schema=TransformationAssessment.SCHEMA,
        source_assessment_ids=tuple(sorted((identity.assessment_id, composite.assessment_id))),
        target_capability_key='witnessed-action-word-closure',
        scientific_meaning_preserved=True,
    )
    inputs.extend(
        (
            ClosureInput(
                'case.mixed-square-only-witnesses',
                (sparse_assessment,),
                sparse_prerequisites,
                sparse_compatibility,
            ),
            ClosureInput(
                'case.complete-category-witnesses',
                (identity, composite),
                full_prerequisites,
                full_compatibility,
            ),
            ObservedStructuralClassInput('case.simulated-interface-state-axes', _isdg_evidence('case.simulated-interface-state-axes')),
            ObservedStructuralClassInput(
                'case.simulated-interface-final-axis-perturbation',
                _isdg_evidence('case.simulated-interface-final-axis-perturbation', failed_axis=ISDG_AXIS_IDS[7]),
            ),
            ObservedStructuralClassInput('case.simulated-interface-state-alternate-identity', _isdg_evidence('case.simulated-interface-state-alternate-identity')),
            ObservedStructuralClassInput('case.retrospective-channel-semantics', _isdg_evidence('case.retrospective-channel-semantics', archive=True)),
        )
    )
    truth = (
        MethodCaseTruth(
            'case.cycle-balanced-coefficients', MethodOperation.COHOMOLOGY, "EXACT_COBOUNDARY", "MATERIAL", "cycle-sum"
        ),
        MethodCaseTruth(
            'case.cycle-uniform-coefficients',
            MethodOperation.COHOMOLOGY,
            "EXACT_NONTRIVIAL_H1_CLASS",
            "MATERIAL",
            "cycle-sum",
        ),
        MethodCaseTruth(
            'case.cycle-small-exact-coefficients',
            MethodOperation.COHOMOLOGY,
            "EXACT_NONTRIVIAL_H1_CLASS",
            "SUBMATERIAL",
            "cycle-sum",
        ),
        MethodCaseTruth(
            'case.two-coordinate-small-second-coefficient',
            MethodOperation.COHOMOLOGY,
            "EXACT_NONCOCYCLE",
            "SUBMATERIAL",
            "direct-delta1",
        ),
        MethodCaseTruth(
            'case.cycle-empirical-small-coefficients', MethodOperation.COHOMOLOGY, "NOT_APPLICABLE_EMPIRICAL", "SUBMATERIAL"
        ),
        MethodCaseTruth(
            'case.cycle-zero-coefficients',
            MethodOperation.COHOMOLOGY,
            "EXACT_ZERO_UNOCCUPIED",
            "SUBMATERIAL",
            "cycle-sum",
        ),
        MethodCaseTruth(
            'case.changed-restriction-matrices',
            MethodOperation.INVALID_RESTRICTION_REFUSAL,
            "METHOD_REFUSAL",
            "INVALID_RESTRICTIONS",
        ),
        MethodCaseTruth(
            'case.identity-map-state-enrichment',
            MethodOperation.STATE_ENRICHMENT,
            "STATE_ONTOLOGY_DEFECT_RESOLVED",
            "EXACT_RESOLUTION",
        ),
        MethodCaseTruth(
            'case.uniform-cycle-refinement', MethodOperation.REFINEMENT, "STABLE", "EXACT_CLASS_PRESERVED", "cycle-sum"
        ),
        MethodCaseTruth(
            'case.mixed-square-only-witnesses', MethodOperation.CLOSURE, "TYPED_PARTIAL_GRAPH", "MIXED_SQUARE_ALONE_CANNOT_PROMOTE_CONJUNCTIVE_CLOSURE"
        ),
        MethodCaseTruth(
            'case.complete-category-witnesses', MethodOperation.CLOSURE, "CONJUNCTIVE_SQUARE_CLOSURE", "CONJUNCTIVE_SQUARE_CLOSURE_PREREQUISITES_COMPLETE"
        ),
        MethodCaseTruth(
            'case.simulated-interface-state-axes',
            MethodOperation.ISDG,
            "INTERFACE_STATE_DEPENDENT_GLOBALIZATION",
            "ALL_EIGHT_ISDG_AXES_PASS",
        ),
        MethodCaseTruth(
            'case.simulated-interface-final-axis-perturbation',
            MethodOperation.ISDG,
            "APPARENT_ANALOGY_ONLY",
            "NONCOMPENSATING_CONJUNCTION_OPPOSED",
        ),
        MethodCaseTruth(
            'case.simulated-interface-state-alternate-identity',
            MethodOperation.ISDG,
            "INTERFACE_STATE_DEPENDENT_GLOBALIZATION",
            "ALL_EIGHT_ISDG_AXES_PASS",
        ),
        MethodCaseTruth(
            'case.retrospective-channel-semantics', MethodOperation.ISDG, "UNEVALUABLE", "AT_LEAST_ONE_AXIS_UNEVALUABLE"
        ),
    )
    return tuple(inputs), truth


def execute_method_input(case: StructuralClassMethodInput) -> MethodPrediction:
    """Execute one input without access to its expected truth."""

    if isinstance(case, CohomologyMethodInput):
        assignment_result = assess_assignment(
            case.complex,
            case.assignment,
            consistency_floor=case.consistency_floor,
            materiality_floor=case.materiality_floor,
        )
        return MethodPrediction(
            case.case_id,
            MethodOperation.COHOMOLOGY,
            assignment_result.exact.status.value,
            assignment_result.operational.assignment_materiality.value,
            assignment_result.reason_codes,
        )
    if isinstance(case, InvalidRestrictionInput):
        try:
            assess_assignment(
                case.complex,
                case.assignment,
                consistency_floor=Decimal("1e-9"),
                materiality_floor=Decimal("1e-6"),
            )
        except ValueError as error:
            return MethodPrediction(
                case.case_id,
                MethodOperation.INVALID_RESTRICTION_REFUSAL,
                "METHOD_REFUSAL",
                "INVALID_RESTRICTIONS",
                (type(error).__name__,),
            )
        return MethodPrediction(
            case.case_id,
            MethodOperation.INVALID_RESTRICTION_REFUSAL,
            "INVALID_PROMOTION",
            "INVALID_RESTRICTIONS",
            ("INVALID_RESTRICTIONS_ACCEPTED",),
        )
    if isinstance(case, StateEnrichmentInput):
        omitted = assess_assignment(
            case.omitted_complex,
            case.omitted_assignment,
            consistency_floor=Decimal("1e-9"),
            materiality_floor=Decimal("1e-6"),
        )
        enriched = assess_assignment(
            case.enriched_complex,
            case.enriched_assignment,
            consistency_floor=Decimal("1e-9"),
            materiality_floor=Decimal("1e-6"),
        )
        enrichment_result = compare_state_enrichment(
            omitted, enriched, assessment_id=f"enrichment.{case.case_id}"
        )
        return MethodPrediction(
            case.case_id,
            MethodOperation.STATE_ENRICHMENT,
            enrichment_result.disposition.value,
            "EXACT_RESOLUTION" if enrichment_result.exact_resolution else "OPERATIONAL_ONLY",
            (),
        )
    if isinstance(case, RefinementInput):
        source = assess_assignment(
            case.complex,
            case.assignment,
            consistency_floor=Decimal("1e-9"),
            materiality_floor=Decimal("1e-6"),
        )
        refinement = compare_refinement(
            case.complex,
            case.complex,
            case.assignment.coefficients,
            case.assignment.coefficients,
            (
                (Decimal("1"), Decimal("0"), Decimal("0")),
                (Decimal("0"), Decimal("1"), Decimal("0")),
                (Decimal("0"), Decimal("0"), Decimal("1")),
            ),
            floor=Decimal("1e-12"),
            assessment_id=f"refinement.{case.case_id}",
        )
        return MethodPrediction(
            case.case_id,
            MethodOperation.REFINEMENT,
            "STABLE" if refinement.stable else "UNSTABLE",
            "EXACT_CLASS_PRESERVED"
            if source.exact.status is ExactClassStatus.EXACT_NONTRIVIAL_H1_CLASS
            else "EXACT_CLASS_NOT_PRESERVED",
            (),
        )
    if isinstance(case, ClosureInput):
        closure_result = assess_closure(
            case.assessments,
            case.prerequisites,
            case.compatibility,
            assessment_id=f"closure.{case.case_id}",
        )
        secondary = next(
            reason
            for reason in closure_result.reason_codes
            if reason
            in {
                "MIXED_SQUARE_ALONE_CANNOT_PROMOTE_CONJUNCTIVE_CLOSURE",
                "CONJUNCTIVE_SQUARE_CLOSURE_PREREQUISITES_COMPLETE",
            }
        )
        return MethodPrediction(
            case.case_id,
            MethodOperation.CLOSURE,
            closure_result.maximum_formal_rung.value,
            secondary,
            closure_result.reason_codes,
        )
    isdg_result = evaluate_isdg(case.evidence)
    return MethodPrediction(
        case.case_id,
        MethodOperation.ISDG,
        isdg_result.mechanism_disposition.value,
        isdg_result.reason_codes[0],
        tuple(
            sorted({reason for axis in isdg_result.axis_results for reason in axis.reason_codes})
        ),
    )


def _independent_exact_status(case: StructuralClassMethodInput, rule_id: str) -> str:
    """Second, closed-form exact check independent of the rank implementation."""

    if rule_id == "cycle-sum":
        if not isinstance(case, (CohomologyMethodInput, RefinementInput)):
            raise ValueError("cycle-sum rule received an incompatible input")
        values = case.assignment.coefficients
        if sum(values, start=Decimal("0")) != 0:
            return ExactClassStatus.EXACT_NONTRIVIAL_H1_CLASS.value
        if all(value == 0 for value in values):
            return ExactClassStatus.EXACT_ZERO_UNOCCUPIED.value
        return ExactClassStatus.EXACT_COBOUNDARY.value
    if rule_id == "direct-delta1":
        if not isinstance(case, CohomologyMethodInput):
            raise ValueError("direct-delta1 rule received an incompatible input")
        return (
            ExactClassStatus.EXACT_NONCOCYCLE.value
            if case.assignment.coefficients[1] != 0
            else ExactClassStatus.EXACT_COBOUNDARY.value
        )
    raise ValueError("unknown independent exact rule")


def score_method_predictions(
    inputs: tuple[StructuralClassMethodInput, ...],
    predictions: tuple[MethodPrediction, ...],
    truth: tuple[MethodCaseTruth, ...],
) -> tuple[MethodScore, ...]:
    """Join evaluator-only truth only after predictions have been issued."""

    input_by_id = {case.case_id: case for case in inputs}
    prediction_by_id = {prediction.case_id: prediction for prediction in predictions}
    if len(input_by_id) != len(inputs) or len(prediction_by_id) != len(predictions):
        raise ValueError("method suite identities are not unique")
    if set(input_by_id) != {item.case_id for item in truth} or set(prediction_by_id) != set(
        input_by_id
    ):
        raise ValueError("method suite input, prediction and truth identities differ")
    scores: list[MethodScore] = []
    for expected in truth:
        prediction = prediction_by_id[expected.case_id]
        exact_check: bool | None = None
        if expected.independent_exact_rule_id is not None:
            independent = _independent_exact_status(
                input_by_id[expected.case_id], expected.independent_exact_rule_id
            )
            expected_exact = (
                expected.expected_secondary
                if expected.operation is MethodOperation.REFINEMENT
                else expected.expected_primary
            )
            if expected.operation is MethodOperation.REFINEMENT:
                expected_exact = ExactClassStatus.EXACT_NONTRIVIAL_H1_CLASS.value
            exact_check = independent == expected_exact
        primary = prediction.primary == expected.expected_primary
        secondary = prediction.secondary == expected.expected_secondary
        reasons = tuple(
            sorted(
                reason
                for condition, reason in (
                    (not primary, "PRIMARY_DISPOSITION_MISMATCH"),
                    (not secondary, "SECONDARY_DISPOSITION_MISMATCH"),
                    (exact_check is False, "INDEPENDENT_EXACT_CHECK_MISMATCH"),
                )
                if condition
            )
        )
        scores.append(
            MethodScore(
                case_id=expected.case_id,
                operation=expected.operation,
                primary_correct=primary,
                secondary_correct=secondary,
                independent_exact_check_passed=exact_check,
                passed=primary and secondary and exact_check is not False,
                reason_codes=reasons,
            )
        )
    return tuple(scores)


__all__ = [
    'MethodCaseTruth',
    "StructuralClassMethodInput",
    'MethodOperation',
    'MethodPrediction',
    'MethodScore',
    'execute_method_input',
    'generate_method_conformance_suite',
    'score_method_predictions',
]
