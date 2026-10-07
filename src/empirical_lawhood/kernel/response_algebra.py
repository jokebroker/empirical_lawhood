"""Pure contracts for denominator-local response-algebra identification.

The records in this module carry categorical scientific dispositions and exact
identities only.  Trajectories, fitted matrices, arrays, figures and model
payloads remain external artifacts behind evidence links.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from .evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling, inherited_visibility
from .obligations import ObligationStatus
from .provenance import EvidenceLink, ObjectIdentity
from .references import ExecutableReference, NamedDecimal
from .serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_stable_id,
)
from .status import OperationalStatus, ReadinessStatus, ScientificStatus
from .systems import RelationalIdentity


class ActionStage(StrEnum):
    REQUESTED = "REQUESTED"
    ACCEPTED = "ACCEPTED"
    APPLIED = "APPLIED"
    REALIZED = "REALIZED"


class ActionWordMode(StrEnum):
    IDENTITY = "IDENTITY"
    SEQUENTIAL = "SEQUENTIAL"
    SIMULTANEOUS = "SIMULTANEOUS"


class PortMateriality(StrEnum):
    NULL = "NULL"
    MATERIAL = "MATERIAL"
    MIXED = "MIXED"
    UNEVALUABLE = "UNEVALUABLE"


class TemporalComposition(StrEnum):
    STATIONARY_SEMIGROUP = "STATIONARY_SEMIGROUP"
    NONSTATIONARY_COCYCLE = "NONSTATIONARY_COCYCLE"
    HISTORY_AUGMENTED_CLOSURE = "HISTORY_AUGMENTED_CLOSURE"
    NONCLOSED = "NONCLOSED"
    UNEVALUABLE = "UNEVALUABLE"


class SimultaneousComposition(StrEnum):
    ADDITIVE_EQUIVALENT = "ADDITIVE_EQUIVALENT"
    NONLINEAR_INTERACTION = "NONLINEAR_INTERACTION"
    BELOW_RESOLUTION = "BELOW_RESOLUTION"
    UNEVALUABLE = "UNEVALUABLE"


class SequentialComposition(StrEnum):
    COMMUTATOR_EQUIVALENT = "COMMUTATOR_EQUIVALENT"
    LTI_OR_DRIFT_EXPLAINED = "LTI_OR_DRIFT_EXPLAINED"
    MATERIAL_NONCOMMUTATIVE = "MATERIAL_NONCOMMUTATIVE"
    BELOW_RESOLUTION = "BELOW_RESOLUTION"
    UNEVALUABLE = "UNEVALUABLE"


class AlgebraClosure(StrEnum):
    CLOSED_IN_GENERATOR_SPAN = "CLOSED_IN_GENERATOR_SPAN"
    GENERATES_NEW_DIRECTIONS = "GENERATES_NEW_DIRECTIONS"
    PROJECTED_OR_STATE_NONCLOSED = "PROJECTED_OR_STATE_NONCLOSED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNEVALUABLE = "UNEVALUABLE"


class StateSufficiency(StrEnum):
    MARKOV_SUFFICIENT = "MARKOV_SUFFICIENT"
    FINITE_HISTORY_SUFFICIENT = "FINITE_HISTORY_SUFFICIENT"
    HIDDEN_STATE_LIMITED = "HIDDEN_STATE_LIMITED"
    TIME_VARYING_DENOMINATOR = "TIME_VARYING_DENOMINATOR"
    UNEVALUABLE = "UNEVALUABLE"


class AlgebraRepresentation(StrEnum):
    DETERMINISTIC_SMOOTH = "DETERMINISTIC_SMOOTH"
    DETERMINISTIC_FINITE = "DETERMINISTIC_FINITE"
    STOCHASTIC_KERNEL = "STOCHASTIC_KERNEL"
    HYBRID_SWITCHING = "HYBRID_SWITCHING"
    UNEVALUABLE = "UNEVALUABLE"


class ReceiverVisibility(StrEnum):
    FAITHFUL_AT_TESTED_RESOLUTION = "FAITHFUL_AT_TESTED_RESOLUTION"
    BRACKET_HIDDEN_BY_PROJECTION = "BRACKET_HIDDEN_BY_PROJECTION"
    GAUGE_DEPENDENT = "GAUGE_DEPENDENT"
    UNEVALUABLE = "UNEVALUABLE"


class ActionQuotient(StrEnum):
    FAITHFUL_PORTS = "FAITHFUL_PORTS"
    EQUIVALENT_PORTS = "EQUIVALENT_PORTS"
    PARTIALLY_COLLAPSED = "PARTIALLY_COLLAPSED"
    UNEVALUABLE = "UNEVALUABLE"


class AdmissionAnnotation(StrEnum):
    NOT_TESTED = "NOT_TESTED"
    ADMITTED = "ADMITTED"
    EMPTY = "EMPTY"
    PARTIAL = "PARTIAL"
    UNEVALUABLE = "UNEVALUABLE"


class PairScientificLabel(StrEnum):
    NO_MATERIAL_RESPONSE = "NO_MATERIAL_RESPONSE"
    COMPOSITION_NULL_AT_RESOLUTION = "COMPOSITION_NULL_AT_RESOLUTION"
    MATERIAL_AFFINE_COMMUTATIVE = "MATERIAL_AFFINE_COMMUTATIVE"
    NONLINEAR_COMMUTATIVE = "NONLINEAR_COMMUTATIVE"
    PATH_ORDER_EXPLAINED_BY_LTI_OR_DRIFT = "PATH_ORDER_EXPLAINED_BY_LTI_OR_DRIFT"
    FINITE_NONCOMMUTATIVE_CLOSED = "FINITE_NONCOMMUTATIVE_CLOSED"
    SMOOTH_NONCOMMUTATIVE_LIE_LOCAL = "SMOOTH_NONCOMMUTATIVE_LIE_LOCAL"
    HYBRID_NONCOMMUTATIVE = "HYBRID_NONCOMMUTATIVE"
    APPARENT_NONCOMMUTATIVITY_STATE_NOT_CLOSED = "APPARENT_NONCOMMUTATIVITY_STATE_NOT_CLOSED"
    STOCHASTIC_NONCOMMUTATIVE = "STOCHASTIC_NONCOMMUTATIVE"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


class SignatureAxis(StrEnum):
    PORT_MATERIALITY = "PORT_MATERIALITY"
    TEMPORAL_COMPOSITION = "TEMPORAL_COMPOSITION"
    SIMULTANEOUS_COMPOSITION = "SIMULTANEOUS_COMPOSITION"
    SEQUENTIAL_COMPOSITION = "SEQUENTIAL_COMPOSITION"
    ALGEBRA_CLOSURE = "ALGEBRA_CLOSURE"
    STATE_SUFFICIENCY = "STATE_SUFFICIENCY"
    REPRESENTATION = "REPRESENTATION"
    RECEIVER_VISIBILITY = "RECEIVER_VISIBILITY"
    ACTION_QUOTIENT = "ACTION_QUOTIENT"
    ADMISSION_ANNOTATION = "ADMISSION_ANNOTATION"


class StructuralClassPrototype(StrEnum):
    RESPONSE_NULL = "RESPONSE_NULL"
    TEMPORAL_COCYCLE__COMPOSITION_NULL_AT_RESOLUTION = (
        "TEMPORAL_COCYCLE__COMPOSITION_NULL_AT_RESOLUTION"
    )
    MATERIAL_AFFINE_COMMUTATIVE = "MATERIAL_AFFINE_COMMUTATIVE"
    NONLINEAR_COMMUTATIVE = "NONLINEAR_COMMUTATIVE"
    CLOSED_NONCOMMUTATIVE = "CLOSED_NONCOMMUTATIVE"
    HISTORY_AUGMENTED_NONCOMMUTATIVE = "HISTORY_AUGMENTED_NONCOMMUTATIVE"
    HYBRID_SWITCHING_NONCOMMUTATIVE = "HYBRID_SWITCHING_NONCOMMUTATIVE"
    PROJECTED_OR_UNCLOSED = "PROJECTED_OR_UNCLOSED"


class StructuralClassDisposition(StrEnum):
    MATCHED = "MATCHED"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"


class ClassPromotionLevel(StrEnum):
    LOCAL_SIGNATURE = "LOCAL_SIGNATURE"
    SUBSTRATE_CLASS = "SUBSTRATE_CLASS"
    STRUCTURAL_CLASS_CANDIDATE = "STRUCTURAL_CLASS_CANDIDATE"
    STRUCTURAL_UNIVERSALITY_CLASS = "STRUCTURAL_UNIVERSALITY_CLASS"


class ResponseAlgebraCheckKind(StrEnum):
    MEASUREMENT = "MEASUREMENT"
    ACTION_DELIVERY = "ACTION_DELIVERY"
    RESET = "RESET"
    CLOCK_AND_UNIT = "CLOCK_AND_UNIT"
    WORD_AND_PREFIX_SUPPORT = "WORD_AND_PREFIX_SUPPORT"
    MATERIALITY_AND_SENSITIVITY = "MATERIALITY_AND_SENSITIVITY"
    TEMPORAL_COMPOSITION = "TEMPORAL_COMPOSITION"
    ACTION_COMPOSITION = "ACTION_COMPOSITION"
    STATE_AND_ALGEBRA_CLOSURE = "STATE_AND_ALGEBRA_CLOSURE"
    STRUCTURAL_CONVERGENCE = "STRUCTURAL_CONVERGENCE"
    DECISIVE_FALSIFIERS = "DECISIVE_FALSIFIERS"
    OUTCOME_VISIBILITY = "OUTCOME_VISIBILITY"


_AXIS_VALUE_TYPES: dict[SignatureAxis, type[StrEnum]] = {
    SignatureAxis.PORT_MATERIALITY: PortMateriality,
    SignatureAxis.TEMPORAL_COMPOSITION: TemporalComposition,
    SignatureAxis.SIMULTANEOUS_COMPOSITION: SimultaneousComposition,
    SignatureAxis.SEQUENTIAL_COMPOSITION: SequentialComposition,
    SignatureAxis.ALGEBRA_CLOSURE: AlgebraClosure,
    SignatureAxis.STATE_SUFFICIENCY: StateSufficiency,
    SignatureAxis.REPRESENTATION: AlgebraRepresentation,
    SignatureAxis.RECEIVER_VISIBILITY: ReceiverVisibility,
    SignatureAxis.ACTION_QUOTIENT: ActionQuotient,
    SignatureAxis.ADMISSION_ANNOTATION: AdmissionAnnotation,
}

_REASON_PREFIXES = (
    "ACTION_DELIVERY_",
    "ACTION_QUOTIENT_",
    "AUTHORITY_",
    "CLOCK_",
    "EVIDENCE_CEILING_",
    "FALSIFIER_",
    "FLOOR_",
    "MATERIALITY_",
    "MEASUREMENT_",
    "NUMERICAL_VIEW_",
    "ORDER_",
    "OUTCOME_ACCESS_",
    "POWER_",
    "PREFIX_",
    "PREREQUISITE_",
    "PROMOTION_",
    "RECEIVER_",
    "RESET_",
    "RESOURCE_",
    "SIMULTANEOUS_",
    "SOURCE_",
    "STATE_CLOSURE_",
    "STATIONARITY_",
    "SUPPORT_",
    "TEMPORAL_",
    "UNCERTAINTY_",
    "UNIT_",
)


def _validate_reason_codes(values: tuple[str, ...]) -> None:
    require_sorted_unique_strings(values, field_name="reason_codes")
    if any(not value.startswith(_REASON_PREFIXES) for value in values):
        raise ValueError("response-algebra reason code is outside the frozen families")


@dataclass(frozen=True, slots=True)
class ActionStageValue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-stage-value'

    stage: ActionStage
    value: Decimal
    native_unit: str
    clock_id: str
    clock_coordinate: Decimal

    def __post_init__(self) -> None:
        validate_decimal(self.value, field_name="value")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_stable_id(self.clock_id, field_name="clock_id")
        validate_decimal(self.clock_coordinate, field_name="clock_coordinate")


@dataclass(frozen=True, slots=True)
class ActionLetter(CanonicalRecord):
    """One delivered action occurrence with all four action stages."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-letter'

    letter_id: str
    port_id: str
    requested: ActionStageValue
    accepted: ActionStageValue
    applied: ActionStageValue
    realized: ActionStageValue
    duration: Decimal
    duration_unit: str
    support_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.letter_id, field_name="letter_id")
        validate_stable_id(self.port_id, field_name="port_id")
        validate_stable_id(self.support_id, field_name="support_id")
        expected = (
            (self.requested, ActionStage.REQUESTED),
            (self.accepted, ActionStage.ACCEPTED),
            (self.applied, ActionStage.APPLIED),
            (self.realized, ActionStage.REALIZED),
        )
        if any(value.stage is not stage for value, stage in expected):
            raise ValueError("action-stage field and stage identity differ")
        units = {value.native_unit for value, _stage in expected}
        if len(units) != 1:
            raise ValueError("all action stages must retain one native unit")
        validate_decimal(self.duration, field_name="duration", minimum=Decimal("0"))
        if self.duration == 0:
            raise ValueError("action-letter duration must be positive")
        validate_nonempty(self.duration_unit, field_name="duration_unit")


@dataclass(frozen=True, slots=True)
class LetterActionWord(CanonicalRecord):
    """A chronological delivered word; repeated letters remain repeated records."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/letter-action-word'

    word_id: str
    mode: ActionWordMode
    letters: tuple[ActionLetter, ...]
    denominator_id: str
    retained_history_id: str
    receiver_id: str
    horizon_id: str
    prefix_support_ids: tuple[str, ...]
    supported: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("word_id", self.word_id),
            ("denominator_id", self.denominator_id),
            ("retained_history_id", self.retained_history_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
        ):
            validate_stable_id(value, field_name=name)
        if len(self.letters) > 32:
            raise ValueError("action word exceeds its bounded length")
        if self.mode is ActionWordMode.IDENTITY and self.letters:
            raise ValueError("identity action word cannot contain letters")
        if self.mode is ActionWordMode.SEQUENTIAL:
            if not self.letters:
                raise ValueError("sequential action word requires at least one letter")
            sequential_clocks = tuple(letter.applied.clock_coordinate for letter in self.letters)
            if any(
                later <= earlier for earlier, later in zip(sequential_clocks, sequential_clocks[1:])
            ):
                raise ValueError("sequential action-word clocks must be strictly increasing")
        if self.mode is ActionWordMode.SIMULTANEOUS:
            if len(self.letters) < 2:
                raise ValueError("simultaneous action word requires at least two letters")
            if len({letter.letter_id for letter in self.letters}) != len(self.letters):
                raise ValueError("simultaneous action word requires distinct action letters")
            simultaneous_clocks = {letter.applied.clock_coordinate for letter in self.letters}
            if len(simultaneous_clocks) != 1:
                raise ValueError("simultaneous action-word clocks must match exactly")
        if len(self.prefix_support_ids) != len(self.letters) + 1:
            raise ValueError("action word requires identity and every successive prefix")
        _validate_reason_codes(self.reason_codes)
        for prefix_id in self.prefix_support_ids:
            validate_stable_id(prefix_id, field_name="prefix_support_ids")
        if self.supported and self.reason_codes:
            raise ValueError("supported action word cannot carry support-failure reasons")
        if not self.supported and not self.reason_codes:
            raise ValueError("unsupported action word requires reason codes")

    @property
    def chronological_letter_ids(self) -> tuple[str, ...]:
        return tuple(letter.letter_id for letter in self.letters)


@dataclass(frozen=True, slots=True)
class TemporalCompositionAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/temporal-composition-assessment'

    assessment_id: str
    disposition: TemporalComposition
    held_out_independent_units: int
    metrics: tuple[NamedDecimal, ...]
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        if self.held_out_independent_units < 0:
            raise ValueError("held-out independent-unit count must be nonnegative")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        _validate_reason_codes(self.reason_codes)
        if self.disposition is TemporalComposition.UNEVALUABLE:
            if not self.reason_codes:
                raise ValueError("unevaluable temporal composition requires reasons")
        elif self.held_out_independent_units == 0 or not self.evidence_link_ids:
            raise ValueError("evaluated temporal composition requires held-out evidence")


@dataclass(frozen=True, slots=True)
class ActionCompositionAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-composition-assessment'

    assessment_id: str
    first_letter_id: str
    second_letter_id: str
    first_port_materiality: PortMateriality
    second_port_materiality: PortMateriality
    simultaneous: SimultaneousComposition
    sequential: SequentialComposition
    pair_label: PairScientificLabel
    held_out_independent_units: int
    metrics: tuple[NamedDecimal, ...]
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("first_letter_id", self.first_letter_id),
            ("second_letter_id", self.second_letter_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.first_letter_id == self.second_letter_id:
            raise ValueError("action-composition assessment requires two distinct letters")
        if self.held_out_independent_units < 0:
            raise ValueError("held-out independent-unit count must be nonnegative")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        _validate_reason_codes(self.reason_codes)
        unevaluable = (
            self.first_port_materiality is PortMateriality.UNEVALUABLE
            or self.second_port_materiality is PortMateriality.UNEVALUABLE
            or self.simultaneous is SimultaneousComposition.UNEVALUABLE
            or self.sequential is SequentialComposition.UNEVALUABLE
            or self.pair_label is PairScientificLabel.UNEVALUABLE
        )
        if unevaluable and not self.reason_codes:
            raise ValueError("unevaluable action composition requires reasons")
        if not unevaluable and (self.held_out_independent_units == 0 or not self.evidence_link_ids):
            raise ValueError("evaluated action composition requires held-out evidence")
        equivalent = self.sequential is SequentialComposition.COMMUTATOR_EQUIVALENT
        material = (
            self.first_port_materiality is PortMateriality.MATERIAL
            and self.second_port_materiality is PortMateriality.MATERIAL
        )
        if equivalent and not material:
            raise ValueError("commutator equivalence requires two material constituent ports")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraSignature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/response-algebra-signature'

    signature_id: str
    port_materiality: PortMateriality
    temporal_composition: TemporalComposition
    simultaneous_composition: SimultaneousComposition
    sequential_composition: SequentialComposition
    algebra_closure: AlgebraClosure
    state_sufficiency: StateSufficiency
    representation: AlgebraRepresentation
    receiver_visibility: ReceiverVisibility
    action_quotient: ActionQuotient
    admission_annotation: AdmissionAnnotation
    metrics: tuple[NamedDecimal, ...]
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.signature_id, field_name="signature_id")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        _validate_reason_codes(self.reason_codes)
        values = self.axis_values()
        if "UNEVALUABLE" in values.values() and not self.reason_codes:
            raise ValueError("an unevaluable signature axis requires reason codes")

    def axis_values(self) -> dict[SignatureAxis, str]:
        return {
            SignatureAxis.PORT_MATERIALITY: self.port_materiality.value,
            SignatureAxis.TEMPORAL_COMPOSITION: self.temporal_composition.value,
            SignatureAxis.SIMULTANEOUS_COMPOSITION: self.simultaneous_composition.value,
            SignatureAxis.SEQUENTIAL_COMPOSITION: self.sequential_composition.value,
            SignatureAxis.ALGEBRA_CLOSURE: self.algebra_closure.value,
            SignatureAxis.STATE_SUFFICIENCY: self.state_sufficiency.value,
            SignatureAxis.REPRESENTATION: self.representation.value,
            SignatureAxis.RECEIVER_VISIBILITY: self.receiver_visibility.value,
            SignatureAxis.ACTION_QUOTIENT: self.action_quotient.value,
            SignatureAxis.ADMISSION_ANNOTATION: self.admission_annotation.value,
        }


@dataclass(frozen=True, slots=True)
class SignatureCriterion(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/signature-criterion'

    axis: SignatureAxis
    allowed_values: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_strings(
            self.allowed_values,
            field_name="allowed_values",
            allow_empty=False,
        )
        allowed = {value.value for value in _AXIS_VALUE_TYPES[self.axis]}
        if not set(self.allowed_values) <= allowed:
            raise ValueError("signature criterion contains a value from another axis")


@dataclass(frozen=True, slots=True)
class StructuralClassDefinition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/structural-class-definition'

    class_id: str
    prototype: StructuralClassPrototype
    criteria: tuple[SignatureCriterion, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.class_id, field_name="class_id")
        require_sorted_unique_ids(self.criteria, attribute="axis", field_name="criteria")
        if not self.criteria:
            raise ValueError("structural class requires at least one categorical criterion")

    def assess(self, signature: ResponseAlgebraSignature) -> StructuralClassDisposition:
        observed = signature.axis_values()
        if any(observed[criterion.axis] == "UNEVALUABLE" for criterion in self.criteria):
            return StructuralClassDisposition.UNEVALUABLE
        if all(observed[value.axis] in value.allowed_values for value in self.criteria):
            return StructuralClassDisposition.MATCHED
        return StructuralClassDisposition.OPPOSED


def prototype_class_definitions() -> tuple[StructuralClassDefinition, ...]:
    "Return the frozen prototype conjunctions in canonical ID order."

    def criterion(axis: SignatureAxis, *values: StrEnum) -> SignatureCriterion:
        return SignatureCriterion(
            axis=axis,
            allowed_values=tuple(sorted(value.value for value in values)),
        )

    definitions = (
        StructuralClassDefinition(
            class_id="closed-noncommutative",
            prototype=StructuralClassPrototype.CLOSED_NONCOMMUTATIVE,
            criteria=tuple(
                sorted(
                    (
                        criterion(
                            SignatureAxis.ALGEBRA_CLOSURE,
                            AlgebraClosure.CLOSED_IN_GENERATOR_SPAN,
                        ),
                        criterion(SignatureAxis.PORT_MATERIALITY, PortMateriality.MATERIAL),
                        criterion(
                            SignatureAxis.SEQUENTIAL_COMPOSITION,
                            SequentialComposition.MATERIAL_NONCOMMUTATIVE,
                        ),
                        criterion(
                            SignatureAxis.STATE_SUFFICIENCY,
                            StateSufficiency.MARKOV_SUFFICIENT,
                        ),
                    ),
                    key=lambda value: value.axis,
                )
            ),
        ),
        StructuralClassDefinition(
            class_id="history-augmented-noncommutative",
            prototype=StructuralClassPrototype.HISTORY_AUGMENTED_NONCOMMUTATIVE,
            criteria=tuple(
                sorted(
                    (
                        criterion(SignatureAxis.PORT_MATERIALITY, PortMateriality.MATERIAL),
                        criterion(
                            SignatureAxis.SEQUENTIAL_COMPOSITION,
                            SequentialComposition.MATERIAL_NONCOMMUTATIVE,
                        ),
                        criterion(
                            SignatureAxis.STATE_SUFFICIENCY,
                            StateSufficiency.FINITE_HISTORY_SUFFICIENT,
                        ),
                        criterion(
                            SignatureAxis.TEMPORAL_COMPOSITION,
                            TemporalComposition.HISTORY_AUGMENTED_CLOSURE,
                        ),
                    ),
                    key=lambda value: value.axis,
                )
            ),
        ),
        StructuralClassDefinition(
            class_id="hybrid-switching-noncommutative",
            prototype=StructuralClassPrototype.HYBRID_SWITCHING_NONCOMMUTATIVE,
            criteria=tuple(
                sorted(
                    (
                        criterion(SignatureAxis.PORT_MATERIALITY, PortMateriality.MATERIAL),
                        criterion(
                            SignatureAxis.REPRESENTATION,
                            AlgebraRepresentation.HYBRID_SWITCHING,
                        ),
                        criterion(
                            SignatureAxis.SEQUENTIAL_COMPOSITION,
                            SequentialComposition.MATERIAL_NONCOMMUTATIVE,
                        ),
                    ),
                    key=lambda value: value.axis,
                )
            ),
        ),
        StructuralClassDefinition(
            class_id="material-affine-commutative",
            prototype=StructuralClassPrototype.MATERIAL_AFFINE_COMMUTATIVE,
            criteria=tuple(
                sorted(
                    (
                        criterion(SignatureAxis.PORT_MATERIALITY, PortMateriality.MATERIAL),
                        criterion(
                            SignatureAxis.SEQUENTIAL_COMPOSITION,
                            SequentialComposition.COMMUTATOR_EQUIVALENT,
                        ),
                        criterion(
                            SignatureAxis.SIMULTANEOUS_COMPOSITION,
                            SimultaneousComposition.ADDITIVE_EQUIVALENT,
                        ),
                        criterion(
                            SignatureAxis.TEMPORAL_COMPOSITION,
                            TemporalComposition.NONSTATIONARY_COCYCLE,
                            TemporalComposition.STATIONARY_SEMIGROUP,
                        ),
                    ),
                    key=lambda value: value.axis,
                )
            ),
        ),
        StructuralClassDefinition(
            class_id="nonlinear-commutative",
            prototype=StructuralClassPrototype.NONLINEAR_COMMUTATIVE,
            criteria=tuple(
                sorted(
                    (
                        criterion(SignatureAxis.PORT_MATERIALITY, PortMateriality.MATERIAL),
                        criterion(
                            SignatureAxis.SEQUENTIAL_COMPOSITION,
                            SequentialComposition.COMMUTATOR_EQUIVALENT,
                        ),
                        criterion(
                            SignatureAxis.SIMULTANEOUS_COMPOSITION,
                            SimultaneousComposition.NONLINEAR_INTERACTION,
                        ),
                    ),
                    key=lambda value: value.axis,
                )
            ),
        ),
        StructuralClassDefinition(
            class_id="projected-or-unclosed",
            prototype=StructuralClassPrototype.PROJECTED_OR_UNCLOSED,
            criteria=(
                criterion(
                    SignatureAxis.ALGEBRA_CLOSURE,
                    AlgebraClosure.PROJECTED_OR_STATE_NONCLOSED,
                ),
            ),
        ),
        StructuralClassDefinition(
            class_id="response-null",
            prototype=StructuralClassPrototype.RESPONSE_NULL,
            criteria=(criterion(SignatureAxis.PORT_MATERIALITY, PortMateriality.NULL),),
        ),
        StructuralClassDefinition(
            class_id="temporal-cocycle-composition-null-at-resolution",
            prototype=(StructuralClassPrototype.TEMPORAL_COCYCLE__COMPOSITION_NULL_AT_RESOLUTION),
            criteria=tuple(
                sorted(
                    (
                        criterion(
                            SignatureAxis.PORT_MATERIALITY,
                            PortMateriality.MIXED,
                            PortMateriality.NULL,
                        ),
                        criterion(
                            SignatureAxis.SEQUENTIAL_COMPOSITION,
                            SequentialComposition.BELOW_RESOLUTION,
                        ),
                        criterion(
                            SignatureAxis.SIMULTANEOUS_COMPOSITION,
                            SimultaneousComposition.BELOW_RESOLUTION,
                        ),
                        criterion(
                            SignatureAxis.TEMPORAL_COMPOSITION,
                            TemporalComposition.NONSTATIONARY_COCYCLE,
                            TemporalComposition.STATIONARY_SEMIGROUP,
                        ),
                    ),
                    key=lambda value: value.axis,
                )
            ),
        ),
    )
    return tuple(sorted(definitions, key=lambda value: value.class_id))


@dataclass(frozen=True, slots=True)
class StructuralClassAssignment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/structural-class-assignment'

    assignment_id: str
    class_definition_id: str
    signature_id: str
    disposition: StructuralClassDisposition
    promotion_level: ClassPromotionLevel
    matched_axes: tuple[SignatureAxis, ...]
    opposed_axes: tuple[SignatureAxis, ...]
    supporting_substrate_ids: tuple[str, ...]
    supporting_world_ids: tuple[str, ...]
    prospective_confirmation_ids: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assignment_id", self.assignment_id),
            ("class_definition_id", self.class_definition_id),
            ("signature_id", self.signature_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.matched_axes, field_name="matched_axes")
        require_sorted_unique_strings(self.opposed_axes, field_name="opposed_axes")
        for name, values in (
            ("supporting_substrate_ids", self.supporting_substrate_ids),
            ("supporting_world_ids", self.supporting_world_ids),
            ("prospective_confirmation_ids", self.prospective_confirmation_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
            for value in values:
                validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        _validate_reason_codes(self.reason_codes)
        if set(self.matched_axes) & set(self.opposed_axes):
            raise ValueError("a signature axis cannot both match and oppose a class")
        if self.disposition is StructuralClassDisposition.MATCHED:
            if self.opposed_axes or not self.matched_axes or not self.evidence_link_ids:
                raise ValueError("matched class assignment requires matched evidence only")
        elif self.disposition is StructuralClassDisposition.OPPOSED:
            if not self.opposed_axes or not self.evidence_link_ids:
                raise ValueError("opposed class assignment requires opposing evidence")
        elif not self.reason_codes:
            raise ValueError("unevaluable class assignment requires reason codes")
        self._validate_promotion_evidence()

    def _validate_promotion_evidence(self) -> None:
        if not self.supporting_substrate_ids or not self.supporting_world_ids:
            raise ValueError("class assignment requires explicit substrate and evidence worlds")
        if self.promotion_level in {
            ClassPromotionLevel.LOCAL_SIGNATURE,
            ClassPromotionLevel.SUBSTRATE_CLASS,
        }:
            if self.prospective_confirmation_ids:
                raise ValueError("local/substrate class cannot claim out-of-family confirmation")
            return
        if len(self.supporting_substrate_ids) < 2 or len(self.supporting_world_ids) < 2:
            raise ValueError("structural class candidate requires a second substrate and world")
        if self.promotion_level is ClassPromotionLevel.STRUCTURAL_CLASS_CANDIDATE:
            if self.prospective_confirmation_ids:
                raise ValueError("candidate status precedes out-of-family confirmation")
            return
        if (
            len(self.supporting_substrate_ids) < 3
            or len(self.supporting_world_ids) < 3
            or not self.prospective_confirmation_ids
        ):
            raise ValueError("universality class requires preregistered out-of-family confirmation")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraCheckResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/response-algebra-check-result'

    check_id: str
    kind: ResponseAlgebraCheckKind
    status: ObligationStatus
    decisive: bool
    metrics: tuple[NamedDecimal, ...]
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.check_id, field_name="check_id")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        _validate_reason_codes(self.reason_codes)
        if self.status is ObligationStatus.SATISFIED:
            if not self.evidence_link_ids or self.reason_codes:
                raise ValueError("satisfied algebra check requires evidence and no reasons")
        elif self.status is ObligationStatus.NOT_APPLICABLE:
            if self.reason_codes:
                raise ValueError("not-applicable algebra check cannot carry failure reasons")
        elif not self.reason_codes:
            raise ValueError("unresolved algebra check requires reason codes")

    @property
    def passed(self) -> bool:
        return self.status in {ObligationStatus.SATISFIED, ObligationStatus.NOT_APPLICABLE}


_REQUIRED_CHECKS = frozenset(ResponseAlgebraCheckKind)


@dataclass(frozen=True, slots=True)
class ResponseAlgebraIdentificationResult(CanonicalRecord):
    "Terminal response-algebra result for one exact relation, with at most local-law evidence."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/response-algebra-identification-result'

    result_id: str
    dataset: ObjectIdentity
    config: ObjectIdentity
    system_id: str
    world_id: str
    relation: RelationalIdentity
    method_key: str
    method_version: str
    evaluator: ExecutableReference
    checks: tuple[ResponseAlgebraCheckResult, ...]
    temporal_assessments: tuple[TemporalCompositionAssessment, ...]
    action_assessments: tuple[ActionCompositionAssessment, ...]
    signature: ResponseAlgebraSignature
    class_assignment: StructuralClassAssignment | None
    operational_status: OperationalStatus
    readiness_status: ReadinessStatus
    scientific_status: ScientificStatus
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    evidence_links: tuple[EvidenceLink, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("result_id", self.result_id),
            ("system_id", self.system_id),
            ("world_id", self.world_id),
            ("method_key", self.method_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.method_version)
        require_sorted_unique_ids(self.checks, attribute="check_id", field_name="checks")
        if {check.kind for check in self.checks} != _REQUIRED_CHECKS:
            raise ValueError("response-algebra result lacks the complete check family")
        require_sorted_unique_ids(
            self.temporal_assessments,
            attribute="assessment_id",
            field_name="temporal_assessments",
        )
        require_sorted_unique_ids(
            self.action_assessments,
            attribute="assessment_id",
            field_name="action_assessments",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        _validate_reason_codes(self.reason_codes)
        if self.evidence_ceiling in {
            EvidenceCeiling.ADMISSION,
            EvidenceCeiling.CONTROLLER_USE,
        }:
            raise ValueError("response algebra is capped at local law")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("response-algebra visibility cannot be lowered")
        if not self.visibility_ceiling.is_promotable:
            if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
                raise ValueError("outcome-visible algebra evidence must be non-promotable")
        if self.class_assignment is not None:
            if self.class_assignment.signature_id != self.signature.signature_id:
                raise ValueError("class assignment and response-algebra signature differ")
            if (
                not self.visibility_ceiling.is_promotable
                and self.class_assignment.promotion_level is not ClassPromotionLevel.LOCAL_SIGNATURE
            ):
                raise ValueError("non-promotable evidence cannot advance a structural class")
        all_checks_pass = all(check.passed for check in self.checks)
        if self.scientific_status is ScientificStatus.SUPPORTED:
            if (
                self.operational_status is not OperationalStatus.SUCCEEDED
                or self.readiness_status is not ReadinessStatus.READY
                or not all_checks_pass
                or not self.evidence_links
                or self.reason_codes
            ):
                raise ValueError("supported response algebra bypasses a terminal obligation")
        elif not self.reason_codes:
            raise ValueError("non-supported response algebra requires terminal reasons")
