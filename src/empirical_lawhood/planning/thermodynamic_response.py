"Frozen vocabulary for thermodynamic response categories."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    require_unique_ids,
    validate_decimal,
    validate_nonempty,
    validate_schema,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.kernel.thermodynamic_response import ChronologyConvention, CompositeSignatureAxis, FiniteWordMode, FiniteWord, PreparedResponseObject, ThermodynamicPromotionLevel, ThermodynamicResponseTerminalStatus, ThermodynamicSignConvention


THERMODYNAMIC_RESPONSE_CONFORMANCE_CASE_TOKENS = tuple(
    f"thermodynamic-reference-{index:02d}" for index in range(16)
)


PRESERVED_RESPONSE_ALGEBRA_SCHEMA_IDS = tuple(
    sorted(
        {
            'empirical-lawhood/kernel/action-composition-assessment',
            'empirical-lawhood/kernel/action-letter',
            'empirical-lawhood/kernel/action-stage-value',
            'empirical-lawhood/kernel/letter-action-word',
            'empirical-lawhood/kernel/response-algebra-check-result',
            'empirical-lawhood/kernel/response-algebra-identification-result',
            'empirical-lawhood/kernel/signature-criterion',
            'empirical-lawhood/kernel/response-algebra-signature',
            'empirical-lawhood/kernel/structural-class-assignment',
            'empirical-lawhood/kernel/structural-class-definition',
            'empirical-lawhood/kernel/temporal-composition-assessment',
            'empirical-lawhood/methods/frozen-class-assessment',
            'empirical-lawhood/methods/affine-transition',
            'empirical-lawhood/methods/response-algebra-class-card',
            'empirical-lawhood/methods/delivered-generator-record',
            'empirical-lawhood/methods/delivered-word-record',
            'empirical-lawhood/methods/response-algebra-development-freeze',
            'empirical-lawhood/methods/discrete-transition',
            'empirical-lawhood/methods/response-algebra-evaluation-output',
            'empirical-lawhood/methods/generator-transition',
            'empirical-lawhood/methods/response-algebra-method-input',
            'empirical-lawhood/methods/response-algebra-method-config',
            'empirical-lawhood/methods/response-algebra-method-result',
            'empirical-lawhood/methods/response-algebra-method-selection',
            'empirical-lawhood/methods/response-algebra-numerical-qualification',
            'empirical-lawhood/methods/receiver-equivalence-criterion',
            'empirical-lawhood/methods/state-vector',
            'empirical-lawhood/methods/state-view-spec',
            'empirical-lawhood/methods/response-algebra-unit-contrast-table',
            'empirical-lawhood/methods/response-algebra-unit-contrast',
            'empirical-lawhood/methods/word-response',
            'empirical-lawhood/planning/action-alphabet-spec',
            'empirical-lawhood/planning/action-word-design-spec',
            'empirical-lawhood/planning/compiled-response-algebra-conformance',
            'empirical-lawhood/planning/planned-action-letter',
            'empirical-lawhood/planning/planned-action-word',
            'empirical-lawhood/planning/response-algebra-conformance-authorization',
            'empirical-lawhood/planning/response-algebra-conformance-execution-package',
            'empirical-lawhood/planning/response-algebra-conformance-spec',
            'empirical-lawhood/planning/equivalence-spec',
            'empirical-lawhood/planning/response-algebra-hypothesis-freeze',
            'empirical-lawhood/planning/multiplicity-spec',
            'empirical-lawhood/planning/response-algebra-protocol-spec',
            'empirical-lawhood/planning/nonadaptive-stopping-spec',
            'empirical-lawhood/planning/structural-class-hypothesis',
        }
    )
)


THERMODYNAMIC_RESPONSE_ADDITIONAL_SCHEMA_IDS = tuple(
    sorted(
        {
            'empirical-lawhood/kernel/action-delivery',
            'empirical-lawhood/kernel/action-interval',
            'empirical-lawhood/kernel/action-stage-observation',
            'empirical-lawhood/kernel/balance-closure-assessment',
            'empirical-lawhood/kernel/composite-structural-class-assessment',
            'empirical-lawhood/kernel/composite-structural-class-definition',
            'empirical-lawhood/kernel/entropy-term-spec',
            'empirical-lawhood/kernel/finite-response-signature',
            'empirical-lawhood/kernel/finite-word',
            'empirical-lawhood/kernel/delivered-dose-component',
            'empirical-lawhood/kernel/ontology-compatibility-witness',
            'empirical-lawhood/kernel/prepared-response-object',
            'empirical-lawhood/kernel/return-qualification',
            'empirical-lawhood/kernel/stored-energy-term-spec',
            'empirical-lawhood/kernel/structural-transport-witness',
            'empirical-lawhood/kernel/thermodynamic-ledger-observation',
            'empirical-lawhood/kernel/thermodynamic-response-signature',
            'empirical-lawhood/kernel/thermodynamic-role-binding',
            'empirical-lawhood/kernel/thermodynamic-term-spec',
            'empirical-lawhood/kernel/scalar-dose-action-word-applicability-check',
            'empirical-lawhood/kernel/scalar-dose-action-word-projection-witness',
            'empirical-lawhood/methods/thermodynamic-response-method-input',
            'empirical-lawhood/methods/thermodynamic-response-method-config',
            'empirical-lawhood/methods/thermodynamic-response-method-result',
            'empirical-lawhood/methods/finite-trajectory-panel',
            'empirical-lawhood/physical/thermodynamic-response/action-journal-entry',
            'empirical-lawhood/physical/thermodynamic-response/external-authority-evidence',
            'empirical-lawhood/physical/thermodynamic-response/physical-episode-bundle',
            'empirical-lawhood/physical/thermodynamic-response/physical-episode-publication',
            'empirical-lawhood/planning/finite-contrast-spec',
            'empirical-lawhood/planning/finite-word-family-spec',
            'empirical-lawhood/planning/thermodynamic-response-conformance-spec',
            'empirical-lawhood/planning/word-role-map',
        }
    )
)


class WordRole(StrEnum):
    IDENTITY = "IDENTITY"
    SHAM = "SHAM"
    SINGLE = "SINGLE"
    REPEAT = "REPEAT"
    ORDER = "ORDER"
    OVERLAP = "OVERLAP"
    LONGER_WORD = "LONGER_WORD"
    CYCLE = "CYCLE"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True, slots=True)
class WordRoleMap(CanonicalRecord):
    """Closed, many-valued role assignments over one exact word inventory."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/word-role-map'

    role_map_id: str
    family_id: str
    assignments: tuple[tuple[str, tuple[WordRole, ...]], ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.role_map_id, field_name="role_map_id")
        validate_stable_id(self.family_id, field_name="family_id")
        word_ids = tuple(word_id for word_id, _roles in self.assignments)
        require_sorted_unique_strings(word_ids, field_name="assignments", allow_empty=False)
        for word_id, roles in self.assignments:
            validate_stable_id(word_id, field_name="assignments.word_id")
            if not roles or tuple(sorted(set(roles), key=lambda value: value.value)) != roles:
                raise ValueError("each word requires sorted, unique declared roles")


@dataclass(frozen=True, slots=True)
class FiniteWordFamilySpec(CanonicalRecord):
    """One ordered, arbitrary finite-word inventory with declared action pairs."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-word-family-spec'

    family_id: str
    relation_id: str
    objects: tuple[PreparedResponseObject, ...]
    words: tuple[FiniteWord, ...]
    declared_action_pairs: tuple[tuple[str, str], ...]
    role_map: WordRoleMap
    receiver_ids: tuple[str, ...]
    horizon_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.family_id, field_name="family_id")
        validate_stable_id(self.relation_id, field_name="relation_id")
        require_sorted_unique_ids(self.objects, attribute="object_id", field_name="objects")
        if not self.objects:
            raise ValueError("finite-word family requires prepared response objects")
        object_by_id = {value.object_id: value for value in self.objects}
        if any(value.relation_id != self.relation_id for value in self.objects):
            raise ValueError("finite-word family objects belong to another relation")
        require_unique_ids(self.words, attribute="word_id", field_name="words")
        if not self.words:
            raise ValueError("finite-word family must not be empty")
        if len(self.words) > 1024:
            raise ValueError("finite-word family exceeds its bounded size")
        identity_words = tuple(word for word in self.words if word.mode is FiniteWordMode.IDENTITY)
        if len(identity_words) != 1:
            raise ValueError("finite-word family requires exactly one identity word")
        if self.role_map.family_id != self.family_id:
            raise ValueError("word-role map belongs to another finite-word family")
        inventory = {word.word_id for word in self.words}
        mapped = {word_id for word_id, _roles in self.role_map.assignments}
        if mapped != inventory:
            raise ValueError("word-role map must cover every and only family word")
        identity_roles = {
            word_id for word_id, roles in self.role_map.assignments if WordRole.IDENTITY in roles
        }
        if identity_roles != {identity_words[0].word_id}:
            raise ValueError("identity role must identify exactly the identity word")
        for word in self.words:
            try:
                prefix_objects = tuple(object_by_id[value] for value in word.prefix_object_ids)
            except KeyError as error:
                raise ValueError("finite word references an undeclared prepared object") from error
            clocks = {value.clock_id for value in prefix_objects}
            if len(clocks) != 1:
                raise ValueError("finite-word objects require one mapped clock")
            if any(
                later.clock_coordinate <= earlier.clock_coordinate
                for earlier, later in zip(prefix_objects, prefix_objects[1:])
            ):
                raise ValueError("finite-word object chronology must increase")
            elapsed = prefix_objects[-1].clock_coordinate - prefix_objects[0].clock_coordinate
            if elapsed != word.elapsed_time:
                raise ValueError("finite-word elapsed time differs from its object chronology")
            if word.mode is FiniteWordMode.IDENTITY:
                continue
            if word.mode is FiniteWordMode.SEQUENTIAL:
                delivery_prefixes = prefix_objects[:-1]
                delivery_targets = prefix_objects[1:]
            else:
                delivery_prefixes = (prefix_objects[0],) * len(word.deliveries)
                delivery_targets = (prefix_objects[-1],) * len(word.deliveries)
            for delivery, prefix, target in zip(
                word.deliveries, delivery_prefixes, delivery_targets, strict=True
            ):
                if delivery.preparation_id != prefix.preparation_id:
                    raise ValueError("delivery preparation differs from its source object")
                interval = delivery.realized_interval
                if interval.clock_id != prefix.clock_id:
                    raise ValueError("delivery interval differs from its object clock")
                if (
                    interval.start < prefix.clock_coordinate
                    or interval.end > target.clock_coordinate
                ):
                    raise ValueError("delivery interval lies outside its source-target segment")
        for name, values in (
            ("receiver_ids", self.receiver_ids),
            ("horizon_ids", self.horizon_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            for value in values:
                validate_stable_id(value, field_name=name)
        seen_pairs: set[tuple[str, str]] = set()
        available_letters = {
            delivery.letter_id for word in self.words for delivery in word.deliveries
        }
        for pair in self.declared_action_pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("declared action pair requires two distinct letters")
            for letter_id in pair:
                validate_stable_id(letter_id, field_name="declared_action_pairs")
            if not set(pair) <= available_letters:
                raise ValueError("declared action pair is outside the delivered word inventory")
            if pair in seen_pairs:
                raise ValueError("declared action pairs must be unique")
            seen_pairs.add(pair)
        if tuple(sorted(seen_pairs)) != self.declared_action_pairs:
            raise ValueError("declared action pairs must have canonical ordering")


@dataclass(frozen=True, slots=True)
class FiniteContrastSpec(CanonicalRecord):
    """A static linear functional over declared words, receiver and horizon."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/finite-contrast-spec'

    contrast_id: str
    family_id: str
    relation_id: str
    estimand_id: str
    receiver_id: str
    horizon_id: str
    coefficients: tuple[tuple[str, Decimal], ...]
    coordinate_ids: tuple[str, ...]
    response_units: tuple[str, ...]
    zero_sum_required: bool

    def __post_init__(self) -> None:
        for name, identifier in (
            ("contrast_id", self.contrast_id),
            ("family_id", self.family_id),
            ("relation_id", self.relation_id),
            ("estimand_id", self.estimand_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
        ):
            validate_stable_id(identifier, field_name=name)
        word_ids = tuple(word_id for word_id, _coefficient in self.coefficients)
        require_sorted_unique_strings(word_ids, field_name="coefficients", allow_empty=False)
        for word_id, coefficient in self.coefficients:
            validate_stable_id(word_id, field_name="coefficients.word_id")
            validate_decimal(coefficient, field_name="coefficients.coefficient")
            if coefficient == 0:
                raise ValueError("zero finite-contrast coefficients must be omitted")
        if (
            self.zero_sum_required
            and sum(
                (coefficient for _word_id, coefficient in self.coefficients),
                start=Decimal("0"),
            )
            != 0
        ):
            raise ValueError("declared zero-sum finite contrast does not sum to zero")
        if not self.coordinate_ids or len(set(self.coordinate_ids)) != len(self.coordinate_ids):
            raise ValueError("finite contrast coordinates must be nonempty and unique")
        for coordinate_id in self.coordinate_ids:
            validate_stable_id(coordinate_id, field_name="coordinate_ids")
        if len(self.response_units) != len(self.coordinate_ids):
            raise ValueError("finite contrast coordinates and response units differ")
        for response_unit in self.response_units:
            validate_nonempty(response_unit, field_name="response_units")


@dataclass(frozen=True, slots=True)
class ThermodynamicResponseVocabulary(CanonicalRecord):
    """One immutable catalogue of preserved and additive contract identities."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/thermodynamic-response-vocabulary'

    vocabulary_id: str
    preserved_response_algebra_schema_ids: tuple[str, ...]
    additional_schema_ids: tuple[str, ...]
    chronology_conventions: tuple[ChronologyConvention, ...]
    sign_conventions: tuple[ThermodynamicSignConvention, ...]
    composite_signature_axes: tuple[CompositeSignatureAxis, ...]
    terminal_statuses: tuple[ThermodynamicResponseTerminalStatus, ...]
    promotion_levels: tuple[ThermodynamicPromotionLevel, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.vocabulary_id, field_name="vocabulary_id")
        for name, values in (
            ("preserved_response_algebra_schema_ids", self.preserved_response_algebra_schema_ids),
            ("additional_schema_ids", self.additional_schema_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if self.preserved_response_algebra_schema_ids != PRESERVED_RESPONSE_ALGEBRA_SCHEMA_IDS:
            raise ValueError("preserved response-algebra schema catalogue differs")
        if self.additional_schema_ids != THERMODYNAMIC_RESPONSE_ADDITIONAL_SCHEMA_IDS:
            raise ValueError("thermodynamic-response schema catalogue differs")
        if self.chronology_conventions != tuple(ChronologyConvention):
            raise ValueError("chronology convention vocabulary differs")
        if self.sign_conventions != tuple(ThermodynamicSignConvention):
            raise ValueError("thermodynamic sign vocabulary differs")
        if self.composite_signature_axes != tuple(CompositeSignatureAxis):
            raise ValueError("composite signature-axis vocabulary differs")
        if self.terminal_statuses != tuple(ThermodynamicResponseTerminalStatus):
            raise ValueError("terminal-status vocabulary differs")
        if self.promotion_levels != tuple(ThermodynamicPromotionLevel):
            raise ValueError("promotion-level vocabulary differs")


@dataclass(frozen=True, slots=True)
class ThermodynamicResponseConformanceSpec(CanonicalRecord):
    """Oracle-separated thresholds and identities for the sixteen reference cases."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/thermodynamic-response-conformance-spec'

    conformance_id: str
    specification_version: str
    method_key: str
    method_version: str
    evaluator_key: str
    evaluator_version: str
    case_tokens: tuple[str, ...]
    independent_units_per_case: int
    construction_seed: int
    time_coordinates: tuple[Decimal, ...]
    time_unit: str
    coordinate_ids: tuple[str, ...]
    native_units: tuple[str, ...]
    numerical_floors: tuple[NamedDecimal, ...]
    decision_thresholds: tuple[NamedDecimal, ...]
    truth_blind_input_schema: str
    privileged_oracle_schema: str
    method_result_schema: str
    maximum_false_noncommutativity: int
    maximum_false_entropy_production: int
    maximum_false_cycle_qualification: int
    privileged_truth_outside_method_input: bool
    prohibited_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.conformance_id, field_name="conformance_id")
        validate_semantic_version(self.specification_version)
        validate_semantic_version(self.method_version)
        validate_semantic_version(self.evaluator_version)
        for name, value in (
            ("method_key", self.method_key),
            ("evaluator_key", self.evaluator_key),
        ):
            validate_stable_id(value, field_name=name)
        if self.case_tokens != THERMODYNAMIC_RESPONSE_CONFORMANCE_CASE_TOKENS:
            raise ValueError("thermodynamic conformance case-token inventory differs")
        if self.independent_units_per_case != 8:
            raise ValueError("truth-known conformance requires eight physical units per case")
        if not 0 <= self.construction_seed <= 2**63 - 1:
            raise ValueError("truth-known construction seed is outside its bound")
        if len(self.time_coordinates) < 4:
            raise ValueError("truth-known conformance requires at least four time anchors")
        for time_coordinate in self.time_coordinates:
            validate_decimal(
                time_coordinate,
                field_name="time_coordinates",
                minimum=Decimal("0"),
            )
        if any(
            later <= earlier
            for earlier, later in zip(self.time_coordinates, self.time_coordinates[1:])
        ):
            raise ValueError("truth-known time coordinates must strictly increase")
        validate_nonempty(self.time_unit, field_name="time_unit")
        if not self.coordinate_ids or len(set(self.coordinate_ids)) != len(self.coordinate_ids):
            raise ValueError("truth-known coordinates must be nonempty and unique")
        for coordinate_id in self.coordinate_ids:
            validate_stable_id(coordinate_id, field_name="coordinate_ids")
        if len(self.native_units) != len(self.coordinate_ids):
            raise ValueError("truth-known coordinate identities and units differ")
        for native_unit in self.native_units:
            validate_nonempty(native_unit, field_name="native_units")
        for values, name in (
            (self.numerical_floors, "numerical_floors"),
            (self.decision_thresholds, "decision_thresholds"),
        ):
            require_sorted_unique_ids(values, attribute="value_id", field_name=name)
            if not values:
                raise ValueError(f"{name} must not be empty")
        floor_by_id = {value.value_id: value.unit for value in self.numerical_floors}
        if set(floor_by_id) != set(self.coordinate_ids):
            raise ValueError("truth-known numerical floors differ from coordinates")
        if any(
            floor_by_id[coordinate_id] != unit
            for coordinate_id, unit in zip(
                self.coordinate_ids,
                self.native_units,
                strict=True,
            )
        ):
            raise ValueError("truth-known numerical-floor native units differ")
        required_threshold_ids = {
            "balance-closure",
            "controlled-order",
            "cocycle",
            "naturality",
            "return-bath",
            "return-state",
            "stationarity",
        }
        if {value.value_id for value in self.decision_thresholds} != required_threshold_ids:
            raise ValueError("truth-known decision-threshold inventory differs")
        for name, value in (
            ("truth_blind_input_schema", self.truth_blind_input_schema),
            ("privileged_oracle_schema", self.privileged_oracle_schema),
            ("method_result_schema", self.method_result_schema),
        ):
            validate_schema(value)
        if any(
            value != 0
            for value in (
                self.maximum_false_noncommutativity,
                self.maximum_false_entropy_production,
                self.maximum_false_cycle_qualification,
            )
        ):
            raise ValueError("truth-known conformance permits no false scientific promotion")
        if not self.privileged_truth_outside_method_input:
            raise ValueError("privileged truth must remain outside method input")
        require_sorted_unique_strings(
            self.prohibited_component_ids,
            field_name="prohibited_component_ids",
            allow_empty=False,
        )
        if not {"llm", "rl"} <= set(self.prohibited_component_ids):
            raise ValueError("truth-known conformance prohibits LLM and RL components")


def reference_thermodynamic_response_conformance_spec() -> ThermodynamicResponseConformanceSpec:
    return ThermodynamicResponseConformanceSpec(
        conformance_id="thermodynamic-response-conformance",
        specification_version="1.0.0",
        method_key="thermodynamic-response.finite-analysis",
        method_version="1.0.0",
        evaluator_key="thermodynamic-response.sealed-evaluator",
        evaluator_version="1.0.0",
        case_tokens=THERMODYNAMIC_RESPONSE_CONFORMANCE_CASE_TOKENS,
        independent_units_per_case=8,
        construction_seed=20260719,
        time_coordinates=(Decimal("0"), Decimal("1"), Decimal("2"), Decimal("3")),
        time_unit="s",
        coordinate_ids=("bath-state", "mode-state", "receiver-x", "receiver-y"),
        native_units=("K", "1", "K", "K"),
        numerical_floors=(
            NamedDecimal("bath-state", Decimal("0.05"), "K"),
            NamedDecimal("mode-state", Decimal("0.05"), "1"),
            NamedDecimal("receiver-x", Decimal("0.05"), "K"),
            NamedDecimal("receiver-y", Decimal("0.05"), "K"),
        ),
        decision_thresholds=(
            NamedDecimal("balance-closure", Decimal("0.1"), "J"),
            NamedDecimal("cocycle", Decimal("0.05"), "scaled-rms"),
            NamedDecimal("controlled-order", Decimal("0.1"), "floor-scaled-rms"),
            NamedDecimal("naturality", Decimal("0.000001"), "K"),
            NamedDecimal("return-bath", Decimal("0.1"), "K"),
            NamedDecimal("return-state", Decimal("0.1"), "K"),
            NamedDecimal("stationarity", Decimal("0.05"), "scaled-rms"),
        ),
        truth_blind_input_schema='empirical-lawhood/reference-worlds/thermodynamic-truth-blind-input',
        privileged_oracle_schema='empirical-lawhood/reference-worlds/privileged-thermodynamic-oracle',
        method_result_schema='empirical-lawhood/methods/thermodynamic-response-method-result',
        maximum_false_noncommutativity=0,
        maximum_false_entropy_production=0,
        maximum_false_cycle_qualification=0,
        privileged_truth_outside_method_input=True,
        prohibited_component_ids=("llm", "rl"),
    )


def thermodynamic_response_vocabulary() -> ThermodynamicResponseVocabulary:
    return ThermodynamicResponseVocabulary(
        vocabulary_id="thermodynamic-response-vocabulary",
        preserved_response_algebra_schema_ids=PRESERVED_RESPONSE_ALGEBRA_SCHEMA_IDS,
        additional_schema_ids=THERMODYNAMIC_RESPONSE_ADDITIONAL_SCHEMA_IDS,
        chronology_conventions=tuple(ChronologyConvention),
        sign_conventions=tuple(ThermodynamicSignConvention),
        composite_signature_axes=tuple(CompositeSignatureAxis),
        terminal_statuses=tuple(ThermodynamicResponseTerminalStatus),
        promotion_levels=tuple(ThermodynamicPromotionLevel),
    )


__all__ = [
    "FiniteContrastSpec",
    "FiniteWordFamilySpec",
    "PRESERVED_RESPONSE_ALGEBRA_SCHEMA_IDS",
    "THERMODYNAMIC_RESPONSE_ADDITIONAL_SCHEMA_IDS",
    "THERMODYNAMIC_RESPONSE_CONFORMANCE_CASE_TOKENS",
    "ThermodynamicResponseConformanceSpec",
    "ThermodynamicResponseVocabulary",
    "WordRole",
    "WordRoleMap",
    'thermodynamic_response_vocabulary',
    "reference_thermodynamic_response_conformance_spec",
]
