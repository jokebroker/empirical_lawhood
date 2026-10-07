"""Strict authoring records for bounded response-algebra experiments."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.response_algebra import (
    StructuralClassDefinition,
    prototype_class_definitions,
)
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.kernel.worlds import WorldKind


class WordExecutionMode(StrEnum):
    IDENTITY = "IDENTITY"
    SEQUENTIAL = "SEQUENTIAL"
    SIMULTANEOUS = "SIMULTANEOUS"


class MultiplicityKind(StrEnum):
    PRIMARY_PAIR = "PRIMARY_PAIR"
    FAMILY_WISE_ERROR = "FAMILY_WISE_ERROR"
    FALSE_DISCOVERY_RATE = "FALSE_DISCOVERY_RATE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class PlannedActionLetter(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/planned-action-letter'

    letter_id: str
    port_id: str
    requested_value: Decimal
    native_unit: str
    duration: Decimal
    duration_unit: str
    clock_id: str
    two_sided_partner_id: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("letter_id", self.letter_id),
            ("port_id", self.port_id),
            ("clock_id", self.clock_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.requested_value, field_name="requested_value")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_decimal(self.duration, field_name="duration", minimum=Decimal("0"))
        if self.duration == 0:
            raise ValueError("planned action duration must be positive")
        validate_nonempty(self.duration_unit, field_name="duration_unit")
        if self.two_sided_partner_id is not None:
            validate_stable_id(
                self.two_sided_partner_id,
                field_name="two_sided_partner_id",
            )
            if self.two_sided_partner_id == self.letter_id:
                raise ValueError("a two-sided action partner must be another letter")


@dataclass(frozen=True, slots=True)
class ActionAlphabetSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/action-alphabet-spec'

    alphabet_id: str
    letters: tuple[PlannedActionLetter, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.alphabet_id, field_name="alphabet_id")
        require_sorted_unique_ids(self.letters, attribute="letter_id", field_name="letters")
        if len(self.letters) < 2 or len(self.letters) > 32:
            raise ValueError("response-algebra alphabet size must be in [2, 32]")
        by_id = {letter.letter_id: letter for letter in self.letters}
        for letter in self.letters:
            partner_id = letter.two_sided_partner_id
            if partner_id is None:
                continue
            partner = by_id.get(partner_id)
            if partner is None or partner.two_sided_partner_id != letter.letter_id:
                raise ValueError("two-sided action partners must be reciprocal")
            if (
                partner.port_id != letter.port_id
                or partner.native_unit != letter.native_unit
                or partner.clock_id != letter.clock_id
                or partner.duration != letter.duration
            ):
                raise ValueError(
                    "two-sided action partners must share port, unit, clock and duration"
                )


@dataclass(frozen=True, slots=True)
class PlannedActionWord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/planned-action-word'

    word_id: str
    mode: WordExecutionMode
    letter_ids: tuple[str, ...]
    slot_offsets: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.word_id, field_name="word_id")
        if len(self.letter_ids) > 32:
            raise ValueError("planned action word exceeds its bounded length")
        if len(self.slot_offsets) != len(self.letter_ids):
            raise ValueError("each planned action occurrence requires one slot offset")
        for letter_id in self.letter_ids:
            validate_stable_id(letter_id, field_name="letter_ids")
        for offset in self.slot_offsets:
            validate_decimal(offset, field_name="slot_offsets", minimum=Decimal("0"))
        if self.mode is WordExecutionMode.IDENTITY and self.letter_ids:
            raise ValueError("identity word cannot contain action letters")
        if self.mode is WordExecutionMode.SEQUENTIAL and not self.letter_ids:
            raise ValueError("sequential word requires at least one action letter")
        if self.mode is WordExecutionMode.SEQUENTIAL:
            if any(
                later <= earlier for earlier, later in zip(self.slot_offsets, self.slot_offsets[1:])
            ):
                raise ValueError("sequential word slot offsets must be strictly increasing")
        if self.mode is WordExecutionMode.SIMULTANEOUS:
            if len(self.letter_ids) < 2:
                raise ValueError("simultaneous word requires at least two action letters")
            require_sorted_unique_strings(self.letter_ids, field_name="letter_ids")
            if len(set(self.slot_offsets)) != 1:
                raise ValueError("simultaneous word letters must share one slot offset")


@dataclass(frozen=True, slots=True)
class ActionWordDesignSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/action-word-design-spec'

    design_id: str
    alphabet: ActionAlphabetSpec
    words: tuple[PlannedActionWord, ...]
    primary_pair: tuple[str, str]
    receiver_ids: tuple[str, ...]
    horizon_ids: tuple[str, ...]
    development_unit_ids: tuple[str, ...]
    evaluation_unit_ids: tuple[str, ...]
    randomization_seed: int

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        require_sorted_unique_ids(self.words, attribute="word_id", field_name="words")
        if not self.words:
            raise ValueError("word design must not be empty")
        if type(self.randomization_seed) is not int or self.randomization_seed < 0:
            raise ValueError("randomization seed must be a nonnegative integer")
        for name, values in (
            ("receiver_ids", self.receiver_ids),
            ("horizon_ids", self.horizon_ids),
            ("development_unit_ids", self.development_unit_ids),
            ("evaluation_unit_ids", self.evaluation_unit_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            for value in values:
                validate_stable_id(value, field_name=name)
        if set(self.development_unit_ids) & set(self.evaluation_unit_ids):
            raise ValueError("development and evaluation units must be disjoint")
        if len(self.primary_pair) != 2 or self.primary_pair[0] == self.primary_pair[1]:
            raise ValueError("primary pair requires two distinct action letters")
        alphabet_ids = {letter.letter_id for letter in self.alphabet.letters}
        if not set(self.primary_pair) <= alphabet_ids:
            raise ValueError("primary pair is outside the action alphabet")
        letter_by_id = {letter.letter_id: letter for letter in self.alphabet.letters}
        if len({letter_by_id[value].port_id for value in self.primary_pair}) != 2:
            raise ValueError("primary pair requires two distinct native action ports")
        observed_keys: set[tuple[WordExecutionMode, tuple[str, ...], tuple[Decimal, ...]]] = set()
        for word in self.words:
            if not set(word.letter_ids) <= alphabet_ids:
                raise ValueError("planned word contains a letter outside its alphabet")
            key = (word.mode, word.letter_ids, word.slot_offsets)
            if key in observed_keys:
                raise ValueError("word design contains duplicate action semantics")
            observed_keys.add(key)
        identity_key = (WordExecutionMode.IDENTITY, (), ())
        if identity_key not in observed_keys:
            raise ValueError("word design requires an identity branch")
        sequential_prefixes = {
            (word.letter_ids, word.slot_offsets)
            for word in self.words
            if word.mode is WordExecutionMode.SEQUENTIAL
        }
        for word in self.words:
            if word.mode is WordExecutionMode.SEQUENTIAL:
                required = tuple(
                    (word.letter_ids[:index], word.slot_offsets[:index])
                    for index in range(1, len(word.letter_ids))
                )
            elif word.mode is WordExecutionMode.SIMULTANEOUS:
                required = tuple(
                    ((letter_id,), (offset,))
                    for letter_id, offset in zip(
                        word.letter_ids,
                        word.slot_offsets,
                        strict=True,
                    )
                )
            else:
                required = ()
            if any(prefix not in sequential_prefixes for prefix in required):
                raise ValueError("word design omits a required action prefix")
        self._validate_primary_pair_family(letter_by_id)

    def _validate_primary_pair_family(
        self,
        letter_by_id: dict[str, PlannedActionLetter],
    ) -> None:
        first, second = self.primary_pair
        sequential = {
            (word.letter_ids, word.slot_offsets)
            for word in self.words
            if word.mode is WordExecutionMode.SEQUENTIAL
        }
        simultaneous = {
            (word.letter_ids, word.slot_offsets)
            for word in self.words
            if word.mode is WordExecutionMode.SIMULTANEOUS
        }
        pair_words = [
            word
            for word in self.words
            if word.mode is WordExecutionMode.SEQUENTIAL
            and word.letter_ids in {(first, second), (second, first)}
        ]
        offset_pairs = {word.slot_offsets for word in pair_words}
        matched_slots = [
            offsets
            for offsets in offset_pairs
            if ((first, second), offsets) in sequential and ((second, first), offsets) in sequential
        ]
        if len(matched_slots) != 1:
            raise ValueError("primary AB/BA words require one exact matched two-slot schedule")
        early, late = matched_slots[0]
        required_sequential = {
            ((first,), (early,)),
            ((first,), (late,)),
            ((second,), (early,)),
            ((second,), (late,)),
            ((first, first), (early, late)),
            ((second, second), (early, late)),
        }
        if not required_sequential <= sequential:
            raise ValueError("primary pair lacks early/late singles or repeat words")
        ordered_simultaneous_letters = tuple(sorted((first, second)))
        simultaneous_offsets = (early, early)
        if (ordered_simultaneous_letters, simultaneous_offsets) not in simultaneous:
            raise ValueError("primary pair lacks its matched simultaneous word")
        first_duration = letter_by_id[first].duration
        second_duration = letter_by_id[second].duration
        if late < early + max(first_duration, second_duration):
            raise ValueError("primary pair temporal slots overlap")


@dataclass(frozen=True, slots=True)
class EquivalenceSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/equivalence-spec'

    equivalence_id: str
    receiver_id: str
    native_unit: str
    numerical_or_observation_floor_upper: Decimal
    equivalence_width: Decimal
    constituent_materiality_lower: Decimal
    confidence_level: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.equivalence_id, field_name="equivalence_id")
        validate_stable_id(self.receiver_id, field_name="receiver_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        for name, value in (
            ("numerical_or_observation_floor_upper", self.numerical_or_observation_floor_upper),
            ("equivalence_width", self.equivalence_width),
            ("constituent_materiality_lower", self.constituent_materiality_lower),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal("0"))
        if not (
            self.numerical_or_observation_floor_upper
            < self.equivalence_width
            <= self.constituent_materiality_lower
        ):
            raise ValueError(
                "equivalence requires floor < equivalence width <= materiality threshold"
            )
        validate_decimal(self.confidence_level, field_name="confidence_level")
        if not Decimal("0") < self.confidence_level < Decimal("1"):
            raise ValueError("confidence level must be strictly between zero and one")


@dataclass(frozen=True, slots=True)
class MultiplicitySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/multiplicity-spec'

    multiplicity_id: str
    kind: MultiplicityKind
    family_ids: tuple[str, ...]
    error_rate: Decimal | None

    def __post_init__(self) -> None:
        validate_stable_id(self.multiplicity_id, field_name="multiplicity_id")
        require_sorted_unique_strings(self.family_ids, field_name="family_ids")
        for family_id in self.family_ids:
            validate_stable_id(family_id, field_name="family_ids")
        if self.kind is MultiplicityKind.NOT_APPLICABLE:
            if self.family_ids or self.error_rate is not None:
                raise ValueError("not-applicable multiplicity cannot carry a family or rate")
            return
        if not self.family_ids or self.error_rate is None:
            raise ValueError("multiplicity control requires a family and exact error rate")
        validate_decimal(self.error_rate, field_name="error_rate")
        if not Decimal("0") < self.error_rate < Decimal("1"):
            raise ValueError("multiplicity error rate must be strictly between zero and one")


@dataclass(frozen=True, slots=True)
class NonadaptiveStoppingSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/nonadaptive-stopping-spec'

    stopping_id: str
    maximum_preparations: int
    maximum_words_per_preparation: int
    maximum_invalid_unit_fraction: Decimal
    maximum_retries: int
    adaptive: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.stopping_id, field_name="stopping_id")
        if not 1 <= self.maximum_preparations <= 100_000:
            raise ValueError("maximum preparations must be in [1, 100000]")
        if not 1 <= self.maximum_words_per_preparation <= 10_000:
            raise ValueError("maximum words per preparation must be in [1, 10000]")
        validate_decimal(
            self.maximum_invalid_unit_fraction,
            field_name="maximum_invalid_unit_fraction",
        )
        if not Decimal("0") <= self.maximum_invalid_unit_fraction < Decimal("1"):
            raise ValueError("maximum invalid-unit fraction must be in [0, 1)")
        if not 0 <= self.maximum_retries <= 3:
            raise ValueError("maximum retries must be in [0, 3]")
        if self.adaptive:
            raise ValueError("response-algebra stopping must be nonadaptive")


@dataclass(frozen=True, slots=True)
class StructuralClassHypothesis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/structural-class-hypothesis'

    hypothesis_id: str
    class_definition: StructuralClassDefinition
    primary: bool
    decisive_falsifier_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.hypothesis_id, field_name="hypothesis_id")
        require_sorted_unique_strings(
            self.decisive_falsifier_ids,
            field_name="decisive_falsifier_ids",
            allow_empty=False,
        )
        for falsifier_id in self.decisive_falsifier_ids:
            validate_stable_id(falsifier_id, field_name="decisive_falsifier_ids")


def response_algebra_prototype_catalogue_sha256() -> str:
    "Fingerprint the exact ordered structural-class prototype catalogue."

    return sha256(canonical_json_bytes(prototype_class_definitions())).hexdigest()


@dataclass(frozen=True, slots=True)
class ResponseAlgebraHypothesisFreeze(CanonicalRecord):
    """Pre-evaluation class nomination and shared-method freeze for one child."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/response-algebra-hypothesis-freeze'

    freeze_id: str
    substrate_id: str
    primary_hypothesis: StructuralClassHypothesis
    prototype_catalogue_sha256: str
    prototype_class_ids: tuple[str, ...]
    nominated_result_ids: tuple[str, ...]
    nomination_metrics: tuple[NamedDecimal, ...]
    allowed_method_ids: tuple[str, ...]
    selected_method_id: str
    evaluator_key: str
    interpretation_constraint_ids: tuple[str, ...]
    prohibited_component_ids: tuple[str, ...]
    child_specific_estimator_allowed: bool
    evaluation_authored: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("freeze_id", self.freeze_id),
            ("substrate_id", self.substrate_id),
            ("evaluator_key", self.evaluator_key),
        ):
            validate_stable_id(value, field_name=name)
        expected_definitions = prototype_class_definitions()
        expected_ids = tuple(value.class_id for value in expected_definitions)
        if self.prototype_catalogue_sha256 != response_algebra_prototype_catalogue_sha256():
            raise ValueError("response-algebra prototype catalogue fingerprint differs")
        if self.prototype_class_ids != expected_ids:
            raise ValueError("response-algebra prototype class identities differ")
        if not self.primary_hypothesis.primary:
            raise ValueError("hypothesis freeze requires a primary class hypothesis")
        if self.primary_hypothesis.class_definition.prototype not in {
            value.prototype for value in expected_definitions
        }:
            raise ValueError("primary hypothesis uses an unregistered class prototype")
        for name, values in (
            ("nominated_result_ids", self.nominated_result_ids),
            ("allowed_method_ids", self.allowed_method_ids),
            ("interpretation_constraint_ids", self.interpretation_constraint_ids),
            ("prohibited_component_ids", self.prohibited_component_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        for result_id in self.nominated_result_ids:
            validate_stable_id(result_id, field_name="nominated_result_ids")
        require_sorted_unique_ids(
            self.nomination_metrics,
            attribute="value_id",
            field_name="nomination_metrics",
        )
        if not self.nomination_metrics:
            raise ValueError("hypothesis freeze requires empirical nomination metrics")
        validate_nonempty(self.selected_method_id, field_name="selected_method_id")
        if self.selected_method_id not in self.allowed_method_ids:
            raise ValueError("selected method is outside the frozen bounded menu")
        if self.child_specific_estimator_allowed:
            raise ValueError("child-specific response-algebra estimator copies are forbidden")
        if self.evaluation_authored:
            raise ValueError("class hypothesis must freeze before evaluation authoring")
        if not {"live-actuation", "llm", "rl"} <= set(self.prohibited_component_ids):
            raise ValueError("hypothesis freeze must prohibit live actuation, LLMs and RL")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraProtocolSpec(CanonicalRecord):
    """One bounded, claim-bearing authoring root for a fresh child."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/response-algebra-protocol-spec'

    protocol_id: str
    relation_id: str
    method_key: str
    method_version: str
    evaluator_key: str
    word_design: ActionWordDesignSpec
    equivalence_specs: tuple[EquivalenceSpec, ...]
    class_hypotheses: tuple[StructuralClassHypothesis, ...]
    multiplicity: MultiplicitySpec
    stopping: NonadaptiveStoppingSpec
    resource_budget: ResourceBudget
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    prohibited_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("protocol_id", self.protocol_id),
            ("relation_id", self.relation_id),
            ("method_key", self.method_key),
            ("evaluator_key", self.evaluator_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.method_version)
        require_sorted_unique_ids(
            self.equivalence_specs,
            attribute="equivalence_id",
            field_name="equivalence_specs",
        )
        if not self.equivalence_specs:
            raise ValueError("response-algebra protocol requires native-unit equivalence specs")
        receiver_ids = set(self.word_design.receiver_ids)
        if {value.receiver_id for value in self.equivalence_specs} != receiver_ids:
            raise ValueError("equivalence specs must cover every and only declared receiver")
        require_sorted_unique_ids(
            self.class_hypotheses,
            attribute="hypothesis_id",
            field_name="class_hypotheses",
        )
        if not self.class_hypotheses:
            raise ValueError("response-algebra protocol requires a class hypothesis")
        if sum(value.primary for value in self.class_hypotheses) != 1:
            raise ValueError("response-algebra protocol requires exactly one primary hypothesis")
        if self.evidence_ceiling in {
            EvidenceCeiling.ADMISSION,
            EvidenceCeiling.CONTROLLER_USE,
        }:
            raise ValueError("response-algebra protocol cannot exceed local law")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("response-algebra protocol visibility cannot be lowered")
        if not self.visibility_ceiling.is_promotable:
            if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
                raise ValueError("outcome-visible protocol must be non-promotable")
        require_sorted_unique_strings(
            self.prohibited_component_ids,
            field_name="prohibited_component_ids",
            allow_empty=False,
        )
        required_prohibitions = {"llm", "live-actuation", "rl"}
        if not required_prohibitions <= set(self.prohibited_component_ids):
            raise ValueError("protocol must explicitly prohibit LLM, RL and live actuation")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraConformanceSpec(CanonicalRecord):
    """Frozen truth-known suite counts, sensitivity and error gates."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/response-algebra-conformance-spec'

    conformance_id: str
    case_tokens: tuple[str, ...]
    development_word_units_per_case: int
    evaluation_word_units_per_case: int
    development_temporal_units_per_case: int
    evaluation_temporal_units_per_case: int
    generator_units_per_split_per_case: int
    master_seed: int
    bootstrap_replicates: int
    bootstrap_seed_base: int
    floor_upper: Decimal
    equivalence_width: Decimal
    minimum_material_effect: Decimal
    maximum_affine_closure_error: Decimal
    maximum_stationarity_error: Decimal
    minimum_wrong_horizon_ratio: Decimal
    minimum_stochastic_commutator: Decimal
    maximum_dose_scaling_relative_error: Decimal
    minimum_required_passed_cases: int
    maximum_allowed_false_positives: int
    maximum_allowed_missed_material_cases: int
    prohibited_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.conformance_id, field_name="conformance_id")
        require_sorted_unique_strings(
            self.case_tokens,
            field_name="case_tokens",
            allow_empty=False,
        )
        for token in self.case_tokens:
            validate_stable_id(token, field_name="case_tokens")
        if len(self.case_tokens) != 10:
            raise ValueError("Response-algebra conformance requires exactly ten cases")
        for count_name, count in (
            ("development_word_units_per_case", self.development_word_units_per_case),
            ("evaluation_word_units_per_case", self.evaluation_word_units_per_case),
            ("development_temporal_units_per_case", self.development_temporal_units_per_case),
            ("evaluation_temporal_units_per_case", self.evaluation_temporal_units_per_case),
            ("generator_units_per_split_per_case", self.generator_units_per_split_per_case),
        ):
            if count < 8 or count > 10_000:
                raise ValueError(f"{count_name} must be in [8, 10000]")
        if self.master_seed < 0 or self.bootstrap_seed_base < 0:
            raise ValueError("conformance seeds must be nonnegative")
        if not 100 <= self.bootstrap_replicates <= 100_000:
            raise ValueError("bootstrap replicate count must be in [100, 100000]")
        for threshold_name, threshold in (
            ("floor_upper", self.floor_upper),
            ("equivalence_width", self.equivalence_width),
            ("minimum_material_effect", self.minimum_material_effect),
            ("maximum_affine_closure_error", self.maximum_affine_closure_error),
            ("maximum_stationarity_error", self.maximum_stationarity_error),
            ("minimum_wrong_horizon_ratio", self.minimum_wrong_horizon_ratio),
            ("minimum_stochastic_commutator", self.minimum_stochastic_commutator),
            ("maximum_dose_scaling_relative_error", self.maximum_dose_scaling_relative_error),
        ):
            validate_decimal(threshold, field_name=threshold_name, minimum=Decimal("0"))
        if not self.floor_upper < self.equivalence_width <= self.minimum_material_effect:
            raise ValueError("conformance requires floor < equivalence <= material effect")
        if not 1 <= self.minimum_required_passed_cases <= len(self.case_tokens):
            raise ValueError("minimum passed-case count is outside the frozen suite")
        for count_name, count in (
            ("maximum_allowed_false_positives", self.maximum_allowed_false_positives),
            (
                "maximum_allowed_missed_material_cases",
                self.maximum_allowed_missed_material_cases,
            ),
        ):
            if count < 0 or count > len(self.case_tokens):
                raise ValueError(f"{count_name} is outside the frozen suite")
        require_sorted_unique_strings(
            self.prohibited_component_ids,
            field_name="prohibited_component_ids",
            allow_empty=False,
        )
        if not {"llm", "rl"} <= set(self.prohibited_component_ids):
            raise ValueError("conformance suite must explicitly prohibit LLM and RL")


class ConformanceAuthorizationDecision(StrEnum):
    APPROVED_NONACTUATING = "APPROVED_NONACTUATING"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"


@dataclass(frozen=True, slots=True)
class ResponseAlgebraConformanceExecutionPackage(CanonicalRecord):
    """Scoped, outcome-blind execution package for the truth-known method gate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/response-algebra-conformance-execution-package'

    package_id: str
    run_id: str
    conformance_spec: ResponseAlgebraConformanceSpec
    method_key: str
    method_version: str
    capability_registry_sha256: str
    implementation_commit: str
    implementation_sources: tuple[ArtifactIdentity, ...]
    authority_action: AuthorityAction
    world_kind: WorldKind
    source_access: SourceAccessClass
    requested_scope_id: str
    resource_budget: ResourceBudget
    storage_root_id: str
    external_relative_root: str
    prohibited_component_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("package_id", self.package_id),
            ("run_id", self.run_id),
            ("method_key", self.method_key),
            ("requested_scope_id", self.requested_scope_id),
            ("storage_root_id", self.storage_root_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.method_version)
        if self.method_key != "response-algebra.direct-affine-finite":
            raise ValueError("conformance package selects another method")
        if self.method_version != "1.0.0":
            raise ValueError("conformance package selects an unsupported method version")
        for name, digest in (("capability_registry_sha256", self.capability_registry_sha256),):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if len(self.implementation_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.implementation_commit
        ):
            raise ValueError("implementation commit must be a lowercase Git SHA-1")
        require_sorted_unique_ids(
            self.implementation_sources,
            attribute="artifact_id",
            field_name="implementation_sources",
        )
        if not self.implementation_sources:
            raise ValueError("conformance package requires exact implementation sources")
        if self.authority_action is not AuthorityAction.REFERENCE_WORLD_EXECUTION:
            raise ValueError("conformance package requires reference-world authority")
        if self.world_kind is not WorldKind.ANALYTIC_REFERENCE:
            raise ValueError("conformance package must remain an analytic reference world")
        if self.source_access is not SourceAccessClass.NONE:
            raise ValueError("truth-known conformance cannot request external source access")
        if self.storage_root_id != "operator-external-root":
            raise ValueError("conformance package names another production storage root")
        expected_root = f"runs/{self.run_id}"
        if self.external_relative_root != expected_root:
            raise ValueError("conformance external root must be derived from its run ID")
        require_sorted_unique_strings(
            self.prohibited_component_ids,
            field_name="prohibited_component_ids",
            allow_empty=False,
        )
        if self.prohibited_component_ids != self.conformance_spec.prohibited_component_ids:
            raise ValueError("package prohibitions differ from the frozen conformance spec")


@dataclass(frozen=True, slots=True)
class ResponseAlgebraConformanceAuthorization(CanonicalRecord):
    """Outcome-blind, nonactuating authorization evidence for one exact package."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/response-algebra-conformance-authorization'

    authorization_id: str
    package_id: str
    package_sha256: str
    policy_id: str
    policy_sha256: str
    decision: ConformanceAuthorizationDecision
    passed_gate_ids: tuple[str, ...]
    failed_gate_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    proposer_id: str
    approver_id: str
    decided_at_utc: str
    implementation_commit: str
    outcome_access: OutcomeAccess
    plan_mutated: bool
    grants_claim_promotion: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("authorization_id", self.authorization_id),
            ("package_id", self.package_id),
            ("policy_id", self.policy_id),
            ("proposer_id", self.proposer_id),
            ("approver_id", self.approver_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, digest in (
            ("package_sha256", self.package_sha256),
            ("policy_sha256", self.policy_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        require_sorted_unique_strings(self.passed_gate_ids, field_name="passed_gate_ids")
        require_sorted_unique_strings(self.failed_gate_ids, field_name="failed_gate_ids")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if set(self.passed_gate_ids) & set(self.failed_gate_ids):
            raise ValueError("a conformance authorization gate cannot pass and fail")
        if self.proposer_id == self.approver_id:
            raise ValueError("conformance authorization forbids self-approval")
        parse_utc_timestamp(self.decided_at_utc, field_name="decided_at_utc")
        if len(self.implementation_commit) != 40 or any(
            character not in "0123456789abcdef" for character in self.implementation_commit
        ):
            raise ValueError("authorization implementation commit must be a Git SHA-1")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("conformance authorization must remain outcome-blind")
        if self.plan_mutated or self.grants_claim_promotion:
            raise ValueError("conformance authorization cannot edit science or promote claims")
        if self.decision is ConformanceAuthorizationDecision.APPROVED_NONACTUATING:
            if self.failed_gate_ids:
                raise ValueError("approved conformance authorization cannot retain failed gates")
        elif not self.failed_gate_ids:
            raise ValueError("authority-required conformance authorization needs failed gates")


@dataclass(frozen=True, slots=True)
class CompiledResponseAlgebraConformance(CanonicalRecord):
    """Deterministic pre-execution preview of the bounded reference task graph."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/compiled-response-algebra-conformance'

    compiled_id: str
    package_id: str
    package_sha256: str
    authorization_id: str
    authorization_sha256: str
    task_ids: tuple[str, ...]
    case_tokens: tuple[str, ...]
    expected_truth_blind_invocations: int
    expected_privileged_oracles: int
    expected_case_scores: int
    maximum_output_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("compiled_id", self.compiled_id),
            ("package_id", self.package_id),
            ("authorization_id", self.authorization_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, digest in (
            ("package_sha256", self.package_sha256),
            ("authorization_sha256", self.authorization_sha256),
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        require_sorted_unique_strings(self.task_ids, field_name="task_ids", allow_empty=False)
        require_sorted_unique_strings(self.case_tokens, field_name="case_tokens", allow_empty=False)
        expected = len(self.case_tokens)
        if (
            self.expected_truth_blind_invocations != expected
            or self.expected_privileged_oracles != expected
            or self.expected_case_scores != expected
        ):
            raise ValueError("compiled conformance cardinalities differ from its case set")
        if self.maximum_output_bytes <= 0:
            raise ValueError("compiled conformance requires a positive output byte ceiling")


def authorize_response_algebra_conformance(
    *,
    package: ResponseAlgebraConformanceExecutionPackage,
    policy: AuthorityPolicy,
    authorization_id: str,
    passed_gate_ids: tuple[str, ...],
    proposer_id: str,
    approver_id: str,
    decided_at_utc: str,
) -> ResponseAlgebraConformanceAuthorization:
    """Apply the existing pure authority policy to one exact conformance package."""

    parse_utc_timestamp(decided_at_utc, field_name="decided_at_utc")
    refusal_reasons = policy.refusal_reasons(
        action=package.authority_action,
        world_kind=package.world_kind,
        source_access=package.source_access,
        passed_gate_ids=frozenset(passed_gate_ids),
        requested_budget=package.resource_budget,
        requested_outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
        requested_scope_id=package.requested_scope_id,
        proposer_id=proposer_id,
        approver_id=approver_id,
        at_utc=decided_at_utc,
    )
    missing_required = set(policy.required_gate_ids) - set(passed_gate_ids)
    policy_failures = {
        f"authority-policy-{value.lower().replace('_', '-')}" for value in refusal_reasons
    }
    failed = tuple(sorted(missing_required | policy_failures))
    approved = not refusal_reasons
    return ResponseAlgebraConformanceAuthorization(
        authorization_id=authorization_id,
        package_id=package.package_id,
        package_sha256=package.fingerprint(),
        policy_id=policy.policy_id,
        policy_sha256=policy.fingerprint(),
        decision=(
            ConformanceAuthorizationDecision.APPROVED_NONACTUATING
            if approved
            else ConformanceAuthorizationDecision.AUTHORITY_REQUIRED
        ),
        passed_gate_ids=tuple(sorted(passed_gate_ids)),
        failed_gate_ids=failed,
        reason_codes=(
            ("AUTHORITY_APPROVED_NONACTUATING",)
            if approved
            else tuple(sorted(f"AUTHORITY_{value}" for value in refusal_reasons))
        ),
        proposer_id=proposer_id,
        approver_id=approver_id,
        decided_at_utc=decided_at_utc,
        implementation_commit=package.implementation_commit,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        plan_mutated=False,
        grants_claim_promotion=False,
    )


def compile_response_algebra_conformance(
    package: ResponseAlgebraConformanceExecutionPackage,
    authorization: ResponseAlgebraConformanceAuthorization,
) -> CompiledResponseAlgebraConformance:
    """Fail closed unless authorization binds the exact clean package."""

    if authorization.decision is not ConformanceAuthorizationDecision.APPROVED_NONACTUATING:
        raise PermissionError("response-algebra conformance authority is required")
    if (
        authorization.package_id != package.package_id
        or authorization.package_sha256 != package.fingerprint()
        or authorization.implementation_commit != package.implementation_commit
    ):
        raise ValueError("conformance authorization differs from its execution package")
    case_count = len(package.conformance_spec.case_tokens)
    return CompiledResponseAlgebraConformance(
        compiled_id=f"compiled.{package.package_id}",
        package_id=package.package_id,
        package_sha256=package.fingerprint(),
        authorization_id=authorization.authorization_id,
        authorization_sha256=authorization.fingerprint(),
        task_ids=(
            "authorize",
            "freeze-truth-blind-inputs",
            "identify",
            "privileged-score",
            "publish-and-verify",
        ),
        case_tokens=package.conformance_spec.case_tokens,
        expected_truth_blind_invocations=case_count,
        expected_privileged_oracles=case_count,
        expected_case_scores=case_count,
        maximum_output_bytes=package.resource_budget.output_bytes,
    )
