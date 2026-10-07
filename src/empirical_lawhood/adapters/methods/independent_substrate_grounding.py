"Shared scientific design and adjudication records for independent substrate grounding.\n\nThis adapter-local module freezes the finite design grammar used by the four\nin-scope software lanes.  It does not decode target data and does not change\nthe structural recurrence predictor.  Target-specific states are instantiated only after\nsource qualification, from this closed grammar, before development outcomes.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Iterable

from empirical_lawhood.adapters.methods.structural_recurrence import TargetLevel
from empirical_lawhood.adapters.methods.observed_structural_classes import EvidenceWorld
from empirical_lawhood.adapters.methods.scientific_description_code import scientific_description_octets

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)


class IndependentSubstrateTargetKind(StrEnum):
    BOPTEST_BRIDGE = "boptest-bridge"
    FREEGSNKE = "freegsnke"
    GRID2OP = "grid2op"
    NREL_INVERTER = "nrel-inverter-archive"


class IndependentSubstrateIndependenceClass(StrEnum):
    NOVEL_EXTERNAL_IMPLEMENTATION = "NOVEL_EXTERNAL_IMPLEMENTATION"
    DONOR_VISIBLE_BRIDGE = "DONOR_VISIBLE_BRIDGE"
    PHYSICAL_EXTERNAL_APPARATUS = "PHYSICAL_EXTERNAL_APPARATUS"
    CONTAMINATED = "CONTAMINATED"
    UNEVALUABLE = "UNEVALUABLE"


class IndependentSubstrateComparatorKind(StrEnum):
    STRUCTURAL_RECURRENCE = "structural-recurrence"
    WILDCARD_ALL = "wildcard-all"
    ALWAYS_HOLD = "always-hold"
    ALWAYS_ADMIT_OR_ACT = "always-admit-or-act"
    DEVELOPMENT_MAJORITY = "development-majority"
    DENOMINATOR_BLIND = "denominator-blind"
    RECEIVER_BLIND = "receiver-blind"
    SATURATED_DEVELOPMENT_LOOKUP = "saturated-development-lookup"


class IndependentSubstrateMappingConstructor(StrEnum):
    DIRECT_NATIVE_ROLE_BINDING = "direct-native-role-binding"
    DECLARED_TARGET_NATIVE_AGGREGATE = "declared-target-native-aggregate"
    DECLARED_TARGET_NATIVE_RESTRICTION = "declared-target-native-restriction"
    DECLARED_RECEIVER_GAUGE_PROJECTION = "declared-receiver-gauge-projection"
    DECLARED_HORIZON_SELECTION = "declared-horizon-selection"


class IndependentSubstrateCompatibilityDisposition(StrEnum):
    COMPATIBLE = "COMPATIBLE"
    METHOD_DOMAIN_UNRESOLVED = "METHOD_DOMAIN_UNRESOLVED"
    TARGET_OPERAND_INCOMPLETE = "TARGET_OPERAND_INCOMPLETE"


class IndependentSubstrateInterfaceShape(StrEnum):
    IRREGULAR_GRAPH = "IRREGULAR_GRAPH"
    CONTINUOUS_PHYSICAL_ARCHIVE = "CONTINUOUS_PHYSICAL_ARCHIVE"
    ACTION_LEDGER = "ACTION_LEDGER"
    RECEIVER_EXCHANGE = "RECEIVER_EXCHANGE"
    VARIABLE_LEVEL = "VARIABLE_LEVEL"


class IndependentSubstrateConformanceCaseKind(StrEnum):
    INTERFACE_IRREGULAR_GRAPH = "INTERFACE_IRREGULAR_GRAPH"
    INTERFACE_CONTINUOUS_PHYSICAL_ARCHIVE = "INTERFACE_CONTINUOUS_PHYSICAL_ARCHIVE"
    INTERFACE_ACTION_LEDGER = "INTERFACE_ACTION_LEDGER"
    INTERFACE_RECEIVER_EXCHANGE_BACKACTION = "INTERFACE_RECEIVER_EXCHANGE_BACKACTION"
    INTERFACE_VARIABLE_LEVEL = "INTERFACE_VARIABLE_LEVEL"
    PREDICTOR_POSITIVE = "PREDICTOR_POSITIVE"
    PREDICTOR_OPPOSED = "PREDICTOR_OPPOSED"
    COMPARATOR_WILDCARD = "COMPARATOR_WILDCARD"
    COMPARATOR_SATURATED_LOOKUP = "COMPARATOR_SATURATED_LOOKUP"
    UNSAFE_FALSE_ADMISSION = "UNSAFE_FALSE_ADMISSION"
    INCOMPLETE_OPERAND = "INCOMPLETE_OPERAND"
    NONMINIMAL_DENOMINATOR = "NONMINIMAL_DENOMINATOR"
    POST_ISSUE_DOMAIN_LOSS = "POST_ISSUE_DOMAIN_LOSS"
    POST_ISSUE_MISSINGNESS = "POST_ISSUE_MISSINGNESS"


class IndependentSubstrateIndependentAxis(StrEnum):
    SUPPORTED = "INDEPENDENT_IMPLEMENTATION_RECURRENCE_SUPPORTED"
    MIXED = "INDEPENDENT_IMPLEMENTATION_RECURRENCE_MIXED"
    OPPOSED = "INDEPENDENT_IMPLEMENTATION_RECURRENCE_OPPOSED"
    UNEVALUABLE = "INDEPENDENT_IMPLEMENTATION_RECURRENCE_UNEVALUABLE"


class IndependentSubstrateRestrictivenessAxis(StrEnum):
    SUPPORTED = "PREDICTIVE_RESTRICTIVENESS_SUPPORTED"
    NOT_DISTINGUISHED = "PREDICTIVE_RESTRICTIVENESS_NOT_DISTINGUISHED"
    OPPOSED = "PREDICTIVE_RESTRICTIVENESS_OPPOSED"
    UNEVALUABLE = "PREDICTIVE_RESTRICTIVENESS_UNEVALUABLE"


class IndependentSubstrateTopologyMetricAxis(StrEnum):
    TOPOLOGY_STRONGER = "TOPOLOGY_VS_METRIC_TRANSPORT_TOPOLOGY_STRONGER"
    EQUIVALENT_OR_UNEVALUABLE = "TOPOLOGY_VS_METRIC_TRANSPORT_EQUIVALENT_OR_UNEVALUABLE"
    METRIC_STRONGER_OR_TOPOLOGY_OPPOSED = (
        "TOPOLOGY_VS_METRIC_TRANSPORT_METRIC_STRONGER_OR_TOPOLOGY_OPPOSED"
    )


class IndependentSubstratePhysicalAxis(StrEnum):
    SUPPORTED = "PHYSICAL_STRUCTURE_CONSISTENCY_SUPPORTED"
    MIXED = "PHYSICAL_STRUCTURE_CONSISTENCY_MIXED"
    OPPOSED = "PHYSICAL_STRUCTURE_CONSISTENCY_OPPOSED"
    UNEVALUABLE = "PHYSICAL_STRUCTURE_CONSISTENCY_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class IndependentSubstrateTargetSlot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-target-slot'

    slot_id: str
    slot: IndependentSubstrateTargetKind
    evidence_world: EvidenceWorld
    fixed_independence_class: IndependentSubstrateIndependenceClass | None
    required_support_level: TargetLevel
    counts_for_independent_recurrence: bool
    conditional_prospective_validation_allowed: bool
    substitution_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.slot_id, field_name="slot_id")
        if self.substitution_allowed:
            raise ValueError("independent substrate grounding target slots prohibit substitution")
        if self.slot in {IndependentSubstrateTargetKind.FREEGSNKE, IndependentSubstrateTargetKind.GRID2OP}:
            if (
                not self.counts_for_independent_recurrence
                or self.required_support_level is not TargetLevel.ADMISSION
                or not self.conditional_prospective_validation_allowed
                or self.fixed_independence_class is not None
            ):
                raise ValueError("novel target slot contract differs")
        elif self.counts_for_independent_recurrence:
            raise ValueError("bridge/archive slots cannot count as novel targets")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateForecastStateCode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-forecast-state-code'

    state_id: str
    integer_code: int

    def __post_init__(self) -> None:
        validate_stable_id(self.state_id, field_name="state_id")
        if self.integer_code < 0:
            raise ValueError("forecast state code must be nonnegative")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateForecastAlphabetGrammar(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-forecast-alphabet-grammar'

    grammar_id: str
    role_vocabulary: tuple[str, ...]
    constructor_ids: tuple[str, ...]
    common_state_codes: tuple[IndependentSubstrateForecastStateCode, ...]
    prohibited_fallbacks: tuple[str, ...]
    forecast_denominator_rules: tuple[str, ...]
    target_instantiation_stage: str

    def __post_init__(self) -> None:
        validate_stable_id(self.grammar_id, field_name="grammar_id")
        require_sorted_unique_strings(
            self.role_vocabulary,
            field_name="role_vocabulary",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.constructor_ids,
            field_name="constructor_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.common_state_codes,
            attribute="state_id",
            field_name="common_state_codes",
        )
        codes = tuple(value.integer_code for value in self.common_state_codes)
        if len(codes) != len(set(codes)):
            raise ValueError("forecast integer codes must be unique")
        require_sorted_unique_strings(
            self.prohibited_fallbacks,
            field_name="prohibited_fallbacks",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.forecast_denominator_rules,
            field_name="forecast_denominator_rules",
            allow_empty=False,
        )
        if set(self.forecast_denominator_rules) != {
            "all-issued-complete-evaluation-units",
            "no-post-issue-outcome-exclusion",
            "panel-limited-remains-in-issued-denominator",
            "pre-issue-nonentry-excluded-with-typed-reason",
            "target-local-no-cross-target-pooling",
        }:
            raise ValueError("forecast denominator rules differ")
        if self.target_instantiation_stage != "G2_BEFORE_DEVELOPMENT":
            raise ValueError("target alphabets must freeze before development")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateMappingCandidateGrammar(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-mapping-candidate-grammar'

    grammar_id: str
    required_structural_recurrence_role_ids: tuple[str, ...]
    constructor_ids: tuple[IndependentSubstrateMappingConstructor, ...]
    maximum_candidates_per_target: int
    roster_instantiation_stage: str
    selection_stage: str
    selection_precedence: tuple[str, ...]
    prohibited_operations: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.grammar_id, field_name="grammar_id")
        if self.required_structural_recurrence_role_ids != ("A", "D", "H", "R", "tau"):
            raise ValueError("mapping grammar must bind exact D/H/A/R/tau roles")
        if tuple(sorted(set(self.constructor_ids), key=lambda value: value.value)) != tuple(
            sorted(self.constructor_ids, key=lambda value: value.value)
        ) or set(self.constructor_ids) != set(IndependentSubstrateMappingConstructor):
            raise ValueError("mapping constructor roster differs")
        if not 1 <= self.maximum_candidates_per_target <= 64:
            raise ValueError("mapping candidate bound is outside the frozen range")
        if self.roster_instantiation_stage != "G2_BEFORE_DEVELOPMENT":
            raise ValueError("mapping candidates must freeze before development")
        if self.selection_stage != "DEVELOPMENT_ONLY_BEFORE_PREDICTION_ISSUE":
            raise ValueError("mapping selection must remain development-only")
        if self.selection_precedence != (
            "unsafe-false-admission",
            "categorical-mismatch-count",
            "prediction-set-cardinality",
            "canonical-description-bits",
            "stable-candidate-id",
        ):
            raise ValueError("mapping candidate selection precedence differs")
        require_sorted_unique_strings(
            self.prohibited_operations,
            field_name="prohibited_operations",
            allow_empty=False,
        )
        if set(self.prohibited_operations) != {
            "add-candidate-after-development",
            "derive-binding-from-evaluation-outcome",
            "merge-roles-after-evaluation",
            "select-by-evaluation-score",
            "substitute-target-native-unit-or-frame",
        }:
            raise ValueError("mapping candidate prohibited operations differ")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateRoleBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-role-binding'

    binding_id: str
    structural_recurrence_role_id: str
    target_native_role_ids: tuple[str, ...]
    constructor: IndependentSubstrateMappingConstructor
    native_units_preserved: bool
    native_frame_preserved: bool
    receiver_direction_preserved: bool
    causal_cutoff_preserved: bool
    outcome_derived: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if self.structural_recurrence_role_id not in {"A", "D", "H", "R", "tau"}:
            raise ValueError('role binding names an unknown structural recurrence role')
        require_sorted_unique_strings(
            self.target_native_role_ids,
            field_name="target_native_role_ids",
            allow_empty=False,
        )
        if not all(
            (
                self.native_units_preserved,
                self.native_frame_preserved,
                self.receiver_direction_preserved,
                self.causal_cutoff_preserved,
            )
        ):
            raise ValueError("mapping candidate must preserve native role semantics")
        if self.outcome_derived:
            raise ValueError("mapping bindings cannot be outcome-derived")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateTargetMappingCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-target-mapping-candidate'

    candidate_id: str
    target_slot: IndependentSubstrateTargetKind
    finite_roster_id: str
    role_bindings: tuple[IndependentSubstrateRoleBinding, ...]
    requested_accepted_applied_realized_distinct: bool
    target_native_action_chart_preserved: bool
    development_authored: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_stable_id(self.finite_roster_id, field_name="finite_roster_id")
        require_sorted_unique_ids(
            self.role_bindings,
            attribute="binding_id",
            field_name="role_bindings",
        )
        if tuple(value.structural_recurrence_role_id for value in self.role_bindings) != (
            "A",
            "D",
            "H",
            "R",
            "tau",
        ):
            raise ValueError("mapping candidate must bind exact D/H/A/R/tau roles")
        if not all(
            (
                self.requested_accepted_applied_realized_distinct,
                self.target_native_action_chart_preserved,
                self.development_authored,
            )
        ):
            raise ValueError("mapping candidate violates the frozen authoring boundary")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateMappingDevelopmentAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-mapping-development-assessment'

    assessment_id: str
    candidate: IndependentSubstrateTargetMappingCandidate
    same_complete_unit_ids_sha256: str
    unsafe_false_admission_count: int
    categorical_mismatch_count: int
    prediction_set_cardinality: int
    evaluation_outcome_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_sha256(
            self.same_complete_unit_ids_sha256,
            field_name="same_complete_unit_ids_sha256",
        )
        for name in (
            "unsafe_false_admission_count",
            "categorical_mismatch_count",
            "prediction_set_cardinality",
            "evaluation_outcome_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.evaluation_outcome_count:
            raise ValueError("mapping selection cannot access evaluation outcomes")

    @property
    def lexicographic_key(self) -> tuple[int, int, int, int, str]:
        return (
            self.unsafe_false_admission_count,
            self.categorical_mismatch_count,
            self.prediction_set_cardinality,
            scientific_description_octets(self.candidate) * 8,
            self.candidate.candidate_id,
        )


def select_mapping_candidate(
    assessments: Iterable[IndependentSubstrateMappingDevelopmentAssessment],
    *,
    maximum_candidates: int,
) -> IndependentSubstrateMappingDevelopmentAssessment:
    """Select one finite development-only mapping without evaluation rescue."""

    values = tuple(assessments)
    if not values or len(values) > maximum_candidates:
        raise ValueError("mapping candidate roster is empty or exceeds its frozen bound")
    if len({value.candidate.candidate_id for value in values}) != len(values):
        raise ValueError("mapping candidate roster contains duplicate identities")
    if len({value.candidate.target_slot for value in values}) != 1:
        raise ValueError("mapping candidates cannot cross target slots")
    if len({value.candidate.finite_roster_id for value in values}) != 1:
        raise ValueError("mapping candidates must belong to one frozen roster")
    if len({value.same_complete_unit_ids_sha256 for value in values}) != 1:
        raise ValueError("mapping candidates must use identical complete units")
    return min(values, key=lambda value: value.lexicographic_key)


@dataclass(frozen=True, slots=True)
class IndependentSubstratePowerAndUncertaintyGrammar(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-power-and-uncertainty-grammar'

    grammar_id: str
    complete_unit_role: str
    resampling_unit_role: str
    jointly_powered_axis_ids: tuple[str, ...]
    simultaneous_family_ids: tuple[str, ...]
    nested_rows_or_views_inflate_replication: bool
    cross_target_pooling_allowed: bool
    panel_limited_postissue_disposition: str
    physical_response_and_certification_margins_distinct: bool
    freeze_stage: str

    def __post_init__(self) -> None:
        validate_stable_id(self.grammar_id, field_name="grammar_id")
        if self.complete_unit_role != "TARGET_PHYSICAL_PREPARATION_OR_NATIVE_EPISODE":
            raise ValueError("power grammar complete-unit role differs")
        if self.resampling_unit_role != "COMPLETE_UNIT":
            raise ValueError("power grammar must resample complete units")
        if self.jointly_powered_axis_ids != (
            "observation-grid",
            "panel",
            "receiver-coordinate",
        ):
            raise ValueError("joint power axes differ")
        require_sorted_unique_strings(
            self.simultaneous_family_ids,
            field_name="simultaneous_family_ids",
            allow_empty=False,
        )
        if set(self.simultaneous_family_ids) != {
            "all-primary-estimands",
            "all-decisive-falsifiers",
            "all-topology-metric-exchanges",
        }:
            raise ValueError("simultaneous uncertainty families differ")
        if self.nested_rows_or_views_inflate_replication or self.cross_target_pooling_allowed:
            raise ValueError("power grammar cannot inflate or pool replication")
        if self.panel_limited_postissue_disposition != "PANEL_LIMITED_NOT_NONENTRY":
            raise ValueError("panel limitation cannot become post-issue nonentry")
        if not self.physical_response_and_certification_margins_distinct:
            raise ValueError("physical and certification margins must remain distinct")
        if self.freeze_stage != "G3_BEFORE_PREDICTION_ISSUE":
            raise ValueError("target power must freeze before prediction issue")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateComparatorCodebook(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-comparator-codebook'

    codebook_id: str
    comparator_kinds: tuple[IndependentSubstrateComparatorKind, ...]
    score_precedence: tuple[str, ...]
    saturated_out_of_cell_rule: str
    description_length_rule: str
    target_averaging_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.codebook_id, field_name="codebook_id")
        if tuple(sorted(set(self.comparator_kinds), key=lambda value: value.value)) != tuple(
            sorted(self.comparator_kinds, key=lambda value: value.value)
        ):
            raise ValueError("comparator roster must be sorted and unique")
        if set(self.comparator_kinds) != set(IndependentSubstrateComparatorKind):
            raise ValueError("comparator codebook must contain the exact common roster")
        if self.score_precedence != (
            "unsafe-false-admission",
            "categorical-mismatch-count",
            "prediction-set-cardinality",
            "canonical-description-bits",
        ):
            raise ValueError("comparator score precedence differs")
        if self.saturated_out_of_cell_rule != "WILDCARD_ALL":
            raise ValueError("saturated lookup out-of-cell behavior differs")
        if self.description_length_rule != 'SCIENTIFIC_DESCRIPTION_CODEBOOK_BITS':
            raise ValueError("comparator description-length rule differs")
        if self.target_averaging_allowed:
            raise ValueError("comparator scores cannot average across targets")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateClaimContrastRequirement(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-claim-contrast-requirement'

    contrast_id: str
    law_operand: str
    minimum_contrast: str
    absence_narrows_claim: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.contrast_id, field_name="contrast_id")
        if self.law_operand not in {"D", "H", "A", "R", "tau"}:
            raise ValueError("claim contrast names an unknown law operand")
        validate_nonempty(self.minimum_contrast, field_name="minimum_contrast")
        if not self.absence_narrows_claim:
            raise ValueError("missing within-system contrasts must narrow claims")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateScientificDesignBasis(CanonicalRecord):
    """The complete outcome-blind scientific grammar outcome-blind scientific grammar."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-scientific-design-basis'

    design_basis_id: str
    method_port_freeze: ObjectIdentity
    applicability_overlay: ObjectIdentity
    target_slots: tuple[IndependentSubstrateTargetSlot, ...]
    mapping_grammar: IndependentSubstrateMappingCandidateGrammar
    forecast_grammar: IndependentSubstrateForecastAlphabetGrammar
    comparator_codebook: IndependentSubstrateComparatorCodebook
    power_and_uncertainty_grammar: IndependentSubstratePowerAndUncertaintyGrammar
    claim_contrasts: tuple[IndependentSubstrateClaimContrastRequirement, ...]
    action_clock_role_ids: tuple[str, ...]
    mandatory_hold_rules: tuple[str, ...]
    denominator_minimality_order: tuple[str, ...]
    topology_metric_rules: tuple[str, ...]
    nonentry_precedence: tuple[str, ...]
    target_issue_preconditions: tuple[str, ...]
    axis_ids: tuple[str, ...]
    excluded_scope_ids: tuple[str, ...]
    unresolved_in_scope_scientific_choice_ids: tuple[str, ...]
    generated_target_parameters_imported: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.design_basis_id, field_name="design_basis_id")
        require_sorted_unique_ids(
            self.target_slots,
            attribute="slot_id",
            field_name="target_slots",
        )
        if tuple(value.slot for value in self.target_slots) != (
            IndependentSubstrateTargetKind.BOPTEST_BRIDGE,
            IndependentSubstrateTargetKind.FREEGSNKE,
            IndependentSubstrateTargetKind.GRID2OP,
            IndependentSubstrateTargetKind.NREL_INVERTER,
        ):
            raise ValueError("design basis target slot roster/order differs")
        require_sorted_unique_ids(
            self.claim_contrasts,
            attribute="contrast_id",
            field_name="claim_contrasts",
        )
        if {value.law_operand for value in self.claim_contrasts} != {
            "D",
            "H",
            "A",
            "R",
            "tau",
        }:
            raise ValueError("design basis must cover D/H/A/R/tau contrasts")
        if self.action_clock_role_ids != (
            "accepted",
            "applied",
            "realized",
            "requested",
        ):
            raise ValueError("design basis action-clock ontology differs")
        for name in (
            "mandatory_hold_rules",
            "topology_metric_rules",
            "nonentry_precedence",
            "target_issue_preconditions",
            "axis_ids",
            "excluded_scope_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        if self.denominator_minimality_order != (
            "source-qualified-factor-and-stratum-count",
            "canonical-comparator-codebook-byte-length",
            "stable-candidate-id",
        ):
            raise ValueError("denominator minimality order differs")
        if set(self.mandatory_hold_rules) != {
            "hold-on-action-clock-ambiguity",
            "hold-on-authority-absence",
            "hold-on-domain-loss",
            "hold-on-invalid-receiver-or-gauge",
            'hold-on-admission-failure',
            "hold-on-support-or-reachability-absence",
        }:
            raise ValueError("mandatory hold rules differ")
        if self.unresolved_in_scope_scientific_choice_ids:
            raise ValueError("outcome-blind scientific grammar cannot freeze with unresolved in-scope science")
        if self.generated_target_parameters_imported:
            raise ValueError("outcome-blind scientific grammar cannot import generated structural recurrence follow-up parameters")
        if set(self.excluded_scope_ids) != {
            "CPAIM_AND_ALL_PROSPECTIVE_PHYSICAL_WORK",
            "NATURE_OUTLINE_MANUSCRIPT_FIGURES_AND_SUBMISSION",
            "SCALE_COVARIANCE_COMPOSITION_AND_FIXED_POINT_CLAIMS",
        }:
            raise ValueError("design basis excluded-scope boundary differs")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateDenominatorCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-denominator-candidate'

    candidate_id: str
    factor_ids: tuple[str, ...]
    stratum_ids: tuple[str, ...]
    impossible: bool
    impossible_reason: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        require_sorted_unique_strings(self.factor_ids, field_name="factor_ids")
        require_sorted_unique_strings(self.stratum_ids, field_name="stratum_ids")
        if self.impossible != (self.impossible_reason is not None):
            raise ValueError("impossible denominator candidate requires exactly one reason")

    @property
    def factor_and_stratum_count(self) -> int:
        return len(self.factor_ids) + len(self.stratum_ids)


@dataclass(frozen=True, slots=True)
class IndependentSubstrateDenominatorDevelopmentAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-denominator-development-assessment'

    assessment_id: str
    candidate: IndependentSubstrateDenominatorCandidate
    same_complete_unit_ids_sha256: str
    legal_action_alphabet_preserved: bool
    forecast_roster_preserved: bool
    policy_outputs_preserved: bool
    unsafe_error_count: int
    simultaneous_precision_passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_sha256(
            self.same_complete_unit_ids_sha256,
            field_name="same_complete_unit_ids_sha256",
        )
        if self.unsafe_error_count < 0:
            raise ValueError("unsafe error count must be nonnegative")

    @property
    def eligible(self) -> bool:
        return not self.candidate.impossible and all(
            (
                self.legal_action_alphabet_preserved,
                self.forecast_roster_preserved,
                self.policy_outputs_preserved,
                self.unsafe_error_count == 0,
                self.simultaneous_precision_passed,
            )
        )


def select_minimal_denominator(
    assessments: Iterable[IndependentSubstrateDenominatorDevelopmentAssessment],
) -> IndependentSubstrateDenominatorDevelopmentAssessment:
    """Apply the outcome-blind scientific grammar common order without evaluation-time lattice growth."""

    values = tuple(assessments)
    if not values:
        raise ValueError("DENOMINATOR_MINIMALITY_UNRESOLVED: empty candidate lattice")
    unit_hashes = {value.same_complete_unit_ids_sha256 for value in values}
    if len(unit_hashes) != 1:
        raise ValueError("denominator candidates must use identical complete units")
    eligible = tuple(value for value in values if value.eligible)
    if not eligible:
        raise ValueError("DENOMINATOR_MINIMALITY_UNRESOLVED: no candidate closes")
    return min(
        eligible,
        key=lambda value: (
            value.candidate.factor_and_stratum_count,
            scientific_description_octets(value.candidate),
            value.candidate.candidate_id,
        ),
    )


@dataclass(frozen=True, slots=True)
class IndependentSubstrateLookupCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-lookup-cell'

    cell_id: str
    key_codes: tuple[int, ...]
    emitted_state_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if not self.key_codes or not self.emitted_state_codes:
            raise ValueError("lookup cells require keys and emitted states")
        if tuple(sorted(set(self.emitted_state_codes))) != self.emitted_state_codes:
            raise ValueError("lookup emitted states must be sorted and unique")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateComparatorEncoding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-comparator-encoding'

    encoding_id: str
    kind: IndependentSubstrateComparatorKind
    selected_parameter_codes: tuple[int, ...]
    denominator_dependency_ids: tuple[str, ...]
    receiver_dependency_ids: tuple[str, ...]
    lookup_cells: tuple[IndependentSubstrateLookupCell, ...]
    out_of_cell_rule: str

    def __post_init__(self) -> None:
        validate_stable_id(self.encoding_id, field_name="encoding_id")
        require_sorted_unique_strings(
            self.denominator_dependency_ids,
            field_name="denominator_dependency_ids",
        )
        require_sorted_unique_strings(
            self.receiver_dependency_ids,
            field_name="receiver_dependency_ids",
        )
        require_sorted_unique_ids(
            self.lookup_cells,
            attribute="cell_id",
            field_name="lookup_cells",
        )
        if any(value < 0 for value in self.selected_parameter_codes):
            raise ValueError("comparator parameter codes must be nonnegative")
        lookup_kinds = {
            IndependentSubstrateComparatorKind.DENOMINATOR_BLIND,
            IndependentSubstrateComparatorKind.RECEIVER_BLIND,
            IndependentSubstrateComparatorKind.SATURATED_DEVELOPMENT_LOOKUP,
        }
        if self.kind in lookup_kinds:
            if self.out_of_cell_rule != "WILDCARD_ALL":
                raise ValueError("lookup comparator must wildcard outside development cells")
        elif self.lookup_cells:
            raise ValueError("only lookup comparators may carry lookup cells")

    @property
    def scientific_description_bits(self) -> int:
        return 8 * scientific_description_octets(self)


@dataclass(frozen=True, slots=True)
class IndependentSubstrateComparatorScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-comparator-score'

    score_id: str
    encoding: ObjectIdentity
    unsafe_false_admission_count: int
    categorical_mismatch_count: int
    prediction_set_cardinality: int
    scientific_description_bits: int

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        for name in (
            "unsafe_false_admission_count",
            "categorical_mismatch_count",
            "prediction_set_cardinality",
            "scientific_description_bits",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def lexicographic_key(self) -> tuple[int, int, int, int]:
        return (
            self.unsafe_false_admission_count,
            self.categorical_mismatch_count,
            self.prediction_set_cardinality,
            self.scientific_description_bits,
        )


@dataclass(frozen=True, slots=True)
class IndependentSubstrateTargetPredictionContract(CanonicalRecord):
    "One immutable target selection/freeze before structural recurrence issue and reveal.\n\n    Target adapters own their native forecast, power and metric/topology\n    schemas. This common record binds those exact identities while retaining\n    the finite outcome-blind scientific grammar mapping and denominator alternatives needed to prove that\n    development selected from a pre-existing roster rather than inventing a\n    rescue model.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-target-prediction-contract'

    contract_id: str
    target_slot: IndependentSubstrateTargetKind
    scientific_design_basis: ObjectIdentity
    source_qualification: ObjectIdentity
    independence_dossier: IndependentImplementationDossier
    mapping_candidate_roster: tuple[IndependentSubstrateTargetMappingCandidate, ...]
    mapping_development_assessments: tuple[IndependentSubstrateMappingDevelopmentAssessment, ...]
    selected_mapping_candidate_id: str
    denominator_candidate_lattice: tuple[IndependentSubstrateDenominatorCandidate, ...]
    denominator_development_assessments: tuple[IndependentSubstrateDenominatorDevelopmentAssessment, ...]
    selected_denominator_candidate_id: str
    selected_claimed_law_operand_ids: tuple[str, ...]
    selected_unsupported_law_operand_ids: tuple[str, ...]
    proposed_denominator_candidate_id: str
    meaningful_split_candidate_ids: tuple[str, ...]
    merge_or_omission_candidate_ids: tuple[str, ...]
    impossible_candidate_ids: tuple[str, ...]
    target_forecast_alphabet: ObjectIdentity
    comparator_encodings: tuple[IndependentSubstrateComparatorEncoding, ...]
    target_power_freeze: ObjectIdentity
    metric_topology_design: ObjectIdentity
    target_development_evidence: ObjectIdentity
    structural_recurrence_target_design: ObjectIdentity
    structural_recurrence_development_evidence: ObjectIdentity
    same_complete_unit_ids_sha256: str
    frozen_before_prediction_issue: bool
    published_before_evaluator_access: bool
    evaluation_outcome_access_count: int

    def __post_init__(self) -> None:
        for name in (
            "contract_id",
            "selected_mapping_candidate_id",
            "selected_denominator_candidate_id",
            "proposed_denominator_candidate_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.scientific_design_basis.object_schema != IndependentSubstrateScientificDesignBasis.SCHEMA:
            raise ValueError("target contract scientific design basis differs")
        if (
            self.independence_dossier.slot is not self.target_slot
            or not self.independence_dossier.prediction_precedes_outcome_access
        ):
            raise ValueError("target contract independence dossier differs")
        if self.structural_recurrence_target_design.object_schema != ('empirical-lawhood/methods/structural-recurrence-target-design-freeze'):
            raise ValueError('target contract structural recurrence target design differs')
        if self.structural_recurrence_development_evidence.object_schema != ('empirical-lawhood/methods/structural-recurrence-stage-evidence'):
            raise ValueError('target contract structural recurrence development evidence differs')
        validate_sha256(
            self.same_complete_unit_ids_sha256,
            field_name="same_complete_unit_ids_sha256",
        )

        require_sorted_unique_ids(
            self.mapping_candidate_roster,
            attribute="candidate_id",
            field_name="mapping_candidate_roster",
        )
        require_sorted_unique_ids(
            self.mapping_development_assessments,
            attribute="assessment_id",
            field_name="mapping_development_assessments",
        )
        if not self.mapping_candidate_roster or len(self.mapping_candidate_roster) > 32:
            raise ValueError("target mapping roster is empty or exceeds outcome-blind scientific grammar")
        if any(
            value.target_slot is not self.target_slot for value in self.mapping_candidate_roster
        ):
            raise ValueError("target contract mapping roster crosses target slots")
        if len({value.finite_roster_id for value in self.mapping_candidate_roster}) != 1:
            raise ValueError("target contract mapping roster is not one finite freeze")
        mapping_by_id = {value.candidate_id: value for value in self.mapping_candidate_roster}
        assessed_mapping_ids = {
            value.candidate.candidate_id for value in self.mapping_development_assessments
        }
        if (
            len(self.mapping_development_assessments) != len(mapping_by_id)
            or assessed_mapping_ids != set(mapping_by_id)
            or any(
                mapping_by_id[value.candidate.candidate_id] != value.candidate
                for value in self.mapping_development_assessments
            )
        ):
            raise ValueError("target contract mapping assessments differ from the frozen roster")
        if any(
            value.same_complete_unit_ids_sha256 != self.same_complete_unit_ids_sha256
            for value in self.mapping_development_assessments
        ):
            raise ValueError("target contract mapping selection changed complete units")
        selected_mapping = select_mapping_candidate(
            self.mapping_development_assessments,
            maximum_candidates=32,
        )
        if selected_mapping.candidate.candidate_id != self.selected_mapping_candidate_id:
            raise ValueError("target contract mapping selection differs from outcome-blind scientific grammar precedence")

        require_sorted_unique_ids(
            self.denominator_candidate_lattice,
            attribute="candidate_id",
            field_name="denominator_candidate_lattice",
        )
        require_sorted_unique_ids(
            self.denominator_development_assessments,
            attribute="assessment_id",
            field_name="denominator_development_assessments",
        )
        denominator_by_id = {
            value.candidate_id: value for value in self.denominator_candidate_lattice
        }
        assessed_denominator_ids = {
            value.candidate.candidate_id for value in self.denominator_development_assessments
        }
        if (
            len(self.denominator_development_assessments) != len(denominator_by_id)
            or assessed_denominator_ids != set(denominator_by_id)
            or any(
                denominator_by_id[value.candidate.candidate_id] != value.candidate
                for value in self.denominator_development_assessments
            )
        ):
            raise ValueError(
                "target contract denominator assessments differ from the frozen lattice"
            )
        for name in (
            "meaningful_split_candidate_ids",
            "merge_or_omission_candidate_ids",
            "impossible_candidate_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        categories = (
            (self.proposed_denominator_candidate_id,),
            self.meaningful_split_candidate_ids,
            self.merge_or_omission_candidate_ids,
            self.impossible_candidate_ids,
        )
        flattened_categories = tuple(value for group in categories for value in group)
        if len(flattened_categories) != len(set(flattened_categories)) or set(
            flattened_categories
        ) != set(denominator_by_id):
            raise ValueError("target denominator kinds do not partition the frozen lattice")
        if any(
            denominator_by_id[candidate_id].impossible
            != (candidate_id in self.impossible_candidate_ids)
            for candidate_id in denominator_by_id
        ):
            raise ValueError("target denominator impossible disposition differs from its kind")
        if any(
            value.same_complete_unit_ids_sha256 != self.same_complete_unit_ids_sha256
            for value in self.denominator_development_assessments
        ):
            raise ValueError("target denominator selection changed complete units")
        selected_denominator = select_minimal_denominator(self.denominator_development_assessments)
        if selected_denominator.candidate.candidate_id != self.selected_denominator_candidate_id:
            raise ValueError("target contract denominator selection differs from outcome-blind scientific grammar precedence")
        for name in (
            "selected_claimed_law_operand_ids",
            "selected_unsupported_law_operand_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        claimed = set(self.selected_claimed_law_operand_ids)
        unsupported = set(self.selected_unsupported_law_operand_ids)
        if claimed & unsupported or claimed | unsupported != {"A", "D", "H", "R", "tau"}:
            raise ValueError(
                "target contract selected claimed/unsupported operands do not partition the law"
            )

        require_sorted_unique_ids(
            self.comparator_encodings,
            attribute="encoding_id",
            field_name="comparator_encodings",
        )
        if len(self.comparator_encodings) != len(IndependentSubstrateComparatorKind) or {
            value.kind for value in self.comparator_encodings
        } != set(IndependentSubstrateComparatorKind):
            raise ValueError("target contract comparator roster differs from outcome-blind scientific grammar")
        if not all(
            (
                self.frozen_before_prediction_issue,
                self.published_before_evaluator_access,
                self.evaluation_outcome_access_count == 0,
            )
        ):
            raise ValueError("target prediction contract crossed issue or reveal")


def saturated_lookup_emit(
    *,
    cells: tuple[IndependentSubstrateLookupCell, ...],
    key_codes: tuple[int, ...],
    wildcard_state_codes: tuple[int, ...],
) -> tuple[int, ...]:
    """Exact union lookup with the prospectively frozen wildcard fallback."""

    matches = tuple(value for value in cells if value.key_codes == key_codes)
    if not matches:
        return tuple(sorted(set(wildcard_state_codes)))
    return tuple(sorted({code for match in matches for code in match.emitted_state_codes}))


@dataclass(frozen=True, slots=True)
class IndependentSubstrateMetricTopologyExchange(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-metric-topology-exchange'

    exchange_id: str
    complete_unit_ids_sha256: str
    topology_success_count: int
    metric_success_count: int
    complete_unit_count: int
    delta: Decimal
    simultaneous_lower: Decimal
    simultaneous_upper: Decimal
    physical_response_margin: Decimal
    certification_margin: Decimal
    measurement_backaction_status: str
    claim_bearing: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.exchange_id, field_name="exchange_id")
        validate_sha256(
            self.complete_unit_ids_sha256,
            field_name="complete_unit_ids_sha256",
        )
        if self.complete_unit_count <= 0:
            raise ValueError("metric/topology exchange requires complete units")
        for name in ("topology_success_count", "metric_success_count"):
            value = getattr(self, name)
            if not 0 <= value <= self.complete_unit_count:
                raise ValueError(f"{name} exceeds the complete-unit denominator")
        for name in (
            "delta",
            "simultaneous_lower",
            "simultaneous_upper",
            "physical_response_margin",
            "certification_margin",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        expected = Decimal(self.topology_success_count - self.metric_success_count) / Decimal(
            self.complete_unit_count
        )
        if self.delta != expected:
            raise ValueError("metric/topology delta differs from complete-unit counts")
        if not self.simultaneous_lower <= self.delta <= self.simultaneous_upper:
            raise ValueError("metric/topology interval does not contain its estimate")
        validate_nonempty(
            self.measurement_backaction_status,
            field_name="measurement_backaction_status",
        )


@dataclass(frozen=True, slots=True)
class IndependentSubstrateInterfaceConformanceFixture(CanonicalRecord):
    """Independently authored truth-known truth-known interface conformance external-interface shape."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-interface-conformance-fixture'

    fixture_id: str
    interface_shape: IndependentSubstrateInterfaceShape
    prediction_issued: bool
    required_operand_ids: tuple[str, ...]
    missing_operand_ids: tuple[str, ...]
    action_clock_ids: tuple[str, ...]
    receiver_count: int
    measurement_backaction_status: str
    unfavorable_response: bool
    domain_loss: bool
    attained_level: TargetLevel

    def __post_init__(self) -> None:
        validate_stable_id(self.fixture_id, field_name="fixture_id")
        require_sorted_unique_strings(
            self.required_operand_ids,
            field_name="required_operand_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.missing_operand_ids,
            field_name="missing_operand_ids",
        )
        if not set(self.missing_operand_ids).issubset(self.required_operand_ids):
            raise ValueError("missing operands must belong to the required roster")
        require_sorted_unique_strings(self.action_clock_ids, field_name="action_clock_ids")
        if self.receiver_count < 0:
            raise ValueError("receiver count must be nonnegative")
        validate_nonempty(
            self.measurement_backaction_status,
            field_name="measurement_backaction_status",
        )
        if self.unfavorable_response and self.domain_loss:
            raise ValueError("fixture separates unfavorable response from domain loss")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateInterfaceConformanceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-interface-conformance-result'

    result_id: str
    fixture: ObjectIdentity
    compatibility: IndependentSubstrateCompatibilityDisposition
    scientific_outcome: str
    receiver_relative_claim_eligible: bool
    action_ledger_complete: bool
    attained_level: TargetLevel
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_nonempty(self.scientific_outcome, field_name="scientific_outcome")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def classify_interface_fixture(
    fixture: IndependentSubstrateInterfaceConformanceFixture,
) -> IndependentSubstrateInterfaceConformanceResult:
    """Prove that only outcome-blind pre-issue facts can produce nonentry."""

    required_action_clocks = {
        "accepted",
        "applied",
        "realized",
        "requested",
    }
    action_complete = required_action_clocks.issubset(fixture.action_clock_ids)
    reasons: set[str] = set()
    if fixture.missing_operand_ids and not fixture.prediction_issued:
        compatibility = IndependentSubstrateCompatibilityDisposition.TARGET_OPERAND_INCOMPLETE
        outcome = "PRE_ISSUE_NONENTRY"
        reasons.add("PRE_ISSUE_REQUIRED_OPERAND_ABSENT")
    elif not action_complete and not fixture.prediction_issued:
        compatibility = IndependentSubstrateCompatibilityDisposition.TARGET_OPERAND_INCOMPLETE
        outcome = "PRE_ISSUE_NONENTRY"
        reasons.add("PRE_ISSUE_ACTION_CLOCK_INCOMPLETE")
    else:
        compatibility = IndependentSubstrateCompatibilityDisposition.COMPATIBLE
        if fixture.domain_loss:
            outcome = "POST_ISSUE_DOMAIN_LOSS"
            reasons.add("DOMAIN_LOSS_IS_SCIENTIFIC_OUTCOME")
        elif fixture.unfavorable_response:
            outcome = "POST_ISSUE_NEGATIVE"
            reasons.add("UNFAVORABLE_RESPONSE_IS_SCIENTIFIC_OUTCOME")
        else:
            outcome = "TRUTH_KNOWN_CONFORMANT"
    receiver_eligible = fixture.receiver_count >= 2 and fixture.measurement_backaction_status in {
        "BOUNDED",
        "MEASURED",
    }
    if not receiver_eligible:
        reasons.add("RECEIVER_RELATIVE_CLAIM_CLOSED")
    if fixture.prediction_issued and fixture.missing_operand_ids:
        reasons.add("POST_ISSUE_MISSINGNESS_CANNOT_BECOME_NONENTRY")
    return IndependentSubstrateInterfaceConformanceResult(
        result_id=f"{fixture.fixture_id}.result",
        fixture=ObjectIdentity.from_record(fixture.fixture_id, fixture),
        compatibility=compatibility,
        scientific_outcome=outcome,
        receiver_relative_claim_eligible=receiver_eligible,
        action_ledger_complete=action_complete,
        attained_level=fixture.attained_level,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class IndependentSubstrateConformanceCaseResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-conformance-case-result'

    case_id: str
    case_kind: IndependentSubstrateConformanceCaseKind
    observed_code: str
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        validate_nonempty(self.observed_code, field_name="observed_code")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.passed == bool(self.reason_codes):
            raise ValueError("truth-known interface conformance case reasons must be present exactly on failure")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateConformanceReport(CanonicalRecord):
    """Canonical truth-known report; never target or physical evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-conformance-report'

    report_id: str
    design_basis: ObjectIdentity
    evidence_world: EvidenceWorld
    case_results: tuple[IndependentSubstrateConformanceCaseResult, ...]
    independent_of_structural_recurrence_wrapper: bool
    all_passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        require_sorted_unique_ids(
            self.case_results,
            attribute="case_id",
            field_name="case_results",
        )
        if {value.case_kind for value in self.case_results} != set(IndependentSubstrateConformanceCaseKind):
            raise ValueError("truth-known interface conformance report does not contain the exact conformance roster")
        if self.evidence_world is not EvidenceWorld.TRUTH_KNOWN_GENERATED:
            raise ValueError("truth-known interface conformance report must remain truth-known generated evidence")
        if not self.independent_of_structural_recurrence_wrapper:
            raise ValueError("truth-known interface conformance fixtures must be independent of the structural recurrence wrapper")
        if self.all_passed != all(value.passed for value in self.case_results):
            raise ValueError("truth-known interface conformance report status is not fact-derived")


def execute_independent_substrate_truth_known_conformance(
    *,
    design_basis: ObjectIdentity,
) -> IndependentSubstrateConformanceReport:
    """Execute the closed truth-known interface conformance adversarial matrix without target outcomes."""

    results: list[IndependentSubstrateConformanceCaseResult] = []

    def record(kind: IndependentSubstrateConformanceCaseKind, observed: str, passed: bool) -> None:
        results.append(
            IndependentSubstrateConformanceCaseResult(
                case_id=f"independent-substrate-grounding.truth-known-conformance.{kind.value.lower().replace('_', '-')}",
                case_kind=kind,
                observed_code=observed,
                passed=passed,
                reason_codes=() if passed else ("TRUTH_KNOWN_EXPECTATION_MISMATCH",),
            )
        )

    def fixture(
        fixture_id: str,
        shape: IndependentSubstrateInterfaceShape,
        *,
        issued: bool = False,
        missing: tuple[str, ...] = (),
        clocks: tuple[str, ...] = ("accepted", "applied", "realized", "requested"),
        receivers: int = 2,
        backaction: str = "BOUNDED",
        unfavorable: bool = False,
        domain_loss: bool = False,
        level: TargetLevel = TargetLevel.ADMISSION,
    ) -> IndependentSubstrateInterfaceConformanceResult:
        return classify_interface_fixture(
            IndependentSubstrateInterfaceConformanceFixture(
                fixture_id=fixture_id,
                interface_shape=shape,
                prediction_issued=issued,
                required_operand_ids=("A", "D", "H", "R", "tau"),
                missing_operand_ids=missing,
                action_clock_ids=clocks,
                receiver_count=receivers,
                measurement_backaction_status=backaction,
                unfavorable_response=unfavorable,
                domain_loss=domain_loss,
                attained_level=level,
            )
        )

    shape_cases = (
        (
            IndependentSubstrateConformanceCaseKind.INTERFACE_IRREGULAR_GRAPH,
            IndependentSubstrateInterfaceShape.IRREGULAR_GRAPH,
            TargetLevel.ADMISSION,
        ),
        (
            IndependentSubstrateConformanceCaseKind.INTERFACE_CONTINUOUS_PHYSICAL_ARCHIVE,
            IndependentSubstrateInterfaceShape.CONTINUOUS_PHYSICAL_ARCHIVE,
            TargetLevel.LAW_QUALIFICATION,
        ),
        (
            IndependentSubstrateConformanceCaseKind.INTERFACE_ACTION_LEDGER,
            IndependentSubstrateInterfaceShape.ACTION_LEDGER,
            TargetLevel.ADMISSION,
        ),
        (
            IndependentSubstrateConformanceCaseKind.INTERFACE_VARIABLE_LEVEL,
            IndependentSubstrateInterfaceShape.VARIABLE_LEVEL,
            TargetLevel.ADMISSION,
        ),
    )
    for kind, shape, level in shape_cases:
        value = fixture(f"fixture.{kind.value.lower()}", shape, level=level)
        record(
            kind,
            value.scientific_outcome,
            value.scientific_outcome == "TRUTH_KNOWN_CONFORMANT",
        )

    receiver = fixture(
        "fixture.receiver-exchange-backaction",
        IndependentSubstrateInterfaceShape.RECEIVER_EXCHANGE,
        issued=True,
        backaction="UNMEASURED",
    )
    record(
        IndependentSubstrateConformanceCaseKind.INTERFACE_RECEIVER_EXCHANGE_BACKACTION,
        "CLAIM_OPEN" if receiver.receiver_relative_claim_eligible else "CLAIM_CLOSED",
        not receiver.receiver_relative_claim_eligible,
    )

    positive = fixture(
        "fixture.predictor-positive",
        IndependentSubstrateInterfaceShape.VARIABLE_LEVEL,
        issued=True,
    )
    record(
        IndependentSubstrateConformanceCaseKind.PREDICTOR_POSITIVE,
        positive.scientific_outcome,
        positive.scientific_outcome == "TRUTH_KNOWN_CONFORMANT",
    )
    opposed = fixture(
        "fixture.predictor-opposed",
        IndependentSubstrateInterfaceShape.VARIABLE_LEVEL,
        issued=True,
        unfavorable=True,
    )
    record(
        IndependentSubstrateConformanceCaseKind.PREDICTOR_OPPOSED,
        opposed.scientific_outcome,
        opposed.scientific_outcome == "POST_ISSUE_NEGATIVE",
    )

    wildcard = saturated_lookup_emit(
        cells=(),
        key_codes=(99,),
        wildcard_state_codes=(0, 1, 2, 3),
    )
    record(
        IndependentSubstrateConformanceCaseKind.COMPARATOR_WILDCARD,
        ".".join(str(value) for value in wildcard),
        wildcard == (0, 1, 2, 3),
    )
    lookup_cells = (
        IndependentSubstrateLookupCell(
            cell_id="independent-substrate-grounding.truth-known-conformance.lookup-cell",
            key_codes=(1, 2),
            emitted_state_codes=(3,),
        ),
    )
    lookup = saturated_lookup_emit(
        cells=lookup_cells,
        key_codes=(1, 2),
        wildcard_state_codes=(0, 1, 2, 3),
    )
    out_of_cell = saturated_lookup_emit(
        cells=lookup_cells,
        key_codes=(9, 9),
        wildcard_state_codes=(0, 1, 2, 3),
    )
    record(
        IndependentSubstrateConformanceCaseKind.COMPARATOR_SATURATED_LOOKUP,
        f"IN={lookup};OUT={out_of_cell}",
        lookup == (3,) and out_of_cell == (0, 1, 2, 3),
    )

    structural_recurrence_encoding = IndependentSubstrateComparatorEncoding(
        encoding_id='independent-substrate-grounding.truth-known-conformance.encoding.structural-recurrence',
        kind=IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE,
        selected_parameter_codes=(1,),
        denominator_dependency_ids=("D",),
        receiver_dependency_ids=("R",),
        lookup_cells=(),
        out_of_cell_rule="NOT_APPLICABLE",
    )
    wildcard_encoding = IndependentSubstrateComparatorEncoding(
        encoding_id="independent-substrate-grounding.truth-known-conformance.encoding.wildcard",
        kind=IndependentSubstrateComparatorKind.WILDCARD_ALL,
        selected_parameter_codes=(),
        denominator_dependency_ids=(),
        receiver_dependency_ids=(),
        lookup_cells=(),
        out_of_cell_rule="WILDCARD_ALL",
    )
    unsafe_structural_recurrence = IndependentSubstrateComparatorScore(
        score_id='independent-substrate-grounding.truth-known-conformance.score.unsafe-structural-recurrence',
        encoding=ObjectIdentity.from_record(structural_recurrence_encoding.encoding_id, structural_recurrence_encoding),
        unsafe_false_admission_count=1,
        categorical_mismatch_count=0,
        prediction_set_cardinality=1,
        scientific_description_bits=structural_recurrence_encoding.scientific_description_bits,
    )
    safe_comparator = IndependentSubstrateComparatorScore(
        score_id="independent-substrate-grounding.truth-known-conformance.score.safe-wildcard",
        encoding=ObjectIdentity.from_record(
            wildcard_encoding.encoding_id,
            wildcard_encoding,
        ),
        unsafe_false_admission_count=0,
        categorical_mismatch_count=2,
        prediction_set_cardinality=4,
        scientific_description_bits=wildcard_encoding.scientific_description_bits,
    )
    record(
        IndependentSubstrateConformanceCaseKind.UNSAFE_FALSE_ADMISSION,
        (
            "COMPARATOR_PRECEDES"
            if safe_comparator.lexicographic_key < unsafe_structural_recurrence.lexicographic_key
            else 'STRUCTURAL_RECURRENCE_PRECEDES'
        ),
        safe_comparator.lexicographic_key < unsafe_structural_recurrence.lexicographic_key,
    )

    incomplete = fixture(
        "fixture.incomplete-operand",
        IndependentSubstrateInterfaceShape.ACTION_LEDGER,
        missing=("A",),
    )
    record(
        IndependentSubstrateConformanceCaseKind.INCOMPLETE_OPERAND,
        incomplete.scientific_outcome,
        incomplete.scientific_outcome == "PRE_ISSUE_NONENTRY",
    )

    unit_hash = "1" * 64
    minimal = IndependentSubstrateDenominatorCandidate(
        candidate_id="independent-substrate-grounding.truth-known-conformance.denominator-minimal",
        factor_ids=("preparation",),
        stratum_ids=("all",),
        impossible=False,
        impossible_reason=None,
    )
    nonminimal = IndependentSubstrateDenominatorCandidate(
        candidate_id="independent-substrate-grounding.truth-known-conformance.denominator-nonminimal",
        factor_ids=("load", "preparation"),
        stratum_ids=("high", "low"),
        impossible=False,
        impossible_reason=None,
    )
    selected = select_minimal_denominator(
        tuple(
            IndependentSubstrateDenominatorDevelopmentAssessment(
                assessment_id=f"independent-substrate-grounding.truth-known-conformance.assessment.{candidate.candidate_id.split('.')[-1]}",
                candidate=candidate,
                same_complete_unit_ids_sha256=unit_hash,
                legal_action_alphabet_preserved=True,
                forecast_roster_preserved=True,
                policy_outputs_preserved=True,
                unsafe_error_count=0,
                simultaneous_precision_passed=True,
            )
            for candidate in (minimal, nonminimal)
        )
    )
    record(
        IndependentSubstrateConformanceCaseKind.NONMINIMAL_DENOMINATOR,
        selected.candidate.candidate_id,
        selected.candidate == minimal,
    )

    domain_loss = fixture(
        "fixture.post-issue-domain-loss",
        IndependentSubstrateInterfaceShape.VARIABLE_LEVEL,
        issued=True,
        domain_loss=True,
        level=TargetLevel.MEASUREMENT_READINESS,
    )
    record(
        IndependentSubstrateConformanceCaseKind.POST_ISSUE_DOMAIN_LOSS,
        domain_loss.scientific_outcome,
        domain_loss.scientific_outcome == "POST_ISSUE_DOMAIN_LOSS",
    )
    post_issue_missing = fixture(
        "fixture.post-issue-missingness",
        IndependentSubstrateInterfaceShape.ACTION_LEDGER,
        issued=True,
        missing=("A",),
    )
    record(
        IndependentSubstrateConformanceCaseKind.POST_ISSUE_MISSINGNESS,
        post_issue_missing.scientific_outcome,
        post_issue_missing.scientific_outcome != "PRE_ISSUE_NONENTRY"
        and "POST_ISSUE_MISSINGNESS_CANNOT_BECOME_NONENTRY" in post_issue_missing.reason_codes,
    )

    ordered = tuple(sorted(results, key=lambda value: value.case_id))
    return IndependentSubstrateConformanceReport(
        report_id="independent-substrate-grounding.truth-known-conformance.truth-known-conformance-report",
        design_basis=design_basis,
        evidence_world=EvidenceWorld.TRUTH_KNOWN_GENERATED,
        case_results=ordered,
        independent_of_structural_recurrence_wrapper=True,
        all_passed=all(value.passed for value in ordered),
    )


@dataclass(frozen=True, slots=True)
class IndependentImplementationDossier(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/independent-implementation-dossier'

    dossier_id: str
    slot: IndependentSubstrateTargetKind
    outcome_generator_owner: str
    outcome_generator_source_sha256: str
    pairwise_generator_disjoint: bool
    preparation_rosters_disjoint: bool
    target_runner_imports_structural_recurrence: bool
    prediction_precedes_outcome_access: bool
    analysis_implementation_statement: str
    evaluator_identity: str
    apparatus_site_count: int
    classification: IndependentSubstrateIndependenceClass
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.dossier_id, field_name="dossier_id")
        validate_nonempty(
            self.outcome_generator_owner,
            field_name="outcome_generator_owner",
        )
        validate_sha256(
            self.outcome_generator_source_sha256,
            field_name="outcome_generator_source_sha256",
        )
        validate_nonempty(
            self.analysis_implementation_statement,
            field_name="analysis_implementation_statement",
        )
        validate_stable_id(self.evaluator_identity, field_name="evaluator_identity")
        if self.apparatus_site_count < 0:
            raise ValueError("apparatus/site count must be nonnegative")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.slot is IndependentSubstrateTargetKind.BOPTEST_BRIDGE:
            expected = IndependentSubstrateIndependenceClass.DONOR_VISIBLE_BRIDGE
        elif self.slot is IndependentSubstrateTargetKind.NREL_INVERTER:
            expected = IndependentSubstrateIndependenceClass.PHYSICAL_EXTERNAL_APPARATUS
        elif not self.prediction_precedes_outcome_access:
            expected = IndependentSubstrateIndependenceClass.CONTAMINATED
        elif all(
            (
                self.pairwise_generator_disjoint,
                self.preparation_rosters_disjoint,
                not self.target_runner_imports_structural_recurrence,
            )
        ):
            expected = IndependentSubstrateIndependenceClass.NOVEL_EXTERNAL_IMPLEMENTATION
        else:
            expected = IndependentSubstrateIndependenceClass.UNEVALUABLE
        if self.classification is not expected:
            raise ValueError("independence classification differs from its vector")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateTargetTerminalHandoff(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-target-terminal-handoff'

    handoff_id: str
    slot: IndependentSubstrateTargetKind
    evidence_world: EvidenceWorld
    attained_level: TargetLevel
    independence_class: IndependentSubstrateIndependenceClass
    evaluation_eligible: bool
    categorical_support: bool
    decisive_opposition: bool
    unsafe_false_admission_count: int
    action_ontology_clock_error_count: int
    panel_envelope_limited: bool
    structural_recurrence_restrictiveness_supported: bool
    comparator_tied_or_won: bool
    structural_recurrence_less_safe_or_exact_than_comparator: bool
    metric_topology_exchanges: tuple[IndependentSubstrateMetricTopologyExchange, ...]
    physical_consistency: str
    prediction_receipt_sha256: str | None
    match_receipt_sha256: str | None
    maximum_claim_ceiling: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.handoff_id, field_name="handoff_id")
        for name in (
            "unsafe_false_admission_count",
            "action_ontology_clock_error_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        require_sorted_unique_ids(
            self.metric_topology_exchanges,
            attribute="exchange_id",
            field_name="metric_topology_exchanges",
        )
        for name in ("prediction_receipt_sha256", "match_receipt_sha256"):
            value = getattr(self, name)
            if value is not None:
                validate_sha256(value, field_name=name)
        validate_nonempty(self.maximum_claim_ceiling, field_name="maximum_claim_ceiling")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.categorical_support and self.decisive_opposition:
            raise ValueError("one target cannot simultaneously support and oppose")
        if self.panel_envelope_limited and self.categorical_support:
            raise ValueError("panel-limited target cannot support recurrence")
        if self.slot is not IndependentSubstrateTargetKind.NREL_INVERTER and (
            self.physical_consistency != "NOT_APPLICABLE"
        ):
            raise ValueError("only NREL can populate physical consistency")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateAxisAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-axis-adjudication'

    adjudication_id: str
    target_handoff_sha256s: tuple[str, ...]
    independent_axis: IndependentSubstrateIndependentAxis
    restrictiveness_axis: IndependentSubstrateRestrictivenessAxis
    topology_metric_axis: IndependentSubstrateTopologyMetricAxis
    physical_axis: IndependentSubstratePhysicalAxis
    reason_codes: tuple[str, ...]
    prospective_physical_axis_present: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_strings(
            self.target_handoff_sha256s,
            field_name="target_handoff_sha256s",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.prospective_physical_axis_present:
            raise ValueError("independent substrate grounding excludes a prospective physical axis")


def _novel(
    handoffs: tuple[IndependentSubstrateTargetTerminalHandoff, ...],
) -> tuple[IndependentSubstrateTargetTerminalHandoff, ...]:
    by_slot = {value.slot: value for value in handoffs}
    return tuple(
        by_slot[slot]
        for slot in (IndependentSubstrateTargetKind.FREEGSNKE, IndependentSubstrateTargetKind.GRID2OP)
        if slot in by_slot
    )


def adjudicate_axes(
    handoffs: tuple[IndependentSubstrateTargetTerminalHandoff, ...],
) -> IndependentSubstrateAxisAdjudication:
    """Apply counterexample-first, noncompensating four-axis precedence."""

    if len({value.slot for value in handoffs}) != len(handoffs):
        raise ValueError("cross-target handoffs must have unique slots")
    novel = _novel(handoffs)
    eligible_novel = tuple(value for value in novel if value.evaluation_eligible)
    decisive = tuple(
        value
        for value in eligible_novel
        if (
            value.decisive_opposition
            or value.unsafe_false_admission_count
            or value.action_ontology_clock_error_count
        )
    )
    reasons: set[str] = set()
    if decisive:
        independent = IndependentSubstrateIndependentAxis.OPPOSED
        reasons.add("DECISIVE_NOVEL_TARGET_COUNTEREXAMPLE")
    elif len(novel) == 2 and all(
        value.independence_class is IndependentSubstrateIndependenceClass.NOVEL_EXTERNAL_IMPLEMENTATION
        and value.attained_level in {TargetLevel.ADMISSION, TargetLevel.PROSPECTIVE_USE, TargetLevel.CONTROLLER_USE_CEILING}
        and value.evaluation_eligible
        and value.categorical_support
        and not value.panel_envelope_limited
        for value in novel
    ):
        independent = IndependentSubstrateIndependentAxis.SUPPORTED
    elif eligible_novel:
        independent = IndependentSubstrateIndependentAxis.MIXED
        reasons.add("ONE_OR_MORE_NOVEL_TARGETS_INCOMPLETE_OR_NONSUPPORTING")
    else:
        independent = IndependentSubstrateIndependentAxis.UNEVALUABLE
        reasons.add("NO_ELIGIBLE_NOVEL_TARGET_HANDOFF")

    if any(value.structural_recurrence_less_safe_or_exact_than_comparator for value in eligible_novel):
        restrictiveness = IndependentSubstrateRestrictivenessAxis.OPPOSED
        reasons.add('STRUCTURAL_RECURRENCE_LESS_SAFE_OR_EXACT_THAN_COMPARATOR')
    elif len(eligible_novel) == 2 and all(
        value.structural_recurrence_restrictiveness_supported for value in eligible_novel
    ):
        restrictiveness = IndependentSubstrateRestrictivenessAxis.SUPPORTED
    elif eligible_novel and any(value.comparator_tied_or_won for value in eligible_novel):
        restrictiveness = IndependentSubstrateRestrictivenessAxis.NOT_DISTINGUISHED
        reasons.add("SIMPLE_OR_SATURATED_COMPARATOR_NOT_BEATEN")
    else:
        restrictiveness = IndependentSubstrateRestrictivenessAxis.UNEVALUABLE
        reasons.add("COMPARATOR_AXIS_INCOMPLETE")

    exchanges = tuple(
        exchange
        for target in eligible_novel
        for exchange in target.metric_topology_exchanges
        if exchange.claim_bearing
    )
    if any(
        target.decisive_opposition
        or any(exchange.simultaneous_upper < 0 for exchange in target.metric_topology_exchanges)
        for target in eligible_novel
    ):
        topology_metric = IndependentSubstrateTopologyMetricAxis.METRIC_STRONGER_OR_TOPOLOGY_OPPOSED
        reasons.add("METRIC_STRONGER_OR_TOPOLOGY_OPPOSED_IN_TARGET")
    elif len(eligible_novel) == 2 and all(
        target.metric_topology_exchanges
        and all(
            exchange.claim_bearing and exchange.simultaneous_lower > 0
            for exchange in target.metric_topology_exchanges
        )
        and target.unsafe_false_admission_count == 0
        for target in eligible_novel
    ):
        topology_metric = IndependentSubstrateTopologyMetricAxis.TOPOLOGY_STRONGER
    else:
        topology_metric = IndependentSubstrateTopologyMetricAxis.EQUIVALENT_OR_UNEVALUABLE
        reasons.add("TOPOLOGY_METRIC_CONJUNCTION_NOT_ESTABLISHED")
    if not exchanges:
        reasons.add("NO_CLAIM_BEARING_METRIC_TOPOLOGY_EXCHANGE")

    nrel = next(
        (value for value in handoffs if value.slot is IndependentSubstrateTargetKind.NREL_INVERTER),
        None,
    )
    if nrel is None or nrel.physical_consistency == "UNEVALUABLE":
        physical = IndependentSubstratePhysicalAxis.UNEVALUABLE
        reasons.add("PHYSICAL_ARCHIVE_UNEVALUABLE_OR_ABSENT")
    elif nrel.physical_consistency == "OPPOSED":
        physical = IndependentSubstratePhysicalAxis.OPPOSED
    elif nrel.physical_consistency == "SUPPORTED":
        physical = IndependentSubstratePhysicalAxis.SUPPORTED
    else:
        physical = IndependentSubstratePhysicalAxis.MIXED

    return IndependentSubstrateAxisAdjudication(
        adjudication_id="independent-substrate-grounding.four-axis-adjudication",
        target_handoff_sha256s=tuple(sorted(value.fingerprint() for value in handoffs)),
        independent_axis=independent,
        restrictiveness_axis=restrictiveness,
        topology_metric_axis=topology_metric,
        physical_axis=physical,
        reason_codes=tuple(sorted(reasons)),
        prospective_physical_axis_present=False,
    )


def scientific_design_basis(
    *,
    method_port_freeze: ObjectIdentity,
    applicability_overlay: ObjectIdentity,
) -> IndependentSubstrateScientificDesignBasis:
    """Return the exact outcome-blind outcome-blind scientific grammar design basis."""

    slots = (
        IndependentSubstrateTargetSlot(
            slot_id="independent-substrate-grounding.slot.boptest-bridge",
            slot=IndependentSubstrateTargetKind.BOPTEST_BRIDGE,
            evidence_world=EvidenceWorld.RESETTABLE_SIMULATOR,
            fixed_independence_class=IndependentSubstrateIndependenceClass.DONOR_VISIBLE_BRIDGE,
            required_support_level=TargetLevel.MEASUREMENT_READINESS,
            counts_for_independent_recurrence=False,
            conditional_prospective_validation_allowed=True,
            substitution_allowed=False,
        ),
        IndependentSubstrateTargetSlot(
            slot_id="independent-substrate-grounding.slot.freegsnke",
            slot=IndependentSubstrateTargetKind.FREEGSNKE,
            evidence_world=EvidenceWorld.RESETTABLE_SIMULATOR,
            fixed_independence_class=None,
            required_support_level=TargetLevel.ADMISSION,
            counts_for_independent_recurrence=True,
            conditional_prospective_validation_allowed=True,
            substitution_allowed=False,
        ),
        IndependentSubstrateTargetSlot(
            slot_id="independent-substrate-grounding.slot.grid2op",
            slot=IndependentSubstrateTargetKind.GRID2OP,
            evidence_world=EvidenceWorld.RESETTABLE_SIMULATOR,
            fixed_independence_class=None,
            required_support_level=TargetLevel.ADMISSION,
            counts_for_independent_recurrence=True,
            conditional_prospective_validation_allowed=True,
            substitution_allowed=False,
        ),
        IndependentSubstrateTargetSlot(
            slot_id="independent-substrate-grounding.slot.nrel-inverter",
            slot=IndependentSubstrateTargetKind.NREL_INVERTER,
            evidence_world=EvidenceWorld.RETROSPECTIVE_PHYSICAL_ARCHIVE,
            fixed_independence_class=IndependentSubstrateIndependenceClass.PHYSICAL_EXTERNAL_APPARATUS,
            required_support_level=TargetLevel.LAW_QUALIFICATION,
            counts_for_independent_recurrence=False,
            conditional_prospective_validation_allowed=False,
            substitution_allowed=False,
        ),
    )
    state_ids = (
        "ACTION",
        "ADMIT",
        "DOMAIN_LOSS",
        "FAILURE",
        "HOLD",
        "NONENTRY",
        "SINK",
        "SUPPORT",
        "UNSAFE",
        "VALID",
    )
    grammar = IndependentSubstrateForecastAlphabetGrammar(
        grammar_id="independent-substrate-grounding.forecast-alphabet-grammar",
        role_vocabulary=tuple(
            sorted(
                (
                    "action",
                    "admission",
                    "denominator",
                    "failure",
                    "history",
                    "hold",
                    "horizon",
                    "nonentry",
                    "receiver",
                    "sink",
                    "support",
                    "validity",
                )
            )
        ),
        constructor_ids=tuple(
            sorted(
                (
                    "ACTION_OR_HOLD",
                    "BOOLEAN",
                    "CLOSED_ENUMERATION",
                    "PREREQUISITE_NONENTRY",
                    "SUBSET_OF_CLOSED_ENUMERATION",
                )
            )
        ),
        common_state_codes=tuple(
            IndependentSubstrateForecastStateCode(
                state_id=f"independent-substrate-grounding.forecast-state.{state_id.lower().replace('_', '-')}",
                integer_code=index,
            )
            for index, state_id in enumerate(state_ids)
        ),
        prohibited_fallbacks=tuple(
            sorted(
                (
                    "ADD_OR_REMOVE_STATE_AFTER_DEVELOPMENT",
                    "BORROW_NEAREST_LOOKUP_CELL",
                    "MAP_UNSEEN_TO_FAVORABLE_DEFAULT",
                    "POST_EVALUATION_STATE_MERGE",
                )
            )
        ),
        forecast_denominator_rules=tuple(
            sorted(
                (
                    "all-issued-complete-evaluation-units",
                    "no-post-issue-outcome-exclusion",
                    "panel-limited-remains-in-issued-denominator",
                    "pre-issue-nonentry-excluded-with-typed-reason",
                    "target-local-no-cross-target-pooling",
                )
            )
        ),
        target_instantiation_stage="G2_BEFORE_DEVELOPMENT",
    )
    mapping_grammar = IndependentSubstrateMappingCandidateGrammar(
        grammar_id="independent-substrate-grounding.mapping-candidate-grammar",
        required_structural_recurrence_role_ids=("A", "D", "H", "R", "tau"),
        constructor_ids=tuple(sorted(IndependentSubstrateMappingConstructor, key=lambda value: value.value)),
        maximum_candidates_per_target=32,
        roster_instantiation_stage="G2_BEFORE_DEVELOPMENT",
        selection_stage="DEVELOPMENT_ONLY_BEFORE_PREDICTION_ISSUE",
        selection_precedence=(
            "unsafe-false-admission",
            "categorical-mismatch-count",
            "prediction-set-cardinality",
            "canonical-description-bits",
            "stable-candidate-id",
        ),
        prohibited_operations=tuple(
            sorted(
                (
                    "add-candidate-after-development",
                    "derive-binding-from-evaluation-outcome",
                    "merge-roles-after-evaluation",
                    "select-by-evaluation-score",
                    "substitute-target-native-unit-or-frame",
                )
            )
        ),
    )
    codebook = IndependentSubstrateComparatorCodebook(
        codebook_id="independent-substrate-grounding.comparator-codebook",
        comparator_kinds=tuple(sorted(IndependentSubstrateComparatorKind, key=lambda value: value.value)),
        score_precedence=(
            "unsafe-false-admission",
            "categorical-mismatch-count",
            "prediction-set-cardinality",
            "canonical-description-bits",
        ),
        saturated_out_of_cell_rule="WILDCARD_ALL",
        description_length_rule='SCIENTIFIC_DESCRIPTION_CODEBOOK_BITS',
        target_averaging_allowed=False,
    )
    contrast_text = {
        "D": "split-merge-or-preparation-exchange-with-other-operands-fixed",
        "H": "history-inclusion-omission-or-lag-exchange-preserving-causal-cutoff",
        "A": "two-actions-or-action-versus-hold-on-common-preparation",
        "R": "two-receiver-or-observation-maps-over-shared-preparations",
        "tau": "two-predeclared-horizons-when-horizon-dependence-is-claimed",
    }
    return IndependentSubstrateScientificDesignBasis(
        design_basis_id="independent-substrate-grounding.scientific-design-basis",
        method_port_freeze=method_port_freeze,
        applicability_overlay=applicability_overlay,
        target_slots=slots,
        mapping_grammar=mapping_grammar,
        forecast_grammar=grammar,
        comparator_codebook=codebook,
        power_and_uncertainty_grammar=IndependentSubstratePowerAndUncertaintyGrammar(
            grammar_id="independent-substrate-grounding.power-and-uncertainty-grammar",
            complete_unit_role="TARGET_PHYSICAL_PREPARATION_OR_NATIVE_EPISODE",
            resampling_unit_role="COMPLETE_UNIT",
            jointly_powered_axis_ids=(
                "observation-grid",
                "panel",
                "receiver-coordinate",
            ),
            simultaneous_family_ids=tuple(
                sorted(
                    (
                        "all-decisive-falsifiers",
                        "all-primary-estimands",
                        "all-topology-metric-exchanges",
                    )
                )
            ),
            nested_rows_or_views_inflate_replication=False,
            cross_target_pooling_allowed=False,
            panel_limited_postissue_disposition="PANEL_LIMITED_NOT_NONENTRY",
            physical_response_and_certification_margins_distinct=True,
            freeze_stage="G3_BEFORE_PREDICTION_ISSUE",
        ),
        claim_contrasts=tuple(
            IndependentSubstrateClaimContrastRequirement(
                contrast_id=f"independent-substrate-grounding.claim-contrast.{operand.lower()}",
                law_operand=operand,
                minimum_contrast=text,
                absence_narrows_claim=True,
            )
            for operand, text in sorted(contrast_text.items())
        ),
        action_clock_role_ids=("accepted", "applied", "realized", "requested"),
        mandatory_hold_rules=tuple(
            sorted(
                (
                    "hold-on-action-clock-ambiguity",
                    "hold-on-authority-absence",
                    "hold-on-domain-loss",
                    "hold-on-invalid-receiver-or-gauge",
                    'hold-on-admission-failure',
                    "hold-on-support-or-reachability-absence",
                )
            )
        ),
        denominator_minimality_order=(
            "source-qualified-factor-and-stratum-count",
            "canonical-comparator-codebook-byte-length",
            "stable-candidate-id",
        ),
        topology_metric_rules=tuple(
            sorted(
                (
                    "complete-predeclared-exchange-roster",
                    "identical-complete-unit-roster",
                    "minimum-delta-across-exchanges",
                    "no-cross-target-native-metric-pooling",
                    "physical-response-margin-separate-from-certification-margin",
                    "simultaneous-lower-bound-positive-for-every-exchange",
                )
            )
        ),
        nonentry_precedence=tuple(
            sorted(
                (
                    "pre-issue-missing-operand-may-yield-target-operand-incomplete",
                    "pre-issue-unmappable-source-may-yield-method-domain-unresolved",
                    "post-issue-domain-loss-is-scientific-outcome-not-nonentry",
                    "post-issue-unfavorable-response-is-negative-not-nonentry",
                )
            )
        ),
        target_issue_preconditions=tuple(
            sorted(
                (
                    "25-entry-and-48-gap-closure",
                    'development-evaluation-prospective-rosters-disjoint',
                    "exact-forecast-alphabet-and-denominator",
                    "independence-dossier",
                    "joint-panel-grid-receiver-simultaneous-power",
                    "mapping-candidate-and-denominator-lattice-frozen",
                    "outcome-blind-approval-and-storage-authority",
                )
            )
        ),
        axis_ids=tuple(
            sorted(
                (
                    "INDEPENDENT_IMPLEMENTATION_RECURRENCE",
                    "PHYSICAL_STRUCTURE_CONSISTENCY",
                    "PREDICTIVE_RESTRICTIVENESS",
                    "TOPOLOGY_VS_METRIC_TRANSPORT",
                )
            )
        ),
        excluded_scope_ids=tuple(
            sorted(
                (
                    "CPAIM_AND_ALL_PROSPECTIVE_PHYSICAL_WORK",
                    "NATURE_OUTLINE_MANUSCRIPT_FIGURES_AND_SUBMISSION",
                    "SCALE_COVARIANCE_COMPOSITION_AND_FIXED_POINT_CLAIMS",
                )
            )
        ),
        unresolved_in_scope_scientific_choice_ids=(),
        generated_target_parameters_imported=False,
    )


__all__ = [
    'IndependentImplementationDossier',
    'IndependentSubstrateAxisAdjudication',
    'IndependentSubstrateClaimContrastRequirement',
    'IndependentSubstrateComparatorCodebook',
    'IndependentSubstrateComparatorEncoding',
    'IndependentSubstrateComparatorKind',
    'IndependentSubstrateComparatorScore',
    'IndependentSubstrateCompatibilityDisposition',
    'IndependentSubstrateDenominatorCandidate',
    'IndependentSubstrateDenominatorDevelopmentAssessment',
    'IndependentSubstrateForecastAlphabetGrammar',
    'IndependentSubstrateForecastStateCode',
    'IndependentSubstrateIndependentAxis',
    'IndependentSubstrateIndependenceClass',
    'IndependentSubstrateInterfaceConformanceFixture',
    'IndependentSubstrateInterfaceConformanceResult',
    'IndependentSubstrateInterfaceShape',
    'IndependentSubstrateLookupCell',
    'IndependentSubstrateConformanceCaseKind',
    'IndependentSubstrateConformanceCaseResult',
    'IndependentSubstrateConformanceReport',
    'IndependentSubstrateMappingCandidateGrammar',
    'IndependentSubstrateMappingConstructor',
    'IndependentSubstrateMappingDevelopmentAssessment',
    'IndependentSubstrateMetricTopologyExchange',
    'IndependentSubstratePhysicalAxis',
    'IndependentSubstratePowerAndUncertaintyGrammar',
    'IndependentSubstrateRestrictivenessAxis',
    'IndependentSubstrateScientificDesignBasis',
    'IndependentSubstrateTargetKind',
    'IndependentSubstrateTargetSlot',
    'IndependentSubstrateTargetMappingCandidate',
    'IndependentSubstrateTargetPredictionContract',
    'IndependentSubstrateTargetTerminalHandoff',
    'IndependentSubstrateTopologyMetricAxis',
    "adjudicate_axes",
    "classify_interface_fixture",
    'execute_independent_substrate_truth_known_conformance',
    "saturated_lookup_emit",
    "scientific_design_basis",
    "select_mapping_candidate",
    "select_minimal_denominator",
    'IndependentSubstrateRoleBinding',
]
