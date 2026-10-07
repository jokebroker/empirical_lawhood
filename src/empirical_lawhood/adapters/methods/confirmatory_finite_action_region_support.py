"""Prepared-denominator/local-support production for confirmatory finite actions.

This additive seam reuses the current confirmatory source validation and pure
Student arithmetic, but emits response rows only for three disjoint local
regions.  The prepared denominator remains the shared action/preparation
identity and deliberately has no response row.  Prefix G1 and tier-only G1R
are noncompensating method evidence; this module never finalizes a law.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ActionWordSupportStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawMethodKind, LawQualificationResult
from empirical_lawhood.kernel.laws import LawRepresentationKind
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import HorizonSpec, parse_utc_timestamp
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.identification_evidence import (
    IdentificationEvidenceManifest,
    IdentificationEvidenceProjection,
)
from empirical_lawhood.planning.identification_evidence_extensions import IdentificationEvidenceProjectionExtension
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority

from .confirmatory_finite_action import CONFIRMATORY_FINITE_ACTION_KEY, CONFIRMATORY_FINITE_ACTION_VERSION, CONFIRMATORY_REGION_SUPPORT_SPEC_SCHEMA, ConfirmatoryActionRole, ConfirmatoryActionRoleBinding, ConfirmatoryFiniteActionBindingReceipt, ConfirmatoryFiniteActionConfig, ConfirmatoryFiniteActionDesign, ConfirmatoryPrefixSpec, ConfirmatoryResponseQuantity, _ValidatedConfirmatoryOperands, bind_confirmatory_finite_action_design, _median, _receipt, _student_interval, _validated_confirmatory_operands, _value
from .contracts import ComponentUncertaintyFamilyAssessment, CandidateFamilyLedger, CandidateFamilyMember, CandidateEvaluatorImplementation, CandidateMethodEvidenceReceipt, CandidateRosterDisposition, FiniteActionCellDisposition, FiniteActionCompatibilitySetExtension, FiniteActionNativeValue, FiniteActionResponseEntry, LawCandidateAxisMap, LawCandidateEvidence, LawObligationTemplate
from .evidence_projection import TaggedIdentificationProjectionPlan
from .finite_action_identification import FiniteActionCandidateScaffold, FiniteActionProductionCell, FiniteActionProductionDisposition
from .law_assessment import (
    CandidateFamilyAssembler,
    CandidatePayloadPublisher,
    LawAssessmentAssembler,
    ResponseLawQualificationService,
)
from .qualification_profiles import ComponentQualificationProfile


CONFIRMATORY_LOCAL_REGION_FAMILY_ALPHA = Decimal("0.025")
CONFIRMATORY_LOCAL_REGION_PER_TEST_TAIL_PROBABILITY = Decimal("0.004166666666666666666666666667")
CONFIRMATORY_LOCAL_REGION_DEGREES_OF_FREEDOM = 5
CONFIRMATORY_LOCAL_REGION_CRITICAL_VALUE = Decimal("4.2193091158834291")
CONFIRMATORY_LOCAL_REGION_MATERIALITY = Decimal("0.0017")
CONFIRMATORY_LOCAL_REGION_UNIT_COUNT = 6
CONFIRMATORY_LOCAL_REGION_MINIMUM_POSITIVE_UNITS = 5
CONFIRMATORY_LOCAL_REGION_TEST_COUNT = 6

_PREFIX_CRITICALS = (
    (1, 5, CONFIRMATORY_LOCAL_REGION_CRITICAL_VALUE),
    (2, 11, Decimal("3.2081223331681157")),
    (3, 17, Decimal("2.9840416584797573")),
)


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionTierRegionBinding(CanonicalRecord):
    """One preissued tier-to-local map and its six physical units."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-finite-action-tier-region-binding'

    binding_id: str
    tier_id: str
    local_support_id: str
    admission_support_cell_id: str
    physical_independent_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("binding_id", self.binding_id),
            ("tier_id", self.tier_id),
            ("local_support_id", self.local_support_id),
            ("admission_support_cell_id", self.admission_support_cell_id),
        ):
            validate_stable_id(value, field_name=field_name)
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
            allow_empty=False,
        )
        if len(self.physical_independent_unit_ids) != CONFIRMATORY_LOCAL_REGION_UNIT_COUNT:
            raise ValueError("confirmatory tier region requires exactly six physical units")
        if self.local_support_id != self.admission_support_cell_id:
            raise ValueError("confirmatory local support must equal its current admission cell ID")


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionRegionSupportSpec(CanonicalRecord):
    """Outcome-blind three-region G1R design with no observation or result."""

    SCHEMA: ClassVar[str] = CONFIRMATORY_REGION_SUPPORT_SPEC_SCHEMA

    spec_id: str
    prepared_denominator_id: str
    chart_id: str
    horizon: ObjectIdentity
    retained_history_ids: tuple[str, ...]
    action_words: tuple[OccurrenceActionWord, ...]
    action_role_bindings: tuple[ConfirmatoryActionRoleBinding, ...]
    axis_map: LawCandidateAxisMap
    tier_region_bindings: tuple[ConfirmatoryFiniteActionTierRegionBinding, ...]
    prefixes: tuple[ConfirmatoryPrefixSpec, ...]
    response_quantity: ConfirmatoryResponseQuantity
    information_cutoff_id: str
    region_family_id: str
    family_alpha: Decimal
    simultaneous_test_count: int
    per_test_tail_probability: Decimal
    degrees_of_freedom: int
    critical_value: Decimal
    materiality: Decimal
    minimum_complete_units: int
    minimum_positive_units: int
    median_materiality: Decimal
    producer_implementation_id: str
    producer_implementation_sha256: str
    interpolation_allowed: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for field_name, value in (
            ("spec_id", self.spec_id),
            ("prepared_denominator_id", self.prepared_denominator_id),
            ("chart_id", self.chart_id),
            ("information_cutoff_id", self.information_cutoff_id),
            ("region_family_id", self.region_family_id),
            ("producer_implementation_id", self.producer_implementation_id),
        ):
            validate_stable_id(value, field_name=field_name)
        validate_sha256(
            self.producer_implementation_sha256,
            field_name="producer_implementation_sha256",
        )
        if self.horizon.object_schema != HorizonSpec.SCHEMA:
            raise ValueError("confirmatory region support requires an exact horizon")
        require_sorted_unique_strings(
            self.retained_history_ids,
            field_name="retained_history_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        require_sorted_unique_ids(
            self.action_role_bindings,
            attribute="binding_id",
            field_name="action_role_bindings",
        )
        require_sorted_unique_ids(
            self.tier_region_bindings,
            attribute="binding_id",
            field_name="tier_region_bindings",
        )
        require_sorted_unique_ids(
            self.prefixes,
            attribute="prefix_id",
            field_name="prefixes",
        )
        if len(self.action_words) != 4 or len(self.action_role_bindings) != 4:
            raise ValueError("confirmatory region support requires all four action roles")
        word_ids = {value.word_id for value in self.action_words}
        if {value.action_word_id for value in self.action_role_bindings} != word_ids:
            raise ValueError("confirmatory region action roles differ from the word chart")
        if {value.role for value in self.action_role_bindings} != set(ConfirmatoryActionRole):
            raise ValueError("confirmatory region action-role roster differs")
        if any(
            value.denominator_id != self.prepared_denominator_id
            or value.horizon_id != self.horizon.object_id
            or value.retained_history_id not in self.retained_history_ids
            for value in self.action_words
        ):
            raise ValueError("confirmatory region words change denominator/history/horizon")
        if (
            len(self.axis_map.bindings) != 2
            or len(self.axis_map.denominator_member_ids) != 2
            or any(len(value.qualification_view_ids) != 1 for value in self.axis_map.bindings)
        ):
            raise ValueError("confirmatory region support requires two exact member/view axes")
        if len(self.tier_region_bindings) != 3:
            raise ValueError("confirmatory region support requires exactly three tier maps")
        tier_ids = {value.tier_id for value in self.tier_region_bindings}
        local_ids = {value.local_support_id for value in self.tier_region_bindings}
        if len(tier_ids) != 3 or len(local_ids) != 3:
            raise ValueError("confirmatory tier/local maps must be one-to-one")
        unit_ids = tuple(
            sorted(
                unit_id
                for binding in self.tier_region_bindings
                for unit_id in binding.physical_independent_unit_ids
            )
        )
        if len(unit_ids) != 18 or len(set(unit_ids)) != 18:
            raise ValueError("confirmatory tier regions require 18 distinct physical units")
        ordered_prefixes = tuple(sorted(self.prefixes, key=lambda value: value.order_index))
        if len(ordered_prefixes) != 3:
            raise ValueError("confirmatory region support requires the exact J1--J3 prefixes")
        cumulative_tiers: set[str] = set()
        for prefix, (order, degrees_of_freedom, critical) in zip(
            ordered_prefixes,
            _PREFIX_CRITICALS,
            strict=True,
        ):
            current = set(prefix.constituent_tier_ids)
            if (
                prefix.order_index != order
                or not cumulative_tiers < current
                or not current <= tier_ids
                or len(current) != order
                or prefix.degrees_of_freedom != degrees_of_freedom
                or prefix.one_sided_tail_probability
                != CONFIRMATORY_LOCAL_REGION_PER_TEST_TAIL_PROBABILITY
                or prefix.one_sided_critical_value != critical
            ):
                raise ValueError("confirmatory G1 prefix constants or tier closure differ")
            cumulative_tiers = current
        if cumulative_tiers != tier_ids:
            raise ValueError("confirmatory J3 prefix omits a tier")
        prefix_ids = {value.prefix_id for value in self.prefixes}
        if (
            self.prepared_denominator_id in local_ids | prefix_ids | tier_ids
            or local_ids & prefix_ids
            or local_ids & tier_ids
            or prefix_ids & tier_ids
        ):
            raise ValueError("confirmatory prepared/local/prefix/tier identities alias")
        for field_name, decimal_value in (
            ("family_alpha", self.family_alpha),
            ("per_test_tail_probability", self.per_test_tail_probability),
            ("critical_value", self.critical_value),
            ("materiality", self.materiality),
            ("median_materiality", self.median_materiality),
        ):
            validate_decimal(
                decimal_value,
                field_name=field_name,
                minimum=Decimal(0),
            )
        if (
            self.family_alpha != CONFIRMATORY_LOCAL_REGION_FAMILY_ALPHA
            or self.simultaneous_test_count != CONFIRMATORY_LOCAL_REGION_TEST_COUNT
            or self.per_test_tail_probability != CONFIRMATORY_LOCAL_REGION_PER_TEST_TAIL_PROBABILITY
            or self.degrees_of_freedom != CONFIRMATORY_LOCAL_REGION_DEGREES_OF_FREEDOM
            or self.critical_value != CONFIRMATORY_LOCAL_REGION_CRITICAL_VALUE
            or self.materiality != CONFIRMATORY_LOCAL_REGION_MATERIALITY
            or self.minimum_complete_units != CONFIRMATORY_LOCAL_REGION_UNIT_COUNT
            or self.minimum_positive_units != CONFIRMATORY_LOCAL_REGION_MINIMUM_POSITIVE_UNITS
            or self.median_materiality != CONFIRMATORY_LOCAL_REGION_MATERIALITY
        ):
            raise ValueError("confirmatory G1R constants differ from the six-test family")
        if self.interpolation_allowed:
            raise ValueError("confirmatory local support cannot interpolate")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("confirmatory region-support spec must be outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("confirmatory region-support spec must remain prospective")

    @property
    def physical_independent_unit_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                unit_id
                for binding in self.tier_region_bindings
                for unit_id in binding.physical_independent_unit_ids
            )
        )

    @property
    def local_support_ids(self) -> tuple[str, ...]:
        return tuple(sorted(value.local_support_id for value in self.tier_region_bindings))

    @property
    def prefix_ids(self) -> tuple[str, ...]:
        return tuple(sorted(value.prefix_id for value in self.prefixes))


class ConfirmatoryRegionalTestDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionUnitEffect(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-finite-action-unit-effect'

    effect_id: str
    physical_independent_unit_id: str
    value: Decimal
    native_unit: str
    native_frame_id: str
    clock_id: str

    def __post_init__(self) -> None:
        for field_name, identifier in (
            ("effect_id", self.effect_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
            ("native_frame_id", self.native_frame_id),
            ("clock_id", self.clock_id),
        ):
            validate_stable_id(identifier, field_name=field_name)
        validate_decimal(self.value, field_name="value")
        if not self.native_unit:
            raise ValueError("confirmatory regional effect requires a native unit")


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionRegionalTest(CanonicalRecord):
    """One exact tier/member G1R test over six physical units."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-finite-action-regional-test'

    test_id: str
    family_id: str
    tier_id: str
    local_support_id: str
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_ids: tuple[str, ...]
    active_action_word: ObjectIdentity
    hold_action_word: ObjectIdentity
    expected_physical_unit_ids: tuple[str, ...]
    complete_physical_unit_ids: tuple[str, ...]
    excluded_physical_unit_ids: tuple[str, ...]
    effects: tuple[ConfirmatoryFiniteActionUnitEffect, ...]
    mean: Decimal | None
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    median: Decimal | None
    positive_effect_count: int
    degrees_of_freedom: int
    one_sided_tail_probability: Decimal
    one_sided_critical_value: Decimal
    materiality: Decimal
    action_exchange_passed: bool
    causal_obligations_passed: bool
    preservation_passed: bool
    disposition: ConfirmatoryRegionalTestDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("test_id", self.test_id),
            ("family_id", self.family_id),
            ("tier_id", self.tier_id),
            ("local_support_id", self.local_support_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
        ):
            validate_stable_id(value, field_name=field_name)
        for field_name, values in (
            ("qualification_view_ids", self.qualification_view_ids),
            ("expected_physical_unit_ids", self.expected_physical_unit_ids),
            ("complete_physical_unit_ids", self.complete_physical_unit_ids),
            ("excluded_physical_unit_ids", self.excluded_physical_unit_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(
                values,
                field_name=field_name,
                allow_empty=field_name
                in {
                    "complete_physical_unit_ids",
                    "excluded_physical_unit_ids",
                    "reason_codes",
                },
            )
        require_sorted_unique_ids(self.effects, attribute="effect_id", field_name="effects")
        if len(self.expected_physical_unit_ids) != CONFIRMATORY_LOCAL_REGION_UNIT_COUNT:
            raise ValueError("confirmatory regional test requires six expected units")
        if (
            set(self.complete_physical_unit_ids) & set(self.excluded_physical_unit_ids)
            or set(self.complete_physical_unit_ids) | set(self.excluded_physical_unit_ids)
            != set(self.expected_physical_unit_ids)
            or {value.physical_independent_unit_id for value in self.effects}
            != set(self.complete_physical_unit_ids)
        ):
            raise ValueError("confirmatory regional complete/excluded/effect rosters differ")
        if self.positive_effect_count != sum(value.value > 0 for value in self.effects):
            raise ValueError("confirmatory regional positive count differs from its effects")
        if (
            self.degrees_of_freedom != CONFIRMATORY_LOCAL_REGION_DEGREES_OF_FREEDOM
            or self.one_sided_tail_probability != CONFIRMATORY_LOCAL_REGION_PER_TEST_TAIL_PROBABILITY
            or self.one_sided_critical_value != CONFIRMATORY_LOCAL_REGION_CRITICAL_VALUE
            or self.materiality != CONFIRMATORY_LOCAL_REGION_MATERIALITY
        ):
            raise ValueError("confirmatory regional test constants differ")
        summary = (self.mean, self.lower_bound, self.upper_bound, self.median)
        if len(self.complete_physical_unit_ids) == CONFIRMATORY_LOCAL_REGION_UNIT_COUNT:
            if any(value is None for value in summary):
                raise ValueError("complete confirmatory regional test requires its summaries")
            effect_values = tuple(value.value for value in self.effects)
            expected_mean, expected_lower, expected_upper = _student_interval(
                effect_values,
                self.one_sided_critical_value,
            )
            if summary != (
                expected_mean,
                expected_lower,
                expected_upper,
                _median(effect_values),
            ):
                raise ValueError("confirmatory regional summaries differ from unit effects")
        elif any(value is not None for value in summary):
            raise ValueError("incomplete confirmatory regional test cannot impute summaries")
        supported = (
            len(self.complete_physical_unit_ids) == CONFIRMATORY_LOCAL_REGION_UNIT_COUNT
            and self.lower_bound is not None
            and self.lower_bound > self.materiality
            and self.positive_effect_count >= CONFIRMATORY_LOCAL_REGION_MINIMUM_POSITIVE_UNITS
            and self.median is not None
            and self.median >= self.materiality
            and self.action_exchange_passed
            and self.causal_obligations_passed
            and self.preservation_passed
        )
        if self.disposition is ConfirmatoryRegionalTestDisposition.SUPPORTED:
            if not supported or self.reason_codes:
                raise ValueError("supported confirmatory regional test violates G1R")
        elif self.disposition is ConfirmatoryRegionalTestDisposition.UNEVALUABLE:
            if len(self.complete_physical_unit_ids) == CONFIRMATORY_LOCAL_REGION_UNIT_COUNT:
                raise ValueError("complete confirmatory regional test cannot be unevaluable")
            if not self.reason_codes:
                raise ValueError("unevaluable confirmatory regional test requires reasons")
        elif supported or not self.reason_codes:
            raise ValueError("non-supported confirmatory regional test requires a failed rule")


class ConfirmatoryRegionCompatibilityDisposition(StrEnum):
    COMPATIBLE = "COMPATIBLE"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionRegionSupportCompatibility(CanonicalRecord):
    """Exact G1/G1R and prepared-D/local partition receipt; never a law."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/confirmatory-finite-action-region-support-compatibility'
    )

    compatibility_id: str
    spec: ObjectIdentity
    config: ObjectIdentity
    projection: ObjectIdentity
    projection_extension: ObjectIdentity
    pure_prefix_payload: ObjectIdentity
    prefix_cell_ids: tuple[str, ...]
    regional_test_ids: tuple[str, ...]
    regional_entry_ids: tuple[str, ...]
    final_payload: ObjectIdentity
    candidate_obligation_template: LawObligationTemplate
    qualification_profile: ObjectIdentity
    prepared_denominator_id: str
    local_support_ids: tuple[str, ...]
    prefix_ids: tuple[str, ...]
    g1_supported: bool
    g1r_supported: bool
    interpolation_used: bool
    disposition: ConfirmatoryRegionCompatibilityDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.compatibility_id, field_name="compatibility_id")
        validate_stable_id(
            self.prepared_denominator_id,
            field_name="prepared_denominator_id",
        )
        if self.spec.object_schema != ConfirmatoryFiniteActionRegionSupportSpec.SCHEMA:
            raise ValueError("confirmatory compatibility binds another region spec")
        if self.config.object_schema != ConfirmatoryFiniteActionConfig.SCHEMA:
            raise ValueError("confirmatory compatibility binds another config")
        if self.projection.object_schema != IdentificationEvidenceProjection.SCHEMA:
            raise ValueError("confirmatory compatibility binds another projection")
        if self.projection_extension.object_schema != (
            IdentificationEvidenceProjectionExtension.SCHEMA
        ):
            raise ValueError("confirmatory compatibility binds another projection extension")
        if self.pure_prefix_payload.object_schema != FiniteActionCompatibilitySetExtension.SCHEMA:
            raise ValueError("confirmatory compatibility binds another pure prefix payload")
        if self.final_payload.object_schema != FiniteActionCompatibilitySetExtension.SCHEMA:
            raise ValueError("confirmatory compatibility binds another final payload")
        if self.qualification_profile.object_schema != ComponentQualificationProfile.SCHEMA:
            raise ValueError("confirmatory compatibility binds another profile schema")
        for field_name, values in (
            ("prefix_cell_ids", self.prefix_cell_ids),
            ("regional_test_ids", self.regional_test_ids),
            ("regional_entry_ids", self.regional_entry_ids),
            ("local_support_ids", self.local_support_ids),
            ("prefix_ids", self.prefix_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(
                values,
                field_name=field_name,
                allow_empty=field_name == "reason_codes",
            )
        if (
            len(self.local_support_ids) != 3
            or len(self.prefix_ids) != 3
            or len(self.prefix_cell_ids) != 24
            or len(self.regional_test_ids) != CONFIRMATORY_LOCAL_REGION_TEST_COUNT
            or len(self.regional_entry_ids) != 24
            or self.prepared_denominator_id in set(self.local_support_ids) | set(self.prefix_ids)
            or set(self.local_support_ids) & set(self.prefix_ids)
        ):
            raise ValueError("confirmatory prepared/local/prefix product differs")
        if self.candidate_obligation_template.denominator_cell_ids != tuple(
            sorted((self.prepared_denominator_id, *self.local_support_ids))
        ):
            raise ValueError("confirmatory law support is not sorted prepared D plus locals")
        if self.candidate_obligation_template.recurrence_cell_ids != self.prefix_ids:
            raise ValueError("confirmatory compatibility lost exact G1 prefix evidence")
        if self.interpolation_used:
            raise ValueError("confirmatory region compatibility cannot interpolate")
        compatible = self.g1_supported and self.g1r_supported
        if self.disposition is ConfirmatoryRegionCompatibilityDisposition.COMPATIBLE:
            if not compatible or self.reason_codes:
                raise ValueError("compatible confirmatory region receipt lacks G1/G1R")
        elif compatible or not self.reason_codes:
            raise ValueError("noncompatible confirmatory region receipt requires reasons")


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionRegionSupportResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-finite-action-region-support-result'

    result_id: str
    spec: ObjectIdentity
    config: ObjectIdentity
    projection: ObjectIdentity
    projection_extension: ObjectIdentity
    prefix_cells: tuple[FiniteActionProductionCell, ...]
    regional_tests: tuple[ConfirmatoryFiniteActionRegionalTest, ...]
    regional_cells: tuple[FiniteActionProductionCell, ...]
    payload: FiniteActionCompatibilitySetExtension
    compatibility: ConfirmatoryFiniteActionRegionSupportCompatibility
    candidate_evidence: LawCandidateEvidence

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_sorted_unique_ids(
            self.prefix_cells,
            attribute="cell_id",
            field_name="prefix_cells",
        )
        require_sorted_unique_ids(
            self.regional_tests,
            attribute="test_id",
            field_name="regional_tests",
        )
        require_sorted_unique_ids(
            self.regional_cells,
            attribute="cell_id",
            field_name="regional_cells",
        )
        if len(self.regional_tests) != CONFIRMATORY_LOCAL_REGION_TEST_COUNT:
            raise ValueError("confirmatory region result requires exactly six G1R tests")
        if (
            self.compatibility.spec != self.spec
            or self.compatibility.config != self.config
            or self.compatibility.projection != self.projection
            or self.compatibility.projection_extension != self.projection_extension
            or self.compatibility.prefix_cell_ids
            != tuple(sorted(value.cell_id for value in self.prefix_cells))
            or self.compatibility.regional_test_ids
            != tuple(sorted(value.test_id for value in self.regional_tests))
            or self.compatibility.regional_entry_ids
            != tuple(sorted(value.entry_id for value in self.payload.entries))
        ):
            raise ValueError("confirmatory region result and compatibility identities differ")
        g1_supported = all(
            value.disposition is FiniteActionProductionDisposition.SUPPORTED
            for value in self.prefix_cells
        )
        g1r_supported = all(
            value.disposition is ConfirmatoryRegionalTestDisposition.SUPPORTED
            for value in self.regional_tests
        )
        if (
            self.compatibility.g1_supported != g1_supported
            or self.compatibility.g1r_supported != g1r_supported
        ):
            raise ValueError("confirmatory compatibility changes the G1/G1R dispositions")
        local_ids = set(self.compatibility.local_support_ids)
        unit_rosters_by_local = {
            local_id: {
                value.expected_physical_unit_ids
                for value in self.regional_tests
                if value.local_support_id == local_id
            }
            for local_id in local_ids
        }
        if (
            any(len(rosters) != 1 for rosters in unit_rosters_by_local.values())
            or {value.local_support_id for value in self.regional_tests} != local_ids
            or {value.support_cell_id for value in self.payload.entries} != local_ids
            or any(
                value.physical_independent_unit_ids
                not in unit_rosters_by_local[value.support_cell_id]
                for value in self.payload.entries
            )
        ):
            raise ValueError("confirmatory region result changes a tier-local unit roster")
        local_rosters = tuple(next(iter(value)) for value in unit_rosters_by_local.values())
        if any(
            set(left) & set(right)
            for index, left in enumerate(local_rosters)
            for right in local_rosters[index + 1 :]
        ) or {unit_id for roster in local_rosters for unit_id in roster} != set(
            self.payload.physical_independent_unit_ids
        ):
            raise ValueError("confirmatory region result pools or omits tier units")
        expected = {
            (
                value.denominator_member_id,
                value.candidate_version_id,
                value.support_cell_id,
                value.action_word.object_id,
            )
            for value in self.regional_cells
        }
        actual = {
            (
                value.denominator_member_id,
                value.candidate_version_id,
                value.support_cell_id,
                value.action_word.object_id,
            )
            for value in self.payload.entries
        }
        if expected != actual or len(self.regional_cells) != len(self.payload.entries):
            raise ValueError("confirmatory regional cells and payload rows differ")
        if self.compatibility.final_payload != ObjectIdentity.from_record(
            self.payload.extension_id,
            self.payload,
        ):
            raise ValueError("confirmatory compatibility binds another final payload")
        if self.compatibility.candidate_obligation_template != (
            self.candidate_evidence.obligation_template
        ):
            raise ValueError("confirmatory compatibility binds another candidate support")


@dataclass(frozen=True, slots=True)
class _CellCalculation:
    cell: FiniteActionProductionCell
    effects: tuple[tuple[str, Decimal], ...]
    complete: bool
    exchange: bool
    causal: bool
    preservation: bool


def _calculate_cell(
    *,
    config: ConfirmatoryFiniteActionConfig,
    operands: _ValidatedConfirmatoryOperands,
    member: str,
    candidate_version: str,
    support_cell_id: str,
    unit_ids: tuple[str, ...],
    constituent_tier_ids: tuple[str, ...],
    word: OccurrenceActionWord,
    role: ConfirmatoryActionRole,
    hold_id: str,
    critical_value: Decimal,
    previous_active_supported: bool,
    cell_namespace: str,
) -> _CellCalculation:
    values = operands.values
    invalid = operands.invalid
    unit_bindings = operands.unit_bindings
    words = operands.words
    paired: dict[str, tuple[Decimal, Decimal, Decimal]] = {}
    excluded: list[str] = []
    for unit_id in unit_ids:
        action = values.get((member, candidate_version, unit_id, word.word_id))
        hold = values.get((member, candidate_version, unit_id, hold_id))
        if action is None or hold is None:
            excluded.append(unit_id)
            continue
        paired[unit_id] = (
            action.response - hold.response,
            action.precutoff - hold.precutoff,
            action.onset_clock,
        )
    complete_units = tuple(sorted(paired))
    effects = tuple((unit_id, paired[unit_id][0]) for unit_id in complete_units)
    reasons: list[str] = []
    mean = lower = upper = Decimal(0)
    causal_ok = True
    preservation_ok = True
    if word.support_status is not ActionWordSupportStatus.SUPPORTED:
        disposition = FiniteActionProductionDisposition.OUTSIDE_SUPPORT
        reasons.append("CONFIRMATORY_ACTION_WORD_OUTSIDE_SUPPORT")
    elif len(complete_units) != len(unit_ids):
        disposition = (
            FiniteActionProductionDisposition.INVALID
            if any(
                (member, candidate_version, unit_id, word.word_id) in invalid
                or (member, candidate_version, unit_id, hold_id) in invalid
                for unit_id in excluded
            )
            else FiniteActionProductionDisposition.PARTIAL
        )
        reasons.append("CONFIRMATORY_INCOMPLETE_PHYSICAL_UNIT_WORD_GRID")
        causal_ok = False
        preservation_ok = False
    else:
        response_values = tuple(paired[value][0] for value in complete_units)
        mean, lower, upper = _student_interval(response_values, critical_value)
        preservation_ok = all(
            abs(paired[unit_id][1]) <= config.response_quantity.maximum_precutoff_absolute_effect
            for unit_id in complete_units
        )
        tier_effects = {
            tier_id: tuple(
                paired[unit_id][0]
                for unit_id in complete_units
                if unit_bindings[unit_id].tier_id == tier_id
            )
            for tier_id in constituent_tier_ids
        }
        if any(
            len(tier_values) < config.minimum_complete_units_per_tier
            for tier_values in tier_effects.values()
        ):
            reasons.append("CONFIRMATORY_TIER_COMPLETE_UNIT_MINIMUM_NOT_MET")
        if not preservation_ok:
            reasons.append("CONFIRMATORY_PRECUTOFF_HISTORY_FALSIFIER_FAILED")
        if role is ConfirmatoryActionRole.TARGET_ACTIVE:
            causal_ok = all(
                config.response_quantity.earliest_active_response_clock
                <= paired[unit_id][2]
                <= config.response_quantity.latest_active_response_clock
                for unit_id in complete_units
            )
            if lower <= config.materiality:
                reasons.append("CONFIRMATORY_ONE_SIDED_LOWER_NOT_MATERIAL")
            if any(
                sum(effect > 0 for effect in tier_values) < config.minimum_positive_units_per_tier
                for tier_values in tier_effects.values()
            ):
                reasons.append("CONFIRMATORY_TIER_POSITIVE_COUNT_FAILED")
            if any(
                _median(tier_values) < config.materiality for tier_values in tier_effects.values()
            ):
                reasons.append("CONFIRMATORY_TIER_MEDIAN_NOT_MATERIAL")
            if not causal_ok:
                reasons.append("CONFIRMATORY_ACTIVE_CAUSAL_CLOCK_FAILED")
            if not previous_active_supported:
                reasons.append("CONFIRMATORY_PREFIX_CLOSURE_FAILED")
        elif role is ConfirmatoryActionRole.WRONG_SIGN_CONTROL:
            causal_ok = all(
                config.response_quantity.earliest_active_response_clock
                <= paired[unit_id][2]
                <= config.response_quantity.latest_active_response_clock
                for unit_id in complete_units
            )
            if not causal_ok:
                reasons.append("CONFIRMATORY_WRONG_SIGN_CAUSAL_CLOCK_FAILED")
        elif role is ConfirmatoryActionRole.FUTURE_NULL_CONTROL:
            null_ok = all(
                abs(effect) <= config.response_quantity.maximum_null_absolute_effect
                for effect in response_values
            )
            causal_ok = all(
                paired[unit_id][2] >= config.response_quantity.earliest_future_response_clock
                for unit_id in complete_units
            )
            if not null_ok:
                reasons.append("CONFIRMATORY_FUTURE_TARGET_WINDOW_NOT_NULL")
            if not causal_ok:
                reasons.append("CONFIRMATORY_FUTURE_CAUSAL_CLOCK_FAILED")
            causal_ok = causal_ok and null_ok
        disposition = (
            FiniteActionProductionDisposition.SUPPORTED
            if not reasons
            else FiniteActionProductionDisposition.NOT_SUPPORTED
        )
    exchange_ok = (
        word.denominator_id == words[hold_id].denominator_id
        and word.retained_history_id == words[hold_id].retained_history_id
        and word.horizon_id == words[hold_id].horizon_id
    )
    suffix = (f"{member}.{candidate_version}.{support_cell_id}.{word.word_id}").replace("_", "-")
    estimates: tuple[FiniteActionNativeValue, ...] = ()
    lowers: tuple[FiniteActionNativeValue, ...] = ()
    uppers: tuple[FiniteActionNativeValue, ...] = ()
    if disposition in {
        FiniteActionProductionDisposition.SUPPORTED,
        FiniteActionProductionDisposition.NOT_SUPPORTED,
    }:
        estimates = (_value(prefix="response", quantity=config.response_quantity, value=mean),)
        lowers = (_value(prefix="lower", quantity=config.response_quantity, value=lower),)
        uppers = (_value(prefix="upper", quantity=config.response_quantity, value=upper),)
    return _CellCalculation(
        cell=FiniteActionProductionCell(
            cell_id=f"{cell_namespace}.{suffix}",
            denominator_member_id=member,
            candidate_version_id=candidate_version,
            support_cell_id=support_cell_id,
            action_word=ObjectIdentity.from_record(word.word_id, word),
            comparator_action_word_id=hold_id,
            complete_physical_unit_ids=complete_units,
            excluded_physical_unit_ids=tuple(sorted(excluded)),
            response_estimates=estimates,
            lower_bounds=lowers,
            upper_bounds=uppers,
            sink_operands=(),
            disposition=disposition,
            reason_codes=tuple(sorted(reasons)),
        ),
        effects=effects,
        complete=len(complete_units) == len(unit_ids),
        exchange=exchange_ok,
        causal=causal_ok,
        preservation=preservation_ok,
    )


def _payload_disposition(
    disposition: FiniteActionProductionDisposition,
) -> FiniteActionCellDisposition:
    return {
        FiniteActionProductionDisposition.SUPPORTED: FiniteActionCellDisposition.SUPPORTED,
        FiniteActionProductionDisposition.NOT_SUPPORTED: (
            FiniteActionCellDisposition.NOT_SUPPORTED
        ),
        FiniteActionProductionDisposition.OUTSIDE_SUPPORT: (
            FiniteActionCellDisposition.OUTSIDE_SUPPORT
        ),
        FiniteActionProductionDisposition.PARTIAL: FiniteActionCellDisposition.UNEVALUABLE,
        FiniteActionProductionDisposition.INVALID: FiniteActionCellDisposition.UNEVALUABLE,
        FiniteActionProductionDisposition.UNEVALUABLE: FiniteActionCellDisposition.UNEVALUABLE,
    }[disposition]


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionRegionSupportProducer:
    """Pure G1/G1R producer plus one ordinary payload publication."""

    payload_publisher: CandidatePayloadPublisher
    qualification_profile: ObjectIdentity

    def produce(
        self,
        *,
        spec: ConfirmatoryFiniteActionRegionSupportSpec,
        projection: IdentificationEvidenceProjection,
        extension: IdentificationEvidenceProjectionExtension,
        config: ConfirmatoryFiniteActionConfig,
        scaffold: FiniteActionCandidateScaffold,
    ) -> ConfirmatoryFiniteActionRegionSupportResult:
        if self.qualification_profile.object_schema != ComponentQualificationProfile.SCHEMA:
            raise ValueError("confirmatory regional producer requires an exact profile")
        if self.qualification_profile.object_id != (
            "profile.confirmatory-finite-action-local-support.method-equivalent"
        ):
            raise ValueError("confirmatory regional producer requires the local-support profile")
        if (
            spec.prepared_denominator_id != config.prepared_denominator_id
            or spec.chart_id != config.chart_id
            or spec.horizon != config.horizon
            or spec.retained_history_ids != config.retained_history_ids
            or spec.action_words != config.action_words
            or spec.action_role_bindings != config.action_role_bindings
            or spec.axis_map != config.axis_map
            or spec.prefixes != config.prefixes
            or spec.response_quantity != config.response_quantity
            or spec.information_cutoff_id != config.information_cutoff_id
            or spec.physical_independent_unit_ids != config.physical_independent_unit_ids
            or config.materiality != spec.materiality
            or config.minimum_complete_units_per_tier != spec.minimum_complete_units
            or config.minimum_positive_units_per_tier != spec.minimum_positive_units
            or config.family_alpha != spec.family_alpha
            or config.simultaneous_test_count != spec.simultaneous_test_count
        ):
            raise ValueError("confirmatory regional spec differs from the bound config")
        expected_units_by_tier = {
            value.tier_id: value.physical_independent_unit_ids
            for value in spec.tier_region_bindings
        }
        actual_units_by_tier = {
            tier_id: tuple(
                sorted(
                    value.physical_independent_unit_id
                    for value in config.unit_tier_bindings
                    if value.tier_id == tier_id
                )
            )
            for tier_id in expected_units_by_tier
        }
        if actual_units_by_tier != expected_units_by_tier:
            raise ValueError("confirmatory regional tier units differ from the issued map")
        if (
            scaffold.obligation_template.denominator_cell_ids
            != tuple(sorted((spec.prepared_denominator_id, *spec.local_support_ids)))
            or scaffold.obligation_template.recurrence_cell_ids != spec.prefix_ids
        ):
            raise ValueError("confirmatory regional candidate support partition differs")
        operands = _validated_confirmatory_operands(
            projection=projection,
            extension=extension,
            config=config,
            scaffold=scaffold,
        )
        evidence_link_ids = tuple(value.link_id for value in scaffold.evidence_links)

        prefix_cells: list[FiniteActionProductionCell] = []
        prefix_calculations: list[_CellCalculation] = []
        active_prefix_pass: dict[tuple[str, str, str], bool] = {}
        ordered_prefixes = tuple(sorted(config.prefixes, key=lambda value: value.order_index))
        for axis in config.axis_map.bindings:
            member = axis.denominator_member_id
            version = axis.candidate_version_member_id
            for prefix in ordered_prefixes:
                unit_ids = tuple(
                    sorted(
                        value.physical_independent_unit_id
                        for value in config.unit_tier_bindings
                        if value.tier_id in prefix.constituent_tier_ids
                    )
                )
                previous_ok = all(
                    active_prefix_pass.get((member, version, prior.prefix_id), False)
                    for prior in ordered_prefixes
                    if prior.order_index < prefix.order_index
                )
                for word in config.action_words:
                    role = operands.roles[word.word_id]
                    calculation = _calculate_cell(
                        config=config,
                        operands=operands,
                        member=member,
                        candidate_version=version,
                        support_cell_id=prefix.prefix_id,
                        unit_ids=unit_ids,
                        constituent_tier_ids=prefix.constituent_tier_ids,
                        word=word,
                        role=role,
                        hold_id=operands.hold_id,
                        critical_value=prefix.one_sided_critical_value,
                        previous_active_supported=previous_ok,
                        cell_namespace="confirmatory-prefix-cell",
                    )
                    prefix_calculations.append(calculation)
                    prefix_cells.append(calculation.cell)
                    if role is ConfirmatoryActionRole.TARGET_ACTIVE:
                        active_prefix_pass[(member, version, prefix.prefix_id)] = (
                            calculation.cell.disposition
                            is FiniteActionProductionDisposition.SUPPORTED
                        )
        g1_supported = all(
            value.cell.disposition is FiniteActionProductionDisposition.SUPPORTED
            for value in prefix_calculations
        )
        pure_prefix_payload = FiniteActionCompatibilitySetExtension(
            extension_id=f"confirmatory-prefix-calculation.{config.config_id}",
            method_semantics=(
                "Pure unpublished G1 prefix calculation with exact role and causal rules; "
                "no linearity inference or regional relabeling."
            ),
            prepared_denominator_id=config.prepared_denominator_id,
            horizon=config.horizon,
            retained_history_ids=config.retained_history_ids,
            action_words=config.action_words,
            physical_independent_unit_ids=config.physical_independent_unit_ids,
            entries=tuple(
                sorted(
                    (
                        FiniteActionResponseEntry(
                            entry_id=f"prefix-calculation-entry.{value.cell.cell_id}",
                            denominator_member_id=value.cell.denominator_member_id,
                            candidate_version_id=value.cell.candidate_version_id,
                            qualification_view_ids=next(
                                axis.qualification_view_ids
                                for axis in config.axis_map.bindings
                                if axis.denominator_member_id == value.cell.denominator_member_id
                                and axis.candidate_version_member_id
                                == value.cell.candidate_version_id
                            ),
                            support_cell_id=value.cell.support_cell_id,
                            action_word=value.cell.action_word,
                            comparator_action_word_ids=(value.cell.comparator_action_word_id,),
                            physical_independent_unit_ids=tuple(
                                sorted(
                                    (
                                        *value.cell.complete_physical_unit_ids,
                                        *value.cell.excluded_physical_unit_ids,
                                    )
                                )
                            ),
                            response_values=value.cell.response_estimates,
                            sink_values=(),
                            effort_values=(),
                            uncertainty_values=tuple(
                                sorted(
                                    (*value.cell.lower_bounds, *value.cell.upper_bounds),
                                    key=lambda item: item.value_id,
                                )
                            ),
                            disposition=_payload_disposition(value.cell.disposition),
                            evidence_link_ids=evidence_link_ids,
                            reason_codes=value.cell.reason_codes,
                        )
                        for value in prefix_calculations
                    ),
                    key=lambda value: value.entry_id,
                )
            ),
            rejection_entry_ids=tuple(
                sorted(
                    f"prefix-calculation-entry.{value.cell.cell_id}"
                    for value in prefix_calculations
                    if value.cell.disposition is not FiniteActionProductionDisposition.SUPPORTED
                )
            ),
            refusal_reason_codes=("confirmatory-prefix-calculation-cell-unavailable",),
        )

        regional_cells: list[FiniteActionProductionCell] = []
        regional_calculations: dict[tuple[str, str, str, str], _CellCalculation] = {}
        regional_tests: list[ConfirmatoryFiniteActionRegionalTest] = []
        active_id = next(
            value.action_word_id
            for value in config.action_role_bindings
            if value.role is ConfirmatoryActionRole.TARGET_ACTIVE
        )
        hold_word = operands.words[operands.hold_id]
        active_word = operands.words[active_id]
        for axis in config.axis_map.bindings:
            member = axis.denominator_member_id
            version = axis.candidate_version_member_id
            for mapping in spec.tier_region_bindings:
                role_calculations: list[_CellCalculation] = []
                for word in config.action_words:
                    calculation = _calculate_cell(
                        config=config,
                        operands=operands,
                        member=member,
                        candidate_version=version,
                        support_cell_id=mapping.local_support_id,
                        unit_ids=mapping.physical_independent_unit_ids,
                        constituent_tier_ids=(mapping.tier_id,),
                        word=word,
                        role=operands.roles[word.word_id],
                        hold_id=operands.hold_id,
                        critical_value=spec.critical_value,
                        previous_active_supported=True,
                        cell_namespace="confirmatory-region-cell",
                    )
                    role_calculations.append(calculation)
                    regional_cells.append(calculation.cell)
                    regional_calculations[
                        (member, version, mapping.local_support_id, word.word_id)
                    ] = calculation
                active = regional_calculations[
                    (member, version, mapping.local_support_id, active_id)
                ]
                complete_ids = active.cell.complete_physical_unit_ids
                excluded_ids = active.cell.excluded_physical_unit_ids
                effects = tuple(
                    sorted(
                        (
                            ConfirmatoryFiniteActionUnitEffect(
                                effect_id=(
                                    f"regional-effect.{member}.{version}.{mapping.tier_id}."
                                    f"{unit_id}"
                                ),
                                physical_independent_unit_id=unit_id,
                                value=value,
                                native_unit=config.response_quantity.native_unit,
                                native_frame_id=config.response_quantity.native_frame_id,
                                clock_id=config.response_quantity.response_clock_id,
                            )
                            for unit_id, value in active.effects
                        ),
                        key=lambda value: value.effect_id,
                    )
                )
                response_values = tuple(value.value for value in effects)
                summary: tuple[Decimal | None, Decimal | None, Decimal | None, Decimal | None]
                if len(complete_ids) == spec.minimum_complete_units:
                    mean, lower, upper = _student_interval(
                        response_values,
                        spec.critical_value,
                    )
                    summary = (mean, lower, upper, _median(response_values))
                else:
                    summary = (None, None, None, None)
                action_exchange_passed = all(value.exchange for value in role_calculations)
                causal_passed = all(value.causal for value in role_calculations)
                preservation_passed = all(value.preservation for value in role_calculations)
                reasons = set(active.cell.reason_codes)
                if len(complete_ids) != spec.minimum_complete_units:
                    reasons.add("CONFIRMATORY_G1R_INCOMPLETE_TIER")
                    disposition = ConfirmatoryRegionalTestDisposition.UNEVALUABLE
                else:
                    regional_lower = summary[1]
                    regional_median = summary[3]
                    if regional_lower is None or regional_lower <= spec.materiality:
                        reasons.add("CONFIRMATORY_G1R_LOWER_NOT_STRICTLY_MATERIAL")
                    if sum(value.value > 0 for value in effects) < spec.minimum_positive_units:
                        reasons.add("CONFIRMATORY_G1R_POSITIVE_COUNT_FAILED")
                    if regional_median is None or regional_median < spec.median_materiality:
                        reasons.add("CONFIRMATORY_G1R_MEDIAN_NOT_MATERIAL")
                    if not action_exchange_passed:
                        reasons.add("CONFIRMATORY_G1R_ACTION_EXCHANGE_FAILED")
                    if not causal_passed:
                        reasons.add("CONFIRMATORY_G1R_CAUSAL_OBLIGATION_FAILED")
                    if not preservation_passed:
                        reasons.add("CONFIRMATORY_G1R_PRESERVATION_FAILED")
                    disposition = (
                        ConfirmatoryRegionalTestDisposition.SUPPORTED
                        if not reasons
                        else ConfirmatoryRegionalTestDisposition.NOT_SUPPORTED
                    )
                regional_tests.append(
                    ConfirmatoryFiniteActionRegionalTest(
                        test_id=(f"confirmatory-g1r-test.{member}.{version}.{mapping.tier_id}"),
                        family_id=spec.region_family_id,
                        tier_id=mapping.tier_id,
                        local_support_id=mapping.local_support_id,
                        denominator_member_id=member,
                        candidate_version_id=version,
                        qualification_view_ids=axis.qualification_view_ids,
                        active_action_word=ObjectIdentity.from_record(
                            active_word.word_id,
                            active_word,
                        ),
                        hold_action_word=ObjectIdentity.from_record(
                            hold_word.word_id,
                            hold_word,
                        ),
                        expected_physical_unit_ids=mapping.physical_independent_unit_ids,
                        complete_physical_unit_ids=complete_ids,
                        excluded_physical_unit_ids=excluded_ids,
                        effects=effects,
                        mean=summary[0],
                        lower_bound=summary[1],
                        upper_bound=summary[2],
                        median=summary[3],
                        positive_effect_count=sum(value.value > 0 for value in effects),
                        degrees_of_freedom=spec.degrees_of_freedom,
                        one_sided_tail_probability=spec.per_test_tail_probability,
                        one_sided_critical_value=spec.critical_value,
                        materiality=spec.materiality,
                        action_exchange_passed=action_exchange_passed,
                        causal_obligations_passed=causal_passed,
                        preservation_passed=preservation_passed,
                        disposition=disposition,
                        reason_codes=tuple(sorted(reasons)),
                    )
                )
        g1r_supported = all(
            value.disposition is ConfirmatoryRegionalTestDisposition.SUPPORTED
            for value in regional_tests
        )
        axis_by_coordinate = {
            (value.denominator_member_id, value.candidate_version_member_id): value
            for value in config.axis_map.bindings
        }
        entries = tuple(
            sorted(
                (
                    FiniteActionResponseEntry(
                        entry_id=f"confirmatory-region-entry.{cell.cell_id}",
                        denominator_member_id=cell.denominator_member_id,
                        candidate_version_id=cell.candidate_version_id,
                        qualification_view_ids=axis_by_coordinate[
                            (cell.denominator_member_id, cell.candidate_version_id)
                        ].qualification_view_ids,
                        support_cell_id=cell.support_cell_id,
                        action_word=cell.action_word,
                        comparator_action_word_ids=(cell.comparator_action_word_id,),
                        physical_independent_unit_ids=next(
                            value.physical_independent_unit_ids
                            for value in spec.tier_region_bindings
                            if value.local_support_id == cell.support_cell_id
                        ),
                        response_values=cell.response_estimates,
                        sink_values=(),
                        effort_values=(),
                        uncertainty_values=tuple(
                            sorted(
                                (*cell.lower_bounds, *cell.upper_bounds),
                                key=lambda value: value.value_id,
                            )
                        ),
                        disposition=_payload_disposition(cell.disposition),
                        evidence_link_ids=evidence_link_ids,
                        reason_codes=cell.reason_codes,
                    )
                    for cell in regional_cells
                ),
                key=lambda value: value.entry_id,
            )
        )
        payload = FiniteActionCompatibilitySetExtension(
            extension_id=f"confirmatory-region-support-set.{config.config_id}",
            method_semantics=(
                "Fresh member-local finite-word response lookup on exactly three disjoint "
                "tier regions with G1 and G1R intersection; no linearity inference, prepared-"
                "denominator response row, interpolation, coefficient pooling or magnitude-"
                "convergence claim."
            ),
            prepared_denominator_id=spec.prepared_denominator_id,
            horizon=spec.horizon,
            retained_history_ids=spec.retained_history_ids,
            action_words=spec.action_words,
            physical_independent_unit_ids=spec.physical_independent_unit_ids,
            entries=entries,
            rejection_entry_ids=tuple(
                sorted(
                    value.entry_id
                    for value in entries
                    if value.disposition is not FiniteActionCellDisposition.SUPPORTED
                )
            ),
            refusal_reason_codes=("confirmatory-region-support-cell-unavailable",),
        )
        compatibility_reasons = []
        if not g1_supported:
            compatibility_reasons.append("CONFIRMATORY_G1_PREFIX_FAMILY_FAILED")
        if not g1r_supported:
            compatibility_reasons.append("CONFIRMATORY_G1R_REGION_FAMILY_FAILED")
        if any(
            value.disposition is ConfirmatoryRegionalTestDisposition.UNEVALUABLE
            for value in regional_tests
        ):
            compatibility_disposition = ConfirmatoryRegionCompatibilityDisposition.UNEVALUABLE
        elif compatibility_reasons:
            compatibility_disposition = ConfirmatoryRegionCompatibilityDisposition.NOT_SUPPORTED
        else:
            compatibility_disposition = ConfirmatoryRegionCompatibilityDisposition.COMPATIBLE
        compatibility = ConfirmatoryFiniteActionRegionSupportCompatibility(
            compatibility_id=f"confirmatory-region-compatibility.{config.config_id}",
            spec=ObjectIdentity.from_record(spec.spec_id, spec),
            config=ObjectIdentity.from_record(config.config_id, config),
            projection=config.projection,
            projection_extension=config.projection_extension,
            pure_prefix_payload=ObjectIdentity.from_record(
                pure_prefix_payload.extension_id,
                pure_prefix_payload,
            ),
            prefix_cell_ids=tuple(sorted(value.cell_id for value in prefix_cells)),
            regional_test_ids=tuple(sorted(value.test_id for value in regional_tests)),
            regional_entry_ids=tuple(sorted(value.entry_id for value in entries)),
            final_payload=ObjectIdentity.from_record(payload.extension_id, payload),
            candidate_obligation_template=scaffold.obligation_template,
            qualification_profile=self.qualification_profile,
            prepared_denominator_id=spec.prepared_denominator_id,
            local_support_ids=spec.local_support_ids,
            prefix_ids=spec.prefix_ids,
            g1_supported=g1_supported,
            g1r_supported=g1r_supported,
            interpolation_used=False,
            disposition=compatibility_disposition,
            reason_codes=tuple(sorted(compatibility_reasons)),
        )
        payload_bytes = payload.canonical_bytes()
        artifact = ArtifactIdentity(
            artifact_id=f"artifact.{payload.fingerprint()[:32]}",
            role="law-evaluator-payload",
            payload_schema=payload.SCHEMA,
            sha256=payload.fingerprint(),
            media_type="application/json",
            size_bytes=len(payload_bytes),
        )
        evaluator = ExecutableReference(
            reference_id=f"evaluator.region-support.{config.config_id}",
            capability_key="law-evaluator.finite-action",
            capability_version=CONFIRMATORY_FINITE_ACTION_VERSION,
            evaluator_key="canonical-finite-action-law",
            payload=artifact,
            payload_format=SafePayloadFormat.CANONICAL_JSON,
            input_schema='empirical-lawhood/methods/law-evaluation-request',
            output_schema='empirical-lawhood/methods/law-evaluation-result',
            deterministic=True,
        )
        implementation = CandidateEvaluatorImplementation(
            implementation_id=f"implementation.region-support.{config.config_id}",
            capability_key=evaluator.capability_key,
            capability_version=evaluator.capability_version,
            evaluator_key=evaluator.evaluator_key,
            source_sha256=spec.producer_implementation_sha256,
        )
        publication = self.payload_publisher.publish_candidate_payload(
            payload=payload_bytes,
            evaluator=evaluator,
            implementation=ObjectIdentity.from_record(
                implementation.implementation_id,
                implementation,
            ),
            decoder_schema='empirical-lawhood/methods/finite-action/compatibility-set-decoder',
            decoder_version="1.0.0",
            maximum_decode_bytes=4 * 1024 * 1024,
        )
        all_calculations = (*prefix_calculations, *regional_calculations.values())
        target_signs: dict[str, set[int]] = {}
        for test in regional_tests:
            if test.mean is not None:
                target_signs.setdefault(test.local_support_id, set()).add(
                    1 if test.mean > 0 else -1 if test.mean < 0 else 0
                )
        member_structural_agreement = (
            set(target_signs) == set(spec.local_support_ids)
            and all(len(value) == 1 and value != {0} for value in target_signs.values())
            and all(
                len(
                    {
                        test.disposition
                        for test in regional_tests
                        if test.local_support_id == local_id
                    }
                )
                == 1
                for local_id in spec.local_support_ids
            )
        )
        receipt_values: dict[str, bool | None] = {
            "recurrence": all(value.complete for value in all_calculations),
            "declared-word-exchange": all(value.exchange for value in all_calculations),
            "causal-falsifiers": all(value.causal for value in all_calculations),
            "history-prefix-closure": all(value.preservation for value in all_calculations),
            "fresh-confirmatory-evaluation": all(value.complete for value in all_calculations),
            "member-structural-agreement": member_structural_agreement,
            "g1-prefix-support": g1_supported,
            "g1r-region-support": g1r_supported,
            "region-support-compatibility": (
                compatibility.disposition is ConfirmatoryRegionCompatibilityDisposition.COMPATIBLE
            ),
            "uncertainty.numerical": len(config.axis_map.bindings) > 1,
            "uncertainty.aleatoric": len(config.physical_independent_unit_ids) > 1,
            "uncertainty.epistemic": len(config.axis_map.bindings) > 1,
            "uncertainty.transport": None,
            "uncertainty.observation": all(value.complete for value in all_calculations),
        }
        receipts: tuple[CandidateMethodEvidenceReceipt, ...] = tuple(
            sorted(
                (
                    _receipt(
                        candidate_id=scaffold.candidate_id,
                        suffix=suffix,
                        observed=observed,
                        evidence_link_ids=evidence_link_ids,
                    )
                    for suffix, observed in receipt_values.items()
                ),
                key=lambda value: value.receipt_id,
            )
        )
        candidate = LawCandidateEvidence(
            evidence_id=scaffold.evidence_id,
            candidate_id=scaffold.candidate_id,
            system=scaffold.system,
            dataset_or_projection=config.projection,
            config=ObjectIdentity.from_record(config.config_id, config),
            method_key=config.method_key,
            method_version=config.method_version,
            method_kind=LawMethodKind.NONLINEAR_LOCAL,
            representation_kind=LawRepresentationKind.FINITE_ACTION_OPERATOR,
            candidate_evaluator=evaluator,
            axis_map=config.axis_map,
            claim_unit_binding=scaffold.claim_unit_binding,
            physical_independent_unit_ids=config.physical_independent_unit_ids,
            claim_template=scaffold.claim_template,
            obligation_template=scaffold.obligation_template,
            method_receipts=receipts,
            payload_publication=publication,
            evidence_links=scaffold.evidence_links,
            outcome_access=scaffold.outcome_access,
            parent_visibility_ceilings=scaffold.parent_visibility_ceilings,
            visibility_ceiling=scaffold.visibility_ceiling,
            candidate_extension=ExtensionBinding(
                namespace="finite-action-compatibility-set",
                schema=payload.SCHEMA,
                payload_sha256=publication.content_sha256,
            ),
        )
        return ConfirmatoryFiniteActionRegionSupportResult(
            result_id=f"confirmatory-region-support-result.{config.fingerprint()[:32]}",
            spec=ObjectIdentity.from_record(spec.spec_id, spec),
            config=ObjectIdentity.from_record(config.config_id, config),
            projection=config.projection,
            projection_extension=config.projection_extension,
            prefix_cells=tuple(sorted(prefix_cells, key=lambda value: value.cell_id)),
            regional_tests=tuple(sorted(regional_tests, key=lambda value: value.test_id)),
            regional_cells=tuple(sorted(regional_cells, key=lambda value: value.cell_id)),
            payload=payload,
            compatibility=compatibility,
            candidate_evidence=candidate,
        )


_PROSPECTIVE_CONFIRMATORY_INPUT_SCHEMAS = tuple(
    sorted(
        (
            ConfirmatoryFiniteActionBindingReceipt.SCHEMA,
            ConfirmatoryFiniteActionConfig.SCHEMA,
            ConfirmatoryFiniteActionDesign.SCHEMA,
            ConfirmatoryFiniteActionRegionSupportSpec.SCHEMA,
            FiniteActionCandidateScaffold.SCHEMA,
            IdentificationEvidenceManifest.SCHEMA,
            IdentificationEvidenceProjection.SCHEMA,
            IdentificationEvidenceProjectionExtension.SCHEMA,
            TaggedIdentificationProjectionPlan.SCHEMA,
            StudyOperationAuthority.SCHEMA,
            SystemSpec.SCHEMA,
        )
    )
)
_PROSPECTIVE_CONFIRMATORY_OUTPUT_SCHEMAS = tuple(
    sorted(
        (
            ComponentUncertaintyFamilyAssessment.SCHEMA,
            ConfirmatoryFiniteActionRegionSupportResult.SCHEMA,
            LawQualificationResult.SCHEMA,
            'empirical-lawhood/methods/prospective-confirmatory-finite-action-evaluation',
        )
    )
)


@dataclass(frozen=True, slots=True)
class ProspectiveConfirmatoryFiniteActionEvaluatorBinding(CanonicalRecord):
    """Static direct-service binding for one authorized post-reveal invocation."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/prospective-confirmatory-finite-action-evaluator-binding'
    )

    binding_id: str
    evaluator_id: str
    implementation_sha256: str
    binding_implementation_id: str
    binding_implementation_sha256: str
    canonical_codec_id: str
    input_schema_ids: tuple[str, ...]
    output_schema_ids: tuple[str, ...]
    qualification_profile: ObjectIdentity
    maximum_outcome_access: OutcomeAccess
    deterministic: bool
    acquisition_allowed: bool
    threshold_derivation_allowed: bool
    region_selection_allowed: bool
    candidate_publication_count: int
    finalizer_invocation_count: int

    def __post_init__(self) -> None:
        for field_name, identifier in (
            ("binding_id", self.binding_id),
            ("evaluator_id", self.evaluator_id),
            ("binding_implementation_id", self.binding_implementation_id),
            ("canonical_codec_id", self.canonical_codec_id),
        ):
            validate_stable_id(identifier, field_name=field_name)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        validate_sha256(
            self.binding_implementation_sha256,
            field_name="binding_implementation_sha256",
        )
        require_sorted_unique_strings(
            self.input_schema_ids,
            field_name="input_schema_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.output_schema_ids,
            field_name="output_schema_ids",
            allow_empty=False,
        )
        if self.input_schema_ids != _PROSPECTIVE_CONFIRMATORY_INPUT_SCHEMAS:
            raise ValueError("prospective confirmatory evaluator input codecs differ")
        if self.output_schema_ids != _PROSPECTIVE_CONFIRMATORY_OUTPUT_SCHEMAS:
            raise ValueError("prospective confirmatory evaluator output codecs differ")
        if self.qualification_profile.object_schema != ComponentQualificationProfile.SCHEMA:
            raise ValueError("prospective confirmatory evaluator binds another profile schema")
        if self.qualification_profile.object_id != (
            "profile.confirmatory-finite-action-local-support.method-equivalent"
        ):
            raise ValueError("prospective confirmatory evaluator requires local support")
        if self.maximum_outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("prospective confirmatory evaluator is evaluator-reveal only")
        if not self.deterministic:
            raise ValueError("prospective confirmatory evaluator must be deterministic")
        if (
            self.acquisition_allowed
            or self.threshold_derivation_allowed
            or self.region_selection_allowed
        ):
            raise ValueError("prospective confirmatory evaluator cannot change the experiment")
        if self.candidate_publication_count != 1 or self.finalizer_invocation_count != 1:
            raise ValueError("prospective confirmatory evaluator has one candidate/finalizer")


def prospective_confirmatory_finite_action_evaluator_binding(
    *,
    implementation_sha256: str,
    binding_implementation_sha256: str,
    qualification_profile: ObjectIdentity,
) -> ProspectiveConfirmatoryFiniteActionEvaluatorBinding:
    """Build the one static, substrate-neutral direct-service binding."""

    return ProspectiveConfirmatoryFiniteActionEvaluatorBinding(
        binding_id="binding.prospective-confirmatory-finite-action-evaluator",
        evaluator_id="evaluator.prospective-confirmatory-finite-action",
        implementation_sha256=implementation_sha256,
        binding_implementation_id="implementation.confirmatory-design-binder",
        binding_implementation_sha256=binding_implementation_sha256,
        canonical_codec_id="codec.canonical-json-strict",
        input_schema_ids=_PROSPECTIVE_CONFIRMATORY_INPUT_SCHEMAS,
        output_schema_ids=_PROSPECTIVE_CONFIRMATORY_OUTPUT_SCHEMAS,
        qualification_profile=qualification_profile,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        deterministic=True,
        acquisition_allowed=False,
        threshold_derivation_allowed=False,
        region_selection_allowed=False,
        candidate_publication_count=1,
        finalizer_invocation_count=1,
    )


@dataclass(frozen=True, slots=True)
class ProspectiveConfirmatoryFiniteActionEvaluation(CanonicalRecord):
    """Sole direct-service result; includes the ordinary terminal law result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-confirmatory-finite-action-evaluation'

    evaluation_id: str
    binding: ObjectIdentity
    design: ObjectIdentity
    binding_receipt: ConfirmatoryFiniteActionBindingReceipt
    reveal_authority: StudyOperationAuthority
    region_support: ConfirmatoryFiniteActionRegionSupportResult
    candidate_family: ComponentUncertaintyFamilyAssessment
    qualification_result: LawQualificationResult
    evaluated_at_utc: str
    candidate_publication_count: int
    finalizer_invocation_count: int
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        if self.binding.object_schema != (
            ProspectiveConfirmatoryFiniteActionEvaluatorBinding.SCHEMA
        ):
            raise ValueError("prospective confirmatory result binds another implementation")
        if self.design.object_schema != ConfirmatoryFiniteActionDesign.SCHEMA:
            raise ValueError("prospective confirmatory result binds another design")
        if self.binding_receipt.design != self.design:
            raise ValueError("prospective confirmatory result changes its bound design")
        if self.region_support.config != self.binding_receipt.derived_config:
            raise ValueError("prospective confirmatory result changes its derived config")
        candidate = self.region_support.candidate_evidence
        if len(self.candidate_family.ledger.members) != 1:
            raise ValueError("prospective confirmatory result requires one candidate")
        if self.candidate_family.ledger.dataset_or_projection != candidate.dataset_or_projection:
            raise ValueError("prospective confirmatory family binds another projection")
        if self.candidate_family.ledger.members[0].candidate_id != candidate.candidate_id:
            raise ValueError("prospective confirmatory family binds another candidate")
        if self.qualification_result.candidate_family_assessment != ObjectIdentity.from_record(
            self.candidate_family.assessment_id,
            self.candidate_family,
        ):
            raise ValueError("prospective confirmatory finalizer binds another family")
        if self.candidate_publication_count != 1 or self.finalizer_invocation_count != 1:
            raise ValueError("prospective confirmatory result must retain sole ownership")
        if (
            self.reveal_authority.kind is not StudyAuthorityKind.OUTCOME_REVEAL
            or not self.reveal_authority.allows_reveal
            or self.reveal_authority.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        ):
            raise ValueError("prospective confirmatory result lacks reveal authority")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("prospective confirmatory result must remain prospective")
        parse_utc_timestamp(self.evaluated_at_utc, field_name="evaluated_at_utc")


@dataclass(frozen=True, slots=True)
class ProspectiveConfirmatoryFiniteActionEvaluator:
    """Authorize, bind, publish and finalize one predeclared region candidate."""

    binding: ProspectiveConfirmatoryFiniteActionEvaluatorBinding
    payload_publisher: CandidatePayloadPublisher
    assessment_assembler: LawAssessmentAssembler
    family_assembler: CandidateFamilyAssembler
    qualification_service: ResponseLawQualificationService

    def evaluate(
        self,
        *,
        design: ConfirmatoryFiniteActionDesign,
        manifest: IdentificationEvidenceManifest,
        projection_plan: TaggedIdentificationProjectionPlan,
        projection: IdentificationEvidenceProjection,
        projection_extension: IdentificationEvidenceProjectionExtension,
        scaffold: FiniteActionCandidateScaffold,
        region_support_spec: ConfirmatoryFiniteActionRegionSupportSpec,
        config: ConfirmatoryFiniteActionConfig,
        binding_receipt: ConfirmatoryFiniteActionBindingReceipt,
        system: SystemSpec,
        reveal_authority: StudyOperationAuthority,
        authority_subject: ObjectIdentity,
        prerequisite_authority: ObjectIdentity,
        evaluated_at_utc: str,
    ) -> ProspectiveConfirmatoryFiniteActionEvaluation:
        require_study_authority(
            reveal_authority,
            kind=StudyAuthorityKind.OUTCOME_REVEAL,
            subject=authority_subject,
            prerequisite_authority=prerequisite_authority,
            grantee_id=self.binding.evaluator_id,
            at_utc=evaluated_at_utc,
        )
        if (
            not reveal_authority.allows_reveal
            or reveal_authority.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        ):
            raise PermissionError("prospective confirmatory evaluator lacks reveal access")
        if design.region_support_spec != ObjectIdentity.from_record(
            region_support_spec.spec_id,
            region_support_spec,
        ):
            raise ValueError("prospective confirmatory design binds another region spec")
        expected_config, expected_receipt = bind_confirmatory_finite_action_design(
            design=design,
            manifest=manifest,
            projection_plan=projection_plan,
            projection=projection,
            projection_extension=projection_extension,
            scaffold=scaffold,
            binding_implementation_id=self.binding.binding_implementation_id,
            binding_implementation_sha256=(self.binding.binding_implementation_sha256),
        )
        if config != expected_config or binding_receipt != expected_receipt:
            raise ValueError("prospective confirmatory binding receipt is not reproducible")
        producer = ConfirmatoryFiniteActionRegionSupportProducer(
            payload_publisher=self.payload_publisher,
            qualification_profile=self.binding.qualification_profile,
        )
        region_result = producer.produce(
            spec=region_support_spec,
            projection=projection,
            extension=projection_extension,
            config=config,
            scaffold=scaffold,
        )
        candidate = region_result.candidate_evidence
        assessment = self.assessment_assembler.assemble(
            system=system,
            candidate=candidate,
            qualification_profile=self.binding.qualification_profile,
        )
        binding_identity = ObjectIdentity.from_record(self.binding.binding_id, self.binding)
        ledger = CandidateFamilyLedger(
            family_id=f"candidate-family.{config.config_id}",
            system=ObjectIdentity.from_record(system.system_id, system),
            dataset_or_projection=config.projection,
            method_key=CONFIRMATORY_FINITE_ACTION_KEY,
            method_version=CONFIRMATORY_FINITE_ACTION_VERSION,
            method_kind=LawMethodKind.NONLINEAR_LOCAL,
            representation_kind=LawRepresentationKind.FINITE_ACTION_OPERATOR,
            chart_id=config.chart_id,
            axis_map=scaffold.axis_map,
            claim_unit_binding=scaffold.claim_unit_binding,
            claim_template=scaffold.claim_template,
            obligation_template=scaffold.obligation_template,
            members=(
                CandidateFamilyMember(
                    candidate_id=candidate.candidate_id,
                    config=ObjectIdentity.from_record(config.config_id, config),
                    complexity_rank=0,
                    disposition=CandidateRosterDisposition.ASSESS_REQUIRED,
                    predeclared_reason_codes=(),
                ),
            ),
            candidate_generation_rule_ids=(
                "generation.confirmatory-g1-g1r-single-predeclared-candidate",
            ),
            selection_threshold_ids=("threshold.confirmatory-g1-g1r-noncompensating-intersection",),
            development_input_ids=tuple(
                sorted((design.design_id, projection_plan.plan_id, region_support_spec.spec_id))
            ),
            selector_capability=binding_identity,
            selector_config=self.binding.qualification_profile,
            selector_implementation=binding_identity,
            multiplicity_family_id=config.multiplicity_family_id,
            multiplicity_rule_id="bonferroni-confirmatory-g1-g1r-fixed-family",
            randomness_seed_ids=(),
            tie_break_rule="single-predeclared-candidate-no-ranking",
            outcome_access=candidate.outcome_access,
            parent_visibility_ceilings=candidate.parent_visibility_ceilings,
            visibility_ceiling=candidate.visibility_ceiling,
        )
        family = self.family_assembler.assemble(ledger, (assessment,))
        qualification = self.qualification_service.qualify(
            system,
            config.projection,
            family,
        )
        evaluation_id = f"prospective-confirmatory-evaluation.{config.fingerprint()[:32]}"
        return ProspectiveConfirmatoryFiniteActionEvaluation(
            evaluation_id=evaluation_id,
            binding=binding_identity,
            design=ObjectIdentity.from_record(design.design_id, design),
            binding_receipt=binding_receipt,
            reveal_authority=reveal_authority,
            region_support=region_result,
            candidate_family=family,
            qualification_result=qualification,
            evaluated_at_utc=evaluated_at_utc,
            candidate_publication_count=self.binding.candidate_publication_count,
            finalizer_invocation_count=self.binding.finalizer_invocation_count,
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        )


__all__ = [
    'CONFIRMATORY_LOCAL_REGION_UNIT_COUNT',
    'CONFIRMATORY_LOCAL_REGION_CRITICAL_VALUE',
    'CONFIRMATORY_LOCAL_REGION_DEGREES_OF_FREEDOM',
    'CONFIRMATORY_LOCAL_REGION_FAMILY_ALPHA',
    'CONFIRMATORY_LOCAL_REGION_MATERIALITY',
    'CONFIRMATORY_LOCAL_REGION_MINIMUM_POSITIVE_UNITS',
    'CONFIRMATORY_LOCAL_REGION_PER_TEST_TAIL_PROBABILITY',
    'CONFIRMATORY_LOCAL_REGION_TEST_COUNT',
    'ConfirmatoryFiniteActionRegionSupportCompatibility',
    'ConfirmatoryFiniteActionRegionSupportProducer',
    'ConfirmatoryFiniteActionRegionSupportResult',
    'ConfirmatoryFiniteActionRegionSupportSpec',
    'ConfirmatoryFiniteActionRegionalTest',
    'ConfirmatoryFiniteActionTierRegionBinding',
    'ConfirmatoryFiniteActionUnitEffect',
    'ConfirmatoryRegionCompatibilityDisposition',
    'ConfirmatoryRegionalTestDisposition',
    'ProspectiveConfirmatoryFiniteActionEvaluation',
    'ProspectiveConfirmatoryFiniteActionEvaluatorBinding',
    'ProspectiveConfirmatoryFiniteActionEvaluator',
    'prospective_confirmatory_finite_action_evaluator_binding',
]
