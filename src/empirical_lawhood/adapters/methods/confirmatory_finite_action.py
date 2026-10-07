"Confirmatory finite-word identification on a fresh fixed evaluation panel.\n\nThis additive method is deliberately separate from the exploratory/bootstrap\n``finite-action.compatibility-set`` method.  It implements predeclared\nprefix closure, one-sided independent-unit inference, role-specific control\nsemantics and structural (not magnitude) agreement across denominator\nmembers.  It emits method evidence and a finite-action evaluator payload; the\nsole response-law finalizer remains the only terminal law authority.\n"

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ActionWordSupportStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.laws import LawRepresentationKind
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    NamedDecimal,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import HorizonSpec
from empirical_lawhood.planning.identification_evidence import (
    IdentificationEvidenceManifest,
    IdentificationEvidenceProjection,
    ObservationDisposition,
    ProjectedIdentificationObservation,
)
from .evidence_projection import TaggedIdentificationProjectionPlan
from empirical_lawhood.planning.identification_evidence_extensions import ActionOccurrenceBinding, AuthorityAxis, ComputabilityEffectStatus, IdentificationEvidenceProjectionExtension, IdentificationMethodEvidenceKind, NumericalValidityAxis, ObservationValidityAxis, PhysicalSinkAxis, ScientificTerminalAxis

from .contracts import (
    CandidateEvaluatorImplementation,
    CandidateMethodEvidenceReceipt,
    FiniteActionCellDisposition,
    FiniteActionCompatibilitySetExtension,
    FiniteActionNativeValue,
    FiniteActionResponseEntry,
    LawCandidateAxisMap,
    LawCandidateEvidence,
    MethodEvidenceAvailability,
)
from .evidence_projection_extensions import require_method_projection_truth
from .evidence_projection_templates import ConditionalIdentificationProjectionTemplate, ConditionalQualificationRosterSelectionReceipt, ConditionalQualificationRosterTemplate, ConditionalQualificationSelectionKind, IdentificationProjectionTemplateBindingReceipt, IdentificationProjectionRole, IdentificationProjectionTemplate
from .finite_action_identification import FiniteActionCandidateScaffold, FiniteActionProductionCell, FiniteActionProductionDisposition
from .law_assessment import CandidatePayloadPublisher


CONFIRMATORY_FINITE_ACTION_KEY = "finite-action.confirmatory-prefix"
CONFIRMATORY_FINITE_ACTION_VERSION = "1.0.0"
MAX_CONFIRMATORY_CELLS = 100_000
CONFIRMATORY_DESIGN_CONFIG_ID_RULE = "sha256-design-and-revealed-projection-prefix-16-bytes"
CONFIRMATORY_REGION_SUPPORT_SPEC_SCHEMA = (
    'empirical-lawhood/methods/confirmatory-finite-action-region-support-spec'
)


class ConfirmatoryActionRole(StrEnum):
    MEASURED_HOLD = "MEASURED_HOLD"
    TARGET_ACTIVE = "TARGET_ACTIVE"
    WRONG_SIGN_CONTROL = "WRONG_SIGN_CONTROL"
    FUTURE_NULL_CONTROL = "FUTURE_NULL_CONTROL"


@dataclass(frozen=True, slots=True)
class ConfirmatoryActionRoleBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-action-role-binding'

    binding_id: str
    action_word_id: str
    role: ConfirmatoryActionRole

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.action_word_id, field_name="action_word_id")


@dataclass(frozen=True, slots=True)
class ConfirmatoryUnitTierBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-unit-tier-binding'

    binding_id: str
    physical_independent_unit_id: str
    tier_id: str
    denominator_cell_id: str

    def __post_init__(self) -> None:
        for name, identifier in (
            ("binding_id", self.binding_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
            ("tier_id", self.tier_id),
            ("denominator_cell_id", self.denominator_cell_id),
        ):
            validate_stable_id(identifier, field_name=name)


@dataclass(frozen=True, slots=True)
class ConfirmatoryPrefixSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-prefix-spec'

    prefix_id: str
    order_index: int
    constituent_tier_ids: tuple[str, ...]
    degrees_of_freedom: int
    one_sided_tail_probability: Decimal
    one_sided_critical_value: Decimal
    critical_value_source_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.prefix_id, field_name="prefix_id")
        validate_stable_id(self.critical_value_source_id, field_name="critical_value_source_id")
        if self.order_index < 1:
            raise ValueError("confirmatory prefix order must be positive")
        if self.degrees_of_freedom < 1:
            raise ValueError("confirmatory Student degrees of freedom must be positive")
        require_sorted_unique_strings(
            self.constituent_tier_ids,
            field_name="constituent_tier_ids",
            allow_empty=False,
        )
        for name, value in (
            ("one_sided_tail_probability", self.one_sided_tail_probability),
            ("one_sided_critical_value", self.one_sided_critical_value),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if not Decimal(0) < self.one_sided_tail_probability < Decimal(1):
            raise ValueError("confirmatory one-sided tail probability is invalid")
        if self.one_sided_critical_value == 0:
            raise ValueError("confirmatory one-sided critical value must be positive")


@dataclass(frozen=True, slots=True)
class ConfirmatoryResponseQuantity(CanonicalRecord):
    """Projection fields needed for target, causal and pre-cutoff checks."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-response-quantity'

    binding_id: str
    response_projection_value_id: str
    precutoff_projection_value_id: str
    onset_clock_projection_value_id: str
    quantity_id: str
    native_unit: str
    native_frame_id: str
    response_clock_id: str
    maximum_null_absolute_effect: Decimal
    maximum_precutoff_absolute_effect: Decimal
    earliest_active_response_clock: Decimal
    latest_active_response_clock: Decimal
    earliest_future_response_clock: Decimal

    def __post_init__(self) -> None:
        for name, identifier in (
            ("binding_id", self.binding_id),
            ("response_projection_value_id", self.response_projection_value_id),
            ("precutoff_projection_value_id", self.precutoff_projection_value_id),
            ("onset_clock_projection_value_id", self.onset_clock_projection_value_id),
            ("quantity_id", self.quantity_id),
            ("native_frame_id", self.native_frame_id),
            ("response_clock_id", self.response_clock_id),
        ):
            validate_stable_id(identifier, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        for name, decimal_value in (
            ("maximum_null_absolute_effect", self.maximum_null_absolute_effect),
            ("maximum_precutoff_absolute_effect", self.maximum_precutoff_absolute_effect),
            ("earliest_active_response_clock", self.earliest_active_response_clock),
            ("latest_active_response_clock", self.latest_active_response_clock),
            ("earliest_future_response_clock", self.earliest_future_response_clock),
        ):
            validate_decimal(decimal_value, field_name=name, minimum=Decimal(0))
        if self.latest_active_response_clock < self.earliest_active_response_clock:
            raise ValueError("confirmatory active response clock interval is reversed")


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-finite-action-config'

    config_id: str
    method_key: str
    method_version: str
    implementation_sha256: str
    projection: ObjectIdentity
    projection_extension: ObjectIdentity
    horizon: ObjectIdentity
    prepared_denominator_id: str
    chart_id: str
    retained_history_ids: tuple[str, ...]
    action_words: tuple[OccurrenceActionWord, ...]
    action_role_bindings: tuple[ConfirmatoryActionRoleBinding, ...]
    axis_map: LawCandidateAxisMap
    unit_tier_bindings: tuple[ConfirmatoryUnitTierBinding, ...]
    prefixes: tuple[ConfirmatoryPrefixSpec, ...]
    evaluation_split_ids: tuple[str, ...]
    response_quantity: ConfirmatoryResponseQuantity
    minimum_complete_units_per_tier: int
    minimum_positive_units_per_tier: int
    materiality: Decimal
    family_alpha: Decimal
    simultaneous_test_count: int
    multiplicity_family_id: str
    information_cutoff_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.axis_map, LawCandidateAxisMap):
            raise TypeError("confirmatory config requires a LawCandidateAxisMap")
        for name, identifier in (
            ("config_id", self.config_id),
            ("method_key", self.method_key),
            ("prepared_denominator_id", self.prepared_denominator_id),
            ("chart_id", self.chart_id),
            ("multiplicity_family_id", self.multiplicity_family_id),
            ("information_cutoff_id", self.information_cutoff_id),
        ):
            validate_stable_id(identifier, field_name=name)
        if self.method_key != CONFIRMATORY_FINITE_ACTION_KEY:
            raise ValueError("confirmatory finite-action config selects another method")
        validate_semantic_version(self.method_version)
        if self.method_version != CONFIRMATORY_FINITE_ACTION_VERSION:
            raise ValueError("confirmatory finite-action config selects another version")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.projection.object_schema != IdentificationEvidenceProjection.SCHEMA:
            raise ValueError("confirmatory config requires an evidence projection")
        if self.projection_extension.object_schema != (
            IdentificationEvidenceProjectionExtension.SCHEMA
        ):
            raise ValueError("confirmatory config requires the projection companion")
        if self.horizon.object_schema != HorizonSpec.SCHEMA:
            raise ValueError("confirmatory config requires the exact response horizon")
        require_sorted_unique_strings(
            self.retained_history_ids,
            field_name="retained_history_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.evaluation_split_ids,
            field_name="evaluation_split_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.action_words, attribute="word_id", field_name="action_words")
        require_sorted_unique_ids(
            self.action_role_bindings,
            attribute="binding_id",
            field_name="action_role_bindings",
        )
        require_sorted_unique_ids(
            self.unit_tier_bindings,
            attribute="binding_id",
            field_name="unit_tier_bindings",
        )
        require_sorted_unique_ids(self.prefixes, attribute="prefix_id", field_name="prefixes")
        if not self.action_words or not self.unit_tier_bindings or not self.prefixes:
            raise ValueError("confirmatory finite-action design cannot be empty")
        word_ids = {value.word_id for value in self.action_words}
        if {value.action_word_id for value in self.action_role_bindings} != word_ids:
            raise ValueError("confirmatory action roles must cover every word exactly once")
        role_counts = {
            role: sum(value.role is role for value in self.action_role_bindings)
            for role in ConfirmatoryActionRole
        }
        if role_counts != {
            ConfirmatoryActionRole.MEASURED_HOLD: 1,
            ConfirmatoryActionRole.TARGET_ACTIVE: 1,
            ConfirmatoryActionRole.WRONG_SIGN_CONTROL: 1,
            ConfirmatoryActionRole.FUTURE_NULL_CONTROL: 1,
        }:
            raise ValueError("confirmatory identification requires the exact four-role word chart")
        if any(
            value.denominator_id != self.prepared_denominator_id
            or value.horizon_id != self.horizon.object_id
            or value.retained_history_id not in self.retained_history_ids
            for value in self.action_words
        ):
            raise ValueError("confirmatory action words change denominator/history/horizon")
        unit_ids = tuple(
            sorted(value.physical_independent_unit_id for value in self.unit_tier_bindings)
        )
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("confirmatory physical units may bind only one tier")
        tier_ids = {value.tier_id for value in self.unit_tier_bindings}
        ordered = tuple(sorted(self.prefixes, key=lambda value: value.order_index))
        if tuple(value.order_index for value in ordered) != tuple(range(1, len(ordered) + 1)):
            raise ValueError("confirmatory prefixes must have contiguous order")
        previous: set[str] = set()
        for prefix in ordered:
            current = set(prefix.constituent_tier_ids)
            if not current <= tier_ids or not previous < current:
                raise ValueError("confirmatory prefixes must be strict nested tier prefixes")
            previous = current
        if previous != tier_ids:
            raise ValueError("deepest confirmatory prefix must cover every declared tier")
        counts = {
            tier: sum(value.tier_id == tier for value in self.unit_tier_bindings)
            for tier in tier_ids
        }
        if self.minimum_complete_units_per_tier < 2 or any(
            count < self.minimum_complete_units_per_tier for count in counts.values()
        ):
            raise ValueError("confirmatory tiers lack the required independent units")
        if not 1 <= self.minimum_positive_units_per_tier <= (self.minimum_complete_units_per_tier):
            raise ValueError("confirmatory positive-count rule is outside the complete roster")
        for name, decimal_value in (
            ("materiality", self.materiality),
            ("family_alpha", self.family_alpha),
        ):
            validate_decimal(decimal_value, field_name=name, minimum=Decimal(0))
        if not Decimal(0) < self.family_alpha < Decimal(1):
            raise ValueError("confirmatory family alpha must lie between zero and one")
        expected_test_count = len(self.axis_map.bindings) * len(self.prefixes)
        if self.simultaneous_test_count != expected_test_count:
            raise ValueError("confirmatory simultaneous test count differs from its axis product")
        per_test_alpha = self.family_alpha / Decimal(self.simultaneous_test_count)
        if any(
            prefix.one_sided_tail_probability != per_test_alpha
            or prefix.degrees_of_freedom
            != sum(counts[tier] for tier in prefix.constituent_tier_ids) - 1
            for prefix in self.prefixes
        ):
            raise ValueError("confirmatory Student/Bonferroni prefix metadata differs")
        if len(self.axis_map.bindings) * len(self.prefixes) * len(self.action_words) > (
            MAX_CONFIRMATORY_CELLS
        ):
            raise ValueError("confirmatory finite-action config exceeds its cell bound")

    @property
    def physical_independent_unit_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(value.physical_independent_unit_id for value in self.unit_tier_bindings)
        )


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionDesign(CanonicalRecord):
    "Outcome-blind design from which one revealed config is derived."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-finite-action-design'

    design_id: str
    method_key: str
    method_version: str
    implementation_sha256: str
    projection_plan: ObjectIdentity
    allowed_consumer_id: str
    scaffold_template: ObjectIdentity
    expected_physical_unit_ids: tuple[str, ...]
    expected_source_payload_role: str
    config_id_derivation_rule: str
    region_support_spec: ObjectIdentity
    horizon: ObjectIdentity
    prepared_denominator_id: str
    chart_id: str
    retained_history_ids: tuple[str, ...]
    action_words: tuple[OccurrenceActionWord, ...]
    action_role_bindings: tuple[ConfirmatoryActionRoleBinding, ...]
    axis_map: LawCandidateAxisMap
    unit_tier_bindings: tuple[ConfirmatoryUnitTierBinding, ...]
    prefixes: tuple[ConfirmatoryPrefixSpec, ...]
    evaluation_split_ids: tuple[str, ...]
    response_quantity: ConfirmatoryResponseQuantity
    minimum_complete_units_per_tier: int
    minimum_positive_units_per_tier: int
    materiality: Decimal
    family_alpha: Decimal
    simultaneous_test_count: int
    multiplicity_family_id: str
    information_cutoff_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    conditional_roster_selection: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        for field_name, identifier in (
            ("design_id", self.design_id),
            ("allowed_consumer_id", self.allowed_consumer_id),
            ("expected_source_payload_role", self.expected_source_payload_role),
            ("config_id_derivation_rule", self.config_id_derivation_rule),
        ):
            validate_stable_id(identifier, field_name=field_name)
        if self.projection_plan.object_schema != TaggedIdentificationProjectionPlan.SCHEMA:
            raise ValueError("confirmatory design requires an exact projection plan")
        if self.scaffold_template.object_schema != FiniteActionCandidateScaffold.SCHEMA:
            raise ValueError("confirmatory design requires an exact candidate scaffold")
        if self.region_support_spec.object_schema != CONFIRMATORY_REGION_SUPPORT_SPEC_SCHEMA:
            raise ValueError("confirmatory design requires the exact region-support spec")
        require_sorted_unique_strings(
            self.expected_physical_unit_ids,
            field_name="expected_physical_unit_ids",
            allow_empty=False,
        )
        if self.config_id_derivation_rule != CONFIRMATORY_DESIGN_CONFIG_ID_RULE:
            raise ValueError("confirmatory design selects another config-ID rule")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("confirmatory design must be outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("confirmatory design must remain prospective")
        if self.conditional_roster_selection is not None and (
            self.conditional_roster_selection.object_schema
            != ConditionalQualificationRosterSelectionReceipt.SCHEMA
        ):
            raise ValueError("confirmatory design names another roster-selection schema")
        # Reuse the unchanged validator through a status-free placeholder
        # identity pair.  No projection values are present in this object.
        placeholder = ObjectIdentity(
            object_id=f"placeholder.{self.design_id}",
            object_schema=IdentificationEvidenceProjection.SCHEMA,
            object_version="1.0.0",
            object_fingerprint="0" * 64,
        )
        placeholder_extension = ObjectIdentity(
            object_id=f"placeholder-extension.{self.design_id}",
            object_schema=IdentificationEvidenceProjectionExtension.SCHEMA,
            object_version="1.0.0",
            object_fingerprint="0" * 64,
        )
        config = ConfirmatoryFiniteActionConfig(
            config_id=f"prospective-validation.{self.design_id}",
            method_key=self.method_key,
            method_version=self.method_version,
            implementation_sha256=self.implementation_sha256,
            projection=placeholder,
            projection_extension=placeholder_extension,
            horizon=self.horizon,
            prepared_denominator_id=self.prepared_denominator_id,
            chart_id=self.chart_id,
            retained_history_ids=self.retained_history_ids,
            action_words=self.action_words,
            action_role_bindings=self.action_role_bindings,
            axis_map=self.axis_map,
            unit_tier_bindings=self.unit_tier_bindings,
            prefixes=self.prefixes,
            evaluation_split_ids=self.evaluation_split_ids,
            response_quantity=self.response_quantity,
            minimum_complete_units_per_tier=self.minimum_complete_units_per_tier,
            minimum_positive_units_per_tier=self.minimum_positive_units_per_tier,
            materiality=self.materiality,
            family_alpha=self.family_alpha,
            simultaneous_test_count=self.simultaneous_test_count,
            multiplicity_family_id=self.multiplicity_family_id,
            information_cutoff_id=self.information_cutoff_id,
        )
        if config.physical_independent_unit_ids != self.expected_physical_unit_ids:
            raise ValueError("confirmatory design unit roster differs from tier bindings")


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionProspectiveCore(CanonicalRecord):
    """All confirmatory science that is independent of a future manifest fingerprint."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-finite-action-prospective-core'

    method_key: str
    method_version: str
    implementation_sha256: str
    allowed_consumer_id: str
    scaffold_template: ObjectIdentity
    expected_physical_unit_ids: tuple[str, ...]
    expected_source_payload_role: str
    config_id_derivation_rule: str
    region_support_spec: ObjectIdentity
    horizon: ObjectIdentity
    prepared_denominator_id: str
    chart_id: str
    retained_history_ids: tuple[str, ...]
    action_words: tuple[OccurrenceActionWord, ...]
    action_role_bindings: tuple[ConfirmatoryActionRoleBinding, ...]
    axis_map: LawCandidateAxisMap
    unit_tier_bindings: tuple[ConfirmatoryUnitTierBinding, ...]
    prefixes: tuple[ConfirmatoryPrefixSpec, ...]
    evaluation_split_ids: tuple[str, ...]
    response_quantity: ConfirmatoryResponseQuantity
    minimum_complete_units_per_tier: int
    minimum_positive_units_per_tier: int
    materiality: Decimal
    family_alpha: Decimal
    simultaneous_test_count: int
    multiplicity_family_id: str
    information_cutoff_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    conditional_roster_selection: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("confirmatory prospective core must remain outcome-blind/prospective")
        if self.conditional_roster_selection is not None and (
            self.conditional_roster_selection.object_schema
            != ConditionalQualificationRosterSelectionReceipt.SCHEMA
        ):
            raise ValueError("confirmatory core names another roster-selection schema")
        _materialize_confirmatory_design(
            design_id="prospective-validation.confirmatory-core",
            projection_plan=ObjectIdentity(
                object_id="projection-plan.prospective-validation.confirmatory-core",
                object_schema=TaggedIdentificationProjectionPlan.SCHEMA,
                object_version="1.0.0",
                object_fingerprint="0" * 64,
            ),
            core=self,
        )


def _materialize_confirmatory_design(
    *,
    design_id: str,
    projection_plan: ObjectIdentity,
    core: ConfirmatoryFiniteActionProspectiveCore,
) -> ConfirmatoryFiniteActionDesign:
    return ConfirmatoryFiniteActionDesign(
        design_id=design_id,
        method_key=core.method_key,
        method_version=core.method_version,
        implementation_sha256=core.implementation_sha256,
        projection_plan=projection_plan,
        allowed_consumer_id=core.allowed_consumer_id,
        scaffold_template=core.scaffold_template,
        expected_physical_unit_ids=core.expected_physical_unit_ids,
        expected_source_payload_role=core.expected_source_payload_role,
        config_id_derivation_rule=core.config_id_derivation_rule,
        region_support_spec=core.region_support_spec,
        horizon=core.horizon,
        prepared_denominator_id=core.prepared_denominator_id,
        chart_id=core.chart_id,
        retained_history_ids=core.retained_history_ids,
        action_words=core.action_words,
        action_role_bindings=core.action_role_bindings,
        axis_map=core.axis_map,
        unit_tier_bindings=core.unit_tier_bindings,
        prefixes=core.prefixes,
        evaluation_split_ids=core.evaluation_split_ids,
        response_quantity=core.response_quantity,
        minimum_complete_units_per_tier=core.minimum_complete_units_per_tier,
        minimum_positive_units_per_tier=core.minimum_positive_units_per_tier,
        materiality=core.materiality,
        family_alpha=core.family_alpha,
        simultaneous_test_count=core.simultaneous_test_count,
        multiplicity_family_id=core.multiplicity_family_id,
        information_cutoff_id=core.information_cutoff_id,
        outcome_access=core.outcome_access,
        visibility_ceiling=core.visibility_ceiling,
        conditional_roster_selection=core.conditional_roster_selection,
    )


@dataclass(frozen=True, slots=True)
class ConditionalConfirmatoryUnitTierSlot(CanonicalRecord):
    """One confirmatory BT unit selected with its complete acquisition block."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/conditional-confirmatory-unit-tier-slot'

    binding_id: str
    slot_id: str
    tier_id: str
    primary_physical_independent_unit_id: str
    primary_denominator_cell_id: str
    alternative_physical_independent_unit_id: str
    alternative_denominator_cell_id: str

    def __post_init__(self) -> None:
        for field_name, value in (
            ("binding_id", self.binding_id),
            ("slot_id", self.slot_id),
            ("tier_id", self.tier_id),
            (
                "primary_physical_independent_unit_id",
                self.primary_physical_independent_unit_id,
            ),
            ("primary_denominator_cell_id", self.primary_denominator_cell_id),
            (
                "alternative_physical_independent_unit_id",
                self.alternative_physical_independent_unit_id,
            ),
            ("alternative_denominator_cell_id", self.alternative_denominator_cell_id),
        ):
            validate_stable_id(value, field_name=field_name)
        if (
            self.primary_physical_independent_unit_id
            == self.alternative_physical_independent_unit_id
            or self.primary_denominator_cell_id == self.alternative_denominator_cell_id
        ):
            raise ValueError("conditional confirmatory primary/alternative aliases")


@dataclass(frozen=True, slots=True)
class ConditionalConfirmatoryFiniteActionCoreTemplate(CanonicalRecord):
    """Preissue confirmatory core with exact whole-block unit alternatives."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/conditional-confirmatory-finite-action-core-template'
    )

    template_id: str
    roster_template: ConditionalQualificationRosterTemplate
    primary_core: ConfirmatoryFiniteActionProspectiveCore
    unit_tier_slots: tuple[ConditionalConfirmatoryUnitTierSlot, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.template_id, field_name="template_id")
        require_sorted_unique_ids(
            self.unit_tier_slots,
            attribute="binding_id",
            field_name="unit_tier_slots",
        )
        slots = {value.slot_id: value for value in self.roster_template.slots}
        by_slot = {value.slot_id: value for value in self.unit_tier_slots}
        if (
            len(by_slot) != len(self.unit_tier_slots)
            or set(by_slot) != set(slots)
            or len(by_slot) != 18
        ):
            raise ValueError("conditional confirmatory core requires all 18 exact slots")
        primary_bindings = {
            value.physical_independent_unit_id: value
            for value in self.primary_core.unit_tier_bindings
        }
        if len(primary_bindings) != len(self.primary_core.unit_tier_bindings):
            raise ValueError("conditional confirmatory primary unit bindings alias")
        for slot_id, value in by_slot.items():
            roster_slot = slots[slot_id]
            primary_binding = primary_bindings.get(value.primary_physical_independent_unit_id)
            if (
                value.tier_id != roster_slot.tier_id
                or value.primary_physical_independent_unit_id
                not in roster_slot.primary.physical_unit_instance_ids
                or value.alternative_physical_independent_unit_id
                not in roster_slot.conditional_alternative.physical_unit_instance_ids
                or primary_binding is None
                or primary_binding.binding_id != value.binding_id
                or primary_binding.tier_id != value.tier_id
                or primary_binding.denominator_cell_id != value.primary_denominator_cell_id
            ):
                raise ValueError("conditional confirmatory slot differs from roster/core")
        if set(primary_bindings) != {
            value.primary_physical_independent_unit_id for value in self.unit_tier_slots
        }:
            raise ValueError("conditional confirmatory primary core has foreign units")
        if self.primary_core.conditional_roster_selection is not None:
            raise ValueError("conditional confirmatory primary core is already selected")


def materialize_conditional_confirmatory_finite_action_core(
    *,
    template: ConditionalConfirmatoryFiniteActionCoreTemplate,
    selection: ConditionalQualificationRosterSelectionReceipt,
    region_support_spec: ObjectIdentity,
    scaffold_template: ObjectIdentity,
) -> ConfirmatoryFiniteActionProspectiveCore:
    """Materialize exactly 18 selected BT units before ordinary design binding."""

    if selection.roster_template != template.roster_template:
        raise ValueError("conditional confirmatory selection names another roster")
    if region_support_spec.object_schema != CONFIRMATORY_REGION_SUPPORT_SPEC_SCHEMA:
        raise ValueError("conditional confirmatory core requires a region-support spec")
    if scaffold_template.object_schema != FiniteActionCandidateScaffold.SCHEMA:
        raise ValueError("conditional confirmatory core requires a candidate scaffold")
    selections = {value.slot_id: value for value in selection.selections}
    unit_bindings = tuple(
        sorted(
            (
                ConfirmatoryUnitTierBinding(
                    binding_id=value.binding_id,
                    physical_independent_unit_id=(
                        value.primary_physical_independent_unit_id
                        if selections[value.slot_id].kind
                        is ConditionalQualificationSelectionKind.PRIMARY
                        else value.alternative_physical_independent_unit_id
                    ),
                    tier_id=value.tier_id,
                    denominator_cell_id=(
                        value.primary_denominator_cell_id
                        if selections[value.slot_id].kind
                        is ConditionalQualificationSelectionKind.PRIMARY
                        else value.alternative_denominator_cell_id
                    ),
                )
                for value in template.unit_tier_slots
            ),
            key=lambda value: value.binding_id,
        )
    )
    selected_ids = tuple(sorted(value.physical_independent_unit_id for value in unit_bindings))
    if len(selected_ids) != 18 or len(set(selected_ids)) != 18:
        raise ValueError("conditional confirmatory materialization aliases selected units")
    return replace(
        template.primary_core,
        expected_physical_unit_ids=selected_ids,
        unit_tier_bindings=unit_bindings,
        region_support_spec=region_support_spec,
        scaffold_template=scaffold_template,
        conditional_roster_selection=ObjectIdentity.from_record(
            selection.receipt_id,
            selection,
        ),
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionDesignTemplate(CanonicalRecord):
    """Preissue template that defers only the future projection-plan fingerprint."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-finite-action-design-template'

    template_id: str
    expected_design_id: str
    expected_projection_plan_id: str
    projection_template: ObjectIdentity
    core: ConfirmatoryFiniteActionProspectiveCore
    binder_implementation_id: str
    binder_implementation_sha256: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    conditional_core_template: ConditionalConfirmatoryFiniteActionCoreTemplate | None = None
    conditional_projection_template: ObjectIdentity | None = None
    conditional_roster_selection: ObjectIdentity | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            ("template_id", self.template_id),
            ("expected_design_id", self.expected_design_id),
            ("expected_projection_plan_id", self.expected_projection_plan_id),
            ("binder_implementation_id", self.binder_implementation_id),
        ):
            validate_stable_id(value, field_name=field_name)
        validate_sha256(
            self.binder_implementation_sha256,
            field_name="binder_implementation_sha256",
        )
        if self.projection_template.object_schema != IdentificationProjectionTemplate.SCHEMA:
            raise ValueError("confirmatory template requires a projection template")
        if (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("confirmatory design template must remain outcome-blind/prospective")
        if self.conditional_core_template is None:
            if self.conditional_projection_template is not None:
                raise ValueError("confirmatory template has a conditional projection without core")
        else:
            if (
                self.conditional_projection_template is None
                or self.conditional_projection_template.object_schema
                != ConditionalIdentificationProjectionTemplate.SCHEMA
                or self.core != self.conditional_core_template.primary_core
                or self.conditional_roster_selection is not None
            ):
                raise ValueError("unresolved confirmatory conditional topology differs")
        if self.core.conditional_roster_selection != self.conditional_roster_selection:
            raise ValueError("confirmatory template/core selection lineage differs")
        if self.conditional_roster_selection is not None and (
            self.conditional_roster_selection.object_schema
            != ConditionalQualificationRosterSelectionReceipt.SCHEMA
        ):
            raise ValueError("confirmatory template names another roster-selection schema")


def materialize_conditional_confirmatory_finite_action_design_template(
    *,
    template: ConfirmatoryFiniteActionDesignTemplate,
    selection: ConditionalQualificationRosterSelectionReceipt,
    projection_template: IdentificationProjectionTemplate,
    region_support_spec: ObjectIdentity,
    scaffold_template: ObjectIdentity,
) -> ConfirmatoryFiniteActionDesignTemplate:
    """Resolve one preissue conditional design without reading scientific values."""

    conditional_core = template.conditional_core_template
    conditional_projection = template.conditional_projection_template
    if conditional_core is None or conditional_projection is None:
        raise ValueError("confirmatory design template has no conditional topology")
    if template.conditional_roster_selection is not None:
        raise ValueError("confirmatory design template is already conditionally selected")
    selection_identity = ObjectIdentity.from_record(selection.receipt_id, selection)
    if (
        projection_template.role is not IdentificationProjectionRole.PRIMARY_LAW
        or projection_template.conditional_roster_selection != selection_identity
        or conditional_projection.object_id != f"conditional.{projection_template.template_id}"
    ):
        raise ValueError("confirmatory design selects another projection topology")
    core = materialize_conditional_confirmatory_finite_action_core(
        template=conditional_core,
        selection=selection,
        region_support_spec=region_support_spec,
        scaffold_template=scaffold_template,
    )
    return replace(
        template,
        projection_template=ObjectIdentity.from_record(
            projection_template.template_id,
            projection_template,
        ),
        core=core,
        conditional_core_template=None,
        conditional_projection_template=None,
        conditional_roster_selection=selection_identity,
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionDesignTemplateBindingReceipt(CanonicalRecord):
    """Topology-only post-manifest materialization; never a scientific result."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/methods/confirmatory-finite-action-design-template-binding-receipt'
    )

    receipt_id: str
    template: ObjectIdentity
    projection_template: ObjectIdentity
    projection_template_binding: ObjectIdentity
    projection_plan: ObjectIdentity
    design: ObjectIdentity
    binder_implementation_id: str
    binder_implementation_sha256: str
    compact_values_read: bool
    grants_scientific_promotion: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.binder_implementation_id,
            field_name="binder_implementation_id",
        )
        validate_sha256(
            self.binder_implementation_sha256,
            field_name="binder_implementation_sha256",
        )
        expected = (
            (self.template, ConfirmatoryFiniteActionDesignTemplate.SCHEMA),
            (self.projection_template, IdentificationProjectionTemplate.SCHEMA),
            (
                self.projection_template_binding,
                IdentificationProjectionTemplateBindingReceipt.SCHEMA,
            ),
            (self.projection_plan, TaggedIdentificationProjectionPlan.SCHEMA),
            (self.design, ConfirmatoryFiniteActionDesign.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected):
            raise ValueError("confirmatory design-template receipt binds another schema")
        if self.compact_values_read or self.grants_scientific_promotion:
            raise ValueError("confirmatory design-template binding cannot read/promote science")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("confirmatory design-template binding requires evaluator reveal")


def bind_confirmatory_finite_action_design_template(
    *,
    receipt_id: str,
    template: ConfirmatoryFiniteActionDesignTemplate,
    projection_template: IdentificationProjectionTemplate,
    projection_template_binding: IdentificationProjectionTemplateBindingReceipt,
    projection_plan: TaggedIdentificationProjectionPlan,
    binder_implementation_id: str,
    binder_implementation_sha256: str,
) -> tuple[
    ConfirmatoryFiniteActionDesign,
    ConfirmatoryFiniteActionDesignTemplateBindingReceipt,
]:
    """Materialize the ordinary design only after the exact future plan exists."""

    if template.conditional_core_template is not None:
        raise ValueError("conditional confirmatory template requires roster selection")
    if (
        binder_implementation_id != template.binder_implementation_id
        or binder_implementation_sha256 != template.binder_implementation_sha256
    ):
        raise ValueError("confirmatory design-template binder implementation differs")
    projection_template_identity = ObjectIdentity.from_record(
        projection_template.template_id,
        projection_template,
    )
    projection_plan_identity = ObjectIdentity.from_record(
        projection_plan.plan_id,
        projection_plan,
    )
    if (
        template.projection_template != projection_template_identity
        or template.expected_projection_plan_id != projection_plan.plan_id
        or projection_template_binding.template != projection_template_identity
        or projection_template_binding.projection_plan != projection_plan_identity
        or projection_template.allowed_consumer_id != template.core.allowed_consumer_id
        or projection_plan.allowed_consumer_id != template.core.allowed_consumer_id
        or projection_plan.projection_config != projection_template.projection_config
        or projection_plan.claim_unit_binding.binding_id
        != projection_template.claim_unit_binding_id
        or projection_plan.value_partition != projection_template.value_partition
        or projection_template.conditional_roster_selection != template.conditional_roster_selection
    ):
        raise ValueError("confirmatory design-template projection topology differs")
    design = _materialize_confirmatory_design(
        design_id=template.expected_design_id,
        projection_plan=projection_plan_identity,
        core=template.core,
    )
    receipt = ConfirmatoryFiniteActionDesignTemplateBindingReceipt(
        receipt_id=receipt_id,
        template=ObjectIdentity.from_record(template.template_id, template),
        projection_template=projection_template_identity,
        projection_template_binding=ObjectIdentity.from_record(
            projection_template_binding.binding_id,
            projection_template_binding,
        ),
        projection_plan=projection_plan_identity,
        design=ObjectIdentity.from_record(design.design_id, design),
        binder_implementation_id=binder_implementation_id,
        binder_implementation_sha256=binder_implementation_sha256,
        compact_values_read=False,
        grants_scientific_promotion=False,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    return design, receipt


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionBindingReceipt(CanonicalRecord):
    """Identity-only receipt for deterministic post-reveal config binding."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-finite-action-binding-receipt'

    receipt_id: str
    design: ObjectIdentity
    manifest: ObjectIdentity
    projection_plan: ObjectIdentity
    projection: ObjectIdentity
    projection_extension: ObjectIdentity
    scaffold: ObjectIdentity
    derived_config: ObjectIdentity
    binding_implementation_id: str
    binding_implementation_sha256: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    scientific_verdict: None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.binding_implementation_id,
            field_name="binding_implementation_id",
        )
        validate_sha256(
            self.binding_implementation_sha256,
            field_name="binding_implementation_sha256",
        )
        expected_schemas = (
            (self.design, ConfirmatoryFiniteActionDesign.SCHEMA),
            (self.manifest, IdentificationEvidenceManifest.SCHEMA),
            (self.projection_plan, TaggedIdentificationProjectionPlan.SCHEMA),
            (self.projection, IdentificationEvidenceProjection.SCHEMA),
            (
                self.projection_extension,
                IdentificationEvidenceProjectionExtension.SCHEMA,
            ),
            (self.scaffold, FiniteActionCandidateScaffold.SCHEMA),
            (self.derived_config, ConfirmatoryFiniteActionConfig.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected_schemas):
            raise ValueError("confirmatory binding receipt contains another schema")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("confirmatory binding receipt is evaluator-reveal only")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("confirmatory binding receipt must remain prospective")
        if self.scientific_verdict is not None:
            raise ValueError("confirmatory binding receipt cannot carry a verdict")


def _derived_confirmatory_config_id(
    design: ConfirmatoryFiniteActionDesign,
    projection: IdentificationEvidenceProjection,
) -> str:
    digest = hashlib.sha256(
        design.canonical_bytes() + b"\0" + projection.canonical_bytes()
    ).hexdigest()
    return f"confirmatory-config.bound.{digest[:32]}"


def bind_confirmatory_finite_action_design(
    *,
    design: ConfirmatoryFiniteActionDesign,
    manifest: IdentificationEvidenceManifest,
    projection_plan: TaggedIdentificationProjectionPlan,
    projection: IdentificationEvidenceProjection,
    projection_extension: IdentificationEvidenceProjectionExtension,
    scaffold: FiniteActionCandidateScaffold,
    binding_implementation_id: str,
    binding_implementation_sha256: str,
) -> tuple[ConfirmatoryFiniteActionConfig, ConfirmatoryFiniteActionBindingReceipt]:
    """Bind revealed evidence without selecting or changing scientific parameters."""

    validate_stable_id(binding_implementation_id, field_name="binding_implementation_id")
    validate_sha256(binding_implementation_sha256, field_name="binding_implementation_sha256")
    manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
    plan_identity = ObjectIdentity.from_record(projection_plan.plan_id, projection_plan)
    projection_identity = ObjectIdentity.from_record(projection.projection_id, projection)
    extension_identity = ObjectIdentity.from_record(
        projection_extension.extension_id,
        projection_extension,
    )
    scaffold_identity = ObjectIdentity.from_record(scaffold.scaffold_id, scaffold)
    if design.projection_plan != plan_identity:
        raise ValueError("confirmatory design binds another projection plan")
    if projection_plan.manifest != manifest_identity:
        raise ValueError("confirmatory projection plan binds another manifest")
    if (
        projection.manifest != manifest_identity
        or projection_extension.manifest != manifest_identity
    ):
        raise ValueError("confirmatory revealed evidence binds another manifest")
    if projection_extension.projection != projection_identity:
        raise ValueError("confirmatory companion binds another projection")
    if (
        design.allowed_consumer_id != projection_plan.allowed_consumer_id
        or design.allowed_consumer_id != projection.allowed_consumer_id
        or design.allowed_consumer_id not in manifest.allowed_consumer_ids
    ):
        raise ValueError("confirmatory consumer identity differs")
    if (
        projection.projection_capability != projection_plan.projection_capability
        or projection.projection_config != projection_plan.projection_config
        or projection.projection_implementation != projection_plan.projection_implementation
        or projection.claim_unit_binding != projection_plan.claim_unit_binding
    ):
        raise ValueError("confirmatory projection differs from its issued plan")
    if design.scaffold_template != scaffold_identity:
        raise ValueError("confirmatory design binds another candidate scaffold")
    if (
        design.information_cutoff_id != manifest.information_cutoff_id
        or design.information_cutoff_id != projection.information_cutoff_id
    ):
        raise ValueError("confirmatory information cutoff differs")
    expected_units = design.expected_physical_unit_ids
    if (
        projection.claim_unit_binding.independent_unit_instance_ids != expected_units
        or projection_extension.independent_unit_ids != expected_units
        or scaffold.claim_unit_binding
        != ObjectIdentity.from_record(
            projection.claim_unit_binding.binding_id,
            projection.claim_unit_binding,
        )
    ):
        raise ValueError("confirmatory physical-unit/claim-unit binding differs")
    payloads = {value.payload_id: value for value in manifest.payloads}
    if set(projection.source_payload_ids) - set(payloads) or any(
        payloads[payload_id].artifact.role != design.expected_source_payload_role
        for payload_id in projection.source_payload_ids
    ):
        raise ValueError("confirmatory projection uses another source payload role")
    expected_words = {
        ObjectIdentity.from_record(value.word_id, value) for value in design.action_words
    }
    if tuple(sorted(manifest.action_words, key=lambda value: value.word_id)) != (
        design.action_words
    ):
        raise ValueError("confirmatory manifest changes the exact action-word chart")
    observed_words = {
        value.action_word for value in projection.observations if value.action_word is not None
    }
    if observed_words != expected_words:
        raise ValueError("confirmatory projection changes the exact action-word chart")
    if projection.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
        raise ValueError("confirmatory binding requires evaluator-revealed projection")
    if (
        projection.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        or scaffold.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
    ):
        raise ValueError("confirmatory binding cannot lower prospective visibility")
    config = ConfirmatoryFiniteActionConfig(
        config_id=_derived_confirmatory_config_id(design, projection),
        method_key=design.method_key,
        method_version=design.method_version,
        implementation_sha256=design.implementation_sha256,
        projection=projection_identity,
        projection_extension=extension_identity,
        horizon=design.horizon,
        prepared_denominator_id=design.prepared_denominator_id,
        chart_id=design.chart_id,
        retained_history_ids=design.retained_history_ids,
        action_words=design.action_words,
        action_role_bindings=design.action_role_bindings,
        axis_map=design.axis_map,
        unit_tier_bindings=design.unit_tier_bindings,
        prefixes=design.prefixes,
        evaluation_split_ids=design.evaluation_split_ids,
        response_quantity=design.response_quantity,
        minimum_complete_units_per_tier=design.minimum_complete_units_per_tier,
        minimum_positive_units_per_tier=design.minimum_positive_units_per_tier,
        materiality=design.materiality,
        family_alpha=design.family_alpha,
        simultaneous_test_count=design.simultaneous_test_count,
        multiplicity_family_id=design.multiplicity_family_id,
        information_cutoff_id=design.information_cutoff_id,
    )
    receipt = ConfirmatoryFiniteActionBindingReceipt(
        receipt_id=f"receipt.{config.config_id}",
        design=ObjectIdentity.from_record(design.design_id, design),
        manifest=manifest_identity,
        projection_plan=plan_identity,
        projection=projection_identity,
        projection_extension=extension_identity,
        scaffold=scaffold_identity,
        derived_config=ObjectIdentity.from_record(config.config_id, config),
        binding_implementation_id=binding_implementation_id,
        binding_implementation_sha256=binding_implementation_sha256,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return config, receipt


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/confirmatory-finite-action-result'

    result_id: str
    config: ObjectIdentity
    projection: ObjectIdentity
    projection_extension: ObjectIdentity
    cells: tuple[FiniteActionProductionCell, ...]
    payload: FiniteActionCompatibilitySetExtension
    candidate_evidence: LawCandidateEvidence

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.config.object_schema != ConfirmatoryFiniteActionConfig.SCHEMA:
            raise ValueError("confirmatory result binds another config")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        expected = {
            (
                value.denominator_member_id,
                value.candidate_version_id,
                value.support_cell_id,
                value.action_word.object_id,
            )
            for value in self.cells
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
        if not self.cells or expected != actual:
            raise ValueError("confirmatory result and payload cell rosters differ")


def _mean(values: tuple[Decimal, ...]) -> Decimal:
    return sum(values, Decimal(0)) / Decimal(len(values))


def _median(values: tuple[Decimal, ...]) -> Decimal:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def _student_interval(
    values: tuple[Decimal, ...], critical: Decimal
) -> tuple[Decimal, Decimal, Decimal]:
    mean = _mean(values)
    if len(values) < 2:
        raise ValueError("confirmatory Student interval requires at least two units")
    variance = sum(((value - mean) ** 2 for value in values), Decimal(0)) / Decimal(len(values) - 1)
    standard_error = (variance / Decimal(len(values))).sqrt()
    return mean, mean - critical * standard_error, mean + critical * standard_error


def _value(
    *, prefix: str, quantity: ConfirmatoryResponseQuantity, value: Decimal
) -> FiniteActionNativeValue:
    return FiniteActionNativeValue(
        value_id=f"{prefix}.{quantity.quantity_id}",
        quantity_id=quantity.quantity_id,
        value=value,
        native_unit=quantity.native_unit,
        native_frame_id=quantity.native_frame_id,
        clock_id=quantity.response_clock_id,
    )


def _receipt(
    *,
    candidate_id: str,
    suffix: str,
    observed: bool | None,
    evidence_link_ids: tuple[str, ...],
) -> CandidateMethodEvidenceReceipt:
    availability = (
        MethodEvidenceAvailability.NOT_APPLICABLE
        if observed is None
        else MethodEvidenceAvailability.OBSERVED
    )
    return CandidateMethodEvidenceReceipt(
        receipt_id=f"method-evidence.{candidate_id}.{suffix}",
        evidence_kind_id=f"confirmatory-finite-action.{suffix}",
        metrics=(
            NamedDecimal(
                value_id=f"confirmatory-finite-action-{suffix.replace('.', '-')}-criterion",
                value=Decimal(int(observed)),
                unit="1",
            ),
        )
        if observed is not None
        else (),
        artifact_ids=(),
        evidence_link_ids=evidence_link_ids,
        method_reason_codes=(),
        availability=availability,
    )


@dataclass(frozen=True, slots=True)
class _ObservedWord:
    observation: ProjectedIdentificationObservation
    response: Decimal
    precutoff: Decimal
    onset_clock: Decimal


@dataclass(frozen=True, slots=True)
class _ValidatedConfirmatoryOperands:
    """Pure authenticated operands shared by prefix and regional calculations."""

    roles: dict[str, ConfirmatoryActionRole]
    words: dict[str, OccurrenceActionWord]
    hold_id: str
    unit_bindings: dict[str, ConfirmatoryUnitTierBinding]
    values: dict[tuple[str, str, str, str], _ObservedWord]
    invalid: frozenset[tuple[str, str, str, str]]


def _validated_confirmatory_operands(
    *,
    projection: IdentificationEvidenceProjection,
    extension: IdentificationEvidenceProjectionExtension,
    config: ConfirmatoryFiniteActionConfig,
    scaffold: FiniteActionCandidateScaffold,
) -> _ValidatedConfirmatoryOperands:
    """Validate source lineage once without publishing a payload or candidate."""

    if not isinstance(config.axis_map, LawCandidateAxisMap):  # pragma: no cover
        raise TypeError("confirmatory config lost its axis map")
    if config.projection != ObjectIdentity.from_record(projection.projection_id, projection):
        raise ValueError("confirmatory config binds another projection")
    if projection.allowed_consumer_id != config.method_key:
        raise ValueError("confirmatory projection is authorized for another consumer")
    if config.projection_extension != ObjectIdentity.from_record(extension.extension_id, extension):
        raise ValueError("confirmatory config binds another projection extension")
    if extension.projection != config.projection or projection.information_cutoff_id != (
        config.information_cutoff_id
    ):
        raise ValueError("confirmatory projection lineage or causal cutoff differs")
    require_method_projection_truth(
        extension=extension,
        method_kind=IdentificationMethodEvidenceKind.FINITE_ACTION,
    )
    if config.physical_independent_unit_ids != extension.independent_unit_ids:
        raise ValueError("confirmatory config changes the independent-unit roster")
    if scaffold.axis_map != config.axis_map:
        raise ValueError("confirmatory scaffold axis map differs")
    if scaffold.obligation_template.physical_unit_count != len(
        config.physical_independent_unit_ids
    ):
        raise ValueError("confirmatory obligation scope changes physical replication")
    if scaffold.claim_unit_binding != ObjectIdentity.from_record(
        projection.claim_unit_binding.binding_id,
        projection.claim_unit_binding,
    ):
        raise ValueError("confirmatory scaffold changes the claim-unit binding")

    roles = {value.action_word_id: value.role for value in config.action_role_bindings}
    words = {value.word_id: value for value in config.action_words}
    word_identities = {
        value.word_id: ObjectIdentity.from_record(value.word_id, value)
        for value in config.action_words
    }
    hold_id = next(
        word_id for word_id, role in roles.items() if role is ConfirmatoryActionRole.MEASURED_HOLD
    )
    unit_bindings = {
        value.physical_independent_unit_id: value for value in config.unit_tier_bindings
    }
    observations_by_id = {
        value.projected_observation_id: value for value in projection.observations
    }
    terminal_by_observation = {
        value.projected_observation_id: value for value in extension.terminal_dispositions
    }
    terminals_by_episode = {value.episode_id: value for value in extension.terminal_dispositions}
    if (
        len(terminal_by_observation) != len(extension.terminal_dispositions)
        or len(terminals_by_episode) != len(extension.terminal_dispositions)
        or set(terminal_by_observation) != set(observations_by_id)
    ):
        raise ValueError("confirmatory terminal roster differs from the projection")
    occurrence_bindings: dict[str, list[ActionOccurrenceBinding]] = {}
    for binding in extension.action_occurrences:
        terminal = terminals_by_episode.get(binding.episode_id)
        observation = observations_by_id.get(binding.projected_observation_id)
        if (
            terminal is None
            or observation is None
            or terminal.projected_observation_id != binding.projected_observation_id
            or observation.action_word != binding.action_word
            or word_identities.get(binding.action_word.object_id) != binding.action_word
        ):
            raise ValueError("confirmatory occurrence changes episode/action identity")
        occurrence_bindings.setdefault(binding.episode_id, []).append(binding)
    for terminal in extension.terminal_dispositions:
        observation = observations_by_id[terminal.projected_observation_id]
        if terminal.physical_independent_unit_id != observation.physical_unit_instance_id:
            raise ValueError("confirmatory terminal changes the physical unit")
        if observation.action_delivery is None:
            raise ValueError("confirmatory observation lacks exact action delivery")
        word = words.get(observation.action_delivery.action_word.object_id)
        if word is None or observation.action_delivery.action_word != word_identities[word.word_id]:
            raise ValueError("confirmatory observation changes exact action word")
        bindings = occurrence_bindings.get(terminal.episode_id, [])
        binding_word_ids = {value.action_word for value in bindings}
        by_occurrence = {value.occurrence.occurrence_id: value.occurrence for value in bindings}
        expected_occurrence_ids = tuple(value.occurrence_id for value in word.occurrences)
        delivery = observation.action_delivery
        if (
            binding_word_ids != {word_identities[word.word_id]}
            or len(by_occurrence) != len(bindings)
            or tuple(by_occurrence.get(value) for value in expected_occurrence_ids)
            != word.occurrences
            or set(delivery.occurrence_ids) != set(expected_occurrence_ids)
            or any(
                len(stage_ids) != len(expected_occurrence_ids)
                for stage_ids in (
                    delivery.requested_event_ids,
                    delivery.accepted_event_ids,
                    delivery.applied_event_ids,
                    delivery.realized_event_ids,
                )
            )
        ):
            raise ValueError(
                "confirmatory episode changes exact occurrence order/content or delivery"
            )
    effects_by_member_view: dict[tuple[str, str], set[ComputabilityEffectStatus]] = {}
    for effect in extension.computability_entries:
        effects_by_member_view.setdefault(
            (
                effect.coordinate.denominator_member_id,
                effect.coordinate.numerical_view_id,
            ),
            set(),
        ).add(effect.status)
    expected_axes = {
        (value.denominator_member_id, value.candidate_version_member_id): value
        for value in config.axis_map.bindings
    }
    values: dict[tuple[str, str, str, str], _ObservedWord] = {}
    invalid: set[tuple[str, str, str, str]] = set()
    if {value.split_id for value in projection.observations} != set(config.evaluation_split_ids):
        raise ValueError("confirmatory projection crosses its fresh evaluation split")
    for observation in projection.observations:
        unit = unit_bindings.get(observation.physical_unit_instance_id)
        if unit is None:
            raise ValueError("confirmatory projection contains a foreign evaluation unit")
        axis = expected_axes.get((observation.member_id, observation.candidate_version_id))
        if axis is None or observation.qualification_view_id not in axis.qualification_view_ids:
            raise ValueError("confirmatory projection contains a foreign member/view axis")
        if observation.denominator_cell_id != unit.denominator_cell_id:
            raise ValueError("confirmatory projection changes a unit denominator cell")
        if (
            observation.chart_id != config.chart_id
            or observation.receiver_id != config.response_quantity.quantity_id
            or observation.receiver_clock_id != config.response_quantity.response_clock_id
            or observation.native_frame_id != config.response_quantity.native_frame_id
        ):
            raise ValueError("confirmatory projection changes chart/receiver/frame/clock")
        if observation.action_word is None or observation.action_word.object_id not in words:
            raise ValueError("confirmatory projection lacks an exact declared action word")
        word_id = observation.action_word.object_id
        if observation.action_word != ObjectIdentity.from_record(word_id, words[word_id]):
            raise ValueError("confirmatory projection changes action-word identity")
        coordinate = (
            observation.member_id,
            observation.candidate_version_id,
            observation.physical_unit_instance_id,
            word_id,
        )
        if coordinate in values or coordinate in invalid:
            raise ValueError("confirmatory projection duplicates a unit/member/word cell")
        terminal = terminal_by_observation.get(observation.projected_observation_id)
        effect_statuses = effects_by_member_view.get(
            (observation.member_id, observation.qualification_view_id),
            set(),
        )
        valid = (
            observation.disposition is ObservationDisposition.COMPLETE
            and terminal is not None
            and terminal.physical_sink is PhysicalSinkAxis.NONE
            and terminal.observation_validity is ObservationValidityAxis.VALID
            and terminal.numerical_validity is NumericalValidityAxis.VALID
            and terminal.authority in {AuthorityAxis.NOT_REQUIRED, AuthorityAxis.AUTHORIZED}
            and terminal.scientific_status
            in {ScientificTerminalAxis.OBSERVED, ScientificTerminalAxis.NEGATIVE}
            and bool(effect_statuses)
            and effect_statuses
            <= {
                ComputabilityEffectStatus.REPRESENTED,
                ComputabilityEffectStatus.ASSUMPTION_CLOSED,
            }
        )
        if not valid:
            invalid.add(coordinate)
            continue
        compact = {value.value_id: value for value in observation.compact_values}
        quantity = config.response_quantity
        try:
            response = compact[quantity.response_projection_value_id]
            precutoff = compact[quantity.precutoff_projection_value_id]
            onset = compact[quantity.onset_clock_projection_value_id]
        except KeyError as error:
            raise ValueError("confirmatory projection lacks a required response field") from error
        if response.unit != quantity.native_unit or precutoff.unit != quantity.native_unit:
            raise ValueError("confirmatory response/precutoff native unit drifted")
        if onset.unit != "1":
            raise ValueError("confirmatory onset clock must be a dimensionless native index")
        values[coordinate] = _ObservedWord(
            observation,
            response.value,
            precutoff.value,
            onset.value,
        )
    return _ValidatedConfirmatoryOperands(
        roles=roles,
        words=words,
        hold_id=hold_id,
        unit_bindings=unit_bindings,
        values=values,
        invalid=frozenset(invalid),
    )


@dataclass(frozen=True, slots=True)
class ConfirmatoryFiniteActionProducer:
    payload_publisher: CandidatePayloadPublisher

    def identify(
        self,
        *,
        projection: IdentificationEvidenceProjection,
        extension: IdentificationEvidenceProjectionExtension,
        config: ConfirmatoryFiniteActionConfig,
        scaffold: FiniteActionCandidateScaffold,
    ) -> ConfirmatoryFiniteActionResult:
        operands = _validated_confirmatory_operands(
            projection=projection,
            extension=extension,
            config=config,
            scaffold=scaffold,
        )
        roles = operands.roles
        words = operands.words
        hold_id = operands.hold_id
        unit_bindings = operands.unit_bindings
        values = operands.values
        invalid = operands.invalid

        prefixes = tuple(sorted(config.prefixes, key=lambda value: value.order_index))
        evidence_link_ids = tuple(value.link_id for value in scaffold.evidence_links)
        cells: list[FiniteActionProductionCell] = []
        entries: list[FiniteActionResponseEntry] = []
        active_prefix_pass: dict[tuple[str, str, str], bool] = {}
        estimates_by_role_member_prefix: dict[
            tuple[ConfirmatoryActionRole, str, str, str], Decimal
        ] = {}
        support_by_role_prefix: dict[tuple[ConfirmatoryActionRole, str], set[bool]] = {}
        all_complete = True
        all_exchange = True
        all_causal = True
        all_history = True
        for axis in config.axis_map.bindings:
            member = axis.denominator_member_id
            candidate_version = axis.candidate_version_member_id
            for prefix in prefixes:
                prefix_units = tuple(
                    sorted(
                        value.physical_independent_unit_id
                        for value in config.unit_tier_bindings
                        if value.tier_id in prefix.constituent_tier_ids
                    )
                )
                for word in config.action_words:
                    role = roles[word.word_id]
                    paired: dict[str, tuple[Decimal, Decimal, Decimal]] = {}
                    excluded: list[str] = []
                    for unit_id in prefix_units:
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
                    response_values: tuple[Decimal, ...] = tuple(
                        paired[value][0] for value in complete_units
                    )
                    reasons: list[str] = []
                    if word.support_status is not ActionWordSupportStatus.SUPPORTED:
                        disposition = FiniteActionProductionDisposition.OUTSIDE_SUPPORT
                        reasons.append("CONFIRMATORY_ACTION_WORD_OUTSIDE_SUPPORT")
                        mean = lower = upper = Decimal(0)
                    elif len(complete_units) != len(prefix_units):
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
                        mean = lower = upper = Decimal(0)
                        all_complete = False
                    else:
                        mean, lower, upper = _student_interval(
                            response_values,
                            prefix.one_sided_critical_value,
                        )
                        precutoff_ok = all(
                            abs(paired[unit_id][1])
                            <= config.response_quantity.maximum_precutoff_absolute_effect
                            for unit_id in complete_units
                        )
                        all_history = all_history and precutoff_ok
                        tier_effects = {
                            tier_id: tuple(
                                paired[unit_id][0]
                                for unit_id in complete_units
                                if unit_bindings[unit_id].tier_id == tier_id
                            )
                            for tier_id in prefix.constituent_tier_ids
                        }
                        if any(
                            len(tier_values) < config.minimum_complete_units_per_tier
                            for tier_values in tier_effects.values()
                        ):
                            reasons.append("CONFIRMATORY_TIER_COMPLETE_UNIT_MINIMUM_NOT_MET")
                        if not precutoff_ok:
                            reasons.append("CONFIRMATORY_PRECUTOFF_HISTORY_FALSIFIER_FAILED")
                        if role is ConfirmatoryActionRole.TARGET_ACTIVE:
                            causal_ok = all(
                                config.response_quantity.earliest_active_response_clock
                                <= paired[unit_id][2]
                                <= config.response_quantity.latest_active_response_clock
                                for unit_id in complete_units
                            )
                            tier_counts_ok = all(
                                sum(effect > 0 for effect in tier_values)
                                >= config.minimum_positive_units_per_tier
                                for tier_values in tier_effects.values()
                            )
                            tier_medians_ok = all(
                                _median(tier_values) >= config.materiality
                                for tier_values in tier_effects.values()
                            )
                            previous_ok = all(
                                active_prefix_pass.get(
                                    (member, candidate_version, prior.prefix_id), False
                                )
                                for prior in prefixes
                                if prior.order_index < prefix.order_index
                            )
                            if lower <= config.materiality:
                                reasons.append("CONFIRMATORY_ONE_SIDED_LOWER_NOT_MATERIAL")
                            if not tier_counts_ok:
                                reasons.append("CONFIRMATORY_TIER_POSITIVE_COUNT_FAILED")
                            if not tier_medians_ok:
                                reasons.append("CONFIRMATORY_TIER_MEDIAN_NOT_MATERIAL")
                            if not causal_ok:
                                reasons.append("CONFIRMATORY_ACTIVE_CAUSAL_CLOCK_FAILED")
                            if not previous_ok:
                                reasons.append("CONFIRMATORY_PREFIX_CLOSURE_FAILED")
                            active_prefix_pass[
                                (member, candidate_version, prefix.prefix_id)
                            ] = not reasons
                            all_causal = all_causal and causal_ok
                        elif role is ConfirmatoryActionRole.WRONG_SIGN_CONTROL:
                            causal_ok = all(
                                config.response_quantity.earliest_active_response_clock
                                <= paired[unit_id][2]
                                <= config.response_quantity.latest_active_response_clock
                                for unit_id in complete_units
                            )
                            if not causal_ok:
                                reasons.append("CONFIRMATORY_WRONG_SIGN_CAUSAL_CLOCK_FAILED")
                            all_causal = all_causal and causal_ok
                        elif role is ConfirmatoryActionRole.FUTURE_NULL_CONTROL:
                            null_ok = all(
                                abs(effect) <= config.response_quantity.maximum_null_absolute_effect
                                for effect in response_values
                            )
                            causal_ok = all(
                                paired[unit_id][2]
                                >= config.response_quantity.earliest_future_response_clock
                                for unit_id in complete_units
                            )
                            if not null_ok:
                                reasons.append("CONFIRMATORY_FUTURE_TARGET_WINDOW_NOT_NULL")
                            if not causal_ok:
                                reasons.append("CONFIRMATORY_FUTURE_CAUSAL_CLOCK_FAILED")
                            all_causal = all_causal and null_ok and causal_ok
                        # HOLD and wrong-sign controls are supported measured
                        # fibres when complete and history-valid.  Their signs
                        # are retained, never forced to fit the target.
                        disposition = (
                            FiniteActionProductionDisposition.SUPPORTED
                            if not reasons
                            else FiniteActionProductionDisposition.NOT_SUPPORTED
                        )
                    all_exchange = all_exchange and (
                        word.denominator_id == words[hold_id].denominator_id
                        and word.retained_history_id == words[hold_id].retained_history_id
                        and word.horizon_id == words[hold_id].horizon_id
                    )
                    suffix = (
                        f"{member}.{candidate_version}.{prefix.prefix_id}.{word.word_id}"
                    ).replace("_", "-")
                    estimates: tuple[FiniteActionNativeValue, ...] = ()
                    lowers: tuple[FiniteActionNativeValue, ...] = ()
                    uppers: tuple[FiniteActionNativeValue, ...] = ()
                    if disposition in {
                        FiniteActionProductionDisposition.SUPPORTED,
                        FiniteActionProductionDisposition.NOT_SUPPORTED,
                    }:
                        estimates = (
                            _value(
                                prefix="response", quantity=config.response_quantity, value=mean
                            ),
                        )
                        lowers = (
                            _value(prefix="lower", quantity=config.response_quantity, value=lower),
                        )
                        uppers = (
                            _value(prefix="upper", quantity=config.response_quantity, value=upper),
                        )
                        estimates_by_role_member_prefix[
                            (role, member, candidate_version, prefix.prefix_id)
                        ] = mean
                    cell = FiniteActionProductionCell(
                        cell_id=f"confirmatory-finite-action-cell.{suffix}",
                        denominator_member_id=member,
                        candidate_version_id=candidate_version,
                        support_cell_id=prefix.prefix_id,
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
                    )
                    cells.append(cell)
                    support_by_role_prefix.setdefault((role, prefix.prefix_id), set()).add(
                        disposition is FiniteActionProductionDisposition.SUPPORTED
                    )
                    payload_disposition = {
                        FiniteActionProductionDisposition.SUPPORTED: FiniteActionCellDisposition.SUPPORTED,
                        FiniteActionProductionDisposition.NOT_SUPPORTED: FiniteActionCellDisposition.NOT_SUPPORTED,
                        FiniteActionProductionDisposition.OUTSIDE_SUPPORT: FiniteActionCellDisposition.OUTSIDE_SUPPORT,
                        FiniteActionProductionDisposition.PARTIAL: FiniteActionCellDisposition.UNEVALUABLE,
                        FiniteActionProductionDisposition.INVALID: FiniteActionCellDisposition.UNEVALUABLE,
                        FiniteActionProductionDisposition.UNEVALUABLE: FiniteActionCellDisposition.UNEVALUABLE,
                    }[disposition]
                    entries.append(
                        FiniteActionResponseEntry(
                            entry_id=f"confirmatory-finite-action-entry.{suffix}",
                            denominator_member_id=member,
                            candidate_version_id=candidate_version,
                            qualification_view_ids=axis.qualification_view_ids,
                            support_cell_id=prefix.prefix_id,
                            action_word=ObjectIdentity.from_record(word.word_id, word),
                            comparator_action_word_ids=(hold_id,),
                            physical_independent_unit_ids=prefix_units,
                            response_values=cell.response_estimates,
                            sink_values=(),
                            effort_values=(),
                            uncertainty_values=tuple(
                                sorted(
                                    (*cell.lower_bounds, *cell.upper_bounds),
                                    key=lambda value: value.value_id,
                                )
                            ),
                            disposition=payload_disposition,
                            evidence_link_ids=evidence_link_ids,
                            reason_codes=cell.reason_codes,
                        )
                    )

        target_signs: dict[str, set[int]] = {}
        for (
            role,
            _member,
            _version,
            prefix_id,
        ), estimate in estimates_by_role_member_prefix.items():
            if role is ConfirmatoryActionRole.TARGET_ACTIVE:
                target_signs.setdefault(prefix_id, set()).add(
                    1 if estimate > 0 else -1 if estimate < 0 else 0
                )
        expected_prefix_ids = {prefix.prefix_id for prefix in prefixes}
        member_structural_agreement = (
            set(target_signs) == expected_prefix_ids
            and all(len(signs) == 1 and signs != {0} for signs in target_signs.values())
            and all(len(values_) == 1 for values_ in support_by_role_prefix.values())
        )

        payload = FiniteActionCompatibilitySetExtension(
            extension_id=f"confirmatory-finite-action-set.{config.config_id}",
            method_semantics=(
                "Fresh fixed-panel member-local finite-word response lookup with Bonferroni "
                "one-sided Student bounds, exact tier-prefix closure and role-specific causal "
                "controls; no linearity inference, coefficient pooling or magnitude-convergence "
                "claim."
            ),
            prepared_denominator_id=config.prepared_denominator_id,
            horizon=config.horizon,
            retained_history_ids=config.retained_history_ids,
            action_words=config.action_words,
            physical_independent_unit_ids=config.physical_independent_unit_ids,
            entries=tuple(sorted(entries, key=lambda value: value.entry_id)),
            rejection_entry_ids=tuple(
                sorted(
                    value.entry_id
                    for value in entries
                    if value.disposition is not FiniteActionCellDisposition.SUPPORTED
                )
            ),
            refusal_reason_codes=("confirmatory-finite-action-cell-unavailable",),
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
            reference_id=f"evaluator.{config.config_id}",
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
            implementation_id=f"implementation.{config.config_id}",
            capability_key=evaluator.capability_key,
            capability_version=evaluator.capability_version,
            evaluator_key=evaluator.evaluator_key,
            source_sha256=config.implementation_sha256,
        )
        publication = self.payload_publisher.publish_candidate_payload(
            payload=payload_bytes,
            evaluator=evaluator,
            implementation=ObjectIdentity.from_record(
                implementation.implementation_id, implementation
            ),
            decoder_schema='empirical-lawhood/methods/finite-action/compatibility-set-decoder',
            decoder_version="1.0.0",
            maximum_decode_bytes=4 * 1024 * 1024,
        )
        receipt_values = {
            "recurrence": all_complete,
            "declared-word-exchange": all_exchange,
            "causal-falsifiers": all_causal,
            "history-prefix-closure": all_history,
            "fresh-confirmatory-evaluation": all_complete,
            "member-structural-agreement": member_structural_agreement,
            "uncertainty.numerical": len(config.axis_map.bindings) > 1,
            "uncertainty.aleatoric": len(config.physical_independent_unit_ids) > 1,
            "uncertainty.epistemic": len(config.axis_map.bindings) > 1,
            "uncertainty.transport": None,
            "uncertainty.observation": all_complete,
        }
        receipts = tuple(
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
                # The consolidated decoder owns this payload schema under its
                # stable namespace; method/profile identity remains separate.
                namespace="finite-action-compatibility-set",
                schema=payload.SCHEMA,
                payload_sha256=publication.content_sha256,
            ),
        )
        return ConfirmatoryFiniteActionResult(
            result_id=f"confirmatory-finite-action-result.{config.fingerprint()[:32]}",
            config=ObjectIdentity.from_record(config.config_id, config),
            projection=config.projection,
            projection_extension=config.projection_extension,
            cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
            payload=payload,
            candidate_evidence=candidate,
        )


__all__ = [
    'CONFIRMATORY_DESIGN_CONFIG_ID_RULE',
    "CONFIRMATORY_FINITE_ACTION_KEY",
    "CONFIRMATORY_FINITE_ACTION_VERSION",
    'CONFIRMATORY_REGION_SUPPORT_SPEC_SCHEMA',
    'ConditionalConfirmatoryFiniteActionCoreTemplate',
    'ConditionalConfirmatoryUnitTierSlot',
    "ConfirmatoryActionRole",
    'ConfirmatoryActionRoleBinding',
    'ConfirmatoryFiniteActionConfig',
    'ConfirmatoryFiniteActionBindingReceipt',
    'ConfirmatoryFiniteActionDesign',
    'ConfirmatoryFiniteActionDesignTemplateBindingReceipt',
    'ConfirmatoryFiniteActionDesignTemplate',
    'ConfirmatoryFiniteActionProducer',
    'ConfirmatoryFiniteActionProspectiveCore',
    'ConfirmatoryFiniteActionResult',
    'ConfirmatoryPrefixSpec',
    'ConfirmatoryResponseQuantity',
    'ConfirmatoryUnitTierBinding',
    'bind_confirmatory_finite_action_design',
    'bind_confirmatory_finite_action_design_template',
    'materialize_conditional_confirmatory_finite_action_core',
    'materialize_conditional_confirmatory_finite_action_design_template',
]
