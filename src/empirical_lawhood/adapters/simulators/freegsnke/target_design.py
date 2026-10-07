"""Outcome-blind independent substrate grounding target-design freezes for FreeGSNKE.

These records instantiate the common outcome-blind scientific grammar grammar with source-native names,
units, finite alternatives and complete-preparation power semantics.  They do
not choose a mapping or denominator from development data and they do not
decode simulator outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateDenominatorCandidate, IndependentSubstrateForecastAlphabetGrammar, IndependentSubstrateTargetMappingCandidate, IndependentSubstrateTargetKind
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)

from .contracts import (
    FREEGSNKE_ACTIVE_CIRCUIT_IDS,
    FREEGSNKE_PASSIVE_CURRENT_SUMMARY_IDS,
    FREEGSNKE_RECEIVER_UNITS,
)
from .design import FreeGsnkeActionDesign, FreeGsnkePhaseRequestRoster


FREEGSNKE_FORECAST_STATE_IDS = (
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
FREEGSNKE_REQUIRED_FORECAST_ROLE_IDS = (
    "action",
    "admission",
    "denominator",
    "failure",
    "history",
    "hold",
    "sink",
    "support",
)
FREEGSNKE_LAW_OPERAND_IDS = ("A", "D", "H", "R", "tau")
FREEGSNKE_POWER_RECEIVER_IDS = tuple(
    sorted(
        (
            *FREEGSNKE_RECEIVER_UNITS,
            *(f"current-{value}" for value in FREEGSNKE_ACTIVE_CIRCUIT_IDS),
            "current-ip",
            *FREEGSNKE_PASSIVE_CURRENT_SUMMARY_IDS,
        )
    )
)


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetForecast(CanonicalRecord):
    """One closed-alphabet forecast over an exact native target subject."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-forecast'

    forecast_id: str
    role_id: str
    subject_native_ids: tuple[str, ...]
    alphabet_state_ids: tuple[str, ...]
    unsafe_state_ids: tuple[str, ...]
    evaluation_unit_denominator: str
    claim_bearing: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.forecast_id, field_name="forecast_id")
        validate_stable_id(self.role_id, field_name="role_id")
        require_sorted_unique_strings(
            self.subject_native_ids,
            field_name="subject_native_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.alphabet_state_ids,
            field_name="alphabet_state_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.unsafe_state_ids,
            field_name="unsafe_state_ids",
        )
        if not set(self.alphabet_state_ids).issubset(FREEGSNKE_FORECAST_STATE_IDS):
            raise ValueError("FreeGSNKE forecast contains a state outside the outcome-blind scientific grammar alphabet")
        if not set(self.unsafe_state_ids).issubset(self.alphabet_state_ids):
            raise ValueError("FreeGSNKE unsafe states must belong to the forecast alphabet")
        if self.evaluation_unit_denominator != ("ALL_ISSUED_COMPLETE_EVALUATION_PREPARATIONS"):
            raise ValueError("FreeGSNKE forecast denominator differs")


@dataclass(frozen=True, slots=True)
class FreeGsnkeForecastAlphabetFreeze(CanonicalRecord):
    """Exact G2 forecast roster frozen before any development outcome."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-forecast-alphabet-freeze'

    freeze_id: str
    common_grammar: ObjectIdentity
    target_encoder: ObjectIdentity
    source_binding: ObjectIdentity
    action_design: ObjectIdentity
    forecasts: tuple[FreeGsnkeTargetForecast, ...]
    postissue_outcome_exclusion_allowed: bool
    frozen_before_development: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        if self.common_grammar.object_schema != IndependentSubstrateForecastAlphabetGrammar.SCHEMA:
            raise ValueError("FreeGSNKE forecast grammar identity differs")
        if self.target_encoder.object_schema != (
            'empirical-lawhood/simulators/freegsnke/free-gsnke-forecast-encoder-freeze'
        ):
            raise ValueError("FreeGSNKE forecast encoder identity differs")
        if self.action_design.object_schema != FreeGsnkeActionDesign.SCHEMA:
            raise ValueError("FreeGSNKE forecast action-design identity differs")
        require_sorted_unique_ids(
            self.forecasts,
            attribute="forecast_id",
            field_name="forecasts",
        )
        role_ids = {value.role_id for value in self.forecasts if value.claim_bearing}
        if not set(FREEGSNKE_REQUIRED_FORECAST_ROLE_IDS).issubset(role_ids):
            raise ValueError("FreeGSNKE claim-bearing forecast roster is incomplete")
        if self.postissue_outcome_exclusion_allowed:
            raise ValueError("FreeGSNKE cannot exclude issued units after outcomes")
        if not self.frozen_before_development or self.protected_outcome_access_count:
            raise ValueError("FreeGSNKE forecast alphabet crossed development access")


@dataclass(frozen=True, slots=True)
class FreeGsnkeMetricTopologyLevelBinding(CanonicalRecord):
    """One outcome-blind native selector in a within-system exchange."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-metric-topology-level-binding'

    binding_id: str
    level_id: str
    denominator_stratum_id: str | None
    history_stratum_id: str | None
    action_branch_id: str
    receiver_id: str
    horizon_s: Decimal

    def __post_init__(self) -> None:
        for name in (
            "binding_id",
            "level_id",
            "action_branch_id",
            "receiver_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("denominator_stratum_id", "history_stratum_id"):
            value = getattr(self, name)
            if value is not None:
                validate_stable_id(value, field_name=name)
        validate_decimal(self.horizon_s, field_name="horizon_s", minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class FreeGsnkeMetricTopologyExchangeDesign(CanonicalRecord):
    """One prospectively frozen topology/native-metric paired comparison."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-metric-topology-exchange-design'

    exchange_id: str
    law_operand_id: str
    contrast_level_ids: tuple[str, ...]
    topology_endpoint_id: str
    metric_coordinate_id: str
    metric_native_unit: str
    metric_tolerance: Decimal
    topology_deadband: Decimal
    response_direction: Decimal
    level_bindings: tuple[FreeGsnkeMetricTopologyLevelBinding, ...]
    action_branch_ids: tuple[str, ...]
    receiver_ids: tuple[str, ...]
    horizons_s: tuple[Decimal, ...]
    physical_response_margin_rule: str
    certification_margin_rule: str
    measurement_backaction_rule: str
    topology_classification_rule: str
    metric_reference_rule: str
    complete_unit_success_rule: str
    missing_observation_rule: str
    claim_bearing: bool

    def __post_init__(self) -> None:
        for name in (
            "exchange_id",
            "topology_endpoint_id",
            "metric_coordinate_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.law_operand_id not in FREEGSNKE_LAW_OPERAND_IDS:
            raise ValueError("FreeGSNKE exchange names an unknown law operand")
        require_sorted_unique_strings(
            self.contrast_level_ids,
            field_name="contrast_level_ids",
            allow_empty=False,
        )
        if len(self.contrast_level_ids) < 2:
            raise ValueError("FreeGSNKE exchange requires at least two contrast levels")
        require_sorted_unique_ids(
            self.level_bindings,
            attribute="binding_id",
            field_name="level_bindings",
        )
        if (
            tuple(sorted(value.level_id for value in self.level_bindings))
            != self.contrast_level_ids
        ):
            raise ValueError("FreeGSNKE exchange level bindings differ from its roster")
        require_sorted_unique_strings(
            self.action_branch_ids,
            field_name="action_branch_ids",
        )
        require_sorted_unique_strings(
            self.receiver_ids,
            field_name="receiver_ids",
        )
        for value in self.horizons_s:
            validate_decimal(value, field_name="horizons_s", minimum=Decimal(0))
        if tuple(sorted(set(self.horizons_s))) != self.horizons_s:
            raise ValueError("FreeGSNKE exchange horizons must be sorted and unique")
        validate_nonempty(self.metric_native_unit, field_name="metric_native_unit")
        validate_decimal(
            self.metric_tolerance,
            field_name="metric_tolerance",
            minimum=Decimal(0),
        )
        if self.metric_tolerance == 0:
            raise ValueError("FreeGSNKE metric tolerance must be positive")
        validate_decimal(
            self.topology_deadband,
            field_name="topology_deadband",
            minimum=Decimal(0),
        )
        validate_decimal(self.response_direction, field_name="response_direction")
        if self.topology_deadband == 0 or self.response_direction not in {
            Decimal(-1),
            Decimal(1),
        }:
            raise ValueError("FreeGSNKE topology boundary/direction differs")
        for name in (
            "physical_response_margin_rule",
            "certification_margin_rule",
            "measurement_backaction_rule",
            "topology_classification_rule",
            "metric_reference_rule",
            "complete_unit_success_rule",
            "missing_observation_rule",
        ):
            validate_nonempty(getattr(self, name), field_name=name)
        if self.physical_response_margin_rule == self.certification_margin_rule:
            raise ValueError("FreeGSNKE physical and certification margins are conflated")
        if self.law_operand_id == "A" and len(self.action_branch_ids) < 2:
            raise ValueError("FreeGSNKE action exchange requires two native branches")
        if self.law_operand_id == "R" and len(self.receiver_ids) < 2:
            raise ValueError("FreeGSNKE receiver exchange requires two receiver maps")
        if self.law_operand_id == "tau" and len(self.horizons_s) < 2:
            raise ValueError("FreeGSNKE horizon exchange requires two horizons")
        if (
            self.topology_classification_rule != "SIGNED_THREE_BAND_AROUND_ZERO"
            or self.metric_reference_rule != "DEVELOPMENT_COMPLETE_UNIT_CELL_MEDIAN"
            or self.complete_unit_success_rule != "ALL_APPLICABLE_LEVELS"
            or self.missing_observation_rule != "FAIL_BOTH_RETAIN_ISSUED_UNIT"
        ):
            raise ValueError("FreeGSNKE metric/topology construction rule differs")
        bindings = self.level_bindings
        denominator_ids = tuple(value.denominator_stratum_id for value in bindings)
        history_ids = tuple(value.history_stratum_id for value in bindings)
        branch_ids = tuple(value.action_branch_id for value in bindings)
        receiver_ids = tuple(value.receiver_id for value in bindings)
        horizons = tuple(value.horizon_s for value in bindings)
        if self.law_operand_id == "D":
            valid_exchange = (
                all(value is not None for value in denominator_ids)
                and len(set(denominator_ids)) == len(bindings)
                and set(history_ids) == {None}
                and len(set(branch_ids)) == len(set(receiver_ids)) == len(set(horizons)) == 1
            )
        elif self.law_operand_id == "H":
            valid_exchange = (
                all(value is not None for value in history_ids)
                and len(set(history_ids)) == len(bindings)
                and set(denominator_ids) == {None}
                and len(set(branch_ids)) == len(set(receiver_ids)) == len(set(horizons)) == 1
            )
        elif self.law_operand_id == "A":
            valid_exchange = (
                set(denominator_ids) == set(history_ids) == {None}
                and len(set(branch_ids)) == len(bindings)
                and tuple(sorted(set(branch_ids))) == self.action_branch_ids
                and len(set(receiver_ids)) == len(set(horizons)) == 1
            )
        elif self.law_operand_id == "R":
            valid_exchange = (
                set(denominator_ids) == set(history_ids) == {None}
                and len(set(receiver_ids)) == len(bindings)
                and tuple(sorted(set(receiver_ids))) == self.receiver_ids
                and len(set(branch_ids)) == len(set(horizons)) == 1
            )
        else:
            valid_exchange = (
                set(denominator_ids) == set(history_ids) == {None}
                and len(set(horizons)) == len(bindings)
                and tuple(sorted(set(horizons))) == self.horizons_s
                and len(set(branch_ids)) == len(set(receiver_ids)) == 1
            )
        if not valid_exchange:
            raise ValueError("FreeGSNKE exchange does not vary exactly its named law operand")


@dataclass(frozen=True, slots=True)
class FreeGsnkeMetricTopologyDesign(CanonicalRecord):
    """Complete target-local exchange roster, including explicit claim narrowing."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-metric-topology-design'

    design_id: str
    source_binding: ObjectIdentity
    action_design: ObjectIdentity
    numerical_view: ObjectIdentity
    exchanges: tuple[FreeGsnkeMetricTopologyExchangeDesign, ...]
    claimed_law_operand_ids: tuple[str, ...]
    unsupported_law_operand_ids: tuple[str, ...]
    cross_target_native_metric_pooling_allowed: bool
    frozen_before_development: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        if self.action_design.object_schema != FreeGsnkeActionDesign.SCHEMA:
            raise ValueError("FreeGSNKE exchange action design differs")
        if self.numerical_view.object_schema != (
            'empirical-lawhood/simulators/freegsnke/free-gsnke-numerical-view'
        ):
            raise ValueError("FreeGSNKE exchange numerical view differs")
        require_sorted_unique_ids(
            self.exchanges,
            attribute="exchange_id",
            field_name="exchanges",
        )
        for name in ("claimed_law_operand_ids", "unsupported_law_operand_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if set(self.claimed_law_operand_ids) & set(self.unsupported_law_operand_ids) or set(
            self.claimed_law_operand_ids
        ) | set(self.unsupported_law_operand_ids) != set(FREEGSNKE_LAW_OPERAND_IDS):
            raise ValueError("FreeGSNKE claimed/unsupported operands do not partition the law")
        claim_bearing_operands = {
            value.law_operand_id for value in self.exchanges if value.claim_bearing
        }
        if claim_bearing_operands != set(self.claimed_law_operand_ids):
            raise ValueError("FreeGSNKE exchange roster and claimed operands differ")
        if self.cross_target_native_metric_pooling_allowed:
            raise ValueError("FreeGSNKE native metrics cannot be pooled across targets")
        if not self.frozen_before_development or self.protected_outcome_access_count:
            raise ValueError("FreeGSNKE exchange design crossed development access")


@dataclass(frozen=True, slots=True)
class FreeGsnkeJointPowerAnalysis(CanonicalRecord):
    "Compact qualification analysis with separated power axes."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-joint-power-analysis'

    analysis_id: str
    excluded_qualification_evidence: ObjectIdentity
    method_implementation: ObjectIdentity
    candidate_panel_sizes: tuple[int, ...]
    candidate_numerical_views: tuple[ObjectIdentity, ...]
    selected_numerical_view: ObjectIdentity
    receiver_coordinate_ids: tuple[str, ...]
    observation_grid_coordinate_count: int
    separated_axis_ids: tuple[str, ...]
    transition_width_upper: Decimal
    grid_discretization_error_upper: Decimal
    receiver_projection_error_upper: Decimal
    complete_unit_standard_error_upper: Decimal
    simultaneous_half_width_upper: Decimal
    minimum_material_response: Decimal
    minimum_certification_margin: Decimal
    familywise_alpha: Decimal
    power_floor: Decimal
    worst_case_power: Decimal
    selected_development_units: int
    selected_evaluation_units: int
    selected_prospective_evaluation_units: int
    complete_unit_bootstrap_replications: int
    qualification_units_disjoint_from_target_rosters: bool
    qualification_outcomes_imported_into_target: bool
    complete_unit_resampling: bool
    nested_observations_count_as_units: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        if (
            not self.candidate_panel_sizes
            or tuple(sorted(set(self.candidate_panel_sizes))) != self.candidate_panel_sizes
            or any(value < 2 for value in self.candidate_panel_sizes)
        ):
            raise ValueError("FreeGSNKE power candidate panel sizes differ")
        require_sorted_unique_ids(
            self.candidate_numerical_views,
            attribute="object_id",
            field_name="candidate_numerical_views",
        )
        if self.selected_numerical_view not in self.candidate_numerical_views:
            raise ValueError("FreeGSNKE power selected an unqualified numerical view")
        require_sorted_unique_strings(
            self.receiver_coordinate_ids,
            field_name="receiver_coordinate_ids",
            allow_empty=False,
        )
        if self.receiver_coordinate_ids != FREEGSNKE_POWER_RECEIVER_IDS:
            raise ValueError("FreeGSNKE joint power receiver roster differs")
        if self.observation_grid_coordinate_count < len(self.receiver_coordinate_ids):
            raise ValueError("FreeGSNKE power observation grid is incomplete")
        if self.separated_axis_ids != (
            "grid-resolution",
            "panel-size",
            "receiver-coordinate",
            "simultaneous-inference",
            "transition-width",
        ):
            raise ValueError("FreeGSNKE qualification power axes are not separated")
        for name in (
            "transition_width_upper",
            "grid_discretization_error_upper",
            "receiver_projection_error_upper",
            "complete_unit_standard_error_upper",
            "simultaneous_half_width_upper",
            "minimum_material_response",
            "minimum_certification_margin",
            "familywise_alpha",
            "power_floor",
            "worst_case_power",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.simultaneous_half_width_upper < self.complete_unit_standard_error_upper:
            raise ValueError("FreeGSNKE simultaneous width cannot understate unit uncertainty")
        if self.minimum_material_response == 0 or self.minimum_certification_margin == 0:
            raise ValueError("FreeGSNKE power margins must be positive")
        for name in ("familywise_alpha", "power_floor", "worst_case_power"):
            if not Decimal(0) < getattr(self, name) < Decimal(1):
                raise ValueError(f"FreeGSNKE {name} must be strictly interior")
        selected_sizes = (
            self.selected_development_units,
            self.selected_evaluation_units,
            self.selected_prospective_evaluation_units,
        )
        if any(value not in self.candidate_panel_sizes for value in selected_sizes):
            raise ValueError("FreeGSNKE selected panel size was not prospectively tested")
        if self.complete_unit_bootstrap_replications < 2048:
            raise ValueError("FreeGSNKE joint power bootstrap is underspecified")
        if not all(
            (
                self.qualification_units_disjoint_from_target_rosters,
                not self.qualification_outcomes_imported_into_target,
                self.complete_unit_resampling,
                not self.nested_observations_count_as_units,
            )
        ):
            raise ValueError("FreeGSNKE power analysis violates qualification/unit isolation")

    @property
    def physical_response_margin(self) -> Decimal:
        return self.minimum_material_response - (
            self.transition_width_upper
            + self.grid_discretization_error_upper
            + self.receiver_projection_error_upper
        )

    @property
    def certification_margin(self) -> Decimal:
        return self.minimum_certification_margin - self.simultaneous_half_width_upper

    @property
    def joint_power_passed(self) -> bool:
        return all(
            (
                self.physical_response_margin > 0,
                self.certification_margin >= 0,
                self.worst_case_power >= self.power_floor,
            )
        )


@dataclass(frozen=True, slots=True)
class FreeGsnkePowerFreeze(CanonicalRecord):
    """Joint panel/grid/receiver-coordinate power and roster freeze."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-power-freeze'

    freeze_id: str
    source_binding: ObjectIdentity
    action_design: ObjectIdentity
    development_roster: ObjectIdentity
    evaluation_roster: ObjectIdentity
    selected_numerical_view: ObjectIdentity
    qualified_numerical_views: tuple[ObjectIdentity, ...]
    joint_power_analysis: FreeGsnkeJointPowerAnalysis
    development_unit_ids: tuple[str, ...]
    development_seeds: tuple[int, ...]
    evaluation_unit_ids: tuple[str, ...]
    evaluation_seeds: tuple[int, ...]
    prospective_validation_unit_ids: tuple[str, ...]
    prospective_validation_seeds: tuple[int, ...]
    required_horizons_s: tuple[Decimal, ...]
    required_receiver_ids: tuple[str, ...]
    development_minimum_complete_units: int
    evaluation_minimum_complete_units: int
    prospective_validation_minimum_complete_units: int
    simultaneous_family_ids: tuple[str, ...]
    familywise_alpha: Decimal
    complete_unit_bootstrap_replications: int
    joint_panel_grid_receiver_power_passed: bool
    panel_envelope_limited: bool
    complete_unit_role: str
    resampling_unit_role: str
    nested_observations_count_as_units: bool
    physical_response_and_certification_margins_distinct: bool
    frozen_before_development: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        if self.action_design.object_schema != FreeGsnkeActionDesign.SCHEMA:
            raise ValueError("FreeGSNKE power action design differs")
        for name in ("development_roster", "evaluation_roster"):
            if getattr(self, name).object_schema != FreeGsnkePhaseRequestRoster.SCHEMA:
                raise ValueError(f"FreeGSNKE power {name} schema differs")
        if self.selected_numerical_view.object_schema != (
            'empirical-lawhood/simulators/freegsnke/free-gsnke-numerical-view'
        ):
            raise ValueError("FreeGSNKE selected numerical view differs")
        require_sorted_unique_ids(
            self.qualified_numerical_views,
            attribute="object_id",
            field_name="qualified_numerical_views",
        )
        if self.selected_numerical_view not in self.qualified_numerical_views:
            raise ValueError("FreeGSNKE selected view was not jointly qualified")
        if (
            self.joint_power_analysis.selected_numerical_view != self.selected_numerical_view
            or self.joint_power_analysis.candidate_numerical_views != self.qualified_numerical_views
        ):
            raise ValueError("FreeGSNKE power freeze and joint analysis views differ")
        for name in (
            "development_unit_ids",
            "evaluation_unit_ids",
            "prospective_validation_unit_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        all_unit_ids = (
            *self.development_unit_ids,
            *self.evaluation_unit_ids,
            *self.prospective_validation_unit_ids,
        )
        if len(all_unit_ids) != len(set(all_unit_ids)):
            raise ValueError("FreeGSNKE development/evaluation/prospective validation units overlap")
        all_seeds = (*self.development_seeds, *self.evaluation_seeds, *self.prospective_validation_seeds)
        if (
            any(
                len(unit_ids) != len(seeds)
                for unit_ids, seeds in (
                    (self.development_unit_ids, self.development_seeds),
                    (self.evaluation_unit_ids, self.evaluation_seeds),
                    (self.prospective_validation_unit_ids, self.prospective_validation_seeds),
                )
            )
            or any(value < 0 for value in all_seeds)
            or len(all_seeds) != len(set(all_seeds))
        ):
            raise ValueError("FreeGSNKE power requires one disjoint seed per unit")
        for value in self.required_horizons_s:
            validate_decimal(value, field_name="required_horizons_s", minimum=Decimal(0))
        if (
            len(self.required_horizons_s) < 2
            or tuple(sorted(set(self.required_horizons_s))) != self.required_horizons_s
        ):
            raise ValueError("FreeGSNKE power horizon roster differs")
        require_sorted_unique_strings(
            self.required_receiver_ids,
            field_name="required_receiver_ids",
            allow_empty=False,
        )
        if self.required_receiver_ids != FREEGSNKE_POWER_RECEIVER_IDS:
            raise ValueError("FreeGSNKE power must retain every frozen receiver coordinate")
        minima = (
            self.development_minimum_complete_units,
            self.evaluation_minimum_complete_units,
            self.prospective_validation_minimum_complete_units,
        )
        if any(value < 2 for value in minima):
            raise ValueError("FreeGSNKE power minima require plural complete units")
        expected_limited = any(
            count < minimum
            for count, minimum in zip(
                (
                    len(self.development_unit_ids),
                    len(self.evaluation_unit_ids),
                    len(self.prospective_validation_unit_ids),
                ),
                minima,
                strict=True,
            )
        )
        if self.panel_envelope_limited != expected_limited:
            raise ValueError("FreeGSNKE power panel limitation is not roster-derived")
        if (
            (
                self.joint_power_analysis.selected_development_units,
                self.joint_power_analysis.selected_evaluation_units,
                self.joint_power_analysis.selected_prospective_evaluation_units,
            )
            != (
                len(self.development_unit_ids),
                len(self.evaluation_unit_ids),
                len(self.prospective_validation_unit_ids),
            )
            or self.joint_panel_grid_receiver_power_passed
            != self.joint_power_analysis.joint_power_passed
        ):
            raise ValueError("FreeGSNKE joint power disposition is not analysis-derived")
        require_sorted_unique_strings(
            self.simultaneous_family_ids,
            field_name="simultaneous_family_ids",
            allow_empty=False,
        )
        if set(self.simultaneous_family_ids) != {
            "all-decisive-falsifiers",
            "all-primary-estimands",
            "all-topology-metric-exchanges",
        }:
            raise ValueError("FreeGSNKE simultaneous family roster differs")
        validate_decimal(
            self.familywise_alpha,
            field_name="familywise_alpha",
            minimum=Decimal(0),
        )
        if not Decimal(0) < self.familywise_alpha < Decimal(1):
            raise ValueError("FreeGSNKE familywise alpha must be strictly interior")
        if self.complete_unit_bootstrap_replications < 2048:
            raise ValueError("FreeGSNKE complete-unit bootstrap is underspecified")
        if (
            self.familywise_alpha != self.joint_power_analysis.familywise_alpha
            or self.complete_unit_bootstrap_replications
            != self.joint_power_analysis.complete_unit_bootstrap_replications
            or self.required_receiver_ids != self.joint_power_analysis.receiver_coordinate_ids
        ):
            raise ValueError("FreeGSNKE power freeze changed its joint analysis")
        if self.complete_unit_role != ("OUTCOME_BLIND_PREPARED_EQUILIBRIUM_PROFILE_CURRENT_STATE"):
            raise ValueError("FreeGSNKE independent-unit role differs")
        if self.resampling_unit_role != "COMPLETE_PREPARATION":
            raise ValueError("FreeGSNKE resampling unit differs")
        if self.nested_observations_count_as_units:
            raise ValueError("FreeGSNKE nested branches/views cannot inflate replication")
        if not self.physical_response_and_certification_margins_distinct:
            raise ValueError("FreeGSNKE physical and certification margins are conflated")
        if not self.frozen_before_development or self.protected_outcome_access_count:
            raise ValueError("FreeGSNKE power freeze crossed development access")

    @property
    def issue_eligible(self) -> bool:
        return self.joint_panel_grid_receiver_power_passed and not self.panel_envelope_limited


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetDesignFreeze(CanonicalRecord):
    """Complete source-qualified G2/G3 design before target development."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-design-freeze'

    freeze_id: str
    scientific_design_basis: ObjectIdentity
    source_qualification: ObjectIdentity
    source_binding: ObjectIdentity
    action_design: ObjectIdentity
    mapping_candidate_roster: tuple[IndependentSubstrateTargetMappingCandidate, ...]
    denominator_candidate_lattice: tuple[IndependentSubstrateDenominatorCandidate, ...]
    proposed_denominator_candidate_id: str
    meaningful_split_candidate_ids: tuple[str, ...]
    merge_or_omission_candidate_ids: tuple[str, ...]
    impossible_candidate_ids: tuple[str, ...]
    forecast_alphabet: ObjectIdentity
    power_freeze: ObjectIdentity
    metric_topology_design: ObjectIdentity
    frozen_before_development: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        validate_stable_id(
            self.proposed_denominator_candidate_id,
            field_name="proposed_denominator_candidate_id",
        )
        if self.scientific_design_basis.object_schema != (
            'empirical-lawhood/independent-substrate-grounding/independent-substrate-scientific-design-basis'
        ):
            raise ValueError("FreeGSNKE design basis identity differs")
        if self.action_design.object_schema != FreeGsnkeActionDesign.SCHEMA:
            raise ValueError("FreeGSNKE target action design differs")
        if self.forecast_alphabet.object_schema != FreeGsnkeForecastAlphabetFreeze.SCHEMA:
            raise ValueError("FreeGSNKE target forecast alphabet differs")
        if self.power_freeze.object_schema != FreeGsnkePowerFreeze.SCHEMA:
            raise ValueError("FreeGSNKE target power freeze differs")
        if self.metric_topology_design.object_schema != FreeGsnkeMetricTopologyDesign.SCHEMA:
            raise ValueError("FreeGSNKE target metric/topology design differs")
        require_sorted_unique_ids(
            self.mapping_candidate_roster,
            attribute="candidate_id",
            field_name="mapping_candidate_roster",
        )
        if (
            not self.mapping_candidate_roster
            or len(self.mapping_candidate_roster) > 32
            or any(
                value.target_slot is not IndependentSubstrateTargetKind.FREEGSNKE
                for value in self.mapping_candidate_roster
            )
            or len({value.finite_roster_id for value in self.mapping_candidate_roster}) != 1
        ):
            raise ValueError("FreeGSNKE finite mapping roster differs from outcome-blind scientific grammar")
        require_sorted_unique_ids(
            self.denominator_candidate_lattice,
            attribute="candidate_id",
            field_name="denominator_candidate_lattice",
        )
        by_id = {value.candidate_id: value for value in self.denominator_candidate_lattice}
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
        groups = (
            (self.proposed_denominator_candidate_id,),
            self.meaningful_split_candidate_ids,
            self.merge_or_omission_candidate_ids,
            self.impossible_candidate_ids,
        )
        flattened = tuple(value for group in groups for value in group)
        if len(flattened) != len(set(flattened)) or set(flattened) != set(by_id):
            raise ValueError("FreeGSNKE denominator kinds do not partition the lattice")
        if any(
            by_id[candidate_id].impossible != (candidate_id in self.impossible_candidate_ids)
            for candidate_id in by_id
        ):
            raise ValueError("FreeGSNKE denominator impossible kind differs")
        if not self.frozen_before_development or self.protected_outcome_access_count:
            raise ValueError("FreeGSNKE target design crossed development access")


__all__ = [
    "FREEGSNKE_FORECAST_STATE_IDS",
    "FREEGSNKE_LAW_OPERAND_IDS",
    "FREEGSNKE_POWER_RECEIVER_IDS",
    "FREEGSNKE_REQUIRED_FORECAST_ROLE_IDS",
    'FreeGsnkeForecastAlphabetFreeze',
    'FreeGsnkeJointPowerAnalysis',
    'FreeGsnkeMetricTopologyDesign',
    'FreeGsnkeMetricTopologyExchangeDesign',
    'FreeGsnkeMetricTopologyLevelBinding',
    'FreeGsnkePowerFreeze',
    'FreeGsnkeTargetDesignFreeze',
    'FreeGsnkeTargetForecast',
]
