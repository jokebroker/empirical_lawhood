"""Exact, outcome-protected FAIR-MAST archive law contracts.

These records instantiate Section 7.6 of the flagship plan.  They translate
archive-native values but do not finalize a generic response law.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
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


class ArchiveRosterRole(StrEnum):
    ARCHIVE_DEVELOPMENT = "ARCHIVE_DEVELOPMENT"
    ARCHIVE_CONSTRUCT = "ARCHIVE_CONSTRUCT"
    ARCHIVE_PROTECTED = "ARCHIVE_PROTECTED"


class ArchiveActionClass(StrEnum):
    NBI_DOWN = "NBI_DOWN"
    NBI_HOLD = "NBI_HOLD"
    NBI_UP = "NBI_UP"


class ArchiveResponseClass(StrEnum):
    DOWN_OR_NEGATIVE = "DOWN_OR_NEGATIVE"
    WITHIN_MATERIAL_EQUIVALENCE = "WITHIN_MATERIAL_EQUIVALENCE"
    UP_OR_POSITIVE = "UP_OR_POSITIVE"


class ArchiveRepresentationKind(StrEnum):
    ACTION_ONLY = "ACTION_ONLY"
    CAPACITY_MATCHED_UNTYPED_HISTORY = "CAPACITY_MATCHED_UNTYPED_HISTORY"
    CURRENT_STATE = "CURRENT_STATE"
    SELECTED_RESTRICTIVE = "SELECTED_RESTRICTIVE"


@dataclass(frozen=True, slots=True)
class ArchiveExposureEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-exposure-entry'

    entry_id: str
    shot_id: str
    exposure_kind: str
    source_identity: ObjectIdentity
    exact_outcome_access_known: bool
    prior_access_description: str

    def __post_init__(self) -> None:
        validate_stable_id(self.entry_id, field_name="entry_id")
        validate_stable_id(self.shot_id, field_name="shot_id")
        validate_stable_id(self.exposure_kind, field_name="exposure_kind")
        validate_nonempty(
            self.prior_access_description,
            field_name="prior_access_description",
        )


@dataclass(frozen=True, slots=True)
class MastArchiveExposureLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mast-archive-exposure-ledger'

    ledger_id: str
    entries: tuple[ArchiveExposureEntry, ...]
    locally_retained_materialization_ids: tuple[str, ...]
    protected_ineligible_shot_ids: tuple[str, ...]
    residual_limitation_codes: tuple[str, ...]
    source_classification: str
    completed_before_source_access: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        require_sorted_unique_ids(self.entries, attribute="entry_id", field_name="entries")
        for name, values in (
            (
                "locally_retained_materialization_ids",
                self.locally_retained_materialization_ids,
            ),
            ("protected_ineligible_shot_ids", self.protected_ineligible_shot_ids),
            ("residual_limitation_codes", self.residual_limitation_codes),
        ):
            require_sorted_unique_strings(values, field_name=name)
        exposed_shots = {value.shot_id for value in self.entries}
        if not exposed_shots.issubset(self.protected_ineligible_shot_ids):
            raise ValueError("exposure ledger permits a previously accessed protected shot")
        if self.source_classification != (
            "PROJECT_EXPOSED_SOURCE / OUTCOME_PROTECTED_EVALUATION_ROSTER"
        ):
            raise ValueError("archive source novelty classification differs")
        if (
            not self.completed_before_source_access
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("archive exposure ledger was not frozen outcome-blind")


@dataclass(frozen=True, slots=True)
class ArchiveRosterSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-roster-spec'

    roster_spec_id: str
    campaign_ids: tuple[str, ...]
    development_per_campaign: int
    construct_per_campaign: int
    preferred_protected_per_campaign: int
    minimum_protected_per_campaign: int
    maximum_events_per_shot: int
    protected_selection_order: tuple[str, ...]
    minimum_evaluable_active_per_campaign: int
    minimum_evaluable_active_total: int

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_spec_id, field_name="roster_spec_id")
        if (
            self.campaign_ids != ("M7", "M8", "M9")
            or self.development_per_campaign != 30
            or self.construct_per_campaign != 20
            or self.preferred_protected_per_campaign != 80
            or self.minimum_protected_per_campaign != 60
            or self.maximum_events_per_shot != 1
            or self.protected_selection_order != ("PROTECTED_EIGHTY_PER_CAMPAIGN", "PROTECTED_SIXTY_PER_CAMPAIGN")
            or self.minimum_evaluable_active_per_campaign != 10
            or self.minimum_evaluable_active_total != 30
        ):
            raise ValueError("archive roster differs from development30/construct20/protected80-or60 per campaign")


@dataclass(frozen=True, slots=True)
class ArchiveEventSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-event-spec'

    event_spec_id: str
    valid_interval_leading_exclusion_ms: Decimal
    valid_interval_trailing_exclusion_ms: Decimal
    pre_summary_start_ms: Decimal
    pre_summary_end_ms: Decimal
    post_summary_start_ms: Decimal
    post_summary_end_ms: Decimal
    minimum_valid_samples_per_summary: int
    active_threshold_kw: Decimal
    hold_minimum_tolerance_kw: Decimal
    hold_quantization_multiplier: Decimal
    stability_start_ms: Decimal
    stability_end_ms: Decimal
    concurrent_window_start_ms: Decimal
    concurrent_window_end_ms: Decimal
    concurrent_quantization_multiplier: Decimal
    active_tie_disposition: str
    intermediate_disposition: str
    incomplete_concurrency_disposition: str
    event_uses_endpoint_fields: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.event_spec_id, field_name="event_spec_id")
        expected = (
            self.valid_interval_leading_exclusion_ms == Decimal("20"),
            self.valid_interval_trailing_exclusion_ms == Decimal("80"),
            self.pre_summary_start_ms == Decimal("-10"),
            self.pre_summary_end_ms == Decimal("-2"),
            self.post_summary_start_ms == Decimal("2"),
            self.post_summary_end_ms == Decimal("10"),
            self.minimum_valid_samples_per_summary == 2,
            self.active_threshold_kw == Decimal("100"),
            self.hold_minimum_tolerance_kw == Decimal("1"),
            self.hold_quantization_multiplier == Decimal("3"),
            self.stability_start_ms == Decimal("-40"),
            self.stability_end_ms == Decimal("-2"),
            self.concurrent_window_start_ms == Decimal("-10"),
            self.concurrent_window_end_ms == Decimal("10"),
            self.concurrent_quantization_multiplier == Decimal("3"),
            self.active_tie_disposition == "MULTIPLE_EQUAL_EVENTS",
            self.intermediate_disposition == "ACTION_MAGNITUDE_AMBIGUOUS",
            self.incomplete_concurrency_disposition == "CONCURRENT_ACTUATOR_STATUS_UNQUALIFIED",
            not self.event_uses_endpoint_fields,
        )
        if not all(expected):
            raise ValueError("archive event rule differs from Section 7.6.2")


@dataclass(frozen=True, slots=True)
class ArchiveStateHistorySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-state-history-spec'

    state_history_spec_id: str
    denominator_field_ids: tuple[str, ...]
    receiver_field_id: str
    history_signal_ids: tuple[str, ...]
    history_blocks_ms: tuple[str, ...]
    block_reducer_ids: tuple[str, ...]
    latest_sample_exclusive_ms: Decimal
    missing_value_policy: str

    def __post_init__(self) -> None:
        validate_stable_id(self.state_history_spec_id, field_name="state_history_spec_id")
        for name, values in (
            ("denominator_field_ids", self.denominator_field_ids),
            ("history_signal_ids", self.history_signal_ids),
            ("history_blocks_ms", self.history_blocks_ms),
            ("block_reducer_ids", self.block_reducer_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        required_denominator = {
            "abs-b0",
            "a-minor-lcfs",
            "campaign",
            "electron-density-core-profile",
            "elongation",
            "plasma-current",
            "r-major",
            "shot-source-identity",
        }
        if (
            not required_denominator.issubset(self.denominator_field_ids)
            or self.receiver_field_id != "t-e-core"
            or self.history_signal_ids
            != ("electron-density", "nbi-power", "plasma-current", "t-e-core", "wmhd")
            or self.history_blocks_ms != ("-12:-2", "-22:-12", "-32:-22", "-42:-32")
            or self.block_reducer_ids != ("least-squares-native-time-slope", "median")
            or self.latest_sample_exclusive_ms != Decimal("-2")
            or self.missing_value_policy != "UNEVALUABLE_NO_IMPUTATION"
        ):
            raise ValueError("archive state/history contract differs")


@dataclass(frozen=True, slots=True)
class ArchiveEndpointSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-endpoint-spec'

    endpoint_spec_id: str
    reference_window_start_ms: Decimal
    reference_window_end_ms: Decimal
    endpoint_offset_ms: Decimal
    endpoint_tolerance_ms: Decimal
    response_unit: str
    materiality_ev: Decimal
    receiver_valid_min_ev: Decimal
    receiver_valid_max_ev: Decimal
    validity_disposition_ids: tuple[str, ...]
    diagnostics_nonpromotable: tuple[str, ...]
    unavailable_uncertainty_policy: str

    def __post_init__(self) -> None:
        validate_stable_id(self.endpoint_spec_id, field_name="endpoint_spec_id")
        require_sorted_unique_strings(
            self.validity_disposition_ids,
            field_name="validity_disposition_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.diagnostics_nonpromotable,
            field_name="diagnostics_nonpromotable",
            allow_empty=False,
        )
        if (
            self.reference_window_start_ms != Decimal("-10")
            or self.reference_window_end_ms != Decimal("-2")
            or self.endpoint_offset_ms != Decimal("40")
            or self.endpoint_tolerance_ms != Decimal("1")
            or self.response_unit != "eV"
            or self.materiality_ev != Decimal("100")
            or self.receiver_valid_min_ev != Decimal("-1000")
            or self.receiver_valid_max_ev != Decimal("2000")
            or set(self.validity_disposition_ids)
            != {"INVALID_MEASUREMENT", "MISSING_DIAGNOSTIC", "OUT_OF_WINDOW"}
            or set(self.diagnostics_nonpromotable)
            != {"profile-temperature", "thermal-content", "wmhd"}
            or self.unavailable_uncertainty_policy != "RETAIN_UNAVAILABLE"
        ):
            raise ValueError("archive endpoint contract differs")


@dataclass(frozen=True, slots=True)
class ArchiveRepresentationSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-representation-spec'

    representation_spec_id: str
    representations: tuple[ArchiveRepresentationKind, ...]
    model_family: str
    loss: str
    penalty_multipliers: tuple[Decimal, ...]
    blocked_fold_count: int
    block_size_per_campaign: int
    minimum_evaluable_per_block: int
    minimum_active_per_campaign: int
    capacity_match_edf_tolerance: Decimal
    unpenalized_intercept: bool
    continuous_standardization_role: ArchiveRosterRole
    zero_variance_policy: str

    def __post_init__(self) -> None:
        validate_stable_id(
            self.representation_spec_id,
            field_name="representation_spec_id",
        )
        if (
            self.representations != tuple(ArchiveRepresentationKind)
            or self.model_family != "INTERCEPT_PLUS_RIDGE_LINEAR"
            or self.loss != "CAMPAIGN_STANDARDIZED_BLOCKED_SQUARED_ERROR"
            or self.penalty_multipliers
            != (Decimal("0"), Decimal("1e-8"), Decimal("1e-6"), Decimal("1e-4"))
            or self.blocked_fold_count != 3
            or self.block_size_per_campaign != 10
            or self.minimum_evaluable_per_block != 8
            or self.minimum_active_per_campaign != 8
            or self.capacity_match_edf_tolerance != Decimal("0.5")
            or not self.unpenalized_intercept
            or self.continuous_standardization_role is not ArchiveRosterRole.ARCHIVE_DEVELOPMENT
            or self.zero_variance_policy != "REMOVE_IDENTICALLY_FROM_OWNING_MODELS"
        ):
            raise ValueError("archive representation/fitting ladder differs")


@dataclass(frozen=True, slots=True)
class ArchiveModelSupportSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-model-support-spec'

    support_spec_id: str
    support_roles: tuple[ArchiveRosterRole, ...]
    categorical_rule: str
    continuous_rule: str
    coordinate_space: str
    zero_width_rule: str
    clipping_allowed: bool
    extrapolation_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.support_spec_id, field_name="support_spec_id")
        if (
            self.support_roles
            != (
                ArchiveRosterRole.ARCHIVE_DEVELOPMENT,
                ArchiveRosterRole.ARCHIVE_CONSTRUCT,
            )
            or self.categorical_rule != "ALL_REQUIRED_PATTERNS_OBSERVED"
            or self.continuous_rule != "CLOSED_FEATUREWISE_MIN_MAX"
            or self.coordinate_space != "NATIVE_PRE_STANDARDIZATION_MODEL_PROJECTION"
            or self.zero_width_rule != "EXACT_EQUALITY_ONLY"
            or self.clipping_allowed
            or self.extrapolation_allowed
        ):
            raise ValueError("archive model support rule differs")


@dataclass(frozen=True, slots=True)
class ArchiveCounterfactualActionTemplateSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-counterfactual-action-template-spec'

    template_spec_id: str
    action_classes: tuple[ArchiveActionClass, ...]
    source_roles: tuple[ArchiveRosterRole, ...]
    reducer: str
    campaign_weighting: str
    receiver_outcome_used: bool
    requires_complete_action_support: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.template_spec_id, field_name="template_spec_id")
        if (
            self.action_classes != (ArchiveActionClass.NBI_DOWN, ArchiveActionClass.NBI_UP)
            or self.source_roles
            != (
                ArchiveRosterRole.ARCHIVE_DEVELOPMENT,
                ArchiveRosterRole.ARCHIVE_CONSTRUCT,
            )
            or self.reducer != "EQUAL_CAMPAIGN_COMPONENTWISE_MEDIAN_RELATIVE_TRACE"
            or self.campaign_weighting != "ONE_THIRD_EACH"
            or self.receiver_outcome_used
            or not self.requires_complete_action_support
        ):
            raise ValueError("archive counterfactual action template differs")


@dataclass(frozen=True, slots=True)
class ArchiveTemporalPlaceboSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-temporal-placebo-spec'

    placebo_spec_id: str
    future_action_used: bool
    pseudo_endpoint_late_ms: Decimal
    pseudo_endpoint_early_ms: Decimal
    sample_tolerance_ms: Decimal
    predictor_cutoff_ms: Decimal
    raw_contrast_lower_bound_ev: Decimal
    recurrence_wilson_upper_bound: Decimal
    same_campaign_action_minima_required: bool
    h10_control_kind: str

    def __post_init__(self) -> None:
        validate_stable_id(self.placebo_spec_id, field_name="placebo_spec_id")
        if (
            not self.future_action_used
            or self.pseudo_endpoint_late_ms != Decimal("-2")
            or self.pseudo_endpoint_early_ms != Decimal("-42")
            or self.sample_tolerance_ms != Decimal("1")
            or self.predictor_cutoff_ms != Decimal("-42")
            or self.raw_contrast_lower_bound_ev != Decimal("100")
            or self.recurrence_wilson_upper_bound != Decimal("0.65")
            or not self.same_campaign_action_minima_required
            or self.h10_control_kind != "PREACTION_TIME_SHIFT"
        ):
            raise ValueError("archive temporal placebo differs")


@dataclass(frozen=True, slots=True)
class ArchiveLawSpec(CanonicalRecord):
    """Complete fixed archive law; F2 materializes but cannot tune it."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-law-spec'

    law_spec_id: str
    roster: ArchiveRosterSpec
    event: ArchiveEventSpec
    state_history: ArchiveStateHistorySpec
    endpoint: ArchiveEndpointSpec
    representation: ArchiveRepresentationSpec
    model_support: ArchiveModelSupportSpec
    action_template: ArchiveCounterfactualActionTemplateSpec
    temporal_placebo: ArchiveTemporalPlaceboSpec
    action_trace_offset_ms: tuple[Decimal, Decimal]
    action_projection_offsets_ms: tuple[Decimal, ...]
    action_sample_tolerance_ms: Decimal
    calibration_gamma_candidates: tuple[Decimal, ...]
    calibration_target_coverage: Decimal
    calibration_minimum_coverage: Decimal
    calibration_minimum_evaluable_per_campaign: int
    calibration_minimum_active_per_campaign: int
    bootstrap_replicates: int
    campaign_estimand: str
    false_safe_rule: str
    undeclared_counts_incorrect: bool
    protected_outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.law_spec_id, field_name="law_spec_id")
        if (
            self.action_trace_offset_ms != (Decimal("-10"), Decimal("40"))
            or self.action_projection_offsets_ms
            != (
                Decimal("-10"),
                Decimal("-2"),
                Decimal("2"),
                Decimal("10"),
                Decimal("20"),
                Decimal("30"),
                Decimal("40"),
            )
            or self.action_sample_tolerance_ms != Decimal("1")
            or self.calibration_gamma_candidates
            != (Decimal("0.80"), Decimal("0.90"), Decimal("0.95"))
            or self.calibration_target_coverage != Decimal("0.50")
            or self.calibration_minimum_coverage != Decimal("0.25")
            or self.calibration_minimum_evaluable_per_campaign != 16
            or self.calibration_minimum_active_per_campaign != 5
            or self.bootstrap_replicates != 10_000
            or self.campaign_estimand != "EQUAL_CAMPAIGN_ACTION_MEAN"
            or self.false_safe_rule
            != "SUPPORTED_DECLARED_CLASS_WRONG_OR_RECEIVER_OUTSIDE_VALID_INTERVAL"
            or not self.undeclared_counts_incorrect
            or self.protected_outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("archive law specification differs from the final plan")


@dataclass(frozen=True, slots=True)
class MastArchiveSourceFeasibilityAmendment(CanonicalRecord):
    "Additive source-feasibility amendment over the immutable archive law."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mast-archive-source-feasibility-amendment'

    amendment_id: str
    predecessor_law: ObjectIdentity
    campaign_ids: tuple[str, ...]
    development_per_campaign: int
    construct_per_campaign: int
    preferred_protected_per_campaign: int
    minimum_protected_per_campaign: int
    minimum_evaluable_active_per_campaign: int
    minimum_evaluable_active_total: int
    action_classes: tuple[str, ...]
    nbi_hold_available: bool
    metadata_absence_predicates: tuple[str, ...]
    concurrent_command_array_paths: tuple[str, ...]
    concurrent_command_rule: str
    realized_feedback_array_paths: tuple[str, ...]
    realized_feedback_disposition: str
    active_maximum_plateau_rule: str
    receiver_sample_tolerance_ms: Decimal
    receiver_nearest_tie_rule: str
    source_clock_semantics: str
    campaign_estimand: str
    unchanged_predecessor_component_ids: tuple[str, ...]
    protected_outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.amendment_id, field_name="amendment_id")
        for name in (
            "campaign_ids",
            "action_classes",
            "metadata_absence_predicates",
            "concurrent_command_array_paths",
            "realized_feedback_array_paths",
            "unchanged_predecessor_component_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        validate_decimal(
            self.receiver_sample_tolerance_ms,
            field_name="receiver_sample_tolerance_ms",
        )
        if (
            self.campaign_ids != ("M8", "M9")
            or self.development_per_campaign != 30
            or self.construct_per_campaign != 20
            or self.preferred_protected_per_campaign != 80
            or self.minimum_protected_per_campaign != 60
            or self.minimum_evaluable_active_per_campaign != 10
            or self.minimum_evaluable_active_total != 20
            or self.action_classes != ("NBI_DOWN", "NBI_UP")
            or self.nbi_hold_available
            or self.metadata_absence_predicates != ("pellets-false", "rmp-coil-false")
            or self.concurrent_command_array_paths
            != (
                "gas_injection/valve_target_voltage",
                "pulse_schedule/i_plasma",
                "pulse_schedule/n_e_line",
                "pulse_schedule/z_ref",
            )
            or self.concurrent_command_rule
            != "EXACT_CONSTANCY_ON_LEVEL2_ZERO_ORDER_GRID_-10_TO_PLUS10_MS"
            or self.realized_feedback_array_paths
            != (
                "gas_injection/valve_voltage",
                "pf_active/coil_current",
                "pf_active/coil_voltage",
            )
            or self.realized_feedback_disposition
            != "CONTROLLED_DELIVERY_DIAGNOSTIC_NOT_INDEPENDENT_EVENT_EXCLUSION"
            or self.active_maximum_plateau_rule
            != "ONE_CONTIGUOUS_MAXIMUM_PLATEAU_EARLIEST_PIVOT_ELSE_MULTIPLE_EVENTS"
            or self.receiver_sample_tolerance_ms != Decimal("2.5")
            or self.receiver_nearest_tie_rule != "EARLIER_SOURCE_SAMPLE"
            or self.source_clock_semantics != "FAIR_MAST_LEVEL2_INTERPOLATED_CLOCKS"
            or self.campaign_estimand != "EQUAL_M8_M9_CAMPAIGN_ACTION_MEAN"
            or self.unchanged_predecessor_component_ids
            != (
                "action-descriptor-and-projection",
                "bootstrap-calibration-and-support",
                "pre-action-state-history",
                "representations-and-model-family",
                "response-materiality-and-validity",
            )
            or self.protected_outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("archive source-feasibility amendment differs from the accepted source remedy")


@dataclass(frozen=True, slots=True)
class ArchiveActionTraceIdentity(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/archive-action-trace-identity'

    trace_id: str
    shot_id: str
    action_class: ArchiveActionClass
    source_clock_id: str
    native_unit: str
    pre_median_kw: Decimal
    post_median_kw: Decimal
    signed_change_kw: Decimal
    signed_dose_kw_ms: Decimal
    source_payload_sha256: str
    requested_stage_observed: bool
    accepted_stage_observed: bool
    applied_stage_observed: bool
    realized_stage_observed: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("trace_id", self.trace_id),
            ("shot_id", self.shot_id),
            ("source_clock_id", self.source_clock_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        for decimal_name, decimal_value in (
            ("pre_median_kw", self.pre_median_kw),
            ("post_median_kw", self.post_median_kw),
            ("signed_change_kw", self.signed_change_kw),
            ("signed_dose_kw_ms", self.signed_dose_kw_ms),
        ):
            validate_decimal(decimal_value, field_name=decimal_name)
        validate_sha256(self.source_payload_sha256, field_name="source_payload_sha256")
        if (
            self.requested_stage_observed
            or self.accepted_stage_observed
            or self.applied_stage_observed
            or not self.realized_stage_observed
        ):
            raise ValueError("archive action trace fabricates an unobserved delivery stage")


def classify_archive_response(
    response_ev: Decimal,
    *,
    endpoint: ArchiveEndpointSpec,
) -> ArchiveResponseClass:
    """Apply the exact native-eV material classes without coercion or fitting."""

    validate_decimal(response_ev, field_name="response_ev")
    if response_ev < -endpoint.materiality_ev:
        return ArchiveResponseClass.DOWN_OR_NEGATIVE
    if response_ev > endpoint.materiality_ev:
        return ArchiveResponseClass.UP_OR_POSITIVE
    return ArchiveResponseClass.WITHIN_MATERIAL_EQUIVALENCE


def build_archive_law_spec() -> ArchiveLawSpec:
    """Build the sole plan-fixed archive law specification without source outcomes."""

    return ArchiveLawSpec(
        law_spec_id='archive-law-spec.mastu-beam-heating-temperature-response',
        roster=ArchiveRosterSpec(
            roster_spec_id='archive-roster-spec.mastu-beam-heating-temperature-response',
            campaign_ids=("M7", "M8", "M9"),
            development_per_campaign=30,
            construct_per_campaign=20,
            preferred_protected_per_campaign=80,
            minimum_protected_per_campaign=60,
            maximum_events_per_shot=1,
            protected_selection_order=("PROTECTED_EIGHTY_PER_CAMPAIGN", "PROTECTED_SIXTY_PER_CAMPAIGN"),
            minimum_evaluable_active_per_campaign=10,
            minimum_evaluable_active_total=30,
        ),
        event=ArchiveEventSpec(
            event_spec_id='archive-event-spec.mastu-beam-heating-temperature-response',
            valid_interval_leading_exclusion_ms=Decimal("20"),
            valid_interval_trailing_exclusion_ms=Decimal("80"),
            pre_summary_start_ms=Decimal("-10"),
            pre_summary_end_ms=Decimal("-2"),
            post_summary_start_ms=Decimal("2"),
            post_summary_end_ms=Decimal("10"),
            minimum_valid_samples_per_summary=2,
            active_threshold_kw=Decimal("100"),
            hold_minimum_tolerance_kw=Decimal("1"),
            hold_quantization_multiplier=Decimal("3"),
            stability_start_ms=Decimal("-40"),
            stability_end_ms=Decimal("-2"),
            concurrent_window_start_ms=Decimal("-10"),
            concurrent_window_end_ms=Decimal("10"),
            concurrent_quantization_multiplier=Decimal("3"),
            active_tie_disposition="MULTIPLE_EQUAL_EVENTS",
            intermediate_disposition="ACTION_MAGNITUDE_AMBIGUOUS",
            incomplete_concurrency_disposition=("CONCURRENT_ACTUATOR_STATUS_UNQUALIFIED"),
            event_uses_endpoint_fields=False,
        ),
        state_history=ArchiveStateHistorySpec(
            state_history_spec_id='archive-state-history-spec.mastu-beam-heating-temperature-response',
            denominator_field_ids=(
                "a-minor-lcfs",
                "abs-b0",
                "campaign",
                "electron-density-core-profile",
                "elongation",
                "plasma-current",
                "r-major",
                "shot-source-identity",
            ),
            receiver_field_id="t-e-core",
            history_signal_ids=(
                "electron-density",
                "nbi-power",
                "plasma-current",
                "t-e-core",
                "wmhd",
            ),
            history_blocks_ms=("-12:-2", "-22:-12", "-32:-22", "-42:-32"),
            block_reducer_ids=("least-squares-native-time-slope", "median"),
            latest_sample_exclusive_ms=Decimal("-2"),
            missing_value_policy="UNEVALUABLE_NO_IMPUTATION",
        ),
        endpoint=ArchiveEndpointSpec(
            endpoint_spec_id='archive-endpoint-spec.mastu-beam-heating-temperature-response',
            reference_window_start_ms=Decimal("-10"),
            reference_window_end_ms=Decimal("-2"),
            endpoint_offset_ms=Decimal("40"),
            endpoint_tolerance_ms=Decimal("1"),
            response_unit="eV",
            materiality_ev=Decimal("100"),
            receiver_valid_min_ev=Decimal("-1000"),
            receiver_valid_max_ev=Decimal("2000"),
            validity_disposition_ids=(
                "INVALID_MEASUREMENT",
                "MISSING_DIAGNOSTIC",
                "OUT_OF_WINDOW",
            ),
            diagnostics_nonpromotable=(
                "profile-temperature",
                "thermal-content",
                "wmhd",
            ),
            unavailable_uncertainty_policy="RETAIN_UNAVAILABLE",
        ),
        representation=ArchiveRepresentationSpec(
            representation_spec_id='archive-representation-spec.mastu-beam-heating-temperature-response',
            representations=tuple(ArchiveRepresentationKind),
            model_family="INTERCEPT_PLUS_RIDGE_LINEAR",
            loss="CAMPAIGN_STANDARDIZED_BLOCKED_SQUARED_ERROR",
            penalty_multipliers=(
                Decimal("0"),
                Decimal("1e-8"),
                Decimal("1e-6"),
                Decimal("1e-4"),
            ),
            blocked_fold_count=3,
            block_size_per_campaign=10,
            minimum_evaluable_per_block=8,
            minimum_active_per_campaign=8,
            capacity_match_edf_tolerance=Decimal("0.5"),
            unpenalized_intercept=True,
            continuous_standardization_role=ArchiveRosterRole.ARCHIVE_DEVELOPMENT,
            zero_variance_policy="REMOVE_IDENTICALLY_FROM_OWNING_MODELS",
        ),
        model_support=ArchiveModelSupportSpec(
            support_spec_id='archive-model-support-spec.mastu-beam-heating-temperature-response',
            support_roles=(
                ArchiveRosterRole.ARCHIVE_DEVELOPMENT,
                ArchiveRosterRole.ARCHIVE_CONSTRUCT,
            ),
            categorical_rule="ALL_REQUIRED_PATTERNS_OBSERVED",
            continuous_rule="CLOSED_FEATUREWISE_MIN_MAX",
            coordinate_space="NATIVE_PRE_STANDARDIZATION_MODEL_PROJECTION",
            zero_width_rule="EXACT_EQUALITY_ONLY",
            clipping_allowed=False,
            extrapolation_allowed=False,
        ),
        action_template=ArchiveCounterfactualActionTemplateSpec(
            template_spec_id='archive-action-template-spec.mastu-beam-heating-temperature-response',
            action_classes=(ArchiveActionClass.NBI_DOWN, ArchiveActionClass.NBI_UP),
            source_roles=(
                ArchiveRosterRole.ARCHIVE_DEVELOPMENT,
                ArchiveRosterRole.ARCHIVE_CONSTRUCT,
            ),
            reducer="EQUAL_CAMPAIGN_COMPONENTWISE_MEDIAN_RELATIVE_TRACE",
            campaign_weighting="ONE_THIRD_EACH",
            receiver_outcome_used=False,
            requires_complete_action_support=True,
        ),
        temporal_placebo=ArchiveTemporalPlaceboSpec(
            placebo_spec_id='archive-temporal-placebo-spec.mastu-beam-heating-temperature-response',
            future_action_used=True,
            pseudo_endpoint_late_ms=Decimal("-2"),
            pseudo_endpoint_early_ms=Decimal("-42"),
            sample_tolerance_ms=Decimal("1"),
            predictor_cutoff_ms=Decimal("-42"),
            raw_contrast_lower_bound_ev=Decimal("100"),
            recurrence_wilson_upper_bound=Decimal("0.65"),
            same_campaign_action_minima_required=True,
            h10_control_kind="PREACTION_TIME_SHIFT",
        ),
        action_trace_offset_ms=(Decimal("-10"), Decimal("40")),
        action_projection_offsets_ms=(
            Decimal("-10"),
            Decimal("-2"),
            Decimal("2"),
            Decimal("10"),
            Decimal("20"),
            Decimal("30"),
            Decimal("40"),
        ),
        action_sample_tolerance_ms=Decimal("1"),
        calibration_gamma_candidates=(
            Decimal("0.80"),
            Decimal("0.90"),
            Decimal("0.95"),
        ),
        calibration_target_coverage=Decimal("0.50"),
        calibration_minimum_coverage=Decimal("0.25"),
        calibration_minimum_evaluable_per_campaign=16,
        calibration_minimum_active_per_campaign=5,
        bootstrap_replicates=10_000,
        campaign_estimand="EQUAL_CAMPAIGN_ACTION_MEAN",
        false_safe_rule=("SUPPORTED_DECLARED_CLASS_WRONG_OR_RECEIVER_OUTSIDE_VALID_INTERVAL"),
        undeclared_counts_incorrect=True,
        protected_outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )


__all__ = [
    "ArchiveActionClass",
    'ArchiveActionTraceIdentity',
    'ArchiveCounterfactualActionTemplateSpec',
    'ArchiveEndpointSpec',
    'ArchiveEventSpec',
    'ArchiveExposureEntry',
    'ArchiveLawSpec',
    'MastArchiveSourceFeasibilityAmendment',
    'ArchiveModelSupportSpec',
    "ArchiveRepresentationKind",
    'ArchiveRepresentationSpec',
    "ArchiveResponseClass",
    "ArchiveRosterRole",
    'ArchiveRosterSpec',
    'ArchiveStateHistorySpec',
    'MastArchiveExposureLedger',
    'ArchiveTemporalPlaceboSpec',
    'build_archive_law_spec',
    "classify_archive_response",
]
