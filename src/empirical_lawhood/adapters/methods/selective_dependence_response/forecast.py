"""Development-parent and frozen forecast construction for selective dependence response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .analysis import SelectiveDependenceResponseFiniteLawCalibration, SelectiveDependenceResponseSelectiveDependenceSignature
from .comparators import SelectiveDependenceResponseComparatorEncoding, SelectiveDependenceResponseForecastSeparationReport
from .denominator import SelectiveDependenceResponseDenominatorAlternative, SelectiveDependenceResponseDenominatorSelection
from .power import SelectiveDependenceResponsePowerDesignQualification
from .contracts import SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID, digest_ids
from .contracts import SelectiveDependenceResponseContextDecision, SelectiveDependenceResponseDisposition, SelectiveDependenceResponseExchangeForecast, SelectiveDependenceResponseExchangeExpectation, SelectiveDependenceResponseSelectiveLawForecast


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseForecastChallengeQualification(CanonicalRecord):
    """Pre-issue proof that the finite forecast can test its own proposition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-forecast-challenge-qualification'

    qualification_id: str
    target_id: str
    active_exchange_count: int
    invariant_exchange_count: int
    support_boundary_count: int
    target_sink_conflict_case_ids: tuple[str, ...]
    policy_dispositions: tuple[SelectiveDependenceResponseDisposition, ...]
    hold_viability_states: tuple[bool, ...]
    mixed_hold_viability_present: bool
    target_has_two_policy_dispositions: bool
    qualified: bool
    stop_codes: tuple[str, ...]
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("qualification_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "active_exchange_count",
            "invariant_exchange_count",
            "support_boundary_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        require_sorted_unique_strings(
            self.target_sink_conflict_case_ids,
            field_name="target_sink_conflict_case_ids",
        )
        if tuple(sorted(set(self.policy_dispositions), key=lambda value: value.value)) != (
            self.policy_dispositions
        ):
            raise ValueError("policy dispositions must be sorted and unique")
        expected_two = len(self.policy_dispositions) >= 2
        if self.target_has_two_policy_dispositions != expected_two:
            raise ValueError("target policy-diversity flag is not data-derived")
        if self.hold_viability_states != tuple(sorted(set(self.hold_viability_states))):
            raise ValueError("hold viability states must be sorted and unique")
        expected_mixed_hold = self.hold_viability_states == (False, True)
        if self.mixed_hold_viability_present != expected_mixed_hold:
            raise ValueError("mixed hold-viability flag is not data-derived")
        require_sorted_unique_strings(self.stop_codes, field_name="stop_codes")
        if self.qualified == bool(self.stop_codes):
            raise ValueError("forecast challenge qualification and stops differ")
        if self.evaluation_outcome_count:
            raise ValueError("forecast challenge cannot use evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("forecast challenge must remain development visible")


def qualify_forecast_challenge(
    *,
    target_id: str,
    exchange_forecasts: tuple[SelectiveDependenceResponseExchangeForecast, ...],
    context_forecasts: tuple[SelectiveDependenceResponseContextDecision, ...],
    target_sink_conflict_case_ids: tuple[str, ...],
) -> SelectiveDependenceResponseForecastChallengeQualification:
    counts = {
        expectation: sum(value.expectation is expectation for value in exchange_forecasts)
        for expectation in SelectiveDependenceResponseExchangeExpectation
    }
    dispositions = tuple(
        sorted({value.disposition for value in context_forecasts}, key=lambda value: value.value)
    )
    hold_states = tuple(
        sorted({value.hold_viable for value in context_forecasts if value.hold_viable is not None})
    )
    stops = []
    if not counts[SelectiveDependenceResponseExchangeExpectation.ACTIVE]:
        stops.append("active-exchange-absent")
    if not counts[SelectiveDependenceResponseExchangeExpectation.INVARIANT]:
        stops.append("invariant-exchange-absent")
    if not counts[SelectiveDependenceResponseExchangeExpectation.SUPPORT_BOUNDARY]:
        stops.append("support-boundary-absent")
    if not target_sink_conflict_case_ids:
        stops.append("target-sink-conflict-absent")
    if len(dispositions) < 2:
        stops.append("policy-disposition-diversity-absent")
    if hold_states != (False, True):
        stops.append("hold-viability-contrast-absent")
    ordered_stops = tuple(sorted(stops))
    return SelectiveDependenceResponseForecastChallengeQualification(
        qualification_id=f"qualification.{target_id}.forecast-challenge",
        target_id=target_id,
        active_exchange_count=counts[SelectiveDependenceResponseExchangeExpectation.ACTIVE],
        invariant_exchange_count=counts[SelectiveDependenceResponseExchangeExpectation.INVARIANT],
        support_boundary_count=counts[SelectiveDependenceResponseExchangeExpectation.SUPPORT_BOUNDARY],
        target_sink_conflict_case_ids=tuple(sorted(target_sink_conflict_case_ids)),
        policy_dispositions=dispositions,
        hold_viability_states=hold_states,
        mixed_hold_viability_present=hold_states == (False, True),
        target_has_two_policy_dispositions=len(dispositions) >= 2,
        qualified=not ordered_stops,
        stop_codes=ordered_stops,
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseDevelopmentLineage(CanonicalRecord):
    """Receipt- and implementation-bound lineage visible to development analysis."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-development-lineage'

    lineage_id: str
    target_id: str
    source_qualification: ObjectIdentity
    source_completion: ObjectIdentity
    method_completion: ObjectIdentity
    implementation_sha256: str
    dependency_receipt_ids: tuple[str, ...]
    dependency_materialization_ids: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("lineage_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        for name in ("dependency_receipt_ids", "dependency_materialization_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("development lineage must remain development visible")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseDevelopmentParent(CanonicalRecord):
    """Sole immutable science parent of one conditional evaluation package."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-development-parent'

    parent_id: str
    target_id: str
    method_question: ObjectIdentity
    contamination_ledger: ObjectIdentity
    preparation_freeze: ObjectIdentity
    analysis_freeze: ObjectIdentity
    design: ObjectIdentity
    construct_review: ObjectIdentity
    selected_role_map: ObjectIdentity
    role_map_selection: ObjectIdentity
    source_qualification: ObjectIdentity
    source_completion: ObjectIdentity
    method_completion: ObjectIdentity
    implementation_sha256: str
    development_receipt_ids: tuple[str, ...]
    development_materialization_ids: tuple[str, ...]
    development_panel: ObjectIdentity
    finite_law: ObjectIdentity
    selective_signature: ObjectIdentity
    comparator_encodings: tuple[ObjectIdentity, ...]
    power_qualification: ObjectIdentity
    separation_report: ObjectIdentity
    challenge_qualification: ObjectIdentity
    denominator_selection: ObjectIdentity
    target_relation_id: str
    evaluation_complete_unit_ids: tuple[str, ...]
    evaluation_complete_unit_ids_sha256: str
    reserve_complete_unit_ids: tuple[str, ...]
    reserve_complete_unit_ids_sha256: str
    eligible_for_evaluation: bool
    stop_codes: tuple[str, ...]
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("parent_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.target_relation_id != SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID:
            raise ValueError("development parent target relation differs")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        for name in ("development_receipt_ids", "development_materialization_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        require_sorted_unique_ids(
            self.comparator_encodings,
            attribute="object_id",
            field_name="comparator_encodings",
        )
        if len(self.comparator_encodings) != 10:
            raise ValueError("development parent requires the exact ten comparators")
        require_sorted_unique_strings(
            self.evaluation_complete_unit_ids,
            field_name="evaluation_complete_unit_ids",
            allow_empty=False,
        )
        validate_sha256(
            self.evaluation_complete_unit_ids_sha256,
            field_name="evaluation_complete_unit_ids_sha256",
        )
        if self.evaluation_complete_unit_ids_sha256 != digest_ids(
            self.evaluation_complete_unit_ids
        ):
            raise ValueError("development-parent evaluation roster digest differs")
        require_sorted_unique_strings(
            self.reserve_complete_unit_ids,
            field_name="reserve_complete_unit_ids",
        )
        validate_sha256(
            self.reserve_complete_unit_ids_sha256,
            field_name="reserve_complete_unit_ids_sha256",
        )
        if self.reserve_complete_unit_ids_sha256 != digest_ids(self.reserve_complete_unit_ids):
            raise ValueError("development-parent reserve roster digest differs")
        if set(self.evaluation_complete_unit_ids) & set(self.reserve_complete_unit_ids):
            raise ValueError("evaluation and reserve complete-unit rosters overlap")
        require_sorted_unique_strings(self.stop_codes, field_name="stop_codes")
        if self.eligible_for_evaluation == bool(self.stop_codes):
            raise ValueError("development eligibility and typed stops are inconsistent")
        if self.evaluation_outcome_count:
            raise ValueError("development parent cannot access evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("development parent must remain development visible")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseDevelopmentBundle(CanonicalRecord):
    """One deterministic analysis output with an optional eligible forecast."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-development-bundle'

    bundle_id: str
    target_id: str
    parent: SelectiveDependenceResponseDevelopmentParent
    law: SelectiveDependenceResponseFiniteLawCalibration
    signature: SelectiveDependenceResponseSelectiveDependenceSignature
    comparators: tuple[SelectiveDependenceResponseComparatorEncoding, ...]
    power: SelectiveDependenceResponsePowerDesignQualification
    separation: SelectiveDependenceResponseForecastSeparationReport
    challenge: SelectiveDependenceResponseForecastChallengeQualification
    denominator: SelectiveDependenceResponseDenominatorSelection
    forecast: SelectiveDependenceResponseSelectiveLawForecast | None
    evaluation_eligible: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("bundle_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.comparators, attribute="encoding_id", field_name="comparators"
        )
        if len(self.comparators) != 10:
            raise ValueError("development bundle lacks the exact comparator family")
        fixed_target_ids = (
            self.parent.target_id,
            self.law.target_id,
            self.signature.target_id,
            self.power.target_id,
            self.separation.target_id,
            self.challenge.target_id,
            self.denominator.target_id,
        )
        if any(value != self.target_id for value in fixed_target_ids) or any(
            value.target_id != self.target_id for value in self.comparators
        ):
            raise ValueError("development bundle crosses targets")
        expected_eligible = self.parent.eligible_for_evaluation and self.forecast is not None
        if self.evaluation_eligible != expected_eligible:
            raise ValueError("development bundle eligibility is not parent-derived")
        if self.forecast is not None and self.forecast.target_id != self.target_id:
            raise ValueError("development bundle forecast crosses targets")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("development bundle must remain development visible")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseStudyForecastQualification(CanonicalRecord):
    """Outcome-free two-target policy-diversity gate before evaluation issue."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-study-forecast-qualification'

    qualification_id: str
    target_ids: tuple[str, ...]
    target_forecasts: tuple[ObjectIdentity, ...]
    forecast_policy_dispositions: tuple[SelectiveDependenceResponseDisposition, ...]
    all_three_policy_dispositions_present: bool
    qualified: bool
    stop_codes: tuple[str, ...]
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        require_sorted_unique_strings(self.target_ids, field_name="target_ids", allow_empty=False)
        require_sorted_unique_ids(
            self.target_forecasts,
            attribute="object_id",
            field_name="target_forecasts",
        )
        if len(self.target_ids) != 2 or len(self.target_forecasts) != 2:
            raise ValueError("programme forecast qualification requires two targets")
        if (
            tuple(sorted(set(self.forecast_policy_dispositions), key=lambda value: value.value))
            != self.forecast_policy_dispositions
        ):
            raise ValueError("programme forecast dispositions are not sorted and unique")
        expected_three = {
            SelectiveDependenceResponseDisposition.ACTION_AVAILABLE,
            SelectiveDependenceResponseDisposition.HOLD_ONLY,
            SelectiveDependenceResponseDisposition.NONATTEMPT,
        }.issubset(self.forecast_policy_dispositions)
        if self.all_three_policy_dispositions_present != expected_three:
            raise ValueError("programme forecast policy-diversity flag is not derived")
        require_sorted_unique_strings(self.stop_codes, field_name="stop_codes")
        expected_stops = (
            () if expected_three else ("programme-policy-disposition-roster-incomplete",)
        )
        if self.stop_codes != expected_stops or self.qualified != expected_three:
            raise ValueError("programme forecast qualification differs from its roster")
        if self.evaluation_outcome_count:
            raise ValueError("programme forecast qualification cannot inspect evaluation")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("programme forecast qualification must remain development visible")


def qualify_study_forecasts(
    bundles: tuple[SelectiveDependenceResponseDevelopmentBundle, SelectiveDependenceResponseDevelopmentBundle],
) -> SelectiveDependenceResponseStudyForecastQualification:
    ordered = tuple(sorted(bundles, key=lambda value: value.target_id))
    if len({value.target_id for value in ordered}) != 2:
        raise ValueError("programme forecast targets repeat")
    if any(not value.evaluation_eligible or value.forecast is None for value in ordered):
        raise ValueError("programme forecast gate requires two target-eligible forecasts")
    forecasts = tuple(value.forecast for value in ordered)
    assert all(value is not None for value in forecasts)
    typed_forecasts = tuple(value for value in forecasts if value is not None)
    dispositions = tuple(
        sorted(
            {
                decision.disposition
                for forecast in typed_forecasts
                for decision in forecast.context_forecasts
            },
            key=lambda value: value.value,
        )
    )
    expected = {
        SelectiveDependenceResponseDisposition.ACTION_AVAILABLE,
        SelectiveDependenceResponseDisposition.HOLD_ONLY,
        SelectiveDependenceResponseDisposition.NONATTEMPT,
    }
    qualified = expected.issubset(dispositions)
    return SelectiveDependenceResponseStudyForecastQualification(
        qualification_id="qualification.selective-dependence-response.programme-forecast",
        target_ids=tuple(value.target_id for value in ordered),
        target_forecasts=tuple(
            sorted(
                (ObjectIdentity.from_record(value.forecast_id, value) for value in typed_forecasts),
                key=lambda value: value.object_id,
            )
        ),
        forecast_policy_dispositions=dispositions,
        all_three_policy_dispositions_present=qualified,
        qualified=qualified,
        stop_codes=() if qualified else ("programme-policy-disposition-roster-incomplete",),
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


def freeze_development_parent(
    *,
    target_id: str,
    design: CanonicalRecord,
    design_id: str,
    method_question: CanonicalRecord,
    method_question_id: str,
    contamination_ledger: CanonicalRecord,
    contamination_ledger_id: str,
    preparation_freeze: CanonicalRecord,
    preparation_freeze_id: str,
    analysis_freeze: CanonicalRecord,
    analysis_freeze_id: str,
    construct_review: CanonicalRecord,
    construct_review_id: str,
    selected_role_map: ObjectIdentity,
    role_map_selection: ObjectIdentity,
    development_panel: CanonicalRecord,
    development_panel_id: str,
    lineage: SelectiveDependenceResponseDevelopmentLineage,
    law: SelectiveDependenceResponseFiniteLawCalibration,
    signature: SelectiveDependenceResponseSelectiveDependenceSignature,
    comparators: tuple[SelectiveDependenceResponseComparatorEncoding, ...],
    power: SelectiveDependenceResponsePowerDesignQualification,
    separation: SelectiveDependenceResponseForecastSeparationReport,
    challenge: SelectiveDependenceResponseForecastChallengeQualification,
    denominator: SelectiveDependenceResponseDenominatorSelection,
    evaluation_complete_unit_ids: tuple[str, ...],
    reserve_complete_unit_ids: tuple[str, ...],
) -> SelectiveDependenceResponseDevelopmentParent:
    stop_codes = []
    if not law.supported:
        stop_codes.append("local-law-unqualified")
    if not signature.all_active_supported or not signature.all_invariant_supported:
        stop_codes.append("selective-dependence-unresolved")
    if not signature.support_boundary_refused:
        stop_codes.append("support-boundary-opposed")
    if not power.attainable:
        stop_codes.append("power-or-resolution-unattainable")
    if not separation.all_claim_relevant_comparators_separated:
        stop_codes.append("no-predictively-distinguishing-challenge")
    if not challenge.qualified:
        stop_codes.extend(challenge.stop_codes)
    if not denominator.resolved:
        stop_codes.extend(denominator.stop_codes)
    elif denominator.selected_alternative is not SelectiveDependenceResponseDenominatorAlternative.PROPOSED:
        stop_codes.append("proposed-denominator-not-minimal-adequate")
    expected_count = power.selected_evaluation_unit_count
    if expected_count is None or len(evaluation_complete_unit_ids) != expected_count:
        stop_codes.append("evaluation-roster-power-mismatch")
    ordered_stops = tuple(sorted(set(stop_codes)))
    return SelectiveDependenceResponseDevelopmentParent(
        parent_id=f"parent.{target_id}.development",
        target_id=target_id,
        method_question=ObjectIdentity.from_record(method_question_id, method_question),
        contamination_ledger=ObjectIdentity.from_record(
            contamination_ledger_id, contamination_ledger
        ),
        preparation_freeze=ObjectIdentity.from_record(preparation_freeze_id, preparation_freeze),
        analysis_freeze=ObjectIdentity.from_record(analysis_freeze_id, analysis_freeze),
        design=ObjectIdentity.from_record(design_id, design),
        construct_review=ObjectIdentity.from_record(construct_review_id, construct_review),
        selected_role_map=selected_role_map,
        role_map_selection=role_map_selection,
        source_qualification=lineage.source_qualification,
        source_completion=lineage.source_completion,
        method_completion=lineage.method_completion,
        implementation_sha256=lineage.implementation_sha256,
        development_receipt_ids=lineage.dependency_receipt_ids,
        development_materialization_ids=lineage.dependency_materialization_ids,
        development_panel=ObjectIdentity.from_record(development_panel_id, development_panel),
        finite_law=ObjectIdentity.from_record(law.calibration_id, law),
        selective_signature=ObjectIdentity.from_record(signature.signature_id, signature),
        comparator_encodings=tuple(
            sorted(
                (ObjectIdentity.from_record(value.encoding_id, value) for value in comparators),
                key=lambda value: value.object_id,
            )
        ),
        power_qualification=ObjectIdentity.from_record(power.qualification_id, power),
        separation_report=ObjectIdentity.from_record(separation.report_id, separation),
        challenge_qualification=ObjectIdentity.from_record(challenge.qualification_id, challenge),
        denominator_selection=ObjectIdentity.from_record(denominator.selection_id, denominator),
        target_relation_id=SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID,
        evaluation_complete_unit_ids=evaluation_complete_unit_ids,
        evaluation_complete_unit_ids_sha256=digest_ids(evaluation_complete_unit_ids),
        reserve_complete_unit_ids=reserve_complete_unit_ids,
        reserve_complete_unit_ids_sha256=digest_ids(reserve_complete_unit_ids),
        eligible_for_evaluation=not ordered_stops,
        stop_codes=ordered_stops,
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


__all__ = [
    'SelectiveDependenceResponseDevelopmentBundle',
    'SelectiveDependenceResponseDevelopmentLineage',
    'SelectiveDependenceResponseDevelopmentParent',
    'SelectiveDependenceResponseForecastChallengeQualification',
    'SelectiveDependenceResponseStudyForecastQualification',
    "freeze_development_parent",
    'qualify_study_forecasts',
    "qualify_forecast_challenge",
]
