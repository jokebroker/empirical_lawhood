'Facewise prospective structural recurrence over the preserved structural recurrence kernels.'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceDevelopmentHandoff, StructuralRecurrenceDonorCompilerFreeze, StructuralRecurrencePredictionIssue, StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetDesignFreeze, StructuralRecurrenceTargetStage
from empirical_lawhood.adapters.methods.structural_recurrence import MethodQualificationVerdict, ProspectiveDisposition, PolicyBranch, PredictiveMatch, PredictiveStructuralAdjudicator, PredictiveStructuralMatcher, PredictiveStructuralRecurrenceCompiler, PredictorKind, StructuralReason, StructuralObservation, StructuralPredictionInput
from empirical_lawhood.adapters.methods.action_fiber_structural_recurrence import HoldViabilityDisposition, PolicySafetySignature, PolicyValidationDisposition, build_policy_signature
from empirical_lawhood.adapters.methods.structural_bootstrap_inputs import StructuralBootstrapInputCensus, require_structural_bootstrap_census
from empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast import ForecastMatch, MarginPolicySignature, PanelAdmissionForecast, MarginStructuralRecurrenceForecastTargetMatch, build_margin_policy, forecast_panel_admission, forecast_policy_decision, forecast_validation_disposition, require_margin_bootstrap_inputs
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.planning.contrasting_objectives import StructuralFaceHandoffProfile

from .interval_property_comparison import IntervalPropertyComparisonResult
from .transformed_property_comparison import TransformedPropertyComparisonMethodSpec, TransformedPropertyComparisonOperands, TransformedPropertyComparisonResult, PropertyCompatibilityAxisMap, PropertyOperandSeries, bridge_transformed_property_result_to_interval


class StructuralRecurrenceFaceApplicability(StrEnum):
    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class StructuralRecurrencePredictiveLevel(StrEnum):
    CATEGORICAL = "CATEGORICAL"
    METRIC = "METRIC"
    DYNAMICAL = "DYNAMICAL"


class ProspectiveStructuralRecurrenceDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    MIXED = "MIXED"
    UNEVALUABLE = "UNEVALUABLE"
    MISSING_TARGET = "MISSING_TARGET"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ProspectiveStructuralRecurrenceVerdict(StrEnum):
    METHOD_SUPPORTED = "METHOD_SUPPORTED"
    METHOD_OPPOSED = "METHOD_OPPOSED"
    METHOD_MIXED = "METHOD_MIXED"
    METHOD_UNEVALUABLE = "METHOD_UNEVALUABLE"
    METHOD_INCOMPLETE = "METHOD_INCOMPLETE"
    METHOD_PREREQUISITE_NONATTEMPT = "METHOD_PREREQUISITE_NONATTEMPT"


_COMPATIBILITY_COMPARATORS = ("ASSUME_HOLD_SAFE", 'CATEGORICAL_TARGET_WIDE_RESTRICTION')
PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_KEY = 'method.predictive-structural-recurrence-prospective'
PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_VERSION = "1.0.0"
PROSPECTIVE_STRUCTURAL_RECURRENCE_IMPLEMENTATION_SHA256 = sha256(
    b'empirical-lawhood/prospective-structural-recurrence:facewise-observation-qualification-with-action-fiber-margin-overlays'
).hexdigest()
PROSPECTIVE_STRUCTURAL_RECURRENCE_COMPANION_POLICY_ID = 'companion.complete-categorical-roster'


def _identity(record_id: str, value: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(record_id, value)


def _derived_id(prefix: str, values: object) -> str:
    return f"{prefix}.{sha256(canonical_json_bytes(values)).hexdigest()[:32]}"


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrenceMethodSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-structural-recurrence-method-spec'

    spec_id: str
    method_key: str
    method_version: str
    implementation_sha256: str
    structural_ontology: ObjectIdentity
    categorical_method_identity: ObjectIdentity
    margin_conformance_identity: ObjectIdentity
    predecessor_identity: ObjectIdentity
    binary_robustness_margin_min: Decimal
    target_robustness_ratio_min: Decimal
    bootstrap_replications: int
    compatibility_comparator_ids: tuple[str, ...]
    companion_policy_id: str
    grants_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        validate_stable_id(self.method_key, field_name="method_key")
        validate_semantic_version(self.method_version)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if (
            self.method_key != PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_KEY
            or self.method_version != PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_VERSION
            or self.implementation_sha256 != PROSPECTIVE_STRUCTURAL_RECURRENCE_IMPLEMENTATION_SHA256
        ):
            raise ValueError('structural recurrence prospective method spec names another implementation')
        validate_decimal(
            self.binary_robustness_margin_min,
            field_name="binary_robustness_margin_min",
        )
        validate_decimal(
            self.target_robustness_ratio_min,
            field_name="target_robustness_ratio_min",
        )
        if self.binary_robustness_margin_min < 0 or not (
            Decimal("0") <= self.target_robustness_ratio_min <= Decimal("1")
        ):
            raise ValueError('structural recurrence prospective thresholds are outside their domains')
        if self.bootstrap_replications < 512:
            raise ValueError('structural recurrence prospective bootstrap roster is below 512')
        if self.compatibility_comparator_ids != _COMPATIBILITY_COMPARATORS:
            raise ValueError('structural recurrence prospective method changes code-owned comparators')
        validate_stable_id(self.companion_policy_id, field_name="companion_policy_id")
        if self.companion_policy_id != PROSPECTIVE_STRUCTURAL_RECURRENCE_COMPANION_POLICY_ID:
            raise ValueError('structural recurrence prospective companion policy differs from the registered method')
        if self.structural_ontology.object_schema != (
            'empirical-lawhood/methods/structural-ontology-identity'
        ):
            raise ValueError('structural recurrence prospective method binds another ontology contract')
        if self.grants_authority or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError('structural recurrence prospective method spec cannot grant authority or see outcomes')


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrenceFaceSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-structural-recurrence-face-spec'

    face_id: str
    target_slot_id: str
    property_id: str
    receiver_action_face_id: str
    predictive_level: StructuralRecurrencePredictiveLevel
    face_applicability: StructuralRecurrenceFaceApplicability
    admission_applicability: StructuralRecurrenceFaceApplicability
    prospective_validation_applicability: StructuralRecurrenceFaceApplicability
    base_operand_schema_ids: tuple[str, ...]
    physical_independent_unit_ids: tuple[str, ...]
    categorical_companion_required: bool

    def __post_init__(self) -> None:
        for name in ("face_id", "target_slot_id", "property_id", "receiver_action_face_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.base_operand_schema_ids,
            field_name="base_operand_schema_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
            allow_empty=False,
        )
        if self.face_applicability is StructuralRecurrenceFaceApplicability.NOT_APPLICABLE and (
            self.admission_applicability is not StructuralRecurrenceFaceApplicability.NOT_APPLICABLE
            or self.prospective_validation_applicability is not StructuralRecurrenceFaceApplicability.NOT_APPLICABLE
        ):
            raise ValueError('inapplicable structural recurrence face cannot require overlays')
        if (
            self.prospective_validation_applicability is not StructuralRecurrenceFaceApplicability.NOT_APPLICABLE
            and self.admission_applicability is StructuralRecurrenceFaceApplicability.NOT_APPLICABLE
        ):
            raise ValueError('an applicable prospective validation face requires an applicable admission face')
        if self.categorical_companion_required != (
            self.predictive_level is StructuralRecurrencePredictiveLevel.CATEGORICAL
        ):
            raise ValueError("categorical companion applicability differs from face level")

    @property
    def face_key(self) -> tuple[str, str, str, str]:
        return (
            self.target_slot_id,
            self.property_id,
            self.receiver_action_face_id,
            self.predictive_level.value,
        )


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrencePlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-structural-recurrence-plan'

    plan_id: str
    face_profile: StructuralFaceHandoffProfile
    method_spec: ProspectiveStructuralRecurrenceMethodSpec
    faces: tuple[ProspectiveStructuralRecurrenceFaceSpec, ...]
    target_designs: tuple[StructuralRecurrenceTargetDesignFreeze, ...]
    donor_compiler_freezes: tuple[StructuralRecurrenceDonorCompilerFreeze, ...]
    decisive_falsifier_ids: tuple[str, ...]
    frozen_before_protected_target_outcomes: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        require_sorted_unique_ids(self.faces, attribute="face_id", field_name="faces")
        require_sorted_unique_ids(
            self.target_designs,
            attribute="design_id",
            field_name="target_designs",
        )
        require_sorted_unique_ids(
            self.donor_compiler_freezes,
            attribute="freeze_id",
            field_name="donor_compiler_freezes",
        )
        require_sorted_unique_strings(
            self.decisive_falsifier_ids,
            field_name="decisive_falsifier_ids",
            allow_empty=False,
        )
        if not self.faces or not self.target_designs:
            raise ValueError('structural recurrence prospective plan requires faces and targets')
        if len({value.face_key for value in self.faces}) != len(self.faces):
            raise ValueError('structural recurrence prospective plan contains duplicate face keys')
        profile_faces = {value.face_id: value for value in self.face_profile.faces}
        if set(profile_faces) != {value.face_id for value in self.faces}:
            raise ValueError('structural recurrence plan and face-profile rosters differ')
        for face in self.faces:
            profile_face = profile_faces[face.face_id]
            if face.face_key != profile_face.face_key or (
                face.face_applicability.value,
                face.admission_applicability.value,
                face.prospective_validation_applicability.value,
            ) != (
                profile_face.face_applicability,
                profile_face.admission_applicability,
                profile_face.prospective_validation_applicability,
            ):
                raise ValueError('structural recurrence face semantics differ from their frozen profile')
        targets = tuple(value.target_slot.target_slot_id for value in self.target_designs)
        if len(set(targets)) != len(targets) or set(targets) != {
            value.target_slot_id for value in self.faces
        }:
            raise ValueError('structural recurrence plan requires exactly one design per target')
        design_ids = {_identity(value.design_id, value) for value in self.target_designs}
        if (
            len(self.donor_compiler_freezes) != len(self.target_designs)
            or {value.design_freeze for value in self.donor_compiler_freezes} != design_ids
        ):
            raise ValueError('structural recurrence plan requires exactly one donor freeze per design')
        design_by_identity = {
            _identity(value.design_id, value): value for value in self.target_designs
        }
        design_by_target = {
            value.target_slot.target_slot_id: value for value in self.target_designs
        }
        for face in self.faces:
            evaluation_units = (
                design_by_target[face.target_slot_id].roster(StructuralRecurrenceTargetStage.EVALUATION).unit_ids
            )
            if face.physical_independent_unit_ids != evaluation_units:
                raise ValueError(
                    'structural recurrence face physical units differ from its frozen evaluation roster'
                )
        for freeze in self.donor_compiler_freezes:
            design = design_by_identity[freeze.design_freeze]
            if (
                freeze.donor_corpus.ontology != self.method_spec.structural_ontology
                or design.target_slot.ontology != self.method_spec.structural_ontology
                or design.compatibility.ontology != self.method_spec.structural_ontology
                or design.target_slot.target_slot_id
                not in freeze.donor_corpus.excluded_target_slot_ids
                or freeze.decision_table_sha256 != freeze.donor_corpus.decision_table_sha256
            ):
                raise ValueError('structural recurrence donor freeze differs from its ontology/target corpus')
        if not self.frozen_before_protected_target_outcomes:
            raise ValueError('structural recurrence plan must freeze before protected target outcomes')


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrencePredictionIssue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-structural-recurrence-prediction-issue'

    issue_id: str
    method_spec: ObjectIdentity
    face: ObjectIdentity
    target_design: ObjectIdentity
    development_evidence: ObjectIdentity
    normalized_prediction_input: ObjectIdentity
    categorical_issue: StructuralRecurrencePredictionIssue | None
    metric_or_dynamical_forecast_operand: ObjectIdentity | None
    published_before_evaluation: bool
    protected_outcome_access_count: int
    development_policy: PolicySafetySignature | None = None
    development_margins: MarginPolicySignature | None = None
    forecasts: tuple[PanelAdmissionForecast, ...] = ()
    predicted_evaluation_policy_branch: PolicyBranch | None = None
    predicted_evaluation_action_id: str | None = None
    predicted_prospective_validation_disposition: ProspectiveDisposition | None = None
    predicted_validation_disposition: PolicyValidationDisposition | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        if (self.categorical_issue is None) == (self.metric_or_dynamical_forecast_operand is None):
            raise ValueError('structural recurrence face issue requires exactly one predictive-level product')
        if not self.published_before_evaluation or self.protected_outcome_access_count:
            raise ValueError('structural recurrence face issue is not prospective')
        overlay_fields = (
            self.development_policy,
            self.development_margins,
            self.predicted_evaluation_policy_branch,
            self.predicted_evaluation_action_id,
            self.predicted_prospective_validation_disposition,
            self.predicted_validation_disposition,
        )
        if all(value is None for value in overlay_fields):
            if self.forecasts:
                raise ValueError('overlay-inapplicable structural recurrence issue contains forecasts')
            return
        if any(value is None for value in overlay_fields):
            raise ValueError('overlay-applicable structural recurrence issue has a partial forecast contract')
        assert self.development_policy is not None
        assert self.development_margins is not None
        assert self.predicted_evaluation_policy_branch is not None
        assert self.predicted_evaluation_action_id is not None
        assert self.predicted_prospective_validation_disposition is not None
        assert self.predicted_validation_disposition is not None
        require_sorted_unique_ids(
            self.forecasts,
            attribute="forecast_id",
            field_name="forecasts",
        )
        if self.development_margins.base_policy != self.development_policy:
            raise ValueError('structural recurrence issue margins differ from its development policy')
        action_count = len(self.development_margins.action_margins)
        if len(self.forecasts) != 2 * action_count:
            raise ValueError('structural recurrence issue requires evaluation/prospective validation forecasts for every action')
        evaluation = tuple(
            value for value in self.forecasts if value.future_stage is StructuralRecurrenceTargetStage.EVALUATION
        )
        prospective_validation = tuple(value for value in self.forecasts if value.future_stage is StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION)
        branch, action_id = forecast_policy_decision(self.development_policy, evaluation)
        prospective_validation_disposition, validation = forecast_validation_disposition(
            branch=branch,
            action_id=action_id,
            forecasts=prospective_validation,
        )
        if (
            self.predicted_evaluation_policy_branch is not branch
            or self.predicted_evaluation_action_id != action_id
            or self.predicted_prospective_validation_disposition is not prospective_validation_disposition
            or self.predicted_validation_disposition is not validation
        ):
            raise ValueError('structural recurrence issue decisions are not forecast-derived')


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceMetricDynamicalForecast(CanonicalRecord):
    """Face-specific donor forecast frozen before a fresh target issue.

    This additive product replaces the historical use of a generic normalized
    prediction-input identity for metric and dynamical faces.  It contains the
    exact source-side property series and typed compatibility map that the
    property-comparison method will later join to a target observation.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-metric-dynamical-forecast'

    forecast_id: str
    face_key: tuple[str, str, str, str]
    donor_law: ObjectIdentity
    structural_recurrence_method_spec: ObjectIdentity
    property_method_spec: TransformedPropertyComparisonMethodSpec
    property_method_config: ObjectIdentity
    target_design: ObjectIdentity
    normalized_prediction_input: ObjectIdentity
    domain_cell_ids: tuple[str, ...]
    compatibility_maps: tuple[PropertyCompatibilityAxisMap, ...]
    predicted_source_series: PropertyOperandSeries
    decisive_falsifier_ids: tuple[str, ...]
    frozen_before_fresh_target_issue: bool
    target_outcome_access_count: int
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.forecast_id, field_name="forecast_id")
        if len(self.face_key) != 4:
            raise ValueError('structural recurrence typed forecast lacks its complete face key')
        for value in self.face_key[:3]:
            validate_stable_id(value, field_name="face_key")
        if self.face_key[3] not in {
            StructuralRecurrencePredictiveLevel.METRIC.value,
            StructuralRecurrencePredictiveLevel.DYNAMICAL.value,
        }:
            raise ValueError('structural recurrence typed forecast is not metric or dynamical')
        require_sorted_unique_strings(
            self.domain_cell_ids,
            field_name="domain_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.compatibility_maps,
            attribute="value_id",
            field_name="compatibility_maps",
        )
        require_sorted_unique_strings(
            self.decisive_falsifier_ids,
            field_name="decisive_falsifier_ids",
            allow_empty=False,
        )
        if {value.value_id for value in self.compatibility_maps} != {
            value.value_id for value in self.predicted_source_series.metric_values
        }:
            raise ValueError('structural recurrence typed forecast map/predicted quantity rosters differ')
        if self.property_method_config != _identity(
            self.property_method_spec.spec_id,
            self.property_method_spec,
        ):
            raise ValueError('structural recurrence typed forecast property method/config differ')
        if (
            not self.frozen_before_fresh_target_issue
            or self.target_outcome_access_count
            or self.grants_authority
        ):
            raise ValueError('structural recurrence typed forecast crossed target issue/outcome/authority')


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrenceRosterIssue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-structural-recurrence-roster-issue'

    roster_issue_id: str
    plan: ObjectIdentity
    face_specs: tuple[ProspectiveStructuralRecurrenceFaceSpec, ...]
    face_issues: tuple[ProspectiveStructuralRecurrencePredictionIssue, ...]
    roster_commitment_sha256: str
    evaluation_outcome_access_count: int
    prospective_validation_outcome_access_count: int
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_issue_id, field_name="roster_issue_id")
        require_sorted_unique_ids(self.face_issues, attribute="issue_id", field_name="face_issues")
        require_sorted_unique_ids(
            self.face_specs,
            attribute="face_id",
            field_name="face_specs",
        )
        expected_faces = {_identity(value.face_id, value): value for value in self.face_specs}
        if len({value.face_key for value in self.face_specs}) != len(self.face_specs):
            raise ValueError('structural recurrence roster issue contains duplicate face keys')
        observed_faces = {value.face for value in self.face_issues}
        if len(observed_faces) != len(self.face_issues) or observed_faces != set(expected_faces):
            raise ValueError('structural recurrence roster issue differs from its complete face plan')
        validate_sha256(self.roster_commitment_sha256, field_name="roster_commitment_sha256")
        expected = sha256(
            canonical_json_bytes(
                tuple(_identity(value.issue_id, value) for value in self.face_issues)
            )
        ).hexdigest()
        if self.roster_commitment_sha256 != expected:
            raise ValueError('structural recurrence roster commitment differs from its face issues')
        if (
            self.evaluation_outcome_access_count
            or self.prospective_validation_outcome_access_count
            or self.grants_authority
        ):
            raise ValueError('structural recurrence roster issue contains outcome access or authority')


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceFrozenLawTransportForecasts(CanonicalRecord):
    """Exact donor-law forecast bundle authenticated by a fresh target issue."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/structural-recurrence-frozen-law-transport-forecasts'

    forecast_id: str
    source_law: ObjectIdentity
    method_spec: ObjectIdentity
    method_config: ObjectIdentity
    roster_issue: ProspectiveStructuralRecurrenceRosterIssue
    metric_dynamical_forecasts: tuple[StructuralRecurrenceMetricDynamicalForecast, ...]
    scientific_bootstrap_inputs: StructuralBootstrapInputCensus
    frozen_before_fresh_target_issue: bool
    target_outcome_access_count: int
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.forecast_id, field_name="forecast_id")
        require_structural_bootstrap_census(self.scientific_bootstrap_inputs)
        if self.source_law.object_schema != 'empirical-lawhood/kernel/response-law':
            raise ValueError('structural recurrence transport forecasts bind another donor-law schema')
        if self.method_spec.object_schema != ProspectiveStructuralRecurrenceMethodSpec.SCHEMA:
            raise ValueError('structural recurrence transport forecasts bind another method specification')
        if self.method_config.object_schema != ProspectiveStructuralRecurrencePlan.SCHEMA:
            raise ValueError('structural recurrence transport forecasts bind another method configuration')
        if self.roster_issue.plan != self.method_config:
            raise ValueError('structural recurrence transport roster differs from its method configuration')
        require_sorted_unique_ids(
            self.metric_dynamical_forecasts,
            attribute="forecast_id",
            field_name="metric_dynamical_forecasts",
        )
        if any(
            value.donor_law != self.source_law or value.structural_recurrence_method_spec != self.method_spec
            for value in self.metric_dynamical_forecasts
        ):
            raise ValueError('structural recurrence transport forecast member substitutes donor law or method')
        issued_forecasts = {
            value.metric_or_dynamical_forecast_operand
            for value in self.roster_issue.face_issues
            if value.metric_or_dynamical_forecast_operand is not None
        }
        if issued_forecasts != {
            _identity(value.forecast_id, value) for value in self.metric_dynamical_forecasts
        }:
            raise ValueError('structural recurrence transport forecast bundle differs from its issued roster')
        if (
            not self.frozen_before_fresh_target_issue
            or self.target_outcome_access_count
            or self.grants_authority
        ):
            raise ValueError('structural recurrence transport forecasts crossed target issue/outcome/authority')


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrenceTargetObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-structural-recurrence-target-observation'

    observation_id: str
    face_issue: ObjectIdentity
    target_episode_or_stage_evidence: ObjectIdentity
    normalized_observation_operand: ObjectIdentity
    categorical_observation: StructuralObservation | None
    property_comparison_result: IntervalPropertyComparisonResult | None
    physical_independent_unit_ids: tuple[str, ...]
    independent_unit_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
            allow_empty=False,
        )
        if self.independent_unit_count != len(self.physical_independent_unit_ids):
            raise ValueError('structural recurrence observation inflates or omits independent units')
        if (self.categorical_observation is None) == (self.property_comparison_result is None):
            raise ValueError('structural recurrence observation requires one predictive-level operand')


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrenceDecisionOverlay(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-structural-recurrence-decision-overlay'

    overlay_id: str
    face_issue: ObjectIdentity
    face: ObjectIdentity | None
    target_design: ObjectIdentity | None
    sealed_admission_evidence: ObjectIdentity | None
    reveal_authority: ObjectIdentity | None
    policy_signature: PolicySafetySignature | None
    margin_signature: MarginPolicySignature | None
    forecast_matches: tuple[ForecastMatch, ...]
    policy_branch: str | None
    selected_action_id: str | None
    hold_viability_disposition: str | None
    primary_reason: str | None
    evaluation_policy_exact: bool
    selected_action_exact: bool
    safety_error_count: int
    physical_independent_unit_ids: tuple[str, ...]
    action_outcome_ids: tuple[str, ...]
    independent_unit_count: int
    disposition: ProspectiveStructuralRecurrenceDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.overlay_id, field_name="overlay_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
        )
        require_sorted_unique_strings(self.action_outcome_ids, field_name="action_outcome_ids")
        require_sorted_unique_ids(
            self.forecast_matches,
            attribute="match_id",
            field_name="forecast_matches",
        )
        if self.independent_unit_count < 0 or self.safety_error_count < 0:
            raise ValueError('structural recurrence decision overlay has negative independent units')
        empty = all(
            value is None
            for value in (
                self.face,
                self.target_design,
                self.sealed_admission_evidence,
                self.reveal_authority,
                self.policy_signature,
                self.margin_signature,
                self.policy_branch,
                self.selected_action_id,
                self.hold_viability_disposition,
                self.primary_reason,
            )
        )
        if self.disposition is ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE:
            if (
                not empty
                or self.forecast_matches
                or self.evaluation_policy_exact
                or self.selected_action_exact
                or self.safety_error_count
                or self.physical_independent_unit_ids
                or self.action_outcome_ids
                or self.independent_unit_count
                or not self.reason_codes
            ):
                raise ValueError("not-applicable decision overlay fabricates evidence")
            return
        if empty or any(
            value is None
            for value in (
                self.sealed_admission_evidence,
                self.face,
                self.target_design,
                self.reveal_authority,
                self.policy_signature,
                self.margin_signature,
                self.policy_branch,
                self.selected_action_id,
                self.hold_viability_disposition,
                self.primary_reason,
            )
        ):
            raise ValueError('applicable decision overlay lacks an owning admission product')
        if self.independent_unit_count != len(self.physical_independent_unit_ids):
            raise ValueError('structural recurrence decision overlay inflates or omits independent units')
        if not self.physical_independent_unit_ids or not self.action_outcome_ids:
            raise ValueError("applicable decision overlay omits complete-unit action evidence")
        assert self.policy_branch is not None
        assert self.hold_viability_disposition is not None
        assert self.primary_reason is not None
        try:
            branch = PolicyBranch(self.policy_branch)
            hold = HoldViabilityDisposition(self.hold_viability_disposition)
            primary_reason = StructuralReason(self.primary_reason)
        except ValueError as error:
            raise ValueError('structural recurrence decision overlay has an unknown owner disposition') from error
        selected = self.selected_action_id
        if (
            (branch is PolicyBranch.EXACT_ACTION and selected in {"hold", "nonattempt"})
            or (branch is PolicyBranch.HOLD and selected != "hold")
            or (branch is PolicyBranch.NONATTEMPT and selected != "nonattempt")
        ):
            raise ValueError('structural recurrence decision overlay changes the owner-selected action')
        assert self.policy_signature is not None
        assert self.margin_signature is not None
        if (
            self.margin_signature.base_policy != self.policy_signature
            or self.policy_signature.policy_branch is not branch
            or self.policy_signature.selected_action_id != selected
            or self.policy_signature.hold_viability.disposition is not hold
            or self.policy_signature.primary_reason is not primary_reason
            or self.policy_signature.stage is not StructuralRecurrenceTargetStage.EVALUATION
            or {value.action_id for value in self.forecast_matches}
            != {value.action_fiber.action_id for value in self.margin_signature.action_margins}
        ):
            raise ValueError('structural recurrence decision overlay changes its pure policy/margin owner')
        evidence_owners = {
            value.evidence
            for value in (
                *self.policy_signature.action_fibers,
                self.policy_signature.hold_viability.hold_fiber,
            )
        }
        if evidence_owners != {self.sealed_admission_evidence}:
            raise ValueError('structural recurrence decision policy signature binds another admission corpus')
        forecast_pass = all(value.admission_exact for value in self.forecast_matches)
        recurrence_pass = all(
            (
                forecast_pass,
                self.evaluation_policy_exact,
                self.selected_action_exact,
                self.safety_error_count == 0,
            )
        )
        expected = (
            ProspectiveStructuralRecurrenceDisposition.SUPPORTED
            if branch in {PolicyBranch.EXACT_ACTION, PolicyBranch.HOLD} and recurrence_pass
            else ProspectiveStructuralRecurrenceDisposition.UNEVALUABLE
            if hold in {HoldViabilityDisposition.ABSENT, HoldViabilityDisposition.UNEVALUABLE}
            or primary_reason in {StructuralReason.UNCERTAINTY, StructuralReason.VALIDITY}
            else ProspectiveStructuralRecurrenceDisposition.OPPOSED
        )
        if self.disposition is not expected or not self.reason_codes:
            raise ValueError('structural recurrence decision disposition is not policy/margin derived')


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrenceValidationOverlay(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-structural-recurrence-validation-overlay'

    overlay_id: str
    face_issue: ObjectIdentity
    decision_overlay: ObjectIdentity | None
    prospective_validation_evidence: ObjectIdentity | None
    validation_policy_signature: PolicySafetySignature | None
    prospective_validation_issue_authority: ObjectIdentity | None
    prospective_validation_execution_authority: ObjectIdentity | None
    validation_authority: ObjectIdentity | None
    policy_branch: str | None
    validation_disposition: str | None
    prospective_validation_disposition: str | None
    validation_exact: bool
    physical_independent_unit_ids: tuple[str, ...]
    action_outcome_ids: tuple[str, ...]
    action_executed: bool
    hold_executed: bool
    nonattempt: bool
    disposition: ProspectiveStructuralRecurrenceDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.overlay_id, field_name="overlay_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
        )
        require_sorted_unique_strings(self.action_outcome_ids, field_name="action_outcome_ids")
        if sum((self.action_executed, self.hold_executed, self.nonattempt)) > 1:
            raise ValueError('structural recurrence validation action/HOLD/NONATTEMPT flags overlap')
        if self.disposition is ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE:
            if any(
                (
                    self.decision_overlay,
                    self.prospective_validation_evidence,
                    self.validation_policy_signature,
                    self.prospective_validation_issue_authority,
                    self.prospective_validation_execution_authority,
                    self.validation_authority,
                    self.policy_branch,
                    self.validation_disposition,
                    self.prospective_validation_disposition,
                    self.validation_exact,
                )
            ) or any(
                (
                    self.physical_independent_unit_ids,
                    self.action_outcome_ids,
                    self.action_executed,
                    self.hold_executed,
                    self.nonattempt,
                )
            ):
                raise ValueError("not-applicable validation overlay fabricates evidence")
            if not self.reason_codes:
                raise ValueError("not-applicable validation overlay lacks its reason")
            return
        if self.disposition is ProspectiveStructuralRecurrenceDisposition.PREREQUISITE_NONATTEMPT:
            if (
                self.prospective_validation_evidence is not None
                or self.validation_policy_signature is not None
                or self.prospective_validation_issue_authority is not None
                or self.prospective_validation_execution_authority is not None
                or self.validation_authority is not None
                or self.physical_independent_unit_ids
                or self.action_outcome_ids
                or any((self.action_executed, self.hold_executed))
                or not self.nonattempt
                or self.decision_overlay is None
                or not self.reason_codes
            ):
                raise ValueError('structural recurrence prerequisite nonattempt contains a prospective validation effect')
            return
        required = (
            self.decision_overlay,
            self.prospective_validation_evidence,
            self.validation_policy_signature,
            self.prospective_validation_issue_authority,
            self.prospective_validation_execution_authority,
            self.validation_authority,
            self.policy_branch,
            self.validation_disposition,
            self.prospective_validation_disposition,
        )
        if any(value is None for value in required):
            raise ValueError('applicable validation overlay lacks an owning prospective validation product')
        if (
            len(
                {
                    self.prospective_validation_issue_authority,
                    self.prospective_validation_execution_authority,
                    self.validation_authority,
                }
            )
            != 3
        ):
            raise ValueError('prospective validation issue, execution and validation authorities must be distinct')
        if not self.physical_independent_unit_ids or not self.action_outcome_ids:
            raise ValueError("applicable validation overlay omits action-delivery evidence")
        assert self.policy_branch is not None
        assert self.validation_disposition is not None
        assert self.prospective_validation_disposition is not None
        try:
            branch = PolicyBranch(self.policy_branch)
            validation = PolicyValidationDisposition(self.validation_disposition)
            prospective_validation = ProspectiveDisposition(self.prospective_validation_disposition)
        except ValueError as error:
            raise ValueError('structural recurrence validation overlay has an unknown owner disposition') from error
        expected_flags = (
            branch is PolicyBranch.EXACT_ACTION,
            branch is PolicyBranch.HOLD,
            branch is PolicyBranch.NONATTEMPT,
        )
        if (
            self.action_executed,
            self.hold_executed,
            self.nonattempt,
        ) != expected_flags:
            raise ValueError('structural recurrence validation action flags differ from the owner result')
        assert self.validation_policy_signature is not None
        expected_prospective_validation = (
            ProspectiveDisposition.VALIDATED
            if branch is PolicyBranch.EXACT_ACTION
            and validation is PolicyValidationDisposition.VALIDATED
            else ProspectiveDisposition.OPPOSED
            if branch is PolicyBranch.EXACT_ACTION
            else ProspectiveDisposition.CONDITION_FALSE
            if branch is PolicyBranch.HOLD
            else ProspectiveDisposition.NONATTEMPT
        )
        if (
            self.validation_policy_signature.stage is not StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION
            or prospective_validation is not expected_prospective_validation
        ):
            raise ValueError('structural recurrence validation overlay changes the pure validation owner')
        evidence_owners = {
            value.evidence
            for value in (
                *self.validation_policy_signature.action_fibers,
                self.validation_policy_signature.hold_viability.hold_fiber,
            )
        }
        if evidence_owners != {self.prospective_validation_evidence}:
            raise ValueError('structural recurrence validation policy signature binds another prospective validation corpus')
        expected_disposition = (
            ProspectiveStructuralRecurrenceDisposition.SUPPORTED
            if validation is PolicyValidationDisposition.VALIDATED and self.validation_exact
            else ProspectiveStructuralRecurrenceDisposition.OPPOSED
        )
        if self.disposition is not expected_disposition or not self.reason_codes:
            raise ValueError('structural recurrence validation disposition is not owner-result derived')


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrenceFaceTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-structural-recurrence-face-terminal'

    terminal_id: str
    face_key: tuple[str, str, str, str]
    roster_issue: ObjectIdentity
    prediction_issue: ObjectIdentity | None
    observation: ObjectIdentity | None
    decision_overlay: ObjectIdentity | None
    validation_overlay: ObjectIdentity | None
    match_identities: tuple[ObjectIdentity, ...]
    disposition: ProspectiveStructuralRecurrenceDisposition
    independent_unit_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.terminal_id, field_name="terminal_id")
        if len(self.face_key) != 4:
            raise ValueError('structural recurrence face terminal lacks its complete face key')
        require_sorted_unique_ids(
            self.match_identities,
            attribute="object_id",
            field_name="match_identities",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.independent_unit_count < 0:
            raise ValueError('structural recurrence face terminal has negative independent units')


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrenceTargetTerminal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-structural-recurrence-target-terminal'

    terminal_id: str
    target_slot_id: str
    face_terminals: tuple[ObjectIdentity, ...]
    disposition: ProspectiveStructuralRecurrenceDisposition
    independent_unit_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.terminal_id, field_name="terminal_id")
        validate_stable_id(self.target_slot_id, field_name="target_slot_id")
        require_sorted_unique_ids(
            self.face_terminals,
            attribute="object_id",
            field_name="face_terminals",
        )
        if not self.face_terminals or self.independent_unit_count < 0:
            raise ValueError('structural recurrence target terminal is incomplete')


@dataclass(frozen=True, slots=True)
class ProspectiveStructuralRecurrenceAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/prospective-structural-recurrence-adjudication'

    adjudication_id: str
    roster_issue: ObjectIdentity
    face_terminals: tuple[ObjectIdentity, ...]
    target_terminals: tuple[ObjectIdentity, ...]
    categorical_adjudications: tuple[ObjectIdentity, ...]
    categorical_supported_face_count: int
    metric_supported_face_count: int
    dynamical_supported_face_count: int
    missing_face_keys: tuple[str, ...]
    nonattempt_face_keys: tuple[str, ...]
    unevaluable_face_keys: tuple[str, ...]
    opposed_face_keys: tuple[str, ...]
    target_ids: tuple[str, ...]
    verdict: ProspectiveStructuralRecurrenceVerdict
    positive_claim_eligible: bool
    independent_generality_claim_eligible: bool
    no_cross_target_pooling: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(
            self.face_terminals,
            attribute="object_id",
            field_name="face_terminals",
        )
        require_sorted_unique_ids(
            self.target_terminals,
            attribute="object_id",
            field_name="target_terminals",
        )
        require_sorted_unique_ids(
            self.categorical_adjudications,
            attribute="object_id",
            field_name="categorical_adjudications",
        )
        for name in (
            "missing_face_keys",
            "nonattempt_face_keys",
            "unevaluable_face_keys",
            "opposed_face_keys",
            "target_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if self.positive_claim_eligible != (
            self.verdict is ProspectiveStructuralRecurrenceVerdict.METHOD_SUPPORTED
        ):
            raise ValueError('structural recurrence positive-claim eligibility differs from verdict')
        if self.independent_generality_claim_eligible or not self.no_cross_target_pooling:
            raise ValueError('structural recurrence prospective adjudication fabricates generality or pooling')


def _face_for_issue(
    plan: ProspectiveStructuralRecurrencePlan,
    issue: ProspectiveStructuralRecurrencePredictionIssue,
) -> ProspectiveStructuralRecurrenceFaceSpec:
    return next(value for value in plan.faces if _identity(value.face_id, value) == issue.face)


def issue_structural_recurrence_prospective_roster(
    plan: ProspectiveStructuralRecurrencePlan,
    development_evidence: tuple[StructuralRecurrenceStageEvidence, ...],
    normalized_prediction_inputs: tuple[StructuralPredictionInput, ...],
    metric_dynamical_forecasts: tuple[StructuralRecurrenceMetricDynamicalForecast, ...] = (),
    *, scientific_bootstrap_inputs: StructuralBootstrapInputCensus,
) -> ProspectiveStructuralRecurrenceRosterIssue:
    evidence_by_target = {value.target_slot_id: value for value in development_evidence}
    input_by_target = {
        value.target_slot.target_slot_id: value for value in normalized_prediction_inputs
    }
    designs = {value.target_slot.target_slot_id: value for value in plan.target_designs}
    freezes = {value.design_freeze: value for value in plan.donor_compiler_freezes}
    forecasts_by_face = {value.face_key: value for value in metric_dynamical_forecasts}
    noncategorical_face_keys = {
        value.face_key
        for value in plan.faces
        if value.predictive_level is not StructuralRecurrencePredictiveLevel.CATEGORICAL
    }
    if set(forecasts_by_face) != noncategorical_face_keys:
        raise ValueError("noncategorical structural prediction requires its complete exact typed forecast census")
    if (
        len(evidence_by_target) != len(development_evidence)
        or len(input_by_target) != len(normalized_prediction_inputs)
        or set(evidence_by_target) != set(designs)
        or set(input_by_target) != set(designs)
        or len(forecasts_by_face) != len(metric_dynamical_forecasts)
    ):
        raise ValueError('structural recurrence prospective issue lacks one development input per target')
    census = require_structural_bootstrap_census(scientific_bootstrap_inputs)
    applicable_targets = {
        face.target_slot_id for face in plan.faces
        if face.admission_applicability is not StructuralRecurrenceFaceApplicability.NOT_APPLICABLE
    }
    if {row.current_target_id for row in census.inputs} != applicable_targets:
        raise ValueError("structural roster requires exactly its applicable target bootstrap census")
    seed_inputs_by_target = {}
    for target_id in sorted(applicable_targets):
        current_design = designs[target_id]
        target_census = StructuralBootstrapInputCensus(
            census_id=census.census_id,
            inputs=tuple(row for row in census.inputs if row.current_target_id == target_id),
        )
        seed_inputs_by_target[target_id] = require_margin_bootstrap_inputs(
            scientific_bootstrap_inputs=target_census, design=current_design,
            development=evidence_by_target[target_id],
            evaluation_count=len(current_design.roster(StructuralRecurrenceTargetStage.EVALUATION).unit_ids),
            prospective_validation_count=len(current_design.roster(StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION).unit_ids),
            bootstrap_replications=plan.method_spec.bootstrap_replications,
        )
    compiler = PredictiveStructuralRecurrenceCompiler()
    issues: list[ProspectiveStructuralRecurrencePredictionIssue] = []
    for face in plan.faces:
        design = designs[face.target_slot_id]
        evidence = evidence_by_target[face.target_slot_id]
        prediction_input = input_by_target[face.target_slot_id]
        if (
            evidence.stage is not StructuralRecurrenceTargetStage.DEVELOPMENT
            or evidence.target_design != _identity(design.design_id, design)
            or prediction_input.target_slot.target_slot_id != face.target_slot_id
            or prediction_input.outcome_access
            not in {OutcomeAccess.OUTCOME_BLIND, OutcomeAccess.DEVELOPMENT_VISIBLE}
        ):
            raise ValueError('structural recurrence prospective issue input crosses its target or cutoff')
        freeze = freezes[_identity(design.design_id, design)]
        development_roster = design.roster(StructuralRecurrenceTargetStage.DEVELOPMENT)
        if (
            prediction_input.donor_corpus != freeze.donor_corpus
            or _identity(prediction_input.ontology.ontology_id, prediction_input.ontology)
            != freeze.donor_corpus.ontology
            or prediction_input.donor_corpus.decision_table_sha256 != freeze.decision_table_sha256
            or evidence.source_implementation != design.selected_source_implementation
            or tuple(value.unit_id for value in evidence.units) != development_roster.unit_ids
            or prediction_input.independent_development_unit_count
            != evidence.independent_unit_count
            or prediction_input.raw_trajectory_values_present
            or prediction_input.native_threshold_values_present
            or prediction_input.target_admission_outcomes_accessed
            or prediction_input.target_prospective_validation_outcomes_accessed
            or prediction_input.post_reveal_diagnostics_accessed
        ):
            raise ValueError('structural recurrence prospective issue changes its frozen donor/development seam')
        categorical_issue: StructuralRecurrencePredictionIssue | None = None
        forecast_operand: ObjectIdentity | None = None
        development_policy: PolicySafetySignature | None = None
        development_margins: MarginPolicySignature | None = None
        panel_forecasts: tuple[PanelAdmissionForecast, ...] = ()
        predicted_branch: PolicyBranch | None = None
        predicted_action_id: str | None = None
        predicted_prospective_validation: ProspectiveDisposition | None = None
        predicted_validation: PolicyValidationDisposition | None = None
        if face.admission_applicability is not StructuralRecurrenceFaceApplicability.NOT_APPLICABLE:
            development_policy, _, _ = build_policy_signature(
                design=design,
                evidence=evidence,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            )
            development_margins = build_margin_policy(
                design=design,
                evidence=evidence,
                base_policy=development_policy,
                binary_robustness_margin_min=(plan.method_spec.binary_robustness_margin_min),
                target_robustness_ratio_min=(plan.method_spec.target_robustness_ratio_min),
            )
            panel_forecasts = tuple(
                sorted(
                    (
                        forecast_panel_admission(
                            design=design,
                            development=evidence,
                            development_margin=margin,
                            future_stage=stage,
                            future_independent_unit_count=len(design.roster(stage).unit_ids),
                            bootstrap_replications=plan.method_spec.bootstrap_replications,
                            scientific_seed_input=seed_inputs_by_target[face.target_slot_id][(margin.action_fiber.action_id, stage)],
                        )
                        for margin in development_margins.action_margins
                        for stage in (StructuralRecurrenceTargetStage.EVALUATION, StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION)
                    ),
                    key=lambda value: value.forecast_id,
                )
            )
            predicted_branch, predicted_action_id = forecast_policy_decision(
                development_policy,
                tuple(
                    value
                    for value in panel_forecasts
                    if value.future_stage is StructuralRecurrenceTargetStage.EVALUATION
                ),
            )
            predicted_prospective_validation, predicted_validation = forecast_validation_disposition(
                branch=predicted_branch,
                action_id=predicted_action_id,
                forecasts=tuple(
                    value for value in panel_forecasts if value.future_stage is StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION
                ),
            )
        if face.predictive_level is StructuralRecurrencePredictiveLevel.CATEGORICAL:
            prediction = compiler.compile(prediction_input)
            panel = compiler.comparator_panel(prediction_input)
            candidates = tuple(
                value for value in prediction_input.action_candidates if value.admitted
            )
            selected_action = (
                min(
                    candidates, key=lambda value: (value.development_rank, value.action_id)
                ).action_id
                if candidates
                else "hold"
            )
            handoff_seed = {
                "design": _identity(design.design_id, design),
                "evidence": _identity(evidence.evidence_id, evidence),
                "input": _identity(prediction_input.input_id, prediction_input),
            }
            handoff = StructuralRecurrenceDevelopmentHandoff(
                handoff_id=_derived_id('structural-recurrence-development-handoff', handoff_seed),
                target_design=handoff_seed["design"],
                development_evidence=handoff_seed["evidence"],
                prediction_input=prediction_input,
                selected_action_id=selected_action,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            )
            categorical_issue = StructuralRecurrencePredictionIssue(
                issue_id=_derived_id(
                    'structural-recurrence-prediction-issue',
                    (
                        handoff_seed,
                        _identity(freeze.freeze_id, freeze),
                        _identity(face.face_id, face),
                    ),
                ),
                donor_compiler_freeze=_identity(freeze.freeze_id, freeze),
                target_design=_identity(design.design_id, design),
                development_handoff=_identity(handoff.handoff_id, handoff),
                prediction=prediction,
                comparators=panel,
                evaluator_implementation=design.evaluator_implementation,
                target_specific_falsifier_ids=plan.decisive_falsifier_ids,
                published_before_evaluation=True,
                protected_outcome_access_count=0,
            )
        else:
            forecast = forecasts_by_face.get(face.face_key)
            if forecast is None:
                raise ValueError("noncategorical structural prediction requires its exact typed scientific forecast")
            if (
                forecast.structural_recurrence_method_spec
                != _identity(
                    plan.method_spec.spec_id,
                    plan.method_spec,
                )
                or forecast.target_design != _identity(design.design_id, design)
                or forecast.face_key != face.face_key
                or forecast.decisive_falsifier_ids != plan.decisive_falsifier_ids
                or forecast.normalized_prediction_input
                != _identity(prediction_input.input_id, prediction_input)
                or forecast.donor_law not in freeze.donor_corpus.donor_handoffs
            ):
                raise ValueError('structural recurrence typed forecast substitutes its face/method/design')
            forecast_operand = _identity(forecast.forecast_id, forecast)
        method_identity = _identity(plan.method_spec.spec_id, plan.method_spec)
        face_identity = _identity(face.face_id, face)
        design_identity = _identity(design.design_id, design)
        evidence_identity = _identity(evidence.evidence_id, evidence)
        input_identity = _identity(prediction_input.input_id, prediction_input)
        identity_seed = {
            "method": method_identity,
            "face": face_identity,
            "design": design_identity,
            "evidence": evidence_identity,
            "input": input_identity,
            "categorical": (
                None
                if categorical_issue is None
                else _identity(categorical_issue.issue_id, categorical_issue)
            ),
            "metric_dynamical_forecast": forecast_operand,
            "development_policy": development_policy,
            "development_margins": development_margins,
            "forecasts": panel_forecasts,
            "predicted_evaluation_policy_branch": predicted_branch,
            "predicted_evaluation_action_id": predicted_action_id,
            'predicted_prospective_validation_disposition': predicted_prospective_validation,
            "predicted_validation_disposition": predicted_validation,
        }
        issues.append(
            ProspectiveStructuralRecurrencePredictionIssue(
                issue_id=_derived_id('structural-recurrence-prospective-prediction-issue', identity_seed),
                method_spec=method_identity,
                face=face_identity,
                target_design=design_identity,
                development_evidence=evidence_identity,
                normalized_prediction_input=input_identity,
                categorical_issue=categorical_issue,
                metric_or_dynamical_forecast_operand=forecast_operand,
                published_before_evaluation=True,
                protected_outcome_access_count=0,
                development_policy=development_policy,
                development_margins=development_margins,
                forecasts=panel_forecasts,
                predicted_evaluation_policy_branch=predicted_branch,
                predicted_evaluation_action_id=predicted_action_id,
                predicted_prospective_validation_disposition=predicted_prospective_validation,
                predicted_validation_disposition=predicted_validation,
            )
        )
    ordered = tuple(sorted(issues, key=lambda value: value.issue_id))
    commitment = sha256(
        canonical_json_bytes(tuple(_identity(value.issue_id, value) for value in ordered))
    ).hexdigest()
    roster_parents = (ObjectIdentity.from_record(plan.plan_id, plan), commitment)
    return ProspectiveStructuralRecurrenceRosterIssue(
        roster_issue_id=_derived_id('structural-recurrence-prospective-roster-issue', roster_parents),
        plan=roster_parents[0],
        face_specs=plan.faces,
        face_issues=ordered,
        roster_commitment_sha256=commitment,
        evaluation_outcome_access_count=0,
        prospective_validation_outcome_access_count=0,
        grants_authority=False,
    )


def observe_structural_recurrence_prospective_face(
    issue: ProspectiveStructuralRecurrencePredictionIssue,
    target_evidence: StructuralRecurrenceStageEvidence,
    normalized_observation: StructuralObservation | IntervalPropertyComparisonResult,
) -> ProspectiveStructuralRecurrenceTargetObservation:
    if target_evidence.stage is not StructuralRecurrenceTargetStage.EVALUATION:
        raise ValueError('structural recurrence prospective observation requires protected evaluation evidence')
    if target_evidence.target_design != issue.target_design:
        raise ValueError('structural recurrence prospective observation substitutes its target design')
    is_categorical = isinstance(normalized_observation, StructuralObservation)
    if is_categorical != (issue.categorical_issue is not None):
        raise ValueError('structural recurrence prospective observation uses another predictive level')
    if isinstance(normalized_observation, StructuralObservation):
        if (
            normalized_observation.target_slot_id != target_evidence.target_slot_id
            or normalized_observation.independent_unit_count
            != target_evidence.independent_unit_count
        ):
            raise ValueError('structural recurrence categorical observation crosses target or unit roster')
    elif normalized_observation.face_key[0] != target_evidence.target_slot_id:
        raise ValueError('structural recurrence property result crosses its target member')
    operand_id = (
        normalized_observation.observation_id
        if isinstance(normalized_observation, StructuralObservation)
        else normalized_observation.result_id
    )
    units = tuple(value.unit_id for value in target_evidence.units)
    parents = {
        "issue": _identity(issue.issue_id, issue),
        "evidence": _identity(target_evidence.evidence_id, target_evidence),
        "operand": _identity(operand_id, normalized_observation),
    }
    return ProspectiveStructuralRecurrenceTargetObservation(
        observation_id=_derived_id('structural-recurrence-prospective-target-observation', parents),
        face_issue=parents["issue"],
        target_episode_or_stage_evidence=parents["evidence"],
        normalized_observation_operand=parents["operand"],
        categorical_observation=(
            normalized_observation
            if isinstance(normalized_observation, StructuralObservation)
            else None
        ),
        property_comparison_result=(
            normalized_observation
            if isinstance(normalized_observation, IntervalPropertyComparisonResult)
            else None
        ),
        physical_independent_unit_ids=units,
        independent_unit_count=len(units),
    )


def observe_structural_recurrence_typed_property_face(
    *,
    issue: ProspectiveStructuralRecurrencePredictionIssue,
    forecast: StructuralRecurrenceMetricDynamicalForecast,
    target_evidence: StructuralRecurrenceStageEvidence,
    operands: TransformedPropertyComparisonOperands,
    result: TransformedPropertyComparisonResult,
) -> ProspectiveStructuralRecurrenceTargetObservation:
    'Authenticate a typed forecast/observation/result join for frozen categorical structural recurrence.'

    forecast_identity = _identity(forecast.forecast_id, forecast)
    if issue.metric_or_dynamical_forecast_operand != forecast_identity:
        raise ValueError('structural recurrence typed observation substitutes its frozen forecast')
    if (
        operands.face_key != forecast.face_key
        or result.face_key != forecast.face_key
        or operands.source_law_or_property != forecast_identity
        or operands.domain_cell_ids != forecast.domain_cell_ids
        or operands.compatibility_maps != forecast.compatibility_maps
        or operands.source != forecast.predicted_source_series
        or operands.target_law_or_observation
        != _identity(target_evidence.evidence_id, target_evidence)
        or result.method_spec
        != _identity(forecast.property_method_spec.spec_id, forecast.property_method_spec)
        or result.operands != _identity(operands.operands_id, operands)
    ):
        raise ValueError('structural recurrence typed observation changes forecast/property operands')
    bridged = bridge_transformed_property_result_to_interval(
        result=result,
        operands=operands,
    )
    return observe_structural_recurrence_prospective_face(issue, target_evidence, bridged)


def _evaluation_forecast_matches(
    issue: ProspectiveStructuralRecurrencePredictionIssue,
    observed_margins: MarginPolicySignature,
) -> tuple[ForecastMatch, ...]:
    if issue.development_margins is None:
        raise ValueError('structural recurrence overlay issue lacks its prospective development margins')
    evaluation_forecasts = {
        value.action_id: value
        for value in issue.forecasts
        if value.future_stage is StructuralRecurrenceTargetStage.EVALUATION
    }
    predicted_bands = {
        value.action_fiber.action_id: value.band
        for value in issue.development_margins.action_margins
    }
    observed = {value.action_fiber.action_id: value for value in observed_margins.action_margins}
    if set(evaluation_forecasts) != set(predicted_bands) or set(observed) != set(predicted_bands):
        raise ValueError('structural recurrence prospective margin match changes the native action chart')
    return tuple(
        sorted(
            (
                ForecastMatch(
                    match_id=_derived_id(
                        'structural-recurrence-prospective-forecast-match',
                        (_identity(issue.issue_id, issue), action_id),
                    ),
                    action_id=action_id,
                    predicted_probability=forecast.admission_probability,
                    predicted_admitted=forecast.predicted_admitted,
                    observed_admitted=observed[action_id].action_fiber.admitted,
                    brier_score=(
                        forecast.admission_probability
                        - Decimal(1 if observed[action_id].action_fiber.admitted else 0)
                    )
                    ** 2,
                    binary_only_comparator_probability=(
                        forecast.binary_only_comparator_probability
                    ),
                    binary_only_comparator_brier=(
                        forecast.binary_only_comparator_probability
                        - Decimal(1 if observed[action_id].action_fiber.admitted else 0)
                    )
                    ** 2,
                    margin_band_predicted=predicted_bands[action_id],
                    margin_band_observed=observed[action_id].band,
                    admission_exact=(
                        forecast.predicted_admitted == observed[action_id].action_fiber.admitted
                    ),
                    band_exact=(predicted_bands[action_id] is observed[action_id].band),
                )
                for action_id, forecast in evaluation_forecasts.items()
            ),
            key=lambda value: value.match_id,
        )
    )


def _decision_safety_error_count(policy: PolicySafetySignature) -> int:
    selected = policy.selected_fiber
    false_action = int(
        policy.policy_branch is PolicyBranch.EXACT_ACTION
        and (selected is None or not selected.admitted)
    )
    hold_error = int(
        policy.policy_branch is PolicyBranch.HOLD and not policy.hold_viability.hold_fiber.admitted
    )
    return false_action + hold_error


def evaluate_structural_recurrence_prospective_decision_overlay(
    issue: ProspectiveStructuralRecurrencePredictionIssue,
    evaluation: StructuralRecurrenceStageEvidence | None,
    reveal_authority: ObjectIdentity | None,
    *,
    face: ProspectiveStructuralRecurrenceFaceSpec | None = None,
    design: StructuralRecurrenceTargetDesignFreeze | None = None,
) -> ProspectiveStructuralRecurrenceDecisionOverlay:
    supplied = (evaluation, reveal_authority, face, design)
    if evaluation is None and reveal_authority is None and design is None:
        if face is not None and face.admission_applicability is not (
            StructuralRecurrenceFaceApplicability.NOT_APPLICABLE
        ):
            raise ValueError('applicable structural recurrence decision overlay lacks its owning admission inputs')
        if issue.development_policy is not None or issue.forecasts:
            raise ValueError('forecast-bearing structural recurrence issue cannot erase its decision overlay')
        disposition = ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
        reasons: tuple[str, ...] = ("ADMISSION_OVERLAY_NOT_APPLICABLE",)
        evidence_identity = None
        face_identity = None
        target_design = None
        policy = None
        margins = None
        forecast_matches: tuple[ForecastMatch, ...] = ()
        policy_branch = None
        selected_action_id = None
        hold_disposition = None
        primary_reason = None
        policy_exact = False
        action_exact = False
        safety_errors = 0
        unit_ids: tuple[str, ...] = ()
        outcome_ids: tuple[str, ...] = ()
    elif any(value is None for value in supplied):
        raise ValueError(
            'applicable structural recurrence decision overlay lacks evidence, face, design or authority'
        )
    else:
        assert evaluation is not None
        assert reveal_authority is not None
        assert face is not None
        assert design is not None
        if (
            issue.development_policy is None
            or issue.development_margins is None
            or issue.predicted_evaluation_policy_branch is None
            or issue.predicted_evaluation_action_id is None
        ):
            raise ValueError('applicable structural recurrence decision was not prospectively forecast')
        face_identity = _identity(face.face_id, face)
        target_design = _identity(design.design_id, design)
        evidence_identity = _identity(evaluation.evidence_id, evaluation)
        if face.admission_applicability is StructuralRecurrenceFaceApplicability.NOT_APPLICABLE:
            raise ValueError('not-applicable structural recurrence decision received admission evidence')
        if issue.face != face_identity or issue.target_design != target_design:
            raise ValueError('structural recurrence decision overlay substitutes its frozen face/design')
        if (
            evaluation.stage is not StructuralRecurrenceTargetStage.EVALUATION
            or evaluation.target_design != target_design
            or evaluation.target_slot_id != face.target_slot_id
            or evaluation.source_implementation != design.selected_source_implementation
            or tuple(value.unit_id for value in evaluation.units)
            != design.roster(StructuralRecurrenceTargetStage.EVALUATION).unit_ids
            or face.physical_independent_unit_ids
            != design.roster(StructuralRecurrenceTargetStage.EVALUATION).unit_ids
        ):
            raise ValueError('structural recurrence decision overlay changes its frozen admission corpus')
        ranks = {
            value.action_id: value.development_rank
            for value in issue.development_policy.action_fibers
        }
        policy, _, _ = build_policy_signature(
            design=design,
            evidence=evaluation,
            development_ranks=ranks,
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        )
        template_margin = issue.development_margins.action_margins[0]
        margins = build_margin_policy(
            design=design,
            evidence=evaluation,
            base_policy=policy,
            binary_robustness_margin_min=(template_margin.binary_robustness_margin_min),
            target_robustness_ratio_min=(template_margin.target_robustness_ratio_min),
        )
        forecast_matches = _evaluation_forecast_matches(issue, margins)
        branch = policy.policy_branch
        hold = policy.hold_viability.disposition
        primary = policy.primary_reason
        policy_exact = issue.predicted_evaluation_policy_branch is branch
        action_exact = issue.predicted_evaluation_action_id == policy.selected_action_id
        safety_errors = _decision_safety_error_count(policy)
        recurrence_pass = all(
            (
                all(value.admission_exact for value in forecast_matches),
                policy_exact,
                action_exact,
                safety_errors == 0,
            )
        )
        disposition = (
            ProspectiveStructuralRecurrenceDisposition.SUPPORTED
            if branch in {PolicyBranch.EXACT_ACTION, PolicyBranch.HOLD} and recurrence_pass
            else ProspectiveStructuralRecurrenceDisposition.UNEVALUABLE
            if hold in {HoldViabilityDisposition.ABSENT, HoldViabilityDisposition.UNEVALUABLE}
            or primary in {StructuralReason.UNCERTAINTY, StructuralReason.VALIDITY}
            else ProspectiveStructuralRecurrenceDisposition.OPPOSED
        )
        reason_set: set[str] = set()
        if not all(value.admission_exact for value in forecast_matches):
            reason_set.add("ACTION_MARGIN_FORECAST_MISMATCH")
        if not policy_exact:
            reason_set.add("POLICY_BRANCH_MISMATCH")
        if not action_exact:
            reason_set.add("ACTION_IDENTITY_MISMATCH")
        if safety_errors:
            reason_set.add("SAFETY_TYPING_ERROR")
        reason_set.add(
            "ADMISSION_EXACT_ACTION_ADMITTED"
            if branch is PolicyBranch.EXACT_ACTION and recurrence_pass
            else "ADMISSION_MEASURED_HOLD_SUPPORTED"
            if branch is PolicyBranch.HOLD and recurrence_pass
            else "ADMISSION_POLICY_UNEVALUABLE"
            if disposition is ProspectiveStructuralRecurrenceDisposition.UNEVALUABLE
            else "ADMISSION_POLICY_OR_FORECAST_OPPOSED"
        )
        reasons = tuple(sorted(reason_set))
        policy_branch = branch.value
        selected_action_id = policy.selected_action_id
        hold_disposition = hold.value
        primary_reason = primary.value
        unit_ids = tuple(value.unit_id for value in evaluation.units)
        outcome_ids = tuple(
            sorted(value.outcome_id for unit in evaluation.units for value in unit.outcomes)
        )
    parents = (
        _identity(issue.issue_id, issue),
        evidence_identity,
        policy,
        margins,
        reveal_authority,
    )
    return ProspectiveStructuralRecurrenceDecisionOverlay(
        overlay_id=_derived_id('structural-recurrence-prospective-decision-overlay', parents),
        face_issue=parents[0],
        face=face_identity,
        target_design=target_design,
        sealed_admission_evidence=evidence_identity,
        reveal_authority=reveal_authority,
        policy_signature=policy,
        margin_signature=margins,
        forecast_matches=forecast_matches,
        policy_branch=policy_branch,
        selected_action_id=selected_action_id,
        hold_viability_disposition=hold_disposition,
        primary_reason=primary_reason,
        evaluation_policy_exact=policy_exact,
        selected_action_exact=action_exact,
        safety_error_count=safety_errors,
        physical_independent_unit_ids=unit_ids,
        action_outcome_ids=outcome_ids,
        independent_unit_count=len(unit_ids),
        disposition=disposition,
        reason_codes=reasons,
    )


def finalize_structural_recurrence_prospective_validation_overlay(
    decision: ProspectiveStructuralRecurrenceDecisionOverlay,
    prospective_validation_evidence: StructuralRecurrenceStageEvidence | None,
    validation_authority: ObjectIdentity | None,
    *,
    issue: ProspectiveStructuralRecurrencePredictionIssue | None = None,
    face: ProspectiveStructuralRecurrenceFaceSpec | None = None,
    design: StructuralRecurrenceTargetDesignFreeze | None = None,
    prospective_validation_issue_authority: ObjectIdentity | None = None,
    prospective_validation_execution_authority: ObjectIdentity | None = None,
) -> ProspectiveStructuralRecurrenceValidationOverlay:
    effects = (
        prospective_validation_evidence,
        validation_authority,
        prospective_validation_issue_authority,
        prospective_validation_execution_authority,
    )
    if decision.disposition is ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE or (
        face is not None and face.prospective_validation_applicability is StructuralRecurrenceFaceApplicability.NOT_APPLICABLE
    ):
        if any(effects):
            raise ValueError('not-applicable structural recurrence validation received prospective validation effects')
        disposition = ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
        reasons: tuple[str, ...] = ("CONTROLLER_USE_OVERLAY_NOT_APPLICABLE",)
        decision_identity = None
        evidence_identity = None
        validation_policy = None
        policy_branch = None
        validation_disposition = None
        prospective_validation_disposition = None
        validation_exact = False
        unit_ids: tuple[str, ...] = ()
        outcome_ids: tuple[str, ...] = ()
        action_executed = False
        hold_executed = False
        nonattempt = False
    elif decision.disposition is not ProspectiveStructuralRecurrenceDisposition.SUPPORTED or not any(effects):
        if any(effects):
            raise ValueError('structural recurrence prerequisite nonattempt cannot contain a partial prospective validation effect')
        disposition = ProspectiveStructuralRecurrenceDisposition.PREREQUISITE_NONATTEMPT
        reasons = ("CONTROLLER_USE_PREREQUISITE_NONATTEMPT",)
        decision_identity = _identity(decision.overlay_id, decision)
        evidence_identity = None
        validation_policy = None
        policy_branch = decision.policy_branch
        validation_disposition = None
        prospective_validation_disposition = None
        validation_exact = False
        unit_ids = ()
        outcome_ids = ()
        action_executed = False
        hold_executed = False
        nonattempt = True
    elif any(value is None for value in (*effects, issue, face, design)):
        raise ValueError(
            'applicable structural recurrence validation lacks issue, face, design, evidence or authority'
        )
    else:
        assert prospective_validation_evidence is not None
        assert validation_authority is not None
        assert prospective_validation_issue_authority is not None
        assert prospective_validation_execution_authority is not None
        assert issue is not None
        assert face is not None
        assert design is not None
        if issue.development_policy is None:
            raise ValueError('applicable structural recurrence validation lacks its prospective policy')
        decision_identity = _identity(decision.overlay_id, decision)
        evidence_identity = _identity(prospective_validation_evidence.evidence_id, prospective_validation_evidence)
        design_identity = _identity(design.design_id, design)
        if (
            face.prospective_validation_applicability is StructuralRecurrenceFaceApplicability.NOT_APPLICABLE
            or decision.face != _identity(face.face_id, face)
            or decision.face_issue != _identity(issue.issue_id, issue)
            or decision.target_design != design_identity
            or issue.target_design != design_identity
            or prospective_validation_evidence.stage is not StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION
            or prospective_validation_evidence.target_design != design_identity
            or prospective_validation_evidence.target_slot_id != face.target_slot_id
            or prospective_validation_evidence.source_implementation != design.selected_source_implementation
            or tuple(value.unit_id for value in prospective_validation_evidence.units)
            != design.roster(StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION).unit_ids
        ):
            raise ValueError('structural recurrence validation overlay changes its admission/prospective validation lineage')
        authorities = {
            decision.reveal_authority,
            prospective_validation_issue_authority,
            prospective_validation_execution_authority,
            validation_authority,
        }
        if None in authorities or len(authorities) != 4:
            raise ValueError('admission reveal and prospective validation issue/execution/reveal authorities must be distinct')
        assert decision.policy_branch is not None
        assert decision.selected_action_id is not None
        branch = PolicyBranch(decision.policy_branch)
        if branch is PolicyBranch.NONATTEMPT:
            raise ValueError('structural recurrence NONATTEMPT cannot execute a prospective validation corpus')
        ranks = {
            value.action_id: value.development_rank
            for value in issue.development_policy.action_fibers
        }
        validation_policy, _, _ = build_policy_signature(
            design=design,
            evidence=prospective_validation_evidence,
            development_ranks=ranks,
            nonhold_action_ids=(decision.selected_action_id,)
            if branch is PolicyBranch.EXACT_ACTION
            else (),
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        )
        if branch is PolicyBranch.EXACT_ACTION:
            selected = validation_policy.selected_fiber
            viable = (
                selected is not None
                and selected.action_id == decision.selected_action_id
                and selected.admitted
                and validation_policy.hold_viability.disposition is HoldViabilityDisposition.VIABLE
            )
            owner_validation = (
                PolicyValidationDisposition.VALIDATED
                if viable
                else PolicyValidationDisposition.OPPOSED
            )
            owner_prospective_validation = ProspectiveDisposition.VALIDATED if viable else ProspectiveDisposition.OPPOSED
        else:
            viable = validation_policy.hold_viability.disposition is HoldViabilityDisposition.VIABLE
            owner_validation = (
                PolicyValidationDisposition.VALIDATED
                if viable
                else PolicyValidationDisposition.OPPOSED
            )
            owner_prospective_validation = ProspectiveDisposition.CONDITION_FALSE
        validation_exact = (
            issue.predicted_validation_disposition is owner_validation
            and issue.predicted_prospective_validation_disposition is owner_prospective_validation
        )
        disposition = (
            ProspectiveStructuralRecurrenceDisposition.SUPPORTED
            if owner_validation is PolicyValidationDisposition.VALIDATED and validation_exact
            else ProspectiveStructuralRecurrenceDisposition.OPPOSED
        )
        reason_set = {
            value
            for unit in prospective_validation_evidence.units
            for outcome in unit.outcomes
            for value in outcome.reason_codes
        }
        if not validation_exact:
            reason_set.add("POLICY_VALIDATION_MISMATCH")
        reason_set.add(
            "CONTROLLER_USE_ACTION_VALIDATED"
            if branch is PolicyBranch.EXACT_ACTION
            and disposition is ProspectiveStructuralRecurrenceDisposition.SUPPORTED
            else "CONTROLLER_USE_HOLD_VALIDATED"
            if branch is PolicyBranch.HOLD and disposition is ProspectiveStructuralRecurrenceDisposition.SUPPORTED
            else "CONTROLLER_USE_POLICY_VALIDATION_OPPOSED"
        )
        reasons = tuple(sorted(reason_set))
        policy_branch = branch.value
        validation_disposition = owner_validation.value
        prospective_validation_disposition = owner_prospective_validation.value
        unit_ids = tuple(value.unit_id for value in prospective_validation_evidence.units)
        outcome_ids = tuple(
            sorted(value.outcome_id for unit in prospective_validation_evidence.units for value in unit.outcomes)
        )
        action_executed = branch is PolicyBranch.EXACT_ACTION
        hold_executed = branch is PolicyBranch.HOLD
        nonattempt = False
    parents = (
        _identity(decision.overlay_id, decision),
        evidence_identity,
        validation_policy,
        prospective_validation_issue_authority,
        prospective_validation_execution_authority,
        validation_authority,
    )
    return ProspectiveStructuralRecurrenceValidationOverlay(
        overlay_id=_derived_id('structural-recurrence-prospective-validation-overlay', parents),
        face_issue=decision.face_issue,
        decision_overlay=decision_identity,
        prospective_validation_evidence=evidence_identity,
        validation_policy_signature=validation_policy,
        prospective_validation_issue_authority=prospective_validation_issue_authority,
        prospective_validation_execution_authority=prospective_validation_execution_authority,
        validation_authority=validation_authority,
        policy_branch=policy_branch,
        validation_disposition=validation_disposition,
        prospective_validation_disposition=prospective_validation_disposition,
        validation_exact=validation_exact,
        physical_independent_unit_ids=unit_ids,
        action_outcome_ids=outcome_ids,
        action_executed=action_executed,
        hold_executed=hold_executed,
        nonattempt=nonattempt,
        disposition=disposition,
        reason_codes=reasons,
    )


def _categorical_matches(
    issue: ProspectiveStructuralRecurrencePredictionIssue,
    observation: ProspectiveStructuralRecurrenceTargetObservation,
) -> tuple[PredictiveMatch, ...]:
    if issue.categorical_issue is None or observation.categorical_observation is None:
        return ()
    matcher = PredictiveStructuralMatcher()
    predictions = (
        issue.categorical_issue.prediction,
        *issue.categorical_issue.comparators.predictions,
    )
    return tuple(
        sorted(
            (matcher.match(value, observation.categorical_observation) for value in predictions),
            key=lambda value: value.match_id,
        )
    )


def _prospective_target_match(
    issue: ProspectiveStructuralRecurrencePredictionIssue,
    decision: ProspectiveStructuralRecurrenceDecisionOverlay,
    validation: ProspectiveStructuralRecurrenceValidationOverlay | None,
) -> MarginStructuralRecurrenceForecastTargetMatch:
    if (
        decision.disposition is ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
        or decision.margin_signature is None
        or decision.policy_signature is None
        or decision.face_issue != _identity(issue.issue_id, issue)
    ):
        raise ValueError('structural recurrence prospective target match lacks an applicable decision')
    expected_forecasts = _evaluation_forecast_matches(issue, decision.margin_signature)
    if decision.forecast_matches != expected_forecasts:
        raise ValueError('structural recurrence decision substitutes its forecast-match products')
    validation_exact = True
    target_result = _identity(decision.overlay_id, decision)
    if validation is not None and validation.disposition is not (
        ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
    ):
        if validation.face_issue != decision.face_issue or validation.decision_overlay != _identity(
            decision.overlay_id, decision
        ):
            raise ValueError('structural recurrence target match substitutes its validation overlay')
        validation_exact = validation.validation_exact
        target_result = _identity(validation.overlay_id, validation)
    reasons: set[str] = set()
    if not all(value.admission_exact for value in expected_forecasts):
        reasons.add("ACTION_MARGIN_FORECAST_MISMATCH")
    if not decision.evaluation_policy_exact:
        reasons.add("POLICY_BRANCH_MISMATCH")
    if not decision.selected_action_exact:
        reasons.add("ACTION_IDENTITY_MISMATCH")
    if not validation_exact:
        reasons.add("POLICY_VALIDATION_MISMATCH")
    if decision.safety_error_count:
        reasons.add("SAFETY_TYPING_ERROR")
    passed = all(
        (
            all(value.admission_exact for value in expected_forecasts),
            decision.evaluation_policy_exact,
            decision.selected_action_exact,
            validation_exact,
            decision.safety_error_count == 0,
        )
    )
    return MarginStructuralRecurrenceForecastTargetMatch(
        match_id=_derived_id(
            'structural-recurrence-prospective-target-match',
            (
                _identity(issue.issue_id, issue),
                _identity(decision.overlay_id, decision),
                None if validation is None else _identity(validation.overlay_id, validation),
            ),
        ),
        prediction_issue=_identity(issue.issue_id, issue),
        target_result=target_result,
        target_slot_id=decision.policy_signature.target_slot_id,
        forecast_matches=expected_forecasts,
        evaluation_policy_exact=decision.evaluation_policy_exact,
        selected_action_exact=decision.selected_action_exact,
        validation_exact=validation_exact,
        safety_error_count=decision.safety_error_count,
        exact_forecast_count=sum(value.admission_exact for value in expected_forecasts),
        exact_band_count=sum(value.band_exact for value in expected_forecasts),
        margin_model_comparator_win_count=sum(
            value.brier_score < value.binary_only_comparator_brier for value in expected_forecasts
        ),
        passed=passed,
        reason_codes=tuple(sorted(reasons)),
    )


def match_structural_recurrence_prospective_face(
    issue: ProspectiveStructuralRecurrencePredictionIssue,
    observation: ProspectiveStructuralRecurrenceTargetObservation,
    decision: ProspectiveStructuralRecurrenceDecisionOverlay | None = None,
    validation: ProspectiveStructuralRecurrenceValidationOverlay | None = None,
) -> tuple[ObjectIdentity, ...]:
    if observation.face_issue != _identity(issue.issue_id, issue):
        raise ValueError('structural recurrence match substitutes its face issue')
    for overlay in (decision, validation):
        if overlay is not None and overlay.face_issue != observation.face_issue:
            raise ValueError('structural recurrence match overlay belongs to another face')
    matches: list[ObjectIdentity] = [
        _identity(value.match_id, value) for value in _categorical_matches(issue, observation)
    ]
    if decision is not None and decision.disposition is not (
        ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
    ):
        target_match = _prospective_target_match(issue, decision, validation)
        matches.append(_identity(target_match.match_id, target_match))
    return tuple(sorted(matches, key=lambda value: value.object_id))


def finalize_structural_recurrence_prospective_face(
    roster_issue: ProspectiveStructuralRecurrenceRosterIssue,
    face_key: tuple[str, str, str, str],
    issue: ProspectiveStructuralRecurrencePredictionIssue | None = None,
    observation: ProspectiveStructuralRecurrenceTargetObservation | None = None,
    decision: ProspectiveStructuralRecurrenceDecisionOverlay | None = None,
    validation: ProspectiveStructuralRecurrenceValidationOverlay | None = None,
    matches: tuple[ObjectIdentity, ...] = (),
    prerequisite_reason_codes: tuple[str, ...] = (),
) -> ProspectiveStructuralRecurrenceFaceTerminal:
    require_sorted_unique_strings(prerequisite_reason_codes, field_name="prerequisite_reason_codes")
    try:
        face = next(value for value in roster_issue.face_specs if value.face_key == face_key)
    except StopIteration as error:
        raise ValueError('structural recurrence face terminal is outside its frozen roster') from error
    expected_issue = next(
        value for value in roster_issue.face_issues if value.face == _identity(face.face_id, face)
    )
    if issue is not None and issue != expected_issue:
        raise ValueError('structural recurrence face terminal substitutes its prediction issue')
    if observation is not None and (
        issue is None or observation.face_issue != _identity(issue.issue_id, issue)
    ):
        raise ValueError('structural recurrence face terminal substitutes its observation')
    for overlay in (decision, validation):
        if overlay is not None and (
            issue is None or overlay.face_issue != _identity(issue.issue_id, issue)
        ):
            raise ValueError('structural recurrence face terminal substitutes an overlay')
    if observation is not None:
        if observation.physical_independent_unit_ids != face.physical_independent_unit_ids:
            raise ValueError('structural recurrence face observation changes its physical-unit roster')
        result = observation.property_comparison_result
        if result is not None and result.face_key != face_key:
            raise ValueError('structural recurrence property result belongs to another face')
    if (issue is not None and issue.categorical_issue is not None) != (
        face.predictive_level is StructuralRecurrencePredictiveLevel.CATEGORICAL
    ):
        raise ValueError('structural recurrence prediction issue differs from its declared level')
    if issue is not None and observation is not None:
        expected_matches = match_structural_recurrence_prospective_face(
            issue,
            observation,
            decision,
            validation,
        )
        if matches != expected_matches:
            raise ValueError('structural recurrence face match identities were substituted')
    elif matches:
        raise ValueError('structural recurrence face without an observation contains matches')
    if face.admission_applicability is StructuralRecurrenceFaceApplicability.NOT_APPLICABLE:
        if decision is not None and decision.disposition is not (
            ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
        ):
            raise ValueError('inapplicable admission overlay received decision evidence')
    if face.prospective_validation_applicability is StructuralRecurrenceFaceApplicability.NOT_APPLICABLE:
        if validation is not None and validation.disposition is not (
            ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
        ):
            raise ValueError('inapplicable prospective validation overlay received validation evidence')

    reasons: tuple[str, ...]
    if face.face_applicability is StructuralRecurrenceFaceApplicability.NOT_APPLICABLE:
        if any((observation, decision, validation, matches, prerequisite_reason_codes)):
            raise ValueError('not-applicable structural recurrence face received target evidence')
        disposition = ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
        reasons = ("FACE_NOT_APPLICABLE_BY_ISSUED_PLAN",)
        unit_count = 0
    elif issue is None:
        disposition = ProspectiveStructuralRecurrenceDisposition.MISSING_TARGET
        reasons = ("FACE_ISSUE_MISSING",)
        unit_count = 0
    elif prerequisite_reason_codes:
        disposition = ProspectiveStructuralRecurrenceDisposition.PREREQUISITE_NONATTEMPT
        reasons = prerequisite_reason_codes
        unit_count = 0 if observation is None else observation.independent_unit_count
    elif observation is None:
        disposition = ProspectiveStructuralRecurrenceDisposition.MISSING_TARGET
        reasons = ("TARGET_OBSERVATION_MISSING",)
        unit_count = 0
    elif issue.categorical_issue is not None:
        computed = _categorical_matches(issue, observation)
        primary = next(value for value in computed if value.predictor_kind is PredictorKind.PRIMARY)
        disposition = (
            ProspectiveStructuralRecurrenceDisposition.SUPPORTED
            if primary.passed
            else ProspectiveStructuralRecurrenceDisposition.OPPOSED
        )
        reasons = ("CATEGORICAL_PRIMARY_MATCH",) if primary.passed else primary.reason_codes
        unit_count = observation.independent_unit_count
    else:
        result = observation.property_comparison_result
        if result is None or result.disposition is ScientificStatus.UNEVALUABLE:
            disposition = ProspectiveStructuralRecurrenceDisposition.UNEVALUABLE
            reasons = ("PROPERTY_METHOD_RESULT_UNEVALUABLE",)
        elif result.disposition is ScientificStatus.SUPPORTED:
            disposition = ProspectiveStructuralRecurrenceDisposition.SUPPORTED
            reasons = result.reason_codes
        elif result.disposition is ScientificStatus.NOT_SUPPORTED:
            disposition = ProspectiveStructuralRecurrenceDisposition.OPPOSED
            reasons = result.reason_codes
        else:
            disposition = ProspectiveStructuralRecurrenceDisposition.MIXED
            reasons = result.reason_codes
        unit_count = observation.independent_unit_count
    if face.face_applicability is not StructuralRecurrenceFaceApplicability.NOT_APPLICABLE:
        if (
            face.admission_applicability is StructuralRecurrenceFaceApplicability.REQUIRED
            and (
                decision is None
                or decision.disposition is ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
            )
        ) or (
            face.prospective_validation_applicability is StructuralRecurrenceFaceApplicability.REQUIRED
            and (
                validation is None
                or validation.disposition is ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
            )
        ):
            disposition = ProspectiveStructuralRecurrenceDisposition.PREREQUISITE_NONATTEMPT
            reasons = ("REQUIRED_OVERLAY_NONATTEMPT",)
    for overlay in (decision, validation):
        if overlay is None or overlay.disposition is ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE:
            continue
        if overlay.disposition in {
            ProspectiveStructuralRecurrenceDisposition.PREREQUISITE_NONATTEMPT,
            ProspectiveStructuralRecurrenceDisposition.UNEVALUABLE,
            ProspectiveStructuralRecurrenceDisposition.OPPOSED,
        }:
            disposition = overlay.disposition
            reasons = overlay.reason_codes
            break
    roster_identity = _identity(roster_issue.roster_issue_id, roster_issue)
    issue_identity = None if issue is None else _identity(issue.issue_id, issue)
    observation_identity = (
        None if observation is None else _identity(observation.observation_id, observation)
    )
    decision_identity = None if decision is None else _identity(decision.overlay_id, decision)
    validation_identity = (
        None if validation is None else _identity(validation.overlay_id, validation)
    )
    identity_seed = {
        "roster": roster_identity,
        "face_key": face_key,
        "issue": issue_identity,
        "observation": observation_identity,
        "decision": decision_identity,
        "validation": validation_identity,
        "disposition": disposition,
    }
    return ProspectiveStructuralRecurrenceFaceTerminal(
        terminal_id=_derived_id('structural-recurrence-prospective-face-terminal', identity_seed),
        face_key=face_key,
        roster_issue=roster_identity,
        prediction_issue=issue_identity,
        observation=observation_identity,
        decision_overlay=decision_identity,
        validation_overlay=validation_identity,
        match_identities=matches,
        disposition=disposition,
        independent_unit_count=unit_count,
        reason_codes=tuple(sorted(set(reasons))),
    )


def _summary_disposition(
    values: tuple[ProspectiveStructuralRecurrenceDisposition, ...],
) -> ProspectiveStructuralRecurrenceDisposition:
    active = tuple(
        value for value in values if value is not ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
    )
    if not active:
        return ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
    for value in (
        ProspectiveStructuralRecurrenceDisposition.MISSING_TARGET,
        ProspectiveStructuralRecurrenceDisposition.PREREQUISITE_NONATTEMPT,
        ProspectiveStructuralRecurrenceDisposition.UNEVALUABLE,
    ):
        if value in active:
            return value
    if all(value is ProspectiveStructuralRecurrenceDisposition.SUPPORTED for value in active):
        return ProspectiveStructuralRecurrenceDisposition.SUPPORTED
    if all(value is ProspectiveStructuralRecurrenceDisposition.OPPOSED for value in active):
        return ProspectiveStructuralRecurrenceDisposition.OPPOSED
    return ProspectiveStructuralRecurrenceDisposition.MIXED


def finalize_structural_recurrence_prospective_target(
    roster_issue: ProspectiveStructuralRecurrenceRosterIssue,
    target_slot_id: str,
    face_terminals: tuple[ProspectiveStructuralRecurrenceFaceTerminal, ...],
) -> ProspectiveStructuralRecurrenceTargetTerminal:
    validate_stable_id(target_slot_id, field_name="target_slot_id")
    expected_keys = {
        value.face_key
        for value in roster_issue.face_specs
        if value.target_slot_id == target_slot_id
    }
    observed_keys = {value.face_key for value in face_terminals}
    if (
        not expected_keys
        or len(observed_keys) != len(face_terminals)
        or observed_keys != expected_keys
    ):
        raise ValueError('structural recurrence target terminal crosses or omits its target faces')
    identities = tuple(
        sorted(
            (_identity(value.terminal_id, value) for value in face_terminals),
            key=lambda value: value.object_id,
        )
    )
    disposition = _summary_disposition(tuple(value.disposition for value in face_terminals))
    active_unit_rosters = {
        value.physical_independent_unit_ids
        for value in roster_issue.face_specs
        if value.target_slot_id == target_slot_id
        and value.face_applicability is not StructuralRecurrenceFaceApplicability.NOT_APPLICABLE
    }
    if len(active_unit_rosters) > 1:
        raise ValueError('structural recurrence target faces disagree on their physical-unit roster')
    independent_unit_count = 0 if not active_unit_rosters else len(next(iter(active_unit_rosters)))
    if any(
        value.disposition is not ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
        and value.independent_unit_count not in {0, independent_unit_count}
        for value in face_terminals
    ):
        raise ValueError('structural recurrence target terminal inflates or drops shared target units')
    parents = (_identity(roster_issue.roster_issue_id, roster_issue), target_slot_id, identities)
    return ProspectiveStructuralRecurrenceTargetTerminal(
        terminal_id=_derived_id('structural-recurrence-prospective-target-terminal', parents),
        target_slot_id=target_slot_id,
        face_terminals=identities,
        disposition=disposition,
        independent_unit_count=independent_unit_count,
    )


def _verdict(values: tuple[ProspectiveStructuralRecurrenceDisposition, ...]) -> ProspectiveStructuralRecurrenceVerdict:
    active = tuple(
        value for value in values if value is not ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
    )
    if ProspectiveStructuralRecurrenceDisposition.MISSING_TARGET in active:
        return ProspectiveStructuralRecurrenceVerdict.METHOD_INCOMPLETE
    if ProspectiveStructuralRecurrenceDisposition.PREREQUISITE_NONATTEMPT in active:
        return ProspectiveStructuralRecurrenceVerdict.METHOD_PREREQUISITE_NONATTEMPT
    if ProspectiveStructuralRecurrenceDisposition.UNEVALUABLE in active or not active:
        return ProspectiveStructuralRecurrenceVerdict.METHOD_UNEVALUABLE
    if all(value is ProspectiveStructuralRecurrenceDisposition.SUPPORTED for value in active):
        return ProspectiveStructuralRecurrenceVerdict.METHOD_SUPPORTED
    if ProspectiveStructuralRecurrenceDisposition.SUPPORTED in active:
        return ProspectiveStructuralRecurrenceVerdict.METHOD_MIXED
    return ProspectiveStructuralRecurrenceVerdict.METHOD_OPPOSED


def adjudicate_structural_recurrence_prospective_roster(
    roster_issue: ProspectiveStructuralRecurrenceRosterIssue,
    face_terminals: tuple[ProspectiveStructuralRecurrenceFaceTerminal, ...],
    terminals: tuple[ProspectiveStructuralRecurrenceTargetTerminal, ...],
    categorical_matches: tuple[PredictiveMatch, ...],
) -> ProspectiveStructuralRecurrenceAdjudication:
    if not face_terminals or not terminals:
        raise ValueError('structural recurrence prospective adjudication requires complete terminal rosters')
    expected_face_keys = {value.face_key for value in roster_issue.face_specs}
    observed_face_keys = {value.face_key for value in face_terminals}
    expected_targets = {value.target_slot_id for value in roster_issue.face_specs}
    observed_targets = {value.target_slot_id for value in terminals}
    if (
        len(observed_face_keys) != len(face_terminals)
        or observed_face_keys != expected_face_keys
        or len(observed_targets) != len(terminals)
        or observed_targets != expected_targets
    ):
        raise ValueError('structural recurrence prospective adjudication omits a frozen face or target')
    roster_identity = _identity(roster_issue.roster_issue_id, roster_issue)
    if any(value.roster_issue != roster_identity for value in face_terminals):
        raise ValueError('structural recurrence prospective adjudication substitutes a face roster')
    by_target = {value.target_slot_id: value for value in terminals}
    for target_id, terminal in by_target.items():
        expected = {
            _identity(value.terminal_id, value)
            for value in face_terminals
            if value.face_key[0] == target_id
        }
        if set(terminal.face_terminals) != expected:
            raise ValueError('structural recurrence target terminal differs from its face terminals')
    match_by_id = {value.match_id: value for value in categorical_matches}
    if len(match_by_id) != len(categorical_matches):
        raise ValueError('structural recurrence prospective adjudication contains duplicate categorical matches')
    categorical_terminals = tuple(
        value
        for value in face_terminals
        if value.face_key[3] == StructuralRecurrencePredictiveLevel.CATEGORICAL.value
        and value.disposition is not ProspectiveStructuralRecurrenceDisposition.NOT_APPLICABLE
    )
    expected_match_ids = {
        identity.object_id
        for terminal in categorical_terminals
        for identity in terminal.match_identities
    }
    if set(match_by_id) != expected_match_ids:
        raise ValueError('structural recurrence categorical companion matches differ from face terminals')
    categorical_records = []
    categorical_complete = True
    categorical_qualified = True
    for categorical_terminal in categorical_terminals:
        face_matches = tuple(
            sorted(
                (match_by_id[value.object_id] for value in categorical_terminal.match_identities),
                key=lambda value: value.match_id,
            )
        )
        if len(face_matches) != len(PredictorKind):
            categorical_complete = False
            categorical_qualified = False
            continue
        categorical = PredictiveStructuralAdjudicator().adjudicate(
            adjudication_id=_derived_id(
                'structural-recurrence-categorical-adjudication',
                tuple(_identity(value.match_id, value) for value in face_matches),
            ),
            matches=face_matches,
        )
        categorical_records.append(_identity(categorical.adjudication_id, categorical))
        categorical_qualified = categorical_qualified and (
            categorical.verdict is MethodQualificationVerdict.METHOD_QUALIFIED
        )
    dispositions = tuple(value.disposition for value in face_terminals)
    verdict = _verdict(dispositions)
    if (
        verdict is ProspectiveStructuralRecurrenceVerdict.METHOD_SUPPORTED
        and categorical_terminals
        and (not categorical_complete or not categorical_qualified)
    ):
        verdict = ProspectiveStructuralRecurrenceVerdict.METHOD_OPPOSED
    encoded_keys = {value.terminal_id: "|".join(value.face_key) for value in face_terminals}
    identities = tuple(
        sorted(
            (_identity(value.terminal_id, value) for value in face_terminals),
            key=lambda value: value.object_id,
        )
    )
    target_identities = tuple(
        sorted(
            (_identity(value.terminal_id, value) for value in terminals),
            key=lambda value: value.object_id,
        )
    )

    def keys(disposition: ProspectiveStructuralRecurrenceDisposition) -> tuple[str, ...]:
        return tuple(
            sorted(
                encoded_keys[value.terminal_id]
                for value in face_terminals
                if value.disposition is disposition
            )
        )

    levels = {value.terminal_id: value.face_key[3] for value in face_terminals}
    parents = (
        _identity(roster_issue.roster_issue_id, roster_issue),
        identities,
        target_identities,
        verdict,
    )
    return ProspectiveStructuralRecurrenceAdjudication(
        adjudication_id=_derived_id('structural-recurrence-prospective-adjudication', parents),
        roster_issue=parents[0],
        face_terminals=identities,
        target_terminals=target_identities,
        categorical_adjudications=tuple(
            sorted(categorical_records, key=lambda value: value.object_id)
        ),
        categorical_supported_face_count=sum(
            value.disposition is ProspectiveStructuralRecurrenceDisposition.SUPPORTED
            and levels[value.terminal_id] == "CATEGORICAL"
            for value in face_terminals
        ),
        metric_supported_face_count=sum(
            value.disposition is ProspectiveStructuralRecurrenceDisposition.SUPPORTED
            and levels[value.terminal_id] == "METRIC"
            for value in face_terminals
        ),
        dynamical_supported_face_count=sum(
            value.disposition is ProspectiveStructuralRecurrenceDisposition.SUPPORTED
            and levels[value.terminal_id] == "DYNAMICAL"
            for value in face_terminals
        ),
        missing_face_keys=keys(ProspectiveStructuralRecurrenceDisposition.MISSING_TARGET),
        nonattempt_face_keys=keys(ProspectiveStructuralRecurrenceDisposition.PREREQUISITE_NONATTEMPT),
        unevaluable_face_keys=keys(ProspectiveStructuralRecurrenceDisposition.UNEVALUABLE),
        opposed_face_keys=keys(ProspectiveStructuralRecurrenceDisposition.OPPOSED),
        target_ids=tuple(sorted(value.target_slot_id for value in terminals)),
        verdict=verdict,
        positive_claim_eligible=verdict is ProspectiveStructuralRecurrenceVerdict.METHOD_SUPPORTED,
        independent_generality_claim_eligible=False,
        no_cross_target_pooling=True,
    )


__all__ = [
    'PROSPECTIVE_STRUCTURAL_RECURRENCE_COMPANION_POLICY_ID',
    'PROSPECTIVE_STRUCTURAL_RECURRENCE_IMPLEMENTATION_SHA256',
    'PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_KEY',
    'PROSPECTIVE_STRUCTURAL_RECURRENCE_METHOD_VERSION',
    'StructuralRecurrenceFaceApplicability',
    'StructuralRecurrenceFrozenLawTransportForecasts',
    'StructuralRecurrenceMetricDynamicalForecast',
    'StructuralRecurrencePredictiveLevel',
    'ProspectiveStructuralRecurrenceAdjudication',
    'ProspectiveStructuralRecurrenceDecisionOverlay',
    'ProspectiveStructuralRecurrenceDisposition',
    'ProspectiveStructuralRecurrenceFaceSpec',
    'ProspectiveStructuralRecurrenceFaceTerminal',
    'ProspectiveStructuralRecurrenceMethodSpec',
    'ProspectiveStructuralRecurrencePlan',
    'ProspectiveStructuralRecurrencePredictionIssue',
    'ProspectiveStructuralRecurrenceRosterIssue',
    'ProspectiveStructuralRecurrenceTargetObservation',
    'ProspectiveStructuralRecurrenceTargetTerminal',
    'ProspectiveStructuralRecurrenceValidationOverlay',
    'ProspectiveStructuralRecurrenceVerdict',
    'adjudicate_structural_recurrence_prospective_roster',
    'evaluate_structural_recurrence_prospective_decision_overlay',
    'finalize_structural_recurrence_prospective_face',
    'finalize_structural_recurrence_prospective_target',
    'finalize_structural_recurrence_prospective_validation_overlay',
    'issue_structural_recurrence_prospective_roster',
    'match_structural_recurrence_prospective_face',
    'observe_structural_recurrence_prospective_face',
    'observe_structural_recurrence_typed_property_face',
]
