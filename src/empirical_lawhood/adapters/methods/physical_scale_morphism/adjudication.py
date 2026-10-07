"""Noncompensating physical scale morphism axis, fixed-section and overall grammar."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority

from .comparators import PhysicalScaleMorphismCoordinateDisposition
from .contracts import PhysicalScaleMorphismEvidenceWorld, PhysicalScaleMorphismForecastLevel, PhysicalScaleMorphismForecastState, PhysicalScaleMorphismHoldDisposition, PhysicalScaleMorphismMapFamily, PhysicalScaleMorphismObstructionKind
from .evaluation_evidence import PhysicalScaleMorphismAdjudicationEvidenceBundle, PhysicalScaleMorphismEvidenceState, PhysicalScaleMorphismSectionKind
from .governance import PhysicalScaleMorphismCompleteFanIn, PhysicalScaleMorphismConstructTerminal, PhysicalScaleMorphismImplementationRecurrenceState, PhysicalScaleMorphismPredictionFreeze, PhysicalScaleMorphismSourceApparatusTerminal
from .inference import PhysicalScaleMorphismPowerTerminal


class PhysicalScaleMorphismAxisTerminal(StrEnum):
    BOUNDED_PHYSICAL_MORPHISM_CLASS_SUPPORTED = "BOUNDED_PHYSICAL_MORPHISM_CLASS_SUPPORTED"
    CALIBRATION_ONLY = "CALIBRATION_ONLY"
    CATEGORICAL_ONLY_MORPHISM = "CATEGORICAL_ONLY_MORPHISM"
    COMPOSITION_OPPOSED = "COMPOSITION_OPPOSED"
    CONSTRUCT_INDEPENDENCE_NOT_QUALIFIED = "CONSTRUCT_INDEPENDENCE_NOT_QUALIFIED"
    COORDINATE_NECESSITY_IDENTIFIED = "COORDINATE_NECESSITY_IDENTIFIED"
    COORDINATE_PREDICTION_OPPOSED = "COORDINATE_PREDICTION_OPPOSED"
    DECISION_MORPHISM_OPPOSED = "DECISION_MORPHISM_OPPOSED"
    DEVELOPMENT_EFFECT_ON_WRONG_SIDE = "DEVELOPMENT_EFFECT_ON_WRONG_SIDE"
    FIBREWISE_HETEROGENEITY_MIXED = "FIBREWISE_HETEROGENEITY_MIXED"
    HOLD_FIBRE_UNQUALIFIED_OR_UNSAFE = "HOLD_FIBRE_UNQUALIFIED_OR_UNSAFE"
    INDEPENDENT_IMPLEMENTATION_RECURRENCE_NOT_TESTED = (
        "INDEPENDENT_IMPLEMENTATION_RECURRENCE_NOT_TESTED"
    )
    INDEPENDENT_IMPLEMENTATION_RECURRENCE_OPPOSED = "INDEPENDENT_IMPLEMENTATION_RECURRENCE_OPPOSED"
    INDEPENDENT_IMPLEMENTATION_RECURRENCE_SUPPORTED = (
        "INDEPENDENT_IMPLEMENTATION_RECURRENCE_SUPPORTED"
    )
    INTERVENTIONAL_MORPHISM_OPPOSED = "INTERVENTIONAL_MORPHISM_OPPOSED"
    METHOD_NOT_QUALIFIED = "METHOD_NOT_QUALIFIED"
    MORPHISM_DOMAIN_INCOMPLETE = "MORPHISM_DOMAIN_INCOMPLETE"
    NO_FINITE_COUNT_UNDER_OBSERVED_RATE = "NO_FINITE_COUNT_UNDER_OBSERVED_RATE"
    NUMERICAL_REFINEMENT_OPPOSED = "NUMERICAL_REFINEMENT_OPPOSED"
    NUMERICAL_REFINEMENT_SUPPORTED = "NUMERICAL_REFINEMENT_SUPPORTED"
    OBSERVATIONAL_ONLY = "OBSERVATIONAL_ONLY"
    PAIRED_FORECAST_TRIPLET_SUPPORTED = "PAIRED_FORECAST_TRIPLET_SUPPORTED"
    PHYSICAL_SCALE_CROSSOVER_IDENTIFIED = "PHYSICAL_SCALE_CROSSOVER_IDENTIFIED"
    PHYSICAL_SCALE_MORPHISM_OPPOSED = "PHYSICAL_SCALE_MORPHISM_OPPOSED"
    PHYSICAL_SCALE_MORPHISM_SUPPORTED = "PHYSICAL_SCALE_MORPHISM_SUPPORTED"
    PRECISION_LIMITED_AT_RESOURCE_CEILING = "PRECISION_LIMITED_AT_RESOURCE_CEILING"
    RECEIVER_MORPHISM_OPPOSED = "RECEIVER_MORPHISM_OPPOSED"
    RECEIVER_MORPHISM_SUPPORTED = "RECEIVER_MORPHISM_SUPPORTED"
    SEMANTIC_FUTURE_CLOSURE_OPPOSED = "SEMANTIC_FUTURE_CLOSURE_OPPOSED"
    SOURCE_OR_APPARATUS_NOT_QUALIFIED = "SOURCE_OR_APPARATUS_NOT_QUALIFIED"
    TASK_SATURATED_BY_ACTION_ONLY_OR_WILDCARD = "TASK_SATURATED_BY_ACTION_ONLY_OR_WILDCARD"
    UNEVALUABLE_MORPHISM_PANEL = "UNEVALUABLE_MORPHISM_PANEL"


class PhysicalScaleMorphismSectionTerminal(StrEnum):
    BOUNDARY_SECTION_ONLY = "BOUNDARY_SECTION_ONLY"
    CATEGORICAL_SECTION_ONLY = "CATEGORICAL_SECTION_ONLY"
    DIMENSIONLESS_FIXED_SECTION_SUPPORTED = "DIMENSIONLESS_FIXED_SECTION_SUPPORTED"
    DYNAMICAL_SECTION_OPPOSED = "DYNAMICAL_SECTION_OPPOSED"
    METRIC_SECTION_ONLY = "METRIC_SECTION_ONLY"
    NATIVE_QUANTITY_NONFIXED_AS_PREDICTED = "NATIVE_QUANTITY_NONFIXED_AS_PREDICTED"
    SCALE_FLOW_NOT_IDENTIFIED = "SCALE_FLOW_NOT_IDENTIFIED"


class PhysicalScaleMorphismOverallTerminal(StrEnum):
    BOUNDED_PHYSICAL_MORPHISM_CLASS_SUPPORTED = "BOUNDED_PHYSICAL_MORPHISM_CLASS_SUPPORTED"
    PARTIAL_OR_OPPOSED = "PARTIAL_OR_OPPOSED"
    PREREQUISITE_STOP = "PREREQUISITE_STOP"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismAdjudicationFacts(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-adjudication-facts'

    facts_id: str
    method_qualified: bool
    construct_qualified: bool
    source_apparatus_qualified: bool
    power_terminal: PhysicalScaleMorphismPowerTerminal
    morphism_domain_complete: bool
    coordinate_disposition: PhysicalScaleMorphismCoordinateDisposition
    forecast_triplet_supported: bool | None
    categorical_forecast_supported: bool | None
    calibration_supported: bool | None
    observational_supported: bool | None
    semantic_future_closure_supported: bool | None
    interventional_supported: bool | None
    decision_supported: bool | None
    all_required_holds_measured_safe: bool
    false_safe_count: int
    receiver_supported: bool | None
    numerical_supported: bool | None
    physical_scale_supported: bool | None
    composition_supported: bool | None
    categorical_section_supported: bool | None
    metric_section_supported: bool | None
    dynamical_section_supported: bool | None
    boundary_section_supported: bool | None
    dimensionless_fixed_section_supported: bool | None
    native_quantity_nonfixed_as_predicted: bool | None
    scale_flow_identified: bool | None
    physical_scale_crossover_identified: bool
    heterogeneity_evaluated: bool
    heterogeneity_opposes_uniformity: bool
    negative_controls_rejected: bool
    unresolved_obstruction_count: int
    authority_or_independence_violation: bool
    multiplicity_violation: bool
    implementation_recurrence_claimed: bool
    implementation_recurrence_state: PhysicalScaleMorphismImplementationRecurrenceState
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.facts_id, field_name="facts_id")
        if self.false_safe_count < 0 or self.unresolved_obstruction_count < 0:
            raise ValueError("adjudication counts must be nonnegative")
        if (
            self.forecast_triplet_supported is True
            and self.categorical_forecast_supported is not True
        ):
            raise ValueError("paired triplet support requires categorical support")
        if self.dimensionless_fixed_section_supported is True and not all(
            value is True
            for value in (
                self.categorical_section_supported,
                self.metric_section_supported,
                self.dynamical_section_supported,
                self.boundary_section_supported,
            )
        ):
            raise ValueError("fixed-section support requires every nominated section member")
        if self.physical_scale_crossover_identified and self.physical_scale_supported is True:
            raise ValueError(
                "complete physical scale support cannot coexist with a terminating crossover"
            )
        if (
            self.implementation_recurrence_claimed
            and self.implementation_recurrence_state
            is PhysicalScaleMorphismImplementationRecurrenceState.NOT_TESTED
        ):
            raise ValueError(
                "claimed independent recurrence cannot remain an intentional nonattempt"
            )
        if (
            not self.implementation_recurrence_claimed
            and self.implementation_recurrence_state is PhysicalScaleMorphismImplementationRecurrenceState.SUPPORTED
        ):
            raise ValueError(
                "unclaimed implementation recurrence cannot acquire a positive terminal"
            )
        if self.maximum_evidence_ceiling is EvidenceCeiling.CONTROLLER_USE:
            raise ValueError("physical scale morphism has a hard admission ceiling")


def _evidence_bool(state: PhysicalScaleMorphismEvidenceState) -> bool | None:
    if state is PhysicalScaleMorphismEvidenceState.SUPPORTED:
        return True
    if state is PhysicalScaleMorphismEvidenceState.OPPOSED:
        return False
    return None


def derive_adjudication_facts(
    *,
    facts_id: str,
    evidence: PhysicalScaleMorphismAdjudicationEvidenceBundle,
) -> PhysicalScaleMorphismAdjudicationFacts:
    """Derive the terminal grammar inputs from exact compact evaluator panels."""

    axes = {value.map_family: value for value in evidence.morphism_axes}
    sections = {value.section: value for value in evidence.section_panels}
    forecast_member_states = tuple(
        member.state for score in evidence.forecast_scores for member in score.member_scores
    )
    if any(value is PhysicalScaleMorphismForecastState.UNEVALUABLE for value in forecast_member_states):
        forecast_supported: bool | None = None
    else:
        forecast_supported = all(
            value.paired_triplet_supported for value in evidence.forecast_scores
        )
    categorical_states = tuple(
        member.state
        for score in evidence.forecast_scores
        for member in score.member_scores
        if member.level is PhysicalScaleMorphismForecastLevel.CATEGORICAL
    )
    if any(value is PhysicalScaleMorphismForecastState.UNEVALUABLE for value in categorical_states):
        categorical_supported: bool | None = None
    else:
        categorical_supported = all(
            value is PhysicalScaleMorphismForecastState.SUPPORTED for value in categorical_states
        )

    def defect_supported(attribute: str, threshold: Decimal) -> bool | None:
        values = tuple(getattr(value, attribute) for value in evidence.defect_panels)
        if any(value.unmatched_operand_ids for value in evidence.defect_panels) or any(
            value is None for value in values
        ):
            return None
        return all(value <= threshold for value in values if value is not None)

    calibration = defect_supported("calibration_defect", evidence.defect_thresholds.calibration)
    observational = defect_supported(
        "observational_defect", evidence.defect_thresholds.observational
    )
    held_future = defect_supported(
        "held_future_semantic_defect", evidence.defect_thresholds.held_future_semantic
    )
    interventional = defect_supported(
        "interventional_defect", evidence.defect_thresholds.interventional
    )
    decision = defect_supported("decision_defect", evidence.defect_thresholds.decision)
    if held_future is None:
        semantic_future: bool | None = None
    else:
        semantic_future = held_future and all(
            value.closure_supported for value in evidence.semantic_panels
        )

    hold_safe = all(
        value.disposition is PhysicalScaleMorphismHoldDisposition.HOLD_MEASURED_SAFE
        for value in evidence.hold_records
    )
    false_safe_count = sum(value.false_safe_hold for value in evidence.hold_records)
    composition_axis = _evidence_bool(axes[PhysicalScaleMorphismMapFamily.COMPOSITION].state)
    if composition_axis is None:
        composition_supported: bool | None = None
    elif not composition_axis:
        composition_supported = False
    else:
        composition_supported = all(
            value.composition_supported for value in evidence.composition_panels
        )

    section_states = {
        section: _evidence_bool(sections[section].state)
        for section in (
            PhysicalScaleMorphismSectionKind.CATEGORICAL,
            PhysicalScaleMorphismSectionKind.METRIC,
            PhysicalScaleMorphismSectionKind.DYNAMICAL,
            PhysicalScaleMorphismSectionKind.BOUNDARY,
        )
    }
    if any(value is None for value in section_states.values()):
        fixed_section_supported: bool | None = None
    else:
        fixed_section_supported = all(value is True for value in section_states.values())
    heterogeneity_evaluated = all(
        value.leave_one_recomputed and value.full_panel_terminal_supplied
        for value in evidence.heterogeneity_panels
    )
    recurrence = evidence.implementation_recurrence
    return PhysicalScaleMorphismAdjudicationFacts(
        facts_id=facts_id,
        method_qualified=evidence.method_qualification.method_qualified,
        construct_qualified=evidence.construct.terminal
        is PhysicalScaleMorphismConstructTerminal.CONSTRUCT_INDEPENDENCE_QUALIFIED,
        source_apparatus_qualified=evidence.source_apparatus.terminal
        is PhysicalScaleMorphismSourceApparatusTerminal.SOURCE_AND_APPARATUS_QUALIFIED,
        power_terminal=evidence.development.power_terminal,
        morphism_domain_complete=evidence.development.morphism_domain_complete,
        coordinate_disposition=evidence.coordinate_panel.disposition,
        forecast_triplet_supported=forecast_supported,
        categorical_forecast_supported=categorical_supported,
        calibration_supported=calibration,
        observational_supported=observational,
        semantic_future_closure_supported=semantic_future,
        interventional_supported=interventional,
        decision_supported=decision,
        all_required_holds_measured_safe=hold_safe,
        false_safe_count=false_safe_count,
        receiver_supported=_evidence_bool(axes[PhysicalScaleMorphismMapFamily.RECEIVER].state),
        numerical_supported=_evidence_bool(axes[PhysicalScaleMorphismMapFamily.NUMERICAL].state),
        physical_scale_supported=_evidence_bool(axes[PhysicalScaleMorphismMapFamily.PHYSICAL_SCALE].state),
        composition_supported=composition_supported,
        categorical_section_supported=section_states[PhysicalScaleMorphismSectionKind.CATEGORICAL],
        metric_section_supported=section_states[PhysicalScaleMorphismSectionKind.METRIC],
        dynamical_section_supported=section_states[PhysicalScaleMorphismSectionKind.DYNAMICAL],
        boundary_section_supported=section_states[PhysicalScaleMorphismSectionKind.BOUNDARY],
        dimensionless_fixed_section_supported=fixed_section_supported,
        native_quantity_nonfixed_as_predicted=_evidence_bool(
            evidence.native_quantity_control.state
        ),
        scale_flow_identified=_evidence_bool(sections[PhysicalScaleMorphismSectionKind.SCALE_FLOW].state),
        physical_scale_crossover_identified=bool(axes[PhysicalScaleMorphismMapFamily.PHYSICAL_SCALE].crossover_ids),
        heterogeneity_evaluated=heterogeneity_evaluated,
        heterogeneity_opposes_uniformity=any(
            value.mixed for value in evidence.heterogeneity_panels
        ),
        negative_controls_rejected=evidence.negative_control_audit.all_rejected,
        unresolved_obstruction_count=sum(
            PhysicalScaleMorphismObstructionKind.UNRESOLVED in value.kinds for value in evidence.obstruction_sets
        ),
        authority_or_independence_violation=(evidence.authority_independence_audit.violation),
        multiplicity_violation=evidence.multiplicity_audit.violation,
        implementation_recurrence_claimed=(len(recurrence.entered_implementation_ids) == 2),
        implementation_recurrence_state=recurrence.state,
        maximum_evidence_ceiling=EvidenceCeiling.ADMISSION,
    )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismAdjudicationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-adjudication-result'

    result_id: str
    facts_id: str
    axis_terminals: tuple[PhysicalScaleMorphismAxisTerminal, ...]
    section_terminals: tuple[PhysicalScaleMorphismSectionTerminal, ...]
    overall_terminal: PhysicalScaleMorphismOverallTerminal
    evidence_ceiling: EvidenceCeiling
    controller_use_claim_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.facts_id, field_name="facts_id")
        if (
            tuple(sorted(set(self.axis_terminals), key=lambda value: value.value))
            != self.axis_terminals
        ):
            raise ValueError("axis terminals must be sorted and unique")
        if (
            tuple(sorted(set(self.section_terminals), key=lambda value: value.value))
            != self.section_terminals
        ):
            raise ValueError("section terminals must be sorted and unique")
        positive = (
            PhysicalScaleMorphismAxisTerminal.BOUNDED_PHYSICAL_MORPHISM_CLASS_SUPPORTED in self.axis_terminals
        )
        if positive != (
            self.overall_terminal is PhysicalScaleMorphismOverallTerminal.BOUNDED_PHYSICAL_MORPHISM_CLASS_SUPPORTED
        ):
            raise ValueError("overall and axis positive terminals differ")
        if positive and self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("positive bounded physical class requires the exact admission ceiling")
        if self.evidence_ceiling is EvidenceCeiling.CONTROLLER_USE:
            raise ValueError("physical scale morphism physical result cannot exceed admission")
        if self.controller_use_claim_allowed:
            raise ValueError("physical scale morphism cannot authorize or claim controller use")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismAdjudicationSeal(CanonicalRecord):
    """Exact fan-in/reveal/evidence binding for one frozen IP-12 adjudication."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-adjudication-seal'

    seal_id: str
    prediction_freeze: ObjectIdentity
    complete_fan_in: ObjectIdentity
    reveal_authority: StudyOperationAuthority
    evaluator_id: str
    evidence_bundle: PhysicalScaleMorphismAdjudicationEvidenceBundle
    facts: PhysicalScaleMorphismAdjudicationFacts
    adjudication: PhysicalScaleMorphismAdjudicationResult
    evidence_panel_identities: tuple[ObjectIdentity, ...]
    input_artifact_locator_ids: tuple[str, ...]
    output_artifact_locator_ids: tuple[str, ...]
    publication_receipt_ids: tuple[str, ...]
    outcome_access_event_ids: tuple[str, ...]
    outcome_access: OutcomeAccess
    raw_evaluation_rows_returned: bool
    frozen: bool

    def __post_init__(self) -> None:
        for name in ("seal_id", "evaluator_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.prediction_freeze.object_schema != PhysicalScaleMorphismPredictionFreeze.SCHEMA:
            raise ValueError("adjudication seal binds the wrong prediction-freeze schema")
        if self.complete_fan_in.object_schema != PhysicalScaleMorphismCompleteFanIn.SCHEMA:
            raise ValueError("adjudication seal binds the wrong complete-fan-in schema")
        if (
            self.reveal_authority.kind is not StudyAuthorityKind.OUTCOME_REVEAL
            or not self.reveal_authority.allows_reveal
            or self.reveal_authority.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        ):
            raise ValueError("adjudication seal lacks exact evaluator-reveal authority")
        if self.reveal_authority.grantee_id != self.evaluator_id:
            raise ValueError("adjudication evaluator differs from the reveal-authority grantee")
        expected_issue = ObjectIdentity.from_record(
            self.evidence_bundle.issue.issue_id,
            self.evidence_bundle.issue,
        )
        if self.reveal_authority.subject != expected_issue:
            raise ValueError("adjudication reveal authority binds another issue record")
        expected_freeze = ObjectIdentity.from_record(
            self.evidence_bundle.prediction_freeze.freeze_id,
            self.evidence_bundle.prediction_freeze,
        )
        if self.prediction_freeze != expected_freeze:
            raise ValueError("adjudication evidence binds another prediction freeze")
        expected_fan_in = ObjectIdentity.from_record(
            self.evidence_bundle.fan_in.fan_in_id,
            self.evidence_bundle.fan_in,
        )
        if self.complete_fan_in != expected_fan_in:
            raise ValueError("adjudication evidence binds another complete fan-in")
        expected_facts = derive_adjudication_facts(
            facts_id=self.facts.facts_id,
            evidence=self.evidence_bundle,
        )
        if self.facts != expected_facts:
            raise ValueError("adjudication facts are not exactly evidence-derived")
        if self.adjudication.facts_id != self.facts.facts_id:
            raise ValueError("adjudication seal result binds another facts record")
        expected = adjudicate_physical_scale_morphism(result_id=self.adjudication.result_id, facts=self.facts)
        if self.adjudication != expected:
            raise ValueError("adjudication result is not exactly facts-derived")
        object_keys = tuple(
            (value.object_schema, value.object_id, value.object_fingerprint)
            for value in self.evidence_panel_identities
        )
        if tuple(sorted(set(object_keys))) != object_keys or not object_keys:
            raise ValueError("adjudication evidence panels must be sorted, unique and nonempty")
        bundle_identity = ObjectIdentity.from_record(
            self.evidence_bundle.bundle_id,
            self.evidence_bundle,
        )
        if bundle_identity not in self.evidence_panel_identities:
            raise ValueError("adjudication panel roster omits the derived evidence bundle")
        for name in (
            "input_artifact_locator_ids",
            "output_artifact_locator_ids",
            "publication_receipt_ids",
            "outcome_access_event_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        if (
            self.input_artifact_locator_ids
            != self.evidence_bundle.fan_in.sealed_payload_artifact_ids
        ):
            raise ValueError("adjudication input artifacts differ from complete sealed fan-in")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("adjudication seal must record the authorized revealed state")
        if self.raw_evaluation_rows_returned:
            raise ValueError("ordinary physical scale morphism closeout cannot return raw evaluation rows")
        if not self.frozen:
            raise ValueError("adjudication seal must be immutable")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismCompletionEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-completion-envelope'

    envelope_id: str
    plan_sha256: str
    candidate_sha256: str
    run_sha256: str
    result_sha256: str
    candidate: ObjectIdentity
    run: ObjectIdentity
    prediction_freeze: ObjectIdentity
    complete_fan_in: ObjectIdentity
    adjudication_seal: PhysicalScaleMorphismAdjudicationSeal
    evidence_world_ids: tuple[str, ...]
    board_ids: tuple[str, ...]
    batch_ids: tuple[str, ...]
    scale_ids: tuple[str, ...]
    map_ids: tuple[str, ...]
    surviving_property_ids: tuple[str, ...]
    opposed_property_ids: tuple[str, ...]
    unevaluable_property_ids: tuple[str, ...]
    boundary_summary_ids: tuple[str, ...]
    obstruction_ids: tuple[str, ...]
    decisive_falsifier_ids: tuple[str, ...]
    uncertainty_method_id: str
    multiplicity_method_id: str
    artifact_locator_ids: tuple[str, ...]
    receipt_locator_ids: tuple[str, ...]
    recovery_locator_ids: tuple[str, ...]
    outcome_access_event_ids: tuple[str, ...]
    source_status_id: str
    apparatus_status_id: str
    method_status_id: str
    numerical_status_id: str
    physical_status_id: str
    operational_status_id: str
    follow_up_nomination_id: str
    raw_payload_embedded: bool

    def __post_init__(self) -> None:
        for name in (
            "envelope_id",
            "uncertainty_method_id",
            "multiplicity_method_id",
            "source_status_id",
            "apparatus_status_id",
            "method_status_id",
            "numerical_status_id",
            "physical_status_id",
            "operational_status_id",
            "follow_up_nomination_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("plan_sha256", "candidate_sha256", "run_sha256", "result_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.candidate.object_schema not in {
            'empirical-lawhood/runtime/draft-study-candidate',
            'empirical-lawhood/runtime/study-candidate',
        }:
            raise ValueError("completion envelope binds the wrong candidate schema")
        if self.run.object_schema != 'empirical-lawhood/runtime/candidate-run-plan':
            raise ValueError("completion envelope binds the wrong run-plan schema")
        if self.candidate_sha256 != self.candidate.object_fingerprint:
            raise ValueError("completion candidate hash differs from its exact identity")
        if self.run_sha256 != self.run.object_fingerprint:
            raise ValueError("completion run hash differs from its exact identity")
        evidence = self.adjudication_seal.evidence_bundle
        freeze = evidence.prediction_freeze
        if self.plan_sha256 != freeze.plan_sha256:
            raise ValueError("completion plan hash differs from the prediction freeze")
        if self.candidate != freeze.candidate:
            raise ValueError("completion candidate differs from the prediction freeze")
        if self.run != evidence.issue.issued_run or self.run != evidence.execution.run:
            raise ValueError("completion run differs from the issued/executed run")
        if self.prediction_freeze != self.adjudication_seal.prediction_freeze:
            raise ValueError("completion envelope binds another prediction freeze")
        if self.complete_fan_in != self.adjudication_seal.complete_fan_in:
            raise ValueError("completion envelope binds another fan-in")
        if self.result_sha256 != self.adjudication_seal.adjudication.fingerprint():
            raise ValueError("completion result hash is not adjudication-derived")
        required_nonempty = {
            "evidence_world_ids",
            "board_ids",
            "batch_ids",
            "scale_ids",
            "map_ids",
            "boundary_summary_ids",
            "obstruction_ids",
            "decisive_falsifier_ids",
            "artifact_locator_ids",
            "receipt_locator_ids",
            "recovery_locator_ids",
            "outcome_access_event_ids",
        }
        for name in (
            "evidence_world_ids",
            "board_ids",
            "batch_ids",
            "scale_ids",
            "map_ids",
            "surviving_property_ids",
            "opposed_property_ids",
            "unevaluable_property_ids",
            "boundary_summary_ids",
            "obstruction_ids",
            "decisive_falsifier_ids",
            "artifact_locator_ids",
            "receipt_locator_ids",
            "recovery_locator_ids",
            "outcome_access_event_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name), field_name=name, allow_empty=name not in required_nonempty
            )
        property_sets = (
            set(self.surviving_property_ids),
            set(self.opposed_property_ids),
            set(self.unevaluable_property_ids),
        )
        if any(
            left & right
            for index, left in enumerate(property_sets)
            for right in property_sets[index + 1 :]
        ):
            raise ValueError("completion property dispositions must be disjoint")
        expected_properties = _completion_property_dispositions(evidence)
        if (
            self.surviving_property_ids,
            self.opposed_property_ids,
            self.unevaluable_property_ids,
        ) != expected_properties:
            raise ValueError("completion property dispositions are not evidence-derived")
        expected_rosters = {
            "evidence_world_ids": tuple(sorted(value.value for value in PhysicalScaleMorphismEvidenceWorld)),
            "board_ids": freeze.evaluation_board_ids,
            "batch_ids": freeze.evaluation_batch_ids,
            "scale_ids": tuple(f"scale.n{value}" for value in freeze.evaluation_scale_cells),
            "map_ids": tuple(
                sorted(
                    morphism_id
                    for axis in evidence.morphism_axes
                    for morphism_id in axis.required_morphism_ids
                )
            ),
            "boundary_summary_ids": tuple(
                sorted(
                    boundary_id
                    for panel in evidence.section_panels
                    for boundary_id in panel.boundary_complex_ids
                )
            ),
            "obstruction_ids": tuple(value.obstruction_id for value in evidence.obstruction_sets),
            "decisive_falsifier_ids": tuple(
                sorted(
                    {
                        *evidence.negative_control_audit.required_control_ids,
                        *(
                            control_id
                            for axis in evidence.morphism_axes
                            for control_id in axis.required_negative_control_ids
                        ),
                    }
                )
            ),
        }
        for name, expected in expected_rosters.items():
            if getattr(self, name) != expected:
                raise ValueError(f"completion {name} is not evidence-derived")
        if self.uncertainty_method_id != freeze.uncertainty_method_id:
            raise ValueError("completion uncertainty method differs from the prediction freeze")
        if self.multiplicity_method_id != evidence.development.multiplicity_rule_id:
            raise ValueError("completion multiplicity method differs from development freeze")
        expected_statuses = _completion_statuses(evidence, self.adjudication)
        for status_name, expected_status in expected_statuses.items():
            if getattr(self, status_name) != expected_status:
                raise ValueError(f"completion {status_name} is not evidence-derived")
        if self.raw_payload_embedded:
            raise ValueError("completion envelope cannot embed physical/scientific payloads")
        expected_locator_sets = _completion_required_locators(evidence, self.adjudication_seal)
        for name in ("artifact_locator_ids", "receipt_locator_ids"):
            if not set(expected_locator_sets[name]).issubset(getattr(self, name)):
                raise ValueError(f"completion envelope omits required {name}")
        for name in ("recovery_locator_ids", "outcome_access_event_ids"):
            if getattr(self, name) != expected_locator_sets[name]:
                raise ValueError(f"completion {name} is not evidence-derived")

    @property
    def adjudication(self) -> PhysicalScaleMorphismAdjudicationResult:
        return self.adjudication_seal.adjudication


def _completion_property_dispositions(
    evidence: PhysicalScaleMorphismAdjudicationEvidenceBundle,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    disposition_by_property: dict[str, str] = {}

    def add(property_ids: tuple[str, ...], disposition: str) -> None:
        for property_id in property_ids:
            previous = disposition_by_property.setdefault(property_id, disposition)
            if previous != disposition:
                raise ValueError(
                    "evidence assigns one completion property to conflicting dispositions"
                )

    for panel in evidence.section_panels:
        add(panel.surviving_property_ids, "surviving")
        add(panel.opposed_property_ids, "opposed")
        add(panel.unevaluable_property_ids, "unevaluable")
    native = evidence.native_quantity_control
    add(native.observed_nonfixed_quantity_ids, "surviving")
    add(native.fixed_against_prediction_ids, "opposed")
    add(native.unevaluable_quantity_ids, "unevaluable")
    surviving = tuple(
        sorted(
            property_id
            for property_id, observed in disposition_by_property.items()
            if observed == "surviving"
        )
    )
    opposed = tuple(
        sorted(
            property_id
            for property_id, observed in disposition_by_property.items()
            if observed == "opposed"
        )
    )
    unevaluable = tuple(
        sorted(
            property_id
            for property_id, observed in disposition_by_property.items()
            if observed == "unevaluable"
        )
    )
    return surviving, opposed, unevaluable


def _completion_statuses(
    evidence: PhysicalScaleMorphismAdjudicationEvidenceBundle,
    adjudication: PhysicalScaleMorphismAdjudicationResult,
) -> dict[str, str]:
    source_qualified = (
        evidence.source_apparatus.terminal
        is PhysicalScaleMorphismSourceApparatusTerminal.SOURCE_AND_APPARATUS_QUALIFIED
    )
    return {
        "source_status_id": (
            "status.source-qualified" if source_qualified else "status.source-not-qualified"
        ),
        "apparatus_status_id": (
            "status.apparatus-qualified" if source_qualified else "status.apparatus-not-qualified"
        ),
        "method_status_id": (
            "status.method-qualified"
            if evidence.method_qualification.method_qualified
            else "status.method-not-qualified"
        ),
        "numerical_status_id": (
            "status.numerical-supported"
            if PhysicalScaleMorphismAxisTerminal.NUMERICAL_REFINEMENT_SUPPORTED in adjudication.axis_terminals
            else "status.numerical-opposed-or-unevaluable"
        ),
        "physical_status_id": (
            "status.physical-complete"
            if evidence.execution.complete and evidence.fan_in.complete
            else "status.physical-incomplete"
        ),
        "operational_status_id": (
            "status.complete"
            if evidence.execution.complete and evidence.fan_in.complete
            else "status.incomplete"
        ),
        "follow_up_nomination_id": {
            PhysicalScaleMorphismOverallTerminal.BOUNDED_PHYSICAL_MORPHISM_CLASS_SUPPORTED: "nomination.none",
            PhysicalScaleMorphismOverallTerminal.PARTIAL_OR_OPPOSED: "nomination.targeted-obstruction-study",
            PhysicalScaleMorphismOverallTerminal.PREREQUISITE_STOP: "nomination.resolve-prerequisite",
            PhysicalScaleMorphismOverallTerminal.UNEVALUABLE: "nomination.resolve-unevaluable-operands",
        }[adjudication.overall_terminal],
    }


def _completion_required_locators(
    evidence: PhysicalScaleMorphismAdjudicationEvidenceBundle,
    seal: PhysicalScaleMorphismAdjudicationSeal,
) -> dict[str, tuple[str, ...]]:
    source_artifacts = tuple(
        artifact_id
        for gate in evidence.source_apparatus.gate_assessments
        for artifact_id in gate.evidence_artifact_locator_ids
    )
    return {
        "artifact_locator_ids": tuple(
            sorted(
                {
                    *source_artifacts,
                    *evidence.execution.physical_bundle_artifact_ids,
                    *evidence.execution.numerical_artifact_ids,
                    *evidence.fan_in.sealed_payload_artifact_ids,
                    *seal.input_artifact_locator_ids,
                    *seal.output_artifact_locator_ids,
                }
            )
        ),
        "receipt_locator_ids": tuple(
            sorted(
                {
                    *evidence.source_apparatus.publication_receipt_ids,
                    *evidence.issue.qualification_receipt_ids,
                    *evidence.execution.publication_receipt_ids,
                    *evidence.fan_in.publication_receipt_ids,
                    *seal.publication_receipt_ids,
                }
            )
        ),
        "recovery_locator_ids": tuple(
            sorted(
                {
                    *evidence.fan_in.recovery_event_ids,
                    *evidence.fan_in.recovery_proof_ids,
                }
            )
        ),
        "outcome_access_event_ids": tuple(
            sorted(
                {
                    *evidence.fan_in.outcome_access_event_ids,
                    *seal.outcome_access_event_ids,
                }
            )
        ),
    }


def make_physical_scale_morphism_completion_envelope(
    *,
    envelope_id: str,
    adjudication_seal: PhysicalScaleMorphismAdjudicationSeal,
    additional_artifact_locator_ids: tuple[str, ...] = (),
    additional_receipt_locator_ids: tuple[str, ...] = (),
) -> PhysicalScaleMorphismCompletionEnvelope:
    """Build an IP-13 envelope from exact sealed evidence, never summary inputs."""

    evidence = adjudication_seal.evidence_bundle
    freeze = evidence.prediction_freeze
    freeze_identity = ObjectIdentity.from_record(freeze.freeze_id, freeze)
    fan_in_identity = ObjectIdentity.from_record(evidence.fan_in.fan_in_id, evidence.fan_in)
    properties = _completion_property_dispositions(evidence)
    statuses = _completion_statuses(evidence, adjudication_seal.adjudication)
    locators = _completion_required_locators(evidence, adjudication_seal)
    artifacts = tuple(sorted({*locators["artifact_locator_ids"], *additional_artifact_locator_ids}))
    receipts = tuple(sorted({*locators["receipt_locator_ids"], *additional_receipt_locator_ids}))
    return PhysicalScaleMorphismCompletionEnvelope(
        envelope_id=envelope_id,
        plan_sha256=freeze.plan_sha256,
        candidate_sha256=freeze.candidate.object_fingerprint,
        run_sha256=evidence.issue.issued_run.object_fingerprint,
        result_sha256=adjudication_seal.adjudication.fingerprint(),
        candidate=freeze.candidate,
        run=evidence.issue.issued_run,
        prediction_freeze=freeze_identity,
        complete_fan_in=fan_in_identity,
        adjudication_seal=adjudication_seal,
        evidence_world_ids=tuple(sorted(value.value for value in PhysicalScaleMorphismEvidenceWorld)),
        board_ids=freeze.evaluation_board_ids,
        batch_ids=freeze.evaluation_batch_ids,
        scale_ids=tuple(f"scale.n{value}" for value in freeze.evaluation_scale_cells),
        map_ids=tuple(
            sorted(
                morphism_id
                for axis in evidence.morphism_axes
                for morphism_id in axis.required_morphism_ids
            )
        ),
        surviving_property_ids=properties[0],
        opposed_property_ids=properties[1],
        unevaluable_property_ids=properties[2],
        boundary_summary_ids=tuple(
            sorted(
                boundary_id
                for panel in evidence.section_panels
                for boundary_id in panel.boundary_complex_ids
            )
        ),
        obstruction_ids=tuple(value.obstruction_id for value in evidence.obstruction_sets),
        decisive_falsifier_ids=tuple(
            sorted(
                {
                    *evidence.negative_control_audit.required_control_ids,
                    *(
                        control_id
                        for axis in evidence.morphism_axes
                        for control_id in axis.required_negative_control_ids
                    ),
                }
            )
        ),
        uncertainty_method_id=freeze.uncertainty_method_id,
        multiplicity_method_id=evidence.development.multiplicity_rule_id,
        artifact_locator_ids=artifacts,
        receipt_locator_ids=receipts,
        recovery_locator_ids=locators["recovery_locator_ids"],
        outcome_access_event_ids=locators["outcome_access_event_ids"],
        raw_payload_embedded=False,
        **statuses,
    )


def _axis_terminals(facts: PhysicalScaleMorphismAdjudicationFacts) -> set[PhysicalScaleMorphismAxisTerminal]:
    terminals: set[PhysicalScaleMorphismAxisTerminal] = set()
    if not facts.method_qualified:
        terminals.add(PhysicalScaleMorphismAxisTerminal.METHOD_NOT_QUALIFIED)
    if not facts.construct_qualified:
        terminals.add(PhysicalScaleMorphismAxisTerminal.CONSTRUCT_INDEPENDENCE_NOT_QUALIFIED)
    if not facts.source_apparatus_qualified:
        terminals.add(PhysicalScaleMorphismAxisTerminal.SOURCE_OR_APPARATUS_NOT_QUALIFIED)
    if facts.power_terminal is PhysicalScaleMorphismPowerTerminal.PRECISION_LIMITED_AT_RESOURCE_CEILING:
        terminals.add(PhysicalScaleMorphismAxisTerminal.PRECISION_LIMITED_AT_RESOURCE_CEILING)
    elif facts.power_terminal is PhysicalScaleMorphismPowerTerminal.NO_FINITE_COUNT_UNDER_OBSERVED_RATE:
        terminals.add(PhysicalScaleMorphismAxisTerminal.NO_FINITE_COUNT_UNDER_OBSERVED_RATE)
    elif facts.power_terminal is PhysicalScaleMorphismPowerTerminal.DEVELOPMENT_EFFECT_ON_WRONG_SIDE:
        terminals.add(PhysicalScaleMorphismAxisTerminal.DEVELOPMENT_EFFECT_ON_WRONG_SIDE)
    if not facts.morphism_domain_complete:
        terminals.add(PhysicalScaleMorphismAxisTerminal.MORPHISM_DOMAIN_INCOMPLETE)

    coordinate_terminal = {
        PhysicalScaleMorphismCoordinateDisposition.COORDINATE_NECESSITY_IDENTIFIED: (
            PhysicalScaleMorphismAxisTerminal.COORDINATE_NECESSITY_IDENTIFIED
        ),
        PhysicalScaleMorphismCoordinateDisposition.COORDINATE_PREDICTION_OPPOSED: (
            PhysicalScaleMorphismAxisTerminal.COORDINATE_PREDICTION_OPPOSED
        ),
        PhysicalScaleMorphismCoordinateDisposition.TASK_SATURATED_BY_ACTION_ONLY_OR_WILDCARD: (
            PhysicalScaleMorphismAxisTerminal.TASK_SATURATED_BY_ACTION_ONLY_OR_WILDCARD
        ),
        PhysicalScaleMorphismCoordinateDisposition.UNEVALUABLE: PhysicalScaleMorphismAxisTerminal.UNEVALUABLE_MORPHISM_PANEL,
    }[facts.coordinate_disposition]
    terminals.add(coordinate_terminal)

    if facts.forecast_triplet_supported is True:
        terminals.add(PhysicalScaleMorphismAxisTerminal.PAIRED_FORECAST_TRIPLET_SUPPORTED)
    elif facts.categorical_forecast_supported is True:
        terminals.add(PhysicalScaleMorphismAxisTerminal.CATEGORICAL_ONLY_MORPHISM)
    elif facts.forecast_triplet_supported is None or facts.categorical_forecast_supported is None:
        terminals.add(PhysicalScaleMorphismAxisTerminal.UNEVALUABLE_MORPHISM_PANEL)

    if facts.calibration_supported is True and facts.observational_supported is False:
        terminals.add(PhysicalScaleMorphismAxisTerminal.CALIBRATION_ONLY)
    if (
        facts.calibration_supported is True
        and facts.observational_supported is True
        and (
            facts.semantic_future_closure_supported is False
            or facts.interventional_supported is False
            or facts.decision_supported is False
        )
    ):
        terminals.add(PhysicalScaleMorphismAxisTerminal.OBSERVATIONAL_ONLY)
    if facts.semantic_future_closure_supported is False:
        terminals.add(PhysicalScaleMorphismAxisTerminal.SEMANTIC_FUTURE_CLOSURE_OPPOSED)
    if facts.interventional_supported is False:
        terminals.add(PhysicalScaleMorphismAxisTerminal.INTERVENTIONAL_MORPHISM_OPPOSED)
    if facts.decision_supported is False:
        terminals.add(PhysicalScaleMorphismAxisTerminal.DECISION_MORPHISM_OPPOSED)
    if not facts.all_required_holds_measured_safe or facts.false_safe_count:
        terminals.update(
            {
                PhysicalScaleMorphismAxisTerminal.HOLD_FIBRE_UNQUALIFIED_OR_UNSAFE,
                PhysicalScaleMorphismAxisTerminal.DECISION_MORPHISM_OPPOSED,
            }
        )

    for state, supported, opposed in (
        (
            facts.receiver_supported,
            PhysicalScaleMorphismAxisTerminal.RECEIVER_MORPHISM_SUPPORTED,
            PhysicalScaleMorphismAxisTerminal.RECEIVER_MORPHISM_OPPOSED,
        ),
        (
            facts.numerical_supported,
            PhysicalScaleMorphismAxisTerminal.NUMERICAL_REFINEMENT_SUPPORTED,
            PhysicalScaleMorphismAxisTerminal.NUMERICAL_REFINEMENT_OPPOSED,
        ),
        (
            facts.physical_scale_supported,
            PhysicalScaleMorphismAxisTerminal.PHYSICAL_SCALE_MORPHISM_SUPPORTED,
            PhysicalScaleMorphismAxisTerminal.PHYSICAL_SCALE_MORPHISM_OPPOSED,
        ),
    ):
        terminals.add(
            supported
            if state is True
            else opposed
            if state is False
            else PhysicalScaleMorphismAxisTerminal.UNEVALUABLE_MORPHISM_PANEL
        )
    if facts.composition_supported is False:
        terminals.add(PhysicalScaleMorphismAxisTerminal.COMPOSITION_OPPOSED)
    elif facts.composition_supported is None:
        terminals.add(PhysicalScaleMorphismAxisTerminal.UNEVALUABLE_MORPHISM_PANEL)
    if facts.physical_scale_crossover_identified:
        terminals.add(PhysicalScaleMorphismAxisTerminal.PHYSICAL_SCALE_CROSSOVER_IDENTIFIED)
    if facts.heterogeneity_opposes_uniformity:
        terminals.add(PhysicalScaleMorphismAxisTerminal.FIBREWISE_HETEROGENEITY_MIXED)
    if not facts.heterogeneity_evaluated:
        terminals.add(PhysicalScaleMorphismAxisTerminal.UNEVALUABLE_MORPHISM_PANEL)

    recurrence_terminal = {
        PhysicalScaleMorphismImplementationRecurrenceState.SUPPORTED: (
            PhysicalScaleMorphismAxisTerminal.INDEPENDENT_IMPLEMENTATION_RECURRENCE_SUPPORTED
        ),
        PhysicalScaleMorphismImplementationRecurrenceState.OPPOSED: (
            PhysicalScaleMorphismAxisTerminal.INDEPENDENT_IMPLEMENTATION_RECURRENCE_OPPOSED
        ),
        PhysicalScaleMorphismImplementationRecurrenceState.NOT_TESTED: (
            PhysicalScaleMorphismAxisTerminal.INDEPENDENT_IMPLEMENTATION_RECURRENCE_NOT_TESTED
        ),
        PhysicalScaleMorphismImplementationRecurrenceState.UNEVALUABLE: (
            PhysicalScaleMorphismAxisTerminal.UNEVALUABLE_MORPHISM_PANEL
        ),
    }[facts.implementation_recurrence_state]
    terminals.add(recurrence_terminal)
    return terminals


def _section_terminals(facts: PhysicalScaleMorphismAdjudicationFacts) -> set[PhysicalScaleMorphismSectionTerminal]:
    terminals: set[PhysicalScaleMorphismSectionTerminal] = set()
    if facts.dimensionless_fixed_section_supported is True:
        terminals.add(PhysicalScaleMorphismSectionTerminal.DIMENSIONLESS_FIXED_SECTION_SUPPORTED)
    else:
        if facts.categorical_section_supported is True:
            terminals.add(PhysicalScaleMorphismSectionTerminal.CATEGORICAL_SECTION_ONLY)
        if facts.metric_section_supported is True:
            terminals.add(PhysicalScaleMorphismSectionTerminal.METRIC_SECTION_ONLY)
        if facts.boundary_section_supported is True:
            terminals.add(PhysicalScaleMorphismSectionTerminal.BOUNDARY_SECTION_ONLY)
    if facts.dynamical_section_supported is False:
        terminals.add(PhysicalScaleMorphismSectionTerminal.DYNAMICAL_SECTION_OPPOSED)
    if facts.native_quantity_nonfixed_as_predicted is True:
        terminals.add(PhysicalScaleMorphismSectionTerminal.NATIVE_QUANTITY_NONFIXED_AS_PREDICTED)
    if facts.scale_flow_identified is False:
        terminals.add(PhysicalScaleMorphismSectionTerminal.SCALE_FLOW_NOT_IDENTIFIED)
    return terminals


def adjudicate_physical_scale_morphism(*, result_id: str, facts: PhysicalScaleMorphismAdjudicationFacts) -> PhysicalScaleMorphismAdjudicationResult:
    terminals = _axis_terminals(facts)
    sections = _section_terminals(facts)
    prerequisites = all(
        (
            facts.method_qualified,
            facts.construct_qualified,
            facts.source_apparatus_qualified,
            facts.power_terminal is PhysicalScaleMorphismPowerTerminal.METHOD_POWER_QUALIFIED,
            facts.morphism_domain_complete,
        )
    )
    recurrence_requirement_met = (
        not facts.implementation_recurrence_claimed
        or facts.implementation_recurrence_state is PhysicalScaleMorphismImplementationRecurrenceState.SUPPORTED
    )
    all_positive = all(
        (
            prerequisites,
            facts.coordinate_disposition
            is PhysicalScaleMorphismCoordinateDisposition.COORDINATE_NECESSITY_IDENTIFIED,
            facts.forecast_triplet_supported is True,
            facts.calibration_supported is True,
            facts.observational_supported is True,
            facts.semantic_future_closure_supported is True,
            facts.interventional_supported is True,
            facts.decision_supported is True,
            facts.all_required_holds_measured_safe,
            facts.false_safe_count == 0,
            facts.receiver_supported is True,
            facts.numerical_supported is True,
            facts.physical_scale_supported is True,
            facts.composition_supported is True,
            facts.dimensionless_fixed_section_supported is True,
            facts.native_quantity_nonfixed_as_predicted is True,
            not facts.physical_scale_crossover_identified,
            facts.heterogeneity_evaluated,
            not facts.heterogeneity_opposes_uniformity,
            facts.negative_controls_rejected,
            facts.unresolved_obstruction_count == 0,
            not facts.authority_or_independence_violation,
            not facts.multiplicity_violation,
            recurrence_requirement_met,
            facts.maximum_evidence_ceiling is EvidenceCeiling.ADMISSION,
        )
    )
    evaluability_values = (
        facts.forecast_triplet_supported,
        facts.categorical_forecast_supported,
        facts.calibration_supported,
        facts.observational_supported,
        facts.semantic_future_closure_supported,
        facts.interventional_supported,
        facts.decision_supported,
        facts.receiver_supported,
        facts.numerical_supported,
        facts.physical_scale_supported,
        facts.composition_supported,
        facts.categorical_section_supported,
        facts.metric_section_supported,
        facts.dynamical_section_supported,
        facts.boundary_section_supported,
        facts.dimensionless_fixed_section_supported,
        facts.native_quantity_nonfixed_as_predicted,
        facts.scale_flow_identified,
        facts.heterogeneity_evaluated,
    )
    if all_positive:
        terminals.add(PhysicalScaleMorphismAxisTerminal.BOUNDED_PHYSICAL_MORPHISM_CLASS_SUPPORTED)
        overall = PhysicalScaleMorphismOverallTerminal.BOUNDED_PHYSICAL_MORPHISM_CLASS_SUPPORTED
    elif not prerequisites:
        overall = PhysicalScaleMorphismOverallTerminal.PREREQUISITE_STOP
    elif not facts.heterogeneity_evaluated or any(value is None for value in evaluability_values):
        overall = PhysicalScaleMorphismOverallTerminal.UNEVALUABLE
    else:
        overall = PhysicalScaleMorphismOverallTerminal.PARTIAL_OR_OPPOSED
    return PhysicalScaleMorphismAdjudicationResult(
        result_id=result_id,
        facts_id=facts.facts_id,
        axis_terminals=tuple(sorted(terminals, key=lambda value: value.value)),
        section_terminals=tuple(sorted(sections, key=lambda value: value.value)),
        overall_terminal=overall,
        evidence_ceiling=facts.maximum_evidence_ceiling,
        controller_use_claim_allowed=False,
    )


__all__ = [
    'PhysicalScaleMorphismAdjudicationFacts',
    'PhysicalScaleMorphismAdjudicationResult',
    'PhysicalScaleMorphismAdjudicationSeal',
    'PhysicalScaleMorphismAxisTerminal',
    'PhysicalScaleMorphismCompletionEnvelope',
    'PhysicalScaleMorphismOverallTerminal',
    'PhysicalScaleMorphismSectionTerminal',
    'adjudicate_physical_scale_morphism',
    "derive_adjudication_facts",
    'make_physical_scale_morphism_completion_envelope',
]
