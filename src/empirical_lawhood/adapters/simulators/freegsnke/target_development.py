"Development-only independent substrate grounding selection and structural recurrence prediction freeze for FreeGSNKE.\n\nThis module is the causal development-freeze boundary.  It evaluates only the finite,\nsource-qualified mapping/denominator designs frozen upstream, uses every issued\ndevelopment preparation, selects by the common outcome-blind scientific grammar precedence, and constructs\nthe immutable target contract and unchanged structural recurrence request.  It cannot read\nevaluation outcomes and it does not generate simulator responses.\n"

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentImplementationDossier, IndependentSubstrateComparatorEncoding, IndependentSubstrateComparatorKind, IndependentSubstrateDenominatorDevelopmentAssessment, IndependentSubstrateForecastAlphabetGrammar, IndependentSubstrateMappingDevelopmentAssessment, IndependentSubstrateTargetPredictionContract, IndependentSubstrateTargetKind, select_mapping_candidate, select_minimal_denominator
from empirical_lawhood.adapters.methods.independent_substrate_comparators import IndependentSubstrateCategoricalForecastPanel, IndependentSubstrateForecastPanelPhase
from empirical_lawhood.adapters.methods.structural_bootstrap_inputs import StructuralBootstrapInputCensus
from empirical_lawhood.adapters.methods.structural_recurrence_runtime import MarginStructuralRecurrenceForecastPredictionRequest
from empirical_lawhood.adapters.methods.action_fiber_structural_recurrence import ActionFiberSignature, PolicySafetySignature, build_policy_signature
from empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast import MarginStructuralRecurrenceForecastConformance, MarginStructuralRecurrenceForecastMethodFreeze
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceActionOutcome, StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetStage
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .design import FreeGsnkeActionDesign
from .forecasting import FreeGsnkeForecastEncoderFreeze, build_freegsnke_forecast_panel, fit_freegsnke_comparator_encodings
from .structural_recurrence_bridge import FreeGsnkeStructuralRecurrenceBridgeFreeze, FreeGsnkeStructuralRecurrenceDesignCandidate, lower_freegsnke_phase_to_structural_recurrence
from .target_analysis import FreeGsnkePhaseReduction, FreeGsnkeTargetReductionConfig
from .target_design import FREEGSNKE_LAW_OPERAND_IDS, FreeGsnkeForecastAlphabetFreeze, FreeGsnkeMetricTopologyDesign, FreeGsnkePowerFreeze, FreeGsnkeTargetDesignFreeze
from .target_metric_topology import FreeGsnkeMetricTopologyReferenceFreeze, freeze_freegsnke_metric_topology_reference


def _digest_ids(values: tuple[str, ...]) -> str:
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


def _policy_fibers(policy: PolicySafetySignature) -> tuple[ActionFiberSignature, ...]:
    return (*policy.action_fibers, policy.hold_viability.hold_fiber)


def _outcome_by_action(
    evidence: StructuralRecurrenceStageEvidence,
) -> dict[str, tuple[StructuralRecurrenceActionOutcome, ...]]:
    action_ids = evidence.units[0].outcomes
    return {
        exemplar.action_id: tuple(
            next(outcome for outcome in unit.outcomes if outcome.action_id == exemplar.action_id)
            for unit in evidence.units
        )
        for exemplar in action_ids
    }


def _native_unit_admitted(outcome: StructuralRecurrenceActionOutcome) -> bool:
    "Target-local unit safety predicate used only for mapping comparison.\n\n    This is deliberately stricter than the panel-level Wilson decision: one\n    unsafe unit remains an unsafe false admission even when the panel rule can\n    tolerate failures.  Numeric thresholds remain owned by the frozen structural recurrence\n    design and are reflected in the outcome's fact-derived booleans/margins.\n    "

    return all(
        (
            outcome.realization_error == 0,
            outcome.sink_margin >= 0,
            outcome.support_preserved,
            outcome.validity_passed,
            outcome.preservation_passed,
            outcome.dynamics_passed,
            outcome.reachability_passed,
            outcome.authority_passed,
            outcome.uncertainty_evaluable,
        )
    )


def _unsafe_and_mismatch_counts(
    *,
    policy: PolicySafetySignature,
    evidence: StructuralRecurrenceStageEvidence,
) -> tuple[int, int, int]:
    by_action = _outcome_by_action(evidence)
    unsafe = 0
    mismatch = 0
    admitted_action_count = 0
    for fiber in _policy_fibers(policy):
        outcomes = by_action[fiber.action_id]
        native = tuple(_native_unit_admitted(value) for value in outcomes)
        if fiber.admitted:
            admitted_action_count += 1
            unsafe += sum(not value for value in native)
        mismatch += sum(fiber.admitted != value for value in native)
    cardinality = max(1, admitted_action_count) * evidence.independent_unit_count
    return unsafe, mismatch, cardinality


def _policy_category(policy: PolicySafetySignature) -> tuple[object, ...]:
    return (
        policy.denominator_structure,
        tuple(value.category_key for value in _policy_fibers(policy)),
        policy.policy_branch,
        policy.selected_action_id,
        policy.primary_reason,
    )


@dataclass(frozen=True, slots=True)
class FreeGsnkeDevelopmentSelection(CanonicalRecord):
    'Auditable G4 result before structural recurrence prediction issue or evaluator access.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-development-selection'

    selection_id: str
    target_design: ObjectIdentity
    structural_recurrence_bridge: ObjectIdentity
    development_reduction: ObjectIdentity
    mapping_development_assessments: tuple[IndependentSubstrateMappingDevelopmentAssessment, ...]
    denominator_development_assessments: tuple[IndependentSubstrateDenominatorDevelopmentAssessment, ...]
    selected_mapping_candidate_id: str
    selected_denominator_candidate_id: str
    selected_design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate
    selected_structural_recurrence_policy: PolicySafetySignature
    structural_recurrence_development_evidence: StructuralRecurrenceStageEvidence
    comparator_encodings: tuple[IndependentSubstrateComparatorEncoding, ...]
    selected_claimed_law_operand_ids: tuple[str, ...]
    selected_unsupported_law_operand_ids: tuple[str, ...]
    same_complete_unit_ids_sha256: str
    frozen_before_prediction_issue: bool
    evaluation_outcome_access_count: int

    def __post_init__(self) -> None:
        for name in (
            "selection_id",
            "selected_mapping_candidate_id",
            "selected_denominator_candidate_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.target_design.object_schema != FreeGsnkeTargetDesignFreeze.SCHEMA:
            raise ValueError("FreeGSNKE selection target design differs")
        if self.structural_recurrence_bridge.object_schema != FreeGsnkeStructuralRecurrenceBridgeFreeze.SCHEMA:
            raise ValueError('FreeGSNKE selection structural recurrence bridge differs')
        if self.development_reduction.object_schema != FreeGsnkePhaseReduction.SCHEMA:
            raise ValueError("FreeGSNKE selection development reduction differs")
        validate_sha256(
            self.same_complete_unit_ids_sha256,
            field_name="same_complete_unit_ids_sha256",
        )
        require_sorted_unique_ids(
            self.mapping_development_assessments,
            attribute="assessment_id",
            field_name="mapping_development_assessments",
        )
        require_sorted_unique_ids(
            self.denominator_development_assessments,
            attribute="assessment_id",
            field_name="denominator_development_assessments",
        )
        selected_mapping = select_mapping_candidate(
            self.mapping_development_assessments,
            maximum_candidates=32,
        )
        selected_denominator = select_minimal_denominator(self.denominator_development_assessments)
        if (
            selected_mapping.candidate.candidate_id != self.selected_mapping_candidate_id
            or selected_denominator.candidate.candidate_id != self.selected_denominator_candidate_id
        ):
            raise ValueError("FreeGSNKE selection differs from outcome-blind scientific grammar precedence")
        if (
            self.selected_design_candidate.mapping_candidate_id
            != self.selected_mapping_candidate_id
            or self.selected_design_candidate.denominator_candidate_id
            != self.selected_denominator_candidate_id
        ):
            raise ValueError('FreeGSNKE selected structural recurrence design differs')
        mapping_units_changed = any(
            value.same_complete_unit_ids_sha256 != self.same_complete_unit_ids_sha256
            for value in self.mapping_development_assessments
        )
        denominator_units_changed = any(
            value.same_complete_unit_ids_sha256 != self.same_complete_unit_ids_sha256
            for value in self.denominator_development_assessments
        )
        if mapping_units_changed or denominator_units_changed:
            raise ValueError("FreeGSNKE selection changed complete development units")
        evidence = self.structural_recurrence_development_evidence
        if (
            evidence.stage is not StructuralRecurrenceTargetStage.DEVELOPMENT
            or evidence.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or evidence.target_design
            != ObjectIdentity.from_record(
                self.selected_design_candidate.design.design_id,
                self.selected_design_candidate.design,
            )
        ):
            raise ValueError('FreeGSNKE selection structural recurrence development evidence differs')
        policy, _, _ = build_policy_signature(
            design=self.selected_design_candidate.design,
            evidence=evidence,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
        if policy != self.selected_structural_recurrence_policy:
            raise ValueError("FreeGSNKE selected policy is not evidence-derived")
        require_sorted_unique_ids(
            self.comparator_encodings,
            attribute="encoding_id",
            field_name="comparator_encodings",
        )
        if {value.kind for value in self.comparator_encodings} != set(IndependentSubstrateComparatorKind):
            raise ValueError("FreeGSNKE selection comparator roster differs")
        for name in (
            "selected_claimed_law_operand_ids",
            "selected_unsupported_law_operand_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        claimed = set(self.selected_claimed_law_operand_ids)
        unsupported = set(self.selected_unsupported_law_operand_ids)
        if (
            claimed & unsupported
            or claimed | unsupported != set(FREEGSNKE_LAW_OPERAND_IDS)
            or (
                "D" in claimed
                and not self.selected_design_candidate.denominator_exchange_claim_bearing
            )
        ):
            raise ValueError("FreeGSNKE selected law-operand ceiling differs")
        if not self.frozen_before_prediction_issue or self.evaluation_outcome_access_count:
            raise ValueError("FreeGSNKE development selection crossed prediction/reveal")


@dataclass(frozen=True, slots=True)
class FreeGsnkeDevelopmentForecastFreeze(CanonicalRecord):
    """G4 binding from finite selection to its fitted outcome-blind scientific grammar forecast codebook."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-development-forecast-freeze'

    freeze_id: str
    selection: FreeGsnkeDevelopmentSelection
    target_encoder: ObjectIdentity
    forecast_alphabet: ObjectIdentity
    development_panel: IndependentSubstrateCategoricalForecastPanel
    comparator_encodings: tuple[IndependentSubstrateComparatorEncoding, ...]
    metric_topology_reference: FreeGsnkeMetricTopologyReferenceFreeze
    frozen_before_prediction_issue: bool
    evaluation_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        if self.target_encoder.object_schema != FreeGsnkeForecastEncoderFreeze.SCHEMA:
            raise ValueError("FreeGSNKE development forecast encoder differs")
        if self.forecast_alphabet.object_schema != FreeGsnkeForecastAlphabetFreeze.SCHEMA:
            raise ValueError("FreeGSNKE development forecast alphabet differs")
        panel = self.development_panel
        if (
            panel.phase is not IndependentSubstrateForecastPanelPhase.DEVELOPMENT
            or panel.complete_unit_ids_sha256 != self.selection.same_complete_unit_ids_sha256
        ):
            raise ValueError("FreeGSNKE development forecast panel/units differ")
        require_sorted_unique_ids(
            self.comparator_encodings,
            attribute="encoding_id",
            field_name="comparator_encodings",
        )
        if self.comparator_encodings != self.selection.comparator_encodings or {
            value.kind for value in self.comparator_encodings
        } != set(IndependentSubstrateComparatorKind):
            raise ValueError("FreeGSNKE development forecast comparator freeze differs")
        if self.metric_topology_reference.development_reduction != (
            self.selection.development_reduction
        ):
            raise ValueError("FreeGSNKE metric reference development evidence differs")
        if not self.frozen_before_prediction_issue or self.evaluation_outcome_access_count:
            raise ValueError("FreeGSNKE development forecast freeze crossed prediction/reveal")


def assess_freegsnke_development_candidates(
    *,
    target_design: FreeGsnkeTargetDesignFreeze,
    bridge: FreeGsnkeStructuralRecurrenceBridgeFreeze,
    reduction_config: FreeGsnkeTargetReductionConfig,
    development_reduction: FreeGsnkePhaseReduction,
    action_design: FreeGsnkeActionDesign,
    execution_authority_verified: bool,
) -> tuple[
    tuple[IndependentSubstrateMappingDevelopmentAssessment, ...],
    tuple[IndependentSubstrateDenominatorDevelopmentAssessment, ...],
]:
    """Assess the exact finite G2 lattice on one unchanged development panel."""

    target_identity = ObjectIdentity.from_record(target_design.freeze_id, target_design)
    if bridge.target_design != target_identity:
        raise ValueError("FreeGSNKE development bridge/target design differs")
    mapping_by_id = {value.candidate_id: value for value in target_design.mapping_candidate_roster}
    denominator_by_id = {
        value.candidate_id: value for value in target_design.denominator_candidate_lattice
    }
    if set(mapping_by_id) != set(bridge.mapping_candidate_ids):
        raise ValueError("FreeGSNKE development mapping roster differs")
    eligible_denominator_ids = {
        candidate_id
        for candidate_id, candidate in denominator_by_id.items()
        if not candidate.impossible
    }
    if eligible_denominator_ids != set(bridge.denominator_candidate_ids):
        raise ValueError("FreeGSNKE development denominator lattice differs")
    same_units = development_reduction.issued_unit_ids_sha256

    mapping_values = []
    mapping_evidence: dict[str, StructuralRecurrenceStageEvidence] = {}
    mapping_policies: dict[str, PolicySafetySignature] = {}
    for mapping_id in sorted(mapping_by_id):
        design_candidate = bridge.candidate(
            mapping_candidate_id=mapping_id,
            denominator_candidate_id=target_design.proposed_denominator_candidate_id,
        )
        evidence = lower_freegsnke_phase_to_structural_recurrence(
            bridge=bridge,
            design_candidate=design_candidate,
            reduction_config=reduction_config,
            phase_reduction=development_reduction,
            action_design=action_design,
            execution_authority_verified=execution_authority_verified,
        )
        policy, _, _ = build_policy_signature(
            design=design_candidate.design,
            evidence=evidence,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
        unsafe, mismatch, cardinality = _unsafe_and_mismatch_counts(
            policy=policy,
            evidence=evidence,
        )
        mapping_values.append(
            IndependentSubstrateMappingDevelopmentAssessment(
                assessment_id=f"assessment.freegsnke.mapping.{mapping_id}",
                candidate=mapping_by_id[mapping_id],
                same_complete_unit_ids_sha256=same_units,
                unsafe_false_admission_count=unsafe,
                categorical_mismatch_count=mismatch,
                prediction_set_cardinality=cardinality,
                evaluation_outcome_count=0,
            )
        )
        mapping_evidence[mapping_id] = evidence
        mapping_policies[mapping_id] = policy
    mapping_assessments = tuple(sorted(mapping_values, key=lambda value: value.assessment_id))
    selected_mapping_id = select_mapping_candidate(
        mapping_assessments,
        maximum_candidates=32,
    ).candidate.candidate_id

    proposed_candidate = bridge.candidate(
        mapping_candidate_id=selected_mapping_id,
        denominator_candidate_id=target_design.proposed_denominator_candidate_id,
    )
    proposed_evidence = mapping_evidence[selected_mapping_id]
    proposed_policy = mapping_policies[selected_mapping_id]
    if proposed_evidence.target_design != ObjectIdentity.from_record(
        proposed_candidate.design.design_id,
        proposed_candidate.design,
    ):
        raise ValueError("FreeGSNKE proposed mapping/design identity differs")
    proposed_category = _policy_category(proposed_policy)

    denominator_values = []
    for denominator_id, denominator in sorted(denominator_by_id.items()):
        if denominator.impossible:
            denominator_values.append(
                IndependentSubstrateDenominatorDevelopmentAssessment(
                    assessment_id=(f"assessment.freegsnke.denominator.{denominator_id}"),
                    candidate=denominator,
                    same_complete_unit_ids_sha256=same_units,
                    legal_action_alphabet_preserved=False,
                    forecast_roster_preserved=False,
                    policy_outputs_preserved=False,
                    unsafe_error_count=0,
                    simultaneous_precision_passed=False,
                )
            )
            continue
        design_candidate = bridge.candidate(
            mapping_candidate_id=selected_mapping_id,
            denominator_candidate_id=denominator_id,
        )
        evidence = lower_freegsnke_phase_to_structural_recurrence(
            bridge=bridge,
            design_candidate=design_candidate,
            reduction_config=reduction_config,
            phase_reduction=development_reduction,
            action_design=action_design,
            execution_authority_verified=execution_authority_verified,
        )
        policy, _, _ = build_policy_signature(
            design=design_candidate.design,
            evidence=evidence,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
        unsafe, _, _ = _unsafe_and_mismatch_counts(policy=policy, evidence=evidence)
        expected_actions = tuple(
            sorted(value.structural_recurrence_action_id for value in design_candidate.action_bindings)
        )
        observed_actions = tuple(sorted(value.action_id for value in evidence.units[0].outcomes))
        complete_roster = all(
            tuple(sorted(value.action_id for value in unit.outcomes)) == expected_actions
            for unit in evidence.units
        )
        precision = all(
            outcome.uncertainty_evaluable for unit in evidence.units for outcome in unit.outcomes
        )
        denominator_values.append(
            IndependentSubstrateDenominatorDevelopmentAssessment(
                assessment_id=f"assessment.freegsnke.denominator.{denominator_id}",
                candidate=denominator,
                same_complete_unit_ids_sha256=same_units,
                legal_action_alphabet_preserved=(
                    observed_actions == expected_actions and complete_roster
                ),
                forecast_roster_preserved=(
                    evidence.independent_unit_count == development_reduction.issued_unit_count
                ),
                policy_outputs_preserved=(_policy_category(policy) == proposed_category),
                unsafe_error_count=unsafe,
                simultaneous_precision_passed=(
                    precision and not development_reduction.panel_envelope_limited
                ),
            )
        )
    return mapping_assessments, tuple(
        sorted(denominator_values, key=lambda value: value.assessment_id)
    )


def freeze_freegsnke_development_selection(
    *,
    selection_id: str,
    target_design: FreeGsnkeTargetDesignFreeze,
    bridge: FreeGsnkeStructuralRecurrenceBridgeFreeze,
    reduction_config: FreeGsnkeTargetReductionConfig,
    development_reduction: FreeGsnkePhaseReduction,
    action_design: FreeGsnkeActionDesign,
    metric_topology_design: FreeGsnkeMetricTopologyDesign,
    comparator_encodings: tuple[IndependentSubstrateComparatorEncoding, ...],
    execution_authority_verified: bool,
) -> FreeGsnkeDevelopmentSelection:
    'Select and freeze one existing structural recurrence design without evaluation rescue.'

    mapping_assessments, denominator_assessments = assess_freegsnke_development_candidates(
        target_design=target_design,
        bridge=bridge,
        reduction_config=reduction_config,
        development_reduction=development_reduction,
        action_design=action_design,
        execution_authority_verified=execution_authority_verified,
    )
    selected_mapping_id = select_mapping_candidate(
        mapping_assessments,
        maximum_candidates=32,
    ).candidate.candidate_id
    selected_denominator_id = select_minimal_denominator(
        denominator_assessments
    ).candidate.candidate_id
    selected_design = bridge.candidate(
        mapping_candidate_id=selected_mapping_id,
        denominator_candidate_id=selected_denominator_id,
    )
    evidence = lower_freegsnke_phase_to_structural_recurrence(
        bridge=bridge,
        design_candidate=selected_design,
        reduction_config=reduction_config,
        phase_reduction=development_reduction,
        action_design=action_design,
        execution_authority_verified=execution_authority_verified,
    )
    policy, _, _ = build_policy_signature(
        design=selected_design.design,
        evidence=evidence,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    claimed = set(metric_topology_design.claimed_law_operand_ids)
    if not selected_design.denominator_exchange_claim_bearing:
        claimed.discard("D")
    unsupported = set(FREEGSNKE_LAW_OPERAND_IDS) - claimed
    return FreeGsnkeDevelopmentSelection(
        selection_id=selection_id,
        target_design=ObjectIdentity.from_record(target_design.freeze_id, target_design),
        structural_recurrence_bridge=ObjectIdentity.from_record(bridge.freeze_id, bridge),
        development_reduction=ObjectIdentity.from_record(
            development_reduction.reduction_id,
            development_reduction,
        ),
        mapping_development_assessments=mapping_assessments,
        denominator_development_assessments=denominator_assessments,
        selected_mapping_candidate_id=selected_mapping_id,
        selected_denominator_candidate_id=selected_denominator_id,
        selected_design_candidate=selected_design,
        selected_structural_recurrence_policy=policy,
        structural_recurrence_development_evidence=evidence,
        comparator_encodings=comparator_encodings,
        selected_claimed_law_operand_ids=tuple(sorted(claimed)),
        selected_unsupported_law_operand_ids=tuple(sorted(unsupported)),
        same_complete_unit_ids_sha256=(development_reduction.issued_unit_ids_sha256),
        frozen_before_prediction_issue=True,
        evaluation_outcome_access_count=0,
    )


def freeze_freegsnke_development_forecasts(
    *,
    freeze_id: str,
    selection_id: str,
    target_design: FreeGsnkeTargetDesignFreeze,
    bridge: FreeGsnkeStructuralRecurrenceBridgeFreeze,
    reduction_config: FreeGsnkeTargetReductionConfig,
    development_reduction: FreeGsnkePhaseReduction,
    action_design: FreeGsnkeActionDesign,
    metric_topology_design: FreeGsnkeMetricTopologyDesign,
    encoder: FreeGsnkeForecastEncoderFreeze,
    forecast_grammar: IndependentSubstrateForecastAlphabetGrammar,
    forecast_alphabet: FreeGsnkeForecastAlphabetFreeze,
    execution_authority_verified: bool,
) -> FreeGsnkeDevelopmentForecastFreeze:
    """Execute the exact G4 selection/panel/comparator order before issue."""

    mapping_assessments, denominator_assessments = assess_freegsnke_development_candidates(
        target_design=target_design,
        bridge=bridge,
        reduction_config=reduction_config,
        development_reduction=development_reduction,
        action_design=action_design,
        execution_authority_verified=execution_authority_verified,
    )
    mapping_id = select_mapping_candidate(
        mapping_assessments,
        maximum_candidates=32,
    ).candidate.candidate_id
    denominator_id = select_minimal_denominator(denominator_assessments).candidate.candidate_id
    selected_design = bridge.candidate(
        mapping_candidate_id=mapping_id,
        denominator_candidate_id=denominator_id,
    )
    evidence = lower_freegsnke_phase_to_structural_recurrence(
        bridge=bridge,
        design_candidate=selected_design,
        reduction_config=reduction_config,
        phase_reduction=development_reduction,
        action_design=action_design,
        execution_authority_verified=execution_authority_verified,
    )
    panel = build_freegsnke_forecast_panel(
        panel_id=f"panel.{freeze_id}",
        encoder=encoder,
        grammar=forecast_grammar,
        alphabet=forecast_alphabet,
        design_candidate=selected_design,
        phase_reduction=development_reduction,
        structural_recurrence_evidence=evidence,
        evaluator_reveal_authorized=False,
    )
    encodings = fit_freegsnke_comparator_encodings(
        namespace_id=freeze_id,
        encoder=encoder,
        development_panel=panel,
        selected_mapping_candidate_id=mapping_id,
        selected_denominator_candidate_id=denominator_id,
    )
    selection = freeze_freegsnke_development_selection(
        selection_id=selection_id,
        target_design=target_design,
        bridge=bridge,
        reduction_config=reduction_config,
        development_reduction=development_reduction,
        action_design=action_design,
        metric_topology_design=metric_topology_design,
        comparator_encodings=encodings,
        execution_authority_verified=execution_authority_verified,
    )
    if (
        selection.selected_mapping_candidate_id != mapping_id
        or selection.selected_denominator_candidate_id != denominator_id
        or selection.structural_recurrence_development_evidence != evidence
    ):
        raise ValueError("FreeGSNKE G4 selection changed while freezing forecasts")
    metric_reference = freeze_freegsnke_metric_topology_reference(
        freeze_id=f"metric-topology-reference.{freeze_id}",
        design=metric_topology_design,
        development_reduction=development_reduction,
    )
    return FreeGsnkeDevelopmentForecastFreeze(
        freeze_id=freeze_id,
        selection=selection,
        target_encoder=ObjectIdentity.from_record(encoder.encoder_id, encoder),
        forecast_alphabet=ObjectIdentity.from_record(
            forecast_alphabet.freeze_id,
            forecast_alphabet,
        ),
        development_panel=panel,
        comparator_encodings=encodings,
        metric_topology_reference=metric_reference,
        frozen_before_prediction_issue=True,
        evaluation_outcome_access_count=0,
    )


def build_freegsnke_target_prediction_contract(
    *,
    contract_id: str,
    target_design: FreeGsnkeTargetDesignFreeze,
    forecast_alphabet: FreeGsnkeForecastAlphabetFreeze,
    power_freeze: FreeGsnkePowerFreeze,
    metric_topology_design: FreeGsnkeMetricTopologyDesign,
    independence_dossier: IndependentImplementationDossier,
    development_forecast_freeze: FreeGsnkeDevelopmentForecastFreeze,
) -> IndependentSubstrateTargetPredictionContract:
    """Bind the selected G4 result into the common immutable target contract."""

    selection = development_forecast_freeze.selection
    target_identity = ObjectIdentity.from_record(target_design.freeze_id, target_design)
    if selection.target_design != target_identity:
        raise ValueError("FreeGSNKE target contract selection/design differs")
    expected_identities = (
        (
            target_design.forecast_alphabet,
            ObjectIdentity.from_record(forecast_alphabet.freeze_id, forecast_alphabet),
            "forecast alphabet",
        ),
        (
            target_design.power_freeze,
            ObjectIdentity.from_record(power_freeze.freeze_id, power_freeze),
            "power freeze",
        ),
        (
            target_design.metric_topology_design,
            ObjectIdentity.from_record(
                metric_topology_design.design_id,
                metric_topology_design,
            ),
            "metric/topology design",
        ),
    )
    for frozen, actual, label in expected_identities:
        if frozen != actual:
            raise ValueError(f"FreeGSNKE target contract {label} differs")
    if development_forecast_freeze.forecast_alphabet != ObjectIdentity.from_record(
        forecast_alphabet.freeze_id,
        forecast_alphabet,
    ):
        raise ValueError("FreeGSNKE target contract development forecast differs")
    if (
        development_forecast_freeze.metric_topology_reference.metric_topology_design
        != ObjectIdentity.from_record(
            metric_topology_design.design_id,
            metric_topology_design,
        )
        or not development_forecast_freeze.metric_topology_reference.issue_eligible
    ):
        raise ValueError("FreeGSNKE target contract metric/topology reference is ineligible")
    source_and_action_bindings = (
        (
            forecast_alphabet.source_binding,
            target_design.source_binding,
            "forecast source",
        ),
        (
            forecast_alphabet.action_design,
            target_design.action_design,
            "forecast action design",
        ),
        (power_freeze.source_binding, target_design.source_binding, "power source"),
        (power_freeze.action_design, target_design.action_design, "power action design"),
        (
            metric_topology_design.source_binding,
            target_design.source_binding,
            "metric/topology source",
        ),
        (
            metric_topology_design.action_design,
            target_design.action_design,
            "metric/topology action design",
        ),
    )
    for actual, expected, label in source_and_action_bindings:
        if actual != expected:
            raise ValueError(f"FreeGSNKE target contract {label} differs")
    evidence_unit_ids = tuple(value.unit_id for value in selection.structural_recurrence_development_evidence.units)
    if (
        evidence_unit_ids != power_freeze.development_unit_ids
        or _digest_ids(power_freeze.development_unit_ids) != selection.same_complete_unit_ids_sha256
    ):
        raise ValueError("FreeGSNKE target contract development power roster differs")
    if (
        independence_dossier.slot is not IndependentSubstrateTargetKind.FREEGSNKE
        or not power_freeze.issue_eligible
    ):
        raise ValueError("FreeGSNKE target contract is not G3 issue eligible")
    selected_design = selection.selected_design_candidate.design
    evidence = selection.structural_recurrence_development_evidence
    return IndependentSubstrateTargetPredictionContract(
        contract_id=contract_id,
        target_slot=IndependentSubstrateTargetKind.FREEGSNKE,
        scientific_design_basis=target_design.scientific_design_basis,
        source_qualification=target_design.source_qualification,
        independence_dossier=independence_dossier,
        mapping_candidate_roster=target_design.mapping_candidate_roster,
        mapping_development_assessments=(selection.mapping_development_assessments),
        selected_mapping_candidate_id=selection.selected_mapping_candidate_id,
        denominator_candidate_lattice=target_design.denominator_candidate_lattice,
        denominator_development_assessments=(selection.denominator_development_assessments),
        selected_denominator_candidate_id=(selection.selected_denominator_candidate_id),
        selected_claimed_law_operand_ids=(selection.selected_claimed_law_operand_ids),
        selected_unsupported_law_operand_ids=(selection.selected_unsupported_law_operand_ids),
        proposed_denominator_candidate_id=(target_design.proposed_denominator_candidate_id),
        meaningful_split_candidate_ids=target_design.meaningful_split_candidate_ids,
        merge_or_omission_candidate_ids=(target_design.merge_or_omission_candidate_ids),
        impossible_candidate_ids=target_design.impossible_candidate_ids,
        target_forecast_alphabet=target_design.forecast_alphabet,
        comparator_encodings=development_forecast_freeze.comparator_encodings,
        target_power_freeze=target_design.power_freeze,
        metric_topology_design=target_design.metric_topology_design,
        target_development_evidence=selection.development_reduction,
        structural_recurrence_target_design=ObjectIdentity.from_record(
            selected_design.design_id,
            selected_design,
        ),
        structural_recurrence_development_evidence=ObjectIdentity.from_record(
            evidence.evidence_id,
            evidence,
        ),
        same_complete_unit_ids_sha256=(selection.same_complete_unit_ids_sha256),
        frozen_before_prediction_issue=True,
        published_before_evaluator_access=True,
        evaluation_outcome_access_count=0,
    )


def build_freegsnke_structural_recurrence_prediction_request(
    *,
    request_id: str,
    issue_id: str,
    method_freeze: MarginStructuralRecurrenceForecastMethodFreeze,
    conformance: MarginStructuralRecurrenceForecastConformance,
    power_freeze: FreeGsnkePowerFreeze,
    development_forecast_freeze: FreeGsnkeDevelopmentForecastFreeze,
    contract: IndependentSubstrateTargetPredictionContract,
    scientific_bootstrap_inputs: StructuralBootstrapInputCensus,
) -> MarginStructuralRecurrenceForecastPredictionRequest:
    'Construct the unchanged registered margin structural recurrence forecast prediction request.'

    selection = development_forecast_freeze.selection
    design = selection.selected_design_candidate.design
    evidence = selection.structural_recurrence_development_evidence
    if (
        contract.structural_recurrence_target_design != ObjectIdentity.from_record(design.design_id, design)
        or contract.structural_recurrence_development_evidence
        != ObjectIdentity.from_record(evidence.evidence_id, evidence)
        or contract.target_power_freeze
        != ObjectIdentity.from_record(power_freeze.freeze_id, power_freeze)
        or contract.same_complete_unit_ids_sha256 != selection.same_complete_unit_ids_sha256
        or contract.comparator_encodings != development_forecast_freeze.comparator_encodings
        or not power_freeze.issue_eligible
    ):
        raise ValueError('FreeGSNKE structural recurrence request changed its target prediction contract')
    return MarginStructuralRecurrenceForecastPredictionRequest(
        request_id=request_id,
        issue_id=issue_id,
        method_freeze=method_freeze,
        conformance=conformance,
        design=design,
        development=evidence,
        evaluation_count=len(power_freeze.evaluation_unit_ids),
        prospective_validation_count=len(power_freeze.prospective_validation_unit_ids),
        scientific_bootstrap_inputs=scientific_bootstrap_inputs,
    )


__all__ = [
    'FreeGsnkeDevelopmentForecastFreeze',
    'FreeGsnkeDevelopmentSelection',
    "assess_freegsnke_development_candidates",
    'build_freegsnke_structural_recurrence_prediction_request',
    "build_freegsnke_target_prediction_contract",
    "freeze_freegsnke_development_selection",
    "freeze_freegsnke_development_forecasts",
]
