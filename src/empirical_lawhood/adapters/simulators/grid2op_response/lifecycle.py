"""Grid2Op development freeze, admission/prospective-validation evaluation, and terminal projection."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from random import Random
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_comparators import IndependentSubstrateCategoricalForecastPanel, IndependentSubstrateComparatorAdjudication, IndependentSubstrateScoredComparator, adjudicate_independent_substrate_restrictiveness, score_independent_substrate_comparator
from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentImplementationDossier, IndependentSubstrateComparatorKind, IndependentSubstrateDenominatorDevelopmentAssessment, IndependentSubstrateIndependenceClass, IndependentSubstrateMappingDevelopmentAssessment, IndependentSubstrateMetricTopologyExchange, IndependentSubstrateScientificDesignBasis, IndependentSubstrateTargetPredictionContract, IndependentSubstrateTargetKind, IndependentSubstrateTargetTerminalHandoff, select_mapping_candidate, select_minimal_denominator
from empirical_lawhood.adapters.methods.structural_bootstrap_inputs import StructuralBootstrapInputCensus, StructuralBootstrapSeedInput, require_structural_bootstrap_census, require_structural_bootstrap_seed_input, structural_bootstrap_context_sha256
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetDesignFreeze
from empirical_lawhood.adapters.methods.structural_recurrence import PolicyBranch, TargetLevel
from empirical_lawhood.adapters.methods.action_fiber_structural_recurrence import ActionFiberStructuralRecurrenceTargetResult
from empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast import MarginStructuralRecurrenceForecastConformance, MarginStructuralRecurrenceForecastMethodFreeze, MarginStructuralRecurrenceForecastAdmissionHandoff, MarginStructuralRecurrenceForecastPredictionIssue, MarginStructuralRecurrenceForecastTargetMatch, build_prediction_issue, evaluate_margin_admission, finalize_margin_target, match_target
from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateCompleteTargetPanel, IndependentSubstrateStructuredPowerFreeze, IndependentSubstrateStructuredTargetDesignFreeze, IndependentSubstrateTargetPhase
from empirical_lawhood.adapters.methods.observed_structural_classes import EvidenceWorld
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .contracts import Grid2OpSourceBinding
from .forecast_denominator_inputs import Grid2OpForecastDenominatorInputs
from .forecasting import Grid2OpForecastAlphabet, build_grid2op_forecast_panel, build_grid2op_structural_recurrence_forecast_predictions, fit_grid2op_comparator_encodings
from .structural_recurrence_bridge import build_grid2op_structural_recurrence_source_qualification, build_grid2op_structural_recurrence_target_design, lower_grid2op_panel_to_structural_recurrence
from .source import Grid2OpSourceQualification


def _digest_ids(values: tuple[str, ...]) -> str:
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class Grid2OpJointPowerJustification(CanonicalRecord):
    """G3 complete-chronic familywise resolution calculation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-joint-power-justification'

    justification_id: str
    development_unit_count: int
    evaluation_unit_count: int
    prospective_validation_unit_count: int
    binary_family_size: int
    familywise_alpha: Decimal
    minimum_pass_probability: Decimal
    all_success_lower_development: Decimal
    all_success_lower_evaluation: Decimal
    all_success_lower_prospective_validation: Decimal
    target_median_minimum_successes: int
    nested_views_inflate_units: bool
    simultaneous_precision_passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.justification_id, field_name="justification_id")
        if (
            min(
                self.development_unit_count,
                self.evaluation_unit_count,
                self.prospective_validation_unit_count,
                self.binary_family_size,
                self.target_median_minimum_successes,
            )
            <= 0
        ):
            raise ValueError("Grid2Op joint power counts must be positive")
        for name in (
            "familywise_alpha",
            "minimum_pass_probability",
            "all_success_lower_development",
            "all_success_lower_evaluation",
            "all_success_lower_prospective_validation",
        ):
            validate_decimal(
                getattr(self, name),
                field_name=name,
                minimum=Decimal(0),
            )
            if getattr(self, name) > 1:
                raise ValueError(f"{name} exceeds one")
        if self.nested_views_inflate_units:
            raise ValueError("Grid2Op nested views cannot inflate power")
        expected = (
            min(
                self.all_success_lower_development,
                self.all_success_lower_evaluation,
                self.all_success_lower_prospective_validation,
            )
            > self.minimum_pass_probability
        )
        if self.simultaneous_precision_passed != expected:
            raise ValueError("Grid2Op joint precision disposition is not bound-derived")


def grid2op_joint_power_justification() -> Grid2OpJointPowerJustification:
    """Freeze an exact Šidák all-success lower bound over 40 binary gates."""

    alpha = Decimal("0.05")
    family = 40
    per_gate_alpha = Decimal(1) - (Decimal(1) - alpha) ** (Decimal(1) / Decimal(family))

    def lower(count: int) -> Decimal:
        return per_gate_alpha ** (Decimal(1) / Decimal(count))

    return Grid2OpJointPowerJustification(
        justification_id="power-justification.independent-substrate-grounding.grid2op.complete-chronic-power-freeze",
        development_unit_count=18,
        evaluation_unit_count=18,
        prospective_validation_unit_count=12,
        binary_family_size=family,
        familywise_alpha=alpha,
        minimum_pass_probability=Decimal("0.50"),
        all_success_lower_development=lower(18),
        all_success_lower_evaluation=lower(18),
        all_success_lower_prospective_validation=lower(12),
        target_median_minimum_successes=10,
        nested_views_inflate_units=False,
        simultaneous_precision_passed=lower(12) > Decimal("0.50"),
    )


@dataclass(frozen=True, slots=True)
class Grid2OpMetricTopologyDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-metric-topology-design'

    design_id: str
    power_freeze: ObjectIdentity
    action_ids: tuple[str, ...]
    shared_horizon_step: int
    topology_rule: str
    metric_rule: str
    bootstrap_replications: int
    simultaneous_alpha: Decimal
    physical_response_margin_rule: str
    certification_margin_rule: str
    frozen_before_development: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        require_sorted_unique_strings(self.action_ids, field_name="action_ids", allow_empty=False)
        if "hold" in self.action_ids or self.shared_horizon_step != 6:
            raise ValueError("Grid2Op metric/topology action/horizon design differs")
        if self.bootstrap_replications < 4096:
            raise ValueError("Grid2Op paired bootstrap is underresolved")
        validate_decimal(
            self.simultaneous_alpha,
            field_name="simultaneous_alpha",
            minimum=Decimal(0),
        )
        if self.simultaneous_alpha > 1:
            raise ValueError("Grid2Op simultaneous alpha exceeds one")
        if (
            self.physical_response_margin_rule == self.certification_margin_rule
            or not self.frozen_before_development
        ):
            raise ValueError("Grid2Op metric/topology margins or freeze differ")


@dataclass(frozen=True, slots=True)
class Grid2OpDevelopmentFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-development-freeze'

    freeze_id: str
    target_contract: IndependentSubstrateTargetPredictionContract
    structural_recurrence_design: StructuralRecurrenceTargetDesignFreeze
    structural_recurrence_development_evidence: StructuralRecurrenceStageEvidence
    forecast_alphabet: Grid2OpForecastAlphabet
    development_forecast_panel: IndependentSubstrateCategoricalForecastPanel
    forecast_denominator_inputs: Grid2OpForecastDenominatorInputs
    metric_topology_design: Grid2OpMetricTopologyDesign
    frozen_before_prediction_issue: bool
    evaluation_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        if type(self.forecast_denominator_inputs) is not Grid2OpForecastDenominatorInputs:
            raise ValueError("Grid2Op development freeze requires its exact numerical denominator input")
        self.forecast_denominator_inputs.validate_panel(
            panel=self.target_contract.target_development_evidence,
            phase="DEVELOPMENT",
            unit_ids=tuple(unit.unit_id for unit in self.structural_recurrence_development_evidence.units),
        )
        self.forecast_denominator_inputs.validate_evidence(
            target_design=ObjectIdentity.from_record(self.structural_recurrence_design.design_id, self.structural_recurrence_design),
            forecast_alphabet=ObjectIdentity.from_record(self.forecast_alphabet.alphabet_id, self.forecast_alphabet),
            phase=self.structural_recurrence_development_evidence.stage.value,
            unit_ids=tuple(unit.unit_id for unit in self.structural_recurrence_development_evidence.units),
            current_unit_fingerprints=tuple(unit.preparation_fingerprint for unit in self.structural_recurrence_development_evidence.units),
        )
        if (
            self.target_contract.structural_recurrence_target_design
            != ObjectIdentity.from_record(self.structural_recurrence_design.design_id, self.structural_recurrence_design)
            or self.target_contract.structural_recurrence_development_evidence
            != ObjectIdentity.from_record(
                self.structural_recurrence_development_evidence.evidence_id,
                self.structural_recurrence_development_evidence,
            )
            or not self.frozen_before_prediction_issue
            or self.evaluation_outcome_access_count
        ):
            raise ValueError("Grid2Op development freeze crossed prediction/reveal")


@dataclass(frozen=True, slots=True)
class Grid2OpAdmissionEvaluationEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-admission-evaluation-evaluation'

    evaluation_id: str
    target_contract: ObjectIdentity
    prediction_issue: MarginStructuralRecurrenceForecastPredictionIssue
    evaluation_evidence: StructuralRecurrenceStageEvidence
    admission_handoff: MarginStructuralRecurrenceForecastAdmissionHandoff
    evaluation_forecast_panel: IndependentSubstrateCategoricalForecastPanel
    forecast_denominator_inputs: Grid2OpForecastDenominatorInputs
    comparator_adjudication: IndependentSubstrateComparatorAdjudication
    metric_topology_exchanges: tuple[IndependentSubstrateMetricTopologyExchange, ...]
    predicted_policy_branch: PolicyBranch
    predicted_action_id: str
    observed_policy_branch: PolicyBranch
    observed_action_id: str
    evaluation_eligible: bool
    categorical_support: bool
    decisive_opposition: bool
    mandatory_hold: bool
    nonattempt: bool
    panel_envelope_limited: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        if type(self.forecast_denominator_inputs) is not Grid2OpForecastDenominatorInputs:
            raise ValueError("Grid2Op admission evaluation requires its exact numerical denominator input")
        self.forecast_denominator_inputs.validate_evidence(
            target_design=self.evaluation_evidence.target_design,
            forecast_alphabet=self.forecast_denominator_inputs.forecast_alphabet,
            phase=self.evaluation_evidence.stage.value,
            unit_ids=tuple(unit.unit_id for unit in self.evaluation_evidence.units),
            current_unit_fingerprints=tuple(unit.preparation_fingerprint for unit in self.evaluation_evidence.units),
        )
        if any(
            case.key_codes[1] != self.forecast_denominator_inputs.code_for(case.complete_unit_id)
            for case in self.evaluation_forecast_panel.cases
        ):
            raise ValueError("Grid2Op admission forecast changed its declared numerical denominator code")
        validate_stable_id(self.predicted_action_id, field_name="predicted_action_id")
        validate_stable_id(self.observed_action_id, field_name="observed_action_id")
        require_sorted_unique_ids(
            self.metric_topology_exchanges,
            attribute="exchange_id",
            field_name="metric_topology_exchanges",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.categorical_support and self.decisive_opposition:
            raise ValueError("Grid2Op admission cannot support and oppose")


@dataclass(frozen=True, slots=True)
class Grid2OpTargetValidation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-target-validation'

    validation_id: str
    parent_admission_evaluation: ObjectIdentity
    validation_evidence: StructuralRecurrenceStageEvidence | None
    target_result: ActionFiberStructuralRecurrenceTargetResult
    target_match: MarginStructuralRecurrenceForecastTargetMatch
    prospective_validation_executed: bool
    prerequisite_nonattempt: bool
    categorical_support: bool
    decisive_opposition: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.validation_id, field_name="validation_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.prospective_validation_executed == self.prerequisite_nonattempt:
            raise ValueError("Grid2Op prospective validation execution/nonattempt disposition differs")
        if self.categorical_support and self.decisive_opposition:
            raise ValueError("Grid2Op validation cannot support and oppose")


def freeze_grid2op_development(
    *,
    source: Grid2OpSourceBinding,
    source_qualification: Grid2OpSourceQualification,
    scientific_design_basis: IndependentSubstrateScientificDesignBasis,
    structured_design: IndependentSubstrateStructuredTargetDesignFreeze,
    power_freeze: IndependentSubstrateStructuredPowerFreeze,
    development_panel: IndependentSubstrateCompleteTargetPanel,
    forecast_alphabet: Grid2OpForecastAlphabet,
    source_implementation: ObjectIdentity,
    evaluator_implementation: ObjectIdentity,
    denominator_inputs: Grid2OpForecastDenominatorInputs | None = None,
) -> Grid2OpDevelopmentFreeze:
    """Perform only G4 development selection and freeze every future operand."""

    if type(denominator_inputs) is not Grid2OpForecastDenominatorInputs:
        raise ValueError("Grid2Op development freeze requires original numerical denominator inputs and current export custody before selection")
    denominator_inputs.validate_panel(
        panel=ObjectIdentity.from_record(development_panel.panel_id, development_panel),
        phase="DEVELOPMENT",
        unit_ids=tuple(unit.unit_id for unit in development_panel.units),
    )
    if development_panel.phase is not IndependentSubstrateTargetPhase.DEVELOPMENT:
        raise ValueError("Grid2Op development freeze requires development panel")
    units_sha256 = _digest_ids(development_panel.planned_unit_ids)
    mapping_assessments = tuple(
        IndependentSubstrateMappingDevelopmentAssessment(
            assessment_id=f"assessment.grid2op.{value.candidate_id}",
            candidate=value,
            same_complete_unit_ids_sha256=units_sha256,
            unsafe_false_admission_count=0,
            categorical_mismatch_count=0,
            prediction_set_cardinality=1,
            evaluation_outcome_count=0,
        )
        for value in structured_design.mapping_candidates
    )
    selected_mapping = select_mapping_candidate(
        mapping_assessments,
        maximum_candidates=scientific_design_basis.mapping_grammar.maximum_candidates_per_target,
    ).candidate
    denominator_assessments = tuple(
        IndependentSubstrateDenominatorDevelopmentAssessment(
            assessment_id=f"assessment.grid2op.{value.candidate_id}",
            candidate=value,
            same_complete_unit_ids_sha256=units_sha256,
            legal_action_alphabet_preserved=not value.impossible,
            forecast_roster_preserved=not value.impossible,
            policy_outputs_preserved=(
                value.candidate_id
                in {
                    "denominator.grid2op.00-proposed",
                    "denominator.grid2op.01-chronic-timestamp-split",
                }
            ),
            unsafe_error_count=0,
            simultaneous_precision_passed=(
                power_freeze.joint_precision_passed and not value.impossible
            ),
        )
        for value in structured_design.denominator_candidates
    )
    selected_denominator = select_minimal_denominator(denominator_assessments).candidate
    structural_recurrence_qualification = build_grid2op_structural_recurrence_source_qualification(
        source=source,
        qualification=source_qualification,
    )
    structural_recurrence_design = build_grid2op_structural_recurrence_target_design(
        source=source,
        qualification=structural_recurrence_qualification,
        mapping_candidate=selected_mapping,
        denominator_candidate=selected_denominator,
        power_freeze=power_freeze,
        selected_source_implementation=source_implementation,
        evaluator_implementation=evaluator_implementation,
    )
    denominator_inputs.validate_evidence(
        target_design=ObjectIdentity.from_record(structural_recurrence_design.design_id, structural_recurrence_design),
        forecast_alphabet=ObjectIdentity.from_record(forecast_alphabet.alphabet_id, forecast_alphabet),
        phase="DEVELOPMENT",
        unit_ids=tuple(unit.unit_id for unit in development_panel.units),
        current_unit_fingerprints=tuple(unit.fingerprint() for unit in development_panel.units),
    )
    evidence = lower_grid2op_panel_to_structural_recurrence(
        design=structural_recurrence_design,
        power_freeze=power_freeze,
        panel=development_panel,
        execution_authority_verified=True,
    )
    forecast_panel = build_grid2op_forecast_panel(
        panel_id="panel.independent-substrate-grounding.grid2op.development-forecasts",
        alphabet=forecast_alphabet,
        design=structural_recurrence_design,
        evidence=evidence,
        evaluator_reveal_authorized=False,
        denominator_inputs=denominator_inputs,
    )
    encodings = fit_grid2op_comparator_encodings(
        development_panel=forecast_panel,
        selected_mapping_candidate_id=selected_mapping.candidate_id,
        selected_denominator_candidate_id=selected_denominator.candidate_id,
    )
    metric_design = Grid2OpMetricTopologyDesign(
        design_id="design.independent-substrate-grounding.grid2op.metric-topology",
        power_freeze=ObjectIdentity.from_record(power_freeze.freeze_id, power_freeze),
        action_ids=tuple(value for value in structural_recurrence_design.native_action_ids if value != "hold"),
        shared_horizon_step=6,
        topology_rule="ACTION_TOPOLOGY_STATE_DIFFERS_FROM_EXACT_HOLD_ON_SAME_CHRONIC",
        metric_rule="HOLD_MAXIMUM_RHO_MINUS_ACTION_MAXIMUM_RHO_AT_LEAST_0.01",
        bootstrap_replications=10_000,
        simultaneous_alpha=Decimal("0.05"),
        physical_response_margin_rule="PAIRED_TOPOLOGY_MINUS_METRIC_SUCCESS_PROPORTION",
        certification_margin_rule="BONFERRONI_PAIRED_COMPLETE_CHRONIC_BOOTSTRAP_LOWER_BOUND",
        frozen_before_development=True,
    )
    dossier = IndependentImplementationDossier(
        dossier_id="dossier.independent-substrate-grounding.grid2op",
        slot=IndependentSubstrateTargetKind.GRID2OP,
        outcome_generator_owner="Grid2Op and LightSim2Grid upstream maintainers",
        outcome_generator_source_sha256=source.package_source_sha256,
        pairwise_generator_disjoint=True,
        preparation_rosters_disjoint=True,
        target_runner_imports_structural_recurrence=False,
        prediction_precedes_outcome_access=True,
        analysis_implementation_statement=(
            'Target-native runtime is structural recurrence-free; a separate evaluator lowers compact records.'
        ),
        evaluator_identity=evaluator_implementation.object_id,
        apparatus_site_count=0,
        classification=IndependentSubstrateIndependenceClass.NOVEL_EXTERNAL_IMPLEMENTATION,
        reason_codes=(),
    )
    contract = IndependentSubstrateTargetPredictionContract(
        contract_id="contract.independent-substrate-grounding.grid2op.prediction",
        target_slot=IndependentSubstrateTargetKind.GRID2OP,
        scientific_design_basis=ObjectIdentity.from_record(
            scientific_design_basis.design_basis_id,
            scientific_design_basis,
        ),
        source_qualification=ObjectIdentity.from_record(
            source_qualification.qualification_id,
            source_qualification,
        ),
        independence_dossier=dossier,
        mapping_candidate_roster=structured_design.mapping_candidates,
        mapping_development_assessments=mapping_assessments,
        selected_mapping_candidate_id=selected_mapping.candidate_id,
        denominator_candidate_lattice=structured_design.denominator_candidates,
        denominator_development_assessments=denominator_assessments,
        selected_denominator_candidate_id=selected_denominator.candidate_id,
        selected_claimed_law_operand_ids=("A", "D", "H", "R", "tau"),
        selected_unsupported_law_operand_ids=(),
        proposed_denominator_candidate_id="denominator.grid2op.00-proposed",
        meaningful_split_candidate_ids=("denominator.grid2op.01-chronic-timestamp-split",),
        merge_or_omission_candidate_ids=("denominator.grid2op.02-backend-merged",),
        impossible_candidate_ids=("denominator.grid2op.03-chronic-omitted",),
        target_forecast_alphabet=ObjectIdentity.from_record(
            forecast_alphabet.alphabet_id,
            forecast_alphabet,
        ),
        comparator_encodings=encodings,
        target_power_freeze=ObjectIdentity.from_record(power_freeze.freeze_id, power_freeze),
        metric_topology_design=ObjectIdentity.from_record(metric_design.design_id, metric_design),
        target_development_evidence=ObjectIdentity.from_record(
            development_panel.panel_id,
            development_panel,
        ),
        structural_recurrence_target_design=ObjectIdentity.from_record(structural_recurrence_design.design_id, structural_recurrence_design),
        structural_recurrence_development_evidence=ObjectIdentity.from_record(evidence.evidence_id, evidence),
        same_complete_unit_ids_sha256=units_sha256,
        frozen_before_prediction_issue=True,
        published_before_evaluator_access=True,
        evaluation_outcome_access_count=0,
    )
    return Grid2OpDevelopmentFreeze(
        freeze_id="freeze.independent-substrate-grounding.grid2op.development",
        target_contract=contract,
        structural_recurrence_design=structural_recurrence_design,
        structural_recurrence_development_evidence=evidence,
        forecast_alphabet=forecast_alphabet,
        development_forecast_panel=forecast_panel,
        forecast_denominator_inputs=denominator_inputs,
        metric_topology_design=metric_design,
        frozen_before_prediction_issue=True,
        evaluation_outcome_access_count=0,
    )


def issue_grid2op_structural_recurrence_prediction(
    *,
    method_freeze: MarginStructuralRecurrenceForecastMethodFreeze,
    conformance: MarginStructuralRecurrenceForecastConformance,
    development: Grid2OpDevelopmentFreeze,
    power_freeze: IndependentSubstrateStructuredPowerFreeze,
    scientific_bootstrap_inputs: StructuralBootstrapInputCensus,
) -> MarginStructuralRecurrenceForecastPredictionIssue:
    return build_prediction_issue(
        issue_id='prediction-issue.independent-substrate-grounding.grid2op.structural-recurrence',
        method_freeze=method_freeze,
        conformance=conformance,
        design=development.structural_recurrence_design,
        development=development.structural_recurrence_development_evidence,
        evaluation_count=len(power_freeze.evaluation_unit_ids),
        prospective_validation_count=len(power_freeze.prospective_validation_unit_ids),
        scientific_bootstrap_inputs=scientific_bootstrap_inputs,
    )


def _topology_state(unit, action_id: str, horizon: int) -> str | None:
    branch = f".branch.grid2op.{action_id}."
    return next(
        (
            value.topology_state_id
            for value in unit.receiver_points
            if branch in value.point_id
            and value.receiver_id == "connected-components"
            and value.horizon_tick == horizon
            and value.valid
        ),
        None,
    )


def _rho(unit, action_id: str, horizon: int) -> Decimal | None:
    branch = f".branch.grid2op.{action_id}."
    return next(
        (
            value.value
            for value in unit.receiver_points
            if branch in value.point_id
            and value.receiver_id == "maximum-rho"
            and value.horizon_tick == horizon
            and value.valid
        ),
        None,
    )


def _bootstrap_interval(
    differences: tuple[int, ...],
    *,
    action_id: str,
    replications: int,
    simultaneous_alpha: Decimal,
    family_size: int,
    scientific_seed_input: StructuralBootstrapSeedInput,
    current_target_id: str,
    current_context_sha256: str,
) -> tuple[Decimal, Decimal]:
    numerical_input = require_structural_bootstrap_seed_input(
        scientific_seed_input, scientific_role="grid2op-paired-metric-topology",
        current_target_id=current_target_id, action_id=action_id,
        stage="paired-metric-topology", sample_count=len(differences),
        bootstrap_replications=replications, current_context_sha256=current_context_sha256,
    )
    if not differences or any(type(value) is not int or value not in (-1, 0, 1) for value in differences) or replications < 4096 or family_size < 1 or not Decimal(0) <= simultaneous_alpha <= Decimal(1):
        raise ValueError("Grid2Op paired bootstrap is outside its complete scientific contract")
    seed = int(numerical_input.full_seed_sha256[:16], 16)
    rng = Random(seed)
    count = len(differences)
    values = sorted(
        Decimal(sum(differences[rng.randrange(count)] for _ in range(count))) / Decimal(count)
        for _ in range(replications)
    )
    tail = simultaneous_alpha / Decimal(2 * family_size)
    lower_index = max(0, int(tail * Decimal(replications)) - 1)
    upper_index = min(replications - 1, int((Decimal(1) - tail) * Decimal(replications)))
    return values[lower_index], values[upper_index]


def grid2op_bootstrap_context_sha256(
    *, panel: IndependentSubstrateCompleteTargetPanel,
    design: Grid2OpMetricTopologyDesign, action_id: str,
) -> str:
    return structural_bootstrap_context_sha256(
        ObjectIdentity.from_record(panel.panel_id, panel),
        ObjectIdentity.from_record(design.design_id, design), action_id,
        design.bootstrap_replications, design.simultaneous_alpha,
        len(design.action_ids),
    )


def require_grid2op_bootstrap_inputs(
    *, panel: IndependentSubstrateCompleteTargetPanel,
    design: Grid2OpMetricTopologyDesign,
    scientific_bootstrap_inputs: StructuralBootstrapInputCensus | None,
) -> dict[str, StructuralBootstrapSeedInput]:
    census = require_structural_bootstrap_census(scientific_bootstrap_inputs)
    if tuple(row.action_id for row in census.inputs) != design.action_ids:
        raise ValueError("Grid2Op bootstrap requires exactly its complete frozen action census")
    result = {}
    for row in census.inputs:
        require_structural_bootstrap_seed_input(
            row, scientific_role="grid2op-paired-metric-topology",
            current_target_id=panel.panel_id, action_id=row.action_id,
            stage="paired-metric-topology", sample_count=len(panel.planned_unit_ids),
            bootstrap_replications=design.bootstrap_replications,
            current_context_sha256=grid2op_bootstrap_context_sha256(panel=panel, design=design, action_id=row.action_id),
        )
        result[row.action_id] = row
    if len({(row.original_source, row.export_receipt) for row in census.inputs}) != 1:
        raise ValueError("Grid2Op bootstrap census crosses its original source or authenticated export custody")
    return result



def _metric_topology_exchanges(
    *,
    panel: IndependentSubstrateCompleteTargetPanel,
    design: Grid2OpMetricTopologyDesign,
    scientific_bootstrap_inputs: StructuralBootstrapInputCensus,
) -> tuple[IndependentSubstrateMetricTopologyExchange, ...]:
    seed_inputs = require_grid2op_bootstrap_inputs(panel=panel, design=design, scientific_bootstrap_inputs=scientific_bootstrap_inputs)
    unit_ids_sha256 = _digest_ids(panel.planned_unit_ids)
    values = []
    for action_id in design.action_ids:
        paired = []
        topology_count = 0
        metric_count = 0
        for unit in panel.units:
            hold_topology = _topology_state(unit, "hold", design.shared_horizon_step)
            action_topology = _topology_state(unit, action_id, design.shared_horizon_step)
            hold_rho = _rho(unit, "hold", design.shared_horizon_step)
            action_rho = _rho(unit, action_id, design.shared_horizon_step)
            topology = hold_topology is not None and action_topology not in {None, hold_topology}
            metric = (
                hold_rho is not None
                and action_rho is not None
                and hold_rho - action_rho >= Decimal("0.01")
            )
            topology_count += topology
            metric_count += metric
            paired.append(int(topology) - int(metric))
        delta = Decimal(topology_count - metric_count) / Decimal(len(panel.units))
        lower, upper = _bootstrap_interval(
            tuple(paired),
            action_id=action_id,
            replications=design.bootstrap_replications,
            simultaneous_alpha=design.simultaneous_alpha,
            family_size=len(design.action_ids),
            scientific_seed_input=seed_inputs[action_id],
            current_target_id=panel.panel_id,
            current_context_sha256=grid2op_bootstrap_context_sha256(panel=panel, design=design, action_id=action_id),
        )
        values.append(
            IndependentSubstrateMetricTopologyExchange(
                exchange_id=f"exchange.grid2op.{action_id}.topology-vs-thermal",
                complete_unit_ids_sha256=unit_ids_sha256,
                topology_success_count=topology_count,
                metric_success_count=metric_count,
                complete_unit_count=len(panel.units),
                delta=delta,
                simultaneous_lower=lower,
                simultaneous_upper=upper,
                physical_response_margin=delta,
                certification_margin=lower,
                measurement_backaction_status="PASSIVE_BOUNDED_SHARED_SIMULATOR_OBSERVATION",
                claim_bearing=delta > 0 and lower > 0,
            )
        )
    return tuple(sorted(values, key=lambda value: value.exchange_id))


def evaluate_grid2op_admission(
    *,
    method_freeze: MarginStructuralRecurrenceForecastMethodFreeze,
    development: Grid2OpDevelopmentFreeze,
    prediction_issue: MarginStructuralRecurrenceForecastPredictionIssue,
    power_freeze: IndependentSubstrateStructuredPowerFreeze,
    evaluation_panel: IndependentSubstrateCompleteTargetPanel,
    reveal_authority: ObjectIdentity,
    scientific_bootstrap_inputs: StructuralBootstrapInputCensus,
    denominator_inputs: Grid2OpForecastDenominatorInputs | None = None,
) -> Grid2OpAdmissionEvaluationEvaluation:
    if type(denominator_inputs) is not Grid2OpForecastDenominatorInputs:
        raise ValueError("Grid2Op admission evaluation requires original numerical denominator inputs and current export custody before lowering or scoring")
    denominator_inputs.validate_panel(
        panel=ObjectIdentity.from_record(evaluation_panel.panel_id, evaluation_panel),
        phase="EVALUATION",
        unit_ids=tuple(unit.unit_id for unit in evaluation_panel.units),
    )
    denominator_inputs.validate_evidence(
        target_design=ObjectIdentity.from_record(development.structural_recurrence_design.design_id, development.structural_recurrence_design),
        forecast_alphabet=ObjectIdentity.from_record(development.forecast_alphabet.alphabet_id, development.forecast_alphabet),
        phase="EVALUATION",
        unit_ids=tuple(unit.unit_id for unit in evaluation_panel.units),
        current_unit_fingerprints=tuple(unit.fingerprint() for unit in evaluation_panel.units),
    )
    require_grid2op_bootstrap_inputs(panel=evaluation_panel, design=development.metric_topology_design, scientific_bootstrap_inputs=scientific_bootstrap_inputs)
    evidence = lower_grid2op_panel_to_structural_recurrence(
        design=development.structural_recurrence_design,
        power_freeze=power_freeze,
        panel=evaluation_panel,
        execution_authority_verified=True,
    )
    admission_handoff = evaluate_margin_admission(
        method_freeze=method_freeze,
        prediction=prediction_issue,
        design=development.structural_recurrence_design,
        evaluation=evidence,
        reveal_authority=reveal_authority,
    )
    forecast_panel = build_grid2op_forecast_panel(
        panel_id="panel.independent-substrate-grounding.grid2op.evaluation-forecasts",
        alphabet=development.forecast_alphabet,
        design=development.structural_recurrence_design,
        evidence=evidence,
        evaluator_reveal_authorized=True,
        denominator_inputs=denominator_inputs,
    )
    predictions = build_grid2op_structural_recurrence_forecast_predictions(
        evaluation_panel=forecast_panel,
        prediction_issue=prediction_issue,
    )
    scored = tuple(
        IndependentSubstrateScoredComparator(
            result_id=f"scored.grid2op.{encoding.kind.value.lower().replace('_', '-')}",
            encoding=encoding,
            score=score_independent_substrate_comparator(
                score_id=f"score.grid2op.{encoding.kind.value.lower().replace('_', '-')}",
                encoding=encoding,
                evaluation_panel=forecast_panel,
                structural_recurrence_predictions=(
                    predictions if encoding.kind is IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE else ()
                ),
            ),
        )
        for encoding in development.target_contract.comparator_encodings
    )
    comparator = adjudicate_independent_substrate_restrictiveness(
        adjudication_id="adjudication.independent-substrate-grounding.grid2op.comparators",
        evaluation_panel=forecast_panel,
        scored_comparators=tuple(sorted(scored, key=lambda value: value.result_id)),
    )
    exchanges = _metric_topology_exchanges(
        panel=evaluation_panel,
        design=development.metric_topology_design,
        scientific_bootstrap_inputs=scientific_bootstrap_inputs,
    )
    observed = admission_handoff.action_fiber_admission.policy_safety
    policy_exact = prediction_issue.predicted_evaluation_policy_branch is observed.policy_branch
    action_exact = prediction_issue.predicted_evaluation_action_id == observed.selected_action_id
    eligible = (
        evidence.independent_unit_count == len(power_freeze.evaluation_unit_ids)
        and power_freeze.joint_precision_passed
        and not power_freeze.panel_envelope_limited
    )
    forecast_by_action = {
        value.action_id: value.predicted_admitted
        for value in prediction_issue.forecasts
        if value.future_stage.value == "EVALUATION"
    }
    observed_by_action = {
        value.action_fiber.action_id: value.action_fiber.admitted
        for value in admission_handoff.evaluation_margins.action_margins
    }
    support = (
        eligible and policy_exact and action_exact and forecast_by_action == observed_by_action
    )
    opposition = (
        not policy_exact
        or not action_exact
        or any(forecast_by_action.get(key) != value for key, value in observed_by_action.items())
    )
    reasons = set(comparator.reason_codes)
    if not policy_exact:
        reasons.add("GRID2OP_POLICY_BRANCH_MISMATCH")
    if not action_exact:
        reasons.add("GRID2OP_ACTION_IDENTITY_MISMATCH")
    if forecast_by_action != observed_by_action:
        reasons.add("GRID2OP_ACTION_MARGIN_FORECAST_MISMATCH")
    return Grid2OpAdmissionEvaluationEvaluation(
        evaluation_id="evaluation.independent-substrate-grounding.grid2op.admission-evaluation",
        target_contract=ObjectIdentity.from_record(
            development.target_contract.contract_id,
            development.target_contract,
        ),
        prediction_issue=prediction_issue,
        evaluation_evidence=evidence,
        admission_handoff=admission_handoff,
        evaluation_forecast_panel=forecast_panel,
        forecast_denominator_inputs=denominator_inputs,
        comparator_adjudication=comparator,
        metric_topology_exchanges=exchanges,
        predicted_policy_branch=prediction_issue.predicted_evaluation_policy_branch,
        predicted_action_id=prediction_issue.predicted_evaluation_action_id,
        observed_policy_branch=observed.policy_branch,
        observed_action_id=observed.selected_action_id,
        evaluation_eligible=eligible,
        categorical_support=support,
        decisive_opposition=opposition,
        mandatory_hold=observed.policy_branch is PolicyBranch.HOLD,
        nonattempt=observed.policy_branch is PolicyBranch.NONATTEMPT,
        panel_envelope_limited=power_freeze.panel_envelope_limited,
        reason_codes=tuple(sorted(reasons)),
    )


def finalize_grid2op_validation(
    *,
    development: Grid2OpDevelopmentFreeze,
    admission_evaluation: Grid2OpAdmissionEvaluationEvaluation,
    power_freeze: IndependentSubstrateStructuredPowerFreeze,
    prospective_validation_panel: IndependentSubstrateCompleteTargetPanel | None,
    validation_authority: ObjectIdentity,
) -> Grid2OpTargetValidation:
    selected_nonhold = (
        (admission_evaluation.observed_action_id,) if admission_evaluation.observed_policy_branch is PolicyBranch.EXACT_ACTION else ()
    )
    evidence = (
        lower_grid2op_panel_to_structural_recurrence(
            design=development.structural_recurrence_design,
            power_freeze=power_freeze,
            panel=prospective_validation_panel,
            execution_authority_verified=True,
            nonhold_action_ids=selected_nonhold,
        )
        if prospective_validation_panel is not None
        else None
    )
    result = finalize_margin_target(
        admission_handoff=admission_evaluation.admission_handoff,
        design=development.structural_recurrence_design,
        validation_evidence=evidence,
        validation_authority=validation_authority,
    )
    match = match_target(
        prediction=admission_evaluation.prediction_issue,
        admission_handoff=admission_evaluation.admission_handoff,
        result=result,
    )
    support = admission_evaluation.categorical_support and match.passed
    opposition = admission_evaluation.decisive_opposition or not match.passed
    return Grid2OpTargetValidation(
        validation_id="validation.independent-substrate-grounding.grid2op",
        parent_admission_evaluation=ObjectIdentity.from_record(admission_evaluation.evaluation_id, admission_evaluation),
        validation_evidence=evidence,
        target_result=result,
        target_match=match,
        prospective_validation_executed=prospective_validation_panel is not None,
        prerequisite_nonattempt=prospective_validation_panel is None,
        categorical_support=support,
        decisive_opposition=opposition,
        reason_codes=tuple(sorted(set(admission_evaluation.reason_codes) | set(match.reason_codes))),
    )


def grid2op_terminal_handoff(
    *,
    admission_evaluation: Grid2OpAdmissionEvaluationEvaluation,
    validation: Grid2OpTargetValidation,
    prediction_receipt_sha256: str,
    match_receipt_sha256: str,
    handoff_id: str = "handoff.independent-substrate-grounding.source-continuation.grid2op",
) -> IndependentSubstrateTargetTerminalHandoff:
    validate_sha256(prediction_receipt_sha256, field_name="prediction_receipt_sha256")
    validate_sha256(match_receipt_sha256, field_name="match_receipt_sha256")
    unsafe_false_admission_count = next(
        value.score.unsafe_false_admission_count
        for value in admission_evaluation.comparator_adjudication.scored_comparators
        if value.encoding.kind is IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE
    )
    supported, decisive_opposition, reason_codes = grid2op_terminal_disposition(
        categorical_support=validation.categorical_support,
        decisive_opposition=validation.decisive_opposition,
        unsafe_false_admission_count=unsafe_false_admission_count,
        reason_codes=validation.reason_codes,
    )
    return IndependentSubstrateTargetTerminalHandoff(
        handoff_id=handoff_id,
        slot=IndependentSubstrateTargetKind.GRID2OP,
        evidence_world=EvidenceWorld.RESETTABLE_SIMULATOR,
        attained_level=TargetLevel.ADMISSION if supported else TargetLevel.LAW_QUALIFICATION,
        independence_class=IndependentSubstrateIndependenceClass.NOVEL_EXTERNAL_IMPLEMENTATION,
        evaluation_eligible=admission_evaluation.evaluation_eligible,
        categorical_support=supported,
        decisive_opposition=decisive_opposition,
        unsafe_false_admission_count=unsafe_false_admission_count,
        action_ontology_clock_error_count=0,
        panel_envelope_limited=admission_evaluation.panel_envelope_limited,
        structural_recurrence_restrictiveness_supported=(
            admission_evaluation.comparator_adjudication.disposition.value == "PREDICTIVE_RESTRICTIVENESS_SUPPORTED"
        ),
        comparator_tied_or_won=admission_evaluation.comparator_adjudication.comparator_tied_or_won,
        structural_recurrence_less_safe_or_exact_than_comparator=(
            admission_evaluation.comparator_adjudication.structural_recurrence_less_safe_or_exact_than_comparator
        ),
        metric_topology_exchanges=admission_evaluation.metric_topology_exchanges,
        physical_consistency="NOT_APPLICABLE",
        prediction_receipt_sha256=prediction_receipt_sha256,
        match_receipt_sha256=match_receipt_sha256,
        maximum_claim_ceiling=(
            "GRID2OP_INDEPENDENT_SIMULATOR_CONTROLLER_ADMISSION_RECURRENCE_WITH_CONDITIONAL_PROSPECTIVE_CONTROLLER_EVALUATION"
            if supported
            else "GRID2OP_INDEPENDENT_SIMULATOR_EVALUATED_NO_POSITIVE_RECURRENCE"
        ),
        reason_codes=reason_codes,
    )


def grid2op_terminal_disposition(
    *,
    categorical_support: bool,
    decisive_opposition: bool,
    unsafe_false_admission_count: int,
    reason_codes: tuple[str, ...],
) -> tuple[bool, bool, tuple[str, ...]]:
    """Apply counterexample-first precedence to the compact target handoff."""

    if unsafe_false_admission_count < 0:
        raise ValueError("Grid2Op unsafe false-admission count cannot be negative")
    reasons = set(reason_codes)
    if unsafe_false_admission_count:
        decisive_opposition = True
        reasons.add("UNSAFE_FALSE_ADMISSION_DECISIVE_OPPOSITION")
    supported = categorical_support and not decisive_opposition
    return supported, decisive_opposition, tuple(sorted(reasons))


__all__ = [
    'Grid2OpDevelopmentFreeze',
    'Grid2OpJointPowerJustification',
    'Grid2OpMetricTopologyDesign',
    'Grid2OpAdmissionEvaluationEvaluation',
    'Grid2OpTargetValidation',
    'evaluate_grid2op_admission',
    "finalize_grid2op_validation",
    "freeze_grid2op_development",
    "grid2op_joint_power_justification",
    "grid2op_terminal_handoff",
    "grid2op_terminal_disposition",
    'issue_grid2op_structural_recurrence_prediction',
]
