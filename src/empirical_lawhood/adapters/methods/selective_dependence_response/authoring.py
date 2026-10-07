"""Accepted planning and standard-candidate bridge for selective dependence response targets.

This module does not create construct review, method-completion, source-canary
outcomes or authority.  Target adapters provide a native planning projection;
the shared code maps it into the accepted kernel/planning ontology and the
ordinary standard candidate compiler.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

from empirical_lawhood.adapters.methods.formal_analysis import standard_formal_method_catalog
from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import (
    ClaimSpec,
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.experiments import (
    AssignmentKind,
    AssignmentSpec,
    ControlKind,
    ControlSpec,
    ExperimentSpec,
    PrecisionGoal,
    RevealBarrierSpec,
)
from empirical_lawhood.kernel.obligations import (
    ClosureSpec,
    ComputabilityEvidence,
    FalsifierKind,
    FalsifierSpec,
    ObligationStatus,
    ScientificObligations,
    StructuralConvergenceSpec,
    SupportSpec,
    UncertaintySpec,
    ValiditySpec,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.quantities import (
    QuantityKind,
    QuantitySpec,
    ResponseDirection,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.kernel.references import QuantityBound
from empirical_lawhood.kernel.status import LifecycleStatus, ReadinessStatus
from empirical_lawhood.kernel.systems import (
    BalanceRole,
    IndependentUnitSpec,
    PortDirection,
    PortSpec,
    RelationalIdentity,
    SystemBoundaryKind,
    SystemSpec,
)
from empirical_lawhood.kernel.time import (
    CausalPhase,
    AvailabilitySpec,
    ClockLabelSemantics,
    ClockSpec,
    HoldSemantics,
    HorizonSpec,
    InformationCutoff,
    SamplingSemantics,
)
from empirical_lawhood.kernel.worlds import (
    ComputabilityEnvelope,
    EvidenceUnitScope,
    NumericalCoordinateSpec,
    NumericalViewSpec,
    RandomnessSemantics,
    WorldKind,
    WorldSpec,
)
from empirical_lawhood.planning.campaigns import (
    CampaignLane,
    CampaignNode,
    CampaignNodeKind,
    CampaignSpec,
    DecisionRight,
)
from empirical_lawhood.planning.experiment_entry import ExperimentEntryChecklist, ExperimentEntryPackage, ExperimentEntryRequirement, ExperimentEntryRequirementBinding, ExperimentEntryTransition, ExperimentTerminalClass, StudyDefinition
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    derive_formal_gap_applicability,
)
from empirical_lawhood.planning.formal_gaps import (
    FormalGapCoverage,
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
    FormalGapEvidenceWorld,
    FormalGapRegister,
)
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.candidate_compiler import AuthoringMaterializationIdentity, CandidateCompilationContext, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, StandardCandidateCompilationContext, StudyCompilationReport, compile_study_candidate, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.plans import ScientificInputRole
from empirical_lawhood.runtime.source_resolution import SourceMaterializationConfig, SourceReadMode

from .composition import SelectiveDependenceResponseStageComposition, compose_target_development
from .contracts import SelectiveDependenceResponseContaminationLedger
from .source_completion import SelectiveDependenceResponseSourceCanaryCompletionEnvelope
from .target_protocol import SelectiveDependenceResponseTargetBinding, SelectiveDependenceResponseTargetStage


_PROGRAMME_BUDGET = ResourceBudget(
    cpu_cores=4,
    memory_bytes=4 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=21_600,
    source_scan_bytes=1024**3,
    output_bytes=1024**3,
)


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetPlanningProjection:
    """Target-owned native facts needed by the accepted planning ontology."""

    target_slug: str
    target_label: str
    binding: SelectiveDependenceResponseTargetBinding
    source_composition: SelectiveDependenceResponseStageComposition
    represented_physics: tuple[str, ...]
    unrepresented_physics: tuple[str, ...]
    equations_id: str
    closure_ids: tuple[str, ...]
    boundary_condition_ids: tuple[str, ...]
    solver_id: str
    solver_version: str
    precision: str
    device_class: str
    runtime_id: str
    numerical_coordinates: tuple[NumericalCoordinateSpec, ...]
    action_stage_units: tuple[str, str, str, str]
    receiver_units: tuple[tuple[str, str, ResponseDirection], ...]
    action_stage_bounds: tuple[
        tuple[Decimal | None, Decimal | None],
        tuple[Decimal | None, Decimal | None],
        tuple[Decimal | None, Decimal | None],
        tuple[Decimal | None, Decimal | None],
    ]
    maximum_horizon_seconds: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.target_slug, field_name="target_slug")
        if self.binding.target_slug != self.target_slug:
            raise ValueError("planning projection and target binding differ")
        if self.source_composition.stage_id != (f"selective-dependence-response.{self.target_slug}.source-canary"):
            raise ValueError("planning projection does not bind the source-canary stage")
        if len(self.action_stage_units) != 4:
            raise ValueError("planning projection must retain four action stages")
        receiver_ids = tuple(value[0] for value in self.receiver_units)
        if receiver_ids != tuple(sorted(set(receiver_ids))):
            raise ValueError("planning projection receiver roster differs")
        for lower, upper in self.action_stage_bounds:
            if lower is None and upper is None:
                raise ValueError("planning projection action stage is unbounded")
            if lower is not None and upper is not None and lower >= upper:
                raise ValueError("planning projection action support is empty")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetAuthoringAct:
    """Pure, authority-pending standard authoring root for one target stage."""

    projection: SelectiveDependenceResponseTargetPlanningProjection
    stage_composition: SelectiveDependenceResponseStageComposition
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    source_configs: tuple[SourceMaterializationConfig, ...]
    source_records: tuple[CanonicalRecord, ...]
    qualifications: tuple[MaterializationQualificationReceipt, ...]
    inventory: FormalGapSourceCapabilityInventory
    template: StudyTemplate
    catalog: CandidateCapabilityCatalog
    draft: StudyDraft
    package: StudyDefinition
    compilation: StudyCompilationReport

    @property
    def candidate_input_payloads(self) -> tuple[bytes, ...]:
        """Return every compact content-addressed byte string needed to compile."""

        return (
            *self.stage_composition.config_payloads,
            *(value.canonical_bytes() for value in self.source_configs),
            *(value.canonical_bytes() for value in self.source_records),
            *(value.canonical_bytes() for value in self.qualifications),
        )

    @property
    def candidate_composition(self) -> SelectiveDependenceResponseStageComposition:
        """Bind the stage runtime composition to this exact accepted authoring root."""

        return replace(
            self.stage_composition,
            catalog=self.catalog,
            known_design_inputs=self.draft.design_inputs,
            formal_source_inventories=(self.inventory,),
        )


def _quantity(
    *,
    quantity_id: str,
    kind: QuantityKind,
    unit: str,
    frame: str,
    clock_id: str,
    phase: CausalPhase,
    outcome_access: OutcomeAccess,
    direction: ResponseDirection = ResponseDirection.NOT_APPLICABLE,
) -> QuantitySpec:
    return QuantitySpec(
        quantity_id=quantity_id,
        label=quantity_id.replace(".", " "),
        kind=kind,
        dimension=unit,
        native_unit=unit,
        coordinate_frame=frame,
        clock_id=clock_id,
        availability=AvailabilitySpec(
            clock_id=clock_id,
            phase=phase,
            outcome_access=outcome_access,
        ),
        response_direction=direction,
    )


def build_target_system(projection: SelectiveDependenceResponseTargetPlanningProjection) -> SystemSpec:
    """Map one native simulator target to an exact L(D,H,A,R,tau) system."""

    slug = projection.target_slug
    binding = projection.binding
    preparation = binding.preparation
    design = binding.design
    denominator_ids = tuple(f"denominator.{value}" for value in getattr(design, "denominator_ids"))
    history_ids = tuple(f"history.{value}" for value in getattr(design, "history_ids"))
    action_stage_rows = (
        ("action.requested", projection.action_stage_units[0], CausalPhase.ACTION_REQUESTED),
        ("action.accepted", projection.action_stage_units[1], CausalPhase.ACTION_REQUESTED),
        ("action.applied", projection.action_stage_units[2], CausalPhase.ACTION_APPLIED),
        ("action.realized", projection.action_stage_units[3], CausalPhase.ACTION_APPLIED),
    )
    action_ids = tuple(sorted(value[0] for value in action_stage_rows))
    receiver_ids = tuple(f"receiver.{value[0]}" for value in projection.receiver_units)
    clock = ClockSpec(
        clock_id=f"clock.selective-dependence-response.{slug}.simulator",
        label=f"selective dependence response {projection.target_label} simulator clock",
        time_unit="second",
        coordinate_frame=f"{slug}-simulator-time",
        sampling=SamplingSemantics.EVENT_DRIVEN,
        hold=HoldSemantics.SOURCE_DEFINED,
        label_semantics=ClockLabelSemantics.EVENT,
        nominal_period=None,
        alignment_tolerance=Decimal("1e-12"),
    )
    quantities: list[QuantitySpec] = []
    for quantity_id in denominator_ids:
        quantities.append(
            _quantity(
                quantity_id=quantity_id,
                kind=QuantityKind.DENOMINATOR,
                unit="category-indicator",
                frame=f"{slug}-native-denominator",
                clock_id=clock.clock_id,
                phase=CausalPhase.PRE_ACTION,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
        )
    for quantity_id in history_ids:
        quantities.append(
            _quantity(
                quantity_id=quantity_id,
                kind=QuantityKind.HISTORY,
                unit="category-indicator",
                frame=f"{slug}-pre-action-history",
                clock_id=clock.clock_id,
                phase=CausalPhase.PRE_ACTION,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
        )
    for quantity_id, unit, phase in action_stage_rows:
        quantities.append(
            _quantity(
                quantity_id=quantity_id,
                kind=QuantityKind.ACTION,
                unit=unit,
                frame=f"{slug}-native-action-chain",
                clock_id=clock.clock_id,
                phase=phase,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
        )
    for native_id, unit, direction in projection.receiver_units:
        quantities.append(
            _quantity(
                quantity_id=f"receiver.{native_id}",
                kind=QuantityKind.RECEIVER,
                unit=unit,
                frame=f"{slug}-native-receiver",
                clock_id=clock.clock_id,
                phase=CausalPhase.RECEIVER,
                outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                direction=direction,
            )
        )
    quantities.append(
        _quantity(
            quantity_id="sink.validity-intersection",
            kind=QuantityKind.SINK,
            unit="signed-native-margin",
            frame=f"{slug}-receiver-gate",
            clock_id=clock.clock_id,
            phase=CausalPhase.RECEIVER,
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            direction=ResponseDirection.HIGHER_IS_BETTER,
        )
    )
    quantities_tuple = tuple(sorted(quantities, key=lambda value: value.quantity_id))
    world = WorldSpec(
        world_id=f"world.selective-dependence-response.{slug}.numerical-simulator",
        label=f"selective dependence response {projection.target_label} integrated numerical simulator",
        kind=WorldKind.NUMERICAL_SIMULATOR,
        represented_physics=projection.represented_physics,
        unrepresented_physics=projection.unrepresented_physics,
        privileged_truth_quantity_ids=(),
        maximum_evidence=EvidenceCeiling.ADMISSION,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.EVALUATION_REVEALED,
            }
        ),
    )
    independent_unit = IndependentUnitSpec(
        unit_id=f"unit.selective-dependence-response.{slug}.complete-preparation",
        label=(
            f"one complete independently generated {projection.target_label} "
            "preparation; all D/H/A/R/tau branches are nested"
        ),
        grouping_key=f"group.selective-dependence-response.{slug}.complete-preparation",
        scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
    )
    relation = RelationalIdentity(
        relation_id=f"relation.selective-dependence-response.{slug}.denominator-local",
        denominator_quantity_ids=denominator_ids,
        history_quantity_ids=history_ids,
        memoryless=False,
        action_quantity_ids=action_ids,
        receiver_quantity_ids=receiver_ids,
        horizon=HorizonSpec(
            horizon_id=f"horizon.selective-dependence-response.{slug}.maximum",
            clock_id=clock.clock_id,
            duration=projection.maximum_horizon_seconds,
            time_unit="second",
        ),
    )
    envelope = ComputabilityEnvelope(
        envelope_id=f"compute.selective-dependence-response.{slug}.bounded",
        represented_effect_ids=projection.represented_physics,
        unresolved_effect_ids=(),
        required_structure_ids=(
            "complete-unit-full-fan-in",
            "finite-action-response",
            "noncompensating-admission",
            "selective-dependence-exchanges",
            "stage-resolved-action-chain",
        ),
        computable_structure_ids=(
            "complete-unit-full-fan-in",
            "finite-action-response",
            "noncompensating-admission",
            "selective-dependence-exchanges",
            "stage-resolved-action-chain",
        ),
        max_cpu_cores=_PROGRAMME_BUDGET.cpu_cores,
        max_memory_bytes=_PROGRAMME_BUDGET.memory_bytes,
        max_gpu_devices=0,
        max_wall_time_seconds=_PROGRAMME_BUDGET.wall_time_seconds,
        max_output_bytes=_PROGRAMME_BUDGET.output_bytes,
        worst_case_latency_seconds=Decimal(_PROGRAMME_BUDGET.wall_time_seconds),
    )
    view = NumericalViewSpec(
        view_id=f"view.selective-dependence-response.{slug}.frozen",
        world_id=world.world_id,
        physical_preparation_id=independent_unit.unit_id,
        equations_id=projection.equations_id,
        closure_ids=projection.closure_ids,
        boundary_condition_ids=projection.boundary_condition_ids,
        coordinates=projection.numerical_coordinates,
        solver_id=projection.solver_id,
        solver_version=projection.solver_version,
        precision=projection.precision,
        device_class=projection.device_class,
        runtime_id=projection.runtime_id,
        randomness=RandomnessSemantics.GENERATIVE_PREPARATION,
        observation_operator_id=f"observer.selective-dependence-response.{slug}.complete-unit",
        computability_envelope_id=envelope.envelope_id,
    )
    authority = AuthorityPolicy(
        policy_id=f"authority-policy.selective-dependence-response.{slug}.simulation",
        delegator_id="human.project-owner",
        delegate_id=f"gate.selective-dependence-response.{slug}.simulation",
        scope_ids=(f"campaign.selective-dependence-response.{slug}",),
        allowed_world_kinds=frozenset({WorldKind.NUMERICAL_SIMULATOR}),
        allowed_actions=frozenset(
            {
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.SIMULATION_EXECUTION,
            }
        ),
        allowed_source_classes=frozenset({SourceAccessClass.NONE}),
        required_gate_ids=(
            "accountable-construct-review",
            "clean-implementation",
            "exact-stage-authoring-package",
            "separate-execution-and-reveal-authority",
        ),
        nondelegable_actions=frozenset(
            {
                AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
                AuthorityAction.HIL_ACTUATION,
                AuthorityAction.HUMAN_OR_ANIMAL_INTERVENTION,
                AuthorityAction.LIVE_ACTUATION,
                AuthorityAction.SAFETY_SIGNIFICANT_OPERATION,
            }
        ),
        budget_ceiling=_PROGRAMME_BUDGET,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    ports = tuple(
        sorted(
            (
                *(
                    PortSpec(
                        port_id=f"port.{quantity_id}.input",
                        quantity_id=quantity_id,
                        clock_id=clock.clock_id,
                        direction=PortDirection.INPUT,
                        balance_role=BalanceRole.COMMAND,
                        authority_action=AuthorityAction.SIMULATION_EXECUTION,
                    )
                    for quantity_id in action_ids
                ),
                *(
                    PortSpec(
                        port_id=f"port.{quantity_id}.output",
                        quantity_id=quantity_id,
                        clock_id=clock.clock_id,
                        direction=PortDirection.OUTPUT,
                        balance_role=(
                            BalanceRole.ENERGY
                            if quantity_id == "sink.validity-intersection"
                            else BalanceRole.OBSERVATION
                        ),
                    )
                    for quantity_id in (*receiver_ids, "sink.validity-intersection")
                ),
            ),
            key=lambda value: value.port_id,
        )
    )
    if preparation.target_id != binding.target_id:
        raise ValueError("planning projection preparation crosses targets")
    return SystemSpec(
        system_id=f"system.selective-dependence-response.{slug}",
        label=f"selective dependence response {projection.target_label}",
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=quantities_tuple,
        independent_unit=independent_unit,
        authority_policy=authority,
        ports=ports,
        computability_envelopes=(envelope,),
        numerical_views=(view,),
    )


def _target_obligations(
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    system: SystemSpec,
    cutoff: InformationCutoff,
) -> ScientificObligations:
    slug = projection.target_slug
    binding = projection.binding
    freeze = binding.analysis_freeze
    qualifier_key = next(
        value.capability_key
        for value in projection.source_composition.registry.capabilities
        if value.capability_key.endswith("qualify-source")
    )
    return ScientificObligations(
        obligations_id=f"obligations.selective-dependence-response.{slug}",
        support=SupportSpec(
            support_id=f"support.selective-dependence-response.{slug}.finite-native-chart",
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=len(binding.preparation.evaluation_unit_ids),
            nested_numerical_view_count=1,
            information_cutoff_id=cutoff.cutoff_id,
            chart_ids=tuple(
                f"chart.selective-dependence-response.{slug}.{value}" for value in getattr(binding.design, "action_ids")
            ),
            denominator_cell_ids=freeze.primary_cell_ids,
            action_bounds=tuple(
                sorted(
                    (
                        QuantityBound(
                            bound_id=f"bound.selective-dependence-response.{slug}.action-requested",
                            quantity_id="action.requested",
                            native_unit=projection.action_stage_units[0],
                            lower=projection.action_stage_bounds[0][0],
                            upper=projection.action_stage_bounds[0][1],
                        ),
                        QuantityBound(
                            bound_id=f"bound.selective-dependence-response.{slug}.action-accepted",
                            quantity_id="action.accepted",
                            native_unit=projection.action_stage_units[1],
                            lower=projection.action_stage_bounds[1][0],
                            upper=projection.action_stage_bounds[1][1],
                        ),
                        QuantityBound(
                            bound_id=f"bound.selective-dependence-response.{slug}.action-applied",
                            quantity_id="action.applied",
                            native_unit=projection.action_stage_units[2],
                            lower=projection.action_stage_bounds[2][0],
                            upper=projection.action_stage_bounds[2][1],
                        ),
                        QuantityBound(
                            bound_id=f"bound.selective-dependence-response.{slug}.action-realized",
                            quantity_id="action.realized",
                            native_unit=projection.action_stage_units[3],
                            lower=projection.action_stage_bounds[3][0],
                            upper=projection.action_stage_bounds[3][1],
                        ),
                    ),
                    key=lambda value: value.bound_id,
                )
            ),
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id=f"validity.selective-dependence-response.{slug}.exact-simulator-view",
            validity_domain_ids=(
                f"domain.selective-dependence-response.{slug}.frozen-design",
                f"domain.selective-dependence-response.{slug}.native-action-support",
            ),
            assumption_ids=(
                "simulator-is-prepared-response-medium-not-oracle",
                "source-runtime-view-and-observation-operator-exact",
            ),
            exclusion_reason_codes=(
                "NO_PHYSICAL_CLAIM_PROMOTION",
                "NO_TARGET_SPECIFIC_RESCUE",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id=f"uncertainty.selective-dependence-response.{slug}.complete-unit-local",
            method_key="bonferroni-complete-unit-student-t",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal(1) - freeze.exchange_familywise_alpha,
            interval_quantity_ids=system.relation.receiver_quantity_ids,
            limitation_codes=(
                "NESTED_CONDITIONS_NOT_REPLICATION",
                "SINGLE_FROZEN_NUMERICAL_VIEW",
            ),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=tuple(
            sorted(
                (
                    FalsifierSpec(
                        falsifier_id=f"falsifier.selective-dependence-response.{slug}.wrong-action",
                        kind=FalsifierKind.WRONG_ACTION,
                        capability_key=qualifier_key,
                        description=(
                            "Requested, accepted, applied and realized actions remain distinct."
                        ),
                        decisive_rule=(
                            "Any echo, rejection, clipping or realization mismatch is retained "
                            "and can stop or render the unit unevaluable."
                        ),
                        status=ObligationStatus.REQUIRED,
                    ),
                    FalsifierSpec(
                        falsifier_id=f"falsifier.selective-dependence-response.{slug}.one-factor-exchange",
                        kind=FalsifierKind.ONE_FACTOR_EXCHANGE,
                        capability_key=qualifier_key,
                        description="Every frozen active/invariant exchange changes one role.",
                        decisive_rule=(
                            "An exact opposed exchange or stronger predeclared counterexample "
                            "precedes aggregate positive evidence."
                        ),
                        status=ObligationStatus.REQUIRED,
                    ),
                    FalsifierSpec(
                        falsifier_id=f"falsifier.selective-dependence-response.{slug}.receiver-gate",
                        kind=FalsifierKind.RECEIVER_GATE,
                        capability_key=qualifier_key,
                        description=("Target and every sink/validity margin are noncompensating."),
                        decisive_rule=(
                            "No favourable receiver can compensate a failed sink, invalid "
                            "action fibre or unmeasured hold."
                        ),
                        status=ObligationStatus.REQUIRED,
                    ),
                ),
                key=lambda value: value.falsifier_id,
            )
        ),
        closure=ClosureSpec(
            closure_id=f"closure.selective-dependence-response.{slug}.selective-dependence",
            recurrence_cell_ids=freeze.primary_cell_ids,
            exchange_factor_ids=system.relation.denominator_quantity_ids,
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id=f"convergence.selective-dependence-response.{slug}.frozen-view",
            required_structure_ids=(
                "finite-law-calibration",
                "hold-measured-separately",
                "selective-active-and-invariant-exchanges",
                "support-boundary-refusal",
            ),
            numerical_view_ids=(system.numerical_views[0].view_id,),
            tolerances=(),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id=f"computability.selective-dependence-response.{slug}.bounded",
            envelope_id=system.computability_envelopes[0].envelope_id,
            numerical_view_ids=(system.numerical_views[0].view_id,),
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=(projection.binding.analysis_freeze.freeze_id,),
        ),
    )


def build_target_experiment(
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    system: SystemSpec,
) -> ExperimentSpec:
    "Build the full future local law/admission target design; controller use remains absent."

    slug = projection.target_slug
    binding = projection.binding
    cutoff = InformationCutoff(
        cutoff_id=f"cutoff.selective-dependence-response.{slug}.pre-action",
        clock_id=system.clocks[0].clock_id,
        phase=CausalPhase.PRE_ACTION,
        coordinate=Decimal(0),
    )
    view_ids = (system.numerical_views[0].view_id,)
    claims = (
        ClaimSpec(
            claim_id=f"claim.selective-dependence-response.{slug}.admission",
            world_id=system.world.world_id,
            relation_id=system.relation.relation_id,
            proposition=(
                "Within the exact finite native support, each action fibre is admitted "
                "only by the intersection of target, sink, validity, uncertainty and "
                "support gates, with measured hold mandatory outside admitted action."
            ),
            estimand=(
                "Complete-unit-local noncompensating gate margins and the categorical "
                "ACTION_AVAILABLE/HOLD_ONLY/NONATTEMPT/UNEVALUABLE disposition."
            ),
            physical_independent_unit_id=system.independent_unit.unit_id,
            requested_rung=EvidenceRung.ADMISSION,
            evidence_ceiling=EvidenceCeiling.ADMISSION,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            promotion_rule=(
                "controller admission requires supported law qualification, exact full evaluation fan-in, every gate "
                "nonnegative and measured hold; it never implies a prospective controller evaluation controller."
            ),
            assumption_ids=(
                "complete-preparation-is-the-independent-unit",
                "noncompensating-admission",
                "receiver-relative-direction",
            ),
            numerical_view_ids=view_ids,
        ),
        ClaimSpec(
            claim_id=f"claim.selective-dependence-response.{slug}.local-law",
            world_id=system.world.world_id,
            relation_id=system.relation.relation_id,
            proposition=(
                "The frozen finite response law has the predeclared selective pattern of "
                "active and invariant D/H/A/R/tau exchanges and outperforms the exact "
                "history-blind comparator on fresh complete evaluation units."
            ),
            estimand=(
                "Complete-unit finite-law accuracy, Bonferroni exchange intervals, "
                "support refusal, and comparator advantage."
            ),
            physical_independent_unit_id=system.independent_unit.unit_id,
            requested_rung=EvidenceRung.LOCAL_LAW,
            evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            promotion_rule=(
                "Every predeclared exchange must be supported and no exact counterexample, "
                "source failure, excess nonevaluability or comparator tie may be rescued."
            ),
            assumption_ids=(
                "complete-preparation-is-the-independent-unit",
                "finite-categorical-law-not-continuum-extrapolation",
                "source-local-no-coefficient-pooling",
            ),
            numerical_view_ids=view_ids,
        ),
    )
    analysis_key = f"simulator.selective-dependence-response.{slug}.analyze-development"
    controls = (
        ControlSpec(
            control_id="control.negative-action",
            kind=ControlKind.NEGATIVE_ACTION,
            capability_key=analysis_key,
            target_quantity_ids=system.relation.receiver_quantity_ids,
            decisive_rule=(
                "The frozen opposite/hold/outside-support action semantics are retained."
            ),
        ),
        ControlSpec(
            control_id="control.support-matched-comparator",
            kind=ControlKind.BASELINE_COMPARATOR,
            capability_key=analysis_key,
            target_quantity_ids=system.relation.receiver_quantity_ids,
            decisive_rule=(
                "The full law and history-blind comparator share units, support, complete "
                "units and evaluation roster; a tie blocks distinctiveness."
            ),
        ),
    )
    evaluation_manifest = "\n".join(binding.preparation.evaluation_unit_ids).encode("ascii")
    from hashlib import sha256

    return ExperimentSpec(
        experiment_id=f"experiment.selective-dependence-response.{slug}",
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=tuple(sorted(claims, key=lambda value: value.claim_id)),
        assignment=AssignmentSpec(
            assignment_id=f"assignment.selective-dependence-response.{slug}.simulator-intervention",
            kind=AssignmentKind.SIMULATOR_INTERVENTION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism=(
                "Frozen native action labels generate separate requested, accepted, "
                "applied and realized records within each complete preparation."
            ),
            support_restriction_ids=tuple(
                sorted(
                    (
                        f"support.selective-dependence-response.{slug}.finite-native-chart",
                        binding.analysis_freeze.freeze_id,
                    )
                )
            ),
        ),
        measurement_quantity_ids=tuple(
            value.quantity_id
            for value in system.quantities
            if value.kind in {QuantityKind.RECEIVER, QuantityKind.SINK}
        ),
        controls=controls,
        precision_goals=(
            PrecisionGoal(
                goal_id=f"precision.selective-dependence-response.{slug}.frozen-panel",
                metric_id="predeclared-complete-unit-exchange-intervals",
                target_width=max(
                    value.equivalence_margin for value in binding.analysis_freeze.exchanges
                ),
                native_unit="target-native-exchange-unit",
                maximum_independent_units=len(binding.preparation.evaluation_unit_ids),
                stopping_rule=(
                    "Run the exact frozen roster or emit a typed stop; never add units, "
                    "change thresholds or retune after outcome access."
                ),
            ),
        ),
        information_cutoffs=(cutoff,),
        reveal_barrier=RevealBarrierSpec(
            barrier_id=f"reveal.selective-dependence-response.{slug}.evaluation",
            development_unit_ids=tuple(
                sorted(
                    (
                        *binding.preparation.canary_unit_ids,
                        *binding.preparation.development_unit_ids,
                    )
                )
            ),
            evaluation_cohort_id=f"cohort.selective-dependence-response.{slug}.evaluation",
            evaluation_manifest_sha256=sha256(evaluation_manifest).hexdigest(),
            sealed_outcome_artifact_ids=(
                f"artifact.selective-dependence-response.{slug}.evaluation-manifest",
                f"artifact.selective-dependence-response.{slug}.sealed-evaluation-panel",
            ),
            evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            fresh_evidence=True,
        ),
        obligations=_target_obligations(projection, system, cutoff),
        design_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
    )


def build_target_campaign(
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    system: SystemSpec,
    experiment: ExperimentSpec,
) -> CampaignSpec:
    slug = projection.target_slug
    node = CampaignNode(
        node_id=f"campaign-node.selective-dependence-response.{slug}.experiment",
        kind=CampaignNodeKind.EXPERIMENT_SPEC,
        lane=CampaignLane.PROSPECTIVE,
        object_identity=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        parent_node_ids=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    return CampaignSpec(
        campaign_id=f"campaign.selective-dependence-response.{slug}",
        objective=(
            "Determine the exact finite target-local law qualification law and controller admission action-fibre "
            "disposition without prospective controller evaluation or physical promotion."
        ),
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(value.claim_id for value in experiment.claims),
        budget=_PROGRAMME_BUDGET,
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id=f"decision-right.selective-dependence-response.{slug}.simulation",
                action=AuthorityAction.SIMULATION_EXECUTION,
                decision_maker_id=system.authority_policy.delegate_id,
                authority_policy_id=system.authority_policy.policy_id,
                delegated=True,
            ),
        ),
        nodes=(node,),
        root_node_ids=(node.node_id,),
        active_node_ids=(node.node_id,),
        evidence_state=(),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )


def _source_records(binding: SelectiveDependenceResponseTargetBinding) -> dict[str, tuple[str, CanonicalRecord]]:
    records: dict[str, tuple[str, CanonicalRecord]] = {
        "input.construct-review": (
            binding.construct_review.attestation_id,
            binding.construct_review,
        ),
        "input.contamination-ledger": (
            binding.contamination_ledger.ledger_id,
            binding.contamination_ledger,
        ),
        "input.method-question": (
            binding.method_question.freeze_id,
            binding.method_question,
        ),
        "input.preparation-freeze": (
            binding.preparation.freeze_id,
            binding.preparation,
        ),
        "input.target-analysis-freeze": (
            binding.analysis_freeze.freeze_id,
            binding.analysis_freeze,
        ),
        "input.target-design": (binding.design_id, binding.design),
    }
    if binding.source_qualification is not None:
        records["input.source-qualification"] = (
            str(getattr(binding.source_qualification, "qualification_id")),
            binding.source_qualification,
        )
    return records


def _source_role(role: ScientificInputRole) -> SourceMaterializationRole:
    return {
        ScientificInputRole.MODEL: SourceMaterializationRole.NUMERICAL_CONFIGURATION,
        ScientificInputRole.PREPARED_MEDIUM: SourceMaterializationRole.PREPARED_MEDIUM,
        ScientificInputRole.QUALIFICATION: SourceMaterializationRole.CALIBRATION,
        ScientificInputRole.RECEIVER: SourceMaterializationRole.RECEIVER_DEFINITION,
        ScientificInputRole.SOURCE: SourceMaterializationRole.OBSERVATION_STREAM,
    }[role]


def _authoring_sources(
    *,
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    binding: SelectiveDependenceResponseTargetBinding,
    stage_composition: SelectiveDependenceResponseStageComposition,
    system: SystemSpec,
    experiment: ExperimentSpec,
    template: StudyTemplate,
) -> tuple[
    tuple[SourceMaterializationConfig, ...],
    tuple[CanonicalRecord, ...],
    tuple[MaterializationQualificationReceipt, ...],
    tuple[SourceMaterializationRef, ...],
]:
    records = _source_records(binding)
    external = {value.input_id: value for value in template.graph.external_inputs}
    expected_source_ids = {
        input_id
        for input_id, value in external.items()
        if value.scientific_role
        in {
            ScientificInputRole.MODEL,
            ScientificInputRole.PREPARED_MEDIUM,
            ScientificInputRole.QUALIFICATION,
            ScientificInputRole.RECEIVER,
            ScientificInputRole.SOURCE,
        }
    }
    if set(records) != expected_source_ids:
        raise ValueError("authoring source record roster differs from the exact graph")
    observer = ObjectIdentity(
        object_id=(
            f"observer.selective-dependence-response.{projection.target_slug}."
            f"{stage_composition.stage_id.rsplit('.', 1)[-1]}.canonical-input"
        ),
        object_schema='empirical-lawhood/methods/selective-dependence-response/stage-authoring-observation-operator',
        object_version="1.0.0",
        object_fingerprint=stage_composition.implementation_sha256,
    )
    view_ids = tuple(value.view_id for value in system.numerical_views)
    native_units = tuple(sorted({value.native_unit for value in system.quantities}))
    frames = tuple(sorted({value.coordinate_frame for value in system.quantities}))
    clocks = tuple(value.clock_id for value in system.clocks)
    configs: list[SourceMaterializationConfig] = []
    qualifications: list[MaterializationQualificationReceipt] = []
    refs: list[SourceMaterializationRef] = []
    for input_id in sorted(records):
        record_id, record = records[input_id]
        spec = external[input_id]
        role = _source_role(spec.scientific_role)
        config = SourceMaterializationConfig(
            config_id=(
                f"source-config.selective-dependence-response.{projection.target_slug}."
                f"{stage_composition.stage_id.rsplit('.', 1)[-1]}.{input_id}"
            ),
            source_id=input_id,
            role=role,
            content_sha256=record.fingerprint(),
            expected_size_bytes=len(record.canonical_bytes()),
            maximum_bytes=spec.maximum_size_bytes,
            payload_schema=record.SCHEMA,
            media_type=spec.media_type,
            read_mode=SourceReadMode.ORDINARY_BOUNDED,
            outcome_access=spec.outcome_access,
            visibility_ceiling=spec.visibility_ceiling,
        )
        materialization = ObjectIdentity.from_record(record_id, record)
        qualification = MaterializationQualificationReceipt(
            receipt_id=(
                f"qualification.selective-dependence-response.{projection.target_slug}."
                f"{stage_composition.stage_id.rsplit('.', 1)[-1]}."
                f"{input_id}.authoring"
            ),
            source_id=input_id,
            materialization=materialization,
            content_sha256=record.fingerprint(),
            evidence_world_id=system.world.world_id,
            observation_operator=observer,
            numerical_view_ids=view_ids,
            native_unit_ids=native_units,
            frame_ids=frames,
            clock_ids=clocks,
            receiver_semantics_id=system.relation.relation_id,
            validity_contract_id=experiment.obligations.validity.validity_id,
            uncertainty_contract_id=experiment.obligations.uncertainty.uncertainty_id,
            access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
            outcome_access=spec.outcome_access,
            visibility_ceiling=spec.visibility_ceiling,
        )
        ref = SourceMaterializationRef(
            source_id=input_id,
            role=role,
            evidence_world_id=system.world.world_id,
            materialization=materialization,
            content_sha256=record.fingerprint(),
            source_config_sha256=config.fingerprint(),
            observation_operator=observer,
            numerical_view_ids=view_ids,
            qualification_receipt=ObjectIdentity.from_record(
                qualification.receipt_id,
                qualification,
            ),
            access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        )
        configs.append(config)
        qualifications.append(qualification)
        refs.append(ref)
    source_records = tuple(records[value][1] for value in sorted(expected_source_ids))
    return tuple(configs), source_records, tuple(qualifications), tuple(refs)


def _design_inputs(
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    binding: SelectiveDependenceResponseTargetBinding,
    cutoff: InformationCutoff,
) -> tuple[DesignInputRecord, ...]:
    if binding.method_completion is None:
        raise ValueError("authoring requires exact terminal method completion")
    records: list[
        tuple[
            str,
            CanonicalRecord,
            DesignInputRole,
            OutcomeAccess,
            VisibilityCeiling,
            tuple[str, ...],
            str,
        ]
    ] = [
        (
            binding.analysis_freeze.freeze_id,
            binding.analysis_freeze,
            DesignInputRole.READINESS_METADATA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            (),
            binding.construct_review.reviewer_id,
        ),
        (
            binding.construct_review.attestation_id,
            binding.construct_review,
            DesignInputRole.READINESS_METADATA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            (),
            binding.construct_review.reviewer_id,
        ),
        (
            binding.contamination_ledger.ledger_id,
            binding.contamination_ledger,
            DesignInputRole.MOTIVATION,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.OUTCOME_VISIBLE,
            (),
            binding.construct_review.reviewer_id,
        ),
        (
            binding.design_id,
            binding.design,
            DesignInputRole.READINESS_METADATA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            (),
            binding.construct_review.reviewer_id,
        ),
        (
            binding.method_completion.envelope_id,
            binding.method_completion,
            DesignInputRole.READINESS_METADATA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            (),
            binding.construct_review.reviewer_id,
        ),
        (
            binding.method_question.freeze_id,
            binding.method_question,
            DesignInputRole.READINESS_METADATA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            (),
            binding.construct_review.reviewer_id,
        ),
        (
            binding.preparation.freeze_id,
            binding.preparation,
            DesignInputRole.READINESS_METADATA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            (),
            binding.construct_review.reviewer_id,
        ),
    ]
    if binding.source_qualification is not None and binding.source_completion is not None:
        records.extend(
            (
                (
                    str(getattr(binding.source_qualification, "qualification_id")),
                    binding.source_qualification,
                    DesignInputRole.DEVELOPMENT_TUNING,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    binding.preparation.canary_unit_ids,
                    "service.selective-dependence-response-source-custody",
                ),
                (
                    binding.source_completion.envelope_id,
                    binding.source_completion,
                    DesignInputRole.DEVELOPMENT_TUNING,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    VisibilityCeiling.DEVELOPMENT_ONLY,
                    binding.preparation.canary_unit_ids,
                    "service.selective-dependence-response-source-custody",
                ),
            )
        )
    return tuple(
        sorted(
            (
                DesignInputRecord(
                    input_id=f"design-input.selective-dependence-response.{projection.target_slug}.{record_id}",
                    object_identity=ObjectIdentity.from_record(record_id, record),
                    materialization_sha256=record.fingerprint(),
                    information_cutoff=cutoff,
                    role=role,
                    outcome_access=access,
                    visibility_ceiling=visibility,
                    operator_id=operator_id,
                    physical_unit_ids=physical_unit_ids,
                )
                for (
                    record_id,
                    record,
                    role,
                    access,
                    visibility,
                    physical_unit_ids,
                    operator_id,
                ) in records
            ),
            key=lambda value: value.input_id,
        )
    )


def _complete_coverage(
    *,
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    stage: SelectiveDependenceResponseTargetStage,
    experiment: ExperimentSpec,
    template: StudyTemplate,
) -> StudyTemplate:
    bindings = {value.obligation_id: value for value in template.coverage.bindings}
    stage_slug = stage.value.lower().replace("_", "-")
    terminal_node_id = (
        "qualify-target-source"
        if stage is SelectiveDependenceResponseTargetStage.SOURCE_CANARY
        else "analyze-development"
    )
    terminal_step = next(
        value for value in template.protocol.steps if value.step_id == terminal_node_id
    )
    terminal_output_id = terminal_step.outputs[0].output_id
    incoming = tuple(
        sorted(
            value.edge_id
            for value in template.graph.edges
            if value.consumer_node_id == terminal_node_id
        )
    )
    for obligation_id in required_candidate_obligation_ids(
        experiment,
        template.protocol,
    ):
        bindings.setdefault(
            obligation_id,
            ObligationCoverageBinding(
                obligation_id=obligation_id,
                proof_owner_node_id=terminal_node_id,
                required_output_id=terminal_output_id,
                contributor_edge_ids=incoming,
            ),
        )
    return replace(
        template,
        coverage=ObligationCoverage(
            coverage_id=(f"coverage.selective-dependence-response.{projection.target_slug}.{stage_slug}.standard"),
            bindings=tuple(sorted(bindings.values(), key=lambda value: value.obligation_id)),
        ),
    )


def _formal_inventory(
    *,
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    binding: SelectiveDependenceResponseTargetBinding,
    stage: SelectiveDependenceResponseTargetStage,
    register: FormalGapRegister,
    system: SystemSpec,
    experiment: ExperimentSpec,
    source_refs: tuple[SourceMaterializationRef, ...],
) -> FormalGapSourceCapabilityInventory:
    stage_slug = stage.value.lower().replace("_", "-")
    units = (
        binding.preparation.canary_unit_ids
        if stage is SelectiveDependenceResponseTargetStage.SOURCE_CANARY
        else binding.preparation.development_unit_ids
    )
    return FormalGapSourceCapabilityInventory(
        inventory_id=(f"formal-source-inventory.selective-dependence-response.{projection.target_slug}.{stage_slug}"),
        denominator_id=system.system_id,
        evidence_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        source_materializations=tuple(
            sorted(
                (value.materialization for value in source_refs),
                key=lambda value: value.object_id,
            )
        ),
        present_operand_ids=(),
        satisfied_prerequisite_ids=(),
        independent_unit_ids=units,
        independent_unit_scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        numerical_view_ids=tuple(value.view_id for value in system.numerical_views),
        available_estimator_family_ids=(),
        available_control_ids=tuple(value.control_id for value in experiment.controls),
        multiplicity_family_ids=(),
        denominator_inapplicable_gap_ids=(),
        resource_blocked_gap_ids=tuple(value.gap_id for value in register.gaps),
        requested_claim_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _formal_coverage(
    *,
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    stage: SelectiveDependenceResponseTargetStage,
    register: FormalGapRegister,
    draft_id: str,
    inventory: FormalGapSourceCapabilityInventory,
) -> FormalGapCoverage:
    stage_slug = stage.value.lower().replace("_", "-")
    applicability = derive_formal_gap_applicability(register, inventory)
    assignments = tuple(
        FormalGapCoverageAssignment(
            gap_id=value.gap_id,
            disposition=FormalGapCoverageDisposition.DEFER_WITH_TYPED_PREREQUISITE,
            readiness_reason=ReadinessStatus.COMPUTABILITY_BOUNDARY,
            reason_codes=("FORMAL_LOWERING_NOT_IN_BOUNDED_SELECTIVE_DEPENDENCE_RESPONSE_STAGE",),
            selected_estimator_family_id=None,
            selected_control_ids=(),
            selected_multiplicity_family_id=None,
            obligation_ids=(),
            output_ids=(),
            adjudication_owner_ids=(),
        )
        for value in register.gaps
    )
    return FormalGapCoverage(
        coverage_id=(f"formal-gap-coverage.selective-dependence-response.{projection.target_slug}.{stage_slug}"),
        register=ObjectIdentity.from_record(register.register_id, register),
        denominator_id=inventory.denominator_id,
        candidate_act_id=draft_id,
        applicability=applicability,
        assignments=assignments,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _entry_package(
    *,
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    stage: SelectiveDependenceResponseTargetStage,
    draft: StudyDraft,
    register: FormalGapRegister,
    coverage: FormalGapCoverage,
) -> ExperimentEntryPackage:
    stage_slug = stage.value.lower().replace("_", "-")
    bindings = tuple(
        ExperimentEntryRequirementBinding(
            requirement=requirement,
            object_ids=(
                f"binding.selective-dependence-response.{projection.target_slug}.{stage_slug}."
                f"{requirement.value.lower()}",
            ),
            readiness=(
                ReadinessStatus.AUTHORITY_REQUIRED
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ReadinessStatus.READY
            ),
            reason_codes=(
                ("SEPARATE_SIMULATION_AND_REVEAL_AUTHORITY_REQUIRED",)
                if requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
                else ()
            ),
        )
        for requirement in sorted(ExperimentEntryRequirement, key=lambda value: value.value)
    )
    checklist = ExperimentEntryChecklist(
        checklist_id=(f"entry-checklist.selective-dependence-response.{projection.target_slug}.{stage_slug}"),
        draft=ObjectIdentity.from_record(draft.draft_id, draft),
        formal_gap_register=ObjectIdentity.from_record(register.register_id, register),
        formal_gap_coverage=ObjectIdentity.from_record(coverage.coverage_id, coverage),
        portfolio_world=FormalGapEvidenceWorld.NUMERICAL_SIMULATOR,
        execution_route_id="route.standard-candidate-issue-compile-execute",
        durability_disposition_id="durability.external-receipt-first-seamios",
        bindings=bindings,
        transitions=tuple(sorted(ExperimentEntryTransition, key=lambda value: value.value)),
        allowed_terminal_classes=tuple(
            sorted(ExperimentTerminalClass, key=lambda value: value.value)
        ),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return ExperimentEntryPackage(
        package_id=(f"experiment-entry-package.selective-dependence-response.{projection.target_slug}.{stage_slug}"),
        register=register,
        coverage=coverage,
        checklist=checklist,
    )


def build_target_source_authoring_act(
    *,
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    register: FormalGapRegister,
) -> SelectiveDependenceResponseTargetAuthoringAct:
    """Compile one source/canary package to authority-pending, without I/O."""

    validate_sha256(
        projection.source_composition.implementation_sha256,
        field_name="implementation_sha256",
    )
    if projection.binding.method_completion is None:
        raise ValueError("source authoring cannot bypass terminal method completion")
    system = build_target_system(projection)
    experiment = build_target_experiment(projection, system)
    campaign = build_target_campaign(projection, system, experiment)
    base_template = projection.source_composition.catalog.templates[0]
    template = _complete_coverage(
        projection=projection,
        stage=SelectiveDependenceResponseTargetStage.SOURCE_CANARY,
        experiment=experiment,
        template=base_template,
    )
    catalog = CandidateCapabilityCatalog(
        catalog_id=(f"selective-dependence-response-{projection.target_slug}-source-canary-authoring-catalog"),
        registrations=projection.source_composition.catalog.registrations,
        templates=(template,),
    )
    source_configs, source_records, qualifications, source_refs = _authoring_sources(
        projection=projection,
        binding=projection.binding,
        stage_composition=projection.source_composition,
        system=system,
        experiment=experiment,
        template=template,
    )
    cutoff = experiment.information_cutoffs[0]
    design_inputs = _design_inputs(projection, projection.binding, cutoff)
    draft_id = f"draft.selective-dependence-response.{projection.target_slug}.source-canary"
    inventory = _formal_inventory(
        projection=projection,
        binding=projection.binding,
        stage=SelectiveDependenceResponseTargetStage.SOURCE_CANARY,
        register=register,
        system=system,
        experiment=experiment,
        source_refs=source_refs,
    )
    formal_coverage = _formal_coverage(
        projection=projection,
        stage=SelectiveDependenceResponseTargetStage.SOURCE_CANARY,
        register=register,
        draft_id=draft_id,
        inventory=inventory,
    )
    nomination_input = next(
        value
        for value in design_inputs
        if value.object_identity.object_schema == SelectiveDependenceResponseContaminationLedger.SCHEMA
    )
    draft = StudyDraft(
        draft_id=draft_id,
        lifecycle=StudyDraftLifecycle.DRAFT,
        question="; ".join(value.proposition for value in experiment.claims),
        alternative_ids=(
            f"alternative.selective-dependence-response.{projection.target_slug}.mixed",
            f"alternative.selective-dependence-response.{projection.target_slug}.not-supported",
            f"alternative.selective-dependence-response.{projection.target_slug}.supported",
            f"alternative.selective-dependence-response.{projection.target_slug}.unevaluable",
        ),
        design_origin=DesignOrigin(
            origin_id=f"origin.selective-dependence-response.{projection.target_slug}.visible-nomination",
            kind=DesignOriginKind.PROSPECTIVE_NOMINATION,
            declared_input_ids=tuple(value.input_id for value in design_inputs),
            parent_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            nomination=nomination_input.object_identity,
            requests_fresh_child=True,
        ),
        design_inputs=design_inputs,
        development_unit_ids=experiment.reveal_barrier.development_unit_ids,
        evaluation_unit_ids=projection.binding.preparation.evaluation_unit_ids,
        development_seed_ids=tuple(
            f"seed.{value}" for value in experiment.reveal_barrier.development_unit_ids
        ),
        evaluation_seed_ids=tuple(
            f"seed.{value}" for value in projection.binding.preparation.evaluation_unit_ids
        ),
        unresolved_decisions=(),
        system=system,
        experiment=experiment,
        campaign=campaign,
        dag_template_key=template.template_key,
        capability_selections=tuple(
            CapabilitySelection(
                capability_key=value.capability_key,
                capability_version=value.capability_version,
                implementation_sha256=value.implementation_sha256,
            )
            for value in projection.source_composition.registry.capabilities
        ),
        source_materializations=source_refs,
        resource_ceiling=_PROGRAMME_BUDGET,
    )
    entry = _entry_package(
        projection=projection,
        stage=SelectiveDependenceResponseTargetStage.SOURCE_CANARY,
        draft=draft,
        register=register,
        coverage=formal_coverage,
    )
    package = StudyDefinition(
        package_id=(
            f"programme-authoring-package.selective-dependence-response.{projection.target_slug}.source-canary"
        ),
        draft=draft,
        entry_package=entry,
    )
    base_context = CandidateCompilationContext(
        context_id=(f"candidate-context.selective-dependence-response.{projection.target_slug}.source-canary"),
        registry=projection.source_composition.registry,
        templates=(template,),
        qualifications=qualifications,
        known_design_inputs=design_inputs,
        implementation_sha256=projection.source_composition.implementation_sha256,
    )
    context = StandardCandidateCompilationContext(
        context_id=(f"standard-context.selective-dependence-response.{projection.target_slug}.source-canary"),
        base=base_context,
        formal_methods=standard_formal_method_catalog(register),
        source_inventories=(inventory,),
    )
    compilation = compile_study_candidate(
        authoring_package=package,
        authoring_materialization=AuthoringMaterializationIdentity(
            media_type="application/json",
            byte_count=len(package.canonical_bytes()),
            raw_materialization_sha256=package.fingerprint(),
        ),
        context=context,
    )
    return SelectiveDependenceResponseTargetAuthoringAct(
        projection=projection,
        stage_composition=projection.source_composition,
        system=system,
        experiment=experiment,
        campaign=campaign,
        source_configs=source_configs,
        source_records=source_records,
        qualifications=qualifications,
        inventory=inventory,
        template=template,
        catalog=catalog,
        draft=draft,
        package=package,
        compilation=compilation,
    )


def build_target_development_authoring_act(
    *,
    projection: SelectiveDependenceResponseTargetPlanningProjection,
    source_qualification: CanonicalRecord,
    source_completion: SelectiveDependenceResponseSourceCanaryCompletionEnvelope,
    register: FormalGapRegister,
) -> SelectiveDependenceResponseTargetAuthoringAct:
    """Compile one development-only package from exact terminal source custody."""

    implementation_sha256 = projection.source_composition.implementation_sha256
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    if projection.binding.method_completion is None:
        raise ValueError("development authoring cannot bypass terminal method completion")
    if not isinstance(source_qualification, projection.binding.source_qualification_type):
        raise ValueError("development source qualification type differs")
    if source_completion.source_implementation_sha256 != implementation_sha256:
        raise ValueError("development source implementation identity differs")
    development_binding = replace(
        projection.binding,
        source_qualification=source_qualification,
        source_completion=source_completion,
    )
    stage_composition = compose_target_development(
        projection.binding,
        source_qualification=source_qualification,
        source_completion=source_completion,
        implementation_sha256=implementation_sha256,
    )
    system = build_target_system(projection)
    experiment = build_target_experiment(projection, system)
    campaign = build_target_campaign(projection, system, experiment)
    base_template = stage_composition.catalog.templates[0]
    template = _complete_coverage(
        projection=projection,
        stage=SelectiveDependenceResponseTargetStage.DEVELOPMENT,
        experiment=experiment,
        template=base_template,
    )
    catalog = CandidateCapabilityCatalog(
        catalog_id=(f"selective-dependence-response-{projection.target_slug}-development-authoring-catalog"),
        registrations=stage_composition.catalog.registrations,
        templates=(template,),
    )
    source_configs, source_records, qualifications, source_refs = _authoring_sources(
        projection=projection,
        binding=development_binding,
        stage_composition=stage_composition,
        system=system,
        experiment=experiment,
        template=template,
    )
    cutoff = experiment.information_cutoffs[0]
    design_inputs = _design_inputs(projection, development_binding, cutoff)
    draft_id = f"draft.selective-dependence-response.{projection.target_slug}.development"
    inventory = _formal_inventory(
        projection=projection,
        binding=development_binding,
        stage=SelectiveDependenceResponseTargetStage.DEVELOPMENT,
        register=register,
        system=system,
        experiment=experiment,
        source_refs=source_refs,
    )
    formal_coverage = _formal_coverage(
        projection=projection,
        stage=SelectiveDependenceResponseTargetStage.DEVELOPMENT,
        register=register,
        draft_id=draft_id,
        inventory=inventory,
    )
    nomination_input = next(
        value
        for value in design_inputs
        if value.object_identity.object_schema == SelectiveDependenceResponseContaminationLedger.SCHEMA
    )
    draft = StudyDraft(
        draft_id=draft_id,
        lifecycle=StudyDraftLifecycle.DRAFT,
        question="; ".join(value.proposition for value in experiment.claims),
        alternative_ids=(
            f"alternative.selective-dependence-response.{projection.target_slug}.mixed",
            f"alternative.selective-dependence-response.{projection.target_slug}.not-supported",
            f"alternative.selective-dependence-response.{projection.target_slug}.supported",
            f"alternative.selective-dependence-response.{projection.target_slug}.unevaluable",
        ),
        design_origin=DesignOrigin(
            origin_id=(f"origin.selective-dependence-response.{projection.target_slug}.development-visible-source"),
            kind=DesignOriginKind.PROSPECTIVE_NOMINATION,
            declared_input_ids=tuple(value.input_id for value in design_inputs),
            parent_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            nomination=nomination_input.object_identity,
            requests_fresh_child=True,
        ),
        design_inputs=design_inputs,
        development_unit_ids=experiment.reveal_barrier.development_unit_ids,
        evaluation_unit_ids=development_binding.preparation.evaluation_unit_ids,
        development_seed_ids=tuple(
            f"seed.{value}" for value in experiment.reveal_barrier.development_unit_ids
        ),
        evaluation_seed_ids=tuple(
            f"seed.{value}" for value in development_binding.preparation.evaluation_unit_ids
        ),
        unresolved_decisions=(),
        system=system,
        experiment=experiment,
        campaign=campaign,
        dag_template_key=template.template_key,
        capability_selections=tuple(
            CapabilitySelection(
                capability_key=value.capability_key,
                capability_version=value.capability_version,
                implementation_sha256=value.implementation_sha256,
            )
            for value in stage_composition.registry.capabilities
        ),
        source_materializations=source_refs,
        resource_ceiling=_PROGRAMME_BUDGET,
    )
    entry = _entry_package(
        projection=projection,
        stage=SelectiveDependenceResponseTargetStage.DEVELOPMENT,
        draft=draft,
        register=register,
        coverage=formal_coverage,
    )
    package = StudyDefinition(
        package_id=(
            f"programme-authoring-package.selective-dependence-response.{projection.target_slug}.development"
        ),
        draft=draft,
        entry_package=entry,
    )
    base_context = CandidateCompilationContext(
        context_id=f"candidate-context.selective-dependence-response.{projection.target_slug}.development",
        registry=stage_composition.registry,
        templates=(template,),
        qualifications=qualifications,
        known_design_inputs=design_inputs,
        implementation_sha256=implementation_sha256,
    )
    context = StandardCandidateCompilationContext(
        context_id=f"standard-context.selective-dependence-response.{projection.target_slug}.development",
        base=base_context,
        formal_methods=standard_formal_method_catalog(register),
        source_inventories=(inventory,),
    )
    compilation = compile_study_candidate(
        authoring_package=package,
        authoring_materialization=AuthoringMaterializationIdentity(
            media_type="application/json",
            byte_count=len(package.canonical_bytes()),
            raw_materialization_sha256=package.fingerprint(),
        ),
        context=context,
    )
    return SelectiveDependenceResponseTargetAuthoringAct(
        projection=projection,
        stage_composition=stage_composition,
        system=system,
        experiment=experiment,
        campaign=campaign,
        source_configs=source_configs,
        source_records=source_records,
        qualifications=qualifications,
        inventory=inventory,
        template=template,
        catalog=catalog,
        draft=draft,
        package=package,
        compilation=compilation,
    )


__all__ = [
    'SelectiveDependenceResponseTargetAuthoringAct',
    'SelectiveDependenceResponseTargetPlanningProjection',
    "build_target_campaign",
    "build_target_development_authoring_act",
    "build_target_experiment",
    "build_target_source_authoring_act",
    "build_target_system",
]
