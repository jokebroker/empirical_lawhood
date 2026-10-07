"""Canonical Prospective conformance over truth-known fresh-evidence branches."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import ClassVar

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from empirical_lawhood.adapters.reference_worlds import ReferenceWorldKind, get_reference_world
from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget, SourceAccessClass
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.experiments import AssignmentKind, ControlKind, ControlSpec, PrecisionGoal
from empirical_lawhood.kernel.provenance import (
    EvidenceLink,
    EvidenceRelation,
    EvidenceSnapshot,
    ObjectIdentity,
)
from empirical_lawhood.kernel.references import ArtifactIdentity, QuantityBound
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import LifecycleStatus, ScientificStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.approval import (
    APPROVAL_SIGNATURE_VERSION,
    ApprovalCheckerRegistration,
    ApprovalCheckerRegistry,
    ApprovalGateAttestation,
    ApprovalGateKind,
    ApprovalSignatureAlgorithm,
    AttestationResult,
    CompleteApprovalService,
    DurableAuthorizationRecord,
    FrozenApprovalProposal,
    issue_approval_gate_attestation,
)
from empirical_lawhood.planning.authority import (
    AuthorizationDecision,
)
from empirical_lawhood.planning.campaigns import (
    CampaignLane,
    CampaignNode,
    CampaignNodeKind,
    CampaignSpec,
    DecisionRight,
)
from empirical_lawhood.planning.discovery import (
    CampaignGraph,
    DiscoveryEdge,
    DiscoveryEdgeKind,
    DiscoveryLane,
    DiscoveryNode,
    DiscoveryNodeKind,
)
from empirical_lawhood.planning.exploration import (
    AnalysisAttempt,
    AnalysisAttemptStatus,
    ExplorationPlan,
    ExploratoryFinding,
    ExploratoryFindingStatus,
    Hypothesis,
    HypothesisDisposition,
    HypothesisSet,
    OutcomeInterpretationKind,
    ProspectiveNomination,
)
from empirical_lawhood.planning.prospective import (
    NativeMeasurementRequirement,
    NominationDecisionKind,
    NominationObligations,
    ProspectiveAdjudication,
    ProspectiveControlRequirement,
    ProspectiveControlRole,
    ProspectiveDesignContext,
    ProspectiveDesignKind,
    ProspectiveEvidenceContract,
    ProspectiveEvidenceResult,
    ProspectiveObjectiveVector,
    nomination_obligations_extension,
)
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.compiler import compile_run_plan, lower_run_plan
from empirical_lawhood.runtime.plans import BarrierKind, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ProtocolRunPlan, ScientificStage

from .adjudication import ProspectiveAdjudicator
from .designers import DesignedExperiment, default_designer_registry
from .nominator import NominationBatch, ProspectiveNominator
from .registry import prospective_capability_keys, prospective_capability_registry


@dataclass(frozen=True, slots=True)
class ProspectiveConformanceOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/prospective/prospective-conformance-outcome'

    case_id: str
    reference_world_id: str
    nomination_decision: NominationDecisionKind
    design_kind: ProspectiveDesignKind | None
    authorization_decision: AuthorizationDecision | None
    fresh_status: ScientificStatus | None
    adjudication_disposition_ids: tuple[str, ...]
    fresh_independent_unit_count: int
    run_compiled: bool
    source_fingerprints_preserved: bool
    outcome_fingerprint: str

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        validate_stable_id(self.reference_world_id, field_name="reference_world_id")
        require_sorted_unique_strings(
            self.adjudication_disposition_ids,
            field_name="adjudication_disposition_ids",
        )
        if self.fresh_independent_unit_count < 0:
            raise ValueError("fresh independent-unit count must be nonnegative")
        validate_sha256(self.outcome_fingerprint, field_name="outcome_fingerprint")
        stopped = self.nomination_decision is NominationDecisionKind.STOP_NO_ACQUISITION
        if stopped:
            if any(
                value is not None
                for value in (
                    self.design_kind,
                    self.authorization_decision,
                    self.fresh_status,
                )
            ):
                raise ValueError("no-acquisition outcome cannot contain execution state")
            if self.run_compiled or self.fresh_independent_unit_count:
                raise ValueError("no-acquisition outcome cannot compile or count fresh units")
        elif not self.run_compiled or self.fresh_status is None:
            raise ValueError("fresh-evidence conformance must compile and grade a run")
        if not self.source_fingerprints_preserved:
            raise ValueError("Prospective conformance cannot rewrite motivating evidence")


@dataclass(frozen=True, slots=True)
class ProspectiveReferenceConformanceReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/prospective/prospective-reference-conformance-report'

    report_id: str
    registry_fingerprint: str
    capability_keys: tuple[str, ...]
    designer_kind_ids: tuple[str, ...]
    authorization_branch_ids: tuple[str, ...]
    outcomes: tuple[ProspectiveConformanceOutcome, ...]
    fresh_evidence_graph: CampaignGraph
    physical_observation_to_controller_use_execution: str
    status: str

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        validate_sha256(self.registry_fingerprint, field_name="registry_fingerprint")
        for field_name, values in (
            ("capability_keys", self.capability_keys),
            ("designer_kind_ids", self.designer_kind_ids),
            ("authorization_branch_ids", self.authorization_branch_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        require_sorted_unique_ids(self.outcomes, attribute="case_id", field_name="outcomes")
        validate_nonempty(self.physical_observation_to_controller_use_execution, field_name="physical_observation_to_controller_use_execution")
        validate_nonempty(self.status, field_name="status")
        if set(self.designer_kind_ids) != {kind.value for kind in ProspectiveDesignKind}:
            raise ValueError("Prospective report must exercise the exact four design families")
        expected_authorization = {
            AuthorizationDecision.APPROVED_NONACTUATING.value,
            AuthorizationDecision.AUTHORITY_REQUIRED.value,
            AuthorizationDecision.REFUSED.value,
        }
        if set(self.authorization_branch_ids) != expected_authorization:
            raise ValueError("Prospective report must retain approve, refuse and external-authority paths")
        if self.physical_observation_to_controller_use_execution != "NONE":
            raise ValueError("reference conformance cannot claim physical measurement through controller-use evidence")
        if self.status != "PASS":
            raise ValueError("canonical Prospective conformance report must pass")


@dataclass(frozen=True, slots=True)
class _PreparedSource:
    system: SystemSpec
    snapshot_identity: ObjectIdentity
    wave_identity: ObjectIdentity
    finding: ExploratoryFinding
    hypotheses: HypothesisSet
    fingerprints: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CompletedCase:
    outcome: ProspectiveConformanceOutcome
    source: _PreparedSource
    batch: NominationBatch
    designed: DesignedExperiment
    run_plan: ProtocolRunPlan
    result: ProspectiveEvidenceResult
    adjudication: ProspectiveAdjudication
    campaign: CampaignSpec


class _ConformanceClock:
    clock_id = "prospective-conformance-clock"

    def now_utc(self) -> str:
        return "2026-07-15T00:00:00Z"


class _FrozenApprovalStore:
    def __init__(self) -> None:
        self.value: FrozenApprovalProposal | None = None

    def persist(self, frozen: FrozenApprovalProposal) -> ObjectIdentity:
        self.value = frozen
        return ObjectIdentity.from_record(frozen.frozen_proposal_id, frozen)

    def load(self, frozen_proposal_id: str) -> FrozenApprovalProposal:
        if self.value is None or self.value.frozen_proposal_id != frozen_proposal_id:
            raise KeyError(frozen_proposal_id)
        return self.value


class _AuthorizationStore:
    def __init__(self) -> None:
        self.value: DurableAuthorizationRecord | None = None

    def persist(self, record: DurableAuthorizationRecord) -> ObjectIdentity:
        self.value = record
        return ObjectIdentity.from_record(record.authorization_id, record)

    def load(self, authorization_id: str) -> DurableAuthorizationRecord:
        if self.value is None or self.value.authorization_id != authorization_id:
            raise KeyError(authorization_id)
        return self.value


class _AttestationStore:
    def __init__(self, values: tuple[ApprovalGateAttestation, ...] = ()) -> None:
        self.values = {value.attestation_id: value for value in values}

    def load(self, attestation_id: str) -> ApprovalGateAttestation:
        return self.values[attestation_id]


class _ConformanceAttestationSigner:
    """Deterministic private issuer confined to reference conformance fixtures."""

    signature_algorithm = ApprovalSignatureAlgorithm.ED25519
    signature_version = APPROVAL_SIGNATURE_VERSION

    def __init__(self, label: str) -> None:
        seed = hashlib.sha256(f"prospective-conformance-signer:{label}".encode()).digest()
        self._private_key = Ed25519PrivateKey.from_private_bytes(seed)
        self.verification_key_hex = (
            self._private_key.public_key()
            .public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            .hex()
        )

    def sign(self, payload: bytes) -> bytes:
        return self._private_key.sign(payload)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=100_000_000,
        gpu_devices=0,
        wall_time_seconds=30,
        source_scan_bytes=0,
        output_bytes=100_000,
    )


def _prepare_source(case_id: str, kind: ReferenceWorldKind) -> _PreparedSource:
    world = get_reference_world(kind)
    system = world.system
    system_identity = ObjectIdentity.from_record(system.system_id, system)
    snapshot_identity = ObjectIdentity(
        object_id=f"snapshot.prospective-source.{case_id}",
        object_schema=EvidenceSnapshot.SCHEMA,
        object_version="1.0.0",
        object_fingerprint=_digest(f"prospective-source-snapshot:{case_id}:{world.fingerprint()}"),
    )
    wave_identity = ObjectIdentity(
        object_id=f"wave.prospective-source.{case_id}",
        object_schema=ExplorationPlan.SCHEMA,
        object_version="1.0.0",
        object_fingerprint=_digest(f"prospective-source-wave:{case_id}:{world.fingerprint()}"),
    )
    finding = ExploratoryFinding(
        finding_id=f"finding.prospective-source.{case_id}",
        plan_id=wave_identity.object_id,
        proposal_id=f"proposal.prospective-source.{case_id}",
        status=ExploratoryFindingStatus.PATTERN,
        attempts=(
            AnalysisAttempt(
                attempt_id=f"attempt.prospective-source.{case_id}",
                proposal_id=f"proposal.prospective-source.{case_id}",
                analysis_id=f"analysis.prospective-source.{case_id}",
                family_coordinate_ids=("registered-reference-coordinate",),
                status=AnalysisAttemptStatus.SUCCEEDED,
                artifact_ids=(f"artifact.prospective-source.{case_id}",),
                reason_codes=(),
            ),
        ),
        effects=(),
        interpretation="The outcome-visible reference pattern motivates a fresh adjudication.",
        limitation_codes=("outcome-visible",),
        evidence_links=(
            EvidenceLink(
                link_id=f"evidence-link.prospective-source.{case_id}",
                relation=EvidenceRelation.DERIVED_FROM,
                source=system_identity,
                target=system_identity,
                artifact_ids=(f"artifact.prospective-source.{case_id}",),
                world_id=system.world.world_id,
                information_cutoff_id=f"cutoff.prospective-source.{case_id}",
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                parent_visibility_ceilings=(VisibilityCeiling.OUTCOME_VISIBLE,),
                reason="The finding is a read-only view of completed reference evidence.",
            ),
        ),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        parent_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    hypotheses = HypothesisSet(
        hypothesis_set_id=f"hypotheses.prospective-source.{case_id}",
        finding_ids=(finding.finding_id,),
        hypotheses=(
            Hypothesis(
                hypothesis_id=f"hypothesis.relational-response.{case_id}",
                statement="The declared action produces the relational response.",
                disposition=HypothesisDisposition.PREFERRED,
                supporting_finding_ids=(finding.finding_id,),
                opposing_observation="Fresh assigned actions fail to reproduce the response.",
                missing_evidence="Fresh assigned independent units with sealed outcomes.",
            ),
            Hypothesis(
                hypothesis_id=f"hypothesis.retrospective-selection.{case_id}",
                statement="Outcome-visible selection produced the apparent response.",
                disposition=HypothesisDisposition.UNRESOLVED,
                supporting_finding_ids=(finding.finding_id,),
                opposing_observation="The effect survives the sealed prospective assignment.",
                missing_evidence="A separately authorized prospective replication.",
            ),
        ),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        parent_visibility_ceilings=(VisibilityCeiling.OUTCOME_VISIBLE,),
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    return _PreparedSource(
        system=system,
        snapshot_identity=snapshot_identity,
        wave_identity=wave_identity,
        finding=finding,
        hypotheses=hypotheses,
        fingerprints=(
            snapshot_identity.object_fingerprint,
            wave_identity.object_fingerprint,
            finding.fingerprint(),
            hypotheses.fingerprint(),
        ),
    )


def _context(
    source: _PreparedSource, case_id: str, *, acquirable: bool
) -> ProspectiveDesignContext:
    quantities = {quantity.quantity_id: quantity for quantity in source.system.quantities}
    receiver = quantities[source.system.relation.receiver_quantity_ids[0]]
    sink = quantities["sink"]
    action = quantities[source.system.relation.action_quantity_ids[0]]
    kind = ProspectiveDesignKind.MODEL_DISCRIMINATION
    return ProspectiveDesignContext(
        context_id=f"prospective-context.{case_id}",
        source_snapshot=source.snapshot_identity,
        system=ObjectIdentity.from_record(source.system.system_id, source.system),
        design_kinds=(kind,),
        evidence_contract=(
            ProspectiveEvidenceContract(
                contract_id=f"fresh-evidence-contract.{case_id}",
                development_unit_ids=(f"development-unit.{case_id}.1",),
                fresh_independent_unit_ids=(
                    f"fresh-unit.{case_id}.1",
                    f"fresh-unit.{case_id}.2",
                ),
                evaluation_cohort_id=f"evaluation-cohort.{case_id}",
                evaluation_manifest_sha256=_digest(f"evaluation-cohort:{case_id}"),
                sealed_outcome_artifact_ids=(f"sealed-outcomes.{case_id}",),
            )
            if acquirable
            else None
        ),
        assignment_kind=AssignmentKind.RANDOMIZED_INTERVENTION,
        assignment_mechanism="Assign the frozen native action contrast by preparation.",
        assignment_support_restriction_ids=(),
        randomization_unit_id=source.system.independent_unit.unit_id,
        measurements=tuple(
            sorted(
                (
                    NativeMeasurementRequirement(
                        requirement_id=f"measurement.receiver.{case_id}",
                        quantity_id=receiver.quantity_id,
                        native_unit=receiver.native_unit,
                        clock_id=receiver.clock_id,
                    ),
                    NativeMeasurementRequirement(
                        requirement_id=f"measurement.sink.{case_id}",
                        quantity_id=sink.quantity_id,
                        native_unit=sink.native_unit,
                        clock_id=sink.clock_id,
                    ),
                ),
                key=lambda item: item.requirement_id,
            )
        ),
        action_bounds=(
            QuantityBound(
                bound_id=f"action-bound.{case_id}",
                quantity_id=action.quantity_id,
                native_unit=action.native_unit,
                lower=Decimal("-1"),
                upper=Decimal("1"),
            ),
        ),
        denominator_cell_ids=(f"denominator-cell.{case_id}",),
        controls=(
            ProspectiveControlRequirement(
                requirement_id=f"control-requirement.negative.{case_id}",
                role=ProspectiveControlRole.NEGATIVE,
                control=ControlSpec(
                    control_id=f"control.negative.{case_id}",
                    kind=ControlKind.NEGATIVE_ACTION,
                    capability_key="reference.negative-action",
                    target_quantity_ids=(receiver.quantity_id,),
                    decisive_rule="The negative action must not retain the declared effect.",
                ),
            ),
            ProspectiveControlRequirement(
                requirement_id=f"control-requirement.positive.{case_id}",
                role=ProspectiveControlRole.POSITIVE,
                control=ControlSpec(
                    control_id=f"control.positive.{case_id}",
                    kind=ControlKind.BASELINE_COMPARATOR,
                    capability_key="reference.positive-baseline",
                    target_quantity_ids=(receiver.quantity_id,),
                    decisive_rule="The known reference contrast must remain detectable.",
                ),
            ),
        ),
        admission_gate_ids=(f"receiver-gate.{case_id}", f"sink-gate.{case_id}"),
        precision_goals=(
            PrecisionGoal(
                goal_id=f"precision-goal.{case_id}",
                metric_id="independent-unit-interval-width",
                target_width=Decimal("0.2"),
                native_unit=receiver.native_unit,
                maximum_independent_units=20,
                stopping_rule="Stop at target precision or the frozen independent-unit cap.",
            ),
        ),
        objectives=(
            ProspectiveObjectiveVector(
                objective_id=f"objective.model-discrimination.{case_id}",
                design_kind=kind,
                falsification_value=Decimal("0.9"),
                hypothesis_discrimination=Decimal("0.9"),
                support_or_rank_gain=Decimal("0.7"),
                acquisition_cost=Decimal("0.2"),
                safety_or_authority_risk=Decimal("0.1"),
            ),
        ),
        alternative_design_ids=(f"alternative.factorial.{case_id}",),
        safety_constraint_ids=("reference-action-bounds", "resource-envelope"),
        budget=_budget(),
        information_cutoff=InformationCutoff(
            cutoff_id=f"fresh-pre-action-cutoff.{case_id}",
            clock_id=action.clock_id,
            phase=CausalPhase.PRE_ACTION,
            coordinate=Decimal(0),
        ),
        authority_action=AuthorityAction.REFERENCE_WORLD_EXECUTION,
        source_access=SourceAccessClass.NONE,
        claim_proposition="Fresh assignment supports the declared relational response.",
        claim_estimand="Fresh independent-unit receiver contrast in native coordinates.",
        claim_promotion_rule="All frozen support, control, falsifier and precision gates pass.",
        claim_assumption_ids=("fresh-independent-units", "sealed-evaluation"),
        requested_rung=EvidenceRung.LOCAL_LAW,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        no_acquisition_reason_codes=() if acquirable else ("fresh-evidence-unavailable",),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )


def _protocol(case_id: str) -> tuple[ProtocolTemplate, CapabilityRegistry]:
    step_budget = ResourceBudget(
        cpu_cores=1,
        memory_bytes=10_000_000,
        gpu_devices=0,
        wall_time_seconds=5,
        source_scan_bytes=0,
        output_bytes=10_000,
    )

    def config(step: str, schema: str) -> CapabilityConfigRef:
        return CapabilityConfigRef(
            config_id=f"config.{step}.{case_id}",
            config_schema=schema,
            config_schema_sha256=_digest(schema),
            content_sha256=_digest(f"config:{step}:{case_id}"),
            artifact_id=f"config-artifact.{step}.{case_id}",
        )

    frozen_schema = 'empirical-lawhood/prospective/synthetic-frozen-reference-design'
    freeze = ProtocolStepTemplate(
        step_id="freeze",
        stage=ScientificStage.FREEZE,
        capability_key="prospective.reference-freeze",
        capability_version="1.0.0",
        config=config("freeze", frozen_schema),
        dependency_step_ids=(),
        outputs=(
            OutputTemplate(
                output_id="frozen-design",
                payload_schema=frozen_schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                filename_suffix=".json",
            ),
        ),
        required_permissions=(),
        requested_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        resource_budget=step_budget,
        resource_lock_ids=("freeze-writer",),
        barrier=BarrierKind.FREEZE,
        maximum_attempts=1,
        obligation_ids=("complete-protocol-freeze",),
    )
    evaluator_permissions = tuple(
        sorted(
            (
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            )
        )
    )
    evaluate = ProtocolStepTemplate(
        step_id="evaluate",
        stage=ScientificStage.EVALUATE,
        capability_key="prospective.reference-evaluator",
        capability_version="1.0.0",
        config=config("evaluate", ProspectiveEvidenceResult.SCHEMA),
        dependency_step_ids=("freeze",),
        outputs=(
            OutputTemplate(
                output_id="fresh-result",
                payload_schema=ProspectiveEvidenceResult.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                filename_suffix=".json",
            ),
        ),
        required_permissions=evaluator_permissions,
        requested_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        resource_budget=step_budget,
        resource_lock_ids=("sealed-evaluator",),
        barrier=BarrierKind.REVEAL,
        maximum_attempts=1,
        obligation_ids=("sealed-fresh-evaluation",),
    )
    template = ProtocolTemplate(
        template_id=f"prospective-reference-protocol.{case_id}",
        template_version="1.0.0",
        steps=(evaluate, freeze),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )
    manifests = (
        CapabilityManifest(
            capability_key=freeze.capability_key,
            capability_version="1.0.0",
            kind=CapabilityKind.TRANSFORM,
            config_schema=freeze.config.config_schema,
            config_schema_sha256=freeze.config.config_schema_sha256,
            input_schema_ids=(),
            output_schema_ids=(frozen_schema,),
            permissions=(),
            maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
            resource_ceiling=step_budget,
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11",
            requires_clean_commit=True,
            requires_active_mount=False,
            requires_network=False,
            conformance_check_ids=("prospective-complete-freeze",),
            implementation_sha256=_digest("prospective.reference-freeze"),
        ),
        CapabilityManifest(
            capability_key=evaluate.capability_key,
            capability_version="1.0.0",
            kind=CapabilityKind.EVALUATOR,
            config_schema=evaluate.config.config_schema,
            config_schema_sha256=evaluate.config.config_schema_sha256,
            input_schema_ids=(frozen_schema,),
            output_schema_ids=(ProspectiveEvidenceResult.SCHEMA,),
            permissions=evaluator_permissions,
            maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            resource_ceiling=step_budget,
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11",
            requires_clean_commit=True,
            requires_active_mount=False,
            requires_network=False,
            conformance_check_ids=("prospective-sealed-evaluator-only",),
            implementation_sha256=_digest("prospective.reference-evaluator"),
        ),
    )
    return template, CapabilityRegistry(
        registry_id=f"prospective-reference-runtime.{case_id}",
        capabilities=tuple(sorted(manifests, key=lambda item: item.registry_id)),
    )


def _campaign(
    case_id: str,
    system: SystemSpec,
    designed: DesignedExperiment,
) -> CampaignSpec:
    experiment = designed.proposal.candidate_experiment
    experiment_identity = ObjectIdentity.from_record(experiment.experiment_id, experiment)
    return CampaignSpec(
        campaign_id=f"prospective-campaign.{case_id}",
        objective="Adjudicate the outcome-visible hypothesis with separately sealed evidence.",
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(claim.claim_id for claim in experiment.claims),
        budget=designed.proposal.budget,
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id, system.authority_policy
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id=f"execution-right.{case_id}",
                action=AuthorityAction.REFERENCE_WORLD_EXECUTION,
                decision_maker_id=system.authority_policy.delegate_id,
                authority_policy_id=system.authority_policy.policy_id,
                delegated=True,
            ),
        ),
        nodes=(
            CampaignNode(
                node_id=f"experiment-node.{case_id}",
                kind=CampaignNodeKind.EXPERIMENT_SPEC,
                lane=CampaignLane.PROSPECTIVE,
                object_identity=experiment_identity,
                parent_node_ids=(),
                lifecycle_status=LifecycleStatus.ACTIVE,
            ),
        ),
        root_node_ids=(f"experiment-node.{case_id}",),
        active_node_ids=(f"experiment-node.{case_id}",),
        evidence_state=(experiment_identity,),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )


def _approval_components(
    source: _PreparedSource,
    nomination: ProspectiveNomination,
    obligations: NominationObligations,
    designed: DesignedExperiment,
    implementation_commit: str,
    authorization_id: str,
) -> tuple[
    CompleteApprovalService,
    ObjectIdentity,
    FrozenApprovalProposal,
    tuple[ObjectIdentity, ...],
]:
    proposal_store = _FrozenApprovalStore()
    authorization_store = _AuthorizationStore()
    freeze_service = CompleteApprovalService(
        clock=_ConformanceClock(),
        proposal_store=proposal_store,
        checker_registry=ApprovalCheckerRegistry(
            registry_id="prospective-freeze-checkers",
            registrations=(),
        ),
        attestation_store=_AttestationStore(),
        store=authorization_store,
    )
    identity = freeze_service.freeze_proposal(
        proposal=designed.proposal,
        nomination=nomination,
        design_obligations=obligations,
        system=source.system,
        requested_scope_id="reference-world",
        implementation_commit=implementation_commit,
    )
    frozen = proposal_store.load(identity.object_id)
    signers = tuple(
        _ConformanceAttestationSigner(gate_id)
        for gate_id in source.system.authority_policy.required_gate_ids
    )
    registrations = tuple(
        ApprovalCheckerRegistration(
            checker_registration_id=f"prospective-checker-registration.{gate_id}",
            gate_id=gate_id,
            gate_kind=(
                ApprovalGateKind.IMPLEMENTATION
                if "implementation" in gate_id
                else ApprovalGateKind.RESOURCE
            ),
            checker_id=f"prospective-checker.{gate_id}",
            implementation_sha256=_digest(f"prospective-checker:{gate_id}"),
            implementation_version="1.0.0",
            signature_algorithm=ApprovalSignatureAlgorithm.ED25519,
            signature_version=APPROVAL_SIGNATURE_VERSION,
            verification_key_hex=signer.verification_key_hex,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        for gate_id, signer in zip(
            source.system.authority_policy.required_gate_ids,
            signers,
            strict=True,
        )
    )
    attestations = tuple(
        issue_approval_gate_attestation(
            registration=registration,
            signer=signer,
            attestation_id=f"prospective-attestation.{authorization_id}.{gate_id}",
            authorization_id=authorization_id,
            subject=identity,
            result=AttestationResult.PASSED,
            checked_at_utc="2026-07-15T00:00:00Z",
            issued_at_utc="2026-07-15T00:00:00Z",
            information_cutoff=frozen.obligations.information_cutoff,
        )
        for gate_id, registration, signer in zip(
            source.system.authority_policy.required_gate_ids,
            registrations,
            signers,
            strict=True,
        )
    )
    service = CompleteApprovalService(
        clock=_ConformanceClock(),
        proposal_store=proposal_store,
        checker_registry=ApprovalCheckerRegistry(
            registry_id="prospective-approval-checkers",
            registrations=registrations,
        ),
        attestation_store=_AttestationStore(attestations),
        store=authorization_store,
    )
    identities = tuple(
        ObjectIdentity.from_record(attestation.attestation_id, attestation)
        for attestation in attestations
    )
    return service, identity, frozen, identities


def _complete_case(
    *,
    case_id: str,
    kind: ReferenceWorldKind,
    status: ScientificStatus,
    favored_alternative: bool,
    implementation_commit: str,
) -> _CompletedCase:
    source = _prepare_source(case_id, kind)
    context = _context(source, case_id, acquirable=True)
    batch = ProspectiveNominator().nominate(
        hypotheses=source.hypotheses,
        findings=(source.finding,),
        system=source.system,
        context=context,
    )
    nomination = batch.decision.nominations[0]
    obligations = batch.obligations[0]
    designed = (
        default_designer_registry()
        .resolve(obligations.design_kind)
        .design(
            nomination=nomination,
            obligations=obligations,
            system=source.system,
        )
    )
    authorization_id = f"authorization.{case_id}"
    service, frozen_identity, frozen, attestations = _approval_components(
        source,
        nomination,
        obligations,
        designed,
        implementation_commit,
        authorization_id,
    )
    authorization = service.authorize(
        policy=source.system.authority_policy,
        frozen_proposal=frozen_identity,
        attestations=attestations,
        authorization_id=authorization_id,
        approver_id=source.system.authority_policy.delegate_id,
    )
    experiment = service.replay(
        system=source.system,
        frozen_proposal=frozen_identity,
        authorization=ObjectIdentity.from_record(
            authorization.authorization_id,
            authorization,
        ),
    )
    campaign = _campaign(case_id, source.system, designed)
    template, runtime_registry = _protocol(case_id)
    run_plan = compile_run_plan(
        run_plan_id=f"run-plan.{case_id}",
        campaign=campaign,
        system=source.system,
        experiment=experiment,
        frozen_proposal=frozen,
        authorization=authorization,
        template=template,
        registry=runtime_registry,
        implementation_commit=implementation_commit,
        approval_service=service,
    )
    lower_run_plan(run_plan, runtime_registry)
    expected_kind = (
        OutcomeInterpretationKind.SUPPORTING
        if status is ScientificStatus.SUPPORTED
        else (
            OutcomeInterpretationKind.OPPOSING
            if status is ScientificStatus.NOT_SUPPORTED
            else OutcomeInterpretationKind.UNRESOLVED
        )
    )
    interpretation = next(
        item for item in nomination.outcome_interpretations if item.kind is expected_kind
    )
    hypotheses = source.hypotheses.hypotheses
    favored = (
        ()
        if status
        in {ScientificStatus.UNEVALUABLE, ScientificStatus.MIXED, ScientificStatus.PARTIAL}
        else (
            (hypotheses[1].hypothesis_id,)
            if favored_alternative
            else (hypotheses[0].hypothesis_id,)
        )
    )
    result = ProspectiveEvidenceResult(
        result_id=f"prospective-result.{case_id}",
        experiment=ObjectIdentity.from_record(experiment.experiment_id, experiment),
        run_plan=ObjectIdentity.from_record(run_plan.run_plan_id, run_plan),
        claim_ids=tuple(claim.claim_id for claim in experiment.claims),
        fresh_independent_unit_ids=(
            f"fresh-unit.{case_id}.1",
            f"fresh-unit.{case_id}.2",
        ),
        status=status,
        awarded_rung=EvidenceRung.LOCAL_LAW if status is ScientificStatus.SUPPORTED else None,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        interpreted_outcome_id=interpretation.interpretation_id,
        favored_hypothesis_ids=favored,
        terminal_reason_codes=(f"reference-{status.value.lower().replace('_', '-')}",),
        evidence_artifact_ids=(f"fresh-evidence-artifact.{case_id}",),
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    adjudication = ProspectiveAdjudicator().adjudicate(
        hypotheses=source.hypotheses,
        nomination=nomination,
        obligations=obligations,
        proposal=designed.proposal,
        experiment=experiment,
        result=result,
    )
    current = (
        source.snapshot_identity.object_fingerprint,
        source.wave_identity.object_fingerprint,
        source.finding.fingerprint(),
        source.hypotheses.fingerprint(),
    )
    outcome = ProspectiveConformanceOutcome(
        case_id=case_id,
        reference_world_id=kind.value,
        nomination_decision=batch.decision.kind,
        design_kind=obligations.design_kind,
        authorization_decision=authorization.decision,
        fresh_status=status,
        adjudication_disposition_ids=tuple(
            sorted({item.disposition.value for item in adjudication.hypothesis_adjudications})
        ),
        fresh_independent_unit_count=len(result.fresh_independent_unit_ids),
        run_compiled=True,
        source_fingerprints_preserved=current == source.fingerprints,
        outcome_fingerprint=adjudication.fingerprint(),
    )
    return _CompletedCase(
        outcome=outcome,
        source=source,
        batch=batch,
        designed=designed,
        run_plan=run_plan,
        result=result,
        adjudication=adjudication,
        campaign=campaign,
    )


def _stop_case() -> ProspectiveConformanceOutcome:
    case_id = "prospective-no-acquisition"
    source = _prepare_source(case_id, ReferenceWorldKind.OBSERVATIONAL_EQUIVALENCE)
    batch = ProspectiveNominator().nominate(
        hypotheses=source.hypotheses,
        findings=(source.finding,),
        system=source.system,
        context=_context(source, case_id, acquirable=False),
    )
    return ProspectiveConformanceOutcome(
        case_id=case_id,
        reference_world_id=ReferenceWorldKind.OBSERVATIONAL_EQUIVALENCE.value,
        nomination_decision=batch.decision.kind,
        design_kind=None,
        authorization_decision=None,
        fresh_status=None,
        adjudication_disposition_ids=(),
        fresh_independent_unit_count=0,
        run_compiled=False,
        source_fingerprints_preserved=source.fingerprints
        == (
            source.snapshot_identity.object_fingerprint,
            source.wave_identity.object_fingerprint,
            source.finding.fingerprint(),
            source.hypotheses.fingerprint(),
        ),
        outcome_fingerprint=batch.decision.fingerprint(),
    )


def _fresh_evidence_graph(case: _CompletedCase) -> CampaignGraph:
    nomination = case.batch.decision.nominations[0]
    post_snapshot = EvidenceSnapshot(
        snapshot_id=f"snapshot.adjudicated.{case.outcome.case_id}",
        campaign_id=case.campaign.campaign_id,
        run_id=case.run_plan.run_plan_id,
        world_id=case.source.system.world.world_id,
        parent_objects=tuple(
            sorted(
                (
                    ObjectIdentity.from_record(
                        case.adjudication.adjudication_id, case.adjudication
                    ),
                    ObjectIdentity.from_record(case.result.result_id, case.result),
                ),
                key=lambda item: item.object_id,
            )
        ),
        parent_visibility_ceilings=(
            VisibilityCeiling.OUTCOME_VISIBLE,
            VisibilityCeiling.PROSPECTIVE,
        ),
        artifacts=(
            ArtifactIdentity(
                artifact_id=f"artifact.adjudicated.{case.outcome.case_id}",
                role="prospective-adjudication",
                payload_schema=ProspectiveAdjudication.SCHEMA,
                sha256=case.adjudication.fingerprint(),
                media_type="application/json",
                size_bytes=len(case.adjudication.canonical_bytes()),
            ),
        ),
        claim_ids=case.result.claim_ids,
        information_cutoff=InformationCutoff(
            cutoff_id=f"post-outcome-cutoff.{case.outcome.case_id}",
            clock_id=case.source.system.relation.horizon.clock_id,
            phase=CausalPhase.POST_OUTCOME,
            coordinate=case.source.system.relation.horizon.duration,
        ),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        projection_ids=("prospective-adjudication",),
    )
    node_values = (
        (
            "source-snapshot",
            DiscoveryNodeKind.EVIDENCE_SNAPSHOT,
            DiscoveryLane.EXPLORATORY,
            case.source.snapshot_identity,
        ),
        (
            "source-wave",
            DiscoveryNodeKind.EXPLORATION_PLAN,
            DiscoveryLane.EXPLORATORY,
            case.source.wave_identity,
        ),
        (
            "source-finding",
            DiscoveryNodeKind.EXPLORATORY_FINDING,
            DiscoveryLane.EXPLORATORY,
            ObjectIdentity.from_record(case.source.finding.finding_id, case.source.finding),
        ),
        (
            "source-hypotheses",
            DiscoveryNodeKind.HYPOTHESIS_SET,
            DiscoveryLane.EXPLORATORY,
            ObjectIdentity.from_record(
                case.source.hypotheses.hypothesis_set_id, case.source.hypotheses
            ),
        ),
        (
            "prospective-nomination",
            DiscoveryNodeKind.PROSPECTIVE_NOMINATION,
            DiscoveryLane.EXPLORATORY,
            ObjectIdentity.from_record(nomination.nomination_id, nomination),
        ),
        (
            "experiment-proposal",
            DiscoveryNodeKind.EXPERIMENT_PROPOSAL,
            DiscoveryLane.PROSPECTIVE,
            ObjectIdentity.from_record(case.designed.proposal.proposal_id, case.designed.proposal),
        ),
        (
            "fresh-run",
            DiscoveryNodeKind.RUN,
            DiscoveryLane.PROSPECTIVE,
            ObjectIdentity.from_record(case.run_plan.run_plan_id, case.run_plan),
        ),
        (
            "fresh-result",
            DiscoveryNodeKind.EVIDENCE_RESULT,
            DiscoveryLane.PROSPECTIVE,
            ObjectIdentity.from_record(case.result.result_id, case.result),
        ),
        (
            "adjudicated-snapshot",
            DiscoveryNodeKind.EVIDENCE_SNAPSHOT,
            DiscoveryLane.EXPLORATORY,
            ObjectIdentity.from_record(post_snapshot.snapshot_id, post_snapshot),
        ),
    )
    nodes = tuple(
        sorted(
            (
                DiscoveryNode(
                    node_id=f"node.{case.outcome.case_id}.{node_id}",
                    kind=kind,
                    lane=lane,
                    object_identity=identity,
                    lifecycle_status=LifecycleStatus.TERMINAL,
                )
                for node_id, kind, lane, identity in node_values
            ),
            key=lambda item: item.node_id,
        )
    )
    edge_values = (
        ("01", DiscoveryEdgeKind.MOTIVATED_BY, "source-snapshot", "source-wave"),
        ("02", DiscoveryEdgeKind.MOTIVATED_BY, "source-wave", "source-finding"),
        ("03", DiscoveryEdgeKind.MOTIVATED_BY, "source-finding", "source-hypotheses"),
        (
            "04",
            DiscoveryEdgeKind.MOTIVATED_BY,
            "source-hypotheses",
            "prospective-nomination",
        ),
        (
            "05",
            DiscoveryEdgeKind.NOMINATES,
            "prospective-nomination",
            "experiment-proposal",
        ),
        ("06", DiscoveryEdgeKind.EXECUTED, "experiment-proposal", "fresh-run"),
        ("07", DiscoveryEdgeKind.EXECUTED, "fresh-run", "fresh-result"),
        (
            "08",
            DiscoveryEdgeKind.ADJUDICATED_BY,
            "fresh-result",
            "adjudicated-snapshot",
        ),
    )
    edges = tuple(
        DiscoveryEdge(
            edge_id=f"edge.{case.outcome.case_id}.{edge_id}",
            kind=kind,
            source_node_id=f"node.{case.outcome.case_id}.{source}",
            target_node_id=f"node.{case.outcome.case_id}.{target}",
            reason_codes=(),
        )
        for edge_id, kind, source, target in edge_values
    )
    return CampaignGraph(
        graph_id=f"campaign-graph.{case.outcome.case_id}",
        campaign=ObjectIdentity.from_record(case.campaign.campaign_id, case.campaign),
        nodes=nodes,
        edges=edges,
        root_node_ids=(f"node.{case.outcome.case_id}.source-snapshot",),
        active_node_ids=(f"node.{case.outcome.case_id}.adjudicated-snapshot",),
    )


def _authorization_branches(
    source: _PreparedSource,
    nomination: ProspectiveNomination,
    obligations: NominationObligations,
    designed: DesignedExperiment,
    implementation_commit: str,
) -> tuple[str, ...]:
    approved_id = "authorization.prospective-approved"
    service, identity, _frozen, attestations = _approval_components(
        source,
        nomination,
        obligations,
        designed,
        implementation_commit,
        approved_id,
    )
    approved = service.preview(
        policy=source.system.authority_policy,
        frozen_proposal=identity,
        attestations=attestations,
        authorization_id=approved_id,
        approver_id=source.system.authority_policy.delegate_id,
    )
    refused_id = "authorization.prospective-refused"
    refused_service, refused_identity, _refused_frozen, refused_attestations = _approval_components(
        source,
        nomination,
        obligations,
        designed,
        implementation_commit,
        refused_id,
    )
    refused = refused_service.preview(
        policy=source.system.authority_policy,
        frozen_proposal=refused_identity,
        attestations=refused_attestations[:1],
        authorization_id=refused_id,
        approver_id=source.system.authority_policy.delegate_id,
    )
    private_obligations = replace(
        obligations,
        source_access=SourceAccessClass.PRIVATE_CREDENTIAL,
    )
    private_binding = nomination_obligations_extension(private_obligations)
    private_nomination = replace(
        nomination,
        extensions=tuple(
            private_binding if value.namespace == private_binding.namespace else value
            for value in nomination.extensions
        ),
    )
    private_designed = replace(
        designed,
        proposal=replace(
            designed.proposal,
            nomination=ObjectIdentity.from_record(
                private_nomination.nomination_id,
                private_nomination,
            ),
        ),
    )
    private_id = "authorization.prospective-external-authority"
    private_service, private_identity, _private_frozen, private_attestations = _approval_components(
        source,
        private_nomination,
        private_obligations,
        private_designed,
        implementation_commit,
        private_id,
    )
    authority_required = private_service.preview(
        policy=source.system.authority_policy,
        frozen_proposal=private_identity,
        attestations=private_attestations,
        authorization_id=private_id,
        approver_id=source.system.authority_policy.delegate_id,
    )
    return tuple(
        sorted(
            {
                approved.decision.value,
                refused.decision.value,
                authority_required.decision.value,
            }
        )
    )


def run_prospective_reference_conformance(
    *, implementation_commit: str = "e" * 40
) -> ProspectiveReferenceConformanceReport:
    success = _complete_case(
        case_id="prospective-successful-adjudication",
        kind=ReferenceWorldKind.STABLE_LINEAR,
        status=ScientificStatus.SUPPORTED,
        favored_alternative=False,
        implementation_commit=implementation_commit,
    )
    defeated = _complete_case(
        case_id="prospective-prospective-replication-failure",
        kind=ReferenceWorldKind.RETROSPECTIVE_DEFEATED,
        status=ScientificStatus.NOT_SUPPORTED,
        favored_alternative=True,
        implementation_commit=implementation_commit,
    )
    unresolved = _complete_case(
        case_id="prospective-information-limited",
        kind=ReferenceWorldKind.OBSERVATIONAL_EQUIVALENCE,
        status=ScientificStatus.UNEVALUABLE,
        favored_alternative=False,
        implementation_commit=implementation_commit,
    )
    keys = prospective_capability_keys()
    registry = prospective_capability_registry({key: _digest(key) for key in keys})
    return ProspectiveReferenceConformanceReport(
        report_id="prospective-reference-conformance",
        registry_fingerprint=registry.fingerprint(),
        capability_keys=keys,
        designer_kind_ids=tuple(sorted(kind.value for kind in ProspectiveDesignKind)),
        authorization_branch_ids=_authorization_branches(
            success.source,
            success.batch.decision.nominations[0],
            success.batch.obligations[0],
            success.designed,
            implementation_commit,
        ),
        outcomes=tuple(
            sorted(
                (
                    defeated.outcome,
                    unresolved.outcome,
                    success.outcome,
                    _stop_case(),
                ),
                key=lambda item: item.case_id,
            )
        ),
        fresh_evidence_graph=_fresh_evidence_graph(success),
        physical_observation_to_controller_use_execution="NONE",
        status="PASS",
    )
