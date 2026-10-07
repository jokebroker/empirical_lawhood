# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest
from tests.runtime_platform.support import (
    IMPLEMENTATION_COMMIT,
    ExplorationFixture,
    ProtocolFixture,
    budget,
    capability_config,
    digest,
    permissions,
)
from tests.approval_support import deterministic_approval_signer

from empirical_lawhood.adapters.reference_worlds import reference_worlds
from empirical_lawhood.kernel.authority import AuthorityAction, SourceAccessClass
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
from empirical_lawhood.kernel.provenance import EvidenceSnapshot, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, QuantityBound
from empirical_lawhood.kernel.status import LifecycleStatus, ReadinessStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.authority import (
    ApprovalRequest,
    AuthorizationDecision,
    AuthorizationRecord,
    authorize_experiment,
)
from empirical_lawhood.planning.approval import (
    ApprovalCheckerRegistration,
    ApprovalCheckerRegistry,
    ApprovalGateAttestation,
    ApprovalGateKind,
    ApprovalObligations,
    ApprovalSignatureAlgorithm,
    AttestationResult,
    CompleteApprovalService,
    DurableAuthorizationRecord,
    FrozenApprovalProposal,
    approval_obligations_extension,
    issue_approval_gate_attestation,
)
from empirical_lawhood.planning.campaigns import (
    CampaignLane,
    CampaignNode,
    CampaignNodeKind,
    CampaignSpec,
    DecisionRight,
)
from empirical_lawhood.planning.design import ExperimentProposal
from empirical_lawhood.planning.exploration import (
    AnalysisProposal,
    AnalysisSpec,
    ProposalDisposition,
    ProposalSelection,
    SearchAxis,
)
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.compiler import (
    compile_exploration_plan,
    compile_run_plan,
    lower_run_plan,
)
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
    SnapshotVerification,
)


def _required_obligations(system: SystemSpec) -> ScientificObligations:
    view_ids = tuple(view.view_id for view in system.numerical_views)
    return ScientificObligations(
        obligations_id='synthetic-reference-obligations',
        support=SupportSpec(
            support_id='synthetic-reference-support',
            relation_id=system.relation.relation_id,
            independent_unit_id=system.independent_unit.unit_id,
            physical_unit_count=0,
            nested_numerical_view_count=len(view_ids),
            information_cutoff_id='synthetic-reference-pre-action-cutoff',
            chart_ids=("reference-chart",),
            denominator_cell_ids=("reference-cell",),
            action_bounds=(
                QuantityBound(
                    bound_id="reference-action-bound",
                    quantity_id=system.relation.action_quantity_ids[0],
                    native_unit="1",
                    lower=Decimal("-1"),
                    upper=Decimal("1"),
                ),
            ),
            status=ObligationStatus.REQUIRED,
        ),
        validity=ValiditySpec(
            validity_id='synthetic-reference-validity',
            validity_domain_ids=("reference-cell",),
            assumption_ids=("declared-reference-dynamics",),
            exclusion_reason_codes=(),
            status=ObligationStatus.REQUIRED,
        ),
        uncertainty=UncertaintySpec(
            uncertainty_id='synthetic-reference-uncertainty',
            method_key="independent-preparation-interval",
            independent_unit_id=system.independent_unit.unit_id,
            confidence_level=Decimal("0.95"),
            interval_quantity_ids=system.relation.receiver_quantity_ids,
            limitation_codes=(),
            status=ObligationStatus.REQUIRED,
        ),
        falsifiers=(
            FalsifierSpec(
                falsifier_id='synthetic-reference-wrong-action',
                kind=FalsifierKind.WRONG_ACTION,
                capability_key="reference.wrong-action",
                description="Displace the declared response action.",
                decisive_rule="Wrong-action displacement must remain below tolerance.",
                status=ObligationStatus.REQUIRED,
            ),
        ),
        closure=ClosureSpec(
            closure_id='synthetic-reference-closure',
            recurrence_cell_ids=("reference-cell",),
            exchange_factor_ids=system.relation.denominator_quantity_ids,
            retained_history_ids=system.relation.history_quantity_ids,
            status=ObligationStatus.REQUIRED,
        ),
        structural_convergence=StructuralConvergenceSpec(
            convergence_id='synthetic-reference-convergence',
            required_structure_ids=("response-rank",),
            numerical_view_ids=view_ids,
            tolerances=(),
            status=ObligationStatus.REQUIRED,
        ),
        computability=ComputabilityEvidence(
            computability_id='synthetic-reference-computability',
            envelope_id=system.computability_envelopes[0].envelope_id,
            numerical_view_ids=view_ids,
            readiness=ReadinessStatus.READY,
            unresolved_reason_codes=(),
            evidence_link_ids=("reference-envelope-preflight",),
        ),
    )


def _candidate_experiment(system: SystemSpec) -> ExperimentSpec:
    view_ids = tuple(view.view_id for view in system.numerical_views)
    claim = ClaimSpec(
        claim_id="synthetic-reference-response-claim",
        world_id=system.world.world_id,
        relation_id=system.relation.relation_id,
        proposition="The prepared reference system supports its declared local response.",
        estimand="Finite-horizon receiver displacement under the native action chart.",
        physical_independent_unit_id=system.independent_unit.unit_id,
        requested_rung=EvidenceRung.LOCAL_LAW,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        promotion_rule="The frozen response and decisive falsifier gates all pass.",
        assumption_ids=("fresh-reference-preparations",),
        numerical_view_ids=view_ids,
    )
    return ExperimentSpec(
        experiment_id='synthetic-reference-experiment',
        system_id=system.system_id,
        world_id=system.world.world_id,
        relation=system.relation,
        independent_unit_id=system.independent_unit.unit_id,
        claims=(claim,),
        assignment=AssignmentSpec(
            assignment_id='synthetic-reference-assignment',
            kind=AssignmentKind.SIMULATOR_INTERVENTION,
            independent_unit_id=system.independent_unit.unit_id,
            action_quantity_ids=system.relation.action_quantity_ids,
            mechanism="Deterministic reference-world intervention on fresh preparations.",
            support_restriction_ids=(),
        ),
        measurement_quantity_ids=tuple(sorted((*system.relation.receiver_quantity_ids, "sink"))),
        controls=(
            ControlSpec(
                control_id='synthetic-reference-negative-control',
                kind=ControlKind.WRONG_ACTION,
                capability_key="reference.wrong-action",
                target_quantity_ids=system.relation.receiver_quantity_ids,
                decisive_rule="Wrong-action displacement cannot retain response advantage.",
            ),
        ),
        precision_goals=(
            PrecisionGoal(
                goal_id='synthetic-reference-precision',
                metric_id="receiver-displacement",
                target_width=Decimal("0.1"),
                native_unit="1",
                maximum_independent_units=8,
                stopping_rule="Stop at eight preparations or the target interval width.",
            ),
        ),
        information_cutoffs=(
            InformationCutoff(
                cutoff_id='synthetic-reference-pre-action-cutoff',
                clock_id=system.clocks[0].clock_id,
                phase=CausalPhase.PRE_ACTION,
                coordinate=Decimal("0"),
            ),
        ),
        reveal_barrier=RevealBarrierSpec(
            barrier_id='synthetic-reference-reveal',
            development_unit_ids=("reference-development-a", "reference-development-b"),
            evaluation_cohort_id="reference-evaluation-cohort",
            evaluation_manifest_sha256=digest('synthetic reference evaluation cohort'),
            sealed_outcome_artifact_ids=("reference-sealed-outcomes",),
            evaluation_outcome_access=OutcomeAccess.EVALUATION_SEALED,
            fresh_evidence=True,
        ),
        obligations=_required_obligations(system),
        design_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evaluation_visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        authority_policy_id=system.authority_policy.policy_id,
        readiness=ReadinessStatus.AUTHORITY_REQUIRED,
    )


def _protocol_template() -> ProtocolTemplate:
    development_budget = budget(source_scan_bytes=1_000)
    step_budget = budget(source_scan_bytes=1_500)

    def output(output_id: str, schema: str) -> tuple[OutputTemplate, ...]:
        return (
            OutputTemplate(
                output_id=output_id,
                payload_schema=schema,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                filename_suffix=".json",
            ),
        )

    steps = (
        ProtocolStepTemplate(
            step_id="develop-a",
            stage=ScientificStage.DEVELOP,
            capability_key="reference.develop-a",
            capability_version="1.0.0",
            config=capability_config(
                "reference.develop-a",
                'empirical-lawhood/testing/fixtures/reference-world-pipeline/development-first-config',
            ),
            dependency_step_ids=("prepare",),
            outputs=output("model-a", 'empirical-lawhood/testing/fixtures/reference-world-pipeline/development-output'),
            required_permissions=permissions(
                CapabilityPermission.READ_DEVELOPMENT,
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            requested_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            resource_budget=development_budget,
            resource_lock_ids=("development-a",),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("fit-development-a",),
        ),
        ProtocolStepTemplate(
            step_id="develop-b",
            stage=ScientificStage.FALSIFY,
            capability_key="reference.develop-b",
            capability_version="1.0.0",
            config=capability_config(
                "reference.develop-b",
                'empirical-lawhood/testing/fixtures/reference-world-pipeline/development-second-config',
            ),
            dependency_step_ids=("prepare",),
            outputs=output("model-b", 'empirical-lawhood/testing/fixtures/reference-world-pipeline/development-output'),
            required_permissions=permissions(
                CapabilityPermission.READ_DEVELOPMENT,
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            requested_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            resource_budget=development_budget,
            resource_lock_ids=("development-b",),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("run-wrong-action",),
        ),
        ProtocolStepTemplate(
            step_id="evaluate",
            stage=ScientificStage.EVALUATE,
            capability_key="reference.evaluate",
            capability_version="1.0.0",
            config=capability_config(
                "reference.evaluate",
                'empirical-lawhood/testing/fixtures/reference-world-pipeline/evaluation-config',
            ),
            dependency_step_ids=("freeze",),
            outputs=output("evaluation", 'empirical-lawhood/testing/fixtures/reference-world-pipeline/evaluation-output'),
            required_permissions=permissions(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            requested_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            resource_budget=budget(output_bytes=3_000, source_scan_bytes=2_500),
            resource_lock_ids=("evaluator",),
            barrier=BarrierKind.REVEAL,
            maximum_attempts=2,
            obligation_ids=("held-out-evaluation",),
        ),
        ProtocolStepTemplate(
            step_id="freeze",
            stage=ScientificStage.FREEZE,
            capability_key="reference.freeze",
            capability_version="1.0.0",
            config=capability_config(
                "reference.freeze",
                'empirical-lawhood/testing/fixtures/reference-world-pipeline/freeze-config',
            ),
            dependency_step_ids=("develop-a", "develop-b"),
            outputs=output("frozen", 'empirical-lawhood/testing/fixtures/reference-world-pipeline/frozen-input'),
            required_permissions=permissions(
                CapabilityPermission.READ_DEVELOPMENT,
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            requested_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            resource_budget=step_budget,
            resource_lock_ids=("freeze",),
            barrier=BarrierKind.FREEZE,
            maximum_attempts=2,
            obligation_ids=("freeze-complete-protocol",),
        ),
        ProtocolStepTemplate(
            step_id="prepare",
            stage=ScientificStage.PREPARE,
            capability_key="reference.prepare",
            capability_version="1.0.0",
            config=capability_config(
                "reference.prepare",
                'empirical-lawhood/testing/fixtures/reference-world-pipeline/preparation-config',
            ),
            dependency_step_ids=(),
            outputs=output("prepared", 'empirical-lawhood/testing/fixtures/reference-world-pipeline/prepared-input'),
            required_permissions=permissions(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            requested_outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            resource_budget=budget(source_scan_bytes=500),
            resource_lock_ids=("reference-simulator",),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("prepare-independent-units",),
        ),
        ProtocolStepTemplate(
            step_id="report",
            stage=ScientificStage.REPORT,
            capability_key="reference.report",
            capability_version="1.0.0",
            config=capability_config(
                "reference.report",
                'empirical-lawhood/testing/fixtures/reference-world-pipeline/report-config',
            ),
            dependency_step_ids=("evaluate",),
            outputs=output("report", 'empirical-lawhood/testing/fixtures/reference-world-pipeline/report-output'),
            required_permissions=permissions(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
            resource_budget=budget(source_scan_bytes=2_500),
            resource_lock_ids=("report",),
            barrier=BarrierKind.NONE,
            maximum_attempts=2,
            obligation_ids=("report-without-promotion",),
        ),
    )
    return ProtocolTemplate(
        template_id='synthetic-branched-reference-protocol',
        template_version="1.0.0",
        steps=steps,
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _protocol_registry(template: ProtocolTemplate) -> CapabilityRegistry:
    kinds = {
        "reference.develop-a": CapabilityKind.ANALYSIS,
        "reference.develop-b": CapabilityKind.FALSIFIER,
        "reference.evaluate": CapabilityKind.EVALUATOR,
        "reference.freeze": CapabilityKind.TRANSFORM,
        "reference.prepare": CapabilityKind.SIMULATOR,
        "reference.report": CapabilityKind.REPORTER,
    }
    input_schemas = {
        "reference.develop-a": ('empirical-lawhood/testing/fixtures/reference-world-pipeline/prepared-input',),
        "reference.develop-b": ('empirical-lawhood/testing/fixtures/reference-world-pipeline/prepared-input',),
        "reference.evaluate": ('empirical-lawhood/testing/fixtures/reference-world-pipeline/frozen-input',),
        "reference.freeze": ('empirical-lawhood/testing/fixtures/reference-world-pipeline/development-output',),
        "reference.prepare": (),
        "reference.report": ('empirical-lawhood/testing/fixtures/reference-world-pipeline/evaluation-output',),
    }
    manifests = tuple(
        CapabilityManifest(
            capability_key=step.capability_key,
            capability_version=step.capability_version,
            kind=kinds[step.capability_key],
            config_schema=step.config.config_schema,
            config_schema_sha256=step.config.config_schema_sha256,
            input_schema_ids=input_schemas[step.capability_key],
            output_schema_ids=tuple(sorted({output.payload_schema for output in step.outputs})),
            permissions=step.required_permissions,
            maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            maximum_outcome_access=step.requested_outcome_access,
            resource_ceiling=budget(
                wall_time_seconds=10,
                output_bytes=10_000,
                source_scan_bytes=2_500,
            ),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=('synthetic-reference-capability-contract',),
            implementation_sha256=digest(step.capability_key),
        )
        for step in template.steps
    )
    return CapabilityRegistry(
        registry_id='synthetic-reference-registry',
        capabilities=manifests,
    )


class _ReferenceDecisionClock:
    clock_id = "reference-fixture-trusted-clock"

    def now_utc(self) -> str:
        return "2026-07-15T00:00:00Z"


class _ReferenceProposalStore:
    def __init__(self, frozen: FrozenApprovalProposal) -> None:
        self.frozen = frozen

    def persist(self, frozen: FrozenApprovalProposal) -> ObjectIdentity:
        if frozen != self.frozen:
            raise ValueError("immutable reference proposal conflict")
        return ObjectIdentity.from_record(frozen.frozen_proposal_id, frozen)

    def load(self, frozen_proposal_id: str) -> FrozenApprovalProposal:
        if frozen_proposal_id != self.frozen.frozen_proposal_id:
            raise KeyError(frozen_proposal_id)
        return self.frozen


class _ReferenceAuthorizationStore:
    def __init__(self) -> None:
        self.record: DurableAuthorizationRecord | None = None

    def persist(self, record: DurableAuthorizationRecord) -> ObjectIdentity:
        if self.record is not None and self.record != record:
            raise ValueError("immutable reference authorization conflict")
        self.record = record
        return ObjectIdentity.from_record(record.authorization_id, record)

    def load(self, authorization_id: str) -> DurableAuthorizationRecord:
        if self.record is None or authorization_id != self.record.authorization_id:
            raise KeyError(authorization_id)
        return self.record


class _ReferenceAttestationStore:
    def __init__(self, values: tuple[ApprovalGateAttestation, ...]) -> None:
        self.values = {value.attestation_id: value for value in values}

    def load(self, attestation_id: str) -> ApprovalGateAttestation:
        return self.values[attestation_id]


def build_protocol_fixture() -> ProtocolFixture:
    system = reference_worlds()[0].system
    candidate = _candidate_experiment(system)
    proposal_budget = budget(
        wall_time_seconds=10,
        output_bytes=20_000,
        source_scan_bytes=10_000,
    )
    proposal = ExperimentProposal(
        proposal_id='synthetic-reference-proposal',
        nomination=ObjectIdentity(
            object_id='synthetic-reference-nomination',
            object_schema='empirical-lawhood/testing/fixtures/nomination',
            object_version="1.0.0",
            object_fingerprint=digest('synthetic reference nomination'),
        ),
        candidate_experiment=candidate,
        alternative_design_ids=("reference-observational-alternative",),
        expected_discrimination="Separates the declared response from wrong-action structure.",
        risk_codes=("reference-only",),
        budget=proposal_budget,
        decision_cutoff=candidate.information_cutoffs[0],
        proposed_by="protocol-planner",
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        parent_visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )
    approval_request = ApprovalRequest(
        authorization_id='synthetic-reference-authorization',
        action=AuthorityAction.REFERENCE_WORLD_EXECUTION,
        world_kind=system.world.kind,
        source_access=SourceAccessClass.NONE,
        requested_scope_id="reference-world",
        requested_budget=proposal_budget,
        passed_gate_ids=system.authority_policy.required_gate_ids,
        proposer_id=proposal.proposed_by,
        approver_id=system.authority_policy.delegate_id,
        decided_at_utc="2026-07-14T00:00:00Z",
        implementation_commit=IMPLEMENTATION_COMMIT,
    )
    # Reference authorization record used for replay comparison.
    comparison_authorization = AuthorizationRecord(
        authorization_id=approval_request.authorization_id,
        policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        proposal=ObjectIdentity.from_record(proposal.proposal_id, proposal),
        experiment=ObjectIdentity.from_record(candidate.experiment_id, candidate),
        decision=AuthorizationDecision.APPROVED_NONACTUATING,
        action=approval_request.action,
        world_kind=approval_request.world_kind,
        source_access=approval_request.source_access,
        requested_scope_id=approval_request.requested_scope_id,
        requested_budget=approval_request.requested_budget,
        passed_gate_ids=approval_request.passed_gate_ids,
        failed_gate_ids=(),
        reason_codes=("frozen-reference-comparison-fixture",),
        proposer_id=approval_request.proposer_id,
        approver_id=approval_request.approver_id,
        decided_at_utc=approval_request.decided_at_utc,
        implementation_commit=approval_request.implementation_commit,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        plan_mutated=False,
        grants_claim_promotion=False,
    )
    comparison_experiment = authorize_experiment(
        system.authority_policy,
        proposal,
        comparison_authorization,
    )

    obligations = ApprovalObligations(
        obligations_id=f"approval-obligations.{proposal.proposal_id}",
        system=ObjectIdentity.from_record(system.system_id, system),
        world=ObjectIdentity.from_record(system.world.world_id, system.world),
        policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        action=approval_request.action,
        world_kind=system.world.kind,
        source_access=approval_request.source_access,
        requested_scope_id=approval_request.requested_scope_id,
        requested_budget=proposal_budget,
        information_cutoff=proposal.decision_cutoff,
        implementation_commit=IMPLEMENTATION_COMMIT,
        required_gate_ids=system.authority_policy.required_gate_ids,
        proposer_id=proposal.proposed_by,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    proposal = replace(
        proposal,
        extensions=(approval_obligations_extension(obligations),),
    )
    frozen_proposal = FrozenApprovalProposal(
        frozen_proposal_id=f"frozen-approval.{proposal.proposal_id}",
        proposal=proposal,
        obligations=obligations,
    )
    frozen_identity = ObjectIdentity.from_record(
        frozen_proposal.frozen_proposal_id,
        frozen_proposal,
    )
    proposal_store = _ReferenceProposalStore(frozen_proposal)
    authorization_store = _ReferenceAuthorizationStore()
    gate_kinds = {
        "clean-implementation": ApprovalGateKind.IMPLEMENTATION,
        "resource-envelope": ApprovalGateKind.RESOURCE,
    }
    signers = tuple(
        deterministic_approval_signer(f"synthetic-reference:{gate_id}")
        for gate_id in system.authority_policy.required_gate_ids
    )
    registrations = tuple(
        ApprovalCheckerRegistration(
            checker_registration_id=f"checker-registration.synthetic-reference.{gate_id}",
            gate_id=gate_id,
            gate_kind=gate_kinds[gate_id],
            checker_id=f"checker.synthetic-reference.{gate_id}",
            implementation_sha256=digest(f"checker:{gate_id}"),
            implementation_version="1.0.0",
            signature_algorithm=ApprovalSignatureAlgorithm.ED25519,
            signature_version="1.0.0",
            verification_key_hex=signer.verification_key_hex,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        for gate_id, signer in zip(
            system.authority_policy.required_gate_ids,
            signers,
            strict=True,
        )
    )
    checker_registry = ApprovalCheckerRegistry(
        registry_id='synthetic-reference-approval-checkers',
        registrations=registrations,
    )
    attestations = tuple(
        issue_approval_gate_attestation(
            registration=registration,
            signer=signer,
            attestation_id=f"attestation.synthetic-reference.{gate_id}",
            authorization_id=approval_request.authorization_id,
            subject=frozen_identity,
            result=AttestationResult.PASSED,
            checked_at_utc="2026-07-15T00:00:00Z",
            issued_at_utc="2026-07-15T00:00:00Z",
            information_cutoff=obligations.information_cutoff,
        )
        for gate_id, registration, signer in zip(
            system.authority_policy.required_gate_ids,
            registrations,
            signers,
            strict=True,
        )
    )
    approval_service = CompleteApprovalService(
        clock=_ReferenceDecisionClock(),
        proposal_store=proposal_store,
        checker_registry=checker_registry,
        attestation_store=_ReferenceAttestationStore(attestations),
        store=authorization_store,
    )
    authorization = approval_service.authorize(
        policy=system.authority_policy,
        frozen_proposal=frozen_identity,
        attestations=tuple(
            ObjectIdentity.from_record(attestation.attestation_id, attestation)
            for attestation in attestations
        ),
        authorization_id=approval_request.authorization_id,
        approver_id=system.authority_policy.delegate_id,
    )
    experiment = approval_service.replay(
        system=system,
        frozen_proposal=frozen_identity,
        authorization=ObjectIdentity.from_record(
            authorization.authorization_id,
            authorization,
        ),
    )
    assert experiment == comparison_experiment
    experiment_identity = ObjectIdentity.from_record(experiment.experiment_id, experiment)
    campaign = CampaignSpec(
        campaign_id='synthetic-reference-campaign',
        objective="Execute a frozen branched reference-world response protocol.",
        system_ids=(system.system_id,),
        world_ids=(system.world.world_id,),
        target_claim_ids=tuple(claim.claim_id for claim in experiment.claims),
        budget=proposal_budget,
        authority_policy=ObjectIdentity.from_record(
            system.authority_policy.policy_id,
            system.authority_policy,
        ),
        decision_rights=(
            DecisionRight(
                decision_right_id='synthetic-reference-execution-right',
                action=AuthorityAction.REFERENCE_WORLD_EXECUTION,
                decision_maker_id=system.authority_policy.delegate_id,
                authority_policy_id=system.authority_policy.policy_id,
                delegated=True,
            ),
        ),
        nodes=(
            CampaignNode(
                node_id='synthetic-reference-experiment-node',
                kind=CampaignNodeKind.EXPERIMENT_SPEC,
                lane=CampaignLane.PROSPECTIVE,
                object_identity=experiment_identity,
                parent_node_ids=(),
                lifecycle_status=LifecycleStatus.ACTIVE,
            ),
        ),
        root_node_ids=('synthetic-reference-experiment-node',),
        active_node_ids=('synthetic-reference-experiment-node',),
        evidence_state=(experiment_identity,),
        lifecycle_status=LifecycleStatus.ACTIVE,
    )
    template = _protocol_template()
    registry = _protocol_registry(template)
    run_plan = compile_run_plan(
        run_plan_id='synthetic-reference-run-plan',
        campaign=campaign,
        system=system,
        experiment=experiment,
        frozen_proposal=frozen_proposal,
        authorization=authorization,
        template=template,
        registry=registry,
        implementation_commit=IMPLEMENTATION_COMMIT,
        approval_service=approval_service,
    )
    execution_plan = lower_run_plan(run_plan, registry)
    return ProtocolFixture(
        system=system,
        proposal=proposal,
        approval_request=approval_request,
        comparison_authorization=comparison_authorization,
        frozen_proposal=frozen_proposal,
        authorization=authorization,
        approval_service=approval_service,
        experiment=experiment,
        campaign=campaign,
        template=template,
        registry=registry,
        run_plan=run_plan,
        execution_plan=execution_plan,
    )


@pytest.fixture
def protocol_fixture() -> ProtocolFixture:
    return build_protocol_fixture()


@pytest.fixture
def exploration_fixture(protocol_fixture: ProtocolFixture) -> ExplorationFixture:
    system = protocol_fixture.system
    artifact = ArtifactIdentity(
        artifact_id='synthetic-exploration-source',
        role="canonical-evidence",
        payload_schema='empirical-lawhood/testing/fixtures/reference-world-pipeline/exploration-source',
        sha256=digest('synthetic source data'),
        media_type="text/plain",
        size_bytes=len(b'synthetic source data'),
    )
    cutoff = InformationCutoff(
        cutoff_id='synthetic-exploration-cutoff',
        clock_id=system.clocks[0].clock_id,
        phase=CausalPhase.POST_OUTCOME,
        coordinate=Decimal("10"),
    )
    snapshot = EvidenceSnapshot(
        snapshot_id='synthetic-exploration-snapshot',
        campaign_id=protocol_fixture.campaign.campaign_id,
        run_id='synthetic-exploration-parent-run',
        world_id=system.world.world_id,
        parent_objects=(ObjectIdentity.from_record(system.system_id, system),),
        parent_visibility_ceilings=(VisibilityCeiling.OUTCOME_VISIBLE,),
        artifacts=(artifact,),
        claim_ids=tuple(claim.claim_id for claim in protocol_fixture.experiment.claims),
        information_cutoff=cutoff,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        projection_ids=("response-summary",),
    )

    def proposal(index: int) -> AnalysisProposal:
        analysis_id = f"synthetic-exploration-analysis-{index}"
        analysis = AnalysisSpec(
            analysis_id=analysis_id,
            snapshot_id=snapshot.snapshot_id,
            snapshot_fingerprint=snapshot.fingerprint(),
            system_id=system.system_id,
            relation=system.relation,
            question=f"Does registered exploratory family member {index} retain structure?",
            estimand="Outcome-visible receiver displacement within the frozen projection.",
            independent_unit_id=system.independent_unit.unit_id,
            projection_ids=snapshot.projection_ids,
            receiver_quantity_ids=system.relation.receiver_quantity_ids,
            denominator_quantity_ids=system.relation.denominator_quantity_ids,
            action_quantity_ids=system.relation.action_quantity_ids,
            horizon_ids=(system.relation.horizon.horizon_id,),
            grouping_clock_ids=(system.clocks[0].clock_id,),
            registered_pipeline_key="exploration.pipeline",
            registered_pipeline_version="1.0.0",
            null_capability_keys=("exploration.null",),
            falsifier_capability_keys=("exploration.skeptic",),
            search_axes=(
                SearchAxis(
                    axis_id="denominator-cell",
                    candidate_ids=(f"cell-{index}",),
                ),
            ),
            uncertainty_method_key="physical-unit-resampling",
            budget=budget(source_scan_bytes=10_000),
            stopping_rule="Execute the complete singleton registered family.",
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            parent_visibility_ceiling=snapshot.visibility_ceiling,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        )
        return AnalysisProposal(
            proposal_id=f"synthetic-exploration-analysis-proposal-{index}",
            analysis=analysis,
            anomaly_signal_ids=(f"synthetic-exploration-signal-{index}",),
            intent="Test a registered outcome-visible diagnostic without promotion.",
            prerequisite_ids=(),
            selection_rationale="Retain the complete family and explicit disposition.",
        )

    proposals = (proposal(1), proposal(2))
    verification = SnapshotVerification(
        verification_id='synthetic-exploration-snapshot-verification',
        snapshot=ObjectIdentity.from_record(snapshot.snapshot_id, snapshot),
        artifact_ids=(artifact.artifact_id,),
        verification_check_ids=("content-sha256", "read-only-projection"),
        verifier_key="external-artifact-verifier",
        verifier_version="1.0.0",
    )
    plan = compile_exploration_plan(
        plan_id='synthetic-exploration-plan',
        snapshot=snapshot,
        snapshot_verification=verification,
        proposals=proposals,
        selections=(
            ProposalSelection(
                proposal_id=proposals[0].proposal_id,
                disposition=ProposalDisposition.SELECTED,
                reason_codes=("representative-family-member",),
                planner_key="bounded-portfolio",
            ),
            ProposalSelection(
                proposal_id=proposals[1].proposal_id,
                disposition=ProposalDisposition.REJECTED,
                reason_codes=("redundant-under-budget",),
                planner_key="bounded-portfolio",
            ),
        ),
        budget=budget(
            wall_time_seconds=5,
            output_bytes=10_000,
            source_scan_bytes=20_000,
        ),
    )
    analysis_permissions = permissions(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.READ_OUTCOME_VISIBLE,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    )
    synthesis_permissions = permissions(
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.READ_OUTCOME_VISIBLE,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    )
    registry = CapabilityRegistry(
        registry_id='synthetic-exploration-registry',
        capabilities=(
            CapabilityManifest(
                capability_key="exploration.pipeline",
                capability_version="1.0.0",
                kind=CapabilityKind.ANALYSIS,
                config_schema=AnalysisSpec.SCHEMA,
                config_schema_sha256=digest(AnalysisSpec.SCHEMA),
                input_schema_ids=(artifact.payload_schema,),
                output_schema_ids=('empirical-lawhood/planning/exploratory-finding',),
                permissions=analysis_permissions,
                maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                resource_ceiling=budget(
                    wall_time_seconds=5,
                    output_bytes=10_000,
                    source_scan_bytes=20_000,
                ),
                deterministic=True,
                seed_required=False,
                language_id="python",
                runtime_id="cpython-3.11",
                requires_clean_commit=True,
                requires_active_mount=True,
                requires_network=False,
                conformance_check_ids=('synthetic-exploration-read-only',),
                implementation_sha256=digest("exploration.pipeline"),
            ),
            CapabilityManifest(
                capability_key="exploration.synthesis",
                capability_version="1.0.0",
                kind=CapabilityKind.ANALYSIS,
                config_schema=plan.SCHEMA,
                config_schema_sha256=digest(plan.SCHEMA),
                input_schema_ids=('empirical-lawhood/planning/exploratory-finding',),
                output_schema_ids=('empirical-lawhood/planning/hypothesis-set',),
                permissions=synthesis_permissions,
                maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                resource_ceiling=budget(
                    wall_time_seconds=5,
                    output_bytes=10_000,
                    source_scan_bytes=20_000,
                ),
                deterministic=True,
                seed_required=False,
                language_id="python",
                runtime_id="cpython-3.11",
                requires_clean_commit=True,
                requires_active_mount=True,
                requires_network=False,
                conformance_check_ids=('synthetic-exploration-synthesis-nonpromotable',),
                implementation_sha256=digest("exploration.synthesis"),
            ),
        ),
    )
    return ExplorationFixture(
        snapshot=snapshot,
        verification=verification,
        plan=plan,
        registry=registry,
    )
