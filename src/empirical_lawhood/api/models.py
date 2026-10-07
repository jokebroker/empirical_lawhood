"""Public authoring packages and bounded result payloads."""

from __future__ import annotations

from empirical_lawhood.kernel.retrospective_prediction_experiments import RetrospectiveExperimentSpec
from empirical_lawhood.planning.approval import FrozenRetrospectiveApproval
from empirical_lawhood.runtime.study_issue import RetrospectiveIssuedBase, RetrospectiveIssuedStudy, RetrospectivePublicationReceipt
from empirical_lawhood.runtime.candidate_compiler import RetrospectiveExtensionReport, RetrospectiveStandardReport

import re
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import ClassVar, TypeAlias

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.models import ViewModelSetSpec, ModelSetSpec
from empirical_lawhood.kernel.provenance import EvidenceSnapshot, ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.approval import DurableAuthorizationRecord, FrozenApprovalProposal, FrozenIssuedStudyApprovalProposal, validate_durable_authorization_structure
from empirical_lawhood.planning.authority import ApprovalRequest
from empirical_lawhood.planning.campaigns import CampaignSpec
from empirical_lawhood.planning.exploration import (
    ExplorationPlan,
    ExploratoryFinding,
    HypothesisSet,
    ProposalDisposition,
)
from empirical_lawhood.planning.linked_campaign import LinkedCampaignPackageRole
from empirical_lawhood.planning.prospective import ProspectiveDesignContext
from empirical_lawhood.runtime.capabilities import CapabilityPermission, CapabilityRegistry
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.exploration import (
    ExplorationExecutionPackage as ExplorationExecutionPackage,
)
from empirical_lawhood.runtime.datasets import DatasetCatalogRecordKind
from empirical_lawhood.runtime.extension_bundles import CapabilityBundleBinding
from empirical_lawhood.runtime.executable_bindings import CapabilityExecutionAvailability
from empirical_lawhood.runtime.plans import ProtocolTemplate, SnapshotVerification
from empirical_lawhood.runtime.candidate_compiler import DraftStudyCandidate, StudyCompilationReport, ExecutableStudyCompilationReport, StudyCandidate, ExecutableStudyCandidate
from empirical_lawhood.runtime.execution_envelope import ExecutionEnvelopeSpec, ExecutionResourceEnvelopeSpec, JitGraphSignatureManifest, PredevelopmentJitSignatureCensus, validate_execution_jit_evidence
from empirical_lawhood.runtime.campaign_elapsed_budget import CampaignElapsedBudgetSpec, CampaignElapsedReservationPlan
from empirical_lawhood.runtime.study_issue import IssuedDraftManifest, StudyPublicationReceipt, ExtensionPublicationReceipt, IssuedStudyManifest, IssuedExecutableStudyManifest
from empirical_lawhood.runtime.multi_world_study_issue import MultiWorldStudyIssueManifest, MultiWorldStudyPublicationReceipt


from empirical_lawhood.runtime.study_bundle_compiler import StudyBundleCompilationReport
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority


ProgrammePublicationReceipt: TypeAlias = (
    StudyPublicationReceipt | ExtensionPublicationReceipt
)
FrozenModelSet: TypeAlias = ViewModelSetSpec | ModelSetSpec


@dataclass(frozen=True, slots=True)
class CampaignPackage(CanonicalRecord):
    """Durable-authorized follow-up root containing every compilation input."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/campaign-package'

    package_id: str
    run_plan_id: str
    execution_plan_id: str
    implementation_commit: str
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    frozen_proposal: FrozenApprovalProposal
    authorization: DurableAuthorizationRecord
    protocol: ProtocolTemplate
    registry: CapabilityRegistry
    model_set: FrozenModelSet | None = None

    def __post_init__(self) -> None:
        _validate_campaign_package_bindings(self)
        validate_durable_authorization_structure(
            system=self.system,
            frozen=self.frozen_proposal,
            record=self.authorization,
        )
        expected = replace(
            self.frozen_proposal.proposal.candidate_experiment,
            authorization_record_id=self.authorization.authorization_id,
            readiness=ReadinessStatus.READY,
        )
        if self.experiment != expected:
            raise ValueError(
                "campaign package experiment differs from complete authorization replay"
            )


@dataclass(frozen=True, slots=True)
class IssuedCampaignPackage(CanonicalRecord):
    """Generated executable-package root for an externally issued programme."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/issued-campaign-package'

    package_id: str
    run_plan_id: str
    execution_plan_id: str
    issued_study: IssuedDraftManifest
    publication_receipt: StudyPublicationReceipt
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    frozen_proposal: FrozenIssuedStudyApprovalProposal
    scientific_approval: DurableAuthorizationRecord
    execution_authority: StudyOperationAuthority
    protocol: ProtocolTemplate
    registry: CapabilityRegistry
    model_set: FrozenModelSet | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("package_id", self.package_id),
            ("run_plan_id", self.run_plan_id),
            ("execution_plan_id", self.execution_plan_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.execution_plan_id != f"execution.{self.run_plan_id}":
            raise ValueError("issued execution-plan ID must derive from run-plan ID")
        candidate = self.issued_study.candidate
        if (
            self.system != candidate.system
            or self.campaign != candidate.campaign
            or self.protocol != candidate.protocol
            or self.registry != self.issued_study.registry
        ):
            raise ValueError("generated package differs from issued scientific components")
        manifest_identity = ObjectIdentity.from_record(
            self.issued_study.issue_id,
            self.issued_study,
        )
        receipt_identity = ObjectIdentity.from_record(
            self.publication_receipt.receipt_id,
            self.publication_receipt,
        )
        candidate_identity = ObjectIdentity.from_record(candidate.candidate_id, candidate)
        proposal = self.frozen_proposal.proposal
        if (
            proposal.issued_study != manifest_identity
            or proposal.publication_receipt != receipt_identity
            or proposal.candidate != candidate_identity
            or proposal.candidate_experiment != candidate.experiment
            or proposal.design_origin != candidate.design_origin
            or proposal.proposer_attestation
            != ObjectIdentity.from_record(
                self.issued_study.proposer_attestation.attestation_id,
                self.issued_study.proposer_attestation,
            )
            or proposal.source_qualification_receipts != candidate.source_qualification_receipts
        ):
            raise ValueError("frozen programme proposal differs from the issued candidate")
        validate_durable_authorization_structure(
            system=self.system,
            frozen=self.frozen_proposal,
            record=self.scientific_approval,
        )
        expected_experiment = replace(
            candidate.experiment,
            authorization_record_id=self.scientific_approval.authorization_id,
            readiness=ReadinessStatus.READY,
        )
        if self.experiment != expected_experiment:
            raise ValueError("generated package experiment differs from approval replay")
        if self.publication_receipt.issue_manifest != manifest_identity:
            raise ValueError("package publication receipt binds another issued programme")
        if self.execution_authority.kind is not StudyAuthorityKind.EXPERIMENT_EXECUTION:
            raise ValueError("generated package lacks experiment execution authority")
        if self.execution_authority.subject != manifest_identity:
            raise ValueError("execution authority binds another issued programme")
        if self.execution_authority.prerequisite_authority != ObjectIdentity.from_record(
            self.scientific_approval.authorization_id,
            self.scientific_approval,
        ):
            raise ValueError("execution authority does not depend on exact scientific approval")
        if self.protocol.nonactuating and self.execution_authority.allows_actuation:
            raise ValueError("nonactuating package received actuation authority")
        if self.protocol.requires_model_set and self.model_set is None:
            raise ValueError("issued controller package lacks its frozen model set")
        if self.model_set is not None:
            model_set_identity = ObjectIdentity.from_record(
                self.model_set.model_set_id,
                self.model_set,
            )
            matching_design_inputs = tuple(
                value
                for value in self.issued_study.draft.design_inputs
                if value.object_identity == model_set_identity
                and value.materialization_sha256 == self.model_set.fingerprint()
            )
            if len(matching_design_inputs) != 1:
                raise ValueError(
                    "issued model set was not frozen as an exact candidate design input"
                )

    @property
    def implementation_commit(self) -> str:
        return self.issued_study.implementation_commit

    @property
    def authorization(self) -> DurableAuthorizationRecord:
        """Compatibility property for current compiler call sites."""

        return self.scientific_approval


@dataclass(frozen=True, slots=True)
class IssuedStudyPackage(CanonicalRecord):
    """Additive executable root retaining the mandatory standard issue identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/issued-study-package'

    package_id: str
    run_plan_id: str
    execution_plan_id: str
    issued_study: IssuedStudyManifest
    publication_receipt: StudyPublicationReceipt
    system: SystemSpec
    experiment: ExperimentSpec
    campaign: CampaignSpec
    frozen_proposal: FrozenIssuedStudyApprovalProposal
    scientific_approval: DurableAuthorizationRecord
    execution_authority: StudyOperationAuthority
    protocol: ProtocolTemplate
    registry: CapabilityRegistry
    model_set: FrozenModelSet | None = None

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveCampaignBase):
            if type(self.issued_study) is not IssuedStudyManifest:
                raise ValueError(
                    "StandardIssuedCampaignPackage requires its original issued_programme schema"
                )
            if type(self.experiment) is not ExperimentSpec:
                raise ValueError(
                    "StandardIssuedCampaignPackage requires its original experiment schema"
                )
            if type(self.frozen_proposal) is not FrozenIssuedStudyApprovalProposal:
                raise ValueError(
                    "StandardIssuedCampaignPackage requires its original frozen_proposal schema"
                )
        for name, value in (
            ("package_id", self.package_id),
            ("run_plan_id", self.run_plan_id),
            ("execution_plan_id", self.execution_plan_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.execution_plan_id != f"execution.{self.run_plan_id}":
            raise ValueError("standard issued execution-plan ID must derive from run-plan ID")
        standard_candidate = self.issued_study.candidate
        candidate = standard_candidate.base_candidate
        if (
            self.system != candidate.system
            or self.campaign != candidate.campaign
            or self.protocol != candidate.protocol
            or self.registry != self.issued_study.registry
        ):
            raise ValueError("generated standard package differs from issued scientific components")
        manifest_identity = ObjectIdentity.from_record(
            self.issued_study.issue_id,
            self.issued_study,
        )
        receipt_identity = ObjectIdentity.from_record(
            self.publication_receipt.receipt_id,
            self.publication_receipt,
        )
        candidate_identity = ObjectIdentity.from_record(
            standard_candidate.candidate_id,
            standard_candidate,
        )
        proposal = self.frozen_proposal.proposal
        if (
            proposal.issued_study != manifest_identity
            or proposal.publication_receipt != receipt_identity
            or proposal.candidate != candidate_identity
            or proposal.candidate_experiment != candidate.experiment
            or proposal.design_origin != candidate.design_origin
            or proposal.proposer_attestation
            != ObjectIdentity.from_record(
                self.issued_study.proposer_attestation.attestation_id,
                self.issued_study.proposer_attestation,
            )
            or proposal.source_qualification_receipts != candidate.source_qualification_receipts
        ):
            raise ValueError("frozen programme proposal differs from the issued standard candidate")
        validate_durable_authorization_structure(
            system=self.system,
            frozen=self.frozen_proposal,
            record=self.scientific_approval,
        )
        expected_experiment = replace(
            candidate.experiment,
            authorization_record_id=self.scientific_approval.authorization_id,
            readiness=ReadinessStatus.READY,
        )
        if self.experiment != expected_experiment:
            raise ValueError("generated standard package experiment differs from approval replay")
        if self.publication_receipt.issue_manifest != manifest_identity:
            raise ValueError("standard package publication receipt binds another issue")
        if self.execution_authority.kind is not StudyAuthorityKind.EXPERIMENT_EXECUTION:
            raise ValueError("generated standard package lacks experiment execution authority")
        if self.execution_authority.subject != manifest_identity:
            raise ValueError("execution authority binds another standard issued programme")
        if self.execution_authority.prerequisite_authority != ObjectIdentity.from_record(
            self.scientific_approval.authorization_id,
            self.scientific_approval,
        ):
            raise ValueError(
                "standard execution authority does not depend on exact scientific approval"
            )
        if self.protocol.nonactuating and self.execution_authority.allows_actuation:
            raise ValueError("nonactuating standard package received actuation authority")
        if self.protocol.requires_model_set and self.model_set is None:
            raise ValueError("standard issued controller package lacks its frozen model set")
        if self.model_set is not None:
            model_set_identity = ObjectIdentity.from_record(
                self.model_set.model_set_id,
                self.model_set,
            )
            matching_design_inputs = tuple(
                value
                for value in self.issued_study.authoring_package.draft.design_inputs
                if value.object_identity == model_set_identity
                and value.materialization_sha256 == self.model_set.fingerprint()
            )
            if len(matching_design_inputs) != 1:
                raise ValueError(
                    "standard issued model set was not frozen as an exact design input"
                )

    @property
    def implementation_commit(self) -> str:
        return self.issued_study.implementation_commit

    @property
    def authorization(self) -> DurableAuthorizationRecord:
        """Compatibility property for current compiler call sites."""

        return self.scientific_approval


@dataclass(frozen=True, slots=True)
class EnvelopeExperimentPackage(CanonicalRecord):
    """Extension and execution-envelope wrapper retaining the complete IssuedStudyPackage."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/envelope-experiment-package'

    package_id: str
    run_plan_id: str
    execution_plan_id: str
    base: IssuedStudyPackage
    issued_study: IssuedExecutableStudyManifest
    publication_receipt: ProgrammePublicationReceipt
    frozen_proposal: FrozenIssuedStudyApprovalProposal
    scientific_approval: DurableAuthorizationRecord
    experiment: ExperimentSpec
    execution_authority: StudyOperationAuthority
    execution_envelope_spec: ExecutionEnvelopeSpec

    def __post_init__(self) -> None:
        for name, value in (
            ("package_id", self.package_id),
            ("run_plan_id", self.run_plan_id),
            ("execution_plan_id", self.execution_plan_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.execution_plan_id != f"execution.{self.run_plan_id}":
            raise ValueError("EnvelopeExperimentPackage execution-plan ID must derive from run-plan ID")
        if self.issued_study.candidate.base_candidate != (self.base.issued_study.candidate):
            raise ValueError("EnvelopeExperimentPackage changes its base standard candidate")
        manifest_identity = ObjectIdentity.from_record(
            self.issued_study.issue_id,
            self.issued_study,
        )
        if (
            self.publication_receipt.issue_manifest != manifest_identity
            or self.publication_receipt.declared_member_ids
            != tuple(value.member_id for value in self.issued_study.members)
        ):
            raise ValueError("EnvelopeExperimentPackage lacks complete extension publication custody")
        candidate_identity = ObjectIdentity.from_record(
            self.issued_study.candidate.candidate_id,
            self.issued_study.candidate,
        )
        candidate_experiment = (
            self.issued_study.candidate.base_candidate.base_candidate.experiment
        )
        proposal = self.frozen_proposal.proposal
        if (
            proposal.issued_study != manifest_identity
            or proposal.publication_receipt
            != ObjectIdentity.from_record(
                self.publication_receipt.receipt_id,
                self.publication_receipt,
            )
            or proposal.candidate != candidate_identity
            or proposal.candidate_experiment != candidate_experiment
            or proposal.design_origin
            != self.issued_study.candidate.base_candidate.base_candidate.design_origin
        ):
            raise ValueError("EnvelopeExperimentPackage approval proposal differs from its issued extension candidate")
        validate_durable_authorization_structure(
            system=self.base.system,
            frozen=self.frozen_proposal,
            record=self.scientific_approval,
        )
        expected_experiment = replace(
            candidate_experiment,
            authorization_record_id=self.scientific_approval.authorization_id,
            readiness=ReadinessStatus.READY,
        )
        if self.experiment != expected_experiment:
            raise ValueError("EnvelopeExperimentPackage experiment differs from its extension approval replay")
        if (
            self.execution_authority.kind is not StudyAuthorityKind.EXPERIMENT_EXECUTION
            or self.execution_authority.subject != manifest_identity
            or self.execution_authority.prerequisite_authority
            != ObjectIdentity.from_record(
                self.scientific_approval.authorization_id,
                self.scientific_approval,
            )
        ):
            raise ValueError("EnvelopeExperimentPackage execution authority differs from extension issue/approval")
        if self.base.protocol.nonactuating and self.execution_authority.allows_actuation:
            raise ValueError("nonactuating EnvelopeExperimentPackage received actuation authority")
        if self.execution_envelope_spec.issued_study_extensions != (
            ObjectIdentity.from_record(
                self.issued_study.issued_extensions.issued_extension_set_id,
                self.issued_study.issued_extensions,
            )
        ):
            raise ValueError("EnvelopeExperimentPackage execution envelope binds another extension set")

    @property
    def system(self) -> SystemSpec:
        return self.base.system

    @property
    def campaign(self) -> CampaignSpec:
        return self.base.campaign

    @property
    def protocol(self) -> ProtocolTemplate:
        return self.base.protocol

    @property
    def registry(self) -> CapabilityRegistry:
        return self.base.registry

    @property
    def model_set(self) -> FrozenModelSet | None:
        return self.base.model_set

    @property
    def implementation_commit(self) -> str:
        return self.issued_study.implementation_commit

    @property
    def authorization(self) -> DurableAuthorizationRecord:
        return self.scientific_approval


@dataclass(frozen=True, slots=True)
class ExperimentPackage(CanonicalRecord):
    """Current package with compilation evidence only for declared JIT tasks."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/experiment-package'

    package_id: str
    run_plan_id: str
    execution_plan_id: str
    base: IssuedStudyPackage
    issued_study: IssuedExecutableStudyManifest
    publication_receipt: ProgrammePublicationReceipt
    frozen_proposal: FrozenIssuedStudyApprovalProposal
    scientific_approval: DurableAuthorizationRecord
    experiment: ExperimentSpec
    execution_authority: StudyOperationAuthority
    execution_resource_envelope_spec: ExecutionResourceEnvelopeSpec
    predevelopment_jit_signature_census: PredevelopmentJitSignatureCensus | None
    jit_graph_signature_manifest: JitGraphSignatureManifest | None

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveCampaignPackage):
            if type(self.base) is not IssuedStudyPackage:
                raise ValueError(
                    "ExperimentPackage requires its original base schema"
                )
            if type(self.issued_study) is not IssuedExecutableStudyManifest:
                raise ValueError(
                    "ExperimentPackage requires its original issued_programme schema"
                )
            if type(self.frozen_proposal) is not FrozenIssuedStudyApprovalProposal:
                raise ValueError(
                    "ExperimentPackage requires its original frozen_proposal schema"
                )
            if type(self.experiment) is not ExperimentSpec:
                raise ValueError(
                    "ExperimentPackage requires its original experiment schema"
                )
        for name, value in (
            ("package_id", self.package_id),
            ("run_plan_id", self.run_plan_id),
            ("execution_plan_id", self.execution_plan_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.execution_plan_id != f"execution.{self.run_plan_id}":
            raise ValueError("ExperimentPackage execution-plan ID must derive from run-plan ID")
        if self.issued_study.candidate.base_candidate != self.base.issued_study.candidate:
            raise ValueError("ExperimentPackage changes its base standard candidate")
        manifest_identity = ObjectIdentity.from_record(
            self.issued_study.issue_id,
            self.issued_study,
        )
        if (
            self.publication_receipt.issue_manifest != manifest_identity
            or self.publication_receipt.declared_member_ids
            != tuple(value.member_id for value in self.issued_study.members)
        ):
            raise ValueError("ExperimentPackage lacks complete extension publication custody")
        candidate_identity = ObjectIdentity.from_record(
            self.issued_study.candidate.candidate_id,
            self.issued_study.candidate,
        )
        candidate_experiment = (
            self.issued_study.candidate.base_candidate.base_candidate.experiment
        )
        proposal = self.frozen_proposal.proposal
        if (
            proposal.issued_study != manifest_identity
            or proposal.publication_receipt
            != ObjectIdentity.from_record(
                self.publication_receipt.receipt_id,
                self.publication_receipt,
            )
            or proposal.candidate != candidate_identity
            or proposal.candidate_experiment != candidate_experiment
            or proposal.design_origin
            != self.issued_study.candidate.base_candidate.base_candidate.design_origin
        ):
            raise ValueError("ExperimentPackage approval proposal differs from its issued extension candidate")
        validate_durable_authorization_structure(
            system=self.base.system,
            frozen=self.frozen_proposal,
            record=self.scientific_approval,
        )
        expected_experiment = replace(
            candidate_experiment,
            authorization_record_id=self.scientific_approval.authorization_id,
            readiness=ReadinessStatus.READY,
        )
        if self.experiment != expected_experiment:
            raise ValueError("ExperimentPackage experiment differs from its extension approval replay")
        if (
            self.execution_authority.kind is not StudyAuthorityKind.EXPERIMENT_EXECUTION
            or self.execution_authority.subject != manifest_identity
            or self.execution_authority.prerequisite_authority
            != ObjectIdentity.from_record(
                self.scientific_approval.authorization_id,
                self.scientific_approval,
            )
        ):
            raise ValueError("ExperimentPackage execution authority differs from extension issue/approval")
        if self.base.protocol.nonactuating and self.execution_authority.allows_actuation:
            raise ValueError("nonactuating ExperimentPackage received actuation authority")
        extension_identity = ObjectIdentity.from_record(
            self.issued_study.issued_extensions.issued_extension_set_id,
            self.issued_study.issued_extensions,
        )
        if self.execution_resource_envelope_spec.issued_study_extensions != extension_identity:
            raise ValueError("ExperimentPackage resource envelope differs from its issued extensions")
        validate_execution_jit_evidence(
            self.execution_resource_envelope_spec,
            self.predevelopment_jit_signature_census,
            self.jit_graph_signature_manifest,
        )

    @property
    def system(self) -> SystemSpec:
        return self.base.system

    @property
    def campaign(self) -> CampaignSpec:
        return self.base.campaign

    @property
    def protocol(self) -> ProtocolTemplate:
        return self.base.protocol

    @property
    def registry(self) -> CapabilityRegistry:
        return self.base.registry

    @property
    def model_set(self) -> FrozenModelSet | None:
        return self.base.model_set

    @property
    def implementation_commit(self) -> str:
        return self.issued_study.implementation_commit

    @property
    def authorization(self) -> DurableAuthorizationRecord:
        return self.scientific_approval


@dataclass(frozen=True, slots=True)
class ElapsedExperimentPackage(ExperimentPackage):
    """ExperimentPackage with an immutable cumulative campaign-time binding."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/elapsed-experiment-package'

    campaign_elapsed_budget: CampaignElapsedBudgetSpec
    campaign_elapsed_reservation_plan: CampaignElapsedReservationPlan

    def __post_init__(self) -> None:
        ExperimentPackage.__post_init__(self)
        campaign_identity = ObjectIdentity.from_record(
            self.base.campaign.campaign_id, self.base.campaign
        )
        budget_identity = ObjectIdentity.from_record(
            self.campaign_elapsed_budget.budget_id,
            self.campaign_elapsed_budget,
        )
        envelope_identity = ObjectIdentity.from_record(
            self.execution_resource_envelope_spec.envelope_spec_id,
            self.execution_resource_envelope_spec,
        )
        if (
            self.campaign_elapsed_budget.campaign_anchor != campaign_identity
            or self.campaign_elapsed_reservation_plan.budget != budget_identity
            or self.campaign_elapsed_reservation_plan.governed_envelope != envelope_identity
        ):
            raise ValueError(
                "ElapsedExperimentPackage budget differs from its campaign or resource envelope"
            )
        self.campaign_elapsed_reservation_plan.validate_task_census(
            self.execution_resource_envelope_spec
        )


IssuedCampaignPackageRoot = (
    IssuedCampaignPackage
    | IssuedStudyPackage
    | EnvelopeExperimentPackage
    | ExperimentPackage
    | ElapsedExperimentPackage
)


def issued_standard_candidate(
    package: IssuedCampaignPackageRoot,
) -> DraftStudyCandidate | StudyCandidate | ExecutableStudyCandidate:
    """Return the exact externally issued candidate root."""

    return package.issued_study.candidate


def issued_base_candidate(package: IssuedCampaignPackageRoot) -> DraftStudyCandidate:
    """Return the executable graph-bearing candidate without losing its outer root."""

    candidate = issued_standard_candidate(package)
    return (
        candidate.base_candidate.base_candidate
        if isinstance(candidate, ExecutableStudyCandidate)
        else candidate.base_candidate
        if isinstance(candidate, StudyCandidate)
        else candidate
    )


def assemble_issued_campaign_package(
    *,
    issued_study: IssuedDraftManifest,
    publication_receipt: StudyPublicationReceipt,
    frozen_proposal: FrozenIssuedStudyApprovalProposal,
    scientific_approval: DurableAuthorizationRecord,
    execution_authority: StudyOperationAuthority,
    run_plan_id: str,
    grantee_id: str,
    at_utc: str,
    model_set: FrozenModelSet | None = None,
) -> IssuedCampaignPackage:
    """Generate the content-complete package after two separate authorities."""

    manifest_identity = ObjectIdentity.from_record(
        issued_study.issue_id,
        issued_study,
    )
    approval_identity = ObjectIdentity.from_record(
        scientific_approval.authorization_id,
        scientific_approval,
    )
    require_study_authority(
        execution_authority,
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=manifest_identity,
        prerequisite_authority=approval_identity,
        grantee_id=grantee_id,
        at_utc=at_utc,
    )
    candidate = issued_study.candidate
    experiment = replace(
        candidate.experiment,
        authorization_record_id=scientific_approval.authorization_id,
        readiness=ReadinessStatus.READY,
    )
    package_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                'issued_study': manifest_identity,
                "publication_receipt": ObjectIdentity.from_record(
                    publication_receipt.receipt_id,
                    publication_receipt,
                ),
                "frozen_proposal": ObjectIdentity.from_record(
                    frozen_proposal.frozen_proposal_id,
                    frozen_proposal,
                ),
                "scientific_approval": approval_identity,
                "execution_authority": ObjectIdentity.from_record(
                    execution_authority.authority_id,
                    execution_authority,
                ),
                "run_plan_id": run_plan_id,
                "model_set": (
                    None
                    if model_set is None
                    else ObjectIdentity.from_record(model_set.model_set_id, model_set)
                ),
            }
        )
    ).hexdigest()
    return IssuedCampaignPackage(
        package_id=f"package.{package_digest[:32]}",
        run_plan_id=run_plan_id,
        execution_plan_id=f"execution.{run_plan_id}",
        issued_study=issued_study,
        publication_receipt=publication_receipt,
        system=candidate.system,
        experiment=experiment,
        campaign=candidate.campaign,
        frozen_proposal=frozen_proposal,
        scientific_approval=scientific_approval,
        execution_authority=execution_authority,
        protocol=candidate.protocol,
        registry=issued_study.registry,
        model_set=model_set,
    )


def assemble_issued_study_package(
    *,
    issued_study: IssuedStudyManifest,
    publication_receipt: StudyPublicationReceipt,
    frozen_proposal: FrozenIssuedStudyApprovalProposal,
    scientific_approval: DurableAuthorizationRecord,
    execution_authority: StudyOperationAuthority,
    run_plan_id: str,
    grantee_id: str,
    at_utc: str,
    model_set: FrozenModelSet | None = None,
) -> IssuedStudyPackage:
    """Generate IssuedStudyPackage while retaining standard candidate and entry identity."""

    record_type_standardissuedcampaignpackage: type[IssuedStudyPackage] = (
        RetrospectiveCampaignBase
        if isinstance(issued_study, RetrospectiveIssuedBase)
        else IssuedStudyPackage
    )
    manifest_identity = ObjectIdentity.from_record(
        issued_study.issue_id,
        issued_study,
    )
    approval_identity = ObjectIdentity.from_record(
        scientific_approval.authorization_id,
        scientific_approval,
    )
    require_study_authority(
        execution_authority,
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=manifest_identity,
        prerequisite_authority=approval_identity,
        grantee_id=grantee_id,
        at_utc=at_utc,
    )
    standard_candidate = issued_study.candidate
    candidate = standard_candidate.base_candidate
    experiment = replace(
        candidate.experiment,
        authorization_record_id=scientific_approval.authorization_id,
        readiness=ReadinessStatus.READY,
    )
    package_digest = hashlib.sha256(
        canonical_json_bytes(
            {
                'issued_study': manifest_identity,
                "publication_receipt": ObjectIdentity.from_record(
                    publication_receipt.receipt_id,
                    publication_receipt,
                ),
                "frozen_proposal": ObjectIdentity.from_record(
                    frozen_proposal.frozen_proposal_id,
                    frozen_proposal,
                ),
                "scientific_approval": approval_identity,
                "execution_authority": ObjectIdentity.from_record(
                    execution_authority.authority_id,
                    execution_authority,
                ),
                "run_plan_id": run_plan_id,
                "model_set": (
                    None
                    if model_set is None
                    else ObjectIdentity.from_record(model_set.model_set_id, model_set)
                ),
            }
        )
    ).hexdigest()
    return record_type_standardissuedcampaignpackage(
        package_id=f"package.{package_digest[:32]}",
        run_plan_id=run_plan_id,
        execution_plan_id=f"execution.{run_plan_id}",
        issued_study=issued_study,
        publication_receipt=publication_receipt,
        system=candidate.system,
        experiment=experiment,
        campaign=candidate.campaign,
        frozen_proposal=frozen_proposal,
        scientific_approval=scientific_approval,
        execution_authority=execution_authority,
        protocol=candidate.protocol,
        registry=issued_study.registry,
        model_set=model_set,
    )


def assemble_envelope_experiment_package(
    *,
    base: IssuedStudyPackage,
    issued_study: IssuedExecutableStudyManifest,
    publication_receipt: ProgrammePublicationReceipt,
    frozen_proposal: FrozenIssuedStudyApprovalProposal,
    scientific_approval: DurableAuthorizationRecord,
    execution_authority: StudyOperationAuthority,
    execution_envelope_spec: ExecutionEnvelopeSpec,
    run_plan_id: str,
    grantee_id: str,
    at_utc: str,
) -> EnvelopeExperimentPackage:
    """Bind validated extensions and an execution envelope while retaining the IssuedStudyPackage unchanged."""

    manifest_identity = ObjectIdentity.from_record(
        issued_study.issue_id,
        issued_study,
    )
    approval_identity = ObjectIdentity.from_record(
        scientific_approval.authorization_id,
        scientific_approval,
    )
    require_study_authority(
        execution_authority,
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=manifest_identity,
        prerequisite_authority=approval_identity,
        grantee_id=grantee_id,
        at_utc=at_utc,
    )
    candidate_experiment = issued_study.candidate.base_candidate.base_candidate.experiment
    experiment = replace(
        candidate_experiment,
        authorization_record_id=scientific_approval.authorization_id,
        readiness=ReadinessStatus.READY,
    )
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "base": ObjectIdentity.from_record(base.package_id, base),
                'issued_study': manifest_identity,
                "publication_receipt": ObjectIdentity.from_record(
                    publication_receipt.receipt_id,
                    publication_receipt,
                ),
                "frozen_proposal": ObjectIdentity.from_record(
                    frozen_proposal.frozen_proposal_id,
                    frozen_proposal,
                ),
                "scientific_approval": approval_identity,
                "execution_authority": ObjectIdentity.from_record(
                    execution_authority.authority_id,
                    execution_authority,
                ),
                "execution_envelope_spec": ObjectIdentity.from_record(
                    execution_envelope_spec.envelope_spec_id,
                    execution_envelope_spec,
                ),
                "run_plan_id": run_plan_id,
            }
        )
    ).hexdigest()
    return EnvelopeExperimentPackage(
        package_id=f"envelope-experiment-package.{digest[:32]}",
        run_plan_id=run_plan_id,
        execution_plan_id=f"execution.{run_plan_id}",
        base=base,
        issued_study=issued_study,
        publication_receipt=publication_receipt,
        frozen_proposal=frozen_proposal,
        scientific_approval=scientific_approval,
        experiment=experiment,
        execution_authority=execution_authority,
        execution_envelope_spec=execution_envelope_spec,
    )


def assemble_experiment_package(
    *,
    base: IssuedStudyPackage,
    issued_study: IssuedExecutableStudyManifest,
    publication_receipt: ProgrammePublicationReceipt,
    frozen_proposal: FrozenIssuedStudyApprovalProposal,
    scientific_approval: DurableAuthorizationRecord,
    execution_authority: StudyOperationAuthority,
    execution_resource_envelope_spec: ExecutionResourceEnvelopeSpec,
    predevelopment_jit_signature_census: PredevelopmentJitSignatureCensus | None,
    jit_graph_signature_manifest: JitGraphSignatureManifest | None,
    run_plan_id: str,
    grantee_id: str,
    at_utc: str,
) -> ExperimentPackage:
    """Assemble ExperimentPackage from issued extensions, deadline-free resource limits and declared JIT evidence."""

    record_type: type[ExperimentPackage] = (
        RetrospectiveCampaignPackage
        if isinstance(issued_study, RetrospectiveIssuedStudy)
        else ExperimentPackage
    )
    manifest_identity = ObjectIdentity.from_record(
        issued_study.issue_id,
        issued_study,
    )
    approval_identity = ObjectIdentity.from_record(
        scientific_approval.authorization_id,
        scientific_approval,
    )
    require_study_authority(
        execution_authority,
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=manifest_identity,
        prerequisite_authority=approval_identity,
        grantee_id=grantee_id,
        at_utc=at_utc,
    )
    candidate_experiment = issued_study.candidate.base_candidate.base_candidate.experiment
    experiment = replace(
        candidate_experiment,
        authorization_record_id=scientific_approval.authorization_id,
        readiness=ReadinessStatus.READY,
    )
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "base": ObjectIdentity.from_record(base.package_id, base),
                'issued_study': manifest_identity,
                "publication_receipt": ObjectIdentity.from_record(
                    publication_receipt.receipt_id,
                    publication_receipt,
                ),
                "frozen_proposal": ObjectIdentity.from_record(
                    frozen_proposal.frozen_proposal_id,
                    frozen_proposal,
                ),
                "scientific_approval": approval_identity,
                "execution_authority": ObjectIdentity.from_record(
                    execution_authority.authority_id,
                    execution_authority,
                ),
                "execution_resource_envelope_spec": ObjectIdentity.from_record(
                    execution_resource_envelope_spec.envelope_spec_id,
                    execution_resource_envelope_spec,
                ),
                "predevelopment_jit_signature_census": None
                if predevelopment_jit_signature_census is None
                else ObjectIdentity.from_record(
                    predevelopment_jit_signature_census.census_id,
                    predevelopment_jit_signature_census,
                ),
                "jit_graph_signature_manifest": None
                if jit_graph_signature_manifest is None
                else ObjectIdentity.from_record(
                    jit_graph_signature_manifest.manifest_id,
                    jit_graph_signature_manifest,
                ),
                "run_plan_id": run_plan_id,
            }
        )
    ).hexdigest()
    return record_type(
        package_id=f"experiment-package.{digest[:32]}",
        run_plan_id=run_plan_id,
        execution_plan_id=f"execution.{run_plan_id}",
        base=base,
        issued_study=issued_study,
        publication_receipt=publication_receipt,
        frozen_proposal=frozen_proposal,
        scientific_approval=scientific_approval,
        experiment=experiment,
        execution_authority=execution_authority,
        execution_resource_envelope_spec=execution_resource_envelope_spec,
        predevelopment_jit_signature_census=predevelopment_jit_signature_census,
        jit_graph_signature_manifest=jit_graph_signature_manifest,
    )


def assemble_elapsed_experiment_package(
    *,
    base: IssuedStudyPackage,
    issued_study: IssuedExecutableStudyManifest,
    publication_receipt: ProgrammePublicationReceipt,
    frozen_proposal: FrozenIssuedStudyApprovalProposal,
    scientific_approval: DurableAuthorizationRecord,
    execution_authority: StudyOperationAuthority,
    execution_resource_envelope_spec: ExecutionResourceEnvelopeSpec,
    predevelopment_jit_signature_census: PredevelopmentJitSignatureCensus | None,
    jit_graph_signature_manifest: JitGraphSignatureManifest | None,
    campaign_elapsed_budget: CampaignElapsedBudgetSpec,
    campaign_elapsed_reservation_plan: CampaignElapsedReservationPlan,
    run_plan_id: str,
    grantee_id: str,
    at_utc: str,
) -> ElapsedExperimentPackage:
    """Bind the cumulative campaign-time contract into the current package."""

    execution_package = assemble_experiment_package(
        base=base,
        issued_study=issued_study,
        publication_receipt=publication_receipt,
        frozen_proposal=frozen_proposal,
        scientific_approval=scientific_approval,
        execution_authority=execution_authority,
        execution_resource_envelope_spec=execution_resource_envelope_spec,
        predevelopment_jit_signature_census=predevelopment_jit_signature_census,
        jit_graph_signature_manifest=jit_graph_signature_manifest,
        run_plan_id=run_plan_id,
        grantee_id=grantee_id,
        at_utc=at_utc,
    )
    return bind_issued_execution_package_elapsed_budget(
        execution_package=execution_package,
        campaign_elapsed_budget=campaign_elapsed_budget,
        campaign_elapsed_reservation_plan=campaign_elapsed_reservation_plan,
    )


def bind_issued_execution_package_elapsed_budget(
    *,
    execution_package: ExperimentPackage,
    campaign_elapsed_budget: CampaignElapsedBudgetSpec,
    campaign_elapsed_reservation_plan: CampaignElapsedReservationPlan,
) -> ElapsedExperimentPackage:
    """Bind cumulative elapsed time while retaining the authorized execution package unchanged."""

    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "execution_package": ObjectIdentity.from_record(execution_package.package_id, execution_package),
                "campaign_elapsed_budget": ObjectIdentity.from_record(
                    campaign_elapsed_budget.budget_id,
                    campaign_elapsed_budget,
                ),
                "campaign_elapsed_reservation_plan": ObjectIdentity.from_record(
                    campaign_elapsed_reservation_plan.plan_id,
                    campaign_elapsed_reservation_plan,
                ),
            }
        )
    ).hexdigest()
    return ElapsedExperimentPackage(
        package_id=f"elapsed-experiment-package.{digest[:32]}",
        run_plan_id=execution_package.run_plan_id,
        execution_plan_id=execution_package.execution_plan_id,
        base=execution_package.base,
        issued_study=execution_package.issued_study,
        publication_receipt=execution_package.publication_receipt,
        frozen_proposal=execution_package.frozen_proposal,
        scientific_approval=execution_package.scientific_approval,
        experiment=execution_package.experiment,
        execution_authority=execution_package.execution_authority,
        execution_resource_envelope_spec=execution_package.execution_resource_envelope_spec,
        predevelopment_jit_signature_census=execution_package.predevelopment_jit_signature_census,
        jit_graph_signature_manifest=execution_package.jit_graph_signature_manifest,
        campaign_elapsed_budget=campaign_elapsed_budget,
        campaign_elapsed_reservation_plan=campaign_elapsed_reservation_plan,
    )


def _validate_campaign_package_bindings(
    package: CampaignPackage,
) -> None:
    """Validate bindings owned by the current campaign package."""

    for name, value in (
        ("package_id", package.package_id),
        ("run_plan_id", package.run_plan_id),
        ("execution_plan_id", package.execution_plan_id),
    ):
        validate_stable_id(value, field_name=name)
    if re.fullmatch(r"[0-9a-f]{40}", package.implementation_commit) is None:
        raise ValueError("implementation_commit must be a lowercase Git SHA-1")
    if package.execution_plan_id != f"execution.{package.run_plan_id}":
        raise ValueError("execution_plan_id must be derived from run_plan_id")
    if package.system.system_id not in package.campaign.system_ids:
        raise ValueError("campaign package binds a system outside its campaign")
    if package.experiment.experiment_id != package.authorization.experiment.object_id:
        raise ValueError("campaign package authorization binds another experiment")
    if package.implementation_commit != package.authorization.implementation_commit:
        raise ValueError("campaign package and authorization implementation commits differ")


@dataclass(frozen=True, slots=True)
class DualLoopPackage(CanonicalRecord):
    """Decode-only fixture replay containing authored expected outputs.

    Its findings and hypotheses may exercise non-promotable compatibility
    summaries, but can never enter claim-bearing execution or authorization.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/dual-loop-package'

    package_id: str
    execution_plan_id: str
    implementation_commit: str
    system: SystemSpec
    snapshot: EvidenceSnapshot
    snapshot_verification: SnapshotVerification
    exploration_plan: ExplorationPlan
    registry: CapabilityRegistry
    findings: tuple[ExploratoryFinding, ...]
    hypotheses: HypothesisSet
    design_context: ProspectiveDesignContext
    approval_request: ApprovalRequest
    synthesis_capability_key: str
    synthesis_capability_version: str

    def __post_init__(self) -> None:
        for name, value in (
            ("package_id", self.package_id),
            ("execution_plan_id", self.execution_plan_id),
            ("synthesis_capability_key", self.synthesis_capability_key),
        ):
            validate_stable_id(value, field_name=name)
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit) is None:
            raise ValueError("implementation_commit must be a lowercase Git SHA-1")
        if self.execution_plan_id != f"execution.{self.exploration_plan.plan_id}":
            raise ValueError("execution_plan_id must derive from the exploration plan")
        snapshot_identity = ObjectIdentity.from_record(self.snapshot.snapshot_id, self.snapshot)
        if self.exploration_plan.snapshot != snapshot_identity:
            raise ValueError("exploration plan binds another evidence snapshot")
        if self.snapshot_verification.snapshot != snapshot_identity:
            raise ValueError("snapshot verification binds another evidence snapshot")
        if self.design_context.source_snapshot != snapshot_identity:
            raise ValueError("prospective context binds another evidence snapshot")
        if self.design_context.system != ObjectIdentity.from_record(
            self.system.system_id, self.system
        ):
            raise ValueError("prospective context binds another system")
        selected = {
            item.proposal_id
            for item in self.exploration_plan.selections
            if item.disposition is ProposalDisposition.SELECTED
        }
        if {finding.proposal_id for finding in self.findings} != selected:
            raise ValueError("findings must cover exactly the selected exploration proposals")
        if any(finding.plan_id != self.exploration_plan.plan_id for finding in self.findings):
            raise ValueError("exploratory finding binds another plan")
        finding_ids = tuple(sorted(finding.finding_id for finding in self.findings))
        if finding_ids != self.hypotheses.finding_ids:
            raise ValueError("hypothesis set does not bind the package findings")
        if (
            any(
                value.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
                for value in self.findings
            )
            or self.hypotheses.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.design_context.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("dual-loop inputs must retain outcome-visible visibility")
        if self.approval_request.action != self.design_context.authority_action:
            raise ValueError("approval request changes the proposed authority action")
        if any(
            capability.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            for capability in self.registry.capabilities
        ):
            raise ValueError("exploration registry cannot contain promotion authority")
        forbidden = {
            CapabilityPermission.APPROVE_NONACTUATING,
            CapabilityPermission.COMMAND_ACTUATOR,
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
            CapabilityPermission.WRITE_CATALOG,
        }
        if any(
            forbidden.intersection(capability.permissions)
            for capability in self.registry.capabilities
        ):
            raise ValueError("exploration registry contains forbidden authority")


@dataclass(frozen=True, slots=True)
class DocumentSummary:
    object_id: str
    object_kind: str
    schema: str
    version: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class DatasetManifestSummary:
    manifest_id: str
    manifest_kind: str
    schema: str
    version: str
    fingerprint: str
    source_scope_ids: tuple[str, ...]
    destination_scope_id: str | None
    control_write_scope_id: str
    capability_registry_keys: tuple[str, ...]
    work_envelope_fingerprint: str
    outcome_access: str
    visibility_ceiling: str
    exception_codes: tuple[str, ...]
    source_bytes_read: bool
    catalog_written: bool


@dataclass(frozen=True, slots=True)
class DatasetStoragePreflightSummary:
    scope_id: str
    scope_fingerprint: str
    storage_root_id: str
    access: str
    ready: bool
    observed_free_bytes: int | None
    effective_write_floor_bytes: int | None
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublicSourceCustodySummary:
    receipt_id: str
    receipt_fingerprint: str
    source_sha256: str
    source_size_bytes: int
    raw_relative_locator: str


@dataclass(frozen=True, slots=True)
class PublicSourceAcquisitionSummary:
    plan_id: str
    plan_fingerprint: str
    member_count: int
    declared_transfer_bytes: int
    maximum_attempts: int
    source_contact_permitted: bool
    effects: tuple[str, ...]
    receipts: tuple[PublicSourceCustodySummary, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetOperationPreviewSummary:
    preview_id: str
    state: str
    manifest_id: str
    manifest_fingerprint: str
    policy_id: str
    policy_fingerprint: str
    request_id: str
    request_fingerprint: str
    authorization_id: str | None
    authorization_fingerprint: str | None
    decision: str
    expected_implementation_commit: str
    observed_implementation_commit: str | None
    repository_verified: bool
    source_preflights: tuple[DatasetStoragePreflightSummary, ...]
    destination_preflight: DatasetStoragePreflightSummary | None
    control_preflight: DatasetStoragePreflightSummary
    reason_codes: tuple[str, ...]
    source_bytes_read: bool
    network_bytes: int
    external_bytes_written: bool
    catalog_written: bool
    outcome_access: str


@dataclass(frozen=True, slots=True)
class DatasetRegistrationSummary:
    receipt_id: str
    receipt_fingerprint: str
    receipt_storage_root_id: str
    receipt_relative_locator: str
    dataset_snapshot_fingerprint: str
    materialization_id: str
    materialization_fingerprint: str
    release_id: str
    custody_state: str
    storage_root_id: str
    relative_locator: str
    distinct_source_bytes: int
    source_full_hash_passes: int
    source_full_hash_read_ceiling_bytes: int
    network_bytes: int
    source_mutated: bool
    download_performed: bool
    dataset_bytes_written: bool
    control_batch_published: bool
    replayed_existing: bool
    new_external_artifacts_created: bool
    catalog_written: bool
    outcome_access: str


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildSummary:
    installation_id: str
    receipt_id: str
    receipt_fingerprint: str
    receipt_storage_root_id: str
    receipt_relative_locator: str
    database_relative_path: str
    database_sha256: str
    database_size_bytes: int
    projection_sha256: str
    projection_state_fingerprint: str
    source_bytes_read: int
    network_bytes: int
    external_receipt_written: bool
    catalog_written: bool
    outcome_access: str


@dataclass(frozen=True, slots=True)
class DatasetCatalogRecordSummary:
    record_id: str
    record_kind: DatasetCatalogRecordKind
    schema: str
    version: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class DatasetListSummary:
    records: tuple[DatasetCatalogRecordSummary, ...]
    returned_count: int
    limit: int
    records_examined: int
    query_steps: int
    max_records_examined: int
    max_query_steps: int
    has_more: bool
    next_cursor: str | None
    snapshot_fingerprint: str
    projection_state_fingerprint: str
    projection_anchor_fingerprint: str


@dataclass(frozen=True, slots=True)
class DatasetShowSummary:
    record_id: str
    record_kind: DatasetCatalogRecordKind
    schema: str
    version: str
    fingerprint: str
    metadata_document: Mapping[str, object]
    payload_read: bool
    source_bytes_read: bool
    outcome_read: bool


@dataclass(frozen=True, slots=True)
class CapabilityListSummary(CanonicalRecord):
    """Bounded static capability discovery without provider execution."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/capability-list-summary'

    catalog_id: str
    catalog_sha256: str
    registrations: tuple[CandidateCapabilityRegistration, ...]
    bundle_bindings: tuple[CapabilityBundleBinding, ...]
    execution_availability: tuple[CapabilityExecutionAvailability, ...]
    returned_count: int
    limit: int
    has_more: bool
    next_cursor: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.catalog_id, field_name="catalog_id")
        validate_sha256(self.catalog_sha256, field_name="catalog_sha256")
        if self.returned_count != len(self.registrations):
            raise ValueError("capability returned_count differs from registrations")
        page_keys = {
            (
                value.manifest.capability_key,
                value.manifest.capability_version,
            )
            for value in self.registrations
        }
        binding_keys = {
            (value.capability_key, value.capability_version) for value in self.bundle_bindings
        }
        if len(binding_keys) != len(self.bundle_bindings) or not binding_keys <= page_keys:
            raise ValueError("capability bundle bindings differ from the returned page")
        availability_ids = tuple(value.availability_id for value in self.execution_availability)
        if tuple(sorted(set(availability_ids))) != availability_ids:
            raise ValueError("capability execution availability must be sorted and unique")
        manifest_identities = {
            ObjectIdentity.from_record(
                "capability-manifest."
                f"{value.manifest.capability_key}."
                f"{value.manifest.capability_version.replace('.', '-')}",
                value.manifest,
            )
            for value in self.registrations
        }
        if any(
            value.discovery_subject not in manifest_identities
            for value in self.execution_availability
        ):
            raise ValueError("capability execution availability differs from the returned page")
        if self.limit <= 0 or self.returned_count > self.limit:
            raise ValueError("capability list exceeds its positive page limit")
        if self.has_more != (self.next_cursor is not None):
            raise ValueError("capability pagination state is inconsistent")


@dataclass(frozen=True, slots=True)
class CapabilityShowSummary(CanonicalRecord):
    """One exact static capability registration and catalog identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/capability-show-summary'

    catalog_id: str
    catalog_sha256: str
    registration: CandidateCapabilityRegistration
    bundle_binding: CapabilityBundleBinding | None
    execution_availability: CapabilityExecutionAvailability | None

    def __post_init__(self) -> None:
        validate_stable_id(self.catalog_id, field_name="catalog_id")
        validate_sha256(self.catalog_sha256, field_name="catalog_sha256")
        if self.bundle_binding is not None and (
            self.bundle_binding.capability_key != self.registration.manifest.capability_key
            or self.bundle_binding.capability_version
            != self.registration.manifest.capability_version
        ):
            raise ValueError("capability bundle binding differs from registration")
        if self.execution_availability is not None:
            expected = ObjectIdentity.from_record(
                "capability-manifest."
                f"{self.registration.manifest.capability_key}."
                f"{self.registration.manifest.capability_version.replace('.', '-')}",
                self.registration.manifest,
            )
            if self.execution_availability.discovery_subject != expected:
                raise ValueError("capability execution availability differs from registration")


def _validate_compact_local_emission(
    *,
    emitted: bool,
    relative_path: str | None,
    payload_sha256: str | None,
    byte_count: int,
    label: str,
) -> None:
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise ValueError(f"{label} byte count must be a nonnegative integer")
    if emitted:
        if relative_path is None or payload_sha256 is None or byte_count <= 0:
            raise ValueError(f"{label} emission lacks its exact local identity")
        validate_relative_locator(relative_path)
        validate_sha256(payload_sha256, field_name=f"{label}_sha256")
    elif relative_path is not None or payload_sha256 is not None or byte_count != 0:
        raise ValueError(f"{label} preview cannot report local writes")


@dataclass(frozen=True, slots=True)
class SourceProfileCompilationSummary:
    profile_id: str
    profile_fingerprint: str
    compilation_id: str
    compilation_fingerprint: str
    disposition: str
    reason_codes: tuple[str, ...]
    source_read: bool
    authority_granted: bool
    local_compilation_emitted: bool
    local_compilation_relative_path: str | None
    local_compilation_sha256: str | None
    local_bytes_written: int

    def __post_init__(self) -> None:
        for name in ("profile_id", "compilation_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("profile_fingerprint", "compilation_fingerprint"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("source-profile reason codes must be sorted and unique")
        positive = self.disposition == "REQUESTS_COMPILED_AUTHORITY_PENDING"
        if self.disposition not in {
            "REQUESTS_COMPILED_AUTHORITY_PENDING",
            "AUTHORITY_REQUIRED",
        } or positive == bool(self.reason_codes):
            raise ValueError("source-profile disposition/reasons are inconsistent")
        if self.source_read or self.authority_granted:
            raise ValueError("source-profile compilation crossed its nonacquiring boundary")
        _validate_compact_local_emission(
            emitted=self.local_compilation_emitted,
            relative_path=self.local_compilation_relative_path,
            payload_sha256=self.local_compilation_sha256,
            byte_count=self.local_bytes_written,
            label="source-profile compilation",
        )


@dataclass(frozen=True, slots=True)
class LinkedCampaignCompilationSummary:
    profile_id: str
    profile_fingerprint: str
    compilation_id: str
    compilation_fingerprint: str
    disposition: str
    reason_codes: tuple[str, ...]
    package_roles: tuple[str, ...]
    provider_registry_fingerprints: tuple[str, ...]
    authority_granted: bool
    executed: bool
    local_compilation_emitted: bool
    local_compilation_relative_path: str | None
    local_compilation_sha256: str | None
    local_bytes_written: int

    def __post_init__(self) -> None:
        for name in ("profile_id", "compilation_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("profile_fingerprint", "compilation_fingerprint"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.reason_codes != tuple(sorted(set(self.reason_codes))):
            raise ValueError("linked-profile reason codes must be sorted and unique")
        if self.package_roles != tuple(sorted(value.value for value in LinkedCampaignPackageRole)):
            raise ValueError("linked-profile summary requires the exact four package roles")
        if (
            self.provider_registry_fingerprints
            != tuple(sorted(set(self.provider_registry_fingerprints)))
            or not self.provider_registry_fingerprints
        ):
            raise ValueError("linked-profile provider fingerprints must be sorted and unique")
        for value in self.provider_registry_fingerprints:
            validate_sha256(value, field_name="provider_registry_fingerprint")
        positive = self.disposition == "COMPILED_AUTHORITY_PENDING"
        if self.disposition not in {
            "COMPILED_AUTHORITY_PENDING",
            "SOURCE_AUTHORITY_REQUIRED",
            "EVIDENCE_PROFILE_BLOCKED",
            "CANDIDATE_COMPILATION_BLOCKED",
            "EXECUTABLE_BINDING_REQUIRED",
        } or positive == bool(self.reason_codes):
            raise ValueError("linked-profile disposition/reasons are inconsistent")
        if self.authority_granted or self.executed:
            raise ValueError("linked-profile compilation crossed its nonactuating boundary")
        _validate_compact_local_emission(
            emitted=self.local_compilation_emitted,
            relative_path=self.local_compilation_relative_path,
            payload_sha256=self.local_compilation_sha256,
            byte_count=self.local_bytes_written,
            label="linked-profile compilation",
        )


@dataclass(frozen=True, slots=True)
class ExperimentPackageAssemblySummary:
    package_id: str
    package_fingerprint: str
    run_plan_id: str
    execution_plan_id: str
    registry_fingerprint: str
    authority_replayed: bool
    executed: bool
    local_package_emitted: bool
    local_package_relative_path: str | None
    local_package_sha256: str | None
    local_bytes_written: int

    def __post_init__(self) -> None:
        for name in ("package_id", "run_plan_id", "execution_plan_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("package_fingerprint", "registry_fingerprint"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.execution_plan_id != f"execution.{self.run_plan_id}":
            raise ValueError("ExperimentPackageAssemblySummary execution-plan ID differs from its run-plan ID")
        if not self.authority_replayed or self.executed:
            raise ValueError("ExperimentPackageAssemblySummary requires replay without execution")
        _validate_compact_local_emission(
            emitted=self.local_package_emitted,
            relative_path=self.local_package_relative_path,
            payload_sha256=self.local_package_sha256,
            byte_count=self.local_bytes_written,
            label="ExperimentPackage",
        )


@dataclass(frozen=True, slots=True)
class SystemSummary:
    system_id: str
    world_id: str
    world_kind: str
    relation_id: str
    denominator_quantity_ids: tuple[str, ...]
    history_quantity_ids: tuple[str, ...]
    action_quantity_ids: tuple[str, ...]
    receiver_quantity_ids: tuple[str, ...]
    horizon_id: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class CompilationSummary:
    package_id: str
    package_fingerprint: str
    run_plan_id: str
    run_plan_fingerprint: str
    execution_plan_id: str
    execution_plan_fingerprint: str
    execution_topology_sha256: str
    registry_fingerprint: str
    topological_step_ids: tuple[str, ...]
    parallel_task_groups: tuple[tuple[str, ...], ...]
    nonactuating: bool
    controller_requested: bool


@dataclass(frozen=True, slots=True)
class PreissueReadinessSummary:
    candidate_id: str
    candidate_fingerprint: str
    prospective_extension_set_id: str
    prospective_extension_set_fingerprint: str
    run_plan_id: str
    registry_fingerprint: str
    execution_topology_sha256: str
    task_count: int
    stage_task_counts: tuple[tuple[str, int], ...]
    runner_ids: tuple[str, ...]
    external_input_ids: tuple[str, ...]
    output_relative_locators: tuple[str, ...]
    adjudication_relative_locator: str
    minimum_free_bytes: int
    source_byte_limit: int
    maximum_parallel_tasks: int
    aggregate_memory_ceiling_bytes: int
    authority_input_ids: tuple[str, ...]
    closure_sections: tuple[tuple[str, bool, tuple[str, ...]], ...]
    source_closure_verified: bool = True
    authority_issued: bool = False
    study_issued: bool = False
    workers_created: bool = False
    source_contacted: bool = False
    outcomes_read: bool = False
    external_bytes_written: int = 0
    catalog_accessed: bool = False
    campaign_elapsed_budget_verified: bool = False
    campaign_elapsed_reservation_plan_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class StudyCompilationSummary(CanonicalRecord):
    """Public candidate result plus its deliberately narrow side-effect profile."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/study-compilation-summary'

    report: StudyCompilationReport
    local_candidate_emitted: bool
    local_candidate_relative_path: str | None
    local_candidate_sha256: str | None
    local_bytes_written: int
    workers_created: bool = False
    external_bytes_written: int = 0
    catalog_mutated: bool = False
    authority_issued: bool = False
    protected_outcomes_read: bool = False

    def __post_init__(self) -> None:
        if self.local_candidate_emitted:
            if (
                self.local_candidate_relative_path is None
                or self.local_candidate_sha256 is None
                or self.local_bytes_written <= 0
            ):
                raise ValueError("local candidate emission lacks its exact identity")
            validate_relative_locator(self.local_candidate_relative_path)
            validate_sha256(
                self.local_candidate_sha256,
                field_name="local_candidate_sha256",
            )
        elif (
            self.local_candidate_relative_path is not None
            or self.local_candidate_sha256 is not None
            or self.local_bytes_written != 0
        ):
            raise ValueError("candidate preview cannot report local writes")
        if (
            self.workers_created
            or self.external_bytes_written != 0
            or self.catalog_mutated
            or self.authority_issued
            or self.protected_outcomes_read
        ):
            raise ValueError("candidate compilation crossed its nonactuating boundary")


@dataclass(frozen=True, slots=True)
class ExecutableStudyCompilationSummary(CanonicalRecord):
    """Extension-aware public compilation with exact byte-custody receipts."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/executable-study-compilation-summary'

    report: ExecutableStudyCompilationReport
    local_candidate_emitted: bool
    local_candidate_relative_path: str | None
    local_candidate_sha256: str | None
    local_bytes_written: int
    workers_created: bool = False
    external_bytes_written: int = 0
    catalog_mutated: bool = False
    authority_issued: bool = False
    protected_outcomes_read: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveCompilationSummary):
            if type(self.report) is not ExecutableStudyCompilationReport:
                raise ValueError(
                    "ExecutableStudyCompilationSummary requires its original report schema"
                )
        if self.local_candidate_emitted:
            if (
                self.local_candidate_relative_path is None
                or self.local_candidate_sha256 is None
                or self.local_bytes_written <= 0
            ):
                raise ValueError("local executable study candidate emission lacks exact identity")
            validate_relative_locator(self.local_candidate_relative_path)
            validate_sha256(self.local_candidate_sha256, field_name="local_candidate_sha256")
        elif (
            self.local_candidate_relative_path is not None
            or self.local_candidate_sha256 is not None
            or self.local_bytes_written != 0
        ):
            raise ValueError("executable study candidate preview cannot report local writes")
        if (
            self.workers_created
            or self.external_bytes_written != 0
            or self.catalog_mutated
            or self.authority_issued
            or self.protected_outcomes_read
        ):
            raise ValueError("executable study candidate compilation crossed its nonactuating boundary")


@dataclass(frozen=True, slots=True)
class StudyBundleCompilationSummary(CanonicalRecord):
    """Pure public bundle result with an explicit zero-side-effect contract."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/study-bundle-compilation-summary'

    report: StudyBundleCompilationReport
    local_candidate_emitted: bool
    local_candidate_relative_path: str | None
    local_candidate_sha256: str | None
    local_bytes_written: int
    workers_created: bool = False
    external_bytes_written: int = 0
    catalog_mutated: bool = False
    authority_issued: bool = False
    protected_outcomes_read: bool = False

    def __post_init__(self) -> None:
        if self.local_candidate_emitted:
            if (
                self.local_candidate_relative_path is None
                or self.local_candidate_sha256 is None
                or self.local_bytes_written <= 0
            ):
                raise ValueError("local bundle candidate emission lacks its exact identity")
            validate_relative_locator(self.local_candidate_relative_path)
            validate_sha256(
                self.local_candidate_sha256,
                field_name="local_candidate_sha256",
            )
        elif (
            self.local_candidate_relative_path is not None
            or self.local_candidate_sha256 is not None
            or self.local_bytes_written != 0
        ):
            raise ValueError("bundle candidate preview cannot report local writes")
        if (
            self.workers_created
            or self.external_bytes_written != 0
            or self.catalog_mutated
            or self.authority_issued
            or self.protected_outcomes_read
        ):
            raise ValueError("bundle compilation crossed its nonactuating boundary")


@dataclass(frozen=True, slots=True)
class StudyIssueSummary(CanonicalRecord):
    """Public preview or confirmed external issue result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/study-issue-summary'

    manifest: IssuedStudyManifest
    confirmed: bool
    publication_receipt: StudyPublicationReceipt | None
    planned_member_count: int
    planned_payload_bytes: int
    external_bytes_written: int
    publication_receipt_bytes: int
    workers_created: bool = False
    catalog_mutated: bool = False
    authority_issued: bool = False
    source_acquired: bool = False
    protected_outcomes_read: bool = False

    def __post_init__(self) -> None:
        if self.planned_member_count != len(self.manifest.members):
            raise ValueError("issue summary member count differs from its manifest")
        if self.planned_payload_bytes <= 0:
            raise ValueError("issue summary requires a positive planned byte count")
        if self.confirmed:
            if (
                self.publication_receipt is None
                or self.external_bytes_written
                != (self.publication_receipt.total_payload_bytes + self.publication_receipt_bytes)
                or self.publication_receipt_bytes != len(self.publication_receipt.canonical_bytes())
            ):
                raise ValueError("confirmed issue lacks its exact publication receipt")
        elif (
            self.publication_receipt is not None
            or self.external_bytes_written != 0
            or self.publication_receipt_bytes != 0
        ):
            raise ValueError("issue preview cannot report external publication")
        if (
            self.workers_created
            or self.catalog_mutated
            or self.authority_issued
            or self.source_acquired
            or self.protected_outcomes_read
        ):
            raise ValueError("programme issue crossed its custody-only boundary")


@dataclass(frozen=True, slots=True)
class ExecutableStudyIssueSummary(CanonicalRecord):
    """Preview or confirmed composite extension publication result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/executable-study-issue-summary'

    manifest: IssuedExecutableStudyManifest
    confirmed: bool
    publication_receipt: ExtensionPublicationReceipt | None
    base_member_count: int
    extension_member_count: int
    planned_extension_payload_bytes: int
    external_bytes_written: int
    publication_receipt_bytes: int
    workers_created: bool = False
    catalog_mutated: bool = False
    authority_issued: bool = False
    source_acquired: bool = False
    protected_outcomes_read: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveIssueSummary):
            if type(self.manifest) is not IssuedExecutableStudyManifest:
                raise ValueError("ExecutableStudyIssueSummary requires its original manifest schema")
        if self.base_member_count != len(self.manifest.base.members):
            raise ValueError("executable study issue base member count differs")
        if self.extension_member_count != len(self.manifest.issued_extensions.members):
            raise ValueError("executable study issue extension member count differs")
        if self.planned_extension_payload_bytes <= 0:
            raise ValueError("executable study issuance requires positive planned extension bytes")
        if self.confirmed:
            if (
                self.publication_receipt is None
                or self.publication_receipt.extension_payload_bytes
                != self.planned_extension_payload_bytes
                or self.publication_receipt_bytes != len(self.publication_receipt.canonical_bytes())
                or self.external_bytes_written
                != self.planned_extension_payload_bytes + self.publication_receipt_bytes
            ):
                raise ValueError("confirmed executable study issue lacks exact publication custody")
        elif (
            self.publication_receipt is not None
            or self.external_bytes_written != 0
            or self.publication_receipt_bytes != 0
        ):
            raise ValueError("executable study issue preview cannot report external publication")
        if (
            self.workers_created
            or self.catalog_mutated
            or self.authority_issued
            or self.source_acquired
            or self.protected_outcomes_read
        ):
            raise ValueError("executable study issuance crossed its custody-only boundary")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyIssueSummary(CanonicalRecord):
    """Preview or confirmed parent-only publication over three immutable child issues."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/multi-world-study-issue-summary'

    manifest: MultiWorldStudyIssueManifest
    confirmed: bool
    publication_receipt: MultiWorldStudyPublicationReceipt | None
    child_issue_count: int
    planned_manifest_payload_bytes: int
    external_bytes_written: int
    publication_receipt_bytes: int
    workers_created: bool = False
    catalog_mutated: bool = False
    authority_issued: bool = False
    source_acquired: bool = False
    protected_outcomes_read: bool = False

    def __post_init__(self) -> None:
        if self.child_issue_count != 3 or self.child_issue_count != len(self.manifest.children):
            raise ValueError("bundle issue summary changes its three-child roster")
        if self.planned_manifest_payload_bytes != len(self.manifest.canonical_bytes()):
            raise ValueError("bundle issue summary changes parent manifest bytes")
        if self.confirmed:
            if (
                self.publication_receipt is None
                or self.publication_receipt.manifest_payload_bytes
                != self.planned_manifest_payload_bytes
                or self.publication_receipt_bytes != len(self.publication_receipt.canonical_bytes())
                or self.external_bytes_written
                != self.planned_manifest_payload_bytes + self.publication_receipt_bytes
            ):
                raise ValueError("confirmed bundle issue lacks exact parent publication custody")
        elif (
            self.publication_receipt is not None
            or self.external_bytes_written != 0
            or self.publication_receipt_bytes != 0
        ):
            raise ValueError("bundle issue preview cannot report external publication")
        if (
            self.workers_created
            or self.catalog_mutated
            or self.authority_issued
            or self.source_acquired
            or self.protected_outcomes_read
        ):
            raise ValueError("bundle issue crossed its custody-only boundary")


@dataclass(frozen=True, slots=True)
class CampaignAuthorityRequired(CanonicalRecord):
    """Nonterminal reveal-barrier result for same-plan public resume."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/campaign-authority-required'

    barrier_id: str
    run_id: str
    issued_study: ObjectIdentity
    execution_plan: ObjectIdentity
    execution_authority: ObjectIdentity
    barrier_task_ids: tuple[str, ...]
    completed_task_ids: tuple[str, ...]
    sealed_predecessor_receipt_ids: tuple[str, ...]
    immutable_resume_prefix_sha256: str
    required_authority_kind: StudyAuthorityKind
    required_outcome_access: OutcomeAccess
    same_plan_resume_required: bool
    protected_outcomes_read: bool

    def __post_init__(self) -> None:
        for name, value in (("barrier_id", self.barrier_id), ("run_id", self.run_id)):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("barrier_task_ids", self.barrier_task_ids),
            ("completed_task_ids", self.completed_task_ids),
            ("sealed_predecessor_receipt_ids", self.sealed_predecessor_receipt_ids),
        ):
            if tuple(sorted(set(values))) != values:
                raise ValueError(f"{name} must be sorted and unique")
        validate_sha256(
            self.immutable_resume_prefix_sha256,
            field_name="immutable_resume_prefix_sha256",
        )
        if not self.barrier_task_ids:
            raise ValueError("authority-required result lacks a reveal barrier")
        if self.required_authority_kind is not StudyAuthorityKind.OUTCOME_REVEAL:
            raise ValueError("campaign barrier requests another authority kind")
        if self.required_outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("campaign barrier requests another outcome-access ceiling")
        if not self.same_plan_resume_required or self.protected_outcomes_read:
            raise ValueError("authority-required result changed or crossed the sealed prefix")


@dataclass(frozen=True, slots=True)
class ExplorationCompilationSummary:
    package_id: str
    package_fingerprint: str
    plan_id: str
    plan_fingerprint: str
    execution_plan_id: str
    execution_plan_fingerprint: str
    registry_fingerprint: str
    topological_task_ids: tuple[str, ...]
    parallel_task_groups: tuple[tuple[str, ...], ...]
    outcome_access: str
    visibility_ceiling: str
    promotable: bool
    nonactuating: bool


@dataclass(frozen=True, slots=True)
class ExplorationPortfolioSummary:
    package_id: str
    snapshot_id: str
    plan_id: str
    proposal_ids: tuple[str, ...]
    selected_proposal_ids: tuple[str, ...]
    rejected_proposal_ids: tuple[str, ...]
    projection_ids: tuple[str, ...]
    outcome_access: str
    visibility_ceiling: str
    promotable: bool


@dataclass(frozen=True, slots=True)
class HypothesisSummary:
    hypothesis_set_id: str
    finding_ids: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]
    visibility_ceiling: str
    promotable: bool
    result_basis: str


@dataclass(frozen=True, slots=True)
class NominationSummary:
    decision_id: str
    decision_kind: str
    nomination_ids: tuple[str, ...]
    requested_rungs: tuple[str, ...]
    fresh_evidence_required: bool
    source_visibility_ceiling: str
    promotable: bool
    result_basis: str


@dataclass(frozen=True, slots=True)
class ProposalSummary:
    proposal_ids: tuple[str, ...]
    experiment_ids: tuple[str, ...]
    readiness: tuple[str, ...]
    authority_action: str
    evaluation_visibility_ceiling: str
    source_visibility_ceiling: str
    grants_claim_promotion: bool
    result_basis: str


@dataclass(frozen=True, slots=True)
class AuthorizationSummary:
    authorization_id: str
    decision: str
    action: str
    reason_codes: tuple[str, ...]
    outcome_access: str
    plan_mutated: bool
    grants_claim_promotion: bool


@dataclass(frozen=True, slots=True)
class ScientificInspectionSummary:
    requested_kind: str
    object_id: str
    found: bool
    catalog_kind: str | None
    object_schema: str | None
    object_fingerprint: str | None
    visibility_ceiling: str | None
    categorical_status: str | None
    external_relative_path: str | None
    payload_read: bool
    sealed_outcome_read: bool


@dataclass(frozen=True, slots=True)
class WriteEffectSummary:
    action: str
    confirmed: bool
    effects: tuple[str, ...]
    observed_free_bytes: int | None = None
    effective_write_floor_bytes: int | None = None
    storage_write_ready: bool | None = None
    storage_reason_codes: tuple[str, ...] = ()
    execution_assurance_profile: str | None = None
    execution_assurance_codes: tuple[str, ...] = ()
    execution_resource_reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CampaignMaintenanceStopSummary:
    run_id: str
    stopped: bool
    ready: bool
    completed_tasks: int
    unstarted_tasks: int
    checkpoint_relative_path: str | None


@dataclass(frozen=True, slots=True)
class CampaignMaintenanceInterruptionSummary:
    run_id: str
    stopped: bool
    completed_tasks: int
    interrupted_tasks: int
    unstarted_tasks: int
    continuation_run_id: str | None
    checkpoint_relative_path: str | None


@dataclass(frozen=True, slots=True)
class DoctorBackendSummary:
    backend_id: str
    state: str
    enforces_resources: bool
    enforces_no_network: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DoctorDependencySummary:
    dependency_id: str
    available: bool
    required_for_actions: tuple[str, ...]
    reason_codes: tuple[str, ...]
    installed_version: str | None = None
    required_version: str | None = None
    version_matches: bool | None = None


@dataclass(frozen=True, slots=True)
class DoctorAuthorityBoundary:
    boundary_id: str
    nondelegable: bool
    reason_code: str


@dataclass(frozen=True, slots=True)
class DoctorActionReadiness:
    action_id: str
    infrastructure_ready: bool
    authority_required: tuple[str, ...]
    executable_now: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DoctorSummary:
    package_version: str
    python_version: str
    repository_root: str
    git_branch: str | None
    git_commit: str | None
    git_dirty: bool | None
    capability_count: int
    capability_registry_fingerprint: str | None
    catalog_relative_path: str
    catalog_present: bool
    catalog_state: str
    catalog_reason_codes: tuple[str, ...]
    external_root: str
    external_present: bool
    external_writable: bool
    mount_active: bool | None
    canonical_contained: bool | None
    symlink_safe: bool | None
    mount_source: str | None
    volume_identity: str | None
    filesystem_type: str | None
    observed_free_bytes: int | None
    effective_write_floor_bytes: int | None
    storage_read_ready: bool
    storage_write_ready: bool
    backends: tuple[DoctorBackendSummary, ...]
    optional_dependencies: tuple[DoctorDependencySummary, ...]
    authority_boundaries: tuple[DoctorAuthorityBoundary, ...]
    action_readiness: tuple[DoctorActionReadiness, ...]
    warnings: tuple[str, ...]
    environment_route: str | None = None
    operating_system: str | None = None
    architecture: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogSummary:
    catalog_relative_path: str
    present: bool
    schema_valid: bool | None
    schema_version: int | None
    alembic_revision: str | None
    page_count: int | None
    page_size: int | None
    forbidden_payload_columns: tuple[str, ...]
    application_id: int | None = None
    integrity_result: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogObjectSummary:
    object_id: str
    kind: str
    object_schema: str
    object_fingerprint: str
    visibility_ceiling: str
    categorical_status: str
    storage_root_id: str | None
    external_relative_path: str | None
    external_identity_kind: str | None


@dataclass(frozen=True, slots=True)
class CatalogQuerySummary:
    matched_count: int
    returned_count: int
    objects: tuple[CatalogObjectSummary, ...]
    next_cursor: str | None = None
    match_count_limit: int = 1_000
    matched_count_truncated: bool = False
    has_more: bool = False
    query_work_limit: int = 100_000
    query_work_steps: int = 0


@dataclass(frozen=True, slots=True)
class DatabaseMutationSummary:
    action: str
    catalog_relative_path: str
    schema_valid: bool
    schema_version: int
    alembic_revision: str
    page_count: int
    page_size: int


@dataclass(frozen=True, slots=True)
class CatalogRebuildSummary:
    database_relative_path: str
    database_sha256: str
    database_size_bytes: int
    projection_sha256: str
    snapshot_sha256: str
    integrity_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrospectiveCampaignBase(IssuedStudyPackage):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/retrospective-campaign-base'

    issued_study: RetrospectiveIssuedBase
    experiment: RetrospectiveExperimentSpec
    frozen_proposal: FrozenRetrospectiveApproval

    def __post_init__(self) -> None:
        if type(self.issued_study) is not RetrospectiveIssuedBase:
            raise ValueError("RetrospectiveCampaignBase requires its exact issued_programme schema")
        if type(self.experiment) is not RetrospectiveExperimentSpec:
            raise ValueError("RetrospectiveCampaignBase requires its exact experiment schema")
        if type(self.frozen_proposal) is not FrozenRetrospectiveApproval:
            raise ValueError("RetrospectiveCampaignBase requires its exact frozen_proposal schema")
        IssuedStudyPackage.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveCampaignPackage(ExperimentPackage):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/retrospective-campaign-package'

    base: RetrospectiveCampaignBase
    issued_study: RetrospectiveIssuedStudy
    frozen_proposal: FrozenRetrospectiveApproval
    experiment: RetrospectiveExperimentSpec

    publication_receipt: RetrospectivePublicationReceipt

    def __post_init__(self) -> None:
        if (
            self.publication_receipt is not None
            and type(self.publication_receipt) is not RetrospectivePublicationReceipt
        ):
            raise ValueError("historical publication receipt schema differs")
        if type(self.base) is not RetrospectiveCampaignBase:
            raise ValueError("RetrospectiveCampaignPackage requires its exact base schema")
        if type(self.issued_study) is not RetrospectiveIssuedStudy:
            raise ValueError(
                "RetrospectiveCampaignPackage requires its exact issued_programme schema"
            )
        if type(self.frozen_proposal) is not FrozenRetrospectiveApproval:
            raise ValueError(
                "RetrospectiveCampaignPackage requires its exact frozen_proposal schema"
            )
        if type(self.experiment) is not RetrospectiveExperimentSpec:
            raise ValueError("RetrospectiveCampaignPackage requires its exact experiment schema")
        ExperimentPackage.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveCompilationSummary(ExecutableStudyCompilationSummary):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/retrospective-compilation-summary'

    report: RetrospectiveExtensionReport

    def __post_init__(self) -> None:
        if type(self.report) is not RetrospectiveExtensionReport:
            raise ValueError("RetrospectiveCompilationSummary requires its exact report schema")
        ExecutableStudyCompilationSummary.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveIssueSummary(ExecutableStudyIssueSummary):
    """Explicit historical compatibility; the predecessor schema is unchanged."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/retrospective-issue-summary'

    manifest: RetrospectiveIssuedStudy

    publication_receipt: RetrospectivePublicationReceipt | None

    def __post_init__(self) -> None:
        if (
            self.publication_receipt is not None
            and type(self.publication_receipt) is not RetrospectivePublicationReceipt
        ):
            raise ValueError("historical publication receipt schema differs")
        if type(self.manifest) is not RetrospectiveIssuedStudy:
            raise ValueError("RetrospectiveIssueSummary requires its exact manifest schema")
        ExecutableStudyIssueSummary.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveBaseCompilationSummary(StudyCompilationSummary):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/retrospective-base-compilation-summary'

    report: RetrospectiveStandardReport

    def __post_init__(self) -> None:
        if type(self.report) is not RetrospectiveStandardReport:
            raise ValueError("historical summary requires its exact derived schema")
        StudyCompilationSummary.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveBaseIssueSummary(StudyIssueSummary):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/api/retrospective-base-issue-summary'

    manifest: RetrospectiveIssuedBase

    def __post_init__(self) -> None:
        if type(self.manifest) is not RetrospectiveIssuedBase:
            raise ValueError("historical summary requires its exact derived schema")
        StudyIssueSummary.__post_init__(self)
