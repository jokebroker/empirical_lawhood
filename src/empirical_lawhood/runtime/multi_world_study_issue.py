"Bundle issue, staged reveal and recovery overlay for three-child bundles."

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.multi_world_study import ArchiveOutcomeProtectionPlan, ArchiveToSimulatorPartialMorphismSpec, MultiWorldJointAdjudicationPlan, MultiWorldChildRole, MultiWorldOutcomeBarrierPlan, MultiWorldOutcomeDomain, MultiWorldStudyChildScientificBinding
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority

from .artifacts import (
    ArtifactMaterialization,
    ArtifactPublicationBinding,
    LogicalArtifactIdentity,
)
from .plans import FrozenCompilationExecutionPlan
from .study_bundle_compiler import StudyBundleCandidate
from .study_issue import ExtensionPublicationReceipt, IssuedExecutableStudyManifest


PROGRAMME_BUNDLE_PUBLICATION_RELATIVE_ROOT = "issued-programme-bundles"


@dataclass(frozen=True, slots=True)
class IssuedMultiWorldStudyChild(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/issued-multi-world-study-child'

    child_id: str
    science: MultiWorldStudyChildScientificBinding
    manifest: IssuedExecutableStudyManifest
    publication_receipt: ExtensionPublicationReceipt

    def __post_init__(self) -> None:
        validate_stable_id(self.child_id, field_name="child_id")
        if self.science.child_id != self.child_id:
            raise ValueError("issued bundle child changes its scientific child identity")
        if self.science.base_candidate != ObjectIdentity.from_record(
            self.manifest.base.candidate.candidate_id,
            self.manifest.base.candidate,
        ):
            raise ValueError("issued bundle child changes its base candidate")
        if self.publication_receipt.issue_manifest != ObjectIdentity.from_record(
            self.manifest.issue_id,
            self.manifest,
        ):
            raise ValueError("issued bundle child publication binds another manifest")


@dataclass(frozen=True, slots=True)
class IssuedMultiWorldJointDescendant(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/issued-multi-world-joint-descendant'

    descendant_id: str
    bundle_candidate: ObjectIdentity
    descendant_node_id: str
    template: ObjectIdentity
    registry: ObjectIdentity
    graph_sha256: str
    expected_authority_gate_ids: tuple[str, ...]
    issued: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.descendant_id, field_name="descendant_id")
        validate_stable_id(self.descendant_node_id, field_name="descendant_node_id")
        require_sorted_unique_strings(
            self.expected_authority_gate_ids,
            field_name="expected_authority_gate_ids",
            allow_empty=False,
        )
        validate_sha256(self.graph_sha256, field_name="graph_sha256")
        if self.bundle_candidate.object_schema != StudyBundleCandidate.SCHEMA:
            raise ValueError("issued joint descendant binds another bundle candidate schema")
        if not self.issued or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("joint descendant issue crossed its outcome-blind boundary")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyIssueManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-study-issue-manifest'

    issue_id: str
    candidate: StudyBundleCandidate
    children: tuple[IssuedMultiWorldStudyChild, ...]
    joint_descendant: IssuedMultiWorldJointDescendant
    archive_outcome_protection: ArchiveOutcomeProtectionPlan
    outcome_barriers: MultiWorldOutcomeBarrierPlan
    partial_morphism: ArchiveToSimulatorPartialMorphismSpec
    joint_adjudication: MultiWorldJointAdjudicationPlan
    custody_authority: ObjectIdentity
    storage_root_id: str
    publication_relative_root: str
    issued_at_utc: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    created_empirical_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        validate_stable_id(self.storage_root_id, field_name="storage_root_id")
        validate_relative_locator(self.publication_relative_root)
        parse_utc_timestamp(self.issued_at_utc, field_name="issued_at_utc")
        require_sorted_unique_ids(self.children, attribute="child_id", field_name="children")
        if len(self.children) != 3 or {value.science.role for value in self.children} != set(
            MultiWorldChildRole
        ):
            raise ValueError("issued flagship bundle requires the exact three child roles")
        candidate_children = {value.child_id: value for value in self.candidate.children}
        for child in self.children:
            candidate_child = candidate_children.get(child.child_id)
            if candidate_child is None or candidate_child.candidate != child.science.base_candidate:
                raise ValueError("issued flagship child differs from the compiled bundle")
        if set(candidate_children) != {value.child_id for value in self.children}:
            raise ValueError("issued flagship bundle changes the compiled child roster")
        candidate_identity = ObjectIdentity.from_record(
            self.candidate.candidate_id,
            self.candidate,
        )
        if (
            self.joint_descendant.bundle_candidate != candidate_identity
            or self.joint_descendant.template != self.candidate.joint_template
            or self.joint_descendant.registry != self.candidate.joint_registry
            or self.joint_descendant.graph_sha256 != self.candidate.joint_graph_sha256
            or self.joint_descendant.expected_authority_gate_ids
            != self.candidate.expected_authority_gate_ids
        ):
            raise ValueError("issued joint descendant differs from the compiled fan-in")
        roles = {value.science.role: value for value in self.children}
        archive = roles[MultiWorldChildRole.FAIR_MAST_ARCHIVE]
        mapped = roles[MultiWorldChildRole.MAPPED_DIRECT_TORAX]
        generated = roles[MultiWorldChildRole.GENERATED_GYM_TORAX]
        if (
            self.archive_outcome_protection.archive_child_id != archive.child_id
            or self.outcome_barriers.archive_child_id != archive.child_id
            or set(self.outcome_barriers.simulator_child_ids)
            != {mapped.child_id, generated.child_id}
            or self.partial_morphism.archive_child_id != archive.child_id
            or self.partial_morphism.simulator_child_id != mapped.child_id
            or (
                self.joint_adjudication.archive_child_id,
                self.joint_adjudication.mapped_child_id,
                self.joint_adjudication.generated_child_id,
            )
            != (archive.child_id, mapped.child_id, generated.child_id)
        ):
            raise ValueError("issued flagship scientific plans bind another child topology")
        barrier_by_child = {value.child_id: value for value in self.outcome_barriers.barriers}
        if any(
            barrier_by_child[value.child_id].authority_subject
            != value.science.reveal_authority_subject
            for value in self.children
        ):
            raise ValueError("issued flagship reveal authority subjects differ from barriers")
        science = tuple(value.science for value in self.children)
        for attribute in (
            "base_candidate",
            "independent_unit_roster",
            "evidence_world",
            "law",
            "protected_outcome_domain_id",
            "execution_authority_subject",
            "reveal_authority_subject",
        ):
            if len({getattr(value, attribute) for value in science}) != 3:
                raise ValueError(f"issued flagship child {attribute} identities collapsed")
        if self.custody_authority.object_schema != StudyOperationAuthority.SCHEMA:
            raise ValueError("issued flagship bundle lacks typed custody authority")
        if (
            self.publication_relative_root != PROGRAMME_BUNDLE_PUBLICATION_RELATIVE_ROOT
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or self.created_empirical_evidence
        ):
            raise ValueError("issued flagship bundle crosses its prospective boundary")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyPublicationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-study-publication-receipt'

    receipt_id: str
    issue_manifest: ObjectIdentity
    child_publication_receipts: tuple[ObjectIdentity, ...]
    manifest_logical: LogicalArtifactIdentity
    manifest_materialization: ArtifactMaterialization
    publication: ArtifactPublicationBinding
    manifest_payload_bytes: int
    published_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        parse_utc_timestamp(self.published_at_utc, field_name="published_at_utc")
        if self.issue_manifest.object_schema != MultiWorldStudyIssueManifest.SCHEMA:
            raise ValueError("bundle publication receipt binds another manifest schema")
        require_sorted_unique_ids(
            self.child_publication_receipts,
            attribute="object_id",
            field_name="child_publication_receipts",
        )
        if len(self.child_publication_receipts) != 3 or any(
            value.object_schema != ExtensionPublicationReceipt.SCHEMA
            for value in self.child_publication_receipts
        ):
            raise ValueError("bundle publication receipt changes child custody")
        if self.manifest_payload_bytes <= 0:
            raise ValueError("bundle publication manifest payload must be nonempty")
        if (
            self.manifest_logical.logical_artifact_id
            != self.manifest_materialization.logical_artifact_id
            or tuple(value.logical_artifact_id for value in self.publication.members)
            != (self.manifest_logical.logical_artifact_id,)
        ):
            raise ValueError("bundle manifest publication identity differs")


@dataclass(frozen=True, slots=True)
class IssuedMultiWorldStudy(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/issued-multi-world-study'

    bundle_id: str
    manifest: MultiWorldStudyIssueManifest
    publication_receipt: MultiWorldStudyPublicationReceipt

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        if self.publication_receipt.issue_manifest != ObjectIdentity.from_record(
            self.manifest.issue_id,
            self.manifest,
        ):
            raise ValueError("issued bundle publication binds another manifest")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyIssuePreparation:
    manifest: MultiWorldStudyIssueManifest
    custody_authority: StudyOperationAuthority

    def __post_init__(self) -> None:
        if self.manifest.custody_authority != ObjectIdentity.from_record(
            self.custody_authority.authority_id,
            self.custody_authority,
        ):
            raise ValueError("bundle issue preparation changes custody authority")


def prepare_study_bundle_issue(
    *,
    candidate: StudyBundleCandidate,
    children: tuple[IssuedMultiWorldStudyChild, ...],
    archive_outcome_protection: ArchiveOutcomeProtectionPlan,
    outcome_barriers: MultiWorldOutcomeBarrierPlan,
    partial_morphism: ArchiveToSimulatorPartialMorphismSpec,
    joint_adjudication: MultiWorldJointAdjudicationPlan,
    custody_authority: StudyOperationAuthority,
    storage_root_id: str,
    grantee_id: str,
    at_utc: str,
) -> MultiWorldStudyIssuePreparation:
    candidate_identity = ObjectIdentity.from_record(candidate.candidate_id, candidate)
    require_study_authority(
        custody_authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=candidate_identity,
        prerequisite_authority=None,
        grantee_id=grantee_id,
        storage_root_id=storage_root_id,
        relative_root=PROGRAMME_BUNDLE_PUBLICATION_RELATIVE_ROOT,
        at_utc=at_utc,
    )
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "candidate": candidate_identity,
                "children": tuple(
                    ObjectIdentity.from_record(value.child_id, value)
                    for value in sorted(children, key=lambda value: value.child_id)
                ),
                "archive_outcome_protection": ObjectIdentity.from_record(
                    archive_outcome_protection.plan_id,
                    archive_outcome_protection,
                ),
                "outcome_barriers": ObjectIdentity.from_record(
                    outcome_barriers.plan_id,
                    outcome_barriers,
                ),
                "partial_morphism": ObjectIdentity.from_record(
                    partial_morphism.morphism_id,
                    partial_morphism,
                ),
                "joint_adjudication": ObjectIdentity.from_record(
                    joint_adjudication.plan_id,
                    joint_adjudication,
                ),
                "custody_authority": ObjectIdentity.from_record(
                    custody_authority.authority_id,
                    custody_authority,
                ),
                "storage_root_id": storage_root_id,
            }
        )
    ).hexdigest()
    issue_id = f"bundle-issue.{digest[:32]}"
    consumer_node_ids = {value.consumer_node_id for value in candidate.receipt_requirements}
    if len(consumer_node_ids) != 1:
        raise ValueError("compiled bundle does not have one deterministic joint descendant")
    joint = IssuedMultiWorldJointDescendant(
        descendant_id=f"joint.{issue_id}",
        bundle_candidate=candidate_identity,
        descendant_node_id=next(iter(consumer_node_ids)),
        template=candidate.joint_template,
        registry=candidate.joint_registry,
        graph_sha256=candidate.joint_graph_sha256,
        expected_authority_gate_ids=candidate.expected_authority_gate_ids,
        issued=True,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    manifest = MultiWorldStudyIssueManifest(
        issue_id=issue_id,
        candidate=candidate,
        children=tuple(sorted(children, key=lambda value: value.child_id)),
        joint_descendant=joint,
        archive_outcome_protection=archive_outcome_protection,
        outcome_barriers=outcome_barriers,
        partial_morphism=partial_morphism,
        joint_adjudication=joint_adjudication,
        custody_authority=ObjectIdentity.from_record(
            custody_authority.authority_id,
            custody_authority,
        ),
        storage_root_id=storage_root_id,
        publication_relative_root=PROGRAMME_BUNDLE_PUBLICATION_RELATIVE_ROOT,
        issued_at_utc=at_utc,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        created_empirical_evidence=False,
    )
    return MultiWorldStudyIssuePreparation(
        manifest=manifest,
        custody_authority=custody_authority,
    )


@dataclass(frozen=True, slots=True)
class MultiWorldStudyChildExecutionBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-study-child-execution-binding'

    child_id: str
    role: MultiWorldChildRole
    issue_manifest: ObjectIdentity
    execution_plan: FrozenCompilationExecutionPlan
    independent_unit_roster: ObjectIdentity
    outcome_domain_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.child_id, field_name="child_id")
        validate_stable_id(self.outcome_domain_id, field_name="outcome_domain_id")
        if self.issue_manifest.object_schema != IssuedExecutableStudyManifest.SCHEMA:
            raise ValueError("bundle child execution binds another issue schema")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyExecutionPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-study-execution-plan'

    plan_id: str
    issued_bundle: ObjectIdentity
    children: tuple[MultiWorldStudyChildExecutionBinding, ...]
    joint_descendant: ObjectIdentity
    outcome_barrier_plan: MultiWorldOutcomeBarrierPlan
    stage_order: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        if self.issued_bundle.object_schema != IssuedMultiWorldStudy.SCHEMA:
            raise ValueError("bundle execution plan binds another issued-bundle schema")
        require_sorted_unique_ids(self.children, attribute="child_id", field_name="children")
        if len(self.children) != 3 or {value.role for value in self.children} != set(
            MultiWorldChildRole
        ):
            raise ValueError("bundle execution plan changes the three-child roster")
        if len({value.independent_unit_roster for value in self.children}) != 3:
            raise ValueError("bundle execution plan collapses independent-unit rosters")
        if len({value.outcome_domain_id for value in self.children}) != 3:
            raise ValueError("bundle execution plan collapses outcome domains")
        expected_order = (
            "STAGE_ARCHIVE_PREACTION_AND_ACTION_TRACE",
            "EXECUTE_AND_REVEAL_SIMULATOR_CHILDREN",
            "REVEAL_ARCHIVE_RECEIVER",
            "JOINT_FANIN",
        )
        if (
            self.stage_order != expected_order
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("bundle execution plan changes staged outcome access")


class MultiWorldBarrierEventKind(StrEnum):
    DOMAIN_SEALED = "DOMAIN_SEALED"
    DOMAIN_REVEALED = "DOMAIN_REVEALED"
    CHILD_RESULT_BOUND = "CHILD_RESULT_BOUND"


@dataclass(frozen=True, slots=True)
class MultiWorldOutcomeBarrierEvent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-outcome-barrier-event'

    event_id: str
    sequence_index: int
    barrier_id: str
    domain: MultiWorldOutcomeDomain
    kind: MultiWorldBarrierEventKind
    authority: ObjectIdentity | None
    child_result: ObjectIdentity | None

    def __post_init__(self) -> None:
        validate_stable_id(self.event_id, field_name="event_id")
        validate_stable_id(self.barrier_id, field_name="barrier_id")
        if self.sequence_index < 0:
            raise ValueError("barrier event sequence index must be nonnegative")
        if self.kind is MultiWorldBarrierEventKind.DOMAIN_SEALED:
            if self.authority is not None or self.child_result is not None:
                raise ValueError("domain seal cannot reveal authority or result")
        elif self.kind is MultiWorldBarrierEventKind.DOMAIN_REVEALED:
            if (
                self.authority is None
                or self.authority.object_schema != StudyOperationAuthority.SCHEMA
                or self.child_result is not None
            ):
                raise ValueError("domain reveal lacks its exact authority")
        elif self.child_result is None or self.authority is not None:
            raise ValueError("child-result binding has invalid authority/result fields")


@dataclass(frozen=True, slots=True)
class MultiWorldOutcomeBarrierPrefix(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-outcome-barrier-prefix'

    prefix_id: str
    plan: ObjectIdentity
    events: tuple[MultiWorldOutcomeBarrierEvent, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.prefix_id, field_name="prefix_id")
        if self.plan.object_schema != MultiWorldOutcomeBarrierPlan.SCHEMA:
            raise ValueError("multi-world event prefix binds another barrier plan")
        require_sorted_unique_ids(self.events, attribute="event_id", field_name="events")
        if tuple(value.sequence_index for value in self.events) != tuple(range(len(self.events))):
            raise ValueError("multi-world barrier prefix is not contiguous")


def _build_barrier_prefix(
    *,
    plan: ObjectIdentity,
    events: tuple[MultiWorldOutcomeBarrierEvent, ...],
) -> MultiWorldOutcomeBarrierPrefix:
    digest = hashlib.sha256(canonical_json_bytes({"plan": plan, "events": events})).hexdigest()
    return MultiWorldOutcomeBarrierPrefix(
        prefix_id=f"barrier-prefix.{len(events):02d}.{digest[:24]}",
        plan=plan,
        events=events,
    )


@dataclass(frozen=True, slots=True)
class MultiWorldAuthorityRequired(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-authority-required'

    requirement_id: str
    prefix: ObjectIdentity
    barrier_id: str
    domain: MultiWorldOutcomeDomain
    authority_subject: ObjectIdentity
    reason_code: str

    def __post_init__(self) -> None:
        validate_stable_id(self.requirement_id, field_name="requirement_id")
        validate_stable_id(self.barrier_id, field_name="barrier_id")
        if self.prefix.object_schema != MultiWorldOutcomeBarrierPrefix.SCHEMA:
            raise ValueError("authority-required record binds another event prefix")
        if self.reason_code != "AUTHORITY_REQUIRED":
            raise ValueError("multi-world reveal pause has another reason code")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyBarrierStatus(CanonicalRecord):
    """Read-only status derived solely from one immutable barrier prefix."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-study-barrier-status'

    status_id: str
    plan: ObjectIdentity
    prefix: ObjectIdentity
    revealed_barrier_ids: tuple[str, ...]
    result_bound_barrier_ids: tuple[str, ...]
    eligible_reveal_barrier_ids: tuple[str, ...]
    terminal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.status_id, field_name="status_id")
        if self.plan.object_schema != MultiWorldOutcomeBarrierPlan.SCHEMA:
            raise ValueError("bundle barrier status binds another plan schema")
        if self.prefix.object_schema != MultiWorldOutcomeBarrierPrefix.SCHEMA:
            raise ValueError("bundle barrier status binds another prefix schema")
        for name, values in (
            ("revealed_barrier_ids", self.revealed_barrier_ids),
            ("result_bound_barrier_ids", self.result_bound_barrier_ids),
            ("eligible_reveal_barrier_ids", self.eligible_reveal_barrier_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.terminal != (len(self.result_bound_barrier_ids) == 3):
            raise ValueError("bundle barrier terminal status differs from bound results")


def inspect_study_bundle_barrier_status(
    *,
    plan: MultiWorldOutcomeBarrierPlan,
    prefix: MultiWorldOutcomeBarrierPrefix,
) -> MultiWorldStudyBarrierStatus:
    if prefix.plan != ObjectIdentity.from_record(plan.plan_id, plan):
        raise ValueError("bundle barrier status prefix changed its plan")
    revealed = {
        value.barrier_id
        for value in prefix.events
        if value.kind is MultiWorldBarrierEventKind.DOMAIN_REVEALED
    }
    results = {
        value.barrier_id
        for value in prefix.events
        if value.kind is MultiWorldBarrierEventKind.CHILD_RESULT_BOUND
    }
    eligible = {
        value.barrier_id
        for value in plan.barriers
        if value.barrier_id not in revealed and set(value.predecessor_barrier_ids).issubset(results)
    }
    return MultiWorldStudyBarrierStatus(
        status_id=f"bundle-barrier-status.{prefix.prefix_id}",
        plan=prefix.plan,
        prefix=ObjectIdentity.from_record(prefix.prefix_id, prefix),
        revealed_barrier_ids=tuple(sorted(revealed)),
        result_bound_barrier_ids=tuple(sorted(results)),
        eligible_reveal_barrier_ids=tuple(sorted(eligible)),
        terminal=len(results) == len(plan.barriers),
    )


def build_multi_world_sealed_prefix(
    plan: MultiWorldOutcomeBarrierPlan,
) -> MultiWorldOutcomeBarrierPrefix:
    events = tuple(
        MultiWorldOutcomeBarrierEvent(
            event_id=f"barrier-event.{index:02d}.{barrier.barrier_id}.sealed",
            sequence_index=index,
            barrier_id=barrier.barrier_id,
            domain=barrier.domain,
            kind=MultiWorldBarrierEventKind.DOMAIN_SEALED,
            authority=None,
            child_result=None,
        )
        for index, barrier in enumerate(plan.barriers)
    )
    return _build_barrier_prefix(
        plan=ObjectIdentity.from_record(plan.plan_id, plan),
        events=events,
    )


def request_multi_world_reveal(
    *,
    plan: MultiWorldOutcomeBarrierPlan,
    prefix: MultiWorldOutcomeBarrierPrefix,
    barrier_id: str,
    authority: StudyOperationAuthority | None,
    grantee_id: str,
    at_utc: str,
) -> MultiWorldOutcomeBarrierPrefix | MultiWorldAuthorityRequired:
    if prefix.plan != ObjectIdentity.from_record(plan.plan_id, plan):
        raise ValueError("multi-world reveal prefix changed its barrier plan")
    barriers = {value.barrier_id: value for value in plan.barriers}
    barrier = barriers.get(barrier_id)
    if barrier is None:
        raise ValueError("multi-world reveal names an unknown barrier")
    revealed = {
        value.barrier_id
        for value in prefix.events
        if value.kind is MultiWorldBarrierEventKind.DOMAIN_REVEALED
    }
    results = {
        value.barrier_id
        for value in prefix.events
        if value.kind is MultiWorldBarrierEventKind.CHILD_RESULT_BOUND
    }
    if barrier_id in revealed:
        raise ValueError("multi-world outcome domain cannot be revealed twice")
    if not set(barrier.predecessor_barrier_ids).issubset(results):
        raise ValueError("archive receiver reveal precedes terminal simulator child results")
    if authority is None:
        return MultiWorldAuthorityRequired(
            requirement_id=f"authority-required.{barrier.barrier_id}",
            prefix=ObjectIdentity.from_record(prefix.prefix_id, prefix),
            barrier_id=barrier.barrier_id,
            domain=barrier.domain,
            authority_subject=barrier.authority_subject,
            reason_code="AUTHORITY_REQUIRED",
        )
    require_study_authority(
        authority,
        kind=StudyAuthorityKind.OUTCOME_REVEAL,
        subject=barrier.authority_subject,
        prerequisite_authority=barrier.prerequisite_execution_authority,
        grantee_id=grantee_id,
        at_utc=at_utc,
    )
    if (
        authority.scope_id != barrier.reveal_scope_id
        or authority.grantee_id != barrier.reveal_grantee_id
        or authority.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
    ):
        raise PermissionError("multi-world reveal authority differs from the domain barrier")
    event = MultiWorldOutcomeBarrierEvent(
        event_id=f"barrier-event.{len(prefix.events):02d}.{barrier.barrier_id}.revealed",
        sequence_index=len(prefix.events),
        barrier_id=barrier.barrier_id,
        domain=barrier.domain,
        kind=MultiWorldBarrierEventKind.DOMAIN_REVEALED,
        authority=ObjectIdentity.from_record(authority.authority_id, authority),
        child_result=None,
    )
    return _build_barrier_prefix(
        plan=prefix.plan,
        events=(*prefix.events, event),
    )


def resolve_multi_world_reveal_authority(
    *,
    plan: MultiWorldOutcomeBarrierPlan,
    prefix: MultiWorldOutcomeBarrierPrefix,
    barrier_id: str,
    authority: StudyOperationAuthority | None,
    grantee_id: str,
    at_utc: str,
) -> MultiWorldOutcomeBarrierPrefix | MultiWorldAuthorityRequired:
    """Named public wrapper retained for orchestration/runbook discovery."""

    return request_multi_world_reveal(
        plan=plan,
        prefix=prefix,
        barrier_id=barrier_id,
        authority=authority,
        grantee_id=grantee_id,
        at_utc=at_utc,
    )


def bind_multi_world_child_result(
    *,
    plan: MultiWorldOutcomeBarrierPlan,
    prefix: MultiWorldOutcomeBarrierPrefix,
    barrier_id: str,
    child_result: ObjectIdentity,
) -> MultiWorldOutcomeBarrierPrefix:
    barrier = next((value for value in plan.barriers if value.barrier_id == barrier_id), None)
    if barrier is None or prefix.plan != ObjectIdentity.from_record(plan.plan_id, plan):
        raise ValueError("child result binds another multi-world barrier plan")
    events = tuple(value for value in prefix.events if value.barrier_id == barrier_id)
    if not any(value.kind is MultiWorldBarrierEventKind.DOMAIN_REVEALED for value in events):
        raise ValueError("child result cannot bind before its authorized reveal")
    if any(value.kind is MultiWorldBarrierEventKind.CHILD_RESULT_BOUND for value in events):
        raise ValueError("multi-world child result cannot be rebound")
    event = MultiWorldOutcomeBarrierEvent(
        event_id=f"barrier-event.{len(prefix.events):02d}.{barrier.barrier_id}.result",
        sequence_index=len(prefix.events),
        barrier_id=barrier.barrier_id,
        domain=barrier.domain,
        kind=MultiWorldBarrierEventKind.CHILD_RESULT_BOUND,
        authority=None,
        child_result=child_result,
    )
    return _build_barrier_prefix(
        plan=prefix.plan,
        events=(*prefix.events, event),
    )


@dataclass(frozen=True, slots=True)
class MultiWorldStudyChildRecoveryBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-study-child-recovery-binding'

    child_id: str
    execution_plan: ObjectIdentity
    recovery_index: ObjectIdentity
    child_result: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.child_id, field_name="child_id")
        if self.execution_plan.object_schema != FrozenCompilationExecutionPlan.SCHEMA:
            raise ValueError("bundle recovery child binds another execution-plan schema")
        if self.recovery_index.object_schema != 'empirical-lawhood/runtime/run-recovery-reference':
            raise ValueError("bundle recovery child binds another recovery-index schema")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyRecoveryIndex(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-study-recovery-index'

    recovery_id: str
    execution_plan: ObjectIdentity
    child_recoveries: tuple[MultiWorldStudyChildRecoveryBinding, ...]
    barrier_prefix: MultiWorldOutcomeBarrierPrefix
    joint_result: ObjectIdentity
    catalog_rebuild_complete: bool
    child_results_rebound_immutably: bool
    scientific_recomputation_count: int
    scientific_retuning_count: int
    terminal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.recovery_id, field_name="recovery_id")
        if self.execution_plan.object_schema != MultiWorldStudyExecutionPlan.SCHEMA:
            raise ValueError("bundle recovery binds another execution-plan schema")
        require_sorted_unique_ids(
            self.child_recoveries,
            attribute="child_id",
            field_name="child_recoveries",
        )
        result_events = tuple(
            value
            for value in self.barrier_prefix.events
            if value.kind is MultiWorldBarrierEventKind.CHILD_RESULT_BOUND
        )
        if (
            len(self.child_recoveries) != 3
            or len(result_events) != 3
            or {value.child_result for value in self.child_recoveries}
            != {value.child_result for value in result_events}
            or not self.catalog_rebuild_complete
            or not self.child_results_rebound_immutably
            or self.scientific_recomputation_count != 0
            or self.scientific_retuning_count != 0
            or not self.terminal
        ):
            raise ValueError("bundle recovery is incomplete or changed scientific evidence")


__all__ = [
    'IssuedMultiWorldStudyChild',
    'IssuedMultiWorldJointDescendant',
    'IssuedMultiWorldStudy',
    'MultiWorldAuthorityRequired',
    "MultiWorldBarrierEventKind",
    'MultiWorldOutcomeBarrierEvent',
    'MultiWorldOutcomeBarrierPrefix',
    "PROGRAMME_BUNDLE_PUBLICATION_RELATIVE_ROOT",
    'MultiWorldStudyChildExecutionBinding',
    'MultiWorldStudyChildRecoveryBinding',
    'MultiWorldStudyExecutionPlan',
    'MultiWorldStudyBarrierStatus',
    'MultiWorldStudyIssueManifest',
    'MultiWorldStudyIssuePreparation',
    'MultiWorldStudyPublicationReceipt',
    'MultiWorldStudyRecoveryIndex',
    "bind_multi_world_child_result",
    "build_multi_world_sealed_prefix",
    'inspect_study_bundle_barrier_status',
    'prepare_study_bundle_issue',
    "request_multi_world_reveal",
    "resolve_multi_world_reveal_authority",
]
