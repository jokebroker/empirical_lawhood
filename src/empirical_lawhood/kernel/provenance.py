"""Machine-auditable object, evidence, claim and snapshot lineage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .evidence import (
    ClaimSpec,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from .references import ArtifactIdentity
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from .status import LifecycleStatus, ScientificStatus
from .time import InformationCutoff


@dataclass(frozen=True, slots=True)
class ObjectIdentity(CanonicalRecord):
    """Stable semantic and content identity of another immutable IR object."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/object-identity'

    object_id: str
    object_schema: str
    object_version: str
    object_fingerprint: str

    def __post_init__(self) -> None:
        validate_stable_id(self.object_id, field_name="object_id")
        validate_schema(self.object_schema)
        validate_semantic_version(self.object_version)
        validate_sha256(self.object_fingerprint, field_name="object_fingerprint")

    @classmethod
    def from_record(cls, object_id: str, value: CanonicalRecord) -> ObjectIdentity:
        return cls(
            object_id=object_id,
            object_schema=value.SCHEMA,
            object_version=value.VERSION,
            object_fingerprint=value.fingerprint(),
        )


class EvidenceRelation(StrEnum):
    SUPPORTS = "SUPPORTS"
    OPPOSES = "OPPOSES"
    LIMITS = "LIMITS"
    DERIVED_FROM = "DERIVED_FROM"
    SUPERSEDES = "SUPERSEDES"
    INVALIDATES = "INVALIDATES"
    CALIBRATES = "CALIBRATES"
    VALIDATES = "VALIDATES"


@dataclass(frozen=True, slots=True)
class EvidenceLink(CanonicalRecord):
    """Typed relation between immutable objects and supporting artifacts."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/evidence-link'

    link_id: str
    relation: EvidenceRelation
    source: ObjectIdentity
    target: ObjectIdentity
    artifact_ids: tuple[str, ...]
    world_id: str
    information_cutoff_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    reason: str
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.link_id, field_name="link_id")
        validate_stable_id(self.world_id, field_name="world_id")
        validate_stable_id(self.information_cutoff_id, field_name="information_cutoff_id")
        require_sorted_unique_strings(self.artifact_ids, field_name="artifact_ids")
        validate_nonempty(self.reason, field_name="reason")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("evidence-link visibility cannot be lowered")
        require_extensions(self.extensions)


@dataclass(frozen=True, slots=True)
class Claim(CanonicalRecord):
    """One graded scientific claim and its exact evidence links."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/claim'

    claim: ClaimSpec
    scientific_status: ScientificStatus
    observed_rung: EvidenceRung | None
    result_summary: str
    evidence_links: tuple[EvidenceLink, ...]
    reason_codes: tuple[str, ...]
    predecessor_claim_ids: tuple[str, ...] = ()
    superseded_by_claim_id: str | None = None
    lifecycle_status: LifecycleStatus = LifecycleStatus.ACTIVE
    extensions: tuple[ExtensionBinding, ...] = ()

    @property
    def claim_id(self) -> str:
        return self.claim.claim_id

    def __post_init__(self) -> None:
        validate_nonempty(self.result_summary, field_name="result_summary")
        require_sorted_unique_ids(
            self.evidence_links, attribute="link_id", field_name="evidence_links"
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.predecessor_claim_ids, field_name="predecessor_claim_ids"
        )
        if self.superseded_by_claim_id is not None:
            validate_stable_id(self.superseded_by_claim_id, field_name="superseded_by_claim_id")
        if self.observed_rung is not None:
            if not self.claim.visibility_ceiling.is_promotable:
                raise ValueError("a non-promotable claim cannot record a measurement through controller use rung")
            if not self.claim.evidence_ceiling.allows(self.observed_rung):
                raise ValueError("observed evidence rung exceeds the claim ceiling")
        if self.scientific_status is ScientificStatus.SUPPORTED:
            if self.observed_rung is None or not self.evidence_links:
                raise ValueError("a supported claim requires a rung and evidence links")
        if self.scientific_status is ScientificStatus.NOT_TESTED:
            if self.observed_rung is not None:
                raise ValueError("an untested claim cannot have an observed rung")
        if self.lifecycle_status is LifecycleStatus.SUPERSEDED:
            if self.superseded_by_claim_id is None:
                raise ValueError("a superseded claim must name its replacement claim")
        elif self.superseded_by_claim_id is not None:
            raise ValueError("only a superseded claim may name a replacement claim")
        require_extensions(self.extensions)


@dataclass(frozen=True, slots=True)
class EvidenceSnapshot(CanonicalRecord):
    """Immutable read-only evidence view for one analysis/exploration wave."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/evidence-snapshot'

    snapshot_id: str
    campaign_id: str
    run_id: str
    world_id: str
    parent_objects: tuple[ObjectIdentity, ...]
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    artifacts: tuple[ArtifactIdentity, ...]
    claim_ids: tuple[str, ...]
    information_cutoff: InformationCutoff
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    projection_ids: tuple[str, ...]
    read_only: bool = True
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("snapshot_id", self.snapshot_id),
            ("campaign_id", self.campaign_id),
            ("run_id", self.run_id),
            ("world_id", self.world_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.parent_objects, attribute="object_id", field_name="parent_objects"
        )
        if len(self.parent_objects) != len(self.parent_visibility_ceilings):
            raise ValueError("each snapshot parent requires a visibility ceiling")
        require_sorted_unique_ids(self.artifacts, attribute="artifact_id", field_name="artifacts")
        require_sorted_unique_strings(self.claim_ids, field_name="claim_ids")
        require_sorted_unique_strings(
            self.projection_ids, field_name="projection_ids", allow_empty=False
        )
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("snapshot visibility cannot be lowered")
        if not self.read_only:
            raise ValueError("an EvidenceSnapshot must be read-only")
        require_extensions(self.extensions)

    def artifact(self, artifact_id: str) -> ArtifactIdentity:
        validate_stable_id(artifact_id, field_name="artifact_id")
        for artifact in self.artifacts:
            if artifact.artifact_id == artifact_id:
                return artifact
        raise KeyError(artifact_id)
