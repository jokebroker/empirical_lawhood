"""Reusable contracts for outcome-visible, multi-partition post-hoc tranches.

The ordinary exploration package intentionally binds one system, world and
independent unit.  These records cover the different case where a frozen
portfolio enumerates heterogeneous evidence partitions without pretending
that they are one statistical population.  They add no evidence-world or
claim-promotion semantics to the kernel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.kernel.worlds import WorldKind


# Scientific purpose sequence; canonical metadata is sorted separately.
POSTHOC_ANALYSIS_IDS = (
    "analysis.coordinate-necessity-and-construct-diversity",
    "analysis.forecast-level-comparability",
    "analysis.noncompensating-gate-semantics",
    "analysis.obstruction-signatures",
    "analysis.power-design-robustness",
    "analysis.semantic-role-preservation",
    "analysis.complete-unit-localization",
    "analysis.independent-scientific-axes",
    "analysis.action-stage-observability",
)


class PosthocInferenceMode(StrEnum):
    WITHIN_PARTITION_UNIT = "WITHIN_PARTITION_UNIT"
    WITHIN_PARTITION_DESCRIPTIVE = "WITHIN_PARTITION_DESCRIPTIVE"
    CROSS_PARTITION_LOGICAL = "CROSS_PARTITION_LOGICAL"


class ParentEligibilityStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    PARTIALLY_ELIGIBLE = "PARTIALLY_ELIGIBLE"
    INELIGIBLE = "INELIGIBLE"


class PosthocAnalysisStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    MIXED = "MIXED"
    OPPOSED = "OPPOSED"
    NULL = "NULL"
    UNEVALUABLE = "UNEVALUABLE"
    NONATTEMPT = "NONATTEMPT"


@dataclass(frozen=True, slots=True)
class PosthocEvidencePartition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/posthoc-evidence-partition'

    partition_id: str
    study_id: str
    target_id: str
    world_id: str
    world_kind: WorldKind
    independent_unit_id: str
    source_artifact_ids: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    numerical_pooling_allowed: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("partition_id", self.partition_id),
            ('study_id', self.study_id),
            ("target_id", self.target_id),
            ("world_id", self.world_id),
            ("independent_unit_id", self.independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.source_artifact_ids,
            field_name="source_artifact_ids",
            allow_empty=False,
        )
        if self.visibility_ceiling.is_promotable:
            raise ValueError("post-hoc partitions must remain non-promotable")
        if self.numerical_pooling_allowed:
            raise ValueError("heterogeneous post-hoc partitions cannot authorize pooling")


@dataclass(frozen=True, slots=True)
class PosthocSourceArtifact(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/posthoc-source-artifact'

    artifact_id: str
    role_id: str
    relative_locator: str
    payload_schema: str | None
    payload_version: str | None
    identity_version: str | None
    version_normalization: str | None
    qualification_status: str
    content_sha256: str
    size_bytes: int
    custody_sha256: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.artifact_id, field_name="artifact_id")
        validate_stable_id(self.role_id, field_name="role_id")
        validate_nonempty(self.relative_locator, field_name="relative_locator")
        if self.relative_locator.startswith("/") or ".." in self.relative_locator.split("/"):
            raise ValueError("source locator must be external-root-relative")
        if self.qualification_status not in {"QUALIFIED", "UNSUPPORTED"}:
            raise ValueError("unsupported source-artifact qualification status")
        if self.qualification_status == "QUALIFIED":
            if self.payload_schema is None or self.payload_version is None:
                raise ValueError("qualified source requires a canonical schema and version")
            if self.identity_version is None:
                raise ValueError("qualified source requires an identity version")
            validate_schema(self.payload_schema)
            validate_nonempty(self.payload_version, field_name="payload_version")
            validate_semantic_version(self.identity_version)
            if self.version_normalization != "EXACT":
                raise ValueError("current source identity requires exact payload version custody")
            if self.identity_version != self.payload_version:
                raise ValueError("exact identity version must equal the payload version")
        elif self.identity_version is not None or self.version_normalization is not None:
            raise ValueError("unsupported source cannot acquire a canonical identity version")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        if self.size_bytes <= 0:
            raise ValueError("source artifact must be nonempty")
        if self.custody_sha256 is not None:
            validate_sha256(self.custody_sha256, field_name="custody_sha256")


@dataclass(frozen=True, slots=True)
class ParentEligibilityRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/parent-eligibility-record'

    parent_id: str
    lane_id: str
    partition_id: str | None
    controlling_artifact_id: str
    source_artifacts: tuple[PosthocSourceArtifact, ...]
    status: ParentEligibilityStatus
    available_role_ids: tuple[str, ...]
    analysis_ids: tuple[str, ...]
    independent_unit_count: int | None
    nested_observations_count_as_units: bool
    reason_codes: tuple[str, ...]
    claim_promotion_allowed: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("parent_id", self.parent_id),
            ("lane_id", self.lane_id),
            ("controlling_artifact_id", self.controlling_artifact_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.partition_id is not None:
            validate_stable_id(self.partition_id, field_name="partition_id")
        require_sorted_unique_ids(
            self.source_artifacts,
            attribute="artifact_id",
            field_name="source_artifacts",
        )
        if self.controlling_artifact_id not in {
            value.artifact_id for value in self.source_artifacts
        }:
            raise ValueError("controlling artifact is absent from source artifacts")
        for name, values in (
            ("available_role_ids", self.available_role_ids),
            ("analysis_ids", self.analysis_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.status is ParentEligibilityStatus.ELIGIBLE and not self.analysis_ids:
            raise ValueError("eligible parent requires at least one analysis")
        if self.status is not ParentEligibilityStatus.ELIGIBLE and not self.reason_codes:
            raise ValueError("partial/ineligible parent requires reasons")
        if self.independent_unit_count is not None and self.independent_unit_count < 0:
            raise ValueError("independent unit count must be nonnegative")
        if self.nested_observations_count_as_units:
            raise ValueError("nested observations cannot become independent units")
        if self.claim_promotion_allowed:
            raise ValueError("outcome-visible post-hoc eligibility cannot promote claims")


@dataclass(frozen=True, slots=True)
class PosthocAnalysisSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/posthoc-analysis-spec'

    analysis_id: str
    question: str
    inference_modes: tuple[PosthocInferenceMode, ...]
    required_role_ids: tuple[str, ...]
    eligible_parent_ids: tuple[str, ...]
    family_coordinate_ids: tuple[str, ...]
    typed_stop_codes: tuple[str, ...]
    decisive_counterexample_rule: str
    output_schema: str
    seed: int | None
    evidence_ceiling: EvidenceCeiling = EvidenceCeiling.NON_PROMOTABLE

    def __post_init__(self) -> None:
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        validate_nonempty(self.question, field_name="question")
        require_sorted_unique_strings(
            self.inference_modes,
            field_name="inference_modes",
            allow_empty=False,
        )
        for name, values in (
            ("required_role_ids", self.required_role_ids),
            ("eligible_parent_ids", self.eligible_parent_ids),
            ("family_coordinate_ids", self.family_coordinate_ids),
            ("typed_stop_codes", self.typed_stop_codes),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        validate_nonempty(
            self.decisive_counterexample_rule,
            field_name="decisive_counterexample_rule",
        )
        validate_schema(self.output_schema)
        if self.seed is not None and self.seed < 0:
            raise ValueError("analysis seed must be nonnegative")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("post-hoc analysis specifications are non-promotable")


@dataclass(frozen=True, slots=True)
class PosthocTrancheFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/posthoc-tranche-freeze'

    freeze_id: str
    plan_sha256: str
    implementation_commit: str
    implementation_sha256: str
    source_qualification: ObjectIdentity
    partitions: tuple[PosthocEvidencePartition, ...]
    parents: tuple[ParentEligibilityRecord, ...]
    analyses: tuple[PosthocAnalysisSpec, ...]
    synthesis_codebook_sha256: str
    frozen_at_utc: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    frozen: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        for name, value in (
            ("plan_sha256", self.plan_sha256),
            ("implementation_sha256", self.implementation_sha256),
            ("synthesis_codebook_sha256", self.synthesis_codebook_sha256),
        ):
            validate_sha256(value, field_name=name)
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit) is None:
            raise ValueError("implementation_commit must be a lowercase Git SHA-1")
        require_sorted_unique_ids(
            self.partitions,
            attribute="partition_id",
            field_name="partitions",
        )
        require_sorted_unique_ids(self.parents, attribute="parent_id", field_name="parents")
        require_sorted_unique_ids(
            self.analyses,
            attribute="analysis_id",
            field_name="analyses",
        )
        if tuple(value.analysis_id for value in self.analyses) != tuple(sorted(POSTHOC_ANALYSIS_IDS)):
            raise ValueError("tranche freeze must contain the complete nine-purpose analysis family")
        known_partitions = {value.partition_id for value in self.partitions}
        if any(
            parent.partition_id is not None and parent.partition_id not in known_partitions
            for parent in self.parents
        ):
            raise ValueError("parent names an unknown evidence partition")
        known_parents = {value.parent_id for value in self.parents}
        for analysis in self.analyses:
            if not set(analysis.eligible_parent_ids).issubset(known_parents):
                raise ValueError("analysis names an unknown parent")
        parse_utc_timestamp(self.frozen_at_utc, field_name="frozen_at_utc")
        if self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }:
            raise ValueError("post-hoc freeze must acknowledge outcome visibility")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("post-hoc freeze must remain non-promotable")
        if not self.frozen:
            raise ValueError("PosthocTrancheFreeze must be frozen")


@dataclass(frozen=True, slots=True)
class PosthocExecutionAuthorization(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/posthoc-execution-authorization'

    authorization_id: str
    freeze: ObjectIdentity
    requested_scope_id: str
    external_run_root: str
    requested_budget: ResourceBudget
    proposer_id: str
    approver_id: str
    authorized_at_utc: str
    implementation_commit: str
    passed_gate_ids: tuple[str, ...]
    plan_mutated: bool
    grants_claim_promotion: bool
    nonactuating: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("authorization_id", self.authorization_id),
            ("requested_scope_id", self.requested_scope_id),
            ("proposer_id", self.proposer_id),
            ("approver_id", self.approver_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.external_run_root, field_name="external_run_root")
        if self.external_run_root.startswith("/") or ".." in self.external_run_root.split("/"):
            raise ValueError("external run root must be an external-root-relative locator")
        require_sorted_unique_strings(
            self.passed_gate_ids,
            field_name="passed_gate_ids",
            allow_empty=False,
        )
        parse_utc_timestamp(self.authorized_at_utc, field_name="authorized_at_utc")
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit) is None:
            raise ValueError("implementation_commit must be a lowercase Git SHA-1")
        if self.proposer_id == self.approver_id:
            raise ValueError("post-hoc execution cannot self-approve")
        if self.plan_mutated or self.grants_claim_promotion or not self.nonactuating:
            raise ValueError("authorization must be nonmutating, nonpromoting and nonactuating")


@dataclass(frozen=True, slots=True)
class PosthocPortfolioPackage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/posthoc-portfolio-package'

    package_id: str
    freeze: PosthocTrancheFreeze
    authorization: PosthocExecutionAuthorization
    execution_plan: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.package_id, field_name="package_id")
        expected = ObjectIdentity.from_record(self.freeze.freeze_id, self.freeze)
        if self.authorization.freeze != expected:
            raise ValueError("execution authorization binds another freeze")


__all__ = [
    "POSTHOC_ANALYSIS_IDS",
    "ParentEligibilityRecord",
    "ParentEligibilityStatus",
    "PosthocAnalysisSpec",
    "PosthocAnalysisStatus",
    "PosthocEvidencePartition",
    "PosthocExecutionAuthorization",
    "PosthocInferenceMode",
    "PosthocPortfolioPackage",
    "PosthocSourceArtifact",
    "PosthocTrancheFreeze",
]
