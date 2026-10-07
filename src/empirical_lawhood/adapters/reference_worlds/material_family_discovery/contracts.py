"""Strict source and experiment contracts for superconductor first discovery."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION = '1.0.0'
UNPARTITIONED_CANDIDATE_ALGORITHM_ID = 'candidate.nims-formula-digest-unpartitioned'
PHASE_DISJOINT_CANDIDATE_ALGORITHM_ID = 'candidate.nims-formula-digest-phase-disjoint-three-quarter-development'


class MaterialFamilyDiscoveryPhase(StrEnum):
    REPRODUCTION = "REPRODUCTION"
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


class ReproductionDisposition(StrEnum):
    EXACT_REPRODUCIBLE = "EXACT_REPRODUCIBLE"
    SOURCE_OPERAND_REQUIRED_SC_EXACT = "SOURCE_OPERAND_REQUIRED_SC_EXACT"


@dataclass(frozen=True, slots=True)
class MaterialSourceObject(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/material-family-discovery/material-source-object'

    object_id: str
    relative_locator: str
    size_bytes: int
    sha256: str
    media_type: str
    encoding: str
    role_id: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("object_id", "role_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            not self.relative_locator
            or "/" in self.relative_locator
            or "\\" in self.relative_locator
            or self.relative_locator in {".", ".."}
            or "\x00" in self.relative_locator
        ):
            raise ValueError("relative_locator must be one plain filename")
        if isinstance(self.size_bytes, bool) or not 0 < self.size_bytes <= 128 * 1024**2:
            raise ValueError("source object byte bound differs")
        validate_sha256(self.sha256, field_name="sha256")
        if self.media_type not in {"application/json", "application/yaml", "text/plain"}:
            raise ValueError("source media type is unsupported")
        if self.encoding not in {"ascii", "cp932", "utf-8"}:
            raise ValueError("source encoding is unsupported")


@dataclass(frozen=True, slots=True)
class MaterialSourceManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/material-family-discovery/material-source-manifest'

    manifest_id: str
    source_id: str
    publisher_id: str
    version_id: str
    doi: str
    official_locator: str
    retrieved_at_utc: str
    license_id: str
    license_locator: str
    redistribution_class_id: str
    credential_required: bool
    terms_acceptance_required: bool
    source_root_id: str
    objects: tuple[MaterialSourceObject, ...]

    def __post_init__(self) -> None:
        for name in (
            "manifest_id",
            "source_id",
            "publisher_id",
            "version_id",
            "license_id",
            "redistribution_class_id",
            "source_root_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.doi != "10.48505/nims.3837":
            raise ValueError("SC source manifest binds the wrong dated DOI")
        if not self.official_locator.startswith("https://mdr.nims.go.jp/"):
            raise ValueError("SC source manifest requires the official NIMS locator")
        if self.license_id != "cc-by-4.0" or not self.license_locator.startswith(
            "https://creativecommons.org/licenses/by/4.0"
        ):
            raise ValueError("SC source manifest license differs")
        if self.credential_required or self.terms_acceptance_required:
            raise ValueError("public NIMS v220808 source must not infer credentialed access")
        object_ids = tuple(value.object_id for value in self.objects)
        if object_ids != tuple(sorted(set(object_ids))) or len(self.objects) != 4:
            raise ValueError("SC source objects must be the exact sorted four-object roster")
        roles = {value.role_id for value in self.objects}
        if roles != {"data.nims-organic", "data.nims-o-and-m", "schema.json", "schema.yaml"}:
            raise ValueError("SC source object roles differ")


@dataclass(frozen=True, slots=True)
class SourceQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/material-family-discovery/source-qualification'

    qualification_id: str
    source_manifest_sha256: str
    qualified_object_ids: tuple[str, ...]
    o_and_m_row_count: int
    organic_row_count: int
    o_and_m_column_count: int
    resolved_tc_row_count: int
    detection_limit_negative_row_count: int
    unlabelled_row_count: int
    above_ceiling_row_count: int
    canonical_candidate_count: int
    ambiguous_family_candidate_count: int
    exact_reproduction_disposition: ReproductionDisposition
    missing_exact_operand_ids: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_sha256(self.source_manifest_sha256, field_name="source_manifest_sha256")
        if self.qualified_object_ids != tuple(sorted(set(self.qualified_object_ids))):
            raise ValueError("qualified object IDs must be sorted and unique")
        for name in (
            "o_and_m_row_count",
            "organic_row_count",
            "o_and_m_column_count",
            "resolved_tc_row_count",
            "detection_limit_negative_row_count",
            "unlabelled_row_count",
            "above_ceiling_row_count",
            "canonical_candidate_count",
            "ambiguous_family_candidate_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.o_and_m_column_count != 191:
            raise ValueError("NIMS O&M schema width differs")
        if self.exact_reproduction_disposition is ReproductionDisposition.EXACT_REPRODUCIBLE:
            if self.missing_exact_operand_ids:
                raise ValueError("exact reproduction cannot omit operands")
        elif self.missing_exact_operand_ids != tuple(sorted(set(self.missing_exact_operand_ids))):
            raise ValueError("missing exact operands must be sorted and unique")


@dataclass(frozen=True, slots=True)
class MaterialFamilyConfig(CanonicalRecord):
    """Frozen reconstruction/evaluation design; no source path or callable injection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/material-family-discovery/material-family-config'

    config_id: str
    plan_id: str
    phase: MaterialFamilyDiscoveryPhase
    source_manifest_sha256: str
    source_qualification_sha256: str
    family_ontology_id: str
    feature_schema_id: str
    canonical_candidate_algorithm_id: str
    duplicate_policy_id: str
    tc_resolution_policy_id: str
    minimum_family_candidates: int
    minimum_family_positive_candidates: int
    target_tc_threshold_kelvin: Decimal
    target_candidates_per_world: int
    candidate_pool_size: int
    measured_candidate_test_modulus: int
    measured_candidate_test_remainder: int
    world_seed: int
    development_family_ids: tuple[str, ...]
    evaluation_family_ids: tuple[str, ...]
    policy_config_sha256s: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    requests_physical_execution: bool
    requests_controller: bool

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "plan_id",
            "family_ontology_id",
            "feature_schema_id",
            "canonical_candidate_algorithm_id",
            "duplicate_policy_id",
            "tc_resolution_policy_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.plan_id != 'material-family-discovery-benchmark':
            raise ValueError("SC config binds the wrong plan")
        if self.canonical_candidate_algorithm_id not in {
            UNPARTITIONED_CANDIDATE_ALGORITHM_ID,
            PHASE_DISJOINT_CANDIDATE_ALGORITHM_ID,
        }:
            raise ValueError("SC candidate/phase-partition algorithm is unknown")
        for name in ("source_manifest_sha256", "source_qualification_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        for value in self.policy_config_sha256s:
            validate_sha256(value, field_name="policy_config_sha256s")
        if self.policy_config_sha256s != tuple(sorted(set(self.policy_config_sha256s))):
            raise ValueError("policy config hashes must be sorted and unique")
        for name in (
            "minimum_family_candidates",
            "minimum_family_positive_candidates",
            "target_candidates_per_world",
            "candidate_pool_size",
            "measured_candidate_test_modulus",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.measured_candidate_test_remainder < self.measured_candidate_test_modulus:
            raise ValueError("measured candidate test remainder is invalid")
        if self.minimum_family_positive_candidates > self.minimum_family_candidates:
            raise ValueError("positive-family floor exceeds candidate floor")
        validate_decimal(
            self.target_tc_threshold_kelvin,
            field_name="target_tc_threshold_kelvin",
            minimum=Decimal(0),
        )
        if self.target_tc_threshold_kelvin != Decimal(5):
            raise ValueError("material-family discovery target threshold differs from the frozen 5 K rule")
        if self.target_candidates_per_world >= self.candidate_pool_size:
            raise ValueError("target roster must be rarer than the complete candidate pool")
        if isinstance(self.world_seed, bool) or self.world_seed < 0:
            raise ValueError("world_seed must be nonnegative")
        for name in ("development_family_ids", "evaluation_family_ids"):
            values = getattr(self, name)
            if values != tuple(sorted(set(values))):
                raise ValueError(f"{name} must be sorted and unique")
        if set(self.development_family_ids) & set(self.evaluation_family_ids):
            raise ValueError("development and evaluation family rosters overlap")
        expected_access = {
            MaterialFamilyDiscoveryPhase.REPRODUCTION: OutcomeAccess.EVALUATION_REVEALED,
            MaterialFamilyDiscoveryPhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            MaterialFamilyDiscoveryPhase.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.phase]
        if self.outcome_access is not expected_access:
            raise ValueError("SC phase outcome access differs")
        expected_visibility = {
            MaterialFamilyDiscoveryPhase.REPRODUCTION: VisibilityCeiling.OUTCOME_VISIBLE,
            MaterialFamilyDiscoveryPhase.DEVELOPMENT: VisibilityCeiling.DEVELOPMENT_ONLY,
            # SuperCon 220808 is an already outcome-visible historical source.
            # Evaluation labels remain sealed from policies, but descendants
            # cannot regain a prospective visibility ceiling from that lineage.
            MaterialFamilyDiscoveryPhase.EVALUATION: VisibilityCeiling.OUTCOME_VISIBLE,
        }[self.phase]
        if self.visibility_ceiling is not expected_visibility:
            raise ValueError("SC phase visibility ceiling differs")
        if self.requests_physical_execution or self.requests_controller:
            raise ValueError("SC historical-material benchmark is nonactuating and not admission/controller use")


@dataclass(frozen=True, slots=True)
class MaterialFamilyDiscoveryAdjudicationConfig(CanonicalRecord):
    """Exact family-level inference and B2 decision contract."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/reference-worlds/material-family-discovery/material-family-discovery-adjudication-config'

    config_id: str
    phase: MaterialFamilyDiscoveryPhase
    family_config_sha256: str
    method_policy_config_sha256: str
    primary_bo_policy_config_sha256: str
    query_budget: int
    minimum_independent_worlds: int
    confidence_level: Decimal
    bootstrap_resamples: int
    bootstrap_seed: int
    interval_method_id: str
    censoring_rule_id: str
    superiority_margin_queries: Decimal
    pareto_noninferiority_margin_queries: Decimal
    pareto_false_promotion_margin: Decimal

    def __post_init__(self) -> None:
        for name in ("config_id", "interval_method_id", "censoring_rule_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "family_config_sha256",
            "method_policy_config_sha256",
            "primary_bo_policy_config_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        for name in (
            "query_budget",
            "minimum_independent_worlds",
            "bootstrap_resamples",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.bootstrap_resamples < 1_000:
            raise ValueError("SC bootstrap requires at least 1,000 family resamples")
        if isinstance(self.bootstrap_seed, bool) or self.bootstrap_seed < 0:
            raise ValueError("SC bootstrap seed must be nonnegative")
        for name in (
            "confidence_level",
            "superiority_margin_queries",
            "pareto_noninferiority_margin_queries",
            "pareto_false_promotion_margin",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if not Decimal(0) < self.confidence_level < Decimal(1):
            raise ValueError("SC confidence level must lie inside (0, 1)")


__all__ = [
    'UNPARTITIONED_CANDIDATE_ALGORITHM_ID',
    "MaterialFamilyConfig",
    "MaterialSourceManifest",
    "MaterialSourceObject",
    "PHASE_DISJOINT_CANDIDATE_ALGORITHM_ID",
    "ReproductionDisposition",
    'MaterialFamilyDiscoveryPhase',
    'MaterialFamilyDiscoveryAdjudicationConfig',
    'MATERIAL_FAMILY_DISCOVERY_PROTOCOL_VERSION',
    "SourceQualification",
]
