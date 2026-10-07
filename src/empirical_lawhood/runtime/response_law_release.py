"""Canonical closeout and exact downstream pin for response-law consolidation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind
from empirical_lawhood.runtime.route_conformance import RunEnvelopeRouteComposition, RunEnvelopeRouteConformanceReceipt, RunEnvelopeConformanceSuiteReceipt, RunEnvelopeOperationalConformanceReceipt, PublicRouteStageConfiguration


class ConsolidationVerificationKind(StrEnum):
    FOCUSED_TESTS = "FOCUSED_TESTS"
    LINT = "LINT"
    TYPE_CHECK = "TYPE_CHECK"
    DOCUMENTATION = "DOCUMENTATION"
    LOCAL_VERIFICATION = "LOCAL_VERIFICATION"


class ConsolidationAuthorityKind(StrEnum):
    SOURCE = "SOURCE"
    STORAGE = "STORAGE"
    ISSUE = "ISSUE"
    RUN = "RUN"
    REVEAL = "REVEAL"
    ACTUATION = "ACTUATION"
    PUBLICATION = "PUBLICATION"
    MERGE = "MERGE"


class ConsolidationClaimCeiling(StrEnum):
    ARCHITECTURE_CONFORMANCE_ONLY = "ARCHITECTURE_CONFORMANCE_ONLY"


REQUIRED_RELEASE_CURRENT_SCHEMAS = (
    'empirical-lawhood/api/envelope-experiment-package',
    'empirical-lawhood/kernel/law-qualification-result',
    'empirical-lawhood/kernel/model-set-spec',
    'empirical-lawhood/planning/admission-controller-study',
    'empirical-lawhood/planning/receipt-admission-spec',
    'empirical-lawhood/planning/controlled-map-reachability-spec',
    'empirical-lawhood/planning/admission-coordinate-gate-receipt',
    'empirical-lawhood/planning/repeated-delivery-controller-evaluation-plan',
    'empirical-lawhood/planning/executable-study-definition',
    'empirical-lawhood/planning/controlled-map-reachability-receipt',
    'empirical-lawhood/runtime/compiled-admission-controller-study',
    'empirical-lawhood/runtime/repeated-delivery-controller-cohort-adjudication',
    'empirical-lawhood/runtime/repeated-delivery-controller-unit-evaluation',
    'empirical-lawhood/runtime/execution-envelope-spec',
    'empirical-lawhood/runtime/envelope-execution-plan',
    'empirical-lawhood/runtime/sealed-repeated-delivery-controller-bundle',
    'empirical-lawhood/runtime/run-envelope-conformance-graph',
    'empirical-lawhood/runtime/run-envelope-route-conformance-receipt',
    'empirical-lawhood/runtime/run-envelope-operational-conformance-receipt',
    'empirical-lawhood/runtime/revealed-repeated-delivery-controller-bundle',
    'empirical-lawhood/runtime/run-execution-envelope',
    'empirical-lawhood/runtime/envelope-run-plan',
    'empirical-lawhood/runtime/envelope-run-recovery-index',
    'empirical-lawhood/runtime/envelope-graph-preservation-receipt',
    'empirical-lawhood/runtime/issued-executable-study-manifest',
    'empirical-lawhood/runtime/executable-study-candidate',
)


@dataclass(frozen=True, slots=True)
class ReleaseSchemaBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/release-schema-binding'

    binding_id: str
    current_schema: str
    replay_source_schemas: tuple[str, ...]
    compatibility_projection: ObjectIdentity | None

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_schema(self.current_schema)
        for value in self.replay_source_schemas:
            validate_schema(value)
        require_sorted_unique_strings(
            self.replay_source_schemas,
            field_name="replay_source_schemas",
        )
        if self.replay_source_schemas and self.compatibility_projection is None:
            raise ValueError("release schema replay source lacks its exact projection")
        if not self.replay_source_schemas and self.compatibility_projection is not None:
            raise ValueError("release schema projection lacks a replay source")


@dataclass(frozen=True, slots=True)
class ReleaseRepositoryStateReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/release-repository-state-receipt'

    receipt_id: str
    implementation_commit: str
    source_tree_sha256: str
    clean_worktree: bool
    observed_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit) is None:
            raise ValueError("release repository commit is not an exact Git SHA-1")
        validate_sha256(self.source_tree_sha256, field_name="source_tree_sha256")
        parse_utc_timestamp(self.observed_at_utc, field_name="observed_at_utc")
        if not self.clean_worktree:
            raise ValueError("response-law release requires a clean source worktree")


@dataclass(frozen=True, slots=True)
class ReleaseVerificationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/release-verification-receipt'

    receipt_id: str
    kind: ConsolidationVerificationKind
    implementation_commit: str
    command_sha256: str
    result_sha256: str
    passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if re.fullmatch(r"[0-9a-f]{40}", self.implementation_commit) is None:
            raise ValueError("release verification commit is not an exact Git SHA-1")
        validate_sha256(self.command_sha256, field_name="command_sha256")
        validate_sha256(self.result_sha256, field_name="result_sha256")
        if not self.passed:
            raise ValueError("response-law release contains a failed verification")


@dataclass(frozen=True, slots=True)
class ReleaseConformanceReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/release-conformance-receipt'

    receipt_id: str
    public_composition: ObjectIdentity
    candidate_graph: ObjectIdentity
    external_test_plane_id: str
    conformance_case_ids: tuple[str, ...]
    terminal: bool
    created_empirical_evidence: bool
    result_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.external_test_plane_id,
            field_name="external_test_plane_id",
        )
        require_sorted_unique_strings(
            self.conformance_case_ids,
            field_name="conformance_case_ids",
            allow_empty=False,
        )
        validate_sha256(self.result_sha256, field_name="result_sha256")
        if (
            self.public_composition.object_schema != 'empirical-lawhood/runtime/run-envelope-route-composition'
            or self.candidate_graph.object_schema
            != 'empirical-lawhood/runtime/run-envelope-conformance-graph'
        ):
            raise ValueError("release conformance binds another public route")
        if not self.terminal or self.created_empirical_evidence:
            raise ValueError("release conformance is nonterminal or claims empirical evidence")


@dataclass(frozen=True, slots=True)
class ReleaseRecoveryInjectionReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/release-recovery-injection-receipt'

    receipt_id: str
    execution_envelope: ObjectIdentity
    recovery_index: ObjectIdentity
    injection_case_ids: tuple[str, ...]
    source_bytes_sha256: str
    recovered_bytes_sha256: str
    scientific_recomputation_count: int
    scientific_retuning_count: int
    terminal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(
            self.injection_case_ids,
            field_name="injection_case_ids",
            allow_empty=False,
        )
        validate_sha256(self.source_bytes_sha256, field_name="source_bytes_sha256")
        validate_sha256(self.recovered_bytes_sha256, field_name="recovered_bytes_sha256")
        if self.source_bytes_sha256 != self.recovered_bytes_sha256:
            raise ValueError("recovery injection changed scientific bytes")
        if self.scientific_recomputation_count or self.scientific_retuning_count:
            raise ValueError("recovery injection recomputed or retuned science")
        if not self.terminal:
            raise ValueError("release recovery injection is not terminal")


@dataclass(frozen=True, slots=True)
class ReleaseAuthorityStop(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/release-authority-stop'

    stop_id: str
    kind: ConsolidationAuthorityKind
    reason_code: str
    required_external_act: str

    def __post_init__(self) -> None:
        validate_stable_id(self.stop_id, field_name="stop_id")
        validate_nonempty(self.reason_code, field_name="reason_code")
        validate_nonempty(self.required_external_act, field_name="required_external_act")


@dataclass(frozen=True, slots=True)
class ResponseLawConsolidationRelease(CanonicalRecord):
    """Machine-readable release boundary; never itself empirical evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/response-law-consolidation-release'

    release_id: str
    plan: ObjectIdentity
    accepted_decision: ObjectIdentity
    schema_bindings: tuple[ReleaseSchemaBinding, ...]
    public_capabilities: tuple[ObjectIdentity, ...]
    config_schema_ids: tuple[str, ...]
    implementation_bindings: tuple[ObjectIdentity, ...]
    public_composition: ObjectIdentity
    candidate_graph: ObjectIdentity
    semantic_output_contracts: tuple[ObjectIdentity, ...]
    source_closure: ImplementationSourceClosure
    repository_state: ReleaseRepositoryStateReceipt
    verification_receipts: tuple[ReleaseVerificationReceipt, ...]
    conformance_receipt: ReleaseConformanceReceipt
    recovery_receipt: ReleaseRecoveryInjectionReceipt
    external_artifact_plane_id: str
    external_test_plane_rebuildable: bool
    repository_contains_scientific_payloads: bool
    maximum_claim: ConsolidationClaimCeiling
    limitations: tuple[str, ...]
    authority_stops: tuple[ReleaseAuthorityStop, ...]
    released_at_utc: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.release_id, field_name="release_id")
        validate_stable_id(
            self.external_artifact_plane_id,
            field_name="external_artifact_plane_id",
        )
        parse_utc_timestamp(self.released_at_utc, field_name="released_at_utc")
        require_sorted_unique_ids(
            self.schema_bindings,
            attribute="binding_id",
            field_name="schema_bindings",
        )
        observed_schemas = {value.current_schema for value in self.schema_bindings}
        if len(observed_schemas) != len(self.schema_bindings):
            raise ValueError("release output schema is duplicated")
        if not set(REQUIRED_RELEASE_CURRENT_SCHEMAS).issubset(observed_schemas):
            raise ValueError("release output schema roster is incomplete")
        require_sorted_unique_ids(
            self.public_capabilities,
            attribute="object_id",
            field_name="public_capabilities",
        )
        require_sorted_unique_strings(
            self.config_schema_ids,
            field_name="config_schema_ids",
            allow_empty=False,
        )
        for value in self.config_schema_ids:
            validate_schema(value)
        require_sorted_unique_ids(
            self.implementation_bindings,
            attribute="object_id",
            field_name="implementation_bindings",
        )
        require_sorted_unique_ids(
            self.semantic_output_contracts,
            attribute="object_id",
            field_name="semantic_output_contracts",
        )
        if (
            not self.public_capabilities
            or not self.implementation_bindings
            or not self.semantic_output_contracts
        ):
            raise ValueError("release public capability/composition roster is incomplete")
        require_sorted_unique_ids(
            self.verification_receipts,
            attribute="receipt_id",
            field_name="verification_receipts",
        )
        require_sorted_unique_strings(
            self.limitations,
            field_name="limitations",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.authority_stops,
            attribute="stop_id",
            field_name="authority_stops",
        )
        required_verifications = set(ConsolidationVerificationKind)
        if {value.kind for value in self.verification_receipts} != required_verifications:
            raise ValueError("release verification roster is incomplete")
        commit = self.source_closure.implementation_commit
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or self.repository_state.implementation_commit != commit
            or self.repository_state.source_tree_sha256 != self.source_closure.source_tree_sha256
            or any(value.implementation_commit != commit for value in self.verification_receipts)
        ):
            raise ValueError("release records differ from the clean source closure")
        if (
            self.conformance_receipt.public_composition != self.public_composition
            or self.conformance_receipt.candidate_graph != self.candidate_graph
            or self.conformance_receipt.external_test_plane_id != self.external_artifact_plane_id
        ):
            raise ValueError("release public composition differs from conformance evidence")
        if (
            not self.external_test_plane_rebuildable
            or self.repository_contains_scientific_payloads
            or self.maximum_claim is not ConsolidationClaimCeiling.ARCHITECTURE_CONFORMANCE_ONLY
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("release overclaims its architecture-only evidence boundary")
        if {value.kind for value in self.authority_stops} != set(ConsolidationAuthorityKind):
            raise ValueError("release authority-stop roster is incomplete")


@dataclass(frozen=True, slots=True)
class ResponseLawDependencyPin(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/response-law-dependency-pin'

    pin_id: str
    consumer_plan: ObjectIdentity
    release: ObjectIdentity
    pinned_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.pin_id, field_name="pin_id")
        parse_utc_timestamp(self.pinned_at_utc, field_name="pinned_at_utc")
        if self.release.object_schema != ResponseLawConsolidationRelease.SCHEMA:
            raise ValueError("downstream dependency pin does not bind the canonical release")


@dataclass(frozen=True, slots=True)
class ReleaseConformanceAttestation(CanonicalRecord):
    "Additive correction binding a release to actually executed conformance."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/release-conformance-attestation'

    attestation_id: str
    superseded_release: ObjectIdentity
    corrected_release: ObjectIdentity
    public_composition: ObjectIdentity
    candidate_graph: ObjectIdentity
    conformance_suite: ObjectIdentity
    public_route_receipt: ObjectIdentity
    operational_conformance: ObjectIdentity
    provider_registry_sha256: str
    source_closure: ImplementationSourceClosure
    correction_reason_codes: tuple[str, ...]
    terminal: bool
    created_empirical_evidence: bool
    attested_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.attestation_id, field_name="attestation_id")
        parse_utc_timestamp(self.attested_at_utc, field_name="attested_at_utc")
        for value in (self.superseded_release, self.corrected_release):
            if value.object_schema != ResponseLawConsolidationRelease.SCHEMA:
                raise ValueError("release attestation binds another release schema")
        if self.superseded_release == self.corrected_release:
            raise ValueError("release correction requires a fresh canonical identity")
        expected_schemas = (
            (self.public_composition, RunEnvelopeRouteComposition.SCHEMA),
            (
                self.candidate_graph,
                'empirical-lawhood/runtime/run-envelope-conformance-graph',
            ),
            (self.conformance_suite, RunEnvelopeConformanceSuiteReceipt.SCHEMA),
            (self.public_route_receipt, RunEnvelopeRouteConformanceReceipt.SCHEMA),
            (
                self.operational_conformance,
                RunEnvelopeOperationalConformanceReceipt.SCHEMA,
            ),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("release attestation binds another conformance topology")
        validate_sha256(self.provider_registry_sha256, field_name="provider_registry_sha256")
        require_sorted_unique_strings(
            self.correction_reason_codes,
            field_name="correction_reason_codes",
            allow_empty=False,
        )
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or not self.terminal
            or self.created_empirical_evidence
        ):
            raise ValueError("release attestation is not a clean terminal correction")


@dataclass(frozen=True, slots=True)
class AttestedResponseLawDependencyPin(CanonicalRecord):
    """Downstream pin that cannot omit the corrective conformance attestation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/attested-response-law-dependency-pin'

    pin_id: str
    consumer_plan: ObjectIdentity
    release: ObjectIdentity
    conformance_attestation: ObjectIdentity
    pinned_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.pin_id, field_name="pin_id")
        parse_utc_timestamp(self.pinned_at_utc, field_name="pinned_at_utc")
        if self.release.object_schema != ResponseLawConsolidationRelease.SCHEMA:
            raise ValueError("downstream dependency pin does not bind the corrected release")
        if self.conformance_attestation.object_schema != ReleaseConformanceAttestation.SCHEMA:
            raise ValueError("downstream dependency pin omits the conformance correction")


REQUIRED_FLAGSHIP_READINESS_DECISION_IDS = (
    "deadline-free-execution-resource-successors",
    "fixed-round-evaluation-acquisition-successors",
    "multi-world-archive-simulator-readiness",
)

REQUIRED_FLAGSHIP_READINESS_CURRENT_SCHEMAS = (
    'empirical-lawhood/api/issued-compilation-package-reference',
    'empirical-lawhood/api/executable-study-compilation-summary',
    'empirical-lawhood/api/executable-study-issue-summary',
    'empirical-lawhood/api/multi-world-study-issue-summary',
    'empirical-lawhood/planning/archive-outcome-protection-plan',
    'empirical-lawhood/planning/archive-overlap-qualification',
    'empirical-lawhood/planning/archive-to-simulator-partial-morphism-spec',
    'empirical-lawhood/planning/bounded-query-acquisition-plan',
    'empirical-lawhood/planning/finite-chart-reference-design',
    'empirical-lawhood/planning/action-aware-controller-evaluation-plan',
    'empirical-lawhood/planning/prospective-evaluation-precommitment',
    'empirical-lawhood/planning/multi-world-joint-adjudication-plan',
    'empirical-lawhood/planning/morphism-negative-control-plan',
    'empirical-lawhood/planning/multi-world-outcome-barrier-plan',
    'empirical-lawhood/planning/multi-world-study-child-scientific-binding',
    'empirical-lawhood/planning/property-transport-expectation',
    'empirical-lawhood/runtime/archive-overlap-qualification-result',
    'empirical-lawhood/runtime/frozen-compilation-execution-plan',
    'empirical-lawhood/runtime/compiled-resource-envelope-reference',
    'empirical-lawhood/runtime/synthetic-controller-route-conformance-receipt',
    'empirical-lawhood/runtime/multi-world-study-operation-runbook',
    'empirical-lawhood/runtime/multi-world-public-route-conformance-receipt',
    'empirical-lawhood/runtime/multi-world-executed-conformance-suite',
    'empirical-lawhood/runtime/extension-publication-receipt',
    'empirical-lawhood/runtime/issued-multi-world-study',
    'empirical-lawhood/runtime/multi-world-joint-adjudication-result',
    'empirical-lawhood/runtime/morphism-control-contrast-receipt',
    'empirical-lawhood/runtime/multi-world-outcome-barrier-prefix',
    'empirical-lawhood/runtime/multi-world-study-execution-plan',
    'empirical-lawhood/runtime/multi-world-study-issue-manifest',
    'empirical-lawhood/runtime/multi-world-study-recovery-index',
    'empirical-lawhood/runtime/study-extension-materialization-receipt',
    'empirical-lawhood/runtime/property-morphism-verdict',
    'empirical-lawhood/runtime/compiled-run-plan-reference',
    'empirical-lawhood/runtime/run-recovery-reference',
    'empirical-lawhood/runtime/executable-study-compilation-report',
)


class MultiWorldReadinessConformanceKind(StrEnum):
    SINGLE_WORLD_CONFORMANCE = "SINGLE_WORLD_CONFORMANCE"
    MULTI_WORLD_THREE_CHILD = "MULTI_WORLD_THREE_CHILD"
    HOSTILE_IDENTITY_ACCESS = "HOSTILE_IDENTITY_ACCESS"
    RECOVERY_AND_CATALOG_REBUILD = "RECOVERY_AND_CATALOG_REBUILD"
    CURRENT_CONTRACT_REPLAY = "CURRENT_CONTRACT_REPLAY"


@dataclass(frozen=True, slots=True)
class MultiWorldReadinessConformanceEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-readiness-conformance-evidence'

    evidence_id: str
    kind: MultiWorldReadinessConformanceKind
    evidence: ObjectIdentity
    terminal: bool
    architecture_conformance_only: bool
    created_empirical_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        if (
            not self.terminal
            or not self.architecture_conformance_only
            or self.created_empirical_evidence
        ):
            raise ValueError("flagship readiness conformance overclaims its evidence")


@dataclass(frozen=True, slots=True)
class MultiWorldReadinessReleaseGenerator(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-readiness-release-generator'

    generator_id: str
    generator_version: str
    implementation_sha256: str
    release_schema: str
    strict_decoder_schema: str

    def __post_init__(self) -> None:
        validate_stable_id(self.generator_id, field_name="generator_id")
        if self.generator_version != "1.0.0":
            raise ValueError("flagship readiness generator version differs")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.release_schema != MultiWorldReadinessRelease.SCHEMA:
            raise ValueError("flagship readiness generator targets another release schema")
        if self.strict_decoder_schema != self.release_schema:
            raise ValueError("flagship readiness generator/decoder schema differs")


@dataclass(frozen=True, slots=True)
class MultiWorldReadinessRelease(CanonicalRecord):
    "Architecture-only release for the accepted multi-world readiness composition."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-readiness-release'

    release_id: str
    corrected_base_release: ObjectIdentity
    corrected_base_attestation: ObjectIdentity
    accepted_decisions: tuple[ObjectIdentity, ...]
    generator: ObjectIdentity
    source_closure: ImplementationSourceClosure
    schema_bindings: tuple[ReleaseSchemaBinding, ...]
    public_capabilities: tuple[ObjectIdentity, ...]
    proof_owner_bindings: tuple[ObjectIdentity, ...]
    semantic_output_contracts: tuple[ObjectIdentity, ...]
    conformance_evidence: tuple[MultiWorldReadinessConformanceEvidence, ...]
    verification_receipts: tuple[ReleaseVerificationReceipt, ...]
    limitations: tuple[str, ...]
    authority_stops: tuple[ReleaseAuthorityStop, ...]
    maximum_claim: ConsolidationClaimCeiling
    released_at_utc: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.release_id, field_name="release_id")
        parse_utc_timestamp(self.released_at_utc, field_name="released_at_utc")
        if self.corrected_base_release.object_schema != ResponseLawConsolidationRelease.SCHEMA:
            raise ValueError("readiness release does not bind the corrected base release")
        if self.corrected_base_attestation.object_schema != ReleaseConformanceAttestation.SCHEMA:
            raise ValueError("readiness release does not bind the corrected base attestation")
        if self.generator.object_schema != MultiWorldReadinessReleaseGenerator.SCHEMA:
            raise ValueError("readiness release does not bind its registered generator")
        require_sorted_unique_ids(
            self.accepted_decisions,
            attribute="object_id",
            field_name="accepted_decisions",
        )
        if tuple(value.object_id for value in self.accepted_decisions) != (
            REQUIRED_FLAGSHIP_READINESS_DECISION_IDS
        ):
            raise ValueError("readiness release changes the its accepted decision roster")
        require_sorted_unique_ids(
            self.schema_bindings,
            attribute="binding_id",
            field_name="schema_bindings",
        )
        if not set(REQUIRED_FLAGSHIP_READINESS_CURRENT_SCHEMAS).issubset(
            {value.current_schema for value in self.schema_bindings}
        ):
            raise ValueError("readiness release output schema roster is incomplete")
        for name, values in (
            ("public_capabilities", self.public_capabilities),
            ("proof_owner_bindings", self.proof_owner_bindings),
            ("semantic_output_contracts", self.semantic_output_contracts),
        ):
            require_sorted_unique_ids(
                values,
                attribute="object_id",
                field_name=name,
            )
            if not values:
                raise ValueError(f"readiness release {name} cannot be empty")
        require_sorted_unique_ids(
            self.conformance_evidence,
            attribute="evidence_id",
            field_name="conformance_evidence",
        )
        if {value.kind for value in self.conformance_evidence} != set(
            MultiWorldReadinessConformanceKind
        ):
            raise ValueError("readiness release conformance roster is incomplete")
        require_sorted_unique_ids(
            self.verification_receipts,
            attribute="receipt_id",
            field_name="verification_receipts",
        )
        if {value.kind for value in self.verification_receipts} != set(
            ConsolidationVerificationKind
        ):
            raise ValueError("readiness release verification roster is incomplete")
        if any(
            value.implementation_commit != self.source_closure.implementation_commit
            for value in self.verification_receipts
        ):
            raise ValueError("readiness verification differs from source closure")
        require_sorted_unique_strings(
            self.limitations,
            field_name="limitations",
            allow_empty=False,
        )
        required_limitations = {
            "no-fair-mast-or-torax-qualification",
            "no-fresh-law-qualification-or-prospective-controller-evaluation-evidence",
            "no-issue-run-reveal-or-actuation-authority",
            "no-simulator-or-physical-portability-claim",
        }
        if not required_limitations.issubset(self.limitations):
            raise ValueError("readiness release limitations are incomplete")
        require_sorted_unique_ids(
            self.authority_stops,
            attribute="stop_id",
            field_name="authority_stops",
        )
        if {value.kind for value in self.authority_stops} != set(ConsolidationAuthorityKind):
            raise ValueError("readiness release authority-stop roster is incomplete")
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or self.maximum_claim is not ConsolidationClaimCeiling.ARCHITECTURE_CONFORMANCE_ONLY
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("readiness release crosses its architecture-only boundary")


@dataclass(frozen=True, slots=True)
class MultiWorldReadinessReleaseAttestation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-readiness-release-attestation'

    attestation_id: str
    release: ObjectIdentity
    generator: ObjectIdentity
    conformance_evidence: tuple[ObjectIdentity, ...]
    source_closure: ImplementationSourceClosure
    terminal: bool
    created_empirical_evidence: bool
    attested_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.attestation_id, field_name="attestation_id")
        parse_utc_timestamp(self.attested_at_utc, field_name="attested_at_utc")
        if self.release.object_schema != MultiWorldReadinessRelease.SCHEMA:
            raise ValueError("readiness attestation binds another release schema")
        if self.generator.object_schema != MultiWorldReadinessReleaseGenerator.SCHEMA:
            raise ValueError("readiness attestation binds another generator schema")
        require_sorted_unique_ids(
            self.conformance_evidence,
            attribute="object_id",
            field_name="conformance_evidence",
        )
        if not self.conformance_evidence:
            raise ValueError("readiness attestation lacks conformance evidence")
        if not self.terminal or self.created_empirical_evidence:
            raise ValueError("readiness attestation is nonterminal or empirical")


@dataclass(frozen=True, slots=True)
class MultiWorldReadinessConsumerPin(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-readiness-consumer-pin'

    pin_id: str
    consumer_plan: ObjectIdentity
    corrected_base_release: ObjectIdentity
    corrected_base_attestation: ObjectIdentity
    readiness_release: ObjectIdentity
    readiness_attestation: ObjectIdentity
    predecessor_pin: ObjectIdentity
    pinned_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.pin_id, field_name="pin_id")
        parse_utc_timestamp(self.pinned_at_utc, field_name="pinned_at_utc")
        expected = (
            (self.corrected_base_release, ResponseLawConsolidationRelease.SCHEMA),
            (self.corrected_base_attestation, ReleaseConformanceAttestation.SCHEMA),
            (self.readiness_release, MultiWorldReadinessRelease.SCHEMA),
            (
                self.readiness_attestation,
                MultiWorldReadinessReleaseAttestation.SCHEMA,
            ),
            (self.predecessor_pin, AttestedResponseLawDependencyPin.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected):
            raise ValueError("multi-world release pin changes its release/attestation lineage")


@dataclass(frozen=True, slots=True)
class StudyReadinessExportPayload:
    relative_path: str
    payload_schema: str
    payload: bytes

    def __post_init__(self) -> None:
        if self.relative_path.startswith("/") or ".." in self.relative_path.split("/"):
            raise ValueError("release export path must be safe and relative")
        validate_schema(self.payload_schema)
        if not self.payload:
            raise ValueError("release export payload must be nonempty")


def generate_multi_world_consumer_readiness_release(
    *,
    generator: MultiWorldReadinessReleaseGenerator,
    release_id: str,
    corrected_base_release: ObjectIdentity,
    corrected_base_attestation: ObjectIdentity,
    accepted_decisions: tuple[ObjectIdentity, ...],
    source_closure: ImplementationSourceClosure,
    schema_bindings: tuple[ReleaseSchemaBinding, ...],
    public_capabilities: tuple[ObjectIdentity, ...],
    proof_owner_bindings: tuple[ObjectIdentity, ...],
    semantic_output_contracts: tuple[ObjectIdentity, ...],
    conformance_evidence: tuple[MultiWorldReadinessConformanceEvidence, ...],
    verification_receipts: tuple[ReleaseVerificationReceipt, ...],
    limitations: tuple[str, ...],
    authority_stops: tuple[ReleaseAuthorityStop, ...],
    released_at_utc: str,
) -> MultiWorldReadinessRelease:
    """Dedicated constructor; generic authoring codecs never construct releases."""

    return MultiWorldReadinessRelease(
        release_id=release_id,
        corrected_base_release=corrected_base_release,
        corrected_base_attestation=corrected_base_attestation,
        accepted_decisions=accepted_decisions,
        generator=ObjectIdentity.from_record(generator.generator_id, generator),
        source_closure=source_closure,
        schema_bindings=schema_bindings,
        public_capabilities=public_capabilities,
        proof_owner_bindings=proof_owner_bindings,
        semantic_output_contracts=semantic_output_contracts,
        conformance_evidence=conformance_evidence,
        verification_receipts=verification_receipts,
        limitations=limitations,
        authority_stops=authority_stops,
        maximum_claim=ConsolidationClaimCeiling.ARCHITECTURE_CONFORMANCE_ONLY,
        released_at_utc=released_at_utc,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def decode_multi_world_consumer_readiness_release(payload: bytes) -> MultiWorldReadinessRelease:
    return decode_canonical_bytes(
        payload,
        MultiWorldReadinessRelease,
        maximum_bytes=64 * 1024 * 1024,
    )


def study_readiness_release_export_payloads(
    *,
    release: MultiWorldReadinessRelease,
    attestation: MultiWorldReadinessReleaseAttestation,
    pin: MultiWorldReadinessConsumerPin,
) -> tuple[StudyReadinessExportPayload, ...]:
    """Return the exact no-write export set for a guarded no-replace publisher."""

    if (
        attestation.release != ObjectIdentity.from_record(release.release_id, release)
        or pin.readiness_release != ObjectIdentity.from_record(release.release_id, release)
        or pin.readiness_attestation
        != ObjectIdentity.from_record(attestation.attestation_id, attestation)
    ):
        raise ValueError("readiness release export lineage differs")
    return (
        StudyReadinessExportPayload(
            relative_path="multi-world-readiness-release.json",
            payload_schema=release.SCHEMA,
            payload=release.canonical_bytes(),
        ),
        StudyReadinessExportPayload(
            relative_path="multi-world-readiness-release-attestation.json",
            payload_schema=attestation.SCHEMA,
            payload=attestation.canonical_bytes(),
        ),
        StudyReadinessExportPayload(
            relative_path="multi-world-readiness-consumer-pin.json",
            payload_schema=pin.SCHEMA,
            payload=pin.canonical_bytes(),
        ),
    )


def attest_response_law_release(
    *,
    attestation_id: str,
    superseded_release: ObjectIdentity,
    corrected_release: ResponseLawConsolidationRelease,
    public_composition: RunEnvelopeRouteComposition,
    public_route_receipt: RunEnvelopeRouteConformanceReceipt,
    conformance_suite: RunEnvelopeConformanceSuiteReceipt,
    correction_reason_codes: tuple[str, ...],
    attested_at_utc: str,
) -> ReleaseConformanceAttestation:
    """Validate exact source/capability/provider/recovery continuity before attesting."""

    release_identity = ObjectIdentity.from_record(corrected_release.release_id, corrected_release)
    composition_identity = ObjectIdentity.from_record(
        public_composition.composition_id,
        public_composition,
    )
    graph_identity = ObjectIdentity.from_record(
        public_composition.graph.graph_id,
        public_composition.graph,
    )
    route_identity = ObjectIdentity.from_record(
        public_route_receipt.receipt_id,
        public_route_receipt,
    )
    operational = public_route_receipt.operational_conformance
    operational_identity = ObjectIdentity.from_record(operational.receipt_id, operational)
    suite_identity = ObjectIdentity.from_record(conformance_suite.suite_id, conformance_suite)
    expected_capabilities = tuple(
        sorted(
            (
                ObjectIdentity.from_record(value.capability_key, value)
                for value in public_composition.capabilities
            ),
            key=lambda value: value.object_id,
        )
    )
    expected_owners = tuple(
        sorted(
            (
                ObjectIdentity.from_record(value.owner_id, value)
                for value in public_composition.proof_owners
            ),
            key=lambda value: value.object_id,
        )
    )
    expected_semantics = tuple(
        sorted(
            (
                ObjectIdentity.from_record(value.contract_id, value)
                for value in public_composition.semantic_output_contracts
            ),
            key=lambda value: value.object_id,
        )
    )
    expected_config_schemas = {
        PublicRouteStageConfiguration.SCHEMA,
        'empirical-lawhood/methods/tagged-identification-projection-plan',
    }
    if (
        corrected_release.public_composition != composition_identity
        or corrected_release.candidate_graph != graph_identity
        or corrected_release.public_capabilities != expected_capabilities
        or corrected_release.implementation_bindings != expected_owners
        or corrected_release.semantic_output_contracts != expected_semantics
        or not expected_config_schemas.issubset(corrected_release.config_schema_ids)
        or corrected_release.source_closure != public_composition.implementation_source
        or public_route_receipt.public_composition != composition_identity
        or public_route_receipt.graph != public_composition.graph
        or operational.provider_registry_sha256 != public_composition.provider_registry_sha256
        or conformance_suite.primary_receipt != route_identity
        or conformance_suite.public_composition != composition_identity
        or conformance_suite.operational_conformance != operational_identity
        or conformance_suite.provider_registry_sha256 != operational.provider_registry_sha256
        or corrected_release.conformance_receipt.result_sha256 != conformance_suite.fingerprint()
        or corrected_release.recovery_receipt.execution_envelope != operational.execution_envelope
        or corrected_release.recovery_receipt.recovery_index != operational.recovery_index
        or corrected_release.recovery_receipt.source_bytes_sha256
        != operational.source_envelope_bytes_sha256
        or corrected_release.recovery_receipt.recovered_bytes_sha256
        != operational.recovered_envelope_bytes_sha256
    ):
        raise ValueError("corrected release differs from executed public-route evidence")
    return ReleaseConformanceAttestation(
        attestation_id=attestation_id,
        superseded_release=superseded_release,
        corrected_release=release_identity,
        public_composition=composition_identity,
        candidate_graph=graph_identity,
        conformance_suite=suite_identity,
        public_route_receipt=route_identity,
        operational_conformance=operational_identity,
        provider_registry_sha256=operational.provider_registry_sha256,
        source_closure=corrected_release.source_closure,
        correction_reason_codes=correction_reason_codes,
        terminal=True,
        created_empirical_evidence=False,
        attested_at_utc=attested_at_utc,
    )


__all__ = [
    "ConsolidationAuthorityKind",
    "ConsolidationClaimCeiling",
    "ConsolidationVerificationKind",
    'MultiWorldReadinessConsumerPin',
    'MultiWorldReadinessConformanceEvidence',
    'MultiWorldReadinessConformanceKind',
    'StudyReadinessExportPayload',
    "ReleaseAuthorityStop",
    "ReleaseConformanceReceipt",
    'ReleaseConformanceAttestation',
    "ReleaseRecoveryInjectionReceipt",
    "ReleaseRepositoryStateReceipt",
    "ReleaseSchemaBinding",
    "ReleaseVerificationReceipt",
    "REQUIRED_RELEASE_CURRENT_SCHEMAS",
    "REQUIRED_FLAGSHIP_READINESS_DECISION_IDS",
    "REQUIRED_FLAGSHIP_READINESS_CURRENT_SCHEMAS",
    'ResponseLawDependencyPin',
    'AttestedResponseLawDependencyPin',
    'ResponseLawConsolidationRelease',
    'MultiWorldReadinessReleaseAttestation',
    'MultiWorldReadinessReleaseGenerator',
    'MultiWorldReadinessRelease',
    "attest_response_law_release",
    'decode_multi_world_consumer_readiness_release',
    'study_readiness_release_export_payloads',
    'generate_multi_world_consumer_readiness_release',
]
