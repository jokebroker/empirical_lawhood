"""Claim-limited release records for the executable metatheory platform.

These records bind source-free contract conformance.  They are not empirical
metatheory evidence and they grant no source, experiment, reveal or actuation
authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import ClassVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.planning.metatheory import MetatheoryEvidenceCeiling
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind
from empirical_lawhood.runtime.executable_bindings import GeneratedExecutableBindingAggregate
from empirical_lawhood.runtime.extension_bundles import GeneratedExtensionBundleAggregate
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.response_law_release import (
    ConsolidationClaimCeiling,
    ReleaseRepositoryStateReceipt,
    ReleaseVerificationReceipt,
)

EXECUTABLE_METATHEORY_RELEASE_SCHEMA_ROSTER = tuple(
    sorted(
        (
            'empirical-lawhood/planning/atlas-qualification-cell-result',
            'empirical-lawhood/planning/atlas-qualification-result',
            'empirical-lawhood/planning/atlas-qualification-spec',
            'empirical-lawhood/planning/categorical-forecast',
            'empirical-lawhood/planning/chart-transition-evidence',
            'empirical-lawhood/planning/coordinate-candidate',
            'empirical-lawhood/planning/coordinate-challenge-cell-result',
            'empirical-lawhood/planning/coordinate-challenge-evidence',
            'empirical-lawhood/planning/coordinate-challenge-nomination',
            'empirical-lawhood/planning/coordinate-challenge-result',
            'empirical-lawhood/planning/coordinate-challenge-spec',
            'empirical-lawhood/planning/decision-assurance-cell',
            'empirical-lawhood/planning/decision-assurance-result',
            'empirical-lawhood/planning/decision-assurance-spec',
            'empirical-lawhood/planning/decision-assurance-target',
            'empirical-lawhood/planning/decision-comparison-evidence',
            'empirical-lawhood/planning/decision-error-interval',
            'empirical-lawhood/planning/decision-error-rule',
            'empirical-lawhood/planning/dynamical-forecast',
            'empirical-lawhood/planning/metatheory-adjudication-cell',
            'empirical-lawhood/planning/metatheory-adjudication-result',
            'empirical-lawhood/planning/metatheory-adjudication-spec',
            'empirical-lawhood/planning/metatheory-adjudication-stop',
            'empirical-lawhood/planning/metatheory-campaign-artifact-binding',
            'empirical-lawhood/planning/metatheory-campaign-capability-binding',
            'empirical-lawhood/planning/metatheory-campaign-owner-binding',
            'empirical-lawhood/planning/metatheory-campaign-profile',
            'empirical-lawhood/planning/metatheory-campaign-rosters',
            'empirical-lawhood/planning/metatheory-forecast-cell',
            'empirical-lawhood/planning/metatheory-method-selection',
            'empirical-lawhood/planning/metatheory-prediction-issue-receipt',
            'empirical-lawhood/planning/metatheory-prediction-package',
            'empirical-lawhood/planning/metatheory-reveal-authorization-binding',
            'empirical-lawhood/planning/metatheory-scoring-method-binding',
            'empirical-lawhood/planning/metatheory-sealed-target',
            'empirical-lawhood/planning/metatheory-stage-applicability',
            'empirical-lawhood/planning/metatheory-target-acquisition-index',
            'empirical-lawhood/planning/metatheory-target-outcome',
            'empirical-lawhood/planning/metatheory-target-spec',
            'empirical-lawhood/planning/metric-forecast',
            'empirical-lawhood/planning/obstruction-atlas-build-stop',
            'empirical-lawhood/planning/obstruction-atlas-spec',
            'empirical-lawhood/planning/obstruction-atlas',
            'empirical-lawhood/planning/obstruction-cell',
            'empirical-lawhood/planning/obstruction-expected-cell',
            'empirical-lawhood/planning/obstruction-mapping-registry',
            'empirical-lawhood/planning/obstruction-mapping',
            'empirical-lawhood/planning/obstruction-source-binding',
            'empirical-lawhood/planning/property-path-consistency-cell',
            'empirical-lawhood/planning/property-survival-assessment',
            'empirical-lawhood/planning/property-survival-cell-evidence',
            'empirical-lawhood/planning/property-survival-cell',
            'empirical-lawhood/planning/property-survival-member-spec',
            'empirical-lawhood/planning/property-survival-signature',
            'empirical-lawhood/planning/property-survival-spec',
            'empirical-lawhood/planning/scientific-source-gate-evidence',
            'empirical-lawhood/planning/scientific-source-gate-result',
            'empirical-lawhood/planning/scientific-source-gate-spec',
            'empirical-lawhood/planning/scientific-source-qualification-result',
            'empirical-lawhood/planning/scientific-source-qualification-spec',
            'empirical-lawhood/planning/scientific-source-qualification-stop',
            'empirical-lawhood/planning/set-valued-chart-region',
            'empirical-lawhood/runtime/metatheory-campaign-compilation',
            'empirical-lawhood/runtime/metatheory-campaign-condition-edge',
            'empirical-lawhood/runtime/executable-metatheory-conformance-case-receipt',
            'empirical-lawhood/runtime/executable-metatheory-conformance-suite',
            'empirical-lawhood/runtime/executable-metatheory-recovery-receipt',
            'empirical-lawhood/runtime/scientific-adjudication-record',
        )
    )
)


class ExecutableMetatheoryReleaseStatus(StrEnum):
    PLATFORM_CONFORMANCE_COMPLETE = "PLATFORM_CONFORMANCE_COMPLETE"


class ExecutableMetatheoryRouteKind(StrEnum):
    FAMILY_NEUTRAL_PUBLIC_EXECUTION = "FAMILY_NEUTRAL_PUBLIC_EXECUTION"


class MetatheoryConformanceCase(StrEnum):
    "Closed, nonauthoritative case labels used by release records."

    SUPPORTED = "SUPPORTED"
    OPPOSED_PROPERTY_DECISION = "OPPOSED_PROPERTY_DECISION"
    COLLISION_ABSENT_UNEVALUABLE = "COLLISION_ABSENT_UNEVALUABLE"
    ATLAS_LOCALIZATION_TOO_WIDE = "ATLAS_LOCALIZATION_TOO_WIDE"
    MISSING_PERMITTED_TARGET = "MISSING_PERMITTED_TARGET"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    AUTHORITY_STOP = "AUTHORITY_STOP"


REQUIRED_EXECUTABLE_METATHEORY_CASES = tuple(
    sorted(value.value for value in MetatheoryConformanceCase)
)


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryNoEffectReadiness(CanonicalRecord):
    """Exact full-topology proof made before the accepted conformance issue."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-no-effect-readiness'

    witness_id: str
    source_closure: ImplementationSourceClosure
    case_fixture_ids: tuple[str, ...]
    campaign_compilations: tuple[ObjectIdentity, ...]
    capability_registry: ObjectIdentity
    discovery_aggregate: ObjectIdentity
    executable_aggregate: ObjectIdentity
    provider_factory_keys: tuple[str, ...]
    decoder_registration_ids: tuple[str, ...]
    artifact_validator_ids: tuple[str, ...]
    output_schema_ids: tuple[str, ...]
    output_relative_locator_templates: tuple[str, ...]
    authority_inputs: tuple[ObjectIdentity, ...]
    storage_root_id: str
    catalog_relative_locator: str
    task_count: int
    external_input_count: int
    total_source_byte_limit: int
    total_output_byte_limit: int
    total_memory_byte_limit: int
    source_contact_count: int
    target_contact_count: int
    protected_outcome_read_count: int
    network_request_count: int
    external_write_count: int
    catalog_mutation_count: int
    scientific_effect_count: int
    artifact_persistence_ready: bool
    control_record_persistence_ready: bool
    ready: bool
    observed_at_utc: str
    created_empirical_evidence: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.witness_id, field_name="witness_id")
        parse_utc_timestamp(self.observed_at_utc, field_name="observed_at_utc")
        require_sorted_unique_strings(
            self.case_fixture_ids,
            field_name="case_fixture_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.provider_factory_keys,
            field_name="provider_factory_keys",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.decoder_registration_ids,
            field_name="decoder_registration_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.artifact_validator_ids,
            field_name="artifact_validator_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.output_schema_ids,
            field_name="output_schema_ids",
            allow_empty=False,
        )
        for value in self.output_schema_ids:
            validate_schema(value)
        require_sorted_unique_strings(
            self.output_relative_locator_templates,
            field_name="output_relative_locator_templates",
            allow_empty=False,
        )
        for value in self.output_relative_locator_templates:
            validate_relative_locator(value)
        require_sorted_unique_ids(
            self.authority_inputs,
            attribute="object_id",
            field_name="authority_inputs",
        )
        validate_stable_id(self.storage_root_id, field_name="storage_root_id")
        validate_relative_locator(self.catalog_relative_locator)
        compilation_fingerprints = tuple(
            value.object_fingerprint for value in self.campaign_compilations
        )
        if (
            tuple(sorted(set(compilation_fingerprints))) != compilation_fingerprints
            or not self.campaign_compilations
            or any(
                value.object_schema != 'empirical-lawhood/runtime/metatheory-campaign-compilation'
                for value in self.campaign_compilations
            )
        ):
            raise ValueError("metatheory readiness compilation roster differs")
        expected_schemas = (
            (self.capability_registry, 'empirical-lawhood/runtime/capability-registry'),
            (self.discovery_aggregate, GeneratedExtensionBundleAggregate.SCHEMA),
            (self.executable_aggregate, GeneratedExecutableBindingAggregate.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("metatheory readiness binds another topology")
        nonnegative_counts = (
            self.task_count,
            self.external_input_count,
            self.total_source_byte_limit,
            self.total_output_byte_limit,
            self.total_memory_byte_limit,
        )
        if any(type(value) is not int or value <= 0 for value in nonnegative_counts):
            raise ValueError("metatheory readiness requires positive derived totals")
        effect_counts = (
            self.source_contact_count,
            self.target_contact_count,
            self.protected_outcome_read_count,
            self.network_request_count,
            self.external_write_count,
            self.catalog_mutation_count,
            self.scientific_effect_count,
        )
        if any(type(value) is not int or value != 0 for value in effect_counts):
            raise ValueError("metatheory readiness proof had an effect")
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or set(self.case_fixture_ids) != set(REQUIRED_EXECUTABLE_METATHEORY_CASES)
            or not self.authority_inputs
            or not self.artifact_persistence_ready
            or not self.control_record_persistence_ready
            or not self.ready
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("metatheory readiness proof exceeds or misses its boundary")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryConformanceCaseReceipt(CanonicalRecord):
    """One actually executed public-route terminal case."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-conformance-case-receipt'

    case_id: str
    case_kind: MetatheoryConformanceCase
    authoritative_case_relative_root: str
    fixture: ObjectIdentity
    campaign_compilation: ObjectIdentity
    issued_package: ObjectIdentity
    run_plan: ObjectIdentity
    execution_plan: ObjectIdentity
    execution_resource_envelope: ObjectIdentity
    recovery_index: ObjectIdentity
    task_receipts: tuple[ObjectIdentity, ...]
    terminal_outputs: tuple[ObjectIdentity, ...]
    adjudication: ObjectIdentity
    pre_reveal_public_result: ObjectIdentity
    terminal_public_result: ObjectIdentity
    provider_registry_sha256: str
    scientific_status: ScientificStatus
    target_contact_count: int
    reveal_count: int
    plumbing_only: bool
    terminal: bool
    created_empirical_evidence: bool
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        validate_relative_locator(self.authoritative_case_relative_root)
        validate_sha256(
            self.provider_registry_sha256,
            field_name="provider_registry_sha256",
        )
        require_sorted_unique_ids(
            self.task_receipts,
            attribute="object_id",
            field_name="task_receipts",
        )
        require_sorted_unique_ids(
            self.terminal_outputs,
            attribute="object_id",
            field_name="terminal_outputs",
        )
        expected_schemas = (
            (self.fixture, 'empirical-lawhood/methods/backbone-structural-transport/metatheory-conformance-fixture'),
            (
                self.campaign_compilation,
                'empirical-lawhood/runtime/metatheory-campaign-compilation',
            ),
            (self.issued_package, 'empirical-lawhood/api/issued-compilation-package-reference'),
            (self.run_plan, 'empirical-lawhood/runtime/compiled-run-plan-reference'),
            (self.execution_plan, 'empirical-lawhood/runtime/frozen-compilation-execution-plan'),
            (
                self.execution_resource_envelope,
                'empirical-lawhood/runtime/compiled-resource-envelope-reference',
            ),
            (self.recovery_index, 'empirical-lawhood/runtime/run-recovery-reference'),
            (self.adjudication, ScientificAdjudicationRecord.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("metatheory conformance case binds another route")
        if not self.task_receipts or not self.terminal_outputs:
            raise ValueError("metatheory conformance case lacks actual receipts")
        for name in ("target_contact_count", "reveal_count"):
            value = getattr(self, name)
            if type(value) is not int or value < 0 or value > 1:
                raise ValueError("metatheory conformance contact count is invalid")
        if (
            not self.plumbing_only
            or not self.terminal
            or self.created_empirical_evidence
            or self.maximum_structural_evidence_ceiling
            is not MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE
        ):
            raise ValueError("metatheory conformance case overclaims its evidence")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryRecoveryReceipt(CanonicalRecord):
    """Actual total-catalog-loss recovery without executor re-entry."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-recovery-receipt'

    receipt_id: str
    case_receipt: ObjectIdentity
    recovery_index: ObjectIdentity
    output_materializations: tuple[ObjectIdentity, ...]
    source_output_set_sha256: str
    recovered_output_set_sha256: str
    catalog_loss_injected: bool
    executor_call_count: int
    scientific_recomputation_count: int
    scientific_retuning_count: int
    terminal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if (
            self.case_receipt.object_schema != ExecutableMetatheoryConformanceCaseReceipt.SCHEMA
            or self.recovery_index.object_schema != 'empirical-lawhood/runtime/run-recovery-reference'
        ):
            raise ValueError("metatheory recovery binds another route")
        require_sorted_unique_ids(
            self.output_materializations,
            attribute="object_id",
            field_name="output_materializations",
        )
        validate_sha256(self.source_output_set_sha256, field_name="source_output_set_sha256")
        validate_sha256(
            self.recovered_output_set_sha256,
            field_name="recovered_output_set_sha256",
        )
        if (
            not self.output_materializations
            or self.source_output_set_sha256 != self.recovered_output_set_sha256
            or not self.catalog_loss_injected
            or self.executor_call_count != 0
            or self.scientific_recomputation_count != 0
            or self.scientific_retuning_count != 0
            or not self.terminal
        ):
            raise ValueError("metatheory total-loss recovery changed science")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryConformanceAttemptStop(CanonicalRecord):
    """Honest terminal for an issued suite attempt stopped by harness failure."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-conformance-attempt-stop'

    attempt_id: str
    source_closure: ImplementationSourceClosure
    readiness: ObjectIdentity
    completed_case_kind: MetatheoryConformanceCase
    issued_package: ObjectIdentity
    run_plan: ObjectIdentity
    execution_plan: ObjectIdentity
    recovery_index: ObjectIdentity
    task_receipts: tuple[ObjectIdentity, ...]
    terminal_outputs: tuple[ObjectIdentity, ...]
    adjudication: ObjectIdentity
    reason_codes: tuple[str, ...]
    supported_case_completed: bool
    catalog_loss_recovery_completed: bool
    suite_published: bool
    new_issue_requires_owner_direction: bool
    created_empirical_evidence: bool
    stopped_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.attempt_id, field_name="attempt_id")
        parse_utc_timestamp(self.stopped_at_utc, field_name="stopped_at_utc")
        require_sorted_unique_ids(
            self.task_receipts,
            attribute="object_id",
            field_name="task_receipts",
        )
        require_sorted_unique_ids(
            self.terminal_outputs,
            attribute="object_id",
            field_name="terminal_outputs",
        )
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        expected_schemas = (
            (self.readiness, ExecutableMetatheoryNoEffectReadiness.SCHEMA),
            (self.issued_package, 'empirical-lawhood/api/issued-compilation-package-reference'),
            (self.run_plan, 'empirical-lawhood/runtime/compiled-run-plan-reference'),
            (self.execution_plan, 'empirical-lawhood/runtime/frozen-compilation-execution-plan'),
            (self.recovery_index, 'empirical-lawhood/runtime/run-recovery-reference'),
            (self.adjudication, ScientificAdjudicationRecord.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("metatheory attempt stop binds another route")
        if (
            self.completed_case_kind is not MetatheoryConformanceCase.SUPPORTED
            or not self.task_receipts
            or not self.terminal_outputs
            or not self.supported_case_completed
            or not self.catalog_loss_recovery_completed
            or self.suite_published
            or not self.new_issue_requires_owner_direction
            or self.created_empirical_evidence
        ):
            raise ValueError("metatheory attempt stop misstates the interrupted attempt")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryConformanceSuite(CanonicalRecord):
    """Exact source-free public-route suite and recovery product."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-conformance-suite'

    suite_id: str
    route_kind: ExecutableMetatheoryRouteKind
    source_closure: ImplementationSourceClosure
    readiness: ExecutableMetatheoryNoEffectReadiness
    cases: tuple[ExecutableMetatheoryConformanceCaseReceipt, ...]
    recovery: ExecutableMetatheoryRecoveryReceipt
    authoritative_relative_root: str
    terminal: bool
    created_empirical_evidence: bool
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.suite_id, field_name="suite_id")
        validate_relative_locator(self.authoritative_relative_root)
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        if {value.case_kind.value for value in self.cases} != set(
            REQUIRED_EXECUTABLE_METATHEORY_CASES
        ):
            raise ValueError("metatheory conformance suite case roster is incomplete")
        compilation_ids = set(self.readiness.campaign_compilations)
        if any(value.campaign_compilation not in compilation_ids for value in self.cases):
            raise ValueError("metatheory suite compilation identity drifted")
        case_receipts = {ObjectIdentity.from_record(value.case_id, value) for value in self.cases}
        if self.recovery.case_receipt not in case_receipts:
            raise ValueError("metatheory recovery is outside the executed suite")
        if (
            self.source_closure != self.readiness.source_closure
            or not self.terminal
            or self.created_empirical_evidence
            or self.maximum_structural_evidence_ceiling
            is not MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("metatheory conformance suite overclaims its boundary")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryReleaseAuthority(CanonicalRecord):
    """Bounded owner authorization for publishing only this release product."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-release-authority'

    authority_id: str
    grantee_id: str
    source_closure: ImplementationSourceClosure
    conformance_suite: ObjectIdentity
    authorization_basis: str
    allows_release_issue: bool
    allows_external_publication: bool
    allows_source_contact: bool
    allows_scientific_execution: bool
    allows_protected_outcome_read: bool
    allows_actuation: bool
    allows_merge_or_push: bool
    issued_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.authority_id, field_name="authority_id")
        validate_stable_id(self.grantee_id, field_name="grantee_id")
        validate_nonempty(self.authorization_basis, field_name="authorization_basis")
        parse_utc_timestamp(self.issued_at_utc, field_name="issued_at_utc")
        if (
            self.conformance_suite.object_schema != ExecutableMetatheoryConformanceSuite.SCHEMA
            or not self.allows_release_issue
            or not self.allows_external_publication
            or self.allows_source_contact
            or self.allows_scientific_execution
            or self.allows_protected_outcome_read
            or self.allows_actuation
            or self.allows_merge_or_push
        ):
            raise ValueError("metatheory release authority exceeds owner direction")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryPlatformRelease(CanonicalRecord):
    """Clean-source EMPC implementation release; never empirical support."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-platform-release'

    release_id: str
    status: ExecutableMetatheoryReleaseStatus
    plan: ObjectIdentity
    architecture_decision: ObjectIdentity
    predecessor_terminals: tuple[ObjectIdentity, ...]
    source_closure: ImplementationSourceClosure
    repository_state: ReleaseRepositoryStateReceipt
    planning_runtime_schema_roster: tuple[str, ...]
    discovery_aggregate: ObjectIdentity
    executable_aggregate: ObjectIdentity
    config_decoders: tuple[ObjectIdentity, ...]
    artifact_validators: tuple[ObjectIdentity, ...]
    provider_factories: tuple[ObjectIdentity, ...]
    method_implementations: tuple[ObjectIdentity, ...]
    conformance_suite: ObjectIdentity
    recovery_receipt: ObjectIdentity
    release_authority: ExecutableMetatheoryReleaseAuthority
    verification_receipts: tuple[ReleaseVerificationReceipt, ...]
    maximum_claim: ConsolidationClaimCeiling
    limitations: tuple[str, ...]
    repository_contains_scientific_payloads: bool
    created_empirical_evidence: bool
    released_at_utc: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.release_id, field_name="release_id")
        parse_utc_timestamp(self.released_at_utc, field_name="released_at_utc")
        require_sorted_unique_ids(
            self.predecessor_terminals,
            attribute="object_id",
            field_name="predecessor_terminals",
        )
        if len(self.predecessor_terminals) != 2:
            raise ValueError("metatheory release requires door and Gym terminals")
        require_sorted_unique_strings(
            self.planning_runtime_schema_roster,
            field_name="planning_runtime_schema_roster",
            allow_empty=False,
        )
        for value in self.planning_runtime_schema_roster:
            validate_schema(value)
        for name in (
            "config_decoders",
            "artifact_validators",
            "provider_factories",
            "method_implementations",
        ):
            require_sorted_unique_ids(
                getattr(self, name),
                attribute="object_id",
                field_name=name,
            )
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        require_sorted_unique_ids(
            self.verification_receipts,
            attribute="receipt_id",
            field_name="verification_receipts",
        )
        require_sorted_unique_strings(self.limitations, field_name="limitations", allow_empty=False)
        expected_schemas = (
            (self.discovery_aggregate, GeneratedExtensionBundleAggregate.SCHEMA),
            (self.executable_aggregate, GeneratedExecutableBindingAggregate.SCHEMA),
            (self.conformance_suite, ExecutableMetatheoryConformanceSuite.SCHEMA),
            (self.recovery_receipt, ExecutableMetatheoryRecoveryReceipt.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("metatheory release binds another implementation route")
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or self.repository_state.implementation_commit
            != self.source_closure.implementation_commit
            or self.repository_state.source_tree_sha256 != self.source_closure.source_tree_sha256
            or self.release_authority.source_closure != self.source_closure
            or self.release_authority.conformance_suite != self.conformance_suite
            or any(
                value.implementation_commit != self.source_closure.implementation_commit
                for value in self.verification_receipts
            )
            or not self.verification_receipts
            or self.maximum_claim is not ConsolidationClaimCeiling.ARCHITECTURE_CONFORMANCE_ONLY
            or self.repository_contains_scientific_payloads
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("metatheory platform release overclaims its boundary")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryConsumerPin(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-consumer-pin'

    pin_id: str
    release: ObjectIdentity
    source_closure: ImplementationSourceClosure
    allowed_consumer_ids: tuple[str, ...]
    maximum_claim: ConsolidationClaimCeiling
    empirical_promotion_permitted: bool
    terminal: bool
    pinned_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.pin_id, field_name="pin_id")
        parse_utc_timestamp(self.pinned_at_utc, field_name="pinned_at_utc")
        require_sorted_unique_strings(
            self.allowed_consumer_ids,
            field_name="allowed_consumer_ids",
            allow_empty=False,
        )
        if (
            self.release.object_schema != ExecutableMetatheoryPlatformRelease.SCHEMA
            or self.maximum_claim is not ConsolidationClaimCeiling.ARCHITECTURE_CONFORMANCE_ONLY
            or self.empirical_promotion_permitted
            or not self.terminal
        ):
            raise ValueError("metatheory consumer pin widens the release")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryExecutedRouteAttestation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-executed-route-attestation'

    attestation_id: str
    release: ObjectIdentity
    consumer_pin: ObjectIdentity
    conformance_suite: ObjectIdentity
    executed_case_receipts: tuple[ObjectIdentity, ...]
    recovery_receipt: ObjectIdentity
    source_closure: ImplementationSourceClosure
    route_kind: ExecutableMetatheoryRouteKind
    terminal: bool
    created_empirical_evidence: bool
    attested_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.attestation_id, field_name="attestation_id")
        parse_utc_timestamp(self.attested_at_utc, field_name="attested_at_utc")
        require_sorted_unique_ids(
            self.executed_case_receipts,
            attribute="object_id",
            field_name="executed_case_receipts",
        )
        expected_schemas = (
            (self.release, ExecutableMetatheoryPlatformRelease.SCHEMA),
            (self.consumer_pin, ExecutableMetatheoryConsumerPin.SCHEMA),
            (self.conformance_suite, ExecutableMetatheoryConformanceSuite.SCHEMA),
            (self.recovery_receipt, ExecutableMetatheoryRecoveryReceipt.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("metatheory attestation binds another release route")
        if (
            len(self.executed_case_receipts) != len(REQUIRED_EXECUTABLE_METATHEORY_CASES)
            or any(
                value.object_schema != ExecutableMetatheoryConformanceCaseReceipt.SCHEMA
                for value in self.executed_case_receipts
            )
            or not self.terminal
            or self.created_empirical_evidence
        ):
            raise ValueError("metatheory attestation is incomplete or overclaims")


def decode_executable_metatheory_suite(
    payload: bytes,
) -> ExecutableMetatheoryConformanceSuite:
    return decode_canonical_bytes(
        payload,
        ExecutableMetatheoryConformanceSuite,
        maximum_bytes=max(len(payload), 1),
    )


def decode_executable_metatheory_attempt_stop(
    payload: bytes,
) -> ExecutableMetatheoryConformanceAttemptStop:
    return decode_canonical_bytes(
        payload,
        ExecutableMetatheoryConformanceAttemptStop,
        maximum_bytes=max(len(payload), 1),
    )


def decode_executable_metatheory_release(
    payload: bytes,
) -> ExecutableMetatheoryPlatformRelease:
    return decode_canonical_bytes(
        payload,
        ExecutableMetatheoryPlatformRelease,
        maximum_bytes=max(len(payload), 1),
    )


def decode_executable_metatheory_pin(
    payload: bytes,
) -> ExecutableMetatheoryConsumerPin:
    return decode_canonical_bytes(
        payload,
        ExecutableMetatheoryConsumerPin,
        maximum_bytes=max(len(payload), 1),
    )


def decode_executable_metatheory_attestation(
    payload: bytes,
) -> ExecutableMetatheoryExecutedRouteAttestation:
    return decode_canonical_bytes(
        payload,
        ExecutableMetatheoryExecutedRouteAttestation,
        maximum_bytes=max(len(payload), 1),
    )


def validate_exact_git_commit(value: str, *, field_name: str) -> None:
    """Shared release-generator guard for source identities."""

    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise ValueError(f"{field_name} is not an exact Git SHA-1")


__all__ = [
    'ExecutableMetatheoryConformanceCaseReceipt',
    'ExecutableMetatheoryConformanceAttemptStop',
    'ExecutableMetatheoryConformanceSuite',
    'ExecutableMetatheoryConsumerPin',
    'ExecutableMetatheoryExecutedRouteAttestation',
    'ExecutableMetatheoryNoEffectReadiness',
    'ExecutableMetatheoryPlatformRelease',
    'ExecutableMetatheoryRecoveryReceipt',
    'ExecutableMetatheoryReleaseAuthority',
    'ExecutableMetatheoryReleaseStatus',
    'ExecutableMetatheoryRouteKind',
    'MetatheoryConformanceCase',
    "EXECUTABLE_METATHEORY_RELEASE_SCHEMA_ROSTER",
    "REQUIRED_EXECUTABLE_METATHEORY_CASES",
    'decode_executable_metatheory_attestation',
    'decode_executable_metatheory_attempt_stop',
    'decode_executable_metatheory_pin',
    'decode_executable_metatheory_release',
    'decode_executable_metatheory_suite',
    "validate_exact_git_commit",
]
