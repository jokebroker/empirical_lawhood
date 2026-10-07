"Claim-limited corrective release records for the source-free public route.\n\nRetained executable-metatheory records remain decode-only historical evidence.\nThese corrective records bind the real-owner source-free public route. They do\nnot grant experiment, source, reveal, merge or actuation authority, and they do\nnot promote contract conformance to empirical science.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
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
from empirical_lawhood.runtime.artifacts import CanonicalTaskReceipt
from empirical_lawhood.runtime.executable_bindings import GeneratedExecutableBindingAggregate
from empirical_lawhood.runtime.extension_bundles import GeneratedExtensionBundleAggregate
from empirical_lawhood.runtime.issued_extension_payloads import ExecutableProviderReconstructionReceipt, ExecutableProviderRecoveryReceipt
from empirical_lawhood.runtime.metatheory_campaigns import MetatheoryStageOwnerContract
from empirical_lawhood.runtime.response_law_release import (
    ConsolidationClaimCeiling,
    ReleaseRepositoryStateReceipt,
    ReleaseVerificationReceipt,
)


REQUIRED_EXECUTABLE_METATHEORY_CORRECTIVE_CASES = tuple(
    sorted(
        (
            "ATLAS_LOCALIZATION_TOO_WIDE",
            "AUTHORITY_STOP",
            "COLLISION_ABSENT_UNEVALUABLE",
            "MISSING_PERMITTED_TARGET",
            "MIXED_PREDICTIVE_LEVELS",
            "OPPOSED_PROPERTY_DECISION",
            "PREREQUISITE_NONATTEMPT",
            "SUPPORTED",
        )
    )
)


class ExecutableMetatheoryCorrectiveReleaseStatus(StrEnum):
    EXECUTABLE_METATHEORY_CORRECTIVE_COMPLETE = "EXECUTABLE_METATHEORY_CORRECTIVE_COMPLETE"


class ExecutableMetatheoryPredecessorDisposition(StrEnum):
    PLUMBING_AND_RECOVERY_ONLY = "PLUMBING_AND_RECOVERY_ONLY"


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryImplementationPathEntry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-implementation-path-entry'

    relative_path: str
    content_sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        validate_relative_locator(self.relative_path)
        validate_sha256(self.content_sha256, field_name="content_sha256")
        if self.size_bytes < 0:
            raise ValueError("implementation path size must be nonnegative")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryImplementationManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-implementation-manifest'

    manifest_id: str
    source_closure: ImplementationSourceClosure
    entries: tuple[ExecutableMetatheoryImplementationPathEntry, ...]
    total_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        paths = tuple(value.relative_path for value in self.entries)
        if tuple(sorted(set(paths))) != paths or not paths:
            raise ValueError("implementation manifest paths must be sorted and unique")
        if self.total_bytes != sum(value.size_bytes for value in self.entries):
            raise ValueError("implementation manifest byte total differs")
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
        ):
            raise ValueError("implementation manifest requires a clean source closure")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryCorrectiveNoEffectReadiness(CanonicalRecord):
    """One complete, preissue, effect-free proof over the corrected topology."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/runtime/executable-metatheory-corrective-no-effect-readiness'
    )

    witness_id: str
    source_closure: ImplementationSourceClosure
    implementation_manifest: ObjectIdentity
    case_ids: tuple[str, ...]
    campaign_compilations: tuple[ObjectIdentity, ...]
    candidate_compilations: tuple[ObjectIdentity, ...]
    capability_registries: tuple[ObjectIdentity, ...]
    owner_contracts: tuple[ObjectIdentity, ...]
    discovery_aggregate: ObjectIdentity
    executable_aggregate: ObjectIdentity
    provider_binding_ids: tuple[str, ...]
    decoder_registration_ids: tuple[str, ...]
    artifact_validator_ids: tuple[str, ...]
    output_schema_ids: tuple[str, ...]
    output_relative_locator_templates: tuple[str, ...]
    adjudication_relative_locator_templates: tuple[str, ...]
    authority_input_kinds: tuple[str, ...]
    external_root_contract: ObjectIdentity
    catalog_relative_locator: str
    task_count: int
    external_input_count: int
    total_source_byte_limit: int
    total_output_byte_limit: int
    total_memory_byte_limit: int
    control_setup_write_count: int
    external_namespace_setup_write_count: int
    source_contact_count: int
    target_contact_count: int
    protected_outcome_read_count: int
    network_request_count: int
    immutable_issue_count: int
    reveal_count: int
    external_artifact_write_count: int
    catalog_mutation_count: int
    scientific_effect_count: int
    counter_scope: str
    artifact_persistence_ready: bool
    control_record_persistence_ready: bool
    provider_recovery_ready: bool
    ready: bool
    created_empirical_evidence: bool
    observed_at_utc: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.witness_id, field_name="witness_id")
        parse_utc_timestamp(self.observed_at_utc, field_name="observed_at_utc")
        validate_nonempty(self.counter_scope, field_name="counter_scope")
        require_sorted_unique_strings(self.case_ids, field_name="case_ids", allow_empty=False)
        if self.case_ids != REQUIRED_EXECUTABLE_METATHEORY_CORRECTIVE_CASES:
            raise ValueError("corrective readiness case roster differs")
        for name in (
            "campaign_compilations",
            "candidate_compilations",
            "capability_registries",
            "owner_contracts",
        ):
            require_sorted_unique_ids(
                getattr(self, name),
                attribute="object_id",
                field_name=name,
            )
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        for name in (
            "provider_binding_ids",
            "decoder_registration_ids",
            "artifact_validator_ids",
            "output_schema_ids",
            "output_relative_locator_templates",
            "adjudication_relative_locator_templates",
            "authority_input_kinds",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        for value in self.output_schema_ids:
            validate_schema(value)
        for name in (
            "output_relative_locator_templates",
            "adjudication_relative_locator_templates",
        ):
            for value in getattr(self, name):
                validate_relative_locator(value)
        validate_relative_locator(self.catalog_relative_locator)
        expected_schemas = (
            (
                self.implementation_manifest,
                ExecutableMetatheoryImplementationManifest.SCHEMA,
            ),
            (self.discovery_aggregate, GeneratedExtensionBundleAggregate.SCHEMA),
            (self.executable_aggregate, GeneratedExecutableBindingAggregate.SCHEMA),
            (self.external_root_contract, 'empirical-lawhood/runtime/external-root-contract'),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("corrective readiness binds another topology")
        positive_totals = (
            self.task_count,
            self.external_input_count,
            self.total_source_byte_limit,
            self.total_output_byte_limit,
            self.total_memory_byte_limit,
        )
        if any(type(value) is not int or value <= 0 for value in positive_totals):
            raise ValueError("corrective readiness requires positive derived totals")
        if self.control_setup_write_count < 0 or self.external_namespace_setup_write_count < 0:
            raise ValueError("control setup write counts must be nonnegative")
        effect_counts = (
            self.source_contact_count,
            self.target_contact_count,
            self.protected_outcome_read_count,
            self.network_request_count,
            self.immutable_issue_count,
            self.reveal_count,
            self.external_artifact_write_count,
            self.catalog_mutation_count,
            self.scientific_effect_count,
        )
        if any(type(value) is not int or value != 0 for value in effect_counts):
            raise ValueError("corrective readiness proof had a prohibited effect")
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or len(self.campaign_compilations) != len(self.case_ids)
            or len(self.candidate_compilations) != len(self.case_ids)
            or not self.artifact_persistence_ready
            or not self.control_record_persistence_ready
            or not self.provider_recovery_ready
            or not self.ready
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("corrective readiness misses or exceeds its boundary")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryStageOwnerEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-stage-owner-evidence'

    evidence_id: str
    task_id: str
    task_receipt: ObjectIdentity
    capability_key: str
    capability_version: str
    config: ObjectIdentity
    canonical_owner: ObjectIdentity
    owner_implementation_sha256: str
    input_materialization_ids: tuple[str, ...]
    output_products: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        for name in ("evidence_id", "task_id", "capability_key"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(
            self.owner_implementation_sha256,
            field_name="owner_implementation_sha256",
        )
        if self.task_receipt.object_schema != CanonicalTaskReceipt.SCHEMA:
            raise ValueError("stage owner evidence binds another receipt schema")
        if self.config.object_schema != 'empirical-lawhood/runtime/capability-config-ref':
            raise ValueError("stage owner evidence binds another config schema")
        require_sorted_unique_strings(
            self.input_materialization_ids,
            field_name="input_materialization_ids",
        )
        require_sorted_unique_ids(
            self.output_products,
            attribute="object_id",
            field_name="output_products",
        )
        if not self.output_products:
            raise ValueError("stage owner evidence lacks a canonical product")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryCorrectiveReadinessStop(CanonicalRecord):
    """Immutable readiness stop for one rejected corrective source identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-corrective-readiness-stop'

    stop_id: str
    source_closure: ImplementationSourceClosure
    failed_section: str
    reason_codes: tuple[str, ...]
    failure_diagnostic_sha256: str
    proof_body_entered: bool
    source_contact_count: int
    target_contact_count: int
    protected_outcome_read_count: int
    network_request_count: int
    immutable_issue_count: int
    reveal_count: int
    external_artifact_write_count: int
    catalog_mutation_count: int
    scientific_effect_count: int
    external_namespace_setup_write_count: int
    stop_publication_external_control_write_count: int
    readiness_published: bool
    new_source_required: bool
    created_empirical_evidence: bool
    stopped_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.stop_id, field_name="stop_id")
        validate_nonempty(self.failed_section, field_name="failed_section")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        validate_sha256(
            self.failure_diagnostic_sha256,
            field_name="failure_diagnostic_sha256",
        )
        parse_utc_timestamp(self.stopped_at_utc, field_name="stopped_at_utc")
        prohibited_counts = (
            self.source_contact_count,
            self.target_contact_count,
            self.protected_outcome_read_count,
            self.network_request_count,
            self.immutable_issue_count,
            self.reveal_count,
            self.external_artifact_write_count,
            self.catalog_mutation_count,
            self.scientific_effect_count,
        )
        if any(type(value) is not int or value != 0 for value in prohibited_counts):
            raise ValueError("corrective readiness stop had a prohibited effect")
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or self.external_namespace_setup_write_count < 0
            or self.stop_publication_external_control_write_count != 3
            or self.readiness_published
            or not self.new_source_required
            or self.created_empirical_evidence
        ):
            raise ValueError("corrective readiness stop misstates the failed checkpoint")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryCorrectiveCaseReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-corrective-case-receipt'

    case_id: str
    case_kind: str
    campaign_compilation: ObjectIdentity
    base_issue: ObjectIdentity
    base_issue_publication: ObjectIdentity
    extension_issue: ObjectIdentity
    extension_issue_publication: ObjectIdentity
    issued_package: ObjectIdentity | None
    run_plan: ObjectIdentity | None
    execution_plan: ObjectIdentity | None
    execution_resource_envelope: ObjectIdentity | None
    provider_reconstruction: ObjectIdentity | None
    recovery_index: ObjectIdentity | None
    stage_owner_evidence: tuple[ExecutableMetatheoryStageOwnerEvidence, ...]
    adjudication: ObjectIdentity | None
    pre_reveal_public_status: str
    pre_reveal_reason_codes: tuple[str, ...]
    operational_status: str
    scientific_status: ScientificStatus | None
    reason_codes: tuple[str, ...]
    target_contact_count: int
    reveal_count: int
    terminal: bool
    created_empirical_evidence: bool
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        validate_nonempty(self.case_kind, field_name="case_kind")
        if self.case_kind not in REQUIRED_EXECUTABLE_METATHEORY_CORRECTIVE_CASES:
            raise ValueError("unknown corrective case label")
        validate_nonempty(self.operational_status, field_name="operational_status")
        validate_nonempty(
            self.pre_reveal_public_status,
            field_name="pre_reveal_public_status",
        )
        require_sorted_unique_strings(
            self.pre_reveal_reason_codes,
            field_name="pre_reveal_reason_codes",
        )
        require_sorted_unique_ids(
            self.stage_owner_evidence,
            attribute="evidence_id",
            field_name="stage_owner_evidence",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        for name in ("target_contact_count", "reveal_count"):
            if getattr(self, name) not in {0, 1}:
                raise ValueError("corrective case contact count is invalid")
        authority_stop = self.case_kind == "AUTHORITY_STOP"
        optional_route_records = (
            self.issued_package,
            self.run_plan,
            self.execution_plan,
            self.execution_resource_envelope,
            self.provider_reconstruction,
            self.recovery_index,
        )
        if authority_stop != all(value is None for value in optional_route_records):
            raise ValueError("authority-stop route state differs")
        if not authority_stop and any(value is None for value in optional_route_records):
            raise ValueError("executed corrective case lacks a route identity")
        if authority_stop != (self.adjudication is None):
            raise ValueError("authority-stop adjudication state differs")
        if authority_stop != (self.scientific_status is None):
            raise ValueError("authority-stop scientific status differs")
        if authority_stop and (
            self.stage_owner_evidence or self.target_contact_count or self.reveal_count
        ):
            raise ValueError("authority-stop crossed execution")
        if not authority_stop and (
            not self.stage_owner_evidence
            or self.issued_package is None
            or self.adjudication is None
            or self.scientific_status is None
        ):
            raise ValueError("executed corrective case lacks terminal owner evidence")
        if (
            not self.terminal
            or self.created_empirical_evidence
            or self.maximum_structural_evidence_ceiling
            is not MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE
        ):
            raise ValueError("corrective case overclaims its evidence")
        expected = (
            (
                self.campaign_compilation,
                'empirical-lawhood/runtime/metatheory-campaign-compilation',
            ),
            (self.base_issue, 'empirical-lawhood/runtime/issued-study-manifest'),
            (
                self.base_issue_publication,
                'empirical-lawhood/runtime/study-publication-receipt',
            ),
            (self.extension_issue, 'empirical-lawhood/runtime/issued-executable-study-manifest'),
            (
                self.extension_issue_publication,
                'empirical-lawhood/runtime/extension-publication-receipt',
            ),
        )
        if any(value.object_schema != schema for value, schema in expected):
            raise ValueError("corrective case issue lineage differs")
        if not authority_stop:
            assert self.issued_package is not None
            assert self.run_plan is not None
            assert self.execution_plan is not None
            assert self.execution_resource_envelope is not None
            assert self.provider_reconstruction is not None
            assert self.recovery_index is not None
            assert self.adjudication is not None
            executed_expected = (
                (self.issued_package, 'empirical-lawhood/api/issued-compilation-package-reference'),
                (self.run_plan, 'empirical-lawhood/runtime/compiled-run-plan-reference'),
                (self.execution_plan, 'empirical-lawhood/runtime/frozen-compilation-execution-plan'),
                (
                    self.execution_resource_envelope,
                    'empirical-lawhood/runtime/compiled-resource-envelope-reference',
                ),
                (
                    self.provider_reconstruction,
                    ExecutableProviderReconstructionReceipt.SCHEMA,
                ),
                (self.recovery_index, 'empirical-lawhood/runtime/run-recovery-reference'),
                (self.adjudication, 'empirical-lawhood/runtime/scientific-adjudication-record'),
            )
            if any(value.object_schema != schema for value, schema in executed_expected):
                raise ValueError("corrective case executed-route lineage differs")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryCorrectiveAdversarialReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-corrective-adversarial-receipt'

    receipt_id: str
    nominal_campaign_compilation: ObjectIdentity
    substituted_payload_sha256: str
    substituted_issued_payload_reason_codes: tuple[str, ...]
    sealed_custody_reason_codes: tuple[str, ...]
    case_label_invariant: bool
    sealed_custody_run_id: str
    sealed_stage_receipts: tuple[ObjectIdentity, ...]
    sealed_reveal_receipt_count: int
    created_empirical_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.sealed_custody_run_id, field_name="sealed_custody_run_id")
        validate_sha256(
            self.substituted_payload_sha256,
            field_name="substituted_payload_sha256",
        )
        if (
            self.nominal_campaign_compilation.object_schema
            != 'empirical-lawhood/runtime/metatheory-campaign-compilation'
        ):
            raise ValueError("corrective adversary binds another campaign compilation")
        require_sorted_unique_ids(
            self.sealed_stage_receipts,
            attribute="object_id",
            field_name="sealed_stage_receipts",
        )
        for name in (
            "substituted_issued_payload_reason_codes",
            "sealed_custody_reason_codes",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        if (
            not self.case_label_invariant
            or not self.sealed_stage_receipts
            or self.sealed_reveal_receipt_count != 0
            or self.created_empirical_evidence
        ):
            raise ValueError("corrective adversarial receipt did not fail closed")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryCorrectiveRecoveryReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-corrective-recovery-receipt'

    receipt_id: str
    supported_case_receipt: ObjectIdentity
    provider_reconstruction: ObjectIdentity
    provider_recovery: ObjectIdentity
    recovery_index: ObjectIdentity
    task_receipts: tuple[ObjectIdentity, ...]
    source_output_set_sha256: str
    recovered_output_set_sha256: str
    catalog_loss_injected: bool
    executor_call_count: int
    scientific_recomputation_count: int
    scientific_retuning_count: int
    source_or_target_reacquisition_count: int
    terminal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        expected = (
            (
                self.supported_case_receipt,
                ExecutableMetatheoryCorrectiveCaseReceipt.SCHEMA,
            ),
            (
                self.provider_reconstruction,
                ExecutableProviderReconstructionReceipt.SCHEMA,
            ),
            (self.provider_recovery, ExecutableProviderRecoveryReceipt.SCHEMA),
            (self.recovery_index, 'empirical-lawhood/runtime/run-recovery-reference'),
        )
        if any(value.object_schema != schema for value, schema in expected):
            raise ValueError("corrective recovery binds another route")
        require_sorted_unique_ids(
            self.task_receipts,
            attribute="object_id",
            field_name="task_receipts",
        )
        validate_sha256(self.source_output_set_sha256, field_name="source_output_set_sha256")
        validate_sha256(
            self.recovered_output_set_sha256,
            field_name="recovered_output_set_sha256",
        )
        if (
            not self.task_receipts
            or self.source_output_set_sha256 != self.recovered_output_set_sha256
            or not self.catalog_loss_injected
            or any(
                value != 0
                for value in (
                    self.executor_call_count,
                    self.scientific_recomputation_count,
                    self.scientific_retuning_count,
                    self.source_or_target_reacquisition_count,
                )
            )
            or not self.terminal
        ):
            raise ValueError("corrective recovery changed scientific work")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryCorrectiveSuite(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-corrective-suite'

    suite_id: str
    source_closure: ImplementationSourceClosure
    readiness: ObjectIdentity
    cases: tuple[ExecutableMetatheoryCorrectiveCaseReceipt, ...]
    adversarial: ExecutableMetatheoryCorrectiveAdversarialReceipt
    recovery: ExecutableMetatheoryCorrectiveRecoveryReceipt
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
        if {value.case_kind for value in self.cases} != set(
            REQUIRED_EXECUTABLE_METATHEORY_CORRECTIVE_CASES
        ):
            raise ValueError("corrective suite case roster is incomplete")
        case_ids = {ObjectIdentity.from_record(value.case_id, value) for value in self.cases}
        if self.recovery.supported_case_receipt not in case_ids:
            raise ValueError("corrective recovery is outside the suite")
        if (
            self.readiness.object_schema != ExecutableMetatheoryCorrectiveNoEffectReadiness.SCHEMA
            or not self.terminal
            or self.created_empirical_evidence
            or self.maximum_structural_evidence_ceiling
            is not MetatheoryEvidenceCeiling.CONTRACT_CONFORMANCE
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("corrective suite overclaims its boundary")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryCorrectiveAttemptStop(CanonicalRecord):
    """Immutable stop for one issued corrective-suite attempt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-corrective-attempt-stop'

    attempt_id: str
    source_closure: ImplementationSourceClosure
    readiness: ObjectIdentity
    completed_cases: tuple[ExecutableMetatheoryCorrectiveCaseReceipt, ...]
    failed_case_kind: str
    reason_codes: tuple[str, ...]
    failure_diagnostic_sha256: str
    suite_published: bool
    new_issue_requires_owner_direction: bool
    created_empirical_evidence: bool
    stopped_at_utc: str

    def __post_init__(self) -> None:
        validate_stable_id(self.attempt_id, field_name="attempt_id")
        parse_utc_timestamp(self.stopped_at_utc, field_name="stopped_at_utc")
        validate_sha256(
            self.failure_diagnostic_sha256,
            field_name="failure_diagnostic_sha256",
        )
        validate_nonempty(self.failed_case_kind, field_name="failed_case_kind")
        if self.failed_case_kind not in REQUIRED_EXECUTABLE_METATHEORY_CORRECTIVE_CASES:
            raise ValueError("corrective attempt stop names another case")
        require_sorted_unique_ids(
            self.completed_cases,
            attribute="case_id",
            field_name="completed_cases",
        )
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if (
            self.readiness.object_schema != ExecutableMetatheoryCorrectiveNoEffectReadiness.SCHEMA
            or self.suite_published
            or not self.new_issue_requires_owner_direction
            or self.created_empirical_evidence
        ):
            raise ValueError("corrective attempt stop does not preserve the failure")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryCorrectiveReleaseAuthority(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-corrective-release-authority'

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
            self.conformance_suite.object_schema != ExecutableMetatheoryCorrectiveSuite.SCHEMA
            or not self.allows_release_issue
            or not self.allows_external_publication
            or self.allows_source_contact
            or self.allows_scientific_execution
            or self.allows_protected_outcome_read
            or self.allows_actuation
            or self.allows_merge_or_push
        ):
            raise ValueError("corrective release authority exceeds its scope")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryCorrectivePlatformRelease(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-corrective-platform-release'

    release_id: str
    status: ExecutableMetatheoryCorrectiveReleaseStatus
    source_closure: ImplementationSourceClosure
    implementation_manifest: ObjectIdentity
    repository_state: ReleaseRepositoryStateReceipt
    controlling_plan: ObjectIdentity
    predecessor_release: ObjectIdentity
    predecessor_disposition: ExecutableMetatheoryPredecessorDisposition
    planning_runtime_schema_roster: tuple[str, ...]
    discovery_aggregate: ObjectIdentity
    executable_aggregate: ObjectIdentity
    owner_contracts: tuple[ObjectIdentity, ...]
    conformance_suite: ObjectIdentity
    recovery_receipt: ObjectIdentity
    release_authority: ExecutableMetatheoryCorrectiveReleaseAuthority
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
        require_sorted_unique_strings(
            self.planning_runtime_schema_roster,
            field_name="planning_runtime_schema_roster",
            allow_empty=False,
        )
        for value in self.planning_runtime_schema_roster:
            validate_schema(value)
        require_sorted_unique_ids(
            self.owner_contracts,
            attribute="object_id",
            field_name="owner_contracts",
        )
        require_sorted_unique_ids(
            self.verification_receipts,
            attribute="receipt_id",
            field_name="verification_receipts",
        )
        require_sorted_unique_strings(self.limitations, field_name="limitations", allow_empty=False)
        expected = (
            (
                self.implementation_manifest,
                ExecutableMetatheoryImplementationManifest.SCHEMA,
            ),
            (self.discovery_aggregate, GeneratedExtensionBundleAggregate.SCHEMA),
            (self.executable_aggregate, GeneratedExecutableBindingAggregate.SCHEMA),
            (self.conformance_suite, ExecutableMetatheoryCorrectiveSuite.SCHEMA),
            (self.recovery_receipt, ExecutableMetatheoryCorrectiveRecoveryReceipt.SCHEMA),
            (self.controlling_plan, 'empirical-lawhood/document/implementation-specification'),
            (
                self.predecessor_release,
                'empirical-lawhood/runtime/executable-metatheory-platform-release',
            ),
        )
        if any(value.object_schema != schema for value, schema in expected):
            raise ValueError("corrective release binds another implementation route")
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
            or not self.owner_contracts
            or any(
                value.object_schema != MetatheoryStageOwnerContract.SCHEMA
                for value in self.owner_contracts
            )
            or not self.verification_receipts
            or self.maximum_claim is not ConsolidationClaimCeiling.ARCHITECTURE_CONFORMANCE_ONLY
            or self.repository_contains_scientific_payloads
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("corrective platform release overclaims its boundary")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryCorrectiveConsumerPin(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-corrective-consumer-pin'

    pin_id: str
    release: ObjectIdentity
    source_closure: ImplementationSourceClosure
    allowed_consumer_ids: tuple[str, ...]
    maximum_claim: ConsolidationClaimCeiling
    empirical_promotion_permitted: bool
    automatic_issue_permitted: bool
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
            self.release.object_schema != ExecutableMetatheoryCorrectivePlatformRelease.SCHEMA
            or self.maximum_claim is not ConsolidationClaimCeiling.ARCHITECTURE_CONFORMANCE_ONLY
            or self.empirical_promotion_permitted
            or self.automatic_issue_permitted
            or not self.terminal
        ):
            raise ValueError("corrective consumer pin widens the release")


@dataclass(frozen=True, slots=True)
class ExecutableMetatheoryCorrectiveExecutedRouteAttestation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/executable-metatheory-corrective-executed-route-attestation'

    attestation_id: str
    release: ObjectIdentity
    consumer_pin: ObjectIdentity
    conformance_suite: ObjectIdentity
    executed_case_receipts: tuple[ObjectIdentity, ...]
    recovery_receipt: ObjectIdentity
    source_closure: ImplementationSourceClosure
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
        expected = (
            (self.release, ExecutableMetatheoryCorrectivePlatformRelease.SCHEMA),
            (self.consumer_pin, ExecutableMetatheoryCorrectiveConsumerPin.SCHEMA),
            (self.conformance_suite, ExecutableMetatheoryCorrectiveSuite.SCHEMA),
            (self.recovery_receipt, ExecutableMetatheoryCorrectiveRecoveryReceipt.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected):
            raise ValueError("corrective attestation binds another route")
        if (
            len(self.executed_case_receipts)
            != len(REQUIRED_EXECUTABLE_METATHEORY_CORRECTIVE_CASES)
            or any(
                value.object_schema != ExecutableMetatheoryCorrectiveCaseReceipt.SCHEMA
                for value in self.executed_case_receipts
            )
            or not self.terminal
            or self.created_empirical_evidence
        ):
            raise ValueError("corrective attestation is incomplete or overclaims")


def _decode(payload: bytes, record_type: type[CanonicalRecord]) -> CanonicalRecord:
    return decode_canonical_bytes(payload, record_type, maximum_bytes=max(len(payload), 1))


def decode_executable_metatheory_corrective_readiness(
    payload: bytes,
) -> ExecutableMetatheoryCorrectiveNoEffectReadiness:
    value = _decode(payload, ExecutableMetatheoryCorrectiveNoEffectReadiness)
    assert isinstance(value, ExecutableMetatheoryCorrectiveNoEffectReadiness)
    return value


def decode_executable_metatheory_corrective_readiness_stop(
    payload: bytes,
) -> ExecutableMetatheoryCorrectiveReadinessStop:
    value = _decode(payload, ExecutableMetatheoryCorrectiveReadinessStop)
    assert isinstance(value, ExecutableMetatheoryCorrectiveReadinessStop)
    return value


def decode_executable_metatheory_corrective_suite(
    payload: bytes,
) -> ExecutableMetatheoryCorrectiveSuite:
    value = _decode(payload, ExecutableMetatheoryCorrectiveSuite)
    assert isinstance(value, ExecutableMetatheoryCorrectiveSuite)
    return value


def decode_executable_metatheory_corrective_release(
    payload: bytes,
) -> ExecutableMetatheoryCorrectivePlatformRelease:
    value = _decode(payload, ExecutableMetatheoryCorrectivePlatformRelease)
    assert isinstance(value, ExecutableMetatheoryCorrectivePlatformRelease)
    return value


def decode_executable_metatheory_corrective_pin(
    payload: bytes,
) -> ExecutableMetatheoryCorrectiveConsumerPin:
    value = _decode(payload, ExecutableMetatheoryCorrectiveConsumerPin)
    assert isinstance(value, ExecutableMetatheoryCorrectiveConsumerPin)
    return value


def decode_executable_metatheory_corrective_attestation(
    payload: bytes,
) -> ExecutableMetatheoryCorrectiveExecutedRouteAttestation:
    value = _decode(payload, ExecutableMetatheoryCorrectiveExecutedRouteAttestation)
    assert isinstance(value, ExecutableMetatheoryCorrectiveExecutedRouteAttestation)
    return value


__all__ = [
    "REQUIRED_EXECUTABLE_METATHEORY_CORRECTIVE_CASES",
    'ExecutableMetatheoryCorrectiveAdversarialReceipt',
    'ExecutableMetatheoryCorrectiveAttemptStop',
    'ExecutableMetatheoryCorrectiveCaseReceipt',
    'ExecutableMetatheoryCorrectiveConsumerPin',
    'ExecutableMetatheoryCorrectiveExecutedRouteAttestation',
    'ExecutableMetatheoryCorrectiveNoEffectReadiness',
    'ExecutableMetatheoryCorrectiveReadinessStop',
    'ExecutableMetatheoryCorrectivePlatformRelease',
    'ExecutableMetatheoryCorrectiveRecoveryReceipt',
    'ExecutableMetatheoryCorrectiveReleaseAuthority',
    'ExecutableMetatheoryCorrectiveReleaseStatus',
    'ExecutableMetatheoryCorrectiveSuite',
    'ExecutableMetatheoryImplementationManifest',
    'ExecutableMetatheoryImplementationPathEntry',
    'ExecutableMetatheoryPredecessorDisposition',
    'ExecutableMetatheoryStageOwnerEvidence',
    'decode_executable_metatheory_corrective_attestation',
    'decode_executable_metatheory_corrective_pin',
    'decode_executable_metatheory_corrective_readiness',
    'decode_executable_metatheory_corrective_readiness_stop',
    'decode_executable_metatheory_corrective_release',
    'decode_executable_metatheory_corrective_suite',
]
