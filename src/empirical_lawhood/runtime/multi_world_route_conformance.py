"""Terminal architecture-only receipt for the fake three-child public route."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.multi_world_study import MultiWorldChildRole, MultiWorldStudyChildScientificBinding
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind

from .synthetic_controller_comparison_conformance import SyntheticControllerRouteConformanceReceipt
from .multi_world_study_runbook import MultiWorldStudyOperationRunbook
from .multi_world_study import ArchiveOverlapQualificationResult, MultiWorldJointAdjudicationResult, MorphismControlContrastReceipt, PropertyMorphismDisposition, PropertyMorphismVerdict
from .multi_world_study_issue import IssuedMultiWorldStudy, MultiWorldOutcomeBarrierPrefix, MultiWorldStudyRecoveryIndex


class MultiWorldConformanceCommandKind(StrEnum):
    FOCUSED_MULTI_WORLD = "FOCUSED_MULTI_WORLD"
    PUBLIC_API_CLI = "PUBLIC_API_CLI"
    HOSTILE_ACCESS_IDENTITY = "HOSTILE_ACCESS_IDENTITY"
    RECOVERY_CATALOG_REBUILD = "RECOVERY_CATALOG_REBUILD"
    CURRENT_ROUTE_REPLAY = "CURRENT_ROUTE_REPLAY"


@dataclass(frozen=True, slots=True)
class MultiWorldConformanceCommandReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-conformance-command-receipt'

    receipt_id: str
    kind: MultiWorldConformanceCommandKind
    argument_vector: tuple[str, ...]
    result_sha256: str
    exit_code: int
    passed_count: int
    failed_count: int
    implementation_commit: str
    terminal: bool
    created_empirical_evidence: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if not self.argument_vector or any(not value for value in self.argument_vector):
            raise ValueError("conformance command receipt lacks an exact argument vector")
        validate_sha256(self.result_sha256, field_name="result_sha256")
        if len(self.implementation_commit) != 40 or any(
            value not in "0123456789abcdef" for value in self.implementation_commit
        ):
            raise ValueError("conformance command receipt has an invalid implementation commit")
        if (
            self.exit_code != 0
            or self.passed_count <= 0
            or self.failed_count != 0
            or not self.terminal
            or self.created_empirical_evidence
        ):
            raise ValueError("conformance command receipt is not a passing architecture check")


@dataclass(frozen=True, slots=True)
class MultiWorldExecutedConformanceSuite(CanonicalRecord):
    "Actual focused-command evidence for the complete architecture-only multi-world route."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-executed-conformance-suite'

    suite_id: str
    source_closure: ImplementationSourceClosure
    operation_runbook: ObjectIdentity
    command_receipts: tuple[MultiWorldConformanceCommandReceipt, ...]
    broad_regression_composition_note: str
    external_artifact_plane_required: bool
    native_archive_reads: int
    native_simulator_launches: int
    terminal: bool
    architecture_conformance_only: bool
    created_empirical_evidence: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.suite_id, field_name="suite_id")
        if self.operation_runbook.object_schema != MultiWorldStudyOperationRunbook.SCHEMA:
            raise ValueError("Multi-world route conformance suite binds another operation runbook")
        require_sorted_unique_ids(
            self.command_receipts,
            attribute="receipt_id",
            field_name="command_receipts",
        )
        if {value.kind for value in self.command_receipts} != set(
            MultiWorldConformanceCommandKind
        ) or any(
            value.implementation_commit != self.source_closure.implementation_commit
            for value in self.command_receipts
        ):
            raise ValueError("Multi-world route conformance suite command evidence is incomplete or drifted")
        if self.broad_regression_composition_note != (
            "No terminal full-suite pass was claimed: prior broad runs reached 4519/4520 "
            "passes with generated-reference-only failures; focused current repairs and "
            "the fixed architecture conformance roster are bound here."
        ):
            raise ValueError("Multi-world route conformance suite misstates its broad-regression evidence")
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or not self.external_artifact_plane_required
            or self.native_archive_reads != 0
            or self.native_simulator_launches != 0
            or not self.terminal
            or not self.architecture_conformance_only
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("Multi-world route conformance suite crosses its architecture-only boundary")


@dataclass(frozen=True, slots=True)
class MultiWorldPublicRouteConformanceReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/runtime/multi-world-public-route-conformance-receipt'
    )

    receipt_id: str
    source_closure: ImplementationSourceClosure
    issued_bundle: ObjectIdentity
    child_science: tuple[MultiWorldStudyChildScientificBinding, ...]
    fake_generated_child: ObjectIdentity
    operation_runbook: ObjectIdentity
    terminal_barrier_prefix: ObjectIdentity
    barrier_event_count: int
    reveal_event_count: int
    result_binding_event_count: int
    recovery_index: ObjectIdentity
    archive_overlap_result: ObjectIdentity
    primary_property_verdicts: tuple[ObjectIdentity, ...]
    observed_morphism_dispositions: tuple[PropertyMorphismDisposition, ...]
    negative_control_contrast: ObjectIdentity
    joint_result: ObjectIdentity
    command_receipts: tuple[MultiWorldConformanceCommandReceipt, ...]
    historical_current_import_scan_passed: bool
    child_identity_isolation_passed: bool
    simulator_only_nonpromotion_passed: bool
    external_artifact_plane_required: bool
    native_archive_reads: int
    native_simulator_launches: int
    terminal: bool
    architecture_conformance_only: bool
    created_empirical_evidence: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_ids(
            self.child_science,
            attribute="child_id",
            field_name="child_science",
        )
        if len(self.child_science) != 3 or {value.role for value in self.child_science} != set(
            MultiWorldChildRole
        ):
            raise ValueError("multi-world conformance changes the exact child roster")
        for attribute in (
            "base_candidate",
            "independent_unit_roster",
            "evidence_world",
            "law",
            "protected_outcome_domain_id",
            "execution_authority_subject",
            "reveal_authority_subject",
        ):
            if len({getattr(value, attribute) for value in self.child_science}) != 3:
                raise ValueError(f"multi-world conformance collapses child {attribute}")
        expected_schemas = (
            (self.issued_bundle, IssuedMultiWorldStudy.SCHEMA),
            (self.fake_generated_child, SyntheticControllerRouteConformanceReceipt.SCHEMA),
            (self.operation_runbook, MultiWorldStudyOperationRunbook.SCHEMA),
            (self.terminal_barrier_prefix, MultiWorldOutcomeBarrierPrefix.SCHEMA),
            (self.recovery_index, MultiWorldStudyRecoveryIndex.SCHEMA),
            (self.archive_overlap_result, ArchiveOverlapQualificationResult.SCHEMA),
            (self.negative_control_contrast, MorphismControlContrastReceipt.SCHEMA),
            (self.joint_result, MultiWorldJointAdjudicationResult.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("multi-world conformance binds another route schema")
        require_sorted_unique_ids(
            self.primary_property_verdicts,
            attribute="object_id",
            field_name="primary_property_verdicts",
        )
        if len(self.primary_property_verdicts) != 18 or any(
            value.object_schema != PropertyMorphismVerdict.SCHEMA
            for value in self.primary_property_verdicts
        ):
            raise ValueError("multi-world conformance changes the contacted property roster")
        require_sorted_unique_strings(
            tuple(value.value for value in self.observed_morphism_dispositions),
            field_name="observed_morphism_dispositions",
            allow_empty=False,
        )
        if set(self.observed_morphism_dispositions) != set(PropertyMorphismDisposition):
            raise ValueError("multi-world conformance did not exercise all morphism dispositions")
        require_sorted_unique_ids(
            self.command_receipts,
            attribute="receipt_id",
            field_name="command_receipts",
        )
        if {value.kind for value in self.command_receipts} != set(MultiWorldConformanceCommandKind):
            raise ValueError("multi-world conformance command evidence is incomplete")
        if any(
            value.implementation_commit != self.source_closure.implementation_commit
            for value in self.command_receipts
        ):
            raise ValueError("multi-world conformance command source closure differs")
        if (
            self.source_closure.kind is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.source_closure.clean_worktree
            or self.barrier_event_count != 9
            or self.reveal_event_count != 3
            or self.result_binding_event_count != 3
            or not self.historical_current_import_scan_passed
            or not self.child_identity_isolation_passed
            or not self.simulator_only_nonpromotion_passed
            or not self.external_artifact_plane_required
            or self.native_archive_reads != 0
            or self.native_simulator_launches != 0
            or not self.terminal
            or not self.architecture_conformance_only
            or self.created_empirical_evidence
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("multi-world conformance crosses its architecture-only boundary")


__all__ = [
    'MultiWorldConformanceCommandKind',
    'MultiWorldConformanceCommandReceipt',
    'MultiWorldPublicRouteConformanceReceipt',
    'MultiWorldExecutedConformanceSuite',
]
