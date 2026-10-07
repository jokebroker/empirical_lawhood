"""Exact runtime resolution for planning-owned evidence profiles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_schema,
    validate_stable_id,
)
from empirical_lawhood.planning.evidence_profiles import ExperimentObjectiveKind, EvidenceWorldKind, EvidenceProfileReasonCode, EvidenceProfileSelection, EvidenceWorldProfileRegistry, RungFeasibilityCell, RungFeasibilityStatus
from empirical_lawhood.planning.experiment_entry import StudyDefinition

from .candidate_compiler import AuthoringMaterializationIdentity, StandardCandidateCompilationContext, StudyCompilationReport, compile_study_candidate


class ProfileFeasibilityDisposition(StrEnum):
    READY_FOR_CANDIDATE_COMPILATION = "READY_FOR_CANDIDATE_COMPILATION"
    OBJECTIVE_SPECIFIC_TERMINAL = "OBJECTIVE_SPECIFIC_TERMINAL"
    INFEASIBLE = "INFEASIBLE"
    INAPPLICABLE = "INAPPLICABLE"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    PROFILE_IDENTITY_MISMATCH = "PROFILE_IDENTITY_MISMATCH"


@dataclass(frozen=True, slots=True)
class EvidenceProfileRegistryResolver(CanonicalRecord):
    """Closed injected registry set; no document-selected import or dispatch."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/evidence-profile-registry-resolver'

    resolver_id: str
    registries: tuple[EvidenceWorldProfileRegistry, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.resolver_id, field_name="resolver_id")
        require_sorted_unique_ids(self.registries, attribute="registry_id", field_name="registries")
        keys = tuple((value.registry_id, value.registry_version) for value in self.registries)
        if len(set(keys)) != len(keys):
            raise ValueError("profile resolver contains a duplicate key/version")

    def resolve_registry(
        self, selection: EvidenceProfileSelection
    ) -> EvidenceWorldProfileRegistry:
        matches = [
            value
            for value in self.registries
            if value.registry_id == selection.registry_id
            and value.fingerprint() == selection.registry_fingerprint
        ]
        if len(matches) != 1:
            raise ValueError("profile selection does not resolve to one exact registry")
        return matches[0]


@dataclass(frozen=True, slots=True)
class CandidateProfileFeasibilityResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-profile-feasibility-result'

    result_id: str
    selection: ObjectIdentity
    disposition: ProfileFeasibilityDisposition
    cells: tuple[RungFeasibilityCell, ...]
    terminal_product_schema: str
    reason_codes: tuple[EvidenceProfileReasonCode, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.selection.object_schema != EvidenceProfileSelection.SCHEMA:
            raise ValueError("feasibility result requires an evidence profile selection")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        validate_schema(self.terminal_product_schema)
        if tuple(reason.value for reason in self.reason_codes) != tuple(
            sorted({reason.value for reason in self.reason_codes})
        ):
            raise ValueError("feasibility reasons must be sorted and unique")
        if self.disposition is not ProfileFeasibilityDisposition.READY_FOR_CANDIDATE_COMPILATION:
            if not self.reason_codes:
                raise ValueError("non-ready profile result requires a typed reason")


@dataclass(frozen=True, slots=True)
class ProfiledStandardCandidateCompilationReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/profiled-standard-candidate-compilation-report'

    report_id: str
    feasibility: CandidateProfileFeasibilityResult
    candidate_report: StudyCompilationReport | None

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        ready = (
            self.feasibility.disposition
            is ProfileFeasibilityDisposition.READY_FOR_CANDIDATE_COMPILATION
        )
        if ready != (self.candidate_report is not None):
            raise ValueError("profile feasibility and candidate compilation differ")


def resolve_candidate_profile_feasibility(
    *,
    selection: EvidenceProfileSelection,
    resolver: EvidenceProfileRegistryResolver,
) -> CandidateProfileFeasibilityResult:
    """Resolve exact identities and stop impossible work before compilation."""

    selection_identity = ObjectIdentity.from_record(selection.selection_id, selection)
    try:
        registry = resolver.resolve_registry(selection)
        worlds = {
            (value.profile_id, value.profile_version, value.fingerprint()): value
            for value in registry.world_profiles
        }
        objectives = {
            (value.profile_id, value.profile_version, value.fingerprint()): value
            for value in registry.objective_profiles
        }
        world = worlds[
            (
                selection.evidence_world_profile_id,
                selection.evidence_world_profile_version,
                selection.evidence_world_profile_fingerprint,
            )
        ]
        objective = objectives[
            (
                selection.objective_profile_id,
                selection.objective_profile_version,
                selection.objective_profile_fingerprint,
            )
        ]
    except (KeyError, ValueError):
        return CandidateProfileFeasibilityResult(
            result_id=f"profile-feasibility.{selection.fingerprint()[:32]}",
            selection=selection_identity,
            disposition=ProfileFeasibilityDisposition.PROFILE_IDENTITY_MISMATCH,
            cells=(),
            terminal_product_schema='empirical-lawhood/runtime/candidate-profile-feasibility-result',
            reason_codes=(EvidenceProfileReasonCode.PROFILE_IDENTITY_MISMATCH,),
        )

    if objective.objective_kind is ExperimentObjectiveKind.HISTORICAL_PREDICTION:
        if selection.requested_rungs:
            raise ValueError("historical prediction does not request a measurement through controller use rung")
        held = world.world_kind in {
            EvidenceWorldKind.FIXED_ARCHIVE,
            EvidenceWorldKind.BENCHMARK,
            EvidenceWorldKind.TRUTH_KNOWN_REFERENCE,
        }
        return CandidateProfileFeasibilityResult(
            result_id=f"profile-feasibility.{selection.fingerprint()[:32]}",
            selection=selection_identity,
            # A profile selection alone does not supply a historical experiment
            # root or its contract closure. The dedicated candidate route must
            # bind those records; this reason does not mean the type is absent.
            disposition=(
                ProfileFeasibilityDisposition.OBJECTIVE_SPECIFIC_TERMINAL
                if held
                else ProfileFeasibilityDisposition.INAPPLICABLE
            ),
            cells=(),
            terminal_product_schema=objective.terminal_product_schema,
            reason_codes=(
                (
                    EvidenceProfileReasonCode.HISTORICAL_EXPERIMENT_ROOT_REQUIRED,
                    EvidenceProfileReasonCode.OBJECTIVE_TERMINAL_OUTSIDE_EVIDENCE_RUNGS,
                )
                if held
                else (EvidenceProfileReasonCode.OBJECTIVE_TERMINAL_OUTSIDE_EVIDENCE_RUNGS,)
            ),
        )
    if not objective.uses_evidence_rungs:
        return CandidateProfileFeasibilityResult(
            result_id=f"profile-feasibility.{selection.fingerprint()[:32]}",
            selection=selection_identity,
            disposition=ProfileFeasibilityDisposition.OBJECTIVE_SPECIFIC_TERMINAL,
            cells=(),
            terminal_product_schema=objective.terminal_product_schema,
            reason_codes=(EvidenceProfileReasonCode.OBJECTIVE_TERMINAL_OUTSIDE_EVIDENCE_RUNGS,),
        )
    if not selection.requested_rungs:
        raise ValueError("Measurement through controller use objective requires at least one requested rung")
    cells_by_key = {
        (value.evidence_world_profile_id, value.objective_profile_id, value.rung): value
        for value in registry.rung_cells
    }
    cells = tuple(
        sorted(
            (
                cells_by_key[(world.profile_id, objective.profile_id, rung)]
                for rung in selection.requested_rungs
            ),
            key=lambda value: value.cell_id,
        )
    )
    statuses = {value.status for value in cells}
    if RungFeasibilityStatus.INFEASIBLE in statuses:
        disposition = ProfileFeasibilityDisposition.INFEASIBLE
    elif RungFeasibilityStatus.INAPPLICABLE in statuses:
        disposition = ProfileFeasibilityDisposition.INAPPLICABLE
    elif RungFeasibilityStatus.AUTHORITY_REQUIRED in statuses:
        disposition = ProfileFeasibilityDisposition.AUTHORITY_REQUIRED
    else:
        disposition = ProfileFeasibilityDisposition.READY_FOR_CANDIDATE_COMPILATION
    reasons = tuple(
        sorted(
            {reason for cell in cells for reason in cell.reason_codes},
            key=lambda value: value.value,
        )
    )
    return CandidateProfileFeasibilityResult(
        result_id=f"profile-feasibility.{selection.fingerprint()[:32]}",
        selection=selection_identity,
        disposition=disposition,
        cells=cells,
        terminal_product_schema=objective.terminal_product_schema,
        reason_codes=reasons,
    )


def compile_profiled_study_candidate(
    *,
    authoring_package: StudyDefinition,
    authoring_materialization: AuthoringMaterializationIdentity,
    context: StandardCandidateCompilationContext,
    selection: EvidenceProfileSelection,
    resolver: EvidenceProfileRegistryResolver,
) -> ProfiledStandardCandidateCompilationReport:
    """Profile gate followed by the unchanged sole standard candidate compiler."""

    if selection.draft_id != authoring_package.draft.draft_id:
        raise ValueError("profile selection binds another programme draft")
    feasibility = resolve_candidate_profile_feasibility(
        selection=selection,
        resolver=resolver,
    )
    candidate_report = (
        compile_study_candidate(
            authoring_package=authoring_package,
            authoring_materialization=authoring_materialization,
            context=context,
        )
        if feasibility.disposition is ProfileFeasibilityDisposition.READY_FOR_CANDIDATE_COMPILATION
        else None
    )
    return ProfiledStandardCandidateCompilationReport(
        report_id=f"profiled-standard-report.{feasibility.fingerprint()[:32]}",
        feasibility=feasibility,
        candidate_report=candidate_report,
    )


__all__ = [
    'CandidateProfileFeasibilityResult',
    'EvidenceProfileRegistryResolver',
    "ProfileFeasibilityDisposition",
    'ProfiledStandardCandidateCompilationReport',
    'compile_profiled_study_candidate',
    "resolve_candidate_profile_feasibility",
]
