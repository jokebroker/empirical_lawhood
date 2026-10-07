# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
from __future__ import annotations

from dataclasses import replace

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.planning.experiment_entry import StudyDefinition
from empirical_lawhood.planning.formal_gaps import (
    FormalGapCoverageAssignment,
    FormalGapCoverageDisposition,
)
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationDisposition, CandidateDiagnosticCode, StandardCandidateCompilationContext, compile_study_candidate
from tests.experiment_entry_support import standard_authoring_for_draft
from tests.runtime_platform.test_candidate_compiler import (
    _fixture,
    _materialization,
)


def _standard_fixture() -> tuple[
    StudyDefinition,
    StandardCandidateCompilationContext,
]:
    draft, base_context = _fixture()
    return standard_authoring_for_draft(draft, base_context)


def test_standard_candidate_binds_entry_methods_source_and_graph() -> None:
    authoring, context = _standard_fixture()

    report = compile_study_candidate(
        authoring_package=authoring,
        authoring_materialization=_materialization(),
        context=context,
    )

    assert report.disposition is CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING
    assert report.candidate is not None
    assert report.candidate.authoring_package == ObjectIdentity.from_record(
        authoring.package_id,
        authoring,
    )
    assert report.candidate.formal_gap_coverage == ObjectIdentity.from_record(
        authoring.entry_package.coverage.coverage_id,
        authoring.entry_package.coverage,
    )


def test_standard_candidate_rejects_fictitious_graph_owner() -> None:
    authoring, context = _standard_fixture()
    first = replace(
        authoring.entry_package.coverage.assignments[0],
        adjudication_owner_ids=("node.fictitious",),
    )
    coverage = replace(
        authoring.entry_package.coverage,
        assignments=(first, *authoring.entry_package.coverage.assignments[1:]),
    )
    checklist = replace(
        authoring.entry_package.checklist,
        formal_gap_coverage=ObjectIdentity.from_record(
            coverage.coverage_id,
            coverage,
        ),
    )
    entry = replace(
        authoring.entry_package,
        coverage=coverage,
        checklist=checklist,
    )
    hostile = replace(authoring, entry_package=entry)

    report = compile_study_candidate(
        authoring_package=hostile,
        authoring_materialization=_materialization(),
        context=context,
    )

    assert report.disposition is CandidateCompilationDisposition.INVALID_DRAFT
    assert CandidateDiagnosticCode.FORMAL_GAP_GRAPH_BINDING_INVALID in {
        value.code for value in report.diagnostics
    }


def test_standard_candidate_rejects_source_feasible_operand_omission() -> None:
    authoring, context = _standard_fixture()
    applicability = tuple(
        replace(value, present_operand_ids=())
        for value in authoring.entry_package.coverage.applicability
    )
    assignments = tuple(
        FormalGapCoverageAssignment(
            gap_id=value.gap_id,
            disposition=FormalGapCoverageDisposition.UNEVALUABLE_FROM_SOURCE,
            readiness_reason=ReadinessStatus.REQUIRES_NEW_DATA,
            reason_codes=("SOURCE_OPERAND_ABSENT",),
            selected_estimator_family_id=None,
            selected_control_ids=(),
            selected_multiplicity_family_id=None,
            obligation_ids=(),
            output_ids=(),
            adjudication_owner_ids=(),
        )
        for value in authoring.entry_package.coverage.assignments
    )
    coverage = replace(
        authoring.entry_package.coverage,
        applicability=applicability,
        assignments=assignments,
    )
    checklist = replace(
        authoring.entry_package.checklist,
        formal_gap_coverage=ObjectIdentity.from_record(
            coverage.coverage_id,
            coverage,
        ),
    )
    entry = replace(
        authoring.entry_package,
        coverage=coverage,
        checklist=checklist,
    )
    hostile = replace(authoring, entry_package=entry)

    report = compile_study_candidate(
        authoring_package=hostile,
        authoring_materialization=_materialization(),
        context=context,
    )

    assert report.disposition is CandidateCompilationDisposition.INVALID_DRAFT
    assert CandidateDiagnosticCode.FORMAL_GAP_FEASIBLE_OPERAND_OMITTED in {
        value.code for value in report.diagnostics
    }
