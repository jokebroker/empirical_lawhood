"""Pure Matrix source nomination dual-source nomination over persisted preparation scheduling factor ledgers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Iterable

from empirical_lawhood.adapters.simulators.six_matrix_response.preparation_schedule import SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .prospective_window_scheduling import MatrixResponsePreparationWindowSchedulingFactorRecordDisposition, MatrixResponsePreparationWindowSchedulingFactorSample, MatrixResponsePreparationWindowSchedulingTrajectoryFactorRecord


PLAN_ID = "matrix-dual-source-nomination"
SCHEDULE_IDS = ("m64-r075", "p64-r125")
EXPECTED_STEPS = tuple(range(256, 769, 16))
CURRENT_STEPS = tuple(range(256, 641, 16))
LATE_STEPS = tuple(range(656, 769, 16))
MINIMUM_ROOT_HITS = 6
MINIMUM_HALF_HITS = 3


class MatrixResponseDualSourceNominationCohort(StrEnum):
    HALF_A = "HALF_A"
    HALF_B = "HALF_B"


class MatrixResponseDualSourceNominationNominationRoute(StrEnum):
    CURRENT = "CURRENT"
    LATE = "LATE"


class MatrixResponseDualSourceNominationTerminal(StrEnum):
    CURRENT_HORIZON_SOURCE_NOMINATED = "CURRENT_HORIZON_SOURCE_NOMINATED"
    LATE_HORIZON_SOURCE_NOMINATED_FOR_REACTIVE_SOURCE_LAW_ONLY = (
        "LATE_HORIZON_SOURCE_NOMINATED_FOR_REACTIVE_SOURCE_LAW_ONLY"
    )
    DUAL_SOURCE_NOMINATION_AMBIGUOUS = "DUAL_SOURCE_NOMINATION_AMBIGUOUS"
    NO_TESTED_SOURCE_FEASIBILITY = "NO_TESTED_SOURCE_FEASIBILITY"
    DUAL_SOURCE_NOMINATION_TECHNICAL_INVALID = (
        "DUAL_SOURCE_NOMINATION_TECHNICAL_INVALID"
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseDualSourceNominationDesign(CanonicalRecord):
    """Outcome-inaccessible two-word, 64-root source-screen design."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-dual-source-nomination-design'

    design_id: str
    plan_id: str
    cohort_a_roster: ObjectIdentity
    cohort_b_roster: ObjectIdentity
    cohort_a_root_ids: tuple[str, ...]
    cohort_b_root_ids: tuple[str, ...]
    schedule_ids: tuple[str, ...]
    current_steps: tuple[int, ...]
    late_steps: tuple[int, ...]
    minimum_root_hits: int
    minimum_half_hits: int
    physical_root_count: int
    trajectory_count: int
    planned_integration_updates: int
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        if self.plan_id != PLAN_ID:
            raise ValueError("Matrix source nomination plan identity differs")
        if (
            self.cohort_a_roster.object_schema != SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster.SCHEMA
            or self.cohort_b_roster.object_schema != SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster.SCHEMA
            or self.cohort_a_roster == self.cohort_b_roster
        ):
            raise ValueError("Matrix source nomination cohort roster identities differ")
        require_sorted_unique_strings(
            self.cohort_a_root_ids,
            field_name="cohort_a_root_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.cohort_b_root_ids,
            field_name="cohort_b_root_ids",
            allow_empty=False,
        )
        if set(self.cohort_a_root_ids) & set(self.cohort_b_root_ids):
            raise ValueError("Matrix source nomination cohorts share physical roots")
        if (
            len(self.cohort_a_root_ids) != 32
            or len(self.cohort_b_root_ids) != 32
            or self.schedule_ids != SCHEDULE_IDS
            or self.current_steps != CURRENT_STEPS
            or self.late_steps != LATE_STEPS
            or self.minimum_root_hits != MINIMUM_ROOT_HITS
            or self.minimum_half_hits != MINIMUM_HALF_HITS
            or self.physical_root_count != 64
            or self.trajectory_count != 128
            or self.planned_integration_updates != 143_360
            or self.grants_authority
        ):
            raise ValueError("Matrix source nomination frozen design differs")

    @property
    def root_ids(self) -> tuple[str, ...]:
        return self.cohort_a_root_ids + self.cohort_b_root_ids


@dataclass(frozen=True, slots=True)
class MatrixResponseDualSourceNominationRootEndpoint(CanonicalRecord):
    """One schedule occurrence reduced without losing factor-record custody."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-dual-source-nomination-root-endpoint'

    endpoint_id: str
    root_id: str
    root_index: int
    cohort: MatrixResponseDualSourceNominationCohort
    schedule_id: str
    source_factor_record: ObjectIdentity
    preparation_tape: ObjectIdentity
    observation_tape: ObjectIdentity
    sample_count: int
    current_structural_sample_count: int
    late_structural_sample_count: int
    amplitude_pass_count: int
    closure_pass_count: int
    spectrum_valid_count: int
    kernel_pass_count: int
    persistence_pass_count: int
    strict_kernel_pass_count: int
    x_exclusion_pass_count: int
    response_law_pass_count: int
    structural_pass_count: int
    full_intersection_pass_count: int
    one_factor_near_miss_count: int
    two_factor_near_miss_count: int
    maximum_persistence_count: int
    longest_current_run_samples: int
    longest_late_run_samples: int
    technically_valid: bool
    reason_codes: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.endpoint_id, field_name="endpoint_id")
        validate_stable_id(self.root_id, field_name="root_id")
        validate_stable_id(self.schedule_id, field_name="schedule_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            not 0 <= self.root_index < 32
            or self.schedule_id not in SCHEDULE_IDS
            or self.source_factor_record.object_schema
            != MatrixResponsePreparationWindowSchedulingTrajectoryFactorRecord.SCHEMA
            or self.preparation_tape == self.observation_tape
            or self.grants_authority
        ):
            raise ValueError("Matrix source nomination root endpoint coordinate differs")
        counts = (
            self.sample_count,
            self.current_structural_sample_count,
            self.late_structural_sample_count,
            self.amplitude_pass_count,
            self.closure_pass_count,
            self.spectrum_valid_count,
            self.kernel_pass_count,
            self.persistence_pass_count,
            self.strict_kernel_pass_count,
            self.x_exclusion_pass_count,
            self.response_law_pass_count,
            self.structural_pass_count,
            self.full_intersection_pass_count,
            self.one_factor_near_miss_count,
            self.two_factor_near_miss_count,
            self.maximum_persistence_count,
            self.longest_current_run_samples,
            self.longest_late_run_samples,
        )
        if min(counts) < 0 or max(counts[:-3], default=0) > self.sample_count:
            raise ValueError("Matrix source nomination endpoint count is out of range")
        if not 0 <= self.maximum_persistence_count <= 16:
            raise ValueError("Matrix source nomination persistence count is out of range")
        if self.current_structural_sample_count + self.late_structural_sample_count != (
            self.structural_pass_count
        ):
            raise ValueError("Matrix source nomination structural interval partition differs")
        if self.technically_valid:
            if self.sample_count != 33 or self.reason_codes:
                raise ValueError("Matrix source nomination valid endpoint is incomplete")
        elif not self.reason_codes:
            raise ValueError("Matrix source nomination invalid endpoint lacks a reason")

    @property
    def current_hit(self) -> bool:
        return self.current_structural_sample_count > 0

    @property
    def late_hit(self) -> bool:
        return self.late_structural_sample_count > 0


def _longest_true_run(values: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def build_source_root_endpoint(
    record: MatrixResponsePreparationWindowSchedulingTrajectoryFactorRecord,
    *,
    cohort: MatrixResponseDualSourceNominationCohort,
) -> MatrixResponseDualSourceNominationRootEndpoint:
    """Reduce one exact persisted factor record into source-screen operands."""

    reasons: set[str] = set(record.reason_codes)
    technically_valid = bool(
        record.disposition is MatrixResponsePreparationWindowSchedulingFactorRecordDisposition.EVALUATED
        and record.outcome is not None
        and record.outcome.technically_valid
        and len(record.samples) == len(EXPECTED_STEPS)
        and tuple(value.operands.parent_step for value in record.samples) == EXPECTED_STEPS
    )
    if not technically_valid:
        reasons.add("source-factor-record-invalid")
    samples = record.samples if technically_valid else ()
    current = tuple(value for value in samples if value.operands.parent_step in CURRENT_STEPS)
    late = tuple(value for value in samples if value.operands.parent_step in LATE_STEPS)
    if technically_valid and (len(current), len(late)) != (25, 8):
        technically_valid = False
        reasons.add("source-factor-interval-partition-invalid")
        samples = ()
        current = ()
        late = ()

    def missing_structural_factors(value: MatrixResponsePreparationWindowSchedulingFactorSample) -> int:
        return sum(
            not factor
            for factor in (
                value.amplitude_pass,
                value.closure_pass,
                value.spectrum_valid,
                value.kernel_pass,
                value.persistence_count_pass,
                value.strict_kernel_window_pass,
                value.x_exclusion_pass,
            )
        )

    return MatrixResponseDualSourceNominationRootEndpoint(
        endpoint_id=(
            f"matrix-source-nomination.endpoint.{cohort.value.lower()}."
            f"{record.root_id}.{record.schedule_id}"
        ),
        root_id=record.root_id,
        root_index=record.root_index,
        cohort=cohort,
        schedule_id=record.schedule_id,
        source_factor_record=ObjectIdentity.from_record(record.record_id, record),
        preparation_tape=record.preparation_tape,
        observation_tape=record.observation_tape,
        sample_count=len(samples),
        current_structural_sample_count=sum(value.structural_geometry_pass for value in current),
        late_structural_sample_count=sum(value.structural_geometry_pass for value in late),
        amplitude_pass_count=sum(value.amplitude_pass for value in samples),
        closure_pass_count=sum(value.closure_pass for value in samples),
        spectrum_valid_count=sum(value.spectrum_valid for value in samples),
        kernel_pass_count=sum(value.kernel_pass for value in samples),
        persistence_pass_count=sum(value.persistence_count_pass for value in samples),
        strict_kernel_pass_count=sum(value.strict_kernel_window_pass for value in samples),
        x_exclusion_pass_count=sum(value.x_exclusion_pass for value in samples),
        response_law_pass_count=sum(value.response_law_pass for value in samples),
        structural_pass_count=sum(value.structural_geometry_pass for value in samples),
        full_intersection_pass_count=sum(value.full_intersection_pass for value in samples),
        one_factor_near_miss_count=sum(
            missing_structural_factors(value) == 1 for value in samples
        ),
        two_factor_near_miss_count=sum(
            missing_structural_factors(value) == 2 for value in samples
        ),
        maximum_persistence_count=max(
            (value.operands.persistence_pass_count for value in samples),
            default=0,
        ),
        longest_current_run_samples=_longest_true_run(
            value.structural_geometry_pass for value in current
        ),
        longest_late_run_samples=_longest_true_run(
            value.structural_geometry_pass for value in late
        ),
        technically_valid=technically_valid,
        reason_codes=() if technically_valid else tuple(sorted(reasons)),
        grants_authority=False,
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseDualSourceNominationScheduleSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-dual-source-nomination-schedule-summary'

    summary_id: str
    schedule_id: str
    observed_root_count: int
    current_hit_root_count: int
    current_half_a_hit_count: int
    current_half_b_hit_count: int
    late_hit_root_count: int
    late_half_a_hit_count: int
    late_half_b_hit_count: int
    current_structural_sample_count: int
    late_structural_sample_count: int
    amplitude_pass_count: int
    closure_pass_count: int
    spectrum_valid_count: int
    kernel_pass_count: int
    persistence_pass_count: int
    strict_kernel_pass_count: int
    x_exclusion_pass_count: int
    response_law_pass_count: int
    structural_pass_count: int
    full_intersection_pass_count: int
    one_factor_near_miss_count: int
    two_factor_near_miss_count: int
    maximum_persistence_count: int
    longest_current_run_samples: int
    longest_late_run_samples: int
    current_candidate: bool
    late_candidate: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_stable_id(self.schedule_id, field_name="schedule_id")
        if self.schedule_id not in SCHEDULE_IDS or self.grants_authority:
            raise ValueError("Matrix source nomination schedule summary coordinate differs")
        counts = (
            self.observed_root_count,
            self.current_hit_root_count,
            self.current_half_a_hit_count,
            self.current_half_b_hit_count,
            self.late_hit_root_count,
            self.late_half_a_hit_count,
            self.late_half_b_hit_count,
            self.current_structural_sample_count,
            self.late_structural_sample_count,
            self.amplitude_pass_count,
            self.closure_pass_count,
            self.spectrum_valid_count,
            self.kernel_pass_count,
            self.persistence_pass_count,
            self.strict_kernel_pass_count,
            self.x_exclusion_pass_count,
            self.response_law_pass_count,
            self.structural_pass_count,
            self.full_intersection_pass_count,
            self.one_factor_near_miss_count,
            self.two_factor_near_miss_count,
            self.maximum_persistence_count,
            self.longest_current_run_samples,
            self.longest_late_run_samples,
        )
        if min(counts, default=0) < 0:
            raise ValueError("Matrix source nomination schedule summary count is negative")
        expected_current = bool(
            self.observed_root_count == 64
            and self.current_hit_root_count >= MINIMUM_ROOT_HITS
            and self.current_half_a_hit_count >= MINIMUM_HALF_HITS
            and self.current_half_b_hit_count >= MINIMUM_HALF_HITS
        )
        expected_late = bool(
            self.observed_root_count == 64
            and self.late_hit_root_count >= MINIMUM_ROOT_HITS
            and self.late_half_a_hit_count >= MINIMUM_HALF_HITS
            and self.late_half_b_hit_count >= MINIMUM_HALF_HITS
        )
        if (self.current_candidate, self.late_candidate) != (expected_current, expected_late):
            raise ValueError("Matrix source nomination candidate truth differs from frozen counts")


def _schedule_summary(
    schedule_id: str,
    endpoints: tuple[MatrixResponseDualSourceNominationRootEndpoint, ...],
) -> MatrixResponseDualSourceNominationScheduleSummary:
    selected = tuple(value for value in endpoints if value.schedule_id == schedule_id)

    def total(field: str) -> int:
        return sum(int(getattr(value, field)) for value in selected)

    observed = len({value.root_id for value in selected})
    current_a = sum(
        value.current_hit and value.cohort is MatrixResponseDualSourceNominationCohort.HALF_A for value in selected
    )
    current_b = sum(
        value.current_hit and value.cohort is MatrixResponseDualSourceNominationCohort.HALF_B for value in selected
    )
    late_a = sum(
        value.late_hit and value.cohort is MatrixResponseDualSourceNominationCohort.HALF_A for value in selected
    )
    late_b = sum(
        value.late_hit and value.cohort is MatrixResponseDualSourceNominationCohort.HALF_B for value in selected
    )
    current_hits = current_a + current_b
    late_hits = late_a + late_b
    return MatrixResponseDualSourceNominationScheduleSummary(
        summary_id=f"matrix-source-nomination.summary.{schedule_id}",
        schedule_id=schedule_id,
        observed_root_count=observed,
        current_hit_root_count=current_hits,
        current_half_a_hit_count=current_a,
        current_half_b_hit_count=current_b,
        late_hit_root_count=late_hits,
        late_half_a_hit_count=late_a,
        late_half_b_hit_count=late_b,
        current_structural_sample_count=total("current_structural_sample_count"),
        late_structural_sample_count=total("late_structural_sample_count"),
        amplitude_pass_count=total("amplitude_pass_count"),
        closure_pass_count=total("closure_pass_count"),
        spectrum_valid_count=total("spectrum_valid_count"),
        kernel_pass_count=total("kernel_pass_count"),
        persistence_pass_count=total("persistence_pass_count"),
        strict_kernel_pass_count=total("strict_kernel_pass_count"),
        x_exclusion_pass_count=total("x_exclusion_pass_count"),
        response_law_pass_count=total("response_law_pass_count"),
        structural_pass_count=total("structural_pass_count"),
        full_intersection_pass_count=total("full_intersection_pass_count"),
        one_factor_near_miss_count=total("one_factor_near_miss_count"),
        two_factor_near_miss_count=total("two_factor_near_miss_count"),
        maximum_persistence_count=max(
            (value.maximum_persistence_count for value in selected), default=0
        ),
        longest_current_run_samples=max(
            (value.longest_current_run_samples for value in selected), default=0
        ),
        longest_late_run_samples=max(
            (value.longest_late_run_samples for value in selected), default=0
        ),
        current_candidate=bool(
            observed == 64
            and current_hits >= MINIMUM_ROOT_HITS
            and current_a >= MINIMUM_HALF_HITS
            and current_b >= MINIMUM_HALF_HITS
        ),
        late_candidate=bool(
            observed == 64
            and late_hits >= MINIMUM_ROOT_HITS
            and late_a >= MINIMUM_HALF_HITS
            and late_b >= MINIMUM_HALF_HITS
        ),
        grants_authority=False,
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseDualSourceNominationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-dual-source-nomination-result'

    result_id: str
    attempt_id: str
    authorization_id: str
    design: ObjectIdentity
    endpoint_identities: tuple[ObjectIdentity, ...]
    schedule_summaries: tuple[MatrixResponseDualSourceNominationScheduleSummary, ...]
    terminal: MatrixResponseDualSourceNominationTerminal
    nomination_route: MatrixResponseDualSourceNominationNominationRoute | None
    selected_schedule_id: str | None
    technically_valid: bool
    reason_codes: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("result_id", "attempt_id", "authorization_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.endpoint_identities,
            attribute="object_id",
            field_name="endpoint_identities",
        )
        require_sorted_unique_ids(
            self.schedule_summaries,
            attribute="schedule_id",
            field_name="schedule_summaries",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.design.object_schema != MatrixResponseDualSourceNominationDesign.SCHEMA or self.grants_authority:
            raise ValueError("Matrix source nomination result design/authority differs")
        if tuple(value.schedule_id for value in self.schedule_summaries) != SCHEDULE_IDS:
            raise ValueError("Matrix source nomination result lacks its two summaries")
        if self.terminal is MatrixResponseDualSourceNominationTerminal.DUAL_SOURCE_NOMINATION_TECHNICAL_INVALID:
            if self.technically_valid or self.nomination_route is not None or self.selected_schedule_id:
                raise ValueError("Matrix source nomination technical result selects a schedule")
            if not self.reason_codes:
                raise ValueError("Matrix source nomination technical result lacks a reason")
            return
        if (
            not self.technically_valid
            or len(self.endpoint_identities) != 128
            or self.reason_codes
        ):
            raise ValueError("Matrix source nomination scientific result has invalid custody")
        current_positive = self.terminal is MatrixResponseDualSourceNominationTerminal.CURRENT_HORIZON_SOURCE_NOMINATED
        late_positive = (
            self.terminal
            is MatrixResponseDualSourceNominationTerminal.LATE_HORIZON_SOURCE_NOMINATED_FOR_REACTIVE_SOURCE_LAW_ONLY
        )
        if current_positive or late_positive:
            expected_route = (
                MatrixResponseDualSourceNominationNominationRoute.CURRENT
                if current_positive
                else MatrixResponseDualSourceNominationNominationRoute.LATE
            )
            if self.nomination_route is not expected_route or self.selected_schedule_id not in SCHEDULE_IDS:
                raise ValueError("Matrix source nomination positive terminal lacks its exact nomination")
        elif self.nomination_route is not None or self.selected_schedule_id is not None:
            raise ValueError("Matrix source nomination nonpositive terminal selects a schedule")


def _choose_candidate(
    candidates: tuple[MatrixResponseDualSourceNominationScheduleSummary, ...],
    *,
    route: MatrixResponseDualSourceNominationNominationRoute,
) -> tuple[str | None, bool]:
    """Return selected ID and whether an exact outcome tie remains."""

    if not candidates:
        return None, False
    if len(candidates) == 1:
        return candidates[0].schedule_id, False
    hit_field = (
        "current_hit_root_count"
        if route is MatrixResponseDualSourceNominationNominationRoute.CURRENT
        else "late_hit_root_count"
    )
    sample_field = (
        "current_structural_sample_count"
        if route is MatrixResponseDualSourceNominationNominationRoute.CURRENT
        else "late_structural_sample_count"
    )
    ranked = sorted(
        candidates,
        key=lambda value: (getattr(value, hit_field), getattr(value, sample_field)),
        reverse=True,
    )
    first_key = (getattr(ranked[0], hit_field), getattr(ranked[0], sample_field))
    second_key = (getattr(ranked[1], hit_field), getattr(ranked[1], sample_field))
    if first_key == second_key:
        return None, True
    return ranked[0].schedule_id, False


def reduce_dual_source_nomination(
    endpoints: Iterable[MatrixResponseDualSourceNominationRootEndpoint],
    *,
    design: MatrixResponseDualSourceNominationDesign,
    attempt_id: str,
    authorization_id: str,
) -> MatrixResponseDualSourceNominationResult:
    """Apply the frozen current-first, late-second dual-source result map."""

    ordered = tuple(sorted(endpoints, key=lambda value: value.endpoint_id))
    summaries = tuple(_schedule_summary(schedule, ordered) for schedule in SCHEDULE_IDS)
    reasons: set[str] = set()
    expected_coordinates = {
        (root_id, schedule_id)
        for root_id in design.root_ids
        for schedule_id in design.schedule_ids
    }
    observed_coordinates = {(value.root_id, value.schedule_id) for value in ordered}
    if len(ordered) != 128 or len(observed_coordinates) != len(ordered):
        reasons.add("source-endpoint-count-or-coordinate-duplicate")
    if observed_coordinates != expected_coordinates:
        reasons.add("source-endpoint-roster-incomplete")
    if len({value.source_factor_record.object_id for value in ordered}) != len(ordered):
        reasons.add("source-factor-record-reused")
    for value in ordered:
        expected_cohort = (
            MatrixResponseDualSourceNominationCohort.HALF_A
            if value.root_id in design.cohort_a_root_ids
            else MatrixResponseDualSourceNominationCohort.HALF_B
        )
        if value.cohort is not expected_cohort:
            reasons.add("source-endpoint-cohort-mismatch")
        if not value.technically_valid:
            reasons.add("source-endpoint-technically-invalid")
    by_root: dict[str, list[MatrixResponseDualSourceNominationRootEndpoint]] = {}
    for value in ordered:
        by_root.setdefault(value.root_id, []).append(value)
    for root_id, values in by_root.items():
        if len(values) != 2:
            reasons.add("source-root-pair-incomplete")
            continue
        if (
            values[0].preparation_tape != values[1].preparation_tape
            or values[0].observation_tape != values[1].observation_tape
        ):
            reasons.add("source-root-common-noise-mismatch")
        if root_id not in design.root_ids:
            reasons.add("source-root-outside-design")

    identity_by_id = {
        value.endpoint_id: ObjectIdentity.from_record(value.endpoint_id, value)
        for value in ordered
    }
    identities = tuple(identity_by_id[key] for key in sorted(identity_by_id))
    design_identity = ObjectIdentity.from_record(design.design_id, design)
    if reasons:
        return MatrixResponseDualSourceNominationResult(
            result_id=f"matrix-source-nomination.result.{attempt_id}",
            attempt_id=attempt_id,
            authorization_id=authorization_id,
            design=design_identity,
            endpoint_identities=identities,
            schedule_summaries=summaries,
            terminal=MatrixResponseDualSourceNominationTerminal.DUAL_SOURCE_NOMINATION_TECHNICAL_INVALID,
            nomination_route=None,
            selected_schedule_id=None,
            technically_valid=False,
            reason_codes=tuple(sorted(reasons)),
            grants_authority=False,
        )

    current = tuple(value for value in summaries if value.current_candidate)
    selected, tied = _choose_candidate(
        current,
        route=MatrixResponseDualSourceNominationNominationRoute.CURRENT,
    )
    if tied:
        terminal = MatrixResponseDualSourceNominationTerminal.DUAL_SOURCE_NOMINATION_AMBIGUOUS
        route = None
    elif selected is not None:
        terminal = MatrixResponseDualSourceNominationTerminal.CURRENT_HORIZON_SOURCE_NOMINATED
        route = MatrixResponseDualSourceNominationNominationRoute.CURRENT
    else:
        late = tuple(value for value in summaries if value.late_candidate)
        selected, tied = _choose_candidate(
            late,
            route=MatrixResponseDualSourceNominationNominationRoute.LATE,
        )
        if tied:
            terminal = MatrixResponseDualSourceNominationTerminal.DUAL_SOURCE_NOMINATION_AMBIGUOUS
            route = None
        elif selected is not None:
            terminal = (
                MatrixResponseDualSourceNominationTerminal.LATE_HORIZON_SOURCE_NOMINATED_FOR_REACTIVE_SOURCE_LAW_ONLY
            )
            route = MatrixResponseDualSourceNominationNominationRoute.LATE
        else:
            terminal = MatrixResponseDualSourceNominationTerminal.NO_TESTED_SOURCE_FEASIBILITY
            route = None
    return MatrixResponseDualSourceNominationResult(
        result_id=f"matrix-source-nomination.result.{attempt_id}",
        attempt_id=attempt_id,
        authorization_id=authorization_id,
        design=design_identity,
        endpoint_identities=identities,
        schedule_summaries=summaries,
        terminal=terminal,
        nomination_route=route,
        selected_schedule_id=selected,
        technically_valid=True,
        reason_codes=(),
        grants_authority=False,
    )


__all__ = [
    'MatrixResponseDualSourceNominationCohort',
    'MatrixResponseDualSourceNominationDesign',
    'MatrixResponseDualSourceNominationNominationRoute',
    'MatrixResponseDualSourceNominationResult',
    'MatrixResponseDualSourceNominationRootEndpoint',
    'MatrixResponseDualSourceNominationScheduleSummary',
    'MatrixResponseDualSourceNominationTerminal',
    'build_source_root_endpoint',
    'reduce_dual_source_nomination',
]
