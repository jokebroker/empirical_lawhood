"Pure Matrix preparation scheduling development factors and the closed schedule decision.\n\nThe module is intentionally limited to the direct 32-root, nine-schedule\ndevelopment assay.  It does not fit a support law, construct a controller,\nopen confirmation, or grant authority.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, Iterable

from empirical_lawhood.adapters.simulators.six_matrix_response.preparation_schedule import MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS, SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster, SixMatrixResponsePreparationWindowSchedulingPreparationWord, SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche, six_matrix_response_preparation_window_scheduling_preparation_words
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .full_intersection_causal_authority import MatrixResponseCausalIntersectionResidenceResponseFactorSample


PRIMARY_TIMESTEP = Decimal("0.001")
RECEIVER_CADENCE = Decimal("0.016")
FIRST_ELIGIBLE_POST_ARRIVAL_TIME = Decimal("0.256")
LAST_POST_ARRIVAL_TIME = Decimal("0.768")
WINDOW_DURATION = Decimal("0.064")
INNER_WINDOW_DURATION = Decimal("0.128")
S_HIT_GAIN_MINIMUM = Decimal("0.10")
TOTAL_RESIDENCE_GAIN_MINIMUM = Decimal("0.032")
LONGEST_RESIDENCE_GAIN_MINIMUM = Decimal("0.032")
CENTRAL_SCHEDULE_ID = "p00-r100"
EXPECTED_RECEIVER_STEPS = tuple(range(256, 769, 16))
EXPECTED_RECEIVER_TIMES = tuple(
    Decimal(value) * PRIMARY_TIMESTEP for value in EXPECTED_RECEIVER_STEPS
)
NONCENTRAL_SCHEDULE_IDS = tuple(
    value for value in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS if value != CENTRAL_SCHEDULE_ID
)


class MatrixResponsePreparationWindowSchedulingFactorRecordDisposition(StrEnum):
    EVALUATED = "EVALUATED"
    TECHNICAL_INVALID = "TECHNICAL_INVALID"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingFactorThresholds(CanonicalRecord):
    """Frozen causal intersection residence factor thresholds used without pulse/action semantics."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-factor-thresholds'

    thresholds_id: str
    phi_min: Decimal
    phi_max: Decimal
    closure_ratio_max: Decimal
    kernel_band_ratio_max: Decimal
    persistence_pass_count: int
    response_geometry_loss_max: Decimal
    response_radius_ratio_max: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.thresholds_id, field_name="thresholds_id")
        for name in (
            "phi_min",
            "phi_max",
            "closure_ratio_max",
            "kernel_band_ratio_max",
            "response_geometry_loss_max",
            "response_radius_ratio_max",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.phi_min > self.phi_max or not 1 <= self.persistence_pass_count <= 16:
            raise ValueError("Matrix preparation scheduling factor thresholds differ")


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingFactorOperands(CanonicalRecord):
    """All scalar operands needed by the exact causal intersection residence/preparation scheduling factor conjunction."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-factor-operands'

    operand_id: str
    root_id: str
    root_index: int
    tranche: SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche
    schedule_id: str
    numerical_view_id: str
    scientific_view_id: str
    acquisition_group_id: str
    physical_independent_unit_id: str
    parent_step: int
    post_arrival_time: Decimal
    thresholds: MatrixResponsePreparationWindowSchedulingFactorThresholds
    phi_y: Decimal
    closure_y: Decimal
    kernel_band_y: Decimal
    spectrum_valid: bool
    persistence_pass_count: int
    strict_kernel_window_pass: bool
    x_rolling_geometric: bool
    resolved_probe_cells: int
    required_probe_cells: int
    probe_invariants_pass: bool
    geometry_loss: Decimal | None
    radius_loss: Decimal | None
    geometry_radius_ratio: Decimal | None

    def __post_init__(self) -> None:
        for name in (
            "operand_id",
            "root_id",
            "schedule_id",
            "numerical_view_id",
            "scientific_view_id",
            "acquisition_group_id",
            "physical_independent_unit_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.tranche is not SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche.DEVELOPMENT
            or not 0 <= self.root_index < 32
            or self.schedule_id not in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS
            or self.parent_step < 0
            or self.physical_independent_unit_id != self.root_id
            or self.post_arrival_time != Decimal(self.parent_step) * PRIMARY_TIMESTEP
        ):
            raise ValueError("Matrix preparation scheduling factor coordinate differs from development")
        if self.required_probe_cells != 18 or not 0 <= self.resolved_probe_cells <= 18:
            raise ValueError("Matrix preparation scheduling factor operand probe roster differs")
        if not 0 <= self.persistence_pass_count <= 16:
            raise ValueError("Matrix preparation scheduling factor persistence count differs")
        for name in ("post_arrival_time", "phi_y", "closure_y", "kernel_band_y"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for name in ("geometry_loss", "radius_loss", "geometry_radius_ratio"):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name, minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingFactorSample(CanonicalRecord):
    """One receiver sample with every noncompensating factor exposed."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-factor-sample'

    sample_id: str
    operands: MatrixResponsePreparationWindowSchedulingFactorOperands
    amplitude_pass: bool
    closure_pass: bool
    spectrum_valid: bool
    kernel_pass: bool
    persistence_count_pass: bool
    strict_kernel_window_pass: bool
    x_exclusion_pass: bool
    response_law_pass: bool
    structural_geometry_pass: bool
    full_intersection_pass: bool
    predictive_skill_only: bool
    structure_only: bool
    amplitude_only: bool
    radius_only_without_structural_geometry: bool
    false_x_admission: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.sample_id, field_name="sample_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        thresholds = self.operands.thresholds
        expected_factors = (
            thresholds.phi_min <= self.operands.phi_y <= thresholds.phi_max,
            self.operands.closure_y <= thresholds.closure_ratio_max,
            self.operands.spectrum_valid,
            self.operands.spectrum_valid
            and self.operands.kernel_band_y <= thresholds.kernel_band_ratio_max,
            self.operands.persistence_pass_count >= thresholds.persistence_pass_count,
            self.operands.strict_kernel_window_pass,
            not self.operands.x_rolling_geometric,
        )
        observed_factors = (
            self.amplitude_pass,
            self.closure_pass,
            self.spectrum_valid,
            self.kernel_pass,
            self.persistence_count_pass,
            self.strict_kernel_window_pass,
            self.x_exclusion_pass,
        )
        if observed_factors != expected_factors:
            raise ValueError("Matrix preparation scheduling factor ledger differs from its primitive operands")
        structural = all(expected_factors)
        response = bool(
            self.operands.resolved_probe_cells == self.operands.required_probe_cells
            and self.operands.probe_invariants_pass
            and self.operands.geometry_loss is not None
            and self.operands.geometry_radius_ratio is not None
            and self.operands.geometry_loss <= thresholds.response_geometry_loss_max
            and self.operands.geometry_radius_ratio <= thresholds.response_radius_ratio_max
        )
        if self.structural_geometry_pass != structural or self.response_law_pass != response:
            raise ValueError("Matrix preparation scheduling factor truth differs")
        if self.full_intersection_pass != (response and structural):
            raise ValueError("Matrix preparation scheduling full endpoint is not response AND structure")
        if self.predictive_skill_only != (response and not structural):
            raise ValueError("Matrix preparation scheduling predictive-only diagnostic differs")
        if self.structure_only != (structural and not response):
            raise ValueError("Matrix preparation scheduling structure-only diagnostic differs")
        if self.amplitude_only != (self.amplitude_pass and not self.full_intersection_pass):
            raise ValueError("Matrix preparation scheduling amplitude-only diagnostic differs")
        radius_only = bool(
            self.operands.radius_loss is not None
            and self.operands.geometry_loss is not None
            and self.operands.radius_loss <= self.operands.geometry_loss
            and not structural
        )
        if self.radius_only_without_structural_geometry != radius_only:
            raise ValueError("Matrix preparation scheduling radius-only diagnostic differs")
        pre_x_structural = all(expected_factors[:-1])
        if self.false_x_admission != (
            response and pre_x_structural and self.operands.x_rolling_geometric
        ):
            raise ValueError("Matrix preparation scheduling X-false-admission diagnostic differs")


def evaluate_factor_sample(operands: MatrixResponsePreparationWindowSchedulingFactorOperands) -> MatrixResponsePreparationWindowSchedulingFactorSample:
    """Apply the exact causal intersection residence factor rules to scalar preparation scheduling operands."""

    thresholds = operands.thresholds
    amplitude = thresholds.phi_min <= operands.phi_y <= thresholds.phi_max
    closure = operands.closure_y <= thresholds.closure_ratio_max
    spectrum = operands.spectrum_valid
    kernel = spectrum and operands.kernel_band_y <= thresholds.kernel_band_ratio_max
    persistence = operands.persistence_pass_count >= thresholds.persistence_pass_count
    strict_kernel = operands.strict_kernel_window_pass
    x_exclusion = not operands.x_rolling_geometric
    response = bool(
        operands.resolved_probe_cells == operands.required_probe_cells
        and operands.probe_invariants_pass
        and operands.geometry_loss is not None
        and operands.geometry_radius_ratio is not None
        and operands.geometry_loss <= thresholds.response_geometry_loss_max
        and operands.geometry_radius_ratio <= thresholds.response_radius_ratio_max
    )
    structural = bool(
        amplitude
        and closure
        and spectrum
        and kernel
        and persistence
        and strict_kernel
        and x_exclusion
    )
    reasons: set[str] = set()
    if not amplitude:
        reasons.add("amplitude-failed")
    if not closure:
        reasons.add("closure-failed")
    if not spectrum:
        reasons.add("spectrum-invalid")
    if not kernel:
        reasons.add("kernel-failed")
    if not persistence:
        reasons.add("persistence-count-failed")
    if not strict_kernel:
        reasons.add("strict-kernel-window-failed")
    if not x_exclusion:
        reasons.add("x-exclusion-failed")
    if operands.resolved_probe_cells != operands.required_probe_cells:
        reasons.add("probe-cells-unresolved")
    if not operands.probe_invariants_pass:
        reasons.add("probe-invariants-failed")
    if (
        operands.geometry_loss is None
        or operands.geometry_loss > thresholds.response_geometry_loss_max
    ):
        reasons.add("geometry-loss-failed")
    if (
        operands.geometry_radius_ratio is None
        or operands.geometry_radius_ratio > thresholds.response_radius_ratio_max
    ):
        reasons.add("radius-contrast-failed")
    radius_only = bool(
        operands.radius_loss is not None
        and operands.geometry_loss is not None
        and operands.radius_loss <= operands.geometry_loss
        and not structural
    )
    pre_x_structural = bool(
        amplitude and closure and spectrum and kernel and persistence and strict_kernel
    )
    return MatrixResponsePreparationWindowSchedulingFactorSample(
        sample_id=(
            f"matrix-preparation-scheduling.sample.{operands.root_id}.{operands.schedule_id}."
            f"{operands.scientific_view_id}.step-{operands.parent_step}"
        ),
        operands=operands,
        amplitude_pass=amplitude,
        closure_pass=closure,
        spectrum_valid=spectrum,
        kernel_pass=kernel,
        persistence_count_pass=persistence,
        strict_kernel_window_pass=strict_kernel,
        x_exclusion_pass=x_exclusion,
        response_law_pass=response,
        structural_geometry_pass=structural,
        full_intersection_pass=response and structural,
        predictive_skill_only=response and not structural,
        structure_only=structural and not response,
        amplitude_only=amplitude and not (response and structural),
        radius_only_without_structural_geometry=radius_only,
        false_x_admission=response and pre_x_structural and operands.x_rolling_geometric,
        reason_codes=tuple(sorted(reasons)),
    )


def causal_intersection_residence_hold_compatible_operands(
    sample: MatrixResponseCausalIntersectionResidenceResponseFactorSample,
    *,
    root_id: str,
    root_index: int,
    schedule_id: str,
    numerical_view_id: str,
    scientific_view_id: str,
    acquisition_group_id: str,
    post_arrival_time: Decimal,
    thresholds: MatrixResponsePreparationWindowSchedulingFactorThresholds,
) -> MatrixResponsePreparationWindowSchedulingFactorOperands:
    """Project an causal intersection residence HOLD sample without constructing an causal intersection residence branch for preparation scheduling."""

    return MatrixResponsePreparationWindowSchedulingFactorOperands(
        operand_id=f"matrix-preparation-scheduling.operand.parity.{sample.sample_id}",
        root_id=root_id,
        root_index=root_index,
        tranche=SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche.DEVELOPMENT,
        schedule_id=schedule_id,
        numerical_view_id=numerical_view_id,
        scientific_view_id=scientific_view_id,
        acquisition_group_id=acquisition_group_id,
        physical_independent_unit_id=root_id,
        parent_step=sample.parent_step,
        post_arrival_time=post_arrival_time,
        thresholds=thresholds,
        phi_y=sample.phi_y,
        closure_y=sample.closure_y,
        kernel_band_y=sample.kernel_band_y,
        spectrum_valid=sample.spectrum_valid,
        persistence_pass_count=sample.persistence_pass_count,
        strict_kernel_window_pass=sample.strict_kernel_window_pass,
        x_rolling_geometric=sample.x_rolling_geometric,
        resolved_probe_cells=sample.resolved_probe_cells,
        required_probe_cells=sample.required_probe_cells,
        probe_invariants_pass=sample.probe_invariants_pass,
        geometry_loss=sample.geometry_loss,
        radius_loss=sample.radius_loss,
        geometry_radius_ratio=sample.geometry_radius_ratio,
    )


def assert_causal_intersection_residence_hold_factor_parity(
    source: MatrixResponseCausalIntersectionResidenceResponseFactorSample,
    target: MatrixResponsePreparationWindowSchedulingFactorSample,
) -> None:
    """Require exact numeric and Boolean parity on an causal intersection residence-produced HOLD sample."""

    scalar_fields = (
        "phi_y",
        "closure_y",
        "kernel_band_y",
        "spectrum_valid",
        "persistence_pass_count",
        "strict_kernel_window_pass",
        "x_rolling_geometric",
        "resolved_probe_cells",
        "required_probe_cells",
        "probe_invariants_pass",
        "geometry_loss",
        "radius_loss",
        "geometry_radius_ratio",
    )
    if any(getattr(source, name) != getattr(target.operands, name) for name in scalar_fields):
        raise ValueError("Matrix preparation scheduling numeric causal intersection residence parity failed")
    decision_fields = (
        "amplitude_pass",
        "closure_pass",
        "spectrum_valid",
        "kernel_pass",
        "persistence_count_pass",
        "strict_kernel_window_pass",
        "x_exclusion_pass",
        "response_law_pass",
        "structural_geometry_pass",
        "full_intersection_pass",
        "predictive_skill_only",
        "structure_only",
        "reason_codes",
    )
    if any(getattr(source, name) != getattr(target, name) for name in decision_fields):
        raise ValueError("Matrix preparation scheduling Boolean causal intersection residence parity failed")


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingFactorParityProjection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-factor-parity-projection'

    phi_y: Decimal
    amplitude_pass: bool
    closure_y: Decimal
    closure_pass: bool
    kernel_band_y: Decimal
    spectrum_valid: bool
    kernel_pass: bool
    persistence_pass_count: int
    persistence_count_pass: bool
    strict_kernel_window_pass: bool
    x_rolling_geometric: bool
    x_exclusion_pass: bool
    resolved_probe_cells: int
    required_probe_cells: int
    probe_invariants_pass: bool
    geometry_loss: Decimal | None
    radius_loss: Decimal | None
    geometry_radius_ratio: Decimal | None
    response_law_pass: bool
    structural_geometry_pass: bool
    full_intersection_pass: bool
    predictive_skill_only: bool
    structure_only: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def _parity_projection(source: MatrixResponseCausalIntersectionResidenceResponseFactorSample | MatrixResponsePreparationWindowSchedulingFactorSample) -> MatrixResponsePreparationWindowSchedulingFactorParityProjection:
    operands = source.operands if isinstance(source, MatrixResponsePreparationWindowSchedulingFactorSample) else source
    return MatrixResponsePreparationWindowSchedulingFactorParityProjection(
        phi_y=operands.phi_y,
        amplitude_pass=source.amplitude_pass,
        closure_y=operands.closure_y,
        closure_pass=source.closure_pass,
        kernel_band_y=operands.kernel_band_y,
        spectrum_valid=source.spectrum_valid,
        kernel_pass=source.kernel_pass,
        persistence_pass_count=operands.persistence_pass_count,
        persistence_count_pass=source.persistence_count_pass,
        strict_kernel_window_pass=source.strict_kernel_window_pass,
        x_rolling_geometric=operands.x_rolling_geometric,
        x_exclusion_pass=source.x_exclusion_pass,
        resolved_probe_cells=operands.resolved_probe_cells,
        required_probe_cells=operands.required_probe_cells,
        probe_invariants_pass=operands.probe_invariants_pass,
        geometry_loss=operands.geometry_loss,
        radius_loss=operands.radius_loss,
        geometry_radius_ratio=operands.geometry_radius_ratio,
        response_law_pass=source.response_law_pass,
        structural_geometry_pass=source.structural_geometry_pass,
        full_intersection_pass=source.full_intersection_pass,
        predictive_skill_only=source.predictive_skill_only,
        structure_only=source.structure_only,
        reason_codes=source.reason_codes,
    )


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingFactorParityReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-factor-parity-receipt'

    receipt_id: str
    source_sample: ObjectIdentity
    target_sample: ObjectIdentity
    source_projection_sha256: str
    target_projection_sha256: str
    numeric_parity: bool
    byte_parity: bool
    inference_unit_count: int = 0

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if (
            self.source_sample.object_schema != MatrixResponseCausalIntersectionResidenceResponseFactorSample.SCHEMA
            or self.target_sample.object_schema != MatrixResponsePreparationWindowSchedulingFactorSample.SCHEMA
        ):
            raise ValueError("Matrix preparation scheduling parity receipt requires exact causal intersection residence/preparation scheduling samples")
        validate_sha256(self.source_projection_sha256, field_name="source_projection_sha256")
        validate_sha256(self.target_projection_sha256, field_name="target_projection_sha256")
        if (
            not self.numeric_parity
            or not self.byte_parity
            or self.source_projection_sha256 != self.target_projection_sha256
            or self.inference_unit_count != 0
        ):
            raise ValueError("Matrix preparation scheduling factor parity did not close")


def factor_parity_receipt(
    source: MatrixResponseCausalIntersectionResidenceResponseFactorSample,
    target: MatrixResponsePreparationWindowSchedulingFactorSample,
) -> MatrixResponsePreparationWindowSchedulingFactorParityReceipt:
    assert_causal_intersection_residence_hold_factor_parity(source, target)
    source_digest = sha256(canonical_json_bytes(_parity_projection(source))).hexdigest()
    target_digest = sha256(canonical_json_bytes(_parity_projection(target))).hexdigest()
    return MatrixResponsePreparationWindowSchedulingFactorParityReceipt(
        receipt_id=f"matrix-preparation-scheduling.factor-parity.{source.sample_id}",
        source_sample=ObjectIdentity.from_record(source.sample_id, source),
        target_sample=ObjectIdentity.from_record(target.sample_id, target),
        source_projection_sha256=source_digest,
        target_projection_sha256=target_digest,
        numeric_parity=True,
        byte_parity=source_digest == target_digest,
    )


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingResidenceInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-residence-interval'

    interval_id: str
    entry_time: Decimal
    exit_time: Decimal
    duration: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.interval_id, field_name="interval_id")
        for name in ("entry_time", "exit_time", "duration"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if (
            self.entry_time < FIRST_ELIGIBLE_POST_ARRIVAL_TIME
            or self.exit_time > LAST_POST_ARRIVAL_TIME
            or self.exit_time < self.entry_time
            or self.duration != self.exit_time - self.entry_time
        ):
            raise ValueError("Matrix preparation scheduling interval lies outside the post-arrival search grid")


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingTrajectoryOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-trajectory-outcome'

    outcome_id: str
    root_id: str
    root_index: int
    tranche: SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche
    schedule_id: str
    numerical_view_id: str
    scientific_view_id: str
    acquisition_group_id: str
    physical_independent_unit_id: str
    interval_count: int
    intervals: tuple[MatrixResponsePreparationWindowSchedulingResidenceInterval, ...]
    first_entry_time: Decimal | None
    final_exit_time: Decimal | None
    total_residence: Decimal
    longest_residence: Decimal
    s_hit: bool
    s_window: bool
    s_inner_window: bool
    predictive_skill_only_sample_count: int
    structure_only_sample_count: int
    amplitude_only_sample_count: int
    radius_only_sample_count: int
    false_x_admission_count: int
    technically_valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "outcome_id",
            "root_id",
            "schedule_id",
            "numerical_view_id",
            "scientific_view_id",
            "acquisition_group_id",
            "physical_independent_unit_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.tranche is not SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche.DEVELOPMENT
            or not 0 <= self.root_index < 32
            or self.schedule_id not in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS
            or self.physical_independent_unit_id != self.root_id
        ):
            raise ValueError("Matrix preparation scheduling trajectory outcome coordinate differs")
        require_sorted_unique_ids(self.intervals, attribute="interval_id", field_name="intervals")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        counts = (
            self.interval_count,
            self.predictive_skill_only_sample_count,
            self.structure_only_sample_count,
            self.amplitude_only_sample_count,
            self.radius_only_sample_count,
            self.false_x_admission_count,
        )
        if min(counts) < 0 or self.interval_count != len(self.intervals):
            raise ValueError("Matrix preparation scheduling trajectory outcome counts differ")
        expected_first = self.intervals[0].entry_time if self.intervals else None
        expected_final = self.intervals[-1].exit_time if self.intervals else None
        total = sum((value.duration for value in self.intervals), Decimal(0))
        longest = max((value.duration for value in self.intervals), default=Decimal(0))
        if (
            self.first_entry_time != expected_first
            or self.final_exit_time != expected_final
            or self.total_residence != total
            or self.longest_residence != longest
            or self.s_hit != bool(self.intervals)
            or self.s_window != (longest >= WINDOW_DURATION)
            or self.s_inner_window != (longest >= INNER_WINDOW_DURATION)
        ):
            raise ValueError("Matrix preparation scheduling trajectory residence reduction differs")


def reduce_trajectory_window(
    samples: Iterable[MatrixResponsePreparationWindowSchedulingFactorSample],
) -> MatrixResponsePreparationWindowSchedulingTrajectoryOutcome:
    """Reduce the exact 33-point post-arrival receiver ledger."""

    ordered = tuple(samples)
    if len(ordered) != len(EXPECTED_RECEIVER_STEPS):
        raise ValueError("Matrix preparation scheduling trajectory requires the exact 33-sample receiver grid")
    first = ordered[0].operands
    coordinates = (
        first.root_id,
        first.root_index,
        first.tranche,
        first.schedule_id,
        first.numerical_view_id,
        first.scientific_view_id,
        first.acquisition_group_id,
        first.physical_independent_unit_id,
        first.thresholds,
    )
    if any(
        (
            value.operands.root_id,
            value.operands.root_index,
            value.operands.tranche,
            value.operands.schedule_id,
            value.operands.numerical_view_id,
            value.operands.scientific_view_id,
            value.operands.acquisition_group_id,
            value.operands.physical_independent_unit_id,
            value.operands.thresholds,
        )
        != coordinates
        for value in ordered
    ):
        raise ValueError("Matrix preparation scheduling receiver ledger crosses a trajectory/config coordinate")
    if (
        tuple(value.operands.parent_step for value in ordered) != EXPECTED_RECEIVER_STEPS
        or tuple(value.operands.post_arrival_time for value in ordered)
        != EXPECTED_RECEIVER_TIMES
    ):
        raise ValueError("Matrix preparation scheduling receiver ledger is incomplete or unordered")

    intervals: list[MatrixResponsePreparationWindowSchedulingResidenceInterval] = []
    start: Decimal | None = None
    for index, sample in enumerate(ordered):
        current_time = sample.operands.post_arrival_time
        if sample.full_intersection_pass and start is None:
            start = current_time
        if start is not None and (
            not sample.full_intersection_pass or index == len(ordered) - 1
        ):
            end_index = index if sample.full_intersection_pass else index - 1
            end = ordered[end_index].operands.post_arrival_time
            intervals.append(
                MatrixResponsePreparationWindowSchedulingResidenceInterval(
                    interval_id=(
                        f"matrix-preparation-scheduling.interval.{first.root_id}.{first.schedule_id}."
                        f"{first.scientific_view_id}.{len(intervals):03d}"
                    ),
                    entry_time=start,
                    exit_time=end,
                    duration=end - start,
                )
            )
            start = None
    longest = max((value.duration for value in intervals), default=Decimal(0))
    reasons = {reason for sample in ordered for reason in sample.reason_codes}
    if not intervals:
        reasons.add("no-full-intersection-hit")
    if longest < WINDOW_DURATION:
        reasons.add("s-window-duration-below-0p064")
    if longest < INNER_WINDOW_DURATION:
        reasons.add("s-inner-window-duration-below-0p128")
    technically_valid = all(value.operands.probe_invariants_pass for value in ordered)
    if not technically_valid:
        reasons.add("trajectory-probe-invariants-invalid")
    return MatrixResponsePreparationWindowSchedulingTrajectoryOutcome(
        outcome_id=(
            f"matrix-preparation-scheduling.outcome.development.{first.root_id}.{first.schedule_id}."
            f"{first.scientific_view_id}"
        ),
        root_id=first.root_id,
        root_index=first.root_index,
        tranche=first.tranche,
        schedule_id=first.schedule_id,
        numerical_view_id=first.numerical_view_id,
        scientific_view_id=first.scientific_view_id,
        acquisition_group_id=first.acquisition_group_id,
        physical_independent_unit_id=first.physical_independent_unit_id,
        interval_count=len(intervals),
        intervals=tuple(intervals),
        first_entry_time=intervals[0].entry_time if intervals else None,
        final_exit_time=intervals[-1].exit_time if intervals else None,
        total_residence=sum((value.duration for value in intervals), Decimal(0)),
        longest_residence=longest,
        s_hit=bool(intervals),
        s_window=longest >= WINDOW_DURATION,
        s_inner_window=longest >= INNER_WINDOW_DURATION,
        predictive_skill_only_sample_count=sum(value.predictive_skill_only for value in ordered),
        structure_only_sample_count=sum(value.structure_only for value in ordered),
        amplitude_only_sample_count=sum(value.amplitude_only for value in ordered),
        radius_only_sample_count=sum(value.radius_only_without_structural_geometry for value in ordered),
        false_x_admission_count=sum(value.false_x_admission for value in ordered),
        technically_valid=technically_valid,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingTrajectoryFactorRecord(CanonicalRecord):
    """Method result for one independently persisted development trajectory."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-trajectory-factor-record'

    record_id: str
    source_trajectory_result: ObjectIdentity
    source_measurement_receipt: ObjectIdentity
    source_development_roster: ObjectIdentity
    source_scientific_view: ObjectIdentity
    root_id: str
    root_index: int
    tranche: SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche
    schedule_id: str
    numerical_view_id: str
    scientific_view_id: str
    preparation_tape: ObjectIdentity
    observation_tape: ObjectIdentity
    factor_config: ObjectIdentity
    samples: tuple[MatrixResponsePreparationWindowSchedulingFactorSample, ...]
    outcome: MatrixResponsePreparationWindowSchedulingTrajectoryOutcome | None
    disposition: MatrixResponsePreparationWindowSchedulingFactorRecordDisposition
    reason_codes: tuple[str, ...]
    physical_independent_unit_id: str
    acquisition_group_id: str
    grants_authority: bool = False

    def __post_init__(self) -> None:
        for name in (
            "record_id",
            "root_id",
            "schedule_id",
            "numerical_view_id",
            "scientific_view_id",
            "physical_independent_unit_id",
            "acquisition_group_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.source_trajectory_result.object_schema
            != 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-trajectory-result'
            or self.source_measurement_receipt.object_schema
            != 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-measurement-artifact-receipt'
            or self.source_development_roster.object_schema
            != 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-development-roster'
            or self.source_scientific_view.object_schema
            != 'empirical-lawhood/planning/response-acquisition-view'
            or self.source_scientific_view.object_id != self.scientific_view_id
            or self.preparation_tape.object_schema
            != 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-noise-tape-receipt'
            or self.observation_tape.object_schema
            != 'empirical-lawhood/simulators/six-matrix-response/six-matrix-response-preparation-window-scheduling-noise-tape-receipt'
            or self.preparation_tape == self.observation_tape
            or self.factor_config.object_schema
            != 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-factor-evaluation-config'
        ):
            raise ValueError("Matrix preparation scheduling factor record custody differs")
        if (
            self.tranche is not SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche.DEVELOPMENT
            or self.schedule_id not in MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS
            or not 0 <= self.root_index < 32
            or self.physical_independent_unit_id != self.root_id
            or self.grants_authority
        ):
            raise ValueError("Matrix preparation scheduling factor record unit/tranche differs")
        require_sorted_unique_ids(self.samples, attribute="sample_id", field_name="samples")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is MatrixResponsePreparationWindowSchedulingFactorRecordDisposition.EVALUATED:
            if not self.samples or self.outcome is None or self.reason_codes:
                raise ValueError("Matrix preparation scheduling evaluated factor record is incomplete")
            if self.outcome != reduce_trajectory_window(self.samples):
                raise ValueError("Matrix preparation scheduling factor record is not the exact ledger reduction")
            if (
                self.outcome.root_id != self.root_id
                or self.outcome.root_index != self.root_index
                or self.outcome.schedule_id != self.schedule_id
                or self.outcome.numerical_view_id != self.numerical_view_id
                or self.outcome.scientific_view_id != self.scientific_view_id
                or self.outcome.acquisition_group_id != self.acquisition_group_id
            ):
                raise ValueError("Matrix preparation scheduling factor outcome crosses its trajectory")
        elif self.samples or self.outcome is not None or not self.reason_codes:
            raise ValueError("Matrix preparation scheduling unresolved record cannot fabricate factor evidence")

    def intent_to_treat_outcome(self) -> MatrixResponsePreparationWindowSchedulingTrajectoryOutcome:
        if self.outcome is not None:
            return self.outcome
        reasons = tuple(sorted((*self.reason_codes, "intent-to-treat-technical-failure")))
        return MatrixResponsePreparationWindowSchedulingTrajectoryOutcome(
            outcome_id=(
                f"matrix-preparation-scheduling.outcome.development.{self.root_id}.{self.schedule_id}."
                f"{self.scientific_view_id}"
            ),
            root_id=self.root_id,
            root_index=self.root_index,
            tranche=self.tranche,
            schedule_id=self.schedule_id,
            numerical_view_id=self.numerical_view_id,
            scientific_view_id=self.scientific_view_id,
            acquisition_group_id=self.acquisition_group_id,
            physical_independent_unit_id=self.physical_independent_unit_id,
            interval_count=0,
            intervals=(),
            first_entry_time=None,
            final_exit_time=None,
            total_residence=Decimal(0),
            longest_residence=Decimal(0),
            s_hit=False,
            s_window=False,
            s_inner_window=False,
            predictive_skill_only_sample_count=0,
            structure_only_sample_count=0,
            amplitude_only_sample_count=0,
            radius_only_sample_count=0,
            false_x_admission_count=0,
            technically_valid=False,
            reason_codes=reasons,
        )


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingJoinedFactorRootBlock(CanonicalRecord):
    """Exact nine-arm DEVELOPMENT join; one block remains one physical unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-joined-factor-root-block'

    block_id: str
    root_id: str
    root_index: int
    numerical_view_id: str
    source_development_roster: ObjectIdentity
    factor_config: ObjectIdentity
    preparation_tape: ObjectIdentity
    observation_tape: ObjectIdentity
    trajectories: tuple[MatrixResponsePreparationWindowSchedulingTrajectoryFactorRecord, ...]
    scientific_view_ids: tuple[str, ...]
    acquisition_group_ids: tuple[str, ...]
    source_trajectory_result_ids: tuple[str, ...]
    source_measurement_receipt_ids: tuple[str, ...]
    physical_independent_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("block_id", "root_id", "numerical_view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if not 0 <= self.root_index < 32:
            raise ValueError("Matrix preparation scheduling development block index differs")
        require_sorted_unique_ids(
            self.trajectories, attribute="schedule_id", field_name="trajectories"
        )
        if tuple(value.schedule_id for value in self.trajectories) != MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS:
            raise ValueError("Matrix preparation scheduling factor block lacks the exact nine-schedule roster")
        if any(
            value.root_id != self.root_id
            or value.root_index != self.root_index
            or value.tranche is not SixMatrixResponsePreparationWindowSchedulingTrajectoryTranche.DEVELOPMENT
            or value.numerical_view_id != self.numerical_view_id
            or value.source_development_roster != self.source_development_roster
            or value.factor_config != self.factor_config
            or value.preparation_tape != self.preparation_tape
            or value.observation_tape != self.observation_tape
            for value in self.trajectories
        ):
            raise ValueError("Matrix preparation scheduling factor block crosses a root, config, view or tape")
        for name in (
            "scientific_view_ids",
            "acquisition_group_ids",
            "source_trajectory_result_ids",
            "source_measurement_receipt_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        expected_views = tuple(sorted(value.scientific_view_id for value in self.trajectories))
        expected_groups = tuple(sorted(value.acquisition_group_id for value in self.trajectories))
        expected_results = tuple(
            sorted(value.source_trajectory_result.object_id for value in self.trajectories)
        )
        expected_receipts = tuple(
            sorted(value.source_measurement_receipt.object_id for value in self.trajectories)
        )
        if (
            self.scientific_view_ids != expected_views
            or self.acquisition_group_ids != expected_groups
            or self.source_trajectory_result_ids != expected_results
            or self.source_measurement_receipt_ids != expected_receipts
            or any(len(value) != 9 for value in (
                self.scientific_view_ids,
                self.acquisition_group_ids,
                self.source_trajectory_result_ids,
                self.source_measurement_receipt_ids,
            ))
            or self.physical_independent_unit_ids != (self.root_id,)
        ):
            raise ValueError("Matrix preparation scheduling nine-arm occurrences/custody are not distinct with n=1")


def join_factor_trajectory_block(
    records: Iterable[MatrixResponsePreparationWindowSchedulingTrajectoryFactorRecord],
) -> MatrixResponsePreparationWindowSchedulingJoinedFactorRootBlock:
    """Join nine independently persisted records or fail before inference."""

    values = tuple(records)
    if len(values) != 9:
        raise ValueError("Matrix preparation scheduling factor block requires exactly nine records")
    ordered = tuple(sorted(values, key=lambda value: value.schedule_id))
    if tuple(value.schedule_id for value in ordered) != MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS:
        raise ValueError("Matrix preparation scheduling factor block is incomplete or duplicates a schedule")
    first = ordered[0]
    return MatrixResponsePreparationWindowSchedulingJoinedFactorRootBlock(
        block_id=f"matrix-preparation-scheduling.factor-block.development.{first.root_id}.{first.numerical_view_id}",
        root_id=first.root_id,
        root_index=first.root_index,
        numerical_view_id=first.numerical_view_id,
        source_development_roster=first.source_development_roster,
        factor_config=first.factor_config,
        preparation_tape=first.preparation_tape,
        observation_tape=first.observation_tape,
        trajectories=ordered,
        scientific_view_ids=tuple(sorted(value.scientific_view_id for value in ordered)),
        acquisition_group_ids=tuple(sorted(value.acquisition_group_id for value in ordered)),
        source_trajectory_result_ids=tuple(
            sorted(value.source_trajectory_result.object_id for value in ordered)
        ),
        source_measurement_receipt_ids=tuple(
            sorted(value.source_measurement_receipt.object_id for value in ordered)
        ),
        physical_independent_unit_ids=(first.root_id,),
    )


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingScheduleCell(CanonicalRecord):
    """One of the eight frozen noncentral paired DEVELOPMENT contrasts."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-schedule-cell'

    cell_id: str
    preparation_word: SixMatrixResponsePreparationWindowSchedulingPreparationWord
    root_count: int
    s_hit_gain: Decimal
    total_residence_gain: Decimal
    longest_residence_gain: Decimal
    first_half_s_hit_gain: Decimal
    first_half_total_residence_gain: Decimal
    first_half_longest_residence_gain: Decimal
    second_half_s_hit_gain: Decimal
    second_half_total_residence_gain: Decimal
    second_half_longest_residence_gain: Decimal
    selected_s_window_count: int
    selection_margin: Decimal
    minimum_half_margin: Decimal
    central_distance: Decimal
    final_arrival_step: int
    evidence_valid: bool
    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if self.preparation_word.schedule_id not in NONCENTRAL_SCHEDULE_IDS:
            raise ValueError("Matrix preparation scheduling development cell must be noncentral")
        for name in (
            "s_hit_gain",
            "total_residence_gain",
            "longest_residence_gain",
            "first_half_s_hit_gain",
            "first_half_total_residence_gain",
            "first_half_longest_residence_gain",
            "second_half_s_hit_gain",
            "second_half_total_residence_gain",
            "second_half_longest_residence_gain",
            "selection_margin",
            "minimum_half_margin",
            "central_distance",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        ratio = Decimal(self.preparation_word.y_duration_steps) / Decimal(
            self.preparation_word.x_duration_steps
        )
        expected_distance = (
            abs(Decimal(self.preparation_word.delta_start_steps) / Decimal(64))
            + abs((ratio - Decimal(1)) / Decimal("0.25"))
        )
        if (
            self.root_count != 32
            or not 0 <= self.selected_s_window_count <= 32
            or self.central_distance != expected_distance
            or self.final_arrival_step != self.preparation_word.final_arrival_step
            or self.eligible != (self.evidence_valid and not self.reason_codes)
        ):
            raise ValueError("Matrix preparation scheduling development-cell eligibility/geometry differs")


def _mean(values: Iterable[Decimal]) -> Decimal:
    items = tuple(values)
    if not items:
        raise ValueError("Matrix preparation scheduling mean requires observations")
    return sum(items, Decimal(0)) / Decimal(len(items))


def _paired_schedule_cell(
    *,
    word: SixMatrixResponsePreparationWindowSchedulingPreparationWord,
    blocks: tuple[MatrixResponsePreparationWindowSchedulingJoinedFactorRootBlock, ...],
    evidence_valid: bool,
) -> MatrixResponsePreparationWindowSchedulingScheduleCell:
    outcomes = {
        block.root_index: {
            record.schedule_id: record.intent_to_treat_outcome()
            for record in block.trajectories
        }
        for block in blocks
    }
    pairs = tuple(
        (outcomes[index][word.schedule_id], outcomes[index][CENTRAL_SCHEDULE_ID])
        for index in range(32)
    )

    def effects(
        group: tuple[tuple[MatrixResponsePreparationWindowSchedulingTrajectoryOutcome, MatrixResponsePreparationWindowSchedulingTrajectoryOutcome], ...]
    ) -> tuple[Decimal, Decimal, Decimal]:
        return (
            _mean(Decimal(int(left.s_hit) - int(right.s_hit)) for left, right in group),
            _mean(left.total_residence - right.total_residence for left, right in group),
            _mean(left.longest_residence - right.longest_residence for left, right in group),
        )

    overall = effects(pairs)
    first_half = effects(pairs[:16])
    second_half = effects(pairs[16:])
    selection_margin = min(
        (overall[0] - S_HIT_GAIN_MINIMUM) / S_HIT_GAIN_MINIMUM,
        (overall[1] - TOTAL_RESIDENCE_GAIN_MINIMUM) / TOTAL_RESIDENCE_GAIN_MINIMUM,
        (overall[2] - LONGEST_RESIDENCE_GAIN_MINIMUM) / LONGEST_RESIDENCE_GAIN_MINIMUM,
    )
    minimum_half_margin = min(
        (first_half[0] - S_HIT_GAIN_MINIMUM) / S_HIT_GAIN_MINIMUM,
        (first_half[1] - TOTAL_RESIDENCE_GAIN_MINIMUM) / TOTAL_RESIDENCE_GAIN_MINIMUM,
        (first_half[2] - LONGEST_RESIDENCE_GAIN_MINIMUM) / LONGEST_RESIDENCE_GAIN_MINIMUM,
        (second_half[0] - S_HIT_GAIN_MINIMUM) / S_HIT_GAIN_MINIMUM,
        (second_half[1] - TOTAL_RESIDENCE_GAIN_MINIMUM) / TOTAL_RESIDENCE_GAIN_MINIMUM,
        (second_half[2] - LONGEST_RESIDENCE_GAIN_MINIMUM) / LONGEST_RESIDENCE_GAIN_MINIMUM,
    )
    selected_s_window_count = sum(left.s_window for left, _right in pairs)
    reasons: set[str] = set()
    if overall[0] < S_HIT_GAIN_MINIMUM:
        reasons.add("s-hit-gain-below-threshold")
    if overall[1] < TOTAL_RESIDENCE_GAIN_MINIMUM:
        reasons.add("total-residence-gain-below-threshold")
    if overall[2] < LONGEST_RESIDENCE_GAIN_MINIMUM:
        reasons.add("longest-residence-gain-below-threshold")
    if any(value <= 0 for value in (*first_half, *second_half)):
        reasons.add("half-specific-effect-not-strictly-positive")
    if selected_s_window_count < 4:
        reasons.add("selected-s-window-count-below-four")
    if not evidence_valid:
        reasons.add("development-factor-or-custody-invalid")
    ratio = Decimal(word.y_duration_steps) / Decimal(word.x_duration_steps)
    distance = (
        abs(Decimal(word.delta_start_steps) / Decimal(64))
        + abs((ratio - Decimal(1)) / Decimal("0.25"))
    )
    return MatrixResponsePreparationWindowSchedulingScheduleCell(
        cell_id=f"matrix-preparation-scheduling.schedule-cell.{word.schedule_id}",
        preparation_word=word,
        root_count=32,
        s_hit_gain=overall[0],
        total_residence_gain=overall[1],
        longest_residence_gain=overall[2],
        first_half_s_hit_gain=first_half[0],
        first_half_total_residence_gain=first_half[1],
        first_half_longest_residence_gain=first_half[2],
        second_half_s_hit_gain=second_half[0],
        second_half_total_residence_gain=second_half[1],
        second_half_longest_residence_gain=second_half[2],
        selected_s_window_count=selected_s_window_count,
        selection_margin=selection_margin,
        minimum_half_margin=minimum_half_margin,
        central_distance=distance,
        final_arrival_step=word.final_arrival_step,
        evidence_valid=evidence_valid,
        eligible=evidence_valid and not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


class MatrixResponsePreparationWindowSchedulingDevelopmentDisposition(StrEnum):
    SELECTED = "SELECTED"
    NO_PREPARATION_HISTORY_CAUSAL_AUTHORITY = "NO_PREPARATION_HISTORY_CAUSAL_AUTHORITY"
    DEVELOPMENT_EVIDENCE_INCOMPLETE = "DEVELOPMENT_EVIDENCE_INCOMPLETE"
    DEVELOPMENT_TECHNICAL_INVALID = "DEVELOPMENT_TECHNICAL_INVALID"


@dataclass(frozen=True, slots=True)
class MatrixResponsePreparationWindowSchedulingDevelopmentDecision(CanonicalRecord):
    "Mutually exclusive schedule decision over one closed direct development table."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-preparation-window-scheduling-development-decision'

    decision_id: str
    disposition: MatrixResponsePreparationWindowSchedulingDevelopmentDisposition
    observed_root_block_count: int
    complete_valid_root_block_count: int
    observed_trajectory_count: int
    inference_unit_count: int
    root_block_ids: tuple[str, ...]
    physical_independent_unit_ids: tuple[str, ...]
    source_development_roster: ObjectIdentity | None
    factor_config: ObjectIdentity | None
    cells: tuple[MatrixResponsePreparationWindowSchedulingScheduleCell, ...]
    selected_cell: MatrixResponsePreparationWindowSchedulingScheduleCell | None
    reason_codes: tuple[str, ...]
    no_top_up: bool
    grants_authority: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        require_sorted_unique_strings(self.root_block_ids, field_name="root_block_ids")
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
        )
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.observed_root_block_count != len(self.root_block_ids)
            or self.inference_unit_count != len(self.physical_independent_unit_ids)
            or self.observed_trajectory_count != 9 * self.observed_root_block_count
            or not self.no_top_up
            or self.grants_authority
        ):
            raise ValueError("Matrix preparation scheduling unit accounting or no-top-up contract differs")
        closed = self.observed_root_block_count == self.inference_unit_count == 32
        if closed:
            if (
                self.source_development_roster is None
                or self.factor_config is None
                or len(self.cells) != 8
                or tuple(value.preparation_word.schedule_id for value in self.cells)
                != NONCENTRAL_SCHEDULE_IDS
            ):
                raise ValueError("Matrix preparation scheduling closed schedule-decision table lacks its exact eight cells")
        elif self.cells or self.selected_cell is not None:
            raise ValueError("Matrix preparation scheduling incomplete schedule-decision evidence cannot select or construct cells")
        if self.disposition is MatrixResponsePreparationWindowSchedulingDevelopmentDisposition.SELECTED:
            if (
                not closed
                or self.selected_cell is None
                or not self.selected_cell.eligible
                or self.complete_valid_root_block_count != 32
                or self.reason_codes
            ):
                raise ValueError("Matrix preparation scheduling selected disposition lacks a valid complete table")
        elif self.disposition is MatrixResponsePreparationWindowSchedulingDevelopmentDisposition.NO_PREPARATION_HISTORY_CAUSAL_AUTHORITY:
            if (
                not closed
                or self.complete_valid_root_block_count != 32
                or self.selected_cell is not None
                or any(value.eligible for value in self.cells)
                or self.reason_codes
            ):
                raise ValueError("Matrix preparation scheduling scientific negative is not a complete valid table")
        elif self.disposition is MatrixResponsePreparationWindowSchedulingDevelopmentDisposition.DEVELOPMENT_EVIDENCE_INCOMPLETE:
            if closed or not self.reason_codes:
                raise ValueError("Matrix preparation scheduling incomplete evidence disposition differs")
        elif (
            not closed
            or self.complete_valid_root_block_count == 32
            or self.selected_cell is not None
            or not self.reason_codes
        ):
            raise ValueError("Matrix preparation scheduling technical-invalid disposition differs")


def reduce_development_blocks(
    blocks: Iterable[MatrixResponsePreparationWindowSchedulingJoinedFactorRootBlock],
    *,
    roster: SixMatrixResponsePreparationWindowSchedulingDevelopmentRoster,
) -> MatrixResponsePreparationWindowSchedulingDevelopmentDecision:
    "Close the schedule decision from exactly 32 nine-arm blocks; never top up or partially select."

    values = tuple(blocks)
    unique_ids = tuple(sorted({value.block_id for value in values}))
    unique_roots = tuple(sorted({value.root_id for value in values}))
    root_indices = {value.root_index for value in values}
    complete_roster = bool(
        len(values) == 32
        and len(unique_ids) == 32
        and len(unique_roots) == 32
        and root_indices == set(range(32))
    )
    if not complete_roster:
        reasons = []
        if len(values) != 32:
            reasons.append("development-root-block-count-not-32")
        if len(unique_ids) != len(values):
            reasons.append("duplicate-development-block-identity")
        if len(unique_roots) != len(values):
            reasons.append("duplicate-physical-independent-unit")
        if root_indices != set(range(32)):
            reasons.append("development-root-index-roster-incomplete")
        return MatrixResponsePreparationWindowSchedulingDevelopmentDecision(
            decision_id="matrix-preparation-scheduling.development-decision",
            disposition=MatrixResponsePreparationWindowSchedulingDevelopmentDisposition.DEVELOPMENT_EVIDENCE_INCOMPLETE,
            observed_root_block_count=len(unique_ids),
            complete_valid_root_block_count=0,
            observed_trajectory_count=9 * len(unique_ids),
            inference_unit_count=len(unique_roots),
            root_block_ids=unique_ids,
            physical_independent_unit_ids=unique_roots,
            source_development_roster=None,
            factor_config=None,
            cells=(),
            selected_cell=None,
            reason_codes=tuple(sorted(reasons)),
            no_top_up=True,
        )
    ordered = tuple(sorted(values, key=lambda value: value.root_index))
    first = ordered[0]
    roster_identity = ObjectIdentity.from_record(roster.roster_id, roster)
    compatible = all(
        value.source_development_roster == roster_identity
        and value.factor_config == first.factor_config
        and value.numerical_view_id == first.numerical_view_id
        and value.numerical_view_id == roster.numerical_view_id
        and value.root_id == roster.physical_independent_unit_ids[value.root_index]
        for value in ordered
    )
    block_valid = tuple(
        compatible
        and all(
            record.disposition is MatrixResponsePreparationWindowSchedulingFactorRecordDisposition.EVALUATED
            and record.outcome is not None
            and record.outcome.technically_valid
            for record in block.trajectories
        )
        for block in ordered
    )
    complete_valid_count = sum(block_valid)
    evidence_valid = compatible and complete_valid_count == 32
    words = {
        value.schedule_id: value for value in six_matrix_response_preparation_window_scheduling_preparation_words()
    }
    cells = tuple(
        _paired_schedule_cell(
            word=words[schedule_id],
            blocks=ordered,
            evidence_valid=evidence_valid,
        )
        for schedule_id in NONCENTRAL_SCHEDULE_IDS
    )
    if not evidence_valid:
        reasons = []
        if not compatible:
            reasons.append("development-config-or-view-identity-mismatch")
        if complete_valid_count != 32:
            reasons.append("development-factor-record-technical-invalid")
        disposition = MatrixResponsePreparationWindowSchedulingDevelopmentDisposition.DEVELOPMENT_TECHNICAL_INVALID
        selected = None
    else:
        eligible = tuple(value for value in cells if value.eligible)
        selected = (
            min(
                eligible,
                key=lambda value: (
                    -value.selection_margin,
                    -value.minimum_half_margin,
                    value.central_distance,
                    value.final_arrival_step,
                    value.preparation_word.schedule_id,
                ),
            )
            if eligible
            else None
        )
        reasons = []
        disposition = (
            MatrixResponsePreparationWindowSchedulingDevelopmentDisposition.SELECTED
            if selected is not None
            else MatrixResponsePreparationWindowSchedulingDevelopmentDisposition.NO_PREPARATION_HISTORY_CAUSAL_AUTHORITY
        )
    return MatrixResponsePreparationWindowSchedulingDevelopmentDecision(
        decision_id="matrix-preparation-scheduling.development-decision",
        disposition=disposition,
        observed_root_block_count=32,
        complete_valid_root_block_count=complete_valid_count,
        observed_trajectory_count=288,
        inference_unit_count=32,
        root_block_ids=tuple(sorted(value.block_id for value in ordered)),
        physical_independent_unit_ids=tuple(sorted(value.root_id for value in ordered)),
        source_development_roster=roster_identity,
        factor_config=first.factor_config,
        cells=cells,
        selected_cell=selected,
        reason_codes=tuple(sorted(reasons)),
        no_top_up=True,
    )


__all__ = [
    "MATRIX_RESPONSE_PREPARATION_SCHEDULING_SCHEDULE_IDS",
    'MatrixResponsePreparationWindowSchedulingDevelopmentDecision',
    'MatrixResponsePreparationWindowSchedulingDevelopmentDisposition',
    'MatrixResponsePreparationWindowSchedulingFactorOperands',
    'MatrixResponsePreparationWindowSchedulingFactorParityReceipt',
    'MatrixResponsePreparationWindowSchedulingFactorRecordDisposition',
    'MatrixResponsePreparationWindowSchedulingFactorSample',
    'MatrixResponsePreparationWindowSchedulingFactorThresholds',
    'MatrixResponsePreparationWindowSchedulingJoinedFactorRootBlock',
    'MatrixResponsePreparationWindowSchedulingResidenceInterval',
    'MatrixResponsePreparationWindowSchedulingScheduleCell',
    'MatrixResponsePreparationWindowSchedulingTrajectoryFactorRecord',
    'MatrixResponsePreparationWindowSchedulingTrajectoryOutcome',
    "EXPECTED_RECEIVER_STEPS",
    'assert_causal_intersection_residence_hold_factor_parity',
    'evaluate_factor_sample',
    'factor_parity_receipt',
    'join_factor_trajectory_block',
    'reduce_development_blocks',
    'reduce_trajectory_window',
    'causal_intersection_residence_hold_compatible_operands',
]
