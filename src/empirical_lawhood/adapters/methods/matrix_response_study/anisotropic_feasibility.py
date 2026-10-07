"""Frozen anisotropic feasibility anisotropic feasibility experiment and noncompensating finalizer."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from math import sqrt
from typing import Callable, ClassVar, Iterable

import numpy as np

from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView, SixMatrixResponsePreparationSchedule, SixMatrixResponseSixMatrixSourceConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ideal_state
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import BAOABGradientCache, SixMatrixResponseConstitution, baoab_step, derive_rng_stream, fast_receiver
from empirical_lawhood.adapters.simulators.six_matrix_response.spectral import spectral_receiver
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


from .scientific_seed_inputs import anisotropic_member_scientific_ordinal
from .geometry_inputs import MatrixGeometryAllocation
from empirical_lawhood.adapters.simulators.six_matrix_response.history_preparation import FOUR_FAMILY_PREPARATION_IDS


class MatrixResponseAnisotropicFeasibilityStage(StrEnum):
    FEASIBILITY = "FEASIBILITY"
    CONFIRMATION = "CONFIRMATION"
    NUMERICAL_CONCORDANCE = "NUMERICAL_CONCORDANCE"


class MatrixResponseAnisotropicFeasibilityDisposition(StrEnum):
    SUBSTRATE_QUALIFIED = "SUBSTRATE_QUALIFIED"
    NO_CONSTITUTIVE_SUBSTRATE = "NO_CONSTITUTIVE_SUBSTRATE_WITHIN_PREDECLARED_MODEL_FAMILY"


@dataclass(frozen=True, slots=True)
class MatrixResponseAnisotropicFeasibilityFactorDiagnostic(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-anisotropic-feasibility-factor-diagnostic'

    factor_id: str
    alpha_tilde: Decimal
    final_phi: Decimal | None
    final_closure_ratio: Decimal | None
    maximum_kernel_band_ratio: Decimal
    persistence_fraction: Decimal
    qualification_sample_count: int
    spectral_sample_count: int
    geometric: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.factor_id not in {"X", "Y"}:
            raise ValueError("Six-matrix anisotropic feasibility factor must be X or Y")
        validate_decimal(self.alpha_tilde, field_name="alpha_tilde", minimum=Decimal(0))
        for name in ("final_phi", "final_closure_ratio"):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name, minimum=Decimal(0))
        validate_decimal(
            self.maximum_kernel_band_ratio,
            field_name="maximum_kernel_band_ratio",
            minimum=Decimal(0),
        )
        validate_decimal(
            self.persistence_fraction,
            field_name="persistence_fraction",
            minimum=Decimal(0),
        )
        if self.persistence_fraction > Decimal(1):
            raise ValueError("Six-matrix anisotropic feasibility persistence cannot exceed one")
        if min(self.qualification_sample_count, self.spectral_sample_count) < 1:
            raise ValueError("Six-matrix anisotropic feasibility factor requires fast and spectral samples")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.geometric == bool(self.reason_codes):
            raise ValueError("Six-matrix anisotropic feasibility factor reasons differ from geometric status")


@dataclass(frozen=True, slots=True)
class MatrixResponseAnisotropicFeasibilityRolloutSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-anisotropic-feasibility-rollout-summary'

    rollout_id: str
    stage: MatrixResponseAnisotropicFeasibilityStage
    member_id: str
    numerical_view_id: str
    q: int
    alpha_x_index: int
    alpha_y_index: int
    alpha_tilde_x: Decimal
    alpha_tilde_y: Decimal
    history_id: str
    seed_index: int
    requested_steps: int
    completed_steps: int
    rng_seed_sha256: str
    final_state_sha256: str
    factor_x: MatrixResponseAnisotropicFeasibilityFactorDiagnostic
    factor_y: MatrixResponseAnisotropicFeasibilityFactorDiagnostic
    constitution: SixMatrixResponseConstitution
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("rollout_id", "member_id", "numerical_view_id", "history_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.q not in {2, 3} or min(self.alpha_x_index, self.alpha_y_index) < 0:
            raise ValueError("Six-matrix anisotropic feasibility rollout grid geometry differs")
        for name in ("alpha_tilde_x", "alpha_tilde_y"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.seed_index < 0 or self.requested_steps < 1:
            raise ValueError("Six-matrix anisotropic feasibility rollout seed/step bounds differ")
        if not 0 <= self.completed_steps <= self.requested_steps:
            raise ValueError("Six-matrix anisotropic feasibility completed steps exceed requested steps")
        validate_sha256(self.rng_seed_sha256, field_name="rng_seed_sha256")
        validate_sha256(self.final_state_sha256, field_name="final_state_sha256")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_constitution = SixMatrixResponseConstitution(
            f"{int(self.factor_x.geometric)}{int(self.factor_y.geometric)}"
        )
        if self.constitution is not expected_constitution:
            raise ValueError("Six-matrix anisotropic feasibility constitution differs from factor admission")
        expected_valid = self.completed_steps == self.requested_steps and not self.reason_codes
        if self.valid != expected_valid:
            raise ValueError("Six-matrix anisotropic feasibility rollout validity differs from its execution record")


@dataclass(frozen=True, slots=True)
class MatrixResponseAnisotropicFeasibilityCellSupport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-anisotropic-feasibility-cell-support'

    support_id: str
    stage: MatrixResponseAnisotropicFeasibilityStage
    member_id: str
    alpha_x_index: int
    alpha_y_index: int
    alpha_tilde_x: Decimal
    alpha_tilde_y: Decimal
    constitution: SixMatrixResponseConstitution
    seed_recurrence: int
    history_recurrence: int
    recurrent: bool
    interior: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.support_id, field_name="support_id")
        validate_stable_id(self.member_id, field_name="member_id")
        if min(self.alpha_x_index, self.alpha_y_index) < 0:
            raise ValueError("Six-matrix anisotropic feasibility support indices cannot be negative")
        for name in ("alpha_tilde_x", "alpha_tilde_y"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if min(self.seed_recurrence, self.history_recurrence) < 0:
            raise ValueError("Six-matrix anisotropic feasibility recurrence counts cannot be negative")
        if self.interior and not self.recurrent:
            raise ValueError("Six-matrix anisotropic feasibility interior support must first be recurrent")


@dataclass(frozen=True, slots=True)
class MatrixResponseAnisotropicFeasibilityMemberScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-anisotropic-feasibility-member-score'

    score_id: str
    member_id: str
    all_four_constitutions: bool
    minimum_axial_interior_width: int
    seed_recurrence_score: int
    history_diversity_score: int
    numerical_concordance: Decimal
    kernel_separation: Decimal
    false_geometric_admissions: int
    selected: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.score_id, field_name="score_id")
        validate_stable_id(self.member_id, field_name="member_id")
        if (
            min(
                self.minimum_axial_interior_width,
                self.seed_recurrence_score,
                self.history_diversity_score,
                self.false_geometric_admissions,
            )
            < 0
        ):
            raise ValueError("Six-matrix anisotropic feasibility score counts cannot be negative")
        for name in ("numerical_concordance", "kernel_separation"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
            if getattr(self, name) > Decimal(1):
                raise ValueError(f"Six-matrix anisotropic feasibility {name} cannot exceed one")
        if self.selected and not self.all_four_constitutions:
            raise ValueError("Six-matrix anisotropic feasibility cannot select a member without all four constitutions")


@dataclass(frozen=True, slots=True)
class MatrixResponseAnisotropicFeasibilityViewComparison(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-anisotropic-feasibility-view-comparison'

    comparison_id: str
    primary_rollout_id: str
    secondary_rollout_id: str
    primary_constitution: SixMatrixResponseConstitution
    secondary_constitution: SixMatrixResponseConstitution
    concordant: bool

    def __post_init__(self) -> None:
        for name in ("comparison_id", "primary_rollout_id", "secondary_rollout_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.concordant != (self.primary_constitution is self.secondary_constitution):
            raise ValueError("Six-matrix anisotropic feasibility view concordance differs from phase identities")


@dataclass(frozen=True, slots=True)
class MatrixResponseAnisotropicFeasibilityQualificationReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-anisotropic-feasibility-qualification-report'

    report_id: str
    implementation_commit: str
    numerical_qualification_report_sha256: str
    source_config: ObjectIdentity
    rollouts: tuple[MatrixResponseAnisotropicFeasibilityRolloutSummary, ...]
    cell_support: tuple[MatrixResponseAnisotropicFeasibilityCellSupport, ...]
    member_scores: tuple[MatrixResponseAnisotropicFeasibilityMemberScore, ...]
    selected_member_id: str | None
    view_comparisons: tuple[MatrixResponseAnisotropicFeasibilityViewComparison, ...]
    disposition: MatrixResponseAnisotropicFeasibilityDisposition
    reason_codes: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        if len(self.implementation_commit) != 40:
            raise ValueError("Six-matrix anisotropic feasibility report requires one Git SHA-1 implementation commit")
        try:
            int(self.implementation_commit, 16)
        except ValueError as error:
            raise ValueError("Six-matrix anisotropic feasibility implementation commit is not hexadecimal") from error
        validate_sha256(self.numerical_qualification_report_sha256, field_name="numerical_qualification_report_sha256")
        require_sorted_unique_ids(self.rollouts, attribute="rollout_id", field_name="rollouts")
        require_sorted_unique_ids(
            self.cell_support,
            attribute="support_id",
            field_name="cell_support",
        )
        require_sorted_unique_ids(
            self.member_scores,
            attribute="score_id",
            field_name="member_scores",
        )
        require_sorted_unique_ids(
            self.view_comparisons,
            attribute="comparison_id",
            field_name="view_comparisons",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        selected_scores = tuple(value for value in self.member_scores if value.selected)
        if len(selected_scores) > 1 or (
            self.selected_member_id != (selected_scores[0].member_id if selected_scores else None)
        ):
            raise ValueError("Six-matrix anisotropic feasibility selected member differs from the frozen score")
        positive = self.disposition is MatrixResponseAnisotropicFeasibilityDisposition.SUBSTRATE_QUALIFIED
        if positive != bool(self.selected_member_id and not self.reason_codes):
            raise ValueError("Six-matrix anisotropic feasibility disposition differs from its noncompensating gate")
        if (
            self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
            or self.grants_authority
        ):
            raise ValueError("Six-matrix anisotropic feasibility is development-visible nonpromotable evidence")


@dataclass(frozen=True, slots=True)
class _RolloutTask:
    source: SixMatrixResponseSixMatrixSourceConfig
    member: SixMatrixResponseModelFamilyMember
    stage: MatrixResponseAnisotropicFeasibilityStage
    q: int
    alpha_x_index: int
    alpha_y_index: int
    alpha_tilde_x: Decimal
    alpha_tilde_y: Decimal
    history_id: str
    seed_index: int
    numerical_view: SixMatrixResponseNumericalView
    schedule: SixMatrixResponsePreparationSchedule
    step_multiplier: int
    scientific_seed_sha256: str


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("Six-matrix anisotropic feasibility metric is nonfinite")
    return Decimal(repr(float(value)))


def _grid(lower: Decimal, upper: Decimal, count: int) -> tuple[Decimal, ...]:
    increment = (upper - lower) / Decimal(count - 1)
    return tuple(lower + increment * index for index in range(count))


def _couplings(
    *,
    history_id: str,
    step: int,
    ramp_steps: int,
    target_x: float,
    target_y: float,
) -> tuple[float, float]:
    fraction = min(step, ramp_steps) / ramp_steps
    if history_id == "matrix-history.joint-increasing-coupling":
        return target_x * fraction, target_y * fraction
    if history_id == "matrix-history.joint-decreasing-coupling":
        return 8.0 + (target_x - 8.0) * fraction, 8.0 + (target_y - 8.0) * fraction
    first = min(1.0, 2.0 * fraction)
    second = max(0.0, min(1.0, 2.0 * fraction - 1.0))
    if history_id == "matrix-history.x-first-increasing-coupling":
        return target_x * first, target_y * second
    if history_id == "matrix-history.y-first-increasing-coupling":
        return target_x * second, target_y * first
    raise ValueError("Six-matrix anisotropic feasibility history is outside the frozen roster")


def _closure_ratio(*, raw_residual: float, radius: float, q: int, alpha_tilde: float) -> float:
    if alpha_tilde <= 0:
        return float("inf")
    scale = 2.0 * (alpha_tilde / q) / 3.0
    rms_generator_norm = abs(scale) * sqrt((q**2) * max(radius, 0.0) / 3.0)
    return raw_residual / max(rms_generator_norm, np.finfo(np.float64).tiny)


def _phi(*, radius: float, q: int, alpha_tilde: float) -> float:
    c2 = (q**2 - 1.0) / 4.0
    return sqrt(max(radius, 0.0) / ((alpha_tilde / q) ** 2 * c2))


def _kernel_band_ratio(eigenvalues: tuple[Decimal, ...], q: int) -> float:
    values = np.asarray(tuple(float(value) for value in eigenvalues), dtype=np.float64)
    numerator = float(values[q**2 - 1])
    denominator = float(values[q**2])
    if denominator <= np.finfo(np.float64).eps:
        return 1.0
    return numerator / denominator


def _factor_diagnostic(
    *,
    factor_id: str,
    alpha_tilde: float,
    q: int,
    radii: tuple[float, ...],
    closure_residuals: tuple[float, ...],
    kernel_ratios: tuple[float, ...],
    source: SixMatrixResponseSixMatrixSourceConfig,
) -> MatrixResponseAnisotropicFeasibilityFactorDiagnostic:
    thresholds = source.numerical_thresholds
    reasons: set[str] = set()
    if alpha_tilde <= 0:
        reasons.add("zero-coupling-factor")
        final_phi = None
        final_closure = None
        persistence = 0.0
    else:
        phis = tuple(_phi(radius=value, q=q, alpha_tilde=alpha_tilde) for value in radii)
        closures = tuple(
            _closure_ratio(
                raw_residual=residual,
                radius=radius,
                q=q,
                alpha_tilde=alpha_tilde,
            )
            for radius, residual in zip(radii, closure_residuals, strict=True)
        )
        instantaneous = tuple(
            float(thresholds.radius_phi_min) <= phi <= float(thresholds.radius_phi_max)
            and closure <= float(thresholds.su2_closure_ratio_max)
            for phi, closure in zip(phis, closures, strict=True)
        )
        final_phi = phis[-1]
        final_closure = closures[-1]
        persistence = sum(instantaneous) / len(instantaneous)
        if not (float(thresholds.radius_phi_min) <= final_phi <= float(thresholds.radius_phi_max)):
            reasons.add("radius-phi-outside-geometric-band")
        if final_closure > float(thresholds.su2_closure_ratio_max):
            reasons.add("su2-closure-ratio-exceeded")
        if persistence < float(thresholds.persistence_fraction_min):
            reasons.add("qualification-persistence-insufficient")
    maximum_kernel = max(kernel_ratios)
    if maximum_kernel > float(thresholds.approximate_kernel_band_ratio_max):
        reasons.add("approximate-kernel-band-not-separated")
    return MatrixResponseAnisotropicFeasibilityFactorDiagnostic(
        factor_id=factor_id,
        alpha_tilde=_decimal(alpha_tilde),
        final_phi=None if final_phi is None else _decimal(final_phi),
        final_closure_ratio=(None if final_closure is None else _decimal(final_closure)),
        maximum_kernel_band_ratio=_decimal(maximum_kernel),
        persistence_fraction=_decimal(persistence),
        qualification_sample_count=len(radii),
        spectral_sample_count=len(kernel_ratios),
        geometric=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


def _rollout_id(task: _RolloutTask) -> str:
    suffix = (
        f"{task.member.member_id.removeprefix('six-matrix-response.member.')}"
        f".x{task.alpha_x_index:02d}.y{task.alpha_y_index:02d}"
        f".{task.history_id.removeprefix('six-matrix-response.history.')}"
        f".s{task.seed_index}.{task.numerical_view.view_id.removeprefix('six-matrix-response.view.')}"
    )
    return f"six-matrix-response.rollout.{task.stage.value.lower().replace('_', '-')}.{suffix}"


def _run_rollout(task: _RolloutTask) -> MatrixResponseAnisotropicFeasibilityRolloutSummary:
    validate_sha256(task.scientific_seed_sha256, field_name="scientific_seed_sha256")
    gradient_cache_state = BAOABGradientCache()
    rollout_id = _rollout_id(task)
    purpose_id = f"purpose.{rollout_id}"
    rng, rng_receipt = derive_rng_stream(
        seed_root_id="seed.matrix-anisotropic-feasibility",
        purpose_id=purpose_id,
        stream_index=task.seed_index,
        derivation_rule_id=task.numerical_view.stream_derivation_rule_id,
        scientific_seed_sha256=task.scientific_seed_sha256,
    )
    if task.history_id == "matrix-history.joint-decreasing-coupling":
        state = ideal_state(
            q=task.q,
            alpha_tilde_x=8.0,
            alpha_tilde_y=8.0,
            constitution="11",
        )
    else:
        state = ideal_state(
            q=task.q,
            alpha_tilde_x=0.0,
            alpha_tilde_y=0.0,
            constitution="00",
        )
    multiplier = task.step_multiplier
    ramp_steps = task.schedule.ramp_steps * multiplier
    qualification_start = (task.schedule.ramp_steps + task.schedule.dwell_steps) * multiplier
    total_steps = task.schedule.total_steps * multiplier
    fast_cadence = task.schedule.fast_receiver_cadence_steps * multiplier
    spectral_cadence = task.schedule.spectral_receiver_cadence_steps * multiplier
    radii: list[list[float]] = [[], []]
    closures: list[list[float]] = [[], []]
    kernel_ratios: list[list[float]] = [[], []]
    reasons: set[str] = set()
    try:
        for step in range(1, total_steps + 1):
            alpha_x, alpha_y = _couplings(
                history_id=task.history_id,
                step=step,
                ramp_steps=ramp_steps,
                target_x=float(task.alpha_tilde_x),
                target_y=float(task.alpha_tilde_y),
            )
            state = baoab_step(
                state,
                member=task.member,
                numerical_view=task.numerical_view,
                next_alpha_tilde_x=alpha_x,
                next_alpha_tilde_y=alpha_y,
                rng=rng,
                gradient_cache=gradient_cache_state,
            )
            if step > qualification_start and step % fast_cadence == 0:
                receiver = fast_receiver(state=state, member=task.member)
                radii[0].append(float(receiver.radius_x))
                radii[1].append(float(receiver.radius_y))
                closures[0].append(float(receiver.closure_residual_x))
                closures[1].append(float(receiver.closure_residual_y))
                if not receiver.valid:
                    reasons.update(receiver.reason_codes or ("invalid-fast-receiver",))
            if step > qualification_start and step % spectral_cadence == 0:
                spectra = spectral_receiver(
                    receiver_prefix=f"spectrum.{rollout_id}.{step}",
                    q=task.q,
                    positions=state.positions,
                )
                by_sector = {value.sector: value for value in spectra}
                for factor_index, sector in enumerate(("X", "Y")):
                    kernel_ratios[factor_index].append(
                        _kernel_band_ratio(by_sector[sector].eigenvalues, task.q)
                    )
                if not all(value.valid for value in spectra):
                    reasons.add("invalid-spectral-receiver")
        if not state.finite:
            reasons.add("nonfinite-final-state")
    except (FloatingPointError, ValueError, np.linalg.LinAlgError):
        reasons.add("numerical-execution-failure")
    completed_steps = state.step_index
    if completed_steps != total_steps:
        reasons.add("incomplete-integration")
    if not all(radii) or not all(kernel_ratios):
        reasons.add("missing-qualification-receivers")
        # A typed invalid result still needs canonical finite diagnostics.
        for values in radii:
            if not values:
                values.append(0.0)
        for values in closures:
            if not values:
                values.append(0.0)
        for values in kernel_ratios:
            if not values:
                values.append(1.0)
    factors = tuple(
        _factor_diagnostic(
            factor_id=factor_id,
            alpha_tilde=float(alpha),
            q=task.q,
            radii=tuple(radii[index]),
            closure_residuals=tuple(closures[index]),
            kernel_ratios=tuple(kernel_ratios[index]),
            source=task.source,
        )
        for index, (factor_id, alpha) in enumerate(
            (("X", task.alpha_tilde_x), ("Y", task.alpha_tilde_y))
        )
    )
    state_digest = sha256()
    state_digest.update(state.positions.tobytes(order="C"))
    state_digest.update(state.momenta.tobytes(order="C"))
    state_digest.update(str(state.step_index).encode("ascii"))
    state_digest.update(str(task.alpha_tilde_x).encode("ascii"))
    state_digest.update(str(task.alpha_tilde_y).encode("ascii"))
    constitution = SixMatrixResponseConstitution(f"{int(factors[0].geometric)}{int(factors[1].geometric)}")
    return MatrixResponseAnisotropicFeasibilityRolloutSummary(
        rollout_id=rollout_id,
        stage=task.stage,
        member_id=task.member.member_id,
        numerical_view_id=task.numerical_view.view_id,
        q=task.q,
        alpha_x_index=task.alpha_x_index,
        alpha_y_index=task.alpha_y_index,
        alpha_tilde_x=task.alpha_tilde_x,
        alpha_tilde_y=task.alpha_tilde_y,
        history_id=task.history_id,
        seed_index=task.seed_index,
        requested_steps=total_steps,
        completed_steps=completed_steps,
        rng_seed_sha256=rng_receipt.derived_seed_sha256,
        final_state_sha256=state_digest.hexdigest(),
        factor_x=factors[0],
        factor_y=factors[1],
        constitution=constitution,
        valid=completed_steps == total_steps and not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


def run_anisotropic_feasibility_rollout(
    *,
    source: SixMatrixResponseSixMatrixSourceConfig,
    member: SixMatrixResponseModelFamilyMember,
    stage: MatrixResponseAnisotropicFeasibilityStage,
    q: int,
    alpha_x_index: int,
    alpha_y_index: int,
    alpha_tilde_x: Decimal,
    alpha_tilde_y: Decimal,
    history_id: str,
    seed_index: int,
    numerical_view: SixMatrixResponseNumericalView,
    schedule: SixMatrixResponsePreparationSchedule,
    scientific_seed_sha256: str,
    step_multiplier: int = 1,
) -> MatrixResponseAnisotropicFeasibilityRolloutSummary:
    """Execute one fully identified independent anisotropic feasibility matrix trajectory."""

    return _run_rollout(
        _RolloutTask(
            source=source,
            member=member,
            stage=stage,
            q=q,
            alpha_x_index=alpha_x_index,
            alpha_y_index=alpha_y_index,
            alpha_tilde_x=alpha_tilde_x,
            alpha_tilde_y=alpha_tilde_y,
            history_id=history_id,
            seed_index=seed_index,
            numerical_view=numerical_view,
            schedule=schedule,
            step_multiplier=step_multiplier,
            scientific_seed_sha256=scientific_seed_sha256,
        )
    )


def _tasks(
    *,
    source: SixMatrixResponseSixMatrixSourceConfig,
    member: SixMatrixResponseModelFamilyMember,
    stage: MatrixResponseAnisotropicFeasibilityStage,
    allocation: MatrixGeometryAllocation,
) -> tuple[_RolloutTask, ...]:
    envelope = source.feasibility_envelope
    if stage is MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY:
        q = 2
        grid = _grid(
            envelope.feasibility_alpha_tilde_min,
            envelope.feasibility_alpha_tilde_max,
            envelope.feasibility_axis_count,
        )
        histories = envelope.feasibility_history_ids
        seeds = range(envelope.feasibility_seed_count)
        view = source.primary_view
        schedule = envelope.feasibility_schedule
        multiplier = 1
        selected_cells = None
    elif stage is MatrixResponseAnisotropicFeasibilityStage.CONFIRMATION:
        q = 3
        grid = _grid(
            envelope.confirmation_alpha_tilde_min,
            envelope.confirmation_alpha_tilde_max,
            envelope.confirmation_axis_count,
        )
        histories = envelope.confirmation_history_ids
        seeds = range(envelope.confirmation_seed_count)
        view = source.primary_view
        schedule = envelope.confirmation_schedule
        multiplier = 1
        selected_cells = None
    else:
        q = 3
        grid = _grid(
            envelope.confirmation_alpha_tilde_min,
            envelope.confirmation_alpha_tilde_max,
            envelope.confirmation_axis_count,
        )
        histories = envelope.confirmation_history_ids
        seeds = range(1)
        view = source.secondary_view
        schedule = envelope.confirmation_schedule
        multiplier = 2
        # Exact frozen boundary/interior grid strata: 1/4/7 on each axis.
        selected_cells = frozenset((x, y) for x in (1, 4, 7) for y in (1, 4, 7))
    allocated = {cell.coordinate: cell.full_seed_sha256 for cell in allocation.cells}
    return tuple(
        _RolloutTask(
            source=source,
            member=member,
            stage=stage,
            q=q,
            alpha_x_index=x_index,
            alpha_y_index=y_index,
            alpha_tilde_x=alpha_x,
            alpha_tilde_y=alpha_y,
            history_id=history,
            seed_index=seed,
            numerical_view=view,
            schedule=schedule,
            step_multiplier=multiplier,
            scientific_seed_sha256=allocated[(
                anisotropic_member_scientific_ordinal(
                    mass_x=member.mass_x, mass_y=member.mass_y, cross_coupling_gamma=member.cross_coupling_gamma
                ),
                tuple(MatrixResponseAnisotropicFeasibilityStage).index(stage),
                x_index, y_index, FOUR_FAMILY_PREPARATION_IDS.index(history),
                seed, 0 if multiplier == 1 else 1,
            )],
        )
        for x_index, alpha_x in enumerate(grid)
        for y_index, alpha_y in enumerate(grid)
        if selected_cells is None or (x_index, y_index) in selected_cells
        for history in histories
        for seed in seeds
    )


def _execute_tasks(
    tasks: tuple[_RolloutTask, ...],
    *,
    workers: int,
    progress: Callable[[int, int], None] | None,
) -> tuple[MatrixResponseAnisotropicFeasibilityRolloutSummary, ...]:
    if workers < 1:
        raise ValueError("Six-matrix anisotropic feasibility worker count must be positive")
    results: list[MatrixResponseAnisotropicFeasibilityRolloutSummary] = []
    if workers == 1:
        iterator: Iterable[MatrixResponseAnisotropicFeasibilityRolloutSummary] = map(_run_rollout, tasks)
        for value in iterator:
            results.append(value)
            if progress is not None:
                progress(len(results), len(tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for value in executor.map(_run_rollout, tasks, chunksize=4):
                results.append(value)
                if progress is not None:
                    progress(len(results), len(tasks))
    return tuple(sorted(results, key=lambda value: value.rollout_id))


def aggregate_cell_support(
    rollouts: tuple[MatrixResponseAnisotropicFeasibilityRolloutSummary, ...],
    *,
    stage: MatrixResponseAnisotropicFeasibilityStage,
    seed_minimum: int,
    history_minimum: int,
) -> tuple[MatrixResponseAnisotropicFeasibilityCellSupport, ...]:
    """Reduce independent trajectories to recurrent cells, then erode to interiors."""

    if not rollouts:
        return ()
    member_ids = {value.member_id for value in rollouts}
    if len(member_ids) != 1 or any(value.stage is not stage for value in rollouts):
        raise ValueError("Six-matrix anisotropic feasibility cell aggregation cannot pool members or stages")
    grouped: dict[tuple[int, int, SixMatrixResponseConstitution], list[MatrixResponseAnisotropicFeasibilityRolloutSummary]] = {}
    coordinates: dict[tuple[int, int], tuple[Decimal, Decimal]] = {}
    for value in rollouts:
        key = (value.alpha_x_index, value.alpha_y_index, value.constitution)
        grouped.setdefault(key, []).append(value)
        coordinates[(value.alpha_x_index, value.alpha_y_index)] = (
            value.alpha_tilde_x,
            value.alpha_tilde_y,
        )
    recurrence: dict[tuple[int, int, SixMatrixResponseConstitution], tuple[int, int, bool]] = {}
    for key, values in grouped.items():
        valid = tuple(value for value in values if value.valid)
        seed_count = len({value.seed_index for value in valid})
        history_count = len({value.history_id for value in valid})
        recurrence[key] = (
            seed_count,
            history_count,
            seed_count >= seed_minimum and history_count >= history_minimum,
        )
    outputs = []
    member_id = next(iter(member_ids))
    for key in sorted(recurrence, key=lambda value: (value[0], value[1], value[2].value)):
        x_index, y_index, constitution = key
        seed_count, history_count, recurrent = recurrence[key]
        neighbors = (
            (x_index - 1, y_index, constitution),
            (x_index + 1, y_index, constitution),
            (x_index, y_index - 1, constitution),
            (x_index, y_index + 1, constitution),
        )
        interior = recurrent and all(
            recurrence.get(neighbor, (0, 0, False))[2] for neighbor in neighbors
        )
        alpha_x, alpha_y = coordinates[(x_index, y_index)]
        outputs.append(
            MatrixResponseAnisotropicFeasibilityCellSupport(
                support_id=(
                    f"six-matrix-response.support.{stage.value.lower().replace('_', '-')}"
                    f".{member_id.removeprefix('six-matrix-response.member.')}"
                    f".x{x_index:02d}.y{y_index:02d}.{constitution.value}"
                ),
                stage=stage,
                member_id=member_id,
                alpha_x_index=x_index,
                alpha_y_index=y_index,
                alpha_tilde_x=alpha_x,
                alpha_tilde_y=alpha_y,
                constitution=constitution,
                seed_recurrence=seed_count,
                history_recurrence=history_count,
                recurrent=recurrent,
                interior=interior,
            )
        )
    return tuple(outputs)


def _member_score(
    *,
    member_id: str,
    rollouts: tuple[MatrixResponseAnisotropicFeasibilityRolloutSummary, ...],
    support: tuple[MatrixResponseAnisotropicFeasibilityCellSupport, ...],
) -> MatrixResponseAnisotropicFeasibilityMemberScore:
    interiors = tuple(value for value in support if value.interior)
    widths: list[int] = []
    for constitution in SixMatrixResponseConstitution:
        values = tuple(value for value in interiors if value.constitution is constitution)
        widths.extend(
            (
                len({value.alpha_x_index for value in values}),
                len({value.alpha_y_index for value in values}),
            )
        )
    all_four = all(
        any(value.constitution is constitution for value in interiors)
        for constitution in SixMatrixResponseConstitution
    )
    recurrent = tuple(value for value in support if value.recurrent)
    geometric_factors = tuple(
        factor
        for value in rollouts
        if value.valid
        for factor in (value.factor_x, value.factor_y)
        if factor.geometric
    )
    kernel_separation = (
        sum(1.0 - float(value.maximum_kernel_band_ratio) for value in geometric_factors)
        / len(geometric_factors)
        if geometric_factors
        else 0.0
    )
    false_admissions = sum(
        int(value.factor_x.geometric and value.alpha_tilde_x == 0)
        + int(value.factor_y.geometric and value.alpha_tilde_y == 0)
        for value in rollouts
    )
    return MatrixResponseAnisotropicFeasibilityMemberScore(
        score_id=f"six-matrix-response.score.{member_id.removeprefix('six-matrix-response.member.')}",
        member_id=member_id,
        all_four_constitutions=all_four,
        minimum_axial_interior_width=min(widths),
        seed_recurrence_score=sum(value.seed_recurrence for value in recurrent),
        history_diversity_score=sum(value.history_recurrence for value in recurrent),
        numerical_concordance=_decimal(sum(value.valid for value in rollouts) / len(rollouts)),
        kernel_separation=_decimal(max(0.0, min(1.0, kernel_separation))),
        false_geometric_admissions=false_admissions,
        selected=False,
    )


def _selection_key(score: MatrixResponseAnisotropicFeasibilityMemberScore) -> tuple[object, ...]:
    return (
        score.all_four_constitutions,
        score.minimum_axial_interior_width,
        score.seed_recurrence_score,
        score.history_diversity_score,
        score.numerical_concordance,
        score.kernel_separation,
        -score.false_geometric_admissions,
    )


def select_feasibility_member(
    scores: tuple[MatrixResponseAnisotropicFeasibilityMemberScore, ...],
    *, scientific_member_order: tuple[str, ...],
) -> tuple[MatrixResponseAnisotropicFeasibilityMemberScore, ...]:
    if not scores:
        raise ValueError("Six-matrix anisotropic feasibility selection requires the frozen family scores")
    if (
        type(scientific_member_order) is not tuple
        or len(set(scientific_member_order)) != len(scientific_member_order)
        or len({value.member_id for value in scores}) != len(scores)
        or {value.member_id for value in scores} != set(scientific_member_order)
    ):
        raise ValueError("anisotropic selection requires its complete explicit scientific member order")
    by_member = {value.member_id: value for value in scores}
    ordered = tuple(by_member[member_id] for member_id in scientific_member_order)
    eligible = tuple(value for value in ordered if value.all_four_constitutions)
    selected_id = None
    if eligible:
        best_key = max(_selection_key(value) for value in eligible)
        selected_id = next(
            value.member_id for value in eligible if _selection_key(value) == best_key
        )
    return tuple(
        MatrixResponseAnisotropicFeasibilityMemberScore(
            score_id=value.score_id,
            member_id=value.member_id,
            all_four_constitutions=value.all_four_constitutions,
            minimum_axial_interior_width=value.minimum_axial_interior_width,
            seed_recurrence_score=value.seed_recurrence_score,
            history_diversity_score=value.history_diversity_score,
            numerical_concordance=value.numerical_concordance,
            kernel_separation=value.kernel_separation,
            false_geometric_admissions=value.false_geometric_admissions,
            selected=value.member_id == selected_id,
        )
        for value in ordered
    )


def _confirmation_reasons(
    *,
    primary: tuple[MatrixResponseAnisotropicFeasibilityRolloutSummary, ...],
    support: tuple[MatrixResponseAnisotropicFeasibilityCellSupport, ...],
    comparisons: tuple[MatrixResponseAnisotropicFeasibilityViewComparison, ...],
    source: SixMatrixResponseSixMatrixSourceConfig,
) -> tuple[str, ...]:
    reasons: set[str] = set()
    if any(not value.valid for value in primary):
        reasons.add("confirmation-invalid-primary-rollout")
    interiors = tuple(value for value in support if value.interior)
    for constitution in SixMatrixResponseConstitution:
        if not any(value.constitution is constitution for value in interiors):
            reasons.add(f"confirmation-missing-{constitution.value}-interior")
    minimum = source.numerical_thresholds.confirmation_minimum_cells_per_axis
    for constitution in (SixMatrixResponseConstitution.X_ONLY, SixMatrixResponseConstitution.Y_ONLY):
        values = tuple(value for value in interiors if value.constitution is constitution)
        if len({value.alpha_x_index for value in values}) < minimum:
            reasons.add(f"confirmation-{constitution.value}-x-width-insufficient")
        if len({value.alpha_y_index for value in values}) < minimum:
            reasons.add(f"confirmation-{constitution.value}-y-width-insufficient")
    if not comparisons or any(not value.concordant for value in comparisons):
        reasons.add("confirmation-primary-half-step-phase-disagreement")
    return tuple(sorted(reasons))


def run_matrix_response_study_anisotropic_feasibility_feasibility(
    *,
    source: SixMatrixResponseSixMatrixSourceConfig,
    implementation_commit: str,
    numerical_qualification_report_sha256: str,
    allocation: MatrixGeometryAllocation,
    workers: int = 8,
    progress: Callable[[str, int, int], None] | None = None,
    task_executor: Callable[..., tuple[MatrixResponseAnisotropicFeasibilityRolloutSummary, ...]] | None = None,
) -> MatrixResponseAnisotropicFeasibilityQualificationReport:
    """Execute the exact feasibility roster and conditionally execute the exact confirmation roster."""

    validate_sha256(source.fingerprint(), field_name="source_config_fingerprint")
    validate_sha256(numerical_qualification_report_sha256, field_name="numerical_qualification_report_sha256")
    envelope = source.feasibility_envelope
    execute_tasks = _execute_tasks if task_executor is None else task_executor
    expected_primary_steps = (
        len(source.anisotropic_model.family_members)
        * envelope.feasibility_rollouts_per_member
        * envelope.feasibility_schedule.total_steps
        + envelope.confirmation_rollouts * envelope.confirmation_schedule.total_steps
    )
    secondary_steps = 9 * len(envelope.confirmation_history_ids) * 2 * envelope.confirmation_schedule.total_steps
    if expected_primary_steps != 11_999_232 or (
        expected_primary_steps + secondary_steps > envelope.maximum_anisotropic_feasibility_integration_steps
    ):
        raise ValueError("Six-matrix anisotropic feasibility exact work roster differs from the frozen ceiling")
    all_rollouts: list[MatrixResponseAnisotropicFeasibilityRolloutSummary] = []
    all_support: list[MatrixResponseAnisotropicFeasibilityCellSupport] = []
    raw_scores: list[MatrixResponseAnisotropicFeasibilityMemberScore] = []
    for member in source.anisotropic_model.family_members:
        tasks = _tasks(source=source, member=member, stage=MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY, allocation=allocation)
        progress_callback: Callable[[int, int], None] | None = None
        if progress is not None:
            member_id = member.member_id

            def progress_callback(done: int, total: int, member_id: str = member_id) -> None:
                assert progress is not None
                progress(f"feasibility:{member_id}", done, total)

        member_rollouts = execute_tasks(
            tasks,
            workers=workers,
            progress=progress_callback,
        )
        member_support = aggregate_cell_support(
            member_rollouts,
            stage=MatrixResponseAnisotropicFeasibilityStage.FEASIBILITY,
            seed_minimum=2,
            history_minimum=2,
        )
        all_rollouts.extend(member_rollouts)
        all_support.extend(member_support)
        raw_scores.append(
            _member_score(
                member_id=member.member_id,
                rollouts=member_rollouts,
                support=member_support,
            )
        )
    scores = select_feasibility_member(
        tuple(raw_scores),
        scientific_member_order=tuple(
            member.member_id for member in sorted(
                source.anisotropic_model.family_members,
                key=lambda value: anisotropic_member_scientific_ordinal(
                    mass_x=value.mass_x, mass_y=value.mass_y, cross_coupling_gamma=value.cross_coupling_gamma
                ),
            )
        ),
    )
    selected_score = next((value for value in scores if value.selected), None)
    comparisons: tuple[MatrixResponseAnisotropicFeasibilityViewComparison, ...] = ()
    reasons: tuple[str, ...]
    if selected_score is None:
        reasons = ("feasibility-no-member-with-all-four-constitution-interiors",)
    else:
        member = next(
            value
            for value in source.anisotropic_model.family_members
            if value.member_id == selected_score.member_id
        )
        confirmation_primary = execute_tasks(
            _tasks(source=source, member=member, stage=MatrixResponseAnisotropicFeasibilityStage.CONFIRMATION, allocation=allocation),
            workers=workers,
            progress=(
                None
                if progress is None
                else lambda done, total: progress("confirmation:primary", done, total)
            ),
        )
        confirmation_support = aggregate_cell_support(
            confirmation_primary,
            stage=MatrixResponseAnisotropicFeasibilityStage.CONFIRMATION,
            seed_minimum=source.numerical_thresholds.confirmation_seed_recurrence_minimum,
            history_minimum=source.numerical_thresholds.confirmation_history_recurrence_minimum,
        )
        secondary = execute_tasks(
            _tasks(
                source=source,
                member=member,
                stage=MatrixResponseAnisotropicFeasibilityStage.NUMERICAL_CONCORDANCE,
                allocation=allocation,
            ),
            workers=workers,
            progress=(
                None
                if progress is None
                else lambda done, total: progress("confirmation:secondary", done, total)
            ),
        )
        primary_by_key = {
            (
                value.alpha_x_index,
                value.alpha_y_index,
                value.history_id,
                value.seed_index,
            ): value
            for value in confirmation_primary
        }
        comparison_values = []
        for value in secondary:
            primary = primary_by_key[
                (
                    value.alpha_x_index,
                    value.alpha_y_index,
                    value.history_id,
                    value.seed_index,
                )
            ]
            comparison_values.append(
                MatrixResponseAnisotropicFeasibilityViewComparison(
                    comparison_id=(
                        f"six-matrix-response.view-comparison.x{value.alpha_x_index:02d}"
                        f".y{value.alpha_y_index:02d}"
                        f".{value.history_id.removeprefix('six-matrix-response.history.')}"
                    ),
                    primary_rollout_id=primary.rollout_id,
                    secondary_rollout_id=value.rollout_id,
                    primary_constitution=primary.constitution,
                    secondary_constitution=value.constitution,
                    concordant=primary.constitution is value.constitution,
                )
            )
        comparisons = tuple(sorted(comparison_values, key=lambda value: value.comparison_id))
        reasons = _confirmation_reasons(
            primary=confirmation_primary,
            support=confirmation_support,
            comparisons=comparisons,
            source=source,
        )
        if any(not value.valid for value in secondary):
            reasons = tuple(sorted(set(reasons) | {"confirmation-invalid-secondary-rollout"}))
        all_rollouts.extend(confirmation_primary)
        all_rollouts.extend(secondary)
        all_support.extend(confirmation_support)
    # This remains the outcome-visible feasibility candidate even if confirmation rejects it.
    selected_member_id = selected_score.member_id if selected_score is not None else None
    return MatrixResponseAnisotropicFeasibilityQualificationReport(
        report_id="six-matrix-response.anisotropic-feasibility-qualification-report",
        implementation_commit=implementation_commit,
        numerical_qualification_report_sha256=numerical_qualification_report_sha256,
        source_config=ObjectIdentity.from_record(source.config_id, source),
        rollouts=tuple(sorted(all_rollouts, key=lambda value: value.rollout_id)),
        cell_support=tuple(sorted(all_support, key=lambda value: value.support_id)),
        member_scores=scores,
        selected_member_id=selected_member_id,
        view_comparisons=comparisons,
        disposition=(
            MatrixResponseAnisotropicFeasibilityDisposition.SUBSTRATE_QUALIFIED
            if selected_member_id is not None and not reasons
            else MatrixResponseAnisotropicFeasibilityDisposition.NO_CONSTITUTIVE_SUBSTRATE
        ),
        reason_codes=reasons,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        grants_authority=False,
    )


__all__ = [
    'MatrixResponseAnisotropicFeasibilityCellSupport',
    'MatrixResponseAnisotropicFeasibilityDisposition',
    'MatrixResponseAnisotropicFeasibilityFactorDiagnostic',
    'MatrixResponseAnisotropicFeasibilityMemberScore',
    'MatrixResponseAnisotropicFeasibilityQualificationReport',
    'MatrixResponseAnisotropicFeasibilityRolloutSummary',
    'MatrixResponseAnisotropicFeasibilityStage',
    'MatrixResponseAnisotropicFeasibilityViewComparison',
    'aggregate_cell_support',
    'run_anisotropic_feasibility_rollout',
    'run_matrix_response_study_anisotropic_feasibility_feasibility',
    'select_feasibility_member',
]
