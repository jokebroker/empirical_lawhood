"""Sole noncompensating matrix response numerical qualification finalizer."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from math import sqrt
from time import perf_counter
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView, SixMatrixResponseSixMatrixSourceConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.extension_bundle import SIX_MATRIX_RESPONSE_CAPABILITY
from empirical_lawhood.adapters.simulators.six_matrix_response.gradients import SixMatrixParameters, analytic_gradient_terms, optimized_action_terms, reference_action_terms
from empirical_lawhood.adapters.simulators.six_matrix_response.model import SixMatrixState, conjugate_state, hermitian_part, ideal_positions
from empirical_lawhood.adapters.simulators.six_matrix_response.simulation import SixMatrixResponseConstitution, SixMatrixResponseEpisodeRequest, SixMatrixResponseEpisodeTerminal, SixMatrixResponseNativeActionRequest, SixMatrixResponseScaledCouplings, SixMatrixResponseSixMatrixEpisodeEngine, fast_receiver, hermitian_noise, state_from_checkpoint
from empirical_lawhood.adapters.simulators.six_matrix_response.spectral import adjoint_laplacian_basis, adjoint_laplacian_explicit, ideal_joint_laplacian_eigenvalues, ideal_laplacian_eigenvalues, laplacian_eigenvalues, spectral_receiver
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


class MatrixResponseNumericalQualificationDisposition(StrEnum):
    NUMERICALLY_QUALIFIED = "NUMERICALLY_QUALIFIED"
    NUMERICAL_MODEL_UNQUALIFIED = "NUMERICAL_MODEL_UNQUALIFIED"


class MatrixResponseNumericalQualificationPhaseLabel(StrEnum):
    DISORDERED = "DISORDERED"
    GEOMETRIC_PRODUCT = "GEOMETRIC_PRODUCT"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class MatrixResponseNumericalQualificationCheck(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-numerical-qualification-check'

    check_id: str
    measured_value: Decimal
    threshold_value: Decimal
    comparison: str
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.check_id, field_name="check_id")
        validate_decimal(self.measured_value, field_name="measured_value")
        validate_decimal(self.threshold_value, field_name="threshold_value")
        if self.comparison not in {"LE", "GE", "EQ"}:
            raise ValueError("matrix response numerical qualification check comparison is outside LE/GE/EQ")
        expected = {
            "LE": self.measured_value <= self.threshold_value,
            "GE": self.measured_value >= self.threshold_value,
            "EQ": self.measured_value == self.threshold_value,
        }[self.comparison]
        if self.passed != expected:
            raise ValueError("matrix response numerical qualification check pass differs from its comparison")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.passed == bool(self.reason_codes):
            raise ValueError("matrix response numerical qualification check reasons differ from pass status")


@dataclass(frozen=True, slots=True)
class MatrixResponseNumericalQualificationCanonicalPoint(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-numerical-qualification-canonical-point'

    point_id: str
    canonical_model: ObjectIdentity
    terminal: SixMatrixResponseEpisodeTerminal
    phase_label: MatrixResponseNumericalQualificationPhaseLabel
    expected_phase_labels: tuple[MatrixResponseNumericalQualificationPhaseLabel, ...]
    phi_x: Decimal | None
    phi_y: Decimal | None
    action_density: Decimal
    hermiticity_residual: Decimal
    passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.point_id, field_name="point_id")
        if not self.expected_phase_labels or len(set(self.expected_phase_labels)) != len(
            self.expected_phase_labels
        ):
            raise ValueError("matrix response numerical qualification expected phase labels must be nonempty and unique")
        for name in ("phi_x", "phi_y"):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name, minimum=Decimal(0))
        validate_decimal(self.action_density, field_name="action_density")
        validate_decimal(
            self.hermiticity_residual,
            field_name="hermiticity_residual",
            minimum=Decimal(0),
        )
        expected = (
            self.terminal is SixMatrixResponseEpisodeTerminal.COMPLETED
            and self.phase_label in self.expected_phase_labels
        )
        if self.passed != expected:
            raise ValueError("matrix response numerical qualification canonical point pass differs from its terminal/label")


@dataclass(frozen=True, slots=True)
class MatrixResponseNumericalQualificationBenchmark(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-numerical-qualification-benchmark'

    benchmark_id: str
    q: int
    integration_steps: int
    elapsed_seconds: Decimal
    steps_per_second: Decimal
    serialized_checkpoint_bytes: int
    spectral_seconds: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.benchmark_id, field_name="benchmark_id")
        if (
            self.q not in {2, 3, 4}
            or min(self.integration_steps, self.serialized_checkpoint_bytes) < 1
        ):
            raise ValueError("matrix response numerical qualification benchmark geometry/bounds differ")
        for name in ("elapsed_seconds", "steps_per_second", "spectral_seconds"):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value == 0:
                raise ValueError("matrix response numerical qualification benchmark values must be positive")


@dataclass(frozen=True, slots=True)
class MatrixResponseNumericalQualificationQualificationReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-numerical-qualification-qualification-report'

    report_id: str
    implementation_commit: str
    source_config: ObjectIdentity
    simulator_capability: ObjectIdentity
    checks: tuple[MatrixResponseNumericalQualificationCheck, ...]
    canonical_points: tuple[MatrixResponseNumericalQualificationCanonicalPoint, ...]
    benchmarks: tuple[MatrixResponseNumericalQualificationBenchmark, ...]
    disposition: MatrixResponseNumericalQualificationDisposition
    reason_codes: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        if len(self.implementation_commit) != 40:
            raise ValueError("matrix response numerical qualification report requires one Git SHA-1 implementation commit")
        try:
            int(self.implementation_commit, 16)
        except ValueError as error:
            raise ValueError("matrix response numerical qualification implementation commit is not hexadecimal") from error
        require_sorted_unique_ids(self.checks, attribute="check_id", field_name="checks")
        require_sorted_unique_ids(
            self.canonical_points,
            attribute="point_id",
            field_name="canonical_points",
        )
        require_sorted_unique_ids(
            self.benchmarks,
            attribute="benchmark_id",
            field_name="benchmarks",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        all_pass = all(value.passed for value in self.checks) and all(
            value.passed for value in self.canonical_points
        )
        expected = (
            MatrixResponseNumericalQualificationDisposition.NUMERICALLY_QUALIFIED
            if all_pass
            else MatrixResponseNumericalQualificationDisposition.NUMERICAL_MODEL_UNQUALIFIED
        )
        if self.disposition is not expected:
            raise ValueError("matrix response numerical qualification disposition differs from the noncompensating intersection")
        if all_pass == bool(self.reason_codes):
            raise ValueError("matrix response numerical qualification report reasons differ from its intersection")
        if (
            self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
            or self.visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY
            or self.grants_authority
        ):
            raise ValueError("matrix response numerical qualification is development-visible nonpromotable numerical evidence")


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("matrix response numerical qualification metric is nonfinite")
    return Decimal(repr(float(value)))


def _member(source: SixMatrixResponseSixMatrixSourceConfig, *, mass: float) -> SixMatrixResponseModelFamilyMember:
    return next(
        value
        for value in source.anisotropic_model.family_members
        if float(value.mass_x) == mass and float(value.cross_coupling_gamma) == 1.0
    )


from .scientific_seed_inputs import canonical_model_scientific_ordinal, qualification_episode_scientific_seed_sha256


def _safe_suffix(value: float) -> str:
    return str(value).replace(".", "p")


def _episode_request(
    source: SixMatrixResponseSixMatrixSourceConfig,
    *,
    suffix: str,
    member: SixMatrixResponseModelFamilyMember,
    view: SixMatrixResponseNumericalView,
    q: int,
    alpha: float,
    constitution: SixMatrixResponseConstitution,
    seed_index: int,
    steps: int,
    cadence: int,
    scientific_seed_sha256: str,
) -> SixMatrixResponseEpisodeRequest:
    validate_sha256(scientific_seed_sha256, field_name="scientific_seed_sha256")
    coupling = SixMatrixResponseScaledCouplings(Decimal(str(alpha)), Decimal(str(alpha)))
    return SixMatrixResponseEpisodeRequest(
        request_id=f"request.matrix-response-numerical-qualification.{suffix}",
        task_id=f"task.matrix-response-numerical-qualification.{suffix}",
        output_id=f"output.matrix-response-numerical-qualification.{suffix}",
        source_config=ObjectIdentity.from_record(source.config_id, source),
        family_member=member,
        numerical_view=view,
        q=q,
        seed_root_id="seed.matrix-response-numerical-qualification",
        rng_purpose_id=f"purpose.matrix-response-numerical-qualification.{suffix}",
        rng_stream_index=seed_index,
        scientific_seed_sha256=scientific_seed_sha256,
        initial_constitution=constitution,
        action=SixMatrixResponseNativeActionRequest(
            action_id=f"action.matrix-response-numerical-qualification.{suffix}",
            requested_start=coupling,
            requested_target=coupling,
            ramp_steps=steps,
            allow_clipping=False,
        ),
        integration_steps=steps,
        receiver_cadence_steps=cadence,
        evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        grants_authority=False,
    )


def _phi(*, radius: Decimal, q: int, alpha_tilde: float) -> float:
    c2 = (q**2 - 1.0) / 4.0
    return sqrt(float(radius) / ((alpha_tilde / q) ** 2 * c2))


def _phase_label(
    *,
    q: int,
    alpha: float,
    radius_x: Decimal,
    radius_y: Decimal,
    source: SixMatrixResponseSixMatrixSourceConfig,
) -> tuple[MatrixResponseNumericalQualificationPhaseLabel, Decimal | None, Decimal | None]:
    if alpha == 0:
        return MatrixResponseNumericalQualificationPhaseLabel.DISORDERED, None, None
    phi_x = _phi(radius=radius_x, q=q, alpha_tilde=alpha)
    phi_y = _phi(radius=radius_y, q=q, alpha_tilde=alpha)
    lower = float(source.numerical_thresholds.radius_phi_min)
    upper = float(source.numerical_thresholds.radius_phi_max)
    label = (
        MatrixResponseNumericalQualificationPhaseLabel.GEOMETRIC_PRODUCT
        if lower <= phi_x <= upper and lower <= phi_y <= upper
        else MatrixResponseNumericalQualificationPhaseLabel.AMBIGUOUS
    )
    return label, _decimal(phi_x), _decimal(phi_y)


def _check(
    check_id: str,
    measured: float,
    threshold: float,
    comparison: str,
    reason: str,
) -> MatrixResponseNumericalQualificationCheck:
    passed = {
        "LE": measured <= threshold,
        "GE": measured >= threshold,
        "EQ": measured == threshold,
    }[comparison]
    return MatrixResponseNumericalQualificationCheck(
        check_id=check_id,
        measured_value=_decimal(measured),
        threshold_value=_decimal(threshold),
        comparison=comparison,
        passed=passed,
        reason_codes=() if passed else (reason,),
    )


def _gradient_error() -> tuple[float, float]:
    rng = np.random.default_rng(20260829)
    q = 2
    shape = (2, 3, q**2, q**2)
    positions = 0.2 * hermitian_part(rng.normal(size=shape) + 1.0j * rng.normal(size=shape))
    direction = hermitian_part(rng.normal(size=shape) + 1.0j * rng.normal(size=shape))
    parameters = SixMatrixParameters(q, 1.0, 1.0, 1.0, 2.1, 3.2)
    reference = reference_action_terms(positions, parameters)
    optimized = optimized_action_terms(positions, parameters)
    action_error = max(
        abs(getattr(reference, name) - getattr(optimized, name))
        for name in reference.__dataclass_fields__
    )
    gradient = analytic_gradient_terms(positions, parameters)
    epsilon = 1e-6
    plus = optimized_action_terms(positions + epsilon * direction, parameters)
    minus = optimized_action_terms(positions - epsilon * direction, parameters)
    errors = []
    for name in reference.__dataclass_fields__:
        numerical = (getattr(plus, name) - getattr(minus, name)) / (2 * epsilon)
        analytic = float(np.vdot(direction, getattr(gradient, name)).real)
        errors.append(abs(numerical - analytic) / max(abs(numerical), abs(analytic), 1.0))
    return action_error, max(errors)


def _stationarity_and_spectra() -> tuple[float, float, float]:
    stationarity = 0.0
    spectrum_error = 0.0
    construction_error = 0.0
    for q in (2, 3, 4):
        positions = ideal_positions(
            q=q,
            alpha_tilde_x=4.2,
            alpha_tilde_y=3.7,
            constitution="11",
        )
        parameters = SixMatrixParameters(q, 1.0, 1.0, 1.0, 4.2, 3.7)
        stationarity = max(
            stationarity,
            float(np.linalg.norm(analytic_gradient_terms(positions, parameters).total)),
        )
        x_explicit = adjoint_laplacian_explicit(positions[0])
        y_explicit = adjoint_laplacian_explicit(positions[1])
        x_basis = adjoint_laplacian_basis(positions[0])
        y_basis = adjoint_laplacian_basis(positions[1])
        construction_error = max(
            construction_error,
            float(np.max(np.abs(x_explicit - x_basis))),
            float(np.max(np.abs(y_explicit - y_basis))),
        )
        spectrum_error = max(
            spectrum_error,
            float(
                np.max(
                    np.abs(
                        laplacian_eigenvalues(x_basis)
                        - ideal_laplacian_eigenvalues(q=q, alpha_tilde=4.2, active=True)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        laplacian_eigenvalues(y_basis)
                        - ideal_laplacian_eigenvalues(q=q, alpha_tilde=3.7, active=True)
                    )
                )
            ),
            float(
                np.max(
                    np.abs(
                        laplacian_eigenvalues(x_basis + y_basis)
                        - ideal_joint_laplacian_eigenvalues(
                            q=q,
                            alpha_tilde_x=4.2,
                            alpha_tilde_y=3.7,
                            constitution="11",
                        )
                    )
                )
            ),
        )
    return stationarity, spectrum_error, construction_error


def _noise_covariance_error() -> float:
    rng = np.random.Generator(np.random.PCG64DXSM(99173))
    diagonal: list[float] = []
    off_real: list[float] = []
    off_imag: list[float] = []
    for _ in range(1000):
        noise = hermitian_noise(rng=rng, q=2)
        diagonal.extend(noise[..., np.arange(4), np.arange(4)].real.ravel())
        off_real.extend((sqrt(2.0) * noise[..., 0, 1].real).ravel())
        off_imag.extend((sqrt(2.0) * noise[..., 0, 1].imag).ravel())
    return max(
        abs(float(np.var(values, ddof=1)) - 1.0) for values in (diagonal, off_real, off_imag)
    )


def _metropolis_phi(*, draws: int = 5000) -> tuple[float, float]:
    q = 2
    alpha = 6.0
    parameters = SixMatrixParameters(q, 1.0, 1.0, 1.0, alpha, alpha)
    positions = ideal_positions(
        q=q,
        alpha_tilde_x=alpha,
        alpha_tilde_y=alpha,
        constitution="11",
    )
    rng = np.random.default_rng(61423)
    action = optimized_action_terms(positions, parameters).total
    accepted = 0
    values = []
    for index in range(draws):
        proposal = positions + 0.015 * hermitian_part(
            rng.normal(size=positions.shape) + 1.0j * rng.normal(size=positions.shape)
        )
        proposal_action = optimized_action_terms(proposal, parameters).total
        if np.log(rng.random()) < -(proposal_action - action):
            positions = proposal
            action = proposal_action
            accepted += 1
        if index >= draws // 5 and index % 10 == 0:
            radius = float(
                np.trace(np.sum(positions[0] @ positions[0], axis=0)).real / parameters.n
            )
            values.append(sqrt(radius / ((alpha / q) ** 2 * parameters.c2)))
    return float(np.mean(values)), accepted / draws


def _invariance_error(source: SixMatrixResponseSixMatrixSourceConfig) -> float:
    q = 2
    member = _member(source, mass=1.0)
    positions = ideal_positions(
        q=q,
        alpha_tilde_x=4.2,
        alpha_tilde_y=3.7,
        constitution="11",
    )
    rng = np.random.default_rng(483)
    random = hermitian_part(rng.normal(size=(q**2, q**2)) + 1.0j * rng.normal(size=(q**2, q**2)))
    _, unitary = np.linalg.eigh(random)
    state = SixMatrixState(q, positions, np.zeros_like(positions), 0, 4.2, 3.7)
    transformed = conjugate_state(state, unitary)
    first = fast_receiver(state=state, member=member)
    second = fast_receiver(state=transformed, member=member)
    scalar_names = (
        "action_density",
        "radius_x",
        "radius_y",
        "closure_residual_x",
        "closure_residual_y",
        "cross_commutator_norm",
        "kinetic_density",
        "coupling_derivative_x",
        "coupling_derivative_y",
        "hermiticity_residual",
    )
    errors = [
        abs(float(getattr(first, name)) - float(getattr(second, name)))
        / max(abs(float(getattr(first, name))), abs(float(getattr(second, name))), 1.0)
        for name in scalar_names
    ]
    first_spectra = spectral_receiver(
        receiver_prefix="spectrum.numerical-qualification-first", q=q, positions=positions
    )
    second_spectra = spectral_receiver(
        receiver_prefix="spectrum.numerical-qualification-second", q=q, positions=transformed.positions
    )
    for left, right in zip(first_spectra, second_spectra, strict=True):
        errors.append(
            max(
                abs(float(a) - float(b))
                for a, b in zip(left.eigenvalues, right.eigenvalues, strict=True)
            )
            / max(float(left.eigenvalues[-1]), float(right.eigenvalues[-1]), 1.0)
        )
    return max(errors)


def run_matrix_response_study_numerical_qualification_qualification(
    *,
    source: SixMatrixResponseSixMatrixSourceConfig,
    implementation_commit: str,
    canonical_steps: int = 4096,
    chain_steps: int = 8192,
    view_primary_steps: int = 2048,
    benchmark_steps: int = 512,
) -> MatrixResponseNumericalQualificationQualificationReport:
    """Execute the frozen numerical qualification intersection; smaller values are test injection only."""

    validate_sha256(source.fingerprint(), field_name="source_config_fingerprint")
    engine = SixMatrixResponseSixMatrixEpisodeEngine(source)
    checks: list[MatrixResponseNumericalQualificationCheck] = []
    action_error, gradient_error = _gradient_error()
    stationarity, spectrum_error, construction_error = _stationarity_and_spectra()
    thresholds = source.numerical_thresholds
    checks.extend(
        (
            _check(
                "matrix-response-numerical-qualification.action-reference-optimized",
                action_error,
                float(thresholds.ideal_spectrum_absolute_error_max),
                "LE",
                "action-reference-drift",
            ),
            _check(
                "matrix-response-numerical-qualification.gradient-finite-difference",
                gradient_error,
                float(thresholds.gradient_relative_error_max),
                "LE",
                "gradient-relative-error",
            ),
            _check(
                "matrix-response-numerical-qualification.hermitian-noise-covariance",
                _noise_covariance_error(),
                0.08,
                "LE",
                "noise-covariance-error",
            ),
            _check(
                "matrix-response-numerical-qualification.ideal-background-stationarity",
                stationarity,
                1e-10,
                "LE",
                "ideal-background-not-stationary",
            ),
            _check(
                "matrix-response-numerical-qualification.ideal-spectrum",
                spectrum_error,
                float(thresholds.ideal_spectrum_absolute_error_max),
                "LE",
                "ideal-spectrum-error",
            ),
            _check(
                "matrix-response-numerical-qualification.laplacian-construction-crosscheck",
                construction_error,
                float(thresholds.ideal_spectrum_absolute_error_max),
                "LE",
                "laplacian-construction-drift",
            ),
            _check(
                "matrix-response-numerical-qualification.receiver-hidden-transform-invariance",
                _invariance_error(source),
                float(thresholds.invariance_relative_error_max),
                "LE",
                "receiver-invariance-error",
            ),
        )
    )

    points = []
    for model in source.canonical_reference_models:
        alpha = float(model.scaled_coupling_alpha_tilde)
        suffix = model.model_id.removeprefix("six-matrix-response.canonical.")
        request = _episode_request(
            source,
            suffix=f"canonical.{suffix}",
            scientific_seed_sha256=qualification_episode_scientific_seed_sha256(
                role_ordinal=0, case_ordinal=canonical_model_scientific_ordinal(
                    mass_m=model.mass_m, q=model.representation_dimension_q, alpha_tilde=model.scaled_coupling_alpha_tilde
                ), view_ordinal=0, seed_index=0,
            ),
            member=_member(source, mass=float(model.mass_m)),
            view=source.primary_view,
            q=model.representation_dimension_q,
            alpha=alpha,
            constitution=(SixMatrixResponseConstitution.EMPTY if alpha == 0 else SixMatrixResponseConstitution.PRODUCT),
            seed_index=0,
            steps=canonical_steps,
            cadence=max(1, canonical_steps // 4),
        )
        episode = engine.run(request)
        receiver = episode.receivers[-1]
        label, phi_x, phi_y = _phase_label(
            q=model.representation_dimension_q,
            alpha=alpha,
            radius_x=receiver.radius_x,
            radius_y=receiver.radius_y,
            source=source,
        )
        expected = (
            (MatrixResponseNumericalQualificationPhaseLabel.DISORDERED,)
            if alpha == 0
            else (
                (
                    MatrixResponseNumericalQualificationPhaseLabel.AMBIGUOUS,
                    MatrixResponseNumericalQualificationPhaseLabel.GEOMETRIC_PRODUCT,
                )
                if alpha == 4.2
                else (MatrixResponseNumericalQualificationPhaseLabel.GEOMETRIC_PRODUCT,)
            )
        )
        points.append(
            MatrixResponseNumericalQualificationCanonicalPoint(
                point_id=f"matrix-response-numerical-qualification.point.{suffix}",
                canonical_model=ObjectIdentity.from_record(model.model_id, model),
                terminal=episode.terminal,
                phase_label=label,
                expected_phase_labels=expected,
                phi_x=phi_x,
                phi_y=phi_y,
                action_density=receiver.action_density,
                hermiticity_residual=receiver.hermiticity_residual,
                passed=episode.terminal is SixMatrixResponseEpisodeTerminal.COMPLETED and label in expected,
            )
        )
    points.sort(key=lambda value: value.point_id)

    action_by_family: dict[tuple[int, Decimal], list[tuple[Decimal, Decimal]]] = {}
    model_by_fingerprint = {
        value.fingerprint(): value for value in source.canonical_reference_models
    }
    for point in points:
        model = model_by_fingerprint[point.canonical_model.object_fingerprint]
        action_by_family.setdefault((model.representation_dimension_q, model.mass_m), []).append(
            (model.scaled_coupling_alpha_tilde, point.action_density)
        )
    ordering_margins: list[float] = []
    for values in action_by_family.values():
        ordered = sorted(values)
        ordering_margins.extend(
            float(left_action - right_action)
            for (_, left_action), (_, right_action) in zip(ordered, ordered[1:])
        )
    checks.append(
        _check(
            "matrix-response-numerical-qualification.canonical-action-ordering",
            min(ordering_margins),
            0.0,
            "GE",
            "canonical-action-ordering-failed",
        )
    )

    member = _member(source, mass=1.0)
    history_differences = []
    for alpha in (4.2, 8.0):
        history_phi = []
        for constitution in (SixMatrixResponseConstitution.EMPTY, SixMatrixResponseConstitution.PRODUCT):
            episode = engine.run(
                _episode_request(
                    source,
                    suffix=f"history.dimension-two-a{_safe_suffix(alpha)}.{constitution.value}",
                    scientific_seed_sha256=qualification_episode_scientific_seed_sha256(
                        role_ordinal=1, case_ordinal=(4.2, 8.0).index(alpha) * 2 + int(constitution is SixMatrixResponseConstitution.PRODUCT), view_ordinal=0, seed_index=31,
                    ),
                    member=member,
                    view=source.primary_view,
                    q=2,
                    alpha=alpha,
                    constitution=constitution,
                    seed_index=31,
                    steps=chain_steps,
                    cadence=max(1, chain_steps // 16),
                )
            )
            history_phi.append(
                _phi(
                    radius=episode.receivers[-1].radius_x,
                    q=2,
                    alpha_tilde=alpha,
                )
            )
        history_differences.append(abs(history_phi[0] - history_phi[1]))
    checks.append(
        _check(
            "matrix-response-numerical-qualification.canonical-history-convergence",
            max(history_differences),
            0.10,
            "LE",
            "canonical-history-convergence-failed",
        )
    )
    chain_means = []
    maximum_hermiticity = 0.0
    for seed in range(4):
        episode = engine.run(
            _episode_request(
                source,
                suffix=f"chain.dimension-two-a6.s{seed}",
                scientific_seed_sha256=qualification_episode_scientific_seed_sha256(
                    role_ordinal=2, case_ordinal=0, view_ordinal=0, seed_index=seed,
                ),
                member=member,
                view=source.primary_view,
                q=2,
                alpha=6.0,
                constitution=SixMatrixResponseConstitution.PRODUCT,
                seed_index=seed,
                steps=chain_steps,
                cadence=max(1, chain_steps // 16),
            )
        )
        retained = episode.receivers[len(episode.receivers) // 2 :]
        chain_means.append(
            float(
                np.mean([_phi(radius=value.radius_x, q=2, alpha_tilde=6.0) for value in retained])
            )
        )
        maximum_hermiticity = max(
            maximum_hermiticity,
            max(float(value.hermiticity_residual) for value in episode.receivers),
        )
    chain_spread = max(chain_means) - min(chain_means)
    checks.extend(
        (
            _check(
                "matrix-response-numerical-qualification.chain-recurrence",
                chain_spread,
                0.10,
                "LE",
                "independent-chain-disagreement",
            ),
            _check(
                "matrix-response-numerical-qualification.integrator-hermiticity",
                maximum_hermiticity,
                float(thresholds.hermiticity_residual_max),
                "LE",
                "integrator-hermiticity-error",
            ),
        )
    )

    view_differences = []
    view_concordance = []
    for alpha in (4.2, 6.0, 8.0):
        results = []
        for view, steps, view_name in (
            (source.primary_view, view_primary_steps, "primary"),
            (source.secondary_view, view_primary_steps * 2, "secondary"),
        ):
            episode = engine.run(
                _episode_request(
                    source,
                    suffix=f"view.dimension-two-a{_safe_suffix(alpha)}.{view_name}",
                    scientific_seed_sha256=qualification_episode_scientific_seed_sha256(
                        role_ordinal=3, case_ordinal=(4.2, 6.0, 8.0).index(alpha), view_ordinal=0 if view is source.primary_view else 1, seed_index=17,
                    ),
                    member=member,
                    view=view,
                    q=2,
                    alpha=alpha,
                    constitution=SixMatrixResponseConstitution.PRODUCT,
                    seed_index=17,
                    steps=steps,
                    cadence=max(1, steps // 8),
                )
            )
            receiver = episode.receivers[-1]
            phase, phi_x, _ = _phase_label(
                q=2,
                alpha=alpha,
                radius_x=receiver.radius_x,
                radius_y=receiver.radius_y,
                source=source,
            )
            assert phi_x is not None
            results.append((phase, float(phi_x)))
        view_concordance.append(float(results[0][0] is results[1][0]))
        view_differences.append(abs(results[0][1] - results[1][1]))
    checks.extend(
        (
            _check(
                "matrix-response-numerical-qualification.numerical-view-phase-concordance",
                min(view_concordance),
                float(thresholds.numerical_view_phase_concordance_min),
                "GE",
                "numerical-view-phase-disagreement",
            ),
            _check(
                "matrix-response-numerical-qualification.numerical-view-radius-difference",
                max(view_differences),
                0.10,
                "LE",
                "numerical-view-radius-drift",
            ),
        )
    )

    metropolis_mean, acceptance = _metropolis_phi()
    langevin_mean = float(np.mean(chain_means))
    checks.extend(
        (
            _check(
                "matrix-response-numerical-qualification.metropolis-acceptance-lower",
                acceptance,
                0.20,
                "GE",
                "metropolis-acceptance-too-low",
            ),
            _check(
                "matrix-response-numerical-qualification.metropolis-acceptance-upper",
                acceptance,
                0.80,
                "LE",
                "metropolis-acceptance-too-high",
            ),
            _check(
                "matrix-response-numerical-qualification.metropolis-langevin-radius",
                abs(metropolis_mean - langevin_mean),
                0.10,
                "LE",
                "metropolis-langevin-disagreement",
            ),
        )
    )

    benchmarks = []
    for q in (2, 3, 4):
        request = _episode_request(
            source,
            suffix=f"benchmark.q{q}",
            scientific_seed_sha256=qualification_episode_scientific_seed_sha256(
                role_ordinal=4, case_ordinal=q - 2, view_ordinal=0, seed_index=q,
            ),
            member=member,
            view=source.primary_view,
            q=q,
            alpha=6.0,
            constitution=SixMatrixResponseConstitution.PRODUCT,
            seed_index=q,
            steps=benchmark_steps,
            cadence=benchmark_steps,
        )
        started = perf_counter()
        episode = engine.run(request)
        elapsed = perf_counter() - started
        assert episode.checkpoint is not None
        state, _ = state_from_checkpoint(episode.checkpoint)
        spectral_started = perf_counter()
        spectral_receiver(
            receiver_prefix=f"spectrum.matrix-response-numerical-qualification-benchmark-q{q}",
            q=q,
            positions=state.positions,
        )
        spectral_elapsed = perf_counter() - spectral_started
        benchmarks.append(
            MatrixResponseNumericalQualificationBenchmark(
                benchmark_id=f"matrix-response-numerical-qualification.benchmark.q{q}",
                q=q,
                integration_steps=benchmark_steps,
                elapsed_seconds=_decimal(elapsed),
                steps_per_second=_decimal(benchmark_steps / elapsed),
                serialized_checkpoint_bytes=len(episode.checkpoint.canonical_bytes()),
                spectral_seconds=_decimal(spectral_elapsed),
            )
        )
    projected_anisotropic_feasibility_seconds = 15_000_000 / (
        min(float(value.steps_per_second) for value in benchmarks) * source.maximum_worker_processes
    )
    checks.append(
        _check(
            "matrix-response-numerical-qualification.anisotropic-feasibility-projected-wall-seconds",
            projected_anisotropic_feasibility_seconds,
            24 * 60 * 60,
            "LE",
            "anisotropic-feasibility-compute-model-exceeds-wall-bound",
        )
    )

    checks.sort(key=lambda value: value.check_id)
    failed_reasons = {
        reason for value in checks if not value.passed for reason in value.reason_codes
    }
    if any(not value.passed for value in points):
        failed_reasons.add("canonical-phase-ordering-failed")
    failed = tuple(sorted(failed_reasons))
    return MatrixResponseNumericalQualificationQualificationReport(
        report_id="matrix-response-study.numerical-qualification-report",
        implementation_commit=implementation_commit,
        source_config=ObjectIdentity.from_record(source.config_id, source),
        simulator_capability=ObjectIdentity.from_record(
            SIX_MATRIX_RESPONSE_CAPABILITY.capability_key,
            SIX_MATRIX_RESPONSE_CAPABILITY,
        ),
        checks=tuple(checks),
        canonical_points=tuple(points),
        benchmarks=tuple(benchmarks),
        disposition=(
            MatrixResponseNumericalQualificationDisposition.NUMERICALLY_QUALIFIED
            if not failed
            else MatrixResponseNumericalQualificationDisposition.NUMERICAL_MODEL_UNQUALIFIED
        ),
        reason_codes=failed,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
        grants_authority=False,
    )


__all__ = [
    'MatrixResponseNumericalQualificationBenchmark',
    'MatrixResponseNumericalQualificationCanonicalPoint',
    'MatrixResponseNumericalQualificationCheck',
    'MatrixResponseNumericalQualificationDisposition',
    'MatrixResponseNumericalQualificationPhaseLabel',
    'MatrixResponseNumericalQualificationQualificationReport',
    'run_matrix_response_study_numerical_qualification_qualification',
]
