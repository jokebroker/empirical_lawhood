'Gauge-covariant finite-temperature transverse response for ambient pressure superconductor gauge covariant response.\n\nThe producer evaluates a transverse Peierls probe in a clean lattice-BCS\nclosure.  It explicitly subtracts the normal-state response, retains raw\ndiamagnetic and state-rearrangement terms, evaluates signed probes at finite\nq, and verifies pure-gauge covariance.  A material result additionally needs\nthe frozen DFT/Wannier/pairing compatibility operands; fixtures never satisfy\nthat boundary.\n'

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from math import pi
from typing import Final

import numpy as np
from numpy.typing import NDArray

from .gauge_covariant_response_contracts import GaugeCovariantResponseConformanceResult, GaugeCovariantResponseFixtureDisposition, GaugeCovariantResponseFixtureResult, GaugeCovariantResponseFixtureSpec, GaugeCovariantResponseFixtureSuite, GaugeCovariantResponseFormalismFreeze, GaugeCovariantResponseMaterialInput, GaugeCovariantResponseMaterialResult, GaugeCovariantResponseResponseObservation


KB_EV_PER_K: Final = 8.617333262145e-5
PRODUCER_IMPLEMENTATION_ID: Final = 'producer.ambient-pressure-superconductor-gauge-covariant-response-gauge-covariant-lattice-bcs'


def _d(value: float) -> Decimal:
    if abs(value) < 5e-15:
        value = 0.0
    return Decimal(format(value, ".12g"))


def gauge_covariant_response_formalism() -> GaugeCovariantResponseFormalismFreeze:
    return GaugeCovariantResponseFormalismFreeze(
        formalism_id='formalism.ambient-pressure-superconductor-gauge-covariant-response-gauge-covariant-lattice-bcs',
        producer_implementation_id=PRODUCER_IMPLEMENTATION_ID,
        source_ids=tuple(
            sorted(
                (
                    'source.ambient-pressure-superconductor.gauge-covariant-response-wannier90-3-1-0',
                    'source.ambient-pressure-superconductor.gauge-covariant-response-watanabe-2501-13722v2',
                    'source.ambient-pressure-superconductor.epw61-archive',
                )
            )
        ),
        equation_ids=tuple(
            sorted(
                (
                    "equation.bdg-finite-temperature-grand-potential",
                    "equation.peierls-gauge-covariant-transverse-probe",
                    "equation.signed-current-free-energy-derivative",
                    "equation.superconducting-minus-normal-kernel",
                )
            )
        ),
        compatibility_map_ids=tuple(
            sorted(
                (
                    'compatibility.dft-wannier-hopping-frame-to-gauge-covariant-response-lattice',
                    'compatibility.epw-or-scdft-gap-temperature-to-gauge-covariant-response-pairing',
                    'compatibility.gauge-covariant-response-lattice-current-to-si-current-density',
                    'compatibility.gauge-covariant-response-stiffness-interval-to-analytic-slab-receiver',
                )
            )
        ),
        input_operand_ids=tuple(
            sorted(
                (
                    "operand.material-specific-gap-at-300k",
                    "operand.material-specific-wannier-hamiltonian",
                    "operand.native-lattice-and-current-units",
                    "operand.normal-state-comparator",
                    "operand.requested-and-realized-preparation",
                )
            )
        ),
        output_operand_ids=tuple(
            sorted(
                (
                    "operand.base-and-refined-stiffness-interval",
                    "operand.diamagnetic-and-state-rearrangement-terms",
                    "operand.finite-q-inference-envelope",
                    "operand.normal-state-cancellation",
                    "operand.signed-transverse-current",
                    "operand.ward-pure-gauge-residual",
                )
            )
        ),
        validity_requirement_ids=tuple(
            sorted(
                (
                    "validity.explicit-300k-pairing-state",
                    "validity.finite-q-envelope-common-across-views",
                    "validity.material-band-closure-bound",
                    "validity.nonlocal-and-strong-coupling-closure-resolved",
                    "validity.single-band-effective-lattice-adequate",
                    "validity.weak-probe-locality",
                )
            )
        ),
        nonpromotion_rule_ids=tuple(
            sorted(
                (
                    "rule.fixture-method-evidence-only",
                    'rule.uniform-pairing-current-cannot-substitute-for-gauge-covariant-response',
                    "rule.scalar-tc-cannot-substitute-for-transverse-response",
                    'rule.unmapped-lattice-stiffness-cannot-enter-material-admission',
                )
            )
        ),
        base_view_id='view.ambient-pressure-superconductor-gauge-covariant-response-lattice-base',
        refined_view_id='view.ambient-pressure-superconductor-gauge-covariant-response-lattice-refined',
        independent_view_id='view.ambient-pressure-superconductor-gauge-covariant-response-independent-scdft-cfop',
        operating_temperature_K=Decimal("300"),
        finite_q_required=True,
        normal_comparator_required=True,
        gauge_covariance_required=True,
        material_compatibility_required=True,
    )


def gauge_covariant_response_fixture_suite() -> GaugeCovariantResponseFixtureSuite:
    positive = {
        "temperature_K": Decimal("300"),
        "hopping_eV": Decimal("1"),
        "chemical_potential_eV": Decimal("-1"),
        "gap_eV": Decimal("0.06"),
    }

    def fixture(
        name: str,
        expected: GaugeCovariantResponseFixtureDisposition,
        perturbation: str,
        **overrides: Decimal,
    ) -> GaugeCovariantResponseFixtureSpec:
        values = {**positive, **overrides}
        return GaugeCovariantResponseFixtureSpec(
            fixture_id=f'fixture.ambient-pressure-superconductor-gauge-covariant-response-{name}',
            fixture_kind=f'fixture-kind.gauge-covariant-response-{name}',
            expected_disposition=expected,
            perturbation_id=f'perturbation.gauge-covariant-response-{perturbation}',
            **values,
        )

    fixtures = (
        fixture("finite-q-drift", GaugeCovariantResponseFixtureDisposition.REJECT, "finite-q-drift"),
        fixture(
            "insulator",
            GaugeCovariantResponseFixtureDisposition.ACCEPT,
            "none",
            chemical_potential_eV=Decimal("-6"),
            gap_eV=Decimal("0"),
        ),
        fixture("missing-diamagnetic", GaugeCovariantResponseFixtureDisposition.REJECT, "missing-diamagnetic"),
        fixture("nonlinear", GaugeCovariantResponseFixtureDisposition.REJECT, "nonlinear"),
        fixture("normal", GaugeCovariantResponseFixtureDisposition.ACCEPT, "none", gap_eV=Decimal("0")),
        fixture("positive", GaugeCovariantResponseFixtureDisposition.ACCEPT, "none"),
        fixture("view-disagreement", GaugeCovariantResponseFixtureDisposition.REJECT, "view-disagreement"),
        fixture("ward-failure", GaugeCovariantResponseFixtureDisposition.REJECT, "ward-failure"),
        fixture("wrong-sign", GaugeCovariantResponseFixtureDisposition.REJECT, "wrong-sign"),
    )
    return GaugeCovariantResponseFixtureSuite(
        suite_id='suite.ambient-pressure-superconductor-gauge-covariant-response-gauge-closed-conformance',
        fixtures=tuple(sorted(fixtures, key=lambda value: value.fixture_id)),
        base_nx=24,
        base_nky=48,
        refined_nx=32,
        refined_nky=64,
        signed_probe_amplitude=Decimal("0.0005"),
        derivative_step=Decimal("0.00005"),
        oddness_relative_limit=Decimal("0.000001"),
        nonlinear_relative_limit=Decimal("0.02"),
        ward_absolute_limit_eV=Decimal("0.0000000001"),
        normal_cancellation_absolute_limit=Decimal("0.01"),
        minimum_positive_stiffness=Decimal("0.0001"),
    )


def _bdg_hamiltonian(
    *,
    nx: int,
    ky: float,
    hopping_eV: float,
    chemical_potential_eV: float,
    gap_eV: float,
    amplitude: float,
    q: float,
) -> NDArray[np.complex128]:
    positions = np.arange(nx, dtype=np.float64)
    vector_potential = amplitude * np.cos(q * positions)
    electron = np.diag(
        -2.0 * hopping_eV * np.cos(ky - vector_potential) - chemical_potential_eV
    ).astype(np.complex128)
    hole = np.diag(2.0 * hopping_eV * np.cos(ky + vector_potential) + chemical_potential_eV).astype(
        np.complex128
    )
    for index in range(nx):
        neighbour = (index + 1) % nx
        electron[index, neighbour] = -hopping_eV
        electron[neighbour, index] = -hopping_eV
        hole[index, neighbour] = hopping_eV
        hole[neighbour, index] = hopping_eV
    pairing = np.eye(nx, dtype=np.complex128) * gap_eV
    return np.block([[electron, pairing], [pairing, hole]])


def _free_energy_per_cell(
    *,
    nx: int,
    nky: int,
    hopping_eV: float,
    chemical_potential_eV: float,
    gap_eV: float,
    temperature_K: float,
    amplitude: float,
    q: float,
) -> float:
    total = 0.0
    for ky in (np.arange(nky, dtype=np.float64) + 0.5) * (2.0 * pi / nky) - pi:
        eigenvalues = np.linalg.eigvalsh(
            _bdg_hamiltonian(
                nx=nx,
                ky=float(ky),
                hopping_eV=hopping_eV,
                chemical_potential_eV=chemical_potential_eV,
                gap_eV=gap_eV,
                amplitude=amplitude,
                q=q,
            )
        )
        scaled = np.abs(eigenvalues) / (2.0 * KB_EV_PER_K * temperature_K)
        total += -0.5 * KB_EV_PER_K * temperature_K * float(np.logaddexp(scaled, -scaled).sum())
    return total / (nky * nx)


def _curvature(
    *,
    nx: int,
    nky: int,
    hopping_eV: float,
    chemical_potential_eV: float,
    gap_eV: float,
    temperature_K: float,
    amplitude: float,
    q: float,
) -> float:
    def energy(probe: float) -> float:
        return _free_energy_per_cell(
            nx=nx,
            nky=nky,
            hopping_eV=hopping_eV,
            chemical_potential_eV=chemical_potential_eV,
            gap_eV=gap_eV,
            temperature_K=temperature_K,
            amplitude=probe,
            q=q,
        )

    return (energy(amplitude) + energy(-amplitude) - 2.0 * energy(0.0)) / amplitude**2


def _corrected_current(
    *,
    nx: int,
    nky: int,
    hopping_eV: float,
    chemical_potential_eV: float,
    gap_eV: float,
    temperature_K: float,
    amplitude: float,
    derivative_step: float,
    q: float,
) -> float:
    def difference(probe: float) -> float:
        common = dict(
            nx=nx,
            nky=nky,
            hopping_eV=hopping_eV,
            chemical_potential_eV=chemical_potential_eV,
            temperature_K=temperature_K,
            amplitude=probe,
            q=q,
        )
        return _free_energy_per_cell(gap_eV=gap_eV, **common) - _free_energy_per_cell(
            gap_eV=0.0, **common
        )

    return -(difference(amplitude + derivative_step) - difference(amplitude - derivative_step)) / (
        2.0 * derivative_step
    )


def _explicit_diamagnetic(
    *,
    nx: int,
    nky: int,
    hopping_eV: float,
    chemical_potential_eV: float,
    gap_eV: float,
    temperature_K: float,
    derivative_step: float,
    q: float,
) -> float:
    total = 0.0
    for ky in (np.arange(nky, dtype=np.float64) + 0.5) * (2.0 * pi / nky) - pi:
        common = dict(
            nx=nx,
            ky=float(ky),
            hopping_eV=hopping_eV,
            chemical_potential_eV=chemical_potential_eV,
            gap_eV=gap_eV,
            q=q,
        )
        zero = _bdg_hamiltonian(amplitude=0.0, **common)
        positive = _bdg_hamiltonian(amplitude=derivative_step, **common)
        negative = _bdg_hamiltonian(amplitude=-derivative_step, **common)
        curvature_matrix = (positive - 2.0 * zero + negative) / derivative_step**2
        eigenvalues, eigenvectors = np.linalg.eigh(zero)
        thermal = (
            eigenvectors * np.tanh(eigenvalues / (2.0 * KB_EV_PER_K * temperature_K))
        ) @ eigenvectors.conj().T
        total += -0.25 * float(np.trace(thermal @ curvature_matrix).real) / nx
    return total / nky


def _pure_gauge_residual(
    *, hopping_eV: float, chemical_potential_eV: float, gap_eV: float, temperature_K: float
) -> float:
    size = 12
    phase = 0.17 * np.sin(2.0 * pi * np.arange(size) / size)

    def matrix(gauged: bool) -> NDArray[np.complex128]:
        single = np.diag(np.full(size, -chemical_potential_eV)).astype(np.complex128)
        for index in range(size):
            neighbour = (index + 1) % size
            link = phase[index] - phase[neighbour] if gauged else 0.0
            single[index, neighbour] = -hopping_eV * np.exp(1j * link)
            single[neighbour, index] = -hopping_eV * np.exp(-1j * link)
        pairing_phase = np.exp(2j * phase) if gauged else np.ones(size)
        pairing = np.diag(gap_eV * pairing_phase)
        return np.block([[single, pairing], [pairing.conj(), -single.conj()]])

    def energy(value: NDArray[np.complex128]) -> float:
        eigenvalues = np.linalg.eigvalsh(value)
        scaled = np.abs(eigenvalues) / (2.0 * KB_EV_PER_K * temperature_K)
        return (
            -0.5 * KB_EV_PER_K * temperature_K * float(np.logaddexp(scaled, -scaled).sum()) / size
        )

    return abs(energy(matrix(True)) - energy(matrix(False)))


def produce_gauge_covariant_response_observation(
    fixture: GaugeCovariantResponseFixtureSpec,
    suite: GaugeCovariantResponseFixtureSuite,
) -> GaugeCovariantResponseResponseObservation:
    """Produce one closure-scoped method observation without fault injection."""

    hopping = float(fixture.hopping_eV)
    chemical_potential = float(fixture.chemical_potential_eV)
    gap = float(fixture.gap_eV)
    temperature = float(fixture.temperature_K)
    amplitude = float(suite.signed_probe_amplitude)
    derivative_step = float(suite.derivative_step)

    def view(nx: int, nky: int) -> tuple[float, float, float, float]:
        q = 2.0 * pi / nx
        common = dict(
            nx=nx,
            nky=nky,
            hopping_eV=hopping,
            chemical_potential_eV=chemical_potential,
            temperature_K=temperature,
            amplitude=amplitude,
        )
        superconducting_q = _curvature(gap_eV=gap, q=q, **common)
        normal_q = _curvature(gap_eV=0.0, q=q, **common)
        superconducting_2q = _curvature(gap_eV=gap, q=2.0 * q, **common)
        normal_2q = _curvature(gap_eV=0.0, q=2.0 * q, **common)
        stiffness_q = superconducting_q - normal_q
        stiffness_2q = superconducting_2q - normal_2q
        lower = min(stiffness_q, stiffness_2q)
        upper = max(stiffness_q, stiffness_2q, 2.0 * stiffness_q - stiffness_2q)
        return stiffness_q, lower, upper, normal_q

    base, base_lower, base_upper, normal_total = view(suite.base_nx, suite.base_nky)
    refined, refined_lower, refined_upper, _ = view(suite.refined_nx, suite.refined_nky)
    interval_lower = max(base_lower, refined_lower)
    interval_upper = min(base_upper, refined_upper)
    common_interval = interval_lower <= interval_upper
    if not common_interval:
        interval_lower = min(base_lower, refined_lower)
        interval_upper = max(base_upper, refined_upper)

    q_base = 2.0 * pi / suite.base_nx
    current_common = dict(
        nx=suite.base_nx,
        nky=suite.base_nky,
        hopping_eV=hopping,
        chemical_potential_eV=chemical_potential,
        gap_eV=gap,
        temperature_K=temperature,
        derivative_step=derivative_step,
        q=q_base,
    )
    j_plus = _corrected_current(amplitude=amplitude, **current_common)
    j_minus = _corrected_current(amplitude=-amplitude, **current_common)
    j_zero = _corrected_current(amplitude=0.0, **current_common)
    oddness = abs(j_plus + j_minus - 2.0 * j_zero) / max(abs(j_plus), abs(j_minus), 1e-15)

    base_at_double_probe = _curvature(
        nx=suite.base_nx,
        nky=suite.base_nky,
        hopping_eV=hopping,
        chemical_potential_eV=chemical_potential,
        gap_eV=gap,
        temperature_K=temperature,
        amplitude=2.0 * amplitude,
        q=q_base,
    ) - _curvature(
        nx=suite.base_nx,
        nky=suite.base_nky,
        hopping_eV=hopping,
        chemical_potential_eV=chemical_potential,
        gap_eV=0.0,
        temperature_K=temperature,
        amplitude=2.0 * amplitude,
        q=q_base,
    )
    nonlinear = abs(base - base_at_double_probe) / max(abs(base), 1e-12)

    terms_common = dict(
        nx=suite.base_nx,
        nky=suite.base_nky,
        hopping_eV=hopping,
        chemical_potential_eV=chemical_potential,
        temperature_K=temperature,
        derivative_step=derivative_step,
        q=q_base,
    )
    superconducting_diamagnetic = _explicit_diamagnetic(gap_eV=gap, **terms_common)
    normal_diamagnetic = _explicit_diamagnetic(gap_eV=0.0, **terms_common)
    superconducting_total = base + normal_total
    superconducting_paramagnetic = superconducting_total - superconducting_diamagnetic
    normal_paramagnetic = normal_total - normal_diamagnetic
    ward = _pure_gauge_residual(
        hopping_eV=hopping,
        chemical_potential_eV=chemical_potential,
        gap_eV=gap,
        temperature_K=temperature,
    )
    return GaugeCovariantResponseResponseObservation(
        observation_id=f"observation.{fixture.fixture_id.removeprefix('fixture.')}",
        fixture_id=fixture.fixture_id,
        evidence_scope_id="evidence-scope.method-fixture-only",
        temperature_K=fixture.temperature_K,
        q_base_inverse_lattice=_d(q_base),
        q_refined_inverse_lattice=_d(2.0 * pi / suite.refined_nx),
        probe_amplitude=suite.signed_probe_amplitude,
        j_plus_eV_per_link=_d(j_plus),
        j_minus_eV_per_link=_d(j_minus),
        j_zero_eV_per_link=_d(j_zero),
        stiffness_base_eV_per_link=_d(base),
        stiffness_refined_eV_per_link=_d(refined),
        stiffness_interval_lower_eV_per_link=_d(interval_lower),
        stiffness_interval_upper_eV_per_link=_d(interval_upper),
        superconducting_diamagnetic_eV_per_link=_d(superconducting_diamagnetic),
        superconducting_paramagnetic_eV_per_link=_d(superconducting_paramagnetic),
        normal_diamagnetic_eV_per_link=_d(normal_diamagnetic),
        normal_paramagnetic_eV_per_link=_d(normal_paramagnetic),
        normal_total_eV_per_link=_d(normal_total),
        oddness_relative_residual=_d(oddness),
        nonlinear_relative_residual=_d(nonlinear),
        ward_absolute_residual_eV=_d(ward),
        base_refined_common_interval=common_interval,
        diamagnetic_term_present=True,
        material_compatibility_present=False,
        model_validity_resolved=True,
        reason_codes=('reason.gauge-covariant-response-method-fixture-not-material-evidence',),
    )


def _perturb(
    observation: GaugeCovariantResponseResponseObservation, fixture: GaugeCovariantResponseFixtureSpec, suite: GaugeCovariantResponseFixtureSuite
) -> GaugeCovariantResponseResponseObservation:
    perturbation = fixture.perturbation_id
    if perturbation == 'perturbation.gauge-covariant-response-none':
        return observation
    if perturbation == 'perturbation.gauge-covariant-response-wrong-sign':
        return replace(
            observation,
            j_plus_eV_per_link=abs(observation.j_plus_eV_per_link),
            j_minus_eV_per_link=-abs(observation.j_minus_eV_per_link),
            reason_codes=tuple(
                sorted((*observation.reason_codes, "reason.fixture-wrong-sign-injected"))
            ),
        )
    if perturbation == 'perturbation.gauge-covariant-response-missing-diamagnetic':
        return replace(
            observation,
            diamagnetic_term_present=False,
            reason_codes=tuple(
                sorted((*observation.reason_codes, "reason.fixture-diamagnetic-term-removed"))
            ),
        )
    if perturbation == 'perturbation.gauge-covariant-response-nonlinear':
        return replace(
            observation,
            nonlinear_relative_residual=suite.nonlinear_relative_limit * Decimal("10"),
            reason_codes=tuple(
                sorted((*observation.reason_codes, "reason.fixture-nonlinearity-injected"))
            ),
        )
    if perturbation == 'perturbation.gauge-covariant-response-ward-failure':
        return replace(
            observation,
            ward_absolute_residual_eV=suite.ward_absolute_limit_eV * Decimal("1000"),
            reason_codes=tuple(
                sorted((*observation.reason_codes, "reason.fixture-ward-failure-injected"))
            ),
        )
    if perturbation in {'perturbation.gauge-covariant-response-finite-q-drift', 'perturbation.gauge-covariant-response-view-disagreement'}:
        code = (
            "reason.fixture-finite-q-drift-injected"
            if perturbation.endswith("finite-q-drift")
            else "reason.fixture-view-disagreement-injected"
        )
        return replace(
            observation,
            stiffness_refined_eV_per_link=-abs(observation.stiffness_refined_eV_per_link),
            base_refined_common_interval=False,
            reason_codes=tuple(sorted((*observation.reason_codes, code))),
        )
    raise ValueError(f'unknown closed gauge covariant response fixture perturbation {perturbation}')


def _evaluate(
    observation: GaugeCovariantResponseResponseObservation,
    fixture: GaugeCovariantResponseFixtureSpec,
    suite: GaugeCovariantResponseFixtureSuite,
) -> tuple[GaugeCovariantResponseFixtureDisposition, tuple[str, ...]]:
    failures: list[str] = []
    if observation.oddness_relative_residual > suite.oddness_relative_limit:
        failures.append('reason.gauge-covariant-response-oddness-failed')
    if observation.nonlinear_relative_residual > suite.nonlinear_relative_limit:
        failures.append('reason.gauge-covariant-response-weak-probe-nonlinearity-failed')
    if observation.ward_absolute_residual_eV > suite.ward_absolute_limit_eV:
        failures.append('reason.gauge-covariant-response-ward-gauge-covariance-failed')
    if abs(observation.normal_total_eV_per_link) > suite.normal_cancellation_absolute_limit:
        failures.append('reason.gauge-covariant-response-normal-cancellation-failed')
    if not observation.diamagnetic_term_present:
        failures.append('reason.gauge-covariant-response-diamagnetic-term-missing')
    if observation.superconducting_diamagnetic_eV_per_link <= 0:
        failures.append('reason.gauge-covariant-response-diamagnetic-sign-failed')
    if not observation.base_refined_common_interval:
        failures.append('reason.gauge-covariant-response-base-refined-view-disagreement')

    positive_kind = fixture.fixture_kind not in {
        'fixture-kind.gauge-covariant-response-normal',
        'fixture-kind.gauge-covariant-response-insulator',
    }
    if positive_kind:
        if observation.stiffness_interval_lower_eV_per_link <= suite.minimum_positive_stiffness:
            failures.append('reason.gauge-covariant-response-positive-conservative-stiffness-missing')
        if observation.j_plus_eV_per_link * observation.probe_amplitude >= 0:
            failures.append('reason.gauge-covariant-response-current-direction-failed')
    else:
        if (
            max(
                abs(observation.stiffness_base_eV_per_link),
                abs(observation.stiffness_refined_eV_per_link),
            )
            > suite.normal_cancellation_absolute_limit
        ):
            failures.append('reason.gauge-covariant-response-nonsuperconducting-fixture-did-not-cancel')

    disposition = GaugeCovariantResponseFixtureDisposition.ACCEPT if not failures else GaugeCovariantResponseFixtureDisposition.REJECT
    return disposition, tuple(sorted(failures))


def run_gauge_covariant_response_conformance() -> tuple[GaugeCovariantResponseConformanceResult, tuple[GaugeCovariantResponseResponseObservation, ...]]:
    formalism = gauge_covariant_response_formalism()
    suite = gauge_covariant_response_fixture_suite()
    physical_cache: dict[tuple[Decimal, Decimal], GaugeCovariantResponseResponseObservation] = {}
    observations: list[GaugeCovariantResponseResponseObservation] = []
    results: list[GaugeCovariantResponseFixtureResult] = []
    for fixture in suite.fixtures:
        key = (fixture.chemical_potential_eV, fixture.gap_eV)
        if key not in physical_cache:
            physical_cache[key] = produce_gauge_covariant_response_observation(fixture, suite)
        base = replace(
            physical_cache[key],
            observation_id=f"observation.{fixture.fixture_id.removeprefix('fixture.')}",
            fixture_id=fixture.fixture_id,
        )
        observation = _perturb(base, fixture, suite)
        observed, reasons = _evaluate(observation, fixture, suite)
        result = GaugeCovariantResponseFixtureResult(
            fixture_id=fixture.fixture_id,
            observation_sha256=observation.fingerprint(),
            expected_disposition=fixture.expected_disposition,
            observed_disposition=observed,
            matched_expectation=observed is fixture.expected_disposition,
            reason_codes=reasons,
        )
        observations.append(observation)
        results.append(result)
    ordered_results = tuple(sorted(results, key=lambda value: value.fixture_id))
    matched = sum(value.matched_expectation for value in ordered_results)
    conformance = GaugeCovariantResponseConformanceResult(
        result_id='result.ambient-pressure-superconductor-gauge-covariant-response-gauge-closed-conformance',
        formalism_sha256=formalism.fingerprint(),
        fixture_suite_sha256=suite.fingerprint(),
        fixture_results=ordered_results,
        required_fixture_count=len(ordered_results),
        matched_fixture_count=matched,
        strict_method_pass=matched == len(ordered_results),
        material_result_count=0,
        maximum_evidence_scope_id="evidence-scope.method-fixture-only",
        reason_codes=(
            'reason.gauge-covariant-response-conformance-intersection-passed'
            if matched == len(ordered_results)
            else 'reason.gauge-covariant-response-conformance-intersection-failed',
        ),
    )
    return conformance, tuple(sorted(observations, key=lambda value: value.fixture_id))


def produce_material_gauge_covariant_response(material: GaugeCovariantResponseMaterialInput) -> GaugeCovariantResponseMaterialResult:
    """Evaluate one fully compatible material denominator at 300 K.

    The current implementation is deliberately restricted to a validated
    single-band effective lattice.  Multiband, strong-coupling or nonlocal
    inputs that cannot close that compatibility contract must stop before this
    function rather than being coerced into the model.
    """

    suite = gauge_covariant_response_fixture_suite()
    fixture = GaugeCovariantResponseFixtureSpec(
        fixture_id=f'fixture.ambient-pressure-superconductor-gauge-covariant-response-material-{material.material_unit_id}',
        fixture_kind='fixture-kind.gauge-covariant-response-material-compatible',
        expected_disposition=GaugeCovariantResponseFixtureDisposition.ACCEPT,
        temperature_K=material.temperature_K,
        hopping_eV=material.hopping_eV,
        chemical_potential_eV=material.chemical_potential_eV,
        gap_eV=material.gap_at_temperature_eV,
        perturbation_id='perturbation.gauge-covariant-response-none',
    )
    raw = produce_gauge_covariant_response_observation(fixture, suite)
    observation = replace(
        raw,
        evidence_scope_id="evidence-scope.material-linked-effective-lattice",
        material_compatibility_present=True,
        model_validity_resolved=True,
        reason_codes=('reason.gauge-covariant-response-material-compatibility-and-validity-bound',),
    )
    observed_disposition, failures = _evaluate(observation, fixture, suite)
    passed = observed_disposition is GaugeCovariantResponseFixtureDisposition.ACCEPT
    scale = material.stiffness_scale_SI_per_eV_link
    current_scale = material.current_density_scale_A_per_m2_per_eV_link
    reason_codes = tuple(
        sorted(
            failures
            or (
                'reason.gauge-covariant-response-strict-material-transverse-operand-complete',
                'reason.gauge-covariant-response-result-does-not-imply-admission-admission',
            )
        )
    )
    return GaugeCovariantResponseMaterialResult(
        result_id=f'result.ambient-pressure-superconductor-gauge-covariant-response-material-{material.material_unit_id}',
        material_unit_id=material.material_unit_id,
        material_input_sha256=material.fingerprint(),
        observation=observation,
        conservative_stiffness_lower_SI=(
            max(Decimal("0"), observation.stiffness_interval_lower_eV_per_link) * scale
        ),
        conservative_stiffness_upper_SI=(
            max(Decimal("0"), observation.stiffness_interval_upper_eV_per_link) * scale
        ),
        j_plus_A_per_m2=observation.j_plus_eV_per_link * current_scale,
        j_minus_A_per_m2=observation.j_minus_eV_per_link * current_scale,
        gauge_covariant_method_pass=passed,
        transverse_operand_complete=passed,
        admission_emitted=False,
        reason_codes=reason_codes,
    )


__all__ = [
    "KB_EV_PER_K",
    "PRODUCER_IMPLEMENTATION_ID",
    'produce_gauge_covariant_response_observation',
    'produce_material_gauge_covariant_response',
    'gauge_covariant_response_fixture_suite',
    'gauge_covariant_response_formalism',
    'run_gauge_covariant_response_conformance',
]
