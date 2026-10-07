'Fresh, method-only ambient pressure superconductor gauge covariant response development input and falsifier result.'

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_decimal,
    validate_stable_id,
)

from .gauge_covariant_response_contracts import GaugeCovariantResponseConformanceResult, GaugeCovariantResponseFixtureSuite, GaugeCovariantResponseFormalismFreeze, GaugeCovariantResponseResponseObservation
from .synthetic_gauge_covariant_response import gauge_covariant_response_fixture_suite, gauge_covariant_response_formalism


@dataclass(frozen=True, slots=True)
class SyntheticMaterialResponseMethodConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/synthetic-material-response-method-config'

    config_id: str
    independent_unit_id: str
    formalism: GaugeCovariantResponseFormalismFreeze
    fixture_suite: GaugeCovariantResponseFixtureSuite
    operating_pressure_Pa: Decimal
    requested_clock: int
    accepted_clock: int
    applied_clock: int
    receiver_clock: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        if not self.config_id.startswith(
            "empirical-lawhood-"
        ) or not self.independent_unit_id.startswith("unit.empirical-lawhood-"):
            raise ValueError(
                'ambient pressure superconductor fresh method input requires target-owned identities'
            )
        if self.formalism != gauge_covariant_response_formalism() or self.fixture_suite != gauge_covariant_response_fixture_suite():
            raise ValueError(
                'ambient pressure superconductor method input differs from the source-traced gauge covariant response gauge covariant response formalism or nine-fixture chart'
            )
        validate_decimal(
            self.operating_pressure_Pa,
            field_name="operating_pressure_Pa",
            minimum=Decimal(0),
        )
        if self.operating_pressure_Pa != Decimal(101325):
            raise ValueError('ambient pressure superconductor method chart requires ambient pressure')
        clocks = (
            self.requested_clock,
            self.accepted_clock,
            self.applied_clock,
            self.receiver_clock,
        )
        if any(type(value) is not int for value in clocks) or clocks != (0, 1, 2, 3):
            raise ValueError(
                'ambient pressure superconductor method clocks must preserve request/accept/apply/receive order'
            )


@dataclass(frozen=True, slots=True)
class SyntheticMaterialResponseMethodEvaluationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/ambient-pressure-superconductor/synthetic-material-response-method-evaluation-config'
    )

    config_id: str
    source: SyntheticMaterialResponseMethodConfig
    method_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if not self.method_only:
            raise ValueError('gauge covariant response fixtures cannot qualify a material at 300 K')


@dataclass(frozen=True, slots=True)
class SyntheticMaterialResponseMethodPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/synthetic-material-response-method-panel'

    panel_id: str
    independent_unit_id: str
    config_sha256: str
    conformance: GaugeCovariantResponseConformanceResult
    observations: tuple[GaugeCovariantResponseResponseObservation, ...]
    method_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        if len(self.config_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.config_sha256
        ):
            raise ValueError('ambient pressure superconductor method panel requires a canonical config digest')
        if not self.method_only or len(self.observations) != 9:
            raise ValueError(
                'ambient pressure superconductor method panel is one nonpromotable nine-fixture suite'
            )
        fixture_ids = tuple(value.fixture_id for value in self.observations)
        if fixture_ids != tuple(sorted(set(fixture_ids))):
            raise ValueError(
                'ambient pressure superconductor method observations must have sorted unique fixture IDs'
            )


@dataclass(frozen=True, slots=True)
class SyntheticMaterialResponseMethodCheck(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/synthetic-material-response-method-check'

    check_id: str
    independent_unit_id: str
    matched_fixture_count: int
    required_fixture_count: int
    passed: bool
    reason_codes: tuple[str, ...]
    method_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.check_id, field_name="check_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        if self.required_fixture_count != 9 or not 0 <= self.matched_fixture_count <= 9:
            raise ValueError('ambient pressure superconductor method check requires the nine nested gauge covariant response fixtures')
        if self.passed != (self.matched_fixture_count == 9) or not self.method_only:
            raise ValueError(
                'ambient pressure superconductor method check cannot promote a material or contradict fixture results'
            )
        if (
            self.passed
            and self.reason_codes
            or not self.passed
            and not self.reason_codes
        ):
            raise ValueError('ambient pressure superconductor method check reasons differ from result')


def check_synthetic_material_response_method_conformance(
    config: SyntheticMaterialResponseMethodConfig, panel: SyntheticMaterialResponseMethodPanel
) -> SyntheticMaterialResponseMethodCheck:
    """Recheck every sealed observation against the frozen method chart."""
    from .synthetic_gauge_covariant_response import _evaluate

    if (
        panel.config_sha256 != config.fingerprint()
        or panel.independent_unit_id != config.independent_unit_id
    ):
        raise ValueError('ambient pressure superconductor method panel differs from selected input')
    result = panel.conformance
    if (
        result.formalism_sha256 != config.formalism.fingerprint()
        or result.fixture_suite_sha256 != config.fixture_suite.fingerprint()
        or result.material_result_count != 0
        or result.maximum_evidence_scope_id != "evidence-scope.method-fixture-only"
        or tuple(row.fixture_id for row in result.fixture_results)
        != tuple(row.fixture_id for row in config.fixture_suite.fixtures)
        or any(
            row.expected_disposition != fixture.expected_disposition
            for row, fixture in zip(
                result.fixture_results, config.fixture_suite.fixtures, strict=True
            )
        )
    ):
        raise ValueError(
            'ambient pressure superconductor gauge covariant response result differs from the frozen method-only fixture chart'
        )
    for fixture, row, observation in zip(
        config.fixture_suite.fixtures,
        result.fixture_results,
        panel.observations,
        strict=True,
    ):
        observed, reasons = _evaluate(observation, fixture, config.fixture_suite)
        if (
            observation.fixture_id != fixture.fixture_id
            or row.observation_sha256 != observation.fingerprint()
            or row.observed_disposition is not observed
            or row.reason_codes != reasons
        ):
            raise ValueError('ambient pressure superconductor gauge covariant response observation fails sealed fixture re-evaluation')
    passed = result.strict_method_pass and result.matched_fixture_count == 9
    return SyntheticMaterialResponseMethodCheck(
        f'{config.config_id}.method-check',
        config.independent_unit_id,
        result.matched_fixture_count,
        result.required_fixture_count,
        passed,
        () if passed else ("SYNTHETIC_MATERIAL_RESPONSE_FIXTURE_CONFORMANCE_FAILED",),
        True,
    )
