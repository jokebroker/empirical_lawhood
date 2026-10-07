'Additive contracts for the ambient pressure superconductor gauge covariant response staged-gauge covariant response follow-up.\n\nThese records import the immutable material source design source design audit design candidates by fingerprint and\nadd the method/source/provider objects required by amendment gauge covariant response.  They do not\nchange the historical material source design schemas or promote method fixtures to material\nevidence.\n'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
from typing import ClassVar, Final, Mapping

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .material_source_design_contracts import SourceAssetLock


GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA: Final = (
    'empirical-lawhood/simulators/ambient-pressure-superconductor/gauge-covariant-response/config'
)
GAUGE_COVARIANT_RESPONSE_CONFIG_VERSION: Final = "1.0.0"
GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION: Final = "1.0.0"
GAUGE_COVARIANT_RESPONSE_STUDY_ID: Final = 'ambient-pressure-superconductor-gauge-covariant-development'
GAUGE_COVARIANT_RESPONSE_EXTERNAL_ROOT: Final = 'runs/run.ambient-pressure-superconductor-gauge-covariant-development'
GAUGE_COVARIANT_RESPONSE_MAXIMUM_CONFIG_BYTES: Final = 256 * 1024
GAUGE_COVARIANT_RESPONSE_MAXIMUM_OUTPUT_BYTES: Final = 8 * 1024 * 1024

GAUGE_COVARIANT_RESPONSE_EXTENSION_QUALIFICATION_CAPABILITY_KEY: Final = 'source.ambient-pressure-superconductor-gauge-covariant-response-gauge-covariant-response-extension-qualification'
GAUGE_COVARIANT_RESPONSE_CONFORMANCE_CAPABILITY_KEY: Final = 'simulator.ambient-pressure-superconductor-gauge-covariant-response-gauge-closed-gauge-covariant-response-conformance'
GAUGE_COVARIANT_RESPONSE_MATERIAL_CAPABILITY_KEY: Final = 'simulator.ambient-pressure-superconductor-gauge-covariant-response-material-gauge-closed'
GAUGE_COVARIANT_RESPONSE_DEVELOPMENT_BASIS_FREEZE_CAPABILITY_KEY: Final = 'transform.ambient-pressure-superconductor-gauge-covariant-response-development-basis-freeze'
GAUGE_COVARIANT_RESPONSE_CONTROL_SCIENCE_FREEZE_CAPABILITY_KEY: Final = 'evaluator.ambient-pressure-superconductor-gauge-covariant-response-material-control-control-science-freeze'
GAUGE_COVARIANT_RESPONSE_MATCHED_WAVE_CAPABILITY_KEY: Final = 'transform.ambient-pressure-superconductor-gauge-covariant-response-response-guided-exploration-matched-wave'
GAUGE_COVARIANT_RESPONSE_TRANSPORT_NOMINATION_CAPABILITY_KEY: Final = 'transform.ambient-pressure-superconductor-gauge-covariant-response-transport-nomination-transport-nomination'
GAUGE_COVARIANT_RESPONSE_ADMISSION_COMPILER_CAPABILITY_KEY: Final = 'evaluator.ambient-pressure-superconductor-gauge-covariant-response-computational-admission-gauge-covariant-response-admission-compiler'
GAUGE_COVARIANT_RESPONSE_CONDITIONAL_PROSPECTIVE_CONTROL_CAPABILITY_KEY: Final = 'simulator.ambient-pressure-superconductor-gauge-covariant-response-prospective-controller-validation-conditional-controller-use'
GAUGE_COVARIANT_RESPONSE_INDEPENDENT_RECURRENCE_CAPABILITY_KEY: Final = 'simulator.ambient-pressure-superconductor-gauge-covariant-response-independent-recurrence-independent-recurrence'
GAUGE_COVARIANT_RESPONSE_CLOSEOUT_CAPABILITY_KEY: Final = 'evaluator.ambient-pressure-superconductor-gauge-covariant-response-development-closeout-closeout'
GAUGE_COVARIANT_RESPONSE_ADJUDICATION_SCHEMA: Final = 'empirical-lawhood/simulators/ambient-pressure-superconductor/gauge-covariant-response/scientific-adjudication'


class GaugeCovariantResponseLifecycle(StrEnum):
    DRAFT = "DRAFT"
    FROZEN = "FROZEN"


class GaugeCovariantResponseFixtureDisposition(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


class GaugeCovariantResponseStage(StrEnum):
    MATERIAL_SOURCE_DESIGN = 'MATERIAL_SOURCE_DESIGN'
    MATERIAL_CONTROL = 'MATERIAL_CONTROL'
    DIVERSE_SEED_ACTIONS = 'DIVERSE_SEED_ACTIONS'
    RESPONSE_GUIDED_WAVE_1 = 'RESPONSE_GUIDED_WAVE_1'
    RESPONSE_GUIDED_WAVE_2 = 'RESPONSE_GUIDED_WAVE_2'
    TRANSPORT_NOMINATION = 'TRANSPORT_NOMINATION'
    COMPUTATIONAL_ADMISSION = 'COMPUTATIONAL_ADMISSION'
    PROSPECTIVE_CONTROLLER_VALIDATION = 'PROSPECTIVE_CONTROLLER_VALIDATION'
    INDEPENDENT_RECURRENCE = 'INDEPENDENT_RECURRENCE'
    DEVELOPMENT_CLOSEOUT = 'DEVELOPMENT_CLOSEOUT'


class GaugeCovariantResponseStageDisposition(StrEnum):
    SOURCE_DESIGN_PASS = "MATERIAL_SOURCE_DESIGN_SOURCE_QUALIFIED__DESIGN_BASIS_FROZEN"
    MATERIAL_CONTROL_PASS = "MATERIAL_CONTROL_RECOVERY_PASS__SEARCH_EXPLORATION_CONFORMANCE__SCIENCE_FROZEN"
    SOURCE_OPERAND_REQUIRED = "SOURCE_OPERAND_REQUIRED"
    NONATTEMPT = "NONATTEMPT_UPSTREAM_STOP"
    RESPONSE_GUIDED_WAVES_COMPLETE = 'RESPONSE_GUIDED_DEVELOPMENT_WAVES_COMPLETE'
    TRANSPORT_NOMINATION_COMPLETE = 'DEVELOPMENT_ATLAS_AND_GAUGE_COVARIANT_NOMINATION_RESULT'
    COMPUTATIONAL_ADMISSION_HOLD = "GAUGE_COVARIANT_RESPONSE_OR_ADMISSION_HOLD"
    CONTROLLER_USE_PASS = "CONTROLLER_USE_SIMULATOR_LOCAL_VALIDATED"
    CONTROLLER_USE_NOT_VALIDATED = "CONTROLLER_USE_NOT_VALIDATED"
    COMPUTATIONAL_ADMISSION_ELIGIBLE = 'COMPUTATIONAL_ADMISSION_ELIGIBLE'
    UNEVALUABLE = "UNEVALUABLE"


def _stable(
    values: tuple[str, ...], *, field_name: str, allow_empty: bool = False
) -> None:
    require_sorted_unique_strings(
        values, field_name=field_name, allow_empty=allow_empty
    )
    for value in values:
        validate_stable_id(value, field_name=field_name)


def _nonnegative(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{field_name} must be a nonnegative integer')


def _positive(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{field_name} must be a positive integer')


@dataclass(frozen=True, slots=True)
class SyntheticMaterialMethodPrerequisites(CanonicalRecord):
    """Externally frozen input hashes; these identify bytes, not authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/synthetic-material-method-prerequisites'
    base_source_design_result_sha256: str
    base_source_sha256: str
    base_roster_sha256: str
    base_exploration_sha256: str
    base_science_sha256: str

    def __post_init__(self) -> None:
        for key in self.__dataclass_fields__:
            if key.startswith("base_"):
                validate_sha256(getattr(self, key), field_name=key)

    @classmethod
    def from_config(cls, config: 'GaugeCovariantResponseConfig') -> 'SyntheticMaterialMethodPrerequisites':
        return cls(
            *(
                getattr(config, key)
                for key in cls.__dataclass_fields__
                if key.startswith("base_")
            )
        )


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseConfig:
    """Closed configuration; dispatch remains a static registered capability set."""

    payload_sha256: str
    lifecycle: GaugeCovariantResponseLifecycle
    campaign_id: str
    revision: str
    freeze_id: str
    frozen_at_utc: str
    amendment_sha256: str
    base_source_design_result_sha256: str
    base_source_sha256: str
    base_roster_sha256: str
    base_exploration_sha256: str
    base_science_sha256: str
    source_extension_sha256: str
    gauge_covariant_formalism_sha256: str
    gauge_covariant_fixture_suite_sha256: str
    development_protocol_id: str
    wave_count: int
    actions_per_policy_wave: int
    survivor_limit: int
    operating_temperature_K: Decimal
    operating_pressure_Pa: Decimal
    probe_amplitude: Decimal
    storage_root: str
    external_root: str
    minimum_free_bytes: int
    resources: tuple[tuple[str, int | bool], ...]
    allowed_authority_actions: tuple[str, ...]

    def resource(self, key: str) -> int | bool:
        try:
            return dict(self.resources)[key]
        except KeyError as error:
            raise ValueError(f'unknown gauge covariant response resource key {key}') from error


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseSourceQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-source-qualification'
    )

    qualification_id: str
    base_source_qualification_sha256: str
    extension_assets: tuple[SourceAssetLock, ...]
    credentials_required: bool
    clickthrough_required: bool
    paid_resource_required: bool
    exact_public_use_terms_accepted: bool
    material_target_values_projected: bool
    synthesis_corpus_promoting_route_evidence: bool
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_sha256(self.base_source_qualification_sha256)
        require_sorted_unique_ids(
            self.extension_assets, attribute="source_id", field_name="extension_assets"
        )
        if (
            self.credentials_required
            or self.clickthrough_required
            or self.paid_resource_required
        ):
            raise ValueError(
                'gauge covariant response source extension must remain credential-free and unbilled'
            )
        if not self.exact_public_use_terms_accepted:
            raise ValueError('gauge covariant response exact permitted use must be accepted before freeze')
        if self.material_target_values_projected:
            raise ValueError(
                'gauge covariant response source qualification cannot project material target outcomes'
            )
        if self.synthesis_corpus_promoting_route_evidence:
            raise ValueError(
                "unlicensed synthesis corpus cannot promote route admission"
            )
        _stable(self.limitations, field_name="limitations", allow_empty=False)


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseFixtureSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-fixture-spec'

    fixture_id: str
    fixture_kind: str
    expected_disposition: GaugeCovariantResponseFixtureDisposition
    temperature_K: Decimal
    hopping_eV: Decimal
    chemical_potential_eV: Decimal
    gap_eV: Decimal
    perturbation_id: str

    def __post_init__(self) -> None:
        for field_name in ("fixture_id", "fixture_kind", "perturbation_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        for field_name in ("temperature_K", "hopping_eV", "gap_eV"):
            validate_decimal(
                getattr(self, field_name), field_name=field_name, minimum=Decimal("0")
            )
        validate_decimal(self.chemical_potential_eV, field_name="chemical_potential_eV")


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseFixtureSuite(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-fixture-suite'

    suite_id: str
    fixtures: tuple[GaugeCovariantResponseFixtureSpec, ...]
    base_nx: int
    base_nky: int
    refined_nx: int
    refined_nky: int
    signed_probe_amplitude: Decimal
    derivative_step: Decimal
    oddness_relative_limit: Decimal
    nonlinear_relative_limit: Decimal
    ward_absolute_limit_eV: Decimal
    normal_cancellation_absolute_limit: Decimal
    minimum_positive_stiffness: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.suite_id, field_name="suite_id")
        require_sorted_unique_ids(
            self.fixtures, attribute="fixture_id", field_name="fixtures"
        )
        for field_name in ("base_nx", "base_nky", "refined_nx", "refined_nky"):
            _positive(getattr(self, field_name), field_name=field_name)
        if self.refined_nx <= self.base_nx or self.refined_nky <= self.base_nky:
            raise ValueError('gauge covariant response refined view must strictly refine both coordinates')
        for field_name in (
            "signed_probe_amplitude",
            "derivative_step",
            "oddness_relative_limit",
            "nonlinear_relative_limit",
            "ward_absolute_limit_eV",
            "normal_cancellation_absolute_limit",
            "minimum_positive_stiffness",
        ):
            validate_decimal(
                getattr(self, field_name), field_name=field_name, minimum=Decimal("0")
            )


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseFormalismFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-formalism-freeze'

    formalism_id: str
    producer_implementation_id: str
    source_ids: tuple[str, ...]
    equation_ids: tuple[str, ...]
    compatibility_map_ids: tuple[str, ...]
    input_operand_ids: tuple[str, ...]
    output_operand_ids: tuple[str, ...]
    validity_requirement_ids: tuple[str, ...]
    nonpromotion_rule_ids: tuple[str, ...]
    base_view_id: str
    refined_view_id: str
    independent_view_id: str
    operating_temperature_K: Decimal
    finite_q_required: bool
    normal_comparator_required: bool
    gauge_covariance_required: bool
    material_compatibility_required: bool

    def __post_init__(self) -> None:
        for field_name in (
            "formalism_id",
            "producer_implementation_id",
            "base_view_id",
            "refined_view_id",
            "independent_view_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "source_ids",
            "equation_ids",
            "compatibility_map_ids",
            "input_operand_ids",
            "output_operand_ids",
            "validity_requirement_ids",
            "nonpromotion_rule_ids",
        ):
            _stable(getattr(self, field_name), field_name=field_name, allow_empty=False)
        validate_decimal(
            self.operating_temperature_K, field_name="operating_temperature_K"
        )
        if self.operating_temperature_K != Decimal("300"):
            raise ValueError('strict gauge covariant response operating temperature must remain 300 K')
        if not all(
            (
                self.finite_q_required,
                self.normal_comparator_required,
                self.gauge_covariance_required,
                self.material_compatibility_required,
            )
        ):
            raise ValueError('strict gauge covariant response cannot relax a required construction boundary')


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseResponseObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-response-observation'
    )

    observation_id: str
    fixture_id: str
    evidence_scope_id: str
    temperature_K: Decimal
    q_base_inverse_lattice: Decimal
    q_refined_inverse_lattice: Decimal
    probe_amplitude: Decimal
    j_plus_eV_per_link: Decimal
    j_minus_eV_per_link: Decimal
    j_zero_eV_per_link: Decimal
    stiffness_base_eV_per_link: Decimal
    stiffness_refined_eV_per_link: Decimal
    stiffness_interval_lower_eV_per_link: Decimal
    stiffness_interval_upper_eV_per_link: Decimal
    superconducting_diamagnetic_eV_per_link: Decimal
    superconducting_paramagnetic_eV_per_link: Decimal
    normal_diamagnetic_eV_per_link: Decimal
    normal_paramagnetic_eV_per_link: Decimal
    normal_total_eV_per_link: Decimal
    oddness_relative_residual: Decimal
    nonlinear_relative_residual: Decimal
    ward_absolute_residual_eV: Decimal
    base_refined_common_interval: bool
    diamagnetic_term_present: bool
    material_compatibility_present: bool
    model_validity_resolved: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("observation_id", "fixture_id", "evidence_scope_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "temperature_K",
            "q_base_inverse_lattice",
            "q_refined_inverse_lattice",
            "probe_amplitude",
            "stiffness_interval_lower_eV_per_link",
            "stiffness_interval_upper_eV_per_link",
            "superconducting_diamagnetic_eV_per_link",
            "normal_diamagnetic_eV_per_link",
            "oddness_relative_residual",
            "nonlinear_relative_residual",
            "ward_absolute_residual_eV",
        ):
            validate_decimal(
                getattr(self, field_name), field_name=field_name, minimum=Decimal("0")
            )
        for field_name in (
            "j_plus_eV_per_link",
            "j_minus_eV_per_link",
            "j_zero_eV_per_link",
            "stiffness_base_eV_per_link",
            "stiffness_refined_eV_per_link",
            "superconducting_paramagnetic_eV_per_link",
            "normal_paramagnetic_eV_per_link",
            "normal_total_eV_per_link",
        ):
            validate_decimal(getattr(self, field_name), field_name=field_name)
        if (
            self.stiffness_interval_lower_eV_per_link
            > self.stiffness_interval_upper_eV_per_link
        ):
            raise ValueError('gauge covariant response stiffness interval is inverted')
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=True)


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseFixtureResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-fixture-result'

    fixture_id: str
    observation_sha256: str
    expected_disposition: GaugeCovariantResponseFixtureDisposition
    observed_disposition: GaugeCovariantResponseFixtureDisposition
    matched_expectation: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.fixture_id, field_name="fixture_id")
        validate_sha256(self.observation_sha256)
        if self.matched_expectation != (
            self.expected_disposition is self.observed_disposition
        ):
            raise ValueError('gauge covariant response fixture expectation flag differs from dispositions')
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=True)


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseConformanceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-conformance-result'

    result_id: str
    formalism_sha256: str
    fixture_suite_sha256: str
    fixture_results: tuple[GaugeCovariantResponseFixtureResult, ...]
    required_fixture_count: int
    matched_fixture_count: int
    strict_method_pass: bool
    material_result_count: int
    maximum_evidence_scope_id: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_sha256(self.formalism_sha256)
        validate_sha256(self.fixture_suite_sha256)
        require_sorted_unique_ids(
            self.fixture_results, attribute="fixture_id", field_name="fixture_results"
        )
        _positive(self.required_fixture_count, field_name="required_fixture_count")
        _nonnegative(self.matched_fixture_count, field_name="matched_fixture_count")
        _nonnegative(self.material_result_count, field_name="material_result_count")
        validate_stable_id(
            self.maximum_evidence_scope_id, field_name="maximum_evidence_scope_id"
        )
        if self.required_fixture_count != len(self.fixture_results):
            raise ValueError('gauge covariant response required fixture count differs from result roster')
        if self.matched_fixture_count != sum(
            value.matched_expectation for value in self.fixture_results
        ):
            raise ValueError('gauge covariant response matched fixture count differs')
        if self.strict_method_pass != (
            self.matched_fixture_count == self.required_fixture_count
        ):
            raise ValueError(
                'gauge covariant response strict method pass differs from exact fixture intersection'
            )
        if self.material_result_count != 0:
            raise ValueError('material source design gauge covariant response conformance cannot contain a material result')
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseMaterialInput(CanonicalRecord):
    'A material-linked effective-lattice denominator accepted by strict gauge covariant response.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-material-input'

    input_id: str
    material_unit_id: str
    preparation_sha256: str
    wannier_hamiltonian_sha256: str
    pairing_state_sha256: str
    compatibility_record_sha256: str
    compatibility_map_id: str
    temperature_K: Decimal
    hopping_eV: Decimal
    chemical_potential_eV: Decimal
    gap_at_temperature_eV: Decimal
    lattice_spacing_m: Decimal
    current_density_scale_A_per_m2_per_eV_link: Decimal
    stiffness_scale_SI_per_eV_link: Decimal
    requested_realized_preparation_match: bool
    finite_temperature_pairing_resolved: bool
    strong_coupling_validity_resolved: bool
    nonlocal_validity_resolved: bool
    single_band_effective_lattice_valid: bool
    native_unit_map_complete: bool

    def __post_init__(self) -> None:
        for field_name in ("input_id", "material_unit_id", "compatibility_map_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "preparation_sha256",
            "wannier_hamiltonian_sha256",
            "pairing_state_sha256",
            "compatibility_record_sha256",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "temperature_K",
            "hopping_eV",
            "gap_at_temperature_eV",
            "lattice_spacing_m",
            "current_density_scale_A_per_m2_per_eV_link",
            "stiffness_scale_SI_per_eV_link",
        ):
            validate_decimal(
                getattr(self, field_name), field_name=field_name, minimum=Decimal("0")
            )
        validate_decimal(self.chemical_potential_eV, field_name="chemical_potential_eV")
        if self.temperature_K != Decimal("300"):
            raise ValueError(
                'material gauge covariant response input must be the explicit 300 K pairing state'
            )
        if any(
            getattr(self, field_name) <= 0
            for field_name in (
                "hopping_eV",
                "lattice_spacing_m",
                "current_density_scale_A_per_m2_per_eV_link",
                "stiffness_scale_SI_per_eV_link",
            )
        ):
            raise ValueError('material gauge covariant response physical scales must be positive')
        if not all(
            (
                self.requested_realized_preparation_match,
                self.finite_temperature_pairing_resolved,
                self.strong_coupling_validity_resolved,
                self.nonlocal_validity_resolved,
                self.single_band_effective_lattice_valid,
                self.native_unit_map_complete,
            )
        ):
            raise ValueError('incomplete material compatibility cannot enter strict gauge covariant response')


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseMaterialResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-material-result'

    result_id: str
    material_unit_id: str
    material_input_sha256: str
    observation: GaugeCovariantResponseResponseObservation
    conservative_stiffness_lower_SI: Decimal
    conservative_stiffness_upper_SI: Decimal
    j_plus_A_per_m2: Decimal
    j_minus_A_per_m2: Decimal
    gauge_covariant_method_pass: bool
    transverse_operand_complete: bool
    admission_emitted: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("result_id", "material_unit_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        validate_sha256(self.material_input_sha256)
        if not isinstance(self.observation, GaugeCovariantResponseResponseObservation):
            raise ValueError('material gauge covariant response result requires its typed observation')
        for field_name in (
            "conservative_stiffness_lower_SI",
            "conservative_stiffness_upper_SI",
        ):
            validate_decimal(
                getattr(self, field_name), field_name=field_name, minimum=Decimal("0")
            )
        for field_name in ("j_plus_A_per_m2", "j_minus_A_per_m2"):
            validate_decimal(getattr(self, field_name), field_name=field_name)
        if self.conservative_stiffness_lower_SI > self.conservative_stiffness_upper_SI:
            raise ValueError('material gauge covariant response SI stiffness interval is inverted')
        if self.gauge_covariant_method_pass != self.transverse_operand_complete:
            raise ValueError(
                'material gauge covariant response pass must exactly close the transverse operand'
            )
        if self.gauge_covariant_method_pass and not self.observation.material_compatibility_present:
            raise ValueError('material gauge covariant response pass lacks material compatibility')
        if self.admission_emitted:
            raise ValueError('strict gauge covariant response producer cannot emit or imply admission admission')
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseDevelopmentBasisFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-development-basis-freeze'
    )

    freeze_id: str
    config_sha256: str
    amendment_sha256: str
    base_source_design_result_sha256: str
    base_source_sha256: str
    base_roster_sha256: str
    base_exploration_sha256: str
    base_science_sha256: str
    source_extension_sha256: str
    gauge_covariant_formalism_sha256: str
    gauge_covariant_conformance_sha256: str
    development_protocol_id: str
    development_protocol_sha256: str
    development_registry_sha256: str
    conditional_protocol_sha256s: tuple[str, ...]
    provider_capability_ids: tuple[str, ...]
    compatibility_map_ids: tuple[str, ...]
    calibration_derived_field_ids: tuple[str, ...]
    target_contact_count: int
    full_development_template_present: bool
    exact_provider_set_present: bool
    gauge_covariant_response_provider_present: bool
    computational_admission_slots_frozen: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        for field_name in (
            "config_sha256",
            "amendment_sha256",
            'base_source_design_result_sha256',
            "base_source_sha256",
            "base_roster_sha256",
            "base_exploration_sha256",
            "base_science_sha256",
            "source_extension_sha256",
            'gauge_covariant_formalism_sha256',
            'gauge_covariant_conformance_sha256',
            "development_protocol_sha256",
            "development_registry_sha256",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        validate_stable_id(
            self.development_protocol_id, field_name="development_protocol_id"
        )
        for value in self.conditional_protocol_sha256s:
            validate_sha256(value)
        require_sorted_unique_strings(
            self.conditional_protocol_sha256s,
            field_name="conditional_protocol_sha256s",
            allow_empty=False,
        )
        _stable(self.provider_capability_ids, field_name="provider_capability_ids")
        _stable(self.compatibility_map_ids, field_name="compatibility_map_ids")
        _stable(
            self.calibration_derived_field_ids,
            field_name="calibration_derived_field_ids",
        )
        _nonnegative(self.target_contact_count, field_name="target_contact_count")
        if self.target_contact_count != 0:
            raise ValueError('gauge covariant response freeze cannot contact development atlas or sealed prospective target outcomes')
        if not all(
            (
                self.full_development_template_present,
                self.exact_provider_set_present,
                self.gauge_covariant_response_provider_present,
                self.computational_admission_slots_frozen,
            )
        ):
            raise ValueError('incomplete gauge covariant response basis must not use the freeze schema')
        _stable(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseStageResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-stage-result'

    result_id: str
    stage: GaugeCovariantResponseStage
    disposition: GaugeCovariantResponseStageDisposition
    attempted: bool
    evaluable: bool
    upstream_result_sha256: str
    target_contact_count: int
    calibration_control_count: int
    development_material_count: int
    prospective_material_count: int
    gauge_covariant_nomination_count: int
    admission_count: int
    controller_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_sha256(self.upstream_result_sha256)
        for field_name in (
            "target_contact_count",
            'calibration_control_count',
            'development_material_count',
            'prospective_material_count',
            'gauge_covariant_nomination_count',
            'admission_count',
            "controller_count",
        ):
            _nonnegative(getattr(self, field_name), field_name=field_name)
        if self.disposition is GaugeCovariantResponseStageDisposition.NONATTEMPT and (
            self.attempted or self.evaluable
        ):
            raise ValueError("typed nonattempt cannot claim attempt or evaluability")
        if self.controller_count and not self.admission_count:
            raise ValueError('controller construction requires complete admission admission')
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


@dataclass(frozen=True, slots=True)
class GaugeCovariantResponseCloseout(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/gauge-covariant-response-closeout'

    closeout_id: str
    config_sha256: str
    design_freeze_sha256: str
    terminal_parent_sha256: str
    operational_status: str
    constructive_path_disposition_id: str
    discovery_advantage_disposition_id: str
    material_target_disposition_id: str
    maximum_claim_ceiling_id: str
    target_contact_count: int
    admission_count: int
    controller_use_validation_count: int
    computational_admission_candidate_count: int
    downstream_nonattempt_stage_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.closeout_id, field_name="closeout_id")
        for field_name in (
            "config_sha256",
            "design_freeze_sha256",
            "terminal_parent_sha256",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "operational_status",
            "constructive_path_disposition_id",
            "discovery_advantage_disposition_id",
            "material_target_disposition_id",
            "maximum_claim_ceiling_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "target_contact_count",
            'admission_count',
            'controller_use_validation_count',
            'computational_admission_candidate_count',
        ):
            _nonnegative(getattr(self, field_name), field_name=field_name)
        _stable(
            self.downstream_nonattempt_stage_ids,
            field_name="downstream_nonattempt_stage_ids",
            allow_empty=True,
        )
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f'{field_name} must be an object')
    return value


def _keys(
    value: Mapping[str, object], expected: tuple[str, ...], *, field_name: str
) -> None:
    if set(value) != set(expected):
        raise ValueError(f'{field_name} keys differ from the closed gauge covariant response schema')


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f'{field_name} must be nonempty trimmed text')
    return value


def _decimal(value: object, *, field_name: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f'{field_name} must be a decimal string')
    result = Decimal(value)
    validate_decimal(result, field_name=field_name)
    return result


def _sha(value: object, *, field_name: str) -> str:
    result = _text(value, field_name=field_name)
    validate_sha256(result, field_name=field_name)
    return result


def _int(value: object, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{field_name} must be an integer >= {minimum}')
    return value


def _strings(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f'{field_name} must be a string array')
    result = tuple(value)
    require_sorted_unique_strings(result, field_name=field_name, allow_empty=False)
    return result


def _reject_floats(value: object, *, field_name: str = "config") -> None:
    if isinstance(value, float):
        raise ValueError(f'{field_name} contains a binary float')
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_floats(item, field_name=f'{field_name}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_floats(item, field_name=f'{field_name}[{index}]')


def decode_gauge_covariant_response_config(
    document: Mapping[str, object],
    *,
    payload_sha256: str,
    expected_parents: SyntheticMaterialMethodPrerequisites,
) -> GaugeCovariantResponseConfig:
    _reject_floats(document)
    _keys(
        document,
        (
            "authority",
            "campaign_id",
            "design",
            "lifecycle",
            "operating_domain",
            "plan_id",
            "resources",
            "schema",
            "search",
            "storage",
            "version",
        ),
        field_name="config",
    )
    if (
        document["schema"] != GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA
        or document["version"] != GAUGE_COVARIANT_RESPONSE_CONFIG_VERSION
    ):
        raise ValueError('unsupported gauge covariant response config schema or version')
    if document["campaign_id"] != GAUGE_COVARIANT_RESPONSE_STUDY_ID:
        raise ValueError('gauge covariant response campaign identity differs')
    if (
        document["plan_id"]
        != 'ambient-pressure-superconductor-response-study'
    ):
        raise ValueError('gauge covariant response plan identity differs')

    lifecycle = _mapping(document["lifecycle"], field_name="lifecycle")
    _keys(
        lifecycle,
        ("freeze_id", "frozen_at_utc", "revision", "status"),
        field_name="lifecycle",
    )
    status = GaugeCovariantResponseLifecycle(_text(lifecycle["status"], field_name="lifecycle.status"))
    if status is not GaugeCovariantResponseLifecycle.FROZEN:
        raise ValueError('the issued gauge covariant response runner accepts only a frozen config')

    design = _mapping(document["design"], field_name="design")
    _keys(
        design,
        (
            "amendment_sha256",
            "base_exploration_sha256",
            'base_source_design_result_sha256',
            "base_roster_sha256",
            "base_science_sha256",
            "base_source_sha256",
            "development_protocol_id",
            'gauge_covariant_fixture_suite_sha256',
            'gauge_covariant_formalism_sha256',
            "source_extension_sha256",
        ),
        field_name="design",
    )
    if (
        _sha(design['base_source_design_result_sha256'], field_name='base_source_design_result_sha256')
        != expected_parents.base_source_design_result_sha256
    ):
        raise ValueError('gauge covariant response config does not import the immutable source design audit result')
    for key, expected in (
        ("base_source_sha256", expected_parents.base_source_sha256),
        ("base_roster_sha256", expected_parents.base_roster_sha256),
        ("base_exploration_sha256", expected_parents.base_exploration_sha256),
        ("base_science_sha256", expected_parents.base_science_sha256),
    ):
        if _sha(design[key], field_name=key) != expected:
            raise ValueError(f'gauge covariant response config changed immutable {key}')

    search = _mapping(document["search"], field_name="search")
    _keys(
        search,
        ("actions_per_policy_wave", "survivor_limit", "wave_count"),
        field_name="search",
    )
    operating = _mapping(document["operating_domain"], field_name="operating_domain")
    _keys(
        operating,
        ("operating_pressure_Pa", "operating_temperature_K", "probe_amplitude"),
        field_name="operating_domain",
    )
    temperature = _decimal(
        operating["operating_temperature_K"], field_name="operating_temperature_K"
    )
    pressure = _decimal(
        operating["operating_pressure_Pa"], field_name="operating_pressure_Pa"
    )
    if temperature != Decimal("300") or pressure != Decimal("101325"):
        raise ValueError('gauge covariant response operating state changed')

    storage = _mapping(document["storage"], field_name="storage")
    _keys(
        storage,
        ("external_root", "minimum_free_bytes", "storage_root"),
        field_name="storage",
    )
    if (
        storage["storage_root"] != 'storage.semi-os-external'
        or storage["external_root"] != GAUGE_COVARIANT_RESPONSE_EXTERNAL_ROOT
    ):
        raise ValueError('gauge covariant response storage binding differs')

    resources = _mapping(document["resources"], field_name="resources")
    resource_keys = (
        "cpu_cores",
        "gpu_devices",
        "memory_bytes",
        "network_required",
        "output_bytes",
        "scratch_bytes",
        "wall_time_seconds",
    )
    _keys(resources, resource_keys, field_name="resources")
    resource_values: list[tuple[str, int | bool]] = []
    for key in resource_keys:
        value = resources[key]
        if key == "network_required":
            if value is not False:
                raise ValueError('gauge covariant response execution must remain network-disabled')
        else:
            value = _int(value, field_name=f'resources.{key}')
        resource_values.append((key, value))

    authority = _mapping(document["authority"], field_name="authority")
    _keys(authority, ("allowed_actions", "owner_id"), field_name="authority")
    if authority["owner_id"] != "human.project-owner":
        raise ValueError('gauge covariant response authority owner differs')

    return GaugeCovariantResponseConfig(
        payload_sha256=payload_sha256,
        lifecycle=status,
        campaign_id=GAUGE_COVARIANT_RESPONSE_STUDY_ID,
        revision=_text(lifecycle["revision"], field_name="revision"),
        freeze_id=_text(lifecycle["freeze_id"], field_name="freeze_id"),
        frozen_at_utc=_text(lifecycle["frozen_at_utc"], field_name="frozen_at_utc"),
        amendment_sha256=_sha(
            design["amendment_sha256"], field_name="amendment_sha256"
        ),
        base_source_design_result_sha256=expected_parents.base_source_design_result_sha256,
        base_source_sha256=expected_parents.base_source_sha256,
        base_roster_sha256=expected_parents.base_roster_sha256,
        base_exploration_sha256=expected_parents.base_exploration_sha256,
        base_science_sha256=expected_parents.base_science_sha256,
        source_extension_sha256=_sha(
            design["source_extension_sha256"], field_name="source_extension_sha256"
        ),
        gauge_covariant_formalism_sha256=_sha(
            design['gauge_covariant_formalism_sha256'], field_name='gauge_covariant_formalism_sha256'
        ),
        gauge_covariant_fixture_suite_sha256=_sha(
            design['gauge_covariant_fixture_suite_sha256'], field_name='gauge_covariant_fixture_suite_sha256'
        ),
        development_protocol_id=_text(
            design["development_protocol_id"], field_name="development_protocol_id"
        ),
        wave_count=_int(search["wave_count"], field_name="wave_count", minimum=1),
        actions_per_policy_wave=_int(
            search["actions_per_policy_wave"],
            field_name="actions_per_policy_wave",
            minimum=1,
        ),
        survivor_limit=_int(
            search["survivor_limit"], field_name="survivor_limit", minimum=1
        ),
        operating_temperature_K=temperature,
        operating_pressure_Pa=pressure,
        probe_amplitude=_decimal(
            operating["probe_amplitude"], field_name="probe_amplitude"
        ),
        storage_root=_text(storage["storage_root"], field_name="storage_root"),
        external_root=_text(storage["external_root"], field_name="external_root"),
        minimum_free_bytes=_int(
            storage["minimum_free_bytes"], field_name="minimum_free_bytes"
        ),
        resources=tuple(resource_values),
        allowed_authority_actions=_strings(
            authority["allowed_actions"], field_name="allowed_actions"
        ),
    )


def decode_gauge_covariant_response_config_bytes(
    payload: bytes, *, expected_parents: SyntheticMaterialMethodPrerequisites
) -> GaugeCovariantResponseConfig:
    if len(payload) > GAUGE_COVARIANT_RESPONSE_MAXIMUM_CONFIG_BYTES:
        raise ValueError('gauge covariant response config exceeds its byte bound')
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('gauge covariant response config is not strict JSON') from error
    if not isinstance(document, dict):
        raise ValueError('gauge covariant response config root must be an object')
    return decode_gauge_covariant_response_config(
        document,
        payload_sha256=sha256(payload).hexdigest(),
        expected_parents=expected_parents,
    )


__all__ = [
    'GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA',
    'GAUGE_COVARIANT_RESPONSE_CONFIG_VERSION',
    'GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION',
    'GAUGE_COVARIANT_RESPONSE_STUDY_ID',
    'GAUGE_COVARIANT_RESPONSE_EXTERNAL_ROOT',
    'GAUGE_COVARIANT_RESPONSE_MAXIMUM_CONFIG_BYTES',
    'GAUGE_COVARIANT_RESPONSE_MAXIMUM_OUTPUT_BYTES',
    'GAUGE_COVARIANT_RESPONSE_EXTENSION_QUALIFICATION_CAPABILITY_KEY',
    'GAUGE_COVARIANT_RESPONSE_CONFORMANCE_CAPABILITY_KEY',
    'GAUGE_COVARIANT_RESPONSE_MATERIAL_CAPABILITY_KEY',
    'GAUGE_COVARIANT_RESPONSE_DEVELOPMENT_BASIS_FREEZE_CAPABILITY_KEY',
    'GAUGE_COVARIANT_RESPONSE_CONTROL_SCIENCE_FREEZE_CAPABILITY_KEY',
    'GAUGE_COVARIANT_RESPONSE_MATCHED_WAVE_CAPABILITY_KEY',
    'GAUGE_COVARIANT_RESPONSE_TRANSPORT_NOMINATION_CAPABILITY_KEY',
    'GAUGE_COVARIANT_RESPONSE_ADMISSION_COMPILER_CAPABILITY_KEY',
    'GAUGE_COVARIANT_RESPONSE_CONDITIONAL_PROSPECTIVE_CONTROL_CAPABILITY_KEY',
    'GAUGE_COVARIANT_RESPONSE_INDEPENDENT_RECURRENCE_CAPABILITY_KEY',
    'GAUGE_COVARIANT_RESPONSE_CLOSEOUT_CAPABILITY_KEY',
    'GAUGE_COVARIANT_RESPONSE_ADJUDICATION_SCHEMA',
    'GaugeCovariantResponseLifecycle',
    'GaugeCovariantResponseStage',
    'GaugeCovariantResponseStageDisposition',
    'SyntheticMaterialMethodPrerequisites',
    'GaugeCovariantResponseConfig',
    'GaugeCovariantResponseSourceQualification',
    'GaugeCovariantResponseDevelopmentBasisFreeze',
    'GaugeCovariantResponseStageResult',
    'GaugeCovariantResponseCloseout',
    'GaugeCovariantResponseConformanceResult',
    'GaugeCovariantResponseFixtureDisposition',
    'GaugeCovariantResponseFixtureResult',
    'GaugeCovariantResponseFixtureSpec',
    'GaugeCovariantResponseFixtureSuite',
    'GaugeCovariantResponseFormalismFreeze',
    'GaugeCovariantResponseMaterialInput',
    'GaugeCovariantResponseMaterialResult',
    'GaugeCovariantResponseResponseObservation',
    'decode_gauge_covariant_response_config_bytes',
]
