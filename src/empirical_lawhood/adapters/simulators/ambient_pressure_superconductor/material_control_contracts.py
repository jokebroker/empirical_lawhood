'Additive material control control-bundle and multiband-gauge covariant response qualification contracts.'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .material_source_design_contracts import SourceAssetLock


MATERIAL_CONTROL_CONFIG_SCHEMA = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-control/config'
MATERIAL_CONTROL_CONFIG_VERSION = "1.0.0"
MATERIAL_CONTROL_CAPABILITY_VERSION = "1.0.8"
MATERIAL_CONTROL_CAMPAIGN_ID = 'ambient-pressure-superconductor-material-control'
MATERIAL_CONTROL_EXTERNAL_ROOT = 'runs/run.ambient-pressure-superconductor-material-control'
MATERIAL_CONTROL_MAXIMUM_CONFIG_BYTES = 256 * 1024
MATERIAL_CONTROL_ADJUDICATION_SCHEMA = (
    'empirical-lawhood/simulators/ambient-pressure-superconductor/material-control/scientific-adjudication'
)

MATERIAL_CONTROL_SOURCE_CAPABILITY_KEY = 'source.ambient-pressure-superconductor-material-control-control-bundle'
MATERIAL_CONTROL_METHOD_CAPABILITY_KEY = 'falsifier.ambient-pressure-superconductor-material-control-gauge-covariant-response-methods'
MATERIAL_CONTROL_WORKFLOW_CAPABILITY_KEY = 'simulator.ambient-pressure-superconductor-material-control-control-workflow'
MATERIAL_CONTROL_PANEL_CAPABILITY_KEY = 'transform.ambient-pressure-superconductor-material-control-control-panel'
MATERIAL_CONTROL_SEARCH_WORLD_CAPABILITY_KEY = 'falsifier.ambient-pressure-superconductor-material-control-constructive-search-search-worlds'
MATERIAL_CONTROL_CLOSEOUT_CAPABILITY_KEY = 'evaluator.ambient-pressure-superconductor-material-control-control-science-freeze'


class MaterialControlLifecycle(StrEnum):
    DRAFT = "DRAFT"
    FROZEN = "FROZEN"


class ControlClass(StrEnum):
    POSITIVE_ISOTROPIC = "POSITIVE_ISOTROPIC"
    POSITIVE_ANISOTROPIC = "POSITIVE_ANISOTROPIC"
    NORMAL_METAL = "NORMAL_METAL"
    INSULATOR = "INSULATOR"
    DYNAMICALLY_UNSTABLE = "DYNAMICALLY_UNSTABLE"


class ControlDisposition(StrEnum):
    PASS = "CONTROL_CLASS_RECOVERED"
    FAIL = "CONTROL_CLASS_NOT_RECOVERED"
    UNEVALUABLE = "CONTROL_UNEVALUABLE"


class MaterialControlDisposition(StrEnum):
    PASS = "MATERIAL_CONTROL_RECOVERY_PASS__SEARCH_EXPLORATION_CONFORMANCE__SCIENCE_FROZEN"
    CONTROL_FAIL = "MATERIAL_CONTROL_RECOVERY_FAIL"
    SOURCE_FAIL = "MATERIAL_CONTROL_SOURCE_OR_ENVIRONMENT_FAIL"
    METHOD_FAIL = "MATERIAL_CONTROL_RESPONSE_OR_SEARCH_METHOD_FAIL"
    RESOURCE_STOP = "MATERIAL_CONTROL_RESOURCE_PATH_INADEQUATE"


@dataclass(frozen=True, slots=True)
class MaterialControlInputPrerequisites(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-control-input-prerequisites'
    gauge_covariant_development_freeze_sha256: str
    gauge_covariant_conformance_sha256: str
    development_roster_sha256: str
    development_source_sha256: str

    def __post_init__(self) -> None:
        for key in (
            'gauge_covariant_development_freeze_sha256',
            'gauge_covariant_conformance_sha256',
            'development_roster_sha256',
            'development_source_sha256',
        ):
            validate_sha256(getattr(self, key), field_name=key)


@dataclass(frozen=True, slots=True)
class MaterialControlConfig:
    predecessors: MaterialControlInputPrerequisites
    payload_sha256: str
    lifecycle: MaterialControlLifecycle
    campaign_id: str
    revision: str
    freeze_id: str
    frozen_at_utc: str
    gauge_covariant_development_freeze_sha256: str
    gauge_covariant_conformance_sha256: str
    control_bundle_sha256: str
    bridge_qualification_sha256: str
    implementation_sha256: str
    control_input_archive_sha256: str
    workflow_profile_ids: tuple[str, ...]
    target_contact_count: int
    storage_root: str
    external_root: str
    minimum_free_bytes: int
    resources: tuple[tuple[str, int | bool], ...]
    allowed_authority_actions: tuple[str, ...]

    def resource(self, key: str) -> int | bool:
        try:
            return dict(self.resources)[key]
        except KeyError as error:
            raise ValueError(f'unknown material control resource key {key}') from error


class BridgeDisposition(StrEnum):
    CONDITIONAL_METHOD_PASS = 'MULTIBAND_STRONG_COUPLING_GAUGE_COVARIANT_CONDITIONAL_METHOD_PASS'
    METHOD_FAIL = 'MULTIBAND_STRONG_COUPLING_GAUGE_COVARIANT_METHOD_FAIL'


class MaterialGaugeCovariantCompatibilityDisposition(StrEnum):
    PASS = 'MATERIAL_GAUGE_COVARIANT_COMPATIBILITY_PASS'
    UNRESOLVED = "MODEL_VALIDITY_UNRESOLVED"


class MaterialGaugeCovariantViewDisposition(StrEnum):
    PASS = "MATERIAL_GAUGE_COVARIANT_VIEW_PASS"
    NUMERICAL_FAIL = "MATERIAL_GAUGE_COVARIANT_VIEW_NUMERICAL_FAIL"
    VALIDITY_UNRESOLVED = "MODEL_VALIDITY_UNRESOLVED"


def _stable(
    values: tuple[str, ...], *, field_name: str, allow_empty: bool = False
) -> None:
    require_sorted_unique_strings(
        values, field_name=field_name, allow_empty=allow_empty
    )
    for value in values:
        validate_stable_id(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class MaterialControlControlBundleQualification(CanonicalRecord):
    """Exact held Pb/MgB2 workflow assets and their scientific role boundary."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-control-control-bundle-qualification'

    qualification_id: str
    base_source_qualification_sha256: str
    gauge_covariant_source_extension_sha256: str
    tutorial_asset: SourceAssetLock
    tutorial_page_asset: SourceAssetLock
    control_input_asset: SourceAssetLock
    environment_profile_id: str
    environment_image_sha256: str
    container_image_digest: str
    binary_sha256s: tuple[tuple[str, str], ...]
    workflow_control_ids: tuple[str, ...]
    science_view_ids: tuple[str, ...]
    tutorial_view_id: str
    credentials_required: bool
    clickthrough_required: bool
    paid_resource_required: bool
    target_outcomes_projected: bool
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_sha256(self.base_source_qualification_sha256)
        validate_sha256(self.gauge_covariant_source_extension_sha256)
        validate_stable_id(
            self.environment_profile_id, field_name="environment_profile_id"
        )
        validate_sha256(self.environment_image_sha256)
        if not self.container_image_digest.startswith("sha256:"):
            raise ValueError("container image must use a sha256 identity")
        validate_sha256(self.container_image_digest.removeprefix("sha256:"))
        names = tuple(name for name, _digest in self.binary_sha256s)
        if names != tuple(sorted(set(names))) or not names:
            raise ValueError("binary identities must be sorted and unique")
        for _name, digest in self.binary_sha256s:
            validate_sha256(digest)
        _stable(self.workflow_control_ids, field_name="workflow_control_ids")
        _stable(self.science_view_ids, field_name="science_view_ids")
        validate_stable_id(self.tutorial_view_id, field_name="tutorial_view_id")
        if self.tutorial_view_id in self.science_view_ids:
            raise ValueError(
                "tutorial workflow view cannot impersonate a frozen SSSP science view"
            )
        if (
            self.credentials_required
            or self.clickthrough_required
            or self.paid_resource_required
        ):
            raise ValueError('material control control bundle must remain public and unbilled')
        if self.target_outcomes_projected:
            raise ValueError('calibration source qualification cannot project development atlas/sealed prospective outcomes')
        _stable(self.limitations, field_name="limitations")


@dataclass(frozen=True, slots=True)
class MultibandStrongCouplingBridgeQualification(CanonicalRecord):
    """Conformance result for the multiband strong-coupling material bridge."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/multiband-strong-coupling-bridge-qualification'
    )

    qualification_id: str
    implementation_sha256: str
    formalism_source_ids: tuple[str, ...]
    equation_ids: tuple[str, ...]
    compatibility_map_ids: tuple[str, ...]
    validity_requirement_ids: tuple[str, ...]
    orbital_count: int
    matsubara_count: int
    temperature_K: Decimal
    gauge_covariance_residual_eV_per_cell: Decimal
    signed_current_residual_eV_per_cell: Decimal
    decomposition_residual_eV_per_cell: Decimal
    finite_q_fit_residual_eV_per_cell: Decimal
    normal_intercept_eV_per_cell: Decimal
    raw_normal_intercept_eV_per_cell: Decimal
    matsubara_convergence_relative: Decimal
    finite_difference_convergence_relative: Decimal
    paired_intercept_eV_per_cell: Decimal
    inverse_penetration_depth_sq_per_m2: Decimal
    disposition: BridgeDisposition
    material_promotion_authorized: bool
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_sha256(self.implementation_sha256)
        for field_name in (
            "formalism_source_ids",
            "equation_ids",
            "compatibility_map_ids",
            "validity_requirement_ids",
        ):
            _stable(getattr(self, field_name), field_name=field_name)
        if self.orbital_count < 2 or self.matsubara_count <= 0:
            raise ValueError(
                "bridge qualification requires multiband Matsubara evidence"
            )
        for field_name in (
            "temperature_K",
            "gauge_covariance_residual_eV_per_cell",
            "signed_current_residual_eV_per_cell",
            "decomposition_residual_eV_per_cell",
            "finite_q_fit_residual_eV_per_cell",
            "normal_intercept_eV_per_cell",
            "raw_normal_intercept_eV_per_cell",
            "matsubara_convergence_relative",
            "finite_difference_convergence_relative",
            "paired_intercept_eV_per_cell",
            "inverse_penetration_depth_sq_per_m2",
        ):
            validate_decimal(
                getattr(self, field_name), field_name=field_name, minimum=Decimal("0")
            )
        if self.material_promotion_authorized:
            raise ValueError(
                "method qualification alone cannot authorize material promotion"
            )
        _stable(self.limitations, field_name="limitations")
        passed = (
            self.gauge_covariance_residual_eV_per_cell <= Decimal("1e-10")
            and self.signed_current_residual_eV_per_cell <= Decimal("1e-9")
            and self.decomposition_residual_eV_per_cell <= Decimal("1e-8")
            and self.finite_q_fit_residual_eV_per_cell
            <= Decimal("0.1") * self.paired_intercept_eV_per_cell
            and self.normal_intercept_eV_per_cell <= Decimal("1e-12")
            and self.matsubara_convergence_relative <= Decimal("0.05")
            and self.finite_difference_convergence_relative <= Decimal("0.05")
            and self.paired_intercept_eV_per_cell > Decimal("0")
            and self.inverse_penetration_depth_sq_per_m2 > Decimal("0")
        )
        if (self.disposition is BridgeDisposition.CONDITIONAL_METHOD_PASS) != passed:
            raise ValueError(
                "bridge disposition differs from noncompensating conformance checks"
            )


@dataclass(frozen=True, slots=True)
class MultibandStrongCouplingMaterialCompatibility(CanonicalRecord):
    'One-view compatibility of material operands with the conditional gauge covariant response method.\n\n    The result is deliberately noncompensating.  A small fit residual cannot\n    rescue a missing common gauge, omitted-position bound, Matsubara-decay\n    proxy, Migdal-parameter proxy, static transverse pair condition or\n    native-to-SI map.\n    '

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/multiband-strong-coupling-material-compatibility'
    )

    compatibility_id: str
    structure_id: str
    view_id: str
    normal_hamiltonian_sha256: str
    eliashberg_state_sha256: str
    orbital_centres_sha256: str
    common_wannier_gauge: bool
    local_self_energy_supported: bool
    phonon_spectrum_available: bool
    migdal_parameter_proxy_resolved: bool
    static_transverse_pair_decoupling_supported: bool
    static_coulomb_gauge_supported: bool
    native_to_si_map_complete: bool
    pairing_projection_relative_residual: Decimal
    pairing_projection_relative_limit: Decimal
    hamiltonian_truncation_eV: Decimal
    hamiltonian_truncation_limit_eV: Decimal
    position_matrix_omission_relative_upper: Decimal
    position_matrix_omission_relative_limit: Decimal
    matsubara_decay_proxy_relative_upper: Decimal
    matsubara_decay_proxy_relative_limit: Decimal
    migdal_parameter_relative_upper: Decimal
    migdal_parameter_relative_limit: Decimal
    disposition: MaterialGaugeCovariantCompatibilityDisposition
    material_promotion_authorized: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("compatibility_id", "structure_id", "view_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "normal_hamiltonian_sha256",
            "eliashberg_state_sha256",
            "orbital_centres_sha256",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "pairing_projection_relative_residual",
            "pairing_projection_relative_limit",
            "hamiltonian_truncation_eV",
            "hamiltonian_truncation_limit_eV",
            "position_matrix_omission_relative_upper",
            "position_matrix_omission_relative_limit",
            "matsubara_decay_proxy_relative_upper",
            "matsubara_decay_proxy_relative_limit",
            "migdal_parameter_relative_upper",
            "migdal_parameter_relative_limit",
        ):
            validate_decimal(
                getattr(self, field_name), field_name=field_name, minimum=Decimal("0")
            )
        if self.material_promotion_authorized:
            raise ValueError(
                'one-view material-gauge covariant response compatibility cannot authorize promotion'
            )
        passed = all(
            (
                self.common_wannier_gauge,
                self.local_self_energy_supported,
                self.phonon_spectrum_available,
                self.migdal_parameter_proxy_resolved,
                self.static_transverse_pair_decoupling_supported,
                self.static_coulomb_gauge_supported,
                self.native_to_si_map_complete,
                self.pairing_projection_relative_residual
                <= self.pairing_projection_relative_limit,
                self.hamiltonian_truncation_eV <= self.hamiltonian_truncation_limit_eV,
                self.position_matrix_omission_relative_upper
                <= self.position_matrix_omission_relative_limit,
                self.matsubara_decay_proxy_relative_upper
                <= self.matsubara_decay_proxy_relative_limit,
                self.migdal_parameter_relative_upper
                <= self.migdal_parameter_relative_limit,
            )
        )
        if (self.disposition is MaterialGaugeCovariantCompatibilityDisposition.PASS) != passed:
            raise ValueError(
                'material-gauge covariant response disposition differs from its validity intersection'
            )
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


@dataclass(frozen=True, slots=True)
class MultibandStrongCouplingMaterialViewResult(CanonicalRecord):
    """One material preparation and numerical view; never a cross-view promotion."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/multiband-strong-coupling-material-view-result'
    )

    result_id: str
    structure_id: str
    view_id: str
    compatibility_sha256: str
    compatibility_passed: bool
    temperature_K: Decimal
    mode_count: int
    gauge_covariance_residual_eV_per_cell: Decimal
    signed_current_residual_eV_per_cell: Decimal
    decomposition_residual_eV_per_cell: Decimal
    finite_q_fit_residual_eV_per_cell: Decimal
    finite_q_fit_relative_limit: Decimal
    intercept_eV_per_cell: Decimal
    raw_normal_intercept_eV_per_cell: Decimal
    inverse_penetration_depth_sq_per_m2: Decimal
    finite_difference_convergence_relative: Decimal
    spatial_grid_convergence_relative: Decimal
    matsubara_convergence_relative: Decimal
    uncertainty_relative_upper: Decimal
    stiffness_lower_per_m2: Decimal
    stiffness_upper_per_m2: Decimal
    disposition: MaterialGaugeCovariantViewDisposition
    material_promotion_authorized: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("result_id", "structure_id", "view_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        validate_sha256(self.compatibility_sha256, field_name="compatibility_sha256")
        if self.mode_count < 3:
            raise ValueError('material gauge covariant response requires at least three finite-q modes')
        for field_name in (
            "temperature_K",
            "gauge_covariance_residual_eV_per_cell",
            "signed_current_residual_eV_per_cell",
            "decomposition_residual_eV_per_cell",
            "finite_q_fit_residual_eV_per_cell",
            "finite_q_fit_relative_limit",
            "raw_normal_intercept_eV_per_cell",
            "finite_difference_convergence_relative",
            "spatial_grid_convergence_relative",
            "matsubara_convergence_relative",
            "uncertainty_relative_upper",
        ):
            validate_decimal(
                getattr(self, field_name), field_name=field_name, minimum=Decimal("0")
            )
        for field_name in (
            "intercept_eV_per_cell",
            "inverse_penetration_depth_sq_per_m2",
            "stiffness_lower_per_m2",
            "stiffness_upper_per_m2",
        ):
            validate_decimal(getattr(self, field_name), field_name=field_name)
        if self.temperature_K <= 0:
            raise ValueError('material gauge covariant response temperature must be positive')
        if self.stiffness_lower_per_m2 > self.stiffness_upper_per_m2:
            raise ValueError('material gauge covariant response stiffness interval is inverted')
        if self.material_promotion_authorized:
            raise ValueError("one numerical view cannot authorize material promotion")
        numerical_pass = all(
            (
                self.gauge_covariance_residual_eV_per_cell <= Decimal("1e-9"),
                self.signed_current_residual_eV_per_cell <= Decimal("1e-8"),
                self.decomposition_residual_eV_per_cell <= Decimal("1e-7"),
                self.finite_q_fit_residual_eV_per_cell
                <= self.finite_q_fit_relative_limit * abs(self.intercept_eV_per_cell),
                self.finite_difference_convergence_relative <= Decimal("0.05"),
                self.spatial_grid_convergence_relative <= Decimal("0.10"),
                self.matsubara_convergence_relative <= Decimal("0.05"),
                self.intercept_eV_per_cell > 0,
                self.inverse_penetration_depth_sq_per_m2 > 0,
                self.stiffness_lower_per_m2 > 0,
            )
        )
        expected = (
            MaterialGaugeCovariantViewDisposition.PASS
            if self.compatibility_passed and numerical_pass
            else (
                MaterialGaugeCovariantViewDisposition.NUMERICAL_FAIL
                if self.compatibility_passed
                else MaterialGaugeCovariantViewDisposition.VALIDITY_UNRESOLVED
            )
        )
        if self.disposition is not expected:
            raise ValueError(
                'material-gauge covariant response view disposition differs from its intersection'
            )
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


@dataclass(frozen=True, slots=True)
class MaterialControlTutorialReproduction(CanonicalRecord):
    """Official EPW workflow reproduction kept outside the SSSP science panel."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-control-tutorial-reproduction'
    )

    reproduction_id: str
    structure_id: str
    control_class: ControlClass
    view_id: str
    source_asset_sha256: str
    preparation_sha256: str
    exit_code: int
    timed_out: bool
    terminal_step_id: str
    wall_time_seconds: Decimal
    peak_scratch_bytes: int
    workflow_completed: bool
    pairing_recovered: bool
    electron_phonon_lambda: Decimal
    tc_estimate_K: Decimal
    gap_min_meV: Decimal
    gap_max_meV: Decimal
    raw_archive_sha256: str
    raw_archive_bytes: int
    science_calibration_authorized: bool
    physical_independent_unit_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "reproduction_id",
            "structure_id",
            "view_id",
            "terminal_step_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        if self.control_class not in {
            ControlClass.POSITIVE_ISOTROPIC,
            ControlClass.POSITIVE_ANISOTROPIC,
        }:
            raise ValueError("tutorial reproduction is reserved for positive workflows")
        for field_name in (
            "source_asset_sha256",
            "preparation_sha256",
            "raw_archive_sha256",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "wall_time_seconds",
            "electron_phonon_lambda",
            "tc_estimate_K",
            "gap_min_meV",
            "gap_max_meV",
        ):
            validate_decimal(
                getattr(self, field_name), field_name=field_name, minimum=Decimal("0")
            )
        if self.peak_scratch_bytes <= 0:
            raise ValueError("tutorial reproduction peak scratch must be positive")
        if self.workflow_completed != (self.exit_code == 0 and not self.timed_out):
            raise ValueError(
                "tutorial workflow completion differs from process evidence"
            )
        if self.gap_min_meV > self.gap_max_meV:
            raise ValueError("tutorial gap interval is inverted")
        if self.pairing_recovered and not (
            self.workflow_completed and self.tc_estimate_K > 0 and self.gap_max_meV > 0
        ):
            raise ValueError("tutorial pairing recovery lacks workflow evidence")
        if self.science_calibration_authorized:
            raise ValueError(
                "tutorial numerics cannot calibrate the frozen SSSP science views"
            )
        if self.physical_independent_unit_count != 1:
            raise ValueError(
                "one tutorial preparation is one physical independent unit"
            )
        if self.raw_archive_bytes <= 0:
            raise ValueError("tutorial reproduction raw archive must be nonempty")
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


@dataclass(frozen=True, slots=True)
class MaterialControlControlObservation(CanonicalRecord):
    """One physical-material/numerical-view result; nested rows are not replicates."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-control-control-observation'
    )

    observation_id: str
    structure_id: str
    control_class: ControlClass
    view_id: str
    source_asset_sha256: str
    input_member_sha256: str
    requested_preparation_sha256: str
    accepted_preparation_sha256: str
    applied_preparation_sha256: str
    realized_preparation_sha256: str
    requested_realized_match: bool
    exit_code: int
    timed_out: bool
    terminal_step_id: str
    wall_time_seconds: Decimal
    peak_scratch_bytes: int
    workflow_completed: bool
    scf_completed: bool
    scf_converged: bool
    scf_repeat_evaluable: bool
    phonon_evaluable: bool
    epw_evaluable: bool
    gauge_covariant_response_evaluable: bool
    metallic: bool
    dynamically_stable: bool
    positive_superconductor_recovered: bool
    total_energy_Ry: Decimal
    scf_accuracy_Ry: Decimal
    scf_repeat_residual_Ry: Decimal
    minimum_phonon_frequency_cm1: Decimal
    electron_phonon_lambda: Decimal
    tc_estimate_K: Decimal
    gap_min_meV: Decimal
    gap_max_meV: Decimal
    band_gap_eV: Decimal
    gauge_covariant_stiffness_lower_per_m2: Decimal
    gauge_covariant_stiffness_upper_per_m2: Decimal
    raw_archive_sha256: str
    raw_archive_bytes: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "observation_id",
            "structure_id",
            "view_id",
            "terminal_step_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "source_asset_sha256",
            "input_member_sha256",
            "requested_preparation_sha256",
            "accepted_preparation_sha256",
            "applied_preparation_sha256",
            "realized_preparation_sha256",
            "raw_archive_sha256",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        if self.requested_realized_match != (
            len(
                {
                    self.requested_preparation_sha256,
                    self.accepted_preparation_sha256,
                    self.applied_preparation_sha256,
                    self.realized_preparation_sha256,
                }
            )
            == 1
        ):
            raise ValueError("requested/accepted/applied/realized match flag differs")
        for field_name in (
            "wall_time_seconds",
            "scf_accuracy_Ry",
            "scf_repeat_residual_Ry",
            "electron_phonon_lambda",
            "tc_estimate_K",
            "gap_min_meV",
            "gap_max_meV",
            "band_gap_eV",
            'gauge_covariant_stiffness_lower_per_m2',
            'gauge_covariant_stiffness_upper_per_m2',
        ):
            validate_decimal(
                getattr(self, field_name), field_name=field_name, minimum=Decimal("0")
            )
        if self.peak_scratch_bytes <= 0:
            raise ValueError("control observation peak scratch must be positive")
        if self.workflow_completed != (self.exit_code == 0 and not self.timed_out):
            raise ValueError(
                "control workflow completion differs from process evidence"
            )
        for field_name in ("total_energy_Ry", "minimum_phonon_frequency_cm1"):
            validate_decimal(getattr(self, field_name), field_name=field_name)
        if self.gap_min_meV > self.gap_max_meV:
            raise ValueError("control gap interval is inverted")
        if self.gauge_covariant_stiffness_lower_per_m2 > self.gauge_covariant_stiffness_upper_per_m2:
            raise ValueError('control gauge covariant response interval is inverted')
        if self.raw_archive_bytes <= 0:
            raise ValueError("control raw archive must be nonempty")
        if self.scf_repeat_evaluable and not self.scf_completed:
            raise ValueError("SCF repeat evidence requires a completed SCF")
        if self.positive_superconductor_recovered and not (
            self.control_class
            in {ControlClass.POSITIVE_ISOTROPIC, ControlClass.POSITIVE_ANISOTROPIC}
            and self.epw_evaluable
            and self.tc_estimate_K > 0
            and self.gap_max_meV > 0
        ):
            raise ValueError("positive-control recovery lacks pairing evidence")
        if (
            self.gauge_covariant_response_evaluable
            and self.control_class
            in {ControlClass.POSITIVE_ISOTROPIC, ControlClass.POSITIVE_ANISOTROPIC}
            and self.gauge_covariant_stiffness_upper_per_m2 <= 0
        ):
            raise ValueError('evaluable control gauge covariant response requires a positive upper stiffness')
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=True)


@dataclass(frozen=True, slots=True)
class MaterialControlControlPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-control-control-panel'

    panel_id: str
    structure_id: str
    control_class: ControlClass
    observations: tuple[MaterialControlControlObservation, ...]
    physical_independent_unit_count: int
    nested_numerical_view_count: int
    disposition: ControlDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("panel_id", "structure_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        if not self.observations:
            raise ValueError("control panel cannot be empty")
        if tuple(sorted(value.view_id for value in self.observations)) != tuple(
            value.view_id for value in self.observations
        ):
            raise ValueError("control observations must be sorted by numerical view")
        if len({value.view_id for value in self.observations}) != len(
            self.observations
        ):
            raise ValueError("control panel cannot duplicate a numerical view")
        if any(
            value.structure_id != self.structure_id
            or value.control_class is not self.control_class
            for value in self.observations
        ):
            raise ValueError("control panel mixes material units or classes")
        if self.physical_independent_unit_count != 1:
            raise ValueError(
                "nested numerical views cannot inflate physical replication"
            )
        if self.nested_numerical_view_count != len(self.observations):
            raise ValueError("nested numerical-view count differs")
        science = tuple(
            value
            for value in self.observations
            if value.view_id
            in {"view.pbe-efficiency-base", "view.pbe-precision-refined"}
        )
        if len(science) != 2 or not all(
            value.requested_realized_match for value in self.observations
        ):
            recovered = False
        elif self.control_class in {
            ControlClass.POSITIVE_ISOTROPIC,
            ControlClass.POSITIVE_ANISOTROPIC,
        }:
            recovered = (
                len(self.observations) == 2
                and all(
                    value.positive_superconductor_recovered
                    for value in self.observations
                )
                and all(
                    value.workflow_completed and value.scf_repeat_evaluable
                    for value in self.observations
                )
                and all(value.gauge_covariant_response_evaluable for value in science)
                and max(value.gauge_covariant_stiffness_lower_per_m2 for value in science) > 0
                and max(value.gauge_covariant_stiffness_lower_per_m2 for value in science)
                <= min(value.gauge_covariant_stiffness_upper_per_m2 for value in science)
            )
        elif self.control_class is ControlClass.NORMAL_METAL:
            recovered = all(
                value.workflow_completed
                and value.scf_repeat_evaluable
                and value.scf_converged
                and value.metallic
                for value in science
            )
        elif self.control_class is ControlClass.INSULATOR:
            recovered = all(
                value.workflow_completed
                and value.scf_repeat_evaluable
                and value.scf_converged
                and not value.metallic
                and value.band_gap_eV > 0
                for value in science
            )
        else:
            recovered = all(
                value.workflow_completed
                and value.scf_repeat_evaluable
                and value.phonon_evaluable
                and not value.dynamically_stable
                for value in science
            )
        if (self.disposition is ControlDisposition.PASS) != recovered:
            raise ValueError(
                "control-panel disposition differs from its class intersection"
            )
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=True)


@dataclass(frozen=True, slots=True)
class MaterialControlTruthWorldConformance(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-control-truth-world-conformance'

    result_id: str
    exploration_design_sha256: str
    world_result_ids: tuple[str, ...]
    required_world_count: int
    matched_world_count: int
    response_policy_pass: bool
    scalar_policy_pass: bool
    random_policy_pass: bool
    accounting_pass: bool
    hold_logic_pass: bool
    target_contact_count: int
    disposition_id: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_sha256(self.exploration_design_sha256)
        _stable(self.world_result_ids, field_name="world_result_ids")
        validate_stable_id(self.disposition_id, field_name="disposition_id")
        if self.required_world_count <= 0 or self.matched_world_count < 0:
            raise ValueError('constructive search world counts are invalid')
        if self.matched_world_count > self.required_world_count:
            raise ValueError('constructive search matched count exceeds its roster')
        if self.target_contact_count != 0:
            raise ValueError('material control constructive search conformance cannot contact target materials')
        passed = self.matched_world_count == self.required_world_count and all(
            (
                self.response_policy_pass,
                self.scalar_policy_pass,
                self.random_policy_pass,
                self.accounting_pass,
                self.hold_logic_pass,
            )
        )
        if (self.disposition_id == 'constructive-search.exploration-conformance-pass') != passed:
            raise ValueError('constructive search disposition differs from its intersection')
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=True)


@dataclass(frozen=True, slots=True)
class MaterialControlTruthWorldResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-control-truth-world-result'

    result_id: str
    world_id: str
    public_graph_sha256: str
    privileged_truth_sha256: str
    expected_disposition_id: str
    observed_disposition_id: str
    policy_nomination_sha256s: tuple[tuple[str, str], ...]
    policy_history_separated: bool
    accounting_passed: bool
    hold_logic_passed: bool
    matched_expected: bool
    target_contact_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "result_id",
            "world_id",
            "expected_disposition_id",
            "observed_disposition_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        validate_sha256(self.public_graph_sha256)
        validate_sha256(self.privileged_truth_sha256)
        policy_ids = tuple(value[0] for value in self.policy_nomination_sha256s)
        if policy_ids != tuple(sorted(set(policy_ids))) or len(policy_ids) != 3:
            raise ValueError('constructive search world result requires three sorted policy records')
        for policy_id, digest in self.policy_nomination_sha256s:
            validate_stable_id(
                policy_id, field_name="policy_nomination_sha256s.policy_id"
            )
            validate_sha256(digest)
        if self.target_contact_count != 0:
            raise ValueError('constructive search world result cannot contact a target material')
        matched = (
            self.expected_disposition_id == self.observed_disposition_id
            and self.policy_history_separated
            and self.accounting_passed
            and self.hold_logic_passed
        )
        if self.matched_expected != matched:
            raise ValueError('constructive search world match flag differs from its intersection')
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


@dataclass(frozen=True, slots=True)
class MaterialControlCalibrationFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-control-calibration-freeze'
    )

    freeze_id: str
    control_panel_sha256s: tuple[str, ...]
    numerical_noise_Ry: Decimal
    cross_view_relative_limit: Decimal
    phonon_stability_tolerance_cm1: Decimal
    receiver_tolerance: Decimal
    receiver_identity_residual_limit_eV_per_cell: Decimal
    finite_q_fit_relative_limit: Decimal
    rounding_rule_ids: tuple[str, ...]
    target_outcomes_used: bool
    all_predeclared_fields_resolved: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        require_sorted_unique_strings(
            self.control_panel_sha256s,
            field_name="control_panel_sha256s",
            allow_empty=False,
        )
        for value in self.control_panel_sha256s:
            validate_sha256(value)
        for field_name in (
            "numerical_noise_Ry",
            "cross_view_relative_limit",
            "phonon_stability_tolerance_cm1",
            "receiver_tolerance",
            "receiver_identity_residual_limit_eV_per_cell",
            "finite_q_fit_relative_limit",
        ):
            validate_decimal(
                getattr(self, field_name), field_name=field_name, minimum=Decimal("0")
            )
        _stable(self.rounding_rule_ids, field_name="rounding_rule_ids")
        if self.target_outcomes_used:
            raise ValueError('material control calibration cannot consume development atlas or sealed prospective outcomes')
        if not self.all_predeclared_fields_resolved:
            raise ValueError("incomplete calibration cannot use the freeze schema")
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


@dataclass(frozen=True, slots=True)
class MaterialControlScienceFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-control-science-freeze'

    freeze_id: str
    config_sha256: str
    gauge_covariant_development_freeze_sha256: str
    control_bundle_sha256: str
    bridge_qualification_sha256: str
    gauge_covariant_conformance_sha256: str
    tutorial_reproduction_sha256s: tuple[str, ...]
    material_gauge_covariant_result_sha256s: tuple[str, ...]
    control_panel_sha256s: tuple[str, ...]
    calibration_freeze_sha256: str
    constructive_search_conformance_sha256: str
    development_roster_sha256: str
    development_source_sha256: str
    resource_envelope_sha256: str
    target_contact_count: int
    calibration_control_count: int
    all_controls_passed: bool
    constructive_search_passed: bool
    science_frozen_before_development_contact: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        for field_name in (
            "config_sha256",
            'gauge_covariant_development_freeze_sha256',
            "control_bundle_sha256",
            "bridge_qualification_sha256",
            'gauge_covariant_conformance_sha256',
            "calibration_freeze_sha256",
            'constructive_search_conformance_sha256',
            'development_roster_sha256',
            'development_source_sha256',
            "resource_envelope_sha256",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        require_sorted_unique_strings(
            self.control_panel_sha256s,
            field_name="control_panel_sha256s",
            allow_empty=False,
        )
        for value in self.control_panel_sha256s:
            validate_sha256(value)
        for field_name, values, expected_count in (
            ("tutorial_reproduction_sha256s", self.tutorial_reproduction_sha256s, 2),
            ('material_gauge_covariant_result_sha256s', self.material_gauge_covariant_result_sha256s, 4),
        ):
            require_sorted_unique_strings(
                values, field_name=field_name, allow_empty=False
            )
            if len(values) != expected_count:
                raise ValueError(
                    f'{field_name} requires exactly {expected_count} records'
                )
            for value in values:
                validate_sha256(value, field_name=field_name)
        if self.target_contact_count != 0 or self.calibration_control_count != 5:
            raise ValueError(
                'material control freeze requires five calibration units and zero target contact'
            )
        if not all(
            (self.all_controls_passed, self.constructive_search_passed, self.science_frozen_before_development_contact)
        ):
            raise ValueError(
                'failed material control intersection cannot use the science-freeze schema'
            )
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


@dataclass(frozen=True, slots=True)
class MaterialControlStageResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-control-stage-result'

    result_id: str
    disposition: MaterialControlDisposition
    attempted: bool
    evaluable: bool
    science_freeze_sha256: str
    calibration_control_count: int
    development_material_count: int
    prospective_material_count: int
    target_contact_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.science_freeze_sha256:
            validate_sha256(self.science_freeze_sha256)
        for field_name in (
            'calibration_control_count',
            'development_material_count',
            'prospective_material_count',
            "target_contact_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f'{field_name} must be nonnegative')
        if self.disposition is MaterialControlDisposition.PASS and (
            not self.evaluable or len(self.science_freeze_sha256) != 64
        ):
            raise ValueError('material control pass requires its exact science freeze')
        if (
            self.development_material_count
            or self.prospective_material_count
            or self.target_contact_count
        ):
            raise ValueError('material control cannot contain target-material contact')
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


@dataclass(frozen=True, slots=True)
class MaterialControlCloseout(CanonicalRecord):
    'Total material control result with an optional, genuinely successful science freeze.\n\n    The wrapper is emitted for every scientifically valid terminal disposition.\n    It avoids inventing a freeze-shaped object when controls, source custody,\n    resources or method validity do not pass their noncompensating intersection.\n    '

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-control-closeout'

    closeout_id: str
    control_bundle_sha256: str
    bridge_qualification_sha256: str
    gauge_covariant_conformance_sha256: str
    tutorial_reproduction_sha256s: tuple[str, ...]
    material_gauge_covariant_result_sha256s: tuple[str, ...]
    control_panel_sha256s: tuple[str, ...]
    constructive_search_conformance_sha256: str
    calibration_freeze: MaterialControlCalibrationFreeze | None
    science_freeze: MaterialControlScienceFreeze | None
    stage_result: MaterialControlStageResult
    target_contact_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.closeout_id, field_name="closeout_id")
        for field_name in (
            "control_bundle_sha256",
            "bridge_qualification_sha256",
            'gauge_covariant_conformance_sha256',
            'constructive_search_conformance_sha256',
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        for field_name, values in (
            ("tutorial_reproduction_sha256s", self.tutorial_reproduction_sha256s),
            ('material_gauge_covariant_result_sha256s', self.material_gauge_covariant_result_sha256s),
            ("control_panel_sha256s", self.control_panel_sha256s),
        ):
            require_sorted_unique_strings(
                values, field_name=field_name, allow_empty=False
            )
            for value in values:
                validate_sha256(value, field_name=field_name)
        if (
            self.target_contact_count != 0
            or self.stage_result.target_contact_count != 0
        ):
            raise ValueError('material control closeout cannot contain target contact')
        passed = self.stage_result.disposition is MaterialControlDisposition.PASS
        if passed != (
            self.calibration_freeze is not None and self.science_freeze is not None
        ):
            raise ValueError(
                'material control closeout freeze presence differs from its disposition'
            )
        if passed:
            if self.science_freeze is None:
                raise ValueError('passed material control closeout lacks a science freeze')
            if (
                self.stage_result.science_freeze_sha256
                != self.science_freeze.fingerprint()
            ):
                raise ValueError(
                    'material control stage result differs from the nested science freeze'
                )
        if self.science_freeze is not None:
            if self.calibration_freeze is None:
                raise ValueError('material control science freeze lacks a calibration freeze')
            if (
                self.science_freeze.calibration_freeze_sha256
                != self.calibration_freeze.fingerprint()
            ):
                raise ValueError('material control science and calibration freezes differ')
        _stable(self.reason_codes, field_name="reason_codes", allow_empty=False)


def _config_mapping(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f'{field_name} must be an object')
    return value


def _config_keys(
    value: dict[str, object], expected: tuple[str, ...], *, field_name: str
) -> None:
    if set(value) != set(expected):
        raise ValueError(f'{field_name} keys differ from the closed material control schema')


def _config_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f'{field_name} must be nonempty trimmed text')
    return value


def _config_sha(value: object, *, field_name: str) -> str:
    result = _config_text(value, field_name=field_name)
    validate_sha256(result, field_name=field_name)
    return result


def _config_int(value: object, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{field_name} must be an integer >= {minimum}')
    return value


def _config_strings(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f'{field_name} must be a string array')
    result = tuple(value)
    require_sorted_unique_strings(result, field_name=field_name, allow_empty=False)
    return result


def _reject_config_floats(value: object, *, field_name: str = "config") -> None:
    if isinstance(value, float):
        raise ValueError(f'{field_name} contains a binary float')
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_config_floats(item, field_name=f'{field_name}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_config_floats(item, field_name=f'{field_name}[{index}]')


def decode_material_control_config(
    document: dict[str, object],
    *,
    payload_sha256: str,
    expected_parents: MaterialControlInputPrerequisites,
) -> MaterialControlConfig:
    _reject_config_floats(document)
    _config_keys(
        document,
        (
            "authority",
            "campaign_id",
            "controls",
            "design",
            "lifecycle",
            "plan_id",
            "resources",
            "schema",
            "storage",
            "version",
        ),
        field_name="config",
    )
    if (
        document["schema"] != MATERIAL_CONTROL_CONFIG_SCHEMA
        or document["version"] != MATERIAL_CONTROL_CONFIG_VERSION
    ):
        raise ValueError('unsupported material control config schema or version')
    if document["campaign_id"] != MATERIAL_CONTROL_CAMPAIGN_ID:
        raise ValueError('material control campaign identity differs')
    if (
        document["plan_id"]
        != 'ambient-pressure-superconductor-response-study'
    ):
        raise ValueError('material control plan identity differs')

    lifecycle = _config_mapping(document["lifecycle"], field_name="lifecycle")
    _config_keys(
        lifecycle,
        ("freeze_id", "frozen_at_utc", "revision", "status"),
        field_name="lifecycle",
    )
    status = MaterialControlLifecycle(
        _config_text(lifecycle["status"], field_name="lifecycle.status")
    )
    if status is not MaterialControlLifecycle.FROZEN:
        raise ValueError('the issued material control runner accepts only a frozen config')
    revision = _config_text(lifecycle["revision"], field_name="lifecycle.revision")
    if revision != 'ambient-pressure-superconductor-material-control':
        raise ValueError('material control follow-up revision differs')

    design = _config_mapping(document["design"], field_name="design")
    _config_keys(
        design,
        (
            'gauge_covariant_development_freeze_sha256',
            'gauge_covariant_conformance_sha256',
            "bridge_qualification_sha256",
            "control_bundle_sha256",
            "control_input_archive_sha256",
            'development_roster_sha256',
            'development_source_sha256',
            "implementation_sha256",
        ),
        field_name="design",
    )
    gauge_covariant_freeze = _config_sha(
        design['gauge_covariant_development_freeze_sha256'],
        field_name='gauge_covariant_development_freeze_sha256',
    )
    gauge_covariant_conformance = _config_sha(
        design['gauge_covariant_conformance_sha256'], field_name='gauge_covariant_conformance_sha256'
    )
    for key in ('development_roster_sha256', 'development_source_sha256'):
        if _config_sha(design[key], field_name=key) != getattr(expected_parents, key):
            raise ValueError('material control development atlas predeclaration differs from explicit parents')
    if (
        gauge_covariant_freeze != expected_parents.gauge_covariant_development_freeze_sha256
        or gauge_covariant_conformance != expected_parents.gauge_covariant_conformance_sha256
    ):
        raise ValueError('material control config changed its immutable gauge covariant response predecessors')

    controls = _config_mapping(document["controls"], field_name="controls")
    _config_keys(
        controls,
        ("target_contact_count", "workflow_profile_ids"),
        field_name="controls",
    )
    target_contact_count = _config_int(
        controls["target_contact_count"], field_name="target_contact_count"
    )
    if target_contact_count != 0:
        raise ValueError('material control config cannot authorize target contact')

    storage = _config_mapping(document["storage"], field_name="storage")
    _config_keys(
        storage,
        ("external_root", "minimum_free_bytes", "storage_root"),
        field_name="storage",
    )
    if storage["storage_root"] != 'storage.semi-os-external':
        raise ValueError('material control storage identity differs')
    if storage["external_root"] != MATERIAL_CONTROL_EXTERNAL_ROOT:
        raise ValueError('material control external run root differs')

    resources = _config_mapping(document["resources"], field_name="resources")
    resource_keys = (
        "cpu_cores",
        "gpu_devices",
        "memory_bytes",
        "network_required",
        "output_bytes",
        "scratch_bytes",
        "wall_time_seconds",
    )
    _config_keys(resources, resource_keys, field_name="resources")
    resource_values: list[tuple[str, int | bool]] = []
    for key in resource_keys:
        value = resources[key]
        if key == "network_required":
            if value is not False:
                raise ValueError(
                    'material control scientific execution must remain network-disabled'
                )
        else:
            value = _config_int(value, field_name=f'resources.{key}')
        resource_values.append((key, value))
    resource_map = dict(resource_values)
    if (
        resource_map["cpu_cores"] != 8
        or resource_map["gpu_devices"] != 0
        or int(resource_map["memory_bytes"]) < 24 * 1024**3
        or int(resource_map["wall_time_seconds"]) < 86_400
        or int(resource_map["wall_time_seconds"]) > 7 * 86_400
        or int(resource_map["output_bytes"]) < 8 * 1024**3
        or int(resource_map["output_bytes"]) > 64 * 1024**3
        or resource_map["scratch_bytes"] != resource_map["output_bytes"]
    ):
        raise ValueError('material control resources differ from the enforceable QE/EPW envelope')

    authority = _config_mapping(document["authority"], field_name="authority")
    _config_keys(authority, ("allowed_actions", "owner_id"), field_name="authority")
    if authority["owner_id"] != "human.project-owner":
        raise ValueError('material control authority owner differs')
    allowed_actions = _config_strings(
        authority["allowed_actions"], field_name="allowed_actions"
    )
    if allowed_actions != (
        "EVALUATOR_REVEAL",
        "NONACTUATING_PROSPECTIVE_FREEZE",
        "SIMULATION_EXECUTION",
    ):
        raise ValueError(
            'material control authority actions differ from the nonactuating control/reveal act'
        )

    return MaterialControlConfig(
        predecessors=expected_parents,
        payload_sha256=payload_sha256,
        lifecycle=status,
        campaign_id=MATERIAL_CONTROL_CAMPAIGN_ID,
        revision=revision,
        freeze_id=_config_text(lifecycle["freeze_id"], field_name="freeze_id"),
        frozen_at_utc=_config_text(
            lifecycle["frozen_at_utc"], field_name="frozen_at_utc"
        ),
        gauge_covariant_development_freeze_sha256=gauge_covariant_freeze,
        gauge_covariant_conformance_sha256=gauge_covariant_conformance,
        control_bundle_sha256=_config_sha(
            design["control_bundle_sha256"], field_name="control_bundle_sha256"
        ),
        bridge_qualification_sha256=_config_sha(
            design["bridge_qualification_sha256"],
            field_name="bridge_qualification_sha256",
        ),
        implementation_sha256=_config_sha(
            design["implementation_sha256"], field_name="implementation_sha256"
        ),
        control_input_archive_sha256=_config_sha(
            design["control_input_archive_sha256"],
            field_name="control_input_archive_sha256",
        ),
        workflow_profile_ids=_config_strings(
            controls["workflow_profile_ids"], field_name="workflow_profile_ids"
        ),
        target_contact_count=0,
        storage_root='storage.semi-os-external',
        external_root=MATERIAL_CONTROL_EXTERNAL_ROOT,
        minimum_free_bytes=_config_int(
            storage["minimum_free_bytes"],
            field_name="minimum_free_bytes",
            minimum=256 * 1024**3,
        ),
        resources=tuple(resource_values),
        allowed_authority_actions=allowed_actions,
    )


def decode_material_control_config_bytes(
    payload: bytes, *, expected_parents: MaterialControlInputPrerequisites
) -> MaterialControlConfig:
    if len(payload) > MATERIAL_CONTROL_MAXIMUM_CONFIG_BYTES:
        raise ValueError('material control config exceeds its byte bound')
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('material control config is not strict JSON') from error
    if not isinstance(document, dict) or not all(
        isinstance(key, str) for key in document
    ):
        raise ValueError('material control config root must be an object')
    return decode_material_control_config(
        document,
        payload_sha256=sha256(payload).hexdigest(),
        expected_parents=expected_parents,
    )


__all__ = [
    "MATERIAL_CONTROL_ADJUDICATION_SCHEMA",
    'MaterialControlCalibrationFreeze',
    'MaterialControlCloseout',
    'MaterialControlConfig',
    'MaterialControlControlBundleQualification',
    'MaterialControlControlObservation',
    'MaterialControlControlPanel',
    'MaterialControlDisposition',
    'MaterialControlLifecycle',
    'MaterialControlScienceFreeze',
    'MaterialControlStageResult',
    'MaterialControlTutorialReproduction',
    'MaterialControlTruthWorldConformance',
    'MaterialControlTruthWorldResult',
    "BridgeDisposition",
    "ControlClass",
    "ControlDisposition",
    'MultibandStrongCouplingMaterialCompatibility',
    'MaterialGaugeCovariantCompatibilityDisposition',
    'MaterialGaugeCovariantViewDisposition',
    'MultibandStrongCouplingMaterialViewResult',
    'MultibandStrongCouplingBridgeQualification',
    'decode_material_control_config_bytes',
]
