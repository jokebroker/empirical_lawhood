'Typed ambient pressure superconductor material source design source and design-freeze contracts.\n\nThe records in this module are adapter-owned additions over the existing\nprogramme, source-resolution, candidate, receipt and recovery contracts.  They\ndo not redefine the response law, admission geometry or controller compiler.\n'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
from typing import ClassVar, Final, Mapping

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)


MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA: Final = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-source-design/config'
MATERIAL_SOURCE_DESIGN_CONFIG_VERSION: Final = "1.0.0"
MATERIAL_SOURCE_DESIGN_CAPABILITY_VERSION: Final = "1.0.0"
MATERIAL_SOURCE_DESIGN_CAMPAIGN_ID: Final = 'ambient-pressure-superconductor-material-source-design'
MATERIAL_SOURCE_DESIGN_EXTERNAL_ROOT: Final = 'runs/ambient-pressure-300k-superconductor/material-source-design-source-design-audit'
MATERIAL_SOURCE_DESIGN_MAXIMUM_CONFIG_BYTES: Final = 512 * 1024
MATERIAL_SOURCE_DESIGN_MAXIMUM_OUTPUT_BYTES: Final = 8 * 1024 * 1024

MATERIAL_SOURCE_DESIGN_SOURCE_CAPABILITY_KEY: Final = 'source.ambient-pressure-superconductor-public-source-qualification.material-source-design'
MATERIAL_SOURCE_DESIGN_DESIGN_CAPABILITY_KEY: Final = 'transform.ambient-pressure-superconductor-design-basis-freeze.material-source-design'
MATERIAL_SOURCE_DESIGN_EVALUATOR_CAPABILITY_KEY: Final = 'evaluator.ambient-pressure-superconductor-design-basis-freeze.material-source-design'
MATERIAL_SOURCE_DESIGN_ADJUDICATION_SCHEMA: Final = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-source-design/scientific-adjudication'
MATERIAL_SOURCE_DESIGN_SOURCE_PLAN_ID: Final = 'source.ambient-pressure-superconductor-material-source-design-public-roster'


class MaterialSourceDesignConfigLifecycle(StrEnum):
    DRAFT = "DRAFT"
    FROZEN = "FROZEN"


class SplitPartition(StrEnum):
    CALIBRATION = 'CALIBRATION'
    DEVELOPMENT_ATLAS = 'DEVELOPMENT_ATLAS'
    SEALED_PROSPECTIVE = 'SEALED_PROSPECTIVE'


class ActionStage(StrEnum):
    CALIBRATION = "CALIBRATION"
    INITIAL_DEVELOPMENT = "INITIAL_DEVELOPMENT"
    ELIGIBLE_WAVE = "ELIGIBLE_WAVE"
    SEALED_PROSPECTIVE = "SEALED_PROSPECTIVE"


class SourceOutcomeRole(StrEnum):
    NO_OUTCOMES = "NO_OUTCOMES"
    METHOD_EVIDENCE = "METHOD_EVIDENCE"
    HISTORICAL_OUTCOMES_NOT_PROJECTED = "HISTORICAL_OUTCOMES_NOT_PROJECTED"


class RouteEvidenceStatus(StrEnum):
    CONTROL_TUTORIAL = "CONTROL_TUTORIAL"
    PROPOSAL_ONLY = "PROPOSAL_ONLY"
    UNRESOLVED = "UNRESOLVED"


class MaterialSourceDesignDisposition(StrEnum):
    PASS = "MATERIAL_SOURCE_DESIGN_SOURCE_QUALIFIED__DESIGN_BASIS_FROZEN"
    SOURCE_NOT_QUALIFIED = "MATERIAL_SOURCE_DESIGN_SOURCE_NOT_QUALIFIED"
    DESIGN_NOT_FROZEN = "MATERIAL_SOURCE_DESIGN_BASIS_NOT_FROZEN"


class MaterialGateDisposition(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNEVALUABLE = "UNEVALUABLE"


def _stable_values(values: tuple[str, ...], *, field_name: str, allow_empty: bool = True) -> None:
    require_sorted_unique_strings(values, field_name=field_name, allow_empty=allow_empty)
    for value in values:
        validate_stable_id(value, field_name=field_name)


def _positive_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f'{field_name} must be a positive integer')


def _nonnegative_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{field_name} must be a nonnegative integer')


@dataclass(frozen=True, slots=True)
class MaterialSourceDesignConfig:
    'Closed material source design configuration; executable paths and dispatch are not data.'

    payload_sha256: str
    lifecycle: MaterialSourceDesignConfigLifecycle
    campaign_id: str
    revision: str
    freeze_id: str | None
    frozen_at_utc: str | None
    source_qualification_sha256: str
    material_roster_sha256: str
    exploration_design_sha256: str
    science_design_sha256: str
    development_protocol_id: str
    wave_count: int
    actions_per_policy_wave: int
    initial_development_action_count: int
    policy_action_budget: int
    policy_compute_budget_units: int
    high_fidelity_survivor_limit: int
    family_restart_quota: int
    bridge_action_quota_per_wave: int
    policy_ids: tuple[str, ...]
    deterministic_seed: str
    operating_temperature_K: Decimal
    operating_pressure_Pa: Decimal
    probe_field_T: Decimal
    slab_thickness_m: Decimal
    required_ambient_lifetime_s: Decimal
    maximum_process_pressure_Pa: Decimal
    maximum_process_temperature_K: Decimal
    maximum_process_duration_s: Decimal
    storage_root: str
    external_root: str
    minimum_free_bytes: int
    resources: tuple[tuple[str, int | bool], ...]
    allowed_authority_actions: tuple[str, ...]

    def resource(self, key: str) -> int | bool:
        try:
            return dict(self.resources)[key]
        except KeyError as error:
            raise ValueError(f'unknown material source design resource key {key}') from error


@dataclass(frozen=True, slots=True)
class SourceMemberLock(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/source-member-lock'

    member_id: str
    member_locator: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        validate_relative_locator(self.member_locator)
        _positive_int(self.size_bytes, field_name="size_bytes")
        validate_sha256(self.sha256)


@dataclass(frozen=True, slots=True)
class SourceAssetLock(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/source-asset-lock'

    source_id: str
    role_id: str
    release_id: str
    relative_locator: str
    public_locator: str
    content_sha256: str
    size_bytes: int
    published_checksum: str
    license_id: str
    license_scope: str
    outcome_role: SourceOutcomeRole
    archive_member_count: int
    archive_regular_file_count: int
    archive_directory_count: int
    archive_symlink_count: int
    archive_expanded_regular_bytes: int
    archive_maximum_member_bytes: int
    member_locks: tuple[SourceMemberLock, ...]
    qualification_checks: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("source_id", "role_id", "release_id", "license_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        validate_relative_locator(self.relative_locator)
        validate_nonempty(self.public_locator, field_name="public_locator")
        validate_sha256(self.content_sha256)
        _positive_int(self.size_bytes, field_name="size_bytes")
        validate_nonempty(self.published_checksum, field_name="published_checksum")
        validate_nonempty(self.license_scope, field_name="license_scope")
        for field_name in (
            "archive_member_count",
            "archive_regular_file_count",
            "archive_directory_count",
            "archive_symlink_count",
            "archive_expanded_regular_bytes",
            "archive_maximum_member_bytes",
        ):
            _nonnegative_int(getattr(self, field_name), field_name=field_name)
        require_sorted_unique_ids(
            self.member_locks,
            attribute="member_id",
            field_name="member_locks",
        )
        _stable_values(
            self.qualification_checks,
            field_name="qualification_checks",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class PseudopotentialLock(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/pseudopotential-lock'

    pseudopotential_id: str
    set_id: str
    element: str
    filename: str
    size_bytes: int
    md5: str
    sha256: str
    functional: str
    family: str
    wavefunction_cutoff_Ry: Decimal
    charge_density_cutoff_Ry: Decimal
    license_id: str

    def __post_init__(self) -> None:
        for field_name in ("pseudopotential_id", "set_id", "license_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        validate_nonempty(self.element, field_name="element")
        validate_nonempty(self.filename, field_name="filename")
        _positive_int(self.size_bytes, field_name="size_bytes")
        if len(self.md5) != 32 or any(
            character not in "0123456789abcdef" for character in self.md5
        ):
            raise ValueError("md5 must be a lowercase MD5 digest")
        validate_sha256(self.sha256)
        validate_nonempty(self.functional, field_name="functional")
        validate_nonempty(self.family, field_name="family")
        validate_decimal(
            self.wavefunction_cutoff_Ry,
            field_name="wavefunction_cutoff_Ry",
            minimum=Decimal("0"),
        )
        validate_decimal(
            self.charge_density_cutoff_Ry,
            field_name="charge_density_cutoff_Ry",
            minimum=Decimal("0"),
        )
        if self.wavefunction_cutoff_Ry <= 0 or self.charge_density_cutoff_Ry <= 0:
            raise ValueError("pseudopotential cutoffs must be positive")


@dataclass(frozen=True, slots=True)
class MaterialSourceDesignSourceQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-source-design-source-qualification'

    qualification_id: str
    source_plan_id: str
    assets: tuple[SourceAssetLock, ...]
    pseudopotentials: tuple[PseudopotentialLock, ...]
    selected_element_symbols: tuple[str, ...]
    three_dsc_row_count: int
    three_dsc_column_count: int
    three_dsc_target_column_present: bool
    three_dsc_target_values_projected: bool
    xi_reference_value: Decimal
    xi_reference_temperature: Decimal
    xi_reference_tolerance: Decimal
    credentials_required: bool
    clickthrough_required: bool
    paid_resource_required: bool
    exact_public_terms_accepted: bool
    source_values_used_for_candidate_ranking: bool
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_stable_id(self.source_plan_id, field_name="source_plan_id")
        require_sorted_unique_ids(self.assets, attribute="source_id", field_name="assets")
        require_sorted_unique_ids(
            self.pseudopotentials,
            attribute="pseudopotential_id",
            field_name="pseudopotentials",
        )
        require_sorted_unique_strings(
            self.selected_element_symbols,
            field_name="selected_element_symbols",
            allow_empty=False,
        )
        _positive_int(self.three_dsc_row_count, field_name="three_dsc_row_count")
        _positive_int(self.three_dsc_column_count, field_name="three_dsc_column_count")
        for field_name in (
            "xi_reference_value",
            "xi_reference_temperature",
            "xi_reference_tolerance",
        ):
            validate_decimal(getattr(self, field_name), field_name=field_name, minimum=Decimal("0"))
        if self.credentials_required or self.clickthrough_required or self.paid_resource_required:
            raise ValueError('material source design source roster must remain credential-free and unbilled')
        if not self.exact_public_terms_accepted:
            raise ValueError('material source design source terms must be accepted before freeze')
        if not self.three_dsc_target_column_present or self.three_dsc_target_values_projected:
            raise ValueError('3DSC target values must exist but remain outside the material source design projection')
        if self.source_values_used_for_candidate_ranking:
            raise ValueError('material source design sources cannot rank development atlas or sealed prospective candidates')
        _stable_values(self.limitations, field_name="limitations")


@dataclass(frozen=True, slots=True)
class MaterialSiteSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-site-spec'

    site_id: str
    element: str
    fractional_coordinates: tuple[Decimal, Decimal, Decimal]
    occupancy: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.site_id, field_name="site_id")
        validate_nonempty(self.element, field_name="element")
        for coordinate in self.fractional_coordinates:
            validate_decimal(coordinate, field_name="fractional_coordinate")
            if coordinate < 0 or coordinate >= 1:
                raise ValueError("fractional coordinates must lie in [0, 1)")
        validate_decimal(self.occupancy, field_name="occupancy")
        if self.occupancy != Decimal("1"):
            raise ValueError('material source design admits only full-occupancy ordered prototypes')


@dataclass(frozen=True, slots=True)
class MaterialStructureSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-structure-spec'

    structure_id: str
    candidate_id: str
    formula: tuple[tuple[str, int], ...]
    family_id: str
    partition: SplitPartition
    prototype_id: str
    space_group_number: int
    lattice_vectors_A: tuple[
        tuple[Decimal, Decimal, Decimal],
        tuple[Decimal, Decimal, Decimal],
        tuple[Decimal, Decimal, Decimal],
    ]
    sites: tuple[MaterialSiteSpec, ...]
    charge_state: str
    magnetic_state: str
    source_mode: str
    process_history_id: str
    outcome_visibility: str

    def __post_init__(self) -> None:
        for field_name in (
            "structure_id",
            "candidate_id",
            "family_id",
            "prototype_id",
            "process_history_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        if not self.formula or tuple(sorted(self.formula)) != self.formula:
            raise ValueError("formula must be sorted and nonempty")
        if len({element for element, _count in self.formula}) != len(self.formula):
            raise ValueError("formula contains duplicate elements")
        for element, count in self.formula:
            validate_nonempty(element, field_name="formula.element")
            _positive_int(count, field_name="formula.count")
        if isinstance(self.space_group_number, bool) or not 1 <= self.space_group_number <= 230:
            raise ValueError("space_group_number must lie in [1, 230]")
        for vector in self.lattice_vectors_A:
            for coordinate in vector:
                validate_decimal(coordinate, field_name="lattice_vectors_A")
        require_sorted_unique_ids(self.sites, attribute="site_id", field_name="sites")
        expected = {element: count for element, count in self.formula}
        observed: dict[str, int] = {}
        for site in self.sites:
            observed[site.element] = observed.get(site.element, 0) + 1
        if observed != expected:
            raise ValueError("site elements do not reproduce the declared integer formula")
        validate_nonempty(self.charge_state, field_name="charge_state")
        validate_nonempty(self.magnetic_state, field_name="magnetic_state")
        validate_stable_id(self.source_mode, field_name="source_mode")
        validate_stable_id(self.outcome_visibility, field_name="outcome_visibility")


@dataclass(frozen=True, slots=True)
class MaterialFamilySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-family-spec'

    family_id: str
    partition: SplitPartition
    prototype_id: str
    chemical_domain_id: str
    mechanism_lane_id: str
    grouping_key: str
    seed_structure_ids: tuple[str, ...]
    coefficient_pooling_allowed: bool

    def __post_init__(self) -> None:
        for field_name in (
            "family_id",
            "prototype_id",
            "chemical_domain_id",
            "mechanism_lane_id",
            "grouping_key",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        _stable_values(self.seed_structure_ids, field_name="seed_structure_ids", allow_empty=False)


@dataclass(frozen=True, slots=True)
class FormationRouteSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/formation-route-spec'

    route_id: str
    process_history_id: str
    route_class_id: str
    evidence_status: RouteEvidenceStatus
    pressure_lower_Pa: Decimal
    pressure_upper_Pa: Decimal
    temperature_lower_K: Decimal
    temperature_upper_K: Decimal
    duration_upper_s: Decimal
    atmosphere_id: str
    equipment_class_id: str
    evidence_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "route_id",
            "process_history_id",
            "route_class_id",
            "atmosphere_id",
            "equipment_class_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "pressure_lower_Pa",
            "pressure_upper_Pa",
            "temperature_lower_K",
            "temperature_upper_K",
            "duration_upper_s",
        ):
            validate_decimal(getattr(self, field_name), field_name=field_name, minimum=Decimal("0"))
        if self.pressure_lower_Pa > self.pressure_upper_Pa:
            raise ValueError("formation pressure interval is reversed")
        if self.temperature_lower_K > self.temperature_upper_K:
            raise ValueError("formation temperature interval is reversed")
        _stable_values(self.evidence_source_ids, field_name="evidence_source_ids")


@dataclass(frozen=True, slots=True)
class MaterialActionSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-action-spec'

    action_id: str
    parent_structure_id: str
    child_structure_id: str
    action_kind: str
    site_class_id: str
    from_species: str
    to_species: str
    fraction: Decimal
    stage: ActionStage
    route_id: str
    action_cost_units: int
    compute_cost_units: int
    discontinuous_bridge: bool

    def __post_init__(self) -> None:
        for field_name in (
            "action_id",
            "parent_structure_id",
            "child_structure_id",
            "action_kind",
            "site_class_id",
            "route_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        validate_nonempty(self.from_species, field_name="from_species")
        validate_nonempty(self.to_species, field_name="to_species")
        validate_decimal(self.fraction, field_name="fraction")
        if self.fraction <= 0 or self.fraction > 1:
            raise ValueError("action fraction must lie in (0, 1]")
        _positive_int(self.action_cost_units, field_name="action_cost_units")
        _positive_int(self.compute_cost_units, field_name="compute_cost_units")


@dataclass(frozen=True, slots=True)
class MaterialRosterFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-roster-freeze'

    roster_id: str
    families: tuple[MaterialFamilySpec, ...]
    structures: tuple[MaterialStructureSpec, ...]
    actions: tuple[MaterialActionSpec, ...]
    formation_routes: tuple[FormationRouteSpec, ...]
    calibration_structure_ids: tuple[str, ...]
    development_initial_action_ids: tuple[str, ...]
    development_eligible_wave_action_ids: tuple[str, ...]
    prospective_action_ids: tuple[str, ...]
    grouping_algorithm_id: str
    canonicalization_algorithm_id: str
    duplicate_action_child_groups: tuple[tuple[str, tuple[str, ...]], ...]
    unresolved_disorder_count: int
    unresolved_partial_occupancy_count: int
    family_partition_collision_count: int
    target_outcome_values_used: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        require_sorted_unique_ids(self.families, attribute="family_id", field_name="families")
        require_sorted_unique_ids(
            self.structures,
            attribute="structure_id",
            field_name="structures",
        )
        require_sorted_unique_ids(self.actions, attribute="action_id", field_name="actions")
        require_sorted_unique_ids(
            self.formation_routes,
            attribute="route_id",
            field_name="formation_routes",
        )
        for field_name in (
            'calibration_structure_ids',
            'development_initial_action_ids',
            'development_eligible_wave_action_ids',
            'prospective_action_ids',
        ):
            _stable_values(getattr(self, field_name), field_name=field_name, allow_empty=False)
        validate_stable_id(self.grouping_algorithm_id, field_name="grouping_algorithm_id")
        validate_stable_id(
            self.canonicalization_algorithm_id,
            field_name="canonicalization_algorithm_id",
        )
        for child_id, action_ids in self.duplicate_action_child_groups:
            validate_stable_id(child_id, field_name="duplicate_action_child_groups.child")
            _stable_values(
                action_ids, field_name="duplicate_action_child_groups.actions", allow_empty=False
            )
            if len(action_ids) < 2:
                raise ValueError("a duplicate child group must contain at least two actions")
        for field_name in (
            "unresolved_disorder_count",
            "unresolved_partial_occupancy_count",
            "family_partition_collision_count",
        ):
            _nonnegative_int(getattr(self, field_name), field_name=field_name)
        if any(
            getattr(self, field_name) != 0
            for field_name in (
                "unresolved_disorder_count",
                "unresolved_partial_occupancy_count",
                "family_partition_collision_count",
            )
        ):
            raise ValueError(
                'the material source design frozen narrow roster must close identity and split collisions'
            )
        if self.target_outcome_values_used:
            raise ValueError('target outcomes cannot construct the material source design roster')

        family_by_id = {family.family_id: family for family in self.families}
        structure_by_id = {structure.structure_id: structure for structure in self.structures}
        route_ids = {route.route_id for route in self.formation_routes}
        action_by_id = {action.action_id: action for action in self.actions}
        for structure in self.structures:
            family = family_by_id.get(structure.family_id)
            if family is None or family.partition is not structure.partition:
                raise ValueError("every structure must reference a same-partition frozen family")
            a, b, c = structure.lattice_vectors_A
            determinant = (
                a[0] * (b[1] * c[2] - b[2] * c[1])
                - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0])
            )
            if determinant == 0:
                raise ValueError("a frozen material lattice must have nonzero volume")
            coordinates = tuple(site.fractional_coordinates for site in structure.sites)
            if len(set(coordinates)) != len(coordinates):
                raise ValueError("a frozen ordered structure cannot contain coincident sites")
        for family in self.families:
            for seed_id in family.seed_structure_ids:
                seed = structure_by_id.get(seed_id)
                if seed is None or seed.family_id != family.family_id:
                    raise ValueError("family seeds must reference structures in that family")

        accounted_action_ids = (
            set(self.development_initial_action_ids)
            | set(self.development_eligible_wave_action_ids)
            | set(self.prospective_action_ids)
        )
        if accounted_action_ids != set(action_by_id):
            raise ValueError('every material source design action must occur in exactly one declared stage roster')
        if (
            set(self.development_initial_action_ids) & set(self.development_eligible_wave_action_ids)
            or set(self.development_initial_action_ids) & set(self.prospective_action_ids)
            or set(self.development_eligible_wave_action_ids) & set(self.prospective_action_ids)
        ):
            raise ValueError('material source design action stage rosters must be disjoint')
        expected_stage_sets = {
            ActionStage.INITIAL_DEVELOPMENT: set(self.development_initial_action_ids),
            ActionStage.ELIGIBLE_WAVE: set(self.development_eligible_wave_action_ids),
            ActionStage.SEALED_PROSPECTIVE: set(self.prospective_action_ids),
        }
        for action in self.actions:
            parent = structure_by_id.get(action.parent_structure_id)
            child = structure_by_id.get(action.child_structure_id)
            if parent is None or child is None or action.route_id not in route_ids:
                raise ValueError("actions must close over frozen structures and formation routes")
            if parent.family_id != child.family_id or parent.partition is not child.partition:
                raise ValueError('an material source design action cannot cross a frozen family or partition')
            if action.action_id not in expected_stage_sets[action.stage]:
                raise ValueError("action stage differs from its declared roster")
            expected_partition = (
                SplitPartition.SEALED_PROSPECTIVE
                if action.stage is ActionStage.SEALED_PROSPECTIVE
                else SplitPartition.DEVELOPMENT_ATLAS
            )
            if parent.partition is not expected_partition:
                raise ValueError("action stage and development/prospective partition differ")
        for structure_id in self.calibration_structure_ids:
            structure = structure_by_id.get(structure_id)
            if structure is None or structure.partition is not SplitPartition.CALIBRATION:
                raise ValueError('calibration roster must contain only calibration structures')

        observed_duplicate_groups: dict[str, tuple[str, ...]] = {}
        for structure_id in structure_by_id:
            child_actions = tuple(
                sorted(
                    action.action_id
                    for action in self.actions
                    if action.child_structure_id == structure_id
                )
            )
            if len(child_actions) > 1:
                observed_duplicate_groups[structure_id] = child_actions
        if dict(self.duplicate_action_child_groups) != observed_duplicate_groups:
            raise ValueError("duplicate child audit must exactly reproduce the action graph")


@dataclass(frozen=True, slots=True)
class SearchPolicySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/search-policy-spec'

    policy_id: str
    method_id: str
    implementation_id: str
    visible_input_ids: tuple[str, ...]
    forbidden_input_ids: tuple[str, ...]
    ordering_rule_id: str
    tie_breaker_id: str
    hold_rule_id: str

    def __post_init__(self) -> None:
        for field_name in (
            "policy_id",
            "method_id",
            "implementation_id",
            "ordering_rule_id",
            "tie_breaker_id",
            "hold_rule_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        _stable_values(self.visible_input_ids, field_name="visible_input_ids", allow_empty=False)
        _stable_values(
            self.forbidden_input_ids, field_name="forbidden_input_ids", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class TruthWorldLock(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/truth-world-lock'

    world_id: str
    case_kind_id: str
    public_graph_sha256: str
    privileged_truth_sha256: str
    expected_disposition_id: str
    selector_outcome_access: str

    def __post_init__(self) -> None:
        for field_name in (
            "world_id",
            "case_kind_id",
            "expected_disposition_id",
            "selector_outcome_access",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        validate_sha256(self.public_graph_sha256, field_name="public_graph_sha256")
        validate_sha256(self.privileged_truth_sha256, field_name="privileged_truth_sha256")


@dataclass(frozen=True, slots=True)
class ExplorationDesignFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/exploration-design-freeze'

    design_id: str
    eligible_action_graph_sha256: str
    policy_specs: tuple[SearchPolicySpec, ...]
    truth_worlds: tuple[TruthWorldLock, ...]
    wave_count: int
    actions_per_policy_wave: int
    policy_action_budget: int
    policy_compute_budget_units: int
    family_restart_quota: int
    bridge_action_quota_per_wave: int
    initial_history_rule_id: str
    family_diversity_rule_id: str
    cost_accounting_rule_id: str
    union_reuse_rule_id: str
    primary_estimand_ids: tuple[str, ...]
    advantage_rule_id: str
    advantage_confidence_level: Decimal
    advantage_minimum_relative_cost_reduction: Decimal
    independent_family_unit_count: int
    target_outcome_values_used: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        validate_sha256(self.eligible_action_graph_sha256)
        require_sorted_unique_ids(
            self.policy_specs,
            attribute="policy_id",
            field_name="policy_specs",
        )
        require_sorted_unique_ids(
            self.truth_worlds, attribute="world_id", field_name="truth_worlds"
        )
        for field_name in (
            "wave_count",
            "actions_per_policy_wave",
            "policy_action_budget",
            "policy_compute_budget_units",
            "family_restart_quota",
            "bridge_action_quota_per_wave",
            "independent_family_unit_count",
        ):
            _positive_int(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "initial_history_rule_id",
            "family_diversity_rule_id",
            "cost_accounting_rule_id",
            "union_reuse_rule_id",
            "advantage_rule_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        _stable_values(
            self.primary_estimand_ids, field_name="primary_estimand_ids", allow_empty=False
        )
        validate_decimal(
            self.advantage_confidence_level,
            field_name="advantage_confidence_level",
        )
        validate_decimal(
            self.advantage_minimum_relative_cost_reduction,
            field_name="advantage_minimum_relative_cost_reduction",
        )
        if not Decimal("0") < self.advantage_confidence_level < Decimal("1"):
            raise ValueError("advantage confidence level must lie in (0, 1)")
        if not Decimal("0") < self.advantage_minimum_relative_cost_reduction < Decimal("1"):
            raise ValueError("minimum relative cost reduction must lie in (0, 1)")
        if self.target_outcome_values_used:
            raise ValueError("target outcomes cannot construct the search design")
        if self.policy_action_budget != self.wave_count * self.actions_per_policy_wave:
            raise ValueError("policy action budget must equal wave_count times actions_per_wave")
        if len(self.policy_specs) != 3:
            raise ValueError('material source design requires response, scalar and stratified-random policies')
        if len(self.truth_worlds) != 6:
            raise ValueError('material source design requires all six predeclared truth-world cases')
        if self.independent_family_unit_count < 5:
            raise ValueError('material source design discovery comparison requires at least five family units')


@dataclass(frozen=True, slots=True)
class SolverViewFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/solver-view-freeze'

    view_id: str
    role_id: str
    solver_id: str
    solver_version: str
    functional: str
    pseudopotential_set_id: str
    k_mesh: tuple[int, int, int]
    q_mesh: tuple[int, int, int]
    fine_k_mesh: tuple[int, int, int]
    fine_q_mesh: tuple[int, int, int]
    smearing_Ry: Decimal
    precision: str
    model_closure_ids: tuple[str, ...]
    promotes_positive_claim: bool

    def __post_init__(self) -> None:
        for field_name in ("view_id", "role_id", "solver_id", "pseudopotential_set_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        validate_nonempty(self.solver_version, field_name="solver_version")
        validate_nonempty(self.functional, field_name="functional")
        validate_nonempty(self.precision, field_name="precision")
        for field_name in ("k_mesh", "q_mesh", "fine_k_mesh", "fine_q_mesh"):
            values = getattr(self, field_name)
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in values
            ):
                raise ValueError(f'{field_name} must contain positive integers')
        validate_decimal(self.smearing_Ry, field_name="smearing_Ry", minimum=Decimal("0"))
        _stable_values(self.model_closure_ids, field_name="model_closure_ids", allow_empty=False)


@dataclass(frozen=True, slots=True)
class CalibrationAlgorithmFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/calibration-algorithm-freeze'

    algorithm_id: str
    output_id: str
    control_structure_ids: tuple[str, ...]
    estimator_id: str
    rounding_rule_id: str
    output_unit: str
    unresolved_state: str
    target_outcomes_forbidden: bool

    def __post_init__(self) -> None:
        for field_name in (
            "algorithm_id",
            "output_id",
            "estimator_id",
            "rounding_rule_id",
            "unresolved_state",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        _stable_values(
            self.control_structure_ids,
            field_name="control_structure_ids",
            allow_empty=False,
        )
        validate_nonempty(self.output_unit, field_name="output_unit")
        if not self.target_outcomes_forbidden:
            raise ValueError('material source design calibration algorithms must forbid development atlas/sealed prospective outcomes')


@dataclass(frozen=True, slots=True)
class MaterialGateFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-gate-freeze'

    gate_id: str
    generic_gate_id: str
    required_operand_ids: tuple[str, ...]
    threshold_rule_id: str
    failure_reason_id: str
    support_required: bool

    def __post_init__(self) -> None:
        for field_name in (
            "gate_id",
            "generic_gate_id",
            "threshold_rule_id",
            "failure_reason_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        _stable_values(
            self.required_operand_ids, field_name="required_operand_ids", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class MethodFormalismFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/method-formalism-freeze'

    formalism_id: str
    source_ids: tuple[str, ...]
    applicability_ids: tuple[str, ...]
    equation_id: str
    promotion_role_id: str
    hard_rule_id: str
    diagnostic_rule_ids: tuple[str, ...]
    nonuniversal_parameter_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "formalism_id",
            "equation_id",
            "promotion_role_id",
            "hard_rule_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        _stable_values(self.source_ids, field_name="source_ids", allow_empty=False)
        _stable_values(self.applicability_ids, field_name="applicability_ids", allow_empty=False)
        _stable_values(self.diagnostic_rule_ids, field_name="diagnostic_rule_ids")
        _stable_values(self.nonuniversal_parameter_ids, field_name="nonuniversal_parameter_ids")


@dataclass(frozen=True, slots=True)
class ScienceDesignFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/science-design-freeze'

    design_id: str
    operating_temperature_K: Decimal
    operating_pressure_Pa: Decimal
    pressure_tolerance_Pa: Decimal
    probe_field_T: Decimal
    probe_field_fraction_of_hc1_upper: Decimal
    slab_thickness_m: Decimal
    required_ambient_lifetime_s: Decimal
    maximum_process_pressure_Pa: Decimal
    maximum_process_temperature_K: Decimal
    maximum_process_duration_s: Decimal
    mechanism_lane_id: str
    physical_independent_unit_id: str
    solver_views: tuple[SolverViewFreeze, ...]
    calibration_algorithms: tuple[CalibrationAlgorithmFreeze, ...]
    material_gates: tuple[MaterialGateFreeze, ...]
    method_formalisms: tuple[MethodFormalismFreeze, ...]
    compatibility_map_ids: tuple[str, ...]
    receiver_level_ids: tuple[str, ...]
    receiver_survivor_limit: int
    refinement_rule_ids: tuple[str, ...]
    stop_rule_ids: tuple[str, ...]
    only_calibration_derived_ids: tuple[str, ...]
    target_outcome_values_used: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        for field_name in (
            "operating_temperature_K",
            "operating_pressure_Pa",
            "pressure_tolerance_Pa",
            "probe_field_T",
            "probe_field_fraction_of_hc1_upper",
            "slab_thickness_m",
            "required_ambient_lifetime_s",
            "maximum_process_pressure_Pa",
            "maximum_process_temperature_K",
            "maximum_process_duration_s",
        ):
            validate_decimal(getattr(self, field_name), field_name=field_name, minimum=Decimal("0"))
        if self.operating_temperature_K != Decimal("300"):
            raise ValueError('ambient pressure superconductor operating temperature must remain exactly 300 K')
        if self.operating_pressure_Pa != Decimal("101325"):
            raise ValueError('ambient pressure superconductor ambient pressure convention must remain 101325 Pa')
        if not Decimal("0") < self.probe_field_fraction_of_hc1_upper <= Decimal("0.1"):
            raise ValueError("probe field must remain at most one tenth of conservative Hc1")
        for field_name in ("mechanism_lane_id", "physical_independent_unit_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        require_sorted_unique_ids(self.solver_views, attribute="view_id", field_name="solver_views")
        require_sorted_unique_ids(
            self.calibration_algorithms,
            attribute="algorithm_id",
            field_name="calibration_algorithms",
        )
        require_sorted_unique_ids(
            self.material_gates,
            attribute="gate_id",
            field_name="material_gates",
        )
        require_sorted_unique_ids(
            self.method_formalisms,
            attribute="formalism_id",
            field_name="method_formalisms",
        )
        _stable_values(
            self.compatibility_map_ids, field_name="compatibility_map_ids", allow_empty=False
        )
        _stable_values(self.receiver_level_ids, field_name="receiver_level_ids", allow_empty=False)
        _positive_int(self.receiver_survivor_limit, field_name="receiver_survivor_limit")
        _stable_values(
            self.refinement_rule_ids, field_name="refinement_rule_ids", allow_empty=False
        )
        _stable_values(self.stop_rule_ids, field_name="stop_rule_ids", allow_empty=False)
        _stable_values(
            self.only_calibration_derived_ids,
            field_name="only_calibration_derived_ids",
            allow_empty=False,
        )
        if self.target_outcome_values_used:
            raise ValueError('development atlas/sealed prospective outcomes cannot freeze the science design')
        calibration_outputs = tuple(
            sorted(algorithm.output_id for algorithm in self.calibration_algorithms)
        )
        if calibration_outputs != self.only_calibration_derived_ids:
            raise ValueError('the only open material source design fields must exactly equal calibration outputs')
        required_material_gate_ids = {
            f'gate.material.{name}'
            for name in (
                "authority",
                "dynamic-sink",
                "effort",
                "identity",
                "material-constraints",
                "meissner-receiver",
                "metallic-state",
                "order-at-300k",
                "preservation",
                "support",
                "synthesis-reachability",
                "target-tc",
                "thermodynamic-sink",
                "transverse-response",
                "uncertainty",
                "validity",
            )
        }
        if {gate.gate_id for gate in self.material_gates} != required_material_gate_ids:
            raise ValueError('material source design science design must preserve all 16 material gates')
        allowed_generic_gate_ids = {
            "generic.atlas-native",
            "generic.authority",
            "generic.baseline-preservation",
            "generic.dynamics",
            "generic.effort",
            "generic.observation-validity",
            "generic.physical-sink",
            "generic.reachability",
            "generic.target",
            "generic.uncertainty",
        }
        if {gate.generic_gate_id for gate in self.material_gates} - allowed_generic_gate_ids:
            raise ValueError(
                "material gates may lower only to the nine generic gates plus atlas support"
            )
        if self.receiver_level_ids != tuple(sorted((
            'receiver.counterfactual',
            'receiver.material-linked',
            'receiver.gauge-covariant',
        ))):
            raise ValueError('material source design receiver hierarchy must preserve counterfactual receiver, material linked receiver and gauge covariant response')


@dataclass(frozen=True, slots=True)
class MaterialSourceDesignDesignBasisFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-source-design-design-basis-freeze'

    freeze_id: str
    config_sha256: str
    source_qualification_sha256: str
    material_roster_sha256: str
    exploration_design_sha256: str
    science_design_sha256: str
    development_protocol_id: str
    development_protocol_sha256: str
    development_registry_sha256: str
    target_contact_count: int
    only_open_field_ids: tuple[str, ...]
    frozen_component_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        for field_name in (
            "config_sha256",
            "source_qualification_sha256",
            "material_roster_sha256",
            "exploration_design_sha256",
            "science_design_sha256",
            "development_protocol_sha256",
            "development_registry_sha256",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        validate_stable_id(self.development_protocol_id, field_name="development_protocol_id")
        _nonnegative_int(self.target_contact_count, field_name="target_contact_count")
        if self.target_contact_count != 0:
            raise ValueError('material source design design freeze cannot contact development atlas or sealed prospective target outcomes')
        _stable_values(
            self.only_open_field_ids, field_name="only_open_field_ids", allow_empty=False
        )
        _stable_values(
            self.frozen_component_ids, field_name="frozen_component_ids", allow_empty=False
        )
        _stable_values(self.reason_codes, field_name="reason_codes", allow_empty=False)


@dataclass(frozen=True, slots=True)
class MaterialSourceDesignDesignBasisAudit(CanonicalRecord):
    'Outcome-blind material source design assessment when the candidate basis cannot be frozen.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-source-design-design-basis-audit'

    audit_id: str
    config_sha256: str
    source_qualification_sha256: str
    material_roster_sha256: str
    exploration_design_sha256: str
    science_design_sha256: str
    intended_development_protocol_id: str
    target_contact_count: int
    source_qualified: bool
    candidate_components_closed: bool
    full_development_template_present: bool
    exact_provider_set_present: bool
    missing_capability_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        for field_name in (
            "config_sha256",
            "source_qualification_sha256",
            "material_roster_sha256",
            "exploration_design_sha256",
            "science_design_sha256",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        validate_stable_id(
            self.intended_development_protocol_id,
            field_name="intended_development_protocol_id",
        )
        _nonnegative_int(self.target_contact_count, field_name="target_contact_count")
        if self.target_contact_count != 0:
            raise ValueError('material source design design audit cannot contact target outcomes')
        _stable_values(
            self.missing_capability_ids,
            field_name="missing_capability_ids",
            allow_empty=False,
        )
        _stable_values(self.reason_codes, field_name="reason_codes", allow_empty=False)
        if self.full_development_template_present and self.exact_provider_set_present:
            raise ValueError('a complete material source design basis must use AP3DesignBasisFreeze, not an audit')


@dataclass(frozen=True, slots=True)
class MaterialSourceDesignResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/ambient-pressure-superconductor/material-source-design-result'

    result_id: str
    config_sha256: str
    source_qualification_sha256: str
    design_basis_record_sha256: str
    disposition: MaterialSourceDesignDisposition
    source_qualified: bool
    design_basis_frozen: bool
    target_contact_count: int
    calibration_structure_count: int
    development_family_count: int
    development_initial_action_count: int
    development_eligible_wave_action_count: int
    prospective_family_count: int
    prospective_action_count: int
    truth_world_count: int
    policy_count: int
    calibration_derived_field_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        for field_name in (
            "config_sha256",
            "source_qualification_sha256",
            "design_basis_record_sha256",
        ):
            validate_sha256(getattr(self, field_name), field_name=field_name)
        for field_name in (
            "target_contact_count",
            'calibration_structure_count',
            'development_family_count',
            'development_initial_action_count',
            'development_eligible_wave_action_count',
            'prospective_family_count',
            'prospective_action_count',
            "truth_world_count",
            "policy_count",
            "calibration_derived_field_count",
        ):
            _nonnegative_int(getattr(self, field_name), field_name=field_name)
        if self.target_contact_count != 0:
            raise ValueError('material source design result cannot include target contact')
        if self.disposition is MaterialSourceDesignDisposition.PASS and not (
            self.source_qualified and self.design_basis_frozen
        ):
            raise ValueError('material source design pass requires source qualification and design freeze')
        _stable_values(self.reason_codes, field_name="reason_codes", allow_empty=False)


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f'{field_name} must be an object with string keys')
    return value


def _keys(value: Mapping[str, object], expected: tuple[str, ...], *, field_name: str) -> None:
    if set(value) != set(expected):
        raise ValueError(f'{field_name} keys differ from the closed material source design schema')


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f'{field_name} must be nonempty trimmed text')
    return value


def _integer(value: object, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{field_name} must be an integer >= {minimum}')
    return value


def _decimal(value: object, *, field_name: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f'{field_name} must be a decimal string')
    try:
        result = Decimal(value)
    except Exception as error:
        raise ValueError(f'{field_name} must be a decimal string') from error
    validate_decimal(result, field_name=field_name)
    return result


def _sha(value: object, *, field_name: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    result = _text(value, field_name=field_name)
    validate_sha256(result, field_name=field_name)
    return result


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f'{field_name} must be a string array')
    result = tuple(value)
    require_sorted_unique_strings(result, field_name=field_name, allow_empty=False)
    return result


def _reject_binary_floats(value: object, *, field_name: str = "config") -> None:
    if isinstance(value, float):
        raise ValueError(f'{field_name} contains a binary float')
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_binary_floats(item, field_name=f'{field_name}.{key}')
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_binary_floats(item, field_name=f'{field_name}[{index}]')


def decode_material_source_design_config(document: Mapping[str, object], *, payload_sha256: str) -> MaterialSourceDesignConfig:
    _reject_binary_floats(document)
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
    if document["schema"] != MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA or document["version"] != MATERIAL_SOURCE_DESIGN_CONFIG_VERSION:
        raise ValueError('unsupported material source design config schema or version')
    if document["plan_id"] != 'ambient-pressure-superconductor-response-study':
        raise ValueError('material source design config plan identity differs')
    if document["campaign_id"] != MATERIAL_SOURCE_DESIGN_CAMPAIGN_ID:
        raise ValueError('material source design campaign identity differs')

    lifecycle = _mapping(document["lifecycle"], field_name="lifecycle")
    _keys(
        lifecycle,
        ("freeze_id", "frozen_at_utc", "revision", "status"),
        field_name="lifecycle",
    )
    try:
        status = MaterialSourceDesignConfigLifecycle(_text(lifecycle["status"], field_name="lifecycle.status"))
    except ValueError as error:
        raise ValueError('unsupported material source design lifecycle') from error
    freeze_id = lifecycle["freeze_id"]
    frozen_at_utc = lifecycle["frozen_at_utc"]
    if freeze_id is not None and not isinstance(freeze_id, str):
        raise ValueError("lifecycle.freeze_id must be text or null")
    if frozen_at_utc is not None and not isinstance(frozen_at_utc, str):
        raise ValueError("lifecycle.frozen_at_utc must be text or null")
    if status is MaterialSourceDesignConfigLifecycle.FROZEN and (not freeze_id or not frozen_at_utc):
        raise ValueError('frozen material source design config requires freeze identity and timestamp')
    if status is MaterialSourceDesignConfigLifecycle.DRAFT and (freeze_id is not None or frozen_at_utc is not None):
        raise ValueError('draft material source design config cannot carry freeze metadata')

    design = _mapping(document["design"], field_name="design")
    _keys(
        design,
        (
            "development_protocol_id",
            "exploration_design_sha256",
            "material_roster_sha256",
            "science_design_sha256",
            "source_qualification_sha256",
        ),
        field_name="design",
    )
    allow_empty = status is MaterialSourceDesignConfigLifecycle.DRAFT

    search = _mapping(document["search"], field_name="search")
    _keys(
        search,
        (
            "actions_per_policy_wave",
            "bridge_action_quota_per_wave",
            "deterministic_seed",
            "family_restart_quota",
            "high_fidelity_survivor_limit",
            "initial_development_action_count",
            "policy_action_budget",
            "policy_compute_budget_units",
            "policy_ids",
            "wave_count",
        ),
        field_name="search",
    )

    operating = _mapping(document["operating_domain"], field_name="operating_domain")
    _keys(
        operating,
        (
            "maximum_process_duration_s",
            "maximum_process_pressure_Pa",
            "maximum_process_temperature_K",
            "operating_pressure_Pa",
            "operating_temperature_K",
            "probe_field_T",
            "required_ambient_lifetime_s",
            "slab_thickness_m",
        ),
        field_name="operating_domain",
    )

    storage = _mapping(document["storage"], field_name="storage")
    _keys(
        storage,
        ("external_root", "minimum_free_bytes", "storage_root"),
        field_name="storage",
    )
    if storage["storage_root"] != 'storage.semi-os-external':
        raise ValueError('material source design storage root identity differs')
    if storage["external_root"] != MATERIAL_SOURCE_DESIGN_EXTERNAL_ROOT:
        raise ValueError('material source design external root differs')

    resources = _mapping(document["resources"], field_name="resources")
    expected_resource_keys = (
        "cpu_cores",
        "gpu_devices",
        "memory_bytes",
        "network_required",
        "output_bytes",
        "scratch_bytes",
        "wall_time_seconds",
    )
    _keys(resources, expected_resource_keys, field_name="resources")
    resource_values: list[tuple[str, int | bool]] = []
    for key in expected_resource_keys:
        value = resources[key]
        if key == "network_required":
            if not isinstance(value, bool) or value:
                raise ValueError('material source design execution must be network-disabled')
        else:
            value = _integer(value, field_name=f'resources.{key}', minimum=0)
        resource_values.append((key, value))

    authority = _mapping(document["authority"], field_name="authority")
    _keys(authority, ("allowed_actions", "owner_id"), field_name="authority")
    if authority["owner_id"] != "human.project-owner":
        raise ValueError('material source design authority owner differs')
    allowed_actions = _string_tuple(authority["allowed_actions"], field_name="allowed_actions")

    result = MaterialSourceDesignConfig(
        payload_sha256=payload_sha256,
        lifecycle=status,
        campaign_id=MATERIAL_SOURCE_DESIGN_CAMPAIGN_ID,
        revision=_text(lifecycle["revision"], field_name="lifecycle.revision"),
        freeze_id=freeze_id,
        frozen_at_utc=frozen_at_utc,
        source_qualification_sha256=_sha(
            design["source_qualification_sha256"],
            field_name="design.source_qualification_sha256",
            allow_empty=allow_empty,
        ),
        material_roster_sha256=_sha(
            design["material_roster_sha256"],
            field_name="design.material_roster_sha256",
            allow_empty=allow_empty,
        ),
        exploration_design_sha256=_sha(
            design["exploration_design_sha256"],
            field_name="design.exploration_design_sha256",
            allow_empty=allow_empty,
        ),
        science_design_sha256=_sha(
            design["science_design_sha256"],
            field_name="design.science_design_sha256",
            allow_empty=allow_empty,
        ),
        development_protocol_id=_text(
            design["development_protocol_id"],
            field_name="design.development_protocol_id",
        ),
        wave_count=_integer(search["wave_count"], field_name="search.wave_count", minimum=1),
        actions_per_policy_wave=_integer(
            search["actions_per_policy_wave"],
            field_name="search.actions_per_policy_wave",
            minimum=1,
        ),
        initial_development_action_count=_integer(
            search["initial_development_action_count"],
            field_name="search.initial_development_action_count",
            minimum=5,
        ),
        policy_action_budget=_integer(
            search["policy_action_budget"],
            field_name="search.policy_action_budget",
            minimum=1,
        ),
        policy_compute_budget_units=_integer(
            search["policy_compute_budget_units"],
            field_name="search.policy_compute_budget_units",
            minimum=1,
        ),
        high_fidelity_survivor_limit=_integer(
            search["high_fidelity_survivor_limit"],
            field_name="search.high_fidelity_survivor_limit",
            minimum=1,
        ),
        family_restart_quota=_integer(
            search["family_restart_quota"],
            field_name="search.family_restart_quota",
            minimum=1,
        ),
        bridge_action_quota_per_wave=_integer(
            search["bridge_action_quota_per_wave"],
            field_name="search.bridge_action_quota_per_wave",
            minimum=1,
        ),
        policy_ids=_string_tuple(search["policy_ids"], field_name="search.policy_ids"),
        deterministic_seed=_text(
            search["deterministic_seed"],
            field_name="search.deterministic_seed",
        ),
        operating_temperature_K=_decimal(
            operating["operating_temperature_K"],
            field_name="operating_domain.operating_temperature_K",
        ),
        operating_pressure_Pa=_decimal(
            operating["operating_pressure_Pa"],
            field_name="operating_domain.operating_pressure_Pa",
        ),
        probe_field_T=_decimal(
            operating["probe_field_T"],
            field_name="operating_domain.probe_field_T",
        ),
        slab_thickness_m=_decimal(
            operating["slab_thickness_m"],
            field_name="operating_domain.slab_thickness_m",
        ),
        required_ambient_lifetime_s=_decimal(
            operating["required_ambient_lifetime_s"],
            field_name="operating_domain.required_ambient_lifetime_s",
        ),
        maximum_process_pressure_Pa=_decimal(
            operating["maximum_process_pressure_Pa"],
            field_name="operating_domain.maximum_process_pressure_Pa",
        ),
        maximum_process_temperature_K=_decimal(
            operating["maximum_process_temperature_K"],
            field_name="operating_domain.maximum_process_temperature_K",
        ),
        maximum_process_duration_s=_decimal(
            operating["maximum_process_duration_s"],
            field_name="operating_domain.maximum_process_duration_s",
        ),
        storage_root=str(storage["storage_root"]),
        external_root=str(storage["external_root"]),
        minimum_free_bytes=_integer(
            storage["minimum_free_bytes"],
            field_name="storage.minimum_free_bytes",
            minimum=1,
        ),
        resources=tuple(resource_values),
        allowed_authority_actions=allowed_actions,
    )
    if result.operating_temperature_K != Decimal("300"):
        raise ValueError('material source design operating temperature differs from 300 K')
    if result.operating_pressure_Pa != Decimal("101325"):
        raise ValueError('material source design ambient pressure convention differs')
    if not 5 <= result.initial_development_action_count <= 20:
        raise ValueError('material source design initial development action count must lie in [5, 20]')
    if len(result.policy_ids) != 3:
        raise ValueError('material source design requires exactly the three qualified policies')
    if result.policy_action_budget != result.wave_count * result.actions_per_policy_wave:
        raise ValueError('material source design policy action budget must equal W times K')
    return result


def decode_material_source_design_config_bytes(payload: bytes) -> MaterialSourceDesignConfig:
    if len(payload) > MATERIAL_SOURCE_DESIGN_MAXIMUM_CONFIG_BYTES:
        raise ValueError('material source design config exceeds its byte limit')
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError('material source design config is not valid UTF-8 JSON') from error
    if not isinstance(document, dict):
        raise ValueError('material source design config root must be an object')
    return decode_material_source_design_config(document, payload_sha256=sha256(payload).hexdigest())


def load_material_source_design_config(path: Path) -> tuple[MaterialSourceDesignConfig, bytes]:
    observed = path.lstat()
    if path.is_symlink() or not path.is_file() or observed.st_size > MATERIAL_SOURCE_DESIGN_MAXIMUM_CONFIG_BYTES:
        raise ValueError('material source design config must be a bounded regular non-symlink file')
    payload = path.read_bytes()
    return decode_material_source_design_config_bytes(payload), payload


__all__ = [
    "MATERIAL_SOURCE_DESIGN_ADJUDICATION_SCHEMA",
    "MATERIAL_SOURCE_DESIGN_CAMPAIGN_ID",
    "MATERIAL_SOURCE_DESIGN_CAPABILITY_VERSION",
    "MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA",
    "MATERIAL_SOURCE_DESIGN_CONFIG_VERSION",
    "MATERIAL_SOURCE_DESIGN_DESIGN_CAPABILITY_KEY",
    "MATERIAL_SOURCE_DESIGN_EVALUATOR_CAPABILITY_KEY",
    "MATERIAL_SOURCE_DESIGN_EXTERNAL_ROOT",
    "MATERIAL_SOURCE_DESIGN_MAXIMUM_CONFIG_BYTES",
    "MATERIAL_SOURCE_DESIGN_MAXIMUM_OUTPUT_BYTES",
    "MATERIAL_SOURCE_DESIGN_SOURCE_CAPABILITY_KEY",
    "MATERIAL_SOURCE_DESIGN_SOURCE_PLAN_ID",
    'MaterialSourceDesignConfig',
    'MaterialSourceDesignConfigLifecycle',
    'MaterialSourceDesignDesignBasisFreeze',
    'MaterialSourceDesignDesignBasisAudit',
    'MaterialSourceDesignDisposition',
    'MaterialSourceDesignResult',
    'MaterialSourceDesignSourceQualification',
    "ActionStage",
    "CalibrationAlgorithmFreeze",
    "ExplorationDesignFreeze",
    "FormationRouteSpec",
    "MaterialActionSpec",
    "MaterialFamilySpec",
    "MaterialGateDisposition",
    "MaterialGateFreeze",
    "MaterialRosterFreeze",
    "MaterialSiteSpec",
    "MaterialStructureSpec",
    "MethodFormalismFreeze",
    "PseudopotentialLock",
    "RouteEvidenceStatus",
    "ScienceDesignFreeze",
    "SearchPolicySpec",
    "SolverViewFreeze",
    "SourceAssetLock",
    "SourceMemberLock",
    "SourceOutcomeRole",
    "SplitPartition",
    "TruthWorldLock",
    'decode_material_source_design_config',
    'decode_material_source_design_config_bytes',
    'load_material_source_design_config',
]
