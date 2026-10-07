"""Outcome-blind mapped-state, TORAX ensemble and policy bindings."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_schema,
    validate_stable_id,
)


class MappedStateDisposition(StrEnum):
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNEVALUABLE = "UNEVALUABLE"


class MappedFieldOriginKind(StrEnum):
    OBSERVED = "OBSERVED"
    DERIVED = "DERIVED"
    ASSUMED = "ASSUMED"
    MARGINALIZED = "MARGINALIZED"


class MappedActionWord(StrEnum):
    TORAX_DOWN = "TORAX_DOWN"
    TORAX_HOLD = "TORAX_HOLD"
    TORAX_UP = "TORAX_UP"


class MappedNumericalViewKind(StrEnum):
    PRIMARY = "PRIMARY"
    CHALLENGER = "CHALLENGER"


@dataclass(frozen=True, slots=True)
class MappedFieldOrigin(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-field-origin'

    field_binding_id: str
    archive_field_id: str
    torax_field_id: str | None
    origin: MappedFieldOriginKind
    archive_unit: str
    torax_unit: str | None
    unit_transform_id: str
    lower_bound: Decimal | None
    upper_bound: Decimal | None
    acceptance_predicate_id: str
    archive_outcome_used: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("field_binding_id", self.field_binding_id),
            ("archive_field_id", self.archive_field_id),
            ("unit_transform_id", self.unit_transform_id),
            ("acceptance_predicate_id", self.acceptance_predicate_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.archive_unit, field_name="archive_unit")
        if self.archive_outcome_used:
            raise ValueError("mapped field origin cannot use an archive receiver outcome")
        if self.origin is MappedFieldOriginKind.MARGINALIZED:
            if any(
                value is not None
                for value in (
                    self.torax_field_id,
                    self.torax_unit,
                    self.lower_bound,
                    self.upper_bound,
                )
            ):
                raise ValueError("marginalized mapping field transports a value")
            return
        if self.torax_field_id is None or self.torax_unit is None:
            raise ValueError("mapped field lacks its TORAX coordinate and unit")
        validate_stable_id(self.torax_field_id, field_name="torax_field_id")
        validate_nonempty(self.torax_unit, field_name="torax_unit")
        if self.origin is MappedFieldOriginKind.ASSUMED:
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("assumed TORAX field lacks a bounded native interval")
            validate_decimal(self.lower_bound, field_name="lower_bound")
            validate_decimal(self.upper_bound, field_name="upper_bound")
            if self.lower_bound >= self.upper_bound:
                raise ValueError("assumed TORAX field interval is empty")


@dataclass(frozen=True, slots=True)
class MappedToraxAssumptionMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-torax-assumption-member'

    member_id: str
    quantile: Decimal
    assumed_field_binding_ids: tuple[str, ...]
    joint_vector_required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        require_sorted_unique_strings(
            self.assumed_field_binding_ids,
            field_name="assumed_field_binding_ids",
            allow_empty=False,
        )
        if self.quantile not in {Decimal("0.20"), Decimal("0.50"), Decimal("0.80")}:
            raise ValueError("mapped assumption member is outside the frozen quantiles")
        if not self.joint_vector_required:
            raise ValueError("mapped assumption coordinates cannot be mixed memberwise")


@dataclass(frozen=True, slots=True)
class MappedToraxNumericalView(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-torax-numerical-view'

    view_id: str
    kind: MappedNumericalViewKind
    radial_cell_count: int
    timestep_seconds: Decimal
    corrector_depth: int
    selection_rule_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.view_id, field_name="view_id")
        validate_stable_id(self.selection_rule_id, field_name="selection_rule_id")
        if self.radial_cell_count < 2 or self.timestep_seconds <= 0 or self.corrector_depth < 0:
            raise ValueError("mapped numerical view contains an invalid native coordinate")


@dataclass(frozen=True, slots=True)
class MappedToraxEnsembleSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-torax-ensemble-spec'

    ensemble_id: str
    field_origins: tuple[MappedFieldOrigin, ...]
    assumption_members: tuple[MappedToraxAssumptionMember, ...]
    numerical_views: tuple[MappedToraxNumericalView, ...]
    complete_cartesian_product_required: bool
    archive_law_is_member: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.ensemble_id, field_name="ensemble_id")
        require_sorted_unique_ids(
            self.field_origins,
            attribute="field_binding_id",
            field_name="field_origins",
        )
        require_sorted_unique_ids(
            self.assumption_members,
            attribute="member_id",
            field_name="assumption_members",
        )
        require_sorted_unique_ids(
            self.numerical_views,
            attribute="view_id",
            field_name="numerical_views",
        )
        if (
            {value.quantile for value in self.assumption_members}
            != {Decimal("0.20"), Decimal("0.50"), Decimal("0.80")}
            or len(self.assumption_members) != 3
            or {value.kind for value in self.numerical_views}
            != {MappedNumericalViewKind.PRIMARY, MappedNumericalViewKind.CHALLENGER}
            or len(self.numerical_views) != 2
            or not self.complete_cartesian_product_required
            or self.archive_law_is_member
        ):
            raise ValueError("mapped ensemble is not the complete three-by-two product")

    @property
    def member_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                f"{assumption.member_id}.{view.view_id}"
                for assumption in self.assumption_members
                for view in self.numerical_views
            )
        )


@dataclass(frozen=True, slots=True)
class MappedActionChart(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-action-chart'

    chart_id: str
    actions: tuple[MappedActionWord, ...]
    active_fraction_of_available_range: Decimal
    ramp_duration_ms: Decimal
    receiver_horizon_ms: Decimal
    delivery_relative_tolerance: Decimal
    ordinal_transport_only: bool
    archive_threshold_transport_forbidden: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.chart_id, field_name="chart_id")
        if (
            self.actions != tuple(MappedActionWord)
            or self.active_fraction_of_available_range != Decimal("0.20")
            or self.ramp_duration_ms != Decimal("2")
            or self.receiver_horizon_ms != Decimal("40")
            or self.delivery_relative_tolerance != Decimal("1e-9")
            or not self.ordinal_transport_only
            or not self.archive_threshold_transport_forbidden
        ):
            raise ValueError("mapped TORAX action chart differs from Section 7.7")


@dataclass(frozen=True, slots=True)
class MappedReceiverSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-receiver-spec'

    receiver_spec_id: str
    quantity: str
    radial_coordinate: str
    radial_min: Decimal
    radial_max: Decimal
    weighting: str
    minimum_valid_cells: int
    response_definition: str
    native_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.receiver_spec_id, field_name="receiver_spec_id")
        if (
            self.quantity != "electron-temperature"
            or self.radial_coordinate != "normalized-toroidal-flux-rho"
            or self.radial_min != Decimal("0")
            or self.radial_max != Decimal("0.20")
            or self.weighting != "GEOMETRY_CELL_VOLUME"
            or self.minimum_valid_cells != 2
            or self.response_definition != "T40MS_MINUS_PREACTION"
            or self.native_unit != "eV"
        ):
            raise ValueError("mapped TORAX receiver differs")


@dataclass(frozen=True, slots=True)
class MappedPreparationPlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-preparation-plan'

    plan_id: str
    mapped_state_id: str
    nominal_preparation_id: str
    confirmatory_preparation_ids: tuple[str, ...]
    uncertainty_polytope: ObjectIdentity
    metric: ObjectIdentity
    selection_rule: str
    physical_independent_unit_count: int
    required_realized_distinctness: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("plan_id", self.plan_id),
            ("mapped_state_id", self.mapped_state_id),
            ("nominal_preparation_id", self.nominal_preparation_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.confirmatory_preparation_ids,
            field_name="confirmatory_preparation_ids",
            allow_empty=False,
        )
        if (
            len(self.confirmatory_preparation_ids) != 3
            or self.nominal_preparation_id in self.confirmatory_preparation_ids
            or self.selection_rule
            != "FARTHEST_POINT_UNCERTAINTY_VERTICES_AND_EXTREMA_EXCLUDING_NOMINAL"
            or self.physical_independent_unit_count != 1
            or self.required_realized_distinctness != 3
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("mapped preparation plan differs from the one-plus-three design")


@dataclass(frozen=True, slots=True)
class MappedMethodBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-method-binding'

    binding_id: str
    arm_id: str
    method_family_id: str
    method_config_schema: str
    method_config: ObjectIdentity
    shared_capability_ids: tuple[str, ...]
    fit_role: str
    contact_role: str
    protected_refit_allowed: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("arm_id", self.arm_id),
            ("method_family_id", self.method_family_id),
            ("fit_role", self.fit_role),
            ("contact_role", self.contact_role),
        ):
            validate_stable_id(value, field_name=name)
        validate_schema(self.method_config_schema)
        require_sorted_unique_strings(
            self.shared_capability_ids,
            field_name="shared_capability_ids",
            allow_empty=False,
        )
        if (
            self.method_config.object_schema != self.method_config_schema
            or self.fit_role != "mapped-development-fit"
            or self.contact_role != "mapped-development-contact"
            or self.protected_refit_allowed
        ):
            raise ValueError("mapped method does not preserve 16-fit/8-contact development separation")


@dataclass(frozen=True, slots=True)
class MappedChildScientificConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-child-scientific-config'

    config_id: str
    ensemble: MappedToraxEnsembleSpec
    action_chart: MappedActionChart
    receiver: MappedReceiverSpec
    method_bindings: tuple[MappedMethodBinding, ...]
    mapped_development_count: int
    mapped_development_fit_count: int
    mapped_development_contact_count: int
    mapped_prospective_attempt_count: int
    mapped_prospective_capacity_options: tuple[int, ...]
    nominal_reference_episode_count: int
    maximum_confirmatory_episode_count: int
    protected_adaptive_acquisition_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_ids(
            self.method_bindings,
            attribute="binding_id",
            field_name="method_bindings",
        )
        if (
            len(self.method_bindings) != 4
            or {value.arm_id for value in self.method_bindings}
            != {"arm-a", "arm-b", "arm-c", "arm-d"}
            or self.mapped_development_count != 24
            or self.mapped_development_fit_count != 16
            or self.mapped_development_contact_count != 8
            or self.mapped_prospective_attempt_count != 45
            or self.mapped_prospective_capacity_options != (24, 30)
            or self.nominal_reference_episode_count != 18
            or self.maximum_confirmatory_episode_count != 54
            or self.protected_adaptive_acquisition_allowed
        ):
            raise ValueError("mapped child design differs from 24 development states/45 prospective attempts and 18+54 controller use")


def validate_mapped_member_product(
    *,
    ensemble: MappedToraxEnsembleSpec,
    observed_member_ids: tuple[str, ...],
) -> None:
    require_sorted_unique_strings(
        observed_member_ids,
        field_name="observed_member_ids",
        allow_empty=False,
    )
    if observed_member_ids != ensemble.member_ids:
        raise ValueError("MAPPED_MODEL_ENSEMBLE_INCOMPLETE")


__all__ = [
    'MappedActionChart',
    "MappedActionWord",
    'MappedChildScientificConfig',
    "MappedFieldOriginKind",
    'MappedFieldOrigin',
    'MappedMethodBinding',
    "MappedNumericalViewKind",
    'MappedPreparationPlan',
    'MappedReceiverSpec',
    "MappedStateDisposition",
    'MappedToraxAssumptionMember',
    'MappedToraxEnsembleSpec',
    'MappedToraxNumericalView',
    "validate_mapped_member_product",
]
