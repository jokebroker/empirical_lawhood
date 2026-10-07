"""Relational open-system, component and interface contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .authority import AuthorityAction, AuthorityPolicy
from .evidence import ClaimSpec
from .quantities import QuantityKind, QuantitySpec
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)
from .status import ReadinessStatus
from .time import (
    CausalPhase,
    ClockRelationKind,
    ClockRelationSpec,
    ClockSpec,
    HorizonSpec,
    InformationCutoff,
)
from .worlds import (
    ComputabilityEnvelope,
    EvidenceUnitScope,
    NumericalViewSpec,
    WorldKind,
    WorldSpec,
    validate_claim_in_world,
)


class SystemBoundaryKind(StrEnum):
    OPEN_DRIVEN_DISSIPATIVE = "OPEN_DRIVEN_DISSIPATIVE"
    CLOSED_REFERENCE = "CLOSED_REFERENCE"
    HYBRID = "HYBRID"


class PortDirection(StrEnum):
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"


class BalanceRole(StrEnum):
    MASS = "MASS"
    ENERGY = "ENERGY"
    MOMENTUM = "MOMENTUM"
    CHARGE = "CHARGE"
    INFORMATION = "INFORMATION"
    COMMAND = "COMMAND"
    OBSERVATION = "OBSERVATION"
    NONE = "NONE"


class BalanceSemantics(StrEnum):
    CONSERVED = "CONSERVED"
    DISSIPATIVE = "DISSIPATIVE"
    INFORMATIONAL = "INFORMATIONAL"
    COMMAND = "COMMAND"


@dataclass(frozen=True, slots=True)
class RelationalIdentity(CanonicalRecord):
    """The fixed L(D,H,A,R,tau) identity of one candidate law."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/relational-identity'

    relation_id: str
    denominator_quantity_ids: tuple[str, ...]
    history_quantity_ids: tuple[str, ...]
    memoryless: bool
    action_quantity_ids: tuple[str, ...]
    receiver_quantity_ids: tuple[str, ...]
    horizon: HorizonSpec
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.relation_id, field_name="relation_id")
        for field_name, values, allow_empty in (
            ("denominator_quantity_ids", self.denominator_quantity_ids, False),
            ("history_quantity_ids", self.history_quantity_ids, True),
            ("action_quantity_ids", self.action_quantity_ids, True),
            ("receiver_quantity_ids", self.receiver_quantity_ids, False),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=allow_empty)
        if self.memoryless and self.history_quantity_ids:
            raise ValueError("a memoryless relation cannot retain history quantities")
        if not self.memoryless and not self.history_quantity_ids:
            raise ValueError("a history-dependent relation must name retained history")
        require_extensions(self.extensions)


@dataclass(frozen=True, slots=True)
class IndependentUnitSpec(CanonicalRecord):
    """Physical preparation/acquisition unit for splits and uncertainty."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/independent-unit-spec'

    unit_id: str
    label: str
    grouping_key: str
    scope: EvidenceUnitScope = EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        validate_stable_id(self.grouping_key, field_name="grouping_key")
        validate_nonempty(self.label, field_name="label")
        if self.scope is not EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT:
            raise ValueError("independent units must have physical evidence scope")


@dataclass(frozen=True, slots=True)
class PortSpec(CanonicalRecord):
    """One typed component/system boundary port."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/port-spec'

    port_id: str
    quantity_id: str
    clock_id: str
    direction: PortDirection
    balance_role: BalanceRole
    authority_action: AuthorityAction | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.port_id, field_name="port_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_stable_id(self.clock_id, field_name="clock_id")


@dataclass(frozen=True, slots=True)
class PortRef(CanonicalRecord):
    """Reference to a port owned by the system or one component."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/port-ref'

    owner_id: str
    port_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.owner_id, field_name="owner_id")
        validate_stable_id(self.port_id, field_name="port_id")


@dataclass(frozen=True, slots=True)
class InterfaceSpec(CanonicalRecord):
    """Directed, unit/clock/balance-checked component connection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/interface-spec'

    interface_id: str
    source: PortRef
    target: PortRef
    balance_role: BalanceRole
    balance_semantics: BalanceSemantics
    uncertainty_contract_id: str
    validity_contract_id: str
    clock_relation: ClockRelationSpec | None = None
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("interface_id", self.interface_id),
            ("uncertainty_contract_id", self.uncertainty_contract_id),
            ("validity_contract_id", self.validity_contract_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.source == self.target:
            raise ValueError("an interface cannot connect a port to itself")
        physical_roles = {
            BalanceRole.CHARGE,
            BalanceRole.ENERGY,
            BalanceRole.MASS,
            BalanceRole.MOMENTUM,
        }
        if self.balance_role in physical_roles and self.balance_semantics not in {
            BalanceSemantics.CONSERVED,
            BalanceSemantics.DISSIPATIVE,
        }:
            raise ValueError("a physical balance requires conserved/dissipative semantics")
        if self.balance_role in {BalanceRole.INFORMATION, BalanceRole.OBSERVATION}:
            if self.balance_semantics is not BalanceSemantics.INFORMATIONAL:
                raise ValueError("an information interface requires informational semantics")
        if self.balance_role is BalanceRole.COMMAND:
            if self.balance_semantics is not BalanceSemantics.COMMAND:
                raise ValueError("a command interface requires command semantics")
        if self.balance_role is BalanceRole.NONE:
            raise ValueError("an interface must declare a nonempty balance role")
        require_extensions(self.extensions)


@dataclass(frozen=True, slots=True)
class ComponentSpec(CanonicalRecord):
    """One recursively parented component inside a normalized system graph."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/component-spec'

    component_id: str
    label: str
    relation: RelationalIdentity
    ports: tuple[PortSpec, ...]
    numerical_view_ids: tuple[str, ...] = ()
    parent_component_id: str | None = None
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.component_id, field_name="component_id")
        validate_nonempty(self.label, field_name="label")
        if self.parent_component_id is not None:
            validate_stable_id(self.parent_component_id, field_name="parent_component_id")
            if self.parent_component_id == self.component_id:
                raise ValueError("a component cannot parent itself")
        require_sorted_unique_ids(self.ports, attribute="port_id", field_name="ports")
        require_sorted_unique_strings(self.numerical_view_ids, field_name="numerical_view_ids")
        require_extensions(self.extensions)


def _validate_relation(
    relation: RelationalIdentity,
    quantities: dict[str, QuantitySpec],
    clocks: dict[str, ClockSpec],
) -> None:
    role_contracts = (
        (
            relation.denominator_quantity_ids,
            {
                QuantityKind.DENOMINATOR,
                QuantityKind.STATE,
                QuantityKind.BOUNDARY,
                QuantityKind.RESOURCE,
            },
            "denominator",
        ),
        (
            relation.history_quantity_ids,
            {QuantityKind.HISTORY, QuantityKind.STATE, QuantityKind.OBSERVATION},
            "history",
        ),
        (relation.action_quantity_ids, {QuantityKind.ACTION}, "action"),
        (relation.receiver_quantity_ids, {QuantityKind.RECEIVER}, "receiver"),
    )
    for identifiers, allowed_kinds, role in role_contracts:
        for identifier in identifiers:
            quantity = quantities.get(identifier)
            if quantity is None:
                raise ValueError(f"{role} quantity {identifier!r} is not registered")
            if quantity.kind not in allowed_kinds:
                raise ValueError(
                    f"quantity {identifier!r} has kind {quantity.kind.value}, "
                    f"not a valid {role} kind"
                )
            phase = quantity.availability.phase
            if role in {"denominator", "history"} and not phase.precedes_or_equals(
                CausalPhase.PRE_ACTION
            ):
                raise ValueError(f"{role} quantity {identifier!r} is post-cutoff")
            if role == "action" and phase not in {
                CausalPhase.ACTION_REQUESTED,
                CausalPhase.ACTION_APPLIED,
            }:
                raise ValueError(f"action quantity {identifier!r} lacks requested/applied timing")
            if role == "receiver" and phase not in {
                CausalPhase.RECEIVER,
                CausalPhase.POST_OUTCOME,
            }:
                raise ValueError(f"receiver quantity {identifier!r} lacks receiver timing")
    horizon_clock = clocks.get(relation.horizon.clock_id)
    if horizon_clock is None:
        raise ValueError("relation horizon clock is not registered")
    if horizon_clock.time_unit != relation.horizon.time_unit:
        raise ValueError("relation horizon unit differs from its clock unit")


@dataclass(frozen=True, slots=True)
class SystemSpec(CanonicalRecord):
    """Prepared open system plus its normalized recursive component graph."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/system-spec'

    system_id: str
    label: str
    boundary_kind: SystemBoundaryKind
    world: WorldSpec
    relation: RelationalIdentity
    clocks: tuple[ClockSpec, ...]
    quantities: tuple[QuantitySpec, ...]
    independent_unit: IndependentUnitSpec
    authority_policy: AuthorityPolicy
    ports: tuple[PortSpec, ...]
    components: tuple[ComponentSpec, ...] = ()
    interfaces: tuple[InterfaceSpec, ...] = ()
    computability_envelopes: tuple[ComputabilityEnvelope, ...] = ()
    numerical_views: tuple[NumericalViewSpec, ...] = ()
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.system_id, field_name="system_id")
        validate_nonempty(self.label, field_name="label")
        self._validate_registry_order()
        clocks = {clock.clock_id: clock for clock in self.clocks}
        quantities = {quantity.quantity_id: quantity for quantity in self.quantities}
        envelopes = {envelope.envelope_id: envelope for envelope in self.computability_envelopes}
        views = {view.view_id: view for view in self.numerical_views}
        self._validate_world_and_quantities(clocks, quantities)
        _validate_relation(self.relation, quantities, clocks)
        self._validate_boundary(quantities)
        owner_ports = self._validate_components(quantities, clocks, views)
        self._validate_numerical_views(envelopes)
        for interface in self.interfaces:
            self._validate_interface(interface, owner_ports, quantities, clocks)
        require_extensions(self.extensions)

    def _validate_registry_order(self) -> None:
        for values, attribute, name in (
            (self.clocks, "clock_id", "clocks"),
            (self.quantities, "quantity_id", "quantities"),
            (self.ports, "port_id", "ports"),
            (self.components, "component_id", "components"),
            (self.interfaces, "interface_id", "interfaces"),
            (
                self.computability_envelopes,
                "envelope_id",
                "computability_envelopes",
            ),
            (self.numerical_views, "view_id", "numerical_views"),
        ):
            require_sorted_unique_ids(values, attribute=attribute, field_name=name)

    def _validate_world_and_quantities(
        self,
        clocks: dict[str, ClockSpec],
        quantities: dict[str, QuantitySpec],
    ) -> None:
        if self.world.kind not in self.authority_policy.allowed_world_kinds:
            raise ValueError("system world is outside the bound authority policy")
        for truth_id in self.world.privileged_truth_quantity_ids:
            if truth_id not in quantities:
                raise ValueError("world privileged truth quantity is not registered")
        for quantity in self.quantities:
            if quantity.clock_id not in clocks:
                raise ValueError(f"quantity {quantity.quantity_id!r} uses an unknown clock")
            if quantity.availability.outcome_access not in (self.world.available_outcome_access):
                raise ValueError(
                    f"quantity {quantity.quantity_id!r} requests unavailable outcome access"
                )

    def _validate_components(
        self,
        quantities: dict[str, QuantitySpec],
        clocks: dict[str, ClockSpec],
        views: dict[str, NumericalViewSpec],
    ) -> dict[str, dict[str, PortSpec]]:
        owner_ports: dict[str, dict[str, PortSpec]] = {
            self.system_id: {port.port_id: port for port in self.ports}
        }
        self._validate_ports(self.ports, quantities)
        component_ids = {component.component_id for component in self.components}
        for component in self.components:
            if (
                component.parent_component_id is not None
                and component.parent_component_id not in component_ids
            ):
                raise ValueError(f"component {component.component_id!r} has an unknown parent")
            _validate_relation(component.relation, quantities, clocks)
            self._validate_ports(component.ports, quantities)
            owner_ports[component.component_id] = {port.port_id: port for port in component.ports}
            unknown_views = set(component.numerical_view_ids) - set(views)
            if unknown_views:
                raise ValueError(
                    f"component {component.component_id!r} references unknown "
                    f"numerical views: {sorted(unknown_views)}"
                )
        self._validate_component_acyclicity()
        return owner_ports

    def _validate_numerical_views(self, envelopes: dict[str, ComputabilityEnvelope]) -> None:
        for view in self.numerical_views:
            if view.world_id != self.world.world_id:
                raise ValueError("numerical view and system world differ")
            if view.physical_preparation_id != self.independent_unit.unit_id:
                raise ValueError(
                    "numerical view must be nested under the physical independent unit"
                )
            if view.computability_envelope_id not in envelopes:
                raise ValueError("numerical view uses an unknown computability envelope")
        if self.world.kind is WorldKind.NUMERICAL_SIMULATOR and not self.numerical_views:
            raise ValueError("a numerical-simulator system requires a numerical view")

    def _validate_boundary(self, quantities: dict[str, QuantitySpec]) -> None:
        kinds = {quantity.kind for quantity in quantities.values()}
        if self.boundary_kind is SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE:
            if QuantityKind.ACTION not in kinds:
                raise ValueError("an open driven system requires an action quantity")
            if not kinds.intersection({QuantityKind.SINK, QuantityKind.BOUNDARY}):
                raise ValueError("an open driven system requires a sink or boundary")

    def _validate_ports(
        self, ports: tuple[PortSpec, ...], quantities: dict[str, QuantitySpec]
    ) -> None:
        for port in ports:
            quantity = quantities.get(port.quantity_id)
            if quantity is None:
                raise ValueError(f"port {port.port_id!r} quantity is not registered")
            if port.clock_id != quantity.clock_id:
                raise ValueError(f"port {port.port_id!r} clock differs from its quantity")
            if quantity.kind is QuantityKind.ACTION:
                if port.authority_action is None:
                    raise ValueError("an action port must bind an authority action")
                if (
                    port.authority_action not in self.authority_policy.allowed_actions
                    or port.authority_action in self.authority_policy.nondelegable_actions
                ):
                    raise ValueError("action port exceeds the bound authority policy")
            elif port.authority_action is not None:
                raise ValueError("only an action quantity may bind actuation authority")

    def _validate_component_acyclicity(self) -> None:
        parents = {
            component.component_id: component.parent_component_id for component in self.components
        }
        for start in parents:
            seen: set[str] = set()
            current: str | None = start
            while current is not None:
                if current in seen:
                    raise ValueError("component parent graph contains a cycle")
                seen.add(current)
                current = parents.get(current)

    @staticmethod
    def _resolve_port(reference: PortRef, owner_ports: dict[str, dict[str, PortSpec]]) -> PortSpec:
        owner = owner_ports.get(reference.owner_id)
        if owner is None or reference.port_id not in owner:
            raise ValueError(
                f"interface references unknown port {reference.owner_id}.{reference.port_id}"
            )
        return owner[reference.port_id]

    @classmethod
    def _validate_interface(
        cls,
        interface: InterfaceSpec,
        owner_ports: dict[str, dict[str, PortSpec]],
        quantities: dict[str, QuantitySpec],
        clocks: dict[str, ClockSpec],
    ) -> None:
        source_port = cls._resolve_port(interface.source, owner_ports)
        target_port = cls._resolve_port(interface.target, owner_ports)
        if source_port.direction is not PortDirection.OUTPUT:
            raise ValueError("interface source must be an output port")
        if target_port.direction is not PortDirection.INPUT:
            raise ValueError("interface target must be an input port")
        if source_port.balance_role is not interface.balance_role:
            raise ValueError("interface source balance role mismatch")
        if target_port.balance_role is not interface.balance_role:
            raise ValueError("interface target balance role mismatch")
        source_quantity = quantities[source_port.quantity_id]
        target_quantity = quantities[target_port.quantity_id]
        source_quantity.require_compatible(target_quantity)
        source_clock = clocks[source_port.clock_id]
        target_clock = clocks[target_port.clock_id]
        relation = interface.clock_relation
        if source_port.clock_id == target_port.clock_id:
            if relation is not None and relation.kind is not ClockRelationKind.IDENTITY:
                raise ValueError("one clock cannot have a non-identity self relation")
            if relation is not None and (
                relation.source_clock_id != source_port.clock_id
                or relation.target_clock_id != target_port.clock_id
            ):
                raise ValueError("identity clock relation references the wrong clock")
        else:
            if relation is None:
                raise ValueError("distinct interface clocks require a clock relation")
            if (
                relation.source_clock_id != source_port.clock_id
                or relation.target_clock_id != target_port.clock_id
            ):
                raise ValueError("interface clock relation references the wrong clocks")
            if (
                source_clock.time_unit != target_clock.time_unit
                or source_clock.coordinate_frame != target_clock.coordinate_frame
            ):
                raise ValueError("interface clock relation requires common native units and frame")

    @property
    def readiness(self) -> ReadinessStatus:
        if any(
            envelope.readiness is ReadinessStatus.COMPUTABILITY_BOUNDARY
            for envelope in self.computability_envelopes
        ):
            return ReadinessStatus.COMPUTABILITY_BOUNDARY
        return ReadinessStatus.READY

    def validate_claim(self, claim: ClaimSpec) -> None:
        validate_claim_in_world(claim, self.world)
        known_relations = {self.relation.relation_id} | {
            component.relation.relation_id for component in self.components
        }
        if claim.relation_id not in known_relations:
            raise ValueError("claim relation does not match the system relation")
        if claim.physical_independent_unit_id != self.independent_unit.unit_id:
            raise ValueError("claim independent unit does not match the system")
        known_views = {view.view_id for view in self.numerical_views}
        unknown = set(claim.numerical_view_ids) - known_views
        if unknown:
            raise ValueError(f"claim references unknown numerical views: {sorted(unknown)}")

    def require_inputs_available(
        self, cutoff: InformationCutoff, quantity_ids: tuple[str, ...]
    ) -> None:
        quantities = {quantity.quantity_id: quantity for quantity in self.quantities}
        for quantity_id in quantity_ids:
            quantity = quantities.get(quantity_id)
            if quantity is None:
                raise ValueError(f"unknown input quantity: {quantity_id}")
            cutoff.require_allows(quantity.availability)
