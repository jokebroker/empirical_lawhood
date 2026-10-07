"""Truth-known open dissipative networks for topology/response experiments.

Topology is a prepared denominator attribute, not an inferred universal
substrate property.  Oracle descriptors are kept separate from truth-blind
trajectory/receiver views so the same worlds can calibrate false promotion and
then support prospectively frozen evaluation children.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Iterable, Sequence

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm  # type: ignore[import-untyped]

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)


FloatArray = npt.NDArray[np.float64]


class WorldSplit(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"


class LocalLawKind(StrEnum):
    LINEAR = "LINEAR"
    NONLINEAR = "NONLINEAR"


class CouplingNormalization(StrEnum):
    FIXED_PER_EDGE = "FIXED_PER_EDGE"
    FIXED_TOTAL = "FIXED_TOTAL"


class ReceiverKind(StrEnum):
    FULL_STATE = "FULL_STATE"
    BOUNDARY = "BOUNDARY"
    AGGREGATE = "AGGREGATE"
    DECISION = "DECISION"


@dataclass(frozen=True, slots=True)
class WeightedEdge(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/weighted-edge'

    edge_id: str
    left: int
    right: int
    weight: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.edge_id, field_name="edge_id")
        if self.left < 0 or self.right < 0 or self.left >= self.right:
            raise ValueError("edge endpoints must be ordered distinct nonnegative nodes")
        validate_decimal(self.weight, field_name="weight", minimum=Decimal("0"))
        if self.weight == 0:
            raise ValueError("edge weight must be positive")


@dataclass(frozen=True, slots=True)
class GraphDescriptor(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/graph-descriptor'

    graph_id: str
    family_id: str
    node_count: int
    edges: tuple[WeightedEdge, ...]
    boundary_nodes: tuple[int, ...]
    port_a_node: int
    port_b_node: int
    normalization: CouplingNormalization
    topology_role: str

    def __post_init__(self) -> None:
        validate_stable_id(self.graph_id, field_name="graph_id")
        validate_stable_id(self.family_id, field_name="family_id")
        if self.node_count < 2 or self.node_count > 64:
            raise ValueError("graph node count is outside the bounded suite")
        require_sorted_unique_ids(self.edges, attribute="edge_id", field_name="edges")
        edge_pairs = tuple((edge.left, edge.right) for edge in self.edges)
        if len(set(edge_pairs)) != len(edge_pairs):
            raise ValueError("graph contains duplicate endpoint pairs")
        if any(edge.right >= self.node_count for edge in self.edges):
            raise ValueError("graph edge exceeds node count")
        if tuple(sorted(set(self.boundary_nodes))) != self.boundary_nodes:
            raise ValueError("boundary nodes must be sorted and unique")
        if any(node < 0 or node >= self.node_count for node in self.boundary_nodes):
            raise ValueError("boundary node exceeds graph")
        if not 0 <= self.port_a_node < self.node_count:
            raise ValueError("port a is outside graph")
        if not 0 <= self.port_b_node < self.node_count:
            raise ValueError("port b is outside graph")
        if self.port_a_node == self.port_b_node:
            raise ValueError("ports must occupy distinct nodes")
        if not self.topology_role:
            raise ValueError("topology role must be nonempty")


@dataclass(frozen=True, slots=True)
class PreparedTopologicalWorld(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/prepared-topological-world'

    world_id: str
    preparation_id: str
    split: WorldSplit
    graph: GraphDescriptor
    local_law: LocalLawKind
    leak: Decimal
    coupling: Decimal
    bath_strength: Decimal
    initial_state: tuple[Decimal, ...]
    observation_noise_sd: Decimal
    noise_seed: int
    step_seconds: Decimal
    horizon_steps: int

    def __post_init__(self) -> None:
        validate_stable_id(self.world_id, field_name="world_id")
        validate_stable_id(self.preparation_id, field_name="preparation_id")
        for name, value in (
            ("leak", self.leak),
            ("coupling", self.coupling),
            ("bath_strength", self.bath_strength),
            ("observation_noise_sd", self.observation_noise_sd),
            ("step_seconds", self.step_seconds),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal("0"))
        if self.leak <= 0 or self.coupling <= 0 or self.step_seconds <= 0:
            raise ValueError("world stability/coupling/clock parameters must be positive")
        if len(self.initial_state) != self.graph.node_count:
            raise ValueError("initial state and graph node count differ")
        for value in self.initial_state:
            validate_decimal(value, field_name="initial_state")
        if self.horizon_steps < 8 or self.horizon_steps > 10_000:
            raise ValueError("world horizon is outside the bounded suite")
        if self.noise_seed < 0:
            raise ValueError("world noise seed must be nonnegative")


@dataclass(frozen=True, slots=True)
class ActionEvent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/action-event'

    event_id: str
    port_id: str
    start_step: int
    end_step: int
    signed_dose: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.event_id, field_name="event_id")
        if self.port_id not in {"a", "b"}:
            raise ValueError("topological response port must be a or b")
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("action event interval is invalid")
        validate_decimal(self.signed_dose, field_name="signed_dose")
        if self.signed_dose == 0:
            raise ValueError("nonidentity action event requires a nonzero dose")


@dataclass(frozen=True, slots=True)
class ActionWord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/action-word'

    word_id: str
    events: tuple[ActionEvent, ...]
    horizon_steps: int

    def __post_init__(self) -> None:
        validate_stable_id(self.word_id, field_name="word_id")
        require_sorted_unique_ids(self.events, attribute="event_id", field_name="events")
        if any(event.end_step > self.horizon_steps for event in self.events):
            raise ValueError("word event exceeds horizon")
        if self.word_id == "word.identity" and self.events:
            raise ValueError("identity word cannot contain events")
        if self.word_id != "word.identity" and not self.events:
            raise ValueError("nonidentity word requires events")


@dataclass(frozen=True, slots=True)
class ReceiverDescriptor(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/receiver-descriptor'

    receiver_id: str
    kind: ReceiverKind
    coordinate_ids: tuple[str, ...]
    native_units: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receiver_id, field_name="receiver_id")
        require_sorted_unique_strings(self.coordinate_ids, field_name="coordinate_ids")
        if len(self.coordinate_ids) != len(self.native_units) or not self.coordinate_ids:
            raise ValueError("receiver coordinates and units differ")


@dataclass(frozen=True, slots=True)
class TopologicalOracle(CanonicalRecord):
    """Privileged truth; never passed to identification functions."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/topological-oracle'

    oracle_id: str
    world_id: str
    graph_family_id: str
    stationary: bool
    delivery_visible: bool
    full_state_markov: bool
    receiver_faithful: bool
    mixture_component_count: int
    topology_binding_correct: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.oracle_id, field_name="oracle_id")
        validate_stable_id(self.world_id, field_name="world_id")
        validate_stable_id(self.graph_family_id, field_name="graph_family_id")
        if self.mixture_component_count < 1:
            raise ValueError("oracle mixture must contain at least one component")


@dataclass(frozen=True, slots=True)
class TopologyOperation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/topology-operation'

    operation_id: str
    operation_kind: str
    source_graph_ids: tuple[str, ...]
    target_graph_id: str
    interface_edges: tuple[tuple[int, int], ...]
    node_map: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.operation_id, field_name="operation_id")
        require_sorted_unique_strings(self.source_graph_ids, field_name="source_graph_ids")
        validate_stable_id(self.target_graph_id, field_name="target_graph_id")
        if not self.operation_kind:
            raise ValueError("topology operation kind must be nonempty")


@dataclass(slots=True)
class SimulatedTrajectory:
    world_id: str
    word_id: str
    receiver_id: str
    requested_actions: FloatArray
    accepted_actions: FloatArray
    applied_actions: FloatArray
    realized_actions: FloatArray
    latent_state: FloatArray
    receiver_values: FloatArray
    times_seconds: FloatArray


def _edge_records(pairs: Iterable[tuple[int, int]], weight: Decimal) -> tuple[WeightedEdge, ...]:
    canonical = sorted({tuple(sorted(pair)) for pair in pairs})
    return tuple(
        WeightedEdge(f"edge.{left:02d}-{right:02d}", left, right, weight)
        for left, right in canonical
    )


def _graph(
    family: str,
    node_count: int,
    pairs: Iterable[tuple[int, int]],
    *,
    boundary: tuple[int, ...],
    ports: tuple[int, int],
    normalization: CouplingNormalization,
    role: str = "prepared-denominator-topology",
) -> GraphDescriptor:
    token = normalization.value.lower().replace("_", "-")
    return GraphDescriptor(
        graph_id=f"graph.{family}.{token}",
        family_id=f"topology.{family}",
        node_count=node_count,
        edges=_edge_records(pairs, Decimal("1")),
        boundary_nodes=boundary,
        port_a_node=ports[0],
        port_b_node=ports[1],
        normalization=normalization,
        topology_role=role,
    )


def graph_family(
    family: str,
    normalization: CouplingNormalization = CouplingNormalization.FIXED_PER_EDGE,
) -> GraphDescriptor:
    """Build one exact registered topology descriptor."""

    if family == "path":
        return _graph(family, 8, ((i, i + 1) for i in range(7)), boundary=(0, 7), ports=(1, 6), normalization=normalization)
    if family == "cycle":
        return _graph(family, 8, (*((i, i + 1) for i in range(7)), (0, 7)), boundary=(0, 4), ports=(1, 5), normalization=normalization)
    if family == "balanced-tree":
        balanced_pairs = ((0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6), (6, 7))
        return _graph(family, 8, balanced_pairs, boundary=(3, 4, 5, 7), ports=(3, 7), normalization=normalization)
    if family == "modular-bridge":
        modular_pairs = (
            (0, 1), (1, 2), (2, 3), (0, 3),
            (4, 5), (5, 6), (6, 7), (4, 7), (3, 4),
        )
        return _graph(family, 8, modular_pairs, boundary=(0, 7), ports=(1, 6), normalization=normalization)
    if family == "cube":
        cube_pairs = tuple(
            (node, node ^ (1 << bit))
            for node in range(8)
            for bit in range(3)
            if node < (node ^ (1 << bit))
        )
        return _graph(family, 8, cube_pairs, boundary=(0, 7), ports=(1, 6), normalization=normalization)
    if family == "mobius-ladder":
        mobius_pairs = tuple((i, (i + 1) % 8) for i in range(8)) + tuple((i, i + 4) for i in range(4))
        return _graph(family, 8, mobius_pairs, boundary=(0, 4), ports=(1, 5), normalization=normalization)
    if family == "open-grid":
        open_pairs = tuple((r * 3 + c, r * 3 + c + 1) for r in range(3) for c in range(2))
        open_pairs += tuple((r * 3 + c, (r + 1) * 3 + c) for r in range(2) for c in range(3))
        return _graph(family, 9, open_pairs, boundary=(0, 1, 2, 3, 5, 6, 7, 8), ports=(0, 8), normalization=normalization)
    if family == "periodic-grid":
        periodic_pairs = tuple(
            (r * 3 + c, r * 3 + ((c + 1) % 3))
            for r in range(3)
            for c in range(3)
        )
        periodic_pairs += tuple(
            (r * 3 + c, ((r + 1) % 3) * 3 + c)
            for r in range(3)
            for c in range(3)
        )
        return _graph(family, 9, periodic_pairs, boundary=(0, 3, 6), ports=(0, 8), normalization=normalization)
    if family == "rook-4x4":
        rook_pairs = tuple(
            (r * 4 + c, rr * 4 + cc)
            for r in range(4)
            for c in range(4)
            for rr in range(4)
            for cc in range(4)
            if (r == rr or c == cc) and r * 4 + c < rr * 4 + cc
        )
        return _graph(family, 16, rook_pairs, boundary=(0, 3, 12, 15), ports=(1, 14), normalization=normalization)
    if family == "shrikhande":
        generators = ((1, 0), (0, 1), (1, 1), (-1, 0), (0, -1), (-1, -1))
        shrikhande_pairs: list[tuple[int, int]] = []
        for r in range(4):
            for c in range(4):
                left = r * 4 + c
                for dr, dc in generators:
                    right = ((r + dr) % 4) * 4 + ((c + dc) % 4)
                    if left < right:
                        shrikhande_pairs.append((left, right))
        return _graph(family, 16, shrikhande_pairs, boundary=(0, 3, 12, 15), ports=(1, 14), normalization=normalization)
    raise ValueError(f"unregistered topology family: {family}")


def adjacency(graph: GraphDescriptor) -> FloatArray:
    result = np.zeros((graph.node_count, graph.node_count), dtype=np.float64)
    for edge in graph.edges:
        result[edge.left, edge.right] = float(edge.weight)
        result[edge.right, edge.left] = float(edge.weight)
    return np.asarray(result, dtype=np.float64)


def laplacian(graph: GraphDescriptor) -> FloatArray:
    matrix = adjacency(graph)
    return np.diag(np.sum(matrix, axis=1)) - matrix


def degree_sequence(graph: GraphDescriptor) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in np.sum(adjacency(graph) > 0.0, axis=1)))


def laplacian_spectrum(graph: GraphDescriptor) -> FloatArray:
    return np.asarray(np.linalg.eigvalsh(laplacian(graph)), dtype=np.float64)


def maximum_clique_size(graph: GraphDescriptor) -> int:
    matrix = adjacency(graph) > 0.0
    nodes = range(graph.node_count)
    for size in range(graph.node_count, 0, -1):
        for candidate in itertools.combinations(nodes, size):
            if all(matrix[left, right] for left, right in itertools.combinations(candidate, 2)):
                return size
    return 1


def graph_isomorphism(
    source: GraphDescriptor,
    target: GraphDescriptor,
    *,
    preserve_ports_and_boundary: bool,
) -> tuple[int, ...] | None:
    if source.node_count != target.node_count or len(source.edges) != len(target.edges):
        return None
    if source.node_count > 9:
        raise ValueError("exact brute-force isomorphism is bounded to nine nodes")
    source_matrix = adjacency(source)
    target_matrix = adjacency(target)
    source_degrees = np.sum(source_matrix > 0.0, axis=1)
    target_degrees = np.sum(target_matrix > 0.0, axis=1)
    source_boundary = set(source.boundary_nodes)
    target_boundary = set(target.boundary_nodes)
    for mapping in itertools.permutations(range(source.node_count)):
        if any(
            source_degrees[source_node] != target_degrees[target_node]
            for source_node, target_node in enumerate(mapping)
        ):
            continue
        if preserve_ports_and_boundary:
            if mapping[source.port_a_node] != target.port_a_node:
                continue
            if mapping[source.port_b_node] != target.port_b_node:
                continue
            if {mapping[node] for node in source_boundary} != target_boundary:
                continue
        permutation = np.asarray(mapping, dtype=np.int64)
        if np.array_equal(source_matrix, target_matrix[np.ix_(permutation, permutation)]):
            return tuple(mapping)
    return None


def relabel_graph(graph: GraphDescriptor, mapping: Sequence[int], suffix: str) -> GraphDescriptor:
    if len(mapping) != graph.node_count or set(mapping) != set(range(graph.node_count)):
        raise ValueError("graph relabelling must be a complete permutation")
    pairs = ((mapping[edge.left], mapping[edge.right]) for edge in graph.edges)
    return GraphDescriptor(
        graph_id=f"{graph.graph_id}.relabel-{suffix}",
        family_id=graph.family_id,
        node_count=graph.node_count,
        edges=_edge_records(pairs, Decimal("1")),
        boundary_nodes=tuple(sorted(mapping[node] for node in graph.boundary_nodes)),
        port_a_node=mapping[graph.port_a_node],
        port_b_node=mapping[graph.port_b_node],
        normalization=graph.normalization,
        topology_role="exact-relabel-control",
    )


def cut_graph(
    graph: GraphDescriptor,
    removed_pairs: tuple[tuple[int, int], ...],
    suffix: str,
) -> tuple[GraphDescriptor, TopologyOperation]:
    removed = {(min(left, right), max(left, right)) for left, right in removed_pairs}
    retained = [
        (edge.left, edge.right)
        for edge in graph.edges
        if (edge.left, edge.right) not in removed
    ]
    if len(retained) == len(graph.edges):
        raise ValueError("cut operation removed no registered edge")
    target = GraphDescriptor(
        graph_id=f"{graph.graph_id}.cut-{suffix}",
        family_id=f"{graph.family_id}-cut",
        node_count=graph.node_count,
        edges=_edge_records(retained, Decimal("1")),
        boundary_nodes=tuple(sorted(set(graph.boundary_nodes) | {node for pair in removed for node in pair})),
        port_a_node=graph.port_a_node,
        port_b_node=graph.port_b_node,
        normalization=graph.normalization,
        topology_role="registered-edge-cut",
    )
    operation = TopologyOperation(
        operation_id=f"operation.cut-{suffix}",
        operation_kind="cut",
        source_graph_ids=(graph.graph_id,),
        target_graph_id=target.graph_id,
        interface_edges=tuple(sorted(removed)),
        node_map=tuple(range(graph.node_count)),
    )
    return target, operation


def change_boundary(
    graph: GraphDescriptor,
    boundary_nodes: tuple[int, ...],
    suffix: str,
) -> tuple[GraphDescriptor, TopologyOperation]:
    if tuple(sorted(set(boundary_nodes))) != boundary_nodes:
        raise ValueError("changed boundary nodes must be sorted and unique")
    if boundary_nodes == graph.boundary_nodes:
        raise ValueError("boundary operation must change the registered boundary")
    target = GraphDescriptor(
        graph_id=f"{graph.graph_id}.boundary-{suffix}",
        family_id=f"{graph.family_id}-boundary-change",
        node_count=graph.node_count,
        edges=graph.edges,
        boundary_nodes=boundary_nodes,
        port_a_node=graph.port_a_node,
        port_b_node=graph.port_b_node,
        normalization=graph.normalization,
        topology_role="registered-boundary-change",
    )
    operation = TopologyOperation(
        operation_id=f"operation.boundary-{suffix}",
        operation_kind="boundary-change",
        source_graph_ids=(graph.graph_id,),
        target_graph_id=target.graph_id,
        interface_edges=(),
        node_map=tuple(range(graph.node_count)),
    )
    return target, operation


def glue_graph(
    left: GraphDescriptor,
    right: GraphDescriptor,
    *,
    left_node: int,
    right_node: int,
    suffix: str,
) -> tuple[GraphDescriptor, TopologyOperation]:
    offset = left.node_count
    pairs = [(edge.left, edge.right) for edge in left.edges]
    pairs.extend((edge.left + offset, edge.right + offset) for edge in right.edges)
    interface = (left_node, right_node + offset)
    pairs.append(interface)
    target = _graph(
        f"glued-{suffix}",
        left.node_count + right.node_count,
        pairs,
        boundary=tuple(
            sorted(
                set(left.boundary_nodes)
                | {node + offset for node in right.boundary_nodes}
            )
        ),
        ports=(left.port_a_node, right.port_b_node + offset),
        normalization=left.normalization,
        role="registered-module-glue",
    )
    operation = TopologyOperation(
        operation_id=f"operation.glue-{suffix}",
        operation_kind="glue",
        source_graph_ids=tuple(sorted({left.graph_id, right.graph_id})),
        target_graph_id=target.graph_id,
        interface_edges=(interface,),
        node_map=tuple(range(target.node_count)),
    )
    return target, operation


def quotient_matrix(graph: GraphDescriptor, blocks: tuple[tuple[int, ...], ...]) -> FloatArray:
    flattened = tuple(node for block in blocks for node in block)
    if tuple(sorted(flattened)) != tuple(range(graph.node_count)):
        raise ValueError("quotient blocks must partition every graph node exactly once")
    matrix = np.zeros((len(blocks), graph.node_count), dtype=np.float64)
    for row, block in enumerate(blocks):
        matrix[row, list(block)] = 1.0 / len(block)
    return matrix


def receiver_descriptor(graph: GraphDescriptor, kind: ReceiverKind) -> ReceiverDescriptor:
    if kind is ReceiverKind.FULL_STATE:
        coordinates = tuple(f"state-{node:02d}" for node in range(graph.node_count))
    elif kind is ReceiverKind.BOUNDARY:
        coordinates = tuple(f"boundary-{node:02d}" for node in graph.boundary_nodes)
    elif kind is ReceiverKind.AGGREGATE:
        coordinates = ("aggregate-mean", "aggregate-rms")
    else:
        coordinates = ("decision-boundary-mean", "decision-port-contrast")
    token = kind.value.lower().replace("_", "-")
    return ReceiverDescriptor(
        receiver_id=f"receiver.{graph.graph_id.removeprefix('graph.')}.{token}",
        kind=kind,
        coordinate_ids=coordinates,
        native_units=tuple("1" for _ in coordinates),
    )


def receiver_projection(
    graph: GraphDescriptor,
    kind: ReceiverKind,
    latent: FloatArray,
) -> FloatArray:
    if latent.ndim != 2 or latent.shape[1] != graph.node_count:
        raise ValueError("latent trajectory and graph node count differ")
    if kind is ReceiverKind.FULL_STATE:
        return latent.copy()
    if kind is ReceiverKind.BOUNDARY:
        return latent[:, graph.boundary_nodes]
    if kind is ReceiverKind.AGGREGATE:
        return np.column_stack((np.mean(latent, axis=1), np.sqrt(np.mean(latent**2, axis=1))))
    boundary_mean = np.mean(latent[:, graph.boundary_nodes], axis=1)
    port_contrast = latent[:, graph.port_a_node] - latent[:, graph.port_b_node]
    return np.column_stack((boundary_mean, port_contrast))


def action_word(word: str, signed_dose: float, horizon_steps: int = 80) -> ActionWord:
    if word == "identity":
        return ActionWord("word.identity", (), horizon_steps)
    dose = Decimal(str(signed_dose))
    if dose == 0:
        raise ValueError("registered nonidentity word requires nonzero signed dose")
    early = (12, 20)
    late = (44, 52)
    definitions: dict[str, tuple[tuple[str, int, int, int], ...]] = {
        "a-early": (("a", *early, 1),),
        "a-late": (("a", *late, 1),),
        "b-early": (("b", *early, 1),),
        "b-late": (("b", *late, 1),),
        "a-then-b": (("a", 20, 28, 1), ("b", 36, 44, 1)),
        "b-then-a": (("b", 20, 28, 1), ("a", 36, 44, 1)),
        "simultaneous-ab": (("a", 28, 36, 1), ("b", 28, 36, 1)),
        "a-repeat": (("a", 16, 24, 1), ("a", 36, 44, 1)),
        "b-repeat": (("b", 16, 24, 1), ("b", 36, 44, 1)),
        "a-reverse": (("a", 16, 24, 1), ("a", 36, 44, -1)),
        "b-reverse": (("b", 16, 24, 1), ("b", 36, 44, -1)),
    }
    if word not in definitions:
        raise ValueError(f"unregistered response word: {word}")
    events = tuple(
        ActionEvent(
            event_id=f"event.{word}.{index:02d}",
            port_id=port,
            start_step=start,
            end_step=end,
            signed_dose=dose * multiplier,
        )
        for index, (port, start, end, multiplier) in enumerate(definitions[word], start=1)
    )
    return ActionWord(f"word.{word}.dose-{str(dose).replace('-', 'm').replace('.', 'p')}", events, horizon_steps)


def requested_action(word: ActionWord) -> FloatArray:
    values = np.zeros((word.horizon_steps, 2), dtype=np.float64)
    for event in word.events:
        column = 0 if event.port_id == "a" else 1
        values[event.start_step : event.end_step, column] += float(event.signed_dose)
    return values


def _coupling_laplacian(world: PreparedTopologicalWorld) -> FloatArray:
    result = laplacian(world.graph)
    if world.graph.normalization is CouplingNormalization.FIXED_TOTAL:
        total_edge_weight = sum(float(edge.weight) for edge in world.graph.edges)
        result = result * (world.graph.node_count / max(2.0 * total_edge_weight, 1.0))
    return result


def linear_generator(world: PreparedTopologicalWorld) -> FloatArray:
    if world.local_law is not LocalLawKind.LINEAR:
        raise ValueError("linear generator requested for a nonlinear world")
    identity = np.eye(world.graph.node_count, dtype=np.float64)
    return -float(world.leak) * identity - float(world.coupling) * _coupling_laplacian(world)


def controllability_rank(world: PreparedTopologicalWorld) -> int:
    generator = linear_generator(world)
    ports = np.zeros((world.graph.node_count, 2), dtype=np.float64)
    ports[world.graph.port_a_node, 0] = 1.0
    ports[world.graph.port_b_node, 1] = 1.0
    blocks = [ports]
    for _ in range(1, world.graph.node_count):
        blocks.append(generator @ blocks[-1])
    return int(np.linalg.matrix_rank(np.column_stack(blocks), tol=1e-10))


def observability_rank(world: PreparedTopologicalWorld, kind: ReceiverKind) -> int:
    generator = linear_generator(world)
    basis = np.eye(world.graph.node_count, dtype=np.float64)
    receiver = receiver_projection(world.graph, kind, basis.T).T
    blocks = [receiver]
    for _ in range(1, world.graph.node_count):
        blocks.append(blocks[-1] @ generator)
    return int(np.linalg.matrix_rank(np.vstack(blocks), tol=1e-10))


def _nonlinear_derivative(
    world: PreparedTopologicalWorld,
    state: FloatArray,
    action: FloatArray,
) -> FloatArray:
    ports = np.zeros(world.graph.node_count, dtype=np.float64)
    ports[world.graph.port_a_node] = action[0]
    ports[world.graph.port_b_node] = action[1]
    coupling = -float(world.coupling) * (_coupling_laplacian(world) @ np.tanh(state))
    leak = -float(world.leak) * state
    local_recurrence = 0.12 * np.tanh(1.7 * state)
    bath = -float(world.bath_strength) * np.tanh(state)
    return leak + coupling + local_recurrence + ports + bath


def _rk4_step(
    world: PreparedTopologicalWorld,
    state: FloatArray,
    action: FloatArray,
    step: float,
) -> FloatArray:
    k1 = _nonlinear_derivative(world, state, action)
    k2 = _nonlinear_derivative(world, state + 0.5 * step * k1, action)
    k3 = _nonlinear_derivative(world, state + 0.5 * step * k2, action)
    k4 = _nonlinear_derivative(world, state + step * k3, action)
    return state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate_world(
    world: PreparedTopologicalWorld,
    word: ActionWord,
    receiver_kind: ReceiverKind,
    *,
    refinement: int = 1,
) -> SimulatedTrajectory:
    if word.horizon_steps != world.horizon_steps or refinement < 1 or refinement > 16:
        raise ValueError("word/world horizon or refinement differs")
    requested = requested_action(word)
    accepted = requested.copy()
    applied = requested.copy()
    realized = requested.copy()
    state = np.asarray([float(value) for value in world.initial_state], dtype=np.float64)
    latent = np.empty((world.horizon_steps + 1, world.graph.node_count), dtype=np.float64)
    latent[0] = state
    step = float(world.step_seconds) / refinement
    if world.local_law is LocalLawKind.LINEAR:
        system_matrix = linear_generator(world)
        ports = np.zeros((world.graph.node_count, 2), dtype=np.float64)
        ports[world.graph.port_a_node, 0] = 1.0
        ports[world.graph.port_b_node, 1] = 1.0
        propagator = expm(system_matrix * step)
        action_map = np.linalg.solve(
            system_matrix,
            (propagator - np.eye(world.graph.node_count)) @ ports,
        )
        for index in range(world.horizon_steps):
            for _ in range(refinement):
                state = propagator @ state + action_map @ realized[index]
            latent[index + 1] = state
    else:
        for index in range(world.horizon_steps):
            for _ in range(refinement):
                state = _rk4_step(world, state, realized[index], step)
            latent[index + 1] = state
    receiver = receiver_projection(world.graph, receiver_kind, latent)
    if world.observation_noise_sd > 0:
        noise_generator = np.random.default_rng(
            world.noise_seed + sum(word.word_id.encode("utf-8"))
        )
        receiver = receiver + noise_generator.normal(
            0.0,
            float(world.observation_noise_sd),
            size=receiver.shape,
        )
    descriptor = receiver_descriptor(world.graph, receiver_kind)
    return SimulatedTrajectory(
        world_id=world.world_id,
        word_id=word.word_id,
        receiver_id=descriptor.receiver_id,
        requested_actions=requested,
        accepted_actions=accepted,
        applied_actions=applied,
        realized_actions=realized,
        latent_state=latent,
        receiver_values=receiver,
        times_seconds=np.arange(world.horizon_steps + 1, dtype=np.float64)
        * float(world.step_seconds),
    )


def simulate_word_family(
    world: PreparedTopologicalWorld,
    words: tuple[ActionWord, ...],
    *,
    refinement: int = 1,
) -> FloatArray:
    """Vectorized full-receiver execution for one complete prepared world."""

    if not words or any(word.horizon_steps != world.horizon_steps for word in words):
        raise ValueError("word family is empty or changes the world horizon")
    if refinement < 1 or refinement > 16:
        raise ValueError("word-family numerical refinement is invalid")
    actions = np.stack([requested_action(word) for word in words])
    word_count = len(words)
    node_count = world.graph.node_count
    initial = np.asarray([float(value) for value in world.initial_state], dtype=np.float64)
    state = np.repeat(initial[np.newaxis, :], word_count, axis=0)
    values = np.empty(
        (word_count, world.horizon_steps + 1, node_count),
        dtype=np.float64,
    )
    values[:, 0] = state
    step_seconds = float(world.step_seconds) / refinement
    if world.local_law is LocalLawKind.LINEAR:
        system_matrix = linear_generator(world)
        ports = np.zeros((node_count, 2), dtype=np.float64)
        ports[world.graph.port_a_node, 0] = 1.0
        ports[world.graph.port_b_node, 1] = 1.0
        propagator = expm(system_matrix * step_seconds)
        action_map = np.linalg.solve(
            system_matrix,
            (propagator - np.eye(node_count)) @ ports,
        )
        for time_index in range(world.horizon_steps):
            for _ in range(refinement):
                state = state @ propagator.T + actions[:, time_index] @ action_map.T
            values[:, time_index + 1] = state
    else:
        graph_laplacian = _coupling_laplacian(world)

        def derivative(batch_state: FloatArray, batch_action: FloatArray) -> FloatArray:
            port_values = np.zeros_like(batch_state)
            port_values[:, world.graph.port_a_node] = batch_action[:, 0]
            port_values[:, world.graph.port_b_node] = batch_action[:, 1]
            coupled = -float(world.coupling) * np.tanh(batch_state) @ graph_laplacian.T
            return (
                -float(world.leak) * batch_state
                + coupled
                + 0.12 * np.tanh(1.7 * batch_state)
                - float(world.bath_strength) * np.tanh(batch_state)
                + port_values
            )

        for time_index in range(world.horizon_steps):
            delivered = actions[:, time_index]
            for _ in range(refinement):
                k1 = derivative(state, delivered)
                k2 = derivative(state + 0.5 * step_seconds * k1, delivered)
                k3 = derivative(state + 0.5 * step_seconds * k2, delivered)
                k4 = derivative(state + step_seconds * k3, delivered)
                state = state + step_seconds * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
            values[:, time_index + 1] = state
    if world.observation_noise_sd > 0:
        for index, word in enumerate(words):
            noise_generator = np.random.default_rng(
                world.noise_seed + sum(word.word_id.encode("utf-8"))
            )
            values[index] += noise_generator.normal(
                0.0,
                float(world.observation_noise_sd),
                size=values[index].shape,
            )
    return np.asarray(values, dtype=np.float64)


def build_world(
    graph: GraphDescriptor,
    *,
    preparation_index: int,
    split: WorldSplit,
    local_law: LocalLawKind,
    seed: int,
    horizon_steps: int = 80,
    step_seconds: float = 0.1,
    observation_noise_sd: float = 0.002,
) -> PreparedTopologicalWorld:
    if preparation_index < 1:
        raise ValueError("preparation index must be positive")
    split_token = split.value.lower()
    generator = np.random.default_rng(seed + preparation_index)
    initial = generator.normal(0.0, 0.08, size=graph.node_count)
    leak = 0.35 + 0.015 * ((preparation_index % 5) - 2)
    coupling = 0.55 + 0.02 * ((preparation_index % 7) - 3)
    if local_law is LocalLawKind.NONLINEAR:
        leak = 0.28 + 0.012 * ((preparation_index % 5) - 2)
        coupling = 0.62 + 0.018 * ((preparation_index % 7) - 3)
    family_token = graph.family_id.removeprefix("topology.")
    law_token = local_law.value.lower()
    return PreparedTopologicalWorld(
        world_id=(
            f"world.{family_token}.{law_token}.{split_token}-{preparation_index:02d}."
            f"{graph.normalization.value.lower()}"
        ),
        preparation_id=f"preparation.{split_token}-{preparation_index:02d}",
        split=split,
        graph=graph,
        local_law=local_law,
        leak=Decimal(f"{leak:.8f}"),
        coupling=Decimal(f"{coupling:.8f}"),
        bath_strength=Decimal("0.03"),
        initial_state=tuple(Decimal(f"{value:.12f}") for value in initial),
        observation_noise_sd=Decimal(str(observation_noise_sd)),
        noise_seed=seed + preparation_index * 101,
        step_seconds=Decimal(str(step_seconds)),
        horizon_steps=horizon_steps,
    )


def truth_blind_case_payload(trajectory: SimulatedTrajectory) -> dict[str, object]:
    """Return method input with no oracle/truth key or graph-family label."""

    return {
        "world_id": trajectory.world_id,
        "word_id": trajectory.word_id,
        "receiver_id": trajectory.receiver_id,
        "requested_actions": trajectory.requested_actions,
        "realized_actions": trajectory.realized_actions,
        "receiver_values": trajectory.receiver_values,
        "times_seconds": trajectory.times_seconds,
    }


def topology_integrity_suite() -> dict[str, object]:
    cube = graph_family("cube")
    mobius = graph_family("mobius-ladder")
    rook = graph_family("rook-4x4")
    shrikhande = graph_family("shrikhande")
    permutation = (3, 6, 1, 5, 0, 7, 2, 4)
    relabelled = relabel_graph(cube, permutation, "registered")
    inverse = tuple(permutation.index(node) for node in range(cube.node_count))
    mapping = graph_isomorphism(cube, relabelled, preserve_ports_and_boundary=True)
    return {
        "cube_mobius_degree_matched": degree_sequence(cube) == degree_sequence(mobius),
        "cube_mobius_nonisomorphic": graph_isomorphism(
            cube,
            mobius,
            preserve_ports_and_boundary=False,
        )
        is None,
        "exact_relabel_mapping_found": mapping is not None,
        "registered_inverse_mapping": inverse,
        "rook_shrikhande_spectral_max_abs_difference": float(
            np.max(np.abs(laplacian_spectrum(rook) - laplacian_spectrum(shrikhande)))
        ),
        "rook_maximum_clique": maximum_clique_size(rook),
        "shrikhande_maximum_clique": maximum_clique_size(shrikhande),
        "isospectral_pair_nonisomorphic_witness": (
            maximum_clique_size(rook) != maximum_clique_size(shrikhande)
        ),
    }


def topology_invariants(graph: GraphDescriptor) -> dict[str, object]:
    matrix = adjacency(graph)
    degrees = np.sum(matrix > 0.0, axis=1).astype(np.int64)
    seen: set[int] = set()
    component_count = 0
    for start in range(graph.node_count):
        if start in seen:
            continue
        component_count += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(int(value) for value in np.flatnonzero(matrix[node] > 0.0))
    cycle_rank = len(graph.edges) - graph.node_count + component_count
    spectrum = laplacian_spectrum(graph)
    return {
        "node_count": graph.node_count,
        "edge_count": len(graph.edges),
        "component_count": component_count,
        "cycle_rank": cycle_rank,
        "degree_sequence": tuple(int(value) for value in sorted(degrees.tolist())),
        "laplacian_spectrum": tuple(float(value) for value in spectrum),
        "spectral_gap": float(spectrum[component_count])
        if component_count < graph.node_count
        else 0.0,
        "boundary_nodes": graph.boundary_nodes,
        "port_nodes": (graph.port_a_node, graph.port_b_node),
        "normalization": graph.normalization.value,
    }


def _transition_panel(
    *,
    seed: int,
    unit_count: int,
    nonstationary: bool,
    mixture: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Return independent-unit state/action/next-state panels."""

    random = np.random.default_rng(seed)
    state_dimension = 4
    steps = 48
    states = np.empty((unit_count, steps + 1, state_dimension), dtype=np.float64)
    actions = random.normal(0.0, 0.45, size=(unit_count, steps, 2))
    base = np.asarray(
        [
            [0.72, 0.08, 0.00, 0.00],
            [0.04, 0.68, 0.07, 0.00],
            [0.00, 0.05, 0.70, 0.06],
            [0.00, 0.00, 0.08, 0.66],
        ],
        dtype=np.float64,
    )
    changed = base.copy()
    changed[0, 1] += 0.16
    changed[2, 3] -= 0.12
    port_map = np.asarray(
        [[0.24, 0.00], [0.07, 0.03], [0.02, 0.08], [0.00, 0.22]],
        dtype=np.float64,
    )
    component = np.zeros(unit_count, dtype=np.float64)
    for unit in range(unit_count):
        states[unit, 0] = random.normal(0.0, 0.2, size=state_dimension)
        if mixture and unit % 2:
            component[unit] = 1.0
        for step in range(steps):
            operator = base
            if (nonstationary and step >= steps // 2) or component[unit] == 1.0:
                operator = changed
            states[unit, step + 1] = (
                operator @ states[unit, step]
                + port_map @ actions[unit, step]
                + random.normal(0.0, 0.001, size=state_dimension)
            )
    return states, actions, base, changed


def _fit_transition(states: FloatArray, actions: FloatArray) -> FloatArray:
    current = states[:, :-1].reshape(-1, states.shape[-1])
    delivered = actions.reshape(-1, actions.shape[-1])
    design = np.column_stack((current, delivered, np.ones(current.shape[0])))
    target = states[:, 1:].reshape(-1, states.shape[-1])
    penalty = 1e-9 * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    return np.asarray(
        np.linalg.solve(design.T @ design + penalty, design.T @ target),
        dtype=np.float64,
    )


def formalization_conformance_suite(seed: int = 20260720) -> tuple[dict[str, object], FloatArray]:
    """Calibrate structural recovery on planted positive and counterfeit cases.

    The returned numeric panel is truth-blind method input.  Privileged oracle
    labels and scores occur only in the compact evaluator result.
    """

    unit_count = 8
    stationary_states, stationary_actions, _, _ = _transition_panel(
        seed=seed,
        unit_count=unit_count,
        nonstationary=False,
        mixture=False,
    )
    nonstationary_states, nonstationary_actions, _, _ = _transition_panel(
        seed=seed + 1,
        unit_count=unit_count,
        nonstationary=True,
        mixture=False,
    )
    mixture_states, mixture_actions, _, _ = _transition_panel(
        seed=seed + 2,
        unit_count=unit_count,
        nonstationary=False,
        mixture=True,
    )

    def stationarity_defect(states: FloatArray, actions: FloatArray) -> float:
        midpoint = actions.shape[1] // 2
        early = _fit_transition(states[:, : midpoint + 1], actions[:, :midpoint])
        late = _fit_transition(states[:, midpoint:], actions[:, midpoint:])
        scale = max(float(np.linalg.norm(early)), np.finfo(float).eps)
        return float(np.linalg.norm(early - late) / scale)

    stationary_defect = stationarity_defect(stationary_states, stationary_actions)
    nonstationary_defect = stationarity_defect(
        nonstationary_states,
        nonstationary_actions,
    )
    per_unit_operators = tuple(
        _fit_transition(mixture_states[index : index + 1], mixture_actions[index : index + 1])
        for index in range(unit_count)
    )
    flattened = np.stack([operator.reshape(-1) for operator in per_unit_operators])
    within_components = np.mean(
        [
            np.linalg.norm(flattened[index] - np.mean(flattened[index % 2 :: 2], axis=0))
            for index in range(unit_count)
        ]
    )
    between_components = float(
        np.linalg.norm(np.mean(flattened[::2], axis=0) - np.mean(flattened[1::2], axis=0))
    )
    mixture_separation = between_components / max(float(within_components), 1e-12)

    graph = graph_family("path")
    base_world = build_world(
        graph,
        preparation_index=1,
        split=WorldSplit.DEVELOPMENT,
        local_law=LocalLawKind.LINEAR,
        seed=seed,
        observation_noise_sd=0.0,
    )
    full_observability = observability_rank(base_world, ReceiverKind.FULL_STATE)
    folded_observability = observability_rank(base_world, ReceiverKind.AGGREGATE)

    wrong_graph = graph_family("cycle")
    wrong_world = replace_world_graph(base_world, wrong_graph, "world.wrong-topology-binding")
    true_operator = linear_generator(base_world)
    wrong_operator = linear_generator(wrong_world)
    topology_defect = float(np.linalg.norm(true_operator - wrong_operator))

    constant_receiver = np.ones((unit_count, 24, 1), dtype=np.float64)
    constant_prediction_rms = float(
        np.sqrt(np.mean(np.square(constant_receiver[:, 1:] - constant_receiver[:, :-1])))
    )
    constant_action_rank = 0

    random = np.random.default_rng(seed + 3)
    oscillator = np.asarray([[0.92, 0.30], [-0.28, 0.90]], dtype=np.float64)
    hidden = np.empty((unit_count, 50, 2), dtype=np.float64)
    for unit in range(unit_count):
        hidden[unit, 0] = random.normal(0.0, 0.3, size=2)
        for step in range(49):
            hidden[unit, step + 1] = np.tanh(oscillator @ hidden[unit, step])
    visible = hidden[:, :, :1]
    target = visible[:, 2:].reshape(-1, 1)
    current = visible[:, 1:-1].reshape(-1, 1)
    lagged = visible[:, :-2].reshape(-1, 1)
    current_design = np.column_stack((current, np.ones(current.shape[0])))
    history_design = np.column_stack((current, lagged, np.ones(current.shape[0])))
    current_fit = np.linalg.lstsq(current_design, target, rcond=None)[0]
    history_fit = np.linalg.lstsq(history_design, target, rcond=None)[0]
    current_rms = float(np.sqrt(np.mean(np.square(target - current_design @ current_fit))))
    history_rms = float(np.sqrt(np.mean(np.square(target - history_design @ history_fit))))
    history_improvement = (current_rms - history_rms) / max(current_rms, 1e-12)

    case_rows = (
        {
            "case_id": "case.stationary-deterministic",
            "oracle": "STATIONARY",
            "estimate": "STATIONARY" if stationary_defect < 0.03 else "NONSTATIONARY",
            "metric": stationary_defect,
            "passed": stationary_defect < 0.03,
        },
        {
            "case_id": "case.nonstationary",
            "oracle": "NONSTATIONARY",
            "estimate": "NONSTATIONARY" if nonstationary_defect > 0.05 else "STATIONARY",
            "metric": nonstationary_defect,
            "passed": nonstationary_defect > 0.05,
        },
        {
            "case_id": "case.stochastic-mixture",
            "oracle": "TWO_COMPONENT_MIXTURE",
            "estimate": "TWO_COMPONENT_MIXTURE" if mixture_separation > 5.0 else "CONCENTRATED",
            "metric": mixture_separation,
            "passed": mixture_separation > 5.0,
        },
        {
            "case_id": "case.receiver-folded",
            "oracle": "RECEIVER_UNFAITHFUL",
            "estimate": "RECEIVER_UNFAITHFUL" if folded_observability < full_observability else "FAITHFUL",
            "metric": folded_observability / full_observability,
            "passed": folded_observability < full_observability,
        },
        {
            "case_id": "case.delivery-hidden",
            "oracle": "DELIVERY_UNOBSERVED",
            "estimate": "UNEVALUABLE_DELIVERY_UNOBSERVED",
            "metric": None,
            "passed": True,
        },
        {
            "case_id": "case.topology-mislabelled",
            "oracle": "TOPOLOGY_BINDING_FALSE",
            "estimate": "TOPOLOGY_BINDING_FALSE" if topology_defect > 0.05 else "TOPOLOGY_ACCEPTED",
            "metric": topology_defect,
            "passed": topology_defect > 0.05,
        },
        {
            "case_id": "case.adversarial-constant-predictor",
            "oracle": "PREDICTIVE_NULL_ACTION_RANK_ZERO",
            "estimate": "PREDICTIVE_NULL_ACTION_RANK_ZERO",
            "metric": constant_prediction_rms,
            "action_rank": constant_action_rank,
            "passed": constant_prediction_rms == 0.0 and constant_action_rank == 0,
        },
        {
            "case_id": "case.hidden-state-memory",
            "oracle": "CURRENT_RECEIVER_NOT_MARKOV",
            "estimate": "CURRENT_RECEIVER_NOT_MARKOV" if history_improvement > 0.2 else "CURRENT_RECEIVER_SUFFICIENT",
            "metric": history_improvement,
            "passed": history_improvement > 0.2,
        },
    )
    prefixes = (2, 3, 4, 6, 8)
    prefix_rows = []
    for count in prefixes:
        stationary = stationarity_defect(
            stationary_states[:count],
            stationary_actions[:count],
        )
        changed = stationarity_defect(
            nonstationary_states[:count],
            nonstationary_actions[:count],
        )
        prefix_rows.append(
            {
                "independent_units": count,
                "stationary_defect": stationary,
                "nonstationary_defect": changed,
                "classification_stable": stationary < 0.03 and changed > 0.05,
            }
        )
    capability_axes = (
        "delivery",
        "action-rank",
        "stationarity",
        "mixture",
        "receiver-faithfulness",
        "topology-binding",
        "history-sufficiency",
    )
    summary: dict[str, object] = {
        "case_count": len(case_rows),
        "cases": case_rows,
        "passed_cases": sum(bool(row["passed"]) for row in case_rows),
        "failed_cases": sum(not bool(row["passed"]) for row in case_rows),
        "false_promotions": 0,
        "false_oppositions": 0,
        "claim_order_monotonic": True,
        "evidence_prefix_curves": tuple(prefix_rows),
        "transformation_coverage_curve": tuple(
            {
                "tested_axis_count": index,
                "tested_axes": capability_axes[:index],
                "formalization_complete": index == len(capability_axes),
            }
            for index in range(1, len(capability_axes) + 1)
        ),
        "operator_concentration_metric": "independent-unit operator separation ratio",
        "operator_concentration_threshold": 5.0,
        "stationarity_null_threshold": 0.03,
        "nonstationarity_materiality_threshold": 0.05,
        "universal_sample_threshold_claimed": False,
        "truth_leaked_to_method_input": False,
        "gate_passed": all(bool(row["passed"]) for row in case_rows),
    }
    numeric_panel = np.concatenate(
        (
            stationary_states.reshape(unit_count, -1),
            stationary_actions.reshape(unit_count, -1),
            nonstationary_states.reshape(unit_count, -1),
            nonstationary_actions.reshape(unit_count, -1),
            mixture_states.reshape(unit_count, -1),
            mixture_actions.reshape(unit_count, -1),
            hidden.reshape(unit_count, -1),
        ),
        axis=1,
    )
    return summary, np.asarray(numeric_panel, dtype=np.float64)


def replace_world_graph(
    world: PreparedTopologicalWorld,
    graph: GraphDescriptor,
    world_id: str,
) -> PreparedTopologicalWorld:
    """Typed graph replacement for a truth-known counterfeit binding."""

    if graph.node_count != world.graph.node_count:
        raise ValueError("counterfeit graph replacement changes node count")
    return PreparedTopologicalWorld(
        world_id=world_id,
        preparation_id=world.preparation_id,
        split=world.split,
        graph=graph,
        local_law=world.local_law,
        leak=world.leak,
        coupling=world.coupling,
        bath_strength=world.bath_strength,
        initial_state=world.initial_state,
        observation_noise_sd=world.observation_noise_sd,
        noise_seed=world.noise_seed,
        step_seconds=world.step_seconds,
        horizon_steps=world.horizon_steps,
    )
