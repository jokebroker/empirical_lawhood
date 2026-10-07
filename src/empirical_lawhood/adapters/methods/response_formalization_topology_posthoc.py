"Pure estimators for the outcome-visible topology-response archive analysis.\n\nThe functions in this module never discover paths or read evidence.  They keep\nindependent-unit grouping explicit and return finite certificates for numerical,\nformal-context, graph and reliability analyses used by the bounded runner.\n"

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import log, sqrt
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt
from scipy.linalg import orthogonal_procrustes  # type: ignore[import-untyped]
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]
from scipy.stats import beta  # type: ignore[import-untyped]


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class EvidenceUnitRef:
    unit_id: str
    evidence_world: str
    denominator: str

    def __post_init__(self) -> None:
        if not self.unit_id or not self.evidence_world or not self.denominator:
            raise ValueError("evidence-unit fields must be nonempty")


@dataclass(frozen=True, slots=True)
class TypedMapRef:
    map_id: str
    domain: str
    codomain: str
    native_unit: str
    valid: bool

    def __post_init__(self) -> None:
        if not self.map_id or not self.domain or not self.codomain or not self.native_unit:
            raise ValueError("typed-map identity, sections and native unit are required")
        if not isinstance(self.valid, bool):
            raise ValueError("typed-map validity must be explicit")


@dataclass(frozen=True, slots=True)
class ReceiverSection:
    receiver_id: str
    node_ids: tuple[int, ...]
    native_unit: str

    def __post_init__(self) -> None:
        if (
            not self.receiver_id
            or not self.node_ids
            or not self.native_unit
            or len(set(self.node_ids)) != len(self.node_ids)
            or any(value < 0 for value in self.node_ids)
        ):
            raise ValueError("receiver identity, unique nodes and native unit are required")


@dataclass(frozen=True, slots=True)
class ResidualTensor:
    unit_ids: tuple[str, ...]
    axis_names: tuple[str, ...]
    values: FloatArray

    def __post_init__(self) -> None:
        if self.values.ndim != len(self.axis_names):
            raise ValueError("residual axes differ from tensor rank")
        if self.values.shape[0] != len(self.unit_ids):
            raise ValueError("first residual axis must contain independent units")
        if not np.isfinite(self.values).all():
            raise ValueError("residual tensor must be finite")


@dataclass(frozen=True, slots=True)
class TransportFold:
    source_unit_id: str
    target_unit_id: str
    method_id: str
    train_label_visible: bool
    target_label_visible_during_fit: bool

    def __post_init__(self) -> None:
        if self.target_label_visible_during_fit:
            raise ValueError("held-out target labels cannot fit a transport map")


@dataclass(frozen=True, slots=True)
class RestrictionDiagram:
    object_ids: tuple[str, ...]
    maps: tuple[TypedMapRef, ...]
    identity_valid: bool
    composition_valid: bool

    def __post_init__(self) -> None:
        if not self.object_ids or len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("restriction objects must be unique and nonempty")
        if not all(value.valid for value in self.maps):
            raise ValueError("invalid restrictions cannot define a diagram")


@dataclass(frozen=True, slots=True)
class CohomologyEligibility:
    status: str
    d_squared_zero: bool | None
    h0_dimension: int | None
    h1_dimension: int | None
    reason: str

    def __post_init__(self) -> None:
        allowed = {
            "NO_RESTRICTION_SYSTEM",
            "RESTRICTION_SYSTEM_DEFINED",
            "COCHAIN_COMPLEX_DEFINED",
            "CLASS_COMPUTABLE",
            "NONTRIVIAL_CLASS_IN_THIS_FINITE_MODEL",
        }
        if self.status not in allowed or not self.reason:
            raise ValueError("cohomology eligibility status or reason differs")
        if self.status == "NO_RESTRICTION_SYSTEM" and any(
            value is not None for value in (self.d_squared_zero, self.h0_dimension, self.h1_dimension)
        ):
            raise ValueError("an undefined restriction system cannot carry cohomology dimensions")


@dataclass(frozen=True, slots=True)
class ResponseKernelEstimate:
    unit_id: str
    action_id: str
    clock: float
    response: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.unit_id
            or not self.action_id
            or not self.response
            or not np.isfinite(self.clock)
            or not all(np.isfinite(value) for value in self.response)
        ):
            raise ValueError("response-kernel fields must be finite and complete")


@dataclass(frozen=True, slots=True)
class StateAugmentationCandidate:
    candidate_id: str
    feature_indices: tuple[int, ...]
    uses_future: bool = False

    def __post_init__(self) -> None:
        if self.uses_future:
            raise ValueError("future-state leakage is forbidden")
        if (
            not self.candidate_id
            or not self.feature_indices
            or len(set(self.feature_indices)) != len(self.feature_indices)
            or any(value < 0 for value in self.feature_indices)
        ):
            raise ValueError("state-augmentation coordinates differ")


@dataclass(frozen=True, slots=True)
class AdmissionMarginRecord:
    unit_id: str
    decision_index: int
    margins: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.margins or not all(np.isfinite(value) for value in self.margins.values()):
            raise ValueError("admission margins must be present and finite")


@dataclass(frozen=True, slots=True)
class ClosureDependencyRecord:
    node_id: str
    seam_type: str
    available: bool

    def __post_init__(self) -> None:
        if not self.node_id or not self.seam_type or not isinstance(self.available, bool):
            raise ValueError("closure-dependency typing is incomplete")


@dataclass(frozen=True, slots=True)
class AnalysisResultCard:
    workstream: str
    parent_analysis_ids: tuple[str, ...]
    evidence_world: str
    independent_unit: str
    independent_unit_count: int
    status: str
    claim_ceiling: str
    non_entailments: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.workstream not in {f"hv{index}" for index in range(1, 10)}:
            raise ValueError("unknown high-value workstream")
        if not self.parent_analysis_ids:
            raise ValueError("parent analysis IDs are required")
        if self.independent_unit_count < 0:
            raise ValueError("independent-unit count must be nonnegative")
        if not self.evidence_world or not self.independent_unit or not self.claim_ceiling:
            raise ValueError("result-card evidence typing is incomplete")
        if not self.status or not self.non_entailments:
            raise ValueError("result-card status and non-entailments are required")


def require_single_evidence_world(units: Sequence[EvidenceUnitRef]) -> str:
    """Reject numeric pooling across separately typed evidence worlds."""

    if not units:
        raise ValueError("at least one evidence unit is required")
    worlds = {value.evidence_world for value in units}
    if len(worlds) != 1:
        raise ValueError("cross-world numeric pooling requires an explicit transport contract")
    return next(iter(worlds))


def _as_matrix(values: npt.ArrayLike) -> FloatArray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.size == 0 or not np.isfinite(matrix).all():
        raise ValueError("a finite nonempty matrix is required")
    return matrix


def rms(values: npt.ArrayLike) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("RMS requires finite nonempty values")
    return float(np.sqrt(np.mean(np.square(array))))


def singular_spectrum(
    values: npt.ArrayLike,
    *,
    relative_tolerances: Sequence[float],
) -> dict[str, Any]:
    """Return complete spectrum and non-selected rank sensitivity."""

    matrix = _as_matrix(values)
    spectrum = np.linalg.svd(matrix, compute_uv=False)
    energy = np.square(spectrum)
    total = float(np.sum(energy))
    stable_rank = 0.0 if spectrum[0] == 0 else total / float(spectrum[0] ** 2)
    probabilities = energy / total if total > 0 else np.zeros_like(energy)
    nonzero = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-np.sum(nonzero * np.log(nonzero))))
    ranks = {
        str(tolerance): int(np.sum(spectrum > tolerance * spectrum[0]))
        if spectrum[0] > 0
        else 0
        for tolerance in relative_tolerances
    }
    return {
        "shape": tuple(int(value) for value in matrix.shape),
        "singular_values": tuple(float(value) for value in spectrum),
        "stable_rank": float(stable_rank),
        "effective_rank": effective_rank,
        "relative_tolerance_ranks": ranks,
        "frobenius_norm": sqrt(total),
    }


def principal_subspace_angle(
    left: npt.ArrayLike,
    right: npt.ArrayLike,
    *,
    rank: int,
) -> float:
    """Largest principal angle in radians between row-spanned response subspaces."""

    left_matrix = _as_matrix(left)
    right_matrix = _as_matrix(right)
    if left_matrix.shape[1] != right_matrix.shape[1] or rank <= 0:
        raise ValueError("subspaces must share coordinates and have positive rank")
    left_basis = np.linalg.svd(left_matrix, full_matrices=False)[2][:rank].T
    right_basis = np.linalg.svd(right_matrix, full_matrices=False)[2][:rank].T
    singular = np.linalg.svd(left_basis.T @ right_basis, compute_uv=False)
    return float(np.arccos(np.clip(np.min(singular), -1.0, 1.0)))


def fit_ridge_map(source: npt.ArrayLike, target: npt.ArrayLike, alpha: float) -> FloatArray:
    """Fit an affine ridge map after appending a constant coordinate."""

    x = _as_matrix(source)
    y = _as_matrix(target)
    if x.shape[0] != y.shape[0] or alpha < 0:
        raise ValueError("ridge correspondence rows differ or alpha is negative")
    design = np.column_stack((x, np.ones(x.shape[0], dtype=np.float64)))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    return np.asarray(
        np.linalg.solve(design.T @ design + penalty, design.T @ y),
        dtype=np.float64,
    )


def apply_affine_map(values: npt.ArrayLike, coefficients: npt.ArrayLike) -> FloatArray:
    matrix = _as_matrix(values)
    fitted = _as_matrix(coefficients)
    if fitted.shape[0] != matrix.shape[1] + 1:
        raise ValueError("affine coefficient shape differs")
    return np.column_stack((matrix, np.ones(matrix.shape[0], dtype=np.float64))) @ fitted


def fit_orthogonal_alignment(source: npt.ArrayLike, target: npt.ArrayLike) -> dict[str, FloatArray]:
    x = _as_matrix(source)
    y = _as_matrix(target)
    if x.shape != y.shape:
        raise ValueError("orthogonal correspondences must have equal shape")
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    rotation, _ = orthogonal_procrustes(x - x_mean, y - y_mean)
    return {"source_mean": x_mean, "target_mean": y_mean, "rotation": rotation}


def apply_orthogonal_alignment(values: npt.ArrayLike, fit: Mapping[str, FloatArray]) -> FloatArray:
    matrix = _as_matrix(values)
    return (matrix - fit["source_mean"]) @ fit["rotation"] + fit["target_mean"]


def assignment_correspondence(source: npt.ArrayLike, target: npt.ArrayLike) -> tuple[int, ...]:
    """Return the minimum-distance one-to-one nominal correspondence."""

    x = _as_matrix(source)
    y = _as_matrix(target)
    if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
        raise ValueError("assignment panels must have equal shape")
    x_scale = np.std(x, axis=0) + 1e-12
    y_scale = np.std(y, axis=0) + 1e-12
    distance = np.sum(
        np.square(x[:, None, :] / x_scale - y[None, :, :] / y_scale),
        axis=2,
    )
    rows, columns = linear_sum_assignment(distance)
    order = np.empty(x.shape[0], dtype=np.int64)
    order[rows] = columns
    return tuple(int(value) for value in order)


def nearest_centroid_predictions(
    train_values: npt.ArrayLike,
    train_labels: Sequence[str],
    score_values: npt.ArrayLike,
) -> tuple[str, ...]:
    train = _as_matrix(train_values)
    score = _as_matrix(score_values)
    if train.shape[0] != len(train_labels) or train.shape[1] != score.shape[1]:
        raise ValueError("classification panel shape differs")
    labels = tuple(sorted(set(train_labels)))
    centroids = np.stack(
        [np.mean(train[np.asarray(train_labels) == label], axis=0) for label in labels]
    )
    scale = np.std(train, axis=0) + 1e-12
    distance = np.sum(np.square((score[:, None, :] - centroids[None, :, :]) / scale), axis=2)
    return tuple(labels[int(index)] for index in np.argmin(distance, axis=1))


def classification_summary(truth: Sequence[str], predicted: Sequence[str]) -> dict[str, Any]:
    if len(truth) != len(predicted) or not truth:
        raise ValueError("classification vectors must be nonempty and equally sized")
    labels = tuple(sorted(set(truth) | set(predicted)))
    index = {label: position for position, label in enumerate(labels)}
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for expected, observed in zip(truth, predicted, strict=True):
        confusion[index[expected], index[observed]] += 1
    accuracy = float(np.trace(confusion) / np.sum(confusion))
    return {
        "labels": labels,
        "confusion": tuple(tuple(int(value) for value in row) for row in confusion),
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "count": len(truth),
    }


def discrete_mutual_information(
    labels: Sequence[str],
    codes: Sequence[str],
    *,
    miller_madow: bool = True,
) -> dict[str, float]:
    """Finite plug-in mutual information with an optional small-sample correction."""

    if len(labels) != len(codes) or not labels:
        raise ValueError("mutual-information inputs differ")
    n = len(labels)
    label_values = tuple(sorted(set(labels)))
    code_values = tuple(sorted(set(codes)))
    joint = np.zeros((len(label_values), len(code_values)), dtype=np.float64)
    li = {value: index for index, value in enumerate(label_values)}
    ci = {value: index for index, value in enumerate(code_values)}
    for label, code in zip(labels, codes, strict=True):
        joint[li[label], ci[code]] += 1.0
    joint /= n
    row = np.sum(joint, axis=1, keepdims=True)
    column = np.sum(joint, axis=0, keepdims=True)
    expected = row @ column
    valid = joint > 0
    estimate = float(np.sum(joint[valid] * np.log2(joint[valid] / expected[valid])))
    correction = (
        (len(label_values) - 1) * (len(code_values) - 1) / (2.0 * n * log(2.0))
        if miller_madow
        else 0.0
    )
    entropy = -float(np.sum(row[row > 0] * np.log2(row[row > 0])))
    return {
        "plugin_bits": estimate,
        "bias_corrected_bits": max(0.0, estimate - correction),
        "label_entropy_bits": entropy,
    }


def grouped_leave_one_out_ridge(
    features: npt.ArrayLike,
    targets: npt.ArrayLike,
    groups: Sequence[str],
    *,
    alpha: float,
    standardize: bool = True,
) -> dict[str, Any]:
    """Score an affine ridge model while holding out every complete group."""

    x = _as_matrix(features)
    y = _as_matrix(targets)
    if x.shape[0] != y.shape[0] or x.shape[0] != len(groups):
        raise ValueError("grouped cross-fit rows differ")
    predictions = np.full_like(y, np.nan)
    baseline_predictions = np.full_like(y, np.nan)
    fold_rows: list[dict[str, Any]] = []
    group_array = np.asarray(groups)
    for group in sorted(set(groups)):
        test = group_array == group
        train = ~test
        if int(np.sum(train)) <= x.shape[1] or not np.any(test):
            fold_rows.append({"group": group, "status": "INSUFFICIENT_TRAINING_ROWS"})
            continue
        train_x = x[train]
        test_x = x[test]
        if standardize:
            location = np.mean(train_x, axis=0)
            scale = np.std(train_x, axis=0)
            scale[scale < 1e-12] = 1.0
            train_x = (train_x - location) / scale
            test_x = (test_x - location) / scale
        coefficients = fit_ridge_map(train_x, y[train], alpha)
        predictions[test] = apply_affine_map(test_x, coefficients)
        baseline_predictions[test] = np.mean(y[train], axis=0)
        fold_rows.append(
            {
                "group": group,
                "status": "SCORED",
                "row_count": int(np.sum(test)),
                "rmse": rms(predictions[test] - y[test]),
            }
        )
    valid = np.isfinite(predictions).all(axis=1)
    return {
        "status": "SCORED" if np.any(valid) else "UNEVALUABLE",
        "scored_rows": int(np.sum(valid)),
        "rmse": rms(predictions[valid] - y[valid]) if np.any(valid) else None,
        "baseline_rmse": rms(baseline_predictions[valid] - y[valid]) if np.any(valid) else None,
        "folds": tuple(fold_rows),
        "predictions": predictions,
    }


def graph_descriptors(adjacency: npt.ArrayLike, *, ports: Sequence[int]) -> dict[str, Any]:
    """Compute finite graph, spectrum and port-rooted descriptors."""

    matrix = _as_matrix(adjacency)
    if matrix.shape[0] != matrix.shape[1] or not np.allclose(matrix, matrix.T):
        raise ValueError("an undirected square adjacency matrix is required")
    node_count = matrix.shape[0]
    degree = np.sum(matrix > 0, axis=1).astype(np.float64)
    laplacian = np.diag(np.sum(matrix, axis=1)) - matrix
    laplacian_spectrum = np.linalg.eigvalsh(laplacian)
    adjacency_spectrum = np.linalg.eigvalsh(matrix)
    edge_count = int(np.sum(matrix > 0) // 2)
    component_count = _component_count(matrix > 0)
    cycle_rank = edge_count - node_count + component_count
    distances = np.full((node_count, node_count), np.inf, dtype=np.float64)
    np.fill_diagonal(distances, 0.0)
    distances[matrix > 0] = 1.0
    for pivot in range(node_count):
        distances = np.minimum(distances, distances[:, pivot, None] + distances[None, pivot, :])
    port_distances = np.min(distances[:, np.asarray(ports, dtype=np.int64)], axis=1)
    pseudoinverse = np.linalg.pinv(laplacian)
    resistance = []
    for left, right in combinations(ports, 2):
        value = pseudoinverse[left, left] + pseudoinverse[right, right]
        value -= 2.0 * pseudoinverse[left, right]
        resistance.append(float(value))
    return {
        "node_count": int(node_count),
        "edge_count": edge_count,
        "component_count": component_count,
        "cycle_rank": int(cycle_rank),
        "degree_sequence": tuple(float(value) for value in np.sort(degree)),
        "mean_degree": float(np.mean(degree)),
        "degree_variance": float(np.var(degree)),
        "laplacian_spectrum": tuple(float(value) for value in laplacian_spectrum),
        "adjacency_spectrum": tuple(float(value) for value in adjacency_spectrum),
        "spectral_gap": float(laplacian_spectrum[1]) if node_count > 1 else 0.0,
        "port_distance_mean": float(np.mean(port_distances)),
        "port_distance_maximum": float(np.max(port_distances)),
        "effective_port_resistance_mean": float(np.mean(resistance)) if resistance else 0.0,
        "boundary_to_volume": float(len(set(ports)) / node_count),
        "trace_a2": float(np.trace(matrix @ matrix)),
        "trace_a3": float(np.trace(matrix @ matrix @ matrix)),
        "trace_a4": float(np.trace(np.linalg.matrix_power(matrix, 4))),
    }


def _component_count(adjacency: npt.NDArray[np.bool_]) -> int:
    unseen = set(range(adjacency.shape[0]))
    count = 0
    while unseen:
        count += 1
        frontier = [unseen.pop()]
        while frontier:
            node = frontier.pop()
            neighbours = {int(value) for value in np.flatnonzero(adjacency[node])}
            fresh = neighbours & unseen
            unseen -= fresh
            frontier.extend(fresh)
    return count


def formal_closure(
    objects: Sequence[frozenset[str]],
    attributes: frozenset[str],
    *,
    universe: Iterable[str] | None = None,
) -> frozenset[str]:
    """Galois closure of an attribute set in a finite binary context."""

    domain = (
        frozenset(universe)
        if universe is not None
        else frozenset().union(*objects, attributes)
    )
    if not objects:
        return domain
    containing = [obj for obj in objects if attributes <= obj]
    if not containing:
        return domain
    result = containing[0]
    for obj in containing[1:]:
        result = result & obj
    return result


def duquenne_guigues_basis(
    objects: Sequence[frozenset[str]],
    universe: Sequence[str],
) -> tuple[dict[str, tuple[str, ...]], ...]:
    """Enumerate the canonical pseudo-intent implication basis."""

    ordered = tuple(universe)
    subsets = [
        frozenset(ordered[index] for index in range(len(ordered)) if mask & (1 << index))
        for mask in range(1 << len(ordered))
    ]
    subsets.sort(key=lambda value: (len(value), tuple(sorted(value))))
    pseudo_intents: list[frozenset[str]] = []
    implications: list[dict[str, tuple[str, ...]]] = []
    for candidate in subsets:
        closure = formal_closure(objects, candidate, universe=ordered)
        if closure == candidate:
            continue
        if all(
            not (prior < candidate)
            or formal_closure(objects, prior, universe=ordered) <= candidate
            for prior in pseudo_intents
        ):
            pseudo_intents.append(candidate)
            implications.append(
                {
                    "premise": tuple(sorted(candidate)),
                    "conclusion": tuple(sorted(closure - candidate)),
                }
            )
    return tuple(implications)


def implication_holds(
    objects: Sequence[frozenset[str]],
    premise: Iterable[str],
    conclusion: Iterable[str],
) -> bool:
    left = frozenset(premise)
    right = frozenset(conclusion)
    return all(not left <= obj or right <= obj for obj in objects)


def exact_completeness_shapley(attributes: Sequence[str]) -> dict[str, float]:
    """Exact Shapley values for a noncompensating all-attributes predicate."""

    if not attributes:
        raise ValueError("at least one completeness attribute is required")
    value = 1.0 / len(attributes)
    return {attribute: value for attribute in attributes}


def two_set_cech_dimensions(
    *,
    left_dimension: int,
    right_dimension: int,
    overlap_dimension: int,
    left_restriction: npt.ArrayLike,
    right_restriction: npt.ArrayLike,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Compute H0/H1 dimensions for a finite two-set real coefficient cover."""

    left = np.asarray(left_restriction, dtype=np.float64)
    right = np.asarray(right_restriction, dtype=np.float64)
    if left.shape != (overlap_dimension, left_dimension):
        raise ValueError("left restriction shape differs")
    if right.shape != (overlap_dimension, right_dimension):
        raise ValueError("right restriction shape differs")
    d0 = np.column_stack((-left, right))
    rank = int(np.linalg.matrix_rank(d0, tol=tolerance))
    return {
        "c0_dimension": left_dimension + right_dimension,
        "c1_dimension": overlap_dimension,
        "d0_rank": rank,
        "h0_dimension": left_dimension + right_dimension - rank,
        "h1_dimension": overlap_dimension - rank,
        "d_squared_zero": True,
        "coefficient_field": "REAL",
    }


def clopper_pearson(successes: int, trials: int, confidence: float = 0.95) -> dict[str, float]:
    if trials <= 0 or successes < 0 or successes > trials or not 0 < confidence < 1:
        raise ValueError("invalid exact-binomial interval inputs")
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(
        beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    )
    return {"estimate": successes / trials, "lower": lower, "upper": upper}


def directed_reachable(
    edges: Sequence[tuple[str, str]],
    sources: Iterable[str],
    *,
    removed_nodes: frozenset[str] = frozenset(),
    removed_edges: frozenset[tuple[str, str]] = frozenset(),
) -> frozenset[str]:
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        if left in removed_nodes or right in removed_nodes or (left, right) in removed_edges:
            continue
        adjacency.setdefault(left, set()).add(right)
    reached = {value for value in sources if value not in removed_nodes}
    frontier = list(reached)
    while frontier:
        node = frontier.pop()
        for neighbour in adjacency.get(node, set()):
            if neighbour not in reached:
                reached.add(neighbour)
                frontier.append(neighbour)
    return frozenset(reached)


def dominators(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    source: str,
) -> dict[str, tuple[str, ...]]:
    """Classic fixed-point dominators for a finite directed graph."""

    node_set = set(nodes)
    predecessors: dict[str, set[str]] = {node: set() for node in nodes}
    for left, right in edges:
        predecessors[right].add(left)
    dom: dict[str, set[str]] = {
        node: ({source} if node == source else set(node_set)) for node in nodes
    }
    changed = True
    while changed:
        changed = False
        for node in nodes:
            if node == source:
                continue
            incoming = predecessors[node]
            intersection = set.intersection(*(dom[value] for value in incoming)) if incoming else set()
            update = {node} | intersection
            if update != dom[node]:
                dom[node] = update
                changed = True
    return {node: tuple(sorted(values)) for node, values in dom.items()}


def minimal_node_cutsets(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
    sources: Sequence[str],
    target: str,
    *,
    maximum_size: int,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate inclusion-minimal non-source/non-target node cuts."""

    candidates = [node for node in nodes if node not in set(sources) | {target}]
    cuts: list[tuple[str, ...]] = []
    for size in range(1, maximum_size + 1):
        for selected in combinations(candidates, size):
            selected_set = frozenset(selected)
            if any(set(prior) <= selected_set for prior in cuts):
                continue
            if target not in directed_reachable(edges, sources, removed_nodes=selected_set):
                cuts.append(tuple(sorted(selected)))
    return tuple(cuts)


def minimal_edge_cutsets(
    edges: Sequence[tuple[str, str]],
    sources: Sequence[str],
    target: str,
    *,
    maximum_size: int,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Enumerate inclusion-minimal directed edge cuts up to a bounded size."""

    unique_edges = tuple(sorted(set(edges)))
    cuts: list[tuple[tuple[str, str], ...]] = []
    for size in range(1, maximum_size + 1):
        for selected in combinations(unique_edges, size):
            selected_set = frozenset(selected)
            if any(set(prior) <= selected_set for prior in cuts):
                continue
            if target not in directed_reachable(
                unique_edges,
                sources,
                removed_edges=selected_set,
            ):
                cuts.append(tuple(selected))
    return tuple(cuts)


def minimal_directed_paths(
    edges: Sequence[tuple[str, str]],
    sources: Sequence[str],
    target: str,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate every simple source-to-target path in a finite digraph."""

    adjacency: dict[str, tuple[str, ...]] = {}
    for node in {left for left, _ in edges} | {right for _, right in edges}:
        adjacency[node] = tuple(sorted(right for left, right in edges if left == node))
    paths: list[tuple[str, ...]] = []

    def visit(node: str, path: tuple[str, ...]) -> None:
        if node == target:
            paths.append(path)
            return
        for neighbour in adjacency.get(node, ()):
            if neighbour not in path:
                visit(neighbour, (*path, neighbour))

    for source in sorted(set(sources)):
        visit(source, (source,))
    return tuple(sorted(set(paths), key=lambda value: (len(value), value)))


def exact_adaptive_resolution_policy(
    experiment_ids: Sequence[str],
    costs: Mapping[str, int],
    outcome_masks: Mapping[str, Sequence[int]],
    observable_class_masks: Sequence[int],
    *,
    initial_mask: int,
    maximum_acts: int,
    maximum_cost: int,
) -> dict[str, Any]:
    """Solve the finite uniform-prior adaptive class-resolution problem exactly.

    The lexicographic objective maximizes the probability of reaching a full
    observable-equivalence class within the budget, then minimizes expected
    observation cost and expected act count.  Truth labels are used only to
    construct the evaluator's finite partitions, never as a policy feature.
    """

    identifiers = tuple(experiment_ids)
    if (
        not identifiers
        or len(set(identifiers)) != len(identifiers)
        or initial_mask <= 0
        or maximum_acts < 0
        or maximum_cost < 0
    ):
        raise ValueError("adaptive decision inputs are invalid")
    if set(costs) != set(identifiers) or set(outcome_masks) != set(identifiers):
        raise ValueError("adaptive experiment dictionaries differ")
    if any(int(costs[value]) <= 0 for value in identifiers):
        raise ValueError("adaptive experiment costs must be positive")
    classes = tuple(int(value) for value in observable_class_masks)
    if not classes or any(value <= 0 for value in classes):
        raise ValueError("observable classes are empty")
    if any(left & right for index, left in enumerate(classes) for right in classes[index + 1 :]):
        raise ValueError("observable classes overlap")
    if initial_mask & ~sum(classes):
        raise ValueError("observable classes do not cover the initial version space")
    partitions: tuple[tuple[int, ...], ...] = tuple(
        tuple(int(mask) for mask in outcome_masks[identifier] if int(mask) > 0)
        for identifier in identifiers
    )
    for values in partitions:
        union = 0
        for index, value in enumerate(values):
            if any(value & prior for prior in values[:index]):
                raise ValueError("an experiment outcome partition overlaps")
            union |= value
        if initial_mask & ~union:
            raise ValueError("an experiment outcome partition is incomplete")

    def resolved(mask: int) -> bool:
        return any(mask & ~equivalence == 0 for equivalence in classes)

    # probability, expected cost, expected acts, first experiment index
    @lru_cache(maxsize=None)
    def solve(
        mask: int,
        remaining: int,
        acts_left: int,
        cost_left: int,
    ) -> tuple[Fraction, Fraction, Fraction, int | None]:
        if resolved(mask):
            return Fraction(1), Fraction(0), Fraction(0), None
        best: tuple[Fraction, Fraction, Fraction, int | None] = (
            Fraction(0),
            Fraction(0),
            Fraction(0),
            None,
        )
        total = mask.bit_count()
        if acts_left == 0 or cost_left == 0:
            return best
        for index, identifier in enumerate(identifiers):
            bit = 1 << index
            cost = int(costs[identifier])
            if not remaining & bit or cost > cost_left:
                continue
            branches = tuple(mask & value for value in partitions[index] if mask & value)
            if len(branches) <= 1:
                continue
            probability = Fraction(0)
            expected_cost = Fraction(cost)
            expected_acts = Fraction(1)
            for branch in branches:
                weight = Fraction(branch.bit_count(), total)
                child = solve(branch, remaining ^ bit, acts_left - 1, cost_left - cost)
                probability += weight * child[0]
                expected_cost += weight * child[1]
                expected_acts += weight * child[2]
            candidate: tuple[Fraction, Fraction, Fraction, int | None] = (
                probability,
                expected_cost,
                expected_acts,
                index,
            )
            prior_index = best[3]
            better = probability > best[0]
            better |= probability == best[0] and expected_cost < best[1]
            better |= (
                probability == best[0]
                and expected_cost == best[1]
                and expected_acts < best[2]
            )
            better |= (
                probability == best[0]
                and expected_cost == best[1]
                and expected_acts == best[2]
                and prior_index is not None
                and identifier < identifiers[prior_index]
            )
            if better:
                best = candidate
        return best

    solution = solve(initial_mask, (1 << len(identifiers)) - 1, maximum_acts, maximum_cost)
    return {
        "status": "EXACT_ADAPTIVE_POLICY_SOLVED",
        "resolution_probability": float(solution[0]),
        "resolution_probability_exact": f"{solution[0].numerator}/{solution[0].denominator}",
        "expected_cost": float(solution[1]),
        "expected_cost_exact": f"{solution[1].numerator}/{solution[1].denominator}",
        "expected_acts": float(solution[2]),
        "expected_acts_exact": f"{solution[2].numerator}/{solution[2].denominator}",
        "first_experiment_id": (
            None if solution[3] is None else identifiers[solution[3]]
        ),
        "dynamic_states_evaluated": solve.cache_info().currsize,
        "maximum_acts": maximum_acts,
        "maximum_cost": maximum_cost,
        "uniform_hypothesis_prior": True,
        "truth_available_to_policy": False,
        "objective": "MAX_RESOLUTION_PROBABILITY_THEN_MIN_EXPECTED_COST_AND_ACTS",
    }


def kaplan_meier_discrete(
    event_times: Sequence[int | None],
    *,
    maximum_time: int,
) -> tuple[dict[str, Any], ...]:
    """Discrete survival table with each supplied row treated as one unit."""

    if not event_times or maximum_time <= 0:
        raise ValueError("survival inputs are empty")
    survival = 1.0
    rows: list[dict[str, Any]] = []
    for clock in range(1, maximum_time + 1):
        at_risk = sum(value is None or value >= clock for value in event_times)
        events = sum(value == clock for value in event_times)
        hazard = events / at_risk if at_risk else 0.0
        survival *= 1.0 - hazard
        rows.append(
            {
                "clock": clock,
                "at_risk": at_risk,
                "events": events,
                "hazard": hazard,
                "survival": survival,
            }
        )
    return tuple(rows)


def bootstrap_group_mean(
    values: Sequence[float],
    *,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    if not values or resamples <= 0:
        raise ValueError("bootstrap inputs are empty")
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(resamples, len(array)))
    estimates = np.mean(array[indices], axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "estimate": float(np.mean(array)),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "independent_unit_count": len(values),
    }


def binary_auc(labels: Sequence[bool], scores: Sequence[float]) -> float | None:
    """Rank-sum AUC without treating ties as ordered."""

    if len(labels) != len(scores) or not labels:
        raise ValueError("AUC vectors differ")
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += float(positive > negative) + 0.5 * float(positive == negative)
    return wins / (len(positives) * len(negatives))


def exhaustive_subset_resolution(
    experiment_ids: Sequence[str],
    costs: Mapping[str, int],
    survivor_mask: Callable[[tuple[str, ...]], int],
    target_mask: int,
    *,
    maximum_acts: int,
    maximum_cost: int,
) -> dict[str, Any]:
    """Prove the least-cost static separating set by exhaustive enumeration."""

    candidates: list[tuple[int, int, tuple[str, ...]]] = []
    evaluated = 0
    for size in range(maximum_acts + 1):
        for selected in combinations(experiment_ids, size):
            evaluated += 1
            cost = sum(costs[value] for value in selected)
            if cost <= maximum_cost and survivor_mask(selected) == target_mask:
                candidates.append((cost, size, selected))
    if not candidates:
        return {"status": "NO_FEASIBLE_STATIC_SEPARATING_SET", "subsets_evaluated": evaluated}
    cost, size, selected = min(candidates)
    return {
        "status": "EXACT_STATIC_SUBSET_ORACLE",
        "minimum_cost": cost,
        "act_count": size,
        "experiment_ids": selected,
        "subsets_evaluated": evaluated,
        "adaptive_dynamic_programming_claimed": False,
    }


__all__ = [
    "AdmissionMarginRecord",
    "AnalysisResultCard",
    "ClosureDependencyRecord",
    "CohomologyEligibility",
    "EvidenceUnitRef",
    "ReceiverSection",
    "ResidualTensor",
    "ResponseKernelEstimate",
    "RestrictionDiagram",
    "StateAugmentationCandidate",
    "TransportFold",
    "TypedMapRef",
    "apply_affine_map",
    "apply_orthogonal_alignment",
    "assignment_correspondence",
    "binary_auc",
    "bootstrap_group_mean",
    "classification_summary",
    "clopper_pearson",
    "directed_reachable",
    "discrete_mutual_information",
    "dominators",
    "duquenne_guigues_basis",
    "exact_completeness_shapley",
    "exact_adaptive_resolution_policy",
    "exhaustive_subset_resolution",
    "fit_orthogonal_alignment",
    "fit_ridge_map",
    "formal_closure",
    "graph_descriptors",
    "grouped_leave_one_out_ridge",
    "implication_holds",
    "kaplan_meier_discrete",
    "minimal_directed_paths",
    "minimal_edge_cutsets",
    "minimal_node_cutsets",
    "nearest_centroid_predictions",
    "principal_subspace_angle",
    "require_single_evidence_world",
    "rms",
    "singular_spectrum",
    "two_set_cech_dimensions",
]
