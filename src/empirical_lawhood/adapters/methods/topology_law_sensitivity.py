"Prospectively frozen sensitivity diagnostics for topology-law assessment.\n\nThese diagnostics keep the primary estimator in :mod:`topology_law` fixed.\nThey add only the analyses declared in the sensitivity design:\ntopology-family deletion, descriptor-controlled leave-family-out prediction,\nempirical linearized reachability/observability, and finite action-fibre/loop\ndiagnostics. Every fitted quantity uses development preparations; evaluation\npreparations are scored only after the source bytes for this module are frozen.\n"

from __future__ import annotations

from typing import Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.reference_worlds.topological_response import (
    ReceiverKind,
    adjacency,
    graph_family,
    laplacian_spectrum,
)

from .response_formalization import (
    affine_prediction,
    bootstrap_mean_interval,
    fit_affine_operator,
    floor_certified_rank,
    rms,
)
from .topology_law import _fit_models, _operator_errors, _project


FloatArray = npt.NDArray[np.float64]


def _word_index(word_ids: Sequence[str], word_id: str) -> int:
    try:
        return word_ids.index(word_id)
    except ValueError as error:
        raise ValueError(f"required registered word is absent: {word_id}") from error


def _component_count(matrix: FloatArray) -> int:
    seen: set[int] = set()
    count = 0
    for start in range(matrix.shape[0]):
        if start in seen:
            continue
        count += 1
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(int(value) for value in np.flatnonzero(matrix[node] > 0.0))
    return count


def _shortest_distances(matrix: FloatArray, start: int) -> npt.NDArray[np.int64]:
    distances = np.full(matrix.shape[0], matrix.shape[0] + 1, dtype=np.int64)
    distances[start] = 0
    queue = [start]
    for node in queue:
        for neighbour in np.flatnonzero(matrix[node] > 0.0):
            candidate = int(neighbour)
            if distances[candidate] > distances[node] + 1:
                distances[candidate] = distances[node] + 1
                queue.append(candidate)
    return distances


def _descriptor_vectors(family: str) -> tuple[FloatArray, FloatArray, dict[str, object]]:
    graph = graph_family(family)
    matrix = np.asarray(adjacency(graph) > 0.0, dtype=np.float64)
    degrees = np.sum(matrix, axis=1)
    spectrum = laplacian_spectrum(graph)
    distances = np.stack(
        [_shortest_distances(matrix, node) for node in range(graph.node_count)]
    )
    finite_distances = distances[distances <= graph.node_count]
    diameter = int(np.max(finite_distances))
    port_distance = int(distances[graph.port_a_node, graph.port_b_node])
    triangles = int(round(float(np.trace(matrix @ matrix @ matrix) / 6.0)))
    bridges = 0
    for edge in graph.edges:
        reduced = matrix.copy()
        reduced[edge.left, edge.right] = 0.0
        reduced[edge.right, edge.left] = 0.0
        bridges += _component_count(reduced) > 1
    components = _component_count(matrix)
    cycle_rank = len(graph.edges) - graph.node_count + components
    baseline = np.asarray(
        (
            graph.node_count,
            float(np.mean(degrees)),
            float(np.std(degrees)),
            float(spectrum[components]) if components < graph.node_count else 0.0,
            float(np.max(spectrum)),
            float(np.mean(np.square(spectrum))),
            port_distance,
            len(graph.boundary_nodes) / graph.node_count,
        ),
        dtype=np.float64,
    )
    extension = np.asarray(
        (
            cycle_rank / graph.node_count,
            triangles / graph.node_count,
            bridges / max(len(graph.edges), 1),
            diameter / graph.node_count,
        ),
        dtype=np.float64,
    )
    return (
        baseline,
        np.concatenate((baseline, extension)),
        {
            "family": family,
            "baseline": tuple(float(value) for value in baseline),
            "extension": tuple(float(value) for value in extension),
            "cycle_rank": cycle_rank,
            "triangle_count": triangles,
            "bridge_count": bridges,
            "diameter": diameter,
            "port_distance": port_distance,
        },
    )


def _ridge_predict(
    train_features: FloatArray,
    train_targets: FloatArray,
    test_features: FloatArray,
    *,
    ridge: float,
) -> FloatArray:
    mean = np.mean(train_features, axis=0)
    scale = np.std(train_features, axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (train_features - mean) / scale
    design = np.column_stack(
        (standardized, np.ones(standardized.shape[0], dtype=np.float64))
    )
    penalty = ridge * np.eye(design.shape[1], dtype=np.float64)
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        design.T @ design + penalty,
        design.T @ train_targets,
    )
    test = np.append((test_features - mean) / scale, 1.0)
    return np.asarray(test @ coefficients, dtype=np.float64)


def _decision_unit_errors(
    values: FloatArray,
    family: str,
    actions: FloatArray,
    operator: FloatArray,
) -> FloatArray:
    decision = _project(values, graph_family(family), ReceiverKind.DECISION)
    errors = np.empty(decision.shape[0], dtype=np.float64)
    for unit_index, unit in enumerate(decision):
        current = unit[:, :-1]
        target = unit[:, 1:]
        predictors = np.concatenate(
            (current, np.broadcast_to(actions, (*current.shape[:-1], 2))),
            axis=-1,
        )
        prediction = affine_prediction(operator, predictors.reshape(-1, 4))
        errors[unit_index] = rms(target.reshape(-1, 2) - prediction)
    return errors


def topology_descriptor_holdout(
    *,
    development: Mapping[str, FloatArray],
    evaluation: Mapping[str, FloatArray],
    actions: FloatArray,
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
    ridge: float = 1e-3,
) -> dict[str, object]:
    """Compare registered global descriptors beyond a fixed covariate basis.

    A topology family is omitted from fitting, its decision operator is
    predicted from descriptors of the remaining families, and that prediction
    is scored on the omitted family's evaluation preparations.  This is a
    stringent topology-family extrapolation diagnostic, not a claim that the
    four extension coordinates form a sufficient invariant basis.
    """

    families = tuple(sorted(development))
    if families != tuple(sorted(evaluation)) or len(families) < 4:
        raise ValueError("descriptor holdout requires at least four matching families")
    descriptors = {family: _descriptor_vectors(family) for family in families}
    rows = []
    block_improvements = []
    for law_index, law in enumerate(("LINEAR", "NONLINEAR")):
        for normalization_index, normalization in enumerate(
            ("FIXED_PER_EDGE", "FIXED_TOTAL")
        ):
            models = _fit_models(
                development,
                actions,
                families,
                law_index,
                normalization_index,
            )
            family_operators = dict(cast(Mapping[str, FloatArray], models["family"]))
            family_improvements = []
            family_rows = []
            for left_out in families:
                retained = tuple(family for family in families if family != left_out)
                baseline_features = np.stack(
                    [descriptors[family][0] for family in retained]
                )
                extended_features = np.stack(
                    [descriptors[family][1] for family in retained]
                )
                targets = np.stack(
                    [family_operators[family].reshape(-1) for family in retained]
                )
                baseline_operator = _ridge_predict(
                    baseline_features,
                    targets,
                    descriptors[left_out][0],
                    ridge=ridge,
                ).reshape(5, 2)
                extended_operator = _ridge_predict(
                    extended_features,
                    targets,
                    descriptors[left_out][1],
                    ridge=ridge,
                ).reshape(5, 2)
                held_out_values = evaluation[left_out][law_index, normalization_index]
                baseline_errors = _decision_unit_errors(
                    held_out_values,
                    left_out,
                    actions,
                    baseline_operator,
                )
                extended_errors = _decision_unit_errors(
                    held_out_values,
                    left_out,
                    actions,
                    extended_operator,
                )
                improvement = baseline_errors - extended_errors
                family_improvements.append(float(np.mean(improvement)))
                family_rows.append(
                    {
                        "left_out_family": left_out,
                        "baseline_mean_error": float(np.mean(baseline_errors)),
                        "extended_mean_error": float(np.mean(extended_errors)),
                        "mean_improvement": float(np.mean(improvement)),
                        "independent_unit_improvements": tuple(
                            float(value) for value in improvement
                        ),
                    }
                )
            interval = bootstrap_mean_interval(
                np.asarray(family_improvements, dtype=np.float64),
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
                seed=seed + law_index * 10 + normalization_index,
            )
            block_improvements.append(interval)
            rows.append(
                {
                    "law": law,
                    "normalization": normalization,
                    "family_rows": tuple(family_rows),
                    "family_block_improvement": {
                        "estimate": interval[0],
                        "lower": interval[1],
                        "upper": interval[2],
                    },
                    "extension_supported": interval[1] > 0.0,
                }
            )
    return {
        "baseline_coordinate_ids": (
            "node-count",
            "degree-mean",
            "degree-sd",
            "laplacian-gap",
            "laplacian-maximum",
            "laplacian-second-moment",
            "port-distance",
            "boundary-fraction",
        ),
        "extension_coordinate_ids": (
            "cycle-rank-fraction",
            "triangle-density",
            "bridge-fraction",
            "diameter-fraction",
        ),
        "descriptor_records": tuple(descriptors[family][2] for family in families),
        "blocks": tuple(rows),
        "extension_supported_all_blocks": all(
            interval[1] > 0.0 for interval in block_improvements
        ),
        "topology_identity_used_as_predictor": False,
        "evaluation_model_selection_performed": False,
        "interpretation_ceiling": (
            "GENERATED_FAMILY_HOLDOUT_DESCRIPTOR_DIAGNOSTIC; "
            "FINITE_BASIS_NOT_PROVEN_SUFFICIENT"
        ),
    }


def _receiver_matrix(family: str, kind: ReceiverKind) -> FloatArray:
    graph = graph_family(family)
    if kind is ReceiverKind.FULL_STATE:
        return np.eye(graph.node_count, dtype=np.float64)
    if kind is ReceiverKind.BOUNDARY:
        return np.eye(graph.node_count, dtype=np.float64)[list(graph.boundary_nodes)]
    if kind is ReceiverKind.AGGREGATE:
        return np.full((1, graph.node_count), 1.0 / graph.node_count, dtype=np.float64)
    matrix = np.zeros((2, graph.node_count), dtype=np.float64)
    matrix[0, list(graph.boundary_nodes)] = 1.0 / len(graph.boundary_nodes)
    matrix[1, graph.port_a_node] = 1.0
    matrix[1, graph.port_b_node] = -1.0
    return matrix


def _full_state_operator(
    values: FloatArray,
    actions: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    current = values[:, :, :-1]
    target = values[:, :, 1:]
    delivered = np.broadcast_to(actions[np.newaxis, ...], (*current.shape[:-1], 2))
    predictors = np.concatenate((current, delivered), axis=-1)
    operator = fit_affine_operator(
        predictors.reshape(-1, predictors.shape[-1]),
        target.reshape(-1, target.shape[-1]),
    )
    node_count = values.shape[-1]
    return (
        np.asarray(operator[:node_count].T, dtype=np.float64),
        np.asarray(operator[node_count : node_count + 2].T, dtype=np.float64),
        operator,
    )


def _matrix_rank(values: FloatArray, relative_tolerance: float) -> tuple[int, tuple[float, ...]]:
    rank, singular = floor_certified_rank(
        values,
        absolute_floor=1e-8,
        relative_tolerance=relative_tolerance,
    )
    return rank, tuple(float(value) for value in singular)


def topology_system_geometry(
    *,
    development: Mapping[str, FloatArray],
    evaluation: Mapping[str, FloatArray],
    actions: FloatArray,
    rank_relative_tolerance: float,
) -> dict[str, object]:
    """Fit development-only full-state linearizations and grade their geometry."""

    families = tuple(sorted(development))
    rows = []
    for family in families:
        graph = graph_family(family)
        for law_index, law in enumerate(("LINEAR", "NONLINEAR")):
            for normalization_index, normalization in enumerate(
                ("FIXED_PER_EDGE", "FIXED_TOTAL")
            ):
                a_matrix, b_matrix, operator = _full_state_operator(
                    development[family][law_index, normalization_index],
                    actions,
                )
                powers = [np.eye(graph.node_count, dtype=np.float64)]
                for _ in range(1, graph.node_count):
                    powers.append(powers[-1] @ a_matrix)
                controllability = np.column_stack(
                    [power @ b_matrix for power in powers]
                )
                controllability_rank, controllability_singular = _matrix_rank(
                    controllability,
                    rank_relative_tolerance,
                )
                gramian = sum(
                    (power @ b_matrix) @ (power @ b_matrix).T for power in powers
                )
                gramian_eigenvalues = np.linalg.eigvalsh(gramian)
                receiver_rows = []
                for kind in ReceiverKind:
                    c_matrix = _receiver_matrix(family, kind)
                    observability = np.vstack(
                        [c_matrix @ power for power in powers]
                    )
                    observable_rank, observable_singular = _matrix_rank(
                        observability,
                        rank_relative_tolerance,
                    )
                    receiver_rows.append(
                        {
                            "receiver": kind.value,
                            "observable_rank": observable_rank,
                            "latent_dimension": graph.node_count,
                            "observable_fraction": observable_rank / graph.node_count,
                            "singular_values": observable_singular,
                            "aggregate_rms_coordinate_omitted_from_linear_test": (
                                kind is ReceiverKind.AGGREGATE
                            ),
                        }
                    )
                evaluation_errors = []
                for unit in evaluation[family][law_index, normalization_index]:
                    current = unit[:, :-1]
                    target = unit[:, 1:]
                    delivered = np.broadcast_to(actions, (*current.shape[:-1], 2))
                    predictors = np.concatenate((current, delivered), axis=-1)
                    evaluation_errors.append(
                        rms(
                            target.reshape(-1, graph.node_count)
                            - affine_prediction(
                                operator,
                                predictors.reshape(-1, graph.node_count + 2),
                            )
                        )
                    )
                rows.append(
                    {
                        "family": family,
                        "law": law,
                        "normalization": normalization,
                        "controllable_rank": controllability_rank,
                        "latent_dimension": graph.node_count,
                        "input_dimension": 2,
                        "input_kernel_dimension": 2
                        - _matrix_rank(b_matrix, rank_relative_tolerance)[0],
                        "controllability_singular_values": controllability_singular,
                        "finite_horizon_gramian_eigenvalues": tuple(
                            float(value) for value in gramian_eigenvalues
                        ),
                        "receiver_observability": tuple(receiver_rows),
                        "evaluation_recurrence_rms": tuple(evaluation_errors),
                        "nonlinear_row_is_global_empirical_linearization": law == "NONLINEAR",
                    }
                )
    return {
        "rows": tuple(rows),
        "reachable_geometry": "FINITE_LINEARIZED_CONTROLLABILITY_GEOMETRY",
        "admitted_set_geometry": "UNEVALUABLE_NO_RECEIVER_ADMISSION_SPECIFICATION",
        "rank_absolute_tolerance": 1e-8,
        "rank_relative_tolerance": rank_relative_tolerance,
        "evaluation_refit_performed": False,
    }


def topology_action_fibres_and_loops(
    *,
    evaluation: Mapping[str, FloatArray],
    word_ids: tuple[str, ...],
    equivalence_floor: float,
    rank_relative_tolerance: float,
) -> dict[str, object]:
    """Measure finite action fibres and registered inverse-loop residuals."""

    dose_indices = {
        name: _word_index(word_ids, word)
        for name, word in (
            ("a-negative", "word.a-early.dose-m0p25"),
            ("a-positive", "word.a-early.dose-0p25"),
            ("b-negative", "word.b-early.dose-m0p25"),
            ("b-positive", "word.b-early.dose-0p25"),
        )
    }
    loop_indices = {
        port: _word_index(word_ids, f"word.{port}-reverse.dose-1p0")
        for port in ("a", "b")
    }
    identity_index = _word_index(word_ids, "word.identity")
    rows = []
    for family in sorted(evaluation):
        graph = graph_family(family)
        for law_index, law in enumerate(("LINEAR", "NONLINEAR")):
            for normalization_index, normalization in enumerate(
                ("FIXED_PER_EDGE", "FIXED_TOTAL")
            ):
                decision = _project(
                    evaluation[family][law_index, normalization_index],
                    graph,
                    ReceiverKind.DECISION,
                )
                response = decision - decision[:, identity_index : identity_index + 1]
                ranks = []
                fibre_pairs = []
                labels = tuple(dose_indices)
                final_signatures = np.stack(
                    [response[:, dose_indices[label], -1] for label in labels],
                    axis=1,
                )
                for unit in range(response.shape[0]):
                    derivative = np.column_stack(
                        (
                            (
                                final_signatures[unit, labels.index("a-positive")]
                                - final_signatures[unit, labels.index("a-negative")]
                            )
                            / 0.5,
                            (
                                final_signatures[unit, labels.index("b-positive")]
                                - final_signatures[unit, labels.index("b-negative")]
                            )
                            / 0.5,
                        )
                    )
                    rank, _ = floor_certified_rank(
                        derivative,
                        absolute_floor=equivalence_floor,
                        relative_tolerance=rank_relative_tolerance,
                    )
                    ranks.append(rank)
                for left_index, left in enumerate(labels):
                    for right in labels[left_index + 1 :]:
                        defects = np.sqrt(
                            np.mean(
                                np.square(
                                    final_signatures[:, labels.index(left)]
                                    - final_signatures[:, labels.index(right)]
                                ),
                                axis=1,
                            )
                        )
                        fibre_pairs.append(
                            {
                                "left_action": left,
                                "right_action": right,
                                "mean_defect": float(np.mean(defects)),
                                "equivalent_unit_fraction": float(
                                    np.mean(defects <= equivalence_floor)
                                ),
                            }
                        )
                loop_residuals = {
                    port: tuple(
                        float(value)
                        for value in np.sqrt(
                            np.mean(
                                np.square(response[:, index]),
                                axis=(1, 2),
                            )
                        )
                    )
                    for port, index in loop_indices.items()
                }
                rows.append(
                    {
                        "family": family,
                        "law": law,
                        "normalization": normalization,
                        "rank_counts": {
                            str(rank): ranks.count(rank) for rank in sorted(set(ranks))
                        },
                        "finite_action_fibres": tuple(fibre_pairs),
                        "inverse_loop_residuals": loop_residuals,
                        "loop_is_expected_to_close_exactly": False,
                    }
                )
    return {
        "rows": tuple(rows),
        "equivalence_floor": equivalence_floor,
        "kernel_interpretation": (
            "LOCAL_FINITE_DIFFERENCE_ACTION_KERNEL_AT_DECISION_RECEIVER"
        ),
        "loop_interpretation": (
            "DISSIPATIVE_TIMING-CONTROLLED RESIDUAL; NOT A GROUP-INVERSE TEST"
        ),
    }


def topology_law_leave_out_sensitivity(
    *,
    development: Mapping[str, FloatArray],
    evaluation: Mapping[str, FloatArray],
    actions: FloatArray,
) -> dict[str, object]:
    """Assess conclusion stability without refitting scientific thresholds.

    Family deletion removes that family's evaluation contribution. Unit
    deletion removes the same matched preparation index across every family.
    Operators remain development-only; no evaluation outcome selects a model.
    """

    families = tuple(sorted(development))
    if families != tuple(sorted(evaluation)) or len(families) < 3:
        raise ValueError("leave-out sensitivity requires matching topology families")
    unit_count = next(iter(evaluation.values())).shape[2]
    family_rows = []
    unit_rows = []
    for law_index, law in enumerate(("LINEAR", "NONLINEAR")):
        for normalization_index, normalization in enumerate(
            ("FIXED_PER_EDGE", "FIXED_TOTAL")
        ):
            models = _fit_models(
                development,
                actions,
                families,
                law_index,
                normalization_index,
            )
            errors = _operator_errors(
                evaluation,
                actions,
                families,
                law_index,
                normalization_index,
                models,
            )
            paired = errors["single-stationary"] - errors["topology-conditioned"]
            for family_index, family in enumerate(families):
                retained = np.delete(paired, family_index, axis=0)
                family_rows.append(
                    {
                        "law": law,
                        "normalization": normalization,
                        "left_out_family": family,
                        "retained_mean_improvement": float(np.mean(retained)),
                        "retained_unit_block_improvements": tuple(
                            float(value) for value in np.mean(retained, axis=0)
                        ),
                        "conclusion_positive": bool(np.mean(retained) > 0.0),
                    }
                )
            for unit_index in range(unit_count):
                retained = np.delete(paired, unit_index, axis=1)
                unit_rows.append(
                    {
                        "law": law,
                        "normalization": normalization,
                        "left_out_unit_index": unit_index,
                        "retained_mean_improvement": float(np.mean(retained)),
                        "conclusion_positive": bool(np.mean(retained) > 0.0),
                    }
                )
    return {
        "leave_one_family": tuple(family_rows),
        "leave_one_matched_independent_unit": tuple(unit_rows),
        "all_family_deletions_positive": all(
            bool(row["conclusion_positive"]) for row in family_rows
        ),
        "all_unit_deletions_positive": all(
            bool(row["conclusion_positive"]) for row in unit_rows
        ),
        "threshold_refit_performed": False,
        "evaluation_model_selection_performed": False,
    }
