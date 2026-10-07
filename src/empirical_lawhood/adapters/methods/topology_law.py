"""Prospective estimators for topology-conditioned generated response laws."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.reference_worlds.topological_response import (
    GraphDescriptor,
    ReceiverKind,
    graph_family,
)

from .response_formalization import (
    affine_prediction,
    bootstrap_mean_interval,
    fit_affine_operator,
    floor_certified_rank,
    operator_concentration,
    rms,
)


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class FrozenTopologyMethod:
    equivalence_floor: float
    materiality_floor: float
    rank_relative_tolerance: float
    relabel_relative_tolerance: float
    numerical_guard: float
    bootstrap_resamples: int
    confidence_level: float
    bootstrap_seed: int

    def __post_init__(self) -> None:
        if not 0.0 < self.equivalence_floor < self.materiality_floor:
            raise ValueError("topology method floors are not ordered")
        if min(
            self.rank_relative_tolerance,
            self.relabel_relative_tolerance,
            self.numerical_guard,
        ) <= 0.0:
            raise ValueError("topology method tolerances must be positive")
        if self.bootstrap_resamples < 100 or not 0.5 < self.confidence_level < 1.0:
            raise ValueError("topology bootstrap controls are invalid")


def _validate_arrays(
    arrays: Mapping[str, FloatArray],
    *,
    unit_count: int,
    word_count: int,
) -> None:
    if not arrays:
        raise ValueError("topology analysis requires at least one family")
    for family, values in arrays.items():
        expected_nodes = graph_family(family).node_count
        if values.shape != (2, 2, unit_count, word_count, 81, expected_nodes):
            raise ValueError(f"topology array shape differs for {family}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"topology array contains nonfinite values for {family}")


def _project(values: FloatArray, graph: GraphDescriptor, kind: ReceiverKind) -> FloatArray:
    if values.shape[-1] != graph.node_count:
        raise ValueError("receiver projection and graph dimensions differ")
    if kind is ReceiverKind.FULL_STATE:
        return values.copy()
    if kind is ReceiverKind.BOUNDARY:
        return values[..., graph.boundary_nodes]
    if kind is ReceiverKind.AGGREGATE:
        return np.stack(
            (
                np.mean(values, axis=-1),
                np.sqrt(np.mean(np.square(values), axis=-1)),
            ),
            axis=-1,
        )
    return np.stack(
        (
            np.mean(values[..., graph.boundary_nodes], axis=-1),
            values[..., graph.port_a_node] - values[..., graph.port_b_node],
        ),
        axis=-1,
    )


def _word_index(word_ids: Sequence[str], prefix: str) -> int:
    try:
        return next(index for index, value in enumerate(word_ids) if value == prefix)
    except StopIteration as error:
        raise ValueError(f"required word is absent: {prefix}") from error


def _transition_rows(
    arrays: Mapping[str, FloatArray],
    actions: FloatArray,
    families: Sequence[str],
    law_index: int,
    normalization_index: int,
    time_indices: npt.NDArray[np.int64] | None = None,
) -> tuple[FloatArray, FloatArray]:
    states = []
    targets = []
    for family in families:
        graph = graph_family(family)
        decision = _project(
            arrays[family][law_index, normalization_index],
            graph,
            ReceiverKind.DECISION,
        )
        current = decision[:, :, :-1]
        target = decision[:, :, 1:]
        delivered = np.broadcast_to(actions[np.newaxis, ...], (*current.shape[:-1], 2))
        if time_indices is not None:
            current = current[:, :, time_indices]
            target = target[:, :, time_indices]
            delivered = delivered[:, :, time_indices]
        states.append(
            np.column_stack(
                (
                    current.reshape(-1, 2),
                    delivered.reshape(-1, 2),
                )
            )
        )
        targets.append(target.reshape(-1, 2))
    return (
        np.asarray(np.vstack(states), dtype=np.float64),
        np.asarray(np.vstack(targets), dtype=np.float64),
    )


def _fit_models(
    development: Mapping[str, FloatArray],
    actions: FloatArray,
    families: tuple[str, ...],
    law_index: int,
    normalization_index: int,
) -> dict[str, object]:
    states, targets = _transition_rows(
        development,
        actions,
        families,
        law_index,
        normalization_index,
    )
    stationary = fit_affine_operator(states, targets)
    time_bins = tuple(np.arange(start, start + 20, dtype=np.int64) for start in range(0, 80, 20))
    cocycle = []
    for indices in time_bins:
        bin_states, bin_targets = _transition_rows(
            development,
            actions,
            families,
            law_index,
            normalization_index,
            indices,
        )
        cocycle.append(fit_affine_operator(bin_states, bin_targets))
    family_operators = {}
    for family in families:
        family_states, family_targets = _transition_rows(
            development,
            actions,
            (family,),
            law_index,
            normalization_index,
        )
        family_operators[family] = fit_affine_operator(family_states, family_targets)
    vectors = np.stack([family_operators[family].reshape(-1) for family in families])
    first = 0
    second = int(np.argmax(np.linalg.norm(vectors - vectors[first], axis=1)))
    centroids = np.stack((vectors[first], vectors[second]))
    assignments = np.zeros(len(families), dtype=np.int64)
    for _ in range(20):
        distances = np.linalg.norm(vectors[:, np.newaxis] - centroids[np.newaxis], axis=2)
        updated = np.argmin(distances, axis=1)
        if np.array_equal(updated, assignments) and _ > 0:
            break
        assignments = updated
        for component in range(2):
            members = vectors[assignments == component]
            if members.size:
                centroids[component] = np.mean(members, axis=0)
    mixture_operators = []
    for component in range(2):
        component_families = tuple(
            family for family, assignment in zip(families, assignments, strict=True)
            if int(assignment) == component
        )
        component_states, component_targets = _transition_rows(
            development,
            actions,
            component_families,
            law_index,
            normalization_index,
        )
        mixture_operators.append(fit_affine_operator(component_states, component_targets))
    high_capacity: dict[str, tuple[FloatArray, ...]] = {}
    for family in families:
        rows = []
        for indices in time_bins:
            family_states, family_targets = _transition_rows(
                development,
                actions,
                (family,),
                law_index,
                normalization_index,
                indices,
            )
            rows.append(fit_affine_operator(family_states, family_targets))
        high_capacity[family] = tuple(rows)
    return {
        "stationary": stationary,
        "cocycle": tuple(cocycle),
        "family": family_operators,
        "mixture": tuple(mixture_operators),
        "mixture_assignment": {
            family: int(assignment)
            for family, assignment in zip(families, assignments, strict=True)
        },
        "high_capacity": high_capacity,
        "concentration": operator_concentration(
            tuple(family_operators[family] for family in families),
            temperature=0.05,
        ),
    }


def _operator_errors(
    evaluation: Mapping[str, FloatArray],
    actions: FloatArray,
    families: tuple[str, ...],
    law_index: int,
    normalization_index: int,
    models: Mapping[str, object],
) -> dict[str, FloatArray]:
    unit_count = next(iter(evaluation.values())).shape[2]
    names = (
        "persistence",
        "single-stationary",
        "absolute-clock-cocycle",
        "two-component-mixture",
        "topology-conditioned",
        "high-capacity-diagnostic",
    )
    result = {
        name: np.empty((len(families), unit_count), dtype=np.float64) for name in names
    }
    stationary = np.asarray(models["stationary"], dtype=np.float64)
    cocycle_values = cast(tuple[FloatArray, ...], models["cocycle"])
    cocycle = tuple(np.asarray(value, dtype=np.float64) for value in cocycle_values)
    family_operators = dict(cast(Mapping[str, FloatArray], models["family"]))
    mixture_values = cast(tuple[FloatArray, ...], models["mixture"])
    mixture_operators = tuple(
        np.asarray(value, dtype=np.float64) for value in mixture_values
    )
    assignment = dict(cast(Mapping[str, int], models["mixture_assignment"]))
    high_capacity = dict(
        cast(Mapping[str, tuple[FloatArray, ...]], models["high_capacity"])
    )
    for family_index, family in enumerate(families):
        graph = graph_family(family)
        decision = _project(
            evaluation[family][law_index, normalization_index],
            graph,
            ReceiverKind.DECISION,
        )
        for unit_index in range(unit_count):
            current = decision[unit_index, :, :-1]
            target = decision[unit_index, :, 1:]
            delivered = actions
            predictors = np.concatenate((current, delivered), axis=-1)
            result["persistence"][family_index, unit_index] = rms(target - current)
            result["single-stationary"][family_index, unit_index] = rms(
                target.reshape(-1, 2)
                - affine_prediction(stationary, predictors.reshape(-1, 4))
            )
            family_operator = np.asarray(family_operators[family], dtype=np.float64)
            result["topology-conditioned"][family_index, unit_index] = rms(
                target.reshape(-1, 2)
                - affine_prediction(family_operator, predictors.reshape(-1, 4))
            )
            component = int(assignment[family])
            result["two-component-mixture"][family_index, unit_index] = rms(
                target.reshape(-1, 2)
                - affine_prediction(mixture_operators[component], predictors.reshape(-1, 4))
            )
            cocycle_prediction = np.empty_like(target)
            high_prediction = np.empty_like(target)
            family_high = tuple(
                np.asarray(value, dtype=np.float64) for value in high_capacity[family]
            )
            for bin_index, start in enumerate(range(0, 80, 20)):
                indices = slice(start, start + 20)
                cocycle_prediction[:, indices] = affine_prediction(
                    cocycle[bin_index],
                    predictors[:, indices].reshape(-1, 4),
                ).reshape(predictors.shape[0], 20, 2)
                high_prediction[:, indices] = affine_prediction(
                    family_high[bin_index],
                    predictors[:, indices].reshape(-1, 4),
                ).reshape(predictors.shape[0], 20, 2)
            result["absolute-clock-cocycle"][family_index, unit_index] = rms(
                target - cocycle_prediction
            )
            result["high-capacity-diagnostic"][family_index, unit_index] = rms(
                target - high_prediction
            )
    return result


def _bootstrap_row(
    values: FloatArray,
    method: FrozenTopologyMethod,
    seed_offset: int,
) -> dict[str, float]:
    estimate, lower, upper = bootstrap_mean_interval(
        np.asarray(values, dtype=np.float64),
        resamples=method.bootstrap_resamples,
        confidence_level=method.confidence_level,
        seed=method.bootstrap_seed + seed_offset,
    )
    return {"estimate": estimate, "lower": lower, "upper": upper}


def _model_assessment(
    development: Mapping[str, FloatArray],
    evaluation: Mapping[str, FloatArray],
    actions: FloatArray,
    families: tuple[str, ...],
    method: FrozenTopologyMethod,
) -> tuple[dict[str, object], ...]:
    rows = []
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
            summaries = {
                name: _bootstrap_row(
                    np.mean(values, axis=0),
                    method,
                    law_index * 100 + normalization_index * 10 + index,
                )
                for index, (name, values) in enumerate(sorted(errors.items()))
            }
            improvement = np.mean(
                errors["single-stationary"] - errors["topology-conditioned"],
                axis=0,
            )
            improvement_interval = _bootstrap_row(
                improvement,
                method,
                500 + law_index * 10 + normalization_index,
            )
            rows.append(
                {
                    "law": law,
                    "normalization": normalization,
                    "model_errors": summaries,
                    "topology_conditioning_improvement": improvement_interval,
                    "topology_conditioning_supported": (
                        improvement_interval["lower"] > 0.0
                    ),
                    "mixture_assignment": models["mixture_assignment"],
                    "operator_concentration": models["concentration"],
                    "high_capacity_is_diagnostic_only": True,
                }
            )
    return tuple(rows)


def _decision_response(
    values: FloatArray,
    family: str,
    law_index: int,
    normalization_index: int,
) -> FloatArray:
    projected = _project(
        values[law_index, normalization_index],
        graph_family(family),
        ReceiverKind.DECISION,
    )
    return projected - projected[:, :1]


def _paired_topology_exchange(
    evaluation: Mapping[str, FloatArray],
    word_ids: Sequence[str],
    left_family: str,
    right_family: str,
    method: FrozenTopologyMethod,
    seed_offset: int,
) -> tuple[dict[str, object], ...]:
    selected_words = tuple(
        _word_index(word_ids, word)
        for word in (
            "word.a-early.dose-1p0",
            "word.b-early.dose-1p0",
            "word.a-then-b.dose-1p0",
        )
        if word in word_ids
    )
    if not selected_words:
        selected_words = (
            _word_index(word_ids, "word.a-early.dose-1"),
            _word_index(word_ids, "word.b-early.dose-1"),
            _word_index(word_ids, "word.a-then-b.dose-1"),
        )
    rows = []
    for law_index, law in enumerate(("LINEAR", "NONLINEAR")):
        for normalization_index, normalization in enumerate(
            ("FIXED_PER_EDGE", "FIXED_TOTAL")
        ):
            left = _decision_response(
                evaluation[left_family],
                left_family,
                law_index,
                normalization_index,
            )
            right = _decision_response(
                evaluation[right_family],
                right_family,
                law_index,
                normalization_index,
            )
            unit_effect = np.sqrt(
                np.mean(
                    np.square(left[:, selected_words] - right[:, selected_words]),
                    axis=(1, 2, 3),
                )
            )
            interval = _bootstrap_row(
                np.asarray(unit_effect, dtype=np.float64),
                method,
                seed_offset + law_index * 10 + normalization_index,
            )
            rows.append(
                {
                    "law": law,
                    "normalization": normalization,
                    "effect": interval,
                    "material": interval["lower"] > method.materiality_floor,
                }
            )
    return tuple(rows)


def _controlled_order(
    evaluation: Mapping[str, FloatArray],
    word_ids: Sequence[str],
    families: tuple[str, ...],
    method: FrozenTopologyMethod,
) -> tuple[dict[str, object], ...]:
    indices = {
        name: _word_index(word_ids, f"word.{name}.dose-1p0")
        for name in (
            "a-early",
            "a-late",
            "b-early",
            "b-late",
            "a-then-b",
            "b-then-a",
        )
    }
    rows = []
    for family_index, family in enumerate(families):
        for law_index, law in enumerate(("LINEAR", "NONLINEAR")):
            for normalization_index, normalization in enumerate(
                ("FIXED_PER_EDGE", "FIXED_TOTAL")
            ):
                response = _decision_response(
                    evaluation[family],
                    family,
                    law_index,
                    normalization_index,
                )
                order = response[:, indices["a-then-b"]] - response[:, indices["b-then-a"]]
                timing = (
                    response[:, indices["a-early"]]
                    - response[:, indices["a-late"]]
                    - response[:, indices["b-early"]]
                    + response[:, indices["b-late"]]
                )
                unit_defect = np.sqrt(np.mean(np.square(order - timing), axis=(1, 2)))
                interval = _bootstrap_row(
                    np.asarray(unit_defect, dtype=np.float64),
                    method,
                    1_000 + family_index * 20 + law_index * 2 + normalization_index,
                )
                rows.append(
                    {
                        "family": family,
                        "law": law,
                        "normalization": normalization,
                        "controlled_order": interval,
                        "material": interval["lower"] > method.materiality_floor,
                    }
                )
    return tuple(rows)


def _action_rank(
    evaluation: Mapping[str, FloatArray],
    word_ids: Sequence[str],
    families: tuple[str, ...],
    method: FrozenTopologyMethod,
) -> tuple[dict[str, object], ...]:
    rows = []
    for family in families:
        indices = {
            key: _word_index(word_ids, f"word.{port}-early.dose-{token}")
            for key, port, token in (
                ("a-positive", "a", "0p25"),
                ("a-negative", "a", "m0p25"),
                ("b-positive", "b", "0p25"),
                ("b-negative", "b", "m0p25"),
            )
        }
        for law_index, law in enumerate(("LINEAR", "NONLINEAR")):
            for normalization_index, normalization in enumerate(
                ("FIXED_PER_EDGE", "FIXED_TOTAL")
            ):
                response = _decision_response(
                    evaluation[family],
                    family,
                    law_index,
                    normalization_index,
                )
                ranks = []
                singular_values = []
                for unit in response:
                    derivative_a = (
                        unit[indices["a-positive"], -1]
                        - unit[indices["a-negative"], -1]
                    ) / 0.5
                    derivative_b = (
                        unit[indices["b-positive"], -1]
                        - unit[indices["b-negative"], -1]
                    ) / 0.5
                    rank, singular = floor_certified_rank(
                        np.column_stack((derivative_a, derivative_b)),
                        absolute_floor=method.equivalence_floor,
                        relative_tolerance=method.rank_relative_tolerance,
                    )
                    ranks.append(rank)
                    singular_values.append(tuple(float(value) for value in singular))
                rows.append(
                    {
                        "family": family,
                        "law": law,
                        "normalization": normalization,
                        "rank_counts": {
                            str(rank): ranks.count(rank) for rank in sorted(set(ranks))
                        },
                        "rank_two_fraction": ranks.count(2) / len(ranks),
                        "singular_values": tuple(singular_values),
                    }
                )
    return tuple(rows)


def _receiver_classification(
    development: Mapping[str, FloatArray],
    evaluation: Mapping[str, FloatArray],
    word_ids: Sequence[str],
    families: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    selected = tuple(
        _word_index(word_ids, word)
        for word in (
            "word.a-early.dose-1p0",
            "word.b-early.dose-1p0",
            "word.a-then-b.dose-1p0",
            "word.b-then-a.dose-1p0",
            "word.a-reverse.dose-1p0",
        )
    )

    def signatures(values: FloatArray, graph: GraphDescriptor, kind: ReceiverKind) -> FloatArray:
        projected = _project(values, graph, kind)
        response = projected - projected[:, :1]
        chosen = response[:, selected]
        return np.column_stack(
            (
                np.sqrt(np.mean(np.square(chosen[:, :, -1]), axis=-1)),
                np.sqrt(np.mean(np.square(chosen), axis=(2, 3))),
                np.max(np.abs(chosen), axis=(2, 3)),
            )
        )

    rows = []
    for kind in (
        ReceiverKind.FULL_STATE,
        ReceiverKind.BOUNDARY,
        ReceiverKind.AGGREGATE,
        ReceiverKind.DECISION,
    ):
        correct = 0
        total = 0
        for law_index in range(2):
            for normalization_index in range(2):
                centroids = {}
                for family in families:
                    graph = graph_family(family)
                    centroids[family] = np.mean(
                        signatures(
                            development[family][law_index, normalization_index],
                            graph,
                            kind,
                        ),
                        axis=0,
                    )
                for family in families:
                    graph = graph_family(family)
                    observed = signatures(
                        evaluation[family][law_index, normalization_index],
                        graph,
                        kind,
                    )
                    for signature in observed:
                        predicted = min(
                            families,
                            key=lambda candidate: float(
                                np.linalg.norm(signature - centroids[candidate])
                            ),
                        )
                        correct += predicted == family
                        total += 1
        rows.append(
            {
                "receiver": kind.value,
                "topology_classification_accuracy": correct / total,
                "correct": correct,
                "total": total,
                "selection_role": "FROZEN_FINITE_RECEIVER_HIERARCHY",
            }
        )
    return tuple(rows)


def analyse_topology_law(
    *,
    development: Mapping[str, FloatArray],
    evaluation: Mapping[str, FloatArray],
    actions: FloatArray,
    word_ids: tuple[str, ...],
    relabel_controls: Mapping[str, FloatArray],
    cut_controls: Mapping[str, FloatArray],
    glue_control: FloatArray,
    method: FrozenTopologyMethod,
) -> dict[str, object]:
    families = tuple(sorted(development))
    _validate_arrays(development, unit_count=8, word_count=len(word_ids))
    _validate_arrays(evaluation, unit_count=8, word_count=len(word_ids))
    if actions.shape != (len(word_ids), 80, 2):
        raise ValueError("topology action array shape differs")
    delivery_equal = bool(np.all(np.isfinite(actions)))
    identity_index = _word_index(word_ids, "word.identity")
    identity_actions_zero = bool(np.all(actions[identity_index] == 0.0))
    relabel_rows = []
    maximum_relabel_defect = 0.0
    for family in families:
        values = relabel_controls[family]
        if values.shape[:4] != (2, 2, 2, 3):
            raise ValueError("relabel control shape differs")
        defect = float(np.max(np.abs(values[0] - values[1])))
        scale = max(float(np.max(np.abs(values[0]))), 1.0)
        relative = defect / scale
        maximum_relabel_defect = max(maximum_relabel_defect, relative)
        relabel_rows.append(
            {
                "family": family,
                "maximum_absolute_defect": defect,
                "maximum_relative_defect": relative,
                "passed": relative <= method.relabel_relative_tolerance,
            }
        )
    model_rows = _model_assessment(
        development,
        evaluation,
        actions,
        families,
        method,
    )
    exchanges = {
        "degree-matched-cube-vs-mobius": _paired_topology_exchange(
            evaluation,
            word_ids,
            "cube",
            "mobius-ladder",
            method,
            2_000,
        ),
        "boundary-open-vs-periodic-grid": _paired_topology_exchange(
            evaluation,
            word_ids,
            "open-grid",
            "periodic-grid",
            method,
            2_100,
        ),
        "path-vs-cycle": _paired_topology_exchange(
            evaluation,
            word_ids,
            "path",
            "cycle",
            method,
            2_200,
        ),
        "tree-vs-cycle": _paired_topology_exchange(
            evaluation,
            word_ids,
            "balanced-tree",
            "cycle",
            method,
            2_300,
        ),
        "modular-vs-cycle": _paired_topology_exchange(
            evaluation,
            word_ids,
            "modular-bridge",
            "cycle",
            method,
            2_400,
        ),
    }
    cut_rows = []
    for family in families:
        values = cut_controls[family]
        unit_effect = float(np.sqrt(np.mean(np.square(values[1] - values[0]))))
        cut_rows.append(
            {
                "family": family,
                "paired_cut_response_rms": unit_effect,
                "material": unit_effect > method.materiality_floor,
                "independent_unit_count": 1,
                "uncertainty_status": "SINGLE_REGISTERED_INTERVENTION_NO_POPULATION_CI",
            }
        )
    if glue_control.shape != (2, 2, 2, 8, 3, 81, 16):
        raise ValueError("glue control shape differs")
    glue_rows = []
    for law_index, law in enumerate(("LINEAR", "NONLINEAR")):
        for normalization_index, normalization in enumerate(
            ("FIXED_PER_EDGE", "FIXED_TOTAL")
        ):
            effects = np.sqrt(
                np.mean(
                    np.square(
                        glue_control[0, law_index, normalization_index]
                        - glue_control[1, law_index, normalization_index]
                    ),
                    axis=(1, 2, 3),
                )
            )
            interval = _bootstrap_row(
                np.asarray(effects, dtype=np.float64),
                method,
                2_500 + law_index * 10 + normalization_index,
            )
            glue_rows.append(
                {
                    "law": law,
                    "normalization": normalization,
                    "interface_effect": interval,
                    "material": interval["lower"] > method.materiality_floor,
                }
            )
    controlled_order = _controlled_order(
        evaluation,
        word_ids,
        families,
        method,
    )
    action_rank = _action_rank(evaluation, word_ids, families, method)
    receiver_rows = _receiver_classification(
        development,
        evaluation,
        word_ids,
        families,
    )
    topology_model_supported = all(
        bool(row["topology_conditioning_supported"]) for row in model_rows
    )
    material_exchange_count = sum(
        bool(row["material"])
        for exchange_rows in exchanges.values()
        for row in exchange_rows
    )
    relabel_pass = all(bool(row["passed"]) for row in relabel_rows)
    topology_conditioned = relabel_pass and material_exchange_count > 0
    adjudications = (
        {
            "topology_type": "exact-relabel",
            "disposition": "SUPPORTED_INVARIANCE" if relabel_pass else "OPPOSED_METHOD_BINDING",
        },
        {
            "topology_type": "degree-matched-global-connectivity",
            "disposition": "SUPPORTED_MATERIAL" if any(
                bool(row["material"])
                for row in exchanges["degree-matched-cube-vs-mobius"]
            ) else "NULL_AT_FROZEN_FLOOR",
        },
        {
            "topology_type": "open-vs-periodic-boundary",
            "disposition": "SUPPORTED_MATERIAL" if any(
                bool(row["material"])
                for row in exchanges["boundary-open-vs-periodic-grid"]
            ) else "NULL_AT_FROZEN_FLOOR",
        },
        {
            "topology_type": "edge-cut-fault",
            "disposition": "SUPPORTED_MATERIAL" if any(
                bool(row["material"]) for row in cut_rows
            ) else "NULL_AT_FROZEN_FLOOR",
        },
        {
            "topology_type": "module-glue-interface",
            "disposition": "SUPPORTED_MATERIAL" if any(
                bool(row["material"]) for row in glue_rows
            ) else "NULL_AT_FROZEN_FLOOR",
        },
        {
            "topology_type": "laplacian-isospectral-nonisomorphic",
            "disposition": "UNEVALUABLE_RESPONSE_DESCRIPTOR_CONTROL_ONLY",
        },
    )
    mixture_law_supported = any(
        float(
            cast(Mapping[str, float], row["operator_concentration"])[
                "effective_operator_count"
            ]
        )
        > 1.5
        for row in model_rows
    )
    return {
        'measurement_delivery_checks': {
            "delivery_stages_equal": delivery_equal,
            "identity_actions_zero": identity_actions_zero,
            "all_arrays_finite": True,
            "passed": delivery_equal and identity_actions_zero,
        },
        "relabel_controls": tuple(relabel_rows),
        "maximum_relabel_relative_defect": maximum_relabel_defect,
        "operator_models": model_rows,
        "topology_exchanges": exchanges,
        "controlled_order": controlled_order,
        "action_rank": action_rank,
        "receiver_hierarchy": receiver_rows,
        "receiver_pushforward_naturality": "EXACT_BY_REGISTERED_PROJECTION",
        "dynamic_quotient_claim": "UNEVALUABLE_WITHOUT_LUMPABILITY_TEST",
        "cut_faults": tuple(cut_rows),
        "glue_interfaces": tuple(glue_rows),
        "adjudications": adjudications,
        "topology_conditioned_operator_improvement_all_blocks": topology_model_supported,
        "material_exchange_block_count": material_exchange_count,
        "topology_conditioned_response_supported": topology_conditioned,
        "stable_mixture_valued_law_supported": mixture_law_supported,
        "cross_substrate_claim": False,
        "physical_topology_claim": False,
        "scientific_ceiling": "GENERATED_TRUTH_KNOWN_TOPOLOGY_CONDITIONED_LOCAL_LAW",
    }
