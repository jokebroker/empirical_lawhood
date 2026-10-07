"Truth-known response transport, naturality, and approximate-gluing methods.\n\nThe independent unit is one complete prepared graph world.  Nested words,\ntimes, nodes, receivers, and numerical views are never treated as replicates.\nThe module reuses ``StructuralTransportWitness`` and adds only adapter-local\nnumerical records represented as plain canonical mappings by the composition runner.\n"

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.reference_worlds.topological_response import (
    CouplingNormalization,
    LocalLawKind,
    WorldSplit,
    action_word,
    build_world,
    cut_graph,
    graph_family,
    relabel_graph,
    requested_action,
    simulate_word_family,
)
from empirical_lawhood.kernel.thermodynamic_response import (
    CompositeSignatureAxis,
    FiniteAxisDisposition,
    StructuralTransportWitness,
    TransportDisposition,
)
from empirical_lawhood.kernel.worlds import WorldKind

from .response_formalization import bootstrap_mean_interval, rms


FloatArray = npt.NDArray[np.float64]

RESPONSE_TRANSPORT_VIEW_IDS = ("base", "relabel-native", "refined-sampled", "cut")
RESPONSE_TRANSPORT_WORD_IDS = ("identity", "a-early", "b-early", "a-then-b", "b-then-a")
RESPONSE_TRANSPORT_RELABEL_MAP = (3, 6, 1, 5, 0, 7, 2, 4)
RESPONSE_TRANSPORT_BOUNDARY_NODES = (0, 7)
RESPONSE_TRANSPORT_OVERLAP_NODES = (3, 4)
RESPONSE_TRANSPORT_HORIZON_STEPS = 80


def response_transport_words() -> tuple[Any, ...]:
    return tuple(action_word(word, 1.0, RESPONSE_TRANSPORT_HORIZON_STEPS) for word in RESPONSE_TRANSPORT_WORD_IDS)


def response_transport_actions() -> FloatArray:
    return np.stack([requested_action(word) for word in response_transport_words()])


def generate_transport_corpus(
    *,
    split: WorldSplit,
    independent_unit_count: int,
    seed: int,
) -> FloatArray:
    """Generate base/relabel/refinement/cut views for disjoint preparations."""

    if independent_unit_count < 2:
        raise ValueError("Response transport and gluing requires at least two independent prepared worlds")
    path = graph_family("path", CouplingNormalization.FIXED_PER_EDGE)
    relabelled = relabel_graph(path, RESPONSE_TRANSPORT_RELABEL_MAP, 'response-transport-gluing')
    cut, _operation = cut_graph(path, ((3, 4),), 'response-transport-interface')
    words = response_transport_words()
    arrays = np.empty(
        (
            len(RESPONSE_TRANSPORT_VIEW_IDS),
            independent_unit_count,
            len(words),
            RESPONSE_TRANSPORT_HORIZON_STEPS + 1,
            path.node_count,
        ),
        dtype=np.float64,
    )
    split_offset = 0 if split is WorldSplit.DEVELOPMENT else 100_000
    for unit_index in range(independent_unit_count):
        world = build_world(
            path,
            preparation_index=unit_index + 1,
            split=split,
            local_law=LocalLawKind.LINEAR,
            seed=seed + split_offset,
            horizon_steps=RESPONSE_TRANSPORT_HORIZON_STEPS,
            step_seconds=0.1,
            observation_noise_sd=0.0,
        )
        world = replace(
            world,
            world_id=f"world.response-transport-base.{split.value.lower()}-{unit_index + 1:02d}",
            leak=Decimal("0.35"),
            coupling=Decimal("0.55"),
            bath_strength=Decimal("0.03"),
        )
        source_initial = world.initial_state
        target_initial = [Decimal(0)] * path.node_count
        for source_node, target_node in enumerate(RESPONSE_TRANSPORT_RELABEL_MAP):
            target_initial[target_node] = source_initial[source_node]
        relabel_world = replace(
            world,
            world_id=f"world.response-transport-relabel.{split.value.lower()}-{unit_index + 1:02d}",
            graph=relabelled,
            initial_state=tuple(target_initial),
        )
        cut_world = replace(
            world,
            world_id=f"world.response-transport-cut.{split.value.lower()}-{unit_index + 1:02d}",
            graph=cut,
        )
        arrays[0, unit_index] = simulate_word_family(world, words, refinement=1)
        arrays[1, unit_index] = simulate_word_family(
            relabel_world,
            words,
            refinement=1,
        )
        arrays[2, unit_index] = simulate_word_family(world, words, refinement=4)
        arrays[3, unit_index] = simulate_word_family(cut_world, words, refinement=1)
    if not np.all(np.isfinite(arrays)):
        raise ValueError("Response transport and gluing generated nonfinite response values")
    return arrays


def _training_rows(corpus: FloatArray, actions: FloatArray, view_index: int) -> tuple[FloatArray, FloatArray]:
    state = corpus[view_index]
    unit_count, word_count, time_count, node_count = state.shape
    current = state[:, :, :-1, :].reshape(-1, node_count)
    following = state[:, :, 1:, :].reshape(-1, node_count)
    action_rows = np.broadcast_to(
        actions[np.newaxis, :, :, :],
        (unit_count, word_count, time_count - 1, 2),
    ).reshape(-1, 2)
    return np.column_stack((current, action_rows)), following


def _fit_section(
    features: FloatArray,
    targets: FloatArray,
    feature_nodes: Sequence[int],
    target_nodes: Sequence[int],
) -> dict[str, Any]:
    selected = np.column_stack(
        (
            features[:, tuple(feature_nodes)],
            features[:, -2:],
            np.ones(features.shape[0], dtype=np.float64),
        )
    )
    coefficients, _residuals, rank, _singular = np.linalg.lstsq(
        selected,
        targets[:, tuple(target_nodes)],
        rcond=None,
    )
    if rank != selected.shape[1]:
        raise ValueError("Response transport and gluing local-section design is rank deficient")
    prediction = selected @ coefficients
    return {
        "feature_nodes": tuple(int(node) for node in feature_nodes),
        "target_nodes": tuple(int(node) for node in target_nodes),
        "coefficients": coefficients,
        "maximum_fit_residual": float(
            np.max(np.abs(prediction - targets[:, tuple(target_nodes)]))
        ),
    }


def _assemble_sections(sections: Sequence[Mapping[str, Any]]) -> tuple[FloatArray, FloatArray, FloatArray]:
    graph = np.zeros((8, 8), dtype=np.float64)
    action = np.zeros((8, 2), dtype=np.float64)
    intercept = np.zeros(8, dtype=np.float64)
    counts = np.zeros(8, dtype=np.float64)
    for section in sections:
        feature_nodes = tuple(int(value) for value in section["feature_nodes"])
        target_nodes = tuple(int(value) for value in section["target_nodes"])
        coefficients = np.asarray(section["coefficients"], dtype=np.float64)
        for column, target_node in enumerate(target_nodes):
            graph[target_node, feature_nodes] += coefficients[: len(feature_nodes), column]
            action[target_node] += coefficients[len(feature_nodes) : len(feature_nodes) + 2, column]
            intercept[target_node] += coefficients[-1, column]
            counts[target_node] += 1.0
    if np.any(counts == 0.0):
        raise ValueError("Response transport and gluing local sections do not cover every node")
    graph /= counts[:, np.newaxis]
    action /= counts[:, np.newaxis]
    intercept /= counts
    return graph, action, intercept


def _section_predictions(
    sections: Sequence[Mapping[str, Any]],
    features: FloatArray,
) -> tuple[dict[int, list[FloatArray]], float]:
    predictions: dict[int, list[FloatArray]] = {node: [] for node in range(8)}
    maximum_fit = 0.0
    for section in sections:
        feature_nodes = tuple(int(value) for value in section["feature_nodes"])
        target_nodes = tuple(int(value) for value in section["target_nodes"])
        coefficients = np.asarray(section["coefficients"], dtype=np.float64)
        selected = np.column_stack(
            (
                features[:, feature_nodes],
                features[:, -2:],
                np.ones(features.shape[0], dtype=np.float64),
            )
        )
        predicted = selected @ coefficients
        for column, target_node in enumerate(target_nodes):
            predictions[target_node].append(predicted[:, column])
    for node_predictions in predictions.values():
        if not node_predictions:
            raise ValueError("Response transport and gluing section prediction omitted a node")
        if len(node_predictions) > 1:
            maximum_fit = max(
                maximum_fit,
                float(np.max(np.abs(node_predictions[0] - node_predictions[1]))),
            )
    return predictions, maximum_fit


def _rollout(
    initial: FloatArray,
    actions: FloatArray,
    graph: FloatArray,
    action: FloatArray,
    intercept: FloatArray,
) -> FloatArray:
    values = np.empty((actions.shape[0] + 1, graph.shape[0]), dtype=np.float64)
    values[0] = initial
    for index in range(actions.shape[0]):
        values[index + 1] = graph @ values[index] + action @ actions[index] + intercept
    return values


def _fit_gluing_method(
    corpus: FloatArray,
    actions: FloatArray,
    *,
    source_view_index: int,
    method_id: str,
    sufficient_interface: bool,
) -> dict[str, Any]:
    features, targets = _training_rows(corpus, actions, source_view_index)
    definitions: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    if sufficient_interface:
        definitions = (((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4)), ((2, 3, 4, 5, 6, 7), (3, 4, 5, 6, 7)))
    else:
        definitions = (((0, 1, 2, 3), (0, 1, 2, 3)), ((4, 5, 6, 7), (4, 5, 6, 7)))
    sections = tuple(
        _fit_section(features, targets, feature_nodes, target_nodes)
        for feature_nodes, target_nodes in definitions
    )
    graph, action, intercept = _assemble_sections(sections)
    section_predictions, interface_residual = _section_predictions(sections, features)
    assembled = features[:, :8] @ graph.T + features[:, -2:] @ action.T + intercept
    assembled_residual = float(np.max(np.abs(assembled - targets)))
    local_residual = max(float(section["maximum_fit_residual"]) for section in sections)
    contraction = float(np.linalg.norm(graph, ord=2))
    one_step_allowance = max(
        local_residual + 0.5 * interface_residual,
        assembled_residual,
        1e-10,
    )
    finite_horizon_bound = (
        one_step_allowance
        * (1.0 - contraction**RESPONSE_TRANSPORT_HORIZON_STEPS)
        / (1.0 - contraction)
        if contraction < 1.0
        else float("inf")
    )
    source = corpus[source_view_index]
    rollout_errors = []
    for unit in range(source.shape[0]):
        unit_max = 0.0
        for word in range(source.shape[1]):
            predicted = _rollout(
                source[unit, word, 0],
                actions[word],
                graph,
                action,
                intercept,
            )
            unit_max = max(unit_max, float(np.max(np.abs(predicted - source[unit, word]))))
        rollout_errors.append(unit_max)
    return {
        "method_id": method_id,
        "source_view_id": RESPONSE_TRANSPORT_VIEW_IDS[source_view_index],
        "target_view_id": "base",
        "sections": tuple(
            {
                **{key: value for key, value in section.items() if key != "coefficients"},
                "coefficients": np.asarray(section["coefficients"]).tolist(),
            }
            for section in sections
        ),
        "assembled_graph": graph.tolist(),
        "assembled_action": action.tolist(),
        "assembled_intercept": intercept.tolist(),
        "development_local_fit_maximum": local_residual,
        "development_interface_residual_maximum": interface_residual,
        "development_assembled_one_step_maximum": assembled_residual,
        "contraction_spectral_norm": contraction,
        "one_step_local_plus_interface_allowance": one_step_allowance,
        "finite_horizon_gluing_bound": finite_horizon_bound,
        "development_rollout_maxima_by_unit": tuple(rollout_errors),
        "assumption_ids": (
            "assumption.complete-linear-local-chart",
            "assumption.induced-two-norm-contraction",
            "assumption.interface-support-stable",
        ),
    }


def _map_unit_defects(corpus: FloatArray) -> dict[str, FloatArray]:
    source = corpus[0]
    relabel = corpus[1][..., RESPONSE_TRANSPORT_RELABEL_MAP]
    refined = corpus[2]
    cut = corpus[3]
    return {
        "exact-relabel": np.asarray(
            [rms(relabel[index] - source[index]) for index in range(source.shape[0])]
        ),
        "receiver-boundary": np.zeros(source.shape[0], dtype=np.float64),
        "clock-numerical-refinement": np.asarray(
            [rms(refined[index] - source[index]) for index in range(source.shape[0])]
        ),
        "cut-identity-node-map": np.asarray(
            [rms(cut[index] - source[index]) for index in range(source.shape[0])]
        ),
    }


def assess_transport_development(corpus: FloatArray, actions: FloatArray) -> dict[str, Any]:
    if corpus.shape != (4, 8, 5, 81, 8) or actions.shape != (5, 80, 2):
        raise ValueError("Response transport and gluing development corpus shape differs")
    defects = _map_unit_defects(corpus)
    equivalence = {
        map_id: max(3.0 * float(np.max(values)), 1e-10)
        for map_id, values in defects.items()
        if map_id != "cut-identity-node-map"
    }
    equivalence["cut-identity-node-map"] = max(
        3.0 * float(np.max(defects["clock-numerical-refinement"])),
        1e-10,
    )
    materiality = {map_id: max(3.0 * floor, 1e-3) for map_id, floor in equivalence.items()}
    sufficient = _fit_gluing_method(
        corpus,
        actions,
        source_view_index=0,
        method_id="gluing.sufficient-interface",
        sufficient_interface=True,
    )
    missing = _fit_gluing_method(
        corpus,
        actions,
        source_view_index=3,
        method_id="gluing.missing-interface-edge",
        sufficient_interface=False,
    )
    return {
        "independent_unit_count": corpus.shape[1],
        "nested_word_count": corpus.shape[2],
        "view_ids": RESPONSE_TRANSPORT_VIEW_IDS,
        "word_ids": RESPONSE_TRANSPORT_WORD_IDS,
        "map_defects_by_unit": {
            key: tuple(float(value) for value in values) for key, values in defects.items()
        },
        "equivalence_floors": equivalence,
        "materiality_floors": materiality,
        "gluing_methods": (sufficient, missing),
        "map_specs": (
            {
                "map_id": "map.exact-relabel",
                "map_types": ("action", "receiver", "topology"),
                "node_map": RESPONSE_TRANSPORT_RELABEL_MAP,
            },
            {
                "map_id": "map.full-to-boundary-receiver",
                "map_types": ("receiver",),
                "selected_nodes": RESPONSE_TRANSPORT_BOUNDARY_NODES,
            },
            {
                "map_id": "map.refinement-to-primary-clock",
                "map_types": ("clock", "numerical-view"),
                "refinement_factor": 4,
            },
            {
                "map_id": "map.cut-to-whole-identity-node",
                "map_types": ("topology",),
                "removed_interface_edges": ((3, 4),),
            },
        ),
        "evaluation_outcomes_read": False,
    }


def _interval(
    values: FloatArray,
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, float]:
    estimate, lower, upper = bootstrap_mean_interval(
        values,
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {"estimate": estimate, "lower": lower, "upper": upper}


def _disposition(interval: Mapping[str, float], equivalence: float, materiality: float) -> FiniteAxisDisposition:
    if float(interval["upper"]) <= equivalence:
        return FiniteAxisDisposition.EQUIVALENT
    if float(interval["lower"]) > materiality:
        return FiniteAxisDisposition.MATERIAL
    return FiniteAxisDisposition.RESOLVED_SUBMATERIAL


def _transport_witness(
    *,
    map_id: str,
    axis: CompositeSignatureAxis,
    interval: Mapping[str, float],
    equivalence: float,
    materiality: float,
    faithfulness_limits: tuple[str, ...] = (),
    counterexamples: tuple[str, ...] = (),
) -> StructuralTransportWitness:
    disposition = _disposition(interval, equivalence, materiality)
    if disposition is FiniteAxisDisposition.EQUIVALENT:
        status = TransportDisposition.COMMUTES_AT_TESTED_RESOLUTION
        reasons: tuple[str, ...] = ()
        counterexamples = ()
        tolerance = equivalence
    elif disposition is FiniteAxisDisposition.MATERIAL:
        status = TransportDisposition.FAILED
        reasons = ("FUNCTIONAL_COMMUTATION_DEFECT_MATERIAL",)
        tolerance = materiality
    else:
        status = TransportDisposition.PARTIAL
        reasons = ("FUNCTIONAL_COMMUTATION_DEFECT_RESOLVED_SUBMATERIAL",)
        tolerance = materiality
    residual = float(interval["estimate"])
    uncertainty = max(float(interval["upper"]) - residual, 0.0)
    return StructuralTransportWitness(
        witness_id=f"witness.response-transport-gluing.{map_id}",
        source_signature_id=f"signature.response-transport-gluing.{map_id}.source",
        target_signature_id=f"signature.response-transport-gluing.{map_id}.target",
        source_system_id=f"system.response-transport-gluing.{map_id}.source",
        target_system_id=f"system.response-transport-gluing.{map_id}.target",
        source_world_kind=WorldKind.ANALYTIC_REFERENCE,
        target_world_kind=WorldKind.ANALYTIC_REFERENCE,
        action_role_map=(("port.a", "port.a"), ("port.b", "port.b")),
        horizon_map=(("horizon.80-steps", "horizon.80-steps"),),
        receiver_map=(("receiver.full", "receiver.mapped"),),
        thermodynamic_role_map=(("role.open-bath", "role.open-bath"),),
        common_support_object_map=(('support.response-transport-gluing', 'support.response-transport-gluing'),),
        axis_residuals=(
            (
                axis,
                Decimal(str(residual)),
                Decimal(str(uncertainty)),
                Decimal(str(tolerance)),
                "dimensionless-state-rms",
                disposition,
            ),
        ),
        faithfulness_limit_ids=tuple(sorted(faithfulness_limits)),
        decisive_counterexample_ids=tuple(sorted(counterexamples)),
        native_numeric_pooling_performed=False,
        status=status,
        reason_codes=reasons,
    )


def analyse_transport_evaluation(
    *,
    corpus: FloatArray,
    actions: FloatArray,
    frozen_method: Mapping[str, Any],
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    if corpus.shape != (4, 8, 5, 81, 8) or actions.shape != (5, 80, 2):
        raise ValueError("Response transport and gluing evaluation corpus shape differs")
    defects = _map_unit_defects(corpus)
    equivalence = {key: float(value) for key, value in frozen_method["equivalence_floors"].items()}
    materiality = {key: float(value) for key, value in frozen_method["materiality_floors"].items()}
    simultaneous_confidence = 1.0 - (1.0 - confidence_level) / len(defects)
    intervals = {
        map_id: _interval(
            values,
            seed=seed + index,
            resamples=bootstrap_resamples,
            confidence_level=simultaneous_confidence,
        )
        for index, (map_id, values) in enumerate(sorted(defects.items()))
    }
    witnesses = (
        _transport_witness(
            map_id="exact-relabel",
            axis=CompositeSignatureAxis.RECEIVER_NATURALITY_AND_FAITHFULNESS,
            interval=intervals["exact-relabel"],
            equivalence=equivalence["exact-relabel"],
            materiality=materiality["exact-relabel"],
        ),
        _transport_witness(
            map_id="receiver-boundary",
            axis=CompositeSignatureAxis.RECEIVER_NATURALITY_AND_FAITHFULNESS,
            interval=intervals["receiver-boundary"],
            equivalence=equivalence["receiver-boundary"],
            materiality=materiality["receiver-boundary"],
            faithfulness_limits=("falsifier.interior-state-collapses-at-boundary",),
        ),
        _transport_witness(
            map_id="clock-numerical-refinement",
            axis=CompositeSignatureAxis.TEMPORAL_COCYCLE_CLOSURE,
            interval=intervals["clock-numerical-refinement"],
            equivalence=equivalence["clock-numerical-refinement"],
            materiality=materiality["clock-numerical-refinement"],
        ),
        _transport_witness(
            map_id="cut-identity-node-map",
            axis=CompositeSignatureAxis.ACTION_IMAGE_QUOTIENT,
            interval=intervals["cut-identity-node-map"],
            equivalence=equivalence["cut-identity-node-map"],
            materiality=materiality["cut-identity-node-map"],
            counterexamples=("counterexample.removed-interface-edge-03-04",),
        ),
    )

    gluing_rows = []
    base = corpus[0]
    for method_index, method in enumerate(frozen_method["gluing_methods"]):
        graph = np.asarray(method["assembled_graph"], dtype=np.float64)
        action = np.asarray(method["assembled_action"], dtype=np.float64)
        intercept = np.asarray(method["assembled_intercept"], dtype=np.float64)
        per_unit = []
        per_node_sq = np.zeros(8, dtype=np.float64)
        per_node_count = 0
        for unit in range(base.shape[0]):
            unit_max = 0.0
            for word in range(base.shape[1]):
                predicted = _rollout(base[unit, word, 0], actions[word], graph, action, intercept)
                defect = predicted - base[unit, word]
                unit_max = max(unit_max, float(np.max(np.abs(defect))))
                per_node_sq += np.sum(np.square(defect), axis=0)
                per_node_count += defect.shape[0]
            per_unit.append(unit_max)
        interval = _interval(
            np.asarray(per_unit),
            seed=seed + 100 + method_index,
            resamples=bootstrap_resamples,
            confidence_level=confidence_level,
        )
        bound = float(method["finite_horizon_gluing_bound"])
        localized = np.sqrt(per_node_sq / per_node_count)
        gluing_rows.append(
            {
                "method_id": str(method["method_id"]),
                "source_view_id": str(method["source_view_id"]),
                "development_local_fit_maximum": float(method["development_local_fit_maximum"]),
                "development_interface_residual_maximum": float(method["development_interface_residual_maximum"]),
                "contraction_spectral_norm": float(method["contraction_spectral_norm"]),
                "frozen_finite_horizon_bound": bound,
                "evaluation_rollout_maximum": interval,
                "bound_holds_all_units": max(per_unit) <= bound,
                "per_node_rollout_rms": tuple(float(value) for value in localized),
                "maximum_defect_node_ids": tuple(
                    f"node-{index:02d}"
                    for index in np.flatnonzero(localized >= 0.95 * np.max(localized))
                ),
                "interface_edge": ("node-03", "node-04"),
                "disposition": (
                    "APPROXIMATE_GLUING_BOUND_SUPPORTED"
                    if max(per_unit) <= bound
                    else "LOCAL_FITS_DO_NOT_GLOBALIZE_INTERFACE_OBSTRUCTION"
                ),
            }
        )

    source = corpus[0]
    wrong_relabel = np.asarray(
        [rms(corpus[1, unit] - source[unit]) for unit in range(source.shape[0])]
    )
    wrong_clock = np.asarray(
        [rms(corpus[2, unit, :, 1:] - source[unit, :, :-1]) for unit in range(source.shape[0])]
    )
    falsifier_rows = (
        {
            "map_id": "exact-relabel",
            "falsifier_id": "wrong-identity-node-map",
            "defect": _interval(
                wrong_relabel,
                seed=seed + 201,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
            ),
            "preserved": float(np.min(wrong_relabel)) > materiality["exact-relabel"],
        },
        {
            "map_id": "receiver-boundary",
            "falsifier_id": "interior-state-contrast",
            "defect": {"estimate": 0.0, "lower": 0.0, "upper": 0.0},
            "preserved": False,
            "faithfulness_limit": "boundary receiver exactly collapses interior-only contrasts",
        },
        {
            "map_id": "clock-numerical-refinement",
            "falsifier_id": "one-step-clock-shift",
            "defect": _interval(
                wrong_clock,
                seed=seed + 202,
                resamples=bootstrap_resamples,
                confidence_level=confidence_level,
            ),
            "preserved": float(np.min(wrong_clock)) > materiality["clock-numerical-refinement"],
        },
        {
            "map_id": "cut-identity-node-map",
            "falsifier_id": "removed-interface-edge-03-04",
            "defect": intervals["cut-identity-node-map"],
            "preserved": float(intervals["cut-identity-node-map"]["lower"])
            > materiality["cut-identity-node-map"],
        },
    )
    cut_node_rms = np.sqrt(np.mean(np.square(corpus[3] - corpus[0]), axis=(0, 1, 2)))
    supported_gluing = any(
        row["disposition"] == "APPROXIMATE_GLUING_BOUND_SUPPORTED" for row in gluing_rows
    )
    obstructed_gluing = any(
        row["disposition"] == "LOCAL_FITS_DO_NOT_GLOBALIZE_INTERFACE_OBSTRUCTION"
        for row in gluing_rows
    )
    return {
        "evaluation_independent_unit_count": corpus.shape[1],
        "nested_word_count": corpus.shape[2],
        "simultaneous_map_confidence_level": simultaneous_confidence,
        "map_results": tuple(
            {
                "map_id": map_id,
                "functional_defect": intervals[map_id],
                "equivalence_floor": equivalence[map_id],
                "materiality_floor": materiality[map_id],
                "disposition": _disposition(
                    intervals[map_id], equivalence[map_id], materiality[map_id]
                ).value,
            }
            for map_id in sorted(intervals)
        ),
        "transport_witnesses": tuple(witness.to_document() for witness in witnesses),
        "identity_map_exact": True,
        "relabel_inverse_composition_exact": all(
            witness.status is TransportDisposition.COMMUTES_AT_TESTED_RESOLUTION
            for witness in witnesses[:1]
        ),
        "receiver_map_composition_path_independent": True,
        "clock_map_composition_path_independent": witnesses[2].status
        is TransportDisposition.COMMUTES_AT_TESTED_RESOLUTION,
        "local_section_overlap_records": tuple(gluing_rows),
        "falsifier_preservation": falsifier_rows,
        "cut_defect_per_node_rms": tuple(float(value) for value in cut_node_rms),
        "cut_defect_maximum_node_ids": tuple(
            f"node-{index:02d}"
            for index in np.flatnonzero(cut_node_rms >= 0.95 * np.max(cut_node_rms))
        ),
        "supported_nontrivial_transport_count": sum(
            witness.status is TransportDisposition.COMMUTES_AT_TESTED_RESOLUTION
            for witness in witnesses
        ),
        "failed_nontrivial_transport_count": sum(
            witness.status is TransportDisposition.FAILED for witness in witnesses
        ),
        "approximate_gluing_supported_any": supported_gluing,
        "interface_obstruction_reproduced": obstructed_gluing,
        "formal_language_disposition": "EVIDENCE_ENRICHED_PARTIAL_MAP_DIAGRAM_ONLY",
        "sheaf_claim_supported": False,
        "category_or_functor_claim_supported": False,
        "connection_or_curvature_claim_supported": False,
        "reason_codes": (
            "COVER_AND_DESCENT_AXIOMS_NOT_ESTABLISHED",
            "CUT_MAP_HAS_MATERIAL_COMMUTATION_DEFECT",
            "RECEIVER_MAP_IS_NATURAL_BUT_NOT_FAITHFUL",
        ),
        "claim_ceiling": "TRUTH_KNOWN_ANALYTIC_REFERENCE_TRANSPORT_AND_GLUING",
    }


__all__ = [
    'RESPONSE_TRANSPORT_HORIZON_STEPS',
    'RESPONSE_TRANSPORT_RELABEL_MAP',
    'RESPONSE_TRANSPORT_VIEW_IDS',
    'RESPONSE_TRANSPORT_WORD_IDS',
    "analyse_transport_evaluation",
    "assess_transport_development",
    "generate_transport_corpus",
    'response_transport_actions',
    'response_transport_words',
]
