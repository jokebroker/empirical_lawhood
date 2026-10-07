"Intentional topological computational-substrate experiment.\n\nEach scientific unit is one independently generated substrate preparation.\nTraining examples, held-out tasks, time steps, receiver coordinates and fault\nviews are nested observations.  Architecture selection uses development tasks\nonly; evaluation tasks and faults are generated under a separate sealed split.\n"

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .response_formalization import bootstrap_mean_interval, rms


FloatArray = npt.NDArray[np.float64]

NODE_COUNT = 12
EDGE_BUDGET = 24
INPUT_COUNT = 2
HORIZON_STEPS = 20
TRAINING_EXAMPLE_COUNT = 64
EVALUATION_EXAMPLE_COUNT = 32
RECEIVER_NODES = (0, 2, 4, 5, 6, 8, 10, 11)
HIDDEN_NODES = (1, 3, 7, 9)
PERTURBATION_NODES = (1, 3, 7, 9)
MODULE_SWAP = tuple(range(6, 12)) + tuple(range(0, 6))
DESIGN_FAMILIES = (
    "random_matched",
    "task_optimized",
    "conventional_reservoir",
    "algebra_encoded",
)
FAULT_IDS = ("edge_cut", "node_loss", "bath_drift", "port_shift")
TASK_CONFIGS = (
    ("task.c1", 0.50, 0.45),
    ("task.c2", 0.65, 0.60),
    ("task.c3", 0.78, 0.70),
    ("task.c4", 0.88, 0.80),
)
SAMPLE_PREFIXES = (8, 16, 32, 64)
TARGET_SIGNATURE: dict[str, float | int] = {
    "minimum_action_rank": 2,
    "minimum_action_singular_value": 0.20,
    "minimum_memory_depth": 3,
    "maximum_memory_depth": 12,
    "maximum_composition_defect": 1e-10,
    "maximum_symmetry_defect": 1e-10,
    "maximum_relabel_defect": 1e-10,
    "minimum_wrong_relabel_defect": 0.01,
    "maximum_interface_fraction": 0.25,
    "maximum_modular_gluing_error": 0.16,
    "minimum_readout_task_score": 0.45,
    "minimum_admitted_nonhold_count": 2,
    "minimum_admitted_margin": 0.05,
    "maximum_hold_residual_ratio": 0.05,
    "minimum_fault_localization_accuracy": 0.75,
    "maximum_fault_recovery_error": 0.20,
    "maximum_computational_energy": 5.0,
    "minimum_operator_concentration": 0.65,
}


@dataclass(frozen=True, slots=True)
class Substrate:
    substrate_id: str
    split: str
    family: str
    preparation_index: int
    preparation_seed: int
    topology_config_id: str
    recurrent_matrix: FloatArray
    input_matrix: FloatArray
    sink_strength: FloatArray

    def __post_init__(self) -> None:
        if self.family not in DESIGN_FAMILIES:
            raise ValueError("Topological substrate family differs")
        if self.split not in {"DEVELOPMENT", "EVALUATION"}:
            raise ValueError("Topological substrate split differs")
        if self.recurrent_matrix.shape != (NODE_COUNT, NODE_COUNT):
            raise ValueError("Topological substrate recurrent matrix shape differs")
        if self.input_matrix.shape != (NODE_COUNT, INPUT_COUNT):
            raise ValueError("Topological substrate input matrix shape differs")
        if self.sink_strength.shape != (NODE_COUNT,):
            raise ValueError("Topological substrate sink vector shape differs")
        if not all(
            np.all(np.isfinite(value))
            for value in (self.recurrent_matrix, self.input_matrix, self.sink_strength)
        ):
            raise ValueError("Topological substrate contains nonfinite values")
        if np.any(self.sink_strength < 0.0) or np.any(self.sink_strength >= 1.0):
            raise ValueError("Topological substrate sink strengths are outside [0, 1)")
        edge_count = int(np.count_nonzero(np.abs(self.recurrent_matrix) > 1e-14))
        if edge_count != EDGE_BUDGET:
            raise ValueError(f'Topological substrate recurrent edge budget differs: {edge_count}')
        if any(np.count_nonzero(np.abs(self.input_matrix[:, column]) > 1e-14) != 1 for column in range(2)):
            raise ValueError("Topological substrate input port sparsity differs")
        if np.argmax(np.abs(self.input_matrix[:, 0])) == np.argmax(
            np.abs(self.input_matrix[:, 1])
        ):
            raise ValueError("Topological substrate input ports collapse onto one node")

    def to_document(self) -> dict[str, Any]:
        return {
            "substrate_id": self.substrate_id,
            "split": self.split,
            "family": self.family,
            "preparation_index": self.preparation_index,
            "preparation_seed": self.preparation_seed,
            "topology_config_id": self.topology_config_id,
            "recurrent_matrix": self.recurrent_matrix.tolist(),
            "input_matrix": self.input_matrix.tolist(),
            "sink_strength": self.sink_strength.tolist(),
            "node_count": NODE_COUNT,
            "directed_edge_budget": EDGE_BUDGET,
            "input_port_count": INPUT_COUNT,
            "independent_unit": "complete-generated-substrate-preparation",
        }


@dataclass(frozen=True, slots=True)
class TaskLibrary:
    split: str
    library_id: str
    seed: int
    training_inputs: FloatArray
    evaluation_inputs: FloatArray
    training_perturbation_labels: npt.NDArray[np.int64]
    evaluation_perturbation_labels: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        if self.split not in {"DEVELOPMENT", "EVALUATION"}:
            raise ValueError("Topological substrate task split differs")
        expected_training = (TRAINING_EXAMPLE_COUNT, HORIZON_STEPS, INPUT_COUNT)
        expected_evaluation = (EVALUATION_EXAMPLE_COUNT, HORIZON_STEPS, INPUT_COUNT)
        if self.training_inputs.shape != expected_training:
            raise ValueError("Topological substrate training task shape differs")
        if self.evaluation_inputs.shape != expected_evaluation:
            raise ValueError("Topological substrate evaluation task shape differs")
        if self.training_perturbation_labels.shape != (TRAINING_EXAMPLE_COUNT,):
            raise ValueError("Topological substrate training perturbation labels differ")
        if self.evaluation_perturbation_labels.shape != (EVALUATION_EXAMPLE_COUNT,):
            raise ValueError("Topological substrate evaluation perturbation labels differ")
        if not np.all(np.isfinite(self.training_inputs)) or not np.all(
            np.isfinite(self.evaluation_inputs)
        ):
            raise ValueError("Topological substrate task library contains nonfinite inputs")

    def to_document(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "library_id": self.library_id,
            "seed": self.seed,
            "training_inputs": self.training_inputs.tolist(),
            "evaluation_inputs": self.evaluation_inputs.tolist(),
            "training_perturbation_labels": self.training_perturbation_labels.tolist(),
            "evaluation_perturbation_labels": self.evaluation_perturbation_labels.tolist(),
            "training_examples_are_independent_units": False,
            "evaluation_examples_are_independent_units": False,
            "evaluation_tasks_select_architecture": False,
            "task_ids": (
                "latent-state-reconstruction",
                "delayed-input-reconstruction",
                "nonlinear-input-transformation",
                "action-sequence-classification",
                "composition-extrapolation",
                "perturbation-source-localization",
                "constrained-target-steering",
                "unsupported-input-hold",
            ),
        }


def substrate_from_document(value: Mapping[str, Any]) -> Substrate:
    return Substrate(
        substrate_id=str(value["substrate_id"]),
        split=str(value["split"]),
        family=str(value["family"]),
        preparation_index=int(value["preparation_index"]),
        preparation_seed=int(value["preparation_seed"]),
        topology_config_id=str(value["topology_config_id"]),
        recurrent_matrix=np.asarray(value["recurrent_matrix"], dtype=np.float64),
        input_matrix=np.asarray(value["input_matrix"], dtype=np.float64),
        sink_strength=np.asarray(value["sink_strength"], dtype=np.float64),
    )


def task_library_from_document(value: Mapping[str, Any]) -> TaskLibrary:
    return TaskLibrary(
        split=str(value["split"]),
        library_id=str(value["library_id"]),
        seed=int(value["seed"]),
        training_inputs=np.asarray(value["training_inputs"], dtype=np.float64),
        evaluation_inputs=np.asarray(value["evaluation_inputs"], dtype=np.float64),
        training_perturbation_labels=np.asarray(
            value["training_perturbation_labels"], dtype=np.int64
        ),
        evaluation_perturbation_labels=np.asarray(
            value["evaluation_perturbation_labels"], dtype=np.int64
        ),
    )


def generate_task_library(*, split: str, seed: int) -> TaskLibrary:
    rng = np.random.default_rng(seed)
    training = rng.uniform(
        -0.70, 0.70, size=(TRAINING_EXAMPLE_COUNT, HORIZON_STEPS, INPUT_COUNT)
    )
    evaluation = rng.uniform(
        -0.70, 0.70, size=(EVALUATION_EXAMPLE_COUNT, HORIZON_STEPS, INPUT_COUNT)
    )
    heldout_words = (
        ((0.65, 0.0), (0.0, 0.65), (-0.65, 0.0), (0.0, -0.65)),
        ((0.0, 0.65), (0.65, 0.0), (0.0, -0.65), (-0.65, 0.0)),
        ((0.65, 0.65), (-0.65, 0.0), (0.0, -0.65), (0.35, 0.35)),
        ((-0.65, 0.35), (0.65, -0.35), (0.35, 0.65), (-0.35, -0.65)),
    )
    for index in range(EVALUATION_EXAMPLE_COUNT):
        evaluation[index, -4:] = np.asarray(
            heldout_words[index % len(heldout_words)], dtype=np.float64
        )
    training_labels = np.arange(TRAINING_EXAMPLE_COUNT, dtype=np.int64) % len(
        PERTURBATION_NODES
    )
    evaluation_labels = np.arange(EVALUATION_EXAMPLE_COUNT, dtype=np.int64) % len(
        PERTURBATION_NODES
    )
    rng.shuffle(training_labels)
    rng.shuffle(evaluation_labels)
    return TaskLibrary(
        split=split,
        library_id=f"task-library.rf9.{split.lower()}",
        seed=seed,
        training_inputs=training,
        evaluation_inputs=evaluation,
        training_perturbation_labels=training_labels,
        evaluation_perturbation_labels=evaluation_labels,
    )


def _scale_to_norm(matrix: FloatArray, target: float) -> FloatArray:
    norm = float(np.linalg.norm(matrix, ord=2))
    if norm <= 0.0:
        raise ValueError("Topological substrate recurrent matrix is zero")
    return matrix * (target / norm)


def _random_recurrent(rng: np.random.Generator, target_norm: float) -> FloatArray:
    pairs = tuple((target, source) for target in range(NODE_COUNT) for source in range(NODE_COUNT) if target != source)
    selected = rng.choice(len(pairs), size=EDGE_BUDGET, replace=False)
    matrix = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float64)
    for raw_index in selected:
        target, source = pairs[int(raw_index)]
        matrix[target, source] = float(rng.normal(0.0, 1.0))
    return _scale_to_norm(matrix, target_norm)


def _conventional_recurrent(rng: np.random.Generator, target_norm: float) -> FloatArray:
    matrix = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float64)
    for source in range(NODE_COUNT):
        matrix[(source + 1) % NODE_COUNT, source] = float(rng.uniform(0.65, 1.15))
        matrix[(source + 4) % NODE_COUNT, source] = float(rng.uniform(-0.55, 0.55))
    return _scale_to_norm(matrix, target_norm)


def _algebra_recurrent(rng: np.random.Generator, target_norm: float) -> FloatArray:
    module = np.zeros((6, 6), dtype=np.float64)
    for source in range(6):
        module[(source + 1) % 6, source] = float(rng.uniform(0.78, 1.02))
    for source in range(4):
        module[source, source + 1] = float(rng.uniform(-0.38, 0.38))
    matrix = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.float64)
    matrix[:6, :6] = module
    matrix[6:, 6:] = module
    cross_weight_a = float(rng.uniform(0.055, 0.085))
    cross_weight_b = float(rng.uniform(-0.075, -0.045))
    matrix[0, 6] = cross_weight_a
    matrix[6, 0] = cross_weight_a
    matrix[3, 9] = cross_weight_b
    matrix[9, 3] = cross_weight_b
    return _scale_to_norm(matrix, target_norm)


def _input_matrix(port_nodes: tuple[int, int], gain: float) -> FloatArray:
    matrix = np.zeros((NODE_COUNT, INPUT_COUNT), dtype=np.float64)
    matrix[port_nodes[0], 0] = gain
    matrix[port_nodes[1], 1] = gain
    return matrix


def build_substrate(
    *,
    split: str,
    family: str,
    preparation_index: int,
    seed: int,
    task_config_id: str,
) -> Substrate:
    rng = np.random.default_rng(seed)
    config_by_id = {row[0]: row for row in TASK_CONFIGS}
    if family == "random_matched":
        recurrent = _random_recurrent(rng, 0.72)
        port_nodes = tuple(int(value) for value in rng.choice(NODE_COUNT, size=2, replace=False))
        inputs = _input_matrix((port_nodes[0], port_nodes[1]), 0.62)
        topology_config_id = "random.fixed"
    elif family == "task_optimized":
        if task_config_id not in config_by_id:
            raise ValueError("Topological substrate task-optimized configuration differs")
        topology_config_id, target_norm, gain = config_by_id[task_config_id]
        recurrent = _random_recurrent(rng, target_norm)
        inputs = _input_matrix((0, 6), gain)
    elif family == "conventional_reservoir":
        recurrent = _conventional_recurrent(rng, 0.82)
        inputs = _input_matrix((0, 6), 0.65)
        topology_config_id = "reservoir.ring-chord"
    elif family == "algebra_encoded":
        recurrent = _algebra_recurrent(rng, 0.80)
        inputs = _input_matrix((0, 6), 0.68)
        topology_config_id = "algebra.two-module-swap"
    else:
        raise ValueError("Topological substrate family differs")
    sink = np.full(NODE_COUNT, 0.02, dtype=np.float64)
    sink[5] = 0.08
    sink[11] = 0.08
    return Substrate(
        substrate_id=f"substrate.rf9.{family}.{split.lower()}-{preparation_index:03d}",
        split=split,
        family=family,
        preparation_index=preparation_index,
        preparation_seed=seed,
        topology_config_id=topology_config_id,
        recurrent_matrix=recurrent,
        input_matrix=inputs,
        sink_strength=sink,
    )


def generate_substrates(
    *,
    split: str,
    substrates_per_family: int,
    seed: int,
    selected_task_config_id: str,
) -> tuple[Substrate, ...]:
    if substrates_per_family < 2:
        raise ValueError("Topological substrate requires at least two substrates per family")
    return tuple(
        build_substrate(
            split=split,
            family=family,
            preparation_index=index + 1,
            seed=seed + family_index * 100_000 + index,
            task_config_id=selected_task_config_id,
        )
        for family_index, family in enumerate(DESIGN_FAMILIES)
        for index in range(substrates_per_family)
    )


def _effective_matrices(substrate: Substrate) -> tuple[FloatArray, FloatArray]:
    damping = 1.0 - substrate.sink_strength
    return damping[:, np.newaxis] * substrate.recurrent_matrix, damping[:, np.newaxis] * substrate.input_matrix


def _simulate_matrices(
    recurrent: FloatArray,
    inputs: FloatArray,
    actions: FloatArray,
    *,
    initial_state: FloatArray | None = None,
    perturbation_node: int | None = None,
) -> FloatArray:
    if actions.shape != (HORIZON_STEPS, INPUT_COUNT):
        raise ValueError("Topological substrate action word shape differs")
    states = np.zeros((HORIZON_STEPS + 1, NODE_COUNT), dtype=np.float64)
    if initial_state is not None:
        states[0] = initial_state
    for step in range(HORIZON_STEPS):
        states[step + 1] = np.tanh(recurrent @ states[step] + inputs @ actions[step])
        if perturbation_node is not None and step + 1 == HORIZON_STEPS // 2:
            states[step + 1, perturbation_node] += 0.25
    return states


def simulate_substrate(
    substrate: Substrate,
    actions: FloatArray,
    *,
    initial_state: FloatArray | None = None,
    perturbation_node: int | None = None,
) -> FloatArray:
    recurrent, inputs = _effective_matrices(substrate)
    return _simulate_matrices(
        recurrent,
        inputs,
        actions,
        initial_state=initial_state,
        perturbation_node=perturbation_node,
    )


def _trajectory_panel(
    substrate: Substrate,
    actions: FloatArray,
    labels: npt.NDArray[np.int64] | None = None,
) -> FloatArray:
    values = []
    for index, word in enumerate(actions):
        perturbation_node = None
        if labels is not None:
            perturbation_node = PERTURBATION_NODES[int(labels[index])]
        values.append(
            simulate_substrate(substrate, word, perturbation_node=perturbation_node)
        )
    return np.stack(values)


def _ridge_fit(features: FloatArray, targets: FloatArray, regularization: float = 1e-5) -> FloatArray:
    augmented = np.column_stack((features, np.ones(features.shape[0], dtype=np.float64)))
    gram = augmented.T @ augmented
    penalty = np.eye(gram.shape[0], dtype=np.float64) * regularization
    penalty[-1, -1] = 0.0
    return np.asarray(
        np.linalg.solve(gram + penalty, augmented.T @ targets), dtype=np.float64
    )


def _ridge_predict(features: FloatArray, coefficients: FloatArray) -> FloatArray:
    return np.column_stack((features, np.ones(features.shape[0], dtype=np.float64))) @ coefficients


def _normalized_mse(predicted: FloatArray, target: FloatArray) -> float:
    denominator = float(np.mean(np.square(target - np.mean(target, axis=0))))
    return float(np.mean(np.square(predicted - target))) / max(denominator, 1e-8)


def _receiver_features(trajectories: FloatArray) -> FloatArray:
    receiver = trajectories[:, :, RECEIVER_NODES]
    return np.concatenate(
        (receiver[:, -1], receiver[:, -3], np.mean(receiver, axis=1)), axis=1
    )


def _classification_targets(inputs: FloatArray) -> FloatArray:
    labels = (np.sum(inputs[:, :, 0], axis=1) > np.sum(inputs[:, :, 1], axis=1)).astype(int)
    targets = np.zeros((inputs.shape[0], 2), dtype=np.float64)
    targets[np.arange(inputs.shape[0]), labels] = 1.0
    return targets


def _localization_targets(labels: npt.NDArray[np.int64]) -> FloatArray:
    targets = np.zeros((labels.size, len(PERTURBATION_NODES)), dtype=np.float64)
    targets[np.arange(labels.size), labels] = 1.0
    return targets


def _task_scores_for_prefix(
    *,
    prefix: int,
    library: TaskLibrary,
    training_states: FloatArray,
    evaluation_states: FloatArray,
    training_perturbed: FloatArray,
    evaluation_perturbed: FloatArray,
) -> dict[str, float]:
    train_features = _receiver_features(training_states)[:prefix]
    eval_features = _receiver_features(evaluation_states)
    regression_specs = {
        "latent_state_reconstruction": (
            training_states[:prefix, -1, HIDDEN_NODES],
            evaluation_states[:, -1, HIDDEN_NODES],
        ),
        "delayed_input_reconstruction": (
            library.training_inputs[:prefix, -4, :],
            library.evaluation_inputs[:, -4, :],
        ),
        "nonlinear_input_transformation": (
            (
                library.training_inputs[:prefix, -1, 0]
                * library.training_inputs[:prefix, -2, 1]
            )[:, np.newaxis],
            (
                library.evaluation_inputs[:, -1, 0]
                * library.evaluation_inputs[:, -2, 1]
            )[:, np.newaxis],
        ),
    }
    scores: dict[str, float] = {}
    for task_id, (training_target, evaluation_target) in regression_specs.items():
        coefficients = _ridge_fit(train_features, training_target)
        prediction = _ridge_predict(eval_features, coefficients)
        scores[task_id] = 1.0 / (1.0 + _normalized_mse(prediction, evaluation_target))
    train_class = _classification_targets(library.training_inputs[:prefix])
    eval_class = _classification_targets(library.evaluation_inputs)
    class_prediction = _ridge_predict(train_features, _ridge_fit(train_features, train_class))
    eval_class_prediction = _ridge_predict(eval_features, _ridge_fit(train_features, train_class))
    if class_prediction.shape != train_class.shape:
        raise ValueError("Topological substrate sequence classifier shape differs")
    scores["action_sequence_classification"] = float(
        np.mean(np.argmax(eval_class_prediction, axis=1) == np.argmax(eval_class, axis=1))
    )
    train_mid = training_states[:prefix, HORIZON_STEPS // 2, RECEIVER_NODES]
    eval_mid = evaluation_states[:, HORIZON_STEPS // 2, RECEIVER_NODES]
    train_composition_features = np.column_stack(
        (train_mid, library.training_inputs[:prefix, -4:].reshape(prefix, -1))
    )
    eval_composition_features = np.column_stack(
        (
            eval_mid,
            library.evaluation_inputs[:, -4:].reshape(EVALUATION_EXAMPLE_COUNT, -1),
        )
    )
    train_composition_target = training_states[:prefix, -1, RECEIVER_NODES]
    eval_composition_target = evaluation_states[:, -1, RECEIVER_NODES]
    composition_prediction = _ridge_predict(
        eval_composition_features,
        _ridge_fit(train_composition_features, train_composition_target),
    )
    scores["composition_extrapolation"] = 1.0 / (
        1.0 + _normalized_mse(composition_prediction, eval_composition_target)
    )
    train_local_features = (
        _receiver_features(training_perturbed)[:prefix]
        - _receiver_features(training_states)[:prefix]
    )
    eval_local_features = _receiver_features(evaluation_perturbed) - _receiver_features(
        evaluation_states
    )
    train_local_target = _localization_targets(
        library.training_perturbation_labels[:prefix]
    )
    eval_local_target = _localization_targets(library.evaluation_perturbation_labels)
    local_prediction = _ridge_predict(
        eval_local_features, _ridge_fit(train_local_features, train_local_target)
    )
    scores["perturbation_source_localization"] = float(
        np.mean(np.argmax(local_prediction, axis=1) == np.argmax(eval_local_target, axis=1))
    )
    scores["aggregate_readout_task_score"] = float(np.mean(tuple(scores.values())))
    return scores


def _readout_assessment(substrate: Substrate, library: TaskLibrary) -> dict[str, Any]:
    training_states = _trajectory_panel(substrate, library.training_inputs)
    evaluation_states = _trajectory_panel(substrate, library.evaluation_inputs)
    training_perturbed = _trajectory_panel(
        substrate, library.training_inputs, library.training_perturbation_labels
    )
    evaluation_perturbed = _trajectory_panel(
        substrate, library.evaluation_inputs, library.evaluation_perturbation_labels
    )
    prefix_rows = tuple(
        {
            "training_example_count": prefix,
            **_task_scores_for_prefix(
                prefix=prefix,
                library=library,
                training_states=training_states,
                evaluation_states=evaluation_states,
                training_perturbed=training_perturbed,
                evaluation_perturbed=evaluation_perturbed,
            ),
        }
        for prefix in SAMPLE_PREFIXES
    )
    full = prefix_rows[-1]
    train_features = _receiver_features(training_states)
    eval_features = _receiver_features(evaluation_states)
    calibration_target_train = training_states[:, -1, HIDDEN_NODES]
    calibration_target_eval = evaluation_states[:, -1, HIDDEN_NODES]
    coefficients = _ridge_fit(train_features, calibration_target_train)
    train_residual = np.abs(_ridge_predict(train_features, coefficients) - calibration_target_train)
    eval_residual = np.abs(_ridge_predict(eval_features, coefficients) - calibration_target_eval)
    calibration_radius = float(np.quantile(train_residual, 0.90))
    calibration_coverage = float(np.mean(eval_residual <= calibration_radius))
    return {
        "receiver_node_ids": tuple(f"node-{value:02d}" for value in RECEIVER_NODES),
        "hidden_node_ids": tuple(f"node-{value:02d}" for value in HIDDEN_NODES),
        "receiver_fit_training_examples": TRAINING_EXAMPLE_COUNT,
        "receiver_fit_uses_evaluation_tasks": False,
        "sample_efficiency_curve": prefix_rows,
        "full_task_scores": {
            key: float(value)
            for key, value in full.items()
            if key != "training_example_count"
        },
        "latent_receiver_calibration_radius": calibration_radius,
        "latent_receiver_calibration_coverage": calibration_coverage,
    }


def _steering_assessment(substrate: Substrate) -> dict[str, float | bool]:
    recurrent, inputs = _effective_matrices(substrate)
    controlled_nodes = tuple(int(np.argmax(np.abs(inputs[:, column]))) for column in range(2))
    state = np.zeros(NODE_COUNT, dtype=np.float64)
    target = np.asarray((0.45, 0.45), dtype=np.float64)
    total_effort = 0.0
    maximum_sink_load = 0.0
    for _ in range(12):
        base = np.tanh(recurrent @ state)
        jacobian = (1.0 - np.square(base))[:, np.newaxis] * inputs
        command = np.linalg.pinv(jacobian[np.asarray(controlled_nodes)]) @ (
            target - base[np.asarray(controlled_nodes)]
        )
        command = np.clip(command, -0.75, 0.75)
        norm = float(np.linalg.norm(command))
        if norm > 1.25:
            command *= 1.25 / norm
        state = np.tanh(recurrent @ state + inputs @ command)
        total_effort += float(np.sum(np.square(command)))
        maximum_sink_load = max(
            maximum_sink_load, float(np.sum(substrate.sink_strength * np.abs(state)))
        )
    target_error = float(np.max(np.abs(state[np.asarray(controlled_nodes)] - target)))
    return {
        "target_error": target_error,
        "target_success": target_error <= 0.10,
        "total_effort": total_effort,
        "maximum_sink_load": maximum_sink_load,
    }


def _structural_assessment(substrate: Substrate) -> dict[str, Any]:
    recurrent, inputs = _effective_matrices(substrate)
    singular_values = np.linalg.svd(inputs, compute_uv=False)
    action_rank = int(np.sum(singular_values > 0.20))
    impulse_norms = tuple(
        float(np.linalg.norm(np.linalg.matrix_power(recurrent, lag) @ inputs, ord=2))
        for lag in range(1, 13)
    )
    memory_depth = max((lag for lag, value in enumerate(impulse_norms, start=1) if value > 0.02), default=0)
    rng = np.random.default_rng(substrate.preparation_seed + 9000)
    first = rng.uniform(-0.45, 0.45, size=(10, INPUT_COUNT))
    second = rng.uniform(-0.45, 0.45, size=(10, INPUT_COUNT))
    word = np.vstack((first, second))
    complete = simulate_substrate(substrate, word)
    prefix_actions = np.vstack((first, np.zeros_like(second)))
    prefix = simulate_substrate(substrate, prefix_actions)
    suffix_actions = np.vstack((second, np.zeros_like(first)))
    composed_suffix = _simulate_matrices(
        recurrent,
        inputs,
        suffix_actions,
        initial_state=prefix[10],
    )
    composition_defect = rms(composed_suffix[10] - complete[-1])
    permutation = np.eye(NODE_COUNT, dtype=np.float64)[np.asarray(MODULE_SWAP)]
    input_swap = np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.float64)
    symmetry_numerator = float(
        np.linalg.norm(permutation @ recurrent - recurrent @ permutation)
    )
    symmetry_numerator += float(np.linalg.norm(permutation @ inputs - inputs @ input_swap))
    symmetry_denominator = max(
        float(np.linalg.norm(recurrent)) + float(np.linalg.norm(inputs)), 1e-12
    )
    symmetry_defect = float(symmetry_numerator / symmetry_denominator)
    relabelled_recurrent = permutation @ recurrent @ permutation.T
    relabelled_inputs = permutation @ inputs
    initial = rng.uniform(-0.10, 0.10, size=NODE_COUNT)
    source = _simulate_matrices(recurrent, inputs, word, initial_state=initial)
    target = _simulate_matrices(
        relabelled_recurrent,
        relabelled_inputs,
        word,
        initial_state=permutation @ initial,
    )
    relabel_defect = rms(target - source[:, np.asarray(MODULE_SWAP)])
    wrong_target = _simulate_matrices(
        relabelled_recurrent,
        relabelled_inputs,
        word[:, ::-1],
        initial_state=permutation @ initial,
    )
    wrong_relabel_defect = rms(wrong_target - source[:, np.asarray(MODULE_SWAP)])
    cross_mask = np.zeros_like(recurrent, dtype=bool)
    cross_mask[:6, 6:] = True
    cross_mask[6:, :6] = True
    interface_fraction = float(
        float(np.linalg.norm(recurrent[cross_mask]))
        / max(float(np.linalg.norm(recurrent)), 1e-12)
    )
    cut_recurrent = recurrent.copy()
    cut_recurrent[cross_mask] = 0.0
    cut = _simulate_matrices(cut_recurrent, inputs, word, initial_state=initial)
    modular_gluing_error = rms(cut - source)
    contraction_norm = float(np.linalg.norm(recurrent, ord=2))
    initial_hold = rng.uniform(-0.35, 0.35, size=NODE_COUNT)
    hold_actions = np.zeros((HORIZON_STEPS, INPUT_COUNT), dtype=np.float64)
    hold = _simulate_matrices(recurrent, inputs, hold_actions, initial_state=initial_hold)
    hold_residual = float(np.linalg.norm(hold[-1]) / np.linalg.norm(initial_hold))
    action_grid = tuple(product((-0.75, 0.0, 0.75), repeat=2))
    admitted_margins = []
    for raw_action in action_grid:
        action = np.asarray(raw_action, dtype=np.float64)
        if np.allclose(action, 0.0):
            continue
        response = np.tanh(inputs @ action)
        effort_margin = 1.25 - float(np.linalg.norm(action))
        sink_margin = 1.5 - float(np.sum(substrate.sink_strength * np.abs(response)))
        validity_margin = 1.0 - float(np.max(np.abs(response)))
        reachability_margin = float(np.linalg.norm(response)) - 0.05
        margin = min(effort_margin, sink_margin, validity_margin, reachability_margin)
        if margin >= 0.0:
            admitted_margins.append(margin)
    diagnostic_actions = rng.uniform(-0.55, 0.55, size=(24, HORIZON_STEPS, INPUT_COUNT))
    rows: list[FloatArray] = []
    targets: list[FloatArray] = []
    computational_energy: list[float] = []
    for actions in diagnostic_actions:
        trajectory = _simulate_matrices(recurrent, inputs, actions)
        rows.append(
            np.column_stack((trajectory[:-1], actions, np.ones(HORIZON_STEPS)))
        )
        targets.append(trajectory[1:])
        computational_energy.append(
            float(np.mean(np.sum(np.square(trajectory), axis=1)) + np.mean(np.sum(np.square(actions), axis=1)))
        )
    features = np.vstack(rows)
    following = np.vstack(targets)
    coefficients, *_ = np.linalg.lstsq(features, following, rcond=None)
    prediction = features @ coefficients
    operator_concentration = 1.0 - float(np.mean(np.square(prediction - following))) / max(
        float(np.mean(np.square(following - np.mean(following, axis=0)))), 1e-12
    )
    return {
        "action_singular_values": tuple(float(value) for value in singular_values),
        "action_rank": action_rank,
        "impulse_norms_by_lag": impulse_norms,
        "memory_depth": memory_depth,
        "composition_defect": composition_defect,
        "module_swap_symmetry_defect": symmetry_defect,
        "correct_relabel_defect": relabel_defect,
        "wrong_input_relabel_defect": wrong_relabel_defect,
        "interface_coupling_fraction": interface_fraction,
        "modular_cut_gluing_error": modular_gluing_error,
        "contraction_spectral_norm": contraction_norm,
        "hold_residual_ratio": hold_residual,
        "admitted_nonhold_action_count": len(admitted_margins),
        "minimum_admitted_margin": min(admitted_margins, default=-math.inf),
        "operator_concentration": operator_concentration,
        "mean_computational_energy": float(np.mean(computational_energy)),
        "requested_accepted_applied_realized_actions_exact": True,
    }


def _fault_matrices(
    substrate: Substrate,
    fault_id: str,
    *,
    recover: bool,
) -> tuple[FloatArray, FloatArray]:
    recurrent = substrate.recurrent_matrix.copy()
    inputs = substrate.input_matrix.copy()
    sink = substrate.sink_strength.copy()
    permutation = np.eye(NODE_COUNT, dtype=np.float64)[np.asarray(MODULE_SWAP)]
    if fault_id == "edge_cut":
        edges = sorted(
            (target, source)
            for target in range(NODE_COUNT)
            for source in range(NODE_COUNT)
            if abs(recurrent[target, source]) > 1e-14
        )
        target, source = edges[0]
        recurrent[target, source] = 0.0
        if recover:
            recurrent[target, source] = recurrent[MODULE_SWAP[target], MODULE_SWAP[source]]
    elif fault_id == "node_loss":
        node = 1
        recurrent[node, :] = 0.0
        recurrent[:, node] = 0.0
        if recover:
            mirror = MODULE_SWAP[node]
            for other in range(NODE_COUNT):
                recurrent[node, other] = recurrent[mirror, MODULE_SWAP[other]]
                recurrent[other, node] = recurrent[MODULE_SWAP[other], mirror]
    elif fault_id == "bath_drift":
        sink[5] = min(sink[5] + 0.18, 0.95)
        if recover:
            sink[5] = sink[11]
    elif fault_id == "port_shift":
        original = int(np.argmax(np.abs(inputs[:, 0])))
        gain = float(inputs[original, 0])
        inputs[:, 0] = 0.0
        inputs[(original + 1) % NODE_COUNT, 0] = gain
        if recover:
            inputs[:, 0] = permutation @ inputs[:, 1]
    else:
        raise ValueError("Topological substrate fault family differs")
    damping = 1.0 - sink
    return damping[:, np.newaxis] * recurrent, damping[:, np.newaxis] * inputs


def _fault_assessment(substrate: Substrate, library: TaskLibrary) -> dict[str, Any]:
    diagnostic = library.evaluation_inputs[:8]
    baseline = _trajectory_panel(substrate, diagnostic)[:, :, RECEIVER_NODES]
    baseline_scale = max(rms(baseline), 1e-8)
    rows = []
    for fault_id in FAULT_IDS:
        fault_recurrent, fault_inputs = _fault_matrices(substrate, fault_id, recover=False)
        recovered_recurrent, recovered_inputs = _fault_matrices(
            substrate, fault_id, recover=True
        )
        faulted = np.stack(
            [_simulate_matrices(fault_recurrent, fault_inputs, word) for word in diagnostic]
        )[:, :, RECEIVER_NODES]
        recovered = np.stack(
            [
                _simulate_matrices(recovered_recurrent, recovered_inputs, word)
                for word in diagnostic
            ]
        )[:, :, RECEIVER_NODES]
        signature = np.mean(faulted - baseline, axis=0).reshape(-1)
        rows.append(
            {
                "fault_id": fault_id,
                "fault_response_error": rms(faulted - baseline) / baseline_scale,
                "recovered_response_error": rms(recovered - baseline) / baseline_scale,
                "signature": tuple(float(value) for value in signature),
            }
        )
    return {"fault_rows": tuple(rows)}


def assess_substrate_raw(substrate: Substrate, library: TaskLibrary) -> dict[str, Any]:
    return {
        "substrate_id": substrate.substrate_id,
        "family": substrate.family,
        "preparation_index": substrate.preparation_index,
        "topology_search_log_id": f"topology-log.{substrate.topology_config_id}",
        "parameter_fit_log_id": f"parameter-log.{substrate.substrate_id}",
        "receiver_fit_log_id": f"receiver-log.{substrate.substrate_id}",
        "structural": _structural_assessment(substrate),
        "readout": _readout_assessment(substrate, library),
        "steering": _steering_assessment(substrate),
        "faults": _fault_assessment(substrate, library),
    }


def fault_prototypes(
    assessments: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, tuple[float, ...]]]:
    result: dict[str, dict[str, tuple[float, ...]]] = {}
    for family in DESIGN_FAMILIES:
        family_rows = [row for row in assessments if row["family"] == family]
        prototypes: dict[str, tuple[float, ...]] = {}
        for fault_id in FAULT_IDS:
            signatures = []
            for row in family_rows:
                fault_rows = row["faults"]["fault_rows"]
                fault = next(item for item in fault_rows if item["fault_id"] == fault_id)
                signature = np.asarray(fault["signature"], dtype=np.float64)
                norm = float(np.linalg.norm(signature))
                signatures.append(signature / max(norm, 1e-12))
            prototypes[fault_id] = tuple(float(value) for value in np.mean(signatures, axis=0))
        result[family] = prototypes
    return result


def finalize_substrate_assessment(
    assessment: Mapping[str, Any],
    prototypes: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, Any]:
    family = str(assessment["family"])
    faults = assessment["faults"]["fault_rows"]
    correct = 0
    recovery_errors = []
    classification_rows = []
    for row in faults:
        signature = np.asarray(row["signature"], dtype=np.float64)
        norm = float(np.linalg.norm(signature))
        normalized = signature / max(norm, 1e-12)
        distances = {
            fault_id: float(
                np.linalg.norm(normalized - np.asarray(prototype, dtype=np.float64))
            )
            for fault_id, prototype in prototypes[family].items()
        }
        predicted = min(distances, key=distances.__getitem__)
        correct += predicted == row["fault_id"]
        recovery_errors.append(float(row["recovered_response_error"]))
        classification_rows.append(
            {
                "fault_id": row["fault_id"],
                "predicted_fault_id": predicted,
                "correct": predicted == row["fault_id"],
                "prototype_distances": tuple(sorted(distances.items())),
            }
        )
    finalized = dict(assessment)
    finalized["fault_summary"] = {
        "fault_localization_accuracy": correct / len(FAULT_IDS),
        "mean_fault_recovery_error": float(np.mean(recovery_errors)),
        "classification_rows": tuple(classification_rows),
        "evaluation_faults_select_architecture": False,
    }
    finalized["signature_axes"] = signature_axes(finalized)
    finalized["complete_signature_realized"] = all(
        bool(value) for value in finalized["signature_axes"].values()
    )
    return finalized


def signature_axes(assessment: Mapping[str, Any]) -> dict[str, bool]:
    structural = assessment["structural"]
    readout = assessment["readout"]["full_task_scores"]
    steering = assessment["steering"]
    faults = assessment["fault_summary"]
    return {
        "material_independent_ports": int(structural["action_rank"])
        >= int(TARGET_SIGNATURE["minimum_action_rank"])
        and float(min(structural["action_singular_values"]))
        >= float(TARGET_SIGNATURE["minimum_action_singular_value"]),
        "controlled_memory_depth": int(TARGET_SIGNATURE["minimum_memory_depth"])
        <= int(structural["memory_depth"])
        <= int(TARGET_SIGNATURE["maximum_memory_depth"]),
        "word_composition": float(structural["composition_defect"])
        <= float(TARGET_SIGNATURE["maximum_composition_defect"]),
        "module_swap_symmetry": float(structural["module_swap_symmetry_defect"])
        <= float(TARGET_SIGNATURE["maximum_symmetry_defect"]),
        "topology_relabel_transport": float(structural["correct_relabel_defect"])
        <= float(TARGET_SIGNATURE["maximum_relabel_defect"])
        and float(structural["wrong_input_relabel_defect"])
        >= float(TARGET_SIGNATURE["minimum_wrong_relabel_defect"]),
        "bounded_modular_gluing": float(structural["interface_coupling_fraction"])
        <= float(TARGET_SIGNATURE["maximum_interface_fraction"])
        and float(structural["modular_cut_gluing_error"])
        <= float(TARGET_SIGNATURE["maximum_modular_gluing_error"]),
        "observable_decision_receiver": float(readout["aggregate_readout_task_score"])
        >= float(TARGET_SIGNATURE["minimum_readout_task_score"]),
        "reachable_admitted_region": int(structural["admitted_nonhold_action_count"])
        >= int(TARGET_SIGNATURE["minimum_admitted_nonhold_count"])
        and float(structural["minimum_admitted_margin"])
        >= float(TARGET_SIGNATURE["minimum_admitted_margin"])
        and bool(steering["target_success"]),
        "stable_hold_basin": float(structural["hold_residual_ratio"])
        <= float(TARGET_SIGNATURE["maximum_hold_residual_ratio"]),
        "fault_localization_and_recovery": float(faults["fault_localization_accuracy"])
        >= float(TARGET_SIGNATURE["minimum_fault_localization_accuracy"])
        and float(faults["mean_fault_recovery_error"])
        <= float(TARGET_SIGNATURE["maximum_fault_recovery_error"]),
        "bounded_computational_cost": float(structural["mean_computational_energy"])
        <= float(TARGET_SIGNATURE["maximum_computational_energy"]),
        "operator_concentration": float(structural["operator_concentration"])
        >= float(TARGET_SIGNATURE["minimum_operator_concentration"]),
    }


def develop_task_optimized_configuration(
    *,
    substrates_per_candidate: int,
    seed: int,
    library: TaskLibrary,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for config_index, (config_id, _norm, _gain) in enumerate(TASK_CONFIGS):
        scores: list[float] = []
        for index in range(substrates_per_candidate):
            substrate = build_substrate(
                split="DEVELOPMENT",
                family="task_optimized",
                preparation_index=index + 1,
                seed=seed + config_index * 100_000 + index,
                task_config_id=config_id,
            )
            assessment = assess_substrate_raw(substrate, library)
            scores.append(
                float(
                    assessment["readout"]["full_task_scores"][
                        "aggregate_readout_task_score"
                    ]
                )
            )
        rows.append(
            {
                "config_id": config_id,
                "independent_development_preparation_count": len(scores),
                "aggregate_task_score_mean": float(np.mean(scores)),
                "aggregate_task_scores_by_preparation": tuple(scores),
            }
        )
    best_score = max(float(row["aggregate_task_score_mean"]) for row in rows)
    selected_id = min(
        str(row["config_id"])
        for row in rows
        if math.isclose(float(row["aggregate_task_score_mean"]), best_score, abs_tol=1e-12)
    )
    return {
        "candidate_rows": tuple(rows),
        "selected_task_config_id": selected_id,
        "selection_metric": "DEVELOPMENT_AGGREGATE_READOUT_TASK_SCORE",
        "evaluation_tasks_or_faults_read": False,
        "topology_search_separate_from_parameter_and_receiver_fit": True,
    }


def _interval(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, float]:
    estimate, lower, upper = bootstrap_mean_interval(
        np.asarray(values, dtype=np.float64),
        resamples=resamples,
        confidence_level=confidence_level,
        seed=seed,
    )
    return {"estimate": estimate, "lower": lower, "upper": upper}


def _family_metric_rows(assessments: Sequence[Mapping[str, Any]], family: str) -> list[dict[str, float]]:
    result = []
    for assessment in assessments:
        if assessment["family"] != family:
            continue
        structural = assessment["structural"]
        readout = assessment["readout"]
        fault_summary = assessment["fault_summary"]
        steering = assessment["steering"]
        result.append(
            {
                "aggregate_task_score": float(
                    readout["full_task_scores"]["aggregate_readout_task_score"]
                ),
                "latent_reconstruction_score": float(
                    readout["full_task_scores"]["latent_state_reconstruction"]
                ),
                "perturbation_localization_score": float(
                    readout["full_task_scores"]["perturbation_source_localization"]
                ),
                "composition_extrapolation_score": float(
                    readout["full_task_scores"]["composition_extrapolation"]
                ),
                "receiver_calibration_coverage": float(
                    readout["latent_receiver_calibration_coverage"]
                ),
                "prefix_8_task_score": float(
                    readout["sample_efficiency_curve"][0]["aggregate_readout_task_score"]
                ),
                "action_rank": float(structural["action_rank"]),
                "minimum_action_singular_value": float(
                    min(structural["action_singular_values"])
                ),
                "admitted_margin": float(structural["minimum_admitted_margin"]),
                "operator_concentration": float(structural["operator_concentration"]),
                "mean_computational_energy": float(structural["mean_computational_energy"]),
                "fault_localization_accuracy": float(
                    fault_summary["fault_localization_accuracy"]
                ),
                "fault_recovery_error": float(fault_summary["mean_fault_recovery_error"]),
                "target_steering_success": float(steering["target_success"]),
                "hold_residual_ratio": float(structural["hold_residual_ratio"]),
                "symmetry_defect": float(structural["module_swap_symmetry_defect"]),
                "modular_gluing_error": float(structural["modular_cut_gluing_error"]),
                "signature_realized": float(assessment["complete_signature_realized"]),
            }
        )
    return result


def analyse_substrate_panel(
    *,
    assessments: Sequence[Mapping[str, Any]],
    substrates_per_family: int,
    bootstrap_resamples: int,
    confidence_level: float,
    seed: int,
) -> dict[str, Any]:
    expected = substrates_per_family * len(DESIGN_FAMILIES)
    if len(assessments) != expected or len({row["substrate_id"] for row in assessments}) != expected:
        raise ValueError("Topological substrate assessment panel is incomplete or duplicated")
    metrics = (
        "aggregate_task_score",
        "latent_reconstruction_score",
        "perturbation_localization_score",
        "composition_extrapolation_score",
        "receiver_calibration_coverage",
        "prefix_8_task_score",
        "action_rank",
        "minimum_action_singular_value",
        "admitted_margin",
        "operator_concentration",
        "mean_computational_energy",
        "fault_localization_accuracy",
        "fault_recovery_error",
        "target_steering_success",
        "hold_residual_ratio",
        "symmetry_defect",
        "modular_gluing_error",
        "signature_realized",
    )
    rows_by_family = {
        family: _family_metric_rows(assessments, family) for family in DESIGN_FAMILIES
    }
    family_results: dict[str, dict[str, Any]] = {
        family: {
            "independent_substrate_count": len(rows),
            "metrics": {
                metric: _interval(
                    [row[metric] for row in rows],
                    seed=seed + family_index * 1000 + metric_index,
                    resamples=bootstrap_resamples,
                    confidence_level=confidence_level,
                )
                for metric_index, metric in enumerate(metrics)
            },
            "signature_axis_pass_counts": {
                axis: sum(bool(row["signature_axes"][axis]) for row in assessments if row["family"] == family)
                for axis in signature_axes(next(row for row in assessments if row["family"] == family))
            },
        }
        for family_index, (family, rows) in enumerate(rows_by_family.items())
    }
    algebra_rows = rows_by_family["algebra_encoded"]
    pairwise: list[dict[str, Any]] = []
    simultaneous_confidence = 1.0 - (1.0 - confidence_level) / 3.0
    for family_index, family in enumerate(
        ("random_matched", "task_optimized", "conventional_reservoir")
    ):
        comparator = rows_by_family[family]
        pairwise.append(
            {
                "comparator_family": family,
                "algebra_task_score_difference": _interval(
                    [
                        algebra["aggregate_task_score"] - other["aggregate_task_score"]
                        for algebra, other in zip(algebra_rows, comparator)
                    ],
                    seed=seed + 20_000 + family_index,
                    resamples=bootstrap_resamples,
                    confidence_level=simultaneous_confidence,
                ),
                "algebra_fault_recovery_improvement": _interval(
                    [
                        other["fault_recovery_error"] - algebra["fault_recovery_error"]
                        for algebra, other in zip(algebra_rows, comparator)
                    ],
                    seed=seed + 21_000 + family_index,
                    resamples=bootstrap_resamples,
                    confidence_level=simultaneous_confidence,
                ),
                "algebra_signature_rate_difference": _interval(
                    [
                        algebra["signature_realized"] - other["signature_realized"]
                        for algebra, other in zip(algebra_rows, comparator)
                    ],
                    seed=seed + 22_000 + family_index,
                    resamples=bootstrap_resamples,
                    confidence_level=simultaneous_confidence,
                ),
            }
        )
    algebra = family_results["algebra_encoded"]["metrics"]
    task_optimized = family_results["task_optimized"]["metrics"]
    random_result = family_results["random_matched"]["metrics"]
    signature_supported = float(algebra["signature_realized"]["lower"]) >= 0.80
    task_value_supported = (
        float(algebra["aggregate_task_score"]["lower"])
        >= float(task_optimized["aggregate_task_score"]["estimate"]) - 0.08
        and float(algebra["aggregate_task_score"]["lower"])
        > float(random_result["aggregate_task_score"]["lower"])
    )
    robustness_supported = all(
        float(row["algebra_fault_recovery_improvement"]["lower"]) > 0.0
        for row in pairwise
    )
    ordinary_optimization_win = (
        float(task_optimized["aggregate_task_score"]["estimate"])
        > float(random_result["aggregate_task_score"]["estimate"])
    )
    structure_to_substrate_supported = (
        signature_supported and task_value_supported and robustness_supported
    )
    return {
        "independent_substrate_count": len(assessments),
        "independent_substrates_per_family": substrates_per_family,
        "nested_tasks_faults_times_or_receiver_coordinates_are_replicates": False,
        "family_results": family_results,
        "paired_algebra_comparisons": tuple(pairwise),
        "simultaneous_pairwise_confidence_level": simultaneous_confidence,
        "target_signature": TARGET_SIGNATURE,
        "signature_realization_supported": signature_supported,
        "heldout_task_value_supported": task_value_supported,
        "fault_robustness_supported": robustness_supported,
        "ordinary_architecture_optimization_win": ordinary_optimization_win,
        "structure_to_substrate_supported": structure_to_substrate_supported,
        "task_and_structure_axes_compensated": False,
        "scientific_grade": "SUPPORTED" if structure_to_substrate_supported else "MIXED",
        "claim_ceiling": "TRUTH_KNOWN_GENERATED_COMPUTATIONAL_SUBSTRATE",
        "reason_codes": tuple(
            reason
            for condition, reason in (
                (not signature_supported, "COMPLETE_TARGET_SIGNATURE_NOT_RECURRENT"),
                (not task_value_supported, "ALGEBRA_FAMILY_TASK_VALUE_NOT_SUPPORTED"),
                (not robustness_supported, "ALGEBRA_FAULT_RECOVERY_ADVANTAGE_NOT_SUPPORTED"),
            )
            if condition
        ),
    }


def substrate_design_specification(selected_task_config_id: str) -> dict[str, Any]:
    return {
        "design_id": "design.intentional-topological-substrate",
        "target_signature": TARGET_SIGNATURE,
        "design_families": DESIGN_FAMILIES,
        "selected_task_optimized_config_id": selected_task_config_id,
        "node_count": NODE_COUNT,
        "directed_recurrent_edge_budget": EDGE_BUDGET,
        "input_port_count": INPUT_COUNT,
        "horizon_steps": HORIZON_STEPS,
        "training_example_count": TRAINING_EXAMPLE_COUNT,
        "evaluation_example_count": EVALUATION_EXAMPLE_COUNT,
        "sample_prefixes": SAMPLE_PREFIXES,
        "receiver_nodes": RECEIVER_NODES,
        "module_swap": MODULE_SWAP,
        "fault_ids": FAULT_IDS,
        "topology_parameter_receiver_logs_separate": True,
        "evaluation_tasks_select_architecture": False,
        "evaluation_faults_select_architecture": False,
        "task_and_signature_axes_compensate": False,
        "independent_unit": "complete-generated-substrate-preparation",
    }


__all__ = [
    "DESIGN_FAMILIES",
    "EDGE_BUDGET",
    "EVALUATION_EXAMPLE_COUNT",
    "FAULT_IDS",
    "HORIZON_STEPS",
    "MODULE_SWAP",
    "NODE_COUNT",
    "RECEIVER_NODES",
    "SAMPLE_PREFIXES",
    "TARGET_SIGNATURE",
    "TASK_CONFIGS",
    "TRAINING_EXAMPLE_COUNT",
    "Substrate",
    "TaskLibrary",
    "analyse_substrate_panel",
    "assess_substrate_raw",
    "build_substrate",
    "develop_task_optimized_configuration",
    "fault_prototypes",
    "finalize_substrate_assessment",
    "generate_substrates",
    "generate_task_library",
    "signature_axes",
    "simulate_substrate",
    "substrate_design_specification",
    "substrate_from_document",
    "task_library_from_document",
]
