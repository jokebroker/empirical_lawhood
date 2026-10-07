"""Sparse, independently derived trajectory generator for simulator morphism challenges.

This module intentionally shares only simulator morphism challenge interchange records and ordinary
NumPy/SciPy.  It does not import the physical scale morphism method implementation or the physical-scale
dense RC solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.linalg import expm_multiply

from empirical_lawhood.adapters.simulator_morphism_challenges.contracts import SimulatorMorphismChallengeChallengeNomination, SimulatorMorphismChallengeConfig, SimulatorMorphismChallengeDenominatorDescriptor, SimulatorMorphismChallengeGeneratorActionOutcome, SimulatorMorphismChallengeGeneratorOutcome, SimulatorMorphismChallengeGeneratorPairOutcome
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import canonical_json_bytes


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SparseGeneratorOperator:
    state_matrix: sparse.csc_matrix
    left_input: FloatArray
    capacitances: FloatArray
    left_resistance: float
    right_resistance: float


@dataclass(frozen=True, slots=True)
class GeneratorExecution:
    outcome: SimulatorMorphismChallengeGeneratorOutcome
    arrays: dict[str, FloatArray]


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("simulator morphism challenges generator output must be finite")
    return Decimal(str(float(value)))


def assemble_sparse_operator(
    descriptor: SimulatorMorphismChallengeDenominatorDescriptor,
    *,
    left_resistance_factor: float = 1.0,
    right_resistance_factor: float = 1.0,
) -> SparseGeneratorOperator:
    """Derive C dV/dt + G V = boundary inputs using sparse diagonals."""

    if left_resistance_factor <= 0.0 or right_resistance_factor <= 0.0:
        raise ValueError("simulator morphism challenges resistance factors must be positive")
    n_cells = descriptor.scale_cells
    capacitances = np.asarray(descriptor.capacitances_farads, dtype=np.float64)
    interior_resistances = np.asarray(descriptor.interior_resistances_ohms, dtype=np.float64)
    if (
        capacitances.shape != (n_cells,)
        or interior_resistances.shape != (n_cells - 1,)
        or np.min(capacitances) <= 0.0
        or (interior_resistances.size and np.min(interior_resistances) <= 0.0)
    ):
        raise ValueError("simulator morphism challenges sparse generator received an invalid descriptor")
    left_resistance = float(descriptor.left_source_resistance_ohms) * left_resistance_factor
    right_resistance = float(descriptor.right_termination_resistance_ohms) * right_resistance_factor
    edge_conductances = 1.0 / interior_resistances
    diagonal = np.zeros(n_cells, dtype=np.float64)
    if edge_conductances.size:
        diagonal[:-1] += edge_conductances
        diagonal[1:] += edge_conductances
    left_conductance = 1.0 / left_resistance
    right_conductance = 1.0 / right_resistance
    diagonal[0] += left_conductance
    diagonal[-1] += right_conductance
    conductance = sparse.diags(
        diagonals=(-edge_conductances, diagonal, -edge_conductances),
        offsets=(-1, 0, 1),
        shape=(n_cells, n_cells),
        format="csc",
        dtype=np.float64,
    )
    inverse_c = sparse.diags(1.0 / capacitances, format="csc")
    state_matrix = sparse.csc_matrix(-(inverse_c @ conductance))
    left_input = np.zeros(n_cells, dtype=np.float64)
    left_input[0] = left_conductance / capacitances[0]
    if not np.all(np.isfinite(state_matrix.data)) or not np.all(np.isfinite(left_input)):
        raise ValueError("simulator morphism challenges sparse operator is nonfinite")
    return SparseGeneratorOperator(
        state_matrix=state_matrix,
        left_input=left_input,
        capacitances=capacitances,
        left_resistance=left_resistance,
        right_resistance=right_resistance,
    )


def receiver_matrix(descriptor: SimulatorMorphismChallengeDenominatorDescriptor, bin_count: int = 8) -> FloatArray:
    """Independently derive capacitance-weighted equal-width receiver bins."""

    n_cells = descriptor.scale_cells
    if bin_count != 8 or n_cells % bin_count:
        raise ValueError("simulator morphism challenges generator requires an exact equal-width R8 partition")
    capacitances = np.asarray(descriptor.capacitances_farads, dtype=np.float64)
    matrix = np.zeros((bin_count, n_cells), dtype=np.float64)
    width = n_cells // bin_count
    for bin_index in range(bin_count):
        start = bin_index * width
        stop = start + width
        weights = capacitances[start:stop]
        matrix[bin_index, start:stop] = weights / np.sum(weights)
    return matrix


def _affine_action(
    operator: SparseGeneratorOperator,
    states: FloatArray,
    *,
    amplitude_volts: float,
    duration_seconds: float,
    endpoint_seconds: float,
) -> FloatArray:
    if not 0.0 < duration_seconds <= endpoint_seconds:
        raise ValueError("simulator morphism challenges action clock lies outside the panel endpoint")
    n_cells = states.shape[0]
    forcing = operator.left_input * amplitude_volts
    augmented = sparse.bmat(
        [
            [operator.state_matrix, sparse.csc_matrix(forcing[:, None])],
            [sparse.csc_matrix((1, n_cells)), sparse.csc_matrix((1, 1))],
        ],
        format="csc",
    )
    augmented_states = np.vstack((states, np.ones((1, states.shape[1]), dtype=np.float64)))
    driven = np.asarray(expm_multiply(augmented * duration_seconds, augmented_states))[:-1]
    coast = endpoint_seconds - duration_seconds
    if coast > 0.0:
        driven = np.asarray(expm_multiply(operator.state_matrix * coast, driven))
    return np.asarray(driven, dtype=np.float64)


def _metrics(
    operator: SparseGeneratorOperator,
    descriptor: SimulatorMorphismChallengeDenominatorDescriptor,
    states: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    denominator = descriptor.scale_cells * float(descriptor.capacitance_bar_farads)
    q_star = operator.capacitances @ states / denominator
    e_star = np.sum(operator.capacitances[:, None] * states**2, axis=0) / denominator
    v_max_star = np.max(states, axis=0) / float(descriptor.voltage_reference_volts)
    return (
        np.asarray(q_star, dtype=np.float64),
        np.asarray(e_star, dtype=np.float64),
        np.asarray(v_max_star, dtype=np.float64),
    )


def _array_digest(arrays: dict[str, FloatArray]) -> str:
    digest = sha256()
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key], dtype=np.float64)
        digest.update(key.encode("utf-8"))
        digest.update(b"\0float64\0")
        digest.update(canonical_json_bytes(tuple(int(value) for value in values.shape)))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _nomination_set_digest(
    nominations: tuple[SimulatorMorphismChallengeChallengeNomination, ...],
) -> str:
    return sha256(
        canonical_json_bytes(tuple(value.fingerprint() for value in nominations))
    ).hexdigest()


def generate_outcomes(
    *,
    config: SimulatorMorphismChallengeConfig,
    descriptor: SimulatorMorphismChallengeDenominatorDescriptor,
    nominations: tuple[SimulatorMorphismChallengeChallengeNomination, ...],
    implementation_sha256: str,
    outcome_access: OutcomeAccess,
    action_clock_factor: float = 1.0,
    left_resistance_factor: float = 1.0,
    right_resistance_factor: float = 1.0,
) -> GeneratorExecution:
    """Realize one complete scale-view challenge without assigning a verdict."""

    if tuple(value.nomination_id for value in nominations) != tuple(
        sorted({value.nomination_id for value in nominations})
    ):
        raise ValueError("simulator morphism challenges nominations must be sorted and unique")
    if any(
        value.unit_id != descriptor.unit_id or value.scale_cells != descriptor.scale_cells
        for value in nominations
    ):
        raise ValueError("simulator morphism challenges nomination/descriptor identity differs")
    if action_clock_factor <= 0.0:
        raise ValueError("simulator morphism challenges action-clock factor must be positive")
    operator = assemble_sparse_operator(
        descriptor,
        left_resistance_factor=left_resistance_factor,
        right_resistance_factor=right_resistance_factor,
    )
    receiver = receiver_matrix(descriptor, config.receiver_bin_count)
    n_pairs = len(nominations)
    if n_pairs == 0:
        empty_arrays = {
            "present-states": np.empty((0, 2, descriptor.scale_cells), dtype=np.float64)
        }
        outcome = SimulatorMorphismChallengeGeneratorOutcome(
            outcome_id=f"generator-outcome.{descriptor.unit_id}.n{descriptor.scale_cells}",
            unit_id=descriptor.unit_id,
            descriptor_sha256=descriptor.fingerprint(),
            nomination_set_sha256=_nomination_set_digest(nominations),
            generator_implementation_sha256=implementation_sha256,
            pair_outcomes=(),
            arrays_sha256=_array_digest(empty_arrays),
            array_count=1,
            outcome_access=outcome_access,
            scientific_verdict_assigned=False,
        )
        return GeneratorExecution(outcome=outcome, arrays=empty_arrays)
    states = np.empty((descriptor.scale_cells, 2 * n_pairs), dtype=np.float64)
    for pair_index, nomination in enumerate(nominations):
        mode = np.asarray(nomination.mode_values, dtype=np.float64)
        center = np.asarray(nomination.center_values, dtype=np.float64)
        hidden_amplitude = float(nomination.amplitude_volts)
        states[:, 2 * pair_index] = center + hidden_amplitude * mode
        states[:, 2 * pair_index + 1] = center - hidden_amplitude * mode
    lower = float(config.hidden_voltage_envelope[0]) + float(config.hidden_voltage_interior_margin)
    upper = float(config.hidden_voltage_envelope[1]) - float(config.hidden_voltage_interior_margin)
    envelope_valid = np.ones(n_pairs, dtype=np.bool_)
    envelope_valid &= (
        np.min(states.reshape(descriptor.scale_cells, n_pairs, 2), axis=(0, 2)) >= lower
    )
    envelope_valid &= (
        np.max(states.reshape(descriptor.scale_cells, n_pairs, 2), axis=(0, 2)) <= upper
    )

    arrays: dict[str, FloatArray] = {"present-states": states.T.reshape(n_pairs, 2, -1)}
    coordinate_defects = np.zeros(n_pairs, dtype=np.float64)
    history = states.copy()
    history_step_seconds = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    for depth in range(config.history_max_depth + 1):
        observed = receiver @ history
        for pair_index, nomination in enumerate(nominations):
            if depth <= nomination.depth:
                coordinate_defects[pair_index] = max(
                    coordinate_defects[pair_index],
                    float(
                        np.max(
                            np.abs(observed[:, 2 * pair_index] - observed[:, 2 * pair_index + 1])
                        )
                    ),
                )
                envelope_valid[pair_index] &= (
                    np.min(history[:, 2 * pair_index : 2 * pair_index + 2]) >= lower
                    and np.max(history[:, 2 * pair_index : 2 * pair_index + 2]) <= upper
                )
        if depth < config.history_max_depth:
            history = np.asarray(
                expm_multiply(-operator.state_matrix * history_step_seconds, history),
                dtype=np.float64,
            )

    maximum_future_receiver = np.zeros(n_pairs, dtype=np.float64)
    maximum_future_metric = np.zeros(n_pairs, dtype=np.float64)
    future_receiver_by_pair = np.zeros((n_pairs, 3), dtype=np.float64)
    for time_index, time_star in enumerate(config.diagnostic_times_t_star):
        future = np.asarray(
            expm_multiply(
                operator.state_matrix * (float(time_star) * float(descriptor.time_scale_seconds)),
                states,
            ),
            dtype=np.float64,
        )
        arrays[f"diagnostic-{time_index:02d}"] = future.T.reshape(n_pairs, 2, -1)
        receiver_values = receiver @ future
        q_star, e_star, v_max_star = _metrics(operator, descriptor, future)
        for pair_index in range(n_pairs):
            left = 2 * pair_index
            right = left + 1
            receiver_defect = float(
                np.max(np.abs(receiver_values[:, left] - receiver_values[:, right]))
            )
            metric_defect = max(
                abs(float(q_star[left] - q_star[right])),
                abs(float(e_star[left] - e_star[right])),
                abs(float(v_max_star[left] - v_max_star[right])),
            )
            future_receiver_by_pair[pair_index, time_index] = receiver_defect
            maximum_future_receiver[pair_index] = max(
                maximum_future_receiver[pair_index], receiver_defect
            )
            maximum_future_metric[pair_index] = max(
                maximum_future_metric[pair_index], metric_defect
            )

    action_outcomes: list[list[SimulatorMorphismChallengeGeneratorActionOutcome]] = [[] for _ in nominations]
    for amplitude_index, action_amplitude in enumerate(config.action_amplitudes_u_star):
        for duration_index, duration in enumerate(config.action_durations_t_star):
            action_id = f"action.a{amplitude_index:02d}.d{duration_index:02d}"
            duration_seconds = (
                float(duration) * float(descriptor.time_scale_seconds) * action_clock_factor
            )
            realized = _affine_action(
                operator,
                states,
                amplitude_volts=float(action_amplitude) * float(descriptor.voltage_reference_volts),
                duration_seconds=duration_seconds,
                endpoint_seconds=duration_seconds,
            )
            arrays[action_id] = realized.T.reshape(n_pairs, 2, -1)
            q_star, e_star, v_max_star = _metrics(operator, descriptor, realized)
            left_current = -realized[0, :] / operator.left_resistance
            right_current = -realized[-1, :] / operator.right_resistance
            for pair_index, nomination in enumerate(nominations):
                left = 2 * pair_index
                right = left + 1
                midpoint_state = ((realized[:, left] + realized[:, right]) / 2.0)[:, None]
                midpoint_q, midpoint_e, midpoint_v_max = _metrics(
                    operator, descriptor, midpoint_state
                )
                maximum_future_metric[pair_index] = max(
                    maximum_future_metric[pair_index],
                    abs(float(q_star[left] - q_star[right])),
                    abs(float(e_star[left] - e_star[right])),
                    abs(float(v_max_star[left] - v_max_star[right])),
                )
                plus_target = bool(q_star[left] >= float(config.target_charge_minimum_q_star))
                minus_target = bool(q_star[right] >= float(config.target_charge_minimum_q_star))
                plus_sink = bool(v_max_star[left] <= float(config.sink_voltage_maximum_v_star))
                minus_sink = bool(v_max_star[right] <= float(config.sink_voltage_maximum_v_star))
                midpoint_target = bool(
                    midpoint_q[0]
                    >= float(config.target_charge_minimum_q_star)
                    - float(config.midpoint_boundary_tie_epsilon)
                )
                midpoint_sink = bool(
                    midpoint_v_max[0]
                    <= float(config.sink_voltage_maximum_v_star)
                    + float(config.midpoint_boundary_tie_epsilon)
                )
                requested_id = action_id
                action_outcomes[pair_index].append(
                    SimulatorMorphismChallengeGeneratorActionOutcome(
                        action_id=action_id,
                        requested_action_id=requested_id,
                        accepted_action_id=requested_id,
                        applied_action_id=requested_id,
                        realized_action_id=requested_id,
                        amplitude_u_star=action_amplitude,
                        duration_t_star=duration,
                        plus_q_star=_decimal(q_star[left]),
                        minus_q_star=_decimal(q_star[right]),
                        plus_e_star=_decimal(e_star[left]),
                        minus_e_star=_decimal(e_star[right]),
                        plus_v_max_star=_decimal(v_max_star[left]),
                        minus_v_max_star=_decimal(v_max_star[right]),
                        midpoint_q_star=_decimal(midpoint_q[0]),
                        midpoint_e_star=_decimal(midpoint_e[0]),
                        midpoint_v_max_star=_decimal(midpoint_v_max[0]),
                        plus_target_pass=plus_target,
                        minus_target_pass=minus_target,
                        midpoint_target_pass=midpoint_target,
                        plus_sink_pass=plus_sink,
                        minus_sink_pass=minus_sink,
                        midpoint_sink_pass=midpoint_sink,
                        plus_admit=plus_target and plus_sink,
                        minus_admit=minus_target and minus_sink,
                        midpoint_admit=midpoint_target and midpoint_sink,
                        plus_left_current_amperes=_decimal(left_current[left]),
                        minus_left_current_amperes=_decimal(left_current[right]),
                        plus_right_current_amperes=_decimal(right_current[left]),
                        minus_right_current_amperes=_decimal(right_current[right]),
                        requested_accepted_applied_realized_parity=True,
                    )
                )

    pair_records = tuple(
        sorted(
            (
                SimulatorMorphismChallengeGeneratorPairOutcome(
                    nomination_id=nomination.nomination_id,
                    realized_coordinate_defect=_decimal(coordinate_defects[pair_index]),
                    maximum_future_receiver_defect=_decimal(maximum_future_receiver[pair_index]),
                    maximum_future_metric_defect=_decimal(maximum_future_metric[pair_index]),
                    future_receiver_defects=tuple(
                        _decimal(value) for value in future_receiver_by_pair[pair_index]
                    ),
                    action_outcomes=tuple(
                        sorted(action_outcomes[pair_index], key=lambda value: value.action_id)
                    ),
                    voltage_envelope_valid=bool(envelope_valid[pair_index]),
                )
                for pair_index, nomination in enumerate(nominations)
            ),
            key=lambda value: value.nomination_id,
        )
    )
    arrays_sha256 = _array_digest(arrays)
    outcome = SimulatorMorphismChallengeGeneratorOutcome(
        outcome_id=f"generator-outcome.{descriptor.unit_id}.n{descriptor.scale_cells}",
        unit_id=descriptor.unit_id,
        descriptor_sha256=descriptor.fingerprint(),
        nomination_set_sha256=_nomination_set_digest(nominations),
        generator_implementation_sha256=implementation_sha256,
        pair_outcomes=pair_records,
        arrays_sha256=arrays_sha256,
        array_count=len(arrays),
        outcome_access=outcome_access,
        scientific_verdict_assigned=False,
    )
    return GeneratorExecution(outcome=outcome, arrays=arrays)


__all__ = [
    "GeneratorExecution",
    "SparseGeneratorOperator",
    "assemble_sparse_operator",
    "generate_outcomes",
    "receiver_matrix",
]
