"""Outcome-blind source, equation, rank, negative-control, and resource canaries."""

from __future__ import annotations

import ast
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import resource
from time import perf_counter, process_time
from typing import Mapping
from itertools import combinations, permutations

import numpy as np
from scipy.linalg import expm, svdvals
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from empirical_lawhood.adapters.methods.receiver_history_closure.evaluator import (
    _lexical_reference_decision_errors,
    adjudicate_unit_scale,
)
from empirical_lawhood.adapters.methods.receiver_history_closure.conditioning import (
    dimensionless_history_operator,
)
from empirical_lawhood.adapters.methods.receiver_history_closure.discrete_rank import (
    residual_certified_rank_bracket,
)
from empirical_lawhood.adapters.methods.receiver_history_closure.history import (
    assemble_dense_operator,
    coordinate_labels,
    observe_history,
    requested_depths,
)
from empirical_lawhood.adapters.methods.receiver_history_closure.preparation_sampler import (
    _raw_state_bits,
    charge_preservation_error,
    scale_coupled_initial_states,
)
from empirical_lawhood.adapters.methods.receiver_history_closure.inference import _transition_disposition
from empirical_lawhood.adapters.methods.receiver_history_closure.structural_rank import (
    maximum_bipartite_matching_rank,
)
from empirical_lawhood.adapters.methods.receiver_history_closure.lexical_targeting import (
    nominate_targeted_challenges,
)
from empirical_lawhood.adapters.simulators.rc_ladder.generator import (
    _affine_action,
    _factorized_action_components,
    assemble_sparse_operator,
    generate_outcomes,
    receiver_matrix,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess

from .contracts import (
    ReceiverHistoryDenominatorDescriptor,
    ReceiverHistoryDisorderFamily,
    ReceiverHistoryEndpoint,
    ReceiverHistoryConfig,
    ReceiverHistoryGeneratorActionOutcome,
    ReceiverHistoryPhase,
    ReceiverHistoryScientificState,
)
from .authoring import RECEIVER_HISTORY_RESOURCE_PROFILE
from .descriptors import default_config, development_unit_ids, generate_descriptor
from .runtime_contracts import (
    ReceiverHistoryCanaryReport,
    ReceiverHistoryTaskOutputMeasurement,
)
from .workflow import (
    denominator_bundle,
    freeze_nominations,
    history_bundle,
    targeter_bundle,
    target_adjudication_bundle,
    target_generator_bundle,
)


_GENERATOR_PATH = "src/empirical_lawhood/adapters/simulators/rc_ladder/generator.py"
_OBSERVER_PATHS = (
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/conditioning.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/discrete_rank.py",
    "src/empirical_lawhood/adapters/methods/_arb.py",
    "src/empirical_lawhood/adapters/methods/certified_rank.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/history.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/preparation_sampler.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/structural_rank.py",
    "src/empirical_lawhood/adapters/methods/receiver_history_closure/lexical_targeting.py",
)
_EVALUATOR_PATH = "src/empirical_lawhood/adapters/methods/receiver_history_closure/evaluator.py"
_SHARED_PREFIX = "src/empirical_lawhood/adapters/receiver_history/"
_OTHER_CIRCUIT_CONTRACTS = (
    "empirical_lawhood.adapters.physical_scale_morphism",
    "empirical_lawhood.adapters.simulator_morphism_challenges",
    "empirical_lawhood.adapters.history_budget_phase_diagram",
    "empirical_lawhood.adapters.split_cohort_history_budget",
)


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("receiver-history conformance metric must be finite and nonnegative")
    return Decimal(str(float(value)))


def _canary_config(config: ReceiverHistoryConfig | None) -> ReceiverHistoryConfig:
    value = config or default_config(ReceiverHistoryPhase.CANARY)
    if value.phase is not ReceiverHistoryPhase.CANARY:
        raise ValueError("receiver-history conformance requires a canary config")
    return value


def _imports(payload: bytes) -> tuple[str, ...]:
    tree = ast.parse(payload.decode("utf-8"))
    values: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            values.add(node.module)
    return tuple(sorted(values))


def audit_source_firewall(source_files: Mapping[str, bytes]) -> tuple[str, ...]:
    """Verify the exact allowed generator--observer implementation intersection."""

    required = {_GENERATOR_PATH, _EVALUATOR_PATH, *_OBSERVER_PATHS}
    if not required.issubset(source_files):
        raise ValueError("receiver-history firewall audit lacks a claim-bearing source file")
    generator_imports = set(_imports(source_files[_GENERATOR_PATH]))
    observer_imports = {value for path in _OBSERVER_PATHS for value in _imports(source_files[path])}
    evaluator_imports = set(_imports(source_files[_EVALUATOR_PATH]))
    forbidden_generator = {
        value
        for value in generator_imports
        if value.startswith("empirical_lawhood.adapters.methods")
        or any(value.startswith(prefix) for prefix in _OTHER_CIRCUIT_CONTRACTS)
    }
    forbidden_observer = {
        value
        for value in observer_imports
        if value.startswith("empirical_lawhood.adapters.simulators")
        or any(value.startswith(prefix) for prefix in _OTHER_CIRCUIT_CONTRACTS)
    }
    if forbidden_generator or forbidden_observer:
        raise ValueError("receiver-history generator--observer source firewall is violated")
    allowed_local_intersection = {
        "empirical_lawhood.adapters.receiver_history.contracts",
        "empirical_lawhood.kernel.serialization",
    }
    local_intersection = {
        value for value in generator_imports & observer_imports if value.startswith("empirical_lawhood")
    }
    if local_intersection != allowed_local_intersection:
        raise ValueError("receiver-history implementations share an undeclared scientific module")
    if "empirical_lawhood.adapters.simulators.rc_ladder.generator" in evaluator_imports:
        raise ValueError("receiver-history evaluator imports the generator implementation")
    shared_paths = tuple(sorted(path for path in source_files if path.startswith(_SHARED_PREFIX)))
    if not {
        f"{_SHARED_PREFIX}contracts.py",
        f"{_SHARED_PREFIX}runtime_contracts.py",
    }.issubset(shared_paths):
        raise ValueError("receiver-history firewall audit lacks the shared record surface")
    return (
        'independent-generator-source-firewall',
        'independent-observer-source-firewall',
        "scientific-intersection-is-contracts-only",
    )


def _report(report_id: str, checks: set[str], started: float) -> ReceiverHistoryCanaryReport:
    return ReceiverHistoryCanaryReport(
        report_id=report_id,
        check_ids=tuple(sorted(checks)),
        passed=True,
        reason_codes=(),
        wall_time_seconds=_decimal(perf_counter() - started),
        peak_memory_bytes=0,
        projected_evaluation_cpu_hours=Decimal(0),
        projected_evaluation_wall_hours=Decimal(0),
        projected_evaluation_output_bytes=0,
        evaluation_roster_access_count=0,
    )


def _brute_matching_rank(support: np.ndarray) -> int:
    rows, columns = support.shape
    for rank in range(min(rows, columns), -1, -1):
        for selected_rows in combinations(range(rows), rank):
            for selected_columns in combinations(range(columns), rank):
                if any(
                    all(
                        support[row, column]
                        for row, column in zip(
                            selected_rows,
                            ordered_columns,
                            strict=True,
                        )
                    )
                    for ordered_columns in permutations(selected_columns)
                ):
                    return rank
    return 0


def run_structural_rank_canary() -> ReceiverHistoryCanaryReport:
    started = perf_counter()
    checks: set[str] = set()
    for bitmask in range(1 << 12):
        support = np.asarray(
            [(bitmask >> index) & 1 for index in range(12)],
            dtype=np.bool_,
        ).reshape(3, 4)
        if maximum_bipartite_matching_rank(support) != _brute_matching_rank(support):
            raise ValueError('RECEIVER_HISTORY_STRUCTURAL_RANK_CONFORMANCE_FAILED')
    checks.add("exhaustive-3x4-boolean-matching")
    return _report("receiver-history.structural-rank-canary", checks, started)


def run_discrete_rank_canary() -> ReceiverHistoryCanaryReport:
    started = perf_counter()
    fixtures = (
        (np.eye(4, dtype=np.float64), 4),
        (np.vstack((np.eye(4), np.eye(4))), 4),
        (np.asarray([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64), 1),
    )
    for matrix, expected in fixtures:
        lower, upper, _residual, _separation = residual_certified_rank_bracket(matrix)
        if (lower, upper) != (expected, expected):
            raise ValueError('RECEIVER_HISTORY_DISCRETE_RANK_CONFORMANCE_FAILED')
    return _report(
        "receiver-history.discrete-rank-canary",
        {
            "arb-gram-inertia-reconstruction",
            "duplicate-row-rank-control",
            "independent-exact-dyadic-rational-rank",
            "known-rank-high-precision-brackets",
        },
        started,
    )


def run_conditioning_canary() -> ReceiverHistoryCanaryReport:
    started = perf_counter()
    config = default_config(ReceiverHistoryPhase.CANARY)
    descriptor = _uniform_descriptor(16)
    coordinates = tuple(value for value in coordinate_labels(config, 16) if value.depth == 4)
    spectra = {
        value.resolution_epsilon: np.linalg.svd(
            dimensionless_history_operator(config, descriptor, value),
            compute_uv=False,
        )
        for value in coordinates
    }
    low = spectra[Decimal("0.0025")]
    primary = spectra[Decimal("0.005")]
    high = spectra[Decimal("0.01")]
    if not (
        np.allclose(low, 2.0 * primary, rtol=1e-12, atol=1e-12)
        and np.allclose(high, 0.5 * primary, rtol=1e-12, atol=1e-12)
    ):
        raise ValueError('RECEIVER_HISTORY_CONDITIONING_CONFORMANCE_FAILED')
    return _report(
        "receiver-history.conditioning-canary",
        {"absolute-resolution-scaling", "capacitance-energy-normalization"},
        started,
    )


def run_untouched_sampler_canary() -> ReceiverHistoryCanaryReport:
    started = perf_counter()
    descriptors = {scale: _uniform_descriptor(scale) for scale in (64, 128, 256)}
    states = scale_coupled_initial_states(
        seed=sha256(b"receiver-history-untouched-canary").digest(),
        unit_id="canary.untouched.block-00",
        descriptors=descriptors,
    )
    if charge_preservation_error(states, descriptors) > 1e-15:
        raise ValueError('RECEIVER_HISTORY_UNTOUCHED_SCALE_MAP_CONFORMANCE_FAILED')
    shuffled = dict(states)
    shuffled[64] = states[64][::-1].copy()
    if charge_preservation_error(shuffled, descriptors) <= 1e-6:
        raise ValueError('RECEIVER_HISTORY_UNTOUCHED_SCALE_SHUFFLE_NEGATIVE_FAILED')
    raw = np.asarray([[0.25, np.nan, np.inf, -np.inf]], dtype=np.float64)
    retained = _raw_state_bits(raw)
    if not np.array_equal(retained.view(np.float64), raw, equal_nan=True):
        raise ValueError('RECEIVER_HISTORY_INVALID_RAW_BIT_RETENTION_FAILED')
    return _report(
        "receiver-history.untouched-sampler-canary",
        {
            "capacitance-weighted-direct-charge-map",
            "invalid-ieee754-bit-retention",
            "scale-shuffled-negative",
        },
        started,
    )


def run_certificate_canary(
    config: ReceiverHistoryConfig | None = None,
) -> ReceiverHistoryCanaryReport:
    """Exercise the independent sparse objective family without observer input."""

    started = perf_counter()
    config = _canary_config(config)
    descriptor = _uniform_descriptor(16)
    generated = generate_outcomes(
        config=config,
        descriptor=descriptor,
        nominations=(),
        implementation_sha256="1" * 64,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    ).outcome
    checks = generated.sparse_certificate_checks
    coordinates = {value.coordinate_id for value in coordinate_labels(config, 16)}
    by_coordinate: dict[str, int] = {value: 0 for value in coordinates}
    for check in checks:
        matches = tuple(
            value
            for value in coordinates
            if check.objective_key.startswith(f"optimization.{value}.")
        )
        if len(matches) != 1:
            raise ValueError('RECEIVER_HISTORY_SPARSE_CERTIFICATE_UNKNOWN_COORDINATE')
        coordinate_id = matches[0]
        by_coordinate[coordinate_id] += 1
        if (
            not check.complete_family_certificate
            or check.primal_residual > config.optimization_tolerance
            or check.duality_gap > config.optimization_tolerance
        ):
            raise ValueError('RECEIVER_HISTORY_SPARSE_CERTIFICATE_INCOMPLETE')
    objective_count = 1 + 4 * (
        len(config.action_amplitudes_u_star) * len(config.action_durations_t_star)
    )
    if not by_coordinate or set(by_coordinate.values()) != {objective_count}:
        raise ValueError('RECEIVER_HISTORY_SPARSE_CERTIFICATE_OBJECTIVE_ROSTER_FAILED')
    # Deleting even one objective must make the exact configured roster fail.
    if len(checks[:-1]) == len(coordinates) * objective_count:
        raise ValueError('RECEIVER_HISTORY_SPARSE_CERTIFICATE_NEGATIVE_CONTROL_FAILED')
    return _report(
        "receiver-history.certificate-canary",
        {
            "independent-sparse-primal-dual-family",
            f"exact-{objective_count}-objective-per-coordinate-roster",
            "missing-objective-negative",
        },
        started,
    )


def _action_fixture(
    config: ReceiverHistoryConfig,
    *,
    plus_admit: bool,
    minus_robust_admit: bool,
    midpoint_admit: bool,
) -> ReceiverHistoryGeneratorActionOutcome:
    target = config.target_charge_minimum_q_star
    sink = config.sink_voltage_maximum_v_star
    target_margin = config.target_decision_margin
    sink_margin = config.sink_decision_margin
    high_q = target + target_margin
    low_q = max(Decimal(0), target - target_margin)
    low_v = max(Decimal(0), sink - sink_margin)
    high_v = sink + sink_margin
    plus_q = high_q if plus_admit else low_q
    plus_v = low_v if plus_admit else high_v
    minus_q = high_q if minus_robust_admit else low_q
    minus_v = low_v if minus_robust_admit else high_v
    midpoint_q = high_q if midpoint_admit else low_q
    midpoint_v = low_v if midpoint_admit else high_v
    return ReceiverHistoryGeneratorActionOutcome(
        action_id="action.canary.lexical",
        requested_action_id="action.canary.lexical",
        accepted_action_id="action.canary.lexical",
        applied_action_id="action.canary.lexical",
        realized_action_id="action.canary.lexical",
        amplitude_u_star=Decimal("0.5"),
        duration_t_star=Decimal("0.5"),
        plus_q_star=plus_q,
        minus_q_star=minus_q,
        plus_e_star=Decimal(0),
        minus_e_star=Decimal(0),
        plus_v_max_star=plus_v,
        minus_v_max_star=minus_v,
        midpoint_q_star=midpoint_q,
        midpoint_e_star=Decimal(0),
        midpoint_v_max_star=midpoint_v,
        plus_target_pass=plus_q >= target,
        minus_target_pass=minus_q >= target,
        midpoint_target_pass=midpoint_q >= target,
        plus_sink_pass=plus_v <= sink,
        minus_sink_pass=minus_v <= sink,
        midpoint_sink_pass=midpoint_v <= sink,
        plus_admit=plus_admit,
        minus_admit=minus_robust_admit,
        midpoint_admit=midpoint_admit,
        plus_left_current_amperes=Decimal(0),
        minus_left_current_amperes=Decimal(0),
        plus_right_current_amperes=Decimal(0),
        minus_right_current_amperes=Decimal(0),
        requested_accepted_applied_realized_parity=True,
    )


def run_lexical_direction_canary() -> ReceiverHistoryCanaryReport:
    started = perf_counter()
    config = default_config(ReceiverHistoryPhase.CANARY)
    for midpoint in (False, True):
        false_safe = _lexical_reference_decision_errors(
            _action_fixture(
                config,
                plus_admit=True,
                minus_robust_admit=False,
                midpoint_admit=midpoint,
            ),
            config,
        )
        false_hold = _lexical_reference_decision_errors(
            _action_fixture(
                config,
                plus_admit=False,
                minus_robust_admit=True,
                midpoint_admit=midpoint,
            ),
            config,
        )
        if false_safe != (True, False) or false_hold != (False, True):
            raise ValueError('RECEIVER_HISTORY_LEXICAL_REFERENCE_DIRECTION_FAILED')
    return _report(
        "receiver-history.lexical-direction-canary",
        {"plus-is-exact-reference", "minus-is-robust-member", "midpoint-label-invariant"},
        started,
    )


def run_rank_separation_canary() -> ReceiverHistoryCanaryReport:
    started = perf_counter()
    reports = (
        run_structural_rank_canary(),
        run_discrete_rank_canary(),
        run_conditioning_canary(),
    )
    return _report(
        "receiver-history.rank-separation-canary",
        {check for report in reports for check in report.check_ids},
        started,
    )


def run_transition_canary() -> ReceiverHistoryCanaryReport:
    started = perf_counter()
    opposed = ReceiverHistoryScientificState.OPPOSED
    limited = ReceiverHistoryScientificState.TARGETABILITY_LIMITED
    informative = ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
    if (
        _transition_disposition((opposed, limited, informative))
        is not ReceiverHistoryScientificState.SUPPORTED
    ):
        raise ValueError('RECEIVER_HISTORY_MONOTONE_TRANSITION_CONTROL_FAILED')
    if (
        _transition_disposition((opposed, ReceiverHistoryScientificState.MIXED, informative))
        is not ReceiverHistoryScientificState.MIXED
    ):
        raise ValueError('RECEIVER_HISTORY_MIXED_TRANSITION_NEGATIVE_FAILED')
    if (
        _transition_disposition((informative, limited, opposed))
        is not ReceiverHistoryScientificState.NONMONOTONE_PHASE_PATTERN
    ):
        raise ValueError('RECEIVER_HISTORY_NONMONOTONE_TRANSITION_NEGATIVE_FAILED')
    return _report(
        "receiver-history.transition-canary",
        {"monotone-three-state-control", "mixed-never-supported", "reversal-never-supported"},
        started,
    )


def run_factorized_action_canary(
    config: ReceiverHistoryConfig | None = None,
) -> ReceiverHistoryCanaryReport:
    started = perf_counter()
    config = _canary_config(config)
    descriptor = _uniform_descriptor(64)
    operator = assemble_sparse_operator(descriptor)
    rng = np.random.default_rng(27052026)
    states = rng.uniform(0.2, 0.8, size=(64, 7))
    endpoint = float(config.panel_endpoint_t_star) * float(descriptor.time_scale_seconds)
    for duration in config.action_durations_t_star:
        duration_seconds = float(duration) * float(descriptor.time_scale_seconds)
        homogeneous, response = _factorized_action_components(
            operator,
            states,
            duration_seconds=duration_seconds,
            endpoint_seconds=endpoint,
        )
        for amplitude in config.action_amplitudes_u_star:
            amplitude_volts = float(amplitude) * float(descriptor.voltage_reference_volts)
            factorized = homogeneous + amplitude_volts * response[:, None]
            direct = _affine_action(
                operator,
                states,
                amplitude_volts=amplitude_volts,
                duration_seconds=duration_seconds,
                endpoint_seconds=endpoint,
            )
            if not np.allclose(factorized, direct, atol=2e-12, rtol=2e-12):
                raise ValueError('RECEIVER_HISTORY_FACTORIZED_ACTION_EQUIVALENCE_FAILED')
    return _report(
        "receiver-history.factorized-action-canary",
        {
            "all-"
            f"{len(config.action_amplitudes_u_star) * len(config.action_durations_t_star)}"
            "-actions-factorized-versus-direct"
        },
        started,
    )


def run_one_pass_history_canary() -> ReceiverHistoryCanaryReport:
    started = perf_counter()
    config = default_config(ReceiverHistoryPhase.CANARY)
    descriptor = _uniform_descriptor(64)
    operator = assemble_sparse_operator(descriptor)
    rng = np.random.default_rng(9062026)
    states = rng.uniform(0.2, 0.8, size=(64, 11))
    lag = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    one_pass = np.asarray(
        expm_multiply(
            -operator.state_matrix,
            states,
            start=0.0,
            stop=lag * config.history_max_depth,
            num=config.history_max_depth + 1,
            endpoint=True,
            traceA=float(-operator.state_matrix.diagonal().sum()),
        )
    )
    sequential = [states]
    for _ in range(config.history_max_depth):
        sequential.append(np.asarray(expm_multiply(-operator.state_matrix * lag, sequential[-1])))
    if not np.allclose(one_pass, np.asarray(sequential), atol=3e-11, rtol=3e-11):
        raise ValueError('RECEIVER_HISTORY_ONE_PASS_HISTORY_EQUIVALENCE_FAILED')
    return _report(
        "receiver-history.one-pass-history-canary",
        {"32-time-krylov-versus-31-sequential-propagations"},
        started,
    )


def _uniform_descriptor(scale_cells: int) -> ReceiverHistoryDenominatorDescriptor:
    seed = sha256(f"receiver-history-uniform-canary-n{scale_cells}".encode("ascii")).digest()
    base = generate_descriptor(
        unit_id=f"canary.truth-known.n{scale_cells}",
        family=ReceiverHistoryDisorderFamily.SMOOTH_PERIODIC_12,
        scale_cells=scale_cells,
        seed=seed,
    )
    return replace(
        base,
        latent_field_sha256=sha256(b"uniform-zero-field").hexdigest(),
        capacitances_farads=(base.capacitance_bar_farads,) * scale_cells,
        interior_resistances_ohms=(base.resistance_bar_ohms,) * (scale_cells - 1),
    )


def _analytic_equilibrium(
    descriptor: ReceiverHistoryDenominatorDescriptor, amplitude: float
) -> np.ndarray:
    left = float(descriptor.left_source_resistance_ohms)
    right = float(descriptor.right_termination_resistance_ohms)
    interior = np.asarray(descriptor.interior_resistances_ohms, dtype=np.float64)
    current = amplitude / (left + float(np.sum(interior)) + right)
    preceding = np.concatenate((np.zeros(1), np.cumsum(interior)))
    return amplitude - current * (left + preceding)


def _rank_audit(descriptor: ReceiverHistoryDenominatorDescriptor) -> None:
    config = default_config(ReceiverHistoryPhase.CANARY)
    operator = assemble_dense_operator(descriptor)
    lag = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    inverse = expm(-operator.state_matrix * lag)
    blocks = []
    current = operator.receiver.copy()
    probe_depths = requested_depths(config, descriptor.scale_cells)
    for depth in range(config.history_max_depth + 1):
        if depth:
            current = current @ inverse
        blocks.append(current.copy())
        if depth not in probe_depths:
            continue
        stack = np.vstack(blocks)
        numpy_values = np.linalg.svd(stack, compute_uv=False)
        scipy_values = svdvals(stack)
        if not np.allclose(numpy_values, scipy_values, atol=1e-10, rtol=1e-11):
            raise ValueError('RECEIVER_HISTORY_CANARY_SVD_AUDIT_FAILED')
        duplicated = np.vstack([operator.receiver] * (depth + 1))
        if np.linalg.matrix_rank(duplicated, tol=1e-12) > 8:
            raise ValueError('RECEIVER_HISTORY_DUPLICATE_PRESENT_CONTROL_FAILED')
        shuffled = np.vstack(tuple(reversed(blocks)))
        if not np.allclose(
            np.linalg.svd(shuffled, compute_uv=False),
            numpy_values,
            atol=1e-10,
            rtol=1e-11,
        ):
            raise ValueError('RECEIVER_HISTORY_SHUFFLED_LAG_CONTROL_FAILED')


def _run_generator_canary(
    config: ReceiverHistoryConfig | None = None,
) -> ReceiverHistoryCanaryReport:
    """Qualify the sparse generator without importing observer implementation code."""

    start = perf_counter()
    config = _canary_config(config)
    checks: set[str] = set()
    for scale_cells in config.scale_cells:
        descriptor = _uniform_descriptor(scale_cells)
        operator = assemble_sparse_operator(descriptor)
        equilibrium = np.asarray(
            spsolve(-operator.state_matrix, operator.left_input), dtype=np.float64
        )
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError('RECEIVER_HISTORY_GENERATOR_MANUFACTURED_SOLUTION_FAILED')
        if np.max(np.abs(operator.state_matrix @ equilibrium + operator.left_input)) > 1e-12:
            raise ValueError('RECEIVER_HISTORY_GENERATOR_CURRENT_BALANCE_RESIDUAL_FAILED')
        initial = np.linspace(0.1, 0.9, scale_cells, dtype=np.float64)
        duration = float(config.panel_endpoint_t_star) * float(descriptor.time_scale_seconds)
        future = np.asarray(
            expm_multiply(
                operator.state_matrix * duration,
                initial,
            ),
            dtype=np.float64,
        )
        two_half_steps = np.asarray(
            expm_multiply(
                operator.state_matrix * (duration / 2.0),
                expm_multiply(
                    operator.state_matrix * (duration / 2.0),
                    initial,
                ),
            ),
            dtype=np.float64,
        )
        if not np.allclose(future, two_half_steps, atol=2e-12, rtol=2e-12):
            raise ValueError('RECEIVER_HISTORY_GENERATOR_PROPAGATOR_SEMIGROUP_FAILED')
        checks.update(
            {
                f"n{scale_cells}-generator-current-balance",
                f"n{scale_cells}-generator-manufactured-equilibrium",
            }
        )
    return ReceiverHistoryCanaryReport(
        report_id="receiver-history.generator-canary",
        check_ids=tuple(sorted(checks)),
        passed=True,
        reason_codes=(),
        wall_time_seconds=_decimal(perf_counter() - start),
        peak_memory_bytes=0,
        projected_evaluation_cpu_hours=Decimal(0),
        projected_evaluation_wall_hours=Decimal(0),
        projected_evaluation_output_bytes=0,
        evaluation_roster_access_count=0,
    )


def run_generator_canary(
    config: ReceiverHistoryConfig | None = None,
) -> ReceiverHistoryCanaryReport:
    with threadpool_limits(limits=2):
        return _run_generator_canary(config)


def _run_observer_canary(
    config: ReceiverHistoryConfig | None = None,
) -> ReceiverHistoryCanaryReport:
    """Qualify dense observer equations, R8 and history controls in isolation."""

    start = perf_counter()
    config = _canary_config(config)
    checks: set[str] = set()
    for scale_cells in config.scale_cells:
        descriptor = _uniform_descriptor(scale_cells)
        operator = assemble_dense_operator(descriptor)
        equilibrium = -np.linalg.solve(operator.state_matrix, operator.left_input)
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError('RECEIVER_HISTORY_OBSERVER_MANUFACTURED_SOLUTION_FAILED')
        if np.max(np.abs(operator.state_matrix @ equilibrium + operator.left_input)) > 1e-12:
            raise ValueError('RECEIVER_HISTORY_OBSERVER_CURRENT_BALANCE_RESIDUAL_FAILED')
        _rank_audit(descriptor)
        checks.update(
            {
                f"n{scale_cells}-observer-manufactured-equilibrium",
                f"n{scale_cells}-observer-rank-and-history-controls",
            }
        )
    return ReceiverHistoryCanaryReport(
        report_id="receiver-history.observer-canary",
        check_ids=tuple(sorted(checks)),
        passed=True,
        reason_codes=(),
        wall_time_seconds=_decimal(perf_counter() - start),
        peak_memory_bytes=0,
        projected_evaluation_cpu_hours=Decimal(0),
        projected_evaluation_wall_hours=Decimal(0),
        projected_evaluation_output_bytes=0,
        evaluation_roster_access_count=0,
    )


def run_observer_canary(
    config: ReceiverHistoryConfig | None = None,
) -> ReceiverHistoryCanaryReport:
    with threadpool_limits(limits=2):
        return _run_observer_canary(config)


def _run_cross_implementation_conformance(
    *,
    implementation_sha256: str,
    generator_report: ReceiverHistoryCanaryReport,
    observer_report: ReceiverHistoryCanaryReport,
    independence_check_ids: tuple[str, ...],
    config: ReceiverHistoryConfig | None = None,
) -> ReceiverHistoryCanaryReport:
    """Compare separately qualified implementations on the shared finite cases."""

    if (
        not generator_report.passed
        or not observer_report.passed
        or generator_report.report_id != "receiver-history.generator-canary"
        or observer_report.report_id != "receiver-history.observer-canary"
        or not independence_check_ids
    ):
        raise ValueError('RECEIVER_HISTORY_CANARY_PREREQUISITE_FAILED')

    start = perf_counter()
    config = _canary_config(config)
    checks: set[str] = {
        *generator_report.check_ids,
        *observer_report.check_ids,
        *independence_check_ids,
    }
    for scale_cells in config.scale_cells:
        descriptor = _uniform_descriptor(scale_cells)
        dense = assemble_dense_operator(descriptor)
        sparse = assemble_sparse_operator(descriptor)
        if not np.allclose(
            dense.state_matrix, sparse.state_matrix.toarray(), atol=1e-14, rtol=1e-14
        ):
            raise ValueError('RECEIVER_HISTORY_CANARY_OPERATOR_MISMATCH')
        if not np.allclose(dense.left_input, sparse.left_input, atol=1e-14, rtol=1e-14):
            raise ValueError('RECEIVER_HISTORY_CANARY_INPUT_MISMATCH')
        if not np.allclose(dense.receiver, receiver_matrix(descriptor), atol=1e-14, rtol=1e-14):
            raise ValueError('RECEIVER_HISTORY_CANARY_RECEIVER_MISMATCH')
        equilibrium = -np.linalg.solve(dense.state_matrix, dense.left_input)
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError('RECEIVER_HISTORY_CANARY_MANUFACTURED_SOLUTION_FAILED')
        if np.max(np.abs(dense.state_matrix @ equilibrium + dense.left_input)) > 1e-12:
            raise ValueError('RECEIVER_HISTORY_CANARY_CURRENT_BALANCE_RESIDUAL_FAILED')
        initial = np.column_stack((np.linspace(0.1, 0.9, scale_cells), analytic))
        duration = 0.013 * float(descriptor.time_scale_seconds)
        dense_future = expm(dense.state_matrix * duration) @ initial
        sparse_future = np.asarray(expm_multiply(sparse.state_matrix * duration, initial))
        if not np.allclose(dense_future, sparse_future, atol=1e-11, rtol=1e-11):
            raise ValueError('RECEIVER_HISTORY_CANARY_PROPAGATOR_MISMATCH')
        _rank_audit(descriptor)

        seed = sha256(f"receiver-history-conformance:{scale_cells}".encode("ascii")).digest()
        challenge_descriptor = generate_descriptor(
            unit_id=f"canary.truth-known.n{scale_cells}",
            family=ReceiverHistoryDisorderFamily.CORRELATED_FIELD_12,
            scale_cells=scale_cells,
            seed=seed,
        )
        history = observe_history(
            config=config,
            descriptor=challenge_descriptor,
        )
        observer = nominate_targeted_challenges(
            config=config,
            descriptor=challenge_descriptor,
            history_forecast=history.forecast,
            implementation_sha256=implementation_sha256,
        )
        if not observer.nominations:
            raise ValueError('RECEIVER_HISTORY_CANARY_NO_NOMINATIONS')
        generator = generate_outcomes(
            config=config,
            descriptor=challenge_descriptor,
            nominations=observer.geometries,
            implementation_sha256=implementation_sha256,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
        adjudication = adjudicate_unit_scale(
            config=config,
            descriptor=challenge_descriptor,
            forecast=history.forecast,
            geometries=observer.geometries,
            nominations=observer.nominations,
            generator_outcome=generator.outcome,
            method_freeze=None,
        )
        if not adjudication.generator_observer_agreement:
            raise ValueError('RECEIVER_HISTORY_CANARY_CROSS_IMPLEMENTATION_FAILED')
        for left_factor, right_factor, clock_factor in (
            (1.25, 1.0, 1.0),
            (1.0, 1.25, 1.0),
            (1.0, 1.0, 0.9),
        ):
            wrong = generate_outcomes(
                config=config,
                descriptor=challenge_descriptor,
                nominations=observer.geometries,
                implementation_sha256=implementation_sha256,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                left_resistance_factor=left_factor,
                right_resistance_factor=right_factor,
                action_clock_factor=clock_factor,
            )
            wrong_adjudication = adjudicate_unit_scale(
                config=config,
                descriptor=challenge_descriptor,
                forecast=history.forecast,
                geometries=observer.geometries,
                nominations=observer.nominations,
                generator_outcome=wrong.outcome,
                method_freeze=None,
            )
            if wrong_adjudication.generator_observer_agreement:
                raise ValueError('RECEIVER_HISTORY_CANARY_NEGATIVE_CONTROL_FAILED')
        checks.update(
            {
                f"n{scale_cells}-dense-sparse-equations",
                f"n{scale_cells}-manufactured-equilibrium",
                f"n{scale_cells}-rank-svd-and-history-controls",
                f"n{scale_cells}-source-termination-clock-negatives",
            }
        )
    return ReceiverHistoryCanaryReport(
        report_id="receiver-history.numerical-conformance",
        check_ids=tuple(sorted(checks)),
        passed=True,
        reason_codes=(),
        wall_time_seconds=_decimal(perf_counter() - start),
        peak_memory_bytes=0,
        projected_evaluation_cpu_hours=Decimal(0),
        projected_evaluation_wall_hours=Decimal(0),
        projected_evaluation_output_bytes=0,
        evaluation_roster_access_count=0,
    )


def run_cross_implementation_conformance(
    *,
    implementation_sha256: str,
    generator_report: ReceiverHistoryCanaryReport,
    observer_report: ReceiverHistoryCanaryReport,
    independence_check_ids: tuple[str, ...],
    config: ReceiverHistoryConfig | None = None,
) -> ReceiverHistoryCanaryReport:
    with threadpool_limits(limits=2):
        return _run_cross_implementation_conformance(
            implementation_sha256=implementation_sha256,
            generator_report=generator_report,
            observer_report=observer_report,
            independence_check_ids=independence_check_ids,
            config=config,
        )


def run_numerical_conformance(*, implementation_sha256: str) -> ReceiverHistoryCanaryReport:
    """Compatibility entry point for the complete split N16/N32 canary."""

    generator = run_generator_canary()
    observer = run_observer_canary()
    return run_cross_implementation_conformance(
        implementation_sha256=implementation_sha256,
        generator_report=generator,
        observer_report=observer,
        independence_check_ids=("scientific-intersection-is-contracts-only",),
    )


def _run_resource_canary(
    *,
    canary_config: ReceiverHistoryConfig,
    implementation_sha256: str,
    block_index: int = 0,
    scale_cells: tuple[int, ...] | None = None,
    disorder_family: ReceiverHistoryDisorderFamily,
    endpoint_mask: tuple[ReceiverHistoryEndpoint, ...],
) -> ReceiverHistoryCanaryReport:
    """Measure one endpoint-projected complete unit without roster access."""

    if canary_config.phase is not ReceiverHistoryPhase.CANARY:
        raise ValueError("resource canary requires the issued canary profile")
    config = replace(
        canary_config,
        config_id='receiver-history.resource-canary-development-profile',
        phase=ReceiverHistoryPhase.DEVELOPMENT,
        scale_cells=canary_config.excluded_resource_canary_scale_cells,
        excluded_resource_canary_scale_cells=(),
        unit_ids=development_unit_ids(),
    )
    scales = config.scale_cells if scale_cells is None else scale_cells
    if scales != config.scale_cells:
        raise ValueError("resource canary scale roster differs from the excluded config")
    if not 0 <= block_index < 6:
        raise ValueError("resource canary block index lies outside the excluded roster")
    frozen_mask = tuple(sorted(set(endpoint_mask), key=lambda value: value.value))
    if not frozen_mask or frozen_mask != endpoint_mask:
        raise ValueError("resource canary endpoint mask must be sorted and unique")
    block_id = f"block-{block_index:02d}"
    seed = sha256(f"receiver-history-resource-canary-{block_index:02d}".encode("ascii")).digest()
    wall_start = perf_counter()
    cpu_start = process_time()
    checks: set[str] = set()
    unit_id = next(
        value
        for value in config.unit_ids
        if disorder_family.value in value and value.endswith("block-00")
    )
    denominators = denominator_bundle(config=config, unit_id=unit_id, seed=seed)
    history = history_bundle(config=config, denominators=denominators)
    targeter = targeter_bundle(
        config=config,
        denominators=denominators,
        history=history.bundle,
        implementation_sha256=implementation_sha256,
        endpoint_mask=frozenset(endpoint_mask),
    )
    nomination_freeze = freeze_nominations(
        history=history.bundle,
        observer=targeter.bundle,
        untouched=None,
        method_freeze=None,
    )
    generated = target_generator_bundle(
        config=config,
        denominators=denominators,
        nomination_freeze=nomination_freeze,
        implementation_sha256=implementation_sha256,
    )
    adjudication = target_adjudication_bundle(
        config=config,
        denominators=denominators,
        history=history.bundle,
        observer=targeter.bundle,
        generator=generated.bundle,
        generator_arrays_payload=generated.packed_arrays.payload,
        generator_arrays_manifest=generated.packed_arrays.manifest,
        method_freeze=None,
        power_atlas=None,
        maximum_array_bytes=512 * 1024**2,
    )
    disagreement_ids = tuple(
        sorted(
            {
                *(
                    value.adjudication_id
                    for value in adjudication.adjudications
                    if not value.generator_observer_agreement
                ),
                *(
                    value.adjudication_id
                    for value in adjudication.targeted_adjudications
                    if not value.valid or not value.generator_observer_agreement
                ),
            }
        )
    )
    for adjudication_id in disagreement_ids:
        checks.add(
            "implementation-disagreement."
            + sha256(adjudication_id.encode("ascii")).hexdigest()[:16]
        )
    task_output_bytes = {
        "descriptor": len(denominators.canonical_bytes()),
        "history": sum(
            (
                len(history.bundle.canonical_bytes()),
                len(history.packed_arrays.payload),
                len(history.packed_arrays.manifest.canonical_bytes()),
            )
        ),
        "targeter": sum(
            (
                len(targeter.bundle.canonical_bytes()),
                len(targeter.packed_arrays.payload),
                len(targeter.packed_arrays.manifest.canonical_bytes()),
            )
        ),
        "generator": sum(
            (
                len(generated.bundle.canonical_bytes()),
                len(generated.packed_arrays.payload),
                len(generated.packed_arrays.manifest.canonical_bytes()),
            )
        ),
        "adjudication": len(adjudication.canonical_bytes()),
    }
    task_output_limits = {
        "descriptor": RECEIVER_HISTORY_RESOURCE_PROFILE.small.output_bytes,
        "history": RECEIVER_HISTORY_RESOURCE_PROFILE.observer.output_bytes,
        "targeter": RECEIVER_HISTORY_RESOURCE_PROFILE.observer.output_bytes,
        "generator": RECEIVER_HISTORY_RESOURCE_PROFILE.generator.output_bytes,
        "adjudication": RECEIVER_HISTORY_RESOURCE_PROFILE.small.output_bytes,
    }
    per_task_output_passed = all(
        task_output_bytes[key] <= task_output_limits[key] for key in task_output_bytes
    )
    task_output_measurements = tuple(
        ReceiverHistoryTaskOutputMeasurement(
            task_class_id=key,
            measured_output_bytes=task_output_bytes[key],
            output_ceiling_bytes=task_output_limits[key],
            passed=task_output_bytes[key] <= task_output_limits[key],
        )
        for key in sorted(task_output_bytes)
    )
    checks.update(
        f"output-ceiling-exceeded.{key}"
        for key in task_output_bytes
        if task_output_bytes[key] > task_output_limits[key]
    )
    output_bytes = sum(task_output_bytes.values())
    checks.update(
        {
            "coordinates-k0-k4-k8-b1-2-b1-only",
            'no-untouched-512-preparation-or-512-cell-work',
            "all-three-primary-scale-views",
            f"family-{disorder_family.value}",
            *(f"endpoint-{value.value.lower()}" for value in endpoint_mask),
            f"resource-profile-{RECEIVER_HISTORY_RESOURCE_PROFILE.profile_id}",
            "per-task-output-ceilings-audited",
            "cpu-projection-bounded-by-issued-two-core-envelope",
            *(
                ('reduced-action-menu-profile',)
                if canary_config.maximum_parallel_tasks == 6
                else ()
            ),
        }
    )
    elapsed_wall = perf_counter() - wall_start
    raw_elapsed_cpu = process_time() - cpu_start
    # six resource qualification tasks can execute concurrently under one scheduler. If an operator
    # selects a same-process executor, process CPU deltas include sibling tasks
    # and cannot be attributed repeatedly. Cap each report by its issued
    # two-core envelope; retain the smaller isolated measurement.
    elapsed_cpu = min(raw_elapsed_cpu, elapsed_wall * 2.0)
    measured_wall_time = _decimal(elapsed_wall)
    measured_cpu_time = _decimal(elapsed_cpu)
    per_complete_unit_factor = 36 if canary_config.maximum_parallel_tasks == 6 else 90
    projected_output = output_bytes * per_complete_unit_factor
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resource_path_passed = (
        projected_output <= 4 * 1024**3
        and per_task_output_passed
        and peak_rss_kib * 1024 <= 12 * 1024**3
    )
    projected_wall_hours = (
        measured_wall_time
        * Decimal(per_complete_unit_factor)
        / Decimal(canary_config.maximum_parallel_tasks * 3600)
    )
    checks.add(
        "runtime-target-met" if projected_wall_hours <= Decimal(7) else "runtime-target-at-risk"
    )
    passed = resource_path_passed and not disagreement_ids
    reasons = tuple(
        sorted(
            {
                *(('RECEIVER_HISTORY_RESOURCE_PATH_INADEQUATE',) if not resource_path_passed else ()),
                *(('RECEIVER_HISTORY_RESOURCE_CANARY_IMPLEMENTATION_DISAGREEMENT',) if disagreement_ids else ()),
            }
        )
    )
    return ReceiverHistoryCanaryReport(
        report_id=f"receiver-history.resource-canary.{block_id}",
        check_ids=tuple(sorted(checks)),
        passed=passed,
        reason_codes=reasons,
        wall_time_seconds=measured_wall_time,
        peak_memory_bytes=int(peak_rss_kib * 1024),
        projected_evaluation_cpu_hours=(
            measured_cpu_time * Decimal(per_complete_unit_factor) / Decimal(3600)
        ),
        projected_evaluation_wall_hours=(projected_wall_hours),
        projected_evaluation_output_bytes=projected_output,
        evaluation_roster_access_count=0,
        task_output_measurements=task_output_measurements,
        measured_cpu_time_seconds=measured_cpu_time,
    )


def run_resource_canary(
    *,
    canary_config: ReceiverHistoryConfig,
    implementation_sha256: str,
    block_index: int = 0,
    scale_cells: tuple[int, ...] | None = None,
    disorder_family: ReceiverHistoryDisorderFamily,
    endpoint_mask: tuple[ReceiverHistoryEndpoint, ...],
) -> ReceiverHistoryCanaryReport:
    with threadpool_limits(limits=2):
        return _run_resource_canary(
            canary_config=canary_config,
            implementation_sha256=implementation_sha256,
            block_index=block_index,
            scale_cells=scale_cells,
            disorder_family=disorder_family,
            endpoint_mask=endpoint_mask,
        )


def merge_concurrent_resource_reports(
    reports: tuple[ReceiverHistoryCanaryReport, ...],
    *,
    maximum_parallel_tasks: int = 4,
) -> ReceiverHistoryCanaryReport:
    """Conservatively qualify six endpoint/family resource paths."""

    if maximum_parallel_tasks not in {4, 6}:
        raise ValueError("receiver-history resource admission concurrency differs")
    expected_ids = tuple(
        f"receiver-history.resource-canary.block-{index:02d}" for index in range(6)
    )
    if tuple(value.report_id for value in reports) != expected_ids:
        raise ValueError("receiver-history resource report roster differs")
    reduced_flags = tuple('reduced-action-menu-profile' in value.check_ids for value in reports)
    if len(set(reduced_flags)) != 1 or maximum_parallel_tasks != (6 if reduced_flags[0] else 4):
        raise ValueError("receiver-history resource report/profile concurrency differs")
    aggregate_peak_memory = sum(
        sorted((value.peak_memory_bytes for value in reports), reverse=True)[
            :maximum_parallel_tasks
        ]
    )
    aggregate_memory_safe = aggregate_peak_memory <= 16 * 1024**3
    passed = all(value.passed for value in reports) and aggregate_memory_safe
    reasons = tuple(
        sorted(
            {
                *(reason for value in reports for reason in value.reason_codes),
                *(('RECEIVER_HISTORY_AGGREGATE_MEMORY_ENVELOPE_EXCEEDED',) if not aggregate_memory_safe else ()),
            }
        )
    )
    if not passed and not reasons:
        reasons = ('RECEIVER_HISTORY_CONCURRENT_RESOURCE_QUALIFICATION_FAILED',)
    return ReceiverHistoryCanaryReport(
        report_id="receiver-history.resource-admission",
        check_ids=tuple(
            sorted(
                {
                    "six-endpoint-family-resource-canaries",
                    f"{maximum_parallel_tasks}-worker-resource-admission",
                    *(check for value in reports for check in value.check_ids),
                }
            )
        ),
        passed=passed,
        reason_codes=reasons,
        wall_time_seconds=max(value.wall_time_seconds for value in reports),
        peak_memory_bytes=aggregate_peak_memory,
        projected_evaluation_cpu_hours=max(
            value.projected_evaluation_cpu_hours for value in reports
        ),
        projected_evaluation_wall_hours=max(
            value.projected_evaluation_wall_hours for value in reports
        ),
        projected_evaluation_output_bytes=max(
            value.projected_evaluation_output_bytes for value in reports
        ),
        evaluation_roster_access_count=0,
    )


def merge_component_canaries(
    *, report_id: str, reports: tuple[ReceiverHistoryCanaryReport, ...]
) -> ReceiverHistoryCanaryReport:
    if not reports or len({value.report_id for value in reports}) != len(reports):
        raise ValueError("receiver-history component canary merge roster differs")
    passed = all(value.passed for value in reports)
    reasons = tuple(sorted({reason for value in reports for reason in value.reason_codes}))
    if not passed and not reasons:
        reasons = ('RECEIVER_HISTORY_COMPONENT_CANARY_MERGE_FAILED',)
    return ReceiverHistoryCanaryReport(
        report_id=report_id,
        check_ids=tuple(sorted({check for value in reports for check in value.check_ids})),
        passed=passed,
        reason_codes=reasons,
        wall_time_seconds=sum((value.wall_time_seconds for value in reports), Decimal(0)),
        peak_memory_bytes=max(value.peak_memory_bytes for value in reports),
        projected_evaluation_cpu_hours=Decimal(0),
        projected_evaluation_wall_hours=Decimal(0),
        projected_evaluation_output_bytes=0,
        evaluation_roster_access_count=0,
    )


def merge_canary_reports(
    *,
    source_check_ids: tuple[str, ...],
    numerical: ReceiverHistoryCanaryReport,
    resource_report: ReceiverHistoryCanaryReport,
) -> ReceiverHistoryCanaryReport:
    passed = numerical.passed and resource_report.passed
    reasons = tuple(sorted({*numerical.reason_codes, *resource_report.reason_codes}))
    if not passed and not reasons:
        reasons = ('RECEIVER_HISTORY_CANARY_QUALIFICATION_FAILED',)
    return ReceiverHistoryCanaryReport(
        report_id="receiver-history.source-canary-qualification",
        check_ids=tuple(
            sorted({*source_check_ids, *numerical.check_ids, *resource_report.check_ids})
        ),
        passed=passed,
        reason_codes=reasons,
        wall_time_seconds=numerical.wall_time_seconds + resource_report.wall_time_seconds,
        peak_memory_bytes=max(numerical.peak_memory_bytes, resource_report.peak_memory_bytes),
        projected_evaluation_cpu_hours=resource_report.projected_evaluation_cpu_hours,
        projected_evaluation_wall_hours=resource_report.projected_evaluation_wall_hours,
        projected_evaluation_output_bytes=resource_report.projected_evaluation_output_bytes,
        evaluation_roster_access_count=0,
        task_output_measurements=resource_report.task_output_measurements,
        measured_cpu_time_seconds=resource_report.measured_cpu_time_seconds,
    )


__all__ = [
    "audit_source_firewall",
    "merge_canary_reports",
    "merge_component_canaries",
    "merge_concurrent_resource_reports",
    "run_certificate_canary",
    "run_cross_implementation_conformance",
    "run_generator_canary",
    "run_factorized_action_canary",
    "run_lexical_direction_canary",
    "run_numerical_conformance",
    "run_observer_canary",
    "run_one_pass_history_canary",
    "run_rank_separation_canary",
    "run_resource_canary",
    "run_transition_canary",
    "run_untouched_sampler_canary",
]
