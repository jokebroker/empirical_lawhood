"""Outcome-blind source, equation, rank, negative-control, and resource canaries."""

from __future__ import annotations

from empirical_lawhood.adapters.history_budget_fixed_scientific_inputs import fixed_history_budget_preparation_input, fixed_history_budget_unit_input

from empirical_lawhood.adapters.history_budget_seed_constants import history_budget_fixed_seed

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

from empirical_lawhood.adapters.methods.split_cohort_history_budget.evaluator import _lexical_reference_decision_errors, adjudicate_unit_scale
from empirical_lawhood.adapters.methods.split_cohort_history_budget.conditioning import dimensionless_history_operator
from empirical_lawhood.adapters.methods.split_cohort_history_budget.discrete_rank import residual_certified_rank_bracket
from empirical_lawhood.adapters.methods.split_cohort_history_budget.history import assemble_dense_operator, coordinate_labels, observe_history
from empirical_lawhood.adapters.methods.split_cohort_history_budget.preparation_sampler import _raw_state_bits, charge_preservation_error, scale_coupled_initial_states
from empirical_lawhood.adapters.methods.split_cohort_history_budget.inference import _transition_disposition
from empirical_lawhood.adapters.methods.split_cohort_history_budget.structural_rank import maximum_bipartite_matching_rank
from empirical_lawhood.adapters.methods.split_cohort_history_budget.targeting import nominate_targeted_challenges
from empirical_lawhood.adapters.simulators.rc_ladder_split_cohort_history.generator import _affine_action, _factorized_action_components, assemble_sparse_operator, generate_outcomes, receiver_matrix
from empirical_lawhood.kernel.evidence import OutcomeAccess

from .contracts import SplitCohortHistoryBudgetDenominatorDescriptor, SplitCohortHistoryBudgetDisorderFamily, SplitCohortHistoryBudgetConfig, SplitCohortHistoryBudgetGeneratorActionOutcome, SplitCohortHistoryBudgetPhase, SplitCohortHistoryBudgetScientificState
from .authoring import SPLIT_COHORT_HISTORY_BUDGET_RESOURCE_PROFILE
from .descriptors import default_config, generate_descriptor
from .runtime_contracts import SplitCohortHistoryBudgetCanaryReport, SplitCohortHistoryBudgetDenominatorBundle, SplitCohortHistoryBudgetTaskOutputMeasurement
from .workflow import adjudication_bundle, generator_bundle, freeze_nominations, history_bundle, targeter_bundle, untouched_bundle


_GENERATOR_PATH = 'src/empirical_lawhood/adapters/simulators/rc_ladder_split_cohort_history/generator.py'
_OBSERVER_PATHS = (
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/conditioning.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/discrete_rank.py',
    "src/empirical_lawhood/adapters/methods/_arb.py",
    "src/empirical_lawhood/adapters/methods/certified_rank.py",
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/history.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/preparation_sampler.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/structural_rank.py',
    'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/targeting.py',
)
_EVALUATOR_PATH = 'src/empirical_lawhood/adapters/methods/split_cohort_history_budget/evaluator.py'
_SHARED_PREFIX = 'src/empirical_lawhood/adapters/split_cohort_history_budget/'


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("split cohort history budget conformance metric must be finite and nonnegative")
    return Decimal(str(float(value)))


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
        raise ValueError("split cohort history budget firewall audit lacks a claim-bearing source file")
    generator_imports = set(_imports(source_files[_GENERATOR_PATH]))
    observer_imports = {value for path in _OBSERVER_PATHS for value in _imports(source_files[path])}
    evaluator_imports = set(_imports(source_files[_EVALUATOR_PATH]))
    forbidden_generator = {
        value
        for value in generator_imports
        if value.startswith("empirical_lawhood.adapters.methods")
        or "rc_ladder_response" in value
        or "physical_scale_morphism" in value
        or "physical_scale_morphism.simulator_challenges" in value
        or "simulator_morphism_challenges" in value
    }
    forbidden_observer = {
        value
        for value in observer_imports
        if value.startswith("empirical_lawhood.adapters.simulators")
        or "physical_scale_morphism" in value
        or "physical_scale_morphism.simulator_challenges" in value
        or "simulator_morphism_challenges" in value
    }
    if forbidden_generator or forbidden_observer:
        raise ValueError("split cohort history budget generator--observer source firewall is violated")
    allowed_local_intersection = {
        'empirical_lawhood.adapters.split_cohort_history_budget.contracts',
        "empirical_lawhood.kernel.serialization",
    }
    local_intersection = {
        value for value in generator_imports & observer_imports if value.startswith("empirical_lawhood")
    }
    if local_intersection != allowed_local_intersection:
        raise ValueError("split cohort history budget implementations share an undeclared scientific module")
    if 'empirical_lawhood.adapters.simulators.rc_ladder_split_cohort_history.generator' in evaluator_imports:
        raise ValueError("split cohort history budget evaluator imports the generator implementation")
    shared_paths = tuple(sorted(path for path in source_files if path.startswith(_SHARED_PREFIX)))
    if not {
        f"{_SHARED_PREFIX}contracts.py",
        f"{_SHARED_PREFIX}runtime_contracts.py",
    }.issubset(shared_paths):
        raise ValueError("split cohort history budget firewall audit lacks the shared record surface")
    return (
        "generator-does-not-import-observer-or-physical-scale-methods",
        "observer-does-not-import-generator-or-physical-scale-methods",
        "scientific-intersection-is-contracts-only",
    )


def _report(report_id: str, checks: set[str], started: float) -> SplitCohortHistoryBudgetCanaryReport:
    return SplitCohortHistoryBudgetCanaryReport(
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


def run_structural_rank_canary() -> SplitCohortHistoryBudgetCanaryReport:
    started = perf_counter()
    checks: set[str] = set()
    for bitmask in range(1 << 12):
        support = np.asarray(
            [(bitmask >> index) & 1 for index in range(12)],
            dtype=np.bool_,
        ).reshape(3, 4)
        if maximum_bipartite_matching_rank(support) != _brute_matching_rank(support):
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_STRUCTURAL_RANK_CONFORMANCE_FAILED")
    checks.add("exhaustive-3x4-boolean-matching")
    return _report("split-cohort-history-budget.structural-rank-canary", checks, started)


def run_discrete_rank_canary() -> SplitCohortHistoryBudgetCanaryReport:
    started = perf_counter()
    fixtures = (
        (np.eye(4, dtype=np.float64), 4),
        (np.vstack((np.eye(4), np.eye(4))), 4),
        (np.asarray([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64), 1),
    )
    for matrix, expected in fixtures:
        lower, upper, _residual, _separation = residual_certified_rank_bracket(matrix)
        if (lower, upper) != (expected, expected):
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_DISCRETE_RANK_CONFORMANCE_FAILED")
    return _report(
        "split-cohort-history-budget.discrete-rank-canary",
        {
            "arb-gram-inertia-reconstruction",
            "duplicate-row-rank-control",
            "independent-exact-dyadic-rational-rank",
            "known-rank-high-precision-brackets",
        },
        started,
    )


def run_conditioning_canary() -> SplitCohortHistoryBudgetCanaryReport:
    started = perf_counter()
    config = default_config(SplitCohortHistoryBudgetPhase.CANARY)
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
        raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CONDITIONING_CONFORMANCE_FAILED")
    return _report(
        "split-cohort-history-budget.conditioning-canary",
        {"absolute-resolution-scaling", "capacitance-energy-normalization"},
        started,
    )


def run_untouched_sampler_canary() -> SplitCohortHistoryBudgetCanaryReport:
    started = perf_counter()
    descriptors = {scale: _uniform_descriptor(scale) for scale in (64, 128, 256)}
    states = scale_coupled_initial_states(
        seed=history_budget_fixed_seed(programme_ordinal=2, scientific_role="untouched-canary", numeric_index=0),
        unit_id="canary.untouched.block-00",
        scientific_input=fixed_history_budget_preparation_input(programme_ordinal=2, unit_id="canary.untouched.block-00"),
        descriptors=descriptors,
    )
    if charge_preservation_error(states, descriptors) > 1e-15:
        raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_UNTOUCHED_SCALE_MAP_CONFORMANCE_FAILED")
    shuffled = dict(states)
    shuffled[64] = states[64][::-1].copy()
    if charge_preservation_error(shuffled, descriptors) <= 1e-6:
        raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_UNTOUCHED_SCALE_SHUFFLE_NEGATIVE_FAILED")
    raw = np.asarray([[0.25, np.nan, np.inf, -np.inf]], dtype=np.float64)
    retained = _raw_state_bits(raw)
    if not np.array_equal(retained.view(np.float64), raw, equal_nan=True):
        raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_INVALID_RAW_BIT_RETENTION_FAILED")
    return _report(
        "split-cohort-history-budget.untouched-sampler-canary",
        {
            "capacitance-weighted-direct-charge-map",
            "invalid-ieee754-bit-retention",
            "scale-shuffled-negative",
        },
        started,
    )


def run_certificate_canary() -> SplitCohortHistoryBudgetCanaryReport:
    """Exercise the independent sparse objective family without observer input."""

    started = perf_counter()
    config = default_config(SplitCohortHistoryBudgetPhase.CANARY)
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
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_SPARSE_CERTIFICATE_UNKNOWN_COORDINATE")
        coordinate_id = matches[0]
        by_coordinate[coordinate_id] += 1
        if (
            not check.complete_family_certificate
            or check.primal_residual > config.optimization_tolerance
            or check.dual_residual > config.optimization_tolerance
        ):
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_SPARSE_CERTIFICATE_INCOMPLETE")
    if not by_coordinate or set(by_coordinate.values()) != {51}:
        raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_SPARSE_CERTIFICATE_OBJECTIVE_ROSTER_FAILED")
    # Deleting even one objective must make the exact 51-objective roster fail.
    if len(checks[:-1]) == len(coordinates) * 51:
        raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_SPARSE_CERTIFICATE_NEGATIVE_CONTROL_FAILED")
    return _report(
        "split-cohort-history-budget.certificate-canary",
        {
            "independent-sparse-primal-dual-family",
            "exact-51-objective-per-coordinate-roster",
            "missing-objective-negative",
        },
        started,
    )


def _action_fixture(
    config: SplitCohortHistoryBudgetConfig,
    *,
    plus_admit: bool,
    minus_robust_admit: bool,
    midpoint_admit: bool,
) -> SplitCohortHistoryBudgetGeneratorActionOutcome:
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
    return SplitCohortHistoryBudgetGeneratorActionOutcome(
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


def run_lexical_direction_canary() -> SplitCohortHistoryBudgetCanaryReport:
    started = perf_counter()
    config = default_config(SplitCohortHistoryBudgetPhase.CANARY)
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
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_LEXICAL_REFERENCE_DIRECTION_FAILED")
    return _report(
        "split-cohort-history-budget.lexical-direction-canary",
        {"plus-is-exact-reference", "minus-is-robust-member", "midpoint-label-invariant"},
        started,
    )


def run_rank_separation_canary() -> SplitCohortHistoryBudgetCanaryReport:
    started = perf_counter()
    reports = (
        run_structural_rank_canary(),
        run_discrete_rank_canary(),
        run_conditioning_canary(),
    )
    return _report(
        "split-cohort-history-budget.rank-separation-canary",
        {check for report in reports for check in report.check_ids},
        started,
    )


def run_transition_canary() -> SplitCohortHistoryBudgetCanaryReport:
    started = perf_counter()
    opposed = SplitCohortHistoryBudgetScientificState.OPPOSED
    limited = SplitCohortHistoryBudgetScientificState.TARGETABILITY_LIMITED
    informative = SplitCohortHistoryBudgetScientificState.INFORMATIVE_NONADVERSE
    if (
        _transition_disposition((opposed, limited, informative))
        is not SplitCohortHistoryBudgetScientificState.SUPPORTED
    ):
        raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_MONOTONE_TRANSITION_CONTROL_FAILED")
    if (
        _transition_disposition((opposed, SplitCohortHistoryBudgetScientificState.MIXED, informative))
        is not SplitCohortHistoryBudgetScientificState.MIXED
    ):
        raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_MIXED_TRANSITION_NEGATIVE_FAILED")
    if (
        _transition_disposition((informative, limited, opposed))
        is not SplitCohortHistoryBudgetScientificState.NONMONOTONE_PHASE_PATTERN
    ):
        raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_NONMONOTONE_TRANSITION_NEGATIVE_FAILED")
    return _report(
        "split-cohort-history-budget.transition-canary",
        {"monotone-three-state-control", "mixed-never-supported", "reversal-never-supported"},
        started,
    )


def run_factorized_action_canary() -> SplitCohortHistoryBudgetCanaryReport:
    started = perf_counter()
    config = default_config(SplitCohortHistoryBudgetPhase.CANARY)
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
                raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_FACTORIZED_ACTION_EQUIVALENCE_FAILED")
    return _report(
        "split-cohort-history-budget.factorized-action-canary",
        {"all-25-actions-factorized-versus-direct"},
        started,
    )


def run_one_pass_history_canary() -> SplitCohortHistoryBudgetCanaryReport:
    started = perf_counter()
    config = default_config(SplitCohortHistoryBudgetPhase.CANARY)
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
        raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_ONE_PASS_HISTORY_EQUIVALENCE_FAILED")
    return _report(
        "split-cohort-history-budget.one-pass-history-canary",
        {"32-time-krylov-versus-31-sequential-propagations"},
        started,
    )


def _uniform_descriptor(scale_cells: int) -> SplitCohortHistoryBudgetDenominatorDescriptor:
    seed = history_budget_fixed_seed(programme_ordinal=2, scientific_role="uniform-canary", numeric_index=scale_cells)
    base = generate_descriptor(
        unit_id=f"canary.truth-known.n{scale_cells}",
        family=SplitCohortHistoryBudgetDisorderFamily.SMOOTH_PERIODIC_12,
        scale_cells=scale_cells,
        seed=seed,
    )
    return replace(
        base,
        latent_field_sha256=sha256(b"uniform-zero-field").hexdigest(),
        capacitances_farads=(base.capacitance_bar_farads,) * scale_cells,
        interior_resistances_ohms=(base.resistance_bar_ohms,) * (scale_cells - 1),
    )


def _analytic_equilibrium(descriptor: SplitCohortHistoryBudgetDenominatorDescriptor, amplitude: float) -> np.ndarray:
    left = float(descriptor.left_source_resistance_ohms)
    right = float(descriptor.right_termination_resistance_ohms)
    interior = np.asarray(descriptor.interior_resistances_ohms, dtype=np.float64)
    current = amplitude / (left + float(np.sum(interior)) + right)
    preceding = np.concatenate((np.zeros(1), np.cumsum(interior)))
    return amplitude - current * (left + preceding)


def _rank_audit(descriptor: SplitCohortHistoryBudgetDenominatorDescriptor) -> None:
    config = default_config(SplitCohortHistoryBudgetPhase.CANARY)
    operator = assemble_dense_operator(descriptor)
    lag = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    inverse = expm(-operator.state_matrix * lag)
    blocks = []
    current = operator.receiver.copy()
    for depth in range(config.history_max_depth + 1):
        if depth:
            current = current @ inverse
        blocks.append(current.copy())
        stack = np.vstack(blocks)
        numpy_values = np.linalg.svd(stack, compute_uv=False)
        scipy_values = svdvals(stack)
        if not np.allclose(numpy_values, scipy_values, atol=1e-10, rtol=1e-11):
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CANARY_SVD_AUDIT_FAILED")
        duplicated = np.vstack([operator.receiver] * (depth + 1))
        if np.linalg.matrix_rank(duplicated, tol=1e-12) > 8:
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_DUPLICATE_PRESENT_CONTROL_FAILED")
        shuffled = np.vstack(tuple(reversed(blocks)))
        if not np.allclose(
            np.linalg.svd(shuffled, compute_uv=False),
            numpy_values,
            atol=1e-10,
            rtol=1e-11,
        ):
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_SHUFFLED_LAG_CONTROL_FAILED")


def _run_generator_canary() -> SplitCohortHistoryBudgetCanaryReport:
    """Qualify the sparse generator without importing observer implementation code."""

    start = perf_counter()
    config = default_config(SplitCohortHistoryBudgetPhase.CANARY)
    checks: set[str] = set()
    for scale_cells in config.scale_cells:
        descriptor = _uniform_descriptor(scale_cells)
        operator = assemble_sparse_operator(descriptor)
        equilibrium = np.asarray(
            spsolve(-operator.state_matrix, operator.left_input), dtype=np.float64
        )
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_GENERATOR_MANUFACTURED_SOLUTION_FAILED")
        if np.max(np.abs(operator.state_matrix @ equilibrium + operator.left_input)) > 1e-12:
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_GENERATOR_CURRENT_BALANCE_RESIDUAL_FAILED")
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
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_GENERATOR_PROPAGATOR_SEMIGROUP_FAILED")
        checks.update(
            {
                f"n{scale_cells}-generator-current-balance",
                f"n{scale_cells}-generator-manufactured-equilibrium",
            }
        )
    return SplitCohortHistoryBudgetCanaryReport(
        report_id="split-cohort-history-budget.generator-canary",
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


def run_generator_canary() -> SplitCohortHistoryBudgetCanaryReport:
    with threadpool_limits(limits=2):
        return _run_generator_canary()


def _run_observer_canary() -> SplitCohortHistoryBudgetCanaryReport:
    """Qualify dense observer equations, R8 and history controls in isolation."""

    start = perf_counter()
    config = default_config(SplitCohortHistoryBudgetPhase.CANARY)
    checks: set[str] = set()
    for scale_cells in config.scale_cells:
        descriptor = _uniform_descriptor(scale_cells)
        operator = assemble_dense_operator(descriptor)
        equilibrium = -np.linalg.solve(operator.state_matrix, operator.left_input)
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_OBSERVER_MANUFACTURED_SOLUTION_FAILED")
        if np.max(np.abs(operator.state_matrix @ equilibrium + operator.left_input)) > 1e-12:
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_OBSERVER_CURRENT_BALANCE_RESIDUAL_FAILED")
        _rank_audit(descriptor)
        checks.update(
            {
                f"n{scale_cells}-observer-manufactured-equilibrium",
                f"n{scale_cells}-observer-rank-and-history-controls",
            }
        )
    return SplitCohortHistoryBudgetCanaryReport(
        report_id="split-cohort-history-budget.observer-canary",
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


def run_observer_canary() -> SplitCohortHistoryBudgetCanaryReport:
    with threadpool_limits(limits=2):
        return _run_observer_canary()


def _run_cross_implementation_conformance(
    *,
    implementation_sha256: str,
    generator_report: SplitCohortHistoryBudgetCanaryReport,
    observer_report: SplitCohortHistoryBudgetCanaryReport,
    independence_check_ids: tuple[str, ...],
) -> SplitCohortHistoryBudgetCanaryReport:
    """Compare separately qualified implementations on the shared finite cases."""

    if (
        not generator_report.passed
        or not observer_report.passed
        or generator_report.report_id != "split-cohort-history-budget.generator-canary"
        or observer_report.report_id != "split-cohort-history-budget.observer-canary"
        or not independence_check_ids
    ):
        raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CANARY_PREREQUISITE_FAILED")

    start = perf_counter()
    config = default_config(SplitCohortHistoryBudgetPhase.CANARY)
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
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CANARY_OPERATOR_MISMATCH")
        if not np.allclose(dense.left_input, sparse.left_input, atol=1e-14, rtol=1e-14):
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CANARY_INPUT_MISMATCH")
        if not np.allclose(dense.receiver, receiver_matrix(descriptor), atol=1e-14, rtol=1e-14):
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CANARY_RECEIVER_MISMATCH")
        equilibrium = -np.linalg.solve(dense.state_matrix, dense.left_input)
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CANARY_MANUFACTURED_SOLUTION_FAILED")
        if np.max(np.abs(dense.state_matrix @ equilibrium + dense.left_input)) > 1e-12:
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CANARY_CURRENT_BALANCE_RESIDUAL_FAILED")
        initial = np.column_stack((np.linspace(0.1, 0.9, scale_cells), analytic))
        duration = 0.013 * float(descriptor.time_scale_seconds)
        dense_future = expm(dense.state_matrix * duration) @ initial
        sparse_future = np.asarray(expm_multiply(sparse.state_matrix * duration, initial))
        if not np.allclose(dense_future, sparse_future, atol=1e-11, rtol=1e-11):
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CANARY_PROPAGATOR_MISMATCH")
        _rank_audit(descriptor)

        seed = history_budget_fixed_seed(programme_ordinal=2, scientific_role="conformance", numeric_index=scale_cells)
        challenge_descriptor = generate_descriptor(
            unit_id=f"canary.truth-known.n{scale_cells}",
            family=SplitCohortHistoryBudgetDisorderFamily.CORRELATED_FIELD_12,
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
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CANARY_NO_NOMINATIONS")
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
            raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CANARY_CROSS_IMPLEMENTATION_FAILED")
        for factors in (
            {"left_resistance_factor": 1.25},
            {"right_resistance_factor": 1.25},
            {"action_clock_factor": 0.9},
        ):
            wrong = generate_outcomes(
                config=config,
                descriptor=challenge_descriptor,
                nominations=observer.geometries,
                implementation_sha256=implementation_sha256,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                **factors,
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
                raise ValueError("SPLIT_COHORT_HISTORY_BUDGET_CANARY_NEGATIVE_CONTROL_FAILED")
        checks.update(
            {
                f"n{scale_cells}-dense-sparse-equations",
                f"n{scale_cells}-manufactured-equilibrium",
                f"n{scale_cells}-rank-svd-and-history-controls",
                f"n{scale_cells}-source-termination-clock-negatives",
            }
        )
    return SplitCohortHistoryBudgetCanaryReport(
        report_id="split-cohort-history-budget.numerical-conformance",
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
    generator_report: SplitCohortHistoryBudgetCanaryReport,
    observer_report: SplitCohortHistoryBudgetCanaryReport,
    independence_check_ids: tuple[str, ...],
) -> SplitCohortHistoryBudgetCanaryReport:
    with threadpool_limits(limits=2):
        return _run_cross_implementation_conformance(
            implementation_sha256=implementation_sha256,
            generator_report=generator_report,
            observer_report=observer_report,
            independence_check_ids=independence_check_ids,
        )


def run_numerical_conformance(*, implementation_sha256: str) -> SplitCohortHistoryBudgetCanaryReport:
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
    implementation_sha256: str,
    block_index: int = 0,
    scale_cells: tuple[int, ...] | None = None,
) -> SplitCohortHistoryBudgetCanaryReport:
    """Measure one excluded complete seed block without evaluation-roster access."""

    config = default_config(SplitCohortHistoryBudgetPhase.CANARY)
    scales = config.excluded_resource_canary_scale_cells if scale_cells is None else scale_cells
    if not scales or not set(scales).issubset(config.excluded_resource_canary_scale_cells):
        raise ValueError("resource canary scale roster differs from the excluded config")
    if not 0 <= block_index < 4:
        raise ValueError("resource canary block index lies outside the excluded roster")
    block_id = f"block-{block_index:02d}"
    seed = history_budget_fixed_seed(programme_ordinal=2, scientific_role="excluded-resource-canary", numeric_index=block_index)
    wall_start = perf_counter()
    cpu_start = process_time()
    output_bytes = 0
    checks: set[str] = set()
    descriptors = tuple(
        generate_descriptor(
            unit_id=f"canary.resource.{block_id}",
            family=SplitCohortHistoryBudgetDisorderFamily.MULTISCALE_WAVELET_12,
            scale_cells=scale,
            seed=seed,
        )
        for scale in scales
    )
    scientific_input = fixed_history_budget_unit_input(programme_ordinal=2, unit_id=f"canary.resource.{block_id}", scale_cells=scales)
    if scientific_input.preparation_input is None:
        raise ValueError("resource canary requires its complete original preparation input")
    preparation_seed = bytes.fromhex(scientific_input.preparation_input.preparation_substream_seed_hex)
    denominators = SplitCohortHistoryBudgetDenominatorBundle(
        bundle_id=f"denominator-bundle.canary.resource.{block_id}",
        unit_id=f"canary.resource.{block_id}",
        preparation_substream_seed_hex=preparation_seed.hex(),
        preparation_substream_seed_sha256=sha256(preparation_seed).hexdigest(),
        descriptors=descriptors,
        scientific_input=scientific_input,
    )
    history = history_bundle(config=config, denominators=denominators)
    targeter = targeter_bundle(
        config=config,
        denominators=denominators,
        history=history.bundle,
        implementation_sha256=implementation_sha256,
    )
    untouched = untouched_bundle(config=config, denominators=denominators)
    nomination_freeze = freeze_nominations(
        history=history.bundle,
        observer=targeter.bundle,
        untouched=untouched.bundle,
        method_freeze=None,
    )
    generated = generator_bundle(
        config=config,
        denominators=denominators,
        nomination_freeze=nomination_freeze,
        untouched=untouched.bundle,
        untouched_float_payload=untouched.float_arrays.payload,
        untouched_float_manifest=untouched.float_arrays.manifest,
        untouched_int_payload=untouched.int_arrays.payload,
        untouched_int_manifest=untouched.int_arrays.manifest,
        implementation_sha256=implementation_sha256,
        maximum_array_bytes=512 * 1024**2,
    )
    adjudication = adjudication_bundle(
        config=config,
        denominators=denominators,
        history=history.bundle,
        observer=targeter.bundle,
        untouched=untouched.bundle,
        untouched_int_payload=untouched.int_arrays.payload,
        untouched_int_manifest=untouched.int_arrays.manifest,
        generator=generated.bundle,
        generator_arrays_payload=generated.packed_arrays.payload,
        generator_arrays_manifest=generated.packed_arrays.manifest,
        method_freeze=None,
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
        "untouched": sum(
            (
                len(untouched.bundle.canonical_bytes()),
                len(untouched.float_arrays.payload),
                len(untouched.float_arrays.manifest.canonical_bytes()),
                len(untouched.int_arrays.payload),
                len(untouched.int_arrays.manifest.canonical_bytes()),
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
        "descriptor": SPLIT_COHORT_HISTORY_BUDGET_RESOURCE_PROFILE.small.output_bytes,
        "history": SPLIT_COHORT_HISTORY_BUDGET_RESOURCE_PROFILE.observer.output_bytes,
        "targeter": SPLIT_COHORT_HISTORY_BUDGET_RESOURCE_PROFILE.observer.output_bytes,
        "untouched": SPLIT_COHORT_HISTORY_BUDGET_RESOURCE_PROFILE.generator.output_bytes,
        "generator": SPLIT_COHORT_HISTORY_BUDGET_RESOURCE_PROFILE.generator.output_bytes,
        "adjudication": SPLIT_COHORT_HISTORY_BUDGET_RESOURCE_PROFILE.small.output_bytes,
    }
    per_task_output_passed = all(
        task_output_bytes[key] <= task_output_limits[key] for key in task_output_bytes
    )
    task_output_measurements = tuple(
        SplitCohortHistoryBudgetTaskOutputMeasurement(
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
            "n256-m512-all-pairs-and-all-resolution-views",
            "all-three-scale-pushforwards",
            "complete-target-dynamic-sink-certificates",
            "complete-25-action-untouched-generation",
            f"resource-profile-{SPLIT_COHORT_HISTORY_BUDGET_RESOURCE_PROFILE.profile_id}",
            "per-task-output-ceilings-audited",
            "cpu-projection-bounded-by-issued-two-core-envelope",
        }
    )
    elapsed_wall = perf_counter() - wall_start
    raw_elapsed_cpu = process_time() - cpu_start
    # C8--C11 execute concurrently in one process. Process CPU deltas include
    # sibling tasks and cannot be attributed four times. Cap each report by
    # its issued two-core envelope; retain the smaller isolated measurement.
    elapsed_cpu = min(raw_elapsed_cpu, elapsed_wall * 2.0)
    measured_wall_time = _decimal(elapsed_wall)
    measured_cpu_time = _decimal(elapsed_cpu)
    per_complete_unit_factor = 90
    projected_output = output_bytes * per_complete_unit_factor
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resource_path_passed = (
        elapsed_cpu * per_complete_unit_factor <= 90 * 3600
        and elapsed_wall * per_complete_unit_factor / 4 <= 12 * 3600
        and projected_output <= 4 * 1024**3
        and per_task_output_passed
        and peak_rss_kib * 1024 <= 12 * 1024**3
    )
    passed = resource_path_passed and not disagreement_ids
    reasons = tuple(
        sorted(
            {
                *(("SPLIT_COHORT_HISTORY_BUDGET_RESOURCE_PATH_INADEQUATE",) if not resource_path_passed else ()),
                *(("SPLIT_COHORT_HISTORY_BUDGET_RESOURCE_CANARY_IMPLEMENTATION_DISAGREEMENT",) if disagreement_ids else ()),
            }
        )
    )
    return SplitCohortHistoryBudgetCanaryReport(
        report_id=f"split-cohort-history-budget.excluded-resource-canary.{block_id}",
        check_ids=tuple(sorted(checks)),
        passed=passed,
        reason_codes=reasons,
        wall_time_seconds=measured_wall_time,
        peak_memory_bytes=int(peak_rss_kib * 1024),
        projected_evaluation_cpu_hours=(
            measured_cpu_time * Decimal(per_complete_unit_factor) / Decimal(3600)
        ),
        projected_evaluation_wall_hours=(
            measured_wall_time * Decimal(per_complete_unit_factor) / Decimal(4 * 3600)
        ),
        projected_evaluation_output_bytes=projected_output,
        evaluation_roster_access_count=0,
        task_output_measurements=task_output_measurements,
        measured_cpu_time_seconds=measured_cpu_time,
    )


def run_resource_canary(
    *,
    implementation_sha256: str,
    block_index: int = 0,
    scale_cells: tuple[int, ...] | None = None,
) -> SplitCohortHistoryBudgetCanaryReport:
    with threadpool_limits(limits=2):
        return _run_resource_canary(
            implementation_sha256=implementation_sha256,
            block_index=block_index,
            scale_cells=scale_cells,
        )


def merge_concurrent_resource_reports(
    reports: tuple[SplitCohortHistoryBudgetCanaryReport, ...],
) -> SplitCohortHistoryBudgetCanaryReport:
    """Conservatively qualify the four service-scheduled excluded unit paths."""

    expected_ids = tuple(
        f"split-cohort-history-budget.excluded-resource-canary.block-{index:02d}" for index in range(4)
    )
    if tuple(value.report_id for value in reports) != expected_ids:
        raise ValueError("Concurrent resource report roster differs")
    passed = all(value.passed for value in reports)
    reasons = tuple(sorted({reason for value in reports for reason in value.reason_codes}))
    if not passed and not reasons:
        reasons = ("SPLIT_COHORT_HISTORY_BUDGET_CONCURRENT_RESOURCE_QUALIFICATION_FAILED",)
    return SplitCohortHistoryBudgetCanaryReport(
        report_id="split-cohort-history-budget.concurrent-resource-qualification",
        check_ids=tuple(
            sorted(
                {
                    "four-excluded-units-service-scheduled",
                    "four-worker-resource-admission",
                    *(check for value in reports for check in value.check_ids),
                }
            )
        ),
        passed=passed,
        reason_codes=reasons,
        wall_time_seconds=max(value.wall_time_seconds for value in reports),
        peak_memory_bytes=sum(value.peak_memory_bytes for value in reports),
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
    *, report_id: str, reports: tuple[SplitCohortHistoryBudgetCanaryReport, ...]
) -> SplitCohortHistoryBudgetCanaryReport:
    if not reports or len({value.report_id for value in reports}) != len(reports):
        raise ValueError("Component canary merge roster differs")
    passed = all(value.passed for value in reports)
    reasons = tuple(sorted({reason for value in reports for reason in value.reason_codes}))
    if not passed and not reasons:
        reasons = ("SPLIT_COHORT_HISTORY_BUDGET_COMPONENT_CANARY_MERGE_FAILED",)
    return SplitCohortHistoryBudgetCanaryReport(
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
    numerical: SplitCohortHistoryBudgetCanaryReport,
    resource_report: SplitCohortHistoryBudgetCanaryReport,
) -> SplitCohortHistoryBudgetCanaryReport:
    passed = numerical.passed and resource_report.passed
    reasons = tuple(sorted({*numerical.reason_codes, *resource_report.reason_codes}))
    if not passed and not reasons:
        reasons = ("SPLIT_COHORT_HISTORY_BUDGET_CANARY_QUALIFICATION_FAILED",)
    return SplitCohortHistoryBudgetCanaryReport(
        report_id="split-cohort-history-budget.source-canary-qualification",
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
