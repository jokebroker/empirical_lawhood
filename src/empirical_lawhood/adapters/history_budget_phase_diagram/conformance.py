"""Outcome-blind source, equation, rank, negative-control, and resource canaries."""

from __future__ import annotations

from empirical_lawhood.adapters.history_budget_fixed_scientific_inputs import fixed_history_budget_descriptor_input, fixed_history_budget_preparation_input, fixed_history_budget_unit_input

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

from empirical_lawhood.adapters.methods.history_budget_phase_diagram.evaluator import adjudicate_unit_scale
from empirical_lawhood.adapters.methods.history_budget_phase_diagram.conditioning import dimensionless_history_operator
from empirical_lawhood.adapters.methods.history_budget_phase_diagram.discrete_rank import residual_certified_rank_bracket
from empirical_lawhood.adapters.methods.history_budget_phase_diagram.history import assemble_dense_operator, observe_and_nominate, coordinate_labels
from empirical_lawhood.adapters.methods.history_budget_phase_diagram.preparation_sampler import charge_preservation_error, scale_coupled_initial_states
from empirical_lawhood.adapters.methods.history_budget_phase_diagram.structural_rank import maximum_bipartite_matching_rank
from empirical_lawhood.adapters.simulators.rc_ladder_history_budget.generator import assemble_sparse_operator, generate_outcomes, receiver_matrix
from empirical_lawhood.kernel.evidence import OutcomeAccess

from .contracts import HistoryBudgetPhaseDiagramDenominatorDescriptor, HistoryBudgetPhaseDiagramDisorderFamily, HistoryBudgetPhaseDiagramPhase
from .authoring import HISTORY_BUDGET_PHASE_DIAGRAM_RESOURCE_PROFILE
from .descriptors import default_config, generate_descriptor
from .runtime_contracts import HistoryBudgetPhaseDiagramCanaryReport, HistoryBudgetPhaseDiagramDenominatorBundle, HistoryBudgetPhaseDiagramTaskOutputMeasurement
from .workflow import adjudication_bundle, generator_bundle, history_bundle, targeter_bundle, untouched_bundle


_GENERATOR_PATH = 'src/empirical_lawhood/adapters/simulators/rc_ladder_history_budget/generator.py'
_OBSERVER_PATHS = (
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/conditioning.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/discrete_rank.py',
    "src/empirical_lawhood/adapters/methods/_arb.py",
    "src/empirical_lawhood/adapters/methods/certified_rank.py",
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/history.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/preparation_sampler.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/structural_rank.py',
    'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/targeting.py',
)
_EVALUATOR_PATH = 'src/empirical_lawhood/adapters/methods/history_budget_phase_diagram/evaluator.py'
_SHARED_PREFIX = 'src/empirical_lawhood/adapters/history_budget_phase_diagram/'


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("history budget phase diagram conformance metric must be finite and nonnegative")
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
        raise ValueError("history budget phase diagram firewall audit lacks a claim-bearing source file")
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
        raise ValueError("history budget phase diagram generator--observer source firewall is violated")
    allowed_local_intersection = {
        'empirical_lawhood.adapters.history_budget_phase_diagram.contracts',
        "empirical_lawhood.kernel.serialization",
    }
    local_intersection = {
        value for value in generator_imports & observer_imports if value.startswith("empirical_lawhood")
    }
    if local_intersection != allowed_local_intersection:
        raise ValueError("history budget phase diagram implementations share an undeclared scientific module")
    if 'empirical_lawhood.adapters.simulators.rc_ladder_history_budget.generator' in evaluator_imports:
        raise ValueError("history budget phase diagram evaluator imports the generator implementation")
    shared_paths = tuple(sorted(path for path in source_files if path.startswith(_SHARED_PREFIX)))
    if not {
        f"{_SHARED_PREFIX}contracts.py",
        f"{_SHARED_PREFIX}runtime_contracts.py",
    }.issubset(shared_paths):
        raise ValueError("history budget phase diagram firewall audit lacks the shared record surface")
    return (
        "generator-does-not-import-observer-or-physical-scale-methods",
        "observer-does-not-import-generator-or-physical-scale-methods",
        "scientific-intersection-is-contracts-only",
    )


def _report(report_id: str, checks: set[str], started: float) -> HistoryBudgetPhaseDiagramCanaryReport:
    return HistoryBudgetPhaseDiagramCanaryReport(
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


def run_structural_rank_canary() -> HistoryBudgetPhaseDiagramCanaryReport:
    started = perf_counter()
    checks: set[str] = set()
    for bitmask in range(1 << 12):
        support = np.asarray(
            [(bitmask >> index) & 1 for index in range(12)],
            dtype=np.bool_,
        ).reshape(3, 4)
        if maximum_bipartite_matching_rank(support) != _brute_matching_rank(support):
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_STRUCTURAL_RANK_CONFORMANCE_FAILED")
    checks.add("exhaustive-3x4-boolean-matching")
    return _report("history-budget-phase-diagram.structural-rank-canary", checks, started)


def run_discrete_rank_canary() -> HistoryBudgetPhaseDiagramCanaryReport:
    started = perf_counter()
    fixtures = (
        (np.eye(4, dtype=np.float64), 4),
        (np.vstack((np.eye(4), np.eye(4))), 4),
        (np.asarray([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64), 1),
    )
    for matrix, expected in fixtures:
        lower, upper, _residual, _separation = residual_certified_rank_bracket(matrix)
        if (lower, upper) != (expected, expected):
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_DISCRETE_RANK_CONFORMANCE_FAILED")
    return _report(
        "history-budget-phase-diagram.discrete-rank-canary",
        {
            "arb-svd-inertia-reconstruction",
            "duplicate-row-rank-control",
            "exact-dyadic-rank-upper-bound",
            "known-rank-high-precision-brackets",
        },
        started,
    )


def run_conditioning_canary() -> HistoryBudgetPhaseDiagramCanaryReport:
    started = perf_counter()
    config = default_config(HistoryBudgetPhaseDiagramPhase.CANARY)
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
        raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CONDITIONING_CONFORMANCE_FAILED")
    return _report(
        "history-budget-phase-diagram.conditioning-canary",
        {"absolute-resolution-scaling", "capacitance-energy-normalization"},
        started,
    )


def run_untouched_sampler_canary() -> HistoryBudgetPhaseDiagramCanaryReport:
    started = perf_counter()
    states = scale_coupled_initial_states(
        seed=history_budget_fixed_seed(programme_ordinal=1, scientific_role="untouched-canary", numeric_index=0),
        unit_id="canary.untouched.block-00",
        scientific_input=fixed_history_budget_preparation_input(programme_ordinal=1, unit_id="canary.untouched.block-00"),
    )
    if charge_preservation_error(states) > 1e-15:
        raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_UNTOUCHED_SCALE_MAP_CONFORMANCE_FAILED")
    shuffled = dict(states)
    shuffled[64] = states[64][::-1].copy()
    if charge_preservation_error(shuffled) <= 1e-6:
        raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_UNTOUCHED_SCALE_SHUFFLE_NEGATIVE_FAILED")
    return _report(
        "history-budget-phase-diagram.untouched-sampler-canary",
        {"charge-preserving-dyadic-map", "scale-shuffled-negative"},
        started,
    )


def _uniform_descriptor(scale_cells: int) -> HistoryBudgetPhaseDiagramDenominatorDescriptor:
    seed = history_budget_fixed_seed(programme_ordinal=1, scientific_role="uniform-canary", numeric_index=scale_cells)
    base = generate_descriptor(
        unit_id=f"canary.truth-known.n{scale_cells}",
        family=HistoryBudgetPhaseDiagramDisorderFamily.SMOOTH_PERIODIC_12,
        scale_cells=scale_cells,
        seed=seed,
    )
    return replace(
        base,
        latent_field_sha256=sha256(b"uniform-zero-field").hexdigest(),
        capacitances_farads=(base.capacitance_bar_farads,) * scale_cells,
        interior_resistances_ohms=(base.resistance_bar_ohms,) * (scale_cells - 1),
    )


def _analytic_equilibrium(descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor, amplitude: float) -> np.ndarray:
    left = float(descriptor.left_source_resistance_ohms)
    right = float(descriptor.right_termination_resistance_ohms)
    interior = np.asarray(descriptor.interior_resistances_ohms, dtype=np.float64)
    current = amplitude / (left + float(np.sum(interior)) + right)
    preceding = np.concatenate((np.zeros(1), np.cumsum(interior)))
    return amplitude - current * (left + preceding)


def _rank_audit(descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor) -> None:
    config = default_config(HistoryBudgetPhaseDiagramPhase.CANARY)
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
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_SVD_AUDIT_FAILED")
        duplicated = np.vstack([operator.receiver] * (depth + 1))
        if np.linalg.matrix_rank(duplicated, tol=1e-12) > 8:
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_DUPLICATE_PRESENT_CONTROL_FAILED")
        shuffled = np.vstack(tuple(reversed(blocks)))
        if not np.allclose(
            np.linalg.svd(shuffled, compute_uv=False),
            numpy_values,
            atol=1e-10,
            rtol=1e-11,
        ):
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_SHUFFLED_LAG_CONTROL_FAILED")


def _run_generator_canary() -> HistoryBudgetPhaseDiagramCanaryReport:
    """Qualify the sparse generator without importing observer implementation code."""

    start = perf_counter()
    config = default_config(HistoryBudgetPhaseDiagramPhase.CANARY)
    checks: set[str] = set()
    for scale_cells in config.scale_cells:
        descriptor = _uniform_descriptor(scale_cells)
        operator = assemble_sparse_operator(descriptor)
        equilibrium = np.asarray(
            spsolve(-operator.state_matrix, operator.left_input), dtype=np.float64
        )
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_GENERATOR_MANUFACTURED_SOLUTION_FAILED")
        if np.max(np.abs(operator.state_matrix @ equilibrium + operator.left_input)) > 1e-12:
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_GENERATOR_CURRENT_BALANCE_RESIDUAL_FAILED")
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
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_GENERATOR_PROPAGATOR_SEMIGROUP_FAILED")
        checks.update(
            {
                f"n{scale_cells}-generator-current-balance",
                f"n{scale_cells}-generator-manufactured-equilibrium",
            }
        )
    return HistoryBudgetPhaseDiagramCanaryReport(
        report_id="history-budget-phase-diagram.generator-canary",
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


def run_generator_canary() -> HistoryBudgetPhaseDiagramCanaryReport:
    with threadpool_limits(limits=2):
        return _run_generator_canary()


def _run_observer_canary() -> HistoryBudgetPhaseDiagramCanaryReport:
    """Qualify dense observer equations, R8 and history controls in isolation."""

    start = perf_counter()
    config = default_config(HistoryBudgetPhaseDiagramPhase.CANARY)
    checks: set[str] = set()
    for scale_cells in config.scale_cells:
        descriptor = _uniform_descriptor(scale_cells)
        operator = assemble_dense_operator(descriptor)
        equilibrium = -np.linalg.solve(operator.state_matrix, operator.left_input)
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_OBSERVER_MANUFACTURED_SOLUTION_FAILED")
        if np.max(np.abs(operator.state_matrix @ equilibrium + operator.left_input)) > 1e-12:
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_OBSERVER_CURRENT_BALANCE_RESIDUAL_FAILED")
        _rank_audit(descriptor)
        checks.update(
            {
                f"n{scale_cells}-observer-manufactured-equilibrium",
                f"n{scale_cells}-observer-rank-and-history-controls",
            }
        )
    return HistoryBudgetPhaseDiagramCanaryReport(
        report_id="history-budget-phase-diagram.observer-canary",
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


def run_observer_canary() -> HistoryBudgetPhaseDiagramCanaryReport:
    with threadpool_limits(limits=2):
        return _run_observer_canary()


def _run_cross_implementation_conformance(
    *,
    implementation_sha256: str,
    generator_report: HistoryBudgetPhaseDiagramCanaryReport,
    observer_report: HistoryBudgetPhaseDiagramCanaryReport,
    independence_check_ids: tuple[str, ...],
) -> HistoryBudgetPhaseDiagramCanaryReport:
    """Compare separately qualified implementations on the shared finite cases."""

    if (
        not generator_report.passed
        or not observer_report.passed
        or generator_report.report_id != "history-budget-phase-diagram.generator-canary"
        or observer_report.report_id != "history-budget-phase-diagram.observer-canary"
        or not independence_check_ids
    ):
        raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_PREREQUISITE_FAILED")

    start = perf_counter()
    config = default_config(HistoryBudgetPhaseDiagramPhase.CANARY)
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
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_OPERATOR_MISMATCH")
        if not np.allclose(dense.left_input, sparse.left_input, atol=1e-14, rtol=1e-14):
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_INPUT_MISMATCH")
        if not np.allclose(dense.receiver, receiver_matrix(descriptor), atol=1e-14, rtol=1e-14):
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_RECEIVER_MISMATCH")
        equilibrium = -np.linalg.solve(dense.state_matrix, dense.left_input)
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_MANUFACTURED_SOLUTION_FAILED")
        if np.max(np.abs(dense.state_matrix @ equilibrium + dense.left_input)) > 1e-12:
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_CURRENT_BALANCE_RESIDUAL_FAILED")
        initial = np.column_stack((np.linspace(0.1, 0.9, scale_cells), analytic))
        duration = 0.013 * float(descriptor.time_scale_seconds)
        dense_future = expm(dense.state_matrix * duration) @ initial
        sparse_future = np.asarray(expm_multiply(sparse.state_matrix * duration, initial))
        if not np.allclose(dense_future, sparse_future, atol=1e-11, rtol=1e-11):
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_PROPAGATOR_MISMATCH")
        _rank_audit(descriptor)

        seed = history_budget_fixed_seed(programme_ordinal=1, scientific_role="conformance", numeric_index=scale_cells)
        challenge_descriptor = generate_descriptor(
            unit_id=f"canary.truth-known.n{scale_cells}",
            family=HistoryBudgetPhaseDiagramDisorderFamily.CORRELATED_FIELD_12,
            scale_cells=scale_cells,
            seed=seed,
        )
        observer = observe_and_nominate(
            config=config,
            descriptor=challenge_descriptor,
            implementation_sha256=implementation_sha256,
            scientific_input=fixed_history_budget_descriptor_input(programme_ordinal=1, current_descriptor_sha256=challenge_descriptor.fingerprint()),
        )
        if not observer.nominations:
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_NO_NOMINATIONS")
        generator = generate_outcomes(
            config=config,
            descriptor=challenge_descriptor,
            nominations=observer.nominations,
            implementation_sha256=implementation_sha256,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
        adjudication = adjudicate_unit_scale(
            config=config,
            descriptor=challenge_descriptor,
            forecast=observer.forecast,
            nominations=observer.nominations,
            generator_outcome=generator.outcome,
            method_freeze=None,
        )
        if not adjudication.generator_observer_agreement:
            raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_CROSS_IMPLEMENTATION_FAILED")
        for factors in (
            {"left_resistance_factor": 1.25},
            {"right_resistance_factor": 1.25},
            {"action_clock_factor": 0.9},
        ):
            wrong = generate_outcomes(
                config=config,
                descriptor=challenge_descriptor,
                nominations=observer.nominations,
                implementation_sha256=implementation_sha256,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                **factors,
            )
            wrong_adjudication = adjudicate_unit_scale(
                config=config,
                descriptor=challenge_descriptor,
                forecast=observer.forecast,
                nominations=observer.nominations,
                generator_outcome=wrong.outcome,
                method_freeze=None,
            )
            if wrong_adjudication.generator_observer_agreement:
                raise ValueError("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_NEGATIVE_CONTROL_FAILED")
        checks.update(
            {
                f"n{scale_cells}-dense-sparse-equations",
                f"n{scale_cells}-manufactured-equilibrium",
                f"n{scale_cells}-rank-svd-and-history-controls",
                f"n{scale_cells}-source-termination-clock-negatives",
            }
        )
    return HistoryBudgetPhaseDiagramCanaryReport(
        report_id="history-budget-phase-diagram.numerical-conformance",
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
    generator_report: HistoryBudgetPhaseDiagramCanaryReport,
    observer_report: HistoryBudgetPhaseDiagramCanaryReport,
    independence_check_ids: tuple[str, ...],
) -> HistoryBudgetPhaseDiagramCanaryReport:
    with threadpool_limits(limits=2):
        return _run_cross_implementation_conformance(
            implementation_sha256=implementation_sha256,
            generator_report=generator_report,
            observer_report=observer_report,
            independence_check_ids=independence_check_ids,
        )


def run_numerical_conformance(*, implementation_sha256: str) -> HistoryBudgetPhaseDiagramCanaryReport:
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
    scale_cells: tuple[int, ...] | None = None,
) -> HistoryBudgetPhaseDiagramCanaryReport:
    """Measure one excluded complete seed block without evaluation-roster access."""

    config = default_config(HistoryBudgetPhaseDiagramPhase.CANARY)
    scales = config.excluded_resource_canary_scale_cells if scale_cells is None else scale_cells
    if not scales or not set(scales).issubset(config.excluded_resource_canary_scale_cells):
        raise ValueError("resource canary scale roster differs from the excluded config")
    seed = history_budget_fixed_seed(programme_ordinal=1, scientific_role="excluded-resource-canary", numeric_index=0)
    wall_start = perf_counter()
    cpu_start = process_time()
    output_bytes = 0
    checks: set[str] = set()
    descriptors = tuple(
        generate_descriptor(
            unit_id="canary.resource.block-00",
            family=HistoryBudgetPhaseDiagramDisorderFamily.MULTISCALE_WAVELET_12,
            scale_cells=scale,
            seed=seed,
        )
        for scale in scales
    )
    scientific_input = fixed_history_budget_unit_input(programme_ordinal=1, unit_id="canary.resource.block-00", scale_cells=scales)
    if scientific_input.preparation_input is None:
        raise ValueError("resource canary requires its complete original preparation input")
    preparation_seed = bytes.fromhex(scientific_input.preparation_input.preparation_substream_seed_hex)
    denominators = HistoryBudgetPhaseDiagramDenominatorBundle(
        bundle_id="denominator-bundle.canary.resource.block-00",
        unit_id="canary.resource.block-00",
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
    generated = generator_bundle(
        config=config,
        denominators=denominators,
        observer=targeter.bundle,
        untouched=untouched.bundle,
        untouched_float_payload=untouched.float_arrays.payload,
        untouched_float_manifest=untouched.float_arrays.manifest,
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
            value.adjudication_id
            for value in adjudication.adjudications
            if not value.generator_observer_agreement
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
        "descriptor": HISTORY_BUDGET_PHASE_DIAGRAM_RESOURCE_PROFILE.small.output_bytes,
        "history": HISTORY_BUDGET_PHASE_DIAGRAM_RESOURCE_PROFILE.observer.output_bytes,
        "targeter": HISTORY_BUDGET_PHASE_DIAGRAM_RESOURCE_PROFILE.observer.output_bytes,
        "untouched": HISTORY_BUDGET_PHASE_DIAGRAM_RESOURCE_PROFILE.generator.output_bytes,
        "generator": HISTORY_BUDGET_PHASE_DIAGRAM_RESOURCE_PROFILE.generator.output_bytes,
        "adjudication": HISTORY_BUDGET_PHASE_DIAGRAM_RESOURCE_PROFILE.small.output_bytes,
    }
    per_task_output_passed = all(
        task_output_bytes[key] <= task_output_limits[key] for key in task_output_bytes
    )
    task_output_measurements = tuple(
        HistoryBudgetPhaseDiagramTaskOutputMeasurement(
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
            f"resource-profile-{HISTORY_BUDGET_PHASE_DIAGRAM_RESOURCE_PROFILE.profile_id}",
            "per-task-output-ceilings-audited",
        }
    )
    elapsed_wall = perf_counter() - wall_start
    elapsed_cpu = process_time() - cpu_start
    measured_wall_time = _decimal(elapsed_wall)
    measured_cpu_time = _decimal(elapsed_cpu)
    per_complete_unit_factor = 90
    projected_output = output_bytes * per_complete_unit_factor
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    resource_path_passed = (
        elapsed_cpu * per_complete_unit_factor <= 1_000 * 3600
        and elapsed_wall * per_complete_unit_factor / 2 <= 180 * 3600
        and projected_output <= 32 * 1024**3
        and per_task_output_passed
        and peak_rss_kib * 1024 <= 12 * 1024**3
    )
    passed = resource_path_passed and not disagreement_ids
    reasons = tuple(
        sorted(
            {
                *(("HISTORY_BUDGET_PHASE_DIAGRAM_RESOURCE_PATH_INADEQUATE",) if not resource_path_passed else ()),
                *(("HISTORY_BUDGET_PHASE_DIAGRAM_RESOURCE_CANARY_IMPLEMENTATION_DISAGREEMENT",) if disagreement_ids else ()),
            }
        )
    )
    return HistoryBudgetPhaseDiagramCanaryReport(
        report_id="history-budget-phase-diagram.excluded-resource-canary",
        check_ids=tuple(sorted(checks)),
        passed=passed,
        reason_codes=reasons,
        wall_time_seconds=measured_wall_time,
        peak_memory_bytes=int(peak_rss_kib * 1024),
        projected_evaluation_cpu_hours=(
            measured_cpu_time * Decimal(per_complete_unit_factor) / Decimal(3600)
        ),
        projected_evaluation_wall_hours=(
            measured_wall_time * Decimal(per_complete_unit_factor) / Decimal(2 * 3600)
        ),
        projected_evaluation_output_bytes=projected_output,
        evaluation_roster_access_count=0,
        task_output_measurements=task_output_measurements,
        measured_cpu_time_seconds=measured_cpu_time,
    )


def run_resource_canary(
    *,
    implementation_sha256: str,
    scale_cells: tuple[int, ...] | None = None,
) -> HistoryBudgetPhaseDiagramCanaryReport:
    with threadpool_limits(limits=2):
        return _run_resource_canary(
            implementation_sha256=implementation_sha256,
            scale_cells=scale_cells,
        )


def merge_canary_reports(
    *,
    source_check_ids: tuple[str, ...],
    numerical: HistoryBudgetPhaseDiagramCanaryReport,
    resource_report: HistoryBudgetPhaseDiagramCanaryReport,
) -> HistoryBudgetPhaseDiagramCanaryReport:
    passed = numerical.passed and resource_report.passed
    reasons = tuple(sorted({*numerical.reason_codes, *resource_report.reason_codes}))
    if not passed and not reasons:
        reasons = ("HISTORY_BUDGET_PHASE_DIAGRAM_CANARY_QUALIFICATION_FAILED",)
    return HistoryBudgetPhaseDiagramCanaryReport(
        report_id="history-budget-phase-diagram.source-canary-qualification",
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
    "run_cross_implementation_conformance",
    "run_generator_canary",
    "run_numerical_conformance",
    "run_observer_canary",
    "run_resource_canary",
]
