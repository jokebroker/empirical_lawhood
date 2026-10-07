"""Outcome-blind source, equation, rank, negative-control, and resource canaries."""

from __future__ import annotations

from empirical_lawhood.adapters.history_budget_fixed_scientific_inputs import fixed_history_budget_descriptor_input

from empirical_lawhood.adapters.history_budget_seed_constants import history_budget_fixed_seed

import ast
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
import resource
from time import perf_counter, process_time
from typing import Mapping

import numpy as np
from scipy.linalg import expm, svdvals
from scipy.sparse.linalg import expm_multiply
from scipy.sparse.linalg import spsolve
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from empirical_lawhood.adapters.methods.simulator_morphism_challenges.evaluator import adjudicate_unit_scale
from empirical_lawhood.adapters.methods.simulator_morphism_challenges.observer import assemble_dense_operator, observe_and_nominate
from empirical_lawhood.adapters.simulators.rc_ladder_morphism_challenges.generator import assemble_sparse_operator, generate_outcomes, receiver_matrix
from empirical_lawhood.kernel.evidence import OutcomeAccess

from .array_io import simulator_morphism_challenges_array_semantics, pack_arrays
from .contracts import SimulatorMorphismChallengeDenominatorDescriptor, SimulatorMorphismChallengeDisorderFamily, SimulatorMorphismChallengePhase
from .descriptors import default_config, generate_descriptor
from .runtime_contracts import SimulatorMorphismChallengeCanaryReport


_GENERATOR_PATH = 'src/empirical_lawhood/adapters/simulators/rc_ladder_morphism_challenges/generator.py'
_OBSERVER_PATH = 'src/empirical_lawhood/adapters/methods/simulator_morphism_challenges/observer.py'
_EVALUATOR_PATH = 'src/empirical_lawhood/adapters/methods/simulator_morphism_challenges/evaluator.py'
_SHARED_PREFIX = 'src/empirical_lawhood/adapters/simulator_morphism_challenges/'


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("simulator morphism challenges conformance metric must be finite and nonnegative")
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

    required = {_GENERATOR_PATH, _OBSERVER_PATH, _EVALUATOR_PATH}
    if not required.issubset(source_files):
        raise ValueError("simulator morphism challenges firewall audit lacks a claim-bearing source file")
    generator_imports = set(_imports(source_files[_GENERATOR_PATH]))
    observer_imports = set(_imports(source_files[_OBSERVER_PATH]))
    evaluator_imports = set(_imports(source_files[_EVALUATOR_PATH]))
    forbidden_generator = {
        value
        for value in generator_imports
        if value.startswith("empirical_lawhood.adapters.methods") or "rc_ladder_response" in value
    }
    forbidden_observer = {
        value
        for value in observer_imports
        if value.startswith("empirical_lawhood.adapters.simulators") or "physical_scale_morphism" in value
    }
    if forbidden_generator or forbidden_observer:
        raise ValueError("simulator morphism challenges generator--observer source firewall is violated")
    allowed_local_intersection = {'empirical_lawhood.adapters.simulator_morphism_challenges.contracts'}
    local_intersection = {
        value for value in generator_imports & observer_imports if value.startswith("empirical_lawhood")
    }
    if local_intersection != allowed_local_intersection:
        raise ValueError("simulator morphism challenges implementations share an undeclared scientific module")
    if 'empirical_lawhood.adapters.simulators.rc_ladder_morphism_challenges.generator' in evaluator_imports:
        raise ValueError("simulator morphism challenges evaluator imports the generator implementation")
    shared_paths = tuple(sorted(path for path in source_files if path.startswith(_SHARED_PREFIX)))
    if not {
        f"{_SHARED_PREFIX}contracts.py",
        f"{_SHARED_PREFIX}runtime_contracts.py",
    }.issubset(shared_paths):
        raise ValueError("simulator morphism challenges firewall audit lacks the shared record surface")
    return (
        "generator-does-not-import-observer-or-physical-scale-methods",
        "observer-does-not-import-generator-or-physical-scale-methods",
        "scientific-intersection-is-contracts-only",
    )


def _uniform_descriptor(scale_cells: int) -> SimulatorMorphismChallengeDenominatorDescriptor:
    seed = history_budget_fixed_seed(programme_ordinal=0, scientific_role="uniform-canary", numeric_index=scale_cells)
    base = generate_descriptor(
        unit_id=f"canary.truth-known.n{scale_cells}",
        family=SimulatorMorphismChallengeDisorderFamily.SMOOTH_PERIODIC_12,
        scale_cells=scale_cells,
        seed=seed,
    )
    return replace(
        base,
        latent_field_sha256=sha256(b"uniform-zero-field").hexdigest(),
        capacitances_farads=(base.capacitance_bar_farads,) * scale_cells,
        interior_resistances_ohms=(base.resistance_bar_ohms,) * (scale_cells - 1),
    )


def _analytic_equilibrium(descriptor: SimulatorMorphismChallengeDenominatorDescriptor, amplitude: float) -> np.ndarray:
    left = float(descriptor.left_source_resistance_ohms)
    right = float(descriptor.right_termination_resistance_ohms)
    interior = np.asarray(descriptor.interior_resistances_ohms, dtype=np.float64)
    current = amplitude / (left + float(np.sum(interior)) + right)
    preceding = np.concatenate((np.zeros(1), np.cumsum(interior)))
    return amplitude - current * (left + preceding)


def _rank_audit(descriptor: SimulatorMorphismChallengeDenominatorDescriptor) -> None:
    config = default_config(SimulatorMorphismChallengePhase.CANARY)
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
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_CANARY_SVD_AUDIT_FAILED")
        duplicated = np.vstack([operator.receiver] * (depth + 1))
        if np.linalg.matrix_rank(duplicated, tol=1e-12) > 8:
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_DUPLICATE_PRESENT_CONTROL_FAILED")
        shuffled = np.vstack(tuple(reversed(blocks)))
        if not np.allclose(
            np.linalg.svd(shuffled, compute_uv=False),
            numpy_values,
            atol=1e-10,
            rtol=1e-11,
        ):
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_SHUFFLED_LAG_CONTROL_FAILED")


def _run_generator_canary() -> SimulatorMorphismChallengeCanaryReport:
    """Qualify the sparse generator without importing observer implementation code."""

    start = perf_counter()
    config = default_config(SimulatorMorphismChallengePhase.CANARY)
    checks: set[str] = set()
    for scale_cells in config.scale_cells:
        descriptor = _uniform_descriptor(scale_cells)
        operator = assemble_sparse_operator(descriptor)
        equilibrium = np.asarray(
            spsolve(-operator.state_matrix, operator.left_input), dtype=np.float64
        )
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_GENERATOR_MANUFACTURED_SOLUTION_FAILED")
        if np.max(np.abs(operator.state_matrix @ equilibrium + operator.left_input)) > 1e-12:
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_GENERATOR_CURRENT_BALANCE_RESIDUAL_FAILED")
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
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_GENERATOR_PROPAGATOR_SEMIGROUP_FAILED")
        checks.update(
            {
                f"n{scale_cells}-generator-current-balance",
                f"n{scale_cells}-generator-manufactured-equilibrium",
            }
        )
    return SimulatorMorphismChallengeCanaryReport(
        report_id="simulator-morphism-challenges.generator-canary",
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


def run_generator_canary() -> SimulatorMorphismChallengeCanaryReport:
    with threadpool_limits(limits=2):
        return _run_generator_canary()


def _run_observer_canary() -> SimulatorMorphismChallengeCanaryReport:
    """Qualify dense observer equations, R8 and history controls in isolation."""

    start = perf_counter()
    config = default_config(SimulatorMorphismChallengePhase.CANARY)
    checks: set[str] = set()
    for scale_cells in config.scale_cells:
        descriptor = _uniform_descriptor(scale_cells)
        operator = assemble_dense_operator(descriptor)
        equilibrium = -np.linalg.solve(operator.state_matrix, operator.left_input)
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_OBSERVER_MANUFACTURED_SOLUTION_FAILED")
        if np.max(np.abs(operator.state_matrix @ equilibrium + operator.left_input)) > 1e-12:
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_OBSERVER_CURRENT_BALANCE_RESIDUAL_FAILED")
        _rank_audit(descriptor)
        checks.update(
            {
                f"n{scale_cells}-observer-manufactured-equilibrium",
                f"n{scale_cells}-observer-rank-and-history-controls",
            }
        )
    return SimulatorMorphismChallengeCanaryReport(
        report_id="simulator-morphism-challenges.observer-canary",
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


def run_observer_canary() -> SimulatorMorphismChallengeCanaryReport:
    with threadpool_limits(limits=2):
        return _run_observer_canary()


def _run_cross_implementation_conformance(
    *,
    implementation_sha256: str,
    generator_report: SimulatorMorphismChallengeCanaryReport,
    observer_report: SimulatorMorphismChallengeCanaryReport,
    independence_check_ids: tuple[str, ...],
) -> SimulatorMorphismChallengeCanaryReport:
    """Compare separately qualified implementations on the shared finite cases."""

    if (
        not generator_report.passed
        or not observer_report.passed
        or generator_report.report_id != "simulator-morphism-challenges.generator-canary"
        or observer_report.report_id != "simulator-morphism-challenges.observer-canary"
        or not independence_check_ids
    ):
        raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_CANARY_PREREQUISITE_FAILED")

    start = perf_counter()
    config = default_config(SimulatorMorphismChallengePhase.CANARY)
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
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_CANARY_OPERATOR_MISMATCH")
        if not np.allclose(dense.left_input, sparse.left_input, atol=1e-14, rtol=1e-14):
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_CANARY_INPUT_MISMATCH")
        if not np.allclose(dense.receiver, receiver_matrix(descriptor), atol=1e-14, rtol=1e-14):
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_CANARY_RECEIVER_MISMATCH")
        equilibrium = -np.linalg.solve(dense.state_matrix, dense.left_input)
        analytic = _analytic_equilibrium(descriptor, 1.0)
        if not np.allclose(equilibrium, analytic, atol=2e-13, rtol=2e-13):
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_CANARY_MANUFACTURED_SOLUTION_FAILED")
        if np.max(np.abs(dense.state_matrix @ equilibrium + dense.left_input)) > 1e-12:
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_CANARY_CURRENT_BALANCE_RESIDUAL_FAILED")
        initial = np.column_stack((np.linspace(0.1, 0.9, scale_cells), analytic))
        duration = 0.013 * float(descriptor.time_scale_seconds)
        dense_future = expm(dense.state_matrix * duration) @ initial
        sparse_future = np.asarray(expm_multiply(sparse.state_matrix * duration, initial))
        if not np.allclose(dense_future, sparse_future, atol=1e-11, rtol=1e-11):
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_CANARY_PROPAGATOR_MISMATCH")
        _rank_audit(descriptor)

        seed = history_budget_fixed_seed(programme_ordinal=0, scientific_role="conformance", numeric_index=scale_cells)
        challenge_descriptor = generate_descriptor(
            unit_id=f"canary.truth-known.n{scale_cells}",
            family=SimulatorMorphismChallengeDisorderFamily.CORRELATED_FIELD_12,
            scale_cells=scale_cells,
            seed=seed,
        )
        observer = observe_and_nominate(
            config=config,
            descriptor=challenge_descriptor,
            implementation_sha256=implementation_sha256,
            scientific_input=fixed_history_budget_descriptor_input(programme_ordinal=0, current_descriptor_sha256=challenge_descriptor.fingerprint()),
        )
        if not observer.nominations:
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_CANARY_NO_NOMINATIONS")
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
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_CANARY_CROSS_IMPLEMENTATION_FAILED")
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
                raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_CANARY_NEGATIVE_CONTROL_FAILED")
        checks.update(
            {
                f"n{scale_cells}-dense-sparse-equations",
                f"n{scale_cells}-manufactured-equilibrium",
                f"n{scale_cells}-rank-svd-and-history-controls",
                f"n{scale_cells}-source-termination-clock-negatives",
            }
        )
    return SimulatorMorphismChallengeCanaryReport(
        report_id="simulator-morphism-challenges.numerical-conformance",
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
    generator_report: SimulatorMorphismChallengeCanaryReport,
    observer_report: SimulatorMorphismChallengeCanaryReport,
    independence_check_ids: tuple[str, ...],
) -> SimulatorMorphismChallengeCanaryReport:
    with threadpool_limits(limits=2):
        return _run_cross_implementation_conformance(
            implementation_sha256=implementation_sha256,
            generator_report=generator_report,
            observer_report=observer_report,
            independence_check_ids=independence_check_ids,
        )


def run_numerical_conformance(*, implementation_sha256: str) -> SimulatorMorphismChallengeCanaryReport:
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
) -> SimulatorMorphismChallengeCanaryReport:
    """Measure one excluded complete seed block without evaluation-roster access."""

    config = default_config(SimulatorMorphismChallengePhase.CANARY)
    scales = config.excluded_resource_canary_scale_cells if scale_cells is None else scale_cells
    if not scales or not set(scales).issubset(config.excluded_resource_canary_scale_cells):
        raise ValueError("resource canary scale roster differs from the excluded config")
    seed = history_budget_fixed_seed(programme_ordinal=0, scientific_role="excluded-resource-canary", numeric_index=0)
    wall_start = perf_counter()
    cpu_start = process_time()
    output_bytes = 0
    checks: set[str] = set()
    for scale in scales:
        descriptor = generate_descriptor(
            unit_id="canary.resource.block-00",
            family=SimulatorMorphismChallengeDisorderFamily.MULTISCALE_WAVELET_12,
            scale_cells=scale,
            seed=seed,
        )
        observer = observe_and_nominate(
            config=config,
            descriptor=descriptor,
            implementation_sha256=implementation_sha256,
            scientific_input=fixed_history_budget_descriptor_input(programme_ordinal=0, current_descriptor_sha256=descriptor.fingerprint()),
        )
        if not observer.nominations:
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_RESOURCE_CANARY_UNTARGETABLE")
        generator = generate_outcomes(
            config=config,
            descriptor=descriptor,
            nominations=observer.nominations,
            implementation_sha256=implementation_sha256,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
        adjudication = adjudicate_unit_scale(
            config=config,
            descriptor=descriptor,
            forecast=observer.forecast,
            nominations=observer.nominations,
            generator_outcome=generator.outcome,
            method_freeze=None,
        )
        if not adjudication.generator_observer_agreement:
            raise ValueError("SIMULATOR_MORPHISM_CHALLENGE_RESOURCE_CANARY_IMPLEMENTATION_DISAGREEMENT")
        observer_arrays = pack_arrays(
            manifest_id=f"manifest.canary-resource.n{scale}.observer",
            unit_id="canary.resource.block-00",
            arrays={f"n{scale}.{key}": value for key, value in observer.arrays.items()},
            semantics={
                f"n{scale}.{key}": simulator_morphism_challenges_array_semantics(f"n{scale}.{key}")
                for key in observer.arrays
            },
        )
        generator_arrays = pack_arrays(
            manifest_id=f"manifest.canary-resource.n{scale}.generator",
            unit_id="canary.resource.block-00",
            arrays={f"n{scale}.{key}": value for key, value in generator.arrays.items()},
            semantics={
                f"n{scale}.{key}": simulator_morphism_challenges_array_semantics(f"n{scale}.{key}")
                for key in generator.arrays
            },
        )
        output_bytes += sum(
            (
                len(descriptor.canonical_bytes()),
                len(observer.forecast.canonical_bytes()),
                sum(len(value.canonical_bytes()) for value in observer.nominations),
                len(generator.outcome.canonical_bytes()),
                len(adjudication.canonical_bytes()),
                len(observer_arrays.payload),
                len(observer_arrays.manifest.canonical_bytes()),
                len(generator_arrays.payload),
                len(generator_arrays.manifest.canonical_bytes()),
            )
        )
        checks.add(f"n{scale}-excluded-resource-unit")
    elapsed_wall = perf_counter() - wall_start
    elapsed_cpu = process_time() - cpu_start
    per_complete_unit_factor = 36.0
    projected_output = int(np.ceil(output_bytes * per_complete_unit_factor))
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    passed = (
        elapsed_cpu * per_complete_unit_factor <= 72 * 3600
        and elapsed_wall * per_complete_unit_factor / 2 <= 36 * 3600
        and projected_output <= 20 * 1024**3
    )
    return SimulatorMorphismChallengeCanaryReport(
        report_id="simulator-morphism-challenges.excluded-resource-canary",
        check_ids=tuple(sorted(checks)),
        passed=passed,
        reason_codes=() if passed else ("SIMULATOR_MORPHISM_CHALLENGE_RESOURCE_PATH_INADEQUATE",),
        wall_time_seconds=_decimal(elapsed_wall),
        peak_memory_bytes=int(peak_rss_kib * 1024),
        projected_evaluation_cpu_hours=_decimal(elapsed_cpu * per_complete_unit_factor / 3600),
        projected_evaluation_wall_hours=_decimal(
            elapsed_wall * per_complete_unit_factor / (2 * 3600)
        ),
        projected_evaluation_output_bytes=projected_output,
        evaluation_roster_access_count=0,
    )


def run_resource_canary(
    *,
    implementation_sha256: str,
    scale_cells: tuple[int, ...] | None = None,
) -> SimulatorMorphismChallengeCanaryReport:
    with threadpool_limits(limits=2):
        return _run_resource_canary(
            implementation_sha256=implementation_sha256,
            scale_cells=scale_cells,
        )


def merge_canary_reports(
    *,
    source_check_ids: tuple[str, ...],
    numerical: SimulatorMorphismChallengeCanaryReport,
    resource_report: SimulatorMorphismChallengeCanaryReport,
) -> SimulatorMorphismChallengeCanaryReport:
    passed = numerical.passed and resource_report.passed
    reasons = () if passed else ("SIMULATOR_MORPHISM_CHALLENGE_RESOURCE_PATH_INADEQUATE",)
    return SimulatorMorphismChallengeCanaryReport(
        report_id="simulator-morphism-challenges.source-canary-qualification",
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
