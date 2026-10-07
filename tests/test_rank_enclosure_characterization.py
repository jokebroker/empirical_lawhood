# SPDX-License-Identifier: MPL-2.0
"""Independent exact-rational and analytic acceptance for A20's repair."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal, getcontext, localcontext
from fractions import Fraction
import importlib
from pathlib import Path
from types import SimpleNamespace

from flint import arb, arb_mat, ctx, fmpq
import mpmath as mp
import numpy as np
import pytest

from empirical_lawhood.adapters.methods import certified_rank as rank
from empirical_lawhood.adapters.methods._arb import endpoint_decimal
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from tests.probes.rank_enclosure_probe import FAMILIES, exact_endpoint, observe_endpoints, observe_residuals


@pytest.mark.parametrize("family", tuple(FAMILIES))
@pytest.mark.parametrize("threshold", (False, True))
@pytest.mark.parametrize("digits", (2, 15, 180))
def test_current_endpoint_counterexamples_enclose_at_every_context(family, threshold, digits):
    observed = observe_endpoints(family, digits, threshold=threshold)
    assert observed["lower_outward"] and observed["upper_outward"]
    assert Fraction(observed["converted_width"]) >= Fraction(observed["reference_width"]) > 0


@pytest.mark.parametrize("midpoint,radius", (
    ((1, 0), (1, -400)), ((-1, 0), (1, -400)), ((0, 0), (1, -400)),
    ((1, -160), (1, -500)), ((1, -1074), (1, -1200)), ((1, 1023), (1, 900)),
))
@pytest.mark.parametrize("bits", (53, 384))
def test_endpoint_direction_against_exact_stored_ball(midpoint, radius, bits):
    with ctx.workprec(bits), localcontext() as context:
        context.prec = 2
        context.clear_flags()
        for signal in context.traps:
            context.traps[signal] = True
        ball = arb(midpoint, radius)
        assert Fraction(endpoint_decimal(ball, upper=False)) <= exact_endpoint(ball, upper=False)
        assert Fraction(endpoint_decimal(ball, upper=True)) >= exact_endpoint(ball, upper=True)
        assert not any(context.flags.values())


def test_more_than_eighty_decimal_digits_are_preserved_at_the_numeric_boundary():
    exact = Fraction(1) + Fraction(1, 2**300)
    with ctx.workprec(384), localcontext() as context:
        context.prec = 7
        for signal in context.traps:
            context.traps[signal] = True
        point = arb(fmpq(exact.numerator, exact.denominator))
        assert Fraction(endpoint_decimal(point, upper=True)) == exact
        assert Fraction(endpoint_decimal(point, upper=False)) == exact


@pytest.mark.parametrize("value", ("nan", "+inf", "-inf"))
def test_nonfinite_endpoints_refuse(value):
    with pytest.raises(ValueError, match="finite"):
        endpoint_decimal(arb(value), upper=True)


@pytest.mark.parametrize("family", tuple(FAMILIES))
def test_rank_residual_and_bytes_do_not_depend_on_mpmath_or_decimal_context(family):
    observations = []
    for digits in (2, 15, 180):
        with localcontext() as context:
            context.prec = digits
            observations.append(observe_residuals(family, digits))
    assert all(row["rank"] == [2, 2] for row in observations)
    assert len({row["residual"] for row in observations}) == 1
    assert Decimal(observations[0]["residual"]) > 0
    assert all(row["separation"] is None for row in observations)


def _fraction(point):
    exact = point.fmpq()
    return Fraction(int(exact.p), int(exact.q))


@pytest.mark.parametrize("matrix", (
    np.array([[0.1, 1e-90], [0.3, 0.5]]),
    np.array([[0.1, 0.3], [1e-90, 0.5], [2.**-600, -0.25]]),
    np.array([[0.1, 0.3, 2.**-600], [1e-90, 0.5, -0.25]]),
))
def test_stored_residual_bounds_independently_exact_gram_error(matrix):
    certificate = rank.certify_rank(matrix)
    # Fraction defines the exact binary64 Gram independently of Arb arithmetic.
    a = [[Fraction(float(value)) for value in row] for row in matrix]
    vectors = list(zip(*a)) if matrix.shape[0] >= matrix.shape[1] else a
    exact_gram = [[sum((x*y for x,y in zip(left,right)), Fraction(0))
                   for right in vectors] for left in vectors]
    with ctx.workprec(certificate.precision_bits):
        entered = arb_mat(matrix.shape[0], matrix.shape[1],
                          [arb(fmpq(v.numerator, v.denominator)) for row in a for v in row])
        gram = entered.transpose() * entered if matrix.shape[0] >= matrix.shape[1] else entered * entered.transpose()
        errors_squared = sum((exact_gram[i][j] - _fraction(gram[i,j].mid()))**2
                             for i in range(len(vectors)) for j in range(len(vectors)))
    residual = Fraction(certificate.residual)
    assert residual > 0 and errors_squared > 0
    assert residual**4 >= errors_squared
    assert 4 * residual <= Fraction(certificate.tau_lower)
    # Independent norm inequalities check threshold direction without a rounded sqrt.
    scale = Fraction(max(matrix.shape), 2**160)
    squared_columns = [sum((v*v for v in column), Fraction(0)) for column in zip(*a)]
    assert (Fraction(certificate.tau_lower)/scale)**2 <= max(Fraction(1), max(squared_columns))
    assert (Fraction(certificate.tau_upper)/scale)**2 >= max(Fraction(1), sum(squared_columns))


_TAU = 2.**-159  # two-dimensional matrices with norm <= 1
@pytest.mark.parametrize("small,expected", (
    (np.nextafter(_TAU, 0.0), 1), (_TAU, 1), (np.nextafter(_TAU, np.inf), 2),
    (1e-100, 1), (0.0, 1),
))
def test_threshold_adjacent_singular_values_use_strict_greater_than(small, expected):
    certificate = rank.certify_rank(np.diag([0.5, small]))
    assert certificate.lower <= expected <= certificate.upper
    if small != _TAU:
        assert certificate.lower == certificate.upper == expected
    else:
        assert certificate.lower < certificate.upper


@pytest.mark.parametrize("matrix,expected", (
    (np.zeros((3, 2)), 0), (np.eye(4), 4), (np.vstack((np.eye(3), np.eye(3))), 3),
    (np.array([[1., 2.], [2., 4.]]), 1),
    (_TAU * np.array([[1., 1.], [0., 1.]]), 1),
    (np.array([[0.5, _TAU], [0., _TAU]]), 1),
    (np.diag([np.finfo(float).max, np.finfo(float).max]), 2),
    (np.diag([np.finfo(float).max, np.nextafter(0., 1.)]), 1),
    (np.diag([np.nextafter(0., 1.), np.nextafter(0., 1.)]), 0),
))
def test_analytic_controls_include_extreme_binary64_magnitudes(matrix, expected):
    for entered in (matrix, matrix.T, -matrix[::-1, ::-1]):
        certificate = rank.certify_rank(entered)
        assert (certificate.lower, certificate.upper) == (expected, expected)


def test_exact_rank_only_caps_and_never_promotes_a_subthreshold_direction():
    result = rank.certify_rank(np.diag([0.5, 1e-100]))
    assert (result.lower, result.upper) == (1, 1)  # exact algebraic rank is two


def test_insufficient_residual_budget_returns_wide_rank_and_actual_bound(monkeypatch):
    original = rank._enclosures
    attempts = []

    def inflated(entered):
        gram, lower, upper, residual = original(entered)
        attempts.append(ctx.prec)
        return gram, lower, upper, lower  # valid loose bound for this exact fixture

    monkeypatch.setattr(rank, "_enclosures", inflated)
    monkeypatch.setattr(rank, "_inertia_above", lambda *a: pytest.fail("uncertified inertia admitted"))
    result = rank.certify_rank(np.eye(2))
    assert attempts == [384, 640]
    assert (result.lower, result.upper) == (0, 2)
    assert result.residual == result.tau_lower > 0


def test_unresolved_pivots_retain_exact_rank_upper_bound(monkeypatch):
    def unresolved(*args):
        raise rank._UnresolvedInertia("synthetic uncertainty")
    monkeypatch.setattr(rank, "_inertia_above", unresolved)
    result = rank.certify_rank(np.array([[1., 2.], [2., 4.]]))
    assert (result.lower, result.upper) == (0, 1)
    assert result.residual == 0  # exact Gram; no fabricated residual sentinel
    assert result.precision_bits == 640


@pytest.mark.parametrize("matrix", (np.array([1.]), np.zeros((0, 2)), np.array([[np.inf]]),
                                   np.array([[np.nan]]), np.array([[1+1j]])))
def test_invalid_inputs_refuse(matrix):
    with pytest.raises(ValueError):
        rank.certify_rank(matrix)


@pytest.mark.parametrize("precision", (53, 255, 256.0, True))
def test_invalid_precision_refuses(precision):
    with pytest.raises(ValueError):
        rank.certify_rank(np.eye(2), precision_bits=precision)


def test_context_restoration_on_success_uncertainty_and_exception(monkeypatch):
    c = getcontext()
    before = (c.copy(), mp.mp.prec, ctx.prec)
    with ctx.workprec(97):
        for matrix in (np.eye(2), np.diag([0.5, _TAU])):
            rank.certify_rank(matrix)
            assert ctx.prec == 97
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda bits: rank.certify_rank(np.eye(2), precision_bits=bits), (384, 512, 768)))
        assert all((result.lower, result.upper) == (2, 2) for result in results)
        assert ctx.prec == 97
        def fail(*args):
            raise RuntimeError("synthetic failure")
        monkeypatch.setattr(rank, "_enclosures", fail)
        with pytest.raises(RuntimeError):
            rank.certify_rank(np.eye(2))
        assert ctx.prec == 97
    assert mp.mp.prec == before[1] and ctx.prec == before[2]
    assert str(c) == str(before[0])


@pytest.mark.parametrize("family", tuple(FAMILIES))
def test_family_record_producers_use_the_shared_certificate(family):
    module = FAMILIES[family]
    assert module.residual_certified_rank_bracket is rank.residual_certified_rank_bracket
    matrix = np.vstack((np.array([[0.1, 1e-90], [0.3, 0.5]]), np.zeros((6, 2))))
    record = module.discrete_rank_bracket(
        SimpleNamespace(discrete_rank_algorithm_id=rank.ALGORITHM_ID),
        SimpleNamespace(scale_cells=2), SimpleNamespace(depth=0, coordinate_id="test-coordinate"),
        SimpleNamespace(coordinate_id="test-coordinate", structural_rank=2), matrix=matrix,
    )
    assert (record.lower_rank, record.upper_rank) == (2, 2)
    assert record.residual_norm > 0 and record.separation_ratio is None
    assert record.algorithm_id == rank.ALGORITHM_ID
    with localcontext() as context:
        context.prec = 7
        payload = record.canonical_bytes()
        decoded = decode_canonical_bytes(payload, type(record), maximum_bytes=len(payload))
        assert decoded.residual_norm == record.residual_norm
        assert decoded.fingerprint() == record.fingerprint()
    with pytest.raises(ValueError, match="real"):
        module.discrete_rank_bracket(
            SimpleNamespace(discrete_rank_algorithm_id=rank.ALGORITHM_ID),
            SimpleNamespace(scale_cells=2), SimpleNamespace(depth=0, coordinate_id="test-coordinate"),
            SimpleNamespace(coordinate_id="test-coordinate", structural_rank=2), matrix=matrix + 1j,
        )


@pytest.mark.parametrize("family,prefix", (
    ("receiver_history", "ReceiverHistory"), ("split_cohort_history_budget", "SplitCohortHistoryBudget"),
    ("history_budget_phase_diagram", "HistoryBudgetPhaseDiagram"),
))
def test_new_algorithm_and_shared_source_are_bound(family, prefix):
    contracts = importlib.import_module(f"empirical_lawhood.adapters.{family}.contracts")
    descriptors = importlib.import_module(f"empirical_lawhood.adapters.{family}.descriptors")
    provider = importlib.import_module(f"empirical_lawhood.adapters.{family}.runtime_provider")
    conformance = importlib.import_module(f"empirical_lawhood.adapters.{family}.conformance")
    config = descriptors.default_config(getattr(contracts, prefix + "Phase").CANARY)
    assert config.discrete_rank_algorithm_id == rank.ALGORITHM_ID
    with pytest.raises(ValueError):
        replace(config, discrete_rank_algorithm_id="obsolete-rank-certificate")
    paths = getattr(provider, family.upper() + "_ALL_SOURCE_CLOSURE_PATHS")
    root = Path(__file__).resolve().parents[1]
    sources = {path: (root / path).read_bytes() for path in paths}
    before = provider.implementation_closures(sources)
    conformance.audit_source_firewall(sources)
    for path in ("src/empirical_lawhood/adapters/methods/_arb.py",
                 "src/empirical_lawhood/adapters/methods/certified_rank.py",
                 "src/empirical_lawhood/kernel/serialization.py"):
        changed = {**sources, path: sources[path] + b"\n# synthetic source change\n"}
        after = provider.implementation_closures(changed)
        assert before.observer_sha256 != after.observer_sha256
        assert before.complete_sha256 != after.complete_sha256
        if "/kernel/" in path:
            continue
        with pytest.raises(ValueError, match="firewall"):
            conformance.audit_source_firewall({**sources, path: b"import empirical_lawhood.adapters.simulators.forbidden\n"})


@pytest.mark.parametrize("family", tuple(FAMILIES))
def test_unresolved_family_record_cannot_be_promoted_by_structural_comparison(family):
    # Eight rows give tau=2^-157. Equality makes interval inertia unresolved;
    # the exact rank upper bound is two, above the supplied structural rank.
    matrix = np.vstack((np.diag([0.5, 2.**-157]), np.zeros((6, 2))))
    record = FAMILIES[family].discrete_rank_bracket(
        SimpleNamespace(discrete_rank_algorithm_id=rank.ALGORITHM_ID),
        SimpleNamespace(scale_cells=2), SimpleNamespace(depth=0, coordinate_id="test-coordinate"),
        SimpleNamespace(coordinate_id="test-coordinate", structural_rank=1), matrix=matrix,
    )
    assert record.lower_rank < record.upper_rank
    assert record.compatibility.value == "RANK_UNRESOLVED"
