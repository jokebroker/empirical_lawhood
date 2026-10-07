"""Independent algebra/passive controls; no history campaign or qualification."""

from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from empirical_lawhood.adapters.methods.matrix_history_analysis import algebra as a, passive as p
from empirical_lawhood.adapters.methods.matrix_history_analysis.analysis import algebra_sample_indices, passive_cells
from empirical_lawhood.adapters.methods.matrix_history_analysis.records import MatrixHistoryProtocol
from empirical_lawhood.adapters.simulators.six_matrix_response.model import su2_generators
from empirical_lawhood.api.matrix_history_analysis import prepare_matrix_history_allocation


def test_tensor_factor_and_irreducible_lie_counterexample():
    tensor, _ = a.algebra(a.coords(a.LEFT).T)
    spin, _ = a.algebra(a.coords(su2_generators(4) / np.sqrt(5)).T)
    assert tensor["lie_leakage"] < 1e-25
    assert tensor["associative_leakage"] < 1e-25
    assert tensor["casimir_relative"] < 1e-14
    assert spin["lie_leakage"] < 1e-25
    assert spin["associative_leakage"] > .1
    assert spin["eigenpair"] > .1
    deformed = a.sector(np.asarray([.5*a.RIGHT[0], 1.3*a.RIGHT[1], 2*a.RIGHT[2]]))
    assert deformed["identifiable"]
    assert deformed["metrics"]["associative_leakage"] < 1e-25


def test_gap_refusal_full_spectrum_and_joint_lifting():
    empty = a.sector(np.zeros((3, 4, 4), dtype="<c16"))
    assert not empty["identifiable"]
    assert len(empty["spectrum"]) == 15
    assert np.isnan(empty["factor"]).all()
    result = a.measure(np.stack((a.LEFT, a.RIGHT)))
    assert result["x"]["identifiable"] and result["y"]["identifiable"]
    assert result["y"]["spectrum"][2] < 1e-12
    assert result["joint_spectrum"][0] > .9
    assert result["y"]["metrics"]["commutant_mixing"] < 1e-25


def test_invariant_spectra_products_and_constructive_factor():
    rng = np.random.default_rng(812)
    unitary, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    triplet = a.RIGHT + .17 * a.LEFT
    baseline = a.sector(triplet)
    rotated = a.sector(unitary @ triplet @ unitary.conj().T)
    np.testing.assert_allclose(baseline["spectrum"], rotated["spectrum"], atol=1e-13)
    assert baseline["metrics"]["associative_leakage"] == pytest.approx(rotated["metrics"]["associative_leakage"], abs=1e-13)
    factor = a.exact_factor(a.LEFT)
    for field in factor:
        np.testing.assert_allclose(np.linalg.eigvalsh(field), [-.5, -.5, .5, .5], atol=1e-14)


def test_positive_gram_matches_independent_kronecker_oracle():
    from empirical_lawhood.adapters.simulators.six_matrix_response.spectral import adjoint_laplacian_explicit
    triplet = a.RIGHT + .17 * a.LEFT
    basis = np.column_stack([field.reshape(16, order="F") for field in a.BASIS])
    independent = basis.conj().T @ adjoint_laplacian_explicit(triplet) @ basis
    np.testing.assert_allclose(a.operator(triplet), independent, atol=1e-14)
    assert not a.sector(1e-11*a.RIGHT)["identifiable"]


def test_band_ratio_is_third_over_fourth_and_public_columns_preserve_it():
    from empirical_lawhood.adapters.methods.matrix_history_analysis.analysis import ALGEBRA_METRICS, algebra_history
    from empirical_lawhood.adapters.simulators.six_matrix_response.spectral import adjoint_laplacian_explicit

    rng = np.random.default_rng(912)
    raw = rng.normal(size=(2, 3, 4, 4)) + 1j * rng.normal(size=(2, 3, 4, 4))
    state = (raw + raw.swapaxes(-1, -2).conj()) / 2
    basis = np.column_stack([field.reshape(16, order="F") for field in a.BASIS])
    expected = []
    for triplet in state:
        spectrum = np.linalg.eigvalsh(basis.conj().T @ adjoint_laplacian_explicit(triplet) @ basis)
        ratio = spectrum[2] / spectrum[3]
        assert abs(ratio - spectrum[:3].mean() / spectrum[3]) > .01
        result = a.sector(triplet)
        assert result["identifiable"]
        assert result["metrics"]["band_ratio"] == pytest.approx(ratio, abs=1e-14)
        expected.append(ratio)
    root = prepare_matrix_history_allocation(allocation_id="test.band", namespace="test.band", master_seed=912).roots[0]
    arrays = {f"{view}_positions": np.broadcast_to(state, (1024 * multiplier + 1, *state.shape))
              for view, multiplier in (("primary", 1), ("fine", 2))}
    public, _, _, _ = algebra_history(arrays, root, compare_commutant_altered=False)
    for sector, ratio in zip(("x", "y"), expected, strict=True):
        np.testing.assert_allclose(public["metrics"][:, ALGEBRA_METRICS.index(sector + ".band_ratio")], ratio, atol=1e-14)


def test_constant_generator_rk4_and_nochange_reference():
    path = np.repeat(a.RIGHT[None], 33, axis=0)
    actual, rates, integrals = p.propagators(path, .001, (1.,))
    expected = expm(-.032 * a.operator(a.RIGHT))
    np.testing.assert_allclose(actual[-1, 0], expected, atol=1e-13)
    np.testing.assert_allclose(integrals[-1], .032 * rates[0], atol=1e-14)
    seal = p.seal_causal_forecast(path[0], kappa=1, horizon=.032)
    evaluated = p.evaluate_forecast(actual[-1, 0], seal=seal, initial_operator=rates[0], endpoint_operator=rates[-1], integrated_operator=integrals[-1])
    assert evaluated["metrics"]["frozen.full"] < 1e-22
    assert evaluated["metrics"]["nochange.full"] == pytest.approx(1.)
    assert evaluated["increment_rank_absolute"] == 12
    assert evaluated["increment_unresolved"]


def test_noncommuting_midpoint_in_y_against_independent_ode():
    rng = np.random.default_rng(27)
    unitary, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    left, right = a.RIGHT, unitary @ a.RIGHT @ unitary.conj().T
    path = np.asarray([(1-t)*left+t*right for t in np.linspace(0, 1, 17)])
    dt = .002
    def derivative(t, state):
        fraction = t / .032
        return (-a.operator((1-fraction)*left+fraction*right) @ state.reshape(15, 15)).ravel()
    expected = solve_ivp(derivative, (0, .032), np.eye(15).ravel(), method="DOP853", rtol=1e-11, atol=1e-13).y[:, -1].reshape(15, 15)
    result, rates, integrals = p.propagators(path, dt, (1.,))
    np.testing.assert_allclose(result[-1, 0], expected, atol=2e-9)
    wrong_midpoint = .032 * (rates[0]+rates[-1])/2
    assert np.linalg.norm(integrals[-1] - wrong_midpoint) > 1e-3
    assert np.linalg.norm(rates[0] @ rates[-1] - rates[-1] @ rates[0]) > 1e-2


def test_origin_only_seal_future_poisoning_and_unresolved_increment():
    seal = p.seal_causal_forecast(a.RIGHT, kappa=1, horizon=.032)
    for future in (a.LEFT, 20*a.RIGHT):
        actual, rates, integrals = p.propagators(np.stack((a.RIGHT, future)), .032, (1.,))
        p.evaluate_forecast(actual[-1, 0], seal=seal, initial_operator=rates[0], endpoint_operator=rates[-1], integrated_operator=integrals[-1])
        assert p.seal_causal_forecast(a.RIGHT, kappa=1, horizon=.032).sha256 == seal.sha256
    assert not seal.predictions[0].flags.writeable
    zero = np.zeros((15, 15))
    empty_seal = p.seal_causal_forecast(np.zeros((3, 4, 4), dtype="<c16"), kappa=1, horizon=.032)
    outcome = p.evaluate_forecast(np.eye(15), seal=empty_seal, initial_operator=zero, endpoint_operator=zero, integrated_operator=zero)
    assert outcome["increment_rank_absolute"] == 0
    assert outcome["increment_unresolved"]
    assert outcome["metrics"]["nochange.full"] == 0


def test_bad_direction_with_small_aggregate_and_distinct_rank_conventions():
    zero = np.zeros((15, 15))
    seal = p.seal_causal_forecast(np.zeros((3, 4, 4), dtype="<c16"), kappa=1, horizon=.032)
    actual = np.diag([.5]*14 + [1-1e-9])
    prediction = actual.copy()
    prediction[-1, -1] += 1e-3
    engineered = replace(seal, predictions=(prediction,)*4)
    result = p.evaluate_forecast(actual, seal=engineered, initial_operator=zero, endpoint_operator=zero, integrated_operator=zero)
    assert result["metrics"]["frozen.full"] < 1e-6
    assert result["metrics"]["frozen.worst"] > 1e11
    tiny = np.eye(15)-np.diag([1e-13]*15)
    result = p.evaluate_forecast(tiny, seal=seal, initial_operator=zero, endpoint_operator=zero, integrated_operator=zero)
    assert result["increment_rank_absolute"] == 0
    assert result["metrics"]["nochange.worst"] == pytest.approx(1)


def test_prebound_dense_role_not_outcome_search_and_protocol_refusal():
    allocation = prepare_matrix_history_allocation(allocation_id="sample.history", namespace="sample", master_seed=71)
    ordinary, dense = allocation.roots[64], allocation.roots[82]
    assert len(algebra_sample_indices(ordinary, 1)) == len(algebra_sample_indices(ordinary, 2)) == 65
    assert len(algebra_sample_indices(dense, 1)) == 1025
    assert len(algebra_sample_indices(dense, 2)) == 2049
    assert len(passive_cells(ordinary)) == 4
    assert (944, 128, 1.) not in passive_cells(dense)
    assert (944, 64, 1.) in passive_cells(dense)
    with pytest.raises(ValueError, match="protocol"):
        replace(MatrixHistoryProtocol(), dense_family_slot=19)
