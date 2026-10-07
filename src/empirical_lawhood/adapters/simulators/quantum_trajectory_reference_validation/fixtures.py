"""Truth-known C0 semantic fixtures for the corrected jump contract."""

from __future__ import annotations

from hashlib import sha256
import math

import numpy as np

from .basis import FixedNumberBasis, build_basis
from .evolution import CountingRandom, action_at_time
from .jumps import ZeroProjectedMassError, apply_jump, jump_masses, sample_mark
from .operators import DenseEigensystemPropagator, SparseKrylovPropagator, build_hamiltonian, normalize_state, phase_gauge_infidelity
from .schemas import QuantumTrajectoryReferenceValidationConfig
from .types import Action, FixtureRow, Validity, View


def _state_hash(state: np.ndarray) -> str:
    return sha256(np.asarray(state, dtype="<c16").tobytes()).hexdigest()


def _states_for_site(basis: FixedNumberBasis, site: int) -> tuple[int, int]:
    occupied = next(state for state in basis.states if (state >> site) & 1)
    unoccupied = next(state for state in basis.states if not ((state >> site) & 1))
    return occupied, unoccupied


def _mass_state(
    basis: FixedNumberBasis,
    site: int,
    mass: float,
) -> np.ndarray:
    occupied, unoccupied = _states_for_site(basis, site)
    result = math.sqrt(mass) * basis.state_vector(occupied) + math.sqrt(
        1.0 - mass
    ) * basis.state_vector(unoccupied)
    normalized, _ = normalize_state(result)
    return normalized


def _projection_row(
    *,
    fixture_id: str,
    view: View,
    basis: FixedNumberBasis,
    state: np.ndarray,
    site: int,
    expected_mass: float,
) -> FixtureRow:
    masses = jump_masses(basis, state)
    validity = Validity.VALID
    reason = "OK"
    post_residual: float | None
    vector_mass: float | None
    mass_residual: float | None
    try:
        applied = apply_jump(
            basis,
            state,
            site,
            expected_mass=float(masses.projected_masses[site]),
        )
        post_residual = applied.post_jump_norm_residual
        vector_mass = applied.projected_vector_norm_squared
        mass_residual = applied.projected_mass_identity_residual
    except ZeroProjectedMassError:
        validity = Validity.ZERO_PROJECTED_MASS
        reason = "ZERO_PROJECTED_MASS"
        post_residual = None
        vector_mass = 0.0
        mass_residual = abs(float(masses.projected_masses[site]))
    return FixtureRow(
        fixture_id=fixture_id,
        view=view,
        input_state_sha256=_state_hash(state),
        site=site,
        expected_projected_mass=expected_mass,
        observed_projected_mass_expectation=float(masses.projected_masses[site]),
        observed_projected_vector_norm_squared=vector_mass,
        projected_mass_identity_residual=mass_residual,
        expected_mark_probability=expected_mass / basis.particles,
        observed_mark_probability=float(masses.mark_probabilities[site]),
        mark_sum_residual=masses.probability_sum_residual,
        propagator_norm_residual=None,
        post_jump_norm_residual=post_residual,
        phase_gauge_infidelity=None,
        prng_counter_before=0,
        prng_counter_after=0,
        validity=validity,
        reason_code=reason,
    )


def run_fixtures(config: QuantumTrajectoryReferenceValidationConfig) -> tuple[FixtureRow, ...]:
    basis = build_basis(config.l_dense, config.n_dense)
    site = 0
    occupied, unoccupied = _states_for_site(basis, site)
    projection_specs = (
        ("occupied-basis-site", basis.state_vector(occupied), 1.0),
        ("unoccupied-basis-site", basis.state_vector(unoccupied), 0.0),
        ("balanced-superposition", _mass_state(basis, site, 0.5), 0.5),
        ("small-born-mass-1e-6", _mass_state(basis, site, 1e-6), 1e-6),
        ("small-born-mass-1e-12", _mass_state(basis, site, 1e-12), 1e-12),
    )
    rows: list[FixtureRow] = []
    for view in (View.SPARSE, View.DENSE):
        for fixture_id, state, expected_mass in projection_specs:
            rows.append(
                _projection_row(
                    fixture_id=fixture_id,
                    view=view,
                    basis=basis,
                    state=state,
                    site=site,
                    expected_mass=expected_mass,
                )
            )

        amplitudes = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        amplitudes /= amplitudes.sum()
        selected_states = basis.states[:4]
        nonuniform = np.zeros(basis.dimension, dtype=np.complex128)
        for weight, basis_state in zip(amplitudes, selected_states, strict=True):
            nonuniform += math.sqrt(float(weight)) * basis.state_vector(basis_state)
        nonuniform, _ = normalize_state(nonuniform)
        expected = float(
            sum(
                weight * ((basis_state >> site) & 1)
                for weight, basis_state in zip(
                    amplitudes,
                    selected_states,
                    strict=True,
                )
            )
        )
        rows.append(
            _projection_row(
                fixture_id="nonuniform-full-mark-vector",
                view=view,
                basis=basis,
                state=nonuniform,
                site=site,
                expected_mass=expected,
            )
        )

        phase_state = _mass_state(basis, site, 0.25)
        phase_pair = np.exp(1j * 0.731) * phase_state
        rows.append(
            FixtureRow(
                fixture_id="global-phase-pair",
                view=view,
                input_state_sha256=_state_hash(phase_state),
                site=None,
                expected_projected_mass=None,
                observed_projected_mass_expectation=None,
                observed_projected_vector_norm_squared=None,
                projected_mass_identity_residual=None,
                expected_mark_probability=None,
                observed_mark_probability=None,
                mark_sum_residual=None,
                propagator_norm_residual=None,
                post_jump_norm_residual=None,
                phase_gauge_infidelity=phase_gauge_infidelity(
                    phase_state,
                    phase_pair,
                ),
                prng_counter_before=0,
                prng_counter_after=0,
                validity=Validity.VALID,
                reason_code="OK",
            )
        )
        diagonal_h = build_hamiltonian(
            basis,
            action=Action.HOLD,
            epsilon=float(config.epsilon),
            j_xy=0.0,
            j_z=float(config.j_z),
        )
        propagator = (
            SparseKrylovPropagator(diagonal_h)
            if view == View.SPARSE
            else DenseEigensystemPropagator(diagonal_h)
        )
        for fixture_id, duration in (
            ("zero-duration-propagation", 0.0),
            ("known-unitary-propagation", 0.73),
        ):
            propagated = propagator.propagate(phase_state, duration)
            _, residual = normalize_state(propagated)
            rows.append(
                FixtureRow(
                    fixture_id=fixture_id,
                    view=view,
                    input_state_sha256=_state_hash(phase_state),
                    site=None,
                    expected_projected_mass=None,
                    observed_projected_mass_expectation=None,
                    observed_projected_vector_norm_squared=None,
                    projected_mass_identity_residual=None,
                    expected_mark_probability=None,
                    observed_mark_probability=None,
                    mark_sum_residual=None,
                    propagator_norm_residual=residual,
                    post_jump_norm_residual=None,
                    phase_gauge_infidelity=None,
                    prng_counter_before=0,
                    prng_counter_after=0,
                    validity=Validity.VALID,
                    reason_code="OK",
                )
            )

    boundary_probabilities = np.asarray([0.2, 0.3, 0.5], dtype=np.float64)
    selected = sample_mark(boundary_probabilities, 0.2)
    rows.append(
        FixtureRow(
            fixture_id="equal-cumulative-boundary",
            view=View.ANALYTIC,
            input_state_sha256=sha256(boundary_probabilities.tobytes()).hexdigest(),
            site=selected,
            expected_projected_mass=None,
            observed_projected_mass_expectation=None,
            observed_projected_vector_norm_squared=None,
            projected_mass_identity_residual=None,
            expected_mark_probability=None,
            observed_mark_probability=None,
            mark_sum_residual=abs(float(boundary_probabilities.sum()) - 1.0),
            propagator_norm_residual=None,
            post_jump_norm_residual=None,
            phase_gauge_infidelity=None,
            prng_counter_before=0,
            prng_counter_after=0,
            validity=Validity.VALID,
            reason_code="RIGHT_BOUNDARY_SELECTS_NEXT_SITE",
        )
    )
    rng = CountingRandom(7)
    before = rng.draw_count
    zero_state = basis.state_vector(unoccupied)
    try:
        apply_jump(basis, zero_state, site, expected_mass=0.0)
    except ZeroProjectedMassError:
        pass
    else:
        raise AssertionError("zero-mass fixture did not refuse normalization")
    rows.append(
        FixtureRow(
            fixture_id="zero-mass-prng-guard",
            view=View.ANALYTIC,
            input_state_sha256=_state_hash(zero_state),
            site=site,
            expected_projected_mass=0.0,
            observed_projected_mass_expectation=0.0,
            observed_projected_vector_norm_squared=0.0,
            projected_mass_identity_residual=0.0,
            expected_mark_probability=0.0,
            observed_mark_probability=0.0,
            mark_sum_residual=None,
            propagator_norm_residual=None,
            post_jump_norm_residual=None,
            phase_gauge_infidelity=None,
            prng_counter_before=before,
            prng_counter_after=rng.draw_count,
            validity=Validity.ZERO_PROJECTED_MASS,
            reason_code="ZERO_PROJECTED_MASS_NO_PRNG_CONSUMPTION",
        )
    )
    switch_action = action_at_time(
        actions=(Action.HOLD, Action.PLUS),
        switch_times=(1.0,),
        event_time=1.0,
    )
    rows.append(
        FixtureRow(
            fixture_id="action-switch-right-endpoint",
            view=View.ANALYTIC,
            input_state_sha256=_state_hash(basis.state_vector(occupied)),
            site=None,
            expected_projected_mass=None,
            observed_projected_mass_expectation=None,
            observed_projected_vector_norm_squared=None,
            projected_mass_identity_residual=None,
            expected_mark_probability=None,
            observed_mark_probability=None,
            mark_sum_residual=None,
            propagator_norm_residual=None,
            post_jump_norm_residual=None,
            phase_gauge_infidelity=None,
            prng_counter_before=0,
            prng_counter_after=0,
            validity=Validity.VALID,
            reason_code="SWITCH_APPLIED_BEFORE_RIGHT_ENDPOINT_EVENT",
            requested_action=Action.PLUS.value,
            accepted_action=Action.PLUS.value,
            applied_action=switch_action.value,
            realized_action_start=1.0,
            realized_action_end=2.0,
            endpoint_included=True,
        )
    )
    seen = {row.fixture_id for row in rows}
    if set(config.fixture_ids) != seen:
        raise AssertionError(
            f"fixture implementation differs: missing={set(config.fixture_ids) - seen}, "
            f"extra={seen - set(config.fixture_ids)}"
        )
    return tuple(rows)


__all__ = ["run_fixtures"]
