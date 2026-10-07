"""Current four-family complete paired history production.

Driver purposes are allocated before effects. Every primary OU innovation is
split into two conditional fine innovations; views are nested, not new roots.
The initial family, ramp and autonomous clocks follow the retained source.
"""

from decimal import Decimal

import numpy as np

from empirical_lawhood.adapters.methods.matrix_history_analysis.records import MatrixHistoryProtocol, MatrixHistoryRootAllocation

from .contracts import BetaCouplingRule, MatrixIntegratorKind, MatrixPrecision, SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView
from .history_preparation import four_family_preparation_couplings
from .model import ideal_state
from .shooting import brownian_bridge_split
from .simulation import BAOABGradientCache, baoab_step_with_hermitian_noise, hermitian_noise


def history_native_parameters():
    member = SixMatrixResponseModelFamilyMember("six-matrix-response.member.mass-0p5.cross-coupling-1", Decimal("0.5"), Decimal("0.5"), Decimal(1), BetaCouplingRule.PUBLISHED_DETERMINISTIC)
    views = tuple(SixMatrixResponseNumericalView(
        "six-matrix-response.view.baoab-dt-" + ("0p001" if multiplier == 1 else "0p0005"),
        MatrixIntegratorKind.BAOAB_UNDERDAMPED_LANGEVIN, Decimal("0.001") / multiplier,
        "dimensionless-langevin-time", Decimal(1), Decimal(1), MatrixPrecision.COMPLEX128,
        "numpy-pcg64dxsm", np.__version__, "matrix-history.explicit-numeric-input", 256,
    ) for multiplier in (1, 2))
    return member, views


def initial_history_state(root):
    decreasing = root.history_index // 64 == 1
    return ideal_state(q=2, alpha_tilde_x=8.0 if decreasing else 0.0,
                       alpha_tilde_y=8.0 if decreasing else 0.0,
                       constitution="11" if decreasing else "00")


def advance_history_pair(primary, fine, *, root, coarse_step, protocol, streams, caches, member, views):
    """One real paired step, reusable for bounded source conformance tests."""
    if type(coarse_step) is not int or not 1 <= coarse_step <= 1024 or primary.q != 2 or fine.q != 2 or primary.step_index != coarse_step - 1 or fine.step_index != 2 * (coarse_step - 1):
        raise ValueError("History paired step changes its physical state/source clock")
    stage = "preparation" if coarse_step <= protocol.ramp_ticks else "autonomous"
    noise = hermitian_noise(rng=streams[f"{stage}-coarse-driver"], q=2)
    bridge = hermitian_noise(rng=streams[f"{stage}-fine-bridge"], q=2)
    halves = brownian_bridge_split(coarse_noise=noise, bridge_noise=bridge, half_decay=float(np.exp(-.0005)))
    def advance(state, tick, multiplier, innovation):
        x, y = four_family_preparation_couplings(root.family_id, tick, ramp_steps=protocol.ramp_ticks,
            integrator_multiplier=multiplier, target_x=float(protocol.target_x), target_y=float(protocol.target_y),
            decreasing_coupling_start=float(protocol.decreasing_start))
        return baoab_step_with_hermitian_noise(state, member=member, numerical_view=views[multiplier - 1],
            next_alpha_tilde_x=x, next_alpha_tilde_y=y, standardized_noise=innovation,
            gradient_cache=caches[multiplier - 1])
    primary = advance(primary, coarse_step, 1, noise)
    fine_first = advance(fine, 2 * coarse_step - 1, 2, halves[0])
    fine = advance(fine_first, 2 * coarse_step, 2, halves[1])
    return primary, fine_first, fine


def acquire_history(root: MatrixHistoryRootAllocation, protocol: MatrixHistoryProtocol = MatrixHistoryProtocol(), *, progress=None):
    """Execute exactly one1024/2048 history; cancellation produces no completion."""
    arrays = {}
    for name, count in (("primary", 1025), ("fine", 2049)):
        for component in ("positions", "momenta"):
            arrays[f"{name}_{component}"] = np.full((count, 2, 3, 4, 4), np.nan, dtype="<c16")
        arrays[f"{name}_alpha"] = np.full((count, 2), np.nan, dtype="<f8")
        arrays[f"{name}_ticks"] = np.arange(count, dtype="<i8")
    def store(name, state):
        for component in ("positions", "momenta"):
            arrays[f"{name}_{component}"][state.step_index] = getattr(state, component)
        arrays[f"{name}_alpha"][state.step_index] = state.alpha_tilde_x, state.alpha_tilde_y
    member, views = history_native_parameters()
    caches = (BAOABGradientCache(), BAOABGradientCache())
    streams = {seed.purpose: np.random.Generator(np.random.PCG64DXSM(seed.effective_seed)) for seed in root.seeds}
    primary = fine = initial_history_state(root)
    store("primary", primary)
    store("fine", fine)
    completed_primary = completed_fine = 0
    try:
        for tick in range(1, protocol.total_ticks + 1):
            primary, fine_first, fine = advance_history_pair(primary, fine, root=root, coarse_step=tick,
                protocol=protocol, streams=streams, caches=caches, member=member, views=views)
            store("primary", primary)
            store("fine", fine_first)
            store("fine", fine)
            completed_primary, completed_fine = tick, 2 * tick
            if not primary.finite or not fine.finite:
                raise FloatingPointError("Native history became nonfinite")
            if progress is not None:
                progress(root.history_id, tick)
    except (FloatingPointError, np.linalg.LinAlgError):
        return arrays, "NUMERICAL_INVALID", completed_primary, completed_fine, "NATIVE_NUMERICAL_INVALID"
    return arrays, "COMPLETE", completed_primary, completed_fine, None


def validate_history_arrays(arrays, root, *, complete):
    expected = {f"{view}_{key}" for view in ("primary", "fine") for key in ("positions", "momenta", "alpha", "ticks")}
    if set(arrays) != expected:
        raise ValueError("History arrays omit native phase-space/clock members")
    initial = initial_history_state(root)
    for view, multiplier in (("primary", 1), ("fine", 2)):
        count = 1024 * multiplier + 1
        if not np.array_equal(arrays[f"{view}_ticks"], np.arange(count)):
            raise ValueError("History native clock differs")
        for key, shape, dtype in (("positions", (count, 2, 3, 4, 4), "<c16"), ("momenta", (count, 2, 3, 4, 4), "<c16"), ("alpha", (count, 2), "<f8")):
            value = arrays[f"{view}_{key}"]
            if value.shape != shape or value.dtype.str != dtype or complete and not np.isfinite(value).all():
                raise ValueError("History phase-space shape/precision/completeness differs")
        if not np.array_equal(arrays[f"{view}_positions"][0], initial.positions) or not np.array_equal(arrays[f"{view}_momenta"][0], initial.momenta):
            raise ValueError("History family initial condition was substituted")
        if complete:
            expected_alpha = np.array([four_family_preparation_couplings(root.family_id, i,
                ramp_steps=256, integrator_multiplier=multiplier, target_x=2/3, target_y=22/3,
                decreasing_coupling_start=8) for i in range(count)])
            if not np.array_equal(arrays[f"{view}_alpha"], expected_alpha):
                raise ValueError("History four-family ramp/autonomous coupling schedule differs")
            for component in ("positions", "momenta"):
                value = arrays[f"{view}_{component}"]
                residual = np.linalg.norm((value - value.conj().swapaxes(-1, -2)).reshape(count, -1), axis=1)
                scale = np.maximum(np.linalg.norm(value.reshape(count, -1), axis=1), 1)
                if (residual > 1e-10 * scale).any():
                    raise ValueError("History phase-space is non-Hermitian")
