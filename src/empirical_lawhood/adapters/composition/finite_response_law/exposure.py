"""Exact new native stream reservations, without native draws or source access."""

from empirical_lawhood.adapters.methods.finite_response_law.science import passive_probe_seed_for, seed_for
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationConfig
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import FiniteResponseLawNativeConfig
from empirical_lawhood.adapters.simulators.finite_response_law.fresh_contracts import FiniteResponseLawCalibrationRoot
from empirical_lawhood.adapters.simulators.finite_response_law.randomness import SUBSTREAMS, native_rng
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import probe_roster_seed_sha256


def native_seed_ids(source: FiniteResponseLawNativeConfig) -> tuple[str, ...]:
    seeds = set()
    if source.stage == "prospective-evaluation":
        bootstrap_unit = (
            source.cohort_namespace
            if type(source) is FiniteResponseLawAssignedEvaluationConfig
            else "prospective-evaluation"
        )
        bootstrap_seed = next((seed for p, i, seed in getattr(source, "scientific_seeds", ()) if p == "bootstrap" and i == -1), None)
        seeds.add(f"seed.pcg64.{seed_for('bootstrap', bootstrap_unit, committed_seed=bootstrap_seed):032x}")
    for root in source.roots:
        root_seeds = dict(getattr(root, "scientific_seeds", ()))
        key = (
            root.stage_unit
            if isinstance(root, FiniteResponseLawCalibrationRoot)
            else "canary.r000"
            if source.stage == "native-canary"
            else f"supplemental-development.r{root.index:03d}"
        )
        if isinstance(root, FiniteResponseLawCalibrationRoot):
            for purpose in ("parent-allocation", f"{source.stage}-request"):
                seeds.add(f"seed.pcg64.{seed_for(purpose, key, committed_seed=root_seeds.get(purpose)):032x}")
        purposes = (
            ("prefix", "parent", "future-1", "future-2")
            if source.stage in ("native-canary", "calibration", "prospective-evaluation")
            else ("future-2",)
        )
        for purpose in purposes:
            for substream in SUBSTREAMS if purpose == "prefix" else SUBSTREAMS[:2]:
                record = native_rng(key, purpose, substream, committed_seed=root_seeds.get(purpose))
                seeds.add(f"seed.pcg64.{int(record.committed_seed_decimal):032x}")
                seeds.add(f"seed-state.pcg64.{record.initial_state_sha256}")
                if substream == "passive-probes":
                    probe = probe_roster_seed_sha256(
                        config_fingerprint=record.initial_state_sha256,
                        rule_id="finite-response-law.passive-probes",
                        scientific_seed=passive_probe_seed_for(key, committed_seed=root_seeds.get("passive-probes")),
                    )
                    seeds.add(f"seed.pcg64dxsm.{probe[:32]}")
    return tuple(sorted(seeds))


def native_seed_collisions(
    proposed: tuple[str, ...], exposed: tuple[str, ...]
) -> set[str]:
    """Reject exact stream IDs and raw seed operands across RNG labels."""
    prior = set(exposed)
    raw = {seed.rsplit(".", 1)[-1] for seed in prior if seed.startswith("seed.")}
    return {
        seed
        for seed in proposed
        if seed in prior
        or seed.startswith("seed.") and seed.rsplit(".", 1)[-1] in raw
    }
