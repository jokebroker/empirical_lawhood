"""Strict configuration and canonical serialization for quantum trajectory burnin screen."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence, cast

from .types import CAMPAIGN_TOKEN, PLAN_ID


def stable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def sha256_hex(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _object(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return cast(Mapping[str, object], value)


def _exact_keys(
    value: Mapping[str, object],
    expected: Sequence[str],
    label: str,
) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        unknown = sorted(set(value) - set(expected))
        raise ValueError(f"{label} fields differ: missing={missing}, unknown={unknown}")


def _decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an exact decimal string")
    result = Decimal(value)
    if not result.is_finite():
        raise ValueError(f"{label} must be finite")
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an exact integer")
    return value


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


@dataclass(frozen=True, slots=True)
class RosterSpec:
    roster_id: str
    units: int
    l_sites: int
    particles: int
    allocation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryBurninScreenConfig:
    raw: Mapping[str, object]
    schema: str
    version: str
    plan_id: str
    campaign_token: str
    l_primary: int
    n_primary: int
    l_dense: int
    n_dense: int
    j_xy: Decimal
    j_z: Decimal
    epsilon: Decimal
    gamma_strong: Decimal
    gamma_weak: Decimal
    actions: tuple[str, ...]
    action_switch_order: str
    checkpoint_clock: Decimal
    comparison_horizon: Decimal
    dense_horizon: Decimal
    ensemble_draws: int
    free_running_draws: int
    free_running_horizon: Decimal
    split_blocks: int
    burnin_candidates: tuple[Decimal, ...]
    burnin_extension: Decimal
    observation_horizon: Decimal
    maximum_clock: Decimal
    bootstrap_draws: int
    permutation_draws: int
    family_alpha: Decimal
    propagator_norm_tolerance: Decimal
    post_jump_norm_tolerance: Decimal
    projected_mass_tolerance: Decimal
    mark_probability_sum_tolerance: Decimal
    particle_number_tolerance: Decimal
    phase_infidelity_tolerance: Decimal
    site_probability_tolerance: Decimal
    event_clock_tolerance: Decimal
    hs_multiplier: Decimal
    mean_margin: Decimal
    variance_margin: Decimal
    event_rate_margin: Decimal
    energy_margin: Decimal
    memory_eta_squared: Decimal
    count_floor: Decimal
    elapsed_floor_factor: Decimal
    trig_floor: Decimal
    one_hot_floor: Decimal
    local_observable_floor: Decimal
    half_occupation_floor: Decimal
    max_workers: int
    initial_minimum_free_bytes: int
    subsequent_minimum_free_bytes: int
    maximum_json_bytes: int
    external_root: str
    roster_specs: tuple[RosterSpec, ...]
    fixture_ids: tuple[str, ...]
    allowed_evaluation_slots: tuple[str, ...]

    @property
    def config_sha256(self) -> str:
        return sha256_hex(stable_json_bytes(self.raw))


TOP_LEVEL_KEYS = (
    "actions",
    "adjudication",
    "analytic_fixtures",
    "authority",
    "burnin_ladder",
    "chart_panel",
    "comparator",
    "conditional_evaluation_template",
    "custody",
    "formal_gap_rules",
    "identity",
    "inference",
    "numerical_denominator",
    "receiver_panel",
    "rosters",
    "source",
    "state_panel",
    "thresholds",
    "typed_jump_contract",
)


def decode_config(document: Mapping[str, object]) -> QuantumTrajectoryBurninScreenConfig:
    _exact_keys(document, TOP_LEVEL_KEYS, "quantum trajectory burnin screen config")
    identity = _object(document["identity"], "identity")
    _exact_keys(identity, ("campaign_token", "plan_id", "schema", "version"), "identity")
    schema = _string(identity["schema"], "identity.schema")
    version = _string(identity["version"], "identity.version")
    plan_id = _string(identity["plan_id"], "identity.plan_id")
    campaign_token = _string(identity["campaign_token"], "identity.campaign_token")
    if (
        schema != 'empirical-lawhood/simulators/quantum-trajectory-burnin-screen/quantum-trajectory-source-qualification'
        or version != "1.0.0"
        or plan_id != PLAN_ID
        or campaign_token != CAMPAIGN_TOKEN
    ):
        raise ValueError("quantum trajectory burnin screen config identity differs from the burnin-screen plan")

    source = _object(document["source"], "source")
    _exact_keys(
        source,
        (
            "boundary_bonds_primary",
            "denominators",
            "j_xy",
            "j_z",
            "l_dense",
            "l_primary",
            "n_dense",
            "n_primary",
            "periodic",
            "unravelling",
        ),
        "source",
    )
    denoms = _object(source["denominators"], "source.denominators")
    _exact_keys(denoms, ("strong", "weak"), "source.denominators")

    actions = _object(document["actions"], "actions")
    _exact_keys(actions, ("action_switch_order", "epsilon", "ids"), "actions")
    action_ids = actions["ids"]
    if not isinstance(action_ids, list) or any(not isinstance(item, str) for item in action_ids):
        raise ValueError("actions.ids must be a string list")

    comparator = _object(document["comparator"], "comparator")
    _exact_keys(
        comparator,
        (
            "checkpoint_clock",
            "comparison_horizon",
            "dense_horizon",
            "ensemble_draws",
            "free_running_draws",
            "free_running_horizon",
            "lindblad_preparations",
            "split_blocks",
        ),
        "comparator",
    )
    burnin = _object(document["burnin_ladder"], "burnin_ladder")
    _exact_keys(
        burnin,
        ("candidates", "extension", "maximum_clock", "observation_horizon", "selection"),
        "burnin_ladder",
    )
    candidate_values = burnin["candidates"]
    if not isinstance(candidate_values, list):
        raise ValueError("burnin_ladder.candidates must be a list")
    candidates = tuple(_decimal(value, "burnin_ladder.candidates[]") for value in candidate_values)

    inference = _object(document["inference"], "inference")
    _exact_keys(
        inference,
        (
            "bootstrap_draws",
            "family_alpha",
            "permutation_draws",
            "resampling_unit",
            "studentization",
        ),
        "inference",
    )
    numerical = _object(document["numerical_denominator"], "numerical_denominator")
    _exact_keys(
        numerical,
        (
            "complex_dtype",
            "dense_propagator",
            "float_dtype",
            "max_workers",
            "native_threads_per_worker",
            "network_requested",
            "prng",
            "sparse_propagator",
        ),
        "numerical_denominator",
    )
    thresholds = _object(document["thresholds"], "thresholds")
    threshold_keys = (
        "elapsed_time_scale_floor_factor",
        "energy_margin",
        "event_clock_tolerance",
        "event_rate_relative_margin",
        "half_occupation_scale_floor",
        "hilbert_schmidt_multiplier",
        "local_observable_scale_floor",
        "mark_probability_sum_tolerance",
        "mean_standardized_margin",
        "one_hot_scale_floor",
        "particle_number_tolerance",
        "phase_infidelity_tolerance",
        "post_jump_norm_tolerance",
        "preparation_memory_eta_squared",
        "projected_mass_tolerance",
        "propagator_norm_tolerance",
        "site_probability_tolerance",
        "trigonometric_scale_floor",
        "variance_standardized_margin",
    )
    _exact_keys(thresholds, threshold_keys, "thresholds")
    custody = _object(document["custody"], "custody")
    _exact_keys(
        custody,
        (
            "external_root",
            "initial_minimum_free_bytes",
            "maximum_json_bytes",
            "subsequent_minimum_free_bytes",
        ),
        "custody",
    )
    rosters = _object(document["rosters"], "rosters")
    expected_rosters = (
        "path-conformance-primary",
        "path-conformance-dense",
        "ensemble-conformance",
        "preparation-development",
        "preparation-evaluation",
        "preparation-reserve",
    )
    _exact_keys(rosters, expected_rosters, "rosters")
    roster_specs: list[RosterSpec] = []
    for roster_id in expected_rosters:
        item = _object(rosters[roster_id], f"rosters.{roster_id}")
        _exact_keys(item, ("allocation", "l_sites", "particles", "units"), roster_id)
        allocation = item["allocation"]
        if not isinstance(allocation, list):
            raise ValueError(f"{roster_id}.allocation must be a list")
        allocation_tuple = tuple(
            _integer(value, f"{roster_id}.allocation[]") for value in allocation
        )
        spec = RosterSpec(
            roster_id=roster_id,
            units=_integer(item["units"], f"{roster_id}.units"),
            l_sites=_integer(item["l_sites"], f"{roster_id}.l_sites"),
            particles=_integer(item["particles"], f"{roster_id}.particles"),
            allocation=allocation_tuple,
        )
        if len(spec.allocation) != spec.particles + 1 or sum(spec.allocation) != spec.units:
            raise ValueError(f"{roster_id} allocation is inconsistent")
        roster_specs.append(spec)

    fixtures = _object(document["analytic_fixtures"], "analytic_fixtures")
    _exact_keys(fixtures, ("ids",), "analytic_fixtures")
    fixture_ids = fixtures["ids"]
    if not isinstance(fixture_ids, list) or any(
        not isinstance(value, str) for value in fixture_ids
    ):
        raise ValueError("analytic_fixtures.ids must be a string list")

    template = _object(document["conditional_evaluation_template"], "template")
    _exact_keys(
        template,
        ("allowed_development_slots", "evaluation_roster", "reserve_roster"),
        "conditional_evaluation_template",
    )
    allowed_slots = template["allowed_development_slots"]
    if not isinstance(allowed_slots, list) or any(
        not isinstance(value, str) for value in allowed_slots
    ):
        raise ValueError("allowed_development_slots must be a string list")

    result = QuantumTrajectoryBurninScreenConfig(
        raw=document,
        schema=schema,
        version=version,
        plan_id=plan_id,
        campaign_token=campaign_token,
        l_primary=_integer(source["l_primary"], "source.l_primary"),
        n_primary=_integer(source["n_primary"], "source.n_primary"),
        l_dense=_integer(source["l_dense"], "source.l_dense"),
        n_dense=_integer(source["n_dense"], "source.n_dense"),
        j_xy=_decimal(source["j_xy"], "source.j_xy"),
        j_z=_decimal(source["j_z"], "source.j_z"),
        epsilon=_decimal(actions["epsilon"], "actions.epsilon"),
        gamma_strong=_decimal(denoms["strong"], "source.denominators.strong"),
        gamma_weak=_decimal(denoms["weak"], "source.denominators.weak"),
        actions=tuple(cast(list[str], action_ids)),
        action_switch_order=_string(actions["action_switch_order"], "actions.action_switch_order"),
        checkpoint_clock=_decimal(comparator["checkpoint_clock"], "checkpoint_clock"),
        comparison_horizon=_decimal(comparator["comparison_horizon"], "comparison_horizon"),
        dense_horizon=_decimal(comparator["dense_horizon"], "dense_horizon"),
        ensemble_draws=_integer(comparator["ensemble_draws"], "ensemble_draws"),
        free_running_draws=_integer(comparator["free_running_draws"], "free_running_draws"),
        free_running_horizon=_decimal(comparator["free_running_horizon"], "free_running_horizon"),
        split_blocks=_integer(comparator["split_blocks"], "split_blocks"),
        burnin_candidates=candidates,
        burnin_extension=_decimal(burnin["extension"], "burnin.extension"),
        observation_horizon=_decimal(burnin["observation_horizon"], "burnin.observation_horizon"),
        maximum_clock=_decimal(burnin["maximum_clock"], "burnin.maximum_clock"),
        bootstrap_draws=_integer(inference["bootstrap_draws"], "bootstrap_draws"),
        permutation_draws=_integer(inference["permutation_draws"], "permutation_draws"),
        family_alpha=_decimal(inference["family_alpha"], "family_alpha"),
        propagator_norm_tolerance=_decimal(
            thresholds["propagator_norm_tolerance"], "propagator_norm_tolerance"
        ),
        post_jump_norm_tolerance=_decimal(
            thresholds["post_jump_norm_tolerance"], "post_jump_norm_tolerance"
        ),
        projected_mass_tolerance=_decimal(
            thresholds["projected_mass_tolerance"], "projected_mass_tolerance"
        ),
        mark_probability_sum_tolerance=_decimal(
            thresholds["mark_probability_sum_tolerance"],
            "mark_probability_sum_tolerance",
        ),
        particle_number_tolerance=_decimal(
            thresholds["particle_number_tolerance"], "particle_number_tolerance"
        ),
        phase_infidelity_tolerance=_decimal(
            thresholds["phase_infidelity_tolerance"], "phase_infidelity_tolerance"
        ),
        site_probability_tolerance=_decimal(
            thresholds["site_probability_tolerance"], "site_probability_tolerance"
        ),
        event_clock_tolerance=_decimal(
            thresholds["event_clock_tolerance"], "event_clock_tolerance"
        ),
        hs_multiplier=_decimal(
            thresholds["hilbert_schmidt_multiplier"], "hilbert_schmidt_multiplier"
        ),
        mean_margin=_decimal(thresholds["mean_standardized_margin"], "mean_standardized_margin"),
        variance_margin=_decimal(
            thresholds["variance_standardized_margin"], "variance_standardized_margin"
        ),
        event_rate_margin=_decimal(
            thresholds["event_rate_relative_margin"], "event_rate_relative_margin"
        ),
        energy_margin=_decimal(thresholds["energy_margin"], "energy_margin"),
        memory_eta_squared=_decimal(
            thresholds["preparation_memory_eta_squared"],
            "preparation_memory_eta_squared",
        ),
        count_floor=Decimal("1"),
        elapsed_floor_factor=_decimal(
            thresholds["elapsed_time_scale_floor_factor"],
            "elapsed_time_scale_floor_factor",
        ),
        trig_floor=_decimal(thresholds["trigonometric_scale_floor"], "trigonometric_scale_floor"),
        one_hot_floor=_decimal(thresholds["one_hot_scale_floor"], "one_hot_scale_floor"),
        local_observable_floor=_decimal(
            thresholds["local_observable_scale_floor"],
            "local_observable_scale_floor",
        ),
        half_occupation_floor=_decimal(
            thresholds["half_occupation_scale_floor"],
            "half_occupation_scale_floor",
        ),
        max_workers=_integer(numerical["max_workers"], "max_workers"),
        initial_minimum_free_bytes=_integer(
            custody["initial_minimum_free_bytes"], "initial_minimum_free_bytes"
        ),
        subsequent_minimum_free_bytes=_integer(
            custody["subsequent_minimum_free_bytes"], "subsequent_minimum_free_bytes"
        ),
        maximum_json_bytes=_integer(custody["maximum_json_bytes"], "maximum_json_bytes"),
        external_root=_string(custody["external_root"], "custody.external_root"),
        roster_specs=tuple(roster_specs),
        fixture_ids=tuple(cast(list[str], fixture_ids)),
        allowed_evaluation_slots=tuple(cast(list[str], allowed_slots)),
    )
    _validate_frozen_values(result)
    return result


def _validate_frozen_values(config: QuantumTrajectoryBurninScreenConfig) -> None:
    if (
        (config.l_primary, config.n_primary) != (12, 6)
        or (config.l_dense, config.n_dense) != (8, 4)
        or config.j_xy != Decimal("1.0")
        or config.j_z != Decimal("1.0")
        or config.epsilon != Decimal("0.20")
        or config.gamma_strong != Decimal("1.0")
        or config.gamma_weak != Decimal("0.1")
        or config.actions != ("hold", "minus", "plus")
        or config.burnin_candidates
        != (Decimal("200"), Decimal("400"), Decimal("800"), Decimal("1600"))
        or config.burnin_extension != Decimal("100")
        or config.observation_horizon != Decimal("20")
        or config.maximum_clock != Decimal("3300")
        or config.bootstrap_draws != 8192
        or config.permutation_draws != 8192
        or config.ensemble_draws != 4096
        or config.free_running_draws != 512
    ):
        raise ValueError("quantum trajectory burnin screen config differs from the declared burnin-screen plan")


def load_config(path: Path) -> tuple[QuantumTrajectoryBurninScreenConfig, bytes]:
    payload = path.read_bytes()
    document = json.loads(payload)
    if not isinstance(document, Mapping):
        raise ValueError("quantum trajectory burnin screen config root must be an object")
    config = decode_config(cast(Mapping[str, object], document))
    canonical = stable_json_bytes(document)
    if canonical != payload:
        raise ValueError("quantum trajectory burnin screen repository config must use canonical JSON bytes")
    return config, payload


__all__ = [
    'QuantumTrajectoryBurninScreenConfig',
    "RosterSpec",
    "decode_config",
    "load_config",
    "sha256_hex",
    "stable_json_bytes",
]
