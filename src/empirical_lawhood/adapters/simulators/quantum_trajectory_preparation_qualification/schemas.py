"""Strict configuration and canonical serialization for quantum trajectory preparation qualification."""

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
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{label} must be a string list")
    return tuple(cast(list[str], value))


def _integer_list(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be an integer list")
    return tuple(_integer(item, f"{label}[]") for item in value)


@dataclass(frozen=True, slots=True)
class RosterSpec:
    roster_id: str
    units: int
    l_sites: int
    particles: int
    allocation: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class IntegratorSpec:
    rtol: Decimal
    atol: Decimal
    max_step: Decimal


@dataclass(frozen=True, slots=True)
class QuantumTrajectoryPreparationQualificationConfig:
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
    free_running_draws: int
    free_running_horizon: Decimal
    dense_horizon: Decimal
    ensemble_ladder: tuple[int, ...]
    ensemble_draws: int
    split_blocks: int
    ensemble_conformance_family_cells: int
    ensemble_conformance_family_looks: int
    ensemble_conformance_precision: Decimal
    ensemble_conformance_density_tolerance: Decimal
    burnin_candidates: tuple[Decimal, ...]
    burnin_extension: Decimal
    observation_horizon: Decimal
    maximum_clock: Decimal
    bootstrap_draws: int
    permutation_draws: int
    family_alpha: Decimal
    statistical_conformance_null_replicates: int
    statistical_conformance_planted_replicates: int
    statistical_conformance_bootstrap_panels: int
    statistical_conformance_memory_panels: int
    statistical_conformance_minimum_adverse_panels: int
    statistical_conformance_null_coverage: Decimal
    statistical_conformance_coverage_lower_bound: Decimal
    statistical_conformance_power_lower_bound: Decimal
    statistical_conformance_density_null_pass_rate: Decimal
    statistical_conformance_planted_shift: Decimal
    reference_tight: IntegratorSpec
    reference_tighter: IntegratorSpec
    propagator_norm_tolerance: Decimal
    post_jump_norm_tolerance: Decimal
    projected_mass_tolerance: Decimal
    mark_probability_sum_tolerance: Decimal
    particle_number_tolerance: Decimal
    phase_infidelity_tolerance: Decimal
    site_probability_tolerance: Decimal
    event_clock_tolerance: Decimal
    reference_trace_tolerance: Decimal
    reference_hermiticity_tolerance: Decimal
    reference_minimum_eigenvalue: Decimal
    reference_route_hs_tolerance: Decimal
    reference_self_convergence_tolerance: Decimal
    reference_observable_normalized_tolerance: Decimal
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
    replication_ladder: tuple[int, ...]
    cumulative_allocations: Mapping[int, tuple[int, ...]]
    predecessor: Mapping[str, object]
    fixture_ids: tuple[str, ...]
    allowed_evaluation_slots: tuple[str, ...]

    @property
    def config_sha256(self) -> str:
        return sha256_hex(stable_json_bytes(self.raw))


TOP_LEVEL_KEYS = (
    "actions",
    "adjudication",
    "authority",
    "burnin_ladder",
    "jump_contract_fixtures",
    "ensemble_conformance_ladder",
    "c1p_comparator",
    "chart_panel",
    "conditional_evaluation_template",
    "custody",
    "deterministic_reference",
    "formal_gap_rules",
    "identity",
    "inference",
    "numerical_denominator",
    "observable_support",
    "predecessor",
    "receiver_panel",
    "replication_grid",
    "rosters",
    "statistical_conformance_calibration",
    "source",
    "state_panel",
    "thresholds",
    "typed_jump_contract",
)


def _decode_rosters(value: object) -> tuple[RosterSpec, ...]:
    rosters = _object(value, "rosters")
    expected = ("preparation-development", "preparation-evaluation", "preparation-reserve")
    _exact_keys(rosters, expected, "rosters")
    result: list[RosterSpec] = []
    for roster_id in expected:
        row = _object(rosters[roster_id], f"rosters.{roster_id}")
        _exact_keys(row, ("allocation", "l_sites", "particles", "units"), roster_id)
        spec = RosterSpec(
            roster_id=roster_id,
            units=_integer(row["units"], f"{roster_id}.units"),
            l_sites=_integer(row["l_sites"], f"{roster_id}.l_sites"),
            particles=_integer(row["particles"], f"{roster_id}.particles"),
            allocation=_integer_list(row["allocation"], f"{roster_id}.allocation"),
        )
        if len(spec.allocation) != spec.particles + 1:
            raise ValueError(f"{roster_id} allocation length is inconsistent")
        if sum(spec.allocation) != spec.units:
            raise ValueError(f"{roster_id} allocation count is inconsistent")
        result.append(spec)
    return tuple(result)


def _decode_integrator(value: object, label: str) -> IntegratorSpec:
    row = _object(value, label)
    _exact_keys(row, ("atol", "max_step", "rtol"), label)
    return IntegratorSpec(
        rtol=_decimal(row["rtol"], f"{label}.rtol"),
        atol=_decimal(row["atol"], f"{label}.atol"),
        max_step=_decimal(row["max_step"], f"{label}.max_step"),
    )


def decode_config(document: Mapping[str, object]) -> QuantumTrajectoryPreparationQualificationConfig:
    _exact_keys(document, TOP_LEVEL_KEYS, "quantum trajectory preparation qualification config")

    identity = _object(document["identity"], "identity")
    _exact_keys(identity, ("campaign_token", "plan_id", "schema", "version"), "identity")
    schema = _string(identity["schema"], "identity.schema")
    version = _string(identity["version"], "identity.version")
    plan_id = _string(identity["plan_id"], "identity.plan_id")
    campaign_token = _string(identity["campaign_token"], "identity.campaign_token")
    if (
        schema != 'empirical-lawhood/simulators/quantum-trajectory-preparation-qualification/quantum-trajectory-source-qualification'
        or version != "1.0.0"
        or plan_id != PLAN_ID
        or campaign_token != CAMPAIGN_TOKEN
    ):
        raise ValueError("quantum trajectory preparation qualification config identity differs from the declared source plan")

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
    denominators = _object(source["denominators"], "source.denominators")
    _exact_keys(denominators, ("strong", "weak"), "source.denominators")

    actions = _object(document["actions"], "actions")
    _exact_keys(actions, ("action_switch_order", "epsilon", "ids"), "actions")

    path_conformance = _object(document["c1p_comparator"], "c1p_comparator")
    _exact_keys(
        path_conformance,
        (
            "checkpoint_clock",
            "comparison_horizon",
            "free_running_draws",
            "free_running_horizon",
        ),
        "c1p_comparator",
    )
    ensemble_conformance = _object(document["ensemble_conformance_ladder"], "ensemble_conformance_ladder")
    _exact_keys(
        ensemble_conformance,
        (
            "density_diagnostic_blocks",
            "density_hilbert_schmidt_tolerance",
            "family_cells",
            "family_looks",
            "horizon",
            "interval",
            "normalized_precision",
            "rungs",
        ),
        "ensemble_conformance_ladder",
    )
    ladder = _integer_list(ensemble_conformance["rungs"], "ensemble_conformance_ladder.rungs")

    burnin = _object(document["burnin_ladder"], "burnin_ladder")
    _exact_keys(
        burnin,
        ("candidates", "extension", "maximum_clock", "observation_horizon", "selection"),
        "burnin_ladder",
    )
    candidates_raw = burnin["candidates"]
    if not isinstance(candidates_raw, list):
        raise ValueError("burnin_ladder.candidates must be a list")
    candidates = tuple(_decimal(item, "burnin_ladder.candidates[]") for item in candidates_raw)

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

    statistical_conformance = _object(document["statistical_conformance_calibration"], "statistical_conformance_calibration")
    _exact_keys(
        statistical_conformance,
        (
            "bootstrap_panels",
            "coverage_lower_bound",
            "density_null_pass_rate",
            "false_qualification_upper_bound",
            "fixture_families",
            "memory_panels",
            "minimum_adverse_panels",
            "null_coverage",
            "null_replicates",
            "planted_replicates",
            "planted_shift",
            "power_lower_bound",
            "scalar_vector_tolerance",
        ),
        "statistical_conformance_calibration",
    )

    reference = _object(document["deterministic_reference"], "deterministic_reference")
    _exact_keys(
        reference,
        ("dense_horizon", "dop853_tight", "dop853_tighter", "routes"),
        "deterministic_reference",
    )
    routes = _string_list(reference["routes"], "deterministic_reference.routes")

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
    _exact_keys(
        thresholds,
        (
            "elapsed_time_scale_floor_factor",
            "energy_margin",
            "event_clock_tolerance",
            "event_rate_relative_margin",
            "half_occupation_scale_floor",
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
            "reference_hermiticity_tolerance",
            "reference_minimum_eigenvalue",
            "reference_observable_normalized_tolerance",
            "reference_route_hilbert_schmidt_tolerance",
            "reference_self_convergence_tolerance",
            "reference_trace_tolerance",
            "site_probability_tolerance",
            "trigonometric_scale_floor",
            "variance_standardized_margin",
        ),
        "thresholds",
    )

    fixtures = _object(document["jump_contract_fixtures"], "jump_contract_fixtures")
    _exact_keys(fixtures, ("ids",), "jump_contract_fixtures")
    template = _object(document["conditional_evaluation_template"], "template")
    _exact_keys(
        template,
        ("allowed_development_slots", "evaluation_roster", "reserve_roster"),
        "conditional_evaluation_template",
    )
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

    _validate_static_sections(document)

    replication = _object(document["replication_grid"], "replication_grid")
    _exact_keys(
        replication,
        (
            "candidates",
            "cumulative_allocations",
            "ladder",
            "looks",
            "maximum_endpoint",
            "selection",
        ),
        "replication_grid",
    )
    replication_ladder = _integer_list(replication["ladder"], "replication_grid.ladder")
    allocations_raw = _object(
        replication["cumulative_allocations"],
        "replication_grid.cumulative_allocations",
    )
    _exact_keys(
        allocations_raw,
        tuple(str(value) for value in replication_ladder),
        "replication_grid.cumulative_allocations",
    )
    cumulative_allocations = {
        value: _integer_list(
            allocations_raw[str(value)],
            f"replication_grid.cumulative_allocations.{value}",
        )
        for value in replication_ladder
    }
    predecessor = _object(document["predecessor"], "predecessor")
    _exact_keys(
        predecessor,
        (
            "ensemble_conformance_result_sha256",
            "path_conformance_result_sha256",
            "closeout_receipt_sha256",
            "closeout_sha256",
            "config_sha256",
            "external_root",
            "freeze_sha256",
            "implementation_sha256",
            "plan_sha256",
            "deterministic_reference_result_sha256",
            "roster_sha256",
        ),
        "predecessor",
    )

    config = QuantumTrajectoryPreparationQualificationConfig(
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
        gamma_strong=_decimal(denominators["strong"], "source.denominators.strong"),
        gamma_weak=_decimal(denominators["weak"], "source.denominators.weak"),
        actions=_string_list(actions["ids"], "actions.ids"),
        action_switch_order=_string(actions["action_switch_order"], "actions.action_switch_order"),
        checkpoint_clock=_decimal(path_conformance["checkpoint_clock"], "c1p.checkpoint_clock"),
        comparison_horizon=_decimal(path_conformance["comparison_horizon"], "c1p.comparison_horizon"),
        free_running_draws=_integer(path_conformance["free_running_draws"], "c1p.free_running_draws"),
        free_running_horizon=_decimal(path_conformance["free_running_horizon"], "c1p.free_running_horizon"),
        dense_horizon=_decimal(reference["dense_horizon"], "reference.dense_horizon"),
        ensemble_ladder=ladder,
        ensemble_draws=ladder[-1],
        split_blocks=_integer(ensemble_conformance["density_diagnostic_blocks"], "ensemble_conformance.density_diagnostic_blocks"),
        ensemble_conformance_family_cells=_integer(ensemble_conformance["family_cells"], "ensemble_conformance.family_cells"),
        ensemble_conformance_family_looks=_integer(ensemble_conformance["family_looks"], "ensemble_conformance.family_looks"),
        ensemble_conformance_precision=_decimal(ensemble_conformance["normalized_precision"], "ensemble_conformance.normalized_precision"),
        ensemble_conformance_density_tolerance=_decimal(
            ensemble_conformance["density_hilbert_schmidt_tolerance"], "ensemble_conformance.density_hs_tolerance"
        ),
        burnin_candidates=candidates,
        burnin_extension=_decimal(burnin["extension"], "burnin.extension"),
        observation_horizon=_decimal(burnin["observation_horizon"], "burnin.observation_horizon"),
        maximum_clock=_decimal(burnin["maximum_clock"], "burnin.maximum_clock"),
        bootstrap_draws=_integer(inference["bootstrap_draws"], "inference.bootstrap_draws"),
        permutation_draws=_integer(inference["permutation_draws"], "inference.permutation_draws"),
        family_alpha=_decimal(inference["family_alpha"], "inference.family_alpha"),
        statistical_conformance_null_replicates=_integer(statistical_conformance["null_replicates"], "statistical_conformance.null_replicates"),
        statistical_conformance_planted_replicates=_integer(statistical_conformance["planted_replicates"], "statistical_conformance.planted_replicates"),
        statistical_conformance_bootstrap_panels=_integer(statistical_conformance["bootstrap_panels"], "statistical_conformance.bootstrap_panels"),
        statistical_conformance_memory_panels=_integer(statistical_conformance["memory_panels"], "statistical_conformance.memory_panels"),
        statistical_conformance_minimum_adverse_panels=_integer(
            statistical_conformance["minimum_adverse_panels"], "statistical_conformance.minimum_adverse_panels"
        ),
        statistical_conformance_null_coverage=_decimal(statistical_conformance["null_coverage"], "statistical_conformance.null_coverage"),
        statistical_conformance_coverage_lower_bound=_decimal(statistical_conformance["coverage_lower_bound"], "statistical_conformance.coverage_lower_bound"),
        statistical_conformance_power_lower_bound=_decimal(statistical_conformance["power_lower_bound"], "statistical_conformance.power_lower_bound"),
        statistical_conformance_density_null_pass_rate=_decimal(
            statistical_conformance["density_null_pass_rate"], "statistical_conformance.density_null_pass_rate"
        ),
        statistical_conformance_planted_shift=_decimal(statistical_conformance["planted_shift"], "statistical_conformance.planted_shift"),
        reference_tight=_decode_integrator(reference["dop853_tight"], "dop853_tight"),
        reference_tighter=_decode_integrator(reference["dop853_tighter"], "dop853_tighter"),
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
        reference_trace_tolerance=_decimal(
            thresholds["reference_trace_tolerance"], "reference_trace_tolerance"
        ),
        reference_hermiticity_tolerance=_decimal(
            thresholds["reference_hermiticity_tolerance"],
            "reference_hermiticity_tolerance",
        ),
        reference_minimum_eigenvalue=_decimal(
            thresholds["reference_minimum_eigenvalue"], "reference_minimum_eigenvalue"
        ),
        reference_route_hs_tolerance=_decimal(
            thresholds["reference_route_hilbert_schmidt_tolerance"],
            "reference_route_hilbert_schmidt_tolerance",
        ),
        reference_self_convergence_tolerance=_decimal(
            thresholds["reference_self_convergence_tolerance"],
            "reference_self_convergence_tolerance",
        ),
        reference_observable_normalized_tolerance=_decimal(
            thresholds["reference_observable_normalized_tolerance"],
            "reference_observable_normalized_tolerance",
        ),
        mean_margin=_decimal(thresholds["mean_standardized_margin"], "mean_margin"),
        variance_margin=_decimal(thresholds["variance_standardized_margin"], "variance_margin"),
        event_rate_margin=_decimal(thresholds["event_rate_relative_margin"], "event_rate_margin"),
        energy_margin=_decimal(thresholds["energy_margin"], "energy_margin"),
        memory_eta_squared=_decimal(
            thresholds["preparation_memory_eta_squared"], "memory_eta_squared"
        ),
        count_floor=Decimal("1"),
        elapsed_floor_factor=_decimal(
            thresholds["elapsed_time_scale_floor_factor"], "elapsed_floor_factor"
        ),
        trig_floor=_decimal(thresholds["trigonometric_scale_floor"], "trigonometric_scale_floor"),
        one_hot_floor=_decimal(thresholds["one_hot_scale_floor"], "one_hot_scale_floor"),
        local_observable_floor=_decimal(
            thresholds["local_observable_scale_floor"], "local_observable_scale_floor"
        ),
        half_occupation_floor=_decimal(
            thresholds["half_occupation_scale_floor"], "half_occupation_scale_floor"
        ),
        max_workers=_integer(numerical["max_workers"], "numerical.max_workers"),
        initial_minimum_free_bytes=_integer(
            custody["initial_minimum_free_bytes"], "custody.initial_minimum_free_bytes"
        ),
        subsequent_minimum_free_bytes=_integer(
            custody["subsequent_minimum_free_bytes"],
            "custody.subsequent_minimum_free_bytes",
        ),
        maximum_json_bytes=_integer(custody["maximum_json_bytes"], "custody.maximum_json_bytes"),
        external_root=_string(custody["external_root"], "custody.external_root"),
        roster_specs=_decode_rosters(document["rosters"]),
        replication_ladder=replication_ladder,
        cumulative_allocations=cumulative_allocations,
        predecessor=predecessor,
        fixture_ids=_string_list(fixtures["ids"], "jump_contract_fixtures.ids"),
        allowed_evaluation_slots=_string_list(
            template["allowed_development_slots"], "template.allowed_development_slots"
        ),
    )
    _validate_frozen_values(config, routes)
    return config


def _validate_static_sections(document: Mapping[str, object]) -> None:
    expected: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "adjudication",
            ("evidence_ceiling", "primary_denominator", "requires_noncompensating_intersection"),
        ),
        (
            "authority",
            (
                "accountable_owner",
                "authorization_basis",
                "evaluation_outcome_access",
                "implementation",
                "reveal",
                "source_execution",
            ),
        ),
        (
            "chart_panel",
            ("boundary_sites_primary", "id", "long_expected_events", "short_expected_events"),
        ),
        (
            "formal_gap_rules",
            ("eligible_direct_roles", "order_relation_through_controller_use_direct_evidence"),
        ),
        (
            "observable_support",
            ("energy", "fourier", "occupation", "pair_correlation"),
        ),
        (
            "receiver_panel",
            ("burst_gap_rule", "coordinates", "id", "observation_horizon"),
        ),
        ("state_panel", ("coordinates_excluding_energy", "id")),
        (
            "typed_jump_contract",
            ("mark_probability", "post_jump_state", "projected_mass", "zero_mass_status"),
        ),
    )
    for section, keys in expected:
        _exact_keys(_object(document[section], section), keys, section)


def _validate_frozen_values(config: QuantumTrajectoryPreparationQualificationConfig, routes: tuple[str, ...]) -> None:
    if (
        (config.l_primary, config.n_primary) != (12, 6)
        or (config.l_dense, config.n_dense) != (8, 4)
        or config.j_xy != Decimal("1.0")
        or config.j_z != Decimal("1.0")
        or config.epsilon != Decimal("0.20")
        or config.gamma_strong != Decimal("1.0")
        or config.gamma_weak != Decimal("0.1")
        or config.actions != ("hold", "minus", "plus")
        or config.ensemble_ladder != (4096, 8192, 16384, 32768)
        or config.ensemble_conformance_family_cells != 152
        or config.ensemble_conformance_family_looks != 4
        or config.ensemble_conformance_precision != Decimal("0.01")
        or config.ensemble_conformance_density_tolerance != Decimal("0.01")
        or config.burnin_candidates
        != (Decimal("200"), Decimal("400"), Decimal("800"), Decimal("1600"))
        or config.burnin_extension != Decimal("100")
        or config.maximum_clock != Decimal("3300")
        or config.bootstrap_draws != 16384
        or config.permutation_draws != 16384
        or config.free_running_draws != 512
        or config.statistical_conformance_null_replicates != 2000
        or config.statistical_conformance_planted_replicates != 2000
        or routes != ("sparse-kron-expm-multiply", "direct-matrix-dop853")
        or config.external_root != "runs/quantum-trajectory-preparation-qualification"
        or config.replication_ladder != (512, 1024, 2048)
        or config.cumulative_allocations
        != {
            512: (16, 32, 112, 192, 112, 32, 16),
            1024: (24, 48, 232, 416, 232, 48, 24),
            2048: (32, 80, 480, 864, 480, 80, 32),
        }
    ):
        raise ValueError("quantum trajectory preparation qualification config differs from the declared preparation-qualification plan")


def load_config(path: Path) -> tuple[QuantumTrajectoryPreparationQualificationConfig, bytes]:
    payload = path.read_bytes()
    document = json.loads(payload)
    if not isinstance(document, Mapping):
        raise ValueError("quantum trajectory preparation qualification config root must be an object")
    config = decode_config(cast(Mapping[str, object], document))
    if stable_json_bytes(document) != payload:
        raise ValueError("quantum trajectory preparation qualification repository config must use canonical JSON bytes")
    return config, payload


__all__ = [
    "IntegratorSpec",
    'QuantumTrajectoryPreparationQualificationConfig',
    "RosterSpec",
    "decode_config",
    "load_config",
    "sha256_hex",
    "stable_json_bytes",
]
