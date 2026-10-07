"""Frozen uniform electron gas transverse receiver screen contracts and strict config decoding.

The records in this module keep source observations, method estimates,
privileged truth and adjudication separate.  They contain no filesystem or
publication behavior.
"""

from __future__ import annotations

from empirical_lawhood._required_inputs import required_external_path

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
from pathlib import Path
from typing import ClassVar, Final, Mapping, Sequence

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


PLAN_ID: Final = "uniform-electron-gas-transverse-receiver-screen"
CAMPAIGN_ID: Final = "uniform-electron-gas-transverse-screen-conformance-closeout"
CONFIG_SCHEMA: Final = 'empirical-lawhood/simulators/uniform-electron-gas-transverse-screen/transverse-receiver-screen-config'
CONFIG_VERSION: Final = "1.0.0"
CAPABILITY_VERSION: Final = "1.0.0"
EXTERNAL_ROOT: Final = "runs/uniform-electron-gas-transverse-receiver-screen"
MAXIMUM_CONFIG_BYTES: Final = 1024 * 1024
MAXIMUM_PANEL_BYTES: Final = 4 * 1024 * 1024

SOURCE_CAPABILITY_KEY: Final = "open-sim.uniform-electron-gas-response-truth-source.transverse-screen"
METHOD_CAPABILITY_KEY: Final = "method.uniform-electron-gas-response-transverse-law.transverse-screen"
ADMISSION_CAPABILITY_KEY: Final = "admission.uniform-electron-gas-response-transverse-screen.transverse-screen"
EVALUATOR_CAPABILITY_KEY: Final = "evaluator.uniform-electron-gas-response-transverse-screen.transverse-screen"
ADJUDICATION_SCHEMA: Final = 'empirical-lawhood/simulators/uniform-electron-gas-transverse-screen/scientific-adjudication'


class EvidenceLane(StrEnum):
    GAUGE_CLOSED_TRANSVERSE = "GAUGE_CLOSED_TRANSVERSE"
    TC_CLOSURE_DIAGNOSTIC = "TC_CLOSURE_DIAGNOSTIC"


class TruthCase(StrEnum):
    MISSING_PRESERVATION = "missing-preservation"
    NONLINEAR = "nonlinear"
    NORMAL = "normal"
    POSITIVE = "positive"
    Q_DRIFT = "q-drift"
    SPURIOUS_NORMAL = "spurious-normal"
    VIEW_DISAGREEMENT = "view-disagreement"
    WARD_FAIL = "ward-fail"
    WRONG_SIGN = "wrong-sign"


class UniformElectronGasMeasurementClass(StrEnum):
    TYPED_TRANSVERSE_PANEL = "MEASUREMENT_TYPED_TRANSVERSE_PANEL"
    CLOSURE_ONLY = "MEASUREMENT_CLOSURE_ONLY"
    PARTIAL_ROLE_LIMITED = "MEASUREMENT_PARTIAL_ROLE_LIMITED"
    SOURCE_IDENTITY_STOP = "MEASUREMENT_SOURCE_IDENTITY_STOP"
    UNEVALUABLE = "MEASUREMENT_UNEVALUABLE"


class UniformElectronGasObservationOrderClass(StrEnum):
    ORDER_OPPORTUNITY_SUPPORTED = "ORDER_RELATION_OPPORTUNITY_SUPPORTED"
    PHENOMENOLOGICAL_CLOSURE_ONLY = "ORDER_RELATION_PHENOMENOLOGICAL_CLOSURE_ONLY"
    NO_ORDER_OPPORTUNITY = "ORDER_RELATION_NO_ORDER_OPPORTUNITY"
    VIEW_OPPOSED = "ORDER_RELATION_VIEW_OPPOSED"
    ORDER_OPERAND_REQUIRED = "ORDER_RELATION_OPERAND_REQUIRED"


class UniformElectronGasResponseClass(StrEnum):
    FINITE_TRANSVERSE_RESPONSE = "RESPONSE_FINITE_TRANSVERSE_RESPONSE"
    RESPONSE_BELOW_FLOOR = "RESPONSE_BELOW_FLOOR"
    WRONG_SIGN = "RESPONSE_WRONG_SIGN"
    CONTROL_FAILED = "RESPONSE_CONTROL_FAILED"
    PARTIAL = "RESPONSE_PARTIAL"
    UNEVALUABLE = "RESPONSE_UNEVALUABLE"
    NOT_ATTEMPTED_PREREQUISITE = "RESPONSE_NOT_ATTEMPTED_PREREQUISITE"


class UniformElectronGasLawQualificationClass(StrEnum):
    TRANSVERSE_LAW_SUPPORTED = "LOCAL_LAW_TRANSVERSE_LAW_SUPPORTED"
    FINITE_Q_ONLY = "LOCAL_LAW_FINITE_Q_ONLY"
    NONLINEAR_BOUNDARY_LIMITED = "LOCAL_LAW_NONLINEAR_BOUNDARY_LIMITED"
    VIEW_LOCAL_ONLY = "LOCAL_LAW_VIEW_LOCAL_ONLY"
    NOT_SUPPORTED = "LOCAL_LAW_NOT_SUPPORTED"
    PARTIAL = "LOCAL_LAW_PARTIAL"
    UNEVALUABLE = "LOCAL_LAW_UNEVALUABLE"
    NOT_ATTEMPTED_PREREQUISITE = "LOCAL_LAW_NOT_ATTEMPTED_PREREQUISITE"


class UniformElectronGasAdmissionClass(StrEnum):
    MODEL_LOCAL_TRANSVERSE_ADMITTED = "ADMISSION_MODEL_LOCAL_TRANSVERSE_ADMITTED"
    BRANCH_LOCAL_ONLY = "ADMISSION_BRANCH_LOCAL_ONLY"
    EMPTY_HOLD = "ADMISSION_EMPTY_HOLD"
    NOT_ENTERED_LOCAL_LAW = "ADMISSION_NOT_ENTERED_LOCAL_LAW"
    PRESERVATION_OPERAND_REQUIRED = "ADMISSION_PRESERVATION_OPERAND_REQUIRED"
    AUTHORITY_REQUIRED = "ADMISSION_AUTHORITY_REQUIRED"
    UNEVALUABLE = "ADMISSION_UNEVALUABLE"
    NOT_ATTEMPTED_PREREQUISITE = "ADMISSION_NOT_ATTEMPTED_PREREQUISITE"


class GateName(StrEnum):
    AUTHORITY = "authority"
    BASELINE_PRESERVATION = "baseline_preservation"
    DYNAMICS = "dynamics"
    EFFORT = "effort"
    OBSERVATION_VALIDITY = "observation_validity"
    PHYSICAL_SINK = "physical_sink"
    REACHABILITY = "reachability"
    TARGET = "target"
    UNCERTAINTY = "uncertainty"


_REFERENCE_GAPS: Final = (
    "gap.algebra.falsifier-preservation",
    "gap.algebra.homogeneity",
    "gap.algebra.signed-opposition",
    "gap.calculus.dose-scaling",
    "gap.calculus.finite-difference-generator-convergence",
    "gap.calculus.numerical-view-convergence",
    "gap.calculus.odd-even-susceptibility",
    "gap.calculus.remainder-regularity",
    "gap.calculus.uncertainty-propagation",
    "gap.dynamics.causal-cones-clock-transport",
    "gap.dynamics.preservation-barriers",
    "gap.geometry.admission-margins",
    "gap.geometry.reachability-viability",
    "gap.geometry.support-charts-atlas",
    "gap.geometry.symmetry-gauge",
    "gap.geometry.tangent-rank",
)

_OUT_OF_SCOPE_GAPS: Final = (
    "gap.algebra.associativity-defect",
    "gap.algebra.cross-context-recurrence",
    "gap.algebra.identity-inverse-repetition",
    "gap.algebra.order-commutator",
    "gap.algebra.quotient-lumpability",
    "gap.algebra.restriction-gluing-transport",
    "gap.algebra.semigroup-embedding",
    "gap.algebra.sequential-composition",
    "gap.algebra.simultaneous-composition",
    "gap.calculus.curvature-hessian",
    "gap.calculus.local-jacobian-rank",
    "gap.calculus.mixed-port-volterra",
    "gap.calculus.nonsmooth-directional-derivative",
    "gap.calculus.time-scaling",
    "gap.dynamics.controllability-observability",
    "gap.dynamics.delay-relaxation",
    "gap.dynamics.gramians",
    "gap.dynamics.hysteresis-return",
    "gap.dynamics.koopman-operator",
    "gap.dynamics.local-evolution-operator",
    "gap.dynamics.nonnormal-amplification",
    "gap.dynamics.recurrence-stationarity",
    "gap.dynamics.sparse-evolution",
    "gap.dynamics.stability-transient",
    "gap.dynamics.state-closure-memory",
    "gap.geometry.boundary-strata",
    "gap.geometry.cohomology",
    "gap.geometry.decision-quotients",
    "gap.geometry.information-geometry",
    "gap.geometry.metric-structure",
    "gap.geometry.receiver-fibers",
    "gap.geometry.topology-restriction",
)


def _decimal(value: object, *, field_name: str) -> Decimal:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a decimal string")
    result = Decimal(value)
    validate_decimal(result, field_name=field_name)
    return result


def _integer(value: object, *, field_name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field_name} must be an integer >= {minimum}")
    return value


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field_name} must be an object with string keys")
    return value


def _require_keys(value: Mapping[str, object], expected: Sequence[str], *, field_name: str) -> None:
    expected_set = set(expected)
    if set(value) != expected_set:
        missing = sorted(expected_set - set(value))
        unknown = sorted(set(value) - expected_set)
        raise ValueError(f"{field_name} fields differ: missing={missing}, unknown={unknown}")


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field_name} must be a string list")
    return tuple(value)


def _decimal_tuple(value: object, *, field_name: str) -> tuple[Decimal, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a decimal-string list")
    return tuple(
        _decimal(item, field_name=f"{field_name}[{index}]") for index, item in enumerate(value)
    )


def _reject_binary_floats(value: object, *, field_name: str = "config") -> None:
    if isinstance(value, float):
        raise ValueError(f"{field_name} contains a binary float")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_binary_floats(item, field_name=f"{field_name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_binary_floats(item, field_name=f"{field_name}[{index}]")


@dataclass(frozen=True, slots=True)
class UniformElectronGasTransverseScreenConfig:
    """Decoded exact transverse-screen scientific configuration."""

    payload_sha256: str
    campaign_id: str
    freeze_id: str
    frozen_at_utc: str
    r_s: Decimal
    temperature_K: Decimal
    u0: Decimal
    u_values: tuple[Decimal, ...]
    q_over_kf: tuple[Decimal, ...]
    action_direction: tuple[Decimal, Decimal, Decimal]
    q_direction: tuple[Decimal, Decimal, Decimal]
    requested_clock: int
    accepted_clock: int
    applied_clock: int
    receiver_clock: int
    field_ceiling_T: Decimal
    slab_thickness_m: Decimal
    analytic_profile_points: int
    thresholds: tuple[tuple[str, Decimal], ...]
    truth_cases: tuple[TruthCase, ...]
    views: tuple[str, ...]
    positive_penetration_depth_m: Decimal
    deterministic_kernel_relative_error: Decimal
    storage_root: str
    external_root: str
    minimum_free_bytes: int
    resources: tuple[tuple[str, int | bool], ...]
    reference_conformance_gaps: tuple[str, ...]
    out_of_scope_gaps: tuple[str, ...]

    def threshold(self, key: str) -> Decimal:
        try:
            return dict(self.thresholds)[key]
        except KeyError as error:
            raise KeyError(f"unknown frozen threshold {key}") from error

    def resource(self, key: str) -> int | bool:
        try:
            return dict(self.resources)[key]
        except KeyError as error:
            raise KeyError(f"unknown frozen resource {key}") from error


def decode_config(document: Mapping[str, object], *, payload_sha256: str) -> UniformElectronGasTransverseScreenConfig:
    """Decode only the exact closed transverse-screen config schema."""

    validate_sha256(payload_sha256, field_name="payload_sha256")
    _reject_binary_floats(document)
    _require_keys(
        document,
        (
            "action",
            "authority",
            "denominator",
            "formal_gap_dispositions",
            "geometry",
            "identity",
            "lifecycle",
            "receiver",
            "resources",
            "source_dispositions",
            "storage",
            "tasks",
            "thresholds",
            "truth_known",
        ),
        field_name="config",
    )
    identity = _mapping(document["identity"], field_name="identity")
    _require_keys(identity, ("campaign_id", "plan_id", "schema", "version"), field_name="identity")
    if identity != {
        "campaign_id": CAMPAIGN_ID,
        "plan_id": PLAN_ID,
        "schema": CONFIG_SCHEMA,
        "version": CONFIG_VERSION,
    }:
        raise ValueError("uniform electron gas config identity differs from frozen transverse-screen configuration")

    lifecycle = _mapping(document["lifecycle"], field_name="lifecycle")
    _require_keys(lifecycle, ("freeze_id", "frozen_at_utc", "state"), field_name="lifecycle")
    if lifecycle["state"] != "FROZEN":
        raise ValueError("uniform electron gas transverse-screen config must be frozen")
    if not isinstance(lifecycle["freeze_id"], str) or not isinstance(
        lifecycle["frozen_at_utc"], str
    ):
        raise ValueError("freeze identity and time must be strings")
    validate_stable_id(lifecycle["freeze_id"], field_name="freeze_id")

    denominator = _mapping(document["denominator"], field_name="denominator")
    _require_keys(
        denominator,
        (
            "background",
            "dimension",
            "electron_charge_model",
            "electron_mass_model",
            "particle_dispersion",
            "r_s",
            "temperature_K",
        ),
        field_name="denominator",
    )
    if (
        denominator["background"] != "neutralizing-jellium"
        or denominator["dimension"] != 3
        or denominator["particle_dispersion"] != "free-electron-parabolic"
        or denominator["electron_charge_model"] != "CODATA-elementary-charge"
        or denominator["electron_mass_model"] != "CODATA-electron-mass"
    ):
        raise ValueError("physical denominator differs from frozen transverse-screen configuration")
    r_s = _decimal(denominator["r_s"], field_name="r_s")
    temperature = _decimal(denominator["temperature_K"], field_name="temperature_K")
    if r_s != Decimal("4") or temperature != Decimal("300"):
        raise ValueError("r_s or temperature differs from frozen transverse-screen configuration")

    action = _mapping(document["action"], field_name="action")
    _require_keys(
        action,
        (
            "accepted_clock",
            "applied_clock",
            "coordinate",
            "direction",
            "field_ceiling_T",
            "q_direction",
            "q_over_kf",
            "receiver_clock",
            "requested_clock",
            "u0",
            "u_values",
        ),
        field_name="action",
    )
    if action["coordinate"] != "u=e*v_F*A_T/E_F":
        raise ValueError("action coordinate differs from frozen transverse-screen configuration")
    u0 = _decimal(action["u0"], field_name="action.u0")
    u_values = _decimal_tuple(action["u_values"], field_name="action.u_values")
    q_values = _decimal_tuple(action["q_over_kf"], field_name="action.q_over_kf")
    a_direction = _decimal_tuple(action["direction"], field_name="action.direction")
    q_direction = _decimal_tuple(action["q_direction"], field_name="action.q_direction")
    if (
        u0 != Decimal("5e-7")
        or u_values
        != (
            Decimal("-5e-7"),
            Decimal("-2.5e-7"),
            Decimal("0"),
            Decimal("2.5e-7"),
            Decimal("5e-7"),
        )
        or q_values
        != (
            Decimal("0.015625"),
            Decimal("0.03125"),
            Decimal("0.046875"),
            Decimal("0.0625"),
        )
        or a_direction != (Decimal("0"), Decimal("1"), Decimal("0"))
        or q_direction != (Decimal("1"), Decimal("0"), Decimal("0"))
    ):
        raise ValueError("action chart differs from frozen transverse-screen configuration")
    clocks = tuple(
        _integer(action[key], field_name=f"action.{key}")
        for key in ("requested_clock", "accepted_clock", "applied_clock", "receiver_clock")
    )
    if clocks != (0, 1, 2, 3):
        raise ValueError("action/receiver clocks differ from frozen transverse-screen configuration")

    geometry = _mapping(document["geometry"], field_name="geometry")
    _require_keys(
        geometry,
        ("analytic_profile_points", "calibration", "slab_thickness_m"),
        field_name="geometry",
    )
    if geometry["calibration"] != "parallel-field-simply-connected-slab":
        raise ValueError("calibration geometry differs from frozen transverse-screen configuration")
    slab_thickness = _decimal(geometry["slab_thickness_m"], field_name="slab_thickness_m")
    profile_points = _integer(
        geometry["analytic_profile_points"],
        field_name="analytic_profile_points",
        minimum=3,
    )
    if slab_thickness != Decimal("5e-7") or profile_points != 101:
        raise ValueError("slab geometry differs from frozen transverse-screen configuration")

    thresholds_document = _mapping(document["thresholds"], field_name="thresholds")
    threshold_keys = (
        "action_realization_relative",
        "amplitude_locality_relative",
        "baseline_relative",
        "even_remainder_floor_multiplier",
        "even_remainder_response_relative",
        "normal_cancellation_relative",
        "order_preservation_fraction",
        "q_intercept_stability_relative",
        "response_floor_relative_to_diamagnetic",
        "shielding_score_minimum",
        "slab_fit_relative",
        "transverse_dot_relative",
        "view_k0_agreement_relative",
        "ward_residual",
    )
    _require_keys(thresholds_document, threshold_keys, field_name="thresholds")
    thresholds = tuple(
        sorted(
            (key, _decimal(thresholds_document[key], field_name=f"thresholds.{key}"))
            for key in threshold_keys
        )
    )
    expected_thresholds = {
        "action_realization_relative": Decimal("1e-12"),
        "amplitude_locality_relative": Decimal("0.05"),
        "baseline_relative": Decimal("1e-12"),
        "even_remainder_floor_multiplier": Decimal("5"),
        "even_remainder_response_relative": Decimal("0.02"),
        "normal_cancellation_relative": Decimal("1e-3"),
        "order_preservation_fraction": Decimal("0.95"),
        "q_intercept_stability_relative": Decimal("0.10"),
        "response_floor_relative_to_diamagnetic": Decimal("1e-6"),
        "shielding_score_minimum": Decimal("0.50"),
        "slab_fit_relative": Decimal("0.01"),
        "transverse_dot_relative": Decimal("1e-12"),
        "view_k0_agreement_relative": Decimal("0.05"),
        "ward_residual": Decimal("1e-6"),
    }
    if dict(thresholds) != expected_thresholds:
        raise ValueError("threshold family differs from frozen transverse-screen configuration")

    truth = _mapping(document["truth_known"], field_name="truth_known")
    _require_keys(
        truth,
        (
            "cases",
            "deterministic_kernel_relative_error",
            "positive_penetration_depth_m",
            "privileged_oracle_separate",
            "views",
        ),
        field_name="truth_known",
    )
    case_values = _string_tuple(truth["cases"], field_name="truth_known.cases")
    if tuple(sorted(case_values)) != case_values or set(case_values) != {
        value.value for value in TruthCase
    }:
        raise ValueError("truth-known case roster differs from frozen transverse-screen configuration")
    views = _string_tuple(truth["views"], field_name="truth_known.views")
    if views != ("base", "refined") or truth["privileged_oracle_separate"] is not True:
        raise ValueError("truth-known view/oracle contract differs from frozen transverse-screen configuration")

    gaps = _mapping(document["formal_gap_dispositions"], field_name="formal gaps")
    _require_keys(gaps, ("out_of_scope", "reference_conformance_only"), field_name="formal gaps")
    reference_gaps = _string_tuple(
        gaps["reference_conformance_only"], field_name="reference_conformance_only"
    )
    out_gaps = _string_tuple(gaps["out_of_scope"], field_name="out_of_scope")
    if reference_gaps != _REFERENCE_GAPS or out_gaps != _OUT_OF_SCOPE_GAPS:
        raise ValueError("formal-gap dispositions differ from frozen transverse-screen configuration")
    if len(set(reference_gaps) | set(out_gaps)) != 48 or set(reference_gaps) & set(out_gaps):
        raise ValueError("formal-gap dispositions must partition all 48 registered gaps")

    source = _mapping(document["source_dispositions"], field_name="source_dispositions")
    _require_keys(
        source,
        ("paper_rpa", "target_gauge_closed_roster", "vertex_reference"),
        field_name="source_dispositions",
    )
    if source["target_gauge_closed_roster"] != []:
        raise ValueError("transverse-screen target gauge-closed roster must be empty")
    for key, source_identity in (
        ("paper_rpa", "arxiv:2511.00625v3"),
        ("vertex_reference", "arxiv:2512.19382v2"),
    ):
        value = _mapping(source[key], field_name=f"source_dispositions.{key}")
        _require_keys(value, ("identity", "lane", "status"), field_name=key)
        if value != {
            "identity": source_identity,
            "lane": EvidenceLane.TC_CLOSURE_DIAGNOSTIC.value,
            "status": "CLOSURE_ONLY_SOURCE",
        }:
            raise ValueError(f"{key} source disposition differs from transverse receiver screen source assessment")

    tasks = _mapping(document["tasks"], field_name="tasks")
    _require_keys(tasks, ("conformance-truth-known", "development-law-commitment", "evaluation-target-reveal", "qualification-target-source", "supplementary-closure-diagnostic", "terminal-closeout"), field_name="tasks")
    if tasks != {
        "conformance-truth-known": "ENTERED_TRUTH_KNOWN_CONFORMANCE",
        "development-law-commitment": "NOT_ATTEMPTED_PREREQUISITE_TARGET_SOURCE",
        "evaluation-target-reveal": "NOT_ATTEMPTED_PREREQUISITE_TARGET_SOURCE",
        "qualification-target-source": "NOT_ATTEMPTED_PREREQUISITE_TARGET_SOURCE",
        "supplementary-closure-diagnostic": "NOT_ATTEMPTED_PREREQUISITE_EXACT_CLOSURE_BINDING",
        "terminal-closeout": "ENTER_AFTER_TRUTH_KNOWN_CONFORMANCE",
    }:
        raise ValueError("task entry/nonattempt contract differs from frozen transverse-screen configuration")

    receiver = _mapping(document["receiver"], field_name="receiver")
    _require_keys(
        receiver,
        (
            "current_direction",
            "current_unit",
            "kernel_unit",
            "limit_order",
            "penetration_depth_unit",
            "projection",
            "static_frequency_rad_s",
        ),
        field_name="receiver",
    )
    if (
        receiver["current_direction"] != ["0", "1", "0"]
        or receiver["current_unit"] != "A/m^2"
        or receiver["kernel_unit"] != "A/(T*m^3)"
        or receiver["penetration_depth_unit"] != "m"
        or receiver["projection"] != "dot(J_T,A_hat)"
        or receiver["static_frequency_rad_s"] != "0"
        or receiver["limit_order"] != "omega=0-after-source-convergence;finite-q-fit-then-q-to-zero"
    ):
        raise ValueError("receiver contract differs from frozen transverse-screen configuration")

    storage = _mapping(document["storage"], field_name="storage")
    _require_keys(
        storage,
        ("external_root", "minimum_free_bytes", "storage_root"),
        field_name="storage",
    )
    if storage["external_root"] != EXTERNAL_ROOT or storage["storage_root"] != (
        str(required_external_path("EMPIRICAL_LAWHOOD_ELECTRON_GAS_EXTERNAL_ROOT"))
    ):
        raise ValueError("storage route differs from frozen transverse-screen configuration")

    resources_document = _mapping(document["resources"], field_name="resources")
    resource_keys = (
        "cpu_cores",
        "gpu_devices",
        "maximum_all_inputs_bytes",
        "maximum_one_panel_bytes",
        "memory_bytes",
        "network_required",
        "output_bytes",
        "scratch_bytes",
        "wall_time_seconds",
    )
    _require_keys(resources_document, resource_keys, field_name="resources")
    resources: list[tuple[str, int | bool]] = []
    for key in resource_keys:
        resource_value = resources_document[key]
        if key == "network_required":
            if not isinstance(resource_value, bool):
                raise ValueError("network_required must be boolean")
        else:
            resource_value = _integer(resource_value, field_name=f"resources.{key}")
        resources.append((key, resource_value))
    expected_resources: dict[str, int | bool] = {
        "cpu_cores": 2,
        "gpu_devices": 0,
        "maximum_all_inputs_bytes": 32 * 1024**2,
        "maximum_one_panel_bytes": MAXIMUM_PANEL_BYTES,
        "memory_bytes": 2 * 1024**3,
        "network_required": False,
        "output_bytes": 64 * 1024**2,
        "scratch_bytes": 256 * 1024**2,
        "wall_time_seconds": 600,
    }
    if dict(resources) != expected_resources:
        raise ValueError("resource envelope differs from frozen transverse-screen configuration")

    authority = _mapping(document["authority"], field_name="authority")
    _require_keys(
        authority,
        (
            "accountable_owner",
            "authorization_basis",
            "external_scientific_write",
            "implementation",
            "public_web_source_screen",
            "reveal",
            "source_acquisition",
            "truth_known_execution",
        ),
        field_name="authority",
    )
    if (
        authority["accountable_owner"] != "project-owner"
        or authority["implementation"] != "OWNER_AUTHORIZED"
        or authority["truth_known_execution"] != "OWNER_AUTHORIZED"
        or authority["external_scientific_write"] != "OWNER_AUTHORIZED"
        or authority["source_acquisition"] != "NOT_ENTERED"
        or authority["reveal"] != "OWNER_AUTHORIZED_SEPARATE_EVALUATOR_ACT"
        or not isinstance(authority["authorization_basis"], str)
    ):
        raise ValueError("authority scope differs from frozen transverse-screen configuration")

    return UniformElectronGasTransverseScreenConfig(
        payload_sha256=payload_sha256,
        campaign_id=CAMPAIGN_ID,
        freeze_id=lifecycle["freeze_id"],
        frozen_at_utc=lifecycle["frozen_at_utc"],
        r_s=r_s,
        temperature_K=temperature,
        u0=u0,
        u_values=u_values,
        q_over_kf=q_values,
        action_direction=(a_direction[0], a_direction[1], a_direction[2]),
        q_direction=(q_direction[0], q_direction[1], q_direction[2]),
        requested_clock=clocks[0],
        accepted_clock=clocks[1],
        applied_clock=clocks[2],
        receiver_clock=clocks[3],
        field_ceiling_T=_decimal(action["field_ceiling_T"], field_name="field_ceiling_T"),
        slab_thickness_m=slab_thickness,
        analytic_profile_points=profile_points,
        thresholds=thresholds,
        truth_cases=tuple(TruthCase(value) for value in case_values),
        views=views,
        positive_penetration_depth_m=_decimal(
            truth["positive_penetration_depth_m"],
            field_name="positive_penetration_depth_m",
        ),
        deterministic_kernel_relative_error=_decimal(
            truth["deterministic_kernel_relative_error"],
            field_name="deterministic_kernel_relative_error",
        ),
        storage_root=storage["storage_root"],
        external_root=EXTERNAL_ROOT,
        minimum_free_bytes=_integer(
            storage["minimum_free_bytes"], field_name="minimum_free_bytes", minimum=1
        ),
        resources=tuple(sorted(resources)),
        reference_conformance_gaps=reference_gaps,
        out_of_scope_gaps=out_gaps,
    )


def decode_config_bytes(payload: bytes) -> UniformElectronGasTransverseScreenConfig:
    if not isinstance(payload, bytes) or len(payload) > MAXIMUM_CONFIG_BYTES:
        raise ValueError("uniform electron gas config bytes are absent or oversized")
    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("uniform electron gas config is not strict JSON") from error
    if not isinstance(document, dict):
        raise ValueError("uniform electron gas config root must be an object")
    return decode_config(document, payload_sha256=sha256(payload).hexdigest())


def load_config(path: Path) -> tuple[UniformElectronGasTransverseScreenConfig, bytes]:
    payload = path.read_bytes()
    return decode_config_bytes(payload), payload


Vector3 = tuple[Decimal, Decimal, Decimal]


def validate_vector(value: Vector3, *, field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{field_name} must be an exact three-vector")
    for index, coordinate in enumerate(value):
        validate_decimal(coordinate, field_name=f"{field_name}[{index}]")


@dataclass(frozen=True, slots=True)
class ActionCurrentRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/action-current-row'

    row_id: str
    q_over_kf: Decimal
    u: Decimal
    requested_A_T: Vector3
    accepted_A_T: Vector3
    applied_A_T: Vector3
    realized_A_T: Vector3
    current_density: Vector3
    current_error_bound_A_m2: Decimal
    requested_clock: int
    accepted_clock: int
    applied_clock: int
    receiver_clock: int
    accepted: bool
    valid: bool
    reason_code: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.row_id, field_name="row_id")
        validate_decimal(self.q_over_kf, field_name="q_over_kf", minimum=Decimal("0"))
        if self.q_over_kf == 0:
            raise ValueError("finite-q action rows require q_over_kf > 0")
        validate_decimal(self.u, field_name="u")
        for name in (
            "requested_A_T",
            "accepted_A_T",
            "applied_A_T",
            "realized_A_T",
            "current_density",
        ):
            validate_vector(getattr(self, name), field_name=name)
        validate_decimal(
            self.current_error_bound_A_m2,
            field_name="current_error_bound_A_m2",
            minimum=Decimal("0"),
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                self.requested_clock,
                self.accepted_clock,
                self.applied_clock,
                self.receiver_clock,
            )
        ):
            raise ValueError("action/receiver clocks must be exact integers")
        if self.reason_code is not None:
            validate_stable_id(self.reason_code, field_name="reason_code")
        if self.valid and (not self.accepted or self.reason_code is not None):
            raise ValueError("a valid row must be accepted without a reason code")
        if not self.valid and self.reason_code is None:
            raise ValueError("an invalid row requires a reason code")


@dataclass(frozen=True, slots=True)
class GaugeControlRow(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/gauge-control-row'

    control_id: str
    q_over_kf: Decimal
    diamagnetic_normal: Decimal
    paramagnetic_normal: Decimal
    ward_residual: Decimal
    wrong_polarization_current_A_m2: Decimal
    converged: bool
    limit_order_supported: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.control_id, field_name="control_id")
        validate_decimal(self.q_over_kf, field_name="q_over_kf", minimum=Decimal("0"))
        for name in (
            "diamagnetic_normal",
            "paramagnetic_normal",
            "ward_residual",
            "wrong_polarization_current_A_m2",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if self.ward_residual < 0:
            raise ValueError("ward_residual must be nonnegative")


@dataclass(frozen=True, slots=True)
class TransversePanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/transverse-panel'

    panel_id: str
    source_id: str
    branch_id: str
    view_id: str
    evidence_lane: EvidenceLane
    evidence_world: str
    independent_unit_id: str
    deterministic: bool
    r_s: Decimal
    temperature_K: Decimal
    q_direction: Vector3
    action_direction: Vector3
    rows: tuple[ActionCurrentRow, ...]
    controls: tuple[GaugeControlRow, ...]
    order_zero: Decimal | None
    order_at_max_action: Decimal | None
    baseline_preserved: bool | None
    source_semantics_valid: bool

    def __post_init__(self) -> None:
        for name in (
            "panel_id",
            "source_id",
            "branch_id",
            "view_id",
            "evidence_world",
            "independent_unit_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.evidence_lane is not EvidenceLane.GAUGE_CLOSED_TRANSVERSE:
            raise ValueError("a transverse panel must use the gauge-closed lane")
        validate_decimal(self.r_s, field_name="r_s", minimum=Decimal("0"))
        validate_decimal(self.temperature_K, field_name="temperature_K", minimum=Decimal("0"))
        validate_vector(self.q_direction, field_name="q_direction")
        validate_vector(self.action_direction, field_name="action_direction")
        require_sorted_unique_ids(self.rows, attribute="row_id", field_name="rows")
        require_sorted_unique_ids(self.controls, attribute="control_id", field_name="controls")
        if not self.rows or not self.controls:
            raise ValueError("a transverse panel requires action rows and gauge controls")
        for name in ("order_zero", "order_at_max_action"):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name, minimum=Decimal("0"))


@dataclass(frozen=True, slots=True)
class BranchInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/branch-input'

    input_id: str
    case_id: str
    panels: tuple[TransversePanel, ...]
    authority_valid: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        validate_stable_id(self.case_id, field_name="case_id")
        require_sorted_unique_ids(self.panels, attribute="panel_id", field_name="panels")
        if not self.panels:
            raise ValueError("branch input requires at least one panel")
        branch_ids = {panel.branch_id for panel in self.panels}
        view_ids = {panel.view_id for panel in self.panels}
        if len(branch_ids) != 1 or len(view_ids) != len(self.panels):
            raise ValueError("branch panels must share one branch and have unique views")


@dataclass(frozen=True, slots=True)
class ClosureDiagnosticInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/closure-diagnostic-input'

    input_id: str
    source_id: str
    branch_id: str
    r_s: Decimal
    temperature_K: Decimal
    supplied_tc_K: Decimal
    stiffness_closure: str

    def __post_init__(self) -> None:
        for name in ("input_id", "source_id", "branch_id", "stiffness_closure"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("r_s", "temperature_K", "supplied_tc_K"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal("0"))


@dataclass(frozen=True, slots=True)
class FiniteQEstimate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/finite-q-estimate'

    estimate_id: str
    q_over_kf: Decimal
    kernel_half: Decimal
    kernel_full: Decimal
    kernel_error_bound: Decimal
    offset_current_A_m2: Decimal
    even_remainder_half_A_m2: Decimal
    even_remainder_full_A_m2: Decimal
    locality_relative: Decimal
    locality_pass: bool
    even_remainder_pass: bool
    zero_offset_pass: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.estimate_id, field_name="estimate_id")
        for name in (
            "q_over_kf",
            "kernel_half",
            "kernel_full",
            "kernel_error_bound",
            "offset_current_A_m2",
            "even_remainder_half_A_m2",
            "even_remainder_full_A_m2",
            "locality_relative",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if self.kernel_error_bound < 0 or self.locality_relative < 0:
            raise ValueError("estimate error/locality must be nonnegative")


@dataclass(frozen=True, slots=True)
class ViewLawResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/view-law-result'

    result_id: str
    panel_id: str
    view_id: str
    measurement_class: UniformElectronGasMeasurementClass
    observation_order_class: UniformElectronGasObservationOrderClass
    response_class: UniformElectronGasResponseClass
    law_qualification_class: UniformElectronGasLawQualificationClass
    finite_q: tuple[FiniteQEstimate, ...]
    kernel0: Decimal | None
    kernel0_error_bound: Decimal | None
    kernel0_lower_bound: Decimal | None
    c2: Decimal | None
    q_intercept_stability_relative: Decimal | None
    normal_cancellation_max_relative: Decimal
    ward_residual_max: Decimal
    maximum_field_T: Decimal
    order_preservation_fraction: Decimal | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("result_id", "panel_id", "view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(self.finite_q, attribute="estimate_id", field_name="finite_q")
        for name in (
            "kernel0",
            "kernel0_error_bound",
            "kernel0_lower_bound",
            "c2",
            "q_intercept_stability_relative",
            "order_preservation_fraction",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name)
        for name in (
            "normal_cancellation_max_relative",
            "ward_residual_max",
            "maximum_field_T",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal("0"))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class BranchLawResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/branch-law-result'

    result_id: str
    input_id: str
    case_id: str
    measurement_class: UniformElectronGasMeasurementClass
    observation_order_class: UniformElectronGasObservationOrderClass
    response_class: UniformElectronGasResponseClass
    law_qualification_class: UniformElectronGasLawQualificationClass
    view_results: tuple[ViewLawResult, ...]
    robust_kernel0_lower_bound: Decimal | None
    view_k0_agreement_relative: Decimal | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("result_id", "input_id", "case_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.view_results, attribute="result_id", field_name="view_results"
        )
        for name in ("robust_kernel0_lower_bound", "view_k0_agreement_relative"):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class AdmissionGateResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/admission-gate-result'

    gate: GateName
    passed: bool
    margin: Decimal | None
    reason_code: str

    def __post_init__(self) -> None:
        if self.margin is not None:
            validate_decimal(self.margin, field_name="margin")
        validate_stable_id(self.reason_code, field_name="reason_code")


@dataclass(frozen=True, slots=True)
class ViewScreenResult(CanonicalRecord):
    """One numerical view's complete, noncompensating receiver admission decision."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/view-screen-result'

    result_id: str
    panel_id: str
    view_id: str
    admission_class: UniformElectronGasAdmissionClass
    gates: tuple[AdmissionGateResult, ...]
    kernel0_lower_bound: Decimal | None
    penetration_depth_upper_m: Decimal | None
    shielding_score_lower: Decimal | None
    hold: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("result_id", "panel_id", "view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if tuple(gate.gate.value for gate in self.gates) != tuple(gate.value for gate in GateName):
            raise ValueError("view result must contain all nine sorted admission gates")
        for name in (
            "kernel0_lower_bound",
            "penetration_depth_upper_m",
            "shielding_score_lower",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        admitted = self.admission_class is UniformElectronGasAdmissionClass.MODEL_LOCAL_TRANSVERSE_ADMITTED
        if admitted == self.hold:
            raise ValueError("view admission and hold must be exact complements")
        if admitted != all(gate.passed for gate in self.gates):
            raise ValueError("view admission must equal the nine-gate intersection")


@dataclass(frozen=True, slots=True)
class BranchScreenResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/branch-screen-result'

    result_id: str
    input_id: str
    case_id: str
    evidence_lane: EvidenceLane
    measurement_class: UniformElectronGasMeasurementClass
    observation_order_class: UniformElectronGasObservationOrderClass
    response_class: UniformElectronGasResponseClass
    law_qualification_class: UniformElectronGasLawQualificationClass
    admission_class: UniformElectronGasAdmissionClass
    view_results: tuple[ViewLawResult, ...]
    view_screens: tuple[ViewScreenResult, ...]
    gates: tuple[AdmissionGateResult, ...]
    robust_kernel0_lower_bound: Decimal | None
    penetration_depth_upper_m: Decimal | None
    shielding_score_lower: Decimal | None
    hold: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("result_id", "input_id", "case_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.view_results, attribute="result_id", field_name="view_results"
        )
        require_sorted_unique_ids(
            self.view_screens, attribute="result_id", field_name="view_screens"
        )
        if tuple(gate.gate.value for gate in self.gates) != tuple(gate.value for gate in GateName):
            raise ValueError("branch result must contain all nine sorted admission gates")
        if self.evidence_lane is EvidenceLane.GAUGE_CLOSED_TRANSVERSE:
            if tuple(value.view_id for value in self.view_results) != tuple(
                value.view_id for value in self.view_screens
            ):
                raise ValueError("branch law and admission views differ")
        elif self.view_screens:
            raise ValueError("closure-only results cannot carry transverse view screens")
        for name in (
            "robust_kernel0_lower_bound",
            "penetration_depth_upper_m",
            "shielding_score_lower",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.admission_class is UniformElectronGasAdmissionClass.MODEL_LOCAL_TRANSVERSE_ADMITTED and self.hold:
            raise ValueError("an admitted result cannot command hold")
        if self.admission_class is not UniformElectronGasAdmissionClass.MODEL_LOCAL_TRANSVERSE_ADMITTED and not self.hold:
            raise ValueError("every non-admitted result must command hold")
        if (self.admission_class is UniformElectronGasAdmissionClass.MODEL_LOCAL_TRANSVERSE_ADMITTED) != all(
            gate.passed for gate in self.gates
        ):
            raise ValueError("branch admission must equal the nine-gate intersection")


@dataclass(frozen=True, slots=True)
class TruthOracle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/truth-oracle'

    oracle_id: str
    case_id: str
    truth_family: TruthCase
    expected_response_class: UniformElectronGasResponseClass
    expected_law_qualification_class: UniformElectronGasLawQualificationClass
    expected_admission_class: UniformElectronGasAdmissionClass
    true_penetration_depth_m: Decimal | None
    decisive_reason_code: str

    def __post_init__(self) -> None:
        for name in ("oracle_id", "case_id", "decisive_reason_code"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.true_penetration_depth_m is not None:
            validate_decimal(
                self.true_penetration_depth_m,
                field_name="true_penetration_depth_m",
                minimum=Decimal("0"),
            )


@dataclass(frozen=True, slots=True)
class TruthKnownCase(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/truth-known-case'

    case_id: str
    method_input: BranchInput
    privileged_oracle: TruthOracle

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        if (
            self.method_input.case_id != self.case_id
            or self.privileged_oracle.case_id != self.case_id
        ):
            raise ValueError("truth-known case input/oracle identity differs")


@dataclass(frozen=True, slots=True)
class CaseConformance(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/case-conformance'

    case_id: str
    result: BranchScreenResult
    oracle: TruthOracle
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        if self.result.case_id != self.case_id or self.oracle.case_id != self.case_id:
            raise ValueError("conformance case identities differ")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class MethodConformanceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/method-conformance-result'

    result_id: str
    config_sha256: str
    cases: tuple[CaseConformance, ...]
    closure_nonpromotion_pass: bool
    passed: bool
    status: str

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_sha256(self.config_sha256, field_name="config_sha256")
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a nonempty string")
        if self.passed != (
            self.closure_nonpromotion_pass and all(case.passed for case in self.cases)
        ):
            raise ValueError("method conformance verdict differs from case evidence")


@dataclass(frozen=True, slots=True)
class SourceStopResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/source-stop-result'

    result_id: str
    config_sha256: str
    source_dispositions: tuple[tuple[str, str], ...]
    task_dispositions: tuple[tuple[str, str], ...]
    target_contact_count: int
    status: str

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_sha256(self.config_sha256, field_name="config_sha256")
        if tuple(sorted(dict(self.source_dispositions))) != tuple(
            key for key, _value in self.source_dispositions
        ):
            raise ValueError("source dispositions must have sorted unique keys")
        if tuple(sorted(dict(self.task_dispositions))) != tuple(
            key for key, _value in self.task_dispositions
        ):
            raise ValueError("task dispositions must have sorted unique keys")
        if self.target_contact_count != 0:
            raise ValueError("transverse-screen source stop must contact zero target panels")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a nonempty string")


@dataclass(frozen=True, slots=True)
class TruthInputBatch(CanonicalRecord):
    """Truth-blind method inputs; privileged family labels are prohibited."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/truth-input-batch'

    batch_id: str
    config_sha256: str
    inputs: tuple[BranchInput, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        validate_sha256(self.config_sha256, field_name="config_sha256")
        require_sorted_unique_ids(self.inputs, attribute="case_id", field_name="inputs")
        if not self.inputs:
            raise ValueError("truth input batch cannot be empty")


@dataclass(frozen=True, slots=True)
class TruthOracleBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/truth-oracle-batch'

    batch_id: str
    config_sha256: str
    oracles: tuple[TruthOracle, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        validate_sha256(self.config_sha256, field_name="config_sha256")
        require_sorted_unique_ids(self.oracles, attribute="case_id", field_name="oracles")
        if not self.oracles:
            raise ValueError("truth oracle batch cannot be empty")


@dataclass(frozen=True, slots=True)
class TruthLawBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/truth-law-batch'

    batch_id: str
    config_sha256: str
    laws: tuple[BranchLawResult, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        validate_sha256(self.config_sha256, field_name="config_sha256")
        require_sorted_unique_ids(self.laws, attribute="case_id", field_name="laws")
        if not self.laws:
            raise ValueError("truth law batch cannot be empty")


@dataclass(frozen=True, slots=True)
class TruthScreenBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/uniform-electron-gas-response/truth-screen-batch'

    batch_id: str
    config_sha256: str
    screens: tuple[BranchScreenResult, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        validate_sha256(self.config_sha256, field_name="config_sha256")
        require_sorted_unique_ids(self.screens, attribute="case_id", field_name="screens")
        if not self.screens:
            raise ValueError("truth screen batch cannot be empty")


__all__ = [
    "ADMISSION_CAPABILITY_KEY",
    "ADJUDICATION_SCHEMA",
    "ActionCurrentRow",
    "AdmissionGateResult",
    "BranchInput",
    "BranchLawResult",
    "BranchScreenResult",
    "CAMPAIGN_ID",
    "CAPABILITY_VERSION",
    "CONFIG_SCHEMA",
    "CaseConformance",
    "ClosureDiagnosticInput",
    "EVALUATOR_CAPABILITY_KEY",
    "EvidenceLane",
    "EXTERNAL_ROOT",
    "FiniteQEstimate",
    "GateName",
    "GaugeControlRow",
    "MAXIMUM_PANEL_BYTES",
    "METHOD_CAPABILITY_KEY",
    "MethodConformanceResult",
    'UniformElectronGasMeasurementClass',
    'UniformElectronGasObservationOrderClass',
    'UniformElectronGasResponseClass',
    'UniformElectronGasLawQualificationClass',
    'UniformElectronGasAdmissionClass',
    "PLAN_ID",
    "SOURCE_CAPABILITY_KEY",
    "SourceStopResult",
    "UniformElectronGasTransverseScreenConfig",
    "TransversePanel",
    "TruthCase",
    "TruthInputBatch",
    "TruthKnownCase",
    "TruthLawBatch",
    "TruthOracle",
    "TruthOracleBatch",
    "TruthScreenBatch",
    "Vector3",
    "ViewLawResult",
    "ViewScreenResult",
    "decode_config",
    "decode_config_bytes",
    "load_config",
]
