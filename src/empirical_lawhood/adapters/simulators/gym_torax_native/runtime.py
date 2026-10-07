"""Bounded, source-only Gym--TORAX native episode acquisition.

This module deliberately owns no persistence, issue, publication, method, or
scientific-finalization authority.  Dependencies are imported only by runtime
preflight/environment construction, and tests can inject a source-compatible
environment without importing Gym--TORAX.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from dataclasses import dataclass
from decimal import Decimal
from importlib import metadata as importlib_metadata
import math
from pathlib import Path
import re
import resource
import sys
import time
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .diagnostic_contracts import GYM_TORAX_HORIZON_REQUESTS, GymToraxActionDelivery, GymToraxDeliveryDisposition, GymToraxNativeAction, GymToraxNumericalDisposition, GymToraxNumericalMember, GymToraxObservationDisposition, GymToraxOperatorSourceSummary, GymToraxPreparation, GymToraxSourceDisposition
from .field_metadata_contracts import GymToraxFieldMetadataEpisodeRequest, GymToraxFieldMetadataFloat64Block, GymToraxFieldMetadataNativeEpisode, encode_gym_torax_float64_block
from .field_metadata import GymToraxFieldMetadataManifest, verify_installed_gym_torax_metadata_sources
from .extraction_manifest import GymToraxBoundedExtractionManifest, verify_gym_torax_extraction_manifest

_EXPECTED_VERSIONS = {
    "python": "3.11.14",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "xarray": "2026.7.0",
    "gymtorax": "1.1.1",
    "torax": "1.4.2",
    "jax": "0.10.2",
    "jaxlib": "0.10.2",
}
_REQUIRED_PROFILES = frozenset(("T_e", "T_i", "n_e", "psi", "q"))
_REQUIRED_SCALARS = frozenset(
    (
        "H98",
        "P_heat_total",
        "P_radiation_e",
        "Q_fusion",
        "beta_N",
        "fgw_n_e_volume_avg",
        "q95",
        "q_min",
    )
)
_REQUIRED_NUMERICS = frozenset(("solver_error_state",))
_NUMERICS_FIELDS = (
    "inner_solver_iterations",
    "outer_solver_iterations",
    "sawtooth_crash",
    "solver_error_state",
)
_OPERATOR_REQUEST_CLOCKS = frozenset(range(104, 110))
_MAX_CAPTURED_ARRAY_ELEMENTS = 2_000_000
_MINIMUM_EPISODE_ENVELOPE_BYTES = 16 * 1024
_BLOCK_METADATA_RESERVE_BYTES = 2 * 1024
_ACTION_COMPONENT_IDS = (
    "ip-a",
    "nbi-power-w",
    "nbi-location",
    "nbi-width",
    "ecrh-power-w",
    "ecrh-location",
    "ecrh-width",
)


class GymToraxEnvironment(Protocol):
    """Minimal injected environment surface used by acquisition."""

    @property
    def current_time(self) -> object: ...

    @property
    def state(self) -> object: ...

    def reset(self, *, seed: int) -> object: ...

    def step(self, action: Mapping[str, npt.NDArray[np.float64]]) -> object: ...

    def close(self) -> object: ...


@dataclass(frozen=True, slots=True)
class GymToraxRuntimeProbe:
    python_version: str
    numpy_version: str
    scipy_version: str
    xarray_version: str
    gymtorax_version: str
    torax_version: str
    jax_version: str
    jaxlib_version: str
    backend: str
    x64_enabled: bool


@dataclass(frozen=True, slots=True)
class GymToraxCapturedState:
    profiles: Mapping[str, npt.NDArray[np.float64]]
    scalars: Mapping[str, npt.NDArray[np.float64]]
    numerics: Mapping[str, npt.NDArray[np.float64]]
    dimensions: Mapping[tuple[str, str], tuple[str, ...]]
    coordinates: Mapping[str, npt.NDArray[np.float64]]
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GymToraxOperatorSnapshot:
    """One authentic source-provided controlled-IO operand triple."""

    request_clock: int
    source_method_id: str
    state_coordinate_ids: tuple[str, ...]
    input_coordinate_ids: tuple[str, ...]
    receiver_coordinate_ids: tuple[str, ...]
    transition_a: npt.NDArray[np.float64]
    input_b: npt.NDArray[np.float64]
    receiver_c: npt.NDArray[np.float64]


class GymToraxRuntimePreflightError(RuntimeError):
    """Fail-closed runtime-denominator mismatch before environment creation."""

    def __init__(self, reason_codes: Sequence[str]) -> None:
        self.reason_codes = tuple(sorted(set(reason_codes)))
        super().__init__(",".join(self.reason_codes))


class GymToraxOutputBoundError(RuntimeError):
    """The requested bound cannot contain even a typed minimal episode."""


@dataclass(slots=True)
class _GymToraxEncodedOutputBudget:
    maximum_bytes: int
    reserved_bytes: int = _MINIMUM_EPISODE_ENVELOPE_BYTES

    def reserve_array(self, values: npt.NDArray[np.float64]) -> None:
        self.reserve_shape(tuple(int(value) for value in values.shape))

    def reserve_shape(self, shape: tuple[int, ...]) -> None:
        element_count = math.prod(shape)
        raw_bytes = element_count * 8
        encoded_bytes = 4 * ((raw_bytes + 2) // 3)
        proposed = self.reserved_bytes + encoded_bytes + _BLOCK_METADATA_RESERVE_BYTES
        if proposed > self.maximum_bytes:
            raise GymToraxOutputBoundError("GYM_TORAX_OUTPUT_BOUND_EXCEEDED")
        self.reserved_bytes = proposed

    def reserve_record(self, value: CanonicalRecord) -> None:
        proposed = self.reserved_bytes + len(value.canonical_bytes())
        if proposed > self.maximum_bytes:
            raise GymToraxOutputBoundError("GYM_TORAX_OUTPUT_BOUND_EXCEEDED")
        self.reserved_bytes = proposed


def _array_output_reservation(values: npt.NDArray[np.float64]) -> int:
    raw_bytes = int(values.size) * 8
    return 4 * ((raw_bytes + 2) // 3) + _BLOCK_METADATA_RESERVE_BYTES


def _reserve_capture_before_append(
    capture: GymToraxCapturedState,
    *,
    reserved_bytes: int,
    maximum_bytes: int,
    include_coordinates: bool,
) -> int:
    arrays = (
        *capture.profiles.values(),
        *capture.scalars.values(),
        *capture.numerics.values(),
        *(capture.coordinates.values() if include_coordinates else ()),
    )
    proposed = reserved_bytes + sum(_array_output_reservation(value) for value in arrays)
    if proposed > maximum_bytes:
        raise GymToraxOutputBoundError("GYM_TORAX_OUTPUT_BOUND_EXCEEDED")
    return proposed


def _reserve_operator_before_append(
    snapshot: GymToraxOperatorSnapshot,
    *,
    reserved_bytes: int,
    maximum_bytes: int,
) -> int:
    proposed = reserved_bytes + sum(
        _array_output_reservation(value)
        for value in (snapshot.transition_a, snapshot.input_b, snapshot.receiver_c)
    )
    if proposed > maximum_bytes:
        raise GymToraxOutputBoundError("GYM_TORAX_OUTPUT_BOUND_EXCEEDED")
    return proposed


GymToraxEnvironmentFactory = Callable[
    [GymToraxPreparation, GymToraxNumericalMember], GymToraxEnvironment
]
GymToraxRuntimeInspector = Callable[[], GymToraxRuntimeProbe]
GymToraxProgressCallback = Callable[[Decimal], None]
GymToraxResetObserver = Callable[[], None]


class GymToraxProgressEmissionError(RuntimeError):
    """The executor's monotone progress port rejected a completed clock."""


def _emit_completed_clock_progress(
    callback: GymToraxProgressCallback | None,
    receiver_clock: int,
) -> None:
    if callback is None:
        return
    try:
        # Counter zero is reserved for "no completed work".  Reset clock zero
        # therefore advances to one, followed by receiver clocks 1--120 as
        # counters 2--121.
        callback(Decimal(receiver_clock + 1))
    except Exception as error:
        raise GymToraxProgressEmissionError(
            "native receiver-clock progress could not be emitted"
        ) from error


def inspect_gym_torax_runtime() -> GymToraxRuntimeProbe:
    """Late-import JAX and bind the exact CPU/float64 source denominator."""

    import jax

    jax_config = cast(Any, jax.config)
    jax_config.update("jax_platforms", "cpu")
    jax_config.update("jax_enable_x64", True)
    return GymToraxRuntimeProbe(
        python_version=(
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        ),
        numpy_version=importlib_metadata.version("numpy"),
        scipy_version=importlib_metadata.version("scipy"),
        xarray_version=importlib_metadata.version("xarray"),
        gymtorax_version=importlib_metadata.version("gymtorax"),
        torax_version=importlib_metadata.version("torax"),
        jax_version=importlib_metadata.version("jax"),
        jaxlib_version=importlib_metadata.version("jaxlib"),
        backend=str(jax.default_backend()),
        x64_enabled=bool(jax_config.read("jax_enable_x64")),
    )


def _validate_runtime_probe(probe: GymToraxRuntimeProbe) -> None:
    reasons = {
        f"RUNTIME_VERSION_MISMATCH_{name.upper()}"
        for name, expected in _EXPECTED_VERSIONS.items()
        if getattr(probe, f"{name}_version") != expected
    }
    if probe.backend != "cpu":
        reasons.add("RUNTIME_BACKEND_NOT_CPU")
    if not probe.x64_enabled:
        reasons.add("RUNTIME_FLOAT64_DISABLED")
    if reasons:
        raise GymToraxRuntimePreflightError(tuple(reasons))


def _validate_exact_native_schedule(request: GymToraxFieldMetadataEpisodeRequest) -> None:
    from .action_word import build_gym_torax_native_schedule

    try:
        expected = build_gym_torax_native_schedule(request.schedule.action_word)
    except ValueError as error:
        raise GymToraxRuntimePreflightError(("NATIVE_ACTION_WORD_IDENTITY_DRIFT",)) from error
    if request.schedule != expected:
        raise GymToraxRuntimePreflightError(("NATIVE_SCHEDULE_IDENTITY_DRIFT",))


def _validate_field_metadata_binding(
    request: GymToraxFieldMetadataEpisodeRequest,
    manifest: GymToraxFieldMetadataManifest,
) -> None:
    expected = ObjectIdentity.from_record(manifest.manifest_id, manifest)
    if request.field_metadata_manifest != expected:
        raise GymToraxRuntimePreflightError(("FIELD_METADATA_MANIFEST_DIVERGENCE",))


def _validate_extraction_binding(
    request: GymToraxFieldMetadataEpisodeRequest,
    manifest: GymToraxBoundedExtractionManifest,
) -> None:
    expected = ObjectIdentity.from_record(manifest.manifest_id, manifest)
    if request.extraction_manifest != expected:
        raise GymToraxRuntimePreflightError(("SURGICAL_EXTRACTION_MANIFEST_DIVERGENCE",))


def _scale_profile(profile: object, scale: float) -> object:
    result = copy.deepcopy(profile)
    if not isinstance(result, Mapping):
        raise TypeError("Gym--TORAX source profile is not a radial mapping")
    for radial_values in result.values():
        if not isinstance(radial_values, Mapping):
            raise TypeError("Gym--TORAX radial profile is not a mapping")
        mutable = cast(dict[object, object], radial_values)
        for radius in tuple(mutable):
            if float(cast(Any, radius)) < 1.0:
                mutable[radius] = float(cast(Any, mutable[radius])) * scale
    return result


def _build_gym_torax_environment_after_preflight(
    preparation: GymToraxPreparation,
    numerical_member: GymToraxNumericalMember,
) -> GymToraxEnvironment:
    from gymtorax.envs.iter_hybrid_env import (  # type: ignore[import-untyped]
        CONFIG,
        IterHybridEnv,
    )

    source_config = copy.deepcopy(CONFIG)
    source_config["numerics"]["t_final"] = GYM_TORAX_HORIZON_REQUESTS
    source_config["numerics"]["fixed_dt"] = float(numerical_member.internal_timestep_s)
    source_config["geometry"]["n_rho"] = numerical_member.radial_cells
    source_config["solver"]["n_corrector_steps"] = numerical_member.corrector_steps
    for field_id in ("T_i", "T_e"):
        source_config["profile_conditions"][field_id] = _scale_profile(
            source_config["profile_conditions"][field_id],
            float(preparation.initial_temperature_scale),
        )
    source_config["profile_conditions"]["nbar"] = float(preparation.initial_density_nbar)
    source_config["neoclassical"]["bootstrap_current"]["bootstrap_multiplier"] = float(
        preparation.bootstrap_multiplier
    )
    for field_id in ("D_e_inner", "chi_i_inner", "chi_e_inner"):
        source_config["transport"][field_id] = float(source_config["transport"][field_id]) * float(
            preparation.inner_transport_scale
        )

    class GymToraxEnvironment(IterHybridEnv):  # type: ignore[misc]
        def _get_torax_config(self) -> dict[str, Any]:
            return {
                "config": source_config,
                "discretization": "fixed",
                "ratio_a_sim": int(round(1.0 / float(numerical_member.internal_timestep_s))),
            }

        def _compute_reward(self, state: object, next_state: object, action: object) -> float:
            del state, next_state, action
            return 0.0

    return cast(
        GymToraxEnvironment,
        GymToraxEnvironment(render_mode=None, store_history=False, log_level="warning"),
    )


def build_gym_torax_environment(
    preparation: GymToraxPreparation,
    numerical_member: GymToraxNumericalMember,
    *,
    runtime_inspector: GymToraxRuntimeInspector = inspect_gym_torax_runtime,
) -> GymToraxEnvironment:
    """Construct the exact source environment only after denominator preflight."""

    _validate_runtime_probe(runtime_inspector())
    return _build_gym_torax_environment_after_preflight(preparation, numerical_member)


class GymToraxNativeActionCodec:
    """Exact native Gym--TORAX action translation without semantic aliases."""

    component_ids = _ACTION_COMPONENT_IDS

    @staticmethod
    def array(action: GymToraxNativeAction) -> npt.NDArray[np.float64]:
        return np.asarray(
            (
                float(action.ip_a),
                float(action.nbi_power_w),
                float(action.nbi_location),
                float(action.nbi_width),
                float(action.ecrh_power_w),
                float(action.ecrh_location),
                float(action.ecrh_width),
            ),
            dtype=np.float64,
        )

    @staticmethod
    def mapping(
        action: GymToraxNativeAction,
    ) -> dict[str, npt.NDArray[np.float64]]:
        values = GymToraxNativeActionCodec.array(action)
        return {
            "Ip": values[0:1].copy(),
            "NBI": values[1:4].copy(),
            "ECRH": values[4:7].copy(),
        }

    @staticmethod
    def decode(values: Mapping[str, object]) -> GymToraxNativeAction:
        ip = _finite_vector(values["Ip"], expected_size=1)
        nbi = _finite_vector(values["NBI"], expected_size=3)
        ecrh = _finite_vector(values["ECRH"], expected_size=3)
        return GymToraxNativeAction(
            ip_a=_decimal(ip[0]),
            nbi_power_w=_decimal(nbi[0]),
            nbi_location=_decimal(nbi[1]),
            nbi_width=_decimal(nbi[2]),
            ecrh_power_w=_decimal(ecrh[0]),
            ecrh_location=_decimal(ecrh[1]),
            ecrh_width=_decimal(ecrh[2]),
        )


def _decimal(value: np.float64) -> Decimal:
    return Decimal(str(float(value)))


def _finite_vector(value: object, *, expected_size: int) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != expected_size or not np.isfinite(array).all():
        raise ValueError("native action stage has invalid shape or nonfinite values")
    return array


def _as_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("Gym--TORAX source value is not a mapping")
    if not all(isinstance(key, str) for key in value):
        raise TypeError("Gym--TORAX source mapping has a non-string key")
    return cast(Mapping[str, object], value)


def _bounded_array(value: object) -> npt.NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.size > _MAX_CAPTURED_ARRAY_ELEMENTS:
        raise ValueError("source array exceeds the in-memory capture bound")
    return np.ascontiguousarray(array, dtype=np.float64)


def _native_numerics(environment: GymToraxEnvironment) -> Mapping[str, object]:
    app = getattr(environment, "torax_app", None)
    current = getattr(app, "current_sim_state", None)
    outputs = getattr(current, "solver_numeric_outputs", None)
    if outputs is None:
        return {}
    return {
        field_id: value
        for field_id in _NUMERICS_FIELDS
        if (value := getattr(outputs, field_id, None)) is not None
    }


def _source_dimensions(
    environment: GymToraxEnvironment,
) -> tuple[
    Mapping[tuple[str, str], tuple[str, ...]],
    Mapping[str, npt.NDArray[np.float64]],
]:
    provider = getattr(environment, 'source_dimensions', None)
    if callable(provider):
        raw_dimensions, raw_coordinates = provider()
        return cast(Mapping[tuple[str, str], tuple[str, ...]], raw_dimensions), {
            str(key): _bounded_array(value).reshape(-1)
            for key, value in _as_mapping(raw_coordinates).items()
        }
    app = getattr(environment, "torax_app", None)
    getter = getattr(app, "get_state_data", None)
    if not callable(getter):
        return {}, {}
    tree = getter()
    dimensions_by_field: dict[tuple[str, str], tuple[str, ...]] = {}
    for category in ("profiles", "scalars"):
        dataset = tree[f"/{category}/"].ds
        variables = dataset.variables
        coordinate_names = set(dataset.coords)
        for field_id, variable in variables.items():
            if field_id in coordinate_names:
                continue
            dimensions = tuple(str(value) for value in variable.dims)
            native_dimensions = dimensions[1:] if dimensions[:1] == ("time",) else dimensions
            dimensions_by_field[(category, str(field_id))] = (
                ("value",) if category == "scalars" and not native_dimensions else native_dimensions
            )
    coordinates = {
        str(name): _bounded_array(value.values).reshape(-1)
        for name, value in tree.coords.items()
        if str(name) in {"rho_norm", "rho_face_norm", "rho_cell_norm"}
    }
    return dimensions_by_field, coordinates


def capture_gym_torax_state(
    environment: GymToraxEnvironment,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
) -> GymToraxCapturedState:
    """Capture only the frozen retained fields and their source dimensions."""

    reasons: set[str] = set()
    state = _as_mapping(environment.state or {})
    profiles_raw = _as_mapping(state.get("profiles", {}))
    scalars_raw = _as_mapping(state.get("scalars", {}))
    dimensions_raw, coordinates = _source_dimensions(environment)
    dimensions = dict(dimensions_raw)
    retained = field_metadata_manifest.by_key()

    def capture_category(
        category: str,
        values: Mapping[str, object],
    ) -> dict[str, npt.NDArray[np.float64]]:
        result: dict[str, npt.NDArray[np.float64]] = {}
        expected_ids = sorted(
            field_id for source_category, field_id in retained if source_category == category
        )
        for field_id in expected_ids:
            if field_id not in values:
                reasons.add(f"SOURCE_FIELD_MISSING_{_reason_token(field_id)}")
                continue
            try:
                result[field_id] = _bounded_array(values[field_id])
            except (TypeError, ValueError, OverflowError):
                reasons.add(f"SOURCE_FIELD_INVALID_{_reason_token(field_id)}")
        return result

    numerics_raw = state.get("numerics")
    numerics = capture_category(
        "numerics",
        _as_mapping(numerics_raw) if numerics_raw is not None else _native_numerics(environment),
    )
    for field_id in numerics:
        dimensions.setdefault(("numerics", field_id), ("value",))
    return GymToraxCapturedState(
        profiles=capture_category("profiles", profiles_raw),
        scalars=capture_category("scalars", scalars_raw),
        numerics=numerics,
        dimensions=dimensions,
        coordinates=coordinates,
        reason_codes=tuple(sorted(reasons)),
    )


def _reason_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_") or "UNKNOWN"


def _stable_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def _environment_clock(environment: GymToraxEnvironment) -> int:
    value = float(cast(Any, environment.current_time))
    rounded = int(round(value))
    if not np.isfinite(value) or abs(value - rounded) > 1e-9:
        raise ValueError("native environment clock is not an exact integer")
    return rounded


def _decode_action_or_none(value: object) -> GymToraxNativeAction | None:
    try:
        return GymToraxNativeActionCodec.decode(_as_mapping(value))
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _applied_action(
    environment: GymToraxEnvironment, info: Mapping[str, object]
) -> GymToraxNativeAction | None:
    if "applied_action" in info:
        return _decode_action_or_none(info["applied_action"])
    app = getattr(environment, "torax_app", None)
    config = getattr(app, "config", None)
    getter = getattr(config, "get_current_action_values", None)
    return _decode_action_or_none(getter()) if callable(getter) else None


def _realized_action(
    environment: GymToraxEnvironment,
    info: Mapping[str, object],
    applied: GymToraxNativeAction | None,
) -> GymToraxNativeAction | None:
    if "realized_action" in info:
        return _decode_action_or_none(info["realized_action"])
    if applied is None:
        return None
    try:
        state = _as_mapping(environment.state or {})
        scalars = _as_mapping(state.get("scalars", {}))
        ip = _finite_vector(scalars["Ip"], expected_size=1)[0]
        nbi = _finite_vector(scalars["P_aux_generic_total"], expected_size=1)[0]
        ecrh = _finite_vector(scalars["P_ecrh_e"], expected_size=1)[0]
        return GymToraxNativeAction(
            ip_a=_decimal(ip),
            nbi_power_w=_decimal(nbi),
            nbi_location=applied.nbi_location,
            nbi_width=applied.nbi_width,
            ecrh_power_w=_decimal(ecrh),
            ecrh_location=applied.ecrh_location,
            ecrh_width=applied.ecrh_width,
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _operator_snapshot(
    environment: GymToraxEnvironment,
    *,
    request_clock: int,
    action_mapping: Mapping[str, npt.NDArray[np.float64]],
) -> GymToraxOperatorSnapshot | None:
    provider = _operator_provider(environment)
    if not callable(provider):
        return None
    raw = provider(request_clock, action_mapping)
    if isinstance(raw, GymToraxOperatorSnapshot):
        snapshot = raw
    else:
        values = _as_mapping(raw)
        snapshot = GymToraxOperatorSnapshot(
            request_clock=request_clock,
            source_method_id=str(values["source_method_id"]),
            state_coordinate_ids=tuple(cast(Sequence[str], values["state_coordinate_ids"])),
            input_coordinate_ids=tuple(cast(Sequence[str], values["input_coordinate_ids"])),
            receiver_coordinate_ids=tuple(cast(Sequence[str], values["receiver_coordinate_ids"])),
            transition_a=_bounded_array(values["A"]),
            input_b=_bounded_array(values["B"]),
            receiver_c=_bounded_array(values["C"]),
        )
    _validate_operator_snapshot(snapshot, expected_clock=request_clock)
    return snapshot


def _operator_provider(environment: GymToraxEnvironment) -> object:
    provider = getattr(environment, "capture_controlled_input_output_operators", None)
    if callable(provider):
        return provider
    app = getattr(environment, "torax_app", None)
    return getattr(app, "capture_controlled_input_output_operators", None)


def _validate_operator_snapshot(
    snapshot: GymToraxOperatorSnapshot, *, expected_clock: int
) -> None:
    if snapshot.request_clock != expected_clock:
        raise ValueError("operator snapshot changed its request clock")
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", snapshot.source_method_id):
        raise ValueError("operator source method identity is invalid")
    for values in (
        snapshot.state_coordinate_ids,
        snapshot.input_coordinate_ids,
        snapshot.receiver_coordinate_ids,
    ):
        if (
            not values
            or len(set(values)) != len(values)
            or any(not isinstance(value, str) or not value for value in values)
        ):
            raise ValueError("operator coordinate basis is empty or ambiguous")
    n_state = len(snapshot.state_coordinate_ids)
    n_input = len(snapshot.input_coordinate_ids)
    n_receiver = len(snapshot.receiver_coordinate_ids)
    if (
        snapshot.transition_a.shape != (n_state, n_state)
        or snapshot.input_b.shape != (n_state, n_input)
        or snapshot.receiver_c.shape != (n_receiver, n_state)
        or not np.isfinite(snapshot.transition_a).all()
        or not np.isfinite(snapshot.input_b).all()
        or not np.isfinite(snapshot.receiver_c).all()
    ):
        raise ValueError("operator operand shape or finiteness differs from its basis")


def _operator_blocks(
    snapshots: Sequence[GymToraxOperatorSnapshot],
    *,
    api_exposed: bool,
    output_budget: _GymToraxEncodedOutputBudget,
    reasons: set[str],
) -> tuple[tuple[GymToraxFieldMetadataFloat64Block, ...], GymToraxOperatorSourceSummary]:
    if not api_exposed:
        reasons.add("OPERATOR_API_UNAVAILABLE")
        return (), GymToraxOperatorSourceSummary(
            summary_id='operator-source.tokamak-control.unavailable',
            disposition=GymToraxSourceDisposition.OPERATOR_API_UNAVAILABLE,
            source_method_id="operator-source.gym-torax.unavailable",
            state_coordinate_ids=(),
            input_coordinate_ids=(),
            receiver_coordinate_ids=(),
            operator_block_ids=(),
            reason_codes=("OPERATOR_API_UNAVAILABLE",),
        )
    expected_clocks = tuple(range(104, 110))
    rosters = {
        (
            value.source_method_id,
            value.state_coordinate_ids,
            value.input_coordinate_ids,
            value.receiver_coordinate_ids,
        )
        for value in snapshots
    }
    if tuple(value.request_clock for value in snapshots) != expected_clocks or len(rosters) != 1:
        reasons.add("OPERATOR_SOURCE_INCOMPLETE")
        return (), GymToraxOperatorSourceSummary(
            summary_id='operator-source.tokamak-control.incomplete',
            disposition=GymToraxSourceDisposition.SOURCE_UNAVAILABLE,
            source_method_id="operator-source.gym-torax.incomplete",
            state_coordinate_ids=(),
            input_coordinate_ids=(),
            receiver_coordinate_ids=(),
            operator_block_ids=(),
            reason_codes=("OPERATOR_SOURCE_INCOMPLETE",),
        )
    source_method_id, state_ids, input_ids, receiver_ids = next(iter(rosters))
    clocks = tuple(value.request_clock for value in snapshots)
    blocks_list: list[GymToraxFieldMetadataFloat64Block] = []
    for name, attribute, dimensions in (
        ("A", "transition_a", ("state-row", "state-column")),
        ("B", "input_b", ("state-row", "input-column")),
        ("C", "receiver_c", ("receiver-row", "state-column")),
    ):
        arrays = [getattr(value, attribute) for value in snapshots]
        shape = (len(arrays), *arrays[0].shape)
        output_budget.reserve_shape(shape)
        blocks_list.append(
            encode_gym_torax_float64_block(
                block_id=f"block.operator-{name.lower()}",
                category="operator",
                native_field_id=name,
                native_unit="source-native",
                native_frame_id='frame.tokamak-control.controlled-io-operator',
                field_metadata_id=None,
                dimension_ids=("request-clock", *dimensions),
                values=np.stack(arrays),
                clock_values=clocks,
            )
        )
    blocks = tuple(sorted(blocks_list, key=lambda value: value.block_id))
    return blocks, GymToraxOperatorSourceSummary(
        summary_id='operator-source.tokamak-control.available',
        disposition=GymToraxSourceDisposition.AVAILABLE,
        source_method_id=source_method_id,
        state_coordinate_ids=state_ids,
        input_coordinate_ids=input_ids,
        receiver_coordinate_ids=receiver_ids,
        operator_block_ids=tuple(value.block_id for value in blocks),
        reason_codes=(),
    )


def _captured_state_blocks(
    captures: Sequence[GymToraxCapturedState],
    clocks: tuple[int, ...],
    *,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
    output_budget: _GymToraxEncodedOutputBudget,
    reasons: set[str],
) -> tuple[GymToraxFieldMetadataFloat64Block, ...]:
    blocks: list[GymToraxFieldMetadataFloat64Block] = []
    coordinates: dict[str, npt.NDArray[np.float64]] = {}
    dimensions_by_capture: list[Mapping[tuple[str, str], tuple[str, ...]]] = []
    for capture in captures:
        coordinates.update(capture.coordinates)
        dimensions_by_capture.append(capture.dimensions)
        reasons.update(capture.reason_codes)
    retained_coordinate_ids = {
        dimension_id
        for field in field_metadata_manifest.fields
        for dimension_id in field.native_dimension_ids
        if dimension_id != "value"
    }
    for coordinate_id in sorted(retained_coordinate_ids):
        if coordinate_id not in coordinates:
            reasons.add(f"SOURCE_COORDINATE_MISSING_{_reason_token(coordinate_id)}")
            continue
        values = coordinates[coordinate_id]
        output_budget.reserve_array(values)
        blocks.append(
            encode_gym_torax_float64_block(
                block_id=f"block.coordinate.{_stable_token(coordinate_id)}",
                category="coordinate",
                native_field_id=coordinate_id,
                native_unit="1",
                native_frame_id=f'frame.tokamak-control.{_stable_token(coordinate_id)}',
                field_metadata_id=None,
                dimension_ids=(coordinate_id,),
                values=values,
            )
        )
    for field_metadata in field_metadata_manifest.fields:
        category = field_metadata.source_category
        field_id = field_metadata.native_field_id
        category_id = {
            "profiles": "profile",
            "scalars": "scalar",
            "numerics": "numerics",
        }[category]
        observed = [getattr(capture, category).get(field_id) for capture in captures]
        exemplar = next((value for value in observed if value is not None), None)
        if exemplar is None:
            reasons.add(f"SOURCE_FIELD_MISSING_{_reason_token(field_id)}")
            continue
        normalized: list[npt.NDArray[np.float64]] = []
        for value in observed:
            if value is None or value.shape != exemplar.shape:
                reasons.add("SOURCE_FIELD_MISSING_OR_SHAPE_DRIFT")
                normalized.append(np.full(exemplar.shape, np.nan, dtype=np.float64))
            else:
                normalized.append(value)
        key = (category, field_id)
        if any(
            dimensions.get(key) != field_metadata.native_dimension_ids
            for dimensions in dimensions_by_capture
        ):
            reasons.add("SOURCE_FIELD_DIMENSION_METADATA_MISMATCH")
        if len(field_metadata.native_dimension_ids) != exemplar.ndim:
            reasons.add("SOURCE_FIELD_DIMENSION_METADATA_MISMATCH")
        stacked_shape = (len(normalized), *exemplar.shape)
        output_budget.reserve_shape(stacked_shape)
        blocks.append(
            encode_gym_torax_float64_block(
                block_id=f"block.source-{category_id}.{_stable_token(field_id)}",
                category=f"source-{category_id}",
                native_field_id=field_id,
                native_unit=field_metadata.native_unit,
                native_frame_id=field_metadata.native_frame_id,
                field_metadata_id=field_metadata.field_metadata_id,
                dimension_ids=("state-clock", *field_metadata.native_dimension_ids),
                values=np.stack(normalized),
                clock_values=clocks,
            )
        )
    return tuple(sorted(blocks, key=lambda value: value.block_id))


def _action_blocks(
    deliveries: Sequence[GymToraxActionDelivery],
    *,
    output_budget: _GymToraxEncodedOutputBudget,
) -> tuple[GymToraxFieldMetadataFloat64Block, ...]:
    clocks = tuple(value.request_clock for value in deliveries)
    blocks: list[GymToraxFieldMetadataFloat64Block] = []
    for stage in ("requested", "accepted", "applied", "realized"):
        values = []
        for delivery in deliveries:
            action = cast(GymToraxNativeAction | None, getattr(delivery, stage))
            values.append(
                GymToraxNativeActionCodec.array(action)
                if action is not None
                else np.full(len(_ACTION_COMPONENT_IDS), np.nan, dtype=np.float64)
            )
        if values:
            output_budget.reserve_shape((len(values), len(_ACTION_COMPONENT_IDS)))
            blocks.append(
                encode_gym_torax_float64_block(
                    block_id=f"block.action-{stage}",
                    category=f"action-{stage}",
                    native_field_id="native-action",
                    native_unit="mixed-native-see-action-component-contract",
                    native_frame_id='frame.tokamak-control.native-action-chart',
                    field_metadata_id=None,
                    dimension_ids=("request-clock", "action-component"),
                    values=np.stack(values),
                    clock_values=clocks,
                )
            )
    return tuple(sorted(blocks, key=lambda value: value.block_id))


def _peak_rss_bytes() -> int:
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _episode_id(request: GymToraxFieldMetadataEpisodeRequest) -> str:
    return f"episode.{request.request_id.removeprefix('request.')}"


def _empty_operator_summary(
    disposition: GymToraxSourceDisposition, reason: str
) -> GymToraxOperatorSourceSummary:
    return GymToraxOperatorSourceSummary(
        summary_id=f'operator-source.tokamak-control.{_stable_token(disposition.value)}',
        disposition=disposition,
        source_method_id=f"operator-source.gym-torax.{_stable_token(disposition.value)}",
        state_coordinate_ids=(),
        input_coordinate_ids=(),
        receiver_coordinate_ids=(),
        operator_block_ids=(),
        reason_codes=(reason,),
    )


def finalize_gym_torax_episode(
    *,
    request: GymToraxFieldMetadataEpisodeRequest,
    extraction_manifest: GymToraxBoundedExtractionManifest,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
    runtime_probe: GymToraxRuntimeProbe,
    source_reset_attempted: bool,
    state_clocks: Sequence[int],
    captures: Sequence[GymToraxCapturedState],
    deliveries: Sequence[GymToraxActionDelivery],
    operator_snapshots: Sequence[GymToraxOperatorSnapshot],
    operator_api_exposed: bool,
    reason_codes: Sequence[str],
    termination: bool,
    truncation: bool,
    runtime_seconds: Decimal,
    peak_rss_bytes: int,
) -> GymToraxFieldMetadataNativeEpisode:
    """Finalize captures from either ordinary or live controlled acquisition.

    This has no environment or action-selection authority.  It only applies
    the same bounded episode encoding and orthogonal disposition derivation as
    the ordinary source path to an already completed capture ledger.
    """

    _validate_exact_native_schedule(request)
    _validate_extraction_binding(request, extraction_manifest)
    _validate_field_metadata_binding(request, field_metadata_manifest)
    _validate_runtime_probe(runtime_probe)
    reasons = set(reason_codes)
    clocks = tuple(state_clocks)
    if clocks != tuple(sorted(set(clocks))):
        raise ValueError("episode finalization state clocks are not exact")
    if len(captures) != len(clocks):
        raise ValueError("episode finalization captures differ from state clocks")
    missing = tuple(value for value in range(GYM_TORAX_HORIZON_REQUESTS + 1) if value not in clocks)
    observation = GymToraxObservationDisposition.COMPLETE
    numerical = (
        GymToraxNumericalDisposition.SOLVER_FAILURE
        if "SOLVER_FAILURE" in reasons
        else GymToraxNumericalDisposition.VALID
    )
    if missing:
        reasons.add("REQUIRED_STATE_CLOCKS_ABSENT")
        observation = (
            GymToraxObservationDisposition.PARTIAL
            if clocks
            else GymToraxObservationDisposition.UNEVALUABLE
        )
    available_profiles = set().union(*(capture.profiles.keys() for capture in captures))
    available_scalars = set().union(*(capture.scalars.keys() for capture in captures))
    available_numerics = set().union(*(capture.numerics.keys() for capture in captures))
    if _REQUIRED_PROFILES - available_profiles:
        reasons.add("REQUIRED_PROFILE_OPERAND_ABSENT")
        observation = GymToraxObservationDisposition.PARTIAL if clocks else observation
    if _REQUIRED_SCALARS - available_scalars:
        reasons.add("REQUIRED_SCALAR_OPERAND_ABSENT")
        observation = GymToraxObservationDisposition.PARTIAL if clocks else observation
    if _REQUIRED_NUMERICS - available_numerics:
        reasons.add("SOLVER_ERROR_STATE_ABSENT")
        observation = GymToraxObservationDisposition.PARTIAL if clocks else observation
    if captures and any(
        _REQUIRED_PROFILES - capture.profiles.keys()
        or _REQUIRED_SCALARS - capture.scalars.keys()
        or _REQUIRED_NUMERICS - capture.numerics.keys()
        for capture in captures
    ):
        reasons.add("REQUIRED_FIELD_CLOCK_COVERAGE_INCOMPLETE")
        observation = GymToraxObservationDisposition.PARTIAL
    if any(
        not np.isfinite(value).all()
        for capture in captures
        for field_id, value in (*capture.profiles.items(), *capture.scalars.items())
        if field_id in _REQUIRED_PROFILES or field_id in _REQUIRED_SCALARS
    ):
        reasons.add("NONFINITE_REQUIRED_STATE")
        numerical = GymToraxNumericalDisposition.INVALID

    output_budget = _GymToraxEncodedOutputBudget(request.maximum_output_bytes)
    for delivery in deliveries:
        output_budget.reserve_record(delivery)
    state_blocks = (
        _captured_state_blocks(
            captures,
            clocks,
            field_metadata_manifest=field_metadata_manifest,
            output_budget=output_budget,
            reasons=reasons,
        )
        if captures
        else ()
    )
    if any(
        reason.startswith(("SOURCE_FIELD_", "SOURCE_COORDINATE_", "REQUIRED_FIELD_"))
        for reason in reasons
    ):
        observation = (
            GymToraxObservationDisposition.PARTIAL
            if clocks
            else GymToraxObservationDisposition.UNEVALUABLE
        )
    action_blocks = _action_blocks(deliveries, output_budget=output_budget)
    if not clocks:
        operator_blocks: tuple[GymToraxFieldMetadataFloat64Block, ...] = ()
        operator_summary = _empty_operator_summary(
            GymToraxSourceDisposition.SOURCE_UNAVAILABLE,
            "SOURCE_EPISODE_UNAVAILABLE",
        )
        reasons.add("SOURCE_EPISODE_UNAVAILABLE")
    else:
        operator_blocks, operator_summary = _operator_blocks(
            operator_snapshots,
            api_exposed=operator_api_exposed,
            output_budget=output_budget,
            reasons=reasons,
        )
    blocks = tuple(
        sorted((*state_blocks, *action_blocks, *operator_blocks), key=lambda value: value.block_id)
    )
    delivery_disposition = GymToraxDeliveryDisposition.COMPLETE
    observed_delivery_dispositions = {value.disposition for value in deliveries}
    for candidate in (
        GymToraxDeliveryDisposition.REJECTED,
        GymToraxDeliveryDisposition.PARTIAL,
        GymToraxDeliveryDisposition.CLIPPED,
    ):
        if candidate in observed_delivery_dispositions:
            delivery_disposition = candidate
            break
    if len(deliveries) != GYM_TORAX_HORIZON_REQUESTS:
        delivery_disposition = GymToraxDeliveryDisposition.PARTIAL
        reasons.add("DELIVERY_HORIZON_INCOMPLETE")

    episode = GymToraxFieldMetadataNativeEpisode(
        episode_id=_episode_id(request),
        request=ObjectIdentity.from_record(request.request_id, request),
        preparation=ObjectIdentity.from_record(
            request.preparation.preparation_id,
            request.preparation,
        ),
        numerical_member=ObjectIdentity.from_record(
            request.numerical_member.member_id,
            request.numerical_member,
        ),
        action_word=ObjectIdentity.from_record(
            request.schedule.action_word.word_id,
            request.schedule.action_word,
        ),
        extraction_manifest=request.extraction_manifest,
        field_metadata_manifest=request.field_metadata_manifest,
        source_reset_attempted=source_reset_attempted,
        state_clocks=clocks,
        missing_required_state_clocks=missing,
        last_valid_state_clock=max(clocks) if clocks else None,
        deliveries=tuple(sorted(deliveries, key=lambda value: value.delivery_id)),
        blocks=blocks,
        operator_source=operator_summary,
        source_disposition=(
            GymToraxSourceDisposition.AVAILABLE
            if any(value.category.startswith("source-") for value in blocks)
            else GymToraxSourceDisposition.SOURCE_UNAVAILABLE
        ),
        delivery_disposition=delivery_disposition,
        numerical_disposition=numerical,
        observation_disposition=observation,
        termination=termination,
        truncation=truncation,
        backend=runtime_probe.backend,
        precision="float64",
        python_version=runtime_probe.python_version,
        numpy_version=runtime_probe.numpy_version,
        scipy_version=runtime_probe.scipy_version,
        xarray_version=runtime_probe.xarray_version,
        gymtorax_version=runtime_probe.gymtorax_version,
        torax_version=runtime_probe.torax_version,
        jax_version=runtime_probe.jax_version,
        jaxlib_version=runtime_probe.jaxlib_version,
        runtime_seconds=runtime_seconds,
        peak_rss_bytes=peak_rss_bytes,
        reason_codes=tuple(sorted(reasons)),
        outcome_access=request.outcome_access,
        evidence_ceiling=request.evidence_ceiling,
    )
    serialized_size = len(episode.canonical_bytes())
    if serialized_size > request.maximum_output_bytes:
        raise GymToraxOutputBoundError("GYM_TORAX_OUTPUT_BOUND_EXCEEDED")
    return episode


def acquire_gym_torax_episode(
    request: GymToraxFieldMetadataEpisodeRequest,
    *,
    extraction_manifest: GymToraxBoundedExtractionManifest,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
    repository_root: Path,
    environment_factory: GymToraxEnvironmentFactory | None = None,
    runtime_inspector: GymToraxRuntimeInspector = inspect_gym_torax_runtime,
    progress_callback: GymToraxProgressCallback | None = None,
    source_reset_observer: GymToraxResetObserver | None = None,
) -> GymToraxFieldMetadataNativeEpisode:
    """Acquire one exact reset plus 120-request native episode.

    Runtime qualification always precedes environment construction.  All
    scientific outcomes remain source observations; this function neither
    retries nor adjudicates them.
    """

    _validate_exact_native_schedule(request)
    _validate_extraction_binding(request, extraction_manifest)
    _validate_field_metadata_binding(request, field_metadata_manifest)
    try:
        verify_gym_torax_extraction_manifest(
            extraction_manifest,
            expected_identity=request.extraction_manifest,
            repository_root=repository_root,
        )
    except (OSError, ValueError) as error:
        raise GymToraxRuntimePreflightError(("SURGICAL_EXTRACTION_SOURCE_DIVERGENCE",)) from error
    verify_installed_gym_torax_metadata_sources(field_metadata_manifest)
    probe = runtime_inspector()
    _validate_runtime_probe(probe)
    if request.maximum_output_bytes <= _MINIMUM_EPISODE_ENVELOPE_BYTES:
        raise GymToraxOutputBoundError("GYM_TORAX_OUTPUT_BOUND_EXCEEDED")
    factory = environment_factory or _build_gym_torax_environment_after_preflight
    started = time.monotonic()
    reasons: set[str] = set()
    state_clocks: list[int] = []
    captures: list[GymToraxCapturedState] = []
    deliveries: list[GymToraxActionDelivery] = []
    operator_snapshots: list[GymToraxOperatorSnapshot] = []
    operator_api_exposed = False
    environment: GymToraxEnvironment | None = None
    source_reset_attempted = False
    numerical = GymToraxNumericalDisposition.VALID
    observation = GymToraxObservationDisposition.COMPLETE
    termination = False
    truncation = False
    capture_reserved_bytes = _MINIMUM_EPISODE_ENVELOPE_BYTES
    output_bound_error: GymToraxOutputBoundError | None = None
    try:
        environment = factory(request.preparation, request.numerical_member)
        operator_api_exposed = callable(_operator_provider(environment))
        if source_reset_observer is not None:
            source_reset_observer()
        source_reset_attempted = True
        environment.reset(seed=request.preparation.environment_seed)
        if _environment_clock(environment) != 0:
            reasons.add("NATIVE_RESET_CLOCK_DIVERGENCE")
            observation = GymToraxObservationDisposition.UNEVALUABLE
        else:
            initial_capture = capture_gym_torax_state(
                environment,
                field_metadata_manifest,
            )
            capture_reserved_bytes = _reserve_capture_before_append(
                initial_capture,
                reserved_bytes=capture_reserved_bytes,
                maximum_bytes=request.maximum_output_bytes,
                include_coordinates=True,
            )
            captures.append(initial_capture)
            state_clocks.append(0)
            _emit_completed_clock_progress(progress_callback, 0)
        for row in request.schedule.rows:
            if observation is GymToraxObservationDisposition.UNEVALUABLE:
                break
            if _environment_clock(environment) != row.request_clock:
                reasons.add("NATIVE_CLOCK_DIVERGENCE")
                observation = GymToraxObservationDisposition.PARTIAL
                break
            action_mapping = GymToraxNativeActionCodec.mapping(row.action)
            if row.request_clock in _OPERATOR_REQUEST_CLOCKS:
                try:
                    snapshot = _operator_snapshot(
                        environment,
                        request_clock=row.request_clock,
                        action_mapping=action_mapping,
                    )
                    if snapshot is not None:
                        capture_reserved_bytes = _reserve_operator_before_append(
                            snapshot,
                            reserved_bytes=capture_reserved_bytes,
                            maximum_bytes=request.maximum_output_bytes,
                        )
                        operator_snapshots.append(snapshot)
                except (KeyError, TypeError, ValueError, OverflowError):
                    reasons.add("OPERATOR_SOURCE_INVALID")
            raw_step = environment.step(action_mapping)
            if not isinstance(raw_step, tuple) or len(raw_step) != 5:
                raise TypeError("Gym--TORAX step result is not the five-field Gym surface")
            _, reward_raw, terminated_raw, truncated_raw, info_raw = raw_step
            info = _as_mapping(info_raw)
            rejected = bool(info.get("action_rejected", False))
            accepted = (
                None
                if rejected
                else _decode_action_or_none(info["accepted_action"])
                if "accepted_action" in info
                else row.action
            )
            applied = _applied_action(environment, info)
            realized = _realized_action(environment, info, applied)
            clipped = bool(info.get("action_clipped", False))
            delivery_reasons: set[str] = set()
            if rejected:
                disposition = GymToraxDeliveryDisposition.REJECTED
                delivery_reasons.add("ACTION_REJECTED")
            elif accepted is None or applied is None or realized is None:
                disposition = GymToraxDeliveryDisposition.PARTIAL
                delivery_reasons.add("ACTION_STAGE_UNAVAILABLE")
            elif clipped:
                disposition = GymToraxDeliveryDisposition.CLIPPED
                delivery_reasons.add("ACTION_CLIPPED")
            else:
                disposition = GymToraxDeliveryDisposition.COMPLETE
            deliveries.append(
                GymToraxActionDelivery(
                    delivery_id=f"delivery.{request.request_id.removeprefix('request.')}.{row.request_clock:04d}",
                    request_clock=row.request_clock,
                    receiver_clock=row.request_clock + 1,
                    occurrence_id=row.controlled_occurrence_id,
                    requested=row.action,
                    accepted=accepted,
                    applied=applied,
                    realized=realized,
                    disposition=disposition,
                    reason_codes=tuple(sorted(delivery_reasons)),
                )
            )
            reasons.update(delivery_reasons)
            receiver_clock = row.request_clock + 1
            if _environment_clock(environment) != receiver_clock:
                reasons.add("NATIVE_RECEIVER_CLOCK_DIVERGENCE")
                observation = GymToraxObservationDisposition.PARTIAL
                break
            capture = capture_gym_torax_state(
                environment,
                field_metadata_manifest,
            )
            capture_reserved_bytes = _reserve_capture_before_append(
                capture,
                reserved_bytes=capture_reserved_bytes,
                maximum_bytes=request.maximum_output_bytes,
                include_coordinates=False,
            )
            captures.append(capture)
            state_clocks.append(receiver_clock)
            _emit_completed_clock_progress(progress_callback, receiver_clock)
            decisive_scalars = {
                field_id: value
                for field_id, value in capture.scalars.items()
                if field_id in _REQUIRED_SCALARS
            }
            decisive_profiles = {
                field_id: value
                for field_id, value in capture.profiles.items()
                if field_id in _REQUIRED_PROFILES
            }
            if any(
                not np.isfinite(value).all()
                for value in (*decisive_scalars.values(), *decisive_profiles.values())
            ):
                numerical = GymToraxNumericalDisposition.INVALID
                reasons.add("NONFINITE_REQUIRED_STATE")
            solver_error = capture.numerics.get("solver_error_state")
            if float(cast(Any, reward_raw)) == -1000.0 or (
                solver_error is not None and np.any(solver_error != 0.0)
            ):
                numerical = GymToraxNumericalDisposition.SOLVER_FAILURE
                reasons.add("SOLVER_FAILURE")
            terminated_now = bool(terminated_raw)
            truncated_now = bool(truncated_raw)
            if terminated_now or truncated_now:
                if receiver_clock < GYM_TORAX_HORIZON_REQUESTS:
                    termination = terminated_now
                    truncation = truncated_now
                    reasons.add(
                        "SIMULATOR_TERMINATED_BEFORE_HORIZON"
                        if terminated_now
                        else "SIMULATOR_TRUNCATED_BEFORE_HORIZON"
                    )
                    observation = GymToraxObservationDisposition.PARTIAL
                break
    except GymToraxProgressEmissionError:
        raise
    except GymToraxOutputBoundError as error:
        output_bound_error = error
    except Exception as error:
        reasons.add(
            f"SIMULATOR_{type(error).__name__.upper()}"
            if state_clocks
            else f"TECHNICAL_INITIALIZATION_{type(error).__name__.upper()}"
        )
        if state_clocks:
            numerical = GymToraxNumericalDisposition.SOLVER_FAILURE
            observation = GymToraxObservationDisposition.PARTIAL
        else:
            numerical = GymToraxNumericalDisposition.UNEVALUABLE
            observation = GymToraxObservationDisposition.UNEVALUABLE
    finally:
        if environment is not None:
            try:
                environment.close()
            except Exception as error:
                reasons.add(f"CLEANUP_{type(error).__name__.upper()}")
    if output_bound_error is not None:
        raise output_bound_error

    clocks = tuple(state_clocks)
    missing = tuple(value for value in range(GYM_TORAX_HORIZON_REQUESTS + 1) if value not in clocks)
    if missing:
        reasons.add("REQUIRED_STATE_CLOCKS_ABSENT")
        observation = (
            GymToraxObservationDisposition.PARTIAL
            if clocks
            else GymToraxObservationDisposition.UNEVALUABLE
        )
    available_profiles = set().union(*(capture.profiles.keys() for capture in captures))
    available_scalars = set().union(*(capture.scalars.keys() for capture in captures))
    available_numerics = set().union(*(capture.numerics.keys() for capture in captures))
    if _REQUIRED_PROFILES - available_profiles:
        reasons.add("REQUIRED_PROFILE_OPERAND_ABSENT")
        observation = GymToraxObservationDisposition.PARTIAL if clocks else observation
    if _REQUIRED_SCALARS - available_scalars:
        reasons.add("REQUIRED_SCALAR_OPERAND_ABSENT")
        observation = GymToraxObservationDisposition.PARTIAL if clocks else observation
    if _REQUIRED_NUMERICS - available_numerics:
        reasons.add("SOLVER_ERROR_STATE_ABSENT")
        observation = GymToraxObservationDisposition.PARTIAL if clocks else observation
    if captures and any(
        _REQUIRED_PROFILES - capture.profiles.keys()
        or _REQUIRED_SCALARS - capture.scalars.keys()
        or _REQUIRED_NUMERICS - capture.numerics.keys()
        for capture in captures
    ):
        reasons.add("REQUIRED_FIELD_CLOCK_COVERAGE_INCOMPLETE")
        observation = GymToraxObservationDisposition.PARTIAL
    if any(
        not np.isfinite(value).all()
        for capture in captures
        for field_id, value in (*capture.profiles.items(), *capture.scalars.items())
        if field_id in _REQUIRED_PROFILES or field_id in _REQUIRED_SCALARS
    ):
        reasons.add("NONFINITE_REQUIRED_STATE")
        numerical = GymToraxNumericalDisposition.INVALID

    output_budget = _GymToraxEncodedOutputBudget(request.maximum_output_bytes)
    for delivery in deliveries:
        output_budget.reserve_record(delivery)
    state_blocks = (
        _captured_state_blocks(
            captures,
            clocks,
            field_metadata_manifest=field_metadata_manifest,
            output_budget=output_budget,
            reasons=reasons,
        )
        if captures
        else ()
    )
    if any(
        reason.startswith(("SOURCE_FIELD_", "SOURCE_COORDINATE_", "REQUIRED_FIELD_"))
        for reason in reasons
    ):
        observation = (
            GymToraxObservationDisposition.PARTIAL
            if clocks
            else GymToraxObservationDisposition.UNEVALUABLE
        )
    action_blocks = _action_blocks(deliveries, output_budget=output_budget)
    operator_blocks: tuple[GymToraxFieldMetadataFloat64Block, ...]
    if not clocks:
        operator_blocks = ()
        operator_summary = _empty_operator_summary(
            GymToraxSourceDisposition.SOURCE_UNAVAILABLE,
            "SOURCE_EPISODE_UNAVAILABLE",
        )
        reasons.add("SOURCE_EPISODE_UNAVAILABLE")
    else:
        operator_blocks, operator_summary = _operator_blocks(
            operator_snapshots,
            api_exposed=operator_api_exposed,
            output_budget=output_budget,
            reasons=reasons,
        )
    blocks = tuple(
        sorted((*state_blocks, *action_blocks, *operator_blocks), key=lambda value: value.block_id)
    )
    delivery_disposition = GymToraxDeliveryDisposition.COMPLETE
    observed_delivery_dispositions = {value.disposition for value in deliveries}
    for candidate in (
        GymToraxDeliveryDisposition.REJECTED,
        GymToraxDeliveryDisposition.PARTIAL,
        GymToraxDeliveryDisposition.CLIPPED,
    ):
        if candidate in observed_delivery_dispositions:
            delivery_disposition = candidate
            break
    if len(deliveries) != GYM_TORAX_HORIZON_REQUESTS:
        delivery_disposition = GymToraxDeliveryDisposition.PARTIAL
        reasons.add("DELIVERY_HORIZON_INCOMPLETE")

    episode = GymToraxFieldMetadataNativeEpisode(
        episode_id=_episode_id(request),
        request=ObjectIdentity.from_record(request.request_id, request),
        preparation=ObjectIdentity.from_record(
            request.preparation.preparation_id, request.preparation
        ),
        numerical_member=ObjectIdentity.from_record(
            request.numerical_member.member_id, request.numerical_member
        ),
        action_word=ObjectIdentity.from_record(
            request.schedule.action_word.word_id, request.schedule.action_word
        ),
        extraction_manifest=request.extraction_manifest,
        field_metadata_manifest=request.field_metadata_manifest,
        source_reset_attempted=source_reset_attempted,
        state_clocks=clocks,
        missing_required_state_clocks=missing,
        last_valid_state_clock=max(clocks) if clocks else None,
        deliveries=tuple(sorted(deliveries, key=lambda value: value.delivery_id)),
        blocks=blocks,
        operator_source=operator_summary,
        source_disposition=(
            GymToraxSourceDisposition.AVAILABLE
            if any(value.category.startswith("source-") for value in blocks)
            else GymToraxSourceDisposition.SOURCE_UNAVAILABLE
        ),
        delivery_disposition=delivery_disposition,
        numerical_disposition=numerical,
        observation_disposition=observation,
        termination=termination,
        truncation=truncation,
        backend=probe.backend,
        precision="float64",
        python_version=probe.python_version,
        numpy_version=probe.numpy_version,
        scipy_version=probe.scipy_version,
        xarray_version=probe.xarray_version,
        gymtorax_version=probe.gymtorax_version,
        torax_version=probe.torax_version,
        jax_version=probe.jax_version,
        jaxlib_version=probe.jaxlib_version,
        runtime_seconds=Decimal(f"{time.monotonic() - started:.9f}"),
        peak_rss_bytes=_peak_rss_bytes(),
        reason_codes=tuple(sorted(reasons)),
        outcome_access=request.outcome_access,
        evidence_ceiling=request.evidence_ceiling,
    )
    serialized_size = len(episode.canonical_bytes())
    if serialized_size > request.maximum_output_bytes:
        raise GymToraxOutputBoundError("GYM_TORAX_OUTPUT_BOUND_EXCEEDED")
    return episode


__all__ = [
    'GymToraxEnvironmentFactory',
    'GymToraxCapturedState',
    'GymToraxEnvironment',
    'GymToraxNativeActionCodec',
    'GymToraxOperatorSnapshot',
    "GymToraxOutputBoundError",
    "GymToraxProgressEmissionError",
    "GymToraxRuntimePreflightError",
    'GymToraxRuntimeProbe',
    'GymToraxRuntimeInspector',
    'acquire_gym_torax_episode',
    'build_gym_torax_environment',
    'capture_gym_torax_state',
    'finalize_gym_torax_episode',
    'inspect_gym_torax_runtime',
]
