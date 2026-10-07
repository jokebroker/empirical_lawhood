"""Bounded preopened-member verifier/decoder for the physical scale morphism draft profile."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import TypeAlias
from zipfile import ZipFile

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import validate_relative_locator

from .contracts import RcLadderResponseActionDeliveryStatus, RcLadderResponseBundleManifest, RcLadderResponseChannelCalibration, RcLadderResponseEpisodeManifest, RcLadderResponsePreparationRecord
from .profile import RcLadderResponsePhysicalSemanticProfile, RcLadderResponseStatusMeaning


FloatArray: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int16]
UIntArray: TypeAlias = npt.NDArray[np.uint8]
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_MEMBER_BYTES = 128 * 1024 * 1024
MAX_UNCOMPRESSED_MEMBER_BYTES = 512 * 1024 * 1024

_ARRAY_KEYS = (
    "accepted_action_volts",
    "applied_action_volts",
    "environment_values",
    "node_voltages_volts",
    "observation_validity",
    "realized_action_volts",
    "requested_action_volts",
    "status_codes",
    "terminal_currents_amperes",
    "times_seconds",
)


@dataclass(frozen=True, slots=True)
class RcLadderResponsePreopenedMember:
    relative_path: str
    payload: bytes

    def __post_init__(self) -> None:
        validate_relative_locator(self.relative_path)
        if not isinstance(self.payload, bytes) or not 0 < len(self.payload) <= MAX_MEMBER_BYTES:
            raise ValueError("preopened physical scale morphism member exceeds its byte contract")


@dataclass(frozen=True, slots=True)
class RcLadderResponseDecodedEpisode:
    manifest: RcLadderResponseEpisodeManifest
    times_seconds: FloatArray
    node_voltages_volts: FloatArray
    terminal_currents_amperes: FloatArray
    environment_values: FloatArray
    observation_validity: UIntArray
    requested_action_volts: FloatArray
    accepted_action_volts: FloatArray
    applied_action_volts: FloatArray
    realized_action_volts: FloatArray
    status_codes: IntArray


@dataclass(frozen=True, slots=True)
class RcLadderResponseDecodedBundle:
    manifest: RcLadderResponseBundleManifest
    episodes: tuple[RcLadderResponseDecodedEpisode, ...]


class RcLadderResponseBundleDecodeError(ValueError):
    """The physical bundle failed an exact source/observation contract."""


def _inspect_npz(payload: bytes) -> None:
    try:
        with ZipFile(BytesIO(payload)) as archive:
            entries = archive.infolist()
            expected_names = {f"{value}.npy" for value in _ARRAY_KEYS}
            if {value.filename for value in entries} != expected_names:
                raise RcLadderResponseBundleDecodeError("NPZ member key roster differs")
            if any(
                "/" in value.filename or "\\" in value.filename or value.filename.startswith(".")
                for value in entries
            ):
                raise RcLadderResponseBundleDecodeError("NPZ contains a path-bearing member")
            total = sum(value.file_size for value in entries)
            if total > MAX_UNCOMPRESSED_MEMBER_BYTES:
                raise RcLadderResponseBundleDecodeError("NPZ uncompressed size exceeds its bound")
            if any(value.compress_size == 0 and value.file_size for value in entries):
                raise RcLadderResponseBundleDecodeError("NPZ contains an invalid compressed member")
    except RcLadderResponseBundleDecodeError:
        raise
    except Exception as error:
        raise RcLadderResponseBundleDecodeError(f"NPZ inspection failed: {error}") from error


def _load_arrays(payload: bytes) -> dict[str, np.ndarray]:
    _inspect_npz(payload)
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in _ARRAY_KEYS}
    except Exception as error:
        raise RcLadderResponseBundleDecodeError(f"safe NumPy decoding failed: {error}") from error
    for key, value in arrays.items():
        expected_dtype = {
            "status_codes": np.dtype("<i2"),
            "observation_validity": np.dtype("u1"),
        }.get(key, np.dtype("<f8"))
        if value.dtype != expected_dtype:
            raise RcLadderResponseBundleDecodeError(f"{key} dtype differs from the draft profile")
        if not value.flags.c_contiguous:
            raise RcLadderResponseBundleDecodeError(f"{key} must be C-contiguous")
        if key in {
            "accepted_action_volts",
            "applied_action_volts",
            "realized_action_volts",
            "requested_action_volts",
            "times_seconds",
        } and not np.all(np.isfinite(value)):
            raise RcLadderResponseBundleDecodeError(f"{key} contains nonfinite values")
    return arrays


def _validate_action_stages(
    *,
    episode: RcLadderResponseEpisodeManifest,
    times: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> None:
    ledger = episode.action_ledger
    stages = (
        ("requested_action_volts", ledger.requested_action_id, ledger.request_time_seconds),
        ("accepted_action_volts", ledger.accepted_action_id, ledger.acceptance_time_seconds),
        ("applied_action_volts", ledger.applied_action_id, ledger.application_time_seconds),
        ("realized_action_volts", ledger.realized_action_id, ledger.realization_time_seconds),
    )
    for key, action_id, clock in stages:
        stream = arrays[key]
        if action_id is None:
            if np.any(stream != 0):
                raise RcLadderResponseBundleDecodeError(f"absent {key} ledger stage has sampled values")
            continue
        assert clock is not None
        if np.any(stream[times < float(clock)] != 0):
            raise RcLadderResponseBundleDecodeError(f"{key} precedes its native ledger clock")
    stage_keys = tuple(value[0] for value in stages)
    for upstream_key, downstream_key in zip(stage_keys, stage_keys[1:]):
        downstream_support = np.any(arrays[downstream_key] != 0, axis=1)
        upstream_support = np.any(arrays[upstream_key] != 0, axis=1)
        if np.any(downstream_support & ~upstream_support):
            raise RcLadderResponseBundleDecodeError(
                f"{downstream_key} contains support absent from {upstream_key}"
            )


def _observed_matrix(
    *,
    episode: RcLadderResponseEpisodeManifest,
    arrays: dict[str, np.ndarray],
    calibration: RcLadderResponseChannelCalibration,
) -> np.ndarray:
    by_channel: dict[str, np.ndarray] = {
        f"node-{index:03d}-voltage": arrays["node_voltages_volts"][:, index]
        for index in range(arrays["node_voltages_volts"].shape[1])
    }
    by_channel.update(
        {
            "terminal-left-current": arrays["terminal_currents_amperes"][:, 0],
            "terminal-right-current": arrays["terminal_currents_amperes"][:, 1],
        }
    )
    by_channel.update(
        {
            channel_id: arrays["environment_values"][:, index]
            for index, channel_id in enumerate(episode.environment_channel_ids)
        }
    )
    if set(by_channel) != set(calibration.channel_ids):
        raise RcLadderResponseBundleDecodeError("sampled and calibrated observed-channel rosters differ")
    return np.asarray(
        np.column_stack(tuple(by_channel[value] for value in calibration.channel_ids))
    )


def _validate_observation_validity(
    *,
    episode: RcLadderResponseEpisodeManifest,
    arrays: dict[str, np.ndarray],
    calibration: RcLadderResponseChannelCalibration,
    profile: RcLadderResponsePhysicalSemanticProfile,
) -> None:
    validity = arrays["observation_validity"]
    if np.any((validity != 0) & (validity != 1)):
        raise RcLadderResponseBundleDecodeError("observation validity contains a non-binary code")
    observed = _observed_matrix(
        episode=episode,
        arrays=arrays,
        calibration=calibration,
    )
    finite = np.isfinite(observed)
    if np.any(~finite & (validity != 0)):
        raise RcLadderResponseBundleDecodeError("nonfinite observation lacks typed invalidity")
    codebook = {value.native_code: value for value in profile.status_codebook}
    invalid_rows = np.asarray(
        [codebook[int(value)].invalidates_observation for value in arrays["status_codes"]],
        dtype=bool,
    )
    if np.any((validity == 0).any(axis=1) != invalid_rows):
        raise RcLadderResponseBundleDecodeError("status invalidity and observation-validity mask disagree")
    lower = np.asarray(tuple(map(float, calibration.range_lower_native)))
    upper = np.asarray(tuple(map(float, calibration.range_upper_native)))
    outside = finite & ((observed < lower) | (observed > upper))
    if np.any(outside & (validity != 0)):
        raise RcLadderResponseBundleDecodeError("valid observation lies outside its calibrated range")


def _decode_episode(
    episode: RcLadderResponseEpisodeManifest,
    *,
    preparation: RcLadderResponsePreparationRecord,
    profile: RcLadderResponsePhysicalSemanticProfile,
    calibration: RcLadderResponseChannelCalibration,
    scale_cells: int,
    payload: bytes,
) -> RcLadderResponseDecodedEpisode:
    arrays = _load_arrays(payload)
    count = episode.sample_count
    shapes = {
        "times_seconds": (count,),
        "node_voltages_volts": (count, scale_cells),
        "terminal_currents_amperes": (count, 2),
        "environment_values": (count, len(episode.environment_channel_ids)),
        "observation_validity": (count, len(calibration.channel_ids)),
        "requested_action_volts": (count, 2),
        "accepted_action_volts": (count, 2),
        "applied_action_volts": (count, 2),
        "realized_action_volts": (count, 2),
        "status_codes": (count,),
    }
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise RcLadderResponseBundleDecodeError(f"{key} shape differs from the manifest")
    expected_channels = tuple(
        sorted(
            (
                *(f"node-{index:03d}-voltage" for index in range(scale_cells)),
                "terminal-left-current",
                "terminal-right-current",
            )
        )
    )
    if episode.channel_ids != expected_channels:
        raise RcLadderResponseBundleDecodeError("channel roster/order differs from the board scale")
    if episode.environment_channel_ids != profile.environment_channel_ids:
        raise RcLadderResponseBundleDecodeError("environment channel roster differs from the profile")
    unit_by_channel = {
        **{value: profile.voltage_unit for value in expected_channels if value.startswith("node-")},
        "terminal-left-current": profile.current_unit,
        "terminal-right-current": profile.current_unit,
        **dict(
            zip(
                profile.environment_channel_ids,
                profile.environment_channel_units,
                strict=True,
            )
        ),
    }
    if calibration.native_units != tuple(
        unit_by_channel[value] for value in calibration.channel_ids
    ):
        raise RcLadderResponseBundleDecodeError("channel calibration units differ from the profile")
    if (
        episode.voltage_unit,
        episode.current_unit,
        episode.clock_unit,
        episode.gauge_id,
        episode.terminal_current_sign_id,
    ) != (
        profile.voltage_unit,
        profile.current_unit,
        profile.clock_unit,
        profile.gauge_id,
        profile.terminal_current_sign_id,
    ):
        raise RcLadderResponseBundleDecodeError("episode units/gauge/sign differ from the semantic profile")
    times = arrays["times_seconds"]
    if times[0] != 0 or np.any(np.diff(times) <= 0):
        raise RcLadderResponseBundleDecodeError("native clock must start at zero and increase strictly")
    window_start, window_end = map(float, episode.receiver_window_seconds)
    if (
        times[-1] < window_end
        or not times[0] <= float(episode.causal_cutoff_seconds) <= window_start
    ):
        raise RcLadderResponseBundleDecodeError("sample clock does not cover cutoff/receiver window")
    codebook = {value.native_code: value for value in profile.status_codebook}
    unknown_codes = set(int(value) for value in np.unique(arrays["status_codes"])) - set(codebook)
    if unknown_codes:
        raise RcLadderResponseBundleDecodeError("episode contains a status code absent from the profile")
    observed_meanings = {
        codebook[int(value)].meaning for value in np.unique(arrays["status_codes"])
    }
    clipping_observed = RcLadderResponseStatusMeaning.CLIPPING in observed_meanings
    interlock_observed = RcLadderResponseStatusMeaning.INTERLOCK in observed_meanings
    if episode.action_ledger.clipping_observed != clipping_observed:
        raise RcLadderResponseBundleDecodeError("clipping flag and exact status codebook disagree")
    if episode.action_ledger.interlock_observed != interlock_observed:
        raise RcLadderResponseBundleDecodeError("interlock flag and exact status codebook disagree")
    _validate_action_stages(episode=episode, times=times, arrays=arrays)
    _validate_observation_validity(
        episode=episode,
        arrays=arrays,
        calibration=calibration,
        profile=profile,
    )
    if (
        episode.action_ledger.delivery_status is RcLadderResponseActionDeliveryStatus.REJECTED
        and RcLadderResponseStatusMeaning.NON_DELIVERY not in observed_meanings
    ):
        raise RcLadderResponseBundleDecodeError("rejected action lacks its non-delivery status code")
    if episode.is_hold:
        if episode.action_ledger.delivery_status is not RcLadderResponseActionDeliveryStatus.REALIZED:
            raise RcLadderResponseBundleDecodeError("matched hold was not separately realized")
        for key in (
            "requested_action_volts",
            "accepted_action_volts",
            "applied_action_volts",
            "realized_action_volts",
        ):
            if np.any(arrays[key] != 0):
                raise RcLadderResponseBundleDecodeError("hold episode contains a nonzero action stage")
    if episode.action_ledger.delivery_status is RcLadderResponseActionDeliveryStatus.REJECTED:
        if np.any(arrays["applied_action_volts"] != 0) or np.any(
            arrays["realized_action_volts"] != 0
        ):
            raise RcLadderResponseBundleDecodeError("rejected action has applied/realized values")
    calibration_index = {value: index for index, value in enumerate(calibration.channel_ids)}
    initial_node_validity = np.asarray(
        [
            arrays["observation_validity"][0, calibration_index[value]]
            for value in expected_channels
            if value.startswith("node-")
        ]
    )
    if np.any(initial_node_validity != 1) or not np.all(
        np.isfinite(arrays["node_voltages_volts"][0])
    ):
        raise RcLadderResponseBundleDecodeError("waveform initial state is not a valid observation")
    initial_defect = np.abs(
        arrays["node_voltages_volts"][0]
        - np.asarray(tuple(map(float, preparation.realized_node_voltages_volts)))
    )
    if np.any(initial_defect > float(preparation.voltage_uncertainty_volts)):
        raise RcLadderResponseBundleDecodeError("waveform initial state differs from the realized preparation")
    return RcLadderResponseDecodedEpisode(
        manifest=episode,
        times_seconds=np.asarray(times, dtype=np.float64),
        node_voltages_volts=np.asarray(arrays["node_voltages_volts"], dtype=np.float64),
        terminal_currents_amperes=np.asarray(arrays["terminal_currents_amperes"], dtype=np.float64),
        environment_values=np.asarray(arrays["environment_values"], dtype=np.float64),
        observation_validity=np.asarray(arrays["observation_validity"], dtype=np.uint8),
        requested_action_volts=np.asarray(arrays["requested_action_volts"], dtype=np.float64),
        accepted_action_volts=np.asarray(arrays["accepted_action_volts"], dtype=np.float64),
        applied_action_volts=np.asarray(arrays["applied_action_volts"], dtype=np.float64),
        realized_action_volts=np.asarray(arrays["realized_action_volts"], dtype=np.float64),
        status_codes=np.asarray(arrays["status_codes"], dtype=np.int16),
    )


def decode_bundle(
    *,
    manifest_payload: bytes,
    members: tuple[RcLadderResponsePreopenedMember, ...],
    semantic_profile: RcLadderResponsePhysicalSemanticProfile,
) -> RcLadderResponseDecodedBundle:
    """Verify exact preopened bytes and decode without choosing a filesystem path."""

    try:
        manifest = decode_canonical_bytes(
            manifest_payload,
            RcLadderResponseBundleManifest,
            maximum_bytes=min(MAX_MANIFEST_BYTES, semantic_profile.maximum_manifest_bytes),
        )
    except Exception as error:
        raise RcLadderResponseBundleDecodeError(f"manifest decoding failed: {error}") from error
    profile_identity = ObjectIdentity.from_record(semantic_profile.profile_id, semantic_profile)
    if manifest.semantic_profile != profile_identity:
        raise RcLadderResponseBundleDecodeError("bundle semantic-profile identity differs")
    if manifest.board.scale_cells not in semantic_profile.allowed_scale_cells:
        raise RcLadderResponseBundleDecodeError("bundle scale lies outside the semantic profile")
    if semantic_profile.exact_dossier_bound and manifest.dossier != semantic_profile.dossier:
        raise RcLadderResponseBundleDecodeError("bundle dossier differs from the exact semantic profile")
    by_path = {value.relative_path: value.payload for value in members}
    if len(by_path) != len(members):
        raise RcLadderResponseBundleDecodeError("preopened member path repeats")
    expected_paths = {value.member_relative_path for value in manifest.episodes}
    if set(by_path) != expected_paths:
        raise RcLadderResponseBundleDecodeError("preopened member roster differs from the manifest")
    decoded = []
    calibration_by_identity = {
        ObjectIdentity.from_record(value.calibration_id, value): value
        for value in manifest.channel_calibrations
    }
    for episode in manifest.episodes:
        payload = by_path[episode.member_relative_path]
        if len(payload) > semantic_profile.maximum_member_bytes:
            raise RcLadderResponseBundleDecodeError("episode member exceeds its semantic-profile bound")
        if len(payload) != episode.member_size_bytes:
            raise RcLadderResponseBundleDecodeError("episode member size differs from the manifest")
        if sha256(payload).hexdigest() != episode.member_sha256:
            raise RcLadderResponseBundleDecodeError("episode member hash differs from the manifest")
        decoded.append(
            _decode_episode(
                episode,
                preparation=next(
                    value
                    for value in manifest.preparations
                    if ObjectIdentity.from_record(value.preparation_id, value)
                    == episode.preparation
                ),
                profile=semantic_profile,
                calibration=calibration_by_identity[episode.channel_calibration],
                scale_cells=manifest.board.scale_cells,
                payload=payload,
            )
        )
    return RcLadderResponseDecodedBundle(
        manifest=manifest,
        episodes=tuple(sorted(decoded, key=lambda value: value.manifest.episode_id)),
    )


__all__ = [
    'RcLadderResponseBundleDecodeError',
    'RcLadderResponseDecodedBundle',
    'RcLadderResponseDecodedEpisode',
    'RcLadderResponsePreopenedMember',
    "decode_bundle",
]
