"""Synthetic draft-profile qualification for the read-only physical adapter.

This module creates no filesystem paths and performs no physical I/O.  Its
positive fixture is generated entirely in memory, while deliberate corruptions
exercise the same strict decoder used for a future dossier-bound source.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
import json
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.evidence import EvidenceCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .contracts import RcLadderResponseActionDeliveryStatus, RcLadderResponseActionLedger, RcLadderResponseBoardIdentity, RcLadderResponseBundleManifest, RcLadderResponseChannelCalibration, RcLadderResponseClockFitRecord, RcLadderResponseComponentMetrology, RcLadderResponseEpisodeManifest, RcLadderResponseEpisodeRole, RcLadderResponsePreparationRecord
from .profile import RcLadderResponsePhysicalSemanticProfile, draft_semantic_profile
from .source import RcLadderResponseBundleDecodeError, RcLadderResponsePreopenedMember, decode_bundle


PHYSICAL_SCALE_MORPHISM_SYNTHETIC_SOURCE_COUNTERFEIT_IDS = (
    "counterfeit.action-delivery-stage-contradiction",
    "counterfeit.channel-calibration-roster-drift",
    "counterfeit.channel-calibration-unit-drift",
    "counterfeit.channel-roster-drift",
    "counterfeit.clock-fit-linkage-drift",
    "counterfeit.component-roster-drift",
    "counterfeit.interlock-without-status-codebook",
    "counterfeit.member-hash-drift",
    "counterfeit.member-roster-drift",
    "counterfeit.metrology-uncertainty-roster-drift",
    "counterfeit.native-clock-nonmonotone",
    "counterfeit.preparation-dimension-drift",
    "counterfeit.previous-episode-linkage-drift",
    "counterfeit.terminal-current-sign-drift",
    "counterfeit.unit-contract-drift",
    "counterfeit.unknown-manifest-field",
    "counterfeit.unsafe-relative-path",
)


@dataclass(frozen=True, slots=True)
class RcLadderResponseSyntheticSourceQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/rc-ladder-response/rc-ladder-response-synthetic-source-qualification'

    qualification_id: str
    semantic_profile_id: str
    semantic_profile_sha256: str
    positive_episode_count: int
    rejected_counterfeit_ids: tuple[str, ...]
    path_selection_count: int
    source_write_count: int
    instrument_command_count: int
    actuator_count: int
    draft_profile_qualified: bool
    exact_dossier_profile_qualified: bool
    physical_evidence_count: int
    reason_codes: tuple[str, ...]
    scientific_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        validate_stable_id(self.semantic_profile_id, field_name="semantic_profile_id")
        validate_sha256(self.semantic_profile_sha256, field_name="semantic_profile_sha256")
        require_sorted_unique_strings(
            self.rejected_counterfeit_ids,
            field_name="rejected_counterfeit_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )
        if any(
            value
            for value in (
                self.path_selection_count,
                self.source_write_count,
                self.instrument_command_count,
                self.actuator_count,
                self.physical_evidence_count,
            )
        ):
            raise ValueError("synthetic source qualification crossed a physical boundary")
        expected = (
            self.positive_episode_count == 2
            and self.rejected_counterfeit_ids == PHYSICAL_SCALE_MORPHISM_SYNTHETIC_SOURCE_COUNTERFEIT_IDS
        )
        if self.draft_profile_qualified != expected:
            raise ValueError("draft-profile qualification is not fixture-derived")
        if self.exact_dossier_profile_qualified:
            raise ValueError("synthetic qualification cannot qualify an absent dossier")
        if self.scientific_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("synthetic source qualification must remain nonpromotable")


def _npz(*, hold: bool, nonmonotone_clock: bool = False) -> bytes:
    count = 11
    times = np.linspace(0, 1, count, dtype="<f8")
    if nonmonotone_clock:
        times[5] = times[4]
    action: np.ndarray = np.zeros((count, 2), dtype="<f8")
    if not hold:
        action[2:5, 0] = 0.25
    handle = BytesIO()
    np.savez(
        handle,
        times_seconds=times,
        node_voltages_volts=np.zeros((count, 16), dtype="<f8"),
        terminal_currents_amperes=np.zeros((count, 2), dtype="<f8"),
        environment_values=np.full((count, 1), 25.0, dtype="<f8"),
        observation_validity=np.ones((count, 19), dtype="u1"),
        requested_action_volts=action,
        accepted_action_volts=action,
        applied_action_volts=action,
        realized_action_volts=action,
        status_codes=np.zeros(count, dtype="<i2"),
    )
    return handle.getvalue()


def _ledger(episode_id: str, action_id: str) -> RcLadderResponseActionLedger:
    return RcLadderResponseActionLedger(
        ledger_id=f"ledger.{episode_id}",
        episode_id=episode_id,
        requested_action_id=action_id,
        accepted_action_id=action_id,
        applied_action_id=action_id,
        realized_action_id=action_id,
        delivery_status=RcLadderResponseActionDeliveryStatus.REALIZED,
        request_time_seconds=Decimal("0"),
        acceptance_time_seconds=Decimal("0.01"),
        application_time_seconds=Decimal("0.02"),
        realization_time_seconds=Decimal("0.03"),
        clipping_observed=False,
        interlock_observed=False,
    )


def synthetic_bundle(
    *, nonmonotone_clock: bool = False
) -> tuple[bytes, tuple[RcLadderResponsePreopenedMember, ...]]:
    board = RcLadderResponseBoardIdentity(
        board_id="board.synthetic-001",
        implementation_id="implementation.synthetic-lab",
        serial_id="serial.synthetic-001",
        scale_cells=16,
        batch_id="batch.synthetic-001",
        panel_id="panel.synthetic-001",
        physical_length_millimetres=Decimal("160"),
        resistance_component_ids=tuple(f"resistor-{index:03d}" for index in range(15)),
        capacitance_component_ids=tuple(f"capacitor-{index:03d}" for index in range(16)),
        source_resistance_component_id="resistor-source",
        termination_resistance_component_id="resistor-termination",
    )
    metrology = RcLadderResponseComponentMetrology(
        metrology_id="metrology.synthetic-001",
        board_id=board.board_id,
        resistance_component_ids=board.resistance_component_ids,
        capacitance_component_ids=board.capacitance_component_ids,
        source_resistance_component_id=board.source_resistance_component_id,
        termination_resistance_component_id=board.termination_resistance_component_id,
        resistance_ohms=tuple(Decimal("1000") for _ in range(15)),
        resistance_uncertainty_ohms=tuple(Decimal("1") for _ in range(15)),
        capacitance_farads=tuple(Decimal("0.000001") for _ in range(16)),
        capacitance_uncertainty_farads=tuple(Decimal("0.000000001") for _ in range(16)),
        source_resistance_ohms=Decimal("50"),
        source_resistance_uncertainty_ohms=Decimal("0.1"),
        termination_resistance_ohms=Decimal("1000"),
        termination_resistance_uncertainty_ohms=Decimal("1"),
        voltage_uncertainty_volts=Decimal("0.001"),
        current_uncertainty_amperes=Decimal("0.000001"),
        measured_before_response=True,
    )
    preparation = RcLadderResponsePreparationRecord(
        preparation_id="preparation.synthetic-zero",
        board_id=board.board_id,
        requested_equivalence_class_id="preparation-class.zero",
        requested_node_voltages_volts=tuple(Decimal("0") for _ in range(16)),
        realized_node_voltages_volts=tuple(Decimal("0") for _ in range(16)),
        voltage_uncertainty_volts=Decimal("0.001"),
        reset_evidence_ids=("evidence.synthetic-reset",),
        equilibration_evidence_ids=("evidence.synthetic-equilibration",),
        realization_complete_before_action_request=True,
    )
    clock_fit = RcLadderResponseClockFitRecord(
        clock_fit_id="clock-fit.synthetic-acquisition",
        native_clock_id="clock.synthetic-acquisition",
        reference_clock_id="clock.synthetic-reference",
        offset_seconds=Decimal("0"),
        drift_seconds_per_second=Decimal("0"),
        maximum_residual_seconds=Decimal("0.000001"),
        uncertainty_seconds=Decimal("0.000001"),
        calibration_evidence_ids=("evidence.synthetic-clock-calibration",),
    )
    payloads = {
        "episode.synthetic-action": _npz(hold=False, nonmonotone_clock=nonmonotone_clock),
        "episode.synthetic-hold": _npz(hold=True),
    }
    board_identity = ObjectIdentity.from_record(board.board_id, board)
    metrology_identity = ObjectIdentity.from_record(metrology.metrology_id, metrology)
    preparation_identity = ObjectIdentity.from_record(preparation.preparation_id, preparation)
    channels = tuple(
        sorted(
            (
                *(f"node-{index:03d}-voltage" for index in range(16)),
                "terminal-left-current",
                "terminal-right-current",
            )
        )
    )
    environment_channels = ("temperature-celsius",)
    calibrated_channels = tuple(sorted((*channels, *environment_channels)))
    channel_calibration = RcLadderResponseChannelCalibration(
        calibration_id="calibration.synthetic-channels",
        channel_ids=calibrated_channels,
        native_units=tuple(
            "degC"
            if value == "temperature-celsius"
            else "A"
            if value.startswith("terminal-")
            else "V"
            for value in calibrated_channels
        ),
        range_lower_native=tuple(
            Decimal("0") if value == "temperature-celsius" else Decimal("-10")
            for value in calibrated_channels
        ),
        range_upper_native=tuple(
            Decimal("50") if value == "temperature-celsius" else Decimal("10")
            for value in calibrated_channels
        ),
        resolution_native=tuple(Decimal("0.000001") for _ in calibrated_channels),
        uncertainty_native=tuple(Decimal("0.001") for _ in calibrated_channels),
        calibration_evidence_ids=("evidence.synthetic-channel-calibration",),
        measured_before_response=True,
    )
    clock_fit_identity = ObjectIdentity.from_record(clock_fit.clock_fit_id, clock_fit)
    calibration_identity = ObjectIdentity.from_record(
        channel_calibration.calibration_id, channel_calibration
    )
    episodes = []
    for episode_id, is_hold in (
        ("episode.synthetic-action", False),
        ("episode.synthetic-hold", True),
    ):
        payload = payloads[episode_id]
        episodes.append(
            RcLadderResponseEpisodeManifest(
                episode_id=episode_id,
                role=RcLadderResponseEpisodeRole.CANARY,
                board=board_identity,
                metrology=metrology_identity,
                preparation=preparation_identity,
                complete_context_id="context.synthetic-001",
                matched_hold_episode_id="episode.synthetic-hold",
                is_hold=is_hold,
                action_ledger=_ledger(episode_id, "hold" if is_hold else "pulse-positive"),
                native_clock_id="clock.synthetic-acquisition",
                clock_fit=clock_fit_identity,
                channel_calibration=calibration_identity,
                previous_episode_id=("episode.synthetic-action" if is_hold else None),
                causal_cutoff_seconds=Decimal("0.2"),
                receiver_window_seconds=(Decimal("0.2"), Decimal("1")),
                sample_count=11,
                channel_ids=channels,
                environment_channel_ids=environment_channels,
                voltage_unit="V",
                current_unit="A",
                clock_unit="s",
                gauge_id="node-to-ground",
                terminal_current_sign_id="positive-current-into-ladder",
                member_relative_path=f"episodes/{episode_id}.npz",
                member_sha256=sha256(payload).hexdigest(),
                member_size_bytes=len(payload),
            )
        )
    profile = draft_semantic_profile()
    manifest = RcLadderResponseBundleManifest(
        manifest_id="manifest.synthetic-001",
        dossier=ObjectIdentity(
            object_id="dossier.synthetic-draft-001",
            object_schema='empirical-lawhood/physical/rc-ladder-response/synthetic-draft-dossier',
            object_version="1.0.0",
            object_fingerprint="1" * 64,
        ),
        protocol=ObjectIdentity(
            object_id="protocol.synthetic-draft-001",
            object_schema='empirical-lawhood/physical/rc-ladder-response/synthetic-draft-protocol',
            object_version="1.0.0",
            object_fingerprint="2" * 64,
        ),
        implementation_id=board.implementation_id,
        board=board,
        metrology=metrology,
        clock_fits=(clock_fit,),
        channel_calibrations=(channel_calibration,),
        preparations=(preparation,),
        episodes=tuple(sorted(episodes, key=lambda value: value.episode_id)),
        semantic_profile=ObjectIdentity.from_record(profile.profile_id, profile),
        laboratory_authority=ObjectIdentity(
            object_id="authority.synthetic-no-physical-act",
            object_schema='empirical-lawhood/physical/rc-ladder-response/synthetic-nonauthority',
            object_version="1.0.0",
            object_fingerprint="3" * 64,
        ),
        custodian_id="custodian.synthetic-001",
        custody_seal=ObjectIdentity(
            object_id="seal.synthetic-001",
            object_schema='empirical-lawhood/physical/rc-ladder-response/synthetic-custody-seal',
            object_version="1.0.0",
            object_fingerprint="4" * 64,
        ),
        manifest_signature=ObjectIdentity(
            object_id="signature.synthetic-manifest-001",
            object_schema='empirical-lawhood/physical/rc-ladder-response/synthetic-manifest-signature',
            object_version="1.0.0",
            object_fingerprint="5" * 64,
        ),
        publication_receipt=ObjectIdentity(
            object_id="receipt.synthetic-publication-001",
            object_schema='empirical-lawhood/physical/rc-ladder-response/synthetic-publication-receipt',
            object_version="1.0.0",
            object_fingerprint="6" * 64,
        ),
        sealed=True,
    )
    members = tuple(
        RcLadderResponsePreopenedMember(relative_path=f"episodes/{episode_id}.npz", payload=payload)
        for episode_id, payload in sorted(payloads.items())
    )
    return manifest.canonical_bytes(), members


def _mutate_manifest(manifest_payload: bytes, path: tuple[str | int, ...], value: object) -> bytes:
    document = json.loads(manifest_payload)
    target: object = document
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    return (
        json.dumps(document, ensure_ascii=True, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("ascii")


def _rejected(
    *,
    manifest: bytes,
    members: tuple[RcLadderResponsePreopenedMember, ...],
    profile: RcLadderResponsePhysicalSemanticProfile,
) -> bool:
    try:
        decode_bundle(manifest_payload=manifest, members=members, semantic_profile=profile)
    except (RcLadderResponseBundleDecodeError, ValueError):
        return True
    return False


def qualify_synthetic_source_adapter() -> RcLadderResponseSyntheticSourceQualification:
    manifest, members = synthetic_bundle()
    profile = draft_semantic_profile()
    positive = decode_bundle(
        manifest_payload=manifest,
        members=members,
        semantic_profile=profile,
    )
    rejected = []
    corrupt = bytearray(members[0].payload)
    corrupt[-1] ^= 1
    try:
        decode_bundle(
            manifest_payload=manifest,
            members=(
                RcLadderResponsePreopenedMember(members[0].relative_path, bytes(corrupt)),
                members[1],
            ),
            semantic_profile=profile,
        )
    except RcLadderResponseBundleDecodeError:
        rejected.append("counterfeit.member-hash-drift")
    bad_manifest, bad_members = synthetic_bundle(nonmonotone_clock=True)
    try:
        decode_bundle(
            manifest_payload=bad_manifest,
            members=bad_members,
            semantic_profile=profile,
        )
    except RcLadderResponseBundleDecodeError:
        rejected.append("counterfeit.native-clock-nonmonotone")
    manifest_paths = {
        "counterfeit.action-delivery-stage-contradiction": (
            "value",
            "episodes",
            0,
            "value",
            "action_ledger",
            "value",
            "delivery_status",
        ),
        "counterfeit.channel-roster-drift": (
            "value",
            "episodes",
            0,
            "value",
            "channel_ids",
        ),
        "counterfeit.channel-calibration-roster-drift": (
            "value",
            "channel_calibrations",
            0,
            "value",
            "channel_ids",
        ),
        "counterfeit.channel-calibration-unit-drift": (
            "value",
            "channel_calibrations",
            0,
            "value",
            "native_units",
        ),
        "counterfeit.clock-fit-linkage-drift": (
            "value",
            "episodes",
            0,
            "value",
            "native_clock_id",
        ),
        "counterfeit.component-roster-drift": (
            "value",
            "metrology",
            "value",
            "capacitance_component_ids",
        ),
        "counterfeit.interlock-without-status-codebook": (
            "value",
            "episodes",
            0,
            "value",
            "action_ledger",
            "value",
            "interlock_observed",
        ),
        "counterfeit.metrology-uncertainty-roster-drift": (
            "value",
            "metrology",
            "value",
            "capacitance_uncertainty_farads",
        ),
        "counterfeit.preparation-dimension-drift": (
            "value",
            "preparations",
            0,
            "value",
            "realized_node_voltages_volts",
        ),
        "counterfeit.previous-episode-linkage-drift": (
            "value",
            "episodes",
            0,
            "value",
            "previous_episode_id",
        ),
        "counterfeit.terminal-current-sign-drift": (
            "value",
            "episodes",
            0,
            "value",
            "terminal_current_sign_id",
        ),
        "counterfeit.unknown-manifest-field": ("value", "unknown_field"),
        "counterfeit.unsafe-relative-path": (
            "value",
            "episodes",
            0,
            "value",
            "member_relative_path",
        ),
        "counterfeit.unit-contract-drift": (
            "value",
            "episodes",
            0,
            "value",
            "voltage_unit",
        ),
    }
    manifest_values: dict[str, object] = {
        "counterfeit.action-delivery-stage-contradiction": "REJECTED",
        "counterfeit.channel-roster-drift": [],
        "counterfeit.channel-calibration-roster-drift": [],
        "counterfeit.channel-calibration-unit-drift": ["1" for _ in range(19)],
        "counterfeit.clock-fit-linkage-drift": "clock.unbound",
        "counterfeit.component-roster-drift": [
            "capacitor-substituted",
            *(f"capacitor-{index:03d}" for index in range(1, 16)),
        ],
        "counterfeit.interlock-without-status-codebook": True,
        "counterfeit.metrology-uncertainty-roster-drift": [],
        "counterfeit.preparation-dimension-drift": [],
        "counterfeit.previous-episode-linkage-drift": "episode.absent",
        "counterfeit.terminal-current-sign-drift": "positive-current-out-of-ladder",
        "counterfeit.unknown-manifest-field": "forbidden",
        "counterfeit.unsafe-relative-path": "../escaped.npz",
        "counterfeit.unit-contract-drift": "mV",
    }
    for counterfeit_id, path in manifest_paths.items():
        if _rejected(
            manifest=_mutate_manifest(manifest, path, manifest_values[counterfeit_id]),
            members=members,
            profile=profile,
        ):
            rejected.append(counterfeit_id)
    if _rejected(manifest=manifest, members=members[:-1], profile=profile):
        rejected.append("counterfeit.member-roster-drift")
    rejected_ids = tuple(sorted(rejected))
    qualified = (
        len(positive.episodes) == 2 and rejected_ids == PHYSICAL_SCALE_MORPHISM_SYNTHETIC_SOURCE_COUNTERFEIT_IDS
    )
    return RcLadderResponseSyntheticSourceQualification(
        qualification_id="qualification.physical-scale-morphism-synthetic-source",
        semantic_profile_id="physical-scale-morphism-synthetic-draft-profile",
        semantic_profile_sha256=profile.fingerprint(),
        positive_episode_count=len(positive.episodes),
        rejected_counterfeit_ids=rejected_ids,
        path_selection_count=0,
        source_write_count=0,
        instrument_command_count=0,
        actuator_count=0,
        draft_profile_qualified=qualified,
        exact_dossier_profile_qualified=False,
        physical_evidence_count=0,
        reason_codes=(
            "DOSSIER_PROFILE_NOT_ATTEMPTED",
            "SYNTHETIC_DRAFT_PROFILE_QUALIFIED"
            if qualified
            else "SYNTHETIC_DRAFT_PROFILE_NOT_QUALIFIED",
        ),
        scientific_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


__all__ = [
    "PHYSICAL_SCALE_MORPHISM_SYNTHETIC_SOURCE_COUNTERFEIT_IDS",
    'RcLadderResponseSyntheticSourceQualification',
    "qualify_synthetic_source_adapter",
    "synthetic_bundle",
]
