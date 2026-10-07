"""Bounded target-owned Gym--TORAX development capture and causal check."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .action_word import GYM_TORAX_FUTURE_IP_ACTION_WORD_ID, GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID, build_gym_torax_action_word, build_gym_torax_native_schedule
from .diagnostic_contracts import GymToraxPreparation, gym_torax_numerical_members
from .field_metadata_contracts import GymToraxFieldMetadataEpisodeRequest, GymToraxRequestRole
from .extraction_manifest import build_gym_torax_extraction_manifest
from .field_metadata import build_gym_torax_field_metadata_manifest
from .runtime import acquire_gym_torax_episode


@dataclass(frozen=True, slots=True)
class GymToraxNativeQuickstart(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-quickstart'

    config_id: str
    preparation: GymToraxPreparation
    numerical_member_id: str
    maximum_output_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        if not self.config_id.startswith("empirical-lawhood-"):
            raise ValueError("Gym--TORAX quick start needs a new target-owned identity")
        if not self.preparation.preparation_id.startswith(
            "preparation.empirical-lawhood-"
        ) or not self.preparation.physical_independent_unit_id.startswith(
            "unit.empirical-lawhood-"
        ):
            raise ValueError(
                "Gym--TORAX development preparation needs new target identities"
            )
        if self.numerical_member_id != 'member.tokamak-control.primary':
            raise ValueError(
                "Gym--TORAX quick start selects only the checked native member"
            )
        if self.maximum_output_bytes != 64 * 1024 * 1024:
            raise ValueError("Gym--TORAX quick start output bound differs")


def run_native_development_check(
    config: GymToraxNativeQuickstart, *, repository_root: Path
) -> dict[str, object]:
    """Run paired native canaries and check a future action cannot alter their prefix."""

    if (
        not repository_root.is_absolute()
        or not (repository_root / "src/empirical_lawhood").is_dir()
    ):
        raise ValueError("Gym--TORAX target source checkout is absent or relative")
    manifest = build_gym_torax_extraction_manifest(repository_root)
    field_metadata = build_gym_torax_field_metadata_manifest()
    member = next(
        value
        for value in gym_torax_numerical_members()
        if value.member_id == config.numerical_member_id
    )

    def acquire(label: str, word_id: str):
        request = GymToraxFieldMetadataEpisodeRequest(
            request_id=f"request.empirical-lawhood-gym-native-{label}-{config.preparation.environment_seed}",
            request_role=GymToraxRequestRole.METADATA_CANARY,
            source_qualification=None,
            extraction_manifest=ObjectIdentity.from_record(
                manifest.manifest_id, manifest
            ),
            field_metadata_manifest=ObjectIdentity.from_record(
                field_metadata.manifest_id, field_metadata
            ),
            preparation=config.preparation,
            numerical_member=member,
            schedule=build_gym_torax_native_schedule(build_gym_torax_action_word(word_id)),
            maximum_output_bytes=config.maximum_output_bytes,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        )
        return acquire_gym_torax_episode(
            request,
            extraction_manifest=manifest,
            field_metadata_manifest=field_metadata,
            repository_root=repository_root,
        )

    hold = acquire("hold", GYM_TORAX_NATIVE_HOLD_ACTION_WORD_ID)
    future = acquire("future", GYM_TORAX_FUTURE_IP_ACTION_WORD_ID)
    expected_clocks = tuple(range(121))
    if hold.state_clocks != expected_clocks or future.state_clocks != expected_clocks:
        raise ValueError(
            "Gym--TORAX native episode did not complete its 121-state clock"
        )

    def scalar(episode, field_id: str):
        blocks = tuple(
            value
            for value in episode.blocks
            if value.category == "source-scalar" and value.native_field_id == field_id
        )
        if len(blocks) != 1 or blocks[0].clock_values != expected_clocks:
            raise ValueError(
                f"Gym--TORAX native {field_id} receiver is absent or unclocked"
            )
        return blocks[0]

    hold_q = scalar(hold, "Q_fusion")
    future_q = scalar(future, "Q_fusion")
    if (
        hold_q.native_unit != future_q.native_unit
        or hold_q.native_frame_id != future_q.native_frame_id
    ):
        raise ValueError("Gym--TORAX paired Q_fusion receiver units or frames differ")
    # The first altered request is clock 111. Its first possible receiver is
    # state 112, so states 0..111 must be unaffected by the future-only word.
    prefix_equal = np.array_equal(hold_q.array()[:112], future_q.array()[:112])
    if not prefix_equal:
        raise ValueError("Gym--TORAX future-only action altered a pre-action receiver")
    post_action_changed = not np.array_equal(
        hold_q.array()[112:], future_q.array()[112:]
    )
    if not post_action_changed:
        raise ValueError("Gym--TORAX native future action did not change Q_fusion")
    future_delivery = next(
        value for value in future.deliveries if value.request_clock == 111
    )
    return {
        "config_id": config.config_id,
        "preparation_id": config.preparation.preparation_id,
        "independent_unit_id": config.preparation.physical_independent_unit_id,
        "independent_units": 1,
        "nested_action_episodes": 2,
        "native_state_clocks": len(expected_clocks),
        "first_future_action_request_clock": 111,
        "pre_action_receiver_clocks_equal": prefix_equal,
        "post_action_q_fusion_changed": post_action_changed,
        "q_fusion_native_unit": hold_q.native_unit,
        "q_fusion_native_frame": hold_q.native_frame_id,
        "future_action_111": {
            "requested_ip_a": str(future_delivery.requested.ip_a),
            "accepted_ip_a": str(future_delivery.accepted.ip_a)
            if future_delivery.accepted is not None
            else None,
            "applied_ip_a": str(future_delivery.applied.ip_a)
            if future_delivery.applied is not None
            else None,
            "realized_ip_a": str(future_delivery.realized.ip_a)
            if future_delivery.realized is not None
            else None,
            "delivery_disposition": future_delivery.disposition.value,
        },
        "hold_observation_disposition": hold.observation_disposition.value,
        "future_observation_disposition": future.observation_disposition.value,
        "hold_reason_codes": hold.reason_codes,
        "future_reason_codes": future.reason_codes,
        "controlled_io_operator_available": not any(
            "OPERATOR_API_UNAVAILABLE" in episode.reason_codes
            for episode in (hold, future)
        ),
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }


__all__ = ['GymToraxNativeQuickstart', "run_native_development_check"]
