"""Target-owned paired direct-TORAX development input."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from importlib.metadata import version
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .contracts import NativeToraxAction, NativeToraxEpisodeDisposition, NativeToraxPreparation, NativeToraxView
from .runtime import execute_native_torax


@dataclass(frozen=True, slots=True)
class NativeToraxQuickstart(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/torax-native/native-torax-quickstart'

    config_id: str
    preparation: NativeToraxPreparation
    view: NativeToraxView
    low_action: NativeToraxAction
    high_action: NativeToraxAction

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        self.view.operational_admission()
        if not self.config_id.startswith(
            "empirical-lawhood-"
        ) or not self.preparation.preparation_id.startswith(
            "preparation.empirical-lawhood-"
        ):
            raise ValueError("TORAX quick start needs new target-owned identities")
        if not self.view.view_id.startswith("view.empirical-lawhood-"):
            raise ValueError("TORAX quick start needs a target-owned view")
        if self.view.radial_cells > 24 or self.view.horizon_s > Decimal("0.04"):
            raise ValueError("TORAX quick-start resource bound differs")
        if (
            self.low_action.action_label != "down"
            or self.high_action.action_label != "up"
        ):
            raise ValueError("TORAX quick-start action labels differ")
        for action in (self.low_action, self.high_action):
            if not action.action_id.startswith("action.empirical-lawhood-"):
                raise ValueError("TORAX quick start needs target-owned action IDs")
            if action.duration_s != self.view.horizon_s or any(
                stage.coordinate_s != 0 for stage in action.stages
            ):
                raise ValueError("TORAX quick-start action clock differs from the view")
        if not (
            Decimal(0)
            < self.low_action.realized_power_w
            < self.high_action.realized_power_w
            <= Decimal(1100000)
        ):
            raise ValueError("TORAX quick-start power chart differs")


def run_native_development_check(config: NativeToraxQuickstart) -> dict[str, object]:
    """Execute two heat actions on one preparation and check native response."""

    config.view.operational_admission()
    if version("torax") != "1.4.2":
        raise ValueError("installed TORAX version differs from the source chart")
    outputs = tuple(
        execute_native_torax(config.preparation, action, config.view)
        for action in (config.low_action, config.high_action)
    )
    for trajectory, episode in outputs:
        if episode.disposition is not NativeToraxEpisodeDisposition.COMPLETE:
            raise ValueError(f"TORAX native episode stopped: {episode.reason_codes}")
        if trajectory.time_s[0] != 0 or trajectory.time_s[-1] != config.view.horizon_s:
            raise ValueError("TORAX native receiver clock differs")
    low, high = (episode for _trajectory, episode in outputs)
    if (
        not high.endpoint_delta_core_temperature_ev
        > low.endpoint_delta_core_temperature_ev
    ):
        raise ValueError("TORAX increased heat power did not increase core response")
    for action, episode in zip(
        (config.low_action, config.high_action), (low, high), strict=True
    ):
        if episode.effort_j != action.realized_power_w * config.view.horizon_s:
            raise ValueError(
                "TORAX native heat effort differs from power-time integral"
            )
    return {
        "config_id": config.config_id,
        "preparation_id": config.preparation.preparation_id,
        "independent_units": 1,
        "nested_action_views": 2,
        "native_torax_version": "1.4.2",
        "receiver_unit": "eV",
        "action_unit": "W",
        "effort_unit": "J",
        "horizon_s": str(config.view.horizon_s),
        "native_time_points": tuple(
            len(trajectory.time_s) for trajectory, _episode in outputs
        ),
        "low": {
            "power_w": str(config.low_action.realized_power_w),
            "stages": {
                stage.stage.value: {
                    "power_w": str(stage.power_w),
                    "coordinate_s": str(stage.coordinate_s),
                }
                for stage in config.low_action.stages
            },
            "effort_j": str(low.effort_j),
            "delta_core_temperature_ev": str(low.endpoint_delta_core_temperature_ev),
            "trajectory_sha256": low.trajectory_sha256,
        },
        "high": {
            "power_w": str(config.high_action.realized_power_w),
            "stages": {
                stage.stage.value: {
                    "power_w": str(stage.power_w),
                    "coordinate_s": str(stage.coordinate_s),
                }
                for stage in config.high_action.stages
            },
            "effort_j": str(high.effort_j),
            "delta_core_temperature_ev": str(high.endpoint_delta_core_temperature_ev),
            "trajectory_sha256": high.trajectory_sha256,
        },
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }


__all__ = ['NativeToraxQuickstart', "run_native_development_check"]
