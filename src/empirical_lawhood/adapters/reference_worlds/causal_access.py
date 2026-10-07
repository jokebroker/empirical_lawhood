"""Truth-known reference worlds for access-versus-scale conformance.

Each family contains one planted coordinate whose observation resolves an
otherwise shifted relation, plus a dimension-matched unit permutation.  Truth
metadata is returned separately from the public tournament observations so a
runner can withhold it until terminal evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.methods.causal_access_tournament import TournamentObservation


FloatArray = npt.NDArray[np.float64]


class AccessMechanism(StrEnum):
    HIDDEN_DELIVERY = "HIDDEN_DELIVERY"
    SHARED_DRIVER = "SHARED_DRIVER"
    HIDDEN_STORAGE = "HIDDEN_STORAGE"
    RECEIVER_PROJECTION = "RECEIVER_PROJECTION"
    SWITCHING_MODE = "SWITCHING_MODE"
    IRRELEVANT_ACCESS_NULL = "IRRELEVANT_ACCESS_NULL"


class FailureForecast(StrEnum):
    SURVIVES = "SURVIVES"
    RELOCATES = "RELOCATES"
    SPLITS = "SPLITS"
    REVERSES = "REVERSES"
    DISAPPEARS = "DISAPPEARS"
    REMAINS_AMBIGUOUS = "REMAINS_AMBIGUOUS"


@dataclass(frozen=True, slots=True)
class GeneratedAccessTruth:
    mechanism: AccessMechanism
    access_coordinate_ids: tuple[str, ...]
    planted_rival: str
    failure_forecast: FailureForecast
    rival_eliminated_by_exact_access: bool
    access_coordinate_causal: bool
    response_present: bool


@dataclass(frozen=True, slots=True)
class GeneratedAccessCohort:
    cohort_id: str
    split: str
    observations: tuple[TournamentObservation, ...]
    truth: GeneratedAccessTruth
    scalar_vector_mirror_max_abs_difference: float

    def __post_init__(self) -> None:
        if self.split not in {"DEVELOPMENT", "EVALUATION"}:
            raise ValueError("generated access cohort split is invalid")
        if not self.observations:
            raise ValueError("generated access cohort cannot be empty")
        if self.scalar_vector_mirror_max_abs_difference > 1e-12:
            raise ValueError("generated scalar/vector mirror implementations disagree")


def truth_for_mechanism(mechanism: AccessMechanism) -> GeneratedAccessTruth:
    values = {
        AccessMechanism.HIDDEN_DELIVERY: GeneratedAccessTruth(
            mechanism,
            ("delivered-action",),
            "REQUEST_EQUALS_DELIVERY",
            FailureForecast.RELOCATES,
            True,
            True,
            True,
        ),
        AccessMechanism.SHARED_DRIVER: GeneratedAccessTruth(
            mechanism,
            ("shared-driver",),
            "DIRECT_LOCAL_COUPLING",
            FailureForecast.DISAPPEARS,
            True,
            True,
            True,
        ),
        AccessMechanism.HIDDEN_STORAGE: GeneratedAccessTruth(
            mechanism,
            ("storage-state",),
            "FINITE_VISIBLE_HISTORY_IS_STATE_COMPLETE",
            FailureForecast.RELOCATES,
            True,
            True,
            True,
        ),
        AccessMechanism.RECEIVER_PROJECTION: GeneratedAccessTruth(
            mechanism,
            ("receiver-component-a", "receiver-component-b"),
            "POOLED_RECEIVER_IS_SINGLE_VALUED",
            FailureForecast.SPLITS,
            True,
            True,
            True,
        ),
        AccessMechanism.SWITCHING_MODE: GeneratedAccessTruth(
            mechanism,
            ("embedded-mode", "mode-action-interaction"),
            "ONE_ACTION_SLOPE_ACROSS_MODES",
            FailureForecast.REVERSES,
            True,
            True,
            True,
        ),
        AccessMechanism.IRRELEVANT_ACCESS_NULL: GeneratedAccessTruth(
            mechanism,
            ("irrelevant-coordinate",),
            "PROPOSED_ACCESS_IS_RELEVANT",
            FailureForecast.SURVIVES,
            False,
            False,
            True,
        ),
    }
    return values[mechanism]


def _target_vectorized(
    mechanism: AccessMechanism,
    *,
    requested: FloatArray,
    local: FloatArray,
    history_one: FloatArray,
    access: FloatArray,
) -> FloatArray:
    if mechanism is AccessMechanism.HIDDEN_DELIVERY:
        return 0.35 * local + 1.40 * access[:, 0]
    if mechanism is AccessMechanism.SHARED_DRIVER:
        return 0.25 * requested + 1.55 * access[:, 0]
    if mechanism is AccessMechanism.HIDDEN_STORAGE:
        return 0.20 * history_one + 1.60 * access[:, 0]
    if mechanism is AccessMechanism.RECEIVER_PROJECTION:
        return 0.20 * requested + 1.20 * access[:, 0] - 0.80 * access[:, 1]
    if mechanism is AccessMechanism.SWITCHING_MODE:
        return 0.30 * local + 1.45 * access[:, 1]
    return 0.55 * local + 0.85 * requested


def _target_scalar(
    mechanism: AccessMechanism,
    *,
    requested: float,
    local: float,
    history_one: float,
    access: tuple[float, ...],
) -> float:
    if mechanism is AccessMechanism.HIDDEN_DELIVERY:
        return 0.35 * local + 1.40 * access[0]
    if mechanism is AccessMechanism.SHARED_DRIVER:
        return 0.25 * requested + 1.55 * access[0]
    if mechanism is AccessMechanism.HIDDEN_STORAGE:
        return 0.20 * history_one + 1.60 * access[0]
    if mechanism is AccessMechanism.RECEIVER_PROJECTION:
        return 0.20 * requested + 1.20 * access[0] - 0.80 * access[1]
    if mechanism is AccessMechanism.SWITCHING_MODE:
        return 0.30 * local + 1.45 * access[1]
    return 0.55 * local + 0.85 * requested


def _latent_arrays(
    mechanism: AccessMechanism,
    *,
    unit_count: int,
    actions_per_unit: int,
    rng: np.random.Generator,
    shifted: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    rows = unit_count * actions_per_unit
    unit_latent = rng.normal(0.0, 1.0, size=unit_count)
    latent = np.repeat(unit_latent, actions_per_unit)
    action_chart = np.linspace(-1.2, 1.2, actions_per_unit, dtype=np.float64)
    requested = np.tile(action_chart, unit_count)
    nuisance = rng.normal(0.0, 1.0, size=rows)
    history_one = 0.7 * latent + 0.25 * requested + rng.normal(0.0, 0.15, size=rows)
    history_two = 0.55 * latent - 0.15 * requested + rng.normal(0.0, 0.18, size=rows)

    if mechanism is AccessMechanism.HIDDEN_DELIVERY:
        gain = np.repeat(rng.uniform(0.45, 1.45, size=unit_count), actions_per_unit)
        delivered = gain * requested + 0.20 * latent
        local = (0.9 if not shifted else -0.15) * delivered + 0.35 * latent + 0.15 * nuisance
        access = delivered[:, None]
    elif mechanism is AccessMechanism.SHARED_DRIVER:
        driver = latent + 0.35 * np.sin(requested * np.pi)
        local_loading = 0.9 if not shifted else -0.45
        local = local_loading * driver + 0.45 * nuisance
        access = driver[:, None]
    elif mechanism is AccessMechanism.HIDDEN_STORAGE:
        storage = 0.9 * latent + 0.55 * requested
        if shifted:
            history_one = -0.20 * storage + rng.normal(0.0, 0.55, size=rows)
            history_two = 0.10 * storage + rng.normal(0.0, 0.55, size=rows)
        local = 0.35 * requested + 0.25 * storage + 0.30 * nuisance
        access = storage[:, None]
    elif mechanism is AccessMechanism.RECEIVER_PROJECTION:
        component_a = latent + 0.60 * requested
        component_b = (
            0.75 * component_a + 0.20 * nuisance
            if not shifted
            else -0.30 * component_a + 0.85 * nuisance
        )
        local = component_a + component_b
        access = np.column_stack((component_a, component_b))
    elif mechanism is AccessMechanism.SWITCHING_MODE:
        if shifted:
            mode_unit = rng.choice(np.asarray([-1.0, 1.0]), size=unit_count, p=(0.65, 0.35))
        else:
            mode_unit = rng.choice(np.asarray([-1.0, 1.0]), size=unit_count, p=(0.15, 0.85))
        mode = np.repeat(mode_unit, actions_per_unit)
        local = 0.45 * latent + 0.20 * requested + 0.25 * nuisance
        access = np.column_stack((mode, mode * requested))
    else:
        local = 0.70 * latent + 0.35 * requested + 0.20 * nuisance
        access = rng.normal(0.0, 1.0, size=(rows, 1))
    return requested, local, history_one, history_two, access


def generate_access_cohort(
    mechanism: AccessMechanism,
    *,
    split: str,
    unit_count: int,
    actions_per_unit: int,
    seed: int,
    shifted: bool,
) -> GeneratedAccessCohort:
    """Generate disjoint public observations and separately typed truth."""

    if split not in {"DEVELOPMENT", "EVALUATION"}:
        raise ValueError("generated access split is invalid")
    if unit_count < 4 or actions_per_unit < 3:
        raise ValueError("generated access cohort is too small")
    rng = np.random.default_rng(seed)
    requested, local, history_one, history_two, access = _latent_arrays(
        mechanism,
        unit_count=unit_count,
        actions_per_unit=actions_per_unit,
        rng=rng,
        shifted=shifted,
    )
    target_noiseless = _target_vectorized(
        mechanism,
        requested=requested,
        local=local,
        history_one=history_one,
        access=access,
    )
    target = target_noiseless + rng.normal(0.0, 0.045, size=target_noiseless.shape)
    scalar = np.asarray(
        [
            _target_scalar(
                mechanism,
                requested=float(requested[index]),
                local=float(local[index]),
                history_one=float(history_one[index]),
                access=tuple(float(value) for value in access[index]),
            )
            for index in range(target_noiseless.size)
        ],
        dtype=np.float64,
    )
    mirror_defect = float(np.max(np.abs(target_noiseless - scalar)))

    access_by_unit = access.reshape(unit_count, actions_per_unit, access.shape[1])
    permutation = np.roll(np.arange(unit_count), 1)
    sham = access_by_unit[permutation].reshape(access.shape)
    rows: list[TournamentObservation] = []
    mechanism_token = mechanism.value.lower().replace("_", "-")
    split_token = split.lower()
    for unit_index in range(unit_count):
        unit_id = f"unit.generated.{mechanism_token}.{split_token}-{unit_index + 1:03d}"
        for action_index in range(actions_per_unit):
            index = unit_index * actions_per_unit + action_index
            rows.append(
                TournamentObservation(
                    observation_id=f"observation.{unit_id}.action-{action_index + 1:02d}",
                    unit_id=unit_id,
                    baseline_features=(
                        float(requested[index]),
                        float(local[index]),
                        float(history_one[index]),
                        float(history_two[index]),
                    ),
                    access_features=tuple(float(value) for value in access[index]),
                    sham_features=tuple(float(value) for value in sham[index]),
                    target=float(target[index]),
                )
            )
    return GeneratedAccessCohort(
        cohort_id=f"cohort.generated.{mechanism_token}.{split_token}.seed-{seed}",
        split=split,
        observations=tuple(rows),
        truth=truth_for_mechanism(mechanism),
        scalar_vector_mirror_max_abs_difference=mirror_defect,
    )


__all__ = [
    "AccessMechanism",
    "FailureForecast",
    "GeneratedAccessCohort",
    "GeneratedAccessTruth",
    "generate_access_cohort",
    "truth_for_mechanism",
]
