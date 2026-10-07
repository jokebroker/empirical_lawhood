"""Development-only analysis for the FSM empirical slice."""

from __future__ import annotations

from pathlib import Path

from .contracts import FineSteeringMirrorBlockResponse, FineSteeringMirrorDevelopmentModel, FineSteeringMirrorProtocolSpec
from .observation import fit_development_model, load_safe_array, reduce_block, sha256_path


class FineSteeringMirrorDevelopmentError(ValueError):
    """Development inputs violate their frozen identities or scientific gates."""


def analyze_development(
    *,
    action_paths: tuple[Path, Path, Path],
    receiver_paths: tuple[Path, Path, Path],
    protocol: FineSteeringMirrorProtocolSpec,
    member_sha256: dict[str, str],
) -> tuple[tuple[FineSteeringMirrorBlockResponse, ...], FineSteeringMirrorDevelopmentModel]:
    """Reduce six development blocks and fit the predeclared scalar action map."""

    observed_digests = tuple(sorted(sha256_path(path) for path in (*action_paths, *receiver_paths)))
    if observed_digests != protocol.expected_development_sha256:
        raise FineSteeringMirrorDevelopmentError("development source bytes differ from the frozen protocol")
    for path in (*action_paths, *receiver_paths):
        expected = member_sha256.get(path.name)
        if expected is None or sha256_path(path) != expected:
            raise FineSteeringMirrorDevelopmentError(f"development member identity differs: {path.name}")
    blocks: list[FineSteeringMirrorBlockResponse] = []
    for action_index, level in enumerate(protocol.action_levels_volts):
        actions = load_safe_array(
            action_paths[action_index],
            expected_realizations=6,
            operator=protocol.observation_operator,
        )
        receivers = load_safe_array(
            receiver_paths[action_index],
            expected_realizations=6,
            operator=protocol.observation_operator,
        )
        for block_index, start in enumerate((0, 3), start=1):
            blocks.append(
                reduce_block(
                    block_id=(f"fsm-development-{int(level * 1000)}mv-block-{block_index}"),
                    action_level_volts=level,
                    actions=actions,
                    receivers=receivers,
                    realization_start=start,
                    operator=protocol.observation_operator,
                )
            )
    ordered = tuple(sorted(blocks, key=lambda block: block.block_id))
    model = fit_development_model(ordered, protocol.action_levels_volts)
    if any(
        block.maximum_input_matrix_condition > protocol.maximum_input_matrix_condition
        for block in ordered
    ):
        raise FineSteeringMirrorDevelopmentError("development input matrix condition gate failed")
    if any(
        block.period_relative_difference > protocol.maximum_period_relative_difference
        for block in ordered
    ):
        raise FineSteeringMirrorDevelopmentError("development period recurrence gate failed")
    if not all(value < 0 for value in model.slope_hz_per_v):
        raise FineSteeringMirrorDevelopmentError("development slopes do not all have the registered sign")
    if (
        model.slope_hz_per_v[protocol.primary_receiver_index]
        > protocol.maximum_primary_slope_hz_per_v
    ):
        raise FineSteeringMirrorDevelopmentError("primary development slope does not clear the frozen gate")
    return ordered, model
