"""Exact supplied-noise source mechanics for the Six-matrix response reactive-nesting follow-up.

This module owns physics reconstruction only.  It does not assign geometric
labels, qualify a source, issue a campaign, or grant authority.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from .contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView
from .model import ComplexArray, SixMatrixState
from .shooting import SixMatrixResponseShootingCommittorCheckpoint, derive_branch_rng, state_from_shooting_checkpoint
from .scientific_inputs import SixMatrixResponseScientificSeedInput, require_six_matrix_scientific_seed_input
from .simulation import BAOABGradientCache, baoab_step_and_capture_hermitian_noise


@dataclass(frozen=True, slots=True)
class SixMatrixResponseRNHistoricalBranchPath:
    """One exact historical shooting branch reconstructed with its original RNG."""

    branch_index: int
    checkpoint_id: str
    checkpoint_sha256: str
    seed_document_sha256: str
    states: tuple[SixMatrixState, ...]
    noises: ComplexArray
    terminal_positions_sha256: str
    terminal_momenta_sha256: str
    terminal_rng_state_sha256: str

    def __post_init__(self) -> None:
        noises = np.asarray(self.noises)
        if (
            not 0 <= self.branch_index < 64
            or len(self.states) != 1025
            or tuple(value.step_index for value in self.states) != tuple(range(1025))
            or noises.shape != (1024, 2, 3, 4, 4)
            or noises.dtype != np.dtype("complex128")
            or not np.isfinite(noises).all()
        ):
            raise ValueError("Matrix reactive nesting historical branch geometry differs")
        for value in (
            self.checkpoint_sha256,
            self.seed_document_sha256,
            self.terminal_positions_sha256,
            self.terminal_momenta_sha256,
            self.terminal_rng_state_sha256,
        ):
            if len(value) != 64:
                raise ValueError("Matrix reactive nesting historical branch digest differs")
            int(value, 16)
        copied = np.ascontiguousarray(noises, dtype="<c16")
        copied.setflags(write=False)
        object.__setattr__(self, "noises", copied)


def replay_retained_branch_path(
    *,
    checkpoint: SixMatrixResponseShootingCommittorCheckpoint,
    branch_index: int,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    scientific_seed_input: SixMatrixResponseScientificSeedInput | None = None,
) -> SixMatrixResponseRNHistoricalBranchPath:
    """Reconstruct exactly one immutable step-816 shooting supplied-noise future."""

    require_six_matrix_scientific_seed_input(
        scientific_seed_input, scientific_role="shooting-branch",
        current_root_id=f"checkpoint-sha256.{checkpoint.combined_state_sha256}",
        current_context_sha256=checkpoint.combined_state_sha256, stream_index=branch_index,
    )
    gradient_cache_state = BAOABGradientCache()
    if (
        checkpoint.parent_step != 816
        or checkpoint.q != 2
        or member.member_id != "six-matrix-response.member.mass-0p5.cross-coupling-1"
        or numerical_view.view_id != "six-matrix-response.view.baoab-dt-0p001"
        or not 0 <= branch_index < 64
    ):
        raise ValueError("Matrix reactive nesting historical replay denominator differs")
    checkpoint_state, _ = state_from_shooting_checkpoint(checkpoint)
    state = SixMatrixState(
        q=checkpoint_state.q,
        positions=checkpoint_state.positions,
        momenta=checkpoint_state.momenta,
        step_index=0,
        alpha_tilde_x=checkpoint_state.alpha_tilde_x,
        alpha_tilde_y=checkpoint_state.alpha_tilde_y,
    )
    rng, seed_document_sha256 = derive_branch_rng(
        checkpoint_combined_sha256=checkpoint.combined_state_sha256,
        branch_index=branch_index,
        scientific_seed_input=scientific_seed_input,
    )
    states = [state]
    noises: list[ComplexArray] = []
    for _ in range(1024):
        state, noise = baoab_step_and_capture_hermitian_noise(
            state,
            member=member,
            numerical_view=numerical_view,
            next_alpha_tilde_x=checkpoint_state.alpha_tilde_x,
            next_alpha_tilde_y=checkpoint_state.alpha_tilde_y,
            rng=rng,
            gradient_cache=gradient_cache_state,
        )
        if not state.finite:
            raise FloatingPointError("Matrix reactive nesting historical branch became nonfinite")
        states.append(state)
        noises.append(noise)
    rng_state = json.dumps(rng.bit_generator.state, sort_keys=True, separators=(",", ":"))
    return SixMatrixResponseRNHistoricalBranchPath(
        branch_index=branch_index,
        checkpoint_id=checkpoint.checkpoint_id,
        checkpoint_sha256=checkpoint.combined_state_sha256,
        seed_document_sha256=seed_document_sha256,
        states=tuple(states),
        noises=np.ascontiguousarray(np.stack(noises), dtype="<c16"),
        terminal_positions_sha256=sha256(state.positions.tobytes(order="C")).hexdigest(),
        terminal_momenta_sha256=sha256(state.momenta.tobytes(order="C")).hexdigest(),
        terminal_rng_state_sha256=sha256(rng_state.encode("utf-8")).hexdigest(),
    )


__all__ = ['SixMatrixResponseRNHistoricalBranchPath', 'replay_retained_branch_path']
