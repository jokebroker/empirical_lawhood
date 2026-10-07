"""Source-compatibility qualification for Six-matrix response reactive-entrance nesting.

The historical 64-branch replay closes a missing passive-probe operand.  It is
strictly a pre-issue compatibility check and contributes zero units to reactive entrance.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.composition.matrix_response_study.causal_intersection_residence_design import MatrixResponseCausalIntersectionResidenceStudyConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseModelFamilyMember, SixMatrixResponseNumericalView
from empirical_lawhood.adapters.simulators.six_matrix_response.controlled_branch import SixMatrixResponseTransientControlledInvarianceActionLedger, SixMatrixResponseTransientControlledInvarianceBranchTrace, build_action_schedule
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import SixMatrixResponseTransientControlledInvarianceProbeRoster
from empirical_lawhood.adapters.simulators.six_matrix_response.reactive_nesting import SixMatrixResponseRNHistoricalBranchPath
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .full_intersection_causal_authority import MatrixResponseCausalIntersectionResidenceResponseFactorSample, evaluate_full_intersection_branch
from .shooting_committor import MatrixResponseShootingCommittorBranchResult, MatrixResponseShootingCommittorBranchTerminal


ENTRY_STEP = 256
LAST_COMPATIBILITY_STEP = 384
COMPATIBILITY_STEPS = tuple(range(ENTRY_STEP, LAST_COMPATIBILITY_STEP + 1, 16))
EXPECTED_HISTORICAL_G_FIRST_INDICES = (6, 21, 27, 31, 43, 44, 62)


class MatrixResponseRNSourceCompatibilityTerminal(StrEnum):
    COMPATIBLE = "SOURCE_COMPATIBILITY_QUALIFIED"
    NONATTEMPT = "REACTIVE_ENTRANCE_PREREQUISITE_NONATTEMPT"


@dataclass(frozen=True, slots=True)
class MatrixResponseRNCompatibilityBranch(CanonicalRecord):
    """One all-intent historical replay and stricter entry-factor ledger."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-rn-compatibility-branch'

    compatibility_id: str
    branch_id: str
    branch_index: int
    seed_document_sha256: str
    retained_branch_sha256: str
    reproduced_branch_sha256: str
    old_branch_exact: bool
    seed_exact: bool
    terminal_positions_exact: bool
    terminal_momenta_exact: bool
    terminal_rng_state_exact: bool
    old_terminal: MatrixResponseShootingCommittorBranchTerminal
    old_tau_g: Decimal | None
    old_tau_00: Decimal | None
    factor_samples: tuple[MatrixResponseCausalIntersectionResidenceResponseFactorSample, ...]
    first_full_entry_step: int | None
    causal_full_entry: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.compatibility_id, field_name="compatibility_id")
        validate_stable_id(self.branch_id, field_name="branch_id")
        if not 0 <= self.branch_index < 64:
            raise ValueError("Matrix reactive nesting compatibility branch index differs")
        for name in (
            "seed_document_sha256",
            "retained_branch_sha256",
            "reproduced_branch_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if tuple(value.parent_step for value in self.factor_samples) != COMPATIBILITY_STEPS:
            raise ValueError("Matrix reactive nesting compatibility factor ledger differs")
        expected_first = next(
            (value.parent_step for value in self.factor_samples if value.full_intersection_pass),
            None,
        )
        if self.first_full_entry_step != expected_first:
            raise ValueError("Matrix reactive nesting first full entry differs")
        expected_entry = bool(
            expected_first is not None
            and (
                self.old_tau_00 is None
                or Decimal(expected_first) / Decimal(1000) < self.old_tau_00
            )
        )
        if self.causal_full_entry != expected_entry:
            raise ValueError("Matrix reactive nesting causal-entry truth differs")
        exact = bool(
            self.old_branch_exact
            and self.seed_exact
            and self.terminal_positions_exact
            and self.terminal_momenta_exact
            and self.terminal_rng_state_exact
        )
        if exact != (not self.reason_codes):
            raise ValueError("Matrix reactive nesting branch compatibility reasons differ")


@dataclass(frozen=True, slots=True)
class MatrixResponseRNSourceCompatibilityReceipt(CanonicalRecord):
    """Closed 64-branch source receipt; never an reactive entrance incidence estimate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-rn-source-compatibility-receipt'

    receipt_id: str
    checkpoint_id: str
    checkpoint_sha256: str
    shooting_config_fingerprint: str
    causal_intersection_factor_config_fingerprint: str
    branches: tuple[MatrixResponseRNCompatibilityBranch, ...]
    historical_g_first_indices: tuple[int, ...]
    response_geometric_entry_indices: tuple[int, ...]
    source_instance_count: int
    historical_replay_count: int
    rn0_denominator_contribution: int
    terminal: MatrixResponseRNSourceCompatibilityTerminal
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.checkpoint_id, field_name="checkpoint_id")
        for name in (
            "checkpoint_sha256",
            "shooting_config_fingerprint",
            "causal_intersection_factor_config_fingerprint",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        require_sorted_unique_ids(self.branches, attribute="compatibility_id", field_name="branches")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            tuple(value.branch_index for value in self.branches) != tuple(range(64))
            or self.response_geometric_entry_indices
            != tuple(value.branch_index for value in self.branches if value.causal_full_entry)
            or (self.source_instance_count, self.historical_replay_count, self.rn0_denominator_contribution)
            != (1, 64, 0)
        ):
            raise ValueError("Matrix reactive nesting source-compatibility roster differs")
        compatible = bool(
            self.historical_g_first_indices == EXPECTED_HISTORICAL_G_FIRST_INDICES
            and all(not value.reason_codes for value in self.branches)
        )
        if (self.terminal is MatrixResponseRNSourceCompatibilityTerminal.COMPATIBLE) != compatible:
            raise ValueError("Matrix reactive nesting source-compatibility terminal differs")
        if compatible != (not self.reason_codes):
            raise ValueError("Matrix reactive nesting source-compatibility receipt reasons differ")


def _compatibility_trace(
    *, path: SixMatrixResponseRNHistoricalBranchPath, numerical_view: SixMatrixResponseNumericalView
) -> SixMatrixResponseTransientControlledInvarianceBranchTrace:
    schedule = build_action_schedule(
        action_word="hold",
        branch_start_step=0,
        trigger_parent_step=None,
        total_primary_steps=LAST_COMPATIBILITY_STEP,
        baseline_x=path.states[0].alpha_tilde_x,
        baseline_y=path.states[0].alpha_tilde_y,
        timestep=float(numerical_view.timestep),
    )
    ledger = SixMatrixResponseTransientControlledInvarianceActionLedger(
        ledger_id=f"matrix-reactive-nesting.compatibility-ledger.b{path.branch_index:02d}",
        action_word="hold",
        trigger_parent_step=None,
        requested_sha256=schedule.requested_sha256,
        accepted_sha256=schedule.requested_sha256,
        applied_sha256=schedule.requested_sha256,
        realized_sha256=schedule.requested_sha256,
        maximum_excursion=Decimal(0),
        maximum_increment=Decimal(0),
        total_variation=Decimal(0),
        squared_action_energy=Decimal(0),
        generalized_absolute_work=Decimal(0),
        pulse_count=0,
        exact_baseline_return=True,
        clipped=False,
        valid=True,
        reason_codes=(),
    )
    y_path = np.ascontiguousarray(
        np.stack(
            tuple(value.positions[1] for value in path.states[: LAST_COMPATIBILITY_STEP + 1])
        ),
        dtype="<c16",
    )
    noises = np.ascontiguousarray(path.noises[:LAST_COMPATIBILITY_STEP], dtype="<c16")
    return SixMatrixResponseTransientControlledInvarianceBranchTrace(
        branch_id=f"matrix-reactive-nesting.compatibility.b{path.branch_index:02d}",
        block_index=path.branch_index,
        numerical_view_id=numerical_view.view_id,
        parent_step_multiplier=1,
        start_state=path.states[0],
        final_state=path.states[LAST_COMPATIBILITY_STEP],
        receiver_states=tuple(
            path.states[step] for step in range(16, LAST_COMPATIBILITY_STEP + 1, 16)
        ),
        y_path=y_path,
        noise_seed_sha256=path.seed_document_sha256,
        noise_block_sha256=sha256(noises.tobytes()).hexdigest(),
        action_ledger=ledger,
        valid=True,
        reason_codes=(),
    )


def evaluate_retained_branch_compatibility(
    *,
    path: SixMatrixResponseRNHistoricalBranchPath,
    retained: MatrixResponseShootingCommittorBranchResult,
    reproduced: MatrixResponseShootingCommittorBranchResult,
    member: SixMatrixResponseModelFamilyMember,
    numerical_view: SixMatrixResponseNumericalView,
    roster: SixMatrixResponseTransientControlledInvarianceProbeRoster,
    config: MatrixResponseCausalIntersectionResidenceStudyConfig,
) -> MatrixResponseRNCompatibilityBranch:
    """Compare exact historical bytes and materialize the missing response factor."""

    if (
        retained.branch_index != path.branch_index
        or reproduced.branch_index != path.branch_index
        or retained.branch_id != reproduced.branch_id
    ):
        raise ValueError("Matrix reactive nesting compatibility branch join differs")
    evaluation = evaluate_full_intersection_branch(
        trace=_compatibility_trace(path=path, numerical_view=numerical_view),
        intent_word="hold",
        member=member,
        roster=roster,
        config=config,
        assessment_start_step=ENTRY_STEP,
        assessment_end_step=LAST_COMPATIBILITY_STEP,
    )
    samples = evaluation.samples
    first_full_entry_step = next(
        (value.parent_step for value in samples if value.full_intersection_pass),
        None,
    )
    reasons: set[str] = set()
    retained_sha = retained.fingerprint()
    reproduced_sha = reproduced.fingerprint()
    if retained_sha != reproduced_sha:
        reasons.add("old-branch-reducer-mismatch")
    if path.seed_document_sha256 != retained.seed_document_sha256:
        reasons.add("branch-seed-mismatch")
    if path.terminal_positions_sha256 != retained.terminal_positions_sha256:
        reasons.add("terminal-positions-mismatch")
    if path.terminal_momenta_sha256 != retained.terminal_momenta_sha256:
        reasons.add("terminal-momenta-mismatch")
    if path.terminal_rng_state_sha256 != retained.terminal_rng_state_sha256:
        reasons.add("terminal-rng-state-mismatch")
    return MatrixResponseRNCompatibilityBranch(
        compatibility_id=f"matrix-reactive-nesting.source-compatibility.b{path.branch_index:02d}",
        branch_id=retained.branch_id,
        branch_index=path.branch_index,
        seed_document_sha256=path.seed_document_sha256,
        retained_branch_sha256=retained_sha,
        reproduced_branch_sha256=reproduced_sha,
        old_branch_exact=retained_sha == reproduced_sha,
        seed_exact=path.seed_document_sha256 == retained.seed_document_sha256,
        terminal_positions_exact=(
            path.terminal_positions_sha256 == retained.terminal_positions_sha256
        ),
        terminal_momenta_exact=(
            path.terminal_momenta_sha256 == retained.terminal_momenta_sha256
        ),
        terminal_rng_state_exact=(
            path.terminal_rng_state_sha256 == retained.terminal_rng_state_sha256
        ),
        old_terminal=retained.terminal,
        old_tau_g=retained.tau_g,
        old_tau_00=retained.tau_00,
        factor_samples=samples,
        first_full_entry_step=first_full_entry_step,
        causal_full_entry=bool(
            first_full_entry_step is not None
            and (
                retained.tau_00 is None
                or Decimal(first_full_entry_step) / Decimal(1000) < retained.tau_00
            )
        ),
        reason_codes=tuple(sorted(reasons)),
    )


def reduce_source_compatibility(
    *,
    checkpoint_id: str,
    checkpoint_sha256: str,
    shooting_config_fingerprint: str,
    causal_intersection_factor_config_fingerprint: str,
    branches: tuple[MatrixResponseRNCompatibilityBranch, ...],
) -> MatrixResponseRNSourceCompatibilityReceipt:
    """Close the exact all-64 compatibility roster without scientific promotion."""

    ordered = tuple(sorted(branches, key=lambda value: value.branch_index))
    reasons = tuple(
        sorted({reason for value in ordered for reason in value.reason_codes})
    )
    old_g = tuple(
        value.branch_index
        for value in ordered
        if value.old_terminal is MatrixResponseShootingCommittorBranchTerminal.G_FIRST_WITHIN_HORIZON
    )
    if old_g != EXPECTED_HISTORICAL_G_FIRST_INDICES:
        reasons = tuple(sorted(set(reasons) | {"historical-g-first-roster-mismatch"}))
    return MatrixResponseRNSourceCompatibilityReceipt(
        receipt_id="matrix-reactive-nesting.source-compatibility.step-0816",
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        shooting_config_fingerprint=shooting_config_fingerprint,
        causal_intersection_factor_config_fingerprint=causal_intersection_factor_config_fingerprint,
        branches=ordered,
        historical_g_first_indices=old_g,
        response_geometric_entry_indices=tuple(
            value.branch_index for value in ordered if value.causal_full_entry
        ),
        source_instance_count=1,
        historical_replay_count=64,
        rn0_denominator_contribution=0,
        terminal=(
            MatrixResponseRNSourceCompatibilityTerminal.COMPATIBLE
            if not reasons
            else MatrixResponseRNSourceCompatibilityTerminal.NONATTEMPT
        ),
        reason_codes=reasons,
    )


__all__ = [
    'MatrixResponseRNCompatibilityBranch',
    'MatrixResponseRNSourceCompatibilityReceipt',
    'MatrixResponseRNSourceCompatibilityTerminal',
    "ENTRY_STEP",
    "COMPATIBILITY_STEPS",
    "EXPECTED_HISTORICAL_G_FIRST_INDICES",
    'evaluate_retained_branch_compatibility',
    'reduce_source_compatibility',
]
