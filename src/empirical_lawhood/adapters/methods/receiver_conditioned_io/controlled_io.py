"""Finite-horizon controlled input-to-receiver construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import (
    CanonicalMatrix,
    ControlledIOMember,
    ControlledIOProductDisposition,
    ControlledIOQualificationConfig,
    MarkovKernel,
    MarkovKernelFamily,
    identity_matrix,
)


@dataclass(frozen=True, slots=True)
class ControlledIOEvaluator:
    """Pure exact-coordinate evaluator for C_j Phi_A(j,l+1) B_l."""

    def evaluate(
        self,
        member: ControlledIOMember,
        config: ControlledIOQualificationConfig,
    ) -> MarkovKernelFamily:
        if config != member.qualification_config:
            raise ValueError("controlled-IO evaluator config differs from the member config")
        identity = ObjectIdentity.from_record(member.member_record_id, member)
        action_identities = tuple(
            sorted(
                (ObjectIdentity.from_record(value.word_id, value) for value in member.action_words),
                key=lambda value: value.object_id,
            )
        )
        if member.disposition is not ControlledIOProductDisposition.SUPPORTED:
            return MarkovKernelFamily(
                family_id=f"markov.{member.member_record_id}",
                controlled_io_member=identity,
                denominator_member_id=member.denominator_member_id,
                candidate_version_id=member.candidate_version_id,
                qualification_view_ids=member.qualification_view_ids,
                action_words=action_identities,
                support_cell_ids=member.support_cell_ids,
                retained_history_id=member.retained_history_id,
                horizon_id=member.horizon_id,
                state_clock_ids=tuple(value.state_clock_id for value in member.steps),
                input_clock_ids=tuple(value.input_clock_id for value in member.steps),
                receiver_clock_ids=tuple(value.receiver_clock_id for value in member.steps),
                kernels=(),
                finite_horizon_map=None,
                disposition=member.disposition,
                evidence_link_ids=(),
                reason_codes=member.reason_codes,
            )

        horizon = len(member.steps) - 1
        receiver_ids = member.receiver_basis.coordinate_ids
        input_ids = member.input_basis.coordinate_ids
        state_ids = member.state_basis.coordinate_ids
        kernels: list[MarkovKernel] = []
        row_blocks: list[np.ndarray] = []
        for receiver_index in range(1, horizon + 1):
            columns: list[np.ndarray] = []
            for input_index in range(horizon):
                if input_index >= receiver_index:
                    block = np.zeros((len(receiver_ids), len(input_ids)), dtype=np.float64)
                else:
                    transition = identity_matrix(state_ids)
                    # A_{j-1} ... A_{l+1}; the adjacent case is the identity.
                    for transition_index in range(input_index + 1, receiver_index):
                        transition = (
                            member.steps[transition_index].state_transition.as_array() @ transition
                        )
                    block = (
                        member.steps[receiver_index].receiver_map.as_array()
                        @ transition
                        @ member.steps[input_index].realized_input_map.as_array()
                    )
                    kernels.append(
                        MarkovKernel(
                            kernel_id=(
                                f"kernel.{member.member_record_id}.r{receiver_index}.u{input_index}"
                            ),
                            receiver_step_index=receiver_index,
                            input_step_index=input_index,
                            value=CanonicalMatrix.from_array(
                                matrix_id=(
                                    f"kernel-matrix.{member.member_record_id}."
                                    f"r{receiver_index}.u{input_index}"
                                ),
                                row_coordinate_ids=receiver_ids,
                                column_coordinate_ids=input_ids,
                                values=block,
                            ),
                        )
                    )
                columns.append(block)
            row_blocks.append(np.hstack(columns))
        stack = np.vstack(row_blocks)
        row_ids = tuple(
            f"receiver-step-{step:04d}.{coordinate}"
            for step in range(1, horizon + 1)
            for coordinate in receiver_ids
        )
        column_ids = tuple(
            f"input-step-{step:04d}.{coordinate}"
            for step in range(horizon)
            for coordinate in input_ids
        )
        return MarkovKernelFamily(
            family_id=f"markov.{member.member_record_id}",
            controlled_io_member=identity,
            denominator_member_id=member.denominator_member_id,
            candidate_version_id=member.candidate_version_id,
            qualification_view_ids=member.qualification_view_ids,
            action_words=action_identities,
            support_cell_ids=member.support_cell_ids,
            retained_history_id=member.retained_history_id,
            horizon_id=member.horizon_id,
            state_clock_ids=tuple(value.state_clock_id for value in member.steps),
            input_clock_ids=tuple(value.input_clock_id for value in member.steps),
            receiver_clock_ids=tuple(value.receiver_clock_id for value in member.steps),
            kernels=tuple(sorted(kernels, key=lambda value: value.kernel_id)),
            finite_horizon_map=CanonicalMatrix.from_array(
                matrix_id=f"finite-horizon-map.{member.member_record_id}",
                row_coordinate_ids=row_ids,
                column_coordinate_ids=column_ids,
                values=stack,
            ),
            disposition=ControlledIOProductDisposition.SUPPORTED,
            evidence_link_ids=member.diagnostics.evidence_link_ids,
            reason_codes=(),
        )
