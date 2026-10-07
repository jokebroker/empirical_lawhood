"""Deterministic finite block-Lanczos receiver-visible Jacobi witnesses."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import (
    CanonicalMatrix,
    ControlledIOMember,
    ControlledIOProductDisposition,
    ControlledIOStep,
    CoordinateBasis,
    decimal_from_float,
)
from .receiver_metric import MetricDisposition, ReceiverRieszFamily, StateMetric


class JacobiDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    METRIC_REFUSED = "METRIC_REFUSED"
    RANK_DEFICIENT = "RANK_DEFICIENT"
    MOMENT_MISMATCH = "MOMENT_MISMATCH"
    MEMBER_UNSTABLE = "MEMBER_UNSTABLE"
    UNEVALUABLE = "UNEVALUABLE"


class JacobiGaugeConvention(StrEnum):
    LARGEST_MAGNITUDE_ENTRY_POSITIVE = "largest-magnitude-entry-positive"


class JacobiDeflationRule(StrEnum):
    SVD_RANK_BELOW_FROZEN_TOLERANCE = "svd-rank-below-frozen-tolerance"


class JacobiMomentNorm(StrEnum):
    FROBENIUS = "frobenius"


@dataclass(frozen=True, slots=True)
class JacobiConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/jacobi-config'

    config_id: str
    state_basis: CoordinateBasis
    input_basis: CoordinateBasis
    receiver_basis: CoordinateBasis
    maximum_order: int
    rank_tolerance: Decimal
    recurrence_tolerance: Decimal
    moment_tolerance: Decimal
    symmetry_tolerance: Decimal
    gauge_convention: JacobiGaugeConvention
    deflation_rule: JacobiDeflationRule
    moment_norm: JacobiMomentNorm
    local_operator_step_id: str
    local_operator_clock_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.maximum_order < 0:
            raise ValueError("Jacobi maximum order must be nonnegative")
        for name, value in (
            ("rank_tolerance", self.rank_tolerance),
            ("recurrence_tolerance", self.recurrence_tolerance),
            ("moment_tolerance", self.moment_tolerance),
            ("symmetry_tolerance", self.symmetry_tolerance),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        validate_stable_id(
            self.local_operator_step_id,
            field_name="local_operator_step_id",
        )
        validate_stable_id(
            self.local_operator_clock_id,
            field_name="local_operator_clock_id",
        )


@dataclass(frozen=True, slots=True)
class JacobiBlock(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/jacobi-block'

    block_id: str
    order_index: int
    diagonal: CanonicalMatrix
    coupling_to_next: CanonicalMatrix | None

    def __post_init__(self) -> None:
        validate_stable_id(self.block_id, field_name="block_id")
        if self.order_index < 0:
            raise ValueError("Jacobi block order must be nonnegative")
        if self.diagonal.shape[0] != self.diagonal.shape[1]:
            raise ValueError("Jacobi diagonal block must be square")
        if self.coupling_to_next is not None and (
            self.coupling_to_next.shape[1] != self.diagonal.shape[0]
        ):
            raise ValueError("Jacobi coupling columns must match the current block")


@dataclass(frozen=True, slots=True)
class JacobiMoment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/jacobi-moment'

    moment_id: str
    order: int
    direct: CanonicalMatrix
    reconstructed: CanonicalMatrix
    reconstruction_error: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.moment_id, field_name="moment_id")
        if self.order < 0:
            raise ValueError("moment order must be nonnegative")
        if self.direct.shape != self.reconstructed.shape:
            raise ValueError("direct and reconstructed moments require the same shape")


@dataclass(frozen=True, slots=True)
class JacobiWitness(CanonicalRecord):
    "Finite receiver-visible witness; explicitly not B/reachability/admission truth."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/jacobi-witness'

    witness_id: str
    controlled_io_member: ObjectIdentity
    local_operator_step: ObjectIdentity | None
    local_operator: ObjectIdentity | None
    metric: ObjectIdentity
    riesz_family: ObjectIdentity
    receiver_coordinate_ids: tuple[str, ...]
    local_operator_clock_id: str
    gauge_convention: JacobiGaugeConvention
    deflation_rule: JacobiDeflationRule
    moment_norm: JacobiMomentNorm
    blocks: tuple[JacobiBlock, ...]
    moments: tuple[JacobiMoment, ...]
    block_ranks: tuple[int, ...]
    deflated_order_indices: tuple[int, ...]
    maximum_reconstruction_error: NamedDecimal
    maximum_recurrence_error: NamedDecimal
    disposition: JacobiDisposition
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.witness_id, field_name="witness_id")
        if self.controlled_io_member.object_schema != ControlledIOMember.SCHEMA:
            raise ValueError("Jacobi witness requires one controlled-IO parent member")
        if self.local_operator_step is not None and (
            self.local_operator_step.object_schema != ControlledIOStep.SCHEMA
        ):
            raise ValueError("Jacobi witness local step must be a ControlledIOStep")
        if self.local_operator is not None and (
            self.local_operator.object_schema != CanonicalMatrix.SCHEMA
        ):
            raise ValueError("Jacobi witness local operator must be a CanonicalMatrix")
        require_sorted_unique_strings(
            self.receiver_coordinate_ids,
            field_name="receiver_coordinate_ids",
            allow_empty=False,
        )
        validate_stable_id(
            self.local_operator_clock_id,
            field_name="local_operator_clock_id",
        )
        require_sorted_unique_ids(self.blocks, attribute="block_id", field_name="blocks")
        require_sorted_unique_ids(self.moments, attribute="moment_id", field_name="moments")
        if any(value < 0 for value in self.block_ranks):
            raise ValueError("Jacobi block ranks must be nonnegative")
        if tuple(sorted(set(self.deflated_order_indices))) != self.deflated_order_indices:
            raise ValueError("Jacobi deflation indices must be sorted and unique")
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is JacobiDisposition.SUPPORTED:
            if self.local_operator_step is None or self.local_operator is None:
                raise ValueError("supported Jacobi witness requires an exact local step/operator")
            if not self.blocks or not self.moments or not self.evidence_link_ids:
                raise ValueError("supported Jacobi witness requires blocks/moments/evidence")
            if self.reason_codes:
                raise ValueError("supported Jacobi witness cannot carry reasons")
        elif self.blocks or self.moments or not self.reason_codes:
            raise ValueError("refused Jacobi witness cannot carry a favorable construction")


def _canonical_qr(values: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    u, singular, _ = np.linalg.svd(values, full_matrices=False)
    rank = int(np.sum(singular > tolerance))
    if rank == 0:
        return np.empty((values.shape[0], 0), dtype=np.float64), singular
    q = u[:, :rank]
    for column in range(q.shape[1]):
        pivot = int(np.argmax(np.abs(q[:, column])))
        if q[pivot, column] < 0:
            q[:, column] *= -1.0
    return q, singular


def _refused(
    *,
    witness_id: str,
    member: ControlledIOMember,
    local_step: ControlledIOStep | None,
    metric: StateMetric,
    riesz: ReceiverRieszFamily,
    config: JacobiConfig,
    disposition: JacobiDisposition,
    reason: str,
) -> JacobiWitness:
    return JacobiWitness(
        witness_id=witness_id,
        controlled_io_member=ObjectIdentity.from_record(member.member_record_id, member),
        local_operator_step=(
            ObjectIdentity.from_record(local_step.step_id, local_step)
            if local_step is not None
            else None
        ),
        local_operator=(
            ObjectIdentity.from_record(
                local_step.state_transition.matrix_id,
                local_step.state_transition,
            )
            if local_step is not None
            else None
        ),
        metric=ObjectIdentity.from_record(metric.metric_id, metric),
        riesz_family=ObjectIdentity.from_record(riesz.family_id, riesz),
        receiver_coordinate_ids=riesz.receiver_coordinate_ids,
        local_operator_clock_id=config.local_operator_clock_id,
        gauge_convention=config.gauge_convention,
        deflation_rule=config.deflation_rule,
        moment_norm=config.moment_norm,
        blocks=(),
        moments=(),
        block_ranks=(),
        deflated_order_indices=(),
        maximum_reconstruction_error=NamedDecimal(
            value_id=f"jacobi-reconstruction-error.{witness_id}",
            value=Decimal(0),
            unit="1",
        ),
        maximum_recurrence_error=NamedDecimal(
            value_id=f"jacobi-recurrence-error.{witness_id}",
            value=Decimal(0),
            unit="1",
        ),
        disposition=disposition,
        evidence_link_ids=(),
        reason_codes=(reason,),
    )


@dataclass(frozen=True, slots=True)
class JacobiWitnessService:
    def construct(
        self,
        *,
        witness_id: str,
        member: ControlledIOMember,
        metric: StateMetric,
        riesz: ReceiverRieszFamily,
        config: JacobiConfig,
    ) -> JacobiWitness:
        local_step = next(
            (value for value in member.steps if value.step_id == config.local_operator_step_id),
            None,
        )
        if local_step is None or local_step.state_clock_id != config.local_operator_clock_id:
            return _refused(
                witness_id=witness_id,
                member=member,
                local_step=local_step,
                metric=metric,
                riesz=riesz,
                config=config,
                disposition=JacobiDisposition.UNEVALUABLE,
                reason="local-operator-step-or-clock-mismatch",
            )
        local_operator = local_step.state_transition
        if (
            config.state_basis != member.state_basis
            or config.input_basis != member.input_basis
            or config.receiver_basis != member.receiver_basis
        ):
            return _refused(
                witness_id=witness_id,
                member=member,
                local_step=local_step,
                metric=metric,
                riesz=riesz,
                config=config,
                disposition=JacobiDisposition.UNEVALUABLE,
                reason="jacobi-config-basis-binding-mismatch",
            )
        if (
            metric.disposition is not MetricDisposition.QUALIFIED
            or (riesz.disposition is not MetricDisposition.QUALIFIED)
            or riesz.metric != ObjectIdentity.from_record(metric.metric_id, metric)
        ):
            return _refused(
                witness_id=witness_id,
                member=member,
                local_step=local_step,
                metric=metric,
                riesz=riesz,
                config=config,
                disposition=JacobiDisposition.METRIC_REFUSED,
                reason="metric-or-riesz-not-qualified-or-not-bound",
            )
        if metric.state_basis != member.state_basis:
            return _refused(
                witness_id=witness_id,
                member=member,
                local_step=local_step,
                metric=metric,
                riesz=riesz,
                config=config,
                disposition=JacobiDisposition.METRIC_REFUSED,
                reason="metric-state-basis-differs-from-member",
            )
        qualification = member.qualification_config
        diagnostics = member.diagnostics
        member_stable = (
            member.disposition is ControlledIOProductDisposition.SUPPORTED
            and bool(member.qualification_receipts)
            and not diagnostics.method_reason_codes
            and diagnostics.maximum_spectral_radius.value <= qualification.maximum_spectral_radius
            and diagnostics.maximum_condition_number.value <= qualification.maximum_condition_number
            and diagnostics.observability_rank >= qualification.minimum_observability_rank
        )
        if not member_stable:
            return _refused(
                witness_id=witness_id,
                member=member,
                local_step=local_step,
                metric=metric,
                riesz=riesz,
                config=config,
                disposition=JacobiDisposition.MEMBER_UNSTABLE,
                reason="controlled-io-member-fails-frozen-stability-rule",
            )
        state_ids = member.state_basis.coordinate_ids
        if local_operator.row_coordinate_ids != state_ids or (
            local_operator.column_coordinate_ids != state_ids
        ):
            raise ValueError("Jacobi local operator must use the member state basis")
        g = metric.matrix.as_array()
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (g + g.T))
        sqrt_g = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
        inverse_sqrt_g = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
        whitened_operator = sqrt_g @ local_operator.as_array() @ inverse_sqrt_g
        symmetric = 0.5 * (whitened_operator + whitened_operator.T)
        if not np.allclose(symmetric, symmetric.T, atol=float(config.symmetry_tolerance)):
            return _refused(
                witness_id=witness_id,
                member=member,
                local_step=local_step,
                metric=metric,
                riesz=riesz,
                config=config,
                disposition=JacobiDisposition.UNEVALUABLE,
                reason="whitened-symmetric-part-not-numerically-symmetric",
            )
        assert riesz.whitened_seed is not None
        q0 = riesz.whitened_seed.as_array()
        tolerance = float(config.rank_tolerance)
        bases: list[np.ndarray] = [q0]
        diagonal: list[np.ndarray] = []
        couplings: list[np.ndarray] = []
        deflations: list[int] = []
        maximum_recurrence_error = 0.0
        maximum_blocks = min(config.maximum_order + 1, len(state_ids))
        for order in range(maximum_blocks):
            q = bases[order]
            a = q.T @ symmetric @ q
            a = 0.5 * (a + a.T)
            diagonal.append(a)
            residual = symmetric @ q - q @ a
            if order > 0:
                residual -= bases[order - 1] @ couplings[order - 1].T
            # Full deterministic reorthogonalization prevents false recurrence.
            for basis in bases:
                residual -= basis @ (basis.T @ residual)
            if order == maximum_blocks - 1:
                break
            next_q, singular = _canonical_qr(residual, tolerance)
            if next_q.shape[1] == 0:
                maximum_recurrence_error = max(
                    maximum_recurrence_error,
                    float(np.linalg.norm(residual, ord="fro")),
                )
                deflations.append(order + 1)
                break
            coupling = next_q.T @ residual
            maximum_recurrence_error = max(
                maximum_recurrence_error,
                float(np.linalg.norm(residual - next_q @ coupling, ord="fro")),
            )
            if int(np.sum(singular > tolerance)) < q.shape[1]:
                deflations.append(order + 1)
            couplings.append(coupling)
            bases.append(next_q)
        if maximum_recurrence_error > float(config.recurrence_tolerance):
            return _refused(
                witness_id=witness_id,
                member=member,
                local_step=local_step,
                metric=metric,
                riesz=riesz,
                config=config,
                disposition=JacobiDisposition.UNEVALUABLE,
                reason="jacobi-recurrence-residual-failed",
            )
        block_sizes = [value.shape[1] for value in bases]
        offsets = np.cumsum([0, *block_sizes])
        jacobi = np.zeros((int(offsets[-1]), int(offsets[-1])), dtype=np.float64)
        for index, a in enumerate(diagonal):
            start, stop = int(offsets[index]), int(offsets[index + 1])
            jacobi[start:stop, start:stop] = a
            if index < len(couplings):
                next_start, next_stop = int(offsets[index + 1]), int(offsets[index + 2])
                b = couplings[index]
                jacobi[next_start:next_stop, start:stop] = b
                jacobi[start:stop, next_start:next_stop] = b.T
        receiver_ids = riesz.receiver_coordinate_ids
        moments: list[JacobiMoment] = []
        maximum_error = 0.0
        selector = np.zeros((jacobi.shape[0], len(receiver_ids)), dtype=np.float64)
        selector[: len(receiver_ids), :] = np.eye(len(receiver_ids))
        direct_power = np.eye(symmetric.shape[0], dtype=np.float64)
        jacobi_power = np.eye(jacobi.shape[0], dtype=np.float64)
        for order in range(2 * len(bases) + 1):
            direct = q0.T @ direct_power @ q0
            reconstructed = selector.T @ jacobi_power @ selector
            error = float(np.linalg.norm(direct - reconstructed, ord="fro"))
            maximum_error = max(maximum_error, error)
            moments.append(
                JacobiMoment(
                    moment_id=f"moment.{witness_id}.{order:04d}",
                    order=order,
                    direct=CanonicalMatrix.from_array(
                        matrix_id=f"moment-direct.{witness_id}.{order:04d}",
                        row_coordinate_ids=receiver_ids,
                        column_coordinate_ids=receiver_ids,
                        values=direct,
                    ),
                    reconstructed=CanonicalMatrix.from_array(
                        matrix_id=f"moment-reconstructed.{witness_id}.{order:04d}",
                        row_coordinate_ids=receiver_ids,
                        column_coordinate_ids=receiver_ids,
                        values=reconstructed,
                    ),
                    reconstruction_error=NamedDecimal(
                        value_id=f"moment-error.{witness_id}.{order:04d}",
                        value=decimal_from_float(error),
                        unit="1",
                    ),
                )
            )
            direct_power = direct_power @ symmetric
            jacobi_power = jacobi_power @ jacobi
        if maximum_error > float(config.moment_tolerance):
            return _refused(
                witness_id=witness_id,
                member=member,
                local_step=local_step,
                metric=metric,
                riesz=riesz,
                config=config,
                disposition=JacobiDisposition.MOMENT_MISMATCH,
                reason="jacobi-moment-reconstruction-failed",
            )
        blocks = tuple(
            JacobiBlock(
                block_id=f"jacobi-block.{witness_id}.{index:04d}",
                order_index=index,
                diagonal=CanonicalMatrix.from_array(
                    matrix_id=f"jacobi-a.{witness_id}.{index:04d}",
                    row_coordinate_ids=tuple(
                        f"block-{index:04d}-coordinate-{item:04d}"
                        for item in range(diagonal[index].shape[0])
                    ),
                    column_coordinate_ids=tuple(
                        f"block-{index:04d}-coordinate-{item:04d}"
                        for item in range(diagonal[index].shape[1])
                    ),
                    values=diagonal[index],
                ),
                coupling_to_next=(
                    CanonicalMatrix.from_array(
                        matrix_id=f"jacobi-b.{witness_id}.{index:04d}",
                        row_coordinate_ids=tuple(
                            f"block-{index + 1:04d}-coordinate-{item:04d}"
                            for item in range(couplings[index].shape[0])
                        ),
                        column_coordinate_ids=tuple(
                            f"block-{index:04d}-coordinate-{item:04d}"
                            for item in range(couplings[index].shape[1])
                        ),
                        values=couplings[index],
                    )
                    if index < len(couplings)
                    else None
                ),
            )
            for index in range(len(diagonal))
        )
        return JacobiWitness(
            witness_id=witness_id,
            controlled_io_member=ObjectIdentity.from_record(member.member_record_id, member),
            local_operator_step=ObjectIdentity.from_record(local_step.step_id, local_step),
            local_operator=ObjectIdentity.from_record(local_operator.matrix_id, local_operator),
            metric=ObjectIdentity.from_record(metric.metric_id, metric),
            riesz_family=ObjectIdentity.from_record(riesz.family_id, riesz),
            receiver_coordinate_ids=receiver_ids,
            local_operator_clock_id=config.local_operator_clock_id,
            gauge_convention=config.gauge_convention,
            deflation_rule=config.deflation_rule,
            moment_norm=config.moment_norm,
            blocks=blocks,
            moments=tuple(moments),
            block_ranks=tuple(block_sizes),
            deflated_order_indices=tuple(deflations),
            maximum_reconstruction_error=NamedDecimal(
                value_id=f"jacobi-reconstruction-error.{witness_id}",
                value=decimal_from_float(maximum_error),
                unit="1",
            ),
            maximum_recurrence_error=NamedDecimal(
                value_id=f"jacobi-recurrence-error.{witness_id}",
                value=decimal_from_float(maximum_recurrence_error),
                unit="1",
            ),
            disposition=JacobiDisposition.SUPPORTED,
            evidence_link_ids=tuple(
                sorted({*metric.evidence_link_ids, *member.diagnostics.evidence_link_ids})
            ),
            reason_codes=(),
        )
