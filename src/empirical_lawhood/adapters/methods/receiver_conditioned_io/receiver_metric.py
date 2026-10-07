"""Positive state-metric qualification and receiver Riesz conversion."""

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
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import CanonicalMatrix, CoordinateBasis, decimal_from_float


class MetricDisposition(StrEnum):
    QUALIFIED = "QUALIFIED"
    NOT_SYMMETRIC = "NOT_SYMMETRIC"
    NOT_POSITIVE_DEFINITE = "NOT_POSITIVE_DEFINITE"
    ILL_CONDITIONED = "ILL_CONDITIONED"
    RECEIVER_RANK_DEFICIENT = "RECEIVER_RANK_DEFICIENT"
    UNEVALUABLE = "UNEVALUABLE"


class WhiteningConvention(StrEnum):
    SYMMETRIC_POSITIVE_SQUARE_ROOT_CANONICAL_QR = "symmetric-positive-square-root-then-canonical-qr"


@dataclass(frozen=True, slots=True)
class StateMetricConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/state-metric-config'

    config_id: str
    state_basis: CoordinateBasis
    receiver_basis: CoordinateBasis
    minimum_eigenvalue: Decimal
    maximum_condition_number: Decimal
    receiver_rank_tolerance: Decimal
    symmetry_tolerance: Decimal
    whitening_convention: WhiteningConvention

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        for name, value in (
            ("minimum_eigenvalue", self.minimum_eigenvalue),
            ("maximum_condition_number", self.maximum_condition_number),
            ("receiver_rank_tolerance", self.receiver_rank_tolerance),
            ("symmetry_tolerance", self.symmetry_tolerance),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if self.minimum_eigenvalue <= 0:
            raise ValueError("metric minimum eigenvalue must be positive")
        if self.maximum_condition_number < 1:
            raise ValueError("metric condition-number bound must be at least one")
        if self.receiver_rank_tolerance <= 0:
            raise ValueError("receiver rank tolerance must be positive")
        if self.symmetry_tolerance <= 0:
            raise ValueError("metric symmetry tolerance must be positive")


@dataclass(frozen=True, slots=True)
class StateMetric(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/state-metric'

    metric_id: str
    state_basis: CoordinateBasis
    matrix: CanonicalMatrix
    config: ObjectIdentity
    minimum_eigenvalue: NamedDecimal
    condition_number: NamedDecimal
    disposition: MetricDisposition
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.metric_id, field_name="metric_id")
        if self.matrix.row_coordinate_ids != self.state_basis.coordinate_ids or (
            self.matrix.column_coordinate_ids != self.state_basis.coordinate_ids
        ):
            raise ValueError("state metric matrix must be square in the state basis")
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is MetricDisposition.QUALIFIED:
            if not self.evidence_link_ids or self.reason_codes:
                raise ValueError("qualified metric requires evidence and no reasons")
        elif not self.reason_codes:
            raise ValueError("unqualified metric requires reasons")


@dataclass(frozen=True, slots=True)
class ReceiverRieszVector(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/receiver-riesz-vector'

    vector_id: str
    receiver_coordinate_id: str
    state_coordinate_ids: tuple[str, ...]
    covector_values: tuple[Decimal, ...]
    riesz_values: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.vector_id, field_name="vector_id")
        validate_stable_id(
            self.receiver_coordinate_id,
            field_name="receiver_coordinate_id",
        )
        require_sorted_unique_strings(
            self.state_coordinate_ids,
            field_name="state_coordinate_ids",
            allow_empty=False,
        )
        dimension = len(self.state_coordinate_ids)
        if len(self.covector_values) != dimension or len(self.riesz_values) != dimension:
            raise ValueError("receiver covector/Riesz vector has the wrong state dimension")
        for value in (*self.covector_values, *self.riesz_values):
            validate_decimal(value, field_name="receiver_riesz_values")


@dataclass(frozen=True, slots=True)
class ReceiverRieszFamily(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/receiver-riesz-family'

    family_id: str
    metric: ObjectIdentity
    receiver_coordinate_ids: tuple[str, ...]
    vectors: tuple[ReceiverRieszVector, ...]
    whitened_seed: CanonicalMatrix | None
    receiver_rank: int
    whitening_convention: WhiteningConvention
    disposition: MetricDisposition
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.family_id, field_name="family_id")
        if self.metric.object_schema != StateMetric.SCHEMA:
            raise ValueError("Riesz family requires one exact state metric")
        require_sorted_unique_strings(
            self.receiver_coordinate_ids,
            field_name="receiver_coordinate_ids",
            allow_empty=False,
        )
        if (
            self.vectors
            and tuple(value.receiver_coordinate_id for value in self.vectors)
            != self.receiver_coordinate_ids
        ):
            raise ValueError("Riesz vector order differs from receiver order")
        if self.receiver_rank < 0 or self.receiver_rank > len(self.receiver_coordinate_ids):
            raise ValueError("receiver rank is outside the receiver block dimension")
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is MetricDisposition.QUALIFIED:
            if self.receiver_rank != len(self.receiver_coordinate_ids):
                raise ValueError("qualified Riesz family requires full receiver-block rank")
            if self.whitened_seed is None or not self.evidence_link_ids or self.reason_codes:
                raise ValueError("qualified Riesz family requires seed evidence")
        elif self.whitened_seed is not None or not self.reason_codes:
            raise ValueError("unqualified Riesz family cannot supply a whitened seed")


def _canonical_sign_columns(values: np.ndarray) -> np.ndarray:
    result = values.copy()
    for column in range(result.shape[1]):
        vector = result[:, column]
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] < 0:
            result[:, column] *= -1.0
    return result


@dataclass(frozen=True, slots=True)
class ReceiverMetricService:
    def qualify_metric(
        self,
        *,
        metric_id: str,
        state_basis: CoordinateBasis,
        matrix: CanonicalMatrix,
        config: StateMetricConfig,
        evidence_link_ids: tuple[str, ...],
    ) -> StateMetric:
        if config.state_basis != state_basis:
            raise ValueError("state basis differs from the metric config")
        if matrix.row_coordinate_ids != state_basis.coordinate_ids or (
            matrix.column_coordinate_ids != state_basis.coordinate_ids
        ):
            raise ValueError("state metric matrix must use the declared state basis")
        values = matrix.as_array()
        asymmetry = float(np.linalg.norm(values - values.T, ord="fro"))
        symmetric = 0.5 * (values + values.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        minimum = float(eigenvalues[0])
        condition = float(eigenvalues[-1] / eigenvalues[0]) if minimum > 0 else float("inf")
        reasons: tuple[str, ...]
        if not np.all(np.isfinite(values)):
            disposition = MetricDisposition.UNEVALUABLE
            reasons = ("metric-nonfinite",)
            minimum_value = Decimal(0)
            condition_value = Decimal(0)
        elif asymmetry > float(config.symmetry_tolerance):
            disposition = MetricDisposition.NOT_SYMMETRIC
            reasons = ("metric-not-symmetric",)
            minimum_value = decimal_from_float(minimum)
            condition_value = (
                Decimal(0) if not np.isfinite(condition) else decimal_from_float(condition)
            )
        elif minimum < float(config.minimum_eigenvalue):
            disposition = MetricDisposition.NOT_POSITIVE_DEFINITE
            reasons = ("metric-not-positive-definite",)
            minimum_value = decimal_from_float(minimum)
            condition_value = (
                Decimal(0) if not np.isfinite(condition) else decimal_from_float(condition)
            )
        elif condition > float(config.maximum_condition_number):
            disposition = MetricDisposition.ILL_CONDITIONED
            reasons = ("metric-ill-conditioned",)
            minimum_value = decimal_from_float(minimum)
            condition_value = decimal_from_float(condition)
        else:
            disposition = MetricDisposition.QUALIFIED
            reasons = ()
            minimum_value = decimal_from_float(minimum)
            condition_value = decimal_from_float(condition)
        return StateMetric(
            metric_id=metric_id,
            state_basis=state_basis,
            matrix=matrix,
            config=ObjectIdentity.from_record(config.config_id, config),
            minimum_eigenvalue=NamedDecimal(
                value_id=f"metric-minimum-eigenvalue.{metric_id}",
                value=minimum_value,
                unit="1",
            ),
            condition_number=NamedDecimal(
                value_id=f"metric-condition-number.{metric_id}",
                value=condition_value,
                unit="1",
            ),
            disposition=disposition,
            evidence_link_ids=evidence_link_ids
            if disposition is MetricDisposition.QUALIFIED
            else (),
            reason_codes=reasons,
        )

    def riesz(
        self,
        *,
        family_id: str,
        metric: StateMetric,
        receiver_coordinate_ids: tuple[str, ...],
        receiver_covectors: CanonicalMatrix,
        config: StateMetricConfig,
    ) -> ReceiverRieszFamily:
        identity = ObjectIdentity.from_record(metric.metric_id, metric)
        if metric.config != ObjectIdentity.from_record(config.config_id, config):
            raise ValueError("Riesz config differs from the metric qualification config")
        if receiver_coordinate_ids != config.receiver_basis.coordinate_ids:
            raise ValueError("receiver covector order differs from the metric config")
        if metric.disposition is not MetricDisposition.QUALIFIED:
            return ReceiverRieszFamily(
                family_id=family_id,
                metric=identity,
                receiver_coordinate_ids=receiver_coordinate_ids,
                vectors=(),
                whitened_seed=None,
                receiver_rank=0,
                whitening_convention=config.whitening_convention,
                disposition=metric.disposition,
                evidence_link_ids=(),
                reason_codes=metric.reason_codes,
            )
        state_ids = metric.state_basis.coordinate_ids
        if receiver_covectors.row_coordinate_ids != state_ids or (
            receiver_covectors.column_coordinate_ids != receiver_coordinate_ids
        ):
            raise ValueError("receiver covectors must be state-by-declared-receiver order")
        g = metric.matrix.as_array()
        q = receiver_covectors.as_array()
        eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (g + g.T))
        inverse = (eigenvectors * (1.0 / eigenvalues)) @ eigenvectors.T
        inverse_sqrt = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
        riesz = inverse @ q
        whitened = inverse_sqrt @ q
        rank = int(np.linalg.matrix_rank(whitened, tol=float(config.receiver_rank_tolerance)))
        vectors = tuple(
            ReceiverRieszVector(
                vector_id=f"riesz.{family_id}.{receiver_id}",
                receiver_coordinate_id=receiver_id,
                state_coordinate_ids=state_ids,
                covector_values=tuple(decimal_from_float(value) for value in q[:, index]),
                riesz_values=tuple(decimal_from_float(value) for value in riesz[:, index]),
            )
            for index, receiver_id in enumerate(receiver_coordinate_ids)
        )
        if rank != len(receiver_coordinate_ids):
            disposition = MetricDisposition.RECEIVER_RANK_DEFICIENT
            return ReceiverRieszFamily(
                family_id=family_id,
                metric=identity,
                receiver_coordinate_ids=receiver_coordinate_ids,
                vectors=vectors,
                whitened_seed=None,
                receiver_rank=rank,
                whitening_convention=config.whitening_convention,
                disposition=disposition,
                evidence_link_ids=(),
                reason_codes=("receiver-block-rank-deficient",),
            )
        seed, _ = np.linalg.qr(whitened, mode="reduced")
        seed = _canonical_sign_columns(seed)
        return ReceiverRieszFamily(
            family_id=family_id,
            metric=identity,
            receiver_coordinate_ids=receiver_coordinate_ids,
            vectors=vectors,
            whitened_seed=CanonicalMatrix.from_array(
                matrix_id=f"whitened-seed.{family_id}",
                row_coordinate_ids=state_ids,
                column_coordinate_ids=receiver_coordinate_ids,
                values=seed,
            ),
            receiver_rank=rank,
            whitening_convention=config.whitening_convention,
            disposition=MetricDisposition.QUALIFIED,
            evidence_link_ids=metric.evidence_link_ids,
            reason_codes=(),
        )
