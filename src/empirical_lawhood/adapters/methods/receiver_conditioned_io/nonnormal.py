"""Full-map singular, transient, pathwise, sink, and Jacobi-residual audit."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
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
    CanonicalVector,
    ControlledIOMember,
    FloatMatrix,
    MarkovKernelFamily,
    decimal_from_float,
)
from .jacobi import JacobiDisposition, JacobiWitness
from .receiver_metric import MetricDisposition, StateMetric


class NonnormalAuditDisposition(StrEnum):
    FAVORABLE = "FAVORABLE"
    ADVERSE = "ADVERSE"
    UNEVALUABLE = "UNEVALUABLE"
    REFUSED = "REFUSED"


@dataclass(frozen=True, slots=True)
class NonnormalAuditConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/nonnormal-audit-config'

    config_id: str
    maximum_transient_gain: Decimal
    maximum_nonnormal_residual: Decimal
    maximum_witness_residual: Decimal
    maximum_hidden_sink_gain: Decimal
    minimum_pathwise_margin: Decimal
    maximum_witness_residual_unit: str
    maximum_hidden_sink_gain_unit: str
    minimum_pathwise_margin_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        for name, value in (
            ("maximum_transient_gain", self.maximum_transient_gain),
            ("maximum_nonnormal_residual", self.maximum_nonnormal_residual),
            ("maximum_witness_residual", self.maximum_witness_residual),
            ("maximum_hidden_sink_gain", self.maximum_hidden_sink_gain),
            ("minimum_pathwise_margin", self.minimum_pathwise_margin),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        for name, unit_value in (
            ("maximum_witness_residual_unit", self.maximum_witness_residual_unit),
            ("maximum_hidden_sink_gain_unit", self.maximum_hidden_sink_gain_unit),
            ("minimum_pathwise_margin_unit", self.minimum_pathwise_margin_unit),
        ):
            if not unit_value.strip():
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class PrefixPathwiseMargin(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/prefix-pathwise-margin'

    margin_id: str
    action_word: ObjectIdentity
    prefix_index: int
    prefix_support_id: str
    margin: Decimal
    native_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.margin_id, field_name="margin_id")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("pathwise margin requires an exact ActionWord")
        if self.prefix_index < 0:
            raise ValueError("pathwise prefix index must be nonnegative")
        validate_stable_id(self.prefix_support_id, field_name="prefix_support_id")
        validate_decimal(self.margin, field_name="margin")
        if not self.native_unit.strip():
            raise ValueError("pathwise margin native unit must not be empty")


@dataclass(frozen=True, slots=True)
class NonnormalAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/nonnormal-audit'

    audit_id: str
    controlled_io_member: ObjectIdentity
    markov_family: ObjectIdentity
    state_metric: ObjectIdentity
    audit_config: ObjectIdentity
    jacobi_witness: ObjectIdentity | None
    hidden_sink_map: ObjectIdentity | None
    controlled_left_singular_vectors: CanonicalMatrix | None
    controlled_singular_values: CanonicalVector | None
    controlled_right_singular_vectors: CanonicalMatrix | None
    largest_controlled_singular_value: NamedDecimal
    maximum_state_transient_gain: NamedDecimal
    nonnormal_residual: NamedDecimal
    witness_residual: NamedDecimal
    hidden_sink_gain: NamedDecimal
    pathwise_margins: tuple[PrefixPathwiseMargin, ...]
    disposition: NonnormalAuditDisposition
    vetoes_mechanistic_interpretation: bool
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        if self.controlled_io_member.object_schema != ControlledIOMember.SCHEMA:
            raise ValueError("nonnormal audit requires a controlled-IO member")
        if self.markov_family.object_schema != MarkovKernelFamily.SCHEMA:
            raise ValueError("nonnormal audit requires a Markov-kernel family")
        if self.state_metric.object_schema != StateMetric.SCHEMA:
            raise ValueError("nonnormal audit requires an exact state metric")
        if self.audit_config.object_schema != NonnormalAuditConfig.SCHEMA:
            raise ValueError("nonnormal audit requires an exact audit config")
        if self.jacobi_witness is not None and (
            self.jacobi_witness.object_schema != JacobiWitness.SCHEMA
        ):
            raise ValueError("nonnormal audit Jacobi operand has another schema")
        if self.hidden_sink_map is not None and (
            self.hidden_sink_map.object_schema != CanonicalMatrix.SCHEMA
        ):
            raise ValueError("nonnormal audit hidden-sink operand has another schema")
        singular_objects = (
            self.controlled_left_singular_vectors,
            self.controlled_singular_values,
            self.controlled_right_singular_vectors,
        )
        if any(value is None for value in singular_objects) and any(
            value is not None for value in singular_objects
        ):
            raise ValueError("controlled singular evidence must be all present or all absent")
        require_sorted_unique_ids(
            self.pathwise_margins,
            attribute="margin_id",
            field_name="pathwise_margins",
        )
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.disposition
            in {
                NonnormalAuditDisposition.FAVORABLE,
                NonnormalAuditDisposition.ADVERSE,
            }
            and self.controlled_singular_values is None
        ):
            raise ValueError("computed audit requires full-map singular directions")
        if self.disposition is NonnormalAuditDisposition.FAVORABLE:
            if self.vetoes_mechanistic_interpretation:
                raise ValueError("favorable audit cannot veto mechanistic interpretation")
            if not self.evidence_link_ids or self.reason_codes:
                raise ValueError("favorable audit requires evidence and no reasons")
        else:
            if not self.vetoes_mechanistic_interpretation or not self.reason_codes:
                raise ValueError("non-favorable full-map audit must retain a typed veto")


def _metric_factors(metric: StateMetric) -> tuple[FloatMatrix, FloatMatrix]:
    g = metric.matrix.as_array()
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (g + g.T))
    sqrt_g = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_sqrt_g = (eigenvectors * (1.0 / np.sqrt(eigenvalues))) @ eigenvectors.T
    return np.asarray(sqrt_g, dtype=np.float64), np.asarray(inverse_sqrt_g, dtype=np.float64)


def _controlled_map(
    member: ControlledIOMember,
    transitions: tuple[FloatMatrix, ...],
    output_map: FloatMatrix | None = None,
) -> FloatMatrix:
    """Stack exact C_j Phi_A(j,l+1) B_l or a declared sink analogue."""

    receiver_count = (
        output_map.shape[0] if output_map is not None else member.receiver_basis.dimension
    )
    horizon = len(member.steps) - 1
    result = np.zeros(
        (horizon * receiver_count, horizon * member.input_basis.dimension),
        dtype=np.float64,
    )
    identity = np.eye(member.state_basis.dimension, dtype=np.float64)
    for receiver_index in range(1, len(member.steps)):
        c = (
            output_map
            if output_map is not None
            else member.steps[receiver_index].receiver_map.as_array()
        )
        for input_index in range(receiver_index):
            propagation = identity
            for transition_index in range(input_index + 1, receiver_index):
                propagation = transitions[transition_index] @ propagation
            block = c @ propagation @ member.steps[input_index].realized_input_map.as_array()
            row_start = (receiver_index - 1) * receiver_count
            column_start = input_index * member.input_basis.dimension
            result[
                row_start : row_start + receiver_count,
                column_start : column_start + member.input_basis.dimension,
            ] = block
    return result


def _canonical_svd(
    values: FloatMatrix,
) -> tuple[FloatMatrix, np.ndarray, FloatMatrix]:
    left, singular, right_transpose = np.linalg.svd(values, full_matrices=False)
    right = right_transpose.T
    for index in range(len(singular)):
        left_vector = left[:, index]
        right_vector = right[:, index]
        left_pivot = int(np.argmax(np.abs(left_vector)))
        pivot_value = left_vector[left_pivot]
        if pivot_value == 0.0:
            right_pivot = int(np.argmax(np.abs(right_vector)))
            pivot_value = right_vector[right_pivot]
        if pivot_value < 0:
            left[:, index] *= -1.0
            right[:, index] *= -1.0
    return left, singular, right


def _uniform_unit(values: tuple[str, ...]) -> str | None:
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else None


@dataclass(frozen=True, slots=True)
class NonnormalAuditService:
    def audit(
        self,
        *,
        audit_id: str,
        member: ControlledIOMember,
        markov: MarkovKernelFamily,
        metric: StateMetric,
        witness: JacobiWitness | None,
        sink_map: CanonicalMatrix | None,
        pathwise_margins: tuple[PrefixPathwiseMargin, ...],
        config: NonnormalAuditConfig,
    ) -> NonnormalAudit:
        member_identity = ObjectIdentity.from_record(member.member_record_id, member)
        markov_identity = ObjectIdentity.from_record(markov.family_id, markov)
        metric_identity = ObjectIdentity.from_record(metric.metric_id, metric)
        config_identity = ObjectIdentity.from_record(config.config_id, config)
        witness_identity = (
            ObjectIdentity.from_record(witness.witness_id, witness) if witness is not None else None
        )
        sink_identity = (
            ObjectIdentity.from_record(sink_map.matrix_id, sink_map)
            if sink_map is not None
            else None
        )
        ordered_margins = tuple(sorted(pathwise_margins, key=lambda value: value.margin_id))
        receiver_unit = _uniform_unit(member.receiver_basis.native_units)
        input_unit = _uniform_unit(member.input_basis.native_units)
        controlled_unit = (
            f"{receiver_unit}/{input_unit}"
            if receiver_unit is not None and input_unit is not None
            else "unavailable"
        )

        def refused(reason: str) -> NonnormalAudit:
            def zero(identifier: str, unit: str) -> NamedDecimal:
                return NamedDecimal(
                    value_id=f"{identifier}.{audit_id}",
                    value=Decimal(0),
                    unit=unit,
                )

            return NonnormalAudit(
                audit_id=audit_id,
                controlled_io_member=member_identity,
                markov_family=markov_identity,
                state_metric=metric_identity,
                audit_config=config_identity,
                jacobi_witness=witness_identity,
                hidden_sink_map=sink_identity,
                controlled_left_singular_vectors=None,
                controlled_singular_values=None,
                controlled_right_singular_vectors=None,
                largest_controlled_singular_value=zero(
                    "controlled-singular-unavailable",
                    controlled_unit,
                ),
                maximum_state_transient_gain=zero("state-transient-unavailable", "1"),
                nonnormal_residual=zero("nonnormal-residual-unavailable", "1"),
                witness_residual=zero(
                    "witness-residual-unavailable",
                    config.maximum_witness_residual_unit,
                ),
                hidden_sink_gain=zero(
                    "hidden-sink-unavailable",
                    config.maximum_hidden_sink_gain_unit,
                ),
                pathwise_margins=ordered_margins,
                disposition=NonnormalAuditDisposition.REFUSED,
                vetoes_mechanistic_interpretation=True,
                evidence_link_ids=(),
                reason_codes=(reason,),
            )

        if metric.disposition is not MetricDisposition.QUALIFIED or (
            markov.finite_horizon_map is None
        ):
            return refused("full-map-operands-unavailable")
        if markov.controlled_io_member != member_identity:
            return refused("markov-family-member-identity-mismatch")
        if metric.state_basis != member.state_basis:
            return refused("metric-state-basis-differs-from-member")
        if witness is None or witness.disposition is not JacobiDisposition.SUPPORTED:
            return refused("supported-jacobi-witness-unavailable")
        if (
            witness.controlled_io_member != member_identity
            or witness.metric != metric_identity
            or witness.local_operator_step is None
            or witness.local_operator is None
        ):
            return refused("jacobi-witness-operand-identity-mismatch")
        selected_step = next(
            (
                step
                for step in member.steps
                if ObjectIdentity.from_record(step.step_id, step) == witness.local_operator_step
            ),
            None,
        )
        if selected_step is None or witness.local_operator != ObjectIdentity.from_record(
            selected_step.state_transition.matrix_id,
            selected_step.state_transition,
        ):
            return refused("jacobi-witness-local-operator-mismatch")
        if receiver_unit is None or input_unit is None:
            return refused("controlled-singular-native-unit-confusion")
        if config.maximum_witness_residual_unit != controlled_unit:
            return refused("witness-residual-native-unit-confusion")

        expected_path_coordinates = {
            (
                ObjectIdentity.from_record(word.word_id, word),
                prefix_index,
                prefix_support_id,
            )
            for word in member.action_words
            for prefix_index, prefix_support_id in enumerate(word.prefix_support_ids)
        }
        observed_path_coordinates = {
            (value.action_word, value.prefix_index, value.prefix_support_id)
            for value in ordered_margins
        }
        if observed_path_coordinates != expected_path_coordinates or len(ordered_margins) != len(
            expected_path_coordinates
        ):
            return refused("pathwise-prefix-margin-roster-incomplete")
        if any(
            value.native_unit != config.minimum_pathwise_margin_unit for value in ordered_margins
        ):
            return refused("pathwise-prefix-margin-native-unit-confusion")

        controlled = markov.finite_horizon_map.as_array()
        transitions = tuple(step.state_transition.as_array() for step in member.steps)
        recomputed_controlled = _controlled_map(member, transitions)
        if not np.array_equal(controlled, recomputed_controlled):
            return refused("markov-family-full-map-recomputation-mismatch")
        left, singular, right = _canonical_svd(controlled)
        largest_singular = float(singular[0]) if len(singular) else 0.0
        sqrt_g, inverse_sqrt_g = _metric_factors(metric)
        whitened_transitions = tuple(
            np.asarray(sqrt_g @ transition @ inverse_sqrt_g, dtype=np.float64)
            for transition in transitions
        )
        symmetric_whitened = tuple(
            0.5 * (transition + transition.T) for transition in whitened_transitions
        )
        symmetric_native = tuple(
            np.asarray(inverse_sqrt_g @ transition @ sqrt_g, dtype=np.float64)
            for transition in symmetric_whitened
        )
        nonnormal_residual = max(
            float(np.linalg.norm(transition - symmetric, ord=2))
            for transition, symmetric in zip(
                whitened_transitions,
                symmetric_whitened,
                strict=True,
            )
        )
        maximum_transient = 1.0
        power = np.eye(member.state_basis.dimension, dtype=np.float64)
        for transition in whitened_transitions:
            power = transition @ power
            maximum_transient = max(maximum_transient, float(np.linalg.norm(power, ord=2)))
        symmetric_controlled = _controlled_map(member, symmetric_native)
        witness_residual = float(np.linalg.norm(controlled - symmetric_controlled, ord=2))
        hidden_sink_gain = 0.0
        if sink_map is not None:
            if sink_map.column_coordinate_ids != member.state_basis.coordinate_ids:
                raise ValueError("hidden-sink map must consume the controlled state basis")
            sink_controlled = _controlled_map(
                member,
                transitions,
                output_map=sink_map.as_array(),
            )
            hidden_sink_gain = float(np.linalg.norm(sink_controlled, ord=2))
        minimum_margin = min(float(value.margin) for value in ordered_margins)
        reasons = []
        if maximum_transient > float(config.maximum_transient_gain):
            reasons.append("nonnormal-transient-gain-exceeded")
        if nonnormal_residual > float(config.maximum_nonnormal_residual):
            reasons.append("nonnormal-residual-exceeded")
        if witness_residual > float(config.maximum_witness_residual):
            reasons.append("symmetric-witness-residual-exceeded")
        if hidden_sink_gain > float(config.maximum_hidden_sink_gain):
            reasons.append("hidden-sink-gain-exceeded")
        if minimum_margin < float(config.minimum_pathwise_margin):
            reasons.append("pathwise-prefix-margin-failed")
        disposition = (
            NonnormalAuditDisposition.ADVERSE if reasons else NonnormalAuditDisposition.FAVORABLE
        )
        evidence = tuple(sorted({*member.diagnostics.evidence_link_ids, *markov.evidence_link_ids}))

        def value(identifier: str, raw: float, unit: str) -> NamedDecimal:
            return NamedDecimal(
                value_id=f"{identifier}.{audit_id}",
                value=decimal_from_float(raw),
                unit=unit,
            )

        singular_ids = tuple(
            f"controlled-singular-direction-{index:04d}" for index in range(len(singular))
        )

        return NonnormalAudit(
            audit_id=audit_id,
            controlled_io_member=member_identity,
            markov_family=markov_identity,
            state_metric=metric_identity,
            audit_config=config_identity,
            jacobi_witness=witness_identity,
            hidden_sink_map=sink_identity,
            controlled_left_singular_vectors=CanonicalMatrix.from_array(
                matrix_id=f"controlled-left-singular-vectors.{audit_id}",
                row_coordinate_ids=markov.finite_horizon_map.row_coordinate_ids,
                column_coordinate_ids=singular_ids,
                values=left,
            ),
            controlled_singular_values=CanonicalVector(
                vector_id=f"controlled-singular-values.{audit_id}",
                coordinate_ids=singular_ids,
                values=tuple(decimal_from_float(raw) for raw in singular),
            ),
            controlled_right_singular_vectors=CanonicalMatrix.from_array(
                matrix_id=f"controlled-right-singular-vectors.{audit_id}",
                row_coordinate_ids=markov.finite_horizon_map.column_coordinate_ids,
                column_coordinate_ids=singular_ids,
                values=right,
            ),
            largest_controlled_singular_value=value(
                "controlled-singular",
                largest_singular,
                controlled_unit,
            ),
            maximum_state_transient_gain=value("state-transient", maximum_transient, "1"),
            nonnormal_residual=value("nonnormal-residual", nonnormal_residual, "1"),
            witness_residual=value(
                "witness-residual",
                witness_residual,
                controlled_unit,
            ),
            hidden_sink_gain=value(
                "hidden-sink",
                hidden_sink_gain,
                config.maximum_hidden_sink_gain_unit,
            ),
            pathwise_margins=ordered_margins,
            disposition=disposition,
            vetoes_mechanistic_interpretation=bool(reasons),
            evidence_link_ids=evidence if not reasons else evidence,
            reason_codes=tuple(sorted(reasons)),
        )
