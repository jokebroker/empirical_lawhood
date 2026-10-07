"Substrate-neutral contracts for controlled input--output response models.\n\nThe records in this module contain method evidence, not generic qualification,\nadmission, reachability, or controller decisions.  Dense numerical objects\nare deliberately small canonical Decimal matrices so truth-known conformance is\ndisk independent and payload publication can remain content addressed.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

import numpy as np
import numpy.typing as npt

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


FloatMatrix = npt.NDArray[np.float64]


def decimal_from_float(value: float) -> Decimal:
    """Deterministic finite float-to-canonical-Decimal boundary."""

    if not np.isfinite(value):
        raise ValueError("controlled-IO numerical result must be finite")
    return Decimal(format(float(value), ".17g"))


class ControlledIOProductDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"
    REFUSED = "REFUSED"


class GammaRule(StrEnum):
    ZERO = "ZERO"
    FIXED_DECLARED_CONTRIBUTION = "FIXED_DECLARED_CONTRIBUTION"


@dataclass(frozen=True, slots=True)
class ControlledIOQualificationConfig(CanonicalRecord):
    """Frozen method thresholds; profile evaluators, not producers, apply them."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-qualification-config'

    config_id: str
    state_dimension: int
    input_dimension: int
    receiver_dimension: int
    maximum_residual_norm: Decimal
    maximum_held_out_prediction_error: Decimal
    maximum_spectral_radius: Decimal
    maximum_condition_number: Decimal
    minimum_controllability_rank: int
    minimum_observability_rank: int
    gamma_rule: GammaRule
    reference_trajectory: ObjectIdentity
    reference_action_word: ObjectIdentity
    clock_contract: ObjectIdentity
    stability_rule_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        for name, dimension in (
            ("state_dimension", self.state_dimension),
            ("input_dimension", self.input_dimension),
            ("receiver_dimension", self.receiver_dimension),
        ):
            if dimension <= 0:
                raise ValueError(f"{name} must be positive")
        for name, threshold in (
            ("maximum_residual_norm", self.maximum_residual_norm),
            ("maximum_held_out_prediction_error", self.maximum_held_out_prediction_error),
            ("maximum_spectral_radius", self.maximum_spectral_radius),
            ("maximum_condition_number", self.maximum_condition_number),
        ):
            validate_decimal(threshold, field_name=name, minimum=Decimal(0))
        if self.maximum_spectral_radius <= 0 or self.maximum_condition_number < 1:
            raise ValueError("controlled-IO stability/conditioning bounds are invalid")
        for name, value in (
            ("minimum_controllability_rank", self.minimum_controllability_rank),
            ("minimum_observability_rank", self.minimum_observability_rank),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.reference_action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("controlled-IO reference action must be an exact ActionWord")
        validate_stable_id(self.stability_rule_id, field_name="stability_rule_id")


@dataclass(frozen=True, slots=True)
class CoordinateBasis(CanonicalRecord):
    """Ordered native coordinate basis with per-coordinate unit and frame."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/coordinate-basis'

    basis_id: str
    coordinate_ids: tuple[str, ...]
    native_units: tuple[str, ...]
    native_frames: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.basis_id, field_name="basis_id")
        require_sorted_unique_strings(
            self.coordinate_ids,
            field_name="coordinate_ids",
            allow_empty=False,
        )
        if len(self.native_units) != len(self.coordinate_ids):
            raise ValueError("coordinate basis requires one native unit per coordinate")
        if len(self.native_frames) != len(self.coordinate_ids):
            raise ValueError("coordinate basis requires one native frame per coordinate")
        for name, values in (
            ("native_units", self.native_units),
            ("native_frames", self.native_frames),
        ):
            if any(not value.strip() for value in values):
                raise ValueError(f"{name} cannot contain an empty value")

    @property
    def dimension(self) -> int:
        return len(self.coordinate_ids)


@dataclass(frozen=True, slots=True)
class CanonicalMatrix(CanonicalRecord):
    """Small exact-shape matrix in declared ordered row/column bases."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/canonical-matrix'

    matrix_id: str
    row_coordinate_ids: tuple[str, ...]
    column_coordinate_ids: tuple[str, ...]
    values: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.matrix_id, field_name="matrix_id")
        for name, coordinates in (
            ("row_coordinate_ids", self.row_coordinate_ids),
            ("column_coordinate_ids", self.column_coordinate_ids),
        ):
            require_sorted_unique_strings(coordinates, field_name=name, allow_empty=False)
        if len(self.values) != len(self.row_coordinate_ids) * len(self.column_coordinate_ids):
            raise ValueError("canonical matrix values do not match its declared shape")
        for value in self.values:
            validate_decimal(value, field_name="values")

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.row_coordinate_ids), len(self.column_coordinate_ids))

    def as_array(self) -> FloatMatrix:
        return np.asarray(tuple(float(value) for value in self.values), dtype=np.float64).reshape(
            self.shape
        )

    @classmethod
    def from_array(
        cls,
        *,
        matrix_id: str,
        row_coordinate_ids: tuple[str, ...],
        column_coordinate_ids: tuple[str, ...],
        values: npt.ArrayLike,
    ) -> CanonicalMatrix:
        array = np.asarray(values, dtype=np.float64)
        expected = (len(row_coordinate_ids), len(column_coordinate_ids))
        if array.shape != expected or not np.all(np.isfinite(array)):
            raise ValueError("matrix array has the wrong shape or nonfinite values")
        return cls(
            matrix_id=matrix_id,
            row_coordinate_ids=row_coordinate_ids,
            column_coordinate_ids=column_coordinate_ids,
            values=tuple(decimal_from_float(value) for value in array.ravel()),
        )


@dataclass(frozen=True, slots=True)
class CanonicalVector(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/canonical-vector'

    vector_id: str
    coordinate_ids: tuple[str, ...]
    values: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.vector_id, field_name="vector_id")
        require_sorted_unique_strings(
            self.coordinate_ids,
            field_name="coordinate_ids",
            allow_empty=False,
        )
        if len(self.values) != len(self.coordinate_ids):
            raise ValueError("canonical vector values do not match its basis")
        for value in self.values:
            validate_decimal(value, field_name="values")

    def as_array(self) -> npt.NDArray[np.float64]:
        return np.asarray(tuple(float(value) for value in self.values), dtype=np.float64)

    @classmethod
    def zeros(cls, *, vector_id: str, coordinate_ids: tuple[str, ...]) -> CanonicalVector:
        return cls(
            vector_id=vector_id,
            coordinate_ids=coordinate_ids,
            values=tuple(Decimal(0) for _ in coordinate_ids),
        )


@dataclass(frozen=True, slots=True)
class ControlledIOStep(CanonicalRecord):
    "One affine time-local step of the declared controlled-IO model."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-step'

    step_id: str
    step_index: int
    state_transition: CanonicalMatrix
    realized_input_map: CanonicalMatrix
    receiver_map: CanonicalMatrix
    state_affine_term: CanonicalVector
    receiver_affine_term: CanonicalVector
    gamma_rule: GammaRule
    fixed_gamma_contribution: CanonicalVector
    state_clock_id: str
    input_clock_id: str
    receiver_clock_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.step_id, field_name="step_id")
        if self.step_index < 0:
            raise ValueError("controlled-IO step index must be nonnegative")
        for name, value in (
            ("state_clock_id", self.state_clock_id),
            ("input_clock_id", self.input_clock_id),
            ("receiver_clock_id", self.receiver_clock_id),
        ):
            validate_stable_id(value, field_name=name)
        state_ids = self.state_transition.row_coordinate_ids
        if self.state_transition.column_coordinate_ids != state_ids:
            raise ValueError("A_k must be square in the declared state basis")
        if self.realized_input_map.row_coordinate_ids != state_ids:
            raise ValueError("B_k rows must use the declared state basis")
        if self.receiver_map.column_coordinate_ids != state_ids:
            raise ValueError("C_k columns must use the declared state basis")
        if self.state_affine_term.coordinate_ids != state_ids:
            raise ValueError("state affine term must use the state basis")
        if self.fixed_gamma_contribution.coordinate_ids != state_ids:
            raise ValueError("Gamma contribution must use the state basis")
        if self.receiver_affine_term.coordinate_ids != self.receiver_map.row_coordinate_ids:
            raise ValueError("receiver affine term must use the receiver basis")
        gamma = self.fixed_gamma_contribution.as_array()
        if self.gamma_rule is GammaRule.ZERO and np.any(gamma != 0.0):
            raise ValueError("ZERO Gamma rule cannot retain a fixed contribution")


@dataclass(frozen=True, slots=True)
class ControlledIOMemberDiagnostics(CanonicalRecord):
    """Raw method facts; deliberately no generic PASS or obligation status."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-member-diagnostics'

    diagnostics_id: str
    residual_norm: NamedDecimal
    held_out_prediction_error: NamedDecimal
    maximum_spectral_radius: NamedDecimal
    controllability_rank: int
    observability_rank: int
    maximum_condition_number: NamedDecimal
    evidence_link_ids: tuple[str, ...]
    method_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.diagnostics_id, field_name="diagnostics_id")
        if self.controllability_rank < 0 or self.observability_rank < 0:
            raise ValueError("controlled-IO ranks must be nonnegative")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.method_reason_codes,
            field_name="method_reason_codes",
        )


@dataclass(frozen=True, slots=True)
class ControlledIOMember(CanonicalRecord):
    """One denominator-local candidate version and refinement-qualified view set."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-member'

    member_record_id: str
    prepared_denominator_id: str
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_ids: tuple[str, ...]
    support_cell_ids: tuple[str, ...]
    state_basis: CoordinateBasis
    input_basis: CoordinateBasis
    receiver_basis: CoordinateBasis
    steps: tuple[ControlledIOStep, ...]
    qualification_config: ControlledIOQualificationConfig
    reference_trajectory: ObjectIdentity
    reference_action_word: ObjectIdentity
    action_words: tuple[OccurrenceActionWord, ...]
    retained_history_id: str
    horizon_id: str
    diagnostics: ControlledIOMemberDiagnostics
    qualification_receipts: tuple[ObjectIdentity, ...]
    disposition: ControlledIOProductDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("member_record_id", self.member_record_id),
            ("prepared_denominator_id", self.prepared_denominator_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("retained_history_id", self.retained_history_id),
            ("horizon_id", self.horizon_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("qualification_view_ids", self.qualification_view_ids),
            ("support_cell_ids", self.support_cell_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        config = self.qualification_config
        if (
            config.state_dimension != self.state_basis.dimension
            or config.input_dimension != self.input_basis.dimension
            or config.receiver_dimension != self.receiver_basis.dimension
        ):
            raise ValueError("controlled-IO config dimensions differ from member bases")
        if config.reference_trajectory != self.reference_trajectory:
            raise ValueError("controlled-IO reference trajectory differs from its config")
        if config.reference_action_word != self.reference_action_word:
            raise ValueError("controlled-IO reference action differs from its config")
        if self.reference_action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("controlled-IO member reference action must be an ActionWord")
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        if not self.action_words:
            raise ValueError("controlled-IO member requires exact current ActionWords")
        action_identities = tuple(
            ObjectIdentity.from_record(value.word_id, value) for value in self.action_words
        )
        if self.reference_action_word not in action_identities:
            raise ValueError("reference action is absent from the controlled-IO action roster")
        if any(
            value.denominator_id != self.prepared_denominator_id
            or value.retained_history_id != self.retained_history_id
            or value.horizon_id != self.horizon_id
            for value in self.action_words
        ):
            raise ValueError("controlled-IO action roster changes denominator/history/horizon")
        step_ids = tuple(value.step_id for value in self.steps)
        if len(set(step_ids)) != len(step_ids):
            raise ValueError("controlled-IO step IDs must be unique")
        if self.steps != tuple(sorted(self.steps, key=lambda value: value.step_index)):
            raise ValueError("controlled-IO steps must be ordered by time index")
        if tuple(value.step_index for value in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("controlled-IO steps must cover contiguous time indices from zero")
        for name, clock_ids in (
            ("state_clock_ids", tuple(value.state_clock_id for value in self.steps)),
            ("input_clock_ids", tuple(value.input_clock_id for value in self.steps)),
            ("receiver_clock_ids", tuple(value.receiver_clock_id for value in self.steps)),
        ):
            if not clock_ids:
                raise ValueError(f"controlled-IO {name} must retain temporal order")
        for step in self.steps:
            if step.state_transition.row_coordinate_ids != self.state_basis.coordinate_ids:
                raise ValueError("controlled-IO A_k uses another state basis")
            if step.realized_input_map.column_coordinate_ids != self.input_basis.coordinate_ids:
                raise ValueError("controlled-IO B_k uses another input basis")
            if step.receiver_map.row_coordinate_ids != self.receiver_basis.coordinate_ids:
                raise ValueError("controlled-IO C_k uses another receiver basis")
            if step.gamma_rule is not config.gamma_rule:
                raise ValueError("controlled-IO Gamma rule differs from its frozen config")
        require_sorted_unique_ids(
            self.qualification_receipts,
            attribute="object_id",
            field_name="qualification_receipts",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is ControlledIOProductDisposition.SUPPORTED:
            if len(self.steps) < 2:
                raise ValueError("supported controlled-IO member requires a nonempty horizon")
            if not self.qualification_receipts or self.reason_codes:
                raise ValueError("supported controlled-IO member requires receipts and no reasons")
        elif not self.reason_codes:
            raise ValueError("non-supported controlled-IO member requires reasons")


@dataclass(frozen=True, slots=True)
class ControlledIOVersionSet(CanonicalRecord):
    """Finite compatibility set; never a pooled coefficient model."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-version-set'

    version_set_id: str
    physical_independent_unit_ids: tuple[str, ...]
    members: tuple[ControlledIOMember, ...]
    accepted_member_record_ids: tuple[str, ...]
    rejected_member_record_ids: tuple[str, ...]
    coverage_claim_id: str | None
    member_stability_contract_id: str
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.version_set_id, field_name="version_set_id")
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
            allow_empty=False,
        )
        if self.coverage_claim_id is not None:
            validate_stable_id(self.coverage_claim_id, field_name="coverage_claim_id")
        validate_stable_id(
            self.member_stability_contract_id,
            field_name="member_stability_contract_id",
        )
        require_sorted_unique_ids(
            self.members,
            attribute="member_record_id",
            field_name="members",
        )
        if not self.members:
            raise ValueError("controlled-IO version set requires a complete member roster")
        for name, values in (
            ("accepted_member_record_ids", self.accepted_member_record_ids),
            ("rejected_member_record_ids", self.rejected_member_record_ids),
            ("evidence_link_ids", self.evidence_link_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        roster = {value.member_record_id for value in self.members}
        accepted = {
            value.member_record_id
            for value in self.members
            if value.disposition is ControlledIOProductDisposition.SUPPORTED
        }
        if set(self.accepted_member_record_ids) != accepted:
            raise ValueError("controlled-IO accepted ledger differs from member dispositions")
        if set(self.rejected_member_record_ids) != roster - accepted:
            raise ValueError("controlled-IO rejection ledger is incomplete")
        axis_pairs = {
            (value.denominator_member_id, value.candidate_version_id) for value in self.members
        }
        if len(axis_pairs) != len(self.members):
            raise ValueError("controlled-IO set duplicates a member/version coordinate")


@dataclass(frozen=True, slots=True)
class MarkovKernel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/markov-kernel'

    kernel_id: str
    receiver_step_index: int
    input_step_index: int
    value: CanonicalMatrix

    def __post_init__(self) -> None:
        validate_stable_id(self.kernel_id, field_name="kernel_id")
        if self.input_step_index < 0 or self.receiver_step_index <= self.input_step_index:
            raise ValueError("Markov kernel requires receiver index strictly after input index")


@dataclass(frozen=True, slots=True)
class MarkovKernelFamily(CanonicalRecord):
    """Exact finite-horizon C_j Phi_A(j,l+1) B_l product family."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/markov-kernel-family'

    family_id: str
    controlled_io_member: ObjectIdentity
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_ids: tuple[str, ...]
    action_words: tuple[ObjectIdentity, ...]
    support_cell_ids: tuple[str, ...]
    retained_history_id: str
    horizon_id: str
    state_clock_ids: tuple[str, ...]
    input_clock_ids: tuple[str, ...]
    receiver_clock_ids: tuple[str, ...]
    kernels: tuple[MarkovKernel, ...]
    finite_horizon_map: CanonicalMatrix | None
    disposition: ControlledIOProductDisposition
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("family_id", self.family_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("retained_history_id", self.retained_history_id),
            ("horizon_id", self.horizon_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.controlled_io_member.object_schema != ControlledIOMember.SCHEMA:
            raise ValueError("Markov family requires one exact controlled-IO member")
        for name, values, allow_empty in (
            ("qualification_view_ids", self.qualification_view_ids, False),
            ("support_cell_ids", self.support_cell_ids, False),
            ("evidence_link_ids", self.evidence_link_ids, True),
            ("reason_codes", self.reason_codes, True),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=allow_empty)
        for name, values in (
            ("state_clock_ids", self.state_clock_ids),
            ("input_clock_ids", self.input_clock_ids),
            ("receiver_clock_ids", self.receiver_clock_ids),
        ):
            if not values:
                raise ValueError(f"{name} must retain a complete ordered clock roster")
            for value in values:
                validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.action_words,
            attribute="object_id",
            field_name="action_words",
        )
        if not self.action_words or any(
            value.object_schema != OccurrenceActionWord.SCHEMA for value in self.action_words
        ):
            raise ValueError("Markov family requires exact ActionWord identities")
        require_sorted_unique_ids(self.kernels, attribute="kernel_id", field_name="kernels")
        if self.disposition is ControlledIOProductDisposition.SUPPORTED:
            if not self.kernels or self.finite_horizon_map is None:
                raise ValueError("supported Markov family requires kernels and stacked map")
            if not self.evidence_link_ids or self.reason_codes:
                raise ValueError("supported Markov family requires evidence and no reasons")
        else:
            if self.kernels or self.finite_horizon_map is not None or not self.reason_codes:
                raise ValueError("refused Markov family cannot carry controlled products")


def identity_matrix(coordinate_ids: tuple[str, ...]) -> FloatMatrix:
    """Exact empty transition product in the declared state basis."""

    return np.eye(len(coordinate_ids), dtype=np.float64)
