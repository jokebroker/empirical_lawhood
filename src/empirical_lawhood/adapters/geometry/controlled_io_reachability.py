"Registered full-map admission reachability from exact controlled-I/O members."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.methods.receiver_conditioned_io.contracts import (
    CanonicalMatrix,
    CanonicalVector,
    ControlledIOMember,
    ControlledIOProductDisposition,
    decimal_from_float,
)
from empirical_lawhood.adapters.methods.receiver_conditioned_io.controlled_io import (
    ControlledIOEvaluator,
)
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import ClockCoordinate, ClockTransport
from empirical_lawhood.planning.evidence_geometry import LawEvaluationBindingDisposition, LawMemberEvaluationBinding, ReceiptAdmissionActionFibre, ReceiptAdmissionRawDisposition, ReceiptAdmissionReachabilityReferenceKind, ReceiptAdmissionReceiptProductionPlan, admission_coordinate_outside_support, validate_law_evaluation_binding

from .admission_receipts import AdmissionReachabilityRawInput


@dataclass(frozen=True, slots=True)
class ActionOccurrenceInputBinding(CanonicalRecord):
    """Exact realized occurrence to one controlled-I/O input step/coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/geometry/action-occurrence-input-binding'

    binding_id: str
    occurrence_id: str
    input_step_index: int
    input_coordinate_id: str
    realized_to_input_clock: ClockTransport
    input_coordinate: ClockCoordinate

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("occurrence_id", self.occurrence_id),
            ("input_coordinate_id", self.input_coordinate_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.input_step_index < 0:
            raise ValueError("controlled input step index must be nonnegative")


@dataclass(frozen=True, slots=True)
class ActionWordInputProjectionSpec(CanonicalRecord):
    """Frozen occurrence-safe projection; simultaneous/repeated words do not collapse."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/geometry/action-word-input-projection-spec'

    projection_id: str
    action_word: ObjectIdentity
    occurrence_bindings: tuple[ActionOccurrenceInputBinding, ...]
    accumulation_rule: str

    def __post_init__(self) -> None:
        validate_stable_id(self.projection_id, field_name="projection_id")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("controlled input projection requires an exact ActionWord")
        require_sorted_unique_ids(
            self.occurrence_bindings,
            attribute="binding_id",
            field_name="occurrence_bindings",
        )
        occurrence_ids = tuple(value.occurrence_id for value in self.occurrence_bindings)
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("controlled input projection binds an occurrence more than once")
        if self.accumulation_rule != "sum-realized-values-within-exact-step-coordinate":
            raise ValueError("controlled input projection uses an unknown accumulation rule")


@dataclass(frozen=True, slots=True)
class ControlledOutputDirectionBound(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/geometry/controlled-output-direction-bound'

    constraint_id: str
    output_coordinate_id: str
    native_unit: str
    native_frame: str
    lower: Decimal | None
    upper: Decimal | None

    def __post_init__(self) -> None:
        validate_stable_id(self.constraint_id, field_name="constraint_id")
        validate_stable_id(self.output_coordinate_id, field_name="output_coordinate_id")
        if not self.native_unit.strip():
            raise ValueError("controlled output bound native unit cannot be empty")
        if not self.native_frame.strip():
            raise ValueError("controlled output bound native frame cannot be empty")
        if self.lower is None and self.upper is None:
            raise ValueError("controlled output bound requires a lower or upper value")
        if self.lower is not None:
            validate_decimal(self.lower, field_name="lower")
        if self.upper is not None:
            validate_decimal(self.upper, field_name="upper")
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("controlled output bound is reversed")


@dataclass(frozen=True, slots=True)
class ControlledIOReachabilityRequest(CanonicalRecord):
    """Frozen method request; it contains no final reachability status."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/geometry/controlled-io-reachability-request'

    request_id: str
    planned_coordinate_id: str
    direction_id: str
    controlled_io_member: ObjectIdentity
    law_evaluation_binding: LawMemberEvaluationBinding
    reference_kind: ReceiptAdmissionReachabilityReferenceKind
    active_projection: ActionWordInputProjectionSpec
    qualified_hold_action_fibre: ObjectIdentity | None
    qualified_hold_law_evaluation_binding: LawMemberEvaluationBinding | None
    hold_projection: ActionWordInputProjectionSpec | None
    declared_native_reference_direction_id: str | None
    declared_native_reference: CanonicalVector | None
    output_bounds: tuple[ControlledOutputDirectionBound, ...]
    rank_tolerance: Decimal
    minimum_output_norm: Decimal
    maximum_condition_number: Decimal
    method: ExecutableReference
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("planned_coordinate_id", self.planned_coordinate_id),
            ("direction_id", self.direction_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.declared_native_reference_direction_id is not None:
            validate_stable_id(
                self.declared_native_reference_direction_id,
                field_name="declared_native_reference_direction_id",
            )
        if self.controlled_io_member.object_schema != ControlledIOMember.SCHEMA:
            raise ValueError("controlled reachability request requires one exact controlled member")
        require_sorted_unique_ids(
            self.output_bounds,
            attribute="constraint_id",
            field_name="output_bounds",
        )
        for name, threshold, minimum in (
            ("rank_tolerance", self.rank_tolerance, Decimal(0)),
            ("minimum_output_norm", self.minimum_output_norm, Decimal(0)),
            ("maximum_condition_number", self.maximum_condition_number, Decimal(1)),
        ):
            validate_decimal(threshold, field_name=name, minimum=minimum)
        if self.rank_tolerance == 0:
            raise ValueError("controlled reachability rank tolerance must be positive")
        if self.reference_kind is ReceiptAdmissionReachabilityReferenceKind.ACTIVE_MINUS_QUALIFIED_HOLD:
            if (
                self.qualified_hold_action_fibre is None
                or self.qualified_hold_action_fibre.object_schema != ReceiptAdmissionActionFibre.SCHEMA
                or self.qualified_hold_law_evaluation_binding is None
                or self.hold_projection is None
                or self.declared_native_reference_direction_id is not None
                or self.declared_native_reference is not None
            ):
                raise ValueError("active-minus-HOLD request requires only an exact HOLD projection")
        elif (
            self.qualified_hold_action_fibre is not None
            or self.qualified_hold_law_evaluation_binding is not None
            or self.hold_projection is not None
            or self.declared_native_reference_direction_id is None
            or self.declared_native_reference is None
        ):
            raise ValueError("native-reference request requires only its declared vector")
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        if (
            not self.method.deterministic
            or self.method.capability_key != "geometry.controlled-io-admission-reachability"
            or self.method.capability_version != "1.0.0"
            or self.method.input_schema != self.SCHEMA
            or self.method.output_schema != AdmissionReachabilityRawInput.SCHEMA
        ):
            raise ValueError(
                "controlled reachability method is not the registered deterministic method"
            )


class _ProjectionUnavailable(ValueError):
    """Internal typed boundary for an unavailable declared clock projection."""


@dataclass(frozen=True, slots=True)
class ControlledInputOutputReachabilityAdmissionMethod:
    """Compute G=C Phi B and map one exact active-minus-HOLD/native direction."""

    capability_key = "geometry.controlled-io-admission-reachability"
    capability_version = "1.0.0"

    def evaluate(
        self,
        *,
        plan: ReceiptAdmissionReceiptProductionPlan,
        member: ControlledIOMember,
        request: ControlledIOReachabilityRequest,
    ) -> AdmissionReachabilityRawInput:
        coordinate = next(
            (
                value
                for value in plan.coordinates
                if value.coordinate_id == request.planned_coordinate_id
            ),
            None,
        )
        if coordinate is None:
            raise ValueError("controlled reachability request uses an unplanned coordinate")
        if (
            coordinate.denominator_member_id != member.denominator_member_id
            or coordinate.candidate_version_id != member.candidate_version_id
            or coordinate.qualification_view_ids != member.qualification_view_ids
        ):
            raise ValueError("controlled reachability member/version/refinement axes differ")
        member_identity = ObjectIdentity.from_record(member.member_record_id, member)
        if request.controlled_io_member != member_identity:
            raise ValueError("controlled reachability request binds another controlled member")
        validate_law_evaluation_binding(coordinate, request.law_evaluation_binding)
        action = plan.action_fibre(coordinate.action_fibre)
        support = plan.support_cell(coordinate.support_cell)
        law = next(
            value
            for value in plan.atlas.laws
            if ObjectIdentity.from_record(value.law_id, value) == coordinate.response_law
        )
        active_identity = ObjectIdentity.from_record(action.action_word.word_id, action.action_word)
        member_action_identities = {
            ObjectIdentity.from_record(value.word_id, value) for value in member.action_words
        }
        if request.active_projection.action_word != active_identity:
            raise ValueError("controlled reachability active projection uses another ActionWord")
        if request.law_evaluation_binding.action_word != active_identity:
            raise ValueError("controlled reachability law evaluation uses another ActionWord")
        if active_identity not in member_action_identities:
            raise ValueError("controlled reachability ActionWord is absent from the member roster")
        if member.horizon_id != plan.horizon.horizon_id:
            raise ValueError("controlled reachability horizon differs from the member")
        if (
            member.retained_history_id != action.action_word.retained_history_id
            or member.input_basis.coordinate_ids != law.relation.action_quantity_ids
            or member.receiver_basis.coordinate_ids != law.relation.receiver_quantity_ids
        ):
            raise ValueError(
                "controlled reachability changes the law history/action/receiver basis"
            )
        if {value.constraint_id for value in request.output_bounds} != set(
            plan.reachability_constraint_ids
        ):
            raise ValueError("controlled reachability constraints differ from the admission plan")
        self._validate_evidence(member, request)
        if admission_coordinate_outside_support(plan, coordinate):
            return self._non_evaluated(plan, request, ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT)
        if (
            member.disposition is not ControlledIOProductDisposition.SUPPORTED
            or support.cell_id not in member.support_cell_ids
        ):
            return self._non_evaluated(plan, request, ReceiptAdmissionRawDisposition.UNEVALUABLE)
        family = ControlledIOEvaluator().evaluate(member, member.qualification_config)
        matrix = family.finite_horizon_map
        if matrix is None:  # pragma: no cover - supported-member evaluator invariant
            raise AssertionError("supported controlled member lost its finite-horizon map")
        try:
            active = self._project_word(
                member,
                action.action_word,
                request.active_projection,
                matrix,
            )
            if request.reference_kind is ReceiptAdmissionReachabilityReferenceKind.ACTIVE_MINUS_QUALIFIED_HOLD:
                hold_identity = request.qualified_hold_action_fibre
                if hold_identity is None:  # pragma: no cover - request invariant
                    raise AssertionError("active-minus-HOLD request lost its fibre")
                hold_fibre = plan.action_fibre(hold_identity)
                if hold_fibre.action_binding_id == action.action_binding_id:
                    raise ValueError("controlled reachability HOLD cannot be the active fibre")
                hold_projection = request.hold_projection
                if hold_projection is None:  # pragma: no cover - request invariant
                    raise AssertionError("active-minus-HOLD request lost its projection")
                hold_word_identity = ObjectIdentity.from_record(
                    hold_fibre.action_word.word_id,
                    hold_fibre.action_word,
                )
                hold_coordinate = next(
                    (
                        value
                        for value in plan.coordinates
                        if value.denominator_member_id == coordinate.denominator_member_id
                        and value.candidate_version_id == coordinate.candidate_version_id
                        and value.qualification_view_ids == coordinate.qualification_view_ids
                        and value.support_cell == coordinate.support_cell
                        and value.action_fibre == hold_identity
                    ),
                    None,
                )
                hold_binding = request.qualified_hold_law_evaluation_binding
                if hold_coordinate is None or hold_binding is None:
                    raise ValueError("controlled reachability HOLD coordinate is absent")
                validate_law_evaluation_binding(hold_coordinate, hold_binding)
                if (
                    hold_projection.action_word != hold_word_identity
                    or hold_binding.action_word != hold_word_identity
                    or hold_binding.evaluation_disposition
                    is not LawEvaluationBindingDisposition.SUPPORTED
                    or hold_word_identity not in member_action_identities
                    or hold_fibre.action_word.denominator_id != action.action_word.denominator_id
                    or hold_fibre.action_word.retained_history_id != member.retained_history_id
                ):
                    raise ValueError(
                        "controlled reachability HOLD is not exact and member-qualified"
                    )
                reference = self._project_word(
                    member,
                    hold_fibre.action_word,
                    hold_projection,
                    matrix,
                )
                direction = active - reference
                declared_direction_id = None
            else:
                declared = request.declared_native_reference
                if declared is None:  # pragma: no cover - request invariant
                    raise AssertionError("native-reference request lost its vector")
                if declared.coordinate_ids != matrix.column_coordinate_ids:
                    raise ValueError(
                        "declared native direction uses another finite-map input basis"
                    )
                direction = declared.as_array()
                declared_direction_id = request.declared_native_reference_direction_id
        except _ProjectionUnavailable:
            return self._unavailable_projection(plan, request)
        controlled = matrix.as_array()
        image = controlled @ direction
        singular_values = np.linalg.svd(controlled, compute_uv=False)
        rank = int(np.sum(singular_values > float(request.rank_tolerance)))
        condition = (
            decimal_from_float(float(singular_values[0] / singular_values[rank - 1]))
            if rank > 0
            else request.maximum_condition_number + Decimal(1)
        )
        viable = np.linalg.norm(image) >= float(
            request.minimum_output_norm
        ) and self._inside_output_bounds(member, matrix, image, request.output_bounds)
        return AdmissionReachabilityRawInput(
            input_id=request.request_id,
            coordinate_id=coordinate.coordinate_id,
            admission_direction_gate_receipt_id=(
                f"receipt.admission-gate.{plan.plan_id}.{coordinate.coordinate_id}.reachability"
            ),
            law_evaluation_binding=request.law_evaluation_binding,
            reference_kind=request.reference_kind,
            reachability_request=ObjectIdentity.from_record(request.request_id, request),
            qualified_hold_action_fibre=request.qualified_hold_action_fibre,
            qualified_hold_law_evaluation_binding=(request.qualified_hold_law_evaluation_binding),
            declared_native_reference_direction_id=declared_direction_id,
            controlled_map_evaluation=ObjectIdentity.from_record(family.family_id, family),
            candidate_direction_ids=(request.direction_id,),
            viable_direction_ids=((request.direction_id,) if viable else ()),
            independent_basis_direction_ids=((request.direction_id,) if viable else ()),
            controlled_map_rank=rank,
            condition_number=NamedDecimal(
                value_id=f"condition.{request.request_id}",
                value=condition,
                unit="1",
            ),
            maximum_condition_number=NamedDecimal(
                value_id=f"maximum-condition.{request.request_id}",
                value=request.maximum_condition_number,
                unit="1",
            ),
            disposition=ReceiptAdmissionRawDisposition.EVALUATED,
            method=request.method,
            input_artifacts=request.input_artifacts,
            evidence_links=request.evidence_links,
        )

    @staticmethod
    def _validate_evidence(
        member: ControlledIOMember,
        request: ControlledIOReachabilityRequest,
    ) -> None:
        if {value.link_id for value in request.evidence_links} != set(
            member.diagnostics.evidence_link_ids
        ):
            raise ValueError("controlled reachability evidence differs from member diagnostics")
        artifacts = {value.artifact_id for value in request.input_artifacts}
        linked = {value for link in request.evidence_links for value in link.artifact_ids}
        if artifacts != linked or request.method.payload.artifact_id not in artifacts:
            raise ValueError("controlled reachability artifacts differ from exact evidence")

    @staticmethod
    def _non_evaluated(
        plan: ReceiptAdmissionReceiptProductionPlan,
        request: ControlledIOReachabilityRequest,
        disposition: ReceiptAdmissionRawDisposition,
    ) -> AdmissionReachabilityRawInput:
        if disposition not in {ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT, ReceiptAdmissionRawDisposition.UNEVALUABLE}:
            raise ValueError("controlled reachability non-evaluation disposition is unsupported")
        return AdmissionReachabilityRawInput(
            input_id=request.request_id,
            coordinate_id=request.planned_coordinate_id,
            admission_direction_gate_receipt_id=(
                f"receipt.admission-gate.{plan.plan_id}.{request.planned_coordinate_id}.reachability"
            ),
            law_evaluation_binding=request.law_evaluation_binding,
            reference_kind=request.reference_kind,
            reachability_request=ObjectIdentity.from_record(request.request_id, request),
            qualified_hold_action_fibre=request.qualified_hold_action_fibre,
            qualified_hold_law_evaluation_binding=(request.qualified_hold_law_evaluation_binding),
            declared_native_reference_direction_id=(request.declared_native_reference_direction_id),
            controlled_map_evaluation=None,
            candidate_direction_ids=(),
            viable_direction_ids=(),
            independent_basis_direction_ids=(),
            controlled_map_rank=None,
            condition_number=None,
            maximum_condition_number=NamedDecimal(
                value_id=f"maximum-condition.{request.request_id}",
                value=request.maximum_condition_number,
                unit="1",
            ),
            disposition=disposition,
            method=request.method,
            input_artifacts=request.input_artifacts,
            evidence_links=request.evidence_links,
        )

    @staticmethod
    def _unavailable_projection(
        plan: ReceiptAdmissionReceiptProductionPlan,
        request: ControlledIOReachabilityRequest,
    ) -> AdmissionReachabilityRawInput:
        return AdmissionReachabilityRawInput(
            input_id=request.request_id,
            coordinate_id=request.planned_coordinate_id,
            admission_direction_gate_receipt_id=(
                f"receipt.admission-gate.{plan.plan_id}.{request.planned_coordinate_id}.reachability"
            ),
            law_evaluation_binding=request.law_evaluation_binding,
            reference_kind=request.reference_kind,
            reachability_request=ObjectIdentity.from_record(request.request_id, request),
            qualified_hold_action_fibre=request.qualified_hold_action_fibre,
            qualified_hold_law_evaluation_binding=(request.qualified_hold_law_evaluation_binding),
            declared_native_reference_direction_id=(request.declared_native_reference_direction_id),
            controlled_map_evaluation=None,
            candidate_direction_ids=(),
            viable_direction_ids=(),
            independent_basis_direction_ids=(),
            controlled_map_rank=None,
            condition_number=None,
            maximum_condition_number=NamedDecimal(
                value_id=f"maximum-condition.{request.request_id}",
                value=request.maximum_condition_number,
                unit="1",
            ),
            disposition=ReceiptAdmissionRawDisposition.UNEVALUABLE,
            method=request.method,
            input_artifacts=request.input_artifacts,
            evidence_links=request.evidence_links,
        )

    @staticmethod
    def _project_word(
        member: ControlledIOMember,
        word: OccurrenceActionWord,
        projection: ActionWordInputProjectionSpec,
        finite_map: CanonicalMatrix,
    ) -> npt.NDArray[np.float64]:
        bindings = {value.occurrence_id: value for value in projection.occurrence_bindings}
        occurrences = {value.occurrence_id: value for value in word.occurrences}
        if set(bindings) != set(occurrences):
            raise ValueError("controlled input projection omits or adds ActionWord occurrences")
        grouped_steps = tuple(
            tuple(bindings[value.occurrence_id].input_step_index for value in group.members)
            for group in word.groups
        )
        if any(len(set(steps)) != 1 for steps in grouped_steps) or any(
            later[0] <= earlier[0]
            for earlier, later in zip(grouped_steps, grouped_steps[1:], strict=False)
        ):
            raise ValueError("controlled input projection collapses or reorders ActionWord groups")
        values = np.zeros(len(finite_map.column_coordinate_ids), dtype=np.float64)
        columns = {value: index for index, value in enumerate(finite_map.column_coordinate_ids)}
        semantics_by_input = dict(
            zip(
                member.input_basis.coordinate_ids,
                zip(
                    member.input_basis.native_units,
                    member.input_basis.native_frames,
                    strict=True,
                ),
                strict=True,
            )
        )
        for occurrence_id, binding in bindings.items():
            occurrence = occurrences[occurrence_id]
            if binding.input_step_index >= len(member.steps) - 1:
                raise ValueError("controlled input projection step lies outside the input horizon")
            if binding.input_coordinate_id not in semantics_by_input:
                raise ValueError("controlled input projection uses another input basis")
            native_unit, native_frame = semantics_by_input[binding.input_coordinate_id]
            if (
                occurrence.channel.controller_quantity_id != binding.input_coordinate_id
                or occurrence.realized.native_unit != native_unit
                or occurrence.realized.native_action_frame != native_frame
            ):
                raise ValueError("controlled input projection changes action quantity/unit/frame")
            if (
                binding.input_coordinate.clock_id
                != member.steps[binding.input_step_index].input_clock_id
            ):
                raise ValueError("controlled input projection changes the realized-input clock")
            clock_projection = binding.realized_to_input_clock.project(
                occurrence.realized.coordinate
            )
            if clock_projection.target is None:
                raise _ProjectionUnavailable("controlled input clock projection is unavailable")
            if not clock_projection.matches(binding.input_coordinate):
                raise ValueError("controlled input projection changes the realized-input clock")
            column_id = f"input-step-{binding.input_step_index:04d}.{binding.input_coordinate_id}"
            values[columns[column_id]] += float(occurrence.realized.value)
        return values

    @staticmethod
    def _inside_output_bounds(
        member: ControlledIOMember,
        finite_map: CanonicalMatrix,
        image: npt.NDArray[np.float64],
        bounds: tuple[ControlledOutputDirectionBound, ...],
    ) -> bool:
        rows = {value: index for index, value in enumerate(finite_map.row_coordinate_ids)}
        receiver_semantics = dict(
            zip(
                member.receiver_basis.coordinate_ids,
                zip(
                    member.receiver_basis.native_units,
                    member.receiver_basis.native_frames,
                    strict=True,
                ),
                strict=True,
            )
        )
        for bound in bounds:
            if bound.output_coordinate_id not in rows:
                raise ValueError("controlled output bound uses another finite-map row")
            receiver_id = bound.output_coordinate_id.split(".", 1)[1]
            if receiver_semantics.get(receiver_id) != (
                bound.native_unit,
                bound.native_frame,
            ):
                raise ValueError("controlled output bound changes receiver native semantics")
            value = image[rows[bound.output_coordinate_id]]
            if bound.lower is not None and value < float(bound.lower):
                return False
            if bound.upper is not None and value > float(bound.upper):
                return False
        return True


__all__ = [
    "ActionOccurrenceInputBinding",
    "ActionWordInputProjectionSpec",
    "ControlledInputOutputReachabilityAdmissionMethod",
    "ControlledIOReachabilityRequest",
    "ControlledOutputDirectionBound",
]
