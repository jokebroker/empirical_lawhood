"""Registered, support-limited evaluation of qualified response-law payloads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Protocol

import numpy as np

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ActionWordSupportStatus
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.laws import LawRepresentationKind, ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import HorizonSpec
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt

from .contracts import (
    FiniteActionCellDisposition,
    FiniteActionCompatibilitySetExtension,
    FiniteActionNativeValue,
    LawCandidateAxisMap,
    ParametricLawModel,
)
from .law_assessment import CandidatePayloadReadError, CandidatePayloadReader
from .receiver_conditioned_io.contracts import (
    ControlledIOProductDisposition,
    ControlledIOVersionSet,
)
from .receiver_conditioned_io.controlled_io import ControlledIOEvaluator


class LawEvaluationKind(StrEnum):
    POINT_NAMED_VALUES = "POINT_NAMED_VALUES"
    FINITE_ACTION_PATH = "FINITE_ACTION_PATH"
    CONTROLLED_IO_VERSION_SET = "CONTROLLED_IO_VERSION_SET"


class LawEvaluationDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    MEMBER_ABSENT = "MEMBER_ABSENT"
    OPERAND_ABSENT = "OPERAND_ABSENT"
    PRODUCT_NOT_CLAIMED = "PRODUCT_NOT_CLAIMED"
    RESOURCE_REFUSED = "RESOURCE_REFUSED"


class LawEvaluationMode(StrEnum):
    OFFLINE = "OFFLINE"
    ONLINE = "ONLINE"


@dataclass(frozen=True, slots=True)
class LawEvaluationValue(CanonicalRecord):
    """One native-unit/frame/clock value at the executable law boundary."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-evaluation-value'

    value_id: str
    quantity_id: str
    value: Decimal
    native_unit: str
    native_frame_id: str
    clock_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.value_id, field_name="value_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_stable_id(self.native_frame_id, field_name="native_frame_id")
        validate_stable_id(self.clock_id, field_name="clock_id")
        if not self.native_unit.strip() or not self.value.is_finite():
            raise ValueError("law evaluation value requires finite native-unit data")


@dataclass(frozen=True, slots=True)
class LawEvaluationRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-evaluation-request'

    request_id: str
    law: ObjectIdentity
    qualification_result: ObjectIdentity
    axis_map: ObjectIdentity
    evaluator: ObjectIdentity
    payload_publication: ObjectIdentity
    evaluation_kind: LawEvaluationKind
    evaluation_mode: LawEvaluationMode
    resource_envelope_id: str
    randomness_seed_id: str | None
    system_id: str
    world_id: str
    relation_id: str
    chart_id: str
    prepared_denominator_id: str
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_id: str
    history_quantity_ids: tuple[str, ...]
    receiver_quantity_ids: tuple[str, ...]
    horizon: ObjectIdentity
    support_cell_id: str
    action_word: OccurrenceActionWord | None
    input_values: tuple[LawEvaluationValue, ...]
    input_artifacts: tuple[ArtifactIdentity, ...]
    requested_product_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("request_id", self.request_id),
            ("system_id", self.system_id),
            ("world_id", self.world_id),
            ("relation_id", self.relation_id),
            ("chart_id", self.chart_id),
            ("prepared_denominator_id", self.prepared_denominator_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("qualification_view_id", self.qualification_view_id),
            ("support_cell_id", self.support_cell_id),
            ("resource_envelope_id", self.resource_envelope_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.law.object_schema != ResponseLaw.SCHEMA:
            raise ValueError("law evaluation request requires a ResponseLaw")
        if self.qualification_result.object_schema != LawQualificationResult.SCHEMA:
            raise ValueError("law evaluation request requires authoritative qualification")
        if self.axis_map.object_schema != LawCandidateAxisMap.SCHEMA:
            raise ValueError("law evaluation request requires the exact three-axis map")
        if self.randomness_seed_id is not None:
            validate_stable_id(self.randomness_seed_id, field_name="randomness_seed_id")
        if self.horizon.object_schema != HorizonSpec.SCHEMA:
            raise ValueError("law evaluation request requires the exact response horizon")
        require_sorted_unique_strings(
            self.history_quantity_ids,
            field_name="history_quantity_ids",
        )
        require_sorted_unique_strings(
            self.receiver_quantity_ids,
            field_name="receiver_quantity_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.input_values,
            attribute="value_id",
            field_name="input_values",
        )
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_strings(
            self.requested_product_ids,
            field_name="requested_product_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class LawEvaluationResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-evaluation-result'

    result_id: str
    request: ObjectIdentity
    evaluator_implementation: ObjectIdentity
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_id: str
    action_word: ObjectIdentity | None
    disposition: LawEvaluationDisposition
    response_values: tuple[LawEvaluationValue, ...]
    sink_values: tuple[LawEvaluationValue, ...]
    effort_values: tuple[LawEvaluationValue, ...]
    uncertainty_values: tuple[LawEvaluationValue, ...]
    path_artifact_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("result_id", self.result_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("qualification_view_id", self.qualification_view_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("response_values", self.response_values),
            ("sink_values", self.sink_values),
            ("effort_values", self.effort_values),
            ("uncertainty_values", self.uncertainty_values),
        ):
            require_sorted_unique_ids(values, attribute="value_id", field_name=name)
        require_sorted_unique_strings(
            self.path_artifact_ids,
            field_name="path_artifact_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is LawEvaluationDisposition.SUPPORTED:
            if not self.response_values or self.reason_codes:
                raise ValueError("supported law evaluation requires response values")
        else:
            if any(
                (
                    self.response_values,
                    self.sink_values,
                    self.effort_values,
                    self.uncertainty_values,
                    self.path_artifact_ids,
                )
            ):
                raise ValueError("refused law evaluation cannot return scientific values")
            if not self.reason_codes:
                raise ValueError("refused law evaluation requires reasons")


@dataclass(frozen=True, slots=True)
class LawEvaluatorRegistration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/law-evaluator-registration'

    registration_id: str
    evaluator_key: str
    evaluation_kind: LawEvaluationKind
    representation_kind: LawRepresentationKind
    payload_schema: str
    extension_namespace: str | None
    decoder_schema: str
    decoder_version: str
    request_schema: str
    result_schema: str
    implementation: ObjectIdentity
    deterministic: bool
    offline_resource_envelope_id: str
    online_resource_envelope_id: str
    maximum_payload_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("registration_id", self.registration_id),
            ("evaluator_key", self.evaluator_key),
            ("offline_resource_envelope_id", self.offline_resource_envelope_id),
            ("online_resource_envelope_id", self.online_resource_envelope_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_schema(self.payload_schema)
        if self.extension_namespace is not None:
            validate_stable_id(
                self.extension_namespace,
                field_name="extension_namespace",
            )
        validate_schema(self.decoder_schema)
        validate_semantic_version(self.decoder_version)
        validate_schema(self.request_schema)
        validate_schema(self.result_schema)
        if self.request_schema != LawEvaluationRequest.SCHEMA:
            raise ValueError("law evaluator registration requires the current request schema")
        if self.result_schema != LawEvaluationResult.SCHEMA:
            raise ValueError("law evaluator registration requires the current result schema")
        if self.maximum_payload_bytes <= 0:
            raise ValueError("law evaluator payload bound must be positive")


class LawEvaluatorImplementation(Protocol):
    registration: LawEvaluatorRegistration

    def decode(
        self,
        payload: bytes,
        *,
        maximum_bytes: int,
    ) -> CanonicalRecord: ...

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult: ...


@dataclass(frozen=True, slots=True)
class LawEvaluatorRegistry:
    implementations: tuple[LawEvaluatorImplementation, ...]

    def __post_init__(self) -> None:
        keys = tuple(value.registration.evaluator_key for value in self.implementations)
        if tuple(sorted(set(keys))) != keys:
            raise ValueError("law evaluator registry must be sorted and unique")

    def require(self, law: ResponseLaw) -> LawEvaluatorImplementation:
        for implementation in self.implementations:
            registration = implementation.registration
            if registration.evaluator_key == law.evaluator.evaluator_key:
                if registration.representation_kind is not law.representation_kind:
                    raise ValueError("law evaluator registration has another representation")
                if registration.payload_schema != law.evaluator.payload.payload_schema:
                    raise ValueError("law evaluator registration has another payload schema")
                return implementation
        raise ValueError("law evaluator is not registered")


def _refusal(
    request: LawEvaluationRequest,
    implementation: ObjectIdentity,
    disposition: LawEvaluationDisposition,
    reason: str,
) -> LawEvaluationResult:
    return LawEvaluationResult(
        result_id=f"evaluation.{request.request_id}",
        request=ObjectIdentity.from_record(request.request_id, request),
        evaluator_implementation=implementation,
        denominator_member_id=request.denominator_member_id,
        candidate_version_id=request.candidate_version_id,
        qualification_view_id=request.qualification_view_id,
        action_word=(
            None
            if request.action_word is None
            else ObjectIdentity.from_record(request.action_word.word_id, request.action_word)
        ),
        disposition=disposition,
        response_values=(),
        sink_values=(),
        effort_values=(),
        uncertainty_values=(),
        path_artifact_ids=(),
        reason_codes=(reason,),
    )


def _action_identity(action_word: OccurrenceActionWord | None) -> ObjectIdentity | None:
    if action_word is None:
        return None
    return ObjectIdentity.from_record(action_word.word_id, action_word)


@dataclass(frozen=True, slots=True)
class PointParametricLawEvaluator:
    registration: LawEvaluatorRegistration

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload,
            ParametricLawModel,
            maximum_bytes=maximum_bytes,
        )

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, ParametricLawModel):
            raise TypeError("point evaluator received another payload type")
        if request.requested_product_ids != ("response-values",):
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                "point-law-product-not-claimed",
            )
        if request.qualification_view_id != payload.numerical_view_id:
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.MEMBER_ABSENT,
                "qualification-view-absent",
            )
        if payload.receiver_quantity_ids != law.relation.receiver_quantity_ids:
            raise ValueError("parametric payload changes the law receiver block")
        if not set(payload.feature_quantity_ids).issubset(law.interface_input_quantity_ids):
            raise ValueError("parametric payload uses inputs outside the qualified law")
        inputs = {value.quantity_id: value.value for value in request.input_values}
        if len(inputs) != len(request.input_values):
            raise ValueError("point evaluation requires one value per input quantity")
        if not set(payload.feature_quantity_ids).issubset(inputs):
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.OPERAND_ABSENT,
                "required-input-absent",
            )
        quantities = {value.quantity_id: value for value in system.quantities}
        outputs: list[LawEvaluationValue] = []
        coefficients = {
            (value.receiver_quantity_id, value.term_id): value for value in payload.coefficients
        }
        for receiver_id in payload.receiver_quantity_ids:
            receiver_quantity = quantities[receiver_id]
            if coefficients[(receiver_id, "intercept")].native_unit != (
                receiver_quantity.native_unit
            ):
                raise ValueError("parametric payload changes a receiver native unit")
            total = coefficients[(receiver_id, "intercept")].value
            for feature_id in payload.feature_quantity_ids:
                total += (
                    coefficients[(receiver_id, f"linear.{feature_id}")].value * inputs[feature_id]
                )
                quadratic = coefficients.get((receiver_id, f"quadratic.{feature_id}"))
                if quadratic is not None:
                    total += quadratic.value * inputs[feature_id] * inputs[feature_id]
            outputs.append(
                LawEvaluationValue(
                    value_id=receiver_id,
                    quantity_id=receiver_id,
                    value=total,
                    native_unit=receiver_quantity.native_unit,
                    native_frame_id=receiver_quantity.coordinate_frame,
                    clock_id=receiver_quantity.clock_id,
                )
            )
        return LawEvaluationResult(
            result_id=f"evaluation.{request.request_id}",
            request=ObjectIdentity.from_record(request.request_id, request),
            evaluator_implementation=self.registration.implementation,
            denominator_member_id=request.denominator_member_id,
            candidate_version_id=request.candidate_version_id,
            qualification_view_id=request.qualification_view_id,
            action_word=_action_identity(request.action_word),
            disposition=LawEvaluationDisposition.SUPPORTED,
            response_values=tuple(sorted(outputs, key=lambda value: value.value_id)),
            sink_values=(),
            effort_values=(),
            uncertainty_values=(),
            path_artifact_ids=(),
            reason_codes=(),
        )


@dataclass(frozen=True, slots=True)
class FiniteActionLawEvaluator:
    registration: LawEvaluatorRegistration

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload,
            FiniteActionCompatibilitySetExtension,
            maximum_bytes=maximum_bytes,
        )

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, FiniteActionCompatibilitySetExtension):
            raise TypeError("finite-action evaluator received another payload type")
        if request.action_word is None:
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.OPERAND_ABSENT,
                "action-word-absent",
            )
        if request.input_values:
            raise ValueError("finite-action lookup cannot accept free input values")
        action_identity = ObjectIdentity.from_record(
            request.action_word.word_id,
            request.action_word,
        )
        if (
            request.action_word.horizon_id != payload.horizon.object_id
            or request.horizon != payload.horizon
            or request.action_word.retained_history_id not in payload.retained_history_ids
            or request.prepared_denominator_id != payload.prepared_denominator_id
        ):
            raise ValueError(
                "finite-action word changes prepared denominator, retained history or horizon"
            )
        payload_action_identities = {
            ObjectIdentity.from_record(value.word_id, value) for value in payload.action_words
        }
        if action_identity not in payload_action_identities:
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.OUTSIDE_SUPPORT,
                "finite-action-word-outside-roster",
            )
        matches = tuple(
            value
            for value in payload.entries
            if value.denominator_member_id == request.denominator_member_id
            and value.candidate_version_id == request.candidate_version_id
            and request.qualification_view_id in value.qualification_view_ids
            and value.support_cell_id == request.support_cell_id
            and value.action_word == action_identity
        )
        if not matches:
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.OUTSIDE_SUPPORT,
                "finite-action-cell-outside-support",
            )
        if len(matches) != 1:
            raise ValueError("finite-action payload has duplicate exact cells")
        entry = matches[0]
        if entry.disposition is not FiniteActionCellDisposition.SUPPORTED:
            disposition = (
                LawEvaluationDisposition.OUTSIDE_SUPPORT
                if entry.disposition is FiniteActionCellDisposition.OUTSIDE_SUPPORT
                else LawEvaluationDisposition.PRODUCT_NOT_CLAIMED
            )
            return _refusal(
                request,
                self.registration.implementation,
                disposition,
                entry.reason_codes[0],
            )
        available = {"response-values"}
        if entry.sink_values:
            available.add("sink-values")
        if entry.effort_values:
            available.add("effort-values")
        if entry.uncertainty_values:
            available.add("uncertainty-values")
        if not set(request.requested_product_ids).issubset(available):
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                "finite-action-product-not-claimed",
            )
        if "response-values" not in request.requested_product_ids:
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                "finite-action-response-product-required",
            )
        quantities = {value.quantity_id: value for value in system.quantities}

        def bound(values: tuple[FiniteActionNativeValue, ...]) -> tuple[LawEvaluationValue, ...]:
            result = []
            for value in values:
                try:
                    quantity = quantities[value.quantity_id]
                except KeyError as error:
                    raise ValueError("finite-action payload names an unknown quantity") from error
                if (
                    value.native_unit != quantity.native_unit
                    or value.native_frame_id != quantity.coordinate_frame
                    or value.clock_id != quantity.clock_id
                ):
                    raise ValueError("finite-action payload changes native unit/frame/clock")
                if value.quantity_id not in law.interface_output_quantity_ids:
                    raise ValueError("finite-action payload exceeds qualified law outputs")
                result.append(
                    LawEvaluationValue(
                        value_id=value.value_id,
                        quantity_id=value.quantity_id,
                        value=value.value,
                        native_unit=value.native_unit,
                        native_frame_id=value.native_frame_id,
                        clock_id=value.clock_id,
                    )
                )
            return tuple(result)

        return LawEvaluationResult(
            result_id=f"evaluation.{request.request_id}",
            request=ObjectIdentity.from_record(request.request_id, request),
            evaluator_implementation=self.registration.implementation,
            denominator_member_id=request.denominator_member_id,
            candidate_version_id=request.candidate_version_id,
            qualification_view_id=request.qualification_view_id,
            action_word=action_identity,
            disposition=LawEvaluationDisposition.SUPPORTED,
            response_values=bound(entry.response_values),
            sink_values=(
                bound(entry.sink_values) if "sink-values" in request.requested_product_ids else ()
            ),
            effort_values=(
                bound(entry.effort_values)
                if "effort-values" in request.requested_product_ids
                else ()
            ),
            uncertainty_values=(
                bound(entry.uncertainty_values)
                if "uncertainty-values" in request.requested_product_ids
                else ()
            ),
            path_artifact_ids=(),
            reason_codes=(),
        )


@dataclass(frozen=True, slots=True)
class ControlledIOVersionSetLawEvaluator:
    """Exact finite-horizon evaluation of one qualified controlled-IO member."""

    registration: LawEvaluatorRegistration

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload,
            ControlledIOVersionSet,
            maximum_bytes=maximum_bytes,
        )

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        request: LawEvaluationRequest,
        payload: CanonicalRecord,
    ) -> LawEvaluationResult:
        if not isinstance(payload, ControlledIOVersionSet):
            raise TypeError("controlled-IO evaluator received another payload type")
        if request.requested_product_ids != ("response-path",):
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                "controlled-io-product-not-claimed",
            )
        if request.action_word is None:
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.OPERAND_ABSENT,
                "controlled-io-action-word-absent",
            )
        action_identity = ObjectIdentity.from_record(
            request.action_word.word_id,
            request.action_word,
        )
        matches = tuple(
            value
            for value in payload.members
            if value.denominator_member_id == request.denominator_member_id
            and value.candidate_version_id == request.candidate_version_id
            and request.qualification_view_id in value.qualification_view_ids
        )
        if not matches:
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.MEMBER_ABSENT,
                "controlled-io-member-absent",
            )
        if len(matches) != 1:
            raise ValueError("controlled-IO payload duplicates a member/version coordinate")
        member = matches[0]
        if member.disposition is not ControlledIOProductDisposition.SUPPORTED:
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                member.reason_codes[0],
            )
        if request.support_cell_id not in member.support_cell_ids:
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.OUTSIDE_SUPPORT,
                "controlled-io-cell-outside-support",
            )
        member_action_identities = {
            ObjectIdentity.from_record(value.word_id, value) for value in member.action_words
        }
        if action_identity not in member_action_identities:
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.OUTSIDE_SUPPORT,
                "controlled-io-action-word-outside-support",
            )
        if (
            request.action_word.denominator_id != member.prepared_denominator_id
            or request.action_word.retained_history_id != member.retained_history_id
            or request.action_word.horizon_id != member.horizon_id
            or request.prepared_denominator_id != member.prepared_denominator_id
        ):
            raise ValueError("controlled-IO action changes prepared denominator/history/horizon")
        if member.horizon_id != request.horizon.object_id:
            raise ValueError("controlled-IO member changes the qualified horizon")
        markov = ControlledIOEvaluator().evaluate(member, member.qualification_config)
        if markov.finite_horizon_map is None:
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.PRODUCT_NOT_CLAIMED,
                "controlled-io-map-unavailable",
            )
        matrix = markov.finite_horizon_map
        inputs = {value.value_id: value for value in request.input_values}
        if set(inputs) != set(matrix.column_coordinate_ids):
            return _refusal(
                request,
                self.registration.implementation,
                LawEvaluationDisposition.OPERAND_ABSENT,
                "controlled-io-input-path-incomplete",
            )
        horizon = len(member.steps) - 1
        if len(request.action_word.groups) != horizon:
            raise ValueError("controlled-IO action word changes the finite input horizon")
        occurrences = {value.occurrence_id: value for value in request.action_word.occurrences}
        expected: dict[str, LawEvaluationValue] = {}
        for step_index, group in enumerate(request.action_word.groups):
            step_occurrences = tuple(occurrences[value.occurrence_id] for value in group.members)
            by_quantity = {
                value.channel.controller_quantity_id: value for value in step_occurrences
            }
            if len(by_quantity) != len(step_occurrences) or set(by_quantity) != set(
                member.input_basis.coordinate_ids
            ):
                raise ValueError("controlled-IO action group does not cover the exact input basis")
            for basis_index, quantity_id in enumerate(member.input_basis.coordinate_ids):
                occurrence = by_quantity[quantity_id]
                realized = occurrence.realized
                if (
                    realized.native_unit != member.input_basis.native_units[basis_index]
                    or realized.native_action_frame != member.input_basis.native_frames[basis_index]
                    or realized.coordinate.clock_id != member.steps[step_index].input_clock_id
                ):
                    raise ValueError("controlled-IO action changes native input unit/frame/clock")
                value_id = f"input-step-{step_index:04d}.{quantity_id}"
                expected[value_id] = LawEvaluationValue(
                    value_id=value_id,
                    quantity_id=quantity_id,
                    value=realized.value,
                    native_unit=realized.native_unit,
                    native_frame_id=realized.native_action_frame,
                    clock_id=realized.coordinate.clock_id,
                )
        if tuple(inputs[value_id] for value_id in matrix.column_coordinate_ids) != tuple(
            expected[value_id] for value_id in matrix.column_coordinate_ids
        ):
            raise ValueError("controlled-IO input path differs from realized action occurrences")
        vector = np.asarray(
            [float(inputs[value_id].value) for value_id in matrix.column_coordinate_ids],
            dtype=np.float64,
        )
        response = matrix.as_array() @ vector
        quantities = {value.quantity_id: value for value in system.quantities}
        output_quantity_ids = tuple(
            coordinate
            for _step in range(1, len(member.steps))
            for coordinate in member.receiver_basis.coordinate_ids
        )
        values_list: list[LawEvaluationValue] = []
        for value_id, quantity_id, raw in zip(
            matrix.row_coordinate_ids,
            output_quantity_ids,
            response,
            strict=True,
        ):
            receiver_index = int(value_id.removeprefix("receiver-step-").split(".", 1)[0])
            basis_index = member.receiver_basis.coordinate_ids.index(quantity_id)
            quantity = quantities[quantity_id]
            if (
                quantity.native_unit != member.receiver_basis.native_units[basis_index]
                or quantity.coordinate_frame != member.receiver_basis.native_frames[basis_index]
                or quantity.clock_id != member.steps[receiver_index].receiver_clock_id
                or quantity_id not in law.interface_output_quantity_ids
            ):
                raise ValueError("controlled-IO member changes native receiver unit/frame/clock")
            values_list.append(
                LawEvaluationValue(
                    value_id=value_id,
                    quantity_id=quantity_id,
                    value=Decimal(format(float(raw), ".17g")),
                    native_unit=quantity.native_unit,
                    native_frame_id=quantity.coordinate_frame,
                    clock_id=quantity.clock_id,
                )
            )
        values = tuple(values_list)
        return LawEvaluationResult(
            result_id=f"evaluation.{request.request_id}",
            request=ObjectIdentity.from_record(request.request_id, request),
            evaluator_implementation=self.registration.implementation,
            denominator_member_id=request.denominator_member_id,
            candidate_version_id=request.candidate_version_id,
            qualification_view_id=request.qualification_view_id,
            action_word=action_identity,
            disposition=LawEvaluationDisposition.SUPPORTED,
            response_values=values,
            sink_values=(),
            effort_values=(),
            uncertainty_values=(),
            path_artifact_ids=(),
            reason_codes=(),
        )


@dataclass(frozen=True, slots=True)
class LawEvaluationService:
    payload_reader: CandidatePayloadReader
    registry: LawEvaluatorRegistry

    def evaluate(
        self,
        system: SystemSpec,
        law: ResponseLaw,
        qualification: LawQualificationResult,
        axis_map: LawCandidateAxisMap,
        request: LawEvaluationRequest,
        publication: CandidatePayloadPublicationReceipt,
    ) -> LawEvaluationResult:
        if request.law != ObjectIdentity.from_record(law.law_id, law):
            raise ValueError("law evaluation request binds another law")
        if (
            system.system_id != law.system_id
            or system.world.world_id != law.world_id
            or system.relation != law.relation
        ):
            raise ValueError("law evaluation system differs from its law")
        if request.system_id != law.system_id or request.world_id != law.world_id:
            raise ValueError("law evaluation request changes denominator/world")
        if request.relation_id != law.relation.relation_id or request.chart_id != law.chart_id:
            raise ValueError("law evaluation request changes relation/chart")
        if request.history_quantity_ids != law.relation.history_quantity_ids:
            raise ValueError("law evaluation request changes retained history")
        if request.receiver_quantity_ids != law.relation.receiver_quantity_ids:
            raise ValueError("law evaluation request changes receiver block")
        if request.horizon != ObjectIdentity.from_record(
            law.relation.horizon.horizon_id,
            law.relation.horizon,
        ):
            raise ValueError("law evaluation request changes response horizon")
        if request.action_word is not None:
            if request.action_word.horizon_id != request.horizon.object_id:
                raise ValueError("law evaluation action changes response horizon")
            if request.action_word.denominator_id != request.prepared_denominator_id:
                raise ValueError("law evaluation action changes prepared denominator")
        quantities = {value.quantity_id: value for value in system.quantities}
        for value in request.input_values:
            try:
                quantity = quantities[value.quantity_id]
            except KeyError as error:
                raise ValueError("law evaluation input names an unknown quantity") from error
            if (
                value.native_unit != quantity.native_unit
                or value.native_frame_id != quantity.coordinate_frame
                or value.clock_id != quantity.clock_id
            ):
                raise ValueError("law evaluation input changes native unit/frame/clock")
        if publication.candidate_evaluator != law.evaluator:
            raise ValueError("law evaluation publication binds another evaluator")
        implementation = self.registry.require(law)
        if publication.implementation != implementation.registration.implementation:
            raise ValueError("law evaluation implementation differs from payload publication")
        if request.evaluator != implementation.registration.implementation:
            raise ValueError("law evaluation request binds another implementation")
        registration = implementation.registration
        if request.evaluation_kind is not registration.evaluation_kind:
            raise ValueError("law evaluation request changes evaluation kind")
        expected_envelope = (
            registration.offline_resource_envelope_id
            if request.evaluation_mode is LawEvaluationMode.OFFLINE
            else registration.online_resource_envelope_id
        )
        if request.resource_envelope_id != expected_envelope:
            return _refusal(
                request,
                registration.implementation,
                LawEvaluationDisposition.RESOURCE_REFUSED,
                "law-evaluation-resource-envelope-not-registered",
            )
        if registration.deterministic and request.randomness_seed_id is not None:
            raise ValueError("deterministic law evaluation cannot accept a randomness seed")
        if not registration.deterministic and request.randomness_seed_id is None:
            return _refusal(
                request,
                registration.implementation,
                LawEvaluationDisposition.RESOURCE_REFUSED,
                "law-evaluation-randomness-seed-required",
            )
        if (
            publication.decoder_schema != registration.decoder_schema
            or publication.decoder_version != registration.decoder_version
        ):
            raise ValueError("law evaluation decoder registration differs from publication")
        if publication.maximum_decode_bytes > registration.maximum_payload_bytes:
            return _refusal(
                request,
                registration.implementation,
                LawEvaluationDisposition.RESOURCE_REFUSED,
                "law-evaluation-payload-bound-exceeds-registration",
            )
        if qualification.response_law != law or qualification.selected_candidate_id is None:
            raise ValueError("law evaluation requires its supported authoritative qualification")
        expected_extensions = (
            ()
            if registration.extension_namespace is None
            else (
                ExtensionBinding(
                    namespace=registration.extension_namespace,
                    schema=registration.payload_schema,
                    payload_sha256=publication.content_sha256,
                ),
            )
        )
        if law.extensions != expected_extensions or qualification.extensions != law.extensions:
            raise ValueError("law evaluation extension binding differs from its registration")
        if request.qualification_result != ObjectIdentity.from_record(
            qualification.result_id,
            qualification,
        ):
            raise ValueError("law evaluation request binds another qualification result")
        if request.axis_map != ObjectIdentity.from_record(axis_map.axis_map_id, axis_map):
            raise ValueError("law evaluation request binds another axis map")
        if qualification.axis_map != request.axis_map:
            raise ValueError("law evaluation axis map differs from qualification")
        if request.payload_publication != ObjectIdentity.from_record(
            publication.receipt_id,
            publication,
        ):
            raise ValueError("law evaluation request binds another payload publication")
        matching_axes = tuple(
            value
            for value in axis_map.bindings
            if value.candidate_version_member_id == request.candidate_version_id
            and value.denominator_member_id == request.denominator_member_id
            and request.qualification_view_id in value.qualification_view_ids
        )
        if not matching_axes:
            return _refusal(
                request,
                implementation.registration.implementation,
                LawEvaluationDisposition.MEMBER_ABSENT,
                "qualified-axis-coordinate-absent",
            )
        if len(matching_axes) != 1:
            raise ValueError("law evaluation coordinate is duplicated in the axis map")
        if request.support_cell_id not in law.obligations.support.denominator_cell_ids:
            return _refusal(
                request,
                implementation.registration.implementation,
                LawEvaluationDisposition.OUTSIDE_SUPPORT,
                "support-cell-outside-law-support",
            )
        if (
            request.action_word is not None
            and request.action_word.support_status is not ActionWordSupportStatus.SUPPORTED
        ):
            return _refusal(
                request,
                registration.implementation,
                LawEvaluationDisposition.OUTSIDE_SUPPORT,
                request.action_word.reason_codes[0],
            )
        input_values = {value.quantity_id: value for value in request.input_values}
        for bound in law.obligations.support.action_bounds:
            bound_value = input_values.get(bound.quantity_id)
            if bound_value is None:
                continue
            if (
                bound_value.native_unit != bound.native_unit
                or (bound.lower is not None and bound_value.value < bound.lower)
                or (bound.upper is not None and bound_value.value > bound.upper)
            ):
                return _refusal(
                    request,
                    implementation.registration.implementation,
                    LawEvaluationDisposition.OUTSIDE_SUPPORT,
                    "action-value-outside-law-support",
                )
        try:
            payload = self.payload_reader.read_candidate_payload(publication)
        except Exception as error:
            raise CandidatePayloadReadError("EVALUATOR_PAYLOAD_UNREADABLE") from error
        if len(payload) > publication.maximum_decode_bytes:
            raise CandidatePayloadReadError("EVALUATOR_PAYLOAD_EXCEEDS_BOUND")
        if hashlib.sha256(payload).hexdigest() != publication.content_sha256:
            raise CandidatePayloadReadError("EVALUATOR_PAYLOAD_DIGEST_MISMATCH")
        decoded = implementation.decode(
            payload,
            maximum_bytes=min(
                publication.maximum_decode_bytes,
                registration.maximum_payload_bytes,
            ),
        )
        if decoded.fingerprint() != publication.content_sha256:
            raise CandidatePayloadReadError("EVALUATOR_PAYLOAD_CANONICAL_IDENTITY_MISMATCH")
        return implementation.evaluate(system, law, request, decoded)
