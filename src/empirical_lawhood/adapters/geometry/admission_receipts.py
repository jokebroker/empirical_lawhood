"Registered raw admission receipt producers over one frozen Cartesian plan."

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.admission import AdmissionGateKind, GateStatus
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.adapters.methods.law_evaluation import (
    LawEvaluationRequest,
    LawEvaluationResult,
)
from empirical_lawhood.planning.evidence_geometry import BaselinePreservationCompatibility, AdmissionCoordinateGateReceipt, LawEvaluationBindingDisposition, LawMemberEvaluationBinding, ReceiptAdmissionPlannedCoordinate, ReceiptAdmissionRawDisposition, ReceiptAdmissionReachabilityReferenceKind, ControlledMapAdmissionReceiptCorpus, ReceiptAdmissionReceiptProductionPlan, ReceiptAdmissionUtilityDirection, ReceiptAdmissionUtilityStatus, ReachabilityCellDisposition, ControlledMapReachabilityReceipt, UtilityEvaluationReceipt, derive_admission_gate_observation, admission_disposition_reason
from empirical_lawhood.planning.gate_margin import GateMarginReceipt
from empirical_lawhood.runtime.gate_margin_projection import CertifiedAdmissionMarginProjection, bind_certified_admission_margin_projection, certified_admission_operand


@dataclass(frozen=True, slots=True)
class LawMemberEvaluationBinder:
    "Sole constructor from full law evaluation request/result records into admission continuity."

    def bind(
        self,
        *,
        plan: ReceiptAdmissionReceiptProductionPlan,
        coordinate_id: str,
        request: LawEvaluationRequest,
        result: LawEvaluationResult | None,
    ) -> LawMemberEvaluationBinding:
        coordinate = _coordinate(plan, coordinate_id)
        action = plan.action_fibre(coordinate.action_fibre)
        support = plan.support_cell(coordinate.support_cell)
        law = next(
            value
            for value in plan.atlas.laws
            if ObjectIdentity.from_record(value.law_id, value) == coordinate.response_law
        )
        action_identity = action.action_word_identity
        if (
            request.law != coordinate.response_law
            or request.qualification_result != coordinate.qualification_result
            or request.denominator_member_id != coordinate.denominator_member_id
            or request.candidate_version_id != coordinate.candidate_version_id
            or request.qualification_view_id not in coordinate.qualification_view_ids
            or request.support_cell_id != support.cell_id
            or request.action_word is None
            or ObjectIdentity.from_record(request.action_word.word_id, request.action_word)
            != action_identity
            or request.system_id != plan.atlas.system_id
            or request.world_id != plan.atlas.world_id
            or request.relation_id != law.relation.relation_id
            or request.chart_id != law.chart_id
            or request.prepared_denominator_id != action.action_word.denominator_id
            or request.history_quantity_ids != law.relation.history_quantity_ids
            or request.receiver_quantity_ids != law.relation.receiver_quantity_ids
            or request.horizon
            != ObjectIdentity.from_record(law.relation.horizon.horizon_id, law.relation.horizon)
        ):
            raise ValueError("law evaluation request rewrites its planned admission coordinate")
        if result is None:
            result_identity = None
            implementation = None
            disposition = None
        else:
            if (
                result.request != ObjectIdentity.from_record(request.request_id, request)
                or result.denominator_member_id != coordinate.denominator_member_id
                or result.candidate_version_id != coordinate.candidate_version_id
                or result.qualification_view_id != request.qualification_view_id
                or result.action_word != action_identity
            ):
                raise ValueError("law evaluation result rewrites its request/admission coordinate")
            result_identity = ObjectIdentity.from_record(result.result_id, result)
            implementation = result.evaluator_implementation
            disposition = LawEvaluationBindingDisposition(result.disposition.value)
        return LawMemberEvaluationBinding(
            binding_id=(
                f"law-member-evaluation-binding.{plan.plan_id}.{coordinate_id}."
                f"{request.qualification_view_id}.{request.request_id}"
            ),
            planned_coordinate=ObjectIdentity.from_record(coordinate.coordinate_id, coordinate),
            law_member_binding=coordinate.law_member_binding,
            response_law=coordinate.response_law,
            qualification_result=coordinate.qualification_result,
            denominator_member_id=coordinate.denominator_member_id,
            candidate_version_id=coordinate.candidate_version_id,
            qualification_view_id=request.qualification_view_id,
            action_fibre=coordinate.action_fibre,
            action_word=action_identity,
            support_cell=coordinate.support_cell,
            law_evaluation_request=ObjectIdentity.from_record(request.request_id, request),
            law_evaluation_result=result_identity,
            evaluator_implementation=implementation,
            evaluation_disposition=disposition,
        )


def _coordinate(
    plan: ReceiptAdmissionReceiptProductionPlan,
    coordinate_id: str,
) -> ReceiptAdmissionPlannedCoordinate:
    validate_stable_id(coordinate_id, field_name="coordinate_id")
    value = next(
        (value for value in plan.coordinates if value.coordinate_id == coordinate_id),
        None,
    )
    if value is None:
        raise ValueError("raw admission input references an unplanned coordinate")
    return value


@dataclass(frozen=True, slots=True)
class AdmissionGateRawInput(CanonicalRecord):
    """Method output only; final gate status/margin/reasons are deliberately absent."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/geometry/admission-gate-raw-input'

    input_id: str
    coordinate_id: str
    gate_kind: AdmissionGateKind
    disposition: ReceiptAdmissionRawDisposition
    observed_scalar: NamedDecimal | None
    observed_boolean: bool | None
    observed_identity: ObjectIdentity | None
    law_evaluation_binding: LawMemberEvaluationBinding
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    baseline_compatibility: BaselinePreservationCompatibility | None

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
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


@dataclass(frozen=True, slots=True)
class AdmissionReachabilityRawInput(CanonicalRecord):
    """Full-map method output without a caller-authored reachability disposition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/geometry/admission-reachability-raw-input'

    input_id: str
    coordinate_id: str
    admission_direction_gate_receipt_id: str
    law_evaluation_binding: LawMemberEvaluationBinding
    reference_kind: ReceiptAdmissionReachabilityReferenceKind
    reachability_request: ObjectIdentity
    qualified_hold_action_fibre: ObjectIdentity | None
    qualified_hold_law_evaluation_binding: LawMemberEvaluationBinding | None
    declared_native_reference_direction_id: str | None
    controlled_map_evaluation: ObjectIdentity | None
    candidate_direction_ids: tuple[str, ...]
    viable_direction_ids: tuple[str, ...]
    independent_basis_direction_ids: tuple[str, ...]
    controlled_map_rank: int | None
    condition_number: NamedDecimal | None
    maximum_condition_number: NamedDecimal
    disposition: ReceiptAdmissionRawDisposition
    method: ExecutableReference
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("input_id", self.input_id),
            ("coordinate_id", self.coordinate_id),
            ("admission_direction_gate_receipt_id", self.admission_direction_gate_receipt_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.declared_native_reference_direction_id is not None:
            validate_stable_id(
                self.declared_native_reference_direction_id,
                field_name="declared_native_reference_direction_id",
            )
        if self.reachability_request.object_schema != (
            'empirical-lawhood/geometry/controlled-io-reachability-request'
        ):
            raise ValueError("admission reachability raw input requires its exact method request")
        for name, values in (
            ("candidate_direction_ids", self.candidate_direction_ids),
            ("viable_direction_ids", self.viable_direction_ids),
            ("independent_basis_direction_ids", self.independent_basis_direction_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
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


@dataclass(frozen=True, slots=True)
class AdmissionUtilityRawInput(CanonicalRecord):
    """Native utility operands; the producer alone derives the robust utility."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/geometry/admission-utility-raw-input'

    input_id: str
    coordinate_id: str
    law_evaluation_binding: LawMemberEvaluationBinding
    utility_definition: ObjectIdentity
    direction: ReceiptAdmissionUtilityDirection
    minimum_utility: NamedDecimal
    predicted_response: NamedDecimal | None
    reference_response: NamedDecimal | None
    uncertainty_allowance: NamedDecimal | None
    disposition: ReceiptAdmissionRawDisposition
    evaluator: ExecutableReference
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
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


@dataclass(frozen=True, slots=True)
class RawAdmissionGateReceiptProducer:
    implementation: ObjectIdentity

    capability_key = "geometry.raw-admission-gate-receipt-producer"
    capability_version = "1.0.0"

    def produce(
        self,
        *,
        plan: ReceiptAdmissionReceiptProductionPlan,
        raw: AdmissionGateRawInput,
    ) -> AdmissionCoordinateGateReceipt:
        if plan.gate_receipt_producer != self.implementation:
            raise ValueError("admission gate producer implementation differs from the frozen plan")
        coordinate = _coordinate(plan, raw.coordinate_id)
        predicate = next(
            value for value in plan.gate_predicates if value.gate_kind is raw.gate_kind
        )
        receipt_id = f"receipt.admission-gate.{plan.plan_id}.{raw.input_id}"
        if raw.disposition is ReceiptAdmissionRawDisposition.EVALUATED:
            status, margin, reasons = derive_admission_gate_observation(
                receipt_id=receipt_id,
                predicate=predicate,
                observed_scalar=raw.observed_scalar,
                observed_boolean=raw.observed_boolean,
                observed_identity=raw.observed_identity,
            )
        else:
            status = (
                GateStatus.FAIL
                if raw.disposition is ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT
                else GateStatus.UNEVALUABLE
            )
            margin = None
            reasons = (admission_disposition_reason(raw.disposition),)
        return AdmissionCoordinateGateReceipt(
            receipt_id=receipt_id,
            planned_coordinate=coordinate,
            predicate=predicate,
            disposition=raw.disposition,
            observed_scalar=raw.observed_scalar,
            observed_boolean=raw.observed_boolean,
            observed_identity=raw.observed_identity,
            law_evaluation_binding=raw.law_evaluation_binding,
            information_cutoff=plan.information_cutoff,
            outcome_access=plan.outcome_access,
            evidence_domain=plan.evidence_domain,
            authority_boundary=plan.authority_boundary,
            producer=self.implementation,
            resource_envelope=plan.gate_resource_envelope,
            evaluator=predicate.evaluator,
            input_artifacts=raw.input_artifacts,
            evidence_links=raw.evidence_links,
            baseline_compatibility=raw.baseline_compatibility,
            status=status,
            margin=margin,
            reason_codes=reasons,
        )


@dataclass(frozen=True, slots=True)
class CertifiedAdmissionMarginReceiptProducer:
    "Feed a native margin certificate through the sole raw admission producer."

    raw_producer: RawAdmissionGateReceiptProducer

    capability_key = "geometry.certified-admission-margin-receipt-producer"
    capability_version = "1.0.0"

    def produce(
        self,
        *,
        plan: ReceiptAdmissionReceiptProductionPlan,
        margin: GateMarginReceipt,
        law_evaluation_binding: LawMemberEvaluationBinding,
        baseline_compatibility: BaselinePreservationCompatibility | None,
    ) -> CertifiedAdmissionMarginProjection:
        coordinate = _coordinate(plan, margin.planned_coordinate.coordinate_id)
        if margin.planned_coordinate != coordinate:
            raise ValueError("certified margin substitutes its planned admission coordinate")
        predicate = next(
            value
            for value in plan.gate_predicates
            if value.gate_kind is margin.native_predicate.gate_kind
        )
        if (
            margin.information_cutoff != plan.information_cutoff
            or margin.outcome_access is not plan.outcome_access
            or margin.evidence_domain != plan.evidence_domain
            or margin.authority_boundary != plan.authority_boundary
            or margin.resource_envelope != plan.gate_resource_envelope
            or margin.evidence_ceiling is not plan.evidence_ceiling
            or margin.visibility_ceiling is not plan.visibility_ceiling
        ):
            raise ValueError("certified margin crosses its frozen admission plan boundary")
        disposition, observed_scalar = certified_admission_operand(margin, predicate)
        receipt = self.raw_producer.produce(
            plan=plan,
            raw=AdmissionGateRawInput(
                input_id=f"certified-margin.{margin.receipt_id}",
                coordinate_id=coordinate.coordinate_id,
                gate_kind=predicate.gate_kind,
                disposition=disposition,
                observed_scalar=observed_scalar,
                observed_boolean=None,
                observed_identity=None,
                law_evaluation_binding=law_evaluation_binding,
                input_artifacts=margin.input_artifacts,
                evidence_links=margin.evidence_links,
                baseline_compatibility=baseline_compatibility,
            ),
        )
        return bind_certified_admission_margin_projection(
            margin=margin,
            admission_gate_receipt=receipt,
        )


def _reachability_terminal(
    raw: AdmissionReachabilityRawInput,
) -> tuple[ReachabilityCellDisposition, int, tuple[str, ...]]:
    if raw.disposition is not ReceiptAdmissionRawDisposition.EVALUATED:
        return (
            (
                ReachabilityCellDisposition.UNREACHABLE
                if raw.disposition is ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT
                else ReachabilityCellDisposition.UNEVALUABLE
            ),
            0,
            (admission_disposition_reason(raw.disposition),),
        )
    if raw.controlled_map_rank is None or raw.condition_number is None:
        raise ValueError("evaluated admission reachability lacks rank or conditioning")
    reasons: set[str] = set()
    if raw.controlled_map_rank == 0:
        reasons.add("ADMISSION_CONTROLLED_MAP_ZERO_RANK")
    if raw.condition_number.value > raw.maximum_condition_number.value:
        reasons.add("ADMISSION_CONTROLLED_MAP_ILL_CONDITIONED")
    if not raw.viable_direction_ids:
        reasons.add("ADMISSION_NO_VIABLE_DIRECTION")
    if not raw.independent_basis_direction_ids:
        reasons.add("ADMISSION_NO_INDEPENDENT_DIRECTION_BASIS")
    if reasons:
        return ReachabilityCellDisposition.UNREACHABLE, 0, tuple(sorted(reasons))
    return (
        ReachabilityCellDisposition.REACHABLE,
        len(raw.independent_basis_direction_ids),
        (),
    )


@dataclass(frozen=True, slots=True)
class RawReachabilityAdmissionReceiptProducer:
    implementation: ObjectIdentity

    capability_key = "geometry.raw-admission-reachability-receipt-producer"
    capability_version = "1.0.0"

    def produce(
        self,
        *,
        plan: ReceiptAdmissionReceiptProductionPlan,
        raw: AdmissionReachabilityRawInput,
    ) -> ControlledMapReachabilityReceipt:
        if plan.reachability_receipt_producer != self.implementation:
            raise ValueError("admission reachability producer differs from the frozen plan")
        status, rank, reasons = _reachability_terminal(raw)
        return ControlledMapReachabilityReceipt(
            receipt_id=f"receipt.admission-reachability.{plan.plan_id}.{raw.input_id}",
            planned_coordinate=_coordinate(plan, raw.coordinate_id),
            law_evaluation_binding=raw.law_evaluation_binding,
            admission_direction_gate_receipt_id=raw.admission_direction_gate_receipt_id,
            reference_kind=raw.reference_kind,
            reachability_request=raw.reachability_request,
            qualified_hold_action_fibre=raw.qualified_hold_action_fibre,
            qualified_hold_law_evaluation_binding=(raw.qualified_hold_law_evaluation_binding),
            declared_native_reference_direction_id=raw.declared_native_reference_direction_id,
            controlled_map_evaluation=raw.controlled_map_evaluation,
            horizon=plan.horizon,
            constraint_ids=plan.reachability_constraint_ids,
            candidate_direction_ids=raw.candidate_direction_ids,
            viable_direction_ids=raw.viable_direction_ids,
            independent_basis_direction_ids=raw.independent_basis_direction_ids,
            controlled_map_rank=raw.controlled_map_rank,
            condition_number=raw.condition_number,
            maximum_condition_number=raw.maximum_condition_number,
            disposition=raw.disposition,
            information_cutoff=plan.information_cutoff,
            outcome_access=plan.outcome_access,
            evidence_domain=plan.evidence_domain,
            authority_boundary=plan.authority_boundary,
            producer=self.implementation,
            resource_envelope=plan.reachability_resource_envelope,
            method=raw.method,
            input_artifacts=raw.input_artifacts,
            evidence_links=raw.evidence_links,
            status=status,
            viable_direction_rank=rank,
            reason_codes=reasons,
        )


@dataclass(frozen=True, slots=True)
class RawUtilityAdmissionReceiptProducer:
    implementation: ObjectIdentity

    capability_key = "geometry.raw-admission-utility-receipt-producer"
    capability_version = "1.0.0"

    def produce(
        self,
        *,
        plan: ReceiptAdmissionReceiptProductionPlan,
        raw: AdmissionUtilityRawInput,
    ) -> UtilityEvaluationReceipt:
        if plan.utility_receipt_producer != self.implementation:
            raise ValueError("admission utility producer differs from the frozen plan")
        derived: NamedDecimal | None
        if raw.disposition is ReceiptAdmissionRawDisposition.EVALUATED:
            if (
                raw.predicted_response is None
                or raw.reference_response is None
                or raw.uncertainty_allowance is None
            ):
                raise ValueError("evaluated admission utility lacks raw operands")
            signed = (
                raw.predicted_response.value - raw.reference_response.value
                if raw.direction is ReceiptAdmissionUtilityDirection.HIGHER_IS_BETTER
                else raw.reference_response.value - raw.predicted_response.value
            )
            derived = NamedDecimal(
                value_id=f"utility.{plan.plan_id}.{raw.input_id}",
                value=signed - raw.uncertainty_allowance.value,
                unit=raw.predicted_response.unit,
            )
            status = (
                ReceiptAdmissionUtilityStatus.VIABLE
                if derived.value >= raw.minimum_utility.value
                else ReceiptAdmissionUtilityStatus.NOT_VIABLE
            )
            reasons = () if status is ReceiptAdmissionUtilityStatus.VIABLE else ("ADMISSION_UTILITY_BELOW_MINIMUM",)
        else:
            derived = None
            status = (
                ReceiptAdmissionUtilityStatus.NOT_VIABLE
                if raw.disposition is ReceiptAdmissionRawDisposition.OUTSIDE_SUPPORT
                else ReceiptAdmissionUtilityStatus.UNEVALUABLE
            )
            reasons = (admission_disposition_reason(raw.disposition),)
        return UtilityEvaluationReceipt(
            receipt_id=f"receipt.admission-utility.{plan.plan_id}.{raw.input_id}",
            planned_coordinate=_coordinate(plan, raw.coordinate_id),
            law_evaluation_binding=raw.law_evaluation_binding,
            utility_definition=raw.utility_definition,
            direction=raw.direction,
            minimum_utility=raw.minimum_utility,
            predicted_response=raw.predicted_response,
            reference_response=raw.reference_response,
            uncertainty_allowance=raw.uncertainty_allowance,
            derived_utility=derived,
            disposition=raw.disposition,
            information_cutoff=plan.information_cutoff,
            outcome_access=plan.outcome_access,
            evidence_domain=plan.evidence_domain,
            authority_boundary=plan.authority_boundary,
            producer=self.implementation,
            resource_envelope=plan.utility_resource_envelope,
            evaluator=raw.evaluator,
            input_artifacts=raw.input_artifacts,
            evidence_links=raw.evidence_links,
            status=status,
            reason_codes=reasons,
        )


class AdmissionReceiptCorpusAssembler:
    capability_key = "geometry.admission-receipt-corpus-assembler"
    capability_version = "1.0.0"

    @staticmethod
    def assemble(
        *,
        corpus_id: str,
        plan: ReceiptAdmissionReceiptProductionPlan,
        gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...],
        reachability_receipts: tuple[ControlledMapReachabilityReceipt, ...],
        utility_receipts: tuple[UtilityEvaluationReceipt, ...],
    ) -> ControlledMapAdmissionReceiptCorpus:
        validate_stable_id(corpus_id, field_name="corpus_id")
        artifacts: dict[str, ArtifactIdentity] = {}
        evidence: dict[str, EvidenceLink] = {}
        all_receipts: tuple[
            AdmissionCoordinateGateReceipt | ControlledMapReachabilityReceipt | UtilityEvaluationReceipt,
            ...,
        ] = (*gate_receipts, *reachability_receipts, *utility_receipts)
        for receipt in all_receipts:
            for artifact in receipt.input_artifacts:
                prior_artifact = artifacts.setdefault(artifact.artifact_id, artifact)
                if prior_artifact != artifact:
                    raise ValueError("admission raw receipts reuse an artifact ID with different bytes")
            for link in receipt.evidence_links:
                prior_link = evidence.setdefault(link.link_id, link)
                if prior_link != link:
                    raise ValueError("admission raw receipts reuse an evidence ID with different bytes")
        return ControlledMapAdmissionReceiptCorpus(
            corpus_id=corpus_id,
            plan=plan,
            gate_receipts=tuple(sorted(gate_receipts, key=lambda value: value.receipt_id)),
            reachability_receipts=tuple(
                sorted(reachability_receipts, key=lambda value: value.receipt_id)
            ),
            utility_receipts=tuple(sorted(utility_receipts, key=lambda value: value.receipt_id)),
            input_artifacts=tuple(artifacts[value] for value in sorted(artifacts)),
            evidence_links=tuple(evidence[value] for value in sorted(evidence)),
        )


__all__ = [
    "CertifiedAdmissionMarginReceiptProducer",
    "LawMemberEvaluationBinder",
    'AdmissionGateRawInput',
    'AdmissionReachabilityRawInput',
    'AdmissionReceiptCorpusAssembler',
    'AdmissionUtilityRawInput',
    "RawAdmissionGateReceiptProducer",
    "RawReachabilityAdmissionReceiptProducer",
    "RawUtilityAdmissionReceiptProducer",
]
