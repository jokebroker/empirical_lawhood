"""Issued mechanistic-evidence product rosters and terminal envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_stable_id,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    ImplementationSourceClosure,
)


class MechanisticProductKind(StrEnum):
    CONTROLLED_MARKOV_KERNELS = "CONTROLLED_MARKOV_KERNELS"
    FINITE_HORIZON_IO_MAP = "FINITE_HORIZON_IO_MAP"
    RECEIVER_METRIC_RIESZ = "RECEIVER_METRIC_RIESZ"
    FINITE_JACOBI_WITNESS = "FINITE_JACOBI_WITNESS"
    NONNORMAL_PATHWISE_AUDIT = "NONNORMAL_PATHWISE_AUDIT"
    WITNESS_RESIDUAL = "WITNESS_RESIDUAL"
    CAUSAL_SHADOW = "CAUSAL_SHADOW"
    DETERMINISTIC_COMPARATOR = "DETERMINISTIC_COMPARATOR"


class MechanisticProductUse(StrEnum):
    DESCRIPTIVE = "DESCRIPTIVE"
    LAW_QUALIFICATION_PROFILE_OPERAND = "LAW_QUALIFICATION_PROFILE_OPERAND"
    ACQUISITION_TREATMENT_INPUT = "ACQUISITION_TREATMENT_INPUT"
    RAW_ADMISSION_PRODUCER_OPERAND = "RAW_ADMISSION_PRODUCER_OPERAND"


class MechanisticProductDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    REFUSED = "REFUSED"


@dataclass(frozen=True, slots=True)
class MechanisticProductCoordinate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/mechanistic-product-coordinate'

    coordinate_id: str
    denominator_member_id: str
    candidate_version_id: str
    qualification_view_id: str
    support_cell_id: str
    action_word_id: str
    receiver_block_id: str
    retained_history_id: str
    horizon_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("coordinate_id", self.coordinate_id),
            ("denominator_member_id", self.denominator_member_id),
            ("candidate_version_id", self.candidate_version_id),
            ("qualification_view_id", self.qualification_view_id),
            ("support_cell_id", self.support_cell_id),
            ("action_word_id", self.action_word_id),
            ("receiver_block_id", self.receiver_block_id),
            ("retained_history_id", self.retained_history_id),
            ("horizon_id", self.horizon_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class MechanisticProductRequest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/mechanistic-product-request'

    request_id: str
    coordinate_id: str
    product_kind: MechanisticProductKind
    allowed_use: MechanisticProductUse
    expected_product_schema: str
    proof_owner_kind: CapabilityKind
    proof_owner_capability: ObjectIdentity
    proof_owner_config: ObjectIdentity
    proof_owner_implementation: ObjectIdentity
    maximum_evidence_ceiling: EvidenceCeiling
    decisive_falsifier_ids: tuple[str, ...]
    refusal_rule_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.request_id, field_name="request_id")
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_schema(self.expected_product_schema)
        if self.proof_owner_capability.object_schema != CapabilityManifest.SCHEMA:
            raise ValueError("mechanistic proof owner must be an exact capability manifest")
        if self.proof_owner_implementation.object_schema != ImplementationSourceClosure.SCHEMA:
            raise ValueError("mechanistic proof owner requires an implementation closure")
        for name, values in (
            ("decisive_falsifier_ids", self.decisive_falsifier_ids),
            ("refusal_rule_ids", self.refusal_rule_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if self.maximum_evidence_ceiling in {
            EvidenceCeiling.ADMISSION,
            EvidenceCeiling.CONTROLLER_USE,
        }:
            raise ValueError("mechanistic products cannot themselves claim admission/controller use")
        analysis_products = {
            MechanisticProductKind.CONTROLLED_MARKOV_KERNELS,
            MechanisticProductKind.FINITE_HORIZON_IO_MAP,
            MechanisticProductKind.RECEIVER_METRIC_RIESZ,
            MechanisticProductKind.FINITE_JACOBI_WITNESS,
            MechanisticProductKind.NONNORMAL_PATHWISE_AUDIT,
            MechanisticProductKind.WITNESS_RESIDUAL,
            MechanisticProductKind.CAUSAL_SHADOW,
        }
        if self.product_kind in analysis_products and (
            self.proof_owner_kind is not CapabilityKind.ANALYSIS
        ):
            raise ValueError("mechanistic analysis product has the wrong capability kind")
        if self.product_kind is MechanisticProductKind.DETERMINISTIC_COMPARATOR and (
            self.proof_owner_kind
            not in {
                CapabilityKind.EXPERIMENT_DESIGNER,
                CapabilityKind.CONTROLLER_SYNTHESIZER,
            }
        ):
            raise ValueError("deterministic comparator has the wrong capability kind")


@dataclass(frozen=True, slots=True)
class MechanisticEvidencePlan(CanonicalRecord):
    """Frozen issued product roster; it contains no product dispositions."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/mechanistic-evidence-plan'

    plan_id: str
    response_law_or_candidate: ObjectIdentity
    evidence_projection: ObjectIdentity
    action_chart_id: str
    coordinates: tuple[MechanisticProductCoordinate, ...]
    requests: tuple[MechanisticProductRequest, ...]
    metric_contract: ObjectIdentity
    whitening_contract: ObjectIdentity
    operator_contract: ObjectIdentity
    seed_block_contract: ObjectIdentity
    clock_contract: ObjectIdentity
    finite_order_contract: ObjectIdentity
    conditioning_contract: ObjectIdentity
    rank_deflation_contract: ObjectIdentity
    residual_contract: ObjectIdentity
    member_stability_contract: ObjectIdentity
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        validate_stable_id(self.action_chart_id, field_name="action_chart_id")
        require_sorted_unique_ids(
            self.coordinates,
            attribute="coordinate_id",
            field_name="coordinates",
        )
        require_sorted_unique_ids(self.requests, attribute="request_id", field_name="requests")
        if not self.coordinates or not self.requests:
            raise ValueError("mechanistic plan requires coordinates and product requests")
        coordinate_ids = {value.coordinate_id for value in self.coordinates}
        if {value.coordinate_id for value in self.requests} - coordinate_ids:
            raise ValueError("mechanistic request names an unknown coordinate")
        pairs = {(value.coordinate_id, value.product_kind) for value in self.requests}
        if len(pairs) != len(self.requests):
            raise ValueError("mechanistic plan duplicates a coordinate/product request")
        inherited = inherited_visibility(
            self.parent_visibility_ceilings,
            self.outcome_access,
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("mechanistic plan visibility cannot be lowered")
        if (
            any(
                value.allowed_use is MechanisticProductUse.RAW_ADMISSION_PRODUCER_OPERAND
                for value in self.requests
            )
            and self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("raw admission mechanistic operands must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class MechanisticProductResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/mechanistic-product-result'

    result_id: str
    request_id: str
    coordinate_id: str
    product_kind: MechanisticProductKind
    allowed_use: MechanisticProductUse
    proof_owner_kind: CapabilityKind
    proof_owner_capability: ObjectIdentity
    proof_owner_config: ObjectIdentity
    proof_owner_implementation: ObjectIdentity
    maximum_evidence_ceiling: EvidenceCeiling
    product: ObjectIdentity | None
    disposition: MechanisticProductDisposition
    evidence_link_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("result_id", self.result_id),
            ("request_id", self.request_id),
            ("coordinate_id", self.coordinate_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("evidence_link_ids", self.evidence_link_ids),
            ("artifact_ids", self.artifact_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.proof_owner_capability.object_schema != CapabilityManifest.SCHEMA:
            raise ValueError("mechanistic result proof owner must be a capability manifest")
        if self.proof_owner_implementation.object_schema != ImplementationSourceClosure.SCHEMA:
            raise ValueError("mechanistic result requires an implementation closure")
        if self.maximum_evidence_ceiling in {
            EvidenceCeiling.ADMISSION,
            EvidenceCeiling.CONTROLLER_USE,
        }:
            raise ValueError("mechanistic result cannot claim admission/controller use")
        if self.disposition is MechanisticProductDisposition.SUPPORTED:
            if self.product is None or not self.evidence_link_ids or self.reason_codes:
                raise ValueError("supported mechanistic product requires evidence and product")
        else:
            if self.product is not None or not self.reason_codes:
                raise ValueError("non-supported mechanistic product cannot carry a product")


@dataclass(frozen=True, slots=True)
class MechanisticEvidenceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/mechanistic-evidence-result'

    result_id: str
    plan: ObjectIdentity
    products: tuple[MechanisticProductResult, ...]
    terminal: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.plan.object_schema != MechanisticEvidencePlan.SCHEMA:
            raise ValueError("mechanistic result requires one exact issued plan")
        require_sorted_unique_ids(self.products, attribute="result_id", field_name="products")
        if not self.products or not self.terminal:
            raise ValueError("mechanistic evidence result must be terminal and complete")


@dataclass(frozen=True, slots=True)
class MechanisticEvidenceAssembler:
    "Completeness validator only; cannot promote products to law or admission."

    def assemble(
        self,
        plan: MechanisticEvidencePlan,
        products: tuple[MechanisticProductResult, ...],
    ) -> MechanisticEvidenceResult:
        expected = {
            (
                value.request_id,
                value.coordinate_id,
                value.product_kind,
                value.allowed_use,
                value.proof_owner_kind,
                value.proof_owner_capability,
                value.proof_owner_config,
                value.proof_owner_implementation,
                value.maximum_evidence_ceiling,
            )
            for value in plan.requests
        }
        observed = {
            (
                value.request_id,
                value.coordinate_id,
                value.product_kind,
                value.allowed_use,
                value.proof_owner_kind,
                value.proof_owner_capability,
                value.proof_owner_config,
                value.proof_owner_implementation,
                value.maximum_evidence_ceiling,
            )
            for value in products
        }
        if observed != expected or len(observed) != len(products):
            raise ValueError("mechanistic result does not cover the issued product roster")
        requests = {value.request_id: value for value in plan.requests}
        for product in products:
            request = requests[product.request_id]
            if product.product is not None and (
                product.product.object_schema != request.expected_product_schema
            ):
                raise ValueError("mechanistic product schema differs from the issued request")
        return MechanisticEvidenceResult(
            result_id=f"mechanistic-result.{plan.plan_id}",
            plan=ObjectIdentity.from_record(plan.plan_id, plan),
            products=tuple(sorted(products, key=lambda value: value.result_id)),
            terminal=True,
            outcome_access=plan.outcome_access,
            visibility_ceiling=plan.visibility_ceiling,
        )
