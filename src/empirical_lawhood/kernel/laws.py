"""Supported local relational response-law representation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .evidence import (
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from .obligations import ScientificObligations
from .provenance import Claim, EvidenceLink
from .references import ExecutableReference
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from .status import ScientificStatus
from .systems import RelationalIdentity, SystemSpec


class LawRepresentationKind(StrEnum):
    PREDICTIVE_RESPONSE_MAP = "PREDICTIVE_RESPONSE_MAP"
    SUSCEPTIBILITY_SYSTEM_ID = "SUSCEPTIBILITY_SYSTEM_ID"
    FINITE_ACTION_OPERATOR = "FINITE_ACTION_OPERATOR"
    LOCAL_STATE_SPACE = "LOCAL_STATE_SPACE"
    STOCHASTIC_RESPONSE_KERNEL = "STOCHASTIC_RESPONSE_KERNEL"


class CausalStrength(StrEnum):
    OBSERVATIONAL_ASSOCIATION = "OBSERVATIONAL_ASSOCIATION"
    LOGGED_INTERVENTION = "LOGGED_INTERVENTION"
    QUASI_EXPERIMENTAL = "QUASI_EXPERIMENTAL"
    RANDOMIZED_INTERVENTION = "RANDOMIZED_INTERVENTION"
    SIMULATOR_INTERVENTION = "SIMULATOR_INTERVENTION"


@dataclass(frozen=True, slots=True)
class ResponseLaw(CanonicalRecord):
    """One supported, support-limited local law L(D,H,A,R,tau)."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/response-law'

    law_id: str
    system_id: str
    world_id: str
    relation: RelationalIdentity
    chart_id: str
    representation_kind: LawRepresentationKind
    causal_strength: CausalStrength
    evaluator: ExecutableReference
    interface_input_quantity_ids: tuple[str, ...]
    interface_output_quantity_ids: tuple[str, ...]
    mapping_assumption_ids: tuple[str, ...]
    joint_response_sink_effort_distribution_identified: bool
    obligations: ScientificObligations
    claim: Claim
    evidence_links: tuple[EvidenceLink, ...]
    scientific_status: ScientificStatus
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("law_id", self.law_id),
            ("system_id", self.system_id),
            ("world_id", self.world_id),
            ("chart_id", self.chart_id),
        ):
            validate_stable_id(value, field_name=name)
        for field_name, values in (
            ("interface_input_quantity_ids", self.interface_input_quantity_ids),
            ("interface_output_quantity_ids", self.interface_output_quantity_ids),
            ("mapping_assumption_ids", self.mapping_assumption_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        if not set(self.relation.action_quantity_ids).issubset(self.interface_input_quantity_ids):
            raise ValueError("law inputs omit a relational action")
        if not set(self.relation.receiver_quantity_ids).issubset(
            self.interface_output_quantity_ids
        ):
            raise ValueError("law outputs omit a relational receiver")
        if self.representation_kind is LawRepresentationKind.STOCHASTIC_RESPONSE_KERNEL:
            if not self.joint_response_sink_effort_distribution_identified:
                raise ValueError("a stochastic response kernel requires the joint distribution")
        elif self.joint_response_sink_effort_distribution_identified:
            raise ValueError("joint-distribution identification is reserved for response kernels")
        self._validate_evidence()
        require_extensions(self.extensions)

    def _validate_evidence(self) -> None:
        if self.scientific_status is not ScientificStatus.SUPPORTED:
            raise ValueError("ResponseLaw represents only a supported local law")
        if not self.evidence_ceiling.allows(EvidenceRung.LOCAL_LAW):
            raise ValueError("response-law evidence ceiling is below local law")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("an outcome-visible object cannot become a ResponseLaw")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("response-law visibility cannot be lowered")
        if not self.obligations.claim_ready:
            raise ValueError(
                f"response-law obligations are blocked: {self.obligations.blocking_reason_codes()}"
            )
        if self.obligations.support.relation_id != self.relation.relation_id:
            raise ValueError("response-law support relation differs")
        if self.claim.scientific_status is not ScientificStatus.SUPPORTED:
            raise ValueError("response law requires a supported claim")
        if self.claim.observed_rung is not EvidenceRung.LOCAL_LAW:
            raise ValueError("response law requires an observed local law claim")
        if self.claim.claim.world_id != self.world_id:
            raise ValueError("response-law claim world differs")
        if self.claim.claim.relation_id != self.relation.relation_id:
            raise ValueError("response-law claim relation differs")
        require_sorted_unique_ids(
            self.evidence_links, attribute="link_id", field_name="evidence_links"
        )
        if not self.evidence_links:
            raise ValueError("response law requires evidence links")

    def supports_quantity(self, quantity_id: str) -> bool:
        validate_stable_id(quantity_id, field_name="quantity_id")
        return quantity_id in {
            *self.interface_input_quantity_ids,
            *self.interface_output_quantity_ids,
        }


def validate_law_against_system(law: ResponseLaw, system: SystemSpec) -> None:
    if law.world_id != system.world.world_id:
        raise ValueError("response law binds the wrong evidence world")
    if law.outcome_access not in system.world.available_outcome_access:
        raise ValueError("response law uses unavailable world outcome access")
    if law.system_id == system.system_id:
        expected_relation = system.relation
    else:
        component = next(
            (
                component
                for component in system.components
                if component.component_id == law.system_id
            ),
            None,
        )
        if component is None:
            raise ValueError("response law subject is not in the prepared system")
        expected_relation = component.relation
    if law.relation != expected_relation:
        raise ValueError("response law relation differs from its system/component")
    quantities = {quantity.quantity_id for quantity in system.quantities}
    if not {
        *law.interface_input_quantity_ids,
        *law.interface_output_quantity_ids,
    }.issubset(quantities):
        raise ValueError("response law references unknown system quantities")
    system.validate_claim(law.claim.claim)
    known_views = {view.view_id for view in system.numerical_views}
    obligation_views = {
        *law.obligations.structural_convergence.numerical_view_ids,
        *law.obligations.computability.numerical_view_ids,
    }
    if not obligation_views.issubset(known_views):
        raise ValueError("response law obligations use unknown numerical views")
    known_envelopes = {envelope.envelope_id for envelope in system.computability_envelopes}
    if law.obligations.computability.envelope_id not in known_envelopes:
        raise ValueError("response law uses an unknown computability envelope")
