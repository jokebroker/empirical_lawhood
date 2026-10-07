"""Proof-owner bindings for additive flagship-readiness capability outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_nonempty,
    validate_schema,
    validate_stable_id,
)

from .capabilities import CapabilityManifest


FLAGSHIP_READINESS_OBLIGATION_IDS = frozenset(
    {
        "order-relation.action-preparation-recurrence",
        "law-qualification.set-valued-finite-chart-reference",
        "admission.gate-margin-certification",
        "controller-use.action-aware-nested-evaluation",
        "prospective.evaluation-prefix-binding",
        "runtime.deadline-free-resource-recovery",
        "bundle.multi-world-issue-recovery",
        "morphism.archive-overlap-controls",
        "adjudication.nonpooling-joint",
    }
)


@dataclass(frozen=True, slots=True)
class MultiWorldReadinessProofOwner(CanonicalRecord):
    "Generic multi-world owner for one closed readiness obligation."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-readiness-proof-owner'

    owner_id: str
    obligation_id: str
    capability_manifest: ObjectIdentity
    owner_module: str
    owner_symbol: str
    rule_semantics: str

    def __post_init__(self) -> None:
        validate_stable_id(self.owner_id, field_name="owner_id")
        validate_stable_id(self.obligation_id, field_name="obligation_id")
        validate_nonempty(self.owner_module, field_name="owner_module")
        validate_nonempty(self.owner_symbol, field_name="owner_symbol")
        validate_nonempty(self.rule_semantics, field_name="rule_semantics")
        if self.obligation_id not in FLAGSHIP_READINESS_OBLIGATION_IDS:
            raise ValueError("flagship multi-world proof owner has an unknown obligation")
        if self.capability_manifest.object_schema != CapabilityManifest.SCHEMA:
            raise ValueError("flagship multi-world proof owner requires a capability manifest")


@dataclass(frozen=True, slots=True)
class ReadinessProofOwnerBinding(CanonicalRecord):
    "Bind one multi-world output schema to its sole registered proof owner."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/readiness-proof-owner-binding'

    binding_id: str
    obligation_id: str
    output_schema: str
    proof_owner: ObjectIdentity
    capability_manifest: ObjectIdentity
    maximum_evidence_ceiling: EvidenceCeiling
    maximum_outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.obligation_id, field_name="obligation_id")
        validate_schema(self.output_schema)
        if self.obligation_id not in FLAGSHIP_READINESS_OBLIGATION_IDS:
            raise ValueError("flagship multi-world proof binding has an unknown obligation")
        if self.proof_owner.object_schema != MultiWorldReadinessProofOwner.SCHEMA:
            raise ValueError("flagship multi-world proof binding has another owner schema")
        if self.capability_manifest.object_schema != CapabilityManifest.SCHEMA:
            raise ValueError("flagship multi-world proof binding requires a capability manifest")


@dataclass(frozen=True, slots=True)
class ReadinessRuleProofOwnerBinding(CanonicalRecord):
    """One named scientific obligation, owner, implementation and output."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/readiness-rule-proof-owner-binding'

    binding_id: str
    obligation_id: str
    output_schema: str
    proof_owner: ObjectIdentity
    capability_manifest: ObjectIdentity
    owner_module: str
    owner_symbol: str
    maximum_evidence_ceiling: EvidenceCeiling
    maximum_outcome_access: OutcomeAccess
    rule_semantics: str

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.obligation_id, field_name="obligation_id")
        validate_schema(self.output_schema)
        validate_nonempty(self.owner_module, field_name="owner_module")
        validate_nonempty(self.owner_symbol, field_name="owner_symbol")
        validate_nonempty(self.rule_semantics, field_name="rule_semantics")
        if self.capability_manifest.object_schema != CapabilityManifest.SCHEMA:
            raise ValueError("flagship proof binding requires an exact capability manifest")
        expected_owner_schema = {
            "order-relation.action-preparation-recurrence": (
                'empirical-lawhood/runtime/action-preparation-recurrence-proof-owner'
            ),
            "admission.gate-margin-certification": ('empirical-lawhood/planning/gate-margin-proof-owner'),
        }.get(self.obligation_id)
        if expected_owner_schema is None or self.proof_owner.object_schema != expected_owner_schema:
            raise ValueError("flagship proof binding has an unknown or substituted owner")


__all__ = [
    "FLAGSHIP_READINESS_OBLIGATION_IDS",
    'ReadinessRuleProofOwnerBinding',
    'ReadinessProofOwnerBinding',
    'MultiWorldReadinessProofOwner',
]
