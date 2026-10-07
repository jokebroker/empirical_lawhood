"""Nonexclusive obstruction classification for every physical scale morphism axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .contracts import PhysicalScaleMorphismObstructionKind, PhysicalScaleMorphismObstructionSet


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismObstructionFacts(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-obstruction-facts'

    facts_id: str
    observed_opposition: bool = False
    absent_operand: bool = False
    support_or_denominator_failure: bool = False
    action_chain_failure: bool = False
    power_failure: bool = False
    authority_absent: bool = False
    nonattempt: bool = False
    unresolved: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.facts_id, field_name="facts_id")


_NEXT_ACT = {
    PhysicalScaleMorphismObstructionKind.ABSENT_OPERAND: "restore-or-requalify-operand",
    PhysicalScaleMorphismObstructionKind.ACTION_CHAIN: "reconcile-action-chain",
    PhysicalScaleMorphismObstructionKind.AUTHORITY: "obtain-exact-authority",
    PhysicalScaleMorphismObstructionKind.NONATTEMPT: "retain-nonattempt-or-issue-follow-up-study",
    PhysicalScaleMorphismObstructionKind.OBSERVED_OPPOSITION: "accept-terminal-opposition",
    PhysicalScaleMorphismObstructionKind.POWER: "accept-power-terminal-or-change-design-identity",
    PhysicalScaleMorphismObstructionKind.SUPPORT_OR_DENOMINATOR: "contract-support-or-change-denominator",
    PhysicalScaleMorphismObstructionKind.UNRESOLVED: "resolve-before-claim",
}


def classify_obstructions(
    *,
    obstruction_id: str,
    owner_id: str,
    facts: PhysicalScaleMorphismObstructionFacts,
) -> PhysicalScaleMorphismObstructionSet:
    pairs = (
        (PhysicalScaleMorphismObstructionKind.OBSERVED_OPPOSITION, facts.observed_opposition),
        (PhysicalScaleMorphismObstructionKind.ABSENT_OPERAND, facts.absent_operand),
        (PhysicalScaleMorphismObstructionKind.SUPPORT_OR_DENOMINATOR, facts.support_or_denominator_failure),
        (PhysicalScaleMorphismObstructionKind.ACTION_CHAIN, facts.action_chain_failure),
        (PhysicalScaleMorphismObstructionKind.POWER, facts.power_failure),
        (PhysicalScaleMorphismObstructionKind.AUTHORITY, facts.authority_absent),
        (PhysicalScaleMorphismObstructionKind.NONATTEMPT, facts.nonattempt),
        (PhysicalScaleMorphismObstructionKind.UNRESOLVED, facts.unresolved),
    )
    kinds = tuple(
        sorted((kind for kind, present in pairs if present), key=lambda value: value.value)
    )
    return PhysicalScaleMorphismObstructionSet(
        obstruction_id=obstruction_id,
        owner_id=owner_id,
        kinds=kinds,
        reason_codes=tuple(value.value.lower().replace("_", "-") for value in kinds),
        permitted_next_act_ids=tuple(sorted(_NEXT_ACT[value] for value in kinds)),
    )


__all__ = ['PhysicalScaleMorphismObstructionFacts', "classify_obstructions"]
