"""Finite controlled-IO version/member qualification without pooling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .contracts import ControlledIOMember, ControlledIOProductDisposition, ControlledIOVersionSet


@dataclass(frozen=True, slots=True)
class ControlledIOVersionSetConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/controlled-io-version-set-config'

    config_id: str
    version_set_id: str
    physical_independent_unit_ids: tuple[str, ...]
    member_stability_contract_id: str
    coverage_claim_id: str | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.version_set_id, field_name="version_set_id")
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
            allow_empty=False,
        )
        validate_stable_id(
            self.member_stability_contract_id,
            field_name="member_stability_contract_id",
        )
        if self.coverage_claim_id is not None:
            validate_stable_id(self.coverage_claim_id, field_name="coverage_claim_id")


@dataclass(frozen=True, slots=True)
class ControlledIOVersionSetAssembler:
    """Assemble a complete roster while retaining every adverse member."""

    def assemble(
        self,
        *,
        members: tuple[ControlledIOMember, ...],
        config: ControlledIOVersionSetConfig,
    ) -> ControlledIOVersionSet:
        ordered = tuple(sorted(members, key=lambda value: value.member_record_id))
        return ControlledIOVersionSet(
            version_set_id=config.version_set_id,
            physical_independent_unit_ids=config.physical_independent_unit_ids,
            members=ordered,
            accepted_member_record_ids=tuple(
                value.member_record_id
                for value in ordered
                if value.disposition is ControlledIOProductDisposition.SUPPORTED
            ),
            rejected_member_record_ids=tuple(
                value.member_record_id
                for value in ordered
                if value.disposition is not ControlledIOProductDisposition.SUPPORTED
            ),
            coverage_claim_id=config.coverage_claim_id,
            member_stability_contract_id=config.member_stability_contract_id,
            evidence_link_ids=tuple(
                sorted(
                    {
                        link_id
                        for value in ordered
                        for link_id in value.diagnostics.evidence_link_ids
                    }
                )
            ),
        )
