"""Historical design-input ledger for the multi-world flagship."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class ExposedDesignInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/multi-world-response-study/contracts/exposed-design-input'

    input_id: str
    source: ObjectIdentity
    knowledge_ids: tuple[str, ...]
    permitted_use: str
    prohibited_use: str
    contaminated_design_input: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        require_sorted_unique_strings(
            self.knowledge_ids,
            field_name="knowledge_ids",
            allow_empty=False,
        )
        validate_nonempty(self.permitted_use, field_name="permitted_use")
        validate_nonempty(self.prohibited_use, field_name="prohibited_use")
        if not self.contaminated_design_input:
            raise ValueError("historical result cannot become fresh flagship evidence")


@dataclass(frozen=True, slots=True)
class ExposedDesignInputLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/multi-world-response-study/contracts/exposed-design-input-ledger'

    ledger_id: str
    inputs: tuple[ExposedDesignInput, ...]
    required_knowledge_ids: tuple[str, ...]
    historical_units_excluded_from_evaluation_endpoints: bool
    historical_threshold_tuning_forbidden: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        require_sorted_unique_ids(self.inputs, attribute="input_id", field_name="inputs")
        require_sorted_unique_strings(
            self.required_knowledge_ids,
            field_name="required_knowledge_ids",
            allow_empty=False,
        )
        observed = {knowledge for value in self.inputs for knowledge in value.knowledge_ids}
        if observed != set(self.required_knowledge_ids):
            raise ValueError("historical design ledger does not cover the exact knowledge roster")
        if (
            not self.historical_units_excluded_from_evaluation_endpoints
            or not self.historical_threshold_tuning_forbidden
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("historical design knowledge can promote into the current study")


__all__ = ['ExposedDesignInputLedger', 'ExposedDesignInput']
