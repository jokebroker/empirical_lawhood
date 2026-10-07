"""Frozen numerical falsifier for one local RC study, without physical claims."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderStudyConfig
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id


@dataclass(frozen=True, slots=True)
class ResistorCapacitorLadderEvaluationConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/rc-ladder-numerical-comparison/resistor-capacitor-ladder-evaluation-config'

    config_id: str
    study: ResistorCapacitorLadderStudyConfig
    numerical_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if not self.numerical_only:
            raise ValueError(
                "RC numerical evaluator cannot claim physical-board evidence"
            )


__all__ = ['ResistorCapacitorLadderEvaluationConfig']
