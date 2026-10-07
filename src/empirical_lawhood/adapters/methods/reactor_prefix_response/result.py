"""The task's scientific outputs, retaining an explicit pre-method stop."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.finite_action_identification import FiniteActionIdentificationResult
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from .chain import FiniteChainConfig
from .projection import ReactorProjection


@dataclass(frozen=True, slots=True)
class ReactorScienceResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/reactor-science-result'
    projection: ReactorProjection
    bound_config: FiniteChainConfig | None
    identification: FiniteActionIdentificationResult | None
    qualification: LawQualificationResult | None

    def __post_init__(self) -> None:
        if self.projection.extension is None:
            if any(
                v is not None for v in (self.bound_config, self.identification, self.qualification)
            ):
                raise ValueError("incomplete native delivery cannot emit a method or law result")
        elif any(v is None for v in (self.bound_config, self.identification, self.qualification)):
            raise ValueError("complete method invocation must retain every owner result")

    @property
    def scientific_status(self) -> ScientificStatus:
        return (
            ScientificStatus.UNEVALUABLE
            if self.qualification is None
            else self.qualification.scientific_status
        )
