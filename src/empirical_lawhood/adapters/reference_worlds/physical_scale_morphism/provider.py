"""Typed batch boundary for the physical scale morphism truth-world provider.

Oracle labels deliberately remain in :mod:`oracle`; the truth-blind method
runner receives only ``physical scale morphismTruthCaseBatch``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.physical_scale_morphism.conformance import run_truth_blind_case
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_stable_id,
)

from .contracts import PhysicalScaleMorphismTruthBlindObservation, PhysicalScaleMorphismTruthCase, PhysicalScaleMorphismTruthMethodSuiteResult, PhysicalScaleMorphismTruthSuiteConfig
from .oracle import adjudicate_truth_suite
from .worlds import generate_truth_cases


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismTruthCaseBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/physical-scale-morphism/physical-scale-morphism-truth-case-batch'

    batch_id: str
    config_id: str
    cases: tuple[PhysicalScaleMorphismTruthCase, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        if len(self.cases) != 30:
            raise ValueError("physical scale morphism truth batch requires fifteen fixtures in two variants")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismTruthObservationBatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/physical-scale-morphism/physical-scale-morphism-truth-observation-batch'

    batch_id: str
    case_batch_id: str
    observations: tuple[PhysicalScaleMorphismTruthBlindObservation, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        validate_stable_id(self.case_batch_id, field_name="case_batch_id")
        require_sorted_unique_ids(
            self.observations,
            attribute="observation_id",
            field_name="observations",
        )
        if len(self.observations) != 30:
            raise ValueError("physical scale morphism method batch requires thirty observations")


def generate_truth_case_batch(
    config: PhysicalScaleMorphismTruthSuiteConfig, *, case_id_prefix: str = "case"
) -> PhysicalScaleMorphismTruthCaseBatch:
    return PhysicalScaleMorphismTruthCaseBatch(
        batch_id="batch.physical-scale-morphism-truth-cases",
        config_id=config.config_id,
        cases=generate_truth_cases(config, case_id_prefix=case_id_prefix),
    )


def execute_truth_blind_method(
    batch: PhysicalScaleMorphismTruthCaseBatch,
) -> PhysicalScaleMorphismTruthObservationBatch:
    return PhysicalScaleMorphismTruthObservationBatch(
        batch_id="batch.physical-scale-morphism-truth-observations",
        case_batch_id=batch.batch_id,
        observations=tuple(
            sorted(
                (run_truth_blind_case(value) for value in batch.cases),
                key=lambda value: value.observation_id,
            )
        ),
    )


def evaluate_truth_method(
    *,
    config: PhysicalScaleMorphismTruthSuiteConfig,
    cases: PhysicalScaleMorphismTruthCaseBatch,
    observations: PhysicalScaleMorphismTruthObservationBatch,
) -> PhysicalScaleMorphismTruthMethodSuiteResult:
    if observations.case_batch_id != cases.batch_id or cases.config_id != config.config_id:
        raise ValueError("physical scale morphism truth provider batch identities differ")
    input_by_case = {value.case_id: value.input_data_sha256 for value in cases.cases}
    observation_by_case = {
        value.case_id: value.truth_input_sha256 for value in observations.observations
    }
    if observation_by_case != input_by_case:
        raise ValueError("physical scale morphism truth observations do not bind the generated challenge inputs")
    return adjudicate_truth_suite(
        result_id="result.physical-scale-morphism-truth-method",
        config=config,
        cases=cases.cases,
        observations=observations.observations,
    )


__all__ = [
    'PhysicalScaleMorphismTruthCaseBatch',
    'PhysicalScaleMorphismTruthObservationBatch',
    "evaluate_truth_method",
    "execute_truth_blind_method",
    "generate_truth_case_batch",
]
