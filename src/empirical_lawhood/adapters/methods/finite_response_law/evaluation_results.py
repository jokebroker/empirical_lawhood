"Frozen finite response-law evaluation reveal configuration and complete-root scientific readout."

import json
from dataclasses import dataclass
from typing import Any, ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.runtime.controller_evaluation_nested import PreparedControllerCohortEvaluation
from .control_records import FiniteResponseLawControlConfig
from .evaluation_readout import FiniteResponseLawRootInferenceOperands, cohort_inference
from .method_records import FiniteResponseLawQualificationReport
from .method_records import FiniteResponseLawAssignedQualificationReport
from empirical_lawhood.adapters.simulators.finite_response_law.randomness import _assigned_stage_unit

COHORT = "finite-response-law.prospective-evaluation.cohort"
ADJUDICATE = "finite-response-law.prospective-evaluation.adjudicate"


def encode_cohort_statistics(statistics: dict[str, Any]) -> str:
    """Preserve finite computed numbers inside the record's declared JSON string.

    The outer canonical record remains float-free. This changes only encoding,
    not the frozen inference or the precision of its binary floating results.
    """
    return json.dumps(statistics, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationRevealConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-evaluation-reveal-config'
    control: FiniteResponseLawControlConfig

    @property
    def config_id(self) -> str:
        if self.control.config_id.endswith(".control-config"):
            return self.control.config_id.removesuffix(".control-config") + ".reveal-config"
        raise ValueError("Finite response-law evaluation reveal loses its frozen control configuration identity")


def reveal_tasks(config: FiniteResponseLawEvaluationRevealConfig) -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                COHORT,
                ADJUDICATE,
                *(f"{r.stage_unit}.evaluate-use" for r in config.control.source.roots),
            )
        )
    )


@dataclass(frozen=True, slots=True)
class FiniteResponseLawEvaluationCohort(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-evaluation-cohort'
    config: ObjectIdentity
    qualification: ObjectIdentity
    generic: PreparedControllerCohortEvaluation
    operands: tuple[FiniteResponseLawRootInferenceOperands, ...]
    lower_qualified: bool
    cached_qualified: bool
    statistics_json: str

    def __post_init__(self) -> None:
        assigned = bool(self.operands) and _assigned_stage_unit(
            self.operands[0].root_id, "prospective-evaluation", 64
        )
        if (
            self.config.object_schema != FiniteResponseLawEvaluationRevealConfig.SCHEMA
            or self.qualification.object_schema
            != (FiniteResponseLawAssignedQualificationReport if assigned else FiniteResponseLawQualificationReport).SCHEMA
        ):
            raise ValueError("Finite response-law evaluation inference loses its frozen configuration or qualification")
        expected = cohort_inference(
            self.operands,
            lower_qualified=self.lower_qualified,
            cached_qualified=self.cached_qualified,
        )
        if self.statistics_json != encode_cohort_statistics(expected):
            raise ValueError("Finite response-law evaluation inference differs from the complete assigned scalar census")
        scalar = {
            (r.root_id, e.policy_id): (e.selected_word >= 0, e.success, e.false_admission)
            for r in self.operands
            for e in r.events
        }
        generic = {
            (u.root_id, u.policy_id): (u.admitted, u.task_success, u.admitted_failure)
            for u in self.generic.units
        }
        if scalar != generic or len(scalar) != 384 or self.generic.independent_root_count != 64:
            raise ValueError("Finite response-law evaluation generic and independent inference censuses disagree")

    @property
    def scientific_status(self) -> ScientificStatus:
        stats = json.loads(self.statistics_json)
        use, information = stats["tier1_use_supported"], stats["tier1_information_supported"]
        if information is None:
            return ScientificStatus.UNEVALUABLE
        if use and information:
            return ScientificStatus.SUPPORTED
        return ScientificStatus.MIXED if use or information else ScientificStatus.NOT_SUPPORTED

    @property
    def reasons(self) -> tuple[str, ...]:
        stats = json.loads(self.statistics_json)
        return tuple(
            sorted(
                (
                    "ALL_64_ASSIGNED_ROOTS_RETAINED",
                    "SIMULATOR_LOCAL_PAIRED_RECEIVER_ONLY",
                    "USE_SUPPORTED" if stats["tier1_use_supported"] else "USE_NOT_SUPPORTED",
                    "INFORMATION_UNEVALUABLE"
                    if stats["tier1_information_supported"] is None
                    else "INFORMATION_SUPPORTED"
                    if stats["tier1_information_supported"]
                    else "INFORMATION_NOT_SUPPORTED",
                    "ADDED_USE_SUPPORTED"
                    if stats["tier1_added_use_supported"]
                    else "ADDED_USE_NOT_ESTABLISHED",
                )
            )
        )
