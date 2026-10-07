"""Shared typed configuration and synthetic implementation readiness result."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.adapters.physical.rc_ladder_response.provider import RcLadderResponseSyntheticSourceQualification
from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.contracts import PhysicalScaleMorphismTruthMethodSuiteResult, PhysicalScaleMorphismTruthSuiteConfig
from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.worlds import default_truth_suite_config
from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderModelConfig
from empirical_lawhood.adapters.simulators.rc_ladder_response.provider import RcLadderResponseNumericalQualification, default_numerical_model_config
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)


class PhysicalScaleMorphismImplementationPhase(StrEnum):
    CONTRACT_DEFINITION = "contract-definition"
    TRUTH_METHOD_VALIDATION = "truth-method-validation"
    NUMERICAL_VALIDATION = "numerical-validation"
    SYNTHETIC_SOURCE_VALIDATION = "synthetic-source-validation"
    RUNTIME_REHEARSAL = "runtime-rehearsal"


# These fixed scientific ordinals preserve the original implementation sequence.
# Public phase labels never determine the sequence by lexical sorting.
PHYSICAL_SCALE_MORPHISM_IMPLEMENTATION_PHASE_ORDINALS = (
    (1, PhysicalScaleMorphismImplementationPhase.CONTRACT_DEFINITION),
    (2, PhysicalScaleMorphismImplementationPhase.TRUTH_METHOD_VALIDATION),
    (3, PhysicalScaleMorphismImplementationPhase.NUMERICAL_VALIDATION),
    (4, PhysicalScaleMorphismImplementationPhase.SYNTHETIC_SOURCE_VALIDATION),
    (5, PhysicalScaleMorphismImplementationPhase.RUNTIME_REHEARSAL),
)
PHYSICAL_SCALE_MORPHISM_IMPLEMENTATION_PHASE_ROSTER = tuple(
    phase
    for _, phase in sorted(
        PHYSICAL_SCALE_MORPHISM_IMPLEMENTATION_PHASE_ORDINALS,
        key=lambda item: item[0],
    )
)


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismMethodFreeze(CanonicalRecord):
    """Outcome-blind identity lock between method execution and truth reveal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-method-freeze'

    freeze_id: str
    case_batch_id: str
    observation_batch_sha256: str
    evaluation_outcome_count_at_freeze: int
    physical_outcome_count_at_freeze: int
    frozen: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        validate_stable_id(self.case_batch_id, field_name="case_batch_id")
        validate_sha256(
            self.observation_batch_sha256,
            field_name="observation_batch_sha256",
        )
        if self.evaluation_outcome_count_at_freeze or self.physical_outcome_count_at_freeze:
            raise ValueError("method freeze observed protected outcomes")
        if not self.frozen:
            raise ValueError("method freeze must be immutable")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismStudyConfig(CanonicalRecord):
    """Closed config for the disk-independent rehearsal; never a source path."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-study-config'

    config_id: str
    truth_suite: PhysicalScaleMorphismTruthSuiteConfig
    numerical_model: ResistorCapacitorLadderModelConfig
    physical_semantic_profile_id: str
    implementation_phases: tuple[PhysicalScaleMorphismImplementationPhase, ...]
    physical_execution_authorized: bool
    external_source_write_authorized: bool
    exact_dossier_present: bool
    protected_physical_outcome_count: int
    outcome_access: OutcomeAccess
    maximum_evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(
            self.physical_semantic_profile_id,
            field_name="physical_semantic_profile_id",
        )
        if self.implementation_phases != PHYSICAL_SCALE_MORPHISM_IMPLEMENTATION_PHASE_ROSTER:
            raise ValueError("synthetic programme must freeze the complete implementation-phase roster")
        if (
            self.physical_execution_authorized
            or self.external_source_write_authorized
            or self.exact_dossier_present
            or self.protected_physical_outcome_count
        ):
            raise ValueError("synthetic implementation config crossed the independent physical gate")
        if self.outcome_access is not OutcomeAccess.PRIVILEGED_TRUTH:
            raise ValueError("synthetic truth evaluator requires privileged truth access")
        if self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("synthetic rehearsal must remain nonpromotable")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismSyntheticReadiness(CanonicalRecord):
    """Nonpromotable terminal for the authorized synthetic tranche only."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-synthetic-readiness'

    readiness_id: str
    config_sha256: str
    truth_result_sha256: str
    numerical_qualification_sha256: str
    source_qualification_sha256: str
    method_qualified: bool
    numerical_method_qualified: bool
    draft_source_profile_qualified: bool
    exact_dossier_profile_qualified: bool
    runtime_rehearsal_qualified: bool
    completed_phase_ids: tuple[str, ...]
    next_required_phase_id: str
    physical_execution_eligible: bool
    claim_promotion_allowed: bool
    reason_codes: tuple[str, ...]
    scientific_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.readiness_id, field_name="readiness_id")
        for name in (
            "config_sha256",
            "truth_result_sha256",
            "numerical_qualification_sha256",
            "source_qualification_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        validate_stable_id(self.next_required_phase_id, field_name="next_required_phase_id")
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )
        if self.completed_phase_ids != tuple(
            value.value for value in PHYSICAL_SCALE_MORPHISM_IMPLEMENTATION_PHASE_ROSTER
        ):
            raise ValueError("synthetic readiness does not close the implementation-phase roster")
        expected_runtime = (
            self.method_qualified
            and self.numerical_method_qualified
            and self.draft_source_profile_qualified
            and not self.exact_dossier_profile_qualified
        )
        if self.runtime_rehearsal_qualified != expected_runtime:
            raise ValueError("synthetic readiness is not derived noncompensatingly")
        if (
            self.exact_dossier_profile_qualified
            or self.physical_execution_eligible
            or self.claim_promotion_allowed
        ):
            raise ValueError("synthetic readiness cannot cross construct/source qualification or promote a claim")
        if self.next_required_phase_id != "independent-construct-review":
            raise ValueError("synthetic readiness must stop at independent construct review")
        if self.scientific_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("synthetic readiness must remain nonpromotable")


def default_study_config() -> PhysicalScaleMorphismStudyConfig:
    return PhysicalScaleMorphismStudyConfig(
        config_id="physical-scale-morphism-synthetic-implementation-rehearsal",
        truth_suite=default_truth_suite_config(),
        numerical_model=default_numerical_model_config(),
        physical_semantic_profile_id="physical-scale-morphism-synthetic-draft-profile",
        implementation_phases=PHYSICAL_SCALE_MORPHISM_IMPLEMENTATION_PHASE_ROSTER,
        physical_execution_authorized=False,
        external_source_write_authorized=False,
        exact_dossier_present=False,
        protected_physical_outcome_count=0,
        outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


def adjudicate_synthetic_readiness(
    *,
    config: PhysicalScaleMorphismStudyConfig,
    truth: PhysicalScaleMorphismTruthMethodSuiteResult,
    numerical: RcLadderResponseNumericalQualification,
    source: RcLadderResponseSyntheticSourceQualification,
) -> PhysicalScaleMorphismSyntheticReadiness:
    if truth.config_id != config.truth_suite.config_id:
        raise ValueError("truth result differs from programme config")
    if numerical.model_config_id != config.numerical_model.config_id:
        raise ValueError("numerical result differs from programme config")
    if source.semantic_profile_id != config.physical_semantic_profile_id:
        raise ValueError("source result differs from programme config")
    qualified = (
        truth.method_qualified
        and numerical.numerical_method_qualified
        and source.draft_profile_qualified
        and not source.exact_dossier_profile_qualified
    )
    return PhysicalScaleMorphismSyntheticReadiness(
        readiness_id="readiness.physical-scale-morphism-synthetic-implementation-rehearsal",
        config_sha256=config.fingerprint(),
        truth_result_sha256=truth.fingerprint(),
        numerical_qualification_sha256=numerical.fingerprint(),
        source_qualification_sha256=source.fingerprint(),
        method_qualified=truth.method_qualified,
        numerical_method_qualified=numerical.numerical_method_qualified,
        draft_source_profile_qualified=source.draft_profile_qualified,
        exact_dossier_profile_qualified=source.exact_dossier_profile_qualified,
        runtime_rehearsal_qualified=qualified,
        completed_phase_ids=tuple(value.value for value in config.implementation_phases),
        next_required_phase_id="independent-construct-review",
        physical_execution_eligible=False,
        claim_promotion_allowed=False,
        reason_codes=(
            "IMPLEMENTATION_REHEARSAL_QUALIFIED"
            if qualified
            else "IMPLEMENTATION_REHEARSAL_NOT_QUALIFIED",
            "INDEPENDENT_DOSSIER_REVIEW_REQUIRED",
            "PHYSICAL_EXECUTION_NOT_AUTHORIZED",
        ),
        scientific_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


__all__ = [
    'PhysicalScaleMorphismImplementationPhase',
    'PhysicalScaleMorphismMethodFreeze',
    'PhysicalScaleMorphismStudyConfig',
    'PhysicalScaleMorphismSyntheticReadiness',
    "adjudicate_synthetic_readiness",
    'default_study_config',
]
